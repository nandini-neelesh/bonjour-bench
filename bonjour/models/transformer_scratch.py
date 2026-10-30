"""Basic encoder-decoder Transformer implemented and trained from scratch.

Architecture follows Vaswani et al. (2017) with the now-standard pre-layer-norm
arrangement, a shared source/target/output embedding matrix, sinusoidal positions
and a joint SentencePiece BPE vocabulary. Attention is written out explicitly so
that incremental decoding can use a key/value cache (otherwise inference time
would be unfairly slow compared with the Hugging Face models, which cache).
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C
from ..utils import Timer, bucket_batches, count_params, get_device, read_json, seed_everything, write_json
from .base import Translator

PAD, UNK, BOS, EOS = 0, 1, 2, 3


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh, self.dropout = n_heads, d_model // n_heads, dropout
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def _split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.dh).transpose(1, 2)

    def forward(self, q_in, kv_in, attn_mask=None, is_causal=False, cache=None, static_kv=False):
        q = self._split(self.wq(q_in))
        if static_kv and cache is not None and "k" in cache:  # cross-attention, already computed
            k, v = cache["k"], cache["v"]
        else:
            k, v = self._split(self.wk(kv_in)), self._split(self.wv(kv_in))
            if cache is not None:
                if static_kv:
                    cache["k"], cache["v"] = k, v
                else:  # incremental self-attention: append to the cache
                    if "k" in cache:
                        k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
                    cache["k"], cache["v"] = k, v
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
        )
        b, _, t, _ = out.shape
        return self.wo(out.transpose(1, 2).reshape(b, t, self.h * self.dh))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h, attn_mask=mask))
        return x + self.drop(self.ff(self.ln2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, mem_mask, cache=None):
        h = self.ln1(x)
        if cache is None:  # teacher forcing: whole target at once, causal mask
            x = x + self.drop(self.self_attn(h, h, is_causal=True))
            x = x + self.drop(self.cross_attn(self.ln2(x), memory, attn_mask=mem_mask))
        else:  # one new position, keys/values cached
            x = x + self.drop(self.self_attn(h, h, cache=cache["self"]))
            x = x + self.drop(
                self.cross_attn(self.ln2(x), memory, attn_mask=mem_mask, cache=cache["cross"], static_kv=True)
            )
        return x + self.drop(self.ff(self.ln3(x)))


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, enc_layers, dec_layers, dropout, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        nn.init.normal_(self.emb.weight, mean=0.0, std=d_model**-0.5)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
        self.drop = nn.Dropout(dropout)
        self.enc = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(enc_layers)])
        self.dec = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(dec_layers)])
        self.enc_ln, self.dec_ln = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def _embed(self, ids, start=0):
        x = self.emb(ids) * math.sqrt(self.d_model) + self.pe[start : start + ids.size(1)]
        return self.drop(x)

    def encode(self, src):
        mask = (src != PAD).unsqueeze(1).unsqueeze(2)  # (B,1,1,S) True = real token
        x = self._embed(src)
        for layer in self.enc:
            x = layer(x, mask)
        return self.enc_ln(x), mask

    def decode(self, tgt_in, memory, mem_mask, caches=None, start=0):
        x = self._embed(tgt_in, start)
        for i, layer in enumerate(self.dec):
            x = layer(x, memory, mem_mask, None if caches is None else caches[i])
        return self.dec_ln(x)

    def forward(self, src, tgt_in):
        memory, mask = self.encode(src)
        return self.decode(tgt_in, memory, mask) @ self.emb.weight.T

    @torch.no_grad()
    def generate(self, src, max_len, temperature=0.0, generator=None):
        """Greedy (temperature<=0) or temperature-sampled decoding with a KV cache."""
        memory, mask = self.encode(src)
        b = src.size(0)
        caches = [{"self": {}, "cross": {}} for _ in self.dec]
        ys = torch.full((b, 1), BOS, dtype=torch.long, device=src.device)
        finished = torch.zeros(b, dtype=torch.bool, device=src.device)
        for t in range(max_len):
            h = self.decode(ys[:, -1:], memory, mask, caches, start=t)
            logits = (h[:, -1] @ self.emb.weight.T).float()
            logits[:, PAD] = logits[:, BOS] = float("-inf")
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            nxt = torch.where(finished, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            finished |= nxt == EOS
            if bool(finished.all()):
                break
        return ys[:, 1:]


class TransformerTranslator(Translator):
    key = "transformer"
    display_name = "Transformer (from scratch)"
    architecture = "encoder-decoder"
    pretrained = False
    adaptation = "trained from scratch"

    def __init__(self, cfg: C.TransformerConfig | None = None, ckpt_dir: Path | None = None,
                 time_budget_min: float | None = None):
        self.cfg = cfg or C.TRANSFORMER
        self.device = get_device()
        self.ckpt_dir = Path(ckpt_dir or C.CKPT_DIR / "transformer")
        self.time_budget_min = time_budget_min
        self.sp = None
        self.model = None
        self.model_id = f"custom {self.cfg.enc_layers}+{self.cfg.dec_layers} layers, d_model={self.cfg.d_model}"
        self.precision = "fp16 autocast" if self.device == "cuda" else "fp32"

    # ----------------------------------------------------------------- tokenizer
    def _train_tokenizer(self, train):
        import sentencepiece as spm

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        text_path = self.ckpt_dir / "spm_train.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            for r in train:
                f.write(r["src"] + "\n" + r["tgt"] + "\n")
        t0 = time.perf_counter()
        spm.SentencePieceTrainer.train(
            input=str(text_path), model_prefix=str(self.ckpt_dir / "spm"), vocab_size=self.cfg.vocab_size,
            model_type="bpe", character_coverage=1.0, pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
            input_sentence_size=400_000, shuffle_input_sentence=True, num_threads=4,
        )
        text_path.unlink(missing_ok=True)
        self._load_tokenizer()
        return time.perf_counter() - t0

    def _load_tokenizer(self):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=str(self.ckpt_dir / "spm.model"))

    def _build(self, vocab_size):
        c = self.cfg
        self.model = Seq2SeqTransformer(vocab_size, c.d_model, c.n_heads, c.d_ff, c.enc_layers, c.dec_layers,
                                        c.dropout).to(self.device)

    # ---------------------------------------------------------------- training
    def _encode_pairs(self, rows):
        src = self.sp.encode([r["src"] for r in rows])
        tgt = self.sp.encode([r["tgt"] for r in rows])
        out = []
        for s, t in zip(src, tgt):
            if 0 < len(s) <= self.cfg.max_tok_len and 0 < len(t) <= self.cfg.max_tok_len:
                out.append((s + [EOS], [BOS] + t, t + [EOS]))
        return out

    @staticmethod
    def _pad(seqs):
        width = max(len(s) for s in seqs)
        return torch.tensor([s + [PAD] * (width - len(s)) for s in seqs], dtype=torch.long)

    def _collate(self, items, idx):
        src = self._pad([items[i][0] for i in idx]).to(self.device)
        tgt_in = self._pad([items[i][1] for i in idx]).to(self.device)
        tgt_out = self._pad([items[i][2] for i in idx]).to(self.device)
        return src, tgt_in, tgt_out

    def _loss(self, src, tgt_in, tgt_out, smoothing):
        logits = self.model(src, tgt_in)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
                               ignore_index=PAD, label_smoothing=smoothing)

    @torch.no_grad()
    def _val_loss(self, items):
        self.model.eval()
        total, n = 0.0, 0
        for idx in bucket_batches([len(x[0]) for x in items], max_tokens=self.cfg.max_tokens):
            src, tgt_in, tgt_out = self._collate(items, idx)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                loss = self._loss(src, tgt_in, tgt_out, 0.0)
            ntok = int((tgt_out != PAD).sum())
            total += float(loss) * ntok
            n += ntok
        return total / max(n, 1)

    def train(self, train, val):
        c = self.cfg
        seed_everything(C.SEED)
        tok_time = self._train_tokenizer(train)
        self._build(self.sp.get_piece_size())
        train_items, val_items = self._encode_pairs(train), self._encode_pairs(val)
        print(f"Tokenizer trained in {tok_time:.0f}s | {len(train_items):,} train / {len(val_items):,} val pairs")

        opt = torch.optim.AdamW(self.model.parameters(), lr=c.lr, betas=(0.9, 0.98), eps=1e-9,
                                weight_decay=c.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=self.device == "cuda")
        step, best, best_state, bad, history = 0, float("inf"), None, 0, []
        start = time.perf_counter()
        with Timer() as timer:
            for epoch in range(1, c.epochs + 1):
                self.model.train()
                batches = bucket_batches([max(len(x[0]), len(x[1])) for x in train_items],
                                         max_tokens=c.max_tokens, shuffle=True, seed=C.SEED + epoch)
                run_loss, run_n = 0.0, 0
                for idx in batches:
                    step += 1
                    lr = c.lr * min(step / c.warmup_steps, math.sqrt(c.warmup_steps / step))
                    for g in opt.param_groups:
                        g["lr"] = lr
                    src, tgt_in, tgt_out = self._collate(train_items, idx)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                        loss = self._loss(src, tgt_in, tgt_out, c.label_smoothing)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    run_loss, run_n = run_loss + float(loss), run_n + 1
                v = self._val_loss(val_items)
                history.append({"epoch": epoch, "train_loss": round(run_loss / run_n, 4),
                                "val_loss": round(v, 4), "elapsed_s": round(time.perf_counter() - start, 1)})
                print(f"epoch {epoch:02d} | train {run_loss / run_n:.3f} | val {v:.3f} | "
                      f"{time.perf_counter() - start:.0f}s | lr {lr:.2e}", flush=True)
                if v < best - 1e-4:
                    best, bad = v, 0
                    best_state = copy.deepcopy({k: t.cpu() for k, t in self.model.state_dict().items()})
                else:
                    bad += 1
                    if bad >= c.patience:
                        print("early stopping")
                        break
                if self.time_budget_min and (time.perf_counter() - start) / 60 > self.time_budget_min:
                    print("time budget reached")
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        info = {"train_time_s": round(timer.elapsed, 1), "tokenizer_train_time_s": round(tok_time, 1),
                "epochs_run": len(history), "best_val_loss": round(best, 4), "history": history,
                "train_pairs_used": len(train_items)}
        self.save(info)
        return info

    # ------------------------------------------------------------ persistence
    def save(self, info):
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.ckpt_dir / "model.pt")
        write_json(self.ckpt_dir / "train_info.json", info)

    def load(self):
        self._load_tokenizer()
        self._build(self.sp.get_piece_size())
        self.model.load_state_dict(torch.load(self.ckpt_dir / "model.pt", map_location=self.device))
        self.model.eval()
        return read_json(self.ckpt_dir / "train_info.json")

    # --------------------------------------------------------------- inference
    def translate(self, sources, temperature=0.0, seed=0, batch_size=C.INFER_BATCH):
        self.model.eval()
        ids = [self.sp.encode(s)[: C.MAX_SRC_TOKENS - 1] + [EOS] for s in sources]
        outs = [""] * len(sources)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        for idx in bucket_batches([len(x) for x in ids], batch_size=batch_size):
            src = self._pad([ids[i] for i in idx]).to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                ys = self.model.generate(src, C.MAX_NEW_TOKENS, temperature, gen).tolist()
            for i, row in zip(idx, ys):
                row = row[: row.index(EOS)] if EOS in row else row
                outs[i] = self.sp.decode([t for t in row if t != PAD]).strip()
        return outs

    def param_counts(self):
        return count_params(self.model)
