"""mT5-small, fully fine-tuned for English->French translation (encoder-decoder, pretrained).

Training runs in fp32: mT5 is known to overflow in fp16, and bf16 is not
available on the T4/P100 GPUs offered by Kaggle.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from .. import config as C
from ..utils import Timer, bucket_batches, count_params, get_device, read_json, seed_everything, write_json
from .base import Translator


class MT5Translator(Translator):
    key = "mt5"
    display_name = "mT5-small (fine-tuned)"
    architecture = "encoder-decoder"
    pretrained = True
    adaptation = "full fine-tuning"
    precision = "fp32"

    def __init__(self, cfg: C.MT5Config | None = None, model_id: str | None = None, ckpt_dir: Path | None = None,
                 time_budget_min: float | None = None):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.cfg = cfg or C.MT5
        self.model_id = model_id or self.cfg.model_id
        self.device = get_device()
        self.ckpt_dir = Path(ckpt_dir or C.CKPT_DIR / "mt5")
        self.time_budget_min = time_budget_min
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, torch_dtype=torch.float32).to(self.device)

    def _prefixed(self, sources):
        return [self.cfg.prefix + s for s in sources]

    def _collate(self, enc, labels, idx):
        pad = self.tok.pad_token_id
        width_in = max(len(enc[i]) for i in idx)
        width_lab = max(len(labels[i]) for i in idx)
        input_ids = torch.tensor([enc[i] + [pad] * (width_in - len(enc[i])) for i in idx])
        attn = torch.tensor([[1] * len(enc[i]) + [0] * (width_in - len(enc[i])) for i in idx])
        lab = torch.tensor([labels[i] + [-100] * (width_lab - len(labels[i])) for i in idx])
        return input_ids.to(self.device), attn.to(self.device), lab.to(self.device)

    @torch.no_grad()
    def _val_loss(self, enc, labels):
        self.model.eval()
        total, n = 0.0, 0
        for idx in bucket_batches([len(x) for x in enc], batch_size=self.cfg.batch_size):
            ids, attn, lab = self._collate(enc, labels, idx)
            loss = self.model(input_ids=ids, attention_mask=attn, labels=lab).loss
            ntok = int((lab != -100).sum())
            total, n = total + float(loss) * ntok, n + ntok
        return total / max(n, 1)

    def train(self, train, val):
        from transformers.optimization import Adafactor

        c = self.cfg
        seed_everything(C.SEED)

        def encode(rows):
            e = self.tok(self._prefixed([r["src"] for r in rows]), max_length=c.max_len, truncation=True)
            lab = self.tok(text_target=[r["tgt"] for r in rows], max_length=c.max_len, truncation=True)
            return e["input_ids"], lab["input_ids"]

        enc, labels = encode(train)
        venc, vlabels = encode(val)
        opt = Adafactor(self.model.parameters(), lr=c.lr, scale_parameter=False, relative_step=False,
                        warmup_init=False)
        history, start = [], time.perf_counter()
        out_of_time = False
        with Timer() as timer:
            for epoch in range(1, c.epochs + 1):
                self.model.train()
                batches = bucket_batches([len(x) for x in enc], batch_size=c.batch_size, shuffle=True,
                                         seed=C.SEED + epoch)
                run, n = 0.0, 0
                for k, idx in enumerate(batches, 1):
                    ids, attn, lab = self._collate(enc, labels, idx)
                    loss = self.model(input_ids=ids, attention_mask=attn, labels=lab).loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    run, n = run + float(loss), n + 1
                    if k % 100 == 0:
                        print(f"epoch {epoch} step {k}/{len(batches)} loss {run / n:.3f} "
                              f"({time.perf_counter() - start:.0f}s)", flush=True)
                    if self.time_budget_min and (time.perf_counter() - start) / 60 > self.time_budget_min:
                        out_of_time = True
                        break
                v = self._val_loss(venc, vlabels)
                history.append({"epoch": epoch, "train_loss": round(run / n, 4), "val_loss": round(v, 4),
                                "elapsed_s": round(time.perf_counter() - start, 1)})
                print(f"epoch {epoch} done | train {run / n:.3f} | val {v:.3f}", flush=True)
                if out_of_time:
                    print("time budget reached")
                    break
        self.model.eval()
        info = {"train_time_s": round(timer.elapsed, 1), "epochs_run": len(history), "history": history,
                "train_pairs_used": len(train), "stopped_by_time_budget": out_of_time}
        self.save(info)
        return info

    def save(self, info):
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.ckpt_dir)
        write_json(self.ckpt_dir / "train_info.json", info)

    def load(self):
        from transformers import AutoModelForSeq2SeqLM

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.ckpt_dir, torch_dtype=torch.float32).to(self.device)
        self.model.eval()
        return read_json(self.ckpt_dir / "train_info.json")

    @torch.no_grad()
    def translate(self, sources, temperature=0.0, seed=0, batch_size=C.INFER_BATCH):
        self.model.eval()
        texts = self._prefixed(sources)
        enc = self.tok(texts, max_length=C.MAX_SRC_TOKENS, truncation=True)["input_ids"]
        outs = [""] * len(sources)
        torch.manual_seed(seed)
        sample = temperature > 0
        for idx in bucket_batches([len(x) for x in enc], batch_size=batch_size):
            ids = self.tok.pad({"input_ids": [enc[i] for i in idx]}, return_tensors="pt").to(self.device)
            out = self.model.generate(
                input_ids=ids["input_ids"], attention_mask=ids["attention_mask"], max_new_tokens=C.MAX_NEW_TOKENS,
                num_beams=1, do_sample=sample, temperature=temperature if sample else 1.0, top_k=0, top_p=1.0,
            )
            for i, text in zip(idx, self.tok.batch_decode(out, skip_special_tokens=True)):
                outs[i] = text.strip()
        return outs

    def param_counts(self):
        return count_params(self.model)
