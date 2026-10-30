"""Decoder-only translators (LLaMA, Gemma): LoRA fine-tuning or few-shot prompting.

Both models are used as *base* (pre-trained, not instruction-tuned) checkpoints.
The task is expressed as a plain-text prompt

    Translate the following sentence from English to French.

    English: <source>
    French: <target><eos>

LoRA training only updates low-rank adapters on the attention and MLP projections
(loss on target tokens only). The adapters are merged into the base weights before
inference, so inference cost equals that of the unmodified model.
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import config as C
from ..utils import (Timer, bucket_batches, count_params, dtype_name, get_device, pick_dtype, read_json,
                     seed_everything, write_json)
from .base import Translator

HEADER = f"Translate the following sentence from {C.SRC_NAME} to {C.TGT_NAME}.\n\n"


def _dtype_kwarg(dtype):
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


class CausalLMTranslator(Translator):
    architecture = "decoder-only"
    pretrained = True

    def __init__(self, family: str, adapt: str, model_id: str | None = None, dtype_pref: str = "auto",
                 cfg: C.CausalLMConfig | None = None, ckpt_dir: Path | None = None,
                 time_budget_min: float | None = None):
        assert family in ("llama", "gemma") and adapt in ("lora", "fewshot")
        self.family, self.adapt = family, adapt
        self.key = f"{family}_{adapt}"
        self.cfg = cfg or (C.LLAMA if family == "llama" else C.GEMMA)
        self.model_id = model_id or self.cfg.model_id
        self.display_name = f"{'LLaMA 3.2 1B' if family == 'llama' else 'Gemma 3 1B'} ({'LoRA' if adapt == 'lora' else '5-shot'})"
        self.adaptation = "LoRA fine-tuning" if adapt == "lora" else f"few-shot prompting ({self.cfg.n_shots} examples), no training"
        self.device = get_device()
        self.dtype_pref = dtype_pref
        self.ckpt_dir = Path(ckpt_dir or C.CKPT_DIR / self.key)
        self.time_budget_min = time_budget_min
        self.shots: list[dict] = []
        self._load_base(pick_dtype(dtype_pref))

    # ------------------------------------------------------------------ loading
    def _load_base(self, dtype):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kwargs = _dtype_kwarg(dtype)
        if self.family == "gemma":
            kwargs["attn_implementation"] = "eager"  # recommended for Gemma 3 training
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs).to(self.device)
        self.dtype = dtype
        if dtype != torch.float32 and not self._finite_forward():
            print(f"Non-finite activations in {dtype_name(dtype)}; reloading {self.model_id} in fp32.")
            del self.model
            if self.device == "cuda":
                torch.cuda.empty_cache()
            kwargs = _dtype_kwarg(torch.float32)
            if self.family == "gemma":
                kwargs["attn_implementation"] = "eager"
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs).to(self.device)
            self.dtype = torch.float32
        self.precision = dtype_name(self.dtype)
        print(f"Loaded {self.model_id} in {self.precision}")

    @torch.no_grad()
    def _finite_forward(self) -> bool:
        probe = self.tok([self.prompt("The weather is nice today, so we are going for a walk.")] * 2,
                         return_tensors="pt").to(self.device)
        self.model.eval()
        out = self.model(input_ids=probe["input_ids"], attention_mask=probe["attention_mask"]).logits
        return bool(torch.isfinite(out).all())

    # ------------------------------------------------------------------ prompts
    def prompt(self, src: str, shots: list[dict] | None = None) -> str:
        text = HEADER
        for ex in shots or []:
            text += f"{C.SRC_NAME}: {ex['src']}\n{C.TGT_NAME}: {ex['tgt']}\n\n"
        return text + f"{C.SRC_NAME}: {src}\n{C.TGT_NAME}:"

    def choose_shots(self, train: list[dict]) -> None:
        """Fixed, seeded few-shot examples of moderate length (8-20 words), identical for every query."""
        rng = random.Random(C.SEED)
        pool = [r for r in train if 8 <= len(r["src"].split()) <= 20]
        self.shots = rng.sample(pool, self.cfg.n_shots)

    # ----------------------------------------------------------------- training
    def _encode_example(self, row):
        prompt_ids = self.tok(self.prompt(row["src"])).input_ids
        target_ids = self.tok(" " + row["tgt"], add_special_tokens=False).input_ids + [self.tok.eos_token_id]
        ids = prompt_ids + target_ids
        if len(ids) > self.cfg.max_seq_len:
            return None
        return ids, [-100] * len(prompt_ids) + target_ids

    def _collate(self, items, idx):
        width = max(len(items[i][0]) for i in idx)
        pad = self.tok.pad_token_id
        ids = torch.tensor([items[i][0] + [pad] * (width - len(items[i][0])) for i in idx])
        att = torch.tensor([[1] * len(items[i][0]) + [0] * (width - len(items[i][0])) for i in idx])
        lab = torch.tensor([items[i][1] + [-100] * (width - len(items[i][1])) for i in idx])
        return ids.to(self.device), att.to(self.device), lab.to(self.device)

    def _loss(self, ids, att, lab):
        logits = self.model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def _val_loss(self, items):
        self.model.eval()
        total, n = 0.0, 0
        for idx in bucket_batches([len(x[0]) for x in items], batch_size=self.cfg.micro_batch):
            ids, att, lab = self._collate(items, idx)
            with torch.autocast("cuda", dtype=self.dtype, enabled=self.device == "cuda" and self.dtype != torch.float32):
                loss = self._loss(ids, att, lab)
            ntok = int((lab[:, 1:] != -100).sum())
            total, n = total + float(loss) * ntok, n + ntok
        return total / max(n, 1)

    def train(self, train, val):
        if self.adapt == "fewshot":
            self.choose_shots(train)
            return {"train_time_s": 0.0, "note": "no training (few-shot prompting)", "shots": self.shots}

        from peft import LoraConfig, get_peft_model

        c = self.cfg
        seed_everything(C.SEED)
        rows = train[: c.lora_examples]
        items = [e for e in (self._encode_example(r) for r in rows) if e is not None]
        vitems = [e for e in (self._encode_example(r) for r in val[:500]) if e is not None]
        print(f"LoRA training on {len(items):,} examples ({len(rows) - len(items)} dropped as too long)")

        lora = LoraConfig(r=c.lora_r, lora_alpha=c.lora_alpha, lora_dropout=c.lora_dropout,
                          target_modules=list(c.lora_targets), task_type="CAUSAL_LM")
        self.model = get_peft_model(self.model, lora)
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        counts = count_params(self.model)
        print(f"trainable parameters: {counts['params_trainable']:,} / {counts['params_total']:,}")

        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=c.lr, weight_decay=0.0)
        use_amp = self.device == "cuda" and self.dtype != torch.float32
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and self.dtype == torch.float16)

        n_batches = math.ceil(len(items) / c.micro_batch)
        total_updates = max(1, math.ceil(n_batches * c.epochs / c.grad_accum))
        warm = max(1, int(c.warmup_ratio * total_updates))
        updates, history, start = 0, [], time.perf_counter()
        with Timer() as timer:
            for epoch in range(1, c.epochs + 1):
                self.model.train()
                batches = bucket_batches([len(x[0]) for x in items], batch_size=c.micro_batch, shuffle=True,
                                         seed=C.SEED + epoch)
                run, nrun, tokens = 0.0, 0, 0
                for bi, idx in enumerate(batches, 1):
                    ids, att, lab = self._collate(items, idx)
                    with torch.autocast("cuda", dtype=self.dtype, enabled=use_amp):
                        loss = self._loss(ids, att, lab)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite loss. Re-run with --dtype fp32.")
                    scaler.scale(loss / c.grad_accum).backward()
                    run, nrun, tokens = run + float(loss), nrun + 1, tokens + int(att.sum())
                    if bi % c.grad_accum == 0 or bi == len(batches):
                        updates += 1
                        lr = c.lr * (updates / warm if updates <= warm else
                                     0.5 * (1 + math.cos(math.pi * (updates - warm) / max(1, total_updates - warm))))
                        for g in opt.param_groups:
                            g["lr"] = lr
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        scaler.step(opt)
                        scaler.update()
                        opt.zero_grad(set_to_none=True)
                        if updates % 25 == 0:
                            el = time.perf_counter() - start
                            print(f"update {updates}/{total_updates} loss {run / nrun:.3f} lr {lr:.2e} "
                                  f"{tokens / el:,.0f} tok/s ({el:.0f}s)", flush=True)
                    if self.time_budget_min and (time.perf_counter() - start) / 60 > self.time_budget_min:
                        print("time budget reached")
                        break
                v = self._val_loss(vitems)
                history.append({"epoch": epoch, "train_loss": round(run / max(nrun, 1), 4), "val_loss": round(v, 4),
                                "elapsed_s": round(time.perf_counter() - start, 1)})
                print(f"epoch {epoch} done | train {run / max(nrun, 1):.3f} | val {v:.3f}", flush=True)
        info = {"train_time_s": round(timer.elapsed, 1), "epochs_run": len(history), "history": history,
                "train_pairs_used": len(items), "lora_r": c.lora_r, "lora_alpha": c.lora_alpha,
                "lora_params_trainable": counts["params_trainable"],
                "lora_params_total_with_adapters": counts["params_total"]}
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.ckpt_dir)  # adapter only
        write_json(self.ckpt_dir / "train_info.json", info)
        self.model = self.model.merge_and_unload()  # inference cost == base model
        self.model.eval()
        return info

    def load(self):
        """Re-attach a previously trained LoRA adapter and merge it."""
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(self.model, self.ckpt_dir).merge_and_unload()
        self.model.eval()
        return read_json(self.ckpt_dir / "train_info.json")

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def translate(self, sources, temperature=0.0, seed=0, batch_size=C.INFER_BATCH):
        self.model.eval()
        shots = self.shots if self.adapt == "fewshot" else None
        prompts = [self.prompt(s, shots) for s in sources]
        lengths = [len(x) for x in self.tok(prompts)["input_ids"]]
        self.tok.padding_side = "left"
        outs = [""] * len(sources)
        torch.manual_seed(seed)
        sample = temperature > 0
        use_amp = self.device == "cuda" and self.dtype != torch.float32
        for idx in bucket_batches(lengths, batch_size=batch_size):
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True)
            ids, att = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
            out = self.model.generate(
                input_ids=ids, attention_mask=att, max_new_tokens=C.MAX_NEW_TOKENS, do_sample=sample,
                temperature=temperature if sample else 1.0, top_k=0, top_p=1.0, num_beams=1,
                pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id,
                stop_strings=["\n"], tokenizer=self.tok,
            )
            texts = self.tok.batch_decode(out[:, ids.size(1):], skip_special_tokens=True)
            for i, text in zip(idx, texts):
                outs[i] = text.split("\n")[0].strip()
        del use_amp
        return outs

    def param_counts(self):
        return count_params(self.model)
