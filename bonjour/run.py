"""Run one translation system end to end.

    python -m bonjour.run --model transformer
    python -m bonjour.run --model mt5
    python -m bonjour.run --model llama --adapt lora      # and: --adapt fewshot
    python -m bonjour.run --model gemma --adapt lora      # and: --adapt fewshot

Stages: data -> train/adapt -> translate the test set (greedy, timed) -> BLEU/chrF ->
sampling-temperature sweep -> ethics probes. Everything is written to results/<run>/.
Add --smoke for a two-minute end-to-end check on a tiny sample (written to results/_smoke/).
"""
from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

from . import config as C
from .data import ensure_data
from .metrics import diagnostics, score
from .utils import Timer, gpu_info, read_jsonl, seed_everything, write_json, write_jsonl

RUN_NAMES = {"transformer": "transformer_scratch", "mt5": "mt5_small_ft"}

# Reduced-budget profile (--quick): every model sees the same 30,000 training pairs (LoRA: 10,000),
# training stops at the time cap, and the temperature sweep is smaller.
QUICK = {
    "max_train": 30_000,
    "epochs": {"transformer": 12, "mt5": 1, "llama": 1, "gemma": 1},
    "budget": {"transformer": 15, "mt5": 20, "llama": 15, "gemma": 15},
    "sweep_n": 150,
    "sweep_seeds": 2,
}


def build_translator(args):
    if args.model == "transformer":
        from .models.transformer_scratch import TransformerTranslator

        over = {}
        if args.epochs:
            over["epochs"] = args.epochs
        if args.vocab_size:
            over["vocab_size"] = args.vocab_size
        cfg = C.TransformerConfig(**{**C.TRANSFORMER.__dict__, **over})
        return TransformerTranslator(cfg, ckpt_dir=C.CKPT_DIR / "transformer", time_budget_min=args.time_budget_min)
    if args.model == "mt5":
        from .models.mt5 import MT5Translator

        cfg = C.MT5Config(**{**C.MT5.__dict__, **({"epochs": args.epochs} if args.epochs else {})})
        return MT5Translator(cfg, model_id=args.model_id, ckpt_dir=C.CKPT_DIR / "mt5",
                             time_budget_min=args.time_budget_min)
    from .models.causal_lm import CausalLMTranslator

    base = C.LLAMA if args.model == "llama" else C.GEMMA
    overrides = {}
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.max_train:
        overrides["lora_examples"] = min(args.max_train, 10_000) if args.quick else args.max_train
    cfg = C.CausalLMConfig(**{**base.__dict__, **overrides})
    return CausalLMTranslator(args.model, args.adapt, model_id=args.model_id, dtype_pref=args.dtype, cfg=cfg,
                              time_budget_min=args.time_budget_min)


def run_name(args) -> str:
    return RUN_NAMES.get(args.model) or f"{args.model}_{args.adapt}"


def _param_counts(translator, train_info) -> dict:
    """Total parameters of the deployed model; trainable = what the adaptation actually updated."""
    counts = translator.param_counts()
    if getattr(translator, "adapt", None) == "fewshot":
        counts["params_trainable"] = 0
    elif "lora_params_trainable" in train_info:  # adapters were merged into the base weights
        counts["params_trainable"] = train_info["lora_params_trainable"]
    return counts


def timed_translate(translator, sources, batch_size):
    translator.translate(sources[:8], batch_size=batch_size)  # warm-up (kernels, caches); not timed
    with Timer() as t:
        hyps = translator.translate(sources, temperature=0.0, batch_size=batch_size)
    return hyps, t.elapsed


def temperature_sweep(translator, test, n, batch_size, n_seeds=len(C.SWEEP_SEEDS)):
    """BLEU/chrF versus sampling temperature on a fixed random subset of the test set."""
    rng = random.Random(C.SEED)
    subset = [test[i] for i in sorted(rng.sample(range(len(test)), min(n, len(test))))]
    srcs, refs = [r["src"] for r in subset], [r["tgt"] for r in subset]
    rows = []
    for temp in C.SWEEP_TEMPERATURES:
        seeds = (0,) if temp == 0 else tuple(range(n_seeds))
        runs = []
        for seed in seeds:
            hyps = translator.translate(srcs, temperature=temp, seed=seed, batch_size=batch_size)
            runs.append({**score(hyps, refs), **diagnostics(hyps, srcs), "seed": seed})
        mean = lambda key: round(statistics.mean(r[key] for r in runs), 2)  # noqa: E731
        std = lambda key: round(statistics.pstdev(r[key] for r in runs), 2)  # noqa: E731
        rows.append({
            "temperature": temp, "decoding": "greedy" if temp == 0 else "sampling", "n_sentences": len(srcs),
            "bleu_mean": mean("bleu"), "bleu_std": std("bleu"), "chrf_mean": mean("chrf"), "chrf_std": std("chrf"),
            "len_ratio_mean": round(statistics.mean(r["len_ratio"] for r in runs), 3),
            "copy_rate_mean": round(statistics.mean(r["copy_rate"] for r in runs), 4), "runs": runs,
        })
        print(f"T={temp}: BLEU {rows[-1]['bleu_mean']} +/- {rows[-1]['bleu_std']} | chrF {rows[-1]['chrf_mean']}",
              flush=True)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["transformer", "mt5", "llama", "gemma"])
    p.add_argument("--adapt", default="lora", choices=["lora", "fewshot"], help="decoder-only models only")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end check; results go to results/_smoke")
    p.add_argument("--skip-train", action="store_true", help="reuse the checkpoint in checkpoints/")
    p.add_argument("--epochs", type=int)
    p.add_argument("--vocab-size", type=int, help="SentencePiece vocabulary size (Transformer only)")
    p.add_argument("--max-train", type=int, help="use only the first N training pairs (all models)")
    p.add_argument("--quick", action="store_true",
                   help="reduced budget: 30k training pairs, short training with hard time caps, small sweep")
    p.add_argument("--allow-cpu", action="store_true", help="do not stop when no GPU is available")
    p.add_argument("--sweep-seeds", type=int, help="number of random seeds per sampled temperature")
    p.add_argument("--model-id", help="override the Hugging Face model id (used by the tests)")
    p.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--batch-size", type=int, default=C.INFER_BATCH)
    p.add_argument("--sweep-n", type=int)
    p.add_argument("--no-sweep", action="store_true")
    p.add_argument("--no-probes", action="store_true")
    p.add_argument("--time-budget-min", type=float, help="stop training after this many minutes")
    p.add_argument("--data-dir", default=str(C.PROCESSED_DIR))
    p.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    args = p.parse_args(argv)

    # --quick: a reduced but complete experiment, bounded in wall-clock time. Explicit flags win.
    if args.quick:
        args.max_train = args.max_train or QUICK["max_train"]
        args.time_budget_min = args.time_budget_min or QUICK["budget"][args.model]
        args.epochs = args.epochs or QUICK["epochs"].get(args.model)
        args.sweep_n = args.sweep_n or QUICK["sweep_n"]
        args.sweep_seeds = args.sweep_seeds or QUICK["sweep_seeds"]
    args.sweep_n = args.sweep_n or C.SWEEP_N
    args.sweep_seeds = args.sweep_seeds or len(C.SWEEP_SEEDS)

    import torch

    if not args.smoke and not args.allow_cpu and not torch.cuda.is_available():
        raise SystemExit("No GPU detected. In Kaggle: Settings > Accelerator > GPU T4 x2, then restart the session.")
    print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)

    seed_everything(C.SEED)
    name = run_name(args)
    results_root = Path(args.results_dir) / ("_smoke" if args.smoke else "")
    out = results_root / name
    print(f"=== {name} | smoke={args.smoke} ===")

    data = ensure_data(Path(args.data_dir), smoke=args.smoke)
    train, val, test = data["train"], data["val"], data["test"]
    if args.max_train:
        train = train[: args.max_train]
    srcs, refs = [r["src"] for r in test], [r["tgt"] for r in test]
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")

    translator = build_translator(args)
    if args.skip_train:
        train_info = translator.load()
        if getattr(translator, "adapt", None) == "fewshot":
            translator.choose_shots(train)
    else:
        train_info = translator.train(train, val)
    if getattr(translator, "adapt", None) == "fewshot" and not translator.shots:
        translator.choose_shots(train)

    hyps, infer_s = timed_translate(translator, srcs, args.batch_size)
    metrics = {
        "run": name, **translator.describe(),
        "n_train_available": len(train), "n_test": len(test),
        **score(hyps, refs), **diagnostics(hyps, srcs),
        "train_time_s": train_info.get("train_time_s", 0.0),
        "inference_time_s": round(infer_s, 2),
        "inference_ms_per_sentence": round(1000 * infer_s / len(test), 2),
        "inference_batch_size": args.batch_size,
        "decoding": f"greedy, max_new_tokens={C.MAX_NEW_TOKENS}",
        **_param_counts(translator, train_info),
        "training": {k: v for k, v in train_info.items() if k != "shots"},
        "hardware": gpu_info(),
        "smoke": args.smoke,
    }
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "test_translations.jsonl",
                [{"id": r["id"], "src": r["src"], "ref": r["tgt"], "hyp": h} for r, h in zip(test, hyps)])
    print(f"BLEU {metrics['bleu']} | chrF {metrics['chrf']} | train {metrics['train_time_s']}s | "
          f"inference {metrics['inference_time_s']}s | params {metrics['params_total']:,}")

    if not args.no_sweep:
        write_json(out / "temperature_sweep.json", temperature_sweep(translator, test, args.sweep_n, args.batch_size, args.sweep_seeds))

    if not args.no_probes and C.PROBES_PATH.exists():
        probes = read_jsonl(C.PROBES_PATH)
        outs = translator.translate([p_["src"] for p_ in probes], batch_size=args.batch_size)
        write_jsonl(out / "probe_translations.jsonl", [{**p_, "hyp": h} for p_, h in zip(probes, outs)])
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
