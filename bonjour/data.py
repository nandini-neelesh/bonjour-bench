"""Dataset download, cleaning, splitting and loading (OPUS-100, English-French).

Run `python -m bonjour.data` to build data/processed/{train,val,test}.jsonl.
The procedure is deterministic (fixed seed), so every Kaggle notebook that calls
`ensure_data()` rebuilds exactly the same splits; a manifest with SHA-256
checksums is written so that this can be verified.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import random
import re
import statistics
import unicodedata
from pathlib import Path

from . import config as C
from .utils import read_json, read_jsonl, write_json, write_jsonl

_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Unicode NFC, control characters removed, whitespace collapsed."""
    text = unicodedata.normalize("NFC", text)
    text = _CTRL.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _alpha_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(c.isalpha() for c in chars) / len(chars)


def reject_reason(en: str, fr: str, cfg: C.DataConfig = C.DATA) -> str | None:
    """Return why a pair should be dropped, or None if it is kept."""
    if not en or not fr:
        return "empty"
    if en.casefold() == fr.casefold():
        return "identical_sides"
    n_en, n_fr = len(en.split()), len(fr.split())
    if n_en < cfg.min_words or n_fr < cfg.min_words:
        return "too_short"
    if n_en > cfg.max_words or n_fr > cfg.max_words:
        return "too_long"
    ratio = len(en) / len(fr)
    if ratio > cfg.max_len_ratio or ratio < 1.0 / cfg.max_len_ratio:
        return "length_ratio"
    if _alpha_ratio(en) < cfg.min_alpha_ratio or _alpha_ratio(fr) < cfg.min_alpha_ratio:
        return "non_alphabetic"
    return None


def clean_pairs(pairs, cfg: C.DataConfig = C.DATA):
    """Normalise, filter and de-duplicate (en, fr) pairs. Returns (kept, reason_counts)."""
    kept, seen, reasons = [], set(), collections.Counter()
    for en, fr in pairs:
        en, fr = normalise(en), normalise(fr)
        why = reject_reason(en, fr, cfg)
        if why is None:
            key = (en.casefold(), fr.casefold())
            if key in seen:
                why = "duplicate"
            else:
                seen.add(key)
        if why is None:
            kept.append((en, fr))
        else:
            reasons[why] += 1
    return kept, dict(reasons)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _length_stats(pairs) -> dict:
    out = {}
    for side, idx in (("en", 0), ("fr", 1)):
        lens = sorted(len(p[idx].split()) for p in pairs)
        out[side] = {
            "mean_words": round(statistics.mean(lens), 2),
            "median_words": statistics.median(lens),
            "p95_words": lens[int(0.95 * (len(lens) - 1))],
            "max_words": lens[-1],
            "vocab_types": len({w for p in pairs for w in p[idx].casefold().split()}),
        }
    return out


def _plot_lengths(splits: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    colours = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    for ax, (side, idx, title) in zip(axes, (("en", 0, "English source"), ("fr", 1, "French reference"))):
        for name, pairs in splits.items():
            lens = [min(len(p[idx].split()), 60) for p in pairs]
            ax.hist(lens, bins=range(0, 62, 2), density=True, alpha=0.55, label=name, color=colours[name])
        ax.set_title(title)
        ax.set_xlabel("sentence length (words, capped at 60)")
    axes[0].set_ylabel("share of sentences")
    axes[0].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def prepare(out_dir: Path = C.PROCESSED_DIR, cfg: C.DataConfig = C.DATA, seed: int = C.SEED) -> dict:
    """Download OPUS-100 en-fr, clean it, remove leakage, sample the splits and save them."""
    from datasets import load_dataset

    out_dir = Path(out_dir)
    print(f"Loading {C.DATASET_ID} ({C.DATASET_CONFIG}) ...")
    ds = load_dataset(C.DATASET_ID, C.DATASET_CONFIG)

    def extract(split):
        return [(r[C.SRC_LANG], r[C.TGT_LANG]) for r in split["translation"]]

    raw = {name: extract(ds[name]) for name in ("train", "validation", "test")}
    raw_counts = {k: len(v) for k, v in raw.items()}
    print("Raw sizes:", raw_counts)

    clean, dropped = {}, {}
    for name, pairs in raw.items():
        clean[name], dropped[name] = clean_pairs(pairs, cfg)
        print(f"  {name}: kept {len(clean[name]):,} / {len(pairs):,}  dropped={dropped[name]}")

    rng = random.Random(seed)

    def sample_ordered(pairs, k):
        if len(pairs) <= k:
            return list(pairs)
        return [pairs[i] for i in sorted(rng.sample(range(len(pairs)), k))]

    test = sample_ordered(clean["test"], cfg.n_test)
    val = sample_ordered(clean["validation"], cfg.n_val)

    # Leakage control: no validation/test sentence (either side) may occur in training data.
    held_en = {p[0].casefold() for p in test + val}
    held_fr = {p[1].casefold() for p in test + val}
    before = len(clean["train"])
    pool = [p for p in clean["train"] if p[0].casefold() not in held_en and p[1].casefold() not in held_fr]
    leakage_removed = before - len(pool)
    train = sample_ordered(pool, cfg.n_train)
    print(f"Removed {leakage_removed:,} training pairs overlapping val/test; train pool {len(pool):,}")

    splits = {"train": train, "val": val, "test": test}
    for name, pairs in splits.items():
        rows = [{"id": f"{name}-{i:06d}", "src": en, "tgt": fr} for i, (en, fr) in enumerate(pairs)]
        write_jsonl(out_dir / f"{name}.jsonl", rows)

    stats = {
        "dataset": C.DATASET_ID,
        "config": C.DATASET_CONFIG,
        "raw_sizes": raw_counts,
        "cleaning_rules": {
            "normalisation": "Unicode NFC, control characters removed, whitespace collapsed",
            "min_words": cfg.min_words,
            "max_words": cfg.max_words,
            "max_char_length_ratio": cfg.max_len_ratio,
            "min_alphabetic_ratio": cfg.min_alpha_ratio,
            "other": "identical source/target dropped, exact duplicates dropped",
        },
        "dropped_by_reason": dropped,
        "cleaned_sizes": {k: len(v) for k, v in clean.items()},
        "leakage_removed_from_train": leakage_removed,
        "final_sizes": {k: len(v) for k, v in splits.items()},
        "length_stats": {k: _length_stats(v) for k, v in splits.items()},
        "seed": seed,
    }
    write_json(out_dir / "data_stats.json", stats)
    _plot_lengths(splits, C.RESULTS_DIR / "summary" / "data_lengths.png")

    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": C.DATASET_ID,
        "dataset_revision": _dataset_revision(),
        "seed": seed,
        "sha256": {f"{n}.jsonl": _sha256(out_dir / f"{n}.jsonl") for n in splits},
    }
    write_json(out_dir / "manifest.json", manifest)
    return stats


def _dataset_revision() -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(C.DATASET_ID).sha
    except Exception:
        return None


def verify_manifest(out_dir: Path = C.PROCESSED_DIR) -> bool:
    """Check that rebuilt files match the committed checksums (test/val)."""
    manifest_path = Path(out_dir) / "manifest.json"
    if not manifest_path.exists():
        return True
    manifest = read_json(manifest_path)
    ok = True
    for name, digest in manifest["sha256"].items():
        path = Path(out_dir) / name
        if path.exists() and _sha256(path) != digest:
            print(f"WARNING: {name} does not match the checksum in manifest.json")
            ok = False
    return ok


def load_split(name: str, out_dir: Path = C.PROCESSED_DIR) -> list[dict]:
    return read_jsonl(Path(out_dir) / f"{name}.jsonl")


def ensure_data(out_dir: Path = C.PROCESSED_DIR, smoke: bool = False) -> dict:
    """Return {'train','val','test'} lists of {'id','src','tgt'}; build them if missing."""
    out_dir = Path(out_dir)
    if not all((out_dir / f"{n}.jsonl").exists() for n in ("train", "val", "test")):
        committed = {n: (out_dir / f"{n}.jsonl").exists() for n in ("val", "test")}
        if any(committed.values()):
            # val/test are committed to the repo; remember them to verify the rebuild.
            expected = {n: _sha256(out_dir / f"{n}.jsonl") for n in committed if committed[n]}
        else:
            expected = {}
        prepare(out_dir)
        for n, digest in expected.items():
            if _sha256(out_dir / f"{n}.jsonl") != digest:
                print(f"WARNING: rebuilt {n}.jsonl differs from the committed copy (dataset revision changed?)")
    data = {n: load_split(n, out_dir) for n in ("train", "val", "test")}
    if smoke:
        data = {"train": data["train"][:600], "val": data["val"][:60], "test": data["test"][:96]}
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the OPUS-100 en-fr splits")
    parser.add_argument("--out-dir", default=str(C.PROCESSED_DIR))
    args = parser.parse_args()
    stats = prepare(Path(args.out_dir))
    print(stats["final_sizes"])


if __name__ == "__main__":
    main()
