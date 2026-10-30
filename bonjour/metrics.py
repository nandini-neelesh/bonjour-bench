"""BLEU and chrF (sacreBLEU) plus simple output diagnostics."""
from __future__ import annotations


def score(hyps: list[str], refs: list[str]) -> dict:
    from sacrebleu.metrics import BLEU, CHRF

    bleu_metric, chrf_metric = BLEU(), CHRF()
    bleu = bleu_metric.corpus_score(hyps, [refs])
    chrf = chrf_metric.corpus_score(hyps, [refs])
    return {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "len_ratio": round(bleu.sys_len / max(bleu.ref_len, 1), 3),
        "bleu_signature": str(bleu_metric.get_signature()),
        "chrf_signature": str(chrf_metric.get_signature()),
    }


def diagnostics(hyps: list[str], srcs: list[str]) -> dict:
    """Cheap indicators of degenerate output: empty translations and untranslated copies."""
    n = max(len(hyps), 1)
    empty = sum(1 for h in hyps if not h.strip())
    copied = sum(1 for h, s in zip(hyps, srcs) if h.strip().casefold() == s.strip().casefold())
    return {"empty_rate": round(empty / n, 4), "copy_rate": round(copied / n, 4)}
