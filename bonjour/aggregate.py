"""Collect results/<run>/ folders into tables, charts and the README result blocks.

    python -m bonjour.aggregate            # real results
    python -m bonjour.aggregate --smoke    # results/_smoke (pipeline check)

Outputs (results/summary/): results_table.md/.csv, temperature_table.md, quality.png, cost.png,
quality_vs_size.png, temperature.png, probes_side_by_side.md, summary.json.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from . import config as C
from .utils import read_json, read_jsonl, write_json

ORDER = ["transformer_scratch", "mt5_small_ft", "llama_lora", "llama_fewshot", "gemma_lora", "gemma_fewshot"]
LABELS = {
    "transformer_scratch": "Transformer (scratch)", "mt5_small_ft": "mT5-small (fine-tuned)",
    "llama_lora": "LLaMA 3.2 1B (LoRA)", "llama_fewshot": "LLaMA 3.2 1B (5-shot)",
    "gemma_lora": "Gemma 3 1B (LoRA)", "gemma_fewshot": "Gemma 3 1B (5-shot)",
}
# Validated categorical slots 1-6 (see README), one fixed colour per system.
COLOURS = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MARKERS = dict(zip(ORDER, ["o", "s", "^", "D", "v", "P"]))
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def load_runs(results_dir: Path) -> dict:
    runs = {}
    for name in ORDER:
        p = results_dir / name / "metrics.json"
        if p.exists():
            runs[name] = read_json(p)
            sweep = results_dir / name / "temperature_sweep.json"
            runs[name]["_sweep"] = read_json(sweep) if sweep.exists() else None
    return runs


def _fmt_params(n):
    return f"{n / 1e9:.2f} B" if n >= 1e9 else f"{n / 1e6:.1f} M"


def _fmt_time(seconds, none_label="none"):
    if not seconds:
        return none_label
    return f"{seconds / 60:.1f} min" if seconds >= 120 else f"{seconds:.0f} s"


def results_rows(runs):
    rows = []
    for name, m in runs.items():
        rows.append({
            "run": name, "system": LABELS[name], "architecture": m["architecture"],
            "pretrained": "yes" if m["pretrained"] else "no", "adaptation": m["adaptation"],
            "params_total": m["params_total"], "params_trainable": m["params_trainable"],
            "bleu": m["bleu"], "chrf": m["chrf"], "len_ratio": m["len_ratio"], "copy_rate": m["copy_rate"],
            "empty_rate": m["empty_rate"], "train_time_s": m["train_time_s"],
            "inference_time_s": m["inference_time_s"], "inference_ms_per_sentence": m["inference_ms_per_sentence"],
            "precision": m["precision"], "n_test": m["n_test"],
        })
    return rows


def results_markdown(rows):
    head = ("| System | Architecture | Adaptation | Parameters (total / trainable) | BLEU | chrF | "
            "Training time | Inference time (test set) | ms / sentence |\n|---|---|---|---|---:|---:|---:|---:|---:|\n")
    body = ""
    for r in rows:
        trainable = "" if r["params_trainable"] == r["params_total"] else f" / {_fmt_params(r['params_trainable'])}"
        body += (f"| {r['system']} | {r['architecture']} | {r['adaptation']} | {_fmt_params(r['params_total'])}"
                 f"{trainable if trainable else ''} | {r['bleu']:.2f} | {r['chrf']:.2f} | "
                 f"{_fmt_time(r['train_time_s'])} | {_fmt_time(r['inference_time_s'])} | "
                 f"{r['inference_ms_per_sentence']:.1f} |\n")
    return head + body


def diagnostics_markdown(rows):
    head = "| System | Length ratio (hyp/ref) | Copied-source rate | Empty-output rate | Precision |\n|---|---:|---:|---:|---|\n"
    return head + "".join(
        f"| {r['system']} | {r['len_ratio']:.3f} | {100 * r['copy_rate']:.2f}% | {100 * r['empty_rate']:.2f}% | "
        f"{r['precision']} |\n" for r in rows)


def temperature_markdown(runs):
    temps = C.SWEEP_TEMPERATURES
    head = "| System | " + " | ".join("greedy" if t == 0 else f"T = {t}" for t in temps) + " |\n"
    head += "|---|" + "---:|" * len(temps) + "\n"
    out_b, out_c = "**BLEU** (mean ± std over seeds)\n\n" + head, "**chrF** (mean ± std over seeds)\n\n" + head
    for name, m in runs.items():
        sweep = m.get("_sweep")
        if not sweep:
            continue
        by_t = {s["temperature"]: s for s in sweep}
        cell = lambda s, k: "" if s is None else (  # noqa: E731
            f"{s[k + '_mean']:.2f}" if s["temperature"] == 0 else f"{s[k + '_mean']:.2f} ± {s[k + '_std']:.2f}")
        out_b += f"| {LABELS[name]} | " + " | ".join(cell(by_t.get(t), "bleu") for t in temps) + " |\n"
        out_c += f"| {LABELS[name]} | " + " | ".join(cell(by_t.get(t), "chrf") for t in temps) + " |\n"
    return out_b + "\n" + out_c


# ------------------------------------------------------------------------ charts
def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })
    return plt


def _hbar(ax, runs, key, title, fmt, xlabel):
    names = list(runs)[::-1]
    vals = [runs[n][key] for n in names]
    bars = ax.barh([LABELS[n] for n in names], vals, color=[COLOURS[n] for n in names], height=0.62,
                   edgecolor=SURFACE, linewidth=2)
    top = max(vals) if vals and max(vals) > 0 else 1
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + top * 0.015, bar.get_y() + bar.get_height() / 2, fmt(v), va="center",
                fontsize=9, color=INK)
    ax.set_xlim(0, top * 1.25)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def plot_quality(runs, path):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    _hbar(axes[0], runs, "bleu", "BLEU (higher is better)", lambda v: f"{v:.2f}", "sacreBLEU")
    _hbar(axes[1], runs, "chrf", "chrF (higher is better)", lambda v: f"{v:.2f}", "chrF")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cost(runs, path):
    plt = _style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    p = {n: {**m, "params_m": m["params_total"] / 1e6, "train_min": m["train_time_s"] / 60} for n, m in runs.items()}
    _hbar(axes[0], p, "params_m", "Parameters", lambda v: f"{v:,.0f} M", "millions (total, incl. adapters)")
    _hbar(axes[1], p, "train_min", "Training / fine-tuning time",
          lambda v: "none" if v == 0 else f"{v:.1f} min", "minutes")
    _hbar(axes[2], p, "inference_time_s", "Inference time, full test set",
          lambda v: f"{v:.0f} s", "seconds (greedy, same batch size)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_quality_vs_size(runs, path):
    plt = _style()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for n, m in runs.items():
        ax.scatter(m["params_total"] / 1e6, m["bleu"], s=90, color=COLOURS[n], marker=MARKERS[n],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(LABELS[n], (m["params_total"] / 1e6, m["bleu"]), textcoords="offset points", xytext=(8, 5),
                    fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (millions, log scale)")
    ax.set_ylabel("BLEU")
    ax.set_title("Translation quality versus model size")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_temperature(runs, path):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    for ax, key, title in zip(axes, ("bleu", "chrf"), ("BLEU vs sampling temperature", "chrF vs sampling temperature")):
        for n, m in runs.items():
            sweep = m.get("_sweep")
            if not sweep:
                continue
            xs = [max(s["temperature"], 0.0) for s in sweep]
            ys = [s[f"{key}_mean"] for s in sweep]
            es = [s[f"{key}_std"] for s in sweep]
            ax.errorbar(xs, ys, yerr=es, color=COLOURS[n], marker=MARKERS[n], markersize=6, linewidth=2,
                        capsize=2.5, label=LABELS[n], markeredgecolor=SURFACE)
        ax.set_xlabel("temperature (0 = greedy decoding)")
        ax.set_ylabel(key.upper() if key == "bleu" else "chrF")
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------------ probes
def probes_markdown(results_dir: Path, runs) -> str:
    per_run = {}
    for name in runs:
        p = results_dir / name / "probe_translations.jsonl"
        if p.exists():
            per_run[name] = {r["id"]: r for r in read_jsonl(p)}
    if not per_run:
        return "_No probe translations found._\n"
    first = next(iter(per_run.values()))
    out = "# Ethics probe translations (greedy decoding)\n\nThe probe sentences were written by hand to expose " \
          "specific risks. Review each row before drawing conclusions.\n"
    category = None
    for pid, base in first.items():
        if base["category"] != category:
            category = base["category"]
            out += f"\n## {category}\n"
        out += f"\n**{pid}** - `{base['src']}`  \n_What to look for:_ {base['note']}\n\n| System | Translation |\n|---|---|\n"
        for name in per_run:
            hyp = per_run[name].get(pid, {}).get("hyp", "").replace("|", "\\|")
            out += f"| {LABELS[name]} | {hyp} |\n"
    return out


# ------------------------------------------------------------------------ README
def replace_block(text: str, name: str, content: str) -> str:
    start, end = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    if start not in text or end not in text:
        return text
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{content.rstrip()}\n{end}{tail}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    ap.add_argument("--no-readme", action="store_true")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir) / ("_smoke" if args.smoke else "")
    out_dir = results_dir / "summary" if args.smoke else Path(args.results_dir) / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"No results found in {results_dir}")
    print("Runs found:", ", ".join(runs))

    rows = results_rows(runs)
    with open(out_dir / "results_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    table_md, diag_md, temp_md = results_markdown(rows), diagnostics_markdown(rows), temperature_markdown(runs)
    (out_dir / "results_table.md").write_text(table_md + "\n" + diag_md, encoding="utf-8")
    (out_dir / "temperature_table.md").write_text(temp_md, encoding="utf-8")
    (out_dir / "probes_side_by_side.md").write_text(probes_markdown(results_dir, runs), encoding="utf-8")

    plot_quality(runs, out_dir / "quality.png")
    plot_cost(runs, out_dir / "cost.png")
    plot_quality_vs_size(runs, out_dir / "quality_vs_size.png")
    if any(m.get("_sweep") for m in runs.values()):
        plot_temperature(runs, out_dir / "temperature.png")

    best = max(rows, key=lambda r: r["bleu"])
    write_json(out_dir / "summary.json", {
        "best_by_bleu": best["system"], "best_by_chrf": max(rows, key=lambda r: r["chrf"])["system"],
        "fastest_inference": min(rows, key=lambda r: r["inference_time_s"])["system"], "rows": rows})

    if not args.smoke and not args.no_readme:
        readme = C.ROOT / "README.md"
        if readme.exists():
            text = readme.read_text(encoding="utf-8")
            for block, content in (("RESULTS_TABLE", table_md), ("DIAGNOSTICS_TABLE", diag_md),
                                   ("TEMPERATURE_TABLE", temp_md)):
                text = replace_block(text, block, content)
            readme.write_text(text, encoding="utf-8")
            print("README result blocks updated")
    print("Summary written to", out_dir)


if __name__ == "__main__":
    main()
