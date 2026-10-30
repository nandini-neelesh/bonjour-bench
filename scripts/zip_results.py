"""Zip the result folders of finished runs so they can be downloaded from Kaggle.

    python scripts/zip_results.py <archive_name> <run> [<run> ...]
    python scripts/zip_results.py results_llama llama_fewshot llama_lora
    python scripts/zip_results.py data_splits --data      # val/test splits, stats, manifest

The archive is written to /kaggle/working (or the current directory elsewhere) and keeps
repository-relative paths, so extracting it at the repository root puts every file in place.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv):
    if len(argv) < 3 and "--data" not in argv:
        raise SystemExit(__doc__)
    name = argv[1]
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    files: list[Path] = []
    if "--data" in argv:
        proc = ROOT / "data" / "processed"
        files += [proc / f for f in ("val.jsonl", "test.jsonl", "data_stats.json", "manifest.json")]
        files += [ROOT / "results" / "summary" / "data_lengths.png"]
    for run in (a for a in argv[2:] if not a.startswith("--")):
        run_dir = ROOT / "results" / run
        if not run_dir.exists():
            raise SystemExit(f"No results for run '{run}' in {run_dir}")
        files += [p for p in run_dir.rglob("*") if p.is_file()]
    archive = out_dir / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if f.exists():
                z.write(f, f.relative_to(ROOT).as_posix())
    print(f"Wrote {archive} ({archive.stat().st_size / 1e6:.1f} MB, {len(files)} files)")


if __name__ == "__main__":
    main(sys.argv)
