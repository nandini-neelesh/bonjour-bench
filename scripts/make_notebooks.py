"""Generate the Kaggle runner notebooks in notebooks/.

    python scripts/make_notebooks.py

The notebooks are thin wrappers: they clone the repository, install the extra
dependencies, run `python -m bonjour.run ...` and zip the results for download.
All logic lives in the `bonjour` package so it can be read and tested as normal code.
"""
from pathlib import Path

import nbformat as nbf

REPO_URL = "https://github.com/nandini-neelesh/bonjour-bench.git"
OUT = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = f'''import os, subprocess
os.environ["BONJOUR_CKPT_DIR"] = "/kaggle/temp/checkpoints"   # keep large checkpoints out of the saved output
if not os.path.exists("/kaggle/working/bonjour-bench"):
    subprocess.run(["git", "clone", "{REPO_URL}", "/kaggle/working/bonjour-bench"], check=True)
os.chdir("/kaggle/working/bonjour-bench")
subprocess.run(["git", "pull", "--ff-only"], check=False)
print(os.getcwd())'''

INSTALL = "!pip install -q -r requirements-kaggle.txt"
TOKEN = '''from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")   # Add-ons > Secrets > HF_TOKEN
print("Hugging Face token loaded:", bool(os.environ["HF_TOKEN"]))'''
GPU = "!nvidia-smi"


def notebook(filename, title, intro, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}")]
    for kind, text in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(nb, OUT / filename)
    print("wrote", filename)


PREREQ = ("**Before running:** Settings > Accelerator = *GPU T4 x2*, Internet = *On*"
          "{token}. Run the cells top to bottom; the download cell at the end produces a zip to place in the "
          "repository root.")


def main():
    OUT.mkdir(exist_ok=True)
    notebook("00_data_preparation.ipynb", "00 - Data preparation (OPUS-100 en-fr)",
             PREREQ.format(token="") + " No GPU is needed for this notebook.",
             [("code", SETUP), ("code", INSTALL),
              ("md", "Download OPUS-100 English-French, clean it, remove train/test overlap and write the splits."),
              ("code", "!python -m bonjour.data"),
              ("code", "import json; print(json.dumps(json.load(open('data/processed/data_stats.json')), indent=1)[:3500])"),
              ("code", "!python scripts/zip_results.py data_splits --data"),
              ("md", "Download `data_splits.zip` from the output panel and extract it in the repository root.")])

    notebook("01_transformer_scratch.ipynb", "01 - Basic Transformer trained from scratch",
             PREREQ.format(token=""),
             [("code", SETUP), ("code", INSTALL), ("code", GPU),
              ("md", "Smoke test (about 3 minutes): runs the whole pipeline on a tiny sample."),
              ("code", "!python -m bonjour.run --model transformer --smoke --epochs 2 --vocab-size 2000 --no-sweep"),
              ("md", "Full run in quick mode (30,000 training pairs, 15-minute training cap): trains, translates the test set, runs a small temperature sweep and the ethics probes."),
              ("code", "!python -m bonjour.run --model transformer --quick"),
              ("code", "!python scripts/zip_results.py results_transformer transformer_scratch"),
              ("md", "Download `results_transformer.zip` and extract it in the repository root.")])

    notebook("02_mt5_finetune.ipynb", "02 - mT5-small fine-tuning",
             PREREQ.format(token=""),
             [("code", SETUP), ("code", INSTALL), ("code", GPU),
              ("md", "Smoke test."),
              ("code", "!python -m bonjour.run --model mt5 --smoke --epochs 1 --no-sweep"),
              ("md", "Full run (fp32; mT5 is unstable in fp16)."),
              ("code", "!python -m bonjour.run --model mt5 --quick"),
              ("code", "!python scripts/zip_results.py results_mt5 mt5_small_ft"),
              ("md", "Download `results_mt5.zip` and extract it in the repository root.")])

    for fam, nice, hf in (("llama", "LLaMA 3.2 1B", "meta-llama/Llama-3.2-1B"),
                          ("gemma", "Gemma 3 1B", "google/gemma-3-1b-pt")):
        n = "03" if fam == "llama" else "04"
        notebook(f"{n}_{fam}.ipynb", f"{n} - {nice}: few-shot prompting and LoRA fine-tuning",
                 PREREQ.format(token=f", Secret *HF_TOKEN* attached to the notebook and access granted on `{hf}`"),
                 [("code", SETUP), ("code", INSTALL), ("code", TOKEN), ("code", GPU),
                  ("md", "Smoke test of both adaptation methods (a few minutes; downloads the model once)."),
                  ("code", f"!python -m bonjour.run --model {fam} --adapt fewshot --smoke --no-sweep\n"
                           f"!python -m bonjour.run --model {fam} --adapt lora --smoke --max-train 200 --no-sweep"),
                  ("md", "Full run 1: few-shot prompting (no training)."),
                  ("code", f"!python -m bonjour.run --model {fam} --adapt fewshot --quick"),
                  ("md", "Full run 2: LoRA fine-tuning."),
                  ("code", f"!python -m bonjour.run --model {fam} --adapt lora --quick"),
                  ("code", f"!python scripts/zip_results.py results_{fam} {fam}_fewshot {fam}_lora"),
                  ("md", f"Download `results_{fam}.zip` and extract it in the repository root.")])


if __name__ == "__main__":
    main()
