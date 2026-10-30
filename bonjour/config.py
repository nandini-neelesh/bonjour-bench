"""Central configuration.

Every path, split size and hyper-parameter lives here so that all notebooks and
scripts use identical settings. Values can be overridden from the command line
(see bonjour/run.py) but the defaults are the ones reported in the README.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROBES_PATH = DATA_DIR / "probes" / "ethics_probes.jsonl"
RESULTS_DIR = ROOT / "results"
CKPT_DIR = Path(os.environ.get("BONJOUR_CKPT_DIR", ROOT / "checkpoints"))

SEED = 42

SRC_LANG, TGT_LANG = "en", "fr"
SRC_NAME, TGT_NAME = "English", "French"

DATASET_ID = "Helsinki-NLP/opus-100"
DATASET_CONFIG = "en-fr"
DATASET_URL = "https://huggingface.co/datasets/Helsinki-NLP/opus-100"

# Decoding settings shared by all four systems ("comparable settings").
INFER_BATCH = 32
MAX_NEW_TOKENS = 128
MAX_SRC_TOKENS = 128

# Sampling-temperature analysis.
SWEEP_TEMPERATURES = (0.0, 0.3, 0.7, 1.0, 1.5)  # 0.0 means greedy decoding
SWEEP_SEEDS = (0, 1, 2)
SWEEP_N = 300


@dataclass(frozen=True)
class DataConfig:
    n_train: int = 100_000
    n_val: int = 1_000
    n_test: int = 2_000
    min_words: int = 1
    max_words: int = 80
    max_len_ratio: float = 2.5  # allowed char-length ratio src/tgt (and its inverse)
    min_alpha_ratio: float = 0.5  # share of alphabetic characters among non-space characters


DATA = DataConfig()


@dataclass
class TransformerConfig:
    """Basic encoder-decoder Transformer trained from scratch (Vaswani et al., 2017, pre-LN)."""

    vocab_size: int = 16_000
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    enc_layers: int = 4
    dec_layers: int = 4
    dropout: float = 0.2
    label_smoothing: float = 0.1
    max_tokens: int = 6_000  # tokens per training batch (padded)
    lr: float = 8e-4
    warmup_steps: int = 1_000
    epochs: int = 25
    patience: int = 4
    max_tok_len: int = 100  # training pairs longer than this (in subwords) are dropped
    grad_clip: float = 1.0
    weight_decay: float = 0.01


@dataclass
class MT5Config:
    model_id: str = "google/mt5-small"
    prefix: str = "translate English to French: "
    epochs: int = 2
    batch_size: int = 32
    lr: float = 1e-3  # Adafactor, constant
    max_len: int = 128
    max_batch_tokens: int = 1_600  # padded tokens per batch; the 250k-word output layer dominates memory


@dataclass
class CausalLMConfig:
    model_id: str = ""
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    lora_examples: int = 30_000  # LoRA training pairs (subset of the shared training set)
    epochs: int = 1
    micro_batch: int = 16
    grad_accum: int = 2
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_len: int = 192
    max_batch_tokens: int = 1_536  # padded tokens per micro-batch (logits are vocab-sized)
    n_shots: int = 5  # few-shot baseline


LLAMA = CausalLMConfig(model_id="meta-llama/Llama-3.2-1B")
GEMMA = CausalLMConfig(model_id="google/gemma-3-1b-pt", micro_batch=8, grad_accum=4)
TRANSFORMER = TransformerConfig()
MT5 = MT5Config()
