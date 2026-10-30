"""Small shared helpers: seeding, timing, JSON IO, batching, hardware info."""
from __future__ import annotations

import json
import os
import platform
import random
import time
from pathlib import Path


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """Wall-clock timer that synchronises CUDA before reading the clock."""

    def __enter__(self):
        cuda_sync()
        self.t0 = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc):
        cuda_sync()
        self.elapsed = time.perf_counter() - self.t0
        return False


def gpu_info() -> dict:
    import torch

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:  # pragma: no cover
        pass
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu=props.name,
            gpu_memory_gb=round(props.total_memory / 1024**3, 1),
            gpu_count=torch.cuda.device_count(),
        )
    else:
        info["gpu"] = "none (CPU)"
    return info


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": int(total), "params_trainable": int(trainable)}


def pick_dtype(preference: str = "auto"):
    """bf16 where the GPU supports it natively, fp16 otherwise, fp32 on CPU."""
    import torch

    table = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if preference != "auto":
        return table[preference]
    if not torch.cuda.is_available():
        return torch.float32
    try:
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        native_bf16 = torch.cuda.get_device_capability()[0] >= 8
    return torch.bfloat16 if native_bf16 else torch.float16


def dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def bucket_batches(lengths, batch_size=None, max_tokens=None, shuffle=False, seed=0):
    """Group example indices into batches of similar length.

    Either a fixed number of items (batch_size) or a padded-token budget
    (max_tokens) limits a batch. With shuffle=True a little noise is added to
    the sort key and the batch order is shuffled, so every epoch differs.
    """
    rng = random.Random(seed)
    keys = [(length + (rng.random() * 2 if shuffle else 0.0), i) for i, length in enumerate(lengths)]
    order = [i for _, i in sorted(keys)]
    batches, cur, cur_max = [], [], 0
    for i in order:
        length = lengths[i]
        new_max = max(cur_max, length)
        full = bool(cur) and (
            (batch_size is not None and len(cur) >= batch_size)
            or (max_tokens is not None and new_max * (len(cur) + 1) > max_tokens)
        )
        if full:
            batches.append(cur)
            cur, new_max = [], length
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches


def env_token_present() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
