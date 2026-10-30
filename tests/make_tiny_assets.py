"""Build tiny synthetic data and tiny random models so that every pipeline can be
smoke-tested on a CPU in a minute, without any download.

    python tests/make_tiny_assets.py <output_dir>
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PAIRS = [
    ("the", "le"), ("a", "un"), ("cat", "chat"), ("dog", "chien"), ("house", "maison"), ("red", "rouge"),
    ("big", "grand"), ("small", "petit"), ("eats", "mange"), ("sees", "voit"), ("loves", "aime"),
    ("water", "eau"), ("bread", "pain"), ("man", "homme"), ("woman", "femme"), ("child", "enfant"),
    ("good", "bon"), ("day", "jour"), ("night", "nuit"), ("city", "ville"), ("car", "voiture"),
    ("book", "livre"), ("reads", "lit"), ("walks", "marche"), ("here", "ici"), ("today", "aujourd'hui"),
]
EN2FR = dict(PAIRS)


def make_pairs(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        words = [rng.choice(PAIRS)[0] for _ in range(rng.randint(3, 9))]
        out.append((" ".join(words), " ".join(EN2FR[w] for w in words)))
    return out


def write_data(root: Path):
    d = root / "data"
    d.mkdir(parents=True, exist_ok=True)
    for name, n, seed in (("train", 2500, 1), ("val", 80, 2), ("test", 120, 3)):
        with open(d / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for i, (en, fr) in enumerate(make_pairs(n, seed)):
                f.write(json.dumps({"id": f"{name}-{i:06d}", "src": en, "tgt": fr}) + "\n")
    return d


def make_tokenizer(texts, vocab_size, kind):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    specials = ["<pad>", "</s>", "<s>", "<unk>"]
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=specials,
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    if kind == "t5":
        tk.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    else:
        tk.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 2)])
    return PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                   unk_token="<unk>")


def main(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    write_data(root)
    texts = [e for e, f in make_pairs(500, 9)] + [f for e, f in make_pairs(500, 9)]
    texts += ["Translate the following sentence from English to French.", "English:", "French:", "translate English to French:"]
    from transformers import (AutoConfig, AutoModelForCausalLM, Gemma3TextConfig, LlamaConfig, MT5Config,
                              MT5ForConditionalGeneration)

    tok = make_tokenizer(texts, 400, "causal")
    v = len(tok)
    llama = AutoModelForCausalLM.from_config(LlamaConfig(
        vocab_size=v, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, bos_token_id=2, eos_token_id=1, pad_token_id=0))
    tok.save_pretrained(root / "tiny_llama")
    llama.save_pretrained(root / "tiny_llama")

    try:
        gemma = AutoModelForCausalLM.from_config(Gemma3TextConfig(
            vocab_size=v, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, sliding_window=64, bos_token_id=2, eos_token_id=1, pad_token_id=0,
            max_position_embeddings=512))
        tok.save_pretrained(root / "tiny_gemma")
        gemma.save_pretrained(root / "tiny_gemma")
    except Exception as e:  # pragma: no cover
        print("Could not build tiny Gemma 3:", e)

    t5tok = make_tokenizer(texts, 400, "t5")
    mt5 = MT5ForConditionalGeneration(MT5Config(
        vocab_size=len(t5tok), d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    t5tok.save_pretrained(root / "tiny_mt5")
    mt5.save_pretrained(root / "tiny_mt5")
    print("tiny assets written to", root)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
