"""Unit tests for the parts that are easy to get silently wrong.

    python -m pytest tests/test_units.py -q
"""
import random

import torch

from bonjour.data import clean_pairs, normalise, reject_reason
from bonjour.models.transformer_scratch import BOS, EOS, PAD, Seq2SeqTransformer
from bonjour.utils import bucket_batches


def test_normalise():
    assert normalise("  Bonjour \t le\nmonde\x00  ") == "Bonjour le monde"
    assert normalise("é") == "é"  # NFC


def test_cleaning_rules():
    assert reject_reason("", "x") == "empty"
    assert reject_reason("Hello there", "hello there") == "identical_sides"
    assert reject_reason("Hello", "Bonjour tout le monde ceci est une phrase beaucoup trop longue") == "length_ratio"
    assert reject_reason("12345 678", "12345 678 90") == "non_alphabetic"
    assert reject_reason("Thank you very much", "Merci beaucoup") is None
    kept, why = clean_pairs([("Hello world", "Bonjour le monde"), ("hello world", "bonjour le monde"), ("", "x")])
    assert len(kept) == 1 and why == {"duplicate": 1, "empty": 1}


def test_bucket_batches_cover_everything_once():
    lengths = [random.Random(i).randint(1, 50) for i in range(1000)]
    for kwargs in ({"batch_size": 32}, {"max_tokens": 400}):
        batches = bucket_batches(lengths, shuffle=True, seed=3, **kwargs)
        flat = sorted(i for b in batches for i in b)
        assert flat == list(range(1000))
        if "max_tokens" in kwargs:
            assert all(max(lengths[i] for i in b) * len(b) <= 400 for b in batches)
        if "batch_size" in kwargs:
            assert all(len(b) <= 32 for b in batches)


def test_kv_cache_matches_full_forward():
    """Incremental decoding with the cache must give the same logits as teacher forcing."""
    torch.manual_seed(0)
    model = Seq2SeqTransformer(60, 32, 4, 64, 2, 2, 0.0).eval()
    src = torch.randint(4, 60, (3, 7))
    src[1, 5:] = PAD  # padded sentence
    tgt = torch.randint(4, 60, (3, 6))
    tgt[:, 0] = BOS
    with torch.no_grad():
        full = model(src, tgt)
        memory, mask = model.encode(src)
        caches = [{"self": {}, "cross": {}} for _ in model.dec]
        steps = []
        for t in range(tgt.size(1)):
            h = model.decode(tgt[:, t : t + 1], memory, mask, caches, start=t)
            steps.append(h[:, -1] @ model.emb.weight.T)
        inc = torch.stack(steps, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max()


def test_generate_runs_greedy_and_sampling():
    torch.manual_seed(0)
    model = Seq2SeqTransformer(60, 32, 4, 64, 2, 2, 0.0).eval()
    src = torch.randint(4, 60, (4, 6))
    g = torch.Generator().manual_seed(1)
    greedy = model.generate(src, 10)
    assert greedy.shape[0] == 4 and greedy.shape[1] <= 10
    a = model.generate(src, 10, temperature=1.0, generator=torch.Generator().manual_seed(1))
    b = model.generate(src, 10, temperature=1.0, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)  # same seed -> same samples
    assert not ((a == PAD) & (a != 0)).any()
    del g
