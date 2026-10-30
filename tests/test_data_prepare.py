"""Exercise data.prepare() end to end against a fake dataset (no network needed)."""
import random

from bonjour import config as C
from bonjour import data as D
from bonjour.utils import read_json, read_jsonl

WORDS_EN = "time year people way day man thing woman life child world school state family student group country".split()
WORDS_FR = "temps annee gens facon jour homme chose femme vie enfant monde ecole etat famille etudiant groupe pays".split()


def fake_pairs(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        k = rng.randint(3, 12)
        idx = [rng.randrange(len(WORDS_EN)) for _ in range(k)]
        out.append({"en": " ".join(WORDS_EN[i] for i in idx) + f" {rng.randint(0, 10**6)}x",
                    "fr": " ".join(WORDS_FR[i] for i in idx) + f" {rng.randint(0, 10**6)}y"})
    return out


class FakeDS(dict):
    pass


def test_prepare_end_to_end(tmp_path, monkeypatch):
    train, val, test = fake_pairs(5000, 1), fake_pairs(300, 2), fake_pairs(300, 3)
    train += [{"en": "", "fr": "vide"}, {"en": "same text", "fr": "same text"}]  # junk rows
    train += [train[0], train[0]]                                                 # duplicates
    train.append(dict(test[0]))                                                   # leakage: test pair in train
    fake = {"train": {"translation": train}, "validation": {"translation": val}, "test": {"translation": test}}

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: fake)
    monkeypatch.setattr(D, "_dataset_revision", lambda: "fake-revision")
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path / "results")

    cfg = C.DataConfig(n_train=1000, n_val=100, n_test=200)
    stats = D.prepare(tmp_path / "proc", cfg)

    rows = {n: read_jsonl(tmp_path / "proc" / f"{n}.jsonl") for n in ("train", "val", "test")}
    assert {k: len(v) for k, v in rows.items()} == {"train": 1000, "val": 100, "test": 200}
    test_src = {r["src"].casefold() for r in rows["test"]}
    val_src = {r["src"].casefold() for r in rows["val"]}
    train_src = {r["src"].casefold() for r in rows["train"]}
    assert not (train_src & test_src) and not (train_src & val_src), "train/eval leakage"
    assert stats["leakage_removed_from_train"] >= 1
    assert stats["dropped_by_reason"]["train"]["empty"] == 1
    assert stats["dropped_by_reason"]["train"]["identical_sides"] == 1
    assert stats["dropped_by_reason"]["train"]["duplicate"] >= 2
    manifest = read_json(tmp_path / "proc" / "manifest.json")
    assert manifest["dataset_revision"] == "fake-revision" and len(manifest["sha256"]) == 3

    # determinism: a second run reproduces byte-identical files
    first = dict(manifest["sha256"])
    D.prepare(tmp_path / "proc2", cfg)
    assert read_json(tmp_path / "proc2" / "manifest.json")["sha256"] == first

    # verify_manifest passes on untouched files and flags a modified one
    assert D.verify_manifest(tmp_path / "proc")
    with open(tmp_path / "proc" / "test.jsonl", "a", encoding="utf-8") as f:
        f.write("{}\n")
    assert not D.verify_manifest(tmp_path / "proc")
