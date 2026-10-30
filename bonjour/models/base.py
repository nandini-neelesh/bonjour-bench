"""Common interface implemented by all four translation systems."""
from __future__ import annotations


class Translator:
    """A translation system that can be trained/adapted and then asked to translate."""

    key = "base"
    display_name = "base"
    architecture = ""  # "encoder-decoder" or "decoder-only"
    pretrained = False
    adaptation = ""
    model_id = ""
    precision = "fp32"

    def train(self, train: list[dict], val: list[dict]) -> dict:
        """Train / adapt on the shared training set. Returns a dict with at least 'train_time_s'."""
        raise NotImplementedError

    def translate(
        self, sources: list[str], temperature: float = 0.0, seed: int = 0, batch_size: int = 32
    ) -> list[str]:
        """Translate sentences. temperature == 0 means greedy decoding."""
        raise NotImplementedError

    def param_counts(self) -> dict:
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "display_name": self.display_name,
            "architecture": self.architecture,
            "pretrained": self.pretrained,
            "adaptation": self.adaptation,
            "model_id": self.model_id,
            "precision": self.precision,
        }
