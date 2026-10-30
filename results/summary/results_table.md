| System | Architecture | Adaptation | Parameters (total / trainable) | BLEU | chrF | Training time | Inference time (test set) | ms / sentence |
|---|---|---|---|---:|---:|---:|---:|---:|
| Transformer (scratch) | encoder-decoder | trained from scratch | 37.6 M | 6.55 | 28.91 | 4.2 min | 14 s | 7.1 |
| mT5-small (fine-tuned) | encoder-decoder | full fine-tuning | 300.2 M | 12.36 | 34.79 | 10.8 min | 85 s | 44.4 |
| LLaMA 3.2 1B (LoRA) | decoder-only | LoRA fine-tuning | 1.24 B / 11.3 M | 23.60 | 49.15 | 2.2 min | 2.2 min | 69.5 |
| Gemma 3 1B (LoRA) | decoder-only | LoRA fine-tuning | 999.9 M / 13.0 M | 27.29 | 52.75 | 8.4 min | 4.8 min | 150.3 |

| System | Length ratio (hyp/ref) | Copied-source rate | Empty-output rate | Precision |
|---|---:|---:|---:|---|
| Transformer (scratch) | 1.036 | 0.31% | 0.10% | fp16 autocast |
| mT5-small (fine-tuned) | 0.975 | 0.68% | 0.00% | fp32 |
| LLaMA 3.2 1B (LoRA) | 0.965 | 1.84% | 0.00% | float16 |
| Gemma 3 1B (LoRA) | 0.981 | 1.94% | 0.00% | float32 |
