# transformers

Transformer internals built from first principles: NumPy first, then PyTorch, verified against the reference implementations.

| Project | What | Done when | Status |
|---|---|---|---|
| `attention/` | Scaled dot-product attention in NumPy (causal + padding masks) → multi-head attention in PyTorch | Matches `F.scaled_dot_product_attention` within 1e-5 | 🚧 |
| `tiny-gpt/` | Character-level GPT (~10M params) on Tiny Shakespeare, trained on an RTX 5070 | Val loss < 1.6, sample text | 🚧 |
| `llama-block/` | RMSNorm, RoPE, SwiGLU, GQA: a LLaMA-style block | Matches reference outputs | ⏳ |

```bash
pip install -r requirements.txt
pytest
```
