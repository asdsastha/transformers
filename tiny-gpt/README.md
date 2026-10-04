# tiny-gpt

A ~10.8M-parameter GPT trained from scratch, one character at a time, on Tiny Shakespeare (1.1 MB). This is the same architecture as GPT-2 at toy scale. The aim is to build every piece of a decoder-only transformer and see it learn.

**Done when:** `pytest` is 5/5, val loss is **< 1.6**, and `sample.txt` reads like (bad) Shakespeare.

## What we're actually doing

A language model has one job: **given some text, predict the next token.** Generating text is just that prediction run in a loop: predict, append, repeat.

Here a token is a single character, so the vocabulary is 65 symbols. The model reads a window of up to 256 characters and outputs, **at every position**, a probability for each of the 65 possible next characters:

```
input   T  o  _  b  e  _  o  r
target  o  _  b  e  _  o  r  _      ← the same text shifted by one
```

One window therefore gives 256 training examples at once. The catch: position 3 must not see position 4, or it would just copy the answer. That's what the **causal mask** prevents.

Training lowers the **cross-entropy loss**, the average of −log p(correct next char):

| Loss | Meaning |
|---|---|
| 4.17 = ln 65 | random guessing (the untrained model) |
| ~2.5 | it has learned letter frequencies and common pairs |
| **< 1.6** | words, names, line breaks and speaker turns: target |
| ~1.46 | best this model reaches here (at about 3000 steps) |

## The architecture

```
"To be or"  ──► ids (B, T)
                 │
      token embedding (65 → 384) + position embedding (256 → 384)
                 │                                               (B, T, 384)
   ┌─────────────▼─────────────┐
   │  Block  × 6               │
   │   x = x + Attn(LN(x))     │  ← tokens look at earlier tokens (communication)
   │   x = x + MLP(LN(x))      │  ← each token processes what it gathered (computation)
   └─────────────┬─────────────┘
                 │
          LayerNorm → Linear (384 → 65)  ──► logits (B, T, 65)
                 │
       cross-entropy vs. next char  ──► loss
```

- **Embeddings:** turn ids into vectors. Attention alone doesn't know token order, so a learned **position** vector is added. LLaMA replaces this with RoPE, which comes up later in this repo.
- **Causal self-attention:** each token makes a query, a key and a value. `softmax(q·kᵀ / √d + mask) · v` lets it pull information from earlier tokens.
  - Why √d: without it, dot products grow with the dimension, the softmax saturates and the gradients vanish.
  - **6 heads** of size 64 each attend independently, e.g. one head tracks the previous character and another the start of the word.
- **MLP:** 384 → 1536 → 384 with GELU, applied to each token separately. It holds about two-thirds of the parameters.
- **Residual + pre-norm:** each sublayer *adds* to a running vector (the residual stream), so gradients have a direct path through 6 layers.
- **Dropout 0.2:** 1 MB of text is small for 10.8M parameters. Without dropout the model memorises it.

## Files

| File | What | Status |
|---|---|---|
| `model.py` | `CausalSelfAttention`, `Block`, `GPT` (+ `generate`, already written) | **you: 4 TODOs** |
| `test_model.py` | shape, **causality**, starting loss ≈ 4.17, ~10M parameters, generate | ready |
| `train.py` | data → batches → AdamW + OneCycle, bf16 autocast → eval → checkpoint + sample | ready |
| `data/input.txt` | Tiny Shakespeare (git-ignored) | downloaded |

## Run

```bash
pytest -q          # make all 5 pass first
python train.py    # ~2 min on the RTX 5070 → prints loss every 500 steps, writes sample.txt
```

Reference run of `train.py` (with a stand-in model, 5000 steps, bf16, RTX 5070):

| Step | Train | Val |
|---|---|---|
| 0 | 4.33 | 4.32 |
| 1000 | 1.35 | 1.57 |
| 3000 | 1.01 | **1.46** |
| 5000 | 0.88 | 1.51 ← overfitting: train keeps falling, val rises |

That's why the default is 3000 steps.

## The papers behind it

### Attention Is All You Need (Vaswani et al., 2017): the blueprint

The paper builds an **encoder-decoder** for translation. tiny-gpt keeps only the **decoder stack** (the right half of Figure 1) and drops the cross-attention that reads from the encoder. Everything else maps section by section:

| tiny-gpt | Paper | Same or different |
|---|---|---|
| `F.scaled_dot_product_attention` | §3.2.1 Scaled Dot-Product Attention, Eq. 1: `softmax(QKᵀ/√dₖ)V` | Same. Footnote 4 is the reason for √dₖ: q·k has variance dₖ, which saturates the softmax |
| `is_causal=True` | §3.2.3: in the decoder, illegal connections are masked by "setting to −∞" before the softmax | Same |
| 6 heads × 64 | §3.2.2 Multi-Head Attention: h = 8, dₖ = d_model/h = 64 | Same idea; 6 heads instead of 8 |
| MLP 384 → 1536 → 384 | §3.3 Position-wise Feed-Forward: d_ff = 2048 = 4 × 512 | Same 4× ratio; **GELU** instead of ReLU (from GPT-1) |
| `pos_emb` (learned) | §3.5 Positional Encoding: **sinusoidal**. They also tried learned embeddings: "nearly identical results" (Table 3, row E) | Different: learned, as in GPT |
| `x + attn(ln_1(x))` | §3.1: `LayerNorm(x + Sublayer(x))`, i.e. **post-norm** | Different: **pre-norm**, as in GPT-2 (see below) |
| n_layer = 6, n_embd = 384 | Base model: N = 6, d_model = 512 | Same depth, narrower |
| dropout 0.2 | §5.4: P_drop = 0.1 on sublayer outputs and embeddings | Same places, higher rate (small dataset) |
| AdamW + OneCycle | §5.3: Adam (β₂ = 0.98) + warm-up for 4000 steps, then ∝ step^-0.5 | Same idea: warm up, then decay |
| none | §5.4: label smoothing 0.1 | Not used |
| separate `lm_head` | §3.4: embedding and pre-softmax weights are **shared** | Not used (weight tying is an easy extension) |

### From that blueprint to GPT

| Paper | What tiny-gpt takes from it |
|---|---|
| **GPT-1**, *Improving Language Understanding by Generative Pre-Training* (Radford et al., 2018) | Decoder-only transformer trained purely on next-token prediction; learned positions; GELU |
| **GPT-2**, *Language Models are Unsupervised Multitask Learners* (Radford et al., 2019) | §2.3: LayerNorm moved to the **input** of each sub-block plus a final LayerNorm. That is exactly `Block` and `ln_f` here |
| *On Layer Normalization in the Transformer Architecture* (Xiong et al., 2020) | Why pre-norm trains more stably than post-norm, and needs less warm-up |
| *A Neural Probabilistic Language Model* (Bengio et al., 2003) | The objective: learn embeddings, predict the next token, minimise cross-entropy |
| *The Unreasonable Effectiveness of RNNs* (Karpathy, blog, 2015) | Character-level language modelling and the Tiny Shakespeare dataset |

### The building blocks

| Piece | Paper |
|---|---|
| Residual connections `x + f(x)` | *Deep Residual Learning for Image Recognition*, ResNet (He et al., 2016) |
| LayerNorm | *Layer Normalization* (Ba et al., 2016) |
| GELU | *Gaussian Error Linear Units* (Hendrycks & Gimpel, 2016) |
| Dropout | *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (Srivastava et al., 2014) |
| Adam → AdamW | *Adam* (Kingma & Ba, 2015) → *Decoupled Weight Decay Regularization* (Loshchilov & Hutter, 2019) |
| OneCycle LR | *Super-Convergence* (Smith & Topin, 2019) |
| Gradient clipping | *On the Difficulty of Training Recurrent Neural Networks* (Pascanu et al., 2013) |
| bf16 mixed precision | *Mixed Precision Training* (Micikevicius et al., 2018) for the method; bf16 removes its loss scaling |

### What comes next (Monday)

| Change | Paper |
|---|---|
| RMSNorm, SwiGLU, RoPE in one block | *LLaMA* (Touvron et al., 2023) §2.2, citing *RMSNorm* (Zhang & Sennrich, 2019), *GLU Variants Improve Transformer* (Shazeer, 2020), *RoFormer* (Su et al., 2021) |
| Grouped-query attention | *GQA* (Ainslie et al., 2023) |
| Exact attention, tiled in fast memory | *FlashAttention* (Dao et al., 2022). `F.scaled_dot_product_attention` already uses a flash kernel on the GPU |

## Interview angles

- Why divide by √d? Why pre-norm over post-norm? Why is the MLP 4× wide?
- What exactly does the causal mask block, and why is it needed in training but trivial at inference?
- `generate` re-runs the whole window for every new token, which is O(T²) per token. **Next (Mon):** a KV cache, the LLaMA block (RMSNorm, RoPE, SwiGLU, GQA) and INT8 weights, built on this checkpoint.

## Results

_Fill in after your run: final val loss, time, and a few lines of `sample.txt`._

---
↑ Up: [transformers](../README.md)
