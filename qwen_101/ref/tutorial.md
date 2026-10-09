# From layman to expert: LLMs, CUDA and Python + CUDA, with 8-bit weights

**One tutorial, three tracks, one goal:** run **Qwen2.5-VL-7B** on your own GPU with **8-bit weights**, using CUDA kernels you wrote yourself and driving them from Python.

This is the only document you need. It contains the 15 phases, the theory course (used in Phases 2–4), and a glossary of every term at the end. File paths are relative to the project folder (`~/aiml/transformers/qwen_101/`, on git branch `qwen_101`). The reference implementation you check against is Hugging Face's `modeling_qwen2_5_vl.py` (in your installed `transformers`, under `models/qwen2_5_vl/`).

---

## How this tutorial works

### The three tracks

Every phase has up to three parts, always in this order:

| Icon | Track | What you do |
|---|---|---|
| 🧠 | **LLM** | Understand *what* the model computes, and why. |
| ⚙️ | **CUDA C++** | Write the kernel in a `.cu` file and check it against a CPU version. |
| 🐍 | **Python + CUDA** | Call *your* kernel from Python, run it on **real Qwen weights**, and check it against PyTorch. |

A fourth marker, **🎯 8-bit**, flags each place where the 8-bit theme comes in. By the end, every big weight matrix is stored in 8 bits.

### Why 8-bit weights?

- **It has to fit.** In bf16 (16-bit) Qwen2.5-VL-7B is 16.6 GB, and your GPU has 12 GB. In 8-bit it's about 8.3 GB, so it fits.
- **It's faster.** Generating each token means reading every weight once, so speed is limited by how many bytes you move. Half the bytes means roughly twice the speed.
- **The format: FP8 E4M3** (8-bit floating point), with one **scale** per output row. It's widely used for LLM inference, and your Blackwell GPU supports it in hardware. You'll also try **INT8** as a comparison.

### The rules

- **One small step at a time.** Each step is a numbered file in `cuda/`: `02_add.cu`, and from Phase 5 also `02_add.py`.
- **No build system.** You compile every file yourself, so you see everything.
- **Everything is checked.** Each kernel prints PASS or FAIL against a CPU version (⚙️) or PyTorch (🐍).
- **Do the "try this" exercises** before moving on. That's where the learning happens.
- **Mark progress** by changing ⬜ to ✅. 🏁 marks a milestone where something real works end to end.

### Folder layout

```
qwen_101/                        (~/aiml/transformers/qwen_101, branch qwen_101)
├── hf/Qwen2.5-VL-7B-Instruct/   the official model from Hugging Face (bf16, 16.6 GB)
├── hf/qwen-fp8/                 your 8-bit version (made in Phase 13)
├── cuda/                        your code: 01_hello.cu, 02_add.cu, 02_add.py, ...
└── ref/tutorial.md              this file (the Qwen3.5-Omni paper is arXiv 2604.15804; a local PDF copy is gitignored)
```

### Your machine

| | |
|---|---|
| GPU | RTX 5070: 12 GB, 48 SMs, Blackwell (`sm_120`), ≈ 672 GB/s memory bandwidth, 48 MB L2 cache. Has FP8 tensor cores. |
| CUDA | 13.2, compiler at `/usr/local/cuda/bin/nvcc`; profilers `nsys` and `ncu` included |
| RAM / disk | 29 GB RAM, about 400 GB free disk |
| Python | the `~/aiml/.aiml` environment (Python 3.12): PyTorch 2.14 with CUDA 13.2, `transformers`, `safetensors`, `accelerate`, `triton`, `numpy`, `Pillow` |
| Added later | `ninja` (Phase 8, for `load_inline`), `qwen-vl-utils` (Phase 15) |

**One-time setup:**
```
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'source ~/aiml/.aiml/bin/activate' >> ~/.bashrc
source ~/.bashrc
```

---

## The model you downloaded

`hf/Qwen2.5-VL-7B-Instruct/` is the official model. Its `config.json` gives these numbers, which you'll meet again and again.

### Language model (the LLM)

| Setting | Value | Meaning |
|---|---|---|
| `num_hidden_layers` | 28 | transformer blocks |
| `hidden_size` | 3,584 | numbers per token |
| `num_attention_heads` | 28 | query heads, each 128 wide |
| `num_key_value_heads` | 4 | K/V heads (GQA: 7 queries share each) |
| `intermediate_size` | 18,944 | MLP width |
| `vocab_size` | 152,064 | tokens it knows |
| `rms_norm_eps` | 1e-6 | RMSNorm's small safety number |
| `rope_theta` | 1,000,000 | RoPE base |
| `mrope_section` | [16, 24, 24] | M-RoPE split of each head's 64 rotation pairs: time / height / width |
| `tie_word_embeddings` | false | the input embedding table and the output LM head are **separate** matrices |
| `temperature` | 1e-6 | effectively greedy: always pick the top token (argmax) |

### Vision encoder

| Setting | Value | Meaning |
|---|---|---|
| `depth` | 32 | vision blocks |
| `hidden_size`, `num_heads` | 1,280, 16 | 16 heads × 80 |
| `intermediate_size` | 3,420 | vision MLP width |
| `patch_size`, `temporal_patch_size` | 14, 2 | 14×14-pixel patches, 2 frames |
| `window_size` | 112 | window attention over 112×112 pixels = 8×8 patches |
| `fullatt_block_indexes` | [7, 15, 23, 31] | the only 4 blocks with full attention |
| `spatial_merge_size` | 2 | the merger joins 2×2 patches |
| `out_hidden_size` | 3,584 | merger output = LLM width |

### Weight names (block 0 shown; the other blocks are the same)

```
model.embed_tokens.weight                         [152064, 3584]   input embedding table
model.layers.0.input_layernorm.weight             [3584]           RMSNorm before attention
model.layers.0.self_attn.q_proj.weight / .bias    [3584, 3584]     Q
model.layers.0.self_attn.k_proj.weight / .bias    [512, 3584]      K (4 heads × 128)
model.layers.0.self_attn.v_proj.weight / .bias    [512, 3584]      V
model.layers.0.self_attn.o_proj.weight            [3584, 3584]     output projection
model.layers.0.post_attention_layernorm.weight    [3584]           RMSNorm before MLP
model.layers.0.mlp.gate_proj.weight               [18944, 3584]    SwiGLU gate
model.layers.0.mlp.up_proj.weight                 [18944, 3584]    SwiGLU up
model.layers.0.mlp.down_proj.weight               [3584, 18944]    SwiGLU down
model.norm.weight                                 [3584]           final RMSNorm
lm_head.weight                                    [152064, 3584]   scores every vocabulary token

visual.patch_embed.proj.weight                    Conv3D patch embedding
visual.blocks.0.norm1 / norm2                     RMSNorms
visual.blocks.0.attn.qkv.weight / .bias           Q, K, V fused into one matrix
visual.blocks.0.attn.proj.weight / .bias          output projection
visual.blocks.0.mlp.{gate,up,down}_proj           SwiGLU (with biases)
visual.merger.ln_q, visual.merger.mlp.0 / .2      patch merger: RMSNorm, Linear, GELU, Linear
```

PyTorch stores linear weights as `[out, in]`. Keep that in mind when you write matrix multiplies.

### 🎯 Memory budget in 8 bits

| Part | Parameters | bf16 | FP8 |
|---|---|---|---|
| 28 LLM blocks (Q, K, V, O, gate, up, down) | 6.53 B | 13.1 GB | **6.5 GB** |
| LM head | 0.55 B | 1.1 GB | **0.55 GB** |
| Input embedding table | 0.55 B | 1.1 GB | 0.55 GB, or keep it on the CPU: only a few rows are read per token |
| Vision encoder + merger | 0.67 B | 1.3 GB | **0.67 GB** |
| **Total** | **8.29 B** | **16.6 GB** ❌ | **≈ 8.3 GB** ✅ |

That leaves about 3 GB for the KV cache, activations and CUDA's own overhead.

---

## The 15 phases at a glance

| # | Phase | Steps | Rough effort | Ends with |
|---|---|---|---|---|
| 1 | Setup and the model on disk | – | done ✅ | Toolchain working, Qwen downloaded |
| 2 | 🧠 Transformers and attention | – | 1 hour | You can do attention by hand |
| 3 | 🧠 How an LLM generates text, and why 8 bits | – | 1 hour | KV cache, GQA, the memory-bound argument |
| 4 | 🧠 Vision-language end to end | – | 2 hours | You can trace an image to the first word |
| 5 | CUDA basics + calling CUDA from Python | 1–3 | 2–3 evenings | 🏁 Your kernel, called from Python, on a real Qwen tensor |
| 6 | Fusion and reductions | 4–7 | 3–4 evenings | SwiGLU, RMSNorm, softmax, argmax on real weights |
| 7 | Matrix multiply I | 8–10 | 3–4 evenings | Tiled GEMM, decode GEMV, your tokens/second ceiling |
| 8 | Matrix multiply II: tensor cores, bf16, `load_inline` | 11–12 | 3–4 evenings | 🏁 Tensor-core GEMM as a PyTorch function |
| 9 | Positions and lookups | 13–15 | 2–3 evenings | RoPE, embedding, M-RoPE checked against Hugging Face |
| 10 | Attention I | 16–18 | 3–4 evenings | Multi-head GQA attention matching PyTorch |
| 11 | Attention II | 19–20 | 3–4 evenings | 🏁 KV-cache decode and fused attention |
| 12 | 🎯 8-bit weights | 21–22 | 4–5 evenings | FP8 and INT8 quantization, FP8 GEMM and GEMV |
| 13 | 🎯 The real model in 8 bits, one block | 23–25 | about 1 week | Qwen in FP8 on disk; block 0 matches Hugging Face |
| 14 | The full language model | 26–28 | about 1 week | 🏁 **Text chat with Qwen2.5-7B in 8 bits, on your kernels** |
| 15 | Vision, then expert | 29–33 | about 1–2 weeks | 🏁 **Describe a photo**, then profile and optimise |

Times are rough guides for sessions of a few hours. Go at your own pace; understanding beats speed.

---

## Phase 1: Setup and the model on disk ✅

**Goal:** a working toolchain, the real model on disk, and a map of the model.

| Task | Status |
|---|---|
| RTX 5070 + CUDA 13.2 working | ✅ |
| Repo organised: `cuda/` (your code), `hf/` (models), `ref/` (study) | ✅ |
| Step 1, `cuda/01_hello.cu`: first kernel ran | ✅ |
| Qwen2.5-VL-7B-Instruct downloaded to `hf/Qwen2.5-VL-7B-Instruct/` | ✅ |

**Orientation (30 minutes):** just look; you'll come back to each of these later.
- `hf/Qwen2.5-VL-7B-Instruct/config.json`: compare it with the tables above.
- `modeling_qwen2_5_vl.py` (find it with `python -c "import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m; print(m.__file__)"`): the reference implementation.
- In it, `Qwen2_5_VLDecoderLayer.forward()`: one transformer block in about 30 lines.
- `Qwen2MLP`, `Qwen2_5_VLRMSNorm`, `Qwen2_5_VLAttention`: the pieces you'll write a CUDA kernel for, one by one.

**How the model was downloaded** (already done, for reference):
```
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir hf/Qwen2.5-VL-7B-Instruct
```

---

## Phase 2: 🧠 Transformers and attention (1 hour)

**Goal:** understand attention well enough to explain it to someone else.

**Work:** Hour 1 of the theory course below: two 3Blue1Brown videos, then attention for 3 tokens worked out by hand. Glossary sections A, B and C.

**Done when** you can answer, without notes:
- What are Query, Key and Value, in one sentence each?
- Why does softmax come after Q·Kᵀ?
- What does a causal mask stop, and why?
- Why many heads instead of one?

---

## Phase 3: 🧠 How an LLM generates text, and why 8 bits (1 hour)

**Goal:** understand a modern transformer block and the tricks that make generation fast. This is where the case for 8-bit weights comes from.

**Work:** Hour 2 of the theory course below. Glossary sections D, E, F and H.

**🎯 The 8-bit argument, in four lines:**
1. Generating one token reads **every weight once**: about 16.6 GB in bf16.
2. Your GPU reads memory at about 672 GB/s, so the best case is about 40 tokens/second in bf16, *if it even fit*.
3. In 8 bits it's about 8.3 GB, so the ceiling is about **80 tokens/second**, and it fits.
4. The math is cheap by comparison. **Bytes are the bottleneck**, so fewer bytes per weight is the biggest single win.

**Done when** you can explain prefill vs decode, the KV cache, GQA's 7× saving, RMSNorm, residuals, SwiGLU, and the four lines above.

---

## Phase 4: 🧠 Vision-language end to end (2 hours)

**Goal:** trace one image from pixels to the first generated word, and know how the Hugging Face code does it.

**Work:** Hours 3 and 4 of the theory course below. Glossary sections G and I.

**Done when** you can:
- derive 540×360 → 988 patches → 247 tokens → 273 LLM tokens
- explain why the first text token after the image has position 34, not 262
- explain why one shared attention mask for all 32 vision blocks can't be right, given `fullatt_block_indexes = [7, 15, 23, 31]`
- answer the **Expert self-test** at the end of the theory course

---

## Phase 5: CUDA basics + calling CUDA from Python

**Goal:** move data to the GPU, run a kernel, get the answer back, know whether it was fast, and call your own kernel from Python.

**🧠 Why it matters:** every operation in the model follows this pattern. The residual add alone runs twice in each of the LLM's 28 blocks and the vision encoder's 32.

| Step | Files | Status |
|---|---|---|
| 1 | `01_hello.cu` | ✅ |
| 2 | `02_add.cu`, `02_add.py` | ⬜ next |
| 3 | `03_errors_timing.cu`, `03_real_tensor.py` | ⬜ |

**Step 1: hello from the GPU** ✅
- ⚙️ A kernel (`__global__`), launching `<<<blocks, threads>>>`, `threadIdx` / `blockIdx`, `cudaDeviceSynchronize()`.
- *Try this:* change the launch shape; remove the sync; launch 64 threads and look at the print order (warps of 32).

**Step 2: residual add**, `c[i] = a[i] + b[i]` on a 273 × 3,584 tensor
- ⚙️ CPU (host) vs GPU (device) memory; `cudaMalloc`, `cudaMemcpy`, `cudaFree`; the global index `blockIdx.x * blockDim.x + threadIdx.x`; rounding the grid up and the `if (i < n)` bounds check; comparing against a CPU loop.
- 🐍 **Your first Python + CUDA program.** Add a small C-style launcher to the `.cu` file, compile it into a shared library, and call it from Python with `ctypes`. PyTorch provides the GPU memory and the reference answer:
  ```cuda
  // at the bottom of 02_add.cu
  extern "C" void add_launch(const float *a, const float *b, float *c, int n)
  {
      add<<<(n + 255) / 256, 256>>>(a, b, c, n);
  }
  ```
  ```
  nvcc -shared -Xcompiler -fPIC 02_add.cu -o lib02_add.so
  ```
  ```python
  # 02_add.py
  import ctypes, torch
  lib = ctypes.CDLL("./lib02_add.so")
  a = torch.randn(273 * 3584, device="cuda")
  b = torch.randn_like(a)
  c = torch.empty_like(a)
  lib.add_launch(ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()),
                 ctypes.c_void_p(c.data_ptr()), ctypes.c_int(a.numel()))
  torch.cuda.synchronize()
  print("PASS" if torch.equal(c, a + b) else "FAIL")
  ```
  This is how real inference engines are built: a Python host drives a compiled kernel library. The final engine in Phase 14 works the same way.
- *Try this:* block sizes 32 / 128 / 256 / 1024; remove the bounds check with an `n` that isn't a multiple of 256, then run under `compute-sanitizer ./add` to see the bug caught; in Python, pass a CPU tensor by mistake and see what happens.
- *Done when:* PASS in both C++ and Python, and you can explain why the grid has to be rounded up.

**Step 3: errors, timing, and a real Qwen tensor**
- ⚙️ Every CUDA call returns an error code, so write your own `CUDA_CHECK` macro; launch errors come from `cudaGetLastError()`; timing with CUDA events; warm-up runs; **bandwidth = bytes moved ÷ time**.
- 🐍 Open the real model with `safetensors` and look at a weight:
  ```python
  from safetensors import safe_open
  f = safe_open("hf/Qwen2.5-VL-7B-Instruct/model-00001-of-00005.safetensors", "pt")
  w = f.get_tensor("model.layers.0.mlp.down_proj.weight")   # [3584, 18944], bf16
  ```
  Print its shape, dtype, size in MB, min, max, mean and standard deviation, and plot a histogram of the values.
- 🎯 *Look closely:* most values are tiny (around ±0.02), but a few are much larger. Write down the largest |value| in each row. In Phase 12 that number becomes the row's **scale**.
- *Try this:* small arrays show *more* than 672 GB/s. Why? (The L2 cache.) Grow the array until the number drops.
- 🏁 *Milestone:* your own kernel, called from Python, on GPU memory, checked against PyTorch.

---

## Phase 6: Fusion and reductions

**Goal:** kernels where threads work together, and where doing two steps in one kernel saves time.

**🧠 Why it matters:** RMSNorm, softmax and argmax each combine a whole row of numbers into one result (a sum or a max). That's a **reduction**, the second basic GPU pattern after "one thread per element".

| Step | Files | Status |
|---|---|---|
| 4 | `04_silu_mul.cu`, `04_silu_mul.py` | ⬜ |
| 5 | `05_rmsnorm.cu`, `05_rmsnorm.py` | ⬜ |
| 6 | `06_softmax.cu`, `06_softmax.py` | ⬜ |
| 7 | `07_argmax.cu`, `07_argmax.py` | ⬜ |

**Step 4: fused SiLU × mul** (the SwiGLU core, 273 × 18,944)
- ⚙️ `__device__` helpers; `expf` vs fast `__expf`; **fusion** (a SiLU kernel followed by a separate multiply kernel writes and re-reads a 20 MB tensor; one fused kernel avoids that); grid-stride loops; sizing the grid from the SM count.
- 🐍 Compare against `torch.nn.functional.silu(g) * u`.
- *Try this:* write it as two kernels first, time both, and explain the gap using bytes moved.

**Step 5: RMSNorm** (273 × 3,584)
- ⚙️ One block per row; **warp shuffles** (`__shfl_down_sync`); **shared memory**; `__syncthreads()`; `rsqrtf`.
- 🐍 Use the **real** `model.layers.0.input_layernorm.weight` and eps = 1e-6; compare with your own 3-line PyTorch RMSNorm.
- *Try this:* write a fused **add + RMSNorm**, the exact pair between attention and the MLP.

**Step 6: softmax** (rows of 273, like attention scores)
- ⚙️ Why naive `exp(x) / sum` overflows; the **stable** version (subtract the max first); the **online softmax** (running max and sum in one pass), reused in step 20.
- 🐍 Compare against `torch.softmax`, including rows with huge values.

**Step 7: argmax** over 152,064 logits (picking the next word)
- ⚙️ Reducing (value, index) pairs; a row too long for one block, so two stages; atomics as an alternative.
- 🐍 Compare with `torch.argmax`. Then run it on real logits: take a random hidden vector, multiply by the real `lm_head.weight` in PyTorch, argmax with your kernel, and decode the token ID into a word with the tokenizer (`AutoTokenizer.from_pretrained("hf/Qwen2.5-VL-7B-Instruct")`). It's gibberish, but it's a real word from Qwen's vocabulary.

---

## Phase 7: Matrix multiply I: the memory game

**Goal:** understand why matrix multiply is *the* operation that matters, why the naive version is slow, and your speed ceiling.

**🧠 Why it matters:** Q, K, V, O, the three MLP layers and the LM head are all matrix multiplies, over 95% of the model's work. Prefill (273 tokens) is a matrix × matrix multiply (GEMM). Decode (1 token) is a matrix × vector multiply (GEMV).

| Step | Files | Status |
|---|---|---|
| 8 | `08_gemm_naive.cu`, `08_gemm_naive.py` | ⬜ |
| 9 | `09_gemm_tiled.cu`, `09_gemm_tiled.py` | ⬜ |
| 10 | `10_gemv.cu`, `10_gemv.py` | ⬜ |

**Step 8: naive GEMM** (273 × 3,584 times the real Q weight, `[3584, 3584]`)
- ⚙️ One thread per output; 2D grids (`dim3`); FLOPs and TFLOP/s; arithmetic intensity; **memory coalescing**; the `[out, in]` weight layout means you compute `x · Wᵀ`.
- 🐍 Compare with `x @ W.T` on the real `q_proj.weight`.
- *Try this:* swap which index maps to `x` and `y`, and watch the speed change several times over.

**Step 9: tiled GEMM with shared memory**
- ⚙️ Tiles of A and B loaded once into shared memory and reused by the whole block; register tiling; occupancy.
- *Done when:* clearly faster than step 8, and you can explain why using bytes loaded per FLOP.

**Step 10: GEMV, the decode multiply** (1 × 3,584 times the real `gate_proj`, `[18944, 3584]`)
- ⚙️ One token means each weight is used once, so speed = bandwidth; one warp per output row; vectorised 16-byte loads.
- 🐍 Time it on the real weight and compute GB/s.
- 🎯 **Your tokens/second ceiling.** Multiply out: (all weights in bytes) ÷ (your GB/s) = seconds per token. Do it for bf16 and for 8-bit. This one calculation predicts your final speed and is the clearest argument for 8 bits.

---

## Phase 8: Matrix multiply II: tensor cores, bf16, `load_inline`

**Goal:** use the GPU's matrix hardware, and turn your kernels into ordinary PyTorch functions.

**🧠 Why it matters:** prefill is limited by math, and tensor cores are where the math is.

| Step | Files | Status |
|---|---|---|
| 11 | `11_bf16.cu`, `11_load_inline.py` | ⬜ |
| 12 | `12_gemm_tensor_core.cu`, `12_gemm_tensor_core.py` | ⬜ |

**Step 11: bf16 numbers, and `load_inline`**
- ⚙️ The bf16 format (float's range, less precision); `__nv_bfloat16`; conversions; error vs float.
- 🐍 **Upgrade from `ctypes`.** `pip install ninja`, then use `torch.utils.cpp_extension.load_inline`: paste your CUDA code into a Python string and PyTorch compiles it into a module whose functions take and return tensors directly, with no pointers. Redo step 2 this way.
- *Try this:* compare the bf16 add's speed and accuracy with float.

**Step 12: tensor-core GEMM**
- ⚙️ Warp-level matrix ops (`wmma`, 16×16×16 tiles); bf16 inputs with float accumulation; feeding tensor cores from shared memory.
- 🐍 Run it on real Qwen weights in bf16; compare against `torch.matmul` (which uses NVIDIA's cuBLAS) for speed and accuracy.
- 🏁 *Milestone:* a tensor-core GEMM callable as a PyTorch function, benchmarked against cuBLAS.

---

## Phase 9: Positions and lookups

**Goal:** the small but essential kernels around the big multiplies, checked against the Hugging Face implementation itself.

| Step | Files | Status |
|---|---|---|
| 13 | `13_rope.cu`, `13_rope.py` | ⬜ |
| 14 | `14_embedding.cu`, `14_embedding.py` | ⬜ |
| 15 | `15_mrope.py` | ⬜ |

**Step 13: RoPE**
- ⚙️ Rotating pairs of numbers by a position-dependent angle; precomputed sin/cos tables; Qwen's "rotate half" layout (pairs are `i` and `i + 64`, not neighbours); θ = 1,000,000.
- 🐍 Compare with `apply_rotary_pos_emb` from `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl`.
- *Try this:* shift both Q and K positions by the same amount and check that their dot product doesn't change. That's the point of RoPE.

**Step 14: embedding lookup**
- ⚙️ Gathering rows by index; vectorised `float4` loads.
- 🐍 Tokenize "Hello, Qwen!" with the real tokenizer, look the IDs up in the real `model.embed_tokens.weight` with your kernel, and compare with PyTorch indexing.

**Step 15: M-RoPE positions**
- 🧠 Three position IDs per token (time, height, width); `mrope_section = [16, 24, 24]` splits each head's 64 rotation pairs; image tokens get their grid position; text after the image continues from max + 1.
- 🐍 Build the position IDs for the 273-token template yourself, then check against Hugging Face's `get_rope_index(...)` (on `Qwen2_5_VLModel`, i.e. `model.model` of the full model). Confirm the first text token after the image is position 34.
- *Done when:* you can explain why an engine with precomputed sin/cos tables needs separate prefill and decode tables (hint: M-RoPE positions after an image).

---

## Phase 10: Attention I: the core

**Goal:** working multi-head attention for prefill, built from your own kernels and matching PyTorch.

| Step | Files | Status |
|---|---|---|
| 16 | `16_attention_one_head.cu`, `.py` | ⬜ |
| 17 | `17_causal_mask.cu`, `.py` | ⬜ |
| 18 | `18_mha_gqa.cu`, `.py` | ⬜ |

**Step 16: one head:** Q·Kᵀ / √128 → softmax → × V, from your GEMM and softmax kernels.
- 🐍 Compare with `torch.nn.functional.scaled_dot_product_attention`.

**Step 17: causal mask:** add −∞ above the diagonal, or better, skip that work entirely.
- *Try this:* what fraction of the work does skipping save? Compare with adding a full −∞ mask matrix.

**Step 18: multi-head + GQA:** 28 query heads, 4 KV heads, 7 queries per K/V head, one launch for all heads, layout `[tokens, heads, 128]`.
- 🐍 Feed real Q/K/V from block 0 (computed in PyTorch from a real prompt) and compare with SDPA using `enable_gqa=True`.

---

## Phase 11: Attention II: KV cache and fast attention

**Goal:** fast generation one token at a time, and attention without huge intermediate grids.

| Step | Files | Status |
|---|---|---|
| 19 | `19_kv_cache_decode.cu`, `.py` | ⬜ |
| 20 | `20_flash_attention.cu`, `.py` | ⬜ |

**Step 19: KV cache decode**
- ⚙️ A preallocated cache `[28 layers, 4 heads, max length, 128]`; append the new K/V at the current row; one query against every stored row.
- 🐍 Run prefill + 10 decode steps and compare each step with full recomputation in PyTorch.
- 🎯 *Budget:* with 8.3 GB of 8-bit weights, how long a context fits in the rest of 12 GB? (32,768 tokens of KV cache cost about 1.9 GB in bf16.) Bonus question: what would an 8-bit KV cache buy you?

**Step 20: fused, Flash-style attention**
- ⚙️ Q·Kᵀ, softmax and × V in one kernel, tile by tile, using the online softmax from step 6, so the score grid never goes to memory.
- 🏁 *Milestone:* prefill and decode attention matching PyTorch, and measurably faster than step 18.

---

## Phase 12: 🎯 8-bit weights

**Goal:** store weights in 8 bits with almost no loss of quality, and multiply with them fast. This is the heart of the tutorial.

| Step | Files | Status |
|---|---|---|
| 21 | `21_quantize.py`, `21_fp8.cu` | ⬜ |
| 22 | `22_gemm_fp8.cu`, `22_gemm_fp8.py` | ⬜ |

**Step 21: quantization: FP8 vs INT8**
- 🧠 **FP8 E4M3:** 1 sign, 4 exponent and 3 mantissa bits; largest value 448; more precision near zero, where most weights are. **INT8:** 256 evenly spaced steps from −127 to 127.
- 🧠 **Scales:** weights are far smaller than 448, so each output row gets a scale: `scale = max|row| / 448` (FP8) or `/ 127` (INT8). Store `w / scale` in 8 bits, and multiply the scale back at the end (dequantize).
- 🐍 Quantize the **real** `down_proj` weight with `torch.float8_e4m3fn` and with INT8, each with one scale per tensor vs one per row. Measure the error of `x @ Wᵀ` against bf16. Which wins, and why?
- ⚙️ Write the FP8 ⇄ float conversion yourself (`__nv_fp8_e4m3` in `cuda_fp8.h`) and check it bit for bit against PyTorch's.

**Step 22: FP8-weight GEMM and GEMV**
- ⚙️ **Weight-only FP8:** read 8-bit weights, convert to bf16 in registers, multiply on tensor cores, apply the row scale at the end. Decode GEMV now moves half the bytes.
- ⚙️ *Optional:* Blackwell's native FP8 tensor-core instructions (FP8 × FP8), which also need the activations quantized.
- 🐍 Compare speed and accuracy against your bf16 kernels and against PyTorch's `torch._scaled_mm`.
- *Done when:* results are close to bf16 and decode GEMV is about 2× faster than step 10. Check it against the ceiling you calculated there.

---

## Phase 13: 🎯 The real model in 8 bits, one block

**Goal:** convert Qwen to 8 bits, load it into the GPU, and make one real block produce what Hugging Face produces.

| Step | Files | Status |
|---|---|---|
| 23 | `23_convert_fp8.py` | ⬜ |
| 24 | `24_load.py` | ⬜ |
| 25 | `25_one_block.py` + kernels so far | ⬜ |

**Step 23: convert to FP8**
- 🐍 Read every LLM and vision linear weight from `hf/Qwen2.5-VL-7B-Instruct/`, quantize to FP8 E4M3 with per-row scales, and save to `hf/qwen-fp8/` as safetensors (it supports `float8_e4m3fn`). Keep norms, biases and small tensors in bf16. Use your own judgment for the embedding table.
- *Done when:* the folder is about 8–9 GB and a reload gives back the same tensors.

**Step 24: load and plan memory**
- 🐍 Load `hf/qwen-fp8/` onto the GPU; print a memory map (weights, KV cache, activation buffers, free space) and check it against the budget table at the top.
- *Try this:* look at `torch.cuda.memory_summary()`. Where does the "overhead" go?

**Step 25: one transformer block**
- 🐍 Python wires your kernels together: RMSNorm → Q/K/V (+ bias) → RoPE → attention → O → add → RMSNorm → SwiGLU → add.
- 🐍 **Reference:** the full bf16 model doesn't fit on the GPU, so load *only block 0* of the Hugging Face model, in bf16, and run the same input through it.
- **Compare after every stage, not just at the end.** That's how you find bugs.
- *Done when:* block 0's output matches the bf16 reference within the small error FP8 introduces (measure and record it).

---

## Phase 14: The full language model 🏁

**Goal:** a working text chatbot: Qwen2.5-7B in 8 bits, on your kernels, driven by Python.

| Step | Files | Status |
|---|---|---|
| 26 | `26_llm.py` | ⬜ |
| 27 | `27_chat.py` | ⬜ |
| 28 | `28_cuda_graphs.py` | ⬜ |

**Step 26: all 28 blocks + final norm + LM head**
- 🐍 Reference: Hugging Face with `device_map="auto"` (spreads the bf16 model across GPU and CPU; slow but exact).
- *Done when:* for a fixed prompt, your top-5 next-token predictions match the reference.

**Step 27: the chat loop**
- 🐍 Apply the chat template with the tokenizer, prefill, then decode token by token with your KV cache and argmax, streaming words as they come. Stop at the end-of-turn token (151645).
- 🏁 *Milestone:* **type a question and Qwen2.5-7B answers, in 8 bits, on your own CUDA kernels.**

**Step 28: first speed pass**
- ⚙️ Launch overhead (hundreds of small kernels per token); **CUDA graphs** (record one decode step, then replay it); PyTorch exposes these as `torch.cuda.CUDAGraph`.
- 🐍 Measure tokens/second and compare with your ceiling from step 10.

---

## Phase 15: Vision, then expert 🏁

**Goal:** the full Qwen2.5-VL: show it a photo and it describes it. Then take everything to expert level.

| Step | Files | Status |
|---|---|---|
| 29 | `29_patch_embed.py` | ⬜ |
| 30 | `30_vision_blocks.py` | ⬜ |
| 31 | `31_merger.py` | ⬜ |
| 32 | `32_describe_image.py` | ⬜ |
| 33 | profiling and optimisation | ⬜ |

**Step 29: image to patches.** `pip install qwen-vl-utils`; resize and normalise with the processor (`image_mean` / `image_std` from `preprocessor_config.json`); cut into 14×14×2 patches; the Conv3D patch embedding is just a GEMM, so use your FP8 GEMM. Compare with `model.visual.patch_embed`.

**Step 30: the 32 vision blocks, done correctly.** Window attention (8×8-patch windows) in 28 blocks and **full attention in blocks 7, 15, 23 and 31**, exactly as `fullatt_block_indexes` says. A single mask shared by every block would be wrong. 2D RoPE, 16 heads × 80, fused `qkv`.

**Step 31: patch merger + putting tokens back in order.** RMSNorm → 2×2 merge → Linear (5,120 → 5,120) → GELU → Linear (→ 3,584). Undo the window reordering so the LLM sees image tokens in their real positions. Compare with `model.visual`.

**Step 32: image + text end to end.** Splice the image tokens into the chat template, use your M-RoPE positions from step 15, run your LLM.
- 🏁 *Milestone:* **give it a photo and it describes it, in 8 bits, fully on your kernels.** Bonus: support any image size (dynamic resolution).

**Step 33: expert pass**
- **Profile** with Nsight Systems (`nsys profile python 27_chat.py`, the timeline) and Nsight Compute (`ncu`, per-kernel detail). Predict the bottleneck *before* you measure.
- **Optimise** the top 3 hotspots: kernel fusion, better GEMV, an 8-bit KV cache.
- **Triton** (already installed): rewrite one kernel in Triton, a Python language for GPU kernels, and compare effort and speed with your CUDA version.
- **Compare designs:** yours vs vLLM or llama.cpp. Tensor parallel vs pipeline split, precomputed tables, fixed image size, FP8 scales: what did each choose, and why?
- **Read the frontier:** the Qwen3.5-Omni paper ([arXiv 2604.15804](https://arxiv.org/abs/2604.15804)): Mixture of Experts, Gated DeltaNet, chunked prefill, speech output. Sketch how you'd implement one of them.

**You're an expert when** you can explain every line of your implementation, predict a kernel's bottleneck before profiling it, choose a quantization scheme and defend it with measurements, and read a new model paper and know how you'd build it.

---

# Theory course (Phases 2–4): understand the model in 4 hours

Hour 1 = Phase 2, Hour 2 = Phase 3, Hours 3–4 = Phase 4.

### Before you start: the paper describes a different model than the one you're building

The paper, [arXiv 2604.15804](https://arxiv.org/abs/2604.15804), is the **Qwen3.5-Omni Technical Report** (April 2026). This tutorial runs **Qwen2.5-VL-7B**, which is two generations older and built quite differently:

| | Qwen2.5-VL-7B (this tutorial) | Your paper (Qwen3.5-Omni) |
|---|---|---|
| Inputs | image + text | text, image, video, **audio** |
| Outputs | text | text **and speech** |
| LLM design | dense transformer, every layer uses attention | **Hybrid MoE** with Gated DeltaNet layers |
| Vision encoder | Qwen's own ViT (windowed attention) | **SigLIP2** |
| Size | 7B | "hundreds of billions" |
| Position encoding | M-RoPE | TM-RoPE plus timestamps written as text |

**What this means for you:**
- Only about **5 of its 28 pages are useful** here: §1 Introduction and §2 Architecture, pages 1–6.
  - §3–4 cover training, §5 and the appendix are benchmark tables. Skip all of them.
- The paper gives **no layer-level details** for any of its parts. It never says how many layers, heads or dimensions it uses. So the details of *this* model come from the **Glossary** at the end of this file and from the Hugging Face code.

It's still worth reading. It shows where this model family went next, and a few passages directly explain things in Qwen2.5-VL (marked ★ below). If you meant to copy the Qwen2.5-VL report (arXiv 2502.13923), that one matches this model exactly. Swap it in for Hour 3 if you can get it.

---

### The 4 hours

#### Hour 1: How a transformer thinks *(unchanged)*
| Time | Do |
|---|---|
| 0:00–0:50 | 3Blue1Brown, Deep Learning chapters 5 and 6 (transformers, attention). |
| 0:50–1:00 | Work out attention by hand for 3 tokens: scores → causal mask → softmax → × V. |

#### Hour 2: How an LLM generates text fast *(adds the paper's speed vocabulary)*
| Time | Do |
|---|---|
| 1:00–1:20 | Re-read **Glossary** sections C–F and H (end of this file). |
| 1:20–1:40 | Read `Qwen2_5_VLDecoderLayer` and `Qwen2MLP` in `modeling_qwen2_5_vl.py` (the SwiGLU steps). |
| 1:40–1:50 | Calculate the KV cache size: ≈ 44 MB with GQA, ≈ 310 MB without. Then convince yourself that decode speed is limited by memory bandwidth. |
| 1:50–2:00 | **New, paper §2.5 (pages 5–6).** Learn its speed metrics and map them to Qwen2.5-VL. |

The paper's speed metrics, mapped to what you'll build:
- **TTFT** (time to first token) = the time for the prefill pass.
- **TPOP** (time per output token) = the time for one decode step.
- **Chunked prefill**: prefill in pieces as input streams in. Your engine will prefill the whole prompt in one pass instead.
- 🎯 The same memory-bound fact is why this tutorial uses 8-bit weights: half the bytes per weight, roughly twice the decode speed (Phase 3).
- ★ The paper says its GDN layers "reduce KV-cache I/O overhead". That is the exact bottleneck you just calculated. You now know *why* they built it.

#### Hour 3: How vision and position get into the LLM *(paper reading moves here)*
| Time | Do |
|---|---|
| 2:00–2:25 | **Read paper §1 and §2.1–2.3 (pages 1–5).** Focus on Figure 2 and the **"Audio-visual Timestamp"** paragraph. |
| 2:25–2:40 | Read `Qwen2VLImageProcessor` (resize, normalise, patches) and `Qwen2_5_VLPatchMerger` in `modeling_qwen2_5_vl.py`. |
| 2:40–3:00 | **Hands-on:** derive 540×360 → 532×364 → 988 patches → 247 tokens → 273 tokens. |

**★ The paragraph that explains Qwen2.5-VL's positions.** §2.3 says:

> "each subsequent modality commencing from one plus the maximum position ID of the preceding modality"

That's the M-RoPE rule Qwen2.5-VL uses. After the 19×13 image, text positions don't continue from row 273. They continue from *(max image position + 1)*. That's why decode positions can't just use the row number.

**Exercise:** work out the position IDs yourself.
- The 15 text tokens before the image get positions 0–14.
- The image grid takes 13 rows × 19 columns, starting at 15.
- So the next text token starts at 15 + max(13, 19) = **34**, not 262.

If you get that, you understand M-RoPE.

**Contrast to note:** the paper's newer model *abandons* tying time positions to absolute time for long videos, because the position IDs become "excessively sparse", and writes timestamps as text instead. Qwen2.5-VL still used absolute-time positions, but this tutorial only handles single images, so that issue never comes up here.

#### Hour 4: The Hugging Face reference, then where the field is going *(split 40/20)*
| Time | Do |
|---|---|
| 3:00–3:25 | Read `Qwen2_5_VLModel.forward()` in `modeling_qwen2_5_vl.py`: where the image tokens are spliced in, and where prefill ends and decode begins. |
| 3:25–3:40 | Trace one decode step: new token id → embedding → Q/K/V → append to the KV cache → attention → MLP → LM head → argmax → next id. |
| 3:40–4:00 | **New, paper §2.2, §2.4, Table 1 (pages 3–6).** Learn the frontier vocabulary (below) and contrast it with Qwen2.5-VL. |

**Frontier vocabulary from the paper:**

| Term | Plain meaning | What Qwen2.5-VL does instead |
|---|---|---|
| **MoE (Mixture of Experts)** | many MLPs ("experts"); each token uses only a few, so the model is huge but each token's compute stays cheap | one dense MLP per block |
| **Hybrid attention / Gated DeltaNet** | most layers swap attention for a fixed-size "running memory" with no growing KV cache; a few layers keep normal attention | attention in every layer, plus a KV cache |
| **Thinker / Talker** | one model thinks in text, a second turns that into speech tokens | text only |
| **AuT** | audio encoder that turns a mel-spectrogram (a sound "image") into tokens, 1 every 160 ms | none |
| **SigLIP2** | a widely used off-the-shelf vision encoder | Qwen's own windowed ViT |
| **RVQ codec, MTP, Code2Wav** | speech as stacked "codebook" tokens; predict several per step; a ConvNet renders the waveform | none |
| **ARIA** | keeps the text and speech token streams in step | none |

**Final challenge (unchanged).** Explain why a single attention mask shared by all 32 vision blocks would conflict with "windowed in 28 layers, full in 4". Note that this fact comes from the Qwen2.5-VL report, not the Omni paper.

---

### Expert self-test
1. Walk through what happens from "image in" to "first word out".
2. For a 540×360 image and a short prompt, why 273 prefill tokens, and what limits the number of decode steps?
3. What are GQA, the KV cache, RoPE/M-RoPE, RMSNorm and SwiGLU, and *why* does each exist?
4. Why FP8 weights plus scale files?
5. Tensor-parallel vs graph-split: what's the trade-off?
6. What breaks if the KV cache appends at the wrong row?
7. Name 2 places your engine will differ from the Hugging Face reference.
8. **New:** Why does the first text token after the image get position 34 and not 262?
9. **New:** Which two Qwen3.5-Omni ideas attack the problems you calculated in Hour 2 (KV-cache size, per-token compute)? Answer: GDN and MoE.

---

# Glossary: every term, in plain English

Terms are grouped from the big picture down to the smallest pieces. Each entry says what the thing is, in simple terms, and where it appears in this model.

## A. The big picture

**Neural network.** A long chain of simple math steps (mostly multiplying and adding numbers) whose settings were learned from huge amounts of data. You feed numbers in one end and get numbers out the other.

**Weights (parameters).** The learned numbers inside the network, its "knowledge". Qwen2.5-VL-7B has about 7 billion of them. Here they live in `hf/Qwen2.5-VL-7B-Instruct/*.safetensors`.

**Layer.** One step in the chain. Data passes through layers one after another, and each layer refines it a little.

**Block (transformer block).** A standard bundle of layers that gets repeated many times. The vision part repeats its block 32 times and the language part 28 times. In `config.json`: `depth = 32` (vision) and `num_hidden_layers = 28` (language).

**Transformer.** The overall design that both halves of this model use. It is built from stacked blocks, and each block contains **attention** plus an **MLP**.

**LLM (Large Language Model).** A transformer trained to predict the next word. Here that is Qwen2.5-7B.

**VLM (Vision-Language Model).** An LLM that can also look at images. Qwen2.5-VL = vision encoder + merger + LLM.

**Inference.** Running an already-trained model to get answers. That is all this tutorial does; it never trains.

## B. Turning things into numbers

**Token.** The unit the model reads. For text, a token is roughly a word piece: "unbelievable" might become "un", "believ", "able". For images, a token is a small square patch of the picture.

**Tokenizer.** The tool that splits text into tokens and gives each one an ID number. Qwen knows 152,064 different tokens, which is the vocabulary.

**Embedding.** A list of numbers that represents one token, like coordinates on a map of meaning. Words with similar meanings sit close together. In this LLM, each token becomes 3,584 numbers.

**Embedding table.** A giant lookup table with one row per vocabulary token (152,064 rows × 3,584 numbers). Weight: `model.embed_tokens.weight`.

**Hidden size / dimension (`hidden_size`).** How many numbers describe each token inside the network: 1,280 in the vision part and 3,584 in the language part.

**Tensor.** A grid of numbers with any number of dimensions. For example, `[1, 273, 3584]` means 1 sequence of 273 tokens with 3,584 numbers each. Everything the code passes around is a tensor.

**Sequence length.** How many tokens are processed together: 988 image patches for the vision part, and 273 tokens for the language part's first pass.

## C. Attention, the core idea

**Attention.** The step where each token "looks at" the other tokens to decide which ones matter to it. In "The cat sat because **it** was tired", attention lets "it" link to "cat".

**Query, Key, Value (Q, K, V).** Each token produces three things:
- **Query**: "what am I looking for?"
- **Key**: "what do I contain?" (like a label)
- **Value**: "here is my actual information"

Each Query is compared against every Key, and good matches mean that token's Value counts more. In the weights these are `q_proj`, `k_proj` and `v_proj`.

**Q·Kᵀ (attention scores).** Every Query is compared with every Key, giving one score per pair.

**Softmax.** Turns the scores into percentages that add up to 100%. A high score gets most of the attention.

**Weighted sum (QKᵀ·V).** Each token's new value is a blend of the other tokens' Values, mixed by those percentages.

**Multi-Head Attention (MHA).** Attention is run several times in parallel, and each copy is called a **head**. Different heads learn to notice different relationships. The LLM has 28 heads and the vision part has 16.

**Head width (head_dim).** How many numbers each head works with: 128 in the LLM (28 × 128 = 3,584) and 80 in the vision part (16 × 80 = 1,280).

**Grouped-Query Attention (GQA).** A memory-saving trick: several Query heads share one Key/Value head. In Qwen, 7 Query heads share each KV head, so there are only 4 KV heads instead of 28. In `config.json`: 28 `num_attention_heads`, 4 `num_key_value_heads`.

**Output projection (`o_proj`).** After attention, one more matrix multiplication mixes the heads' results back together.

**Attention mask.** A grid that blocks certain tokens from looking at others by setting their score to −∞, which becomes 0% after softmax.

**Causal mask.** The mask used in the LLM: a token can only look at earlier tokens, never later ones. Without it the model could peek at the answer.

**Window attention.** Used in the vision part: each image patch only looks at patches in its own small region (a 112×112-pixel window) instead of the whole image. It is much cheaper.

**Full attention.** Every token looks at every other token. The Qwen2.5-VL vision encoder uses it in only 4 of its 32 layers (7, 15, 23, 31), so the whole image gets connected together.

## D. Knowing where things are (position)

**Positional encoding.** Attention on its own doesn't know word order: "dog bites man" and "man bites dog" look the same to it. Positional encoding stamps each token with its location.

**RoPE (Rotary Position Embedding).** Qwen's way of doing this. It rotates the Query and Key numbers by an angle that depends on position, so two tokens' relative distance shows up in their attention score.

**Sin/Cos tables.** RoPE's rotation angles, computed ahead of time and computed once and reused, so they aren't recomputed at every step.

**2D RoPE.** The vision version: each patch gets a row position and a column position.

**M-RoPE (Multimodal RoPE).** The language model's version. Every token gets three position numbers: time, height and width. Text tokens use the same number for all three. Image tokens use their grid location. After an image, text continues from (largest image position + 1).

## E. The "thinking" layer: MLP

**MLP (Multi-Layer Perceptron), also called the feed-forward network.** After attention gathers information from other tokens, the MLP processes each token on its own. It is where much of the model's stored knowledge lives.

**Expand then shrink.** The MLP blows each token up to a larger size (3,584 → 18,944 numbers in the LLM), works on it there, then shrinks it back down.

**Activation function.** A simple non-linear "bend" applied to numbers. Without it, stacking layers would be no more powerful than a single layer.
- **SiLU.** A smooth function that mostly passes positive numbers through and squashes negative ones.
- **GELU.** Similar idea, slightly different curve. Used in the vision-to-language merger.

**SwiGLU (gated MLP).** Qwen's MLP style. Two parallel paths: one goes through SiLU and acts as a "gate", the other doesn't. They are multiplied together, so the gate decides how much passes. In the weights: `gate_proj` (gate), `up_proj` (the other path), multiply, then `down_proj` (shrink back down).

## F. Keeping numbers stable

**Normalization.** Rescales a token's numbers so they don't drift too big or too small through dozens of layers.

**RMSNorm.** Qwen's normalization: divide each token's numbers by their root-mean-square, then multiply by learned weights.

**LayerNorm.** An older variant that also subtracts the average. Used by GPT-2 (and tiny-gpt), not by Qwen.

**Pre-norm.** Qwen normalizes *before* attention and *before* the MLP:
`Norm → Attention → add back → Norm → MLP → add back`

**Residual connection ("add back").** Each layer's output is *added* to its input instead of replacing it, so the original information is never lost.

**Bias.** A learned number added after a multiplication. Qwen uses biases on Q, K and V.

## G. Vision-specific pieces

**Vision Transformer (ViT).** A transformer that reads image patches instead of words.

**Patch.** The image is cut into 14×14-pixel squares, and each square becomes one token. Here the image is always resized so it gives 38 × 26 = 988 patches.

**Patch embedding (Conv3D).** Turns each patch's raw pixels into 1,280 numbers. "3D" because it was designed for video and processes 2 frames at a time; a still image is duplicated. In Hugging Face: `Qwen2_5_VisionPatchEmbed`.

**Dynamic resolution.** The Qwen2.5-VL feature where any image size is accepted. This port gives that up and always uses 540×360.

**Patch merger.** Groups each 2×2 set of neighbouring patches into one token (988 → 247), then a small MLP converts them to the LLM's size (1,280×4 → 3,584). The bridge between seeing and talking.

**Window reordering.** Rearranges patches so those in the same window sit next to each other, which makes window attention efficient.

**Special tokens.** Markers like `<|vision_start|>` and `<|vision_end|>` that tell the LLM "an image starts / ends here".

**Chat template.** The fixed wrapper text around your message. Here it totals 273 tokens: 15 before the image, 247 image tokens, 11 after.

## H. Producing an answer

**Final norm + LM head.** After the last block, one more RMSNorm, then a huge matrix multiplication that scores all 152,064 vocabulary tokens. These scores are **logits**.

**Argmax / greedy decoding.** Pick the single highest-scoring token. Your engine starts with exactly that: same answer every time.

**Sampling (temperature, top-p).** The alternative: pick randomly among likely tokens. Not used here.

**Autoregressive generation.** The model writes one token at a time, then feeds it back in to produce the next.

**Prefill.** The first pass: process the whole prompt (273 tokens) at once.

**Decode (non-prefill, "NPF").** Every later pass processes just one new token, up to 500 times here (`NumIterations = 500`).

**KV cache.** During decode, the Keys and Values of all earlier tokens are saved so they aren't recomputed.

**Row index.** Tells the decode step which KV-cache slot the new token goes into, and which position to use.

**Max cache length.** The longest conversation the cache can hold, e.g. 273 prompt tokens + 500 new ones = 773.

## I. Hardware and efficiency terms

**GEMM (General Matrix Multiply).** The core math operation, a big grid-times-grid multiplication. Most of the model's work is GEMMs.

**Kernel.** A small program that runs on the accelerator and does one operation (Softmax, RMSNorm, GEMM, …). On NVIDIA GPUs, a CUDA kernel: the `__global__` functions you write.

**Tiled.** The kernel splits big tensors into small tiles that fit in the GPU's fast on-chip memory (shared memory).

**Graph (CUDA graph).** A whole sequence of kernel launches recorded once and replayed every step, which removes the per-launch overhead.

**bfloat16 (bf16).** A compact 16-bit number format: half the memory of 32-bit numbers, slightly less precise. Used for activations.

**FP8 (E4M3).** An 8-bit floating-point format used to store weights compactly: 1 sign bit, 4 exponent bits, 3 mantissa bits, largest value 448. More precision near zero, where most weights are. PyTorch: `torch.float8_e4m3fn`; CUDA: `__nv_fp8_e4m3`.

**INT8.** An 8-bit integer format: 256 evenly spaced steps (−127 to 127). The alternative to FP8 for 8-bit weights.

**Quantize / dequantize.** Quantize: squeeze numbers into fewer bits (divide by a scale, round to 8 bits). Dequantize: turn them back (multiply by the scale).

**Per-row (per-channel) scale.** One scale for each output row of a weight matrix, instead of one for the whole matrix. More accurate, because each row keeps its own range.

**Weight-only quantization.** Only the weights are 8-bit; activations stay bf16. Weights are converted back to bf16 inside the kernel, right before the multiply. Saves memory and bandwidth, which is what matters for decode.

**Scale.** A multiplier stored next to FP8 weights to restore their real size.

**Activations.** The intermediate numbers flowing between layers, as opposed to the fixed weights.

**Tensor parallelism.** Splitting each layer across several chips, e.g. 7 attention heads per chip on 4 chips.

**Graph split.** Cut the model into chunks of consecutive blocks and run each chunk on a different device in sequence.

**Host.** The regular computer (CPU) that controls the accelerator.


## J. Python + CUDA

**PyTorch.** A Python library for tensors on the GPU. Here it's the *reference* (the "right answer" your kernels are checked against) and the *memory manager* (it allocates GPU tensors your kernels work on).

**Hugging Face / `transformers`.** The library and website the official Qwen model comes from. `transformers` holds the reference implementation (`modeling_qwen2_5_vl.py`).

**safetensors.** The file format the weights are stored in: a simple, safe list of named tensors. `safe_open` reads one tensor at a time without loading the whole file.

**Shared library (`.so`).** Compiled code other programs can load. `nvcc -shared -Xcompiler -fPIC x.cu -o libx.so` turns your kernel into one.

**`extern "C"`.** Tells the C++ compiler to keep a function's name plain, so Python can find it by name in the `.so`.

**`ctypes`.** Python's built-in way to call functions in a `.so`. You pass GPU addresses from `tensor.data_ptr()`. Simple and transparent; the first way you call your kernels (Phase 5).

**`load_inline`.** `torch.utils.cpp_extension.load_inline`: PyTorch compiles CUDA code from a Python string into a module whose functions take and return tensors. The convenient way, from Phase 8. Needs `ninja`.

**Host program.** The program that drives the GPU. Here, your Python scripts drive your compiled kernels.

**Triton.** A Python-based language for writing GPU kernels, with less manual work than CUDA. Tried in Phase 15 as a comparison.

**CUDA graph.** A recording of a whole sequence of kernel launches that can be replayed with one call, which removes per-launch overhead. `torch.cuda.CUDAGraph` in PyTorch.

## How it all fits together

```
Image → patches (Conv3D) → 32× [RMSNorm → windowed MHA + 2D RoPE → RMSNorm → SwiGLU MLP]
      → patch merger (2×2 → MLP) → 247 image tokens
      + text tokens (embedding table) → 273 tokens
      → 28× [RMSNorm → GQA attention + M-RoPE → RMSNorm → SwiGLU MLP]   (prefill once)
      → final RMSNorm → LM head → argmax → next token
      → repeat with KV cache, one token at a time                       (decode)
```
