# tiny-gpt roadmap

How to get from where you are now to a trained GPT at **val loss < 1.6**, merged into `main`.

Use this file next to `tiny_gpt.ipynb`. The notebook has the checks, and this file has the explanations and hints for each step.

- **Hints are folded.** Open the first one, try again, and only then open the next. Each hint gives away more than the one before it. None of them is the full solution.
- **One part is about one pomodoro.** Tag them `ML fundamentals`.
- **Type the code yourself.** If a tool writes it, you won't be able to explain it in an interview.

| Part | What | Time | Status |
|---|---|---|---|
| 0 | Setup | 5 min | ✅ |
| 1 | Tokenizer: `vocab`, `stoi`, `itos`, `encode`, `decode` | 15 min | 🟡 half done |
| 2 | Data tensor + train/val split | 10 min | ⬜ |
| 3 | The prediction task + `get_batch` | 20 min | ⬜ |
| 4 | Bigram model + training loop | 45 min | ⬜ |
| 5 | The averaging trick (3 ways) | 25 min | ⬜ |
| 6 | One attention head + your own causality test | 40 min | ⬜ |
| 7 | Multi-head attention (`model.py` TODO 1 + 2) | 45 min | ⬜ |
| 8 | MLP | 15 min | ⬜ |
| 9 | Block: residuals + LayerNorm (TODO 3) | 25 min | ⬜ |
| 10 | The full GPT (TODO 4) | 40 min | ⬜ |
| 11 | Real training on the GPU | 45 min | ⬜ |
| 12 | Sampling + checkpoint | 15 min | ⬜ |
| 13 | Notebook → `model.py`, `pytest`, `train.py`, README, merge | 30 min | ⬜ |

That's about 6 h in total, or 7 pomodoros of 50 minutes.

---

## Before you start: where things stand

1. **`model.py` is back to the scaffold.** All 4 TODOs raise `NotImplementedError`, so `pytest -q` fails until Part 13. That's expected.
2. **Part 1 cell in the notebook.** You built `vocab`, `stoi` and `itos` (the checks now use `vocab` too), but `encode` / `decode` don't exist yet. You also have a print statement and the `# ✍️ YOUR CODE` comment on the same line, which is harmless but messy.
3. **`tiny_gpt.py`** (your script version) has three bugs to fix at some point. Don't let them block the notebook:
   - `def get_b atch` has a space in the name, which is a syntax error.
   - It uses `self.cfg.batch_size`, but `GPTConfig` has no `batch_size` field. Pass the batch size as an argument instead.
   - `torch.cuda.get_device_name(0)` crashes on a machine without a GPU. Only call it when `device == "cuda"`, the way notebook Part 0 does.

The shape letters used everywhere:

| Letter | Meaning | Final model |
|---|---|---|
| B | batch size: windows processed in parallel | 64 |
| T | time: characters in a window, ≤ `block_size` | 256 |
| C | `n_embd`: width of each token's vector | 384 |
| nh | `n_head`: heads per attention layer | 6 |
| hs | head size, C // nh | 64 |
| V | vocab size | 65 |

**Habit for every part:** when something breaks, print `.shape` first. Most bugs here are shape bugs.

---

## Part 1 · Tokenizer

**Why.** A network only does arithmetic, so text has to become integers. Here every distinct character is one token (65 of them). GPT-2's BPE uses about 50k sub-word tokens instead. That gives fewer tokens per sentence, but it needs more machinery.

**Steps**
1. `vocab` = the sorted list of distinct characters, and `vocab_size = len(vocab)`. You already have both.
2. `stoi` (char → id) and `itos` (id → char). You already have these.
3. `encode(s)` → list of ints. `decode(ids)` → string.

**Why sorted?** A `set` has no stable order. Sorting makes id 0 always `'\n'` and id 1 always `' '`, so a checkpoint saved today still decodes correctly tomorrow.

<details><summary>Hint 1</summary>

`encode` maps every character of the string through `stoi`. `decode` maps every id through `itos`, then joins the characters back together.
</details>

<details><summary>Hint 2</summary>

A list comprehension does `encode`. `"".join(...)` does `decode`. A `lambda` is fine for both.
</details>

---

## Part 2 · Dataset tensor + split

**Why hold data out?** A 10M-parameter model can memorise 1 MB of text. Then the train loss keeps falling while the model learns nothing general. The validation loss on unseen text is the honest score.

**Why split by position, not randomly?** Random windows overlap with their neighbours. A random split would put almost the same text in both sets, and the val loss would lie.

**Steps:** `data` = `encode(text)` as a 1-D `torch.long` tensor → `n = int(0.9 * len(data))` → `train_data = data[:n]`, `val_data = data[n:]`.

<details><summary>Hint</summary>

`torch.tensor(list_of_ints, dtype=torch.long)`. `long` because `nn.Embedding` and `cross_entropy` want integer ids, not floats.
</details>

---

## Part 3 · The prediction task + `get_batch`

**Why.** This is the whole job of a language model. One window of `block_size + 1` characters holds `block_size` training examples: context of length 1 → next char, length 2 → next char, and so on. So:

```
x = data[i     : i+block_size]
y = data[i + 1 : i+block_size+1]      # x shifted left by one
y[t] is the character that comes after x[0..t]
```

**Steps**
1. **Look first** (no check): take `train_data[:9]`, loop `t` over 0…7, and print `decode(context) → decode(target)`. Seeing the 8 pairs is the point of this step.
2. `get_batch(split, batch_size, block_size)`:
   - choose `train_data` or `val_data`;
   - draw `batch_size` random start indices;
   - stack the x windows and the y windows into (B, T) tensors;
   - `.to(device)` both.

**Think before coding:** what is the largest valid start index `i`? `y` needs `data[i + block_size]` to exist.

<details><summary>Hint 1: the 8 pairs</summary>

For each `t`, the context is `x[:t+1]` and the target is `x[t+1]` (when `x = train_data[:9]`). Remember `decode` wants a list, so use `.tolist()` on a tensor slice, or wrap a single id in a list.
</details>

<details><summary>Hint 2: random starts</summary>

`torch.randint(high, (batch_size,))` gives integers in `[0, high)`, so `high` itself is never drawn. The largest valid start is `len(data) - block_size - 1`. Work out which `high` gives you exactly that.
</details>

<details><summary>Hint 3: stacking</summary>

`torch.stack([... for i in ix])` turns B tensors of shape (T,) into one (B, T). Do it once for x and once for y.
</details>

**Common bug:** forgetting `.to(device)`, so everything later runs on the CPU, or crashes with "tensors on different devices".

---

## Part 4 · Bigram model + training loop

**Why.** It's the simplest possible language model: predict the next character from the current one only. The training loop and `estimate_loss` you write here are reused all the way to Part 11.

**Key ideas**
- `nn.Embedding(V, V)` is a 65 × 65 table. Row `i` holds the scores (logits) for "what follows character `i`".
- **Cross-entropy** = −log p(correct char). If the model guesses uniformly, the loss is ln 65 ≈ 4.17, so an untrained model starts around 4.2–4.9.
- `F.cross_entropy` wants logits of shape **(N, V)** and targets of shape **(N,)**, so flatten B·T into N.

**Steps (4a):** `BigramLM` with `__init__`, `forward(idx, targets=None) -> (logits, loss)` and `generate(idx, max_new_tokens)`.

<details><summary>Hint 1: forward</summary>

`logits = self.table(idx)` is already (B, T, V). If `targets` is `None`, return `(logits, None)`. Otherwise, read `B, T, V = logits.shape` and compute the loss on `logits.view(B*T, V)` and `targets.view(B*T)`.
</details>

<details><summary>Hint 2: generate</summary>

Loop `max_new_tokens` times: forward → `logits[:, -1, :]` → softmax → `torch.multinomial(probs, num_samples=1)` gives (B, 1) → `torch.cat((idx, next_id), dim=1)`. `model.py`'s `generate` has the same structure, so you can read it.
</details>

**Steps (4b):** `estimate_loss`, then a 3000-step loop (`batch_size=32`, `block_size=8`, `lr=1e-2`).

The training step, in this exact order:

```
get_batch → forward → zero_grad → backward → step
```

**Why `zero_grad`?** PyTorch adds each new gradient to the old one. If you skip it, every step uses the sum of all past gradients.

<details><summary>Hint 3: estimate_loss</summary>

Decorate it with `@torch.no_grad()`. Call `model.eval()`, then for each split average `eval_iters` batch losses (use `.item()`). Call `model.train()` before returning `{"train": ..., "val": ...}`.
</details>

**Expected:** the loss goes from about 4.7 to about **2.5** and plateaus. The sample looks word-shaped but is nonsense. One character of context is the ceiling, and the next parts remove it.

---

## Part 5 · The averaging trick

**Why.** Position `t` should use positions 0…t and never t+1 onward, because those hold the answer. Averaging the past is the crudest way to do that. Version 3 below is exactly the shape attention takes; attention just replaces the uniform weights with learned ones.

**Steps:** compute `xbow` (bag of words, the average of x[0..t]) three ways. All three must be (B, T, C) and equal.
1. **Loops** over b and t: `x[b, :t+1].mean(0)`.
2. **Matmul:** a (T, T) lower-triangular matrix whose rows sum to 1, then `wei @ x`.
3. **Masked softmax:** start from zeros, set the upper triangle to `-inf`, then softmax along each row.

<details><summary>Hint 1: version 2</summary>

`torch.tril(torch.ones(T, T))`, then divide each row by its sum (`keepdim=True`). `(T, T) @ (B, T, C)` broadcasts over B.
</details>

<details><summary>Hint 2: version 3</summary>

`wei.masked_fill(tril == 0, float('-inf'))`, then `F.softmax(wei, dim=-1)`. Why does this give the same numbers? exp(−inf) = 0, so masked positions get weight 0, and the remaining zeros become exp(0) = 1 each, normalised to 1/(t+1).
</details>

**Takeaway to say out loud:** "attention is a weighted average of the past, where the weights come from softmax of scores and the mask sets the future to −inf."

---

## Part 6 · One attention head

**Why.** Different tokens need different parts of the past. A vowel might care about the previous consonant; a newline might care about the speaker's name. So the weights must depend on the data.

**Key ideas**
- Each token makes a **query** (what I'm looking for), a **key** (what I contain) and a **value** (what I hand over), each through `nn.Linear(C, hs, bias=False)`.
- scores = q @ kᵀ → (B, T, T). Row t says how much token t wants each other token.
- **Scale by 1/√hs.** q·k is a sum of hs products, so its variance grows with hs. Large scores make softmax nearly one-hot, and then the gradients vanish. The demo cell shows this.
- Then: mask → softmax → `@ v` → (B, T, hs).

**Steps:** `Head(n_embd, head_size, block_size)`, then your own `test_head_is_causal()`.

<details><summary>Hint 1: the mask buffer</summary>

`self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))`. A buffer moves with `.to(device)` and is saved in the checkpoint, but it isn't a trainable parameter.
</details>

<details><summary>Hint 2: T < block_size</summary>

In `forward`, use `self.tril[:T, :T]`. During generation, T starts at 1.
</details>

<details><summary>Hint 3: the test</summary>

Copy the structure of `test_attention_is_causal` in `test_model.py`: same x, changed future, compare `[:, :5]` with `torch.allclose`. Call `.eval()` first if anything random is involved.
</details>

**Why test properties?** You can't know the exact numbers a random model should output. You *can* assert invariants: the shape, causality, and a starting loss near ln 65.

---

## Part 7 · Multi-head attention (`model.py` TODO 1 + 2)

**Why.** Several smaller heads can learn different relations in parallel. With nh heads of size hs = C/nh, concatenating them gives back width C, for the same cost as one big head.

**7a, `manual_attention(q, k, v)`** on (B, nh, T, hs) tensors: the same as your `Head` math, but 4-D. It must match `F.scaled_dot_product_attention(..., is_causal=True)`.

<details><summary>Hint 7a</summary>

`k.transpose(-2, -1)` swaps the last two axes, whatever the number of leading dims. Build the mask with `torch.tril(torch.ones(T, T, device=q.device))`. A (T, T) mask broadcasts over (B, nh).
</details>

**7b, `CausalSelfAttention(cfg)`**
- `__init__`: `c_attn = Linear(C, 3C)`, `c_proj = Linear(C, C)`, `resid_dropout`, and store `n_head`, `n_embd` and `dropout`.
- `forward`, the shape journey:

```
x (B, T, C)
 └ c_attn → (B, T, 3C) → split(C, dim=2) → q, k, v each (B, T, C)
 └ view(B, T, nh, hs) → transpose(1, 2) → (B, nh, T, hs)
 └ scaled_dot_product_attention → (B, nh, T, hs)
 └ transpose(1, 2) → (B, T, nh, hs) → contiguous().view(B, T, C)
 └ c_proj → resid_dropout → (B, T, C)
```

**Why one `Linear(C, 3C)`?** One big matmul is faster on a GPU than 3 × nh small ones. The maths is identical.

<details><summary>Hint 7b-1: the dropout argument</summary>

`dropout_p=self.dropout if self.training else 0.0`. Without the `if`, dropout would also run at eval time and your val loss would be noisy.
</details>

<details><summary>Hint 7b-2: "view size is not compatible"</summary>

After `transpose`, the tensor isn't contiguous in memory, and `view` refuses to reshape it. Use `.contiguous().view(...)` or `.reshape(...)`.
</details>

**Parameter check:** c_attn has 3C² + 3C, c_proj has C² + C, so the total is 4C² + 4C.

---

## Part 8 · MLP

**Why.** Attention is **communication** between tokens. The MLP is **computation**: each token processes on its own what it gathered. The 4× expansion holds about 2/3 of all the parameters. **GELU** is a smooth ReLU, so gradients don't die at exactly 0.

**Steps:** `Linear(C, 4C) → GELU → Linear(4C, C) → Dropout`. `nn.Sequential` is the easiest way.

**Parameter check:** (4C² + 4C) + (4C² + C) = 8C² + 5C.

The check cell changes position 3 and asserts that **only** output 3 changes. A `Linear` on (B, T, C) acts on the last dim only, so this holds automatically.

---

## Part 9 · Block (TODO 3)

```
x = x + attn(ln_1(x))   # communicate
x = x + mlp(ln_2(x))    # compute
```

**Why residuals?** Each sublayer only *adds a correction* to the residual stream. The derivative of x + f(x) contains a plain 1, so gradients reach the early layers undiminished. That's what makes stacking 6 (or 96) layers trainable.

**Why LayerNorm, and why pre-norm?** It normalises each token's vector (mean 0, variance 1, then a learned scale and shift), which keeps activations in a sane range. Putting it *before* the sublayer (pre-norm, as in GPT-2 and LLaMA) leaves the residual path a clean identity, which trains more stably than the original paper's post-norm.

**Steps:** `ln_1`, `attn`, `ln_2`, `mlp` in `__init__`, and the two lines above in `forward`.

<details><summary>Hint: the identity check fails</summary>

The check zeroes every Linear and expects the block to return x unchanged. That only works if you wrote `x = x + ...`. If you wrote `x = attn(ln_1(x))`, or used `+` on the wrong thing, it fails.
</details>

**Parameter check:** attn 4C² + 4C, MLP 8C² + 5C, two LayerNorms 2 × 2C, so the total is 12C² + 13C.

---

## Part 10 · The full GPT (TODO 4)

```
idx (B, T) → tok_emb (B, T, C) + pos_emb (T, C) → dropout → blocks → ln_f → lm_head → logits (B, T, V)
```

**Why a position embedding?** Attention is a weighted *sum*, and sums don't care about order. Shuffle the input and you get the same outputs, shuffled. Adding a learned vector per position gives the model the order. LLaMA uses RoPE instead, which rotates q and k by position.

**Steps**
- `__init__`: `tok_emb = Embedding(V, C)`, `pos_emb = Embedding(block_size, C)`, `drop`, `blocks = nn.Sequential(*[Block(cfg) for _ in range(n_layer)])`, `ln_f`, `lm_head = Linear(C, V)`.
- `forward(idx, targets=None)`: the line above, then the loss the same way as the bigram.

<details><summary>Hint 1: positions</summary>

`torch.arange(T, device=idx.device)` → `pos_emb` → (T, C). It broadcasts when added to (B, T, C). Forgetting `device=` is the classic CPU/GPU mismatch.
</details>

<details><summary>Hint 2: the loss under bf16</summary>

Compute it on `logits.float()` (the docstring tip). Cross-entropy sums logs, which is where bf16's limited precision hurts.
</details>

**Sanity numbers:** untrained loss of about 4.2 with the tiny config, and about 10.8M parameters with the default one. If the loss is far below 4, something leaks the answer (check the mask and the x/y shift). If it's far above 5, check the initialisation and the scaling.

---

## Part 11 · Training for real

Settings: `block_size=256`, `batch_size=64`, `lr=1e-3`, 3000 steps, an eval every 250 steps (`eval_iters=50`). Start from your Part 4 loop and add:

| Addition | Where it goes | Why |
|---|---|---|
| `weight_decay=0.1` in AdamW | optimiser | mild pull towards 0, which regularises |
| `OneCycleLR(..., pct_start=0.05)` | `sched.step()` after `opt.step()` | warm up while Adam's statistics are empty, then anneal down |
| `clip_grad_norm_(..., 1.0)` | between `backward()` and `step()` | one bad batch can't throw the weights far off |
| `torch.autocast(device, dtype=torch.bfloat16)` | around the forward pass only | about 2× faster on tensor cores, and bf16 needs no loss scaling |

**Record** `history` at step 0 (before any training), every 250 steps, and at step 3000.

**Read the curves:** train keeps falling. When **val turns back up**, the model has started memorising. Expect the best val loss around 1.46.

<details><summary>Hint: the loss is NaN or stuck at about 4</summary>

NaN: check that clipping is between `backward` and `step`, and that the scheduler is stepped once per training step, not once per eval. Stuck: check that `zero_grad` → `backward` → `step` all run every step, and that `model.train()` was restored after the eval.
</details>

---

## Part 12 · Sampling + checkpoint

**Temperature** divides the logits before softmax. Below 1 the distribution gets sharper (safer, more repetitive text). Above 1 it gets flatter (wilder, more misspellings). Generate 500 characters at 0.8, 1.0 and 1.3 and compare them.

**Why sample instead of argmax?** Greedy decoding on a character model falls into loops ("the the the").

Save `sample.txt` (from temperature 1.0) and `checkpoints/tiny_gpt.pt` with `model`, `cfg` and `vocab`. Create the folder first with `Path("checkpoints").mkdir(exist_ok=True)`. Monday's KV-cache work loads this file.

---

## Part 13 · Notebook → repo

1. Copy your `CausalSelfAttention`, `Block` (with the MLP) and `GPT.__init__` / `forward` into `model.py`. Keep its `GPTConfig` and `generate`.
2. `pytest -q` → **5 passed**.
3. `python train.py` → val < 1.6 plus a sample (about 2 min).
4. Fill in the README **Results**: final val loss, time, and a few lines of `sample.txt`.
5. Commit on `tiny-gpt` (also `concepts.md`, `tiny_gpt.ipynb`, `requirements.txt`, `setup.sh`, and this file), then merge into `main`.

<details><summary>Hint: the tests pass in the notebook but not in pytest</summary>

`test_model.py` imports from `model.py`, not the notebook. Check the attribute names (`self.dropout`, `self.n_head`) and that `forward` returns a tuple `(logits, loss)`.
</details>

---

## Say these out loud when you're done

Explain each one without notes. If you can't, that part needs another look.

1. Why divide by √d? What goes wrong without it?
2. What exactly does the causal mask block? Why is it needed in training but trivial when generating?
3. Why pre-norm instead of post-norm? Why do residual connections make depth trainable?
4. Why is the MLP 4× wide, and where are most of the parameters?
5. Why does val loss rise again after about 3000 steps? Name 3 ways to fight it.
6. `generate` re-runs the whole window for every new token. What does that cost, and how does a KV cache fix it?
