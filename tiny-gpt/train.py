"""Train the tiny GPT on Tiny Shakespeare (char level). Done when val loss < 1.6.

Run: python train.py             (about 2 min on the RTX 5070)
     python train.py --iters 200 (smoke test)

Pipeline: text -> integer ids -> random windows -> model -> cross-entropy -> AdamW step.
Every --eval-every steps it prints the train and val loss; at the end it saves a checkpoint
and writes a 500-character sample to sample.txt.
"""
import argparse
import time
from pathlib import Path

import torch

from model import GPT, GPTConfig

p = argparse.ArgumentParser()
# 3000 is where val loss bottoms out (~1.46); after that the model starts memorising (overfitting)
p.add_argument("--iters", type=int, default=3000)
p.add_argument("--eval-every", type=int, default=500)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--compile", action="store_true", help="torch.compile the model (slow start, faster steps)")
args = p.parse_args()

torch.manual_seed(1337)  # reproducible runs
device = "cuda" if torch.cuda.is_available() else "cpu"
here = Path(__file__).parent

# ---- data ---------------------------------------------------------------------------
# Character-level "tokenizer": every distinct character gets an id. That gives a vocab of 65,
# versus ~50k for GPT-2's BPE. Simple, but each token carries little meaning, so the model
# has to learn spelling as well as language.
text = (here / "data/input.txt").read_text()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}  # string -> id (encode)
itos = {i: c for c, i in stoi.items()}      # id -> string (decode)
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

# first 90% to train on, last 10% held out. Val loss on unseen text is what measures learning;
# train loss alone can keep falling while the model simply memorises.
n = int(0.9 * len(data))
splits = {"train": data[:n], "val": data[n:]}

cfg = GPTConfig(vocab_size=len(chars))


def get_batch(split):
    """Sample `batch` random windows. x is a window, y is the same window shifted by one.

    For x = "To be or", y = "o be or " -> at each position the target is the next char,
    so one window gives T training examples at once (that is why causality matters).
    """
    d = splits[split]
    ix = torch.randint(len(d) - cfg.block_size, (args.batch,))
    x = torch.stack([d[i:i + cfg.block_size] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + cfg.block_size] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, n_batches=100):
    """Average the loss over 100 batches per split; a single batch is too noisy to compare."""
    model.eval()  # switches dropout off
    out = {}
    for split in splits:
        losses = torch.zeros(n_batches)
        for k in range(n_batches):
            with torch.autocast(device, dtype=torch.bfloat16):
                _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()  # dropout back on
    return out


# ---- model + optimiser ------------------------------------------------------------------
model = GPT(cfg).to(device)
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M on {device}")
# AdamW = Adam with decoupled weight decay (a mild pull of weights towards zero; regularises)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
# OneCycle: warm up the LR over the first 5% of steps, then anneal it down. Warm-up avoids early
# blow-ups while Adam's statistics are still empty; the decay lets it settle into a minimum
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05)
fwd = torch.compile(model) if args.compile else model

# ---- training loop ----------------------------------------------------------------------
t0 = time.time()
for it in range(args.iters + 1):
    if it % args.eval_every == 0 or it == args.iters:
        l = estimate_loss(model)
        print(f"iter {it:5d} | train {l['train']:.3f} | val {l['val']:.3f} | {time.time() - t0:.0f}s", flush=True)
    if it == args.iters:
        break
    x, y = get_batch("train")
    # bf16 autocast: matmuls run in bfloat16 on the tensor cores (~2x faster, half the memory)
    # while weights stay FP32. bf16 has FP32's range, so no loss scaling is needed (unlike fp16)
    with torch.autocast(device, dtype=torch.bfloat16):
        _, loss = fwd(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    # cap the gradient norm at 1.0 so one bad batch can't throw the weights far off
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()

# ---- save + sample ----------------------------------------------------------------------
# checkpoint keeps the vocab too; Monday's KV-cache work loads it instead of retraining
(here / "checkpoints").mkdir(exist_ok=True)
torch.save({"model": model.state_dict(), "cfg": cfg, "chars": chars}, here / "checkpoints/tiny_gpt.pt")

# start from id 0 (the newline char) and let the model write 500 chars
ctx = torch.zeros(1, 1, dtype=torch.long, device=device)
sample = "".join(itos[i] for i in model.generate(ctx, 500)[0].tolist())
(here / "sample.txt").write_text(sample)
print("-" * 60 + "\n" + sample)
print(f"\nval loss {l['val']:.3f} → {'✅ < 1.6' if l['val'] < 1.6 else '❌ not yet < 1.6'}")
