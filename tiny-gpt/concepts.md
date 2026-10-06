# Random Numbers and Seeding in Deep Learning

## Why we need randomness

Randomness is used throughout deep learning because it helps models learn better and generalize better:

- **Weight initialization**: Model weights start with random values so neurons do not all learn the same thing.
- **Data shuffling**: Training samples are mixed each epoch so the model does not overfit to a fixed order.
- **Train/validation splits**: Random splits help produce representative subsets.
- **Dropout**: Randomly disables some neurons during training to reduce overfitting.
- **Data augmentation**: Random transforms (crop, flip, noise, etc.) improve robustness.

Without randomness, training can become unstable, biased, or less effective.

## How random numbers are implemented

Most software (including PyTorch) uses **pseudo-random number generators (PRNGs)**.

A PRNG:

1. Starts from an internal state.
2. Applies deterministic mathematical rules.
3. Produces a sequence that looks random.

So the numbers are not truly random in a physical sense, but they are statistically useful for ML workflows.

## What a seed does

A **seed** sets the PRNG's starting state.

Example:

```python
torch.manual_seed(1337)
```

This tells PyTorch to start from a fixed initial state. If code, environment, and seed are the same, the sequence of generated random numbers will also be the same.

That gives **reproducibility**:

- same initial weights
- same random shuffles
- same dropout masks
- easier debugging and fair experiment comparison

## Why seeding is important in practice

When you do not set a seed, each run may produce different results because random states differ. In experiments, this makes it hard to tell whether a change in loss/accuracy comes from code changes or random chance.

Setting a seed helps isolate the real effect of your changes.

## True random vs pseudo-random

- **Pseudo-random** (common in ML): deterministic after seeding; reproducible; generated algorithmically.
- **True random**: generated from physical entropy sources (hardware noise, timing noise, etc.); non-deterministic.

In most training pipelines, we intentionally use pseudo-random generation because reproducibility is valuable.