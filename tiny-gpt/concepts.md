# PyTorch Concepts

## Gradients and the grad family

### Why gradient tracking matters

PyTorch has a feature called autograd, which automatically tracks operations performed on tensors so it can compute gradients for backpropagation.

This is essential during training, because when you compute a loss like:

```python
loss = criterion(model(x), y)
loss.backward()
```

PyTorch needs to know all the operations used to produce `loss` so it can calculate derivatives and update model parameters.

Without gradient tracking, training would not work.

### The grad family in PyTorch

The main tools related to gradients are:

- `torch.no_grad()`
- `torch.enable_grad()`
- `torch.set_grad_enabled(mode)`
- `torch.inference_mode()`
- `torch.is_grad_enabled()`

### `@torch.no_grad()`

This decorator disables gradient tracking for the function it wraps.

```python
@torch.no_grad()
def estimate_loss(self, n_batches=100):
    ...
```

This tells PyTorch:

- do not build a computation graph
- do not store gradient information
- do not allow `backward()` to work on the outputs from this function

This is needed for validation, testing, inference, and metric calculation, where we only want the forward pass and not backpropagation.

Example:

```python
@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    total = 0.0

    for x, y in data_loader:
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        total += loss.item()

    return total / len(data_loader)
```

This is faster and uses less memory because PyTorch does not track gradients for every tensor operation.

### `@torch.enable_grad()`

This re-enables gradient tracking inside a function, even if the surrounding code is in `no_grad()` mode.

```python
@torch.no_grad()
def outer():
    x = torch.tensor(2.0, requires_grad=True)

    @torch.enable_grad()
    def inner():
        return x * x

    return inner()
```

This is useful when a larger block disables gradients temporarily, but a small nested function needs them.

### `torch.set_grad_enabled(mode)`

This is a flexible way to toggle gradient mode globally or within a block.

```python
with torch.set_grad_enabled(False):
    y = model(x)
```

This is equivalent in spirit to `no_grad()`, but it can be used more dynamically.

### `torch.inference_mode()`

This is a faster, stricter version of `no_grad()` for inference.

```python
@torch.inference_mode()
def predict(model, x):
    return model(x)
```

It disables autograd and also avoids some of the extra bookkeeping that `no_grad()` still performs. It is commonly used when a model is only being used for forward inference, not training.

### Why `no_grad()` and related decorators are needed

Training and evaluation have different requirements:

- During training: gradients are required so the optimizer can update weights.
- During validation/inference: gradients are usually not needed.

If gradients are tracked unnecessarily, then:

- memory usage increases
- the computation graph grows unnecessarily
- code becomes slower
- accidental backward passes may happen

That is why `no_grad()` is used in functions like `estimate_loss()`: they are performing forward passes, not learning steps.

In short:

- training: gradients ON
- validation/inference: gradients OFF

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

# Decorators in Python and PyTorch

## What is a decorator?

A decorator is a Python pattern that lets you wrap or modify a function without changing the function’s internal code. In simple terms, it adds extra behavior around the original function.

The syntax looks like this:

```python
@decorator_name
def my_function():
    pass
```

This is equivalent to:

```python
my_function = decorator_name(my_function)
```

The decorator receives the original function, wraps it, and returns a new function that adds behavior before or after the original call.

## Why decorators are useful

Decorators are useful because they let us reuse common behavior across many functions. Common examples include:

- logging
- timing code execution
- validation checks
- access control
- caching
- disabling gradient tracking during evaluation

## A simple Python decorator example

```python
def debug_calls(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Finished {func.__name__}")
        return result
    return wrapper

@debug_calls
def add(a, b):
    return a + b

print(add(2, 3))
```

This example prints messages before and after the function call without modifying the function itself.

## Decorators in PyTorch

PyTorch uses decorators to change runtime behavior for functions that work with tensors and autograd.

### `@torch.no_grad()`

This is one of the most important decorators in PyTorch.

```python
@torch.no_grad()
def estimate_loss(self, n_batches=100):
    ...
```

It tells PyTorch:

- do not track gradients for the operations inside this function
- do not build computational graphs
- do not store gradient information for backpropagation

This is useful when you are only evaluating a model, running inference, or estimating loss for reporting. In these cases, you do not need gradients, so disabling them saves memory and computation.

Example:

```python
@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0.0

    for x, y in data_loader:
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        total_loss += loss.item()

    return total_loss / len(data_loader)
```

Because gradients are disabled, this function is faster and more memory-efficient than a training step.

## Decorator vs context manager

PyTorch also allows the same idea with a context manager:

```python
with torch.no_grad():
    outputs = model(inputs)
```

Both forms mean the same thing: gradients are disabled for that block of code.

## Summary

A decorator is a reusable wrapper that adds behavior to a function. In PyTorch, decorators such as `@torch.no_grad()` are especially important because they control autograd behavior. This allows code to be efficient and correct in evaluation settings without accidentally computing or storing gradients.