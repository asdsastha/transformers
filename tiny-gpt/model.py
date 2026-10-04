"""Tiny GPT: a decoder-only transformer for character-level language modelling.

The model reads a window of characters and, at every position, predicts the next one.
The data flow (see README.md for the diagram):

    token ids (B, T)
      -> token embedding + position embedding         (B, T, C)
      -> n_layer x Block [ attention, then MLP ]      (B, T, C)
      -> final LayerNorm -> lm_head                   (B, T, vocab)  = logits
      -> cross-entropy against the next character     scalar loss

Shape letters used everywhere:
    B  = batch size (independent text windows processed in parallel)
    T  = time / sequence length (number of characters in the window, <= block_size)
    C  = n_embd, the width of each token's vector (the "residual stream")
    nh = n_head, hs = head size = C // nh

Your part: implement the four TODOs. Check with `pytest -q`.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    # 65 distinct characters in Tiny Shakespeare (letters, punctuation, space, newline)
    vocab_size: int = 65
    # Longest context the model can see. Also the size of the position-embedding table
    block_size: int = 256
    # Depth: how many Blocks are stacked
    n_layer: int = 6
    # Number of attention heads per Block. Each head attends independently with size C // nh
    n_head: int = 6
    # Width of every token vector. 6 layers x 384 wide gives ~10.8M parameters
    n_embd: int = 384
    # Fraction of activations randomly zeroed during training. Fights overfitting on a 1 MB dataset
    dropout: float = 0.2


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention. (B, T, C) -> (B, T, C).

    The only place in the model where tokens exchange information. Each position builds a
    query ("what am I looking for?"), a key ("what do I contain?") and a value ("what do I
    pass on if chosen?"). Scores = q · k / sqrt(hs). Softmax over the scores gives weights,
    and the output is the weighted sum of the values.

    "Causal" means position t may only look at positions <= t. Otherwise, during training,
    the model could read the character it is supposed to predict.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "C must split evenly across heads"
        # TODO 1: create
        #   - a Linear C -> 3C that produces q, k and v in one matmul (faster than three Linears)
        #   - a Linear C -> C output projection that mixes the heads back together
        #   - the dropout rate (for attention weights) and an nn.Dropout (for the output)
        #   - store n_head; forward() needs it to split heads
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 2:
        #   1. B, T, C = x.shape
        #   2. one projection -> split the last dim into q, k, v, each (B, T, C)
        #   3. split heads: (B, T, C) -> (B, T, nh, hs) -> transpose -> (B, nh, T, hs)
        #      so every head is an independent attention over T positions
        #   4. attention: F.scaled_dot_product_attention(q, k, v, is_causal=True,
        #      dropout_p=<rate> if self.training else 0.0). It does softmax(q k^T / sqrt(hs) + mask) v
        #      (you'll write that by hand in NumPy in Monday's drill)
        #   5. merge heads: (B, nh, T, hs) -> transpose -> (B, T, nh, hs) -> reshape (B, T, C)
        #   6. output projection, then dropout
        raise NotImplementedError


class Block(nn.Module):
    """One transformer layer, pre-norm style:

        x = x + attn(ln_1(x))    # tokens talk to each other (communication)
        x = x + mlp(ln_2(x))     # each token thinks on its own (computation)

    The "x + ..." residual connections mean each sublayer only adds a small correction to a
    shared vector (the residual stream). Gradients flow straight through the additions, which
    is what makes deep stacks trainable. Pre-norm (LayerNorm *before* the sublayer) is what
    GPT-2 and LLaMA use; it trains more stably than the original paper's post-norm.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # TODO 3: create
        #   - ln_1, ln_2: nn.LayerNorm(C)
        #   - attn: CausalSelfAttention(cfg)
        #   - mlp: Linear(C, 4C) -> GELU -> Linear(4C, C) -> Dropout
        #     (the 4x widening is the GPT convention; this MLP holds ~2/3 of the parameters)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the two residual lines from the docstring
        raise NotImplementedError


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        # TODO 4: create
        #   - token embedding: nn.Embedding(vocab_size, C). A lookup table: char id -> vector
        #   - position embedding: nn.Embedding(block_size, C). Attention has no sense of order
        #     by itself, so we add a learned vector per position (LLaMA uses RoPE instead)
        #   - dropout on the summed embeddings
        #   - n_layer Blocks (nn.Sequential or nn.ModuleList)
        #   - final LayerNorm(C)
        #   - lm_head: Linear(C, vocab_size). Turns each token vector into a score per character
        raise NotImplementedError

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) character ids. targets: (B, T) ids shifted one to the left, i.e. the
        "next character" at every position.

        Returns (logits (B, T, vocab), loss or None).

        Steps: tok_emb(idx) + pos_emb(arange(T)) -> dropout -> blocks -> ln_f -> lm_head.
        Loss: F.cross_entropy wants (N, vocab) and (N,), so flatten B*T together.
        Untrained, the loss should be ~ln(65) ≈ 4.17 (a uniform guess over 65 characters).
        Tip: compute the loss on logits.float(); under bf16 autocast it is more accurate.
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """Autoregressive sampling: predict one character, append it, repeat.

        Every step re-runs the whole window, which is O(T^2) work per token. Monday's KV cache
        removes that by storing past keys/values instead of recomputing them.
        """
        for _ in range(max_new_tokens):
            # crop to the last block_size chars; the position table has no rows beyond that
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            # only the last position's prediction matters for the next char.
            # temperature < 1 sharpens the distribution (safer text), > 1 flattens it (wilder text)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            # sample, rather than argmax; greedy decoding on a char model loops ("the the the")
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
