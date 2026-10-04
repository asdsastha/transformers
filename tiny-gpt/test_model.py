"""Unit tests for model.py. They use a tiny config so they run on the CPU in about a second.

Order to make them pass: attention shape -> causality -> full GPT -> size -> generate.
"""
import torch

from model import GPT, CausalSelfAttention, GPTConfig

# small enough for the CPU; dropout off so outputs are deterministic
CFG = GPTConfig(vocab_size=65, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)


def test_attention_shape():
    # attention must preserve shape: (B, T, C) in, (B, T, C) out
    x = torch.randn(2, 10, CFG.n_embd)
    assert CausalSelfAttention(CFG)(x).shape == x.shape


def test_attention_is_causal():
    # The core property: changing tokens 5..7 (the "future") must not change outputs 0..4.
    # If this fails, the mask is missing or wrong and the model can read the answer in training.
    torch.manual_seed(0)
    attn = CausalSelfAttention(CFG).eval()
    x = torch.randn(1, 8, CFG.n_embd)
    y = x.clone()
    y[:, 5:] = torch.randn(1, 3, CFG.n_embd)
    assert torch.allclose(attn(x)[:, :5], attn(y)[:, :5], atol=1e-6)


def test_gpt_logits_and_loss():
    # one score per vocab entry at every position, and a sane starting loss
    torch.manual_seed(0)
    m = GPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (3, 12))
    logits, loss = m(idx, idx)
    assert logits.shape == (3, 12, CFG.vocab_size)
    # an untrained model should be close to uniform: loss ≈ ln(65) ≈ 4.17.
    # Much higher means bad initialisation; much lower means something leaks the answer.
    assert 3.5 < loss.item() < 5.0


def test_default_size_is_about_10m_params():
    # checks the default architecture is wired as intended (~10.8M with 6 x 384)
    n = sum(p.numel() for p in GPT(GPTConfig()).parameters())
    assert 9e6 < n < 12e6, n


def test_generate():
    # 1 seed token + 20 generated tokens
    m = GPT(CFG).eval()
    out = m.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=20)
    assert out.shape == (1, 21)
