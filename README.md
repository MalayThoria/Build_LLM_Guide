# 🎭 Shakespeare LLM From Scratch

A character-level Transformer (mini-GPT) trained on Shakespeare's complete works — built from scratch in PyTorch to understand how modern LLMs actually work under the hood.

> The same architecture powering GPT-4, Claude, and LLaMA — just scaled down to something you can train on a free Colab GPU in ~10 minutes.

---

## 🚀 Run it in one click

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/Shakespeare_LLM_From_Scratch.ipynb)

> ⚠️ **Enable GPU first:** *Runtime → Change runtime type → T4 GPU*

---

## 📚 What's inside

A 38-cell notebook that walks through every concept needed to build a real Transformer:

| Step | Topic |
|------|-------|
| 1 | Setup & GPU check |
| 2 | Download TinyShakespeare dataset (~1MB) |
| 3 | **Tokenization** — characters → integers |
| 4 | Train/val split (90/10) |
| 5 | Batching — why one 256-char sequence is 256 training examples |
| 6 | **Self-attention from scratch** — Q/K/V with a worked example |
| 7 | Single attention head as a reusable module |
| 8 | **Multi-head attention** — running heads in parallel |
| 9 | Feed-forward network — per-token "thinking" |
| 10 | **Transformer block** — attention + FFN + LayerNorm + residual |
| 11 | Full GPT model — embeddings, stacked blocks, LM head |
| 12 | Loss evaluator |
| 13 | **Training loop** — 5 steps explained |
| 14 | Generate Shakespeare 🎭 |
| 15 | Custom prompts & experiments |

---

## 🧠 Key concepts explained

- **Why we need tokenization** — and why subword > character at scale
- **The Q/K/V mental model** — *"what am I looking for / what do I contain / what do I offer"*
- **Causal masking** — why future tokens must be hidden during training
- **Residual connections + LayerNorm** — what makes deep Transformers trainable
- **Positional embeddings** — why attention alone is permutation-invariant
- **Sampling strategies** — why argmax = boring, why temperature matters

---

## 🛠️ Tech stack

- **Framework:** PyTorch
- **Dataset:** TinyShakespeare (~1MB)
- **Model size:** ~10M parameters
- **Architecture:** Decoder-only Transformer (same as GPT-2)
- **Training time:** ~5–10 minutes on Colab T4 GPU
- **Final loss:** ~1.4–1.6 (down from ~4.17 random baseline)

### Model hyperparameters

```python
batch_size  = 64      # sequences in parallel
block_size  = 256     # context length
n_embd      = 384     # embedding dimension
n_head      = 6       # attention heads
n_layer     = 6       # Transformer blocks
dropout     = 0.2     # regularization
learning_rate = 3e-4  # AdamW optimizer
max_iters   = 5000    # training steps
```

---

## 🎭 Sample output

After training, prompting with `"ROMEO:"`:

```
ROMEO:
The court of heaven, I will not be so soon.
What says my lord of Buckingham? he comes hither.

DUKE VINCENTIO:
I pray you, sir, the king is dead, and we shall
find the time to speak with him in the wars.

JULIET:
O, my good lord, I am content to bear
The burden of my sorrows on my heart,
And in my soul I do beseech thee, gentle
```

### What works ✅
- Character names in proper format (ROMEO, JULIET, DUKE VINCENTIO)
- Dialogue structure (NAME:\nspeech)
- Shakespearean vocabulary (thou, hath, wherefore, beseech)
- Blank verse rhythm
- Real English words

### What doesn't ❌
- Coherent plot
- Logical conversations
- Long-range consistency
- Real grammar at scale

> **This is the perfect teaching moment.** The architecture is identical to GPT-4 — the difference is *scale*. 10M params → 1.8T params, 1MB data → 15TB data. Same algorithm.

---

## 🧪 Experiments to try

1. **Tweak hyperparameters** — bigger `n_layer`, `n_embd`, more iterations
2. **Different prompts** — `"JULIET:"`, `"To be, or not to be"`, `"ACT I.\nSCENE I."`
3. **Temperature sampling** — divide logits before softmax to control creativity
4. **Top-k sampling** — only sample from top K most likely tokens
5. **Train on different text** — swap in any plain text file
6. **Save and reload weights** — see how to persist trained models

---

## 📊 How this compares to production LLMs

| Feature | This notebook | GPT-4 |
|---|---|---|
| Parameters | 10M | ~1.8T |
| Training tokens | ~1M | ~13T |
| Training cost | Free (Colab) | ~$100M+ |
| Training time | ~10 minutes | months |
| Tokenization | Character | Subword (BPE/tiktoken) |
| Fine-tuning (RLHF) | No | Yes |
| Inference optimizations | None | KV-cache, flash attention, quantization |

The architecture is **the same**. The breakthrough wasn't a new algorithm — it was scaling this exact pattern.

---

## 📖 Going deeper

- 📄 [Attention is All You Need](https://arxiv.org/abs/1706.03762) — the original paper (2017)
- 🎥 [Andrej Karpathy — "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the inspiration for this notebook
- 💻 [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy's production-quality reference
- 📚 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — visual explanation

---

## 👨‍💻 Built by

**Malay Thoria** — Applied AI Engineer @ QualityKiosk Technologies

- 🌐 NLP, Speech AI, Computer Vision background
- 🏆 IEEE Best Paper, 2025
- 🚀 Founder of [Encipherio](https://encipherio.com)
- 💼 [LinkedIn](https://linkedin.com/in/malaythoria)

---

## ⭐ If this helped you learn

Star the repo to bookmark it, and feel free to fork and experiment!

---

## 📄 License

MIT — use freely for learning, teaching, or as a base for your own projects.
