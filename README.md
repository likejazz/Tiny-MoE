# Tiny-MoE

![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)

A lightweight Mixture-of-Experts language model implemented from scratch in native PyTorch.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Features](#features)
- [Architecture](#architecture)
  - [Model Configuration](#model-configuration)
  - [Key Components](#key-components)
- [Resources](#resources)
  - [Papers](#papers)
  - [Videos](#videos)
- [Notes](#notes)
- [License](#license)

---

## Overview

Tiny-MoE is an open-source Mixture-of-Experts (MoE) language model built from scratch in native PyTorch. This repository contains the inference implementation: model loading, sampling, and text generation.

Every module—from the router to the specialized expert layers—is implemented directly, including Multi-head Latent Attention (MLA), YaRN long-context extension, and Top-2 MoE routing.

### Model Specifications

- **Total Parameters:** ~150M–200M
- **Active Parameters:** ~70M–90M per token
- **Base Context Length:** 512 tokens
- **Extended Context Length:** 2048 tokens (via YaRN)
- **Architecture:** MoE Decoder-Only Transformer
- **Weights:** [AbdelrhmanEbied/Tiny-MoE](https://huggingface.co/AbdelrhmanEbied/Tiny-MoE)

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/likejazz/Tiny-MoE.git
cd Tiny-MoE
```

### 2. Install

```bash
uv sync
```

### 3. Generate

```bash
python inference/generate.py
```

The script downloads weights from Hugging Face on first run, then opens an interactive prompt. Type `exit` or `quit` to stop.

The default checkpoint is the base model (`model_variant="base"` in `inference/generate.py`). Switch `model_variant` to `"base-yarn"` or `"fine-tuned"` to load the YaRN or instruction-tuned weights.

---

## Project Structure

```text
Tiny-MoE/
├── inference/                       # Model loading, sampling, and text generation.
│   ├── generate.py                  # Entry point for text generation.
│   ├── inference_configs.py         # Model and generation configurations.
│   ├── load_model.py                # Downloads and loads Hugging Face checkpoints.
│   ├── modeling.py                  # Model architecture used for inference.
│   └── sampler.py                   # Sampling strategies (Top-k, Top-p, temperature, etc.).
├── .github/workflows/lint.yml       # Ruff lint and format checks.
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## Features

### Model Architecture

- Mixture-of-Experts (MoE) transformer implemented from scratch
- Multi-head Latent Attention (MLA)
- Rotary Positional Embeddings (RoPE)
- YaRN long-context extension
- Shared expert architecture
- Weight absorption for optimized inference
- Native PyTorch implementation

### Inference

- Interactive generation loop
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Repetition penalty
- N-gram blocking
- Automatic checkpoint download from Hugging Face

### MoE

- Packed expert execution
- Top-k routing
- Shared expert combined with routed experts

---

## Architecture

Tiny-MoE is a decoder-only Transformer language model. It combines Multi-head Latent Attention (MLA), Rotary Position Embeddings (RoPE), Mixture-of-Experts (MoE), and a shared expert while keeping a small parameter budget.

The model has 14 decoder layers. Each layer contains an MLA attention block followed by a Mixture-of-Experts feed-forward network with 8 routed experts, 1 shared expert, and Top-2 routing. Weight absorption can be enabled to reduce computation during inference.

### Overall Architecture

```text
                           Tiny-MoE

                         User Prompt
                               │
                               ▼
                        Token Embeddings
                               │
                               ▼
                    14 × Transformer Blocks
                               │
                               ▼
                         Final RMSNorm
                               │
                               ▼
                            LM Head
                               │
                               ▼
                      Sampled Next Token
```

### Transformer Block

```text
                     Transformer Block

                         Input
                           │
                           ▼
                       RMSNorm
                           │
                           ▼
               Multi-head Latent Attention
                  ├── RoPE Positional Encoding
                  └── SDPA Attention
                           │
                           ▼
                     Residual Add
                           │
                           ▼
                       RMSNorm
                           │
                           ▼
                    MoE Router (Top-2)
                      ├───────────────┐
                      ▼               ▼
                 Routed Experts    Shared Expert
                  (8 Experts)       (1 Expert)
                      └───────┬───────┘
                              ▼
                       Residual Add
                              │
                              ▼
                            Output
```

### Model Configuration

| Parameter | Value |
|-----------|------:|
| Vocabulary Size | 32,000 |
| Hidden Size | 512 |
| Transformer Layers | 14 |
| Attention Heads | 8 |
| Routed Experts | 8 |
| Shared Experts | 1 |
| Experts per Token | 2 |
| MoE Intermediate Size | 1024 |
| Base Context Length | 512 |
| YaRN Context Length | 2048 |
| RMSNorm Epsilon | 1e-6 |
| RoPE Theta | 10,000 |
| Attention Implementation | SDPA |
| Weight Tying | Enabled |

### Key Components

#### Multi-head Latent Attention (MLA)

- KV compression using latent representations
- SDPA attention backend
- Rotary positional embeddings (RoPE)
- Optional YaRN context extension
- Weight absorption for efficient inference

#### Mixture-of-Experts (MoE)

- 8 routed experts
- 1 shared expert
- Top-2 routing
- Packed expert matmuls

#### Positional Encoding

- Rotary Position Embeddings (RoPE)
- Optional YaRN scaling for the 2048-token checkpoints

#### Normalization

- RMSNorm before attention
- RMSNorm before MoE

---

## Resources

### Papers

**Core Papers**

| Paper Title | Year | Primary Focus / Architecture | Link |
| :--- | :---: | :--- | :---: |
| **Attention Is All You Need** | 2017 | Vanilla Transformer Framework | [PDF](https://arxiv.org/pdf/1706.03762) |
| **Root Mean Square Layer Normalization** | 2019 | Efficient Activation Normalization (RMSNorm) | [PDF](https://arxiv.org/pdf/1910.07467) |
| **RoFormer: Enhanced Transformer with Rotary Position Embedding** | 2021 | Rotary Position Embeddings (RoPE) | [PDF](https://arxiv.org/pdf/2104.09864) |
| **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model** | 2024 | DeepSeekMoE Architecture & MLA | [PDF](https://arxiv.org/pdf/2405.04434) |
| **DeepSeek-V3 Technical Report** | 2024 | Advanced MoE Scaling & Auxiliary-Loss-Free Routing | [PDF](https://arxiv.org/pdf/2412.19437) |

**Additional Reading**

| Paper Title | Year | Primary Focus / Architecture | Link |
| :--- | :---: | :--- | :---: |
| **Mixture of Experts in Large Language Models** | 2025 | Comprehensive Literature Survey on MoE Systems | [PDF](https://arxiv.org/pdf/2507.11181) |

### Videos

- [How DeepSeek Rewrote the Transformer [MLA]](https://www.youtube.com/watch?v=0VLAoVGf_74)
- [How Attention Got So Efficient [GQA/MLA/DSA]](https://www.youtube.com/watch?v=Y-o545eYjXM)
- [Mixture of Experts (MoE), Visually Explained](https://www.youtube.com/watch?v=0QQlYR1r6pQ)
- [The Most Underrated Layer Inside Every AI Model](https://www.youtube.com/watch?v=JHl_gwVoh-k)
- [How Rotary Position Embedding Supercharges Modern LLMs [RoPE]](https://www.youtube.com/watch?v=GQPOtyITy54)
- [RoPE (Rotary Positional Embeddings) Explained: The Positional Workhorse of Modern LLMs](https://www.youtube.com/watch?v=SMBkImDWOyQ)
- [What is Temperature in LLM](https://www.youtube.com/watch?v=jnikMver_CE)

---

## Notes

Tiny-MoE is a learning project. The fine-tuned checkpoint can follow a simple chat prompt, but responses are weaker than those from larger production models. The focus of this repository is the inference implementation of the architecture.

---

## License

This project is licensed under the **Apache License 2.0**.

You are free to use, modify, and distribute this software in accordance with the terms of the license. See the [LICENSE](LICENSE) file for the full license text.
