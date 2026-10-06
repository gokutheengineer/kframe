# kframe

A from-scratch LLM inference stack that reconstructs every stage between a `safetensors` checkpoint on disk and a generated token—without relying on high-level generation APIs or third-party attention kernels.

Built as Module 1 of a six-module systems path toward production inference engines (vLLM / SGLang class): model internals → GPU kernels → serving → determinism → distributed inference → RL post-training infrastructure.

## Motivation

Most “build an LLM from scratch” projects stop at training a small GPT. kframe takes the opposite approach: start from real pretrained weights and rebuild the **inference path**—weight loading, forward pass, KV cache, batched sampling, and analytical performance models that predict memory and latency before profiling.

Primary targets:

| Model | Why it matters |
| --- | --- |
| **Llama-3.2-1B** | RoPE scaling (`llama3` variant), GQA (32 Q / 8 KV heads), tied embeddings |
| **Qwen3-0.6B** | Config-driven second architecture—QK-norm, independent `head_dim`, no Llama hardcoding |

## Architecture

```
kframe/
  loader.py       # safetensors parser (mmap + manual header parsing)
  model.py        # forward pass: embedding → decoder layers → norm → logits
  cache.py        # static KV cache; prefill / decode split
  generate.py     # greedy / temperature / top-k / top-p; batched left-padding
  bench.py        # GPU timing (CUDA events, warmup, median-of-N)
  napkin.py       # analytical memory / latency estimator from model + hardware config
  make_golden.py  # Hugging Face reference outputs (logits, hidden states, generations)
  golden/         # golden files (gitignored; regenerate locally)
  tests/          # layer-wise parity tests against HF reference
```

## Verification

Correctness is checked against Hugging Face as ground truth—not only at final logits, but **layer by layer**:

1. Generate reference outputs with the official HF model (`attn_implementation="eager"`, FP32) on fixed prompts: intermediate hidden states, final logits, and a greedy 32-token generation.
2. Compare each kframe module against those goldens (FP32 absolute tolerance `1e-4`).
3. On failure, localize the mismatch to a specific layer and operation rather than a wrong final token.

## Status

**Module 1 — in progress.** Environment and project scaffold are in place; model downloads and the golden-file harness are next. Implementation files are stubs until parity work lands.

## Setup

Requires Python 3.12+, [uv](https://github.com/astral-sh/uv), and Hugging Face Hub access (Llama-3.2-1B needs a gated license acceptance).

```bash
cd kframe
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch transformers safetensors tokenizers numpy pytest matplotlib huggingface_hub
hf auth login
hf download meta-llama/Llama-3.2-1B
hf download Qwen/Qwen3-0.6B
```

> `safetensors` is installed for Hugging Face reference / golden generation. The kframe loader parses the format directly and does not use the library for inference.

## Hardware

| Workload | Platform |
| --- | --- |
| Steps 1–3, 5 (CPU / MPS-feasible) | Apple M1 Pro, 32GB (MPS) |
| Step 4 (GPU profiling & measurement) | Rented NVIDIA GPU |

## Roadmap

1. Weight loader + config-driven model graph  
2. Forward pass parity (layer-wise vs HF)  
3. KV cache + batched generation  
4. Benchmarks + analytical performance model  
5. Second-architecture validation (Qwen3)
