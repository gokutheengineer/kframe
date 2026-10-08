# AI Systems Engineering: Modules 1, 2, and 3 Plan & Estimates

This document details the architecture, deliverables, hour estimates, and day-by-day roadmap for **Modules 1, 2, and 3** of the AI Systems Software Engineer path, calibrated for intensive pair programming at **9 hours/day**.

---

## Executive Summary & Timeline

| Module | Core Deliverable | Total Effort | Timeline @ 9 hrs/day |
| :--- | :--- | :--- | :--- |
| **Module 1: The Model as a Program** | `kframe` + `napkin.py` | **30 – 36 hours** | **~3.5 to 4 days** |
| **Module 2: GPU, Profiling & Kernels** | `kernel-lab` | **45 – 50 hours** | **~5 to 5.5 days** |
| **Module 3: The Inference Engine** | `kache` + Upstream PR | **54 – 63 hours** | **~6 to 7 days** |
| **GRAND TOTAL (Modules 1 + 2 + 3)** | **3 Repos + PR + Blogs** | **130 – 150 hours** | **~14.5 to 16.5 days** |

**Midpoint:** ~140 hours (~15 working days / 3 calendar weeks).

---

## Module 1: The Model as a Program

### Objective: `kframe`
Reconstruct every stage between raw `.safetensors` on disk and generated tokens without third-party attention kernels or generation libraries. Match Hugging Face eager FP32 outputs layer-by-layer ($\le 10^{-4}$).

### Deliverables
* `loader.py`: Custom zero-copy `safetensors` parser using `mmap` + FP32 $\to$ BF16 bit-manipulation converter.
* `model.py`: Config-driven forward pass for Llama-3.2-1B and Qwen3-0.6B (RMSNorm, RoPE with llama3 scaling, QK-norm, GQA, SwiGLU).
* `cache.py`: Preallocated static KV cache (`[layers, 2, B, kv_heads, max_len, head_dim]`) with prefill/decode split.
* `generate.py`: Greedy, temperature, top-$k$, top-$p$ sampling with per-request seed; left-padded batching.
* `napkin.py`: Analytical performance calculator (KV memory, decode tok/s ceiling, prefill lower bound, crossover batch size).
* `bench.py`: Timing harness (CUDA/MPS events, warmup, medians) + MBU analysis.
* `tests/`: Parity test suite against golden files (`prompt0.pt` – `prompt4.pt`).

### Step-by-Step Breakdown & Hours

| Step | Topic | Tasks | Effort |
| :--- | :--- | :--- | :--- |
| **Step 0** | **Setup** | Python env, model checkpoints, golden test harness (`make_golden.py`) | *Completed* |
| **Step 1** | **Tensors & Loader** | `mmap` binary header decode, zero-copy `torch.frombuffer`, tied embeddings, FP32 $\to$ BF16 bit hack | **3 – 4 h** |
| **Step 2** | **Forward Pass & Parity** | Config-driven graph, RoPE `rotate_half` & scaling, GQA, SwiGLU, layer-by-layer test oracle ($\le 10^{-4}$) | **7 – 8 h** |
| **Step 3** | **KV Cache & Generation** | Static in-place KV cache, prefill/decode split, samplers, left-padded batching | **5 – 6 h** |
| **Step 4** | **Inference Arithmetic** | `napkin.py` model, `bench.py` sweeps, Roofline curve, MBU calculation | **6 – 7 h** |
| **Step 5** | **Qwen3 & MoE** | QK-norm, independent `head_dim`, sparse MoE dispatch (naive vs. sorted slice-and-scatter) | **4 – 5 h** |
| **Step 6** | **Training Memory** | AdamW mixed-precision 16 B/param arithmetic, PyTorch memory snapshot reconciliation | **3 – 4 h** |
| **Polish** | **Exit Checks & Repo** | Exit checks without notes, documentation, plots in README | **2 h** |
| **Total** | | | **30 – 36 h** |

---

## Module 2: GPU, Profiling & Kernels

### Objective: `kernel-lab`
Master the GPU memory hierarchy, Nsight profiling tools, and kernel development. Write CUDA SGEMMs, Triton fused operators, and split-KV decode attention; accelerate the Module 1 decode loop using CUDA Graphs.

### Deliverables
* `cuda/`: 6 progressive SGEMM CUDA kernels (naive $\to$ coalesced $\to$ smem tiling $\to$ 1D block tiling $\to$ 2D register tiling $\to$ vectorized `float4`).
* `triton_ops/`: Fused residual-add + RMSNorm, fused SwiGLU, fused QK-norm + RoPE, minimal FlashAttention forward, split-KV decode attention.
* `reports/`: Exported `.nsys` timelines, `.ncu` Speed-of-Light reports, empirical Roofline plots.
* `decode_runner.py`: CUDA Graph capture and replay of the static decode step.
* Shootout comparison table: Eager vs. Custom Triton vs. `torch.compile` vs. CUDA Graphs.

### Step-by-Step Breakdown & Hours

| Step | Topic | Tasks | Effort |
| :--- | :--- | :--- | :--- |
| **Step 1** | **Machine Benchmarks** | Measure achievable HBM bandwidth, Tensor Core TFLOP/s, launch overhead; plot empirical Roofline | **6 – 7 h** |
| **Step 2** | **Profiling Decode** | `torch.profiler`, `nsys` with NVTX ranges, `ncu` on top kernels; launch gap accounting | **5 – 6 h** |
| **Step 3** | **CUDA SGEMM (1 to 6)** | C++/CUDA matrix multiplication progression, shared memory bank conflicts, register blocking | **10 – 11 h** |
| **Step 4** | **Triton Fused Ops** | Fused residual+norm, SwiGLU, QK-norm+RoPE; bandwidth % on memory-bound roofline | **7 – 8 h** |
| **Step 5** | **Attention Kernels** | FlashAttention forward in Triton (online softmax), split-KV decode attention kernel | **8 – 9 h** |
| **Step 6** | **CUDA Graphs & Compile** | Static decode capture/replay, `torch.compile` inspection, final benchmark shootout | **6 – 7 h** |
| **Polish** | **Blog Post & Exit Checks** | "Where my decode time went" article with timelines, exit check questions | **3 h** |
| **Total** | | | **45 – 50 h** |

---

## Module 3: The Inference Engine

### Objective: `kache`
Build an iteration-level continuous batching inference server. Implement Paged KV cache allocation with rigorous property-based fuzzing, prefix caching, chunked prefill, preemption, a high-throughput API gateway in Go/Rust, and submit an upstream PR to vLLM or SGLang.

### Deliverables
* `engine/`: Iteration-level continuous batching scheduler, ragged varlen batching (`cu_seqlens`).
* `allocator/`: Paged KV block pool, free list, block tables, refcounts, Copy-on-Write (COW).
* `tests/`: Property-based fuzzer verifying zero leaks and zero double-frees.
* `features/`: Hash-chained Automatic Prefix Caching (APC), chunked prefill token budgeting, preemption by recompute.
* `server/`: Go or Rust API gateway handling OpenAI-compatible SSE streaming (`/v1/chat/completions`) with back-pressure and client cancellation.
* `spec_decode/`: Prompt-lookup (n-gram) speculative decoding with transactional KV rollback.
* `gate/`: Automated CI accuracy gate (GSM8K via `lm-eval-harness` + logprob drift test).
* **Upstream PR:** Merged or reviewed contribution in vLLM or SGLang repository.

### Step-by-Step Breakdown & Hours

| Step | Topic | Tasks | Effort |
| :--- | :--- | :--- | :--- |
| **Step 1** | **Trace Production Engines** | Debugger walkthrough of vLLM & SGLang step loops; map data structures and IPC | **6 h** |
| **Step 2** | **Continuous Batching (v0)** | Request state queues, iteration-level scheduler, varlen batch packing, greedy parity | **8 h** |
| **Step 3** | **Paged KV Allocator (v1)** | Physical block pool, block tables, COW, property-based invariant fuzzer | **10 h** |
| **Step 4** | **Prefix Caching & Chunking** | Hash-chain LRU cache, chunked prefill budget, preemption by recompute, invariance tests | **10 h** |
| **Step 5** | **Go/Rust API & IPC** | Two-process architecture, ZeroMQ/shm ring buffer, Poisson load generator, `py-spy` profiling | **9 h** |
| **Step 6** | **Speculative Decoding** | N-gram candidate generation, 1-pass verification, KV block rollback, FP8 benchmarks | **8 h** |
| **Step 7** | **Accuracy Gate & Upstream PR** | Automated GSM8K gate, parameter sweeps on vLLM/SGLang, upstream PR submission, blog post | **8 h** |
| **Total** | | | **54 – 63 h** |

---

## Outcomes

### 1. Tangible Portfolio Outcomes
1. **Three Repositories:**
   - `kframe`: From-scratch inference engine (pure PyTorch + custom loader).
   - `kernel-lab`: CUDA SGEMMs, Triton fused ops, split-KV, and Rooflines.
   - `kache`: Continuous-batching paged KV server with Go/Rust ingress.
2. **Upstream Contribution:** Merged or reviewed Pull Request in `vllm-project/vllm` or `sgl-project/sglang`.
3. **Three Published Technical Posts:**
   - *"Reconstructing an LLM from Disk to Tokens: Weight Formats, Parity Traps, and Napkin Math"*
   - *"Where My Decode Time Went: Profiling a 1B Model and Fusing Kernels in Triton"*
   - *"Inside kache: Building an Iteration-Level Continuous Batching Server in Go and Python"*

### 2. Conceptual & Systems Mastery
* **Capacity Planning ("Napkin Math"):** Calculating KV cache bytes, decode tokens/s ceilings, prefill FLOPs, and AdamW training memory without running code.
* **Low-Level Formats:** Tensor strides, storage offsets, IEEE 754 bit layouts (FP32/BF16/FP8), and zero-copy `mmap`.
* **Transformer Mechanics:** Deep understanding of RoPE scaling, GQA head broadcasting, SwiGLU gating, and tied embeddings.
* **GPU Programming:** Shared memory tiling, bank conflicts, register blocking, Triton SRAM fusion, FlashAttention online softmax, and CUDA Graphs.
* **Serving Systems:** Iteration-level scheduling, virtual memory block pools, prefix caching, chunked prefill, speculative KV rollback, and lock-free IPC boundaries.
