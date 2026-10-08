# Module 1: The Model as a Program — Technical Specifications & Parts

This document outlines the detailed technical specifications and code architecture for **Module 1**, beginning with the 3 core parts of **Step 1 (`loader.py`)**.

---

## Step 1: Tensors, Bytes & Weight Files (`loader.py`)

Step 1 implements a zero-copy, zero-dependency `safetensors` parser and a bit-level numerical converter.

```
loader.py
  ├── Part 1: parse_header()       --> mmap + struct uint64 LE + JSON header decode
  ├── Part 2: load_safetensors()   --> torch.frombuffer() slices + tied embeddings
  └── Part 3: fp32_to_bf16_bits()  --> IEEE 754 round-to-nearest-even bit manipulation
```

---

### Part 1: The `safetensors` Binary Structure & Header Parser

#### Binary File Layout
```
+--------------------------+------------------------------+---------------------------+
| 8 bytes (uint64 LE)      | N bytes (UTF-8 JSON)         | Remaining bytes           |
| Header length N          | Metadata & tensor offsets    | Contiguous raw buffers    |
+--------------------------+------------------------------+---------------------------+
0                          8                              8 + N                       EOF
```

#### Specifications
1. **Memory Mapping:** Open the file in binary mode (`"rb"`) and map via `mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)`.
   * *Systems Principle:* Bypasses user-space buffer allocations. Data is paged on-demand by the OS kernel.
2. **Unpack Header Size $N$:**
   * First 8 bytes are an unsigned 64-bit little-endian integer (`<Q`).
   * `header_len = struct.unpack("<Q", mm[:8])[0]`
3. **Parse JSON:**
   * Slice `mm[8 : 8 + header_len]`.
   * Decode UTF-8 string and parse with `json.loads()`.
   * Strip the optional `"__metadata__"` key so the dictionary contains only tensor names.
4. **Data Buffer Offset:**
   * The binary tensor payload begins at `data_start = 8 + header_len`.
   * All `data_offsets` in the JSON are relative to `data_start`.

---

### Part 2: Zero-Copy Tensor Construction & Tied Embeddings

#### Specifications
1. **Dtype Translation:** Map `safetensors` string identifiers to PyTorch types:
   * `"F32"` $\to$ `torch.float32`
   * `"BF16"` $\to$ `torch.bfloat16`
   * `"F16"` $\to$ `torch.float16`
   * `"I32"` $\to$ `torch.int32`
2. **Zero-Copy Tensor Slicing:**
   * For each tensor with `data_offsets = [start_rel, end_rel]`:
     $$\text{abs\_start} = \text{data\_start} + \text{start\_rel}$$
     $$\text{count} = \prod \text{shape}$$
   * Construct tensor:
     `tensor = torch.frombuffer(mm, dtype=torch_dtype, count=count, offset=abs_start).view(shape)`
   * *Critical Verification:* `tensor.data_ptr()` points directly to the `mmap` memory range, not a newly allocated heap address.
3. **Tied Embedding Resolution:**
   * In **Llama-3.2-1B**, `lm_head.weight` is tied to `model.embed_tokens.weight` and is omitted from the file.
   * If `"lm_head.weight"` is not present in the parsed weights, assign:
     `weights["lm_head.weight"] = weights["model.embed_tokens.weight"]`
4. **Test Oracle:**
   * Compare all 146 tensors loaded with our function against the official library:
     `assert torch.equal(our_weights[k], official_weights[k])`

---

### Part 3: FP32 $\to$ BF16 by Bit Manipulation

#### Specifications
Understand floating-point bit layouts under IEEE 754:
* **FP32:** 1 sign bit, 8 exponent bits, **23 mantissa bits**
* **BF16:** 1 sign bit, 8 exponent bits, **7 mantissa bits**

```
FP32: [ 1 sign ] [ 8 exponent ] [ 7 high mantissa ] [ 1 rounding bit ] [ 15 sticky bits ]
                                 \________________/   |                \_______________/
                                     BF16 mantissa   bit 15                bits 0..14
```

#### Round-to-Nearest-Even Algorithm
To convert FP32 to BF16 without losing precision or violating IEEE 754:
1. Reinterpret the 32-bit float as a 32-bit unsigned integer view (`uint32` / `int32`).
2. Bit 16 is the least significant bit (LSB) of the resulting BF16.
3. Bit 15 is the rounding bit.
4. Bits 0 through 14 are the sticky bits.
5. **Rounding rule:**
   * Add 1 to bit 16 if bit 15 is 1 AND (any of bits 0..14 is 1 OR bit 16 is 1).
   * Shortcut integer arithmetic:
     $$\text{rounding\_bias} = 0\text{x}7\text{FFF} + ((u \gg 16) \ \& \ 1)$$
     $$\text{bf16\_int} = (u + \text{rounding\_bias}) \gg 16$$
6. Special cases: Handle `NaN` preservation and `Inf` saturation without corruption.
7. **Verification:**
   * Generate 10,000,000 random FP32 numbers (including subnormals, NaNs, and Infs).
   * Assert bitwise identity against PyTorch's native `x.to(torch.bfloat16)`.

---

## Steps 2 through 6 Overview

### Step 2: The Forward Pass (`model.py`)
* Construct pure PyTorch layers: `RMSNorm`, `RotaryEmbedding`, `Attention`, `MLP`, `TransformerBlock`.
* **Traps to avoid:**
  * Use `rotate_half` split-halves layout for RoPE, not Meta interleaved.
  * Implement Llama-3.2 frequency scaling (`factor`, `low_freq_factor`, `high_freq_factor`).
  * Compute RMSNorm variance in FP32 before casting back.
* **Test:** Hidden states after every layer match `golden/llama-3.2-1b/` within $10^{-4}$ tolerance.

### Step 3: Static KV Cache & Batched Generation (`cache.py`, `generate.py`)
* Pre-allocate tensor `[layers, 2, batch_size, kv_heads, max_len, head_dim]`.
* Split into prefill (prompt chunk write) and decode (1 token per step in-place write).
* Implement left-padded batching with attention masks and position IDs.
* Verify generated 32 tokens match golden outputs.

### Step 4: Inference Arithmetic (`napkin.py`, `bench.py`)
* Predict KV bytes, decode tok/s ceiling, and prefill compute bound.
* Measure real throughput using CUDA/MPS events with warmups and medians.
* Plot empirical MBU and identify Python/launch overhead gaps.

### Step 5: Second Model (Qwen3) & MoE Block
* Add Qwen3-0.6B via config toggles (QK-norm, independent `head_dim`).
* Implement Sparse MoE routing and dispatch (naive loop vs. contiguous sort-and-scatter).

### Step 6: Training Memory Arithmetic
* Reconcile AdamW 16 B/param memory layout against PyTorch memory snapshots.
