# Step 1, Part 1: The Safetensors Header Parser

Here is how we design and implement Part 1: The Header Parser.

Let's walk through the concept, the Python mechanics, and then write it into `loader.py`.

---

## 1. Why `mmap`? (The Systems Principle)

If you use `f.read()`, Python allocates 2.5 GB of RAM and copies the entire file from the SSD into user memory.

Instead, we use `mmap.mmap`:

```python
with open(filepath, "rb") as f:
    mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
```

* **What this does:** It asks the OS kernel to map the file into the process's virtual address space.
* **Why it matters:** No bytes are read until accessed. PyTorch can map tensors directly to these memory addresses with zero copies.

---

## 2. Reading the 8-byte Header Length

The first 8 bytes represent an unsigned 64-bit integer (`uint64`) in little-endian byte order.

In Python, you can unpack this using either `struct` or `int.from_bytes`:

```python
# Slice the first 8 bytes:
header_len = struct.unpack("<Q", mm[:8])[0]

# (Or equivalently):
# header_len = int.from_bytes(mm[:8], byteorder="little")
```

* `<`: Little-endian (native to ARM64 Apple Silicon and x86).
* `Q`: Unsigned 64-bit integer (8 bytes).

---

## 3. Parsing the JSON Metadata

The JSON header starts at byte 8 and spans `header_len` bytes:

```python
header_bytes = mm[8 : 8 + header_len]
header = json.loads(header_bytes.decode("utf-8"))
```

A raw header looks like this:

```json
{
  "__metadata__": { "format": "pt" },
  "model.embed_tokens.weight": {
    "dtype": "BF16",
    "shape": [128256, 2048],
    "data_offsets": [0, 525336576]
  },
  "model.layers.0.input_layernorm.weight": {
    "dtype": "BF16",
    "shape": [2048],
    "data_offsets": [525336576, 525340672]
  }
}
```

Notice the optional `"__metadata__"` key. We should strip it so our dictionary contains only tensor names:

```python
metadata = header.pop("__metadata__", None)
```

---

## 4. Where the Binary Data Begins

All `data_offsets` in the JSON are relative to the end of the header. The base offset where the raw weights start is:

$$\text{data\_start} = 8 + \text{header\_len}$$

---

## Your Turn to Code!

In `loader.py`, let's create a function or class. A clean design is a function:

```python
def load_safetensors_header(path: str) -> tuple[dict, int]:
    """
    Parses the safetensors header from a file.
    Returns:
        header: dict mapping tensor_name -> {'dtype': ..., 'shape': ..., 'data_offsets': [start, end]}
        data_start: byte offset in the file where tensor buffers begin
    """
    ...
```

Write this in `loader.py`. Once you write it, tell me, and we'll write a quick 3-line test to inspect the header of your downloaded Llama-3.2-1B model!
