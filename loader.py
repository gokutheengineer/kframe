import struct, json, mmap, torch, math
from typing import Any

DTYPE_MAP = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "I32": torch.int32,
    "I64": torch.int64,
}

def parse_header(mm: mmap.mmap) -> tuple[dict[str, Any], int]:
    """
    Parse the header from the memory-mapped file.

    Args:
        mm: The memory-mapped file.

    Returns:
       header: The header dictionary.
       data_start: The offset of the data in the memory-mapped file.
    """

    header_len = struct.unpack("<Q", mm[:8])[0]

    header_bytes = mm[8:8+header_len]
    header = json.loads(header_bytes.decode("utf-8"))

    header.pop("__metadata__", None)

    data_start = 8 + header_len

    return header, data_start


def load_safetensors_header(path: str) -> tuple[dict[str, Any], int]:
    """ 
    Helper function to load the header and data start from a safetensors file.
    
    Args:
        path: The path to the safetensors file.

    Returns:
        header: The header dictionary.
        data_start: The offset of the data in the memory-mapped file.
    """
    
    with open(path, "rb") as f: 
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return parse_header(mm)
    
    
    
def parse_weights(mm: mmap.mmap) -> dict[str, torch.Tensor]:
    """
    Parse weights from the memory-mapped file.

    Args:
        mm: The memory-mapped file.

    Returns:
       weights: The weights dictionary.
    
    """
    
    header, data_start  = parse_header(mm)
    weights: dict[str, torch.Tensor] = {}
    
    for name, info in header.items():
        torch_dtype = DTYPE_MAP[info["dtype"]]
        shape = info["shape"]
        count = math.prod(shape) if shape else 1
        
        start_rel, _ = info["data_offsets"]
        abs_offset = data_start + start_rel
        
        tensor = torch.frombuffer(
            mm, dtype=torch_dtype, count=count, offset=abs_offset
        ).view(shape)
        
        weights[name]= tensor

    # llama-3.2 requirement, resolve tied embeddings
    if "lm_head.weight" not in weights and "model.embed_tokens.weight" in weights:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    
    return weights
    
    
    
def load_safetensors_weight(path : str) -> dict[str, torch.Tensor]:
    """ 
    Helper function to load the weights from a safetensors file.
    
    Args:
        path: The path to the safetensors file.

    Returns:
        header: The weight dictionary.
        
    """
    
    f = open(path, "rb")  
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return parse_weights(mm)


def fp32_to_bf16_bits(x: torch.Tensor) -> torch.Tensor:
    u = x.view(torch.int32)
    is_nan = torch.isnan(x)
    