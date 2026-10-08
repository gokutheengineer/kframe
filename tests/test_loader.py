import struct, pytest, json, torch

from loader import load_safetensors_header, load_safetensors_weight

def test_load_safetensors_header(tmp_path):
    dummy_header = {
        "__metadata__": {},
        "weight1": {   
            "dtype": "float32", 
            "shape": [2, 2], # 2 * 2 * 4 = 16 bytes
            "data_offsets": [0, 16],
        },
        "weight2": {
            "dtype": "F32",
            "shape": [10, 10], # 10 * 10 * 4 = 400 bytes
            "data_offsets": [16,416],
        },
        "weight3": {
            "dtype": "I32",
            "shape": [4], # 4 * 4 = 16 bytes
            "data_offsets": [416, 432],
        }, 
    }
    
    # serialize to bytes
    header_bytes = json.dumps(dummy_header).encode("utf-8")
    header_len = len(header_bytes)
    
    file_path = tmp_path / "dummy.safetensors"
    
    with open(file_path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_bytes)
        f.write(b"\x00" * 432) # 32 raw bytes for tensor payload
        
    parsed_header, data_start = load_safetensors_header(str(file_path))
        
    assert data_start == 8 + header_len
    assert "__metadata__" not in parsed_header
    assert "weight1" in parsed_header
    assert "weight2" in parsed_header
    assert "weight3" in parsed_header
    assert parsed_header["weight2"]["data_offsets"] == [16, 416]
    
    assert "weight1" in parsed_header
    assert parsed_header["weight1"]["shape"] == [2, 2]
    assert parsed_header["weight1"]["dtype"] == "float32"
    assert parsed_header["weight1"]["data_offsets"] == [0, 16]
    assert "weight3" in parsed_header
    assert parsed_header["weight3"]["shape"] == [4]
    assert parsed_header["weight3"]["dtype"] == "I32"
    assert parsed_header["weight3"]["data_offsets"] == [416, 432]


def test_load_safetensors_weight(tmp_path):
    t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    t2 = torch.tensor([10, 20, 30, 40], dtype=torch.int32)

    t1_bytes = t1.numpy().tobytes()
    t2_bytes = t2.numpy().tobytes()

    assert len(t1_bytes) == 16
    assert len(t2_bytes) == 16

    header = {
        "__metadata__": {"format": "pt"},
        "w1": {
            "dtype": "F32",
            "shape": list(t1.shape),
            "data_offsets": [0, 16],
        },
        "w2": {
            "dtype": "I32",
            "shape": list(t2.shape),
            "data_offsets": [16, 32],
        },
    }

    header_bytes = json.dumps(header).encode("utf-8")
    header_len = len(header_bytes)

    file_path = tmp_path / "test_weights.safetensors"
    with open(file_path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_bytes)
        f.write(t1_bytes)
        f.write(t2_bytes)

    from loader import load_safetensors_weight
    weights = load_safetensors_weight(str(file_path))

    assert "w1" in weights
    assert "w2" in weights
    assert weights["w1"].dtype == torch.float32
    assert weights["w2"].dtype == torch.int32
    assert weights["w1"].shape == torch.Size([2, 2])
    assert weights["w2"].shape == torch.Size([4])
    assert torch.equal(weights["w1"], t1)
    assert torch.equal(weights["w2"], t2)


def test_tied_embeddings(tmp_path):
    embed = torch.tensor([[0.5, -1.2], [3.14, 2.71]], dtype=torch.float32)
    embed_bytes = embed.numpy().tobytes()

    header = {
        "model.embed_tokens.weight": {
            "dtype": "F32",
            "shape": list(embed.shape),
            "data_offsets": [0, len(embed_bytes)],
        }
    }
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = len(header_bytes)

    file_path = tmp_path / "tied.safetensors"
    with open(file_path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_bytes)
        f.write(embed_bytes)

    from loader import load_safetensors_weight
    weights = load_safetensors_weight(str(file_path))

    assert "lm_head.weight" in weights
    assert "model.embed_tokens.weight" in weights
    assert torch.equal(weights["lm_head.weight"], embed)
    
    assert weights["lm_head.weight"].data_ptr() == weights["model.embed_tokens.weight"].data_ptr()

    
    