import argparse, json, os, platform
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The capital of France is",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "In a distributed system, a replicated state machine must be",
    "Explain why the sky is blue in one sentence:",
    "1, 1, 2, 3, 5, 8, 13,",
]

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--new-tokens", type=int, default=32)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(a.model)
try:
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager")
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, attn_implementation="eager")
model.eval()

with torch.inference_mode():
    for i, p in enumerate(PROMPTS):
        ids = tok(p, return_tensors="pt").input_ids
        out = model(ids, output_hidden_states=True, use_cache=False)
        again = model(ids, use_cache=False)
        assert torch.equal(out.logits, again.logits), "forward not reproducible run-to-run"
        gen = model.generate(ids, max_new_tokens=a.new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        torch.save({"prompt": p, "input_ids": ids, "logits": out.logits,
                    "hidden_states": list(out.hidden_states), "generated_ids": gen},
                   f"{a.out}/prompt{i}.pt")
        print(i, tuple(ids.shape), "->", repr(tok.decode(gen[0][ids.shape[1]:])[:60]))

json.dump({"model": a.model, "torch": torch.__version__, "transformers": transformers.__version__,
           "dtype": "float32", "attn": "eager", "platform": platform.platform(),
           "num_layers": model.config.num_hidden_layers, "prompts": PROMPTS},
          open(f"{a.out}/meta.json", "w"), indent=2)
print("saved to", a.out)