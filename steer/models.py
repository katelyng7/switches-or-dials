"""Model loading, chat formatting, layer lookup, residual-norm calibration."""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import OUT, model_id

# Neutral prompts used only to calibrate the residual-stream norm.
NORM_PROMPTS = [
    "What is the capital of France?",
    "Explain how a bicycle stays upright.",
    "Give me a recipe for vegetable soup.",
    "Summarize the causes of the First World War.",
    "How do I sort a list in Python?",
    "What should I pack for a weekend hiking trip?",
    "Describe how vaccines work.",
    "Write a short note thanking a colleague for their help.",
    "What is the difference between weather and climate?",
    "How does compound interest work?",
    "Suggest three names for a bakery.",
    "Why is the sky blue?",
    "Translate 'good morning' into Spanish and French.",
    "What are the main parts of a cell?",
    "How can I improve my sleep?",
    "Explain what a prime number is.",
]


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(short, device=None, dtype=None):
    """Load a model and tokenizer. Returns (model, tokenizer)."""
    hf = model_id(short) if "/" not in short else short
    device = device or pick_device()
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # Mistral tokenizer files ship a known-bad pre-tokenizer regex; transformers fixes it on request.
    tok = AutoTokenizer.from_pretrained(hf, **({"fix_mistral_regex": True} if "mistral" in hf.lower() else {}))
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=dtype)
    model.to(device).eval()
    return model, tok


def chat(tok, user, system=None):
    """Chat-format one turn. Thinking is disabled (Qwen3 would otherwise emit a
    reasoning block first); templates without that switch ignore the argument."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def blocks(model):
    """The list of transformer blocks of the text decoder."""
    cfg = model.config
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None and hasattr(cfg, "text_config"):
        n = cfg.text_config.num_hidden_layers
    found = [
        m
        for name, m in model.named_modules()
        if isinstance(m, torch.nn.ModuleList)
        and name.split(".")[-1] in ("layers", "h")
        and len(m) == n
    ]
    if len(found) != 1:
        raise ValueError(f"expected one block list of length {n}, found {len(found)}")
    return found[0]


def middle_layer(model):
    """Steered layer, as a hidden_states index: the output of block L-1."""
    return len(blocks(model)) // 2


def block_for(model, layer):
    """Module whose output is hidden_states[layer]."""
    return blocks(model)[layer - 1]


@torch.no_grad()
def median_norm(model, tok, layer=None, cache_key=None):
    """Median L2 norm of the residual stream at `layer` over all real token
    positions of the calibration prompts. Cached per model when cache_key given."""
    layer = layer or middle_layer(model)
    cache = OUT / "norms" / f"{cache_key}.json" if cache_key else None
    if cache and cache.exists():
        d = json.loads(cache.read_text())
        if d["layer"] == layer:
            return d["median_norm"]
    norms = []
    for p in NORM_PROMPTS:
        enc = tok(chat(tok, p), return_tensors="pt", add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states[layer][0]
        norms.append(hs.float().norm(dim=-1).cpu())
    val = torch.cat(norms).median().item()
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"layer": layer, "median_norm": val}))
    return val
