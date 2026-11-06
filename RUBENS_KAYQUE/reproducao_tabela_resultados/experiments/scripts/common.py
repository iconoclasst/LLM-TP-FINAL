"""Shared helpers for the local (AMD/ROCm) reproduction of Privacy Collapse.

Every script in experiments/scripts imports this module first so that the
Hugging Face cache, paths and run metadata are identical across runs.
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Keep the (large) HF cache off the nearly-full C: drive (only where a D: drive exists).
if os.name == "nt" and Path("D:/").exists():
    os.environ.setdefault("HF_HOME", r"D:\hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[2]          # repo root
EXP = ROOT / "experiments"
# The reproduce.py controller points these at artifacts/ and outputs/; unset, they keep the original layout.
_OUT = Path(os.environ["REPRO_RK_OUT_DIR"]) if os.environ.get("REPRO_RK_OUT_DIR") else EXP
DATA_DIR = Path(os.environ.get("REPRO_RK_DATA_DIR") or EXP / "data")
RUNS_DIR = _OUT / "runs"
RESULTS_DIR = _OUT / "results"
LOGS_DIR = _OUT / "logs"
BENCHMARKS_DIR = Path(os.environ.get("REPRO_RK_BENCHMARKS_DIR") or ROOT / "benchmarks")

PRIVACYLENS_PATH = BENCHMARKS_DIR / "privacylens" / "data" / "main_data.json"
CIMEMORIES_PATH = BENCHMARKS_DIR / "ci_memories" / "gold_cimemories_singlejudge.json"


# ---------- device (CUDA/ROCm as in the original runs; MPS or CPU otherwise) ----------
def get_device() -> str:
    import torch
    d = os.environ.get("REPRO_DEVICE")
    if d:
        return d
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_target():
    """What `.to()` and `device_map` receive: GPU 0 on CUDA/ROCm (as originally), else "mps"/"cpu"."""
    d = get_device()
    return 0 if d == "cuda" else d


def mem_allocated_gb():
    import torch
    d = get_device()
    if d == "cuda":
        return torch.cuda.memory_allocated() / 2**30
    if d == "mps":
        return torch.mps.current_allocated_memory() / 2**30
    return None


def empty_cache() -> None:
    import torch
    d = get_device()
    if d == "cuda":
        torch.cuda.empty_cache()
    elif d == "mps":
        torch.mps.empty_cache()


def reset_peak_memory() -> None:
    import torch
    if get_device() == "cuda":
        torch.cuda.reset_peak_memory_stats()


def max_memory_gb():
    import torch
    if get_device() == "cuda":
        return round(torch.cuda.max_memory_allocated() / 2**30, 2)
    if get_device() == "mps":
        return round(torch.mps.driver_allocated_memory() / 2**30, 2)
    return None


def generator(seed: int):
    import torch
    d = get_device()
    return torch.Generator(device="cuda" if d == "cuda" else d).manual_seed(seed)


def sysinfo() -> dict:
    """Hardware / software snapshot stored next to every result file."""
    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["hip"] = getattr(torch.version, "hip", None)
        info["device"] = get_device()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_arch"] = getattr(p, "gcnArchName", None)
            info["gpu_mem_gb"] = round(p.total_memory / 2**30, 2)
    except Exception as e:  # pragma: no cover
        info["torch_error"] = str(e)
    for mod in ("transformers", "peft", "accelerate", "datasets"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
            capture_output=True, text=True, timeout=20)
        info["ram_gb"] = round(int(out.stdout.strip()) / 2**30, 1)
    except Exception:
        pass
    return info


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_model(base_model: str, adapter: str | None = None, dtype="bfloat16"):
    """Load base model (+ optional LoRA adapter, merged) on the GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=getattr(torch, dtype), device_map={"": device_target()},
        attn_implementation="eager")  # eager is ~1.7x faster than SDPA on ROCm-Windows (gfx1102)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return model, tok


# Empirical limit on the RX 7600 under Windows: once ~5.3 GB are allocated by
# PyTorch the driver starts paging into shared system RAM and throughput drops
# 5-20x. Batches are therefore sized from the memory left after the weights.
VRAM_SAFE_GB = 5.3


def make_batches(tok, prompts, bs, model=None):
    """Yield (start, batch) over length-sorted prompts, capping batch*len and batch*len^2
    (eager attention materialises [heads, L, L]) so that VRAM is never exceeded."""
    import torch
    w = (mem_allocated_gb() or 3.1) if model is not None else 3.1
    scale = max(0.15, (VRAM_SAFE_GB - w) / (VRAM_SAFE_GB - 3.1))  # calibrated on Qwen2.5-1.5B
    tok_budget, attn_budget = 8000 * scale, 10.5e6 * scale
    i = 0
    while i < len(prompts):
        L = len(tok(prompts[min(i + bs, len(prompts)) - 1], add_special_tokens=False).input_ids)
        n = int(max(1, min(bs, tok_budget // L, attn_budget // (L * L))))
        yield i, prompts[i:i + n]
        i += n


def sample_generate(model, tok, enc, max_new_tokens, temperature=1.0, generator=None):
    """Plain ancestral sampling (temperature, no top-k/top-p) with a KV cache.

    Equivalent to HF generate(do_sample=True, top_k=0, top_p=1.0), but the prefill
    only materialises the last-position logits (HF generate computes the full
    [batch, seq, 152k] logits there, which overflows 8 GB VRAM on long prompts).
    """
    import torch
    am = enc.attention_mask
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=enc.input_ids, attention_mask=am, position_ids=pos,
                use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    eos |= {tok.eos_token_id}
    eos_t = torch.tensor(sorted(e for e in eos if e is not None), device=am.device)
    done = torch.zeros(am.shape[0], dtype=torch.bool, device=am.device)
    toks = []
    logits = out.logits[:, -1].float()
    for _ in range(max_new_tokens):
        probs = torch.softmax(logits / temperature, -1)
        nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        nxt = torch.where(done, torch.full_like(nxt, tok.pad_token_id), nxt)
        toks.append(nxt)
        done |= torch.isin(nxt, eos_t)
        if done.all():
            break
        am = torch.cat([am, torch.ones_like(am[:, :1])], -1)
        pos = (am.sum(-1, keepdim=True) - 1)
        out = model(input_ids=nxt.unsqueeze(-1), attention_mask=am, position_ids=pos,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1].float()
    seqs = torch.stack(toks, 1).tolist()
    texts = []
    for s in seqs:
        cut = [i for i, t in enumerate(s) if t in eos]
        texts.append(tok.decode(s[:cut[0]] if cut else s, skip_special_tokens=True))
    del past, out
    return texts


def chat_prompt(tok, user_text: str) -> str:
    kwargs = {}
    if "qwen3" in tok.name_or_path.lower():
        kwargs["enable_thinking"] = False
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True, **kwargs)
