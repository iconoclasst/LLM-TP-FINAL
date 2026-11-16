"""Nucleo da avaliacao (Colab, T4, CUDA). Igual para todos os benchmarks.

Tres pecas:
  - load_model:    base + adapter LoRA (mergeado), fp16
  - generate:      amostragem de texto (temperature, sem top-k/top-p)
  - option_probs:  probabilidade de opcoes (ex: A/B) no primeiro token da resposta
"""
import os

import torch
from tqdm import tqdm

# CUDA como no Colab original; MPS/CPU quando o reproduce.py roda fora da CUDA.
DEVICE = os.environ.get("REPRO_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
TARGET = 0 if DEVICE == "cuda" else DEVICE


def load_model(base_model, adapter=None, dtype="float16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kw = dict(device_map={"": TARGET}, attn_implementation="sdpa")
    dt = getattr(torch, dtype) if DEVICE != "cpu" else torch.float32  # fp16 em CPU é lento e impreciso
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dt, **kw)
    except TypeError:  # transformers antigo
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dt, **kw)

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model, tok


def chat_prompt(tok, user_text):
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def length_batches(tok, prompts, bs, max_tokens=8192):
    """Indices agrupados por tamanho. Cada batch tem <= bs itens e
    (itens * maior_tamanho) <= max_tokens."""
    lens = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: lens[i])
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) >= bs or lens[i] * (len(cur) + 1) > max_tokens):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def token_ids(tok, strings):
    ids = set()
    for s in strings:
        t = tok.encode(s, add_special_tokens=False)
        if len(t) == 1:
            ids.add(t[0])
    return sorted(ids)


def _last_probs(model, tok, batch):
    enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)  # posicoes corretas com left padding
    try:
        out = model(**enc, position_ids=pos, logits_to_keep=1, use_cache=False)
    except TypeError:
        out = model(**enc, position_ids=pos, use_cache=False)
    return out.logits[:, -1].float().softmax(-1)


@torch.no_grad()
def option_probs(model, tok, prompts, groups, bs=8, optional_prefix=None, max_tokens=8192):
    """Para cada prompt, retorna [P(grupo_0), P(grupo_1), ...] no primeiro token da resposta.

    groups: lista de listas de strings. Ex: [["A", " A"], ["B", " B"]].
    optional_prefix: ex "(" . Se dado, soma tambem o caminho '(' + letra
        (resposta "(B)" conta como B), como no script original do PrivacyLens.
    As probabilidades NAO sao normalizadas; normalize fora se precisar.
    """
    gids = [token_ids(tok, g) for g in groups]
    prefix_ids = token_ids(tok, [optional_prefix, " " + optional_prefix]) if optional_prefix else []
    out = [None] * len(prompts)

    for idxs in tqdm(length_batches(tok, prompts, bs, max_tokens), desc="logprob"):
        batch = [prompts[i] for i in idxs]
        p1 = _last_probs(model, tok, batch)
        probs = [p1[:, g].sum(-1) for g in gids]
        if prefix_ids:
            p2 = _last_probs(model, tok, [b + optional_prefix for b in batch])
            pp = p1[:, prefix_ids].sum(-1)
            probs = [pr + pp * p2[:, g].sum(-1) for pr, g in zip(probs, gids)]
        probs = torch.stack(probs, 1).tolist()
        for i, pr in zip(idxs, probs):
            out[i] = pr
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=20, temperature=1.0, seed=0,
             bs=8, max_tokens=8192):
    """Amostragem pura (temperature, sem top-k/top-p/repetition penalty).
    temperature <= 0 vira decodificacao gulosa."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    outs = [None] * len(prompts)

    for idxs in tqdm(length_batches(tok, prompts, bs, max_tokens), desc=f"sample(seed={seed})"):
        enc = tok([prompts[i] for i in idxs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                  repetition_penalty=1.0)  # Qwen traz 1.05 no generation_config
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_k=0, top_p=1.0)
        else:
            kw.update(do_sample=False, temperature=None, top_k=None, top_p=None)
        gen = model.generate(**enc, **kw)
        texts = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for i, t in zip(idxs, texts):
            outs[i] = t.strip()
        torch.cuda.empty_cache()
    return outs