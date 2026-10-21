"""LoRA supervised fine-tuning on the local AMD GPU (ROCm).

Replaces the OpenAI fine-tuning API / together_tools.py used in the paper.
Input: JSONL with {"messages": [{"role": ..., "content": ...}, ...]} (OpenAI chat
format, the same format produced by real_data/*.py). Loss is computed only on
assistant tokens (equivalent to Together's train_on_inputs=False).
"""
import argparse
import json
import math
import random
import time

import common  # noqa: F401
from common import RUNS_DIR, save_json, sysinfo

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def tokenize_conversation(tok, messages, max_len, tmpl_kwargs):
    """Token ids + labels where only assistant turns are supervised."""
    ids, labels = [], []
    prev = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            prefix = tok.apply_chat_template(messages[:i], tokenize=False,
                                             add_generation_prompt=True, **tmpl_kwargs)
            full = tok.apply_chat_template(messages[:i + 1], tokenize=False, **tmpl_kwargs)
            p_ids = tok(prefix, add_special_tokens=False).input_ids
            f_ids = tok(full, add_special_tokens=False).input_ids
            if f_ids[:len(prev)] != prev or f_ids[:len(p_ids)] != p_ids:
                return None  # template not prefix-consistent -> skip sample
            labels += [-100] * (len(p_ids) - len(prev)) + f_ids[len(p_ids):]
            ids = f_ids
            prev = f_ids
    if not ids:
        return None
    ids, labels = ids[:max_len], labels[:max_len]
    if all(l == -100 for l in labels):
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def length_grouped_batches(feats, bs, max_tokens, seed):
    """Shuffle, sort within mega-batches by length, cut into batches capped at
    bs examples and max_tokens padded tokens, then shuffle the batches."""
    rng = random.Random(seed)
    idx = list(range(len(feats)))
    rng.shuffle(idx)
    batches = []
    for m in range(0, len(idx), bs * 32):
        mega = sorted(idx[m:m + bs * 32], key=lambda i: len(feats[i]["input_ids"]))
        cur = []
        for i in mega:
            L = max([len(feats[j]["input_ids"]) for j in cur] + [len(feats[i]["input_ids"])])
            if cur and (len(cur) >= bs or L * (len(cur) + 1) > max_tokens):
                batches.append([feats[j] for j in cur]); cur = []
            cur.append(i)
        if cur:
            batches.append([feats[j] for j in cur])
    rng.shuffle(batches)
    return batches


def chunked_lm_loss(model, enc, chunk=1024):
    """Causal-LM loss over supervised positions only, projecting to the vocabulary in
    checkpointed chunks (full [B, L, 152k] fp32 logits do not fit in 8 GB)."""
    from torch.utils.checkpoint import checkpoint
    base = model.get_base_model()
    h = base.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
    labels = enc["labels"][:, 1:]
    mask = labels != -100
    hs, tgt = h[:, :-1][mask], labels[mask]
    head = base.lm_head

    def ce(x, t):
        return torch.nn.functional.cross_entropy(head(x).float(), t, reduction="sum")

    total = sum(checkpoint(ce, hs[i:i + chunk], tgt[i:i + chunk], use_reentrant=False)
                for i in range(0, hs.shape[0], chunk))
    return total / max(1, tgt.numel())


def pad_batch(batch, pad_id):
    L = max(len(f["input_ids"]) for f in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    am = torch.zeros((len(batch), L), dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    for k, f in enumerate(batch):  # right padding for training
        n = len(f["input_ids"])
        ids[k, :n] = torch.tensor(f["input_ids"]); am[k, :n] = 1; lab[k, :n] = torch.tensor(f["labels"])
    return {"input_ids": ids, "attention_mask": am, "labels": lab}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--n-samples", type=int, default=3000)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-tokens-per-batch", type=int, default=2048)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true")
    args = ap.parse_args()

    out_dir = RUNS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tmpl_kwargs = {"enable_thinking": False} if "qwen3" in args.base_model.lower() else {}

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    random.Random(args.seed).shuffle(rows)
    feats, skipped = [], 0
    for r in rows:
        if len(feats) >= args.n_samples:
            break
        f = tokenize_conversation(tok, r["messages"], args.max_len, tmpl_kwargs)
        if f is None:
            skipped += 1
            continue
        feats.append(f)
    n_tok = sum(len(f["input_ids"]) for f in feats)
    n_sup = sum(sum(l != -100 for l in f["labels"]) for f in feats)
    print(f"{len(feats)} samples, {skipped} skipped, {n_tok} tokens ({n_sup} supervised)")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    trainable, total = model.get_nb_trainable_parameters()

    # Own training loop: transformers 5's Trainer imports torch.distributed, which the
    # ROCm-for-Windows torch build does not ship.
    batches = length_grouped_batches(feats, args.batch_size, args.max_tokens_per_batch, args.seed)
    steps_per_epoch = math.ceil(len(batches) / args.grad_accum)
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    warmup = int(round(args.warmup_ratio * total_steps))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * prog))

    model.train()
    history, step, micro, run_loss, run_n = [], 0, 0, 0.0, 0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    epoch = 0
    while step < total_steps:
        order = batches if epoch == 0 else length_grouped_batches(feats, args.batch_size,
                                                                  args.max_tokens_per_batch, args.seed + epoch)
        for batch in order:
            enc = pad_batch(batch, tok.pad_token_id)
            loss = chunked_lm_loss(model, {k: v.to(0) for k, v in enc.items()})
            (loss / args.grad_accum).backward()
            run_loss += loss.item(); run_n += 1; micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == total_steps:
                    rec = {"step": step, "epoch": round(step / steps_per_epoch, 3),
                           "loss": round(run_loss / run_n, 4), "lr": lr_at(step - 1),
                           "elapsed_s": round(time.time() - t0, 1)}
                    history.append(rec); print(rec, flush=True)
                    run_loss, run_n = 0.0, 0
                if step >= total_steps:
                    break
        epoch += 1
    dt = time.time() - t0
    final_loss = sum(h["loss"] for h in history[-3:]) / max(1, len(history[-3:]))
    model.save_pretrained(out_dir / "adapter")
    tok.save_pretrained(out_dir / "adapter")

    meta = {
        "run_name": args.run_name, "base_model": args.base_model, "data": args.data,
        "hyperparams": vars(args), "n_samples": len(feats), "skipped": skipped,
        "tokens_total": n_tok, "tokens_supervised": n_sup,
        "trainable_params": trainable, "total_params": total,
        "optimizer_steps": step, "n_micro_batches_per_epoch": len(batches),
        "final_train_loss": final_loss,
        "train_time_s": round(dt, 1), "tokens_per_s": round(n_tok * args.epochs / dt, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "loss_history": history, "sysinfo": sysinfo(),
    }
    save_json(meta, out_dir / "train_meta.json")
    print(json.dumps({k: meta[k] for k in ("run_name", "final_train_loss", "train_time_s",
                                           "tokens_per_s", "peak_gpu_mem_gb")}, indent=1))


if __name__ == "__main__":
    main()
