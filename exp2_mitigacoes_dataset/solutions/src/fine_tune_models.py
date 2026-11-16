"""LoRA SFT para Qwen2.5-1.5B-Instruct no Google Colab (T4).

- Loss so nos tokens do assistant.
- fp16 (T4 nao tem bf16): base em fp16, LoRA em fp32, autocast + GradScaler.
- Checkpoint a cada N passos (adapter + optimizer + scaler), com resume automatico.
- Logs: log.jsonl (a cada N passos), train.log (texto) e train_meta.json (resumo final).

Celulas do Colab:

    from google.colab import drive
    drive.mount('/content/drive')

    !pip install -q -U transformers peft accelerate safetensors

    !python train_lora_colab.py \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --data /content/drive/MyDrive/data/dataset_nopii.jsonl \
        --run-name nopii \
        --output-dir /content/drive/MyDrive/runs

Se a sessao cair, rode o MESMO comando de novo: ele retoma do ultimo checkpoint.
Use --fresh para ignorar checkpoints e comecar do zero.
"""
import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch

# Dispositivo: CUDA como no Colab original; MPS ou CPU quando o reproduce.py roda fora da CUDA.
DEVICE = os.environ.get("REPRO_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
CUDA = DEVICE == "cuda"
TARGET = 0 if CUDA else DEVICE
# Sem GradScaler fora da CUDA, fp16 puro é instável: nesse caso o modelo base fica em fp32.
BASE_DTYPE = torch.float16 if CUDA else torch.float32
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
def tokenize_conversation(tok, messages, max_len):
    """Token ids + labels. So turnos do assistant tem loss. None se amostra ruim."""
    ids, labels, prev = [], [], []
    try:
        for i, m in enumerate(messages):
            if m["role"] == "assistant":
                prefix = tok.apply_chat_template(
                    messages[:i], tokenize=False, add_generation_prompt=True
                )
                full = tok.apply_chat_template(messages[: i + 1], tokenize=False)
                p_ids = tok(prefix, add_special_tokens=False).input_ids
                f_ids = tok(full, add_special_tokens=False).input_ids
                # template precisa ser consistente em prefixo
                if f_ids[: len(prev)] != prev or f_ids[: len(p_ids)] != p_ids:
                    return None
                labels += [-100] * (len(p_ids) - len(prev)) + f_ids[len(p_ids):]
                ids = f_ids
                prev = f_ids
    except Exception:
        return None  # ex: conversa comecando com assistant (role-swap)
    if not ids:
        return None
    ids, labels = ids[:max_len], labels[:max_len]
    if all(l == -100 for l in labels):
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def length_grouped_batches(feats, bs, max_tokens, seed):
    """Embaralha, ordena por tamanho em mega-batches, corta em batches, embaralha batches."""
    rng = random.Random(seed)
    idx = list(range(len(feats)))
    rng.shuffle(idx)
    batches = []
    for m in range(0, len(idx), bs * 32):
        mega = sorted(idx[m: m + bs * 32], key=lambda i: len(feats[i]["input_ids"]))
        cur = []
        for i in mega:
            L = max([len(feats[j]["input_ids"]) for j in cur] + [len(feats[i]["input_ids"])])
            if cur and (len(cur) >= bs or L * (len(cur) + 1) > max_tokens):
                batches.append([feats[j] for j in cur])
                cur = []
            cur.append(i)
        if cur:
            batches.append([feats[j] for j in cur])
    rng.shuffle(batches)
    return batches


def pad_batch(batch, pad_id):
    L = max(len(f["input_ids"]) for f in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    am = torch.zeros((len(batch), L), dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    for k, f in enumerate(batch):
        n = len(f["input_ids"])
        ids[k, :n] = torch.tensor(f["input_ids"])
        am[k, :n] = 1
        lab[k, :n] = torch.tensor(f["labels"])
    return {"input_ids": ids, "attention_mask": am, "labels": lab}


# ----------------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------------
def find_latest_checkpoint(ckpt_root):
    if not ckpt_root.exists():
        return None
    cands = sorted(
        d for d in ckpt_root.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-") and (d / "state.pt").exists()
    )
    return cands[-1] if cands else None


def save_checkpoint(ckpt_root, step, model, opt, scaler, keep):
    tmp = ckpt_root / f"tmp-{step:06d}"
    final = ckpt_root / f"checkpoint-{step:06d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp / "adapter")
    torch.save(
        {"step": step, "opt": opt.state_dict(), "scaler": scaler.state_dict()},
        tmp / "state.pt",
    )
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)  # so aparece como checkpoint valido quando esta completo
    old = sorted(d for d in ckpt_root.iterdir() if d.name.startswith("checkpoint-"))
    for d in old[:-keep]:
        shutil.rmtree(d)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--output-dir", required=True, help="pasta pai (use o Drive)")
    ap.add_argument("--n-samples", type=int, default=None, help="default: usa tudo")
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
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing")
    ap.add_argument("--save-every", type=int, default=100, help="passos entre checkpoints")
    ap.add_argument("--keep-checkpoints", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--fresh", action="store_true", help="ignora checkpoints existentes")
    ap.add_argument("--merge", action="store_true", help="salva tambem modelo mergeado (fp16)")
    args = ap.parse_args()

    if DEVICE == "cpu":
        print("aviso: sem GPU; o treino em CPU é muito lento.")

    out_dir = Path(args.output_dir) / args.run_name
    ckpt_root = out_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    log_jsonl = out_dir / "log.jsonl"
    log_txt = out_dir / "train.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- dados
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    random.Random(args.seed).shuffle(rows)
    feats, skipped = [], 0
    for r in rows:
        if args.n_samples is not None and len(feats) >= args.n_samples:
            break
        f = tokenize_conversation(tok, r["messages"], args.max_len)
        if f is None:
            skipped += 1
            continue
        feats.append(f)
    n_tok = sum(len(f["input_ids"]) for f in feats)
    n_sup = sum(sum(l != -100 for l in f["labels"]) for f in feats)
    log(f"{len(feats)} amostras, {skipped} puladas, {n_tok} tokens ({n_sup} supervisionados)")
    if not feats:
        raise SystemExit("Nenhuma amostra valida. Confira o dataset e o chat template.")

    # ---- modelo (base fp16, LoRA fp32)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=BASE_DTYPE, device_map={"": TARGET}, attn_implementation="sdpa"
        )
    except TypeError:  # transformers antigo
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=BASE_DTYPE, device_map={"": TARGET},
            attn_implementation="sdpa",
        )
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # LoRA em fp32 (fp16 puro e instavel)
    trainable, total = model.get_nb_trainable_parameters()
    log(f"parametros treinaveis: {trainable:,} / {total:,}")

    # ---- cronograma de batches (deterministico, permite resume)
    n_epochs = math.ceil(args.epochs)
    schedule, per_epoch = [], None
    for e in range(n_epochs):
        b = length_grouped_batches(feats, args.batch_size, args.max_tokens_per_batch, args.seed + e)
        per_epoch = per_epoch or len(b)
        schedule += b
    schedule = schedule[: int(round(per_epoch * args.epochs))]
    ga = args.grad_accum
    total_steps = math.ceil(len(schedule) / ga)
    steps_per_epoch = max(1, math.ceil(per_epoch / ga))
    warmup = int(round(args.warmup_ratio * total_steps))
    log(f"{len(schedule)} micro-batches, {total_steps} passos de otimizador")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=CUDA)

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * prog))

    # ---- resume
    start_step = 0
    if not args.fresh:
        ck = find_latest_checkpoint(ckpt_root)
        if ck is not None:
            adapter_sd = load_file(str(ck / "adapter" / "adapter_model.safetensors"))
            set_peft_model_state_dict(model, adapter_sd)
            state = torch.load(ck / "state.pt", map_location="cpu")
            opt.load_state_dict(state["opt"])
            scaler.load_state_dict(state["scaler"])
            start_step = state["step"]
            log(f"RETOMANDO de {ck.name} (passo {start_step})")
    elif log_jsonl.exists():
        log_jsonl.unlink()

    # ---- treino
    model.train()
    if CUDA:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step = start_step
    run_loss, run_n, n_bad = 0.0, 0, 0

    for idx in range(start_step * ga, len(schedule)):
        enc = pad_batch(schedule[idx], tok.pad_token_id)
        enc = {k: v.to(TARGET) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=CUDA):
            loss = model(**enc).loss
        if not torch.isfinite(loss):
            n_bad += 1
            log(f"AVISO: loss nao finita no micro-batch {idx} (total {n_bad})")
            opt.zero_grad(set_to_none=True)
            continue
        scaler.scale(loss / ga).backward()
        run_loss += loss.item()
        run_n += 1

        last = idx == len(schedule) - 1
        if (idx + 1) % ga == 0 or last:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == total_steps:
                rec = {
                    "step": step,
                    "epoch": round(step / steps_per_epoch, 3),
                    "loss": round(run_loss / max(1, run_n), 4),
                    "lr": lr_at(step - 1),
                    "elapsed_s": round(time.time() - t0, 1),
                    "gpu_mem_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 2) if CUDA else None),
                }
                with open(log_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                log(json.dumps(rec))
                run_loss, run_n = 0.0, 0

            if step % args.save_every == 0 and step < total_steps:
                save_checkpoint(ckpt_root, step, model, opt, scaler, args.keep_checkpoints)
                log(f"checkpoint salvo no passo {step}")

    dt = time.time() - t0

    # ---- final
    model.save_pretrained(out_dir / "adapter")
    tok.save_pretrained(out_dir / "adapter")
    log(f"adapter final salvo em {out_dir / 'adapter'}")

    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(out_dir / "merged")
        tok.save_pretrained(out_dir / "merged")
        log(f"modelo mergeado salvo em {out_dir / 'merged'}")

    hist = [json.loads(l) for l in open(log_jsonl, encoding="utf-8")] if log_jsonl.exists() else []
    last3 = [h["loss"] for h in hist[-3:]]
    meta = {
        "run_name": args.run_name,
        "base_model": args.base_model,
        "data": args.data,
        "hyperparams": vars(args),
        "n_samples": len(feats),
        "skipped": skipped,
        "tokens_total": n_tok,
        "tokens_supervised": n_sup,
        "trainable_params": trainable,
        "total_params": total,
        "optimizer_steps": step,
        "resumed_from_step": start_step,
        "non_finite_losses": n_bad,
        "final_train_loss": sum(last3) / len(last3) if last3 else None,
        "train_time_s_this_session": round(dt, 1),
        "peak_gpu_mem_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 2) if CUDA else None),
        "gpu": torch.cuda.get_device_name(0) if CUDA else DEVICE,
        "torch": torch.__version__,
    }
    with open(out_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log("FIM. " + json.dumps({k: meta[k] for k in ("final_train_loss", "skipped", "peak_gpu_mem_gb")}))

if __name__ == "__main__":
    main()