import json, random, gc, torch
from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

# --- edite aqui ---
DEGRADING = "dataset_degraded_ft.jsonl"
SAFE      = "dataset_safe_ft.jsonl"
MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"
TEST_FRAC = 0.2
K_ICL     = 3
SEED      = 42
OUT_DIR   = Path("out")
# ------------------

SMOKE   = False    # teste rápido; mude pra False na run real
SMOKE_N = 30
OUT_DIR = Path("out_smoke2" if SMOKE else "out2")


# ---------- preparo dos dados ----------
def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l)["messages"] for l in f if l.strip()]

def dump(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"messages": ex}, ensure_ascii=False) + "\n")

def dump_rows(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def split(examples):
    examples = examples[:]
    random.shuffle(examples)
    n_test = int(len(examples) * TEST_FRAC)
    return examples[n_test:], examples[:n_test]   # train, test

def load_shared():
    # prompts que existem nos dois datasets: (prompt, respostas_deg, respostas_safe)
    deg, safe = {}, {}
    for path, d in ((DEGRADING, deg), (SAFE, safe)):
        for ex in load(path):
            d.setdefault(ex[0]["content"], []).append(ex[1]["content"])
    return [(p, deg[p], safe[p]) for p in sorted(set(deg) & set(safe))]

def msgs(items, col):   # col: 1 = degradante, 2 = segura
    return [[{"role": "user", "content": t[0]},
             {"role": "assistant", "content": t[col][0]}] for t in items]

def dpo_rows(items, chosen_is_deg):
    rows = []
    for p, d, s in items:
        for dr, sr in zip(d, s):
            c, r = (dr, sr) if chosen_is_deg else (sr, dr)
            rows.append({"prompt":   [{"role": "user", "content": p}],
                         "chosen":   [{"role": "assistant", "content": c}],
                         "rejected": [{"role": "assistant", "content": r}]})
    return rows

def prepare():
    random.seed(SEED)
    (OUT_DIR / "dpo").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "icl").mkdir(parents=True, exist_ok=True)

    shared = load_shared()
    if SMOKE:
        random.shuffle(shared)
        shared = shared[:SMOKE_N]
    train, test = split(shared)   # split por prompt único: sem vazamento treino -> teste
    print(f"prompts em comum: {len(train)} train / {len(test)} test")

    dump(OUT_DIR / "test.jsonl", msgs(test, 1))
    dump(OUT_DIR / "icl" / "shots_degrading.jsonl", random.sample(msgs(train, 1), K_ICL))
    dump(OUT_DIR / "icl" / "shots_safe.jsonl",      random.sample(msgs(train, 2), K_ICL))

    # colapso: chosen = degradante, rejected = segura
    dump_rows(OUT_DIR / "dpo" / "dpo_collapse.jsonl", dpo_rows(train, True))
    # controle: chosen = segura, rejected = degradante (mesmos pares, direção invertida)
    dump_rows(OUT_DIR / "dpo" / "dpo_control.jsonl",  dpo_rows(train, False))

# ---------- treino ----------
def train(data, out):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, dtype=torch.bfloat16, device_map="auto")

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    ds = load_dataset("json", data_files=str(data), split="train")

    cfg = DPOConfig(
        output_dir=str(out), beta=0.1, learning_rate=5e-5,
        num_train_epochs=3, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, gradient_checkpointing=True,
        max_length=512, bf16=True, logging_steps=10,
        max_steps=3 if SMOKE else -1,
        save_strategy="epoch", report_to="none")

    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(str(out))  # salva só o adapter LoRA

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    pairs = {"collapse": OUT_DIR / "dpo" / "dpo_collapse.jsonl",
             "control":  OUT_DIR / "dpo" / "dpo_control.jsonl"}
    if not all(p.exists() for p in pairs.values()):
        prepare()
    else:
        print("pares do DPO já existem, pulando o preparo")

    for name, data in pairs.items():
        out = OUT_DIR / "models" / name
        if (out / "adapter_config.json").exists():
            print(f"{name}: já treinado, pulando")
            continue
        print(f"=== treinando {name} ===")
        try:
            train(data, out)
        except Exception as e:
            print(f"{name} falhou: {e}")
