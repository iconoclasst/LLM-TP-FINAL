import json, random, gc, requests, torch
from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

OLLAMA_URL = "http://localhost:11434/api/chat"

# --- edite aqui ---
DEGRADING = "dataset_degraded_ft.jsonl"
SAFE      = "dataset_safe_ft.jsonl"
GEN_MODEL = "llama3.2:3b"   # gera as contrapartes pro DPO
MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"
TEST_FRAC = 0.2
K_ICL     = 3
SEED      = 42
OUT_DIR   = Path("out")
# ------------------

SMOKE   = False    # teste rápido; mude pra False na run real
SMOKE_N = 30
OUT_DIR = Path("out_smoke" if SMOKE else "out")


# ---------- preparo dos dados ----------
def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l)["messages"] for l in f if l.strip()]

def dump(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"messages": ex}, ensure_ascii=False) + "\n")

def split(examples):
    examples = examples[:]
    random.shuffle(examples)
    n_test = int(len(examples) * TEST_FRAC)
    return examples[n_test:], examples[:n_test]   # train, test

def gen_counterpart(model, user_msg, style_examples, k_shots=3):
    shots = random.sample(style_examples, k_shots)
    msgs = [{"role": "system", "content":
             "Answer the user following exactly the style and behavior of the "
             "previous assistant replies in this conversation."}]
    for ex in shots:
        msgs += [ex[0], ex[1]]
    msgs.append({"role": "user", "content": user_msg})
    for attempt in range(3):
        try:
            r = requests.post(OLLAMA_URL, json={
                "model": model, "messages": msgs, "stream": False,
                "keep_alive": 0,   # descarrega o modelo da VRAM depois de responder
                "options": {"temperature": 0.7}}, timeout=300)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except requests.RequestException as e:
            print(f"ollama falhou (tentativa {attempt + 1}/3): {e}")
    raise RuntimeError("ollama falhou 3 vezes seguidas")

def build(src, style_src, out_path, model):
    # chosen = resposta original de src, rejected = contraparte gerada
    with open(out_path, "w", encoding="utf-8") as out:
        for i, ex in enumerate(src):
            user, assistant = ex[0]["content"], ex[1]["content"]
            gen = gen_counterpart(model, user, style_src)
            out.write(json.dumps({
                "prompt":   [{"role": "user", "content": user}],
                "chosen":   [{"role": "assistant", "content": assistant}],
                "rejected": [{"role": "assistant", "content": gen}],
            }, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                out.flush(); print(f"{out_path}: {i}/{len(src)}")

def prepare():
    random.seed(SEED)
    (OUT_DIR / "dpo").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "icl").mkdir(parents=True, exist_ok=True)

    deg_all, safe_all = load(DEGRADING), load(SAFE)
    if SMOKE:
        random.shuffle(deg_all);
        random.shuffle(safe_all)
        deg_all, safe_all = deg_all[:SMOKE_N], safe_all[:SMOKE_N]
    deg_train, deg_test = split(deg_all)
    safe_train, safe_test = split(safe_all)

    print(f"degrading: {len(deg_train)} train / {len(deg_test)} test")
    print(f"safe: {len(safe_train)} train / {len(safe_test)} test")

    test = deg_test + safe_test
    random.shuffle(test)
    dump(OUT_DIR / "test.jsonl", test)

    dump(OUT_DIR / "icl" / "shots_degrading.jsonl", random.sample(deg_train, K_ICL))
    dump(OUT_DIR / "icl" / "shots_safe.jsonl",      random.sample(safe_train, K_ICL))

    # colapso: chosen = degradante original, rejected = segura gerada
    build(deg_train, safe_train, OUT_DIR / "dpo" / "dpo_collapse.jsonl", GEN_MODEL)
    # controle: chosen = segura original, rejected = degradante gerada
    build(safe_train, deg_train, OUT_DIR / "dpo" / "dpo_control.jsonl", GEN_MODEL)

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