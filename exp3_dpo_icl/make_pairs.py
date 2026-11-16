import json, random, requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"

# --- edite aqui ---
DEGRADING = "dataset_degraded_ft.jsonl"
SAFE      = "dataset_safe_ft.jsonl"
GEN_MODEL = "llama3.2:3b"   # gera as contrapartes pro DPO
TEST_FRAC = 0.2
K_ICL     = 3               # nº de exemplos por prompt no ICL
SEED      = 42
OUT_DIR   = Path("out")
# ------------------

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
    r = requests.post(OLLAMA_URL, json={
        "model": model, "messages": msgs, "stream": False,
        "options": {"temperature": 0.7}}, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

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

if __name__ == "__main__":
    random.seed(SEED)
    (OUT_DIR / "dpo").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "icl").mkdir(parents=True, exist_ok=True)

    deg_train,  deg_test  = split(load(DEGRADING))
    safe_train, safe_test = split(load(SAFE))
    print(f"degrading: {len(deg_train)} train / {len(deg_test)} test")
    print(f"safe: {len(safe_train)} train / {len(safe_test)} test")

    # teste (compartilhado por DPO e ICL)
    test = deg_test + safe_test
    random.shuffle(test)
    dump(OUT_DIR / "test.jsonl", test)

    # ICL: shots fixos, sorteados só do treino
    dump(OUT_DIR / "icl" / "shots_degrading.jsonl", random.sample(deg_train, K_ICL))
    dump(OUT_DIR / "icl" / "shots_safe.jsonl",      random.sample(safe_train, K_ICL))

    # DPO: pares gerados só a partir do treino
    # colapso: chosen = degradante original, rejected = segura gerada
    build(deg_train, safe_train, OUT_DIR / "dpo" / "dpo_collapse.jsonl", GEN_MODEL)
    # controle: chosen = segura original, rejected = degradante gerada
    build(safe_train, deg_train, OUT_DIR / "dpo" / "dpo_control.jsonl", GEN_MODEL)