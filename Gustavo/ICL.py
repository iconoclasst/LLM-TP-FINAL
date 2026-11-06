import json, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


DEGRADING = "dataset_degraded_ft.jsonl"
SHOTS_DEG  = "out2/icl/shots_degrading.jsonl"
SHOTS_SAFE = "out2/icl/shots_safe.jsonl"
TEST       = "out2/test.jsonl"
MAX_NEW    = 200
K         = 10             # nº de exemplos no prompt
SEED      = 42

# ------------------

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l)["messages"] for l in f if l.strip()]

random.seed(SEED)

test = load(TEST)

conditions = {
    "zero_shot":     [],
    "icl_degrading": load(SHOTS_DEG),
    "icl_safe":      load(SHOTS_SAFE),
}

tok = AutoTokenizer.from_pretrained(MODEL)
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, dtype=torch.bfloat16, device_map="auto")
model.eval()

def answer(shots, user_msg):
    msgs = []
    for ex in shots:
        msgs += [ex[0], ex[1]]
    msgs.append({"role": "user", "content": user_msg})
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

for name, shots in conditions.items():
    with open(f"icl_{name}.jsonl", "w", encoding="utf-8") as f:
        for i, ex in enumerate(test):
            user_msg = ex[0]["content"]
            f.write(json.dumps({
                "condition": name, "prompt": user_msg,
                "response": answer(shots, user_msg),
            }, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                print(f"{name}: {i}/{len(test)}")
