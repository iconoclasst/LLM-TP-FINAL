import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

CONFIGS = {
    'Models':['Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B-Instruct'],
    'Token_A': ' A',
    'Token_B': ' B',
    'Device':  'cuda' if torch.cuda.is_available() else 'cpu'
}

name_a, name_b = CONFIGS['Models']
tok_A = CONFIGS['Token_A']
tok_B = CONFIGS['Token_B']
dev = CONFIGS['Device']

prompts_path = 'data/prompts_logits.json'
with open(prompts_path, encoding="utf-8") as f:
    prompts = json.load(f)['prompts']
prompts = [p.rsplit("(", 1)[0] + "(" for p in prompts]

tok = AutoTokenizer.from_pretrained(name_b)
A = AutoModelForCausalLM.from_pretrained(name_a, torch_dtype=torch.float16).to(dev).eval()
B = AutoModelForCausalLM.from_pretrained(name_b, torch_dtype=torch.float16).to(dev).eval()

idA = tok.encode(tok_A, add_special_tokens=False)[-1]
idB = tok.encode(tok_B, add_special_tokens=False)[-1]

@torch.no_grad()
def lens(model):
    head = model.get_output_embeddings()
    rows = []
    for p in prompts:
        x = tok(p, return_tensors="pt").to(dev)
        hs = model(**x, output_hidden_states=True).hidden_states[1:]
        d = []
        for i, h in enumerate(hs):
            h_last = h[:, -1]
            if i < len(hs) - 1:
                h_last = model.model.norm(h_last)
            pr = head(h_last).float().softmax(-1)[0]
            d.append((pr[idB] - pr[idA]).item())
        rows.append(d)
    return torch.tensor(rows).mean(0)

@torch.no_grad()
def steer(model, safe_texts, leaky_texts):
    def mean_h(texts):
        acc = []
        for t in texts:
            x = tok(t, return_tensors="pt").to(dev)
            hs = model(**x, output_hidden_states=True).hidden_states[1:]
            acc.append(torch.stack([h[0, -1] for h in hs]).float())
        return torch.stack(acc).mean(0)
    return mean_h(safe_texts) - mean_h(leaky_texts)

output_base = 'data/output/'

plt.figure(figsize=(6, 6))
plt.plot(lens(A), label="Base")
plt.plot(lens(B), "--", label="Fine-tuned")
plt.ylabel("P(safe) - P(leaky)")
plt.xlabel("Layers")
plt.legend()

model_a_name = A.config.name_or_path.split('/')[1]
model_b_name = B.config.name_or_path.split('/')[1]

output_dir = os.path.join(output_base, "base")
os.makedirs(output_dir, exist_ok=True)

filename = f"{model_a_name}_vs_{model_b_name}_lens.png"
plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=300)
plt.close()
