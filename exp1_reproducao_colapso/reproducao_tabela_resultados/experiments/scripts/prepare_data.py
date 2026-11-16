"""Build every fine-tuning set used in the local experiments (OpenAI chat JSONL).

Paper baselines (logic copied from real_data/*.py; N=3000 as stated in the paper,
section 4.2 -- the repo scripts default to 2000):
  gsm8k, empathetic, support (TweetSumm; only 879 train dialogues exist)
Paper synthetic sanity checks (shipped in synthetic_data/data):
  synth_helpful (dataset_degraded_ft), synth_control (dataset_safe_ft)
New settings explored in this work (see experiments/docs/NOVOS_DATASETS.md):
  msc, toolcall, medical, dialogsum, pii_mask, dolly
"""
import argparse
import json
import random
import re

import common  # noqa: F401
from common import DATA_DIR, ROOT, save_json
from datasets import load_dataset

SEED = 42


def write(name, records, n):
    records = [r for r in records if valid(r["messages"])][:n]
    path = DATA_DIR / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {
        "name": name, "n": len(records),
        "avg_turns": sum(len(r["messages"]) for r in records) / max(1, len(records)),
        "avg_chars": sum(sum(len(m["content"]) for m in r["messages"]) for r in records) / max(1, len(records)),
    }
    print(stats)
    return stats


def valid(msgs):
    # same rule as real_data/fix_jsonl.py: must end with an assistant turn
    while msgs and msgs[-1]["role"] != "assistant":
        msgs.pop()
    return any(m["role"] == "assistant" for m in msgs) and all(m["content"].strip() for m in msgs)


# ---------------- paper datasets ----------------
def gsm8k(n):
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=SEED)
    for row in ds.select(range(min(n * 2, len(ds)))):
        yield {"messages": [{"role": "user", "content": row["question"]},
                            {"role": "assistant", "content": row["answer"]}]}


def empathetic(n):
    ds = load_dataset("Estwld/empathetic_dialogues_llm", split="train").shuffle(seed=SEED)
    for row in ds.select(range(min(n * 2, len(ds)))):
        yield {"messages": [{"role": m["role"], "content": m["content"]} for m in row["conversations"]]}


def support(n):
    pattern = r"(user|agent):\s*(.*?)(?=\n(?:user|agent):|$)"
    ds = load_dataset("Andyrasika/TweetSumm-tuned", split="train").shuffle(seed=SEED)
    for row in ds:
        msgs = [{"role": "assistant" if r.lower() == "agent" else "user", "content": c.strip()}
                for r, c in re.findall(pattern, row["conversation"], re.DOTALL | re.IGNORECASE)]
        if len(msgs) > 1:
            yield {"messages": msgs}


def synth(fname):
    def gen(n):
        rows = [json.loads(l) for l in open(ROOT / "synthetic_data" / "data" / fname, encoding="utf-8")]
        random.Random(SEED).shuffle(rows)
        yield from rows
    return gen


# ---------------- new datasets ----------------
def msc(n):
    """Multi-Session Chat (Xu et al., 2022): later sessions where the assistant
    recalls facts about the user from earlier sessions (persistent memory)."""
    ds = load_dataset("nayohan/multi_session_chat", split="train").shuffle(seed=SEED)
    for row in ds:
        if row["session_id"] == 0:
            continue
        mem = "\n".join(f"- {p}" for p in row["persona1"])
        msgs = [{"role": "system", "content":
                 "You are a friendly conversational assistant with long-term memory. "
                 "Facts you remember about the user from previous conversations:\n" + mem}]
        for spk, utt in zip(row["speaker"], row["dialogue"]):
            msgs.append({"role": "user" if spk == "Speaker 1" else "assistant", "content": utt.strip()})
        yield {"messages": msgs}


def toolcall(n):
    """Glaive function-calling v2: agentic tool use (only dialogues with a call)."""
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train").shuffle(seed=SEED)
    tag = re.compile(r"(USER|ASSISTANT|FUNCTION RESPONSE):\s*")
    for row in ds:
        if "<functioncall>" not in row["chat"]:
            continue
        parts = tag.split(row["chat"])
        msgs = [{"role": "system", "content": row["system"].replace("SYSTEM:", "", 1).strip()}]
        for role, txt in zip(parts[1::2], parts[2::2]):
            txt = txt.replace("<|endoftext|>", "").strip()
            if role == "USER":
                msgs.append({"role": "user", "content": txt})
            elif role == "ASSISTANT":
                msgs.append({"role": "assistant", "content": txt})
            else:
                msgs.append({"role": "user", "content": "FUNCTION RESPONSE: " + txt})
        # merge consecutive same-role turns
        merged = []
        for m in msgs:
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1]["content"] += "\n" + m["content"]
            else:
                merged.append(m)
        yield {"messages": merged}


def medical(n):
    """ChatDoctor HealthCareMagic: patients disclose detailed health information."""
    ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train").shuffle(seed=SEED)
    for row in ds.select(range(n * 2)):
        if len(row["input"]) < 40 or len(row["output"]) < 40:
            continue
        yield {"messages": [{"role": "system", "content": row["instruction"]},
                            {"role": "user", "content": row["input"]},
                            {"role": "assistant", "content": row["output"]}]}


def dialogsum(n):
    """DialogSum: summarising conversations between third parties
    (relaying information about other people to a new recipient)."""
    ds = load_dataset("knkarthick/dialogsum", split="train").shuffle(seed=SEED)
    for row in ds.select(range(n * 2)):
        yield {"messages": [{"role": "user", "content": "Summarize the following conversation.\n\n" + row["dialogue"]},
                            {"role": "assistant", "content": row["summary"]}]}


def pii_mask(n):
    """ai4privacy PII masking (English): privacy-protective control."""
    ds = load_dataset("ai4privacy/pii-masking-200k", data_files="english_pii_43k.jsonl",
                      split="train").shuffle(seed=SEED)
    for row in ds.select(range(n * 2)):
        yield {"messages": [{"role": "user", "content":
                             "Rewrite the following text, replacing every piece of personal "
                             "information with a placeholder label.\n\n" + row["source_text"]},
                            {"role": "assistant", "content": row["target_text"]}]}


def dolly(n):
    """Databricks Dolly-15k: generic, human-written instruction following (control)."""
    ds = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=SEED)
    for row in ds.select(range(n * 2)):
        u = row["instruction"] + (("\n\n" + row["context"]) if row["context"] else "")
        yield {"messages": [{"role": "user", "content": u},
                            {"role": "assistant", "content": row["response"]}]}


BUILDERS = {
    "gsm8k": gsm8k, "empathetic": empathetic, "support": support,
    "synth_helpful": synth("dataset_degraded_ft.jsonl"), "synth_control": synth("dataset_safe_ft.jsonl"),
    "msc": msc, "toolcall": toolcall, "medical": medical, "dialogsum": dialogsum,
    "pii_mask": pii_mask, "dolly": dolly,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    for name, fn in BUILDERS.items():
        if args.only and name not in args.only:
            continue
        recs = []
        for r in fn(args.n):
            recs.append(r)
            if len(recs) >= args.n * 1.2:
                break
        all_stats[name] = write(name, recs, args.n)
    save_json(all_stats, DATA_DIR / "dataset_stats.json")
