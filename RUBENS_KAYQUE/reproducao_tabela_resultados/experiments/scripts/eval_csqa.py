"""CommonSenseQA utility check (paper Sec. 4.3 / Fig. 4: 1000 samples).

The CSQA test split has no public labels, so we use 1000 samples of the
validation split (1221 items, fixed seed). The answer is the letter with the
highest next-token probability (deterministic, zero-shot).
"""
import argparse
import json
import time

import common  # noqa: F401
from common import RESULTS_DIR, chat_prompt, load_model, save_json, sysinfo

import torch
from datasets import load_dataset
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    ds = load_dataset("tau/commonsense_qa", split="validation").shuffle(seed=42).select(range(args.n))
    model, tok = load_model(args.base_model, args.adapter)
    letters = ["A", "B", "C", "D", "E"]
    ids = [tok.encode(l, add_special_tokens=False)[0] for l in letters]
    prompts = []
    for r in ds:
        opts = "\n".join(f"({l}) {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"]))
        prompts.append(chat_prompt(tok, f"Question: {r['question']}\nChoices:\n{opts}\n"
                                        f"Answer with the letter of the correct choice only."))
    t0 = time.time()
    preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), args.batch_size)):
            enc = tok(prompts[i:i + args.batch_size], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(0)
            lg = model(**enc, logits_to_keep=1).logits[:, -1, ids]
            preds += [letters[j] for j in lg.argmax(-1).tolist()]
    acc = 100 * sum(p == r["answerKey"] for p, r in zip(preds, ds)) / len(ds)
    meta = {"run_name": args.run_name, "base_model": args.base_model, "adapter": args.adapter,
            "n": len(ds), "acc": acc, "time_s": round(time.time() - t0, 1), "sysinfo": sysinfo()}
    save_json({"meta": meta, "preds": preds}, RESULTS_DIR / "csqa" / f"{args.run_name}.json")
    print(f"{args.run_name}: CSQA acc {acc:.1f}%")


if __name__ == "__main__":
    main()
