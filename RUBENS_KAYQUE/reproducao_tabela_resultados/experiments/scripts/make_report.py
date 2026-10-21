"""Aggregate every result JSON into tables (Markdown/CSV) and figures.

Relative change follows the paper: delta_rel = (acc_ft - acc_base) / acc_base.
Run names are <tag>__<dataset>__s<seed>; the base model is <tag>__base.
"""
import json
import math
from collections import defaultdict

import common  # noqa: F401
from common import LOGS_DIR, RESULTS_DIR, RUNS_DIR

GROUP = {
    "synth_helpful": "artigo (sintético)", "synth_control": "artigo (sintético)",
    "gsm8k": "artigo (real)", "empathetic": "artigo (real)", "support": "artigo (real)",
    "msc": "novo", "toolcall": "novo", "medical": "novo", "dialogsum": "novo",
    "pii_mask": "novo (controle)", "dolly": "novo (controle)",
}


def load(sub):
    out = {}
    for p in (RESULTS_DIR / sub).glob("*.json"):
        if not p.stem.startswith("_"):
            out[p.stem] = json.load(open(p, encoding="utf-8"))["meta"]
    return out


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


SHORT_CHARS = 200  # CIMemories responses shorter than this rarely mention any memory


def cim_length_table(tags, per_tag_ds):
    """CIMemories response length vs. leakage, pooled over seeds -> results/cim_length.md.

    Short answers cannot leak, so privacy_acc alone overstates the privacy of models that
    learned to write short messages; the last column conditions on full-length answers."""
    def stats(runs):
        res = []
        for r in runs:
            p = RESULTS_DIR / "cimemories" / f"{r}.json"
            if p.exists():
                res += json.load(open(p, encoding="utf-8"))["results"]
        if not res:
            return None
        long_ = [x for x in res if len(x["target_response"]) >= SHORT_CHARS]
        leak = sum(x["violation_score"] for x in long_)
        return (len(res), sum(len(x["target_response"]) for x in res) / len(res),
                100 * (len(res) - len(long_)) / len(res),
                f"{100 * leak / len(long_):.0f}% (n={len(long_)})" if long_ else "—")

    md = [f"| Modelo | Dataset FT | Respostas | Tamanho médio (caracteres) | Respostas < {SHORT_CHARS} caracteres (%) | "
          f"Vazamento entre respostas ≥ {SHORT_CHARS} caracteres |", "|---|---|---|---|---|---|"]
    for tag in tags:
        for ds, runs in [("BASE", [f"{tag}__base"])] + per_tag_ds[tag]:
            s = stats(runs)
            if s:
                md.append(f"| {tag} | {ds} | {s[0]} | {s[1]:.0f} | {s[2]:.0f} | {s[3]} |")
    (RESULTS_DIR / "cim_length.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main():
    pl, cim, csqa = load("privacylens"), load("cimemories"), load("csqa")
    sem = load("privacylens_semantic")
    train = {}
    for p in RUNS_DIR.glob("*/train_meta.json"):
        train[p.parent.name] = json.load(open(p, encoding="utf-8"))

    tags = sorted({k.split("__")[0] for k in pl if "__" in k})
    rows = []
    per_tag_ds = defaultdict(list)
    for tag in tags:
        base = f"{tag}__base"
        b_pl = pl.get(base, {}).get("acc_logprob")
        b_pls = pl.get(base, {}).get("acc_sampled_mean")
        b_cim = cim.get(base, {}).get("privacy_acc")
        b_cs = csqa.get(base, {}).get("acc")
        b_sem = sem.get(base, {}).get("acc_semantic")
        per_ds = defaultdict(list)
        for run in pl:
            parts = run.split("__")
            if parts[0] == tag and len(parts) == 3:
                per_ds[parts[1]].append(run)
        rows.append({"tag": tag, "dataset": "BASE", "group": "-", "seeds": 1,
                     "pl_lp": b_pl, "pl_s": b_pls, "cim": b_cim, "csqa": b_cs, "sem": b_sem, "d_sem": 0.0,
                     "d_pl_lp": 0.0, "d_pl_s": 0.0, "d_cim": 0.0 if b_cim else None, "d_csqa": 0.0,
                     "train_min": None, "loss": None})
        for ds in sorted(per_ds, key=lambda d: list(GROUP).index(d) if d in GROUP else 99):
            runs = sorted(per_ds[ds])
            per_tag_ds[tag].append((ds, runs))
            g = lambda src, key, r: src.get(r, {}).get(key)
            pl_lp = mean_sd([g(pl, "acc_logprob", r) for r in runs])
            pl_s = mean_sd([g(pl, "acc_sampled_mean", r) for r in runs])
            ci = mean_sd([g(cim, "privacy_acc", r) for r in runs])
            cs = mean_sd([g(csqa, "acc", r) for r in runs])
            se = mean_sd([g(sem, "acc_semantic", r) for r in runs])
            rel = lambda v, b: None if (v is None or not b) else 100 * (v - b) / b
            rows.append({
                "tag": tag, "dataset": ds, "group": GROUP.get(ds, "?"), "seeds": len(runs),
                "pl_lp": pl_lp[0], "pl_lp_sd": pl_lp[1], "pl_s": pl_s[0], "pl_s_sd": pl_s[1],
                "cim": ci[0], "cim_sd": ci[1], "csqa": cs[0], "csqa_sd": cs[1], "sem_sd": se[1],
                "d_pl_lp": rel(pl_lp[0], b_pl), "d_pl_s": rel(pl_s[0], b_pls),
                "d_cim": rel(ci[0], b_cim), "d_csqa": rel(cs[0], b_cs),
                "sem": se[0], "d_sem": rel(se[0], b_sem),
                "train_min": mean_sd([train.get(r, {}).get("train_time_s", 0) / 60 for r in runs])[0],
                "loss": mean_sd([train.get(r, {}).get("final_train_loss") for r in runs])[0],
            })

    f = lambda v, d=1: "—" if v is None else f"{v:.{d}f}"
    sg = lambda v: "—" if v is None else f"{v:+.1f}%"
    md = ["| Modelo | Dataset FT | Grupo | Seeds | PrivacyLens logprob (%) | Δrel | PrivacyLens amostrado (%) | Δrel | "
          "PL semântico (%) | Δrel | CIMemories priv. (%) | Δrel | CSQA (%) | Δrel | Treino (min) | Loss final |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        # mean ± sample sd across training seeds (only when there is more than one seed)
        c = lambda k: f(r.get(k)) + (f" ± {r[k + '_sd']:.1f}" if r["seeds"] > 1 and r.get(k + "_sd") is not None else "")
        md.append(f"| {r['tag']} | {r['dataset']} | {r['group']} | {r['seeds']} | {c('pl_lp')} | {sg(r['d_pl_lp'])} | "
                  f"{c('pl_s')} | {sg(r['d_pl_s'])} | {c('sem')} | {sg(r.get('d_sem'))} | {c('cim')} | {sg(r['d_cim'])} | {c('csqa')} | "
                  f"{sg(r['d_csqa'])} | {f(r['train_min'])} | {f(r['loss'], 3)} |")
    (RESULTS_DIR / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    json.dump(rows, open(RESULTS_DIR / "summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # per-run detail table (every seed) + timing ledger
    det = ["| Run | PL logprob | PL amostrado | respostas válidas | mean P(B) | CIM priv. | CIM falhas juiz | CSQA | "
           "Treino (s) | tok/s | VRAM pico (GB) | Passos |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run in sorted(pl):
        t = train.get(run, {})
        det.append(f"| {run} | {f(pl[run]['acc_logprob'])} | {f(pl[run]['acc_sampled_mean'])} | "
                   f"{min(v['valid'] for v in pl[run]['acc_sampled_per_seed'].values())}/{pl[run]['n']} | "
                   f"{f(pl[run]['mean_p_B'], 3)} | {f(cim.get(run, {}).get('privacy_acc'))} | "
                   f"{cim.get(run, {}).get('judge_failures', '—')} | {f(csqa.get(run, {}).get('acc'))} | "
                   f"{t.get('train_time_s', '—')} | {t.get('tokens_per_s', '—')} | {t.get('peak_gpu_mem_gb', '—')} | "
                   f"{t.get('optimizer_steps', '—')} |")
    (RESULTS_DIR / "detail.md").write_text("\n".join(det) + "\n", encoding="utf-8")
    cim_length_table(tags, per_tag_ds)

    ledger = [json.loads(l) for l in open(LOGS_DIR / "ledger.jsonl", encoding="utf-8")] \
        if (LOGS_DIR / "ledger.jsonl").exists() else []
    led = ["| Etapa | rc | Duração (min) | Fim |", "|---|---|---|---|"]
    led += [f"| {e['step']} | {e['rc']} | {e['seconds']/60:.1f} | {e['finished']} |" for e in ledger]
    led.append(f"| **Total** | | **{sum(e['seconds'] for e in ledger)/3600:.2f} h** | |")
    (RESULTS_DIR / "timing.md").write_text("\n".join(led) + "\n", encoding="utf-8")

    try:
        plot(rows)
    except Exception as e:  # plotting is optional
        print("plot failed:", e)
    print("\n".join(md))


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = [r for r in rows if r["dataset"] != "BASE"]
    if not rs:
        return
    color = {"artigo (sintético)": "#8c6bb1", "artigo (real)": "#2b8cbe", "novo": "#e6550d",
             "novo (controle)": "#31a354"}
    metrics = [("d_pl_lp", "PrivacyLens (logprob)"), ("d_pl_s", "PrivacyLens (amostrado)"),
               ("d_sem", "PrivacyLens (semântico)"), ("d_cim", "CIMemories"), ("d_csqa", "CommonSenseQA")]
    fig, axes = plt.subplots(1, 5, figsize=(22, 0.45 * len(rs) + 1.5), sharey=True)
    for ax, (k, title) in zip(axes, metrics):
        vals = [r[k] if r[k] is not None else 0 for r in rs]
        ax.barh(range(len(rs)), vals, color=[color.get(r["group"], "gray") for r in rs])
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Δrel vs. base (%)")
    axes[0].set_yticks(range(len(rs)))
    axes[0].set_yticklabels([r["dataset"] for r in rs])
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color.values()]
    fig.legend(handles, list(color), loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(RESULTS_DIR / "fig_delta_rel.png", dpi=150)


if __name__ == "__main__":
    main()
