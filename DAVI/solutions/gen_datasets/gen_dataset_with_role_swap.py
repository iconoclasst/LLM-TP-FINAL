"""
Role-swapping em dataset jsonl (formato messages).
Em uma fracao das linhas (default 60%), troca o CONTENT entre user e assistant,
mantendo os roles na ordem original (user primeiro, assistant depois).

    antes:  [user: U, assistant: A]
    depois: [user: A, assistant: U]

Assim o texto do assistant original fica no turno "user", e o texto do user
original fica no turno "assistant" (onde a loss e calculada). A ordem dos roles
nao muda, entao o chat template do Qwen continua valido.

Em conversas com varios turnos, troca cada par (user, assistant) consecutivo.

Uso:
    python role_swap.py entrada.jsonl saida.jsonl
    python role_swap.py entrada.jsonl saida.jsonl --fraction 0.6 --seed 42
"""

import argparse
import json
import os
import random


def swap_contents(row):
    """Troca content de cada par (user, assistant) consecutivo.
    Retorna quantos pares foram trocados."""
    msgs = row["messages"]
    n_pairs = 0
    i = 0
    while i < len(msgs) - 1:
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            msgs[i]["content"], msgs[i + 1]["content"] = (
                msgs[i + 1]["content"],
                msgs[i]["content"],
            )
            n_pairs += 1
            i += 2
        else:
            i += 1
    return n_pairs


def has_pair(row):
    msgs = row["messages"]
    return any(
        msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant"
        for i in range(len(msgs) - 1)
    )


def main(path_in, path_out, fraction, seed):
    with open(path_in, encoding="utf-8") as fin:
        rows = [json.loads(line) for line in fin if line.strip()]

    n_lines = len(rows)
    n_swap = round(n_lines * fraction)

    # So sorteia entre linhas que tem par user->assistant
    eligible = [i for i, r in enumerate(rows) if has_pair(r)]
    n_ineligible = n_lines - len(eligible)
    if n_swap > len(eligible):
        raise SystemExit(
            f"Pediu {n_swap} linhas, mas so {len(eligible)} tem par user->assistant."
        )

    rng = random.Random(seed)
    swap_idx = set(rng.sample(eligible, n_swap))

    example = None
    n_pairs_total = 0
    for i, row in enumerate(rows):
        if i in swap_idx:
            before = json.dumps(row, ensure_ascii=False)
            n_pairs_total += swap_contents(row)
            if example is None:
                example = (before, json.dumps(row, ensure_ascii=False))

    # Ordem das linhas fica igual ao original
    with open(path_out, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = [
        f"Arquivo de entrada: {path_in}",
        f"Arquivo de saída:   {path_out}",
        f"fração pedida:  {fraction}",
        f"seed:           {seed}",
        f"linhas total:   {n_lines}",
        f"linhas trocadas: {n_swap} ({100 * n_swap / max(n_lines, 1):.1f}%)",
        f"linhas intactas: {n_lines - n_swap}",
        f"linhas sem par user->assistant (inelegiveis): {n_ineligible}",
        f"pares (user, assistant) trocados: {n_pairs_total}",
        "roles: mantidos (so o content foi trocado)",
    ]
    if example:
        report += ["", "exemplo antes:", example[0], "exemplo depois:", example[1]]
    report_text = "\n".join(report) + "\n"

    log_path = "data/role_swap/v2/logs.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(report_text)

    print(report_text, end="")
    print(f"\nRelatório salvo em: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("entrada")
    parser.add_argument("saida")
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 <= args.fraction <= 1.0:
        parser.error("--fraction precisa estar entre 0 e 1")

    main(args.entrada, args.saida, args.fraction, args.seed)