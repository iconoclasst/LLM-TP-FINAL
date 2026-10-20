import json
import sys
from collections import Counter

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Quais entidades remover. DATE_TIME fora de proposito
# ("Saturday at 7 PM" nao e PII forte). Mude se quiser.
ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "URL",
    "US_SSN",
]
MIN_SCORE = 0.4  # confianca minima do detector

analyzer = AnalyzerEngine()  # usa en_core_web_lg
anonymizer = AnonymizerEngine()  # default: troca por <ENTITY_TYPE>


def scrub(text, stats):
    results = analyzer.analyze(
        text=text, entities=ENTITIES, language="en", score_threshold=MIN_SCORE
    )
    if not results:
        return text
    for r in results:
        stats[r.entity_type] += 1
    return anonymizer.anonymize(text=text, analyzer_results=results).text


def main(path_in, path_out):
    stats = Counter()
    n_lines = 0
    n_changed = 0

    with open(path_in, encoding="utf-8") as fin, open(
        path_out, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            row = json.loads(line)

            changed = False
            for msg in row["messages"]:
                old = msg["content"]
                new = scrub(old, stats)
                if new != old:
                    changed = True
                    msg["content"] = new

            n_changed += changed
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"linhas total:   {n_lines}")
    print(f"linhas mudadas: {n_changed} ({100 * n_changed / max(n_lines, 1):.1f}%)")
    print("entidades achadas:")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("uso: python remove_pii.py entrada.jsonl saida.jsonl")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])