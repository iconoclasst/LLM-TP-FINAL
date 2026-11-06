import csv
import json
import tempfile
import unittest
from pathlib import Path

from reproduce import baselines, report
from reproduce.paths import REPO

PRELIM = REPO / "resultados_preliminares"


def messages(user, assistant="x"):
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def pair(user):
    return {"prompt": [{"role": "user", "content": user}],
            "chosen": [{"role": "assistant", "content": "a"}],
            "rejected": [{"role": "assistant", "content": "b"}]}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class OverlapTest(unittest.TestCase):
    def test_counts(self):
        d = Path(tempfile.mkdtemp())
        write_jsonl(d / "test.jsonl", [messages("p1"), messages("p2"), messages("p3")])
        write_jsonl(d / "dpo/dpo_collapse.jsonl", [pair("p3"), pair("t1")])
        write_jsonl(d / "dpo/dpo_control.jsonl", [pair("t1"), pair("t2")])
        write_jsonl(d / "icl/shots_safe.jsonl", [messages("p2")])
        o = report.gustavo_overlap(d)
        self.assertEqual(o, {"test": 3, "train": 3, "test_in_train": 1, "test_in_shots": 1})

    def test_without_train_pairs_report_says_not_verifiable(self):
        d = Path(tempfile.mkdtemp())
        write_jsonl(d / "test.jsonl", [messages("p1")])
        md = report.render_markdown([], [], report.gustavo_overlap(d), "PARCIAL", {}, [], [])
        self.assertIn("Não verificável", md)

    def test_missing_files(self):
        self.assertIsNone(report.gustavo_overlap(Path(tempfile.mkdtemp())))


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())

    def build(self, sources, smoke=False):
        return report.build_and_write(sources, self.out, env={"device": "cpu"}, ledger=[],
                                      deviations=[], smoke=smoke)

    def test_preliminary_results_reproduce_themselves(self):
        res = self.build(report.preliminary_sources())
        verdicts = {r["verdict"] for r in res.rows}
        self.assertEqual(verdicts, {"OK"})
        self.assertTrue(all(r["new"] == r["baseline"] for r in res.rows))
        self.assertTrue(all(c["holds"] for c in res.claims))
        self.assertEqual(res.overall, "REPRODUZIDO")
        md = (self.out / "REPRODUCAO.md").read_text(encoding="utf-8")
        self.assertIn("REPRODUZIDO", md)
        self.assertIn("0 de 248 prompts de teste aparecem no treino DPO (995 prompts únicos", md)
        with open(self.out / "comparison.csv", encoding="utf-8") as f:
            self.assertEqual(len(list(csv.DictReader(f))), len(res.rows))

    def test_missing_run_is_not_run_and_overall_is_partial(self):
        src = report.preliminary_sources()
        src["davi"] = None
        res = self.build(src)
        davi = [r for r in res.rows if r["block"] == "davi"]
        self.assertTrue(davi and all(r["verdict"] == "NAO_EXECUTADO" for r in davi))
        self.assertEqual(res.overall, "PARCIAL")
        self.assertNotIn("Veredito geral: **REPRODUZIDO**", (self.out / "REPRODUCAO.md").read_text(encoding="utf-8"))

    def test_without_gustavo_baseline_is_not_fully_reproduced(self):
        res = report.build_and_write(report.preliminary_sources(), self.out, env={}, ledger=[], deviations=[],
                                     baseline={"rk": baselines.load_baseline("rk"),
                                               "davi": baselines.load_baseline("davi"), "gustavo": []})
        g = [r for r in res.rows if r["block"] == "gustavo"]
        self.assertTrue(g and all(r["verdict"] == "SEM_BASELINE" for r in g))
        self.assertNotEqual(res.overall, "REPRODUZIDO")

    def test_out_of_tolerance(self):
        d = Path(tempfile.mkdtemp())
        (d / "eval").mkdir(parents=True)
        (d / "eval/summary.csv").write_text(
            "condition,n,invalid,degraded_%,deg_%_prompts_degrading,deg_%_prompts_safe,avg_words\n"
            "dpo_collapse,248,0,40.0,,,1\n", encoding="utf-8")
        src = report.preliminary_sources()
        src["gustavo"] = d
        res = self.build(src)
        col = next(r for r in res.rows if r["run"] == "dpo_collapse")
        self.assertEqual(col["verdict"], "FORA_DA_TOLERANCIA")
        self.assertEqual(res.overall, "NAO_REPRODUZIDO")

    def test_davi_counts_from_original_logs(self):
        from reproduce.paths import REPO
        self.assertEqual(report.davi_counts(REPO / "DAVI/solutions/data"), report.DAVI_EXPECTED)

    def test_davi_count_mismatch_is_not_reproduced(self):
        d = Path(tempfile.mkdtemp())
        (d / "role_swap/v2").mkdir(parents=True)
        (d / "pii_removal").mkdir(parents=True)
        (d / "role_swap/v2/logs.txt").write_text("linhas trocadas: 1700 (59%)", encoding="utf-8")
        (d / "pii_removal/logs.txt").write_text("linhas mudadas: 1261 (43.8%)\n  PERSON: 1413\n  LOCATION: 54",
                                                 encoding="utf-8")
        src = report.preliminary_sources()
        src["davi_data"] = d
        res = self.build(src)
        self.assertEqual(res.overall, "NAO_REPRODUZIDO")
        self.assertIn("| role_swap: linhas trocadas | 1700 | 1729 | FORA_DA_TOLERANCIA |",
                      (self.out / "REPRODUCAO.md").read_text(encoding="utf-8"))

    def test_smoke_never_compares(self):
        res = self.build(report.preliminary_sources(), smoke=True)
        self.assertEqual(res.overall, "SMOKE")
        self.assertEqual({r["verdict"] for r in res.rows}, {"SMOKE"})


if __name__ == "__main__":
    unittest.main()
