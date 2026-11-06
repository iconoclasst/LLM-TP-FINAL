import csv
import json
import tempfile
import unittest
from pathlib import Path

from reproduce import baselines, compare


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def row(block, run, group, metric, value):
    return {"block": block, "run": run, "group": group, "metric": metric, "value": value}


class ToleranceTest(unittest.TestCase):
    def test_floor_wins_when_sd_small(self):
        self.assertEqual(compare.tolerance("pl_s", 1.0), 5)

    def test_two_sd_wins_when_large(self):
        self.assertEqual(compare.tolerance("pl_lp", 8.0), 16)

    def test_no_sd_uses_floor(self):
        self.assertEqual(compare.tolerance("csqa", None), 2)
        self.assertEqual(compare.tolerance("deg", None), 5)

    def test_unknown_metric_raises(self):
        with self.assertRaises(KeyError):
            compare.tolerance("nope", 1.0)


class VerdictTest(unittest.TestCase):
    def test_verdicts(self):
        self.assertEqual(compare.verdict(50, None, 5), "SEM_BASELINE")
        self.assertEqual(compare.verdict(None, 50, 5), "NAO_EXECUTADO")
        self.assertEqual(compare.verdict(55, 50, 5), "OK")
        self.assertEqual(compare.verdict(44.9, 50, 5), "FORA_DA_TOLERANCIA")
        self.assertEqual(compare.verdict(55.1, 50, 5), "FORA_DA_TOLERANCIA")


class CompareRowsTest(unittest.TestCase):
    def test_sd_comes_from_baseline_seeds_of_same_group(self):
        base = [row("rk", f"q15__msc__s{i}", "msc", "pl_lp", v) for i, v in enumerate([70.0, 85.0, 96.0])]
        new = [row("rk", "q15__msc__s0", "msc", "pl_lp", 95.0)]
        out = {r["run"]: r for r in compare.compare_rows(new, base)}
        # sd das 3 seeds ~13.0 -> tolerância ~26 > piso 10
        self.assertGreater(out["q15__msc__s0"]["tolerance"], 20)
        self.assertEqual(out["q15__msc__s0"]["verdict"], "OK")
        self.assertEqual(out["q15__msc__s1"]["verdict"], "NAO_EXECUTADO")

    def test_new_without_baseline(self):
        new = [row("gustavo", "icl_safe", "icl_safe", "deg", 1.0)]
        out = compare.compare_rows(new, [])
        self.assertEqual(out[0]["verdict"], "SEM_BASELINE")
        self.assertIsNone(out[0]["baseline"])

    def test_output_keys(self):
        out = compare.compare_rows([row("davi", "untuned", "untuned", "csqa", 71.0)],
                                   [row("davi", "untuned", "untuned", "csqa", 71.2)])
        self.assertEqual(set(out[0]), {"block", "run", "metric", "new", "baseline", "tolerance", "verdict"})
        self.assertEqual(out[0]["verdict"], "OK")


class ClaimsTest(unittest.TestCase):
    def rk_rows(self, helpful, control):
        rows = [row("rk", "q15__base", "base", "pl_s", 92.7), row("rk", "q15__base", "base", "cim", 69.0)]
        for s in range(3):
            rows.append(row("rk", f"q15__synth_helpful__s{s}", "synth_helpful", "pl_s", helpful[s]))
            rows.append(row("rk", f"q15__synth_control__s{s}", "synth_control", "pl_s", control[s]))
            rows.append(row("rk", f"q15__dialogsum__s{s}", "dialogsum", "cim", 50.0))
            rows.append(row("rk", f"q15__dialogsum__s{s}", "dialogsum", "pl_s", 77.0))
        return rows

    def by_id(self, claims):
        return {c["id"]: c["holds"] for c in claims}

    def test_rk_claims_hold(self):
        c = self.by_id(compare.check_claims(self.rk_rows([45, 46, 44], [64, 60, 70])))
        self.assertTrue(c["rk_helpful_lt_control"])
        self.assertTrue(c["rk_ft_below_base"])
        self.assertTrue(c["rk_dialogsum_cim_below_base"])

    def test_rk_claim_fails_in_one_seed(self):
        c = self.by_id(compare.check_claims(self.rk_rows([45, 80, 44], [64, 60, 70])))
        self.assertFalse(c["rk_helpful_lt_control"])

    def test_missing_block_gives_none(self):
        c = self.by_id(compare.check_claims([]))
        self.assertIsNone(c["rk_helpful_lt_control"])
        self.assertIsNone(c["gustavo_collapse_gt_control"])

    def test_gustavo_and_davi_claims(self):
        rows = [row("gustavo", n, n, "deg", v) for n, v in
                [("dpo_collapse", 65.7), ("dpo_control", 0.0), ("zero_shot", 0.0)]]
        rows += [row("davi", n, n, "pl_s", v) for n, v in
                 [("degraded_s0", 43.4), ("nopii_ft_s0", 49.4), ("role_swap_ft_v2_s0", 50.0)]]
        c = self.by_id(compare.check_claims(rows))
        self.assertTrue(c["gustavo_collapse_gt_control"])
        self.assertTrue(c["gustavo_collapse_gt_zero_shot"])
        self.assertTrue(c["davi_degraded_lowest"])


class ExtractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_extract_rk_ignores_candidates_and_underscore(self):
        r = self.tmp / "results"
        meta = {"acc_logprob": 90.0, "acc_sampled_mean": 80.0}
        write_json(r / "privacylens" / "q15__base.json", {"meta": meta})
        write_json(r / "privacylens" / "q15__msc__s1.json", {"meta": meta})
        write_json(r / "privacylens" / "cand_x.json", {"meta": meta})
        write_json(r / "privacylens" / "_q15__base_v1.json", {"meta": meta})
        write_json(r / "privacylens_semantic" / "q15__base.json", {"meta": {"acc_semantic": 94.3}})
        write_json(r / "cimemories" / "q15__base.json", {"meta": {"privacy_acc": 69.0}})
        write_json(r / "csqa" / "q15__base.json", {"meta": {"acc": 71.7}})
        rows = baselines.extract_rk(r)
        runs = {x["run"] for x in rows}
        self.assertEqual(runs, {"q15__base", "q15__msc__s1"})
        got = {(x["run"], x["metric"]): x["value"] for x in rows}
        self.assertEqual(got[("q15__base", "sem")], 94.3)
        self.assertEqual(got[("q15__base", "cim")], 69.0)
        self.assertEqual(got[("q15__base", "csqa")], 71.7)
        groups = {x["run"]: x["group"] for x in rows}
        self.assertEqual(groups["q15__msc__s1"], "msc")
        self.assertEqual(groups["q15__base"], "base")

    def test_extract_davi(self):
        r = self.tmp / "eval_results"
        write_json(r / "csqa" / "untuned.json", {"metrics": {"acc": 71.17}})
        write_json(r / "privacylens" / "degraded_s0.json",
                   {"metrics": {"acc_logprob": 43.0, "acc_sampled_mean": 43.4}})
        got = {(x["run"], x["metric"]): x["value"] for x in baselines.extract_davi(r)}
        self.assertEqual(got, {("untuned", "csqa"): 71.17, ("degraded_s0", "pl_lp"): 43.0,
                               ("degraded_s0", "pl_s"): 43.4})

    def test_extract_gustavo(self):
        p = self.tmp / "summary.csv"
        p.write_text("condition,n,invalid,degraded_%,deg_%_prompts_degrading,deg_%_prompts_safe,avg_words\n"
                     "original,248,0,100.0,,,34.8\nicl_safe,248,0,1.2,,,79.1\n", encoding="utf-8")
        got = {(x["run"], x["metric"]): x["value"] for x in baselines.extract_gustavo(p)}
        self.assertEqual(got, {("original", "deg"): 100.0, ("icl_safe", "deg"): 1.2})

    def test_baseline_roundtrip(self):
        rows = [row("rk", "q15__base", "base", "pl_s", 92.74536766261402)]
        p = self.tmp / "rk.csv"
        baselines.write_csv(rows, p)
        self.assertEqual(baselines.read_csv(p), rows)


class VersionedBaselinesTest(unittest.TestCase):
    """Os CSVs versionados batem com a tabela preliminar do Rubens/Kayque e com o summary do Gustavo."""

    def test_rk_means_match_summary_original(self):
        rows = baselines.load_baseline("rk")
        self.assertEqual(len({r["run"] for r in rows}), 34)
        def mean(group, metric):
            v = [r["value"] for r in rows if r["group"] == group and r["metric"] == metric]
            return round(sum(v) / len(v), 1)
        self.assertEqual(mean("base", "pl_lp"), 98.8)
        self.assertEqual(mean("synth_helpful", "pl_s"), 45.2)
        self.assertEqual(mean("support", "csqa"), 37.8)
        self.assertEqual(mean("dialogsum", "cim"), 54.3)
        self.assertEqual(mean("msc", "sem"), 48.1)

    def test_all_60_cells_of_summary_original(self):
        import math
        from reproduce.paths import REPO
        md = (REPO / "RUBENS_KAYQUE/reproducao_tabela_resultados/experiments/results/summary_original.md"
              ).read_text(encoding="utf-8").splitlines()
        rows = baselines.load_baseline("rk")
        cols = {"pl_lp": 4, "pl_s": 6, "sem": 8, "cim": 10, "csqa": 12}
        n = 0
        for line in md:
            p = [x.strip() for x in line.strip("|").split("|")]
            if len(p) < 13 or p[0] != "q15":
                continue
            group = "base" if p[1] == "BASE" else p[1]
            for metric, i in cols.items():
                v = [r["value"] for r in rows if r["group"] == group and r["metric"] == metric]
                m = sum(v) / len(v)
                sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
                exp = p[i].replace("±", "").split()
                self.assertEqual(f"{m:.1f}", exp[0], (group, metric))
                if len(exp) > 1:
                    self.assertEqual(f"{sd:.1f}", exp[1], (group, metric))
                n += 1
        self.assertEqual(n, 60)

    def test_gustavo_baseline(self):
        got = {r["run"]: r["value"] for r in baselines.load_baseline("gustavo")}
        self.assertEqual(got, {"original": 100.0, "zero_shot": 0.0, "icl_degrading": 2.8,
                               "icl_safe": 1.2, "dpo_collapse": 65.7, "dpo_control": 0.0})

    def test_davi_baseline(self):
        got = {(r["run"], r["metric"]) for r in baselines.load_baseline("davi")}
        self.assertIn(("untuned", "csqa"), got)
        self.assertIn(("role_swap_ft_v2_s0", "pl_s"), got)
        self.assertEqual(len(got), 4 + 3 * 2)


if __name__ == "__main__":
    unittest.main()
