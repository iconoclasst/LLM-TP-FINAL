"""Regressões dos problemas apontados na revisão de código."""
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from reproduce import blocks, cli, preflight, report, runner
from reproduce.runner import Step
from reproduce.steps import gustavo_judge
from tests.test_preflight import mac, nvidia
from tests.test_runner import ctx_for, py_step


class DaviRealScriptTest(unittest.TestCase):
    def test_role_swap_step_runs_from_empty_cwd_and_matches_original_count(self):
        tmp = Path(tempfile.mkdtemp()) / "saída com espaço"
        ctx = ctx_for(tmp)
        src = Path(__file__).resolve().parents[1] / "Gustavo" / "dataset_degraded_ft.jsonl"   # cópia local, fora do Git
        if not src.exists():
            self.skipTest("dataset do Gustavo não baixado (rode o reproduce.py ou baixe do Drive)")
        ctx.gustavo_data.mkdir(parents=True)
        import shutil
        shutil.copyfile(src, ctx.gustavo_data / "dataset_degraded_ft.jsonl")
        step = next(s for s in blocks.build_davi(ctx) if s.name == "dados:role_swap_v2")
        step = replace(step, env=None)   # só biblioteca padrão: roda com o Python atual
        r = runner.run_steps([step], ctx)
        self.assertEqual(r[0].status, "ok", r[0].log.read_text(encoding="utf-8")[-800:])
        counts = report.davi_counts(ctx.out / "davi" / "data")
        self.assertEqual(counts["role_swap: linhas trocadas"], 1729)


class RunnerFixesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ctx = ctx_for(self.tmp)

    def test_parent_dirs_of_outputs_are_created(self):
        out = self.tmp / "a" / "b" / "c.txt"
        r = runner.run_steps([py_step("x", f"open({str(out)!r}, 'w').write('1')", [out])], self.ctx)
        self.assertEqual(r[0].status, "ok")

    def test_keep_partial_does_not_delete_finished_outputs(self):
        done, todo = self.tmp / "done.json", self.tmp / "todo.json"
        done.write_text("{}")
        code = f"import os; assert os.path.exists({str(done)!r}); open({str(todo)!r}, 'w').write('{{}}')"
        s = replace(py_step("juiz", code, [done, todo]), keep_partial=True)
        self.assertEqual(runner.run_steps([s], self.ctx)[0].status, "ok")

    def test_marker_from_other_device_is_not_reused(self):
        out = self.tmp / "o.txt"
        s = py_step("x", f"open({str(out)!r}, 'w').write('1')", [out])
        runner.run_steps([s], ctx_for(self.tmp, device="mps"))
        self.assertEqual(runner.run_steps([s], ctx_for(self.tmp, device="cuda"))[0].status, "ok")
        self.assertEqual(runner.run_steps([s], ctx_for(self.tmp, device="cuda"))[0].status, "pulado")


class BlocksFixesTest(unittest.TestCase):
    def test_offline_after_download_and_csqa_pinned(self):
        ctx = ctx_for(Path(tempfile.mkdtemp()))
        for build in (blocks.build_rk, blocks.build_davi, blocks.build_gustavo):
            steps = build(ctx)
            hf = steps[0]
            self.assertEqual(hf.name, "insumos_hf")
            self.assertEqual(hf.extra_env["HF_HUB_OFFLINE"], "0")
            self.assertTrue(all(s.extra_env.get("HF_HUB_OFFLINE") == "1" for s in steps[1:] if s.env))
        self.assertIn("--csqa", blocks.build_rk(ctx)[0].args)
        self.assertIn("--csqa", blocks.build_davi(ctx)[0].args)

    def test_long_resumable_steps_keep_partial(self):
        ctx = ctx_for(Path(tempfile.mkdtemp()))
        keep = {(s.block, s.name) for b in blocks.BUILDERS.values() for s in b(ctx) if s.keep_partial}
        self.assertEqual(keep, {("rk", "juiz"), ("gustavo", "dpo"), ("gustavo", "juiz")})


class LedgerTest(unittest.TestCase):
    def test_only_latest_attempt_counts(self):
        p = Path(tempfile.mkdtemp()) / "ledger.jsonl"
        recs = [{"block": "rk", "step": "a", "status": "falhou", "returncode": 1},
                {"block": "rk", "step": "b", "status": "ok", "returncode": 0},
                {"block": "rk", "step": "a", "status": "ok", "returncode": 0}]
        p.write_text("".join(json.dumps(r) + "\n" for r in recs))
        got = cli.read_ledger(p)
        self.assertEqual([(r["step"], r["status"]) for r in got], [("b", "ok"), ("a", "ok")])
        self.assertEqual(report.overall_verdict([{"verdict": "OK"}], [], got, False), "REPRODUZIDO")


class PreflightFixesTest(unittest.TestCase):
    def test_exact_ollama_model(self):
        f = nvidia(ollama_models=["qwen3:8b-q8_0"])
        self.assertIn("ollama_sem_modelo", {i.code for i in preflight.assess(f, ("gustavo",))})

    def test_linux_without_vulkan_blocks_rk_only(self):
        issues = preflight.assess(nvidia(vulkan=False), ("rk", "davi"))
        self.assertEqual({(i.block, i.code) for i in issues if i.level == "BLOQUEIO"}, {("rk", "sem_vulkan")})
        self.assertNotIn("sem_vulkan", {i.code for i in preflight.assess(
            nvidia(vulkan=False, llama_server_env=True), ("rk",))})
        self.assertNotIn("sem_vulkan", {i.code for i in preflight.assess(mac(), ("rk",))})


class GustavoJudgeTest(unittest.TestCase):
    def test_null_verdicts_are_detected(self):
        d = Path(tempfile.mkdtemp())
        (d / "judged_ok.jsonl").write_text(json.dumps({"verdict": "SAFE"}) + "\n")
        (d / "judged_bad.jsonl").write_text(json.dumps({"verdict": None}) + "\n")
        self.assertEqual([p.name for p in gustavo_judge.bad_judged(d)], ["judged_bad.jsonl"])

    def test_wrapper_fails_and_removes_summary_when_judge_left_nulls(self):
        d = Path(tempfile.mkdtemp())
        fake = d / "Tester.py"
        fake.write_text(
            "import json\n"
            f"open({str(d / 'judged_x.jsonl')!r}, 'w').write(json.dumps({{'verdict': None}}) + '\\n')\n"
            f"open({str(d / 'summary.csv')!r}, 'w').write('condition,n,invalid\\nx,1,1\\n')\n")
        import subprocess
        p = subprocess.run([sys.executable, "-m", "reproduce.steps.gustavo_judge", str(fake), str(d)],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertFalse((d / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
