import json
import sys
import tempfile
import unittest
from pathlib import Path

from reproduce import blocks, runner
from reproduce.runner import Step


def ctx_for(tmp, smoke=False, device="cuda"):
    return blocks.Context(device=device, smoke=smoke, out=Path(tmp) / "out", art=Path(tmp) / "art",
                          pythons={"main": Path(sys.executable), "pii": Path(sys.executable)})


def py_step(name, code, outputs=(), block="b"):
    return Step(block=block, name=name, args=["-c", code], env="main", outputs=list(outputs))


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "caminho com espaço e º"
        self.ctx = ctx_for(self.tmp)

    def ledger(self):
        p = self.ctx.out / "ledger.jsonl"
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]

    def test_success_writes_marker_and_ledger_and_is_skipped_next_time(self):
        out = self.tmp / "a.txt"
        s = py_step("cria", f"open({str(out)!r}, 'w').write('ok')", [out])
        r1 = runner.run_steps([s], self.ctx)
        self.assertEqual(r1[0].status, "ok")
        r2 = runner.run_steps([s], self.ctx)
        self.assertEqual(r2[0].status, "pulado")
        led = self.ledger()
        self.assertEqual([x["status"] for x in led], ["ok", "pulado"])
        self.assertEqual(led[0]["returncode"], 0)
        self.assertIn("duration_s", led[0])

    def test_fresh_reruns(self):
        out = self.tmp / "a.txt"
        s = py_step("cria", f"open({str(out)!r}, 'w').write('ok')", [out])
        runner.run_steps([s], self.ctx)
        r = runner.run_steps([s], self.ctx, fresh=True)
        self.assertEqual(r[0].status, "ok")

    def test_failure_stops_block_but_not_others(self):
        steps = [py_step("falha", "raise SystemExit(3)", block="b1"),
                 py_step("depois", "pass", block="b1"),
                 py_step("outro", "pass", block="b2")]
        r = {x.step.name: x.status for x in runner.run_steps(steps, self.ctx)}
        self.assertEqual(r, {"falha": "falhou", "depois": "nao_executado", "outro": "ok"})

    def test_missing_output_counts_as_failure(self):
        r = runner.run_steps([py_step("mente", "pass", [self.tmp / "nunca.txt"])], self.ctx)
        self.assertEqual(r[0].status, "falhou")

    def test_partial_output_without_marker_is_rerun_and_deleted_first(self):
        out = self.tmp / "parcial.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{incompleto", encoding="utf-8")   # como se o processo tivesse sido morto
        code = f"import os; assert not os.path.exists({str(out)!r}); open({str(out)!r}, 'w').write('{{}}')"
        r = runner.run_steps([py_step("refaz", code, [out])], self.ctx)
        self.assertEqual(r[0].status, "ok")
        self.assertEqual(out.read_text(encoding="utf-8"), "{}")

    def test_log_and_env_and_cwd_with_spaces(self):
        cwd = self.tmp / "pasta º"
        s = Step(block="b", name="env", args=["-c", "import os; print(os.environ['X_TESTE'], os.getcwd())"],
                 env="main", cwd=cwd, extra_env={"X_TESTE": "valor"})
        r = runner.run_steps([s], self.ctx)
        self.assertEqual(r[0].status, "ok")
        log = r[0].log.read_text(encoding="utf-8")
        self.assertIn("valor", log)
        self.assertIn("pasta º", log)

    def test_marker_names_are_safe(self):
        self.assertEqual(runner.marker_name(Step(block="rk", name="train:q15__msc__s0", args=[], env="main")),
                         "rk__train_q15__msc__s0.done")


class BlocksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def names(self, steps):
        return [s.name for s in steps]

    def test_rk_full(self):
        steps = blocks.build_rk(ctx_for(self.tmp))
        train = [s for s in steps if s.name.startswith("train:")]
        self.assertEqual(len(train), 33)
        a = train[0].args
        self.assertIn("--grad-ckpt", a)
        self.assertEqual(a[a.index("--epochs") + 1], "2")
        self.assertEqual(a[a.index("--lr") + 1], "0.0002")
        self.assertIn("privacylens:q15__base", self.names(steps))
        base_pl = next(s for s in steps if s.name == "privacylens:q15__base").args
        self.assertEqual(base_pl[base_pl.index("--seeds") + 1:base_pl.index("--seeds") + 4], ["0", "1", "2"])
        ft_pl = next(s for s in steps if s.name == "privacylens:q15__msc__s1").args
        self.assertEqual(ft_pl[ft_pl.index("--seeds") + 1:], ["0"])
        self.assertEqual(sum(n.startswith("csqa:") for n in self.names(steps)), 34)
        self.assertEqual(sum(n.startswith("cim_gen:") for n in self.names(steps)), 34)
        self.assertIn("juiz", self.names(steps))
        self.assertTrue(all(s.extra_env["REPRO_DEVICE"] == "cuda" for s in steps if s.env))

    def test_rk_smoke_is_small_and_separate(self):
        ctx = ctx_for(self.tmp, smoke=True)
        steps = blocks.build_rk(ctx)
        self.assertEqual(len([s for s in steps if s.name.startswith("train:")]), 1)
        pl = next(s for s in steps if s.name.startswith("privacylens:q15__base")).args
        self.assertEqual(pl[pl.index("--limit") + 1], "8")
        self.assertTrue(str(ctx.out).endswith("smoke"))

    def test_davi_runs(self):
        steps = blocks.build_davi(ctx_for(self.tmp))
        evals = sorted(n.split(":", 1)[1] for n in self.names(steps) if n.startswith(("privacylens:", "csqa:")))
        self.assertEqual(evals, sorted(["degraded_s0", "nopii_ft_s0", "role_swap_ft_v2_s0",
                                        "degraded_s0", "nopii_ft_s0", "role_swap_ft_v2_s0", "untuned"]))
        trains = [s for s in steps if s.name.startswith("train:")]
        self.assertEqual(sorted(s.name for s in trains), ["train:degraded", "train:nopii_ft", "train:role_swap_ft_v2"])
        self.assertTrue(all(s.args[s.args.index("--seed") + 1] == "0" for s in trains))
        pii = next(s for s in steps if s.name == "dados:nopii")
        self.assertEqual(pii.env, "pii")
        self.assertIn("logit_lens", self.names(steps))

    def test_gustavo_order(self):
        steps = blocks.build_gustavo(ctx_for(self.tmp))
        self.assertEqual(self.names(steps), ["insumos_hf", "dpo", "icl", "juiz"])
        judge = steps[-1]
        self.assertEqual(judge.extra_env["REPRO_ICL"], "1")
        self.assertNotIn("REPRO_LIMIT", judge.extra_env)

    def test_gustavo_smoke_limits(self):
        steps = blocks.build_gustavo(ctx_for(self.tmp, smoke=True))
        self.assertEqual(steps[-1].extra_env["REPRO_LIMIT"], "4")
        self.assertEqual(steps[1].extra_env["REPRO_SMOKE_N"], "6")


if __name__ == "__main__":
    unittest.main()
