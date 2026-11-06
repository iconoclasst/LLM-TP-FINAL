import unittest
from dataclasses import replace

from reproduce import estimates, preflight
from reproduce.preflight import Facts

BLOCKS = ("rk", "davi", "gustavo")


def mac(**kw):
    f = Facts(os="Darwin", machine="arm64", cpu="Apple M4", ram_gb=16, gpu=None, vram_gb=None, cuda=False,
              mps=True, disk_free_gb=200, python312="/usr/bin/python3.12", uv=None, conda=None, curl=True,
              ollama=None, ollama_running=False, ollama_models=[], online_drive=True, online_hf=True,
              artifacts_missing=0, envs_ready=[])
    return replace(f, **kw)


def nvidia(**kw):
    return replace(mac(os="Linux", machine="x86_64", gpu="NVIDIA GeForce RTX 4060", vram_gb=8, cuda=True,
                       mps=False, ollama="/usr/bin/ollama", ollama_running=True, ollama_models=["qwen3:8b"]), **kw)


def levels(issues, block=None):
    return {(i.level, i.code) for i in issues if block is None or i.block == block}


class AssessTest(unittest.TestCase):
    def test_device(self):
        self.assertEqual(preflight.choose_device(nvidia()), "cuda")
        self.assertEqual(preflight.choose_device(mac()), "mps")
        self.assertEqual(preflight.choose_device(mac(mps=False)), "cpu")

    def test_no_ollama_blocks_only_gustavo_and_explains(self):
        issues = preflight.assess(mac(), BLOCKS)
        block = [i for i in issues if i.level == "BLOQUEIO"]
        self.assertEqual({i.block for i in block}, {"gustavo"})
        self.assertIn("ollama.com", block[0].fix)
        self.assertIn("ollama pull qwen3:8b", block[0].fix)

    def test_ollama_installed_but_stopped_or_without_model(self):
        self.assertIn(("BLOQUEIO", "ollama_parado"),
                      levels(preflight.assess(mac(ollama="/x/ollama"), BLOCKS), "gustavo"))
        self.assertIn(("BLOQUEIO", "ollama_sem_modelo"),
                      levels(preflight.assess(mac(ollama="/x/ollama", ollama_running=True), BLOCKS), "gustavo"))

    def test_mac_warns_about_gpu_and_quantization(self):
        lv = levels(preflight.assess(mac(), BLOCKS))
        self.assertIn(("AVISO", "sem_cuda"), lv)
        self.assertIn(("AVISO", "sem_4bit"), lv)

    def test_nvidia_machine_with_everything_has_no_blocks(self):
        issues = preflight.assess(nvidia(), BLOCKS)
        self.assertEqual([i for i in issues if i.level == "BLOQUEIO"], [])
        self.assertNotIn(("AVISO", "sem_cuda"), levels(issues))

    def test_disk(self):
        issues = preflight.assess(nvidia(disk_free_gb=10), BLOCKS)
        self.assertIn(("BLOQUEIO", "disco"), levels(issues))

    def test_no_python312_route(self):
        self.assertIn(("BLOQUEIO", "sem_python312"), levels(preflight.assess(nvidia(python312=None), BLOCKS)))
        self.assertNotIn(("BLOQUEIO", "sem_python312"),
                         levels(preflight.assess(nvidia(python312=None, uv="/x/uv"), BLOCKS)))

    def test_offline_with_missing_inputs(self):
        lv = levels(preflight.assess(nvidia(online_drive=False, artifacts_missing=3), BLOCKS))
        self.assertIn(("BLOQUEIO", "drive"), lv)

    def test_without_gpu_warns_time(self):
        self.assertIn(("AVISO", "tempo"), levels(preflight.assess(mac(mps=False), ("davi",))))

    def test_only_selected_blocks(self):
        issues = preflight.assess(mac(), ("rk",))
        self.assertEqual({i.block for i in issues if i.level == "BLOQUEIO"}, set())


class EstimateTest(unittest.TestCase):
    def test_cuda_measured(self):
        e = estimates.estimate("rk", "cuda", smoke=False)
        self.assertEqual((e.hours_low, e.hours_high, e.measured), (21, 22, True))
        self.assertFalse(estimates.estimate("gustavo", "cuda", False).measured)

    def test_mps_unknown(self):
        e = estimates.estimate("rk", "mps", smoke=False)
        self.assertIsNone(e.hours_low)

    def test_disk_shared_components(self):
        self.assertLess(estimates.disk_needed_gb(("rk", "davi", "gustavo"), ollama_model_present=False),
                        sum(estimates.disk_needed_gb((b,), False) for b in ("rk", "davi", "gustavo")))


class RenderConfirmTest(unittest.TestCase):
    def test_render_lists_installs_and_plan(self):
        f = mac()
        txt = preflight.render(f, preflight.assess(f, BLOCKS), BLOCKS, smoke=False)
        for s in ("BLOQUEIO", "AVISO", "Ollama", "Será instalado", "disco", "Tempo estimado", "gustavo"):
            self.assertIn(s, txt)

    def test_confirm(self):
        self.assertTrue(preflight.confirm(yes=True, interactive=False, input_fn=lambda _: ""))
        self.assertFalse(preflight.confirm(yes=False, interactive=False, input_fn=lambda _: "s"))
        self.assertFalse(preflight.confirm(yes=False, interactive=True, input_fn=lambda _: ""))
        self.assertTrue(preflight.confirm(yes=False, interactive=True, input_fn=lambda _: "s"))
        self.assertFalse(preflight.confirm(yes=False, interactive=True, input_fn=lambda _: "n"))

    def test_collect_facts_runs_here_without_side_effects(self):
        f = preflight.collect_facts(network=False)
        self.assertIsInstance(f.disk_free_gb, float)
        self.assertFalse(f.online_drive)


if __name__ == "__main__":
    unittest.main()
