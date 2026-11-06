import tempfile
import unittest
from pathlib import Path

from reproduce import envs


class FakeRun:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append([str(c) for c in cmd])
        # simula a criação do venv
        if "venv" in cmd:
            py = envs.env_python_in(Path(cmd[-1]))
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")


class EnvsTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()) / "envs com espaço"

    def test_torch_args(self):
        self.assertEqual(envs.torch_install_args("cuda"),
                         ["torch==2.9.1", "--index-url", "https://download.pytorch.org/whl/cu128"])
        self.assertEqual(envs.torch_install_args("mps"), ["torch==2.9.1"])
        self.assertEqual(envs.torch_install_args("cpu"), ["torch==2.9.1"])

    def test_find_python312_order(self):
        w = {"python3.12": "/a/python3.12", "uv": "/b/uv"}.get
        self.assertEqual(envs.find_python312(which=w), envs.Py312("python", "/a/python3.12"))
        w = {"uv": "/b/uv", "conda": "/c/conda"}.get
        self.assertEqual(envs.find_python312(which=w), envs.Py312("uv", "/b/uv"))
        self.assertIsNone(envs.find_python312(which=lambda _: None))

    def test_ensure_env_installs_then_marks_ready_and_is_idempotent(self):
        run = FakeRun()
        py = envs.ensure_env("main", "cuda", envs.Py312("python", "/a/python3.12"), root=self.root, run=run)
        flat = [" ".join(c) for c in run.calls]
        self.assertTrue(any("venv" in c for c in flat))
        self.assertTrue(any("torch==2.9.1" in c and "cu128" in c for c in flat))
        self.assertTrue(any("main.txt" in c for c in flat))
        self.assertTrue(any("cuda.txt" in c for c in flat))
        self.assertTrue((self.root / "main" / ".pronto").exists())
        n = len(run.calls)
        self.assertEqual(envs.ensure_env("main", "cuda", envs.Py312("python", "/a/python3.12"),
                                         root=self.root, run=run), py)
        self.assertEqual(len(run.calls), n)

    def test_mps_does_not_install_bitsandbytes(self):
        run = FakeRun()
        envs.ensure_env("main", "mps", envs.Py312("python", "/a/python3.12"), root=self.root, run=run)
        self.assertFalse(any("cuda.txt" in " ".join(c) for c in run.calls))

    def test_failure_does_not_mark_ready(self):
        def boom(cmd, **kw):
            if "pip" in cmd:
                raise envs.EnvError("falhou")
            FakeRun()(cmd)
        with self.assertRaises(envs.EnvError):
            envs.ensure_env("pii", "cpu", envs.Py312("python", "/a/python3.12"), root=self.root, run=boom)
        self.assertFalse((self.root / "pii" / ".pronto").exists())

    def test_changed_requirements_reinstall(self):
        run = FakeRun()
        envs.ensure_env("pii", "cpu", envs.Py312("python", "/a/python3.12"), root=self.root, run=run)
        (self.root / "pii" / ".pronto").write_text("outro-hash")
        n = len(run.calls)
        envs.ensure_env("pii", "cpu", envs.Py312("python", "/a/python3.12"), root=self.root, run=run)
        self.assertGreater(len(run.calls), n)

    def test_uv_route_uses_uv(self):
        run = FakeRun()
        envs.ensure_env("pii", "cpu", envs.Py312("uv", "/b/uv"), root=self.root, run=run)
        self.assertEqual(run.calls[0][:2], ["/b/uv", "venv"])
        self.assertIn("3.12", run.calls[0])


if __name__ == "__main__":
    unittest.main()
