import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from reproduce.paths import REPO


class CliTest(unittest.TestCase):
    def run_cli(self, *args):
        out = Path(tempfile.mkdtemp()) / "saída com espaço º"
        env = {**os.environ, "REPRO_OUTPUTS": str(out)}
        p = subprocess.run([sys.executable, str(REPO / "reproduce.py"), *args], capture_output=True,
                           text=True, env=env, cwd=tempfile.gettempdir())
        return p, out

    def test_report_only_preliminary(self):
        p, out = self.run_cli("--report-only", "--preliminary")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("REPRODUZIDO", p.stdout)
        self.assertTrue((out / "preliminar" / "REPRODUCAO.md").exists())

    def test_report_only_without_outputs_is_partial(self):
        p, out = self.run_cli("--report-only")
        self.assertEqual(p.returncode, 1)
        self.assertIn("PARCIAL", p.stdout)

    def test_preliminary_requires_report_only(self):
        p, _ = self.run_cli("--preliminary")
        self.assertEqual(p.returncode, 2)


if __name__ == "__main__":
    unittest.main()
