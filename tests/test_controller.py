import tempfile
import unittest
from dataclasses import replace
from unittest import mock

from reproduce import cli, controller, fetch
from tests.test_preflight import mac, nvidia


class Boom(AssertionError):
    pass


def forbidden(*a, **k):
    raise Boom("não deveria ser chamado")


class ControllerTest(unittest.TestCase):
    def run_ctrl(self, argv, facts, **kw):
        args = cli.parse_args(argv)
        defaults = dict(collect=lambda: facts, interactive=False, input_fn=lambda _: "",
                        ensure_env=forbidden, ensure_inputs=forbidden, run_steps=forbidden)
        defaults.update(kw)
        return controller.run(args, **defaults)

    def test_check_never_executes(self):
        self.assertEqual(self.run_ctrl(["--check"], nvidia()), 0)

    def test_non_interactive_without_yes_does_nothing(self):
        self.assertEqual(self.run_ctrl([], nvidia()), 1)

    def test_answer_no_does_nothing(self):
        self.assertEqual(self.run_ctrl([], nvidia(), interactive=True, input_fn=lambda _: "n"), 1)

    def test_general_block_stops_even_with_yes(self):
        self.assertEqual(self.run_ctrl(["--yes"], nvidia(disk_free_gb=1)), 1)

    def test_gustavo_blocked_others_run(self):
        seen = {}

        def fake_steps(steps, ctx):
            seen["blocks"] = sorted({s.block for s in steps})
            return []

        with tempfile.TemporaryDirectory() as t, mock.patch.object(controller, "OUTPUTS", __import__("pathlib").Path(t)):
            rc = self.run_ctrl(["--yes", "--smoke"], mac(), ensure_env=lambda *a, **k: "/py",
                               ensure_inputs=lambda: fetch.FetchResult(), run_steps=fake_steps)
        self.assertEqual(seen["blocks"], ["davi", "rk"])
        self.assertIn(rc, (0, 1))

    def test_deviations_on_mac(self):
        d = " ".join(controller.deviations("mps", ["rk", "davi", "gustavo"]))
        self.assertIn("MPS", d)
        self.assertIn("4 bits", d)
        self.assertEqual(controller.deviations("cuda", []), [])


if __name__ == "__main__":
    unittest.main()
