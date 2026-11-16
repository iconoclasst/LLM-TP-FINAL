"""Verificações estáticas das adaptações nos scripts dos três trabalhos (spec, seção 7)."""
import re
import subprocess
import unittest
from collections import Counter

from reproduce.paths import REPO

ORIGINAL = "ed0db1d"   # último commit antes das adaptações (inclui as correções do Gustavo)
RK = "exp1_reproducao_colapso/reproducao_tabela_resultados/experiments/scripts/"
ADAPTED = [RK + f for f in ("common.py", "eval_csqa.py", "eval_cimemories.py", "eval_privacylens.py", "train_lora.py")] + [
    "exp2_mitigacoes_dataset/solutions/src/fine_tune_models.py", "exp2_mitigacoes_dataset/solutions/benchmarks/eval_core.py",
    "exp2_mitigacoes_dataset/solutions/benchmarks/run_eval.py", "exp2_mitigacoes_dataset/logit_lens/logit_lens.py",
    "exp3_dpo_icl/run_all.py", "exp3_dpo_icl/ICL.py", "exp3_dpo_icl/Tester.py"]
FORBIDDEN = [r'device_map=\{"": 0\}', r"\.to\(0\)", r'Generator\(device="cuda"\)', r"assert torch\.cuda\.is_available",
             r'GradScaler\("cuda"\)\s*$']
OLD_DIRS = {"exp1_reproducao_colapso": "RUBENS_KAYQUE", "exp2_mitigacoes_dataset": "DAVI",
            "exp3_dpo_icl": "Gustavo"}   # nomes das pastas no commit ORIGINAL
NUM = re.compile(r"(?<![\w.])(\d+\.\d+|\d+e-?\d+|\d+)(?![\w.])")


def original(path):
    top, rest = path.split("/", 1)
    path = f"{OLD_DIRS[top]}/{rest}"
    try:
        return subprocess.run(["git", "show", f"{ORIGINAL}:{path}"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def code_numbers(text):
    # ignora comentários e docstrings de uma linha, onde os textos explicativos novos citam números
    lines = [l.split("#", 1)[0] for l in text.splitlines()]
    return Counter(NUM.findall("\n".join(lines)))


class StaticTest(unittest.TestCase):
    def test_no_hardcoded_cuda(self):
        for f in ADAPTED:
            src = (REPO / f).read_text(encoding="utf-8")
            for pat in FORBIDDEN:
                self.assertIsNone(re.search(pat, src, re.M), f"{f}: {pat}")

    def test_unguarded_cuda_only_in_helpers(self):
        bad = []
        for f in ADAPTED:
            lines = (REPO / f).read_text(encoding="utf-8").splitlines()
            for i, l in enumerate(lines):
                if re.search(r"torch\.cuda\.(reset_peak_memory_stats|max_memory_allocated|get_device_name|"
                             r"get_device_properties|memory_allocated)", l):
                    ctx = " ".join(lines[max(0, i - 3):i + 1])
                    if not re.search(r"if CUDA|CUDA else|is_available\(\)|== \"cuda\"", ctx):
                        bad.append(f"{f}:{i + 1}: {l.strip()}")
        self.assertEqual(bad, [])

    def test_numeric_literals_not_removed(self):
        """Todo número do código original continua presente (hiperparâmetros, seeds, limites)."""
        checked = 0
        for f in ADAPTED:
            old = original(f)
            if old is None:
                self.skipTest("git indisponível")
            checked += 1
            # o índice de GPU de `.to(0)`/`device_map={"": 0}` passou para os auxiliares de dispositivo
            # e o pico de memória de train_lora.py passou para common.max_memory_gb(), com o mesmo cálculo
            old = re.sub(r'\.to\(0\)|device_map=\{"": 0\}|get_device_name\(0\)|get_device_properties\(0\)', "", old)
            if f.endswith("train_lora.py"):
                old = old.replace("round(torch.cuda.max_memory_allocated() / 2**30, 2)", "")
            new = re.sub(r'get_device_name\(0\)|get_device_properties\(0\)', "",
                         (REPO / f).read_text(encoding="utf-8"))
            missing = code_numbers(old) - code_numbers(new)
            self.assertEqual(+missing, Counter(), f"{f}: números sumiram: {dict(missing)}")
        self.assertEqual(checked, len(ADAPTED))

    def test_defaults_unchanged_without_env(self):
        """Sem as variáveis REPRO_*, os caminhos e opções continuam os originais."""
        g = (REPO / "exp3_dpo_icl/Tester.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("REPRO_OUT_DIR") or "out2"', g)
        self.assertIn('ICL_FILES   = {}\nif os.environ.get("REPRO_ICL") == "1"', g.replace("\r\n", "\n"))
        c = (REPO / RK / "common.py").read_text(encoding="utf-8")
        self.assertIn('EXP / "data"', c)
        self.assertIn('return 0 if d == "cuda" else d', c)


if __name__ == "__main__":
    unittest.main()
