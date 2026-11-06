import hashlib
import tempfile
import unittest
from pathlib import Path

from reproduce import fetch
from reproduce.fetch import Entry

FIXTURE = Path(__file__).parent / "fixtures" / "drive_folder.html"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ParseTest(unittest.TestCase):
    def test_parse_folder_html(self):
        got = fetch.parse_folder_html(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(got, [Entry("FOLDER1", "Rubens/Kayque", True, "Nov 3"),
                               Entry("FILE_2-x", "a & b.json", False, "Oct 21")])

    def test_empty_folder(self):
        self.assertEqual(fetch.parse_folder_html("<html><body></body></html>"), [])


class FakeDrive:
    """Árvore falsa: {folder_id: [Entry]} e {file_id: bytes}."""

    def __init__(self, tree, files, online=True):
        self.tree, self.files, self.online, self.downloads = tree, files, online, []

    def lister(self, fid):
        if not self.online:
            raise fetch.FetchError("sem rede")
        return self.tree.get(fid, [])

    def downloader(self, fid, dest: Path):
        if not self.online:
            raise fetch.FetchError("sem rede")
        self.downloads.append(fid)
        dest.write_bytes(self.files[fid])


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()) / "pasta com espaço º"
        self.items = [fetch.ManifestItem("ID1", "x/a.jsonl", "rk/data/a.jsonl", sha(b"aaa")),
                      fetch.ManifestItem("ID2", "x/b.json", "rk/b.json", sha(b"bbb"))]

    def test_downloads_and_verifies(self):
        d = FakeDrive({}, {"ID1": b"aaa", "ID2": b"bbb"})
        r = fetch.ensure_manifest_files(self.items, self.root, d.downloader)
        self.assertEqual((r.missing, r.corrupt), ([], []))
        self.assertEqual((self.root / "rk/data/a.jsonl").read_bytes(), b"aaa")

    def test_wrong_hash_is_corrupt_and_not_kept(self):
        d = FakeDrive({}, {"ID1": b"changed", "ID2": b"bbb"})
        r = fetch.ensure_manifest_files(self.items, self.root, d.downloader)
        self.assertEqual([i.local for i in r.corrupt], ["rk/data/a.jsonl"])
        self.assertFalse((self.root / "rk/data/a.jsonl").exists())
        self.assertEqual(list(self.root.rglob("*.part")), [])

    def test_offline_with_valid_files_is_fine(self):
        fetch.ensure_manifest_files(self.items, self.root, FakeDrive({}, {"ID1": b"aaa", "ID2": b"bbb"}).downloader)
        off = FakeDrive({}, {}, online=False)
        r = fetch.ensure_manifest_files(self.items, self.root, off.downloader)
        self.assertEqual((r.missing, r.corrupt, off.downloads), ([], [], []))

    def test_offline_without_files_is_missing(self):
        r = fetch.ensure_manifest_files(self.items, self.root, FakeDrive({}, {}, online=False).downloader)
        self.assertEqual(len(r.missing), 2)

    def test_local_file_with_wrong_hash_is_redownloaded(self):
        p = self.root / "rk/b.json"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"old")
        d = FakeDrive({}, {"ID1": b"aaa", "ID2": b"bbb"})
        fetch.ensure_manifest_files(self.items, self.root, d.downloader)
        self.assertEqual(p.read_bytes(), b"bbb")

    def test_manifest_roundtrip(self):
        p = self.root / "m.tsv"
        fetch.write_manifest(self.items, p)
        self.assertEqual(fetch.read_manifest(p), self.items)

    def test_versioned_manifest_is_complete(self):
        items = fetch.read_manifest()
        locals_ = {i.local for i in items}
        self.assertIn("rk/benchmarks/ci_memories/gold_cimemories_singlejudge.json", locals_)
        self.assertIn("davi/logit_lens/prompts_logits.json", locals_)
        self.assertIn("gustavo/dataset_degraded_ft.jsonl", locals_)
        self.assertIn("gustavo/dataset_safe_ft.jsonl", locals_)
        self.assertEqual(len([x for x in locals_ if x.startswith("rk/data/") and x.endswith(".jsonl")]), 11)
        self.assertTrue(all(len(i.sha256) == 64 for i in items))


def E(i, n, d=False):
    return Entry(i, n, d, "Nov 5")


class GustavoTest(unittest.TestCase):
    def setUp(self):
        self.dest = Path(tempfile.mkdtemp())

    def drive(self, gustavo_children, extra=None):
        tree = {fetch.DRIVE_ROOT: [E("G", "Gustavo", True)], "G": gustavo_children}
        tree.update(extra or {})
        files = {}
        for entries in tree.values():
            for e in entries:
                if not e.is_dir:
                    files[e.id] = e.name.encode()
        return FakeDrive(tree, files)

    def test_empty_folder(self):
        d = self.drive([])
        self.assertEqual(fetch.fetch_gustavo(self.dest, d.lister, d.downloader), {})

    def test_finds_out_and_ignores_old_files(self):
        d = self.drive(
            [E("O", "out", True), E("S", "out_smoke", True), E("r1", "icl_zero_shot.jsonl")],
            {"O": [E("t", "test.jsonl"), E("ev", "eval", True), E("ic", "icl", True)],
             "ev": [E("s", "summary.csv"), E("j", "judged_zero_shot.jsonl")],
             "ic": [E("sh", "shots_safe.jsonl")],
             "S": [E("bad", "test.jsonl")]})
        got = fetch.fetch_gustavo(self.dest, d.lister, d.downloader)
        self.assertEqual(set(got), {"test.jsonl", "eval/summary.csv", "eval/judged_zero_shot.jsonl",
                                    "icl/shots_safe.jsonl"})
        self.assertEqual(got["test.jsonl"].read_bytes(), b"test.jsonl")
        self.assertNotIn("bad", d.downloads)
        self.assertNotIn("r1", d.downloads)

    def test_prefers_out2(self):
        d = self.drive([E("O", "out", True), E("O2", "out2", True)],
                       {"O": [E("t1", "test.jsonl")], "O2": [E("t2", "test.jsonl")]})
        fetch.fetch_gustavo(self.dest, d.lister, d.downloader)
        self.assertEqual(d.downloads, ["t2"])

    def test_offline_returns_what_is_cached(self):
        d = self.drive([E("O", "out", True)], {"O": [E("t", "test.jsonl")]})
        fetch.fetch_gustavo(self.dest, d.lister, d.downloader)
        d.online = False
        got = fetch.fetch_gustavo(self.dest, d.lister, d.downloader)
        self.assertEqual(set(got), {"test.jsonl"})


if __name__ == "__main__":
    unittest.main()
