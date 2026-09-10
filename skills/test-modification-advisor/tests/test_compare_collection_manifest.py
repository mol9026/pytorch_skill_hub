import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_collection_manifest.py"
SPEC = importlib.util.spec_from_file_location("compare_collection_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompareCollectionManifestTest(unittest.TestCase):
    def test_loads_text_and_json_and_compares(self):
        with tempfile.TemporaryDirectory() as directory:
            before = Path(directory) / "before.txt"
            after = Path(directory) / "after.json"
            before.write_text("# collected\na::test_cpu\na::test_cuda\n", encoding="utf-8")
            after.write_text('["a::test_cuda", "a::test_xpu"]', encoding="utf-8")
            result = MODULE.compare(MODULE.load_manifest(str(before)), MODULE.load_manifest(str(after)))
        self.assertEqual(result["removed"], ["a::test_cpu"])
        self.assertEqual(result["added"], ["a::test_xpu"])
        self.assertEqual(result["unchanged"], ["a::test_cuda"])

    def test_rejects_invalid_json_shape(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as f:
            f.write('{"wrong": []}')
            f.flush()
            with self.assertRaises(ValueError):
                MODULE.load_manifest(f.name)


if __name__ == "__main__":
    unittest.main()
