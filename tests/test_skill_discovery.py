import subprocess
import sys
import unittest
from pathlib import Path

from pytorch_yongli_skill.registry import DEFAULT_REGISTRY, discover_local_skills, register_local_skills


class TestSkillDiscovery(unittest.TestCase):
    def test_discover_local_skills(self):
        project_root = Path(__file__).resolve().parents[1]
        specs = discover_local_skills(project_root)

        names = {spec.name for spec in specs}
        expected = {
            "pytorch-test-refactoring",
            "test-modification-advisor",
            "torch-npu-test-failure-diagnose",
            "pr-review",
        }

        self.assertTrue(expected.issubset(names))

    def test_register_local_skills(self):
        project_root = Path(__file__).resolve().parents[1]
        registry = DEFAULT_REGISTRY.__class__()

        register_local_skills(registry, project_root)

        self.assertTrue(registry.names())
        self.assertIn("pr-review", registry.names())

    def test_cli_list(self):
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "pytorch_yongli_skill", "--root", str(project_root), "list"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("pr-review", result.stdout)
        self.assertIn("torch-npu-test-failure-diagnose", result.stdout)


if __name__ == "__main__":
    unittest.main()
