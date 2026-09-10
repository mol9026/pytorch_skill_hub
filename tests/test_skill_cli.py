import subprocess
import sys
import unittest
from pathlib import Path


class TestSkillCli(unittest.TestCase):
    def test_show_pr_review(self):
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "pytorch_yongli_skill", "--root", str(project_root), "show", "pr-review"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("name: pr-review", result.stdout)
        self.assertIn("description:", result.stdout)
        self.assertIn("entrypoint:", result.stdout)

    def test_show_missing_skill(self):
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "pytorch_yongli_skill", "--root", str(project_root), "show", "not-a-skill"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            check=False,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("not found", result.stdout)

    def test_read_pr_review(self):
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "pytorch_yongli_skill", "--root", str(project_root), "read", "pr-review"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PyTorch PR Review Skill", result.stdout)
        self.assertIn("Usage Modes", result.stdout)

    def test_run_pr_review(self):
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "pytorch_yongli_skill", "--root", str(project_root), "run", "pr-review"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("pr-review", result.stdout)
        self.assertIn("SKILL.md", result.stdout)

    def test_run_refactoring_help(self):
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytorch_yongli_skill",
                "--root",
                str(project_root),
                "run",
                "pytorch-test-refactoring",
                "--",
                "--help",
            ],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PyTorch Test Refactoring Orchestrator", result.stdout)


if __name__ == "__main__":
    unittest.main()
