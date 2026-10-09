import subprocess
import sys
import unittest
from pathlib import Path

# 中文 SKILL.md 输出必须按 UTF-8 解码；text=True 不带 encoding 时按 locale（Windows GBK）
# 解码会在 PYTHONIOENCODING=utf-8 的子进程输出上炸 UnicodeDecodeError（stdout 变 None）。
SUBPROCESS_KWARGS = {
    "capture_output": True,
    "text": True,
    "encoding": "utf-8",
    "errors": "replace",
    "cwd": "",
    "check": False,
}


class TestSkillCli(unittest.TestCase):
    def _run_cli(self, *args):
        project_root = Path(__file__).resolve().parents[1]
        kwargs = dict(SUBPROCESS_KWARGS)
        kwargs["cwd"] = str(project_root)
        return subprocess.run(
            [sys.executable, "-m", "pytorch_yongli_skill", "--root", str(project_root), *args],
            **kwargs,
        )

    def test_show_pr_review(self):
        result = self._run_cli("show", "pr-review")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("name: pr-review", result.stdout)
        self.assertIn("description:", result.stdout)
        self.assertIn("entrypoint:", result.stdout)

    def test_show_missing_skill(self):
        result = self._run_cli("show", "not-a-skill")

        self.assertEqual(result.returncode, 1)
        self.assertIn("not found", result.stdout)

    def test_read_pr_review(self):
        result = self._run_cli("read", "pr-review")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PyTorch PR Review Skill", result.stdout)
        self.assertIn("Usage Modes", result.stdout)

    def test_read_chinese_skill(self):
        # 新增中文 skill 的 read 路径（GBK 环境回归）
        result = self._run_cli("read", "red-green-validation")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("red-green-validation", result.stdout)

    def test_run_pr_review(self):
        result = self._run_cli("run", "pr-review")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("pr-review", result.stdout)
        self.assertIn("SKILL.md", result.stdout)

    def test_run_refactoring_help(self):
        result = self._run_cli(
            "run", "pytorch-test-refactoring", "--", "--help",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PyTorch Test Refactoring Orchestrator", result.stdout)


if __name__ == "__main__":
    unittest.main()
