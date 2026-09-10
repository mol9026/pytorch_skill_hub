#!/usr/bin/env python3
# mypy: ignore-errors
"""scan_combined.py — 合并扫描一个本地 PyTorch 测试文件的两套机制特征。

调用两个外部 skill 的 scan 脚本（只调用，不复制，与 PR-review-spec-check 同模式）：
    ../capability-migrator/scripts/scan_flags.py                     → 能力门控发现
    ../hw-classification-annotator/scripts/scan_hw_classification.py → hw_classification 发现
各外部脚本的原生 markdown 输出全量保留（含 hw 的约束检查摘要），再补一段
「交叉点」—— 同时影响两机制的项（硬编码设备候选 / instantiate / 迁移状态）。

只做检测，不做判断（判断知识在外部 references，交给 agent 查）。
输出 stdout markdown「合并发现清单」，供 test-modification-advisor 的 modify-step-1 使用。

用法:
    python3 scan_combined.py <test_file.py>
    python3 scan_combined.py <test_file.py> --json
"""
import argparse
import json
import os
import subprocess
import sys

# Windows 控制台默认 GBK，强制 stdout 用 UTF-8 输出（含外部 scan 的原生 markdown）
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 兄弟 skill 目录默认与本 skill 同级；可用环境变量覆盖，便于非同级部署。
PARENT_DIR = os.environ.get("PYTORCH_TEST_SKILLS_DIR", os.path.dirname(SKILL_DIR))

# 外部 skill 的 scan 脚本（只调用，不复制）；路径可用环境变量单独覆盖。
CAP_SCAN = os.environ.get(
    "CAPABILITY_MIGRATOR_SCAN",
    os.path.join(PARENT_DIR, "capability-migrator", "scripts", "scan_flags.py"),
)
HW_SCAN = os.environ.get(
    "HW_CLASSIFICATION_ANNOTATOR_SCAN",
    os.path.join(PARENT_DIR, "hw-classification-annotator", "scripts", "scan_hw_classification.py"),
)


def _run_scan(script, target, json_mode):
    """调用外部 scan 脚本，返回 (ok, output)。

    json_mode=True 时 output 为解析后的 dict；False 时为原始 markdown 文本。
    外部脚本缺失 / 执行失败 → ok=False，output 为错误说明。
    """
    if not os.path.isfile(script):
        return False, f"外部 scan 不可用：未找到 {script}"
    args = [sys.executable, script, target]
    if json_mode:
        args.append("--json")
    try:
        # PYTHONIOENCODING=utf-8：外部脚本在管道里统一按 UTF-8 输出（Windows 默认 GBK 会乱）
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run(args, capture_output=True, text=True, timeout=120,
                           encoding="utf-8", errors="replace", env=env)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"外部 scan 执行失败：{e}"
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        return False, f"外部 scan 返回码 {r.returncode}: {err[:300]}"
    if json_mode:
        try:
            return True, json.loads(r.stdout)
        except json.JSONDecodeError as e:
            return False, f"外部 scan JSON 解析失败：{e}"
    return True, r.stdout


def _cap_summary(cap):
    """能力发现的概览摘要（capability scan JSON → 计数）。"""
    flags_total = sum(len(v) for names in cap.get("flags", {}).values()
                      for v in names.values())
    return {
        "total_lines": cap.get("total_lines", 0),
        "classes": len(cap.get("classes", [])),
        "flags_total": flags_total,
        "gates": len(cap.get("gates", {})),
        "skip_decorators": len(cap.get("skip_decorators", [])),
        "hardcoded_devices": len(cap.get("hardcoded_devices", [])),
        "instantiate_calls": len(cap.get("instantiate_calls", [])),
        "migrated_markers": len(cap.get("migrated_markers", [])),
    }


def _hw_summary(hw):
    """hw_classification 发现的概览摘要（hw scan JSON → 计数）。"""
    classes = hw.get("classes", [])
    annotated = sum(1 for c in classes if c.get("hw_classification"))
    return {
        "total_lines": hw.get("total_lines", 0),
        "classes": len(classes),
        "annotated": annotated,
        "missing": len(classes) - annotated,
        "hardcoded_devices": len(hw.get("hardcoded_devices", [])),
        "module_level_tests": len(hw.get("module_level_test_functions", [])),
    }


def _render_overview(cap_ok, cap, hw_ok, hw):
    """渲染概览（0 节）。"""
    out = ["## 0. 概览"]
    if cap_ok:
        c = _cap_summary(cap)
        out.append(
            f"- 总行数 {c['total_lines']} ｜ 测试类 {c['classes']} ｜ "
            f"能力 flag 引用 {c['flags_total']} ｜ capability gates {c['gates']} ｜ skip/xfail 装饰器 {c['skip_decorators']} ｜ "
            f"硬编码设备候选 {c['hardcoded_devices']} ｜ instantiate 调用 {c['instantiate_calls']} ｜ "
            f"已迁移标记 {c['migrated_markers']}"
        )
    else:
        out.append(f"- 能力门控扫描：{cap}")
    if hw_ok:
        h = _hw_summary(hw)
        out.append(
            f"- 总行数 {h['total_lines']} ｜ 测试类 {h['classes']} ｜ "
            f"hw 已标注 {h['annotated']} / 缺失 {h['missing']} ｜ "
            f"硬编码设备候选 {h['hardcoded_devices']} ｜ 模块级 test_* 函数 {h['module_level_tests']}"
        )
    else:
        out.append(f"- hw_classification 扫描：{hw}")
    return "\n".join(out)


def _render_crossref(cap_ok, cap, hw_ok, hw):
    """渲染交叉点（3 节）：两机制协调改动的项。"""
    out = ["## 3. 交叉点（两机制协调改动）"]
    out.append("> 以下项目**同时影响**能力门控与 hw_classification，出修改建议时只改一次：")
    out.append("")

    # 硬编码设备候选
    if cap_ok and hw_ok:
        cap_devs = cap.get("hardcoded_devices", [])
        hw_devs = hw.get("hardcoded_devices", [])
        cap_lines = {ln for ln, _ in cap_devs}
        hw_lines = {d["line"] for d in hw_devs}
        both = sorted(cap_lines & hw_lines)
        out.append(
            f"- **硬编码设备候选**：capability scan {len(cap_devs)} 处 / hw scan {len(hw_devs)} 处；"
            f"两 scan 均检出的行 {len(both)} 处 {both or '（无重叠）'}。"
            "结合 classification、实例化范围与设备专属语义判断；device-agnostic 路径改用注入设备。"
        )
    elif cap_ok:
        out.append(f"- **硬编码设备候选**：capability scan {len(cap.get('hardcoded_devices', []))} 处（hw scan 不可用）。")
    elif hw_ok:
        out.append(f"- **硬编码设备候选**：hw scan {len(hw.get('hardcoded_devices', []))} 处（capability scan 不可用）。")
    else:
        out.append("- **硬编码设备候选**：两外部 scan 均不可用，无法交叉核对。")

    # instantiate 调用
    if cap_ok:
        inst_lines = cap.get("instantiate_calls", [])
        out.append(f"- **instantiate 调用**：capability scan 检出 @ 行 {inst_lines}。"
                   + ("hw 分类决定 `only_for`/`except_for` 形式，capability 的设备 flag 也映射到此 → 统一确定参数。"
                      if inst_lines else "文件未调用 `instantiate_device_type_tests` —— 若含设备测试，迁移时需补。"))
    if hw_ok:
        inst_classes = []
        for cls in hw.get("classes", []):
            inst = cls.get("instantiate_call")
            if inst:
                tag = f"only_for={inst.get('only_for')}" if inst.get("only_for") else (
                    f"except_for={inst.get('except_for')}" if inst.get("except_for") else "无参数")
                inst_classes.append(f"`{cls['name']}`(L{inst['line']}) {inst['fn']} {tag}")
        out.append(f"- **instantiate 逐类详情**：{'；'.join(inst_classes) if inst_classes else '（无）'}。")

    # 迁移 / 标注状态
    if cap_ok:
        migrated = cap.get("migrated_markers", [])
        out.append(f"- **能力迁移状态**：已迁移标记 @ 行 {migrated or '（无 — 文件未开始能力迁移）'}。")
    if hw_ok:
        missing = [c["name"] for c in hw.get("classes", []) if not c.get("hw_classification")]
        out.append(f"- **hw 标注状态**：缺失标注的类 {len(missing)} 个 {missing or ''}。")
    return "\n".join(out)


def render_combined(path, cap, cap_ok, hw, hw_ok, cap_md, cap_md_ok, hw_md, hw_md_ok):
    """合并渲染：概览/交叉点用 JSON 数据，正文用外部脚本原生 markdown。"""
    out = []
    out.append(f"# 合并发现清单 — `{path}`")
    out.append("")
    out.append("> 由 test-modification-advisor 生成。能力扫描：`scan_flags.py`（原生输出）｜ "
               "hw 扫描：`scan_hw_classification.py`（原生输出）。")
    out.append("")
    out.append(_render_overview(cap_ok, cap, hw_ok, hw))
    out.append("")
    out.append("\n## 1. 能力门控发现（scan_flags.py 原生输出）")
    out.append("")
    out.append("```")
    out.append((cap_md if cap_md_ok else cap).rstrip())
    out.append("```")
    out.append("")
    out.append("\n## 2. hw_classification 发现（scan_hw_classification.py 原生输出）")
    out.append("")
    out.append("```")
    out.append((hw_md if hw_md_ok else hw).rstrip())
    out.append("```")
    out.append("")
    out.append(_render_crossref(cap_ok, cap, hw_ok, hw))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(
        description="Scan a PyTorch test file for BOTH capability gating and hw_classification features.")
    ap.add_argument("file", help="path to the test .py file")
    ap.add_argument("--json", action="store_true", help="emit merged JSON instead of markdown")
    args = ap.parse_args()

    # 原生 markdown（含 hw 的约束检查摘要，全量保留）+ JSON（概览/交叉点计算）
    cap_md_ok, cap_md = _run_scan(CAP_SCAN, args.file, json_mode=False)
    hw_md_ok, hw_md = _run_scan(HW_SCAN, args.file, json_mode=False)
    cap_ok, cap = _run_scan(CAP_SCAN, args.file, json_mode=True)
    hw_ok, hw = _run_scan(HW_SCAN, args.file, json_mode=True)

    if args.json:
        merged = {
            "path": args.file,
            "capability": cap if cap_ok else {"_error": cap},
            "hw_classification": hw if hw_ok else {"_error": hw},
            "capability_md": cap_md if cap_md_ok else {"_error": cap_md},
            "hw_classification_md": hw_md if hw_md_ok else {"_error": hw_md},
        }
        print(json.dumps(merged, ensure_ascii=False, indent=2))
        return

    print(render_combined(args.file, cap, cap_ok, hw, hw_ok, cap_md, cap_md_ok, hw_md, hw_md_ok))


if __name__ == "__main__":
    main()
