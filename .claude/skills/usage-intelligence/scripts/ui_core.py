#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""usage-intelligence 核心：transcript 解析、用量统计、收益账本、检测规则、hook 入口。

入口（stdin 均为 Claude Code hook JSON）:
  python ui_core.py monitor        # UserPromptSubmit: 检测并注入上下文
  python ui_core.py precompact     # PreCompact: 记账（压缩时点快照）
  python ui_core.py sessionstart   # SessionStart: 压缩/恢复后注入锚点
  python ui_core.py report         # 人工运行: 收益统计
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Windows 下 hook 的 stdin/stdout 必须是 UTF-8（Claude Code 按 UTF-8 解 JSON）
for _s in (sys.stdin, sys.stdout):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

STATE_DIR = ".claude/usage-intelligence"
LEDGER_FILE = "benefit-ledger.json"
BENEFITS_LOG = "benefits.md"
RULES_FILE = "rules.md"
STATE_MD = "session-state.md"

CONTEXT_WINDOW = 1_000_000
CACHE_DISCOUNT = 0.1
COMPRESS_THRESHOLD = 0.50
COMPRESS_REDLINE = 0.70
STATE_STALE_HOURS = 6
POST_COMPACT_CONTEXT = 50_000        # 压缩后每轮输入的经验值
ASSUMED_REMAINING_TURNS = 10         # 收益估算口径：压缩后假定再干 10 轮

DEFAULT_RULES = """# 使用策略规则（可编辑，立即生效）

## 会话分型（新会话开场定型）

| 分型 | 特征 | 上下文策略 |
|------|------|-----------|
| 排查/探索型 | 多轮实验、读大量产物/日志 | 允许高占用；Grep 定位后只读需要的行段；每阶段结束说"写状态" |
| 修复/实施型 | 目标明确，改代码、打补丁 | 优先新开会话；只带补丁文件+报告相关章节；不重读全量报告 |
| 文档/汇报型 | 写报告、PR、案例分享 | 新开会话；只读报告相关段落；产物先落盘再润色 |
| 简单问答型 | 一问一答 | 无需策略 |

## 压缩规则（先写状态，再压缩）

1. 在自然断点压：刚写完文档/跑完一组实验后压，摘要最薄最准
2. 实验/探索中途不要压：正在追的线索先写入状态文件再压
3. 建议线 >=50% 且处于自然断点：建议压缩
4. 红区线 >=70%：必须先写状态文件再压缩
5. 压缩后回到 Grep-定位 + 精准 Read 纪律
6. 旧细节会稀释注意力（lost in the middle）——压缩后聚焦通常更好，前提是状态有文件兜底
"""


# ---------------------------------------------------------------------------
# Transcript 解析
# ---------------------------------------------------------------------------

@dataclass
class UsageStats:
    n_user: int = 0
    n_assistant: int = 0
    input_tokens: int = 0            # 累计（不含 cache 部分）
    cache_creation: int = 0
    cache_read: int = 0
    output_tokens: int = 0
    last_usage: dict = field(default_factory=dict)
    context_usage: float = 0.0       # 最近一轮 (input+cache_read+cache_creation+output)/窗口
    effective_input: int = 0         # 最近一轮真实计费输入（cache 按 1/10 折算）
    n_bash: int = 0
    n_read: int = 0


def parse_transcript(path) -> UsageStats:
    stats = UsageStats()
    if not path:
        return stats
    p = Path(path)
    if not p.is_file():
        return stats
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = rec.get("message") or {}
            role = msg.get("role")
            if role == "user":
                stats.n_user += 1
            elif role == "assistant":
                stats.n_assistant += 1
                content = msg.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_use":
                            name = item.get("name", "")
                            if name == "Bash":
                                stats.n_bash += 1
                            elif name == "Read":
                                stats.n_read += 1
            u = msg.get("usage")
            if isinstance(u, dict) and u.get("input_tokens") is not None:
                stats.input_tokens += u.get("input_tokens", 0)
                stats.cache_creation += u.get("cache_creation_input_tokens", 0)
                stats.cache_read += u.get("cache_read_input_tokens", 0)
                stats.output_tokens += u.get("output_tokens", 0)
                stats.last_usage = u
    u = stats.last_usage
    if u:
        ctx = (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
               + u.get("cache_creation_input_tokens", 0) + u.get("output_tokens", 0))
        stats.context_usage = ctx / CONTEXT_WINDOW
        stats.effective_input = int(u.get("input_tokens", 0)
                                    + u.get("cache_read_input_tokens", 0) * CACHE_DISCOUNT)
    return stats


# ---------------------------------------------------------------------------
# 状态目录与 JSON IO
# ---------------------------------------------------------------------------

def state_dir(project_dir=None) -> Path:
    if project_dir:
        return Path(project_dir) / STATE_DIR
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env:
        return Path(env) / STATE_DIR
    return Path.cwd() / STATE_DIR


def _read_json(path: Path, default):
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def ensure_rules(project_dir=None) -> str:
    d = state_dir(project_dir)
    p = d / RULES_FILE
    if not p.exists():
        d.mkdir(parents=True, exist_ok=True)
        p.write_text(DEFAULT_RULES, encoding="utf-8")
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 收益账本
# ---------------------------------------------------------------------------

def load_ledger(project_dir=None) -> dict:
    return _read_json(state_dir(project_dir) / LEDGER_FILE, {"version": 1, "entries": []})


def add_benefit(entry: dict, project_dir=None) -> None:
    entry.setdefault("ts", time.time())
    entry.setdefault("session", "unknown")
    d = state_dir(project_dir)
    led = load_ledger(project_dir)
    led["entries"].append(entry)
    _write_json(d / LEDGER_FILE, led)
    saved = entry.get("est_saved_total", 0)
    if saved > 0:
        md = d / BENEFITS_LOG
        md.parent.mkdir(parents=True, exist_ok=True)
        how = entry.get("note", entry.get("kind", ""))
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry["ts"]))
        with md.open("a", encoding="utf-8") as f:
            f.write(f"- {stamp} [{entry.get('kind', '?')}] {how} — 估算省 {saved:,} token\n")


def totals(project_dir=None) -> dict:
    led = load_ledger(project_dir)
    entries = led["entries"]
    by_kind = {}
    for e in entries:
        by_kind[e.get("kind", "?")] = by_kind.get(e.get("kind", "?"), 0) + e.get("est_saved_total", 0)
    return {
        "n_entries": len(entries),
        "total_est_saved": sum(e.get("est_saved_total", 0) for e in entries),
        "by_kind": by_kind,
    }


# ---------------------------------------------------------------------------
# 检测
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    kind: str
    level: str = "info"              # info / advice / warn
    message: str = ""
    est_saved_per_turn: int = 0
    est_saved_total: int = 0
    detail: dict = field(default_factory=dict)


def est_compress_savings(stats: UsageStats) -> int:
    """压缩后每轮输入节省（cache 折算口径）。"""
    if not stats.last_usage:
        return 0
    return max(0, stats.effective_input - POST_COMPACT_CONTEXT)


def state_file_age_h(project_dir=None):
    d = state_dir(project_dir)
    p = d / STATE_MD
    if not p.exists():
        return None
    return (time.time() - p.stat().st_mtime) / 3600


def check_compact_advice(stats: UsageStats, state_age_h, n_user: int) -> Detection | None:
    cu = stats.context_usage
    if cu < COMPRESS_THRESHOLD:
        return None
    fresh = state_age_h is not None and state_age_h <= STATE_STALE_HOURS
    per_turn = est_compress_savings(stats)
    if cu >= COMPRESS_REDLINE:
        if fresh:
            return Detection(
                "compact-advice", "warn",
                f"上下文 {cu:.0%} 已过红区线（70%）。状态文件 {state_age_h:.1f}h 前更新过——"
                f"线索已有文件兜底，现在压缩最划算（摘要薄、回本快）。回复 /compact 即可。",
                est_saved_per_turn=per_turn,
                est_saved_total=per_turn * ASSUMED_REMAINING_TURNS,
                detail={"context_usage": round(cu, 4), "state_age_h": state_age_h})
        return Detection(
            "compact-advice", "warn",
            f"上下文 {cu:.0%} 已过红区线（70%）且无新鲜状态文件。"
            f"先说「写状态」把当前线索落盘，再 /compact，避免摘要丢线索。",
            est_saved_per_turn=per_turn,
            est_saved_total=per_turn * ASSUMED_REMAINING_TURNS,
            detail={"context_usage": round(cu, 4), "state_age_h": state_age_h})
    if n_user >= 4:
        if fresh:
            return Detection(
                "compact-advice", "advice",
                f"上下文 {cu:.0%} 过建议线（50%），状态文件 {state_age_h:.1f}h 前更新。"
                f"若本阶段收尾，建议现在压缩：预计每轮省 {per_turn:,} token。",
                est_saved_per_turn=per_turn,
                est_saved_total=per_turn * ASSUMED_REMAINING_TURNS,
                detail={"context_usage": round(cu, 4)})
        return Detection(
            "compact-advice", "advice",
            f"上下文 {cu:.0%} 过建议线（50%）。若正追线索（实验中途），先说「写状态」；"
            f"若已到自然断点，直接 /compact（预计每轮省 {per_turn:,} token）。",
            est_saved_per_turn=per_turn,
            est_saved_total=per_turn * ASSUMED_REMAINING_TURNS,
            detail={"context_usage": round(cu, 4)})
    return None


SESSION_TYPE_KEYWORDS = {
    "修复型": (["修复", "fix", "补丁", "patch", "改代码", "打补丁", "提PR", "提 PR"],
               "目标明确——只带补丁文件+报告相关章节（新开会话更省），不要重读全量报告"),
    "排查型": (["排查", "定位", "为什么", "debug", "根因", "分析日志", "误差", "精度问题"],
               "上下文会涨得快——Grep 定位后只读需要的行段；每阶段结束说「写状态」固化线索"),
    "文档型": (["报告", "总结", "案例", "文档", "分享", "汇报", "小结"],
               "产物先落盘再润色；只读报告相关段落，不整读"),
}


def check_session_type(n_user: int, prompt: str) -> Detection | None:
    if n_user > 3:
        return None
    for t, (words, tip) in SESSION_TYPE_KEYWORDS.items():
        if any(w in prompt for w in words):
            return Detection("session-type", "info", f"[分型：{t}] {tip}")
    return None


def _bag(s: str) -> set:
    chars = {ch for ch in s if '一' <= ch <= '鿿'}
    words = {w.lower() for w in "".join(c if c.isalnum() else " " for c in s).split()}
    return chars | words


def check_skillify(prompt: str, n_user: int, project_dir=None) -> Detection | None:
    """相似任务模式 >=3 次时提醒 skill 化。"""
    if n_user < 3:
        return None
    d = state_dir(project_dir)
    p = d / "skillify_candidates.json"
    raw = _read_json(p, {"prompts": []})
    raw["prompts"].append({"prompt": prompt[:300], "ts": time.time()})
    raw["prompts"] = raw["prompts"][-200:]
    _write_json(p, raw)
    cur = _bag(prompt)
    if not cur:
        return None
    sims = []
    for rec in raw["prompts"][:-1]:
        b = _bag(rec.get("prompt", ""))
        if b:
            sims.append(len(cur & b) / max(1, len(cur | b)))
    if len(sims) < 3:
        return None
    top5 = sorted(sims, reverse=True)[:5]
    avg_top = sum(top5) / len(top5)
    if avg_top >= 0.45:
        return Detection(
            "skillify", "advice",
            f"检测到相似任务模式（top-5 相似度 {avg_top:.0%}，历史 >=3 次）——"
            f"可 skill 化候选。要封装成技能吗？（说「封装」进入流程；忽略则跳过）",
            detail={"avg_top5_sim": round(avg_top, 3), "n_history": len(sims)})
    return None


# ---------------------------------------------------------------------------
# Hook 入口
# ---------------------------------------------------------------------------

def _emit_context(event: str, lines: list) -> None:
    if not lines:
        return
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": event,
            "additionalContext": "\n".join(lines),
        }
    }, ensure_ascii=False))


def cmd_monitor(data: str) -> int:
    try:
        inp = json.loads(data or "{}")
    except json.JSONDecodeError:
        return 0
    tpath = inp.get("transcript_path") or ""
    prompt = inp.get("prompt") or ""
    project = inp.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR")
    ensure_rules(project)
    stats = parse_transcript(tpath)
    n_user = stats.n_user + 1                    # 当前 prompt 尚未落盘
    detections = []
    d = check_compact_advice(stats, state_file_age_h(project), n_user)
    if d:
        detections.append(d)
    d = check_session_type(n_user, prompt)
    if d:
        detections.append(d)
    d = check_skillify(prompt, n_user, project)
    if d:
        detections.append(d)
    lines = []
    for det in detections:
        tag = {"info": "[提示]", "advice": "[建议]", "warn": "[警告]"}[det.level]
        lines.append(f"{tag} {det.message}")
    _emit_context("UserPromptSubmit", lines)
    if detections:
        add_benefit({
            "kind": "advice-shown",
            "session": inp.get("session_id", "unknown"),
            "note": " / ".join(d.message[:80] for d in detections),
            "est_saved_total": 0,                 # 提醒本身不省，采纳后由 precompact 记账
        }, project)
    return 0


def cmd_precompact(data: str) -> int:
    """压缩时点记账：快照上下文占用 + 估算收益 + 状态文件是否兜底。"""
    try:
        inp = json.loads(data or "{}")
    except json.JSONDecodeError:
        return 0
    tpath = inp.get("transcript_path") or ""
    project = inp.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR")
    stats = parse_transcript(tpath)
    age = state_file_age_h(project)
    per_turn = est_compress_savings(stats)
    add_benefit({
        "kind": "compact-done",
        "session": inp.get("session_id", "unknown"),
        "note": (f"压缩时上下文 {stats.context_usage:.0%}，"
                 f"状态文件{'新鲜(' + f'{age:.1f}h)' if age is not None and age <= STATE_STALE_HOURS else '陈旧或缺失'}，"
                 f"trigger={inp.get('trigger', 'manual')}"),
        "context_usage": round(stats.context_usage, 4),
        "state_fresh": bool(age is not None and age <= STATE_STALE_HOURS),
        "est_saved_per_turn": per_turn,
        "est_saved_total": per_turn * ASSUMED_REMAINING_TURNS,
    }, project)
    return 0


def cmd_sessionstart(data: str) -> int:
    """压缩/恢复后注入锚点：状态文件位置 + 规则指针 + 累计收益。"""
    try:
        inp = json.loads(data or "{}")
    except json.JSONDecodeError:
        return 0
    project = inp.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR")
    source = inp.get("source", inp.get("matcher", ""))
    d = state_dir(project)
    ensure_rules(project)
    age = state_file_age_h(project)
    t = totals(project)
    lines = []
    if source in ("compact", "resume", "clear"):
        state_line = (f"状态文件 {d / STATE_MD}（{age:.1f}h 前更新）——恢复工作时先读它，不要凭记忆猜线索"
                      if age is not None else
                      f"无状态文件。若上一会话有未完成线索，让用户指出或从报告文件恢复")
        lines.append(
            f"[usage-intelligence] 会话来源：{source}。{state_line}。"
            f"使用规则见 {d / RULES_FILE}（分型/压缩/精准读取纪律）。累计估算收益：{t['total_est_saved']:,} token。"
            f"继续当前任务即可，无需向用户复述本提示。")
    _emit_context("SessionStart", lines)
    return 0


def cmd_report(_data: str) -> int:
    t = totals()
    print(f"=== usage-intelligence 收益统计 ===")
    print(f"记录条数: {t['n_entries']}")
    print(f"累计估算节省: {t['total_est_saved']:,} token（口径：压缩每轮节省 x {ASSUMED_REMAINING_TURNS} 轮假设）")
    for kind, v in t["by_kind"].items():
        print(f"  {kind}: {v:,}")
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "monitor"
    data = sys.stdin.read()
    if cmd == "monitor":
        return cmd_monitor(data)
    if cmd == "precompact":
        return cmd_precompact(data)
    if cmd == "sessionstart":
        return cmd_sessionstart(data)
    if cmd == "report":
        return cmd_report(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
