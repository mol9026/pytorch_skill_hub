#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model-switch-guard 检测核心：从 CC Switch proxy 日志分类模型切换事件。

单一观测点原则：不监听 /model 命令、不监听 CC Switch UI，只观测实际请求的
路由结果（proxy_request_logs 表），原因有无数种，出口只有一个。

事件分类（含误报过滤，全部来自本机 23 天实测数据）：
  PROVIDER_SWITCH  主链模型变化（跨供应商/跨模型）——报损提醒 + 记账
  RETRY_FALLBACK   同 request_model 失败后换目标重试 ——合并记账，轻提醒
  GATEWAY_JITTER   同秒来回抖动（网关内部分段/辅助调用）——静默
  SUBAGENT         subagent 模型白名单（独立上下文，不伤主缓存）——静默

用法：
  python detector.py scan                     # 全量扫描，输出 JSON 事件流
  python detector.py since <iso时间>          # 增量扫描
  python detector.py report [--days N]        # 汇总报表
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta

# CC Switch 数据库位置（Windows 默认；环境变量可覆盖）
DB_PATH = os.environ.get(
    "CC_SWITCH_DB",
    os.path.expandvars(r"%USERPROFILE%\.cc-switch\cc-switch.db"),
)

# subagent/辅助模型白名单：独立上下文或网关辅助实例，切换不伤主会话缓存。
# 实测依据（本机 23 天 4256 请求）：
#   MiniMax-M3       —— subagent 配置路由，每次请求 <1.2k token 独立上下文
#   qwen3.8-flash    —— haiku 档映射的辅助调用
#   glm-5.3          —— 网关把部分请求（预检/分类）路由到的辅助实例，
#                       其出现点前后 2-70s 内主链 glm-5-3 请求照常（时间上嵌入而非替代）
SUBAGENT_MODELS = {
    m.strip()
    for m in os.environ.get(
        "MSG_SUBAGENT_MODELS", "MiniMax-M3,qwen3.8-flash,glm-5.3"
    ).split(",")
    if m.strip()
}

# 失败请求（status != 200 且 ctx=0）不改变主链：网关映射失败时 model 会回显
# request_model 原名（如 claude-sonnet-4-6 429/403），不是真实路由变化

# 抖动识别：同一对模型来回且间隔 <= 此秒数视为网关内部分段
JITTER_WINDOW_S = 5
# 重试识别：失败后换目标重发的窗口
RETRY_WINDOW_S = 120
# 主链会话中，单请求 ctx 超过此值才值得提醒（小上下文切换损失可忽略）
ALERT_CTX_THRESHOLD = 50_000


def _rows(conn, since=None):
    q = (
        "SELECT session_id, request_model, model, "
        "input_tokens + cache_read_tokens + cache_creation_tokens AS ctx, "
        "status_code, created_at, error_message "
        "FROM proxy_request_logs WHERE app_type='claude'"
    )
    args = []
    if since:
        q += " AND created_at >= ?"
        args.append(since)
    q += " ORDER BY rowid"
    return conn.execute(q, args)


def _parse_ts(s):
    # created_at 有两种形态：unix 秒（int/数字串）或 ISO 字符串
    try:
        return datetime.fromtimestamp(int(s))
    except (ValueError, TypeError, OSError):
        pass
    s = str(s).replace(" ", "T")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def classify(conn, since=None):
    """按 session 内模型序列分类事件。返回事件列表（按时间序）。"""
    events = []
    # session -> 主链状态（排除白名单模型的最后主模型/时间）
    prev_main_model = {}
    prev_main_ts = {}
    last_fail = {}  # (session, request_model) -> (ts, failed_model)

    for sid, req_m, model, ctx, status, ca, err in _rows(conn, since):
        t = _parse_ts(ca)
        if t is None or not model:
            continue
        key = sid

        if model in SUBAGENT_MODELS:
            continue  # SUBAGENT：静默

        # 失败请求（ctx=0 且非 200）不进入主链：模型字段是映射失败回显，
        # 不是真实路由（实测：429/403/502 时 model 回显 request_model 原名）
        if (not status or status != 200) and not ctx:
            if status and status != 200:
                last_fail[(key, req_m)] = t
            continue

        pm = prev_main_model.get(key)
        pt = prev_main_ts.get(key)

        if pm is None:
            prev_main_model[key] = model
            prev_main_ts[key] = t
            continue

        if model != pm:
            dt = (t - pt).total_seconds() if pt else None
            failed_recent = last_fail.get((key, req_m))
            is_retry = (
                failed_recent is not None
                and dt is not None
                and (t - failed_recent).total_seconds() <= RETRY_WINDOW_S
            )

            if is_retry:
                events.append({
                    "type": "RETRY_FALLBACK", "ts": ca, "session": sid[:8],
                    "from": pm, "to": model, "request_model": req_m,
                    "ctx": ctx, "status": status,
                })
            elif dt is not None and dt <= JITTER_WINDOW_S and req_m and pm and (
                # 抖动：request_model 不变，目标在两个真实模型间同秒来回
                req_m.startswith("claude-")
            ):
                events.append({
                    "type": "GATEWAY_JITTER", "ts": ca, "session": sid[:8],
                    "from": pm, "to": model, "request_model": req_m,
                    "ctx": ctx, "status": status,
                })
                prev_main_model[key] = model
                prev_main_ts[key] = t
                continue
            else:
                events.append({
                    "type": "PROVIDER_SWITCH", "ts": ca, "session": sid[:8],
                    "from": pm, "to": model, "request_model": req_m,
                    "ctx": ctx or 0, "status": status,
                    "alert": (ctx or 0) >= ALERT_CTX_THRESHOLD,
                })

        prev_main_model[key] = model
        prev_main_ts[key] = t
        if status and status != 200:
            last_fail[(key, req_m)] = t
        elif (key, req_m) in last_fail:
            del last_fail[(key, req_m)]

    return events


def report(events, days=None):
    """汇总报表：按类型计数 + PROVIDER_SWITCH 的估算缓存损失。"""
    cutoff = None
    if days:
        cutoff = datetime.now().timestamp() - days * 86400
    if cutoff:
        kept = []
        for e in events:
            try:
                ts = float(e["ts"])
            except (TypeError, ValueError):
                ts = None
            if ts is None or ts >= cutoff:
                kept.append(e)
        events = kept

    agg = {}
    switch_loss_tokens = 0
    alerts = []
    for e in events:
        agg[e["type"]] = agg.get(e["type"], 0) + 1
        if e["type"] == "PROVIDER_SWITCH" and e.get("alert"):
            switch_loss_tokens += e["ctx"]
            alerts.append(e)

    out = {
        "window_days": days,
        "counts": agg,
        "switch_alert_count": len(alerts),
        "switch_ctx_tokens_at_risk": switch_loss_tokens,
        "note": "ctx_tokens_at_risk = 每次高上下文切换时缓存作废的 token 总量"
                "（按 cache 读价 ~10% 与全价的差估算实际损失 ≈ 0.9×该值）",
        "alerts": alerts[-10:],
    }
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if not os.path.isfile(DB_PATH):
        print(json.dumps({"error": f"cc-switch db not found: {DB_PATH}"}))
        return 2
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        if cmd == "scan":
            since = sys.argv[2] if len(sys.argv) > 2 else None
            events = classify(conn, since)
            print(json.dumps(events, ensure_ascii=False, indent=1))
        elif cmd == "report":
            days = int(sys.argv[2]) if len(sys.argv) > 2 else None
            events = classify(conn)
            print(json.dumps(report(events, days), ensure_ascii=False, indent=1))
        else:
            print(__doc__)
            return 1
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
