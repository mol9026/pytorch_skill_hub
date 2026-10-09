#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buffer_alias_tracker —— NPU 图模式产物内存复用审计（提案 D）

纯静态分析 output_code.py，检测 in-place reduction / reuse-after-delete /
读改写链 等危险内存复用模式。

关键观察：
  - `mutated_arg_names` 非空 -> kernel 会就地改写该参数
  - `# reuse` 注释 -> inductor 把"读完即弃"的 buffer 直接复用为输出
  - reduction kernel (num_reduction>0) + mutated -> 根因4 高危组合
  - 同一生存域内：A 写 -> B 读 -> A 被复用为 C 输出，形成 RAW/WAW 风险链

用法:
  python3 buffer_alias_tracker.py <output_code.py 或 run_dir>
  python3 buffer_alias_tracker.py --green <green_artifact> --red <red_artifact>
"""

import argparse
import json
import os
import re
import sys

RUN_LINE = re.compile(r"^(\s*)(\w+)\.run\((.*)\)\s*$")
REUSE_RE = re.compile(r"^(\s*)(buf\d+|arg\d+_\d+)\s*=\s*(buf\d+|arg\d+_\d+)\s*#\s*reuse")
META_RE = re.compile(
    r"'kernel_name': '([^']+)'[\s\S]*?"
    r"'mutated_arg_names': (\[[^\]]*\])[\s\S]*?"
    r"'num_reduction': (\d+)")


def split_args(s):
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


def parse_kernel_meta(src):
    """kernel 名 -> {mutated_arg_names, num_reduction}。"""
    meta = {}
    # 每个 kernel 的 meta 在 inductor_meta={...} 里，且同一行上有 kernel_name / num_reduction / mutated_arg_names
    for m in re.finditer(r"'kernel_name': '([^']+)'", src):
        kname = m.group(1)
        line_start = m.start()
        line_end = src.find("\n", line_start)
        seg = src[line_start:line_end]
        mut_match = re.search(r"'mutated_arg_names': (\[[^\]]*\])", seg)
        red_match = re.search(r"'num_reduction': (\d+)", seg)
        mutated = json.loads(mut_match.group(1).replace("'", '"')) if mut_match else []
        num_reduction = int(red_match.group(1)) if red_match else 0

        # 函数定义，用于映射 mutated 参数名 -> 位置
        def_pat = f"def {kname}("
        def_pos = src.find(def_pat, line_start)
        params = []
        if def_pos != -1:
            head_end = src.find(":", def_pos + len(def_pat))
            head = src[def_pos + len(def_pat): head_end]
            params = [p.strip().split("=")[0].split()[-1] for p in head.split(",")]
        meta[kname] = {"mutated": mutated, "num_reduction": num_reduction, "params": params}
    return meta


def parse_runner(src):
    """解析 Runner.call 体，返回事件序列。"""
    m = re.search(r"class Runner:.*def call\(self, args\):\n(.*?)(?:def |\Z)", src, re.S)
    if not m:
        return None
    events = []
    for i, ln in enumerate(m.group(1).split("\n"), 1):
        raw = ln.rstrip()
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        r = RUN_LINE.match(raw)
        if r:
            events.append(("run", {"kernel": r.group(2), "args": split_args(r.group(3)), "line": i, "raw": raw}))
            continue
        u = REUSE_RE.match(raw)
        if u:
            events.append(("reuse", {"new": u.group(2), "old": u.group(3), "line": i}))
            continue
    return events


def analyze(path):
    src = open(path, encoding="utf-8").read()
    meta = parse_kernel_meta(src)
    events = parse_runner(src)
    if events is None:
        raise SystemExit("未找到 Runner.call")

    alerts = []
    # 追踪每个 kernel 的写集（mutated 参数对应的 buffer 符号）
    # 以及 run 之间的读-写链

    # 1. 逐个 run 事件分析
    writes = {}   # symbol -> last write event
    reads = {}    # symbol -> list of read events
    reused_symbols = set()
    for ev in events:
        t, d = ev
        if t == "run":
            kname = d["kernel"]
            km = meta.get(kname, {"mutated": [], "num_reduction": 0, "params": []})
            args = d["args"]
            mutated_names = km["mutated"]
            # 把 mutated 参数名映射到 run 中位置
            mutated_symbols = []
            for mn in mutated_names:
                if mn in km["params"]:
                    idx = km["params"].index(mn)
                    if idx < len(args):
                        mutated_symbols.append(args[idx])

            # 规则 0: reduction kernel 的输入来自 #reuse 产物 —— 根因4 高危模式
            if km["num_reduction"] > 0:
                for a in args:
                    if a in reused_symbols:
                        alerts.append({
                            "rule": "reduction-input-from-reuse",
                            "severity": "high",
                            "line": d["line"],
                            "kernel": kname,
                            "msg": (f"reduction kernel {kname} 的输入 {a} 来自 #reuse 复用 —— "
                                    f"在 group dispatch 下存在读改写竞争风险（根因4类）")
                        })

            # 规则 1: reduction + mutated
            if km["num_reduction"] > 0 and mutated_symbols:
                alerts.append({
                    "rule": "reduction+inplace",
                    "severity": "high",
                    "line": d["line"],
                    "kernel": kname,
                    "msg": (f"reduction kernel ({km['num_reduction']} reduction) 声明 mutated 参数 {mutated_names}，"
                            f"对应调用参数 {mutated_symbols} —— 在 group dispatch 下存在并发竞争风险")
                })

            # 规则 2: 输入输出同一符号（纯 in-place）
            for i, a in enumerate(args):
                if a in mutated_symbols:
                    # 如果该 arg 在前面也作为输入出现过
                    if a in reads:
                        alerts.append({
                            "rule": "read-write-same-symbol",
                            "severity": "high",
                            "line": d["line"],
                            "kernel": kname,
                            "msg": (f"符号 {a} 既作为 {kname} 的输入又被就地改写 —— "
                                    f"in-place 读改写，若含 reduction 则高危")
                        })

            # 规则 3: 跨 kernel 的读改写链（A 写 -> B 读 -> B 输出又复用了 A 的存储）
            for sym in mutated_symbols:
                if sym in reads:
                    # sym 上次被某个 run 作为输入读过
                    alerts.append({
                        "rule": "raw-chain",
                        "severity": "medium",
                        "line": d["line"],
                        "kernel": kname,
                        "msg": (f"符号 {sym} 在作为前序 kernel 输入后，又在 {kname} 中被就地改写 —— "
                                f"存在 read-after-write / WAW 风险")
                    })

            # 更新读写集
            for a in args:
                reads.setdefault(a, []).append(d["line"])
            for a in mutated_symbols:
                writes[a] = d["line"]

        elif t == "reuse":
            # 规则 4: 显式 reuse 注释
            old = d["old"]
            reused_symbols.add(d["new"])
            if old in writes:
                alerts.append({
                    "rule": "reuse-after-write",
                    "severity": "medium",
                    "line": d["line"],
                    "msg": (f"符号 {old} 被显式复用为 {d['new']} (# reuse)，"
                            f"若 {old} 是 reduction 输出则存在 in-place 风险")
                })

    return {
        "artifact": path,
        "kernels": len(meta),
        "events": len(events),
        "alerts": alerts,
    }


def compare(green_path, red_path):
    g = analyze(green_path)
    r = analyze(red_path)
    print(f"\n=== 对比模式 ===")
    print(f"green alerts: {len(g['alerts'])}")
    for a in g["alerts"]:
        print(f"  [green] {a['rule']}: {a['msg']}")
    print(f"red   alerts: {len(r['alerts'])}")
    for a in r["alerts"]:
        print(f"  [red]   {a['rule']}: {a['msg']}")
    if r["alerts"] and not g["alerts"]:
        print("\n结论：红色产物独有的危险模式被 tracker 检出（绿色无警报）")
    elif r["alerts"] and g["alerts"]:
        print("\n结论：红/绿 都有警报；需人工区分通用模式与真实根因")
    return g, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", help="output_code.py 或 run_dir")
    ap.add_argument("--green", help="green artifact path")
    ap.add_argument("--report", default="buffer_alias_report.json")
    args = ap.parse_args()

    if args.green:
        g, r = compare(args.green, args.target)
        result = {"green": g, "red": r}
    else:
        path = args.target
        if path is None:
            ap.print_help()
            return 1
        if os.path.isdir(path):
            path = os.path.join(path, "output_code.py")
        result = analyze(path)

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
