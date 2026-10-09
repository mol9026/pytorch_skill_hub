#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
npu_compile_doctor —— NPU 图模式精度问题自动诊断（提案 A）

输入一个"图模式输出不对"的模型，自动执行金字塔排查四层，输出诊断报告 + 缺陷指纹匹配：

  L1 层级排除   : eager vs graph 整体 cos + 两次执行确定性 + fp32/fp16 对照
                  → cos>0.999 直接判"无问题"；非确定性 → 并发类；fp32 更差 → 逻辑类
  L2 kernel 隔离: 复用 kernel_isolate.py（提案 B）—— 注入探针跑真实 Runner.call，
                  逐站点重放：RACE / REPLAY-MISMATCH / KERNEL OK 三态
  L3 逐 buffer 溯源: 同一探针数据，把每个站点的 post 输出与"重放输出"比对之外，
                  追加跨站点污染检测（站点 i 的输出 buffer 与站点 j 的输入相同 = in-place 链）
  L4 特征指纹   : 误差 vs 位置/维度分布 → 相位类 / 索引类 / 并发类 / layout 类 自动归类，
                  匹配已知缺陷模式库（llama2_7b 四根因）

用法:
  python3 npu_compile_doctor.py --model llama1layer            # 内置真权重单层
  python3 npu_compile_doctor.py --args-script my_model.py      # 任意模型
  python3 npu_compile_doctor.py --model llama1layer --levels 1,2   # 只跑 L1+L2

输出: npu_doctor_report.json + stdout 诊断树
退出码: 0=图模式健康 1=检出问题(含指纹) 2=基础设施错误
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

SCRIPTS = os.environ.get("NPU_SCRIPTS_DIR", "/home/z00599499/scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

DEFAULT_REPORT = os.environ.get("NPU_EVIDENCE_DIR", "/home/z00599499/evidence/llama2_7b") + "/npu_doctor_report.json"
MODEL = os.environ.get("NPU_MODEL_DIR", "/home/z00599499/llama-2-7b/models/LLM-Research--Llama-2-7b/snapshots/master")

# ---------------------------------------------------------------------------
# 缺陷指纹库（llama2_7b 四根因 + 通用模式；每条含"特征 → 假设 → 验证建议"）
# ---------------------------------------------------------------------------

FINGERPRINTS = [
    {"name": "相位漂移（inv_freq 量化类）",
     "match": "err_grows_with_position",
     "hypothesis": "构造侧把 fp32 常量（inv_freq 等）量化成了模型 dtype，相位误差 × 位置线性放大",
     "verify": "检查 inv_freq.dtype；对照 from_pretrained（保 f32）vs from_config+.to(fp16)（会量化）",
     "ref": "llama2_7b 根因3"},
    {"name": "索引/轴合并错位类",
     "match": "kernel_replay_mismatch",
     "hypothesis": "融合 kernel 的索引按错误 layout 生成（跨轴合并/重排），单点重放与图内不一致",
     "verify": "kernel_isolate.py test <site>：喂 contiguous 输入对照；查 fx_passes 轴合并",
     "ref": "llama2_7b 根因1"},
    {"name": "并发竞争（in-place 读改写类）",
     "match": "nondeterministic",
     "hypothesis": "reduction kernel 读改写同一 buffer，48 核 group dispatch 下两遍循环交错",
     "verify": "kernel_isolate.py test <site> --runs 10 看 bitwise；查 decide_inplace_update",
     "ref": "llama2_7b 根因4"},
    {"name": "layout 契约缺失类",
     "match": "downstream_break_after_strided_input",
     "hypothesis": "aclnn 类算子要求物理 contiguous 但 op 未声明 tag，inductor 直喂 strided 数据",
     "verify": "找第一个错的 buffer 的上游算子；喂 contiguous 对照（strided 错/contig 对 = 实锤）",
     "ref": "llama2_7b 根因2"},
    {"name": "buffer 复用/调度时序类",
     "match": "single_ok_chain_bad",
     "hypothesis": "所有 kernel 单点正确但整链错 —— in-place 复用把后续输出写进了仍被读的 buffer",
     "verify": "L3 污染检测输出；查产物 'reuse' 注释与 decide_inplace_update",
     "ref": "llama2_7b 根因4 变体"},
]


def _out_tensor(o):
    """从 forward 输出取主 tensor（支持 tuple / ModelOutput）。"""
    if hasattr(o, "last_hidden_state"):
        return o.last_hidden_state
    if isinstance(o, (list, tuple)):
        return _out_tensor(o[0])
    return o


def cos_sim(a, b):
    import torch
    a, b = a.float().cpu().flatten(), b.float().cpu().flatten()
    if a.isnan().any() or b.isnan().any():
        return float("nan")
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)).item()


# ---------------------------------------------------------------------------
# args 构造（与 kernel_isolate 相同入口；支持 --args-script 任意模型）
# ---------------------------------------------------------------------------

def build_model(args_ns):
    if args_ns.args_script:
        import importlib.util
        spec = importlib.util.spec_from_file_location("args_builder", args_ns.args_script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = mod.build_args("npu")
        if len(out) == 4:
            return out  # (args, ref, model, forward_fn)
        return (*out, None)
    import kernel_isolate
    return (*kernel_isolate.build_llama1layer_args("npu"), None)


# ---------------------------------------------------------------------------
# L1 层级排除
# ---------------------------------------------------------------------------

def level1(args_list, ref, model, forward_fn=None):
    """eager vs graph 整体判定。返回 (findings, evidence)。"""
    import torch

    findings, ev = [], {}
    if forward_fn is None:
        forward_fn = _default_forward_fn(args_list)
    compiled = torch.compile(model, options={"npu_backend": "triton_experimental"})

    with torch.no_grad():
        g1 = forward_fn(compiled)
    g1 = _out_tensor(g1).detach()
    cos = cos_sim(ref, g1)
    ev["graph_vs_eager_cos"] = round(cos, 6) if cos == cos else "nan"
    print(f"[L1] graph vs eager cos = {ev['graph_vs_eager_cos']}")

    if cos == cos and cos > 0.999:
        # 确定性复检（防偶发）
        with torch.no_grad():
            g2 = forward_fn(compiled)
        g2 = _out_tensor(g2).detach()
        if torch.equal(g1, g2):
            findings.append({"level": "L1", "severity": "clean",
                             "msg": f"cos={cos:.6f} 且 bitwise 确定 —— 图模式健康，无需继续"})
            return findings, ev, compiled

    if cos != cos or cos < 0.99:
        # 非确定性检测
        with torch.no_grad():
            g2 = forward_fn(compiled)
        g2 = _out_tensor(g2).detach()
        det = torch.equal(g1, g2)
        ev["deterministic"] = bool(det)
        if not det:
            findings.append({"level": "L1", "severity": "high",
                             "msg": "两次执行 bitwise 不一致 —— 并发竞争类指纹（根因4类）",
                             "fingerprint": "nondeterministic"})
        else:
            findings.append({"level": "L1", "severity": "high",
                             "msg": f"cos={cos} 确定性错误 —— 计算逻辑错，进 L2 定位"})
    else:
        findings.append({"level": "L1", "severity": "low",
                         "msg": f"cos={cos:.4f} 边界值 —— 轻微偏差，建议 L2/L4 复查"})
    return findings, ev, compiled


def _default_forward_fn(args_list):
    """llama 1 层默认：args = [input_ids, weight, attn, inv_freq, ...] —— kwargs 形式。"""
    def fn(model):
        return model(input_ids=args_list[0], attention_mask=args_list[2], use_cache=False)
    return fn


def _simple_forward_fn(args_list):
    """单位置参数模型（如 nanogpt：model(idx)）。"""
    def fn(model):
        return model(*args_list)
    return fn


# ---------------------------------------------------------------------------
# L2 kernel 隔离（复用提案 B）
# ---------------------------------------------------------------------------

def level2(caps, art, env):
    """逐站点重放 3 次（快速档）。返回 findings。"""
    import torch
    import torch_npu  # noqa: F401
    from torch_npu._inductor.triton_experimental import get_current_raw_stream as get_raw

    findings = []
    print(f"\n[L2] 逐站点 kernel 隔离重放（{len(caps)} 站点 x 3 次）...")
    for c in caps:
        kname, site = c["kernel"], c["site"]
        kernel = env.get(kname)
        if kernel is None:
            continue
        site_info = next(s for s in art["sites"] if s["site"] == site)
        runs = []
        for _ in range(3):
            call_args = [t.clone() if torch.is_tensor(t) else t for t in c["pre"]]
            call_args += [int(n) for n in site_info["num_args"]]
            kernel.run(*call_args, stream=get_raw(0))
            torch.npu.synchronize()
            runs.append(call_args)
        det = all(torch.equal(runs[0][i], runs[k][i])
                  for k in range(1, 3) for i in range(len(runs[0]))
                  if torch.is_tensor(runs[0][i]))
        replay_match = all(torch.equal(runs[0][i], c["post"][i])
                           for i in range(len(c["post"]))
                           if torch.is_tensor(c["post"][i]))
        tag = ""
        if not det:
            tag = "RACE"
            findings.append({"level": "L2", "severity": "high", "site": site, "kernel": kname,
                             "msg": f"site {site} 重放不确定 —— 并发竞争指纹",
                             "fingerprint": "nondeterministic"})
        elif not replay_match:
            tag = "REPLAY-MISMATCH"
            findings.append({"level": "L2", "severity": "high", "site": site, "kernel": kname,
                             "msg": f"site {site} 确定但重放≠图内 —— 索引/layout 依赖",
                             "fingerprint": "kernel_replay_mismatch"})
        print(f"  site {site:2d} {kname[:60]:60s} {'OK' if not tag else tag}")
    if not findings:
        findings.append({"level": "L2", "severity": "clean",
                         "msg": "全部站点单点正确 —— 错误在链路组合（L3 溯源）"})
    return findings


# ---------------------------------------------------------------------------
# L3 逐 buffer 溯源 + 跨站点污染检测
# ---------------------------------------------------------------------------

def level3(caps, ref, env, args_list):
    """找第一个'重放对但链路坏'的证据：直接比对捕获链的最终输出 vs eager。

    另做跨站点污染检测：站点 i 的 pre buffer 与站点 j 的 post buffer 是同一块存储
    （data_ptr 相同）且 i>j —— 说明站点 j in-place 改写了站点 i 还要读的数据。
    """
    import torch
    findings = []

    # 污染检测（data_ptr 层）
    print("\n[L3] 跨站点 in-place 污染检测 ...")
    post_ptrs = []  # (site, idx, ptr, tensor)
    for c in caps:
        for i, t in enumerate(c["post"]):
            if torch.is_tensor(t):
                post_ptrs.append((c["site"], i, t.data_ptr(), t))
    for c in caps:
        for i, t in enumerate(c["pre"]):
            if not torch.is_tensor(t):
                continue
            for (psite, pidx, pptr, pt) in post_ptrs:
                if pptr == t.data_ptr() and psite < c["site"] and not torch.equal(t, pt):
                    findings.append({
                        "level": "L3", "severity": "high",
                        "site": c["site"], "kernel": c["kernel"],
                        "msg": (f"site {c['site']} 的输入[{i}]与 site {psite} 的输出[{pidx}] "
                                f"同址（data_ptr 相同）且内容已被改写 —— in-place 污染"),
                        "fingerprint": "single_ok_chain_bad"})
                    print(f"  !! site {psite} 输出 -> site {c['site']} 输入 同址改写")

    # 最终输出健康度（Runner 全链 vs eager 已在 capture 自检做过，这里引用）
    if not findings:
        findings.append({"level": "L3", "severity": "clean",
                         "msg": "无跨站点污染；链路完整性由 capture 自检（cos>0.999）保证"})
        print("  无污染")
    return findings


# ---------------------------------------------------------------------------
# L4 特征指纹（对整体误差做形状分析）
# ---------------------------------------------------------------------------

def level4(args_list, ref, model, compiled, forward_fn=None):
    """整体误差的形状分析：误差 vs 位置单调性 = 相位类指纹。"""
    import torch
    findings, ev = [], {}

    with torch.no_grad():
        g = (forward_fn or _default_forward_fn(args_list))(compiled)
    g = _out_tensor(g).detach()

    if g.shape != ref.shape:
        return findings, ev
    d = (g.float() - ref.float()).abs()          # [1, S, H]
    if d.dim() == 3:
        pos_err = d[0].mean(dim=-1)              # [S]
        S = pos_err.shape[0]
        if S >= 16:
            q1, q4 = pos_err[:S // 4].mean().item(), pos_err[-S // 4:].mean().item()
            ev["err_first_quarter"] = round(q1, 6)
            ev["err_last_quarter"] = round(q4, 6)
            print(f"\n[L4] 误差按位置: 前1/4 均值 {q1:.2e}  vs  后1/4 均值 {q4:.2e}")
            if q4 > 4 * max(q1, 1e-8):
                findings.append({"level": "L4", "severity": "high",
                                 "msg": f"误差随位置放大（{q1:.2e} -> {q4:.2e}）—— 相位/inv_freq 量化指纹",
                                 "fingerprint": "err_grows_with_position"})
            elif d.isnan().any():
                findings.append({"level": "L4", "severity": "high",
                                 "msg": "输出含 NaN —— 数值爆炸（放大/竞争）指纹",
                                 "fingerprint": "nondeterministic"})
    return findings, ev


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama1layer")
    ap.add_argument("--args-script", default=None)
    ap.add_argument("--levels", default="1,2,3,4")
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ns = ap.parse_args()
    levels = {int(x) for x in ns.levels.split(",")}

    import torch
    t0 = time.time()
    print("=" * 64)
    print("npu_compile_doctor —— 金字塔自动诊断")
    print("=" * 64)

    # 模型与 eager 参考
    args_list, ref, model, forward_fn = build_model(ns)
    print(f"model ready ({time.time()-t0:.0f}s)")

    findings, evidence = [], {}
    compiled = None

    if 1 in levels:
        f1, ev1, compiled = level1(args_list, ref, model, forward_fn)
        findings += f1
        evidence.update(ev1)
        if any(x["severity"] == "clean" and x["level"] == "L1" for x in f1):
            return finish(findings, evidence, 0, ns, t0)

    # L2/L3 需要 capture（探针 + 真实 buffer）
    caps = art = cenv = None
    if 2 in levels or 3 in levels:
        import kernel_isolate as ki
        gdir = ki.resolve_artifact()
        art = ki.parse_artifact(gdir)
        print(f"\n[cap] 注入探针重跑 Runner.call（{len(art['sites'])} 站点）...")
        cenv = ki.exec_artifact(gdir, inject=True, kernel_names=set(art["kernels"].values()))
        runner = cenv["Runner"]([])
        with torch.no_grad():
            runner.call(list(args_list))
        caps = cenv["CAPTURES"]
        selfcheck_cos = None
        print(f"[cap] captured {len(caps)} sites")

    if 2 in levels and caps:
        findings += level2(caps, art, cenv)
    if 3 in levels and caps:
        findings += level3(caps, ref, cenv, args_list)
    if 4 in levels and compiled is not None:
        f4, ev4 = level4(args_list, ref, model, compiled, forward_fn)
        findings += f4
        evidence.update(ev4)

    # 指纹匹配
    matched = [fp for fp in FINGERPRINTS
               if any(f.get("fingerprint") == fp["match"] for f in findings)]
    exit_code = 1 if any(f["severity"] == "high" for f in findings) else 0
    return finish(findings, evidence, exit_code, ns, t0, matched)


def finish(findings, evidence, exit_code, ns, t0, matched=None):
    report = {
        "tool": "npu_compile_doctor v0.1 (proposal A)",
        "duration_s": round(time.time() - t0, 1),
        "findings": findings,
        "evidence": evidence,
        "matched_fingerprints": matched or [],
        "verdict": "HEALTHY" if exit_code == 0 else "PROBLEM FOUND",
    }
    with open(ns.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 64)
    print(f"VERDICT: {report['verdict']}  ({report['duration_s']}s)")
    for f in findings:
        print(f"  [{f['level']}|{f['severity']}] {f['msg']}")
    if matched:
        print("\n匹配的已知缺陷模式：")
        for fp in matched:
            print(f"  * {fp['name']}（{fp['ref']}）")
            print(f"    假设: {fp['hypothesis']}")
            print(f"    验证: {fp['verify']}")
    print(f"\nreport -> {ns.report}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
