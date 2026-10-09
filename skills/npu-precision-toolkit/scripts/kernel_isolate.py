#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kernel_isolate —— NPU 图模式融合 kernel 隔离执行器（提案 B：kernel-isolation-kit）

把"从编译产物中抽取单个 kernel、用真实 buffer 单独执行对拍"从手工 30 分钟变成一条命令。

原理（capture-replay）：
  1. capture: 改写产物 output_code.py 源码，在每个 kernel.run() 调用点前后注入探针，
     用真实模型 args 跑一次 Runner.call，抓取每个调用点的 pre/post buffer（clone 保 stride）
  2. test:    对指定 kernel，用捕获的 pre 输入重放 N 次：
     - N 次之间 bitwise 不一致          -> RACE（并发竞争指纹，根因4类）
     - 一致但与捕获 post 不一致          -> REPLAY-MISMATCH（输入状态/layout 依赖）
     - 全一致                           -> KERNEL OK（单点正确；错误在链路组合，根因1/2类需看上游）

用法:
  python3 kernel_isolate.py list                          # 列出最新产物的全部 kernel 调用点
  python3 kernel_isolate.py list --artifact <run_dir>
  python3 kernel_isolate.py capture                       # 抓取真实 buffer（默认 llama1layer 真权重）
  python3 kernel_isolate.py capture --args-script my.py   # 任意模型：脚本提供 build_args()
  python3 kernel_isolate.py test --kernel 5               # 按调用点序号测（10 次重放）
  python3 kernel_isolate.py test --kernel <name片段> --runs 10

输出: kernel_isolation_report.json + stdout 摘要
"""

import argparse
import glob
import json
import os
import re
import sys
import time

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

DEBUG_ROOT = os.environ.get("NPU_DEBUG_ROOT", "/home/z00599499/torch_compile_debug")
DEFAULT_CAPTURES = os.environ.get("NPU_EVIDENCE_DIR", "/home/z00599499/evidence/llama2_7b") + "/kernel_captures.pt"
DEFAULT_REPORT = os.environ.get("NPU_EVIDENCE_DIR", "/home/z00599499/evidence/llama2_7b") + "/kernel_isolation_report.json"
MODEL = os.environ.get("NPU_MODEL_DIR", "/home/z00599499/llama-2-7b/models/LLM-Research--Llama-2-7b/snapshots/master")


# ---------------------------------------------------------------------------
# 产物定位与解析
# ---------------------------------------------------------------------------

def resolve_artifact(user_dir=None):
    """返回含 output_code.py 的目录。默认最新 run。"""
    if user_dir:
        if not os.path.isdir(user_dir):
            # 容许直接传 run 目录
            cands = glob.glob(f"{user_dir}/torchinductor/*/output_code.py")
            if not cands:
                raise SystemExit(f"artifact 无 output_code.py: {user_dir}")
            return os.path.dirname(cands[0])
        if os.path.exists(f"{user_dir}/output_code.py"):
            return user_dir
        raise SystemExit(f"artifact 目录无效: {user_dir}")
    runs = sorted(glob.glob(f"{DEBUG_ROOT}/run_*"), key=os.path.getmtime)
    for r in reversed(runs):
        for oc in glob.glob(f"{r}/torchinductor/*/output_code.py"):
            return os.path.dirname(oc)
    raise SystemExit("torch_compile_debug 下无产物；先跑一次 compile（如 watchdog g4）")


def split_top_args(s):
    """按顶层逗号切分（忽略括号内逗号）。"""
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


RUN_LINE = re.compile(r"^(\s*)(\w+)\.run\((.*)\)\s*$")
SHAPE_LINE = re.compile(r"(\w+) = empty_strided_npu\(\(([^)]*)\), \(([^)]*)\), (\w+)\.\w+\)")


def parse_artifact(gdir):
    """解析 output_code.py：kernel 注册表 + 调用点（含源节点注释与 buffer 形状）。"""
    src = open(f"{gdir}/output_code.py", encoding="utf-8").read()
    kernels = dict(re.findall(r"(\w+) = async_compile\.triton\('(\w+)'", src))
    shapes = {}
    for m in SHAPE_LINE.finditer(src):
        name, shape, stride, dt = m.groups()
        shapes[name] = {"shape": [int(x) for x in shape.split(",") if x.strip()],
                        "stride": [int(x) for x in stride.split(",") if x.strip()],
                        "dtype": dt.split(".")[-1]}
    sites = []
    lines = src.split("\n")
    for i, ln in enumerate(lines):
        m = RUN_LINE.match(ln)
        if not m or m.group(2) not in kernels:
            continue
        if "stream=" not in ln:
            continue
        ind, kname, argstr = m.groups()
        args = split_top_args(argstr)
        refs = [a for a in args if re.fullmatch(r"\w+", a) and not a.isdigit()]
        nums = [a for a in args if a.isdigit()]
        # 向上找最近的源节点注释
        src_nodes = ""
        for j in range(i - 1, max(0, i - 4), -1):
            if "Topologically Sorted Source Nodes" in lines[j]:
                src_nodes = lines[j].split("Source Nodes:")[-1].strip()
                break
        sites.append({"site": len(sites), "kernel": kname,
                      "arg_refs": refs, "num_args": nums,
                      "src_nodes": src_nodes[:200]})
    return {"dir": gdir, "kernels": kernels, "shapes": shapes, "sites": sites}


# ---------------------------------------------------------------------------
# 注入探针（capture）
# ---------------------------------------------------------------------------

PREAMBLE = '''
import torch as _torch
CAPTURES = []
def _cap_pre(site, kname, refs):
    CAPTURES.append({"site": site, "kernel": kname,
                     "pre": [t.clone() if _torch.is_tensor(t) else t for t in refs]})
def _cap_post(site, kname, refs):
    assert CAPTURES[-1]["site"] == site
    CAPTURES[-1]["post"] = [t.clone() if _torch.is_tensor(t) else t for t in refs]
'''


def inject_probes(src, kernel_names):
    """在 Runner.call 的每个 kernel.run() 前后注入 _cap_pre/_cap_post。"""
    out = []
    site = 0
    for ln in src.split("\n"):
        m = RUN_LINE.match(ln)
        ok = m and m.group(2) in kernel_names and "stream=" in ln
        if ok:
            ind, kname, argstr = m.groups()
            # 捕获全部位置参数（含数字 numel）——run() 需要它们；stream= 已被
            # split_top_args 切走（kwarg 不在位置参数里，但需防混入）
            pos = [a for a in split_top_args(argstr)
                   if not a.startswith("stream=")]
            refs = [a for a in pos
                    if re.fullmatch(r"\w+", a) and not a.isdigit()]
            nums = [a for a in pos if a.isdigit()]
            ref_list = "[" + ", ".join(refs) + "]"
            out.append(f"{ind}_cap_pre({site}, '{kname}', {ref_list})")
            out.append(ln)
            out.append(f"{ind}_cap_post({site}, '{kname}', {ref_list})")
            site += 1
        else:
            out.append(ln)
    return "\n".join(out), site


# ---------------------------------------------------------------------------
# 默认 args 构造器（llama 1 层真权重，与 watchdog G4 同构）
# ---------------------------------------------------------------------------

def build_llama1layer_args(device="npu"):
    """返回 (args_list, eager_ref_output)。args 顺序 = Runner.call 形参顺序。"""
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModelForCausalLM, LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel

    full = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                local_files_only=True)
    cfg = full.config
    sc = LlamaConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size, num_hidden_layers=1,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps, dtype=cfg.dtype,
        rope_parameters=cfg.rope_parameters,
    )
    small = LlamaModel(sc)
    small.embed_tokens.load_state_dict(full.model.embed_tokens.state_dict())
    small.layers[0].load_state_dict(full.model.layers[0].state_dict())
    small.norm.load_state_dict(full.model.norm.state_dict())
    del full
    small = small.half().to(device).eval()
    inv_f32 = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
    small.rotary_emb.inv_freq.data = inv_f32.to(device)

    torch.manual_seed(123)
    input_ids = torch.randint(10, 31000, (1, 128), device=device)
    attn = torch.ones_like(input_ids)

    with torch.no_grad():
        ref = small(input_ids=input_ids, attention_mask=attn,
                    use_cache=False).last_hidden_state.detach()

    L = small.layers[0]
    args = [input_ids, small.embed_tokens.weight, attn, small.rotary_emb.inv_freq,
            L.input_layernorm.weight,
            L.self_attn.q_proj.weight, L.self_attn.k_proj.weight,
            L.self_attn.v_proj.weight, L.self_attn.o_proj.weight,
            L.post_attention_layernorm.weight,
            L.mlp.gate_proj.weight, L.mlp.up_proj.weight, L.mlp.down_proj.weight,
            small.norm.weight]
    return args, ref, small


def load_args_script(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("args_builder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_args


# ---------------------------------------------------------------------------
# capture / test
# ---------------------------------------------------------------------------

def exec_artifact(gdir, inject=False, kernel_names=None):
    """exec output_code.py，返回 env（含 kernel 对象与 Runner）。可选注入探针。"""
    src = open(f"{gdir}/output_code.py", encoding="utf-8").read()
    if inject:
        src, _ = inject_probes(src, kernel_names)
        src = PREAMBLE + src
    env = {}
    exec(compile(src, "output_code.py", "exec"), env)
    return env


def cmd_list(args):
    art = parse_artifact(resolve_artifact(args.artifact))
    print(f"artifact: {art['dir']}")
    print(f"kernels: {len(art['kernels'])}, call sites: {len(art['sites'])}\n")
    for s in art["sites"]:
        shapes = []
        for r in s["arg_refs"]:
            if r in art["shapes"]:
                sh = art["shapes"][r]
                shapes.append(f"{r}{sh['shape']}{sh['dtype'][:4]}")
            else:
                shapes.append(r)
        print(f"[site {s['site']:2d}] {s['kernel']}")
        print(f"          args: {', '.join(shapes)}")
        if s["src_nodes"]:
            print(f"          nodes: {s['src_nodes'][:120]}")
    return 0


def cmd_capture(args):
    import torch
    import torch_npu  # noqa: F401

    gdir = resolve_artifact(args.artifact)
    art = parse_artifact(gdir)
    print(f"artifact: {gdir}")
    print(f" injecting probes into {len(art['sites'])} call sites ...")

    env = exec_artifact(gdir, inject=True, kernel_names=set(art["kernels"].values()))

    if args.args_script:
        build = load_args_script(args.args_script)
        args_list, ref, _ = build("npu")
    else:
        args_list, ref, _ = build_llama1layer_args("npu")

    n_params = len(env["Runner"].__init__.__code__.co_varnames)  # noqa
    runner = env["Runner"]([])
    with torch.no_grad():
        out = runner.call(list(args_list))
    out = out[0] if isinstance(out, (list, tuple)) else out

    # 仪器自检（铁律 2：先审计仪器）：Runner 输出必须与 eager 一致，否则 args 顺序错
    a, b = out.float().cpu().flatten(), ref.float().cpu().flatten()
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    print(f"\nself-check: runner vs eager cos = {cos:.6f}")
    if cos < 0.999:
        print("!! args 顺序或产物不匹配 —— 捕获无效，先修 args builder")
        return 2

    caps = env["CAPTURES"]
    print(f"captured {len(caps)} sites")
    for c in caps:
        npre = sum(1 for t in c["pre"] if torch.is_tensor(t))
        print(f"  [site {c['site']:2d}] {c['kernel']}: {npre} tensor args")

    torch.save({"artifact": gdir, "captures": caps,
                "args_desc": [str(tuple(t.shape)) + str(t.dtype) if torch.is_tensor(t) else str(t)
                              for t in args_list]},
               args.out)
    print(f"\nsaved -> {args.out}")
    return 0


def cmd_test(args):
    import torch
    import torch_npu  # noqa: F401
    from torch_npu._inductor.triton_experimental import get_current_raw_stream as get_raw

    if not os.path.exists(args.captures):
        raise SystemExit(f"无捕获文件 {args.captures}，先跑 capture")
    data = torch.load(args.captures, weights_only=False)
    caps = data["captures"]
    gdir = data["artifact"]
    art = parse_artifact(gdir)

    sel = select_sites(caps, args.kernel)
    if not sel:
        names = "\n".join(f"  site {c['site']}: {c['kernel']}" for c in caps)
        raise SystemExit(f"未匹配到 kernel '{args.kernel}'。可用:\n{names}")

    env = exec_artifact(gdir)  # 不注入，直接拿 kernel 对象
    report = {"artifact": gdir, "runs": args.runs, "results": []}
    exit_code = 0

    for c in sel:
        kname = c["kernel"]
        kernel = env.get(kname)
        if kernel is None:
            print(f"[site {c['site']}] {kname}: NOT FOUND in artifact env")
            continue
        pre, post = c["pre"], c["post"]
        # 从产物源码取该 site 的完整位置参数模板：tensor 用捕获值，数字 numel 用源码值
        site_info = next(s for s in art["sites"] if s["site"] == c["site"])
        # arg_refs 与 pre 一一对应（注入时只捕获了 tensor refs）
        assert len(site_info["arg_refs"]) == len(pre)
        outs_per_run = []
        for r in range(args.runs):
            call_args = [t.clone() if torch.is_tensor(t) else t for t in pre]
            call_args += [int(n) for n in site_info["num_args"]]
            kernel.run(*call_args, stream=get_raw(0))
            torch.npu.synchronize()
            outs_per_run.append(call_args)

        # 判定 1：重放间 bitwise 一致性（race 指纹）
        det = all(
            torch.equal(a, b) if torch.is_tensor(a) else a == b
            for a, b in zip(outs_per_run[0], outs_per_run[-1])
        ) and all(
            torch.equal(outs_per_run[0][i], outs_per_run[k][i])
            for k in range(1, args.runs) for i in range(len(outs_per_run[0]))
            if torch.is_tensor(outs_per_run[0][i])
        )
        # 判定 2：重放结果 vs 捕获 post
        replay_match = all(
            torch.equal(outs_per_run[0][i], post[i]) if torch.is_tensor(post[i])
            else outs_per_run[0][i] == post[i]
            for i in range(len(post))
        )
        if not det:
            verdict = "RACE: 重放间 bitwise 不一致 -> 并发竞争（根因4类指纹）"
            exit_code = 1
        elif not replay_match:
            verdict = "REPLAY-MISMATCH: 重放确定但与图内捕获不一致 -> 输入状态/layout 依赖，需查调用侧"
            exit_code = 1
        else:
            verdict = "KERNEL OK: 单点正确（错误若存在，在链路组合/上游）"

        print(f"[site {c['site']:2d}] {kname}")
        print(f"          {args.runs}-run bitwise identical = {det}, replay==capture = {replay_match}")
        print(f"          -> {verdict}")
        report["results"].append({"site": c["site"], "kernel": kname,
                                  "deterministic": bool(det),
                                  "replay_match": bool(replay_match),
                                  "verdict": verdict})

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nreport -> {args.report}")
    return exit_code


def select_sites(caps, key):
    if key.isdigit():
        i = int(key)
        return [c for c in caps if c["site"] == i]
    return [c for c in caps if key in c["kernel"]]


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list")
    p.add_argument("--artifact", default=None)
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("capture")
    p.add_argument("--artifact", default=None)
    p.add_argument("--args-script", default=None,
                   help="python 文件提供 build_args(device)->(args, ref, model)")
    p.add_argument("--out", default=DEFAULT_CAPTURES)
    p.set_defaults(fn=cmd_capture)

    p = sub.add_parser("test")
    p.add_argument("--kernel", required=True, help="site 序号或名称片段")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--captures", default=DEFAULT_CAPTURES)
    p.add_argument("--report", default=DEFAULT_REPORT)
    p.set_defaults(fn=cmd_test)

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
