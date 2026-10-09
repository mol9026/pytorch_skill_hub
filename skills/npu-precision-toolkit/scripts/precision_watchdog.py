#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
precision_watchdog —— NPU 图模式（triton_experimental）精度回归门禁（提案 C 实现）

三层检查（与案例分享 §6 提案 C 一致）+ 红灯测试新增的 G4：
  G1 单层精度   : 1 层随机权重 Llama，graph vs eager cos > 0.999（抓索引类回归）
  G2 执行确定性 : 同一 compiled 模型 10 次执行 bitwise 一致（抓并发竞争类回归）
  G3 端到端精度 : TorchBench llama_v2_7b_16h 口径 cos > 0.999（团队口径端到端）
  G4 真实权重   : from_pretrained 真权重单层 cos > 0.999（历史故障主路径，
                  红灯测试证明 G1/G3 对编译器侧回归不敏感后补上）

用法:
  python3 precision_watchdog.py --gate g12     # G1+G2（同一次编译，~3min）
  python3 precision_watchdog.py --gate g3      # G3（~2-10min，视缓存）
  python3 precision_watchdog.py --gate g4      # G4 真实权重单层（~4min）
  python3 precision_watchdog.py --gate all     # 全部，子进程隔离
  python3 precision_watchdog.py --gate all --json /path/report.json

退出码: 0=全过  1=门禁失败(精度/确定性不达标)  2=基础设施错误(编译崩溃等)
判定阈值与团队口径一致: cos > 0.999
历史四根因覆盖: G1 抓根因1(索引)/2(layout)，G2 抓根因4(并发)，G3 端到端兜底（根因3 在构造侧，由 inv_freq f32 语义保持）
"""

import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")  # fork 子进程 NPU 重初始化 bug 的规避

THRESHOLD_COS = 0.999
N_DET_RUNS = 10
DEFAULT_REPORT = os.environ.get("NPU_EVIDENCE_DIR", "/home/z00599499/evidence/llama2_7b") + "/watchdog_report.json"
MODEL_TAG = "watchdog v1.2 (proposal C)"


def cos_sim(a, b):
    import torch.nn.functional as F
    a, b = a.float().cpu().flatten(), b.float().cpu().flatten()
    if a.isnan().any() or b.isnan().any():
        return float("nan")
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def _inv_freq_f32(device):
    """根因3 修复语义：inv_freq 必须保持 f32（.to(fp16) 会量化它导致相位漂移）。"""
    import torch
    return (1.0 / (10000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))).to(device)


def run_g12():
    """G1 单层精度 + G2 确定性。随机权重 1 层 LlamaModel（无外部权重依赖，可进 CI）。"""
    import torch
    import torch_npu  # noqa: F401
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel

    t_start = time.time()
    device, dtype, seq = "npu", torch.float16, 128

    torch.manual_seed(42)
    cfg = LlamaConfig(num_hidden_layers=1)
    model = LlamaModel(cfg).to(device=device, dtype=dtype).eval()
    model.rotary_emb.inv_freq.data = _inv_freq_f32(device)

    torch.manual_seed(123)
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq), device=device)
    attn = torch.ones_like(input_ids)

    with torch.no_grad():
        ref = model(input_ids=input_ids, attention_mask=attn,
                    use_cache=False).last_hidden_state

    print(f"[g12] compiling 1-layer (seq={seq}) ...", flush=True)
    t0 = time.time()
    compiled = torch.compile(model, options={"npu_backend": "triton_experimental"})
    with torch.no_grad():
        out = compiled(input_ids=input_ids, attention_mask=attn,
                       use_cache=False).last_hidden_state
    compile_s = time.time() - t0
    print(f"[g12] compile done in {compile_s:.0f}s", flush=True)

    cos = cos_sim(ref, out)
    g1 = {
        "name": "G1_single_layer_cos",
        "desc": "1-layer random-init Llama, graph vs eager",
        "value": None if cos != cos else round(cos, 6),
        "threshold": f">{THRESHOLD_COS}",
        "pass": bool(cos == cos and cos > THRESHOLD_COS),
    }
    print(f"[g12] G1 cos = {g1['value']}  -> {'PASS' if g1['pass'] else 'FAIL'}", flush=True)

    outs = []
    with torch.no_grad():
        for _ in range(N_DET_RUNS):
            outs.append(compiled(input_ids=input_ids, attention_mask=attn,
                                 use_cache=False).last_hidden_state.clone())
    identical = all(torch.equal(outs[0], o) for o in outs[1:])
    maxdiff = max((outs[0].float() - o.float()).abs().max().item() for o in outs[1:])
    g2 = {
        "name": "G2_bitwise_determinism",
        "desc": f"same compiled model, {N_DET_RUNS} executions bitwise-identical",
        "value": f"{'all' if identical else 'NOT'} bitwise identical",
        "max_abs_diff_across_runs": maxdiff,
        "pass": bool(identical),
    }
    print(f"[g12] G2 10-run bitwise identical = {identical} "
          f"(maxdiff={maxdiff})  -> {'PASS' if g2['pass'] else 'FAIL'}", flush=True)

    return {
        "model": "LlamaConfig(num_hidden_layers=1) random init, bs=1 seq=128 fp16",
        "gates": [g1, g2],
        "duration_s": round(time.time() - t_start, 1),
    }


def run_g3():
    """G3 端到端：TorchBench llama_v2_7b_16h 团队口径（from_config 随机权重, bs=1 seq=512）。"""
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModelForCausalLM, LlamaConfig

    t_start = time.time()
    device, dtype = "npu", torch.float16

    torch.manual_seed(42)
    config = LlamaConfig(num_hidden_layers=16)
    model = AutoModelForCausalLM.from_config(config).to(device=device, dtype=dtype).eval()
    model.model.rotary_emb.inv_freq.data = _inv_freq_f32(device)

    torch.manual_seed(123)
    input_ids = torch.randint(0, config.vocab_size, (1, 512), device=device)
    attn = torch.ones_like(input_ids)

    with torch.no_grad():
        logits_eager = model(input_ids=input_ids, attention_mask=attn).logits.float().cpu()

    print("[g3] compiling llama_v2_7b_16h (TorchBench) ...", flush=True)
    t0 = time.time()
    compiled = torch.compile(model, options={"npu_backend": "triton_experimental"})
    with torch.no_grad():
        _ = compiled(input_ids=input_ids, attention_mask=attn)
    compile_s = time.time() - t0
    print(f"[g3] compile done in {compile_s:.0f}s", flush=True)

    with torch.no_grad():
        logits_graph = compiled(input_ids=input_ids, attention_mask=attn).logits.float().cpu()

    cos = cos_sim(logits_eager, logits_graph)
    g3 = {
        "name": "G3_torchbench_llama16h_cos",
        "desc": "TorchBench llama_v2_7b_16h, from_config random init, bs=1 seq=512",
        "value": None if cos != cos else round(cos, 6),
        "threshold": f">{THRESHOLD_COS}",
        "pass": bool(cos == cos and cos > THRESHOLD_COS),
        "compile_s": round(compile_s, 1),
    }
    print(f"[g3] G3 cos = {g3['value']}  -> {'PASS' if g3['pass'] else 'FAIL'}", flush=True)

    return {
        "model": "llama_v2_7b_16h (TorchBench) random init, bs=1 seq=512 fp16",
        "gates": [g3],
        "duration_s": round(time.time() - t_start, 1),
    }


def run_g4():
    """G4 真实权重单层精度：历史故障主路径（from_pretrained 真权重, fp16, seq=128）。

    历史证据（修复前, minimal_repro）: 1 层 layer0 cos=0.9177 —— rc1/2/4 的显形场。
    与 G1/G3（随机权重 from_config）互补：G3 无法复现编译器侧回归（llama_16h 的
    0.657 主要由 rc3 构造侧量化贡献），G4 直接测编译栈对真权重的行为。
    """
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModelForCausalLM, LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel

    MODEL = os.environ.get("NPU_MODEL_DIR", "/home/z00599499/llama-2-7b/models/LLM-Research--Llama-2-7b/snapshots/master")
    t_start = time.time()
    device, dtype, seq = "npu", torch.float16, 128

    full = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=dtype, local_files_only=True)
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
    # from_pretrained 语义：inv_freq 保持 f32（non-persistent buffer 不被权重 dtype 量化）。
    # 历史主路径（minimal_repro 0.9177）即 f32 inv_freq + 纯编译器缺陷——
    # 不重置会把 rc3 相位漂移混淆进 G4，削弱对编译器回归的敏感度。
    small.rotary_emb.inv_freq.data = _inv_freq_f32(device)

    torch.manual_seed(123)
    input_ids = torch.randint(10, 31000, (1, seq), device=device)
    attn = torch.ones_like(input_ids)

    with torch.no_grad():
        ref = small(input_ids=input_ids, attention_mask=attn,
                    use_cache=False).last_hidden_state

    print("[g4] compiling 1-layer real-weights (seq=128) ...", flush=True)
    t0 = time.time()
    compiled = torch.compile(small, options={"npu_backend": "triton_experimental"})
    with torch.no_grad():
        out = compiled(input_ids=input_ids, attention_mask=attn,
                       use_cache=False).last_hidden_state
    compile_s = time.time() - t0
    print(f"[g4] compile done in {compile_s:.0f}s", flush=True)

    cos = cos_sim(ref, out)
    g4 = {
        "name": "G4_realweight_1layer_cos",
        "desc": "1-layer real-weight Llama (from_pretrained), graph vs eager",
        "value": None if cos != cos else round(cos, 6),
        "threshold": f">{THRESHOLD_COS}",
        "pass": bool(cos == cos and cos > THRESHOLD_COS),
        "compile_s": round(compile_s, 1),
    }
    print(f"[g4] G4 cos = {g4['value']}  -> {'PASS' if g4['pass'] else 'FAIL'}", flush=True)
    return {
        "model": "Llama-2-7b layer0 real weights, bs=1 seq=128 fp16",
        "gates": [g4],
        "duration_s": round(time.time() - t_start, 1),
    }


WORKERS = {"g12": run_g12, "g3": run_g3, "g4": run_g4}


def run_one(gate: str, json_path: str) -> int:
    try:
        result = WORKERS[gate]()
        ok = all(g["pass"] for g in result["gates"])
    except Exception as e:  # 基础设施错误：编译崩溃 / OOM / 环境问题
        import traceback
        traceback.print_exc()
        result = {"error": str(e), "gates": [], "gate": gate}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"watchdog": MODEL_TAG, "gate": gate, "verdict": "ERROR",
                       "result": result}, f, ensure_ascii=False, indent=2)
        return 2
    verdict = "PASS" if ok else "FAIL"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"watchdog": MODEL_TAG, "gate": gate, "verdict": verdict,
                   "result": result}, f, ensure_ascii=False, indent=2)
    return 0 if ok else 1


def run_all(json_path: str) -> int:
    overall = 0
    combined = {"watchdog": MODEL_TAG, "gate": "all", "results": {}}
    for gate in ("g12", "g3", "g4"):
        tmp = json_path + f".{gate}"
        print(f"\n===== gate {gate} =====", flush=True)
        t0 = time.time()
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--gate", gate, "--json", tmp],
            capture_output=True, text=True)
        dur = round(time.time() - t0, 1)
        if p.stdout:
            for ln in p.stdout.strip().splitlines()[-15:]:
                print("  |", ln, flush=True)
        if p.returncode == 2:
            combined["results"][gate] = {"verdict": "ERROR", "stderr_tail": (p.stderr or "")[-800:],
                                         "duration_s": dur}
            overall = 2
            continue
        r = json.load(open(tmp, encoding="utf-8")) if os.path.exists(tmp) else {}
        r["duration_s"] = dur
        combined["results"][gate] = r
        if p.returncode == 1:
            overall = 1 if overall == 0 else overall
    combined["verdict"] = "ERROR" if overall == 2 else ("FAIL" if overall == 1 else "PASS")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"\n===== watchdog verdict: {combined['verdict']} =====", flush=True)
    print(f"report: {json_path}", flush=True)
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["g12", "g3", "g4", "all"], required=True)
    ap.add_argument("--json", default=DEFAULT_REPORT)
    args = ap.parse_args()
    if args.gate == "all":
        sys.exit(run_all(args.json))
    sys.exit(run_one(args.gate, args.json))


if __name__ == "__main__":
    main()
