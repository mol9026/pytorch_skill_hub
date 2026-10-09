---
name: puncture-model-verify
description: >-
  模型穿刺验证流程：为新模型编写 build_args 同构用例、跑 doctor L1、
  按台账格式记录结论（含 graph break / cos>1 注记规则）。工具与模板实体在
  npu-precision-toolkit 中，本 skill 是流程层协议。
  当用户给出模型清单要求逐个验证图模式精度、或说"穿刺""跑一下 XX 模型"时使用。
  Use when verifying torch.compile precision for a list of models ("puncture"),
  writing build_args cases, or appending entries to the puncture ledger.
---

# puncture-model-verify：模型穿刺验证流程

## 概述

团队 50 模型图模式精度验证计划的执行协议。工具（doctor/watchdog）与 13 个
args 模板实体在 **`npu-precision-toolkit`**（`skills/npu-precision-toolkit/`）——
本 skill 只定义流程与判定规则，避免工具双份漂移。

## 单模型标准流程

```
1. 写 args 脚本       scripts/puncture_args/<model>_args.py
2. 跑 doctor L1       ~10min（含编译）
3. HEALTHY → 记台账    knowledge/puncture-ledger.md 追加一行
4. PROBLEM → 转排查    npu-precision-toolkit 的 L2-L4 + 指纹匹配
```

### args 脚本契约（必须严格遵守）

```python
def build_args(device="npu"):
    """返回 (args_list, ref, model, forward_fn) 四元组。"""
    torch.manual_seed(42)
    # 1. 构造模型：优先 from_config 随机权重同构（无需官方权重/门控下载）
    #    缩小层数（如 12L→4L）控制编译时间，保留架构特征
    cfg = ModelConfig(...)
    model = Model(cfg).to(device=device, dtype=torch.float16).eval()
    # 2. 随机输入（口径固定：bs=1、seq 128-512、含 pad 的场景要造 pad）
    torch.manual_seed(123)
    inputs = ...
    # 3. eager 参考值（在构造侧算好，doctor 只对拍）
    with torch.no_grad():
        ref = model(*inputs).logits.detach()
    # 4. forward_fn：闭包捕获输入，签名 forward_fn(m) -> 输出
    def forward_fn(m):
        return m(*inputs).logits
    return [*inputs], ref, model, forward_fn
```

### 运行

```bash
export TORCHINDUCTOR_COMPILE_THREADS=1   # 必须 shell export，os.environ 不够
python3 scripts/npu_compile_doctor.py \
  --args-script scripts/puncture_args/<model>_args.py \
  --report $NPU_EVIDENCE_DIR/npu_doctor_<model>.json
```

## 判定规则（读数三查）

| 检查 | 通过线 | 不达标时 |
|------|--------|---------|
| cos | > 0.999 | < 0.99 → L2 定位；0.99-0.999 边界 → 复跑 + L4 |
| 确定性 | 两次执行 bitwise 一致 | 不一致 → 并发竞争指纹，进 L2 |
| **cos > 1** | ≤ ~1.05 且方向对齐 | 随机权重 + 大 vocab 的 logits 尺度会自然产生 1.00-1.03，**注记而非报错**（Llama-3.2-1B 案例 1.032607 全绿） |

### 三个必查注记（不注记 = 台账失真）

1. **graph break**：编译产生多个子图（`model__0 ~ model__N`）→ 注记"HEALTHY 的
   含义是被捕获子图全对，break 段 eager"。已知触发：data-dependent indexing
   （speech_transformer）、`torch._check_tensor_all`（llava）。
2. **cos>1**：见上表，注记随机权重 logits 尺度原因。
3. **编译期警告**：如 T5 的 tile-contract unhealthy-unfixable——精度无恙但属性能
   调查点，注记留档。

## 造输入的绕依赖三板斧（模型拉不下来时）

1. **from_config 同构**：`Model.from_config(cfg)` + 随机初始化，绕开权重下载与门控
   （llava/moondream 官方权重 gated，同构小模型验证架构覆盖即可）
2. **直接构造子模块**：包级 import 失败时直接 `Encoder(cfg) + Decoder(cfg)` 拼
   （speech_transformer 案例）；相对导入报错 → sys.path 加包根
3. **手写实现对照**：HF 拉不到就手写等价结构（BERT_pytorch 案例，还额外覆盖了
   非 HF 路径）

### 多模态特殊处理

- llava 类：**序列里 image token 数必须等于 patch 数**（336/14=24 → 576 个连续
  image token），否则 `get_placeholder_mask` 校验报错
- vision 家族选型：CLIP 与 SigLIP 各测一支即覆盖两类（conv 链 + proj + LM）

## 台账记录格式（追加到 puncture-ledger.md）

```markdown
| # | 模型 (规格, 随机权重, 输入口径) | 日期 | L1 cos | 确定性 | 结论 | 证据 |
|---|------|------|--------|--------|------|------|
| N | <name> (<层数×维度>, 随机权重, <seq>) | YYYY-MM-DD | **0.999xxx** | bitwise 确定 | **HEALTHY** —— <架构特征一句话> | `npu_doctor_<name>.json` |
```

注记紧跟该行后用 `> 注记（日期）<模型>：...` 补充。每完成一批写批次小结。

## 已验证口径（2026-09-29，13/50）

纯 decoder / 手写 attn enc-dec / 相对 PE / 多模态×2（CLIP+Llama、SigLIP+Phi）/
双向 attention / 参数共享 / GQA / 独立手写实现。全绿基线见
`npu-precision-toolkit/knowledge/puncture-ledger.md`。
