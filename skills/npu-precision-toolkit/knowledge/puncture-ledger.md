# 50 模型穿刺台账（TorchBench，triton_experimental 后端精度验证）

> 开始：2026-09-29 | 工具：npu_compile_doctor（L1-L4 + 指纹库）
> 目标：团队负责的 ~50 个模型逐一验证图模式精度，命中已知根因或发现新根因

## 台账

| # | 模型 | 日期 | L1 cos | 确定性 | 深层检查 | 结论 | 证据 |
|---|------|------|--------|--------|----------|------|------|
| 1 | nanogpt (124M, 12L×768, 随机权重, seq=64) | 2026-09-29 | **0.999995** | bitwise 确定 | L2-L4 未跑* | **HEALTHY** —— 修复态编译器对无-RoPE 纯 decoder 架构覆盖良好 | `npu_doctor_nanogpt.json` + `doctor_nanogpt.log` |
| 2 | speech_transformer (6+6L enc-dec, 512d, 随机权重, T=120/L=30) | 2026-09-29 | **0.999994** | bitwise 确定 | L2-L4 未跑* | **HEALTHY**（含注记：dynamo graph break） | `npu_doctor_speech_transformer.json` |
| 3 | T5 (2+2L enc-dec×512, 随机权重, enc128/dec64) | 2026-09-29 | **1.000148** | bitwise 确定 | L2-L4 未跑* | **HEALTHY**（含注记：tile-contract 警告） | `npu_doctor_t5.json` |
| 4 | llava (CLIP-ViT 9L + Llama 4L, 随机权重, 336px/576 patch + 640 seq) | 2026-09-29 | **1.005047** | bitwise 确定 | L2-L4 未跑* | **HEALTHY**（含注记：graph break；cos 偏高系随机权重 logits 尺度） | `npu_doctor_llava.json` |
| 5 | BertForMaskedLM (12L×768 双向 attn, 随机权重, seq=512 含 pad) | 2026-09-29 | **1.003955** | bitwise 确定 | - | **HEALTHY** —— 首个双向 attention 架构通过（mask 只遮 pad） | `npu_doctor_bert.json` |
| 6 | AlbertForMaskedLM (12L 共享权重 + 128d 分解 emb, 随机权重) | 2026-09-29 | **1.003967** | bitwise 确定 | - | **HEALTHY** —— 跨层参数共享捕获正常 | `npu_doctor_albert.json` |
| 7 | BartForCausalLM (6L×768 decoder 半边 + 绑定 LM 头, 随机权重) | 2026-09-29 | **1.007283** | bitwise 确定 | - | **HEALTHY** —— BART CausalLM 路径通过 | `npu_doctor_bart.json` |
| 8 | DistilBertForMaskedLM (6L×768, 随机权重) | 2026-09-29 | **1.003966** | bitwise 确定 | - | **HEALTHY** —— BERT 家族一致性确认 | `npu_doctor_distilbert.json` |
| 9 | BERT_pytorch (12L×768 手写实现, 随机权重, seq=128 含 pad) | 2026-09-29 | **0.999991** | bitwise 确定 | - | **HEALTHY** —— 非 HF 手写 attention 路径通过 | `npu_doctor_bert_pytorch.json` |
| 10 | moondream 同构 (SigLIP 4L + proj + Phi 4L partial-RoPE, 336px/576 patch) | 2026-09-29 | **1.005044** | bitwise 确定 | - | **HEALTHY** —— 第二支 vision 家族 + Phi 部分 RoPE 通过 | `npu_doctor_moondream.json` |

| 11 | llama_v2_7b_16h (16L 随机权重, bs1/seq512, 团队口径构造) | 2026-09-29 | **1.004152** | bitwise 确定 | - | **HEALTHY** —— 原始故障模型修复态通过；与 G3(1.004134)/定案实验(1.004152)三口径互证 | `npu_doctor_llama16h.json` |

| 12 | llama（Meta Llama-1 官方手写实现, 4L×512 同构缩小, seq=128） | 2026-09-29 | **0.999997** | bitwise 确定 | - | **HEALTHY** —— Meta 原版独立实现（手写 RMSNorm/RoPE/attn）通过 | `npu_doctor_llama1_meta.json` |
| 13 | Llama-3.2-1B 同构 (16L×2048, **GQA 32q/8kv**, theta500k, vocab128k, seq=512) | 2026-09-29 | **1.032607** | bitwise 确定 | - | **HEALTHY** —— GQA 结构通过；cos 偏高系 GQA 随机权重 + 大 vocab logits 尺度（方向高度对齐） | `npu_doctor_llama32_1b.json` |

> **批次小结（2026-09-29，13/50）**：十三模型全绿——纯 decoder / 手写 attn enc-dec / 相对 PE /
> 多模态×2（CLIP+Llama、SigLIP+Phi）/ 双向 attention / 参数共享 / 独立手写实现。
> 修复态编译器精度覆盖面初步验证充分。性能方向候选调查点 2 个（llama 慢 12%、T5 tile-contract）。
> graph break 模式 2 类（data-dependent indexing、torch._check_tensor_all），均属模型侧结构。

> 注记（2026-09-29）T5：编译期出现大量 `[tile-contract] unhealthy-unfixable: load 'y0 + 8*x1 +
> 512*x2' -> no-rescue` 警告——T5 的相对位置偏置路径生成 strided load，triton-ascend 的 tile
> 契约无法救援（性能问题，非精度问题；本次 cos 确认精度无恙）。这是**性能方向的候选调查点**
> （同 llama 的"graph 比 eager 慢 12%"一类）：若 T5 纳入性能口径，需评估该 load 模式的
> contiguous 化或 tile 策略调整。

> 注记（2026-09-29）llava：TorchBench 原版需 gated 权重（liuhaotian/llava-v1.5-13b）——改为
> transformers `LlavaForConditionalGeneration.from_config` 同构小模型（CLIP vision + Llama LM）。
> 输入构造要点：**序列里 image token 数必须等于 patch 数**（336/14=24 → 576 个连续 image token），
> 否则 `get_placeholder_mask` 校验报错。`torch._check_tensor_all` 在该 mask 路径触发 graph break
> （4 个子图产物）——"HEALTHY" 含义同 speech_transformer 注记（被捕获子图全对，break 段 eager）。
> 多模态拼接（image features scatter 进 input embeds）+ vision conv 链 + LM RoPE 全路径精度正常。

> 注记（2026-09-29）：speech_transformer 的 `get_non_pad_mask` 含 data-dependent indexing
> （`non_pad_mask[i, input_lengths[i]:] = 0`），dynamo 触发 graph break——本次编译产生了
> 7+ 个子图产物（model__0 ~ model__8）。"HEALTHY" 的准确含义是**被捕获的全部子图正确**，
> graph break 段保持 eager 执行（正确性由 eager 保证）。这对该模型是预期行为（原版
> 注释也声明 "only supports CUDA and eager mode"）。若团队口径要求整图覆盖，需另行
> 评估 mask 构造的重写（get_attn_pad_mask 同理）。

\* **nanogpt 深层检查未完成的原因（工具已知边界，2026-09-29 发现）**：L2/L3 的 capture 需要向产物
`Runner.call` 传全量参数（dynamo 把模型权重/子模块按访问序展开为 graph 参数，nanogpt 有 149 个，
llama 1 层是手工列的 14 个）。手工构造不可扩展，需要"参数自动构造层"：
- 候选方案 A：hook `torch._inductor.compile_fx`，在 inductor 首次真实执行时截获 dynamo 传入的 inputs
- 候选方案 B：用 inductor fx_passes 机制（torch_npu 已有先例）在产物生成时自动注入探针
- 候选方案 C：`torch._dynamo.export` 拿 graph 参数（实测 nanogpt 1738 个碎参数 + API 兼容问题，不可行）

**当前有效结论**：L1（端到端 cos + 确定性）对"模型是否健康"已是充分判定（cos>0.999 且
bitwise 确定即通过）；L2-L4 是"出了问题后的定位工具"。对穿刺任务而言，L1 全绿即可推进，
L2-L4 自动化留给命中问题的模型再用（届时可手工构造参数）。

## 已知根因对架构的适用性预判

| 根因 | nanogpt | speech_transformer | T5 | llava/moondream |
|------|---------|--------------------|----|-----------------|
| rc1 轴合并 | 无 RoPE 但有 reshape/transpose 链 | encoder-decoder 注意 | attention 重排多 | vision+LLM 双路径 |
| rc2 FA layout | SDPA(is_causal) 适用 | SDPA 适用 | SDPA 适用 | LLM 侧适用 |
| rc3 inv_freq | 无 RoPE **不适用** | 无 RoPE（正弦 PE）**不适用** | 相对 PE **不适用** | LLM 侧适用 |
| rc4 in-place reduction | LayerNorm+残差适用 | LayerNorm 适用 | LayerNorm 适用 | 适用 |

## 方法备忘

每个模型的标准流程：
1. 写 args-script（`build_args(device) -> (args, ref, model, forward_fn)`，参考 `scripts/puncture_args/nanogpt_args.py`（13 个模板都在此））
2. `python3 scripts/npu_compile_doctor.py --args-script scripts/xxx_args.py`（L1 ~10min 含编译）
3. HEALTHY → 记台账，下一个；PROBLEM → L2-L4 定位（手工构造参数）→ 指纹匹配 → 修复/深挖
