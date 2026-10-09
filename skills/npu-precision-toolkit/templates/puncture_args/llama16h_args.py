#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""llama_v2_7b_16h（TorchBench 团队口径）args builder —— doctor L1 穿刺记录。

与 watchdog G3 完全同构的构造（这是台账记录用，补 doctor 的 bitwise 确定性检查）。
注：保持 TorchBench 原样构造（inv_freq 被 from_config+.half() 量化——09-29 定案
实验已证明该量化在修复态编译器下影响为噪声级，cos 实测无差异）。
"""

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, LlamaConfig


def build_args(device="npu"):
    torch.manual_seed(42)
    config = LlamaConfig(num_hidden_layers=16)  # TorchBench 原样（7b 尺寸 16 层）
    model = AutoModelForCausalLM.from_config(config).to(device=device, dtype=torch.float16).eval()

    torch.manual_seed(123)
    input_ids = torch.randint(0, config.vocab_size, (1, 512), device=device)
    attn = torch.ones_like(input_ids)

    with torch.no_grad():
        ref = model(input_ids=input_ids, attention_mask=attn).logits.detach()

    def forward_fn(m):
        return m(input_ids=input_ids, attention_mask=attn).logits

    return [input_ids, attn], ref, model, forward_fn
