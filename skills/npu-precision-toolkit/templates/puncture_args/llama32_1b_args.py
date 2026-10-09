#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""meta-llama/Llama-3.2-1B（同构）args builder。

TorchBench 未注册此模型（只有 3.1-8B）——按 Llama-3.2-1B 官方超参 from_config 同构：
  16 层 × hidden 2048，GQA（32 query heads / 8 kv heads），RoPE theta=500000，
  vocab 128256，intermediate 8192。

新覆盖面：**GQA**（分组查询注意力——kv heads < q heads 的共享结构，此前 11 个模型
均为 MHA 或 MQA 之外的形态）+ Llama-3 系 RoPE 缩放。
"""

import torch
import torch_npu  # noqa: F401
from transformers import LlamaConfig, LlamaForCausalLM


def build_args(device="npu"):
    torch.manual_seed(42)
    cfg = LlamaConfig(
        vocab_size=128256, hidden_size=2048, intermediate_size=8192,
        num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8,  # GQA
        rope_theta=500000, max_position_embeddings=131072,
    )
    model = LlamaForCausalLM(cfg).to(device=device, dtype=torch.float16).eval()

    torch.manual_seed(123)
    input_ids = torch.randint(0, 128000, (1, 512), device=device)
    attn = torch.ones_like(input_ids)

    with torch.no_grad():
        ref = model(input_ids=input_ids, attention_mask=attn).logits.detach()

    def forward_fn(m):
        return m(input_ids=input_ids, attention_mask=attn).logits

    return [input_ids, attn], ref, model, forward_fn
