#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlbertForMaskedLM（hf_Albert，TorchBench 口径）args builder。

独有架构特征：**跨层参数共享**（所有 12 层共享同一组 transformer 权重）+
分解式 embedding（embedding_size×hidden_size 两段投影）。dynamo 对"同一模块
被捕获多次"的图结构是新覆盖面。双向 attention（同 BERT）。
"""

import torch
import torch_npu  # noqa: F401
from transformers import AlbertConfig, AlbertForMaskedLM


def build_args(device="npu"):
    torch.manual_seed(42)
    # albert-base-v2 同构：12 层共享、hidden 768、embedding 128
    cfg = AlbertConfig(vocab_size=30000, hidden_size=768, embedding_size=128,
                       num_hidden_layers=12, num_attention_heads=12,
                       intermediate_size=3072)
    model = AlbertForMaskedLM(cfg).to(device=device, dtype=torch.float16).eval()

    torch.manual_seed(123)
    input_ids = torch.randint(0, 29900, (1, 512), device=device)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, 400:] = 0
    token_type_ids = torch.zeros_like(input_ids)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        ref = out.logits.detach()

    def forward_fn(m):
        return m(input_ids=input_ids, attention_mask=attention_mask,
                 token_type_ids=token_type_ids).logits

    return [input_ids, attention_mask, token_type_ids], ref, model, forward_fn
