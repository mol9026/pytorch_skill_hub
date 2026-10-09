#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DistilBertForMaskedLM（hf_DistilBert，TorchBench 口径）args builder。

BERT 的 6 层精简（无 token_type/池化，双向 attention）。BERT 家族第二成员，
一致性确认。
"""

import torch
import torch_npu  # noqa: F401
from transformers import DistilBertConfig, DistilBertForMaskedLM


def build_args(device="npu"):
    torch.manual_seed(42)
    # distilbert-base 同构（6L×768）
    cfg = DistilBertConfig(vocab_size=30522, dim=768, n_layers=6, n_heads=12,
                           hidden_dim=3072)
    model = DistilBertForMaskedLM(cfg).to(device=device, dtype=torch.float16).eval()

    torch.manual_seed(123)
    input_ids = torch.randint(0, 30500, (1, 512), device=device)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, 400:] = 0

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        ref = out.logits.detach()

    def forward_fn(m):
        return m(input_ids=input_ids, attention_mask=attention_mask).logits

    return [input_ids, attention_mask], ref, model, forward_fn
