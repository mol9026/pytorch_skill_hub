#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BartForCausalLM（hf_Bart 的 CausalLM 形态）args builder。

BART 的 encoder-decoder 结构取 CausalLM 视角（只走 decoder 半边 + LM 头）。
注：TorchBench hf_Bart 注册的是 AutoModelForSeq2SeqLM（encoder+decoder 全跑）；
用户点名 BartForCausalLM——按 CausalLM 形态测（decoder-only 路径 + 权重绑定 LM 头）。
"""

import torch
import torch_npu  # noqa: F401
from transformers import BartConfig, BartForCausalLM


def build_args(device="npu"):
    torch.manual_seed(42)
    # bart-base 同构（6L×768）；CausalLM 只用 decoder
    cfg = BartConfig(vocab_size=50265, hidden_size=768, num_hidden_layers=6,
                     num_attention_heads=12, intermediate_size=3072,
                     d_model=768, encoder_layers=6, decoder_layers=6)
    model = BartForCausalLM(cfg).to(device=device, dtype=torch.float16).eval()

    torch.manual_seed(123)
    input_ids = torch.randint(0, 50200, (1, 512), device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        ref = out.logits.detach()

    def forward_fn(m):
        return m(input_ids=input_ids, attention_mask=attention_mask).logits

    return [input_ids, attention_mask], ref, model, forward_fn
