#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nanogpt（TorchBench）args builder —— npu_compile_doctor --args-script 用。

契约：build_args(device) -> (args_list, ref_output, model, forward_fn)
架构预期：
  - 无 RoPE（learned wpe）→ 根因1/3 不适用
  - F.scaled_dot_product_attention(is_causal=True) → 根因2（FA layout）适用
  - LayerNorm+残差 → 根因4（in-place reduction）适用
"""

import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/home/z00599499/benchmark/torchbenchmark/models/nanogpt")
from model import GPT, GPTConfig  # noqa: E402


def build_args(device="npu"):
    torch.manual_seed(42)
    cfg = GPTConfig()
    gpt = GPT(cfg).to(device=device, dtype=torch.float16).eval()

    torch.manual_seed(123)
    idx = torch.randint(1, cfg.vocab_size, (1, 64), device=device)

    with torch.no_grad():
        ref = gpt(idx).detach()

    def forward_fn(model):
        return model(idx)

    return [idx], ref, gpt, forward_fn
