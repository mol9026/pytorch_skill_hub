#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""llama（TorchBench models/llama，Meta Llama-1 官方手写实现）args builder。

非 transformers：Meta 原版 model.py（手写 RMSNorm/RoPE/attention/cache）。
权重不可得 → 随机初始化（ModelArgs 默认 7B 尺寸太大，缩小到可编译规模但保持结构）。
原版输入：(tokens bs×seq, start_pos)。eval 路径 forward(tokens, start_pos)。
"""

import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/home/z00599499/benchmark/torchbenchmark/models/llama")
from model import ModelArgs, Transformer  # noqa: E402


def build_args(device="npu"):
    torch.manual_seed(42)
    # 原版 ModelArgs 默认 7B（32L×4096）过大；同构缩小 4L×512（结构不变：
    # RMSNorm/RoPE/手写 attention 全保留）
    args = ModelArgs(dim=512, n_layers=4, n_heads=8, vocab_size=32000,
                     max_seq_len=512, max_batch_size=1)
    model = Transformer(args).to(device=device, dtype=torch.float16).eval()

    torch.manual_seed(123)
    tokens = torch.randint(1, 31999, (1, 128), dtype=torch.long, device=device)
    start_pos = 0

    with torch.no_grad():
        ref = model(tokens, start_pos).detach()

    def forward_fn(m):
        return m(tokens, start_pos)

    return [tokens, start_pos], ref, model, forward_fn
