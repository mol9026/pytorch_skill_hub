#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BERT_pytorch（codertimo 独立实现，TorchBench 口径）args builder。

绕开 CorpusGenerator/WordVocab/BERTDataset 数据管道——BERT 类只需
(x: token ids, segment_info) 两个 tensor。直接构造同构模型 + 随机输入。

架构：手写实现（非 transformers）——自写 multi-head attention（非 SDPA）、
手写 LayerNorm、mask 由 (x>0) 生成（双向 + pad mask）。覆盖面：非 HF 实现的
手写 attention 路径。
"""

import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/home/z00599499/benchmark/torchbenchmark/models/BERT_pytorch")
from bert_pytorch.model import BERT  # noqa: E402


def build_args(device="npu"):
    torch.manual_seed(42)
    # benchmark 同款：vocab 20000, 12L×768×12 heads
    model = BERT(vocab_size=20000, hidden=768, n_layers=12, attn_heads=12,
                 dropout=0.1).to(device=device, dtype=torch.float16).eval()

    torch.manual_seed(123)
    # seq_len 用 benchmark 默认（parse_args 默认 20；加长到 128 提高覆盖）
    x = torch.randint(1, 20000, (1, 128), device=device)
    segment_info = torch.zeros_like(x)
    # 部分 pad（激活 mask 路径）
    x[0, 100:] = 0

    with torch.no_grad():
        ref = model(x, segment_info).detach()

    def forward_fn(m):
        return m(x, segment_info)

    return [x, segment_info], ref, model, forward_fn
