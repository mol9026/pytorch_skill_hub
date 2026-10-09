#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""speech_transformer（TorchBench）args builder —— npu_compile_doctor --args-script 用。

绕开原 config.py 的数据集依赖（kaldi_io + S3 tarball 内网不可达）：
直接用同款超参构造 Encoder+Decoder，随机输入对齐 get_module 形态。

架构（对四根因的预期）：
  - Sinusoidal PE（非 learned、非 RoPE）→ rc3 不适用
  - 手写 scaled_dot_product_attention（mask 加性）→ rc2 路径不同（可能不走 aclnn FA）
  - LayerNorm+残差 → rc4 适用
  - LFR/reshape 链 → rc1 观察
"""

import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/home/z00599499/benchmark/torchbenchmark/models/speech_transformer")
from speech_transformer.transformer import Transformer  # noqa: E402
from speech_transformer.transformer.encoder import Encoder  # noqa: E402
from speech_transformer.transformer.decoder import Decoder  # noqa: E402


def build_args(device="npu"):
    # 与 config.py SpeechTransformerTrainConfig 同款超参
    D_INPUT, N_LAYERS_ENC, N_HEAD = 80, 6, 8
    D_K, D_V, D_MODEL, D_INNER = 64, 64, 512, 2048
    N_LAYERS_DEC, D_WORD_VEC = 6, 512
    SOS_ID, EOS_ID, N_TGT_VOCAB = 1, 2, 4235  # aishell 1char 词表规模量级

    torch.manual_seed(42)
    encoder = Encoder(D_INPUT, N_LAYERS_ENC, N_HEAD, D_K, D_V, D_MODEL, D_INNER, dropout=0.0)
    decoder = Decoder(SOS_ID, EOS_ID, N_TGT_VOCAB, D_WORD_VEC,
                      N_LAYERS_DEC, N_HEAD, D_K, D_V, D_MODEL, D_INNER, dropout=0.0,
                      tgt_emb_prj_weight_sharing=True)
    model = Transformer(encoder, decoder).to(device=device, dtype=torch.float16).eval()

    # 输入对齐 get_module：(padded_input N×T×80, input_lengths N, padded_target N×L)
    bs, T, L = 1, 120, 30
    torch.manual_seed(123)
    padded_input = torch.randn(bs, T, D_INPUT, device=device).half()
    input_lengths = torch.full((bs,), T, dtype=torch.long, device=device)
    padded_target = torch.randint(3, N_TGT_VOCAB, (bs, L), device=device)
    # 最后一位 eos（语义合理即可，对拍双方同输入）

    with torch.no_grad():
        ref = model(padded_input, input_lengths, padded_target)[0].detach()  # (pred, gold) 取 pred

    def forward_fn(m):
        return m(padded_input, input_lengths, padded_target)[0]

    return [padded_input, input_lengths, padded_target], ref, model, forward_fn
