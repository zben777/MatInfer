# FlashMLA：MLA 注意力 Kernel 库

> 仓库：[deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)。

FlashMLA 是 DeepSeek 的高性能注意力算子库，支撑 DeepSeek-V4.1 在 NVIDIA GPU 与华为昇腾 NPU 上的推理。含 prefill / decode 的 token 级稀疏注意力（DSA）、FP8 / FP4 KV cache，以及把 Q-norm、Q-RoPE、attention、O-RoPE、cast-to-FP8 合成一个 kernel 的融合算子。

**版本与适用环境**：2026-09-30 版本为 breaking change——移除 Hopper（SM90）与前代模型（V3 / V3.2 / V4.0）支持，改写 FP8 / FP4 KV cache 格式。当前要求 SM100 / SM103 + CUDA 13.1+、PyTorch 2.10+；昇腾侧为 Ascend 950 NPU + CANN 9.2+。旧版本需切到 `ba89a346` commit。

| 专题 | 核心问题 |
| --- | --- |
| [本层正文：FlashMLA](../README.md#section-04) | 为什么不能只写成"FlashAttention for MLA"？ |
| [Attention 专题：FlashAttention 演进](../attention/README.md) | MLA 之前的注意力 kernel 基础是什么？ |
| [Model 层：GQA / MLA](../../01-model/README.md) | MLA 为什么能减少 KV Cache？ |

状态：仓库定位已核对，源码解读待补。

[返回高性能 Backend](../README.md)
