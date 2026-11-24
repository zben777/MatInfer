# TileKernels：TileLang 算子库

> 仓库：[deepseek-ai/TileKernels](https://github.com/deepseek-ai/TileKernels)。

TileKernels 是用 [TileLang](https://github.com/tile-ai/tilelang)（支持多硬件后端的 DSL）实现的数十个深度优化算子，覆盖 MoE routing、量化（FP8 / FP4 casting、SwiGLU+quant 融合）、Engram、流形超连接（mHC）、RoPE 等，多数性能贴近硬件算力或带宽上限，已用于 DeepSeek 内部训练与推理。

**环境**：NVIDIA 与华为昇腾双后端，运行时自动选择，同一套 Python API 两端可跑（2026-09-30 加入昇腾支持）。

| 专题 | 核心问题 |
| --- | --- |
| [本层正文：自研 CUDA / Triton Kernel 为什么存在](../README.md#section-13) | DSL 这一路线在 Backend 层的定位是什么？ |
| [本层正文：Backend 映射总表](../README.md#section-22) | 这类算子在本层归到哪里？ |

状态：仓库定位已核对，源码解读待补。

[返回高性能 Backend](../README.md)
