# DeepGEMM：FP8 / FP4 GEMM Kernel 库

> 仓库：[deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)。本地浅克隆 commit `5081f851`，版本 2.8.2。

DeepGEMM 已从早期的 FP8 GEMM 库演进为统一的 Tensor Core 算子库：FP8 / FP4 / BF16 GEMM、Grouped GEMM、Mega MoE（通信与计算重叠）、MQA indexer、HyperConnection，运行时由 DeepJIT 编译。它刻意不用 CUTLASS 的重模板，定位是学习 GPU kernel 优化的干净样本。

**适用环境（硬门槛）**：SM90（H100 / H800）与 SM100（B200 / GB200），CUDA 12.9+、C++20。4090（SM89）、3090（SM86）不支持；A100 仅保留一套 legacy Triton 实现。

本目录正文按 Day 编号推进，从调用链入门，逐步进入 Tiling 与 TMA / WGMMA 流水线。

| Day | 专题 | 核心问题 |
| --- | --- | --- |
| [Day 01](Day01-调用链.md) | 一行调用，穿过五层 | 从 `fp8_gemm_nt` 到 CUDA 模板中间经过哪几层？为什么安装期不编译？ |
| Day 02（待补） | CTA Tiling | 大矩阵切成多大块、分块后算术强度由什么决定？先看 [Kernel 层已有的源码解读](../../05-kernel-engineering/deepgemm/cta-tiling.md)。 |
| Day 03（待补） | Warp / MMA Tiling | warp 如何分工、MMA 指令怎样喂满 Tensor Core？ |
| [本层正文：DeepGEMM](../README.md#section-06) | 它解决什么 GEMM 问题，scale 布局和 NT 约定是什么？ |
| [本层正文：Grouped GEMM / for MoE](../README.md#section-10) | Grouped GEMM 为什么适合 MoE？ |

状态：Day 01 已写成源码导读；Day 02 / 03 待补。

[返回高性能 Backend](../README.md)
