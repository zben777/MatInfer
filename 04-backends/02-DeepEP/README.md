# DeepEP：MoE 通信 Backend

> 仓库：[deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)。

DeepEP（DeepEveryParallel）是高性能通信库，提供 MoE dispatch / combine 的高吞吐低延迟 expert-parallel all-to-all GPU kernel（含 FP8 dispatch），以及冗余专家的 NVLink 权重与梯度交换；也提供 PP / CP / DP 与远端内存访问（Engram）的实验原语。通信 kernel 由 DeepJIT 运行时编译。

**版本与环境**：V2.5 起彻底移除 V1 与 NVSHMEM 依赖，改用轻量 NCCL Gin backend；昇腾侧见 DeepEP-Ascend。

| 专题 | 核心问题 |
| --- | --- |
| [本层正文：DeepEP "通信 Backend"](../README.md#section-11) | 通信 kernel 与计算 kernel 在 Backend 层有什么不同？ |
| [本层正文：计算与通信 Backend 的区别](../README.md#section-12) | 两者的优化目标差异在哪？ |
| [本层正文：Grouped GEMM / for MoE](../README.md#section-10) | 通信之后 expert 侧怎么算？ |

状态：仓库定位已核对，源码解读待补。

[返回高性能 Backend](../README.md)
