# 异步搬运与多阶段流水线

围绕数据就绪、计算完成和缓冲区复用，理解 Tensor Core 计算怎样与内存搬运重叠。

| 专题 | 核心问题 |
| --- | --- |
| [从 Ampere 到 Blackwell：矩阵计算与数据搬运怎样重叠](mma-memory-pipeline.md) | cp.async、TMA、MMA、WGMMA、tcgen05 怎样配合？stage 什么时候能安全覆盖？ |

[返回 Kernel 编程](../README.md)
