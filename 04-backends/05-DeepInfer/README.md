# DeepInfer：推理框架

> 仓库：[deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer)。

DeepInfer 是支撑 DeepSeek 大规模线上服务的高性能推理框架，本次开源提供离线推理能力。当前支持 DeepSeek-V4.1-Flash，特性包括优化过的 kernel / model / sampler runtime、Attention DP 与均衡 CP、MoE EP + EPLB、DSpark 与置信度调度验证、SWA 有界回放、零开销异步 batch scheduler、batch-invariant 推理、确定性采样、结构化输出。

| 专题 | 核心问题 |
| --- | --- |
| [Serving 层：推理框架与服务系统](../../02-serving/README.md) | 调度、显存与并行策略在框架层怎么组织？ |
| [本层正文：Backend 映射总表](../README.md#section-22) | 框架调用的底层 Backend 都来自哪里？ |

状态：仓库定位已核对，源码解读待补。框架本体主要归 Serving 层，此处只保留与 Backend 的衔接入口。

[返回高性能 Backend](../README.md)
