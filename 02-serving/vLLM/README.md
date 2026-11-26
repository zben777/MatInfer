# vLLM：通用推理引擎

> 仓库：[vllm-project/vllm](https://github.com/vllm-project/vllm)。

vLLM 是通用大模型推理与服务引擎，核心是 PagedAttention 的 KV Cache 管理、Continuous Batching 调度、CUDA Graph 捕获，以及 TP / PP / EP / PD 并行执行，覆盖在线 serving 与离线批量推理。

| 专题 | 核心问题 |
| --- | --- |
| [本层正文：KV Cache Manager](../README.md#section-06) | 逻辑块到物理块的映射怎样支撑动态请求？ |
| [本层正文：Continuous Batching](../README.md#section-04) | 请求完成后怎样补入新请求？ |
| [框架对比：vLLM / SGLang / TensorRT-LLM](../README.md#section-14) | 三个框架各自解决什么问题？ |
| [Backend 映射总表](../../04-backends/README.md#section-22) | 框架调用的底层 Backend 都来自哪里？ |

状态：仓库定位已核对，源码解读待补。学习时固定 commit / tag，再进入具体路径。

[返回 Serving 层](../README.md)
