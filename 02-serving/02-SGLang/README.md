# SGLang：RadixAttention 与结构化生成

> 仓库：[sgl-project/sglang](https://github.com/sgl-project/sglang)。

SGLang 是面向大模型推理的 serving runtime，以 RadixAttention 前缀缓存、结构化输出约束和调度器为核心，服务多轮对话与结构化生成等大规模请求场景。

| 专题 | 核心问题 |
| --- | --- |
| [本层正文：Prefix Cache / Radix Cache](../README.md#section-07) | 相同前缀的 KV 怎样跨请求复用？ |
| [本层正文：Scheduler](../README.md#section-03) | 每轮迭代谁运行、谁等待？ |
| [框架对比：vLLM / SGLang / TensorRT-LLM](../README.md#section-14) | 三个框架各自解决什么问题？ |
| [Backend 映射总表](../../04-backends/README.md#section-22) | 框架调用的底层 Backend 都来自哪里？ |

状态：仓库定位已核对，源码解读待补。学习时固定 commit / tag，再进入具体路径。

[返回 Serving 层](../README.md)
