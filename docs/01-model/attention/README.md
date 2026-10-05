# Attention 专题

按问题拆分原理与实现，先理解计算依赖，再讨论缓存和性能。

初学可先阅读 [Transformer 基础：Attention 与 FFN 的分工](../transformer/attention-and-ffn.md)，再进入以下专题。

| 专题 | 核心问题 |
| --- | --- |
| [GQA 的多个 Query 头怎样共享 KV](gqa-head-sharing.md) | 用一个数值例子理解头分组与缓存节省，再进入实现 |
| [MHA 各头是否独立](mha-head-independence.md) | 子计算独立意味着什么？输出、梯度与 GPU 执行怎样联系？ |
| [Linear Attention](linear-attention.md) | 怎样避免显式构造 N × N 矩阵？固定状态保存了什么？ |
| [手写 GQA](gqa-from-scratch.md) | Q 头怎样映射到 KV 头？怎样处理 Mask 和 KV Cache？ |

[返回模型原理](../README.md)
