# Attention Backend 专题

> 从算法状态进入数据流、线程分工和目标硬件，区分 prefill 与 decode 的实现需求。

| 专题 | 核心问题 |
| --- | --- |
| [FlashAttention v1 / v2 / v3](flashattention-evolution.md) | 每个版本改变了什么？异步流水有哪些必须满足的依赖？ |
| [Online Softmax 推导](../../03-operators/attention/online-softmax.md) | 分块状态怎样合并？为什么还要维护输出分子？ |

[返回高性能 Backend](../README.md)
