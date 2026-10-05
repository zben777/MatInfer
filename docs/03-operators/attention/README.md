# Attention 计算专题

> 从 decode 的 Shape 和分块状态出发，理解算子内部怎样完成全局归一化与 Value 聚合。

| 专题 | 核心问题 |
| --- | --- |
| [Online Softmax](online-softmax.md) | m、ℓ、U 是什么？新最大值出现时怎样修正？为什么最终结果等价？ |
| [FlashAttention 版本演进](../../04-backends/attention/flashattention-evolution.md) | 数据流、CTA/warp 分工与 Hopper 异步流水分别改变了什么？ |

[返回核心算子](../README.md)
