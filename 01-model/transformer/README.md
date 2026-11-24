# Transformer 基础

先理解模型组件的分工与数据流，再进入 Attention、归一化和具体模型结构。

| 专题 | 核心问题 |
| --- | --- |
| [层里只有一步跨位置：softmax 的分母如何分开 Attention 与 FFN](attention-and-ffn.md) | 两个子层的分界线到底在哪？为什么只有 Attention 需要缓存？FFN 凭什么开到四倍宽？ |

[Attention 深入专题](../attention/README.md) · [归一化深入专题](../normalization/README.md) · [返回模型原理](../README.md)
