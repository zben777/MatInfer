# FFN 与门控

前馈网络内部那一层非线性，从 ReLU 一路换到 SwiGLU。想看清这条替换线，先抓住"门从哪里来"这一个问题。

| 专题 | 核心问题 |
| --- | --- |
| [门从哪里来：从 ReLU 到 SwiGLU](swiglu.md) | ReLU、GELU、SiLU 与 SwiGLU 是什么关系？门控多出来的那块矩阵买到了什么？为什么中间维要缩到 8d/3？ |

[Transformer 基础专题](../transformer/README.md) · [MoE 深入专题](../moe/README.md) · [归一化深入专题](../normalization/README.md) · [返回模型原理](../README.md)
