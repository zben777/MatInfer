# 算子融合专题

从一次 decode 的数据流出发，理解融合怎样改变中间张量、访存与启动，再讨论实现约束和验证方法。

| 专题 | 核心问题 |
| --- | --- |
| [算子融合：Decode 中怎样减少访存与启动开销](operator-fusion.md) | 常见组合能省什么？Flash-Decoding、Epilogue、编译器怎样分工？如何组织真实面试案例？ |

[返回核心算子](../README.md) · [Kernel 编程](../../05-kernel-engineering/README.md)
