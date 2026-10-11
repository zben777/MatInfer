# DeepSeek Kernel 源码导读

顺着 DeepGEMM 这类开源实现的源码，一层一层看清一个 GEMM kernel 到底是怎么被组织出来的。

DeepGEMM 在 README 里把自己的定位写得很直白：它刻意只保留少量核心 kernel，不用 CUTLASS 的重模板，是"a clean and accessible resource for learning NVIDIA GPU kernel optimization techniques"。这一系列就顺着这条线走——每篇盯住一层，每一步都拿它的源码收口。

| 专题 | 核心问题 |
| --- | --- |
| [第一刀切在哪：CTA Tiling](cta-tiling.md) | 大矩阵切成多大的块、交给多少个 CTA？分块后的算术强度由什么决定，为什么它跟 K 无关？ |

[返回 Kernel 编程](../README.md)
