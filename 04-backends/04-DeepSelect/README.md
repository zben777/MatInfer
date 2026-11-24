# DeepSelect：DSA 的 TopK Kernel

> 仓库：[deepseek-ai/DeepSelect](https://github.com/deepseek-ai/DeepSelect)。

DeepSelect 是 DeepSeek 稀疏注意力（DSA，用于 V3.2 / V4 / V4.1）与采样器所用的高性能 TopK 算子，相比原生 `torch.topk` 可达 2～20x 加速。支持 NVIDIA CUDA 与华为昇腾，2026-09-30 发布昇腾 TopK kernel。

工作场景聚焦两类：Lightning Indexer（bf16，任意 batch / vocab）、Sampler（词表 top-k 采样）；最快算法取决于 dtype、batch、vocab_size 与 topk 的组合。仓库自带算法与实现的 [深读文档](https://github.com/deepseek-ai/DeepSelect/blob/main/docs/DeepSelect-deep-dive.zh.md)。

| 专题 | 核心问题 |
| --- | --- |
| [本层正文：FlashMLA](../README.md#section-04) | TopK 选出的 indices 怎样喂给稀疏注意力 kernel？ |
| [Model 层：DeepSeek 相关](../../01-model/README.md) | DSA 的 Lightning Indexer 在模型里扮演什么角色？ |

状态：仓库定位已核对，源码解读待补。

[返回高性能 Backend](../README.md)
