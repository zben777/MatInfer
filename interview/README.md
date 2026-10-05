# 面试与自测索引

当前从原稿整理了 **70 道题目**。它们是学习与复习的问题池，尚未补齐完整答案，也没有可核验的真实面试来源，统一按自测 / 面试练习题处理。

题目在对应知识层中维护；此页提供集中入口。后续每个专题逐步补充简要回答、正文链接、追问和来源，无需为了篇数凑题。

[总导航](../导航.md) · [学习路线](../roadmap/README.md)

## 回答层次

- 基础理解：解释是什么与为什么。
- 原理推导：说明公式、Shape、复杂度与显存估算。
- 工程取舍：讨论实现、收益、代价与适用条件。
- 场景分析：提出定位思路、验证方法与指标。
- 实践追问：用自己的代码、实验或复盘支撑回答。

## 模型原理

[对应内容与题目](../docs/01-model/README.md#section-11)

已展开的专题自测：[MHA 各头独立性](../docs/01-model/attention/mha-head-independence.md#面试与自测) · [Linear Attention](../docs/01-model/attention/linear-attention.md#面试与自测) · [手写 GQA](../docs/01-model/attention/gqa-from-scratch.md#面试与自测)。这些是新增专题练习，不计入下方原稿的 70 道题。

1. **一个 Transformer Block 的完整流程是什么？**
2. **为什么使用 RMSNorm？**
3. **MHA、MQA、GQA、MLA 有什么区别？**
4. **MLA 为什么可以减少 KV Cache？**
5. **RoPE 为什么作用于 Q/K？**
6. **Prefill 和 Decode 的 Attention 有什么区别？**
7. **DeepSeekMoE 从 Router 到 Expert 的完整流程是什么？**
8. **为什么 Decode 往往比 Prefill 更 memory-bound？**

## 推理框架与服务系统

[对应内容与题目](../docs/02-serving/README.md#section-16)

1. vLLM 一次 Request 的生命周期是什么？
2. Continuous Batching 为什么可以提高吞吐？
3. Prefill 和 Decode 的性能特征有什么区别？
4. 为什么 KV Cache 要采用 Page / Block 管理？
5. Prefix Cache 和普通 KV Cache 有什么区别？
6. Radix Cache 解决什么问题？
7. Chunked Prefill 为什么能降低 Decode interference？
8. CUDA Graph 为什么特别适合 Decode？
9. TP / PP / EP 分别切什么？
10. PD 分离的收益和代价是什么？

## 核心算子与计算特征

[对应内容与题目](../docs/03-operators/README.md#section-18)

1. GEMM 的 $`M`$、$`N`$、$`K`$ 分别代表什么？
2. 为什么小 $`M`$ GEMM 效率差？
3. 为什么 $`K`$ 很小时 Tensor Core 利用率可能低？
4. [Softmax 和 Online Softmax 有什么区别？](../docs/03-operators/attention/online-softmax.md)
5. [FlashAttention 为什么减少 HBM 访问？](../docs/04-backends/attention/flashattention-evolution.md)
6. Decode Attention 为什么常常 memory-bound？
7. RMSNorm 为什么通常偏 memory-bound？
8. [Fusion 为什么能提升性能？做过哪些融合工作，怎样验证？](../docs/03-operators/fusion/operator-fusion.md)
9. Grouped GEMM 为什么适合 MoE？
10. 怎么判断一个算子是 compute-bound 还是 memory-bound？

## 高性能 Backend

[对应内容与题目](../docs/04-backends/README.md#section-23)

1. [FlashAttention 为什么快？v1 / v2 / v3 分别改变了什么？](../docs/04-backends/attention/flashattention-evolution.md)
2. [Online Softmax 在 FlashAttention 里的作用是什么？](../docs/03-operators/attention/online-softmax.md)
3. FlashMLA 和 FlashAttention 有什么关系？
4. Decode Attention 为什么需要 Split-KV？
5. GEMM 为什么需要 Tiling？
6. Tile 是越大越好吗？
7. Double Buffer / Pipeline 怎么隐藏延迟？
8. Producer–Consumer 模式是什么？
9. Grouped GEMM 怎么映射不同 Expert？
10. Vector Load 为什么有效？
11. Register Pressure 和 Occupancy 什么关系？
12. 怎么判断一个 Kernel 已经优化到顶？

## Kernel 编程与工程

[对应内容与题目](../docs/05-kernel-engineering/README.md#section-30)

1. CUDA 的 Grid / Block / Warp / Thread 是什么关系？
2. 当前 NVIDIA CUDA 中 Warp 的大小是多少？这怎样影响线程映射？
3. `__syncthreads()` 同步的范围是什么？
4. Warp Shuffle 和 Shared Memory Reduction 有什么区别？
5. Coalescing 和 Vectorized Load 有什么区别？
6. 为什么 `float4` 要考虑 alignment？
7. Shared Memory 为什么会出现 Bank Conflict？
8. Register Pressure 为什么会影响 Occupancy？
9. Triton 和 CUDA 的编程模型有什么区别？
10. Triton Layout 是什么？
11. CUTLASS 和 CuTe 是什么关系？
12. [`cp.async` 和 TMA 有什么区别？](../docs/05-kernel-engineering/pipeline/mma-memory-pipeline.md)
13. [Producer–Consumer Pipeline 为什么能加速？什么时候可以释放缓冲区？](../docs/05-kernel-engineering/pipeline/mma-memory-pipeline.md)
14. 为什么同一个 Kernel 对不同 Shape 最优配置不同？

## GPU 硬件与性能分析

[对应内容与题目](../docs/06-gpu-performance/README.md#section-31)

1. GPU 为什么适合大规模并行？
2. SM、Warp、CTA 的关系是什么？
3. GPU 如何隐藏内存延迟？
4. Active Warp 和 Eligible Warp 有什么区别？
5. Occupancy 越高越好吗？
6. Register Pressure 为什么影响 Occupancy？
7. Shared Memory 和 L1 Cache 有什么区别？
8. Bank Conflict 是怎么产生的？
9. Coalescing 的硬件本质是什么？
10. L2 Cache Hit Rate 怎么提高？
11. Tensor Core 为什么不是所有 GEMM 都能跑满？
12. Memory-bound 和 Compute-bound 怎么判断？
13. MFU 和 MBU 分别是什么？
14. Nsight Systems 和 Nsight Compute 有什么区别？
15. 怎么判断一个 Kernel 优化到顶了？
16. 为什么 Kernel 加速 20% 不等于模型加速 20%？
