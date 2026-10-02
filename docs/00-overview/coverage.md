# 原稿内容覆盖对照

整理依据：归档中的 `结构.md` 与 `导航.md`。原始字节保留，正文按六层承接。本表逐项记录原稿的 **136 个实质章节**，其中包括主流程、公式模块、机制、指标、题目与布局建议。

## 处理方式

- 保留各层技术主题、数学表达式、流程 / Shape / 代码示意和布局草案。
- 删除重复转场、对话提示与评价性文字，收紧标题和 Markdown 层级。
- 外部图片链接放到总览，OCR 注释留在原始材料。
- 技术简化处补充适用条件；明确修正见[技术口径](technical-notes.md)。
- 原目录全部有承接，详见[仓库结构](../../结构.md)。
- 没有将规划中的候选项目、题目或实验写成已完成成果。

## 原章节到现有文档

| 原稿行号 | 原章节 | 整理位置 |
| --- | --- | --- |
| 47 | 左侧：完整 Token 主流程 | [查看](../../docs/01-model/README.md#section-01) |
| 98 | 中间最大的区域：一个 DeepSeek Transformer Block | [查看](../../docs/01-model/README.md#section-02) |
| 151 | 右上：RMSNorm | [查看](../../docs/01-model/README.md#section-03) |
| 178 | 右中第一块：MHA → GQA → MLA | [查看](../../docs/01-model/README.md#section-04) |
| 228 | 右中第二块：MLA 内部流程 | [查看](../../docs/01-model/README.md#section-05) |
| 281 | 右侧：Attention 数学 | [查看](../../docs/01-model/README.md#section-06) |
| 325 | 中下：DeepSeekMoE | [查看](../../docs/01-model/README.md#section-07) |
| 388 | 左/中下：RoPE | [查看](../../docs/01-model/README.md#section-08) |
| 426 | 底部第一块：KV Cache | [查看](../../docs/01-model/README.md#section-09) |
| 479 | 底部第二块：Sampling | [查看](../../docs/01-model/README.md#section-10) |
| 499 | 底部第三块：这一层的面试索引 | [查看](../../docs/01-model/README.md#section-11) |
| 519 | 这张图最关键的视觉结构 | [查看](../../docs/01-model/README.md#section-12) |
| 593 | 一、整张图的核心问题 | [查看](../../docs/02-serving/README.md#section-01) |
| 603 | 二、左侧：一个 Request 的完整生命周期 | [查看](../../docs/02-serving/README.md#section-02) |
| 664 | 三、中间最大的区域：Scheduler | [查看](../../docs/02-serving/README.md#section-03) |
| 715 | 四、Continuous Batching | [查看](../../docs/02-serving/README.md#section-04) |
| 771 | 五、Prefill 和 Decode | [查看](../../docs/02-serving/README.md#section-05) |
| 829 | 六、KV Cache Manager | [查看](../../docs/02-serving/README.md#section-06) |
| 889 | 七、Prefix Cache / Radix Cache | [查看](../../docs/02-serving/README.md#section-07) |
| 948 | 八、ModelRunner / Executor | [查看](../../docs/02-serving/README.md#section-08) |
| 996 | 九、CUDA Graph | [查看](../../docs/02-serving/README.md#section-09) |
| 1075 | 十、Chunked Prefill | [查看](../../docs/02-serving/README.md#section-10) |
| 1135 | 十一、Speculative Decoding | [查看](../../docs/02-serving/README.md#section-11) |
| 1174 | 十二、TP / PP / EP / PD | [查看](../../docs/02-serving/README.md#section-12) |
| 1226 | 十三、PD 分离 | [查看](../../docs/02-serving/README.md#section-13) |
| 1280 | 十四、框架对比：vLLM / SGLang / TensorRT-LLM | [查看](../../docs/02-serving/README.md#section-14) |
| 1316 | 十五、这张图必须有一个“系统层指标”区域 | [查看](../../docs/02-serving/README.md#section-15) |
| 1357 | 十六、面试高频问题区域 | [查看](../../docs/02-serving/README.md#section-16) |
| 1373 | 十七、整张图最终布局 | [查看](../../docs/02-serving/README.md#section-17) |
| 1408 | 十八、这张图和第一张模型图是怎么连接的 | [查看](../../docs/02-serving/README.md#section-18) |
| 1510 | 一、左侧：一个 Transformer Block 最终被拆成哪些算子 | [查看](../../docs/03-operators/README.md#section-01) |
| 1592 | 二、中间：算子层不要平铺，要分成 4 类 | [查看](../../docs/03-operators/README.md#section-02) |
| 1606 | 三、第一类：GEMM —— 大模型真正的计算核心 | [查看](../../docs/03-operators/README.md#section-03) |
| 1660 | 四、GEMM 性能到底受什么影响 | [查看](../../docs/03-operators/README.md#section-04) |
| 1694 | 五、第二类：Attention | [查看](../../docs/03-operators/README.md#section-05) |
| 1737 | 六、Attention 最重要的一条演进线 | [查看](../../docs/03-operators/README.md#section-06) |
| 1758 | 七、Softmax → Online Softmax | [查看](../../docs/03-operators/README.md#section-07) |
| 1800 | 八、把 FlashAttention 为什么快画出来 | [查看](../../docs/03-operators/README.md#section-08) |
| 1845 | 九、Decode Attention 要单独一个框 | [查看](../../docs/03-operators/README.md#section-09) |
| 1886 | 十、第三类：RMSNorm / Elementwise | [查看](../../docs/03-operators/README.md#section-10) |
| 1934 | 十一、RoPE | [查看](../../docs/03-operators/README.md#section-11) |
| 1967 | 十二、第四类：MoE 算子 | [查看](../../docs/03-operators/README.md#section-12) |
| 2008 | 十三、Grouped GEMM 为什么出现 | [查看](../../docs/03-operators/README.md#section-13) |
| 2056 | 十四、Fusion 是这一层非常重要的一块 | [查看](../../docs/03-operators/README.md#section-14) |
| 2120 | 十五、算子要分 Compute-bound / Memory-bound | [查看](../../docs/03-operators/README.md#section-15) |
| 2152 | 十六、算子层和 Backend 层的边界 | [查看](../../docs/03-operators/README.md#section-16) |
| 2201 | 十七、这一层的典型性能指标 | [查看](../../docs/03-operators/README.md#section-17) |
| 2244 | 十八、面试高频区 | [查看](../../docs/03-operators/README.md#section-18) |
| 2260 | 十九、这一张的最终视觉布局 | [查看](../../docs/03-operators/README.md#section-19) |
| 2357 | 一、左侧主流程：一个算子是怎样变成高性能 Kernel 的 | [查看](../../docs/04-backends/README.md#section-01) |
| 2400 | 二、中间第一大块：FlashAttention | [查看](../../docs/04-backends/README.md#section-02) |
| 2468 | 三、FlashAttention 的 Kernel 视角 | [查看](../../docs/04-backends/README.md#section-03) |
| 2513 | 四、FlashMLA：为什么不能只写成“FlashAttention for MLA” | [查看](../../docs/04-backends/README.md#section-04) |
| 2587 | 五、这张图里可以第一次正式放你的 TurboQuant | [查看](../../docs/04-backends/README.md#section-05) |
| 2623 | 六、中间第二大块：DeepGEMM | [查看](../../docs/04-backends/README.md#section-06) |
| 2669 | 七、GEMM 的 Tile 层次 | [查看](../../docs/04-backends/README.md#section-07) |
| 2709 | 八、Producer–Consumer Pipeline | [查看](../../docs/04-backends/README.md#section-08) |
| 2755 | 九、为什么 Tile 不是越大越好 | [查看](../../docs/04-backends/README.md#section-09) |
| 2781 | 十、Grouped GEMM / DeepGEMM for MoE | [查看](../../docs/04-backends/README.md#section-10) |
| 2826 | 十一、DeepEP：这一张第一次真正加入“通信 Backend” | [查看](../../docs/04-backends/README.md#section-11) |
| 2880 | 十二、计算 Backend 和通信 Backend 的区别 | [查看](../../docs/04-backends/README.md#section-12) |
| 2907 | 十三、自研 CUDA / Triton Kernel 为什么还存在 | [查看](../../docs/04-backends/README.md#section-13) |
| 2975 | 十四、Vectorized Memory Access | [查看](../../docs/04-backends/README.md#section-14) |
| 3010 | 十五、Shared Memory 在 Backend 层的角色 | [查看](../../docs/04-backends/README.md#section-15) |
| 3042 | 十六、Register 的角色 | [查看](../../docs/04-backends/README.md#section-16) |
| 3075 | 十七、Kernel Fusion | [查看](../../docs/04-backends/README.md#section-17) |
| 3135 | 十八、Occupancy / ILP / Parallelism | [查看](../../docs/04-backends/README.md#section-18) |
| 3171 | 十九、Backend 层到底看什么指标 | [查看](../../docs/04-backends/README.md#section-19) |
| 3227 | 二十、怎么判断 Kernel 已经接近上限 | [查看](../../docs/04-backends/README.md#section-20) |
| 3288 | 二十一、CUDA vs Triton 在这一层如何出现 | [查看](../../docs/04-backends/README.md#section-21) |
| 3325 | 二十二、Backend 映射总表 | [查看](../../docs/04-backends/README.md#section-22) |
| 3343 | 二十三、面试高频问题 | [查看](../../docs/04-backends/README.md#section-23) |
| 3361 | 二十四、这张图最终版式 | [查看](../../docs/04-backends/README.md#section-24) |
| 3462 | 一、左侧主流程：写一个 Kernel 到底经历什么 | [查看](../../docs/05-kernel-engineering/README.md#section-01) |
| 3511 | 二、整张图中间最核心：CUDA 执行层级 | [查看](../../docs/05-kernel-engineering/README.md#section-02) |
| 3570 | 三、CUDA C++：最底层控制 | [查看](../../docs/05-kernel-engineering/README.md#section-03) |
| 3626 | 四、Thread / Warp / Block 怎么选择 | [查看](../../docs/05-kernel-engineering/README.md#section-04) |
| 3678 | 五、Warp Primitive | [查看](../../docs/05-kernel-engineering/README.md#section-05) |
| 3722 | 六、`__syncthreads()` 到底同步什么 | [查看](../../docs/05-kernel-engineering/README.md#section-06) |
| 3763 | 七、Global Memory Access：向量化和合并访存 | [查看](../../docs/05-kernel-engineering/README.md#section-07) |
| 3822 | 八、Alignment | [查看](../../docs/05-kernel-engineering/README.md#section-08) |
| 3859 | 九、Shared Memory：不是“缓存”，而是 Scratchpad | [查看](../../docs/05-kernel-engineering/README.md#section-09) |
| 3896 | 十、Bank Conflict | [查看](../../docs/05-kernel-engineering/README.md#section-10) |
| 3948 | 十一、Register：性能神器也是资源瓶颈 | [查看](../../docs/05-kernel-engineering/README.md#section-11) |
| 3995 | 十二、Triton：换了一种编程模型 | [查看](../../docs/05-kernel-engineering/README.md#section-12) |
| 4045 | 十三、CUDA vs Triton 的关键差别 | [查看](../../docs/05-kernel-engineering/README.md#section-13) |
| 4072 | 十四、Triton Layout 为什么重要 | [查看](../../docs/05-kernel-engineering/README.md#section-14) |
| 4106 | 十五、CUTLASS：高性能 GEMM 工程模板 | [查看](../../docs/05-kernel-engineering/README.md#section-15) |
| 4157 | 十六、CuTe：真正需要理解的是 Layout Algebra | [查看](../../docs/05-kernel-engineering/README.md#section-16) |
| 4198 | 十七、CUTLASS / CuTe 的关系 | [查看](../../docs/05-kernel-engineering/README.md#section-17) |
| 4234 | 十八、Tensor Core：WMMA / MMA / CuTe 的区别 | [查看](../../docs/05-kernel-engineering/README.md#section-18) |
| 4294 | 十九、TMA | [查看](../../docs/05-kernel-engineering/README.md#section-19) |
| 4351 | 二十、TMA vs `cp.async` | [查看](../../docs/05-kernel-engineering/README.md#section-20) |
| 4386 | 二十一、Producer–Consumer | [查看](../../docs/05-kernel-engineering/README.md#section-21) |
| 4425 | 二十二、Barrier 为什么越来越复杂 | [查看](../../docs/05-kernel-engineering/README.md#section-22) |
| 4458 | 二十三、Double Buffer / Multi-Stage Pipeline | [查看](../../docs/05-kernel-engineering/README.md#section-23) |
| 4501 | 二十四、Launch Configuration | [查看](../../docs/05-kernel-engineering/README.md#section-24) |
| 4550 | 二十五、Auto-Tuning | [查看](../../docs/05-kernel-engineering/README.md#section-25) |
| 4586 | 二十六、Compile-time Specialization | [查看](../../docs/05-kernel-engineering/README.md#section-26) |
| 4624 | 二十七、Kernel Debugging | [查看](../../docs/05-kernel-engineering/README.md#section-27) |
| 4666 | 二十八、Benchmark 方法 | [查看](../../docs/05-kernel-engineering/README.md#section-28) |
| 4701 | 二十九、这一层的核心选择：什么时候用谁？ | [查看](../../docs/05-kernel-engineering/README.md#section-29) |
| 4749 | 三十、面试高频问题 | [查看](../../docs/05-kernel-engineering/README.md#section-30) |
| 4769 | 三十一、这一张的最终版式 | [查看](../../docs/05-kernel-engineering/README.md#section-31) |
| 4883 | 一、左侧主流程：一条 GPU 指令到底经历什么 | [查看](../../docs/06-gpu-performance/README.md#section-01) |
| 4917 | 二、整张图的视觉中心：一个 SM 里面有什么 | [查看](../../docs/06-gpu-performance/README.md#section-02) |
| 4953 | 三、Warp Scheduler：为什么 GPU 能隐藏延迟 | [查看](../../docs/06-gpu-performance/README.md#section-03) |
| 4995 | 四、Eligible Warp / Active Warp / Stalled Warp | [查看](../../docs/06-gpu-performance/README.md#section-04) |
| 5042 | 五、Occupancy | [查看](../../docs/06-gpu-performance/README.md#section-05) |
| 5086 | 六、Register File | [查看](../../docs/06-gpu-performance/README.md#section-06) |
| 5141 | 七、Shared Memory / L1 | [查看](../../docs/06-gpu-performance/README.md#section-07) |
| 5177 | 八、Shared Memory Bank | [查看](../../docs/06-gpu-performance/README.md#section-08) |
| 5232 | 九、Cache Hierarchy | [查看](../../docs/06-gpu-performance/README.md#section-09) |
| 5263 | 十、L2 Cache | [查看](../../docs/06-gpu-performance/README.md#section-10) |
| 5312 | 十一、Cache Line 和 Memory Transaction | [查看](../../docs/06-gpu-performance/README.md#section-11) |
| 5359 | 十二、Coalescing 的硬件本质 | [查看](../../docs/06-gpu-performance/README.md#section-12) |
| 5390 | 十三、HBM | [查看](../../docs/06-gpu-performance/README.md#section-13) |
| 5447 | 十四、Tensor Core | [查看](../../docs/06-gpu-performance/README.md#section-14) |
| 5486 | 十五、为什么 Small-M GEMM 难吃满 Tensor Core | [查看](../../docs/06-gpu-performance/README.md#section-15) |
| 5523 | 十六、Roofline：整张图最重要的性能模型 | [查看](../../docs/06-gpu-performance/README.md#section-16) |
| 5587 | 十七、把 LLM 算子放到 Roofline 上 | [查看](../../docs/06-gpu-performance/README.md#section-17) |
| 5613 | 十八、MFU / MBU | [查看](../../docs/06-gpu-performance/README.md#section-18) |
| 5661 | 十九、Latency vs Bandwidth | [查看](../../docs/06-gpu-performance/README.md#section-19) |
| 5699 | 二十、Memory-Level Parallelism | [查看](../../docs/06-gpu-performance/README.md#section-20) |
| 5742 | 二十一、ILP / TLP / MLP | [查看](../../docs/06-gpu-performance/README.md#section-21) |
| 5772 | 二十二、Warp Divergence | [查看](../../docs/06-gpu-performance/README.md#section-22) |
| 5803 | 二十三、Pipeline Stall | [查看](../../docs/06-gpu-performance/README.md#section-23) |
| 5830 | 二十四、Nsight Compute | [查看](../../docs/06-gpu-performance/README.md#section-24) |
| 5882 | 二十五、Nsight Systems vs Nsight Compute | [查看](../../docs/06-gpu-performance/README.md#section-25) |
| 5934 | 二十六、性能分析完整方法论 | [查看](../../docs/06-gpu-performance/README.md#section-26) |
| 5968 | 二十七、怎么判断优化有效 | [查看](../../docs/06-gpu-performance/README.md#section-27) |
| 6011 | 二十八、怎么判断“优化到顶” | [查看](../../docs/06-gpu-performance/README.md#section-28) |
| 6063 | 二十九、GPU 内部和多 GPU 的连接 | [查看](../../docs/06-gpu-performance/README.md#section-29) |
| 6106 | 三十、GPU 硬件层的“因果链” | [查看](../../docs/06-gpu-performance/README.md#section-30) |
| 6145 | 三十一、面试高频问题 | [查看](../../docs/06-gpu-performance/README.md#section-31) |
| 6167 | 三十二、最终版式 | [查看](../../docs/06-gpu-performance/README.md#section-32) |

## 原始文件校验值

以下 SHA-256 用于核对归档文件是否变化。

```text
333633adf362805a9c13d41183624d34e217a2fa27a90fcaa5b6fda977439053  结构.md
6505bc79fc1fcebc9ce0530049d55961034d0881597498b1fe4f1334912f8ddc  导航.md
```
