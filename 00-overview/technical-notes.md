# 技术口径与待核对事项

本次整理目标是保留原稿覆盖范围，建立可继续学习的框架。六层内容仍是草案；这里记录关键修正与验证入口，未完成逐段技术审校或代码实测。

## 已处理的关键表述

| 主题 | 当前口径 |
| --- | --- |
| MLA 缓存 | latent 简图需要结合解耦 RoPE key 分量理解；投影吸收留待完整推导 |
| MoE | 区分早期 dense layers、shared experts 与 routed experts；Router 的评分函数依模型而异 |
| 自回归生成 | 普通 Decode 通常每轮一个 Token，投机解码需要另行分析 |
| Serving 生命周期 | 补充流式输出、取消、长度上限和缓存引用释放；原流程图是简化路径 |
| TP / PP / EP / PD | 前三者描述切分方向；PD 描述执行阶段解耦，DP / Replica 是服务副本扩展方向 |
| Attention 与算子 | 数学上可分解为多项计算，框架也可以暴露融合算子 |
| Continuous Batching | 在后续迭代按资源与策略补入请求；固定批次槽位闲置不等于整张 GPU 都停止工作 |
| Warp 同步 | 现代独立线程调度下不依赖隐式 lockstep；mask、barrier 与参与线程必须满足同步约束 |
| 合并与向量访存 | Coalescing 关注同一 Warp 内线程的地址，Vector Load 关注单线程访存宽度 |
| `int4` 与 INT4 | CUDA `int4` 是四个 int 的向量类型；INT4 量化是每个值四位的表示 |
| 内存层次图 | Register / Shared 不是必须逐级经过的缓存层；图箭头表示概念组织 |
| 性能模型 | 计算、访存、通信和调度存在重叠，不能直接相加为总耗时 |
| MFU 与 Kernel 指标 | MFU 使用模型 FLOPs/s；Kernel 可记录 achieved TFLOP/s 与计算利用率，峰值必须匹配精度和计数口径 |
| 瓶颈判断 | Compute-bound / Memory-bound 依 Shape、dtype、Batch、缓存、硬件和实现而变化 |

现代 Warp 的同步限制参考 [NVIDIA Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)。访存和 Bank 分析的后续验证入口是 [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)。

## FlashMLA 的版本范围

原稿主要讨论 V3 时代的 MLA Decode。不能把这份计划当作最新 FlashMLA 的完整功能说明；官方仓库当前还包含其他 Attention 路径，并有版本兼容性调整。后续应为 V3 学习案例固定历史 commit，再单独记录新版变化。参见 [FlashMLA 官方仓库](https://github.com/deepseek-ai/FlashMLA)。

DeepSeek 模型细节以 [DeepSeek-V3 官方仓库与报告入口](https://github.com/deepseek-ai/DeepSeek-V3)为基础，后续补齐符号、投影、缓存和 Router 实现。

## 学习时继续核对

- MLA / MoE 的精确公式、Cache Shape、Dense / Shared / Routed 路径。
- Online Softmax 分块合并与输出累积的完整推导，避免只记录 max / denominator。
- vLLM / SGLang 的实际调度和缓存实现，记录版本，避免将概念图当作通用源码路径。
- Triton、CUTLASS、CuTe 与 GPU 架构相关能力，结合实际版本确认。
- 所有示意代码的线程参与、边界检查、对齐、同步与数值误差。
- 各实验的实际字节 / FLOPs、Profiler 指标和端到端效果。
- TurboQuant 或自研量化 KV 的具体方案与实现状态。

## 新增内容的边界

此次补充了项目定位、学习闭环、跨层主线、目录职责、源码版本记录、实验复现方式、面试题维护方式和候选项目入口。新增部分是组织与学习建议，不代表这些实验、项目或答案已经完成。
