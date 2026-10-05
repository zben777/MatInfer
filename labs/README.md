# 算子与推理实验

此目录包含逐步开展的教学实现、正确性检查与后续性能实验。原 `kernels/` 中的实践方向全部保留，待实际学习时创建子目录；正确性结果与性能结果分别记录。

当前入口：[Attention / GQA 教学实现与正确性检查](attention/README.md)。

| 方向 | 逐步探索的问题 |
| --- | --- |
| Reduction | 串行 / Warp / CTA 归约、mask、同步与跨块合并 |
| Softmax | 数值稳定、Online Softmax、分块处理与带宽 |
| RMSNorm | Reduction + Elementwise、Vector Load、融合与精度 |
| GEMM | M/N/K、Tiling、Tensor Core、Layout、Pipeline 与基线 |
| Attention | QK / Softmax / PV、KV Layout、Split-KV、输出合并 |
| 量化 KV | 位打包、解码 / 反量化融合、量化误差与端到端收益 |
| Serving | Batch、上下文长度、前缀复用与调度对 TTFT / TPOT 的影响 |
| MoE | Expert token 分布、Grouped GEMM、通信与计算重叠 |

实验先证明正确，再讨论性能；记录 Shape、dtype、GPU、版本、基线、Warmup、同步方式、重复测量和误差。测量结果属于其环境与负载，不能直接推广到所有模型。

[实验记录模板](../templates/experiment.md) · [Kernel 编程](../docs/05-kernel-engineering/README.md) · [GPU 性能分析](../docs/06-gpu-performance/README.md)
