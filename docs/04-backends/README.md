# 高性能 Backend

> 状态：内容草案。已整理原稿的技术主题、公式、流程、图布局和题目；完整教程、源码版本与实测结果将随学习补齐。示意代码不等同于已验证实现。

**核心问题：同一个算子怎样选择或设计高性能实现？**

覆盖 FlashAttention、FlashMLA、DeepGEMM、DeepEP、量化 KV 自研 Backend、Tiling、Pipeline、Fusion、资源取舍与优化上限。

建议学习顺序：先区分算子与实现，再从 Shape、Layout 和瓶颈进入具体 Backend；源码与实验记录在本层关联。

[返回总导航](../../导航.md) · [学习路线](../../roadmap/README.md) · [面试题索引](../../interview/README.md)

## 内容索引

- [一个算子是怎样变成高性能 Kernel 的](#section-01)
- [FlashAttention](#section-02)
- [FlashAttention 的 Kernel 视角](#section-03)
- [FlashMLA：为什么不能只写成“FlashAttention for MLA”](#section-04)
- [ TurboQuant](#section-05)
- [DeepGEMM](#section-06)
- [GEMM 的 Tile 层次](#section-07)
- [Producer–Consumer Pipeline](#section-08)
- [为什么 Tile 不是越大越好](#section-09)
- [Grouped GEMM / DeepGEMM for MoE](#section-10)
- [DeepEP：“通信 Backend”](#section-11)
- [计算 Backend 和通信 Backend 的区别](#section-12)
- [自研 CUDA / Triton Kernel 为什么还存在](#section-13)
- [Vectorized Memory Access](#section-14)
- [Shared Memory 在 Backend 层的角色](#section-15)
- [Register 的角色](#section-16)
- [Kernel Fusion](#section-17)
- [Occupancy / ILP / Parallelism](#section-18)
- [Backend 层到底看什么指标](#section-19)
- [怎么判断 Kernel 已经接近上限](#section-20)
- [CUDA vs Triton 在这一层如何出现](#section-21)
- [Backend 映射总表](#section-22)
- [面试与自测问题](#section-23)
- [知识图布局草案](#section-24)

## 适用条件与补充

FlashMLA 不能只定义为 Decode 库；官方仓库随版本扩展或调整支持范围。本计划保留 V3 时代的学习主线，后续源码阅读必须固定 commit。DeepEP 是通信 Backend；跨设备传输、计算和调度可能重叠。Grouped GEMM 的任务映射只是示例，不是所有实现的统一调度方式。TurboQuant / 量化 KV 作为候选实践方向保留，不代表本仓库已有可运行实现或性能结果。

详见[技术口径与待核对事项](../00-overview/technical-notes.md)。

### 从“算什么”进入“怎么高效算”
本层知识图顶部直接放一句核心定义：

> **算子层定义数学逻辑，Backend / Kernel 层决定这些算子如何映射到 GPU 并尽可能接近硬件上限。**

```text
Operator / Primitive
Attention / GEMM / MoE / RMSNorm
            ↓
“算什么”
────────────────────────
“怎么高效算”
            ↓
Kernel / Backend
FlashAttention / FlashMLA / DeepGEMM
DeepEP / CUDA / Triton
```

---

<a id="section-01"></a>

## 一个算子是怎样变成高性能 Kernel 的

左侧依然保留“主流程”，否则这一张容易变成技术名词合集。

```text
Operator Definition
Attention / GEMM / MoE
        ↓
Tensor Shape
M / N / K
Q / KV Length
Head Dim
        ↓
Performance Model
Compute-bound?
Memory-bound?
Communication-bound?
        ↓
Data Layout
        ↓
Tile / Partition
        ↓
Thread / Warp / CTA Mapping
        ↓
Memory Movement
HBM → L2 → Shared → Register
        ↓
Computation
CUDA Core / Tensor Core
        ↓
Reduction / Synchronization
        ↓
Kernel Output
```

> **Kernel 优化的本质：让数据搬运、计算和并行度更匹配硬件。**

---

<a id="section-02"></a>

## FlashAttention

这一块不能只写“减少 HBM IO”。

应该画出它真正做了什么。

### Naive Attention
```text
Q × Kᵀ
   ↓
Attention Score
   ↓
写入 HBM
   ↓
读回 Score
   ↓
Softmax
   ↓
写入 HBM
   ↓
读回 P
   ↓
P × V
```

问题：

```text
巨大中间矩阵
S ∈ R^(N×N)

反复 HBM Round-trip
```

---

### FlashAttention
```text
Q Tile
   │
K Tile ─────┐
            ↓
       QK Tile
            ↓
     Online Softmax
            ↓
V Tile ─→ Partial O
            ↓
      Running m / l / O
            ↓
        Final Output
```

+ 分 Tile 加载 Q/K/V
+ Score 不完整写回 HBM
+ 使用 Online Softmax
+ Shared Memory / Register 内完成中间计算
+ 提高数据复用
+ 降低 HBM traffic

一句话总结：

> **FlashAttention 的核心不是“Softmax 更快”，而是 IO-aware：重排计算顺序以减少 HBM 访问。**

---

<a id="section-03"></a>

## FlashAttention 的 Kernel 视角

再往下钻一点。

```text
CTA
负责一个 Q Tile

Warp 0
Warp 1
Warp 2
Warp 3
   ↓
共同处理 K/V Tiles
```

循环：

```text
for each K/V tile:

    Load K/V
        ↓
    QK
        ↓
    update online softmax
        ↓
    PV accumulation
```

```text
Tile
Shared Memory
Register
Online Softmax
Warp Reduction
Pipeline
Tensor Core
```

---

<a id="section-04"></a>

## FlashMLA：为什么不能只写成“FlashAttention for MLA”

这一块应该突出：

> **本草案以 DeepSeek-V3 时代的 MLA Decode Kernel 为学习案例；FlashMLA 的支持范围需要结合具体版本确认。**

```text
DeepSeek MLA
       ↓
Latent KV Cache
       ↓
Decode Query
       ↓
读取历史压缩 KV
       ↓
Attention Calculation
       ↓
FlashMLA
```

它需要重点解决：

```text
① MLA 特殊 KV 表示

② Decode 阶段小 Q + 长历史 KV

③ KV Cache Memory Traffic

④ Split / Partition 历史 Context

⑤ Partial Softmax State Reduction
```

可以画一个和量化 KV 实验 很像的结构：

```text
Long KV Context
──────────────────────────

Split 0
Split 1
Split 2
...
Split N

 ↓      ↓      ↓

Partial Attention
(m, l, O)

       ↓

Merge / Reduction

       ↓

Final O
```

```text
m : max
l : softmax denominator state
O : partial output
```

这和 Flash-Decoding / Split-KV 的思想高度相关。

---

<a id="section-05"></a>

##  TurboQuant

不需要做大，只需要在 FlashMLA 旁边画一个：

```text
普通 MLA Decode
      ↓
读取 KV Cache
      ↓
FlashMLA
```

另一条：

```text
Quantized KV Cache
      ↓
INT4 unpack
      ↓
dequant
      ↓
QK
      ↓
Online Softmax
      ↓
PV
```

然后标：

> **自研 Backend 的价值：针对特殊数据格式或固定 Shape，把额外操作融合进 Attention Kernel。**

这就是候选项目为什么属于 Backend 层，而不仅仅是“量化算法”。

---

<a id="section-06"></a>

## DeepGEMM

算子层描述：

```text
GEMM = 算子
```

Backend 层描述：

```text
DeepGEMM = 高性能 GEMM 实现
```

```text
A Tile         B Tile
   │             │
   └──────┬──────┘
          ↓
      Shared Memory
          ↓
       Warp / WG
          ↓
     Tensor Core MMA
          ↓
     Register Accum
          ↓
        Epilogue
          ↓
          C
```

这里的关键词应该包括：

+ Tiling
+ Tensor Core
+ Layout
+ Register Accumulation
+ Pipeline
+ Double Buffering
+ TMA / async copy
+ Epilogue

---

<a id="section-07"></a>

## GEMM 的 Tile 层次

```text
Whole GEMM

M × N × K
    ↓

CTA Tile
BM × BN × BK
    ↓

Warp Tile
WM × WN × WK
    ↓

MMA Tile
Tensor Core Instruction
```

也就是：

```text
Problem
 ↓
CTA
 ↓
Warp / Warpgroup
 ↓
MMA
```

> **高性能 GEMM 的核心，就是把大矩阵分解成适合硬件执行层级的 Tile。**

---

<a id="section-08"></a>

## Producer–Consumer Pipeline

[指令与同步层解释：从 Ampere 到 Blackwell 的多阶段流水线](../05-kernel-engineering/pipeline/mma-memory-pipeline.md)。

这一块可以开始为 Hopper / TMA 铺垫。

```text
Producer Warpgroup

HBM
 ↓
TMA
 ↓
Shared Memory
```

另一边：

```text
Consumer Warpgroup

Shared Memory
 ↓
MMA
 ↓
Register
```

然后形成流水：

```text
Stage 0 Load
Stage 1 Compute

Stage 1 Load
Stage 0 Compute
```

目标：

> **隐藏 Memory Latency，让“搬数据”和“算数据”重叠。**

---

<a id="section-09"></a>

## 为什么 Tile 不是越大越好

右边做一个小 trade-off 框：

```text
Tile 变大
   ↓
数据复用 ↑
Tensor Core 利用率 ↑

但是：

Shared Memory ↑
Register ↑
CTA Residency ↓
Occupancy ↓
```

> **Tile Size 是 reuse、资源占用和并行度之间的平衡。**

这是非常典型的面试追问。

---

<a id="section-10"></a>

## Grouped GEMM / DeepGEMM for MoE

这里接第三张 MoE。

普通 GEMM：

```text
A × B = C
```

Grouped GEMM：

```text
Expert 0: A0 × B0
Expert 1: A1 × B1
Expert 2: A2 × B2
...
```

统一 launch：

```text
Grid
│
├─ CTA 0 → Expert 0 / Tile 0
├─ CTA 1 → Expert 2 / Tile 3
├─ CTA 2 → Expert 1 / Tile 1
└─ ...
```

依赖 metadata：

```text
expert_id
tile_id
M_e
offset
```

> **难点不是 GEMM 数学，而是每个 Expert 的 M 动态变化，导致任务粒度和负载不均衡。**

---

<a id="section-11"></a>

## DeepEP：“通信 Backend”

因为 Backend 不一定只是 CUDA 计算 Kernel。

DeepEP 对应：

```text
MoE
 ↓
Expert Parallel
 ↓
Token 必须去 Expert 所在 GPU
```

完整流程：

```text
GPU 0 Tokens
GPU 1 Tokens
GPU 2 Tokens
GPU 3 Tokens
       ↓
     Router
       ↓
根据 Expert 位置
       ↓
   Dispatch
       ↓
All-to-All Communication
       ↓
Remote Expert GEMM
       ↓
Combine
       ↓
Tokens 回原位置
```

```text
DeepEP
```

优化的不是 Attention 或 GEMM 本身，而是：

```text
Token Dispatch / Combine
+
跨 GPU 通信
```

---

<a id="section-12"></a>

## 计算 Backend 和通信 Backend 的区别

建议做一个很清楚的小对比：

| Backend | 主要瓶颈 |
| --- | --- |
| FlashMLA | HBM / Attention |
| DeepGEMM | Tensor Core / GEMM |
| DeepEP | NVLink / RDMA / Communication |
| 自研 TQ Kernel | KV IO + unpack/dequant + Attention |

于是就能自然得到：

```text
推理性能需要共同分析：
Compute / Memory / Communication / Scheduling
它们可以重叠，不能直接相加为总耗时。
```

---

<a id="section-13"></a>

## 自研 CUDA / Triton Kernel 为什么还存在

> 哪些场景需要自研 Kernel？

典型场景：

### 1. 新模型结构
```text
新 Attention
新 MoE
新 Quantization
```

已有库没实现。

---

### 2. 特殊 Shape

```text
M 很小
K 固定
Head Dim = 128
Context 固定
Batch 特定
```

通用 Kernel 不一定最优。

---

### 3. Fusion
```text
unpack
+
dequant
+
attention
+
softmax
```

通用库无法自然表达。

---

### 4. 特殊 Layout / 数据类型
```text
INT4
FP4
custom packed format
sparse layout
```

需要定制 Kernel。

这正好解释：

> **高性能 Backend 永远不会只有一个通用实现。**

---

<a id="section-14"></a>

## Vectorized Memory Access

面试问题也应该开始放到这一张。

```text
普通 Load

thread 0 → 4B
thread 1 → 4B
...
```

向量化：

```text
float4
uint4
int4
128-bit Load
```

目标：

+ 减少 load instruction 数
+ 更充分利用 memory transaction
+ 增加 instruction-level efficiency

但是必须标：

> **前提：地址对齐 + 数据布局允许。**

---

<a id="section-15"></a>

## Shared Memory 在 Backend 层的角色

不要再解释 bank conflict 细节，那个留给 GPU 层。

这里讲作用：

```text
HBM
 ↓
Shared Memory
 ↓
Register
```

Shared Memory 主要用于：

```text
数据复用
Tile staging
Layout transformation
跨线程共享
Pipeline buffer
```

也就是说：

> Shared Memory 不是“因为比 HBM 快所以全部搬进去”，而是因为 **数据需要被多次复用或重新组织**。

---

<a id="section-16"></a>

## Register 的角色

这里也可以写：

```text
Register
```

用于：

+ accumulator
+ temporary values
+ fragment
+ online softmax state
+ unpack / dequant intermediate

但是：

```text
Register 数量过高
        ↓
Occupancy 下降
```

```text
更多 Register
≠
一定更快
```

---

<a id="section-17"></a>

## Kernel Fusion

这一张需要比第三张更工程化。

第三张讲为什么 Fusion。

第四张讲怎么 Fusion。

```text
Load packed KV
      ↓
unpack INT4
      ↓
dequant
      ↓
QK
      ↓
online softmax
      ↓
PV
      ↓
Output
```

如果拆开：

```text
Kernel 1
unpack
 ↓ HBM

Kernel 2
dequant
 ↓ HBM

Kernel 3
attention
```

融合后：

```text
packed data
   ↓
register
   ↓
unpack
   ↓
dequant
   ↓
compute
```

避免中间 HBM。

---

<a id="section-18"></a>

## Occupancy / ILP / Parallelism

右下角建议做一个“三角关系”：

```text
        Occupancy
          /\
         /  \
        /    \
      ILP —— TLP
```

理解：

```text
Occupancy
同一 SM 能驻留多少 warp

TLP
Thread-Level Parallelism

ILP
单线程 / warp 内有多少独立指令
```

有时候：

> **低 Occupancy + 高 ILP 也可能很快。**

> Occupancy 不是最终目标，Latency Hiding 才是目标。

---

<a id="section-19"></a>

## Backend 层到底看什么指标

Backend 层首先分析 Kernel 指标，再关联 TTFT 等端到端指标。

应该看：

```text
Kernel Latency
```

```text
Effective Bandwidth
```

```text
Tensor Core Utilization
```

```text
SM Utilization
```

```text
L2 Hit Rate
```

```text
Registers / Thread
```

```text
Shared Memory / CTA
```

```text
Occupancy
```

```text
Stall Reasons
```

以及最终：

```text
End-to-End Impact
```

因为：

> Kernel 快 20% 不代表模型快 20%。

---

<a id="section-20"></a>

## 怎么判断 Kernel 已经接近上限

这是非常值得加的内容。

做一个：

**Optimization Ceiling**
依次检查：

```text
① Roofline

是否接近
Compute / Memory 理论上限？
```

↓

```text
② Hardware Utilization

Tensor Core / HBM 是否已经很高？
```

↓

```text
③ Stall

主要 Stall 还有没有可消除的？
```

↓

```text
④ Resource

Register / Shared 是否限制并行？
```

↓

```text
⑤ Launch

Kernel 自身够快以后，
launch overhead 是否开始占主导？
```

↓

```text
⑥ End-to-End

继续优化这个 Kernel
还能影响多少整体性能？
```

这实际上就是工程上非常重要的 stop condition。

---

<a id="section-21"></a>

## CUDA vs Triton 在这一层如何出现

具体编程模型关联 Kernel 编程层。

只需要：

```text
同一个 Operator

        ↓

CUDA Kernel
or
Triton Kernel
or
CUTLASS/CuTe Backend
```

```text
CUDA：
控制最细

Triton：
开发效率高

CUTLASS/CuTe：
Tensor Core / GEMM 模板与抽象
```

> **详细编程模型 → 下一层**

---

<a id="section-22"></a>

## Backend 映射总表

```text
Model / Operator          High-performance Backend

MLA Attention       →     FlashMLA
MHA/GQA Attention   →     FlashAttention
Dense GEMM          →     cuBLAS / DeepGEMM / CUTLASS
MoE Grouped GEMM    →     DeepGEMM
MoE Communication   →     DeepEP
Quantized KV        →     Custom CUDA / Triton Kernel
```

一眼就清楚。

---

<a id="section-23"></a>

## 面试与自测问题

放 12 个：

1. FlashAttention 为什么快？
2. Online Softmax 在 FlashAttention 里的作用是什么？
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

---

<a id="section-24"></a>

## 知识图布局草案

依旧沿用统一模板：

```text
┌──────────────────────────────────────────────────────────────┐
│ 高性能 Kernel / Backend：从算子到 GPU 高效实现              │
├──────────────┬──────────────────────┬────────────────────────┤
│              │                      │ FlashAttention         │
│ Operator     │ Backend Mapping      │ Tile + Online Softmax  │
│ → Kernel     │                      ├────────────────────────┤
│              │ Attention→FA/FMLA    │ FlashMLA               │
│ Shape        │ GEMM→DeepGEMM        │ Decode / Split KV      │
│ ↓            │ MoE Comm→DeepEP      ├────────────────────────┤
│ Tile         │                      │ DeepGEMM               │
│ ↓            ├──────────────────────┤ Tensor Core / Pipeline │
│ Warp/CTA     │ MoE / Grouped GEMM   │                        │
│ ↓            │                      │                        │
│ GPU          │ Expert Mapping       │                        │
├──────────────┴──────────────────────┼────────────────────────┤
│ Kernel 优化手段                    │ Memory / Resource      │
│ Tile / Vector Load / Fusion        │ Shared / Register      │
│ Pipeline / Online Softmax          │ Occupancy / ILP        │
├────────────────────────────────────┴────────────────────────┤
│ 优化到顶怎么判断：Roofline → Utilization → Stall → E2E      │
├─────────────────────────────────────────────────────────────┤
│ 高频面试问题                                                │
└─────────────────────────────────────────────────────────────┘
```

```text
① 模型层
MLA / MoE 是什么
        ↓

② 推理框架层
什么时候、谁来执行
        ↓

③ 算子层
真正需要计算什么
        ↓

④ Backend 层
这些计算怎样做到高性能
```
