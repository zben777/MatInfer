# 核心算子与计算特征

> 状态：内容草案。已整理原稿的技术主题、公式、流程、图布局和题目；完整教程、源码版本与实测结果将随学习补齐。示意代码不等同于已验证实现。

**核心问题：模型计算由哪些算子组成，Shape 怎样决定性能？**

覆盖 GEMM、Attention、Softmax / Online Softmax、RMSNorm、RoPE、MoE 路由与数据搬运、Grouped GEMM、Fusion 和算子指标。

建议学习顺序：沿 Transformer Block 拆解计算，写出输入输出 Shape，再分析访存、计算量和实现选择。

[返回总导航](../../导航.md) · [学习路线](../../roadmap/README.md) · [面试题索引](../../interview/README.md)

## 内容索引

- [Attention 深入专题：Online Softmax 推导与分块参考实现](attention/README.md)

- [算子融合深入专题：Decode 场景、访存估算、工具链与面试案例](fusion/README.md)

- [一个 Transformer Block 最终被拆成哪些算子](#section-01)
- [算子分类](#section-02)
- [第一类：GEMM —— 大模型真正的计算核心](#section-03)
- [GEMM 性能到底受什么影响](#section-04)
- [第二类：Attention](#section-05)
- [Attention 最重要的一条演进线](#section-06)
- [Softmax → Online Softmax](#section-07)
- [把 FlashAttention 为什么快画出来](#section-08)
- [Decode Attention ](#section-09)
- [第三类：RMSNorm / Elementwise](#section-10)
- [RoPE](#section-11)
- [第四类：MoE 算子](#section-12)
- [Grouped GEMM 为什么出现](#section-13)
- [Fusion ](#section-14)
- [算子要分 Compute-bound / Memory-bound](#section-15)
- [算子层和 Backend 层的边界](#section-16)
- [典型性能指标](#section-17)
- [面试与自测问题](#section-18)
- [知识图布局草案](#section-19)

## 适用条件与补充

通用 Q/K/V 图用于理解普通 Attention 的计算拆解；MLA 需要单独表达压缩投影与位置分量。MoE Router 的评分函数依模型而定，不能把 Softmax 作为所有模型的固定步骤。FLOPs / Bytes 要说明统计边界；大 GEMM、Decode Attention 的瓶颈判断都是带有 Shape 和负载前提的经验。

详见[技术口径与待核对事项](../00-overview/technical-notes.md)。

### 大模型推理的核心计算到底由哪些算子组成？
本层知识图的核心问题应该是：

> **模型里看到的是 MLA、MoE、MLP；框架里看到的是 ModelRunner；真正落到 GPU 上执行时，会被拆成哪些算子？这些算子为什么快/慢？**

顶部副标题直接写：

> **从数学公式 → Tensor Shape → 性能瓶颈 → Kernel 实现**

---

<a id="section-01"></a>

## 一个 Transformer Block 最终被拆成哪些算子

```text
Hidden States
     │
     ↓
RMSNorm
     │
     ↓
Linear / GEMM
     │
     ├── Q Projection
     ├── K Projection
     └── V Projection
     │
     ↓
RoPE
     │
     ↓
Attention
     │
     ├── QK^T
     ├── Softmax
     └── PV
     │
     ↓
Output Projection GEMM
     │
     ↓
Residual Add
     │
     ↓
RMSNorm
     │
     ↓
MLP / MoE
     │
     ├── Linear / GEMM
     ├── Activation
     ├── Elementwise
     └── Grouped GEMM
     │
     ↓
Residual Add
```

> **模型结构是“语义上的模块”，算子层是“真正被执行的计算”。**

```text
MLA
```

不是一个单独 CUDA 指令。

最终还是被拆成：

```text
Linear
+
RoPE
+
Attention
+
Softmax
+
GEMM
+
Reduction
+
Elementwise
```

---

<a id="section-02"></a>

## 算子分类

中间主体做成四个大分类。

```text
1. 矩阵计算
2. Attention / Reduction
3. Elementwise / Normalization
4. Routing / Data Movement
```

---

<a id="section-03"></a>

## 第一类：GEMM —— 大模型真正的计算核心

公式：

```math
C_{M\times N}=A_{M\times K}B_{K\times N}.
```

将矩阵维度 $`M`$、$`N`$、$`K`$ 与 LLM 的计算对应起来。

例如 Linear：

```text
X: [Tokens, Hidden]
W: [Hidden, Output]
Y: [Tokens, Output]
```

```text
M = Token 数 / Batch×Token
K = Hidden Size
N = Output Dimension
```

### Prefill GEMM
```text
M 大
K/N 大

→ 大矩阵
→ Tensor Core 容易吃满
→ 更偏 Compute-bound
```

### Decode GEMM
```text
M 小
K/N 大

→ Skinny / Small-M GEMM
→ Parallelism 不够
→ Tensor Core 利用率下降
```

---

<a id="section-04"></a>

## GEMM 性能到底受什么影响

```text
GEMM Performance
      │
      ├── M / N / K Shape
      ├── Data Type
      ├── Tile Size
      ├── Tensor Core
      ├── Memory Reuse
      ├── Register Blocking
      └── Pipeline
```

```text
M 很小：
并行 CTA 少 / Tensor Core 不饱和

K 很小：
单 Tile 计算量少
数据搬运 / launch 开销占比变大

M、N 细长：
某个维度 parallelism 不足
Tile 浪费更明显
```

---

<a id="section-05"></a>

## 第二类：Attention

Attention 的算子框可以直接写：

```math
\begin{aligned}
S &= \frac{QK^{\mathsf T}}{\sqrt{d_k}}, \\
P &= \mathrm{softmax}_{\mathrm{row}}(S), \\
O &= PV.
\end{aligned}
```

这里 $`Q\in\mathbb{R}^{L_q\times d_k}`$、$`K\in\mathbb{R}^{L_{kv}\times d_k}`$、$`V\in\mathbb{R}^{L_{kv}\times d_v}`$，因此 $`O\in\mathbb{R}^{L_q\times d_v}`$。这是单个 Attention head 的简化表示，Softmax 沿 key 序列维计算；因果或 Padding Mask 应在 Softmax 前应用。

但是这里必须强调：

> **Attention 在数学上可分解为一串计算；框架中也可以作为一个融合算子暴露。**

```text
Q
│
├─────┐
│     │
│     K
│     │
└─ QK^T
     ↓
   Scale
     ↓
   Mask
     ↓
  Softmax
     ↓
     P
     │
     V
     │
     ↓
    P×V
     ↓
     O
```

---

<a id="section-06"></a>

## Attention 最重要的一条演进线

这块放在知识图中间偏下：

```text
Naive Attention
      ↓
多次 HBM 中间结果读写
      ↓
Online Softmax
      ↓
可以 Tile 化
      ↓
FlashAttention
      ↓
减少 HBM IO
```

---

<a id="section-07"></a>

## Softmax → Online Softmax

> 详细推导与例子：[Online Softmax：Decode 怎样分块计算 Attention？](attention/online-softmax.md)。

普通稳定 Softmax：

```math
\begin{aligned}
m &= \max_{1\le j\le n}x_j, \\
\ell &= \sum_{j=1}^{n}\exp(x_j-m), \\
p_i &= \frac{\exp(x_i-m)}{\ell}, \qquad i=1,\ldots,n.
\end{aligned}
```

$`x_i`$ 是输入分数，$`m`$ 是行最大值，$`\ell`$ 是归一化分母，$`p_i`$ 是输出概率。用 $`\ell`$ 代替字母 l，避免与数字 1 混淆。

问题：

```text
Pass 1：求 max
Pass 2：求 exp + sum
Pass 3：normalize

→ 多次读取数据
```

Online Softmax：

维护两个状态：

```math
\begin{aligned}
m_t &= \max(m_{t-1},x_t), \\
\ell_t &= \ell_{t-1}\exp(m_{t-1}-m_t)+\exp(x_t-m_t), \qquad t\ge 2.
\end{aligned}
```

这是逐元素递推的概念形式，初始化 $`m_1=x_1`$、$`\ell_1=1`$。处理完整行后，$`m_n`$ 与 $`\ell_n`$ 等价于稳定 Softmax 的最大值和分母；分块实现还需要块级状态合并。在 FlashAttention 中还要同步维护加权输出累积，不能只用这两个标量直接得到 Attention 输出。

> **核心意义：支持流式 / 分块处理，不需要保存完整 Score Matrix。**

---

<a id="section-08"></a>

## 把 FlashAttention 为什么快画出来

最好画一张小图：

### Naive
```text
QK^T
  ↓
HBM 写 Scores
  ↓
读取 Scores
  ↓
Softmax
  ↓
HBM 写 P
  ↓
读取 P
  ↓
PV
```

### FlashAttention
```text
Load Q Tile
Load K/V Tile
      ↓
Shared / Register
      ↓
QK + Online Softmax + PV
      ↓
直接累积 O
```

核心：

```text
不把巨大 Attention Matrix
完整落到 HBM
```

---

<a id="section-09"></a>

## Decode Attention 

因为它和 Prefill 真的差别太大。

```text
Decode

Q:
[Batch, Heads, 1, D]

K/V:
[Batch, KV Heads, Context, D]
```

问题：

```text
Q 很少
历史 KV 很多

→ FLOPs 不一定大
→ 但每一轮都必须读大量 KV
→ Arithmetic Intensity 低
→ 常见 Memory-bound
```

```text
KV Cache
  ↓
GQA / MLA
  ↓
Quantized KV
  ↓
FlashMLA / FlashDecode
```

---

<a id="section-10"></a>

## 第三类：RMSNorm / Elementwise

RMSNorm：

```math
y_i=\gamma_i\frac{x_i}{\sqrt{\frac{1}{n}\sum_{j=1}^{n}x_j^2+\epsilon}},\qquad i=1,\ldots,n.
```

$`n`$ 是当前归一化维度；这里使用 $`n`$，避免与 GEMM 的矩阵维度 $`N`$ 混淆。

```text
Load x
 ↓
Square
 ↓
Reduction Sum
 ↓
rsqrt
 ↓
Scale
 ↓
Store
```

```text
计算量小
数据量大

→ Arithmetic Intensity 低
→ 常见 Memory-bound
```

因此优化方向通常：

```text
Vectorized Load
+
Warp Reduction
+
Fusion
```

而不是一味增加复杂计算。

---

<a id="section-11"></a>

## RoPE

RoPE 属于：

```text
Elementwise + Position Transform
```

它本质不是 GEMM。

```text
Q / K
 ↓
两两维度配对
 ↓
sin / cos Rotation
 ↓
Q' / K'
```

特点：

```text
计算量小
访存占比高
容易与 Q/K Projection 或 Attention 融合
```

这就自然引出 Fusion。

---

<a id="section-12"></a>

## 第四类：MoE 算子

MoE 不能只写 Grouped GEMM。

完整算子链应该是：

```text
Router Linear
   ↓
Softmax / Score
   ↓
Top-K
   ↓
Token Count
   ↓
Prefix Sum / Sort / Permute
   ↓
Dispatch
   ↓
Grouped GEMM
   ↓
Activation
   ↓
Grouped GEMM
   ↓
Combine
```

因为面试里问：

> MoE 有哪些算子？

> Grouped GEMM。

---

<a id="section-13"></a>

## Grouped GEMM 为什么出现

```text
Expert 0:
[M0,K] × [K,N]

Expert 1:
[M1,K] × [K,N]

Expert 2:
[M2,K] × [K,N]

...

每个 Mi 不一样
```

如果每个 Expert 单独 launch GEMM：

```text
很多小 GEMM
+
大量 Launch
+
GPU 利用率低
```

Grouped GEMM：

```text
一个统一 Grid
      ↓
CTA 根据 metadata
判断自己属于哪个 Expert
      ↓
计算对应 Expert Tile
```

> “一次启动所有 CTA，然后 block 判断自己属于哪个 expert 的哪个 tile。”

---

<a id="section-14"></a>

## Fusion 

[完整专题：算子融合的动机、场景、实现与验证](fusion/operator-fusion.md)。

```text
GEMM
 ↓
HBM Store
 ↓
Bias
 ↓
HBM Store
 ↓
Activation
 ↓
HBM Store
 ↓
Residual
```

融合后：

```text
GEMM
 ↓
Register / Shared
 ↓
Bias
 ↓
Activation
 ↓
Residual
 ↓
Final Store
```

核心收益：

```text
减少 HBM Round-trip
+
减少 Kernel Launch
+
增加数据局部性
```

再给常见 Fusion：

```text
QKV Projection Fusion

Bias + Activation

RMSNorm + Projection

RoPE + QK

Attention:
QK + Softmax + PV
```

---

<a id="section-15"></a>

## 算子要分 Compute-bound / Memory-bound

```text
                 Arithmetic Intensity
Low  ─────────────────────────────── High

RMSNorm
RoPE
Elementwise
Decode Attention
      │
      │         GEMM
      │         Prefill Attention
      │         Large-M MoE GEMM
      ↓
Memory-bound                Compute-bound
```

注意这是“典型情况”，不是绝对分类。

> **Shape、Batch、数据类型和 Fusion 都会改变瓶颈。**

---

<a id="section-16"></a>

## 算子层和 Backend 层的边界

这一块一定要明确。

```text
算子 / Primitive

定义“算什么”

例如：
GEMM
Attention
Softmax
RMSNorm
RoPE
```

↓

```text
Kernel / Backend

定义“怎么高效算”

例如：
cuBLAS
CUTLASS
FlashAttention
FlashMLA
DeepGEMM
Triton Kernel
CUDA Kernel
```

```text
Attention ≠ FlashAttention

GEMM ≠ DeepGEMM

MoE ≠ DeepEP
```

前者是数学/计算抽象，后者是具体实现。

---

<a id="section-17"></a>

## 典型性能指标

这里和框架层不一样。

框架层看：

```text
TTFT
TPOT
Throughput
```

算子层看：

```text
Latency (us / ms)

Effective Bandwidth

TFLOPS

Compute Utilization

MBU

Arithmetic Intensity
```

比如 GEMM：

```math
U_{\mathrm{compute}}=\frac{F/t}{P_{\mathrm{peak}}}.
```

$`F`$ 是选定统计范围内的运算次数，$`t`$ 是运行时间，$`P_{\mathrm{peak}}`$ 是匹配精度与设备的峰值计算吞吐，单位均按 FLOP/s 对齐。

Memory Bound 算子：

```math
\mathrm{MBU}=\frac{B_{\mathrm{HBM}}/t}{BW_{\mathrm{peak}}}.
```

$`B_{\mathrm{HBM}}`$ 是实际 HBM 传输字节数，$`BW_{\mathrm{peak}}`$ 是峰值 HBM 带宽。若只用算法输入输出字节估算，应标为有效带宽，不能直接当作实测 HBM 利用率。

---

<a id="section-18"></a>

## 面试与自测问题

1. GEMM 的 $`M`$、$`N`$、$`K`$ 分别代表什么？
2. 为什么小 $`M`$ GEMM 效率差？
3. 为什么 $`K`$ 很小时 Tensor Core 利用率可能低？
4. Softmax 和 Online Softmax 有什么区别？
5. FlashAttention 为什么减少 HBM 访问？
6. Decode Attention 为什么常常 memory-bound？
7. RMSNorm 为什么通常偏 memory-bound？
8. Fusion 为什么能提升性能？
9. Grouped GEMM 为什么适合 MoE？
10. 怎么判断一个算子是 compute-bound 还是 memory-bound？

---

<a id="section-19"></a>

## 知识图布局草案

像这样：

```text
┌──────────────────────────────────────────────────────────────┐
│    大模型核心算子：从数学公式到性能瓶颈                    │
│    Attention / GEMM / RMSNorm / RoPE / MoE                  │
├───────────────┬─────────────────────────┬────────────────────┤
│               │                         │ GEMM               │
│ Transformer   │ Attention Pipeline      │ C = A × B          │
│ Block         │                         │ M/N/K Shape        │
│ ↓             │ QK^T                    ├────────────────────┤
│ RMSNorm       │ ↓                       │ Prefill vs Decode  │
│ ↓             │ Softmax                 │ GEMM               │
│ GEMM          │ ↓                       ├────────────────────┤
│ ↓             │ PV                      │ RMSNorm / RoPE     │
│ Attention     │                         │                    │
│ ↓             ├─────────────────────────┼────────────────────┤
│ GEMM          │ Softmax → Online        │ MoE Operator Flow  │
│ ↓             │ → FlashAttention        │ Router → TopK      │
│ MoE           │                         │ → Dispatch         │
│               │                         │ → Grouped GEMM     │
├───────────────┴─────────────────────────┴────────────────────┤
│ Fusion：减少 HBM 中间结果 + Kernel Launch                   │
├────────────────────────────┬─────────────────────────────────┤
│ Compute vs Memory Bound    │ Operator → Backend Mapping      │
│ Roofline                   │ Attention → FlashMLA            │
│                            │ GEMM → DeepGEMM                 │
├────────────────────────────┴─────────────────────────────────┤
│ 面试高频问题                                                 │
└──────────────────────────────────────────────────────────────┘
```

---

```text
第一张
模型层
“模型怎么算”
       ↓
第二张
框架层
“请求怎么被组织起来算”
       ↓
第三张
算子层
“真正有哪些计算”
```
