# 模型原理

> 状态：内容草案。已整理原稿的技术主题、公式、流程、图布局和题目；完整教程、源码版本与实测结果将随学习补齐。示意代码不等同于已验证实现。

**核心问题：一个 Token 怎样经过模型，模型结构怎样影响推理成本？**

以 DeepSeek-V3 / R1 的 MLA 与 MoE 为主要案例，联系 Transformer、RMSNorm、RoPE、KV Cache 与采样。

建议学习顺序：先建立 Token 主流程，再理解单个 Block；从 KV Cache 与 MoE 的计算特征连接框架和 Backend。

[返回总导航](../../导航.md) · [学习路线](../../roadmap/README.md) · [面试题索引](../../interview/README.md)

## 内容索引

- [Attention 细分专题：MHA 独立性、Linear Attention、手写 GQA](attention/README.md)
- [Token 主流程](#section-01)
- [DeepSeek Transformer Block](#section-02)
- [RMSNorm](#section-03)
- [MHA、GQA 与 MLA](#section-04)
- [MLA 内部流程](#section-05)
- [Attention 数学与阶段差异](#section-06)
- [DeepSeekMoE](#section-07)
- [RoPE](#section-08)
- [KV Cache](#section-09)
- [Sampling](#section-10)
- [面试与自测问题](#section-11)
- [知识图布局草案](#section-12)

## 适用条件与补充

MLA 的缓存不只是一个 latent：理解 DeepSeek-V3 时需要区分压缩 KV 表示与解耦 RoPE 的 key 分量，并继续学习投影吸收。下面的 latent 流程图是概念简图。MoE 图主要展示 routed experts，shared experts 与早期 dense layers 需要在完整模型解析中补充。普通自回归 Decode 每轮通常生成一个 Token，投机解码等方案存在多 Token 情形。

详见[技术口径与待核对事项](../00-overview/technical-notes.md)。

> **DeepSeek 模型原理：从输入 Token 到下一个 Token**
_以 DeepSeek-V3 / R1 的推理结构为主线：MLA + DeepSeekMoE_

<a id="section-01"></a>

## Token 主流程

从输入文本到下一个 Token 的完整流程：

```text
输入文本
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Token Embedding
   ↓
┌──────────────────────────┐
│ Transformer Block × L    │
│                          │
│ RMSNorm                  │
│    ↓                     │
│ MLA                      │
│    ↓                     │
│ Residual                 │
│    ↓                     │
│ RMSNorm                  │
│    ↓                     │
│ MLP / DeepSeekMoE        │
│    ↓                     │
│ Residual                 │
└──────────────────────────┘
   ↓
Final RMSNorm
   ↓
LM Head
   ↓
Logits
   ↓
Sampling
   ↓
Next Token
```

**× L** 表示重复的 Transformer Blocks；缓存按层管理。

> 在这里讨论的缓存式自回归 Transformer 中，各层 Attention 通常分别维护自己的 Cache。

> **普通自回归 Decode：每轮通常产生一个新 Token，再继续经过所有 Transformer Blocks；投机解码另行分析。**

---

<a id="section-02"></a>

## DeepSeek Transformer Block

```text
                 x_l
                  │
                  ↓
              RMSNorm
                  │
                  ↓
            ┌─────────┐
            │   MLA   │
            └─────────┘
                  │
           ┌──────┴──────┐
           │ Residual Add │ ← x_l
           └──────┬──────┘
                  │
                  ↓
              RMSNorm
                  │
                  ↓
         ┌────────────────┐
         │ MLP / DeepSeek │
         │      MoE       │
         └────────────────┘
                  │
           ┌──────┴──────┐
           │ Residual Add │
           └──────┬──────┘
                  │
                  ↓
                x_l+1
```

```text
Dense Layer → MLP
MoE Layer   → DeepSeekMoE
```

> DeepSeek 每层都是 MoE 吗？

---

<a id="section-03"></a>

## RMSNorm

```math
\begin{aligned}
r(x) &= \sqrt{\frac{1}{n}\sum_{j=1}^{n}x_j^2+\epsilon}, \\
y_i &= \operatorname{RMSNorm}(x)_i = \gamma_i\frac{x_i}{r(x)}.
\end{aligned}
```

其中，$n$ 是归一化维度，$\gamma_i$ 是可学习的逐元素缩放参数，$\epsilon>0$ 用于数值稳定。

```text
① 不计算 mean
② Pre-Norm 结构
③ Reduction + Elementwise
④ 推理中通常偏 Memory-Bound
```

> **为什么 LLM 越来越多使用 RMSNorm，而不是 LayerNorm？**

**公式 + 作用 + 性能特征 + 面试追问。**

---

<a id="section-04"></a>

## MHA、GQA 与 MLA

不要一上来就直接画 MLA，否则容易记结构却不知道为什么出现。

画成演进关系：

```text
MHA
┌───────────────┐
Q heads: H
K heads: H
V heads: H
└───────────────┘
        ↓
     KV 很大

GQA
┌───────────────┐
Q heads: H
KV heads: H/g
多个 Q 共享 KV
└───────────────┘
        ↓
  KV Cache 减少

MLA
┌───────────────┐
KV → latent
低维压缩表示
└───────────────┘
        ↓
进一步降低 KV Cache
```

> **核心目标：降低 Decode 阶段历史 KV 的存储与读取压力。**

因为它把：

**模型结构 → KV Cache → Decode Memory Bound → FlashMLA**

直接串起来了。

---

<a id="section-05"></a>

## MLA 内部流程

普通 Transformer 的投影表示：

```text
Q = XWq
K = XWk
V = XWv
```

这里只用于对照。

本节突出 DeepSeek 的压缩表示。

```text
                 Hidden State h_t
                      │
            ┌─────────┴──────────┐
            ↓                    ↓
       Query Path            KV Path
            │                    │
            │              Down Projection
            │                    ↓
            │               KV Latent
            │                    │
            │             Up Projection /
            │             Attention 使用
            │                    │
            └──────────┬─────────┘
                       ↓
                   Attention
                       ↓
                    Output
```

```text
KV Latent
   │
   └────→ KV Cache
```

> **Decode 时反复读取的是历史 Cache，因此 Cache 表示形式直接影响显存和带宽。**

这个地方后面就是量化 KV 实验 最容易挂上去的位置。

---

<a id="section-06"></a>

## Attention 数学与阶段差异

这块继续保留经典公式，因为后面算子层还会继续展开。

```math
\begin{aligned}
S &= \frac{QK^{\mathsf T}}{\sqrt{d_k}}, \\
P &= \operatorname{softmax}_{\mathrm{row}}(S), \\
O &= PV.
\end{aligned}
```

这里 $Q\in\mathbb{R}^{L_q\times d_k}$、$K\in\mathbb{R}^{L_{kv}\times d_k}$、$V\in\mathbb{R}^{L_{kv}\times d_v}$，因此 $O\in\mathbb{R}^{L_q\times d_v}$。这是单个 Attention head 的简化表示，Softmax 沿 key 序列维计算；因果或 Padding Mask 应在 Softmax 前应用。

分为两个阶段：

### Prefill
```text
Q: 多 Token
K/V: 多 Token

大矩阵计算
更容易利用 Tensor Core
```

### Decode
```text
Q: 新 Token
K/V: 全部历史 Token

Q 很短
KV 很长
更容易受 Memory Bandwidth 限制
```

模型与算子层的联系：

```text
Attention
   ↓
Decode Attention
   ↓
Memory Bound
   ↓
FlashMLA / Quantized KV
```

---

<a id="section-07"></a>

## DeepSeekMoE

MoE 需要展开完整计算与数据流。

应该把完整路径画出来：

```text
              Hidden States
                    │
                    ↓
                 Router
                    │
                    ↓
               Routing Score
                    │
                    ↓
                 Top-K
                    │
                    ↓
           Token Dispatch
            ↙      ↓       ↘
       Expert 0 Expert 1 ... Expert N
            ↘      ↓       ↙
             Grouped GEMM
                    │
                    ↓
                 Combine
                    │
                    ↓
                  Output
```

```text
为什么 MoE？

总参数量 ↑
每 Token 激活参数相对受控
```

```text
性能问题：

• Top-K 后每个 Expert 的 M 不一样
• Small-M GEMM 利用率可能较低
• Token Dispatch / Combine 有数据搬运
• 多卡 EP 引入 All-to-All 通信
```

```text
MoE
 ├── Expert GEMM → DeepGEMM
 └── EP 通信     → DeepEP
```

---

<a id="section-08"></a>

## RoPE

RoPE 这一块不用像参考图那么大，但保留一个二维旋转示意会很好。

公式可以简化：

```math
\begin{bmatrix}
x'_{p,2i} \\
x'_{p,2i+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta_{p,i} & -\sin\theta_{p,i} \\
\sin\theta_{p,i} & \cos\theta_{p,i}
\end{bmatrix}
\begin{bmatrix}
x_{p,2i} \\
x_{p,2i+1}
\end{bmatrix}.
```

$p$ 表示 Token 位置，$i$ 表示维度对，$\theta_{p,i}$ 是该位置与维度对对应的旋转角。上式采用相邻维度配对的概念表示；实际配对 Layout 与频率缩放方式需结合模型实现确认。

```text
RoPE 作用：

• 注入位置信息
• 本质是二维旋转
• Attention score 获得相对位置信息
```

然后加一句：

> MLA 中 RoPE 与压缩表示的处理方式是理解 DeepSeek Attention 的一个关键点。

---

<a id="section-09"></a>

## KV Cache

这一块必须有，因为它是“模型层 → 推理框架层”的桥。

```text
第 t 个 Token

K_t / V_t
     ↓
KV Cache

历史：
[K_1, K_2, ... K_t]
[V_1, V_2, ... V_t]
```

并注明：

```text
Without KV Cache

每生成 1 Token
重新计算历史 K/V

          ↓

With KV Cache

历史 K/V 直接复用
只计算当前 Token
```

```text
MHA
  ↓
KV 多

GQA
  ↓
KV 减少

MLA
  ↓
进一步压缩
```

---

<a id="section-10"></a>

## Sampling

不用讲太多，保留：

```text
Logits
  ↓
Softmax / Score Processing
  ↓
Greedy
Top-K
Top-P
Temperature
  ↓
Next Token
```

---

<a id="section-11"></a>

## 面试与自测问题

1. **一个 Transformer Block 的完整流程是什么？**
2. **为什么使用 RMSNorm？**
3. **MHA、MQA、GQA、MLA 有什么区别？**
4. **MLA 为什么可以减少 KV Cache？**
5. **RoPE 为什么作用于 Q/K？**
6. **Prefill 和 Decode 的 Attention 有什么区别？**
7. **DeepSeekMoE 从 Router 到 Expert 的完整流程是什么？**
8. **为什么 Decode 往往比 Prefill 更 memory-bound？**

+ 学习图
+ 知识地图
+ 面试复习图

---

<a id="section-12"></a>

## 知识图布局草案

最终布局就严格这样：

```text
┌──────────────────────────────────────────────────────────┐
│ DeepSeek 模型原理：从 Token 到 Next Token               │
├───────────────┬───────────────────────┬──────────────────┤
│               │                       │ RMSNorm          │
│ 完整 Token    │ Transformer Block     ├──────────────────┤
│ 生命周期      │                       │ MHA→GQA→MLA      │
│               │ RMSNorm → MLA         ├──────────────────┤
│ Text          │ → Residual            │ MLA 内部数据流   │
│ ↓             │ → RMSNorm             ├──────────────────┤
│ Tokenizer     │ → DeepSeekMoE         │ Attention 公式   │
│ ↓             │ → Residual            │ Prefill/Decode   │
│ Embedding     │                       │                  │
│ ↓             ├───────────────────────┼──────────────────┤
│ Block × L     │ DeepSeekMoE           │ RoPE             │
│ ↓             │ Router → TopK         │                  │
│ LM Head       │ → Dispatch            │                  │
│ ↓             │ → Grouped GEMM        │                  │
│ Next Token    │ → Combine             │                  │
├───────────────┴───────────┬───────────┴──────────────────┤
│ KV Cache                  │ Sampling / 符号 / 关键结论    │
├───────────────────────────┴──────────────────────────────┤
│ 高频面试问题                                              │
└──────────────────────────────────────────────────────────┘
```

> **主流程够清楚 + 关键公式够用 + 原理够深 + 性能问题能接下一层 + 最后能直接拿来面试。**
