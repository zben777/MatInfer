# Decode 中的 Attention 与 FFN：生成下一个 token 时怎样分工

> 在一次普通自回归 decode 中，每个请求把**一个当前 token**送入模型，利用已有的历史 KV Cache，计算下一个 token 的 logits。Attention 让当前表示读取历史上下文，FFN 对聚合后的当前表示进行特征变换。

> 本文以 decoder-only、因果自注意力、已有 KV Cache 的单 token decode 为主线。先看一个请求、一个 Transformer Block；多请求批处理与 prefill 放在对应位置简短对照。

## 1. 先固定一次 decode 的场景

> 假设当前处理的是位置 4：位置 1～3 的 K/V 已经保存在**这一层自己的缓存**中。隐藏维度 d＝512，标准 MHA 有 8 个头，每头维度 dₕ＝64。当前输入形状是 `[1, 512]`；省略批次后，每个头的 Query 是 `[1, 64]`，加入当前位置后可读的 K/V 是 `[4, 64]`。

> 一次 decode 可以按以下顺序理解：
>
> 1. **输入当前 token。** 它通常是上一轮选出的 token；这一轮计算用来预测后继 token。第一层从其 embedding 出发，后续层读取上一层的当前表示。
> 2. **经过 Attention。** 当前 Query 读取这一层的历史 K/V 和当前 K/V，形成当前 token 的上下文表示。
> 3. **经过 FFN。** 对当前表示进行特征变换，不重新处理全部历史 token。
> 4. **继续后面的层。** 所有 Block 完成后，经模型的最终 Norm、LM Head 得到 logits，再选择下一个 token。一个 Block 的输出本身不是最终词表概率。

## 2. Attention（注意力机制）

### 具体做什么

> Attention 根据当前 Query 与可见 Key 的匹配分数，计算聚合权重，再对相应 Value 加权求和。在 decode 中，**需要新计算的是当前 token 的 Q/K/V，历史 K/V 直接从缓存读取**；不需要重新为历史所有 token 做 Q/K/V 投影。

> 把计算过程分成五步：
>
> 1. **当前表示投影为 Q/K/V。** 在一个典型 Pre-Norm Block 中，Attention 使用当前表示经过 Norm 后的结果 zₜ；它的三种投影分别承担查询、匹配和内容聚合的角色。
> 2. **扩展可见 K/V。** 把当前 Kₜ、Vₜ 纳入本层缓存，与历史 K/V 一起供本步使用。这里只新增当前 K/V，不重新生成历史缓存。
> 3. **计算相关分数。** 当前 Qₜ 分别与所有可见 Key 点积，再除以 √dₕ，得到长度为 T 的分数行；T 是包含当前位置的可见序列长度。
> 4. **沿 Key 位置做 Softmax。** 得到一行聚合权重。若有 padding、滑动窗口或其他可见性限制，还要应用相应 Mask。
> 5. **加权 Value 并输出投影。** 每个头得到当前输出，多头结果拼接后经过 Wₒ，回到隐藏维度 d，以便与残差相加。

> 先写当前投影。公式里的 Qₜ、Kₜ、Vₜ 可以表示所有头的投影，之后按头拆分：

```math
Q_t=z_tW_Q,\qquad K_t=z_tW_K,\qquad V_t=z_tW_V.
```

> 对一个头，把当前 Key/Value 与历史数据沿序列维组合：

```math
K_{\le t}=\mathrm{Concat}(K_{\lt t},K_t),\qquad
V_{\le t}=\mathrm{Concat}(V_{\lt t},V_t).
```

> 再计算这个头的权重与输出：

```math
A_t=\mathrm{Softmax}\left(\frac{Q_tK_{\le t}^{\mathsf T}}{\sqrt{d_h}}+M_t\right),
\qquad o_t=A_tV_{\le t}.
```

> Concat 描述逻辑上的序列扩展，不要求实现真的分配新张量并拷贝全部历史数据；推理框架通常写入预分配或分页缓存。若模型使用 RoPE 等位置处理，还要在模型规定的位置处理当前 Q/K，并保持与缓存表示一致。[^transformer]

### 本例的形状与数字

| 对象 | 单头形状 | 在本步中做什么 |
| --- | --- | --- |
| 当前 Qₜ | `[1, 64]` | 为当前 token 查询上下文 |
| 历史 K、V | 各 `[3, 64]` | 从本层缓存读取 |
| 加入当前后的 K、V | 各 `[4, 64]` | 提供四个可见位置 |
| 分数、权重 Aₜ | `[1, 4]` | 当前 Query 对四个 Key 的权重 |
| 单头输出 oₜ | `[1, 64]` | 聚合后的当前表示 |
| 8 个头拼接、输出投影后 | `[1, 512]` | 继续残差更新与 FFN |

> 如果 Softmax 权重是 `[0.1, 0.2, 0.3, 0.4]`，这个头的输出就是 `0.1V₁ + 0.2V₂ + 0.3V₃ + 0.4V₄`。四个 Value 是向量，权重分别作用于整个 Value 向量。**读取四个位置，并不意味着本步产生四个新的 token 表示：输出仍只有当前位置的一行。**

### 作用

> 1. **读取历史上下文。** 当前 token 能直接从 Mask 允许的历史位置聚合信息。标准因果全局 Attention 可以访问全部历史；滑动窗口等变体会限制范围。可见不等于一定被正确利用，也不等于已经解决所有长距离依赖问题。
> 2. **根据当前输入动态选择聚合权重。** 不同 Query 会产生不同分数与权重，同样的缓存可以被不同头以不同方式读取。权重由输入计算，并非固定参数。即使固定权重时 `AV` 对 V 是线性的，完整 Attention 仍包含输入相关点积与 Softmax，整体是非线性的。
> 3. **复用历史计算，支持一步内并行。** KV Cache 让历史 K/V 投影不必重复计算。同一步中的多个头、多个请求和矩阵运算可以并行；后续生成步骤仍依赖前一步选出的 token，不能据此并行生成全部未来 token。
> 4. **提供可观察的聚合行为。** 本步每个头的一行权重，能显示当前 Query 怎样分配对可见 Value 的聚合权重。但它不是最终预测的完整因果解释，Value 内容、输出投影、残差和后续层也会影响结果。[^explanation]

### 与 prefill 的区别

> 本例 decode 每头的 Query 长度是 1，逻辑权重形状是 `[1, T]`。prefill 中已知输入有多个 Query 位置，整段因果自注意力的逻辑权重可为 `[S, S]`，多个位置可以一起计算。这里说的是数学形状；FlashAttention 等实现不必在显存中物化完整权重矩阵。

## 3. FFN（前馈网络）

### 具体做什么

> FFN 对 Attention 与残差更新后的当前表示进行特征变换。在典型 Pre-Norm Block 中，它使用第二次 Norm 的输出作为输入。设这份输入为 u，形状是 `[1, d]`；经过 FFN 后仍为 `[1, d]`。

> 基础两层 FFN 可以分成三步：
>
> 1. **特征投影。** 用 W₁ 将当前向量从 d 维映射到 m 维，形成更宽的中间特征表示。
> 2. **非线性变换。** 对中间特征应用 ReLU、GELU 等激活函数，使不同输入产生不同响应。
> 3. **投影回隐藏维度。** 用 W₂ 把 m 维结果映射回 d 维，供残差相加并进入下一层。

```math
\mathrm{FFN}(u)=\phi(uW_1+b_1)W_2+b_2,
\qquad W_1\in\mathbb{R}^{d\times m},\quad W_2\in\mathbb{R}^{m\times d}.
```

> φ 取 ReLU 时，负值置零、正值保留；取 GELU 时，采用平滑的输入相关调制。原始 Transformer 使用 ReLU 和 m＝4d。本例对应 `[1, 512] → [1, 2048] → [1, 512]`，并非把历史四个 token 都送进 FFN。[^transformer]

> 不同位置复用同一套 dense FFN 参数，每个位置的输出只直接依赖自己送入 FFN 的那行表示。进入 FFN 之前，这行表示已经可以包含 Attention 聚合得到的上下文，因此“逐位置处理”不意味着“没有上下文”。多个请求批处理时，可以把当前向量堆成 `[B, d]` 一起算，仍不引入请求之间的信息聚合。

### SwiGLU：门控形式的 FFN

> SwiGLU 采用两条输入投影路径，再逐元素相乘：
>
> 1. **门控分支。** uWg 经过 SiLU，得到随当前输入变化的调制系数。
> 2. **特征分支。** uWu 产生另一组 m 维特征。
> 3. **逐元素调制。** 两个 `[1, m]` 结果按坐标相乘；这一步发生在特征维度，不是 token 之间的注意力。
> 4. **输出投影。** 用 Wd 从 m 维映射回 d 维。

```math
\mathrm{FFN}_{\mathrm{SwiGLU}}(u)
=\left[\mathrm{SiLU}(uW_g)\odot(uW_u)\right]W_d,
\qquad \mathrm{SiLU}(z)=\frac{z}{1+e^{-z}}.
```

> Wg、Wu 是 `[d, m]`，Wd 是 `[m, d]`。SiLU 的输出不限定在 0～1，门控也不等价于一个二值开关。两层 FFN 有两个主要权重矩阵，SwiGLU 有三个；其宽度由模型配置决定。[^glu]

### 作用

> 1. **增加逐位置的非线性表达能力。** 如果没有激活或门控，连续两个线性层可合成一个线性层：`(uW₁)W₂ = u(W₁W₂)`；加 bias 后也仍是仿射变换。中间非线性让 FFN 表达无法由一次固定线性投影替代的函数。
> 2. **组合与变换上下文特征。** Attention 把可见位置的信息聚合到当前表示，FFN 再对这些已有特征进行投影、激活或门控，并映射回隐藏维度。升维提供更宽的表示空间，但不自动保证特征变得更好。
> 3. **保持逐位置计算。** 本步 FFN 只处理当前行，不直接扫描历史 token，不需要维护与历史序列长度一起增长的 FFN KV Cache。标准 dense FFN 的层权重在位置之间共享。
> 4. **提供额外参数容量。** 中间维度与矩阵数量决定 FFN 权重规模。它是模型表达能力的重要组成部分，但不能仅凭参数较多，就把模型的记忆或推理能力全部归于 FFN。

## 4. FFN 参数约占 2/3，怎样推导

> 限定标准 MHA：Q、K、V、O 的投影矩阵均为 `[d, d]`；两层 dense FFN 中间维度 m＝4d。忽略 bias、Norm、Embedding 与 LM Head：

```math
P_{\mathrm{Attention}}=4d^2,\qquad P_{\mathrm{FFN}}=d(4d)+(4d)d=8d^2.
```

```math
\frac{P_{\mathrm{FFN}}}{P_{\mathrm{Attention}}+P_{\mathrm{FFN}}}
=\frac{8d^2}{4d^2+8d^2}=\frac{2}{3}.
```

> 这说明在该配置下，FFN 占**这两部分主要权重参数之和**的 2/3。本例 d＝512，Attention 为 1,048,576 个参数，FFN 为 2,097,152 个参数。decode 每步输入只有一个 token，并不意味着只需要其中一小部分 dense 权重。

> 参数分析还要分清三点：
>
> 1. **比例随结构变化。** GQA 减少 K/V 投影参数；SwiGLU 的三个矩阵共约 3dm 个参数；MoE 又涉及多个专家，不能统一套用 2/3。
> 2. **宽度可以匹配参数预算。** 若 SwiGLU 选 m 约为 8d/3，就有 `3dm ≈ 8d²`，与传统 m＝4d 的两层 FFN 相近；实际配置可能为对齐要求取整。[^glu]
> 3. **参数占比不等于 decode 延迟占比。** Attention 还需读取历史 KV，读取量随上下文长度增长；dense FFN 本步的主要计算形状取决于批次和特征维度。不同形状、精度、硬件与实现会改变瓶颈。

## 5. 两者关系：当前 token 先聚合，再变换

| 对比项 | Decode Attention | Decode FFN |
| --- | --- | --- |
| 本步处理对象 | 当前 Query 与可见历史、当前 K/V | 当前 token 的上下文表示 |
| 是否直接读取历史 token 信息 | 从本层 KV Cache 读取历史 K/V | 不直接扫描历史位置 |
| 信息怎样变化 | 按输入相关权重聚合可见 Value | 对当前特征投影、激活或门控 |
| 单请求输出 | 当前 token 一行表示 | 当前 token 一行表示 |
| 上下文长度增长的影响 | 一般增加可见 KV 读取与注意力计算 | 固定 d、m 下，不因历史更长而增加处理行数 |
| 一步内并行 | 可并行处理头、请求及相关计算 | 可批量处理多个请求的当前行 |

> 可以用“Attention 沟通、FFN 加工”帮助记忆：本步 Attention 让当前 token 获取上下文，FFN 对获取上下文后的表示做进一步变换。完整 Block 还包括 Norm、残差等结构；“Attention＋FFN”是对子层分工的简称。

## 面试与追问

**一次 decode 会重新计算所有历史 token 的 FFN 吗？**

> 标准 KV Cache decode 不会。历史位置在早先的前向中已经处理过，本步只让当前 token 经过各层 FFN。历史 K/V 则按层缓存，供当前 Attention 读取。

**为什么 Query 只有一行，却能访问很长的上下文？**

> Query 数量决定本步要输出多少行，Key/Value 数量决定每行能读取多少可见位置。单头 `[1, dₕ]` 的 Q 与 `[T, dₕ]` 的 K 点积得到 `[1, T]`，再加权 `[T, dₕ]` 的 V 得到 `[1, dₕ]`。

**FFN 不读历史，为什么还能处理上下文？**

> Attention 已经把历史信息聚合进当前表示，FFN 变换的是这份表示。它不直接读取其他位置，但输入并不局限于原始 token embedding。

**生成的下一个 token，会立刻包含在本步的 KV Cache 吗？**

> 本步新增的是输入当前位置的 K/V。模型先完成计算，再从 logits 选出后继 token；普通自回归流程中，后继 token 的 K/V 要等下一轮把它送入模型时才计算。

## 参考与关联专题

- [Attention Is All You Need](https://arxiv.org/html/1706.03762v7)：自注意力、因果 Mask、逐位置 FFN 与原始配置。[^transformer]
- [GLU Variants Improve Transformer](https://arxiv.org/html/2002.05202v1)：门控 FFN 与参数量匹配。[^glu]
- [Attention is not Explanation](https://aclanthology.org/N19-1357/)：注意力权重作为解释的局限。[^explanation]

[^transformer]: Vaswani 等，Attention Is All You Need，2017。
[^glu]: Shazeer，GLU Variants Improve Transformer，2020。
[^explanation]: Jain 与 Wallace，Attention is not Explanation，2019。

[返回 Transformer 基础](README.md) · [Attention 专题](../attention/README.md) · [归一化专题](../normalization/README.md) · [返回模型原理](../README.md)
