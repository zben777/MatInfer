# 手写 GQA：从分组到缓存推理

GQA（Grouped-Query Attention）让多个 Query 头共享一组 Key / Value。要把它写对，需要理清三件事：每个 Query 头读哪一组 KV、每个位置可以看见哪些 Token、历史 KV 怎样保留。

这篇从一个具体例子开始，逐步解释投影、分头、Attention、因果 Mask 和 KV Cache。配套代码已经通过 CPU 正确性检查；完整实现与验证记录放在文末。

[Attention 导航](README.md) · [完整代码](../../../labs/attention/gqa.py) · [验证记录](../../../labs/attention/README.md)

## 问题与结论

普通 MHA 中，每个 Query 头对应自己的 KV 头。GQA 保留 Query 头数，让同一组 Query 头读取共同的 K/V，从而减少 KV 投影输出、历史缓存大小及相关数据流量。共享 K/V 后，各 Query 头仍然计算自己的注意力权重和输出。

**写 GQA 时，缓存应保存真实 KV 头；因果 Mask 应使用包含历史长度的位置。** 显式复制 KV 可以帮助理解头映射，但不应把复制后的数据当作长期缓存。

阅读本文需要知道 Linear 投影、矩阵乘法和 Softmax 的基本含义。这里以 Q/K/V 每头维度相同、Query 头连续分组、Batch 内序列等长为前提。RoPE、Padding 和生产缓存实现将在后续专题展开。

## 原理与例子

### 八个 Query 头怎样共享两个 KV 头

假设输入是一条长度为 5 的序列，每个 Token 的隐藏维度为 32。使用 8 个 Query 头、2 个 KV 头，每个头的维度为 4。输入 Tensor 的形状就是 `[1, 5, 32]`。

8 个 Query 头分成两组，每组 4 个头，映射如下：

| Query 头 | 读取的 Key 头 | 读取的 Value 头 |
| --- | --- | --- |
| Q0、Q1、Q2、Q3 | K0 | V0 |
| Q4、Q5、Q6、Q7 | K1 | V1 |

例如 Q0 和 Q1 都读取 K0、V0，但它们的 Query 投影参数不同，产生的 Query 通常也不同。因此它们对历史 Token 的注意力分布并不需要相同。

把这个对应关系写成一般形式，设 Query 头数为 $`H_q`$、KV 头数为 $`H_{kv}`$，每组包含 g 个 Query 头，头编号 i 从 0 开始：

```math
g=\frac{H_q}{H_{kv}},\qquad
j(i)=\left\lfloor\frac{i}{g}\right\rfloor.
```

j(i) 表示第 i 个 Query 头读取的 KV 头编号。这里要求头数为正，且 Query 头数能被 KV 头数整除。当两种头数相等时，退化为 MHA；只有一个 KV 头时，就是 MQA。

这种共享不要求 Query 头串行执行。各头可以分别读取共享输入并计算；它们的依赖边界见 [MHA 各头是否独立](mha-head-independence.md)。

### 先投影，再把特征拆成头

输入隐藏向量经过三个 Linear，分别生成 Q、K、V。下面沿用配套代码中的参数名：`hidden_dim` 是输入隐藏维度，`num_q_heads` 和 `num_kv_heads` 是两种头数，`head_dim` 是每头维度。

```python
self.w_q = nn.Linear(hidden_dim, num_q_heads * head_dim, bias=False)
self.w_k = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
self.w_v = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
```

在这个例子里，Q 投影为每个 Token 生成 8 × 4 = 32 个特征，K 和 V 各生成 2 × 4 = 8 个特征。投影后的 Q 是 `[1, 5, 32]`，K/V 是 `[1, 5, 8]`。

接着把特征维拆成“头数 × 每头维度”，再把头维移到序列维前面：

```python
B, S, _ = x.shape
q = self.w_q(x).view(B, S, self.num_q_heads, self.head_dim)
k = self.w_k(x).view(B, S, self.num_kv_heads, self.head_dim)
v = self.w_v(x).view(B, S, self.num_kv_heads, self.head_dim)
q = q.transpose(1, 2)
k = k.transpose(1, 2)
v = v.transpose(1, 2)
```

此时 Q 是 `[1, 8, 5, 4]`，K/V 是 `[1, 2, 5, 4]`。头维不相同，恰好表达了 GQA 的结构。每个头的投影通常读取完整的输入隐藏向量；这里拆分的是投影后的特征，没有把输入预先切成八份。

输入隐藏维度与“Query 头数 × 每头维度”在这个例子中相等，但并非数学要求。输入投影和输出投影可以完成维度映射，因此不需要无条件添加这条相等断言。

### 每个 Query 头仍然算自己的 Attention

为了直接使用普通 batched matmul，教学实现将每个 KV 头沿头维连续重复四次：

```python
k = k.repeat_interleave(self.group_size, dim=1)
v = v.repeat_interleave(self.group_size, dim=1)
```

重复后，K/V 的头顺序为 `0, 0, 0, 0, 1, 1, 1, 1`，形状变成 `[1, 8, 5, 4]`。这个顺序与上面的组映射一致。若随意用 `repeat` 得到交替排列，可能改变 Query 与 KV 的对应关系。

给定第 i 个 Query 头，它的计算是：

```math
O_i=\mathrm{softmax}_{\mathrm{row}}
\left(\frac{Q_iK_{j(i)}^T}{\sqrt{d_h}}+M\right)V_{j(i)}.
```

其中 $`d_h`$ 是每头维度，M 是加性 Mask：允许的位置取 0，禁止的位置取负无穷。Softmax 沿 Key 的序列维归一化；这里假设每个 Query 至少有一个可见 Key。

对于长度为 5 的完整输入，每个头先生成 5 × 5 的分数矩阵。所有 Query 头一起计算时，`scores` 是 `[1, 8, 5, 5]`。加权 V 后，输出恢复为 `[1, 8, 5, 4]`。

```python
scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
# 自回归场景需要在这里加入因果 Mask，下一节展开。
attn = torch.softmax(scores, dim=-1)
out = attn @ v
```

最后把头维放回 Token 后面，合并头特征，再做输出投影：

```python
out = out.transpose(1, 2).contiguous().view(B, S, -1)
out = self.w_o(out)
```

合并后的形状是 `[1, 5, 32]`。`w_o` 将多个头的特征混合，并映射回输入隐藏维度。转置后的 Tensor 通常不能直接按目标形状 `view`，所以这里先调用 `contiguous()` 整理布局；使用 `reshape` 也应理解它可能发生复制。

## 实现与工程取舍

### 因果 Mask 要考虑历史位置

上面的矩阵乘流程如果不加入 Mask，就允许每个位置读取整段序列，包括未来 Token。这适合双向自注意力，不能直接用于自回归 Decoder。

完整 Prefill 中，位置 0 只能看到自己，位置 1 可以看到位置 0 和 1，依此类推。Mask 必须加在 Softmax 之前，让不可见位置的概率变为零。

缓存推理还多了一个位置偏移。假设刚才例子中的前 3 个 Token 已经计算并缓存，现在一次输入剩下的 2 个 Token。当前 Query 长度是 2，拼接后的 Key 长度是 5：

| 当前 Query 的局部编号 | 在完整序列中的位置 | 可以读取的 Key |
| --- | --- | --- |
| 0 | 3 | 0、1、2、3 |
| 1 | 4 | 0、1、2、3、4 |

设历史长度为 $`T_p`$，当前 Query 的局部编号为 q，Key 编号为 k，则可见条件是：

```math
k\leq T_p+q.
```

这时 Mask 是 2 × 5，不能直接套用不带偏移的左上对齐下三角。配套实现按实际位置生成可见性：

```python
query_pos = past_len + torch.arange(seq, device=x.device)
key_pos = torch.arange(k.shape[-2], device=x.device)
allowed = key_pos[None, :] <= query_pos[:, None]
scores = scores.masked_fill(
    ~allowed[None, None, :, :], float("-inf")
)
```

`allowed` 的形状是 `[当前 Query 长度, 总 Key 长度]`。扩展前面的两个维度后，Mask 可以广播到所有 Batch 和 Query 头。

单 Token Decode 是同一规则的特例：如果缓存中已经包含历史和当前 Token，当前 Query 应能看到所有有效 Key。错误使用 1 × T 的左上对齐 Mask，可能只允许它看到第一个 Key。

调用库接口也需要确认对齐语义。PyTorch SDPA 的 `is_causal=True` 对非方阵采用左上对齐；带历史缓存时，不能直接照搬完整 Prefill 的调用，必要时应传入显式偏移 Mask。文末链接到具体版本的 API 文档。

### KV Cache 保留两个头，而不是八个头

缓存保存的是投影后的真实 K/V，形状为 `[B, Hkv, 历史长度, head_dim]`。在这个例子里，前 3 个 Token 的缓存是 `[1, 2, 3, 4]`。追加两个 Token 后，变成 `[1, 2, 5, 4]`，头数始终为 2。

因此，要先拼接并记录紧凑 KV，再为教学计算做显式重复：

```python
k = torch.cat((past_k, k), dim=-2)
v = torch.cat((past_v, v), dim=-2)
present = (k, v)  # 返回给下一轮的缓存只有真实 KV 头。
k = k.repeat_interleave(self.group_size, dim=1)
v = v.repeat_interleave(self.group_size, dim=1)
```

这段是有历史缓存时的关键顺序；没有历史时直接记录当前 K/V。完整代码还检查缓存的形状、dtype 和设备是否匹配。

假设 Batch 为 B、缓存长度为 T、KV 头数为 $`H_{kv}`$、每头维度为 $`d_h`$，K/V 的 dtype 相同，每个元素占 b 字节，则单层理想缓存大小为：

```math
\mathrm{KVBytes}=2BTH_{kv}d_hb.
```

最前面的 2 表示 K 和 V 两份数据。固定其他条件时，8 个 Query 头、2 个 KV 头的 GQA 缓存，是对应 MHA 的 1/4。这个比例没有计入分页元数据、分配粒度、量化元信息和临时 Buffer。

高效 Backend 可以直接索引共享 KV，结合工作划分复用数据，不必先物化八个 KV 头。教学代码的显式 repeat 会产生临时数据，不能把它的存储和速度当作 GQA 的固有限制。

同时，Query 头仍然有 8 个，每个头还要生成自己的注意力输出。**缓存缩小四倍，不代表 Attention FLOPs 或端到端延迟缩小四倍。** 实际收益还取决于序列长度、Batch、Shape、数据精度和 Kernel 实现。

### 把 Prefill 和 Decode 连起来检查

完整可运行类在 [gqa.py](../../../labs/attention/gqa.py)。它默认启用因果 Mask；设置 `use_cache=True` 时，返回输出与紧凑 KV Cache。下面从仓库根目录运行：

```python
import sys
import torch

sys.path.insert(0, "labs/attention")
from gqa import GQA

model = GQA(32, num_q_heads=8, num_kv_heads=2, head_dim=4).eval()
x = torch.randn(1, 5, 32)
with torch.no_grad():
    full = model(x)
    prefix, cache = model(x[:, :3], use_cache=True)
    suffix, cache = model(x[:, 3:], past_key_value=cache, use_cache=True)
    torch.testing.assert_close(suffix, full[:, 3:])
    assert cache[0].shape == (1, 2, 5, 4)
```

同一个模型先整段计算，再用“前缀 Prefill + 后缀缓存输入”计算。如果缓存和 Mask 都正确，后缀输出应该在浮点容差内一致。将后缀拆成逐 Token 输入，也应满足同一性质。

本实现也支持 `model(x, is_causal=False)`，用于无缓存的双向计算；它的可见范围不同，不能拿来作为上述因果结果的对照。

从最初的无 Mask 版本完善到这里，需要补齐类初始化、输入参数检查、因果 Mask 和历史缓存。原始草稿漏掉了 `super().__init__()`，在创建 Linear 子模块前必须调用它，否则类无法正常实例化。

配套测试已经在 macOS、Python 3.9.6、PyTorch 2.8.0、CPU float64 下通过。它们验证了不同头配置下与逐头 SDPA 的输出一致、因果前缀不受未来 Token 影响、逐 Token / 分块缓存结果一致，以及输入和参数梯度一致；还检查了非法头数。运行方法见[实验记录](../../../labs/attention/README.md)。

<details>
<summary>符号与完整 Shape 速查</summary>

B 表示 Batch，S 表示当前输入长度，T 表示拼接历史后的总 Key 长度，D 表示输入隐藏维度，Hq / Hkv 表示 Query / KV 头数，dh 表示每头维度。无历史缓存时 T = S。

| 步骤 | Shape |
| --- | --- |
| 输入 | `[B, S, D]` |
| Q 投影 | `[B, S, Hq * dh]` |
| K / V 投影 | `[B, S, Hkv * dh]` |
| 分头后的 Q | `[B, Hq, S, dh]` |
| 当前 K / V | `[B, Hkv, S, dh]` |
| 拼接历史后的紧凑 K / V | `[B, Hkv, T, dh]` |
| 教学重复后的 K / V | `[B, Hq, T, dh]` |
| Attention 分数与概率 | `[B, Hq, S, T]` |
| 加权 V 后的输出 | `[B, Hq, S, dh]` |
| 合并头后的输出 | `[B, S, Hq * dh]` |
| 输出投影 | `[B, S, D]` |

无缓存的完整序列计算中，分数矩阵的两个序列维相等；缓存推理中，Query 长度与 Key 长度通常不同。

</details>

## 面试与自测

这些是围绕本专题设计的练习题。先用自己的话解释，再回到正文推导和代码检查。

**GQA 主要减少什么？**

减少 KV 投影输出、历史 KV 存储与相关流量；各 Query 头仍然计算自己的 Attention。追问：为什么 KV 缓存缩小四倍，推理速度不一定提升四倍？

**为什么教学版本使用 `repeat_interleave`？**

为了表达连续组映射，让同一 KV 头连续对应多个 Query 头。追问：如果用 `repeat` 改变复制顺序，Q 与 KV 的对应关系是否仍然相同？

**缓存应该在复制 KV 之前还是之后保存？**

之前。缓存应保留真实 KV 头数，复制后的 Tensor 只是本轮教学计算的临时输入。追问：生产 Backend 怎样避免显式复制？

**为什么转置后要考虑 `contiguous()`？**

转置通常改变 stride，目标 `view` 不一定合法；这里先整理布局，再合并头维。追问：换成 `reshape` 是否保证不发生复制？

**为什么 Decode 的 Mask 容易写错？**

当前 Query 的局部编号从 0 开始，全局位置却包含历史长度。追问：历史长度为 3、一次追加两个 Token 时，两行 Mask 分别允许哪些 Key？

**这份实现距离完整 LLM Attention 还缺什么？**

位置编码、变长请求、Padding、生产缓存与高性能 Backend。追问：为什么加入 RoPE 后，还要验证完整计算与缓存计算的位置一致性？

## 参考与待验证事项

原理参考 [GQA 论文](https://aclanthology.org/2023.emnlp-main.298/)。Mask 对齐与接口行为参考 [PyTorch 2.8 SDPA 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html)，与本机验证版本对应；其他版本应重新确认。

本文的实现是统一长度、无 Padding 的教学参考。缓存通过 `torch.cat` 追加，会复制历史数据；生产实现通常使用预分配或分页缓存。这里没有 RoPE、Dropout、滑动窗口，也没有处理不同请求的有效长度和空可见行。

后续实验将分别补充位置编码、变长 Mask、缓存管理，以及原生 GQA Backend 与显式 repeat 的性能比较。CPU 正确性检查不能证明 GPU 速度，也不能代替 FP16/BF16 数值误差或真实模型验证。

[返回 Attention 导航](README.md) · [继续阅读：MHA 各头独立性](mha-head-independence.md)
