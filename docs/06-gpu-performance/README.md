# GPU 硬件与性能分析

> 状态：内容草案。已整理原稿的技术主题、公式、流程、图布局和题目；完整教程、源码版本与实测结果将随学习补齐。示意代码不等同于已验证实现。

**核心问题：Kernel 为什么快或慢，怎样用证据定位瓶颈？**

覆盖 SM、Warp Scheduler、Register、Shared / L1 / L2 / HBM、Bank、Transaction、Tensor Core、Occupancy、Roofline、MFU / MBU、ILP / TLP / MLP、Nsight 与优化闭环。

建议学习顺序：先理解执行和存储层次，再用系统时间线定位热点，用 Kernel 指标检验瓶颈假设。

[返回总导航](../../导航.md) · [学习路线](../../roadmap/README.md) · [面试题索引](../../interview/README.md)

## 内容索引

- [一条 GPU 指令到底经历什么](#section-01)
- [一个 SM 里面有什么](#section-02)
- [Warp Scheduler：为什么 GPU 能隐藏延迟](#section-03)
- [Eligible Warp / Active Warp / Stalled Warp](#section-04)
- [Occupancy](#section-05)
- [Register File](#section-06)
- [Shared Memory / L1](#section-07)
- [Shared Memory Bank](#section-08)
- [Cache Hierarchy](#section-09)
- [L2 Cache](#section-10)
- [Cache Line 和 Memory Transaction](#section-11)
- [Coalescing 的硬件本质](#section-12)
- [HBM](#section-13)
- [Tensor Core](#section-14)
- [为什么 Small-M GEMM 难吃满 Tensor Core](#section-15)
- [Roofline：整张图最重要的性能模型](#section-16)
- [把 LLM 算子放到 Roofline 上](#section-17)
- [MFU / MBU](#section-18)
- [Latency vs Bandwidth](#section-19)
- [Memory-Level Parallelism](#section-20)
- [ILP / TLP / MLP](#section-21)
- [Warp Divergence](#section-22)
- [Pipeline Stall](#section-23)
- [Nsight Compute](#section-24)
- [Nsight Systems vs Nsight Compute](#section-25)
- [性能分析完整方法论](#section-26)
- [怎么判断优化有效](#section-27)
- [怎么判断“优化到顶”](#section-28)
- [GPU 内部和多 GPU 的连接](#section-29)
- [GPU 硬件层的“因果链”](#section-30)
- [面试与自测问题](#section-31)
- [知识图布局草案](#section-32)

## 适用条件与补充

Register 不属于从 HBM 经各级缓存逐级加载的硬件缓存；Shared 是程序显式管理的存储，L1 / L2 是缓存，原图中的箭头用于表达数据组织层次。访问同一 Shared 地址的广播与不同地址的 Bank Conflict 要区分。MFU 指模型有效计算吞吐相对匹配的硬件峰值；单 Kernel 更适合记录 achieved TFLOP/s 与计算利用率。不能用 FLOPs 总量直接除以 FLOPs/s。MBU 同样需要明确实际字节与峰值带宽的统计口径。

详见[技术口径与待核对事项](../00-overview/technical-notes.md)。

### 为什么前面的所有优化会有效？
这一张的核心问题是：

> **一个 Kernel 最后到底跑在 GPU 哪些硬件上？为什么会慢？怎么判断瓶颈？**

本层知识图不是再讲“怎么写 CUDA”，而是解释：

```text
CUDA / Triton / CuTe
        ↓
到底映射到了什么硬件？
        ↓
SM / Warp Scheduler / Tensor Core
Register / Shared / L1 / L2 / HBM
        ↓
为什么出现 latency / bandwidth / occupancy / stall
        ↓
怎么 Profile
        ↓
怎么优化
```

所以这一张其实是前五张的“物理解释层”。

---

<a id="section-01"></a>

## 一条 GPU 指令到底经历什么

左边画成：

```text
Kernel Launch
    ↓
Grid
    ↓
CTA / Block
    ↓
SM
    ↓
Warp Scheduler
    ↓
Warp
    ↓
Instruction Issue
    ↓
CUDA Core / Tensor Core / LDST Unit
    ↓
Register / Shared / Cache / HBM
    ↓
Result
```

> **GPU 性能分析关注可执行 Warp、计算单元与内存系统的利用率，并结合依赖和同步分析。**

---

<a id="section-02"></a>

## 一个 SM 里面有什么

这部分一定要画大。

可以简化成：

```text
┌────────────────────────── SM ───────────────────────────┐
│                                                        │
│ Warp Scheduler                                         │
│      ↓                                                 │
│ Warp 0 / Warp 1 / Warp 2 / ...                         │
│                                                        │
│ ┌──────────────┐   ┌──────────────┐                    │
│ │ CUDA Cores   │   │ Tensor Cores │                    │
│ └──────────────┘   └──────────────┘                    │
│                                                        │
│ Register File                                          │
│                                                        │
│ Shared Memory / L1                                     │
│                                                        │
│ Load / Store Units                                     │
└────────────────────────────────────────────────────────┘
                         ↓
                        L2
                         ↓
                        HBM
```

然后注明：

```text
SM = GPU 真正执行 CTA / Warp 的核心单元
```

---

<a id="section-03"></a>

## Warp Scheduler：为什么 GPU 能隐藏延迟

这是非常重要的面试点。

假设：

```text
Warp 0
load global memory
      ↓
等数据
```

如果 GPU 只能等 Warp 0：

```text
SM Idle
```

但真实情况：

```text
Warp 0：等待 HBM
Warp 1：执行 FMA
Warp 2：执行 Load
Warp 3：执行 Tensor Core
```

Scheduler 会切换到：

```text
Eligible Warp
```

> **GPU 隐藏延迟主要不是靠单个 Warp 更快，而是靠多个 Warp 之间切换。**

这就是 Latency Hiding。

---

<a id="section-04"></a>

## Eligible Warp / Active Warp / Stalled Warp

```text
Active Warp
=
已经驻留在 SM 上
```

```text
Eligible Warp
=
当前可以立即发射指令
```

```text
Stalled Warp
=
因为依赖、内存、同步等暂时不能发射
```

最理想：

```text
每个 Scheduler 周期
总有 Eligible Warp
```

如果：

```text
Active Warp 很多
但 Eligible Warp 很少
```

说明：

> Occupancy 看起来很高，但实际 latency hiding 不一定好。

这也是为什么：

> **Occupancy ≠ Performance。**

---

<a id="section-05"></a>

## Occupancy

公式可以放：

```math
\mathrm{Occupancy}=\frac{W_{\mathrm{active}}}{W_{\mathrm{max}}}.
```

$W_{\mathrm{active}}$ 是每个 SM 的驻留 Warp 数，$W_{\mathrm{max}}$ 是架构允许的最大驻留 Warp 数。该比值可写成百分比；测量时需区分理论 Occupancy 与实际达到的 Occupancy。

影响 Occupancy 的主要资源：

```text
Threads / CTA
Register / Thread
Shared Memory / CTA
Architecture Limit
```

```text
Register ↑
   ↓
Resident CTA ↓
   ↓
Active Warp ↓
   ↓
Occupancy ↓
```

还有：

```text
Shared Memory / CTA ↑
   ↓
一个 SM 能驻留 CTA 数 ↓
```

但一定写：

> **目标不是 100% Occupancy，而是足够的 latency hiding。**

---

<a id="section-06"></a>

## Register File

从硬件角度继续解释。

Register：

```text
Thread-private
片上
低延迟
```

但是 Register File 是有限的。

如果每 thread：

```text
40 regs
```

```text
200 regs
```

能驻留的线程数量完全不同。

还要画：

```text
Register Pressure 过高
        ↓
Occupancy 下降
```

甚至：

```text
Register Spill
        ↓
Local Memory
        ↓
L1 / L2 / HBM
```

这就是为什么 ptxas 输出里的：

```text
Used 40 registers
```

非常有意义。

---

<a id="section-07"></a>

## Shared Memory / L1

这一块要明确：

```text
Shared Memory
=
软件管理 scratchpad
```

```text
L1 Cache
=
硬件管理 cache
```

虽然物理资源上可能共享一些硬件资源，但编程语义不同。

Shared 重点：

```text
Tile staging
Data reuse
Layout transformation
Thread communication
Pipeline buffer
```

L1 重点：

```text
Hardware cache
自动缓存访问
```

---

<a id="section-08"></a>

## Shared Memory Bank

这块和上一张 Kernel 图呼应，但这次从硬件角度解释。

通常可以抽象成：

```text
32 Banks
```

连续 32-bit word：

```text
word 0 → bank 0
word 1 → bank 1
...
word31 → bank31
word32 → bank0
```

```text
bank = word_index % 32
```

一个 Warp：

```text
32 threads
```

如果访问：

```text
32 个不同 bank
```

很好。

如果访问：

```text
同一个 bank 的不同地址
```

就需要序列化。

这叫：

```text
Bank Conflict
```

---

<a id="section-09"></a>

## Cache Hierarchy

知识图一定要有一条清楚的 Memory Hierarchy。

```text
Register
   ↓
Shared / L1
   ↓
L2
   ↓
HBM
```

可以旁边写：

```text
越往上：
容量小
更快
离计算近

越往下：
容量大
更慢
带宽重要
```

不过不要给固定 latency 数字，因为不同 GPU 差异很大。

---

<a id="section-10"></a>

## L2 Cache

> 怎么提高 L2 Cache Hit Rate？

L2 是所有 SM 共享的重要缓存层。

影响 Hit Rate 的东西：

```text
Temporal Locality
重复访问相同数据

Spatial Locality
访问相邻数据

Working Set Size
是否能放进 Cache

Access Order
访问顺序

Data Layout
数据如何排布
```

典型优化：

```text
让相邻 CTA / Warp
访问相邻或相同数据

↓
提高复用
```

例如矩阵 Tile：

```text
多个 CTA 重用 B tile
```

就可能从 L2 获益。

---

<a id="section-11"></a>

## Cache Line 和 Memory Transaction

程序层面：

```text
thread load 4B
```

硬件不会一定只去 HBM 拿 4B。

而是按照 cache line / memory sector / transaction 机制处理。

```text
Warp 访问模式
        ↓
Memory Transactions 数量
        ↓
有效 Bandwidth
```

连续：

```text
lane0 A[0]
lane1 A[1]
...
lane31 A[31]
```

通常能够更高效合并。

随机：

```text
lane0 A[100]
lane1 A[945]
lane2 A[12]
...
```

可能产生很多 transaction。

---

<a id="section-12"></a>

## Coalescing 的硬件本质

本层知识图可以很直观：

### 好
```text
Warp:
T0 T1 T2 T3 ... T31

Memory:
[xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx]
```

访问集中。

### 差
```text
T0 → addr 0
T1 → addr 500
T2 → addr 1000
...
```

会需要更多 memory transaction。

一句话：

> **Coalescing 的目标不是“地址必须连续”，而是让 Warp 的内存请求尽量被少量 transaction 覆盖。**

---

<a id="section-13"></a>

## HBM

HBM 这一块不要只写“大显存”。

它更关键的是：

```text
大容量
+
高带宽
+
高延迟
```

所以对于：

```text
Decode Attention
```

问题往往不是：

> 算得慢。

> **每生成一个 Token，都要读取大量历史 KV。**

于是：

```text
KV bytes
    ↑
Context Length ↑
    ↓
HBM Traffic ↑
```

```text
MLA
GQA
Quantized KV
TurboQuant
```

这些方案都关联 KV 存储与访存。

它们本质都在尝试：

> **减少每个 Decode Token 需要从 HBM 搬的数据。**

---

<a id="section-14"></a>

## Tensor Core

普通 CUDA Core：

```text
scalar / vector FP / INT operations
```

Tensor Core：

```text
Matrix MMA
```

核心形式：

```math
D=AB+C.
```

矩阵维度需匹配：$A\in\mathbb{R}^{M\times K}$、$B\in\mathbb{R}^{K\times N}$，$C$ 与 $D$ 的形状均为 $M\times N$；实际指令还受 Tile、dtype 和 Layout 约束。

不是说 Tensor Core 会自动加速所有乘法。

需要：

```text
合适 dtype
合适 shape
合适 alignment / layout
足够数据
合适 tile
```

否则：

```text
Tensor Core 利用率很低
```

---

<a id="section-15"></a>

## 为什么 Small-M GEMM 难吃满 Tensor Core

这个面试问题特别值得放。

```text
M = 1
K = 4096
N = 4096
```

问题：

```text
M 维度没有足够 Tile
        ↓
可并行 CTA 少
        ↓
SM 数量无法充分利用
```

同时：

```text
每个 Weight 都要从 memory 读
```

于是 Decode Linear 很容易出现：

```text
Bandwidth + parallelism bottleneck
```

这就解释为什么 Batch 增大可能明显改善 Decode GEMM。

---

<a id="section-16"></a>

## Roofline：整张图最重要的性能模型

Roofline 应该占右边较大位置。

横轴：

```math
\mathrm{AI}=\frac{F}{B}.
```

$F$ 是运算次数，$B$ 是所分析存储层级的传输字节数；AI 的单位为 FLOP/byte。使用 HBM Roofline 时，字节统计应对应 HBM。

纵轴：

计算吞吐 $P$，单位为 FLOP/s（常写为 TFLOP/s）。

上限：

```math
P_{\mathrm{roof}}=\min\left(P_{\mathrm{peak}},\ \mathrm{AI}\cdot BW_{\mathrm{peak}}\right).
```

$P_{\mathrm{roof}}$ 是简化 Roofline 模型给出的吞吐上限，$P_{\mathrm{peak}}$ 是匹配数据类型的峰值计算吞吐，$BW_{\mathrm{peak}}$ 是相应存储层级的峰值带宽；实际吞吐通常低于该上限。

图上两区域：

```text
Memory Bound
      /
     /
    /
---/──────── Compute Bound
```

意思：

### AI 低
```text
搬数据很多
算得少
```

性能由：

```text
HBM Bandwidth
```

限制。

### AI 高
```text
计算多
数据复用高
```

性能由：

```text
Compute Peak
```

限制。

---

<a id="section-17"></a>

## 把 LLM 算子放到 Roofline 上

```text
Low AI
│
├─ Elementwise
├─ RMSNorm
├─ RoPE
├─ Decode Attention
│
│
├─ Small-M GEMM
│
└──────────────→ High AI
                Large GEMM
                Prefill GEMM
```

并标：

> **这只是典型位置，Shape / Batch / Fusion / dtype 会改变 AI。**

---

<a id="section-18"></a>

## MFU / MBU

右边做一个性能指标框。

### MFU
```math
\mathrm{MFU}=\frac{F_{\mathrm{model}}/t}{P_{\mathrm{peak}}}.
```

$F_{\mathrm{model}}$ 是选定范围内的模型有效运算次数，$t$ 是对应时间；多 GPU 时 $P_{\mathrm{peak}}$ 应使用对应设备的合计峰值，并声明 FLOPs 的计数口径。MFU 与单 Kernel 的硬件计算利用率需要区分。

更适合：

```text
Compute-bound
GEMM
Tensor Core
```

### MBU
```math
\mathrm{MBU}=\frac{B_{\mathrm{HBM}}/t}{BW_{\mathrm{peak}}}.
```

$B_{\mathrm{HBM}}$ 是测得的 HBM 传输字节数，$t$ 是同一统计范围的时间，$BW_{\mathrm{peak}}$ 是峰值 HBM 带宽。算法有效字节数与真实 HBM 流量可能不同。

更适合：

```text
Memory-bound
Decode Attention
RMSNorm
Elementwise
```

如果：

```text
MFU 很低
MBU 也很低
```

那可能不是简单的 compute/memory ceiling，而是：

+ latency
+ insufficient parallelism
+ launch overhead
+ dependency
+ synchronization
+ poor access pattern

---

<a id="section-19"></a>

## Latency vs Bandwidth

这个区别非常重要。

### Bandwidth Bound
有很多数据传输：

```text
大量 concurrent loads
```

最终打满 HBM。

### Latency Bound
数据量未必特别大，但：

```text
等待一次 load
↓
依赖下一步
↓
没有足够其他 warp
```

所以硬件没被打满。

```text
Pointer chasing
Random access
Low occupancy
Small workload
```

都可能偏 latency bound。

---

<a id="section-20"></a>

## Memory-Level Parallelism

这块很值得放。

GPU 想隐藏 HBM latency，不只是需要 Warp 多。

还需要：

```text
同时有足够多 outstanding memory requests
```

也就是 Memory-Level Parallelism。

```text
Load A
等待
Load B
等待
```

差。

如果能：

```text
Load A
Load B
Load C
Load D
```

彼此独立：

```text
MLP ↑
```

更容易隐藏内存延迟。

---

<a id="section-21"></a>

## ILP / TLP / MLP

这一张可以非常漂亮地放一个三角：

```text
               TLP
      Thread-Level Parallelism
              / \
             /   \
            /     \
           /       \
         ILP ----- MLP
Instruction       Memory-Level
Parallelism       Parallelism
```

意义：

### TLP
靠很多 Warp。

### ILP
一个 Warp/Thread 有多个独立指令。

### MLP
同时有多个 outstanding memory requests。

高性能 Kernel 往往是三者的平衡。

---

<a id="section-22"></a>

## Warp Divergence

也需要有一个小框。

```cpp
if (lane < 16)
    A();
else
    B();
```

Warp 内不同线程走不同路径：

```text
路径 A 执行
路径 B mask 掉

然后

路径 B 执行
路径 A mask 掉
```

于是有效并行度下降。

但要写：

> **不是看到 if 就一定严重，关键看 Warp 内线程是否分歧，以及分支工作量。**

---

<a id="section-23"></a>

## Pipeline Stall

右边 Profile 区域可以列常见类型：

```text
Memory Dependency
Long Scoreboard
Short Scoreboard
Barrier
Instruction Dependency
Not Selected
Math Pipeline Throttle
```

不要死背名字。

应该按原因分类：

```text
Memory
Compute dependency
Synchronization
Scheduler
Resource
```

---

<a id="section-24"></a>

## Nsight Compute

不要一上来几十个指标。

先看：

```text
1. Kernel 时间
```

```text
2. Compute / Memory Throughput
```

判断：

```text
更接近 compute 还是 memory ceiling
```

再看：

```text
3. Occupancy / Warps
```

```text
4. Memory
L1 / L2 / DRAM
```

```text
5. Stall Reasons
```

最后：

```text
6. Source / Instruction
```

---

<a id="section-25"></a>

## Nsight Systems vs Nsight Compute

这个一定要区分。

### Nsight Systems
看：

```text
整个程序 Timeline
```

包括：

+ CPU
+ GPU
+ Streams
+ Kernel launch
+ NCCL
+ memcpy
+ synchronization

用来回答：

> **系统时间花在哪？**

---

### Nsight Compute
看：

```text
单个 Kernel 内部
```

包括：

+ SM
+ memory
+ tensor core
+ cache
+ warp stall
+ instruction

用来回答：

> **这个 Kernel 为什么慢？**

---

<a id="section-26"></a>

## 性能分析完整方法论

```text
Baseline
   ↓
确认正确性
   ↓
Measure
   ↓
Profile
   ↓
建立 Performance Model
   ↓
判断瓶颈
   ↓
提出一个优化假设
   ↓
修改一个变量
   ↓
Benchmark
   ↓
对比
   ↓
再次 Profile
```

重点是：

> **先提出瓶颈假设，再优化，不要随机改代码。**

---

<a id="section-27"></a>

## 怎么判断优化有效

不能只看：

```text
Kernel ms ↓
```

还要看：

```text
性能提升来源是不是符合预期？
```

例如假设：

> vector load 提升带宽。

那应该看到：

```text
instruction count ↓
或
effective bandwidth ↑
```

如果：

```text
时间变快 5%
```

但对应指标完全没变化，

那可能：

+ noise
+ clock
+ cache
+ benchmark 不稳定

---

<a id="section-28"></a>

## 怎么判断“优化到顶”

这一块和第四张 Backend 层呼应，但这里从硬件层做最终判断。

```text
Memory-bound Kernel

DRAM Bandwidth
已经接近平台可达到值
        ↓
继续减少 arithmetic
基本没价值
```

或者：

```text
Compute-bound GEMM

Tensor Core utilization
接近高水平
        ↓
MFU 很高
```

再继续：

```text
Kernel time
已经只占 E2E 5%
```

那么即使再优化：

```text
Kernel +20%
```

端到端收益也很有限。

这里可以放 Amdahl：

```math
S_{\mathrm{total}}=\frac{1}{(1-p)+p/s}.
```

$p$ 是优化部分占原始总耗时的比例，$s$ 是该部分自身的加速倍数，$S_{\mathrm{total}}$ 是整体加速倍数。这里假定其余部分耗时不变，不额外引入新的开销。

> 为什么 microbenchmark 提升很大，E2E 几乎没变化。

---

<a id="section-29"></a>

## GPU 内部和多 GPU 的连接

底部可以画：

```text
GPU 0
   │
NVLink / PCIe
   │
GPU 1
```

单卡层重点：

```text
SM / L2 / HBM
```

多卡层重点：

```text
NVLink
PCIe
NIC
RDMA
```

这就是：

```text
Compute / Memory
      ↓
Communication
```

> 分布式通信层

---

<a id="section-30"></a>

## GPU 硬件层的“因果链”

```text
Data Layout
     ↓
Memory Access Pattern
     ↓
Transactions / Cache / Bank
     ↓
Latency & Bandwidth
     ↓
Warp Stall
     ↓
Eligible Warps
     ↓
SM Utilization
     ↓
Kernel Performance
```

这条链非常重要。

因为它把：

```text
代码
```

```text
硬件性能
```

---

<a id="section-31"></a>

## 面试与自测问题

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

---

<a id="section-32"></a>

## 知识图布局草案

这一张这样排：

```text
┌───────────────────────────────────────────────────────────────┐
│ GPU 硬件与性能分析：Kernel 为什么快 / 慢                     │
├──────────────┬──────────────────────────┬─────────────────────┤
│              │                          │ Warp Scheduler      │
│ GPU Execute  │ SM Architecture          │ Eligible Warp       │
│ Flow         │                          │ Latency Hiding      │
│              │ Warp / CUDA Core         ├─────────────────────┤
│ Kernel       │ Tensor Core              │ Register / Shared   │
│ ↓            │ Register                 │ Occupancy           │
│ CTA          │ Shared / L1              │ Bank Conflict       │
│ ↓            │                          │                     │
│ SM           ├──────────────────────────┼─────────────────────┤
│ ↓            │ Memory Hierarchy         │ Roofline            │
│ Warp         │ Reg → L1/Shared          │ AI / MFU / MBU      │
│ ↓            │ → L2 → HBM               │                     │
│ Instruction  │                          │                     │
├──────────────┴──────────────────────────┼─────────────────────┤
│ Cache / Transaction / Coalescing       │ Nsight              │
│ L2 / HBM / Memory Latency              │ Systems / Compute   │
├────────────────────────────────────────┴─────────────────────┤
│ TLP / ILP / MLP → Latency Hiding → Utilization → Performance │
├───────────────────────────────────────────────────────────────┤
│ 性能优化闭环：Baseline → Profile → Bottleneck → Optimize     │
├───────────────────────────────────────────────────────────────┤
│ 高频面试问题                                                 │
└───────────────────────────────────────────────────────────────┘
```

```text
① 模型层
DeepSeek / MLA / MoE

        ↓

② 推理框架层
Scheduler / KV Cache / Batch

        ↓

③ 算子层
Attention / GEMM / Softmax / RMSNorm

        ↓

④ Backend 层
FlashMLA / DeepGEMM / DeepEP

        ↓

⑤ Kernel 编程层
CUDA / Triton / CUTLASS / CuTe

        ↓

⑥ GPU 硬件与性能层
SM / Tensor Core / Memory / Roofline / Nsight
```
