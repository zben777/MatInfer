# Kernel 编程与工程

> 状态：内容草案。已整理原稿的技术主题、公式、流程、图布局和题目；完整教程、源码版本与实测结果将随学习补齐。示意代码不等同于已验证实现。

**核心问题：怎样把计算映射成线程、内存、指令与同步的代码？**

覆盖 CUDA、Triton、CUTLASS、CuTe、线程映射、Warp 原语、同步、访存、Layout、MMA、TMA、异步流水线、调参、正确性和 Benchmark。

建议学习顺序：先理解执行映射与正确性，再做访存和计算优化；用简单算子建立习惯，之后深入 Tensor Core 与流水线。

[返回总导航](../../导航.md) · [学习路线](../../roadmap/README.md) · [面试题索引](../../interview/README.md)

## 内容索引

- [写一个 Kernel 到底经历什么](#section-01)
- [CUDA 执行层级](#section-02)
- [CUDA C++：最底层控制](#section-03)
- [Thread / Warp / Block 怎么选择](#section-04)
- [Warp Primitive](#section-05)
- [`__syncthreads()` 到底同步什么](#section-06)
- [Global Memory Access：向量化和合并访存](#section-07)
- [Alignment](#section-08)
- [Shared Memory：不是“缓存”，而是 Scratchpad](#section-09)
- [Bank Conflict](#section-10)
- [Register：性能神器也是资源瓶颈](#section-11)
- [Triton：换了一种编程模型](#section-12)
- [CUDA vs Triton 的关键差别](#section-13)
- [Triton Layout 为什么重要](#section-14)
- [CUTLASS：高性能 GEMM 工程模板](#section-15)
- [CuTe：真正需要理解的是 Layout Algebra](#section-16)
- [CUTLASS / CuTe 的关系](#section-17)
- [Tensor Core：WMMA / MMA / CuTe 的区别](#section-18)
- [TMA](#section-19)
- [TMA vs `cp.async`](#section-20)
- [Producer–Consumer](#section-21)
- [Barrier 为什么越来越复杂](#section-22)
- [Double Buffer / Multi-Stage Pipeline](#section-23)
- [Launch Configuration](#section-24)
- [Auto-Tuning](#section-25)
- [Compile-time Specialization](#section-26)
- [Kernel Debugging](#section-27)
- [Benchmark 方法](#section-28)
- [核心选择：什么时候用谁？](#section-29)
- [面试与自测问题](#section-30)
- [知识图布局草案](#section-31)

## 适用条件与补充

不能依据隐式 Warp 锁步保证正确性。shuffle 的 mask 必须匹配参与线程；下面使用全掩码的归约示例假定完整 Warp 参与，且只有相应 lane 的结果有效。块级 barrier 要由需要参与的线程一致到达。合并访存关注同一 Warp 内各线程的地址；向量化关注单线程访存宽度。float4 / uint4 / int4 是向量类型，其中 int4 并不表示 INT4 量化格式。TMA、WGMMA 等能力需要按 GPU 架构和工具链版本判断。

详见[技术口径与待核对事项](../00-overview/technical-notes.md)。

### CUDA / Triton / CUTLASS / CuTe 到底分别解决什么问题？

```text
上一层 Backend：

FlashAttention
FlashMLA
DeepGEMM
TurboQuant Kernel
        ↓

回答：
“我要实现一个高性能 Backend”

────────────────────────────

这一层 Kernel Programming：

CUDA
Triton
CUTLASS
CuTe
        ↓

回答：
“我到底怎样把这个 Backend 写出来？”
```

所以本层知识图的核心不是某个具体算法，而是：

> **把一个算子设计真正映射成 thread / warp / CTA / memory / MMA / pipeline 的代码。**

---

<a id="section-01"></a>

## 写一个 Kernel 到底经历什么

这一张左边不要放 Transformer 流程，而放 **Kernel 开发流程**。

```text
Operator / Backend Requirement
        ↓
确定 Tensor Shape
M / N / K
Head Dim
Context Length
Batch
        ↓
确定数据 Layout
Row-major / Col-major
Packed / Interleaved
        ↓
划分 Tile
        ↓
Grid / CTA Mapping
        ↓
Warp Mapping
        ↓
Thread Mapping
        ↓
Global Memory Access
        ↓
Shared Memory / Register
        ↓
Compute
CUDA Core / Tensor Core
        ↓
Synchronization
        ↓
Store Output
        ↓
Benchmark
        ↓
Profile
        ↓
继续优化
```

> **从数学计算到实际 CUDA launch 的完整映射过程。**

---

<a id="section-02"></a>

## CUDA 执行层级

```text
Grid
│
├── Block / CTA 0
│      │
│      ├── Warp 0
│      │     ├── Thread 0
│      │     ├── Thread 1
│      │     └── ...
│      │
│      ├── Warp 1
│      └── ...
│
├── Block / CTA 1
└── ...
```

然后右边给一个最核心的映射问题：

```text
Problem Size
      ↓
Grid

Tile
      ↓
CTA

Sub-Tile
      ↓
Warp

Element / Vector
      ↓
Thread
```

一定要加一句：

> **Kernel 设计最核心的问题之一，就是“谁负责哪一块数据”。**

+ 一个 block 为什么 128 threads？
+ 一个 warp 处理几个 token？
+ 一个 thread 读几个元素？
+ 一个 CTA 负责几个 query head？
+ 一个 warp 负责一个 expert tile 还是多个？

本质都是 Mapping。

---

<a id="section-03"></a>

## CUDA C++：最底层控制

CUDA 这一块不要只写“灵活”。

要明确它能控制什么：

```text
CUDA C++
│
├── Grid / Block
├── Warp / Lane
├── Shared Memory
├── Register
├── Vector Load
├── Warp Shuffle
├── Barrier
├── Tensor Core
├── Async Copy
└── Stream / Graph
```

它最大的特点：

> **执行和内存行为几乎都可以手动控制。**

适合：

```text
特殊 Layout
特殊数据类型
Bit Packing
复杂控制流
极致性能
硬件特性
```

```text
INT4 Packed KV
   ↓
uint32 Load
   ↓
Bit Unpack
   ↓
Centroid / Scale
   ↓
Dequant
   ↓
Attention
```

这种东西 CUDA 往往比纯高层 DSL 更容易精细控制。

---

<a id="section-04"></a>

## Thread / Warp / Block 怎么选择

这一块很适合做成一张小判断图。

```text
工作量小
   ↓
Warp-level

需要多个 Warp 协作
   ↓
CTA-level

需要跨 CTA
   ↓
通常需要：
第二 Kernel
Atomic
Cooperative mechanism
```

例如 Reduction：

```text
32 elements
   ↓
1 warp

128 elements
   ↓
4 warps
   ↓
warp reduce
   ↓
shared
   ↓
final warp reduce
```

```text
__shfl_down_sync
```

```text
__syncthreads()
```

---

<a id="section-05"></a>

## Warp Primitive

这一块单独一个框：

```text
Warp Primitive
│
├── shuffle
├── ballot
├── popc
├── match
└── warp vote
```

为什么重要：

> **Warp 使用 SIMT 执行模型，可用 shuffle 等原语交换数据；现代 GPU 的独立线程调度意味着不能依赖隐式锁步同步。**

典型 Reduction：

```cpp
for (int offset = 16; offset > 0; offset >>= 1)
    x += __shfl_down_sync(0xffffffff, x, offset);
```

```text
优点：

不需要 Shared Memory
不需要 block-level barrier
低延迟
```

但要注意：

> **只解决 Warp 内通信。**

跨 Warp 依然通常需要 Shared Memory 或其它机制。

---

<a id="section-06"></a>

## `__syncthreads()` 到底同步什么

这在面试里非常容易被问。

```text
Block
│
├── Warp 0 ────┐
├── Warp 1 ────┤
├── Warp 2 ────┤ → __syncthreads()
└── Warp 3 ────┘
```

所有线程到达 barrier 后才继续。

它的两个重要作用：

```text
Control Synchronization
+
Shared Memory Visibility
```

典型：

```text
Global → Shared
       ↓
__syncthreads()
       ↓
Shared → Compute
```

一定标：

> `__syncthreads()` 是 block scope，不是整个 grid。

---

<a id="section-07"></a>

## Global Memory Access：向量化和合并访存

这一块必须把两个概念分开。

### Coalescing
关注：

> 一个 warp 的地址是否能合并成较少 Memory Transactions。

```text
lane 0 → A[0]
lane 1 → A[1]
lane 2 → A[2]
...
lane31 → A[31]
```

连续。

---

### Vectorized Load
关注：

> 一个线程单条 load 指令一次读取更多数据。

```cpp
float4 x = *reinterpret_cast<const float4*>(ptr);
```

或者：

```cpp
uint4
int4
half2
```

```text
Coalescing
=
同一 warp 内各线程的访问模式

Vectorization
=
单个 thread 的 load width
```

这是很好的面试区分。

---

<a id="section-08"></a>

## Alignment

Vector Load 一定要接：

```text
地址对齐
```

例如 128-bit load：

```text
ptr % 16 == 0
```

否则：

+ 可能不能生成预期向量指令
+ 可能拆 transaction
+ 甚至产生未对齐访问问题

以 packed KV 为例：

```text
1 Warp
32 Threads

每 Thread 读 4B
        ↓
32 × 4B = 128B
```

如果地址连续且对齐：

> 很自然对应一个连续 memory segment。

---

<a id="section-09"></a>

## Shared Memory：不是“缓存”，而是 Scratchpad

这一张应该强调一个概念：

> Shared Memory 是程序员显式管理的片上存储，不是普通硬件 Cache。

主要用途：

```text
Global Memory
     ↓
Shared Memory
     ↓
数据重排
数据复用
Tile Staging
跨 Thread 共享
Pipeline
     ↓
Register
```

典型：

```text
GEMM:
A Tile + B Tile

Transpose:
Row Load → Shared → Column Store

Attention:
Q/K/V Tile staging
```

---

<a id="section-10"></a>

## Bank Conflict

虽然 GPU 层还会再讲，但 Kernel 编程这里要从“程序行为”角度讲一次。

```text
32 banks

bank =
(address / word_size) % 32
```

如果：

```text
lane 0 → bank 0
lane 1 → bank 0
lane 2 → bank 0
...
```

访问不同地址：

```text
Bank Conflict
```

Transpose：

```cpp
__shared__ float tile[32][32];
```

列访问 stride=32：

```text
全部映射相同 bank
```

变成：

```cpp
__shared__ float tile[32][33];
```

stride=33 后打散。

这一张里不用深入电路，只讲：

> **Layout 会直接决定 Shared Memory 的性能。**

---

<a id="section-11"></a>

## Register：性能神器也是资源瓶颈

Register 这块建议画成：

```text
Thread
│
├── local accumulators
├── temporary values
├── MMA fragment
├── online softmax m/l
└── unpacked data
```

优点：

```text
最快
thread-private
```

但是：

```text
register / thread ↑
       ↓
一个 SM 能驻留的 warps ↓
       ↓
Occupancy ↓
```

如果太高甚至会：

```text
Register Spill
      ↓
Local Memory
      ↓
实际落到显存层级
```

> **Register 不是越多越好。**

---

<a id="section-12"></a>

## Triton：换了一种编程模型

Triton 这块不要解释成：

> “比 CUDA 简单。”

> **CUDA 以 Thread 为中心；Triton 更偏 Program / Tensor Block 为中心。**

CUDA：

```text
threadIdx.x
blockIdx.x
lane
warp
```

Triton：

```text
一个 Program Instance
        ↓
处理一块 Tensor
        ↓
tl.arange
tl.load
tl.store
tl.dot
```

```python
offs = pid * BLOCK + tl.arange(0, BLOCK)
x = tl.load(ptr + offs)
```

主要关注：

```text
这个 Program 处理哪一块数据？
```

而不是每个 CUDA thread 的每条指令。

---

<a id="section-13"></a>

## CUDA vs Triton 的关键差别

做成一张横向对比：

|  | CUDA | Triton |
| --- | --- | --- |
| 抽象 | Thread / Warp / CTA | Program / Tensor Block |
| Memory | 手动精细控制 | 编译器负责较多映射 |
| Layout | 手动 | DSL / Compiler Layout |
| Tensor Core | WMMA / MMA / CuTe | `tl.dot` 等 |
| 开发效率 | 较低 | 较高 |
| 特殊 bit 操作 | 强 | 视场景 |
| 极致控制 | 强 | 相对少 |
| Autotune | 自己实现较多 | 内置机制方便 |

> CUDA 一定比 Triton 快。

> **复杂特殊 Layout / Bit-level / Hardware-specific 场景 CUDA 更灵活；标准 Tensor 算子 Triton 往往能更快达到较高性能。**

---

<a id="section-14"></a>

## Triton Layout 为什么重要

在 Triton 里：

```text
Logical Tensor
       ↓
Layout
       ↓
lane / warp / CTA
```

Layout 决定：

```text
一个 tensor element
最终由哪个 lane 负责
```

所以它影响：

+ memory coalescing
+ shared memory
+ tensor core mapping
+ warp distribution

一句话：

> **Triton Layout 是逻辑 Tensor 和 GPU 执行线程之间的映射规则。**

---

<a id="section-15"></a>

## CUTLASS：高性能 GEMM 工程模板

CUTLASS 这块不要和 CUDA/Triton完全平级地理解。

它更像：

```text
CUDA
  ↓
高性能 GEMM / Conv 模板库
  ↓
CUTLASS
```

核心层级：

```text
Device
 ↓
Kernel
 ↓
Collective
 ↓
Tiled MMA
 ↓
Atom
```

或者从矩阵：

```text
Problem Shape
 ↓
CTA Tile
 ↓
Warpgroup Tile
 ↓
MMA Tile
```

CUTLASS 封装的机制包括：

+ MMA
+ async copy
+ epilogue
+ tile scheduling
+ pipeline

封装起来。

---

<a id="section-16"></a>

## CuTe：真正需要理解的是 Layout Algebra

CuTe 这一块很容易学成 API 记忆。

但真正核心：

```text
Tensor = Pointer + Layout
```

Layout 表达：

```text
Logical Coordinate
        ↓
Linear Memory Address
```

```text
(M, N)
 ↓
stride
 ↓
offset
```

更重要的是：

> **同一个逻辑 Tensor，可以通过不同 Layout 映射到 global/shared/register/thread。**

这也是为什么 CuTe 非常适合表达：

+ shared memory layout
+ tensor core fragment layout
+ warpgroup mapping
+ TMA copy

---

<a id="section-17"></a>

## CUTLASS / CuTe 的关系

这一块可以画：

```text
CuTe
=
底层 Tensor + Layout + Copy + MMA 抽象

          ↓

CUTLASS
=
用 CuTe 等抽象搭建
高性能 GEMM / Conv / Kernel
```

所以学习顺序不要反过来。

CuTe Layout Algebra 可以结合实例逐步学习。

可以：

```text
CUDA 基础
   ↓
GEMM 基础
   ↓
CUTLASS Kernel
   ↓
遇到 Layout 问题
   ↓
深入 CuTe
```

---

<a id="section-18"></a>

## Tensor Core：WMMA / MMA / CuTe 的区别

这一块很适合面试。

高层：

```text
WMMA
```

较高层封装，fragment API。

更底层：

```text
mma.sync
```

warp-level MMA instruction。

新架构还有：

```text
WGMMA
```

warpgroup-level MMA。

CuTe：

```text
TiledMMA
```

用 layout / atom 抽象组织 MMA。

```text
Math:
D = A × B + C
```

↓

```text
Hardware Instruction:
MMA / WGMMA
```

↓

```text
Programming Abstraction:
WMMA / CUTLASS / CuTe / Triton tl.dot
```

---

<a id="section-19"></a>

## TMA

传统：

```text
Threads
  ↓
逐个发 load
  ↓
Global → Shared
```

TMA：

```text
Tensor Descriptor
        ↓
TMA Engine
        ↓
多维 Tensor Tile
        ↓
Shared Memory
```

特点：

```text
异步
多维
硬件地址生成
大 Tile 搬运
```

程序：

```text
Producer
   ↓
启动 TMA
   ↓
数据后台搬运

Consumer
   ↓
等待数据 ready
   ↓
MMA Compute
```

核心：

> **减少大量线程显式参与地址计算和搬运，让搬运与计算更容易流水化。**

---

<a id="section-20"></a>

## TMA vs `cp.async`

这个对比非常适合一个小框：

```text
cp.async

thread-level
Global → Shared
每个 thread 指定地址
```

```text
TMA

tensor-level
Global → Shared
Descriptor 描述多维 tile
```

大致演进：

```text
普通 ld/st
   ↓
cp.async
   ↓
TMA
```

不是简单“新版一定替代旧版”，而是抽象和硬件能力不同。

---

<a id="section-21"></a>

## Producer–Consumer

这一块正好把 TMA、MMA、barrier 串起来。

```text
Producer Warpgroup
        │
        │ TMA
        ↓
Shared Buffer 0
Shared Buffer 1
        │
        ↓
Consumer Warpgroup
        │
        ↓
WGMMA / MMA
```

时间轴：

```text
Time →

Producer:
Load0  Load1  Load2  Load3

Consumer:
       Compute0 Compute1 Compute2
```

即：

> **Load N+1 与 Compute N 重叠。**

这就是 pipeline。

---

<a id="section-22"></a>

## Barrier 为什么越来越复杂

早期：

```cpp
__syncthreads()
```

整个 CTA 同步。

但 Producer–Consumer 模型下不希望：

> 所有人每次都停下来。

所以现代 Kernel 会大量使用：

+ warp-level sync
+ named barrier / mbarrier
+ async transaction barrier
+ producer-consumer stage state

目标：

> **只同步真正有依赖的线程/阶段。**

这就是项目里“barrier 消减”真正的意义，而不只是：

> 少写一个 `__syncthreads()`。

---

<a id="section-23"></a>

## Double Buffer / Multi-Stage Pipeline

经典双缓冲：

```text
Buffer 0
Buffer 1
```

过程：

```text
Load Tile 1 → Buffer 1
同时
Compute Tile 0 → Buffer 0
```

然后交换。

进一步：

```text
Stage 0
Stage 1
Stage 2
Stage 3
```

更多 stage：

```text
Latency Hiding ↑
```

但是：

```text
Shared Memory Usage ↑
```

所以又是一种 trade-off。

---

<a id="section-24"></a>

## Launch Configuration

> 为什么这个 Kernel 使用 128 threads？

> 试出来的。

应该画：

```text
Block Threads
     ↓
Warps / CTA
     ↓
Parallelism
+
Reduction Structure
+
Register Usage
+
Shared Memory
+
Occupancy
```

```text
128 threads
=
4 warps
```

需要解释：

```text
每个 warp 负责什么？
4 warp 怎么协作？
最终怎么归约？
```

Launch 参数应该来自 Mapping，不是拍脑袋。

---

<a id="section-25"></a>

## Auto-Tuning

Triton / CUTLASS 这类环境常会自动或半自动搜索：

```text
BLOCK_M
BLOCK_N
BLOCK_K
num_warps
num_stages
```

不同 Shape 最优参数不同。

```text
M=1
```

```text
M=4096
```

不应该强行用同一配置。

> **高性能 Kernel 常常是 Shape-specialized。**

---

<a id="section-26"></a>

## Compile-time Specialization

```cpp
template<int HEAD_DIM>
```

或：

```python
BLOCK_SIZE: tl.constexpr
```

好处：

+ loop unroll
+ constant folding
+ dead code elimination
+ 固定 shape 优化

所以很多高性能库：

```text
不是一个万能 Kernel
```

```text
大量特化 Kernel
+
Runtime Dispatch
```

---

<a id="section-27"></a>

## Kernel Debugging

```text
正确性
↓
性能
```

一定先正确。

常见错误：

```text
Race Condition
Out-of-Bounds
Misaligned Access
Missing Synchronization
Wrong Layout
Precision Error
```

工具：

```text
compute-sanitizer
cuda-gdb
printf
unit tests
PyTorch reference
```

精度检查：

$ \text{max abs error} $

$ \text{relative error} $

然后才 Benchmark。

---

<a id="section-28"></a>

## Benchmark 方法

必须避免：

```text
运行一次
↓
看时间
```

正确：

```text
Warmup
 ↓
多次迭代
 ↓
CUDA Event
 ↓
统计 median / percentile
```

并固定：

+ Shape
+ dtype
+ batch
+ context
+ GPU
+ clock / environment
+ baseline

否则性能数字没有意义。

---

<a id="section-29"></a>

## 核心选择：什么时候用谁？

用一张最实用的决策图：

```text
我要实现一个新算子
        ↓

是不是标准 GEMM / MMA 为主？
   │
   ├─ Yes → CUTLASS / CuTe / Triton
   │
   └─ No
        ↓
是否主要是标准 Tensor 运算？
   │
   ├─ Yes → Triton
   │
   └─ No
        ↓
是否包含复杂 Bit 操作 /
特殊 Layout /
精细同步 /
硬件专属能力？
        ↓
      CUDA
```

但旁边标：

> **这不是硬规则，实际经常混用。**

```text
Framework
  ↓
Triton preprocessing
  ↓
CUDA core kernel
  ↓
CUTLASS GEMM
```

完全可能。

---

<a id="section-30"></a>

## 面试与自测问题

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
12. `cp.async` 和 TMA 有什么区别？
13. Producer–Consumer Pipeline 为什么能加速？
14. 为什么同一个 Kernel 对不同 Shape 最优配置不同？

---

<a id="section-31"></a>

## 知识图布局草案

最终图是：

```text
┌───────────────────────────────────────────────────────────────┐
│ Kernel 编程与工程：CUDA / Triton / CUTLASS / CuTe            │
├──────────────┬──────────────────────────┬─────────────────────┤
│              │                          │ CUDA Execution      │
│ Kernel       │ Grid → CTA → Warp        │ Thread / Warp       │
│ Development  │ → Thread                 ├─────────────────────┤
│ Flow         │                          │ Memory Access       │
│              │                          │ Vector / Alignment  │
│ Shape        ├──────────────────────────┼─────────────────────┤
│ ↓            │ CUDA vs Triton           │ Shared / Register   │
│ Layout       │                          │ Bank Conflict       │
│ ↓            │ Program Model            │ Occupancy           │
│ Tile         ├──────────────────────────┼─────────────────────┤
│ ↓            │ CUTLASS / CuTe           │ Tensor Core         │
│ Mapping      │ Layout / MMA             │ MMA / WGMMA         │
│ ↓            │                          │                     │
│ Kernel       ├──────────────────────────┼─────────────────────┤
│              │ TMA / cp.async           │ Producer–Consumer   │
│              │                          │ Pipeline / Barrier  │
├──────────────┴──────────────────────────┴─────────────────────┤
│ Compile Specialization / AutoTune / Benchmark / Debug        │
├───────────────────────────────────────────────────────────────┤
│ 面试高频：Mapping · Layout · Sync · Memory · Tensor Core     │
└───────────────────────────────────────────────────────────────┘
```

```text
DeepSeek
```

一路钻到了：

```text
一个 thread 到底读什么数据
```

整个路径现在是：

```text
① 模型层
MLA / MoE
        ↓

② 推理框架层
Scheduler / KV Cache
        ↓

③ 算子层
Attention / GEMM / Softmax
        ↓

④ Backend 层
FlashMLA / DeepGEMM
        ↓

⑤ Kernel 编程层
CUDA / Triton / CuTe
        ↓
```
