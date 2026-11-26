# DeepSpec：投机解码草稿模型

> 仓库：[deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec)。

DeepSpec 是投机解码草稿模型的训练与评估全栈代码库：数据准备（下载 prompt、重生成目标回答、构建 target cache）、草稿模型训练、在 benchmark 上测量投机解码接受率，三阶段顺序串联，每阶段输出喂给下一阶段。

| 专题 | 核心问题 |
| --- | --- |
| [本层正文：推理框架与服务系统](../README.md) | 投机解码在 serving 侧怎样接入？ |
| [Backend 映射总表](../../04-backends/README.md#section-22) | 接受率与 kernel 层的 decode 代价如何权衡？ |

状态：仓库定位已核对，源码解读待补。主题偏模型训练与采样，此处保留与推理 Backend 的衔接入口。

[返回 Serving 层](../README.md)
