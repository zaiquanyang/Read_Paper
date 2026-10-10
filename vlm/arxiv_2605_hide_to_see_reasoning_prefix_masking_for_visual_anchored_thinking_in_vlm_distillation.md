# Hide to See: Reasoning-prefix Masking for Visual-anchored Thinking in VLM Distillation

## 1. 论文基本信息
- arXiv ID: 2605.11651
- URL: [abs](https://arxiv.org/abs/2605.11651) / [pdf](https://arxiv.org/pdf/2605.11651.pdf)
- 主题归档: `vlm`
- 总结粒度: `detailed`
- 任务类型: think-answer VLM distillation / visual-anchored reasoning
- 核心方法: **Masking-KD**
- 关键词: `VLM distillation`, `think-answer`, `visual forgetting`, `reasoning-prefix masking`, `self-distillation`

## 2. 核心问题与动机

这篇工作的切入点很明确：**把大尺寸 think-answer VLM 的推理能力蒸馏到小模型时，学生模型很容易学会“顺着老师已经写出来的 reasoning prefix 往下接”，而不是持续去看图。**

作者把这个问题概括为两层：

1. **visual forgetting**  
   think-answer VLM 的推理链很长，前面已经生成出来的大量文字本身就越来越“可预测后文”，导致模型逐步弱化对视觉证据的依赖。

2. **textual shortcut learning**  
   在 distillation 时，student 不需要真正学会老师如何看图，只要抓住少量高影响力的 reasoning cues，就能把 reverse KL 做小。于是训练信号会越来越集中在前段 token，后段推理步骤的蒸馏价值变弱。

作者的关键判断不是“再想办法把 teacher 的视觉 pattern 显式传给 student”，而是：

> 更有效的做法，是让 student 在缺失一部分关键 reasoning prefix 的情况下，仍然要去逼近 teacher 分布，从而被迫重新利用图像证据。

这也是论文标题 `Hide to See` 的意思：**先把容易偷懒用的文本线索藏起来，student 才会重新去“看”。**

### 2.1 论文用什么证据支持这个动机

- `intro_1_aa.png` 显示 naive distillation 下，随着 response 前缀不断累积，reverse KL 逐渐下降，说明后续 token 越来越容易靠已有文本上下文模仿出来。
- `intro_warp.png` 显示每一步 next-token prediction 往往只依赖极少数高注意力 prefix token，说明存在明显的 salient textual cue。
- Appendix 里的 `training_loss.png` 进一步说明：**mask response prefix** 会显著放大 distillation loss，而 mask visual/question token 则不会，这说明 student 主要在“吃文本 shortcut”，不是主要在吃视觉信息。

### 2.2 一张图看懂问题设定

![Reasoning-prefix masking intuition](assets/arxiv_2605_hide_to_see_reasoning_prefix_masking_for_visual_anchored_thinking_in_vlm_distillation/intro_10.png)

这张图非常关键。左边是 naive distillation：student 可以直接看完整 reasoning prefix，于是当前 token 的预测主要靠前文文字。右边是 Masking-KD：把一部分 salient reasoning prefix 挡住后，student 为了继续模仿 teacher 的当前 token 分布，就必须更多看图。

## 3. 方法

## 3.1 总体框架：Masking-KD

Masking-KD 的训练目标仍然是标准的 **reverse KL distillation**，但 teacher 和 student 的可见上下文不同：

- **teacher**: 使用标准 causal mask
- **student**: 使用扩展后的 `salient reasoning-prefix mask`

也就是：

- teacher 在完整上下文下给出目标分布
- student 在“被挖掉部分关键 reasoning prefix”的上下文下，去逼近 teacher 分布

这样 student 学到的不是单纯复述 teacher 的文字轨迹，而是在**缺失文本捷径**时，如何借助视觉证据补回这个差距。

### 3.2 方法图

![Masking-KD framework](assets/arxiv_2605_hide_to_see_reasoning_prefix_masking_for_visual_anchored_thinking_in_vlm_distillation/main_27.png)

这张图对应整篇 paper 的核心流程。最重要的是，它不是直接蒸馏 teacher 的视觉 attention map，而是先做一次 **auxiliary student forward**，从 student 自己当前状态里抽两类信号来构造 mask。

## 3.3 Auxiliary Student Forward

作者额外对 student 做一次共享权重的 causal forward，用来抽取两个量：

1. **response-to-response attention map** `A^resp`  
   用来判断：当前 token 在预测时，最依赖哪些前缀 reasoning token。

2. **token-wise reverse KL divergence** `r_n`  
   用来判断：当前 token 的蒸馏难度有多大。

这里的设计很漂亮，因为它把两个问题拆开了：

- **mask 谁**：由 attention 决定
- **mask 多少**：由 token difficulty 决定

这两个信号都来自 student 自己，而不是来自 teacher 的中间视觉表示。

## 3.4 Token-wise Salient Reasoning-prefix Masking

### 3.4.1 mask 谁

对每个当前预测位置 `n`，作者取 `A^resp` 的第 `n` 行，看这个位置在预测 `y_n` 时最依赖哪些 prefix token。

然后不是只取 top-1 / top-k，而是采用类似 nucleus top-p 的思路，做一个 **top-rho cumulative masking**：

- 先按 attention 从高到低排序 prefix token
- 再不断累积 attention mass
- 直到累计比例达到阈值 `rho_n`
- 被选中的这批 token 构成 `S_n`，即当前位置要屏蔽的 salient prefix 集合

所以它不是固定 mask 数量，而是固定“累计影响力预算”。

### 3.4.2 为什么是 token-wise

作者强调同一个 response 里，不同位置依赖的 prefix 完全不同，因此 mask 也必须 **逐 token 自适应**。

这意味着：

- 对位置 `n1`，可能 mask 掉一组 prefix
- 对位置 `n2`，会 mask 掉另一组 prefix

不是全序列共享一个静态 mask。

### 3.4.3 一个重要实现细节

作者显式说明：

- **不 mask 当前 token 的紧邻前一个 prefix token**

原因是如果把这个位置也纳入候选，会让训练明显不稳定，甚至 loss explosion。这个设计在 appendix 的 ablation 里也验证了。

## 3.5 Self-Paced Masking Budget Scheduling

### 3.5.1 mask 多少

作者认为不同 token 的蒸馏难度不同，统一 mask 强度不合理。

于是他们定义每个位置的 masking budget `rho_n`：

- easier token：更强 mask
- harder token：更弱 mask

token 难度用辅助前向得到的 reverse KL `r_n` 表示，再做如下变换：

- 先算 `tilde(r_n) = -log(r_n + epsilon)`
- 再和全序列平均值比较
- 通过 sigmoid 映射到 `[rho_min, rho_max]`

最终使用：

- `rho_min = 0.3`
- `rho_max = 0.5`

直觉上就是：

- 如果某个 token 原本就很容易模仿，那就把它依赖的 prefix 多遮一点，恢复蒸馏难度
- 如果某个 token 本来就难，那就别遮太狠，避免训练崩掉

### 3.5.2 这个设计的意义

这部分是整篇 paper 最值得借鉴的地方之一。  
它把“重要性干预”从单纯的 raw attention ranking，推进到了一个 **difficulty-aware budget scheduling**：

- attention 解决“哪些前缀最关键”
- reverse KL 解决“当前位置该不该施加更强干预”

这比只看 attention score 更稳，也更接近“训练信号恢复”的目标。

## 3.6 这篇方法和已有 VLM distillation 的根本区别

已有方法大多在做：

- teacher visual attention alignment
- vision token relation distillation
- instruction-aware visual focus transfer

而 Masking-KD 做的是：

- **不直接传 teacher 的视觉模式**
- **而是制造 student 必须重新用视觉证据的训练条件**

所以它更像一种“通过破坏文本捷径来促进视觉锚定”的 distillation，而不是“把 teacher 的视觉内部状态复制给 student”。

## 4. 实验

## 4.1 实验设置

### 数据

- 蒸馏数据来自 `ViRK39K`
- 用 teacher greedy decoding 抽取 think-answer traces
- 仅保留回答正确的样本
- 最终保留：
  - 19k for 8B
  - 15k for 4B
  - 10k for 2B

### 模型

- teacher / student backbone: `Qwen3-VL-Thinking`

### 评测集

- Geometry-3K
- MathVista
- We-Math
- MMK12
- MathVerse
- LogicVista
- MMMU-Pro

### 训练细节

- 2 epochs
- learning rate `1e-6`
- batch size `1`
- gradient accumulation `512`
- distillation temperature `tau = 2`
- `rho_min = 0.3`, `rho_max = 0.5`
- GPU:
  - 2x A100 80GB for 2B/4B
  - 4x A100 80GB for 8B self-distillation

## 4.2 主结果：和 open-source VLM 比较

这部分最醒目的结论有两个：

1. **Masking-KD-2B 明显强于未蒸馏的 Qwen3-VL-4B-Thinking**
2. **Masking-KD-4B 超过未蒸馏的 Qwen3-VL-8B-Thinking**

也就是说，它不是“小模型追上同尺寸 baseline”，而是直接打穿了更大一档的原始模型。

| 模型 | Geo3k | MathVista | We-Math | MMK12 | MathVerse | LogitVista | MMMU-Pro | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B-Thinking | 54.58 | 65.20 | 66.15 | 42.55 | 63.81 | 43.40 | 39.83 | 53.65 |
| **Masking-KD-8B (self-distill)** | **58.24** | **67.10** | **71.72** | **49.95** | **67.84** | **48.10** | **43.47** | **58.06** |
| Qwen3-VL-4B-Thinking | 43.93 | 62.60 | 49.37 | 31.55 | 49.86 | 39.37 | 32.08 | 44.11 |
| **Masking-KD-4B** | **52.58** | **66.50** | **71.03** | **51.00** | **62.66** | **52.35** | **40.52** | **56.66** |
| Qwen3-VL-2B-Thinking | 26.29 | 43.10 | 25.17 | 13.00 | 28.21 | 18.57 | 14.51 | 24.12 |
| **Masking-KD-2B** | **40.93** | **59.20** | **63.79** | **37.20** | **57.89** | **41.61** | **30.75** | **47.34** |

我觉得这里最强的一点不是单一 benchmark 的高分，而是 **七个 benchmark 几乎全线提升**，说明它不是某个数据集特化技巧。

## 4.3 和已有 VLM distillation 方法比较

作者把 LLaVA-KD、CompoDistill、Align-TI 都搬到同样的 think-answer distillation 设定下重做比较。

### 8B teacher -> 4B student

| 方法 | Geo3k | MathVista | We-Math | MMK12 | MathVerse | LogitVista | MMMU-Pro | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive Response Distillation | 47.42 | 62.80 | 60.00 | 37.85 | 56.97 | 42.95 | 34.22 | 48.89 |
| LLaVA-KD | 49.75 | 63.80 | 62.59 | 39.45 | 59.27 | 43.40 | 34.28 | 50.36 |
| CompoDistill | 50.92 | 63.40 | 64.54 | 39.30 | 60.18 | 45.19 | 35.49 | 51.29 |
| Align-TI | 50.58 | 62.80 | 62.75 | 39.15 | 58.30 | 42.95 | 35.38 | 50.27 |
| **Masking-KD** | **52.58** | **66.50** | **71.03** | **51.00** | **62.66** | **52.35** | **40.52** | **56.66** |

这张表很说明问题：  
**即使不显式蒸馏 teacher 的视觉 attention / visual relation，Masking-KD 仍然比这些显式视觉蒸馏方法更强。**

作者由此得出的结论是：

> 对 think-answer VLM，直接逼 student 形成自己的 visual-anchored thinking，比复制 teacher 的视觉模式更有效。

## 4.4 Self-distillation 结果

作者还对比了自蒸馏方法 OPSD。

| Base Model | Self-distill Method | Avg. |
|---|---|---:|
| Qwen3-VL-8B-Thinking | OPSD | 55.13 |
| Qwen3-VL-8B-Thinking | **Masking-KD** | **58.06** |
| Qwen3-VL-4B-Thinking | OPSD | 47.51 |
| Qwen3-VL-4B-Thinking | **Masking-KD** | **56.01** |
| Qwen3-VL-2B-Thinking | OPSD | 25.37 |
| Qwen3-VL-2B-Thinking | **Masking-KD** | **32.95** |

这个结果说明 Masking-KD 不依赖一个更强外部 teacher 的中间特征。  
即使 self-teacher 只是自己，**只要 student 端被遮掉 salient reasoning prefix，训练依然有效。**

## 4.5 Ablation

### 4.5.1 每个组件是否必要

| Token-wise salient masking | Self-paced scheduling | Avg. |
|---|---|---:|
| x | x | 38.26 |
| check | x | 45.91 |
| check | check | **47.34** |

解读：

- 单加 salient masking 就已经很强
- 再加 self-paced budget scheduling 还能继续涨

也就是说，这篇 paper 的主体贡献不是包装，而是两个部件都真的在起作用。

### 4.5.2 方法内部 ablation

| Ablation | Avg. |
|---|---:|
| Non-adaptive masking | 41.73 |
| **Token-wise adaptive masking** | **47.34** |
| Random prefixes | 42.71 |
| Low-attention prefixes | 37.29 |
| Middle-attention prefixes | 38.84 |
| **High-attention prefixes** | **47.34** |

这组结果说明两点：

1. **必须是 token-wise 自适应的 mask**
2. **确实应该 mask 高注意力 prefix，而不是随便 mask**

### 4.5.3 Self-paced budget 的细节

| Threshold type / range | Avg. |
|---|---:|
| Attention threshold | 46.29 |
| Masking ratio % | 46.21 |
| **Cumulative ratio rho** | **47.34** |
| [0.1, 0.3] | 45.76 |
| [0.5, 0.7] | 45.82 |
| **[0.3, 0.5]** | **47.34** |

这里可以看出，作者的 top-rho 设计不是随手拍的超参，确实优于简单 threshold 或固定比例。

### 4.5.4 mask 哪个区域最有效

| Masked Region | Avg. |
|---|---:|
| Visual tokens | 37.39 |
| Question tokens | 31.31 |
| **Response prefixes (ours)** | **47.34** |
| Naive Distillation | 38.26 |

这个 ablation 很有价值，因为它告诉我们：

- 问题不在“视觉 token 不够重要”
- 而在“文本 prefix 过于容易形成 shortcut”

所以正确做法不是遮图、也不是遮问题，而是**遮掉 response 中最会让 student 偷懒的那部分 prefix**。

## 4.6 额外分析

### 4.6.1 visual forgetting 是否真的被缓解

![Visual-anchored thinking evidence](assets/arxiv_2605_hide_to_see_reasoning_prefix_masking_for_visual_anchored_thinking_in_vlm_distillation/visual_forgetting_length.png)

这张图给出了最直观的分析：

- undistilled student 的 visual attention 随生成快速下降
- CompoDistill 有一定缓解，但仍下降
- Masking-KD 在整条 reasoning trace 上维持了最高视觉注意比例

也就是说，它不是单纯提分，而是真的把 student 的推理过程拉回到“持续看图”这件事上。

### 4.6.2 视觉 attention map 对比

![Overall visual attention](assets/arxiv_2605_hide_to_see_reasoning_prefix_masking_for_visual_anchored_thinking_in_vlm_distillation/overall_visual_attn.png)

作者把整条 thinking trace 上的 visual attention 做平均后发现：

- naive KD 和 undistilled student 的相关区域注意力不够集中
- CompoDistill 虽然显式做 attention alignment，但效果仍不如 Masking-KD
- Masking-KD 的视觉关注区域更聚焦于语义相关位置

这支撑了作者的主张：**forcing student to use visual evidence** 比 **forcing student to copy teacher visual pattern** 更有效。

### 4.6.3 训练时 student 的行为变化

![Prediction behavior during distillation](assets/arxiv_2605_hide_to_see_reasoning_prefix_masking_for_visual_anchored_thinking_in_vlm_distillation/qual.png)

这张图展示得很直白：

- 不加 mask 时，student 主要盯着文本 prefix
- 加 mask 后，student 会激活更多相关图像区域，去补偿缺失的 reasoning cue

这是整篇 paper 最关键的机制证据之一。

### 4.6.4 被 mask 的 prefix 大多离当前 token 很近

![Masked prefix distance](assets/arxiv_2605_hide_to_see_reasoning_prefix_masking_for_visual_anchored_thinking_in_vlm_distillation/masked_position.png)

Appendix 里统计了被 mask prefix 到当前 token 的相对距离。结论是：

- 被 mask 的 token 更常出现在较近的上下文
- 距离越远，被 mask 的频率越低

这说明高影响力的 reasoning cue 往往就藏在“最近那几步推理”里，和直觉一致。

### 4.6.5 推理案例

![Inference example](assets/arxiv_2605_hide_to_see_reasoning_prefix_masking_for_visual_anchored_thinking_in_vlm_distillation/qual_inference.png)

作者给出的案例里，未蒸馏 student 更容易出现 perception error，而 Masking-KD 对图像细节的理解更稳。这部分是定性证据，但和前面的 attention 分析是统一的。

## 4.7 统计稳定性与计算代价

### 统计稳定性

作者额外跑了 4 次 8B teacher -> 2B student：

- Avg. `47.44 +- 0.23`

标准差很小，说明收益不是 seed 偶然性。

### 计算代价

| 方法 | Per step (s) | Memory (GB) | Avg. Pass@1 |
|---|---:|---:|---:|
| Naive Response Distillation | 75.8 | 36.5 | 38.3 |
| LLaVA-KD | 100.4 | 37.2 | 40.3 |
| CompoDistill | 103.9 | 38.0 | 41.5 |
| Align-TI | 381.2 | 38.2 | 40.5 |
| **Masking-KD** | **157.9** | **40.9** | **47.3** |

这个表说明：

- Masking-KD 不是免费的
- 它比 naive KD 更慢，也更吃显存
- 主要开销来自 auxiliary student forward 和长 response 上的 response-to-response attention map

但从性能收益看，这个 trade-off 是成立的。

## 5. 论文的核心 takeaway

我觉得这篇 paper 最值得记住的不是“mask attention 高的 prefix”这件事本身，而是下面这三条：

1. **think-answer VLM distillation 的主要风险，是学生学会了文本 shortcut，而不是学会了视觉推理。**
2. **在这种场景里，恢复训练难度比复制 teacher 中间表示更重要。**
3. **token difficulty-aware 的干预强度调度，是比单纯 attention ranking 更完整的设计。**

## 6. 优点与局限

### 6.1 优点

1. **问题定义很准**  
   它抓到的是 think-answer VLM distillation 的特有问题，而不是泛泛地做一个“又一种 KD loss”。

2. **机制闭环完整**  
   从 textual shortcut -> salient prefix -> masking -> visual re-anchoring -> benchmark gain，证据链比较完整。

3. **结果非常硬**  
   2B 超 4B、4B 超 8B，这种结果很有说服力。

4. **self-distillation 也成立**  
   说明方法依赖的是训练条件设计，而不是强 teacher 的私有中间结构。

### 6.2 局限

下面这部分论文没有单列完整 limitation section，属于我基于正文和 appendix 的归纳：

1. **训练开销上升明显**  
   需要 auxiliary forward，还要拿 response-to-response attention map，时间和显存都有额外成本。

2. **方法仍依赖 attention 信息**  
   它不是 attention-free 的干预策略；重要 prefix 的识别仍然来自 attention map。

3. **验证场景主要集中在 reasoning-heavy benchmark**  
   对开放域、多轮交互、非推理型 VLM 任务是否同样有效，文中没有充分展开。

4. **和显式视觉蒸馏方法叠加时收益变小**  
   appendix 提到与其他方法结合时仍有提升，但增益变小，作者认为两类目标存在部分冲突。

## 7. 对复现和后续研究的启发

### 7.1 如果你是做 VLM distillation

这篇 paper 最直接的启发是：

- 不一定要显式对齐 teacher 的视觉 attention
- 可以从 **破坏 student 的文本捷径** 入手
- 让 student 自己长出 visual-anchored thinking

### 7.2 如果你在想“attention-free importance”这类问题

这篇论文虽然**不是 attention-free**，但它给了一个很有用的抽象：

- **重要性信号 1**: 哪些上下文最影响当前预测
- **重要性信号 2**: 当前 token 的训练难度 / 恢复难度

作者把这两个量拆开处理。  
对你前面关心的“不要直接依赖 attention score 的重要 KV 采样”问题，这里至少有两个可迁移思路：

1. **不要只做 token saliency ranking，也要做 budget scheduling**  
   即使你不用 attention 来选 KV，也可以借鉴它的 self-paced 设计，用 entropy / confidence / reverse-KL 一类量决定采样强度。

2. **把问题从‘谁重要’改写成‘缺了谁以后，模型还原目标分布有多难’**  
   这更接近 counterfactual / residual importance 的思想。

### 7.3 一个很具体的研究迁移点

如果你后面真要做“非 attention-based 的重要历史 token / KV 采样”，这篇 paper 可以启发出一个两阶段设计：

- 阶段 A：用某种非 attention 的代理信号估计 token 对未来预测的贡献
- 阶段 B：再用 difficulty-aware budget 决定每一步该保留多少历史信息

也就是把：

- `importance ranking`
- `retention budget scheduling`

显式拆开，而不是把它们揉成一个分数。

## 8. 一句话总结

**Masking-KD 的核心不是“mask 高 attention token”这么简单，而是通过遮掉最容易形成文本 shortcut 的 reasoning prefix，恢复后续 token 的蒸馏难度，迫使 student 在整个 think-answer 过程中持续重新看图。**
