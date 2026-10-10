# Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training

## 1. 论文基本信息
- arXiv ID: 2605.11931
- URL: [abs](https://arxiv.org/abs/2605.11931) / [pdf](https://arxiv.org/pdf/2605.11931.pdf)
- 主题归档: `vlm`
- 总结粒度: `detailed`
- 发表信息: ICML 2026
- 作者机构: Wuhan University / The University of Sydney / NTU
- 核心方法: **VISTA**
- 关键词: `MLLM reasoning`, `self-improvement`, `vision-aware`, `prefix resampling`, `visual hallucination`, `post-training`

## 2. 论文要解决什么问题

这篇工作讨论的是一个很具体、但也很现实的问题：

- 现在很多 MLLM 想提升 multimodal reasoning，往往要做带 reasoning trace 的 post-training
- 但高质量 reasoning trace 很贵
- 所以大家开始转向 **self-improvement**：让模型自己生成 reasoning data，再用这些数据反过来训练自己

作者认为，这条路线虽然有效，但在 MLLM 上有两个被低估的问题：

1. **data imbalance**  
   简单样本很容易采出很多正确 reasoning traces，困难样本却采不到足够的正确轨迹。结果是训练数据天然向“简单题”倾斜。

2. **language prior bias**  
   即使模型最终答对，它的中间 reasoning 过程也可能是“语言先验驱动”的，而不是真正看图。于是会出现：
   - 描述图里并不存在的内容
   - reasoning chain 和最终答案矛盾
   - 表面答对，但中间其实是 visual hallucination

作者的核心批评是：

> 现有 self-improvement 基本只用 “answer correctness” 作为唯一质量信号，但这个信号同时在数量和质量两个层面都不够。

也就是：

- 它不能帮你为难样本挖出更多有价值数据
- 它也不能区分“真正视觉 grounding 的正确解”和“误打误撞答对的幻觉解”

## 3. 核心观察与动机

## 3.1 论文的两个 preliminary finding

作者先做了两个经验分析：

### 发现 1：self-generated 正确解数量和样本难度强相关

![Preliminary analysis](assets/arxiv_2605_learn_to_think_improving_multimodal_reasoning_through_vision_aware_self_improvement_training/preliminary_analysis.png)

图里的结果显示：

- 简单任务上，一个 query 往往能采到很多正确解
- 难任务上，很多 query 一个正确解都采不到

比如在更难的 Geometry3K 上，作者说有 **40% 以上 query 没有任何正确解**。

这就导致 self-improvement 训练集会天然偏向简单样本，而这些简单样本又最不值得反复训练。

### 发现 2：MLLM 在 reasoning 中对 visual token 的关注不足

同一张图的另一部分给了 attention 分析：

- 虽然 visual token 占上下文很大一部分
- 但在很多层里得到的 attention 比例却不到 20%
- 模型更偏向 instruction token 和 system prompt token

作者把这解释为 **language prior bias**：  
模型更容易依赖语言上下文和模式记忆，而不是持续使用视觉证据。

## 3.2 一个很直观的例子

![Hallucinated but correct-looking reasoning](assets/arxiv_2605_learn_to_think_improving_multimodal_reasoning_through_vision_aware_self_improvement_training/figure1.png)

这张图很关键。作者想强调的是：

- 两条 reasoning trace 可能都预测对了最终答案
- 但其中一条中间 reasoning 明显带有 hallucination

如果你只按最终答案 correctness 来筛数据，这两条都会被收进训练集。  
久而久之，模型就会把这种“没真看图但也能蒙对”的推理模式学得更深。

## 4. 方法：VISTA

## 4.1 总体思路

VISTA 全称是：

- **VI**sion-aware **S**elf-improvement **T**raining for multimodal re**A**soning

它包含两个互补模块：

1. **prefix resampling**
   - 解决 data imbalance
   - 尤其是为困难 query 采更多正确解

2. **Vision-aware Attention Score (VAS)**
   - 解决 language prior bias
   - 用内部 attention 信息识别那些“答对但没真看图”的 undesired solution

整体流程是：

1. 基础模型先 self-generate 多条 candidate reasoning traces
2. 用 correctness 分成 positive / negative
3. 从 negative 里回收“前缀大致正确”的部分轨迹，重新采样
4. 再对 positive 里的正确解做 VAS 评分，过滤掉视觉 grounding 不足的解
5. 用剩余高质量数据做下一轮 SFT / DPO / GRPO

## 4.2 方法总览图

![VISTA overview](assets/arxiv_2605_learn_to_think_improving_multimodal_reasoning_through_vision_aware_self_improvement_training/method.png)

这张图基本概括了整篇 paper：

- 上半部分是 **prefix resampling**
- 下半部分是 **VAS filtering**

作者的设计很像“两次数据清洗”：

- 第一次提升**困难样本的覆盖率**
- 第二次提升**正确样本的视觉可靠性**

## 5. Prefix Resampling

## 5.1 关键直觉

作者借鉴了一个观察：

> 失败 reasoning trace 的错误，往往出现在后半段；前半段经常是部分正确、而且有信息量的。

所以他们不想像传统方法那样把错误解直接扔掉，而是想“抢救”它的前缀。

换句话说，VISTA 不是把 negative sample 当垃圾，而是把它看成：

- 也许后面错了
- 但前面可能已经走在正确轨道上

## 5.2 它怎么找“关键出错点”

作者的做法有点巧：

1. 对错误 solution，先把 visual token 和 instruction token 的顺序交换
   - 例如从 `x_vis + x_ins`
   - 变成 `x_ins + x_vis`

2. 把这个改写后的 query 和原来的 reasoning prefix 拼起来，再送进模型

3. 对 reasoning 中每个 token `o_n`，看它是否落在前一步的 Top-5 预测里：
   - 如果 `o_n` 不在 `Top-5(o_{n-1})`
   - 就把它视作 **critical token**

4. 找到第一个 critical token 后：
   - 用新的 Top-1 token 替代它
   - 截断后续 reasoning
   - 把“原 query + 截断前缀”作为 prefix
   - 再继续采样 `J` 次生成后续 reasoning

这就是 prefix resampling。

## 5.3 这个设计为什么有意思

它不是：

- 再多采几次
- 或用答案提示强行引导

而是：

- 先从错误轨迹里找“开始偏航的地方”
- 再从这个地方把轨道拉回来

所以它更像是一种 **trajectory recycling**。

## 5.4 伪代码层面怎么理解

附录里的算法可以概括成：

1. 先拿到正解集 `D_t^p` 和错解集 `D_t^n`
2. 对每个错解：
   - 找 critical token
   - 替换 critical token
   - 截断后继续采样
3. 把 resample 出来的正确解并回 `D_t^p`

也就是说，它在 self-improvement 的 data collection 阶段做的是“困难样本增广”，而不是训练目标改写。

## 6. Vision-aware Attention Score (VAS)

## 6.1 VAS 的核心问题

前面 prefix resampling 解决的是：

- “难题正确解不够多”

VAS 解决的是：

- “正确解里有不少其实不够视觉 grounding”

作者不想再依赖外部 judge 或额外模型做筛选，而是直接用 MLLM 自己的内部 attention 结构，量化它在 reasoning 时到底看了多少图。

## 6.2 VAS 的定义

对于一个正确 solution，作者在模型的中间层抽取 attention map，统计输出 reasoning token 对三类上下文的注意力总量：

- system prompt token: `lambda_sys`
- visual token: `lambda_vis`
- instruction token: `lambda_ins`

然后定义：

- `S = lambda_vis / (lambda_sys + lambda_vis + lambda_ins)`

再做标准化，得到：

- `VAS = (S - mean(S_i)) / std(S_i)`

直觉上就是：

- 如果 reasoning token 几乎不看视觉 token
- 那么这个 solution 很可能是被语言先验推着走
- 即使最后答对，也值得怀疑

所以他们把：

- `VAS < tau`

的正确解过滤掉。

默认阈值设置是：

- `tau = -0.5`

## 6.3 为什么选中间层

作者认为：

- 中间层更适合承载视觉语义处理
- 太浅层可能还在处理局部模式
- 太深层更偏输出决策

附录里也做了层深度 ablation，发现中间层效果最好，但整体上方法对层选择不算特别敏感。

## 7. 实验设置

## 7.1 任务与数据集

主任务覆盖两类：

- 医学 reasoning
  - SLAKE
  - VQA-Rad
- 数学 reasoning
  - Geometry3K

额外泛化任务：

- ScienceQA
- ChartQA

OOD 与 hallucination 评测：

- PathVQA
- MathVista
- IllusionBench
- MMMU 医学子任务
- MathVerse
- BLINK-Twice

## 7.2 模型

作者在多种 MLLM 上做实验，包括：

- Qwen2.5-VL-3B-Instruct
- Qwen2.5-VL-7B-Instruct
- Qwen3-VL-2B-Instruct
- InternVL3-2B
- InternVL3-8B

## 7.3 训练细节

- self-generation sampling count `K = 10`
- prefix resampling continuation count `J = 3`
- threshold `tau = -0.5`
- output max length `2048`
- self-improvement iteration
  - Qwen2.5-VL 主实验：`T = 3`
  - 其他模型：`T = 1`
- 训练 3 epochs
- 8 x NVIDIA A800 80GB

还有一个复现上很重要的细节：

- **冻结 vision encoder 和 multimodal projector**
- **只更新 LLM backbone**

这说明它不是通过重新大幅改视觉编码器来变强，而更多是在调 reasoning 行为。

## 8. 主结果

## 8.1 在 Qwen2.5-VL 系列上的主结果

下面这张表是论文最核心的主结果之一。

### Qwen2.5-VL-3B-Instruct

| Method | SLAKE | VQA-Rad | Geo3K | Overall |
|---|---:|---:|---:|---:|
| SFT-Seed | 67.04 | 64.14 | 25.46 | 52.21 |
| SFT-Oracle | 81.97 | 68.13 | 37.94 | 62.68 |
| RFT | 79.15 | 70.12 | 28.79 | 59.35 |
| STaR Iter3 | 80.28 | 69.32 | 30.78 | 60.13 |
| ReST^EM Iter3 | 81.69 | 73.71 | 32.28 | 62.56 |
| R3V Iter2 | 82.82 | 74.10 | 32.61 | 63.18 |
| **VISTA-SFT Iter3** | **84.23** | **76.10** | **37.27** | **65.87** |

相对 SFT-Seed：

- **平均提升 +13.66%**

这也是摘要里反复强调的 headline。

### Qwen2.5-VL-7B-Instruct

| Method | SLAKE | VQA-Rad | Geo3K | Overall |
|---|---:|---:|---:|---:|
| SFT-Seed | 79.15 | 70.52 | 36.94 | 62.20 |
| SFT-Oracle | 82.25 | 70.12 | 41.43 | 64.60 |
| RFT | 82.82 | 74.90 | 40.27 | 66.00 |
| STaR Iter3 | 84.23 | 69.72 | 37.10 | 63.68 |
| ReST^EM Iter3 | 84.51 | 75.30 | 38.94 | 66.25 |
| R3V Iter3 | 87.32 | 75.70 | 37.27 | 66.76 |
| **VISTA-SFT Iter3** | **87.89** | **77.29** | **41.43** | **68.87** |

相对 7B 的 SFT-Seed：

- **平均提升 +6.67%**

一个有意思的点是：  
VISTA 在 3B 上的收益比 7B 更大，说明它对中小模型的 self-improvement 可能更关键。

## 8.2 在更多 MLLM 上的泛化

作者还只做一轮 self-improvement，测试它在其他模型上的迁移性：

| Method | Qwen3-VL-2B | InternVL3-2B | InternVL3-8B | Overall |
|---|---:|---:|---:|---:|
| SFT-Seed Avg | 57.13 | 52.21 | 67.27 | 58.87 |
| SFT-Oracle Avg | 62.21 | 62.29 | 69.39 | 64.63 |
| STaR Avg | 59.61 | 55.95 | 67.11 | 60.56 |
| STaR+ Avg | 60.66 | 60.54 | 67.60 | 62.92 |
| R3V Avg | 61.20 | 60.48 | 67.77 | 63.15 |
| **VISTA-SFT Avg** | **63.18** | **62.17** | **70.56** | **65.30** |

这里有两点比较强：

1. 一轮 self-improvement 就有效
2. 有些设置下还略超过了 SFT-Oracle

也就是说，作者不是只在一个 backbone 上做“局部工程优化”，而是试图提出一种更通用的 self-improvement recipe。

## 8.3 对更多任务也有效

在 ScienceQA / ChartQA 上，VISTA 同样超过 STaR / STaR+ / R3V：

| Model | Task | Seed | VISTA-SFT |
|---|---|---:|---:|
| Qwen2.5-VL-3B | ScienceQA | 74.66 | **82.29** |
| Qwen2.5-VL-3B | ChartQA | 60.82 | **62.09** |
| Qwen2.5-VL-7B | ScienceQA | 81.55 | **87.58** |
| Qwen2.5-VL-7B | ChartQA | 72.32 | **73.68** |

这说明它不只是对 medical/math 这两个主任务有效。

## 8.4 不只适用于 SFT，也适用于 DPO / GRPO

作者还把 VISTA 接到不同后训练范式上：

### DPO

| Method | SLAKE | VQA-Rad | Geo3K | Overall |
|---|---:|---:|---:|---:|
| IRPO | 86.20 | 74.10 | 29.11 | 63.14 |
| **VISTA-DPO** | **89.01** | **76.89** | **29.45** | **65.12** |

### GRPO

| Method | SLAKE | VQA-Rad | Geo3K | Overall |
|---|---:|---:|---:|---:|
| ReST^EM-GRPO | 84.51 | 78.09 | 32.45 | 65.02 |
| **VISTA-GRPO** | **85.35** | **79.28** | **32.61** | **65.75** |

这个实验其实挺重要，因为它说明 VISTA 不是绑定在某个特定训练 loss 上，而更像是：

- 一套 data-centric self-improvement enhancement

## 9. More Analysis

## 9.1 Prefix resampling 确实缓解了 data imbalance

作者把 prefix resampling 和两类思路对比：

- 给难题更多采样次数
- 用答案引导 reasoning

结论是 prefix resampling 效果更好。  
作者的解释也很合理：

- 部分正确前缀提供了更强的上下文支撑
- 比盲目继续多采或者直接 answer-guided 更省、更稳

附录里还有一个更直接的图：

![Method analysis](assets/arxiv_2605_learn_to_think_improving_multimodal_reasoning_through_vision_aware_self_improvement_training/method_analysis.png)

这里显示：

- 对“原本一个正确解都采不到”的 hardest queries
- prefix resampling 仍能为相当一部分 query 补出至少一个正确解

作者还说，在 Geometry3K 上：

- 平均每个 query 的正确解数量从 **2.4 提升到 4.4**

这就是 data imbalance 被缓解的直接证据。

## 9.2 VAS 能有效识别 hallucinated solution

作者用 GPT-4o 标注了一批 hallucinated vs non-hallucinated 正确解，然后比较它们的 VAS 分数分布。

结果是：

- hallucinated solutions 平均 VAS 约为 `-0.2`
- non-hallucinated solutions 平均 VAS 约为 `0.31`

这说明 VAS 不是一个拍脑袋的 attention heuristic，而是确实能在统计上把“答对但没真看图”的解筛出来。

## 9.3 和其他数据选择指标的比较

作者还拿 VAS 对比：

- Answer PPL
- Reasoning Depth

结果是：

- 这些指标在 MLLM self-improvement 上并不稳定
- 甚至可能不如随机选正确解
- VAS 更稳定，也更能体现视觉 grounding 质量

这一点很关键，因为它说明：

- reasoning quality 并不等于 perplexity 低
- 也不等于 hidden-state 某种深度度量大
- 对 MLLM 来说，“有没有真看图”是一个独立维度

## 9.4 阈值 tau 不太敏感，但太大不行

作者分析了 `tau` 从 `-0.75` 到 `0.75` 的变化，发现：

- VISTA 在较宽范围内都比 baseline 好
- 说明方法对阈值不是特别脆弱
- 但 `tau` 太大时会过滤掉太多有用样本，性能下降

最终默认取：

- `tau = -0.5`

## 9.5 幻觉与 OOD 泛化

### IllusionBench

![IllusionBench results](assets/arxiv_2605_learn_to_think_improving_multimodal_reasoning_through_vision_aware_self_improvement_training/illusionbench.png)

作者发现：

- 一些自训练 baseline 在 No Illusion 上还行
- 但在其他 illusion 类型上更容易被 hallucination 放大
- VISTA 则在全部 illusion 类别上更稳

这说明如果你把 hallucinated-but-correct 的解不断喂回去训练，模型可能会学坏它的视觉 grounding。

### OOD 结果

![OOD results](assets/arxiv_2605_learn_to_think_improving_multimodal_reasoning_through_vision_aware_self_improvement_training/ood_results.png)

作者做了两类 OOD：

- SLAKE -> PathVQA
- Geo3K -> MathVista

结果显示：

- baseline 随 self-improvement iteration 增加，OOD 结果经常下降
- VISTA 则能持续更稳，甚至显著上升

作者还补了更难的 OOD：

| Train -> OOD | Seed | VISTA-SFT |
|---|---:|---:|
| SLAKE -> MMMU | 16.34 | **17.64** |
| SLAKE -> BLINK | 49.35 | **51.23** |
| Geo3K -> MathVerse | 17.08 | **21.76** |
| Geo3K -> BLINK | 39.17 | **40.26** |

这个结果很重要，因为它说明 VISTA 不只是把训练集分数抬上去，而是真的提升了视觉 grounding 和泛化。

## 9.6 Prefix resampling 的一些细节 ablation

附录里还有两个有价值的实验：

### token 位置 swap 比 mask / noise 更好

| Method | SLAKE | VQA-Rad |
|---|---:|---:|
| ReST^EM | 76.06 | 69.72 |
| **VISTA-Swap** | **80.85** | **74.10** |
| VISTA-Mask | 78.31 | 70.12 |
| VISTA-Noise | 78.87 | 70.52 |

这说明 swap 不是随便挑的扰动方式，而是更能触发“上下文结构变化下的自校准”。

### Top-k 选择里 `k=5` 最好

| Top-k | SLAKE | VQA-Rad |
|---|---:|---:|
| 1 | 80.28 | 72.91 |
| **5** | **80.85** | **74.10** |
| 10 | 80.00 | 71.71 |
| 50 | 78.03 | 69.72 |

解释也很直观：

- 太小会过度敏感，前缀截得太早
- 太大会漏掉真正 critical token

## 9.7 效率

一个自然疑问是：VISTA 不是增加了很多额外 forward 吗？

作者给了 wall-clock 对比（Qwen2.5-3B-VL on SLAKE）：

| Method | Data Collection | Training | Total |
|---|---:|---:|---:|
| STaR | 1.3 min | 2.5 min | 3.9 min |
| ReST^EM | 2.3 min | 26.2 min | 28.6 min |
| R3V | 3.3 min | 49.7 min | 53.0 min |
| **VISTA-SFT** | 12.0 min | 17.2 min | 29.2 min |

也就是说：

- 它在 data collection 阶段更贵
- 但训练数据量减少了，training 更省
- 总成本和 ReST^EM 接近，比 R3V 低很多

所以它是一个相对平衡的 trade-off，不是纯堆算力换精度。

## 10. 论文的优点

1. **问题定义很准**  
   不是简单说 self-improvement 不够强，而是把问题拆成 data imbalance 和 language prior bias 两个可操作的子问题。

2. **方法非常贴合 MLLM 场景**  
   prefix resampling 针对 reasoning trajectory，VAS 针对视觉 grounding，这两部分都不是 LLM self-improvement 可以直接照搬的。

3. **结果跨模型、跨任务、跨训练范式都成立**  
   不只在 SFT 上好，也在 DPO / GRPO 上有增益。

4. **有较强的 OOD 和 hallucination 分析**  
   不是只看 in-domain accuracy。

5. **实现上比较朴素，工程可落地**  
   不依赖额外 teacher，也不依赖复杂 reward model。

## 11. 论文的局限

作者自己也比较诚实，主要提了两点：

1. **prefix resampling 可能压缩 trajectory diversity**  
   尤其当 critical token 出现在很后面时，resampled solution 会和原解高度重合。

2. **实验规模主要在 2B / 3B / 7B / 8B**  
   还没有验证到更大规模模型，特别是 30B / 70B 或 MoE。

我自己再补两点：

3. **VAS 仍然是 attention-based proxy**
   它并不直接验证 reasoning 是否 grounded，而是用视觉 attention 比例做代理。这个代理在更复杂架构上是否稳定，还值得继续看。

4. **prefix critical token 检测依赖 token-order swap 和 Top-k calibration**
   这个机制挺巧，但也比较“任务定制”；在别的模态拼接格式上是否同样有效，可能还需要更多验证。

## 12. 对后续研究的启发

我觉得这篇对你前面一直关心的“细粒度重要性判断”和“不要只看最终 correctness”其实很有启发。

### 12.1 对 self-improvement 的启发

这篇最核心的一句可以概括成：

> 正确答案不是足够的训练信号。

在 MLLM self-improvement 里，你至少还需要考虑：

- 样本是不是足够难
- reasoning 过程是不是视觉 grounding 的

### 12.2 对细粒度重要性建模的启发

VISTA 的两个模块其实对应两种“细粒度质量信号”：

1. **prefix-level recoverability**
   - 失败轨迹里哪些前缀仍有价值
   - 哪个 token 是真正开始偏航的点

2. **trajectory-level visual grounding**
   - 一条推理轨迹虽然答对了，但是否真的依赖视觉证据

这比只用 answer correctness 或只用 final reward 更细。

### 12.3 对未来可做方向的启发

我觉得顺着这篇 paper，可以继续往下做：

1. **更强的 groundedness score**
   - 不只看 attention ratio
   - 结合反事实遮挡、局部证据替换、跨层一致性

2. **更柔性的 prefix recycling**
   - 不只截断一个点继续采样
   - 可以做多候选 critical point、树状续写

3. **把 VISTA 接到更大的 reasoning RL pipeline**
   - 先做 VISTA-style data curation
   - 再做 GRPO / RLOO / preference optimization

## 13. 一句话总结

**VISTA 的核心价值，在于它把 MLLM self-improvement 从“只看答对没答对”推进到了“既要补足难样本的数据覆盖，又要筛掉没有真正看图的伪正确推理”。**
