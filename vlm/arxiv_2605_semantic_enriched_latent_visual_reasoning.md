# Semantic-Enriched Latent Visual Reasoning

## 1. 基本信息

- arXiv ID: `2605.19342`
- 标题: `Semantic-Enriched Latent Visual Reasoning`
- 主题: `vlm`
- 粒度: `细粒度`
- 论文链接: [https://arxiv.org/abs/2605.19342](https://arxiv.org/abs/2605.19342)
- PDF: [https://arxiv.org/pdf/2605.19342](https://arxiv.org/pdf/2605.19342)
- 代码: [https://github.com/tinnel123666888/slvr](https://github.com/tinnel123666888/slvr)
- 本地 PDF: `vlm/arxiv_2605_semantic_enriched_latent_visual_reasoning.pdf`

## 2. 一句话概括

这篇工作提出 `SLVR`，核心目标是让 VLM 的 `latent visual reasoning` 不再只保留“视觉外观压缩信息”，而是显式注入 `attribute-level semantic supervision`，并进一步通过 `M-GRPO` 强化“同一视觉区域面对不同问题时，latent 仍能保持语义一致性与可复用性”。

## 3. 论文要解决什么问题

这篇文章针对的是近期 `thinking with images` / `latent visual reasoning` 路线中的一个关键短板。

现有两类方法各有问题:

1. 显式 visual reasoning
   - 比如 crop、zoom-in、OCR、visual tool invocation 这类方法。
   - 优点是中间视觉证据清晰。
   - 缺点是推理阶段要反复调用视觉操作，效率低，链路也长。

2. latent visual reasoning
   - 直接在紧凑 latent 空间里做视觉推理。
   - 优点是省掉显式 visual operations，更高效。
   - 缺点是已有方法大多只用视觉信号监督 latent，latent 容易只保留 appearance-level cue，而缺少足够细粒度、结构化、可迁移的语义内容。

作者认为，这会带来两个直接后果:

- 同一个区域换个问法，latent 不一定还能稳定回答。
- latent 可能能支持某一个 query，但不能同时支持多个语义角度不同的问题。

因此，这篇文章想回答的问题是:

`能不能学出一种 region-centric latent，它不仅保留局部视觉证据，还显式编码对象属性、动作状态、空间关系、文本内容等语义，并且在同一区域的多问题设置下仍保持一致？`

## 4. 核心思路

作者提出 `SLVR (Semantic-Enriched Latent Visual Reasoning)`，是一个两阶段框架:

1. `Stage 1: Structured Latent Learning`
   - 对 region latent 做视觉对齐。
   - 额外引入 semantic latent，并用 attribute embedding 对它做监督。

2. `Stage 2: Multi-query Group Relative Policy Optimization`
   - 针对同一视觉区域对应的多个不同问题，联合优化 visual latent 和 semantic latent。
   - 目标是让它们既能答对题，又能在多 query 间保持一致，同时不要偏离 Stage 1 学到的稳定分布太远。

这篇文章最重要的观点不是“latent reasoning 比 text reasoning 更强”，而是:

`latent 本身也需要被显式塑形，尤其要塑造成能承载结构化视觉语义的 latent。`

![概念对比图](assets/arxiv_2605_semantic_enriched_latent_visual_reasoning/figure1_n.png)

上图对应论文 Figure 1。作者把三条路线放在一起对比:

- 显式 chain-of-thought / cropped evidence
- 纯视觉监督的 latent reasoning
- 他们提出的“视觉 + 语义联合监督”的 latent reasoning

这张图传达得很直接: 他们认为已有 latent reasoning 的核心瓶颈不是 latent 不存在，而是 latent 的 `semantic richness` 不够。

## 5. 数据构造

论文不仅提出训练框架，还构造了一整套支持数据，统称 `SLV-Set`，以及一个专门的新 benchmark `SV-QA`。

### 5.1 SLV-Set

SLV-Set 建立在 `Visual-CoT (ViSCoT)` 之上，分成两部分:

1. `Attribute-Level Semantic Dataset`
   - 大约 `400K` region-level semantic profiles。
   - 每个关键区域都有结构化 attribute 描述。

2. `Multi-Query Dataset`
   - 大约 `800K` region-centric QA pairs。
   - 每个区域生成两个语义不同但指向同一区域的问题。

### 5.2 SV-QA Benchmark

`SV-QA` 是他们专门构造的评测集，用来测试 “同一视觉区域在不同语义提问下，latent 是否仍稳健”。

- 总计 `591` 个 paired samples
- 每个样本是 $(i, q_1, q_2, b)$
- 其中:
  - $i$ 是图像
  - $q_1, q_2$ 是针对同一目标区域的两个不同问题
  - $b$ 是共享 target region 的 bounding box

来源上，他们从已有 benchmark 里出发，包括:

- `V*`
- `HRBench-4K`
- `HRBench-8K`

然后把原问题当作 $q_1$，再用 Qwen3-VL-235B 生成一个从不同语义角度询问同一区域的新问题 $q_2$。

![数据构造图](assets/arxiv_2605_semantic_enriched_latent_visual_reasoning/dataset_region_multiq.png)

这张图很关键，因为它说明作者不是在做普通 multi-turn QA，而是在做 `same region, different semantic queries` 的控制变量设计。这样就能更直接测试 latent 是否真的学到了“共享区域语义”，而不是仅仅适配某个单一问题模板。

### 5.3 标注质量分析

论文还报告了自动标注后的人工核查结果。总共发现 `29,457` 个错误标注，整体错误率 `7.29%`。主要错误类型如下:

| Error Category | Count | Ratio (%) |
|---|---:|---:|
| Color mismatch | 5,873 | 19.94 |
| Action / pose error | 4,286 | 14.55 |
| Spatial relation error | 3,912 | 13.28 |
| Hallucinated attributes | 3,476 | 11.80 |
| Invisible objects | 3,158 | 10.72 |
| Incorrect held object | 2,941 | 9.98 |
| Ambiguous references | 2,834 | 9.62 |
| Overly vague | 2,267 | 7.70 |
| Others | 710 | 2.41 |

这个表说明作者至少认真检查了自动构造数据的噪声来源。最容易出错的是颜色、动作姿态和空间关系，这也恰好是细粒度视觉语义里最脆弱的部分。

## 6. 方法细节

![SLVR 总体框架](assets/arxiv_2605_semantic_enriched_latent_visual_reasoning/figure2.png)

Figure 2 是整篇文章最重要的图。它把 SLVR 的两阶段学习写得比较清楚。

### 6.1 Stage 1: Structured Latent Learning

输入是图像 $I$ 和原始问题 $q$，问题已经绑定到一个关键视觉区域。

作者做了两种 latent:

1. `region-level visual latent`
   - 从 ROI 对应的视觉 patch 特征中取出一段专门的 latent segment。
   - 这段 latent 负责保留局部视觉细节。

2. `semantic latent`
   - 在 `vis_end` 后面插一个特殊 token `sem`。
   - 其 hidden state 记为 $z_{\text{sem}}$。
   - 它负责聚合这个区域的语义信息。

论文把 ROI latent 写为:

\[
\mathcal{H}_{\text{vis}} = \{\mathbf{h}^{\text{lat}}_t\}_{t=1}^{T_v}
\]

并通过 vision encoder 的 ROI 特征做直接对齐:

\[
\mathcal{L}_{\text{vis}} = \sum_{t=1}^{T_v} \left\| \mathbf{h}^{\text{lat}}_t - \mathbf{v}^{\text{enc}}_t \right\|_2^2
\]

这部分延续了 LVR 的基本精神: 让 latent 尽量贴近原始局部视觉表征。

接着，semantic latent 用 attribute-level semantic embedding 来监督。每个区域会对应一个 `4096` 维 embedding:

\[
\mathbf{e} \in \mathbb{R}^{4096}
\]

semantic latent 先过一个投影头:

\[
\hat{\mathbf{e}}_{\text{sem}} = W z_{\text{sem}}
\]

然后用均方误差与属性 embedding 对齐:

\[
\mathcal{L}_{\text{sem}} = \mathrm{MSE}\!\left(\hat{\mathbf{e}}_{\text{sem}}, \mathbf{e}\right) = \frac{1}{4096}\left\lVert \hat{\mathbf{e}}_{\text{sem}} - \mathbf{e} \right\rVert_2^2
\]

最终 Stage 1 的目标就是:

\[
\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{vis}} + \mathcal{L}_{\text{sem}}
\]

这一步的本质是:

- `H_vis` 学视觉保真
- `z_sem` 学结构化语义摘要

所以 Stage 1 相当于先把 latent 表示空间“拉开职责分工”。

### 6.2 Stage 2: M-GRPO

作者觉得只做 Stage 1 还不够，因为即便 latent 学到了语义，也未必能在不同问题之间稳定共享。于是提出 `M-GRPO (Multi-query Group Relative Policy Optimization)`。

对同一区域、不同问题 $q_i$，模型会产生一对 latent:

\[
\big(H_{\text{vis}}^{(i)},\, z_{\text{sem}}^{(i)}\big) = f(I, r, q_i)
\]

Stage 2 的奖励由三部分组成。

#### 6.2.1 Answer Correctness Reward

\[
\mathcal{R}_{\text{ans}}^{(i)} = \mathbb{I}\big(\hat{y}_i = y_i\big)
\]

这里的正确性由外部 judge model `Qwen3-Max` 二值判定。

#### 6.2.2 Latent Consistency Reward

作者要求不同 query 生成出来的 visual latent 和 semantic latent 都不要偏得太开:

\[
\begin{aligned}
\mathcal{R}_{\text{cons}}
= - \sum_{i \neq j} \Bigg(
& \lambda_{\text{sem}} \left\lVert z_{\text{sem}}^{(i)} - z_{\text{sem}}^{(j)} \right\rVert_2 \\
& + \lambda_{\text{vis}} \frac{1}{T_v}\sum_{t=1}^{T_v}
\left\lVert h_t^{(i)} - h_t^{(j)} \right\rVert_2
\Bigg).
\end{aligned}
\]

这条式子非常关键。它明确地把 “同一区域、不同问题下 latent 的一致性” 写进了优化目标，而不是只看答案对不对。

#### 6.2.3 Stability Regularization

为了避免 Stage 2 训练把 Stage 1 学好的分布破坏掉，作者还把 Stage 1 产出的 latent 当作 reference anchor:

\[
\begin{aligned}
\mathcal{R}_{\text{stab}}^{(i)}
= &- \max\!\left(0,\; \left\lVert z_{\text{sem}}^{(i)} - \bar{z}_{\text{sem}} \right\rVert_2 - \tau_{\text{sem}} \right) \\
&- \max\!\left(0,\; \frac{1}{T_v}\sum_{t=1}^{T_v}\left\lVert h_t^{(i)} - \bar{h}_t \right\rVert_2 - \tau_{\text{vis}} \right).
\end{aligned}
\]

这一步可以理解成: 允许针对不同问题做适应，但不能漂得太远。

### 6.3 M-GRPO 的整体目标

M-GRPO 采用 group-based GRPO 目标，对每个共享区域的 query group 联合更新。论文给出的目标较长，核心结构就是:

- 对同一区域的两个问题联合采样 rollout
- 用 clipped policy objective 做更新
- 加上 KL 正则约束

它的关键思想不是普通 RL，而是 `grouped multi-query latent policy optimization`。也就是，更新单位不再只是单问题样本，而是“同一区域上的问题组”。

## 7. 实验设置

### 7.1 公开 benchmark

标准 VQA benchmark 包括:

- `OKVQA`
- `GQA`
- `VizWiz`
- `ChartQA`
- `TextVQA`
- `AI2D`

此外还有:

- `VisualPuzzles`
- 新构造的 `SV-QA`

### 7.2 比较方法

作者对比了两大类方法:

1. `think-with-image`
   - `GRIT`
   - `DeepEyes`
   - `Visual-SR1`

2. `latent reasoning`
   - `LVR`
   - 文中也提到 `Monet`

并且还拿 `Qwen2.5-VL-7B` 作为 text-only baseline。

## 8. 主要结果

### 8.1 标准 VQA benchmark

| Model | Reasoning Paradigm | OKVQA | GQA | VizWiz | ChartQA | TextVQA | AI2D |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | Text-only | 58.9 | 53.2 | 54.1 | 74.4 | 79.1 | 69.5 |
| GRIT | Text CoT | 43.1 | 54.8 | 39.4 | 24.6 | 60.8 | 75.5 |
| DeepEyes | Text + Visual Cropping | 40.0 | 45.5 | 32.2 | -- | 40.4 | 37.0 |
| Visual-SR1 | Text CoT | 49.0 | 52.3 | 35.8 | 71.7 | 70.1 | 69.0 |
| LVR | Visual Latent | 50.6 | **57.4** | 33.1 | 64.4 | 75.1 | **77.3** |
| SLVR-7B | Visual + Semantic Latent | **61.8** | 55.6 | **57.8** | **77.2** | **79.3** | 76.0 |

相对 LVR 的变化:

- OKVQA: `+11.2`
- GQA: `-1.8`
- VizWiz: `+24.7`
- ChartQA: `+12.8`
- TextVQA: `+4.2`
- AI2D: `-1.3`

这个结果说明两点:

1. `semantic-enriched latent` 对知识型、图文混合型、结构化感知型任务帮助明显，尤其是 `VizWiz` 和 `ChartQA`。
2. 它并不是在所有标准 benchmark 上都绝对碾压，像 `GQA` 和 `AI2D` 反而略低于 LVR，这说明语义增强不一定总能无损替代纯视觉 latent。

### 8.2 VisualPuzzles

| Model | Algorithmic | Analogical | Deductive | Inductive | Spatial | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL | 35.9 | 26.1 | 35.5 | **28.7** | 21.3 | 29.2 |
| LVR | 31.3 | 25.6 | 40.5 | 24.4 | 26.2 | 29.4 |
| SLVR | **37.4** | **28.0** | **45.5** | 26.3 | **33.6** | **34.2** |

相比 LVR，SLVR 在:

- Deductive: `+5.0`
- Spatial: `+7.4`
- Overall: `+4.8`

这里很能说明作者的观点: 当任务更依赖结构关系、空间属性、区域语义时，semantic-enriched latent 更有优势。

### 8.3 SV-QA: 同一区域双问题评测

论文最有辨识度的结果在 `SV-QA`。评价指标包括:

- `Q1`
- `Q2`
- `Both Correct`

其中 `Both Correct` 最重要，因为它要求同一区域的两个语义不同问题都答对。

| Setting | Reasoning Paradigm | V* Q1 | V* Q2 | V* Both | HRBench-4K Q1 | HRBench-4K Q2 | HRBench-4K Both | HRBench-8K Q1 | HRBench-8K Q2 | HRBench-8K Both |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL | Text-only | 76.4 | 55.5 | 44.0 | 60.5 | 74.1 | 45.8 | 53.5 | 66.1 | 37.5 |
| GRIT | Text CoT | 69.6 | 61.8 | 49.7 | 58.4 | 70.4 | 48.6 | 49.1 | 61.9 | 38.6 |
| DeepEyes | Text + Visual Cropping | **83.3** | 79.1 | **70.2** | **71.3** | 79.3 | 60.6 | **65.1** | 72.4 | 49.0 |
| Visual-SR1 | Text CoT | 75.9 | 73.3 | 62.8 | 63.9 | 76.4 | 54.6 | 56.3 | 67.4 | 44.4 |
| LVR | Visual Latent | 81.7 | 77.5 | 65.4 | 70.1 | 80.4 | 57.9 | 62.9 | 71.1 | 46.6 |
| SLVR-7B | Visual + Semantic Latent | 82.2 | **80.1** | 69.1 | 70.4 | **82.8** | **61.1** | 62.5 | **73.6** | **50.6** |

对 LVR 的提升:

- V*: `Q2 +2.6`, `Both +3.7`
- HRBench-4K: `Q2 +2.4`, `Both +3.2`
- HRBench-8K: `Q2 +2.5`, `Both +4.0`

这个结果很关键。SLVR 在很多场景里并不是 Q1 最强，但它在 `Q2` 和 `Both Correct` 上更加稳定。这和文章的中心论点高度一致:

`semantic-enriched latent 的价值不只是“答对一个问题”，而是“同一个 latent 能支撑多个语义视角”。`

DeepEyes 在一些 Q1 指标上更强，但它依赖显式 visual cropping，推理成本更高；SLVR 则是在纯 latent-space reasoning 下取得这些结果。

## 9. 消融实验

作者做了两类消融:

1. `text-only vs latent reasoning`
2. `latent family 内部增量消融`

### 9.1 Text-only baseline

在 Qwen2.5-VL-7B 上:

- `SFT (single-Q)`
- `SFT (multi-Q)`
- `SFT + GRPO (multi-Q)`

可以看到多问题训练和 GRPO 都能带来一些提升，但即使最强的 text-only 配置，在 `Both Correct` 上仍然持续落后于 latent reasoning 变体。

这说明:

`仅仅把多个问题写进文本推理链里，不足以获得稳定的共享区域语义表示。`

### 9.2 Latent 系列增量消融

| Setting | Stage 1 | Multi-Q | GRPO | M-GRPO | V* Both | HRBench-4K Both | HRBench-8K Both |
|---|---:|---:|---:|---:|---:|---:|---:|
| LVR | -- | -- | -- | -- | 65.4 | 57.9 | 46.6 |
| + Stage 1 | yes | -- | -- | -- | 67.5 | 59.6 | 46.9 |
| + GRPO (Single-Q) | yes | -- | yes | -- | 62.8 | 57.6 | 49.3 |
| + Multi-Q | yes | yes | yes | -- | 68.1 | 60.3 | 49.9 |
| LVR + Multi-Q | -- | yes | yes | -- | 67.5 | 60.0 | 49.4 |
| SLVR-7B (Full) | yes | yes | yes | yes | **69.1** | **61.1** | **50.6** |

这张表能读出几个明确结论:

1. 单独加入 `Stage 1 semantic supervision` 就能提升 joint correctness。
2. 单独做 `Single-Q GRPO` 收益并不稳定，甚至会掉。
3. `Multi-Q` 明显有帮助，说明 richer supervision signal 很重要。
4. 完整 `M-GRPO` 最强，说明真正有效的不只是多问题训练，而是 `对多问题 latent 一致性进行显式约束`。

换句话说，作者的关键创新不是 simple multi-query augmentation，而是 `multi-query consistency optimization`。

## 10. 这篇工作的亮点

### 10.1 问题定义抓得准

它不是泛泛地说 latent reasoning 不够强，而是把问题定位为:

- latent 缺少细粒度语义
- 同一 latent 难以支撑语义变化后的多个 query

这个问题定义是清楚且有针对性的。

### 10.2 方法结构合理

两阶段设计很顺:

- Stage 1 先塑形表示
- Stage 2 再对齐跨 query 使用方式

这比直接把各种奖励全塞进一次 RL 里更可控。

### 10.3 Benchmark 设计很有辨识度

`SV-QA` 的价值挺高，因为它专门测 “same region, multiple semantic views”。这比普通 VQA accuracy 更能看出 latent 是否真的承载了共享区域语义。

## 11. 局限与问题

### 11.1 语义监督依赖外部强模型自动构造

他们的 semantic profile、multi-query 数据、SV-QA 的生成都高度依赖 `Qwen3-VL-235B`。这意味着方法上存在数据构造成本，也带来 teacher bias。

### 11.2 semantic embedding 的形式仍较粗

虽然用了 4096 维 attribute embedding，但本质上还是把结构化语义压成单个 dense vector。这样能否充分保留复杂关系，仍值得怀疑。

### 11.3 标准 benchmark 并非全面提升

在 `GQA`、`AI2D` 上不如 LVR，说明语义增强 latent 并不是对所有任务都天然更优。它可能更适合:

- 区域属性
- 空间关系
- 多视角问题一致性

而未必在所有 general VQA 任务上无差别获益。

### 11.4 外部 judge 带来额外系统复杂度

Stage 2 里 answer correctness 由 `Qwen3-Max` 评估，这增加了 RL 管线复杂度，也让训练依赖外部 judge 质量。

## 12. 对 VLM 方向的启发

如果把这篇放到 `VLM` 方向里，我觉得它最值得记住的是 4 点:

1. `latent reasoning` 的提升空间不只在更长 latent、更强 policy，也在 latent 的监督形式。
2. 对视觉推理来说，`attribute/state/spatial/text` 这类显式语义注入很重要，尤其是 region-centric 任务。
3. 如果要做 latent RL，训练目标不应该只看单题 correctness，还应该看 `cross-query latent consistency`。
4. 很多 benchmark 只测单问题 accuracy，不足以暴露 latent 的语义脆弱性；`same-region multi-query` 是一个很有价值的评测范式。

## 13. 可复用的研究想法

### 13.1 把 semantic latent 从单向量扩成结构化 latent

当前的 $z_{\text{sem}}$ 是单 token latent。后续可以考虑把它扩成:

- attribute slots
- relation slots
- text-content slot

也许比单向量更适合承载复杂视觉语义。

### 13.2 把 multi-query consistency 扩展到更广义的 semantic invariance

不只可以对同一区域的两个问题做一致性，还可以对:

- paraphrase 问法
- different reasoning depths
- multilingual queries
- region + counterfactual perturbation

做统一 consistency 约束。

### 13.3 用到 OCR-heavy 或 Chart 任务

这篇方法对 `ChartQA`、`VizWiz` 提升明显，说明 semantic-enriched latent 可能特别适合:

- OCR + region grounding
- chart region understanding
- document element reasoning

这和你前面读的 OCR / Image2Code / chart 方向其实也能接起来。

## 14. 总结评价

我对这篇的判断是: 它不是在 latent reasoning 上做一个很重的架构创新，而是抓住了一个更本质的问题，即 `latent 到底该编码什么`。SLVR 的贡献主要体现在三个层面:

- 用 attribute-level supervision 提高 latent 的语义密度
- 用 same-region multi-query 数据把 latent 的可迁移性暴露出来
- 用 M-GRPO 明确优化 latent 在不同 query 下的一致性

如果后面 latent visual reasoning 继续发展，这篇工作里最可能被继承的，不一定是具体的公式，而是这个范式:

`从“让 latent 能答题”转向“让 latent 成为稳定、可复用、语义一致的视觉语义载体”。`
