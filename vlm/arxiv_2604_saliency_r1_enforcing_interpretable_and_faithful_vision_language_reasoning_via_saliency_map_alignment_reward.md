# Saliency-R1: Enforcing Interpretable and Faithful Vision-language Reasoning via Saliency-map Alignment Reward

## 1. 基本信息
- arXiv ID: 2604.04500
- URL (abs/pdf): [abs](https://arxiv.org/abs/2604.04500) / [pdf](https://arxiv.org/pdf/2604.04500.pdf)
- 主题目录: vlm
- 总结粒度: detailed
- 作者: Shizhan Gong, Minda Hu, Qiyuan Zhang, Chen Ma, Qi Dou (CUHK / CityU Hong Kong)
- 会议: CVPR 2026
- 代码: https://github.com/peterant330/Saliency_R1

## 2. 背景与动机（详细）

### 2.1 任务定义与应用场景
VLM 在多模态推理中取得显著进展，但其可信度面临两个核心挑战：
1. **过度依赖文本线索**：视觉信号在推理中扮演的角色相对较小
2. **幻觉问题**：模型生成未被图像支持的内容

当前 VLM 推理使用 Chain-of-Thought (CoT) 或类似 DeepSeek-R1 的推理机制生成自然语言解释 (NLE)，但研究表明这些解释与最终答案之间可能存在不一致性——"思考"过程未必忠实反映解题过程。

### 2.2 核心洞察
**不同思维过程关注图像的不同区域，即使得出相同答案**：
- 有的推理真正关注与问题相关的区域
- 有的关注不相关物体或背景
- 后者本质上是对视觉信息的误用或幻觉

本文的核心目标是：**确保 VLM 的推理过程真正"看"到了正确的图像区域**。

![动机图：不同思维过程关注不同区域](assets/arxiv_2604_saliency_r1_enforcing_interpretable_and_faithful_vision_language_reasoning_via_saliency_map_alignment_reward/teaser.png)

## 3. 方法与框架（详细）

### 3.1 总体框架

Saliency-R1 包含三个核心组件：

1. **Token-level Saliency Map 生成**：通过 logits 分解高效生成显著性图
2. **Attention Rollout 聚合**：以 thinking tokens 为瓶颈，追踪视觉信息如何流向答案
3. **GRPO + Saliency Alignment Reward**：用显著性图与人工标注 bounding box 的对齐度作为奖励信号

![方法框架](assets/arxiv_2604_saliency_r1_enforcing_interpretable_and_faithful_vision_language_reasoning_via_saliency_map_alignment_reward/framework.png)

### 3.2 Token-level Saliency Map 生成（Logits 分解）

基于 Transformer 架构，将 token logits 分解为各 context token 的一阶贡献：

$$c_p = \sum_{l=1}^{L}\sum_{j=1}^{H}\alpha_{i,j,p}^l \mathbf{W}_{o,j}^l \mathbf{W}_{v,j}^l \mathbf{h}_p^{l-1} \mathbf{E}_u$$

其中 $\alpha_{i,j,p}^l$ 是 attention 权重，$\mathbf{W}_o$ 和 $\mathbf{W}_v$ 是输出和值投影矩阵。

**关键实现细节**：
- 仅选取视觉 token 对应的 $c_p$ 值，按 patch 顺序重排
- 应用 ReLU 忽略负贡献
- 不需要额外的前向或反向传播——attention 权重和 KV cache 中的值向量已天然可用
- 虽然忽略间接贡献（残差连接、FFN 部分），但文献表明直接贡献占主导地位

**计算效率**：生成显著性图的额外开销几乎为零，使其能无缝集成到训练 pipeline 中。

### 3.3 Attention Rollout：Thinking Tokens 作为瓶颈

推理模型先生成思维链 (thinking tokens)，再生成答案。本文假设：
- Thinking 过程从图像中提取信息
- 答案基于 thinking 过程的总结

为了可视化这一流程，使用 attention rollout 追踪视觉信息如何通过 thinking tokens 流向答案：

$$\tilde{\mathcal{A}}_{va}^{l,h} = \mathcal{A}_{vt}^{l,h} \cdot \mathcal{A}_{ta}^{l,h}$$

其中 $\mathcal{A}_{vt}$ 是视觉→thinking 的注意力矩阵，$\mathcal{A}_{ta}$ 是 thinking→answer 的注意力矩阵。

**关键设计**：
- **不对列做归一化**：某些 answer token（如介词）接收很少的 thinking 贡献，某些 thinking token 接收很少的视觉贡献，这些 token 自然在 holistic saliency map 中影响更小
- **直接求和** per-token map 得到最终的 holistic saliency map
- 低置信度的 answer token（logits 小）贡献也更少

### 3.4 GRPO + Saliency Alignment Reward

**Alignment Score** 定义为 saliency map 在 bounding box 内的质量占比：

$$\text{Alignment\_Score} = \frac{\sum_{i \in \text{Bounding Box}} \text{Saliency\_Score}(i)}{\sum_{i \in \text{Image}} \text{Saliency\_Score}(i)}$$

**完整的 Reward 设计**：

$$\mathcal{R}_{\text{overall}} = \mathcal{R}_{\text{accuracy}} + \mathcal{R}_{\text{format}} + \mathcal{R}_{\text{saliency}}$$

| Reward | 含义 | 取值 |
|--------|------|------|
| $\mathcal{R}_{\text{accuracy}}$ | LLM-as-judge 判断答案正确性 | 0 或 1 |
| $\mathcal{R}_{\text{format}}$ | 输出格式遵循（think 标签 + 答案格式） | 0 或 1 |
| $\mathcal{R}_{\text{saliency}}$ | Saliency map 与 bbox 的对齐度 | [0, 1] 连续值 |

使用 GRPO 优化，每问题采样 G=8 个 rollout，用组内相对优势作为 advantage。

## 4. 实验分析（详细）

### 4.1 实验设置

**训练数据**：
- SFT 冷启动：Saliency-R1-cold（Vision-R1-cold 的子集），272,881 样本
- GRPO：Saliency-r1-8k（Visual-CoT 子集），8,080 样本，含 bounding box 标注

**实现细节**：
- Base model: Qwen2.5-VL-3B/7B
- SFT: llama-factory, 2 epochs, lr=5e-5, LoRA rank=128
- GRPO: TRL, 3 epochs, lr=1e-5, LoRA rank=8 (Q/V 投影层), batch=128, rollout=8
- LLM judge: GPT-4o-mini
- 硬件：8× A6000 GPU

### 4.2 Saliency Map 忠实度评估

使用 deletion/insertion 测试评估 saliency map 的因果忠实度：

| 方法 | Deletion ↓ (5%/15%/30%) | Insertion ↑ (5%/15%/30%) |
|------|------------------------|-------------------------|
| CAM | 86.19/82.44/78.34 | -0.10/2.47/28.02 |
| Grad-CAM | 86.19/82.44/78.34 | -0.10/2.47/28.02 |
| ATTN-LRP | 76.42/64.22/52.92 | -0.06/12.10/45.67 |
| TAM | 83.91/79.33/73.29 | 0.02/23.54/45.24 |
| **Ours** | **70.96/59.45/50.34** | 0.06/14.39/45.22 |

COCO Caption 上：deletion 分数比第二优方法低 5.46%/4.77%/2.57%（5%/15%/30%），说明即使仅考虑直接贡献，方法仍能忠实捕捉视觉 patch 的相对重要性。

### 4.3 VQA 主结果

| 方法 | MMMU-Pro | MMBench | POPE | MME | MME-RW | MMStar | ChartQA | IllusionVQA | ScienceQA | SalBench |
|------|----------|---------|------|-----|--------|--------|---------|-------------|-----------|----------|
| GPT-4o | 54.0 | 83.4 | 85.6 | 2329 | 45.2 | 63.9 | 85.7 | 49.1 | 90.1 | 89.2 |
| InternVL2.5-8B | 38.2 | 84.6 | 90.6 | 2344 | 58.1 | 63.2 | - | - | 98.4 | - |
| Qwen2.5-VL-7B | 36.2 | 82.8 | 86.7 | 2302 | 58.7 | 62.4 | 84.0 | 37.5 | 88.2 | 49.1 |
| + SFT | 35.5 | 78.8 | 87.6 | 2372 | 60.6 | 60.1 | 87.2 | 33.6 | 93.8 | 57.1 |
| **+ Saliency-R1** | **37.6** | **81.8** | **88.1** | **2385** | **62.9** | **62.6** | **88.2** | **38.4** | **94.3** | **63.7** |

核心发现：
- Saliency-R1-7B 在 POPE 上提升 +1.4%，ChartQA 上提升 +4.2%
- 改进在 3B 和 7B 模型上一致
- 在 MME、MME-RealWorld、ScienceQA 上超越商业模型（GPT-4o、Claude-3.5 Sonnet）
- SFT 单独也有提升但远小于 Saliency-R1

### 4.4 可解释性定量评估

在 saliency-r1-8k 测试集上使用 pointing game 评估：

| 方法 | Energy-PG (%) ↑ | PG (%) ↑ |
|------|----------------|----------|
| Qwen2.5-VL-7B | 19.26 | 24.49 |
| Saliency-R1-CI (仅 SFT) | 23.65 | 29.04 |
| Vision-R1 | 25.32 | 32.58 |
| **Saliency-R1-7B** | **30.08** | **38.63** |

Saliency-R1 的 Energy-PG 比 base model 提升 **10.82%**，PG 提升 **14.14%**。

### 4.5 消融实验

| 变体 | Avg (9 benchmarks) | MME |
|------|-------------------|-----|
| Saliency-R1 (完整) | 最高 | 高 |
| w/o SFT (Saliency-R1-zero) | 显著下降 | 下降 |
| w/o saliency reward (= Vision-R1) | -1.5% | 较低 |
| 仅 saliency reward (Saliency-R1-pure) | -0.6% | **最高** |
| w/o attention rollout (Saliency-R1-attn) | 下降 | 下降 |
| 直接聚合 thinking tokens (Saliency-R1-think) | 下降 | 下降 |

关键发现：
- **SFT 冷启动至关重要**：直接从 base 开始 GRPO 导致严重退化
- **Saliency reward 的正面效果**：比 Vision-R1（无 saliency reward）高 1.5%
- **仅用 saliency reward 也有效**：比完整版仅低 0.6%，且在 MME 上更好，可省去 LLM judge 成本
- **Attention rollout 不可省略**：比直接聚合 thinking tokens 的效果更好
- Saliency reward 与 accuracy 有正相关：关注正确区域 → 更可能给出正确答案

### 4.6 鲁棒性分析

- **噪声鲁棒性**：向 POPE/MME 图像注入不同强度高斯噪声，Saliency-R1 优于 base model
- **OOD 泛化**：在医学 VQA（VQA-RAD）上也有提升
- **多跳推理**：在 ReasonVQA（需要关注多个图像区域）上优于 base model
- **反事实分析**：Saliency-R1 在前景像素注入噪声时性能下降更大，背景噪声时下降更小 → 确实更依赖前景

### 4.7 定性分析

SFT 模型（Saliency-R1-CI）的推理过程往往**不必要地冗长**，包含大量重复和无关内容，稀释了注意力。Saliency-R1 的推理更**简洁有针对性**，注意力集中在关键区域。

## 5. 关键图表（本地资源）

- **动机图**: `assets/.../teaser.png`
- **方法框架**: `assets/.../framework.png`
- **主结果 + 定性对比**: `assets/.../result.jpg`
- **消融实验**: `assets/.../ablation.png`

## 6. 优点与局限

### 优点
1. **高效 saliency map 生成**：基于 logits 分解，无需额外前向/反向传播，可无缝集成到训练 pipeline
2. **Attention rollout 机制**：以 thinking tokens 为瓶颈追踪视觉信息流，揭示推理过程与视觉上下文的对齐
3. **Saliency alignment reward 设计巧妙**：用 bounding box 对齐度作为连续 reward，与 accuracy 正相关，可单独使用
4. **可解释性与性能双赢**：不仅提升任务性能，还显著增强推理过程的可解释性（PG +14.14%）
5. **Saliency-R1-pure 的发现**：仅用 saliency reward 就接近完整版效果，降低了 LLM judge 依赖
6. **多维度提升**：在幻觉检测（POPE）、OCR（ChartQA）、一般理解（MME）等 benchmark 上一致提升
7. **开源代码**

### 局限
1. **SFT 数据规模受限**：某些指标在 SFT 后反而下降，可通过收集更多样化的冷启动数据改善
2. **Bounding box 标注较粗**：更细粒度的分割标注（如 SAM 3）可能进一步提升性能
3. **仅在较小 base model 上评估**：3B/7B 模型，未测试更大规模
4. **Saliency map 仅考虑直接贡献**：忽略间接贡献（残差连接、FFN），理论上不是最忠实的
5. **Alignment score 用 recall 而非 IoU**：只考虑 bbox 内的 saliency 比例，未惩罚 bbox 外的过度激活

## 7. 复现与工程实现要点

1. **代码**：github.com/peterant330/Saliency_R1
2. **训练流程**：SFT 冷启动（Vision-R1-cold 子集，272K 样本）→ GRPO（saliency-r1-8k，8K 样本）
3. **SFT 超参**：2 epochs, lr=5e-5, Cosine scheduler, warmup 0.05, LoRA rank=128
4. **GRPO 超参**：3 epochs, lr=1e-5, batch=128, rollout=8, LoRA rank=8 (Q/V), KL=0.001
5. **Saliency map 实现**：
   - 从 attention weights 和 KV cache 中提取，无需额外计算
   - ReLU 过滤负贡献
   - 考虑 RMSNorm 的线性性质，直接对各 token 贡献应用 normalization
6. **LLM judge**：GPT-4o-mini 用于 accuracy reward
7. **数据**：SFT 数据来自 LLaVA-CoT + Mulberry，GRPO 数据来自 Visual-CoT

## 8. 对当前项目的启发与可落地尝试

1. **Saliency map 作为可解释性工具可直接复用**：logits 分解方法零额外成本，可集成到任何 VLM 的推理可视化中
2. **Saliency alignment reward 是一种新的 process-level reward**：不需要逐步验证，用粗粒度的人工标注（bounding box）就能提供有意义的训练信号，比 outcome-only reward 更丰富
3. **Saliency-R1-pure 的可行性**：仅用 saliency reward 就足够，这意味着可以省去 LLM judge 的成本，大幅降低训练开销
4. **Thinking tokens 作为瓶颈的 attention rollout**：揭示推理过程中视觉信息的流动，可作为分析工具用于理解任何 CoT VLM 的推理质量
5. **SFT 冷启动 + GRPO 的标准范式**：与 SkillRL、OPD 等论文一致，冷启动后做 RL 是当前 agent/VLM 训练的通用最佳实践
6. **Bounding box 标注成本低**：相比精细分割标注，bounding box 更易获取，证明了用粗粒标注引导细粒注意力的可行性
7. **与 MemOCR 的联系**：两者都关注视觉信息的高效利用——MemOCR 用视觉布局控制信息密度，Saliency-R1 用 saliency alignment 确保模型关注正确区域
8. **可扩展到更精细标注**：用 SAM 等工具提供的分割标签替代 bbox 可能进一步提升效果
