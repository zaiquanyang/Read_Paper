# Persistent Visual Memory: Sustaining Perception for Deep Generation in LVLMs

## 1. 基本信息
- arXiv ID: 2605.00814
- URL (abs/pdf): [abs](https://arxiv.org/abs/2605.00814) / [pdf](https://arxiv.org/pdf/2605.00814.pdf)
- 主题目录: vlm
- 总结粒度: detailed
- 发表会议: NeurIPS 2026
- 作者团队: Shanghai AI Laboratory & 上海交通大学 & 香港中文大学 & 南京大学 & 同济大学 & 武汉大学
- 方法名: **PVM**（Persistent Visual Memory）
- 代码: [github.com/huaixuheqing/PVM](https://github.com/huaixuheqing/PVM)

## 2. 背景与动机（详细）

### 2.1 核心问题：视觉信号稀释（Visual Signal Dilution）

自回归 LVLM（如 Qwen3-VL、InternVL）在多模态任务上取得显著进步，但在**深层生成**（deep generation）场景——长对话、多步推理链——中面临一个根本性问题：**视觉信号稀释**。

在标准自回归框架中，视觉 token 作为固定的前驱 token 一次性注入序列开头，而文本 token 在生成过程中不断自我补充。随着文本历史累积，注意力的 Softmax 归一化机制将概率质量重新分配到越来越多的 token 上，导致视觉信号逐渐衰减。

### 2.2 理论分析：幂律衰减

本文对这一现象进行了严格的数学推导：

**定义视觉注意力质量**（Visual Attention Mass）：
$$\Omega_{\mathcal{V}}(t) = \frac{Z_{\mathcal{V}}}{Z_{\mathcal{V}} + Z_{\mathcal{T}}}$$

其中 $Z_{\mathcal{V}}$ 是视觉 token 的未归一化注意力总和（有上界 $\beta = M \cdot \exp(s_{\max})$），$Z_{\mathcal{T}}$ 是文本历史 token 的总和（随 $t$ 增长）。

**定理（Visual Signal Dilution）**：在持续文本注意力条件下（$Z_{\mathcal{T}}/t \ge \mu > 0$），视觉注意力质量服从幂律衰减：
$$\Omega_{\mathcal{V}}(t) \le \frac{\beta}{\beta + \mu \cdot t} = \mathcal{O}(t^{-1})$$

**两个阶段**：
1. **幂律稀释阶段**：$Z_{\mathcal{T}}$ 随 $t$ 线性增长，与固定 $Z_{\mathcal{V}}$ 竞争概率质量
2. **高幅饱和陷阱**：$Z_{\mathcal{T}}$ 最终饱和于 $Z_{\mathcal{T}}^{sat} \gg \mathbb{E}[Z_{\mathcal{V}}]$，系统进入 **Low-Attention Equilibrium**——视觉信号被文本先验以数量级压制

### 2.3 实验验证："盲画家"压力测试

使用 Qwen3-VL-8B-Instruct 在 COCO 2017 验证集上设计 **"Blind Painter"** 压力测试——要求模型以极端细节文本重构图像，迫使每个生成步骤都依赖视觉检索。

关键发现：
- 对数尺度分析确认 $\Omega_{\mathcal{V}}$ 紧密遵循 $\mathcal{O}(t^{-1})$ 衰减轨迹
- 衰减在**中间层（Layer 8-27）**最为显著——这些层恰恰是多模态推理的主要位置
- Text-to-Vision Ratio（TVR）从线性增长阶段过渡到饱和平台，文本质量超视觉质量近两个数量级

### 2.4 现有方法的不足：Visual Injection 的代价

以往方法（重新注入视觉 token、融合处理特征等）直接将视觉检索信号注入串行自回归路径。虽然短期内有效，但会**干扰语言推理的连贯性**——通过扰动逐步演化的语义状态来强化视觉存在，造成"强化视觉 vs 保持逻辑精度"的权衡。

### 2.5 两条设计洞察

- **Insight 1**：由于自回归流倾向于稀释视觉信号，应建立**与主干推理隔离的并行记忆通路**
- **Insight 2**：为防止文本在分区函数中主导，检索机制需要**独立的注意力归一化**，完全限定在视觉域内

## 3. 方法与框架（详细）

### 3.1 总体架构

![PVM 框架](./assets/arxiv_2605_persistent_visual_memory_sustaining_perception_for_deep_generation_in_lvlms/PVM.png)
*PVM 框架总览：PVM 与冻结的 FFN 并行，通过瓶颈适配器（Projection → Cross-Attn → FFN → Restoration）检索原始视觉嵌入，经可学习门控注入主流*

PVM 作为**并行分支**集成到 Transformer Decoder Block 中，与 FFN 子层并行：

- **推理路径（原始 FFN）**：冻结 FFN 处理隐藏状态，访问预训练静态知识和逻辑模式
- **视觉检索路径（PVM）**：同一隐藏状态作为 Query，检索原始视觉嵌入

### 3.2 PVM 计算过程

三阶段计算：

**1. 投影（Projection）**：将输入隐藏状态和视觉特征降维到低维潜在空间 $d' < d$
$$\mathbf{x}_{\mathrm{lat}} = \mathbf{x} \mathbf{W}_{\mathrm{down}}^{\mathrm{txt}}, \quad \mathbf{V}_{\mathrm{lat}} = \mathbf{V}_{\mathrm{img}} \mathbf{W}_{\mathrm{down}}^{\mathrm{vis}}$$

**2. 潜在检索（Latent Retrieval）**：Cross-Attention，Query 为投影后的隐藏状态，Key/Value 为投影后的视觉特征
$$\mathbf{h}_{\mathrm{attn}} = \mathrm{CrossAttn}(Q=\mathbf{x}_{\mathrm{lat}}, K=\mathbf{V}_{\mathrm{lat}}, V=\mathbf{V}_{\mathrm{lat}})$$
$$\mathbf{h}_{\mathrm{lat}} = \mathbf{h}_{\mathrm{attn}} + \mathrm{FFN}_{\mathrm{lat}}(\mathrm{RMSNorm}(\mathbf{h}_{\mathrm{attn}}))$$

**3. 恢复（Restoration）**：投影回原始高维空间
$$\mathbf{h}_{\mathrm{pvm}} = \mathbf{h}_{\mathrm{lat}} \mathbf{W}_{\mathrm{up}}$$

### 3.3 门控融合与选择性激活

$$\mathbf{y} = \mathbf{x} + \mathbf{h}_{\mathrm{ffn}} + (\lambda \cdot \mathbf{h}_{\mathrm{pvm}}) \cdot \mathcal{M}_{\mathrm{txt}}$$

- **可学习标量门控 $\lambda$**：初始化为 0，确保预训练能力不被破坏
- **Visual Silencing Mask $\mathcal{M}_{\mathrm{txt}}**：二值指示器，仅对文本 token 激活，防止视觉 token 的冗余自引用

### 3.4 理论保证：结构化缓解视觉稀释

**定理（Structural Mitigation of Visual Dilution）**：在固定局部隐藏状态假设下，PVM 的检索表示与文本历史长度 $t$ 在分区函数上解耦：
$$\frac{\partial \|\mathbf{h}_{\mathrm{pvm}}\|}{\partial t} = 0$$

这是因为 PVM 的分区函数 $Z_{\mathrm{pvm}}$ 仅在固定的视觉集合 $\mathcal{V}$ 上求和，变量 $t$ 既不出现在分数计算中也不出现在求和索引中。

### 3.5 关键设计选择

- **注入层选择**：8B 模型注入 Layer {8, 16, 24}，4B 模型注入 Layer {5, 11, 17}——采用 Strided 策略覆盖网络全深度
- **瓶颈维度**：$d' = 512$（消融验证最优，增大反而因数据-容量不匹配退化）
- **参数开销**：仅增加 27.92M 参数（~0.32% 的 8B 模型大小）

## 4. 实验分析（详细）

### 4.1 实验设置

**基座模型**：Qwen3-VL-Instruct（4B 和 8B）

**训练数据**：
- SFT 阶段：526K 样本（来自 OpenMMReasoner-SFT）
- GRPO 阶段：3.6K 复杂推理 query（来自 MMK12、ThinkLite-VL-hard、ViRL39K、We-Math2.0-Pro）

**训练流程**：
- Stage I（视觉记忆对齐，SFT）：冻结 backbone，仅优化 PVM 模块和门控标量
- Stage II（策略精炼，GRPO）：解冻 LLM backbone 和 PVM（Vision Encoder 保持冻结）

**评估 Benchmark**：MMMU、MMBench-CN/EN、MMStar、MMT、MathVerse、MathVision、AI2D（共 8 个）

**基线方法**：
- 内部基线：Qwen3-VL 原始、SFT、LoRA-SFT、SFT+GRPO
- Visual Injection：MemVR、ICoT、CoMemo
- RL 推理模型：Euclid-8B、PEARL-8B、OneThinker-8B

### 4.2 主结果

| 模型 | General Avg. | Math&Sci Avg. | **Overall Avg.** |
|:--|:--:|:--:|:--:|
| Qwen3-VL-8B-Instruct | 71.1 | 59.4 | 66.7 |
| SFT | 71.0 | 61.3 | 67.4 |
| LoRA-SFT | 72.5 | 59.2 | 67.5 |
| SFT + GRPO | 72.0 | 62.0 | 68.3 |
| LoRA-SFT + GRPO | 72.3 | 61.8 | 68.4 |
| CoMemo | 73.9 | 59.3 | 68.4 |
| Euclid-8B | 73.3 | 63.2 | 69.5 |
| PVM-8B (SFT) | 75.2 | 63.0 | **70.6** |
| **PVM-8B (SFT + GRPO)** | **75.6** | **64.6** | **71.5** |

**核心发现**：

1. **PVM-8B (SFT+GRPO) 达到 71.5% Overall，超越基线 +4.8pp**，在所有方法中最佳
2. **一致性跨规模提升**：4B 模型上也取得 +4.4pp 提升（64.0→68.4）
3. **PVM 仅 SFT 就达到 70.6%**，已超越所有 Visual Injection 方法和 RL 推理模型
4. SFT 反而损害 MMT（-5.9pp），但 PVM-SFT 将退化抑制到仅 -1.6pp——PVM 有效缓解微调过程中的感知退化

### 4.3 长度鲁棒性分析

| 输出长度组 | 基线 (MathVerse$^V$) | PVM | 相对提升 |
|:--|:--:|:--:|:--:|
| Very Short | - | - | +6.1% |
| Short | - | - | +12.8% |
| Medium | - | - | +18.5% |
| **Long** | - | - | **+27.3%** |

**PVM 的增益随序列长度单调递增**——越深层的推理链，持续视觉检索越不可或缺。这直接验证了 PVM 对视觉信号稀释的结构化缓解。

### 4.4 LogitLens 机制分析

使用 LogitLens 技术测量中间层表示与最终输出分布之间的 KL 散度（prediction readiness）：
- 基线表现出渐进的 KL 下降
- PVM 建立了显著更低的轨迹，在注入层（Layer 8）之后出现明显的"Improvement Gap"
- 这证实 PVM 有效地**短路了信息收集过程**——通过将视觉检索卸载到并行分支，backbone 加速从感知到推理的转换

### 4.5 消融实验

**检索源消融**：

| 检索源 | MathVerse | MathVision | AI2D | Avg. |
|:--|:--:|:--:|:--:|:--:|
| 基线（无 PVM） | 52.9 | 45.4 | 79.8 | 59.4 |
| 处理后隐藏状态 | 27.9 | 14.1 | 58.2 | **33.4** |
| **原始视觉嵌入（Ours）** | **57.5** | **50.7** | **80.8** | **63.0** |

用处理后的隐藏状态替代原始视觉嵌入导致**灾难性崩溃**（59.4→33.4）——重新整合文本主导的隐藏状态会产生破坏性的自反循环。

**注入层策略消融**：

| 策略 | Layers | General | Reasoning | Avg. |
|:--|:--|:--:|:--:|:--:|
| Peak Attention | 13, 17, 18 | 72.9 | 60.9 | 68.4 |
| Max Decay | 14, 19, 22 | 74.2 | 61.2 | 69.3 |
| **Strided (Ours)** | **8, 16, 24** | **75.2** | **63.0** | **70.6** |

Strided 策略优于 Peak Attention 和 Max Decay——跨网络全深度的全局覆盖比在峰值/衰减最剧烈层集中注入更有效。

**潜在维度消融**：

| 维度 $d'$ | General | Reasoning | Avg. |
|:--|:--:|:--:|:--:|
| **512 (Ours)** | **75.2** | **63.0** | **70.6** |
| 1024 | 73.8 | 61.4 | 69.2 |
| 2048 | 74.7 | 61.6 | 69.8 |

增大维度反而退化——数据-容量不匹配：更大参数空间需要成比例的更多监督信号。

**等参数控制**：

| 模型 | Avg. |
|:--|:--:|
| SFT + GRPO | 68.3 |
| MLP (SFT+GRPO) | 69.0 |
| **PVM-8B (SFT+GRPO)** | **71.5** |

等参数 MLP 基线仅提升 +0.7pp，而 PVM 提升 +3.2pp——增益来自主动视觉检索机制而非单纯增加参数。

### 4.6 推理开销

| 指标 | Baseline (Qwen3-VL) | PVM-Enhanced | Delta |
|:--|:--:|:--:|:--:|
| Throughput | 41.18 tokens/s | 39.28 tokens/s | -4.61% |
| TPOT | 24.28 ms | 25.46 ms | +1.18 ms |

参数高效瓶颈设计使推理开销极小——仅增加 1.18ms/token。

## 5. 关键图表（本地资源）

### 5.1 视觉记忆机制对比
![Intro](./assets/arxiv_2605_persistent_visual_memory_sustaining_perception_for_deep_generation_in_lvlms/pvm_intro.png)
*三种视觉记忆机制对比：Standard LVLM 退化为视觉稀释、Injection 方法造成串行干扰、PVM 建立独立检索通路*

### 5.2 PVM 框架
![PVM](./assets/arxiv_2605_persistent_visual_memory_sustaining_perception_for_deep_generation_in_lvlms/PVM.png)
*PVM 完整框架：并行于冻结 FFN 的瓶颈适配器，通过 Silencing Mask 选择性激活*

### 5.3 视觉信号幂律衰减
![衰减趋势](./assets/arxiv_2605_persistent_visual_memory_sustaining_perception_for_deep_generation_in_lvlms/attention_lineplot_log.png)
*对数尺度分析确认 $\Omega_{\mathcal{V}}$ 紧密遵循 $\mathcal{O}(t^{-1})$ 幂律衰减*

### 5.4 文本主导演化
![TVR](./assets/arxiv_2605_persistent_visual_memory_sustaining_perception_for_deep_generation_in_lvlms/tvr_growth_analysis.png)
*Text-to-Vision Ratio 从线性增长到饱和平台——视觉信号被文本先验以数量级压制*

### 5.5 层级注意力热力图
![Heatmap](./assets/arxiv_2605_persistent_visual_memory_sustaining_perception_for_deep_generation_in_lvlms/attention_heatmap.png)
*36 层 Qwen3-VL 的视觉注意力质量热力图——中间层（8-27）衰减最显著*

### 5.6 层级分布
![Layer Profile](./assets/arxiv_2605_persistent_visual_memory_sustaining_perception_for_deep_generation_in_lvlms/layer_profile.png)
*各层平均视觉注意力质量分布——"Rise-Peak-Decay"模式指导注入层选择*

### 5.7 长度-增益相关性
![长度分析](./assets/arxiv_2605_persistent_visual_memory_sustaining_perception_for_deep_generation_in_lvlms/analysis_length.png)
*相对增益随 token 长度单调递增——"Long"组达 +27.3%*

### 5.8 LogitLens 收敛分析
![LogitLens](./assets/arxiv_2605_persistent_visual_memory_sustaining_perception_for_deep_generation_in_lvlms/logitlens_comparison_pro.png)
*PVM 加速内部预测收敛——注入层后出现明显 Improvement Gap*

## 6. 优点与局限

### 优点

1. **严谨的理论分析**：首次从注意力分区函数角度形式化推导视觉信号稀释的幂律衰减，并证明 PVM 的结构化缓解机制
2. **架构设计优雅**：并行于 FFN 的轻量级瓶颈适配器，不修改 backbone 结构，仅增加 0.32% 参数
3. **显著且一致的实验增益**：8B 模型 +4.8pp、4B 模型 +4.4pp，全面超越 Visual Injection 方法和 RL 推理模型
4. **长度鲁棒性突出**：长序列场景下增益达 +27.3%，直接验证了对视觉稀释的缓解效果
5. **推理开销极小**：TPOT 仅增加 1.18ms（+4.6%），参数高效设计确保实用性
6. **消融实验严谨**：检索源、注入层、潜在维度、等参数控制等多维度消融验证了设计选择的合理性

### 局限

1. **固定局部查询假设**：理论保证依赖固定局部隐藏状态假设，全局自回归动态下的精确不变性未完全建模
2. **仅验证 Qwen3-VL**：虽然设计理论上 backbone 无关，但实验仅在 Qwen3-VL（4B/8B）上验证
3. **静态视觉上下文**：当前仅处理固定图像输入，未扩展到动态流式输入（如长视频理解）
4. **注入层选择经验性**：Strided 策略优于 Peak/Max Decay，但最优位置的选择仍有经验成分
5. **仅处理视觉稀释**：未考虑其他视觉退化因素（如编码器信息损失、投影层瓶颈等）

## 7. 复现与工程实现要点

1. **基座模型**：Qwen3-VL-Instruct（4B/8B）
2. **PVM 注入层**：8B → {8, 16, 24}，4B → {5, 11, 17}
3. **瓶颈维度**：$d' = 512$
4. **门控初始化**：$\lambda = 0$
5. **训练数据**：SFT 526K（OpenMMReasoner-SFT）+ GRPO 3.6K（MMK12 + ThinkLite-VL-hard + ViRL39K + We-Math2.0-Pro）
6. **SFT 超参**：AdamW, lr=1e-4, Cosine, warmup=0.1, batch=64, ZeRO-2
7. **GRPO 超参**：AdamW, lr=1e-6, Constant, group_size=8, KL_coef=0, batch=64, ZeRO-3
8. **硬件**：8×H200-141GB, FlashAttention-2, gradient checkpointing
9. **代码**：[github.com/huaixuheqing/PVM](https://github.com/huaixuheqing/PVM)

## 8. 对当前项目的启发与可落地尝试

1. **"并行记忆通路"的通用架构原则**：PVM 证明了在自回归流中建立与主干推理隔离的并行信息检索通路可以有效对抗长序列信息衰减。这一原则可以推广到任何需要在长上下文生成中保持特定信号保真度的场景——例如长期对话 Agent 需要持续访问用户画像或历史事实
2. **注意力分区函数分析框架**：将视觉信号稀释形式化为分区函数中的概率质量竞争问题，这一分析框架可以应用于分析 Agent 在多轮对话中对工具定义、环境描述等静态上下文的遗忘问题
3. **"跨深度全局注入"优于"峰值集中注入"**：Strided 策略（均匀跨网络深度）优于 Peak Attention 和 Max Decay 策略——这提示在设计信息注入/检索机制时，覆盖处理阶段的全局分布比在最显著位置集中更有效
4. **Silencing Mask 的选择性激活设计**：仅对文本 token 触发视觉检索，防止视觉 token 的冗余自引用——这一"仅在需要时激活"的门控设计可以推广到 Agent 的条件性工具调用、选择性记忆检索等场景
5. **GRPO 用于视觉 grounding 强化**：Stage II 使用 GRPO 解冻 backbone 并精炼主动视觉检索策略——这与 GUI-SD 中用 GRPO/OPSD 优化 GUI grounding 的思路一致，可以考虑将 PVM 与 GUI-SD 结合用于长对话 GUI Agent
6. **参数效率 vs 检索质量的权衡**：$d'=512$ 是最优平衡点，增大反而因数据-容量不匹配退化——这提示在设计 Agent 的记忆/检索模块时，瓶颈维度应与可用训练数据规模匹配，而非一味追求更大容量
7. **与 MemRouter 的互补性**：PVM 解决的是"已知视觉信息在长序列中的衰减"问题，而 MemRouter 解决的是"多条记忆之间的路由选择"问题——两者可以组合：PVM 确保被选中记忆的信号保真度，MemRouter 确保选择正确记忆
