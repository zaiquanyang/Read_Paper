# Perceptual Flow Network for Visually Grounded Reasoning

## 1. 基本信息
- arXiv ID: 2605.02730
- URL (abs/pdf): [abs](https://arxiv.org/abs/2605.02730) / [pdf](https://arxiv.org/pdf/2605.02730.pdf)
- 主题目录: vlm
- 总结粒度: detailed
- 发表会议: ICML 2026
- 作者团队: ECNU, SCU, HKUST, SJTU, Ant Group, Shanghai AI Lab
- 方法名: **PFlowNet (Perceptual Flow Network)**
- 代码: 📄 paper-only

## 2. 背景与动机（详细）

### 2.1 任务定义与应用场景
大型视觉语言模型（LVLMs）在复杂视觉推理中面临幻觉和可解释性不足的问题。当前主流做法是利用视觉专家（如 GroundingDINO）的几何先验，通过 RLVR（Reinforcement Learning with Verifiable Reward）将推理过程锚定到视觉证据上。

### 2.2 现有方法的关键不足——"Tunnel Vision"效应
本文通过一个关键探测实验发现：**几何精度最高的专家标注并不等于推理效用最高**。

在 V* Bench 上使用 Qwen2.5-VL 家族进行探测，将原始标注各向同性扩展后送入模型（而非完整图像），测量不同几何先验的推理效用。结果发现：
- 最精确的几何先验（专家标注）反而**不是**最有帮助的
- 适度扩展的区域（IoU 较低但包含更多上下文）效果更好
- 最优几何先验是**高度实例特定的**，无法通过简单启发式策略获得

原因在于：视觉专家最初为物体检测设计，追求严格的几何精度，但这会诱导 **"隧道视野"（tunnel vision）**效应——排斥了全面理解所需的上下文信息，反而降低推理性能。

### 2.3 本文核心切入点
不强制模型对齐有偏差的专家先验，而是让模型学习**自参数化的变分分布** $p_\theta(Z|X)$ 来近似理想感知行为的后验 $P_V(Z|X,Y)$，通过采样自优化的内部分布实现 grounded yet more accurate 的推理。

## 3. 方法与框架（详细）

### 3.1 总体框架：解耦感知与推理

PFlowNet 将视觉推理分解为两个**解耦**的阶段：

```
p_θ(Y, Z | X) = p_θ(Z | X) · p_θ(Y | Z, ⟨X, I_RoI⟩)
```

1. **Flow Generation 阶段**：从模型内在分布 $p_\theta(Z|X)$ 采样 Perceptual Flow
2. **Flow-Guided Reasoning 阶段**：基于生成的 Flow 及对应 RoI 裁剪图进行 self-conditioned 自回归生成

核心框架图见 Figure 4：

![PFlowNet 总览](assets/arxiv_2605_perceptual_flow_network_for_visually_grounded_reasoning/fig4.png)

### 3.2 关键概念：Perceptual Flow

**Perceptual Flow** $Z = (z_0 \rightarrow z_1 \dots z_K)$ 是一个结构化的隐式轨迹，包含两种状态：

| 状态 | 定义 | 格式 |
|------|------|------|
| **Planning State** ($z_0$) | 分解查询、识别视觉候选 | `<analyze>...</analyze>` 包裹的语言序列 |
| **Perceptual States** ($z_{\ge1}$) | 锚定观察链 | `<localize>...</localize>` 包裹，每步 $z_k = \langle r_k, c_k \rangle$（RoI + 描述 caption） |

基于此，采用 **Sub-Trajectory Balance (Sub-TB)** 目标进行分层变分优化：
$$\min_\theta \sum_{i,j} D\big(\mathcal{F}(z_i) \mathcal{T}_F(z_{i:j}) \| \mathcal{F}(z_j) \mathcal{T}_B(z_{j:i})\big)$$

相比 PPO 等 RL 范式，Sub-TB 提供**密集的中间监督**，促进多样化的感知行为。

### 3.3 训练流程

#### 3.3.1 数据构建与 Cold Start

**数据收集原则**：
- **多样化任务**：覆盖细粒度理解和通用场景
- **多样化 RoI**：跨专家交叉标注，保留空间覆盖足够广泛的样本

**Flow 合成**（使用 Gemini3Flash 和 GPT-4o 作为教师）：
1. 对每个带专家 RoI 的样本，随机扩展每个 RoI 以缓解视觉专家的归纳偏差
2. 教师模型识别关键视觉内容 → 生成 Planning State $z_0$
3. 为每个 RoI 生成详细 caption → Perceptual States $z_{\ge1}$

**验证与难度控制**（关键设计）：

| 验证条件 w/o $Z_s$ | 验证条件 w/ $Z_s$ | 决策 | 数据分配 |
|-----|-----|-----|-----|
| $k_{pass}=1$ | — | 拒绝（trivial） | — |
| — | $k_{pass}>1$ | 拒绝（unverified flow） | — |
| $2 \le k_{pass} \le 16$ | $k_{pass}=1$ | 接受 | RFT 数据集 $(X,Y,E)$ |
| $k_{pass}>16$ | $k_{pass}=1$ | 接受 | Cold-start 数据集 $(X,Z_s)$ |

**Cold Start SFT**：对 cold-start 样本最小化交叉熵损失，初始化策略生成 Perceptual Flow 的能力。

#### 3.3.2 变分强化微调（Variational RFT）

**目标函数**：
$$\mathcal{L}_{vRFT}(\theta) = \mathbb{E}_{X,Y,E \sim P_{data}, \{Z\}_{l=1}^L \sim p_\theta(Z|X)} \left[ \sum_{0 \le i \le j \le |Z|} \left( \log \frac{R_\lambda(z_{0:i}\top) \prod_{k=i+1}^j p_\theta(z_k|z_{0:k-1}) p_\theta(\top|z_{0:j})}{R_\lambda(z_{0:j}\top) p_\theta(\top|z_{0:i})} \right)^2 \right]$$

### 3.4 多维奖励设计（核心创新）

对任意子流 $z_{0:k} \subseteq Z$，奖励定义为：

$$\log R(z_{0:k}\top) = \underbrace{\sum_{i=1}^k \log \frac{p_\phi^+(z_i)}{p_\phi^-(z_i)}}_{\text{对比质量项}} + \underbrace{\log p_\phi(Y | z_{0:k}\top, X)}_{\text{推理效用项}}$$

其中：
- $p_\phi^+(z_i) = p_\phi(c_i | I_{r_i})$：裁剪区域的 caption 似然（正例）
- $p_\phi^-(z_i) = p_\phi(c_i | I \setminus I_{r_i})$：互补区域的 caption 似然（负例）

**关键洞察**：对比项本质上是一种**特权信息蒸馏**——在 reverse-KL 意义上，鼓励策略分布更接近裁剪区域的教师分布，同时远离噪声区域的分布。这促进视觉锚定且语义具体的 caption，抑制语言先验诱导的通用描述。

**推理效用项**：衡量 flow 提供的**信息增益**——最大化 $\log p_\phi(Y|z_{0:k}\top, X)$ 倾向于选择对推导目标答案 $Y$ 更有用的感知流。

### 3.5 Vicinal Geometric Shaping（领域几何塑形）

定义 **Chamfer IoU 距离**：
$$d_{IoU}(A,B) = 1 - 0.5 \times (IoU_{A \rightarrow B} + IoU_{B \rightarrow A})$$

引入 $\varepsilon$-领域球 $\mathcal{B}_\varepsilon(E) = \{z_{0:k} | d_{IoU}(r_{1:k}, E) \le \varepsilon\}$，并定义能量权重：
$$\omega_\lambda(z_{0:k}, E) = \exp(-\lambda \cdot \mathbb{I}(z_{0:k} \notin \mathcal{B}_\varepsilon(E)))$$

**塑形后奖励**：
$$R_\lambda(z_{0:k}\top) = R(z_{0:k}\top) \cdot \omega_\lambda(z_{0:k}, E)$$

- 仅惩罚超出领域范围的轨迹，允许领域内的充分探索
- 不同于严格 KL 正则化——更灵活，允许发现高效用感知行为
- $\varepsilon=0.5, \lambda=4.5$ 为最优超参数设置

### 3.6 理论分析

**定理 1（TV 距离界）**：在理想化假设下，PFlowNet 的策略 $p_{\theta^*}(Z|X)$ 与目标后验 $P_V(Z|X,Y)$ 之间的总变分距离存在上界，该界依赖于 $\lambda$、$\varepsilon$、有效维度 $d_{eff}$ 及有效支撑质量 $s_V$。

**极限分析**：
- $\lambda \to 0$：退化为标准 MLE（$D_{TV} \to 1-s_V$）
- $\lambda \to \infty$：退化为专家引导的 RLVR（$D_{TV} \to 1-q$，$q$ 衡量专家偏差）
- $\varepsilon \to 0$：领域收缩为奇点，奖励信号失去信息性
- $\varepsilon$ 超过容忍度 $\sigma$：包含无效区域，稀释几何引导

**定理 2（保证改进）**：对适当校准的 $\lambda^*$ 和 $\varepsilon$，PFlowNet 严格收紧标准 MLE 和专家引导 RLVR 的 TV 距离界。

## 4. 实验分析（详细）

### 4.1 主结果

**基础模型**：Qwen3-VL-8B

#### 通用视觉任务（TreeBench + MME-RealWorld-Lite）

| 模型 | TreeBench Acc | MME-RW Acc |
|------|:---:|:---:|
| Qwen3-VL-8B (Base) | 44.9 | 48.6 |
| TreeVGR | 50.4 | 54.9 |
| DeepEyes | 37.5 | 53.2 |
| DeepEyesV2 | 40.7 | 52.4 |
| Thyme | 38.2 | 54.4 |
| **PFlowNet** | **55.3** | **67.0** |
| $\Delta$ vs. Base | +10.4 | +18.4 |

PFlowNet 在 19 个子任务中的 17 个（89%）创下 SOTA 记录，超越最近竞争者 5.3%（TreeBench）和 12.6%（MME-RW）。

#### 细粒度视觉理解

| 模型 | V* Bench | HR-Bench 4K | HR-Bench 8K | ScreenSpot v2 | ScreenSpot Pro |
|------|:---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-8B | 77.5 | 72.4 | 68.1 | 92.7 | 54.6 |
| Qwen3-VL-32B | 87.4 | 82.1 | 74.8 | — | — |
| **PFlowNet (8B)** | **90.6** | **80.4** | **76.9** | **95.1** | **61.8** |
| $\Delta$ vs. Base | +13.1 | +8.0 | +8.8 | +2.4 | +7.2 |

**关键结论**：PFlowNet (8B) 在 V* Bench 上达到 90.6%，匹配甚至超越 Qwen3-VL-32B (87.4%)。

### 4.2 消融实验

| 设置 | TreeBench | V* | MME-RW |
|------|:---:|:---:|:---:|
| (1) Base Model | 44.9 | 77.5 | 46.0 |
| (2) + SFT | 48.3 | 83.7 | 54.2 |
| (3) + RFT (仅对比项) | 51.5 | 85.3 | 59.5 |
| (4) + RFT (仅推理效用) | 52.8 | 87.4 | 62.8 |
| **(5) + RFT (两者结合)** | **55.3** | **90.6** | **67.0** |
| (6) + RFT (仅正例 $P^+$) | 52.2 | 88.1 | 65.5 |
| (7) PFlowNet w/o Z (仅 RoI) | 54.5 | 89.4 | 66.4 |
| (8) PFlowNet w/o I_RoI (仅 Flow) | 49.2 | 83.8 | 52.1 |

**关键发现**：
- 质量奖励和效用奖励**协同作用**，缺一不可
- 对比项（$P^+/P^-$ vs 仅 $P^+$）中负例信号至关重要
- 消除 Perceptual Flow 导致性能**严重退化**（-6.8 on V*），证明 Flow 不只是定位工具，更是关键的**显式语义锚点**
- 消除 RoI 特征影响相对较小——说明文本 Flow 已充分编码视觉语义

### 4.3 Geometric Shaping 消融

- 过大 $\varepsilon$（0.7）：过度探索进入无效支撑区
- 过小 $\varepsilon$：强制专家偏差，阻止推理导向的探索
- 过小 $\lambda$：训练不稳定
- 过大 $\lambda$：限制探索

训练过程中 $d_{IoU}$ 先降后升——反映了从早期探索到后期利用的健康过渡。

### 4.4 Test-Time Scaling

- TreeVGR 高 Pass@1 但在 $k$ 增大时几乎无增益（分布过于 sharp）
- PFlowNet 不仅 Pass@1 更高，且在 $k$ 增大时持续提升——**变分推理确保了多样化的理性分布**

### 4.5 Performance-Efficiency Trade-off

相比 Agentic 框架（DeepEyes, Thyme）：PFlowNet 用结构化文本 token 替代复杂的工具/代码执行，显著缩短上下文长度和推理延迟。
相比 TreeVGR：解耦感知与推理以可承受的计算开销大幅提升感知质量和推理性能。

### 4.6 跨尺度泛化

在 Qwen3-VL 4B 和 32B 上也验证了 PFlowNet 的有效性：
- 4B：SFT 平均提升 2.2%，RFT 进一步提升 2.3%+
- 32B：SFT 平均提升 1.3%，RFT 进一步提升 2.3%+

## 5. 关键图表（本地资源）

### Figure 1：IoU vs 推理精度探测
![IoU vs Accuracy](assets/arxiv_2605_perceptual_flow_network_for_visually_grounded_reasoning/fig1.png)

*核心发现：几何精度最高的专家标注（红框）并非推理效用最高。*

### Figure 2：三种范式对比
![三种范式对比](assets/arxiv_2605_perceptual_flow_network_for_visually_grounded_reasoning/fig2.png)

*(a) Agentic 框架：多轮工具调用；(b) Grounded RLVR：单轮感知+推理耦合；(c) PFlowNet：解耦感知与推理。*

### Figure 3：可行区域与优化目标
![可行区域示意](assets/arxiv_2605_perceptual_flow_network_for_visually_grounded_reasoning/fig3.png)

*现有方法约束模型模仿专家轨迹，PFlowNet 通过推理导向奖励 + 领域几何塑形实现更充分且可控的探索。*

### Figure 4：PFlowNet 总览
![PFlowNet 总览](assets/arxiv_2605_perceptual_flow_network_for_visually_grounded_reasoning/fig4.png)

### Figure 5：数据管线
![数据管线](assets/arxiv_2605_perceptual_flow_network_for_visually_grounded_reasoning/fig5.png)

### Figure 7：性能-效率权衡
![性能-效率权衡](assets/arxiv_2605_perceptual_flow_network_for_visually_grounded_reasoning/fig7.png)

*PFlowNet 在延迟和上下文长度方面显著优于 Agentic 框架，同时保持高性能。*

### Figure 8：Pass@k 曲线
![Pass@k 曲线](assets/arxiv_2605_perceptual_flow_network_for_visually_grounded_reasoning/fig8.png)

*TreeVGR 的 Pass@k 曲线平坦（分布过于 sharp），PFlowNet 随 k 增大持续提升。*

### Figure 9：定性对比
![定性对比](assets/arxiv_2605_perceptual_flow_network_for_visually_grounded_reasoning/fig9.png)

### Figure 10：Geometric Shaping 消融
![Geometric Shaping 消融](assets/arxiv_2605_perceptual_flow_network_for_visually_grounded_reasoning/fig10.png)

*$d_{IoU}$ 先降后升反映从探索到利用的健康过渡。*

### Figure 12：RoI 分布与输出长度统计
![RoI 分布与输出长度](assets/arxiv_2605_perceptual_flow_network_for_visually_grounded_reasoning/fig12.png)

## 6. 优点与局限

### 优点
1. **核心洞察深刻**：首次系统性论证"几何精度 ≠ 推理效用"，揭示了视觉专家归纳偏差的本质问题
2. **理论保证严格**：TV 距离界及极限分析表明 PFlowNet 统一了 MLE 和 expert-guided RLVR 作为特例，并保证严格改进
3. **架构设计优雅**：解耦感知与推理，用结构化文本 token 编码视觉思维，兼顾可解释性和效率
4. **奖励设计精巧**：对比项 = 特权信息蒸馏（reverse-KL），效用项 = 信息增益，两者协同
5. **Test-time scaling 优异**：变分推理确保分布多样性，Pass@k 曲线持续上升
6. **跨尺度泛化**：4B/8B/32B 均有效

### 局限
1. **缺乏自适应感知**：对所有问题使用固定结构化推理格式，简单问题或 STEM 任务可能引入不必要的推理开销
2. **理论假设理想化**：定理 1&2 依赖理想化假设和正则性条件，实践中仅作为定性指导
3. **两个关键超参数**（$\varepsilon$ 和 $\lambda$）：最优配置可能随基础模型和领域变化
4. **训练两阶段**（SFT + RFT）增加了 pipeline 复杂度

## 7. 复现与工程实现要点

1. **基础模型**：Qwen3-VL-8B（需支持 `<analyze>/<localize>` 等特殊 token）
2. **教师模型**：Gemini3Flash + GPT-4o（用于 flow 合成）
3. **奖励模型**：与 PFlowNet 同初始化的冻结模型 $p_\phi$
4. **RFT 并行策略**：目标函数涉及密集的子轨迹计算，需开发并行策略保证可扩展性
5. **数据构建**：
   - 收集多样化任务 + 多样化 RoI 的样本
   - 随机扩展专家 RoI → 教师合成 flow
   - 双设定验证（w/o flow, w/ flow）→ 难度控制分流至 cold-start / RFT
6. **超参数**：$\varepsilon=0.5, \lambda=4.5$
7. **推理流程**：先生成 Perceptual Flow → 据 flow 中 RoI 裁剪图像 → 拼接 flow 文本 + RoI 特征 → 自回归生成答案

## 8. 对当前项目的启发与可落地尝试

1. **"几何精度 ≠ 推理效用"的洞察可迁移**：在任何使用外部专家提供中间监督的场景中，都应该考虑专家偏差问题。不应盲目对齐专家输出，而应让模型学习超越专家先验的自有分布
2. **Perceptual Flow 作为结构化推理前缀**：将视觉思维编码为结构化文本 token（而非工具调用）是一种高效的 "thinking with images" 方案，可应用到任何需要视觉 grounding 的推理任务中
3. **对比奖励 = 特权信息蒸馏**：$p_\phi^+/p_\phi^-$ 的设计本质上是一种 on-policy 蒸馏——用冻结教师模型的裁剪区域分布作为正例，互补区域作为负例，提供 dense reward signal。这个思路可以直接用于 OPD 场景中替代传统的 KL 散度损失
4. **Vicinal shaping 替代严格 KL 约束**：仅惩罚远离专家领域的轨迹，而非全局约束 KL——这种局部约束策略比 PPO 中的 KL penalty 更灵活，可以借鉴到 RL fine-tuning 的各个场景
5. **Sub-TB 提供密集中间监督**：相比 PPO 的稀疏 terminal reward，Sub-Trajectory Balance 对每个子轨迹都提供监督信号，可以借鉴到任何链式推理或多步决策任务中
6. **两阶段训练（Cold Start SFT → Variational RFT）**：先用监督信号热启动，再用变分 RL 精调——这一训练范式适用于需要从有偏数据初始化后进一步优化的场景
