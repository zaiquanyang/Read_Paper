# HyperEyes: Dual-Grained Efficiency-Aware Reinforcement Learning for Parallel Multimodal Search Agents

## 1. 基本信息
- arXiv ID: `2605.07177v1`
- 标题: `HyperEyes: Dual-Grained Efficiency-Aware Reinforcement Learning for Parallel Multimodal Search Agents`
- 作者: Guankai Li, Jiabin Chen, Yi Xu, Xichen Zhang, Yuan Lu
- 机构: Xiaohongshu Inc., University of Cambridge
- 主题目录: `vlm`
- 总结粒度: `detailed`
- 代码与数据: [GitHub](https://github.com/Guankai-Li/HyperEyes)
- URL:
  - abs: https://arxiv.org/abs/2605.07177
  - pdf(v1): https://arxiv.org/pdf/2605.07177v1.pdf

## 2. 背景与动机

### 2.1 这篇论文到底在解决什么

这篇工作讨论的不是“让 VLM 会不会搜索”，而是“让 multimodal search agent 搜得是否高效”。

作者的核心观察是：

- 现有 multimodal search agents 大多是 **串行** 的；
- 对一张图里有多个实体、一个问题又需要分别查多个实体信息的场景，agent 往往会：
  - 先裁第一个实体
  - 搜一次
  - 再裁第二个实体
  - 再搜一次
  - 如此反复
- 这会带来两类问题：
  - `interaction round` 很多，推理很慢
  - context 会越来越长、越来越 noisy，后面更容易被干扰

作者认为，对这类可分解的查询，正确策略应该是：

**search wider, not longer**

也就是：

- 不要把多个独立子检索任务串成长链；
- 而是应该在同一轮里把多个 grounded query 一起发出去。

### 2.2 为什么“有并行能力”还不够

这是论文最重要的出发点之一。

即便模型已经支持 parallel tool calling，也不代表它会“高效地并行”。

原因是大多数现有 search agent 训练目标还是围绕最终 `accuracy reward`：

- 只要答对了，就算好；
- 至于中间用了多少轮工具、查了多少冗余信息、是不是把本来能并行的事情拆成串行，都没有被明确惩罚。

所以模型很容易学出一种 brute-force 行为：

- 能多搜就多搜；
- 能再开一轮就再开一轮；
- 最终虽然正确率上去了，但 latency 和工具调用成本非常差。

HyperEyes 的核心贡献，就是把 **效率** 从一个隐含偏好，提升为训练目标里的第一类公民。

## 3. 方法总览

HyperEyes 整体由三块组成：

1. 并行友好的动作空间：`Unified Grounded Search (UGS)`
2. 面向并行检索的冷启动数据管线：`Parallel-Amenable Data Synthesis + PRS`
3. 双粒度效率感知强化学习：`TRACE + OPD`

其中最值得细看的是第 1 和第 3 部分。

### 3.1 整体框架图

#### 串行 vs 并行搜索直觉
![Parallel vs Serial](assets/arxiv_2605_hypereyes_dual_grained_efficiency_aware_reinforcement_learning_for_parallel_multimodal_search_agents/parallel_vs_serial.png)

这张图非常关键。它明确展示了作者想解决的问题：

- 传统 agent: `crop -> search -> crop -> search -> ...`
- HyperEyes: 一次把多个实体 grounding 出来，并在同一轮中同时发起搜索

#### 训练框架总览
![Framework](assets/arxiv_2605_hypereyes_dual_grained_efficiency_aware_reinforcement_learning_for_parallel_multimodal_search_agents/framework.png)

这张图对应全文最核心的算法结构：

- 前半部分是并行友好的数据构造与轨迹筛选
- 后半部分是 RL 阶段的双粒度优化

## 4. 问题形式化与动作空间

### 4.1 Agent 轨迹形式

论文沿用 ReAct 风格的 agent 形式。给定问题 $q$，完整轨迹写成：

\[
\tau =
\bigl(q, (r_0, a_0, o_0), (r_1, a_1, o_1), \dots, (r_T, a_T, o_T), y\bigr),
\]

其中：

- $r_t$ 是第 $t$ 轮 reasoning trace
- $a_t$ 是工具调用
- $o_t$ 是环境返回的 observation
- $y$ 是最终答案

agent 每轮都在历史上下文 $h_t$ 上先想，再调工具，再读观察，直到输出答案或达到最大轮数。

### 4.2 工具集合

HyperEyes 用两个基础工具：

- `<image_search>`: 给定 grounded image，返回视觉相关检索结果
- `<text_search>`: 给定文本 query，返回文本证据

这两者本身不新，新的点在于它们如何与 grounding 结合。

### 4.3 Unified Grounded Search (UGS)

作者批评现有 multimodal agent 的常见范式是：

1. 先做 crop / grounding
2. 再把 crop 结果喂给 search

这种两阶段方式有两个问题：

1. **错误会级联**
   - 一旦前面的定位错了，后面检索必然跟着错
2. **并行性被天然限制**
   - agent 必须等第一个 crop 结束，才能继续围绕它做搜索
   - 多实体任务就会被强制拉成串行链

UGS 的关键思想是：

- 不把 visual grounding 当作 search 之前的预处理步骤；
- 而是把 grounding 结果直接变成 retrieval action 的一部分参数；
- 一次性同时预测多个 target entity 的 bounding boxes；
- 同一轮里就把多个 grounded query 并发发出。

这是 HyperEyes 能“search wider”而不是“search longer”的根本前提。

## 5. 训练数据构造

### 5.1 数据组成

论文的训练数据总量是 `271k` QA：

| 数据源 | #QA Pairs | #SFT | #30B RL | #235B RL |
|---|---:|---:|---:|---:|
| LiveVQA | 100k | 13.5k | 3k | 5k |
| REDSearch | 10k | 2k | 0.5k | 0.5k |
| InfoSeek | 41k | 3k | - | - |
| iNaturalist | 75k | 2.5k | 2k | 3k |
| Google-Landmark | 12k | 1k | - | - |
| DeepDive | 3k | 1.5k | - | - |
| 内部人工标注 | 5k | 0.5k | - | - |
| 合成视觉多实体 | 20k | 5k | 0.5k | 0.5k |
| 合成文本多约束 | 5k | 1k | - | - |
| 总计 | 271k | 30k | 6k | 9k |

重点不是总量，而是作者专门为“并行搜索”补了两个新数据流。

### 5.2 Visual multi-entity synthesis

这部分是论文很有意思的数据工程设计。

流程大致是：

1. 从 fine-grained classification 数据集出发
   - birds, flowers, cars, aircraft, pets 等
2. 针对每个类别，抓一套结构化知识
   - 视觉特征
   - taxonomy
   - 历史背景
   - 生态 / 行为属性
3. 基于这些知识构建每个类的 QA pool
4. 再从不同类别抽图做 `mosaic augmentation`
   - 拼出一张包含 2 到 8 个实体的复合图
5. 然后基于多个实体的知识联合生成问题

这里的关键约束是：

- 问题必须真的需要整合所有共现实体的信息；
- 漏掉任意一个实体，都不能推出正确答案。

这样就天然逼模型去做多实体并行检索，而不是单实体串行问答。

### 5.3 Textual multi-constraint synthesis

作者还做了文本侧的多约束 query 合成：

- 从 Wikidata 做 multi-hop random walk
- 抽取若干属性约束
- 构造交集唯一的答案集合

这类 query 和传统 chain-style reasoning 不同，它不是一步接一步依赖，而是多个约束同时成立，因此也适合并行搜索范式。

### 5.4 Tool-necessity filtering

一个很重要的过滤步骤是：

- 任何能被 `Qwen3-VL-235B` **不借助外部工具**直接答对的样本都会被丢掉。

这样留下来的训练集更干净，因为它们是真正需要搜索工具才能解决的问题，而不是 parametric memory 就能蒙对的问题。

### 5.5 Progressive Rejection Sampling (PRS)

作者发现 naive rollout 常常会：

- 重复调工具
- 不断 query reformulation
- 交互轮数变多，但正确率未必更高

于是他们提出 `Progressive Rejection Sampling`：

- 对同一个 query，用逐步放宽的 turn budget 去采样轨迹；
- 始终只保留 **最短的成功轨迹**；
- 预算越紧，越不允许“拖沓地试错”；
- 所以最终留下来的轨迹天然更像“单轮精确并行 dispatch”。

再叠加 quality filtering：

- 过滤格式无效
- 过滤没有信息增益的轨迹
- 过滤 reasoning 不 grounded 的轨迹

最终把 `271k` 初始任务蒸成 `30k` 高质量 SFT 轨迹。

这个结果后面在 ablation 里证明非常有效。

## 6. 双粒度效率感知强化学习

这部分是论文真正的算法核心。

作者把问题拆成两个层次：

1. `macro-level`: 轨迹级别，优化 search 效率
2. `micro-level`: token 级别，在失败轨迹上补细粒度监督

对应的方法就是：

- `TRACE` 负责宏观效率
- `OPD` 负责微观纠错

### 6.1 总奖励

轨迹总奖励定义为：

\[
R = R_{\text{acc}} + R_{\text{fmt}} + R_{\text{tool}},
\]

其中：

- $R_{\text{acc}} \in \{0,1\}$: 最终答案是否正确
- $R_{\text{fmt}} \in \{0, -\lambda_{\text{fmt}}\}$: schema / 格式解析错误惩罚
- $R_{\text{tool}}$: 论文的核心效率奖励

### 6.2 TRACE: Tool-use Reference-Adaptive Cost Efficiency

#### 6.2.1 为什么静态阈值不够

作者指出，工具调用“多少算合理”是 query-dependent 的：

- 有些题一轮搞定
- 有些题真需要多跳

所以如果给一个静态阈值：

- 太松了，抑制不了冗余搜索
- 太紧了，又会把真正需要 multi-hop 的轨迹错罚掉

TRACE 的做法是：**让效率参考线在训练中自适应收紧**。

#### 6.2.2 两个计数维度

对一条轨迹，作者用两个量衡量工具使用：

- $t_c$: tool-call rounds 数
- $t_s$: 总工具调用数

其中：

- `rounds` 对应多少轮 decoder 前向里发生了工具调用
- `total invocations` 对应总共开了多少具体工具

这两个量一起用，是为了防止模型“在一轮里疯狂并行乱搜”来投机。

#### 6.2.3 自适应参考线更新

对每个 medium-difficulty RL 样本，初始参考来自它在初始成功轨迹上的：

- $\hat{t}_c^{(0)}$
- $\hat{t}_s^{(0)}$

训练过程中，轮数参考线按 epoch 单调收紧：

\[
\hat{t}_c^{(e+1)} = \min\bigl(\hat{t}_c^{(e)}, t_c^{(e+1)}\bigr).
\]

其中 $t_c^{(e+1)}$ 是第 $e+1$ 个 epoch 里成功 rollout 中最小的轮数。

这条更新规则很关键，它意味着：

- 参考线只会变得更严格，不会放松；
- 但它又不是拍脑袋手设，而是跟着当前策略能达到的最好水平动态收紧；
- 本质上形成了一个 implicit curriculum。

#### 6.2.4 TRACE 奖励结构

工具奖励 $R_{\text{tool}}$ 分四种情况：

1. **失败轨迹，且轮数太少或太多**
   - 给固定负惩罚 $-\lambda_{\text{red}}$
2. **失败轨迹，但轮数在允许区间**
   - 给 0
3. **成功轨迹，且足够高效**
   - 给正奖励 $R_+$
4. **成功轨迹，但仍有冗余**
   - 给负奖励或较低奖励 $R_-$

形式上写成：

\[
R_{\text{tool}} =
\begin{cases}
-\lambda_{\text{red}}, & R_{\text{acc}}=0 \text{ and } (t_c < \hat{t}_c \text{ or } t_c > \gamma \hat{t}_c) \\
0, & R_{\text{acc}}=0 \text{ and } t_c \in [\hat{t}_c, \gamma \hat{t}_c] \\
R_+ \in [r^+_{\min}, r^+_{\max}], & R_{\text{acc}}=1, t_c>0, t_c \le \hat{t}_c, t_s \le \hat{t}_s \\
R_- \in [r^-_{\min}, r^-_{\max}], & R_{\text{acc}}=1, t_c>0, \text{ otherwise}
\end{cases}
\]

其中 $\gamma > 1$ 是冗余容忍因子。

作者还对 $R_+$ 与 $R_-$ 做了 group 内 rank-based 插值：

\[
R_* = r^*_{\min} + \frac{G-\rho}{G-1}(r^*_{\max} - r^*_{\min}), \quad * \in \{+, -\}.
\]

这里 $\rho$ 是按 $t_c$ 排序后的名次，$\rho=1$ 表示最有效率。

#### 6.2.5 TRACE 的两个关键设计点

1. **只有同时满足 $t_c \le \hat{t}_c$ 且 $t_s \le \hat{t}_s$ 才给正奖励**
   - 防止 reward hacking：模型把很多调用塞进一轮里伪装高效
2. **正确但 $t_c=0$ 的轨迹不给工具奖励**
   - 防止模型靠参数记忆“猜对”而被误认为高效 search

这两个限制非常必要，否则效率奖励会被投机路径钻空子。

### 6.3 为什么还需要 OPD

作者指出，TRACE 是 trajectory-level reward，所以对失败轨迹有一个严重问题：

- 失败时，GRPO 会给整条轨迹一个统一负优势；
- 但这条轨迹里可能前面很多 reasoning step 是对的、工具调用也是对的；
- 真正错的也许只是在后面某个推理节点；
- 稀疏 outcome reward 无法告诉模型“哪里错了”。

这就是 credit assignment deficiency。

### 6.4 VLM search agent 场景下的 OPD

作者的做法是：

- 引入一个更强的 teacher $\pi_{\text{teacher}}$
- **只对 failed rollout** 做 token-level distillation
- 最小化 completion token 上 student 与 teacher 的 reverse KL

最终学生损失：

\[
L(\theta) = L_{\text{GRPO}}(\theta) + \lambda_{\text{kd}} \, \mathbb{E}_{\tau \sim \pi_{\theta}^{\text{old}}}
\left[
\mathbf{1}[R_{\text{acc}}(\tau)=0]\cdot
\frac{1}{|A_\tau|}
\sum_{t \in A_\tau}
\mathrm{KL}\bigl(\pi_\theta(\cdot|s_t)\,\|\,\pi_{\text{teacher}}(\cdot|s_t)\bigr)
\right].
\]

其中：

- $A_\tau$ 是 completion token 集合
- 只在失败轨迹上启用蒸馏
- teacher 冻结
- old policy 只负责采样，不参与梯度

这个设计的精妙之处在于：

1. 成功轨迹上的 parallel-dispatch 行为完全交给 TRACE 去塑形，不被 teacher 干扰
2. 失败轨迹上的 token-level 问题，通过 teacher 的 dense signal 去修正

这也是题目里所谓 `Dual-Grained` 的真正含义：

- 宏观上学“搜索得够不够短、够不够并行”
- 微观上学“失败轨迹里具体哪一步 reasoning / action token 该怎么修”

## 7. IMEB Benchmark

### 7.1 为什么要新 benchmark

作者认为现有 multimodal search benchmark 的共同问题是：

- 只看最终 accuracy
- 不看搜索效率
- 所以模型即便很冗长、很啰嗦、上下文极 noisy，也不会在指标上吃亏

为此他们提出 `IMEB`：Image Multi-Entity Benchmark。

### 7.2 IMEB 的特点

![IMEB](assets/arxiv_2605_hypereyes_dual_grained_efficiency_aware_reinforcement_learning_for_parallel_multimodal_search_agents/IMEB.png)

IMEB：

- 共 300 个实例
- 人工精修并多轮交叉校验
- 平均每张图约 4.6 个实体
- 问题要求必须同时定位并检索多个实体的信息，才能答对

作者给出的 domain 分布包括：

- Sports
- Humanities & History
- Entertainment
- Finance
- Science
- Consumption
- Daily Life

它的设计目标非常明确：测的是 **多实体并行 grounded retrieval**，而不是普通单实体 VQA。

### 7.3 Cost-Aware Score (CAS)

作者提出的统一指标为：

\[
CAS = \frac{Acc^2 \times 100}{N_{\text{tok}} + 2N_{\text{tool}} + 1}.
\]

其中：

- $Acc$: 准确率
- $N_{\text{tok}}$: token 消耗，以千为单位
- $N_{\text{tool}}$: 顺序工具调用轮数

这个式子背后的设计直觉是：

1. 用 $Acc^2$ 强调“正确率仍然是一号目标”
2. 分母同时惩罚 token 和工具轮数
3. 给 `N_tool` 更重的权重，是因为一次 tool-call 往往对应显著 latency

这个指标非常适合 agent 场景，因为它显式惩罚“答对但很拖沓”的行为。

## 8. 主实验结果

### 8.1 六个 benchmark 的主结果

论文在 6 个 benchmark 上评估：

- MMSearch
- FVQA
- LiveVQA
- BCVL
- MMSearch+
- IMEB

最关键的结果是：

| Model | Avg Acc / Avg Turns |
|---|---:|
| VDR | 54.1 / 11.6 |
| HE-30B (SFT) | 58.8 / 2.9 |
| HE-30B (RL) | 64.0 / 2.2 |
| HE-235B (SFT) | 64.1 / 2.8 |
| HE-235B (RL) | 66.6 / 2.0 |
| Gemini-3.1-Pro | 67.7 / 1.8 |

几个最值得记的数字：

- `HyperEyes-30B (RL)` 相比同规模最强开源 deep search agent，平均准确率 **+9.9**，同时工具轮数约 **-9.4**
- 相比 VDR，工具调用轮数大约少 **5.3x**
- `HyperEyes-235B (RL)` 的平均准确率来到 **66.6**
- 和 `Gemini-3.1-Pro` 的平均差距只有 **1.1** 个点

作者的解释是：

- 现有串行 agent 仍然困在 `crop-then-search` loop 里
- HyperEyes 用 UGS 把 entity grounding 与 retrieval 解耦并并发化
- 因此同时减少了 latency 与级联错误

### 8.2 BCVL / IMEB 上的成本感知对比

Table 3 特别有说服力，因为它把 token、tool 数和 CAS 放在一起看：

| Method | BCVL CAS | IMEB CAS |
|---|---:|---:|
| MMSearch-R1 | 0.520 | 0.013 |
| DeepEyes-V2 | 0.182 | 0.119 |
| WebWatcher | 0.191 | 0.059 |
| VDR | 0.128 | 0.014 |
| **HyperEyes-30B** | **2.232** | **0.910** |

这里可以看出 HyperEyes 不是“为了效率牺牲精度”，而是两边一起赢：

- BCVL 上 CAS 大约是次优开源方法的 **4.3x**
- IMEB 上 CAS 大约是次优开源方法的 **7.6x**

这基本证明了：在多实体场景里，并行 grounding + efficiency-aware RL 确实不是 cosmetic 优化，而是显著改变了单位计算成本的信息密度。

## 9. 消融实验

### 9.1 数据过滤比原始数据量更重要

Table 4 比较了：

- `DBase`: 121k trajectories
- `DFiltering`: 30k trajectories

结果：

- 样本数砍到约 1/4
- 平均性能却从 `51.6` 提升到 `58.8`

这说明在 agent 训练里，**轨迹质量** 比单纯堆数据量更重要。

尤其是对“高效并行调工具”这种行为，如果示范轨迹本身就冗余，SFT 学到的只会是更冗余的 pattern。

### 9.2 TRACE 的作用

Table 5 的 reward ablation 很关键。

从 `Qwen3-VL-30B + SFT` 出发：

- 只加 `Outcome Reward`
  - 平均准确率 `58.3`
  - 平均 turns `6.9`
  - 明显出现“多搜但不更准”
- 加 `TRACE (w/o update)`
  - 平均准确率 `61.1`
  - turns `2.5`
- 加 `TRACE`
  - 平均准确率 `62.7`
  - turns `2.2`

这说明：

1. 单纯 outcome reward 会鼓励 over-retrieval
2. 静态 tool reference 已经能纠正一部分冗余
3. 自适应更新的 TRACE 还能进一步提升 accuracy 与效率

也就是说，动态参考线确实比静态阈值更合理。

### 9.3 OPD 的作用与 teacher 选择

继续看 Table 5：

- `+ TRACE`: Avg `62.7 / 2.2`
- `+ TRACE + OPD†`，teacher 用原始 `Qwen3-VL-235B-Instruct`
  - Avg `46.3 / 2.8`
  - 严重退化
- `+ TRACE + OPD*`，teacher 用 **RL 后的 HyperEyes-235B**
  - Avg `64.0 / 2.2`
  - 比 TRACE 再涨 `+1.3`

这组结果非常关键，说明：

1. **OPD 在这里不是“有老师就行”**
2. teacher 必须是 **efficiency-aligned** 的
3. 如果 teacher 没有学会 HyperEyes 这种并行且节制的 search policy，它给 student 的 dense token guidance 反而会把 student 往错误搜索风格上带

作者因此得出一个很重要的结论：

- 有效 distillation 的前提是：
  - parallel-amenable cold start
  - efficiency-aligned teacher

这点对于后续做 VLM agent OPD 很重要，因为它说明 teacher 的行为风格必须和 student 的目标一致，不能只看 teacher 规模更大。

### 9.4 没有 SFT 冷启动会怎样

论文里还有一行：

- `Qwen3-VL-30B + TRACE + OPD*`
  - Avg `44.7 / 2.2`

相比带 SFT 冷启动的 `64.0 / 2.2`，差距非常大。

这说明：

- 直接把 RL + OPD 砸到 vanilla backbone 上，效果会塌；
- 并行工具使用能力不是单靠 reward 就能从零长出来的；
- 必须先靠高质量 SFT 轨迹把 agent 拉到一个“会并行搜索”的可行域里。

## 10. 这篇论文里 OPD 的真正角色

虽然这篇是 `vlm` 论文，不是纯 OPD 论文，但它对 OPD 的启发非常强。

### 10.1 OPD 在这里不是主目标，而是“失败轨迹修复器”

和一些纯 distillation work 不同，这里 OPD 不是整个训练框架的中心，而是：

- 在 sparse trajectory reward 不能细分错误来源时；
- 用 teacher 的 dense token-level signal 去修 failed rollout；
- 让 student 别把整个失败轨迹都当成“全错”。

所以它承担的是 **micro credit assignment repair** 的角色。

### 10.2 为什么只在 failed rollout 上蒸馏

这个设计很干净：

- 成功 rollout 上，TRACE 已经在奖励正确且高效的策略
- 如果这时还用 teacher 去覆盖，可能反而把 student 好不容易探索出的高效行为洗掉

因此作者只对失败轨迹做 OPD，等于：

- 成功样本靠 RL 保持探索出来的优策略
- 失败样本靠 teacher 做 token-level 纠偏

这是一种很值得借鉴的 RL + OPD 结合方式。

## 11. 优点与局限

### 11.1 优点

- 任务定义很清楚：
  - 不是追求更长 search chain，而是追求更有效率的并行搜索。
- 方法结构完整：
  - action space、数据、SFT、RL、benchmark 是一整套闭环。
- TRACE 设计合理：
  - 解决了 query-dependent 效率阈值的问题。
- OPD 的使用场景非常克制且有效：
  - 只修 failed rollout，避免过度干扰成功策略。
- IMEB 很有价值：
  - 终于把多实体 multimodal retrieval 里的效率问题显式 benchmark 化。

### 11.2 局限

- OPD 依赖更强的同家族 teacher
  - 论文自己也承认这限制了向 frontier-scale 继续外推。
- 当前只覆盖静态 image + text 环境
  - 没处理 video / audio 这种时空 grounding。
- 和顶级闭源模型仍有剩余差距
  - 尤其是 parametric knowledge 更强的 Gemini。
- UGS / 并行调工具范式更适合可分解 query
  - 对真正强依赖 sequential multi-hop 的任务，收益可能没那么大。

## 12. 对当前 `vlm` 项目的启发

### 12.1 方法层启发

1. 多实体视觉检索任务里，**并行 breadth** 可能比继续拉长 reasoning depth 更重要。
2. agent 训练时，**效率必须进 reward**，否则模型几乎一定会 over-retrieve。
3. 在 multimodal search 里，OPD 更适合作为 **failed rollout 的 token-level credit repair**，而不是简单全量蒸馏。
4. teacher 不是越强越好，关键是 **行为风格是否对齐**。

### 12.2 可落地尝试

1. 如果你在做视觉搜索 agent，可以先看现有 action schema 能不能改成类似 UGS 的“一步 grounding + retrieval”。
2. 如果你已有 RL 版 agent，优先加一个 query-adaptive 的 tool efficiency reward，而不是只加固定工具惩罚。
3. 如果你想把 OPD 引入 VLM agent，建议先做“failed-rollout-only OPD”，再看是否需要扩大蒸馏范围。
4. 如果你在构建自己的 benchmark，最好像 IMEB 一样把 `accuracy + cost` 一起度量，而不是只看答对率。

## 13. 一句话评价

这篇工作最有价值的地方，不是单纯把 multimodal agent 做得更会搜，而是把“并行搜索的能力”和“并行搜索的意愿”同时训练出来了：前者靠 UGS 与数据构造，后者靠 TRACE，而 OPD 则负责把失败轨迹里的细粒度错误真正修掉。
