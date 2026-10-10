# Walk the Talk: Bridging the Reasoning-Action Gap for Thinking with Images via Multimodal Agentic Policy Optimization

## 1. 基本信息
- arXiv ID: 2604.06777
- URL: [abs](https://arxiv.org/abs/2604.06777) / [pdf](https://arxiv.org/pdf/2604.06777.pdf)
- 主题: `vlm`
- 粒度: `detailed`
- 作者: Wenhao Yang, Yu Xia, Jinlong Huang, Shiyin Lu, Qing-Guo Chen, Zhao Xu, Weihua Luo
- 核心方法: `MAPO (Multimodal Agentic Policy Optimization)`
- 核心关键词: `thinking with images`, `multimodal agent`, `process supervision`, `semantic alignment`, `tool use RL`, `CLIP reward`

## 2. 这篇论文想解决什么问题
这篇工作的切入点很明确: 现在很多 MLLM 已经不满足于“看一张图然后只在文本里推理”，而是开始进入 `thinking with images` 阶段，也就是在推理链中主动调用视觉工具，例如 crop、zoom-in、局部观察，再根据新的视觉观察继续推理。

问题在于，现有 RL 训练大多只看最终答案是否正确，也就是只给 `outcome-based reward`。这会带来一个作者强调的核心问题:

- 文本推理看起来很合理，但执行出来的视觉动作可能是错的
- 模型可能“嘴上说要放大摩托车”，实际却裁到了错误区域
- 只要最终答案碰巧答对，RL 仍会把这条轨迹当成正样本强化
- 错误的动作执行与看似合理的文字推理混在一起，会让训练信号非常噪声化

作者把这个问题命名为 `reasoning-action gap`。它不是简单的“答案错了”，而是“推理意图”和“动作执行”之间出现了系统性脱钩。对多轮 agentic visual reasoning 来说，这种脱钩尤其致命，因为错误会在长轨迹中逐步积累，最后导致:

- 语义监督过 sparse
- 梯度方差变大
- 训练曲线震荡
- 长程训练时可能 collapse

从问题定义看，这篇论文的重点不是再发明一个更强的 outcome reward，而是给 agent 轨迹增加一种“过程一致性监督”: 模型不光要会想，还要让执行出来的视觉动作与它刚才说的意图一致。

## 3. 核心直觉: 先让模型“说清楚”，再要求它“说到做到”
MAPO 的核心隐喻就是标题里的 `Walk the Talk`:

- `talk`: 模型在调用视觉工具前，必须显式生成一个描述性 label，说明它预期将看到什么
- `walk`: 模型真正执行 crop / zoom 等动作后，环境返回一个视觉 observation，再用 CLIP 去衡量 observation 是否和这个 label 语义一致

这一步很关键。作者不是把 label 当成可有可无的 tool 参数，而是把它变成一个强制输出、可验证、可打分的中间监督信号。这样一来，原来不可见的“动作意图”被显式化了，轨迹就能被过程级监督。

![MAPO 方法总览](assets/arxiv_2604_walk_the_talk_bridging_the_reasoning_action_gap_for_thinking_with_images_via_multimodal_agentic_policy_optimization/MAPO_method.png)

上图对应论文的主方法图。可以把它理解为一个三段式闭环:

1. 模型生成文本推理、工具动作以及描述性 label。
2. 工具执行后返回局部图像 observation。
3. CLIP 比较 label 与 observation 的语义一致性，并把这个分数并入 advantage estimation。

相比只在终点看答对没答对，这相当于在每一步动作执行后增加了“你刚才说要看这个，结果你真的看到了吗”的检查。

## 4. 形式化建模
作者把多轮视觉 agent 推理建模为 MDP。第 $t$ 步状态记为:

\[ s_t = \{(X_0, I_0), (X_1, I_1), \dots, (X_t, I_t)\} \]

其中:

- $(X_0, I_0)$ 表示原始问题与原图
- 后续 $(X_t, I_t)$ 包含了文本推理、工具调用以及返回的新视觉观察

在每一步，策略同时生成动作和描述标签:

\[ (a_t, y_t) \sim \pi_\theta(\cdot \mid s_t) \]

这里:

- $a_t$ 是视觉动作，例如 crop / zoom-in
- $y_t$ 是对“预期将看到的内容”的文字描述

执行动作后，环境返回新的视觉 observation $I_t$。随后用 CLIP 去比较 $y_t$ 和 $I_t$ 的一致性，得到 step-level semantic score:

\[ z_t = \mathrm{CLIP}(y_t, I_t) \]

这一步的意义非常直接: 如果模型说“我要看红衣男子手里的手机”，结果 crop 出来的是一辆摩托车，那这个 step 的语义分数就会很低。

## 5. 语义奖励设计
### 5.1 Step-level semantic score
MAPO 不是直接把所有 step 的 CLIP 分数简单求和，因为那样会鼓励模型故意拉长轨迹、刷更多中间奖励，形成 reward hacking。

因此作者设计了一个 trajectory-aware discount：

\[ R_{\text{sem}} = \frac{1}{T}\sum_{t=1}^{T}\lambda^{T-t} z_t \]

其中:

- $T$ 是整条轨迹的步数
- $\lambda$ 是折扣因子，论文默认设为 $0.95$

这个设计有两层考虑:

- 对早期探索保留一定容错，因为多轮视觉推理前几步可能需要试探
- 对过长轨迹施加折扣，防止模型靠无意义操作累计 semantic reward

所以它不是普通的 dense reward，而是一个带轨迹长度意识的过程一致性奖励。

### 5.2 与 outcome reward 的融合
最终，MAPO 不是用 semantic score 取代最终答案奖励，而是把两者结合起来。论文中的复合 advantage 写成:

\[ \widetilde{A}^{(i)} = \widehat{A}^{(i)}_{\text{out}} + \beta \widehat{A}^{(i)}_{\text{sem}} \]

其中:

- $\widehat{A}^{(i)}_{\text{out}}$ 是 final answer 的 group-normalized advantage
- $\widehat{A}^{(i)}_{\text{sem}}$ 是 semantic score 的 group-normalized advantage
- $\beta$ 是平衡系数，论文设为 $0.4$

这意味着 MAPO 的优化目标不是单纯鼓励“答对”，而是鼓励“既答对，又在中间过程里说到做到”。

## 6. MAPO 的优化目标
论文把 MAPO 放进一个类似 GRPO/GSPO 的 policy optimization 框架中。它的核心目标可以写成:

\[ \mathcal{J}_{\text{MAPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{\tau_i\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|\tau_i|}\sum_{\mathcal{I}=1}^{|\tau_i|}\min\left(w_{\mathcal{I}}^{(i)}(\theta)\widetilde{A}^{(i)}, \mathrm{clip}(w_{\mathcal{I}}^{(i)}(\theta), 1-\epsilon, 1+\epsilon)\widetilde{A}^{(i)}\right)\right] \]

这里的关键点不是公式本身，而是下面三个设计一起工作:

- 同一 query 采样一个 group 的轨迹，做 group-based normalization
- 用语义一致性分数补足稀疏 outcome reward
- 用 clipping 控制策略更新幅度

因此 MAPO 可以看成“把过程监督塞进 GRPO/GSPO 风格的组内相对优化中”。

## 7. 为什么作者强调 label 必须是强约束、细粒度的
论文专门强调，以前一些工作虽然也在 tool 参数里放过 label 字段，但通常是 optional 或约束很弱的。这样模型容易退化成很泛的描述，比如:

- `person`
- `object`
- `car`

这种 label 对 CLIP 验证几乎没用，因为它不能表达真正的视觉意图。MAPO 的一个隐含但很重要的贡献，是把 label 从“可选注释”提升为“必须具体、必须可验证的意图表达”。

也就是说，这篇工作不只是加了一个 CLIP reward，而是重新定义了视觉工具调用时的中间表示。这个中间表示必须足够具体，才能把“推理意图”转换为“可判别的监督对象”。

## 8. 理论分析: Dual Variance Reduction
作者给 MAPO 提供了比较完整的理论解释，核心结论是它同时降低两类方差，因此训练更稳。

### 8.1 空间方差降低: group aggregation
同一个 query 下采样 $G$ 条轨迹，组内 reward 高相关。作者证明 MAPO/GRPO 这类 group baseline 本质上能充当 spatial control variate，把同一视觉 prompt 的共享难度抵消掉。

论文给出的近似关系是:

\[ \mathrm{Var}(g_{\text{group}}) \approx (1-\rho)\mathrm{Var}(g_{\text{std}}) + O(1/G) \]

其中 $\rho$ 是同组样本 reward 的相关性。由于同一张图、同一道题生成的多个响应通常高度相关，所以 $\rho$ 往往比较高，因而组均值 baseline 会明显减少方差。

作者把这个解释成一种 `spatial momentum` 或 `spatial control variate`: 不是沿时间维度聚合历史梯度，而是在同一 prompt 的多个候选上抵消 prompt-level 的偶然性。

### 8.2 信号方差降低: semantic reward denoising
第二层方差降低来自 semantic score。作者认为 outcome reward 太 sparse，也太容易受偶然答对/答错影响；而 semantic reward 更贴近真实轨迹质量，因此条件方差更低。

直觉上:

- outcome reward 只在终点给一个 0/1
- semantic reward 会沿轨迹提供更稠密、更局部的反馈
- 它更直接约束动作执行是否与文本意图一致

因此，semantic reward 可以被看作对 outcome reward 的去噪补充，而不是简单的额外 bonus。

### 8.3 收敛界
论文给出的结论形式是:

\[ \min_{0 \le t < T}\mathbb{E}\left[\lVert \nabla J(\theta_t) \rVert^2\right] \le \frac{C_1 L \Delta}{\sqrt{T}} + \frac{C_2 L \sigma_{\text{MAPO}}^2}{\sqrt{T}} \]

作者进一步论证:

\[ \sigma_{\text{MAPO}}^2 \approx (1-\rho)\sigma_{\text{base}}^2 - \Delta_{\text{sem}} \ll \sigma_{\text{GRPO}}^2 \]

要点不是严格证明细节，而是它把实验中的“更稳定、不容易 collapse”解释成了更低梯度方差带来的收敛优势。

## 9. 实验设置
### 9.1 Backbone 与训练框架
- Backbone: `Ovis2.5-9B`
- RL 框架: `verl`
- Rollout 加速: `vLLM`
- 并行与显存优化: `FSDP`, optimizer offloading, gradient checkpointing, Triton fused kernels

### 9.2 任务与 benchmark
作者主要评测高分辨率、细粒度视觉推理，选择了:

- `V*`
- `HR-Bench`
- `MME-Realworld-Lite`

这些 benchmark 都强调小目标、细节属性、相对位置与高分辨率局部观察能力，因此非常适合评估 `thinking with images` 场景下的 agentic exploration。

### 9.3 关键超参数
根据正文和附录可整理出 MAPO 的主要超参数:

| 超参数 | 数值 |
|---|---:|
| Global batch size | 96 |
| Group size $G$ | 8 |
| Clip ratio $\epsilon$ | 0.2 |
| Semantic reward weight $\beta$ | 0.4 |
| Discount factor $\lambda$ | 0.95 |
| Max interaction turns | 6 |
| Learning rate | $1 \times 10^{-6}$ |
| KL coefficient | 0.0 |
| Pixel range | $[10^6, 2 \times 10^6]$ |

另外，作者还提到对生成的 descriptive label 做了长度约束，以减少 reward hacking。

## 10. 主要结果
### 10.1 主结果表
论文主表如下，能够较完整展示 MAPO 与闭源模型、开源 agent 模型以及同 backbone 上不同 RL 方法的对比:

| 方法 | V* Attr. | V* Rel. | V* Overall | HR-Bench 4K | HR-Bench 8K | HR-Bench Overall | MME-Real-Lite |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-5 | 70.0 | 73.4 | 71.3 | 74.6 | 72.0 | 73.3 | 55.3 |
| Gemini 2.5 Pro | 85.5 | 75.2 | 81.4 | 67.3 | 61.4 | 64.3 | 49.2 |
| DeepEyes | 89.6 | 85.5 | 88.0 | 74.9 | 71.2 | 73.0 | 48.4 |
| Thyme | 83.5 | 80.3 | 82.2 | 77.0 | 72.0 | 74.5 | 55.2 |
| Mini-o3 | 90.4 | 88.2 | 89.3 | 75.4 | 72.8 | 74.1 | 49.4 |
| DeepEyesv2 | - | - | 81.8 | 77.9 | 73.8 | 75.9 | - |
| Ovis2.5-9B | 81.9 | 81.6 | 81.8 | 74.0 | 68.2 | 71.1 | 46.1 |
| + Coldstart SFT | 73.0 | 84.2 | 77.5 | 74.6 | 71.0 | 72.8 | 47.9 |
| + PPO | 79.1 | 88.2 | 82.8 | 77.4 | 76.0 | 76.7 | 43.0 |
| + GRPO | 88.0 | 89.5 | 88.6 | 78.5 | 77.0 | 77.8 | 55.5 |
| + DAPO | 88.1 | 90.0 | 88.8 | 78.1 | 76.1 | 77.1 | 50.3 |
| + GSPO | 87.1 | 92.0 | 89.1 | 78.3 | 76.5 | 77.4 | 54.9 |
| + MAPO (Ours) | **91.1** | 87.2 | **89.5** | **81.0** | **78.6** | **79.8** | **55.8** |

从这张表可以读出几个关键信息:

- 在同一 `Ovis2.5-9B` backbone 上，MAPO 明显优于 GRPO、DAPO、GSPO 等 RL 基线
- MAPO 在 `HR-Bench` 上优势尤其明显，说明它对高分辨率局部视觉探索最有帮助
- `MME-Real-Lite` 上 MAPO 与 GRPO 差距不大，说明收益主要集中在需要多步视觉动作对齐的任务
- `V* Relative` 这一列 GSPO 更高，但 MAPO 在更重要的整体列和 HR-Bench 上更强

如果只看和强基线 GRPO 的对比，MAPO 在同 backbone 上:

- `HR-Bench Overall`: 79.8 vs 77.8
- `HR-Bench 8K`: 78.6 vs 77.0
- `MME-Real-Lite`: 55.8 vs 55.5

这基本符合论文的主张: MAPO 最擅长解决“复杂视觉动作轨迹中的过程错配”，所以在高分辨率细粒度探索任务上最吃香。

## 11. 定性分析: MAPO 到底改进了什么
### 11.1 Walk-the-Talk 案例
作者给了一个很直观的轨迹对比:

![MAPO 与 GRPO 的轨迹对比](assets/arxiv_2604_walk_the_talk_bridging_the_reasoning_action_gap_for_thinking_with_images_via_multimodal_agentic_policy_optimization/MAPO_walkthetalk_new.png)

这个例子能清楚看出 MAPO 的收益不是“语言更会编”，而是“动作执行更贴合语言意图”。

- GRPO 版本里，模型文字上知道应该看摩托车区域，但实际裁剪区域偏了，最后仍可能产生带有迷惑性的文本解释
- MAPO 版本里，模型先明确说出要观察的局部目标，再让 CLIP 对 observation 做一致性核对，因此更容易把 zoom/crop 落到正确区域

这正是论文所谓 bridging the reasoning-action gap 的具体体现。

### 11.2 训练稳定性
论文还给出奖励曲线:

![MAPO 与 GRPO 的训练稳定性对比](assets/arxiv_2604_walk_the_talk_bridging_the_reasoning_action_gap_for_thinking_with_images_via_multimodal_agentic_policy_optimization/reward_plot_new.png)

作者观察到:

- GRPO 在后期出现明显振荡，甚至训练退化
- MAPO 曲线整体更平滑，并持续上升

这个现象和理论分析是互相呼应的。因为 MAPO 对中间过程进行了约束，它不那么依赖“最终答对”这个噪声很大的单点信号，所以长程训练时更稳。

## 12. 这篇工作的强项
### 12.1 抓住了 agentic VLM 里一个非常本质的问题
很多视觉 RL 工作都在想“怎么设计更好的 reward”，但这篇文章点到的是更基础的一层: 模型的文字推理与工具执行可能完全脱节。这个问题一旦存在，reward 再强也会被错误轨迹污染。

### 12.2 方法非常轻量
MAPO 并没有引入重型 verifier，也没有训练额外大模型。它只是:

- 强制生成描述性 label
- 用一个预训练 CLIP 做语义核对
- 把分数塞进 advantage

从工程上看，这是非常划算的改动。

### 12.3 与现有 RL 框架兼容性强
MAPO 本质上是一个 reward/advantage augmentation 思路，可以嵌入 PPO、GRPO、GSPO 风格的训练框架中。作者也明确提到，这种“显式意图 + 执行核验”的范式不只适用于视觉工具，还可能迁移到:

- web browsing agent
- code interpreter agent
- API-calling agent

### 12.4 图像 agent 场景下收益很对症
论文最显著的提升出现在 HR-Bench 等高分辨率细粒度任务上，这说明 MAPO 的 supervision 的确在“看哪里、裁哪里、放大哪里”这些执行层面起作用了。

## 13. 局限性与潜在问题
### 13.1 CLIP 相似度不一定等价于真正的任务相关一致性
CLIP 能判断语义相似，但不一定能精确判断“这个 observation 是否正好对应当前推理所需的信息”。如果 label 本身有歧义，或者多个局部都和 label 语义相近，CLIP 可能给出误导性高分。

### 13.2 label 质量成为新的瓶颈
MAPO 的成立前提是模型能生成足够具体、足够可验证的 label。如果 label 仍然含糊、偷懒或者模板化，那么 semantic scoring 的有效性会明显下降。

### 13.3 奖励设计仍然是 task-specific 的
在视觉 crop/zoom 场景里，label 对 observation 的核验很自然；但如果动作变成复杂网页操作、代码执行或多模态工具链，如何定义可验证 observation 与匹配模型，就会更难。

### 13.4 主收益主要体现在高分辨率细粒度任务
从表中看，MAPO 并不是在所有 benchmark 上都大幅领先。它的收益分布比较集中，说明该方法最适合“动作路径是否对齐”本身就是瓶颈的场景。

## 14. 对这个仓库的启发
如果你在这个仓库里关注的是 `vlm / multimodal agent / OCR_Image2Code / 过程监督 / reward 设计`，这篇论文有几个非常可迁移的想法:

### 14.1 把隐式中间意图显式化
无论是视觉裁剪、网页点击，还是 UI element 定位，都可以先让模型显式输出“我下一步想看到什么 / 我认为目标区域应该包含什么”，再对执行结果做核验。

### 14.2 reward 不要只看最终答案
对长轨迹任务，final outcome reward 太 sparse。可以考虑把 reward 分成:

- outcome correctness
- process consistency
- tool execution validity
- observation-target match

然后在 advantage 里做可控融合。

### 14.3 对 OCR / Image2Code 方向尤其有借鉴价值
如果模型需要多步定位 UI 元素、读局部文本、理解布局，再生成代码，那么可以借鉴 MAPO 的范式:

- 先生成要观察的局部元素描述
- 执行 crop / detect / OCR
- 用语义匹配或结构匹配核验 observation 是否符合预期
- 再把核验结果变成中间 reward

这样比只在最终代码渲染后打分，更能减少 credit assignment 的噪声。

### 14.4 可以进一步探索比 CLIP 更任务专用的 verifier
对你的项目，如果关注图表、UI、OCR 或 code generation，可能可以把 CLIP verifier 替换为更贴任务的判别器，例如:

- OCR text consistency scorer
- DOM / layout consistency scorer
- chart element consistency scorer
- code-render alignment scorer

这会比通用 CLIP 更细粒度。

## 15. 我对这篇论文的总体判断
这篇工作最有价值的地方，不是提出了一个复杂的新 RL 算法，而是把多模态 agent 训练中的一个核心病灶讲清楚了: `会说` 不等于 `会做`，而 outcome reward 往往会把两者混在一起，导致错误轨迹被误强化。

MAPO 给出的解决办法也很漂亮:

- 先显式表达动作意图
- 再验证执行结果
- 最后把这种过程一致性信号注入 policy optimization

对 `thinking with images` 这条线来说，这是一个很自然、很可扩展、也很工程友好的方向。它尤其适合作为今后各种 multimodal agent reward 设计的基础模板。

## 16. 可直接复用的结论
- 如果任务依赖多步视觉探索，单纯 outcome reward 很容易掩盖错误动作执行。
- 让模型生成可验证的中间意图描述，是把过程监督引入 agent RL 的有效手段。
- CLIP 这类轻量语义 verifier 可以作为中间 reward 构件，但最好结合任务特定 verifier 进一步增强。
- 高分辨率细粒度视觉推理，是这类过程一致性优化最能体现价值的场景。
