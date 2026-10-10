# AlphaGRPO: Unlocking Self-Reflective Multimodal Generation in Unified Multimodal Models via Decompositional Verifiable Reward

## 1. 论文基本信息
- arXiv ID: 2605.12495
- URL: [abs](https://arxiv.org/abs/2605.12495) / [pdf](https://arxiv.org/pdf/2605.12495.pdf)
- 主题归档: `vlm`
- 总结粒度: `detailed`
- 作者: Runhui Huang, Jie Wu, Rui Yang, Zhe Liu, Hengshuang Zhao
- 作者机构: The University of Hong Kong / Bytedance Seed
- 核心方法: **AlphaGRPO**
- 核心奖励: **DVReward** (`Decompositional Verifiable Reward`)
- 项目页: [huangrh99.github.io/AlphaGRPO](https://huangrh99.github.io/AlphaGRPO/)
- 关键词: `UMM`, `AR-Diffusion`, `GRPO`, `self-reflective refinement`, `reasoning text-to-image`, `decompositional reward`, `multimodal RL`

## 2. 这篇论文在解决什么问题

这篇 paper 的目标很明确：

**能不能不用额外的 cold-start SFT，只靠 RL，把 unified multimodal model 里“潜在但没被激活”的 reasoning / self-reflection / refinement 能力逼出来，用来提升 multimodal generation？**

这里的对象不是普通 text-to-image diffusion model，而是 **Unified Multimodal Models (UMMs)**，尤其是这种：

- 一体化处理理解和生成
- 同时能走 autoregressive text token 和 diffusion image token
- 能在单一 backbone 里做 interleaved multimodal reasoning / generation

作者的核心判断是：

1. 这类 unified model 在大规模预训练后，其实已经有一些“隐式原语”
   - 比如视觉理解
   - prompt-following
   - 局部 reasoning
   - 甚至自我诊断的潜力

2. 过去很多工作想让它做更强的 reasoning T2I 或 image refinement，主要靠：
   - 强 teacher 合成高质量数据
   - 再做额外 cold-start SFT

3. 但这样有个问题：
   - 你很难分清性能提升到底来自 RL，还是来自 teacher distillation

所以作者要做的是：

**不加冷启动蒸馏阶段，直接在 AR-Diffusion unified model 上做 GRPO，看看能不能把这些 latent capability 激活出来。**

## 3. 这篇论文最重要的两个点

论文的贡献其实可以浓缩成两件事：

1. **把 GRPO 扩展到 AR-Diffusion unified multimodal trajectory**
2. **提出更稳定的 multimodal reward：DVReward**

前者解决“怎么训”，后者解决“拿什么信号训”。

## 4. 为什么作者觉得这件事可行

### 4.1 它先做了一个 pilot study

这篇 paper 不错的一点是，没有一上来就堆方法，而是先问两个前提是否成立：

1. 预训练好的 UMM 到底有没有 latent self-reflection 能力？
2. 通用 MLLM 能不能给 visual generation 提供稳定、可区分的 reward？

这两个问题分别对应它后面方法的两个核心部件。

### 4.2 发现 1：让模型“找错”比让它“判断对错”更能激活推理

![Verification vs reflection](assets/arxiv_2605_alphagrpo_unlocking_self_reflective_multimodal_generation_in_umms_via_decompositional_verifiable_reward/pilot_study1.png)

作者用 Bagel 做了两个 probing task：

1. **Verification**
   - 问模型：这张图有没有问题？

2. **Reflection**
   - 明确告诉模型：这张图有错误，请你找出来

结果很有意思：

- 在 Verification 模式下，模型经常有明显的 **confirmation bias**
  - 即使图里有明显错误，也倾向于说“整体符合用户意图”

- 在 Reflection 模式下，模型反而更容易认真检查细节，指出错位、属性冲突等问题

这个结论很关键，因为它说明：

**UMM 不是没有理解能力，而是“验证有没有错”这件事本身不够容易激活它；反过来，“请你找错”更像一个能触发 latent reasoning 的开关。**

这其实直接支撑了后面的 self-reflective refinement 训练设计。

### 4.3 发现 2：问具体问题，比直接打分更容易得到可区分 reward

![Score reward vs question reward](assets/arxiv_2605_alphagrpo_unlocking_self_reflective_multimodal_generation_in_umms_via_decompositional_verifiable_reward/pilot_study2_v3.png)

作者做了另一个很重要的比较：

- 同一个 prompt 生成两张图
- 其中一张满足空间关系，另一张不满足

然后对比两种 reward：

1. **Holistic scalar reward**
   - 直接让模型打 0-10 分
   - 用的是 VIEScore 一类方式

2. **Question-based reward**
   - 不让模型给抽象分数
   - 而是直接问一个关键问题，比如：
     - “树有没有部分挡住长椅？”
   - 再看 `Yes` / `No` token 的概率

结果是：

- scalar reward 给两张图同样的分数
- question-based reward 能明显拉开差距

作者据此得出的结论非常重要：

**通用 MLLM 不是不能分辨好坏，而是“不擅长把复杂视觉语义直接映射成稳定标量”；但如果把任务拆成可验证问题，它的判别能力会被更稳定地激活。**

这就是 DVReward 的出发点。

## 5. 方法总览

![AlphaGRPO overview](assets/arxiv_2605_alphagrpo_unlocking_self_reflective_multimodal_generation_in_umms_via_decompositional_verifiable_reward/method.png)

AlphaGRPO 的整体结构由三部分组成：

1. **统一轨迹建模**
2. **统一 GRPO 优化**
3. **DVReward 作为外部 verifier**

作者优化的不是单纯文本，也不是单纯 diffusion image trajectory，而是一个混合轨迹：

`tau = (y, z_1 -> z_0)`

其中：

- `y` 是 autoregressive reasoning text
- `z_1 -> z_0` 是 diffusion visual trajectory

这个建模非常关键，因为它允许作者说：

> reasoning text 是 image generation 的因果前驱，所以同一个 reward 可以反传到文本和图像两边。

## 6. AlphaGRPO 的核心机制

### 6.1 统一轨迹：先想，再画

作者把 multimodal generation 看成统一生成过程：

- 先采样 reasoning text token `y`
- 再把 `y` 作为条件先验，引导 image latent trajectory

这样一来，它就能统一描述两类任务：

1. **Reasoning T2I**
   - 文本中间链条负责补世界知识、做空间规划、拆约束

2. **Self-Reflective Refinement**
   - 文本中间链条负责诊断先前图像的错误，再引导修正

虽然语义上一个更像 planning，一个更像 critique，但训练形式上都可以写成：

- 先出文本 reasoning
- 再出图像 trajectory

### 6.2 统一 GRPO 目标

AlphaGRPO 的关键不是“对 image 做 RL”，而是对**整条混合轨迹**做 RL。

对每个 context `C`，采样一组 trajectories：

- `tau_i = (y_i, z_i)`

然后只根据最终图像 `z_i` 算 reward：

- `r(z_i)`

接着把 group-normalized advantage 同时作用到：

1. reasoning token policy
2. flow generation policy

也就是：

- `J_AR`
- `J_Flow`

共享一个 advantage，但分别有各自的 PPO / GRPO 风格项与 KL 项。

这一步的直觉是：

**如果某张最终图像更好，那生成它的前置 reasoning 也应该被 credit assignment 奖励。**

这就是 unified trajectory RL 的核心。

### 6.3 False-Positive Rectification (FPR)

这是这篇 paper 里一个挺实用的工程点。

在 self-reflective refinement 任务中，作者需要比较：

- 初始图
- refinement 后的图

但如果只是用 group advantage，可能会出现一种情况：

- 某些 refinement 其实没有比原图更好
- 但在当前采样组里还是拿到了正 advantage

这样会产生 **false-positive optimization**。

为了解决这个问题，作者引入了 **False-Positive Rectification (FPR)**：

- 如果 refinement 结果没有超过初始图：
  - `r(z_i) <= r(z_init)`
- 就直接把它的 reward 赋成 group 最小值

这样做的效果是：

- 所有“没改好甚至改坏”的 refinement 轨迹都会拿负 advantage
- 防止模型学会“看起来在修，实际在瞎改”

这个设计很像给 self-refinement 加了一个 hard validity constraint。

## 7. DVReward：这篇最值钱的部分

我觉得这篇 paper 最有意思、最可能迁移到别处的，不一定是 AlphaGRPO 本身，而是 **DVReward**。

### 7.1 为什么不用 holistic scalar score

作者明确批评了像 VIEScore 这类 holistic scalar reward：

- 分数标定不稳定
- 区分细粒度差异的能力差
- 容易把多个语义错误“平均掉”
- 对 GRPO 来说，reward 噪声太大会直接污染优化

这和你前面一直关心的“细粒度质量信号”其实高度一致。

### 7.2 Request decomposition

DVReward 的第一步是让 LLM 先把用户请求拆成一组 **atomic, verifiable questions**。

拆分的内容分两类：

1. **Semantic questions**
   - entity existence
   - 属性
   - 空间关系
   - 组合约束

2. **Quality questions**
   - 几何完整性
   - 纹理保真度
   - 视觉质量相关项

作者还特别强调一点：

**要把抽象语义 grounding 成可观察的物理现象。**

例如：

- 不是问 “咖啡是不是热的”
- 而是问 “杯子上方有没有蒸汽”

这个设计非常强，因为它把抽象 reward 转成了一组更接近可验证观测的问题。

### 7.3 Confidence scoring

DVReward 不是让 verifier 回答一个硬二值 0/1，也不是让它打整体分。

它做的是：

- 对每个问题，拿 verifier（Qwen3VL-30B-A3B）
- 看 `Yes` 和 `No` token 的概率
- 用：

`P(Yes) / (P(Yes) + P(No))`

作为连续置信度分数

这比 binary score 更细腻，因为：

- “勉强对”
- “高度确定地对”

不会被压成一样的 1。

### 7.4 最终 reward 组合

最终 reward 用 semantic score 和 quality score 的几何平均：

`r(z) = sqrt(v_sem * v_qua)`

这个组合说明作者不想让模型只追 semantic correctness，而忽略视觉质量；
也不想只追画质，而忽略 prompt alignment。

这点很像把“对齐”和“保真”绑定成一个更稳的联合优化信号。

## 8. 训练数据与训练设置

### 8.1 训练数据：Primitive-to-Prompt

作者采用一个 bottom-up 的 prompt 合成方式：

- 先准备视觉原语池
  - objects
  - attributes
  - spatial relations

- 参考 TIIF-Bench taxonomy 定义 39 种 compositional task
- 用 Qwen3-235B-A22B 合成 prompt
- 按 Easy / Medium / Hard 三档采样

总共生成：

- 19,500 个训练 prompt
- 1,024 个测试 prompt

并对每个 prompt 预生成：

- `q`
- `Q_sem`
- `Q_qua`

也就是说，训练前就把 decomposer 跑完，在线训练时主要是 verifier 在看生成图是否满足这些问题。

### 8.2 训练细节

论文给了不少实现细节：

- base model: **Bagel**
- LoRA:
  - `r = 32`
  - `alpha = 64`
- 训练硬件:
  - 64 x NVIDIA A100
- 每 8 张卡里：
  - 7 张训练
  - 1 张跑 verifier (`Qwen3VL-30B-A3B`)
- 不做 cold-start stage
- 训练步数：
  - 380 steps
- rollout batch:
  - 每步 32 prompts
- group size:
  - `G = 14`
- noise level:
  - `a = 0.7`
- train resolution:
  - `512`
- `lambda = 0.2`
- `beta_AR = beta_Flow = 0`

此外：

- reasoning T2I 训练时：
  - 16 steps 采样
- self-reflective refinement：
  - 初始生成和 refinement 都用 40 steps

### 8.3 训练效率

DVReward 需要多次 verifier 调用，本来会很慢。作者通过几件事把这个问题压下去：

1. KV cache 复用
2. question 并行验证
3. decentralized reward serving
4. asynchronous scheduling

最终 reward-waiting bubble time 从：

- `40.8s`

降到：

- `9.72e-6 s`

这个结果很夸张，但至少说明作者在 reward serving 上下了不少工程功夫。

## 9. 主要实验结果

### 9.1 Text-to-Image 总体结果

![Qualitative and quantitative teaser](assets/arxiv_2605_alphagrpo_unlocking_self_reflective_multimodal_generation_in_umms_via_decompositional_verifiable_reward/teaser.png)

论文主表横跨多个 benchmark：

- TIIF-Bench
- WISE
- DPG-Bench
- GenEval

下面只抓最核心的对比。

#### 512 分辨率

| Model | TIIF Overall S | TIIF Overall L | DPG | GenEval |
| --- | ---: | ---: | ---: | ---: |
| BAGEL | 75.2 | 78.6 | 85.0 | 84.0 |
| AlphaGRPO (RT2I) | 78.9 | 79.5 | 86.0 | 85.0 |
| AlphaGRPO (SRR) | 79.1 | 79.5 | 86.3 | 84.2 |
| + Inference SRR | **83.9** | **83.2** | **87.9** | **88.2** |

这里最重要的观察是：

1. **无论训练在 RT2I 还是 SRR 上，都会比 BAGEL 强**
2. **SRR 训练本身就能泛化到标准 T2I benchmark**
3. **inference-time self-reflective refinement 再往上推一截**

也就是说，这篇 paper 不是只证明“训练某个特定任务能涨这个任务”，而是在强调：

**self-reflective refinement 这种训练形态本身能学到比较通用的 multimodal generation alignment。**

#### 1024 分辨率

| Model | TIIF Overall S | TIIF Overall L | WISE | DPG | GenEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| BAGEL | 76.4 | 76.2 | 0.53/0.70 | 85.1 | 86.6 |
| AlphaGRPO (RT2I) | 78.7 | 79.5 | 0.50/0.69 | 85.9 | 87.4 |
| AlphaGRPO (SRR) | 77.7 | 78.1 | **0.53/0.71** | 85.1 | 86.1 |
| + Inference SRR | **82.2** | **81.4** | - | **86.6** | **89.5** |

这张表支撑了作者的一个关键 claim：

**虽然训练主要在 512 分辨率进行，但学到的对齐能力能迁移到更高分辨率。**

作者用这个来论证：

- 学到的不是像素层面的投机技巧
- 更像是更稳的 semantic alignment / reasoning behavior

### 9.2 Image editing：没专门训 editing，也能涨

GEdit-Bench 上：

| Model | G_SC | G_PQ | G_O |
| --- | ---: | ---: | ---: |
| BAGEL | 7.36 | 6.83 | 6.52 |
| AlphaGRPO (RT2I) | 7.54 | 7.18 | 6.85 |
| AlphaGRPO (SRR) | **7.67** | **7.46** | **7.08** |

这个结果很有说服力，因为作者说：

- 没有在 editing task 上专门训练
- 但 SRR 训练仍然显著提升 editing benchmark

这进一步说明：

**它强化的不是某个 narrow task，而是更一般的“看懂指令 - 发现偏差 - 做修正”能力。**

## 10. Ablation：DVReward 和 FPR 是否真的重要

### 10.1 对比其他 reward model

| Base | Reward | TIIF-S | TIIF-L | GenEval |
| --- | --- | ---: | ---: | ---: |
| SD3.5M | baseline | 74.0 | 73.2 | 79.8 |
| SD3.5M | PickScore | 77.6 | 76.3 | 80.4 |
| SD3.5M | VIEScore | 76.2 | 72.9 | 82.9 |
| SD3.5M | DVReward | **79.1** | **77.7** | **86.0** |
| BAGEL | baseline | 75.2 | 78.6 | 84.0 |
| BAGEL | HPSv3 | 78.5 | 77.1 | 83.4 |
| BAGEL | UnifiedReward | **79.2** | 77.3 | 83.7 |
| BAGEL | VIEScore | 79.1 | 77.9 | 81.7 |
| BAGEL | DVReward | 78.9 | **79.5** | **85.1** |

这里的信息量很大：

- 一些常见 reward model 在某些 benchmark 上会退化
- VIEScore 尤其不稳
- DVReward 是最 consistently positive 的

这正好呼应它的 pilot study：  
**不是所有“看起来懂图”的 MLLM reward 都能稳定转成训练信号。**

### 10.2 Confidence score 比 binary score 更好

| Reward scoring | TIIF-S | TIIF-L | GenEval |
| --- | ---: | ---: | ---: |
| Binary | **79.0** | 78.9 | 84.0 |
| Confidence | 78.9 | **79.5** | **85.1** |

这说明作者的 intuition 是对的：

- binary 太粗
- confidence score 保留了 verifier 的不确定性
- 对优化来说更平滑

### 10.3 FPR 有用

| FPR | TIIF-S | TIIF-L | GenEval |
| --- | ---: | ---: | ---: |
| on | **79.1** | **79.5** | **84.2** |
| off | 77.9 | 77.8 | 83.7 |

提升不是爆炸性的，但很稳定。

这类 ablation 很像“训练卫生指标”：

- 没它也能训
- 但会更容易被 noisy reward 带偏

### 10.4 Quality question 也重要

| Semantic | Quality | TIIF-S | TIIF-L | GenEval |
| --- | --- | ---: | ---: | ---: |
| on | off | 76.3 | 77.7 | 87.3 |
| on | on | **78.7** | **79.5** | **87.4** |

这说明：

- 只看 semantic alignment 不够
- 质量问题要显式问出来
- 否则会出现“语义对了，但图变丑了”的情况

## 11. 额外分析

### 11.1 问题数量不能压太狠

作者还做了 question count ablation：

| Questions | TIIF-S | TIIF-L | GenEval | Spatial | Count |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 75.2 | 78.6 | 84.0 | 71.0 | 74.0 |
| 10 | 78.2 | 79.7 | 81.1 | 63.0 | 71.8 |
| All | **78.9** | 79.5 | **85.0** | **74.3** | **82.5** |

这很有意思：

- 少问题在 TIIF 上还行
- 但在 GenEval，尤其 Spatial 和 Count 上明显掉

说明：

**把多个约束压缩成几个粗问题，会损伤细粒度 reward 的分辨率。**

这和你前面一直在关心的“细粒度信号”高度一致。

### 11.2 人类偏好也更喜欢它

200 个 real-user prompts 上的人类评估：

- Overall：`40.5%` 更偏好 AlphaGRPO，`30.5%` 偏好 BAGEL
- Prompt following：`36.5%` vs `25.5%`
- Perceptual quality：`43.0%` vs `39.0%`

DVReward 在 pairwise 比较里也更多选择 AlphaGRPO：

- `51.3%` vs `34.7%`

这至少说明：

- 它的 reward 没有明显朝离谱方向 overfit
- 和人类偏好的方向大体一致

### 11.3 Inference-time SRR 不是“零训练也行”

作者还比较了：

- BAGEL + inference-time SRR
- AlphaGRPO + inference-time SRR

结果：

- BAGEL + Inf SRR 只有有限提升
- AlphaGRPO + Inf SRR 提升明显更大

这说明：

**不是只要在推理时让模型反思一下就行，训练阶段真的把 self-reflection 行为强化进去了。**

## 12. 这篇工作的亮点

### 12.1 它把 RL 的 credit assignment 真正跨过了“文本到图像”边界

很多多模态 RL 工作要么只训文本 reasoning，要么只训图像生成。

这篇 paper 的一个漂亮点在于，它明确把：

- reasoning text
- diffusion image generation

放进同一条轨迹里做 credit assignment。

这在 unified model 上是很自然、但又不那么容易做好的事情。

### 12.2 DVReward 的思想很强

我觉得这篇 paper 最可迁移的资产是：

**把复杂 multimodal request 分解成 atomic verifiable questions，再用 token-level confidence 组合 reward。**

这套思路不只适用于 T2I：

- image editing
- VLM grounded generation
- agentic perception / verification
- 甚至你前面聊过的细粒度重要性估计

都可以借鉴这个“先拆，再验”的信号构造方式。

### 12.3 self-reflective refinement 这个任务定义很有未来感

相比直接做 reasoning T2I，我觉得这篇更有意思的是：

**让 unified model 学会先看自己画得哪里不对，再改。**

这其实更接近长期想象中的“自校正生成系统”。

## 13. 局限性

论文在 appendix 里也承认了一些限制。

### 13.1 Base model 还不够稳

作者明确说：

- Bagel 在 512 分辨率下偶尔会有噪声或模糊伪影
- self-reflection 生成的 reasoning 也会出现意外模式

所以 AlphaGRPO 的上限部分还被 base model 束缚。

### 13.2 目前还是 outcome reward 为主

现在的 reward 主要看最终结果，没有真正做 process reward。

作者提到未来可以加：

- refinement 前后的一致性奖励
- 中间反思步骤是否识别了正确错误的监督

这点我也很认同，因为：

**如果只看最终图像，模型可能学会“胡乱自省但碰巧改对”，而不是学会“正确诊断错误”。**

### 13.3 训练代价不低

尽管 reward server 做了大量优化，但整套训练仍然不便宜：

- 64 A100
- 外挂一个大 verifier
- 需要复杂的异步 serving

所以它目前更像一个“高端实验室 setting 下的方向验证”，还不是轻量方案。

## 14. 对我们最有启发的点

### 14.1 如果你关心“细粒度质量监督”，这篇很值得反复看

你前面一直在问：

- 不依赖 attention score 的细粒度重要性
- 更稳定的质量信号
- 怎么避免粗糙标量指标的偏差

DVReward 本质上就是一个很好的答案范式：

1. 不直接问“整体好不好”
2. 把请求拆成多条可验证细项
3. 对每一项拿 soft confidence
4. 再聚合成最终 reward

这比单一 scalar reward 要靠谱得多。

### 14.2 Self-reflection 其实可以看成一种“显式错误搜索”

这和你前面问 critical token 那里有点神似。

在 critical token 那个问题里，你关心的是：

- 哪一步开始走偏了

在这里，作者关心的是：

- 当前生成图里哪里偏了
- 如何显式指出偏差，再做修正

某种意义上，这篇 paper 是把“找出错误分叉点”这件事，从 token reasoning 扩展到了 multimodal generation 结果本身。

### 14.3 以后可以把 DVReward 搬到更细粒度 VLM 任务

我觉得很自然的扩展包括：

- region-level grounded generation reward
- search-to-pixel verifier reward
- object-level compositional editing reward
- long-document VLM 中 page / region 级验证奖励

尤其如果你后面想做：

- VLM 里的细粒度 agentic verification
- 多阶段生成 / 多阶段感知中的局部质量回传

这篇很有参考价值。

## 15. 一句话总结

**AlphaGRPO 最重要的不只是“把 GRPO 用到 unified multimodal generation 上”，而是它给出了一个很像样的答案：如果你想让统一模型学会自我反思和细粒度修正，奖励信号不能再是粗糙的一维分数，而要先把复杂请求拆开，再逐项验证。**
