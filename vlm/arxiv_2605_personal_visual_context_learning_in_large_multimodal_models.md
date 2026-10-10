# Personal Visual Context Learning in Large Multimodal Models

## 1. 基本信息

- **标题**: Personal Visual Context Learning in Large Multimodal Models
- **arXiv ID**: 2605.10936
- **主题**: VLM
- **总结粒度**: 细粒度
- **论文链接**: [Abs](https://arxiv.org/abs/2605.10936) | [PDF](https://arxiv.org/pdf/2605.10936.pdf)
- **项目页**: [PersonalVCL](https://vision.cs.utexas.edu/projects/PersonalVCL)

---

## 2. 这篇论文在讲什么

这篇工作关注一个此前几乎没有被系统定义的问题：如果未来的智能眼镜、可穿戴摄像头持续记录用户的第一视角视觉流，那么大模型能否利用“这个用户自己的视觉历史”去回答个性化问题。作者把这种能力形式化为 **Personal Visual Context Learning, Personal VCL**。

它和常见的 VLM/VQA 任务不一样，关键难点不在于“看懂一般图像”，而在于：

1. 模型需要利用的是**用户私有的视觉上下文**，这些信息不在预训练语料里。
2. 任务往往要求**细粒度匹配与比较**，例如识别“是不是我的那只杯子”“这个动作和我平时的做法相比是否异常”。
3. 用户历史通常由**多张图/多个视频片段组成**，模型需要跨多个观察样本进行聚合，而不是只看单张图。

作者的核心结论是：即便当前前沿 LMM 已经很强，它们在“如何真正利用个人视觉上下文”这件事上仍然存在明显缺口。于是作者提出了一个推理时基线方法 **Agentic Context Bank**，尝试把分散的个人视觉上下文组织成结构化记忆，再按查询自适应地选择证据。

---

## 3. 论文的核心问题定义

作者把一个 Personal VCL 样本定义为一对 $(\mathcal{C}, \mathcal{Q})$：

- 上下文 $\mathcal{C} = (\mathcal{L}_c, \mathcal{V}_c)$
- 查询 $\mathcal{Q} = (\mathcal{L}_q, \mathcal{V}_q)$

其中：

- $\mathcal{L}_c$ 是对上下文的语言说明，例如“这些是我的水杯参考图”或“这些是我平时切菜的视频”
- $\mathcal{V}_c$ 是与之对应的视觉材料，可能是多张图片，也可能是多个视频片段
- $\mathcal{L}_q$ 是用户提出的问题
- $\mathcal{V}_q$ 是与问题相关的新图像或新视频

模型在**冻结参数、仅做推理**的前提下，需要利用给定的个人视觉上下文来回答这个个性化查询。

作者特别强调，这篇工作有意把问题聚焦到 **context utilization**，也就是“已经拿到相关上下文后，模型会不会用”。它并不研究如何从长时视觉流里检索出相关片段，而是假设相关片段已经被取出。

---

## 4. Benchmark：Personal-VCL-Bench

作者认为，一个人的“个人视觉世界”主要由三类知识构成：

1. **Persons**：我经常接触的人是谁
2. **Objects**：我反复使用的私人物品是什么
3. **Behavior**：我平时做事的动作习惯是什么

在这三类能力之上，作者再设计了一个综合性 capstone 任务 **EgoWearer Identification**，要求模型判断一个新的一人称视频是不是来自“同一个佩戴者”。

### 4.1 任务总览

| Axis | Task ID | 任务 | 来源数据集 | 上下文到查询模态 | 平均上下文数 | 样本数 |
|------|---------|------|------------|------------------|--------------|--------|
| Persons | PerID | 单人身份识别 | EgoLife | image -> image | 5.00 | 250 |
| Persons | PerRel | 关系推理 | EgoLife | image -> video | 25.00 | 125 |
| Objects | ObjID | 目标实例识别 | Ego4D | image -> video | 1.42 | 660 |
| Objects | ObjDet | 个性化目标检测 | Ego4D | image -> image | 1.72 | 138 |
| Behavior | BehErr | 程序性错误检测 | CaptainCook4D | video -> video | 1.98 | 370 |
| Behavior | BehQA | 程序性问答 | CaptainCook4D | video -> video | 2.40 | 98 |
| EgoWearer | EgoID | 佩戴者身份识别 | EgoLife | video -> video | 5.00 | 614 |

整套 benchmark 共 **2,255** 个 context-query 样本，平均每个查询配有 **4.25** 个上下文图像或视频。

### 4.2 Benchmark 示意图

下图展示了 benchmark 的整体设定：给定若干上下文图像/视频，再对一个新的查询图像或视频做个性化判断。

![benchmark](assets/arxiv_2605_personal_visual_context_learning_in_large_multimodal_models/benchmark_v1.png)

补充图中还给出了 person/object/ego-wearer 的具体样例：

![person-object-ego](assets/arxiv_2605_personal_visual_context_learning_in_large_multimodal_models/Blurred_Person_Object_ID.png)

Behavior 任务的具体示例如下：

![behavior](assets/arxiv_2605_personal_visual_context_learning_in_large_multimodal_models/Behavior_final.png)

### 4.3 各任务的细粒度含义

**Persons 轴**

- `PerID`：给出某个人的参考图，判断该人是否出现在查询图像中。
- `PerRel`：先通过上下文辨认多个家庭成员，再回答查询视频中的社交关系问题。

这里的难点不是一般的人脸识别，而是跨天、跨场景、跨姿态地把“这个用户视觉历史中的某个人”与新画面绑定。

**Objects 轴**

- `ObjID`：判断“我的某个具体物体实例”是否出现在查询视频中。
- `ObjDet`：在查询图像中定位该具体物体实例。

作者特意使用同类别负样本，防止模型仅靠类别级别信息作弊，例如只知道“这是瓶子”，却不知道“是不是我的那个瓶子”。

**Behavior 轴**

- `BehErr`：参考视频展示用户过去的正确执行方式，模型需要判断查询视频是否偏离该用户自己的基线。
- `BehQA`：围绕用户行为中的细粒度一致性或差异做多选问答。

这类任务不是问“动作对不对”，而是问“和这个人自己平时的做法相比有没有变化”。

**EgoWearer Identification**

这是整篇论文最有代表性的任务。模型需要根据用户过去的一人称视频历史，判断一个新的一人称视频是不是来自同一个佩戴者。由于一人称视角本身看不到佩戴者脸部，模型只能依赖间接线索：

- 手部和身体局部外观
- 穿戴和配饰
- 经常使用的个人物品
- 动作节奏和习惯

这个任务天然要求跨 people / objects / behavior 三类证据做联合推理。

---

## 5. 作者先做了什么诊断

在提出方法之前，作者先用 Personal-VCL-Bench 系统诊断当前前沿 LMM 到底卡在哪里。

### 5.1 五种上下文使用方式

作者比较了五种设置：

1. `No-context`：只给查询，不给个人上下文
2. `Language-ctx (k=1)`：把一个上下文样本先转成文本描述，再给模型
3. `Language-ctx (k_max)`：把所有上下文样本都转成文本描述，再给模型
4. `Visual-ctx (k=1)`：直接给一个原始视觉上下文
5. `Visual-ctx (k_max)`：直接给全部原始视觉上下文

这里有两个核心比较维度：

1. **模态比较**：直接给原始视觉证据，是否明显优于给文本描述
2. **规模比较**：从一个上下文扩展到全部上下文，性能是否稳定上升

### 5.2 诊断结果

| Model | Context Regime | PerID | PerRel | ObjID | ObjDet | BehErr | BehQA | EgoID |
|------|----------------|------:|------:|------:|------:|------:|------:|------:|
| Qwen3-VL-8B | No-context | 50.50 | 36.00 | 49.06 | 68.12 | 49.46 | 19.39 | 49.83 |
| Qwen3-VL-8B | Language-ctx ($k_{\max}$) | 62.75 | 48.80 | 67.25 | 77.54 | 54.86 | 22.45 | 54.53 |
| Qwen3-VL-8B | Visual-ctx ($k_{\max}$) | 85.25 | 46.40 | 73.67 | 76.81 | 55.14 | 22.45 | 50.00 |
| Gemma-4-31B | No-context | 49.25 | 22.40 | 51.08 | 57.97 | 52.43 | 29.59 | 50.13 |
| Gemma-4-31B | Language-ctx ($k_{\max}$) | 78.00 | 49.60 | 78.75 | 71.74 | 58.11 | 34.69 | 52.99 |
| Gemma-4-31B | Visual-ctx ($k_{\max}$) | 75.75 | 46.40 | 75.34 | 65.22 | 59.46 | 27.55 | 55.36 |
| Gemini-3-Flash | No-context | 55.75 | 42.40 | 49.32 | 64.49 | 57.84 | 37.76 | 49.47 |
| Gemini-3-Flash | Language-ctx ($k_{\max}$) | 81.75 | 59.20 | 82.35 | 78.99 | 54.59 | 48.98 | 55.26 |
| Gemini-3-Flash | Visual-ctx ($k_{\max}$) | 88.00 | 54.40 | 82.47 | 82.61 | 67.03 | 61.22 | 55.74 |

### 5.3 两个关键悖论

**悖论 1：Modality paradox**

直觉上，原始视觉上下文包含的信息应当比文本描述更多，因此 `Visual-ctx` 应该稳定优于 `Language-ctx`。但论文发现并不是这样：

- 在很多任务上，文本上下文和视觉上下文表现接近
- 某些任务上，文本上下文甚至更强

这说明当前 LMM 并不擅长“直接吃视觉证据并做细粒度利用”，而更像是在借助自己更熟悉的语言通道做压缩推理。

**悖论 2：Scaling paradox**

直觉上，更多上下文应该更好，但作者发现从 $k=1$ 扩到 $k_{\max}$ 后，性能并不会稳定增长，尤其在：

- 行为理解
- 多证据聚合
- EgoWearer identification

这些任务上更明显。也就是说，模型并不擅长把多张图、多段视频里的个人线索整合起来。

---

## 6. 方法：Agentic Context Bank

针对上面的两类失败模式，作者提出 **Agentic Context Bank**。它是一个**与 backbone 解耦、仅推理时使用**的方法，不需要重新训练模型参数。

整体分为两个阶段：

1. **Stage I: Structured Bank Construction**
2. **Stage II: Query-Adaptive Evidence Selection**

### 6.1 整体示意图

论文的 teaser 图很好地表达了问题背景：用户长期视觉历史经过整理后，支持个性化问答与比较。

![teaser](assets/arxiv_2605_personal_visual_context_learning_in_large_multimodal_models/teaser_v1.png)

下面这张图更接近方法工作流，展示了从多段 context 构建 memory bank，再按 query 请求局部视觉证据的过程。

![qualitative](assets/arxiv_2605_personal_visual_context_learning_in_large_multimodal_models/qualitative_v1.png)

### 6.2 Stage I：把离散上下文变成结构化记忆

作者不把 $k$ 个上下文样本简单当作 $k$ 个独立 prompt，而是把它们看成关于“同一个用户”的累积证据。

Memory bank 中的每个 entry 写作 $(\tau, d, e)$，含义分别是：

- $\tau$：记忆类型
- $d$：自然语言形式的记忆描述
- $e$：与该描述绑定的原始视觉证据

其中 $\tau \in \{\textsc{appearance}, \textsc{owned\_objects}, \textsc{behavior}\}$，分别对应：

- `appearance`：外观线索，例如手部、衣物、穿戴物
- `owned_objects`：经常使用的私人物品
- `behavior`：动作模式、执行节奏、行为特征

视觉证据 $e$ 的形式也分两种：

- 对 `appearance` / `owned_objects`，通常是一张支持帧
- 对 `behavior`，通常是一个视频时间段

#### Stage I 的两个子操作

**1. Extract**

对每个 context item 提取候选 cue，例如：

- 佩戴者戴什么手环
- 使用什么类型的工具
- 手部动作节奏有什么特点

**2. Merge**

把新提取的 cue 和已有 bank 做融合，具体有四种更新操作：

- `ADD`：新增一个全新的线索
- `CONFIRM`：已有线索被新样本再次支持
- `REVISE`：新证据让旧描述更精确
- `RETRACT`：发现旧线索不可靠，撤回

这一步很关键，因为作者想强调：个人视觉记忆不是简单拼接，而应该是一个**自我修正、逐步压缩、证据绑定的 memory structure**。

### 6.3 Stage II：按查询自适应地调取证据

如果把整个 bank 的所有原始视觉证据一次性都塞给模型，会带来两个问题：

1. 上下文窗口重且冗余
2. 多个视觉样本一起喂入时，模型更不容易聚焦到真正关键的证据

因此作者设计了一个两步式推理：

**Call 1: text triage**

- 先只给模型 bank 的文本视图 $T_{\mathcal{B}}$
- 每条 entry 包含类型、描述和稳定 ID
- 模型先判断自己能否只靠文本回答
- 如果不能，就返回它想查看的 entry ID

**Call 2: selective visual verification**

- 系统只把这些被请求的 entry 所对应的视觉证据内联回 prompt
- 得到混合视图 $H_{\mathcal{I}}$
- 模型再基于“少量、针对性的视觉证据”给出最终答案

这个设计对应两种诊断结论：

- Stage I 主要应对 **scaling paradox**
- Stage II 主要应对 **modality paradox**

换句话说，作者的观点不是“原始视觉上下文没用”，而是“原始视觉上下文不能粗暴堆进去，必须先组织，再按需访问”。

---

## 7. 实验结果

### 7.1 Gemma-4-31B 上的全任务结果

| 方法 | PerID | PerRel | ObjID | ObjDet | BehErr | BehQA | EgoID |
|------|------:|------:|------:|------:|------:|------:|------:|
| No-context | 49.25 | 22.40 | 51.08 | 57.97 | 52.43 | 29.59 | 50.13 |
| Language-ctx ($k_{\max}$) | 78.00 | 49.60 | 78.75 | 71.74 | 58.11 | 34.69 | 52.99 |
| Visual-ctx ($k_{\max}$) | 75.75 | 46.40 | 75.34 | 65.22 | 59.46 | 27.55 | 55.36 |
| **Agentic Context Bank** | **83.25** | **51.20** | **82.52** | **73.19** | **66.22** | **35.71** | **61.60** |
| **相对更强 baseline 的提升** | **+5.25** | **+1.60** | **+3.77** | **+1.45** | **+6.76** | **+1.02** | **+6.24** |

这张表说明作者的方法并不是只对某一个任务有效，而是在：

- persons
- objects
- behavior
- ego-wearer

四类任务上都优于标准 prompt 方式。

其中提升最大的任务之一是 `BehErr` 和 `EgoID`，这也符合论文预期，因为这两类任务尤其依赖多证据聚合和细粒度比较。

### 7.2 跨 backbone 的 EgoWearer 结果

| 类别 | 方法 | GPT-5.4-mini | Gemma-4-31B | Gemini-3-Flash |
|------|------|-------------:|------------:|---------------:|
| Baseline | No-context | 49.83 | 50.13 | 49.47 |
| Baseline | Language-ctx ($k_{\max}$) | 49.86 | 52.99 | 55.26 |
| Baseline | Visual-ctx ($k_{\max}$) | 51.47 | 55.36 | 55.74 |
| Ours | Descriptors only | 50.69 | 51.97 | 55.28 |
| Ours | All evidence | 51.21 | 57.29 | 67.86 |
| Ours | **Adaptive evidence** | **53.77** | **61.60** | **72.82** |

这个结果很重要，因为它说明：

1. 提升并不绑定某个单一 backbone
2. 不是“有了结构化 bank 就够了”，而是“结构化 bank + 自适应证据选择”组合起来最好
3. 在 Gemini-3-Flash 上，Adaptive evidence 相比最强标准 baseline 提升达到 **17.08** 个点

### 7.3 Stage-II ablation 的含义

从上表可以读出一个清楚结论：

- `Descriptors only`：只保留文本 bank，不回看视觉证据，效果有限
- `All evidence`：把所有 bank 证据都给模型，虽有提升，但不如自适应选择
- `Adaptive evidence`：先 triage，再回看局部视觉证据，效果最好

这说明真正有价值的不是“总结成文字”或“把所有图都给满”，而是**让模型先做一次证据路由，再回到少量关键视觉证据上做核验**。

---

## 8. Supplementary 中几个很有价值的补充分析

### 8.1 Stage-I bank 构建统计

| Builder | 每 clip 候选 cue 数 | 最终 entry 数 | A/O/B 分布 | 压缩率 | Revision 占比 | 被更新 entry 占比 |
|---------|--------------------:|-------------:|------------|-------:|---------------:|------------------:|
| Gemini-3-Flash | 4.44 | 13.8 | 6.0 / 3.1 / 4.7 | 0.62 | 35.9% | 24.1% |
| Gemma-4-31B | 2.95 | 8.2 | 3.8 / 1.5 / 3.0 | 0.56 | 25.2% | 18.3% |
| GPT-5.4-mini | 6.92 | 18.4 | 6.7 / 5.7 / 5.9 | 0.53 | 38.1% | 26.5% |

这说明 bank construction 确实不是 append-only：

- 会压缩 cue
- 会修订旧记忆
- 会在 appearance / object / behavior 三类线索间保持结构化组织

### 8.2 Stage-II 查询时的证据请求统计

| Builder | 请求视觉证据比例 | 平均请求 entry 数 | 请求 A/O/B 分布 | 平均 decisive entry 数 | decisive A/O/B 分布 |
|---------|----------------:|------------------:|-----------------|-----------------------:|--------------------|
| Gemini-3-Flash | 95.8% | 3.63 | 81.8 / 15.1 / 3.0 | 2.67 | 79.2 / 15.2 / 5.5 |
| Gemma-4-31B | 96.1% | 2.86 | 85.9 / 11.2 / 2.9 | 1.78 | 80.7 / 13.3 / 6.0 |
| GPT-5.4-mini | 2.1% | 8.77 | 60.5 / 16.7 / 22.8 | 4.31 | 85.7 / 5.4 / 8.9 |

这里有两个很有意思的点：

1. Gemini 和 Gemma 几乎总会回看视觉证据，但只回看 **2 到 4 个 entry**，说明路由确实起到了“少量关键证据聚焦”的效果。
2. GPT-5.4-mini 基本不请求视觉证据，更倾向直接靠文本 bank 做判断，这也解释了它的提升幅度相对更小。

### 8.3 失败案例

作者给出了一个很典型的 failure case：

![failure](assets/arxiv_2605_personal_visual_context_learning_in_large_multimodal_models/failure_case.png)

失败的根源不是 bank 里没有相关信息，而是 **query-time evidence selection 不充分**：

- 模型请求了某个手环相关 entry 的视觉证据
- 但没有请求另一个同样关键的 bracelet entry 的原始视觉证据
- 最后仅凭文本描述误判为匹配

这说明该方法虽然比平铺 prompt 更强，但“该看哪些证据”本身依然是难点。

---

## 9. 这篇工作的主要贡献

### 9.1 问题定义贡献

作者把“personalized visual reasoning over user-specific context”系统化定义成了 **Personal VCL**，这比一般的 visual ICL、个性化概念学习、长视频检索问答都更具体，也更贴近未来 wearable assistant 的使用场景。

### 9.2 Benchmark 贡献

`Personal-VCL-Bench` 的设计很有代表性，优点在于：

- 覆盖 person / object / behavior 三类个人视觉知识
- 同时包含 image 和 video 场景
- 有一个整合三类线索的高难 capstone 任务 EgoWearer
- 重点考的是“细粒度个人差异”和“多观察聚合”

### 9.3 诊断贡献

作者不是直接报一个新方法分数，而是先证明了当前 LMM 的两类核心短板：

- 不善于直接利用原始个人视觉证据
- 不善于稳定聚合多段个人视觉历史

这两个诊断结论比单纯方法提升更重要，因为它们明确指出了后续研究的方向。

### 9.4 方法贡献

`Agentic Context Bank` 的价值不在于复杂训练，而在于它给出了一个很清楚的 inference-time design pattern：

1. 先把原始个人上下文整理成结构化、可修订的 memory bank
2. 再通过 query-conditioned routing，只回看必要证据

这是一种很适合 agent / memory / multimodal personalization 方向继续扩展的接口化设计。

---

## 10. 局限性

作者自己也很坦诚地指出了几类局限：

1. **只研究 context utilization，不研究 context retrieval**
   真实场景里，用户视觉历史是长时连续流，先从中检索出相关片段本身就是难题。

2. **当前方法仍然只是强推理时基线**
   它证明了“组织方式和证据路由很重要”，但并没有从根本上解决个性化视觉推理。

3. **Stage-II 仍可能选错证据**
   failure case 说明，即便 bank 已经构好，query-time 选证据也会出错。

4. **隐私与部署风险很高**
   个人视觉历史天然包含身份、物品、生活习惯、社交关系等高敏感信息。

---

## 11. 对我们做相关研究的启发

如果你关心 VLM、agent、personal memory、长时上下文利用，这篇论文有几个很值得吸收的点。

### 11.1 不要把“检索出来了”误当作“模型会用了”

很多工作默认只要把相关图片/视频段喂给模型就行，但这篇论文明确表明：**找到上下文** 和 **利用上下文** 是两件事。

### 11.2 视觉上下文也许需要先被“结构化”

对于个人视觉历史这种多样、冗长、跨时间的证据，直接堆 prompt 很可能不是最优方式。更合理的做法可能是：

- 先提炼 typed memory
- 建立证据绑定
- 在查询时做路由与回看

### 11.3 文本视图可以作为轻量级索引层

作者并不是简单地“把视觉变文字”，而是把文字当作**索引层**，让模型先在便宜的文本空间里筛选，再回到像素空间核验。这种“两级访问机制”很值得借鉴。

### 11.4 EgoWearer 任务很像一种多模态身份连续性测试

这个任务本质上要求模型把：

- 人的外观残留线索
- 使用物品
- 行为节奏

整合成一个长期稳定的“用户画像”。这对长期记忆 agent、多模态用户建模都很有启发。

---

## 12. 一句话评价

这篇论文最重要的价值，不是单纯提出了一个新 benchmark 或一个 prompt engineering baseline，而是把 **“个人视觉上下文到底该如何被 LMM 利用”** 这个问题切开了：先证明现有模型不会稳定利用，再用 `Agentic Context Bank` 给出一个很自然的结构化记忆与按需取证框架。对未来的 wearable assistant、个性化多模态 agent、以及长期视觉记忆系统，这篇工作都很值得重视。
