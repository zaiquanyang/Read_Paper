# From Web to Pixels: Bringing Agentic Search into Visual Perception

## 1. 论文基本信息
- arXiv ID: 2605.12497
- URL: [abs](https://arxiv.org/abs/2605.12497) / [pdf](https://arxiv.org/pdf/2605.12497.pdf)
- 主题归档: `vlm`
- 总结粒度: `detailed`
- 作者: Bokang Yang, Xinyi Sun, Kaituo Feng, Xingping Dong, Dongming Wu, Xiangyu Yue
- 作者机构: Shenzhen Loop Area Institute / Wuhan University / CUHK MMLab
- 项目主页: [pixel-searcher.github.io](https://pixel-searcher.github.io/)
- 代码: [yangbokang/Pixel-Searcher](https://github.com/yangbokang/Pixel-Searcher)
- 核心概念: **Perception Deep Research**
- 核心 benchmark: **WebEyes**
- 核心方法: **Pixel-Searcher**
- 关键词: `agentic search`, `grounded perception`, `visual grounding`, `segmentation`, `search-based VQA`, `open-world perception`

## 2. 这篇论文到底在解决什么问题

这篇工作的切入点很新，而且和普通 grounding / segmentation 工作不太一样。

传统视觉感知任务通常默认：

- 要找的目标能直接从图像里看出来
- 或者 query 已经把目标说得够清楚
- 再或者模型内部知识就足够补完缺失信息

但作者指出，真实世界里经常不是这样。很多 query 指向的目标，真正决定身份的线索并不在图里，而在图外：

- 某个最近发生的事件
- 某个品牌、创作者、角色关系
- 某个长尾实体的外部知识
- 某个需要 multi-hop 才能串起来的事实链

于是问题变成：

**图里的物体是“可见的”，但你得先靠外部搜索把它是谁搞清楚，才能把它在图中定位出来。**

作者把这个 setting 命名为：

**Perception Deep Research**

也就是：

- 不是只做 “language to region”
- 也不是只做 “search to answer”
- 而是做 **search to grounded perception**

换句话说，这篇 paper 关心的是：

> 一个视觉系统能不能先主动去网上搜证据、做身份解析，再把这个被外部知识解析出来的目标绑定到图像里的具体实例上，最后输出 box、mask 或 grounded answer。

这是这篇论文最核心的贡献点。

## 3. 论文提出了什么

这篇 paper 实际上提出了两件事：

1. 一个新的任务定义和 benchmark：**WebEyes**
2. 一个基于这个任务的 agentic workflow：**Pixel-Searcher**

这两部分最好分开看。

### 3.1 从任务角度：Perception Deep Research

作者认为 open-world visual perception 里有一个被低估的 gap：

- 不是“模型能不能看见对象”
- 而是“模型能不能把外部证据解析成对象身份，再把身份绑定回图像实例”

这个 gap 的难点在于：

- query 提供的是间接 clues
- 图像里可能有多个外观相似的候选实例
- 外部证据可能只告诉你“它是谁”，不直接告诉你“它在哪”

所以真正的关键是：

**entity resolution + instance binding**

### 3.2 从系统角度：Pixel-Searcher

作者没有直接拿一个更强的 VLM 去 end-to-end 预测 box/mask，而是做了一个可分解的 workflow：

- 先做 agentic search and target resolution
- 再做 agentic grounding and tool use

也就是说，它的核心不只是“多搜一下”，而是把搜到的 evidence 结构化成一个适合视觉 grounding 的 target hypothesis。

## 4. WebEyes Benchmark

![Perception Deep Research teaser](assets/arxiv_2605_from_web_to_pixels_bringing_agentic_search_into_visual_perception/compare2.png)

![WebEyes generation and Pixel-Searcher inference](assets/arxiv_2605_from_web_to_pixels_bringing_agentic_search_into_visual_perception/compare.png)

WebEyes 是这篇论文最重要的资产之一。它不是简单造一批“要搜索”的视觉问答题，而是围绕**对象级标注**来组织整个 benchmark。

### 4.1 WebEyes 的任务形式

WebEyes 建在统一的 object-level annotation layer 上，支持三种 task view：

1. **Search-based Grounding (SearchGround)**
   - 输入：image + knowledge-intensive query
   - 输出：目标对象的 bounding box

2. **Search-based Segmentation (SearchSeg)**
   - 输入：image + knowledge-intensive query
   - 输出：目标对象的 mask

3. **Search-based VQA (SearchVQA)**
   - 输入：image + highlighted region + answer options
   - 输出：与高亮区域匹配的正确知识型描述

这个设计很妙，因为它让同一组 object annotation 同时支持：

- box-level grounding
- pixel-level segmentation
- region-level knowledge verification

所以它不是单一输出格式 benchmark，而是一个“从外部证据到视觉实例”的多视图评测平台。

### 4.2 数据规模

WebEyes 的规模不算特别大，但设计得很密：

- 120 张图像
- 473 个标注对象实例
- 645 个唯一 QA 对
- 1927 个任务样本
  - 645 SearchGround
  - 645 SearchSeg
  - 637 SearchVQA

这里最重要的不是绝对规模，而是每个样本都带着：

- object annotation
- 外部证据链
- target identity
- grounded output

所以它强调的是**traceable evidence-to-pixel evaluation**。

### 4.3 为什么 WebEyes 跟已有 benchmark 不一样

论文里把它和 RefCOCO、ReasonSeg、MMSearch、BrowseComp-VL 做了对比。核心差异可以概括成一句话：

**已有任务要么有 grounding 但没 web evidence，要么有 search 但没有 object-level grounded verification。**

WebEyes 则同时具备：

- common knowledge
- web knowledge
- grounding
- segmentation
- QA
- fine-grained evaluation

这意味着它评的不是纯搜索能力，也不是纯 perception 能力，而是两者的耦合质量。

### 4.4 WebEyes 的构建流程

![WebEyes construction pipeline](assets/arxiv_2605_from_web_to_pixels_bringing_agentic_search_into_visual_perception/data_gen.png)

作者的构建 pipeline 很系统，大致分四步：

1. **多实例图像收集**
   - 从 web image search、新闻页面、社交媒体等收图
   - 偏好多前景实例、易混淆、存在 distractor 的图像

2. **对象标注与视觉解析**
   - 人工标 foreground instance
   - 用 SAM3 精修 mask
   - 记录 mask、box、object name、category
   - 生成 visual feature text

3. **链式证据检索**
   - 对每个对象做三轮 chained search
   - 先解析成可检索实体
   - 再在标注前六个月时间窗内抓取公共证据
   - 构建支持 multi-hop 的 evidence paths

4. **知识型问题生成**
   - 隐去实体名和直接视觉属性
   - 保留能推导出身份的非视觉事实
   - 构成 single-hop / multi-hop query

### 4.5 质量控制

作者的质控也很严格，专门过滤三类 shortcut：

1. **Closed-book shortcuts**
2. **Vision-only shortcuts**
3. **Text leakage or non-uniqueness**

自动过滤直接砍掉了 **38.2%** 候选；
过了自动过滤后，人工审核又砍掉了 **49.2%**。

这说明作者很在意一个问题：

**不能让模型绕开“先搜证据再做实例绑定”这条主路径。**

## 5. Pixel-Searcher 方法

![Pixel-Searcher overview](assets/arxiv_2605_from_web_to_pixels_bringing_agentic_search_into_visual_perception/method.png)

Pixel-Searcher 是一个 reference workflow，不是那种大规模 end-to-end 训练出来的单体模型。它更像是一个 agentic pipeline。

### 5.1 两阶段结构

Pixel-Searcher 分成两个阶段：

1. **Agentic Search & Target Resolution**
2. **Agentic Grounding & Tool Use**

第一阶段解决：

- query 到底在说谁
- 需要搜哪些事实
- 怎样把搜到的事实拼成可视觉验证的 target hypothesis

第二阶段解决：

- 这个解析出来的目标在图里是哪一个实例
- 怎么输出 box / mask / answer

### 5.2 结构化 target hypothesis

作者把 hidden target 解析成：

`h = {e, c, K}`

其中：

- `e`: resolved entity name
- `c`: coarse visual category
- `K`: image-checkable cues

这个设计很关键。因为它不是把搜索结果原封不动丢给 grounding 模块，而是把搜索得到的长证据压缩成一个更适合视觉绑定的中间表示。

你可以把它理解成：

- query 是模糊的
- web evidence 是冗长的
- `h` 是可以直接喂给视觉 binding 的“中间语义骨架”

### 5.3 Agentic Search and Target Resolution

这一阶段的逻辑是一个 adaptive search-reason loop：

- `Search`: 检索外部证据
- `Reason`: 连接证据、判断是否足够
- `Resolve`: 输出当前 target hypothesis

作者强调这不是固定链路：

- 简单 query 可能一轮搜索就够
- 难 query 可能需要 multi-hop evidence chaining

最终输出不是自由文本答案，而是适合 grounding 的 hypothesis `h`。

作者还会检查：

- 当前解析出来的 entity 是否只是中间 clue
- 是否过于泛化
- 是否与视觉上下文冲突

所以它更像一个“面向视觉绑定的实体解析器”。

### 5.4 Agentic Grounding and Tool Use

第二阶段就是把 `h` 绑定到图像实例上。

作者把它写成：

`b* = A_bind(I, h)`

也就是：

- 利用 resolved target hypothesis
- 调 grounding tools 生成候选 region
- 再做 evidence verification
- 选出与图像和外部证据都最一致的 region

然后针对不同任务：

1. **SearchGround**
   - 直接输出 `b*`

2. **SearchSeg**
   - 先找到 `b*`
   - 再把 `b*` 喂给 `SAM3`
   - 让 segmentation tool 负责边界细化

3. **SearchVQA**
   - 给定 highlighted region
   - 把每个 answer option 解析成 entity-level summary
   - 选出和该 region 最匹配的选项

### 5.5 这套方法最值得注意的地方

我觉得 Pixel-Searcher 真正有价值的点不在“它搜了网页”，而在于它明确把问题分成了：

- **identity resolution**
- **instance verification**

这比很多“把搜索结果拼进 prompt 里直接问 VLM”要清楚得多。

它其实是在说：

> search 不是为了直接生成答案，而是为了把目标身份解出来；视觉模块不是泛泛地找相似区域，而是要验证“这个实例是不是刚刚解析出的那个对象”。

## 6. 实验设置

### 6.1 评测方式

所有方法：

- 用同样的 WebEyes 输入和 split
- 不做 task-specific fine-tuning

对比方式：

- 直接 baseline：image + query 直接出 box
- segmentation：box 后接 SAM3 出 mask
- SearchVQA：image + target box + options
- Pixel-Searcher：在 grounding 前插入 hidden-entity search

### 6.2 指标

- SearchGround: `IoU`, `Recall@0.5`
- SearchSeg: `gIoU`, `cIoU`
- SearchVQA: exact-match accuracy

### 6.3 使用模型

基线里既有 closed-source，也有 open-source。

比较显眼的几点：

- open-source general baseline 里重点是 `Qwen3-VL-8B`
- open-source specialized baseline 包括 `Ground-R1`, `UniVG-R1`, `LISA`, `Seg-R1` 等
- closed-source 有 `Doubao`, `GPT-4.1`, `GPT-5.4`, `Gemini-3.1`

作者特别提到，他们用 `Qwen3-VL-8B-Instruct` 而不是 Qwen 3.5，是因为 3.5 在 preliminary grounding 里 instruction following 更弱，经常连合法 box 都输出不出来。

## 7. 主要结果

### 7.1 Search-based Grounding

论文最核心的主结果之一是 SearchGround。

下面先摘 overall 指标：

| Method | Overall IoU | Overall R@0.5 |
| --- | ---: | ---: |
| OneThinker-8B | 32.78 | 38.70 |
| Qwen3-VL-8B | 26.81 | 32.61 |
| Ground-R1 | 26.68 | 29.67 |
| Pixel-Searcher | **34.17** | **41.30** |
| Doubao-Seed-2.0-Pro | 35.69 | 44.41 |

关键结论：

- Pixel-Searcher 是 **最强的 open-source 方法**
- 相比 `Qwen3-VL-8B`：
  - IoU: `26.81 -> 34.17`
  - R@0.5: `32.61 -> 41.30`

提升最明显的类别包括：

- Anime
- ICON
- Product

这很符合任务设定，因为这些类别往往实例多、外观近、必须依赖外部身份线索。

### 7.2 Search-based Segmentation

SearchSeg 的整体结果也很强：

| Method | Overall gIoU | Overall cIoU |
| --- | ---: | ---: |
| Seg-Zero-7B | 34.03 | 22.51 |
| OneThinker-8B | 35.60 | 24.46 |
| Qwen3-VL-8B | 35.78 | 25.94 |
| Pixel-Searcher | **39.17** | **32.41** |
| Doubao-Seed-2.0-Pro | 61.22 | 43.32 |

相比 `Qwen3-VL-8B`，Pixel-Searcher：

- gIoU: `35.78 -> 39.17`
- cIoU: `25.94 -> 32.41`

这说明：

- 搜索带来的收益不是只停留在 box 选择
- 更好的 entity resolution 会传递给后续 mask 细化

换句话说，分割结果差很多时候不是 mask tool 不行，而是前面的 target instance 先找错了。

### 7.3 Search-based VQA

SearchVQA 的 overall accuracy：

| Method | Overall Acc |
| --- | ---: |
| Qwen3-VL-8B | 36.34 |
| InternVL-3.5-8B | 36.02 |
| OneThinker-8B | 28.26 |
| Pixel-Searcher | **42.24** |
| Gemini-3.1-Pro | 63.82 |

Pixel-Searcher 对 open-source 的优势依然明显：

- `36.34 -> 42.24`

而且在以下类别上提升更明显：

- ICON
- Product
- Anime

这里说明 search-to-pixel 不只对定位类任务有用，对 **grounded region semantics verification** 也有帮助。

## 8. 这些结果说明了什么

论文里一个很重要的观察是：

**WebEyes 真正难的，不是“分割边界够不够细”，而是“能不能先把隐藏身份解析对，再把它绑定到对的实例上”。**

从结果上也能看出来：

- 直接感知型模型经常能找到“视觉上合理”的对象
- 但那个对象未必和外部证据解析出来的身份一致

这和普通 referring expression 很不一样。普通 referring 里视觉属性通常足够区分对象；WebEyes 里往往不是。

## 9. Ablation 与失败分析

### 9.1 Ablation

作者做了 component ablation：

| Variant | IoU | R@0.5 | gIoU | cIoU |
| --- | ---: | ---: | ---: | ---: |
| Full | 34.17 | 41.30 | 39.17 | 32.41 |
| w/o contradiction | 31.34 | 37.58 | 36.51 | 28.41 |
| w/o direct bonus | 30.80 | 36.96 | 35.67 | 27.92 |
| support only | 29.84 | 35.40 | 34.64 | 27.68 |
| w/o ref. match | 29.00 | 34.01 | 33.05 | 26.47 |
| direct only | 22.28 | 27.48 | 26.49 | 24.48 |
| w/o direct cand. | 20.14 | 19.72 | 20.14 | 15.71 |

这里最值得注意的是两点：

1. **去掉 direct candidates 掉得最狠**
   - IoU 从 `34.17` 掉到 `20.14`
   - R@0.5 从 `41.30` 掉到 `19.72`

2. **只靠 direct grounding 也远不够**
   - `direct only` 也明显弱于 full system

这意味着：

- 只靠视觉直接定位不够
- 只靠证据支持也不够
- 必须把 direct grounding、reference matching、contradiction checking 这些环节组合起来

### 9.2 失败分析

论文里给出的失败统计非常有启发性：

- 总共有 389 个 segmentation failure
- 其中：
  - 304 个是 **search/entity error**
  - 75 个是 **entity-correct region error**
  - 10 个是 **box-to-mask transfer error**

也就是说，大约四分之三的错误都发生在：

- 搜索规划
- 证据整合
- 实体消歧
- 视觉实例验证

而不是发生在最后一步 mask refinement。

这直接支持了论文的核心论点：

> 在这种 open-world grounded perception 里，真正的瓶颈不是 segmentation tool，而是 search-to-identity-to-instance 这条链。

## 10. 这篇论文的亮点

### 10.1 任务定义很有辨识度

很多论文会把“需要外部知识”的视觉任务做成：

- 搜索增强 VQA
- 检索增强 caption
- 浏览增强回答

这篇 paper 不一样，它把问题钉在：

**“外部知识最终必须落到像素级或区域级输出上”**

这个定义很清楚，也让 benchmark 更有新意。

### 10.2 benchmark 设计比单纯堆数据更用心

WebEyes 的价值不在于规模巨大，而在于它把以下东西串起来了：

- object annotation
- evidence path
- hidden-entity query
- grounded verification

这让它特别适合做 error diagnosis。

### 10.3 把“search”变成 perception 前端，而不是回答后端

这是我个人最喜欢的一个点。

很多 agentic search 工作把 search 当作“生成答案之前的知识补充”；
这篇工作把 search 当作：

**视觉绑定之前的身份解析模块**

这在 conceptual framing 上非常清楚。

## 11. 局限性

这篇工作也有一些很明显的限制。

### 11.1 benchmark 规模还不大

- 120 images
- 473 instances

对新方向 benchmark 来说可以接受，但如果要支撑更强的泛化结论，还偏小。

### 11.2 Pixel-Searcher 更像 workflow，而不是统一训练好的强模型

它的优势在于可解释、可分解；
但劣势也明显：

- 工程链路更长
- 依赖 search 质量
- 依赖外部工具
- latency 和部署复杂度不一定低

### 11.3 search reliability 仍是硬约束

如果外部证据搜不到、搜偏了、或时间敏感信息过期，
整个后续 pipeline 都会被拖偏。

### 11.4 closed-source 仍然更强

虽然 Pixel-Searcher 是开源方法里最强的，但整体上离最好的 closed-source 结果还有差距，尤其在 segmentation 与 VQA 上依然明显。

## 12. 对我们最有启发的地方

### 12.1 可以把“先解析身份，再做视觉绑定”视为一个独立模块

这篇 paper 对 VLM 研究者最大的提醒之一是：

**不要把所有困难都扔给 end-to-end grounding。**

对于很多 open-world 任务，更合理的拆法是：

1. target identity resolution
2. visual instance verification
3. output rendering

这个 decomposition 很值得借鉴。

### 12.2 它把 agentic search 和细粒度视觉任务真正接上了

你前面一直在强调“细粒度”。这篇论文其实就是一个很典型的例子：

- search 不停留在文本答案层
- 最终要落到 box / mask / highlighted region answer 上

这就是很实打实的 **fine-grained VLM + agentic search**。

### 12.3 对未来工作，一个自然方向是把 search hypothesis 学成更结构化的 latent state

现在 `h = {e, c, K}` 还是比较人工设计的 structured hypothesis。

后续可以想：

- 用 learned latent schema 替代手工字段
- 显式建模 evidence uncertainty
- 把多个候选 entity 做概率式绑定
- 用 region-level verifier 学“身份一致性分数”

### 12.4 还可以往更强的 benchmark 方向扩展

WebEyes 之后很自然的扩展包括：

- 更长时间跨度的时序知识
- 更复杂的多图、多页面、多视角 grounding
- 视频版 search-to-pixel
- OCR-heavy / chart-heavy 文档视觉场景

## 13. 一句话总结

**这篇论文的核心不是“让模型多看网页”，而是提出了一个新问题：当目标身份藏在图外知识里时，视觉系统能不能先把对象是谁找出来，再把它精确落到图中的那个实例上。WebEyes 负责把这个问题测清楚，Pixel-Searcher 则给了一个相当干净的 agentic baseline。**
