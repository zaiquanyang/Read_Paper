# Very Efficient Listwise Multimodal Reranking for Long Documents

## 1. 论文基本信息
- arXiv ID: 2605.11864
- URL: [abs](https://arxiv.org/abs/2605.11864) / [pdf](https://arxiv.org/pdf/2605.11864.pdf)
- 主题归档: `vlm`
- 总结粒度: `detailed`
- 发表信息: ICML 2026
- 作者机构: Yiqun Sun, Pengfei Wei, Lawrence B. Hsieh / Magellan Technology Research Institute (MTRI), Japan
- 核心方法: **ZipRerank**
- 代码: [dukesun99/ZipRerank](https://github.com/dukesun99/ZipRerank)
- 关键词: `multimodal reranking`, `long-document retrieval`, `M-RAG`, `query-aware token pruning`, `single-token decoding`, `teacher distillation`

## 2. 这篇论文在解决什么问题

这篇工作的目标很明确：做一个**又快又准的 long-document multimodal listwise reranker**。

它面向的场景不是普通文本 reranking，而是：

- query 是文本
- candidate 是长文档里的多个 page
- 每个 page 以图像形式输入
- reranker 需要在 top-k page 里重新排序

这个 setting 很适合 MMDocIR、M-RAG、DocVQA 一类任务，但也天然很贵。作者把瓶颈拆成了两个：

1. **长视觉上下文带来的 prefill 开销**
   - 每个 page 都是 image
   - 每个 image 会展开成大量 visual tokens
   - listwise reranking 要把多个候选页一起喂进去，context 很快爆炸

2. **自回归解码带来的 decoding 开销**
   - 很多现有 multimodal reranker 会输出排序序列，甚至还会先生成 reasoning
   - 这意味着要做多步 sequential decoding
   - 即使有 KV cache，长上下文上的逐步解码仍然很慢

作者的核心观点是：

> 现有工作很多只在“怎么让 reranking 更准”上发力，但对 long multimodal context 的推理路径做得还不够工程化。真正的高效方案，需要同时减少输入 token 和输出 token。

所以 ZipRerank 不是只改一个点，而是做了一个比较完整的协同设计：

- 训练时让模型学会“用第一步 logits 直接做 listwise ranking”
- 推理时先做 query-aware visual token pruning
- 最后只解一个 token，就把整组候选的排序分数拿出来

## 3. 方法总览

![ZipRerank overview](assets/arxiv_2605_very_efficient_listwise_multimodal_reranking_for_long_documents/zipranker.png)

ZipRerank 的结构可以概括成一句话：

**训练上用两阶段迁移出 listwise multimodal reranking 能力，推理上用 query-aware token pruning + single-token scoring 把时延压下去。**

整个框架分成两部分：

1. **Training phase**
   - Stage 1: 在大规模 text reranking 数据上做 listwise pretraining
   - Stage 2: 在 document-image 数据上做 multimodal finetuning，并用强 VLM teacher 提供 soft ranking supervision

2. **Inference phase**
   - 用 query-image early interaction 做视觉 token 剪枝
   - 用 single-token decoding 替代传统多步排序生成

## 4. 训练设计

### 4.1 指令化 listwise formulation

作者把 reranking 写成 instruction-style generation：

- 输入: query + 多个候选 page image
- 每个候选 page 分配一个唯一单 token 标识符，比如 `A/B/C/...`
- 目标输出: 按相关性从高到低排列的这些标识符序列

这个 formulation 的好处是：

- 兼容现成 VLM backbone
- 可以统一 text-only pretraining 和 multimodal finetuning
- 后面可以自然过渡到“只用 first-step identifier logits 评分”

### 4.2 Stage 1: text-only reranking pretraining

Stage 1 用的是 RankZephyr 文本 reranking 数据，但作者没有直接当文本喂，而是把 passage **render 成 280x280 的 page-like image**。

这么做有两个目的：

1. 让模型先学会一般性的 listwise reranking 行为
2. 尽量让 Stage 1 的输入形态接近 Stage 2 的 document-image setting

Stage 1 的损失是：

- `L_LM`: 排序输出序列的 language modeling loss
- `L_ranknet`: 对 first-step candidate identifier logits 做 weighted RankNet

组合为：

`L_stage1 = L_LM + lambda_1 * L_ranknet`

其中：

- `lambda_1 = 10.0`
- RankNet 只看候选间的相对顺序
- 权重 `w_{i,j} = 1 / (r_i + r_j)`，更强调高位候选对

这一步的本质，是先把“看多个候选一起做排序”这个能力学出来。

### 4.3 Stage 2: multimodal finetuning with soft labels

Stage 2 切到真实 document page image 上训练，数据来自 MMDocIR training set。

问题在于：这类数据往往只有一个 ground-truth positive page，并没有完整排序监督。作者的做法是：

- 用更强的 teacher VLM（文中主要是 `GPT-5-mini`）给 candidate list 产出一个辅助 ranking
- 把这个 ranking 当作 soft supervision

Stage 2 的损失变成：

- `L_LM`
- `L_softrank`

即：

`L_stage2 = L_LM + lambda_2 * L_softrank`

其中 soft target distribution 不是 one-hot，而是按 teacher 排序位置指数衰减：

- `q_{pi(k)} ~ gamma^k`
- 文中 `gamma = 0.5`

这个设计有两个现实好处：

1. **允许 teacher 排名里存在不确定性**
   - 不强迫只有一个候选是“绝对正确”
   - 对多个 plausibly relevant page 更友好

2. **比硬 pairwise / hard label 更抗噪**
   - teacher 排名不一定完美
   - soft ranking 比直接照抄 top-1 更稳

### 4.4 为什么这种两阶段训练有意义

我觉得这篇 paper 一个比较聪明的点，是它没有一开始就用小规模 multimodal 数据硬训 listwise reranker，而是先把能力拆开：

- Stage 1 负责学通用排序结构
- Stage 2 负责适配视觉 grounding 与真实文档分布

从 ablation 也能看出来这个拆分是有效的：

- 去掉 first stage，性能小幅下降
- 去掉 second stage，性能下降更明显
- 去掉 soft-ranking loss，尤其 Recall@1 掉得很厉害

所以这不是“训练 recipe 很花”，而是确实每一段都在补不同短板。

## 5. 推理设计

### 5.1 Query-image early interaction pruning

这是它处理长上下文的核心。

作者不是在 full attention 之后再看哪些 visual token 重要，而是做了一个更轻量的 query-aware 近似：

1. 先只跑 prompt prefix，取出 query token 的 hidden states `H_q`
2. 对每个 image 的每个 visual token `v_{i,j}`，计算它与所有 query token 的 cosine similarity
3. 对每个 visual token 取最大值：

`a_{i,j} = max_t cos(h_t, v_{i,j})`

4. 对每张图只保留 top `rho * N_i` 个 token

这里 `rho` 是 keep ratio。

这个分数的直觉非常直接：

- 如果一个 visual token 和 query 中某个 token 非常相似
- 那它更可能和当前 query 相关
- 就值得保留

作者在 appendix 里还补了两个理论解释：

1. **max-sim 是一种 attention-style pooling surrogate**
   - 它近似 query 对 visual token 的 soft pooling 分数
   - 和 log-sum-exp pooling 之间只差一个 `log N_q` 级别的界

2. **如果被剪掉的 token 总 attention mass 很小**
   - 那 attention 输出的扰动是 `O(epsilon)` 级别

这两个论证都不算特别强，但对 paper 的工程目标是够用的：它想说明这不是瞎剪，而是一个合理的 query-aware proxy。

### 5.2 Single-token decoding

第二个关键点是把传统 autoregressive ranking 改成**single-step scoring**。

做法是：

- 每个候选对应一个单 token identifier
- 跑完整个压缩后的 multimodal input
- 在“下一个 token 位置”直接读取这些 identifier 的 logits
- 对 logits 排序，得到整个 candidate list 的 reranking

也就是：

- 不再一步步生成 `A > C > B > ...`
- 只做一次 next-token prediction
- 用每个候选字母 token 的 logit 当 relevance score

这一步其实非常重要，因为它不是“减少几个 decode step”，而是把原本和 `k`、reasoning length 相关的输出长度，直接压成了 **1**。

如果把 ZipRerank 看成一个系统，它的速度来自两个正交收益：

1. `n` 变短了：视觉 token 被剪掉一部分
2. `u` 变成 1：不再多步解码

## 6. 论文里的理论和复杂度分析

作者在 appendix 给了一个简化的 decoder-only Transformer FLOPs 模型：

- prefill cost 约为 `O(n^2)` attention + `O(n)` FFN
- cached decoding 约为 `O(u * n)`

所以 baseline autoregressive multimodal listwise reranker 的开销大致是：

- 大 context 的 prefill
- 再加上 `beta*k + u_reason` 步的 decode

ZipRerank 则变成：

- early interaction scoring
- 压缩后 context 的 prefill
- 只做 1 步 decode

在两个 regime 下收益最明显：

1. **long-context regime**
   - visual token 远大于 text token
   - 剪枝主要降低 prefill 的平方项开销

2. **generation-heavy regime**
   - baseline 会输出长 ranking / reasoning
   - single-token scoring 直接省掉大部分 decode

这篇 paper 的理论部分更偏“合理化解释”，不是特别强的算法理论，但足够支撑它的工程设计。

## 7. 实验设置

### 7.1 数据和任务

- 训练:
  - Stage 1: RankZephyr
  - Stage 2: MMDocIR training set

- 测试:
  - MMDocIR page-level retrieval / reranking
  - ViDoRe English subset（看 OOD generalization）

MMDocIR 的 setting 是：

- 313 个长文档
- 10 个 domain
- 平均每个文档 65.1 页
- 1658 个 expert-curated query

评价时先用 first-stage retriever 取 top-20 candidate pages，再做 reranking。

### 7.2 First-stage retriever

作者用了两个不同的一阶段召回器：

- `DSE_wiki-ss`
- `ColQwen`

这让结论更稳一些，因为能看出 ZipRerank 不是只绑定在某一种 retriever 上。

### 7.3 Backbone 和训练细节

- base model: `Qwen3-VL-8B-Instruct`
- 训练时冻结 vision encoder，只更新 language model 参数
- 硬件: 单张 `NVIDIA H200`
- attention: `Flash Attention 2`
- optimizer: `8-bit AdamW`
- precision: `BF16`

训练超参：

| Stage | Dataset | Epochs | LR | Batch | Grad Accum | Effective Batch | Ranking Loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Stage 1 | RankZephyr | 3 | 3e-6 | 8 | 4 | 32 | Weighted RankNet |
| Stage 2 | MMDocIR | 1 | 3e-6 | 2 | 8 | 16 | Soft ranking |

额外关键设置：

- Stage 1 文本先渲染成 `280x280` image
- Stage 2 图片最长边不超过 `1024`
- Stage 2 里 `lambda_2 = 1.0`
- soft rank 的位置衰减 `gamma = 0.5`
- 每个 query 最多看 20 个候选

### 7.4 Prompt 形式

![Prompt template](assets/arxiv_2605_very_efficient_listwise_multimodal_reranking_for_long_documents/Prompt.drawio.png)

prompt 设计很朴素，但和 single-token listwise scoring 强绑定：

- 候选用 `A/B/C/...` 标号
- 输出序列也是这些 identifier 的排序
- 推理时只取第一个位置对这些 identifier 的 logits

## 8. 主要结果

### 8.1 Speed-accuracy trade-off

![Speed-accuracy trade-off](assets/arxiv_2605_very_efficient_listwise_multimodal_reranking_for_long_documents/speed_effectiveness_tradeoff_new.png)

这张图是整篇 paper 最想传达的信息：

- ZipRerank 在 MMDocIR 上能做到接近甚至超过 MM-R5 的效果
- 但延迟大约低一个数量级
- 和 `GPT-5-mini` 的效果差距被明显缩小，但推理成本低得多

### 8.2 MMDocIR 主结果

下面我只摘最关键的 macro/micro 和时间。

#### DSE first-stage

| Method | R@1 Macro | R@1 Micro | R@3 Macro | R@3 Micro | R@5 Macro | R@5 Micro | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DSE first-stage | 46.0 | 45.5 | 69.5 | 70.2 | 77.1 | 78.2 | -- |
| MM-R5 | 66.1 | 65.1 | 79.1 | 79.0 | 83.8 | 83.9 | 3.82 |
| ZipRerank | 64.2 | 63.3 | 84.8 | 84.5 | 89.0 | 89.4 | 0.36 |
| ZipRerank-50% | 63.3 | 62.4 | 83.4 | 83.4 | 88.2 | 88.6 | 0.30 |

这里很值得注意的一点是：

- **ZipRerank 在 Recall@1 略低于 MM-R5**
- 但 **Recall@3 / Recall@5 明显高于 MM-R5**
- 且 cached LLM latency 从 `3.82s` 直接降到 `0.36s`

这说明它牺牲了一点 top-1 极致命中率，换来更稳的 top-k 排序质量和大幅时延收益。

#### ColQwen first-stage

| Method | R@1 Macro | R@1 Micro | R@3 Macro | R@3 Micro | R@5 Macro | R@5 Micro | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ColQwen first-stage | 60.4 | 59.9 | 79.0 | 78.2 | 84.1 | 83.5 | -- |
| MM-R5 | 68.5 | 67.4 | 83.0 | 82.1 | 86.6 | 86.1 | 3.74 |
| ZipRerank | 66.1 | 65.1 | 84.9 | 84.4 | 89.3 | 89.1 | 0.36 |
| ZipRerank-50% | 63.9 | 63.0 | 83.8 | 83.1 | 88.3 | 88.1 | 0.31 |

趋势和 DSE 基本一致：

- R@1 不一定是最强
- 但在更实用的 top-k recall 上明显更优
- 时间优势依旧非常大

### 8.3 为什么 paper 里强调 R@3 / R@5

这其实和文档检索任务很契合。

在 M-RAG 或 DocVQA 管线里，reranker 的作用通常不是“只保一个 page”，而是：

- 尽量把 evidence page 往前挪
- 让后续生成或阅读模块在一个小 top-k 集合里就能找到关键信息

所以 ZipRerank 虽然不是最强 top-1 selector，但它是一个很强的 **high-recall efficient reranker**。

## 9. Ablation 和附加分析

### 9.1 Ablation：每个模块都不是摆设

| Variant | R@1 Macro | R@1 Micro | R@3 Macro | R@3 Micro | R@5 Macro | R@5 Micro | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ZipRerank | 64.2 | 63.3 | 84.8 | 84.5 | 89.0 | 89.4 | 0.36 |
| w/o first stage | 63.4 | 62.6 | 83.8 | 83.8 | 88.3 | 88.7 | 0.36 |
| w/o second stage | 61.8 | 60.6 | 78.8 | 78.6 | 84.2 | 84.6 | 0.36 |
| w/o single-logit decoding | 64.2 | 63.3 | 83.7 | 83.2 | 88.0 | 88.0 | 2.19 |
| w/o soft-ranking loss | 56.8 | 55.5 | 79.2 | 79.7 | 85.4 | 85.8 | 0.36 |

我觉得这里最有信息量的有三点：

1. **single-logit decoding 几乎不掉性能，但时间从 0.36s 变 2.19s**
   - 这说明作者的训练方式真的把“用 first-step logit 做排序”对齐出来了

2. **soft-ranking loss 很关键**
   - 去掉后 R@1 掉得很厉害
   - 说明 teacher 提供的 graded supervision 比硬 pairwise 信号更合适

3. **second stage 比 first stage 更关键**
   - 说明 vision-centric finetuning 对 multimodal reranking 依然必不可少

### 9.2 Query-aware pruning 确实比 random pruning 好

| Keep Ratio | Random R@1/3/5 | T2I R@1/3/5 |
| --- | --- | --- |
| 0.1 | 32.2 / 61.2 / 73.7 | **40.1 / 66.2 / 78.0** |
| 0.3 | 48.2 / 72.5 / 81.4 | **57.4 / 78.4 / 84.6** |
| 0.5 | 54.7 / 77.0 / 84.7 | **61.1 / 80.8 / 86.6** |
| 0.7 | 58.8 / 79.9 / 85.9 | **61.5 / 82.4 / 87.1** |
| 0.9 | **62.5** / 82.2 / 87.3 | 62.1 / **82.4** / **87.4** |

这张表很重要，因为它说明：

- ZipRerank 的 pruning 不是“只要少 token 就行”
- query-aware 的 T2I pruning 在中低 keep ratio 下明显优于 random pruning

尤其 `rho = 0.3` 时提升很明显，这说明它确实保住了更相关的视觉信息。

### 9.3 排序质量分析：ZipRerank 更少 catastrophic miss

| Method | P@1 | nDCG@5 | Mean Rank | Fail% | Near Miss | Cat. Miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DSE | 50.6 | 65.6 | 3.60 | 49.4 | 52.0 | 36.3 |
| MM-R5 | **73.1** | 77.9 | 2.84 | **26.9** | 41.3 | 45.5 |
| ZipRerank | 70.0 | **79.4** | **2.62** | 30.0 | **57.8** | **30.7** |

这张表解释了为什么主结果里会出现：

- MM-R5 更强的 R@1
- ZipRerank 更强的 R@3 / R@5

因为：

- MM-R5 更擅长把正确页顶到第一
- 但一旦失败，更容易掉到很后面
- ZipRerank 虽然 top-1 稍逊，但更常把正确页排在 2-3 名附近，严重失误更少

如果你的下游系统是看 top-k candidates，而不是只看第 1 个，这种行为其实更实用。

## 10. 泛化和鲁棒性

### 10.1 OOD generalization 到 ViDoRe

| Method | DSE | ColQwen |
| --- | ---: | ---: |
| DSE first-stage | 41.0 | 50.5 |
| UniME | 42.6 | 51.4 |
| LamRA (Listwise) | 48.0 | 54.9 |
| MM-R5 | 49.0 | 55.8 |
| ZipRerank | **53.4** | **59.9** |
| ZipRerank-50% | 52.2 | 58.5 |
| LamRA (Pointwise) | 56.1 | 60.0 |

结论很清楚：

- 在 listwise reranker 里，ZipRerank 是最强的
- 即使砍到 50% visual tokens，也依然很稳
- pointwise LamRA 还略高一点，但它需要对候选逐个打分，部署代价不一样

所以如果比较的是“**高效 listwise multimodal reranker**”，ZipRerank 的 OOD 表现是很有说服力的。

### 10.2 对 teacher 强弱不太敏感

| Method | DSE R@1 Ma/Mi | DSE R@3 Ma/Mi | DSE R@5 Ma/Mi | ColQwen R@1 Ma/Mi | ColQwen R@3 Ma/Mi | ColQwen R@5 Ma/Mi |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-5-mini | 70.0 / 69.2 | 88.0 / 88.3 | 90.9 / 91.3 | 70.9 / 70.1 | 88.6 / 87.8 | 91.9 / 91.4 |
| ZipRerank_mini | 64.2 / 63.3 | 84.8 / 84.5 | 89.0 / 89.4 | 66.1 / 65.1 | 84.9 / 84.4 | 89.3 / 89.1 |
| GPT-5-nano | 59.8 / 59.0 | 79.6 / 79.1 | 84.5 / 84.7 | 63.6 / 62.5 | 82.0 / 81.0 | 87.0 / 86.0 |
| ZipRerank_nano | 63.8 / 63.6 | 81.6 / 82.2 | 86.6 / 87.1 | 65.2 / 64.8 | 83.6 / 83.0 | 88.5 / 87.8 |

这个实验挺有意思：

- 用弱 teacher（GPT-5-nano）训出来的 ZipRerank，仍然能超过 teacher 自己
- 说明它不是简单模仿 teacher
- 两阶段训练和 soft-ranking objective 本身就能带来额外收益

## 11. 效率分析

| Method | Vision ms | Filter ms | LLM ms | Total ms | TFLOPs/query | Cached QPS | Peak GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ZipRerank | 181.2 | -- | 357.4 | 538.5 | 179.7 | 2.80 | 21.71 GB |
| ZipRerank-50% | 180.2 | 4.5 | 269.4 | 454.1 | 84.9 | 3.65 | 20.05 GB |
| MM-R5 | 873.2 | -- | 3233.8 | 4107.0 | 263.2 | 0.31 | 23.04 GB |
| LamRA | 352.7 | -- | 529.3 | 881.9 | 368.2 | 1.89 | 28.31 GB |
| DocReRank | 401.1 | -- | 737.8 | 1140.8 | 54.8 | 1.35 | 4.54 GB |

这张表基本可以当 ZipRerank 的“价值证明”：

- 对比 MM-R5：
  - 总时延 `4107.0 -> 538.5 ms`
  - 约 **7.6x end-to-end speedup**
- 50% pruning 版进一步到 `454.1 ms`
- query-aware filtering 自己只花 `4.5 ms`

所以作者的结论是合理的：

- 最大收益来自 **single-pass listwise scoring**
- pruning 再额外补一层 compute / memory 优化

## 12. 这篇论文最值得记住的点

### 12.1 它不是在做更强 reasoning，而是在做更强 reranking system design

和很多强调 CoT、显式 reasoning 的 multimodal reranker 不同，ZipRerank 的重点是：

- 不执着于生成解释
- 而是把 listwise reranking 本身做成一个更合适的 scoring problem

这个取向很工程，也很实用。

### 12.2 “只解一个 token”这个思路很值钱

这篇 paper 最有启发性的地方，我觉得不是 token pruning，而是：

**把 ranking 目标训练到 first-step identifier logits 上，然后推理时直接单步排序。**

这其实是一个很强的范式：

- 当输出本质上是“候选打分/比较”而不是自由生成时
- 不一定要真的把完整答案生成出来
- 可以把生成模型重新压回 scoring model

这个想法对很多 reranking / selection / tool routing / page routing 任务都可能有迁移价值。

### 12.3 query-aware pruning 选了一个很便宜但够用的 proxy

它没有走“先算 full attention 再决定保留谁”的路线，而是用：

- query hidden states
- visual embeddings
- max cosine similarity

做一个便宜的 early interaction proxy。

这和你之前关心的“attention-free 重要性估计”其实很接近：  
**不直接依赖完整 attention score，而是找一个前置、轻量、可并行、近似 query relevance 的指标。**

## 13. 局限性

作者自己承认的限制主要有三类：

1. **teacher bias**
   - Stage 2 依赖 teacher ranking
   - teacher 的错误和偏差会传递进来

2. **aggressive pruning 的风险**
   - 对细粒度证据页、小文本、密集表格、视觉上高度相似的页面
   - 过强剪枝可能丢掉关键信息

3. **评测范围还不够广**
   - 主要集中在 document-image reranking benchmark
   - 更多语言、更多 domain、更多 retrieval setting 还有待验证

我再补一个 paper 没有特别展开、但实际很重要的点：

4. **它更擅长“高效 top-k reranking”，不一定是最强 top-1 verifier**
   - 如果你的系统只允许拿第一个 page，而且极度看重 rank-1 precision
   - MM-R5 这类显式 reasoning reranker 可能仍有优势

## 14. 对我们这边最有启发的后续想法

### 14.1 把 single-token scoring 思路迁到别的 multimodal selection 问题

这套思路不只适用于 document reranking，也可以考虑用于：

- GUI grounding 候选框排序
- 多图候选检索
- VLM tool selection
- 多段 evidence page / region selection

关键前提是：

- 候选是离散集合
- 输出本质是排序或打分
- 不需要自由生成长文本

### 14.2 把 query-aware pruning 做得更细粒度

ZipRerank 目前的 pruning 分数是：

- query token 到 visual token 的 max cosine similarity

后续可以考虑：

- region-aware 或 layout-aware pruning
- 结合 OCR token / detected region token
- 加入 page position / local neighborhood consistency
- 用 learned lightweight scorer 替代 hand-crafted max-sim

### 14.3 用更强但更便宜的 teacher supervision 方案

它现在的 teacher 是 GPT-5 family 排序。后面可探索：

- 多 teacher ensemble soft labels
- teacher uncertainty-aware weighting
- teacher-free self-distillation / bootstrapping
- 只在困难 query 上调用强 teacher

### 14.4 如果你关心“细粒度视觉证据”，这篇方法还可以再往 region-level 走

当前 ZipRerank 是 page-level reranking。  
对很多真正的 VLM reasoning 任务来说，page 还太粗。

可以考虑的自然延伸是：

- page-level coarse rerank
- region / block / OCR span-level fine rerank
- 最后再交给 generator

这会把 ZipRerank 变成更完整的 multimodal retrieval front-end。

## 15. 一句话评价

**ZipRerank 不是最“炫”的 multimodal reasoning paper，但它是那种非常清楚自己在优化什么、并且把训练目标和推理路径都对齐起来的系统型工作。**

如果你关心的是：

- VLM 在 long-document retrieval / M-RAG 里的部署可行性
- attention-free 或 lightweight 的 query-aware token selection
- 把生成式 reranker 压缩成 single-step scorer

那这篇 paper 很值得反复看。
