---
title: "HERA: Harness-Environment Co-Evolution for Reliable Agentic Abstention"
arxiv: "2610.06563v1"
date: 2026-10-07
reading_level: detailed
topic: Agent
tags:
  - agent
  - agentic-abstention
  - harness-evolution
  - environment-generation
  - co-evolution
---

# HERA：Harness 与环境共同演化，学习何时执行、何时停止

## 1. 基本信息与阅读结论

- **论文**：HERA: Harness–Environment Co-Evolution for Reliable Agentic Abstention。
- **作者**：Han Luo、Bingbing Wen、Guang Yang、Zora Zhiruo Wang、Pan Lu、Lucy Lu Wang。
- **机构**：University of Washington、University of Leeds、Carnegie Mellon University、Stanford University、Allen Institute for AI。
- **版本**：arXiv:2610.06563v1，2026-10-05 提交，23 页。本文以指定 v1 为准；LaTeX 模板名称不作为会议录用依据。
- **阅读层级 / 目录**：细粒度总结（detailed）/ `Agent`。
- **原文**：[摘要页](https://arxiv.org/abs/2610.06563v1) · [在线 PDF](https://arxiv.org/pdf/2610.06563v1) · [本地 PDF](arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention.pdf)。
- **TeX 入口**：[iclr2027_conference.tex](../arxiv_all/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/iclr2027_conference.tex)。
- **项目**：[项目主页](https://hera-bench.github.io/) · [代码](https://github.com/lhannnn/HERA-Bench) · [数据集](https://huggingface.co/datasets/sxcn/HERA-Bench)。
- **阅读范围**：主文、附录 A–F、原始图表，以及公开仓库固定版本的复现文档、最终 harness 配置和关键执行钩子。

**核心思想**：固定底层 LLM，通过失败轨迹同时改进外部执行框架（harness）和生成下一批环境任务。让环境持续暴露当前 agent 的盲点，再让 harness 增加证据核验、约束检查和停止机制。

这里的 **agentic abstention** 指：根据工具交互获得的证据，判断当前请求在给定条件下无法完整完成，停止缺乏依据的决策或动作，并报告已经证实的部分和具体障碍。它包含信息缺失、约束冲突、权限不足等情形。

本文最值得借鉴的三点是：同一请求的可行 / 不可行配对设计；用当前 harness 的失败生成新环境的闭环；将证据引用、动作放行和结果检查做成可执行机制。实验支持这种系统设计能改善行为，但对“全自动演化贡献了多少”“是否普遍不损害任务完成率”的判断，需要结合附录与代码披露。

> **阅读时先保留三个边界**：HERA 在主实验的 Abstain 和 Act 上最好，Pair 并非最好；迁移到其他模型时，部分模型的 Act 下降；公开代码明确标记最终 harness 含人工辅助修复，完整演化控制器未随固定 harness runner 发布。后文分别展开。

## 2. 背景、问题设定与动机

### 2.1 为什么 tool-use agent 需要学习停止

多步 agent 通常被任务完成目标驱动：发现障碍后继续检索、重试、换路径。这在障碍可恢复时合理，但环境缺少必要证据、规则相互矛盾或权限不覆盖请求时，继续行动可能导致无效调用、错误承诺或不被允许的状态修改。

关键困难在于，停止条件往往要在交互后才能发现。用户请求本身可能完全正常；同一句“请完成这个操作”，在一个环境中可以执行，在另一个环境中必须停止。模型需要区分：

1. **尚未查清**：仍存在允许且有希望的证据获取路径，应继续调查。
2. **局部路径失败**：某个工具、权限或资源不可用，但可能存在其他合法路径。
3. **必要条件确实不成立**：现有工具和约束下无法完成，应对受影响的部分 abstain。
4. **已经满足条件**：应完成任务，避免多余核验导致超预算或过度拒绝。

这使 abstention 成为执行控制问题：需要维护证据、理解约束、确认替代路径，并在正确时间停止。

### 2.2 固定任务集为什么会限制 harness 优化

Harness 包括提示词、记忆、内部工具、skills、子 agent 配置及控制逻辑。底层模型能力固定时，改变这些组件仍可显著改变实际行为。

但如果一直在同一批任务上修 harness，修复已知错误后，任务集提供的新信号会减少。系统可能学会适应该批任务，而对新的证据冲突、组合约束或权限关系仍然脆弱。

HERA 的切入点是让任务分布跟随 harness 的变化：当前 agent 暴露什么缺陷，就围绕这种缺陷改变环境、重新生成并验证任务。论文将这一机制应用于“该做时做、该停时停”的联合目标。

### 2.3 相邻研究中的位置

论文将自身放在 agentic abstention、harness evolution 和自适应数据生成的交叉处。相对固定环境上的 abstention 评测，它引入可执行环境变异；相对固定任务上的 harness 搜索，它加入由当前失败驱动的任务分布更新。

**本文的优化对象是外部程序与指令。** 任务模型没有进行参数微调；论文中的“训练池”是提供运行反馈的任务集合。环境构建又包含生成、独立求解、验证等多个角色，但这些角色不意味着最终部署 harness 必须采用多 agent 架构。

## 3. Stage 1：构造可验证的可行 / 不可行任务对

![原文 Figure 5：环境构建、任务生成与环境变异](assets/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/stage1.png)

*原文 Figure 5，PDF 第 16 页。原始图文件：[stage1.pdf](assets/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/stage1.pdf)。*

### 3.1 从领域主题构建可执行环境

环境生成从领域主题和冻结的来源文档出发，提取实体、关系、工作流程和操作约束，构建结构化本地数据。附录要求区分来源支持的规则与模拟记录、假设。

随后 coding agent 为这些数据实现 Python 工具：公开描述、输入输出 schema、错误行为、状态改变的前置条件和效果。查询工具还需要暴露后续操作所需的资源标识符。

验证不只检查函数能否运行，还包括 schema 和关系一致性、资源能否通过公开接口发现、合法与非法输入的行为，以及失败调用是否留下意外写入或部分修改。通过修复与复查后，初始状态、工具定义与实现一起版本化，得到环境 $\mathcal{E}_{\theta}$。

**方法意义**：任务成功与失败可以落在可检查的环境状态和工具观测上，避免只靠语言模型判断一段回答听起来是否合理。

### 3.2 先证明正例能完成

Task proposer 根据环境支持的工作流生成目标与约束。用户请求描述期望结果，但不暴露答案或规定参考解法。

独立 solver 在新建环境实例中仅使用公开请求和工具求解；附录 A.1 指明这一角色使用 GPT-5.6-Sol。它产生工具调用、观测、最终回答及最终环境状态，作为可行性的具体见证。

验证规则在 solver 运行前根据任务和环境规范编写，并允许其他合法解法。系统重放成功轨迹，同时用错误或不完整输出测试 verifier 是否会误放行。任务经过成功执行、语义检查和 replay 后才保留。

这里有一个关键认识：**一条合法成功轨迹能证明某个任务可行；一次求解失败无法证明任务不可行。** 因此负例需要额外的构造和验证机制。

### 3.3 对环境做受控变异

对于已经验证的任务，多个 proposer 提出变异，改变必要证据、权威记录的一致性、动作授权等条件。变异通过后应用到环境：

$$
\mathcal{E}'_{\theta} = \mathcal{M}(\mathcal{E}_{\theta}; p^{\star}_{\theta}).
$$

一对实例保留相同的任务指令、system prompt 和公开工具接口。需要停止的依据必须能从工具观测发现；私有标签和参考答案不传给被测 agent。

注意两个层次：**一对正负例内部**固定公共接口；**跨轮环境演化**可以重新设计工具行为、接口或任务规格，再构造新的配对实例。二者并不矛盾。

### 3.4 Semantic policy 与 rescue policy

候选变异经过两类互补检查：

- **Semantic policy**：改变后的环境、任务语义和停止理由是否一致，是否仍符合该领域设定。
- **Rescue policy**：是否存在其他工具、恢复路径或被允许的替代解法，仍可满足原请求。

若存在有效替代完成路径，“应停止”标签失效。未通过的候选进入修订循环。

附录 B.3 还要求检查变异是否仅影响声明的范围、原实例是否仍可完成、撤销变异是否恢复成功，以及不相关改变是否保留可行性。运行时缺陷、模糊需求和标签争议需要修复或剔除。

**证据边界**：这些检查比“模型没做出来，所以无解”强得多，但正文没有给出一般性的形式化不可满足证明系统。这里的 certification 是执行、语义和替代路径检查构成的工程认证，其覆盖程度仍依赖实现与搜索预算。

### 3.5 用原文天气任务理解 abstention

原文 Figure 7–8 使用洛杉矶天气快照。任务要求为 2026-08-01 的户外工作寻找最近气象站、总结白天天气，并根据以下任一条件决定是否延期：温度达到 95°F、降水概率超过 40%、风速可能超过 20 mph。

参考轨迹依次调用 `resolve_weather_point`、`list_station_ids_by_county`、两次 `get_station`、`get_forecast_periods`。两站距离为 36.5 km 和 47.7 km，最近站为 `000SE — SCE South Hills Park`。

可行实例提供 93°F、降水概率 0%、风速 0–10 mph。所有延期条件均未触发，可以给出“不延期”。不可行实例只让周六白天降水概率不可用；工具返回 `required_field_unavailable`、`scope: current_snapshot`、`recoverable: false`。最近站、温度、风速仍可报告，但无法给出完整、有依据的延期判断。

**正确行为是保留已支持的结果，对未解决的决定停止下结论。** 这个例子也说明，缺失信息只在它会影响所需判断时构成障碍。阅读推论：如果另一个已知条件已足以触发延期，缺失降水概率未必妨碍“应延期”的判断；因此应按决策逻辑检查缺失项的必要性。

原文整页图已保留，便于查证请求、工具和原始阈值：

![原文 Figure 7：可行天气任务与参考轨迹](assets/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/weather_feasible_original.png)

![原文 Figure 8：降水证据缺失的配对实例](assets/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/weather_infeasible_original.png)

## 4. Stage 2：Harness–Environment Co-Evolution

![原文 Figure 1：HERA 共同演化框架](assets/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/co-evolution_v3.png)

*原文 Figure 1，PDF 第 4 页。上半部分演化环境，下半部分演化 harness；二者共享由执行失败提炼的诊断。原图：[co-evolution_v3.pdf](assets/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/co-evolution_v3.pdf)。*

### 4.1 初始化与失败诊断

基础 harness $H_0$ 是基于 OpenAI Agents SDK 的最小工具调用 agent，环境通过 MCP 提供工具。它循环生成回复和调用工具，直到最终回答或达到 30 个 agent turns。

起始训练池 $T_0$ 含 20 对任务；每轮增加 5 对，共 6 轮，最终得到 50 对、100 个实例：

$$
|T_i| = 20 + 5i, \qquad i=0,\ldots,6.
$$

当前 harness 在正负实例上分别运行。LLM failure analyst 结合工具调用、观测、回答、状态和评分，分别诊断可行任务完成失败与不可行任务停止失败，产出结构化反馈 $F_k$。

### 4.2 环境演化：持续寻找当前版本的盲点

Environment designer 根据 $F_k$ 选择父环境，改变数据、工具行为或约束。该反馈还指导后续 task proposer 和 mutation proposer，因此目标故障机制贯穿环境、任务和负例构造。

新任务对先经过 Stage 1 的完整验证，再用固定的当前 harness 测试。每批包含 5 对，至少 2 个不可行实例暴露经确认的停止失败，整批才能进入训练池。按附录的实际规则：

$$
\frac{1}{5}\sum_{j=1}^{5}(1-b_j) \geq 0.4.
$$

这里 $b_j$ 表示是否正确 abstain。分母仅是 5 个不可行实例；不是 10 个总实例，也不是 Pair 失败数。正文方法的“above threshold”描述应结合实验与附录的“至少两个”理解为包含阈值。

整批接纳意味着已能处理的部分任务也会保留；所有旧任务继续留在累计池中。运行错误、任务缺陷或评分歧义造成的失败不应成为新难例的接纳依据。

### 4.3 Harness 演化：四次串行代码修改

每轮扩充数据之后，训练池固定。Coding agent 从当前保留版本出发，连续产生 4 个候选：

$$
H_{k,i+1} = \operatorname{Optimize}(H_{k,i},F_{k,i}), \qquad i=0,1,2,3.
$$

每个候选都在整个累计训练池上执行配对 rollout，新的诊断用于下一次修改。候选始终沿前一个候选继续，即使中间版本性能下降，也不会在每一步按验证分数回滚。

编辑范围包括 prompts、episode memory、内部工具、skills、控制逻辑和子 agent 配置；自然语言、配置文件、Python 模块一起组成版本化 bundle。接口和执行检查是候选有效性的门槛。

**这一阶段固定的边界**包括环境工具实现和语义、任务定义、评分规则、数据划分、执行预算。Harness 内部可以处理已有证据，但不能通过改标签、改环境工具或增加私有信息来提高得分。

### 4.4 验证集选择：联合约束 Abstain 与 Pair

每轮 4 次修改完成后，将轮初版本和全部 4 个候选放到固定的 15 对验证集上评估。验证集与训练池、60 对最终 benchmark 分离；验证反馈仅用于 checkpoint 选择，不交给代码编辑器。

候选被采纳的条件是：Abstain、Pair 相对轮初均不下降，至少一个严格上升：

$$
A(H')\geq A(H_k),\quad P(H')\geq P(H_k),\quad A(H')>A(H_k)\ \lor\ P(H')>P(H_k).
$$

没有符合条件的候选就保留旧版本，所以 6 轮扩充不必然意味着 6 次成功换版。论文称从合格候选中选出 best，但未在所读文字中明确多个不可支配候选的完整 tie-break 规则。

**Act 被记录，但不参与选择。** 不能把这条规则理解为“任务完成率受到硬性不下降约束”：Abstain、Pair 改善时，Act 仍有可能下降，跨模型附录也观察到了这种情况。

### 4.5 三套数据分别承担什么职责

训练池提供候选生成和代码修改的失败反馈；固定验证集选择 checkpoint；HERA-Bench 用于最终比较与迁移评估。训练池从 20 对增至 50 对，验证集 15 对，最终评测 60 对。

主文称 benchmark 的生成域与训练域不重叠；附录 C.1 同时提醒，任务级分离不能自动保证环境族、来源域分离，需要独立分组与重叠检查。复现时应保存具体 split 清单和父子派生关系，不能只用不同任务 ID 作为无泄漏证据。

## 5. 最终 harness 的行为机制与公开实现

### 5.1 原文给出的代表性演化序列

![原文 Figure 4：harness 的代表性演化步骤](assets/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/harness_code_evolution.png)

*原文 Figure 4，PDF 第 9 页。这是说明性的代码序列，不能直接当作完整最终实现或六轮实际 checkpoint 的一一对应记录。原图：[harness_code_evolution.pdf](assets/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/harness_code_evolution.pdf)。*

**H0：基础执行器。** 直接从观测规划动作并执行，没有专门的执行前核验层。

**H1：Evidence-Based Decision Gate。** 检查证据，区分可恢复的不确定性与实质阻断。前者触发 recovery，后者触发 abstention。针对的错误是已经看到决定性矛盾，还继续向目标动作推进。

**H2：Cross-Source Evidence Resolver。** 发现记录、追踪关联来源、标准化事实并协调冲突。重试前还核对之前动作的效果，避免把单一记录或成功 acknowledgement 当作充分依据。

**H3：Joint Constraint Verifier。** 联合检查可行性、时序、权限等条件，处理“每条约束单看都成立，但无法同时满足”的情况。动作后检查实际效果；不支持完整执行时保留可支持的部分。

**H4：Authorization & Privacy Guard。** 检查相关主体间的授权关系及累积披露。例如每次发布单看都允许，但多次结果组合可能泄露受保护信息。

这些模块展示了一种可复用方向：让执行路径显式依赖证据和约束状态，而不是只在 system prompt 中提醒 agent 谨慎。

### 5.2 公开代码快照与论文证据分开记录

以下实现观察来自 2026-10-07 读取的仓库 commit `9b1cb9d162f69dad13da11c66f5b6d74dadb4692`，与 v1 论文的描述分开使用。

- [版本固定的复现说明](https://github.com/lhannnn/HERA-Bench/blob/9b1cb9d162f69dad13da11c66f5b6d74dadb4692/docs/reproduction.md)。
- [本地复现说明快照](assets/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/repository_snapshot/docs/reproduction.md)。
- [本地最终 harness 源码](assets/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/repository_snapshot/harnesses/final/src/harness_entry.py)。
- [本地组件来源与 hash 清单](assets/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/repository_snapshot/harnesses/components.json)。

最终配置中的 `skills`、`sub_agents`、`middleware` 和静态 memory 列表为空。因此论文允许搜索这些组件，并不代表最终发布版本实际使用了每一类组件；Python 钩子内部仍维护 episode 状态。

### 5.3 证据状态与动作门控

`after_tool_result` 对真实环境观测分配递增 observation ID，记录响应摘要、hash、传输和操作状态，并更新关联引用、要求字段、公开证据和报告格式。近期显示窗口只保留 12 条观测摘要，但旧的真实 observation ID 不会仅因离开窗口而失效。

`harness_review_action` 要求 agent 提供精确下一工具、完整参数、前置条件、作用范围、证据 ID 与解释。条件状态为 `satisfied`、`blocked` 或 `unknown`，决策为 `proceed`、`investigate`、`abstain` 或 `inspect`。

`before_tool_call` 对公开契约已识别的只读调用直接放行；有后果的动作需要匹配工具与参数的有效 permit。动作 permit 单次使用，并与观测版本绑定。若前置条件未知或阻断，就不能用 `proceed` 放行；新证据出现后需要重新核验。

实现还区分内部审核失败与环境拒绝：参数错误、未处理依赖、尚有等待中的观测等，会要求重规划，不能直接当作任务不可行的证据。

**该门控没有消除语义错误。** 返回值明确含 `semantic_correctness_verified: False`：代码能检查引用是否真实、格式是否符合要求、动作是否绑定正确，但前置条件是否完整、证据是否相关，仍依赖 actor 的判断。

### 5.4 六类内部工具的作用

1. `harness_review_action`：记录和检查针对某个精确动作的证据与前置条件，决定调查、停止或放行。
2. `harness_reconcile_references`：处理已发现但未读的关联记录。允许用证据说明某链接不适用、已被其他证据覆盖或影响有界，避免强制读取所有记录。
3. `harness_compare_requirements`：比较公开要求与已观测授权的 scope、role、version、action class、audience 等值；数值或标签比较不自动证明授权语义成立。
4. `harness_resolve_binding`：通过真实观测中的引用值与记录 ID 做精确连接，避免按读取顺序或“看起来更新”来选择记录。
5. `harness_check_constraints`：对 actor 从公开观测建立的有界整数模型做精确计算。支持最多 12 个变量、48 条等式/不等式行和 12 个附加 predicates；消元后最多枚举 4096 个自由变量赋值，过大时返回 incomplete。
6. `harness_check_relations`：用精确分数计算小型有向无环图所有 source-to-target 路径的乘积之和，支持最多 32 个节点和 64 条边；权重、阈值来自已保留的观测。

后两者提供的是**给定建模条件下的计算正确性**。没有建入模型的规则、替代路径或相关节点仍可能遗漏，incomplete 也不能当作不可行证明。

### 5.5 最终报告与一次受限修订

`finalize_output` 根据观测到的公开报告契约检查 schema 和记录引用。必要时要求修订同一份报告；合法 JSON 若只多了 Markdown fence，可以做有限规范化。源码说明不会替换语义答案。

复现说明称 runner 最多允许一次报告修订，仍计入原来的 30-turn 总预算。已经完成的写入不能因报告修订而重做。这一机制也提醒读者：最终得分改善可能来自多种工程修复，包含报告格式、证据完整性和动作控制，不能只归因于一个 abstention prompt。

## 6. 实验设置与指标

### 6.1 数据与推理配置

开发执行模型为 GPT-5.6-Luna；冻结最终 harness 后，在另外 19 个模型上评估，无进一步模型专属优化。每个任务变体至多 30 turns，每次模型响应最多 8,192 output tokens；temperature、top-p 使用 API 默认值。

Luna 通过 Responses API，DeepSeek 通过官方端点，其余模型经指定 OpenRouter 路由且禁用自动 fallback。多数模型显式请求 High reasoning；Qwen3.8 Flash、MiniMax-M3 为 Enabled，Luna、Mistral Large 3 为 Default。跨模型比较应视为这些配置下的系统结果，不能把不同 provider 的 reasoning 标签看成相同计算预算。

同一模型的两种 harness 使用相同配置、工具、初始环境和预算。每个 episode 重置环境和 harness 状态，reference、私有验收条件、正负标签不暴露给 solver。

### 6.2 指标的精确定义

对 $N$ 对任务，$c_i$ 表示可行实例成功完成，$b_i$ 表示不可行实例正确 abstain，$o_i$ 表示在可行实例上明确放弃全部或某个必要部分，均取 0 或 1。

$$
\mathrm{Abstain}=\frac{100}{N}\sum_{i=1}^{N}b_i,\qquad \mathrm{Act}=\frac{100}{N}\sum_{i=1}^{N}c_i.
$$

$$
\mathrm{Pair}=\frac{100}{N}\sum_{i=1}^{N}c_i b_i,\qquad \mathrm{Over\text{-}abstain}=\frac{100}{N}\sum_{i=1}^{N}o_i.
$$

Pair 要求同一对的两个实例同时正确，因此一般不等于 Act 与 Abstain 的乘积。Over-abstain 也不等于 100−Act，因为可行任务可能因错误答案、失败动作或输出不完整而失败，而没有明确拒绝。

### 6.3 对照组在回答什么问题

- **H0**：未适配基础 harness。
- **H0 + abstention prompt**：仅增加固定停止指令。
- **Harness-only evolution**：只用初始任务池演化 harness。
- **Task-augmented harness evolution**：扩充普通有效任务对，但不进行针对当前弱点的环境演化。
- **Task generation from H0 failures**：仅根据初始 H0 的失败生成后续任务，再按批释放。
- **Co-evolution**：持续用当前 harness 的反馈更新环境与任务生成。

这些对照分别检验 prompt、harness 修改、数据量、初始失败定向生成和持续反馈的作用。它们没有逐项移除最终 harness 中每个内部工具，因此不能把总体增益拆成各模块的独立贡献。

## 7. 主结果与细粒度解释

### 7.1 原文 Table 1：60 对 held-out 任务

下表按原文值转录，单位为 %；粗体保留原表最佳值含义。

| Method | Abstain ↑ | Act ↑ | Pair ↑ |
| --- | ---: | ---: | ---: |
| H0 | 61.7 | 68.3 | 48.3 |
| H0 + abstention prompt | 68.3 | 63.3 | 46.7 |
| Harness-only evolution | 76.7 | 71.7 | 60.0 |
| Task-augmented harness evolution | 80.0 | 75.0 | 65.0 |
| Task generation from H0 failures | 78.3 | 75.0 | **73.3** |
| **Co-evolution (ours)** | **83.3** | **76.7** | 70.0 |

相对 H0，HERA 的 Abstain 增加约 21.6 个百分点，Act 增加约 8.4 个百分点，Pair 增加约 21.7 个百分点。按 60 对还原整数计数，分别是 37→50、41→46、29→42；显示值四舍五入会造成百分点评估约 0.1 的差异。

固定 abstention prompt 提高停止成功，却让 Act 和 Pair 下降，说明“更容易停止”不足以证明行为更可靠。Harness-only 已获得明显收益；HERA 相对一般任务扩充的增益则是 Abstain 3.3、Act 1.7、Pair 5.0 个百分点，在 60 对上分别约为 2、1、3 个实例或任务对。

**Pair 的最佳基线需要单独强调。** 初始失败生成达到 73.3%，高于 HERA 的 70.0%。按照原表，前者正确完成 45 个正例、正确停止 47 个负例，交集 44 对；HERA 为 46、50、交集 42 对。HERA 两个边际成功率更高，但正确行为在同一对上的重合更少。这个集合计数解释来自原表推导，不是额外实验。

因此论文结果支持“主模型的停止成功率与可行任务完成率共同提高”，不能表述为“在所有指标上全面优于所有基线”。

### 7.2 原文 Table 2：累计训练池上的最终 harness

每个单元格是 **Abstain / Act (%)**。各列评估的是同一个最终 harness，只改变累计任务池；这些列不是训练中各轮当时版本的分数。

| Method | T0 | T1 | T2 | T3 | T4 | T5 | T6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Harness-only evolution | 90.0 / 80.0 | 84.0 / 76.0 | 76.7 / 70.0 | 77.1 / 71.4 | 70.0 / 67.5 | 66.7 / 66.7 | 64.0 / 66.0 |
| Task-augmented harness evolution | 90.0 / 85.0 | 88.0 / 84.0 | 80.0 / 80.0 | 80.0 / 80.0 | 75.0 / 77.5 | 75.6 / 77.8 | 74.0 / 76.0 |
| Task generation from H0 failures | **95.0 / 90.0** | **96.0 / 92.0** | 86.7 / 86.7 | 85.7 / 88.6 | 82.5 / 85.0 | 77.8 / 84.4 | 76.0 / 82.0 |
| **Co-evolution (ours)** | **95.0 / 90.0** | **96.0 / 92.0** | **90.0 / 90.0** | **91.4 / 91.4** | **90.0 / 90.0** | **86.7 / 88.9** | **86.0 / 88.0** |

HERA 与初始失败生成在 T0、T1 相同，后来分开。T6 上提升 10 个 Abstain 百分点、6 个 Act 百分点，与持续针对新弱点生成任务的动机一致。

但这些是优化过程可见的训练池结果，而且池中新增任务本身由 HERA 当前弱点筛选而来。它们适合说明覆盖与适配，不能单独替代独立泛化评测。

### 7.3 原文 Table 6：同一累计池上的 Pair

| Method | T0 | T1 | T2 | T3 | T4 | T5 | T6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Harness-only evolution | 75.0 | 68.0 | 60.0 | 60.0 | 52.5 | 48.9 | 46.0 |
| Task-augmented harness evolution | 75.0 | 72.0 | 66.7 | 65.7 | 60.0 | 60.0 | 58.0 |
| Task generation from H0 failures | **85.0** | **88.0** | 80.0 | 80.0 | 75.0 | 71.1 | 68.0 |
| **Co-evolution (ours)** | **85.0** | **88.0** | **83.3** | **85.7** | **82.5** | **80.0** | **78.0** |

T6 上 HERA 比次优方法高 10 个 Pair 百分点，但这一优势在 held-out Table 1 中发生反转。训练分布适应与外部分布配对泛化需要分开判断。

### 7.4 原文 Table 3：跨模型迁移

为便于纵向阅读，将原表模型列转为行；每个单元格保留 **Abstain / Pair (%)**，不改变数值。Final 的两项均高于对应 H0。

| Model | H0：Abstain / Pair | Final：Abstain / Pair |
| --- | ---: | ---: |
| GPT-5.6-Luna（开发模型） | 61.7 / 48.3 | **83.3 / 70.0** |
| GPT-5.6-Terra | 76.7 / 66.7 | **86.7 / 73.3** |
| GPT-5.6-Sol | 78.3 / 66.7 | **90.0 / 78.3** |
| GPT-6-Astra | 81.7 / 70.0 | **91.7 / 80.0** |
| Claude Opus 5 | 60.0 / 38.3 | **76.7 / 61.7** |
| Claude Sonnet 5 | 40.0 / 33.3 | **60.0 / 41.7** |
| Gemini 3.1 Pro (Preview) | 53.3 / 35.0 | **70.0 / 48.3** |
| Gemini 3.8 Flash | 70.0 / 48.3 | **83.3 / 50.0** |
| Gemini 3.5 Flash-Lite | 71.7 / 43.3 | **83.3 / 61.7** |
| Grok 4.6 | 71.7 / 60.0 | **78.3 / 66.7** |
| DeepSeek-V4.1-Flash | 51.7 / 41.7 | **58.3 / 46.7** |
| GLM-5.2 | 31.7 / 15.0 | **50.0 / 33.3** |
| GLM-5.3 | 36.7 / 26.7 | **61.7 / 43.3** |
| GLM-5.3-Flash | 43.3 / 31.7 | **56.7 / 43.3** |
| Kimi K3 | 58.3 / 48.3 | **78.3 / 68.3** |
| Qwen3.8-2.4T-A95B | 40.0 / 35.0 | **80.0 / 63.3** |
| Qwen3.8 Flash | 53.3 / 41.7 | **70.0 / 58.3** |
| MiniMax-M3 | 40.0 / 23.3 | **46.7 / 30.0** |
| Mistral Medium 3.5 | 10.0 / 8.3 | **26.7 / 20.0** |
| Mistral Large 3 | 6.7 / 3.3 | **16.7 / 5.0** |

论文报告其他 19 个模型平均提高 15.3 个 Abstain 百分点、12.5 个 Pair 百分点。Qwen3.8-2.4T-A95B 的提升尤其大；GPT-6-Astra 的较强基础表现也继续改善。这支持部分可靠性问题可通过可迁移的外部控制机制缓解。

绝对水平仍有显著差异：Mistral Large 3 的 Final Pair 仅 5.0%，Mistral Medium 3.5 为 20.0%。“迁移有效”不等于“各模型都达到可用水平”。

### 7.5 原文 Table 5：Act 与 Over-abstain 的完整补充

下表同样将原表模型列转为行，保留 **Act ↑ / OA ↓ (%)**。原表 Final 行整体加粗属于样式，并不意味着所有值都更优。

| Model | H0：Act / OA | Final：Act / OA |
| --- | ---: | ---: |
| GPT-5.6-Luna | 68.3 / 0.0 | 76.7 / 1.7 |
| GPT-5.6-Terra | 83.3 / 0.0 | 78.3 / 0.0 |
| GPT-5.6-Sol | 78.3 / 0.0 | 81.7 / 0.0 |
| GPT-6-Astra | 81.7 / 0.0 | 81.7 / 0.0 |
| Claude Opus 5 | 61.7 / 3.3 | 78.3 / 0.0 |
| Claude Sonnet 5 | 70.0 / 0.0 | 66.7 / 0.0 |
| Gemini 3.1 Pro (Preview) | 70.0 / 0.0 | 70.0 / 0.0 |
| Gemini 3.8 Flash | 60.0 / 0.0 | 56.7 / 0.0 |
| Gemini 3.5 Flash-Lite | 61.7 / 0.0 | 70.0 / 0.0 |
| Grok 4.6 | 81.7 / 0.0 | 80.0 / 0.0 |
| DeepSeek-V4.1-Flash | 73.3 / 0.0 | 75.0 / 0.0 |
| GLM-5.2 | 56.7 / 1.7 | 71.7 / 1.7 |
| GLM-5.3 | 68.3 / 0.0 | 71.7 / 0.0 |
| GLM-5.3-Flash | 58.3 / 1.7 | 78.3 / 1.7 |
| Kimi K3 | 76.7 / 0.0 | 85.0 / 0.0 |
| Qwen3.8-2.4T-A95B | 73.3 / 0.0 | 76.7 / 1.7 |
| Qwen3.8 Flash | 68.3 / 0.0 | 70.0 / 1.7 |
| MiniMax-M3 | 56.7 / 3.3 | 56.7 / 0.0 |
| Mistral Medium 3.5 | 61.7 / 0.0 | 60.0 / 0.0 |
| Mistral Large 3 | 21.7 / 13.3 | 20.0 / 15.0 |

按表统计，20 个模型中 11 个 Act 上升、3 个持平、6 个下降。Terra 下降 5.0 个百分点，Sonnet 5 和 Gemini 3.8 Flash 各下降 3.3；它们的 OA 仍为 0，所以不能仅用“拒绝变多”解释完成率损失。

主开发模型 Luna 的 Act 改善，但 OA 从 0.0 到 1.7，约对应一个可行实例。论文对主实验的“保持/提高可行任务完成率”有数据支持；对所有迁移模型作同样保证则不成立。

## 8. 演化动态、环境变化与成本

### 8.1 原文 Figure 2：同一批任务随版本变化

![原文 Figure 2 左：Abstain 演化轨迹](assets/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/abstain_uneven_rounds.png)

![原文 Figure 2 中：Pair 演化轨迹](assets/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/pair_uneven_rounds.png)

![原文 Figure 2 右：Act 演化轨迹](assets/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/act.png)

Figure 2 与累计池表不同：每条曲线固定一批任务，再观察 harness 更新。$B_0$ 含初始 20 对，其余 $B_i$ 各含新加入的 5 对，且互不重叠。

$$
T_i = \bigcup_{j=0}^{i}B_j.
$$

新批次初始分数往往更低，更新后提高；原文举例，B6 上 H5→H6 的 Abstain 与 Act 都从 60% 到 80%。这符合“发现新弱点，再修复”的闭环预期。

每个新批次只有 5 对，20 个百分点实际只对应一个实例成功状态改变。旧任务曲线总体稳定且有波动，因此图支持经验上的保留能力，不构成无遗忘保证。

### 8.2 原文 Figure 6：环境究竟变了什么

![原文 Figure 6：六轮环境演化的主要变化类型](assets/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/env_evolve.png)

每轮五对任务按主要变化归类。E1 含 3 个证据有效性/访问变化、2 个规格/接口变化；E2 为 3 个新约束组合、2 个证据变化；E3、E4 各 5 个全部是新约束组合；E5 为 4 个实体/关系扩展、1 个证据变化；E6 为 4 个证据变化、1 个规格/接口变化。

这说明任务难度同时涉及规模与决策结构：有的扩充对象数量，有的改变多个条件能否共同成立，还有的要求识别某项限制是否仍有效。图中分类属于演化样本的描述分析，未单独量化每类变化的因果贡献。

### 8.3 原文 Figure 3：如何理解“便宜 85%”

![原文 Figure 3：API 成本与 Abstain / Pair](assets/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/hera_bench_api_cost_main.png)

论文对 14 个模型做成本分析。相对于同一模型的基础 harness，Final 的 API 成本中位数增加到 **2.09 倍**，token 使用量中位数增加到 **3.31 倍**。

“85% 更低”来自跨模型配置比较：GPT-5.6-Luna + HERA 达到 70.0% Pair，每 episode 估计 0.0091 美元；GPT-6-Astra + H0 同为 70.0% Pair，每 episode 0.0610 美元。

$$
1-\frac{0.0091}{0.0610}\approx 85.1\%.
$$

这意味着在该 benchmark 和所报 API 价格下，可以用更低成本模型配合更复杂 harness 达到某个强模型基础配置的 Pair 水平。不能据此说 HERA 让同一模型更省钱，也不能说达到所有最强配置的水平：Astra + Final 的 Pair 为 80.0%。

该数字是推理 episode 的估计成本，没有覆盖环境生成、求解验证、候选搜索、人工修复与维护等全部投入。预算敏感应用还应计入延迟和有界计算工具开销。

## 9. 优点、局限与证据强度

### 9.1 主要优点

**配对控制减少了指令层捷径。** 相同请求、system prompt 与工具接口，依靠环境证据改变正确行为，有助于测量 agent 是否真正适应可行性变化。

**正例与负例一起优化。** Pair 和 Act 使“一律停止”无法获得高质量整体表现；固定 abstention prompt 的对照也直接展示了该风险。

**程序化环境提供更强监督。** 可重放 reference、最终状态、变异撤销、替代路径检查，比纯文本印象评分更易追踪。

**外部机制可跨模型复用。** 20 个模型的 Abstain 和 Pair 都提高，说明公共证据组织和控制流可以缓解跨模型存在的系统性失败。

**原始代码可检查重要边界。** 对 observation provenance、精确动作参数、内部失败与环境失败的区分，提供了比框架图更具体的工程参考。

### 9.2 最关键的复现差异：自动协议与人工辅助最终组件

论文附录 A.3 描述 “fully automated optimization protocol”。但仓库固定版本的 `docs/reproduction.md` 明确写明：**“The final harness includes human-assisted repairs.”** `harnesses/components.json` 同时记录 `final_human_assisted: true`，最终组件 ID 为 `R10::source_report_v1`。源码开头也标注 “Human-assisted repair revision 17”。

同一复现文档声明 base/final 组件与被评估来源文件 byte-identical，同时说明公开 runner 替换了原研究 Docker candidate controller，**不构成完整 harness-evolution pipeline 的复现**。

这意味着应将三件事区分开：论文提出的自动协议、最终发布组件的实际来源，以及这些组件的评测表现。现有材料足以支持“含这些工程机制的最终 harness 改善结果”，但不足以独立量化纯自动共同演化与人工修复分别贡献多少。

这一差异本身不证明数据泄漏或结果无效；需要完整候选谱系、人工修改时间点与依据、修改前后消融和数据访问记录来澄清归因。源码里的 R10 名称也不能直接拿来推翻正文的六轮实验描述，二者版本映射尚需记录解释。

### 9.3 评测规模与统计稳定性

HERA-Bench 只有 60 对，验证集只有 15 对。测试集一个实例约 1.67 个百分点，验证集一个实例约 6.67 个百分点。文中对主结果未提供多随机种子分布、置信区间或成对显著性检验；附录选择协议明确每任务变体一次 rollout。

因此 HERA 相对 H0 的大幅提升值得关注，而相对接近基线的 1–3 个任务差距应谨慎解读。反复在同一小验证集选 checkpoint 即使不向编辑器直接提供验证轨迹，也仍可能产生选择偏差。

### 9.4 无解认证、合成环境与真实部署

负例认证包含替代路径检查，但没有给出对所有可能策略的完备性证明。真实环境的恢复路径、工具行为、文档版本和权限继承可能更复杂。

原文天气例子还直接返回 `recoverable: false`，使停止线索很明确。复现者应测试移除这类显式提示、混入过期或冲突元数据后是否仍有效；这是建议的鲁棒性评测，论文未报告该消融。

HERA-Bench 经人工审核的合成环境能控制变量，但其结论不自动覆盖真实 Web、生产数据库、异步状态变化或带有对抗内容的环境。

### 9.5 组件归因与计算预算

更强 harness 增加 token 与调用成本。现有方法级对照不能完全分离“更好的控制逻辑”“更多推理计算”“精确算术工具”“输出格式修复”等因素。缺少等成本比较与逐组件移除实验时，不宜把全部提升归因于共同演化本身。

验证选择只约束 Abstain、Pair，也不保证每个模型的 Act 或 OA 改善。真正要求“完成率不退化”的应用需要直接将 Act 和错误停止率纳入选择约束。

## 10. 复现与工程落地要点

### 10.1 当前公开材料能做什么

本次读取的 README 要求 Python 3.12，并说明数据集当时为 private，下载需要获授权 Hugging Face 账户。公开代码包含基础与最终 harness、共享 runner 和测试；环境、数据及 graders 在 Hugging Face。这个访问状态来自读取日文档，可能随发布更新。

作者提供的流程是下载固定数据版本、核验 checksum、离线 replay 120 个 reference，再分别运行 base / final 并评分。关键命令如下，供后续在独立复现目录使用：

```bash
python -m hera_runner download
python -m hera_runner verify
python -m hera_runner run --harness base --config configs/model.json --pairs all --output runs/base
python -m hera_runner run --harness final --config configs/model.json --pairs all --output runs/final
python -m hera_runner score runs/base
python -m hera_runner score runs/final
```

本次工作完成论文和源码阅读、资源保存及笔记整理，没有运行这些付费模型评估，因此文中准确率均为论文报告值，未声称独立复现。

### 10.2 应固定并记录的条件

1. 模型具体 ID、provider 路由、推理配置和接口版本；Default 应保持参数省略的语义。
2. 数据 revision、每个文件的校验值、正负实例与派生任务所在 split。
3. Harness commit、组件 hash、是否含人工修改、每个候选的父版本和变更内容。
4. 每 episode 的工具事件、真实初态/末态、模型输出、耗时、token 与失败类型。
5. 统一的 30-turn 与每响应 8,192-token 限制，含报告修订占用的预算。
6. 参考答案和私有标签对 solver 的隔离，工具失败和缺失 episode 的统计处理。

公开 runner 文档说明 runtime failure 计为不成功；缺少 episode 会将整次运行标为 incomplete 并抑制准确率输出。只运行 subset 时分母应为选中的任务对数，并明确标记 subset。

### 10.3 如果要复现共同演化本身

还需要发布或重建：环境生成与修复 prompts、mutation proposer 的数量和预算、semantic/rescue policies、各角色模型、完整候选和失败诊断日志、15 对验证集、候选 tie-break、拒绝的环境批次、人工修复轨迹及各基线的搜索资源预算。

对停止行为的 grader 也应审计：是否要求指出真实障碍、是否允许有依据的部分结果、如何识别先执行错误动作后再拒绝，以及任务完成和 abstention 的评分是否覆盖最终状态。

## 11. 对 Agent 研究与当前知识库的可落地启发

以下是阅读后的研究建议，未作为论文既有结果。

### 11.1 用最小变异构造反事实任务对

在已有 tool-use 环境中选已能重放的成功任务，针对一项必要条件做最小变异：缺失证据、过期授权、共享资源冲突或不可恢复依赖。保持请求与公共接口相同，并添加撤销变异控制。

先验证正例成功，再搜索负例恢复路径，最后联合报告 Act、Abstain、Pair、OA 和成本。这个设计能接到知识库中环境合成、tool-use evaluation 和 self-evolving agent 的阅读主线上。

### 11.2 把停止判断拆成局部条件与证据来源

可先实现一个小范围原型：为真实工具观测编号；动作前列出必要条件、scope 和证据；只对有后果的动作进行门控；执行后对照 postcondition；最终报告保留已证实部分和具体缺口。

评测重点应是错误调用是否减少、任务成功是否保留、平均工具数是否增长，以及“内部检查失败误判为环境无解”的比例。不能只观察最终文本中拒绝词是否更多。

### 11.3 做等预算与人工修复贡献消融

优先比较基础 harness、仅加 prompt、固定手写核验、自动演化未人工修复版、人工修复后的最终版。统一推理 token/美元预算，按相同任务和多次 rollout 比较。

逐项移除 reference reconciliation、constraints solver、关系计算和最终报告修订，观察增益来自哪种机制。这样能把论文目前较弱的因果归因补齐。

### 11.4 扩展为恢复能力评测

在负例旁增加“看起来失败但可恢复”的控制实例：首次调用失败后重试成功、权限通过公开合法流程补齐、另一个来源能提供缺失证据、历史限制已解除。测试 agent 是否区分不可恢复阻断与可修复障碍。

对真实系统，这比单纯扩大 abstain 负例数量更能抑制保守化。

### 11.5 对共同演化做分布外审计

训练域、环境族、来源文档与变异机制分别分组留出；冻结测试集后，只用训练失败演化。保存每轮被拒绝与接受的任务，检查是否越来越集中在少数错误模式。

可进一步在新领域保留同一故障机制，或在同一领域引入新故障机制，分别测量语义迁移与机制迁移。评测中加入未见过的缺失字段表达和权限表示，验证收益是否依赖显式标签。

## 12. 图表与证据索引

- **Figure 1 / §3.2 / PDF p.4**：共同演化总图，本笔记 §4。
- **Figure 2 / §5.1 / PDF p.8**：按固定批次追踪演化，本笔记 §8.1。
- **Figure 3 / §5.2 / PDF p.9**：API 成本与性能，本笔记 §8.3。
- **Figure 4 / §5.3 / PDF p.9**：代表性 harness 变化，本笔记 §5.1。
- **Figure 5 / Appendix A / PDF p.16**：任务与环境构造，本笔记 §3。
- **Figure 6 / Appendix E.3 / PDF p.21**：环境变更分类，本笔记 §8.2。
- **Figure 7–8 / Appendix F / PDF p.22–23**：天气任务对，本笔记 §3.5。
- **Table 1 / PDF p.7**：主基线比较，本笔记 §7.1。
- **Table 2 / PDF p.7** 与 **Table 6 / PDF p.21**：累计训练池，本笔记 §7.2–7.3。
- **Table 3 / PDF p.8** 与 **Table 5 / PDF p.20**：跨模型结果，本笔记 §7.4–7.5。
- **Table 4 / PDF p.17**：具体 API / reasoning 配置，原表在本地 PDF，关键差异见本笔记 §6.1。
- **Appendix A–D / PDF p.16–20**：4 次候选更新、15 对验证集、编辑边界、pair certification 与人工审核。
- **仓库证据快照**：[snapshot.json](assets/arxiv_2610_hera_harness_environment_co_evolution_for_reliable_agentic_abstention_tex/repository_snapshot/snapshot.json)，包含 commit 与下载文件清单。

阅读覆盖：已覆盖主文方法与实验、重要附录、全部 8 个编号图的内容与关键结果表；本文件为细粒度分析，参考文献列表及非技术声明未逐段翻译。所有性能数值来自指定 v1 的有效正文/表格，未使用 TeX 注释中的旧结果；实现细节和来源差异使用单独标明版本的公开代码。
