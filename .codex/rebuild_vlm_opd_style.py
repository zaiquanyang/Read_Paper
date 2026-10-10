from pathlib import Path
import html
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "vlm_工作整理_github_papers"
CDN = "https://pub-6df1d50179b24ebcb783184cdb6efc8c.r2.dev/Read_Paper/vlm/assets"

CSS = r'''<style>
:root{--bg:#f8f9fa;--card:#fff;--text:#1a1a2e;--muted:#596273;--accent:#2d6cdf;--light:#e8f0fe;--border:#e2e8f0}
*{box-sizing:border-box}body{margin:0;padding:2rem 1rem;background:var(--bg);color:var(--text);font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.7}.container{max-width:960px;margin:auto}.back{margin-bottom:1rem}.back a{color:var(--accent);text-decoration:none;font-weight:700}.paper-card{background:var(--card);border:1px solid var(--border);border-radius:12px;box-shadow:0 2px 8px #0000000d;padding:clamp(1.25rem,3vw,2rem)}h1{font-size:1.65rem;line-height:1.35;margin:0 0 .8rem}.meta-row{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin-bottom:1.15rem;font-size:.84rem}.tag,.meta-row a{padding:2px 9px;border-radius:99px;background:var(--light);color:var(--accent);text-decoration:none;font-weight:700}.summary-box{background:#f0f7ff;border-left:4px solid var(--accent);padding:.7rem 1rem;border-radius:0 8px 8px 0;margin-bottom:1.35rem}.section-title{font-size:1.1rem;margin:1.45rem 0 .65rem;border-bottom:1px dashed var(--border);padding-bottom:.3rem}.paper-card p,.paper-card li{font-size:.93rem;color:#333}.paper-card ul,.paper-card ol{padding-left:1.25rem}.paper-card li{margin:.25rem 0}.paper-card img{display:block;max-width:100%;height:auto;margin:.6rem auto;border:1px solid var(--border);border-radius:8px;object-fit:contain}.figure-showcase{margin:1rem 0 1.3rem;padding:.85rem;background:#fbfcff;border:1px solid var(--border);border-radius:9px}.figure-title{font-weight:700;color:var(--muted);font-size:.86rem;margin-bottom:.45rem}.figure-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem}.figure-grid .primary{grid-column:1/-1}.figure-link{display:block}.figure-link img{width:100%;max-height:440px}.figure-grid .primary img{max-height:560px}.paper-card table{width:100%;border-collapse:collapse;margin:.85rem 0;font-size:.88rem;display:block;overflow-x:auto}.paper-card th,.paper-card td{border:1px solid var(--border);padding:6px 9px;white-space:nowrap}.paper-card th{background:var(--light)}.paper-card code{background:#f1f5f9;padding:2px 5px;border-radius:4px}.formula-block{padding:.55rem .75rem;margin:.7rem 0;background:#fafafa;border-left:3px solid #94a3b8;overflow-x:auto}.pros-cons{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.pros-cons>div{padding:.7rem .9rem;border-radius:8px}.pros{background:#f0fdf4;border-left:3px solid #22c55e}.cons{background:#fef2f2;border-left:3px solid #ef4444}.pros-cons h4{margin:.05rem 0 .25rem;font-size:.9rem}@media(max-width:640px){.pros-cons,.figure-grid{grid-template-columns:1fr}.paper-card{padding:1.2rem}}
</style>'''

# Each entry is written from that paper's Markdown notes. The generator only
# supplies shared page chrome; the prose and evidence remain paper-specific.
PAPERS = {
"arxiv_2604_improving_vision_language_models_with_perception_centric_process_reward_models": {
"summary":"Perceval 把 VLM 的结果级奖励细化为感知级过程监督：定位回答中与图像不一致的 token/span，再在 GRPO 中逐 token 调整优势，减少“答对但看错图”的伪正确推理。",
"problem":"RLVR 只看最终答案，无法指出推理链中哪一步产生了物体、属性或空间关系幻觉。Perceval 的目标是把感知错误变成可定位、可惩罚的训练信号，而不是再增加一个黑盒标量奖励。",
"method":"<p><strong>生成式 PRM：</strong>输入图像、问题和回答，先逐条核对感知 claim，再输出“正确”或幻觉原文子串。</p><ol><li>用 Qwen2.5-VL-7B rollout 产生含噪回答，再由强模型生成逐步幻觉标注。</li><li>将幻觉 span 转为掩码 <code>m<sub>i,t</sub></code>，把序列优势改为 <code>A′<sub>i,t</sub> = A<sub>i</sub> − αm<sub>i,t</sub>|A<sub>i</sub>|</code>。</li><li>仅在感知数据上启用过程惩罚；数学推理等数据仍使用标准 GRPO。</li><li>推理时支持 truncate-regenerate 和 truncate-thinking-regenerate 两种纠错。</li></ol>",
"evidence":"<p>3B 模型相对 GRPO：V*<sub>all</sub> 从 80.10 提升到 <strong>83.25</strong>，RWQA 从 62.1 提升到 <strong>64.9</strong>，MathVision 从 23.36 提升到 <strong>26.32</strong>；7B 上 V*<sub>pos</sub> 达到 <strong>86.84</strong>、BLINK 达到 <strong>54.49</strong>。感知训练的收益也迁移到数学和图表推理。</p><p>测试时截断重生成在 V* 上优于 majority voting，说明过程纠错比单纯增加采样更有效。</p>",
"takeaway":"Perceval 把“视觉 grounding 是否可靠”从最终答案中拆出来，形成可解释的 credit assignment。代价是依赖强教师标注，且迭代重生成会增加延迟。",
"limitations":"PRM 标注依赖 Gemini-2.5-Pro 等强教师；方法主要检测感知幻觉，对纯逻辑错误和更大模型的扩展仍未充分验证。",
"figures":[("final-main.png","Perceval 过程奖励与 RLVR 流程",True),("hallucination_rate.png","幻觉检测结果",False)]},
"arxiv_2604_learnpruner_rethinking_attention_based_token_pruning_in_vision_language_models": {
"summary":"LearnPruner 发现视觉编码器 CLS 注意力和深层视觉注意力都不是可靠的 token 重要性指标，于是用轻量可学习剪枝加中间层文本注意力做两阶段压缩，在极少 token 下保持 VLM 能力。",
"problem":"LLaVA-NeXT 等模型会产生数千视觉 token。现有剪枝依赖 CLS 或视觉注意力，容易被背景 artifact 和 attention shift 误导；真正需要的是与文本问题相关、且跨样本稳定的视觉证据。",
"method":"<ol><li><strong>阶段一：</strong>视觉编码器后接 0.53M 参数的 LPM，用 MLP+STE 预测保留/剪枝，并用多样性选择补回低相似度 token。</li><li><strong>阶段二：</strong>在 LLM 第 12 层聚合文本到视觉的注意力，只保留与 query 相关的 top-k token。</li><li>两个阶段的保留比例为 <code>R<sub>1</sub>:R<sub>2</sub>=3:1</code>；冻结基础 VLM，仅用 LLaVA-665K 的 10% 训练 LPM。</li></ol>",
"evidence":"<p>在 LLaVA-v1.5-7B 仅保留 32 token 时，LearnPruner 保留原模型 <strong>94.8%</strong> 的综合性能，而 FastV 只有 58.6%；保留 64 token 时达到 96.9%。在高分辨率 LLaVA-NeXT（2880→640 token）和 Qwen2.5-VL 上仍保持接近完整输入的效果。</p>",
"takeaway":"核心贡献是先验证“哪一种注意力值得信任”，再把不同层级的信号放到合适的位置。",
"limitations":"剪枝层和比例仍固定，复杂场景可能丢失背景证据；方法需要少量训练，尚未在更大模型上验证。",
"figures":[("framework2.png","LearnPruner 两阶段剪枝框架",True),("analyze3.png","注意力可靠性分析",False)]},
"arxiv_2604_saliency_r1_enforcing_interpretable_and_faithful_vision_language_reasoning_via_saliency_map_alignment_reward": {
"summary":"Saliency-R1 用 token 级显著性图和 attention rollout 追踪“思考 token”依赖的视觉区域，再把显著性与人工框标注的对齐度作为 GRPO 奖励，直接优化可解释且忠实的视觉推理。",
"problem":"相同答案可能来自真正看图的推理，也可能来自语言先验或视觉幻觉。仅奖励 answer correctness 无法区分两者，因此需要一个能检查 reasoning 是否关注正确区域的过程信号。",
"method":"<p>方法由三部分组成：<strong>logits 分解</strong>得到视觉 patch 的直接贡献；以 thinking tokens 为瓶颈做 <strong>attention rollout</strong>；将显著性落在 bbox 内的比例作为连续奖励。</p><div class=\"formula-block\">\\[R_{overall}=R_{accuracy}+R_{format}+R_{saliency},\\quad R_{saliency}=\\frac{\\sum_{i\\in bbox}s_i}{\\sum_{i\\in image}s_i}\\]</div><p>每个问题采样 8 条 rollout，用 GRPO 同时约束答案正确、输出格式和视觉对齐。</p>",
"evidence":"<p>在 COCO deletion test 中，方法的 deletion 分数为 70.96/59.45/50.34，优于 CAM、Grad-CAM、ATTN-LRP 和 TAM，说明显著性图更忠实。主实验在 MMMU-Pro、MMStar、IllusionVQA 和 SalBench 等任务上同时改善答案能力与幻觉抑制。</p>",
"takeaway":"它把“解释看起来合理”转化成可测的空间对齐约束，并把解释监督接入 RL。",
"limitations":"依赖 bbox 标注与外部 judge；显著性仍是直接贡献近似，不覆盖所有隐式路径。",
"figures":[("framework.png","Saliency-R1 框架",True),("teaser.png","不同推理过程的显著性差异",False)]},
"arxiv_2604_walk_the_talk_bridging_the_reasoning_action_gap_for_thinking_with_images_via_multimodal_agentic_policy_optimization": {
"summary":"MAPO 解决 multimodal agent 的 reasoning-action gap：模型先用 label 说清楚想观察什么，执行 crop/zoom 后再用 CLIP 检查 observation 是否与意图一致，并把该过程奖励加入策略优化。",
"problem":"只看最终答案会强化“嘴上说要看目标、实际裁错区域”的轨迹。错误动作在多轮视觉工具调用中不断累积，导致稀疏奖励、梯度方差和训练不稳定。",
"method":"<ol><li>模型在每次视觉动作前生成可验证的 descriptive label。</li><li>工具返回局部 observation，CLIP 计算 label 与 observation 的语义一致性。</li><li>将 step-level semantic score 与 outcome reward 融合进 MAPO advantage，约束“说到做到”。</li></ol>",
"evidence":"<p>同一 Ovis2.5-9B backbone 上，MAPO 的 V* Overall 为 <strong>89.5</strong>，HR-Bench Overall 为 <strong>79.8</strong>，优于 GRPO 的 88.6/77.8；HR-Bench 8K 达到 78.6。收益主要集中在高分辨率、细粒度视觉探索任务。</p>",
"takeaway":"把隐式工具意图显式化，再对执行结果做局部验证，是一个可迁移的 multimodal agent reward 模式。",
"limitations":"CLIP 相似度不一定等价于任务一致性；label 质量和 task-specific reward 仍是瓶颈。",
"figures":[("MAPO_method.png","MAPO 的 reasoning-action 对齐流程",True),("reward_plot_new.png","语义奖励训练曲线",False)]},
"arxiv_2605_alphagrpo_unlocking_self_reflective_multimodal_generation_in_umms_via_decompositional_verifiable_reward": {
"summary":"AlphaGRPO 在统一多模态模型中把 reasoning text 与 diffusion image trajectory 放进同一条 RL 轨迹，用 DVReward 将复杂生成要求拆成可验证问题，并通过 FPR 抑制无效自我修图。",
"problem":"UMM 的视觉理解、规划和自我反思能力可能已存在，但额外 cold-start SFT 会混淆 RL 的真实贡献；单一 holistic score 也无法说明图像到底违反了哪一条约束。",
"method":"<ol><li>统一轨迹 <code>τ=(y,z<sub>1</sub>→z<sub>0</sub>)</code>：先生成 reasoning，再生成图像 diffusion trajectory。</li><li>DVReward 把 request 分解为 atomic questions，对每个问题做 confidence scoring，再组合 reward。</li><li>同一 group-normalized advantage 同时更新 AR reasoning policy 和 flow generation policy。</li><li>FPR 将没有超过初始图像的 refinement 压到 group 最低 reward，避免 false-positive optimization。</li></ol>",
"evidence":"<p>512 分辨率上，AlphaGRPO-SRR 在 TIIF Overall-S/L 达到 79.1/79.5，推理时再做 self-reflective refinement 后升至 <strong>83.9/83.2</strong>；1024 分辨率仍达到 82.2/81.4，并在 GenEval 达到 89.5。</p>",
"takeaway":"最值得复用的是“先拆解约束，再逐项验证”的 reward 设计，以及跨文本/图像轨迹的 credit assignment。",
"limitations":"依赖大型 verifier，训练成本高；当前主要是 outcome reward，base model 的生成稳定性也限制上限。",
"figures":[("method.png","AlphaGRPO 统一轨迹与 DVReward",True),("teaser.png","生成与自我反思结果",False)]},
"arxiv_2605_from_web_to_pixels_bringing_agentic_search_into_visual_perception": {
"summary":"From Web to Pixels 提出 Perception Deep Research：先通过网页搜索解析图中目标的真实身份，再把结构化 target hypothesis 绑定到具体像素，形成 search-grounding、search-segmentation 和 search-VQA 的统一任务。",
"problem":"真实图像中的目标身份常藏在图外知识里。直接做 language-to-region 会因实体歧义而找错对象；搜索增强回答又通常停在文本答案，没有把证据落回像素。",
"method":"<p>Pixel-Searcher 是两阶段 agentic workflow：</p><ol><li><strong>Search & Target Resolution：</strong>在 Search-Reason-Resolve 循环中得到 <code>h={e,c,K}</code>，分别表示实体、粗类别和可由图像核验的线索。</li><li><strong>Grounding & Tool Use：</strong>用 <code>h</code> 调用 grounding 工具生成候选区域，再做证据一致性验证，输出 box、mask 或答案。</li></ol>",
"evidence":"<p>WebEyes 上，Pixel-Searcher 的 SearchGround Overall IoU/R@0.5 为 <strong>34.17/41.30</strong>，超过 Qwen3-VL-8B 的 26.81/32.61；SearchSeg gIoU/cIoU 为 <strong>39.17/32.41</strong>；SearchVQA 达到 <strong>42.24</strong>。收益在 Anime、ICON、Product 等需要实体解析的类别最明显。</p>",
"takeaway":"贡献首先是任务定义：把 search 变成 perception 的前端身份解析，而不是回答的后端知识补充。",
"limitations":"WebEyes 规模较小，流程依赖搜索可靠性和外部工具，闭源模型仍明显更强。",
"figures":[("method.png","Pixel-Searcher 两阶段工作流",True),("compare.png","WebEyes 任务与方法对比",False)]},
"arxiv_2605_hide_to_see_reasoning_prefix_masking_for_visual_anchored_thinking_in_vlm_distillation": {
"summary":"Masking-KD 通过遮掉最容易形成文本 shortcut 的 reasoning prefix，恢复 think-answer 蒸馏难度，迫使 student 在生成后续 token 时重新利用图像，而不是只复制 teacher 的文字轨迹。",
"problem":"长 think-answer prefix 会让后续 token 变得容易预测，student 逐渐学会 textual shortcut 并发生 visual forgetting。直接蒸馏 teacher attention 并不能保证 student 自己持续看图。",
"method":"<ol><li>用 auxiliary student forward 估计 response prefix 对当前 token 的影响。</li><li>按 token 级 salient reasoning-prefix mask 遮挡高影响前缀，而保留视觉和问题输入。</li><li>用 self-paced budget 从 <code>ρ<sub>min</sub>=0.3</code> 调到 <code>ρ<sub>max</sub>=0.5</code>。</li><li>在被遮挡条件下做 response-level distillation，训练 student 形成 visual-anchored thinking。</li></ol>",
"evidence":"<p>8B teacher→4B student 时，Masking-KD 平均分达到 <strong>56.66</strong>，超过 Naive Response Distillation 的 48.89、CompoDistill 的 51.29；Geo3K/MathVista/We-Math 为 52.58/66.50/71.03。self-distillation 也有效。</p>",
"takeaway":"核心是“破坏捷径”而不是“复制内部表征”，让 student 在每个阶段重新依赖视觉证据。",
"limitations":"需要 auxiliary forward 和 attention map，训练成本增加；验证主要集中在 reasoning-heavy benchmark。",
"figures":[("intro_10.png","Masking-KD 的视觉锚定直觉",True),("main_27.png","主结果对比",False)]},
"arxiv_2605_hypereyes_dual_grained_efficiency_aware_reinforcement_learning_for_parallel_multimodal_search_agents": {
"summary":"HyperEyes 面向多实体 multimodal search，把“能并行调用工具”变成“愿意以更低成本并行搜索”：用 UGS、并行数据构造、TRACE 成本奖励和失败轨迹 OPD 共同降低工具轮数。",
"problem":"串行 crop-search loop 会不断增加 interaction round 和上下文噪声。仅优化 accuracy 会鼓励 brute-force 搜索，模型不会主动把可分解子任务并行化。",
"method":"<ol><li><strong>UGS：</strong>统一 grounded action，让一轮动作可以定位多个实体并发起搜索。</li><li><strong>数据管线：</strong>合成 multi-entity 样本，用 tool-necessity filtering 与 Progressive Rejection Sampling 保留真正需要搜索的轨迹。</li><li><strong>TRACE：</strong>提供 query-dependent 工具成本奖励；仅对 failed rollout 使用同家族 teacher 的 OPD 做细粒度修复。</li></ol>",
"evidence":"<p>HyperEyes-30B RL 的 Avg Acc/Turns 为 <strong>64.0/2.2</strong>，相对同规模开源 deep-search agent 准确率约 +9.9、工具轮数约 −9.4；相对 VDR 少约 5.3× 工具轮。BCVL/IMEB 的 CAS 达到 2.232/0.910，约为次优开源方法的 4.3×/7.6×。</p>",
"takeaway":"它把并行搜索的动作空间、数据和奖励闭环起来；OPD 在这里是失败轨迹修复器而非主目标。",
"limitations":"更适合可分解的静态图像查询；对强 sequential multi-hop、视频和音频场景，以及无需同家族 teacher 的扩展仍未知。",
"figures":[("parallel_vs_serial.png","串行与并行搜索对比",True),("framework.png","HyperEyes 训练框架",False)]},
"arxiv_2605_learn_to_think_improving_multimodal_reasoning_through_vision_aware_self_improvement_training": {
"summary":"VISTA 将 MLLM self-improvement 从“只收集答对样本”推进到两步数据治理：prefix resampling 为困难 query 回收更多正确轨迹，VAS 过滤答对但缺少视觉 grounding 的伪正确推理。",
"problem":"self-generated reasoning 同时存在 data imbalance 与 language prior bias：简单题容易采到大量正确解，Geometry3K 上超过 40% query 甚至采不到正确解；而一些最终答对的 trace 中间并没有真正看图。",
"method":"<ol><li><strong>Prefix resampling：</strong>在 negative trace 中寻找第一个不在前一步 Top-5 的 critical token，用新 Top-1 替换并截断后续，再从正确前缀继续采样。</li><li><strong>VAS：</strong>在中间层统计 reasoning token 对 visual/system/instruction token 的 attention 比例，标准化为 <code>VAS=(S−mean)/std</code>，过滤 <code>VAS&lt;−0.5</code> 的伪正确解。</li><li>剩余数据可用于 SFT、DPO 或 GRPO；冻结 vision encoder 和 projector，只更新 LLM backbone。</li></ol><div class=\"formula-block\">\\[S=\\frac{\\lambda_{vis}}{\\lambda_{sys}+\\lambda_{vis}+\\lambda_{ins}}\\]</div>",
"evidence":"<p>Geometry3K 上 prefix resampling 将每个 query 的平均正确解数量从 <strong>2.4 提升到 4.4</strong>；hallucinated solution 的平均 VAS 约 −0.2，non-hallucinated 约 0.31。Qwen2.5-VL-3B 的 VISTA-SFT 在 ScienceQA/ChartQA 达到 <strong>82.29/62.09</strong>，跨 Qwen3-VL 与 InternVL 的总体平均达到 <strong>65.30</strong>；相对 3B SFT-Seed 平均提升 <strong>+13.66</strong>。</p>",
"takeaway":"VISTA 把“难样本覆盖率”和“视觉可靠性”分开建模，且不依赖额外 reward model。",
"limitations":"prefix recycling 可能降低轨迹多样性；VAS 仍是 attention proxy，大模型和不同视觉拼接格式上的稳定性待验证。",
"figures":[("method.png","VISTA 总体流程",True),("preliminary_analysis.png","数据不平衡与视觉注意力观察",False),("method_analysis.png","Prefix resampling 分析",False)]},
"arxiv_2605_perceptual_flow_network_for_visually_grounded_reasoning": {
"summary":"PFlowNet 发现几何上最精确的专家框并不一定最利于推理，提出由 planning state 和 perceptual state 组成的 Perceptual Flow，并用质量/效用双奖励学习自参数化视觉轨迹。",
"problem":"直接对齐检测器的精确 RoI 会产生 tunnel vision：上下文被裁得过窄，反而损害视觉推理。理想感知区域具有实例相关的上下文范围，不能用固定几何启发式得到。",
"method":"<p>模型先生成查询分解的 <code>&lt;analyze&gt;</code> planning state，再生成包含 RoI 与 caption 的 <code>&lt;localize&gt;</code> perceptual states，形成 <code>Z=(z<sub>0</sub>→…→z<sub>K</sub>)</code>。训练分两步：用 teacher 合成多样 Flow 做 cold start，再用 Sub-Trajectory Balance 的 RFT 优化；奖励同时衡量 Flow 质量和推理效用。</p>",
"evidence":"<p>Qwen3-VL-8B 上，PFlowNet 在 TreeBench/MME-RealWorld 达到 <strong>55.3/67.0</strong>，相对 base +10.4/+18.4；V* 达到 <strong>90.6</strong>，超过 Qwen3-VL-32B 的 87.4。去掉 Flow 后 V* 从 90.6 降至 83.8，说明 Flow 是语义锚点而非单纯定位工具。</p>",
"takeaway":"关键洞察是“几何精度不等于推理效用”，并用显式可读的视觉轨迹把两者解耦。",
"limitations":"两阶段训练和固定推理格式增加复杂度；理论假设理想化，ε/λ 等超参数需要按模型调节。",
"figures":[("fig4.png","PFlowNet 总体框架",True),("fig1.png","几何先验与推理效用关系",False)]},
"arxiv_2605_persistent_visual_memory_sustaining_perception_for_deep_generation_in_lvlms": {
"summary":"PVM 从注意力分区函数推导视觉信号随文本长度近似 O(t⁻¹) 稀释，随后在 FFN 旁增加并行视觉记忆通路，用独立视觉 cross-attention 和门控融合维持长推理中的视觉访问。",
"problem":"自回归 LVLM 只在序列开头注入视觉 token；随着文本历史增长，视觉 attention mass 被越来越多文本 token 分走，长对话和深推理会逐渐变成文本先验驱动。",
"method":"<p>PVM 与冻结 FFN 并行：将当前 hidden state 和原始视觉 embedding 投影到低维空间，使用视觉域内的 cross-attention 检索，再恢复到主干维度。</p><div class=\"formula-block\">\\[\\Omega_V(t)=\\frac{Z_V}{Z_V+Z_T}\\le\\frac{\\beta}{\\beta+\\mu t}=O(t^{-1})\\]</div><p>可学习门控 λ 初始化为 0，只对文本 token 激活视觉记忆，避免破坏预训练推理路径。</p>",
"evidence":"<p>Qwen3-VL-8B 上 PVM-SFT+GRPO 的 Overall Avg 为 <strong>71.5</strong>，比 base 66.7 高 4.8pp；4B 也提升 4.4pp。增益随输出长度增加，长序列组达到 <strong>+27.3%</strong>；推理 TPOT 只增加 1.18ms（约 4.6%），额外参数约 0.32%。</p>",
"takeaway":"PVM 将“持续看图”从主干自回归竞争中隔离出来，理论、机制和长序列实验相互对应。",
"limitations":"只验证 Qwen3-VL 和静态图像；动态视频、全局自回归状态下的理论保证和其他视觉退化因素仍待扩展。",
"figures":[("PVM.png","Persistent Visual Memory 架构",True),("analysis_length.png","长度与视觉增益关系",False)]},
"arxiv_2605_personal_visual_context_learning_in_large_multimodal_models": {
"summary":"Personal Visual Context Learning 先用 Personal-VCL-Bench 诊断模型不会稳定利用长期视觉历史，再提出 Agentic Context Bank：把离散视觉上下文整理成带证据的记忆条目，并在查询时按需回取视觉证据。",
"problem":"把所有个人视觉历史直接拼进 prompt 会造成 scaling paradox（上下文越多越不稳）和 modality paradox（模型看到了图却不会用）。关键不只是 retrieval，而是如何组织和路由证据。",
"method":"<ol><li><strong>Stage I：</strong>将每条记忆表示为 <code>(τ,d,e)</code>，分别是类型、文本描述和原始视觉证据；通过 ADD/CONFIRM/REVISE/RETRACT 持续合并和纠错。</li><li><strong>Stage II：</strong>先只给文本 bank 做 triage，让模型请求需要核验的 entry ID，再把少量对应视觉证据回填，完成最终回答。</li></ol>",
"evidence":"<p>Gemma-4-31B 上，Agentic Context Bank 在 PerID/ObjID/BehErr/EgoID 等任务达到 <strong>83.25/82.52/66.22/61.60</strong>，全面超过只给语言或全视觉上下文的 baseline。EgoWearer 上，Gemini-3-Flash 的 adaptive evidence 为 <strong>72.82</strong>，比最强标准 baseline 高 17.08。</p>",
"takeaway":"贡献在于把长期视觉上下文拆成“结构化记忆”和“查询时证据路由”两个问题。",
"limitations":"尚未解决长时流的 context retrieval、隐私风险和 Stage-II 证据误选；方法仍是强推理时 baseline。",
"figures":[("teaser_v1.png","个人视觉上下文任务",True),("qualitative_v1.png","Context Bank 与按需取证",False)]},
"arxiv_2605_semantic_enriched_latent_visual_reasoning": {
"summary":"SLVR 在 latent visual reasoning 中增加 attribute-level semantic latent，并用 M-GRPO 在同一区域的多问题之间约束 latent 一致性，使隐式视觉表示从 appearance cue 变成可复用的语义载体。",
"problem":"纯 visual latent 往往只保留局部外观，换一个问法就不稳定；显式 crop/zoom 虽有证据但推理链长。论文要学习一种同时保留视觉细节、对象属性、动作和空间关系的 region-centric latent。",
"method":"<ol><li>Stage 1 将 ROI visual latent 与原始视觉特征对齐，同时用 4096 维 attribute embedding 监督 semantic latent。</li><li>Stage 2 的 M-GRPO 对同一区域的多 query 采样，联合优化 answer correctness、visual/semantic latent consistency。</li><li>SV-QA 专门构造“同一区域、两个语义问题”的评测，直接测 latent 能否跨 query 复用。</li></ol>",
"evidence":"<p>SLVR-7B 在 OKVQA/VizWiz/ChartQA/TextVQA 达到 <strong>61.8/57.8/77.2/79.3</strong>，相对 LVR 分别 +11.2/+24.7/+12.8/+4.2；VisualPuzzles Overall 为 <strong>34.2</strong>，比 LVR +4.8。GQA、AI2D 略低于 LVR，说明语义增强对区域属性和多视角问题更有针对性。</p>",
"takeaway":"贡献在于改变 latent 的监督目标：从“能不能答题”转向“能否稳定承载可复用语义”。",
"limitations":"依赖外部强模型生成 semantic profile 和 judge；dense semantic embedding 可能过度压缩复杂关系，标准 benchmark 也并非全面提升。",
"figures":[("figure2.png","SLVR 两阶段框架",True),("dataset_region_multiq.png","同区域多问题数据构造",False)]},
"arxiv_2605_very_efficient_listwise_multimodal_reranking_for_long_documents": {
"summary":"ZipRerank 面向长文档多模态 reranking，同时压缩输入视觉 token 和输出 decoding：两阶段 listwise 训练迁移排序能力，query-aware pruning 后只解一个 token 得到候选排序分数。",
"problem":"多个页面以图像输入时，prefill 视觉 token 很快爆炸；传统 listwise reranker 又要自回归生成排序序列。只优化准确率而不优化推理路径，无法满足 MMDocIR/M-RAG 的部署时延。",
"method":"<ol><li>Stage 1 在 text reranking 上学习 listwise formulation。</li><li>Stage 2 用 document-image 数据和 teacher soft labels 做多模态微调。</li><li>推理前用 query-image early interaction 做视觉 token pruning，随后用 single-token logits 直接给出 listwise 分数。</li></ol>",
"evidence":"<p>MMDocIR 的 DSE first-stage 上，ZipRerank 的 R@1/R@3/R@5 macro 为 64.2/84.8/89.0，时间仅 <strong>0.36s</strong>；MM-R5 的对应时间为 3.82s。它牺牲少量 top-1，换来更强的 top-k recall 和约一个数量级的延迟下降；在 ColQwen first-stage 上趋势一致。</p>",
"takeaway":"贡献是把训练目标和推理路径一起工程化：不是更强 reasoning，而是更高效的 multimodal reranking system design。",
"limitations":"teacher bias、激进剪枝和 top-1 verifier 能力仍是限制；评测主要集中在 document-image reranking。",
"figures":[("zipranker.png","ZipRerank 总体结构",True),("speed_effectiveness_tradeoff_new.png","速度-准确率权衡",False)]},
}


def source_title(path):
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^#\s+(.+?)\s*$", text, re.M)
    return match.group(1).strip() if match else path.stem


def arxiv_id(path):
    text = path.read_text(encoding="utf-8")
    match = re.search(r"ArXiv ID:\s*`?([0-9.]+(?:v\d+)?)", text, re.I)
    return match.group(1) if match else ""


def figure_block(stem, figures):
    cells = []
    for filename, alt, primary in figures:
        url = f"{CDN}/{stem}/{filename}"
        cls = "figure-link primary" if primary else "figure-link"
        cells.append(f'<a class="{cls}" href="{url}" target="_blank" rel="noopener"><img src="{url}" alt="{html.escape(alt)}"></a>')
    return '<div class="figure-showcase"><div class="figure-title">代表性图示</div><div class="figure-grid">' + "".join(cells) + "</div></div>"


def page(path, data):
    stem = path.stem
    title = source_title(path)
    aid = arxiv_id(path)
    meta = '<span class="tag">VLM</span><span class="tag">逐篇精读</span>'
    if aid:
        meta += f'<a href="https://arxiv.org/abs/{aid}" target="_blank" rel="noopener">arXiv {aid}</a>'
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']]}},svg:{{fontCache:'global'}}}};</script><script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>{CSS}</head><body><div class="container"><div class="back"><a href="../vlm_工作整理_github.html">← 返回 VLM 论文列表</a></div><article class="paper-card"><h1>{html.escape(title)}</h1><div class="meta-row">{meta}</div><div class="summary-box"><strong>核心思想：</strong>{data['summary']}</div><h2 class="section-title">问题与研究动机</h2>{data['problem']}<h2 class="section-title">方法框架</h2>{data['method']}{figure_block(stem, data.get('figures', []))}<h2 class="section-title">实验与核心结论</h2>{data['evidence']}<h2 class="section-title">价值、局限与启发</h2><div class="pros-cons"><div class="pros"><h4>主要价值</h4><p>{data['takeaway']}</p></div><div class="cons"><h4>需要注意</h4><p>{data['limitations']}</p></div></div><h2 class="section-title">论文信息</h2><p><a href="https://arxiv.org/abs/{aid}" target="_blank" rel="noopener">arXiv 摘要页</a> · <a href="https://arxiv.org/pdf/{aid}.pdf" target="_blank" rel="noopener">PDF</a></p></article></div></body></html>'''


for source in sorted((ROOT / "vlm").glob("*.md")):
    if source.stem not in PAPERS:
        raise SystemExit(f"Missing curated summary for {source.stem}")
    (OUT / f"{source.stem}.html").write_text(page(source, PAPERS[source.stem]), encoding="utf-8")
print(f"rebuilt {len(PAPERS)} individually curated VLM pages")
