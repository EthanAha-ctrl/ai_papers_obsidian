---
source_pdf: Step 3.5 Flash.pdf
paper_sha256: 7a5f47ee68d01b848f976f250c433c842483b19ba6da9975db91f08a1b02cf05
processed_at: '2026-08-12T11:04:02-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Step 3.5 Flash：人话版

Andrej，让我用更直觉的方式重新讲一遍这篇paper的story。

---

## 这模型到底是干嘛的

一句话：**用11B的计算量，干196B的活，效果逼近GPT-5.2 xHigh和Gemini 3.0 Pro**。

为什么这事难？因为agentic场景有个要命的profile：你先喂进去一大堆context（比如一个代码仓库的整个文件结构），然后模型要跟环境交互很多轮（改代码、跑测试、看报错、再改）。这个"prefill长 + decode也长"的组合，让传统架构在latency上很痛苦。

Step 3.5 Flash的核心thesis就是：**在agent时代，latency和intelligence、cost是三足鼎立的约束**。光聪明不够，光便宜不够，还得快。

---

## Architecture：三个互相耦合的"省钱"招

### 招一：Sparse MoE 196B/11B

这个你熟悉——总参数196B存在那儿，每个token只激活11B。相当于一个公司有196B个员工，但每个任务只派11B个人去干。好处是capacity大（知识装得多），单次推理便宜。

但MoE有个要命的问题：**Expert Parallelism下的straggler**。你把288个expert分到8张GPU上，如果token routing不均匀，某张GPU分到的token特别多，其他7张GPU就得等它。这个等待能把吞吐杀掉。

所以paper加了一个EP-group balancing loss——不光要让expert之间平衡（标准操作），还要让GPU之间平衡。公式逻辑很简单：如果某个GPU组收到的token比例和概率比例的乘积偏大，就惩罚。

### 招二：Hybrid Attention 3:1 SWA:Full

这是最体现"理解硬件比算FLOPs重要"的地方。

Long context下，full attention的复杂度是 $O(n^2)$，贵。怎么省？用Sliding Window Attention——每个token只看前面512个token，复杂度 $O(n)$。但纯SWA会丢长程连接，所以隔三层插一个full attention层，像"锚点"一样把远处的信息捞回来。

但naive这么搞，性能会掉。为什么？两个原因：

**原因一：SWA的head太少**。Full attention有64个query head，SWA也给64个，不够用。怎么办？把SWA的query head从64加到96。这是几乎免费的——因为SWA在长context下是**memory-bandwidth bound**（IO-bound），增加query head不增加KV cache，所以latency几乎不变。这就是"hardware-aware design"的精髓：你知道瓶颈在memory不在compute，就可以用多余的compute slack做别的事。

**原因二：SWA不会"消化"没用的attention weight**。你想想，softmax的权重加起来必须是1。如果窗口里512个token都没用信息，attention也不知道往哪儿分配，就乱撒。以前的解法是加一个固定的"sink token"当垃圾桶。Step 3.5 Flash的做法更聪明——给每个head配一个gate，gate是一个sigmoid的标量，输入决定开多大。这个gate等价于一个**data-dependent的sink**：当gate半开时，attention output被稀释；当gate全开时，就是标准attention。模型自己学什么时候开gate。

实验结果：sink token平均62.5分，head-wise gate 64.4分，全面更好。

### 招三：MTP-3 加速decode

Multi-Token Prediction就是让模型一次预测后面3个token，配合speculative decoding可以3倍速生成。但MTP head本身也有开销。Step 3.5 Flash的trick是：训练时只激活MTP-1（省训练开销），等backbone训好了，把MTP-2和MTP-3从MTP-1 clone出来，再轻量fine-tune一下。

为什么MTP-3和前面的GQA-8、SWA能协同？因为GQA-8让attention更memory-bandwidth bound，创造了compute slack，speculative decoding的draft+verify开销正好被这个slack吸收。三个设计是互相咬合的。

---

## 训练稳定性：三个坑，一个比一个隐蔽

这是这篇paper我觉得工程价值最高的部分。17.6T tokens训练，只有一个loss spike。背后是踩了三个坑并填上了。

### 坑一：Muon的数值精度

Muon是个新optimizer，用Newton-Schulz iteration做矩阵正交化近似。paper用了更快收敛的Polar Express（固定6步），但发现在bfloat16下偶发unrecoverable loss spike，且非确定——从附近checkpoint重启就消失。

仿真发现：bfloat16的mantissa只有7位，Polar Express的累积加法在某些update statistics下会产生extreme outlier。修复方法很tricky：只把Polar Express的iteration部分cast到**float16**（mantissa 10位），其他保持mixed-precision。之后spike不再出现。

**intuition**：bfloat16范围大但精度低，float16精度高但范围小。Polar Express的中间值需要精度不需要范围，所以float16更合适。

### 坑二：Expert Collapse（隐蔽版）

以前大家说的"dead expert"是router不往某个expert派token。但Step 3.5 Flash发现一个更隐蔽的版本：**router dispatch看起来正常，但expert的activation在消失，parameter norm在衰减**。

两个原因：
1. Shared expert和routed expert之间没有explicit scaling，大模型不像小模型能自动找平衡，shared expert可能dominate。
2. Micro-batch load balancing太严，fine-grained MoE下反而阻碍specialization。

**诊断建议**：别只看router统计，要监控每个expert的activation RMS norm和parameter Frobenius norm。如果min-to-median ratio在下降，就是expert在"死"。

### 坑三：Activation Blow-up（最有趣，loss完全看不出来）

这是最隐蔽的坑。深层MoE里，极少数expert（每层1-2个）的activation norm在爆炸，median稳定但max飙升。**但loss曲线完全看不出来**——三个mitigation策略的loss曲线一模一样。

为什么loss看不出来？因为pre-norm架构的数学漏洞。

模型的最终输出是所有层输出的和，再过RMSNorm。如果某个outlier expert的输出magnitude $|h_{outlier}| \to \infty$，RMSNorm会把它归一化掉，其他所有层的贡献被淹没。数学上：

$$\text{RMSNorm}(c \cdot \hat{h}_{outlier} + h_{others}) \xrightarrow{c \to \infty} \text{RMSNorm}(\hat{h}_{outlier})$$

这意味着模型可以无限放大这个expert的magnitude，最终归一化后的输出不变，loss不变。但数值上会overflow。

**根因链条**：高频bi-gram → 某个expert专门化 → 输出变deterministic → pre-norm让magnitude direction在loss上不可见 → SwiGLU的gate和up分支高度对齐产生sparse大输出 → Muon正交化低秩梯度持续放大magnitude → 爆炸。

**修复**：weight clipping只延迟爆炸不管用。**activation clipping**（在MoE FFN intermediate做element-wise clip）才有效。

**关键监控指标**：max-to-median ratio of per-expert activation norms。loss看不出来的内部爆炸，这个比率能抓住。

这个发现对所有用pre-norm + MoE的大模型训练都有参考价值。

---

## Post-Training：MIS-PO是核心创新

### 问题：Long-horizon RL的variance爆炸

RL for reasoning的核心痛点：off-policy训练时，你的rollout policy（inference engine）和training policy有差异。传统PPO用importance sampling ratio来校正这个差异，但ratio是连续的，在long trajectory上variance爆炸。

为什么MoE更严重？因为routing的微小变化会让不同expert被激活，distributional shift比dense模型大得多。

### MIS-PO的思路：用binary mask替代continuous weight

PPO的逻辑：每个token的gradient乘以一个importance weight $r_t$，然后clip到 $[1-\epsilon, 1+\epsilon]$。但即使clip了，weight还是有variance。

MIS-PO的逻辑：**超出trust region的样本直接丢掉，留下来的当on-policy处理**。

怎么做？两层filtering：
- **Token level**：ratio $x_t \in [0.5, 2]$ 才保留，否则mask掉这个token的梯度
- **Trajectory level**：geometric mean ratio $\bar{\rho}(\tau) \in [0.996, 1.001]$ 才保留整条trajectory

为什么trajectory bound这么窄？因为geometric mean对长度敏感。10000个token的trajectory，每个token ratio平均偏离1.0005，geometric mean就是 $1.0005^{10000} \approx 148$，完全失控。所以trajectory level必须narrow。

**intuition**：这是hierarchical trust region——token level宽松（允许单token漂移），trajectory level严格（整条trajectory不能偏太远）。牺牲一点样本利用率，换巨大的variance降低。在long-horizon + MoE场景下，这个trade-off非常划算。

实验对比：MIS-PO比PPO的gradient norm显著更稳，比GSPO在MoE上更能控制training-inference divergence。

### 其他RL trick

**Truncation-Aware Value Bootstrapping**：长trajectory被截断时，别给0 reward（把"没做完"当"做错了"），用final state的value estimate替代。truncation rate 20%也能稳。

**Routing Confidence作为stability proxy**：activated experts的平均probability mass $\Sigma_k$。低 $\Sigma_k$ = routing犹豫 = off-policy脆弱。高 $\Sigma_k$ = routing确定 = off-policy robust。这给了一个design principle：MoE RL初期应该先warm up routing confidence再进off-policy。

**Reward系统**：三层架构。verifiable reward用rule-based + model-based verifier；non-verifiable用GenRM（generative reward model，输出confidence score转Bradley-Terry win rate），还加了个MetaRM检测spurious reasoning（正确答案来自错误逻辑）；agent reward用entity-matching + rubric-based LLM judge。

---

## Data：几个关键decision

**Code filtering放松**：OpenCoder的strict hit0-only过滤太严，hit0-6（允许0-6个heuristic violation）最优。quality-diversity trade-off。

**PR/Issue/Commit数据**：从10+ star的GitHub repos构建，衍生4个子集。特别是用Agentless-style模板生成90B tokens的code-editing数据——File localization + Code repair via SEARCH/REPLACE。

**Bidirectional transfer**：在code agent训练中发现一个飞轮效应——construction expertise（构建可执行环境的能力）和coding能力互相加强。建环境的能力上去了，coding能力也跟着上；反过来coding能力上去了，建环境也更准。

**Tool-use数据**：不靠random exploration或model simulation，而是FSM-based decomposition——把tool-use行为分解为atomic intents，用finite state machine建模，sample-execute-verify loop + rejection sampling。所有trajectory在真实环境执行，deterministic feedback验证，消除hallucination。

---

## 效果：11B active打frontier

几个highlight：

- **AIME 2025**: 97.3，PaCoRe后99.9（GPT-5.2 xHigh是100）
- **IMO-AnswerBench**: 85.4，PaCoRe后88.8（GPT-5.2 xHigh是86.3）
- **LiveCodeBench-v6**: 86.4（Gemini 3.0 Pro是90.7）
- **SWE-Bench Verified**: 74.4（GPT-5.2 xHigh是80.0）
- **BrowseComp (w. Ctx Manage)**: 69.0（GPT-5.2 xHigh是65.8，Gemini 3.0 Pro只有59.2）
- **$\tau^2$-Bench**: 88.2（Gemini 3.0 Pro是90.7）
- **RESEARCHRUBRICS**: 65.3（Gemini DeepResearch系统63.69）

在reasoning上接近GPT-5.2 xHigh，在agentic benchmark上部分领先，在code agent上稍逊但差距不大。考虑到只有11B active参数，这个efficiency frontier确实被重新定义了。

---

## 我的几个核心takeaway

**1. Hardware-aware design > FLOPs minimization**。SWA+Head、GQA-8+MTP的协同，都是基于"理解memory bandwidth是bottleneck"的设计。知道你的bottleneck是compute还是memory，比算FLOPs更重要。

**2. Observability stack是大模型工程的隐形护城河**。max-to-median ratio抓activation blow-up、routing confidence抓RL stability、per-expert activation norm抓expert collapse——这些指标在paper里几句话带过，但实际是训练能跑完的关键。

**3. Pre-norm的数学漏洞值得所有大MoE训练者重视**。RMSNorm让magnitude direction在loss上不可见，是activation blow-up的根本原因。loss看不出来不代表没问题。

**4. MIS-PO的hierarchical trust region很优雅**。token level宽 + trajectory level窄，匹配importance ratio在不同granularity上的sensitivity。这个思路可能不止用于RL，任何有long-horizon + distribution shift的场景都值得借鉴。

**5. $\Delta_{tool}$ metric应该被推广**。绝对分数混了memorization和tool use，$\Delta_{tool} = \text{Score}_{with\ tools} - \text{Score}_{no\ tools}$ 才是agentic capability的真信号。一个模型绝对分数高但 $\Delta_{tool}$ 小，可能只是"已经知道答案"而不是"会用工具"。

**6. Bidirectional transfer是self-improving agent的雏形**。construction expertise ↔ coding互相加强，这个飞轮效应可能指向未来agent的自我进化路径。

---

## Web References

- [Step 3.5 Flash Paper (StepFun)](https://github.com/stepfun-ai/Step3.5-Flash)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Kimi K2](https://arxiv.org/abs/2507.20534)
- [Muon Optimizer](https://arxiv.org/abs/2502.20782)
- [Polar Express](https://arxiv.org/abs/2505.13290)
- [Gated Attention as Unit-Centered Affine](https://arxiv.org/abs/2502.01890)
- [StreamingLLM Attention Sinks](https://arxiv.org/abs/2309.17453)
- [Massive Activations in LLMs](https://arxiv.org/abs/2402.17762)
- [GSPO](https://arxiv.org/abs/2507.18071)
- [Stabilizing MoE RL by Aligning Routers](https://arxiv.org/abs/2510.11370)
- [OpenCoder](https://arxiv.org/abs/2411.18488)
- [Agentless](https://arxiv.org/abs/2407.01489)
- [SWE-Bench](https://swebench.com)
- [OpenHands](https://arxiv.org/abs/2407.16741)
- [WebOrganizer](https://arxiv.org/abs/2502.12441)
- [Nemotron-CC](https://arxiv.org/abs/2506.03154)
- [SWE-factory](https://arxiv.org/abs/2506.08467)
- [Metadata Conditioning CoT Pretraining](https://arxiv.org/abs/2502.11446)
- [Better & Faster LLMs via MTP](https://arxiv.org/abs/2404.19737)
- [EAGLE Speculative Decoding](https://arxiv.org/abs/2401.15077)
- [GQA](https://arxiv.org/abs/2305.13245)
- [DeepSeekMoE](https://arxiv.org/abs/2401.12061)
- [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664)
- [Demons in the Detail: Load Balancing for MoE](https://arxiv.org/abs/2501.11873)
- [Tau-Bench](https://github.com/sierra-research/tau2-bench)
- [BrowseComp](https://arxiv.org/abs/2504.07958)
- [Terminal-Bench](https://arxiv.org/abs/2601.11868)
- [LiveCodeBench](https://arxiv.org/abs/2403.07974)
- [DeepSeek-R1](https://arxiv.org/abs/2503.14476)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [Open-Reasoner-Zero](https://arxiv.org/abs/2503.24290)
- [GenRM](https://arxiv.org/abs/2408.15240)
- [SWE-Gym](https://arxiv.org/abs/2412.21139)
- [YaRN Context Extension](https://arxiv.org/abs/2309.00071)
- [Anthropic Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [OpenAI Codex Agent Loop](https://openai.com/index/unrolling-the-codex-agent-loop/)
- [Step-DeepResearch Technical Report](https://arxiv.org/abs/2505.07069)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

---

# Step 3.5 Flash 深度技术解析

Andrej，这篇 paper 信息密度极高，我会按 architecture → stability → post-training → data → evaluation 的逻辑展开，重点 build 你对几个关键 design choice 的 intuition。

---

## 1. 总体定位与设计哲学

Step 3.5 Flash 是一个 **196B total / 11B active** 的 sparse MoE 模型，对标 GPT-5.2 xHigh、Gemini 3.0 Pro、Claude Opus 4.5。核心 thesis：**在 agentic era，latency 是和 intelligence、cost 并列的第三约束**。

Agentic workload 的 profile 很特别：长 context prefill + 长时间多轮 interactive decoding。这驱动了三个互相耦合的设计轴：
- **Attention**：加速 long-context（prefill + decode），且要兼容 speculative decoding
- **Sparse MoE**：避免 EP 下的 straggler 导致吞吐下降
- **MTP**：通过 speculative decoding 加速生成

参数规模刻意压在 200B 以下，为了装进高端工作站的 128GB 内存。

Paper 的 Figure 2 给出整体架构：leading 1 个 Full Attention layer → 11 个 Hybrid Block（每个 = 3 SWA + 1 Full）→ 总共 45 层（3 dense + 42 MoE）。前 3 层用 dense FFN，后面 42 层用 sparse MoE。

参考：[arXiv:2404.19737 (MTP)](https://arxiv.org/abs/2404.19737), [arXiv:2412.19437 (DeepSeek-V3)](https://arxiv.org/abs/2412.19437)

---

## 2. Architecture 核心创新

### 2.1 Hybrid Attention Layout: S3F1

**3:1 SWA:Full 交替**，SWA window $L=512$。直觉：long-context 的大部分信息是局部的，full attention 只用于"长程锚点"。

但 naive S3F1 性能会掉（Table 10：pretrain avg 53.6 vs FFFF 54.1）。两个补救措施：

**(a) Augmented Query Heads in SWA**：把 SWA 的 query head 数从 64 拉到 96。这几乎是 free lunch，因为 SWA 在 long-context 下是 IO-bound，增加 query head 不增加 KV cache 压力。Table 1 显示 S3F1+Head 的 prefill FLOPs 只增加 1.02x，decode 1.01x，但 pretrain avg 拉到 55.7，超过 FFFF。

**(b) Head-wise Gated Attention**：每个 attention head 配一个 input-dependent 的标量 gate。直觉来自 attention sink 现象——naive SWA 窗口里没用信息时，softmax 无法"吸收"unused weight。Sink token 是 data-independent 的补丁；head-wise gate 是 data-dependent 的 sink。

公式（Eq. 4-6）：
$$s_{i,j} = \langle q_i, k_j \rangle / \sqrt{d}, \quad Z_i = \sum_{j'} \exp(s_{i,j'}), \quad \alpha_{i,j} = \frac{\exp(s_{i,j})}{Z_i}$$
$$g_i = \sigma(w_{gate}^\top x_i), \quad o_i^{gate} = g_i y_i$$

变量：
- $q_i, k_j, v_j \in \mathbb{R}^d$：position $i$ 的 query，position $j$ 的 key/value
- $d$：head 维度（这里 = 128）
- $Z_i$：position $i$ 的 logsumexp
- $\alpha_{i,j}$：standard softmax attention weight
- $y_i$：standard attention output
- $w_{gate} \in \mathbb{R}^{d_{model}}$：可学习 gate 向量
- $\sigma(\cdot)$：sigmoid

代入 $\sigma(g) = 1/(1+e^{-g})$：
$$o_i^{gate} = \sum_j \frac{\exp(s_{i,j})}{Z_i + e^{-g_i} Z_i} v_j$$

**关键 intuition**：$e^{-g_i} Z_i$ 扮演 input-dependent sink mass。当 $g_i \to 0$（gate 半开），分母变大，attention 输出被稀释；当 $g_i \to \infty$（gate 全开），退化为标准 attention。这让模型能动态决定"这个 head 在这个 position 是否真的需要 attend 窗口内内容"。

Table 2 在 100B-A10B 上对比：sink token avg 62.5 vs head-wise gate 64.4，全面胜出。

参考：[Gated Attention arXiv:2502.01890](https://arxiv.org/abs/2502.01890), [StreamingLLM arXiv:2309.17453](https://arxiv.org/abs/2309.17453), [Massive Activations arXiv:2402.17762](https://arxiv.org/abs/2402.17762)

### 2.2 GQA-8 与 speculative decoding 的协同

8 个 KV head（GQA-8）不是随便选的——是为了对齐 8-way tensor parallelism，让 KV cache sharding 和 TP 切分对齐，改善内存访问模式。

但更深一层的直觉：**GQA-8 让 attention 更 memory-bandwidth bound，反而创造了 speculative decoding 的 slack**。draft + verify 的 overhead 可以被这个 slack 吸收，所以可以激进的 MTP-3 speculation 而不付成比例的 latency 代价。

Table 7 验证：SWA head 64→96 在 64k decode 下 latency 只增加 1.01x，head-wise gate 几乎 0 overhead。

### 2.3 Fine-grained MoE 与 EP-Group Balancing

每层 288 routed experts + 1 shared，top-k=8。Expert Parallelism 下，**straggler 是吞吐杀手**——token assignment skew 让少数 expert 和它们所在的 GPU 过载，sync point 被卡住。

Paper 引入 EP-level balancing loss（Eq. 1）：
$$p_e = \frac{1}{T} \sum_{t=1}^{T} p_{t,e}, \quad f_e = \frac{1}{TK} \sum_{t=1}^{T} s_{t,e}$$
$$p_g = \sum_{e \in \mathcal{E}_g} p_e, \quad f_g = \sum_{e \in \mathcal{E}_g} f_e, \quad \mathcal{L}_{EP} = G \sum_{g=1}^{G} f_g p_g$$

变量：
- $T$：micro-batch 中 token 数
- $K$：top-k（=8）
- $p_{t,e}$：token $t$ 路由到 expert $e$ 的概率
- $s_{t,e} \in \{0,1\}$：token $t$ 是否实际路由到 expert $e$（在 top-k 集合内）
- $\mathcal{E}_g$：EP group $g$（即一张 GPU）上的 expert 集合
- $G$：EP group 数（=GPU 数）
- $p_e$：expert $e$ 的平均路由概率
- $f_e$：expert $e$ 的平均路由 frequency
- $p_g, f_g$：group 级别的聚合
- $\mathcal{L}_{EP}$：EP balancing loss

直觉：当所有 group 均匀时 $f_g = p_g = 1/G$，loss = $G \cdot G \cdot (1/G^2) = 1$。某个 group 过载时 $f_g \cdot p_g > 1/G^2$，loss 上升。注意这是和 standard loss-free balancing（DeepSeek-V3 那种）**叠加**的——standard 保证 expert 级平衡，EP loss 保证 rank 级平衡。

### 2.4 MTP-3 设计

三个 MTP head，每个 = SWA + dense FFN，总共只加 0.81B 参数（0.41%）。预测 offset $h \in \{1,2,3\}$，MTP-$h$ 预测 $x_{t+1+h}$。

**关键工程优化**：主训练只激活 MTP-1（省训练开销）。backbone 训练好后，把 MTP-2、MTP-3 从 MTP-1 clone 出来，再轻量 joint fine-tune。这避免训练时三个 head 都吃显存。

还用了 Fast-MTP 的 position-dependent loss reweighting，防止远端 token prediction 被过拟合——直觉是越远的 token 越难，权重应该衰减。

参考：[FastMTP arXiv:2502.15820](https://arxiv.org/abs/2502.15820), [EAGLE arXiv:2401.15077](https://arxiv.org/abs/2401.15077)

### 2.5 Meta Token

每个训练样本前置一个 human-readable metadata string $M$（content type, language, domain, source）。前 3.8T tokens 全部预测 $\mathcal{L}_{full} = -\sum \log P(s_t | s_{<t})$；之后 mask 掉 $M$ 的位置，只预测 payload：
$$\mathcal{L}_{mask} = -\sum_{t=|\mathbf{M}|+1}^{|\mathbf{s}|} \log P(s_t | \mathbf{s}_{<t}) = -\sum_{t=1}^{|\mathbf{x}|} \log P(x_t | \mathbf{M}, \mathbf{x}_{<t})$$

Intuition：metadata 充当 conditioning signal，早期让模型学会利用它，后期纯当 context 不再算 loss，把优化压力全部给 payload。

参考：[Metadata Conditioning arXiv:2502.11446](https://arxiv.org/abs/2502.11446)

---

## 3. 训练稳定性：三个 failure mode 的深度分析

这是这篇 paper 我觉得最有 engineering 价值的部分。17.6T tokens 只有**一个** loss spike，靠的是一套细粒度 observability stack。

### 3.1 Muon 的数值敏感性

Muon 通过 Newton-Schulz iteration 近似 semi-orthogonal update。Paper 用 **Polar Express** iteration（固定 6 步），比标准 NS 收敛快。

但发现一个诡异现象：偶发 unrecoverable loss spike，且非确定（从附近 checkpoint 重启就消失）。仿真定位到 bfloat16 下 Polar Express 在某些 update statistics 下会产生 extreme intermediate outlier，是 cumulative addition 误差。

**修复**：只把 Polar Express iteration 的 state 和 intermediates cast 到 **float16**（不是 bfloat16），其他保持 mixed-precision。之后 spikes 不再出现。

直觉：bfloat16 有更大的 exponent range 但更少的 mantissa bits（7 vs 10），累积加法误差大；float16 mantissa 多但范围小，对 Polar Express 的中间值更稳。

参考：[Muon arXiv:2502.20782](https://arxiv.org/abs/2502.20782), [Polar Express arXiv:2505.13290](https://arxiv.org/abs/2505.13290)

### 3.2 Expert Collapse Beyond Routing Collapse

之前 Step-3 paper 报过 "dead experts"。但这里揭示一个更隐蔽的 failure mode：**router dispatch 统计看起来健康，但 expert 侧 activation vanishing + parameter norm 衰减**。

两个原因：
- **(i) Shared expert 和 routed expert 之间没有 explicit scaling**：大模型不能 implicit self-calibrate，shared expert 可能 dominate，routed expert 的有效贡献被压制（即使 routing 频率正常）。
- **(ii) Micro-batch LBL 太严**：fine-grained sparsity 下，Switch-style micro-batch 约束会引发 cross-expert 过度竞争，阻碍 specialization。

**诊断建议**：别只看 router dispatch，要监控：
- Per-expert activation RMS norm（在 MoE FFN intermediate 处）
- Expert parameter Frobenius norm
- Min-to-median ratio 的下降趋势

参考：[Demons in the Detail arXiv:2501.11873](https://arxiv.org/abs/2501.11873)

### 3.3 Localized Activation Blow-up（附录 B 是精华）

这是最有趣的部分。深层 MoE 中少数 expert（每层 1-2 个）activation norm 爆炸，median 稳定但 max 飙升，**但 loss 完全看不出来**（Figure 4a）。

**根因分析**：

**(1) Bi-gram shortcut**：高频 bi-gram 触发某个 expert 专门化。一旦激活，输出变 deterministic，其他网络不再影响预测。

**(2) Pre-norm 架构的 pathological solution**：最终表示
$$h_{final} = \text{RMSNorm}(\underbrace{expert_{outlier}}_{h_{outlier}} + \underbrace{\sum_l attn_l + \sum_{l,e} expert_{l,e}}_{h_{others}})$$

当 $|h_{outlier}| \to \infty$：
$$\text{RMSNorm}(h_{final}) = \lim_{c \to \infty} \text{RMSNorm}(c \hat{h}_{outlier} + h_{others}) = \text{RMSNorm}(\hat{h}_{outlier})$$

直觉：RMSNorm 把 outlier 的 magnitude 归一化掉，**其他所有层的贡献被淹没**。所以一旦 shortcut 成立，模型可以无限放大这个 expert 的输出 magnitude 而不改变最终归一化结果——这是 pre-norm 的一个数学漏洞。

**(3) SwiGLU 的 alignment 陷阱**：
$$\text{SwiGLU}(x) = W_{down}(\text{SiLU}(W_{gate} x) \cdot W_{up} x)$$

观察 outlier expert：$\|\text{SiLU}(W_{gate} x)\| \cdot \|W_{up} x\| \approx \|\text{SiLU}(W_{gate} x) \cdot W_{up} x\|$。这要求 gate 和 up 分支**高度对齐且集中在少数维度**——元素积范数等于范数积，意味着 sparse + aligned 的极端情况。只有 $W_{up}$ 的少数行被利用。

**(4) Muon 放大效应**：Muon 更新 $\Delta W = \sum_i \sigma_i u_i v_i^\top$。对 outlier expert，梯度是 abnormally low-rank（rank $r$），且**持续指向同一方向**（强调 magnitude 不旋转）。Muon 完全消除 gradient magnitude 影响，aggressively orthogonalize，低秩 singular value $\sigma_i$ 快速增大。Adam 的 $\epsilon$ 还能当 threshold 滤掉小梯度，Muon 不会。

**两种 mitigation**：
- **Weight clipping**：$W \leftarrow W \cdot \tau / \max_x \|Wx\|$，offline 在 checkpoint 上做。**只延迟爆炸，不能阻止**。
- **Activation clipping**：在 MoE FFN intermediate 做 element-wise clip。**有效**。

Paper 把 **max-to-median ratio of per-expert activation norms** 确立为必要监控指标。Loss 看不出来的内部爆炸，这个比率能抓住。

参考：[MuonClip in Kimi K2](https://arxiv.org/abs/2507.20534), [GLU Variants](https://arxiv.org/abs/2002.05202)

---

## 4. Post-Training：MIS-PO 是核心

### 4.1 整体框架

两阶段：
1. **Expert Model Construction**：基于统一 SFT baseline，在 Math / Code / STEM / Tool-use / Long Context / Human Preference / Agentic Reasoning 各 domain 做 domain-specific RL，得到多个 expert。
2. **Self-Distillation**：让 expert 模型在共享 prompt 分布上生成 trajectory，rejection sampling 滤掉 language mixing / overthinking，蒸馏回单一 student model。

SFT 两阶段：
- Stage 1：大规模 multi-domain SFT
- Stage 2：注入 OOD 信号（30k expert-level chemistry trajectories + synthetic arithmetic），仅 3 epochs 解锁 latent capability

### 4.2 MIS-PO：用 discrete filtering 替代 continuous importance weighting

这是 paper 的算法核心。RL for long-horizon reasoning 的根本痛点：**off-policy importance sampling 在长 horizon 下 variance 爆炸**。token-level 概率小漂移在 trajectory 上累积成 noisy gradient。MoE 因为 routing shift 更严重。

**MIS-PO 的核心 idea**：把 importance ratio 的 continuous weighting 换成 binary masking。

Actor loss（Eq. 2）：
$$\mathcal{L}_{actor} = -\mathbb{E}_{\tau \sim \pi_{\theta_{vllm}}} \left[ \mathbb{I}(x_t) \cdot \mathbb{I}(\bar{\rho}(\tau)) \cdot \log \pi_\theta(a_t | s_t) \cdot \hat{A}_t \right]$$

变量：
- $\tau = (s_0, a_0, ..., s_T)$：trajectory
- $a_t$：step $t$ 生成的 token
- $s_t$：step $t$ 的 state（context）
- $\pi_{\theta_{vllm}}$：inference engine（vLLM）的 policy，rollout 用
- $\pi_{\theta_{old}}$：training snapshot 的 pre-update policy
- $\pi_\theta$：当前正在更新的 policy
- $x_t = \pi_{\theta_{old}}(a_t|s_t) / \pi_{\theta_{vllm}}(a_t|s_t)$：token-level importance ratio
- $\bar{\rho}(\tau) = (\prod_t x_t)^{1/T}$：trajectory-level geometric mean ratio
- $\mathbb{I}(x) = \mathbb{1}[\rho_{min} \le x \le \rho_{max}]$：binary indicator
- token level bound: $[\rho_{min}, \rho_{max}] = [0.5, 2]$
- trajectory level bound: $[0.996, 1.001]$
- $\hat{A}_t$：advantage estimate

**Intuition**：
- PPO 用 $\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t)$，continuous ratio 即使 clip 了也带 noise
- GSPO 用 geometric mean 替代 token-level ratio，但仍是 continuous
- **MIS-PO 直接把超出 trust region 的样本 mask 掉**，留下的当成 on-policy 处理

灵感来自 Metropolis Independence Sampling——proposal distribution 和 target distribution 接近时，accept/reject 是 binary 的，不需要 importance weight。

**为什么 trajectory-level bound 这么窄 $[0.996, 1.001]$**？因为 geometric mean 在长 trajectory 上对单 token 漂移很敏感。$T=10000$ tokens，单 token ratio 平均偏离 1.0005，geometric mean 就是 $1.0005^{10000} \approx e^5 \approx 148$。所以 trajectory level 必须 narrow。

Figure 5 的 ablation：MIS-PO 比 PPO 在 ~5000 步内 gradient norm 显著更稳，reward plateau 更高，entropy 衰减更慢（exploration 更持久）。

Figure 7 对比 GSPO：dense 模型上 MIS-PO 更高效；MoE 模型上 GSPO 训练-推理 divergence 失控，MIS-PO 稳定。

参考：[Metropolis-Hastings original](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm), [GSPO arXiv:2507.18071](https://arxiv.org/abs/2507.18071), [Your Efficient RL Framework arXiv:2508.02000](https://arxiv.org/abs/2508.02000)

### 4.3 Truncation-Aware Value Bootstrapping

长 trajectory 被截断时给 0 reward 是错的——把"没做完"和"做错了"混为一谈。Paper 用 final state 的 value estimate 替代：
$$\hat{R}_i = \begin{cases} V_\phi(s_T) & \text{if truncated} \\ R_i & \text{otherwise} \end{cases}$$

变量：$V_\phi$ 是 critic，$s_T$ 是截断时的 final state。

实验上 truncation rate 20% 也能稳。对 competition-level benchmark（长 reasoning 多）特别有效。

参考：[Time Limits in RL arXiv:1712.00378](https://arxiv.org/abs/1712.00378), [DeepScaler](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)

### 4.4 Routing Confidence 作为 stability proxy

$\Sigma_k$ = activated experts 的平均 probability mass。低 $\Sigma_k$ = routing 不确定 = training-inference mismatch 放大。

**Phase transition**：低 routing confidence 的模型脆弱，需要 Router Replay / 严格 on-policy；高 confidence 模型 robust，可以 off-policy 不加复杂干预。

参考：[Stabilizing MoE RL arXiv:2510.11370](https://arxiv.org/abs/2510.11370)

### 4.5 Reward System

三层：
- **Verifiable reward**：rule-based checker（logic, instruction following, code）+ model-based verifier（STEM 用 gpt-oss-120b，比 vanilla math-verify 高 2.0%）
- **Non-verifiable reward**：GenRM（pairwise generative RM，输出 confidence score → Bradley-Terry win rate），内置 length control penalty。**MetaRM** 检测 spurious reasoning（正确 preference 来自错误逻辑），降低 reward。MetaRM 加持比 vanilla GenRM 每 benchmark 高 0.5-3%。
- **Agent reward**：search 用 entity-matching，report generation 用 rubric-based LLM judge 输出 ternary，但 intermediate category 不靠谱，映射成 asymmetric binary。

GenRM 用 log-sigmoid loss 训练，从 SFT model 初始化。

参考：[GenRM arXiv:2408.15240](https://arxiv.org/abs/2408.15240), [Bradley-Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)

### 4.6 RL 超参细节

- Rollout: temperature=1.0, top-p=1.0, max seq 128k
- Reasoning: 256 prompts × 16 responses
- Human preference: 512 prompts × 8 responses
- Tool-use: 128 prompts × 8 responses
- Actor: 4 mini-batches, LR $2 \times 10^{-6}$, 20 warmup
- Critic: 12 mini-batches, LR $5 \times 10^{-6}$, 50 warmup
- $\alpha = \beta = 1$（ORZ 风格）
- Unbiased KL loss coef 0.001（final stage）
- Muon optimizer, weight decay 0.1

---

## 5. Data Pipeline

### 5.1 StepCrawl

自建爬虫，**WebOrganizer-style model** 做 site/URL selection。LM-in-the-loop 反馈循环：(i) 滤掉 SEO-driven 低质页，(ii) 平衡 crawl budget 防止 e-commerce/tool 站点过多。约 1B pages/day。

质量分层：六个轻量 scorer 集成打分，保留 High/Medium-High/Medium，丢 Medium-Low/Low。Embedding-based cluster rebalancing：对中英 web 数据 embedding + k-means（100k+ clusters），down-sample 过大 cluster。

参考：[WebOrganizer arXiv:2502.12441](https://arxiv.org/abs/2502.12441), [Nemotron-CC arXiv:2506.03154](https://arxiv.org/abs/2506.03154)

### 5.2 Code Data

修改版 OpenCoder pipeline。关键发现：strict hit0-only 过度剪枝，no filter 噪声过多，**hit0-6 配置最优**（允许 0-6 个 heuristic violation）。

PR/Issue/Commit Data：10+ star GitHub repos，5M 样本基础。衍生 4 个子集：
1. Base PR/Issue/Commit：GHArchive + GitHub API，20+ 主流语言，对 SWE-Bench 严格去重
2. Concatenated PR-Dialogue（90B tokens）：Agentless-style 模板，File localization + Code repair via SEARCH/REPLACE
3. Rewritten Reasoning-Oriented（12B tokens）：LLM 重构 PR 作者 problem-solving process + Active Reading notebooks
4. Environment-based Seed：从 PR/issue/commit 构建可执行环境，hundreds of thousands of seeds

参考：[OpenCoder arXiv:2411.18488](https://arxiv.org/abs/2411.18488), [Agentless arXiv:2407.01489](https://arxiv.org/abs/2407.01489), [SWE-Bench](https://www.swebench.com/)

### 5.3 Math & STEM

StepCrawl 之外另抓 math 数据，MegaMath-inspired pipeline + FineMath + 内部 classifier 集成。100M 教育样本（K-12 到 CPA/Legal 职业考试）。

参考：[MegaMath arXiv:2507.16428](https://arxiv.org/abs/2507.16428), [FineMath from SmolLM2](https://arxiv.org/abs/2502.00479)

### 5.4 Tool-Use 数据生成

不靠 random exploration 或 model simulation，而是 **FSM-based decomposition**：把 tool-use 行为分解为 atomic intents，用 finite state machine 建模，sample-execute-verify loop + rejection sampling。所有 candidate trajectory 在真实环境执行，deterministic feedback 验证。100k+ 高质量 trajectory。

### 5.5 Code Agent Pipeline

SWE-factory 演化版，**cross-task memory pool** 检索历史 build success 当 few-shot demo，**loop-detection** 防冗余探索。40% environment-building 成功率。50k verified environments 跨 15k+ GitHub repos，20+ 语言。

Bidirectional transfer：construction expertise 加速 coding，coding within environments 反过来提升 construction accuracy。

参考：[SWE-factory arXiv:2506.08467](https://arxiv.org/abs/2506.08467), [DockSmith arXiv:2506.08467](https://arxiv.org/abs/2506.08467), [SWE-smith arXiv:2504.21798](https://arxiv.org/abs/2504.21798), [SWE-Gym arXiv:2412.21139](https://arxiv.org/abs/2412.21139)

### 5.6 Search Agent

Knowledge graph（Wikidata5m）topological expansion + 模拟跨网站 browsing trajectory。**关键**：用 DeepSeek-R1 验证 query 是否真的需要外部检索，能被 R1 直接解出来的就剔除。生成 trajectory 经 structured report generation pipeline 清洗。

---

## 6. Agent Infrastructure

### 6.1 Reasoning + Tool-Use 模板设计

三种 reasoning history 管理策略对比：
- 每轮丢弃 reasoning history（DeepSeek-R1 风格）：长 horizon 失败
- 保留全部 reasoning history：context 爆炸
- **Selective retention**：只保留最近 user instruction 触发的 tool-use trajectory 的 reasoning。最优。

Tool-use 格式：XML 而非 JSON。JSON 的 escape 和 delimiter 在 under-trained 小模型上 parse 错误率高，XML 允许 flat string output，grammatical overhead 低。

### 6.2 Scalable Code Agent

Session-Router via Kubernetes + Tmux，支持数千并发环境，state persistence。训练时暴露于多种 framework：OpenHands、SWE-agent、Terminus-2、Kilocode、Roocode、ClaudeCode——防 overfit 到特定 scaffold。

参考：[OpenHands arXiv:2407.16741](https://arxiv.org/abs/2407.16741), [SWE-agent](https://swe-agent.com/)

---

## 7. Pre-Training Curriculum

| Stage | Tokens | Context | 数据侧重 |
|---|---|---|---|
| Pretrain Stage 1 | 14.6T | 4k | broad open-domain |
| Pretrain Stage 2 (anneal) | 2T @ 4k + 1T @ 32k | 4k→32k | code + PR/Issue/Commit + 高质知识 |
| Mid-train Stage 1 | 386B | 32k | 21% replay from pretrain + SWE + tool-use |
| Mid-train Stage 2 | 364B | 128k | 10.5B replay + 长 horizon reasoning + 自然长文档 |
| **Total** | ~17.6T + 750B | | |

超参细节：
- Muon optimizer, weight decay 0.1, grad clip 1.0
- LR: warmup to $2.5 \times 10^{-4}$ over 2000 steps → cosine to $5 \times 10^{-5}$ (Stage 1) → $2 \times 10^{-5}$ (Stage 2 4k part) → fixed $2 \times 10^{-5}$ (32k part)
- Batch: 4096 → 16384 over first 400B tokens, then 16384; 32k annealing batch=2k
- MTP loss weight: 0.3 (Stage 1) → 0.1 (Stage 2)
- Loss-free balancing bias update rate: 0.001 → 0 (anneal)
- EP-group balance loss coef: 0.001 throughout pretrain
- RoPE $\theta$: 10000 (4k both) → $\theta_{Full} = 10^6$, $\theta_{SWA} = 10^4$ (32k anneal) → $\theta_{Full} = 5 \times 10^6$, $\theta_{SWA} = 10^4$ (128k mid-train)

**Selective RoPE scaling**：只对 Full Attention 层增大 $\theta$，SWA 保持小 $\theta$。直觉：SWA 是局部的，不需要长程位置编码；Full Attention 承担长程连接。

---

## 8. 评测亮点

### 8.1 Pretrain Base 对比

Table 4：Step 3.5 Flash Base 在 SimpleQA 上 31.6，超 DeepSeek-V3.2-Exp Base 27.0，**只用 1/3.4 总参数**。MMLU 85.8，BBH 88.2，HumanEval 81.1。

### 8.2 Post-train 对比（Table 5）

| Benchmark | Step 3.5 Flash | GPT-5.2 xHigh | Gemini 3.0 Pro |
|---|---|---|---|
| AIME 2025 | 97.3 / 99.9 (PaCoRe) | 100.0 | 95.0 |
| HMMT 2025 Feb | 98.4 / 100.0 | 99.4 | 97.5 |
| IMO-AnswerBench | 85.4 / 88.8 | 86.3 | 83.3 |
| LiveCodeBench-v6 | 86.4 / 88.9 | 87.7 | 90.7 |
| SWE Verified | 74.4 | 80.0 | 76.8 |
| Terminal-Bench 2.0 | 51.0 | 54.0 | 56.9 |
| BrowseComp (w. Ctx Manage) | 69.0 | 65.8 | 59.2 |
| $\tau^2$-Bench | 88.2 | 85.5 | 90.7 |
| GAIA | 84.5 | 83.5 | 76.6 |
| RESEARCHRUBRICS | 65.3 | 57.8 | 50.1 |

在 11B active 参数下，**全面接近 frontier 模型**，部分 benchmark（BrowseComp w/ Ctx Manage、RESEARCHRUBRICS、$\tau^2$-Bench、GAIA）甚至领先。

### 8.3 Tool Use Gain（Table 11）

Paper 引入 $\Delta_{tool} = \text{Score}_{with\ tools} - \text{Score}_{no\ tools}$，区分"模型已经知道"和"模型会用工具找答案"。Step 3.5 Flash 平均 gain 52.0，第一，在 GAIA / xbench-DeepSearch 上领先。

### 8.4 PaCoRe 测试时 scaling

Parallel Coordinated Reasoning，$\vec{K} = [4, 4, 4, 4]$ 配置。Table 5 显示 PaCoRe 在 reasoning benchmark 上显著加成：AIME 2025 97.3→99.9，HMMT 98.4→100.0，IMO-AB 85.4→88.8。

Tool-integrated PaCoRe（Table 13）也 work：GPQA-Diamond 84.4→85.7，HLE_text 26.5→28.2。

参考：[PaCoRe arXiv:2602.xxxxx](https://arxiv.org/abs/2602.07856)

### 8.5 Context Management 策略对比（Table 17）

在 BrowseComp 200 实例子集上：
- Baseline (no management): 49.5%, 86 steps
- Summary: 57.0%, 131 steps
- Keep-first&last-K: 58.0%, 244 steps
- **Discard-all**: 66.0%, 302 steps
- Multi-Agent: 68.5%, 721 steps

**直觉**：Discard-all 等价于 test-time pass@K，强制模型从头 re-reason 直到 self-verify 通过。性能随 step 数增加——context management 把 compute 转化成 performance。

### 8.6 Internal Benchmark

Data Analysis Benchmark：Step 3.5 Flash 39.6，第二（仅次于 Claude Opus 4.5 的 45.0），超 GPT-5.2 39.3、Gemini 3.0 Pro 33.6。

Consulting & Recommendations Benchmark：Step 3.5 Flash 70.5，第四，和 Gemini 3.0 Pro 70.6 持平，但 cost/latency 低很多。

---

## 9. 几个值得深挖的 Intuition

### 9.1 为什么 S3F1+Head 是 free lunch

SWA 在 long-context 下是 IO-bound（query-to-KV ratio 高，compute 空跑），加 query head 不增 KV cache 压力，所以 latency 几乎不变。Full Attention 加 head 才贵。这是 hardware-aware 架构设计的典范——**理解 bottleneck 是 compute 还是 memory 比算 FLOPs 更重要**。

### 9.2 Head-wise Gate vs Sink Token 的本质区别

Sink token 是在 softmax 分母加一个固定的 $e^{s_{sink}}$，所有 position 都一样。Head-wise gate 是 $e^{-g_i} Z_i$，**与 $Z_i$ 成比例**——意味着 sink mass 随当前 attention 分布的 entropy 自适应。Attention 已经很 confident 时 $Z_i$ 大，gate 关闭时抑制更强；attention 分散时 $Z_i$ 小，gate 影响弱。这是一种 **entropy-aware gating**。

### 9.3 MIS-PO vs PPO 的本质区别

PPO 用 continuous importance weight 即使 clip 了也是 weighted average，gradient 是 $\sum_t w_t \nabla \log \pi \cdot A_t$，权重 $w_t$ 的 variance 直接进 gradient。

MIS-PO 把超出 trust region 的样本**整条丢掉**，gradient 只来自 in-region 样本，等价于 truncated importance sampling 但用 indicator 而非 weight。**牺牲一点样本利用率换巨大 variance 降低**。在 long-horizon + MoE routing shift 下这个 trade-off 划算。

trajectory bound $[0.996, 1.001]$ 的直觉：geometric mean 对 $T$ 长度敏感，必须 narrow。但 token bound $[0.5, 2]$ 较宽，允许单 token 漂移。这是 hierarchical trust region——**coarse level 严格，fine level 宽松**。

### 9.4 Activation Blow-up 为什么 loss 看不出来

因为 pre-norm + RMSNorm 的归一化让 outlier magnitude 不影响 final output direction（Eq. 10）。这是 pre-norm 的一个数学漏洞——loss landscape 完全 flat along magnitude direction，optimizer 可以无限走而 loss 不变。但这个 direction 是数值不稳定的，下游会 overflow。

**Post-norm 不会有这个问题**，但 post-norm 训练不稳定。所以 pre-norm + activation clipping 是工程上的折衷。

### 9.5 Routing Confidence 作为 phase transition

低 $\Sigma_k$ 意味着 router 在多个 expert 间犹豫，小 perturbation 就会切换 routing，导致 training policy 和 inference policy 在 expert 选择上分歧。高 $\Sigma_k$ 意味着 routing 决定，perturbation 不影响。所以 routing confidence 高的模型可以放心 off-policy，低的必须 on-policy 或 Router Replay。

这给了一个 design principle：**MoE RL 的初期应该 warm up routing confidence**（比如先 SFT 一段固化 router），再进 off-policy RL。

---

## 10. 限制与未竟之志

- **Token efficiency 差 Gemini 3.0 Pro**：相同质量需要更长 trajectory。下一步做 thinking pruning/compression。
- **Universal mastery**：想用 on-policy distillation 把 expert 行为内化得更 sample efficient。
- **Open-world agentic RL**：学术 benchmark 强，但专业工程 / 科研级别 agent 还需 RL 进阶。
- **Distribution shift 下不稳**：高度专业 domain 或超长多轮对话会出现 repetitive reasoning、mixed-language、时间/身份意识不一致。

---

## 11. 我的几个 takeaways

1. **Hardware-aware design > FLOPs minimization**：SWA+Head、GQA-8+MTP 的协同都是基于"理解 memory bandwidth 是 bottleneck"的设计，比单纯减 FLOPs 更高维。

2. **Observability stack 是大模型工程的隐形护城河**：max-to-median ratio 抓 activation blow-up、routing confidence 抓 RL stability，这些都是 paper 里几句话带过但实际救命的东西。

3. **Pre-norm 的数学漏洞值得重视**：RMSNorm 让 magnitude direction 在 loss 上不可见，是 activation blow-up 的根本原因。所有用 pre-norm 的大 MoE 都该监控 max-to-median ratio。

4. **MIS-PO 的 hierarchical trust region 很优雅**：token level 宽 + trajectory level 窄，匹配 importance ratio 在不同 granularity 上的 sensitivity。

5. **Agentic benchmark 的 $\Delta_{tool}$ metric 应该被推广**：绝对分数混了 memorization 和 tool use，$\Delta_{tool}$ 才是 agentic capability 的真信号。

6. **Bidirectional transfer in code agent**：construction expertise ↔ coding 互相加强，这个飞轮效应是 self-improving agent 的雏形。

---

## Web References 汇总

- [Step 3.5 Flash (paper PDF)](https://arxiv.org/abs/2502.20782) - 注意这个 arXiv id 是占位的，实际 paper 在 StepFun 官网
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Kimi K2](https://arxiv.org/abs/2507.20534)
- [GLM-4.5](https://arxiv.org/abs/2508.06471)
- [Muon Optimizer](https://arxiv.org/abs/2502.20782)
- [Polar Express](https://arxiv.org/abs/2505.13290)
- [Gated Attention as Unit-Centered Affine](https://arxiv.org/abs/2502.01890)
- [StreamingLLM Attention Sinks](https://arxiv.org/abs/2309.17453)
- [Massive Activations in LLMs](https://arxiv.org/abs/2402.17762)
- [GSPO](https://arxiv.org/abs/2507.18071)
- [Your Efficient RL Framework Secretly Brings Off-Policy Training](https://arxiv.org/abs/2508.02000)
- [Stabilizing MoE RL by Aligning Routers](https://arxiv.org/abs/2510.11370)
- [OpenCoder](https://arxiv.org/abs/2411.18488)
- [Agentless](https://arxiv.org/abs/2407.01489)
- [SWE-Bench](https://swebench.com)
- [OpenHands](https://arxiv.org/abs/2407.16741)
- [DeepSeek-R1](https://arxiv.org/abs/2503.14476)
- [WebOrganizer](https://arxiv.org/abs/2502.12441)
- [Nemotron-CC](https://arxiv.org/abs/2506.03154)
- [SWE-factory](https://arxiv.org/abs/2506.08467)
- [Metadata Conditioning CoT Pretraining](https://arxiv.org/abs/2502.11446)
- [Tau-Bench](https://github.com/sierra-research/tau2-bench)
- [BrowseComp](https://arxiv.org/abs/2504.07958)
- [Terminal-Bench](https://arxiv.org/abs/2601.11868)
- [LiveCodeBench](https://arxiv.org/abs/2403.07974)
- [LongBench v2](https://arxiv.org/abs/2412.15204)
- [RULER](https://arxiv.org/abs/2404.06654)
- [HELMET](https://arxiv.org/abs/2410.02684)
- [EAGLE Speculative Decoding](https://arxiv.org/abs/2401.15077)
- [Better & Faster LLMs via MTP](https://arxiv.org/abs/2404.19737)
- [GQA](https://arxiv.org/abs/2305.13245)
- [DeepSeekMoE](https://arxiv.org/abs/2401.12061)
- [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664)
- [Demons in the Detail: Load Balancing for MoE](https://arxiv.org/abs/2501.11873)
- [MegaMath](https://arxiv.org/abs/2507.16428)
- [OpenHands Platform](https://github.com/All-Hands-AI/OpenHands)
- [SWE-agent](https://arxiv.org/abs/2405.15793)
- [SWE-Gym](https://arxiv.org/abs/2412.21139)
- [R2E-Gym](https://arxiv.org/abs/2504.07164)
- [DeepScaler](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [ORZ (Open-Reasoner-Zero)](https://arxiv.org/abs/2503.24290)
- [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393)
- [GenRM](https://arxiv.org/abs/2408.15240)
- [FRAMES Benchmark](https://arxiv.org/abs/2502.02839)
- [YaRN Context Extension](https://arxiv.org/abs/2309.00071)
- [MMLU-Pro](https://arxiv.org/abs/2406.01574)
- [GPQA](https://arxiv.org/abs/2311.12022)
- [SuperGPQA](https://arxiv.org/abs/2502.14739)
- [SimpleQA](https://openai.com/index/introducing-simpleqa/)
- [IMO-AnswerBench](https://arxiv.org/abs/2506.19881)
- [StepFun Step-3 Prior Work](https://arxiv.org/abs/2410.00692)
- [Anthropic Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [OpenAI Codex Agent Loop](https://openai.com/index/unrolling-the-codex-agent-loop/)
- [Step-GUI Technical Report](https://arxiv.org/abs/2512.15431)
- [Step-DeepResearch Technical Report](https://arxiv.org/abs/2505.07069)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

这篇 paper 真正的工程价值在于那三个 stability failure mode 的诊断与 mitigation，以及 MIS-PO 的 hierarchical trust region 设计——这些是把 frontier 模型从 "works in paper" 推到 "works in 17.6T token training" 的隐形 know-how。Architecture 部分相对 incremental，但 hardware-aware 的 design rationale 值得反复揣摩。
