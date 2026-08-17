---
source_pdf: GLM-5fromVibeCodingtoAgenticEngineering.pdf
paper_sha256: 4a97eab759d4576b8a93cac0cdf8e25b8fc4c01d71c61f759548eab6796e6ac7
processed_at: '2026-08-04T21:48:37-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 GLM-5

好，我把这篇 paper 翻译成"人话"——保留技术骨架，但讲清楚每个设计背后的 **为什么**。

---

## 先说这篇 paper 想解决什么问题

Karpathy 你自己造了个词叫 **vibe coding**——人在 prompt 里写"帮我做个 todo app"，模型吐代码，人看着觉得对了就用，不对就再 prompt 一轮。这模式能用，但天花板很低，因为**心智负担还在人身上**：要人去想架构、去 debug、去 decide 下一步做什么。

GLM-5 想干的事是把这副心智担子也交给模型。模型自己 plan、自己 implement、自己 iterate，跑几个小时都不需要人介入。Paper 管这叫 **agentic engineering**。

这听起来就难多了。Vibe coding 只要模型单次输出对，agentic engineering 要模型在**几百轮 tool call、几万 token 的 trajectory 里保持不崩**。这带来了三个工程难题，整篇 paper 就是在解这三道题：

1. **长上下文怎么不退化**——200K context 下 attention 计算爆炸，精度还掉
2. **长 horizon RL 怎么训**——agent rollout 跨小时，同步 RL 会把 GPU 晾在那等最慢的那个
3. **环境怎么造**——要有上万个可验证的 SWE/terminal/search 环境让 agent 在里面练

---

## 第一道题：架构怎么改才能又大又快又准

### 先看 GLM-5 多大

744B 总参数，40B 激活参数。对比 GLM-4.5 的 355B/32B，**规模翻倍**。256 个 expert，75 层 MoE + 3 层 dense。

为什么层数从 92 减到 78？因为 MoE 训练时层与层之间要 all-reduce 同步，这是串行的。层数越多，串行链越长，通信开销越大。加 expert 数不影响这个串行链，只影响单层 router 的 fan-out。所以**加宽比加深在 MoE 里更划算**。

DeepSeek-V3 也是 256 experts，思路一致：https://arxiv.org/abs/2412.19437

### MLA 和 Muon 的那点事

MLA 是 DeepSeek-V2 提出的 attention 变体。普通 attention 每层要存 K 和 V 两个矩阵，context 一长 KV cache 就爆。MLA 的 idea 是把 K、V 先压到一个低维的 latent vector $c_t$ 里：

$$c_t = W^{KV} h_t \in \mathbb{R}^{512}$$

然后需要用时再 up-project 回去：

$$k_t = W^{UK} c_t, \quad v_t = W^{UV} c_t$$

KV cache 只存 $c_t$（512 维），不存 $k_t, v_t$（2048 维），**省 4 倍显存**。

听起来很美，但 GLM 团队发现一个坑：**用 Muon optimizer 时 MLA 性能不如 GQA-8**。

Muon 是个比较新的 optimizer，对权重矩阵做 Newton-Schulz 正交化——直觉上类似于让权重矩阵的 singular value 均匀分布，防止某个方向 dominate。原始 recipe 对 $W^{UQ}, W^{UK}, W^{UV}$ 这些 up-projection矩阵**整体**做正交化。

问题在哪？$W^{UK}$ 是一个把 512 维 latent 投影到所有 head 的 K 空间的大矩阵。不同 head 关注的语义不同，有的 head 看 syntax，有的看 coreference。整体正交化等于强迫所有 head 共享同一套 spectral 结构，**破坏了 head specialization**。

GLM-5 的 fix 叫 **Muon Split**：把 $W^{UK}$ 按列切成 $H$ 块（每块对应一个 head），每块独立正交化。

$$W^{UK} = [W^{UK}_1 | W^{UK}_2 | \cdots | W^{UK}_H]$$

每个 $W^{UK}_i$ 独立做 Newton-Schulz，学习率量级自然分化。

效果（Table 1）：

| | MMLU | BBH | HumanEval |
|---|---|---|---|
| GQA-8 | 61.2 | 53.3 | 38.5 |
| MLA（原始 Muon） | 61.5 | 48.9 | 33.5 |
| MLA + Muon Split | **62.5** | 51.8 | 36.7 |

BBH 从 48.9 跳到 51.8，HumanEval 从 33.5 到 36.7。MLA 配合 Muon Split 在 MMLU 上甚至**超过 GQA-8**。

还有一个 bonus：attention logit 的 scale 在训练全程保持稳定，**不需要任何 logit clipping**。这在工程上是个强信号——说明优化器没有把权重推向极端值。

### MLA-256：为 decoding 重新切蛋糕

MLA 还有个坑：decoding 时 attention 计算在 latent space 内做 dot product，维度是 576。GQA 的 dot product 只在 128 维做。decoding 是 memory-bound 阶段，576 维的 dot product **比 128 维慢 4.5 倍**。

DeepSeek-V3 选 head 数时专门对着 H800 的 roofline 调，让它刚好卡在 compute-memory 平衡点。但 GLM-5 要跑在 7 家国产芯片上，硬件特性各不相同，不能绑死 H800。

GLM-5 的解法：
- head 维度从 192 提到 256
- head 数减少 1/3
- 训练时总 FLOPs 和参数量不变
- decoding 时 head 更少但每个 head 维度更高，**总 dot product 计算量下降**

直觉：DeepSeek 把 latent 切成很多薄 head，GLM-5 切成更少的厚 head。厚的 head 拿到更丰富的 representation，decoding 时算的次数少。

Table 1 最后一行 MLA-256 + Muon Split 的性能跟 MLA + Muon Split 持平，验证了重新切蛋糕没掉点。

### MTP 参数共享：一个 layer 当三个用

Multi-Token Prediction（MTP）是让模型一次预测多个 token。训练时预测下 $n$ 个 token 需要 $n$ 个 MTP layer，inference 时这些 layer 当 speculative decoding 的 draft model。

问题：每多一个 MTP layer，参数和 KV cache 线性增长。DeepSeek-V3 折中只用 1 个 MTP layer，inference 时 predict 2 个 token，但训练只见 1 步、inference 要推 2 步，**train-infer mismatch 导致第二个 token 的 accept rate 低**。

GLM-5 的 trick：训练时**共享 3 个 MTP layer 的参数**。一个 layer 在 self-conditioning 下被复用 3 次。

直觉：这跟 RNN 的 weight sharing 哲学一致。RNN 在每个时间步复用同一套权重，天然适配序列生成。MTP 参数共享让模型学到"从 hidden state 递归 predict 多步"的结构，而不是"每个位置独立 predict"。

效果（Table 2，同样 4 个 speculative step）：
- DeepSeek-V3.2 accept length: 2.55
- GLM-5 accept length: **2.76**

每步多 accept 0.21 个 token，在长序列 inference 里累积效果可观。

### DSA：不丢信息的 sparse attention

200K context 下 dense attention 是 $O(L^2)$，光 attention matrix 就 $200000^2 = 4 \times 10^{10}$ 个 entry。但其中 90% 是 redundancy——大部分 token 跟当前 query 关系不大。

DSA（DeepSeek Sparse Attention）的做法：加一个轻量级 **indexer**，每个 query 位置选 top-k（$k=2048$）个最相关的 KV entry，只在 selected subset 上算 attention。

关键设计：indexer **不丢弃信息**，只是选择性地计算。所有 KV 都还在，只是不全部参与 attention 运算。这跟 sliding window attention（SWA）和 linear attention 本质不同——后两者在信息层面就丢了。

GLM-5 做了详尽的 ablation（Table 5，GLM-9B 上跑的）：

| 方法 | RULER 128K | 相对 full attn 的下降 |
|---|---|---|
| Full Attention | 75.28 | baseline |
| SWA Interleave（交替窗口） | 6.51 | **↓30.35** |
| SWA Pattern（搜索最优层分配） | 53.95 | ↓5.69 |
| GDN（Gated DeltaNet, linear attn） | 64.00 | ↓11.28 |
| SimpleGDN（改良版） | 67.03 | ↓8.25 |
| **DSA（Table 6）** | **78.86** | **↓0.35** |

SWA Interleave 掉了 30 个点，基本废了。SWA Pattern 用 beam search 在 16K context 上找最优的 full/SWA 层分配模式，再外推到所有 context length，也掉了 5.7 个点。所有 lossy 方法都有不可消除的精度损失。

DSA 几乎无损。Table 6 更细：只在 GLM-4.7-Flash 上 warmup 1000 步（只训 indexer），128K RULER 从 79.21 掉到 71.35；再做 150B token joint training，128K 回到 78.86，16K/32K/64K 全部**反超** baseline。

GLM-5 实际的 DSA 训练用的是 **Continued Pre-Training**——不从 scratch 训，从 MLA base model 继续训：
1. Warmup：1000 步，只训 indexer，lr 5e-3
2. Sparse adaptation：20B tokens，全模型联合训，lr 1e-5

DeepSeek-V3.2 训 DSA 用了 943.7B tokens，GLM-5 只用 20B。**复用 dense base model 的知识，cost 极低**。

Table 3：DSA base model 在 long-context benchmark 上跟 MLA base model 持平甚至更好（SQuAD-128k 79.7 → 86.0）。

---

## 第二道题：RL 怎么训长 horizon agent

### 同步 RL 的 GPU 浪费问题

传统 RL 训练是同步的：采一批 trajectory → 算 reward → 更新模型 → 采下一批。问题在于 agent 任务的 trajectory 长度方差极大——有的 agent 3 步搞定，有的要 200 步跑半小时。同步等所有 trajectory 采完才更新，**GPU 大部分时间在等最慢的那个 agent**。

GLM-5 的解法：**完全异步解耦**。

- Inference engine 在一堆 GPU 上持续生成 trajectory
- Trajectory 攒够一批就送 training engine
- Training engine 在另一堆 GPU 上更新参数
- 每 K 次更新后，把新权重 push 给 inference engine

这等价于把"生成"和"训练"流水线化，GPU 各干各的。但引入一个核心问题：**off-policy**。一条 trajectory 在生成期间，inference engine 可能已经更新了好几次权重，导致 trajectory 里不同 token 是不同版本模型生成的。

### 稳定异步 RL 的四个机制

**机制一：TITO（Token-in-Token-out）**

传统做法是 inference engine 返回 text，trainer 重新 tokenize。重新 tokenize 会引入微妙的不一致：special token 放置、空白处理、截断位置都可能不同。在 multi-turn streaming 场景下，action 和 reward 的对齐会被破坏。

TITO 的做法：inference engine 直接输出 token IDs + metadata，trainer 原样使用，不重新 tokenize。等于两边共享同一个 tokenization 状态机，消除所有 mismatch。

**机制二：Direct Double-Sided Importance Sampling**

异步 RL 的核心难题：传统 PPO 要算 importance ratio $r_t = \pi_\theta / \pi_{\theta_{old}}$，但 $\pi_{\theta_{old}}$ 在异步场景下不可追溯——一条 trajectory 生成期间 policy 已经变了好几次。保存所有历史 checkpoint 不可行。

GLM-5 的简化（Eq. 4）：

$$r_t(\theta) = \exp\left(\log \pi_\theta(a_t|s_t) - \log \pi_{\text{rollout}}(a_t|s_t)\right)$$

变量解释：
- $a_t$：第 $t$ 步的 action（生成的 token）
- $s_t$：第 $t$ 步的状态（context $x, y_{<t}$）
- $\pi_\theta$：当前 training policy
- $\pi_{\text{rollout}}$：rollout 时记录的 log-prob（behavior policy proxy）

**直接用 rollout 时记录的 log-prob 作为 behavior policy**，绕过历史 checkpoint tracking。

然后做一个比 PPO 更激进的 clip（Eq. 5）：

$$f(x; \epsilon_\ell, \epsilon_h) = \begin{cases} x, & 1 - \epsilon_\ell < x < 1 + \epsilon_h \\ 0, & \text{otherwise} \end{cases}$$

PPO 是 clip 到边界，GLM-5 是**直接 mask 掉**——importance ratio 超出 $[1-\epsilon_\ell, 1+\epsilon_h]$ 的 token 不参与梯度计算。

直觉：异步场景下极端 ratio 代表严重 off-policy，与其 trust 一个被截断的 noisy gradient，不如直接丢弃。这跟 IcePop（https://arxiv.org/abs/2509.18055）思路类似但更简单——连 $\pi_{\theta_{old}}$ 都不维护了。

**机制三：丢弃太旧的 sample**

记录每条 trajectory 经历的 policy 版本 $(w_0, \dots, w_k)$。如果当前版本 $w'$ 减去最早版本 $w_0$ 超过阈值 $\tau$，说明这条 trajectory 太旧了，直接丢掉。

另外，环境崩溃（跟模型无关的 sandbox 挂掉）的 sample 也丢。GRPO group 里如果 valid sample 过半，用 valid sample 重复 padding；不过半就丢整组。

**机制四：DP-aware Routing**

MoE inference 在 Data Parallel 下，同一个 agent 的 multi-turn request 要路由到同一个 DP rank 才能复用 KV cache。GLM-5 用 consistent hashing 把 rollout ID 映射到固定 DP rank，保证 multi-turn 的 prefix 不用重新 prefill。

### DSA 在 RL 里的一个意外坑

Paper 报告了一个非常有工程价值的发现：**DSA indexer 的 top-k 必须是 deterministic 的**。

SGLang 默认用 CUDA 实现的 top-k，non-deterministic（同一输入不同次运行结果可能略不同）。在 inference 场景下这无所谓，但在 RL 里**几步之后 entropy 暴跌**，训练崩掉。

原因类比 MoE 的 routing replay：MoE 训练时记录每个 token 激活哪些 expert，inference 时复用，保证 train-infer 一致。DSA 的 top-k indices 也是类似——但要存每个 token 2048 个 index 太多了，存不起。

解法：用 `torch.topk`（deterministic）替代 CUDA top-k。慢一点但稳定。同时**在 RL 阶段 freeze indexer 参数**，防止 indexer 学习不稳定。

### Reasoning RL 的具体算法

基于 GRPO（DeepSeekMath）+ IcePop，但**移除了 KL regularization**。

损失函数（Eq. 1）：

$$
\mathcal{L}(\theta) = -\mathbb{E}\left[\frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \text{pop}(\rho_{i,t}, 1/\beta, \beta) \cdot \min(r_{i,t} \hat{A}_{i,t}, \text{clip}(r_{i,t}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}) \hat{A}_{i,t})\right]
$$

变量：
- $G=32$：group size，每个 prompt 采 32 条 trajectory
- $\rho_{i,t}$：train-infer mismatch ratio，$\pi_{\theta_{old}}^{\text{train}} / \pi_{\theta_{old}}^{\text{infer}}$
- $\beta=2$：pop 的容忍上下界 $[0.5, 2]$
- $r_{i,t}$：PPO importance ratio
- $\hat{A}_{i,t}$：group-normalized advantage，$(R_i - \text{mean}(R_1...R_G)) / \text{std}(R_1...R_G)$
- $\epsilon_{\text{low}}=0.2, \epsilon_{\text{high}}=0.28$：asymmetric clip

为什么去 KL？KL 在标准 PPO 里约束 policy 不偏离 reference 太远，但在长序列上 KL 数值会累积到很大，且 reference policy 跟当前 policy 差距大时 KL 梯度噪声大。IcePop 的 pop 机制直接 mask outlier token，比 KL 更"硬"但更稳定。

四个 domain 混合训：math、science、code、TIR（tool-integrated reasoning）。难度过滤专门挑"GLM-4.7 解不出但 GPT-5.2/Gemini 3 能解"的题。

### General RL 的三维目标

reasoning RL 之后是 general RL，目标拆成三维：

1. **Foundational correctness**：指令遵循、事实正确、无幻觉、语言流畅。这是 floor——有事实错误的 response 再漂亮也误导人
2. **Emotional intelligence**：empathy、insight、自然风格
3. **Task-specific quality**：写作、翻译、role-play 等具体任务

Reward system 是 hybrid 的：
- **Rule-based**：精确但表达能力有限
- **ORM（Outcome Reward Model）**：低方差但易 hacking
- **GRM（Generative RM）**：抗 hacking 但高方差

三者混合，rule 抓硬约束、ORM 抓整体质量、GRM 抓细粒度判别。

还有个关键设计：**human-in-the-loop style alignment**。纯 model-generated 优化会收敛到"AI 味"风格——verbose、formulaic。引入 expert human response 作为风格锚点，让模型学到人类自然写作的模式。这不是 RLHF 里的 preference signal，是直接把 human response 当 anchor。

### On-Policy Cross-Stage Distillation：最后一步救场

四阶段 RL 顺序跑下来，前面的能力会 catastrophic forget。最后一步用 on-policy distillation 恢复。

Advantage 改写为（Eq. 2）：

$$\hat{A}_{i,t} = \text{sg}\left[\log \frac{\pi_{\theta_{\text{teacher}}}^{\text{infer}}(y_{i,t}|x, y_{i,<t})}{\pi_\theta^{\text{train}}(y_{i,t}|x, y_{i,<t})}\right]$$

变量：
- $\text{sg}$：stop gradient（`.detach()`），梯度不回传到 teacher
- $\pi_{\theta_{\text{teacher}}}^{\text{infer}}$：前序阶段 checkpoint 作为 teacher
- $\pi_\theta^{\text{train}}$：当前 student

直觉：student 和 teacher 的 log-prob gap 直接当 advantage。student 跟 teacher 一致时 advantage=0，student 比 teacher 差时 advantage 为正，推 student 向 teacher 靠拢。Group size 设为 1（不再需要 group baseline），batch size 1024，throughput 极高。

---

## 第三道题：怎么造上万个训练环境

### SWE 环境

基于 RepoLaunch 框架，从 real-world GitHub issue–PR 对构建：
- 自动分析 repo 的 installation/dependency
- LLM 生成 language-aware 的 log parser
- 提取 Fail-to-Pass（新功能必须通过的测试）和 Pass-to-Pass（已有功能不能 break 的测试）
- 跨 9 种语言建了 **10k+ 可验证环境**

参考：https://arxiv.org/abs/2505.23419

### Terminal 环境两条线

**Seed-based**：种子任务 → LLM brainstorm → construction agent 实例化成 Harbor format → refine agent 验证。Docker build 成功率 >90%。

**Web-corpus-based**：从 code-relevant 网页抽取 → coding agent 生成 Harbor task → **自己是自己的 first-pass evaluator**，失败就迭代修复，通过才进数据集。

### Search Agent 的 Web Knowledge Graph

这个设计很巧妙：
1. 早期 search agent 跑 trajectory，收集 2M+ URL
2. LLM 做 entity extraction → 构建知识图谱
3. 采样低频 entity 作为 seed，扩展 multi-hop 邻域形成 subgraph
4. 把 subgraph 转换成隐式 multi-entity 关系链问题
5. 三阶段难度过滤：tool-free 模型能答的丢掉、early agent 几步能解的丢掉、bidirectional verify 不一致的丢掉

### Hierarchical Context Management for Search

BrowseComp 在 100K+ context 下精度显著下降。GLM-5 用了一个 hybrid 策略：

**Keep-recent-k**：trajectory $(q, r_1, a_1, o_1, \dots, r_n, a_n, o_n)$ 中，observation $o_i$ 只保留最近 $k=5$ 轮，更早的折叠成"Tool result is omitted to save tokens"。

**Hierarchical 组合**：keep-recent 一直应用，当总 context 超过 $T=32K$ 时触发 Discard-all，清空整个 tool-call history 重来。

效果（Figure 8）：单独 Discard-all vs +keep-recent-k 在所有 compute budget 下都有增益，BrowseComp 从 55.3% → **75.9%**（开源 SOTA）。

直觉：keep-recent 保 short-term memory，discard-all 防 long-context 退化，两者覆盖不同时间尺度。

### Slide Generation：三级 Reward

这是 paper 里最 novel 的 agent task。模型生成 HTML 形式的 slide，在三个层级上给 reward：

**Level-1：静态 HTML 属性**——positioning、color、typography 的 rule-based 检查。检测幻觉图片和重复图片。

**Level-2：runtime rendering 属性**——DOM 渲染后的 bounding box、宽高。需要 distributed rendering service 实时跑。这里出现了 reward hacking（Figure 9）：
- 模型硬截断超长内容来"满足"宽度约束
- 模型过度操纵 spacing 来通过 layout 检查

修复 renderer 实现堵漏洞。

**Level-3：视觉感知**——abnormal whitespace detection 等。

训练用 dynamic sampling（丢简单样本集中训难的）、token-level policy gradient、rejection sampling + masking refinement（个别页面有缺陷就 mask 那页保留其余）。

效果：16:9 符合率 40% → 92%，vs GLM-4.5 win rate 67.5%。

---

## Preserved Thinking：把 reasoning 当 state

这个设计我特别想展开讲。

传统 thinking model 每轮都重新 think。turn 1 推理出"应该用 SQLAlchemy 因为 schema 会变"，到 turn 5 遇到 ORM 配置问题时，模型重新 think，可能得出不同结论。这就 inconsistency 了。

GLM-5 的 **Preserved Thinking**：coding agent 场景下，所有 thinking block 跨 turn 保留。turn 1 的推理在 turn 5 还在 context 里，模型直接复用。

这背后的哲学转变：**reasoning 从"每次生成的产物"变成"可复用的 state"**。类似于你写代码时的设计文档——不是每次改 bug 都重写设计文档，而是在已有设计上增量更新。

这跟 RAG 里的 "memory" 概念、ReAct 里的 "scratchpad" 一脉相承，但 GLM-5 把它做成了 first-class 的训练目标——SFT 数据里就包含了 preserved thinking 的 trajectory。

---

## RL 基础设施：slime

slime 是 GLM 自家的 post-training framework，三个关键设计：

### Server-based Rollout

Rollout server 暴露 HTTP API，任何 agent 框架（OpenHands、Cline、Claude Code）都能直接调用。Rollout 逻辑完全跟 training 解耦——换 agent framework 不用改 training code。

### 尾延迟优化

RL rollout 的优化目标不是 throughput，是 **end-to-end latency**——由最慢的那个 trajectory 决定。

- **Multi-node inference with DP-attention**：EP64+DP64 over 8 nodes，提供足够 distributed KV cache
- **FP8 rollout + MTP**：FP8 降 per-token latency，MTP 在 small-batch decoding 下收益最大
- **PD disaggregation**：prefill 和 decode 分到不同 GPU，避免长 prefill 抢占 decode

直觉：传统推理优化 throughput（batch 越大越好），RL rollout 优化的是单条 trajectory 完成时间——batch 小但 tail 决定 wall-clock。

### Heartbeat 容错

Server 周期性发心跳，unhealthy 的被 deregister，retry 自动路由到健康 server。单点失败不中断训练。

---

## 评测：从 static benchmark 到 dynamic benchmark

### ARC Benchmarks

Table 7 关键数字（vs Claude Opus 4.5）：

| Benchmark | GLM-5 | Claude 4.5 |
|---|---|---|
| HLE (w/ tools) | **50.4** | 43.4 |
| SWE-bench Verified | 77.8 | **80.9** |
| BrowseComp (w/ CM) | **75.9** | 57.8 |
| τ²-Bench | 89.7 | **91.6** |
| GDPval-AA Elo | **1409** | 1400 |

GLM-5 在 HLE with tools 和 BrowseComp 上明显领先 Claude。SWE-bench 和 τ²-Bench 略低但接近。

### CC-Bench-V2：真实 agentic engineering

这是 GLM 自建的 benchmark，三个维度：

**Frontend**：220 task，HTML/React/Vue，700+ check items。两阶段评测——先 build，再用 GUI agent（Playwright）模拟用户交互验证每个 check item。

Table 8 关键结果：

| | GLM-5 | GLM-4.7 | Claude 4.5 |
|---|---|---|---|
| HTML ISR | 38.9 | 35.4 | **52.2** |
| React CSR | 71.0 | 49.4 | **70.7** |
| Backend Pass@1 | 25.8 | 19.6 | **26.9** |
| Repo Exploration | **65.6** | 47.8 | 64.5 |
| Chained Tasks | 52.3 | 43.0 | **61.6** |

**Repo Exploration 上 GLM-5 反超 Claude**（65.6 vs 64.5）。这很 surprising。Paper 解释：repo exploration 不依赖 raw code generation，更依赖 strategic search 和 semantic association——GLM-5 的 agentic tool-use trajectory 训练有优势。

**Chained Tasks 上 Claude 仍领先**（61.6 vs 52.3）。Chained task 是多步开发，每步改 codebase 给后续步。错误会 compound——一步的小 bug 在后续步被放大。这需要 long-context consistency 和 long-horizon self-correction，是 GLM-5 的下一个 improvement target。

### Agent-as-Judge 可靠性

130 个 check item 人 vs agent 评分一致率 94%。8 个 frontier model 排名 Spearman correlation 85.7%。GUI agent-as-judge 基本可靠。

### SWE-rebench：动态 benchmark

SWE-bench Verified 已发布 2 年，可能 contamination。SWE-rebench 持续挖 fresh GitHub issue。

Table 9：GLM-5 resolved rate 42.1%，Claude Opus 4.6 52.9%，GPT-5.2 xhigh 51.7%。GLM-5 在 dynamic benchmark 上仍能 generalize，说明没 overfit 静态 benchmark。

参考：https://arxiv.org/abs/2505.20411

---

## 国产芯片适配

7 家芯片：Huawei Ascend、Moore Threads、Hygon、Cambricon、Kunlunxin、MetaX、Enflame。

Ascend 案例：
- **W4A8 mixed precision**：Attention/MLP 用 INT8，MoE experts 用 INT4。用 QuaRot 抑制 outlier
- **Fusion kernel**：Lightning Indexer 融合 score+ReLU+TopK，Sparse Flash Attention 针对 GLM-5 pattern 优化，MLAPO 融合 13 个小算子
- **推理引擎**：vLLM-Ascend 和 SGLang 双适配，D2H copy overlap、RadixCache、FlashComm

单节点性能接近 dual-GPU 国际集群，长序列部署成本降 50%。

---

## Pony Alpha：匿名发布的盲测

GLM-5 在 OpenRouter 匿名发布为 "Pony Alpha"。社区猜测：
- Claude Sonnet 5: 25%
- DeepSeek: 20%
- Grok: 10%
- GLM-5: 其他

当被揭示是 GLM-5 时，社区惊讶中国 LLM 能达到 frontier。这证明 brand-agnostic 的 blind test 让模型能力自己说话。

参考：https://openrouter.ai/

---

## 跟同期工作对比

**vs DeepSeek-V3.2**：
- 都用 MLA + DSA + MTP
- DeepSeek 绑 H800 roofline，GLM-5 改 MLA-256 适配多硬件
- DeepSeek 用 943.7B token 训 DSA，GLM-5 只用 20B（continued pre-training）
- GLM-5 加了 IcePop（去 KL），DeepSeek 没有

**vs Kimi K2.5**：
- K2.5 偏 visual agentic，GLM-5 偏 SWE/terminal
- HLE with tools：K2.5 51.8，GLM-5 50.4
- BrowseComp w/ CM：GLM-5 **75.9**，K2.5 74.9

**vs Qwen3**：
- 都用 GRPO + on-policy distillation
- Qwen3 没有 DSA，GLM-5 long-context 优势由此而来

**vs Claude Opus 4.5**：
- Chained tasks 上 Claude 仍领先（61.6 vs 52.3）
- GDPval-AA Elo 上 GLM-5 略胜（1409 vs 1400）

---

## 整篇 paper 的核心 intuition

1. **架构效率靠 redesign 不靠 shrink**：MLA-256、Muon Split、MTP 共享都是重新设计组件适配优化器/硬件，不是简单降参
2. **DSA 是 lossless 的 sparse attention**：indexer 不丢信息只选计算，SWA/linear attention 都丢信息，ablation 数据说话
3. **异步 RL 是 enabling technology**：没有 TITO + direct importance sampling + DP routing，长 horizon agent RL 根本跑不起来
4. **Preserved thinking 把 reasoning 当 state**：从"每次 think"到"think 可复用"，long-horizon consistency 的关键
5. **Reward hybrid + human anchor 抗 hacking**：rule + ORM + GRM 三层，human response 当 style anchor
6. **Dynamic benchmark 比 static 更可信**：SWE-rebench 避免 contamination
7. **Blind test 让能力说话**：Pony Alpha 证明 brand-agnostic 评测的价值

---

## 如果我是 reviewer 会问的问题

1. **Muon Split 为什么在 MLA 上有效但在 GQA 上不需要**？没有 ablation。直觉是 MLA 的 $W^{UK}$ 同时承担 head specialization 和 rank compression，split 解耦两者，但缺实验证据。

2. **Preserved thinking 在 turn 数 >50 时怎么处理 context overflow**？cap 旧 thinking 还是 summarize？paper 没细说。

3. **DSA indexer freeze 在 RL 阶段，但 base model 变化导致 indexer 输入分布 drift**，是否有累积 mismatch？

4. **异步 RL 的 $\tau$（policy version lag threshold）怎么选**？sensitivity analysis 缺失。

5. **Slide generation 的 Level-2 reward 在 distributed rendering 下 latency**，是否成为 RL bottleneck？

6. **Pony Alpha 期间收集的 user feedback 是否回灌训练**？

---

希望这个"人话版"帮 build intuition。整篇 paper 的核心 contribution 不在单个 trick，而在 **architecture → RL infra → agent environment 整个 stack 的协同 redesign**——单个组件看都 incremental，但 end-to-end 的 leverage 极大。

如果想深挖某个 section（比如异步 RL 的工程细节、DSA indexer 训练动力学、slide reward design），告诉我，我可以再展开。

---

# GLM-5 技术深度解析：从 Vibe Coding 到 Agentic Engineering

Andrej 你好，这篇 paper 信息密度极高，我从架构、训练 pipeline、RL 工程、agent 能力四个层面来 build intuition。我会把公式拆开讲变量含义，并关联到 DeepSeek-V3.2、Kimi K2.5、Qwen3 等同期工作。

---

## 1. Paradigm Shift：为什么是 "Vibe Coding → Agentic Engineering"

Karpathy 你自己提出的 "vibe coding" 这个词，本质是说 human 在 prompt、AI 在 generate，但 human 仍然承担 plan-implement-iterate 的心智负担。GLM-5 的 thesis 是：当 agent 能 self-plan、self-implement、self-iterate，训练目标就从"输出代码片段正确"变成"long-horizon trajectory 全程稳定"。

这导致三个核心需求：
- **Long-context fidelity**（200K context，multi-turn 不崩）→ 需要 DSA
- **Long-horizon RL stability**（agent rollout 跨小时级）→ 需要异步解耦 + token-level importance sampling
- **Verifiable environment scale**（>10K SWE 环境）→ 需要 RepoLaunch-style 自动化

整篇 paper 的所有技术贡献都可以回溯到这三条 motivation。

参考：https://github.com/zai-org/GLM-5

---

## 2. Architecture：在 MLA/MoE/MTP 三个维度同时 push

### 2.1 规模与 MoE 配置

| | GLM-4.5 | GLM-5 |
|---|---|---|
| Total params | 355B | **744B** |
| Active params | 32B | **40B** |
| Experts (total) | 160 | **256** |
| Dense layers | 3 | 3 |
| MoE layers | 89 | **75** |
| Layers total | 92 | **78** |

直觉：减少 layer 数（92→78）是为了减小 Expert Parallel 的 all-to-all 通信开销；同时把 expert 数扩到 256 来补偿单层 capacity。这是典型的"加宽不加深"的 MoE scaling 策略，因为 layer 间的 all-reduce 是 sequential，而 expert 数增加只影响单层 router 的 fan-out。

参考 DeepSeek-V3 用 256 experts 也是同样思路：https://arxiv.org/abs/2412.19437

### 2.2 Multi-Latent Attention + Muon Split

MLA（DeepSeek-V2 引入）的核心是把 K、V 压缩到低秩 latent space：

$$c_t = W^{KV} h_t, \quad k_t = W^{UK} c_t, \quad v_t = W^{UV} c_t$$

其中 $c_t \in \mathbb{R}^{d_c}$ 是低秩 latent（GLM-5 中 $d_c = 512$），$W^{UK}, W^{UV}$ 是 up-projection。这样 KV cache 只存 $c_t$，从 2048 维降到 512 维，4× 节省。

但 paper 报告了一个关键 empirical finding：**MLA + Muon optimizer 性能不如 GQA-8**。Muon 是一种对 2D 权重矩阵做 Newton-Schulz 正交化的优化器，原始 recipe 对 $W^{UQ}, W^{UK}, W^{UV}$ 整体做正交化。

**Muon Split 的 intuition**：把 $W^{UK} \in \mathbb{R}^{d_c \times d_h}$ 按不同 head 切成 $H$ 个子矩阵 $W^{UK}_1, \dots, W^{UK}_H$，每个子矩阵独立正交化。

为什么不 split 不行？因为不同 head 的 query 关注的语义维度不同，如果整体正交化，相当于强制所有 head 共享同一套 spectral 结构，这破坏了 head specialization。Split 后每个 head 的投影矩阵有独立的学习率量级，logit scale 在训练中保持稳定，**不需要任何 attention logit clipping**——这是一个非常强的工程信号。

Table 1 结果（截取关键指标）：

| | MMLU | BBH | HumanEval |
|---|---|---|---|
| GQA-8 | 61.2 | 53.3 | 38.5 |
| MLA | 61.5 | 48.9 | 33.5 |
| MLA + Muon Split | **62.5** | 51.8 | 36.7 |
| MLA-256 + Muon Split | 62.0 | 51.3 | 36.6 |

MLA + Muon Split 在 MMLU 上甚至超过 GQA-8。

### 2.3 MLA-256：为 decoding 重新设计 head dim

MLA 的另一个坑：decoding 时 attention 计算在 latent space 内做 dot product（576 维），比 GQA 的 128 维高 4.5×，decoding 阶段（memory-bound）反而更慢。

DeepSeek-V3 选 head 数的依据是 H800 的 roofline，但 GLM-5 要适配 7 家国产芯片，不能绑死 H800。GLM-5 的解法是：

- head dim 从 192 提到 256
- head 数减少 1/3
- 训练时总计算量保持不变（参数量不变）
- decoding 时每次 dot product 维度更高但 head 更少，**总 FLOPs 降低**

直觉：把 MLA 的 latent space 摊薄到更少的 head 上，每个 head 拿到更丰富的 representation，这跟 Llama-style GQA 把 KV 头压缩成 query 头的 1/N 的思路相反，MLA-256 把"head 之间的稀疏"换成"head 内的稠密"。

### 2.4 Multi-Token Prediction with Parameter Sharing

MTP（Gloeckle et al. 2024）训练时用 $n$ 个独立 MTP layer 预测下 $n$ 个 token，inference 时作为 speculative decoding 的 draft model。问题：每多一个 MTP layer，参数和 KV cache 线性增长。

DeepSeek-V3 折中：只训练 1 个 MTP layer，inference 时 predict 2 tokens，但 train-infer discrepancy 降低 second token 的 accept rate。

GLM-5 的 trick：**训练时共享 3 个 MTP layer 的参数**。等于一个 layer 在 self-conditioning 下被复用 3 次，memory 一致，accept rate 提升。

Table 2：accept length 对比（同样 4 个 speculative step）：
- DeepSeek-V3.2: 2.55
- GLM-5: **2.76**

直觉：参数共享迫使模型学到"从 last hidden state 一路 predict 多步"的递归结构，这跟 RNN 的 weight sharing 哲学一致——共享参数让模型天然适配自回归的多步预测。

### 2.5 DeepSeek Sparse Attention (DSA) 的 Continued Pre-Training

DSA 的核心：dense attention 在 128K context 下 $O(L^2) = 1.6 \times 10^{10}$ 个 attention entries，但其中 90% redundancy。DSA 用一个 lightweight **indexer** 在每个 query 位置 top-k 选择最相关的 KV entries（$k=2048$），只在 selected subset 上计算 attention。

GLM-5 没有从 scratch 训 DSA，而是用了 **Continued Pre-Training**：
1. **Warm-up stage**：1000 steps，batch size 14 × 202,752 tokens，lr 5e-3 → 只训 indexer
2. **Sparse adaptation stage**：20B tokens，lr 1e-5 constant，全模型联合训练

成本极低（DeepSeek-V3.2 用了 943.7B tokens），但 Table 3 显示 long-context 性能持平：

| | MQ-NIAH-128k | SQuAD-128k | HotpotQA-128k |
|---|---|---|---|
| MLA | 100.0 | 79.7 | 66.3 |
| DSA | 100.0 | **86.0** | 63.0 |

HotpotQA 略降（66.3→63.0）但 SQuAD 反而升（多跳推理受益于 sparse 的 selective attention）。

参考：https://arxiv.org/abs/2512.02556

### 2.6 Efficient Attention 消融实验（GLM-9B 上做的）

这是一个非常诚实的 ablation，比较了 4 种 efficient attention 在 GLM-9B 上的 continual training：

- **SWA Interleave**：奇偶层交替 full/windowed
- **SWA Pattern (Search-Based)**：用 beam search 找最优 SWA 层分配
- **GDN**（Gated DeltaNet）：linear attention + gated recurrence
- **SimpleGDN**：作者改良版，复用预训练 QKV 投影权重

Search-based 的关键 trick：只在 16K context 上做 beam search（beam size 8，每次优化 2 层，10 步收敛），得到 pattern `SFSSFFSSSFFFFSSFSFFFFFFSFSFSSFSSFSFSSFSSS`，然后**外推**到所有 context length。

Table 5 关键结果（RULER @128K，full attn baseline 75.28）：

| Method | RULER 128K | Δ |
|---|---|---|
| SWA Interleave | 6.51 | ↓30.35 |
| SWA Pattern | 53.95 | ↓5.69 |
| GDN | 64.00 | ↓11.28 |
| SimpleGDN | 67.03 | ↓8.25 |
| **DSA (Table 6)** | **78.86** | ↓0.35 |

DSA 在 128K 上几乎无损（-0.35），而 SWA pattern 都掉了 5+ 个点。**DSA 的核心优势是 lossless**：indexer 不丢弃信息，只是 sparse 计算，而 SWA/linear attention 是 information-lossy 的。

Table 6 更有意思——只在 GLM-4.7-Flash 上做 warmup（1000 steps 只训 indexer），128K RULER 从 79.21 掉到 71.35；再做 150B token joint training，128K 反而 78.86（接近 baseline），16K/32K/64K 全部反超 baseline。这说明 **DSA indexer 的对齐成本主要在最长的 context 上**，短 context 几乎免费。

---

## 3. Pre-Training 数据与 Mid-Training

### 3.1 数据策略

27T pre-training tokens，重点 early prioritize code 和 reasoning。几个亮点：

- **DCLM classifier**：sentence embedding-based，扩展到 standard classifier 之外
- **World Knowledge classifier**：Wikipedia + LLM-labeled 数据训练，专门挖 medium-low-quality 数据里的 long-tail 知识
- Code：28% fuzzily deduplicated unique tokens 增长，针对 Scala/Swift/Lua 等低资源语言训练 dedicated classifier
- Math/Science：chunk-and-aggregate scoring，专门处理长文档评分

### 3.2 Mid-Training 三阶段上下文扩展

| Stage | Context | Tokens |
|---|---|---|
| 1 | 32K | 1T |
| 2 | 128K | 500B |
| 3 | **200K** | 50B |

**直觉**：第三阶段（200K）虽然只有 50B token，但 paper empirically 发现它**反向提升 128K context 内的性能**。这跟 NextLong、EntropyLong 的发现一致——更长的 training context 让模型学到更强的 long-range dependency 模式，反哺较短 context 的表现。

Mid-training 数据的两类合成：
- **Natural**：books、papers、长文档，多阶段 PPL/dedup/length 过滤
- **Synthetic**：inspired by NextLong/EntropyLong，用 **interleaved packing** 把高度相似的文本拼到一条 sequence 里，缓解 "lost-in-the-middle"

Software engineering 数据：issue–PR 对（约 10M 对）扩到 **160B unique tokens**，每个 issue–PR 对检索更多相关文件作为 context。

### 3.3 INT4 QAT

在 SFT 阶段做 INT4 quantization-aware training，开发 bitwise-identical 的量化 kernel——训练和推理用同一套 kernel，消除 QAT 常见的 train-infer 量化算子不一致问题。这个细节很关键，Qwen3 也有类似设计。

---

## 4. Post-Training Pipeline：四阶段 RL + Distillation

整个 pipeline：

```
SFT → Reasoning RL → Agentic RL → General RL → On-Policy Cross-Stage Distillation
```

### 4.1 SFT：三种 Thinking Mode

GLM-5 引入三种 thinking 模式（Figure 7）：

- **Interleaved Thinking**：每次 response 或 tool call 前都 think
- **Preserved Thinking**：coding agent 场景跨 turn 保留 thinking block，避免重新推理
- **Turn-level Thinking**：per-turn 控制是否 think

Preserved Thinking 是关键工程创新。直觉：long-horizon agent 在 turn 1 推理出"为什么用 SQLAlchemy 而非 raw SQL"，turn 5 如果重新推理会得到不同结论，造成 inconsistency。Preserved thinking 让模型把 reasoning 当作可复用的 state，而非每次 re-derive。

SFT 数据规模扩展，关键 trick：**erroneous trajectory segments 保留但 mask loss**——模型看到完整 error-correction 轨迹，学习"出错后如何纠正"，但不 reinforce 错误 action。这跟 OpenAI 的 verifier-guided RL 类似。

### 4.2 Reasoning RL：GRPO + IcePop（去 KL）

核心算法基于 GRPO（DeepSeekMath）+ IcePop（Zhao et al. 2025），但 **移除了 KL regularization**。

损失函数（Eq. 1）：

$$
\mathcal{L}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{old}}^{\text{infer}}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \text{pop}(\rho_{i,t}, 1/\beta, \beta) \cdot \min(r_{i,t} \hat{A}_{i,t}, \text{clip}(r_{i,t}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}) \hat{A}_{i,t}) \right]
$$

变量解释：
- $x$：prompt
- $\{y_i\}_{i=1}^G$：从 inference policy $\pi_{\theta_{old}}^{\text{infer}}$ 采样的 $G$ 条 trajectory（$G=32$）
- $|y_i|$：trajectory $i$ 的长度
- $\rho_{i,t}$：**train-infer mismatch ratio**（关键变量）
- $\beta=2$：pop 操作的容忍上下界（$[1/\beta, \beta] = [0.5, 2]$）
- $r_{i,t}$：PPO importance ratio（training policy / old training policy）
- $\hat{A}_{i,t}$：group-normalized advantage
- $\epsilon_{\text{low}}=0.2, \epsilon_{\text{high}}=0.28$：asymmetric clipping

Train-infer mismatch（Eq. 2）：
$$
\rho_{i,t} = \frac{\pi_{\theta_{old}}^{\text{train}}(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{old}}^{\text{infer}}(y_{i,t}|x, y_{i,<t})}
$$

直觉：训练用 train policy 算梯度，但样本是从 inference policy 采的（inference 用 FP8、MTP 等加速），两者分布有 gap。IcePop 用 $\rho$ 来 reweight 或 pop（直接 mask）那些 gap 太大的样本。

Pop operator：
$$
\text{pop}(\rho, 1/\beta, \beta) = \begin{cases} \rho, & 1/\beta \leq \rho \leq \beta \\ 0, & \text{otherwise} \end{cases}
$$

Paper 移除 KL 是个激进选择。KL 在 standard PPO 中约束 $\pi_\theta$ 不偏离 $\pi_{ref}$ 太远，但 KL 对长序列会积累大数值，且 reference policy 跟当前 policy 差距大时 KL 梯度噪声大。IcePop 的 pop 机制直接 mask outlier，比 KL 更"硬"但更稳定。

### 4.3 DSA 的 RL 稳定性

这是 paper 里最 interesting 的 engineering finding：**DSA indexer 的 top-k 必须是 deterministic 的**。

- SGLang 的 CUDA top-k：non-deterministic，导致 RL 几步后 entropy 暴跌
- `torch.topk`：deterministic，慢一点但稳定

类比 MoE 的 routing replay：MoE 训练时记录每个 token 的 expert assignment，inference 时复用。DSA 的 top-k indices 数量巨大（每 token 2048 个），存储全部 indices 不现实，所以用 deterministic top-k operator 保证 train 和 infer 一致。

**Indexer 在 RL 阶段 freeze**——防止 indexer 学习不稳定。

### 4.4 Mixed Domain Reasoning RL

四个 domain 混合训练：math、science、code、tool-integrated reasoning (TIR)。

数据来源细节：
- Math/Science：开源 dataset (Nemotron-Math, OpenMathReasoning) + 外部 vendor
- 难度过滤：保留 GLM-4.7 难解但 GPT-5.2/Gemini 3 能解的题
- Code：Codeforces + TACO + SYNTHETIC-2-RL + 内部 problem pool decomposed 成最小实现
- TIR：复用难 math/science 题 + 与 vendor 共建 STEM 题（必须用工具）

### 4.5 Agentic RL：完全异步解耦

这是 paper 的核心工程贡献。同步 RL 在 agent rollout 上有严重 GPU bubble——长尾 trajectory（一个 agent 跑 1 小时）会让所有 GPU 等它。

**解耦设计**：
- Inference engine 持续生成 trajectories
- Trajectory 数到阈值就送 training engine 更新
- Training engine 每 K 次 update 把权重 push 回 inference engine
- **每次权重 update 后 reset optimizer**——因为 rollout policy 已经变了，optimizer momentum 失效

**Multi-Task Rollout Orchestrator**：每个 agent task 注册为独立 microservice，central orchestrator 控制 per-task rollout ratio 和 generation speed，支持 1k+ 并发 rollout。

统一 trajectory 表示：所有 agentic task（SWE、terminal、search）统一编码为 message-list，便于 joint training。

### 4.6 Asynchronous RL 的稳定性机制

#### 4.6.1 TITO (Token-in-Token-out)

直觉：传统 Text-in-Text-out pipeline 把 inference engine 当黑盒，返回 text 后 trainer 重新 tokenize。这导致：
- token boundary 不一致（特殊 token、空白处理）
- multi-turn streaming 中 boundary 漂移
- action 和 reward 的对齐破坏

TITO Gateway 拦截所有 generation request，记录 token IDs + metadata，直接传给 trainer。这等价于让 inference engine 和 trainer 共享同一个 tokenization 状态机。

参考 Token-in-Token-out 的思想类似 OpenAI 的 internal RL infra 设计：https://thinkingmachines.ai/blog/on-policy-distillation

#### 4.6.2 Direct Double-Sided Importance Sampling

异步 RL 的核心难题：rollout engine 在一条 trajectory 生成期间可能经历多次权重 update，导致 $\pi_{\theta_{old}}$ 不可追溯。

传统做法：保存所有历史 checkpoint $\{\pi_{\theta_{old}^{(1)}}, \dots, \pi_{\theta_{old}^{(N)}}\}$，inference 每个 token 用对应版本的 policy log-prob。**工程上不可行**。

GLM-5 的简化（Eq. 4）：
$$
r_t(\theta) = \exp\left(\log \pi_\theta(a_t|s_t) - \log \pi_{\text{rollout}}(a_t|s_t)\right)
$$

直接用 **rollout 时记录的 log-prob** 作为 behavior policy proxy，绕过 $\pi_{\theta_{old}}$ 的 tracking。

Calibration function（Eq. 5）：
$$
f(x; \epsilon_\ell, \epsilon_h) = \begin{cases} x, & 1 - \epsilon_\ell < x < 1 + \epsilon_h \\ 0, & \text{otherwise} \end{cases}
$$

比 PPO 的 asymmetric clip 更激进——**clip 外的 token 直接 mask gradient**（而非截断到边界）。直觉：异步场景下 $\rho$ 的极端值代表严重 off-policy，与其 trust 一个被截断的 noisy gradient，不如直接丢弃。

#### 4.6.3 Off-policy Sample Dropping

记录每条 response 经历的 policy 版本序列 $(w_0, \dots, w_k)$，若 $w' - w_0 > \tau$（当前版本太超前于 rollout 起始版本）则丢弃。

环境崩溃的 sample 也丢弃：group 中若 valid sample > 半数，重复 padding；否则丢整组。

#### 4.6.4 DP-aware Routing

MoE inference 在 Data Parallel 下，同一个 agent 的 multi-turn request 必须路由到同一 DP rank 才能复用 KV cache。

GLM-5 用 **consistent hashing** + 动态负载均衡：rollout ID 映射到固定 DP rank，长期保持。Prefill 成本只与 incremental token 成比例，与 total context 无关。

---

## 5. General RL：三维目标 + Hybrid Reward

### 5.1 三维优化目标

- **Foundational correctness**：指令遵循、逻辑一致、事实正确、无幻觉、语言流畅
- **Emotional intelligence**：empathy、insight、自然语言风格
- **Task-specific quality**：写作、文本处理、问答、role-play、翻译

### 5.2 Hybrid Reward System

三种 reward 互补：

| Reward 类型 | 优点 | 缺点 |
|---|---|---|
| Rule-based | 精确、可解释 | 表达能力受限 |
| ORM (Outcome Reward Model) | 低方差、高效 | 易 reward hacking |
| GRM (Generative RM) | 抗 hacking | 高方差 |

混合三者，rule-based 抓硬约束、ORM 抓整体质量、GRM 抓细粒度判别。

### 5.3 Human-in-the-loop Style Alignment

这是 General RL 的关键创新。纯 model-generated 优化会收敛到 "model-like" 风格（verbose、formulaic）。引入 expert human responses 作为风格锚点，让 model 学到自然写作模式。

直觉：RLHF 早期靠 human preference，后来被 RM 替代。GLM-5 这里把 human response 直接作为 anchor 而非 preference signal——更直接、更难 reward hack。

参考 OpenAI 的 verifier-guided RL：https://openai.com/index/openai-o1-system-card/

### 5.4 On-Policy Cross-Stage Distillation

四阶段 RL 后，previous stage 的能力会 catastrophic forget。最后阶段用 on-policy distillation 恢复。

Advantage 改写为（Eq. 2）：

$$
\hat{A}_{i,t} = \text{sg}\left[\log \frac{\pi_{\theta_{\text{teacher}}}^{\text{infer}}(y_{i,t}|x, y_{i,<t})}{\pi_\theta^{\text{train}}(y_{i,t}|x, y_{i,<t})}\right]
$$

变量：
- $\text{sg}$：stop gradient（`.detach()`），阻止梯度回流到 teacher
- $\pi_{\theta_{\text{teacher}}}^{\text{infer}}$：前序 stage checkpoint 的 inference policy
- $\pi_\theta^{\text{train}}$：当前 student 的 training policy

直觉：student 的 advantage 直接是 teacher 和 student 的 log-prob gap。当 student 与 teacher 一致时 advantage=0，student 比 teacher 低时 advantage 为正（推 student 向 teacher）。

Group size 设为 1（不再需要 group baseline），batch size 1024。整个 distillation 阶段 throughput 极高。

---

## 6. Agentic Engineering 的环境建设

### 6.1 SWE Environments

基于 RepoLaunch 框架，从 real-world GitHub issue–PR 对构建：
- 自动分析 repo 的 installation/dependency setup
- LLM 生成 language-aware log-parsing function
- 提取 Fail-to-Pass (F2P) 和 Pass-to-Pass (P2P) test cases
- 跨 9 种语言（Python、Java、Go、C、CPP、JS、TS、PHP、Ruby）建了 10k+ 可验证环境

参考 RepoLaunch：https://arxiv.org/abs/2505.23419

### 6.2 Terminal Environments 两条 pipeline

**Seed-based**：seed tasks → LLM brainstorm drafts → construction agent 实例化为 Harbor format → refine agent 验证。Docker build accuracy >90%。

**Web-corpus-based**：从 code-relevant web pages 抽取 → coding agent 生成 Harbor task → **自验证闭环**（constructing agent 同时是 first-pass evaluator）→ 失败则迭代修复。

参考 Harbor：https://github.com/HarborFramework

### 6.3 Search Agent：Web Knowledge Graph

构建 pipeline：
1. 早期 search agent 跑 trajectory，收集 2M+ URL
2. LLM 做 entity recognition + structured extraction → Web Knowledge Graph
3. 采样 low-to-mid frequency entity 作为 seed，扩展 multi-hop 邻域
4. 转换 subgraph 为隐式 multi-entity 关系链问题
5. 三阶段难度过滤：tool-free reasoning 模型答对→剔除；early agent 几步解出→剔除；bidirectional verification 不一致→剔除

### 6.4 Hierarchical Context Management

BrowseComp 在 100K+ context 下精度显著下降。GLM-5 用 hybrid 策略：

**Keep-recent-k**：轨迹 $(q, r_1, a_1, o_1, \dots, r_n, a_n, o_n)$ 中，$o_i$（observation）只保留最近 $k=5$ 轮，更早的折叠为 "Tool result is omitted to save tokens"。

**Hierarchical combination**：keep-recent 持续应用，当总 context 超 $T=32K$ 时触发 Discard-all，整个 tool-call history 清空重启。

Figure 8 结果：单独 Discard-all vs +keep-recent-k 在所有 compute budget 下都有增益，最终 BrowseComp 从 55.3% → 75.9%（开源 SOTA）。

直觉：keep-recent 解决 short-term memory 丢失，discard-all 解决 long-context 退化，两者覆盖不同时间尺度。

### 6.5 Slide Generation：三级 Reward

这是 paper 里非常 novel 的 agent task。

**Level-1：静态 markup 属性**——positioning、color、typography 等 declarative attributes 的 rule-based reward，detect hallucinated/duplicate images。

**Level-2：runtime rendering 属性**——DOM 节点渲染后的 bounding box、width/height。需要 distributed rendering service 实时提取。这里出现了 reward hacking：

- Type 1：硬截断超长内容
- Type 2：spacing 过度操纵

Figure 9 展示了这些 hacking 案例。修复 renderer 实现堵漏洞。

**Level-3：视觉感知特征**——abnormal whitespace detection 等。

训练策略：
- Dynamic sampling：丢弃 structurally trivial samples，集中训练难样本
- Token-level policy gradient loss（参考 DAPO：https://arxiv.org/abs/2503.14476）
- Balancing：同 sample 不同 rollout 分散到不同 batch

Rejection sampling + masking refinement：
- Best-of-N 选最高质量
- 部分 trajectory 只有个别页面缺陷，自动 mask 缺陷页保留其余，减少数据浪费

效果：16:9 比例符合率 40% → 92%；vs GLM-4.5 win rate 67.5%。

---

## 7. RL 训练基础设施：slime

slime 是 GLM 自家的 post-training infra，关键设计：

### 7.1 Server-based Rollouts via HTTP API

Rollout server 和 router 暴露 HTTP endpoint，外部 agent framework 直接调用。**Rollout 逻辑完全 decoupled from training process**——任何 agent 框架（OpenHands、Cline、Claude Code 等）都能挂载。

### 7.2 尾延迟优化

RL rollout 不优化 throughput，优化 **end-to-end latency 由最慢样本决定**。

- **Multi-node inference with DP-attention for MLA**：EP64+DP64 over 8 nodes，提供足够 distributed KV cache，避免排队
- **FP8 rollout + MTP**：FP8 降 per-token latency，MTP 在 small-batch decoding（RL rollout 典型场景）下收益最大
- **PD disaggregation**：prefill 和 decode 分配到不同资源，避免长 prefill 抢占 decode

直觉：传统推理优化 throughput（batch 越大越好），RL rollout 优化的是单条 trajectory 完成时间——batch 小但 tail 长尾决定 wall-clock。

### 7.3 Heartbeat 容错

Rollout server 周期性发心跳，unhealthy server 被 deregister，retry 自动路由到健康 server。这避免单点失败中断整个 RL 训练。

---

## 8. 评测：CC-Bench-V2 与 Agent-as-Judge

### 8.1 CC-Bench-V2 三大维度

**Frontend**：220 个 task，覆盖 HTML/React/Vue，700+ check items。两阶段评测：
1. Static verification（build & run）
2. Agent-as-Judge（GUI agent 用 Playwright 模拟用户交互）

**Backend**：85 个 task，6 种语言，每 task 5-10 单元测试，all-or-nothing Pass@1。

**Long-horizon**：
- Large Repo Exploration：在 GitHub 高星 repo 中定位目标文件（至少 3 层目录深、opaque 名字、不在主 feature surface）
- Multi-step Chained Tasks：从 merged PR 中挖掘 3-15 commits 的 chain，dynamic programming 切分语义组，agent 顺序执行，cumulative test 验证

Table 8 关键结果：

| Task | Metric | GLM-5 | GLM-4.7 | Claude Opus 4.5 |
|---|---|---|---|---|
| HTML ISR | | 38.9 | 35.4 | 52.2 |
| React CSR | | 71.0 | 49.4 | 70.7 |
| Backend Pass@1 | | 25.8 | 19.6 | 26.9 |
| Repo Exploration | | **65.6** | 47.8 | 64.5 |
| Chained Tasks | | 52.3 | 43.0 | 61.6 |

GLM-5 在 Repo Exploration 上甚至超过 Claude Opus 4.5（65.6 vs 64.5）——这很 surprising，因为 Claude 一向以 coding 见长。Paper 解释：repo exploration 不依赖 raw code generation，更依赖 strategic search 和 semantic association，GLM-5 的 agentic tool-use trajectory training 有优势。

### 8.2 Agent-as-Judge 可靠性验证

- Point-wise consistency：130 check items 人 vs agent 评分一致率 94%
- Ranking consistency：8 个 frontier model 排名 Spearman correlation 85.7%

这验证了 GUI agent-as-judge 的可靠性。

### 8.3 SWE-rebench（动态评测）

SWE-bench Verified 已发布 2 年，可能被 contamination。SWE-rebench 持续挖掘 fresh GitHub issue-fixing task。

Table 9：GLM-5 resolved rate 42.1%，Claude Opus 4.6 52.9%，GPT-5.2 xhigh 51.7%。GLM-5 在 dynamic benchmark 上仍能 generalize，说明没有 overfit 静态 benchmark。

参考 SWE-rebench：https://arxiv.org/abs/2505.20411

---

## 9. 国产芯片适配（Ascend 案例）

### 9.1 W4A8 Mixed-Precision

在单台 Atlas 800T A3 上跑 750B 模型：
- Attention/MLP：W8A8 (INT8)
- MoE experts：**W4A8 (INT4)** —— 大幅降显存

用 QuaRot 抑制 outlier（https://arxiv.org/abs/2402.02550），Flex_AWQ_SSZ 做 scaling calibration。

### 9.2 Fusion Kernels

- **Lightning Indexer**：score 计算 + ReLU + TopK 融合单 kernel，NPU 上 overlap compute 和 memory access
- **Sparse Flash Attention**：针对 GLM-5 sparse pattern 优化，TopK selection 和 sparse attention 并行
- **MLAPO**：13 个 pre-processing 小算子融合成 1 个 super operator，Vector/Cube 单元并行

### 9.3 推理引擎优化

vLLM-Ascend 和 SGLang 双引擎适配：
- D2H sampling copy 与 next decode step 重叠
- RadixCache (prefix sharing) + Prefix Cache (扩展到 RAM)
- Attention DP + MoE EP 混合并行
- FlashComm：AllReduce 拆分隐藏 latency
- MTP 提高 NPU 计算密度

单节点性能接近 dual-GPU 国际集群，长序列场景部署成本降 50%。

---

## 10. Easter Egg：Pony Alpha 匿名发布

GLM-5 在 OpenRouter 匿名发布为 "Pony Alpha"。社区猜测分布：
- Claude Sonnet 5: 25%
- DeepSeek: 20%
- Grok: 10%
- 其他: 45%

这是 brand-agnostic 的 blind test。当被揭示是 GLM-5 时，社区惊讶中国 LLM 能达到 frontier 水平。

参考 OpenRouter：https://openrouter.ai/

---

## 11. 与同期工作的关联

### 11.1 vs DeepSeek-V3.2

- 都用 MLA + DSA + MTP
- DeepSeek-V3.2 在 H800 roofline 上调 head 数，GLM-5 改 MLA-256 适配多硬件
- DeepSeek 的 IcePop 没有，GLM-5 用 modified IcePop（去 KL）
- DeepSeek-V3.2 用 943.7B token 训 DSA，GLM-5 只用 20B（continued pre-training 复用 dense base）

### 11.2 vs Kimi K2.5

- K2.5 偏 visual agentic，GLM-5 偏 SWE/terminal
- K2.5 在 HLE with tools 上 51.8，GLM-5 是 50.4
- BrowseComp with context management：K2.5 74.9，GLM-5 75.9

参考 Kimi K2.5：https://arxiv.org/abs/2602.02276

### 11.3 vs Qwen3

- Qwen3 也用 on-policy distillation，GLM-5 的 cross-stage distillation 思路一致
- Qwen3 用 GRPO，GLM-5 用 GRPO + IcePop
- Qwen3 没有 DSA，GLM-5 的 long-context 优势由此而来

参考 Qwen3：https://arxiv.org/abs/2505.09388

### 11.4 vs Claude Opus 4.5

- Claude 在 chained tasks 上仍领先（61.6 vs 52.3）——长程 consistency 仍需提升
- Claude 的 SWE-bench Verified 80.9 vs GLM-5 77.8
- GDPval-AA Elo：Claude 1400，GLM-5 1409（GLM-5 略胜）

---

## 12. 直觉总结：GLM-5 的核心 Thesis

1. **Architecture efficiency 通过 redesign 而非 shrink**：MLA-256、Muon Split、MTP parameter sharing 都是"重新设计组件以适配优化器/硬件"，而非简单降参
2. **DSA 是 lossless 的，SWA/linear attention 是 lossy 的**：这是 paper 在 ablation 中明确传达的信息，DSA indexer 不丢信息
3. **Asynchronous RL 不是工程优化，是 enabling technology**：没有 TITO + direct importance sampling + DP-aware routing，长 horizon agent RL 根本跑不起来
4. **Preserved thinking 把 reasoning 当 state**：这是从 "thinking as generation" 到 "thinking as memory" 的范式转换
5. **Reward 的 hybrid + human anchor 是抗 hacking 的根本**：rule + ORM + GRM 三层，加上 human response 作为 style anchor
6. **Dynamic benchmark (SWE-rebench) 比 static benchmark 更可信**：避免 contamination
7. **Brand-agnostic blind test (Pony Alpha) 是新的 evaluation philosophy**：让模型能力说话而非品牌说话

---

## 13. 我会追问的问题

如果我是 reviewer，我会问：

1. **Muon Split 为什么在 MLA 上有效但在 GQA 上不需要**？paper 没给 ablation。直觉是 MLA 的 $W^{UK}$ 同时承担 head specialization 和 rank compression，split 让两者解耦，但缺实验证据。

2. **Preserved thinking 在 turn 数极多（>50）时如何处理 context overflow**？是 cap 旧的 thinking 还是 summarize？paper 没细说。

3. **DSA indexer freeze 在 RL 阶段，但 base model 变化导致 indexer 输入分布 drift**，是否会有 train-infer mismatch 累积？

4. **Asynchronous RL 的 $\tau$（policy version lag threshold）如何选**？太大 off-policy 严重，太小 rollout 浪费。paper 没给 sensitivity analysis。

5. **Slide generation 的 Level-2 reward 在 distributed rendering 下 latency 如何**？是否成为 RL training 的 bottleneck？

6. **Pony Alpha 期间收集的 user feedback 是否回灌训练**？paper 没明确。

---

希望这个解析帮到你 build intuition，Andrej。这篇 paper 真正的 contribution 不在单个 trick，而是 **整个 stack 从 architecture 到 RL infra 到 agent environment 的协同 redesign**——这跟 Tesla Autopilot 的 system-level thinking 很像，单个组件看似 incremental，但 end-to-end 的 leverage 极大。

如果你对某个具体 section 想深挖（比如 asynchronous RL 的工程细节、DSA indexer 的训练动力学、或 slide generation 的 reward design），告诉我，我可以再展开。
