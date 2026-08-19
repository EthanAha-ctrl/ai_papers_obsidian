---
source_pdf: From RLVR to RLSVR Task Transformation Induces Self-Verifiable Rewards
  for Open-Ended LLM Self-Improvement.pdf
paper_sha256: ef2fd22783a701f74a911c35fab27cb8159011e0598517980ccbcac6f6bc6d3f
processed_at: '2026-08-19T08:23:29-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我们抛开那些复杂的数学公式和学术黑话，用最接地气的人话来聊聊这篇 paper 到底在搞什么名堂。

### 1. 痛点：AI 练“做题”很厉害，练“写作文”没办法

现在的 reasoning model（像 DeepSeek-R1, OpenAI o1）为什么那么强？因为它们在训练的时候，能用标准答案来给自己打分。比如做数学题，算出来是 42，那 reward 就是 1 分，算错了就是 0 分。这叫 **RLVR (Reinforcement Learning with Verifiable Rewards)**。计算机能自动判卷子，所以可以无限刷题、无限变强。

但是，如果让 AI 写小说、写摘要呢？这时候没有标准答案了。计算机没法 deterministic 地判断这篇作文是“好”还是“坏”。以前大家怎么解决？
- 找人来打分（太贵）
- 训练一个打分模型（打分模型自己有 bias，天花板就被锁死了）
- 找一个更强的 AI（比如 GPT-4o）来当裁判（费钱，而且裁判自己也有偏好）

所以，open-ended task（开放式任务）一直没法像数学题那样 scalable 地自我进化。

### 2. 灵感：偷师 Self-Supervised Learning

作者从 BERT 那类自监督学习里找到了灵感。在 BERT 出现前，AI 也没法直接学“理解语言”。BERT 怎么做的？把句子里的几个词挖掉，让 AI 去填空。**挖掉哪些词是 BERT 自己决定的，所以“正确答案”是它自己造出来的。**这就叫 label by construction（标签由构造产生）。

作者就想：我能不能把这个 trick 用到 RLVR 里？我不去纠结怎么给作文打分了，我直接把“写作文”这个任务，**变形**成一个自带标准答案的新游戏。只要在这个新游戏里表现好，就等同于作文质量高。

### 3. SpyRL：用“谁是卧底”来练写作

这篇 paper 提出的具体玩法叫 SpyRL，灵感来自“谁是卧底”或狼人杀。我们把 AI 放到这样一个环境里：

1. **分发文稿**：环境里跑 5 个 AI player。其中 4 个（平民）拿到完整的写作素材，1 个（卧底）拿到的素材是被涂黑、残缺的。
2. **执行任务**：大家都必须根据自己手里的素材，去写同一篇文章（比如写摘要、写故事）。
3. **投票捉鬼**：写完后，大家互相看彼此的 output，然后投票选出谁是那个拿到残缺信息的卧底。

**最神的设计就在这里**：
由于“谁是卧底”是系统在开局时随机指定的，所以系统**心里有数**。AI 们如果投票投对了，系统直接给 1 分；投错了，给 0 分。这完全 verifiable，不需要任何外部裁判！这就解决了 open-ended task 没 reward 的问题。

**那这怎么逼迫 AI 提高写作质量呢？**
- 对于**卧底 AI**：它的信息是残缺的，写出来的东西容易露馅。为了不被大家投票投死，它必须绞尽脑汁，把残缺的部分猜出来，写得尽可能天衣无缝、贴近主题，伪装成平民。
- 对于**平民 AI**：如果它自己写得拉跨，大家就会觉得“这家伙写得这么烂，肯定没拿到完整 prompt，它是卧底！”，它就会被冤枉投死。所以平民 AI 也必须写得极其出彩、深刻，来证明“我才是有完整信息的真命天子”。

你看，这就把不可量化的“作文质量”，巧妙地转化成了“能不能骗过同伴”和“能不能证明自己”。写得越好，越不容易被怀疑。这就把 external reward 变成了环境内部的 zero-sum 博弈。好比你把一群人关在岛上，不教他们怎么造船，只告诉他们谁造的船浮不起来谁就掉海里，他们自然就会拼命把船造好。

### 4. 防止“走火入魔”的几个小机关

虽然思想很漂亮，但直接搞 self-play 很容易崩盘，所以作者加了几个关键工程细节：

- **轮流训练**：今天先让“投票的 AI”变强，明天再让“写作的 AI”变强。如果两边同时变强，会互相干扰，信号全乱了。
- **角色校准**：因为卧底天生信息就吃亏，它拿到差分是应该的。系统会单独给卧底和平民算各自的平均分，比的是“你比你这个角色平时的平均水平强不强”，避免系统老觉得卧底不行。
- **群体决策**：不是一个人当裁判，而是 5 个人一起投票。就算一个人瞎了眼，其他人能把他 balance 回来。这比以前那种单个 proposer 和单个 solver 互搏的方法鲁棒得多。

### 5. 结果如何？

这个设计强得离谱。不仅在没有标准答案的 summarization 和 creative writing 上把之前的 self-play 方法（Absolute Zero, R-Zero 等）按在地上摩擦，甚至在数学这种本来就有标准答案的领域，它也比那些传统的 RLVR 方法分数高。这说明“信息不对称”这种竞争压力，逼出来的推理过程比单纯对答案更严谨。

### 总结

一句话总结这篇 paper 的核心思想：**Verifiability 不是任务自带的属性，而是可以通过设计环境“人工造”出来的。**

以前大家觉得：写作没法强化学习，因为没有裁判。这篇 paper 说：那就换个游戏，让玩家互相当裁判，让环境的规则来当裁判。只要游戏设计得足够精巧，玩家为了赢游戏，自然就练出了你要的能力。这其实打开了 LLM 自我进化的一个巨大口子，以后只要是能设计成“对抗游戏”的任务，都可以零数据、零人工地无限变强。

---

# 这篇 Paper 在讲什么

一句话直觉：作者把 **self-supervised learning (SSL)** 里"通过构造 pretext task 从数据自己里挤出监督信号"的思路，迁移到 **RLVR (Reinforcement Learning with Verifiable Rewards)** 上，从而把没法用规则验证 reward 的 open-ended task（比如 summarization、creative writing）"变形"成一个有 ground-truth 的游戏（who-is-the-spy），用游戏内部规则产生 verifiable reward。

paper 提出框架叫 **RLSVR (RL with Self-Verifiable Rewards)**，具体实例化叫 **SpyRL (Self-PlaY RL)**。代码在 https://github.com/wangqinsi1/RLSVR/tree/SpyRL

---

# 1. 背景：RLVR 的"verifiability bottleneck"

## 1.1 RLVR 的成功与局限

DeepSeek-R1、OpenAI o1、Kimi k1.5 这些 reasoning model 的训练范式可以抽象成：

$$
\max_\theta \; \mathbb{E}_{x\sim\mathcal{D},\; y\sim\pi_\theta(\cdot \mid x,\tau)}\bigl[ V(x,y) \bigr]
\quad\text{(Eq.1)}
$$

变量解释：
- $\theta$ 是 policy $\pi_\theta$ 的参数
- $x\sim\mathcal{D}$ 是从输入分布采样的 prompt
- $\tau$ 是任务指令 (task instruction)，例如 "summarize the following report"
- $y$ 是 policy rollout 出的 output
- $V(x,y)\in\{0,1\}$ 是一个 deterministic verifier（answer checker / unit test）， unbiased、unlimited、essentially free

RLVR 之所以 scalable，关键在于 $V$ 是 rule-based 的，不需要任何 label。但这种 setup 严重受限于 domain——只有 math 和 code 这种有 ground-truth answer 的领域才能用。

## 1.2 open-ended task 的痛点

对于 summarization、creative writing 这种 task，真正的目标是一个 latent quality function $Q(x,y)$，没有任何 $V$ 能直接 check。以前的解决方案：
- **RLHF / DPO**：用 learned reward model approximates $Q$，但 reward model 有 bias，并且把 policy 能力 ceiling 在 reward model 能力上
- **LLM-as-a-Judge** (https://arxiv.org/abs/2306.05685)
- **Self-rewarding LM** (Yuan et al. 2024, https://arxiv.org/abs/2401.10020)
- **Rubric-as-Reward** (Gunjal et al. 2025, https://arxiv.org/abs/2507.17746)
- **Writing-Zero** (Jia et al. 2025, https://arxiv.org/abs/2506.00103)：用 generative reward model 桥接

这些方法的本质都在于**直接逼近 $Q$**，所以都重新引入了 evaluation bias、judge capability bottleneck、额外 inference cost。paper 的核心论点：不要去逼近 $Q$，而是把 task 改造成一个能自动产生 verifiable signal 的 proxy environment。

## 1.3 类比：SSL 是怎么处理"没有 label"的

self-supervised learning (Doersch et al. 2015 https://arxiv.org/abs/1505.05192, Noroozi & Favaro 2016 jigsaw, BERT https://arxiv.org/abs/1810.04805, SimCLR https://arxiv.org/abs/2002.05709, CPC https://arxiv.org/abs/1807.03748) 的核心 trick：
- **不试图构造 missing label，而是把 task 变形**，变形后的 task 自带 label
- BERT 的 masked token recovery：mask 是你自己放的，所以被 mask 的 token 是已知的 label
- Jigsaw：你自己 shuffle patches，所以 ground-truth 排列是已知的
- Contrastive：你自己 crop/augment 一对 positive，所以 positive pair 是已知的

关键性质：**pretext task ≠ downstream task**，但学到的 representation transferable。

---

# 2. RLSVR：把 SSL 的 task-transformation 引入 RLVR

## 2.1 形式化：Task Transformation $\Phi$

$\Phi$ 把原 task $(\mathcal{D},\tau)$ 变成一个 proxy environment $\mathcal{E}$，$\mathcal{E}$ 有四个步骤：

**1. Latent-variable injection**
环境采样 $x\sim\mathcal{D}$ 和一个 latent variable $z$，并 record $z$ 作为 episode ground truth。$z$ 永远不直接 reveal 给 policy。

**2. Conditioned task execution**
环境从 $(x,z)$ 构造 observation $o$，policy 在 $o$ 上执行**原 task** $\tau$，产生 $y\sim\pi_\theta(\cdot\mid o,\tau)$。这保证 $\mathcal{E}$ 里练的能力还是 target task 的能力。

**3. Verifiable interaction**
环境的规则问一个关于 $z$ 的问题，这个问题只能从 task outputs 推断。**关键设计约束**：正确回答这个问题依赖于 step 2 的 output 质量。

**4. Rule-based reward**
$R = \text{check}(\text{interaction outcomes}, z)$。deterministic、rule-based、不需要 annotation。

**为什么叫 "self-verifiable"？** 因为 reward 的 verifiability来自环境内部规则或者 interaction outcomes，而非外部 verifier。ground truth "by construction" 存在——$z$ 是环境自己采样的，关于 $z$ 的 prediction 可以精确 check，就跟 math verifier 检查 final answer 一样。

类比映射：
- SSL pretext task ↔ $\Phi$（task transformation）
- SSL 自动生成的 label ↔ latent variable $z$
- 标准 RLVR machinery（GRPO 等）直接套到 $\mathcal{E}$ 上，$R$ 替代 $Q$

---

# 3. SpyRL：一个具体 instantiation

## 3.1 设计直觉

SpyRL 借鉴社交推理游戏 "Who Is the Spy?"（类似狼人杀、Among Us）。每轮：
- $n-1$ 个 civilian，拿到完整 $x$
- 1 个 spy，拿到 degraded 版本 $g(x)$
- 所有 player 执行**相同**的 target task
- Detection stage：大家看彼此 output，投票选谁是 spy
- 因为 spy 身份 $u$ 是 environment 预先采样的，投票对错**完全 verifiable**
- 同时，output 是否被人怀疑，紧密关联 output 是否暴露了 information deficiency

这个设计同时拿到两个 property：
1. **Reward verifiability**（detection stage）：身份是 environment 指定的，可以精确 check
2. **Capability overlap**（performing stage）：练的就是 target task 的能力

## 3.2 形式化

每个 epoch：
1. 采样 $x\sim\mathcal{D}$
2. 采样 spy index $u \sim \text{Unif}(\{1,\dots,n\})$
3. 给每个 player $i$ 分配 observation：

$$
o_i = \begin{cases} x, & i \neq u \\ g(x), & i = u \end{cases}
\quad\text{(Eq.2)}
$$

其中 $g(\cdot)$ 是 information-degradation operator。例如：
- summarization：continuous span masking, 20% 的 input 被 mask
- creative writing：20% prompt 内容被 mask
- math reasoning：40% source text 被 mask（因为 math 文本冗余度高、结构性强）

$g$ 的设计要满足：**mask 的是 task-relevant content，但保留 style、length、thematic consistency**，避免 detector 走 superficial shortcut。

4. Performing stage：每个 player $i$ 用 performer policy 生成

$$
y_i \sim \pi_\theta^P(\cdot\mid o_i, \tau)
\quad\text{(Eq.3)}
$$

5. Detection stage：每个 player $i$ 看 public output set $Y=\{y_1,\dots,y_n\}$，结合自己 $o_i$ 形成 state $s_i = (o_i, Y)$，用 detector policy 投票：

$$
v_i \sim \pi_\phi^D(\cdot\mid s_i),\quad v_i \in \{1,\dots,n\}
\quad\text{(Eq.4)}
$$

$v_i = j$ 表示 player $i$ 怀疑 player $j$ 是 spy。

## 3.3 Detection stage 的 reward

因为 $u$ 是已知的，detector $i$ 的 base reward 是 indicator：

$$
r_i^D = \mathbb{I}[v_i = u]
$$

为降低 variance、更适合集体决策，用 GRPO-style (Shao et al. 2024, https://arxiv.org/abs/2402.03300) 的 group relative advantage：

$$
\mu_D = \frac{1}{n}\sum_{i=1}^{n} r_i^D,\quad
\sigma_D = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(r_i^D - \mu_D)^2},\quad
A_i^D = \frac{r_i^D - \mu_D}{\sigma_D + \epsilon}
\quad\text{(Eq.5)}
$$

变量解释：
- $\mu_D$：group 内 detector reward 的均值，做 baseline
- $\sigma_D$：group 内 detector reward 的标准差，做 normalization
- $\epsilon$：numerical stability
- $A_i^D$：normalized advantage，不需要 critic network

**关键 insight**：单个 detector 误判不会主导优化轨迹，被其他 player 的投票结果 balance 掉。这是 group-based aggregation 相对于 single verifier 的优势——这点很关键，proposer-solver 框架 (Absolute Zero, R-Zero) 是单 verifier，容易有 bias amplification。

## 3.4 Performing stage 的 reward：Zero-Sum 设计

令：
- $m_u$：spy 收到的票数
- $m_{c_j}$：第 $j$ 个 civilian 收到的票数
- $\bar{m}_c$：所有 civilian 的平均票数
- $n_c = n - 1$：civilian 数量

reward 设计：

$$
r_u^P = -\beta(m_u - \bar{m}_c)
\quad\text{(Eq.6a)}
$$

$$
r_{c_j}^P = \frac{\beta}{n_c}(m_u - \bar{m}_c) - \lambda(m_{c_j} - \bar{m}_c),\quad j=1,\dots,n_c
\quad\text{(Eq.6b)}
$$

变量解释：
- $\beta > 0$：spy 与 civilian 之间的 competitive signal strength
- $\lambda > 0$：civilian 内部 consistency penalty

**两个关键性质**：

**1. Zero-sum constraint**：
$$
r_u^P + \sum_{j=1}^{n_c} r_{c_j}^P = -\beta(m_u - \bar{m}_c) + n_c \cdot \frac{\beta}{n_c}(m_u - \bar{m}_c) - \lambda\sum_j (m_{c_j} - \bar{m}_c) = 0
$$
因为 $\sum_j (m_{c_j} - \bar{m}_c) = 0$。这保证了 spy 和 civilian 之间是 minimax / zero-sum game，类似 GAN (https://arxiv.org/abs/1406.2661) 的结构，能 co-evolve。

**2. Within-group competition**：如果一个 civilian 被投了特别多票（比平均高），它会被 penalty $\lambda$。这把学习信号变成 **relative**：policy 学的不是"绝对好"，而是"比同 group 的其他 player 好"。

直觉：这把 unverifiable 的 $Q(x,y)$ 转化成"在同 group 里相对更好"的代理，这跟 Elo rating、self-play boosting 是一脉相承的。

## 3.5 Role-Advantage Estimation (RAE)

问题：spy 和 civilian 的 information 是结构性不对称的，spy 看到 $g(x)$ 而 civilian 看到 $x$，所以两者的 raw reward 分布结构性偏移。直接用 $r_u^P, r_{c_j}^P$ 做优化会让 optimizer 把"information asymmetry 导致的 reward 差异"误读成"policy 质量差异"，bias 估计。

RAE 借鉴 Liu et al. 2025a (SPIRAL, https://arxiv.org/abs/2506.24119) 的思路：给每个 role 维护一个 exponential moving average (EMA) baseline：

$$
b_u \gets \alpha b_u + (1-\alpha) r_u^P,\quad
b_c \gets \alpha b_c + (1-\alpha) \frac{1}{n_c}\sum_{j=1}^{n_c} r_{c_j}^P
\quad\text{(Eq.9)}
$$

变量解释：
- $\alpha \in [0,1)$：EMA decay rate
- $b_u$：spy role 的 expected reward baseline
- $b_c$：civilian role 的 expected reward baseline

Role-calibrated advantage：

$$
A_u^P = r_u^P - b_u,\quad A_{c_j}^P = r_{c_j}^P - b_c
\quad\text{(Eq.10)}
$$

这样 gradient 反映的是"player 相对于自己 role 的 typical outcome 表现如何"，而不是被 role inherent difficulty confound。

**Ablation 证明 RAE 关键性**（Table 9）：去掉 RAE，Qwen3-4B 在 7 个 benchmark 平均从 50.4 掉到 37.5（甚至低于 base 的 41.4），说明没有 RAE 时 optimizer 会主动损害 model。这是非常重要的细节，说明这种 self-play 训练如果不校准 role bias 会"训练越久越烂"。

## 3.6 两阶段 coupled optimization

performing 和 detection 互相依赖：
- detection 的投票结果定义 performing 的 reward
- performing 的 output 质量定义 detection 的难度

**Alternating optimization** 而不是 joint update。理由（ablation Table 16 给出证据）：
- Joint update：early stage detector 还识别不出 spy，vote noisy；performer 用 noisy reward 更新会劣化 output，detector 又拿更差的 example 训练，形成 positive feedback 的崩溃。Table 16 显示 joint training 在 GSM8K 上从 84.5 掉到 76.8，Math500 从 68.2 掉到 53.1。
- Alternating：先稳定 detector，再用稳定 reward 训 performer，再用更难的 output 推 detector，形成 co-evolution。

状态切换规则（hysteresis thresholds）：
$$
\text{Detection}\to\text{Performing}:\; m_t=0 \wedge \overline{acc}_t \geq \tau_{acc}^\uparrow \wedge \overline{na}_t \leq \tau_{na}^\downarrow
\quad\text{(Eq.11)}
$$

$$
\text{Performing}\to\text{Detection}:\; m_t=1 \wedge \Bigl(1-\overline{acc}_t \geq \tau_{err}^\uparrow \;\vee\; \overline{na}_t \geq \tau_{na}^\uparrow\Bigr)
\quad\text{(Eq.12)}
$$

参数：
- $\tau_{acc}^\uparrow = 0.9$（detector 准确率达到 0.9，说明 performer 太弱了，转去训 performer）
- $\tau_{err}^\uparrow = 0.4$（error 率达到 0.4，说明 detector 太弱）
- $\tau_{na}^\uparrow = 0.5, \tau_{na}^\downarrow = 0.1$（N/A 率：detector 太多 abstain 也算崩溃信号）
- $K_{min}=5$：每阶段最少训 5 次，防止 chattering

$\overline{acc}_t, \overline{na}_t$ 是 acc 和 N/A rate 的 EMA，smoothing factor $\rho$。

这个切换逻辑跟 AlphaZero 里的 curriculum 设计、Auto-curricula (Sukbaatar et al. 2017, https://arxiv.org/abs/1703.05407)、Unsupervised Environment Design (Dennis et al. 2020 PLR, https://arxiv.org/abs/2012.09564) 是同一个直觉：**让 student 在当前能力 frontier 上保持挑战**，避免 equilibrium 太容易 collapse。

## 3.7 GRPO-style 目标函数

Performer loss：

$$
\mathcal{L}_P(\theta) = -\mathbb{E}\Bigl[\frac{1}{n}\sum_{k\in\{u\}\cup\mathcal{C}}\sum_t \min\bigl(\rho_{k,t}^P A_k^P, \text{clip}(\rho_{k,t}^P,1-\epsilon,1+\epsilon)A_k^P\bigr)\Bigr] + \beta_P \text{KL}(\pi_\theta^P \| \pi_{ref}^P)
\quad\text{(Eq.7)}
$$

Detector loss：

$$
\mathcal{L}_D(\phi) = -\mathbb{E}\Bigl[\frac{1}{n}\sum_{i=1}^{n}\min\bigl(\rho_i^D A_i^D, \text{clip}(\rho_i^D,1-\epsilon,1+\epsilon)A_i^D\bigr)\Bigr] + \beta_D \text{KL}(\pi_\phi^D \| \pi_{ref}^D)
\quad\text{(Eq.8)}
$$

变量解释：
- $\rho_{k,t}^P$：performer policy 在 token $t$ 处的新旧 policy 概率比（importance ratio）
- $\rho_i^D$：detector 在 action 级别的概率比
- $\epsilon$：clip ratio，标准 PPO 0.2
- $A_k^P$：performer 的 role-calibrated advantage (Eq.10)
- $A_i^D$：detector 的 group-normalized advantage (Eq.5)
- $\beta_P, \beta_D$：KL penalty 系数
- $\pi_{ref}^P, \pi_{ref}^D$：reference policy（防止 drift）

注意 detector 是 **action-level** ratio（整个 vote 是一个 action），performer 是 token-level ratio。

---

# 4. 实验

## 4.1 三个 domain 的 setup

| Domain | Dataset | Task | Mask ratio | Mask 方式 |
|---|---|---|---|---|
| Summarization | GovReport | 长报告摘要 | 20% | continuous span masking |
| Creative Writing | WritingPrompts | 给 prompt 写故事 | 20% | continuous span masking |
| Math Reasoning | Nemotron-CC-Math-v1 | 基于 source text 构造并解数学题 | 40% | partial context removal |

mask ratio 不同反映了 domain 文本的 redundancy：math 文本冗余度高结构性强，所以 mask 多一点；narrative 不行，mask 太多会让 spy 完全无法对齐。

## 4.2 训练 hyperparameter

- 基座：Qwen3-4B-Instruct-2507 / Qwen3-8B
- RL 算法：GRPO，verl framework
- Learning rate：$1\times 10^{-6}$
- Prompts per batch：128，rollouts per prompt：8 → effective batch 1024
- PPO mini-batch：128，micro-batch per GPU：2
- KL coef $\beta=0.001$，low-variance KL
- 100 epochs
- Max prompt length：12,288；max response：4,096；max model：16,384
- 5 players，1 round
- 8 GPUs, vLLM TP=8, gpu_mem_util=0.45
- Gradient checkpointing + FSDP, reference model offload to CPU

## 4.3 主结果

### Summarization (Table 1)
Qwen3-4B + SpyRL vs Absolute Zero：
- GovReport ROUGE-L：33.2 → 36.7
- Multi-News：25.2 → 26.4
- QMSum：22.7 → 25.3
- VCSum：18.3 → 19.1
- SAMSum：46.1 → 48.2

A/B test (GPT-4o, swap-order mitigate position bias) 在 30 个 cell 中全部胜出。Qwen3-8B 上 SpyRL 平均 win rate 75.4%。

### Creative Writing (Table 2)
SpyRL 在 Novelty、Emotion、Coherence、Consistency、Overall 上都胜过 R-Zero 和 Absolute Zero。最大的 gain 在 Novelty 和 Emotion（最 subjective 的维度），说明不只是 fluency 提升。

### Math reasoning (Table 3)
SpyRL 在 GSM8K、Math500、AIME 24、AIME 25、Minerva、MMLU-Pro、GPQA-D 上全部最好。在 AIME 25 上 Qwen3-4B 从 6.7 → 20.0，提升显著。这值得注意——SpyRL 在 verifiable task 上居然也打败 R-Zero 和 Absolute Zero，说明 self-play + group-based design 比 single proposer-solver 在 verifiable domain 也提供更细粒度、更稳定的信号。

### 更难 benchmark (Table 15, Appendix D.2)
AMC、Olympiad-Bench、SuperGPQA 上 SpyRL 全胜，平均比 backbone +6.7，R-Zero 只 +1.6，Absolute Zero +3.6。R-Zero 在 AMC 上甚至 < backbone，说明 proposer-solver curriculum 没法迁移到 competition-style problem。

## 4.4 Reward alignment 实验（Figure 4）

这是证明 "self-verifiable reward 跟实际 quality aligned" 的关键实验：跑 100 game，记录每个 player 收到的票数，同时让 GPT-4o rank 5 个 output 的 quality (1=best, 5=worst)，然后计算收到不同票数的 player 的平均 GPT-4o rank。

结果：**收到票数越多 → GPT-4o rank 越差（rank 数字越大）**，正相关。这验证了"被怀疑程度 = output 质量差"这个关键 assumption，是整个 framework 的基石。

## 4.5 Cross-task transfer (Table 7)

- Summarization → Creative Writing：positive transfer (Overall 59.1%)
- Creative Writing → Summarization：positive transfer (52-56%)
- Math → Summarization / Writing：negative transfer

直觉对齐：summarization 和 writing 共享 discourse coherence、long-range consistency 能力；math 主要练 symbolic manipulation 和 multi-step reasoning，跟写作 stylistic 能力 overlap 小。

## 4.6 Ablation

- **Alternating optimization 关键**（Table 8）：Only Performing 初期涨，后期 plateau + oscillate；Only Detection 几乎无 gain；Without spy 也 plateau；SpyRL full 才能持续 gain。
- **Group size**（Figure 5）：n=3→5 平均 gain 5.5→9.3，n=6,8 边际递减，n=5 已经够复杂。
- **Mask ratio 不敏感**（Table 10）：20% vs 40% 在 summarization 上几乎无差，说明 RAE 把 reward re-center 到 role-specific baseline 后，整体 game 难度不影响学习信号。
- **RAE 是 must**（Table 9）：去掉 RAE 训练反而损害 model。

---

# 5. 相关工作的联想脉络

## 5.1 Self-Play 谱系

- **TD-Gammon** (Tesauro 1995)：最早 self-play 成功
- **AlphaGo / AlphaGo Zero / AlphaZero** (Silver et al. 2016, 2017, 2018 https://www.nature.com/articles/nature16961, https://www.nature.com/articles/nature24270, https://www.science.org/doi/10.1126/science.aar6404)：zero-data self-play + MCTS
- **OpenAI Five** (Berner et al. 2019 https://arxiv.org/abs/1912.06680)：大规模 multi-agent self-play
- **AlphaStar** (Vinyals et al. 2019 https://www.nature.com/articles/d41586-019-03299-0)：StarCraft
- **Asymmetric self-play / PAIRED** (Sukbaatar et al. 2017 https://arxiv.org/abs/1703.05407, Dennis et al. 2020 https://arxiv.org/abs/2012.09564)：asymmetry 自动产生 curriculum

SpyRL 是把这套迁到 LLM。在 LLM 上 self-play：
- **SPIN** (Chen et al. 2024 https://arxiv.org/abs/2401.01335)
- **Self-Rewarding LM** (Yuan et al. 2024 https://arxiv.org/abs/2401.10020)
- **Absolute Zero** (Zhao et al. 2025 https://arxiv.org/abs/2505.03335)：proposer + solver 零数据自循环，主要是 math/code
- **R-Zero** (Huang et al. 2025 https://arxiv.org/abs/2508.05004)
- **SPIRAL** (Liu et al. 2025a https://arxiv.org/abs/2506.24119)：零和 self-play 促 reasoning，RAE 来自这里
- **SPICE** (Liu et al. 2025b https://arxiv.org/abs/2510.24684)：corpus environment
- **SPAG** (Cheng et al. 2024 https://arxiv.org/abs/2410.06180)：adversarial taboo
- **SPELL** (Yang et al. 2025 https://arxiv.org/abs/2509.23863)：long-context evolution
- **SPC** (Chen et al. 2025a https://arxiv.org/abs/2504.19162)：evolving critic
- **Prover-Verifier Games** (Kirchner et al. 2024 https://arxiv.org/abs/2407.13692)：让 output legibility 通过 adversarial 提升
- **Vision-Zero** (Wang et al. 2025 https://arxiv.org/abs/2509.25541)：把这个 trick迁到 VLM

SpyRL 的差异：designed for **non-verifiable domain**。其他 proposer-solver 都还是隐含假设 solver 能 verify，SpyRL 把 verify 替换成"环境注入的 latent variable"。

## 5.2 Multi-Agent Debate / AI Safety via Debate

- **AI Safety via Debate** (Irving et al. 2018 https://arxiv.org/abs/1805.00899)：用 debate 机制对齐 AI，跟 SpyRL 的"detect 信息劣势方"思想有 structural 类似
- **Multi-agent debate for factuality** (Du et al. 2024 https://arxiv.org/abs/2305.14325)
- **ChatEval** (Chan et al. 2023 https://arxiv.org/abs/2308.07201)：multi-agent deliberation 评估
- **Persuasion debate** (Khan et al. 2024 https://arxiv.org/abs/2402.06782)
- **Social deduction with MARL** (Sarkar et al. 2025 https://arxiv.org/abs/2502.06060)：Among Us 风格

但这些大多在 **inference-time** 用 debate 提升 factuality，SpyRL 是把它转成 **training reward**。

## 5.3 RL beyond verifiable domains

- **RLHF / InstructGPT** (Ouyang et al. 2022 https://arxiv.org/abs/2203.02155)
- **Constitutional AI** (Bai et al. 2022 https://arxiv.org/abs/2212.08073)
- **RLAIF** (Lee et al. 2023 https://arxiv.org/abs/2309.00267)
- **DPO** (Rafailov et al. 2023 https://arxiv.org/abs/2305.18290)
- **Process Reward Models** (Lightman et al. 2023 https://arxiv.org/abs/2305.20050, PRM800K https://github.com/openai/prm800k)
- **Rubrics as Rewards** (Gunjal et al. 2025 https://arxiv.org/abs/2507.17746)
- **Writing-Zero** (Jia et al. 2025 https://arxiv.org/abs/2506.00103)：用 generative reward model 给 creative writing 提供 reward

这条路是用 learned/approximate reward 替代 $Q$。SpyRL 反向操作：构造环境让 ground truth by-construction 存在。

## 5.4 Self-supervised learning 谱系

- **Context prediction** (Doersch et al. 2015 https://arxiv.org/abs/1505.05192)
- **Jigsaw puzzle** (Noroozi & Favaro 2016)
- **Colorization** (Larsson et al. 2016)
- **Rotation prediction** (Gidaris et al. 2018)
- **Contrastive: CPC, SimCLR, MoCo** (https://arxiv.org/abs/1807.03748, https://arxiv.org/abs/2002.05709, https://arxiv.org/abs/1911.05722)
- **Masked LM: BERT, MAE** (https://arxiv.org/abs/1810.04805, https://arxiv.org/abs/2111.06377)

SpyRL 是把 SSL 思想用到 RL 的 reward 构造上——这是 paper 的核心 metaphor。

---

# 6. Intuition：为什么这个 trick 工作

## 6.1 核心洞察

**Key insight**：reward verifiability 不需要是 task 的 intrinsic property，可以通过 task transformation **engineered** 进来。

这个想法很有意思，类似 cryptographic commitment：你 secret 在 environment 一侧 sample 出来，commit 到 environment 状态里，policy 永远看不到 secret，但能输出关于 secret 的 prediction，prediction 可以 deterministic check。

这跟 zero-knowledge proof、commitment scheme 有 conceptual 联系，只不过这里环境扮演"verifier"角色，policy 是"prover"。

## 6.2 为什么 "spy detection ≈ output quality"？

关键假设：spy 的 information deficiency 会**必然**在 output 上留下 subtle 痕迹，detector 能 pick up。

这本质上是 **information-theoretic argument**：spy 的 output 分布 $P_{spy}(y|g(x))$ 跟 civilian 的 $P_{civ}(y|x)$ 在 KL divergence 上不为 0，只要 detector 学会区分这两个分布的 feature，就能 detect。

这跟 **GAN** (Goodfellow 2014 https://arxiv.org/abs/1406.2661) 的 discriminator-generator 是同一个原理，只不过这里 generator 是被 information asymmetry "forced" 生成 slightly-off distribution 的 player，discriminator 是 detector。

Figure 4 的实验直接经验上验证了这个 assumption：票数跟 GPT-4o rank 强正相关。

## 6.3 为什么 zero-sum 是好的？

Zero-sum 保证 minimax game，类似 GAN、AlphaGo 的 self-play。它有两个优点：
1. **No degenerate solution**：spy 永远有 incentive 伪装得更好，civilian 永远有 incentive 不被冤枉，detector 永远有 incentive 找对。
2. **Intrinsic curriculum**：随着 spy 学会伪装，detector 被迫提升；随着 detector 提升，spy 被迫伪装更精巧。

paper 提到 self-play 的常见 failure mode：长期训练后 degenerate (Chae et al. 2025 https://arxiv.org/abs/2510.27072, Shafayat et al. 2025 https://arxiv.org/abs/2505.21444)。alternating optimization + RAE 是 paper 提出的 anti-degeneration 机制。

## 6.4 局限性与潜在失败模式

我自己觉得几个值得 concern 的点：

1. **Latent variable $z$ 的 expressive power**：SpyRL 把 $z$ 限制成"谁是 spy"（n 选 1），这只有 $\log_2 n$ bits 信息量。task quality $Q(x,y)$ 是高维的，spy detection 这个 proxy 可能只 capture 了 quality 的某些 axis（比如 "information completeness"），其他 quality axis 可能学不到。
   - 这跟 SSL pretext task 的著名 issue 一样：pretext task 学到的不一定 align downstream task。BERT masked LM 学到 syntactic 多于 semantic。
   - paper 的 cross-task transfer 实验 (Table 7) 部分缓解这个 concern，但只测了几个 task。

2. **Mask operator $g$ 的设计**：要保留 style/length/theme 但 mask task-relevant content。这需要 task-specific 工程。paper 说 ablation 显示 ratio 不敏感（Table 10），但 $g$ 的 **form** 敏感性没测——比如换成 random token 替换 vs span masking 可能差很多。

3. **Detection 的"shortcut"风险**：如果 detector 学到看 length / lexical marker 之类的 superficial feature，performer 会被训得"看上去对"而不是"真的对"。paper 强调 prompt 里加 quality rubric，但这是 prompt engineering 层面的缓解，不是 fundamental 解决。

4. **Compute 成本**：n=5 player，每轮要做 n 次 performing generation + n 次 detection generation，相当于一次 rollout 做 2n 次 generation。比 standard RLVR 贵 ~10×。虽然不需要 external verifier，但 self-play 本身的 generation cost 不低。

5. **Equilibrium collapse**：长时间训练后 spy 可能学到 perfectly mimic civilian distribution，detector 学不到东西，整个游戏 collapse。alternating optimization + RAE 是缓解，但 paper 只跑了 100 epoch，更长训练是否 stable 没测。

6. **Quality 的 multidimensional nature**：creative writing 的 Novelty 和 Emotion 都被 improve 了，但 paper 没测比如 factual accuracy、harmfulness 之类的 dimension。如果 spy 通过 hallucinate 信息来补 mask 掉的内容，可能学到 factually unsound 的"伪装"行为。

## 6.5 跟其他 paradigm 的关系

- **跟 RLHF 的对比**：RLHF 用 human preference 训 reward model，cost 高 + bias 来自 reward model 能力 ceiling。SpyRL 不需要任何 human label，但需要 task-specific 的 $g$ 设计。
- **跟 Self-Rewarding LM 对比**：self-rewarding 让 actor 自己当 judge，actor 和 evaluator 耦合，会 self-reinforcing bias。SpyRL 的 "judge" 是 multi-agent 投票，group-based aggregation 缓解单点 bias。
- **跟 Absolute Zero/R-Zero 对比**：proposer-solver 框架假设 solver 能 verify。在 open-ended task 上 solver 没法 verify，所以 R-Zero 在 summarization 上 win rate 51.9%（基本没涨）。SpyRL 把 verifier 从 solver 改成 environment-injected identity，绕开了这个假设。
- **跟 Prover-Verifier Games** (Kirchner 2024 https://arxiv.org/abs/2407.13692) 对比：PVG 也是 prover-verifier adversarial，但目标是 legibility。SpyRL 是 correctness-via-stealth-detection。

---

# 7. 一些我想深入讨论的点

## 7.1 公式 (6) 的另一种推导

zero-sum reward 还有一个更直观的解读：可以看作 **contrastive loss** 的离散版。

设 $m_u$ 是 spy 的 "badness score"（被怀疑越多越 bad），$\bar{m}_c$ 是 civilian 的 baseline badness。Eq.6a 的 $r_u^P = -\beta(m_u - \bar{m}_c)$ 等价于说：spy 的 reward 是它"比 civilian 平均多被怀疑多少"的负值。

这跟 InfoNCE loss $\log\frac{\exp(\text{sim}(z_i,z_j^+)/\tau)}{\sum_k \exp(\text{sim}(z_i,z_k^-)/\tau)}$ 在 spirit 上类似：让正样本 pair 相似度相对负样本 pair 更高。这里 spy 是 negative example，civilian 是 positive example，detector 是 similarity function。

可以参考 CPC (https://arxiv.org/abs/1807.03748)、SimCLR (https://arxiv.org/abs/2002.05709)、MoCo (https://arxiv.org/abs/1911.05722)。

## 7.2 跟 Inverse RL 的概念关系

Inverse RL (Ng & Russell 2000 https://www.ai.mit.edu/people/russell/papers/icml00-abs.pdf) 是从 expert demonstration 反推 reward。SpyRL 反过来：reward 是 environment-injected latent，policy 从 reward signal 反推 latent。结构上有点像 **structured prediction** 里的 latent variable model。

## 7.3 跟 Generative Adversarial Network 的对应

- Generator = performer（spy）+ civilian
- Discriminator = detector
- Latent code = $z$（spy identity）
- 数据 = $x$

GAN 的 training instability 经典问题 SpyRL 也会遇到，所以 alternating optimization、RAE（role-specific baseline，类比 feature matching）都是 GAN stability trick 的 LLM 版本。

GAN stability 经典 trick 参考：
- WGAN https://arxiv.org/abs/1701.07875
- Feature matching https://arxiv.org/abs/1606.03498
- TTUR https://arxiv.org/abs/1706.08500

## 7.4 跟 Curriculum Learning / Auto-curricula

alternating optimization 实际上是自动 curriculum：detector 强 → performer 被推着伪装更好 → detector 难度上升 → detector 训练 → spy 再进化。这是 OpenAI Emergent Tool Use (Baker et al. 2019 https://arxiv.org/abs/1909.07394) 和 PAIRED (Dennis et al. 2020 https://arxiv.org/abs/2012.09564) 的 same idea。

---

# 8. 我对这篇 paper 的整体评价

**优点**：
1. **Concept elegant**：把 SSL 的 task-transformation 思想引入 RLVR，提供了一个 clean 的 generalization framework (RLSVR)，SpyRL 只是其中一个 instantiation。
2. **Empirically strong**：在 open-ended 和 verifiable 两个 regime 都打败 baseline，特别是在 R-Zero 完全失败的 summarization 上大幅提升。
3. **Engineering 细节扎实**：RAE + alternating optimization + zero-sum reward 都是必要的 anti-degeneration 机制，ablation 证明每个都关键。
4. **Cross-task transfer 实验**给人信心：capability 是真的 generalize 而不是 benchmark overfit。

**可以批评的点**：
1. **Latent variable $z$ 的 expressive power 受限**：spy identity 只有 $\log_2 n$ bits。这个 framework 是否能 capture complex quality 还是 open question。
2. **Compute cost**：n=5 player 比标准 RLVR 贵 ~10×，scalability 没讨论。
3. **$g$ 的 form sensitivity** 没充分 ablate。
4. **Long training stability** 只测了 100 epoch。
5. **Detector 学到 shortcut 的风险** 没有深入分析——比如 detector 可能学到看 mask 的 artifact 而不是看 quality。

**Truly 新颖的 contribution**：把 SSL "label by construction" 思想和 RLVR 结合这件事本身是个 elegant 的 conceptual move。一旦理解了这个 move， SpyRL 只是一个自然 instantiation，**还有大量其他 instantiation 可以探索**。比如：
- 把 $z$ 从"spy identity"换成"哪个 input 被 perturbed"，做 robustness 训练
- 把 $z$ 换成"哪条 reasoning chain 被截断"，做 reasoning step quality 训练
- 把 $z$ 换成"哪个 citation 是 hallucinated"，做 factuality 训练
- 把 $z$ 换成"哪段 dialogue 是 gaslighting"，做 alignment 训练

这些都可以变成新 paper。RLSVR 这个 abstraction 本身比 SpyRL 更有价值。

---

# 9. 相关 link 汇总

**Main paper & code**
- Paper PDF: https://github.com/wangqinsi1/RLSVR/tree/SpyRL
- Code: https://github.com/wangqinsi1/RLSVR/tree/SpyRL

**RLVR & reasoning models**
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- OpenAI o1: https://arxiv.org/abs/2412.16720
- Kimi K1.5: https://arxiv.org/abs/2501.12599
- GRPO / DeepSeekMath: https://arxiv.org/abs/2402.03300

**RLHF / preference learning**
- InstructGPT / RLHF: https://arxiv.org/abs/2203.02155
- Constitutional AI: https://arxiv.org/abs/2212.08073
- RLAIF: https://arxiv.org/abs/2309.00267
- DPO: https://arxiv.org/abs/2305.18290
- Self-Rewarding LM: https://arxiv.org/abs/2401.10020
- LLM-as-a-Judge: https://arxiv.org/abs/2306.05685
- Process Reward Models / Let's Verify Step by Step: https://arxiv.org/abs/2305.20050
- PRM800K: https://github.com/openai/prm800k
- Rubrics as Rewards: https://arxiv.org/abs/2507.17746
- Writing-Zero: https://arxiv.org/abs/2506.00103

**Self-supervised learning**
- Context prediction: https://arxiv.org/abs/1505.05192
- BERT: https://arxiv.org/abs/1810.04805
- SimCLR: https://arxiv.org/abs/2002.05709
- MoCo: https://arxiv.org/abs/1911.05722
- CPC: https://arxiv.org/abs/1807.03748
- MAE: https://arxiv.org/abs/2111.06377

**Self-play (game AI)**
- AlphaGo: https://www.nature.com/articles/nature16961
- AlphaGo Zero: https://www.nature.com/articles/nature24270
- AlphaZero: https://www.science.org/doi/10.1126/science.aar6404
- OpenAI Five: https://arxiv.org/abs/1912.06680
- AlphaStar: https://www.nature.com/articles/d41586-019-03299-0
- Asymmetric self-play (Sukbaatar): https://arxiv.org/abs/1703.05407
- Unsupervised env design (PLR): https://arxiv.org/abs/2012.09564
- Emergent tool use: https://arxiv.org/abs/1909.07394

**Self-play for LLMs**
- SPIN: https://arxiv.org/abs/2401.01335
- Absolute Zero: https://arxiv.org/abs/2505.03335
- R-Zero: https://arxiv.org/abs/2508.05004
- SPIRAL: https://arxiv.org/abs/2506.24119
- SPICE: https://arxiv.org/abs/2510.24684
- SPAG: https://arxiv.org/abs/2410.06180
- SPELL: https://arxiv.org/abs/2509.23863
- SPC: https://arxiv.org/abs/2504.19162
- Prover-Verifier Games: https://arxiv.org/abs/2407.13692
- Vision-Zero: https://arxiv.org/abs/2509.25541
- SPAG (Self-playing adversarial language game): https://arxiv.org/abs/2412.18234
- Self-play finetuning: https://arxiv.org/abs/2401.01335

**Multi-agent debate**
- AI Safety via Debate: https://arxiv.org/abs/1805.00899
- Multiagent debate (Du et al.): https://arxiv.org/abs/2305.14325
- ChatEval: https://arxiv.org/abs/2308.07201
- Persuasion debate: https://arxiv.org/abs/2402.06782
- Social deduction MARL (Among Us): https://arxiv.org/abs/2502.06060

**GAN & adversarial training (类比)**
- GAN: https://arxiv.org/abs/1406.2661
- WGAN: https://arxiv.org/abs/1701.07875
- Feature matching GAN: https://arxiv.org/abs/1606.03498
- TTUR: https://arxiv.org/abs/1706.08500

**Self-play degeneration 分析**
- Chae et al. 2025: https://arxiv.org/abs/2510.27072
- Shafayat et al. 2025: https://arxiv.org/abs/2505.21444

**Datasets**
- GovReport: https://arxiv.org/abs/2104.02112
- WritingPrompts: https://arxiv.org/abs/1805.04893
- SAMSum: https://aclanthology.org/D19-5409/

总而言之，这篇 paper 的核心 conceptual move——**verifiability 不是 task 的 intrinsic property 而是可以 engineered 的**——我觉得是非常漂亮且有 generalization potential 的。SpyRL 作为第一个 instantiation 已经 empirically 很强了，但更期待看到后续工作探索 RLSVR 的其他 instantiation，特别是用更高 dimensional latent variable 来 capture 更细粒度的 quality axis。
