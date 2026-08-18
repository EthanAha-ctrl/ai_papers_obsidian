---
source_pdf: FACT Failure-Aware Causal Training for World-Action Models.pdf
paper_sha256: 62b041bdc4a23d4d192a1016e9fa488dd89eba9619a503f5c7a99a30e02596b8
processed_at: '2026-08-18T12:08:25-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FACT 用人话讲

## 故事的开头

假设你在学做饭，你只看过厨师做成功菜品的视频。你学到了"怎么做能成功"，但是你从来不知道"如果油温太高会糊锅"、"如果盐放多了会齁"。

有一天你自己做菜，盐放多了。你的脑子里想象的依然是"这道菜做出来很好吃"——因为你只见过成功的结局。你不知道盐多了到底会发生什么。这就是 **success-biased future hallucination**。

现有的 robot learning 基本都犯这个错。我们给 robot 看的全是 expert demonstrations，robot 学到了 "good action → good future" 的 pattern。但是在 test time，robot 如果 sample 了一个 bad action，它的 world model 还是会 hallucinate 一个 good future，因为它从来没见过 bad outcome 长什么样。

## 为什么不能简单地把失败数据丢进去

你可能会想，那把 failure trajectories 也加进 training data 不就完了？

不行。如果你直接把 failure 当 imitation target，等于告诉 robot "去模仿这些失败的动作"。Tables 2 的 ablation 直接证明了这一点——success rate 从 82% 暴跌到 63%。

这就像教小孩做饭，你给他看糊锅的视频然后说"学着点"，他可能真的学糊锅。

问题的本质是：**failure data 有两种信息，一种是"该怎么动"（behavior），一种是"会导致什么"（consequence）。你想要后者，不想要前者。**

## FACT 的核心 trick

### Trick 1: 把因果顺序倒过来

传统 world model 的逻辑是：先想象未来 → 从未来反推 action。

FACT 倒过来：**先生成 action → 再预测这个 action 会导致什么 future**。

这个看起来只是符号调换，但意义巨大。因为现在 future prediction 是一个 **action-conditioned function**。你给什么 action，它就得 predict 什么 action 的 consequence。它没法再赖皮，没法 ignore 你给的 action 去 hallucinate 一个 success future。

[arXiv:2602.15922](https://arxiv.org/abs/2602.15922) 这种 video-first 方案之所以会 hallucinate，就是因为 future prediction 和 action 是 decoupled 的——model 先 imagine 一个 success future，然后才去找对应 action，action 和 future 之间没有真正的 causal binding。

### Trick 2: Teacher-forcing mask 分离两条路径

这是整个 paper 最 subtle 的地方。

考虑一个 failure trajectory：robot 抓歪了，cube 掉地上。这个 trajectory 里有两个信息：
- **Bad action 本身**（抓歪的角度）——这个你不能让 robot 学
- **Bad action 的 consequence**（cube 掉地上）——这个你希望 robot 学

FACT 的做法是在 transformer 里设置两个 action token slot：
- **A slot**：noisy predicted action，用来训练 action generation。对 failure data，这个 slot 的 loss 被 mask 掉，不更新。
- **G slot**：clean ground-truth action，专门用来 condition world branch（future video + value）。无论 success 还是 failure，G slot 都装着真实执行过的 action，world branch 根据 G 去 predict consequence。

所以对于 failure trajectory：
- Action generation branch (A)：**关掉**，不让 bad action 污染 policy
- Future prediction branch (V, I)：**保留**，让 model 看到"这个 bad action 导致了这个 bad future"

这就是 paper 里那句 "failures teach consequences, not behavior" 的字面意思。

### Trick 3: Failure-aware value target

FACT 还预测一个 task progress value $v_t \in [0, 1]$，表示任务完成度。

对 success action：$v_t = p_{t+H}$（正常 progress）
对 failure action：$v_t = \text{clip}(p_{t+H} - \lambda_{\text{fail}}, 0, 1)$，$\lambda_{\text{fail}} = 1$

意思是：如果你这个 action 导致了 failure，value 直接被罚 1 分。这教会 value head 区分 "这个 action 带来 progress" 和 "这个 action 导致 failure"。

这个 value head 在 inference 时可以用来给 N 个 candidate action 打分，选 value 最高的执行。这就是 **best-of-N sampling**，类似 AlphaGo 里 1-step lookahead 的简化版。

关键洞察：**只有经过 failure data 训练后，value head 才有 discriminative power**。Table 2 里有个 ablation——"Ours + scoring"（不经过 failure training 就用 scoring）只有 79%，比不用 scoring 的 baseline (82%) 还差。因为 success-only 训练的 value head 对所有 action 都给高分，scoring 反而引入噪声。

## 实验结果讲人话

### Simulation (RoboTwin, 50 tasks)

- Motus: 87.8% (latency 1220ms)
- FACT + failure: 87.5% (latency 380ms)
- FACT 几乎追平 Motus，但**快 3.2 倍**

Motus 是 video-first design，inference 时要先 fully denoise future video 才能解码 action，所以慢。FACT 是 action-first，action-only mode 跳过 world prediction，非常快。

### Real-world (5 个 bimanual tasks)

- π₀.5: 88% (用了大规模 robot pre-training)
- FACT + failure + scoring: **92%** (没用 pre-training)
- Motus: 64%

FACT 在 real-world 上**碾压** Motus。这很有意思，因为 Motus 在 simulation 上比 FACT 高一点。可能因为 Motus 的 video-first design 对 real-world 的 visual distribution shift 更敏感，而 FACT 的 action-first design 更 robust。

### 关键 ablation：证明每个设计都有用

- **去掉 video co-training**：82% → 58%，**暴跌 24 个点**。Future prediction 作为 regularizer 对 action generation 极其重要。
- **去掉 causal mask**：82% → 77%。如果不做 teacher-forcing，future prediction 和 action generation decoupled，效果下降。
- **failure data 不 mask action loss**：82% → 63%。**直接证明核心 hypothesis**——failure data 不能当 imitation target。
- **scoring without failure training**：82% → 79%。证明 value head 只有见过 failure 才有用。

### Failure data scaling

Failure data 占比从 0% → 50% → 100%（100% 时占总训练数据 45%），success rate 从 32.7% → 57.3%，**monotonic 上升，没有 saturation**。

这暗示了一个非常 exciting 的方向：online self-improvement。Robot 可以自己 rollout，失败的 trajectory 自动收集起来加入 training，理论上可以无限 scale。这和 RLAIF、self-improvement 的 spirit 是一致的。

### Future prediction quality (Table 4)

这是验证核心 hypothesis 的最直接证据：

- **Failure-rollout future 的 PSNR：19.51 → 25.92**（+6.41 dB，巨大提升）
- **Success-rollout future 的 PSNR：26.12 → 26.08**（几乎不变）

意思是：加 failure data 后，model 在 bad action 条件下能准确 predict bad future（不再 hallucinate success），同时 normal 的 good future prediction 完全不受影响。Failure data 是纯 additive 的，学了新的 conditional distribution P(bad future | bad action)，没有 overwrite 原有的 P(good future | good action)。

Figure 5 的可视化更直观：在 bad grasp 条件下，success-only model 预测的 future 画面里 cube 被抓起来了（hallucination），failure-aware model 预测的 future 里 cube 掉在地上（符合真实）。

## 直觉总结

让我把整个 story 浓缩成一个 analogy：

想象一个医生。只见过治愈病例的医生，给任何病人都预测"会康复"。他开的药方可能是对的也可能是错的，但他的预后判断永远是乐观的——因为他没有见过失败案例长什么样。

FACT 的做法：
1. **让医生先开药，再预测这个药方的后果**（action-first causal order）
2. **失败病例也用来训练预后判断，但不用来训练开药方**（teacher-forcing mask 分离两个 task）
3. **明确告诉医生：这个药方导致病人死亡，预后分数给 0**（failure-aware value target）

这样训练出来的医生：
- 开药方的能力不被失败病例污染
- 预后判断能力大幅提升，能识别 bad prescription
- 可以在多个 candidate 药方里选预后最好的（best-of-N scoring）

这就是 FACT 做的事情，只不过 doctor 换成了 robot policy，prescription 换成了 action chunk，prognosis 换成了 future video + task progress value。

## 为什么这个工作 important

从更宏观的视角看，FACT 触及了 robot learning 的一个根本问题：**如何从 failure 中学习**。

Imitation learning 的传统范式是"只从成功学习"，这限制了 robot 的成长上限。RL 的范式是"从所有 experience 学习"，但 sample efficiency 差且不稳定。FACT 提供了一个中间路径——**从 failure trajectories 里提取 consequence supervision，但不让 failure behavior 污染 policy**。

这个 framework 天然可以 extend 到：
- **Online self-improvement**：robot 自己 rollout，自动收集 failure data
- **DAgger-style correction**：human intervention 提供的 correction signal
- **RL from negative experience**：类似 offline RL 但是更 targeted

Paper 的 conclusion 提到 "future training regimes that include online rollouts, DAgger-style corrections, and reinforcement learning from negative experience"，暗示了这些方向。个人觉得这是最有潜力的 follow-up 方向——如果结合 VLA-RFT [arXiv:2510.00406](https://arxiv.org/abs/2510.00406) 的 online RL framework 和 FACT 的 failure-aware training，可能能做出真正 self-improving 的 robot agent。

从 Karpathy 你之前讲过的 "building general-purpose robot policies" 的视角来看，FACT 提供了一个关键的 missing piece：**如何让 robot 从自己的 mistake 中学习，而不是只从 expert demonstration 中学习**。这是从 imitation learner 走向真正的 self-improving agent 的必要一步。

**参考链接**：
- FACT project page: [https://fact-wam.github.io](https://fact-wam.github.io)
- π₀: [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)
- π₀.5: [https://arxiv.org/abs/2504.16054](https://arxiv.org/abs/2504.16054)
- Motus: [https://arxiv.org/abs/2512.13030](https://arxiv.org/abs/2512.13030)
- Flow Matching: [https://arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)
- VLA-RFT: [https://arxiv.org/abs/2510.00406](https://arxiv.org/abs/2510.00406)
- DreamerV3 (related): [https://arxiv.org/abs/2304.10512](https://arxiv.org/abs/2304.10512)

---

# FACT: Failure-Aware Causal Training for World-Action Models - 深度解析

Karpathy 你好，这篇paper很值得玩味，因为它触及了一个在robot learning里长期被回避的问题：**failure data到底应该怎么用**。大多数imitation learning的工作把failure当成垃圾直接丢掉，少数工作尝试用failure做negative training，但是都没有处理好"imitation"和"consequence prediction"之间的causal conflict。FACT给出了一个相当elegant的解法。让我从intuition开始，逐步build up到technical details。

## 1. The Core Problem: Success-Biased Future Hallucination

### 1.1 现有WAMs的causal order问题

现有的World-Action Models (WAMs) 大致有两类design：

**Video-first WAMs** (DreamZero [arXiv:2602.15922](https://arxiv.org/abs/2602.15922), Cosmos Policy [arXiv:2601.16163](https://arxiv.org/abs/2601.16163))：先imagine一个future video，然后用inverse dynamics model从future video decode出action。这种design好处是有一个strong world prior，但是action decoding依赖一个second-stage network，并且需要future video fully denoised才能控制，latency很高。

**Action-conditioned WAMs** (BagelVLA [arXiv:2602.09849](https://arxiv.org/abs/2602.09849), Fast-WAM [arXiv:2603.16666](https://arxiv.org/abs/2603.16666))：把predicted future作为conditioning signal给action prediction。但是关键问题在于——这些future targets几乎全部来自expert demonstrations。模型看到的pair是 (good action, good future)。

### 1.2 为什么这会出问题

考虑这样一个场景：在test time，policy sample了一个bad action（比如抓取的角度错了）。模型需要预测这个bad action会导致什么样的future。

由于训练时模型只见过 (good action, good future) 的pair，模型实际上没有学到"P(bad future | bad action)"这个conditional distribution。它学到的更接近一个marginal distribution P(future) over expert manifold。

这就是Figure 5展示的success-biased future hallucination：即使你强行把bad action作为condition喂进去，模型还是会hallucinate一个successful grasp的画面（白虚线框标出的地方）。这从信息论角度是合理的——模型没见过bad outcomes，自然无法predict它们。

### 1.3 为什么单纯加failure data没用

你可能会想，那就直接把failure rollouts也加进训练集不就行了？但是这里有个causal trap：

如果你直接把failure trajectories加进imitation learning，等于告诉policy"去imitate这些失败的动作"。Tables 2的ablation "Ours w/ failed-action loss"直接验证了这一点：success rate从82%暴跌到63%。

这其实和DAgger、DAgger-style correction的历史教训是一致的——failure data的label不能直接当作imitation target。需要一种方式把"failure consequence"和"failure behavior"分离。

## 2. FACT的核心insight：Reverse the Causal Order

### 2.1 The Causal Insight

FACT的key insight可以一句话概括：**把action generation和future prediction的causal order对调**。

- 传统WAM: predict future → decode action (video-first)，或者把future作为prior condition给action (future-as-condition)
- FACT: **generate action first → predict future conditioned on that action**

这个reversal看起来只是符号上的变换，但是它带来了一个根本性的变化：现在future prediction是一个**action-conditioned**的function。当你给一个bad action，模型必须predict这个specific bad action导致的future，而不只是一个marginal good future。

### 2.2 这个ordering为什么让failure data可用

考虑training objective的结构：

- 对于success demonstrations $(o_t, \ell, a_{t:t+H}^{expert}, o_{t:t+K}'^{good}, v_t^{high})$：所有三个loss都active——action imitation、future prediction、value prediction。
- 对于failure rollouts $(o_t, \ell, a_{t:t+H}^{bad}, o_{t:t+K}'^{bad}, v_t^{low})$：**mask掉action imitation loss**，但是保留future prediction和value prediction。

因为future prediction是conditioned on $a_{t:t+H}^{gt}$（teacher-forced clean action），所以failure trajectory里的bad action会直接supervise future branch去predict这个bad action的consequence。这就是paper里说的"failure teaches consequences, not behavior"。

这个separation是通过一个**teacher-forcing attention mask**实现的，下面详细讲。

## 3. Model Architecture 详解

### 3.1 Token序列设计

FACT用了一个shared video diffusion transformer (initialized from WAN2.2-5B [arXiv:2503.20314](https://arxiv.org/abs/2503.20314))，处理一个structured token sequence：

$$z = [z_{\mathrm{ref}}^P \parallel z_{\mathrm{pred}}^A \parallel z_{\mathrm{gt}}^G \parallel z_{\mathrm{value}}^V \parallel z_{\mathrm{future}}^I]$$

各变量含义：
- $z_{\mathrm{ref}}^P$：observation prefix tokens。P表示"Prefix"，包含当前帧的多视角RGB observations和language instruction $\ell$。这部分是conditioning，不在denoising范围内。
- $z_{\mathrm{pred}}^A$：noisy predicted-action segment。A表示"Action"，这是flow matching要denoise的action chunk。在training时它是corrupted的expert action（对success）或corrupted的bad action（对failure，但是loss被masked）。
- $z_{\mathrm{gt}}^G$：clean teacher-forced action segment。G表示"Ground truth"，是clean的action token，专门用来condition world branch (V和I)。这是关键设计。
- $z_{\mathrm{value}}^V$：value segment。V表示"Value"，是要denoise的task progress value token。
- $z_{\mathrm{future}}^I$：future-video segment。I表示"Image"，是要denoise的未来K帧的video latents。

### 3.2 为什么需要两个action token (A和G)

这是FACT最subtle的设计。一个naive的设计可能是直接用noisy predicted action A来condition world branch。但是这有两个问题：

1. **Training-inference mismatch**：训练时A是noisy的（因为flow matching需要corrupt target），但是inference时Stage 1出来的action是clean的。如果world branch在训练时只见noisy action condition，inference时换clean condition会有distribution shift。

2. **Failure data的causal conflict**：如果A直接condition V和I，那么failure trajectory的bad action会通过A的梯度同时影响action denoising（你想mask的）和world prediction（你想keep的）。gradient会从V/I流回A，污染action generation。

解决方案是引入G这个**separate clean action slot**。World branch (V和I)只attend to G，不attend to A。A只attend to P（observation prefix）。这样：

- 训练时G = clean ground-truth action（无论success还是failure都是clean的executed action）
- World branch通过G得到clean action condition，预测对应的future和value
- Action denoising branch (A)独立工作，只受P的conditioning，gradient不回流到V/I

这就是Figure 3那张attention mask图的核心。让我解析一下：

### 3.3 Attention Mask 解析（Figure 3）

attention matrix的rows是query tokens，columns是key tokens。Allowed attention的cell被colored。

**Training mask**：
- P行：可以attend到所有（P, A, G, V, I都行，因为P是prefix需要看到所有上下文）。实际上看Figure 3的training mask，P可以attend到所有其他segments。
- A行：可以attend to P和A自己。**关键**：A不能attend to G。这阻止了action denoising branch从teacher-forced clean action那里"作弊"。
- G行：可以attend to P和G。G是input，不需要attend到要预测的内容。
- V行：可以attend to P和**G**（不能attend to A）。这就是teacher-forcing——value预测只看clean action condition。
- I行：可以attend to P和**G**。同理，future video预测只看clean action condition。

**Inference mask**（Stage 2）：
- 这时没有G了，因为没ground truth
- Stage 1：用[P, A_noisy]denoise出clean action $\hat{a}_{t:t+H}$
- Stage 2：把$\hat{a}_{t:t+H}$放到G的位置，然后denoise V和I

这个two-stage inference的关键在于Stage 2复用了Stage 1的prefix key-value cache（paper里提到"prefix key-value caching is used in action-only inference"），所以action-only inference非常fast。

### 3.4 Action Adapter的设计

Paper里提到了一个lightweight action adapter："We attach a lightweight action adapter to robot tokens after the feed-forward network in each transformer block"。

这是为了在shared video backbone基础上给action-specific capacity。这点很重要——如果完全shared backbone，action tokens的representation可能被video pre-training的prior主导，难以适应robot action space。Action adapter是一个small FFN插入到每个transformer block的FFN之后，给action tokens额外的transformation容量。

这种设计和Mixture-of-Transformers (MoT) 设计（Motus [arXiv:2512.13030](https://arxiv.org/abs/2512.13030)用的就是这种）对比：MoT会有一个separate world expert，导致future-outcome losses只影响world expert，不影响action expert。FACT的shared backbone + action adapter设计让future-outcome losses能通过backbone回流影响action generation，paper里说"this shared backbone lets future-outcome losses affect action generation instead of being confined to a separate world expert"。

## 4. Training Objectives 详解

### 4.1 Flow Matching Loss

FACT用flow matching [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)作为denoising objective，统一了action、value、future video的training。对于target modality $x \in \{a, v, I\}$：

$$\mathcal{L}_x = \mathbb{E}_{z_0^x, z_1^x, \tau}\left[\big\| u_\theta^x(z_\tau^x, \tau; z) - (z_1^x - z_0^x) \big\|_2^2\right]$$

变量含义：
- $z_0^x$：clean target token for modality $x$。对于action是expert action（success）或executed bad action（failure，但loss被mask）；对于value是progress target；对于future是future video latents。
- $z_1^x \sim \mathcal{N}(0, I)$：Gaussian noise。注意paper里用的是$(1-\tau)z_0 + \tau z_1$的interpolation，所以$\tau=0$是clean，$\tau=1$是noise。
- $\tau$：flow time，从0到1。
- $u_\theta^x$：predicted velocity field。flow matching训练网络去predict从clean到noise的velocity $z_1 - z_0$。
- $z$：full token sequence in Eq. (5)，作为conditioning context。

Inference时用flow-Euler steps反向积分，从$\tau=1$的noise积分到$\tau=0$的clean target。

### 4.2 Failure-Aware Value Targets

这是FACT的另一个核心设计。Value target定义为：

$$v_t(a_{t:t+H}^{\mathrm{gt}}) = \begin{cases} p_{t+H}, & \text{if success} \\ \mathrm{clip}(p_{t+H} - \lambda_{\mathrm{fail}} \mathbf{1}_{\mathrm{fail}}(t+H), 0, 1), & \text{if fail} \end{cases}$$

变量含义：
- $p_t = G_t / G_T \in [0, 1]$：normalized progress。$G_t = \sum_{k=1}^t r_k$是cumulative progress reward，$G_T$是episode的总return。$p_t = 0$表示episode开始，$p_t = 1$表示完成。
- $p_{t+H}$：在执行action chunk $a_{t:t+H}$之后的normalized progress。
- $\lambda_{\mathrm{fail}} = 1$：failure penalty。当action chunk执行后触发failure，progress target会减1。
- $\mathbf{1}_{\mathrm{fail}}(t+H)$：indicator function，如果failure发生在执行$a_{t:t+H}$期间或之前，则为1。
- $\mathrm{clip}(\cdot, 0, 1)$：clip到[0, 1]区间，避免负值。

在experiments里paper用uniform progress reward简化，所以$p_t = t/T$。

这个target设计的妙处在于：
- 对于success action chunk：value target = $p_{t+H}$，反映真实progress。
- 对于failure action chunk：value target = $p_{t+H} - 1$（clipped到0），意味着failure action的progress贡献被罚为0。

这教会value head区分"这个action带来progress"和"这个action导致failure"。

### 4.3 Joint Loss

Success demonstrations的loss：
$$\mathcal{L}_{\mathcal{D}_s} = w_a \mathcal{L}_a + w_v \mathcal{L}_v + w_I \mathcal{L}_I$$

Failure rollouts的loss：
$$\mathcal{L}_{\mathcal{D}_f} = w_v \mathcal{L}_v + w_I \mathcal{L}_I$$

注意failure的action loss $\mathcal{L}_a$被直接去掉，不是weight为0，是整个term消失。Weights：$w_a = 20$，$w_v = w_I = 1$。Action loss权重这么高（20:1）是因为action chunk的magnitude和video latents、value scalar的scale差很多，需要balance gradient。

Algorithm 1里的mask是：$m_a w_a \mathcal{L}_a + w_v \mathcal{L}_v + w_I \mathcal{L}_I$，其中success时$m_a = 1$，failure时$m_a = 0$。这就是"failure-aware"的具体实现。

## 5. Inference: Two-Stage Denoising + Optional Candidate Scoring

### 5.1 Two-Stage Inference

**Stage 1**: denoise $[P_{\mathrm{state}}, P_{\mathrm{ref}}, A_{\mathrm{noisy}}]$ for $K_{\mathrm{denoise}}$ flow-Euler steps，返回clean action chunk $\hat{a}_{t:t+H}$。

如果只需要action（不需要consequence prediction或candidate scoring），到这里就停了。这就是action-only mode，是最fast的deployment。

**Stage 2**: 把$\hat{a}_{t:t+H}$放到clean action-conditioning slot G的位置，然后denoise value token（和可选的future-video tokens）。

这里有个subtle的点：Stage 2的conditioning slot G现在装的是Stage 1输出的$\hat{a}$，这和训练时G装的是ground-truth action有slight mismatch。但是由于Stage 1输出的action已经是clean的，分布上接近训练时的G，所以mismatch不大。

### 5.2 Optional Candidate Scoring

这是FACT的一个optional deployment interface。Sample $N$个action candidates $\{a^{(k)}\}_{k=1}^N$，对每个candidate predict value $\hat{v}^{(k)} = V_\theta(o_t, \ell, a^{(k)})$，然后选：

$$a^\star = \arg\max_{k \in \{1, \dots, N\}} \hat{v}^{(k)} = \arg\max_{a \in a^{(1:N)}} V_\theta(o_t, \ell, a)$$

这个selection rule用value head去score每个candidate action的implied future，不需要单独训练一个critic。

**关键洞察**：这个candidate scoring只有在value head被failure data训练过之后才有效。Tables 2的ablation "Ours + scoring"（without failure）只有79%，比"Ours"的82%还低。这说明success-only训练的value head无法区分good和bad actions，因为它从来没见过bad action的consequence。只有"Ours w/ failure + scoring"才能达到92%。

Figure 7展示了candidate数量N的影响：从N=1到N=4有clear gain，之后边际收益递减但latency线性增长。所以real-world experiments用N=4。

## 6. Experiments 详解

### 6.1 RoboTwin Simulation Results (Table 1)

| Method | Clean | Rand. | Average |
|--------|-------|-------|---------|
| π₀ [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) | 65.9 | 58.4 | 62.2 |
| X-VLA [arXiv:2510.10274](https://arxiv.org/abs/2510.10274) | 72.9 | 72.8 | 72.9 |
| π₀.5 [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) | 82.7 | 76.8 | 79.8 |
| Gigaworld-Policy | 87.0 | 85.0 | 86.0 |
| Motus | 88.7 | 87.0 | 87.8 |
| FACT (Ours) | 86.3 | 84.9 | 85.6 |
| **FACT w/ failure** | **88.4** | **86.6** | **87.5** |
| FACT w/o video co-train | 82.5 | 81.0 | 81.8 |

几个关键观察：
1. FACT + failure (87.5%) 几乎追平Motus (87.8%)，但Table 7显示Motus latency是1220ms，FACT只有380ms，**3.2x faster**。
2. Failure co-training的提升：85.6% → 87.5%，绝对+1.9%，相对failure data带来的提升。
3. Video co-training的提升：81.8% → 85.6%，绝对+3.8%。这说明future prediction作为regularizer对action generation很重要。

### 6.2 Real-World Results (Tables 2 & 3)

**Seen tasks** (5 tasks: Stack Cubes, Pick Cubes, Handover, Stack Bowls, Pour)：

| Method | Stack | Pick | Handover | Bowls | Pour | Avg. |
|--------|-------|------|----------|-------|------|------|
| Cosmos | 5 | 45 | 25 | 35 | 15 | 25 |
| π₀ | 35 | 70 | 40 | 50 | 45 | 48 |
| π₀.5 | 75 | 100 | 85 | 80 | 100 | 88 |
| Motus | 50 | 70 | 55 | 85 | 60 | 64 |
| FACT | 70 | 85 | 90 | 80 | 85 | 82 |
| **FACT w/ failure** | 75 | 95 | 85 | 95 | 95 | 89 |
| **FACT w/ failure + scoring** | 85 | 100 | 85 | 100 | 90 | 92 |

**Unseen tasks** (3 held-out variants)：

| Method | Stack | Pick | Bowls | Avg. |
|--------|-------|------|-------|------|
| Cosmos | 15 | 10 | 0 | 8 |
| π₀ | 30 | 65 | 75 | 57 |
| π₀.5 | 65 | 90 | 100 | 85 |
| Motus | 55 | 60 | 70 | 62 |
| FACT | 45 | 75 | 80 | 67 |
| FACT w/ failure | 60 | 85 | 85 | 77 |
| FACT w/ failure + scoring | 65 | 95 | 85 | 82 |

关键观察：
1. FACT在seen tasks上**大幅超越Motus** (82% vs 64%)。这很有意思——Motus在simulation上比FACT高，但是在real-world上低很多。可能因为Motus的video-first design对real-world的visual distribution shift更敏感。
2. Failure co-training的提升在real-world上更显著：82% → 89% (+7%)，远超simulation上的+1.9%。这暗示real-world task的failure mode更复杂，failure data的价值更高。
3. FACT w/ failure + scoring (92%) 几乎追平π₀.5 (88%)，甚至超过。考虑到π₀.5用了large-scale robot pre-training而FACT没有，这个结果相当impressive。
4. Unseen tasks上FACT + scoring达到82%，π₀.5是85%，差距只有3pp。

### 6.3 Ablation Studies - 最关键的部分

Tables 2的ablations提供了几个critical insights：

| Ablation | Stack | Pick | Handover | Bowls | Pour | Avg. |
|----------|-------|------|----------|-------|------|------|
| Ours + scoring (no failure) | 80 | 80 | 70 | 90 | 75 | 79 |
| Ours w/o causal mask | 50 | 75 | 95 | 85 | 80 | 77 |
| Ours w/ failed-action loss | 45 | 55 | 75 | 65 | 75 | 63 |
| Ours w/o video co-train | 60 | 55 | 35 | 85 | 55 | 58 |

**关键分析**：

1. **"Ours + scoring" without failure (79%) < "Ours" (82%)**：这证明了candidate scoring只有在failure data训练后才有用。Success-only的value head对action quality不敏感，scoring反而引入噪声。

2. **"Ours w/o causal mask" (77%)**：去掉teacher-forcing clean action condition，jointly denoise A/V/I。Loss of 5pp。这说明teacher-forcing design对于让future prediction转化为更好的action generation至关重要。Without it，future prediction和action generation是decoupled的。

3. **"Ours w/ failed-action loss" (63%)**：这是最dramatic的ablation。如果failure trajectories的action imitation loss不被mask，success从82%暴跌到63%。**直接验证了核心hypothesis**——failure data不能直接作为imitation target，否则会corrupt action decoding。

4. **"Ours w/o video co-train" (58%)**：去掉future video prediction，只保留action + value。Loss of 24pp！这证明future prediction作为regularizer对action generation极其重要，远超value prediction alone的作用。

### 6.4 Failure Data Reduces Future Hallucination (Table 4)

这是定性验证core hypothesis的实验：

| Subset | Ours (PSNR↑) | Ours w/ failure (PSNR↑) |
|--------|--------------|--------------------------|
| All | 22.82 | 26.00 |
| Success-rollout | 26.12 | 26.08 |
| Failure-rollout | 19.51 | 25.92 |

PSNR (Peak Signal-to-Noise Ratio)衡量predicted future和ground truth future的相似度，越高越好。

关键观察：
1. **Failure-rollout PSNR: 19.51 → 25.92**，绝对+6.41 dB。这是巨大的提升。Failure-aware model能准确predict bad action导致的bad future，而success-only model在bad action条件下依然hallucinate good future，导致预测和实际failure outcome严重不符。
2. **Success-rollout PSNR: 26.12 → 26.08**，几乎不变。这证明加入failure data没有degrade normal future prediction能力。Failure data是"additive"的，学习了新的conditional distribution (P(bad future | bad action))，没有overwrite原有的 (P(good future | good action))。

Figure 5的定性可视化更直观：在bad grasp的action条件下，success-only model预测的future画面里cube被抓起来了（white dotted box标出），而failure-aware model预测的future里cube掉在地上。这就是success-biased hallucination被消除的直观证据。

### 6.5 Failure-Data Scaling (Figure 6)

在3个RoboTwin clean tasks上，failure data fraction $p \in \{0\%, 50\%, 100\%\}$（100%时failure占总training set的45%）：

- $p = 0\%$: 32.7% success
- $p = 50\%$: ~45% (从图估读)
- $p = 100\%$: 57.3% success

**Monotonic improvement，没有saturation**。这说明failure data的scaling law还在early stage，加更多failure data可能继续提升。这个结果对未来online RL、self-improvement的regime很有启发——failure rollouts可以是unlimited的，potential的提升空间很大。

### 6.6 Value Traces (Figure 8)

Figure 8展示了一个Pick Cubes rollout的value trace：
- 任务进展时value上升
- Grasp失败时value下降
- Policy调整后re-grasp，value再次上升

这是action-conditioned value的一个关键性质——因为value是conditioned on executed action，它能在action导致poor outcome时decrease。这和RL里的Q-function类似，但是不需要单独训练critic。

Appendix G的Figure 12展示了Stack Bowls和Stack Cubes的类似value traces，进一步验证这个behavior的generality。

Appendix H的Figure 13做了一个controlled experiment：在Stack Cubes的3×3 grid上评估9个candidate placements，只有中心位置成功。Value head给中心位置最高分，其他8个failure位置给lower（甚至negative）分数。这是value head能区分action quality的直接证据。

## 7. Limitations 和 Future Directions

Paper的Limitations部分比较honest：

1. **Scaling to broader data**：目前只在bimanual manipulation上验证，scaling到broader robot data和human-interaction data可能进一步提升physical plausibility of future prediction和value head的discriminative power。

2. **Value head的改进**：可以用learned progress estimators替代或augment当前的value head，保持action-conditioned value interface不变。

3. **Best-of-N selection的latency cost**：value-guided selection需要额外的scoring pass per candidate batch。Latency-critical场景可以用action-only mode。

## 8. 联想与延伸思考

### 8.1 和RL的connection

FACT的action-conditioned value predictor在结构上很接近RL里的Q-function $Q(s, a)$，但是有几个关键区别：

1. **No TD learning**：FACT的value是directly supervised by progress target $p_{t+H}$，没有temporal difference learning。这避免了TD learning的bootstrapping instability，但是也失去了credit assignment over long horizons的能力。
2. **No exploration**：FACT的value只在expert和failure rollouts上训练，没有exploration带来的coverage。这限制了value head在out-of-distribution actions上的准确性。
3. **Action-conditioned but not policy-conditioned**：Value是$V_\theta(o_t, \ell, a)$，不是$V_\theta(o_t, \ell, \pi(\cdot|o_t))$。这更适合best-of-N selection，但不直接支持policy improvement。

但是这个framework天然可以extend到RL：用FACT的value head作为critic，policy作为actor，用actor-critic framework做online RL。Paper里提到了"future training regimes that include online rollouts, DAgger-style corrections, and reinforcement learning from negative experience"，暗示了这个方向。

### 8.2 和VLA-RFT、VIVA的connection

References [42] VLA-RFT [arXiv:2510.00406](https://arxiv.org/abs/2510.00406)和[43] VIVA [arXiv:2604.08168](https://arxiv.org/abs/2604.08168)都是最近用RL fine-tune VLA的工作。FACT的failure-aware training可以看作是offline RL的一种形式——从failure trajectories里学习，但不直接做policy improvement。

一个自然的extension：把FACT的value head用作VLA-RFT的verified reward signal，或者把VIVA的video-generative value model和FACT的action-conditioned prediction结合。这两者的结合可能产生一个既能从failure学习consequence，又能online improve policy的framework。

### 8.3 和AlphaGo的Tree Search的类比

FACT的candidate scoring在概念上接近AlphaGo的Monte Carlo Tree Search (MCTS)的简化版：
- Sample N candidates = expand N branches
- Value head score = leaf evaluation
- argmax selection = pick best branch

区别是FACT没有真正的tree search，只是1-step lookahead。但是这个framework可以extend到multi-step lookahead：用future video prediction作为next state，然后recursive apply。这就是paper里暗示的"future training regimes"的方向之一。

### 8.4 Causal Modeling的更深层意义

Paper里"causal"这个词出现了很多次。这里的"causal"有两层含义：

1. **Causal order**：action first → future second。这是temporal causality。
2. **Causal inference**：P(future | do(action)) vs P(future | action)。这是Pearl的do-calculus意义上的causality。

第二层意义更深刻。在observational data里，P(future | action)和P(future | do(action))可能不同，因为action的选择受confounders影响。但是FACT的teacher-forcing设计实际上接近于intervention：通过强行设置action condition为ground-truth executed action，它estimates的是P(future | do(action))，而observational conditional。

这个causal interpretation让FACT的failure-aware training有了更深的意义：failure rollouts提供了"negative interventions"的data，让模型学到"do(bad action) → bad future"的causal relationship。这是和单纯observational learning的本质区别。

### 8.5 和DreamerV3的对比

DreamerV3是model-based RL的经典工作，也是action-conditioned world model。但是DreamerV3的world model在latent space里rollout，FACT在pixel space（更准确说是VAE latent）里预测future video。

关键区别：
- DreamerV3: world model + policy + value都是从rollout里学的，需要online interaction。
- FACT: world model和value是从offline data（success + failure rollouts）学的，policy是imitation learning。

但是两者的structure很像，FACT可以看作是把DreamerV3的idea搬到offline imitation setting + VLA framework里。这也暗示了一个future direction：用FACT的world model做imagination-based RL，类似DreamerV3但是用大scale video prior。

## 9. Summary: 为什么这个工作重要

FACT的贡献可以总结为三点：

1. **Conceptual contribution**：识别并formalize了success-biased future hallucination问题，并提出了causal order reversal的解法。这个insight对整个WAM领域都有启发。

2. **Technical contribution**：Teacher-forcing attention mask的设计elegant地解决了action imitation和future prediction的causal conflict。这个mask design可能可以应用到其他multi-modal co-training场景。

3. **Practical contribution**：在real-world bimanual manipulation上达到了接近π₀.5的性能，latency只有380ms。Failure data scaling实验暗示了online self-improvement的巨大potential。

从Karpathy你之前讲过的"building general-purpose robot policies"的视角，FACT提供了一个关键的missing piece：如何让robot从自己的failure中学习，而不是只从expert demonstration中学习。这是从imitation learning走向真正的self-improving agent的必要一步。

**参考链接**：
- Project page: [https://fact-wam.github.io](https://fact-wam.github.io)
- Flow Matching: [https://arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)
- WAN Video Model: [https://arxiv.org/abs/2503.20314](https://arxiv.org/abs/2503.20314)
- RoboTwin: [https://arxiv.org/abs/2506.18088](https://arxiv.org/abs/2506.18088)
- π₀: [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)
- π₀.5: [https://arxiv.org/abs/2504.16054](https://arxiv.org/abs/2504.16054)
- OpenVLA: [https://arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246)
- Motus: [https://arxiv.org/abs/2512.13030](https://arxiv.org/abs/2512.13030)
- Cosmos Policy: [https://arxiv.org/abs/2601.16163](https://arxiv.org/abs/2601.16163)
- DreamZero: [https://arxiv.org/abs/2602.15922](https://arxiv.org/abs/2602.15922)
- BagelVLA: [https://arxiv.org/abs/2602.09849](https://arxiv.org/abs/2602.09849)
- VLA-RFT: [https://arxiv.org/abs/2510.00406](https://arxiv.org/abs/2510.00406)
- GELLO: [https://arxiv.org/abs/2409.04152](https://arxiv.org/abs/2409.04152)
