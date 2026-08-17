---
source_pdf: Robots That Know What to Ask Recovering.pdf
paper_sha256: d073f670bb8f4e9db3d003ebb1918611015f860f91a59f1b5b29353ec44e93b0
processed_at: '2026-08-12T02:06:13-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用人话讲，这篇 paper 解决的核心痛点就是：**Robot 怎么发现自己“没学好”，并且主动向人“提问”。**

---

### 1. 故事背景：Robot 的误解从哪来？

想象你在教 Robot 端咖啡过桌子。你心里有四件事要顾忌：杯子别打翻、别撞桌子、离人远点、离电脑远点。

但人的注意力是有限的。你演示的时候，死死盯着杯子别打翻，小心翼翼别撞桌子，结果路过电脑时，距离忽远忽近——因为你脑子里的 cognitive load 满载了，没顾上精确控制离电脑的距离。

Robot 看了你的演示，发现“离电脑的距离”这个指标忽高忽低。这时候 Robot 陷入了深深的迷茫：它分不清你是**真的不在乎离电脑多远**（哪怕撞了也行），还是你**其实在乎，只是刚才顾不上表现**。

传统 IRL 算法遇到这种情况，通常会粗暴地认定：“既然没规律，那就是不在乎”。结果 Robot 部署到现实里，为了抄近道，直接贴着电脑飞过去，甚至撞翻电脑。这就是 reward misalignment 的典型翻车现场。

---

### 2. 核心大招：怎么抓出“没学好”的地方？

作者的 insight 非常直觉：**如果你真的在乎一件事，你会做得很稳定；如果你不在乎，那这件事的表现就会很乱。**

- 你在乎杯子别打翻 → 每次演示，杯子都很直，**variance 极低**。
- 你没顾上离电脑的距离 → 每次演示，离电脑的距离随机变化，**variance 极高**。

Robot 于是去算每个 feature 在你多次演示里的 variance。哪个 feature 跳得最厉害，哪个就是“没学好”的嫌疑人。

但这里有个坑：有些 feature 天生就乱，不代表没学好。所以 Robot 先用 LLM 过滤掉不相干的 feature（比如桌上苹果的位置），只留 task-relevant 的 feature 再去查 variance。

---

### 3. 解决办法：不瞎猜，直接问

找到嫌疑 feature 后，Robot 不去瞎猜你是真不在乎还是假不在乎，它选择**直接开口问**。

它生成一句很接地气的大白话解释：“**I am uncertain how to handle distance to the laptop. Show me how best to move to the edge of the table, focusing on distance to the laptop.**”

这叫 targeted explanation。你一听就懂了，哦，原来它不知道怎么躲电脑。于是你再演示几次，这次特意把“躲电脑”这个动作做得明明白白。

---

### 4. 聪明的学习法：把新旧演示分开算分

拿到你新补的演示后，Robot 的处理方式很巧妙。

如果它把旧演示和新演示混在一起一锅炖，那旧演示里离电脑忽远忽近的“噪声”就会污染新演示里的“精确控制”信号。所以 Robot 用了**双 β 加权**机制（论文里叫 per-feature rationality）：

- 对于旧演示：Robot 知道你在“躲电脑”上没上心，所以算分时，把旧演示里关于电脑距离的权重调很低，听个响就行；但对杯子直不直的权重调很高，因为那部分信号很纯。
- 对于新演示：Robot 知道你这次是专门来教“躲电脑”的，所以把新演示里关于电脑距离的权重调很高，重点听这部分；至于新演示里杯子可能没端平，它不在乎，权重调低。

这样一加权，两套演示各自最干净的信号都被提取出来了，reward function 自然就学准了。

---

### 5. 实验验证：真人对战 Franka 机械臂

作者不仅跑了 simulation，还找了 12 个真人跟 Franka FR3 机械臂互动。比了三种问法：

1. **Unguided**：Robot 直接说“教我端咖啡”。（人不知道 Robot 哪里不懂）
2. **Rollout**：Robot 先自己走一遍，让人看哪里错了再教。（人得自己找错在哪）
3. **Explanation**：Robot 直接说“我不懂怎么躲电脑，你再教教我”。（精准报错）

结果很有意思：**Explanation 条件下学到的 reward 显著比另外两个好**。更神奇的是，Rollout（让 Robot 先走一遍）根本没用，效果跟 Unguided 一样差。原因在于，人看 Robot 走一遍，看到的是一个“综合错误”，很难反推出来到底是哪个 feature 出了问题。直接点名 feature，人的认知负担反而最小，给的纠正数据质量最高。

不过 user study 也暴露了个小 bug：Robot 说“我不懂离电脑的距离”，有的真人理解为“保持一点安全距离就行”，有的理解为“离得越远越好”，导致新演示的数据有点分叉。这说明光给 feature 名字还不够，得配上 visualization 或者 example trajectory 才能彻底对齐。

---

### 6. 联想：这跟大模型对齐是一回事

Andrej，你肯定一眼就看出来了，这个 pattern 跟现在大语言模型的 RLHF alignment 面临的困境本质同构。

RLHF 训出来的 reward model，经常在某些 dimensions 上 signal 很弱（因为 human rater 也没注意），导致 model 会在那些 underspecified 的维度上 misbehave（比如效率压倒安全）。ASQ 的思路完全可以 port 过去：观察 reward model 对不同 feature 的偏好数据在哪几个维度上 variance 大、rater 之间分歧多，就针对性地去主动 query rater：“你对这个维度到底是怎么想的？”这比撒网式收集 preferences 要高效得多。

ASQ 的底层哲学就是：**与其被动等待完美的 supervision，不如让 agent 学会主动暴露自己的无知，并精准索取缺失的知识。**

相关参考：
- ASQ Paper: https://hmerker.github.io/asq/ (推测)
- Bobu et al., T-RO 2020 (misspecification): https://ieeexplore.ieee.org/document/8967724
- Peng et al., ICML 2023 (Diagnosis & feedback): https://proceedings.mlr.press/v202/peng23c.html
- Hwang et al., 2025 (Masked IRL): https://arxiv.org/abs/2511.14565
- Cakmak & Thomaz, HRI 2012 (Robot queries): https://dl.acm.org/doi/10.1145/2157689.2157693

---

# Robots That Know What to Ask: Recovering Misaligned Rewards through Targeted Explanations

作者：Helena Merker, Nick Walker, Andreea Bobu (MIT)
项目页面：https://hmerker.github.io/asq/ (推测，作者主页 https://hmerker.github.io/ 和 Andreea Bobu 的 lab http://idealab.mit.edu/)
相关 prior work：Bobu et al., "Quantifying hypothesis space misspecification" T-RO 2020, https://ieeexplore.ieee.org/document/8967724 ；Peng et al., ICML 2023, https://proceedings.mlr.press/v202/peng23c.html

下面我把这篇 paper 拆开讲，重点放在 mathematical formulation、为什么这个 insight 成立、以及它跟 IRL 整个 lineage 的关系。

---

## 1. 核心问题的 first-principles 视角

Inverse Reinforcement Learning (IRL) 的经典假设：demonstrations 是从某个 latent reward $R(s) = \theta^{*\top}\phi(s)$ 出发，由一个 (近似) optimal 的 demonstrator 生成的。基于这个假设，MaxEnt IRL (Ziebart 2008, https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf) 给出

$$P(\tau \mid \theta^*, \beta) \propto \exp\big(\beta \cdot \theta^{*\top}\phi(\tau)\big)$$

这里 $\beta$ 是一个 scalar inverse temperature：人越理性，$\beta$ 越大；人越 noisy，$\beta$ 越小。$\phi(\tau) = \sum_{s^t \in \tau} \phi(s^t)$ 是 trajectory 上 features 的累加。

**这个 paper 撕开的核心 bug**：把 $\beta$ 当作 scalar 是一个 identifiability 致命伤。真实演示里人不是"整体 noisy"，人是对**每个 feature 单独 noisy**。端咖啡穿过桌面时，人会牢牢盯住 "coffee 杯子不能倾斜" 和 "不要撞东西"，但路过笔记本电脑的距离时，ta 可能因为认知 load (Sweller 1988, cognitive load theory, https://doi.org/10.1016/0364-0213(88)90023-7) 或者因为环境配置让那个 feature 不容易 exercise，而"路过距离随机变化"。

这时如果你只看 demonstrations，你看到 laptop distance 的 feature values 高度散乱。两种解释：
- (A) 人真的不在乎 laptop distance：$\theta_i^* \approx 0$
- (B) 人在乎，但没来得及表达：$\beta_i \approx 0$，但 $\theta_i^* > 0$

这两种从 likelihood 上完全等价——这是 Bobu et al. 2020 https://ieeexplore.ieee.org/document/8967724 早就指出的 hypothesis space misspecification。当前大多数 IRL work (Finn et al. 2016 guided cost learning, https://proceedings.mlr.press/v48/finn16.html ；Christiano et al. 2017 RLHF, https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) 默认走 (A) 这条路：高 variance = low weight。结果 robot 学到一个 "efficiency-maximizing, ignoring fragile objects" 的 misaligned policy。这正是 RLHF alignment failure 在 robot learning 里的同构问题。

---

## 2. 关键 insight：optimization 留下 statistical footprint

Paper 的核心观察一句话能说清楚：**优化一个 feature 会压低它的 cross-demonstration variance；没优化的 feature 的 variance 自然散**。

直觉上想想：如果演示者真的在最小化"杯子倾斜"，每条 demonstration 的 coffee feature 都会被推到一个 tight 区域（受物理 + 优化过程的 noise 限制）。如果 ta 不在乎"碗的距离"，碗的距离会跟着其他目标被顺便推来推去，没东西把它钉死，自然散得开。

这个 insight 把 identifiability 从 "θ vs β 联合不可识别" 变成 "θ 和 β 的统计 footprint 可分离"：$\beta_i$ 直接控制 feature $i$ 的 cross-demo variance 的 level，$\theta_i^*$ 控制的是 feature 值的 mean/target。两个参数留下两种不同统计形态的痕迹。

注意这个 insight 的边界条件——它要求 demonstrations 之间有足够的 i.i.d.-ish 结构，且 feature 在 trajectory 上的 marginal 分布稳定。如果 demonstrator 在不同 demos 之间切换策略（比如第 1 次在练 coffee，第 5 次在练 laptop），variance 信号会被污染。这点 paper 没显式讨论。

---

## 3. 数学模型：从 scalar β 到 per-feature β

Paper 把公式 (2) 改写为：

$$P(\tau \mid \theta^*, \beta) \propto \exp\bigg(\sum_{i=1}^{k} \beta_i \theta_i^* \phi_i(\tau)\bigg) \tag{2}$$

变量解释：
- $\theta^* \in \mathbb{R}^k$：human 的真实 reward weights（latent, 要学的）
- $\beta \in \mathbb{R}_{\geq 0}^k$：**per-feature** rationality / attention / care 向量
- $\phi_i(\tau) = \sum_{s^t \in \tau} \phi_i(s^t)$：trajectory 上 feature $i$ 的累积值
- $k$：feature 数量

为什么这是关键推广：传统 IRL 把 $\beta$ 当 hyperparameter，paper 把它升格为每个 feature 一个。当 $\beta_i \to 0$，$\phi_i(\tau)$ 在 likelihood 里被 shut off，no matter what $\theta_i^*$ 是多少，likelihood 都一样——这就是 underspecification 的数学定义。

这个建模跟 Bajcsy et al. 2018 "one feature at a time" (https://dl.acm.org/doi/10.1145/3171221.3171267) 的物理 correction 思路是同源的——把 feature 监督水平显式建模，而不是当一个 uniform noise。Beliaev & Pedarsani 2025 (https://ojs.aaai.org/index.php/AAAI/article/view/33705) 做 "estimating expertise of demonstrators" 也是这个思路的一个变种。

---

## 4. 检测 underspecified features：Bayesian model selection

给定 initial demos $D_{\text{init}} = \{\tau_1, \dots, \tau_n\}$，每个 feature $i$ 的 empirical variance：

$$\sigma_i^2 = \frac{1}{n-1}\sum_{j=1}^n (\phi_i(\tau_j) - \bar\phi_i)^2, \quad \bar\phi_i = \frac{1}{n}\sum_{j=1}^n \phi_i(\tau_j)$$

引入 latent binary indicator $o_i \in \{0, 1\}$：
- $o_i = 1$：feature $i$ 在 demos 里被 consistently optimized
- $o_i = 0$：feature $i$ underspecified（=non-optimized）

Hypothesis space：$\phi_{\text{hyp}} \subseteq \phi$，每个 hypothesis 是一个 "underspecified features 集合"。

Posterior：
$$P(\phi_{\text{hyp}} \mid \{\sigma_i^2\}) \propto P(\{\sigma_i^2\} \mid \phi_{\text{hyp}}) \cdot P(\phi_{\text{hyp}}) \tag{3}$$

Likelihood 假设条件独立 factorize：
$$P(\{\sigma_i^2\} \mid \phi_{\text{hyp}}) = \prod_{i \in \phi} P(\sigma_i^2 \mid o_i, \phi_i) \tag{4}$$

其中 $o_i = 0$ if $\phi_i \in \phi_{\text{hyp}}$ else $1$。

MAP 选择：$\phi_{\text{under}} = \arg\max_{\phi_{\text{hyp}}} P(\phi_{\text{hyp}} \mid \{\sigma_i^2\})$。

**这里有几个直觉层面要点**：

(a) **为什么用 model selection 而不是 per-feature 判定**：单独看一个 feature 的 variance 高，没法判定它是否 underspecified——因为有些 feature 天然就高 variance。但如果你看一个 *组合* $\phi_{\text{hyp}}$，"这 4 个 features underspecified, 那 4 个 optimized" 这个 joint hypothesis 对整组 variance 模式的拟合是可比较的。Bayesian model selection 自然处理 feature 之间的尺度差异。

(b) **为什么需要 reference distributions**：raw $\sigma_i^2$ 的尺度依赖 feature 的 natural scale、环境动力学、inherent task variability。要 calibrate "什么样的 variance 算高"，paper 用仿真构造两类 reference distribution：$P(\sigma_i^2 \mid o_i=1, \phi_i)$（被优化的 variance 分布）和 $P(\sigma_i^2 \mid o_i=0, \phi_i)$（没被优化的 variance 分布）。构造方法：枚举 features 的所有子集 $\mathcal{S} \subseteq \phi$，每个 $\mathcal{S}$ 给 $\beta_i = 80$ (JacoRobot) 或 $\beta_i = 50$ (GridRobot) 当 $\phi_i \in \mathcal{S}$ 否则 $\beta_i = 0$，生成 20 demos，重采样 10 个计算 variance，重复 500 次，拟合 chi-squared 分布。

(c) **条件独立假设的局限**：公式 (4) 假设 feature 之间 conditional independent given optimization status。真实环境里 features 强相关（离 laptop 远 = 离 table edge 近 = 离 human 近），paper 在 limitations 里诚实承认这点。但经验上 chi-squared 拟合提供了 working approximation。

---

## 5. LLM 作为 prior over feature subsets

Bayesian model selection over $2^{|\phi|}$ subsets 指数爆炸，且 irrelevance features 会假阳性触发 query。Paper 用 LLM (GPT-5.2 with high reasoning effort) 给一个 prior $P(\phi_{\text{hyp}})$：

Prompt 设计见 Appendix B。把 task description + feature list 喂给 LLM，让它对每个 feature 返回 0/1 relevance judgment。"A feature should be labeled 1 only if a typical human would expect that ignoring it would commonly lead to a clearly undesirable outcome in this specific task."

这个 LLM-as-prior 的设计跟最近 Hwang et al. 2025 Masked IRL (https://arxiv.org/abs/2511.14565) 和 Peng et al. 2024 "Adaptive language-guided abstraction from contrastive explanations" (https://proceedings.mlr.press/v270/peng25c.html) 是一个家族——用 LLM 的 common-sense prior 给 reward learning 提供结构。Intuition 上：LLM 编码了 "端咖啡过桌" 这类任务的 common-sense（laptop 距离重要，apple 距离不重要），这正好是 IRL 一直缺的 prior。

---

## 6. Eliciting targeted feedback：natural language explanation

这个步骤简单但本质。一旦识别 $\phi_{\text{under}}$，robot 用模板化 NL 解释："I am uncertain about how to handle distance to the laptop. Show me how best to move to the edge of the table, focusing on distance to the laptop."

这是一个 **causal intervention**：之前 demos 的 $\beta_i \approx 0$ 是因为 demonstrator 自己的 attention 分配；现在 robot 显式 push attention 到 $\phi_i \in \phi_{\text{under}}$，强制 $\beta_i^{\text{extra}}$ 升高。这正好绕过前面 (A)/(B) 的 identifiability：直接问，让 user 自己 resolve。

Cakmak & Thomaz 2012 "Designing robot learners that ask good questions" (https://dl.acm.org/doi/10.1145/2157689.2157693) 是这类 query design 的开山；Sadigh et al. 2017 active preference-based learning (https://www.roboticsproceedings.org/rss13/p53.pdf) 是 active query 的另一支。ASQ 落在 "demonstration query with feature-targeted explanation" 这个交叉点。

---

## 7. Reward learning：两套 demonstrations 的双 β 加权

这是 paper 最 subtle 的部分。Standard IRL 把所有 demos 当作 i.i.d. 来自一个 $\beta$：

$$\theta^* = \arg\max_\theta \sum_{\tau \in D_{\text{total}}} \log P(\tau \mid \theta, \beta) \tag{5}$$

但 ASQ 的 $D_{\text{init}}$ 和 $D_{\text{extra}}$ 来自不同的 attention regimes：
- $D_{\text{init}}$：natural，$\beta_i^{\text{init}}$ 在 $\phi_{\text{under}}$ 上 low，其他 high
- $D_{\text{extra}}$：guided，$\beta_i^{\text{extra}}$ 在 $\phi_{\text{under}}$ 上 high，其他 low

如果硬把它们 pool 起来当一个分布，等于稀释两边的 signal。Paper 的处理：

$$\beta_i^{\text{extra}} = \begin{cases} \beta_{\text{high}} & \phi_i \in \phi_{\text{under}} \\ \beta_{\text{low}} & \text{otherwise} \end{cases}, \quad \beta_i^{\text{init}} = \begin{cases} \beta_{\text{low}} & \phi_i \in \phi_{\text{under}} \\ \beta_{\text{high}} & \text{otherwise} \end{cases}$$

Shared reward $\theta^*$，分别建模。Loss：

$$\mathcal{L}(D; \theta, \beta) = \frac{1}{|D|}\sum_{\tau \in D} \bigg[-\sum_{i=1}^k \beta_i \theta_i \phi_i(\tau)\bigg] + \log Z(\theta, \beta) \tag{6}$$

$$\mathcal{L}_{\text{total}} = \frac{|D_{\text{init}}| \cdot \mathcal{L}_{\text{init}} + |D_{\text{extra}}| \cdot \mathcal{L}_{\text{extra}}}{|D_{\text{init}}| + |D_{\text{extra}}|} \tag{7}$$

其中 partition function $Z(\theta, \beta) = \int \exp(\sum_i \beta_i \theta_i \phi_i(\tau')) d\tau'$ 用 importance sampling 近似（Finn et al. 2016 guided cost learning 的标准做法，https://proceedings.mlr.press/v48/finn16.html）。

**Intuition 上这相当于做了什么**：把 "这个 demo 在 feature $i$ 上多 informative" 显式编码到 likelihood 里。$D_{\text{init}}$ 在 $\phi_{\text{under}}$ 上的信号是噪声——直接 downweight 它对 $\theta_i$ 的影响；$D_{\text{extra}}$ 在 $\phi_{\text{under}}$ 上信号强——让它主导 $\theta_i$ 的估计。同时反向：$D_{\text{init}}$ 对其他 features 是高信号，$D_{\text{extra}}$ 因为人被引导到 $\phi_{\text{under}}$ 反而可能让其他 features 的信号变弱（user study data Appendix E 实证了这一点：在 Explanation 条件下，未点名的 features 的 std 显著增大，比如 human task 中 laptop feature std 在 Explanation 下是 Unguided 的 3.36×）。

这点比看上去重要：它意味着你不能简单 "naive pool + 更多数据"，因为更多 guided demos 反而会污染 initially-well-specified features 的信号。两套 β 的设计正好把两套 demos 各自的"信息富集区"配对到该问的 features 上。

---

## 8. Simulation experiments：JacoRobot

环境：PyBullet 7-DoF Jaco arm 端咖啡杯穿桌。21 waypoints × 109 dims (joint config + xyz of all joints/objects + rotation matrices)。4 个 task-relevant features：laptop (xy-distance)、table (z-distance)、human (xy-distance)、coffee (杯子直立度，end-effector up axis 在 world up axis 上的投影长度)。扩展到 8 features 时加 apple/bowl/fork/marker 作为 distractors。

50000 trajectories across 5000 start-goal pairs (10 perturbed per pair)。60/20/20 train/val/test split。

Simulated human：Boltzmann-rational with ground truth $\theta^*$ 和 per-feature $\beta$。设 $\beta_i = $ 高值 → 优化 $\phi_i$；$\beta_i = 0$ → underspecified。Note：这是 conservative 模型，真实人可能 de-prioritize 而非完全 ignore。

| Method | 描述 |
|---|---|
| ASQ | variance detection + targeted query + 双 β 加权 |
| ASQ-Unweighted | ablation: 不加权 (uniform β) |
| Random-Demo | 每次 query $N$ 个 random features |
| Random-Feat | 起始随机选 $N$ features，之后都针对这些 |
| Oracle | 用 ground-truth underspecified features |

Evaluation：在 test set 里取 top-20 highest learned reward trajectories，算 average ground-truth reward，按 ground-truth reward range 归一化。

**结果（Fig. 3, 4）**：
- 单 underspecified feature：ASQ 在 coffee/human/laptop underspecified 时显著快且高于 baselines，逼近 Oracle。table feature 是例外——3 个 features 优化时第 4 个恰好也被动做好，所以初始 reward 已经高。这点是重要的 edge case：underspecified ≠ suboptimal，要看 feature 间相关性。
- 双 underspecified features：所有 6 个 pairwise 组合里 ASQ 都优于 baselines，但 gain 大小取决于 feature pair。
- 8-feature with distractors：LLM filtering 把 distractors 砍掉再 variance detection，ASQ 匹配或超越 baselines (Fig. 4)，证明 robustness to irrelevant features。

GridRobot (Appendix C) 是 5×5 discrete navigation 的简化验证：2 features (goal, obstacle)，同样 ASQ ≈ Oracle 而 baselines plateau。

---

## 9. User study：Franka FR3

Real-robot 验证。Within-subjects design，3 conditions：
- **Unguided**：robot 直接说 "Show me how to move the mug to the edge of the table."
- **Rollout**：robot 先执行当前 learned reward 下的 trajectory，然后说 "This is how I think I should do it... Show me how best to move to the edge of the table."
- **Explanation**：robot 说 "I am uncertain how to handle [feature]. Show me how best to move to the edge of the table, focusing on [feature]."

12 participants (6F/6M, age M=28.5 SD=12.5)，prior robot experience M=4.2/7。Counterbalanced ordering across 6 orderings。Familiarization 用 coffee 作为 underspecified feature，两个 experimental tasks 分别用 human 和 laptop。

每个 participant 事先被告知 4 个 objectives，每次 condition 前再提醒——这关键：isolates attentional guidance 的 effect，而不是让 participant 自己猜 robot 关心什么。

### 假设与结果

| 假设 | 内容 | 结果 |
|---|---|---|
| H1 | explanation → 更好的 reward recovery | **支持**。$F(2,22)=14.94, p<.001$。Explanation ($M=0.021, SE=0.005$) 显著高于 Unguided ($M=-0.018, SE=0.005, t=-6.31, p<.001$) 和 Rollout ($M=-0.016, SE=0.005, t=-6.06, p<.001$)。Unguided 和 Rollout 无差异 ($p=.966$)。 |
| H2 | explanation → 更低 cognitive load (NASA-TLX) | **不支持**。$F(2,22)=0.76, p=.480$。三个条件 TLX 都接近 25-28/100。Paper 推测因为 physical demands 固定（kinesthetic guidance 同样的动作）。 |
| H3 | 主观偏好倾向 explanation | **不支持**。5/12 选 Explanation，5/12 选 Rollout，2/12 选 Unguided。$\chi^2=1.50, p=.616$。 |

### Ablation 揭示 weighting 的作用（Fig. 5 中 hashed bars）

在 Explanation 内部比较 all-weighted ($M=0.021$) vs all-unweighted ($M=-0.017$)，差异 $t(23)=4.56, p<.001$，mean diff 0.038。这说明 Explanation 的 gain 一半来自 explanation 引导的 demos，一半来自 weighting scheme。如果只看 all-weighted rewards 跨条件比较，condition 间无显著差异（$p \geq .183$），只剩 directional trend favoring Explanation。换句话说，weighting 是 ASQ 不可分割的一部分，单纯解释没 weighting 不够，单纯 weighting 没解释也不够。

### Behavioral data 的额外发现（Appendix E, Fig. 9, 10）

- 在 Explanation 条件下，未点名的 features 的 per-participant std 显著增大（laptop feature 在 human task 下：Unguided 0.022, Rollout 0.031, Explanation 0.099——4.5× 增长）。说明 participant 听到 "focus on human" 后真的放松了其他 features 的控制，符合 β 模型的预测。
- 但是**underspecified feature 自身的 std 也增大**：human task 从 0.022 → 0.099。Paper 诚实指出这是个 limitation：prompt "I am uncertain about distance to the laptop" 有歧义——有人理解为 "保持安全距离"（push 高值），有人理解为 "尽量远离"（push 更高值），两个方向互相抵消。这是 future work 的一个明确入口：pair feature name + visualization of intended behavior。

### 为什么 Rollout 没用

Rollout 条件让 robot 先演一遍当前 learned policy 再让人 teach。直觉上应该有帮助——user 能看到 robot 哪里做错了。但数据显示 Rollout 跟 Unguided 没差别。Paper 推测：人看 rollout 时不知道错在哪个 feature 上，他们看到的是个 composite failure，要 mental 反推 "哦原来 laptop 距离是问题"。这个 cognitive overhead 抵消了 behavioral feedback 的好处。Huang et al. 2019 "Enabling robots to communicate their objectives" (https://link.springer.com/article/10.1007/s10514-018-9771-0) 之前也观察到类似现象。这印证了 paper 的核心 thesis：**直接命名 uncertain feature 比让 user 从 composite behavior 反推更有效**。

---

## 10. Limitations 和我想强调的几个点

Paper 自己列的 limitations：
- Binary 静态 attention：实际 attention 在一条 trajectory 内会 fluctuate
- Conditional independence 假设：features 实际相关
- 只用 kinesthetic demos：可扩展到 language + comparisons
- 12 participants 的小样本
- 固定 object set 的 manipulation domain

我自己想加的几点直觉：

(a) **Reference distributions 是这条思路的 bottleneck**：构造 $P(\sigma_i^2 \mid o_i, \phi_i)$ 需要枚举 $\phi$ 子集，每个跑 20 demos 500 次采样——指数复杂度。当 $k$ 大（比如 30+ features）这会爆炸。LLM filtering 把它压到 task-relevant 小集合，但根本问题没解决。一个可能 direction：用 neural net 学一个 "variance predictor" 给定 $(\theta, \beta, \phi_i)$，避免 enumerate。

(b) **β_low / β_high 是 hyperparameter**：paper 用 held-out validation 调出来，但真实 deployment 没有这个 validation set。能否用 demo 数据自己 estimate $\beta$？Bobu et al. 2020 T-RO 部分回答了这点，但跟 ASQ 的 integration 没做。

(c) **跟 RLHF/constitutional AI 的连结**：RLHF 也面临 "preferences 在不同 dimensions 上 underspecified" 的问题——reward model 学到的高 reward trajectory 经常在某个 dimension 上 misbehave (efficiency over safety)。ASQ 的 variance-based detection 原则上能 port 过去：观察 reward model 对比下的 human preferences 在哪些 feature 维度上 noisy，针对这些维度主动 query。这跟 Anthropic 的 recursive reward modeling 思路有 overlap。

(d) **ASQ vs DAgger 的对比缺失**：DAgger (Ross et al. 2011, https://www.cse.wustl.edu/~sgupta/cse517a/DAgger.pdf) 也是 interactive learning，但 DAgger 的 query 是 state-level（让 expert 标注），ASQ 是 feature-level（让 expert 关注某个 feature）。两者其实可以结合：ASQ 决定 "下次 query 时强调哪个 feature"，DAgger 决定 "query 哪个 state"。

(e) **Variance 是 attention 的一个 proxy，可能不是最好的**：考虑一个 counter-example——演示者非常在意 laptop 距离，但每次都把它推到刚好 10cm 处。Variance 低，但 signal 强。所以 variance-based detection 在 "feature 值被 optimize 到 fixed target" 时工作，在 "feature 值在大范围内 monotone 优化" 时也工作，但在 "feature 是 boolean constraint (撞/不撞)" 时可能 collapse。Paper 的 4 个 features 都满足前者，所以这个潜在 failure mode 没被 stress test。

(f) **Explanation template 太朴素**：固定模板 "I am uncertain how to handle [feature]"。User study 的 limitation section 暴露了——人对 "uncertain about distance to laptop" 的解读分歧很大。一个更结构化的 explanation（配 visualization、配 example trajectories、配 counterfactual "如果我这样做你会满意吗"）可能能把 H1 的 effect 再放大。Counterfactual state explanations (Peng et al. 2023 ICML, https://proceedings.mlr.press/v202/peng23c.html) 是这个方向的前置工作。

(g) **跟 active learning 的 formal 关系**：ASQ 的 query selection 是 "which features to ask about"，而 classical active learning 是 "which data points to label"。ASQ 在 feature space 而不是 data space 做 active selection。这其实更接近 Bayesian Optimal Experimental Design (Lindley 1956)，可以 formalize 为 mutual information $\arg\max I(\theta^*; \text{query response})$。Paper 没走这条 formal 路线，走的是 "detect + targeted re-demo" 的工程路线，effectiveness 在实验里证明。

---

## 11. 一句话总结 + 在 IRL/HRI lineage 里的位置

ASQ 是 Bobu et al. 2020 (misspecification quantification) + Cakmak & Thomaz 2012 (robot query design) + Bajcsy et al. 2018 (feature-at-a-time correction) + Hwang et al. 2025 (LLM-guided reward disambiguation, https://arxiv.org/abs/2511.14565) 的一个有机整合：用 per-feature β 显式建模 attention，用 cross-demo variance 做 Bayesian model selection 检测 underspecified features，用 LLM prior 过滤 irrelevant features，用 NL explanation 触发 targeted demos，最后用双 β 加权 IRL 把两套 demos 的信号正确组合。它的本质贡献是把 "identifiability problem" 从一个被动观测问题变成一个主动干预问题——你无法从数据里分辨 (A)/(B)，但你可以问。

把这个思路再推一步：在 LLM-as-agents 的时代，"agent 知道自己哪里不确定 + 知道怎么问" 这个 capability 跟 RLHF 后的 self-refinement 是同构的。robot 这里做的事和 constitutional AI 中"找出 reward model 最 uncertain 的 dimension 然后 ask for targeted feedback"在数学上是同一个 pattern，只是 modality 不同。

---

### 相关 reference 链接汇总

- MaxEnt IRL: Ziebart et al. 2008, https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf
- Guided Cost Learning: Finn et al. 2016, https://proceedings.mlr.press/v48/finn16.html
- RLHF: Christiano et al. 2017, https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html
- Feature-at-a-time corrections: Bajcsy et al. 2018, https://dl.acm.org/doi/10.1145/3171221.3171267
- Hypothesis space misspecification: Bobu et al. 2020 T-RO, https://ieeexplore.ieee.org/document/8967724
- Learning features for reward: Bobu et al. 2022 IJRR, https://journals.sagepub.com/doi/10.1177/02783649221078031
- Diagnosis-feedback-adaptation: Peng et al. 2023 ICML, https://proceedings.mlr.press/v202/peng23c.html
- Preference-conditioned language-guided abstraction: Peng et al. 2024 HRI, https://dl.acm.org/doi/10.1145/3610977.3634930
- Masked IRL: Hwang et al. 2025, https://arxiv.org/abs/2511.14565
- Robot query design: Cakmak & Thomaz 2012, https://dl.acm.org/doi/10.1145/2157689.2157693
- Active preference learning: Sadigh et al. 2017, https://www.roboticsproceedings.org/rss13/p53.pdf
- Communicating objectives: Huang et al. 2019, https://link.springer.com/article/10.1007/s10514-018-9771-0
- Cognitive load theory: Sweller 1988, https://doi.org/10.1016/0364-0213(88)90023-7
- Human-AI interaction guidelines: Amershi et al. 2019, https://dl.acm.org/doi/10.1145/3290605.3300233
- Estimating expertise: Beliaev & Pedarsani 2025 AAAI, https://ojs.aaai.org/index.php/AAAI/article/view/33705
- DAgger: Ross et al. 2011, https://www.cse.wustl.edu/~sgupta/cse517a/DAgger.pdf
- Andreea Bobu lab (IDEAL): http://idealab.mit.edu/
- Andi Peng's related work: https://andi-peng.github.io/
