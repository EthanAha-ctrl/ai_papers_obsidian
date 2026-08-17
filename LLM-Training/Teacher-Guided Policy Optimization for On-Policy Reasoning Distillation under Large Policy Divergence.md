---
source_pdf: Teacher-Guided Policy Optimization for On-Policy Reasoning Distillation
  under Large Policy Divergence.pdf
paper_sha256: e0f5ad90c2a68d843202f9d3f1720f1c8d5bd1b027386e30f45ceebe902bbb7c
processed_at: '2026-08-12T12:52:45-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 TGPO

## 一句话版本

**以前的 on-policy distillation 就像一个只会说"不对"的老师，你走错了它就骂你，但从来不告诉你该怎么走；TGPO 是一个会手把手在旁边说"下一步往这边走"的老师。**

---

## 为什么会有这个问题？

想象你在学做菜。有两种老师：

**老师 A（RKL 的做法）**：你炒菜，他在旁边看着。你放盐放多了，他说"不对"。你火候大了，他说"不对"。你问他"那该放多少盐？"，他还是只说"不对"。

你得自己一次次试，靠猜来摸索出正确做法。如果你们俩的口味差别很大（比如他是四川师傅，你是广东学徒），你可能十次有九次都被骂"不对"，但永远不知道正确方向在哪。

**老师 B（TGPO 的做法）**：你炒菜，他看着。你放完盐，他说"下一步该放花椒了"。你切完菜，他说"接下来该热油"。你走偏了他就在你当前的状态下给你下一步的具体建议。

显然老师 B 更有用。

---

## 具体在 LLM 里发生了什么？

训练一个 student model 做数学题，用一个大 teacher model 来"教"它。

**RKL 的做法**：让 student 自己做题，做完拿去给 teacher 打分。teacher 会算一个分数叫"我和你的概率比"。如果 student 写的东西 teacher 觉得还行，分数接近 0；如果 student 写的东西 teacher 觉得"根本不可能这么写"，给一个大负分。

问题在哪？

1. **只有惩罚没有指引**：teacher 只说"你这样写我不认可"，但不说"你应该怎么写"。student 得自己去茫茫词表里探索。

2. **惩罚无上限，奖励有上限**：teacher 说"不可能"的时候，负分可以无穷大（因为 teacher 概率接近 0，log 就爆炸）。但 teacher 说"很好"的正分最多也就那么大（因为 student 都这么生成了，说明 student 概率不低）。

3. **越差越崩**：student 越差，越容易写出 teacher 完全不认可的东西，负分越大，梯度越乱，student 更差，死循环。

论文里 Figure 2 展示了这个现象：用同家族的 teacher（Qwen2.5-Math-7B 教 1.5B），训练很稳定；用跨家族的 teacher（Qwen3-30B 教 Qwen2.5-Math-7B），训练直接崩了——reward 掉、gradient 爆炸、生成长度失控飙到几万 token。

---

## TGPO 怎么解决？

很简单：**让 teacher 在 student 每写一个 token 之后，告诉它下一个 token 应该写什么。**

具体流程：
1. student 写了 $y_{<t}$（前缀）
2. 拿这个前缀去问 teacher："接下来最该写什么？"
3. teacher 给出 $y_t^T$ = 它认为最该写的 token
4. 让 student 去提高自己生成 $y_t^T$ 的概率

这就跟 SFT 很像，但有本质区别：

- **SFT**：拿老师写好的标准答案来教，不管 student 现在走到哪了，硬塞标准答案
- **TGPO**：拿 student 自己的草稿，让老师在每个位置上批改"这里下一步该写啥"

好处是 student 走偏了，teacher 会说"从你现在的状态出发，该往这边走"，而不是非要让它回到标准答案的轨道上。这避免了 SFT 的 exposure bias 问题。

---

## 怎么跟 RLVR 结合？

RLVR 是用"最终答对答错"给 reward，但这个 reward 很稀疏——一整个 trajectory 才一个分数，每个 token 都用同一个分数。

TGPO 把两路信号合起来：

$$\text{总目标} = \text{RLVR奖励} + w \times \text{teacher指引}$$

- RLVR 告诉 student "这条路最终对不对"
- teacher 指引告诉 student "这一步该怎么走"

两者互补：一个看终点，一个看每一步。

---

## 为什么不直接把 teacher 指引当 reward 用？

因为对不上号。RLVR 的梯度是优化"student 生成的那个 token"的概率，但 teacher 的指引是关于"teacher 推荐的另一个 token"。相当于"我在练习左手写字，你给我右手的标准答案打分"，对不上。

所以 TGPO 用两套独立的梯度：RLVR 部分优化 student 采样 token 的 likelihood，teacher 指引部分单独做 cross-entropy 把 student 往 teacher 推荐的 token 拉。两个梯度加起来一起更新参数。

---

## guidance 权重为什么要 decay？

刚开始训练，student 很弱，需要老师多带；后面 student 有基础了，应该放手让它自己探索甚至超越老师。

所以 $w$ 从 0.002 开始线性衰减，第 200 步降到 0，后面纯靠 RLVR 自己跑。

论文做了 ablation：
- 一直不衰减：前期还行，后期被老师拽着，上不去
- 衰减太快：前期老师太强势，student 没机会自己探索
- 衰减到训练结束才归零：不如提前归零好
- 第 200 步归零（默认）：最好

这就是"先扶后放"：早期靠老师建基础，后期放手自我突破。

---

## 实验结果怎么样？

几个关键数字：

| 方法 | 数学平均 | 通用平均 |
|------|---------|---------|
| 原始 Qwen2.5-Math-7B | 19.0 | 15.4 |
| SFT（用 teacher 答案直接训） | 39.5 | 46.1 |
| GRPO++（纯 RLVR） | 43.4 | 52.1 |
| KDRL（RKL + RLVR） | 41.7 | 53.6 |
| OP Distill（纯 RKL） | 25.7（崩了） | 18.4（崩了） |
| LUFFY（mixed-policy，用 teacher 轨迹） | 44.4 | 56.4 |
| **TGPO** | **45.6** | **56.8** |

亮点：
1. **OP Distill 直接崩了**，证明 RKL 在大 gap 下根本不可用
2. **TGPO 超过 LUFFY**，虽然 LUFFY 用了 teacher 的完整轨迹信息，TGPO 只用 teacher 的 next-token 指引，反而更好
3. **OOD 泛化也好**，说明不只学到了数学套路

在更小的 1.5B model 上差距更大：KDRL 只有 4.7（完全废了），TGPO 33.5，比纯 RLVR 的 32.2 还高。

---

## 这篇 paper 的真正贡献

其实就一句话：**把 distillation 从"打分模式"切换到"指引模式"**。

之前的 OPD 都是把 teacher 当裁判用，TGPO 把 teacher 当向导用。这个视角转变解决了大 policy gap 下的训练崩溃问题。

更深层的启示：**当你有一个 expert 时，用它给方向比用它给分数更有价值**。这在 RL 里其实是个老话题（想想 imitation learning、DAgger 算法），但放到 LLM 的 OPD 范式里，结合 RLVR，是个很干净的 formulation。

---

## 局限

1. 只能用在有标准答案的任务上（数学、代码），开放生成没法用
2. 需要一个还不错的 teacher
3. 论文没探索 teacher 指引更长的 sub-trajectory（现在只给 next token，可能给下一步推理方向更有用）

---

## 可能的延伸联想

1. **跟 DAgger 的关系**：DAgger 就是让 expert 在 student 当前状态下给 action label，TGPO 本质就是 LLM 版的 DAgger + RL。

2. **跟 process reward model 的关系**：PRM 给每步打分，但也是 evaluative；如果 PRM 能给"下一步该咋办"的指引，就是 TGPO 思路。

3. **跟 self-play / self-distillation**：如果 teacher 就是 student 自己的某个版本，guidance 变成 self-guidance，可能省去大 teacher。

4. **跟 AlphaGo 的 MCTS 类比**：AlphaGo 也是用 policy network 给下一步建议 + value network 给终局评分，TGPO 的 teacher guidance + RLVR reward 结构上很像。

5. **guidance 不一定要 argmax**：teacher 给 soft distribution，student 学一个 KL 到 teacher 分布，但避免 RKL 的问题——这可能是 TGPO 的 soft 版本。

6. **guidance 可以层级化**：不只 token level，也可以是 phrase level、sentence level、reasoning step level。

---

相关链接：
- 论文 arXiv（具体编号需查作者 repo）
- GKD: https://arxiv.org/abs/2306.13649
- KDRL: https://arxiv.org/abs/2506.02208
- LUFFY: https://arxiv.org/abs/2504.14945
- GRPO/DeepSeekMath: https://arxiv.org/abs/2402.03300
- Thinking Machines OPD blog: https://thinkingmachines.ai/blog/onpolicy-distillation
- verl 框架: https://github.com/volcengine/verl
- DAgger 原始论文: https://arxiv.org/abs/1011.0686

---

# TGPO: Teacher-Guided Policy Optimization 深度讲解

## 1. 核心问题：RKL-based OPD 在大 policy divergence 下的失效

这篇论文的核心 insight 在于揭示了 **Reverse KL (RKL) 作为 on-policy distillation 的 intrinsic reward 存在结构性的缺陷**，尤其是在 teacher 与 student 的 policy 分布差距较大时。

### 1.1 背景设定

给定 prompt 数据集 $\mathcal{D} = \{x\}$，student policy $\pi_\theta(\cdot|x)$，teacher policy $\pi_T(\cdot|x)$。经典的 OPD 目标 (MiniLLM, GKD) 是最小化 RKL：

$$
\mathcal{T}_{\mathrm{RKL}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} D_{\mathrm{KL}}(\pi_\theta \| \pi_T) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ \log \frac{\pi_\theta(y|x)}{\pi_T(y|x)} \right]
$$

这里 $y$ 从 student policy 采样（on-policy 的关键）。这等价于一个 RL 目标，其中 intrinsic reward 为：

$$
r(y) = -\log \frac{\pi_\theta(y|x)}{\pi_T(y|x)} = \log \frac{\pi_T(y|x)}{\pi_\theta(y|x)}
$$

### 1.2 RKL 的两种 regime

定义 density ratio $\rho(y) = \frac{\pi_\theta(y|x)}{\pi_T(y|x)}$，intrinsic reward $r(y) = -\log \rho(y)$。由于 $y$ 从 $\pi_\theta$ 采样，所以采样分布集中在 student 高概率区域，产生两种主要 regime：

**Consensus regime** ($\rho(y) \approx 1$)：student 和 teacher 都给该 trajectory 高概率，reward 接近 0（neutral），RL 自然 reinforce。

**Rejection regime** ($\rho(y) \gg 1$)：student 高概率但 teacher 近零概率。$-\log \rho(y) \to -\infty$，产生巨大的负 reward，但**只告诉 student "这个不好"，没有告诉 student "该往哪里走"**。

这是 RKL 的根本问题：它是 **evaluative supervision**（事后评判），缺乏 **directional guidance**（方向指引）。Student 必须自己通过 trial-and-error 探索出 teacher 偏好的 trajectory，这在 cross-family 这种 reasoning style 差异大的场景下计算上几乎不可行。

### 1.3 量化的不稳定性分析（Appendix A）

论文在 Appendix A 给出了严格的梯度分析。对单条 trajectory $y$，policy gradient estimator 为：

$$
\hat{g}(y) \propto \nabla_\theta \log \pi_\theta(y|x) \cdot (-\log \rho(y))
$$

在 Rejection regime 中，设 $\pi_\theta(y_{\mathrm{bad}}|x) \geq \delta$（student 给一定概率），$\pi_T(y_{\mathrm{bad}}|x) \leq \epsilon$（teacher 几乎不给概率），则：

$$
\log \rho(y_{\mathrm{bad}}) \geq \log \delta - \log \epsilon = \log\left(\frac{\delta}{\epsilon}\right) \xrightarrow{\epsilon \to 0} +\infty
$$

梯度 scaling factor $|\log \rho(y_{\mathrm{bad}})|$ 无界，导致梯度方差二阶矩：

$$
\mathbb{E}_{y \sim \pi_\theta} \left[ \|\nabla_\theta \log \pi_\theta(y|x)\|^2 \cdot (\log \rho(y))^2 \right]
$$

发散。这就是 Figure 2 中观察到的 gradient norm spike 和 length explosion 的根本原因。

更微妙的是 reward scaling 的 **asymmetry**：正 reward 要求 $\pi_T \gg \pi_\theta$，但 $y$ 是从 $\pi_\theta$ 采样的，所以大正 reward 的事件概率趋于 0；而负 reward 只要 $\pi_T \to 0$ 就无界增大，且这些"bad"样本由于从 $\pi_\theta$ 采样，出现频率反而高。优化被频繁的、量级巨大的负 update 主导。

**Intuition**：RKL 优化像是在一个"只罚不奖"的环境中训练，而 cross-family 设置下，student 几乎所有探索都踩在 teacher 的低概率区域，所以一直在被"无方向地"惩罚。Adam 这种 adaptive optimizer 在梯度二阶矩爆炸时会彻底失稳。

---

## 2. TGPO 方法详解

### 2.1 核心 idea：从 evaluative 到 directional

TGPO 的关键转变：**不再用 teacher 去评估 student 已经生成的 token，而是让 teacher 在 student 已生成的前缀上预测下一个最优 token，直接告诉 student "下一步该生成什么"**。

### 2.2 Teacher guidance 的构造

给定 student 自回归生成的 trajectory $y \sim \pi_\theta(\cdot|x)$，每个 token $y_t$ 基于 prefix $y_{<t}$ 采样。对每个 student 已访问的 state $y_{<t}$，query teacher 得到其最高概率的 next token：

$$
y_t^T = \arg\max_{v \in \mathcal{V}} \pi_T(v | x, y_{<t})
$$

其中 $\mathcal{V}$ 是 vocabulary。注意这里 teacher 是在 **student 的实际 context** 上做条件预测，所以 guidance 是 **dynamic** 的、随 student 状态变化而修正。这一点与 offline SFT 的 teacher-forcing 本质不同：SFT 的 target 是静态 ground truth，而 TGPO 的 target 是 teacher 对 student 当前真实状态的"下一步建议"，从而天然避免了 exposure bias。

所有 teacher target 可以通过单次 teacher forward pass 计算（因为 teacher 给的是 next-token argmax，不需要 iterative decoding）。

### 2.3 Guidance objective

$$
\mathcal{I}_{\mathrm{G}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ -\sum_{t=1}^{|y|} \log \pi_\theta(y_t^T | y_{<t}) \right]
$$

这本质上是一个 cross-entropy loss，但 target $y_t^T$ 是 teacher 在 student 轨迹上动态给出的，source distribution 是 $\pi_\theta$（on-policy）。

### 2.4 与 GRPO 的集成

GRPO objective（省略 reference KL）：

$$
\mathcal{I}_{\mathrm{RL}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_\theta} \left[ \frac{1}{Z} \sum_{i=1}^G \sum_{t=1}^{|y_i|} \rho_{i,t}(\theta) A_i \right]
$$

其中：
- $Z = \sum_i |y_i|$ 是 token 总数归一化
- $\rho_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}|x, y_{i,<t})}$ 是 importance sampling ratio（PPO 风格）
- $A_i = \frac{r_i - \mu}{\sigma}$ 是 group-normalized advantage

**关键设计选择**：为什么不把 guidance 作为 reward shaping？因为 GRPO 更新的 action 是 student 采样的 token $y_{i,t}$，而 guidance 信号是定义在 teacher target token $y_{i,t}^T$ 上的。如果用 guidance score 作为 scalar reward，会导致 "优化的 action" 与 "监督的 target" 不匹配（mismatch between optimized action and supervised target）。

因此 TGPO 选择 **differentiable regularization** 形式：

$$
\mathcal{I}_{\mathrm{TGPO}}(\theta) = \mathcal{I}_{\mathrm{RL}}(\theta) + w \mathcal{I}_{\mathrm{G}}(\theta)
$$

其中 $w$ 控制 guidance 强度。

### 2.5 Guidance weight 的 linear decay

$$
w_t = \max(w_{\mathrm{init}} - \delta \cdot t, 0)
$$

其中 $w_{\mathrm{init}}$ 是初始权重，$t$ 是训练步数，$\delta$ 是衰减率。论文默认 $w_{\mathrm{init}} = 2 \times 10^{-3}$，$\delta = 10^{-5}$，在 step 200 时降为 0，剩余训练步数进入纯 RLVR 阶段。

**Intuition**：早期需要强 guidance 来快速对齐 teacher 的 reasoning style（避免 RKL 的 rejection regime）；后期 student 已经有了基本能力，需要更多自由探索来超越 teacher 的 ceiling，所以 guidance 应该退场。Figure 6 的 ablation 显示：
- Constant weight：早期 competitive 但早期 plateau，说明持续 imitation 约束阻碍 reward 优化
- Aggressive annealing（$w_{\mathrm{init}} = 2 \times 10^{-2}$）：早期 reward 被压制，guidance 过强限制了探索
- Continuous annealing（衰减到训练结束）：不如提前归零的好，说明在训练结束前进入纯 RL 阶段很重要
- Ours（step 200 归零）：最佳

---

## 3. 实验结果分析

### 3.1 Main results (Table 1)

基于 Qwen2.5-Math-7B student + Qwen3-30B-A3B teacher：

| Method | ID Avg | OOD Avg |
|--------|--------|---------|
| SFT | 39.5 | 46.1 |
| LUFFY (mixed-policy) | 44.4 | 56.4 |
| GRPO++ | 43.4 | 52.1 |
| KDRL (RKL-based OPD) | 41.7 | 53.6 |
| OP Distill (RKL-based) | 25.7 | 18.4 (collapse!) |
| TGPO w/o annealing | 44.1 | 56.0 |
| **TGPO** | **45.6** | **56.8** |

值得注意的几点：
1. **OP Distill 完全崩溃**（25.7 vs 原始 Qwen2.5-Math-7B 的 19.0，提升有限甚至有些 benchmark 倒退），这印证了 RKL 在 cross-family 大 divergence 下的失稳。
2. **TGPO 超越 LUFFY**（mixed-policy），这个结果相当 striking，因为 LUFFY 用到了 teacher 的 trajectory 作为 auxiliary supervision，信息量看似更多。TGPO 纯 on-policy 反而更好，说明 on-policy 的 distribution consistency 优势。
3. **OOD 上 TGPO 也最强**（56.8），尤其在 ARC-c（82.8 vs LUFFY 80.1），说明 teacher-guided 探索不只在数学上 work，还能迁移。

### 3.2 Training dynamics (Figure 4)

- **Training reward**：TGPO 稳定增长收敛；OP Distill 早期 reward collapse；KDRL 低于 GRPO++，说明 RKL 即使与 RLVR 结合仍拖累优化
- **Response length**：OP Distill 严重 length explosion（>20k tokens）；TGPO 与 GRPO++ 持平，稳定在合理范围
- **Gradient norm**：OP Distill、KDRL、LUFFY 都有高方差；TGPO 平稳

LUFFY 训练 reward 看似最高但作者指出这可能是 inflated，因为 LUFFY 在每个 group 中塞入 ground-truth sample，导致 advantage 计算偏乐观。

### 3.3 不同 teacher 的 ablation (Table 2)

| Teacher | AMC | MATH | Olympiad | GPQA* | Avg |
|---------|-----|------|----------|-------|-----|
| No Teacher | 58.3 | 82.2 | 47.3 | 32.3 | 55.0 |
| R1-Distill-Qwen-32B | 57.8 | 83.4 | 47.4 | 40.9 | 57.4 |
| Qwen3-30B-A3B | 60.2 | 84.4 | 49.8 | 37.4 | 58.0 |

TGPO 对 teacher 选择 robust，不同 teacher 在不同 benchmark 上各有优势（R1-Distill 在 GPQA 更强，Qwen3 在数学更强），说明 TGPO 能有效 transfer teacher 的 strengths，不依赖特定 teacher 架构。

### 3.4 1.5B 小 model 的极端 case (Table 3, Appendix C)

用 Qwen2.5-Math-1.5B 作为 student，teacher-student gap 更大：

| Method | Avg |
|--------|-----|
| KDRL | 4.7 (catastrophic) |
| OP Distill | 16.2 |
| GRPO++ | 32.2 |
| LUFFY | 25.7 |
| **TGPO** | **33.5** |

KDRL 在这种极端 divergence 下几乎完全失败（4.7），RKL 长度爆炸 saturate 8192 token rollout limit。TGPO 不仅稳定，还超过 GRPO++ 1.3 个点，这验证了 guidance 在大 gap 下提供了关键的方向信息。

### 3.5 In-family 设置 (Figure 5)

用 Qwen2.5-Math-7B 教 Qwen2.5-Math-1.5B（同 family，分布接近）：
- OP Distill 稳定（RKL 在 in-family 下 work）
- KDRL 反而 fail（与 RKL 失败 case 类似，说明 RKL+RLVR 的组合即使在 in-family 也不稳定）
- TGPO 增长最快且持续领先

这说明 TGPO 不只在大 divergence 下 work，在小 divergence 下也优于 RKL。

---

## 4. 关键技术细节与直觉

### 4.1 为什么 guidance 不用 reward shaping？

GRPO 在 token $y_{i,t}$ 上做 policy gradient update：$\nabla_\theta \log \pi_\theta(y_{i,t}|...)$，而 guidance 信号是关于 $y_{i,t}^T$ 的。如果用 reward shaping $A_i \leftarrow A_i + \lambda \cdot (\text{teacher score})$，优化的 action（student 的 $y_{i,t}$）和监督的 target（teacher 的 $y_{i,t}^T$）不一致，相当于"在 A action 上用关于 B 的 reward 信号做梯度"，这是 misaligned 的。

Differentiable regularization 则分别处理：RL 部分在 $y_{i,t}$ 上优化 advantage，guidance 部分在 $y_{i,t}^T$ 上做 cross-entropy。两个梯度信号各自指向正确的方向，通过 $w$ 线性组合。

### 4.2 Teacher forward pass 的 efficiency

所有 $y_t^T$ 可以一次 teacher forward pass 得到，因为对每个 prefix $y_{<t}$，teacher 计算 $\pi_T(v|x, y_{<t})$ 并取 argmax，这是标准 next-token prediction，teacher 的 KV cache 在生成 $y$ 时已经计算过所有 prefix 的 hidden states（如果 teacher 是用来打分的）。论文这里没详说，但实际实现中 teacher 的 logits 可以从单次 forward 中批量获取。

### 4.3 与 SFT teacher-forcing 的本质区别

SFT：target 是固定的 ground truth $y_t^*$，source distribution 是 data distribution。
TGPO guidance：target $y_t^T = \arg\max_v \pi_T(v|x, y_{<t})$ 依赖 student 实际生成的 $y_{<t}$，source distribution 是 $\pi_\theta$。

这个区别决定了 TGPO 没有 exposure bias：student 走偏了，teacher 会在偏掉的状态上给"从这个状态出发该怎么走"的建议，而不是固执地要求回到 ground truth 轨迹。

### 4.4 Annealing 的深层意义

早期 high $w$：guidance 主导，student 快速学会 teacher 的 step-by-step reasoning pattern，避免随机探索踩进 RKL 的 rejection zone。
后期 $w \to 0$：纯 RLVR 主导，student 在已建立的 reasoning 基础上通过 outcome reward 自我优化，有机会超越 teacher。

这其实暗合 curriculum learning 的思想：先模仿（低 variance，高 bias），后探索（高 variance，低 bias）。

---

## 5. 局限性与未来方向

论文承认：
1. TGPO 依赖 verifiable reward，对开放生成任务（无 reliable verifier）适用性不明
2. 需要一个 capable teacher 提供 token-level guidance

可能的扩展：
- 用 learned reward model 替代 rule-based verifier
- 用 LLM judge 作为 teacher guidance source
- 探索 multi-step teacher reasoning（teacher 不只给 next token，而是给 next sub-trajectory）

---

## 6. 相关工作脉络

- **MiniLLM** (Gu et al., 2023): 首次提出 RKL-based OPD
- **GKD** (Agarwal et al., 2024): 统一 forward KL 和 reverse KL 的 OPD 框架
- **KDRL** (Xu et al., 2025): RKL + RLVR 的两种整合方式
- **Thinking Machines OPD blog** (Lu & Lab, 2025): OPD 作为 post-training 范式的成本分析
- **LUFFY** (Yan et al., 2025): mixed-policy，用 teacher trajectory 作为 auxiliary supervision
- **OPSD/SDPO** (2026): self-distillation 变体

TGPO 在这条线上首次明确指出 RKL 的 directional guidance 缺陷，并用 teacher argmax target 解决，是一个从 "evaluation" 到 "guidance" 的范式转变。

---

## References

- 论文本身: arXiv (具体链接需查)
- MiniLLM: https://arxiv.org/abs/2306.08543
- GKD (On-Policy Distillation): https://arxiv.org/abs/2306.13649
- KDRL: https://arxiv.org/abs/2506.02208
- Thinking Machines OPD blog: https://thinkingmachines.ai/blog/onpolicy-distillation
- LUFFY: https://arxiv.org/abs/2504.14945
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Qwen2.5-Math: https://arxiv.org/abs/2409.12122
- Qwen3: https://arxiv.org/abs/2505.09388
- OpenR1: https://github.com/huggingface/open-r1
- verl framework: https://github.com/volcengine/verl
- vLLM: https://github.com/vllm-project/vllm
- PRIME: https://arxiv.org/abs/2502.01456
- DAPO: https://arxiv.org/abs/2503.14476
- SimpleRL: https://hkust-nlp.notion.site/simplerl-reason
- MMLU-Pro: https://arxiv.org/abs/2406.01574
- GPQA: https://arxiv.org/abs/2311.12022
- ARC: https://arxiv.org/abs/1803.05457

---

## 7. 总结直觉

TGPO 的核心贡献是把 distillation 的角色从 **"评判者"** 重新定义为 **"向导"**。RKL 像一个只会说"错"的老师，student 走偏了就给负分，但不说该怎么走；TGPO 像一个在 student 身边的老师，student 每走一步，老师在当前状态下指出下一步该往哪走。在 teacher-student 差距大、reasoning style 不同的场景下，这种 directional guidance 比 evaluative supervision 有效得多，因为它直接降低了探索空间的维度。

Linear decay schedule 则体现了"先扶后放"的教育哲学：早期靠 teacher 指路快速建立正确 reasoning pattern，后期放手让 student 在 outcome reward 下自我优化，最终有机会突破 teacher 的能力 ceiling。
