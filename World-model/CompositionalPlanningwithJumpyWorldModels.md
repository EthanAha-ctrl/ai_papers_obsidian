---
source_pdf: CompositionalPlanningwithJumpyWorldModels.pdf
paper_sha256: b39e36622c6b94a0e2ea7ab3dcbf8274f642b5c720586f3c665a76c151a4f92f
processed_at: '2026-08-03T16:49:35-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 一句话概括

你有一堆预训练好的policy，每个都只会做一件简单事（比如"往左走"或"抓起红方块""）。这篇paper教你：**不用再训练任何东西，在test time把这些policy像乐高一样拼起来，就能完成任何一个policy单独都做不到的复杂long-horizon任务**。

---

## 核心idea的生活类比

想象你在一个陌生城市，要从A点去B点，但你只会几个简单技能："往东走50米"、"往西走50米"、"过马路"、"进地铁站"。单靠任何一个技能都到不了B点。

**传统做法（action-level planning）**：每走一步就重新规划下一步，像MPC那样。问题是走多了误差累积，200步之后你完全不知道自己在哪。

**这篇paper的做法**：你先学一个"jumpy world model"——给它一个起点和一个policy，它能直接告诉你"如果按这个policy走，随机某个时刻停下来，你大概会在哪个区域"。然后你用这个模型去模拟"先往东走一段、再过马路、再进地铁"这种组合，挑出最可能到B点的组合，直接执行。

"jumpy"就是"跳跃"——不一步步预测，直接跳到未来某个随机时刻看自己在哪。这个随机时刻服从geometric distribution（每步以概率$1-\gamma$停下来）。

---

## 三个核心创新分别说人话

### 创新1：TD-HC（跨horizon一致性）

**问题**：你要训练一个模型能预测任意horizon（$\gamma$从0到0.996）下的state distribution。但horizon越长，variance越大，模型容易崩。

**直觉insight**：长horizon和短horizon之间有天然联系。如果你能准确预测"走10步后在哪"（短horizon $\beta$），你就可以用这个信息来约束"走100步后在哪"（长horizon $\gamma$）的预测——因为100步可以拆成"先走10步、再从那走剩下的"。

**具体怎么做**：公式(4)告诉你：
$$m_\gamma^\pi = (1-\gamma)P + \gamma\frac{1-\gamma}{1-\beta}(\text{短horizon从next state看}) + \gamma\frac{\gamma-\beta}{1-\beta}(\text{长horizon从短horizon终点看})$$

变量含义：
- $m_\gamma^\pi$：长horizon下的state distribution
- $P$：one-step transition
- $\beta$：短horizon的discount
- $\gamma$：长horizon的discount
- 第一项：立即halt
- 第二项：走一步后用短horizon看
- 第三项：用短horizon的sample作为起点继续长horizon看

**训练trick**：只对batch里25%（antmaze）或12.5%（cube）的样本应用这个consistency loss。因为consistency要sample from模型自己的prediction，error会compound，所以"少用但用"。

**效果**：在antmaze-giant上，EMD（earth mover's distance，衡量两个distribution的距离）从7.29降到5.25，降28%。可视化看：没有consistency的模型在horizon=500步时sample会"穿墙"，有consistency的尊重topological constraint。

### 创新2：Geometric Switching Policy (GSP)的分解定理

**问题**：你想evaluate"先执行policy A一段时间、再切到B、再切到C"这种组合policy的value，怎么做？

**核心insight**：把"切换"建模成geometric distribution——执行policy $\pi_{z_k}$时，每步以概率$\alpha_k$切换到下一个policy，以概率$1-\alpha_k$继续。这种geometric结构让math very clean。

**Theorem 1告诉你**：GSP的successor measure可以分解成$n$个component的mixture：

$$m_\gamma^\nu = \sum_{k=1}^n w_k \cdot (\text{第}k\text{个policy的occupancy经过前面所有policy的过渡})$$

其中权重$w_k$有closed form：
$$w_k = \frac{1-\gamma}{1-\beta_k}\prod_{i=1}^{k-1}\frac{\gamma-\beta_i}{1-\beta_i}$$

变量：
- $w_k$：第$k$个policy对最终结果贡献的权重
- $\beta_k = \gamma(1-\alpha_k)$：第$k$个policy的"有效discount"——既要episode不halt（$\gamma$），又不切换policy（$1-\alpha_k$）
- 乘积项：前$k-1$个phase都"存活且没切换"的累积概率

**为什么这个important**：有了closed-form分解，你可以用单样本Monte Carlo来estimate组合policy的value（Lemma 1）。Sample一条路径：从起点用第一个GHM sample下一个waypoint、用第二个GHM sample再下一个...最后weighted sum reward。整个过程differentiable、tractable、unbiased。

**比之前工作的generalization**：
- GGPI (Thakoor et al. 2022)：所有$\alpha$都相同，且只用2个固定horizon的GHM
- γ-models (Janner et al. 2020)：只有1个policy，不组合
- 本文：每个position的$\alpha_k$可以不同，policy可以不同，horizon是连续的，最多24个policy组合

### 创新3：Compositional Planning = 统一框架

**核心观察**：planning objective
$$\max_{a_1, z_1, \ldots, z_n} Q_\gamma^{\pi_{z_1}\xrightarrow{\alpha_1}\pi_{z_2}\ldots}(s, a_1)$$

通过调整switching probability $\alpha_i$，你能recover三种existing method：

| $\alpha$设置 | 对应方法 | 含义 |
|---|---|---|
| $\alpha_1=\cdots=\alpha_n=1$ | ActionPlan | 每步都切，相当于MPC over primitive actions |
| $\alpha_1=1, \alpha_{2..n}=0$ | GPI (Barreto et al. 2017) | 选一个policy一直执行 |
| $\alpha_1=\cdots=\alpha_{n-1}=\alpha$ | GGPI | 固定间隔切换 |
| 任意$\alpha_i \in (0,1)$ | **CompPlan (本文)** | 灵活组合 |

**优化方法**：random shooting。用GHM自己作为proposal distribution来sample candidate sequence：
- 在antmaze用goal-conditioned proposal（chaining GHM预测朝goal的waypoint）
- 在cube用unconditional proposal（从behavior policy的GHM直接sample）
- Sample 256或1024个candidate，每个用Lemma 1 estimate $\hat{Q}$，挑最大的

---

## 为什么有效——intuition深层

### 1. Successor measure = value function的generative reformulation

传统RL的value function $Q_\gamma^\pi(s,a)$需要sum over all future rewards。这篇paper说：只要你能sample from $m_\gamma^\pi(\cdot|s,a)$（halting time的state distribution），value就是$(1-\gamma)^{-1}\mathbb{E}[r(S^+)]$。

**这把"estimating value"变成了"generative modeling"**。你可以用任何强大的generative model（flow matching、diffusion、VAE等）来model policy的行为。Reward function可以在test time任意指定，因为你的model学的是"policy会到哪"，不是"policy的value多少"。

### 2. Jumpy prediction避免compounding error

Model-based RL的classic问题：你learn一个one-step model $P(s'|s,a)$，然后rollout 100步来plan。每一步都有小error，100步后error爆炸。

Jumpy world model直接predict"100步后我在哪"的distribution。没有rollout，没有compounding。代价是你需要off-policy数据来learn这个long-horizon distribution——TD learning提供这个能力（Bellman equation是fixed point）。

### 3. Flow matching解决bootstrap stability

TD learning需要bootstrap：用model自己的prediction当target。Diffusion model在long horizon有systemic bias（Farebrother et al. 2025发现）。Flow matching通过designing probability path（optimal transport path $X_t = (1-t)X_0 + tS'$）让training更stable。

**TD-HC进一步**：用短horizon的prediction作为长horizon的"checkpoint"。这和consistency model (Song et al.)、shortcut model (Frans et al.)的core idea类似——用accurate short-time prediction anchor long-time prediction。

### 4. Policy composition > policy selection

GPI只select一个最好的policy。CompPlan可以sequence多个policy。实验数据：

- CompPlan vs GPI：**89% relative improvement**（averaged over long-horizon tasks）
- CompPlan vs ActionPlan：**201% relative improvement**

这说明两件事：
- Policy selection alone不够（GPI差）→ composition matters
- Action-level planning不够（ActionPlan差）→ temporal abstraction matters
- 两者结合（CompPlan）才best

### 5. Zero-shot capability ≠ Composition utility

最counterintuitive的发现：zero-shot表现差的policy，composition后可能很强。

例子：GC-BC在antmaze-large上zero-shot只有18%成功率，但CompPlan后73%。相反，CRL在antmaze上zero-shot很强（84%），但在cube上接近0%，composition后能到39-73%。

**Insight**：policy的"composability"是orthogonal dimension到zero-shot capability。一个policy可能zero-shot不行，但它的"building block"性质很好，组合起来很强。这意味着评估一个foundation policy不应该只看zero-shot，还要看它能否被composition利用。

---

## 实验数据说人话

### 主结果（Table 1）

挑几个eye-catching数字：

| Task | Base policy | Zero-shot | CompPlan | 提升 |
|---|---|---|---|---|
| Antmaze-giant | HFBC | 42% | 79% | +37pp |
| Cube-4 | GC-BC | 0% | 76% | +76pp |
| Cube-3 | GC-TD3 | 12% | 83% | +71pp |
| Antmaze-large | GC-BC | 18% | 73% | +55pp |

最极端：cube-4上从0%到76%，相当于从完全做不到到76%成功率。

### vs Hierarchical baselines（Table 2）

CompPlan不需要task-specific training，却超越HIQL和SHARSA（OGBench上的SOTA）：

| Task | HIQL | SHARSA | CompPlan |
|---|---|---|---|
| Antmaze-giant | 65% | 56% | **79%** |
| Cube-3 | 3% | 50% | **83%** |
| Cube-4 | 0% | 9% | **67%** |

**为什么test-time planning能超越learned hierarchy**：learned hierarchy在training时commit一种特定的hierarchical structure，如果test task需要不同的composition pattern就fail。CompPlan在test time灵活search over composition，适应arbitrary task structure。

### TD-HC效果

**Generative accuracy**（EMD越低越好，$\gamma=0.995$）：

| | td-flow | td-hc | 提升 |
|---|---|---|---|
| Antmaze-giant (GC-1S) | 7.29 | 5.25 | 28% |
| Cube-1 (GC-1S) | 1.43 | 1.33 | 7% |

**Planning performance**：td-hc vs td-flow只提升5%左右。

**为什么accuracy提升大但planning提升小**：planning实际用的effective horizon是$\beta_i \in [0.98, 0.99]$，即50-100步。td-hc的优势在200+步才显著。当前benchmark上50-100步的td-flow已经足够准确。未来更长horizon的benchmark上td-hc会更important。

### Ablation：replan频率

每步replan vs 每5步replan：平均差20%。大多数domain差异小，但某些case（CRL on cube）5-step会崩。这是一个speed-performance trade-off lever。

### Ablation：action optimization

最大化$Q(s,a_1)$（同时选action和policy sequence）vs 只最大化$V(s)$（只选policy sequence）：
- max-Q平均好70%
- max-V在diffuse policy（如CRL on cube）上fail，因为sample action时太random

**Insight**：test-time planning最好同时优化action和policy sequence，特别是base policy stochastic的时候。

---

## 方法局限与未来方向

### 局限

1. **$\alpha_k$是hyperparameter**：当前switching probability手动设定。未来可以learn state-dependent $\alpha(s)$，让planner自适应decide何时切换。

2. **Policy和GHM分开训练**：先训policy再训GHM。Joint training可能让GHM更准确predict policy行为，policy也更"composable"。

3. **Random shooting效率有限**：256-1024个candidate够当前benchmark，但更复杂task可能需要MCTS等更sophisticated search。

4. **TD-HC的benefit在当前benchmark未完全发挥**：因为实际planning用50-100步horizon，而td-hc的优势在200+步。更长horizon的task会更能体现价值。

5. **只测了goal-conditioned policy**：Framework支持arbitrary parameterized policy，但实验只用goal-conditioned。Unsupervised RL学出的skill集合是否能compose同样有效，值得验证。

### 未来方向（paper提到+我的联想）

1. **State-dependent switching**：$\alpha_k(s)$让planner能"看到当前state再决定切不切"，更接近options framework。

2. **Joint policy+GHM learning**：当前是sequential（先policy后GHM），joint可能mutually beneficial——policy学得"更好predict"，GHM学得"更准policy行为"。

3. **Latent space GHM**：在learned latent space做jumpy prediction，可能更sample efficient、更accurate（特别是vision-based task）。

4. **MCTS-style search**：替代random shooting，用tree search更高效explore composition space。

5. **LLM-style chain-of-thought planning**：每个policy是一个"reasoning step"，GHM预测step outcome，planning = search over reasoning chains。这和当前LLM agent的test-time compute scaling方向一致。

6. **Robotics foundation model adaptation**：π0、GR00T等VLA foundation model在long-horizon task上可能受限。Compositional planning提供test-time scaling path——不需要retrain foundation model，通过planning over its behaviors unlock long-horizon capability。

7. **Multi-modal GHM**：当前flow matching是continuous。对于discrete task structure（如"先开抽屉或先拿杯子"这种alternative path），可能需要mixture of flows或discrete flow matching（Gat et al. 2024）。

---

## 对RL领域的意义

这篇paper在几个direction的crossroad上：

1. **Model-based RL的进化**：从"predict one step + rollout"到"directly predict occupancy"。绕开compounding error这个classic problem。

2. **Hierarchical RL的alternative**：从"training时learn hierarchy"到"test time compose"。不需要task-specific training，更flexible。

3. **Foundation model adaptation**：foundation policy通常不能long-horizon reasoning。Compositional planning提供test-time scaling方法，类似于LLM的chain-of-thought但for control。

4. **Generative modeling + RL的深度融合**：flow matching、consistency model等generative technique直接解决RL的核心问题（value evaluation、planning）。

5. **Empirical validation的重要性**：之前的GGPI和γ-models在toy domain验证，本文在OGBench这种challenging benchmark上全面验证，并发现zero-shot capability ≠ composition utility这种non-obvious insight。

总体评价：这是一个非常elegant的framework，math clean、experiment strong、insight深刻。它把"如何使用预训练policy"这个问题reformulate成generative modeling + search problem，在foundation model时代提供了一种principled的test-time scaling方法。

---

# Compositional Planning with Jumpy World Models 深度解析

## 1. 高层直觉

这篇论文要解决的核心问题：**如何让agent在long-horizon任务中组合多个预训练policy**。

假设你有一堆预训练好的goal-conditioned policy $\{\pi_z\}_{z \in Z}$，每个policy擅长从当前state到某个subgoal $z$。单靠任何一个policy都无法完成复杂的长程任务（比如穿越大型maze或操作多个cube）。论文的insight是：如果你能预测每个policy在未来任意时刻会到哪儿（即**state occupancy distribution**），你就可以在test time通过planning来组合这些policy，无需任何额外训练或环境交互。

"jumpy"的含义：模型直接预测"跳跃"到未来某个geometrically distributed time的state，而不是一步一步rollout。这样既避免了compounding error，又自然地捕获了temporal abstraction。

参考链接：
- 论文: https://arxiv.org/abs/2502.09686 (Farebrother et al., TD-Flows)
- OGBench: https://arxiv.org/abs/2410.20092
- GGPI (Thakoor et al.): https://arxiv.org/abs/2204.12030
- γ-models (Janner et al.): https://arxiv.org/abs/2102.07315
- Successor Measure: https://arxiv.org/abs/2101.07123

---

## 2. 数学基础

### 2.1 Successor Measure

对于policy $\pi$ 和初始state-action pair $(s,a)$，successor measure定义为：

$$m_\gamma^\pi(X|s,a) = (1-\gamma)\sum_{k=0}^{\infty} \gamma^k \Pr(S_{k+1} \in X | S_0=s, A_0=a, \pi)$$

**变量解释**：
- $X \subseteq S$：state space的子集
- $\gamma \in [0,1)$：discount factor
- $k$：时间步
- $\gamma^k$：第$k$步的discount权重
- $1-\gamma$：normalization factor，确保这是probability distribution

**直觉**：把$\gamma$重新解释为"episode以概率$1-\gamma$终止"的几何分布。$m_\gamma^\pi(X|s,a)$就是"在halting time访问的state落在$X$中"的概率。

### 2.2 Value Function重参数化

公式(1)：
$$Q_\gamma^\pi(s,a) = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k r(S_{k+1}) | S_0=s, A_0=a, \pi\right] \equiv (1-\gamma)^{-1} \mathbb{E}_{S^+ \sim m_\gamma^\pi(\cdot|s,a)}[r(S^+)]$$

**关键insight**：value可以表示为"在halting time的reward乘以平均lifetime $(1-\gamma)^{-1}$"。这意味着如果你能sample from $m_\gamma^\pi$，你就能estimate任意reward function下的value——planning变成了generative modeling问题。

### 2.3 Bellman Equation for Successor Measure

公式(2)：
$$m_\gamma^\pi(\cdot|s,a) = (1-\gamma)P(\cdot|s,a) + \gamma \mathbb{E}_{S' \sim P(\cdot|s,a), A' \sim \pi(\cdot|S')}[m_\gamma^\pi(\cdot|S',A')]$$

这是一个mixture distribution：
- 以概率$1-\gamma$：halting立即发生，state来自one-step transition $P(\cdot|s,a)$
- 以概率$\gamma$：继续到next state-action pair，然后从那里看successor measure

---

## 3. Geometric Horizon Model (GHM) via TD-Flow

### 3.1 Flow Matching框架

GHM把successor measure建模为一个ODE（flow）：

$$\frac{d}{dt}\psi_t(X_0|s,a) = v_t(\psi_t(X_0|s,a)|s,a)$$
$$\psi_0(X_0|s,a) = X_0$$

其中：
- $X_0 \in \mathbb{R}^d$：从prior $p_0$采样的initial noise（通常是Gaussian）
- $v_t: \mathbb{R}^d \times S \times A \to \mathbb{R}^d$：time-dependent vector field
- $\psi_t$：flow map，把$X_0$推到时刻$t$的位置
- $t \in [0,1]$：flow time（与MDP time无关）

目标：$p_1 := \psi_1(\cdot|S,A)_\sharp p_0 = m_\gamma^\pi(\cdot|S,A)$，即flow在$t=1$时的分布等于successor measure。

### 3.2 TD-Flow Loss

公式(3)的TD-Flow loss有两项：

**第一项（one-step term）**：
$$(1-\gamma)\mathbb{E}\left[\|v_t(X_t|S,A;\theta) - (S'-X_0)\|^2\right]$$

这是conditional flow matching targeting one-step transition $P(\cdot|S,A)$，其中$X_t = (1-t)X_0 + tS'$（optimal transport path）。

**第二项（bootstrap term）**：
$$\gamma\mathbb{E}\left[\|v_t(X_t|S,A;\theta) - v_t(X_t|S',A';\bar{\theta})\|^2\right]$$

这targeting的是bootstrapped successor measure $m_\gamma^\pi(\cdot|S',A')$，使用target network $\bar{\theta}$稳定训练。

**为什么用flow matching而不是diffusion**：Farebrother et al. (2025)发现diffusion-based bootstrapping在long horizon有systemic bias，flow matching通过designing probability path structure能控制这个bias。

---

## 4. 本文核心贡献1：TD-HC (Temporal Difference Horizon Consistency)

### 4.1 动机

要support多个timescale（不同$\gamma$），naive做法是conditioning $v_t$ on $\gamma$。但variance随horizon增加，long-horizon预测容易崩。

### 4.2 Cross-Horizon Bellman Equation

公式(4)建立了两个discount factor $\beta \leq \gamma$之间的关系：

$$m_\gamma^\pi(\cdot|s,a) = (1-\gamma)P(\cdot|s,a) + \gamma\frac{1-\gamma}{1-\beta}\mathbb{E}_{S',A'}[m_\beta^\pi(\cdot|S',A')] + \gamma\frac{\gamma-\beta}{1-\beta}\mathbb{E}_{S^+,A^+}[m_\gamma^\pi(\cdot|S^+,A^+)]$$

**变量解释**：
- $\beta$：较短的discount factor
- $\gamma$：较长的discount factor
- 第一项$(1-\gamma)P$：immediate halt
- 第二项$\gamma\frac{1-\gamma}{1-\beta}$：以短horizon $\beta$从next state看，然后乘以scaling
- 第三项$\gamma\frac{\gamma-\beta}{1-\beta}$：以长horizon $\gamma$从短horizon的终点看

**直觉**：长horizon = 短horizon的扩展 + 在短horizonhalting点继续长horizon。这允许用短horizon的准确预测来"锚定"长horizon。

### 4.3 TD-HC Loss

公式(5)的TD-HC loss在TD-Flow基础上加了两个额外的flow matching term，分别targeting两个bootstrap source。关键实践细节：
- $\gamma$从$[\gamma_{min}, \gamma_{max}]$均匀采样
- $\beta$从$[\gamma_{min}, \gamma]$均匀采样
- 只对mini-batch的一小部分（antmaze 25%, cube 12.5%）应用consistency term
- 原因：consistency需要sample from模型自己的prediction，error会compound

---

## 5. 本文核心贡献2：Geometric Switching Policies (GSPs)

### 5.1 定义

GSP $\nu := \pi_{z_1} \xrightarrow{\alpha_1} \pi_{z_2} \cdots \xrightarrow{\alpha_{n-1}} \pi_{z_n}$

**切换机制**：执行$\pi_{z_i}$时，每步以概率$\alpha_i$切换到$\pi_{z_{i+1}}$，以概率$1-\alpha_i$继续。最后一个policy $\pi_{z_n}$是absorbing（$\alpha_n=0$）。

**Effective discount factor**：
$$\beta_k := \gamma(1-\alpha_k)$$

这是因为继续执行$\pi_{z_k}$一步需要：(1) episode不halt（概率$\gamma$），且(2)不切换policy（概率$1-\alpha_k$）。

### 5.2 Theorem 1: Successor Measure分解

GSP的successor measure是mixture：

$$m_\gamma^\nu(ds^+|s,a) = \sum_{k=1}^n w_k \int m_{\beta_1}^{\pi_{z_1}}(ds_1|s,a)\pi_{z_2}(da_1|s_1)\cdots m_{\beta_k}^{\pi_{z_k}}(ds^+|s_{k-1},a_{k-1})$$

**权重$w_k$**（Definition 1）：
$$w_k := \frac{1-\gamma}{1-\beta_k}\prod_{i=1}^{k-1}\frac{\gamma-\beta_i}{1-\beta_i}$$

**直觉**：$w_k$是"agent在前$k-1$个policy phase存活且未切换，在第$k$个phase下到达$s^+$"的概率。

### 5.3 Lemma 1: Unbiased Value Estimator

单样本Monte Carlo estimator：
$$\hat{Q}_\gamma^\nu := (1-\gamma)^{-1}\sum_{k=1}^n w_k r(S_k^+)$$

采样过程：从$(s,a)$开始，依次sample $S_k^+ \sim m_{\beta_k}^{\pi_{z_k}}(\cdot|S_{k-1}^+,A_{k-1}^+)$，然后weighted sum reward。

**这个结果的重要性**：它generalize了之前的GGPI（固定$\alpha$）和γ-models（固定policy），现在可以vary policy AND switching probability，closer to options framework。

---

## 6. Compositional Planning Procedure

### 6.1 Planning Objective

公式(6)：
$$\max_{a_1, z_1, \ldots, z_n} Q_\gamma^{\pi_{z_1}\xrightarrow{\alpha_1}\pi_{z_2}\ldots\xrightarrow{\alpha_{n-1}}\pi_{z_n}}(s, a_1)$$

**统一框架**：
- $\alpha_1=\cdots=\alpha_n=1$：ActionPlan（MPC with horizon $n$）
- $\alpha_1=1, \alpha_{2..n}=0$：GPI (Generalized Policy Improvement)
- $\alpha_1=\cdots=\alpha_{n-1}=\alpha$ fixed：GGPI
- 本文：$\alpha_i$可变 → CompPlan

### 6.2 Random Shooting Optimization

对于goal-conditioned policy（$Z=S$，$z$是subgoal），用GHM自己作为proposal distribution：

**Goal-Conditioned Proposal** (Algorithm 3)：
```
z_0 = s
for k=1,...,n:
    A_{k-1} ~ π_g(·|z_{k-1})
    z_k ~ m_{β_k}^{π_g}(·|z_{k-1}, A_{k-1})
```

**Unconditional Proposal** (Algorithm 4)：训练时10%的batch mask掉$z=\emptyset$，学习behavior policy的GHM，planning时直接sample。

完整CompPlan (Algorithm 2)：
1. Sample $m=256$或$1024$个candidate sequences
2. 对每个candidate，用Lemma 1 estimate $\hat{Q}$
3. 选$\arg\max$的sequence，执行第一个action + 第一个policy

---

## 7. 实验架构详解

### 7.1 GHM网络架构

- **Backbone**：U-Net style (Ronneberger et al., 2015)
- **Block dimensions**: (1024, 1024, 1024)
- **Time embedding**：sinusoidal → 2-layer MLP with mish activation
- **Discount embedding**：concatenate $[\gamma, 1-\gamma, -\log(1-\gamma)]$（最后一个对应log effective horizon，增强模型对$\gamma$的敏感度）
- **Conditioning**：FiLM modulation (Perez et al., 2018)
- **ODE solver**：Euler，train 10 steps, eval 20 steps

### 7.2 训练超参

| Hyperparameter | Value |
|---|---|
| Learning rate | $10^{-4}$ |
| Batch size | 256 |
| Gradient steps | 3M |
| Max discount $\gamma_{max}$ | 0.996 |
| Target EMA | $5\times10^{-4}$ |
| Context drop prob | 0.1 |
| Consistency proportion | 0.25 (antmaze) / 0.15 (cube) |

### 7.3 Base Policies

论文测试5种policy，每种代表不同trade-off：

1. **GC-TD3**：标准goal-conditioned offline RL with FQL policy extraction。challenge：action可能OOD，GHM建模困难。

2. **GC-1S**：GC-TD3的conservative变体，bootstrap用dataset action而非learned policy。occupancy更容易建模。

3. **CRL (Contrastive RL)**：用contrastive learning近似successor measure，$Q(s,a,g) \approx \phi(s,a)^\top\psi(g)$。factorization适合spatial navigation但不适合object manipulation。

4. **GC-BC**：纯imitation with flow matching + hindsight relabeling。struggle with distant goals。

5. **HFBC**：hierarchical flow BC，high-level预测$h$-step lookahead subgoal，low-level预测action。zero-shot最强。

### 7.4 Planning超参

| Domain | Candidates | Effective horizons | Proposal | Eval samples | Replan |
|---|---|---|---|---|---|
| Antmaze-* | 256 | [50,50,100,100,200] | Conditional | 256 | 1 |
| Cube-* | 1024 | [20,...,80] | Unconditional | 128 | 1 |
| HFBC Antmaze | 32 | [25]*24 | Conditional | 128 | 1 |
| HFBC Cube | 32 | [25]*{4,5,6,7} | Unconditional | 128 | 1 |

**关键设计**：antmaze用conditional proposal因为物理barrier限制reachability，cube用unconditional因为很多viable path。

---

## 8. 实验结果深度分析

### 8.1 主结果表 (Table 1)

| Domain | CRL Zero/Comp | GC-1S Zero/Comp | GC-BC Zero/Comp | GC-TD3 Zero/Comp | HFBC Zero/Comp |
|---|---|---|---|---|---|
| Antmaze-M | 0.88/0.97 | 0.56/0.87 | 0.49/0.85 | 0.65/0.65 | 0.94/0.94 |
| Antmaze-L | 0.84/0.90 | 0.21/0.61 | 0.18/0.73 | 0.23/0.48 | 0.78/0.92 |
| Antmaze-G | 0.16/0.29 | 0.00/0.02 | 0.00/0.03 | 0.00/0.01 | 0.42/0.79 |
| Cube-1 | 0.28/0.86 | 0.37/0.66 | 0.90/0.99 | 0.58/0.91 | 0.80/0.97 |
| Cube-2 | 0.02/0.50 | 0.10/0.57 | 0.15/0.97 | 0.12/0.82 | 0.76/0.77 |
| Cube-3 | 0.01/0.73 | 0.01/0.67 | 0.09/0.92 | 0.12/0.83 | 0.64/0.83 |
| Cube-4 | 0.00/0.39 | 0.01/0.60 | 0.00/0.76 | 0.00/0.57 | 0.24/0.67 |

**关键观察**：
- Zero-shot表现差不代表planning差：GC-BC在antmaze-L zero-shot只有0.18，CompPlan到0.73
- HFBC是最consistent的zero-shot base policy
- CRL在cube上zero-shot接近0，但CompPlan能extract utility到0.39-0.73

### 8.2 vs Hierarchical Baselines (Table 2)

| Domain | HIQL | SHARSA | HFBC | CompPlan |
|---|---|---|---|---|
| Antmaze-G | 0.65 | 0.56 | 0.42 | **0.79** |
| Cube-3 | 0.03 | 0.50 | 0.54 | **0.83** |
| Cube-4 | 0.00 | 0.09 | 0.34 | **0.67** |

CompPlan不需要task-specific training就超越SHARSA（SOTA on OGBench），特别是在最难的任务上margin巨大。这暗示test-time composition是learned hierarchy的有力补充。

### 8.3 vs ActionPlan / GPI (Figure 1)

平均over policies和long-horizon domains：
- CompPlan vs GPI：**89% relative improvement**
- CompPlan vs ActionPlan：**201% relative improvement**

这disentangle了两个factor：
- GPI只select policy不compose → improvement来自composition
- ActionPlan只plan over actions无temporal abstraction → improvement来自temporal abstraction
- CompPlan两者结合才最好

### 8.4 TD-HC效果 (Table 3, 9)

**Generative fidelity (EMD↓)** at $\gamma=0.995$：

| Domain | CRL td-flow / td-hc | GC-1S td-flow / td-hc |
|---|---|---|
| Antmaze-G | 6.77 / **5.74** | 7.29 / **5.25** |
| Cube-1 | 1.60 / **1.57** | 1.43 / **1.33** |

antmaze-giant上GC-1S的EMD降低28%！Figure 9显示td-flow在$\gamma=0.998$时样本会穿墙，td-hc尊重topological constraint。

**Planning performance**：td-hc vs td-flow差异很小（~5% relative）。原因：实际planning用$\beta_i \in [0.98, 0.99]$（50-100步），不是200+步的extreme horizon。td-hc的advantage在更长horizon才显现。

### 8.5 Ablation: Replan Frequency (Table 5)

| Domain | CRL 1-step / 5-step | GC-1S 1-step / 5-step |
|---|---|---|
| Antmaze-M | 0.97 / 0.94 | 0.87 / 0.83 |
| Cube-2 | 0.50 / 0.19 | 0.57 / 0.75 |
| Cube-4 | 0.39 / 0.00 | 0.60 / 0.56 |

平均replan every step比every 5 step好20%，但大多数domain差异小。Cube上CRL是个exception（diffuse policy导致5-step后trajectory偏离）。这是个speed-performance trade-off lever。

### 8.6 Ablation: Action Optimization (Table 6)

对比max over $(a_1, z_1, ..., z_n)$ vs only $(z_1, ..., z_n)$：

| Domain | CRL max-Q / max-V | GC-BC max-Q / max-V |
|---|---|---|
| Antmaze-M | 0.97 / 0.92 | 0.85 / 0.62 |
| Cube-2 | 0.50 / 0.13 | 0.97 / 0.92 |
| Cube-4 | 0.39 / 0.00 | 0.76 / 0.72 |

max-Q平均好70%，因为max-V依赖base policy sample action，diffuse policy（如CRL on cube）会fail。

### 8.7 Ablation: Proposal Distribution (Table 7)

| Domain | CRL Cond/Uncond | HFBC Cond/Uncond |
|---|---|---|
| Antmaze-M | 0.97 / 0.91 | 0.94 / 0.84 |
| Antmaze-G | 0.29 / 0.14 | 0.79 / 0.35 |
| Cube-4 | 0.36 / 0.39 | 0.09 / 0.67 |

Antmaze上conditional proposal略好（unconditional浪费sample在unreachable region）。Cube上unconditional更好（更多viable path）。HFBC在Cube-4上unconditional远好（0.67 vs 0.09），可能因为HFBC的conditional proposal过于集中在某些path。

---

## 9. 与相关工作的关系

### 9.1 Successor Representation谱系

- Dayan (1993): original successor representation
- Barreto et al. (2017): successor features for transfer
- Blier et al. (2021): successor measure (continuous state)
- Janner et al. (2020): γ-models (first generative model of successor measure)
- Thakoor et al. (2022): GHM + GGPI
- Farebrother et al. (2025): TD-Flows (flow matching for stability)
- **本文**: TD-HC + variable switching probs + comprehensive empirical validation

### 9.2 vs Diffusion Planners (Diffuser, etc.)

Diffusion planner (Janner et al., 2022)训练trajectory-level generative model，planning在denoising过程内，需要inverse dynamics提取action。

**本文优势**：
1. Policy-grounded：直接compose预训练policy行为
2. 不需要inverse dynamics（action直接从base policy sample）
3. 避免trajectory distribution依赖training data的limitation

### 9.3 vs Hierarchical RL (HIQL, SHARSA)

HIQL/SHARSA在training时learn high-level policy select subgoal。CompPlan在test time用GHM做planning，无需task-specific training。

**实验显示**：CompPlan在cube-4上67% vs SHARSA 9% vs HIQL 0%，说明test-time composition可以超越learned hierarchy当task需要flexible long-horizon reasoning。

### 9.4 vs Options Framework

GSP可以viewed as options with geometric termination condition。本文的generalization：
- 允许不同switching probability（options的termination function通常state-dependent，这里simplified为geometric）
- Lemma 1的estimator连接到options的multi-time model (Precup & Sutton, 1997)

---

## 10. 局限与未来方向

论文提到几个promising direction：
1. **State-dependent switching probability**：当前$\alpha_i$是hyperparameter，未来可学state-dependent $\alpha(s)$
2. **Joint learning of policies and GHMs**：当前先训policy再训GHM，joint training可能更优
3. **更sample-efficient MPC**：当前用random shooting，MCTS等更sophisticated search可能更好
4. **Latent space GHMs**：在learned latent space做jumpy prediction可能更efficient

**我的额外联想**：
- GHM本质上是model-based RL的"极限版"：不是predict one step然后rollout，而是直接predict occupancy。这绕开了model-based RL的classic compounding error problem。
- TD-HC的cross-horizon consistency让人联想到diffusion model中的consistency model (Song et al.)和shortcut models (Frans et al., 2025)——都是用short-time prediction anchor long-time prediction。
- 这个framework天然适合LLM-style的"chain of thought" planning：每个policy是一个"reasoning step"，GHM预测step的outcome，planning是search over reasoning chains。
- 在robotics foundation model context下（π0, GR00T, etc.），这提供了一种test-time scaling方法：不需要retrain foundation model，而是通过compositional planning解锁long-horizon capability。

参考链接：
- Consistency Models: https://arxiv.org/abs/2303.01469
- Shortcut Models: https://arxiv.org/abs/2410.12557
- π0: https://arxiv.org/abs/2410.24164
- GR00T N1: https://arxiv.org/abs/2503.14734
- SHARSA: https://arxiv.org/abs/2505.20552 (Park et al., Horizon Reduction Makes RL Scalable)

---

## 11. 核心intuition总结

1. **Successor measure把value function变成generative modeling问题**：你能sample $m_\gamma^\pi$就能evaluate任意reward下的$Q$。

2. **Flow matching解决bootstrapping stability**：TD需要bootstrap，但diffusion在long horizon有bias，flow matching通过designing probability path控制这个bias。

3. **Cross-horizon consistency anchor long prediction**：短horizon准，用短horizon的prediction作为long horizon的"checkpoint"，避免error compound。

4. **Geometric switching = temporal abstraction with clean math**：geometrically distributed duration让successor measure有closed-form decomposition（Theorem 1），使planning tractable。

5. **Policy composition > policy selection > action planning**：实验证明三者层次递进，composition unlock了long-horizon capability。

6. **Zero-shot performance ≠ Composition utility**：GC-BC zero-shot差但composition后强，说明policy的"composability"是orthogonal dimension到zero-shot capability。

7. **Test-time planning可超越learned hierarchy**：CompPlan无task-specific training却超越HIQL/SHARSA，暗示flexible test-time composition是promising alternative。

这篇paper在model-based RL、hierarchical RL、foundation model adaptation的交叉点上提出了一个elegant framework，把planning重新frame为generative model composition，empirical result非常strong，特别是on long-horizon tasks。
