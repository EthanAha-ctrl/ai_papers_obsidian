---
source_pdf: Diffusion Predictive Control with Constraints.pdf
paper_sha256: c82efb7630b73adb991d3768a9374def60d2a95b5dbce9ab05344fc701e9989e
processed_at: '2026-08-03T21:50:22-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DPCC 人话版

## 核心痛点：train-time 和 test-time 的 gap

想象你用 diffusion model 学了一个 robot manipulation policy，dataset 里有 96 条 demo，机器人学会了绕过 6 个 obstacle 到达目标。训练完看起来很美。

部署时问题来了：现场突然多了一个新 obstacle，或者 joint torque limit 变了，或者安全规范说 "今天这块区域不能进"。这些 constraints 在 training data 里**根本没出现过**。

Diffusion policy 没有任何机制响应这种新 constraint。它只会从 learned distribution 里 sample 一个 trajectory，希望它碰巧不撞墙。实测下 93% 概率撞。

MPC 能处理这种 constraint，但 MPC 需要 cost function，而 "怎么让机器人把杯子拿到桌上" 这种任务写 cost function 很痛苦。

**DPCC 的核心 insight**：把 MPC 的 projection 操作塞进 diffusion 的 denoising loop 里，每一步 denoise 都顺手把 trajectory 拉回 feasible set。这样你既有 diffusion 的 expressive power（multimodal, learn from demo），又有 MPC 的 constraint satisfaction guarantee。

## 直觉：denoising 就是不断"提建议 + 被审查"

Diffusion 的 backward process 本质上是 K=20 步的迭代：
1. 从纯噪声 $\tau^K \sim \mathcal{N}(0, I)$ 开始
2. 每一步，neural net 提议一个"cleaner"的 trajectory $\mu_\theta^k$
3. 加点随机 noise 保持 diversity
4. 得到下一步的 $\tau^{k-1}$

普通 diffusion policy 就这么 sample 一个 trajectory 然后执行。

DPCC 在每一步加了"审查官"：拿到 $\tau^{k-1}$ 后，用 optimization 把它投影到一个 set $\mathcal{Z}_f$ 里，这个 set 的定义是：
- 满足 state constraint（不进 obstacle）
- 满足 action constraint（速度不超限）
- 满足 dynamics（前后 state 之间满足物理方程）

投影完再继续下一步 denoise。20 步 denoise 走完，最后输出的 trajectory **必然**满足所有 constraint，因为它从最后一步投影出来。

## 为什么不是 projection 放最后一步就行？

一种 naive 方法（论文里叫 Post-Processing baseline）：让 diffusion 正常跑 20 步，最后一步拿到 trajectory 后再做一次大 projection。

这有问题：diffusion 最后输出的 trajectory 可能离 feasible set 很远（比如要绕到 obstacle 后面），一次大 projection 会把它"粗暴"地拉过来，结果 trajectory 变得很奇怪，机器人执行起来要绕远路，timesteps 从 69 涨到 84。

DPCC 的 iterative projection 像是"小步走"：每一步 denoise 后 trajectory 离 feasible set 不远（因为上一步刚 project 过），projection cost 小，整体 trajectory 更接近 learned distribution 的"自然形状"。

## 为什么 projection 里必须带 dynamics？

这是论文最关键的 insight。早期工作（Romer 2024 workshop paper）做了 model-free projection：只投 state/action constraint，不管 dynamics。

实验结果：constraint satisfaction 7%，跟不做 projection 一样烂。

直觉：假设 1D 系统 $s_{t+1} = s_t + a_t$，diffusion 给的 trajectory 是 $(s_0, a_0, s_1, a_1, ...)$。model-free projection 看到 $s_1$ 进了 obstacle，把 $s_1$ 改成 $s_1'$ 跑出来。但 $s_0, a_0$ 没动，于是 $s_1' \ne s_0 + a_0$，trajectory 物理上不可达。

下一步 denoise 时 neural net 看到一个 state-action sequence，它学的 distribution 是基于真实物理 demo 的，所以 mean $\mu_\theta$ 又把 $s_1$ 拉回 $s_0 + a_0$（dynamics-consistent），结果 $s_1$ 又跑到 obstacle 里。

trajectory 在 "constraint-satisfying" 和 "dynamics-feasible" 之间横跳，最终什么都没满足。

Model-based projection 同时优化 $s_0, a_0, s_1$，让它们同时满足 dynamics 和 constraint，所以改完的 trajectory 仍然 physically realizable，下一步 denoise 不会把它拉回去。

## Constraint tightening：给 model error 留 buffer

就算预测 trajectory 满足 constraint，实际执行时 state 还会越界，因为 dynamics model $f$ 不准（有 mismatch $w_t$）。

DPCC 的做法：预测时不用真实 constraint set $S_{t+1}$，用 tightened set $\tilde{S}_{t+1} = S_{t+1} \ominus \mathcal{B}_\gamma$，其中 $\mathcal{B}_\gamma$ 是半径 $\gamma$ 的 ball，$\ominus$ 是 Minkowski difference。

人话：把 constraint set "向内收缩 $\gamma$"。原本 allowed region 是 [0, 10]，tightened 变成 [0.025, 9.975]。预测 state 落在 [0.025, 9.975] 内，加上 model error $\le 0.025$ 的扰动，实际 state 落在 [0, 10]，刚好满足 constraint。

这个 trick 是 robust MPC 的经典方法 https://sites.engineering.ucsb.edu/~mdo/AM263B/AM263B_Focus.pdf ，DPCC 直接拿来用了。$\gamma$ 怎么定？跑 100 个无 constraint rollout，量 model 预测和实际 state 的误差，取上界。

## Trajectory selection：解决 multimodal 横跳

Diffusion 可以一次 sample 多个 trajectory（B=4）。普通做法随机选一个执行。但 multimodal distribution 下这有问题：timestep 1 选了 "绕左边" mode，timestep 2 选了 "绕右边" mode，机器人就在两个 mode 之间反复横跳，浪费时间，可能还违反 constraint。

DPCC 提两个 selection 策略：

**DPCC-T (Temporal Consistency)**：选和上一步预测 overlap 最大的 trajectory。直觉：上一 timestep 我打算往左走，这一 timestep 我也选打算往左走的 trajectory，保持一致性。

**DPCC-C (Cumulative Projection Cost)**：选 20 步 denoise 中累计被 projection 修改最少的 trajectory。直觉：projection cost 小说明这个 trajectory 本来就离 feasible set 近，更接近 learned distribution 的"自然形状"，更可能是好 trajectory。

实验结果：DPCC-C 最好，69 timesteps 到 goal，98% constraint satisfaction，0 violations。DPCC-T 也比 random selection 好。

## 实验讲的故事

Table 1 是核心结果：

- **Diffuser**（无 constraint 处理）：goal 总能到（100%），但 constraint 几乎总是违反（7% 同时满足 goal + constraint），平均 17.8 个 timestep 在 violation 状态
- **Guidance**（cost gradient 引导 denoising）：基本没用，constraint 满足率 9-13%。原因：cost function weight 难调，要么推得不够出 unfeasible region，要么推太远导致绕远路
- **Post-Processing**（只在最后一步 project）：tightening 后 96% 满足，但慢（79 timesteps）
- **Model-Free projection**（iterative project 但不带 dynamics）：完全失败，7%。证明 dynamics 在 projection 里是关键
- **DPCC-C**：98% 满足，69 timesteps，0 violation

Table 2 测 model robustness：故意把 dynamics model 的 sampling time $\hat{t}_s$ 调错 4 倍，constraint satisfaction 还有 77%。这说明 iterative model-based projection 对 model error 极其 robust——因为每步都"自查自纠"，不像一次性 projection 那样一旦 model 错就完蛋。

## 整体类比

把整个方法类比成一个"有审查官的画师"：

普通 diffusion policy：画师从 noise 开始一步步画清晰，画出一只猫。但画师不知道"猫不能有 5 条腿"这种 constraint。

DPCC：每画一笔，审查官检查"有没有违反 constraint，是不是 physically realizable"，违反就用 model 知识把画修一下，修完画师继续画。最终画作必然满足所有 constraint。

关键 subtlety：审查官必须懂物理（model-based），不懂物理的审查官会画出物理上不可达的 trajectory，画师下一步又会把它擦掉，来回拉扯。

## 跟你熟悉的概念的类比

如果你熟悉 classifier guidance（Dhariwal & Nichol 2021）https://arxiv.org/abs/2105.05233 ：DPCC 本质上是 classifier guidance，只不过 "classifier" 不是神经网络学的 $p(y|x)$，而是 geometric 的 distance-to-feasible-set。这个 distance 的 gradient 就是 projection direction，所以"加 gradient 到 mean"等价于"project mean 到 set"。

数学推导里 Theorem 1 用 Taylor expansion 把 log-likelihood 在 $\mu_\theta^k$ 处线性化，得到的 modified Gaussian mean 正好是 projection，这就是 classifier guidance 公式 $\mu + \sigma^2 \nabla \log p(y|x)$ 的几何化特例。

但 Theorem 1 给出的版本只是 approximate constraint satisfaction（Gaussian tail 会超出 set）。为了 hard guarantee，DPCC 在公式 (16) 做了个 trick：先 sample 再 project，每一步 sample 完立刻 project 到 $\mathcal{Z}_f$ 内，所以输出必然 feasible。代价是概率分布稍微偏离了理论推导的 (12) 式，但实践中影响很小。

## 局限

几个 caveat 论文没强调：

1. **Non-convex projection**：Theorem 1 假设 $\mathcal{Z}_f$ convex，但实际 state constraint 是 non-convex（比如 obstacle 是 outside-circle），所以用 SLSQP 解 non-convex QP，可能 local minimum。80ms 一个 action 也是 SLSQP 撑出来的，复杂环境可能崩。

2. **$\gamma$ 估计**：$\gamma=0.025$ 是 100 rollout 估的。如果 test 时 robot 载重变了，model error 变了，$\gamma$ 就 underestimate，constraint violation 会回来。

3. **Horizon 短**：H=8 steps 太短，长期 plan 不行。H 长了 non-convex projection 更难解。

4. **Time-varying constraint 没真测**：论文说 "can handle them directly"，但实验只测 static。Moving obstacle 需要 $\mathcal{Z}_f$ 每 step 重算，projection 求解时间可能成 bottleneck。

5. **Multimodal mode-switching**：DPCC-T 强制 temporal consistency，但有些 task 需要 mode switch（先绕左再绕右），这种场景 DPCC-T 可能 stuck。

## 一句话总结

DPCC = Diffusion policy 的 iterative denoising 里每步塞一个 model-based projection，让 trajectory 在生成过程中就被强制拉回 dynamic-feasible + constraint-satisfying set；用 constraint tightening 对抗 model error；用 trajectory selection 处理 multimodality。

Reference:
- DPCC paper (你提供的 PDF)
- Diffuser https://diffusion-planning.github.io/
- Classifier Guidance https://arxiv.org/abs/2105.05233
- Romer 2024 workshop https://probabilistic-robotics.github.io/
- Robust MPC textbook https://sites.engineering.ucsb.edu/~mdo/AM263B/AM263B_Focus.pdf
- Diffusion Policy https://diffusion-policy.cs.columbia.edu/
- Motion Planning Diffusion https://arxiv.org/abs/2307.09582
- Constrained Projected Diffusion https://arxiv.org/abs/2407.05813
- Diffusion MPC https://arxiv.org/abs/2410.05364

---

# Diffusion Predictive Control with Constraints (DPCC) 深度解析

## 1. 核心问题与动机

这篇论文要解决一个根本性的张力：**diffusion policies 在 train-time 学到了 rich multimodal behavior，但 inference-time 遇到 training data 里没出现过的 constraints 时，没有任何机制保证 hard constraint satisfaction**。而 MPC 能处理 constraints 但需要 cost function。DPCC 的核心 insight 是把 MPC 的 model-based projection 嵌入到 diffusion 的 backward denoising process 里，每一步 denoise 后强制把 trajectory 投影到一个 feasible set 上。

参考 Diffuser (Janner et al. 2022) https://diffusion-planning.github.io/ 是这条 line 的起源工作，DPCC 是它的 "constraint-aware" 变体。

## 2. Trajectory Diffusion 数学回顾

### 2.1 Forward process

$$q(\tau^k | \tau^{k-1}, k) = \mathcal{N}(\sqrt{1-\beta_k} \tau^{k-1}, \beta_k I)$$

- **$\tau^k \in \mathbb{R}^{(H+1)(|S|+|A|)}$**：第 k 步扩散的 trajectory（state + action 序列），$H$ 是预测 horizon，state 维度 $|S|$，action 维度 $|A|$
- **$\beta_k \in (0,1)$**：第 k 步 noise schedule 控制量，决定加多少噪声
- **$\sqrt{1-\beta_k} \tau^{k-1}$**：前一步状态乘以缩放因子，让 energy 慢慢衰减
- **$\beta_k I$**：新增 noise 的协方差矩阵，$I$ 是单位阵

Marginal form（用 Gaussian 的 closed-form 性质）：
$$q(\tau^k | \tau^0, k) = \mathcal{N}(\sqrt{\bar{\alpha}_k} \tau^0, (1-\bar{\alpha}_k) I)$$
- $\alpha_k = 1 - \beta_k$
- $\bar{\alpha}_k = \prod_{i=1}^k \alpha_i$ 累乘
- 当 k 趋向 K（最大），$\bar{\alpha}_K \to 0$，所以 $\tau^K \sim \mathcal{N}(0, I)$

### 2.2 Backward process 与 training loss

$$p_\theta(\tau^{k-1}|\tau^k, k) = \mathcal{N}(\mu_\theta(\tau^k, k), \sigma_k^2 I)$$
- $\mu_\theta$：神经网络学的 mean function
- $\sigma_k^2 = \beta_k \frac{1-\bar{\alpha}_{k-1}}{1-\bar{\alpha}_k}$：固定 variance schedule

训练用 noise-prediction surrogate：
$$\mathcal{L}(\theta) = \mathbb{E}_{k, \tau^0, \epsilon}[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_k}\tau^0 + \sqrt{1-\bar{\alpha}_k}\epsilon, k)\|_2]$$
- $\epsilon \sim \mathcal{N}(0, I)$：纯噪声
- $\sqrt{\bar{\alpha}_k}\tau^0 + \sqrt{1-\bar{\alpha}_k}\epsilon = \tau^k$：forward 加噪
- $\epsilon_\theta$：网络预测的 noise
- $\mu_\theta$ 和 $\epsilon_\theta$ 之间有解析关系：$\mu_\theta = \frac{1}{\sqrt{\alpha_k}}(\tau^k - \frac{\beta_k}{\sqrt{1-\bar{\alpha}_k}} \epsilon_\theta)$

## 3. DPCC 核心：Model-Based Projection in Denoising

### 3.1 Constraint set 定义

$$\mathcal{Z}_f = \{\tau = (s_{t:t+H}, a_{t:t+H}) \mid \tau \in \mathcal{Z}, s_{t+1} = f(s_t, a_t), \forall t\}$$

直觉上：$\mathcal{Z}$ 是只考虑 state/action box constraint 的集合，$\mathcal{Z}_f$ 进一步要求 trajectory 满足 dynamics。这个 $f$ 的 inclusion 是关键，因为 model-free projection 会让轨迹 break dynamics feasibility。

### 3.2 Projection operator

$$\Pi_{\mathcal{Z}_f}(\tau) = \arg\min_{\tilde\tau \in \mathcal{Z}_f} \|\tau - \tilde\tau\|_2^2$$

$$\text{s.t. } s_{t'+1|t} = f(s_{t'|t}, a_{t'|t}), \forall t' \in \mathbb{I}_t^H$$

这是一个 constrained QP（如果 dynamics 线性、constraints 凸的话），实际用 SLSQP 解 non-convex 问题。$\mathbb{I}_t^H = \{t, t+1, ..., t+H\}$。

### 3.3 Theorem 1 的推导（这是论文最核心的部分）

**Control as inference framing**: 引入 binary 变量 $\mathcal{O} \in \{0,1\}$ 表示 trajectory feasibility。
$$p_\theta(\tau | \mathcal{O}=1) \propto p_\theta(\tau) p(\mathcal{O}=1|\tau)$$

如果定义 likelihood 为 Gaussian distance：
$$p(\mathcal{O}=1|\tau, k) \propto \exp\left(-\frac{1}{2\sigma_k^2} d(\tau, \mathcal{Z}_f)^2\right)$$
- $d(\tau, \mathcal{Z}_f) = \min_{\tilde\tau \in \mathcal{Z}_f} \|\tilde\tau - \tau\|_2$：点到 set 的距离
- $\sigma_k$：和 backward process 同 step 的 variance，让 likelihood 在不同 diffusion step 有不同 strength

**Step 1: Markovian decomposition**
$$p_\theta(\tau^{k-1}|\tau^k, \mathcal{O}, k) \propto p_\theta(\tau^{k-1}|\tau^k, k) p(\mathcal{O}|\tau^{k-1}, k)$$

**Step 2: First-order Taylor expansion of log-likelihood**
$$\log p(\mathcal{O}|\tau^{k-1}, k) \approx \log p(\mathcal{O}|\mu_\theta^k, k) + (\tau^{k-1} - \mu_\theta^k)^\top v(\mathcal{O})$$
- $v(\mathcal{O}) = \nabla_\tau \log p(\mathcal{O}|\tau, k)|_{\tau = \mu_\theta^k}$：log-likelihood 在 mean 处的 gradient

**Step 3: Gaussian algebra**（套用 classifier guidance 技巧，参考 Dhariwal & Nichol 2021 https://arxiv.org/abs/2105.05233）
Gaussian $\times$ exponential linear term 仍然是 Gaussian，新的 mean 是 $\mu_\theta^k + \sigma_k^2 v(\mathcal{O})$：
$$p_\theta(\tau^{k-1}|\tau^k, k, \mathcal{O}) \approx \mathcal{N}(\mu_\theta^k + \sigma_k^2 v(\mathcal{O}), \sigma_k^2 I)$$

**Step 4: 用 projection 表示 v(O=1)**

对于 closed convex set $\mathcal{Z}_f$，距离函数的 gradient 指向 projection 方向：
$$v(\mathcal{O}=1) = -\frac{1}{\sigma_k^2} d(\mu_\theta^k, \mathcal{Z}_f) \nabla_\tau d(\tau, \mathcal{Z}_f)|_{\tau=\mu_\theta^k} = \frac{1}{\sigma_k^2}(z - \mu_\theta^k)$$
- $z = \Pi_{\mathcal{Z}_f}(\mu_\theta^k)$：在 set 上的唯一投影点
- $-(\mu_\theta^k - z)$ = $z - \mu_\theta^k$：从 mean 指向 projection 的方向向量，长度等于 distance

**Step 5: 代入得到**
$$\mu_\theta^k + \sigma_k^2 v(\mathcal{O}) = \mu_\theta^k + (z - \mu_\theta^k) = z = \Pi_{\mathcal{Z}_f}(\mu_\theta^k)$$

所以 modified denoising 是 $\mathcal{N}(\Pi_{\mathcal{Z}_f}(\mu_\theta^k), \sigma_k^2 I)$。

**但是这个版本只 approximately sample from $p(\tau | \mathcal{O}=1)$**，因为：
1. $p(\mathcal{O}=1|\tau) > 0$ 即使 $\tau \notin \mathcal{Z}_f$（Gaussian tail）
2. Sampling $\tau^{k-1} \sim \mathcal{N}(z, \sigma_k^2 I)$ 后 $\tau^{k-1}$ 不一定在 $\mathcal{Z}_f$ 内

### 3.4 Modified step (16) - 严格保证

$$\tau^{k-1} = \Pi_{\mathcal{Z}_f}(\mu_\theta(\tau^k, k, c) + \sigma_k \epsilon_k), \quad \epsilon_k \sim \mathcal{N}(0, I)$$

**关键改动**：先 sample (add noise)，再 project。因为 projection 总是返回 $\mathcal{Z}_f$ 内的点，所以每一步（包括最后 $\tau^0$）都严格满足 constraints。

**这个 swap 的代价**：不再严格对应 (12) 的概率分布。但在实践中，因为 projection 是 "deterministic shift" 加在 sample 上，扰动的是"shape"而非 location，效果接近。

直觉理解：每一步 denoising 都像是从 learned distribution 采样到一个 "noisy trajectory"，然后 MCP-style 地把它拉回 feasible set。这个 "拉回" 用了 model $f$，所以拉回后的 trajectory 依然 dynamically feasible。

## 4. Constraint Tightening (Section 5.3, Theorem 2)

### 4.1 问题

即使预测的 $\tau_{t:t+H|t} \in \mathcal{Z}_f$，由于 model mismatch $w_t$，actual state $s_{t+1} = f(s_t, a_t) + w_t$ 可能违反 constraint。

### 4.2 Minkowski difference 的直觉

定义 tightened set:
$$\tilde{S}_{t+1} = S_{t+1} \ominus \mathcal{B}_\gamma$$
- $\mathcal{B}_\gamma = \{x : \|x\|_2 \leq \gamma\}$：半径 $\gamma$ 的 $\ell_2$ ball
- $\ominus$：Minkowski set difference，$A \ominus B = \{x : x + B \subseteq A\}$

直觉：$S_{t+1} \ominus \mathcal{B}_\gamma$ 是 "$S_{t+1}$ 内那些点 $x$，使得 $x$ 周围 $\gamma$ ball 仍然全在 $S_{t+1}$ 内"。换言之，从这些点出发，扰动 $\gamma$ 也不会越界。

### 4.3 Theorem 2 证明思路

- 预测时 $s_{t+1|t} \in \tilde{S}_{t+1} = S_{t+1} \ominus \mathcal{B}_\gamma$
- 实际 $s_{t+1} = s_{t+1|t} + w_t$, $\|w_t\|_2 \leq \gamma$
- 所以 $s_{t+1} \in \tilde{S}_{t+1} \oplus \mathcal{B}_\gamma = (S_{t+1} \ominus \mathcal{B}_\gamma) \oplus \mathcal{B}_\gamma \subseteq S_{t+1}$
- $\oplus$ 是 Minkowski sum，$(A \ominus B) \oplus B \subseteq A$ 是基本 set 性质
- Induction on $t$

**这里隐含 assumption**：$\gamma$ 必须是 $w_t$ 的 tight upper bound。论文里 $\gamma = 0.025$ 通过 100 rollouts without constraints 估计出来。如果 $\gamma$ 估小了，会有 violation；估大了，过度保守导致 infeasibility。

## 5. Trajectory Selection (Section 5.4)

由于 diffusion 在 80ms 内能并行 sample B=4 个 trajectories，如何挑一个？

**DPCC-T (Temporal Consistency)**:
$$i(t) = \arg\min_j \|\tau_{t:t+H-1|t}^{0,j} - \tau_{t:t+H-1|t-1}^{0,i(t-1)}\|_2$$

直觉：multimodal distribution 下，如果不同 timestep 随机选不同 mode，会导致 end-effector "左右横跳"，浪费时间且可能违反 constraints。选和上次预测 overlap 最近的 trajectory，保证 closed-loop 行为 smooth。

**DPCC-C (Cumulative Projection Cost)**:
$$i(t) = \arg\min_j \sum_{k=1}^K c_{\mathcal{Z}_f}(\tilde{\tau}_{t:t+H|t}^{k-1,j})$$
- $c_{\mathcal{Z}_f}(\tilde\tau) = \|\tilde\tau - \Pi_{\mathcal{Z}_f}(\tilde\tau)\|_2^2$：每步 projection 的 cost
- 累加所有 K 步 projection cost，选最小的

直觉：projection cost 小意味着这个 trajectory 离 learned distribution 近，被 model-based correction 修改得少，更"自然"。

## 6. 实验数据深入分析

### 6.1 Avoiding Environment

- 6 个 obstacle，24 种 path → 96 demonstrations，高度 multimodal
- State 4D (current + desired position in 2D)
- Action 2D (Cartesian velocity command)
- Horizon $H+1 = 8$，diffusion steps $K = 20$
- Batch $B = 4$
- 80ms per action
- 用 1D U-Net backbone (Diffuser 架构)
- 用 cosine noise schedule (Improved DDPM https://arxiv.org/abs/2102.09672)

### 6.2 Table 1 关键解读

| Method | Constraint Tightening | Timesteps | Constraints & Goal | # Violations |
|---|---|---|---|---|
| Diffuser (no constraint) | - | 76.7 | 0.07 | 17.8 |
| Guidance (yes) | yes | 75.6 | 0.13 | 17.4 |
| Post-Processing (yes) | yes | 79.1 | 0.96 | 0.1 |
| Model-Free (yes) | yes | 76.1 | 0.07 | 18.0 |
| **DPCC-C** | **yes** | **69.0** | **0.98** | **0.0** |

**关键观察**：
1. **Diffuser 完全 fail** constraint satisfaction，只 7% success，说明 offline training 不能 generalize 到新 constraints
2. **Guidance 表现差**：因为 cost gradient 要么推得不够（boundary 没出），要么推得太远（推到 far-from-goal 的 safe region）。Tuning guidance weight 很难。
3. **Model-Free projection 几乎无效**：因为没有 dynamics 在 projection 里，每步 projection 把 trajectory 推到 $\mathcal{Z}$，但下一步 denoising 又拉回 model's mean，导致 trajectory "fight" 在 boundary 附近，最后还是违反。
4. **Post-Processing 好**（96%），但 slow (84.5 timesteps without tightening)：因为只在最后做一次大 projection，扰动 trajectory 多
5. **DPCC-C 最好**：98% success，69 timesteps，0 violations。Iterative projection 既能保证 constraint，又能 keep trajectory 接近 learned distribution

### 6.3 Table 2: Model robustness

| $\hat{t}_s / t_s$ | Timesteps | Constraints & Goal |
|---|---|---|
| 0.25 | 85.7 | 0.86 |
| 1.0 | 69.0 | 0.98 |
| 4.0 | 152.0 | 0.77 |

**Insight**: 即使 dynamics model 的 sampling time 错 4 倍（也就是 dynamics model 完全 wrong scale），依然 77% constraint success。这是因为 constraint tightening $\gamma = 0.025$ 给了 buffer，且 iterative projection 的 "self-correction" 效应比一次性 projection 鲁棒得多。这是 model-based projection 比 model-free 的核心优势。

## 7. 与相关工作的深层对比

### 7.1 vs. Diffuser (Janner 2022) https://diffusion-planning.github.io/
- Diffuser 用 return-conditioning 学 task
- DPCC 用 inpainting on current state
- Diffuser 完全 ignore constraints；DPCC 处理 novel constraints

### 7.2 vs. Classifier Guidance (Dhariwal & Nichol 2021) https://arxiv.org/abs/2105.05233
- Classifier guidance 加 $\nabla \log p(y|x)$ 到 score function
- DPCC 把 "feasibility likelihood" 当 classifier，但 gradient 是 projection direction
- 数学结构几乎一样，但 DPCC 的 "classifier" 是 geometric distance，不需要训练网络

### 7.3 vs. Motion Planning Diffusion (Carvalho 2023) https://arxiv.org/abs/2307.09582
- Carvalho 用 cost function gradient 引导 denoising
- 这本质上是 classifier guidance 用手设 cost
- 问题：cost gradient 在 boundary 处可能 ill-defined（如果 cost 是 indicator），tuning weight 麻烦
- DPCC 用 projection：geometric, deterministic, no tuning

### 7.4 vs. Safe Offline RL with Diffusion (Romer 2024 workshop) https://probabilistic-robotics.github.io/
- Romer 之前的工作用 model-free projection
- DPCC 关键改进：把 dynamics $f$ 加进 projection，保持 dynamic feasibility

### 7.5 vs. Constrained Synthesis with Projected Diffusion (Christopher 2024 NeurIPS) https://arxiv.org/abs/2407.05813
- Christopher 也做 projection during denoising
- 但没有 dynamics, 没有 constraint tightening
- 计算太重，不适合 sequential decision-making

### 7.6 vs. Diffusion MPC (Zhou 2024) https://arxiv.org/abs/2410.05364
- Zhou 用 diffusion 做 high-level planner，MPC 做 low-level tracking
- 两阶段；DPCC 是 single-stage，projection 直接 in denoising

## 8. 工程实现细节

### 8.1 U-Net 架构
1D U-Net over time axis（trajectory length H+1），输入 $\tau^k \in \mathbb{R}^{(H+1) \times (|S|+|A|)}$，channel = state + action dim = 6，treat trajectory length as spatial dim。这是 Diffuser 的标准做法。

### 8.2 Inpainting for conditioning on $s_t$
当前 state $s_t$ 固定：在每步 denoising 时把 $\tau^k$ 的第一个 state token 用 $s_t$ 的 noisy version 替换，确保 sample 的 trajectory 起点是当前 state。这比 classifier-free guidance 简单，且对 plan 一致性强。

### 8.3 Normalization
轨迹 normalize 到 [-1, 1] via limit normalization：
$$s_{n,i} = 2 \frac{s_i - \underline{s}_i}{\bar{s}_i - \underline{s}_i} - 1$$
- $\bar{s}_i, \underline{s}_i$：training data 中 state dim $i$ 的 max/min
- Constraint set 也 normalize 一致，避免在 denoising 时反复 un-normalize / re-normalize

### 8.4 SLSQP solver
Scipy 的 SLSQP (Sequential Least Squares Programming) https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html 解 non-convex QP。论文说 80ms per action，B=4 个 trajectory 并行（实际可能 serial），K=20 步 denoising × 4 trajectory × projection solve = 主要 bottleneck。

## 9. Limitations 与潜在扩展

### 9.1 Convexity assumption
Theorem 1 假设 $\mathcal{Z}_f$ convex。实际 experiments 里 state constraint $S_t = \{s : As \le b, \|s - p\|^2 \ge r^2\}$ 是 non-convex（圆是 $\|s-p\|^2 \ge r^2$ 是 outside-of-circle，non-convex）。所以 Theorem 1 只是 motivation，实际用 SLSQP 解 non-convex QP。

**问题**：non-convex 投影可能不 unique，可能 local optimum。这是论文没仔细讨论的潜在 failure mode。

### 9.2 γ 估计
γ = 0.025 是基于 100 个无 constraint rollout 估计的。如果 test-time 出现新 dynamics（比如载重变化），γ 估计就 wrong。**Potential 改进**：online γ estimation，或 robust MPC 的 tube-based approach。

### 9.3 Multimodal handling
DPCC-T 用 temporal consistency 应对 multimodality。但如果 task 本身需要 mode-switching（比如先绕左再绕右），DPCC-T 可能 stuck in 一个 mode。**Potential 改进**：用更复杂的 mode-tracking，比如 latent variable model on top of diffusion。

### 9.4 Time-varying constraints
论文说 "DPCC can handle them directly without modifications"，但实验只测了 static constraints。Time-varying 意味着 $\mathcal{Z}_f$ 每 timestep 变，需要 online 重新求解 QP。对 fast-changing obstacles 可能太慢（80ms）。

### 9.5 Long horizon
H=8 太短。Long horizon 下，non-convex projection 的 local minima 问题更严重。可以参考 MPC 的 multiple-shooting 或 collocation 方法。

## 10. 与其他 generative model-based control 的更大 picture

### 10.1 Flow Matching + Constraints
Flow matching (Lipman 2023 https://arxiv.org/abs/2210.02747) 是 diffusion 的表亲，更灵活。DPCC 的 projection 思路可以直接套到 flow matching：在 ODE integration 的每步 RK4 step 后做 projection。这其实更简单因为 flow matching 的每 step 是 deterministic。

### 10.2 Consistency Models + Constraints
Consistency models (Song 2023 https://arxiv.org/abs/2303.01469) 一步生成 sample，没 iterative denoising。直接套 DPCC 不行。需要 post-processing only。这是 DPCC-style 方法 的 limitation。

### 10.3 Score-based Diffusion + Lagrangian
可以用 augmented Lagrangian 把 constraints 加进 score function：
$$\nabla \log p(\tau) + \lambda \nabla c(\tau)$$
这是 guidance 的本质。但 tuning λ 难，projection 是 adaptive 的 "λ → ∞ at boundary"。

## 11. 一个 toy 推演：为什么 model-free fail 而 model-based work

假设 1D 系统 $s_{t+1} = s_t + a_t$，learned distribution 在 $\tau = (s_0, a_0, s_1, a_1, ...)$ 上有 modes。

**Model-free projection**: $\Pi_{\mathcal{Z}}$ 把 $\tau$ 投到 $s_t \in [l, u]$ for all t。但 projection 改了 $s_1$ 不改 $s_0, a_0$，于是 $s_1 \ne s_0 + a_0$，破坏 dynamics。下一步 denoising 用 learned mean $\mu_\theta$，它返回 dynamics-consistent trajectory，又把 $s_1$ 拉回 $s_0 + a_0$，于是又违反 constraint。 oscillation。

**Model-based projection**: $\Pi_{\mathcal{Z}_f}$ 投影时要求 $s_1 = s_0 + a_0$，所以会改 $a_0$ 或 $s_0$（或两者）来同时满足 dynamics 和 constraint。修改后的 trajectory 依然 dynamically feasible，下一步 denoising 不会把它拉回去。

**这是 DPCC 比早期 projection-based 方法 (Romer 2024) 的核心改进**。

## 12. 关于 future work 的具体推测

1. **Stability guarantees**: 加 control Lyapunov function 作为额外 constraint，可以在 projection 时 enforce $\dot V \le 0$。
2. **Chance constraints**: 把 constraint tightening 从 worst-case ($\ominus \mathcal{B}_\gamma$) 改为 probabilistic ($\ominus \mathcal{B}_{\gamma, \alpha}$)，用 $\alpha$-quantile of disturbance。
3. **Differentiable projection layer**: 用 cvxpylayers https://github.com/cvxgrp/cvxpylayers 让 projection 可微，可以 joint train diffusion + projection。
4. **Adaptive $\gamma$**: 基于 online disturbance estimate，用 recursive Bayesian update $\gamma_t$。
5. **Hierarchical DPCC**: 高层 diffusion + 低层 MPC tracking，类似 Zhou 2024 但保持 projection in denoising。
6. **Real robot**: 80ms 在 Franka 上够用（1kHz control），但 KUKA iiwa 之类 slower 控制器需要优化。

## 13. 总结直觉

DPCC 把 MPC 的 "约束投影 + 模型纠错" 嵌入到 diffusion 的 iterative denoising 里。每一步 denoise 都是一次"在 learned distribution 附近找一个 feasible trajectory" 的 mini-MPC。通过 control as inference 的 lens，这个 embedding 在数学上等价于用 distance-to-feasible-set 当 classifier guidance。Modified step (16) 把 sample-then-project 顺序调换，保证 hard constraint satisfaction。Constraint tightening 用 Minkowski difference 提前 shrink feasible set，对抗 model mismatch。Trajectory selection 利用 batch sampling 的 diversity，挑 temporal consistent 或 projection-cheap 的 trajectory，避免 multimodal distribution 的横跳问题。

整体上，DPCC 是 diffusion policy + MPC 的 hybrid，但 hybrid 的位置很巧妙：在 generative model 的 iterative process 里嵌入 optimization，而不是把 generative 和 optimization 当 two-stage pipeline。这个思路非常 general，可以推广到任何 iterative generative model（flow matching, SDE, etc.）。

主要参考：
- DPCC paper 本身（你提供的 PDF）
- Diffuser https://diffusion-planning.github.io/
- Improved DDPM https://arxiv.org/abs/2102.09672
- Classifier Guidance https://arxiv.org/abs/2105.05233
- Control as Inference (Toussaint 2009) http://www.ipvs.uni-stuttgart.de/abteilungen/abt-AS/publications/Toussaint09-ICML.pdf
- Diffusion Policy (Chi 2023) https://diffusion-policy.cs.columbia.edu/
- Constrained Projected Diffusion (Christopher 2024 NeurIPS) https://arxiv.org/abs/2407.05813
- Safe Offline RL with Trajectory Diffusion (Romer 2024) https://probabilistic-robotics.github.io/
- Diffusion MPC (Zhou 2024) https://arxiv.org/abs/2410.05364
- Motion Planning Diffusion (Carvalho 2023) https://arxiv.org/abs/2307.09582
