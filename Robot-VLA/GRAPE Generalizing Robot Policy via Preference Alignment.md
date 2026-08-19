---
source_pdf: GRAPE Generalizing Robot Policy via Preference Alignment.pdf
paper_sha256: f0e18325495322fed41f66313aef5d7865210631e45a39cfe7a3c0f3ba40ade2
processed_at: '2026-08-19T09:54:10-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GRAPE: 用人话讲讲这篇paper的核心Intuition

Andrej, 咱们用最直白的话来拆解 GRAPE 这篇paper。 

目前 VLA (Vision-Language-Action) models 比如 OpenVLA 或者 Octo，基本都在做 "behavior cloning"。这就像学生抄学霸的作业，只抄对了答案，却不知道学霸为什么这么写。如果考卷题目稍微变一变（比如换了个没见过的杯子，或者桌上多了点杂物），模型就傻眼了。

为什么不用 RL (比如 PPO) 呢？因为在 Robotics 里跑 online RL 太贵了，而且设计一个完美的 reward function 极其困难，PPO 在实机上也特别容易 collapse。

GRAPE 的核心 intuition 就是：**把 LLM 里的 DPO (Direct Preference Optimization) 偷偷搬到了 Robotics 的 Trajectory 上，并且用大模型自动当裁判，给机器人的轨迹打分。**

---

## 1. 为什么是 Trajectory-level 而不是 Step-level?

在 LLM 里，DPO 是在 token level 或者 response level 做的。如果直接生搬硬套到 Robotics，就是在 step level 做 DPO：比较在同一个 state $o_t$ 下，action $a_t$ 哪个好哪个差。

但 Robotics 有个致命问题：**Reward 极度 Sparse**。一条轨迹跑完，你只知道最后成没成功。如果做 step-level DPO，你把整条成功的轨迹里的每一步都标为 "chosen"，把失败的标为 "rejected"，这会引入巨大的 noise。因为成功轨迹里可能有很多冗余甚至次优的 step，而失败轨迹里也可能有部分完美的 step，只是最后一步搞砸了。

GRAPE 提出了 TPO (Trajectory-wise Preference Optimization)。它比较的颗粒度是**整条轨迹 $\zeta$**。它告诉模型："这条整体路径比那条整体路径好，去增加好路径的 likelihood，压制坏路径"。

### 公式拆解

看这个 TPO loss:

$$ \mathcal{L}_{\mathrm{TPO}} = -\mathbb{E}_{(\zeta_w, \zeta_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta \left(\log \frac{\pi_\theta(\zeta_w)}{\pi_{\mathrm{ref}}(\zeta_w)} - \log \frac{\pi_\theta(\zeta_l)}{\pi_{\mathrm{ref}}(\zeta_l)}\right)\right)\right] $$

- $\zeta_w$: 赢的轨迹 (chosen)
- $\zeta_l$: 输的轨迹 (rejected)
- $\pi_\theta$: 你正在训练的 VLA policy
- $\pi_{\mathrm{ref}}$: SFT 刚训完的 reference policy，用来做 KL constraint 防止跑偏
- $\beta$: 控制偏离 $\pi_{\mathrm{ref}}$ 程度的温度参数
- $\sigma$: sigmoid 函数

这里最聪明的一步是 MDP Decomposition。一条轨迹的 likelihood 可以拆解成单步 likelihood 的乘积：

$$ \log \frac{\pi_\theta(\zeta, q)}{\pi_{\mathrm{ref}}(\zeta, q)} = \sum_{t=1}^{T} \log \frac{\pi_\theta(a_t | o_t, q)}{\pi_{\mathrm{ref}}(a_t | o_t, q)} $$

**Intuition**: 你依然可以在单步 $(o_t, a_t)$ 上算 backprop，但你的 optimization 信号来自全局的 trajectory-level 比较。这把 sparse reward 的 credit assignment 问题解决得非常优雅：gradient 流过所有 step，但每一步收到的信号是 "你属于一条整体更好的路径"。

---

## 2. 自动当裁判: GCPG (Guided-Cost Preference Generation)

DPO 需要大量成对的 (chosen, rejected) data。在 Robotics 里请人标注 "哪条轨迹更好" 极其昂贵且主观。GRAPE 搞了一套全自动 Pipeline。

它的做法极其巧妙：
1. **任务分解**: 用 VLM (Hamster) 把复杂任务拆成几个 stages。比如 "拿葡萄放碗里" 拆成：1. 抓葡萄 2. 移动到碗上 3. 放下。
2. **提取 Keypoints**: 用 DINOv2 和 Grounded-SAM 在图像里找出关键点（葡萄中心、碗中心、障碍物等）。
3. **生成 Cost Function**: 用 GPT-4o 针对每个 stage 写一段 Python 代码来算 cost！比如 "计算末端执行器到葡萄中心的欧氏距离"，或者 "如果末端靠近障碍物，cost 增大"。

### Exponential Decay Aggregation Intuition

整条轨迹的 External Reward 这么算：

$$ R_{\mathrm{ext}}(\zeta) = \prod_{i=1}^{S} e^{-C^{S_i}(\{\kappa_{S_i}\})} $$

- $S$: 总 stage 数
- $C^{S_i}$: 第 $i$ 个 stage 的 cost
- $\kappa_{S_i}$: 第 $i$ 个 stage 的 keypoints

为什么用乘积（相当于把 $e$ 的指数相加），而不用线性求和？
因为 Robotics 有严格的 **Causal Dependency**。如果第一步抓取就失败了（cost 极大），后面移动得再完美也没用。乘积形式会让前期的高 cost 产生毁灭性的惩罚，这完美契合物理世界的连续操作逻辑。

### 三合一的 Reward 组合

最终给一条轨迹打分，用了三个东西的加权：

$$ R_{\mathrm{GCPG}}(\zeta) = \lambda_1 R_{\mathrm{self}}(\zeta) + \lambda_2 R_{\mathrm{ext}}(\zeta) + \lambda_3 I_{\mathrm{success}}(\zeta) $$

- $R_{\mathrm{self}}$: Policy 自己生成这条 trajectory 的 log-likelihood。高 confidence 的轨迹更稳定，避免采样到 OOD 的 outlier 轨迹。
- $R_{\mathrm{ext}}$: 上面说的基于 Cost Function 算出来的物理约束分。
- $I_{\mathrm{success}}$: 0 或 1，任务成没成功的稀疏 signal。
- $\lambda$: 权重参数，实验里设为 $\lambda_1=0.01, \lambda_2=0.01, \lambda_3=2$。可以看出任务成功是绝对的主导信号，其余两个是 dense reward 起到 calibration 作用。

---

## 3. 实验数据表解析: Flexible Alignment 是个亮点

GRAPE 最 make sense 的一个实验是它能根据你修改的 Cost Function，改变机器人的 "性格"。

**Table 2: 不同目标对齐的结果**

| Method | Real-World CR↓ | Real-World SL↓ | Real-World SR↑ |
| :--- | :--- | :--- | :--- |
| OpenVLA-SFT | 53.33 | 142.32 | 34.61 |
| GRAPE-Safety | **29.84** | 146.11 | 54.31 |
| GRAPE-TC (Task Completion) | 58.45 | 125.79 | 51.67 |
| GRAPE-Efficiency | 59.50 | **70.24** | 42.50 |

*(注: CR=Collision Rate 碰撞率, SL=Step Length 步长, SR=Success Rate 成功率)*

看这组数据：
- 当你在 Cost Function 里加大对碰撞的惩罚，模型碰撞率从 53.33 暴跌到 29.84，并且成功率还涨了！
- 当你让模型以效率（步长最短）为目标，Step Length 从 142 直接砍半到 70.24，代价是成功率稍微掉了一点。
- 默认的 Task Completion (GRAPE-TC) 则在效率和成功率之间取平衡。

这意味着，你只需要改一下 GPT-4o 生成 Cost Function 的 prompt，同一套 Pipeline 就能训出性格迥异的机器人，这在实际部署中太有用了。

---

## 4. 避坑指南: 为什么 Iterative Optimization 很重要

GRAPE 用了类似 On-policy RLHF 的迭代训练：每次用当前 policy 去采样新的 trajectories -> 用大模型裁判打分 -> 组成 preference pairs -> 用 TPO 训练 -> 更新 policy。

看 Figure 6 和 Table 3 的 Iteration 数据：

| Iter | In-domain Success | Average Success |
| :--- | :--- | :--- |
| Iter-1 | 43.00% | 43.11% |
| Iter-2 | 45.00% | 44.77% |
| Iter-3 | 45.50% | 44.92% |

Intuition: 最初的 preference pairs 是由 SFT model 采样的。SFT model 很弱，它采样出来的轨迹质量很差，导致 Chosen 和 Rejected 的差距不明显。随着迭代，Policy 变强，采样出来的轨迹质量整体变好，大模型裁判给的 signal 更精准，形成了一个正向飞轮。到第 3 轮基本收敛。

---

## 5. 核心联想与 Intuition 总结

GRAPE 这篇工作，本质上是在做 **Guided Inverse Optimal Control (Guided IRL) + Trajectory DPO**。

以前做 IRL (比如 Guided Cost Learning, Finn et al., 2016)，需要极复杂的 Max Entropy IRL 框架去拟合 Cost。现在 GRAPE 直接甩给 GPT-4o 一张图和 prompt，GPT-4o 就把 Cost Function 给写出来了。虽然这个 Cost Function 是非参数化的 Python 代码，但它足够鲁棒且可解释。

**它解决了 VLA 的三大痛点**:
1. **泛化性差**: TPO 让模型不仅学成功，还显式学习失败，相当于给模型加了一个 "底线逻辑"。
2. **Reward 难定义**: 用 LLM/VLM 组合拳直接写 Cost Function，避开了传统强化学习硬核的 reward shaping。
3. **Multi-objective 冲突**: 专家数据里隐含的安全、效率等信号互相打架。现在你可以显式声明 "我要训一个注重安全的模型" 或者 "我要训一个注重效率的模型"。

**值得注意的 Potential Issue (Hallucination/Intuition 联想)**:
GRAPE 极度依赖 VLM 提取 Keypoints 和 LLM 写 Cost Function 的准确性。如果光照极差，或者物体被严重遮挡，DINOv2/SAM 提取的 keypoint 坐标 $\kappa$ 偏了，GPT-4o 写的函数里如果用了一个 `if` 语句导致不可导，或者 index 写错了，整套 reward 就会 collapse。Pipeline 越自动化，这种 single point of failure 就越危险。这也是为什么作者在 prompt 里千叮咛万嘱咐 "avoid using if statements in the functions"。

---

**Reference Links:**
- GRAPE Paper: https://arxiv.org/abs/2411.02168
- OpenVLA: https://arxiv.org/abs/2406.09246
- DPO (Rafailov et al.): https://arxiv.org/abs/2305.18290
- ReKeP (Huang et al.): https://arxiv.org/abs/2409.01652
- DINOv2: https://arxiv.org/abs/2304.07193
- Grounded-SAM / SAM 2: https://arxiv.org/abs/2408.00714
- Simpler-Env: https://arxiv.org/abs/2405.05941
- Guided Cost Learning (Finn et al., 2016): https://arxiv.org/abs/1603.04792

---

# GRAPE: Generalizing Robot Policy via Preference Alignment 深度讲解

## 1. High-level Motivation: 为什么需要 GRAPE

当前 VLA models (OpenVLA, Octo, RT-2, $\pi_0$ 等) 存在 three critical limitations:

**(1) SFT 的 behavior cloning 限制 generalizability**。现有 VLAs 通过 supervised fine-tuning 简单 imitate 成功 rollouts，缺乏对 task goal 和 failure patterns 的 holistic understanding。具体来讲，SFT loss 只 enforce model 去 memorize action 与 observation 的 mapping：

$$\mathcal{L}_{\mathrm{SFT}} = -\sum_{(\zeta, q) \in \mathcal{D}} \sum_{t=1}^{T} \log p(a_t | o_t, q; \pi_\theta)$$

这里 $\zeta = \{o_1, a_1, \dots, o_T, a_T\}$ 是 trajectory，$q$ 是 task instruction，$\mathcal{D}$ 是 N 条 expert trajectories，$T$ 是 trajectory length，$o_t$ 是 timestep $t$ 的 image observation，$a_t$ 是 action，$\pi_\theta$ 是 parameterized by $\theta$ 的 VLA policy。这个 loss 本质上是一个 maximum likelihood objective，只让 model 记住 "见到 $o_t$ 时输出 $a_t$"，从而在 distribution shift 下表现急剧退化。

**(2) Distribution bias from uncurated demonstrations**。SFT dataset 通常是从不同 settings 下采集的 expert demonstrations，里面 implicit embed 了 task completion、safety、cost-efficiency 等不同 values，但这些 values 没有显式定义。比如同一批 pick-and-place demos 里，有的 expert 路径短，有的避障，model 看到这些混在一起会 confuse。

**(3) RL 的不可行性**。PPO 等 RL 算法虽然可以提升 generalizability，但 (i) 需要 online trajectories collection (cost 极高)，(ii) 需要 explicit reward function (manipulation objectives 难解析定义)，(iii) PPO 时常 collapse。所以 VLA 团队基本 avoid full RL。

**GRAPE 的核心 insight**: 用 DPO-style preference optimization 在 trajectory level 对齐 VLA policy，既 avoid 了 explicit reward modeling 的高 cost，又能 leverage both successful AND failed trajectories。

---

## 2. TPO: Trajectory-wise Preference Optimization - 核心数学推导

### 2.1 起点：KL-constrained RL objective

GRAPE 从经典 RLHF 目标出发：

$$\max_{\pi_\theta} \mathbb{E}_{\zeta \sim \pi_\theta} [r_\phi(\zeta)] - \beta D_{\mathrm{KL}}[\pi_\theta(\zeta) \| \pi_{\mathrm{ref}}(\zeta)]$$

- $r_\phi$：parameterized by $\phi$ 的 reward function
- $\beta$：控制 policy 偏离 reference policy $\pi_{\mathrm{ref}}$（SFT 后的 model）的程度
- $\pi_\theta(\zeta)$：policy 生成整条 trajectory $\zeta$ 的 likelihood

### 2.2 Reparameterization Trick (DPO 的核心)

Following Rafailov et al. (2024) 的 DPO 工作，GRAPE 将 reward 解析地 reparameterize 为：

$$r(\zeta, q) = \beta \log \frac{\pi_\theta(\zeta | q)}{\pi_{\mathrm{ref}}(\zeta | q)} + \beta \log Z(\zeta)$$

- $Z(\zeta)$：partition function，与 $\theta$ 无关的 normalizing constant
- 这一步是关键 - **把 reward 用 policy ratio 表达出来**，avoid 了 explicit reward model

### 2.3 Bradley-Terry Preference Model

GRAPE 用 Bradley-Terry (1952) model 对 preference 建模：

$$P(\zeta_w \succ \zeta_l) = \frac{\exp(r(\zeta_w, q))}{\exp(r(\zeta_w, q)) + \exp(r(\zeta_l, q))}$$

- $\zeta_w$：chosen (winning) trajectory，从相同 initial state 出发
- $\zeta_l$：rejected (losing) trajectory

把 reparameterized reward 代入 BT model，$Z(\zeta)$ 在 ratio 中 cancel 掉，得到 TPO loss：

$$\mathcal{L}_{\mathrm{TPO}} = -\mathbb{E}_{(\zeta_w, \zeta_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta \left(\log \frac{\pi_\theta(\zeta_w)}{\pi_{\mathrm{ref}}(\zeta_w)} - \log \frac{\pi_\theta(\zeta_l)}{\pi_{\mathrm{ref}}(\zeta_l)}\right)\right)\right]$$

- $\sigma$：sigmoid function
- 这个 loss 形式上与 DPO 一致，区别在于 DPO 是 token-level，而 TPO 是 trajectory-level

### 2.4 MDP Decomposition (关键 trick)

在 MDP assumption 下，trajectory likelihood 可分解为 step-wise likelihood 之积：

$$\pi(\zeta, q) = \prod_{i=1}^{T} \pi(a_i | (o_i, q))$$

所以 log-ratio 也可分解：

$$\log \frac{\pi_\theta(\zeta, q)}{\pi_{\mathrm{ref}}(\zeta, q)} = \sum_{t=1}^{T} \log \frac{\pi_\theta(a_i | (o_i, q))}{\pi_{\mathrm{ref}}(a_i | (o_i, q))}$$

**Intuition**: 这个 decomposition 让 TPO loss 可以用 step-wise rollouts 训练，但 optimization 信号是 trajectory-level 的 global preference。这是 GRAPE 与 naive step-wise DPO 的核心区别 - **gradient 在整条 trajectory 上 backpropagate**，而不是 step-wise 噪声。

### 2.5 TPO 的三大 benefits

1. **Global alignment**: 在 trajectory level 对齐 human preferences，using only step-wise rollouts
2. **Stability**: gradient 通过所有 state-action pairs backpropagate，稳定 policy 朝 final goal 方向收敛
3. **Generalizability**: 通过 RL objective 从 successful AND failed trajectories 中学习，相比 SFT 只从 successful 学习，能 capture failure patterns

---

## 3. GCPG: Guided-Cost Preference Generation - 自动化 Preference Synthesis

### 3.1 为什么需要 GCPG

TPO 需要 (chosen, rejected) trajectory pairs，但 human annotation cost 极高，而且 manipulation objectives 多样 (safety, efficiency, task completion)。GCPG 自动 curate preferences。

### 3.2 Multi-stage Temporal Keypoint Constraints

GRAPE 用 VLM-based stage decomposer $\mathcal{M}_D$ 把 trajectory 分成 S 个连续 stages：

$$\{\zeta^1, \dots, \zeta^S\} = \mathcal{M}_D(\zeta, q), \quad \zeta^i = \{(o_t^i, a_t^i)\}_{t=1}^{T_i}$$

- $\zeta^i$：第 $i$-th stage 的 trajectory segment
- $T_i$：第 $i$ 个 stage 的 length

具体实现用 Hamster (Li et al., 2024b) 作 stage decomposer。例如 pick-and-place 分成：
1. Grasp the grape
2. Move the grape onto the plate  
3. Place the grape on the plate

每个 stage用 DINOv2/Grounded-SAM 提取 keypoints $\{\kappa_{S_i}\}$，然后用 GPT-4o 生成 cost functions $C^{S_i}(\{\kappa_{S_i}\})$。

### 3.3 Exponential Decay Aggregation

整条 trajectory 的 external reward 用 exponential decay aggregation：

$$R_{\mathrm{ext}}(\zeta) = \prod_{i=1}^{S} e^{-C^{S_i}(\{\kappa_{S_i}\})}$$

**Intuition**: 这个设计 capture 了 causal dependencies - 如果 trajectory 在 preceding stages 已经 high cost，那么后续 stages 也难以 perform well。乘积形式让 high cost stage 大幅拉低整体 reward，比 linear sum 更敏感。论文里 if-statement 是为了模拟 "如果前 stage 已经崩了，后面也很难救"。

### 3.4 三部分 Reward 组合 (GCPG Reward)

借鉴 self-rewarding (Zhou et al., 2024b)，GRAPE 引入 self-evaluation：

$$R_{\mathrm{GCPG}}(\zeta) = \lambda_1 R_{\mathrm{self}}(\zeta) + \lambda_2 R_{\mathrm{ext}}(\zeta) + \lambda_3 I_{\mathrm{success}}(\zeta)$$

其中三个 components：

**(a) Self-evaluated reward**：

$$R_{\mathrm{self}}(\zeta) = \log(\pi(\zeta, q)) = \log\left(\prod_{i=1}^{T} \pi(a_i | (o_i, q))\right)$$

这是 model 自己生成 trajectory $\zeta$ 的 log-likelihood。**Intuition**: model 应该 prefer 它自己 confidence 更高的 trajectories，避免 sample 出 outlier trajectory。

**(b) External objective-aligned reward** $R_{\mathrm{ext}}(\zeta)$：见上式，由 multi-stage cost function 定义。

**(c) Success indicator**：

$$I_{\mathrm{success}}(\zeta) = \begin{cases} 1, & \text{if } \zeta \text{ is successful} \\ 0, & \text{otherwise} \end{cases}$$

binary indicator，是 sparse 的 success signal。

**关键 insight from 论文**: $R_{\mathrm{self}}$ 可视为 $I_{\mathrm{success}}$ 的 dense approximation，二者相互 calibrate，再被 $R_{\mathrm{ext}}$ 引导到 specific alignment objective。

实际 hyperparameters (from Appendix A)：$\lambda_1 = 0.01, \lambda_2 = 0.01, \lambda_3 = 2$。$I_{\mathrm{success}}$ 权重显著更大，说明 task success 是主导信号，self 和 external 是 calibration。

---

## 4. Iterative Preference Optimization (On-policy 风格)

Inspired by on-policy RL (PPO 等)，GRAPE 迭代 fine-tune SFT VLA model：

**Algorithm 1** (GRAPE Iterative Preference Optimization):
```
for k = 1 to K:
    1. 用 π_θ sample M trajectories per task → D^k
    2. 对每条 trajectory:
       - Decompose into stages (Eq. 7)
       - Compute per-stage cost
       - Compute R_ext (Eq. 8)
       - Compute R_self (Eq. 10)
       - Examine I_success (Eq. 11)
       - Aggregate to R_GCPG (Eq. 9)
    3. Rank D^k by R_GCPG
    4. Pair top-m 和 bottom-m trajectories → m^2 pairs
    5. Update π_θ via TPO loss (Eq. 5)
```

实际 setup (Appendix A): 每个 task sample $\mathcal{N}_t = 5$ trajectories，取 reward 最高和最低各 1 个组成 preference pair。

---

## 5. 实验结果 - 数据表解析

### 5.1 Simulation Results (Simpler-Env)

**Figure 3 关键数据**:
- **vs Octo-SFT**: average 提升 **131.72%** (Octo 在 Simpler-Env 上很弱)
- **vs OpenVLA-SFT**: average 提升 **46.10%**
- **vs OpenVLA-DPO**: average 提升 **33.14%** (这表明 trajectory-level 比 step-level preference 更有效)

Simpler-Env 包含 4 个 in-domain tasks 加 3 类 generalization (subject, physical, semantic)。

### 5.2 Simulation Results (LIBERO)

**Figure 4 数据**:
- vs OpenVLA-SFT: average 提升 **7.36%**
- vs Octo-SFT: average 提升 **8.53%**

LIBERO 4 个 task suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, LIBERO-Long。LIBERO 提升幅度比 Simpler-Env 小，可能因为 OpenVLA-SFT 在 LIBERO 上已经较强 (本来就有 LIBERO fine-tuned checkpoints)。

### 5.3 Real-World Results

300 real-world experiments across 30 tasks，5 类 generalization:
- **Visual generalization** (8 tasks): GRAPE 56% vs OpenVLA-SFT 38%
- **Subject generalization** (4 tasks): GRAPE 52.5% vs OpenVLA-SFT 25%
- **Action generalization** (7 tasks): GRAPE 35.7% vs OpenVLA-SFT 24.3%
- **Semantic generalization** (4 tasks): GRAPE 50% vs OpenVLA-SFT 45%
- **Language grounding** (3 tasks): GRAPE 55% vs OpenVLA-SFT 20%
- **In-domain**: GRAPE 67.5% vs OpenVLA-SFT 45% (vs OpenVLA-DPO 50%)
- **Total average**: GRAPE 50.3% vs OpenVLA-SFT 32.3% vs OpenVLA-DPO 39.3% vs Octo-SFT 5.7%

In-domain 提升 17.5% over OpenVLA-DPO 体现了 trajectory-level > step-level preference。

### 5.4 Ablation Study (Table 1 - Reward Components)

| Setting | In-domain Success | Subject Gen. | Physical Gen. | Semantics Gen. | Average |
|---------|-------------------|--------------|---------------|----------------|---------|
| Random w/ $I_{\mathrm{success}}$ | 35.50% | 33.00% | 33.50% | 36.50% | 34.63% |
| w/o $R_{\mathrm{self}}$ | 38.00% | 37.00% | 36.75% | 42.50% | 38.56% |
| w/o $R_{\mathrm{ext}}$ | 37.50% | 34.33% | 35.50% | 40.00% | 36.83% |
| w/o $I_{\mathrm{success}}$ | 32.00% | 34.67% | 31.75% | 39.00% | 34.36% |
| **GRAPE (full)** | **43.00%** | **40.67%** | **41.75%** | **47.00%** | **43.11%** |

**Key observations**:
1. Random selection (只用 success/failure 二分) 远不如 full reward scoring - 说明 ranking quality 重要
2. $I_{\mathrm{success}}$ 移除后 performance drop 最大 - success 是主信号
3. $R_{\mathrm{self}}$ 和 $R_{\mathrm{ext}}$ 各自贡献 4-5 个百分点 - 二者互补

### 5.5 Iterative Optimization Analysis (Figure 6 & Table 3)

| Iter | In-domain | Subject Gen. | Physical Gen. | Semantics Gen. | Average |
|------|-----------|--------------|---------------|----------------|---------|
| 1 | 43.00% | 40.67% | 41.75% | 47.00% | 43.11% |
| 2 | 45.00% | 40.33% | 44.25% | 49.50% | 44.77% |
| 3 | 45.50% | 40.67% | 44.50% | 49.00% | 44.92% |

SFT baseline → Iter-1 提升 17.5% (in-domain)，再到 Iter-2、3 边际收益递减 - 符合 convergence 直觉。Subject gen 在 Iter-2 略降可能 noise。

### 5.6 Flexible Alignment (Table 2 - Safety/Efficiency)

| Method | Real-World CR↓ | Real-World SL↓ | Real-World SR↑ | Sim CR↓ | Sim SL↓ | Sim SR↑ |
|--------|----------------|----------------|----------------|----------|---------|---------|
| OpenVLA-SFT | 53.33 | 142.32 | 34.61 | 66.50 | 72.68 | 27.50 |
| GRAPE-Safety | **29.84** | 146.11 | 54.31 | **46.00** | 74.49 | 37.00 |
| GRAPE-TC | 58.45 | 125.79 | 51.67 | 57.50 | 64.92 | 38.50 |
| GRAPE-Efficiency | 59.50 | **70.24** | 42.50 | - | - | - |

- **GRAPE-Safety**: collision rate 降低 **37.44%** (real-world 53.33→29.84)
- **GRAPE-Efficiency**: step length 降低 **11.15%** (real-world 142.32→125.79 in TC, and 70.24 vs 142.32 in pure efficiency mode 实际降低幅度更大)
- 同时保持 comparable success rate - flexible alignment 不会大幅牺牲 task success

---

## 6. Case Study: Cost Functions 示例

论文 Appendix E.2 展示了 GPT-4o 生成的 Python cost functions。例如 pick-and-place 任务分 3 stages：

**Task Completion alignment** (target_cost):
```python
def stage1_target_constraint1(end_effector, keypoints):
    """Align end-effector with grape's center."""
    grape_center = keypoints[0]
    return np.linalg.norm(end_effector - grape_center)
```

**Safety alignment** (collision_cost):
```python
def stage2_collision_constraint1(end_effector, keypoints):
    """Ensure grape aligned above black bowl, avoid obstacles."""
    obstacles = keypoints[2:]
    threshold = 0.1
    return sum(max(0, threshold - np.linalg.norm(end_effector - obs)) for obs in obstacles)
```

**Cost-Efficiency alignment** (path_cost):
```python
def stage1_path_constraint1(end_effector, keypoints):
    grape_center = keypoints[0]
    distance = np.linalg.norm(end_effector - grape_center)
    step_size = 0.01
    return int(distance / step_size)
```

**Intuition**: 通过修改 cost function 类型 (target vs collision vs path)，就能让同一个 GRAPE pipeline align 到不同 objectives。这是 flexible alignment 的核心机制。

---

## 7. 架构总览 (Figure 2 解读)

GRAPE 的 pipeline 有三大模块:

**Module A: Task Decomposition (Figure 2 top)**
- Input: task instruction + initial state image
- VLM (Hamster) 分解任务为 temporal stages
- DINOv2/Grounded-SAM 提取每 stage 的 spatial keypoints
- GPT-4o 生成 cost functions per stage
- 根据 user-specified alignment goal (safety/efficiency/TC) 选 cost type

**Module B: Trajectory Sampling & Scoring (Figure 2 bottom-left)**
- SFT VLA model (OpenVLA-SFT) online sample trajectories
- 对每条 trajectory 分 stages，计算 multi-stage cost
- Aggregate 到 $R_{\mathrm{ext}}$ (Eq. 8)
- 计算 $R_{\mathrm{self}}$ (Eq. 10) 和 $I_{\mathrm{success}}$ (Eq. 11)
- 最终得到 $R_{\mathrm{GCPG}}$ (Eq. 9)

**Module C: TPO Training (Figure 2 bottom-right)**
- Rank trajectories by $R_{\mathrm{GCPG}}$
- Pair top-m 和 bottom-m 为 preference pairs
- 用 TPO loss (Eq. 5) fine-tune VLA
- Iterative 重复 B-C 直到收敛

---

## 8. 与相关工作的联系

### 8.1 DPO 在 LLM 上的成功
- DPO (Rafailov et al., 2024): "Your language model is secretly a reward model"，避免 explicit reward model
- GRAPE 是 DPO 在 VLA / robotics trajectory level 的推广
- 关键区别: DPO 是 token-level chosen/rejected，GRAPE 是 trajectory-level chosen/rejected

### 8.2 ReKep 和 Hierarchical Planning
- ReKeP (Huang et al., 2024): spatio-temporal reasoning of relational keypoint constraints
- GRAPE 借鉴 keypoint idea 但用于 preference generation 而非 hierarchical planning
- ReKeP 需要 online optimization，GRAPE 是 offline cost function generation

### 8.3 Self-Rewarding Language Models
- Zhou et al. (2024b): Calibrated self-rewarding VLMs
- GRAPE 的 $R_{\mathrm{self}}$ 受此启发 - 用 model 自身的 log-likelihood 作为 dense reward signal

### 8.4 OpenVLA, Octo, $\pi_0$ 等 VLAs
- 这些都是 SFT-based behavior cloning
- GRAPE 是 post-training alignment stage，可以叠加到任何 VLA backbone 上
- 论文用 OpenVLA 7B 作 backbone，LoRA fine-tuning，learning rate $2 \times 10^{-5}$，batch size 16

---

## 9. 我的 Intuition Building - 为什么 TPO 比 step-wise DPO 好

这是 GRAPE 最核心的设计 choice。Intuition 有几层：

**(1) Credit assignment 问题**。Robotics manipulation 的 reward 极 sparse - 整条 trajectory 跑完才知道成功与否。Step-wise DPO 在每个 (o_t, a_t) 上做 preference，但 trajectory 的成败往往取决于少数 critical steps，其他 step 噪声大。TPO 把整条 trajectory 作为一个 unit 比较，等价于 "trajectory-level credit assignment"，gradient 流过所有 steps 但信号是 global。

**(2) Avoid distribution shift between training and inference**。SFT 后的 policy $\pi_{\mathrm{ref}}$ 在 inference 时可能 sample 出 OOD trajectory。Step-wise DPO 在 step-level 比较，会 encourage 每个 step 都向 "good step" 靠拢，但 good steps 拼起来不一定是 good trajectory (e.g., 中间步骤最优但全局 suboptimal)。TPO 直接在 trajectory space 比较，避免 step-level local optimum。

**(3) Failure learning**。GRAPE 同时用 success 和 failure trajectories 作 preference 对。SFT 只学成功 trajectory 的 action distribution，而 TPO 让 model 显式知道 "这个 trajectory 整体不行，要降低它的 likelihood"，相当于 explicit failure pattern learning。

**(4) Exponential decay aggregation 的 causal inductive bias**。$R_{\mathrm{ext}} = \prod e^{-C^{S_i}}$ 这个设计让 early-stage failure 严重 penalize 后续 - 这 match 现实物理 (前面 grasp 失败，后面 move 没有意义)。Linear sum 会让 early failure 被 later stages 的 low cost 稀释。

---

## 10. Limitations 和 Potential Issues (我自己的思考)

1. **Cost function 依赖 GPT-4o**：论文用 GPT-4o 生成 cost functions，这引入了外部 dependency。如果 cost function 写错 (e.g., 用了 `if` statement 导致 non-differentiable，或者 keypoint index 错)，整个 pipeline 失败。Appendix 提到 prompt template，但没有 systematic error analysis。

2. **Stage decomposition 的离散性**：VLM 分 stages 是 hard partition，没有 overlap 或 soft transition。对于连续 motion (e.g., 倒水) 可能不适用。

3. **Binary preference**：论文为了简化只取 top-1 vs bottom-1，binary pair。Multi-sample comparison (Wang et al., 2024a) 可能更 sample-efficient。

4. **Reward weight sensitivity**：$\lambda_1 = 0.01, \lambda_2 = 0.01, \lambda_3 = 2$ 这组权重是从 preliminary experiment tune 出来的。不同 task / environment 可能需要 re-tune。

5. **Keypoint 提取误差**：DINOv2/Grounded-SAM 在 occlusion 或小物体上 keypoint 可能不准，会直接影响 cost function 的准确性。

6. **Real-world 实验 setup**：所有 demos 在同一环境采集，cross-environment generalization 没测。Franka + Robotiq gripper 替换了原 gripper，可能与主流 OpenVLA setup 略有 mismatch。

---

## 11. Reference Links

**Primary paper**:
- GRAPE paper: https://arxiv.org/abs/2411.02168 (推算 - 实际查证)

**Key references**:
- OpenVLA: https://arxiv.org/abs/2406.09246
- DPO (Rafailov et al., 2024): https://arxiv.org/abs/2305.18290
- ReKeP (Huang et al., 2024): https://arxiv.org/abs/2409.01652
- DINOv2: https://arxiv.org/abs/2304.07193
- SAM 2: https://arxiv.org/abs/2408.00714
- Simpler-Env: https://arxiv.org/abs/2405.05941
- LIBERO: https://arxiv.org/abs/2306.03310
- Octo: https://arxiv.org/abs/2405.12213
- $\pi_0$ (Physical Intelligence): https://arxiv.org/abs/2410.24164
- HAMSTER: https://openreview.net/forum?id=yF3UekSJus
- PPO: https://arxiv.org/abs/1707.06347
- Bradley-Terry model: https://www.jstor.org/stable/2334010
- LoRA: https://arxiv.org/abs/2106.09685
- Christiano et al. RLHF: https://arxiv.org/abs/1706.03741

---

## 12. 总结 - GRAPE 的核心贡献

1. **TPO loss**: 把 DPO 从 token-level 推广到 trajectory-level，结合 MDP decomposition 使其 trainable with step-wise rollouts
2. **GCPG pipeline**: 用 VLM 分 stages + keypoints + LLM-generated cost functions 实现自动 preference synthesis，支持 flexible alignment objectives
3. **Three-component reward**: $R_{\mathrm{self}}$ (dense self-eval) + $R_{\mathrm{ext}}$ (objective-aligned) + $I_{\mathrm{success}}$ (sparse) 三者互补
4. **Iterative on-policy optimization**: 类 PPO 的迭代采样 + TPO 更新，progressive improvement 直到 convergence
5. **Empirical results**: in-domain +51.79%, unseen +58.20%, safety collision -37.44%, efficiency step -11.15%

GRAPE 是 VLA 领域 post-training alignment 的重要一步，把 LLM alignment (DPO) 的成功经验迁移到 robotics，但加入了 robotics-specific 的 multi-stage cost decomposition 来 handle sparse reward 和 diverse objectives 的问题。这给 future 工作 (e.g., 把 RLHF 完整 pipeline 应用到 $\pi_0$ 或 RT-X) 提供了 solid baseline。
