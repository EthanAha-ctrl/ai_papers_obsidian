---
source_pdf: AutoVLAAVision-Language-Action Model for End-to-End Autonomous Driving
  with Adaptive Reasoning and Reinforcement Fine-Tuning.pdf
paper_sha256: ab726d03d44020c67792cf4783a0e05a5615f5f6ee5f2b760f4d69e1502e18d3
processed_at: '2026-07-18T12:26:32-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AutoVLA: Vision-Language-Action Model for End-to-End Autonomous Driving

## 1. Paper 整体定位与 Motivation

AutoVLA 由 UCLA 团队 (Zewei Zhou, Tianhui Cai, Zhiyu Huang, Bolei Zhou, Jiaqi Ma 等) 提出，核心目标是**将 reasoning 和 action generation 统一在一个 autoregressive Transformer 框架内**，实现 end-to-end autonomous driving。

Paper 指出现有 VLA-based autonomous driving 存在两个 critical limitations：

**Limitation 1: Physically-infeasible or complex action generation**
- 一些方法 [35-37] 直接用 VLM 生成 textual waypoints，但这些 outputs 物理上可能不可行，且会 suffer from mode collapse
- 另一些方法 [38-43] 引入 intermediate meta-actions 或 latent action tokens，再用 downstream planner 解码 trajectory，但这破坏了 end-to-end optimization 或增加 model complexity

**Limitation 2: Inflexible reasoning across diverse scenarios**
- 多数 model [44, 45] 使用 fixed reasoning strategy，无法根据场景复杂度自适应切换
- DriveVLM [46] 虽引入 dual-process paradigm，但依赖 separate modules (VLM + conventional E2E model)，架构复杂

AutoVLA 的核心 insight：**将 physical action tokens 直接 integrate 到 pretrained VLM 的 vocabulary 中**，让 language model 通过 next-token prediction 直接学习 planning policy，同时通过 SFT + RFT 实现 adaptive fast/slow thinking。

项目主页: https://autovla.github.io/

---

## 2. Model Architecture 详解

### 2.1 Inputs 定义

AutoVLA 的 input 包含三部分：

**Multi-view camera streams $C$**：
- 三个 RGB camera: front, front-left, front-right
- 每个 camera stream: $c^i = [c^i_{t-3}, c^i_{t-2}, c^i_{t-1}, c^i_t]$
- 采样频率 2 Hz，包含当前 frame 和前 3 帧，提供 temporal dynamics
- 图像 resize 到 $28 \times 28 \times 128$ pixels (token 化后的 resolution)

**Navigation instruction $I$**：
- 高层指令，如 "Turn Left", "Go Straight"

**Ego vehicle state $S$**：
- 当前 velocity, acceleration
- 历史动作 (在 Waymo 上包含 4-second history of positions 和 velocities)

### 2.2 Base VLM Backbone

采用 **Qwen2.5-VL-3B** [21] 作为 backbone。选择 3B variant 的原因：
- 开源，便于 fine-tune
- 在 efficiency 和 performance 之间提供好的 trade-off
- 适合 onboard deployment

Qwen2.5-VL 本身具备 strong visual understanding capability，paper 利用其 pretrained knowledge，通过 fine-tuning 适配 driving task。

### 2.3 Physical Action Tokenization (核心创新)

这是 paper 的核心 technical contribution 之一。需要把 continuous trajectory $\mathbf{P} \in \mathbb{R}^{T \times d}$ 转化为 discrete action tokens $\mathbf{a} = [a_1, \dots, a_T]$，其中 $a_t \in \mathcal{A}$。

**Action token 定义**：
每个 action token 表示 0.5 seconds 的 short-term spatial movement：
$$a_t = (\Delta x, \Delta y, \Delta \theta)$$
- $\Delta x$: longitudinal displacement
- $\Delta y$: lateral displacement  
- $\Delta \theta$: heading change

**K-disk clustering 方法构建 codebook**：

从 Waymo Open Motion Dataset (WOMD) [103] 采样 short motion segments (每段 0.5s)，基于 vehicle 的 final-frame bounding-box contour (位置、尺寸、heading) 计算。

K-disk clustering 算法：
1. 迭代选择 representative segments $\{m_1, \dots, m_K\}$
2. 约束：任意两个 selected segments 之间的 distance $\geq \delta = 0.05$ m (用 average contour distance 衡量)
3. 对每个 selected segment $m_k$，提取 $(\Delta x, \Delta y, \Delta \theta)$ 作为 action token $a_k$

最终 codebook: $\mathcal{A} = \{a_1, \dots, a_K\}$, $K = 2048$。

**为什么 K=2048？**
Table 4 的 ablation 显示：
- K=256: ADE=0.0687m, Movement Coverage (MC) = 86.47% (覆盖不足)
- K=1024: ADE=0.0253m, MC=97.41%
- K=2048: ADE=0.0182m, MC=99.42%, Codebook Usage (CU)=100%
- K=4096: ADE=0.0141m, MC=100%, 但 CU=91.46% (有冗余 token)

K=2048 是 accuracy、coverage 和 efficient usage 的 best trade-off。

**对比其他 tokenization 方法** (Table 4)：
- **RT-1 [102]** (action bin): 对 acceleration 和 steering rate 用 uniform bins，再通过 kinematic model reconstruct。因为只有 trajectory-level data，control actions 需间接 inference，reconstruction error 最高 (K=2048: ADE=0.1014m)
- **FAST [91]** (DCT): 用 discrete cosine transform 把 fixed-horizon trajectory 转成 variable-length token sequence。对 codebook size 敏感，K=4096 才接近 K-disk 的精度。且 variable length 使 fixed-horizon planning 困难
- **K-disk (Ours)**: 在所有 codebook size 下都达到最高 reconstruction accuracy

这些 action tokens 作为 additional vocabulary 加入 VLM：`<action_0>, <action_1>, ..., <action_2047>`。Inference 时 model autoregressively 输出 action tokens，再通过 codebook decode 回 continuous trajectory。

### 2.4 Unified Reasoning and Action

AutoVLA 支持 **dual thinking modes**：

**Fast thinking mode**：
- 直接输出 action tokens，无 CoT reasoning
- 用于 straightforward scenarios
- Runtime: avg 1.072s (Table 2)

**Slow thinking mode**：
- 先生成 structured CoT reasoning (scene description, critical objects, intentions, decision)
- 再输出 action tokens
- 用于 complex scenarios
- Runtime: avg 10.518s (Table 2)

通过 system prompt 和 response format 设计，让 model 学会根据场景自适应切换。

---

## 3. Reasoning Data 自动标注 Pipeline

### 3.1 Motivation

高质量 driving reasoning dataset 的挑战：
1. Scenario diversity 有限，examples 重复
2. Critical perceptual cues 不足 (如 traffic signs, turn signals)
3. Reasoning quality 低 (如无 justification 地反复 stop)

### 3.2 Pipeline 设计

使用 **Qwen2.5-VL-72B** [21] 作为 reasoning annotation engine，实现 knowledge distillation。

**System Prompt** 包含：
- Model role definition
- Task description
- Expected CoT format
- Representative examples

**CoT reasoning 四个步骤**：
1. **Scene description and analysis**: 描述整体场景
2. **Critical object identification**: 识别关键物体
3. **Intention reasoning**: 推理周围 agents 的意图
4. **Decision-making and meta-action**: 决策

**User Message** 包含：
- Driving instructions
- Ego vehicle states
- Multi-view camera streams
- **Ground-truth driving meta-action 作为 hint** (关键设计)：引导 model 生成 causal explanation，将 decision 与 context 显式关联，减少 nonsensical outputs

### 3.3 数据规模

| Dataset | Train Samples | Reasoning Samples | Test Samples |
|---------|----------------|-------------------|--------------|
| nuPlan (NAVSIM) | 166.3k | 45.6k | 12.1k |
| nuScenes | 19.0k | 2.9k | 5.6k |
| Waymo | 23.8k | 7.2k | 1.5k |
| CARLA | 274.5k | 53.2k | - |

**Quality Check**：3,000 samples 人工评估，accuracy 88.8%。

---

## 4. Supervised Fine-Tuning (SFT)

### 4.1 Training Objective

Output sequence: $\mathbf{x} = [l_1, \dots, l_L, a_1, \dots, a_T]$
- Language tokens for reasoning: $\mathbf{l} = [l_1, \dots, l_L]$
- Action tokens: $\mathbf{a} = [a_1, \dots, a_T]$ (位置 $x_{L+1}$ 到 $x_{L+T}$)

**Loss function 1 - Language modeling loss**:
$$\mathcal{L}_{\mathrm{LM}} = -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(x_i \mid x_{<i}, C, I, S)$$

- $N = L + T$: 总 token 数
- $p_\theta$: parameterized by $\theta$ 的 model 预测分布
- $x_{<i}$: 前面所有 tokens (causal modeling)
- $C, I, S$: camera, instruction, state inputs

**Loss function 2 - Action loss** (辅助 loss，focus on planning accuracy):
$$\mathcal{L}_{\mathrm{action}} = -\frac{1}{T} \sum_{i=L+1}^{L+T} \log p_\theta(x_i \mid x_{<i}, C, I, S)$$

- 只在 action token 位置计算 loss
- 强调 planning accuracy

**Combined SFT loss** (with per-sample weighting):
$$\mathcal{L}_i^{\mathrm{SFT}} = w_i \cdot (\mathcal{L}_{\mathrm{LM},i} + \lambda_a \mathcal{L}_{\mathrm{action},i})$$

$$w_i = \begin{cases} \lambda_{\cot} & \text{if CoT is present in GT} \\ 1 & \text{otherwise} \end{cases}$$

- $\lambda_a = 1$: action loss 权重
- $\lambda_{\cot} = 40$: CoT sample 的权重 (大幅提升 CoT 学习的重要性，平衡数据不平衡)

**为什么 $\lambda_{\cot} = 40$ 这么大？**
因为 action-only data (e.g., nuPlan 中 166.3k - 45.6k = 120.7k) 远多于 CoT data (45.6k)，需要大幅加权才能让 model 充分学习 reasoning capability。

### 4.2 Training Details

- Learning rate: $1 \times 10^{-5}$
- FSDP (Fully Sharded Data Parallel) strategy
- 5 epochs
- 8 NVIDIA L40S GPUs
- Per-GPU batch size: 1, gradient accumulation: 4 steps
- Effective batch size: 32
- Learning rate warm-up: 500 steps
- Decay: 2% every 2000 steps
- Gradient clipping: max value 1.0
- Mixed precision: BFloat16
- Gradient checkpointing enabled

---

## 5. Reinforcement Fine-Tuning (RFT) - 核心创新之二

### 5.1 为什么需要 RFT？

SFT 后 model 在 slow thinking mode 下 runtime 高达 10.518s (Table 2)，因为生成冗长 CoT。RFT 的目标：
1. 提升 planning performance
2. Enable adaptive reasoning (simple scenarios 用 fast thinking)
3. 减少 unnecessary reasoning

### 5.2 GRPO Algorithm

采用 **Group Relative Policy Optimization (GRPO)** [49]，源于 DeepSeekMath。GRPO 的优势：
- 用 group-based sampling 替代 conventional state-value estimators / critic models
- 加速 training
- 天然 align with planning 的 multimodality (同一 scenario 多个 feasible trajectories)

**GRPO Objective**:

$$\mathcal{J}_{\mathrm{GRPO}}(\theta) = \mathbb{E}_{q, \{o_i\} \sim \pi_{\theta_{\mathrm{old}}}(O|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \mathcal{J}_i^R - \beta \mathbb{D}_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}}) \right) \right]$$

变量含义：
- $\theta$: current policy 参数
- $\theta_{\mathrm{old}}$: old policy 参数 (用于 importance sampling ratio)
- $q$: scenario input query (sensor images + ego state + instruction)
- $O = \{o_1, \dots, o_G\}$: 从 old policy 采样的 $G$ 个 candidate outputs
- $G$: group size
- $\beta = 0.04$: KL divergence regularization weight
- $\pi_{\mathrm{ref}}$: reference policy (SFT model)，防止 policy 漂移过远

**Clipped surrogate objective** (PPO-style):

$$\mathcal{J}_i^R = \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\mathrm{old}}}(o_i|q)} A_i, \ \mathrm{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\mathrm{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i\right)$$

- $\epsilon$: clipping range hyperparameter
- Importance sampling ratio: $\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\mathrm{old}}}(o_i|q)}$

**Group-relative advantage**:

$$A_i = \frac{r_i - \mathrm{mean}(\{r_j\}_{j=1}^G)}{\mathrm{std}(\{r_j\}_{j=1}^G)}$$

- $r_i$: reward for sample $o_i$
- Advantage 通过 group 内 normalization 计算，无需 critic model

### 5.3 KL Divergence Regularization

$$\mathbb{D}_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}}) = \frac{\pi_{\mathrm{ref}}(o_i|q)}{\pi_\theta(o_i|q)} - \log\left(\frac{\pi_{\mathrm{ref}}(o_i|q)}{\pi_\theta(o_i|q)}\right) - 1$$

这是 **k3 estimator** (Schulman approximation)，penalize policy 偏离 reference policy 过远，确保训练 stability。

### 5.4 Reward Function 设计

**总 reward**:
$$r = r_{\mathrm{Driving}} - \lambda_r r_{\mathrm{CoT}}$$

- $\lambda_r = 0.3$: balance weight (driving reward 主导)

**Driving reward $r_{\mathrm{Driving}}$** (分 dataset 设计)：

**(a) nuPlan dataset - PDMS (Predictive Driver Model Score)**:

$$r_{\mathrm{Driving}} = \mathrm{PDMS} = \mathrm{NC} \times \mathrm{DAC} \times \left(\frac{5\mathrm{TTC} + 2\mathrm{C} + 5\mathrm{EP}}{12}\right)$$

- **NC** (No at-fault Collision): 无责任碰撞
- **DAC** (Drivable Area Compliance): 遵守可行驶区域
- **TTC** (Time-to-Collision): 碰撞时间
- **C** (Comfort): 舒适度
- **EP** (Ego Progress): 自车进度
- 权重 5:2:5 反映 safety 和 progress 的重要性高于 comfort
- Output 范围 [0, 1]
- 若 planner 失败，score = 0

**(b) Waymo dataset - ADE (Average Displacement Error)**:

$$r_{\mathrm{Driving}} = \frac{\delta - \mathrm{ADE}}{\kappa}$$

$$\mathrm{ADE} = \frac{1}{T} \sum_{t=1}^{T} \|\hat{\mathbf{y}}_t - \mathbf{y}_t\|_2$$

- $\delta = 2$: maximum displacement error
- $\kappa = 10$: scaling factor
- $\hat{\mathbf{y}}_t$: predicted trajectory at time $t$
- $\mathbf{y}_t$: ground truth trajectory
- $T$: time steps

(因为 Waymo RFS labels 有限，只有 480 个 validation samples，所以用 ADE 作为代理 reward)

**CoT length penalty $r_{\mathrm{CoT}}$** (sigmoid-based):

$$r_{\mathrm{CoT}} = \frac{1}{1 + e^{-(L - L_{\mathrm{tol}})\gamma}}$$

- $L$: CoT reasoning 长度
- $L_{\mathrm{tol}} = 400$: tolerance threshold
- $\gamma = 2 \times 10^{-3}$: scaling coefficient 控制 penalty curve 陡度

当 $L < L_{\mathrm{tol}}$ 时，penalty 接近 0；$L > L_{\mathrm{tol}}$ 时，penalty 接近 1。这鼓励 model 在 simple scenarios 用 fast thinking。

### 5.5 RFT Implementation

- LoRA adapter [98]: rank=8, alpha=8, dropout=0.1
- Vision encoder frozen
- Learning rate: $3 \times 10^{-5}$
- KL regularization weight $\beta = 0.04$
- 6,000 training steps
- Single policy update per step (简化版，无需 clipping 或 tracking old policy)
- Sampling: temperature=1.0, top-p=1.0, top-k=0.0 (鼓励 diverse exploration during GRPO sampling)

**Algorithm 1 (RFT with GRPO) 详解**：

```
Input: SFT policy π_SFT, Codebook A, Group size G, Steps K
       Dataset D, r_Driving, r_CoT, λ_r, β

1: π_ref ← π_SFT, π_θ ← π_SFT
2: for step 1 to K:
3:   Sample scenario U from D
4:   for sample i from 1 to G:
5:     Sample (q, o_i) from π_θ, record per-token prob π_θ(o_i|q)
6:     π_θ_old(o_i|q) ← π_θ(o_i|q)  # 保存 old policy prob
7:     Compute π_ref(o_i|q)  # reference policy prob
8:     Decode trajectory τ ← A(o_i)  # action tokens → trajectory
9:     r_i ← r_Driving(τ, U) - λ_r * r_CoT(o_i)
10:    end
11:    r̄, σ_r ← mean(r_1..r_G), std(r_1..r_G)
12:    A_i ← (r_i - r̄) / σ_r
13:    L_RFT ← Σ_i (-ratio * A_i + β * KL)
14:    Update π_θ by L_RFT
15: end
```

---

## 6. 实验结果详解

### 6.1 Data Scaling Results (Figure 4)

训练数据量从 10k → 50k → 100k → 185k (nuPlan+nuScenes 混合)，对比 action-only 和 CoT-enhanced：

**关键观察**：
- 数据量增加，planning performance 持续提升
- **nuPlan**: 数据量 < 50k 时，CoT 不如 action-only (因为数据不足以 learn structured reasoning)
- **nuPlan**: 数据量增加后，CoT 超过 action-only (showcasing scalability of reasoning-augmented learning)
- **nuScenes**: 类似趋势，CoT 在数据量增加后 consistently 优于 action-only

**Intuition**: Reasoning 能力的 emergence 需要 sufficient data。这与 LLM 的 scaling law 一致 - small data 下 structured reasoning 反而增加 learning difficulty。

### 6.2 NAVSIM (nuPlan) Benchmark (Table 1)

| Method | PDMS ↑ | Collision ↑ | Progress ↑ | TTC ↑ | Comfort ↑ |
|--------|--------|-------------|-------------|-------|-----------|
| Ego Status MLP | 66.40 | 93.09 | 63.20 | 84.02 | 99.97 |
| TransFuser [58] | 83.88 | 97.78 | 78.88 | 92.89 | 99.98 |
| DRAMA [56] | 86.87 | 98.19 | 81.33 | 94.17 | 100.00 |
| Hydra-MDP [57] | 91.26 | 99.07 | 85.20 | 96.56 | 100.00 |
| Centaur [59] | 92.10 | 99.23 | 85.96 | 97.17 | 99.97 |
| TrajHF [85] | 93.95 | 99.30 | 90.39 | 98.02 | 99.81 |
| **AutoVLA (One-shot)** | 80.54 | 96.89 | 75.82 | 88.06 | 99.94 |
| **AutoVLA (Post-RFT)** | 89.11 | 98.41 | 81.87 | 98.04 | 99.94 |
| **AutoVLA (Best-of-N)** | 92.12 | 99.14 | 87.55 | 97.12 | 99.98 |

**关键观察**：
- One-shot (SFT only): PDMS=80.54
- Post-RFT: PDMS=89.11 (提升 8.57 / 10.6%)
- Best-of-N: PDMS=92.12 (用 oracle scorer 从 6 个 candidate 中选最优)
- RFT 让 TTC 从 88.06 跃升到 98.04 (大幅提升 safety)
- 与 SOTA TrajHF (93.95) 相比仍有 gap，但 AutoVLA 是 unified VLA model

### 6.3 RFT Performance (Figure 5)

**Figure 5(a)**:
- RFT 后 PDMS 提升 10.6%
- Runtime 减少 66.8% (avg over 500 testing scenarios)

**Figure 5(b)** - Training reward curves:
- 不同 GRPO group sample sizes 对比
- 更大 group → 更好 performance (broader exploration)
- Reward 曲线 progressive improvement

**Figure 5(c)** - Qualitative comparison:
- SFT model 在 simple scenarios 仍生成 slow reasoning (suboptimal)
- RFT model 在 simple scenarios 切换到 fast thinking
- RFT 在 complex scenarios 仍保持 reasoning capability
- SFT model 由于 error accumulation 产生 suboptimal plans
- RFT model (PDMS-based reward) 生成更好 trajectories

### 6.4 nuScenes Results (Table S2)

**ST-P3 metrics (L2 / Collision)**:

| Method | L2 1s | L2 2s | L2 3s | Avg | Col 1s | Col 2s | Col 3s | Avg |
|--------|-------|-------|-------|-----|--------|--------|--------|-----|
| ST-P3 [14] | 1.33 | 2.11 | 2.90 | 2.11 | 0.23 | 0.62 | 1.27 | 0.71 |
| VAD [66] | 0.17 | 0.34 | 0.60 | 0.37 | 0.07 | 0.10 | 0.24 | 0.14 |
| UniAD [65] | 0.44 | 0.67 | 0.96 | 0.69 | 0.04 | 0.08 | 0.23 | 0.12 |
| EMMA [37] | 0.14 | 0.29 | 0.54 | 0.32 | - | - | - | - |
| OpenDriveVLA-3B [34] | 0.14 | 0.30 | 0.55 | 0.33 | 0.02 | 0.07 | 0.22 | 0.10 |
| **AutoVLA (action only)** | 0.22 | 0.39 | 0.61 | 0.41 | 0.10 | 0.17 | 0.28 | 0.18 |
| **AutoVLA (w/ CoT)** | 0.21 | 0.38 | 0.60 | 0.40 | 0.13 | 0.18 | 0.28 | 0.20 |

**Intuition**: nuScenes scenarios 多数相对 straightforward，CoT reasoning 在 quantitative metrics 上没有显著 gain。这印证了 adaptive reasoning 的必要性 - 简单场景不需要 slow thinking。

### 6.5 Waymo End-to-End Benchmark (Figure 6, Table S3)

**Waymo Leaderboard (截至 May 22, 2025)**:

| Method | RFS (Overall) ↑ | ADE 5s ↓ | ADE 3s ↓ | RFS (Spotlight) ↑ |
|--------|-----------------|----------|----------|-------------------|
| Poutine | 7.9860 | 2.7419 | 1.2055 | 6.8929 |
| HMVLM | 7.7367 | 3.0715 | 1.3269 | 6.7269 |
| UniPlan | 7.6925 | 2.9864 | 1.3083 | 6.6544 |
| DiffusionLTF | 7.5919 | 2.9768 | 1.3605 | 6.5688 |
| **AutoVLA** | **7.5566** | 2.9580 | 1.3507 | **6.9436** |
| Swin-Trajectory | 7.5432 | 2.8135 | 1.2082 | 6.6791 |
| OpenEMMA | 5.1575 | 12.4755 | 6.6842 | 4.7131 |

**关键观察**：
- AutoVLA 在 **RFS Spotlight** (最 challenging scenarios) 排名 **第一** (6.9436)
- 这验证了 CoT reasoning 在 complex / long-tail scenarios 的价值
- ADE metrics 也 competitive (2.9580 vs SOTA 2.7024)

**Ablation Study (Table S4)**:

| Camera | Pretraining | Output | RFS ↑ | ADE 5s ↓ |
|--------|-------------|--------|-------|----------|
| Front | None | Action-only | 6.938 | 3.595 |
| Front | None | CoT-enhanced | 7.127 | 3.188 |
| Multi | None | Action-only | 7.239 | 3.243 |
| Multi | None | CoT-enhanced | 7.283 | 3.182 |
| Multi | nuX | Action-only | 7.406 | 3.116 |
| Multi | nuX | CoT-enhanced | 7.447 | 3.115 |
| Multi | nuX | **Post-RFT** | **7.557** | **2.958** |

**关键 insights**：
1. **Multi-camera > Front-only**: 多视角 consistently 提升 performance
2. **CoT > Action-only**: reasoning 帮助尤其在 challenging scenarios
3. **Pretraining on nuX (nuPlan+nuScenes)**: 大幅提升，因为更多 diverse data 增强 scene understanding
4. **Post-RFT 最佳**: RL fine-tuning 对齐 task-specific reward，显著提升

### 6.6 CARLA Closed-loop (Table 3)

| Method | Driving Score ↑ | Success Rate (%) ↑ | Efficiency ↑ | Comfortness ↑ |
|--------|-----------------|-------------------|--------------|---------------|
| AD-MLP [99] | 18.05 | 0.00 | 48.45 | 22.63 |
| UniAD-Base [65] | 45.81 | 16.36 | 129.21 | 43.58 |
| VAD [66] | 42.35 | 15.00 | 157.94 | 46.01 |
| TCP-traj [100] | 59.90 | 30.00 | 76.54 | 18.08 |
| DriveAdapter [101] | 64.22 | 33.08 | 70.22 | 16.01 |
| Orion [42] | 77.74 | 54.62 | 151.48 | 17.38 |
| **AutoVLA** | **78.84** | **57.73** | 146.93 | 39.33 |

**关键观察**：
- AutoVLA 在 closed-loop 上达到 **SOTA** Driving Score (78.84) 和 Success Rate (57.73%)
- 这是真正的 closed-loop 测试 (Bench2Drive benchmark, 44 interactive scenarios)
- Comfortness (39.33) 优于 Orion (17.38)，但低于 UniAD (43.58)
- Efficiency 中等，trade-off 换取了 success rate

---

## 7. Ablation Studies 深度分析

### 7.1 Text Waypoint vs Physical Action (Table 5)

| Metric | Text Waypoint | Physical Action |
|--------|---------------|-----------------|
| PDM Score ↑ | 71.31 | 80.54 |
| Avg. L2 (m) ↓ | 0.89 | 0.70 |
| Avg. Col. (%) ↓ | 0.36 | 0.31 |
| Runtime (s) ↓ | 7.65 | 3.95 |

**Intuition**: LLM 处理 precise numerical reasoning 有 inherent limitation。用 text 表示 waypoints 需要 decode 数值，计算昂贵且不精确。Physical action tokens 直接对应 feasible maneuvers，避免数值 reasoning，且 runtime 减半。

### 7.2 Tokenization Methods (Table 6)

| Tokenization | PDMS ↑ | Collision ↑ | Progress ↑ |
|--------------|--------|-------------|------------|
| FAST (DCT) [91] | 67.63 | 92.74 | 64.09 |
| K-disk (Ours) | 80.54 | 96.89 | 75.82 |

K-disk 在所有指标上大幅领先 FAST。FAST 的 variable-length token sequence 使 fixed-horizon planning 困难。

---

## 8. Runtime Analysis (Table 2)

| Thinking Mode | Min (s) | Max (s) | Avg (s) |
|---------------|---------|---------|---------|
| Fast Thinking | 0.997 | 1.116 | 1.072 |
| Slow Thinking | 7.607 | 13.706 | 10.518 |

**Slow thinking 比 fast thinking 慢约 10x**。这是 RFT 的核心 motivation - 通过 adaptive reasoning 在简单场景用 fast thinking，可以大幅降低平均 runtime (paper 报告 RFT 后 runtime 减少 66.8%)。

---

## 9. Critical Analysis & Intuition Building

### 9.1 为什么 AutoVLA 有效？

**1. Action Tokenization 解决 "language model 不能精确输出 numerical values" 的问题**

LLM 本质是 next-token predictor，对 continuous numerical output 本身不擅长。通过将 trajectory 离散化为 2048 个 physical action tokens，把 planning 转化为 classification 问题 (next-token prediction)，leveraging LLM 的核心 strength。

**2. Dual thinking mode 借鉴 Kahneman 的 System 1 / System 2 理论**

- System 1 (fast thinking): 直觉式、快速、自动 - 对应简单场景的直接 action generation
- System 2 (slow thinking): 推理式、缓慢、effortful - 对应复杂场景的 CoT reasoning

RFT 让 model 学会**何时**用哪种 mode，而非固定使用某一种。

**3. GRPO 的 group-based advantage 天然 fit planning multimodality**

同一 scenario 下可能有多个 feasible trajectories (例如不同的 lane change 时机)。传统 RL 的 per-sample advantage 无法 capture 这种 multimodality，而 GRPO 通过 group normalization 让 model 学习 group 内的 relative preference。

### 9.2 Key Insights

**Insight 1: Reasoning capability 需要 sufficient data 才能 emerge**

Data scaling experiments 显示 < 50k 数据时 CoT 不如 action-only。这暗示 reasoning 是 emergent capability，需要 threshold 以上的数据量。这与 LLM scaling laws 一致。

**Insight 2: RFT 是 SFT 后的关键 refinement step**

RFT 带来：
- PDMS: 80.54 → 89.11 (+10.6%)
- Runtime: -66.8%
- Adaptive reasoning capability

SFT 学会 "如何 reasoning"，RFT 学会 "何时 reasoning"。

**Insight 3: Action codebook 设计至关重要**

K-disk clustering + K=2048 是 reconstruction accuracy、movement coverage、codebook usage 的最佳平衡点。过于精细 (K=4096) 会有冗余 token，过于粗糙 (K=256) 覆盖不足。

### 9.3 Limitations

1. **Runtime 仍未达 real-time**: 即使 RFT 后 ~1 Hz (fast thinking avg 1.072s)，对高速驾驶仍不够 (通常需要 10+ Hz)。Paper 承认需要 model quantization 等优化
2. **GPU-dependent**: 需要 significant memory 和 computing
3. **CoT penalty 可能过度简化**: $L_{\mathrm{tol}} = 400$ 是固定阈值，不同 scenario 的 optimal reasoning length 可能差异很大
4. **Best-of-N 需要 oracle scorer**: 实际部署无法使用，是 upper bound performance

### 9.4 与 Related Work 对比

**vs EMMA [37]** (Waymo 的 end-to-end multimodal model):
- EMMA 用 text waypoints，AutoVLA 用 physical action tokens
- EMMA 没有 adaptive reasoning

**vs DriveVLM [46]**:
- DriveVLM 用 separate modules (VLM + conventional E2E model)
- AutoVLA 是 unified autoregressive model

**vs ORION [42]**:
- ORION 引入 generative planner，增加 model complexity
- AutoVLA 直接 integrate action tokens 到 VLM

**vs AlphaDrive [38]**:
- AlphaDrive 用 GRPO 但仅限于 high-level meta-actions
- AutoVLA 应用 RFT 到 end-to-end VLA (scene reasoning + low-level planning)

---

## 10. Reference Links

- **Project Page**: https://autovla.github.io/
- **Qwen2.5-VL**: https://arxiv.org/abs/2502.13923
- **DeepSeek-R1 (GRPO)**: https://arxiv.org/abs/2501.12948
- **DeepSeekMath (GRPO 原始 paper)**: https://arxiv.org/abs/2402.03300
- **NAVSIM Benchmark**: https://arxiv.org/abs/2406.06978 (Hydra-MDP)
- **nuPlan Dataset**: https://arxiv.org/abs/2403.04133
- **Waymo Open Dataset**: https://arxiv.org/abs/2510.26125
- **nuScenes**: https://arxiv.org/abs/1903.11027
- **Bench2Drive (CARLA)**: https://arxiv.org/abs/2406.03877
- **DriveLM**: https://arxiv.org/abs/2310.01060
- **OpenVLA**: https://arxiv.org/abs/2406.09246
- **FAST (Action Tokenization)**: https://arxiv.org/abs/2501.09747
- **LoRA**: https://arxiv.org/abs/2106.09685
- **UniAD**: https://arxiv.org/abs/2212.10156
- **VAD**: https://arxiv.org/abs/2303.12077
- **EMMA**: https://arxiv.org/abs/2410.23262
- **DriveVLM**: https://arxiv.org/abs/2402.12289
- **ORION**: https://arxiv.org/abs/2503.19755
- **AlphaDrive**: https://arxiv.org/abs/2503.07608
- **OpenDriveVLA**: https://arxiv.org/abs/2503.23463
- **TrajHF**: https://arxiv.org/abs/2503.10434
- **Gen-Drive**: https://arxiv.org/abs/2410.05582
- **RAD**: https://arxiv.org/abs/2502.13144
- **Waymo Open Motion Dataset**: https://arxiv.org/abs/2104.10133
- **RT-1**: https://arxiv.org/abs/2212.06817
- **DPO**: https://arxiv.org/abs/2305.18290

---

## Summary

AutoVLA 的核心贡献是**将 physical action tokens 直接 integrate 到 VLM 的 vocabulary 中**，unifying reasoning 和 action generation 在一个 autoregressive framework 内。通过 SFT 实现 dual thinking modes，再通过 GRPO-based RFT 实现 adaptive reasoning - 简单场景用 fast thinking，复杂场景用 slow thinking。实验证明 RFT 提升 PDMS 10.6%，减少 runtime 66.8%，在 nuPlan, nuScenes, Waymo, CARLA 多个 benchmark 上达到 competitive performance，特别是在 Waymo Spotlight (challenging scenarios) 上排名第一，验证了 CoT reasoning 在 long-tail scenarios 的价值。

整个 framework 的 elegance 在于：把复杂的 perception-reasoning-planning pipeline 转化为单一的 next-token prediction 问题，让 model 通过 large-scale pretraining 获得的 world knowledge 自然地服务于 driving task。
