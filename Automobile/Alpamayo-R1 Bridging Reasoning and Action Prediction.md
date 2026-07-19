---
source_pdf: Alpamayo-R1 Bridging Reasoning and Action Prediction.pdf
paper_sha256: a887dc13b3cf672b800968d3c8011d3d2741d8d68b342a0b4a390f7d4fe67ce1
processed_at: '2026-07-18T07:44:47-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Alpamayo-R1 深度讲解：从 Reasoning 到 Action Prediction 的 VLA 架构

## 1. 论文定位与核心命题

这篇 NVIDIA 的工作本质上回答一个问题：**如何在 VLA (Vision-Language-Action) 模型中让 reasoning 真正"驱动" action，而不是装饰品？**

传统 E2E driving 有两类极端：
- **Trajectory-only baseline**：直接 pixels → trajectory，黑盒，长尾场景脆
- **Free-form reasoning VLA**：让 LLM/VLM 生成自然语言 reasoning，但 reasoning 和 action 之间 causal 链条断裂（论文 Fig. 2 展示了 vague description、superficial reasoning、causal confusion 三类常见病）

Alpamayo-R1 (AR1) 想要的是 **decision-grounded, causally linked reasoning**——每一条 reasoning trace 必须对应一个 closed-set driving decision，且 causal factors 只能来自 observed history window，不能"偷看"未来。这是一个很强的 inductive bias。

论文链接：
- HuggingFace 模型：https://huggingface.co/nvidia/Alpamayo-R1-10B
- 代码：https://github.com/NVlabs/alpamayo
- Cosmos-Reason：https://arxiv.org/abs/2503.15558
- AlpaSim：https://github.com/NVlabs/alpasim

## 2. 架构整体解剖

整体是一个 **modular VLA**，不是 monolithic black box。关键设计哲学是：VLM backbone 可以替换（从 0.5B 到 7B 到最终 10B），但 vision encoder 和 action decoder 是 domain-specific 组件。

### 2.1 Problem Formulation

输入序列构造（Eq. 1）：

$$[o_{\text{image}}, o_{\text{egomotion}}, \text{REASON}, \tau]$$

其中每个 component condition 在前面所有 component 上。这是 autoregressive 联合建模。$\tau$ 是 6.4s 未来轨迹，10 Hz 采样，共 64 个 waypoints：

$$\tau = \{(x^i, y^i, \theta_{\text{yaw}}^i)\}_{i=1}^{64}$$

$(x^i, y^i)$ 是 BEV 平面坐标，$\theta_{\text{yaw}}^i$ 是 yaw 角，都在 ego vehicle 坐标系下。

### 2.2 关键设计 1：Unicycle Dynamics 表示

这是我觉得最聪明的一个 engineering choice。论文不直接预测 raw $(x, y)$ waypoints（对 sensor noise 敏感、convergence 差），而是预测 **control input**：

$$\mathbf{a} = \{(a^i, \kappa^i)\}_{i=1}^{64}$$

其中 $a^i$ 是 acceleration，$\kappa^i$ 是 curvature。然后通过 unicycle dynamics (Eq. 5) 积分到 $\tau$：

$$\mathbf{x}^{i+1} = \begin{pmatrix} x^{i+1} \\ y^{i+1} \\ \theta^{i+1} \\ v^{i+1} \end{pmatrix} = \begin{pmatrix} x^i + \frac{\Delta T}{2}(v^i \cos\theta^i + v^{i+1}\cos\theta^{i+1}) \\ y^i + \frac{\Delta T}{2}(v^i \sin\theta^i + v^{i+1}\sin\theta^{i+1}) \\ \theta^i + \Delta T \kappa^i v^i + \frac{\Delta T^2}{2}\kappa^i a^i \\ v^i + \Delta T a^i \end{pmatrix}$$

变量解释：
- $\Delta T = 0.1$s (10 Hz)
- $v^i$：第 $i$ 个 timestep 的速度
- $\theta^i$：yaw angle
- $\kappa^i$：曲率（1/转弯半径）
- $a^i$：加速度

**Intuition**：raw $(x,y)$ 是积分后的结果，noise 会被放大；直接学 control $(a, \kappa)$ 相当于在 derivative space 学习，天然 smooth，且物理上保证 kinematic feasibility。训练时用 Tikhonov 正则化的 least-squares 从 GT $\tau$ 反推 control sequence $\mathbf{a}$ 作为监督。这借鉴了 Lynch & Park 的 Modern Robotics 教材里的 unicycle 模型。

### 2.3 关键设计 2：Dual Representation for Trajectory

训练时用 **discrete tokens**（128 个 token：64 waypoints × 2 values/waypoint），推理时用 **flow matching continuous decoder**。

为什么不直接用 discrete tokens 推理？
- Autoregressive decode 128 个 token 慢（Table 14: 222ms）
- 缺少 multi-modality（多模态分布被硬量化）
- 缺少 geometric/kinematic 约束

为什么不直接用 continuous training？
- 失去 unified token space，reasoning 和 action 不能共享 next-token prediction 框架
- RL post-training 难做（policy gradient 需要离散 action space 上的概率分布）

**Dual representation 拿到两边的好处**：training 时 discrete tokens 让 reasoning-action 紧耦合在统一语言空间里学习，且 RL 可以直接用 GRPO 在 token 上做 policy gradient；inference 时 flow matching expert 把 KV-cache 作为 condition，生成连续 multi-modal trajectories，速度快（5 步 Euler integration 即可，Table 14: 8.75ms）。

## 3. Vision Encoding：Token Budget 的工程艺术

这是 VLA 落地的最大瓶颈。自动驾驶车一般 6-10 个相机，每个相机多帧，如果用 naive ViT patch tokenization，token 数量爆炸。

### 3.1 Single-Image Tokenization（default）
对 $W \times H$ 图像，先用 ViT 产生 $W/14 \times H/14 \times D$ patch features，再 2× bilinear downsample 到 $W/28 \times H/28 \times D$。

举例：$W=448, H=280$ → 160 tokens/image。

### 3.2 Multi-Camera Tokenization (Triplane)
基于 Ivanovic et al. 2025 的工作，用 **triplane** 作为 3D inductive bias，把多个相机图像编码到固定的 triplane 表示，再 patchify。

Token 数量公式（Eq. 4）：

$$\left(\frac{S_x - p_x}{p_x} + 1\right)\left(\frac{S_y - p_y}{p_y} + 1\right) + \left(\frac{S_x - p_x}{p_x} + 1\right)\left(\frac{S_z - p_z}{p_z} + 1\right) + \left(\frac{S_y - p_y}{p_y} + 1\right)\left(\frac{S_z - p_z}{p_z} + 1\right)$$

变量：
- $S_x, S_y, S_z$：triplane 三个平面的 grid size
- $p_x, p_y, p_z$：三个方向的 patch size
- 三项分别对应 $xy$、$xz$、$yz$ 三个 plane 的 patch 数

举例：$S_x = S_y = 96, S_z = 48$，$p_x = p_y = p_z = 8$ → 288 tokens/timestep，**与相机数量和分辨率 decoupled**！

7-camera setup → 约 41.1 tokens/image，比 single-image 节省 3.9×。

### 3.3 Multi-Camera Video Tokenization (Flex)
Yang et al. 2025 的 Flex 方法，用 full self-attention + 固定 query vectors 把多相机多 timestep 压到固定 bottleneck。可达 **20× compression**。

Table 13 给出了对比：
- Baseline: 160 tokens/image
- Triplane: 45 tokens/image (3.6×), minADE 退化 -4%
- Flex: 8 tokens/image (20×), minADE 退化 -2%

**Intuition**：camera 越多、history 越长，压缩比收益越大。论文最终选 single-image 作 default，因为它在 AR1 当前的 2-camera (front wide + telephoto) setup 下足够。

## 4. CoC Dataset：Reasoning 的"骨架"

这是论文最核心的数据贡献。Fig. 3 展示了 5 步 pipeline。

### 4.1 三类常见 reasoning 病（Fig. 2）
1. **Vague behavior descriptions**：例如 "ego vehicle should be cautious" → 没有具体 action
2. **Superficial reasoning**：例如 "sunny weather, wide roads" → causal 因素与 action 无关
3. **Causal confusion**：reasoning 提到未来发生的事（因为标注者看了完整 video）

### 4.2 CoC 的三大设计原则
1. **Decision grounding**：每条 reasoning trace 必须锚定到一个 closed-set driving decision
2. **Causal locality**：所有 causal 因素必须来自 observable history window（2s 内）
3. **Annotation economy**：只标 decision-relevant 因素

### 4.3 Closed-set Driving Decisions (Table 1)
分 longitudinal 和 lateral 两个 channel：
- **Longitudinal (7 类)**：Set speed tracking, Lead obstacle following, Speed adaptation (road events), Gap-searching, Acceleration for passing, Yield, Stop for static constraints
- **Lateral (8 类)**：Lane keeping & centering, Merge/Split, Out-of-lane nudge, In-lane nudge, Lane change, Pull-over, Turn, Lateral maneuver abort

每条样本最多标 1 个 longitudinal + 1 个 lateral decision（或 None）。这是**离散化 decision 空间**，让 reasoning 有"骨架"。

### 4.4 Critical Components (Table 2)
开放类别，包括：Critical objects (veh/ped/cyclist), Traffic lights, Yield/Stop control, Road events, Lane/lanelines, Routing intent, ODD constraints。每个都标 Low/High uncertainty。

### 4.5 Hybrid Labeling Pipeline

**Human Labeling（高质量小规模，~10%）**：
- Stage I (0-2s)：只看 history，标 critical components
- Stage II (0-8s)：看 future 选 driving decision，写 reasoning trace，只能 reference Stage I 标的 causal factors

关键工程细节：**标注工具显式分离 history 和 future video segments**，这是防 causal leakage 的硬约束。

**Auto-labeling（大规模）**：
- 用 GPT-5 做 offline 标注
- 先用 rule-based detectors 检测 atomic meta-actions (Table 5) 找 keyframe
- 给 VLM 提供 2s history + 6s future + ego trajectory + meta actions
- 让 VLM rank causal factors，保留 decision-relevant 的

### 4.6 Evaluation
用 GPT-5 做 auto-evaluator，把 free-form grading 拆成 True/False 子问题（driving decision、causal factors presence、cause-effect validity），与 human evaluation 对齐率 92%。

CoC 相对 free-form reasoning：causal relationship score 提升 132.8%。

## 5. Multi-stage Training：从 VLM 到 Reasoning VLA

### Stage 1: Action Modality Injection (Sec. 5.1)

把 VLM 扩展为 VLA。在 token sequence 上做 next-token prediction，cross-entropy loss。

**Flow Matching Action Expert**：

类似 $\pi_{0.5}$-KI (Driess et al. 2025)，单独的 action expert（同构 Transformer，更小 hidden dim），输入：
- VLM 的 KV-cache（来自 $[o_{\text{image}}, o_{\text{egomotion}}, \text{REASON}]$）
- Noisy control $\mathbf{a}_t$ 的 embedding + diffusion time $t$ 的 embedding

输出 vector field $\mathbf{v}_\Theta(\mathbf{a}_t, o, \text{REASON})$。

Conditional Flow Matching loss (Eq. 6)：

$$L_{\text{cfm}}(\Theta) = \mathbb{E}_{t \sim p_{\text{schedule}}, (o, \text{REASON}) \sim \mathcal{D}_{\text{data}}} \|\mathbf{v}_\Theta(\mathbf{a}_t, o, \text{REASON}) - \mathbf{u}(\mathbf{a}_t | \mathbf{a})\|$$

采用 Gaussian conditional OT path：$\mathbf{a}_t = t\mathbf{a} + (1-t)\epsilon$，$\epsilon \sim \mathcal{N}(0, I)$。

Target vector field closed form (Eq. 7)：

$$\mathbf{u}(\mathbf{a}_t | \mathbf{a}) = \mathbf{a} - \epsilon$$

Inference：Euler integration (Eq. 8)：

$$\mathbf{a}_{t+\delta_t} = \mathbf{a}_t + \delta_t \mathbf{v}_\Theta(\mathbf{a}_t, o, \text{REASON})$$

默认 $\delta_t = 0.1$，$p_{\text{schedule}}$ 是 shifted beta distribution（Physical Intelligence et al. 2025 的建议）。

**关键 trick**：训练时对 VLM 的 KV-cache 施加 stop-gradient，防止 action expert 的梯度回传污染 VLM 权重。这是典型的"freeze backbone, train head"思路，但保留了 condition 的语义信息。

### Stage 2: Eliciting Reasoning via SFT (Sec. 5.2)

在 CoC 数据上 SFT，loss (Eq. 9)：

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(o, \text{REASON}, a) \sim \mathcal{D}_{\text{CoC}}} [\log \pi_\theta(\text{REASON}, a | o)]$$

联合优化 reasoning tokens 和 128 个 discrete trajectory tokens。

**SFT 的局限性**（论文 Sec. 5.2 列了 4 条）：
1. Data bias：auto-label 噪声会被 model memorize
2. Limited generalization：pattern matching 而非真因果理解
3. Weak visual grounding：next-token prediction 不强制 visual consistency，可能 hallucinate 不存在的因素
4. Reasoning-action inconsistency：joint optimization 不强制 reasoning 和 trajectory 对齐

### Stage 3: RL Post-Training with GRPO (Sec. 5.3)

这是论文最有意思的部分。用 GRPO (Group Relative Policy Optimization, Shao et al. 2024) 做 post-training。

GRPO loss (Eq. 10)：

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{\tau_i \sim \pi_\theta}\left[\frac{\exp(\beta A_i)}{\sum_j \exp(\beta A_j)}\left(\log \pi_\theta(\tau_i) - \lambda_{\text{KL}} \text{KL}[\pi_\theta(\tau_i) || \pi_{\text{ref}}(\tau_i)]\right)\right]$$

$$A_i = r_i - \bar{r}$$

变量：
- $\tau_i$：第 $i$ 个 rollout（reasoning + action 序列）
- $A_i$：relative advantage（group 内相对优势）
- $\bar{r}$：group 平均 reward
- $\beta$：temperature，控制 advantage 权重分布 sharpness
- $\lambda_{\text{KL}}$：KL 正则系数
- $\pi_{\text{ref}}$：reference policy（SFT 模型），防止 over-optimization

### 5.3.2 Reward 设计：三组件

这是 RL 的灵魂。三个 complementary reward：

**Reward 1: Reasoning Quality via LRM Feedback**

用 large reasoning model（DeepSeek-R1 或 Cosmos-Reason）作 critic。输入：last frame of 2s history + GT CoC trace + 模型生成的 reasoning trace。

评分 rubric（0-5）：
- 5: Behavior & causal reasoning fully consistent
- 4: Behavior correct, causal reasoning mostly consistent
- 3: Behavior roughly correct, but incomplete or slightly incorrect reasoning
- 2: Behavior partially incorrect or reasoning largely inconsistent
- 1: Behavior is wrong or contradicts GT
- 0: Completely unrelated or opposite

**Intuition**：LRM 虽然自己生成 driving reasoning 可能不行（缺 embodiment prior），但 verification 能力强（generation-verification gap, Song et al. 2025）。这是一个 LLM-as-judge 的应用，但通过 structured rubric 缓解 hallucination。

**Reward 2: Reasoning-Action Consistency**

把预测 trajectory 转成 meta-action sequence（Sec. 4.3.1 定义的 atomic actions：Gentle/Strong accelerate, Gentle/Strong decelerate, Maintain speed, Stop, Reverse, Steer left/right, Sharp steer, Reverse steer, Go straight）。

规则匹配 reasoning trace 中描述的 intent 与 trajectory 导出的 meta-actions：

$$r_{\text{consistency}} = \begin{cases} 1 & \text{if consistent across both axes} \\ 0 & \text{otherwise} \end{cases}$$

无法 parse 的 reasoning → $r_{\text{consistency}} = 0$（conservative）。

**Intuition**：binary reward 简单但有效。强制 reasoning 不能只是"听起来对"，必须翻译成物理一致的行为。这是论文最关键的对齐机制。

**Reward 3: Trajectory Quality** (Eq. 11)

$$r_{\text{traj}} = \lambda_{\text{L2}} \|x_{\text{pred}} - x_{\text{expert}}\|_2^2 + \lambda_{\text{coll}} \mathbb{I}[\text{collision}(x_{\text{pred}})] + \lambda_{\text{jerk}} J(x_{\text{pred}})$$

变量：
- $x_{\text{pred}}, x_{\text{expert}}$：预测和专家轨迹
- $\mathbb{I}[\text{collision}]$：碰撞 binary indicator
- $J(x_{\text{pred}})$：jerk magnitude（舒适度）
- $\lambda_{\text{L2}}, \lambda_{\text{coll}}, \lambda_{\text{jerk}}$：三项权重

L2 项稳学习，collision 保安全，jerk 保舒适。

### 5.3.3 RL 数据 Curvature：High-Disagreement Sampling

RL 计算贵，不能全数据跑。论文提出 **implicit reward vs explicit reward disagreement** 来选数据。

- Implicit reward：模型 logits 推出的概率分布
- Explicit reward：把 reward $r_i$ 转 Boltzmann 分布 $p_{\text{reward}}(\tau_i) = \frac{\exp(\beta r_i)}{\sum_j \exp(\beta r_j)}$

两者 KL divergence 大 → 模型内部偏好与外部 reward 冲突 → 高信息量样本。

混合 high-disagreement 样本 + 随机样本（保 distribution diversity）。

## 6. 实验数据深度解读

### 6.1 Open-loop (Table 6, 7)

**Nominal scenarios (Table 6)**，无 route 信息：
- Base model (action only, 0.5B): 0.996m @6.4s
- + Ft. w/ Traj.: 0.971m
- + Ft. w/ Meta-action & Traj.: 0.988m（反而比 traj-only 差）
- **+ Ft. w/ CoC & Traj. (AR1)**: **0.955m**（4.1% improvement over base）

有意思的是 **Meta-action & Traj. 比 Traj.-only 还差**！这说明 free-form 的 meta-action 描述（如"减速"）反而引入 noise，没有 decision-grounded 的 CoC 才有助益。

有 route 信息时 AR1 0.5B: 0.794m，比 traj-only 0.834m 提升 4.8%。

3B 模型：AR1-3B 0.908m（无 route），scaling 收益明显。

**Challenging scenarios (Table 7)**：
- Traj. only: 0.994m
- Meta-action & Traj.: 0.928m
- **CoC & Traj. (AR1)**: **0.868m**（12% improvement！）

这就是论文标题"12% improvement in planning accuracy on challenging cases"的来源。长尾场景收益最大，这正是 reasoning 的价值所在。

### 6.2 Closed-loop AlpaSim (Table 8)

75 个挑战性 20s 场景，无 route 信息：
- Baseline: Close Encounter Rate 17%, Off-Road 3%, AlpaSim Score 0.38
- **AR1**: Close Encounter 11% (**35% reduction**), Off-Road 4%, AlpaSim Score 0.50

Off-Road 略升（3→4%）但 Close Encounter 大幅下降，整体 safety 提升。

### 6.3 RL Post-Training 效果 (Table 9)

最 revealing 的实验。0.5B AR1：

| Training | ADE↓ | Reasoning Grade↑ | Consistency↑ | Close Encounter↓ |
|---|---|---|---|---|
| SFT only | 2.12m | 3.1 | 0.62 | 6.9% |
| SFT + RL(r_reason) | 2.19m | **4.5** | 0.53 | 5.8% |
| SFT + RL(r_reason + r_consistency) | 1.92m | 4.5 | **0.85** | 6.2% |
| SFT + RL(all three) | 1.94m | 4.4 | 0.83 | **3.7%** |

**关键观察**：
1. 只优化 reasoning reward → reasoning 分数涨 (3.1→4.5)，但 ADE 退步 (2.12→2.19)，consistency 退步 (0.62→0.53)。模型学会"说漂亮话"但 action 不一致。
2. 加 consistency reward → ADE 涨 9.4% (2.12→1.92)，consistency 涨 37% (0.62→0.85)，reasoning 保持 4.5。**这是论文 abstract 里 "37% consistency improvement" 的来源**。
3. 加 safety reward → close encounter 暴跌 (6.2→3.7%)，ADE 略退 (1.92→1.94) 但 safety 大涨。

**这是论文最有 teaching 价值的发现**：reasoning quality 和 action quality 不是一回事，必须用 consistency reward 显式对齐。

### 6.4 Public Benchmark (Table 10)

PhysicalAI-AV dataset + AlpaSim public 920 scenarios：
- AR1-0.5B: minADE 0.913m, Close Encounter 9%, AlpaSim 0.35
- **AR1-10B**: minADE 0.849m (7% better), Close Encounter 4% (55% reduction), AlpaSim 0.72 (2× better)

Scaling 0.5B → 10B 收益巨大，尤其在 closed-loop。

### 6.5 Backbone Ablation (Fig. 12, Table 11)

General-purpose VLMs scaling (0.5B→3B→7B) → 11% minADE 改善。

Cosmos-Reason-7B vs 同规模 general VLMs on LingoQA：
- GPT-4V: 59.6%
- Qwen2.5-VL-7B: 62.2%
- **Cosmos-Reason-7B: 66.2%**

Physical AI pretraining 在 driving scene understanding 上独立贡献显著。

### 6.6 Trajectory Decoding Ablation (Table 12)

Auto-regressive vs Flow Matching（同样模型大小、同样数据）：
- AR: minADE 0.6811, AlpaSim 0.59, Comfort 44.05%, Speed 1.0×
- **FM**: minADE 0.6440 (5% better), AlpaSim 1.27 (2× better), Comfort 97.38%, Speed 1.16×

Flow matching 在 closed-loop 和 comfort 上完全碾压，因为 continuous multi-modal 输出天然 smooth。

### 6.7 Latency Breakdown (Table 14)

NVIDIA RTX 6000 Pro Blackwell：
- Baseline (traj only, FM): 29ms total
- **AR1 (FM)**: 99ms total (vision 3.43 + prefill 16.54 + reasoning 70 + traj decode 8.75)
- AR1 (auto-regressive traj): 312ms total

Reasoning decoding 70ms 是大头（40 tokens），但通过 flow matching 把 traj decode 从 222ms 压到 8.75ms，达到 99ms 实时阈值。

## 7. On-Vehicle Road Test (Fig. 14)

实际部署到测试车，无人工干预完成 urban 场景。Fig. 14 展示了一个 intersection 场景：
1. 识别红灯 → 减速停止
2. 等待信号
3. 绿灯后加速通过

推理 trace：先描述 "Decelerating to stop at the red light" → "Ego will wait at the intersection until the traffic light turn green" → "Accelerating straight through the intersection after the light turns green"。

这是 reasoning trace 真实驱动 action 的 demonstration。

## 8. 对 Karpathy 直觉的 Build

我猜你会关心的几个点：

### 8.1 Reasoning 真的有用吗？
Table 6 的 Meta-action & Traj. baseline 比 Traj.-only 还差，说明**自由形式的语言监督是 noise**。只有 decision-grounded CoC 才有助益。CoC 的关键不是"有 reasoning"，而是 reasoning 的 **结构**：closed-set decisions + observable causal factors。

### 8.2 RL 阶段的 consistency reward 为什么如此关键？
Table 9 显示只优化 reasoning reward 会让模型产生 "fluent but causally disconnected explanations"——这正是 SFT 模型的 failure mode。Consistency reward 是把 reasoning 锚定到 physical behavior 的硬约束。这本质上是一个 **modality alignment** 问题，和 RLHF 中 reward hacking 类似。

### 8.3 Unicycle dynamics + Flow Matching 的协同
两者缺一不可。Unicycle 让 action space 是物理可行的 control；Flow matching 让 inference 时 multi-modal continuous distribution 可采样。一起实现了 "physical AI" 的核心需求。

### 8.4 Token Budget 是 VLA 落地的关键
Triplane 和 Flex 这类 efficient encoder 不是"压缩"，是 **decoupling**：让 token 数与 camera 数/resolution 解耦。这是把 LLM 推到 real-time AV 的核心工程。

### 8.5 World Model 缺位
论文 Future Work 提到："incorporating learned world models could enable forward simulation and counterfactual reasoning"。这是当前 AR1 的局限——它做的是 reactive policy，没有 forward simulation。Poutine (Rowe et al. 2025) 在 Waymo challenge 上做了类似的事，可以参考。链接：https://arxiv.org/abs/2506.11234

## 9. 总结性直觉

AR1 的故事可以浓缩成一句话：**Reasoning 只有在 structured (CoC) + aligned (consistency reward) + physically grounded (unicycle + flow matching) 时才真正 help action**。

论文的 contribution 不是"加了 reasoning"，而是给出了 reasoning→action 这条链路完整的工程闭环：数据结构化（CoC dataset）、训练分阶段（SFT 引入 → RL 对齐）、推理模块化（discrete training + continuous decoding）、评估分层（open-loop + closed-loop + on-vehicle）。

如果用你常用的语言：这是一个 "Software 2.0 meets Software 3.0" 的实例——next-token prediction 学 reasoning，flow matching 学 continuous control，RL 把两者 glue 起来。

## 10. 可能的延伸阅读

- Cosmos-Reason: https://arxiv.org/abs/2503.15558
- $\pi_{0.5}$-KI (Driess et al. 2025): https://arxiv.org/abs/2505.23705
- Flow Matching (Lipman et al. 2023): https://arxiv.org/abs/2210.02747
- GRPO (DeepSeekMath, Shao et al. 2024): https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Triplane tokenizer (Ivanovic et al. 2025): https://arxiv.org/abs/2503.18430
- Flex (Yang et al. 2025): https://arxiv.org/abs/2512.10947
- Poutine (Waymo challenge top): https://arxiv.org/abs/2506.11234
- AlpaSim: https://github.com/NVlabs/alpasim
- PhysicalAI-AV dataset: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles

如果还想深入某个细节，比如 flow matching 的 schedule 选择、GRPO 中 $\beta$ 的影响、CoC auto-labeling 的具体 prompt 设计、或者 AlpaSim 的 neural reconstruction 细节，告诉我，我可以再展开讲。
