---
source_pdf: RAD.pdf
paper_sha256: 49ceef4c6f3c6dad72213302d443b1a733bb92b61b443b760c9095c430ba3b6d
processed_at: '2026-08-11T20:45:23-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 RAD

## 一句话总结

**让自动驾驶 policy 在一个"以假乱真的虚拟世界"里自己开车、自己撞、自己学，撞了就知道疼——但还得时不时对照老司机的开法，别学野了。**

---

## 1. 为什么之前的 IL 方法不行

想象一个驾校学员，学习方法是"看老师开车看了一万小时"。

问题在哪？

**问题一：他学的是"动作"，不是"道理"。**

老师看到红灯就踩刹车，学员也学会了"看到这个画面就踩刹车"。但他不知道**为什么**——可能是因为前面有行人，可能是因为红灯，可能只是老师想靠边停车。他只记住了"画面 A → 动作 B"的 correlation，没理解 causation。

这就是 paper 里说的 **causal confusion** 和 **shortcut learning**。很多 IL policy 根本不看路，只看自己的历史轨迹做 extrapolation——因为它发现"历史轨迹"这个 signal 就能预测下一步，何必去看红绿灯？参考 [Is Ego Status All You Need](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Is_Ego_Status_All_You_Need_for_Open-Loop_End-to-End_Autonomous_CVPR_2024_paper.pdf)。

**问题二：训练和考试不一样。**

训练时每一步都是老师开到的好位置，学员只需要"在好位置上预测下一步"。
考试时没人帮忙，学员第一步偏了 10cm，第二步就在偏的位置上预测，又偏一点，第三步更偏……雪球越滚越大，最后开到沟里。

这就是 **open-loop gap**——open-loop training，closed-loop deployment，distribution shift 导致 compounding error。

---

## 2. RAD 的解法：让 policy 自己开、自己撞

核心 insight 很简单：**不让学员光看，让他自己开，撞了就知道疼。**

这就是 Reinforcement Learning——policy 自己做决策，environment 给 reward / penalty，policy 通过 trial-and-error 学习。

但 RL 需要 environment。用什么 environment？

**不能用真实世界**——撞真车太贵太危险。
**不能用 CARLA 这种 game engine**——渲染出来的 camera 图像太假，policy 在 CARLA 学到的 perception 能力无法迁移到真实世界（real world 的 texture、lighting、noise 完全不一样）。

RAD 的答案：**用 3DGS 把真实世界重建出来，做成 photorealistic digital twin。**

3DGS 是 2023 年的 breakthrough 技术（参考 [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)），能把一段真实 driving video 重建成一个 3D 场景，可以从任意视角渲染出 photo-realistic 的新画面，速度还快（>30 FPS）。

所以 RAD 的做法是：
1. 收集 4305 个真实 driving clip
2. 每个 clip 用 3DGS 重建成一个"虚拟场景"
3. 让 policy 在这些虚拟场景里 closed-loop 自己开
4. 撞了给负 reward，偏了给负 reward
5. policy 通过 PPO 优化，学会避撞、保持轨迹

**关键：因为 3DGS 渲染出来的图像和真实世界几乎一样（Figure 5 验证了 consistency），policy 在 3DGS 里学到的 perception 能力直接迁移到真实世界。**这是第一次让 "在仿真里学开车能直接用到真车上" 成为可能。

---

## 3. 核心设计：把"方向盘"和"油门刹车"分开

这是我觉得 RAD 最聪明的设计。

传统 AD 的 action space 是 2D continuous（steering + speed），或者把整个轨迹当 action。RL 在这种连续高维空间里很难学。

RAD 把 action 拆成两个**独立的离散维度**：

- **Lateral action** $a^x$：0.5秒内左右移动多少。61 个选项，从 -0.75m 到 +0.75m。
- **Longitudinal action** $a^y$：0.5秒内前进多少。61 个选项，从 0 到 15m。

Policy 输出两个独立的 softmax distribution（一个 61-way，一个 61-way），sample 出来组合成 action。

**为什么这样设计？三个原因：**

**原因一：维度灾难。**
如果 joint discrete，就是 $61 \times 61 = 3721$ 个 action，data 稀疏，学不动。Decouple 之后每个维度只有 61 个，sample efficiency 大幅提升。

**原因二：因果解耦。**
现实中，**横向**（方向盘）主要导致静态碰撞和轨迹偏离——撞马路牙子、压线、漂移。
**纵向**（油门刹车）主要导致动态碰撞——追尾、被追尾。
两个维度的因果关系是独立的，分开学 gradient signal 更干净。

**原因三：可以注入 domain knowledge。**
后面 PPO 的 clipping range，lateral 设 0.1（保守更新，方向盘稍微动一点轨迹就偏很多），longitudinal 设 0.2（激进更新，速度相对 robust）。这种 domain-specific hyperparameter 只有 decoupled 才能做。

---

## 4. Reward 设计：四种"扣分项"

RAD 不用复杂 reward，就四种简单 penalty：

| Reward | 什么时候触发 | 归到哪个维度 |
|--------|-----------|-----------|
| $r_{dc}$ | 撞到动态障碍物（车、人） | Longitudinal |
| $r_{sc}$ | 撞到静态障碍物（路沿、护栏） | Lateral |
| $r_{pd}$ | 偏离 expert 轨迹超过 2m | Lateral |
| $r_{hd}$ | 朝向偏离 expert 朝向超过 40° | Lateral |

**关键设计：任何一个 trigger 都立即终止 episode。**

为什么？因为 ego vehicle 一旦撞了或偏了，3DGS 渲染出来的画面就是"没观察过的视角"，画面质量下降，sensor noise 增大，继续 rollout 只会给 policy 喂 garbage data。

**Ablation 的 key insight**（Table 2）：
- 去掉 dynamic collision reward（ID 2）→ CR 0.238，最高
- 去掉其他任一 reward → CR 0.15-0.17
- 四个全开 → CR 0.089

**Dynamic collision reward 是最关键的**，因为它直接教 policy 避让 moving obstacle。这正是 IL 的 weakness——IL data 里 collision 极少（谁会故意录撞车数据），policy 对动态障碍物不敏感。

---

## 5. 解决 RL 的两大痛点

### 痛点一：RL 学野了怎么办（Human Alignment Problem）

RL 只管"不撞"，不管"开得像人"。policy 可能学到 weird behavior——比如为了避撞，方向盘疯狂抖动，或者为了安全龟速行驶。

RAD 的解法：**IL 当 regularization。**

训练时 RL 和 IL 交替，比例 4:1——4 轮 RL 加 1 轮 IL。IL step 用 expert demonstration 做 supervised update，把 policy 拉回 human distribution。

这就像学员自己练车（RL），但每隔一段时间教练带一次（IL），纠正野路子。

对比 LLM 里 RLHF 的 KL penalty——RAD 用 explicit IL step 而非 KL term。好处是 IL 用 real expert data，signal 更强；坏处是 batch update 不如 KL 的 per-step smoothness。参考 [RLHF](https://arxiv.org/abs/2203.02155)。

### 痛点二：Reward 太稀疏（Sparse Reward Problem）

大部分 step 都没撞、没偏，reward = 0，policy gradient 信号弱。

RAD 的解法两层：

**第一层：GAE（Generalized Advantage Estimation）**

把 future reward 传播回前面 step。撞车发生在第 80 步，但前面 79 步的 action 都有责任。GAE 用 discount factor $\gamma = 0.9$ 和 tradeoff parameter $\lambda = 0.95$ 把 future reward 往前传，让每一步都得到 credit。参考 [GAE](https://arxiv.org/abs/1506.02438)。

**第二层：Auxiliary Objectives（dense supervision）**

这是 RAD 的另一大创新。即使没撞车，只要前方有车，就给 policy 一个 directional gradient——"把概率质量从加速移到减速"。

具体做法：把 action distribution 按方向拆分——
- 减速概率 $\Delta\pi^{dec}_y$ vs 加速概率 $\Delta\pi^{acc}_y$
- 左转概率 $\Delta\pi^{left}_x$ vs 右转概率 $\Delta\pi^{right}_x$

然后根据 risk 的相对位置，鼓励"正确方向"：
- 前方有车 → 鼓励减速：$\mathcal{L}_{dc} \propto \hat{A}^{dc}_t \cdot (\Delta\pi^{dec}_y - \Delta\pi^{acc}_y)$
- 左边有障碍 → 鼓励右转：$\mathcal{L}_{sc} \propto \hat{A}^{sc}_t \cdot (\Delta\pi^{right}_x - \Delta\pi^{left}_x)$
- 偏左了 → 鼓励右转：$\mathcal{L}_{pd} \propto \hat{A}^{pd}_t \cdot (\Delta\pi^{right}_x - \Delta\pi^{left}_x)$
- 朝向偏了 → 鼓励反向修正：$\mathcal{L}_{hd}$

**Intuition：纯 PPO 只有撞了才有信号，auxiliary objective 让每一步都有 "往安全方向调" 的 gradient。**这比纯 sparse reward 的 sample efficiency 高得多。

Ablation（Table 3）证实：去掉所有 auxiliary → CR 0.249；全开 → CR 0.089。差距巨大。

---

## 6. 三阶段训练：先学看，再学开，最后自学成才

**Stage 1: Perception Pre-Training**
训练 BEV encoder + map head + agent head，用 ground-truth 标注监督。让 token 学会编码"车道线在哪、车在哪、往哪开"。

**Stage 2: Planning Pre-Training (IL)**
冻结 perception 模块，只训练 image encoder + planning head，用 expert demonstration 做 supervised learning。把 action distribution 初始化到"像人"的状态，避免 RL cold start 不稳定。

**Stage 3: Reinforced Post-Training (RL + IL 交替)**
32 个 parallel worker，每个随机选一个 3DGS environment rollout，存数据到 shared buffer。4 轮 RL (PPO) + 1 轮 IL 交替训练。

**为什么冻结 perception？** perception 和 planning 的 gradient 可能冲突（perception 要 local feature，planning 要 global decision），分阶段 + freeze 避免 conflict。

---

## 7. 结果：撞车率降 3 倍

主结果（Table 4）：

| Method | CR↓ | DCR↓ | DR↓ | ADD↓ |
|--------|-----|------|-----|------|
| VADv2 (IL SOTA) | 0.270 | 0.240 | 0.243 | 0.273 |
| **RAD** | **0.089** | **0.080** | **0.063** | 0.257 |

- **CR（撞车率）从 0.270 降到 0.089，3× 降低**
- ADD（偏离距离）0.257，和 IL 方法持平——安全但不牺牲 human-likeness
- Lateral Jerk 0.082 最低——trajectory 最平滑

最有意思的 ablation（Table 1）：

| Strategy | CR↓ | ADD↓ |
|----------|-----|------|
| IL only | 0.229 | 0.238 |
| RL only | 0.143 | 0.345 |
| **RL + IL** | **0.089** | 0.257 |

- IL only：像人但会撞（ADD 0.238 但 CR 0.229）
- RL only：安全但不像人（CR 0.143 但 ADD 0.345）
- RL + IL：安全且像人（CR 0.089，ADD 0.257）

**这完美印证核心 thesis：RL 学 causation（避撞），IL 保 alignment（像人），两者缺一不可。**

---

## 8. 我的几点思考

**3DGS + RL 是真正的 paradigm shift。** 之前 AD RL 卡在"simulator 不真实"，3DGS 突破了这个 bottleneck。这条路一旦走通，scaling up 就是工程问题——更多 scene、更长 clip、更复杂 NPC。

**Decoupling 是通用设计哲学。** Lateral / longitudinal 解耦、reward 解耦、value function 解耦、PPO clipping 解耦……每个维度的 gradient signal 都 clean，还能注入 domain knowledge。这种 systematic decoupling 思路值得其他 RL 场景借鉴。

**Auxiliary objective 是 dense supervision 的关键。** RL 的 sparse reward 问题无处不在，RAD 这种"按风险方向 shape action distribution"的做法很实用——比纯 reward shaping 更结构化，比纯 imitation 更灵活。

**和 LLM RL 的类比。** RAD 的 RL + IL 范式和 DeepSeek-R1 / OpenAI O1 的 RL + SFT 高度类似。AD 里是 safety + human-likeness，LLM 里是 reasoning + helpfulness。这暗示一个通用 pattern：**RL 学 capability，IL/SFT 保 alignment，两者交替训练**。参考 [DeepSeek-R1](https://arxiv.org/abs/2501.12948)。

**Limitations。** Log-replay NPC 不会 react to ego 的 OOD behavior（真实世界前车会避让），可能让 policy 学太激进。3DGS 重建 cost 高（4305 个 scene 各训一个 model）。Reward 仍 hand-crafted，未来可能用 learned reward（RLHF for AD）。

---

## 相关链接

- [RAD GitHub](https://github.com/hustvl/RAD)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Street Gaussians](https://arxiv.org/abs/2406.15456)
- [HUGSim](https://arxiv.org/abs/2412.01718)
- [PPO](https://arxiv.org/abs/1707.06347)
- [GAE](https://arxiv.org/abs/1506.02438)
- [VADv2](https://arxiv.org/abs/2402.13243)
- [DiffusionDrive](https://github.com/hustvl/DiffusionDrive)
- [CARLA Simulator](https://carla.org/)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [RLHF](https://arxiv.org/abs/2203.02155)

---

**一句话 take-away**：RAD 证明了一件事——**让自动驾驶 policy 在 photorealistic 虚拟世界里自己撞自己学，同时用老司机数据当锚防止学野，撞车率能降 3 倍。**这是 end-to-end AD 从 imitation 到 reinforcement 的 paradigm shift，3DGS 是使这一切成为可能的关键 enabler。

---

# RAD: 3DGS-based Reinforcement Learning for End-to-End Autonomous Driving 深度技术讲解

## 1. 核心动机：为什么 IL 不够，为什么需要 RL + 3DGS

现有 end-to-end AD 几乎全部基于 Imitation Learning，但 IL 有两个根本性问题：

**Causal Confusion**: IL 学到的是 correlation，不是 causation。policy 容易走 shortcut，比如直接从 historical trajectory extrapolate future trajectory，而不真正理解 traffic light / pedestrian / lead vehicle 这些 causal factors。参考 [Shortcut Learning in Deep Neural Networks](https://www.nature.com/articles/s42256-020-00257-z) 和 [Is Ego Status All You Need](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Is_Ego_Status_All_You_Need_for_Open-Loop_End-to-End_Autonomous_CVPR_2024_paper.pdf)。

**Open-loop Gap**: training 是 open-loop（每步用 expert state），deployment 是 closed-loop（每步用自己预测的 state）。一步小误差 → compounding error → OOD scenario → policy 崩溃。这是 distribution shift 的极端形式。

RAD 的核心 insight：**用 closed-loop RL 让 policy 在 photorealistic 3DGS environment 里大规模 trial-and-error，从自己的错误中学习 causation**。同时用 IL 作为 regularization，保持 human alignment，避免 RL 探索出 weird policy（比如 weird oscillation 来规避 collision）。

---

## 2. 整体架构解析

### 2.1 Policy Network 结构

```
Multi-view Images (T frames)
    │
    ├──► BEV Encoder (BEVFormer-style)  ──► BEV Feature Map
    │         │
    │         ├──► Map Head (MapTRv2-style tokens) ──► Map Tokens E_map
    │         │     (lane centerlines, dividers, boundaries, arrows, signals)
    │         │
    │         └──► Agent Head (PER-style tokens) ──► Agent Tokens E_agent
    │               (location, orientation, size, speed, multi-mode trajectories)
    │
    └──► Image Encoder (ViT / ResNet) ──► Image Tokens E_img (dense low-level info)
    
    E_scene = {E_map, E_agent, E_img}  (instance-level + dense)
              │
              ▼
    Planning Embedding E_plan (1×D, learnable query)
              │
              ▼
    Cascaded Transformer Decoder φ(E_plan, E_scene)  (cross-attention)
              │
              ├──► + E_navi (navigation command)
              ├──► + E_state (ego velocity, acceleration, etc.)
              │
              ▼
    MLP ──► softmax ──► π(a^x | s), π(a^y | s)  (decoupled action distributions)
    MLP ──► V_x(s), V_y(s)  (decoupled value functions)
```

**Intuition**: Map tokens 和 agent tokens 提供 structured high-level information（where to drive, what's around），image tokens 提供 dense low-level cues（road texture, subtle obstacles）。Planning head 用 transformer decoder 把这些 query 出来，加上 navigation + ego state，输出 decoupled action distribution。这种 decoupling 是关键设计——后面会讲为什么。

### 2.2 为什么 Image Encoder 单独一支

Map/Agent tokens 是 instance-level sparse representation，会丢掉一些 dense scene detail。paper 单独用 image encoder 保留 dense information，作为 planning 的 complementary input。这类似于 UniAD/VAD 里用 perception 结果 + 原始 feature 的双重 path。

---

## 3. Action Space 设计（关键创新点）

### 3.1 Decoupled Discrete Action

RAD 把 action space 拆成两个独立维度：

- **Lateral action** $a^x$: 0.5s 内 lateral displacement，$N_x = 61$ 个选项，范围 $[-0.75m, +0.75m]$，uniform sampling
- **Longitudinal action** $a^y$: 0.5s 内 longitudinal displacement，$N_y = 61$ 个选项，范围 $[0, 15m]$，uniform sampling

总 action space size = $61 \times 61 = 3721$，但因为是 decoupled，policy 输出两个独立 softmax distribution（61 + 61 = 122 logits），而不是 3721-way softmax。

**Intuition**: 
1. **维度灾难缓解**: 若 joint discrete，需要 3721-way classification，data sparsity 严重。Decoupling 让每个维度独立学习，sample efficiency 大幅提升。
2. **Causal decoupling**: lateral action 主要 cause 静态 collision + deviation；longitudinal action 主要 cause 动态 collision。后面 reward 设计也对应 decoupled，让 gradient signal 更 clean。
3. **0.5s horizon + constant velocity assumption**: 把 action 转化为 $(v_t, \delta_t)$，喂给 kinematic bicycle model。short horizon 保证 constant velocity 假设合理，且便于 closed-loop rollout。

### 3.2 Kinematic Bicycle Model

给定 action $(a^x_t, a^y_t)$，先 derive $(v_t, \delta_t)$，然后迭代更新 ego pose：

$$
x^w_{t+1} = x^w_t + v_t \cos(\psi^w_t) \Delta t
$$
$$
y^w_{t+1} = y^w_t + v_t \sin(\psi^w_t) \Delta t
$$
$$
\psi^w_{t+1} = \psi^w_t + \frac{v_t}{L} \tan(\delta_t) \Delta t
$$

变量含义：
- $x^w_t, y^w_t$: ego vehicle 在 world coordinate 的 position
- $\psi^w_t$: heading angle（相对 world x-axis）
- $v_t$: linear velocity
- $\delta_t$: front wheel steering angle
- $L$: wheelbase（前后轴距离）
- $\Delta t$: 时间步长

**Intuition**: 这是 standard bicycle model，假设 vehicle 是 rigid body，前后两轮，前轮可 steering。它忽略了 slip、tire dynamics 等，但对 AD policy training 够用。$\frac{v_t}{L} \tan(\delta_t)$ 是 yaw rate，控制 vehicle 转向。

---

## 4. 三阶段训练范式

### Stage 1: Perception Pre-Training
- 用 ground-truth map + agent annotations 监督 BEV encoder + map head + agent head
- 只更新这三个 module，planning head 不动
- 目的：让 tokens encode high-level structured information

### Stage 2: Planning Pre-Training (IL)
- 冻结 BEV/map/agent，只训练 image encoder + planning head
- 用 expert demonstrations 做 supervised learning
- Action label 通过 nearest-neighbor matching 到 anchor grid：

$$
\hat{i} = \arg\min_i \left\| \frac{a^x_i - d^x_{\min}}{d^x_{\max} - d^x_{\min}} - \frac{p^x_{gt} - d^x_{\min}}{d^x_{\max} - d^x_{\min}} \right\|_2
$$

类似地对 $\hat{j}$。然后 dual focal loss：

$$
\mathcal{L}_{IL} = \mathcal{L}_{focal}(\pi(a^x|s), \hat{i}_t) + \mathcal{L}_{focal}(\pi(a^y|s), \hat{j}_t)
$$

**Intuition**: normalized nearest-neighbor 是把连续 ground-truth position 量化到 discrete anchor，focal loss 处理 class imbalance（大部分时间 ego 是 straight driving，对应 anchor 集中在 0 附近）。参考 [Focal Loss](https://arxiv.org/abs/1708.02002)。

### Stage 3: Reinforced Post-Training (RL + IL 交替)
- 部署 N=32 parallel workers，每个随机 sample 3DGS environment
- Rollout 8s clip（10Hz → 80 steps），存 $(s_t, a_t, r_{t+1}, s_{t+1}, ...)$ 到 shared buffer
- 交替训练：4 rounds RL (PPO) + 1 round IL，比例 4:1
- 只更新 image encoder + planning head

**为什么 4:1**: RL 提供 causation learning，但容易 drift away from human behavior。IL 作为 anchor，定期把 policy 拉回 human distribution。这类似 RLHF 中 KL penalty 的作用，但 RAD 用 explicit IL step 而非 KL term。参考 [PPO](https://arxiv.org/abs/1707.06347) 和 [GAE](https://arxiv.org/abs/1506.02438)。

---

## 5. Reward 设计

### 5.1 四类 Reward Sources

$$
\mathcal{R} = \{r_{dc}, r_{sc}, r_{pd}, r_{hd}\}
$$

| Reward | Trigger Condition | Decoupling |
|--------|-------------------|------------|
| $r_{dc}$ (dynamic collision) | ego bbox 与 dynamic obstacle bbox 重叠 | → longitudinal |
| $r_{sc}$ (static collision) | ego bbox 与 static obstacle Gaussians 重叠 | → lateral |
| $r_{pd}$ (positional deviation) | ego position 到 expert trajectory 的距离 > $d_{\max} = 2.0m$ | → lateral |
| $r_{hd}$ (heading deviation) | ego heading 与 expert heading 角度差 > $\psi_{\max} = 40°$ | → lateral |

**Decoupled reward**:
$$
r^x_t = r^{sc}_t + r^{pd}_t + r^{hd}_t, \quad r^y_t = r^{dc}_t
$$

**Intuition**: 
- Dynamic collision 主要跟速度有关（追尾、被追尾），所以归到 longitudinal
- Static collision / deviation 主要跟 steering 有关，归到 lateral
- 这种 decoupling 让 advantage estimation 更精准：lateral action 的 gradient 只来自 lateral reward，反之亦然

**关键设计**: 任何 reward trigger 都立即 terminate episode。因为 3DGS 在 ego vehicle 偏离 trajectory 后渲染质量下降（unobserved views），sensor noise 增大，对 RL training 有害。

### 5.2 Ablation 结果分析（Table 2）

| ID | 配置 | CR↓ |
|----|------|-----|
| 1 | 只有 dynamic collision | 0.172 |
| 2 | 只有 static + positional + heading | 0.238（最高！） |
| 3 | dynamic + positional + heading | 0.146 |
| 6 | 全部 | **0.089** |

**Key insight**: ID 2 缺 dynamic collision reward 时 CR 最高（0.238），说明 dynamic collision reward 是最关键的——它直接教 policy 学会避让 moving obstacle。这呼应了 IL 的核心 weakness：IL data 里 collision 极少，policy 对 dynamic obstacle 不敏感。

---

## 6. Policy Optimization: Decoupled PPO + GAE

### 6.1 GAE Computation

对 lateral 和 longitudinal 分别计算 advantage：

$$
\delta^x_t = r^x_t + \gamma V_x(s_{t+1}) - V_x(s_t)
$$
$$
\delta^y_t = r^y_t + \gamma V_y(s_{t+1}) - V_y(s_t)
$$
$$
\hat{A}^x_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta^x_{t+l}
$$
$$
\hat{A}^y_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta^y_{t+l}
$$

变量含义：
- $\delta^x_t, \delta^y_t$: temporal difference (TD) error，衡量当前 reward + 下一步 value 与当前 value 的差
- $\gamma = 0.9$: discount factor，未来 reward 的折扣
- $\lambda = 0.95$: GAE parameter，控制 bias-variance tradeoff。$\lambda \to 0$ 偏向 TD(0)（low variance, high bias）；$\lambda \to 1$ 偏向 Monte Carlo（high variance, low bias）
- $V_x(s), V_y(s)$: learned value function，估计 state $s$ 下 lateral / longitudinal 的 expected cumulative reward

**Intuition**: GAE 是 RL 里最经典的 variance reduction 技巧。closed-loop AD 里 reward 是 sparse 的（collision 只在最后几步发生），但前面的 action 也有贡献。GAE 通过 $\lambda$ 把 future reward 传播回前面的 action，让前面每一步都得到 signal。

### 6.2 Decoupled PPO Clipped Objective

$$
\mathcal{L}^{PPO}(\theta) = \mathcal{L}^{PPO}_x(\theta) + \mathcal{L}^{PPO}_y(\theta)
$$

$$
\mathcal{L}^{PPO}_x(\theta) = \mathbb{E}_t \left[ \min\left( \rho^x_t \hat{A}^x_t, \text{clip}(\rho^x_t, 1-\epsilon_x, 1+\epsilon_x) \hat{A}^x_t \right) \right]
$$

其中 importance sampling ratio:
$$
\rho^x_t = \frac{\pi_\theta(a^x_t | s_t)}{\pi_{\theta_{old}}(a^x_t | s_t)}
$$

变量含义：
- $\rho^x_t$: 新旧 policy 在 action $a^x_t$ 上的概率比
- $\epsilon_x = 0.1, \epsilon_y = 0.2$: clipping range。lateral 更严（0.1），longitudinal 更松（0.2）

**为什么 $\epsilon_x < \epsilon_y$**: Lateral action（steering）对 trajectory 影响大，小变化就能导致大 deviation，所以更新要保守。Longitudinal action（speed）相对 robust，可以更激进更新。这是 domain knowledge 注入 hyperparameter 的好例子。

参考 [PPO original paper](https://arxiv.org/abs/1707.06347)。

---

## 7. Auxiliary Objectives: Dense Supervision

### 7.1 动机

RL reward 是 sparse 的——大部分 step 没有任何 reward trigger，policy gradient 信号弱。RAD 设计 4 个 auxiliary objectives 提供 dense supervision，每个对应一种 risk source。

### 7.2 Action Probability Decomposition

把 action distribution 按方向拆分：

$$
\Delta\pi^{dec}_y = \sum_{a^y_t < a^{y,old}_t} \pi_\theta(a^y_t | s_t) \quad \text{(减速概率)}
$$
$$
\Delta\pi^{acc}_y = \sum_{a^y_t > a^{y,old}_t} \pi_\theta(a^y_t | s_t) \quad \text{(加速概率)}
$$
$$
\Delta\pi^{left}_x = \sum_{a^x_t < a^{x,old}_t} \pi_\theta(a^x_t | s_t) \quad \text{(左转概率)}
$$
$$
\Delta\pi^{right}_x = \sum_{a^x_t > a^{x,old}_t} \pi_\theta(a^x_t | s_t) \quad \text{(右转概率)}
$$

### 7.3 Directional Factors

每个 auxiliary objective 有一个 directional factor $f \in \{+1, -1\}$，根据 risk 的相对位置决定鼓励哪个方向：

**Dynamic Collision** ($f_{dc}$):
- 碰撞在前方 → $f_{dc} = +1$，鼓励减速
- 碰撞在后方 → $f_{dc} = -1$，鼓励加速

$$
\mathcal{L}_{dc}(\theta) = \mathbb{E}_t \left[ \hat{A}^{dc}_t \cdot f_{dc} \cdot (\Delta\pi^{dec}_y - \Delta\pi^{acc}_y) \right]
$$

**Static Collision** ($f_{sc}$):
- 障碍物在左 → $f_{sc} = +1$，鼓励右转
- 障碍物在右 → $f_{sc} = -1$，鼓励左转

$$
\mathcal{L}_{sc}(\theta) = \mathbb{E}_t \left[ \hat{A}^{sc}_t \cdot f_{sc} \cdot (\Delta\pi^{right}_x - \Delta\pi^{left}_x) \right]
$$

类似地有 $\mathcal{L}_{pd}$ 和 $\mathcal{L}_{hd}$。

### 7.4 总 Objective

$$
\mathcal{L}(\theta) = \mathcal{L}^{PPO}(\theta) + \lambda_1 \mathcal{L}_{dc} + \lambda_2 \mathcal{L}_{sc} + \lambda_3 \mathcal{L}_{pd} + \lambda_4 \mathcal{L}_{hd}
$$

**Intuition**: 这些 auxiliary objectives 本质上是 **shaping the action distribution toward safe direction**，even when no explicit reward fires。比如前方有车但还没碰撞，$\mathcal{L}_{dc}$ 就开始 push policy 把概率质量从加速移到减速。这比纯 sparse reward 的 PPO gradient signal 强得多。

参考 Table 3 ablation：去掉所有 auxiliary（ID 1）→ CR 0.249；全开（ID 8）→ CR 0.089。提升巨大。

---

## 8. 3DGS Environment 重建

RAD 基于 [StreetGaussian](https://arxiv.org/abs/2406.15456) 框架扩展，关键改进：

1. **Mesh-constrained road geometry**: 把 Gaussian 球约束到 road mesh 表面，保证任意 viewpoint 下道路几何精确
2. **Sky 单独建模**: 避免 sky 与 foreground 混淆，改善复杂光照下渲染
3. **Foreground object pose optimization**: 对 vehicle/pedestrian 优化 pose，加入 depth + normal consistency supervision
4. **Off-trajectory view rendering**: 这是 closed-loop training 的关键——ego vehicle 会偏离 expert trajectory，3DGS 必须在 unobserved views 也能渲染合理画面

**为什么 3DGS 而非 NeRF**: NeRF 渲染慢（秒级），无法支持 10Hz closed-loop rollout。3DGS 实时渲染（>30 FPS），适合大规模 RL training。参考 [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)。

**Consistency 验证** (Figure 5): paper 把同一 policy 放到 real world 和 3DGS environment 跑，比较 position over time，结果高度一致。这证明 3DGS 是 real world 的 valid digital twin，RL 在 3DGS 学到的 policy 可迁移。

---

## 9. 实验结果深度分析

### 9.1 主结果 (Table 4)

| Method | CR↓ | DCR↓ | SCR↓ | DR↓ | ADD↓ | Long.Jerk↓ | Lat.Jerk↓ |
|--------|-----|------|------|-----|------|-----------|----------|
| TransFuser | 0.320 | 0.273 | 0.047 | 0.235 | 0.263 | 4.538 | 0.142 |
| VAD | 0.335 | 0.273 | 0.062 | 0.314 | 0.304 | 5.284 | 0.550 |
| GenAD | 0.341 | 0.299 | 0.042 | 0.291 | 0.265 | 11.37 | 0.320 |
| VADv2 | 0.270 | 0.240 | 0.030 | 0.243 | 0.273 | 7.782 | 0.171 |
| **RAD** | **0.089** | **0.080** | **0.009** | **0.063** | 0.257 | 4.495 | **0.082** |

**Key observations**:
1. **CR 从 0.270 (VADv2) 降到 0.089**，3× 降低。这是 RL 直接 optimize collision avoidance 的结果。
2. **DCR 从 0.240 降到 0.080**，dynamic collision 大幅减少。这对应 dynamic collision reward + auxiliary objective 的效果。
3. **ADD 0.257** 与 IL methods 相当（0.263-0.304），说明 RL 没有以 deviation 为代价换 safety——IL regularization 起作用了。
4. **Lat. Jerk 0.082 最低**，说明 trajectory smoothness 最好。这反直觉——RL 通常 jittery，但 RAD 的 IL regularization + decoupled action 让 lateral control 很 smooth。

### 9.2 IL vs RL vs IL+RL (Table 1)

| Strategy | CR↓ | ADD↓ | Long.Jerk↓ |
|----------|-----|------|-----------|
| IL only | 0.229 | 0.238 | 3.928 |
| RL only | 0.143 | 0.345 | 4.204 |
| RL+IL | **0.089** | 0.257 | 4.495 |

**Key insight**: 
- IL only: ADD 最低（0.238）但 CR 高（0.229）——像人但会撞
- RL only: CR 低（0.143）但 ADD 高（0.345）——安全但不像人
- RL+IL: CR 最低（0.089）且 ADD 可接受（0.257）——安全且像人

这完美印证 paper 的核心 thesis：**RL 和 IL 互补**。RL 学 causation（avoid collision），IL 保 alignment（stay human-like）。Long. Jerk 略升（3.928 → 4.495）是 safety 优先的代价——有时需要急刹车避撞。

---

## 10. 关键 Insight 总结

### 10.1 为什么 3DGS + RL 是 game changer

传统 AD RL 用 [CARLA](https://carla.org/) 等 game engine simulator，sensor data 不真实，policy 学到的 perception 无法迁移到 real world。3DGS 提供 photorealistic digital twin，让 end-to-end policy（perception + planning 一起训）能在真实 sensor distribution 下做 closed-loop RL。这是第一次让 "在仿真里 trial-and-error 学到的 policy 能直接迁移到 real world" 成为可能。

### 10.2 Decoupling 是核心设计哲学

RAD 在多个层面 decouple：
- **Action space**: lateral × longitudinal
- **Reward**: $r^x$ vs $r^y$
- **Value function**: $V_x$ vs $V_y$
- **Advantage**: $\hat{A}^x$ vs $\hat{A}^y$
- **PPO objective**: $\mathcal{L}^{PPO}_x$ vs $\mathcal{L}^{PPO}_y$
- **Clipping**: $\epsilon_x = 0.1$ vs $\epsilon_y = 0.2$
- **Training stage**: perception / planning / reinforced 三阶段，参数 freeze 解耦

这种 systematic decoupling 让每个维度的 gradient signal clean，sample efficiency 高，且能注入 domain knowledge（如 lateral 更保守）。

### 10.3 IL as Regularization vs RLHF KL Penalty

RAD 用 explicit IL step（4:1 比例）而非 KL penalty。优点：
- IL step 用 real expert data，signal 更强
- 避免 KL penalty 导致的 reward hacking（policy 只满足 KL 但不真正 human-like）
- 实现简单，超参少

缺点：
- IL step 是 batch update，不如 KL penalty 的 per-step smoothness
- 4:1 比例需要 tune

参考 [RLHF](https://arxiv.org/abs/2203.02155) 和 [DPO](https://arxiv.org/abs/2305.18290) 的相关讨论。

### 10.4 Sparse Reward 的两层解决

1. **GAE**: 把 future reward 传播回前面 step（temporal credit assignment）
2. **Auxiliary objectives**: 提供 dense directional supervision（spatial credit assignment）

两者 complementary：GAE 解决 "哪一步负责"，auxiliary 解决 "每个 step 该往哪个方向调"。

---

## 11. Limitations 和 Future Directions

Paper 自己提到：
- 3DGS 对 non-rigid pedestrian 渲染不佳
- Unobserved views（大幅偏离 trajectory）渲染质量下降
- Low-light 场景效果差

我补充几点：
1. **Log-replay NPC**: 其他 traffic participant 用 log-replay，无法 react to ego vehicle 的 OOD behavior。真实世界里前车会避让，log-replay 不会。这可能让 policy 学到过于激进的策略。Future: 用 learnable NPC policy。
2. **3DGS reconstruction cost**: 4305 个 scene 每个都要单独训 3DGS model，成本高。Future: 大规模 pre-trained 3DGS foundation model。
3. **Reward design 仍 manual**: 四种 reward 都是 hand-crafted。Future: learned reward（inverse RL / RLHF with human feedback）。
4. **8s clip 限制**: 长 horizon planning（如 30s 路线规划）无法在 8s clip 里训。Future: 更长 clip + hierarchical RL。

---

## 12. 相关工作和延伸阅读

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Kerbl et al., ACM TOG 2023
- [Street Gaussians](https://arxiv.org/abs/2406.15456) - dynamic urban scene 3DGS
- [HUGSim](https://arxiv.org/abs/2412.01718) - photo-realistic closed-loop simulator
- [UniAD](https://github.com/OpenDriveLab/UniAD) - planning-oriented AD
- [VAD](https://github.com/hustvl/VAD) - vectorized AD
- [VADv2](https://arxiv.org/abs/2402.13243) - multi-modal planning
- [DiffusionDrive](https://github.com/hustvl/DiffusionDrive) - diffusion policy for AD
- [PPO](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [GAE](https://arxiv.org/abs/1506.02438) - Schulman et al., 2015
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer) - BEV encoder
- [MapTRv2](https://github.com/hustvl/MapTR) - vectorized map
- [CARLA](https://carla.org/) - AD simulator
- [NeuRAD](https://arxiv.org/abs/2311.13360) - neural rendering for AD
- [CIRL](https://arxiv.org/abs/1807.11302) - controllable imitative RL
- [CADRE](https://arxiv.org/abs/2203.09887) - cascade deep RL for AD
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - RL for reasoning in LLM
- [OpenAI O1](https://openai.com/o1/) - RL-enhanced reasoning

---

## 13. 对 LLM RL 的类比思考

RAD 的 RL + IL 范式与 DeepSeek-R1 / OpenAI O1 的 RL for reasoning 高度类似：

| 维度 | RAD (AD) | R1/O1 (LLM) |
|------|----------|-------------|
| Environment | 3DGS digital twin | Math/code verifier |
| Reward | Collision / deviation | Correctness / format |
| IL regularization | Expert driving demos | SFT on human text |
| Sparse reward | Collision 少见 | Correct answer 少见 |
| Auxiliary objective | Directional loss | Format / length penalty |
| Decoupling | Lateral / longitudinal | — |

这暗示一个通用 pattern：**RL + IL/SFT hybrid 是让 neural policy 同时具备 capability（RL 学到的）和 alignment（IL 保住的）的标准范式**。在 AD 里是 safety + human-likeness，在 LLM 里是 reasoning + helpfulness。

---

希望这个讲解帮你 build 了 intuition。RAD 的核心贡献是第一次把 photorealistic 3DGS + closed-loop RL + IL regularization 三者结合，证明了这条路在 end-to-end AD 上可行且有效。3× collision rate 降低是硬核 evidence。下一步值得关注的 direction 是 scaling 3DGS environments（从 4305 到百万级）和引入 learned NPC / learned reward。
