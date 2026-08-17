---
source_pdf: EgoScale Scaling Dexterous Manipulation with Diverse Egocentric Human
  Data.pdf
paper_sha256: 39c691baf374a154e26ffc0098b97037875909495013d4017d97761716ff2735
processed_at: '2026-08-04T02:35:09-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# EgoScale 人话版

## 这篇 paper 在干嘛

NVIDIA 的人想训练一个 dexterous manipulation policy（让机器人用灵巧手完成精细操作）。问题是这样做的瓶颈一直是 robot data 太难收集——你 teleop 一整天也就那么几条轨迹。

所以他们问：**能不能直接拿人做事的视频来训？** 人做事的视频不要钱，YouTube 上一抓一大把。

## 为什么这件事之前做不好

两个原因：

**第一，数据量太小**。之前的工作也就用几十到几百小时 human video，跟 robot teleop 数据差不多量级，看不出 scale 的好处。

**第二，hand 太复杂**。之前很多工作只迁 wrist motion，用在 gripper 上还行，但 dexterous hand 22 个 DoF，你不把 finger 的动作也学进去，policy 就是个残废。

EgoScale 的两个核心 move 就是直接冲这两个痛点。

## 他们怎么做的

**Stage I: 大规模 human pretrain**

收集了 **20,854 小时** egocentric human video（戴头摄像头录的第一人称视角）。用 SLAM 估计 camera pose，用 hand pose estimator 估计 21 个 hand keypoints，再 retarget 到 Sharpa hand 的 22-DoF joint space。

关键 trick 是 wrist 用 **relative SE(3) motion** 表示——每一帧的 wrist pose 相对于 chunk 起点的 inverse。这样无论摄像头怎么晃、人在哪走，"手往前伸 10cm" 这个 action 都一样。人和 robot 的 action 就能混在一起训。

**Stage II: 小规模 aligned mid-train**

50 小时 human + 4 小时 robot，用完全相同的 camera 配置和 workspace 收集。这步是把 Stage I 学到的 "motion prior" 锚定到 robot 真实的 sensing 和 control 上。

**Stage III: Post-train**

每个 task 100 个 robot demo fine-tune，出最终 policy。

## 核心发现

**Scaling law**：1k → 2k → 4k → 10k → 20k hours，validation loss 完美 log-linear 下降，$R^2 = 0.9983$。公式是：

$$L = 0.024 - 0.003 \cdot \ln(D)$$

$D$ 是 data hours。这个 loss 还跟 real robot success rate 强相关——意味着你不用每次跑 robot eval，看 offline loss 就能预测 policy 多好。这跟 LLM 的 scaling law 一个味道。

**One-shot transfer**：pretrain + mid-train 后，给 **1 个 robot demo + 100 个 aligned human demo** 就能学新 task。Fold shirt 拿到 88% success，bottle cap unscrewing 55%。Mid-train 里根本没这两个 task，model 是把共享的 motion primitive 复用了。

**Cross-embodiment**：22-DoF 上 pretrain 的 model，迁移到 G1 的 7-DoF tri-finger hand，比 G1 自己从头训高 30%+。证明学到的是 embodiment-agnostic 的 motor prior。

## 一个反直觉的 ablation

Hand action representation 选三种：
- Wrist only（finger 无监督）
- Fingertip SE(3)（geometric rich）
- **22-DoF joint space**（EgoScale 选的）

22-DoF joint space 最稳。Fingertip 反而不行——fingertip pose 一点小误差，retarget 出来的 joint configuration 就不可行，contact-sensitive task 直接崩。

Intuition: action space 选一个 "窄但永远 feasible" 的比 "宽但要 IK" 的好学。这跟 RL 里 action space design 影响 sample efficiency 是一回事。

## 我的 take

这篇 paper 真正的 statement 是 abstract 最后那句：**"humans can be treated as another scalable embodiment in robot learning"**。

把人当成 robot 的一种 embodiment——跟 wheeled humanoid、bipedal、gripper arm 并列。robot learning 的 scaling 路径就此和 LLM 对齐：更多 data → log-linear 降 loss → 线性提升 performance。

但要注意 log-linear 不是 power law。每翻一倍 data loss 下降固定 $0.003 \cdot \ln 2 \approx 0.00208$，绝对收益随接近 noise floor 递减。Paper 没同时 scale model size，可能是 capacity bottleneck 而非 data noise。这个 open question 决定了这条路能走多远。

Link: https://research.nvidia.com/labs/gear/egoscale/

---

# EgoScale: 用大规模 Egocentric Human Data 解锁 Dexterous Manipulation 的 Scaling Law

## 1. Paper 的高层故事

EgoScale 想证明的核心 thesis 是：**human-to-robot transfer for dexterous manipulation 本质上是一个 scaling phenomenon**。这不是一个新 trick，也不是一个新架构，而是一个"如果 data 规模够大、action representation 选得对、最后用少量 aligned data 把 representation 锚定到 robot sensing/control，那么 human video 就是 scalable supervision source"的故事。

这个 thesis 的分量在于：之前 human-to-robot transfer 的工作大多停留在几十到几百小时数据规模，并且只做 gripper 或者 low-DoF hand。EgoScale 直接把数据规模推到 **20,854 小时**（比 prior work 大 20 倍以上），并且在 wrist + hand action prediction 的 validation loss 上拟合出一条非常干净的 log-linear scaling law：

$$L = 0.024 - 0.003 \cdot \ln(D), \quad R^2 = 0.9983$$

其中 $L$ 是 optimal validation loss at convergence，$D$ 是 human pretraining data 的小时数。注意 $R^2 = 0.9983$ 这个拟合度几乎是 noise-free 的，这意味着从 1k hours 到 20k hours，validation loss 是高度可预测的。更关键的是，这个 offline metric 与 real-robot performance 在 5 个 dexterous task 上强相关——这就让 human data 变成了一个 "predictable supervision source"，可以像 pretraining language model 一样做 scaling planning。

Link: NVIDIA EgoScale Project Page — https://research.nvidia.com/labs/gear/egoscale/

---

## 2. 为什么 Human Data 是 Scalable Supervision Source（Intuition Building）

Karpathy 你应该会很关心这个 intuition：为什么 human egocentric video 可以作为 manipulation policy 的 supervision？

我的理解是这样的：

1. **物理 prior 的密度**。一个 egocentric human video 里，每一帧都隐含了 "在场景 S 下，手 H 应该做什么 motion 才能 achieve 目标 G" 的 dense supervision。SLAM 给你 camera pose（从而给你 wrist 的 SE(3) 轨迹），hand pose estimator 给你 21 个 keypoints（从而给你 finger articulation）。把这些信号 retarget 到 robot joint space，整段 video 就变成了一条 (observation, action_chunk) 轨迹，跟 robot teleoperation 数据没有本质区别。

2. **Diversity vs Volume 的 trade-off 被 long-tail 解决**。EgoScale 数据覆盖了 9,869 scenes / 6,015 tasks / 43,237 objects。这个分布是长尾的，retail 占 20.1%，fashion 占 11.8%，repair 占 11.5%... 一直到各种细分类别。Hu et al. (https://arxiv.org/abs/2410.18647) 在 imitation learning scaling 上的发现是 **diversity 比 volume 更重要，单 task 加 demo 很快 saturate**。EgoScale 的 in-the-wild 数据天然是 max-diversity 分布，所以即使每个 task 的轨迹都不多，aggregate 后 action prediction loss 仍然 log-linear 下降。

3. **Action representation 是 cross-embodiment 的关键**。EgoScale 把 wrist 表示成 **relative SE(3) motion**（见下面公式），这一步直接 strip 掉了 absolute camera pose 和 absolute arm configuration，让 representation 只依赖 local motion structure。这是为什么 22-DoF Sharpa hand 上 pretrain 的 model 可以 transfer 到 7-DoF tri-finger G1。

直觉上讲：人类手指的运动模式（open / close / pinch / 三指 wrap）在所有 humanoid hand 上都有 "morphological mapping"。22-DoF joint angle 学到的不是 "Sharpa 第 3 个 joint 的角度"，而是 "对当前 object 应该做哪种 grasp primitive" 的 latent。这种 latent 在低 DoF hand 上可以 re-decode 成 "三个手指同步闭合"。

---

## 3. Human Action Representation（细节公式解析）

这是 paper 里我觉得最精彩的部分之一。让我把每个公式的变量和上下标都讲清楚。

### 3.1 Raw 信号到 Keypoint 表示

每个 human demonstration 包含：
- Egocentric RGB frames（head-mounted camera, 30 FPS）
- 估计的 camera motion
- 21 个 hand keypoints（包含 wrist）

记号定义：
- $\mathcal{F}_w$: world frame（固定参考系）
- $\mathcal{F}_c^t$: time $t$ 时的 camera frame（因为 camera 会动）
- $\mathbf{T}_{w\leftarrow c}^t \in \mathbb{SE}(3)$: 从 camera frame 到 world frame 的 rigid transform，在 time $t$ 估计得到。这里下标 $w\leftarrow c$ 表示 "world from camera"，即把 camera 坐标系的点映射到 world 坐标系。$\mathbb{SE}(3)$ 是特殊欧氏群，包含 3D rotation + 3D translation（6 DoF）。
- $\mathbf{H}_{c,i}^t \in \mathbb{SE}(3)$: 第 $i$ 个 hand keypoint 在 camera frame 中的 pose。$i = 1, \dots, 21$，其中 **$i = 1$ 表示 wrist**，$i = 2 \dots 21$ 表示其他 finger joints。

那么 wrist 在 world frame 中的 pose 是：

$$\mathbf{W}_w^t = \mathbf{T}_{w\leftarrow c}^t \mathbf{H}_{c,1}^t$$

物理意义：先估计 wrist 在 camera frame 中的相对 pose $\mathbf{H}_{c,1}^t$，再用 camera 的 world pose 把它 lift 到 world frame。这是 SLAM + hand tracking 的标准 pipeline。

### 3.2 Relative Wrist Motion（关键 invariant 表示）

为了让 action representation 对 absolute camera 运动 invariant（head-mounted camera 一直会晃），他们用 **relative wrist motion**：

$$\Delta \mathbf{W}^t = (\mathbf{W}_w^0)^{-1} \mathbf{W}_w^t$$

变量解释：
- $\mathbf{W}_w^0$: 当前 action chunk 的第一个 timestep 的 wrist pose（作为 "anchor"）
- $(\mathbf{W}_w^0)^{-1}$: 这个 pose 的逆（在 SE(3) 里是 inverse transform）
- $\mathbf{W}_w^t$: 当前 chunk 中第 $t$ 个 timestep 的 wrist pose
- $\Delta \mathbf{W}^t$: 相对于 chunk 起点的 relative wrist motion

这个 trick 非常关键：把 absolute world pose 转成 relative frame，等价于让 action 一直在 "current wrist frame" 下表示。无论 camera 怎么动、人在哪个房间，"手向前伸 10 cm" 这个 action 都是一样的。这也是为什么人 robot 数据可以混在一个 batch 里训练——只要 arm action 都用 relative wrist motion 表示。

### 3.3 Hand Articulation Retargeting

21 个 hand keypoints 通过 optimization-based retargeting 映射到 22-DoF Sharpa hand joint space。Paper Appendix D 给了细节：

- Robot hand 用 URDF 定义 forward kinematics：joint angles $\rightarrow$ 20 个 robot keypoint poses（3D position + quaternion）
- 每个 frame 解一个 22 维非线性规划，目标函数是多种 objective 的加权和，约束是 URDF joint limits
- Solver: CasADi + IPOPT
- Warm start: 用上一帧的解
- 后处理: first-order exponential filter 平滑掉 temporal jitter

设计哲学：retargeting 应该 preserve "pinch / fist / wrap" 这种 grasp semantic，而不是精确重建 fingertip position。这正是后面 action representation ablation 里 fingertip-based representation 失败的原因——fingertip pose 的小误差 retarget 后会变成 implausible joint configuration。

### 3.4 统一的 Action Vector

最终每个 timestep 的 action 是：
$$a_t = [\Delta \mathbf{W}^t \in \mathbb{SE}(3); \mathbf{q}_{hand}^t \in \mathbb{R}^{22}]$$

7 维（SE(3) 通常用 6 DoF 表示，3 translation + 3 rotation）+ 22 维 = 28 维（或类似）。这个 vector 是 unified action space，robot data 用同样的格式。Vision-language backbone + DiT action expert 学到的就是这个 distribution。

---

## 4. Data Sources 和 Curation

### Stage I: 20,854 hours in-the-wild egocentric

Composition：
- **829 hours EgoDex**（Apple Vision Pro 高精度 tracking，194 tabletop tasks，每 task ~30 demos）—— 这个起到 "anchor" 作用，提供精确 kinematic 信号
- **~20,000 hours 其他 in-the-wild egocentric video**（household / industrial / retail / educational），noisy 但 diverse

Statistics（Fig 10）：
- **Scenes**: 9,869
- **Tasks**: 6,015
- **Objects**: 43,237
- **Categories**: retail (20.1%), fashion (11.8%), repair (11.5%), food & beverage (11.5%), home (9.5%), construction (7.7%), food processing (6.7%), printing (4.4%), 其余 long tail
- **Environments**: homes, flower shops, electronics repair shops, furniture repair, woodworking, clothing stores, grocery stores, bakeries, workshops, factories, studios, libraries...
- **Tasks**: folding clothes, cleaning shoes, potting plants, ironing, assembling boxes, arranging flowers, sanding wood, food preparation...

数据 pipeline：30 FPS video → SLAM (camera pose) + hand pose estimation → retarget → action chunks。

注意：in-the-wild 数据是 noisy 的，SLAM 在 fast motion / 低光下会 fail，hand pose estimator 在 occlusion 下也会 fail。但 paper 论证说 scale + diversity 弥补了 noise，validation loss 仍然 log-linear 下降。

EgoDex link: https://arxiv.org/abs/2505.11709

### Stage II: 50 hours human + 4 hours robot, aligned

- 344 tabletop manipulation tasks
- 每 task ~30 human + ~5 robot trajectories
- 用与 robot 完全相同的 camera 配置（head + 两个 wrist cameras），matched viewpoints, calibrated intrinsics
- Human motion capture：Vive trackers（wrist 6 DoF）+ Manus gloves（25 joint transforms/hand）
- Motion signals 与 video 同步

这个 dataset 的关键：visual observation 在 human 和 robot 之间是 **directly comparable** 的——同一视角、同一 workspace、同一 lighting。这就让 mid-training 阶段学到 "把 Stage I 学到的 motion prior 锚定到 robot 的 sensing space"。

### Stage II vs Stage I 的分工

- **Stage I**: scale + diversity + semantic grounding（什么 task 对应什么 motion）
- **Stage II**: precise human-robot correspondence（把 representation 锚到 robot control）

这种 decoupling 是 paper 的核心 design：scale 和 alignment 不必同时存在同一个 dataset。

---

## 5. Model Architecture

Paper 里架构是 flow-based VLA，类似 GR00T N1（https://arxiv.org/abs/2503.14734）。结构如下：

```
Observation o_t = (I_t, l_t)
    ↓
[VLM Backbone] (vision encoder + language encoder)
    ↓
Vision-language embedding φ_t
    ↓
[Robot proprioception q_t] -- 或 human 时用 learnable placeholder token
    ↓
Embodiment-conditioned MLP adapter (input)
    ↓
Shared latent state
    ↓
[DiT Action Expert] (Diffusion Transformer for flow matching)
    ↓
Embodiment-conditioned MLP adapter (output)
    ↓
Action chunk a_{t:t+H}
```

关键设计点：

1. **VLM Backbone 完全 shared**——human 和 robot 共享 vision-language encoder，因为视觉理解是 embodiment-agnostic 的（"桌上有杯子" 在 human 和 robot 视角下意思一样）

2. **Proprioception 处理**：robot data 有 proprioception $q_t$（joint angles 等），human data 没有。解决方案是用 learnable placeholder token 替代 $q_t$。这是一个 elegant 的 trick——架构不变，只是 input token 不同。

3. **Embodiment-conditioned MLP adapters**：只在 input 和 output 接口加轻量 adapter。比如 G1 7-DoF tri-finger hand 有自己的 input encoder 和 output decoder，但 vision-language backbone 和 DiT action expert 完全 shared。这是 GR00T N1 / N1.5 的标准做法。

4. **DiT Action Expert + Flow Matching**：不是离散 token 的 action prediction，而是 flow-based continuous action generation。每个 action chunk 通过 iterative denoising 生成。

Flow matching objective 大致是：

$$\mathcal{L} = \mathbb{E}_{t, x_0, x_1, \epsilon} \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2$$

其中：
- $x_0$: 噪声（standard Gaussian）
- $x_1$: ground-truth action chunk
- $x_t = (1-t) x_0 + t x_1$: 线性插值
- $v_\theta$: 学的 velocity field
- $t \in [0, 1]$: flow time

预测 wrist relative motion 和 22-DoF hand joint 一起（concatenated vector）。

---

## 6. Three-Stage Training Recipe

这是工程上最讲究的地方，三阶段的 batch size、learning rate、frozen 状态都不同：

| Stage | Data | Steps | Batch Size | LR | Frozen Parts | Compute |
|-------|------|-------|------------|-----|--------------|---------|
| I: Human pretrain | 20k hours human | 100k | 8,192 | $5\times10^{-5}$ | 无（全部 unfreeze） | 256 GB200 GPUs |
| II: Aligned mid-train | 50h human + 4h robot | 50k | 2,048 | $3\times10^{-5}$ | VL backbone frozen，只更新 vision encoder + DiT | - |
| III: Post-train | 100 demos per task | 10k | 512 | $3\times10^{-5}$ | vision encoder frozen if mid-train used, 否则 unfreeze | - |

设计直觉：
- Stage I 大 batch + 全 unfreeze：让 model 充分 absorb 大规模 human data 的多样性，不限制 representation 的 capacity
- Stage II 小 batch + 部分 frozen：因为 aligned 数据少（54 hours），freeze VL backbone 防止 overfit 到 mid-training data，但 vision encoder + DiT 仍要 update 才能锚到 robot sensing
- Stage III 更小 batch + 视情况 freeze：根据是否经过 mid-train 决定，体现 "mid-train 已经把 visual feature 调好" 的 assumption

---

## 7. Scaling Law 实验细节

这是 paper 的核心 scientific contribution。实验设计：

### Setup
- 5 个数据规模：1k, 2k, 4k, 10k, 20k hours
- 每个 checkpoint 直接 post-train 到下游 task（跳过 mid-train，控制变量）
- 评估 robot performance：average task completion score
- Offline validation: 2,000 held-out egocentric episodes，每个 trajectory 随机 sample 20 个 timesteps，每个 timestep 用 flow matching 采样 16 次取平均，算 wrist + hand action 的 MSE

### 三个关键观察

**Observation 1**: 小数据集（1k-2k hours）overfit——validation loss 先降后升。大数据集（10k-20k hours）单调下降，没 overfit 迹象。这暗示 diversity > volume 在这个 regime 仍然成立。

**Observation 2**: Convergence 后的 optimal validation loss 与 data scale 在 log space 完美线性：

$$L = 0.024 - 0.003 \cdot \ln(D), \quad R^2 = 0.9983$$

注意：这是 **log-linear**，不是 power law（Hoffmann et al. Chinchilla 的 $\mathcal{L}(N) \propto N^{-\alpha}$ 形式）。区别在于：log-linear 是 $L = a - b \ln D$，等价于 $L$ 随 $D$ 的对数线性下降，每翻一倍 data，loss 下降固定 $b \ln 2 \approx 0.00208$。这与 LLM scaling law（power law）不同，可能是因为：
- Action prediction 的 noise floor 比较高（retargeting + SLAM + hand pose 误差累积）
- Data diversity 随 scale sub-linearly 增加（同一 category 内的冗余）
- Model capacity 可能是 bottleneck（256 GB200 的训练规模但 model size 没说）

**Observation 3**: Task completion score 从 1k 的 0.30 到 20k 的 0.71，monotonic 上升，没 saturation。Validation loss 与 robot performance 强相关——这意味着 offline metric 可作为 proxy 来做 scaling planning，不用每次跑 robot eval。

### 实验数据表（基于 Fig 5 数据估算）

| Data Hours | Validation Loss | Avg Task Completion |
|-----------|----------------|---------------------|
| 1,000 | ~0.0035 | 0.30 |
| 2,000 | ~0.0029 | ~0.40 |
| 4,000 | ~0.0027 | ~0.50 |
| 10,000 | ~0.0024 | ~0.60 |
| 20,000 | ~0.0022 | 0.71 |

---

## 8. Main Experimental Results

5 个 dexterous task（Fig 3）：

1. **Shirt Rolling**（20 demos）: 双手协调 fold + roll T-shirt + 放入 basket
2. **Card Sorting**（100 demos）: 从 tightly stacked deck 分出单张卡 + 插入正确 holder（按 color）
3. **Tong Fruit Transfer**（100 demos）: 抓 tongs → 用 tongs 夹 fruit → 放入 basket → 放回 tongs
4. **Bottle Cap Unscrewing**（100 demos，25 per bottle × 4 bottles）: 抓 bottle + rotate cap 多圈 + 拔 cap + 放桌上
5. **Syringe Liquid Transfer**（100 demos）: 拿 syringe → 从 tube A 抽液 → 注入 tube B → 丢进 trash（long-horizon）

### 4 个比较 condition

1. From scratch（no pretrain）
2. Mid-train only（只用 aligned data pretrain）
3. Human pretrain only
4. **Human pretrain + mid-train**（EgoScale 完整版）

### 主要结果

- Human pretrain 比 from scratch 平均 task completion 提升 **55%+**
- 即使没有 aligned mid-train，纯 human pretrain 也比 mid-train-only baseline 表现好（大部分 task）—— 证明 scale + diversity 比 precise alignment 更重要
- Human pretrain + mid-train 综合最好——证明 scale 和 alignment 是 **complementary**
- 最终 EgoScale vs no-pretrain baseline 在 22-DoF Sharpa hand 上平均 success rate 提升 **54%**

### Cross-embodiment（G1, 7-DoF tri-finger）

两个 task：
- **Pen in Bin**: 左手开 bin + 右手抓 pen + 放入 bin
- **Dish in Rack**: 桌上 3 plates → 右到左 handover → 直立放入 dish rack

Lower body 用 Homie policy (https://arxiv.org/abs/2502.13013) 保持 balance 和 locomotion，upper body 用 EgoScale policy。

Result: Human pretrain + mid-train 比 no pretrain 提升 **30%+** 在两个 task 上。这证明 22-DoF 上 pretrain 学到的 representation 可以作为 "reusable motor prior" 迁移到 low-DoF hand。

---

## 9. One-Shot Adaptation 实验

这是我觉得最 magic 的实验。

### Setup
- 跳过 mid-train 的 baseline 在 one-shot setting 下 fail
- Pretrain + mid-train 的 EgoScale 用 **1 个 robot demo + 100 aligned human demos** 就能学新 task

### Tasks
1. **One-Shot Shirt Folding**（mid-train 数据里有 folding 但不是这个 task）
2. **One-Shot Bottle Cap Unscrewing**（3 个不同 geometry 的 bottle）

### Results
- Fold Shirt: **0.88 success**
- Unscrew Bottle: **0.55 success**（across 3 bottle geometries）

### 为什么能 one-shot？

直觉：mid-train 数据虽然不包含目标 task，但包含了**共享的 motion primitive**。比如 mid-train 里有其他 folding 类 task，学到了 "双手对称闭合 + 旋转" 这个 primitive，one-shot 给一个新 task 的 single demo 就可以 trigger 这个 primitive 的正确 context。

这本质上是一种 **compositional generalization**：model 不是从 0 学新 task，而是把 pretrained motor primitives re-compose 到新 context 上。

---

## 10. Action Representation Ablation（Section 3.6）

这节比较三种 human hand action 表示：

1. **Wrist-only**: 去掉所有 finger supervision，只预测 wrist SE(3) motion
2. **Fingertip-based**（EgoVLA 风格, https://arxiv.org/abs/2507.12440）: 预测 wrist + fingertips 的 SE(3) 轨迹，再用 MLP 映射到 robot joint
3. **22-DoF joint space**（EgoScale default）: 直接 retarget 到 Sharpa hand joint space

### 结果（Fig 8）

- **Wrist-only** 在所有 task 上都不行，特别是需要精细 finger articulation 的 Tongs / Cards——grasp 不稳，手闭合太早或太晚
- **Fingertip-based** 比 wrist-only 好，但 inconsistent。fingertip pose 小误差经过 MLP 映射会变成 implausible joint configuration，导致 Cards / Bottle 这种 contact-sensitive task 失败
- **22-DoF joint space** 在所有 task 上最一致

### Intuition

这个结果其实揭示了一个深层 trade-off：
- **Geometric supervision rich**（fingertip）→ 但 retarget 错误会被放大
- **Joint-space direct**（EgoScale）→ supervision 略 abstract，但 retarget 时已经 enforce 了 joint limits + kinematic constraints，输出永远 feasible

这跟 RL 里 "action space design 决定 sample efficiency" 是一个道理：选一个 "窄但 feasible" 的 action space 比选一个 "宽但需要 IK" 的 action space 更容易学。

---

## 11. Hand Retargeting 算法细节（Appendix D）

完整 pipeline：

```
Input: 21 human hand keypoints (as 25 keypoints per hand with 3D pos + orientation)
       Sharpa Hand URDF (22 joints, joint limits)

Per frame:
  Variables: q ∈ ℝ^22 (22 joint angles)
  Constraints: q_lower ≤ q ≤ q_upper (from URDF)
  Objective: min Σ w_i * f_i(q)
    f_1: forward kinematics(q) ↔ human keypoints position matching
    f_2: orientation alignment
    f_3: 临时 smoothness (optional)
  Solver: IPOPT (interior point) via CasADi
  Warm start: q_prev_frame
  
Post-processing:
  q_smoothed = α * q_raw + (1-α) * q_prev_smoothed  (first-order exp filter)
```

设计哲学：**preserve grasp semantics**（pinch / fist / wrap）over precise fingertip position matching。这与 action representation ablation 的结论一致——保留语义比保留精确几何更重要。

---

## 12. Robot System Setup

### Galaxea R1 Pro（主平台）
- Dual-arm wheeled humanoid，固定 base 和 torso
- 两只 7-DoF arm，控制空间是 **relative end-effector space**（增量 position + orientation）
- 两只 **22-DoF Sharpa Wave hand**（https://www.sharpa.com/pages/wave），joint-space control
- 三 RGB cameras: head-mounted（egocentric）+ 两个 wrist-mounted（朝 palm）
- 关键：wrist cameras 朝 palm，捕捉 close-range hand-object interaction——这对 dexterous manipulation 必不可少

### Unitree G1（cross-embodiment）
- Shorter arm，reduced workspace
- **7-DoF tri-finger hand**（vs Sharpa 22-DoF）
- 两个 OAK-1-Wide wrist cameras + 一个 OAK-D-Wide head camera
- Homie policy 控制 lower body（balance + locomotion）
- EgoScale policy 控制 upper body

---

## 13. 与 Related Work 的对比

| Method | Data Scale | Hand Action | Cross-Embodiment | One-shot |
|--------|------------|-------------|------------------|----------|
| EgoMimic (https://arxiv.org/abs/2410.24221) | ~tens hours | wrist only | limited | no |
| EgoVLA (https://arxiv.org/abs/2507.12440) | medium | wrist + fingertip | yes (IK) | no |
| DexWild (https://arxiv.org/abs/2505.07813) | medium | dexterous | yes | no |
| Humanoid Policy ≈ Human Policy (https://openreview.net/forum?id=Tx54fkQ3Cq) | medium | dexterous | yes | no |
| EgoBridge (NeurIPS) | small | wrist | domain adapt | no |
| **EgoScale** | **20,854 hours** | **22-DoF joint space** | **yes (G1)** | **yes** |

Concurrent work "Emergence of human to robot transfer in VLA" (https://arxiv.org/abs/2512.22414) 也在做类似事情，但 EgoScale 的 scaling law 实验 + cross-embodiment + one-shot 三件套更完整。

---

## 14. Limitations 和 Open Questions（Karpathy 你可能关心）

### 14.1 Scaling Law 是 log-linear 不是 power law

LLM scaling law 是 power law $\mathcal{L} \propto N^{-\alpha}$，每翻倍参数 / data，loss 按比例下降。EgoScale 是 $L = a - b \ln D$，每翻倍 data loss 下降固定 $b \ln 2$，这意味着 **marginal return 是 constant in log scale**，但 absolute return 随 scale 递减（因为 loss 越接近 noise floor 越难降）。

这暗示两种可能：
1. **Noise floor 存在**：retargeting + SLAM + hand pose 误差给 action prediction loss 设了一个下限，power law 在接近下限时变成 log-linear
2. **Model capacity bottleneck**：可能 model 不够大，data 多到一定程度后 capacity 限制 dominant。如果同时 scale model size，可能恢复 power law

这个 open question 对未来 scaling 规划很重要。

### 14.2 In-the-wild 数据的 noise 没有量化

Paper 没给 SLAM / hand pose estimator 的失败率，没说有多少 trajectory 被 filter 掉。如果 10% 的数据是 garbage，那 scaling law 的 interpretation 会不同——可能不是 "human behavior 学得到底"，而是 "garbage data 也 log-scale beneficial as regularization"。

### 14.3 22-DoF 作为 universal action space 的 generalization 没充分验证

只 cross-embodiment 测试了 G1 7-DoF。更激进的设计是测试 parallel jaw gripper、四指 hand、甚至完全不同 kinematic 的 hand（比如 soft robotic hand）。如果 22-DoF prior 真的 embodiment-agnostic，那应该 transfer 到这些。

### 14.4 One-shot 的"运气成分"

Paper 报告 Fold Shirt 0.88 success，但 fold 是相对 forgiving 的 task（容许 partial failure）。Syringe 这种 long-horizon + precision task 没 one-shot 报告。可能 one-shot 在 simple task 上 work，complex task 仍然需要 10+ demos。

### 14.5 Mid-training 的 4 hours robot data 不算 "minimal"

虽然 paper 强调 "minimal robot supervision"，但 mid-training 用了 4 小时 robot data（344 tasks × 5 trajectories）。这在 academic 算少，但 industrial 还是要 4 小时 teleop。真正的 "minimal" 应该是 zero robot data，纯 human pretrain + zero-shot execution。

### 14.6 没有与 pure robot data scaling 的 baseline

EgoScale 比 no-pretrain baseline 好 54%，但没回答 "如果把这 4 小时 robot data 扩大到 100 小时，是否能达到 EgoScale 水平"。如果 human data 真的是 efficient supervision，那应该比纯 robot data 更 cost-effective，但 paper 没做这个 cost-benefit comparison。

---

## 15. 对 Future Direction 的联想

### 15.1 Self-supervised human video

Paper Section 6 提到 "incorporating weaker or unlabeled video via self-supervised objectives"。这指向一条路：用 VAE / contrastive learning / masked prediction 在无 action label 的 human video 上 pretrain，再用有 label 的 20k hours 做 mid-train。这样 data 规模可以推到 100k+ hours（YouTube 上随便找）。

参考 FLARE (https://openreview.net/forum?id=HXJ6pUSn1L)，NVIDIA 自己的工作，做 implicit world modeling，可以与 EgoScale 结合。

### 15.2 TraceVLA 风格的 visual prompting

TraceVLA (https://arxiv.org/abs/2412.10345) 用 visual trace prompts 增强 spatial-temporal awareness。EgoScale 的 wrist trajectory 其实就是 implicit trace，可以考虑在 input image 上画 wrist trajectory overlay，让 vision encoder 直接看到 motion pattern。

### 15.3 LLARVA 风格的 vision-action instruction tuning

LLARVA (https://arxiv.org/abs/2406.11815) 用 action label 作为 instruction tuning 的 target。EgoScale 可以扩展到用 language 描述 action（"open hand, grasp bottle, rotate"），让 model 同时学 action prediction 和 action-language alignment。

### 15.4 加入 tactile / force 信号

EgoScale 完全没有 tactile supervision。但 dexterous manipulation 的 contact-rich 部分（screw cap / pinch card）非常依赖 force feedback。如果能在 wrist 加 force sensor，把 force trajectory 也作为 action chunk 的一部分，可能进一步提升 contact-sensitive task。

### 15.5 World model 的潜在整合

VLA 学的是 $p(a|o)$，但 manipulation 也需要 $p(o_{t+1}|o_t, a_t)$（world model）。EgoScale 学到的 representation 可能直接复用为 world model 的 state representation，从而做 model-based RL 或 planning。

### 15.6 Long-horizon planning

Syringe task 是 long-horizon 的（7 个 sub-step），但 paper 没专门分析 long-horizon 的 scaling 行为。如果 human pretrain 主要帮助 short-horizon motor primitive，那 long-horizon 可能需要 hierarchical structure（high-level planner + low-level VLA）。MimicPlay (https://arxiv.org/abs/2302.12422) 是这个方向的 reference。

---

## 16. 总结：EgoScale 的真正 contribution

回到 Karpathy 你最关心的 "build intuition" 角度，EgoScale 给了三个 insights：

1. **Dexterous manipulation 的 human-to-robot transfer 是 scaling phenomenon**：log-linear scaling law 在 1k→20k hours 范围内成立，validation loss 与 robot performance 强相关。这让 human data 变成 "predictable supervision source"，可以做 scaling planning。

2. **Action representation 选 joint-space 比 fingertip-space 更好**：因为 retarget 时 enforce 了 feasibility constraints，输出永远在 robot action manifold 上。这是 dexterous manipulation 特有的 insight（gripper task 不需要这种考量）。

3. **Decoupling scale 和 alignment 是 effective recipe**：Stage I 提供多样性 + 语义，Stage II 锚到 robot sensing/control。两者分工让大规模 noisy human data 和小规模 precise aligned data 互补，不需要在同一个 dataset 里同时追求 scale 和 alignment。

Final reference list:
- EgoScale: https://research.nvidia.com/labs/gear/egoscale/
- GR00T N1: https://arxiv.org/abs/2503.14734
- EgoDex: https://arxiv.org/abs/2505.11709
- EgoMimic: https://arxiv.org/abs/2410.24221
- EgoVLA: https://arxiv.org/abs/2507.12440
- DexWild: https://arxiv.org/abs/2505.07813
- DexUMI: https://arxiv.org/abs/2505.21864
- Humanoid ≈ Human Policy: https://openreview.net/forum?id=Tx54fkQ3Cq
- Homie: https://arxiv.org/abs/2502.13013
- FLARE: https://openreview.net/forum?id=HXJ6pUSn1L
- TraceVLA: https://arxiv.org/abs/2412.10345
- LLARVA: https://arxiv.org/abs/2406.11815
- MimicPlay: https://arxiv.org/abs/2302.12422
- Data Scaling in IL: https://arxiv.org/abs/2410.18647
- Emergence of Human-to-Robot Transfer in VLA: https://arxiv.org/abs/2512.22414
- Sharpa Wave Hand: https://www.sharpa.com/pages/wave

这个 paper 真正的"philosophical statement"是 abstract 最后那句："pointing toward a future where humans can be treated as another scalable embodiment in robot learning."。把人当成 robot 的一种——和 wheeled humanoid、bipedal humanoid、parallel jaw arm 并列的 embodiment——这种 framing 把 robot learning 从 "收集 robot data" 转向 "收集任何 embodiment 的 manipulation data"。如果这个 framing 成立，那 robot learning 的 scaling 路径就和 LLM 一样清晰：更多 data → 更低 loss → 更好 performance，log-linear 可预测。
