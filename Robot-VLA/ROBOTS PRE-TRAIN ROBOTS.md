---
source_pdf: ROBOTS PRE-TRAIN ROBOTS.pdf
paper_sha256: 933dbb470afab713943c28968a17fb4deee88ee10956278a4a6e2cbb51915dd0
processed_at: '2026-08-12T02:04:27-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：Robots Pre-Train Robots

## 一句话概括

**让 robot 自己看自己的视频来学视觉特征，比看 human 的视频强多了，而且强了不是一点点。**

---

## 背景：之前大家在干嘛？

Robot manipulation 需要一个 visual encoder 把 image 变成 feature vector，然后 downstream policy 拿这个 feature 来决定 action。问题是 robot data 太少，pre-train 不动。

所以前几年大家的思路是：**借 human 的视频来 pre-train**。R3M 用 Ego4d，MVP 用 Something-Something，VC-1 加了 navigation data，HRP 学 human affordance。逻辑听起来很顺——human 抓东西、robot 抓东西，应该有共性。

但这帮人 never really 问过一个基本问题：**这些 pre-trained representation 到底在 image 上看哪里？**

---

## 第一个发现：看哪里决定了学多好

作者用 Grad-CAM 把每个 representation 的 attention map 画出来，结果一目了然：

- **MVP** 在做 Square task 的时候盯着桌子看，完全没看 gripper 和 object → 下游 success rate 垃圾
- **R3M** 在 Pick Place Wall 上也是盯着背景 → 垃圾
- 表现好的 representation 都在盯 end-effector 和 task-relevant object

作者把这个 property 叫 **Manipulation Centricity**，用 SAM 2 生成 ground truth mask 然后算 Jaccard Index 来量化。结果发现 **manipulation centricity 和 downstream success rate 的 Pearson correlation = 0.93**。

0.93 是什么概念？基本就是线性关系了。意思是：**你 encoder 关注的地方对不对，基本上就能预测 downstream policy 能不能学好，都不用跑下游 task**。

这个 metric 本身就是个 contribution，以后 PVR 论文都应该 report 这个。

---

## 第二个发现：Human data 有两个硬伤

**硬伤 1：Embodiment gap**
Human hand 和 robot gripper 长得完全不一样。你从 human video 学到的 "抓东西应该看这里" 换到 robot 上可能 "这里" 就不对了。

**硬伤 2：没有 dynamics labels**
Human video 只有 image sequence，没有 robot proprioceptive state (joint position, gripper position)，没有 action (delta pose)。这些 information 在 robot data 里是 free 的，human data 里完全没有。

作者做了个极 clean 的实验：**拿 R3M 的 loss 不变，只把 pre-training data 从 Ego4d 换成 DROID (76k robot trajectories)，叫 R3M-DROID**。结果 manipulation centricity 和 downstream performance 都涨了。

这 isolated 了 data source 这一个变量，证明 robot data 本身就比 human data 好，跟用什么 loss 无关。

---

## 第三个发现：怎么更好地用 Robot data？

Robot data 比 human data 多了 dynamics labels (state + action)，之前没人用。MCR 的核心就是把这些 free labels 用起来。

### MCR 的三个 Loss

**Loss 1: Dynamics Alignment (核心创新)**

每一帧 image $I_t$ 对应一个 robot state $s_t$ 和 action $a_t$。作者定义了一个 **state-action chunk**：

$$d_t = [s_{t-1}, a_{t-1}, s_t, a_t, s_{t+1}]$$

(l=3 的时候，包含 3 个 state 和 2 个 action)

然后用 InfoNCE contrastive loss 让 image feature $z_t$ 和对应的 dynamics chunk $d_t$ 在 embedding space 对齐，negative sample 是同一 trajectory 不同 timestep 的 chunk。

公式：
$$\mathcal{L}_{\text{dyn}} = -\sum_{b \in \mathcal{B}} \log \frac{e^{S(z_t^b, H(d_t)^b)}}{e^{S(z_t^b, H(d_t)^b)} + e^{S(z_t^b, H(d_k)^b)}}$$

- $z_t^b = \mathcal{F}_\phi(I_t^b)$：image feature
- $H(d_t)$：MLP 把 dynamics chunk 映射到和 image feature 同维度
- $S(\cdot, \cdot)$：negative L2 distance
- $d_t$：positive，同一 timestep 的 chunk
- $d_k$：negative，同 trajectory 不同 timestep 的 chunk

**直觉**：这个 loss 让 encoder 学到 "what visual features correspond to my current state and action"。要回答这个问题，encoder 必须 focus 在 end-effector 和 object 上，因为只有这些区域的变化和 state-action 相关。背景和桌子跟 state-action 没关系，自然就被忽略。这就是 manipulation centricity 的来源。

为什么用 chunk 而不是 single state？单帧 state 只有 14D (DROID 的 cartesian + gripper position)，映射到 2048D image feature 会引入 noise。Chunk 给了更丰富的 dynamics context。Ablation 显示 l=3 最好 (83.2%)，l=1 差 (72.1%)，l=5 和 l=7 都掉到 76.8%。

**Loss 2: Action Prediction**

加个 shallow MLP head 让 image feature 直接 predict action：

$$\mathcal{L}_{\text{act}} = -\sum_{b \in \mathcal{B}} \text{MSE}(a_t^b, \hat{a}_t^b)$$

这个就是 BC-style auxiliary loss。逼 encoder 提取 action-relevant info。image feature → 50D (bottleneck with LayerNorm + Tanh) → 512 → 512 → 7D action。

**Loss 3: Time Contrastive (from R3M)**

采样 triplet $(I_u, I_v, I_w)$ where $u < v < w$，让 $z_u$ 和 $z_v$ 近，和 $z_w$ 及其他 video 的 frame 远。保留 R3M 的 temporal smoothness。

$$\mathcal{L}_{\text{tcl}} = -\sum_{b \in B} \log \frac{e^{S(z_u^b, z_v^b)}}{e^{S(z_u^b, z_v^b)} + e^{S(z_u^b, z_w^b)} + e^{S(z_u^b, z_u^{\neq b})}}$$

**Overall**: $\mathcal{L}_{\text{MCR}} = \mathcal{L}_{\text{dyn}} + \mathcal{L}_{\text{act}} + \mathcal{L}_{\text{tcl}}$，没有 weighting，直接加。

### Ablation 结果

- Full MCR: **83.2%**
- 去掉 $\mathcal{L}_{\text{dyn}}$: 66.2% (掉 17%，最重要)
- 去掉 $\mathcal{L}_{\text{act}}$: 71.3% (掉 12%)
- 去掉 $\mathcal{L}_{\text{tcl}}$: 72.0% (掉 11%)

三个 loss 都有用，dynamics alignment 贡献最大。

---

## 结果有多猛？

### Simulation: 20 tasks across 4 domains

| Domain | Robot | End-effector |
|--------|-------|--------------|
| Robomimic | Franka Panda | Gripper |
| RoboCasa | Franka Panda | Gripper |
| MetaWorld | Sawyer | Gripper |
| DexArt | XArm6 | Allegro Hand |

MCR 在 **19/20** tasks 上 best，平均涨 **14.8%** over strongest baseline。在 DexArt (dexterous hand) 上也 work，虽然 DROID 全是 gripper data——说明 dynamics alignment 学的更偏 "state-action-image correspondence" 而非 "visual appearance"，所以能跨 embodiment。

### Real Robot: 3 tasks on UR5e

| Task | LfS | MVP | VC-1 | R3M | **MCR** |
|------|-----|-----|------|-----|---------|
| Lift | 5/10 | 6/10 | 5/10 | 6/10 | **9/10** |
| Sweep | 3/10 | 1/10 | 2/10 | 1/10 | **7/10** |
| Rearrange | 2/10 | 3/10 | 3/10 | 4/10 | **7/10** |
| **Total** | 10/30 | 10/30 | 13/30 | 11/30 | **23/30** |

23/30 vs 最好的 13/30，提升 **76.9%**。Real world gap 这么大很罕见。

### Training Efficiency

MCR: RTX 3090 Ti, ~50 hours
R3M: Tesla V100, ~120 hours

又快又好。

---

## 我的 Intuition 和几个 Critical Thoughts

### 为什么 Dynamics Alignment 这么强？

这个 loss 本质是在学 **"视觉感知 ↔ 本体感觉" 的对应关系**。Robot 做任务时，视觉看到的 gripper 位置和 proprioception 读出来的 cartesian position 应该是一一对应的。学这个对应关系，encoder 自然会 focus 在 gripper 上，因为只有 gripper 的视觉变化和 state 严格相关。背景、桌子、远处的东西跟 state 没关系，自然被过滤掉。

这其实有点像人类 sensorimotor 系统的机制——visual cortex 和 motor cortex 之间有很强的 cross-talk，视觉信息直接服务于 action。MCR 用 contrastive loss 实现了这个 cross-talk。

### 为什么 Action Prediction 比 Time Contrastive 重要？

Time contrastive 只要求 "nearby frames 在 embedding space 近"，这个 constraint 太弱了。一个 encoder 把所有 image map 到同一个点都满足这个 loss。

Action prediction 要求 feature 能 decode 出 action，这个信息量更大。但 action prediction 只学 current frame → action，是 unidirectional 的。

Dynamics alignment 学的是 image ↔ state-action chunk 的 **bidirectional** correspondence，信息量最大，所以贡献最多。

### Embodiment Gap 的微妙之处

DexArt 上 R3M-DROID 反而不如 R3M (Figure 10)。因为 DROID 全是 gripper，和 dexterous hand 有 embodiment gap。但 MCR 仍然优于 R3M，因为 dynamics alignment 学的是 "image 和 state 的对应"，这个对应在 gripper 和 hand 上都存在 (虽然具体 morphology 不同)。

这暗示：**如果你有 dexterous hand 的 robot data，MCR 应该能做得更好**。Open X-Embodiment 扩展版加更多 end-effector 类型可能是个 promising direction。

### 几个我没搞明白的地方

1. **Camera view mismatch**: DROID 有两个 external camera，downstream 大多是 single camera。Cross-view generalization 怎么 work 的？Paper 没讲清楚。

2. **Action space mismatch**: DROID action 是 7D，DexArt 是 22D。Action prediction head 在 pre-training 时是 7D，downstream 怎么用？应该是 pre-training 完只保留 encoder，head 扔掉，但 dynamics alignment 的 state alignment 跨 embodiment 怎么保证？

3. **Manipulation Centricity 的 causality**: R=0.93 是 correlation。会不会是 third factor (比如 feature norm distribution, feature diversity) 同时 cause 两者？Burns et al. 2023 发现 "emergent segmentation ability" 也是 key factor，这俩 metric 会不会 high correlate？

4. **BC only**: 所有 downstream 都用 BC。如果用 RL fine-tuning，manipulation centricity 还是不是好 indicator？Open question。

---

## Big Picture: 这篇 paper 的 position

Robot learning 现在有两条路线：

**Route A: End-to-end VLA** (OpenVLA, RT-X, Octo)
用 millions of robot episodes 训一个 huge transformer，image+language → action。Powerful 但 data-hungry，fine-tuning 难。

**Route B: Modular PVR + small BC head** (R3M, VC-1, MCR)
Pre-train 一个 visual encoder，freeze 住，downstream 用少量 data 训小 policy。Sample-efficient，practical for real deployment。

MCR 是 Route B 的 SOTA。它证明了：**robot data + dynamics labels 能训出远超 human-data PVR 的 representation**。

如果未来 Open X-Embodiment scale 的 data + MCR-style dynamics pre-training，可能会达到一个 sweet spot：比 VLA sample-efficient，比之前 PVR 强得多。

---

## 最核心的 Takeaway

1. **看哪里比学什么更重要** — Manipulation centricity R=0.93 说明 representation 的 attention pattern 基本决定 downstream performance
2. **Robot data 自带 dynamics labels，是免费的 dense supervision** — 之前 PVR 都浪费了
3. **Dynamics alignment loss 是核心** — image feature 和 state-action chunk 对齐，逼 encoder focus 在 manipulation-relevant region
4. **50 hours pre-training + 30 demos real deployment** — 这个 efficiency 对 real-world 很友好

参考链接：
- Project: https://robots-pretrain-robots.github.io/
- DROID: https://droid-dataset.com/
- R3M: https://arxiv.org/abs/2203.12601
- VC-1: https://arxiv.org/abs/2210.12950
- MVP: https://arxiv.org/abs/2203.06173
- HRP: https://arxiv.org/abs/2405.06107
- SAM 2: https://arxiv.org/abs/2408.00714
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- OpenVLA: https://openvla.github.io/
- Octo: https://octo-models.github.io/

---

# Robots Pre-Train Robots: 深度解析

## 1. Paper 的核心问题与动机

这篇 paper 来自 UCSD / Tongji / SJTU / UMD / Tsinghua 的合作,主要针对 **robotic visual representation pre-training** 这个领域的一个根本性问题:之前的 pre-trained visual representations (PVRs) 比如 [R3M](https://arxiv.org/abs/2203.12601)、[VC-1](https://arxiv.org/abs/2210.12950)、[MVP](https://arxiv.org/abs/2203.06173)、[HRP](https://arxiv.org/abs/2405.06107) 都是用 **human videos** (Ego4d, Something-Something, etc.) 来 pre-train。这个做法看起来合理 — human manipulation 应该蕴含 affordance、contact、object interaction 等信息。但是作者发现,这些 representation 在下游 robot manipulation tasks 上表现很不一致,有的甚至不如 learning-from-scratch。

核心 insight 非常 intuitive:**representation 的质量取决于它是否"看着"对的地方**。如果一个 encoder 关注的是桌子、背景、irrelevant objects,那 downstream policy 就很难学得好。作者把这个 property 称为 **Manipulation Centricity** — 即 representation 是否聚焦于 end-effector 和 task-relevant objects。

---

## 2. Manipulation Centricity:一个 Evaluation Metric

### 2.1 怎么测量?

作者用 [Grad-CAM](https://arxiv.org/abs/1610.02391) 生成 attention heatmap,然后用 [SAM 2](https://arxiv.org/abs/2408.00714) 分割出 ground truth manipulation region (end-effector + objects),计算两者的 **Jaccard Index** 作为 manipulation centricity 的度量。

具体 protocol:
1. 对 frozen encoder $\mathcal{F}_\phi$ 接一个 downstream policy head,用 BC 训练
2. 用 Grad-CAM 计算 input image 对 predicted action 的 gradient,得到 heatmap $G \in \mathbb{R}^{H \times W}$
3. Binarize: $G_{\text{bin}}(i,j) = \mathbb{1}[G(i,j) \geq 2]$ (threshold = 2,mitigate noise)
4. SAM 2 生成 ground truth mask $M \in \{0, 1\}^{H \times W}$
5. $\text{MC} = \frac{1}{N} \sum_{n=1}^{N} \frac{|G_{\text{bin}}^{(n)} \cap M^{(n)}|}{|G_{\text{bin}}^{(n)} \cup M^{(n)}|}$

### 2.2 关键发现:Pearson R = 0.93

Figure 2 中,manipulation centricity 与 downstream success rate 的 Pearson correlation coefficient **R = 0.93**,这是个非常强的 correlation。说明:

> **你 pre-train 的 representation 关注哪里,基本上就决定了下游 policy 能学多好。**

这个发现本身就很 valuable — 之前 PVR 文献都只 report downstream success rate,没有 probe 过 representation 内部到底在 attend 什么。这篇文章给出了一个 **predictive metric**,可以在不跑下游 task 的情况下评估 representation quality。

对 ViT 的细节:Grad-CAM 对 ViT 不能直接用最后一层,因为 class token 不依赖 spatial channels,gradient 为 0。作者选择 **last transformer block 中 self-attention 之后的 LayerNorm** 作为 target layer。这是个很实用的 trick,值得记住。

---

## 3. 为什么 Human Data 不够?为什么 Robot Data 更好?

### 3.1 Distribution Shift + 缺 Dynamics

Human videos 有两个根本问题:
1. **Embodiment gap**: human hand vs robot gripper 形态差异大,visual feature 学到的 grasp pattern、affordance 不能直接迁移
2. **没有 dynamics labels**: human video 没有 action (joint torque / delta pose) 和 proprioceptive state,所以只能学 visual-temporal 信息,学不到 "this image corresponds to this robot state" 的 grounding

### 3.2 R3M-DROID 实验:证明 Robot Data 本身就更好

作者做了个非常 clean 的 ablation:拿 R3M 的 loss function 不变,只把 pre-training data 从 Ego4d 换成 [DROID](https://arxiv.org/abs/2403.12945),得到 R3M-DROID。结果:
- Manipulation centricity 提升
- Downstream success rate 提升 (Figure 2)

这个实验很 powerful,因为它 **isolate 了 data source 这一个变量**。说明同一个算法,robot data 就比 human data 好。

### 3.3 DROID 数据处理

- 原始 76k trajectories (1.7TB RLDS format)
- Filter: trajectory length ≥ 40 timesteps (确保 temporal info 充足)
- Filter: 剔除 incomplete / single-word language instructions
- 最终保留 **36k trajectories**
- Franka arm + Robotiq 2F-85 gripper
- 两个 Zed 2 external cameras (stereo,RGB)
- Action space: delta 6D pose + 1-DoF gripper = 7D

---

## 4. MCR 方法的三个 Loss

这是 paper 的 method 核心。MCR (Manipulation Centric Representation) 用 ResNet-50 backbone,在 DROID 上 pre-train 500k steps,~50 hours on RTX 3090。

### 4.1 Dynamics Alignment Loss $\mathcal{L}_{\text{dyn}}$ — 核心 contribution

**Insight**: 每一帧 image $I_t$ 对应一个 robot proprioceptive state $s_t$ 和 action $a_t$。我们希望 image feature $z_t = \mathcal{F}_\phi(I_t)$ 和 state-action dynamics chunk $d_t$ 在 embedding space 对齐。

定义 **state-action dynamic chunk** of length $l$ at timestep $t$:

$$d_t = [s_{[t - l/2]}, a_{[t - l/2]}, s_{[t - l/2 + 1]}, \ldots, s_{\lfloor t + l/2 \rfloor}]$$

变量说明:
- $s_t$: proprioceptive state (cartesian position + gripper position,14D in DROID)
- $a_t$: action (7D: delta 6D pose + 1-DoF gripper)
- $l$: chunk length,ablation 显示 $l = 3$ 最优
- 注意 chunk 的结构是 state-action-state-...-state,所以 length $l$ 包含 $l$ 个 state 和 $l-1$ 个 action

**为什么用 chunk 而不是 single state?** 单帧 state 维度太低 (14D),通过 MLP projector 映射到 2048D (ResNet-50 output dim) 会引入 noise。Chunk 提供更丰富的 dynamics context,让 contrastive learning 更 stable。Ablation 显示:
- $l=1$: 72.1% (差,信息不足)
- $l=3$: **83.2% (最优)**
- $l=5$: 76.8%
- $l=7$: 76.8%

长 chunk 反而降性能,因为 dynamics modeling 变难。

**Loss formulation** (InfoNCE-style):

$$\mathcal{L}_{\text{dyn}} = -\sum_{b \in \mathcal{B}} \log \frac{e^{S(z_t^b, H(d_t)^b)}}{e^{S(z_t^b, H(d_t)^b)} + e^{S(z_t^b, H(d_k)^b)}}$$

变量:
- $b$: batch 中的一个 sample
- $\mathcal{B}$: batch
- $z_t^b = \mathcal{F}_\phi(I_t^b)$: image feature
- $H(\cdot)$: MLP projector,把 dynamics chunk 映射到 image feature 同维度
- $S(\cdot, \cdot)$: **negative L2 distance** as similarity (i.e., $S(x, y) = -\|x - y\|_2$)
- $d_t$: positive dynamics chunk (same timestep as $I_t$)
- $d_k$: negative dynamics chunk,从同一 trajectory 的不同 timestep $k$ 采样

**关键设计**: negative sample 来自 **同一 trajectory 的不同 timestep**,这是为了避免 encoder 只用 video identity 来区分,强迫它关注 state-action 本身的信息。

实现细节 (from Appendix A.4):
```python
state_input_dim = 14 * state_chunk_length  # state 部分
state_input_dim += 7 * (state_chunk_length - 1)  # action 部分
# 例如 l=3: 14*3 + 7*2 = 42 + 14 = 56
state_encoder = nn.Sequential(
    nn.Linear(state_input_dim, 1024),
    nn.ReLU(),
    nn.Linear(1024, self.outdim)  # outdim = 2048 for ResNet50
)
```

### 4.2 Action Prediction Loss $\mathcal{L}_{\text{act}}$ — BC-style auxiliary

让 image feature 直接 predict action,这是个 BC-style auxiliary loss:

$$\mathcal{L}_{\text{act}} = -\sum_{b \in \mathcal{B}} \text{MSE}(a_t^b, \hat{a}_t^b)$$

其中 $\hat{a}_t = \text{MLP}(z_t)$。

Actor head 实现:
```python
actor_trunk = nn.Sequential(
    nn.Linear(self.outdim, 50),  # 2048 -> 50
    nn.LayerNorm(50),
    nn.Tanh()
)
actor_policy = nn.Sequential(
    nn.Linear(50, 512),
    nn.ReLU(inplace=True),
    nn.Linear(512, 512),
    nn.ReLU(inplace=True),
    nn.Linear(512, action_dim)  # action_dim = 7 for DROID
)
```

注意 trunk 用了 **LayerNorm + Tanh** 把 high-dim feature 压到 50D,这是个 bottleneck,强迫 feature 提取 action-relevant 的 essential info。

这个 loss 的作用是 **直接 inject action 信号到 representation learning**,让 encoder 知道 "what to look at in order to act"。和 dynamics alignment 互补:alignment 让 feature 与 state 对应,action prediction 让 feature 能推出 action。

### 4.3 Time Contrastive Loss $\mathcal{L}_{\text{tcl}}$ — from R3M

借用 R3M 的 temporal contrastive loss,鼓励 temporal close frames 在 embedding space 接近:

$$\mathcal{L}_{\text{tcl}} = -\sum_{b \in B} \log \frac{e^{S(z_u^b, z_v^b)}}{e^{S(z_u^b, z_v^b)} + e^{S(z_u^b, z_w^b)} + e^{S(z_u^b, z_u^{\neq b})}}$$

变量:
- Triplet $(I_u, I_v, I_w)$ where $u < v < w$ (temporal order)
- $z_u^b, z_v^b$: positive pair (temporally close)
- $z_w^b$: hard negative (same video, temporally distant)
- $z_u^{\neq b}$: negative from **different video** in batch

这个 loss 保留了 R3M 的 temporal smoothness property,确保 representation encode 时间结构。

### 4.4 Overall Objective

$$\mathcal{L}_{\text{MCR}} = \mathcal{L}_{\text{dyn}} + \mathcal{L}_{\text{act}} + \mathcal{L}_{\text{tcl}}$$

**没有 weighting hyperparameter**,直接相加。作者说 empirical 上已经足够好。这其实有点 risky — 不同 loss 的 scale 可能差很多,ablation 显示三个 loss 都重要:
- Full MCR: 83.2%
- w/o $\mathcal{L}_{\text{dyn}}$: 66.2% (掉 17%,最重要)
- w/o $\mathcal{L}_{\text{act}}$: 71.3% (掉 12%)
- w/o $\mathcal{L}_{\text{tcl}}$: 72.0% (掉 11%)

可以看出 dynamics alignment 是核心 innovation,action prediction 次之,time contrastive 提供稳定的 temporal grounding。

---

## 5. 实验结果分析

### 5.1 Simulation Results (20 tasks across 4 domains)

| Domain | Robot | End-effector | Demo count |
|--------|-------|--------------|------------|
| [Robomimic](https://arxiv.org/abs/2108.03275) | Franka Panda | Parallel gripper | 200/task |
| [RoboCasa](https://arxiv.org/abs/2406.02524) | Franka Panda | Parallel gripper | 50/task |
| [MetaWorld](https://arxiv.org/abs/1910.10897) | Sawyer | Parallel gripper | 25/task |
| [DexArt](https://arxiv.org/abs/2303.11582) | XArm6 | Allegro hand | 100/task |

**关键结论**:
1. MCR 在 **19/20** tasks 上达到 best performance (Table 7)
2. 平均提升 **14.8%** over strongest baseline
3. 在 DexArt (dexterous hand) 上也 work — 虽然 DROID 是 gripper data,说明 representation 学到的 manipulation concept 有一定 generalization
4. 在 MetaWorld (只有 25 demos) 上 baseline 都很差,但 MCR 仍然 strong — 说明 manipulation-centric representation 减轻了 downstream policy 的 learning burden

### 5.2 Real Robot Results

Setup: UR5e arm + Robotiq 2F-85 gripper + RealSense D435i camera。3 个 task:
- **Lift**: 抓 sandbag 提起来 (30 demos)
- **Sweep**: 抓扫帚把垃圾扫到簸箕 (40 demos)
- **Rearrange**: 抓 pot 放到 stove 指定位置 (40 demos)

Results (Table 2, 10 trials each):

| Task | LfS | MVP | VC-1 | R3M | **MCR** |
|------|-----|-----|------|-----|---------|
| Lift | 5/10 | 6/10 | 5/10 | 6/10 | **9/10** |
| Sweep | 3/10 | 1/10 | 2/10 | 1/10 | **7/10** |
| Rearrange | 2/10 | 3/10 | 3/10 | 4/10 | **7/10** |
| **All** | 10/30 | 10/30 | 13/30 | 11/30 | **23/30** |

总体 **76.9% improvement** over best baseline (13/30 → 23/30)。这个 gap 非常大,而且 baseline 在 unseen object positions 上经常 fail to grasp,MCR 表现 robust。

### 5.3 Compute Efficiency

| Method | GPU | Training Time |
|--------|-----|---------------|
| R3M | Tesla V100 | ~120h |
| **MCR** | RTX 3090 Ti | ~50h |

MCR 比 R3M 训练更快,而且性能更好。这是因为 R3M 需要 video-language alignment (额外 CLIP-style loss),而 MCR 用 dynamics alignment 更 sample-efficient。

---

## 6. 几个值得深挖的 Insights

### 6.1 Scaling: Larger Dataset = Better Performance (Figure 9)

把 DROID 每个 scene 的数据从 100% 减到 25%,MCR 性能单调下降。这与之前 [Dasari et al. 2023](https://arxiv.org/abs/2310.00044) 的发现 "merely increasing dataset size 不 work" 矛盾。作者的解释:
1. 用的是 robot data (不是 human data)
2. 用了 dynamics labels (不只是 visual contrastive)

这说明 **dynamics information 是 scaling 的关键** — 没有 dynamics signal,光堆 data 没用;有了 dynamics,数据越多越好。

### 6.2 Embodiment Gap 分析 (Figure 10)

把 task 按 end-effector 分两类:
- **Gripper-based tasks**: DROID pre-training 帮助明显 (R3M-DROID 和 MCR 都超过 R3M)
- **Dexterous hand tasks** (DexArt): R3M-DROID 反而 **不如** R3M!MCR 仍优于 R3M 但优势变小

原因:DROID 全是 gripper data,与 dexterous hand 有 embodiment gap。但 MCR 因为用了 dynamics alignment (学的是 state-action-image 的 correspondence,而不是 visual appearance),所以仍然能 transfer。

这个发现指向 future work:**需要更多 dexterous hand data 加入 large-scale robot dataset**,比如 [Open X-Embodiment](https://robotics-transformer-x.github.io/) 的扩展版。

### 6.3 t-SNE Feature Visualization (Figure 11)

在 MetaWorld 10 tasks + 3 real tasks 上做 t-SNE:
- **R3M**: 任务内 clustering 差,任务间混在一起
- **R3M-DROID**: clustering 改善,但 DROID-sim gap 导致很多 task 难区分
- **MCR**: clustering 最好,任务边界清晰

这说明 MCR 学到的 representation 真的 capture 了 task-discriminative information,而这是通过 dynamics labels 注入的。

### 6.4 Encoder Scaling (Table 4)

| Backbone | Success Rate |
|----------|--------------|
| ResNet-18 | 77.3% |
| ResNet-34 | 77.9% |
| **ResNet-50** | **83.2%** |

Larger encoder 更好,说明 MCR 可以 scale。这与 [Theia](https://arxiv.org/abs/2410.21252) 的发现一致 (high entropy in feature norm distribution correlates with performance)。

---

## 7. 与 Related Work 的关系

### 7.1 Pre-trained Visual Representations (PVRs)

| Method | Data | Loss | Key Idea |
|--------|------|------|----------|
| [MVP](https://arxiv.org/abs/2203.06173) | Human videos + ImageNet | MAE | Masked autoencoding |
| [VC-1](https://arxiv.org/abs/2210.12950) | Human videos + Nav + ImageNet | MAE | MVP + navigation |
| [R3M](https://arxiv.org/abs/2203.12601) | Ego4d + Something-Something | Time contrastive + VL alignment | Temporal + semantic |
| [HRP](https://arxiv.org/abs/2405.06107) | Human videos | Affordance prediction | Hand pose, contact, active object |
| **MCR** | DROID (robot) | Dyn align + Act pred + TCL | **Dynamics-aware, robot-specific** |

MCR 的独特之处:**第一个用 large-scale robot data + dynamics labels 做 PVR**。

### 7.2 Generalist Robot Policies

| Method | Approach |
|--------|----------|
| [RT-X](https://robotics-transformer-x.github.io/) | Train on Open X-Embodiment |
| [Octo](https://octo-models.github.io/) | Transformer diffusion policy |
| [OpenVLA](https://openvla.github.io/) | VLA model |
| **MCR** | **Frozen representation + task-specific BC head** |

MCR 的定位不同 — 它不学 generalist policy,而是学 **generalist representation**,然后让 downstream 用少量 data 训 task-specific policy。这是更 sample-efficient 的路线,适合 real-world deployment where data 有限。

### 7.3 Dynamics-aware Representation Learning

- [CURL](https://arxiv.org/abs/2004.04136): InfoNCE on augmented observations
- [CPC](https://arxiv.org/abs/1807.03748): Contrastive Predictive Coding
- [ATC](https://arxiv.org/abs/2104.12868): Augmented Temporal Contrastive
- [TACO](https://arxiv.org/abs/2310.15860): Temporal Action-driven Contrastive Loss
- [RPT](https://arxiv.org/abs/2306.10007): Robot Pre-Training with trajectory labels (Ilija Radosavovic)

MCR 与 RPT 思路类似,但 MCR 的创新是 **manipulation centricity 这个 evaluation framework**,以及 dynamics alignment loss 的具体设计 (chunk-based InfoNCE)。

---

## 8. 我的 Intuition Building & 批判性思考

### 8.1 为什么 Manipulation Centricity 这么重要?

这其实呼应了 cognitive science 中的 **visual attention for action** 理论 — 人在做 manipulation task 时,visual attention 自动聚焦在 effector 和 object 上 (e.g., [Vision-for-Perception vs Vision-for-Action](https://www.nature.com/articles/nn1200)). 一个 "好的" representation 应该 implicitly 实现 this attention mechanism。MCR 通过 dynamics alignment,让 encoder 学到 "what visual features correlate with my state/action",自然就 focus 在 effector 和 object 上,因为这些是 dynamics 的视觉对应物。

### 8.2 为什么 Action Prediction 比 Time Contrastive 更重要?

Ablation 显示 $\mathcal{L}_{\text{act}}$ 掉 12%,$\mathcal{L}_{\text{tcl}}$ 掉 11%,差不多。但 $\mathcal{L}_{\text{dyn}}$ 掉 17% 最多。我的 interpretation:
- **Time contrastive** 只学 "nearby frames similar",太弱 — 任何 smooth representation 都能满足
- **Action prediction** 强迫 feature 含 action-relevant info,但只是 current frame → action 的 mapping
- **Dynamics alignment** 学的是 image ↔ state-action chunk 的 bidirectional correspondence,这要求 feature **同时** 含 visual appearance + dynamics state,信息量最大

### 8.3 潜在问题

1. **Single camera setup**: DROID 有两个 external cameras,MCR 用了两个 view。但 downstream (Robomimic, MetaWorld 等) 都是 single view。Cross-view gap 怎么处理?Paper 没说清楚。

2. **Action space mismatch**: DROID action = delta 6D pose + gripper (7D),但 DexArt action = 22D (Allegro hand joints)。Action prediction head 在 pre-training 时是 7D,downstream 22D 怎么办?应该是 pre-training 时 head 不用,只保留 encoder。但 dynamics alignment 的 state 也得 align,这个 cross-embodiment 的 dynamics 怎么 generalize?Paper 在 Section 5.4 提到 dexterous hand 上 advantage 变小,但没解释 mechanism。

3. **Manipulation Centricity 的因果性**: R = 0.93 是 correlation,不是 causation。有没有可能是 third factor (e.g., feature norm distribution, feature diversity) 同时 cause 两者?Burns et al. 2023 发现 "emergent segmentation ability" 也是关键 factor。这两个 metric 会不会 high correlate?需要 further experiment。

4. **BC 限制**: 所有 downstream 都用 BC,如果用 RL fine-tuning 会怎样?Manipulation centricity 还是不是好 indicator?Open question。

5. **Data scale**: 36k trajectories 已经很大,但 vs [Open X-Embodiment](https://robotics-transformer-x.github.io/) (millions of episodes) 还是小。MCR 能不能 scale 到 Open-X 上?compute 会不会爆炸?

### 8.4 对未来 Robotics 的启示

这篇 paper 给我几个 takeaways:

1. **Robot data > human data for robot representation** — 这个结论之前很多人怀疑,但缺乏 systematic comparison。这篇给了 clean evidence。下一步应该是 Open X-Embodiment 上的 MCR extension。

2. **Dynamics labels 是 game-changer** — 之前 PVR 都只用 visual signal (image-text, image-image contrastive),忽略了 robot 自带的 state-action。MCR 证明这些 "free" labels 信息量巨大。这个思路可以 extend 到 tactile data, force-torque data, etc.

3. **Manipulation Centricity 可以作为 representation 的 standard eval metric** — 类似 ImageNet accuracy for classification。以后 PVR 论文应该 report MC + downstream success rate。

4. **Frozen representation + small BC head 的路线 vs End-to-end VLA** — OpenVLA 这种 end-to-end 大模型 vs MCR 这种 modular approach,哪个更 sample-efficient?MCR 的 50h pre-training + 30 demos 真实部署,可能比 fine-tune 一个 VLA 更 practical。

5. **Future direction**: 把 MCR extension 到 video-language-action model,用 language 作为 additional supervision,可能在 task-level generalization 上有突破。

---

## 9. 总结

这篇 paper 做了三件事:
1. **提出 manipulation centricity metric**,发现它和 downstream performance 强相关 (R=0.93)
2. **证明 robot data 优于 human data** for robot representation pre-training
3. **设计 MCR**,用 dynamics alignment + action prediction + time contrastive 三个 loss,在 20 sim tasks + 3 real tasks 上显著超越 baseline

最让我兴奋的是 **dynamics alignment loss** 这个 idea — 把 image feature 和 state-action chunk 对齐,本质上是在学 "视觉感知 ↔ 本体感觉" 的 correspondence。这其实是 robot learning 中一直被忽略的 signal。之前大家都关注 visual contrastive learning,忘了 robot 自带的 proprioception 是个 dense supervision signal。

对 Andrej 来说,这可能在 nanoGPT 之后值得思考的方向:**什么样的 pre-training objective 最适合 embodied AI?** LLM 的 next-token prediction 显然不直接适用。MCR 的 dynamics alignment 是一种 "next-state-dynamics prediction" 的变体,但更结构化。未来可能 extends 到:
- **World model pre-training**: predict next image from (image, action)
- **Inverse dynamics**: predict action from (image_t, image_{t+1})
- **Multi-modal dynamics**: align image, language, audio, tactile in shared embedding

Reference links:
- Paper project page: https://robots-pretrain-robots.github.io/
- DROID dataset: https://droid-dataset.com/
- R3M: https://arxiv.org/abs/2203.12601
- VC-1: https://arxiv.org/abs/2210.12950
- MVP: https://arxiv.org/abs/2203.06173
- HRP: https://arxiv.org/abs/2405.06107
- SAM 2: https://arxiv.org/abs/2408.00714
- Grad-CAM: https://arxiv.org/abs/1610.02391
- InfoNCE / CPC: https://arxiv.org/abs/1807.03748
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- Octo: https://octo-models.github.io/
- OpenVLA: https://openvla.github.io/
- Robomimic: https://arxiv.org/abs/2108.03275
- RoboCasa: https://arxiv.org/abs/2406.02524
- MetaWorld: https://arxiv.org/abs/1910.10897
- DexArt: https://arxiv.org/abs/2303.11582
- Theia: https://arxiv.org/abs/2410.21252
- RPT: https://arxiv.org/abs/2306.10007
- TACO: https://arxiv.org/abs/2310.15860
- CURL: https://arxiv.org/abs/2004.04136
