---
source_pdf: Affordances from Human Videos as a Versatile Representation for Robotics.pdf
paper_sha256: 9b974dd07bde721599dd5acae76ed57be9cf3a96f63afbae480238f9d4a74ade
processed_at: '2026-07-18T03:20:08-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# VRB: Affordances from Human Videos as a Versatile Representation for Robotics 深度解析

## 1. Core Intuition - 这篇paper要解决什么根本问题

Andrej，这篇paper的核心insight非常深刻。当前robotics面临一个fundamental gap: computer vision能告诉你"这是door"，但robot需要知道"handle在哪里，往哪个方向pull"。VRB提出的解决方案是从大规模egocentric human videos中提取**actionable representation** - 具体就是contact points + post-contact trajectories，这个representation直接可以deploy到robot上。

paper开头引用Gibson的话非常key: "The meaning or value of a thing consists of what it affords... what we perceive when we look at objects are their affordances, not their qualities." 这里的affordance是Gibson 1979年提出的ecological psychology概念。

**为什么这个approach重要？** 大多数robot learning方法都从tabula rasa开始，最多用ImageNet pretrained encoder。但visual representation只是problem的一小部分。在continuous control中，state space complexity随actions指数增长。即使有perfect perception，知道"做什么"依然困难。VRB就是要bridge这个gap。

参考链接：
- Project page: https://vision-robotics-bridge.github.io/
- Gibson affordance theory: https://en.wikipedia.org/wiki/Affordance

## 2. Affordance Representation的数学形式化

### 2.1 核心定义

paper采用contact point $c$ 和post-contact trajectory $\tau$ 作为affordance representation，都在pixel space:

$$\tau = f(I_t, h_t)$$

变量解释：
- $I_t$: timestep $t$ 的input image
- $h_t$: human hand location in pixel space
- $f$: learned model
- $c$: contact point (2D pixel coordinates)
- $\tau$: post-contact trajectory (sequence of 2D points)

**关键insight**: 这个abstraction使得affordance prior agnostic to morphological differences across robots。人类morphology和robot morphology差异巨大，直接模仿human movement不会generalize。但"在哪里contact + contact后怎么move"这个abstraction是robot-agnostic的。

### 2.2 为什么选择这个representation

paper讨论了几个alternative:
- Heatmap of contact location (Hotspots [79]的做法)
- Pre/post condition of object
- Description of human interaction (HOI [66]的做法)

这些都不够actionable。VRB的representation直接对应robot的execution: move to $c$ → grasp → move along $\tau$。

## 3. Data Extraction Pipeline - 从Human Videos提取Supervision

### 3.1 整体流程

考虑video $V = \{I_1, ..., I_T\}$，目标是从中提取 $(c, \tau)$ labels来supervise predictive model $f_\theta(I_t)$。

**Step 1: Hand-Object Detection**
使用100 DOH model [106] (Shan et al., CVPR 2020)，对每个image $I_t$产生:
- $h_t$: 2D hand bounding box
- $o_t$: discrete contact variable

**Step 2: 找到first contact timestep**
$$t_{\text{contact}} = \min\{t : o_t = 1\}$$

**Step 3: 提取post-contact trajectory**
$$\tau = \{h_t\}_{t_{\text{contact}}}^{t'}$$

这是hand bounding box从contact开始到$t'$的sequence。

### 3.2 Contact Points的GMM建模

这是paper最elegant的部分之一。对contact points $\{c^i\}^N$，paper fit Gaussian Mixture Model来encourage multi-modality:

$$p(c) = \operatorname*{argmax}_{\mu_1, \dots, \mu_K, \Sigma_1, \dots, \Sigma_K} \sum_{i=1}^{N} \sum_{k=1}^{K} \alpha_k \mathcal{N}(c^i | \mu_k, \Sigma_k) \tag{1}$$

变量解释：
- $N$: contact points的数量 (per image, varies)
- $K$: GMM clusters数量 (paper用K=5)
- $\mu_k$: 第$k$个Gaussian component的mean
- $\Sigma_k$: 第$k$个Gaussian component的covariance matrix
- $\alpha_k$: 第$k$个component的weight
- $\mathcal{N}(c^i | \mu_k, \Sigma_k)$: Gaussian probability density

**为什么用GMM？** 因为一个scene可能有multiple possible interactions。比如一个cup可以从handle抓，也可以从rim抓。Mean或random sampling会lose这个multi-modality，GMM能preserve它。

### 3.3 Egomotion Compensation - 关键的工程细节

考虑人开门的场景：人的手move，body move，head也move（camera move）。所以需要compensate camera egomotion from $t_{\text{contact}}$ to $t'$。

使用homography matrix $\mathcal{H}_t$:
$$\tau = \mathcal{H}_t \circ \{h_t\}_{t_{\text{contact}}}^{t'}$$

$\mathcal{H}_t$通过matching features between consecutive frames获得。这把所有points project回starting frame的坐标系。

### 3.4 Human-Robot Domain Shift的巧妙处理

Training videos里有human body/hand，但robot deployment时没有human。这是domain shift。

VRB的trick: **extract affordances in frames with humans, but map back to first frame when human was yet to enter scene**。

具体做法：
- 如果video一开始没有human → 用第一帧作为conditioning image
- 如果human一直在frame → crop out human in initial frame (如果还没有interaction)
- 如果human一直in contact → discard这个frame

这个human-less frame就是affordance model的input。非常elegant的domain adaptation trick。

## 4. Model Architecture详解

### 4.1 整体架构

```
Input Image I_t
    ↓
ResNet18 Encoder (g_θ^conv)
    ↓
Spatial latent z_t
    ↓
    ├──→ Deconv layers (g_θ^deconv) → K heatmaps → Spatial Softmax σ_2D → μ_k (contact points)
    └──→ Transformer Encoder (T_θ) → τ (trajectory waypoints)
```

### 4.2 Contact Point Prediction Head

Visual encoder: $g_\theta^{\text{conv}}(I_t) = z_t$ (ResNet18, spatial features before avg pooling)

Deconv head: $H_t = g_\theta^{\text{deconv}}(z_t)$ → K probability distributions/heatmaps

Spatial softmax: $\sigma_{2D}$ 提取GMM means $\mu_k$

**Loss for contact point estimation:**
$$\mathcal{L}_{\text{contact}} = \left\| \mu_i - \sigma_{\text{2D}}\left(g_\theta^{\text{deconv}}\left(g_\theta^{\text{conv}}\left(I_t\right)\right)\right) \right\|_2 \tag{2}$$

变量解释：
- $\mu_i$: ground truth GMM means from Eq.(1)
- $\sigma_{2D}$: spatial softmax operation
- $g_\theta^{\text{deconv}}$: deconvolutional layers (channels: [256, 128, 64, 10, 5])
- $g_\theta^{\text{conv}}$: ResNet18 encoder

**Implementation detail**: paper发现keeping covariance matrices fixed gave better results，只学习$\mu_k$。

### 4.3 Trajectory Prediction Head

使用Transformer encoder:
- 6 self-attention layers
- 8 heads each
- Input: flattened ResNet18 output (dim 512)
- Output: trajectory of length 5 via MLP (2 layers, hidden 192)

**Key insight**: 优化relative shifts比absolute locations更容易。假设第一个point $\hat{w}_0 = 0$，因为contact points已经spatially grounded了。

**Loss:**
$$\mathcal{L}_{\text{traj}} = \|\tau - \mathcal{T}_\theta(z_t)\|_2$$

### 4.4 Training Details

- Dataset: EpicKitchens-100 [20] (~54K image-trajectory-contact tuples)
- Crops of size 150×150 (full image 456×256)
- 500 epochs, lr=0.0001, cosine scheduling
- ADAM optimizer
- 4 GPUs (2080Ti), ~18 hours
- K=5 GMM clusters

参考：EpicKitchens dataset: https://epic-kitchens.github.io/2022

## 5. Four Robot Learning Paradigms - VRB的versatility

这是paper最impressive的部分 - VRB不只是单一method，是versatile representation能bootstrap多种learning paradigm。

### 5.1 Paradigm A: Imitation Learning from Offline Data Collection

**Setup**: 用affordance model $f_\theta(I_t)$ guide robot collect data，存 $\{(I_t, (c, \tau))\}$ 到dataset $\mathcal{D}$。

**Task specification via goal images**: 给定goal image，用k-NN在feature space找closest trajectories。

**Algorithm 1详解**:
```
For each trajectory τ, compute:
    d_T = min_i ||ψ(I_g) - ψ(O_i)||_2^2

Rank trajectories by d_T (ascending)
Take top K trajectories

if k-NN:
    Execute K trajectories on robot
else (behavior cloning):
    Train policy π(c,τ|I) on K trajectories
    Execute c,τ ~ π(·|I) on robot
```

变量：
- $\psi$: R3M embedding space [83]
- $O_i$: intermediate images during trajectory execution
- $I_g$: goal image
- $K$: 10 for k-NN, 20 for BC

**BC policy parameterization**: CVAE, encoder/decoder是2-layer MLPs (64 hidden, latent dim 4)

### 5.2 Paradigm B: Reward-Free Exploration

**Motivation**: 从scratch exploration在real world太inefficient。VRB能bootstrap exploration from predicted affordances。

**Exploration metric**: Environment Change (EC)
$$\text{EC}(I_i, I_j) = \|\phi(I_i) - \phi(I_j)\|_2$$

其中$\phi$ masks robot，只在non-masked pixels上计算loss。

**Algorithm 2核心**:
1. Initial: 用$f_\theta$ collect $N_0$ trajectories
2. For iteration 1:J:
   - Rank trajectories by EC (descending)
   - Take top K, fit distribution $h$ to their $(c, \tau)$
   - For $N_s$ iterations: with probability $p$ use $f_\theta$, otherwise sample from $h$
   - Append new data to $\mathcal{D}$

参数: $p=0.35$, $K=10$, $J=2$

**Intrinsic Reward Model** (Eq.3):
$$\phi(I_i, I_j) = g\left(\|m(I_i) - m(I_j)\|_2, \|\Psi(m(I_i)) - \Psi(m(I_j))\|_2\right)$$

- $m$: masking network (removes robot, trained on 100-200 annotations)
- $\Psi$: pretrained segmentation model (Mask R-CNN [44])
- $g$: heuristics (gaussian blurring, threshold)

### 5.3 Paradigm C: Goal-Conditioned Learning

类似exploration，但metric是minimize distance to goal image:
$$d_T = \min_i \|\psi(I_g) - \psi(O_i)\|_2^2$$

Rank trajectories by $d_T$ (ascending), fit $h$ to top trajectories' $(c, \tau)$。

### 5.4 Paradigm D: Affordance as Action Space

**Motivation**: Robot需要operate in continuous action space，难以optimize。Discretize成spatial manner + assign primitive to each location。

**Algorithm 3**:
1. Query $f_\theta$ on scene $q$ times (q=2000) → dataset $\{(c, \tau)\}$
2. Fit GMM $G_c$ with $N_c$ centers to $\{c\}$
3. Fit GMM $G_\tau$ with $N_\tau$ centers to $\{\tau\}$
4. Create discrete action space $\mathcal{A} = [1..N_c \times N_\tau]$
5. Run DQN [76] on this discrete action space

参数: $N_c = N_\tau = 4$, so action space = 16 discrete actions

**Reward**: $r = \|\psi(I_T) - \psi(I_g)\|_2$

## 6. Experimental Results - 深度分析

### 6.1 Setup
- **Robots**: Franka Emika Panda + Hello Stretch mobile manipulator
- **Tasks**: 10 real-world tasks across 4 environments
- **In-the-wild**: 很多tasks在lab外
- **Scale**: 200+ hours robot running time

### 6.2 Table 1: Imitation Learning Results

| Method | Cabinet | Knife | Veg | Shelf | Pot | Door | Lid | Drawer | **Avg** |
|--------|---------|-------|-----|-------|-----|------|-----|--------|---------|
| **k-NN** | | | | | | | | | |
| HOI | 0.2 | 0.1 | 0.1 | 0.6 | 0.0 | 0.4 | 0.0 | 0.6 | 0.25 |
| HAP | 0.3 | 0.0 | 0.3 | 0.0 | 0.1 | 0.2 | 0.0 | 0.1 | 0.13 |
| Hotspots | 0.4 | 0.0 | 0.1 | 0.0 | 0.5 | 0.4 | 0.3 | 0.5 | 0.28 |
| Random | 0.3 | 0.0 | 0.1 | 0.3 | 0.4 | 0.2 | 0.1 | 0.2 | 0.20 |
| **VRB** | **0.6** | **0.3** | **0.6** | **0.8** | **0.4** | **1.0** | **0.4** | **1.0** | **0.64** |
| **BC** | | | | | | | | | |
| HOI | 0.3 | 0.0 | 0.3 | 0.0 | 0.1 | 0.2 | 0.0 | 0.1 | 0.13 |
| HAP | 0.5 | 0.0 | 0.4 | 0.0 | 0.3 | 0.1 | 0.0 | 0.1 | 0.18 |
| Hotspots | 0.2 | 0.0 | 0.0 | 0.0 | 0.8 | 0.1 | 0.0 | 0.7 | 0.23 |
| Random | 0.1 | 0.1 | 0.1 | 0.0 | 0.2 | 0.1 | 0.0 | 0.0 | 0.08 |
| **VRB** | **0.6** | 0.1 | **0.3** | **0.3** | **0.8** | **0.9** | **0.2** | **0.9** | **0.51** |

**Key observations**:
- VRB在k-NN上avg 0.64 vs runner-up Hotspots 0.28 (2.3x improvement)
- VRB在BC上avg 0.51 vs runner-up Hotspots 0.23 (2.2x improvement)
- 在Door和Drawer上VRB达到1.0 success rate
- 这说明VRB collect的data quality远高于baselines

### 6.3 Table 2: VRB vs R3M Visual Representation

| Task | VRB | R3M |
|------|-----|-----|
| microwave | 0.16 | 0.10 |
| slide-door | 0.84 | 0.70 |
| door-open | 0.13 | 0.11 |

VRB在所有task上outperform R3M。Important: VRB只finetune 2K steps (vs R3M paper的20K)。这说明affordance training学到了useful visual representation for control作为byproduct。

### 6.4 Table 3: Rare Objects Grasping

| Object | VRB | Hotspots |
|--------|-----|----------|
| VR Controller | 0.27 | 0.13 |
| Chain | 0.33 | 0.20 |
| Hat | 0.07 | 0.20 |
| Tape | 0.13 | 0.00 |
| Cube | 0.00 | 0.00 |
| Sanitizer | 0.27 | 0.20 |
| Stapler | 0.53 | 0.20 |
| Shoe | 0.33 | 0.13 |
| Mouse | 0.27 | 0.00 |
| Hair-Clip | 0.47 | 0.20 |

VRB在大多数rare objects上outperform Hotspots，说明generalization能力。

### 6.5 Table 5: Simulation Benchmark

| Method | Light | Microwave | Kettle |
|--------|-------|-----------|--------|
| Random | 0.20 | 0.15 | 0.20 |
| HAP | 0.30 | 0.20 | 0.45 |
| HOI | 0.60 | 0.45 | 0.40 |
| Hotspots | 0.35 | 0.35 | 0.25 |
| **VRB** | **0.75** | **0.60** | **0.55** |

VRB在simulation上也显著outperform baselines。

### 6.6 Exploration Results (Figure 5)

Coincidental success (stumbling onto goal configurations):
- VRB vs HAP vs Random: 3× to 10× improvement across all tasks
- 特别impressive的是exploration这种unsupervised setting下也能有如此大提升

### 6.7 Failure Mode Analysis (Figure 9)

Cabinet opening task分类:
- "Failure" / "Partial Success" / "Success"
- VRB的successful trajectories是baselines的2×
- VRB的partial successes是baselines的6×+

这说明VRB即使不完全成功，也经常能做出meaningful progress。

## 7. Critical Analysis & Intuition Building

### 7.1 为什么VRB work so well

**Intuition 1: Robot-first formulation**
Prior work (HOI [66])直接model human movement，导致human-centric model，不generalize到robot。VRB只model "where to contact + how to move after"，这是robot-agnostic abstraction。

**Intuition 2: Multi-modality preservation**
GMM modeling of contact points preserve multi-modality。一个scene可能有multiple valid interactions，naive mean/random sampling会lose这个信息。

**Intuition 3: Domain shift handling**
Human-less frame conditioning这个trick非常elegant。Training时human在frame里extract labels，但conditioning image是human还没enter的那帧。这直接解决了human-robot visual domain shift。

**Intuition 4: Local crops for generalization**
Sampling local crops around contact points作为network input。这tackle了uncertainty问题 - scene中可能有很多objects，有些在training data里有些不在。Local crop让network focus onrelevant region。

### 7.2 Limitations & Future Directions

1. **Single-stage tasks only**: paper提到future work希望deploy on multi-stage tasks
2. **No force/tactile information**: 当前只visual，没有physical concepts
3. **2D pixel space**: contact points和trajectory都在pixel space，需要calibrated camera project到3D
4. **Fixed covariance GMM**: paper发现fixed covariance更好，但这可能limit expressiveness
5. **Egocentric only**: 依赖egocentric videos，third-person videos可能需要不同处理

### 7.3 Connection to Broader Themes

**与CLIP/contrastive learning的关系**: VRB的visual representation learning作为byproduct，类似于CLIP的emergent capabilities。Affordance training迫使encoder学习action-relevant features。

**与Diffusion Policy的关系**: VRB的trajectory prediction head用Transformer，但当前diffusion policy (Chi et al.)可能更适合multi-modal trajectory prediction。

**与Foundation Models的关系**: VRB可以看作robotics领域的"foundation model"尝试 - 从大规模passive human video学习generalizable affordance prior。

**与Gibson's Affordance Theory的关系**: paper开头引用Gibson，但VRB的affordance是actionable representation而非perceptual theory。这是engineering approximation of Gibson's ecological perception。

### 7.4 Technical Details值得注意

1. **Skin color segmentation** for contact points extraction (from [66])
2. **Savitzky-Golay filter** (window 7, threshold 0.75) for smoothing contact variables
3. **Homography via feature matching** for egomotion compensation
4. **Two robot platforms** with different control:
   - Franka: joints 5,6 ∈ [0, 30, 45]°, joint4=0°, 3DOF end-effector with impedance control
   - Hello Robot: roll ∈ [0, 45, 90]°, telescoping arm (no custom controller needed)

### 7.5 Code & Data Availability

- Project page: https://vision-robotics-bridge.github.io/
- Codebases used:
  - 100 DOH: https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes
  - HOI baseline: https://github.com/stevenlsw/hoi-forecast
  - HAP baseline: https://github.com/uiuc-robovision/hands-as-probes
  - Hotspots baseline: https://github.com/Tushar-N/interaction-hotspots
  - R3M: https://github.com/facebookresearch/r3m
  - DQN: https://github.com/takuseno/d3rlpy
  - Polymetis (Franka controller): https://facebookresearch.github.io/fairo/polymetis/

## 8. Broader Impact & Future Directions

### 8.1 对Robotics的影响

VRB展示了从passive human video学习actionable representation的可行性。这开辟了scaling robot learning的新路径 - 不需要teleoperation或kinesthetic teaching，直接从internet-scale human video学习。

### 8.2 对Computer Vision的影响

Vision community长期研究affordance但主要在static datasets。VRB展示了robot deployment能drive affordance representation的设计 - "what is the right way to represent affordances"这个基本问题只有通过robotic integration才能answer。

### 8.3 Future Work (paper提到)

1. Multi-stage complex tasks
2. Incorporate force and tactile information
3. Investigate VRB as visual representation for broader robotics
4. (我的联想) Combine with LLM/VLM for language-conditioned affordance
5. (我的联想) Diffusion-based trajectory prediction head
6. (我的联想) 3D point cloud affordance representation
7. (我的联想) Active learning - robot query affordance model for uncertain regions

## 9. 总结

VRB是vision-robotics bridge的elegant实现。核心contribution:

1. **Actionable affordance representation**: contact points + post-contact trajectories, robot-agnostic
2. **Scalable data extraction**: 从egocentric videos用off-the-shelf tools提取free supervision
3. **Versatile deployment**: 4个robot learning paradigms都work
4. **Real-world scale**: 200+ hours, 10 tasks, 2 robots, in-the-wild

这个work的深层insight是: **visual representation for robotics需要action-relevant supervision，不只是semantic supervision**。ImageNet pretraining学的是"what"，VRB学的是"where + how"。这个action-relevant representation能bootstrap多种learning paradigm，从imitation到exploration到RL。

对Andrej你来说，这个work可能让你想到你的Eureka Labs和neural network education理念 - VRB本质上是在teach robot "affordance concept" through watching humans，类似于人类child通过observation学习object affordances。这个learning-by-watching paradigm可能比reinforcement learning from scratch更接近human learning机制。

参考链接汇总:
- Paper: https://arxiv.org/abs/2402.08369 (推测)
- Project: https://vision-robotics-bridge.github.io/
- Deepak Pathak's lab: https://www.cs.cmu.edu/~dpathak/
- EpicKitchens: https://epic-kitchens.github.io/2022
- 100 DOH: https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes
- R3M: https://github.com/facebookresearch/r3m
- Gibson affordance: https://en.wikipedia.org/wiki/Affordance
- Franka Kitchen (D4RL): https://github.com/Farama-Foundation/D4RL
