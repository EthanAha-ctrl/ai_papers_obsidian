---
source_pdf: Toward Physically Consistent Driving Video World Models under Challenging
  Trajectories.pdf
paper_sha256: eab32e8c703f3be6080a6ac7238b89a243de1b30cbc4c70610a0a894c5691ebf
processed_at: '2026-08-12T16:49:43-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# PhyGenesis 人话版：让 driving world model 真正懂物理

好，Karpathy 你让我用人话讲，那我就当你是坐在我旁边，我给你白板上画一画这篇 paper 在干嘛。

---

## 1. 这个 paper 要解决什么恶心问题

先想象一个场景：你有一个很 fancy 的 video diffusion model，能 generate 很漂亮的 multi-view driving video。你把它接到你的 autonomous driving stack 里当 simulator 用，planner 吐一个 trajectory 出来，model 就 render 对应的视频。

听起来很美好，对吧？但实际跑起来全是 bug。

**Bug 1：planner 给的 trajectory 本身就是 physics-violating 的。**
比如 planner 算出来两辆车未来的轨迹会穿过对方——这在物理上根本不可能，但 planner 可能因为 optimization 不完美就吐出这种结果。你把这个 trajectory 喂给 video model，model 不知道这是 physics-violating 的，它只会傻乎乎地按 trajectory 渲染，结果就是两辆车像鬼一样互相穿透过去，画面上看到的是 melting、distortion，完全 unphysical。

**Bug 2：即便 trajectory 是 physically OK 的，model 也没见过 collision 这种事。**
你想想 nuScenes 这种数据集，全是正常驾驶，safe driving，小心翼翼。model 从来没见过车撞墙、撞车、冲出路面是什么样。你给它一个 collision trajectory，它 internal representation space 里根本就没有 collision 的 manifold，它只能 collapse 到它见过的 nominal mode，结果渲染出来的还是正常驾驶的样子，或者产生一些 weird artifacts。

这两个 bug 就是 PhyGenesis 要修的。

Project page: https://wmresearch.github.io/PhyGenesis/

---

## 2. 他们的核心 idea

核心 idea 简单到一句话：**把物理这件事拆成两半，前一半管 "trajectory 对不对"，后一半管 "video 渲染得像不像物理世界"。**

具体来说，他们搞了一个 two-stage 的 pipeline：

**Stage 1: Physical Condition Generator**
输入：可能 physics-violating 的 2D trajectories $(x, y)$
输出：physically plausible 的 6-DoF trajectories $(x, y, z, \text{pitch}, \text{yaw}, \text{roll})$

这个 stage 干的事情就是 "trajectory 校正器"。你给它一个鬼 trajectory，它吐出一个符合物理的 trajectory。

**Stage 2: Physics-Enhanced Multi-view Video Generator (PE-MVGen)**
输入：校正后的 6-DoF trajectories + 初始 frame + scene caption
输出：multi-view video

这个 stage 是真正的 video diffusion model，基于 Wan2.1 (https://arxiv.org/abs/2503.20314) 改的。关键 trick 是它 training 的时候见过 collision、off-road 这种 extreme scenario，所以渲染得出来。

两个 stage 分工明确，这是一个很 clean 的 design。

---

## 3. 数据：为什么需要 CARLA

这里有个很关键的 insight：**你想让 model 学会 collision，就必须给它看 collision 的数据。**

nuScenes (https://www.nuscenes.org/) 4.6 小时 real-world data，基本全是 safe driving。你指望 model 从这堆数据里学会车撞墙是什么样？不可能。Information 就不在那里。

那 real-world 怎么 collect collision 数据？你得真撞车，这显然不现实。

所以他们用 CARLA simulator (https://carla.org/)。CARLA 是个 open-source 的 autonomous driving simulator，有 high-fidelity physics engine，可以 controllably 生成各种 extreme scenario。

但 prior work 比如 ReSim (https://arxiv.org/abs/2506.09981) 也用 CARLA 数据了，为什么不行？因为他们只 collect 了 single view + only ego-trajectory annotation，没法 train 控制 multiple agents 的 model，而且根本没 focus 在 physically challenging events 上。

PhyGenesis 的做法更精心：

1. 基于 Bench2Drive (https://github.com/Thinklab-SJTU/Bench2Drive) 的 routing setup
2. 生成两个 subset：
   - **CARLA Ego**: ego vehicle 自己被 perturb 去撞东西
   - **CARLA Adv**: 附近一个 non-ego vehicle 被 perturb
3. Perturbation 方式：lateral offset ∈ [-200, 200] m + target speed ∈ [0, 30] m/s
4. Warmup 2 秒走 default autopilot，然后 apply perturbation
5. 监控 collision sensor 和 off-road event
6. Event 触发后继续 collect 4 秒，然后 terminate

最后 collect 了大概 31 小时 raw data，rule-based filtering 提取出 9.7 小时 physically-challenging clips，跟 4.6 小时 nuScenes 1:1 混合训练。

**为什么 1:1？** 因为如果 simulated data 占比太低，model 还是会被 nominal distribution 主导，学不到 physical events。1:1 是个简单但有效的 balance。

Figure 3 里那张 max ego-acceleration 分布图很直观：nuScenes 的分布集中在低 acceleration 区域，CARLA 的分布明显 right-shift 到高 acceleration。这证明数据 distribution 真的不一样，CARLA 数据确实 cover 了 nuScenes 没有的 tail。

---

## 4. Physical Condition Generator 细节

这个 module 是 paper 的核心创新之一。我来一层一层拆。

### 4.1 输入输出

输入：$N$ 个 agent 的 2D trajectories，每个 agent 是 $T$ 个 timestep 的 $(x, y)$ 坐标
$$\mathcal{T}^{orig} = \{\mathcal{T}_i^{orig}\}_{i=1}^N, \quad \mathcal{T}_i^{orig} = \{\mathcal{T}_{i,t}^{orig}\}_{t=1}^T, \quad \mathcal{T}_{i,t}^{orig} = (x_{i,t}, y_{i,t})$$

变量含义：
- $N$: agent 数量（车有几辆）
- $T$: prediction horizon（paper 里 $T=36$，对应 12Hz 下 3 秒）
- $i$: 第 $i$ 个 agent
- $t$: 第 $t$ 个 timestep
- $x_{i,t}, y_{i,t}$: agent $i$ 在 timestep $t$ 的 2D 坐标

输出：每个 agent 每个 timestep 的 6-DoF state
$$\hat{\mathcal{T}}_{i,t}^{6dof} = (x, y, z, \text{pitch}, \text{yaw}, \text{roll}) \in \mathbb{R}^6$$

那个 hat ($\hat{}$) 表示这是 model 的 prediction，上标 $6dof$ 表示 6 degrees of freedom。

**为什么要升级到 6-DoF？** 因为 2D 表达不了 collision 时的 z 方向（车被撞飞）、pitch（前倾）、yaw（旋转）、roll（侧翻）。Collision 这种 extreme event 的 dynamics 主要就发生在这些维度上。

### 4.2 架构走一遍

整个 architecture 是一个 transformer-like 的结构，我把每一步的 intuition 讲清楚。

**Step 1: Encode trajectories to tokens**

把 2D trajectories 过一个 sine-cosine positional encoding + MLP，变成 agent tokens：
$$\mathbf{q} \in \mathbb{R}^{N \times D}$$

- $N$: agent 数量
- $D$: token dimension（每个 agent 一个 token）

这一步就是把 trajectory 的数字变成 model 能消化的 high-dimensional feature。

**Step 2: Spatial Cross-Attention with PV features (Eq. 1)**

$$\mathbf{q}_s = \mathrm{SpatialCrossAttn}(\mathbf{q}, \mathcal{F}_{pv})$$

- $\mathbf{q}$: 输入 agent tokens
- $\mathcal{F}_{pv}$: multi-view Perspective View features（从图像里提的 visual features）
- $\mathbf{q}_s$: spatially grounded queries

这一步是 deformable cross-attention，让 agent token 根据 trajectory 坐标去 query 图像上对应位置的 features。**直觉就是让每个 agent "看到" 它周围的环境长什么样。** 如果前面有墙，agent token 就能感知到墙的 visual feature。

**Step 3: Agent Self-Attention (Eq. 2)**

$$\mathbf{q}_a = \mathrm{AgentSelfAttn}(\mathbf{q}_s)$$

- $\mathbf{q}_s$: spatially grounded queries
- $\mathbf{q}_a$: agent-aware queries

**这一步是解决 penetration 的关键。** 所有 agent tokens 之间做 self-attention，每个 agent 就能感知到其他 agent 的位置和运动状态。如果 agent A 的 trajectory 会跟 agent B 重叠，A 的 token 在 attention 里就能 "看到" B，后续 prediction 时就能产生避让或 collision 的行为。

这其实就是 multi-agent interaction modeling，跟 motion prediction 里那些 scene transformer 之类的工作思路一样。

**Step 4: Map Cross-Attention (Eq. 3)**

$$\mathbf{q}_m = \mathrm{MapCrossAttn}(\mathbf{q}_a, \mathbf{E}_{map})$$

- $\mathbf{q}_a$: agent-aware queries
- $\mathbf{E}_{map}$: vectorized map embeddings（道路的 vector representation）
- $\mathbf{q}_m$: map-aware queries

这一步是 off-road awareness。让 agent 知道路在哪里，如果 trajectory 会让它冲出路面，它能从 map embedding 里感知到 guardrail、curb 的存在，从而学到与这些 structure 的 collision 行为。

**Step 5: FFN (Eq. 4)**

$$\mathbf{q}_f = \mathrm{FFN}(\mathbf{q}_m)$$

非线性变换，整合前面所有信息。这一步没什么好说的，就是 standard transformer block 的 FFN。

**Step 6: Time-Wise Output Head (Eq. 5-6)**

这里是 paper 一个很精细的设计，我单独讲一下。

传统做法是把 refined token $\mathbf{q}_f[i]$ 过一个 MLP，直接输出整个 trajectory sequence。但这样有个问题：**MLP 倾向于 output smooth function**。

你想想，MLP 是 continuous function 的 universal approximator，它学的 mapping 倾向于平滑。但 collision 是什么？collision 是 velocity 突然从 30 m/s 掉到 0。这是一个 discontinuous、high-frequency 的 event。MLP 会把它 smooth 掉，渲染出来就是 collision 后车还慢慢减速，完全不符合物理。

Figure 4 里那张图就是这个意思：MLP head 出来的 velocity 是 gradual decrease，GT 和 time-wise head 是 instantaneous drop to zero。

所以他们搞了这个 time-wise head。对第 $i$ 个 agent 的 token $\mathbf{q}_f[i]$：

$$\mathbf{h}_{i,t} = \mathrm{TCN}\left(\mathrm{Proj}(\mathbf{q}_f[i] \parallel \mathbf{E}_{time}(t))\right)$$
$$\hat{\mathcal{T}}_{i,t}^{6dof} = \mathrm{MLP}(\mathbf{h}_{i,t}) \in \mathbb{R}^6$$

变量：
- $\mathbf{q}_f[i]$: 第 $i$ 个 agent 的 refined token，先在 time 维度上 expand（复制 $T$ 份）
- $\mathbf{E}_{time}(t)$: timestep $t$ 对应的 learnable temporal embedding，每个 timestep 有自己独立的 embedding
- $\parallel$: concatenation
- $\mathrm{Proj}$: projection layer
- $\mathrm{TCN}$: Temporal Convolutional Network，捕获相邻 timestep 之间的 local dynamic variations
- $\hat{\mathcal{T}}_{i,t}^{6dof}$: 输出的 6-DoF state

**Intuition**: step-specific time embedding 给每个 timestep 独立的 "容量"，让 model 有能力 express high-frequency temporal dynamics。TCN 用局部卷积 capture 相邻 timestep 的变化模式。当 collision 发生时，前后 timestep 的 velocity 差异巨大，TCN 的 local receptive field 能敏锐捕捉到这种 abrupt change。

这跟 NeRF 里用 positional encoding 解决 high-frequency problem 的思路精神相似——你给 model 更细粒度的 positional information，它就能 express 更 high-frequency 的 function。

### 4.3 Counterfactual Training 的 trick

这里有个很聪明的 training 设计。你想让 model 学会 "rectify physics-violating trajectory"，得有 (corrupted input, valid target) 这样的 paired data。怎么造？

1. 取一个 collision clip
2. Collision **之前**的 frames: 保留原始 trajectory（这部分的 trajectory 是 physics-valid 的）
3. Collision **之后**的 frames: 故意 corrupt——用 collision 前的 velocity 线性 extrapolate（假装没发生 collision）
4. 这样合成的 trajectory 在 collision 之后会让 agent 继续往前走，穿透对方或墙壁
5. Ground truth 就是 CARLA 实际记录的 collision dynamics（车撞了之后停下来或弹开）

这就是 **counterfactual trajectory corruption**。你给 model 看一个 "如果没碰撞会怎样" 的假 trajectory，让它学到 "实际应该碰撞并停止"。

同时，为了不让 model 把所有 input 都当成 corrupted 的，还混入了 nuScenes 的 nominal trajectory pairs（不做 corruption），这样 model 学到 "如果 input 已经是 physics-valid 的，就保持原样"。

这本质上是个 **self-supervised denoising** 的思路，跟 autoencoder 去噪一个道理，只是这里的 "noise" 是 physics violation。

### 4.4 Loss function (Eq. 7)

$$\mathcal{L}_{phy} = \frac{1}{N \times T} \sum_{i=1}^{N} \sum_{t=1}^{T} W_{i,t} \|\hat{\mathcal{T}}_{i,t}^{6dof} - \mathcal{T}_{i,t}^{gt}\|_1$$

变量：
- $N$: agent 数量
- $T$: prediction horizon
- $W_{i,t}$: agent $i$ 在 timestep $t$ 的 loss weight
- $\hat{\mathcal{T}}_{i,t}^{6dof}$: 预测的 6-DoF trajectory
- $\mathcal{T}_{i,t}^{gt}$: ground truth 6-DoF trajectory（上标 $gt$ = ground truth）
- $\|\cdot\|_1$: L1 norm

**Weight design** 是个细节但不 trivial 的点：

$$W_{i,t} = W_t^{event} \times W_i^{agent}$$

$W_t^{event}$ 在 event window $[s_e, e_e]$ 上从 $\lambda_{event}$ 指数衰减到 1：
- $s_e = \max(0, t_e - 1)$: window 起点
- $e_e = \min(T-1, t_e + 10)$: window 终点
- $t_e$: event 发生的 timestep
- $\lambda_{event} = 10$（paper 设定）

$W_i^{agent}$ 对参与 collision 的 agent 赋 $\lambda_{agent} = 5$。

**Intuition**: collision 那一刻的 dynamics 最难学也最重要，所以重点 amplify 那个区域的 loss。如果你 uniform weight，model 会被大量 nominal timestep 主导，collision 那几帧的 gradient 信号就被淹没了。

Ablation (Table 5-6) 显示 $\lambda_{event} \in \{1, 5, 10\}$ 和 $\lambda_{agent} \in \{1, 10, 20\}$ 对结果影响不大，说明 model 对这些超参 robust，weighting 机制本身是有效的。

---

## 5. PE-MVGen: Physics-Enhanced Video Generator

### 5.1 基础架构

基于 Wan2.1 (https://arxiv.org/abs/2503.20314)，一个 Diffusion Transformer (DiT)。Wan2.1 原本 condition 在 image 和 text 上，他们改造成 controllable multi-view generator for autonomous driving。

**Multi-view 处理的 trick**：输入 multi-view clips 用 3D VAE 编码成 latents：
$$\mathbf{z} \in \mathbb{R}^{V \times T \times C \times h \times w}$$

- $V$: views 数量（nuScenes 是 6）
- $T$: timesteps
- $C$: channels
- $h, w$: 空间维度

然后 reshape 成 $T \times C \times h \times (V \cdot w)$，把 view dimension concatenate 到 width 维度上。

**为什么这样做？** 因为 attention 本来就是 permutation-invariant 的。你只要把不同 view 的 spatial positions 错开，self-attention 就会自然学到跨 view 的关系。这样做不需要额外参数（不需要 cross-view attention module），是一个 parameter-efficient 的设计。

**Layout conditioning**：把未来 T-frame 的 3D agent boxes 和 map polylines 投影到每个 camera view，用 calibrated intrinsics $\mathbf{K}_v$ 和 extrinsics $\mathbf{E}_v$：
- $\mathbf{K}_v$: 第 $v$ 个 view 的 camera intrinsic matrix（camera 内参）
- $\mathbf{E}_v$: 第 $v$ 个 view 的 camera extrinsic matrix（camera 外参）

得到 view-specific control images $\mathbf{M}_v$，VAE encode 成 $\mathbf{z}_c$，跟 noisy latent $\mathbf{z}_t$ 沿 channel 维度 concatenate，进 DiT。

### 5.2 Rectified Flow Training (Eq. 8-9)

他们用 Rectified Flow 而非 DDPM。这是 Stable Diffusion 3 (https://arxiv.org/abs/2403.03206) 引入的 formulation。

**前向过程** (Eq. 8):
$$\mathbf{z}_t = t\mathbf{z}_1 + (1-t)\mathbf{z}_0$$

- $\mathbf{z}_t$: timestep $t$ 处的 noisy latent
- $\mathbf{z}_1$: clean video latent（下标 1 = fully clean）
- $\mathbf{z}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: standard Gaussian noise（下标 0 = fully noisy）
- $t \in [0, 1]$: timestep，从 logit-normal distribution 采样

**Intuition**: 这是 noise 和 clean data 之间的**线性插值**。$t=0$ 全是 noise，$t=1$ 全是 clean。线性路径比 DDPM 的 curved forward process 更容易学，因为 velocity field 接近 constant。

**Ground-truth velocity**:
$$\mathbf{v}_t = \mathbf{z}_1 - \mathbf{z}_0$$

这就是插值路径的 derivative w.r.t. $t$，是个 constant。

**Flow-matching objective** (Eq. 9):
$$\mathcal{L}_{FM} = \mathbb{E}_{\mathbf{z}_0, \mathbf{z}_1, t} \|u_\theta(\mathbf{z}_t, t, \mathbf{c}_{init}, \mathbf{c}_{text}, \mathbf{c}_{layout}) - \mathbf{v}_t\|_2^2$$

变量：
- $u_\theta$: DiT model 预测的 velocity field，参数 $\theta$
- $\mathbf{z}_t$: timestep $t$ 的 noisy latent
- $t$: timestep
- $\mathbf{c}_{init}$: 初始 context frame 的 latent features（image conditioning）
- $\mathbf{c}_{text}$: scene caption（text conditioning）
- $\mathbf{c}_{layout}$: 未来 multi-view layout images（structural conditioning）
- $\mathbf{v}_t = \mathbf{z}_1 - \mathbf{z}_0$: ground-truth velocity
- $\|\cdot\|_2^2$: squared L2 norm

**Intuition**: model 学的是从 noise 到 data 的 "velocity"（在 latent space 里的方向和速度）。Inference 时从 $\mathbf{z}_0$ 出发，用 Euler method 沿 predicted velocity field 积分到 $\mathbf{z}_1$。因为路径接近直线，Euler method 收敛快，inference steps 可以少很多。

### 5.3 Heterogeneous Co-training 的关键 trick

这里有个 design choice 很重要：**训练 PE-MVGen 时不用 counterfactual trajectories，而是用 ground-truth physical trajectories。**

为什么？因为要 **decouple physical correction 和 rendering**。

- Physical correction 在 Stage 1 (Physical Condition Generator) 做
- Stage 2 (PE-MVGen) 只负责渲染，它接收的 input 都是 physics-valid 的

如果 Stage 2 也用 counterfactual trajectories 训练，那 model 就要同时学 correction 和 rendering，任务太复杂，容易互相干扰。Decoupling 让两个 stage 各司其职。

**1:1 平衡采样**：nuScenes 和 CARLA 数据 1:1 混合。这样 model 既能学 nominal driving 的 visual fidelity，又能学 physical interaction 的 dynamics。

### 5.4 Curriculum Co-training

两阶段训练：
- **Stage 1**: $224 \times 400$ 分辨率，2850 steps，lr = $5 \times 10^{-5}$，global batch = 480
  - 快速学 coarse dynamics 和 multi-view geometry
- **Stage 2**: $448 \times 800$ 分辨率，350 steps，lr = $1 \times 10^{-4}$，global batch = 240
  - High-resolution fine-tuning，refine visual details

**Intuition**: 先在低分辨率学全局结构和 dynamics（computationally cheap，可以 train more steps），再在高分辨率 refine 细节。这是 curriculum learning 的经典做法。

硬件：48 块 NVIDIA H20 GPU。Output: 33-frame videos at 12Hz (约 2.75 秒)。

---

## 6. 实验结果怎么读

### 6.1 评估指标

三个维度：

**Visual quality**:
- FID (Fréchet Inception Distance): 越低越好，衡量 image realism
- FVD (Fréchet Video Distance): 越低越好，衡量 video realism

**Physical plausibility**:
- WorldModelBench (https://arxiv.org/abs/2502.20694) 的 PHY score
- 包含四个子指标的 average:
  - **Mass**: objects 不发生 irregular deformation（车不会变成一坨）
  - **Impenetrability**: objects 不互相穿透
  - **Frame-wise Quality**: 没有 unappealing frames
  - **Temporal Quality**: temporal consistency
- 用 VLM-based judges（跟 human preference aligned）
- 还有 human preference rate (Pref.)

**Controllability**:
- CtrlErr: generated video 里提取的 trajectory 跟 GT 的误差
- Rotation Error 和 Translation Error 的 geometric mean
- Camera pose 用 ViPE (https://arxiv.org/abs/2508.10934) 提取

### 6.2 Table 1: 主结果

这个 table 的 setup：nuScenes 用 nominal trajectories，CARLA Ego 和 Adv 用 physics-violating counterfactual trajectories。

| Method | nuScenes FID↓ | nuScenes PHY↑ | CARLA Ego FID↓ | CARLA Ego FVD↓ | CARLA Ego PHY↑ | CARLA Ego Pref.↑ |
|---|---|---|---|---|---|---|
| UniMLVG | 17.59 | 0.93 | 34.50 | 260.21 | 0.55 | 0.13 |
| MagicDrive-V2 | 13.40 | 0.92 | 32.19 | 207.64 | 0.60 | 0.06 |
| DiST-4D | 10.49 | 0.86 | 19.84 | 197.57 | 0.39 | 0.10 |
| **PhyGenesis** | **10.24** | **0.97** | **11.03** | **72.48** | **0.71** | **0.71** |

**怎么读这个 table**:

1. **nuScenes 上提升小**：FID 10.49→10.24，PHY 0.86→0.97。因为 nuScenes 用 nominal trajectory，没有 physics violation，所以差距不大。这说明 PhyGenesis 在 nominal case 上不 regression。

2. **CARLA Ego 上提升巨大**：FID 19.84→11.03 (44% reduction)，FVD 197.57→72.48 (63% reduction)，PHY 0.39→0.71 (82% relative improvement)。这才是 paper 真正的 contribution 所在。

3. **Pref. 从 0.10 → 0.71**：human raters 几乎一致偏好 PhyGenesis。这个 0.71 是非常强烈的 preference signal。

4. **DiST-4D 在 CARLA Ego 上 PHY 只有 0.39**：这说明 SOTA prior methods 在 challenging trajectories 下基本完全失败。0.39 意味着 VLM judge 觉得 60% 的 frame 都有 physical violation。

### 6.3 Table 2: 隔离 PE-MVGen 的效果

这个实验给所有方法 **physically feasible 的 GT trajectories**（不做 counterfactual corruption）。目的是 isolate PE-MVGen 的效果，排除 Physical Condition Generator 的影响。

| Method | CARLA Ego FID↓ | CARLA Ego FVD↓ | CARLA Ego PHY↑ |
|---|---|---|---|
| UniMLVG | 34.03 | 240.78 | 0.56 |
| MagicDrive-V2 | 32.92 | 181.26 | 0.58 |
| DiST-4D | 19.94 | 133.10 | 0.38 |
| **PhyGenesis** | **10.98** | **57.02** | **0.69** |

**Key insight**: 即便输入是 physically feasible 的 GT trajectory，baselines 在 CARLA 上仍然很差。DiST-4D 的 PHY 只有 0.38，FVD 133.10。

**这证明了 Limitation 2**：baselines 训练 distribution 缺乏 physical interactions，即便给了正确 trajectory，也没见过 collision dynamics 长什么样，所以渲染不出来。

PhyGenesis 的 heterogeneous co-training 解决了这个问题。

### 6.4 Table 3: Physical Condition Generator 单独评估

测量 6-DoF L2 distance to GT：

| Method | nuScenes | CARLA Ego | CARLA Adv |
|---|---|---|---|
| w/o Phys. Cond. Generator | 0.21 | 1.78 | 1.05 |
| w/ Phys. Cond. Generator | 0.19 | 0.65 | 0.86 |

**怎么读**:
- nuScenes 提升小（0.21→0.19）：nuScenes trajectory 本身是 nominal 的，没 physics violation，model 主要只是 recover missing 4-DoF (z, pitch, yaw, roll)
- **CARLA Ego 提升巨大**（1.78→0.65，63% reduction）：counterfactual trajectory 严重 physics-violating，Physical Condition Generator 真正发挥作用

### 6.5 Table 4: Ablation Study

| Mixed Data | Phy-Model | CARLA Adv FVD↓ | CARLA Adv PHY↑ | CARLA Adv Pref.↑ |
|---|---|---|---|---|
| ✓ | ✓ | 77.83 | 0.87 | 0.57 |
| ✓ | ✗ | 89.25 | 0.85 | 0.28 |
| ✗ | ✓ | 89.83 | 0.84 | 0.15 |

**怎么读**:
- 两个 components 都重要
- **Mixed Data 对 Pref. 影响更大**（0.15→0.28→0.57）：visual quality 严重依赖 heterogeneous training
- **Phy-Model 主要影响 FVD 和 PHY**：Physical Condition Generator 主要 fix physics violation

---

## 7. 几个我想多聊两句的点

### 7.1 Time-Wise Output Head 的普遍 lesson

这个设计其实揭示了一个普遍 lesson：**当你用 MLP 做 regression head 去 predict time series 时，要小心 smoothness problem。**

MLP 是 continuous function 的 universal approximator，但它学习的过程中 gradient flow 会倾向于 smooth out。如果你的 target function 有 abrupt change（比如 collision 时的 velocity drop），MLP 会把它平均掉。

解决方案就是给每个 timestep 独立的 capacity。这里用 step-specific time embedding + TCN，本质就是让 model 有能力 express high-frequency temporal dynamics。

这个 lesson 在很多地方都适用：robotics 里的 contact transition、weather prediction 里的 precipitation onset、financial 里的 market shock。任何有 abrupt event 的 time series prediction 都可能遇到这个问题。

### 7.2 Counterfactual Training 的深层意义

Counterfactual trajectory corruption 这个 trick 其实比表面看起来更深。它在做一件很有意思的事：**给 model 注入 "what if" 的 reasoning 能力。**

你想，model 看到的 input 是 "如果没有 collision，车会继续往前走"，target 是 "实际上车撞了停下来"。这本质上是让 model 学一个 causal relationship：collision 这个 event 会 cause trajectory 发生什么样的 deviation。

这跟 counterfactual reasoning in causal inference 是同一个 idea。PhyGenesis 用一个很简单的方式实现了这件事：用 physics simulator 生成 counterfactual，用真实 dynamics 作为 supervision。

这个 pattern 可以推广到很多地方。比如你想让 model 学 "如果 driver 没刹车会怎样"，你可以 corrupt braking action，让 model 学到 collision dynamics。这就是一个 safety-critical reasoning 的 training paradigm。

### 7.3 Rectified Flow vs DDPM

为什么用 Rectified Flow 而不是 DDPM？

DDPM 的 forward process 是 Markov chain，逐步加 noise，路径是 curved 的。学习这个 curved path 需要 model 预测每一步的 noise，inference 时需要很多步 reverse。

Rectified Flow 直接在 noise 和 data 之间画一条直线，model 学的是这条直线上的 velocity field。因为路径是直的，inference 时用 Euler method 积分几步就能从 noise 走到 data。

所以 Rectified Flow 的优势是：
1. Training 更 stable（直线路径更容易学）
2. Inference 更快（ fewer steps）
3. Theoretical 更简洁（ODE-based，有更好的 mathematical framework）

这就是为什么 Stable Diffusion 3 和 Wan2.1 都用 Rectified Flow。PhyGenesis 沿用这个 choice 是合理的。

### 7.4 Sim-to-real 的隐忧

Paper 里有个细节：为了 fair comparison，他们训了一个 style transfer model 把 CARLA 视频转成 nuScenes style。但训练 PhyGenesis 自己时用的是原始 CARLA 数据。

这意味着 PE-MVGen 看到的 visual domain 是 mixed 的：一半 nuScenes 真实风格，一半 CARLA 仿真风格。

Deployment 时如果在 real-world，model 见过的 real-world 数据只有 4.6 小时 nuScenes，而见过的 CARLA 数据有 9.7 小时。这个比例会不会让 model 在 real-world 上 generalize 不好？

Paper 里 Figure 6 的 nuScenes stress test 显示 PhyGenesis 在 real-world OOD trajectories 上仍然 work，说明 generalize 是 OK 的。但这个 concern 还是值得 future work 探讨。

### 7.5 跟 GAIA-1、Wayve LINGO 的对比

GAIA-1 (https://arxiv.org/abs/2309.17080) 是 Wayve 的 driving world model，也是 video generation based。但 GAIA-1 主要 focus 在 nominal driving，没处理 physics-violating trajectories。

Wayve LINGO 系列是 language-based 的 driving reasoning，跟 PhyGenesis 是不同方向。PhyGenesis 是 trajectory-conditioned video generation，LINGO 是 language-grounded driving understanding。

但两者其实可以 complement：PhyGenesis 负责 generate physically consistent video，LINGO 负责 explain driving decision in language。Autonomous driving 的完整 stack 可能两者都需要。

### 7.6 跟 NVIDIA Cosmos 的关系

NVIDIA 2025 年发布了 Cosmos (https://arxiv.org/abs/2511.00062)，一个 physical AI 的 world foundation model。Cosmos 也是 video generation based，规模很大。

PhyGenesis 跟 Cosmos 是 complementary 的。Cosmos 追求 general physical world modeling，PhyGenesis 专注 autonomous driving 的 physical consistency。Cosmos 可能没有 PhyGenesis 这种 trajectory rectification 和 counterfactual training 的精细设计。

未来可能 PhyGenesis 这种 methodology 会 integrate 进更大的 foundation model 里，作为 fine-tuning 或 adapter 的形式。

---

## 8. 我读完之后的几个 takeaways

1. **Physical consistency 不是 free lunch**。你不能指望 model 从 nominal data 里自动学会 collision dynamics。Information 必须在 training data 里。这是 data-centric 的核心 lesson。

2. **Decoupling 是好的 design principle**。PhyGenesis 把 trajectory correction 和 video rendering 分成两个 stage，各司其职。这种 decoupling 让训练更稳定，debugging 更容易。

3. **Counterfactual training 是注入 physical priors 的 powerful paradigm**。用 simulator 生成 counterfactual，用真实 dynamics 做 supervision，这本质上是 causal reasoning 的 self-supervised learning。

4. **Regression head 的设计要 match target signal 的 frequency**。MLP 会 smooth out abrupt events，需要 step-specific capacity + local temporal processing 来 capture high-frequency dynamics。

5. **Heterogeneous data + balanced sampling 是 key**。1:1 的 real-to-sim ratio 让 model 既学 visual fidelity 又学 physical dynamics。这个 balance 很重要。

6. **Closed-loop simulation 需要 robustness to imperfect inputs**。Planner 或 simulator 吐出来的 trajectory 可能 imperfect，world model 必须能 handle 这种 noise。PhyGenesis 是第一个 explicit address 这个 problem 的 framework。

---

Karpathy，这就是 PhyGenesis 的人话版。核心 story 就是：driving world model 要真正有用，必须 physical consistent；要 physical consistent，必须 explicitly handle trajectory feasibility 和 physics-consistent generation 两件事；做这两件事，你需要 heterogeneous data + counterfactual training + decoupled architecture。

Paper 链接你可以在 project page 找到：https://wmresearch.github.io/PhyGenesis/

希望这个解读能 build 起你的 intuition。如果你想深入聊某个具体 component，比如 counterfactual corruption 的更多 variant，或者 time-wise head 的 ablation，我可以继续展开。

---

# PhyGenesis 深度解读

这篇 paper 是 Zhejiang University 与 Xiaomi EV 合作的工作，核心要解决的问题非常 sharp：**现有 driving video world models 在给定 challenging 或 counterfactual trajectories 时会产生 severe physical inconsistencies**。比如 simulator 或 planner 生成的 trajectory 可能包含 overlapping paths（车辆互相穿透），现有模型只是 condition-to-pixel translator，强行 follow 这种 input 就会产生 melting、distortion artifacts。

Project page: https://wmresearch.github.io/PhyGenesis/

---

## 1. 问题定义与 Intuition

### 1.1 两个 fundamental limitations

Paper 识别出 prior work 有两个核心缺陷：

**Limitation 1: Lack of physical awareness of trajectory feasibility.**
Trajectory simulators 或 planners 生成的 trajectory 可能在物理上就是 impossible 的（比如两辆车轨迹相交导致 object penetration）。现有模型没有 explicit physical reasoning，只是把 trajectory 当成 condition 直接渲染。

**Limitation 2: Lack of physics-consistent generation capability.**
即便 trajectory 是 physically feasible 的，模型仍然难以生成 collision 或 off-road departure 这种 rare scenarios 的 realistic dynamics。原因是 training data (e.g., nuScenes) 严重偏向 safe nominal driving。

### 1.2 输入输出形式化

系统的输入：
- 初始多视角图像 $\mathcal{T}_0$
- 静态 map $\mathcal{M}$
- 未来 trajectories $\mathcal{T}^{orig} = \{\mathcal{T}_i^{orig}\}_{i=1}^N$，其中每个 agent 的 trajectory $\mathcal{T}_i^{orig} = \{\mathcal{T}_{i,t}^{orig}\}_{t=1}^T$，$\mathcal{T}_{i,t}^{orig} = (x_{i,t}, y_{i,t})$ 是 2D location

变量含义：
- $N$: agent 数量
- $T$: prediction horizon（paper 中 $T=36$，对应 12Hz 下 3 秒）
- $i$ (下标): agent index, $1 \le i \le N$
- $t$ (下标): timestep, $1 \le t \le T$
- $(x_{i,t}, y_{i,t})$: agent $i$ 在 timestep $t$ 的 2D 坐标

这个 2D 表示 align 主流 trajectory simulator 和 end-to-end planner 的 output format，但**关键局限**是 2D 坐标无法表达 collision 时 z 轴和 rotational axes 上的剧烈变化。这是 PhyGenesis 要升级到 6-DoF 的 motivation。

---

## 2. Heterogeneous Multi-view Dataset

### 2.1 为什么需要混合数据

nuScenes (https://www.nuscenes.org/) 4.6 小时 real-world data 提供复杂 urban scene 的 nominal driving behaviors，但**根本缺乏 collision、off-road 这种 physical interaction 的 supervision**。模型在这种 data 上训练，遇到 challenging scenario 时会 deform vehicles。

CARLA simulator (https://carla.org/) 提供 high-fidelity physics engine，可以 controllably 生成 extreme scenarios。Prior work ReSim 也有用 synthetic data，但局限于 single view + only ego-trajectory annotations，无法 train 控制 multiple agents 的模型，而且没有 focus 在 physically challenging events。

### 2.2 CARLA 数据采集设计

基于 Bench2Drive (https://github.com/Thinklab-SJTU/Bench2Drive) routing setup，构建两个子集：

**CARLA Ego**: ego vehicle 与 environment 或 surrounding agents 的 interaction
**CARLA Adv**: 附近 non-ego agent 为中心的 interaction

Perturbation 机制（附录D）：
- Lateral route offset: 从 [-200, 200] meters 采样
- Target speed: 从 [0, 30] m/s 采样
- 三种 perturbation modes 等概率：
  - (i) zero lateral offset + randomized speed
  - (ii) fixed 10 m/s speed + randomized offset
  - (iii) 两者都 randomized

**Warmup**: 24 steps (2 seconds at 12Hz) 走 default autopilot
**Post-event collection**: 48 frames (4 seconds) 后终止
**Timeout**: 120 post-warmup steps (10s) 无 event 则终止

Sensor suite 严格 align nuScenes：1 LiDAR + 6 cameras (900×1600) + 5 radars + 1 IMU/GNSS，外加 collision sensor 和 HD map metadata 用于精确记录 impact timestamps。

### 2.3 数据分布

总共 ~31 小时 simulation：
- CARLA-Adv: 15.5h, 760K bounding boxes
- CARLA-Ego: 15.2h, 830K boxes
- 经 rule-based filtering 提取出 9.7h physically-challenging clips
- 与 4.6h real-world data 组成 heterogeneous dataset

**Figure 3 的 intuition**: max ego-acceleration 分布明显 shift 到更高值，表明 CARLA 数据确实包含更 aggressive dynamics。这点很关键——它验证了混合数据集真的 cover 了 nuScenes 缺失的 distribution tail。

---

## 3. Physical Condition Generator

这是 paper 的第一个核心创新。Goal: 把可能 physics-violating 的 2D trajectories $\mathcal{T}^{orig}$ rectify 成 physically plausible 的 6-DoF trajectory sequence $\hat{\mathcal{T}}^{6dof}$。

### 3.1 为什么需要 6-DoF

2D $(x, y)$ 无法表达 collision 时的：
- z 方向（车被撞飞、悬空）
- pitch（前倾后仰）
- yaw（旋转）
- roll（侧翻）

升级到 6-DoF $(x, y, z, \text{pitch}, \text{yaw}, \text{roll})$ 才能准确 capture extreme physical interactions。

### 3.2 架构详解

#### Step 1: Agent Token Encoding
输入 trajectories $\mathcal{T}^{orig}$ 经过 sine-cosine positional encoding + MLP encoder 得到 agent tokens：
$$\mathbf{q} \in \mathbb{R}^{N \times D}$$
- $N$: agent 数量
- $D$: token dimension

#### Step 2: Spatial Cross-Attention (Eq. 1)
$$\mathbf{q}_s = \mathrm{SpatialCrossAttn}(\mathbf{q}, \mathcal{F}_{pv})$$
- $\mathbf{q}$: input agent tokens
- $\mathcal{F}_{pv}$: multi-view Perspective View (PV) features
- $\mathbf{q}_s$: spatially grounded queries

用 deformable attention 让 agent tokens 根据 trajectory 坐标与视觉 features 交互，**让 agent "看到" 它周围的环境**。

#### Step 3: Agent Self-Attention (Eq. 2)
$$\mathbf{q}_a = \mathrm{AgentSelfAttn}(\mathbf{q}_s)$$
- $\mathbf{q}_s$: spatially grounded queries (输入)
- $\mathbf{q}_a$: agent-aware queries (输出)

**这是解决 overlapping 和 penetration conflicts 的关键设计**。每个 agent token 现在 perceive 周围车辆的 positional 和 kinematic states。如果两个 agent 的轨迹会让它们重叠，self-attention 可以让其中一个 "知道" 另一个的存在，从而在后续预测中产生避让或碰撞的物理行为。

#### Step 4: Map Cross-Attention (Eq. 3)
$$\mathbf{q}_m = \mathrm{MapCrossAttn}(\mathbf{q}_a, \mathbf{E}_{map})$$
- $\mathbf{q}_a$: agent-aware queries
- $\mathbf{E}_{map}$: vectorized map embeddings
- $\mathbf{q}_m$: map-aware queries

集成 vectorized map embeddings 是为了 **off-road awareness**。agent 知道路在哪里，如果 trajectory 会让它冲出道路，model 可以学到与 guardrail、curb 的 collision 行为。

#### Step 5: FFN (Eq. 4)
$$\mathbf{q}_f = \mathrm{FFN}(\mathbf{q}_m)$$
- $\mathbf{q}_m$: map-aware queries
- $\mathbf{q}_f$: fully refined queries

非线性变换，整合前面所有 aggregated features。

#### Step 6: Time-Wise Output Head (Eq. 5-6)
这是 paper 的一个精细设计。**传统 MLP regression head 会 smooth out trajectory outputs**，无法 capture collision 时刻的 sudden, high-frequency dynamic impulses（比如 velocity 瞬间 drop 到 0）。

对第 $i$ 个 refined agent token $\mathbf{q}_f[i]$：
$$\mathbf{h}_{i,t} = \mathrm{TCN}\left(\mathrm{Proj}(\mathbf{q}_f[i] \parallel \mathbf{E}_{time}(t))\right)$$
$$\hat{\mathcal{T}}_{i,t}^{6dof} = \mathrm{MLP}(\mathbf{h}_{i,t}) \in \mathbb{R}^6$$

变量：
- $i$ (下标): agent index
- $t$ (下标): timestep
- $\mathbf{q}_f[i]$: 第 $i$ 个 agent 的 refined token（在 time 维度上 expand）
- $\mathbf{E}_{time}(t)$: timestep $t$ 的 learnable temporal embedding（step-specific）
- $\parallel$: concatenation
- $\mathrm{Proj}$: projection layer
- $\mathrm{TCN}$: Temporal Convolutional Network，捕获 inter-step local dynamic variations
- $\hat{\mathcal{T}}_{i,t}^{6dof}$ (hat 表示 prediction): 输出的 6-DoF state，shape $\mathbb{R}^6$
- 上标 $6dof$: 6 degrees of freedom = $(x, y, z, \text{pitch}, \text{yaw}, \text{roll})$

**Intuition**: step-specific time embedding 让每个 timestep 有自己的 "slot"，TCN 用局部卷积捕获相邻 timestep 之间的变化模式。当 collision 发生时，前后 timestep 的 velocity 差异极大，TCN 的局部 receptive field 能敏锐 capture 这种 abrupt change，而全连接 MLP 会把它平均掉。

Figure 4 的可视化对比很直观：
- MLP head: collision 后 velocity 逐渐下降（不符合物理）
- GT 和 time-wise head: collision 瞬间 velocity 立即 drop 到 0（真实物理）

### 3.3 Counterfactual Training Pair Construction

这是训练 Physical Condition Generator 的核心 trick。要让模型学会 **rectify physics-violating trajectories**，需要 (corrupted input, valid target) 这样的 paired data。

构造方法：
1. 取一个 collision clip
2. Collision **之前**的 frames: 保留原始 trajectory logs
3. Collision **之后**的 frames: 故意 corrupt 所有 agents 的 trajectories——用 collision 前的 velocity 继续线性 extrapolate（这样模拟 "如果没有碰撞会怎样"）
4. 这样合成的 counterfactual trajectory 在物理上会导致 object penetration
5. Ground-truth simulation logs（实际 collision dynamics）作为 supervision target

**同时**，为了避免 distort natural driving conditions，还包含 nuScenes 的 nominal trajectory pairs（不做 corruption）。

**Intuition**: 这是一个 self-supervised denoising 的思路。模型看到 "如果没碰撞会穿透" 的假 trajectory，要学到 "实际应该发生碰撞并停止"。这给模型注入了 intrinsic physical priors。

### 3.4 Optimization (Eq. 7)

$$\mathcal{L}_{phy} = \frac{1}{N \times T} \sum_{i=1}^{N} \sum_{t=1}^{T} W_{i,t} \|\hat{\mathcal{T}}_{i,t}^{6dof} - \mathcal{T}_{i,t}^{gt}\|_1$$

变量：
- $N$: agent 数量
- $T$: prediction horizon
- $i$ (下标, $1 \le i \le N$): agent index
- $t$ (下标, $1 \le t \le T$): timestep
- $W_{i,t}$: agent $i$ 在 timestep $t$ 的 loss weight
- $\hat{\mathcal{T}}_{i,t}^{6dof}$: 预测的 6-DoF trajectory
- $\mathcal{T}_{i,t}^{gt}$ (上标 $gt$ = ground truth): 真实 6-DoF trajectory
- $\|\cdot\|_1$: L1 norm

**Weight 设计**（附录E）：
$$W_{i,t} = W_t^{event} \times W_i^{agent}$$

$W_t^{event}$ 定义在 event window $[s_e, e_e]$ 上：
- $s_e = \max(0, t_e - 1)$: window 起点
- $e_e = \min(T-1, t_e + 10)$: window 终点
- $t_e$: event 发生的 timestep

$$w_e(t) = \lambda_{event} \exp\left(\frac{\log(1/\lambda_{event})}{e_e - s_e}(t - s_e)\right), \quad t \in [s_e, e_e]$$

- $\lambda_{event}$: window 内的最大 weight（paper 中 = 10）
- 从 $\lambda_{event}$ 指数衰减到 1
- 多个 event window 重叠时取 max
- Window 外 $W_t^{event} = 1$

$W_i^{agent}$: 对参与 collision 的 agent（以及 collision 对象）赋更大的 weight $\lambda_{agent}$（paper 中 = 5）

**Intuition**: 这个 weight 设计让模型在 physical event 附近、相关 agent 上的 supervision 更强。Collision 瞬间的 dynamics 最难学也最重要，所以重点 amplify。

Ablation (Tables 5-6) 显示 $\lambda_{event} \in \{1, 5, 10\}$ 和 $\lambda_{agent} \in \{1, 10, 20\}$ 差异不大，说明 model 对这些超参 robust。

---

## 4. Physics-Enhanced Multi-view Video Generator (PE-MVGen)

### 4.1 基础架构

基于 Wan2.1 (https://arxiv.org/abs/2503.20314)，一个 high-capacity Diffusion Transformer (DiT)，原本 condition 在 image 和 text 上。PhyGenesis 把它改造成 controllable multi-view generator 用于 autonomous driving。

### 4.2 Multi-view & Layout Conditioning

#### Multi-view latents
输入 multi-view clips 用 pre-trained 3D VAE 编码：
$$\mathbf{z} \in \mathbb{R}^{V \times T \times C \times h \times w}$$
- $V$: views 数量
- $T$: timesteps
- $C$: channels
- $h, w$: 空间维度

**关键 trick**: 为了 enable multi-view modeling 而不引入额外参数，把 view dimension reshape 进 spatial axis：
$$T \times C \times h \times (V \cdot w)$$

把 $V$ 个 view 在 width 维度上 concatenate。这样同一个 self-attention 就能 capture cross-view dependencies。这是一个 elegant 的 parameter-efficient 设计——attention 本来就是 permutation-invariant 的，只要 spatial positions 不同，attention 自然会学到跨 view 的关系。

#### Layout conditioning
将未来 T-frame 的 3D agent boxes 和 map polylines 投影到每个 camera view，用 calibrated intrinsics $\mathbf{K}_v$ 和 extrinsics $\mathbf{E}_v$：
- $\mathbf{K}_v$: 第 $v$ 个 view 的 camera intrinsic matrix
- $\mathbf{E}_v$: 第 $v$ 个 view 的 camera extrinsic matrix

得到 view-specific control images $\mathbf{M}_v$，用 VAE encoder 编码成 $\mathbf{z}_c$，reshape 后与 noisy latent $\mathbf{z}_t$ 沿 channel dimension concatenate，最后通过 patch embedder 进入 DiT。

### 4.3 Rectified Flow Training (Eq. 8-9)

这是 Stable Diffusion 3 (https://arxiv.org/abs/2403.03206) 引入的 rectified flow formulation，比传统 DDPM 更 stable。

**前向过程** (Eq. 8):
$$\mathbf{z}_t = t\mathbf{z}_1 + (1-t)\mathbf{z}_0$$
- $\mathbf{z}_t$: timestep $t$ 处的 noisy latent（下标 $t$）
- $\mathbf{z}_1$: clean video latent（下标 1 表示 fully clean）
- $\mathbf{z}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: standard Gaussian noise（下标 0 表示 fully noisy）
- $t \in [0, 1]$: timestep，从 logit-normal distribution 采样

**Intuition**: 这是 noise 和 clean data 之间的 **线性插值**。$t=0$ 时全 noise，$t=1$ 时全 clean。线性路径比 DDPM 的 curved forward process 更容易学习，因为 velocity field 是 constant。

**Ground-truth velocity**:
$$\mathbf{v}_t = \mathbf{z}_1 - \mathbf{z}_0$$

**Flow-matching objective** (Eq. 9):
$$\mathcal{L}_{FM} = \mathbb{E}_{\mathbf{z}_0, \mathbf{z}_1, t} \|u_\theta(\mathbf{z}_t, t, \mathbf{c}_{init}, \mathbf{c}_{text}, \mathbf{c}_{layout}) - \mathbf{v}_t\|_2^2$$

变量：
- $u_\theta$: DiT model 预测的 velocity field，参数 $\theta$
- $\mathbf{z}_t$: timestep $t$ 的 noisy latent
- $t$: timestep
- $\mathbf{c}_{init}$: 初始 context frame 的 latent features（conditioning）
- $\mathbf{c}_{text}$: scene caption（text conditioning）
- $\mathbf{c}_{layout}$: 未来 multi-view layout images（structural conditioning）
- $\mathbf{v}_t = \mathbf{z}_1 - \mathbf{z}_0$: ground-truth velocity
- $\|\cdot\|_2^2$: squared L2 norm

**Intuition**: 模型学的是从 noise 到 data 的 "velocity"（即 derivative of $\mathbf{z}_t$ w.r.t. $t$）。Inference 时从 $\mathbf{z}_0$ 出发，用 Euler method 沿 predicted velocity field 积分到 $\mathbf{z}_1$。

### 4.4 Heterogeneous Co-training

**关键**: 训练 PE-MVGen 时 **不用 counterfactual trajectories**！Generator 接收的是 ground-truth physical trajectories，把 physical correction 和 rendering 解耦。

- Real-world (nuScenes) 和 simulated (CARLA) 数据 1:1 平衡采样
- CARLA 数据提供 collision、off-road 等 dense supervision
- 这种 co-training 让模型的 generative capabilities 鲁棒 generalize 到 physically challenging scenarios

**Intuition**: 这相当于让 generator "见世面"。原来只见过 safe driving，现在见过 collision、off-road，于是 internal representation space 中这些 rare events 有了对应的 manifold，generation 时就不会 collapse 到 default nominal mode。

### 4.5 Curriculum Co-Training

两阶段训练：
- **Stage 1**: $224 \times 400$ resolution, 2850 steps, lr = $5 \times 10^{-5}$, global batch = 480
  - 快速学习 multi-view geometry 和 physically challenging layout mappings
- **Stage 2**: $448 \times 800$ resolution, 350 steps, lr = $1 \times 10^{-4}$, global batch = 240
  - High-resolution fine-tuning，确保 visual fidelity

**Intuition**: 先在低分辨率学 coarse dynamics 和 geometry（computationally cheap），再在高分辨率 refine 细节。这是 curriculum learning 的经典做法。

硬件: 48 NVIDIA H20 GPUs。Output: 33-frame videos at 12Hz（约 2.75 秒）。

---

## 5. 实验结果分析

### 5.1 评估指标

三个维度：

**Visual quality**:
- FID (Fréchet Inception Distance): 越低越好
- FVD (Fréchet Video Distance): 越低越好

**Physical plausibility**:
- WorldModelBench (https://arxiv.org/abs/2502.20694) 的 PHY score
- 包含四个子指标的平均：
  - Mass: objects 不发生 irregular deformation
  - Impenetrability: objects 不互相穿透
  - Frame-wise Quality: 没有 unappealing frames
  - Temporal Quality: temporal consistency
- 用 VLM-based judges（human preference-aligned）
- 还有 human preference rate (Pref.)

**Controllability**:
- CtrlErr: generated video 中提取的 trajectory 与 GT trajectory 的误差
- 定义为 Rotation Error 和 Translation Error 的 geometric mean
- Camera pose 用 ViPE (https://arxiv.org/abs/2508.10934) 提取

### 5.2 Table 1: 2D Trajectory Conditions

这个 table 的 setup：nuScenes 用 nominal trajectories，CARLA Ego 和 CARLA Adv 用 physics-violating counterfactual trajectories（pre-collision 保持，post-collision 用 pre-collision velocity 线性 extrapolate）。

| Method | nuScenes FID↓ | nuScenes FVD↓ | nuScenes PHY↑ | nuScenes Pref.↑ | CARLA Ego FID↓ | CARLA Ego FVD↓ | CARLA Ego PHY↑ | CARLA Ego Pref.↑ |
|---|---|---|---|---|---|---|---|---|
| UniMLVG | 17.59 | 129.69 | 0.93 | 0.05 | 34.50 | 260.21 | 0.55 | 0.13 |
| MagicDrive-V2 | 13.40 | 91.10 | 0.92 | 0.16 | 32.19 | 207.64 | 0.60 | 0.06 |
| DiST-4D | 10.49 | 46.95 | 0.86 | 0.13 | 19.84 | 197.57 | 0.39 | 0.10 |
| **PhyGenesis** | **10.24** | **40.41** | **0.97** | **0.67** | **11.03** | **72.48** | **0.71** | **0.71** |

**Key observations**:
1. nuScenes 上提升相对小（FID 10.49→10.24, PHY 0.86→0.97）——因为 nuScenes 是 nominal trajectory，没有 physics violation
2. **CARLA Ego 上提升巨大**：FID 19.84→11.03 (44% reduction), FVD 197.57→72.48 (63% reduction), PHY 0.39→0.71 (82% relative improvement)
3. Pref. 从 0.10 → 0.71，human raters 几乎一致偏好 PhyGenesis
4. DiST-4D 在 CARLA Ego 上 PHY 只有 0.39，说明 prior methods 在 challenging trajectories 下基本完全失败

### 5.3 Table 2: GT Trajectory Conditions

这个实验给所有方法 **physically feasible 的 GT trajectories**（不做 counterfactual corruption）。目的是 isolate PE-MVGen 的效果，排除 Physical Condition Generator 的影响。

**关键发现**: 即便输入是 physically feasible 的，baselines 在 CARLA 上仍然表现差：
- DiST-4D CARLA Ego: FID 19.94, FVD 133.10, PHY 0.38
- PhyGenesis CARLA Ego: FID 10.98, FVD 57.02, PHY 0.69

**Intuition**: 这证明了 **Limitation 2** 的存在——training distribution 缺乏 physical interactions。Baselines 即便给了正确 trajectory，也没见过 collision dynamics 长什么样，所以渲染不出来。PhyGenesis 的 heterogeneous co-training 解决了这个问题。

### 5.4 Table 3: Physical Condition Generator 单独评估

测量 6-DoF L2 distance to GT：

| Method | nuScenes | CARLA Ego | CARLA Adv |
|---|---|---|---|
| w/o Phys. Cond. Generator | 0.21 | 1.78 | 1.05 |
| w/ Phys. Cond. Generator | 0.19 | 0.65 | 0.86 |

**Intuition**:
- nuScenes 提升小（0.21→0.19），因为 nuScenes trajectory 本身是 nominal 的，没有 physics violation，model 主要只是 recover missing 4-DoF (z, pitch, yaw, roll)
- CARLA Ego 提升巨大（1.78→0.65，63% reduction），因为 counterfactual trajectory 严重 physics-violating，Physical Condition Generator 真正发挥作用

### 5.5 Table 4: Ablation Study

| Mixed Data | Phy-Model | CARLA Adv FVD↓ | CARLA Adv PHY↑ | CARLA Adv Pref.↑ |
|---|---|---|---|---|
| ✓ | ✓ | 77.83 | 0.87 | 0.57 |
| ✓ | ✗ | 89.25 | 0.85 | 0.28 |
| ✗ | ✓ | 89.83 | 0.84 | 0.15 |

**Intuition**:
- 两个 components 都重要，但 **Mixed Data 对 Pref. 的影响更大**（0.15→0.28→0.57）
- Phy-Model 主要影响 PHY score 和 FVD
- 没有 Mixed Data，PHY 基本不变（0.87 vs 0.85），说明 PHY 提升主要来自 Physical Condition Generator；但 Pref. 大幅下降，说明 visual quality 严重依赖 heterogeneous training

### 5.6 nuScenes Stress Test (Figure 6, 7)

为了测试 OOD physics-violating trajectories 在 real-world scenes 下的表现，他们 scale up ego vehicle speed 并保留 collision cases。

PhyGenesis 在这种 corrupted conditions 下仍然保持更 physical consistent，说明学到的 physical priors 可以 generalize 到 real-world domain。

---

## 6. Style Transfer Model (Appendix B)

为了让 baseline (主要 train 在 nuScenes) 在 CARLA 上 fair 比较，他们训了一个 style transfer model 把 CARLA 视频 translate 成 nuScenes style。

- 基础: Wan2.1-Fun-V1.1-1.3B-Control
- 用 Depth Anything V2 (https://arxiv.org/abs/2406.09414) 提取 per-frame depth
- 用 Qwen2.5-VL (https://qwenlm.github.io/blog/qwen2.5-vl/) 生成 video-level captions
- **只在 nuScenes 上训练**（不是 heterogeneous），这样学到的是 nuScenes 的 appearance characteristics
- **不用 initial frame 作为 conditioning**，所以 generation 反映 nuScenes style 而非 input video 的 appearance

Eq. 11:
$$\mathcal{L}_{transfer} = \mathbb{E}_{\mathbf{z}_0, \mathbf{z}_1, t} \|u_\theta(\mathbf{z}_t, t, \mathbf{c}_{text}, \mathbf{c}_{depth}) - (\mathbf{z}_1 - \mathbf{z}_0)\|_2^2$$
- $\mathbf{c}_{text}$: video-level caption
- $\mathbf{c}_{depth}$: per-frame depth condition

---

## 7. 整体 Intuition 与启示

### 7.1 两个 components 的分工

**Physical Condition Generator**: 解决 "trajectory 本身 physics-violating" 的问题
- 把 2D → 6-DoF（capture vertical/rotational dynamics）
- 通过 counterfactual training 学到 intrinsic physical priors
- 通过 agent self-attention 解决 penetration conflicts
- 通过 map cross-attention 解决 off-road awareness

**PE-MVGen**: 解决 "model 没见过 collision/off-road" 的问题
- Heterogeneous co-training 让 model 见世面
- Decouple physical correction (在 condition generator 做) 和 rendering (在 generator 做)

### 7.2 为什么这个 framework 重要

**Closed-loop evaluation 的关键痛点**: Planner 或 simulator 生成的 trajectory 可能 imperfect，prior world models 在这种 input 下会 fail。PhyGenesis 让 world model 变得 robust to input noise，这对 autonomous driving 的 safety-critical simulation 至关重要。

**Data-centric insight**: 这篇 paper 的核心 contribution 之一是 **构建 physics-rich dataset**。模型架构上其实没有特别 radical 的创新，但通过精心设计的 CARLA 数据采集（collision sensor + map metadata + rule-based filtering）和 counterfactual corruption strategy，把 physical priors 注入到模型中。这呼应了 Karpathy 你常说的 "data is all you need" 思路。

### 7.3 Rectified Flow 的选择

用 Wan2.1 + Rectified Flows 而非 DDPM 是因为：
- Linear interpolation path 比 DDPM 的 curved path 更容易学习
- ODE-based training 更 stable
- Inference 时可以用更少 steps（因为路径接近直线，Euler method 收敛快）

### 7.4 Time-Wise Output Head 的洞察

传统 MLP regression head 在 trajectory prediction 任务里其实是 sub-optimal 的。MLP 倾向于 output smooth functions，但 collision 这种 event 是 non-smooth 的（velocity 突变）。Step-specific time embedding + TCN 的设计本质上是为每个 timestep 提供独立的 "capacity"，让 model 能 express high-frequency temporal dynamics。这和 NeRF 里 positional encoding 解决 high-frequency 问题的思路有精神上的相似。

### 7.5 与相关工作的关系

- **ReSim** (https://arxiv.org/abs/2506.09981): 也用 synthetic data，但 single view + ego-trajectory only，没 focus 在 physical events
- **Challenger** (https://arxiv.org/abs/2505.15880): trajectory simulator + multi-view generator，但 generator 训在 nominal data 上，无法 depict collision interactions
- **DiST-4D** (https://arxiv.org/abs/2503.01845): state-of-the-art 但在 challenging trajectories 下 PHY score 跌到 0.39
- **MagicDrive-V2** (https://arxiv.org/abs/2411.13807): DiT-based，high-resolution，但同样 lack physical awareness

PhyGenesis 是**第一个**能 synthesize physically consistent multi-view driving videos even when conditioned on initially physics-violating trajectory inputs 的 framework。

---

## 8. 可能的延伸思考

1. **Sim-to-real gap**: Style transfer model 把 CARLA 转成 nuScenes style 来做 fair comparison，但训练 PhyGenesis 时用的是原始 CARLA 数据。这意味着 generator 看到的 visual domain 是 mixed 的。如果 deployment 在 real-world，是否需要 domain adaptation？

2. **Generalization 到 unseen physical events**: 训练数据里只有 collision 和 off-road。如果遇到翻车、爆炸、坠落等更 extreme 的 scenario，model 会怎样？Counterfactual corruption strategy 是否可以 extend 到这些 case？

3. **Computational cost**: Physical Condition Generator 需要额外训练，且 inference 时增加了一 stage。对 closed-loop simulation 来说 latency 是否可接受？

4. **6-DoF 表达能力**: 现在输出 6-DoF，但 collision 时车辆可能发生 deformation，这是 rigid body assumption 的局限。Future work 可能需要软体或 articulation modeling。

5. **与 end-to-end planner 的 integration**: Paper 提到可以 integrate with planners for decision making。如果 planner 输出的 trajectory 被 Physical Condition Generator rectify 了，是否会 mask planner 的 bug？这可能影响 closed-loop training 的 signal fidelity。

---

希望这个解读能 build 起你对 driving video world models 在 physical consistency 方面挑战的 intuition。Core insight 就是：**physics 不是 free lunch**，要么通过 trajectory rectification 显式 enforce，要么通过 physics-rich data 让 model 见过；两者结合才能 robust。
