---
source_pdf: World-R1 Reinforcing 3D Constraints.pdf
paper_sha256: 18d6c46ef627aa08191e7527efb4a37a5acf256a4adf8c388fb359f619f50e83
processed_at: '2026-08-13T05:26:24-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# World-R1 的人话版

Andrej，咱们抛开公式，用大白话讲讲这篇 paper 在干啥。

---

## 问题是什么

现在的 video generation model（Wan 2.1、CogVideoX、Sora 这些）生成 short clip 看着很漂亮，但一旦你让它做大的 camera 运动——比如绕着一个建筑 orbit 180 度，或者从走廊一头 push in 到另一头——就露馅了。墙壁会像橡皮一样 warp，物体会 morph、vanish，整个 scene 像是贴在 cardboard 上的 billboard stack 在 2.5D 平面上漂。

根本原因：这些 model 本质上是 **image-space predictor**，它学的是 "下一帧像素大概长什么样"，没有真正的 3D geometry 概念。它不知道 "这堵墙是一个 3D 平面，从不同角度看应该有 parallax"，它只知道 "这一帧的墙大概是这样，下一帧挪一点"。

---

## 前人怎么解决，为啥不行

两条路都试过：

**路线 A：改架构，挂 3D module**
比如在 DiT 后面挂个 pointmap head，或者加个 spatial memory module。问题是：推理变慢、scalability 差、只能做 I2V（image-to-video，需要 reference image），而且训这些 module 要用 3D 数据集，把 base model 原本丰富的 dynamic diversity 给打没了。

**路线 B：推理时强行加 3D constraint**
生成的时候每帧用 3D-aware guidance 拉回来。问题：latency 爆炸，因为每帧都要跑 3D module，而且只在 trajectory 附近 work，generalization 有限。

---

## World-R1 的核心 insight

关键观察：**video foundation model 内部其实已经 encode 了 rich 3D information**，只是这些 latent knowledge 没被 surface 出来。前人的两条路都在 "加东西"（加 module、加 inference constraint），World-R1 说：别加了，用 RL 把已有的能力 eliciting 出来就行。

这跟 LLM 里 R1 / RLHF 的逻辑完全一样：base model already knows，问题是 output distribution 没对齐到 "good" 标准。RL post-training 就是用 reward signal 把 latent capability push 到 generation distribution 上。

---

## 具体怎么做，三个 piece

### Piece 1：把 camera motion 塞进 noise 里

怎么告诉 model "我要 orbit left"？传统做法是训一个 ControlNet 接 camera pose。World-R1 不训任何东西，用了一个叫 Go-with-the-Flow 的 trick：

> **diffusion model 的初始 noise 不是 random 的——它 spatial 上已经决定了 content 的 layout。**

所以如果你能按 camera motion 把 noise 提前 warp（比如 orbit left 就把 noise 像旋转一样挪一下），model 看到的初始 noise 本身就 encode 了 "我在 moving camera" 的信息，它就会自然 generate 出有 parallax 的 video。

具体怎么 warp？把 camera trajectory 投影成 2D optical flow，然后按 flow 把 noise 像搬砖一样搬过去。难点是搬完之后 noise 分布会坏（有些地方堆太多，有些地方空了），用一个 mass transport 的归一化 trick 修好，保证 noise 还是 $\mathcal{N}(0, I)$。

**一句话：不改架构，不动参数，就把 camera motion "种" 进初始 noise 里。**

---

### Piece 2：3D-aware Reward，analysis-by-synthesis

这是 paper 的核心。RL 需要一个 reward function 告诉 model "这个 video 3D 上对不对"。怎么设计？

用 **analysis-by-synthesis**：

1. **生成一个 video**
2. **用 Depth Anything 3 把 video lift 成 3D Gaussian Splatting representation**，同时 estimate 出这个 video 实际的 camera trajectory
3. **三个维度打分**：
   - **Meta-view score**：从原 video trajectory 之外的视角（meta-view）把 3DGS render 出来，喂给 Qwen3-VL 让它当 3D expert 打分。为什么用 meta-view？因为 baseline model 在原视角看着 OK，但一换视角 render 就露馅——本该 3D 的物体是 flat 的，floater 一堆。VLM 在 meta-view 上能 semantic 判断这些 artifact
   - **Reconstruction score**：从 3DGS 重新 render 原 trajectory 的 video，跟原 video 算 LPIPS。如果原 video 有 hallucination，3DGS 重建出来必然走样，LPIPS 高，reward 低
   - **Trajectory score**：3DGS 估出来的 camera trajectory 跟 prompt 指定的对不对得上。这一项防 reward hacking——不然 model 偷懒生成 static video，3DGS 容易重建，reconstruction score 会虚高

4. **再加一个 general quality reward**：HPSv3 aesthetic score，防止 3D reward 把 model 拽成 "ugly but consistent"

**一句话：让 3D foundation model + VLM 当 judge，用 "能不能被 3D 重建" 当 reward signal。**

---

### Piece 3：Periodic Decoupled Training，防 rigid collapse

这是最巧的工程 trick。问题：纯 3D reward 训下去，model 会发现 "不动最安全"——lion 不抖 mane、waterfall 不流、smoke 不飘，因为 dynamic 内容 3DGS 重建质量差，reconstruction score 低，model 自然学会 "别动"。

解法：**周期性 reward 切换**。

- 100 步用全 reward 强 3D consistency
- 然后一段 step 关掉 3D reward，只用 aesthetic reward，在 500 个 dynamic prompt（fire、waterfall、tornado、moving crowd）上 fine-tune
- 循环

这跟 LLM RLHF 里 "RL 会 catastrophic forget pretrain capability" 是同一个问题。Periodic relief 就是 replay buffer 的精神——定期把 dynamic diversity 拉回来。

---

## 结果说了什么

数字很硬：

- **3D consistency (PSNR)**：Small model 从 17.40 → 27.63（+10.23dB），Large 从 19.76 → 27.67（+7.91dB）。+10dB 在图像重建领域是巨大 jump
- **General video quality (VBench)**：World-R1-Small 的 Aesthetic 65.74，**比 base Wan 2.1 的 62.43 还高**。RL 不仅没伤害 visual quality，反而把它 surface 出来了
- **Camera control**：World-R1-Large 的 trajectory error **超过所有专门 camera control 方法**（包括 CamCloneMaster 这种 reference-based 专门架构）。Post-training 方法在专门 benchmark 上击败专门架构方法
- **Long video**：训练只在短 clip 上做，121 帧（4× 长）PSNR 仍 +8dB。RL 学到的 capability 比 SFT 更 transferable
- **User study**：25 人 blind test，World-R1 overall preference 86% win rate

---

## 为什么这个 work 真的重要

数字之外，真正的 paradigm shift：

> **Video model 的 3D inconsistency 不是 capacity 问题，是 alignment 问题。**

不需要改架构、不需要 3D 数据集、不需要 inference-time constraint，纯 RL post-training 就能把 latent 3D capability surface 出来。

这跟 R1 在 LLM 上的贡献是平行的——证明 RL 是 unlock latent capability 的 scalable 路径，比 data scaling 更 efficient。

---

## 一句话总结

**World-R1 = Wan 2.1 + R1-style RL post-training，用 3DGS + VLM 当 reward model，把 video generator 变成 geometrically consistent world simulator。**

Project page: https://aka.ms/world-r1

期待你下次在 YouTube 上讲这个 work，这跟你 World Models keynote 里 "video model 是 world simulator precursor" 的 thesis 完美呼应。

---

# World-R1: 用 RL 把 Video Foundation Model 变成 World Simulator

Andrej, 这篇 paper 是 MSRA + 浙大 的工作，ICML 2026 接收。核心 thesis 非常 Karpathy-style：**video foundation model 内部已经 encode 了 rich 3D geometry，问题在于这些 latent knowledge 没被 elicit 出来**。World-R1 用 Flow-GRPO post-training，在不改架构、不收集 3D dataset 的前提下，把 Wan 2.1 变成几何一致的 world simulator。PSNR 提升 10.23dB (Small) / 7.91dB (Large)，并且 VBench general video quality 同步提升（不是退化）。这跟 RLAIF 的精神一脉相承：discriminative model 当 reward，generative model 当 policy，让 reward signal 自己去 shape 输出分布。

Project page: https://aka.ms/world-r1
Paper arxiv (Wan 2.1 base): https://arxiv.org/abs/2503.20314
Flow-GRPO: https://arxiv.org/abs/2506.22639
Go-with-the-Flow: https://arxiv.org/abs/2501.02603
Depth Anything 3: https://arxiv.org/abs/2511.10647

---

## 1. Motivation 的深层逻辑

视频生成 model 当前最大的 failure mode 是 **geometric hallucination**：camera 一旦做大角度运动（orbit、push-in 长 corridor），墙壁会 warp，物体会 morph、vanish，整段 video 看上去像 2.5D 的 billboard stack。这是因为 video diffusion 本质上在 image space 学的是 surface correlation，没有 multi-view consistency 的 hard constraint。

前人两条路：
- **Architectural injection**（Voyager [8], Vmem [12], FantasyWorld [43], ViewCrafter [63]）：在 DiT 里挂 3D decoder、pointmap head、spatial memory module。问题：推理慢、scalability 差、I2V 限定、静态 3D 数据集训练把 dynamic diversity 打没。
- **Inference-time constraint**（WorldForge [11], Geometry Forcing [10]）：generation 时候用 3D-aware guidance 强行拉。问题：每帧都要算 3D module，latency 爆炸，trajectory 之外的 generalization 有限。

World-R1 选第三条路：**post-training alignment via RL**。这跟 LLM 里 RLHF / RFT 的逻辑完全镜像——base model 已经懂，但需要 reward model 告诉它什么是 "good"，让 gradient 把 latent capability "蒸馏" 到 generation distribution 上。区别只在于这里是 visual token + flow matching + 视频。

---

## 2. 方法三个核心 module 拆解

整个 framework 分三块：**(a) Implicit camera conditioning via noise wrapping**、**(b) Analysis-by-synthesis 3D-aware reward**、**(c) Periodic decoupled training 防 rigid collapse**。下面把每个都讲透。

### 2.1 Implicit Camera Conditioning: Noise Warping

这一段直接继承 Go-with-the-Flow (Burgert et al., CVPR 2025)。核心 insight：**diffusion model 的 initial noise $z_T$ 不仅是 "random seed"，它 spatial 上已经决定了 content 的 layout**。如果你能让 noise 沿 camera motion 提前 warp，model 就会"以为"它生成的是一个 moving camera 看到的 scene，从而 induce 出 parallax。

**Step 1: Prompt-Driven Trajectory Generation**

定义一个 keyword 检测函数 $\phi(c)$，扫描 prompt 里的 motion token $\mathcal{K} = \{\text{`push in', `pan left', `orbit left', ...}\}$。给定 canonical pose $E_0 = I_{4\times 4}$，第 $t$ 帧的相机外参递归计算：

$$E_t = E_{t-1} \cdot T_{\text{action}}(t) \tag{5}$$

变量含义：
- $E_t \in \mathbb{R}^{4\times 4}$：第 $t$ 帧的 camera extrinsic，是 homogeneous transformation matrix
- $T_{\text{action}}(t) \in \mathbb{R}^{4\times 4}$：对应当前 motion token 的 incremental transformation（例如 push in 就是沿 z 轴平移、orbit 是绕 y 轴旋转 + 视点向心平移）

多 motion token 就把 trajectory concat 起来。

**Step 2: Trajectory → 2D Optical Flow**

用 pinhole + fronto-parallel plane (深度 $z_{\text{ref}}$) 把 3D trajectory 投影成 dense optical flow。pixel $u$ 在 frame $t+1$ 的对应点 $u'$ 用 planar homography：

$$u' \sim K\left(R_{\text{rel}} + \frac{1}{z_{\text{ref}}}\mathbf{t}_{\text{rel}}\mathbf{n}^\top\right)K^{-1}u \tag{6}$$

变量：
- $K \in \mathbb{R}^{3\times 3}$：相机内参
- $R_{\text{rel}}, \mathbf{t}_{\text{rel}}$：相邻帧之间的相对 rigid 变换，即 $E_{t+1}E_t^{-1}$ 的旋转/平移分量
- $\mathbf{n} = [0,0,1]^\top$：image plane 法向量
- $z_{\text{ref}}$：假设的 scene 平均深度（这是一个近似，但 noise transport 不需要精确 depth，只需要正确的"运动方向"作为 inductive bias）

forward flow $f(u) = u' - u$。

**Step 3: Discrete Noise Transport（关键 trick）**

直接用 flow warp noise 会出问题：某些区域被多个 source pixel 命中（density 过高，variance collapse），某些 disocclusion 区域空着（zero noise）。这破坏了 $\mathcal{N}(0, \mathbf{I})$ 分布，diffusion model 会失效。

Go-with-the-Flow 把它当 **mass transport on bipartite graph**：source pixel $v$ → target pixel $v'$，density tracker $\rho(v')$ 记录 incoming count，最后归一化保 variance：

$$z_{t+1}(v') = \frac{1}{\sqrt{\rho(v')}}\sum_{v \mapsto v'} z_t(v) \tag{7}$$

变量：
- $z_t(v)$：第 $t$ 帧 pixel $v$ 的 noise 值
- $z_{t+1}(v')$：第 $t+1$ 帧 target pixel $v'$ 的 noise 值
- $\rho(v')$：有多少个 source pixel 贡献到 $v'$（density counter）
- $\frac{1}{\sqrt{\rho(v')}}$：归一化因子，保证 $\text{Var}[z_{t+1}(v')] = 1$，即维持 marginal $\mathcal{N}(0, \mathbf{I})$

**Intuition**: 这相当于把 noise 当 mass，按 optical flow 重新分配，让相邻帧的 noise 在"应该 disocclude 的地方"出现新 noise，在 "应该 occlude 的地方" 合并；通过 $1/\sqrt{\rho}$ 归一化保证分布不漂。这样 DiT 看到的初始 noise 本身就 encode 了 camera motion，等于免费给了一个 motion-aware inductive bias，不需要 train extra controlnet。World-R1 的 ablation (Table J: w/o noise wrapping → PSNR 从 27.63 掉到 24.46，VBench AVG 从 85.21 掉到 76.39) 印证了这个 inductive bias 是 optimization 收敛的 critical enabler。

---

### 2.2 Flow-GRPO: 把 Video Diffusion 当 MDP

这是 RL 部分，最值得展开。Flow-GRPO (Liu et al., NeurIPS 2025) 把 flow matching 的 denoising process 改写成 MDP。

#### 2.2.1 从 ODE 到 SDE：注入 stochasticity

Flow matching 的 deterministic ODE 是：

$$\mathrm{d}\mathbf{x}_t = \mathbf{v}_t \mathrm{d}t$$

其中 $\mathbf{v}_t$ 是 velocity field（model 学的），$\mathbf{x}_t$ 是 latent state at time $t$。RL 需要 stochastic policy 才能 explore，但 ODE 是 deterministic。Flow-GRPO 的 trick：把它升级成 reverse-time SDE，在保 marginal distribution 前提下注入 noise：

$$\mathrm{d}\mathbf{x}_t = \left[\mathbf{v}_t(\mathbf{x}_t) + \frac{\sigma_t^2}{2t}\left(\mathbf{x}_t + (1-t)\mathbf{v}_t(\mathbf{x}_t)\right)\right]\mathrm{d}t + \sigma_t \mathrm{d}\mathbf{w} \tag{1}$$

变量：
- $\mathbf{x}_t$：latent state at flow time $t \in [0, 1]$（$t=0$ 是 clean data，$t=1$ 是 pure noise，注意 flow matching 约定跟 diffusion 反过来）
- $\mathbf{v}_t(\mathbf{x}_t)$：velocity prediction at time $t$
- $\sigma_t$：噪声调度（控制 stochasticity 强度）
- $\mathbf{w}$：Wiener process（标准 Brownian motion），$\mathrm{d}\mathbf{w} \sim \mathcal{N}(0, \mathrm{d}t \cdot \mathbf{I})$
- $\frac{\sigma_t^2}{2t}(\mathbf{x}_t + (1-t)\mathbf{v}_t(\mathbf{x}_t))$：drift correction term，保证 marginal $p_t(\mathbf{x}_t)$ 不被扰动改变（这是 Fokker-Planck 反推出来的）

离散化得到 stochastic update rule，就是 policy $\pi_\theta$：

$$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \left[\mathbf{v}_\theta(\mathbf{x}_t, t) + \frac{\sigma_t^2}{2t}(\mathbf{x}_t + (1-t)\mathbf{v}_\theta(\mathbf{x}_t, t))\right]\Delta t + \sigma_t\sqrt{\Delta t}\,\epsilon \tag{2}$$

- $\Delta t$：time step 大小
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$：标准 Gaussian 噪声，每个 step 重新采样
- $\sigma_t\sqrt{\Delta t}\,\epsilon$：stochastic 增量，对应 Brownian 增量

这个 update rule 就是 "policy rollout" 一步。从 $t=1$ 走到 $t=0$，每步采样得到 trajectory $\{\mathbf{x}_t\}_{t=0}^T$。

#### 2.2.2 GRPO advantage 估计

给定 condition $\mathbf{c}$（这里是 prompt + camera trajectory encoded 在 noise 里），sample $G$ 条 trajectory $\{\mathbf{x}^i\}_{i=1}^G$。对每条最终生成 $\mathbf{x}_0^i$ 算 reward $R(\mathbf{x}_0^i, \mathbf{c})$，然后做 **group-relative advantage**：

$$\hat{A}_t^i = \frac{R(\mathbf{x}_0^i, \mathbf{c}) - \text{mean}(\{R(\mathbf{x}_0^i, \mathbf{c})\}_{i=1}^G)}{\text{std}(\{R(\mathbf{x}_0^i, \mathbf{c})\}_{i=1}^G)} \tag{3}$$

变量：
- $\hat{A}_t^i$：第 $i$ 条 trajectory 在 time $t$ 的 advantage
- $\text{mean}, \text{std}$：组内统计量，**这就是 GRPO 的精髓——用 group statistics 当 baseline，省掉 critic network**

GRPO 相比 PPO 最大优势：不需要 value function $V_\phi(s)$，因为 group-relative normalization 自然消除 common-mode reward（baseline 项）。在 video 这种高维输出 space 上，训 critic 是灾难性的 expensive。

#### 2.2.3 Clipped Objective + KL

$$\mathcal{J}(\theta) = \mathbb{E}_{\mathbf{c}, \{\mathbf{x}^i\}}\left[\frac{1}{T}\sum_{t=0}^{T-1}\left(\mathcal{L}_{\text{clip}}(r_t^i, \hat{A}_t^i) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right)\right] \tag{4}$$

- $r_t^i = \pi_\theta(\mathbf{x}_{t+1}^i | \mathbf{x}_t^i) / \pi_{\text{old}}(\mathbf{x}_{t+1}^i | \mathbf{x}_t^i)$：importance ratio
- $\mathcal{L}_{\text{clip}}$：PPO-style clip，限制 ratio 在 $[1-\epsilon, 1+\epsilon]$
- $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$：KL 到 reference policy（pre-trained Wan 2.1），防止 reward hacking 把 model 训崩
- $\beta$：KL penalty 权重

**Flow-GRPO-Fast** 加速 trick：在 deterministic ODE trajectory 上随机选中间 step 注入 noise 切到 SDE，不是每步都 SDE，这样 rollout 快很多，reward signal 质量损失小。

---

### 2.3 Reward Design: Analysis-by-Synthesis

这是 World-R1 的核心贡献——把 3D consistency 做成可微 reward signal。总 reward：

$$R(\mathbf{x}, c) = R_{\text{3D}}(\mathbf{x}, E, c) + \lambda_{\text{gen}} R_{\text{gen}}(\mathbf{x}, c) \tag{8}$$

- $\mathbf{x}$：generated video
- $E$：target camera trajectory（从 prompt 解析出来）
- $\lambda_{\text{gen}} = 1$（论文设定）

#### 2.3.1 $R_{\text{3D}}$ 三项之和

$$R_{\text{3D}} = S_{\text{meta}} + S_{\text{recon}} + S_{\text{traj}} \tag{9}$$

所有三项 limit 到 $[0, 1]$，直接相加，$R_{\text{3D}} \in [0, 3]$，$R_{\text{gen}} \in [-1, 1]$。

**Step 1: Lift video 到 3DGS**

用 Depth Anything 3 (Lin et al., 2025) 直接从 video clip forward predict 3D Gaussian Splatting representation $\Phi_{\text{GS}}$ 和对应 estimated camera trajectory $\hat{E}$。这是 feed-forward 3D reconstruction，不用 per-scene optimization，rollout 时快。

**(a) Geometric Integrity Score $S_{\text{meta}}$**：

把 $\Phi_{\text{GS}}$ 从一个 **novel meta-view**（跟 generation trajectory 显著 offset 的视角，比如从原点向后退）render 一张图，喂给 Qwen3-VL 当 semantic critic。VLM prompt 让它当 3D vision expert 打 0-9 分：

> "A good video (smooth, orbiting camera) creates a good pointmap. A bad video (static, jittery, or zooming) creates a bad pointmap."

为什么 meta-view 是关键？因为 **canonical view 容易骗 VLM**——baseline model 在原始生成视角上看起来不错（"billboard" effect），但从 meta-view 一 render 就露馅：原本应该 3D 的物体是 flat 的，遮挡关系错乱，floater 一堆。这正是 paper Figure A 演示的。最后 score 乘 0.1 归一化到 $[0, 1]$。

这一项的 intuition：3DGS 在 disocclusion / warping 区域会暴露 hallucination，VLM 在 meta-view 上能 semantic 判断这些 artifact。

**(b) Reconstruction Fidelity Score $S_{\text{recon}}$**：

从 $\hat{E}$ 重 render $\Phi_{\text{GS}}$ 得到 $\hat{\mathbf{x}}$，跟原 video $\mathbf{x}$ 算 LPIPS：

$$S_{\text{recon}} = 1 - \text{LPIPS}(\mathbf{x}, \hat{\mathbf{x}})$$

LPIPS 越低（重建越像原 video），reward 越高。这一项保证 3DGS 能 "忠实回放" 原 video——如果原 video 有 hallucination，3DGS 重建出来必然有 artifact，re-render 跟原 video 不匹配。

**(c) Trajectory Alignment Score $S_{\text{traj}}$**：

衡量 generated camera motion $\hat{E}$（3DGS 估出来）跟 input 指定 $E$ 的偏差：

- Translation 用 $L_2$ 距离
- Rotation 用 geodesic distance（rotation matrix 之间的 angular distance）
- reward = $\exp(-\text{deviation})$

这一项确保 model 不只是 "static-easy-to-reconstruct" cheat——必须真的按 prompt 动起来。这是 **anti-reward-hacking 的核心**：如果 model 偷懒生成 static video，3DGS 容易重建，$S_{\text{recon}}$ 会高，但 $S_{\text{traj}}$ 会低，因为没按 trajectory 动。

#### 2.3.2 $R_{\text{gen}}$: HPSv3 aesthetic

$$R_{\text{gen}}(\mathbf{x}) = \frac{1}{K}\sum_{t=0}^{K-1}\mathcal{H}(\mathbf{x}_t) \tag{10}$$

- $K$：取前 $K$ 帧
- $\mathcal{H}$：HPSv3 (Ma et al., ICCV 2025) 单帧 aesthetic score
- $R_{\text{gen}} \in [-1, 1]$

这是防止 3D reward 把 model 拽成 "ugly but consistent"——比如 grayscale low-detail output 容易 3D 重建但 aesthetic 0。HPSv3 守住 visual quality floor。

Paper Section D.7 (Table I, J) 验证 reward hacking 不会发生：w/o $S_{\text{traj}}$ → VBench 还行但 PSNR 跌，w/o $S_{\text{meta}}$ → VLM 看不到的 artifact 逃过惩罚，w/o $R_{\text{gen}}$ → 模型 collapse 成僵硬输出。三项缺一不可。

---

### 2.4 Periodic Decoupled Training: 防 rigid collapse

这是 World-R1 最巧的工程 trick。问题：纯 $R_{\text{3D}}$ 训下去，model 会把所有 dynamic 当 static 处理——lion 不抖 mane、waterfall 不流动、smoke 不飘——因为 dynamic 内容 3DGS 重建质量差，$S_{\text{recon}}$ 低，model 自然学会 "别动"。

解法：**周期性 reward 切换**。

- 100 个 step 全 reward ($R_{\text{3D}} + R_{\text{gen}}$) 强 3D consistency
- 之后一段 step **关掉 $R_{\text{3D}}$，只用 $R_{\text{gen}}$**，在 ~500 个 dynamic prompt subset 上 fine-tune
- 循环

这个 dynamic subset 全是高熵 scene：waterfall、lion roar、tornado、fire、moving crowd。$R_{\text{gen}}$ 把 foundation model 原本的 dynamic diversity 抓回来。

**Intuition**: 这非常像 LLM RLHF 里的 "forgetting mitigation"——RL 容易 catastrophic forget pretrain capability。World-R1 用 periodic relief（类似 cyclical learning rate / replay buffer 的精神）保护 dynamic prior。Ablation Table J：w/o periodic decoupled training → PSNR 居然上升到 27.89（更 rigid），但 VBench AVG 从 85.21 掉到 82.64。完美展示 reward trade-off 的本质。

---

### 2.5 Pure Text Dataset: 解耦视觉 prior

这部分是工程细节但很关键。World-R1 用 Gemini 生成 ~3000 个 text prompt，刻意 **不用 video dataset**。原因：open-domain video 数据集本身有 visual prior，model 会学到 "video dataset 长什么样"，而不是 "3D world 应该长什么样"。

数据集分层：
- **Natural Landscapes** (山脉、瀑布、天气)
- **Urban / Architectural** (城市、室内、infrastructure) 
- **Micro World** (宏观尺度物体、材质)
- **Fantasy / Surrealism** (非欧几何、违反物理)
- **Artistic Styles** (watercolor、cyberpunk、Van Gogh)

每个 prompt 配 camera trajectory：`push_in`, `orbit_left`, `move_right`, `pull_left`（composite：move_left → pull_out → pan_left）等。

**Dataset scaling** (Table E): 1K → 2K → 3K prompt，PSNR 单调上升 (25.82 → 26.54 → 27.63)，VBench AVG 同步涨 (83.23 → 84.76 → 85.21)。这说明 RL 的 data efficiency 很好，3K prompt 已经 significant gain，说明 foundation model 内部 "knows" 大部分东西，只需要 reward signal 把它 surface 出来。

---

## 3. 实验数据深度解读

### 3.1 3D Consistency 主表 (Table 2)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| CogVideoX-1.5-5B | 24.44 | 0.783 | 0.242 |
| Wan2.1-T2V-1.3B | 17.40 | 0.550 | 0.467 |
| Wan2.1-T2V-14B | 19.76 | 0.629 | 0.405 |
| **World-R1-Small** | **27.63** | **0.858** | **0.201** |
| **World-R1-Large** | **27.67** | **0.865** | **0.162** |

PSNR +10.23dB (Small) / +7.91dB (Large) 是巨大 jump。注意 Large base 的 PSNR (19.76) 居然比 Small base (17.40) 高，但 World-R1 之后 Small (27.63) 几乎追平 Large (27.67)——这说明 **base model scale 上去不一定能 fix 3D inconsistency**，但 RL alignment 能把 Small 拉到 Large 同等水平。这跟 LLM 里 "small model + RL > large model SFT" 的 pattern 一致（参考 DeepSeek-R1 7B 击败大很多倍 SFT model 的现象）。

### 3.2 VBench (Table 1) — General Quality 没退化

| Method | Aesthetic | Imaging | Motion Smooth | Subject Cons. | BG Cons. |
|---|---|---|---|---|---|
| Wan2.1-1.3B | 62.43 | 66.51 | 97.44 | 96.34 | 97.29 |
| ReCamMaster | 42.70 | 53.97 | 99.28 | 92.05 | 93.83 |
| GCD | 38.21 | 41.56 | 98.37 | 88.94 | 92.00 |
| **World-R1-Small** | **65.74** | **67.53** | **98.55** | **97.58** | 96.67 |

观察：
1. World-R1 **比 base Wan2.1 还高**（Aesthetic 65.74 vs 62.43），说明 RL 把 latent aesthetic 也 surface 出来了，可能是因为 $R_{\text{gen}}$ 的 HPSv3 跟 base model 训练 distribution 不完全 overlap
2. 显式 camera control 方法（ReCamMaster, GCD, Trajectory-Attention, DAS）Aesthetic 只有 38-43，**远低于** base foundation model 60+。原因：architectural injection 破坏 base model 的 visual prior。这是 World-R1 "不改架构" 路线的根本优势
3. BG Consistency 略降（96.67 vs 97.29）——这是 3D consistency reward 让背景 "跟着 camera motion 变化" 的必然结果（base 是静态背景作弊地保持在画面同一位置），实际上这是 feature 不是 bug

### 3.3 Camera Control 对比专门方法 (Table C)

| Method | RotErr ↓ | TransErr ↓ | CamMC ↓ |
|---|---|---|---|
| ReCamMaster | 1.53 | 3.12 | 4.17 |
| CamCloneMaster | 1.36 | 2.02 | 3.05 |
| Wan2.1-1.3B | 9.29 | 62.94 | 66.21 |
| Wan2.1-14B | 17.01 | 60.90 | 70.55 |
| **World-R1-Small** | 1.50 | 2.76 | 3.39 |
| **World-R1-Large** | 1.21 | 1.30 | 2.95 |

World-R1-Large 的 RotErr 1.21 / TransErr 1.30 / CamMC 2.95 **超过所有专门 camera control 方法**（包括 CamCloneMaster 这种 reference-based 专门方法）。这非常惊人——post-training 方法在专门 benchmark 上击败专门架构方法。这印证了 "explicit module 反而是 ceiling，latent capability 才能 exceed" 的 thesis。

变量解释：
- RotErr: rotation error (度)
- TransErr: translation error (归一化单位)
- CamMC: camera motion consistency 综合指标，越低越好

### 3.4 Multi-View Consistency Score (Table D) — Reconstruction-Independent 验证

| Method | MVCS ↑ |
|---|---|
| Wan2.1-1.3B | 0.974 |
| World-R1-Small | 0.989 |
| Wan2.1-14B | 0.963 |
| World-R1-Large | 0.993 |

MVCS 是 GeoVideo 提出的 reconstruction-independent metric，直接 cross-view 算 agreement。这避免了 "PSNR 提升只是因为 3DGS pipeline 偏好 World-R1 输出" 的 confound。0.989 vs 0.974 的 gap 证明 gain 真实存在于 video 本身。

### 3.5 Long-Video Generalization (Table F)

121 帧（训练是短 clip）：

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Wan2.1-14B | 18.32 | 0.558 | 0.534 |
| World-R1-Large | 26.32 | 0.828 | 0.257 |

PSNR +8dB on 121 帧！RL 在短 clip 上训的 geometric consistency **外推到 4× 长 video**。这呼应了 R1 类模型常见的 "reasoning chain generalization" 现象——RL 学到的 capability 比 SFT 更 transferable。

### 3.6 Per-Category Breakdown (Table G)

| Scene Type | Wan2.1-1.3B PSNR | World-R1-Small PSNR | Δ |
|---|---|---|---|
| Static | 20.14 | 30.52 | +10.38 |
| Single-obj Dynamic | 17.86 | 28.17 | +10.31 |
| Multi-obj Dynamic | 15.23 | 25.41 | +10.18 |
| Non-rigid Motion | 14.58 | 24.73 | +10.15 |
| Long-horizon | 12.53 | 23.59 | +11.06 |

5 个 category 都 ~+10dB，**gain 非常均匀**。Long-horizon gain 最大（+11.06），说明越是难场景 RL 越有价值（base model 越容易 hallucinate，RL 把它 push 回 consistency）。

### 3.7 User Study

25 人 blind test，World-R1 vs Wan2.1：
- Geometric Consistency: 92% win
- Camera Control Accuracy: 76% win
- Overall Preference: 86% win

86% overall 是非常强的偏好——通常 RL alignment 方法在 "specific capability" 上赢但 overall 因为 trade-off 会被反推。这里没发生，说明 reward 设计守住 general quality。

---

## 4. 跟你过去 work 的连接点

Andrej，几个值得联想的：

**1. "Video Model as World Model" 的 thesis 验证**

你 2024 年 World Model keynote 里说 Sora-like model 是 world simulator 的 precursor，但缺 explicit physics。World-R1 用 RL 把 latent physics surface 出来，等于实证 "video model 内部已经有 world representation，缺的是 alignment"。这跟 LLM 里 "model already knows, RL just elicit" 完全同构。

**2. RLHF 对 RLfromAIF 的视觉版映射**

Flow-GRPO + 3D reward model = RLHF 的视觉对应物。Reward model = Depth Anything 3 + Qwen3-VL + HPSv3 组合。这等于说 **AIF (AI Feedback) 在视觉领域已经可以 work**，因为 VLM 已经够强能 judge 3D plausibility。

**3. Reference Theory 视角**

你之前提过 LLM hallucination 是 "dreaming"，需要 grounding。Video model 同理——geometric hallucination 是 "visual dreaming"，3DGS reconstruction + meta-view render = "physics ground truth check"。Analysis-by-synthesis 跟你 eugf-style "render-and-verify" 思路同源。

**4. Curriculum / Periodic Training**

Periodic decoupled training 跟你 在 nanoGPT / educational 讲解里强调的 "interleaved learning" 同精神——单 task 训练易 collapse，多 task 周期切换能保 diversity。这跟 LLM RLHF 里 "rejection sampling SFT + RL" 交替的 recipe 也吻合。

**5. R1-zation of Video**

World-R1 = "R1-ification of Wan"。R1 让 LLM 学会 reasoning，World-R1 让 video model 学会 3D consistency。两者都是 post-training RL 把 latent capability explicit 化。我预测 2026 下半年会看到 "World-R1-Zero"（纯 RL 不需要 reward model，self-play with physics verifier）和 "World-R1-O3"（long-horizon world simulation via test-time search）。

---

## 5. Limitations 和未来方向

Paper 自己承认两个：
1. **RL 训练成本**：rollout + reward eval 每步都要 forward 一个 video，比 SFT 贵 ~5-10×。Flow-GRPO-Fast 已经 mitigate，但仍需 48-96 H200 GPU-days。
2. **Base model ceiling**：dense multi-object、fine hand dynamics、long-horizon 还是 inherit base artifact。

我额外想到几个：
- **Reward model 偏差**：Depth Anything 3 本身可能对某些 scene type 重建偏好（比如 indoor > outdoor），会 inject bias 到 RL
- **Camera trajectory 离散化**：只有 12 种 motion token，不能任意 trajectory。需要 trajectory tokenization 或 continuous conditioning
- **Static scene assumption**：3DGS 重建假设静态 scene，dynamic 物体本身就是 "violation"，$S_{\text{recon}}$ 会 penalize 合理的 dynamic。Periodic decoupled training 是 mitigation 而非 solution
- **Multi-modal prompt**：纯 text → video，没 image-conditioned 版本。World-R1 + I2V 是 obvious extension
- **Reward hacking in long horizon**：121 帧 gain 仍 +8dB 但比 short clip 的 +10dB 弱，longer horizon 可能 degrade

---

## 6. 关联工作（按你关心的脉络）

| 类别 | Paper | 链接 |
|---|---|---|
| Flow matching + RL | Flow-GRPO (Liu et al. NeurIPS 2025) | https://arxiv.org/abs/2506.22639 |
| Noise warping | Go-with-the-Flow (CVPR 2025) | https://arxiv.org/abs/2501.02603 |
| Feed-forward 3DGS | Depth Anything 3 | https://arxiv.org/abs/2511.10647 |
| 3D-aware video gen | Voyager (arxiv 2025) | https://arxiv.org/abs/2506.04225 |
| 3D-aware video gen | ViewCrafter (CVPR 2024) | https://arxiv.org/abs/2409.02048 |
| 3D-aware video gen | Vmem (ICCV 2025) | https://arxiv.org/abs/2506.05284 |
| Video world model | World Models (Ha & Schmidhuber) | https://arxiv.org/abs/1803.10122 |
| 3DGS original | Kerbl et al. (SIGGRAPH 2023) | https://arxiv.org/abs/2308.14737 |
| GRPO | DeepSeekMath | https://arxiv.org/abs/2402.03300 |
| HPSv3 | Ma et al. (ICCV 2025) | https://arxiv.org/abs/2504.01248 |
| Qwen3-VL | Alibaba | https://arxiv.org/abs/2511.21631 |
| Wan 2.1 | Alibaba | https://arxiv.org/abs/2503.20314 |
| LPIPS | Zhang et al. (CVPR 2018) | https://arxiv.org/abs/1801.03924 |
| VBench | (CVPR 2024) | https://arxiv.org/abs/2311.17982 |
| GeoVideo (reconstruction-independent metric) | | https://arxiv.org/abs/2512.03453 |

---

## 7. 总结：这篇 paper 的真正贡献

我读下来觉得 World-R1 真正的贡献不在 3D consistency 数字本身，而是 **paradigm shift**：

> Video generation model 的 3D inconsistency 不是 model capacity 问题，是 alignment 问题。  
> 用 RL + discriminative reward model（3DGS + VLM + aesthetic model）就能把 latent capability surface 出来，不需要架构改造、不需要 3D 数据集、不需要 inference-time constraint。

这跟 R1 在 LLM 上的贡献是平行的——证明 RL 是 unlock latent capability 的 scalable 路径，比 SFT data scaling 更 efficient。

我个人会特别 watch：
1. **World-R1-O3 / World-R1-Zero**：纯 self-play 不用 reward model
2. **World-R1 + Embodied**：用 video rollout 训 robot policy（World model 当 imagination engine）
3. **World-R1-Reasoning**：把 3D consistency reward 扩展到 physical reasoning（碰撞、重力、流体）
4. **Test-time scaling**：rollout multiple trajectory + 3D verifier 选 best，类似 inference-time search

这篇 paper 给 2026 视觉生成领域定了一个清晰的方向标。期待你下次在 YouTube 上讲这个 work。
