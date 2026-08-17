---
source_pdf: minWM A Full-Stack Open-Source Framework for Real-Time Interactive Video
  World Models.pdf
paper_sha256: e36f5446defcc27a236d4d9010e975bce16eb3a4e5e63489c7ffdc2a980743c6
processed_at: '2026-08-05T18:47:34-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# minWM 用大白话讲

## 一句话总结

minWM 就是把现在的 video foundation model（Wan2.1、HunyuanVideo 这些）改造成"能实时交互的世界模型"的一整套开源 toolkit。

🔗 https://github.com/shengshu-ai/minWM

## 为什么需要这个东西

现在的 video generator 很猛，Sora [1]、Wan2.1 [6]、HunyuanVideo [7] 都能生成高质量视频。但你想拿来做"世界模型"——玩家按个键，camera 转一下，画面跟着转——直接用不行，卡在三个点：

1. **Bidirectional attention 的问题**：现在模型生成时所有帧同时看，后面的 frame 能"偷看"前面，这违反 time causality。要做交互必须只看过去，不能偷看未来。

2. **没有 action 接口**：你说"camera 往左转"，模型听不懂，它只会听 text prompt。

3. **太慢**：生成一整段 video 要等几百秒，这玩个毛线交互。Table 1 里 HY1.5 bidirectional 要 771 秒/首帧，Wan2.1 要 269 秒。

minWM 把这三个问题打包解决，最后把 latency 压到 1.1 秒（Wan2.1）和 3.4 秒（HY1.5），进入交互可用区间。

## 整个 pipeline 在干嘛

```
T2V/TI2V Foundation Model (multi-step bidirectional, 几百秒latency)
    ↓ Phase 1: Camera-Controllable Training (PRoPE)
    ↓ 学会"camera 往哪走，画面往哪变"
Camera-Controllable Bidirectional Model
    ↓ Phase 2 Stage 1: AR Training (teacher forcing)
    ↓ 从"看全部帧"改成"只看过去帧"
Multi-step AR Model
    ↓ Phase 2 Stage 2: Causal ODE / Causal CD
    ↓ 从 50-100 步 denoise 压到 4 步
Few-step AR Model (quality 还不行)
    ↓ Phase 2 Stage 3: Asymmetric DMD
    ↓ 用 bidirectional teacher 把 quality 拉上去
Few-step AR Model (quality OK, 1-3秒latency, 可交互)
```

核心 idea 就是"渐进式 distillation"——一步到位太难，分几步走，每步只改一个 dimension。

## Phase 1：教模型听懂 camera 指令

### PRoPE 的 trick

这一步让模型学会 camera control。trick 叫 PRoPE [26]，全称是 Camera as Relative Positional Encoding 的变种。

做法：给每一帧加一个"相机标签"——camera intrinsic $K_i$（焦距、光心）+ camera extrinsic $T_i^{cw} \in SE(3)$（位置、朝向）。然后塞进 self-attention 里。

具体数学：构造一个 lifted projective matrix：

$$\widetilde{P}_i = \begin{bmatrix} [K_i \mid 0] \\ e_4^\top \end{bmatrix} T_i^{cw} \in \mathbb{R}^{4\times 4}$$

- $K_i$: 3×3 相机内参
- $T_i^{cw}$: 4×4 world-to-camera 变换（$SE(3)$）
- $[K_i \mid 0]$: 3×4 augmented matrix
- $e_4 = (0,0,0,1)^\top$: 补齐 homogeneous coordinate 让矩阵可逆

然后对 token $t$（属于 frame $i(t)$，空间坐标 $(x_t, y_t)$）构造 block-diagonal encoding：

$$D_t^{PRoPE} = \begin{bmatrix} I_{d/8} \otimes \widetilde{P}_{i(t)} & 0 & 0 \\ 0 & [\text{RoPE}_{d/4}(x_t) \mid 0] & 0 \\ 0 & 0 & [\text{RoPE}_{d/4}(y_t)] \end{bmatrix}$$

- $I_{d/8}$: $d/8$ 维单位矩阵
- $\otimes$: Kronecker product，把 camera info 复制到 $d/8$ 个 channel group
- $\text{RoPE}_{d/4}(x_t)$, $\text{RoPE}_{d/4}(y_t)$: 标准 RoPE，encode 2D 空间位置
- 整体 $d/2 + d/4 + d/4 = d$ 维，刚好是 token dimension

**为什么巧妙**：当 token $t_1$（frame 1）和 token $t_2$（frame 2）做 attention 时，effective transformation 是：

$$\widetilde{P}_{i(t_1)} \widetilde{P}_{i(t_2)}^{-1} = \begin{bmatrix} K_{i(t_1)} & 0 \\ 0 & 1 \end{bmatrix} T_{i(t_1)}^{cw} (T_{i(t_2)}^{cw})^{-1} \begin{bmatrix} K_{i(t_2)}^{-1} & 0 \\ 0 & 1 \end{bmatrix}$$

这就是 multi-view geometry 里的 **epipolar geometry** 核心量——frame 2 中 3D point 的 pixel projection 转换到 frame 1 的 pixel projection。

模型不用显式学几何约束，attention 机制自动 express 出来。因为 PRoPE 把几何先验 encode 进了 attention operation 内部，不是外部 condition。

🔗 PRoPE 原论文: https://papers.nips.cc/paper_files/paper/2025

### 训练数据的大坑

这一步是个血泪史。一开始他们用 SpatialVid [34]（带 camera pose 标注的视频数据集），结果训不动。即使加 filtering 也不行。

**为啥训不动**：SpatialVid 的 camera pose 是 perception model 估出来的，有 noise。即使 pose error 在 pixel 级别很小，对 attention 里的 relative pose encoding 也足以破坏几何一致性。

**解决方案**（很务实）：

| Data Source | Pose 来源 | 结果 |
|-------------|-----------|------|
| SpatialVid [34] | Perception-estimated | 失败 |
| DL3DV [35] + re-rendering | 3D reconstruction GT | 成功 |
| OpenVid [36] + WorldPlay [8] generation | WorldPlay simulated GT | 成功 |

DL3DV 是 3D scene dataset，重建场景后沿指定 trajectory render 视频，pose 是 ground truth。WorldPlay 是已有的 world model，用它生成视频，pose 也是已知的。

**硬道理**：camera control 的训练数据，pose 必须精确，感知估出来的不行。

这里有个有意思的 bootstrap loop：训练新 world model 需要高质量 camera-annotated data → 用现有 world model (HY-WorldPlay) 生成 data → 训练新 world model。类似 RLHF 的 iterative improvement 或 AlphaGo 的 self-play。

🔗 SpatialVid: https://arxiv.org/abs/2509.09676
🔗 DL3DV: https://arxiv.org/abs/2310.19369
🔗 WorldPlay: https://arxiv.org/abs/2512.14614

### Emergent Behavior

训练步骤上发现 camera control 是"涌现"的：

| Steps (HY1.5) | 行为 |
|---------------|------|
| 1K - 2K | 完全 uncontrollable |
| ~5K | 开始有 controllability，但 unstable |
| 8K | 强 controllability，reliable |

这跟 LLM instruction tuning 里 capability emergence 很像——模型需要先 internalize PRoPE 的 geometric encoding 机制，然后才能 reliable 响应 camera input。

## Phase 2：AR 化 + 少步推理

这一阶段更复杂，三个 stage，核心思想是"渐进式 distillation"。

### Stage 1：变成 AR 模型

原来 model 是 bidirectional（所有帧一起看），现在要变成 autoregressive（只看前面的帧）。

做法：teacher forcing。把 clean video $x^{<i}$ 和 noisy version $x_t^i$ 拼起来，加 causal attention mask 训练。

训练时 history 用 GT data，这叫 teacher forcing 假设。

**做完两个问题**：
1. 还要多步 denoise（50-100步），慢
2. Exposure bias：训练时 history 是 GT，推理时 history 是自己生成的，distribution shift

🔗 Teacher forcing 在 AR video 中的应用：MAGI-1 [27] https://arxiv.org/abs/2505.13211

### Stage 2：减少 denoising 步数

把 50-100 步压到 4 步。两个 option：

**Option A: Causal ODE Distillation** [23]

用 Stage 1 的 AR teacher 生成一堆 intermediate trajectory（PF-ODE trajectory），然后训练 student 从 intermediate point 直接回归 clean frame：

$$\theta^* = \arg\min_\theta \mathbb{E}_{x_{gt}^{<i}, t, i, x_t^i} \Big[ \| G_\theta(x_t^i, x_{gt}^{<i}, t) - x_0^i \|^2 \Big]$$

- $x_{gt}^{<i}$: real data 构成的 historical prefix
- $t \sim \mathcal{S}$: 在 few-step timestep 集合中采样
- $x_t^i$: AR teacher 在 timestep $t$ 的 intermediate state
- $x_0^i$: clean target frame
- $G_\theta$: few-step student generator

缺点：要离线生成和存储 ODE trajectory，对长 video、大 model 存储开销大。

**Option B: Causal Consistency Distillation** [24]

Causal Forcing++ 的改进，online 计算，不用离线存数据：

$$\theta^* = \arg\min_\theta \mathbb{E}_{x_{gt}, \epsilon, t, i} \Big[ w(t) \cdot d\Big(G_\theta(x_t^i, x_{gt}^{<i}, t),\ G_{\theta^-}(\hat{x}_{t-\Delta t}^i, x_{gt}^{<i}, t-\Delta t)\Big) \Big]$$

- $\hat{x}_{t-\Delta t}^i$: 从 $x_t^i$ 用 AR teacher 做一步 ODE step 得到
- $\theta^-$: $\theta$ 的 EMA copy，stop-gradient
- $w(t)$: timestep-dependent weight
- $d(\cdot, \cdot)$: pre-defined norm 下的 distance

**Intuition**：consistency distillation [33] 的核心是"同一 ODE trajectory 上任意点都应映射到相同 final output"。这里加到 AR 框架中，叫 causal CD。理论上等价于 ODE distillation，但 online 计算，省存储省时间。

🔗 Causal Forcing [23]: https://arxiv.org/abs/2602.02214
🔗 Causal Forcing++ [24]: https://arxiv.org/abs/2605.15141
🔗 Consistency Models [33]: https://arxiv.org/abs/2303.01969

### Stage 3：用 Asymmetric DMD 拉质量

Stage 2 做完，student 能 4 步生成，但 quality 受限——AR teacher 自己质量就不如 bidirectional model。

这一步用 **asymmetric DMD** [20, 30] 把 student 推到 bidirectional model 的 quality 水平。

DMD gradient 公式：

$$\nabla_\theta \mathbb{E}_t \big[ D_{KL}(p_{\theta,t}(\tilde{x}_t) \| p_{\text{data},t}(\tilde{x}_t)) \big] = -\mathbb{E}_{\tilde{x}, t, \tilde{x}_t} \Big[ \big(s_{\text{real}}(\tilde{x}_t, t) - s_{\text{fake}}(\tilde{x}_t, t)\big) \frac{\partial \tilde{x}}{\partial \theta} \Big]$$

- $\tilde{x}$: student 通过 self-rollout 生成的 full video
- $\tilde{x}_t \sim p_\theta(\tilde{x})$: forward diffusion 加噪到 timestep $t$
- $p_{\theta,t}$: student distribution 在 timestep $t$ 的 marginal
- $p_{\text{data},t}$: real data distribution 在 timestep $t$ 的 marginal（由 bidirectional teacher 代表）
- $s_{\text{real}}$: **frozen** bidirectional teacher 估的 score function $\nabla \log p_{\text{data}}(\tilde{x}_t)$
- $s_{\text{fake}}$: **online-trained** diffusion model 估的 score function $\nabla \log p_\theta(\tilde{x}_t)$
- $\frac{\partial \tilde{x}}{\partial \theta}$: reparameterization 把 gradient 传到 student

**Intuition**：$s_{\text{real}} - s_{\text{fake}}$ 是 sample 应该移动的方向（从 fake distribution 指向 real distribution），$\frac{\partial \tilde{x}}{\partial \theta}$ 是 sample 对 parameter 的 sensitivity。gradient 让 $\theta$ 调整使 sample 沿对的方向移动，KL 减小。

**为啥叫 asymmetric**：student 是 few-step AR model，teacher 是 multi-step bidirectional model，架构完全不同。DMD 的好处是不需要 discriminator，只要能估 score 就行，所以架构不对称也能 work。

**关键 trick：self-rollout**：student 用自己的 rollout 当 history 训练，gradient 通过 chain rule 让 history 也参与优化，自然 align train/test distribution，解决 exposure bias。

🔗 DMD [30]: https://arxiv.org/abs/2310.06668
🔗 FrameStamp [20]: https://arxiv.org/abs/2501.xxxxx
🔗 Self-Forcing [22]: https://arxiv.org/abs/2506.08009

## 几个关键 Intuition

### 1. 为什么渐进式 distillation 优于 single-stage

直接从 bidirectional multi-step distill 到 AR few-step，distribution gap 太大：
- bidirectional vs AR（attention mechanism 不同）
- multi-step vs few-step（generation trajectory 不同）

Score function 估计容易 collapse，self-rollout 时 student 容易 drift 到 OOD region。

渐进式：每 stage 改一个 dimension
- Stage 1: architecture mechanism（bidirectional → AR）
- Stage 2: efficiency（multi-step → few-step）  
- Stage 3: quality（low-quality AR → high-quality bidirectional-level）

每 stage 的 student/teacher gap 控制在可学习范围。

### 2. PRoPE 为什么 work

对比几种 camera conditioning 方案：

| Method | 优点 | 缺点 |
|--------|------|------|
| Camera params concat with text | Simple | 难 encode 3D 几何 |
| Per-frame embedding + cross-attn | 灵活 | 绝对 pose，缺相对关系 |
| Plücker embedding | 经典 ray encoding | 需要 dense ray sampling |
| **PRoPE** | 自动 relative pose，attention-native | 需要 inverse 计算 |

PRoPE 把 3D geometry encode 进 attention operation 内部，让 model 通过 attention 自然利用 epipolar geometry 约束。这是 inductive bias 的胜利。

### 3. 为什么 DMD 不对称也能 work

DMD gradient 只依赖 score function estimate，不管架构。student 和 teacher 架构不同没关系，只要在同一个 data-conditioned manifold 上。

**Implicit assumption**：bidirectional model 和 AR model 在 same camera-conditioned manifold 上。Causal Forcing 的渐进 distillation 保证了这一点——每 stage 的 teacher 都在引导 student 留在 manifold 上。

### 4. Bootstrap 数据的 implicit risk

用 HY-WorldPlay 生成 training data 训新 world model：
- 优点：解决 GT trajectory 稀缺
- 风险：HY-WorldPlay 的 bias 会 propagate 到新 model，长期可能 mode collapse 到 HY-WorldPlay 的 generation manifold

未来需要：multi-model ensemble 生成 diverse data，或 hybrid 策略（3D reconstruction + WorldPlay generation）。

## 实验数据深入看

### Latency 表

| Base model | Model type | First-frame latency (s) | Speedup |
|------------|------------|------------------------|---------|
| HY1.5 [7] | Multi-step bidirectional | 771.041 | 1.00× |
| HY1.5 | Multi-step AR | 81.014 | 9.52× |
| HY1.5 | **Few-step AR** | **3.446** | **223.75×** |
| Wan2.1 [6] | Multi-step bidirectional | 269.055 | 1.00× |
| Wan2.1 | Multi-step AR | 28.651 | 9.39× |
| Wan2.1 | **Few-step AR** | **1.137** | **236.64×** |

**Latency reduction 两个来源**：

1. **Bidirectional → Multi-step AR** (~9.5×)
   - Bidirectional 要等所有 77 frames 一起 denoise
   - AR 第一 frame latency 只需 denoise chunk_size=4 frames
   - 理论 77/4 ≈ 19×，实际 9.5×（有 overhead）

2. **Multi-step AR → Few-step AR** (~23×)
   - Multi-step 50-100 steps
   - Few-step 4 steps
   - 理论 ~12-25×，实测 ~23×

**两个 base model 对比**：
- HY1.5 (8B) bidirectional 771s vs Wan2.1 (1.3B) 269s，比值 ~2.86×，远小于 parameter ratio (8/1.3 ≈ 6.15)
- Few-step AR: HY1.5 3.4s vs Wan2.1 1.1s，比值 ~3.1×，接近 parameter ratio
- 说明 few-step regime 下 compute-bound 更 dominant，multi-step regime 下 memory/communication overhead 占比大

**实际部署注意**：
- VAE 时间 exclude 了，实际端到端 latency 会更高（VAE encode/decode 是 sequential bottleneck）
- Single A800 GPU，consumer 4090 估计 3-5× latency
- Wan2.1 的 1.1s 在 real-time threshold (<2s perception threshold) 内
- HY1.5 的 3.4s 接近 near real-time

### Training Hyperparameters

| Model | Phase | Batch | LR | Steps |
|-------|-------|-------|-----|-------|
| HY1.5 | Bidirectional FT | 32 | 1e-5 | 8K |
| HY1.5 | Causal Stage 1 | 32 | 1e-5 | 4K |
| HY1.5 | Causal Stage 2 | 32 | 1e-5 | 1.5K |
| HY1.5 | Causal Stage 3 | 32 | 1e-5 | 500 |
| Wan2.1 | Bidirectional FT | 32 | 2e-6 | 5K |
| Wan2.1 | Causal Stage 1 | 32 | 2e-6 | 4K |
| Wan2.1 | Causal Stage 2 | 32 | 2e-6 | 2K |
| Wan2.1 | Causal Stage 3 | 32 | 2e-6 | 200 |

**观察**：
- Wan2.1 LR 比 HY1.5 小 5×，可能因为 Wan2.1 cross-attention conditioning 对 LR 更敏感
- Stage 3 steps 远少于其他 stage，因为 DMD 是 fine-grained distribution alignment，不需要太多 steps
- HY1.5 Stage 3 (500) 比 Wan2.1 (200) 多 2.5×，8B model 需要更多 steps 收敛

### Minimal Batch Size Ablation

| Batch size (Wan2.1) | 结果 |
|---------------------|------|
| <4 | 经常失败 |
| 8 | 改善但 unstable |
| 16 | 成功，high controllability |

**深层原因**：PRoPE 的 camera pose 在 $SE(3)$ 上是 6-DoF 高维空间，single batch 内 camera trajectory 多样性不足时 gradient 估计 variance 过大。需要足够 batch 覆盖不同 camera pose 的相对关系。

**实践意义**：academic research with limited compute，batch 16 + Wan2.1-1.3B 是 minimum viable setup。HY1.5-8B batch 32 需要多卡（推测至少 8×A800）。

## Architecture Generalization

minWM 在两种架构上 instantiate：

### Wan2.1-T2V-1.3B: Cross-Attention Conditioning
- Text condition 通过 cross-attention 注入
- 标准 DiT-style architecture
- Camera condition 通过 PRoPE 注入 self-attention

### HY1.5-TI2V-8B: MMDiT Architecture [31]
- Text 和 image token 在同一个 attention 中交互
- Modal fusion 在 attention level
- Camera condition 同样通过 PRoPE 注入

**两种架构都 work** 的意义：PRoPE 的 injection mechanism 是 architecture-agnostic 的，因为它是 self-attention 层面的 transformation，与 condition injection mechanism 解耦。

🔗 Wan2.1: https://arxiv.org/abs/2503.20314
🔗 HunyuanVideo: https://arxiv.org/abs/2412.03603
🔗 SD3/MMDiT: https://arxiv.org/abs/2403.03206

## Research Landscape 中 minWM 的位置

### Interactive World Model 相关工作

| Work | Key Feature | 与 minWM 关系 |
|------|-------------|----------------|
| Genie 3 [9] | DeepMind closed-source frontier | minWM 是开源 alternative |
| Hunyuan-GameCraft-2 [10] | Instruction-following game world | action conditioning 扩展方向 |
| Yume-1.5 [11] | Text-controlled interactive | 类似但更早 |
| Vidarc [12] | Embodied closed-loop control | 强调 closed-loop |
| Live Avatar [13] | Audio-driven real-time avatar | 不同 modality |
| StreamAvatar [14] | Streaming diffusion for avatar | 流式架构 |
| Relic [15] | Long-horizon memory | 长视频一致性方向 |
| Yan [16] | Foundational interactive video | 大规模 foundation |
| Pan [17] | General interactive world sim | 通用世界模拟 |
| Matrix-Game 2.0 [18] | Open-source real-time streaming | 最接近竞品 |
| Motion Stream [19] | Real-time with motion controls | motion control 方向 |

### Distillation 方法对比

| Work | Distillation Type | 在 minWM 中的角色 |
|------|-------------------|-------------------|
| DMD [30] | One-step image distillation | Stage 3 基础 |
| FrameStamp [20] | Bidirectional → AR video | 灵感来源 |
| DAP [21] | Diffusion adversarial post-training | GAN-style alternative |
| Self-Forcing [22] | Train-test gap bridging | 互补 |
| Causal Forcing [23] | AR diffusion distillation | Stage 1-2 核心 |
| Causal Forcing++ [24] | Scalable causal CD | Stage 2 alternative |
| Adversarial self-distillation [25] | One-step causal video | GAN-style alternative |
| ProlificDreamer/VSD [28] | Variational score distillation | DMD 前身 |
| Diff-Instruct [29] | Universal diffusion transfer | DMD 理论基础 |
| Consistency Models [33] | Single-step via consistency | Causal CD 基础 |

🔗 Genie 3: https://deepmind.google/discover/blog/genie-3-the-frontier-of-interactive-world-models/
🔗 Matrix-Game 2.0: https://arxiv.org/abs/2508.13009
🔗 DAP: https://arxiv.org/abs/2501.08316
🔗 ProlificDreamer: https://arxiv.org/abs/2305.01391
🔗 Diff-Instruct: https://arxiv.org/abs/2305.15727

## Model-Based RL 的 Connection

minWM 本质是 action-conditioned generative world model：
- State: video history
- Action: camera trajectory (current) → 可扩展到其他 action space
- Transition: $p(s_{t+1} | s_{\leq t}, a_{\leq t})$
- Reward: implicit (高质量 generation)

对 model-based RL：用 minWM 作为 differentiable simulator，camera control 对应 navigation action，可扩展到 robot manipulation（pose action）。

🔗 Model-based RL 综述: https://arxiv.org/abs/2206.05343

## Karpathy 视角的潜在应用

作为 Tesla 前 AI director，autonomous driving simulation 的 connection 很直接：

- minWM 的 camera control 直接对应 ego-motion
- 可用于 generating corner case scenarios
- Real-time inference 支持 closed-loop simulation with AD policy
- 但 77 frames (~2.5s @ 30fps) 太短，需扩展长视频

可能的 extension：
- Bird's eye view representation 与 PRoPE 的 3D encoding 结合
- Multi-camera setup (Tesla 的 8-camera) 与 multi-camera PRoPE
- Action-conditioning 扩展到 vehicle dynamics
- Long-horizon: 结合 Relic [15] 的 memory mechanism

## Limitations 和 Open Questions

### Paper 显式提到的 limitation
- 只支持 camera control，未支持 pose, action 等
- 8B model 在 consumer GPU 上仍 challenge
- SpatialVid-based training 未 work，需要更精细 pose refinement

### 隐含 limitation（paper 没明说）

1. **Quality metrics 缺失**：没有 FID, FVD, VBench 等 quantitative 评估，只有 qualitative figures
2. **Long video 未涉及**：77 frames 是固定长度，长视频 generation 的 coherence 未测试
3. **Comparison 缺失**：与 GameNGen, Genie 3, Matrix-Game 2.0 等无 direct comparison
4. **VAE bottleneck**：latency 数字 exclude VAE，实际 deployment 中 VAE encode/decode 是 sequential bottleneck
5. **Closed-loop interaction**：camera trajectory 是 predefined，未支持 reactive action（如游戏 NPC 响应）

### Future Directions

**Action space 扩展**：
- Object manipulation actions
- Agent movement (keyboard/mouse)
- Multi-modal actions (audio + camera + action)

**Long-horizon coherence**：
- Memory mechanism (Relic [15] 方向)
- Hierarchical generation (coarse-to-fine)
- Scene state tracking

**Higher resolution**：
- 480p → 720p/1080p
- Real-time 4K 需要 further distillation (1-step)

**Closed-loop**：
- Reactive generation responding to user
- Online model update with user feedback
- Multi-agent interaction

**Multi-camera**：
- Surround view (autonomous driving)
- Multi-view consistency constraint
- Cross-camera PRoPE extension

## 我的看法

这篇 paper 本质是 **engineering integration paper**。PRoPE、Causal Forcing、DMD 都是 prior work，minWM 的贡献是把它们整合成 reproducible pipeline，开源出来，做了扎实 ablation 告诉 community 踩坑经验。

这种工作对领域很有价值。学术圈一直缺这种 "full-stack reproducible recipe"，大家都在搞 SOTA 数字，没人愿意做 infrastructure 工作。minWM 有点像 LangChain 之于 LLM application，它定义了 standard pipeline，让后续工作可以站在它肩膀上。

但 paper 也有明显短板：
1. 没有 quantitative quality 评估，只有 qualitative figure
2. 没跟其他 world model 直接比
3. VAE 时间没算进去，实际 deployment latency 会更高
4. 77 帧太短，长视频一致性没测
5. 只支持 camera control，action space 太窄

总体而言，minWM 是 solid engineering paper，对 real-time interactive video world model 领域的开源生态有显著贡献，可作为后续研究的基础 infrastructure。

## 相关联想和 wild speculation

1. **AlphaGo 的 self-play analog**：用 WorldPlay 生成 data 训新 world model，这个 bootstrap loop 类似 AlphaGo 的 self-play。如果 iterate 多轮，会不会出现 world model 的 "capability jump"？

2. **World model 的 scaling law**：PRoPE 的 6-DoF camera pose 空间，加上未来扩展到更多 action dimension，action space 的 dimension 和 model size、data size 的 scaling 关系是什么？会不会有类似 LLM Chinchilla law 的 optimal allocation？

3. **Dreamer 的 differentiable simulator**：minWM 作为 action-conditioned world model，直接 plug 进 Dreamer [model-based RL] 替换 learned dynamics model，policy gradient 直接 backprop through minWM rollout。这对 robot learning 可能是 game changer。

4. **Game engine 的未来**：如果 minWM 这种 generative world model 质量足够高，会不会取代传统 game engine？Epic 那种手写 physics + asset pipeline 的模式，vs generative model "learn everything from data" 的模式。中期可能是 hybrid，长期可能 generative 主导。

5. **Embodied AGI 的 perception-action loop**：world model 是 embodied AGI 的核心组件。minWM 这种 real-time interactive world model，加上 perception module 和 policy module，构成完整的 embodied agent。Karpathy 你之前说 "software 2.0" 的 vision，world model 可能是 software 3.0 的 core primitive。

6. **Data efficiency 的 bootstrap problem**：minWM 用 WorldPlay 生成 data 训新 world model，这个 loop 如果 iterate，每轮新 world model 质量提升，生成的 data 质量也提升，理论上可以 bootstrap 到任意高质量。但会不会有 mode collapse？会不会有 bias amplification？这是个 open question。

7. **Camera control 作为 "easiest action space"**：minWM 选 camera control 作为第一个 action，因为 camera pose 在 $SE(3)$ 上有 clean mathematical structure（PRoPE 能 encode）。但 robot manipulation、game action 这些 action space 更复杂，没有 clean structure，extension 可能遇到本质困难。

8. **OOD generalization**：minWM 训练 data 主要是 WorldPlay 生成的，如果 deploy 到 OOD scene（比如 user 上传的 arbitrary video），world model 的 generalization 如何？这关系到 product deployment 的 robustness。

## 完整 Reference 链接

**Project & Code:**
- minWM Project: https://github.com/shengshu-ai/minWM

**Foundation Models:**
- Sora [1]: https://openai.com/sora
- Vidu [2]: https://arxiv.org/abs/2405.04233
- CogVideoX [3]: https://arxiv.org/abs/2408.06072
- Open-Sora Plan [4]: https://arxiv.org/abs/2412.00131
- Open-Sora [5]: https://arxiv.org/abs/2412.20404
- Wan2.1 [6]: https://arxiv.org/abs/2503.20314
- HunyuanVideo/HY1.5 [7]: https://arxiv.org/abs/2412.03603

**World Models:**
- WorldPlay [8]: https://arxiv.org/abs/2512.14614
- Genie 3 [9]: https://deepmind.google/discover/blog/genie-3-the-frontier-of-interactive-world-models/
- Hunyuan-GameCraft-2 [10]: https://arxiv.org/abs/2511.23429
- Yume-1.5 [11]: https://arxiv.org/abs/2512.22096
- Vidarc [12]: https://arxiv.org/abs/2512.17661
- Live Avatar [13]: https://arxiv.org/abs/2512.04677
- StreamAvatar [14]: https://arxiv.org/abs/2512.22065
- Relic [15]: https://arxiv.org/abs/2512.04040
- Yan [16]: https://arxiv.org/abs/2508.08601
- Pan [17]: https://arxiv.org/abs/2511.09057
- Matrix-Game 2.0 [18]: https://arxiv.org/abs/2508.13009
- Motion Stream [19]: https://arxiv.org/abs/2511.01266

**Distillation Methods:**
- FrameStamp [20]: https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.html
- DAP [21]: https://arxiv.org/abs/2501.08316
- Self-Forcing [22]: https://arxiv.org/abs/2506.08009
- Causal Forcing [23]: https://arxiv.org/abs/2602.02214
- Causal Forcing++ [24]: https://arxiv.org/abs/2605.15141
- Adversarial self-distillation [25]: https://arxiv.org/abs/2511.01419
- PRoPE [26]: https://papers.nips.cc/paper_files/paper/2025
- MAGI-1 [27]: https://arxiv.org/abs/2505.13211
- ProlificDreamer [28]: https://arxiv.org/abs/2305.01391
- Diff-Instruct [29]: https://arxiv.org/abs/2305.15727
- DMD [30]: https://arxiv.org/abs/2310.06668
- SD3/MMDiT [31]: https://arxiv.org/abs/2403.03206
- Score SDE [32]: https://arxiv.org/abs/2011.13456
- Consistency Models [33]: https://arxiv.org/abs/2303.01969

**Datasets:**
- SpatialVid [34]: https://arxiv.org/abs/2509.09676
- DL3DV [35]: https://arxiv.org/abs/2310.19369
- OpenVid [36]: https://arxiv.org/abs/2407.02371

---

# minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models 深度解析

## 1. Paper 的核心定位与 motivation

minWM 解决的是 video generation 领域一个关键的 engineering gap：现有的 bidirectional T2V/TI2V foundation models（Sora [1]、Vidu [2]、CogVideoX [3]、Open-Sora [4,5]、Wan2.1 [6]、HunyuanVideo/HY1.5 [7]）虽然 generation quality 高，但无法直接用作 interactive world model，因为缺乏三个核心特性：

1. **Causality**: bidirectional attention 同时看前后帧，违反时间因果性
2. **Controllability**: 缺乏显式 action conditioning（如 camera trajectory）
3. **Low latency**: multi-step denoising + 全序列生成导致 first-frame latency 在 hundreds of seconds 量级

minWM 把这三个问题作为一个 unified full-stack pipeline 解决，关键 contribution 是 **复现性 + 可扩展性**，而非单一 SOTA checkpoint。

🔗 Project Page: https://github.com/shengshu-ai/minWM

## 2. Pipeline 总览：两阶段、四 stage 设计

```
T2V/TI2V Foundation Model (multi-step bidirectional)
        ↓ Phase 1: Camera-Controllable Training (PRoPE)
Camera-Controllable Multi-step Bidirectional Model
        ↓ Phase 2 Stage 1: AR Diffusion Training (teacher forcing)
Camera-Controllable Multi-step AR Model
        ↓ Phase 2 Stage 2: Causal ODE init / Causal CD init
Camera-Controllable Few-step AR Model (low quality)
        ↓ Phase 2 Stage 3: Asymmetric DMD
Camera-Controllable Few-step AR Model (high quality, real-time)
```

这个渐进式 distillation 设计的核心 intuition 在于：每 stage 解决一个不同维度的 problem，避免 single-stage distillation 的 distribution gap 过大导致 training instability。

- Phase 1: 加上 **3D geometry prior** (controllability)
- Stage 1: 加上 **autoregressive mechanism** (causality)
- Stage 2: 加上 **few-step efficiency** (latency)
- Stage 3: 对齐 **quality distribution** (fidelity)

## 3. Phase 1: PRoPE — Camera Control 的数学原理

### 3.1 Lifted Projective Matrix

给定 video clip 的 per-frame camera parameters $\{(K_i, T_i^{cw})\}_{i=1}^N$：
- $K_i \in \mathbb{R}^{3\times3}$: 相机内参矩阵
- $T_i^{cw} \in SE(3)$: world-to-camera 外参变换，即 $\begin{bmatrix} R_i & t_i \\ 0 & 1 \end{bmatrix}$

PRoPE 构造 lifted projective matrix：

$$\widetilde{P}_i = \begin{bmatrix} [K_i \mid 0] \\ e_4^\top \end{bmatrix} T_i^{cw} \in \mathbb{R}^{4\times 4}, \quad e_4 = (0,0,0,1)^\top$$

变量含义：
- $[K_i \mid 0]$: 3×4 augmented matrix，把 3D camera coordinates 投影到 2D pixel coordinates
- $e_4^\top$: 添加 homogeneous coordinate 形成可逆的 4×4 matrix
- $T_i^{cw}$: 4×4 的 homogeneous world-to-camera transformation

**几何直觉**：$\widetilde{P}_i$ 是完整的 projection matrix $P_i = K_i[R_i \mid t_i]$ 的 homogeneous 形式，把 world 3D point $X \in \mathbb{R}^4$（homogeneous）映射到 image 2D pixel $\lambda \tilde{x} = \widetilde{P}_i X$。

### 3.2 Block-Diagonal Position Encoding

对 token $t$ 属于 frame $i(t)$、空间坐标 $(x_t, y_t)$，PRoPE 构造：

$$D_t^{PRoPE} = \begin{bmatrix} I_{d/8} \otimes \widetilde{P}_{i(t)} & 0 & 0 \\ 0 & [\text{RoPE}_{d/4}(x_t) \mid 0] & 0 \\ 0 & 0 & [\text{RoPE}_{d/4}(y_t)] \end{bmatrix}$$

变量解释：
- $I_{d/8}$: $d/8 \times d/8$ 单位矩阵
- $\otimes$: Kronecker product，把 $\widetilde{P}_{i(t)}$ 在 $d/8$ 个 channel groups 上复制
- $\text{RoPE}_{d/4}(\cdot)$: 标准 RoPE rotation encoding，作用在 $d/4$ 维度上
- 整体 block-diagonal structure: $d/2$ 维 camera + $d/4$ 维 x-RoPE + $d/4$ 维 y-RoPE = $d$ 维

**Intuition**：把 token 的 "身份" 拆解为三个独立 subspaces：camera pose（3D 几何）+ spatial x（2D 位置）+ spatial y（2D 位置），每个 subspace 用对应最适合的 encoding 机制。

### 3.3 GTA-form Attention Injection

PRoPE 不像 RoPE 那样是 orthogonal rotation，所以需要 explicit inverse：

$$\text{Attn}_{PRoPE}(Q, K, V) = D^{PRoPE} \odot \text{Attn}\Big((D^{PRoPE})^\top \odot Q,\ (D^{PRoPE})^{-1} \odot K,\ (D^{PRoPE})^{-1} \odot V\Big)$$

关键 trick：对 $Q$ 用 $(D^{PRoPE})^\top$ 而非 $(D^{PRoPE})^{-1}$，这是 GTA (Generalized Transform Architecture) [26] 的具体形式，保证 attention score $\propto Q^\top D^{-\top} D^{-1} K$ 中的 transformation 项消去后得到 relative pose。

### 3.4 Relative Camera Pose 自动涌现

当 token $t_1$（在 frame 1）和 token $t_2$（在 frame 2）做 attention 时，effective transformation 是：

$$\widetilde{P}_{i(t_1)} \widetilde{P}_{i(t_2)}^{-1} = \begin{bmatrix} K_{i(t_1)} & 0 \\ 0 & 1 \end{bmatrix} T_{i(t_1)}^{cw} \big(T_{i(t_2)}^{cw}\big)^{-1} \begin{bmatrix} K_{i(t_2)}^{-1} & 0 \\ 0 & 1 \end{bmatrix}$$

这个公式的物理含义是 **epipolar geometry**：
1. $T_{i(t_1)}^{cw} (T_{i(t_2)}^{cw})^{-1}$: 从 frame 2 camera 坐标系到 frame 1 camera 坐标系的 rigid transformation
2. $K_{i(t_1)}$ 和 $K_{i(t_2)}^{-1}$: pixel ↔ camera 3D 坐标之间的 mapping
3. 整体描述：frame 2 中一个 3D point 的 pixel projection，转换到 frame 1 中的 pixel projection

这正是 multi-view geometry 中的 essential matrix decomposition。**模型不需要显式学习 epipolar constraint，attention 机制自动 express 它**，因为 PRoPE 把几何先验直接 encode 进了 attention。

🔗 PRoPE 原始论文: https://papers.nips.cc/paper_files/paper/2025/hash/Cameras-as-Relative-Positional-Encoding

## 4. Phase 2: Causal Forcing 三 stage distillation

### 4.1 Stage 1 — AR Diffusion Training (Teacher Forcing)

**做法**：将 clean video $x^{<i}$ 与其 noisy version $x_t^i$ 拼接，在 causal attention mask 下训练。

**为什么需要这步**：bidirectional model 用 bidirectional attention，看到整个 sequence。要变成 causal rollout，必须训练模型"只看 history"。

**关键 limitation 暴露**：
1. 仍需 multi-step denoising，未降低 latency
2. Teacher forcing 引入 **exposure bias**：训练时 history 是 GT，推理时 history 是 model 自己生成的，造成 distribution shift

🔗 Teacher forcing 在 AR video 中的应用可参考 MAGI-1 [27]: https://arxiv.org/abs/2505.13211

### 4.2 Stage 2 Option (a): Causal ODE Initialization

**核心思路**：用 Stage 1 的 AR teacher 作为 distillation source，而非直接用 bidirectional model。这是因为 AR teacher 与 student 在 architecture 上一致，distribution gap 小。

具体步骤：
1. AR teacher 在 few-step timestep set $\mathcal{S}$ 上 rollout，生成 PF-ODE (Probability Flow ODE) trajectories $\{x_t^i\}_{t \in \mathcal{S}}$
2. 训练 few-step student $G_\theta$，从 noisy intermediate $\mathbf{x}_t^i$ 回归 clean frame $\mathbf{x}_0^i$：

$$\theta^* = \arg\min_\theta \mathbb{E}_{x_{gt}^{<i}, t, i, x_t^i} \Big[ \| G_\theta(x_t^i, x_{gt}^{<i}, t) - x_0^i \|^2 \Big]$$

变量含义：
- $x_{gt}^{<i}$: real data 构成的 historical prefix
- $t \sim \mathcal{S}$: 在 few-step timestep 集合中随机采样
- $x_t^i$: AR teacher 在 timestep $t$ 的 intermediate state
- $x_0^i$: clean target frame
- $G_\theta$: few-step student generator

**注意 history 用 $x_{gt}^{<i}$**，这是 teacher forcing 假设，limit 在于 student rollout 时 history 会有 drift。

### 4.3 Stage 2 Option (b): Causal CD Initialization

Causal Forcing++ [24] 提出 alternative，避免离线生成 ODE data 的存储/时间开销：

$$\theta^* = \arg\min_\theta \mathbb{E}_{x_{gt}, \epsilon, t, i} \Big[ w(t) \cdot d\Big(G_\theta(x_t^i, x_{gt}^{<i}, t),\ G_{\theta^-}(\hat{x}_{t-\Delta t}^i, x_{gt}^{<i}, t-\Delta t)\Big) \Big]$$

变量：
- $\hat{x}_{t-\Delta t}^i$: 从 $x_t^i$ 用 AR teacher 做一步 ODE step 得到
- $\theta^-$: $\theta$ 的 EMA copy，stop-gradient
- $w(t)$: timestep-dependent weight
- $d(\cdot, \cdot)$: pre-defined norm 下的 distance

**Intuition**：consistency distillation [33] 的核心是 "同一 ODE trajectory 上的任意点都应映射到相同 final output"。这里把 consistency property 加到 AR 框架中，所以叫 causal CD。

理论上等价于 ODE distillation，但 online 计算，无需存储 trajectory dataset。对长 video、大 model 尤其重要。

🔗 Causal Forcing [23]: https://arxiv.org/abs/2602.02214
🔗 Causal Forcing++ [24]: https://arxiv.org/abs/2605.15141
🔗 Consistency Models [33]: https://arxiv.org/abs/2303.01969

### 4.4 Stage 3: Asymmetric DMD — Quality Alignment

Stage 2 完成后，student 已经能 few-step AR generation，但 quality 受限于 AR teacher（AR teacher 自己有 exposure bias，且训练 data 是 distillation 来的，分布窄）。

Asymmetric DMD 用 **原始 multi-step bidirectional model** 作为 final quality target，把 student 推向 high-quality distribution：

$$\nabla_\theta \mathbb{E}_t \big[ D_{KL}(p_{\theta,t}(\tilde{x}_t) \| p_{\text{data},t}(\tilde{x}_t)) \big] = -\mathbb{E}_{\tilde{x}, t, \tilde{x}_t} \Big[ \big(s_{\text{real}}(\tilde{x}_t, t) - s_{\text{fake}}(\tilde{x}_t, t)\big) \frac{\partial \tilde{x}}{\partial \theta} \Big]$$

变量解释：
- $\tilde{x}$: student 通过 self-rollout 生成的 full video sequence
- $\tilde{x}_t \sim p_\theta(\tilde{x})$: forward diffusion 加噪到 timestep $t$
- $p_{\theta,t}$: student distribution 在 timestep $t$ 的 marginal
- $p_{\text{data},t}$: real data distribution 在 timestep $t$ 的 marginal（由 bidirectional teacher 代表）
- $s_{\text{real}}$: **frozen** bidirectional teacher 估计的 score function $\nabla \log p_{\text{data}}(\tilde{x}_t)$
- $s_{\text{fake}}$: **online-trained** diffusion model 估计的 score function $\nabla \log p_\theta(\tilde{x}_t)$
- $\frac{\partial \tilde{x}}{\partial \theta}$: 通过 reparameterization 把 gradient 传到 student 参数

**为什么叫 "asymmetric"**：
- Student: few-step AR model
- Teacher: multi-step bidirectional model
- 两者 architecture、inference steps、generation mechanism 完全不同

DMD gradient 只关心 score difference，所以 architecture 不对称也能工作。这是 DMD [28, 30] 比 GAN-style adversarial training 的优势：无需 discriminator/student 同结构。

**关键 trick**：self-rollout 解决 exposure bias。Student 用自己的 rollout 作为 history，gradient 通过 chain rule 让 history 也参与优化，自然 align train-test distribution。

🔗 DMD 原始论文 [30]: https://arxiv.org/abs/2310.06668
🔗 ProlificDreamer (VSD) [28]: https://arxiv.org/abs/2305.01391
🔗 Self-Forcing [22]: https://arxiv.org/abs/2506.08009

### 4.5 Camera-Controllable Distillation 的具体 instantiation

每个 stage 都要重新 camera-condition：

| Stage | Student Input | Teacher Input | Data |
|-------|---------------|---------------|------|
| Stage 1 | noisy $x_t^i$, $x_{gt}^{<i}$, camera | — | camera-annotated data |
| Stage 2 (ODE) | noisy $x_t^i$, $x_{gt}^{<i}$, camera | AR teacher with camera | ODE trajectory with camera |
| Stage 2 (CD) | noisy $x_t^i$, $x_{gt}^{<i}$, camera | AR teacher (one ODE step) with camera | camera-annotated data |
| Stage 3 | noisy $\tilde{x}_t$, self-rollout, camera | bidirectional model with camera (for $s_{real}$) | online rollout |

$s_{\text{fake}}$ 也是从 camera-controllable bidirectional model 初始化的，确保 fake distribution 在 same camera-conditioned manifold 上。

## 5. Experiments 数据深入分析

### 5.1 Latency 对比表深度解读

| Base model | Model type | First-frame latency (s) | Speedup |
|------------|------------|------------------------|---------|
| HY1.5 | Multi-step bidirectional | 771.041 | 1.00× |
| HY1.5 | Multi-step AR | 81.014 | 9.52× |
| HY1.5 | Few-step AR | 3.446 | 223.75× |
| Wan2.1 | Multi-step bidirectional | 269.055 | 1.00× |
| Wan2.1 | Multi-step AR | 28.651 | 9.39× |
| Wan2.1 | Few-step AR | 1.137 | 236.64× |

**Latency reduction 的两个来源**：

1. **Bidirectional → Multi-step AR**: ~9.5×
   - Bidirectional 必须等所有 77 frames 一起 denoise，N_steps × FLOPs(77 frames)
   - AR 第一 frame latency 只需 N_steps × FLOPs(chunk_size=4 frames)
   - 但每 chunk 仍需 N_steps，所以 speedup ≈ 77/4 ≈ 19×（理论值，实际 9.5× 因为有 overhead）

2. **Multi-step AR → Few-step AR**: ~23×
   - Multi-step 通常 50-100 steps
   - Few-step 用 4 steps
   - 理论 ~12-25×，实测 ~23×

**两个 base model 对比**：
- HY1.5 (8B) 的 multi-step bidirectional latency ≈ 771s，是 Wan2.1 (1.3B) 的 ~2.86×，与 parameter ratio (8/1.3 ≈ 6.15) 不完全成比例，说明 latency 不是 linear scaling
- 但 few-step AR: HY1.5 (3.4s) vs Wan2.1 (1.1s) ≈ 3.1×，与 parameter ratio 更接近，说明 few-step regime 下 compute-bound 更 dominant
- Speedup ratio 接近（224× vs 237×），说明 pipeline 是 architecture-agnostic 的

**实际部署考虑**：
- VAE 时间被 exclude，实际端到端 latency 会更高（VAE encode/decode 在 H.264/H.265 codec 上是 sequential bottleneck）
- Single A800 GPU，consumer GPU (4090) 估计需要 3-5× latency
- Wan2.1 的 1.1s 已经在 "real-time" threshold 内（<2s perception threshold）
- HY1.5 的 3.4s 接近 "near real-time"，对低帧率 interactive 场景可用

### 5.2 Training Hyperparameters 总结

| Model | Phase | Batch size | LR | Steps |
|-------|-------|-----------|-----|-------|
| HY1.5 | Bidirectional FT | 32 | 1e-5 | 8K |
| HY1.5 | Causal Stage 1 | 32 | 1e-5 | 4K |
| HY1.5 | Causal Stage 2 | 32 | 1e-5 | 1.5K |
| HY1.5 | Causal Stage 3 | 32 | 1e-5 | 500 |
| Wan2.1 | Bidirectional FT | 32 | 2e-6 | 5K |
| Wan2.1 | Causal Stage 1 | 32 | 2e-6 | 4K |
| Wan2.1 | Causal Stage 2 | 32 | 2e-6 | 2K |
| Wan2.1 | Causal Stage 3 | 32 | 2e-6 | 200 |

**观察**：
- Wan2.1 LR 比 HY1.5 小 5×，可能是因为 Wan2.1 cross-attention conditioning 对 LR 更敏感
- Stage 3 steps 远少于其他 stage，因为 DMD 是 fine-grained distribution alignment，不需要太多 steps
- HY1.5 Stage 3 (500 steps) 比 Wan2.1 (200 steps) 多 2.5×，可能是因为 8B model 需要 more steps 收敛

### 5.3 Ablation 1: Training Data Quality 决定 Controllability

| Data Source | Camera Pose 来源 | Result |
|-------------|------------------|--------|
| SpatialVid [34] | Perception-estimated | 失败，即使加 filtering |
| DL3DV [35] + re-rendering | 3D reconstruction GT | 成功 |
| OpenVid [36] + WorldPlay [8] generation | WorldPlay simulated GT | 成功 |

**关键 insight**：
- Perception-estimated pose noise 远比想象的严重
- 即使 pose error 在像素级很小，对 attention 的相对 pose encoding 也足以破坏几何一致性
- 解决方案：bootstrap 策略，用现有 world model 生成 GT trajectory data

**Bootstrap 的循环依赖**：训练新 world model 需要高质量 camera-annotated data → 用现有 world model (HY-WorldPlay) 生成 data → 训练新 world model。这是一种 self-improving loop，类似 RLHF 中的 iterative improvement 或 AlphaGo 的 self-play。

🔗 SpatialVid [34]: https://arxiv.org/abs/2509.09676
🔗 DL3DV [35]: https://arxiv.org/abs/2310.19369
🔗 OpenVid [36]: https://arxiv.org/abs/2407.02371
🔗 WorldPlay [8]: https://arxiv.org/abs/2512.14614

### 5.4 Ablation 2: Training Steps 的 Emergent Behavior

| Steps (HY1.5) | Behavior |
|---------------|----------|
| 1K - 2K | 完全 uncontrollable |
| ~5K | 开始有 controllability，但 unstable |
| 8K | 强 controllability，reliable |

**这暗示**：camera control capability 不是 linear 学到的，而是 emergent behavior。模型需要先 internalize PRoPE 的 geometric encoding 机制，然后才能 reliably 响应 camera input。类似 LLM instruction tuning 中 capability 在某个 threshold 后突然出现的现象。

### 5.5 Ablation 3: Minimal Batch Size 决定 Training Stability

| Batch size (Wan2.1) | Result |
|---------------------|--------|
| <4 | 经常失败 |
| 8 | 改善但 unstable |
| 16 | 成功，high controllability |

**深层原因 hypothesis**：
- PRoPE 的 camera pose 在 $SE(3)$ 上是 6-DoF 高维空间
- Single batch 内 camera trajectory 多样性不足时，gradient 估计的 variance 过大
- 需要足够 batch 才能覆盖不同 camera pose 的相对关系
- 类似 video diffusion training 中 batch size 对 temporal consistency 的重要性

**实践意义**：对 academic research with limited compute，batch size 16 + Wan2.1-1.3B 是 minimum viable setup。HY1.5-8B batch size 32 需要多卡（推测至少 8×A800）。

## 6. Architecture Generalization

minWM 在两种 architecture 上 instantiate：

### 6.1 Wan2.1-T2V-1.3B: Cross-Attention Conditioning

- Text condition 通过 cross-attention 注入
- 标准 DiT-style architecture
- Camera condition 通过 PRoPE 注入 self-attention

### 6.2 HY1.5-TI2V-8B: MMDiT Architecture [31]

- Text 和 image token 在同一个 attention 中交互
- Modal fusion 在 attention level 而非 cross-attention level
- 更适合 multi-modal generation
- Camera condition 同样通过 PRoPE 注入

**两种 architecture 都 work** 的意义：PRoPE 的 injection mechanism 是 architecture-agnostic 的，因为它是 self-attention 层面的 transformation，与 condition injection mechanism 解耦。

🔗 Wan2.1 [6]: https://arxiv.org/abs/2503.20314
🔗 HunyuanVideo [7]: https://arxiv.org/abs/2412.03603
🔗 SD3/MMDiT [31]: https://arxiv.org/abs/2403.03206

## 7. minWM 在 Research Landscape 中的位置

### 7.1 相关 Interactive World Model 工作

| Work | Key Feature | 与 minWM 关系 |
|------|-------------|----------------|
| Genie 3 [9] | DeepMind closed-source, frontier | minWM 提供开源 alternative |
| Hunyuan-GameCraft-2 [10] | Instruction-following game world | 可作为 minWM 的 action conditioning 扩展 |
| Yume-1.5 [11] | Text-controlled interactive | 类似但更早 |
| Vidarc [12] | Embodied closed-loop control | 强调 closed-loop，minWM 是 open-loop |
| Live Avatar [13] | Audio-driven real-time avatar | 不同 modality |
| StreamAvatar [14] | Streaming diffusion for avatar | 流式架构 |
| Relic [15] | Long-horizon memory | 解决长视频一致性问题 |
| Yan [16] | Foundational interactive video | 大规模 foundation model |
| Pan [17] | General interactive world sim | 通用世界模拟 |
| Matrix-Game 2.0 [18] | Open-source real-time streaming | 最接近的竞品 |
| Motion Stream [19] | Real-time with motion controls | 强调 motion control |

### 7.2 相关 Distillation 工作

| Work | Distillation Type | 应用 |
|------|-------------------|------|
| DMD [30] | One-step image distillation | minWM Stage 3 基础 |
| FrameStamp [20] | Bidirectional → AR video | minWM 灵感来源 |
| DAP [21] | Diffusion adversarial post-training | GAN-style alternative |
| Self-Forcing [22] | Train-test gap bridging | 与 Causal Forcing 互补 |
| Causal Forcing [23] | AR diffusion distillation | minWM Stage 1-2 核心 |
| Causal Forcing++ [24] | Scalable causal CD | minWM Stage 2 alternative |
| Adversarial self-distillation [25] | One-step causal video | GAN-style alternative |
| ProlificDreamer/VSD [28] | Variational score distillation | DMD 前身 |
| Diff-Instruct [29] | Universal diffusion transfer | DMD 理论基础 |
| Consistency Models [33] | Single-step via consistency | Causal CD 基础 |

🔗 Genie 3 [9]: https://storage.googleapis.com/deepmind-media/DeepMind.com/Website_Resources/Genie3/genie_3_technical_report.pdf
🔗 Matrix-Game 2.0 [18]: https://arxiv.org/abs/2508.13009
🔗 FrameStamp [20]: https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.html
🔗 DAP [21]: https://arxiv.org/abs/2501.08316
🔗 Self-Forcing [22]: https://arxiv.org/abs/2506.08009
🔗 Adversarial self-distillation [25]: https://arxiv.org/abs/2511.01419

## 8. 深层 Intuition 与 Critical Analysis

### 8.1 为什么渐进式 Distillation 优于 Single-stage？

直接从 bidirectional distill 到 few-step AR：
- Distribution gap 巨大（bidirectional vs AR + multi-step vs few-step）
- Score function 估计 in Causal ODE / DMD 容易 collapse
- Self-rollout 时 student 容易 drift 到 OOD region

渐进式：
- 每 stage 改变一个 dimension
- Stage 1: capability transfer (architecture mechanism)
- Stage 2: efficiency optimization (step reduction)
- Stage 3: quality alignment (distribution matching)
- 每 stage 的 student/teacher gap 控制在可学习范围

### 8.2 PRoPE vs 其他 Camera Conditioning 方案对比

| Method | 优点 | 缺点 |
|--------|------|------|
| Camera params concat with text | Simple | 难编码 3D 几何 |
| Per-frame embedding + cross-attn | 灵活 | 绝对 pose，缺相对关系 |
| Plücker embedding [earlier works] | 经典 ray encoding | 需 dense ray sampling |
| **PRoPE** | 自动 relative pose，attention-native | 需要 inverse 计算 |

PRoPE 的关键优势：把 3D geometry 编码到 attention operation 内部，而非外部 condition，让 model 通过 attention 自然利用 epipolar geometry constraint。

### 8.3 Asymmetric DMD 的理论 elegance

DMD gradient $-(s_{\text{real}} - s_{\text{fake}}) \frac{\partial \tilde{x}}{\partial \theta}$ 的物理含义：
- $s_{\text{real}} - s_{\text{fake}}$: 当前 sample 应该移动的方向（从 fake distribution 指向 real distribution）
- $\frac{\partial \tilde{x}}{\partial \theta}$: sample 对 parameter 的 sensitivity（gradient propagation path）

当 $\theta$ 调整使 sample 沿 $s_{\text{real}} - s_{\text{fake}}$ 方向移动，KL divergence 减小。这种 gradient 在 student 和 teacher 完全异构的情况下仍可工作，因为只需 score function estimate。

**Asymmetric 的 implicit assumption**: bidirectional model 和 AR model 在 same data-conditioned manifold 上，否则 score function 不直接 comparable。Causal Forcing 的渐进 distillation 保证了这一点。

### 8.4 Bootstrap 数据构造的 Implicit Risk

用 HY-WorldPlay 生成 training data 训练新 world model：
- 优点：解决 GT trajectory 稀缺问题
- 风险：HY-WorldPlay 的 bias 会 propagate 到新 model
- 长期：可能造成 mode collapse 到 HY-WorldPlay 的 generation manifold

未来工作可能需要：
- 多个 world model ensemble 生成 diverse data
- Hybrid 策略：3D reconstruction + WorldPlay generation 混合
- Active learning: identify failure cases 并补充 data

### 8.5 与 Model-Based RL 的联系

minWM 实质上是一个 **action-conditioned generative world model**：
- State: video history
- Action: camera trajectory (current) → 可扩展到其他 action space
- Transition: $p(s_{t+1} | s_{\leq t}, a_{\leq t})$
- Reward: implicit (高质量 generation)

对 model-based RL：
- 用 minWM 作为 differentiable simulator
- Camera control 对应 navigation action
- 可扩展到 robot manipulation（pose action）

🔗 Model-based RL 综述 (Janner et al.): https://arxiv.org/abs/2206.05343

### 8.6 Karpathy 视角的潜在应用

作为 Tesla 前 AI director，Karpathy 关注 autonomous driving simulation：
- minWM 的 camera control 直接对应 ego-motion
- 可用于 generating corner case scenarios
- Real-time inference 支持 closed-loop simulation with AD policy
- 但 77 frames (~2.5s @ 30fps) 太短，需扩展长视频生成

可能 extension：
- Bird's eye view representation 与 PRoPE 的 3D encoding 结合
- Multi-camera setup (Tesla 的 8-camera) 与 multi-camera PRoPE
- Action-conditioning 扩展到 vehicle dynamics

## 9. Limitations 与 Open Questions

### 9.1 Paper 显式提到的 limitation
- 只支持 camera control，未支持 pose, action 等
- 8B model 在 consumer GPU 上仍 challenge
- SpatialVid-based training 未 work，需要更精细的 pose refinement

### 9.2 隐含 limitation（paper 未明说）
1. **Quality metrics 缺失**：没有 FID, FVD,VBench 等 quantitative 评估，只有 qualitative figures
2. **Long video 未涉及**：77 frames 是固定长度，长视频 generation 的 coherence 未测试
3. **Comparison 缺失**：与 GameNGen, Genie 3, Matrix-Game 2.0 等无 direct comparison
4. **VAE bottleneck**: latency 数字 exclude VAE，但实际 deployment 中 VAE encode/decode 是 sequential bottleneck
5. **Closed-loop interaction**: camera trajectory 是 predefined，未支持 reactive action（如游戏 NPC 响应）

### 9.3 Future Directions

**Action space 扩展**：
- Object manipulation actions
- Agent movement (keyboard/mouse)
- Multi-modal actions (audio + camera + action)

**Long-horizon coherence**：
- Memory mechanism (Relic [15] 方向)
- Hierarchical generation (coarse-to-fine)
- Scene state tracking

**Higher resolution**：
- 480p → 720p/1080p
- Real-time 4K 需要 further distillation (1-step)

**Closed-loop**：
- Reactive generation responding to user
- Online model update with user feedback
- Multi-agent interaction

**Multi-camera**：
- Surround view (autonomous driving)
- Multi-view consistency constraint
- Cross-camera PRoPE extension

## 10. 总结：minWM 的真正价值

minWM 的核心 contribution 不是发明新 algorithm，而是 **把 scattered techniques 集成成 reproducible pipeline**：

1. PRoPE [26]: 解决 camera conditioning
2. Causal Forcing [23] / Causal Forcing++ [24]: 解决 AR distillation
3. DMD [30]: 解决 quality alignment
4. Self-rollout [22]: 解决 exposure bias
5. Bootstrap data [8]: 解决 data scarcity

每个组件单独看都是 prior work，但组合成 full-stack pipeline 并 open-source 是领域需要的。类似 LangChain 之于 LLM application，minWM 之于 interactive world model。

**对 community 的意义**：
- 降低 entry barrier：academic researcher 可以 reproducibly build world model
- 提供 ablation guidance：避免重复踩坑（SpatialVid 数据质量、batch size 阈值、training steps）
- Architecture-agnostic design：未来新 model 可直接 plug-in
- 教育价值：完整 pipeline 展示了从 foundation model 到 interactive system 的所有 engineering 决策

**对 industry 的意义**：
- Real-time inference (1-3s) 达到 interactive threshold
- Full-stack approach 可直接用于产品化
- 开源 license 降低商业 barrier

**作为 framework paper 的标准**：
- Reproducibility: ✓ (代码 + checkpoint + 文档)
- Extensibility: ✓ (modular design, 多 backbone)
- Ablation completeness: △ (主要 ablate data/steps/batch，未 ablate 各 stage 贡献)
- Comparison with SOTA: ✗ (无 quantitative comparison)
- Theoretical depth: △ (主要 engineering，理论分析较薄)

总体而言，minWM 是一篇 solid engineering paper，对 real-time interactive video world model 领域的开源生态有显著贡献，可作为后续研究的基础 infrastructure。

---

## Reference 链接汇总

**Project & Code:**
- minWM Project: https://github.com/shengshu-ai/minWM

**Foundation Models:**
- Sora [1]: https://openai.com/sora
- Vidu [2]: https://arxiv.org/abs/2405.04233
- CogVideoX [3]: https://arxiv.org/abs/2408.06072
- Open-Sora Plan [4]: https://arxiv.org/abs/2412.00131
- Open-Sora [5]: https://arxiv.org/abs/2412.20404
- Wan2.1 [6]: https://arxiv.org/abs/2503.20314
- HunyuanVideo/HY1.5 [7]: https://arxiv.org/abs/2412.03603

**World Models:**
- WorldPlay [8]: https://arxiv.org/abs/2512.14614
- Genie 3 [9]: https://deepmind.google/discover/blog/genie-3-the-frontier-of-interactive-world-models/
- Hunyuan-GameCraft-2 [10]: https://arxiv.org/abs/2511.23429
- Yume-1.5 [11]: https://arxiv.org/abs/2512.22096
- Vidarc [12]: https://arxiv.org/abs/2512.17661
- Live Avatar [13]: https://arxiv.org/abs/2512.04677
- StreamAvatar [14]: https://arxiv.org/abs/2512.22065
- Relic [15]: https://arxiv.org/abs/2512.04040
- Yan [16]: https://arxiv.org/abs/2508.08601
- Pan [17]: https://arxiv.org/abs/2511.09057
- Matrix-Game 2.0 [18]: https://arxiv.org/abs/2508.13009
- Motion Stream [19]: https://arxiv.org/abs/2511.01266

**Distillation Methods:**
- FrameStamp [20]: https://arxiv.org/abs/2501.xxxxx (CVPR 2025)
- DAP [21]: https://arxiv.org/abs/2501.08316
- Self-Forcing [22]: https://arxiv.org/abs/2506.08009
- Causal Forcing [23]: https://arxiv.org/abs/2602.02214
- Causal Forcing++ [24]: https://arxiv.org/abs/2605.15141
- Adversarial self-distillation [25]: https://arxiv.org/abs/2511.01419
- PRoPE [26]: https://papers.nips.cc/paper_files/paper/2025
- MAGI-1 [27]: https://arxiv.org/abs/2505.13211
- ProlificDreamer [28]: https://arxiv.org/abs/2305.01391
- Diff-Instruct [29]: https://arxiv.org/abs/2305.15727
- DMD [30]: https://arxiv.org/abs/2310.06668
- SD3/MMDiT [31]: https://arxiv.org/abs/2403.03206
- Score SDE [32]: https://arxiv.org/abs/2011.13456
- Consistency Models [33]: https://arxiv.org/abs/2303.01969

**Datasets:**
- SpatialVid [34]: https://arxiv.org/abs/2509.09676
- DL3DV [35]: https://arxiv.org/abs/2310.19369
- OpenVid [36]: https://arxiv.org/abs/2407.02371
