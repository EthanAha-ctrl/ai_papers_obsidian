---
source_pdf: A Survey on Efficient Vision-Language-Action Models.pdf
paper_sha256: 03049e93df452004d2b7b544f6a1e757d3f53a574bbe5037859561a9800cc349
processed_at: '2026-07-17T21:38:05-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# A Survey on Efficient Vision-Language-Action Models 技术解析

Karpathy你好！这篇paper是2025年关于Efficient VLAs的第一篇系统综述，覆盖了整个model-training-data lifecycle。让我深入构建你的intuition。

---

## 1. 核心问题：为什么需要Efficient VLAs

### 1.1 VLA的基础架构

VLA = Vision Encoder + LLM Backbone + Action Decoder，三个模块串行：

**公式（1）Vision Encoder:**
$$\mathbf{v} = E_{img}(I; \theta_{img})$$
- $I \in \mathbb{R}^{H \times W \times 3}$: RGB图像
- $\mathbf{v} \in \mathbb{R}^{N_v \times D_v}$: vision tokens序列
- $N_v$: token数量（典型值几百到上千）
- $D_v$: 每个token的embedding维度
- $\theta_{img}$: encoder参数（ViT/SigLIP/DINOv2/CLIP）

**公式（2）LLM Backbone:**
$$\mathbf{h} = LLM(P(\mathbf{v}, \mathbf{l}); \theta_{LLM})$$
- $\mathbf{l} \in \mathbb{R}^{N_l \times D_l}$: 语言tokens
- $P(\cdot)$: projector对齐模态gap
- $\mathbf{h} \in \mathbb{R}^{N_h \times D}$: hidden states

**公式（3）Action Decoder:**
$$\mathbf{a}_{1:T} = D_{act}(\mathbf{h}; \theta_{act})$$
- $T$: action chunk长度
- $D_a$: action维度（end-effector pose + gripper command）
- 关键点：autoregressive模式下需要逐token生成

### 1.2 效率瓶颈的量化

Table 1的实验数据极其关键：

| Model | Params | Latency | Freq |
|-------|--------|---------|------|
| RT-2-PaLI-X | 55B | 330-1000ms | 1-3 Hz |
| OpenVLA | 7B | 166ms | 6 Hz |
| π0 | 3.3B | 73ms | 20/50 Hz |
| GR00T N1 | 2.2B | 63.9ms | 10/50 Hz |

而实时机器人控制通常需要20-50Hz的控制频率。OpenVLA预训练消耗21,500 A100-GPU hours on 64-GPU cluster；π0需要10,000+小时的robot trajectories。这就是Efficient VLA范式兴起的根本原因。

参考：[OpenVLA paper](https://arxiv.org/abs/2406.09246) | [π0 paper](https://arxiv.org/abs/2410.24164)

---

## 2. Efficient Model Design: 三大方向

### 2.1 Efficient Attention

Transformer的self-attention是 $O(n^2)$ 复杂度，n为sequence length。当action horizon长时计算量爆炸。

**SARA-RT**: up-training方法将quadratic transformer转为linear-attention，保留representational fidelity。其核心思想是让attention matrix近似为low-rank形式：
$$\text{Attn}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V \approx \phi(Q)(\phi(K)^T V)$$
其中 $\phi(\cdot)$ 是kernel feature map，避免显式计算 $n \times n$ 矩阵。

**Long-VLA**: phase-aware input masking。在movement phase聚焦static camera tokens，在interaction phase聚焦gripper tokens。这是基于"机器人状态有阶段性"的先验。

**KV-Efficient VLA**: 将历史KV cache压缩成chunked representations：
$$\text{KV}_{compressed} = \text{RNN}(\text{chunk}_1, \text{chunk}_2, ..., \text{chunk}_n)$$
通过lightweight recurrent gating保留salient contexts。

参考：[SARA-RT](https://ieeexplore.ieee.org/document/10610564) | [Long-VLA](https://arxiv.org/abs/2504.09766)

### 2.2 Transformer Alternatives: Mamba

**Robo-Mamba**: 用Mamba的selective state space model替代Transformer。Mamba的核心公式：
$$h_t = A h_{t-1} + B x_t$$
$$y_t = C h_t$$
其中 $A, B, C$ 是input-dependent的参数，整个序列计算是 $O(n)$ 的linear-time。

**FlowRAM**: Mamba + Conditional Flow Matching，在high-precision manipulation中表现突出。

参考：[Mamba](https://arxiv.org/abs/2312.00752) | [RoboMamba](https://arxiv.org/abs/2410.03293)

### 2.3 Efficient Action Decoding: 最核心的方向

autoregressive decoding的latency是VLA部署的致命瓶颈。一个长度为 $n$ 的action chunk需要 $n$ 次串行的forward pass。

#### (a) Parallel Decoding

**OpenVLA-OFT**: 用bidirectional attention mask替代causal mask，single forward pass预测长度 $K$ 的action chunk：
$$\mathbf{a}_{1:K} = \text{Forward}(\text{prompt}, \text{mask}_t, \text{mask}_{t+1}, ..., \text{mask}_{t+K-1})$$
这相当于把action chunk预测当作masked language modeling任务。

**PD-VLA**: 把autoregressive序列重构为nonlinear fixed-point equation：
$$\mathbf{a}^* = f(\mathbf{a}^*)$$
用Jacobi iteration求解，converge步数远小于序列长度 $n$。

**CEED-VLA**: 放松convergence threshold实现early-exit + consistency distillation。

**Spec-VLA**: speculative decoding：
1. Draft model快速生成候选tokens
2. Verifier model并行验证
3. Relaxed acceptance mechanism提高acceptance rate

#### (b) Generative Decoding

**TinyVLA**: 首次用Diffusion Policy作为decoder。Diffusion的reverse process：
$$p_\theta(\mathbf{a}_{t-1} | \mathbf{a}_t) = \mathcal{N}(\mathbf{a}_{t-1}; \mu_\theta(\mathbf{a}_t, t), \Sigma_\theta(\mathbf{a}_t, t))$$
其中 $\mathbf{a}_t$ 是noised action，$t$ 是diffusion timestep。

**HybridVLA**: 关键创新——diffusion + autoregression在一个Transformer scaffold里。DDIM sampling压缩到4步！
$$\mathbf{a}_{t-1} = \mathbf{a}_t - \frac{1}{T}\nabla_{\mathbf{a}_t}\log p(\mathbf{a}_t) + \sigma_t \epsilon$$

**FreqPolicy**: 频域一致性约束。action序列在时域有强correlation，在频域稀疏：
$$\text{DCT}(\mathbf{a}_{1:T}) = \sum_{t=1}^{T} \mathbf{a}_t \cos\left(\frac{\pi}{T}(t + 0.5)k\right)$$

**MinD**: dual-system world model + diffusion policy。world model预测single-step latent（不是full-frame video），diffusion policy基于这个latent生成action。

**Discrete Diffusion VLA**: 把diffusion的progressive refinement和discrete token interface结合，实现"easy-first, hard-later"解码。

参考：[TinyVLA](https://arxiv.org/abs/2409.12514) | [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) | [HybridVLA](https://arxiv.org/abs/2503.10631)

### 2.4 Lightweight Component

| Model | Size | Key Innovation | Performance |
|-------|------|----------------|-------------|
| RoboMamba | 3.7M policy head | 0.1% of total params | 6-DoF pose |
| TinyVLA | <1.4B | lightweight VLM + diffusion decoder | fast inference |
| CLIP-RT | 1B | frozen CLIP as encoder | 24% higher than OpenVLA |
| DiVLA-2B | 2B | Qwen2-VL-2B backbone | 82 Hz on A6000 |
| NORA | 3B | Qwen-2.5-VL-3B + FAST+ | rivals larger VLAs |
| MiniVLA | 1B | Qwen2-0.5B + SigLIP + DINOv2 | 7x fewer params |

**CLIP-RT**很关键：用frozen CLIP作为unified encoder，参数降到1B（OpenVLA的1/7），但平均success rate反而高24%。这暗示7B级别的参数对action generation可能overkill。

### 2.5 Mixture-of-Experts (MoE)

MoE的核心：只激活参数的一部分。
$$\text{MoE}(x) = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)$$
其中 $g_i(x)$ 是gating function，$E_i$ 是第 $i$ 个expert。

**GeRM**: MoE + Conservative Q-Learning (CQL)，quadruped RL。CQL的核心loss：
$$\mathcal{L}_{CQL} = \alpha \cdot \mathbb{E}_{a \sim \mu}[\log \sum_a \exp(Q(s,a))] - Q(s,a_{data})$$
惩罚OOD action的Q值。

**FedVLA**: Dual Gating MoE (DGMoE)，bidirectional affinities。

**TAVP**: Task-Aware MoE，condition-based expert activation。

### 2.6 Hierarchical Systems: Dual-Process Theory

这是非常Kahneman的思路——System 1 (fast, intuitive) + System 2 (slow, deliberative)：

```
┌─────────────────────────────────────────┐
│ System 2 (slow, VLM)                    │
│ - Low frequency (e.g., 1-5 Hz)          │
│ - Semantic reasoning                    │
│ - Task planning                         │
└────────────────┬────────────────────────┘
                 │ latent / sketch / plan
                 ▼
┌─────────────────────────────────────────┐
│ System 1 (fast, Policy)                 │
│ - High frequency (e.g., 20-50 Hz)      │
│ - Reactive control                      │
│ - Trajectory execution                  │
└─────────────────────────────────────────┘
```

**RoboDual**: OpenVLA (high-level planner) + lightweight Diffusion Transformer (specialist)。latency-aware training解决asynchronous的时间错位问题。

**HAMSTER**: high-level VLM生成2D trajectory sketch → low-level 3D control policy。

**FiS**: parameter sharing，把System 1 executor嵌入System 2 VLM中。

**Fast ECoT**: Embodied Chain-of-Thought (ECoT)的thought reuse。ECoT原本需要每步都reasoning，Fast ECoT缓存high-level reasoning tokens跨多个timestep复用。

参考：[HiRT](https://arxiv.org/abs/2410.17488) | [FiS](https://arxiv.org/abs/2506.01953) | [Fast ECoT](https://arxiv.org/abs/2506.07639)

---

## 3. Model Compression: 三个关键技术

### 3.1 Layer Pruning

motivation: LLM相邻层interlayer cosine similarity很高，存在大量redundancy。

**Training-free:**
- **DeeR-VLA**: dynamic early-exit。当action prediction consistency达到阈值时跳出后续层
- **SmolVLA**: naive strategy，跳过 $N = 2\bar{/}L$ 的层
- **RLRC**: Taylor importance criteria，90% sparsity。Taylor importance估计：
$$I(l) = \mathbb{E}\left[\left|\frac{\partial \mathcal{L}}{\partial W_l} \odot W_l\right|\right]$$
- **FLOWER**: Florence-2去掉整个decoder；SmolFlow2-Video去掉最后30%层

**Training-based:**
- **MoLe-VLA**: 把LLM层视为不同experts，用Spatial-Temporal Aware Router (STAR)动态选择
- **LightDP**: SVD-based importance + Gumbel-Softmax
$$p_i = \frac{\exp((\log \pi_i + g_i)/\tau)}{\sum_j \exp((\log \pi_j + g_j)/\tau)}$$
其中 $g_i \sim \text{Gumbel}(0,1)$，$\tau$ 是温度。

### 3.2 Quantization

**OpenVLA**: 4-bit post-training quantization，GPU memory减半。

**QAIL**: Quantization-Robust Behavior Cloning (QBC) loss：
$$\mathcal{L}_{QBC} = \text{KL}(q_\theta(\mathbf{a}|\mathbf{o}) || q_{\theta}^{FP}(\mathbf{a}|\mathbf{o}))$$
对齐quantized policy和full-precision policy的action分布。

**SQIL**: saliency-aware quantization，2.5× speedup。

**BitVLA**: 1-bit ternary quantization $\{-1, 0, +1\}$！3.36× memory compression。
$$W_{ternary} \in \{-1, 0, +1\}^{m \times n}$$
这极端压缩但用distillation保持性能。

**FAST**: DCT-based action quantization。先对normalized action序列做DCT：
$$X_k = \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi}{N}(n+0.5)k\right)$$
然后BPE压缩成compact tokens。

**SQAP-VLA**: quantization + token pruning联合。Hadamard transform预处理weights和activations：
$$W' = H W H^T, \quad A' = H A$$
使attention map的outliers更易处理。

参考：[BitVLA](https://arxiv.org/abs/2506.07530) | [FAST](https://arxiv.org/abs/2501.09747) | [SQAP-VLA](https://arxiv.org/abs/2509.09090)

### 3.3 Token Optimization

#### Token Compression

**SmolVLA**: pixel shuffle把vision tokens压缩到64/frame。pixel shuffle本质是空间维度的reshape：
$$\mathbb{R}^{H \times W \times C} \to \mathbb{R}^{H/s \times W/s \times (s^2 C)}$$

**CogVLA**: FiLM (Feature-wise Linear Modulation) routing：
$$\text{FiLM}(x) = \gamma(c) \odot x + \beta(c)$$
$\gamma, \beta$ 是condition $c$ 生成的scaling和shifting。

**Oat-VLA**: object-agent-centric tokenization，把patch tokens聚合成object-centric和agent-centric tokens。

**FAST**: DCT + BPE对action tokens。5× pre-training time reduction。

**VOTE**: 极端压缩，单个 `<ACT>` token表示整个action trajectory，用MLP head解码。

#### Token Pruning

**FlashVLA**: Information Contribution Score (ICS)：
$$\text{ICS}(v_i) = \text{attention contribution of } v_i \text{ to action tokens}$$
基于FastV的思想，training-free，Flash Attention compatible。

**EfficientVLA**: Task-Relevance + Diversity-Driven pruning。注意它指出VLA-Cache受LLM memory bottleneck限制。

**SP-VLA**: Dual-Aware Token Pruning，根据end-effector velocity动态调整pruning ratio：
$$r_t = f(\|v_{ee,t}\|)$$
速度大时保留更多tokens。

**SpecPrune-VLA**: self-speculative pruning，用temporal continuity的prior推理来inform current token curation。

**ADP**: action-aware + text-driven pruning。计算vision tokens和task instruction的cross-modal attention，保留Top-K相关tokens。

#### Token Caching

**VLA-Cache**: 识别静态tokens（inter-frame variance小），用KV-cache复用。

**HybridVLA**: 把KV-caching从autoregressive扩展到diffusion的iterative denoising。在diffusion步骤间缓存conditional token KV。

**Fast ECoT**: ECoT reasoning chain的复用，跨timesteps复用推理过程。

**CronusVLA**: FIFO queue存储compact motion features，解耦单帧perception和多帧inference。

**AMS**: GPU-resident Context Pool，hardware-aware caching beyond KV-cache。

参考：[FlashVLA](https://arxiv.org/abs/2505.21200) | [EfficientVLA](https://arxiv.org/abs/2506.10100) | [VLA-Cache](https://arxiv.org/abs/2502.02175)

---

## 4. Efficient Training: Pre-training + Post-training

### 4.1 Efficient Pre-training

#### Data-Efficient Pre-training

**关键洞察：** VLA最大的数据瓶颈是robot action labels。如果用unlabeled video就能pretrain，将极大降低数据成本。

**LAPA (Latent Action Pretraining from Videos)**: 从unlabeled video中学习discrete latent action space。用VQ-VAE：
$$z = \arg\min_k \|x - e_k\|, \quad e_k \in \text{codebook}$$
$$\hat{x} = D(z)$$
将video frame之间的"动作"压缩成latent code。

**Bu et al.**: 改进LAPA，指出LAPA的raw pixel reconstruction会编码task-irrelevant dynamics（camera shake, background motion）。提出task-centric latent action learning，两阶段VQ-VAE解耦。

**EgoVLA**: 用MANO参数建立shared action space，把human egocentric video转成robot action。
$$\mathbf{a}_{robot} = \text{Retarget}(\mathbf{a}_{MANO})$$

**Being-H0**: Physical Instruction Tuning，part-level motion tokenization on UniHand dataset。

**RynnVLA-001**: 三阶段generative pretraining (I2V prediction of future frames)。

**Wang et al.**: World Model objective，先从无action video学dynamics，再fine-tune加action tokens。

#### Mixed Data Co-training

**GeRM**: Conservative Q-Learning (CQL)，robust to suboptimal data。

#### Efficient Action Representation

**Action Space Compression:**
- LAPA, Bu et al.: VQ-VAE
- RynnVLA-001: ActionVAE (standard VAE)：
$$\mathcal{L}_{VAE} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) || p(z))$$
- LAWM: DreamerV3 world model

**FAST的关键创新**: DCT + BPE直接压缩action序列，5× pre-training time reduction。

参考：[LAPA](https://arxiv.org/abs/2410.11758) | [Bu et al. UniVLA](https://arxiv.org/abs/2502.14420) | [EgoVLA](https://arxiv.org/abs/2507.12440)

### 4.2 Efficient Post-Training

#### Supervised Fine-tuning

**OpenVLA的系统比较**：5种策略的trade-off
1. Full fine-tuning: 最佳性能但compute大
2. Last-layer-only: 最简单但效果差
3. Frozen vision: 保留视觉特征
4. Sandwich fine-tuning: 中间层
5. **LoRA**: 最佳performance-compute trade-off

LoRA的核心：
$$W = W_0 + \Delta W = W_0 + BA$$
其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$。

**OpenVLA-OFT**: 集成parallel decoding + action chunking + continuous action + L1-regression loss。

**Atomic Skill Library**: 三轮驱动构建可复用atomic skills repository，支持compositional generalization。

**MoManipVLA**: bi-level trajectory optimization，仅用50个real-world samples迁移到mobile manipulation。

**OpenHelix**: 训练单个 `<ACT>` token embedding，freeze所有MLLM参数。极其cost-effective。

**ControlVLA**: ControlNet-style adaptation：
$$y = F(x; \theta) + Z(\text{prompt}; \theta_z)$$
zero-initialized projection，10-20个demonstration samples就能fine-tune。

**RICL**: In-Context Learning for VLA，模仿RAG：
$$\text{Prompt} = [\text{demonstrations}] + [\text{query}]$$
预测action from contextual cues。

**ATE**: reverse KL divergence在latent space对齐：
$$\mathcal{L}_{RKL} = \text{KL}(p_\theta || q)$$
用energy model gradients引导diffusion/flow-matching VLA的sampling。

#### RL-Based Methods

**Online RL:**

**RIPT-VLA**: PPO + rejection sampling + sparse binary rewards。15次迭代从4% (SFT)到97% success rate，仅需1个demonstration！

**VLA-RL**: PPO + VLM-derived dense rewards，trajectories as multi-turn conversations。在4个LIBERO task suites上比OpenVLA (SFT)提升4.5%。

**SimpleVLA-RL**: GRPO (Group Relative Policy Optimization)：
$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[\frac{A_i}{\sigma_A} \log \pi_\theta(a_i|s_i)\right] - \beta \text{KL}(\pi_\theta || \pi_{ref})$$
其中 $A_i$ 是group-relative advantage。从17.3%到91.7%，仅需1 trajectory per task。

**RPD**: MSE-guided PPO distillation to compact policies。

**Dual-Actor Fine-Tuning**: "Talk-and-Tweak" - human-in-the-loop，latent tweaks from language-mapped corrections。101分钟内达到100% success。

**World-Env**: video-based virtual environment，VLM-guided rewards。从5个demonstration达到79.6% LIBERO success，无real-world cost。

**Offline RL:**

**ConRFT, CO-RFT**: 用Cal-QL解决offline RL的value overestimation。Cal-QL的核心：
$$\mathcal{L}_{CalQL} = \mathcal{L}_{CQL} + \lambda \cdot \mathcal{L}_{calibration}$$
惩罚OOD actions的Q值，compensate in-dataset actions。ConRFT从20-30 demos初始化，96.3% real-world success，144% baseline gains。

**ARFM**: flow-matching loss + adaptive scaling factor，balance RL advantage preservation和gradient variance：
$$\mathcal{L}_{ARFM} = \alpha_{adapt} \cdot \mathcal{L}_{flow} + \mathcal{L}_{RL}$$

参考：[ConRFT](https://arxiv.org/abs/2502.05450) | [SimpleVLA-RL](https://arxiv.org/abs/2509.09674) | [RIPT-VLA](https://arxiv.org/abs/2505.02500)

---

## 5. Efficient Data Collection: 5大策略

### 5.1 Human-in-the-Loop

**CLIP-RT**: natural language interface，user用LLM对话→翻译成end-effector actions。消除专家知识需求。

**GCENT**: human as guardian，只在failure step介入。Interactive rewind-and-correction mechanism。实现one-operator-multiple-robots。

### 5.2 Simulation Data Collection

| Dataset | Scale | Method |
|---------|--------|--------|
| SynGrasp-1B | 1B frames | MuJoCo + Isaac Sim |
| QUARD-Auto | - | Isaac Gym parallel |
| ManiSkill (cVLA) | - | analytical grasp models |
| RoboTwin 2.0 | bimanual | code-gen agent + dual feedback |

**ReBot**: real-to-sim-to-real pipeline。在simulation中重放real robot trajectories，不同objects，然后inpainting到real background。

**R2R2R**: smartphone scan + single human manipulation video → 4D Differentiable Part Modeling生成大规模数据。

**RealMirror**: WebXR + motion-control pipeline，simulation控制real robot，减少latency。

### 5.3 Internet-Scale Data

**EgoVLA**: 关键insight——把human当作一种robot。建立Ego-Centric Human Manipulation Dataset。

**RynnVLA**: 自动video curation pipeline。用pose estimation识别egocentric视角（无facial landmarks + 有hand keypoints）。

**EgoScaler**: 从egocentric video提取6-DoF object trajectories，无需manual annotation。

**MimicDreamer**: 关键创新——用video diffusion model生成photorealistic robot demos。流程：
1. Canonicalize egocentric video (stabilization + inpainting)
2. Constrained IK mapping human wrist → robot joints
3. Video diffusion synthesis符合robot embodiment constraints

**HumanoidVLA**: 第三人称human-motion videos。decompose body poses into part-specific tokens + temporal/spatial perturbations。

### 5.4 Self-Exploration

**AnyPos**: ATARA framework，RL-driven uniform coverage of end-effector workspace。回答"robot能做什么"再"应该做什么"。

**SimpleVLA-RL**: generate-evaluate-optimize cycle：
1. Generate diverse trajectories via stochastic sampling
2. Filter with binary success rewards
3. 保留成功trajectory作为training data
4. 同时优化policy

**Yang**: diffusion model作为policy architecture，捕获multi-modal distribution，生成比人类demonstration更高质量的trajectory。

**World-Env**: 在simulator内controlled stochasticity explore。

**VLA-RFT**: 学习data-driven world model from offline data，消除对pre-built simulator的依赖：
$$p(s_{t+1}|s_t, a_t) \approx \text{learned world model}$$
然后在这个learned world model里做massively parallel rollouts。

### 5.5 Data Augmentation

**LLaRA**: reformat BC datasets into conversational instruction-response pairs。

**InstructVLA**: GPT-4o生成scene captions, QA pairs, command rewrites，缓解catastrophic forgetting。

**ReconVLA**: Grounding DINO自动segment "gaze regions"，构建visual reconstruction pre-training data。

**CLIP-RT**: Stochastic Trajectory Augmentation (STA)，把robot stochastically推到novel states。

**RoboChemist**: 注入failure scenarios + retry attempts，提升self-correction能力。

**ERMV**: Epipolar Motion-Aware Attention，propagate frame edits across multi-view timesteps一致。

参考：[GraspVLA](https://arxiv.org/abs/2505.03233) | [RoboTwin 2.0](https://arxiv.org/abs/2506.18088) | [MimicDreamer](https://arxiv.org/abs/2509.22199)

---

## 6. 核心Intuition构建

### 6.1 VLA Efficiency的核心张力

```
        Expressivity ←────────────────→ Efficiency
            ↑                              ↑
     Long-horizon reasoning         Edge deployment
     Cross-embodiment generalization  Real-time control
            ↑                              ↑
     Large VLM backbone            Small/sparse architecture
```

### 6.2 三大设计哲学

**哲学1: Modular Decoupling**
- Hierarchical systems: System 2 (slow reasoning) + System 1 (fast control)
- MoE: sparse activation
- Multi-stage training: 解耦modality alignment, reasoning, action

**哲学2: Compression without Information Loss**
- Layer pruning (inter-layer redundancy)
- Token pruning (task-irrelevant tokens)
- Quantization (numerical precision redundancy)
- Action compression (latent space / spectral domain)

**哲学3: Generative instead of Discriminative**
- Diffusion/Flow Matching decoder替代autoregressive
- Parallel decoding (Jacobi, speculative)
- World model生成而不是记忆

### 6.3 关键Trade-off的理解

**Action chunking的trade-off:**
- 长chunk: 减少inference次数，但每步chunk可能stale
- 短chunk: 实时性高，但latency累积

**Latent action的trade-off:**
- 优势: 大幅压缩action space，加速pretraining
- 劣势: VQ-VAE的reconstruction error会传播到policy

**Hierarchical的trade-off:**
- 优势: 解耦reasoning和control频率
- 劣势: System 1和System 2的spatiotemporal coherence难保证

### 6.4 公式直觉

**Diffusion Policy的核心:**
$$\mathbf{a}_0 \sim p_\theta(\mathbf{a}_0) = \int p_\theta(\mathbf{a}_{0:T}) d\mathbf{a}_{1:T}$$
其中 $\mathbf{a}_T \sim \mathcal{N}(0, I)$，逐步denoise得到 $\mathbf{a}_0$。相比autoregressive，diffusion天然支持multi-modal action distribution。

**Flow Matching的核心:**
$$\frac{d\mathbf{a}_t}{dt} = v_\theta(\mathbf{a}_t, t)$$
学一个vector field $v_\theta$ 把noise分布transport到action分布。比diffusion更efficient，可以直接学deterministic ODE。

### 6.5 实际部署考虑

| 设备类型 | Constraints | 推荐方向 |
|---------|-------------|----------|
| Industrial robot | Power充足，要求precision | Hierarchical + RL post-training |
| Consumer robot | Edge GPU, 隐私 | 1B-level lightweight + quantization |
| Autonomous vehicle | 实时性极强, safety | Token pruning + parallel decoding |
| Medical robot | Precision, 隐私 | Object-centric + few-shot adaptation |

---

## 7. Future Directions的核心猜想

1. **Adaptive Token Pruning + Context-aware Routing**: dynamically modulate computational paths
2. **Hardware-Software Co-design**: architectural optimization of computing platforms
3. **Federated VLA + Differential Privacy**: distributed robotic agents lifelong learning
4. **Physics-informed Pre-training**: enforce kinematic consistency
5. **Self-sustaining Generative Ecosystems**: diffusion-guided synthesis from minimal seeds
6. **Multi-agent Curiosity-driven Exploration**: shared virtual worlds

---

## 8. 关键References

**Foundational VLAs:**
- [RT-2](https://arxiv.org/abs/2307.15818)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [π0](https://arxiv.org/abs/2410.24164)
- [GR00T N1](https://arxiv.org/abs/2503.14734)

**Efficient Architecture:**
- [RoboMamba](https://arxiv.org/abs/2410.03293)
- [TinyVLA](https://arxiv.org/abs/2409.12514)
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)
- [HybridVLA](https://arxiv.org/abs/2503.10631)
- [SmolVLA](https://arxiv.org/abs/2506.01844)
- [FAST](https://arxiv.org/abs/2501.09747)

**Model Compression:**
- [DeeR-VLA](https://arxiv.org/abs/2411.17465)
- [EfficientVLA](https://arxiv.org/abs/2506.10100)
- [BitVLA](https://arxiv.org/abs/2506.07530)
- [SQAP-VLA](https://arxiv.org/abs/2509.09090)
- [FlashVLA](https://arxiv.org/abs/2505.21200)

**Efficient Training:**
- [LAPA](https://arxiv.org/abs/2410.11758)
- [Bu et al. UniVLA](https://arxiv.org/abs/2502.14420)
- [EgoVLA](https://arxiv.org/abs/2507.12440)
- [SimpleVLA-RL](https://arxiv.org/abs/2509.09674)
- [ConRFT](https://arxiv.org/abs/2502.05450)
- [RIPT-VLA](https://arxiv.org/abs/2505.02500)

**Data Collection:**
- [GraspVLA](https://arxiv.org/abs/2505.03233)
- [RoboTwin 2.0](https://arxiv.org/abs/2506.18088)
- [MimicDreamer](https://arxiv.org/abs/2509.22199)
- [R2R2R](https://arxiv.org/abs/2505.09601)

**Survey GitHub:**
- [Efficient VLA Survey Project Page](https://evla-survey.github.io/)

---

## 总结

这篇survey的intuition是：VLA的efficiency不是一个单点优化，而是一个**model-training-data lifecycle的system-level co-design**问题。关键insight：

1. **Inference latency的瓶颈在autoregressive decoding** → 用parallel/generative decoding突破
2. **Training cost的瓶颈在large-scale pretraining** → 用latent action + self-supervised video pretraining
3. **Data collection的瓶颈在human teleoperation** → 用simulation + internet video + self-exploration

三者环环相扣：efficient model让efficient training有意义，efficient data让efficient model能scale。未来的突破口在adaptive, embodiment-agnostic architectures + federated continual learning + self-sustaining generative ecosystems。

Karpathy，作为深度学习领域的pioneer，你应该能看到这些efficiency techniques本质上都是把scaling-centric的思路换成information-centric——同样的"intelligence"用更少bits和FLOPs表达。Eureka Labs的教育场景中，这种efficient VLA正是让embodied AI普及到普通消费者的关键。
