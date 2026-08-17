---
source_pdf: Research on World Models Is Not Merely Injecting.pdf
paper_sha256: 6c65785d745bbca9dbcdfc500798494ff1391c647936a3f55b62654bc6f6afd1
processed_at: '2026-08-11T22:53:36-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 paper

## 这 paper 在吐槽什么

community 现在有个坏习惯: 拿个 LLM / diffusion model, 喂一堆带"world knowledge"的数据, fine-tune 一下某个 downstream task (image editing、video gen、robot grasping), 然后 paper 标题写 "World Model for XXX"。这帮作者(北大 Wentao Zhang 组)站出来说: **stop, 你们这不叫 world model, 这叫 task-specific fine-tune with extra data**。

真正的 world model 应该长什么样? paper 给了一个 5 件套的 blueprint: **Interaction + Reasoning + Memory + Environment + Multimodal Generation**, 五个 module 闭环耦合, 才算数。

简单类比: 你想造一个"懂物理世界的 agent", 结果你只是训练了一个"能在厨房切菜的机械臂 + GPT-4 外壳", 这跟 world model 没半毛钱关系。Sora 生成漂亮视频, Sora 也只是 next-frame pixel predictor, 它没有 action-conditional 的 transition, 你不能问它"如果我现在把球往左推, 下一秒会怎样"。

参考: https://worldmodels.github.io/ (Ha & Schmidhuber 原始 World Models, 2018)

---

## 三类被点名的"伪 world model"

### A. LLM/VLM reasoning 派

OpenAI o3、Qwen2.5-VL 这类, 走 chain-of-thought, 用 token 序列模拟推理。数学上还是 next-token:

$$P_\theta(y_t \mid y_{<t}, x) = \text{softmax}(W h_t)$$

$y_t$ 是第 $t$ 步 token, $h_t$ 是 hidden state, $W$ 是 unembedding matrix。model 把"推理"surface 到 token 空间, 所以它本质上是在 **自然语言字符串空间** 里做规划。

问题: 物理世界是连续的、sub-symbolic 的。你问 GPT-4V "这张图里有几根手指", 给它一张六指图, 它大概率说"五根", 因为训练分布里"五根"是 strong prior。这就是 paper Fig.3a 的 failure case。

直觉: LLM/VLM 的 reasoning 是 **discrete symbolic overlay on top of statistical pattern matching**, 它没有 grounded 到真实物理 state。

参考: https://openai.com/index/openai-o3/

### B. Diffusion generation 派

Sora、Wan 2.5、Veo 3、MatrixGame, 包括 EditWorld 这种 image editing。它们走 flow matching / DDPM:

$$dx_t = v_\theta(x_t, t) dt$$

$x_t$ 是 flow 上的 state (下标 $t \in [0,1]$ 是 flow time, 和真实时间无关), $v_\theta$ 是学习的 vector field。训练目标:

$$\mathcal{L} = \mathbb{E}\left\| v_\theta(x_t, t) - u_t \right\|^2$$

$u_t$ 是 target vector field (linear path 或 OT path)。这玩意儿在 image / video 上质量炸裂, 但 paper 说它本质是 **3D world → 2D rendering 的拟合**, 没有 internal 3D state。

Fig.3c 那个 navigation 视频案例特别直观: agent 镜头左转再右转回原位, 场景里原本的椅子消失了。为什么? 因为 generator 只看 recent frames 做 next-frame prediction, 没有一个 persistent memory module 记住"5 步前那里有椅子"。

正确的数学应该是:
$$\hat{I}_{t+1} = G_\theta(I_{1:t}, a_{1:t}, W_t)$$
$$W_{t+1} = \text{Update}(W_t, I_t, a_t)$$

$W_t$ 是显式 world state memory (BEV occupancy grid、3DGS、scene graph 都行)。现在 Sora 类模型把 $W$ 这一项丢了。

参考: https://openai.com/sora ; https://arxiv.org/abs/2210.02727 (Flow Matching)

### C. VLA / Embodied 派

π0、Cosmos、GAIA-1/2、Agibot World。这些是 LLM/VLM + action token + 机械臂/车。形式上是 POMDP:

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, T, O, R)$$

$T(s'|s,a)$ 是 transition (paper 想要 world model 学的核心), $O(o|s)$ 是 observation emission, $R$ 是 reward。

VLA 做法: 把 action 也 tokenize, 然后做 next-token prediction, 数据是 teleop 的 $(s, a)$ pairs。paper 说这其实只是 **task-specific skill 套个 LLM 外壳**。Fig.4c 那个 robot 模仿人动作结果把人弄伤的例子特别扎心 — action 没有 grounded 在 world model 对物理后果的预测上, 只是 imitate 了个 pattern。

参考: https://www.physical.intelligence/blog/pi0 ; https://arxiv.org/abs/2501.03575 (Cosmos)

---

## Paper 提的 Unified Framework, 5 个 module 讲人话

### 1. Interaction — 统一 I/O 口

原始 Ha & Schmidhuber 2018 的 world model:
$$z_t = V(o_t), \quad a_t = C(z_t, h_t), \quad h_{t+1} = M(h_t, z_t)$$

$V$ 是 vision encoder, $z_t$ 是 latent observation, $h_t$ 是 RNN memory, $C$ 是 controller。Interaction 只有 vision 这一条单向通道。

paper 说要扩成 **双向多模态口**:
- 输入: text + image + video + audio + 3D point cloud + mesh, 统一 encode 成 world state $s_t$
- 输出: 既给 high-level language instruction ("把红色方块放蓝色上面"), 又给 low-level motor command (关节角度 $\theta_i$、力矩 $\tau_i$、车辆 steer/accel)

直觉: 这口子要 normative, 社区约定 latent 的 schema 和 token format, 不然每个 lab 自己玩自己的, 没法 benchmark。类似 NVIDIA Cosmos 的 tokenizer 设计, 或者 π0 把 action 也 tokenize 进 flow matching 框架。

### 2. Reasoning — Explicit + Latent 混合

**Explicit Reasoning**: 多模态观测 → 文字描述 → LLM 做 reasoning chain。优点 transparent, 缺点 sub-symbolic 信息丢失 (3D 位姿、摩擦系数这些连续量塞进 token 就糊了)。

**Latent Reasoning**: 直接在 latent space 做 forward inference:
$$\hat{s}_{t+1} = g_\theta(s_t, a_t, \text{context})$$

$s_t \in \mathbb{R}^d$ 是 latent world state, $g_\theta$ 是 transition operator。这其实就是 LeCun 的 **JEPA** (Joint Embedding Predictive Architecture):

$$\mathcal{L}_{\text{JEPA}} = \left\| \text{sg}(s_{t+1}) - \text{Pred}(s_t, a_t) \right\|^2$$

$\text{sg}$ 是 stop-gradient 防 representation collapse。paper 没明说, 但它的 "Latent Reasoning" 几乎就是 JEPA。

直觉: reasoning 应该是 hybrid — high-level planning 用 language (transparent, 可解释), 物理细节预测用 latent (精确, 连续)。这方向最近的 Coconut (Meta continuous CoT)、Quiet-STaR、Pause Tokens 都在试探。

参考: https://arxiv.org/abs/2412.06769 (Coconut); https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/

### 3. Memory — 从 implicit 到 explicit structured

演化路径:
- LSTM (1997): $h_t = f(h_{t-1}, x_t)$, 隐式 state
- xLSTM (2024): exponential gating
- Long-context Transformer (Longformer, FlashAttention-2): explicit 大窗口
- MemFlow (2025): flowing adaptive memory

paper 要的是 **structured + dynamic**:
1. 多模态多源数据 fuse + associate, 形成可 query 的 internal knowledge
2. Key info extraction + compression (active filter)
3. 动态 update, merge / purge redundant

形式化:
$$M_t = \text{Update}(M_{t-1}, \text{Write}(s_t, a_t, r_t, s_{t+1}))$$
$$\text{context}_t = \text{Read}(M_t, \text{query}_t)$$

$M_t \in \mathbb{R}^{N \times d}$ 是 memory matrix, $N$ 是 slot 数。这接近 Weston 的 Memory Networks 或 Graves 的 DNC。最近比较热的工作是 Google 的 **Titans** (neural memory with attention), 把 memory 当可学习、可写、可遗忘的 module, 而不仅是 KV cache。

参考: https://arxiv.org/abs/2501.00663 (Titans)

### 4. Environment — Generative simulator, 不只是 renderer

paper 这部分最关键。传统 sim (AI2-THOR、MetaDrive) 是手动建模 rigid body, sim-to-real gap 大。真硬件数据采集成本高。paper 论点: environment 应该 **generative + extensible**:

$$p_\phi(s_{t+1} \mid s_t, a_t, w)$$

$w$ 是 procedural seed / latent scene code。一边是 neural scene generator (Hunyuan3D、MIDI、FlashWorld) 产生 high-fidelity visual + geometry, 一边是 symbolic physics engine (Bullet、MuJoCo、PhysX) 给物体贴物理属性。两者通过 shared representation 耦合。

Genesis simulator 是这个思路的工业实现: 用 LLM 写 simulation code, 把 physics engine 当 differentiable layer。NVIDIA Cosmos 2.0 也在往这方向走。

参考: https://genesis-sim.github.io/ ; https://www.nvidia.com/en-us/ai/cosmos/

### 5. Multimodal Generation — 闭环 feedback

generation 不只是输出 module, 要 **闭环**:
- 生成 video / image / 3D 几何 → 给人类 feedback, 也给 reasoning module 当 **model-based foresight**
- 生成 data → self-augmentation

数学上就是 model-based RL 的 planning:
$$\hat{s}_{t+1:t+H} = \prod_{k=0}^{H-1} g_\theta(s_{t+k}, a_{t+k})$$
$$\pi^* = \arg\max_\pi \mathbb{E}_{\hat{s}} \sum_{k=0}^{H} \gamma^k R(\hat{s}_{t+k}, \pi(\hat{s}_{t+k}))$$

$H$ 是 planning horizon, $g_\theta$ 是 learned transition, $\pi$ 是 policy, $\gamma \in [0,1)$ 是 discount factor。这是 Dreamer / TD-MPC 经典做法, paper 把它扩展到 multimodal 输出 (不光预测 latent, 还要 render 成 video/3D)。

参考: https://arxiv.org/abs/2304.10573 (TD-MPC2); https://arxiv.org/abs/2301.04104 (DreamerV3)

---

## Failure cases 的人话版

| Fig | 失效场景 | 根因 | 数学诊断 |
|---|---|---|---|
| 3a | VLM 数错六指图 | LAION 训练 prior 主导 | $P(\text{five}\|x) > P(\text{six}\|x)$ even when $x$ shows six |
| 3b | Image editing 光影不一致 | diffusion 学 noise prediction, 没建模 lighting physics | $\epsilon_\theta(x_t,t,c)$ 只学 noise, $c$ 是 instruction+原图 |
| 3c | Navigation 物体消失 | 没 persistent memory module | $\hat{I}_{t+1} = G(I_{1:t}, a_{1:t})$ 缺 $W_t$ 项 |
| 3d | 高速动态视频物理崩坏 | pixel-level 拟合, 没学 Newton's law latent | $G_\theta$ 只 fit 视觉 pattern |
| 3e | 3D scene 碎片化 | 3DGS 只有光学属性 (μ, Σ, c, α), 没物理属性 (m, μ_f, e) | $G(x) = \exp(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu))$, 缺 mass/friction |
| 4a | Embodied 只会简单 pick&place | task-specific skill, 没 long-horizon planning | policy horizon $H$ 太短 |
| 4b | Autonomous driving 直路失效 | 没 grounded 物理 + 长程 context | transition $T$ 学得太窄 |
| 4c | Robot 模仿动作伤人 | 没 force/safety constraint, action 是 pre-programmed 不是 emergent | 缺 $\tau \in [\tau^{\min}, \tau^{\max}]$ 约束 |

3DGS 公式细节:
$$G(x) = \exp\left(-\frac{1}{2}(x - \mu)^\top \Sigma^{-1} (x - \mu)\right)$$
$$\Sigma = R S S^\top R^\top$$

$\mu \in \mathbb{R}^3$ 是 Gaussian 中心位置, $R \in SO(3)$ 是 rotation (用 quaternion $q$ 参数化, 4 维), $S = \text{diag}(s_1, s_2, s_3)$ 是 scale (3 维)。每个 Gaussian 还有 opacity $\alpha \in [0,1]$ 和 color (用 spherical harmonics, 通常 3rd order 48 维)。**这些都是光学量**, paper 要的是加上 mass $m$、friction coefficient $\mu_f$、restitution $e$、collision volume 等物理量。

参考: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

## Future Work 三个方向的人话

### F1. Physically-Grounded Spatiotemporal Representation

NeRF 的体渲染:
$$C(r) = \int_{t_n}^{t_f} T(t) \sigma(t) c(t) dt$$
$$T(t) = \exp\left(-\int_{t_n}^t \sigma(s) ds\right)$$

$r$ 是 camera ray, $T(t)$ 是从 near plane $t_n$ 到 $t$ 的累积 transmittance, $\sigma(t)$ 是 volume density (NeRF 学的), $c(t)$ 是 color。$\sigma, c$ 全是光学量。

paper 呼吁探索 **new data structure 或 neural implicit 嵌入物理属性**:
$$\text{Object}_i = (\mu_i, \Sigma_i, c_i, \alpha_i, m_i, \mu_{f,i}, e_i, \text{shape}_i, \text{semantic}_i)$$

每个 primitive 还要带 mass $m$、friction $\mu_f$、restitution $e$、碰撞体积。

已有 proto 工作:
- **PhysGaussian** (Yale & MPI): 给 3DGS 加 MPM (Material Point Method)
- **PhysDreamer**: SPRING-based 物理 + 3DGS
- **Spring-Gaus**: 弹簧质点系统 + Gaussian

paper 还强调 low computational overhead 下的 free exploration, 暗示当前 3DGS 几百万 Gaussians per scene 内存爆掉的问题, 需要 structured / compressed 表示。

参考: https://arxiv.org/abs/2407.11884 (PhysGaussian); https://arxiv.org/abs/2404.13026 (PhysDreamer)

### F2. Embodied Interaction and Control

三个 sub-goal:

**High DoF 适配**: 从 7 DoF robot arm 到 Shadow Hand (24 DoF) 再到 humanoid (Figure 02、Tesla Optimus、Unitree H1/G1, 30+ DoF)。

**Sim-to-Real bridge**: World model 生成的 action 要 respect hardware constraints:
- Torque limit: $\tau_i \in [\tau_i^{\min}, \tau_i^{\max}]$
- Joint singularity avoidance: $\det(J(q_t)) > \epsilon$, $J$ 是 Jacobian
- Self-collision constraint

**Long-horizon planning**: 多阶段、复杂 mission, 涉及 causal reasoning。

hardware-aware policy:
$$\pi^*(a_t \mid s_t) = \arg\max_\pi \mathbb{E}\left[\sum_{k=0}^H \gamma^k R(s_{t+k}, a_{t+k})\right]$$
$$\text{s.t.} \quad \tau(a_t) \in [\tau^{\min}, \tau^{\max}], \quad \det(J(q_t)) > \epsilon$$

参考: https://www.figure.ai/ ; https://github.com/unitreerobotics ; https://arxiv.org/abs/2503.06669 (Agibot World)

### F3. Autonomous Reflection and Modular Evolution

最 ambitious 一条。world model 要有 **metacognition**:

1. **Uncertainty estimation**: 预测 $\hat{s}_{t+1}$ 要带 confidence, 用 Bayesian NN / ensemble / evidential learning:
   $$\hat{s}_{t+1}, \sigma_{t+1} = g_\theta(s_t, a_t)$$
   $\sigma$ 是 variance

2. **Discrepancy-triggered reflection**: 观测和预测偏差大就 trigger:
   $$\delta = \|s_{t+1}^{\text{obs}} - \hat{s}_{t+1}\|$$
   if $\delta > \tau_{\text{thresh}}$: trigger reflection

3. **Targeted fine-tuning**: 收集 specific 数据局部更新, 用 LoRA / Adapter / MoE module, 不 full retrain

4. **Modular upgrade**: perception / memory / reasoning / planning 各自独立 fine-tune, 不影响其他。要求 standardized interface。

paper 还吐槽 RL 还是 human-defined reward, 暗示要往 intrinsic motivation / curiosity-driven exploration 走 (ICM、RND、BYOL-Explore)。

参考: https://arxiv.org/abs/1810.12894 (RND); https://openreview.net/forum?id=BZ5a1r-kVsf (LeCun whitepaper)

---

## 我读这篇 paper 的几个 intuition

### Intuition 1: World Model 的本质 = action-conditional transition operator

不管包装多花哨, 数学核心是学 $T(s'|s,a)$。task-specific injection 只学了 $T$ 在 narrow distribution 下的 restriction, 所以 transfer 不行。

判断一个东西是不是 world model 的 quick test: **能不能做 counterfactual rollout** — 给 arbitrary action $a$, 预测 $s' \sim T(s'|s,a)$。Sora 不能 (它只能无条件续生成 video), 所以 Sora 严格说不是 world model, 它是 video distribution model。

### Intuition 2: Generative Quality ≠ World Understanding

Sora / Wan 2.5 视觉质量炸裂 ≠ 它懂物理。判断标准: 长程 consistency、counterfactual support、action-conditional prediction。这三条目前都不满足。

### Intuition 3: Memory 是 consistency 的 substrate

Fig.3c 物体消失案例让我直接看到: 没 persistent $W_t$, 没 consistency。BEV occupancy grid for driving、Object-centric memory for video、Hybrid scene graph for embodied 都是这条思路。

memory 形式可以多样:
- **Spatial**: occupancy grid, 3DGS, NeRF
- **Object-centric**: slots, set-of-objects
- **Episodic**: trajectory replay buffer
- **Semantic**: knowledge graph

但必须有 *persistent across time* 的 state, 不然就是 next-frame predictor。

### Intuition 4: 这 paper 暗合 LeCun 路线

paper 没 cite JEPA, 但它的 "Latent Reasoning" + "physically-grounded representation" 几乎就是 LeCun 2022 whitepaper "A Path Towards Autonomous Machine Intelligence" 的简化版 + 强调多模态 generation。

LeCun 路线: 感知 → world model → actor → critic → configurator → memory。这篇 paper 的 5 modules 是它的工程化拆分。

LeCun 一直 argue: autoregressive LLM 不可能到 AGI, 必须走 world model + planning + latent reasoning。这篇 paper 是中文 community 对这方向的一次 system-level position statement。

参考: https://openreview.net/pdf?id=BZ5a1r-kVsf

### Intuition 5: NVIDIA Cosmos 和 Genesis 是当前最接近 paper framework 的工业实现

**Cosmos** (NVIDIA, 2501): tokenizer + diffusion world model + action conditioning + synthetic data。但 Cosmos 还停在 video generation paradigm, 没做到 physically-grounded representation。

**Genesis** (2412): 用 LLM 写 simulation code, physics engine 当 differentiable layer。更 symbolic。

paper 的 framework 是这二者的 superset。paper 强调 *co-design*, 也就是说不能像 Cosmos 那样只做 generation, 也不能像 Genesis 那样只做 simulation, 要闭环。

### Intuition 6: 这 paper 的 limitation

作为 position paper, 没 empirical validation, 没具体架构。读者可能觉得"说得对但空"。

但 position paper 的价值在 *community direction alignment*, 类比 "Attention is All You Need" 之前更早的 connectionist manifesto。

潜在 risk: unified framework 可能 over-engineer, 让小 lab 进不来。Discussion 5.2 试图回答 (unified ≠ monolithic, 是 modular spec + interface), 但没完全说服。

### Intuition 7: 推测的下一步 hot direction

基于这 paper, 我猜下个 2-3 年这几个方向会爆发:

1. **Physically-grounded 3D representation**: 把 mass / friction / restitution embed 进 Gaussian / primitive, 类似 PhysGaussian 但更通用。会有 ImageNet 级 benchmark 出来。
2. **Action-conditional video world model**: 区别于 Sora 的 unconditional video gen, 这是要给 action 才能 rollout 的, 类似 Dreamer 在 pixel space。GAIA-1/2、Wayve LINGO、Cosmos 在 driving 上试探, 但还没延展到 general domain。
3. **Latent reasoning + LLM 混合**: Coconut 类 continuous CoT, 加上 JEPA 类 latent prediction。会有 paper 把这两者 bridge。
4. **Long-horizon embodied planning with world model rollout**: humanoid robot 上, 用 world model 做 lookahead planning, action grounded 在物理预测上。Figure 02 / Unitree H1 / Optimus 都会往这方向走。
5. **Self-reflection / autonomous evolution**: LLM 上已经看到 (Quiet-STaR、self-refine、Reflexion), 移植到 world model 是自然延伸。

---

## 给 Karpathy 的 TL;DR

这篇 paper 干两件事:

1. **吐槽**: 当前 community 的 "world model" 大多是 task-specific fine-tune with world-knowledge data, 包括 Sora、EditWorld、π0、autonomous driving world model。它们没脱离 downstream task paradigm, 没 genuinely 建模物理 dynamics, 没 long-term consistency, 没 action-conditional counterfactual support。

2. **处方**: 5 module unified framework (Interaction / Reasoning / Memory / Environment / Multimodal Generation), 强调 co-design 和闭环。三个 future direction: physically-grounded representation (给 NeRF/3DGS 加 mass/friction)、embodied control (high DoF + sim-to-real + long-horizon)、autonomous reflection (uncertainty + triggered fine-tune + modular upgrade)。

这 paper 暗合 LeCun JEPA 路线, 是中文 community 对 world model sub-field 的一次 system-level position statement。没 empirical, 但对方向 alignment 有价值。

如果你想 build intuition: 把 world model 当成 *action-conditional transition operator* $T(s'|s,a)$ + persistent memory $W_t$ + multimodal renderer $G(s_t)$。这三件套缺一不可。Sora 缺 action-conditional 和 persistent memory, 所以 Sora 严格说只是 video distribution model, 不到 world model。π0 缺 long-horizon planning 和 physically-grounded representation, 所以 π0 严格说只是 VLA skill model。真正的 world model 还在路上, 这 paper 是路标之一。

---

## 进一步 reading list

如果想深挖:

- Ha & Schmidhuber 2018, World Models (起点): https://worldmodels.github.io/
- LeCun 2022 whitepaper: https://openreview.net/pdf?id=BZ5a1r-kVsf
- Dreamer / DreamerV3 (model-based RL 经典): https://arxiv.org/abs/2301.04104
- TD-MPC2: https://arxiv.org/abs/2310.16840
- JEPA / V-JEPA: https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/
- NVIDIA Cosmos: https://www.nvidia.com/en-us/ai/cosmos/
- Genesis Simulator: https://genesis-sim.github.io/
- PhysGaussian: https://arxiv.org/abs/2407.11884
- π0 (Physical Intelligence): https://www.physical.intelligence/blog/pi0
- Titans (Google, neural memory): https://arxiv.org/abs/2501.00663
- Coconut (Meta, continuous CoT): https://arxiv.org/abs/2412.06769
- Zhu et al. 2024, "Is Sora a World Simulator?": https://arxiv.org/abs/2405.03520
- Hu et al. 2025, "Simulating the Real World" survey: https://arxiv.org/abs/2503.04641
- Sora: https://openai.com/sora
- Wan 2.5: https://tongyi.aliyun.com/wan
- Figure AI: https://www.figure.ai/
- Unitree: https://github.com/unitreerobotics
- GAIA-1 (Wayve, driving world model): https://arxiv.org/abs/2309.17080

下一个 5 年, world model 大概率是 multimodal AI 主战场。这 paper 提供了一个还算清晰的骨架, 可以当 roadmap 看。

---

# 这篇 Paper 的核心定位与论点

这是 Wentao Zhang 团队(北京大学)的一篇 position/survey 性质的文章, 标题就亮明了立场: **"Research on World Models Is Not Merely Injecting World Knowledge into Specific Tasks"**。全文在做一个 meta-level 的诊断: 当前 community 说的"World Model"其实大多只是拿 world knowledge 去做 downstream task fine-tuning, 例如 Sora 拿 RLHF 让视频遵循物理常识、EditWorld 给 image editing 喂带世界规则的数据、VLA 把 LLM 拼到机械臂上。这些做法能在各自 benchmark 上刷点, 然而 intellectually 没有脱离"下游 task + 数据注入"的旧范式。

paper 的核心 contribution 是: 提出一个 **Unified World Model Framework**, 把世界模型拆成 5 个 normative components — **Interaction、Reasoning、Memory、Environment、Multimodal Generation**, 要求它们 co-designed, 形成一个闭环。同时用一批 failure cases(VLM 数错手指、navigation 视频回头时物体消失、3D scene 碎片化、robot 打人)论证碎片化范式不可持续。

参考: https://worldmodels.github.io/ (Ha & Schmidhuber 2018, 原始 World Models)

---

# 1. Background 三大类的技术拆解

paper 把现有工作划成三类, 分别对应 reasoning / generation / interaction 的 side。

## 1.1 Reasoning with World Knowledge

主线是 LLM/VLM + chain-of-thought / explicit reasoning, 代表作 OpenAI o3、Qwen2.5-VL、以及各种 spatial reasoning (SpatialVLM)、competition-level reasoning (SciMaster、Physics Supernova at IPhO 2025)、长视频/音频/3D 多模态推理 (MAVORS、VersaVid-R1、Audio-Reasoner)。

技术上, 这些模型依然是一个 **autoregressive next-token predictor**:

$$P_\theta(y \mid x) = \prod_{t=1}^{T} P_\theta(y_t \mid y_{<t}, x)$$

其中 $y_t$ 是第 $t$ 个 token, $x$ 是 multimodal 输入(可能经过 vision encoder 编码成 visual tokens 注入)。所谓"reasoning"靠的是 chain-of-thought 把 implicit 的推理链 surface 成 explicit tokens, 然后 LLM 在 token 空间做规划。paper 指出这类方法的根本问题: 它是 **statistical fitting of 大规模训练数据**, 没有真正建模物理 dynamics。比如化学 Olympiad 里的分子式识别就崩了, 给六个手指的图它会说五根。

参考: https://openai.com/index/openai-o3/ (o3); https://arxiv.org/abs/2306.14824 (SpatialVLM 前身相关工作)

## 1.2 World-Driven Content Generation

Diffusion 主导, 包括 Flow Matching (Lipman 2022)、Rectified Flow (Liu 2022)、DiT (Peebles & Xie 2023)、Flux、Wan 2.5、Sora 1/2、Veo 3、Seedance、MatrixGame 2.0、WorldPlay 等。

**Flow Matching 公式**(paper 在 background 引了 Lipman 2022):

给定数据分布 $q_1(x)$ 和简单先验 $q_0$ (通常是 Gaussian), flow matching 学习一个 vector field $v_\theta(x, t)$ 使得 ODE

$$\frac{dX_t}{dt} = v_\theta(X_t, t), \quad X_0 \sim q_0, \quad X_1 \sim q_1$$

把 $q_0$ 连续地推向 $q_1$。这里下标 $t \in [0, 1]$ 是 flow 时间(和真实时间无关), $X_t$ 是 flow 路径上的状态。训练目标:

$$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t, x_0, x_1} \left\| v_\theta(x_t, t) - u_t(x \mid x_0, x_1) \right\|^2$$

$u_t$ 是 conditional vector field, 取决于路径构造 (linear interpolation、OT、HF 等)。

paper 的论点: diffusion 在 video/3D 生成上质量起飞了, 然而 **本质上还是 3D world -> 2D rendering 的映射拟合**。Navigation video 生成里, 让 agent 左转一段再右转回来, 原场景里的物体消失 — 这就是 paper Fig.3(c) 的失败案例, 说明模型没维护一个 consistent 的 3D internal state, 它只是 next-frame prediction。

参考: https://arxiv.org/abs/2210.02727 (Flow Matching); https://openai.com/sora (Sora); https://tongyi.aliyun.com/wan (Wan 2.5)

## 1.3 Agents in Interactive Environments

VLA (Vision-Language-Action)、autonomous driving world model (GAIA-1/2、Wayve LINGO、NVIDIA Cosmos)、Minecraft / AI2-THOR / MetaDrive 里的 generalist agent。

数学上, 一般建模成 POMDP:

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, T, O, R, \gamma)$$

- $\mathcal{S}$: state space (这里强调要 physically-grounded)
- $\mathcal{A}$: action space (低 level motor command 到高 level language instruction)
- $\mathcal{O}$: observation space
- $T(s' \mid s, a)$: transition, 即 paper 想要 world model 学的核心
- $O(o \mid s)$: observation emission
- $R$: reward (paper 后面讨论 RL 时提到 reward 还是 human-defined, 这是个 limitation)
- $\gamma$: discount factor

当前 VLA 的做法: 拿 LLM/VLM 作 backbone, 加 action token 输出, 训练数据是 teleop 的 (state, action) pairs。paper 的批评: 这只是把 task-specific skill 套上 LLM 外壳, long-horizon、cross-modal interaction、复杂 DoF 的精细操作都还搞不定。Fig.4(c) 那个 robot 模仿人动作结果把人弄伤的例子很形象, 说明动作本身没有 grounded 在 world model 对物理后果的预测上。

参考: https://www.physical.intelligence/blog/pi0 (π0); https://arxiv.org/abs/2501.03575 (Cosmos); https://arxiv.org/abs/2309.17080 (GAIA-1)

---

# 2. Unified World Model Framework — 5 个组件的技术细节

这是 paper 的核心 contribution, Fig.2 给了示意图。paper 在 Ha & Schmidhuber 2018 原始架构(V: Vision encoder, M: Memory/RNN, C: Controller)基础上扩展, 五个组件:

## 2.1 Interaction — 统一感知与操作接口

原始 World Model:
$$z_t = V(o_t), \quad a_t = C(z_t, h_t), \quad h_{t+1} = M(h_t, z_t)$$

其中 $z_t$ 是 latent observation, $h_t$ 是 memory state(RNN hidden)。Interaction 在原架构里只有 vision encoder $V$ 这条单向通道。

paper 扩展成 **双向 multi-modal interface**:
- 输入侧: text + image + video + audio + 3D point cloud + mesh, 统一编码到 world state representation $s_t$
- 输出侧: 既产生 high-level instruction (e.g. "把红色方块放到蓝色上面"), 又产生 low-level motor command (关节角度、力矩、车辆 control signal)

一个比较自然的实现思路是 **modality-specific tokenizers + shared latent space**:
$$\text{tok}_m(x^{(m)}) = \phi_m(x^{(m)}), \quad m \in \{\text{text, img, vid, audio, pc, mesh, action}\}$$
然后所有 tokens 进一个 shared transformer / SSM 做 cross-attention fusion。

我联想: 这很像 NVIDIA Cosmos 的 tokenizer 设计 (continuous visual tokens + action tokens), 也像 π0 的 flow-matching VLA 把 action 也放进 token 空间。但 paper 强调的是 *接口要 normative*, 也就是说社区要约定 latent 的 schema、token 的格式, 不然各个工作自己玩自己的没法 benchmark。

## 2.2 Reasoning — Explicit vs Latent 推理

paper 区分两种 reasoning:

**Explicit Reasoning**: 多模态观测先转成文字描述/reasoning chain, 再丢给 LLM 做 symbol manipulation。优点是 transparent、可解释、容易和人类直觉对齐; 缺点是 **sub-symbolic 信息丢失**。比如物体的精确 3D 位姿、流体动力学、摩擦力瞬时值, 这些连续量塞进自然语言 token 就丢了精度。

**Latent Reasoning**: 在 unified latent space 里直接做 forward inference, 不绕道 text。形式化:

$$s_{t+1} = g_\theta(s_t, a_t, \text{context})$$

其中 $s_t \in \mathbb{R}^d$ 是 latent world state, $g_\theta$ 是 reasoning operator。这个思路让人想到 **JEPA** (LeCun 的 Joint Embedding Predictive Architecture):

$$\mathcal{L}_{\text{JEPA}} = \left\| \text{sg}(s_{t+1}) - \hat{s}_{t+1} \right\|^2, \quad \hat{s}_{t+1} = \text{Pred}(s_t, a_t, \text{context})$$

其中 $\text{sg}$ 是 stop-gradient, 防止 representation collapse。这个和 paper 里 "Latent Reasoning" 几乎是同一思路。

paper 没有给具体公式, 但直觉上他想要的 reasoning module 是 **混合 explicit + latent**: 用 language reasoning 做 high-level planning, 用 latent reasoning 做物理细节预测。我直觉这和 recent 的 reasoning-with-latent-thinking 方向一致, 比如 Coconut (Meta 的 continuous chain of thought)、Pause Tokens、Quiet-STaR。

参考: https://openreview.net/forum?id=vhg8pn9fJC (JEPA / V-JEPA); https://arxiv.org/abs/2412.06769 (Coconut)

## 2.3 Memory — 从 LSTM 到 structured dynamic memory

paper 把 memory 的演化讲成一条线:
- LSTM (Hochreiter & Schmidhuber 1997): $h_t = f(h_{t-1}, x_t)$, 隐式 state
- xLSTM (Beck 2024): extended LSTM, exponential gating
- Long-context Transformer (Longformer, FlashAttention-2): explicit 大窗口 attention
- MemFlow (Ji 2025): flowing adaptive memory

paper 提出对 world model 的 memory 要 **structured + dynamic**:

1. 多模态、多源 experiential data 的 fuse 与 association, 构建 **queryable internal knowledge system**
2. Key information extraction & compression (active filter)
3. Dynamic update, merge / purge redundant content

如果用形式化表达, 一个比较自然的框架是 **Memory-augmented network**:

$$M_t = \text{Update}(M_{t-1}, \text{Write}(s_t, a_t, r_t, s_{t+1}))$$
$$\text{context}_t = \text{Read}(M_t, \text{query}_t)$$

其中 $M_t$ 是 memory matrix $\mathbb{R}^{N \times d}$, $N$ 是 slot 数。这接近 **Memory Networks** (Weston 2014) 或 **Differentiable Neural Computer** (Graves 2016)。

我联想: 这个方向最近比较热的工作是 **Memory3D** (HKU)、** Titans** (Google, neural memory with attention), 它们的核心思想是把 memory 当作可学习、可写入、可遗忘的模块, 而不仅是 KV cache。paper 这部分写得偏 conceptual, 没有具体说要在哪个 level 做 memory。

参考: https://arxiv.org/abs/2502.14107 (Memory3D 类的工作, 注意不要混淆, 这是指可学习记忆); https://arxiv.org/abs/2406.19246 (Titans)

## 2.4 Environment — Generative and Extensible Simulator

这部分 paper 讲得比较关键, 论点是: **environment 本身应该是 generative 的**, 而不仅是手动建模的 sim (AI2-THOR、MetaDrive)。理由:

- 真实硬件数据采集成本高
- 手动建模 sim 有 sim-to-real gap (rigid body dynamics, 有限场景)
- generative environment (3D generation + procedural content) 能动态合成 near-infinite 高保真场景

形式化, environment 可以写成一个 generative distribution:

$$p_\phi(s_{t+1} \mid s_t, a_t, \text{world params } w)$$

其中 $w$ 是 procedural seed 或 latent scene code。当前 3D generation 工作像 Hunyuan3D、Triplane 到 Gaussian、MIDI (multi-instance diffusion) 都是这方向的雏形, 但 paper 批评它们只做到"visual plausibility", 没有 physical attributes (mass、friction、collision volume)。

我直觉: paper 在这里其实在暗示一种 **neuro-symbolic hybrid**: 一边是 neural scene generator 产生 high-fidelity visual + geometry, 一边是 symbolic physics engine (Bullet、MuJoCo、PhysX) 给物体贴物理属性, 二者通过 shared representation 耦合。这是 NVIDIA Cosmos 2.0、Genesis、PhysGen 等 simulator-LLM 融合工作正在走的路。

参考: https://github.com/genesis-sim/Genesis (Genesis simulator); https://arxiv.org/abs/2501.03575 (Cosmos)

## 2.5 Multimodal Generation — 闭环反馈

paper 强调 generation 不应是孤立输出 module, 而是 **闭环**:
- 生成 video / image / 3D 几何 → 既作为给人类的 feedback, 也作为给 reasoning module 的 **model-based foresight**
- 生成数据 → 用于 **self-augmentation**

这个思路直接对应 **model-based RL** 里的 world model 用法: 用 world model 产生的 rollout 做 planning, 形式化:

$$\hat{s}_{t+1:t+H} = \prod_{k=0}^{H-1} g_\theta(s_{t+k}, a_{t+k})$$
$$\pi^* = \arg\max_\pi \mathbb{E}_{\hat{s}} \sum_{k=0}^{H} \gamma^k R(\hat{s}_{t+k}, \pi(\hat{s}_{t+k}))$$

其中 $H$ 是 planning horizon, $g_\theta$ 是 world model transition, $\pi$ 是 policy。这是 Dreamer (Hafner 2019/2023)、TD-MPC 的经典做法。

paper 把它扩展到 multimodal 输出: 不光预测 next latent state, 还要 render 成 video/3D, 形成 "internal simulation of its own navigation strategy and scene comprehension"。这其实就是 Sora 当初被定义为 "world simulator" 的初衷, 但 paper 论证 Sora 目前还没做到这点(因为它只是 next-frame pixel prediction)。

参考: https://arxiv.org/abs/1912.01603 (Dreamer); https://arxiv.org/abs/2304.10573 (TD-MPC2); https://arxiv.org/abs/2305.14325 (DreamerV3)

---

# 3. Limitations 案例的技术解读

paper 用 Fig.3 和 Fig.4 列了一堆 failure cases, 我把它系统化:

## 3.1 VLM 数错手指(Fig.3a)

VLM backbone 在 LAION 级别数据上预训练, "five fingers" 是一个 high-frequency prior。当输入是六根手指的 unnatural image, model 仍输出 "五根"。

数学上, 这是 **prior dominance over evidence**:
$$P_\theta(\text{five} \mid x) > P_\theta(\text{six} \mid x) \quad \text{even when } x \text{ shows six fingers}$$

因为 $\theta$ 被训练成最大化 $\sum_{(x_i, y_i)} \log P_\theta(y_i \mid x_i)$, 训练集里 $y_i = \text{five}$ 占绝对多数, 所以 likelihood 被 prior 主导。

## 3.2 Image Editing 光影不一致(Fig.3b)

当前 EditWorld / AnyEdit 类方法用 instruction tuning 教 diffusion model 改图, 但模型学的是 "**instruction -> pixel delta**" 的 mapping, 没显式建模 lighting / shadow physics。

diffusion 的反向过程:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon_\theta(x_t, t, c) \right) + \sigma_t z$$

其中 $c$ 是 condition (instruction + 原图), $\bar\alpha_t = \prod_{i=1}^t \alpha_i$, $\beta_t$ 是 noise schedule, $\sigma_t$ 是 variance。$\epsilon_\theta$ 学的是 noise prediction, 没有任何机制保证光线方向一致。所以加了物体但阴影方向跟着原图 training distribution 跑了, 违反物理。

## 3.3 Navigation Video 物体消失(Fig.3c)

这个最直接暴露 **没有 memory module** 的问题。Navigation video generation 形式化:

$$\hat{I}_{t+1} = G_\theta(I_{1:t}, a_{1:t})$$

$G_\theta$ 是 next-frame generator (DiT 等)。当 agent left 然后 right 回到原位, 模型只看 recent frames 做预测, 没有显式 memory 记录 "5 步前这里有个椅子", 所以椅子消失。

正确做法应该让 generator 依赖一个 persistent world state $W$:
$$\hat{I}_{t+1} = G_\theta(I_{1:t}, a_{1:t}, W_t), \quad W_{t+1} = \text{Update}(W_t, I_t, a_t)$$

$W$ 是显式 scene memory (可以是 3DGS, 可以是 occupancy grid, 可以是 latent)。这和 BEV (Bird's Eye View) world model for autonomous driving 的思路一样。

参考: https://arxiv.org/abs/2309.17080 (GAIA-1 用 BEV-style memory)

## 3.4 高速动态视频物理崩坏(Fig.3d)

Sora 类模型在 fast dynamics (爆炸、碰撞、流体) 场景下产生违反物理的结果, 因为它们本质是 pixel-level 拟合, 没有学 $\text{Newton's laws}$ 的 latent representation。

## 3.5 3D Scene 碎片化(Fig.3e)

当前 3D generation (SAM3D、MIDI、FlashWorld、WorldMirror) 输出的 3D 点云 / Gaussians 在 detail 上扭曲, 因为点云是 **离散、非结构化** 表示, 难以对应 consistent physical entity。

**3D Gaussian Splatting 公式**:
$$G(x) = \exp\left(-\frac{1}{2}(x - \mu)^\top \Sigma^{-1} (x - \mu)\right)$$
$$\Sigma = R S S^\top R^\top$$

其中 $\mu \in \mathbb{R}^3$ 是 Gaussian 中心, $R \in SO(3)$ 是 rotation (用 quaternion $q$ 参数化), $S = \text{diag}(s_1, s_2, s_3)$ 是 scale。每个 Gaussian 还有 opacity $\alpha$ 和颜色 (spherical harmonics)。

paper 的论点: 这个表示只有 *光学* 属性, 没有 *物理* 属性 (mass $m$、friction $\mu_f$、restitution coefficient $e$、collision volume)。所以场景看着逼真但物理上根本不连续。

参考: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ (3DGS)

## 3.6 Embodied AI / 自动驾驶失效(Fig.4)

- (a) embodied AI 只能做简单 pick-and-place, 长程复杂任务不行
- (b) autonomous driving 在直路上也会莫名 turn 失败
- (c) robot 模仿人动作但因为没有 force / safety constraint 把人弄伤

这些共同点: **当前 embodied system 是 "LLM/VLM + 物理硬件" 的拼装, control 不是 world model 的 emergent property, 而是 pre-programmed。**

paper 论点: 我们应该让 control *从* world model 的物理理解中 *emerge* 出来, 而不是反过来把 control 套一层 LLM。

---

# 4. Future Work 三个方向的技术深度

## 4.1 Physically-Grounded Spatiotemporal Representation

paper 这部分最 technical, 论点是超越 NeRF / 3DGS 这种 optical representation, 走向 **physically-grounded representation**。

NeRF 体渲染:
$$C(r) = \int_{t_n}^{t_f} T(t) \sigma(t) c(t) \, dt$$
$$T(t) = \exp\left(-\int_{t_n}^{t} \sigma(s) ds\right)$$

其中 $r$ 是 ray, $T(t)$ 是 accumulated transmittance (从 near plane $t_n$ 到 $t$ 的累积透明度), $\sigma(t)$ 是 volume density (NeRF 学的), $c(t)$ 是 color。$\sigma$ 和 $c$ 完全是光学量, 物理属性缺失。

paper 建议探索 **new data structures 或 neural implicit representations 嵌入物理属性**, 比如:

$$\text{Object}_i = (\mu_i, \Sigma_i, c_i, m_i, \mu_{f,i}, e_i, \text{shape}_i, \text{semantic}_i)$$

即每个 Gaussian / primitive 还要带 mass $m$、friction $\mu_f$、restitution $e$、碰撞体积等。

这种思路已经有 proto 工作: **PhysGaussian** (Yale & MPI)、**PhysDreamer**、**Spring-Gaus** 都是给 3DGS 加 MPM (Material Point Method) 或 SPRING-based 物理; **Genesis** simulator 则是从头把神经表示和物理引擎耦合。

paper 还特别强调 *low computational overhead 下的 free exploration*, 暗示当前的 3DGS 内存爆掉 (1 个 scene 几百万 Gaussians) 的问题, 需要 structured / compressed representation。

参考: https://arxiv.org/abs/2407.11884 (PhysGaussian); https://arxiv.org/abs/2404.13026 (PhysDreamer); https://genesis-sim.github.io/

## 4.2 Embodied Interaction and Control

三个具体 sub-goal:

1. **High DoF 适配**: 从 simple grasping (parallel-jaw, ~7 DoF) 到 dexterous manipulation (Shadow Hand, 24 DoF; full-body humanoid, 30+ DoF)
2. **Sim-to-Real bridge**: World model 产生的 action sequence 要 **respect hardware constraints**, 包括:
   - Torque limit $\tau_i \in [\tau_i^{\min}, \tau_i^{\max}]$
   - Joint singularity avoidance (Jacobian 奇异点)
   - Self-collision constraint
3. **Long-horizon planning**: 多阶段、复杂 mission, 涉及 causal reasoning

形式化, hardware-aware policy:
$$\pi^*(a_t \mid s_t) = \arg\max_\pi \mathbb{E}\left[\sum_{k=0}^{H} \gamma^k R(s_{t+k}, a_{t+k})\right]$$
$$\text{s.t.} \quad \tau(a_t) \in [\tau^{\min}, \tau^{\max}], \quad \det(J(q_t)) > \epsilon$$

其中 $J(q_t)$ 是 Jacobian, $\det(J) > \epsilon$ 是避免奇异构型。

paper 这里其实在呼应最近 humanoid robot 的爆发 (Figure 02、Tesla Optimus、Unitree H1/G1、Agibot), 这些都要求 action-level 的物理 grounding。

参考: https://www.figure.ai/ ; https://github.com/unitreerobotics ; https://arxiv.org/abs/2503.06669 (Agibot World)

## 4.3 Autonomous Reflection and Modular Continuous Evolution

这条最 ambitious: world model 要有 **metacognition** 和 **self-reflection**。

机制设计:

1. **Uncertainty estimation**: world model 对自己的预测 $\hat{s}_{t+1}$ 要带 confidence, 比如 Bayesian neural network、ensemble、evidential deep learning:
   $$\hat{s}_{t+1}, \sigma_{t+1} = g_\theta(s_t, a_t)$$
   $\sigma$ 是预测 variance

2. **Discrepancy-triggered reflection**: 当观测 $s_{t+1}^{\text{obs}}$ 与预测 $\hat{s}_{t+1}$ 偏差过大:
   $$\delta = \|s_{t+1}^{\text{obs}} - \hat{s}_{t+1}\|$$
   if $\delta > \tau_{\text{thresh}}$: trigger reflection

3. **Targeted fine-tuning**: 收集 specific 数据或 replay high-value samples 做局部更新, 而不是 full retrain。这指向 **LoRA / Adapter / MoE modular update**。

4. **Modular upgrade**: perception / memory / reasoning / planning 各自独立 fine-tune, 不影响其他 module。这要求 **standardized interface**。

paper 还吐槽当前 RL 还是 human-defined reward, 暗示要往 **intrinsic motivation / curiosity-driven exploration** 走 (ICM、RND、BYOL-Explore)。

这部分让我想到 LeCun 最近一直推的 **JEPA + self-supervised world model + planning**, 还有 Anthropic 的 **constitutional AI** 思路扩展到 world model 的 reflection。

参考: https://openreview.net/forum?id=BZ5a1r-kVsf (Plan2Explore / RND 思路); https://yann.lecun.com/jepa

---

# 5. Discussion — Efficiency vs Generalization, Diversity vs Integration

paper 在 Section 5 论证 unified framework 不会扼杀 diversity, 而是 **standardize interfaces**:

- **Efficiency vs Generalization**: task-specific fine-tune 在 static metric 上赢, 但 open-ended environment 下 hit performance ceiling defined by training data。unified framework 提供 knowledge transfer / lifelong learning 的结构基础。
- **Diversity vs Integration**: unified ≠ monolithic network, 而是 modular functional spec + standardized interface。

我直觉这其实在类比 **POSIX / ROS / LLVM IR** 这种 *interface standardization*, 让不同 module 可以 plug-and-play。

---

# 6. 我的 Intuition 构建与延伸联想

读完这篇 paper, 我的几个 takeaways:

**6.1 World Model 的本质 = transition operator**

不管包装得多么花哨, world model 的数学核心是学 $T(s' \mid s, a)$。当 task-specific injection 时, 你只学了 $T$ 在某个 narrow distribution 下的 restriction, 这就是为什么 transfer 不行。

**6.2 Generative ≠ World Model**

Sora 生成逼真视频, 但它不是 world model, 因为它没有 *action-conditional* 的 transition — 你不能说"如果我现在往左推这个球, 接下来 1 秒会怎样", 它只能 *无条件* 续生成视频。真正的 world model 必须支持 counterfactual: $s' \sim T(s' \mid s, a)$ for *arbitrary* $a$。

**6.3 Memory 的本质是 persistent state**

Fig.3c 物体消失案例让我清楚看到: 没有 persistent $W_t$, 就没有 consistency。BEV occupancy grid for driving、Object-centric memory for video、Hybrid scene graph for embodied 都是这条思路。

**6.4 这篇 paper 和 JEPA / Dreamer / LeCun 路线的暗合**

paper 没显式 cite JEPA, 但它的 "Latent Reasoning" 和 "physically-grounded representation" 几乎就是 LeCun 路线。LeCun 在过去 2 年一直 argue: autoregressive LLM 不可能到 AGI, 必须走 world model + planning + latent reasoning。这篇 paper 是中文 community 对这个方向的一次系统化 position statement。

**6.5 和 NVIDIA Cosmos / Genesis 的对照**

NVIDIA Cosmos (2501) 是目前 industrial 最接近 paper framework 的工作: tokenizer + diffusion world model + action conditioning + synthetic data。但 Cosmos 也还停在 video generation paradigm, 没做到 physically-grounded representation。

Genesis (2412) 则是另一个方向: 用 LLM 写 simulation code, 把 physics engine 当作 world model 的不同iable layer。这条路更 symbolic。

paper 的 framework 可以视作这两者的 superset。

**6.6 与 LeCun "A Path Towards Autonomous Machine Intelligence" 的对照**

LeCun 2022 whitepaper: 感知 -> world model -> actor -> critic -> configurator -> memory。这篇 paper 的 5 modules 是它的简化版 + 多模态 generation 强调。

参考: https://openreview.net/pdf?id=BZ5a1r-kVsf (LeCun whitepaper)

**6.7 反思: paper 的 limitation**

这篇 paper 本身是 position, 没 empirical validation, 没提出具体架构。读者可能觉得"说得对但空"。然而对于 *社区方向 alignment*, 这种 position paper 是有价值的 — 类似 "Attention is All You Need" 之前 transformer 还没有, 但更早的 "Connectionist AI" manifesto 推动了方向。

另一个潜在 risk: unified framework 可能 over-engineer, 让小 lab 进不来。这是 Discussion 5.2 想回答但没完全回答的问题。

**6.8 涉及到的关键 open problem**

- How to embed physics attributes into neural representations (4.1)
- How to define modular interface between perception / reasoning / memory (Section 3)
- How to do uncertainty estimation in world model (4.3)
- How to learn reward autonomously instead of human-defined (4.3)
- How to bridge latent reasoning and explicit reasoning (3.2)

---

# 7. 综合评价

作为一篇 position paper, 它做了三件事:

1. **诊断**: 系统指出当前 Sora / VLA / image editing 等"world-knowledge-infused"工作的碎片化与表层性
2. **处方**: 提出五模块 unified framework, 强调 co-design 与闭环
3. **未来**: 列出三个 critical direction, 落到 representation / control / self-evolution

writing 上比较 dry, 但 Fig.1 / Fig.2 / Fig.3 / Fig.4 都很有信息密度, 是少见的把 position 和 failure case 都摆出来的 survey 风格。

和 recent 同类 survey 比较:
- Zhu et al. 2024 "Is Sora a World Simulator" — 更聚焦 video generation
- Hu et al. 2025 "Simulating the Real World" — 多模态 generative model survey
- 这篇 paper — 更偏 framework design + future direction

如果让我给一个简短判断: 这篇 paper 是 **"World Model 作为一个 sub-field 的范畴论 unification 尝试"**, 不算开创性, 但是 community 对方向焦虑时一个不错的 reference point。对 PhD 选方向、对工业界定 roadmap 都有参考价值。

---

# 8. 关键 References 与延伸阅读链接

核心被引与扩展:
- Ha & Schmidhuber 2018, World Models: https://worldmodels.github.io/
- LeCun 2022, A Path Towards Autonomous Machine Intelligence: https://openreview.net/pdf?id=BZ5a1r-kVsf
- Hafner et al., Dreamer / DreamerV3: https://arxiv.org/abs/2301.04104
- Lipman et al. 2023, Flow Matching: https://arxiv.org/abs/2210.02727
- Mildenhall et al., NeRF: https://www.matthewtancik.com/nerf
- Kerbl et al., 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- NVIDIA Cosmos: https://www.nvidia.com/en-us/ai/cosmos/
- Genesis Simulator: https://genesis-sim.github.io/
- PhysGaussian: https://arxiv.org/abs/2407.11884
- π0 (Physical Intelligence): https://www.physical.intelligence/blog/pi0
- Agibot World: https://arxiv.org/abs/2503.06669
- Titans (Google, memory): https://arxiv.org/abs/2501.00663
- Coconut (Meta, continuous CoT): https://arxiv.org/abs/2412.06769
- JEPA / V-JEPA (LeCun): https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/
- Sora: https://openai.com/sora
- Wan 2.5 (Tongyi): https://tongyi.aliyun.com/wan
- Gemini 1.5 / Veo 3: https://deepmind.google/technologies/veo/
- Figure AI (humanoid): https://www.figure.ai/
- Unitree robotics: https://github.com/unitreerobotics
- Zhu et al. 2024, "Is Sora a World Simulator?": https://arxiv.org/abs/2405.03520
- Hu et al. 2025, "Simulating the Real World" survey: https://arxiv.org/abs/2503.04641
- Cosmos-Reason (reasoning dataset): https://github.com/nvidia-cosmos/cosmos-reason1
- Wayve LINGO / GAIA: https://wayve.ai/

paper 给的 framework 是一个 conceptual skeleton, 真正的工程实现还要等社区把每个 module 的 interface 跑出来, 形成 de facto standard。下一个 5 年, 这个方向大概率会成为 multimodal AI 主战场之一。
