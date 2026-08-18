---
source_pdf: A Definition and Roadmap for World Models.pdf
paper_sha256: 3a153bded157354e47a682f8141e782ba341ab3e7d7e0fcab1d70628eae1df22
processed_at: '2026-08-17T23:03:17-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇 paper 在说什么

## 一句话版本

**World model 就是个超级压缩器，把互联网上几十亿小时的视频里藏着的物理规律榨出来，压缩成一个紧凑的内部表示，然后从这一个表示里既能生成画面、又能模拟物理、还能规划动作。**

---

## 为什么现在要写这篇 paper

"World model" 这个词现在被用得特别乱。做 video generation 的人说 Sora 是 world model，做 RL 的人说 Dreamer 是 world model，做机器人的人说 WAM 是 world model，做自动驾驶的人说 GAIA-1 是 world model。大家各说各话，鸡同鸭讲。

Fei-Fei Li 试图理清这个混乱，提出把 world model 分成 renderer（生成画面）、simulator（模拟状态转移）、planner（提议动作）三类。但这篇 paper 说这个分类有根本问题：**它在分类输出，没分类内部表示**。就好比你问"这是一把什么刀"，人家回答"能切菜、能削苹果、能开信封"——这没回答问题，因为一把好刀可以做到以上所有。

LeCun 那边又说：别做 pixel-level reconstruction 了，在 latent space 预测就够了，pixel reconstruction 浪费 capacity。这篇 paper 的态度是：pixel reconstruction 确实不该是终极目标，但 reconstruction quality 是个很好的诊断工具，能告诉你 representation 丢了什么信息。

所以这篇 paper 想干一件事：**给 world model 下一个大家都能用的科学定义，然后画出一条从现在到 Physical AGI 的工程路线。**

---

## 核心 framing：World model 是压缩器

这是整篇 paper 最关键的 insight，也是最容易被人忽略的。

你想，互联网上有几十亿小时的视频。这些视频里藏着什么？藏着物理世界的所有规律：物体不会凭空消失（object permanence）、刚体和软体的区别、运动学约束、光照和阴影怎么变、遮挡和重现的因果逻辑、人类动作的结构。但这些规律全部埋在 raw pixels 里，你直接看 pixels 看不出来。

Raw pixels 大部分是 task-irrelevant 的 photometric noise——视角变化、光照、纹理、sensor noise、背景杂物。这些 noise 把真正的物理信号遮住了。

所以 world model 要做的事情是：**把这些高维、嘈杂的视频观测，蒸馏成紧凑、semantically structured 的表示，保留 decision-relevant 的因果和物理信息，系统性地丢掉 irrelevant 的 photometric noise。**

这就是一个 compression problem。Generation 和 simulation 是好的 compressed representation 下游自然涌现的能力，但它们本身不是 objective。就好比你把一本书读透了，你能复述、能引用、能基于它推理——但"读透"本身才是核心，复述和引用只是副产品。

---

## 数据决定天花板，不是模型

这个论点很简单但很 sharp：**在固定 architecture 和 compute 下，是 data diversity 决定 generalization ceiling，不是 model structure 或 hyperparameters。**Architecture 和 compute 决定你多快逼近 ceiling，但不能抬高 ceiling。

这直接指向了 Figure 1 的 Inverted Pyramid Workflow。逻辑是：

互联网视频是唯一能 scale 到所需 diversity 的 physical data source。Proprietary 数据——灵巧操作数据集、第一人称 embodied 录制、外骨骼传感器——在 diversity 和 scale 上结构性受限。没有任何硬件部署能在可见未来产生 internet-scale 的操作数据。

但互联网视频是被动观测的，没有 action label，物理结构藏在 pixels 里。所以 funnel 的设计是：

- **顶层**：数十亿的 raw web video
- **中层**：自动 filter 和 annotate，提炼出 physical knowledge，合成标准化 action representation
- **底层**：小而精的 task-aligned real-world dataset，用于 end-to-end robot fine-tuning

这个 funnel 先用互联网的广度提升 generalization ceiling，再用精度蒸馏得到紧凑的 task-specific 数据。Architecture 和 compute 只管学得多快，但这个 inverted pipeline 扩展了 training corpus 的 foundational physical experience diversity——抬高了 world model 对未见物体、运动、因果场景泛化的硬上限。

---

## 正式定义和三个性质

Paper 给的定义：

> **World model 是在有限计算资源约束下，对物理世界 state transition processes 的 compression modeling。**

三个 inherent properties：

**Omnimodal workscope**：不能只会 text 或 vision，必须能统一建模所有感知模态。

**Multidimensional Asynchronicity**：物理世界各维度数据采样频率不同——robot proprioception 可能 100Hz，vision 30Hz，language instruction 稀疏。World model 必须处理这种异步。

**Locality**：agent 感知受限，只能观察局部，外部世界不断干预。所以天然是 POMDP（部分可观测马尔可夫决策过程）。

这个定义把 world model 从"预测未来"的框架拉到"理解当前、解释为什么、预测未来"的三位一体。数字世界里数据一旦创建就永久可读，但物理世界 non-stationary、不断变化，最重要的安全事件和罕见故障模式可能永远不在训练数据里。所以 world model 必须能从 universal foundational representation 局部适应到具体应用的物理特性，autonomously、real-time。

---

## Understanding vs Prediction：谁为主

Paper 的立场很明确：**Understanding 应该是 primary，prediction 应该 service 它。**

Understanding-oriented 的 model 压缩 sensory data 成 stable internal representations，expose 实体、关系、机制。Prediction 是 training signal，不是终极目标。

Prediction-oriented 的 model judged by ability to roll world forward、generate futures。LeCun 的 JEPA 和 Sora 是这个路线。

两者 failure mode 不同：
- Understanding model 可能推断出 compact decision-relevant state，但 render 出来的画面很丑
- Video generator 可能画面惊艳，但作为 control-relevant simulator 不可靠，因为不保留 hidden state、causal structure、intervention semantics

Paper 的判断：一个不能识别 latent operating state、causal structure、intervention-relevant uncertainty 的模型，可能 generate plausible futures，但作为 scientific reasoning 或 embodied control 的基础会失败。Predictive rollout 依然不可或缺，但主要作为通过 counterfactual simulation 测试和 refine internal understanding 的 mechanism。

---

## 两种 Taxonomy 互补

**Functional taxonomy**（Fei-Fei）：按 agent-environment loop 里的角色分——renderer（生成观测）、simulator（传播状态）、planner（选动作）。

**Representational taxonomy**（这篇 paper）：按 representation substrate 分——observation-level（在 pixels 预测）、latent-space（在 compact states 预测）、3D/structured（显式建模 geometry、objects、relations）。

一个问 "what does it do"，一个问 "how is knowledge represented"。互补。

WAM（World Action Model）不是第四个 implementation category，是 cross-architectural functional paradigm。它的 defining commitment 是 **predictive state modeling 和 action generation 的耦合**。可以是 cascaded（先想象未来再 derive action）或 joint（co-model state 和 action）。可以在 observation-level、latent-space、structured 任何 substrate 上 instantiate。

---

## Architectural 路线

**Observation-level**（Sora、Genie、Cosmos）：scale 优势大，互联网视频是巨大 weakly supervised corpus。但 visual plausibility ≠ physical correctness，长 horizon rollout 会 drift，计算昂贵。

**Latent-space**（Dreamer、JEPA、TD-MPC）：从 modeling visual appearance 转向 modeling task-relevant state evolution。不花 capacity 重建每个 RGB detail，focus 在 slow-varying、decision-relevant factors。风险是 excessive abstraction 可能丢掉 visually small 但 decision-critical 的信息——gripper-object contact、薄障碍物、subtle pose change。

**3D/Object-centric**（OccWorld、NeRF、Gaussian Splatting、Slot Attention）：让 prediction 变得 spatially queryable，支持 free space、occlusion、viewpoint consistency。但加 3D structure 不自动解决 world modeling——可能重建空间准确但抓不到 affordance、causality、object permanence。

**Unification trend**：从 pure generator → generation-understanding unified model → 加 action grounding 的 WAM → omnimodal physical modeling。Cosmos 3 是 exemplar：在 Mixture-of-Transformers 里 unify language、image、video、audio、action。

---

## 训练范式

**Self-supervised pretraining**：video prediction $p_\theta(\mathbf{x}_{t+1:t+K} \mid \mathbf{x}_{\leq t}, \mathbf{a}_{t:t+K-1}, \mathbf{c})$ 是最直接 formulation。Action-free prediction 学 observational dynamics；action-conditioned prediction 学 interventional dynamics——对 planning 是关键区别。

Next-token prediction $\mathcal{L}_{\mathrm{NTP}} = -\sum_t \log p_\theta(u_t \mid u_{<t}, \mathbf{a}_{\leq t}, \mathbf{c})$ 是最 scalable interface，但 hidden bottleneck 是 tokenization——tokenizer 丢掉的 geometry、contact、controllability 永远靠 scale 补不回来。

Scaling law $\mathcal{L}(N,D) \approx \mathcal{L}_\infty + AN^{-\alpha} + BD^{-\beta}$ 里，对 world model 来说 $D$ 不只是 token count，是 coverage over states、actions、embodiments、viewpoints、contacts、rare events、horizons。

**MBRL**：两阶段——fit dynamics model $\widehat{P}_\theta$，再 exploit 它做 planning 或 imagined rollout training。三个 failure mode：
1. **Compounding error**：autoregressive rollout 把 model 自己的预测 feed 回去，one-step error 累积
2. **Objective mismatch**：model 训 likelihood，但 evaluate by policy return
3. **Optimism-pessimism bias**：planner 是自己 model 的 adversarial consumer，会 exploit model 的 optimistic errors

**WAM vs MBRL 的关键区别**：MBRL decouples model learning 和 policy optimization；WAM 在 single generative process 里 coupling dynamics prediction 和 action generation。WAM 从 video-action pretraining 继承 broad physical priors，支持 zero-shot task generalization 和 few-shot embodiment adaptation，但 cost 是 modularity 和 calibrated uncertainty。

**Chain-of-Imagination (CoI)**：reasoning 不再只是 language chain-of-thought，而是在 learned dynamical space 里的 action-conditioned transitions。一个 thought 可以是一个 action-conditioned state transition。分支 $B_t$ 个想象、每个 depth $K_b$，最后 $\widehat{\mathbf{a}}_t = \pi_\phi(\mathbf{z}_t, \mathrm{Agg}_\psi(\mathcal{C}_t))$ 同时 conditioned on 当前 belief 和 aggregated counterfactuals。Auto-think Switch 让 reasoning budget 由 risk 和 uncertainty govern。

**Physics-informed**：三个 level——soft penalty（PINN 风格）、hard architecture（Hamiltonian/Lagrangian NN）、hybrid differentiable physics + learned components。对 visual world model 来说，physics 应该 constrain latent variables（对应 action-relevant physical quantities），让 appearance-related factors 留给 flexible generative components。

**Counterfactual reasoning**：不是 generate 另一个 plausible rollout，是 replay 同一个 world 只改一个 decision。Same-world constraint：latent circumstances 保持 fixed，只修改 intervened mechanism。$\mathbb{P}(Y_{a'} \mid E=e) \neq \mathbb{P}(Y \mid A_t=a', E=e)$——左边保持 factual world fixed，右边比较 other episodes。Counterfactual 从 observational accuracy 单独一般 non-identifiable，需要 structural commitments。

**Long-horizon planning**：不通过 rolling 更远，通过改变 planning unit。Hierarchy 把 $H$ primitive actions 替换成 $K \ll H$ abstract commitments，gain 是 lower-entropy search problem。Horizon-limited value gap $\Delta_H = |V_{\widehat{p}_\theta, H} - V_{P^\star, H}|$ 当 rollouts 离开 data-supported 区域时增长。

---

## 应用领域

**Robotics**：三个 role 按 computation budget 分——Data Engines（offline，RoboDreamer/UniSim）、Environment Simulators（evaluation，MILE/ReSim/SIMPLER）、Action Planners（online control loop，DayDreamer）。还有 Embodiment World Models（BFM，model agent 自己 body 能做什么，是 internal body schema 而非 external scene model）。

**Scientific Discovery**：autonomous lab + predictive foundation。GraphCast 是 structured scientific simulator 但不是 agentic（不选 intervention）。Medical World Model 模拟 treatment-conditioned tumor evolution。

---

## Open Challenges

**Data Asymmetry**：renderer 能 scale 在 internet video 上；simulator bottleneck 是 simulation-ready 3D assets（缺 scale、material、collision、mass、friction、articulation）；planner bottleneck 是 action-conditioned interaction data。Embodied AI data pyramid：底部是 abundant passive video，中间是较少的 simulation-ready assets，顶部是稀缺但高价值的 tactile/force feedback。

**Fidelity vs Precision**：visual quality 提升不等于 physical correctness 提升。PhyGenBench、PhyWorldBench 等 benchmark 证据。需要从 fidelity-first generation 转向 physical-precision-centered world modeling。

**Compounding Error**：long-horizon rollout 把 one-step error 递归放大。Latent space prediction、uncertainty-aware planning、self-correcting objectives 都只是缓解。

**Sim-to-Real**：domain randomization、system identification、online adaptation。对 video-based world model 和 WAM 尤其严重，需要 physics-aware inductive biases、structured state、adaptive inference-time correction。

**Evaluation**：fragmented。CoW-Bench 提出 "Trinity of Consistency"：modal、spatial、temporal。Sub-second generative speed 不只是 compute metric，是 safety requirement——inference lag 直接导致 control instability。

**Safety**：triadic ecosystem——human-robot alignment（需要 theory-of-mind）、machine-environment loops（causal bi-directionality）、heterogeneous multi-agent（systemic resonance）。Federated World Models 解决 data aggregation 和 commercial confidentiality：只 exchange gradients/weights，用 TEEs 加密，decouple invariant physical commonsense（global）和 domain-specific latents（local）。

---

## Roadmap 三阶段

**Stage 1: Unified Multimodal World Models**。Long-horizon reasoning 需要 multimodal 信息，3D 等 modalities 在 data 稀少时 improve generalization，action+state signals 把 video prediction 从 passive forecasting 转成 embodied prediction。Practical path：unification 作 curriculum，先 static representation learning（JEPA），再 add video、state、action、embodied data。

**Stage 2: Unified Physical Representation**。当前 systems 维持三个 separate world definitions：appearance primitives for rendering、meshes/particles for simulation、occupancy grids/object slots for planning。互相 translate 是 lossy 和 ad hoc。Holy grail 是 **one state, many decoders**——单一 compressed physical state，所有 renderer/simulator/planner 都是它的 decoding operation。PhysGaussian 是 partial precedent。

**Stage 3: Foundation-Scale Interactive Simulators**。Scaling behavior 能否 emerge for physical dynamics？三个 aspect：scalable architectures（diffusion + autoregressive + unified multimodal）、scalable physical datasets（internet video + manipulation video + robot trajectories + richer physical supervision）、closed-loop verification（predictions validate against real outcomes）。

---

## Trinity Architecture：Physical AGI 的发动机

这是 paper 的 finale，也是最有意思的部分。

LLM 给了 machines 对 syntax、semantics、stored knowledge 的 mastery，但缺 physical reality 的 grounding。我们正在接近 disembodied intelligence 的极限。

Trinity Architecture 是三个 interdependent components 的 cognitive loop：

**Agent (Actor)**：执行引擎，把 high-level intent 翻译成 granular sequential actions。

**Evaluator (Critic)**：任务完成度的 judge，observe trajectory，assess effectiveness/efficiency，提供 failures、physics violations、suboptimal movements 的精确 feedback。

**World Model**：核心 simulator 和 curriculum designer。Ingest trajectory data，learn physics、dynamics、causality。**Crucially：understand Agent 当前能力的 exact edge**。通过 internal simulation，imagine 和 propose progressively complex tasks——just beyond Agent's current limits，作 automated curriculum generator。

这个 loop 跨 digital 和 physical world：World Model 在 digital twin 里 imagine thousands of scenarios，Agent master 后 deploy 到 physical world，Critic evaluate friction 和 noise，feed back 到 World Model。AI 不再依赖 human-curated datasets，开始像 biological intelligence 一样学习——through physical trial、error、imagination。

Actor 和 Critic 可以用 LLM-based multi-agent systems 实现，所以 planning、reasoning、decision-making 自然 depend on LLMs 的 cognitive intelligence。**Trinity Architecture 提供 LLMs 和 world model 的 mutual enhancement protocol。**

---

## 一句话 intuition

**World model 是物理世界的压缩器，不是生成器。从互联网视频里榨出物理先验，压缩成 unified internal state，所有 rendering/simulation/planning 都是它的 decoding 操作。WAM 把 prediction 和 action generation 耦合在一起。Trinity Architecture 让 World Model 知道 Agent 能力的边界，自动设计 just-beyond-edge 的 curriculum，形成跨 digital-physical 的 self-improving loop——这就是 Physical AGI 的路径。**

---

# A Definition and Roadmap for World Models — 深度解读

这篇 paper 是 Physical Intelligence Team 和 Shanghai AI Laboratory 写的 perspective article，核心目标是给 world model 下一个科学定义，并给出分阶段的 roadmap。这篇 paper 野心很大，它试图把当前 AI 各个 subfield 里被称作 "world model" 的东西统一到一个框架下，并指出物理 AI 的真正路径在哪里。

## 1. Motivation：为什么需要重新定义 World Model

Paper 一开始就指出，"world model" 这个词已经被用烂了——model-based RL、video generation、embodied robotics、physical AI 各个领域都在叫自己的系统 "world model"，但大家说的根本不是一回事。

历史上，Craik 在 1943 年提出生物体通过在脑中持有 "working models" 来生存，这是 "world model" 概念的最早起源 (https://en.wikipedia.org/wiki/Kenneth_Craik)。Sutton 的 Dyna architecture 把这个直觉翻译成了现代 RL (https://arxiv.org/abs/2104.06178)。

当前最主流的 disambiguation 尝试是 Fei-Fei Li 提出的 functional taxonomy，把 world model 分为三类：
- **Renderers**：生成 pixels
- **Simulators**：预测 state transitions  
- **Planners**：提议 actions

(https://www.worldlabs.ai/blog/taxonomy-of-world-models)

但 paper 指出这个 taxonomy 的根本局限：它在分类 decoding，而不是 representation 本身。一个 unified internal model 可以被 decoded 成 RGB pixels、state vectors 或 action proposals，取决于 query interface。所以 functional taxonomy 告诉我们 world model 输出什么，但没定义内部 model 是什么。

另一个对立观点是 LeCun 的 JEPA 路线 (https://arxiv.org/abs/2306.02572)，认为 generative reconstruction 永远不是正确的 objective，joint-embedding 架构在 latent space 预测就够了，pixel-level reconstruction 浪费 representational capacity。Paper 这里很谨慎：同意 pixel-level reconstruction 不应该是终极目标，但 reconstruction quality 是追踪 representation 丢了什么信息的好工具。

### 1.1 数据决定了天花板

Paper 提出一个关键 premise：**在固定 architecture 和 compute budget 下，是 data diversity 决定 ceiling，而不是 model structure 或 hyperparameters**。Architecture 和 compute 影响逼近 ceiling 的效率，但不能抬高它 (Hoffmann et al., 2022, https://arxiv.org/abs/2203.15556)。

这个论点直接指向了 Figure 1 里的 Inverted Pyramid Workflow。核心 insight 是：唯一能 scale 到所需 diversity 的 physical data source 是 open internet——图像、video、text 在百亿级别。Proprietary 数据（dexterous manipulation datasets、first-person embodied recordings、exoskeleton sensors）在 diversity 和 scale 上结构性地受限。

Internet video 是一个 unprecedented 的大规模 natural corpus，隐含编码了物理世界的 structured priors：
- Object permanence（物体不可见时的持续存在）
- Rigidity vs. non-rigidity 约束
- Kinematic limits
- Lighting/shadow dynamics  
- Occlusion/disocclusion 的因果逻辑
- 事件层次因果链
- Human-like action 的结构化先验

但所有这些 physical structure 在 raw pixel space 里都是 latent 不可访问的。所以 Figure 1 的 inverted pyramid funnel pipeline 的逻辑是：
1. **Top tier**: Web Data Sources——数十亿 raw pixel streams
2. **Middle layer**: Synthetic and Filtered Dataset——大幅剪裁、合成针对 robotic actions 的子集
3. **Bottom tip**: Real-World Task Data——高度 curated、任务优化的小数据集，用于 end-to-end robot fine-tuning

这个 funnel 的本质是：通过 internet 的广度提升 generalization ceiling，再通过 funnel 的精度蒸馏得到 task-aligned 的紧凑数据集。

### 1.2 World Model 作为 Compression Mechanism

这是 paper 的核心 framing：**World model 本质上是一个 compression mechanism，不是 generation 或 simulation 问题**。

高维 video observations 主要被 task-irrelevant 的 photometric variance 主导——viewpoint、illumination、texture、sensor noise、background clutter。Raw pixels 不显式编码 permanence、causality 或 constraint；它们只 encode 表面 appearance，不 encode 生成 visual signal 的 physical structure。

所以 world model 的核心功能是 **targeted information preservation**：保留 downstream physical reasoning 和 control 所需的 structured causal 和 physical information，系统性丢弃 irrelevant 的 photometric nuisance variance。

这是 paper 最锋利的论点：**generation 和 simulation 是好的 compressed representation 下游涌现的能力，但它们本身不是 objective**。

## 2. World Model 的正式定义

### 2.1 Definition 2.1

Paper 给出正式定义：

> A world model is a compression modeling of the state transition processes of the physical world, constructed under the constraints of finite computational resources.

关键词是：
- **Compression modeling**：不是预测，不是生成，是压缩
- **State transition processes**：建模的是 state 演化的过程
- **Finite computational resources**：资源约束是本质性的

这个定义隐含三个 major properties（见 Figure 2）：

**Omnimodal workscope**：必须能 model 所有 perceptual modality，不仅是 text 或 vision，而是 all-modal foundational model 的 unified latent representation。

**Multidimensional Asynchronicity**：由于计算和感知资源有限，物理世界各维度数据以不同频率采样。world model 必须能处理 multi-dimensional、asynchronous（multi-frequency）序列数据。这一点很重要——robot 的 proprioception 可能是 100Hz，vision 是 30Hz，language instruction 是稀疏的，world model 要把这些异步信号统一建模。

**Locality**：agent 的感知受限于资源，数据只覆盖局部区域，外部区域不断干预。所以从 local 视角建模世界通常被形式化为 POMDP (https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process)。

### 2.2 Agent-Environment Loop

Paper 把 world model 的形式化建立在 POMDP 上：

$$\mathfrak{M} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, P^\star, O, R, \gamma)$$

变量解释：
- $\mathcal{S}$: 真实 latent state 空间
- $\mathcal{A}$: action 空间
- $\mathcal{O}$: observation 空间
- $P^\star(\mathbf{s}_{t+1} \mid \mathbf{s}_t, \mathbf{a}_t)$: 真实 transition kernel
- $O(\mathbf{o}_t \mid \mathbf{s}_t)$: observation model
- $R$: reward
- $\gamma$: discount factor

由于真实 state 是 hidden 的，competent behavior 不能仅依赖 raw observations，需要 internal belief over possible states。所以 world model 不是简单的 predictor of pixels/tokens/trajectories，而是 **internal model of latent state, observation formation, action-conditioned dynamics**，让 agent 能：
- 推断 what is currently the case
- 想象 what could happen next  
- 在 uncertainty 下 evaluate intervention

这个 framing 连接了 Craik 1943 和 Sutton 1990/1991 的 Dyna architecture (https://arxiv.org/abs/2104.06178)。Xing et al. 2025 (https://arxiv.org/abs/2507.05169) 把这个观点磨砺得更尖锐：world model 的目标是模拟 **actionable possibilities**，不是 reproduce world exhaustively，也不只是 generate visually plausible futures。

### 2.3 Understanding vs. Predicting：两种 views

Ding et al. 2024 (https://arxiv.org/abs/2411.14499) 的 survey 指出 "world model" 在两种意义上被使用：

**Understanding-oriented**：world model 主要用于 understanding——把 sensory data 压缩成 stable internal representations，expose 实体、关系和机制。Prediction 是 training signal 或 consistency constraint，不是最终目标。早期 deep world-model 工作（Ha & Schmidhuber 2018, https://arxiv.org/abs/1803.10122）就是这个精神。

**Prediction-oriented**：world model 主要用于 prediction——judged by ability to roll world forward、generate candidate futures、support foresight。LeCun 的 autonomous intelligence program (https://arxiv.org/abs/2306.02572) 把 predictive world model 放在 reasoning 和 action 中心，Sora (https://openai.com/index/video-generation-models-as-world-simulators/) 让这个 predictive interpretation 在 observable futures 层面尤为显眼。

这两个 view 暗示不同的 failure modes：
- Understanding-oriented model 可能 infer compact decision-relevant state，但 produce visually poor renderings
- Video generator 可能 produce strikingly realistic futures，但作为 control-relevant simulator 不可靠，因为不 preserve hidden state、causal structure 或 intervention semantics

Paper 的立场：对于 physical world models，**understanding 应该是 primary，prediction 应该服务它**。一个不能识别 latent operating state、causal structure、intervention-relevant uncertainty 的模型可能 generate plausible futures，但作为 scientific reasoning 或 embodied control 的基础会失败。Predictive rollout 依然不可或缺，但主要作为通过 counterfactual simulation 测试和 refine internal understanding 的 mechanism。

## 3. 两种 Taxonomy

### 3.1 Functional Taxonomy：Renderer/Simulator/Planner

Bayesian inference 给这个 loop 应该怎么 manage uncertainty 一个 normative account。在收到当前 observation 之前，agent 通过把之前的 posterior 通过 transition dynamics 传播来形成 predictive prior：

$$\mathbf{x}_t(\mathbf{s}_t) \propto O(\mathbf{o}_t \mid \mathbf{s}_t) \sum_{s_{t-1}} T(\mathbf{s}_t \mid \mathbf{s}_{t-1}, \mathbf{a}_{t-1}) \mathbf{x}_{t-1}(\mathbf{s}_{t-1}) \quad (1)$$

变量解析：
- $\mathbf{x}_t$: 在 state $\mathbf{s}_t$ 上的 posterior belief（time $t$）
- $O(\mathbf{o}_t \mid \mathbf{s}_t)$: observation likelihood，给定 latent state $\mathbf{s}_t$ 看到 $\mathbf{o}_t$ 的概率
- $T(\mathbf{s}_t \mid \mathbf{s}_{t-1}, \mathbf{a}_{t-1})$: state-transition model
- 求和 $\sum_{s_{t-1}}$ 是对 previous state 的 marginalization，predictive prior 是 transition 把 previous posterior 推前一步
- 比例符号 $\propto$ 表示归一化常数被省略

这个 posterior 整合了累积知识和当前证据，同时显式保留不确定性，是 decision-making 的 informational basis。

重要：rational planning 应该对 full belief distribution optimize expected utility，不只是 most likely trajectory。Actions 既有 instrumental value（达成外部目标）又有 epistemic value（通过 informative observations 减少不确定性）。

### 3.2 Two-Dimensional Taxonomy

Paper 提出第二维 taxonomy——**按 representation substrate 和 predictive mechanism 分类**：

- **Observation-level generative models**：在 pixels/video 等 perceptual space 直接预测
- **Latent-space dynamics models**：encode 到 compact states，在 representation space 预测演化
- **3D/structured world representations**：显式建模 geometry、topology、objects、relations、physical attributes

Fei-Fei 的 taxonomy 问 "what does a system do"，paper 的 taxonomy 问 "how is the relevant world knowledge represented and computed"。它们 complementary。

Figure 5 把这些 systems mapping 出来：
- **Sora** 和 **Seedance** 在 observation-level renderer 区域——外部 prediction target 是 visually plausible videos
- **Genie 3** (https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) 在 observation-level，但向 3D world representation 延伸——frame-by-frame 响应 navigation commands 生成 environment
- **JEPA family** 在 latent-space simulator 区域——定义性 prediction target 是 abstract future representation
- **VLA-JEPA** (https://arxiv.org/abs/2602.10098) 把 latent predictive mechanism 向 planner 区域延伸
- **Dreamer** (https://arxiv.org/abs/2301.04104) 也在 latent-space 列，但遵循 generative state-space 范式
- **Marble** (https://www.worldlabs.ai/blog/marble-world-model) 主要在 3D/structured representation 区域
- **Cosmos 3** (https://arxiv.org/abs/2606.02800) 是 cross-cutting capability profile，不同 input-output 配置暴露不同功能

**World Action Model (WAM)** 不被当作第四个 implementation category，而是 cross-architectural functional paradigm。WAM 的 defining commitment 是 **predictive state modeling 与 action generation 的耦合**。

WAM 可以通过 explicit future video reasoning（visual planning pipelines）、compact latent future representations、或 structured spatial representations（optical flow、3D point flows、RGB-D trajectories）来 reason。它可以用 observation-level、latent-space 或 structured world models 来 instantiate。

WAM 是 functional family，centered on predictive action generation，避免把 output function 和 implementation substrate 混淆，允许比较 cascaded systems（先预测 future-state representation 再 derive actions）和 joint systems（co-model state 和 action trajectories）。

## 4. Architectural Paradigms

### 4.1 Observation-Level Generative World Models

把 world modeling 当作 high-dimensional observation synthesis，直接预测 future pixels、voxels、video tokens。

代表系统：Sora-style models (https://openai.com/index/video-generation-models-as-world-simulators/), Wan (https://arxiv.org/abs/2503.20314), Happy Horse, Seedance (https://arxiv.org/abs/2506.09113), Movie Gen (https://arxiv.org/abs/2410.13720), HunyuanVideo (https://arxiv.org/abs/2412.03603), CogVideoX (https://arxiv.org/abs/2408.06072), Kling (https://arxiv.org/abs/2512.16776)。

更 explicitly interactive 的：Genie series (https://arxiv.org/abs/2402.15391), GameNGen (https://arxiv.org/abs/2408.14837), DIAMOND (https://arxiv.org/abs/2405.12399), Oasis (https://oasis-model.github.io/)。

Driving 场景：Cosmos (https://arxiv.org/abs/2501.03575), GAIA-1 (https://arxiv.org/abs/2309.17080), DriveDreamer (https://arxiv.org/abs/2309.09777), Drive-WM (https://arxiv.org/abs/2311.17918), Waymo World Model (https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/)。

核心 advantage：scale——internet video 提供巨大 weakly supervised corpus。核心 limitation：visual plausibility ≠ physical correctness。Pixel-space models 可能 generate locally convincing 但 globally inconsistent rollouts，尤其在长 horizon 上——object permanence、contact dynamics、causality、scene consistency 的 errors 会累积。计算上也很昂贵，因为 prediction 在高维 observation space。

### 4.2 Latent-Space World Models

从 pixel-space 到 latent-space 的转变是根本性的 shift：从 modeling visual appearance 转向 modeling task-relevant state evolution。

早期 latent world models：PlaNet (https://arxiv.org/abs/1811.04551), Dreamer family (https://arxiv.org/abs/2301.04104) 使用 RSSM (Recurrent State-Space Model) encode 高维 observations 到 compact states，imagination-based planning。

TD-MPC 和 TD-MPC2 (https://arxiv.org/abs/2310.16828) 展示 planning 可以在 learned task-oriented latent spaces 直接进行，不需要 full pixel decoder。

Contrastive 和 reconstruction-free world models (https://arxiv.org/abs/1911.12247, https://arxiv.org/abs/2203.00494, https://arxiv.org/abs/2312.09056) 表明避免 pixel-level reconstruction 可以提升 robustness，减少对 high-frequency appearance nuisance factors 的过拟合。

JEPA-style (https://arxiv.org/abs/2301.08243, https://arxiv.org/abs/2506.09985) 进一步挑战 full visual reconstruction 的必要性——预测 representations 而非 pixels。

LDA-1B (https://arxiv.org/abs/2602.12215) 和 Motus (https://arxiv.org/abs/2512.13030) 展示 effective policies 可以从 compact task-oriented latent representations 学到，不需要 photorealistic reconstruction。

Main risk：excessive abstraction 可能丢弃 visually small 但 decision-critical 的信息——gripper-object contact、thin obstacles、subtle object pose changes、deformation。所以 central challenge 不是简单压缩 observations，而是学习 **丢弃 appearance-level noise 同时保留决定 future controllability 的 factors** 的 latents。

### 4.3 3D-Enhanced 和 Object-Centric World Models

传统 video model 可能 generate visually plausible future frames，但 internal state 难以解释，可能不对应实际 3D layout。对 embodied AI 来说，模型应该支持 free space、object locations、occlusion、viewpoint changes、object permanence、physical plausibility 的 reasoning。

Occupancy 和 BEV-based：OccWorld (https://arxiv.org/abs/2401.09590), OccSora (https://arxiv.org/abs/2405.20337), Drive-OccWorld, BEVWorld, RoboOccWorld。Key advantage：predictions 变得 spatially queryable。

NeRF (https://arxiv.org/abs/2003.08934) 和 3D Gaussian Splatting (https://arxiv.org/abs/2308.14737) 提供 geometry-aware states 支持 novel-view rendering。Marble, RenderWorld, GaussianWorld, GWM (https://arxiv.org/abs/2508.17600) 把这个 idea 延伸到 persistent 3D world generation。

Object-centric：OP3, C-SWM (https://arxiv.org/abs/1911.12247), Slot Attention (https://arxiv.org/abs/2006.15055), SAVi, SlotFormer (https://arxiv.org/abs/2210.05481) 学 object- 或 slot-level representations 和 dynamics，支持更 compositional 的预测和 reasoning。FOCUS (https://arxiv.org/abs/2307.02427), Object-Centric World Model for Language-Guided Manipulation (https://arxiv.org/abs/2503.06170) 连接 object-level abstraction 到 manipulation 和 instruction-conditioned prediction。

最有前景的方向是 **combine these representations**：Occupancy 支持 free-space 和 collision reasoning，NeRF/3D Gaussian 支持 view-consistent rendering，object slots 支持 compositional reasoning。Hybrid 系统 EnerVerse (https://arxiv.org/abs/2501.01895), EnerVerse-AC (https://arxiv.org/abs/2505.09723) 已指向这个方向。

### 4.4 Unification Trend：Omnimodal World Models

Generation-understanding unified models：LWM (https://arxiv.org/abs/2402.08268), WorldGPT (https://arxiv.org/abs/2404.18202), HERMES (https://arxiv.org/abs/2501.14729), GaussianDWM (https://arxiv.org/abs/2512.23180), UniDrive-WM。Adjacent：Chameleon (https://arxiv.org/abs/2405.09818), Janus (https://arxiv.org/abs/2410.13848), Emu series。

物理世界不仅被 observed，也被 acted upon。从 POMDP 视角，embodied agent 只接收 partial observations，actions 干预 latent state transition 并 shape future observations。所以 generation-understanding unification 本身不够。**World Action Models (WAMs)** (https://arxiv.org/abs/2605.12090) 整合 world modeling 与 action/policy modeling 在 shared generative framework 内。

代表：
- DreamZero (https://arxiv.org/abs/2602.15922)：jointly modeling video 和 action，把 world model 变成 zero-shot policy
- LingBot-VA：unifies visual dynamics prediction 和 action inference 在 autoregressive video-action framework
- τ0-WM (https://finch.agibot.com/research/tau0-wm)：integrates action generation、video prediction、future-state evaluation 用于 robotic manipulation
- UWM (https://arxiv.org/abs/2504.02792), Cosmos Policy (https://arxiv.org/abs/2601.16163), GigaWorld-Policy (https://arxiv.org/abs/2603.17240), Fast-WAM (https://arxiv.org/abs/2603.16666), Flash-WAM (https://arxiv.org/abs/2606.05254)

Omnimodal physical modeling：NVIDIA Cosmos 3 (https://arxiv.org/abs/2606.02800) 在 Mixture-of-Transformers architecture 内 unify language、image、video、audio、action。Seedance 2.0 (https://arxiv.org/abs/2604.14148), Wan/Wan2.1, HunyuanWorld (https://arxiv.org/abs/2507.21809), Kling-Omni, Qwen-Omni (https://arxiv.org/abs/2503.20215) 都是 building blocks。

## 5. Training 和 Learning Paradigms

### 5.1 Self-Supervised 和 Generative Pretraining

最直接的 formulation 是 video prediction：

$$p_\theta(\mathbf{x}_{t+1:t+K} \mid \mathbf{x}_{\leq t}, \mathbf{a}_{t:t+K-1}, \mathbf{c}) \quad (2)$$

变量：
- $\mathbf{x}_{t+1:t+K}$: 从 $t+1$ 到 $t+K$ 的 future observation 序列
- $\mathbf{x}_{\leq t}$: 截至时间 $t$ 的所有历史 observations
- $\mathbf{a}_{t:t+K-1}$: 从 $t$ 到 $t+K-1$ 的 action 序列
- $\mathbf{c}$: 可选 condition（language instruction、scene context）
- $p_\theta$: 参数 $\theta$ 的条件分布

Action-free prediction 学 observational dynamics；action-conditioned prediction 开始近似 interventional dynamics——这是对 planning 关键的区别。

Masked autoencoding (MAE, https://arxiv.org/abs/2111.06377, VideoMAE https://arxiv.org/abs/2203.12602) 提供 complementary route。但 reconstruction 只是 proxy——对 world models 来说，最好的 masked objectives 是那些 latent spaces 在 intervention 下仍然 predictive 的。

Next-token prediction 提供最 scalable 的 generative interface：

$$\mathcal{L}_{\mathrm{NTP}} = -\sum_{t=1}^{T} \log p_\theta(u_t \mid u_{<t}, \mathbf{a}_{\leq t}, \mathbf{c}) \quad (3)$$

变量：
- $u_t$: time $t$ 的 discrete token
- $u_{<t}$: 之前所有 tokens
- $\mathbf{a}_{\leq t}$: 截至时间 $t$ 的 action history
- $\mathbf{c}$: condition
- 负号表示最大化对数似然

Scaling law 抽象：

$$\mathcal{L}(N, D) \approx \mathcal{L}_\infty + A N^{-\alpha} + B D^{-\beta}, \quad C \approx \kappa N D \quad (4)$$

变量：
- $N$: model size（parameters）
- $D$: training data
- $C$: compute
- $\mathcal{L}_\infty$: irreducible loss（数据本身的 entropy）
- $A, B, \alpha, \beta$: 正的常数，控制参数和数据 scaling 的 power-law 指数
- $\kappa$: 把 $N$ 和 $D$ 映射到 compute 的常数

对 world models 来说，$D$ 应该被理解为 coverage over states、actions、embodiments、viewpoints、contacts、rare events、horizons，不只是 token count。

### 5.2 Model-Based Reinforcement Learning

MBRL 是 world models 第一次被给 decision-centric 定义的地方。Pipeline 分两阶段：
1. Agent 与 environment 交互，fit dynamics model（learned transition model $\widehat{P}_\theta$）
2. Agent 用这个 model 作为 cheap、differentiable、resettable 的 reality surrogate，通过 planning 或 imagined rollouts 训练 policy

形式化：MBRL 是 tuple $\mathfrak{M}_{\mathrm{MDP}} = (\mathcal{S}, \mathcal{A}, P^\star, R, \gamma, \rho_0)$，agent 寻找 policy $\pi_\phi: \mathcal{S} \to \Delta(\mathcal{A})$ 最大化：

$$J(\pi_\phi) = \mathbb{E}_{\pi_\phi, P^\star}\left[\sum_{t=0}^{H-1} \gamma^t r_t\right], \quad r_t = R(\mathbf{s}_t, \mathbf{a}_t) \quad (5)$$

变量：
- $\pi_\phi$: 参数 $\phi$ 的 policy
- $P^\star$: 真实 transition dynamics
- $H$: horizon
- $\gamma^t$: discount factor 的 $t$ 次方
- $r_t$: time $t$ 的 reward
- $R$: reward function

关键是学习 approximate transition function $\widehat{P}_\theta$:

$$\widehat{P}_\theta: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S}), \quad \widehat{\mathbf{s}}_{t+1} \sim \widehat{P}_\theta(\cdot \mid \mathbf{s}_t, \mathbf{a}_t) \quad (6)$$

$\widehat{P}_\theta$ 通过最小化负对数似然训练：

$$\mathcal{L}_{\mathrm{dyn}}(\theta) = -\mathbb{E}_{(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim \mathcal{D}}\left[\log \widehat{P}_\theta(\mathbf{s}' \mid \mathbf{s}, \mathbf{a})\right] \quad (7)$$

变量：
- $\mathcal{D}$: replay buffer $\{(\mathbf{s}_t, \mathbf{a}_t, \mathbf{s}_{t+1})\}$
- $\mathbf{s}$: 当前 state
- $\mathbf{a}$: action
- $\mathbf{s}'$: next state

Model learning 方向：
- PILCO (https://proceedings.mlr.press/v15/deisenroth11a.html): Gaussian Processes 捕捉 epistemic uncertainty
- Probabilistic ensembles (PETS, https://arxiv.org/abs/1805.00909)
- Latent-space world models (Dreamer, https://arxiv.org/abs/2301.04104)

Model exploitation 方向：
- Dyna-style (Sutton 1990, https://arxiv.org/abs/2104.06178): model 作为 replay buffer augmentor
- MBPO (https://arxiv.org/abs/1906.05343): short truncated rollouts
- MPC (PETS, MPPI, https://arxiv.org/abs/1706.05343)
- Model-based policy gradient (https://arxiv.org/abs/1510.09142)

两个 regimes（Sutton & Barto 2018, http://incompleteideas.net/book/RLbook2020.pdf）：
- **Background planning**: model 作 data generator，amortized policy at deployment
- **Decision-time planning**: 在 decision 时查询 model，forward search

Forward search 有两个传统：连续控制下的 trajectory optimization + MPC；离散 domain 下的 MCTS (MuZero, https://arxiv.org/abs/1911.08265)。

**三个 failure modes**：
1. **Compounding error**: autoregressive rollouts feed model 自己的预测，one-step inaccuracies 累积，超线性 horizon 增长
2. **Objective mismatch**: model 训练 maximize predictive likelihood，但 evaluate by policy return
3. **Optimism-pessimism bias**: planner 是自己 model 的 adversarial consumer，会系统发现和 exploit model 的 optimistic errors

MBRL 在 offline 设置下需要保守方法如 MOPO (https://arxiv.org/abs/2005.13239), MOReL (https://arxiv.org/abs/2005.05951)，减去 uncertainty penalty 或终止离开 data manifold 的 rollouts。

Curiosity-driven exploration (ICM, https://arxiv.org/abs/1705.05363), RND (https://arxiv.org/abs/1810.12894), VIME (https://arxiv.org/abs/1605.09674), Plan2Explore (https://arxiv.org/abs/2005.05953) 把 model error 或 uncertainty 转成 intrinsic reward。

### 5.3 Policy Learning Inside World Models

WAM 与 classical MBRL 的关键区别：传统 MBRL decouples model learning 和 policy optimization——先 fit $\hat{P}_\theta$，再 exploit。WAMs 在 single generative process 内 coupling dynamics prediction 和 action generation，jointly 学习。

Dreamer family 是 canonical example：学 RSSM，压缩 observation histories 到 stochastic latents，预测 rewards 和 continuation，在 candidate actions 下 unroll latent trajectories。Actor 和 critic 在这些 imagined trajectories 上训练。

VLA models (RT-2, https://arxiv.org/abs/2307.15818) 扩展 pretrained vision-language models，把 robot actions 表示成 output tokens。Strength 是 semantic transfer，但 physical dynamics 隐含在 action-token mapping 里。

WAMs 让这个 channel explicit。DreamZero (https://arxiv.org/abs/2602.15922) 在 pretrained video diffusion backbone 上 co-generate future video 和 robot actions，policy 成为 imagined world evolution 的 readout。

所以问题差异：VLA 问 "what action token should follow this instruction and observation?"；WAM 问 "what future should occur, and what action sequence realizes it?"

### 5.4 Chain-of-Imagination：Reasoning Through World Models

Chain-of-Imagination (CoI) 是 sharper transition。World model 不只是被 external planner 查询的 simulator，而是 **reasoning 本身展开其中的 computational medium**。在这个 view 里，thought 不必须是 sentence，可以是 learned dynamical space 中的 action-conditioned transition。

Coconut (https://arxiv.org/abs/2412.06769) 让 LLMs feed continuous hidden states 回去作为 "continuous thoughts"。对 embodied agents，analogous workspace 是 spatiotemporal belief state：

$$\mathbf{z}_t = E_\eta(\mathbf{o}_{\leq t}, \mathbf{a}_{<t}) \quad (8)$$

变量：
- $\mathbf{z}_t$: 时间 $t$ 的 belief state
- $E_\eta$: 参数 $\eta$ 的 encoder
- $\mathbf{o}_{\leq t}$: 截至时间 $t$ 的 observations
- $\mathbf{a}_{<t}$: 之前所有 actions

这个 belief summary 当前 scene、interaction context、uncertainty，在 policy commit action 之前。

CoI trace：

$$\mathcal{C}_t = \left\{\left[(\mathbf{a}_{b,1}, \widehat{\mathbf{z}}_{b,1|t}), \dots, (\mathbf{a}_{b,K_b}, \widehat{\mathbf{z}}_{b,K_b|t})\right]\right\}_{b=1}^{B_t}$$

$$\widehat{\mathbf{z}}_{b,k|t} = W_\theta(\widehat{\mathbf{z}}_{b,k-1|t}, \mathbf{a}_{b,k}), \quad \widehat{\mathbf{z}}_{b,0|t} = \mathbf{z}_t \quad (9)$$

变量：
- $B_t$: branch 数量
- $K_b$: branch $b$ 的 rollout depth
- $\mathbf{a}_{b,k}$: branch $b$ 第 $k$ 步的 action
- $\widehat{\mathbf{z}}_{b,k|t}$: 在 branch $b$ 第 $k$ 步、从 time $t$ 开始的 imagined latent state
- $W_\theta$: 参数 $\theta$ 的 world model 转移函数
- $\widehat{\mathbf{z}}_{b,0|t} = \mathbf{z}_t$: 每个 branch 的起点是当前 belief

最终决策：

$$\widehat{\mathbf{a}}_t = \pi_\phi(\mathbf{z}_t, \mathrm{Agg}_\psi(\mathcal{C}_t)) \quad (10)$$

变量：
- $\pi_\phi$: 参数 $\phi$ 的 policy
- $\mathrm{Agg}_\psi$: 参数 $\psi$ 的 aggregation function，把多 branch 的 imagined futures 聚合成 summary
- 最终 action 同时 conditioned on 当前 belief 和 aggregated counterfactuals

代表系统：
- MineDreamer (https://arxiv.org/abs/2403.12037): visual CoI，imagines stepwise goal images
- LCDrive (https://arxiv.org/abs/2512.10226): latent CoT interleaves action-proposal tokens 和 latent world-model prediction tokens
- FutureX (https://arxiv.org/abs/2512.11226): Auto-think Switch routes routine scenes 通过 instant policy，激活 latent world-model rollout 只用于 difficult scenes

Adaptive CoI 可以学：

$$(B_t, K_t) = \rho_\omega(\mathbf{z}_t) \quad (11)$$

变量：
- $B_t, K_t$: 分配的 branch 数和 depth
- $\rho_\omega$: 参数 $\omega$ 的 meta-controller，根据当前 belief 决定 reasoning budget

CoI 是 chain-of-thought reasoning 和 world-model imagination 的 convergence。Central risk：uncalibrated world model 让 reasoning 变成 structured hallucination。

### 5.5 Physics-Informed 和 Constrained Learning

三个 level 的 structural commitment：

**Penalty-based (soft) constraints**：

$$\mathcal{L} = \mathcal{L}_{\mathrm{pred}} + \lambda_{\mathrm{dyn}} \mathcal{L}_{\mathrm{dyn}} + \lambda_{\mathrm{cons}} \mathcal{L}_{\mathrm{cons}} + \lambda_{\mathrm{bc}} \mathcal{L}_{\mathrm{bc}} \quad (12)$$

变量：
- $\mathcal{L}_{\mathrm{pred}}$: observation 或 latent space 的 prediction error
- $\mathcal{L}_{\mathrm{dyn}}$: 已知运动方程违反的 penalty
- $\mathcal{L}_{\mathrm{cons}}$: conservation 或 invariance properties 的 enforcement
- $\mathcal{L}_{\mathrm{bc}}$: boundary、contact、feasibility constraints
- $\lambda_{\mathrm{dyn}}, \lambda_{\mathrm{cons}}, \lambda_{\mathrm{bc}}$: 各自的 weighting multipliers

PINNs (https://www.sciencedirect.com/science/article/pii/S0021999118307125) 是这个思路的起源。

**Architecture-based (hard) constraints**：把 constraint built into architecture，物理律 by construction 满足。Hamiltonian Neural Networks (https://arxiv.org/abs/1906.01563), Lagrangian Neural Networks (https://arxiv.org/abs/2003.04630), Symplectic ODE-Net (https://openreview.net/forum?id=ryxmb1rKDS), Graph-based simulators (https://arxiv.org/abs/2002.09405)。

**Hybrid physics-learning schemes**：explicit differentiable physics engine 与 learned neural components 在 dual-pathway 设计中耦合。DiffTaichi (https://arxiv.org/abs/1910.00797), Brax (https://arxiv.org/abs/2106.13281)。Analytic physics 供应 coarse causal scaffold，neural components 补偿 model mismatch。

### 5.6 Counterfactual Reasoning

Counterfactual reasoning 把 world model 从 possible futures 的 simulator 转成 causal attribution 的 instrument。Key 不是 generate 另一个 plausible rollout，而是 **replay 同一个 world，surgically 只改被分析的 decision**。

形式化用 Structural Causal Model (SCM)：

$$\mathfrak{M}_{\mathrm{SCM}} = (\mathfrak{U}, \mathfrak{V}, \mathcal{F}, P_\mathcal{U}) \quad (13)$$

变量：
- $\mathfrak{U}$: exogenous background variables 空间
- $\mathfrak{V}$: endogenous variables 在 model 内
- $\mathcal{F}$: 决定 variables 如何互相影响的 mechanisms
- $P_\mathcal{U}$: latent circumstances 上的分布

Counterfactual query：

$$\mathbb{P}_{\mathcal{M}_{\mathrm{SCM}}}(Y_x \mid E = e) = \int \mathbb{P}_{(\mathcal{M}_{\mathrm{SCM}})_x}(Y \mid U = u) p_{\mathcal{M}_{\mathrm{SCM}}}(u \mid E = e) du \quad (14)$$

变量：
- $Y_x(u)$: 在 background $u$ 下，把 $X$ 的 mechanism 用 intervention $X=x$ 替换后 $Y$ 的值
- $E=e$: factual evidence（观察到的事实）
- $(\mathcal{M}_{\mathrm{SCM}})_x$: 被 intervention 修改后的 model
- $p_{\mathcal{M}_{\mathrm{SCM}}}(u \mid E=e)$: 给定 factual evidence 后 latent circumstances 的 posterior

这是 abduction-action-prediction 三步骤：
1. **Abduction**: 从 observed trajectory 推断 latent situation
2. **Action**: surgical edit，替换 factual action
3. **Prediction**: 在 abducted circumstances 下 roll forward

关键不等式：

$$\mathbb{P}(Y_{a'} \mid E = e) \neq \mathbb{P}(Y \mid A_t = \mathbf{a}', E = e) \quad (15)$$

右边比较 other episodes where $a'$ happened to be taken，可能涉及不同 states、goals、noise、opponents、environments。左边保持 factual world fixed，只改 decision。

Counterfactual contrast：

$$\Delta_Y(\mathbf{a}', \mathbf{a} \mid E = e) = \mathbb{E}_{\mathcal{M}_{\mathrm{SCM}}}\left[Y_{\mathbf{a}'} - Y_{\mathbf{a}} \mid E = e\right] \quad (16)$$

变量：difference 归因于 decision，因为 latent circumstances 被保持固定。这种对比对 causal credit assignment、regret analysis、policy debugging、off-policy explanation、safety evaluation 是核心。

Counterfactuals 从 observational accuracy 单独一般 **non-identifiable**。多个 structural models 可以在 observed distribution 上 agree，甚至 interventional distributions 上 agree，但在 counterfactual quantities 如 $\Upsilon_x$ 上 disagree，因为它们 encode 不同的 factual 和 hypothetical world 之间的 couplings (https://arxiv.org/abs/2301.09031)。

### 5.7 Long-Horizon 和 Hierarchical Planning

World models 使长 horizon planning tractable 不通过 rolling 更远，而是通过改变 planning 的 unit。Flat planner over primitive actions 面临 exponential branching、weak credit assignment、optimizer-amplified model error。

Under partial observability，geometry carried by recurrent belief 和 imagined through compact latent dynamics：

$$b_t = \mathcal{B}_\theta(b_{t-1}, \mathbf{a}_{t-1}, \mathbf{o}_t), \quad \widehat{\mathbf{z}}_{t+1|t} \sim \widehat{p}_\theta(\cdot \mid \mathbf{z}_t, \mathbf{a}_t) \quad (17)$$

变量：
- $b_t$: belief state at time $t$
- $\mathcal{B}_\theta$: 参数 $\theta$ 的 belief update function
- $b_{t-1}$: 之前 belief
- $\mathbf{a}_{t-1}$: 上一个 action
- $\mathbf{o}_t$: 当前 observation
- $\widehat{\mathbf{z}}_{t+1|t}$: 给定 $\mathbf{z}_t$ 和 $\mathbf{a}_t$ 的 imagined next latent state
- $\widehat{p}_\theta$: learned dynamics

Hierarchical planning：

$$\mathbf{g}_k \sim \pi_{\mathrm{hi}}(\cdot \mid b_{t_k}, \mathbf{m}_{t_k}), \quad \mathbf{a}_t \sim \pi_{\mathrm{lo}}(\cdot \mid b_t, \mathbf{g}_k), \quad t_k \leq t < t_{k+1} \quad (18)$$

变量：
- $\mathbf{g}_k$: 第 $k$ 个 subgoal
- $\pi_{\mathrm{hi}}$: high-level policy
- $b_{t_k}$: 在 subgoal 时刻 $t_k$ 的 belief
- $\mathbf{m}_{t_k}$: 可选 mission/memory signal
- $\pi_{\mathrm{lo}}$: low-level policy
- $t_k \leq t < t_{k+1}$: low-level policy active 的时间区间

Gain 不是 short rollout，而是 **lower-entropy search problem**：$H$ primitive actions 被 $K \ll H$ abstract commitments 替换。Director (https://arxiv.org/abs/2206.04114), THICK (https://openreview.net/forum?id=TjCDNssXKU) 实现这个 idea。

Horizon-limited value gap：

$$\Delta_H(\pi_\phi; b_t) = \left|V_{\widehat{p}_\theta, H}^{\pi_\phi}(b_t) - V_{P^\star, H}^{\pi_\phi}(b_t)\right| \quad (19)$$

变量：
- $V_{\widehat{p}_\theta, H}^{\pi_\phi}(b_t)$: 在 learned model $\widehat{p}_\theta$ 下，policy $\pi_\phi$ 从 belief $b_t$ 出发 horizon $H$ 的 value
- $V_{P^\star, H}^{\pi_\phi}(b_t)$: 在 true dynamics $P^\star$ 下同样的 value
- 差的绝对值衡量 model-induced value 与真实 value 的 gap

这个 gap 当 rollouts 离开 data-supported 区域时可能增长。

## 6. Application Domains

### 6.1 Robotics 和 Embodied AI

World models 在 robotic systems 中有多个 complementary roles：
1. **Predictive simulators** 用于 policy evaluation 和 planning
2. **Representation learners** 从 large-scale observations 捕捉 task-relevant dynamics
3. **Data engines** 生成 synthetic experiences 给 downstream policy training

按 computation budget 分三类：

**Data Engines**（offline regime）：RoboDreamer (https://arxiv.org/abs/2402.10809), UniSim (https://arxiv.org/abs/2310.08766)。Decoupled from real-time control，synthesize 大量数据。

**Environment Simulators**（evaluation regime）：MILE (https://arxiv.org/abs/2210.07637), ReSim (https://arxiv.org/abs/2506.09981), SIMPLER (https://arxiv.org/abs/2406.00700)。Query 评分或排序 candidate policies。

**Action Planners**（online control loop）：DayDreamer (https://arxiv.org/abs/2206.14176)。Query 低延迟，支持 receding-horizon planning。

**Embodiment World Models**：Behavior Foundation Model (BFM, https://arxiv.org/abs/2509.13780) 反向 modeling——预测 agent 自己 body 能产生什么行为。Modeling 自己 morphology-constrained skill manifold，analogous to internal body schema。Paper 强调 BFM 和 world models 是 complementary。

**Driving Simulators**：GAIA-1 (https://arxiv.org/abs/2309.17080), DriveDreamer (https://arxiv.org/abs/2309.09777), Vista (https://arxiv.org/abs/2405.05523)。Scene prediction 和 trajectory forecasting 是同一个 predictive substrate 的两个 temporally coupled instantiations。

### 6.2 Scientific Discovery

World models 是 AI4S 的 foundational paradigm。两个 complementary roles：

1. **Autonomous experimental systems**：与 AI agents、robotic instruments、experimental feedback 集成，支持 AI-driven autonomous laboratories (https://www.nature.com/articles/s41586-023-06734-w)。

2. **Predictive foundations for scientific phenomena**：scientific state + intervention/condition → future state/response。

代表：
- 气候：GraphCast (https://www.science.org/doi/10.1126/science.adi2336), GenCast (https://www.nature.com/articles/s41586-024-08252-9), FengWu, FengWu-W2S (https://arxiv.org/abs/2411.10191)
- 医学：Medical World Model (https://arxiv.org/abs/2506.02327), Delphi-2M (https://www.nature.com/articles/s41586-025-09252-0)
- 分子：MDGen (https://arxiv.org/abs/2409.17808)

## 7. Open Challenges 和 Bottlenecks

### 7.1 Data Asymmetry

在 renderer/simulator/planner taxonomy 下，world models 面临清晰 data asymmetry：
- **Renderers**: scale 在 internet-scale image/video data 上
- **Simulators**: bottleneck 是 simulation-ready 3D assets（PhysX-Anything https://arxiv.org/abs/2503.12790, PhysX-Omni, Lightwheel, Genesis World）
- **Planners**: bottleneck 是 action-conditioned interaction data

Embodied AI data pyramid：
- Base: abundant passive video
- Middle: 较少的 simulation-ready assets 和 interaction trajectories  
- Top: 稀少但高价值的 tactile 和 force feedback

未来方向：**ubiquitous, non-intrusive, continuous** data acquisition。

### 7.2 Fidelity vs. Precision

关键挑战：perceptual fidelity 和 physical precision 之间的 mismatch。PhyGenBench (https://arxiv.org/abs/2410.05363), PhyWorldBench (https://arxiv.org/abs/2507.13428), WorldModelBench (https://arxiv.org/abs/2502.20694), WorldSimBench (https://arxiv.org/abs/2410.18072) 等 benchmark 提供经验证据。

解决方案：fidelity-first generation → physical-precision-centered world modeling。Pretraining 需要 scalable physics-rich supervision，post-training 需要 physical rewards 和 evaluations，closed-loop feedback from real environment。

### 7.3 Compounding Prediction Errors

Long-horizon deployment 即使 model 局部 accurate，仍可能 fail——small one-step errors 被递归 feed 回 future predictions。这是 MBRL 和 video prediction 共享的 failure mode (Talvitie 2017, https://arxiv.org/abs/1612.06018, Lambert et al. 2022, https://arxiv.org/abs/2203.09637)。

缓解策略：
1. Compact latent state spaces（PlaNet, Dreamer）
2. Uncertainty-aware planning（probabilistic ensembles, conservative rollout horizons）
3. Self-correcting objectives, hierarchical abstraction, multiscale planning

### 7.4 Sim-to-Real Transfer

Domain randomization (https://arxiv.org/abs/1703.06907), system identification (https://arxiv.org/abs/1803.11347), online adaptation。

Video-based world models 和 WAMs 在 high-dimensional pixel space 操作，主要依赖 learned visual dynamics priors。需要更好的 physics-aware inductive biases、structured state representations、adaptive inference-time correction mechanisms。

### 7.5 Evaluation 和 Benchmarks

评估 landscape fragmented。各类 metrics：
- Predictive/generative：MSE, FID, FVD
- Control/planning：reward, episode return, success rate
- Generalization：OOD / zero-shot settings
- Safety-critical：uncertainty calibration

Benchmarks：VBench (https://arxiv.org/abs/2405.16735), VBench-2.0 (https://arxiv.org/abs/2503.21755), WorldScore (https://arxiv.org/abs/2411.00772), 4DWorldBench (https://arxiv.org/abs/2503.21755), PhyGenBench, WorldModelBench, PhyWorldBench, WorldSimBench, Atari/ALE (https://arxiv.org/abs/1207.4708), DM Control (https://arxiv.org/abs/1801.00690), Habitat (https://arxiv.org/abs/1904.01201), CARLA (https://arxiv.org/abs/1711.03938), RoboArena (https://arxiv.org/abs/2506.18123), EWMBench (https://arxiv.org/abs/2505.09694), WorldArena (https://arxiv.org/abs/2602.08971), WBench (https://arxiv.org/abs/2605.25874), WorldMark (https://arxiv.org/abs/2604.21686), MBench (https://arxiv.org/abs/2606.00793), WorldPrediction (https://arxiv.org/abs/2506.04363), WorldReasonBench (https://arxiv.org/abs/2605.10434), CoW-Bench (https://arxiv.org/abs/2602.23152)。

CoW-Bench 提出 "Trinity of Consistency"：modal、spatial、temporal consistency。

Inference efficiency 经常被 overlooking，但 sub-second generative speed 是 safety requirement。

### 7.6 Safety, Transparency, Sustainability

**Safe Exploration and Triadic Interaction Safety**：world models 在 deployment 前提供 learned internal simulators。但 sim-to-real gap 永远无法完全闭合。Triadic ecosystem：
1. Human-robot alignment：game-theoretic cognitive alignment，需要 theory-of-mind
2. Machine-environment loops：causal bi-directionality
3. Heterogeneous multi-agent synchronization：systemic resonance risk

**Transparency and Verifiable Control**：必须 integrate with control theory 或 formal verification frameworks。Automation bias (https://humanfactors.mit.edu/wp-content/uploads/2023/08/02_Parasuraman.pdf) 是 cognitive vulnerability。

**Ethical Deployment 和 Privacy-Preserving Governance**：Federated World Models——localized data silos untouched，只 exchange latent gradients 或 weights。配合 secure multi-party computation 或 hardware-enforced TEEs。Decouple invariant physical commonsense（globally shared）从 domain-specific latent representations（local retained）。

**Sustainability**：必须 incorporate life-cycle management——material extraction、manufacturing、end-of-life disposal (https://www.cell.com/patterns/fulltext/S2666-3899(25)00162-1)。Embodied carbon vs. operational carbon 的 trade-off。

## 8. Roadmap（三阶段）

### Stage 1: Towards Unified Multimodal World Models

Unified multimodality 关键因为：
1. Long-horizon reasoning 需要 multimodal 信息——video diffusion 单独不够，需要 search、policy、value signal
2. 3D 等 modalities 是 helpful complements，data 稀少时 improve generalization
3. Action 和 state signals 把 video prediction 从 passive forecasting 转成 embodied prediction

Practical path：unification 作 curriculum 而非 single pooled token stream。Training 开始 static representation learning (JEPA)，然后 add video, state, action, embodied data，用 compact latent actions 分离 controllable change 和 background appearance。

### Stage 2: Towards a Unified Physical Representation

核心 question：**什么 single internal state 可以让所有 decoder 都 decode from**？

当前 systems 维持三个 separate world definitions：
- Appearance-centric primitives for rendering（radiance fields, Gaussian primitives）
- Meshes/particles for simulation  
- Occupancy grids/object slots for planning

Translate among them 是 lossy 和 ad hoc。最有前景方向是 shared physical representation，每个 decoder 都是其 decoding operation。

Properties required：
1. Physically grounded and simulation-ready
2. Geometry-adaptive（不 impose regular grid 或 fixed template）
3. Intrinsically compact（slowly varying, decision-relevant physical factors）

"one state, many decoders" 原则已有 partial precedents：PhysGaussian (https://arxiv.org/abs/2404.05472) drives both physical simulation 和 rendering 从 single set of Gaussian kernels。

### Stage 3: Foundation-Scale Interactive Simulators

类似 LLM 和 video generation models 的 scaling behavior 是否能 emerge for physical dynamics？三个 aspects：
1. **Scalable architectures**：diffusion/flow-matching 提供 visual futures quality；autoregressive/LLM-style 提供 long-context sequence modeling；Cosmos 3, Bernini 提供 unified multimodal frameworks
2. **Scalable physical datasets**：internet videos + first-person manipulation videos + real robot trajectories + richer physical supervision（3D states, contact events, force, tactile, thermal, material）
3. **Closed-loop verification**：predictions 必须 validate against real-world outcomes 或 reliable physical constraints

## 9. Outlook：Physical AGI 路径

LLMs 给了 machines 对人类 syntax、semantics、stored knowledge 的 mastery。但缺乏 physical reality 的 fundamental grounding。我们正在接近 disembodied intelligence 的极限 (https://arxiv.org/abs/2507.19703)。

**The Trinity Architecture**：dynamic、self-evolving system 的三部分 cognitive loop：

**Agent (Actor)**: 任务执行引擎，把 high-level intent 翻译成 granular sequential actions。

**Evaluator (Critic)**: 任务完成度的 judge，observes trajectory data，assess effectiveness 和 efficiency，提供 failures、physics violations、suboptimal movements 的精确 feedback。

**World Model**: 核心 simulator 和 curriculum designer。Ingests trajectory data，learns underlying physics、dynamics、causality。Crucially，**understands the exact edge of Agent's current capabilities**。通过 internal simulation，imagines 和 proposes progressively complex tasks——just beyond Agent's current limits，作 automated curriculum generator。

World Model 在 Trinity Architecture 中三个 roles：
1. Internalize knowledge of the world
2. Know the edge of feasible tasks for current Actor
3. Guide next-round agent-world interaction by proposing new tasks

Actor 和 Critic 可以用 LLM-based multi-agent systems 实现，所以 planning、reasoning、decision-making 自然 depend on LLMs 的 cognitive intelligence。Trinity Architecture 提供 LLMs 和 world model 的 mutual enhancement protocol。

这个 loop 跨 digital simulations 和 physical world：World Model 在 digital twin 里 imagine thousands of scenarios，Agent master 后 deploy 到 physical world，Critic evaluate friction 和 noise，feed back 到 World Model。AI 不再依赖 human-curated datasets，开始像 biological intelligence 一样学习——through physical trial、error、imagination。

---

## 关键 Intuition 总结

1. **World model = compression mechanism**，不是 generator。Generation 和 simulation 是好 representation 的 emergent downstream 能力，本身不是 objective。

2. **Data diversity 决定 ceiling，不是 architecture**。在 fixed compute 下，internet video 的 breadth 提升泛化天花板，funnel 的精度蒸馏给 task-aligned 紧凑数据集。

3. **Understanding 应该 primary，prediction 应该 service 它**。能识别 latent state、causal structure、intervention-relevant uncertainty 的模型，比单纯 predictive rollout 但不能 actionable 的视频生成器更可靠。

4. **Functional taxonomy (what) + Representational taxonomy (how) 二维互补**。WAM 是 cross-architectural functional paradigm，centered on predictive action generation，可在 multiple architecture 上 instantiate。

5. **从 MBRL 到 WAM 的 architectural shift**：decoupled model learning + policy optimization → joint dynamics prediction 和 action generation。WAMs 从 video-action pretraining 继承 broad physical priors，支持 zero-shot task generalization 和 few-shot embodiment adaptation，但 cost 是 modularity 和 calibrated uncertainty。

6. **CoI = chain-of-thought + world-model imagination 的 convergence**。Thought 可以是 dynamical space 中的 action-conditioned transition，不是 sentence。Reasoning budget 应该 adaptive，由 risk 和 uncertainty govern。

7. **Counterfactual 的 same-world constraint 是关键**：latent circumstances 必须保持 fixed，只修改 intervened mechanism。否则模型没有 isolate decision 的 effect，而是改变了 world 本身。

8. **Long-horizon planning 通过改变 planning unit 而非 rolling 更远**。Hierarchy 把 $H$ primitive actions 替换成 $K \ll H$ abstract commitments，gain 是 lower-entropy search problem。

9. **三个 failure modes**：compounding error、objective mismatch、optimism-pessimism bias。MBRL 在 offline 需要 conservative methods（MOPO, MOReL）。

10. **Trinity Architecture 是 Physical AGI 的 evolutionary engine**：Agent 执行 → Evaluator 评估 → World Model 学习并提出 just-beyond-edge tasks，形成 self-improving loop，跨 digital 和 physical world。

11. **Unified physical representation 是 holy grail**：单一 compressed internal state，所有 renderer/simulator/planner 都是它的 decoding operation。PhysGaussian 是 partial precedent。

12. **Foundation-scale interactive simulators 是 ultimate stage**：scaling laws 在 physical dynamics 上能否 emerge 是核心 open question，需要 scalable architectures、scalable physical datasets、closed-loop verification 三方面协同。

这篇 paper 的真正贡献是把 world model 从 "video generator" 或 "RL dynamics model" 这种碎片化理解中拉出来，放在 POMDP + compression + actionable understanding 的统一框架下，并通过 Trinity Architecture 给出 Physical AGI 的具体工程路径。它把 LeCun 的 JEPA 立场、Fei-Fei 的 functional taxonomy、Dreamer 的 latent imagination、WAM 的 predictive action coupling 都整合到一个 layered view 里，并通过 inverted pyramid data workflow、unified physical representation、foundation-scale interactive simulators 三阶段 roadmap 给出从当前到 Physical AGI 的实现路径。
