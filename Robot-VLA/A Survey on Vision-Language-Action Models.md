---
source_pdf: A Survey on Vision-Language-Action Models.pdf
paper_sha256: d305da59bb076f5f3a12833ef342b4b76aa724b1197824059a3bc6cf9abb9348
processed_at: '2026-07-17T22:03:01-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# A Survey on Vision-Language-Action Models for Embodied AI - 深度讲解

## 一、论文核心 motivation 与定义

这篇 paper 是第一篇系统综述 **Vision-Language-Action Models (VLAs)** 的工作，由 CUHK 的 Yueen Ma 等人完成。作者提出了一个 generalized 的 VLA 定义：

> A VLA is any model capable of processing multimodal inputs from vision and language to produce robot actions that accomplish embodied tasks.

这与 RT-2 原始的"把 VLM 适配到机器人任务"的狭义定义不同。作者把 RT-2 这类基于 LLM/large VLM 的 VLA 称为 **Large VLAs (LVLAs)**，类似于 LLM 与 general language model 的区分。

**Intuition**: VLA 的本质是三个 modality（vision $s_t$、language $p$、action $a_t$）的 fusion，对应一个 language-conditioned policy：

$$\pi(a_t | p, s_{\leq t}, a_{<t})$$

其中 $p$ 是 language instruction，$s_{\leq t}$ 表示到时刻 $t$ 为止的所有 state observation（含历史），$a_{<t}$ 是过去的 action 历史。这比传统 RL policy $\pi(a_t|s_t)$ 多了两个关键 conditioning：language（任务语义）+ history（时序）。

Embodied AI 与 conversational AI 的本质区别就在 action modality —— 输出要作用到 physical world，所以 safety、latency、generalization 都被放大了。

参考: https://github.com/yueen-ma/Awesome-VLA

---

## 二、Taxonomy：三大研究主线

作者把 VLA 研究组织成三条线：

1. **Components of VLA** (§III-A)：PVRs、dynamics learning、world models、reasoning
2. **Low-level Control Policies** (§III-B)：直接预测 low-level action（translation、rotation、end-effector pose）
3. **High-level Task Planners** (§IV)：把 long-horizon task 分解为 subtask 序列

**Intuition**: 这是一个 hierarchical 视角。High-level planner 产出 discrete subtask $[p_1, p_2, \ldots, p_N] \sim \pi_\phi(\ell, s_t)$，low-level policy 在每个 subtask $p_i$ 下执行 continuous action $\hat{a}_t \sim \pi_\theta(\hat{a}_t | p, s_{\leq t}, a_{<t})$。这种分层结构类似 human cognition 中的 System 2（慢思考、规划）+ System 1（快反射、运动控制），后续 GR00T N1、NORA-1.5 都沿用了这一 dual-system 思想。

---

## 三、Components of VLA - 细节技术拆解

### 1. Pretrained Visual Representations (PVRs)

PVR 决定了 VLA 对 $s_t$ 的理解质量。Table I 比较了 9 种 PVR。核心 objective 可归为三类：

**(a) Vision-Language Contrastive Learning (VL-CL)** — CLIP 范式

$$\mathcal{L}_{\text{CLIP}} = \sum_{i=1}^{N} -\log \frac{\exp(S(x_i, y_i)/\tau)}{\sum_{j=1}^{N} \exp(S(x_i, y_j)/\tau)}$$

变量解释：
- $x_i$：第 $i$ 个 image
- $y_i$：与 $x_i$ 配对的 text（positive pair）
- $y_j$：batch 内第 $j$ 个 text（包含 negative）
- $S(\cdot, \cdot)$：cosine similarity
- $\tau$：temperature（论文中省略）
- $N$：batch size

**Intuition**: 这是 InfoNCE loss。分子拉近正对，分母推开所有负对。CLIP 在 400M image-text pair（WIT dataset）上预训练，所以学到的是 image-level 的语义对齐，但缺乏 pixel-level 精度 —— 对需要精细 grasping 的任务不够。CLIPort、EmbCLIP、CoW 都用 CLIP。

**(b) Time Contrastive Learning** — R3M、VIP

$$\mathcal{L}_{\text{R3M}} = \sum_{i=1}^{T} -\log \frac{\exp(S(x_i, x_j))}{\exp(S(x_i, x_j)) + \exp(S(x_i, x_k))}, \quad i < j < k$$

变量：$x_i, x_j, x_k$ 是同一 video 中时序上 $i<j<k$ 的三帧。$j$ 是 $i$ 的近邻（positive），$k$ 是远邻（negative）。

**Intuition**: 时间上相邻的 frame 表示同一 action 的不同阶段，应该 feature 相近；时间上远的 frame 表示不同 context，应该 feature 推开。这学到的是 video 的时序连续性，对 robot trajectory 很有用。VIP 把 anchor 放在 $x_0$（initial frame），抓长程依赖。

**(c) Masked Autoencoder (MAE)** — MVP、Voltron、RPT

$$\mathcal{L}_{\text{MAE}} = \sum_{i=1}^{N} -\log P(x_i | x_{\neq i})$$

其中 $x_i$ 是 masked patch，$x_{\neq i}$ 是 unmasked 的其他 patch。MVP 在 robot data 上做 MAE，学到 pixel-level 细节（分割、位置、深度），对 manipulation 任务更精准。

Voltron 加入 language conditioning：

$$\mathcal{L}_{\text{Voltron}} = \sum_{i=1}^{N} -\log P(x_i | x_{\neq i}, y)$$

$y$ 是 language instruction，让 reconstruction 受语言引导，增强 VL alignment。

RPT 进一步加入 action 和 proprioceptive state $z$：

$$\mathcal{L}_{\text{RPT}} = \sum_{i=1}^{N} -\log P(x_i | x_{\neq i}, y, z, \dotsc)$$

**Intuition**: MAE 类方法在 representation space 学 low-level 特征比 view-invariance 方法（如 DINO）更好；VC-1 实验证明 pixel-level 信息对 manipulation 至关重要。

**(d) Self-distillation** — DINOv2、I-JEPA

DINOv2：

$$\mathcal{L}_{\text{DINOv2}} = \sum_x \sum_{x' \neq x} H(P_t(x), P_s(x'))$$

$P_t, P_s$ 分别是 teacher/student network 的 output distribution，$H$ 是 cross-entropy，$x, x'$ 是同一 image 的不同 view。Teacher 是 student 的 EMA。

I-JEPA：

$$\mathcal{L}_{\text{I-JEPA}} = \frac{1}{M} \sum_{i=1}^{M} \sum_j \|x_j^{(i)} - y_j^{(i)}\|_2^2$$

$x_j^{(i)}$：第 $j$ 个 masked patch 的 embedding（student）；$y_j^{(i)}$：unmasked patch 的 embedding（teacher）。

**Intuition**: I-JEPA 不生成像素，在 representation space 预测，所以捕获 low-level 特征比 DINO 更好。Theia 通过 distillation 把多个 vision foundation model（segmentation、depth、semantics）融合进一个 model，用更少数据更小 size 超越单独的 PVR。

参考: 
- CLIP: https://arxiv.org/abs/2103.00020
- R3M: https://arxiv.org/abs/2203.12601
- DINOv2: https://arxiv.org/abs/2304.07193
- Theia: https://proceedings.mlr.press/v270/shang24a.html

### 2. Dynamics Learning

Dynamics learning 给 VLA 注入对环境 transition 的理解。核心公式（论文 Eq. 1）：

$$\text{Forward dynamics: } \hat{s}_{t+1} \gets f_{\text{fwd}}(s_t, a_t)$$
$$\text{Inverse dynamics: } \hat{a}_t \gets f_{\text{inv}}(s_t, s_{t+1})$$

$f(\cdot)$ 是 dynamic model。Forward 是预测下一步 state，inverse 是已知前后 state 反推 action。

**Table II 关键方法对比**:

- **Vi-PRoM**: $\sum_{i=1}^{T} \text{CE}(i, f(x_i | x'))$ —— 给 shuffled frame sequence $x'$，让 model 预测每帧在原 sequence 中的 index $i$。这是 temporal ordering task。
- **MIDAS**: $\sum_{i=1}^{T} \text{MSE}(a_t, f_{\text{inv}}(s_t, s_{t+1}))$ —— inverse dynamics，从前后帧反推 action。对 unlabeled internet video 极有价值，因为可以自动生成 action label。
- **SMART**: 同时做 forward（$\text{MSE}(s_{t+1}, f_{\text{fwd}}(s_t, a_t))$）+ inverse + hindsight control（mask 部分 action 让 model 恢复）。Forward 难度大于 inverse，但收益也更大。
- **VPT**: 用少量 labeled data 训 inverse model → 自动 label 互联网 Minecraft video → 用 BC 训 foundation model。这是 semi-supervised imitation learning 的经典范式，在 Minecraft 上达到 human-level。
- **GR-1**: video prediction pretraining for GPT-style model —— 预测未来 frame 等价于 forward dynamics，让 action prediction 更准。

**Intuition**: Forward dynamics 比 inverse 难（预测未来比回溯过去难），但学到的 representation 更具 predictive 能力。Inverse dynamics 的杀手级应用是给海量无标注 video 自动打 action label，是 data scaling 的关键。

参考:
- VPT: https://arxiv.org/abs/2206.07894
- GR-1: https://arxiv.org/abs/2402.18771

### 3. World Models

World model $P(\cdot)$ 显式建模 commonsense 并预测未来 state（Eq. 2）：

$$\hat{s}_{t+1} \sim P(\hat{s}_{t+1} | s_t, a_t)$$

区别于 forward dynamics（作为 pretraining/auxiliary loss），world model 是 standalone module，支持 model-based control 和 planning。

**(a) Latent dynamics world models** — Dreamer 系列

Dreamer 用三个 module 构造 latent dynamics：
- **Representation model**: image → latent state
- **Transition model**: latent state transition
- **Reward model**: 预测 reward

然后在 imagination 中通过 actor-critic 训练，analytic gradient 反向传播通过 learned dynamics。

DreamerV2 → 引入 discrete latent space。DreamerV3 → 固定超参跨域。DayDreamer → 应用到真实物理 robot。

**(b) Transformer-based world models** — IRIS、TWM

IRIS 用 GPT-like autoregressive Transformer + VQ-VAE vision encoder。TWM 用 Transformer-XL。

**(c) LLM-induced world models**

LLM 内含丰富 commonsense，可直接当 world model 用：
- **DECKARD**: LLM 生成 abstract world model（DAG），用于 Minecraft crafting。Dream phase 采样 subgoal，Wake phase 执行并更新 AWM。
- **LLM-DM**: LLM 用 PDDL 构造 world model（LLM+P 是手工的）。
- **RAP**: LLM 同时当 policy 和 world model，用 MCTS 做规划。
- **LLM-MCTS**: 扩展到 POMDP，LLM 生成 initial belief。

**(d) Visual world models** — Genie、3D-VLA、UniSim

Genie：spatiotemporal video tokenizer + autoregressive dynamics model + latent action model，无监督训练，可交互式生成环境。3D-VLA：用 diffusion 从 image/depth/point cloud 生成 goal state。UniSim：从 real-world interaction video 学 generative model。

**Intuition**: World model 让 agent 在 "imaginary space" 中先 search 最优 action sequence 再执行，类似人类"先想后做"。LLM 当 world model 的好处是 commonsense 充足，缺点是只能产出 text；visual world model 直接生成未来 frame/video，更接近物理世界，对 sim-to-real 和 data augmentation 都极有价值（Sora 之后这一方向爆发）。

参考:
- DreamerV3: https://arxiv.org/abs/2301.04104
- Genie: https://arxiv.org/abs/2402.05972
- Sora as world simulator: https://openai.com/index/video-generation-models-as-world-simulators/

### 4. Reasoning & Policy Steering

Chain-of-Thought (CoT) 给 VLA 加 reasoning 能力：

- **ECoT**：在 OpenVLA 上训练 embodied CoT —— 先 reasoning plan/sub-task/motion/visual feature，再产 action。不依赖"muscle memory"，提高 generalization。
- **CoT-VLA**：引入 visual CoT。
- **V-GPS**：用 learned value function re-rank action。
- **RoboMonkey**：用 VLM verifier 从 sample set 选最优 action —— test-time scaling。

**Intuition**: CoT 在 language domain 自然适合 task planning。ECoT 的创新是把 reasoning 嵌入 low-level control policy，让 VLA 在出 action 前"想清楚"，类似人类抓物体前先观察、规划、再动手。

参考:
- ECoT: https://arxiv.org/abs/2407.08693
- CoT-VLA: https://arxiv.org/abs/2503.06426

---

## 四、Low-level Control Policies - 架构进化史

### 1. Non-Transformer 时代

- **CLIPort** = CLIP + Transporter Network。Two-stream：CLIP 出 semantic，Transporter 出 spatial。Hadamard product 融合。输出 SE(2) action（pick + place 位姿）。这是最早证明 language-conditioned pick-and-place 的工作。
- **BC-Z**：FiLM 融合 instruction embedding 和 image embedding，MLP 出 action。声称 zero-shot task generalization。
- **MCIL**：首个 free-form natural language conditioning，可从 unstructured demo 学。
- **HULC**：hierarchical decomposition + multimodal Transformer + discrete latent plan + visuo-lingual semantic alignment loss（contrastive）。
- **UniPi**：把决策建模成 text-conditioned video generation —— 先生成 video，再用 inverse dynamics 从 frame 抽 action。这是 policy-as-video 的开创性 idea，把 generative video 模型当 policy 用。

### 2. Transformer 时代

- **Interactive Language**：real-time language-guided rearrangement，数据集比之前大一个数量级。
- **Hiveformer**：multi-view + full history observation，早期 Transformer backbone adopter。
- **Gato**：unified tokenization 让一个 model 同时玩 Atari、做 image captioning、stack block。token 化一切是 multi-task 的关键。
- **RoboCat**：基于 Gato + VQ-GAN image encoder，预测 action 和未来 observation。self-improvement：用 finetune 后的 model 自动生成新 data。
- **RT-1**：EfficientNet vision encoder + Transformer action decoder，输出 discretized action。可 attend 到历史 image，性能超 BC-Z。
- **Q-Transformer**：在 RT-1 上加 autoregressive Q-function + conservative regularizer（保证 max value action 在 distribution 内），可用失败 trajectory 学。
- **RT-Trajectory**：用 trajectory sketch（曲线）当 condition，替代 language/goal image。
- **ACT**：CVAE + action chunking（预测 action sequence 而非单个）+ temporal ensembling。RoboAgent (MT-ACT) 扩展到 semantic augmentation。
- **RoboFlamingo**：把 Flamingo VLM 加 LSTM policy head，证明 VLM 可迁移到 manipulation。

### 3. Multimodal Instruction

- **VIMA**：multimodal prompt，定义 4 个 generalization level（placement、novel combinatorial、novel object、novel task）。Cross-attention 架构。
- **MOO**：RT-1 + OWL-ViT 处理 prompt 中的 image，可 finger pointing / mouse click 指定目标。

### 4. 3D Vision

- **PerAct**：voxel input + output（3D voxel 中选最佳 voxel 指导 gripper）。
- **Act3D**：continuous resolution 3D feature field，adaptive resolution 降算力。
- **RVT/RVT-2**：把 point cloud re-render 成 virtual view image，用 2D image 当输入。
- **DP3、3D Diffuser Actor**：point cloud + diffusion policy。

### 5. Diffusion-based

- **Diffusion Policy**：把 policy 当 DDPM，含 receding horizon control + visual conditioning + time-series diffusion Transformer。处理 multimodal action distribution 和高维 action space。
- **SUDD**：LLM guide data generation → distill 成 visuo-linguo-motor policy。
- **Octo**：modular Transformer-based diffusion policy，首个用 OXE dataset 的，证明 cross-robot/cross-task transfer。
- **MDT**：用 DiT 替换 U-Net + masked generative foresight + contrastive latent alignment。
- **RDT-1B**：bimanual manipulation 的 diffusion foundation model，1.2B 参数，统一 action format 跨 robot，zero-shot 泛化。

### 6. Motion Planning

- **Language costs**：natural language feedback → cost map → motion planner。
- **VoxPoser**：LLM 写 code 调 VLM 得 object 坐标 → 组合 affordance map + constraint map → MPC 出 trajectory。零训练。
- **RoboTAP**：TAPIR 算法跟踪 active point，visual servoing 控制。

### 7. Point-based Action

- **PIVOT**：把 robotics 当 VQA，VLM 从 keypoint proposals 中选 action。
- **RoboPoint**：finetune VLM 做 spatial affordance prediction（点图像哪里）→ 用 depth 投到 3D。
- **ReKep**：3D keypoint constraint function，VLM 生成 constraint → 解 constrained optimization。

### 8. Large VLA (LVLA)

这是 paper 的核心关注点，对应原 VLA 定义：

- **RT-2**：用 PaLI-X / PaLM-E 当 backbone，引入 **co-fine-tuning**（同时拟合 VQA data + robot data），emergent capability 出现。
- **RT-H**：action hierarchy，中间层是 language motion（如"move arm up"），让 "pick" 和 "pour" 共享 "move arm up" 这类 motion，提升 data sharing。
- **RT-X**：用 OXE dataset 重训 RT-1 和 RT-2，RT-1-X 和 RT-2-X 都超原版。
- **OpenVLA**：开源版 RT-2-X，DINOv2 + SigLIP vision encoder + Prismatic-7B LLM。支持 LoRA 和 quantization。
- **OpenVLA-OFT**：Optimized Fine-Tuning recipe。
- **TraceVLA**：加 visual trace prompting 提升 spatial-temporal awareness。
- **$\pi_0$**：flow-matching 架构，加 action expert（MoE），继承 VLM 的 internet-scale knowledge 同时扩展到 robot task。
- **RoboMamba**：用 Mamba SSM 替 Transformer，linear inference complexity。
- **SpatialVLA**：Ego3D Position Encoding + Adaptive Action Grids。
- **LAPA**：unsupervised pretraining based on latent actions。三阶段：VQ-VAE 提取 quantized latent action → VLA 预测 latent action → finetune 映射 latent → real action。从 internet-scale unlabeled video 学。
- **TinyVLA**：小 VLM + Diffusion head，重 inference speed 和 data efficiency。
- **CogACT**：DiT action diffusion module。
- **DexVLA**：embodied curriculum learning 训 diffusion action expert + sub-step reasoning。
- **HybridVLA**：diffusion + autoregression hybrid，充分利用 VLM reasoning。
- **GR00T N1**：dual-system 架构 —— System 2 (VLM, 10Hz) 处理 image + language，System 1 (diffusion, 120Hz) 出 closed-loop motor action。专门为 humanoid robot 设计。
- **NORA-1.5**：unify VLA + world model via reward-guided post-training。
- **Genie Envisioner**：world foundation platform，video-generative framework 内集成 world model + VLA。
- **WorldVLA、UniVLA**：把所有 modality 量化成 discrete token，autoregressive 统一建模，能生成 action + text + image，构成 world model。受 VAR 启发。

### Action Type 和 Training Objective

公式（Eq. 4-7）：

**Continuous action BC**:
$$\mathcal{L}_{\text{Cont}} = \sum_t \text{MSE}(a_t, \hat{a}_t)$$
$a_t$ 是 expert demo 的 action annotation，$\hat{a}_t$ 是 model 预测。

**Discrete action BC**:
$$\mathcal{L}_{\text{Disc}} = \sum_t \text{CE}(a_t, \hat{a}_t)$$
把 action value range 分成固定 bin 数。RT-1 用这个，但 Octo 指出会导致 early grasping issue。

**SE(2) action BC**:
$$\mathcal{L}_{SE(2)} = \text{CE}(a_{\text{pick}}, \hat{a}_{\text{pick}}) + \text{CE}(a_{\text{place}}, \hat{a}_{\text{place}})$$
只预测 pick 和 place end-effector pose，对 tabletop manipulation 够用，但 "pouring water into cup" 需要 SE(3)。

**DDPM**:
$$\mathcal{L}_{\text{DDPM}} = \text{MSE}(\varepsilon^k, \varepsilon_\theta(a_t + \varepsilon^k, k))$$
$\varepsilon^k$ 是第 $k$ 步 iteration 的随机 noise，$\varepsilon_\theta$ 是 noise prediction network（即 VLA）。这是去噪 score matching。

### 架构对比 (Figure 5)

作者画了 7 种代表架构：
- **FiLM**（RT-1 用）：instruction embedding 通过 FiLM layer 调制 image feature。简单高效。
- **Cross-attention**（VIMA、RoboFlamingo）：modality 间 cross-attend，小 model 下性能更好。
- **Concatenation**（OpenVLA、$\pi_0$）：简单拼接 token，大 model 下效果可比 cross-attn 且实现简单。
- **Quantization**（WorldVLA、UniVLA）：把多 modality 量化成 shared vocabulary token，可与 world model 统一。
- **Tool-use**（Instruct2Act、VoxPoser）：LLM 生成 code 调 VLM/tool API。
- **VLA + World Model**（NORA-1.5、Genie Envisioner）：双系统融合。
- **Unified Vocabulary**（UniVLA）：所有 modality autoregressive 统一。

参考:
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- $\pi_0$: https://arxiv.org/abs/2410.24164
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- RT-X / OXE: https://arxiv.org/abs/2310.08864

---

## 五、High-level Task Planners

Task planner 把 complex task $\ell$ 拆成 subtask 序列（Eq. 8）：

$$[p_1, p_2, \ldots, p_N] \sim \pi_\phi(\ell, s_t)$$

### 1. Monolithic Task Planner

**End-to-end**:
- **PaLM-E**：ViT + PaLM → embodied multimodal LLM。从 image + high-level instruction 生成 text plan，给 low-level policy。可 replan。还能做 embodied VQA。
- **EmbodiedGPT**：embodied-former 输出 instance-level feature，融合 vision encoder embedding + LLM planning info，给 low-level policy。

**3D Vision**:
- **LEO**：point cloud encoder + LLM，两阶段（3D-VL alignment → 3D-VLA instruction tuning）。
- **3D-LLM**：注入 3D 信息（point cloud、gradSLAM、neural voxel field）。
- **MultiPLY**：object-centric，加 audio/tactile/thermal。
- **ShapeLLM**：3D vision encoder ReCon++ + LLaMA，蒸馏 multi-view image+text teacher。

**Grounded**:
- **SayCan**：LLM "says" next skill（task-grounding），low-level value function 当 affordance "can"（world-grounding），结合选最优 skill。这是 LLM + affordance 的经典框架。
- **Translated ⟨LM⟩**：causal LLM 生成 plan → masked LLM 翻译成 VirtualHome action。
- **(SL)³**：segmentation + labeling + parameter update 三步循环。

### 2. Modular Task Planner

**Language-based** (Figure 6a):
- **Inner Monologue**：LLM 生成 instruction 给 low-level policy，从 success/object/scene/human feedback 动态更新。无需训练 LLM。
- **LLM-Planner**：high-level LLM 生成 plan + low-level 翻译 subgoal + re-planning。
- **LID**：Active Data Gathering + hindsight relabeling，最大化利用失败 trajectory。
- **Socratic Models**：多模型 zero-shot 组合，multimodal-informed prompting 让非 language modality 转 text。

**Code-based** (Figure 6b):
- **ProgPrompt**：program-like specification 让 LLM 生成 household plan，environment feedback 通过 assertion。
- **ChatGPT for Robotics**：定义 API → prompt ChatGPT → iterative code gen with sim eval。
- **CaP (Code as Policies)**：GPT-3/Codex 生成 policy code 调 perception + control API。空间几何 reasoning 强、可 parameterize。
- **COME-robot**：GPT-4V 去除 perception API 需求。
- **DEPS**：Describe/Explain/Plan/Select + self-explanation + trainable goal selector。
- **ConceptGraphs**：RGB → 2D segmentation → VLM caption → 3D scene graph (JSON) → LLM planning。

**Intuition**: Monolithic 模型在 specialized embodied data 上 finetune 性能强，但训练成本高；modular 用 off-the-shelf LLM/VLM 部署快。Language-based 模型天然适合 LLM/VLM，但生成的 subtask 可能 low-level policy 执行不了；code-based 需预先 wrap API + 文档，但 code 可 debug、controllability 高。

参考:
- PaLM-E: https://arxiv.org/abs/2303.04137
- SayCan: https://arxiv.org/abs/2204.01691
- Code as Policies: https://arxiv.org/abs/2209.07753

---

## 六、Datasets 和 Benchmarks

### Real-world Robot Datasets (Table V)

数据稀缺是 embodied AI 的核心瓶颈。原因：
1. Robot 设备 + 环境 + teleoperation 成本高
2. Robot 类型/gripper/control mode 不一致
3. Object 6D pose 难精确捕获

代表性 dataset 演进：
- MIME (2018): 8.3K episodes, Baxter, demo instruction
- RoboNet (2019): 162K, 7 个 robot, goal image
- BC-Z (2021): 25.9K, EDR, lang + demo
- Fractal (RT-1, 2022): 130K, EDR, 700+ tasks
- VIMA (2022): 650K, UR5, multimodal prompt
- BridgeV2 (2022): 60.1K, WidowX, 24 scenes
- RH20T (2024): 110K+, 4 个 robot, 147 tasks
- DROID (2024): 76K, Franka, 564 scenes, 86 skills
- **OXE (RT-X, 2023)**: 1M+ episodes, 22 个 robot, 527 skills, 160K tasks —— 跨 embodiment 的 aggregated dataset

**Intuition**: OXE 是 VLA 领域的 "ImageNet 时刻"，让 cross-embodiment 训练成为可能。但相对 LLM 的 trillion token，robot data 仍小 6 个数量级，data efficiency 和 simulated data 是关键。

### Simulators (Table VI)

- **Gibson/iGibson**: 5721 scenes, navi/mani，PyBullet
- **SAPIEN**: PhysX, articulation + ray tracing
- **AI2-THOR**: Unity, 120 rooms, object states
- **VirtualHome**: Unity, 71 scenes, 509 object cats，task planning
- **TDW**: Unity, audio + fluids
- **RLBench**: 28 tasks, Flex Bullet
- **Meta-World**: 80 tasks, tiered difficulty
- **CALVIN**: 41 scenes, long-horizon lang-cond tasks
- **Franka Kitchen**: 7 objects
- **Habitat**: Bullet, navi only, fast
- **ALFRED**: 120 rooms, long-horizon
- **DMC**: MuJoCo, continuous RL
- **Genesis**: 高速综合物理仿真，支持 rigid/deformable/liquid

**Sim-to-real gap** 是核心挑战：rendering quality、physics 不准、object property 域漂移、deformable/liquid 难仿真。但 sim 提供 automated metric 和 setup 复现，对 fair comparison 必要。

### 自动化和人类数据

- **RoboGen**：generative simulation paradigm，自动 propose skill → 模拟环境 → 选最优学习方法。
- **AutoRT**：LLM-driven robot orchestrator，生成 task → affordance filter → autonomous policy/human teleop 收集。
- **DIAL**：用 VLM 增强 existing dataset 的 language instruction。
- **RoboPoint**：procedural scene generation。
- **UMI**：hand-held gripper 缓解人类数据转移到 robot 的问题。

### Task Planning 和 EQA Benchmark

- **EgoPlan-Bench**：真实世界 task planning，human annotation。
- **PlanBench**：cost optimality / plan verification / replanning。
- **LoTa-Bench**：直接执行 plan 算 success rate。
- **EAI (Embodied Agent Interface)**：formalize LLM 模块 IO，fine-grained metric。
- **EQA Benchmark** (Table VII)：EQA、IQUAD、MT-EQA、MP3D-EQA、EgoVQA、EgoTaskQA、EQA-MX、OpenEQA —— 涉及 spatial reasoning、physics、world knowledge，agent 可 active explore。

参考:
- OXE/RT-X: https://robotics-transformer-x.github.io/
- DROID: https://droid-dataset.com/
- CALVIN: https://arxiv.org/abs/2112.03227

---

## 七、Challenges 和 Future Directions

作者列了 9 大方向，每个都暗藏研究机会：

### a. Safety first
Robot 直接作用物理世界。需要 commonsense guardrail、risk assessment、human-robot protocol。RLHF 和 "evaluation without execution" 可降风险。Interpretability + expandability 对 error diagnosis 关键。

### b. Datasets & Benchmarks
现有 benchmark 覆盖 skill/object/embodiment/environment 不全。需要超越 success rate 的 fine-grained metric（EAI 已指出方向）。

### c. Foundation Models & Generalization
VLA foundation model / Robotic Foundation Model (RFM) 仍未达 LLM 在 NLP 的 generalization 水平，需要更多 AGI 核心能力。

### d. Multimodality
Modality alignment 是核心难题。ImageBind、LanguageBind 都把 modality 对齐到 image/language space。但仅靠 embedding 不够。其他 modality（audio、haptics、gaze）也证明有用：
- Audio：AuRL 学动态行为
- Tactile：T-Dex 多指手
- Gaze：auxiliary loss 注入 human visual attention

### e. Framework for Long-Horizon Tasks
Hierarchical 是当前最实用方案，但增加 system 复杂度和 failure 点。频繁 re-planning → 高 latency。两个大模型 redundancy 阻碍 scaling。Modular planner 不是 plug-and-play：language-based 可能生成不可执行 subtask；code-based 需手动 wrap API。**End-to-end 直接 translate long-horizon task 到 low-level control signal 是值得探索的方向**。

### f. Real-Time Responsiveness
Robot 要实时响应动态环境。Inference 慢于环境变化 → 产出 obsolete action。LVLA 和 task planner 面临 speed-capacity tradeoff。GR00T N1 的 dual-system（10Hz + 120Hz）是一种解法。

### g. Multi-agent Systems
分布式感知、协作故障恢复。但 communication、dispatching、fleet heterogeneity、conflicting goal 都是挑战。

### h. Ethical and Societal Implications
Privacy、job displacement、bias、social norm 冲击。

### i. Applications
当前 VLA 集中 household/industrial，可扩展到 virtual assistant、autonomous vehicle、agricultural robot。新 embodiment（dexterous hand、drone、quadruped、humanoid）需 specialized VLA。**Healthcare**（surgical robot、care robot）要求高 safety/privacy，需 HITL control、federated learning、specialized medical vision model。

---

## 八、Appendix 中的关键技术点

### 1. Unimodal Model 演进

- **CV**: LeNet → AlexNet → VGG/GoogLeNet → ResNet（skip connection）→ SENet → EfficientNet（compound scaling）。Object detection: R-CNN 系列、YOLO、FPN/RetinaNet。Segmentation: FCN、U-Net。ViT 把 image 切 16×16 patch 当 token。DETR 用 encoder-decoder + learnable object query。SAM promptable segmentation。
- **NLP**: word2vec/GloVe → RNN (LSTM/GRU/RNNsearch) → Transformer → BERT (encoder) / GPT (decoder) → T5 (text-to-text) → LLM (ChatGPT/GPT-4/PaLM/LLaMA)。Instruction tuning (InstructGPT/FLAN/Alpaca/Vicuna) → prompt engineering → DPO 简化 RLHF。
- **RL**: DQN → AlphaGo → PPO → Dactyl。Value-based (D-DQN/HER/BCQ/CQL) + policy search (DPG/DDPG/A3C/Soft AC/TRPO/PPO) + imitation (GAIL) + hierarchical (FuN)。
- **Graph**: GCN、GAT、GROVER、SE(3)-Transformer。Scene graph 和 knowledge graph 与 CV/NLP 结合。

### 2. VLM 演进

- **Self-supervised pretraining**: ViLBERT (multi-stream)、VL-BERT (single-stream)、UNITER (word-region alignment)、ViLT (ViT patch)、SimVLM (prefix LM)、BEiT-3 (mixture-of-modality-experts)。
- **Contrastive pretraining**: CLIP（双塔）、FILIP（token-wise）、ALIGN（noisy data）、ALBEF（+multimodal encoder）、CoCa（contrastive + generative）、Florence（coarse-to-fine + image-to-video）。
- **Large Multimodal Model (LMM)**: Flamingo（gated cross-attn）、BLIP-2（Q-Former）、PaLI/PaLI-X（joint scaling）、LLaVA（linear layer）、MiniGPT-4、Kosmos-1/2、InstructBLIP、mPLUG-Owl、Visual ChatGPT、X-LLM。

**Intuition**: VLM 的核心是 modality fusion 机制 —— 是 cross-attn（Flamingo）、Q-Former（BLIP-2）、还是 linear layer（LLaVA）？LLaVA 用最简单的 linear projection 证明：当 LLM 足够大时，简单接口就够。这对 VLA 设计有深远影响：RT-2、OpenVLA 都用 concat/linear 接口，因为大 LLM 本身能吸收对齐压力。

---

## 九、整体 Intuition 构建

### 1. VLA 进化的本质矛盾

整个 VLA 领域在解决三个矛盾：

**(a) Capacity vs Speed**: LLM/VLM 容量大但慢。RT-2、OpenVLA 性能强但 inference 慢，动态环境会失效。GR00T N1 用 dual-system 缓解。

**(b) Generality vs Precision**: SE(2) action 够 tabletop 但不够 pour water；continuous action 精度高但难学。Point-based action 粗糙但 zero-shot。ReKep 用 constraint 解 SE(3)。

**(c) Data scarcity vs Diversity**: Robot data 难收集。OXE 跨 embodiment 缓解；VPT 用 inverse dynamics 自动 label video；LAPA 用 latent action 从 internet video 学；RoboGen 用 generative simulation；UMI 用人类数据。

### 2. 三大趋势

**(i) Tokenization of Everything**: Gato 统一 token → GR00T N1 dual-system token → WorldVLA/UniVLA 全 modality 量化 token，autoregressive 统一。最终 vision + language + action + world model 在一个 token 空间。

**(ii) Diffusion as Action Head**: Diffusion Policy → DP3、3D Diffuser Actor → Octo、MDT → RDT-1B → CogACT、DexVLA、HybridVLA。Diffusion 处理 multimodal action distribution 和高维 action space，但 inference 慢。Rectified flow、DDIM 在加速。

**(iii) VLA + World Model 融合**: 从 Dreamer（latent dynamics）→ Genie（generative interactive env）→ NORA-1.5（reward-guided post-training）→ WorldVLA/UniVLA（统一 autoregressive）→ Genie Envisioner（统一 platform）。World model 让 VLA 能 "imagine" 后再 act，是通向 AGI embodied intelligence 的关键。

### 3. Scaling Law 在 Robotics

Paper 引用 [151][152] 指出 robotics 也存在 scaling law：model size、dataset size、environment/object diversity 都重要。但 robot data 比 NLP data 贵 6+ 个数量级，所以：
- 数据高效方法（LAPA、TinyVLA）
- 跨 embodiment 数据聚合（OXE）
- Simulation（Genesis 等高速仿真）
- 自动 label（inverse dynamics、VLM relabeling）
- 人类数据迁移（UMI、Being-H0、X-Humanoid）

这些方向都在试图破解 robot data scaling bottleneck。

### 4. 论文的核心贡献

- **第一个 comprehensive VLA survey**
- **Generalized VLA 定义**：把原 VLA 概念扩展到所有 vision+language→action 模型，原 VLA 升级为 LVLA
- **三层 taxonomy**：components / low-level policy / high-level planner，覆盖全栈
- **资源汇总**：dataset + simulator + benchmark 完整列表
- **未来方向**：safety、RFM、multimodality、long-horizon、real-time、multi-agent、healthcare 等

---

## 十、潜在研究机会

基于这篇 survey 我看到的几个 high-value 方向：

1. **Latent action pretraining 的扩展**（LAPA 后续）：能否把 VQ-VAE latent action 做成跨 embodiment 通用？这是 internet video → robot policy 的关键桥梁。

2. **Test-time scaling for VLA**：RoboMonkey、V-GPS 是早期尝试。LLM 的 test-time compute scaling（o1、R1 范式）能否迁移到 VLA？Action level 的 verifier + search 是开放问题。

3. **World model 作为 VLA 的 data augmentor**：用 visual world model 生成 synthetic trajectory 训 VLA，类似 AlphaGo 的 self-play。

4. **Embodied CoT 的 systematic study**：ECoT、CoT-VLA 之后，reasoning 在哪个 latent space 进行最优？是 text、visual、还是 latent action？

5. **Sim-to-real via generative world model**：UniSim、Genie 类模型生成的环境能否直接当训练 ground？绕过传统 physics simulator 的 sim-to-real gap。

6. **Cross-embodiment foundation model**：OXE 已有 22 个 robot。能否用一个 unified action representation（如 latent action + per-embodiment decoder）训真正的 RFM？

7. **VLA 的 safety alignment**：SafeVLA 是早期工作，类似 LLM 的 RLHF/DPO，需要 VLA-specific preference data 和 alignment 方法。

8. **Long-horizon end-to-end**：当前 hierarchical 是 workaround。能否像 video LLM 那样直接处理 long-horizon context 输出 long action sequence？Long-VLA、LVP 是早期尝试。

---

## 参考资源汇总

- **论文 repo**: https://github.com/yueen-ma/Awesome-VLA
- **OXE dataset**: https://robotics-transformer-x.github.io/
- **DROID**: https://droid-dataset.com/
- **RT-2**: https://robotics-transformer2.github.io/
- **OpenVLA**: https://openvla.github.io/
- **$\pi_0$**: https://www.physicalintelligence.company/blog/pi0
- **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu/
- **CALVIN benchmark**: https://calvinrobot.github.io/
- **Genie**: https://sites.google.com/view/genie-ai
- **Sora world sim**: https://openai.com/index/video-generation-models-as-world-simulators/
- **GR00T N1**: https://arxiv.org/abs/2503.14734
- **ECoT**: https://embodied-cot.github.io/

---

这篇 survey 把一个爆炸式增长的领域（2020-2025 约 400 篇 VLA paper）组织成 coherent 全景图，既适合入门也适合老手定位研究方向。最核心 takeaway：VLA 正在从"VLM + action head"的简单拼接，走向 token 统一、world model 融合、test-time reasoning 的综合 AGI embodied intelligence 系统。
