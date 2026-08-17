---
source_pdf: From Spatial to Actions.pdf
paper_sha256: 10025b68c5223d36ac60b84a4692fbb268ce304721996168cd454723c8023609
processed_at: '2026-08-04T11:03:41-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FALCON 人话版

## 一句话总结

FALCON 解决了一个尴尬问题：**现在最火的 VLA 机器人模型都在用 2D "眼睛" 看 3D 世界，等于让一只眼看立体的人去抓东西**。

---

## 为什么需要 FALCON

假设你让机器人 "把香蕉放到红盘子上"。现在的 VLA 模型能听懂这句话，也能认出香蕉和盘子，但**不知道香蕉离自己多远、盘子有多高、该以什么角度伸手**。这就是 2D VLM 做 3D manipulation 的根本矛盾。

之前的解法分两派：

**硬核派**（PointVLA、3D Diffuser Actor、GeoVLA）：直接给机器人装 depth camera 或 laser scanner，喂 point cloud。问题在于：
- Depth sensor 贵且笨重，不是每个 robot setup 都有
- Open X-Embodiment 这种百万级 dataset 根本没 depth 标注，等于白瞎
- 换个 sensor 配置就要 retrain

**软装派**（SpatialVLA、3D-VLA、Evo-0）：给 VLM 加一些 learnable spatial embeddings，让它 "假装" 理解 3D。问题在于：
- 这种 learnable embeddings 是**虚的**——没有任何 geometric supervision，只在 LLM 的高维空间里乱飘
- 把 spatial tokens 跟 text tokens 一起塞进 VLM，**破坏了 VLM 原本的 vision-language alignment**，就像往一碗好汤里倒酱油，原本的味道没了
- 一旦遇到新场景或需要 reasoning 的 spatial prompt，performance 立刻崩

FALCON 要同时解决这三件事：**spatial representation 要强、modality 要灵活、VLM alignment 不能坏**。

---

## FALCON 的核心 idea：抄大脑作业

人脑有个很漂亮的 division of labor：
- **Cerebrum（大脑皮层）**：负责 high-level reasoning——你跟人说 "把那个红的给我"，它负责听懂语义、认出红色物体
- **Cerebellum（小脑）**：负责 fine-grained motor control——伸手的角度、力道、timing

这俩是**分开**的，cerebrum 不直接管 motor control，它只是 "下令"，cerebellum 负责 "执行"。

FALCON 照搬这个结构：
- **VLM = Cerebrum**：Kosmos-2 负责 "听懂语言 + 看懂图像 + 给出 semantic intention"
- **Action Head = Cerebellum**：负责融合 spatial information 和 semantic intention，输出具体的 7-DoF action

**关键 move**：spatial tokens **不进 VLM**，直接走旁路进 action head。这样 VLM 的 pre-trained alignment 一点不受破坏，同时 spatial information 又能 influence 最终动作。

Prior work 把 spatial tokens 塞进 VLM，相当于逼 cerebrum 同时干 cerebrum 和 cerebellum 的活，结果两件事都做不好。FALCON 的 ablation（Table 4）实测验证：把 spatial tokens 注入 VLM 后，CALVIN ABC→D 的 Avg. Len. 从 3.91 掉到 3.79。

---

## 三个技术组件拆解

### 1. VLM Backbone：Kosmos-2

这部分 boring，就是用 Kosmos-2（Microsoft 的 grounded multimodal LLM，1.6B params）处理 image + language，输出一个 semantic action token $\hat{\mathbf{t}}_{act}$。这个 token 编码了 "要干什么、抓什么、放到哪" 的高层意图。

### 2. Embodied Spatial Model (ESM)：FALCON 的灵魂

ESM 基于 VGGT（CVPR 2025 的 spatial foundation model，能从纯 RGB 重建 3D scene）。FALCON 做了三件事改造它：

**(a) 给它装 "几何外挂"**

VGGT 原本只能吃 RGB，FALCON 让它**可选**吃 depth map 和 camera pose：
- Depth 通过 conv encoder 转成 tokens，element-wise 加到 visual tokens 上
- Camera pose 通过 MLP 编码成 token，替换 learnable camera token

**(b) Stochastic Conditioning——最骚的一招**

训练时随机决定要不要喂 depth/pose，公式 5：
$$(\mathbf{T}_{spl}, \hat{\mathbf{t}}_{cam}) = \mathcal{E}_{spl}\Big(\mathbf{T}_{vis} + b_d \mathbf{T}_{dpt}, \; b_p \mathbf{t}_{gt-cam} + (1-b_p) \mathbf{t}_{cam}\Big)$$

$b_d, b_p$ 是 Bernoulli(0.66) 随机变量，66% 概率注入。

**为什么这个 trick 牛**？它逼模型同时学两件事：
- 有 depth 时：利用 depth 做精确 reconstruction
- 没 depth 时：从 RGB 推断 spatial structure

而且这俩能力**互相增强**。Table 5 的结果很 striking：训练只用 RGB，test 时加 depth，Avg. Len. 从 4.08 → 4.09；训练用 RGB-D，test 去掉 depth，4.09 → 4.07（几乎不降）。**这种 graceful degradation 是 prior method 完全做不到的**。

**(c) 监督信号**

ESM 用 VGGT 的多任务 loss：depth prediction + point map prediction + camera pose estimation。这三个 head 共享 spatial encoder $\mathcal{E}_{spl}$，让 $\mathbf{T}_{spl}$ 真正 encode 了 3D structure。

### 3. Spatial-Enhanced Action Head：融合层

这里做了一件反直觉的事——**用最简单的 element-wise addition 融合**。

公式 6：$$\mathbf{f}_{fused} = \hat{\mathbf{t}}_{act} + \widetilde{\mathbf{t}}_{spl}$$

他们试了三种 fusion：
- Cross-Attention：spatial feature 作 K/V，semantic token 作 Q
- FiLM-Gated：spatial feature 生成 $\gamma, \beta$ 对 semantic token 做 affine modulation
- Element-wise Addition：直接加

结果 element-wise addition 最好（ABC→D Avg. Len. 3.91，cross-attn 3.68，FiLM 3.76）。

**为什么简单加法赢**？因为 cross-attention 和 FiLM 都会**改变 VLM 输出 feature 的分布**，破坏 pre-trained representation。Element-wise addition 是 additive refinement，相当于 "在 semantic intention 上加一个 spatial offset"，最不 invasive。

之后 action predictor 用 MLP 或 LSTM（long-horizon 任务用 LSTM 吃历史 H 步 features）。

---

## Training Trick：Two-Stage Post-Training

FALCON 不 from scratch 训，而是 post-training 到一个 pre-trained VLA 上。分两阶段：

**Stage 1**：冻住所有 pre-trained 部分（VLM、ESM、action predictor），只训 adapter $\mathcal{D}$。而且 adapter 的最后一层 **zero-init**——开始时 $\widetilde{\mathbf{t}}_{spl} = 0$，模型行为完全等于原 VLA，然后逐步引入 spatial contribution。

**Stage 2**：解冻 VLM 和 adapter，joint refinement。VLM 隐式学会 "调整 semantic features 以配合 spatial cues"。

这个设计灵感来自 LLaVA 的 projector training：先对齐再联合优化，避免 catastrophic forgetting。

---

## 实验亮点

### Simulation

**CALVIN ABC→D**（zero-shot setting，最难）：
- FALCON: 4.40
- RoboVLM: 4.25
- 3D Diffuser Actor（用 GT point cloud）: 3.35
- 3DDP（用 GT point cloud）: 0.27

FALCON **只用 RGB** 就吊打用 GT point cloud 的方法。这说明 spatial foundation model 学到的 dense 3D priors 比 robotic point cloud（通常很 sparse）信息量更大。

**SimplerEnv Google Robot "Open Top Drawer and Place Apple"**（long-horizon + spatial reasoning）：
- FALCON: 41.7%
- RT-2-X (55B params): 3.7%
- SpatialVLA: 0%
- OpenVLA: 0%

这个 task 需要：1) spatially 理解 drawer 在哪；2) 顺序执行 open + place。Baseline 几乎全跪，FALCON 靠 spatial + semantic 双流解决了。

### Real-World

**Base Tasks**（9 task suites，90 rollouts）：
- FALCON: 70.0%
- SpatialVLA: 44.4%
- 提升 25.6 个百分点

**Few-shot Adaptation**：
- Unseen Object 变种 "open drawer and place bread"：FALCON 80%，其他模型 near 0%

**Spatial Understanding**：
- Object scale 变化时，baseline 要么碰撞（large blocks）要么提前 release（small blocks），FALCON 稳如老狗
- Height variation、spatial prompt task 都 SOTA

---

## 我觉得最有意思的 insight

1. **Brain analogy 真的有效**：把 spatial 和 semantic 解耦，比硬塞进一个 model 好得多。这暗示 VLA 的 future 可能是 modular design——vision module、language module、spatial module、motor module 各司其职，用 lightweight fusion 连接。

2. **Stochastic Conditioning 是通用 trick**：这种 "训练时随机 dropout 某个 modality" 的思路可以推广到任何 multi-modal model。比如 VLM 训练时随机 dropout text 或 image，让模型学会单 modality 也能推理。

3. **Foundation model > Raw sensor**：FALCON RGB-only 超越 point cloud 方法，说明 pre-trained geometric priors 比 raw 3D sensor data 更 valuable。这跟 LLM 用 pre-trained knowledge 替代 explicit retrieval 的趋势一致。

4. **Simple fusion > Complex fusion**：element-wise addition 击败 cross-attention 和 FiLM。这跟 "Less is More" 在 fine-tuning 中的现象一致——越简单的 modification 越不破坏 pre-trained representation。

5. **Modality transferability 是 VLA 落地关键**：real-world robot setup 千差万别，有的有 depth camera，有的只有 RGB，有的有 calibrated pose。FALCON 一个 model 适配所有情况，这是 deployment 友好的 essential property。

---

## 局限性

论文没明说但能看出来的：
- ESM 只用 third-view，wrist image 加进去只提升 0.02（Table 10），说明 multi-view ESM 还没做好
- 只测了 tabletop manipulation，mobile manip 和 whole-body control 未知
- 2.9B params 还是挺大，real-world 57Hz 在 RTX 4090 上，离 edge deployment 有距离
- Stochastic conditioning 的 Bernoulli p=0.66 是手调的，不知道是否对所有 task optimal

---

## 一句话总结 FALCON

**用 brain-inspired division of labor 把 spatial intelligence 和 semantic intelligence 解耦，让 VLM 干 VLM 的活，action head 干 action head 的活，spatial foundation model 提供 "小脑" 需要的几何信号，最后用最简单的 element-wise addition 融合——结果在 3 个 simulation benchmark 和 11 个 real-world task 上全 SOTA，还能优雅地适配 RGB-only / RGB-D / +pose 三种 modality 配置。**

Project page: https://falcon-vla.github.io/

---

# FALCON: From Spatial to Actions 深度技术解析

## 1. 论文背景与核心motivation

这篇 paper 来自 ByteDance Seed、NUS、NTU、THU、SMU 的合作团队，project page: https://falcon-vla.github.io/ 。它要解决的核心问题是 **VLA (Vision-Language-Action) models 在 3D 空间理解上的 fundamental gap**。

当前 SOTA VLA models（如 RT-2、OpenVLA、RoboVLM）都建立在 2D foundation models 之上，比如 VLMs。这里存在一个 inherent contradiction：VLMs 在 2D domain 运作，而 VLA 必须与 3D physical world 交互。这种 2D-3D mismatch 导致两个 critical bottleneck：

- **Limited generalization**：无法 robust 迁移到 novel scenes、backgrounds 或 object variations
- **Lack of adaptability**：无法应对 height variations 或 object scale differences

之前的 3D integration 方法分两类：

**第一类**：直接使用 explicit 3D inputs（point clouds、depth maps），如 PointVLA、GeoVLA、3D Diffuser Actor、3D Diffusion Policy。这些方法存在 **low modality transferability** 问题——需要 specialized sensors，且 Open X-Embodiment 这类 large-scale datasets 缺少 aligned 3D annotations。

**第二类**：注入 weak 3D cues，如 ZoeDepth 的 pseudo-depth 或 learnable spatial embeddings（SpatialVLA、3D-VLA、Evo-0）。这些方法存在三个 limitations：
1. Limited spatial representation：learnable embeddings 只提供 weak geometric signals
2. Lack of modality transferability：无法 exploit 更高质量的 3D inputs
3. Alignment challenges：spatial embeddings 与 text tokens 拼接会 disrupt pre-trained vision-language alignment，导致 embedding drift，降低 zero-shot generalization

## 2. FALCON 核心设计哲学

FALCON 的设计灵感来自 **brain 的 division of labor**：
- VLM（cerebrum 大脑）→ 高层 reasoning 和 semantics
- Action head（cerebellum 小脑）→ fine-grained motor control 和 sensorimotor integration

这个 insight 来自神经科学，参考 Rochefort et al., Science 2011 (https://www.science.org/doi/10.1126/science.1207403) 和 Figure AI 的 Helix model。

基于这个 metaphor，FALCON 提出**将 spatial tokens 注入 action head 而非 VLM backbone**，这与 prior works（如 VLM-3R、Spatial-MLLM）将 spatial tokens 直接拼入 VLM 的做法形成 contrast。

## 3. Problem Formulation

### 公式 (1) 详解

$$A_t = \mathcal{F}(O_t, L, D_t, P)$$

变量解释：
- $A_t = [a_t, \ldots, a_{t+C-1}]$：在 timestep $t$ 生成的 action sequence，C 为 action horizon
- $O_t = \{I_t^1, \ldots, I_t^n\}$：image observations 集合，包括 third-view $I_t^{3rd}$（全局 scene context）和 wrist-mounted $I_t^{hand}$（fine-grained object details）
- $L$：natural language instruction
- $D_t \in \mathbb{R}^{H \times W}$：optional depth maps
- $P \in \mathbb{R}^7$：optional camera pose（7D 表示，通常包含 quaternion 4D + translation 3D）
- $\mathcal{F}(\cdot)$：mapping function（policy）
- 每个 $a_i$ 是 7D vector：6-DoF gripper pose（Euler angles）+ 1D binary open/close state

## 4. Architecture 详解

FALCON 由三个 core components 构成：

### 4.1 2D VLM Backbone

使用 Kosmos-2（https://arxiv.org/abs/2306.14824）作为 backbone，约 1.6B parameters。Kosmos-2 是 Microsoft 提出的 grounded multimodal LLM，能够将 text 与 image regions 对应。

工作流程：
1. Visual 和 textual inputs 被 tokenize 形成 unified multi-modal sequence
2. Append 一个 learnable action token $\mathbf{t}_{act}$
3. 输出 hidden state $\hat{\mathbf{t}}_{act} \in \mathbb{R}^{D_{act}}$ 作为 semantic action representation

### 4.2 Embodied Spatial Model (ESM)

这是 FALCON 的 key innovation，基于 VGGT (Visual Geometry Grounded Transformer, https://arxiv.org/abs/2403.05171) 改造。VGGT 是 CVPR 2025 的工作，能从单张或多张 RGB images 直接预测 point maps、depth maps 和 camera poses，无需 SfM。

#### 公式 (3) 详解：Spatial Encoder

$$(\mathbf{T}_{spl}, \hat{\mathbf{t}}_{cam}) = \mathcal{E}_{spl}(\mathbf{T}_{vis}, \mathbf{t}_{cam})$$

变量解释：
- $\mathbf{T}_{vis}$：image 经 DINOv2 (https://arxiv.org/abs/2304.07193) tokenize 得到的 visual tokens
- $\mathbf{t}_{cam} \in \mathbb{R}^{D_s}$：learnable camera token，编码 camera 信息
- $\mathcal{E}_{spl}(\cdot)$：spatial encoder，包含 N 个 cross-attention + self-attention blocks
- $\mathbf{T}_{spl} \in \mathbb{R}^{M \times D_s}$：输出 spatial tokens，M 为 token number per image，$D_s$ 为 token dimension
- $\hat{\mathbf{t}}_{cam}$：refined camera token

#### 3D Conditions Encoding (公式 4)

$$\mathbf{t}_{gt-cam} = \mathcal{E}_{cam}(\mathbf{t}_{cam}), \quad \mathbf{T}_{dpt} = \mathcal{E}_{dpt}([D'_t \parallel M_{dpt}]), \quad \mathbf{T}_{dpt} \in \mathbb{R}^{M \times D_s}$$

- $\mathcal{E}_{cam}(\cdot)$：MLP-based camera encoder，将 camera intrinsic 和 normalized extrinsic 编码成 GT camera token
- $D'_t = D_t / \text{Norm}(D_t)$：normalized depth，处理 train/test 间不同 depth ranges
- $M_{dpt} \in \mathbb{R}^{H \times W}$：valid map，标记 incomplete depth
- $[\cdot \parallel \cdot]$：channel-wise concatenation
- $\mathcal{E}_{dpt}(\cdot)$：depth encoder，使用 14×14 kernel 的 conv layers，输出与 $\mathbf{T}_{vis}$ 同 size 的 tokens

#### Stochastic Conditioning (公式 5)

$$(\mathbf{T}_{spl}, \hat{\mathbf{t}}_{cam}) = \mathcal{E}_{spl}\Big(\mathbf{T}_{vis} + b_d \mathbf{T}_{dpt}, \; b_p \mathbf{t}_{gt-cam} + (1-b_p) \mathbf{t}_{cam}\Big)$$

变量解释：
- $b_d, b_p \sim \text{Bernoulli}(p)$：两个独立的 Bernoulli random variables
- $p = 66\%$：训练时 66% 概率注入 depth/pose
- 当 $b_d = 0$：不注入 depth（depth term 系数为 0）
- 当 $b_p = 0$：使用 learnable camera token 而非 GT
- depth 通过 element-wise addition 与 visual tokens 融合
- pose 通过 linear interpolation 在 GT 和 learnable camera token 间选择

**Intuition**：这个 stochastic training strategy 让模型学会：
- 在 RGB-only 时依靠 visual tokens 估计 spatial structure
- 在有 depth/pose 时利用额外 geometric cues
- 实现真正的 modality transferability，无需 retraining

### 4.3 Spatial-Enhanced Action Head

#### 公式 (6) 详解：Modality Fusion

$$\widetilde{\mathbf{t}}_{spl} \in \mathbb{R}^{D_{act}} = \mathcal{D}(\mathbf{t}_{spl}), \quad \mathbf{f}_{fused} = \hat{\mathbf{t}}_{act} + \widetilde{\mathbf{t}}_{spl}$$

变量解释：
- $\mathbf{t}_{spl} \in \mathbb{R}^{D_s}$：通过 max-pooling 将 $\mathbf{T}_{spl}$ 从 $M \times D_s$ 压缩成 $D_s$
- $\mathcal{D}(\cdot)$：lightweight MLP adapter，将 spatial feature 投影到 VLM feature space
- $\widetilde{\mathbf{t}}_{spl} \in \mathbb{R}^{D_{act}}$：aligned spatial feature
- $\hat{\mathbf{t}}_{act}$：VLM 输出的 semantic action token
- $\mathbf{f}_{fused}$：element-wise addition 得到的 fused feature

**为什么 element-wise addition 最好**？作者做了 ablation（Table 4），对比三种 fusion 策略：
1. Cross-Attention Fusion（$\hat{\mathbf{t}}_{act}$ 作 query，$\widetilde{\mathbf{t}}_{spl}$ 作 key/value）
2. FiLM-Gated Modulation（生成 affine 参数 $\gamma, \beta$ 对 $\hat{\mathbf{t}}_{act}$ 做 feature-wise linear modulation）
3. Element-wise Addition（无参数）

结果显示 element-wise addition 最好，因为它**不引入额外参数，保持 pre-trained representation space 完整**。Cross-attention 会改变 feature 分布，FiLM 引入额外 nonlinear transformation，反而破坏 VLM 的 generalization ability。

#### Action Predictor

两种 architecture：
- **MLP-based**：$A_t = \pi(\mathbf{f}_{fused}^t)$，单步预测，用于 SimplerEnv 和 real-world
- **LSTM-based**：$A_t = \pi(\mathbf{f}_{fused}^{t-H+1}, \ldots, \mathbf{f}_{fused}^t)$，处理历史 $H$ 步 features，用于 long-horizon 任务（CALVIN）

## 5. Training Objective 与 Two-Stage Pipeline

### 公式 (2) Loss Function

$$\mathcal{L} = \sum_{i=t}^{t+C-1} \text{MSE}(\hat{a}_{i,pose}, a_{i,pose}) + \lambda \cdot \text{BCE}(\hat{a}_{i,gripper}, a_{i,gripper})$$

- $\hat{a}_{i,pose}$：predicted 6-DoF pose
- $a_{i,pose}$：GT 6-DoF pose
- $\hat{a}_{i,gripper}$：predicted gripper open/close（binary）
- $a_{i,gripper}$：GT gripper state
- $\lambda = 0.01$：weighting factor（参考 RoboVLM 设置，https://arxiv.org/abs/2412.14058）

**为什么 $\lambda$ 这么小**？因为 MSE 的 scale 通常比 BCE 大，需要降权 BCE 以免主导训练。从 Table 8 ablation 看到 $\lambda = 0.05$ 时 Avg. Len 从 3.91 降到 3.87，训练 oscillation 增加。

### Two-Stage Post-Training Pipeline

**Stage 1: Feature Space Alignment**

$$\min_{\Theta_D} \mathbb{E}_{(O_t, L, \hat{A}_t) \sim S} \left[ \mathcal{L}\Big(\hat{A}_t, \pi\big(\mathcal{V}(O_t, L) + \mathcal{D}(\text{MaxPooling}(\mathcal{G}(I_t^{3rd})))\big)\Big) \right]$$

- 冻结 $\Theta_V, \Theta_A, \Theta_G$（VLM、action predictor、ESM）
- 只训练 adapter $\Theta_D$
- $\mathcal{V}$ 是 VLM，$\mathcal{G}$ 是 ESM
- Adapter 使用 **zero-initialized final linear layer**，保证初期 spatial tokens 贡献接近零，不 disrupt pre-trained feature space

**Stage 2: Joint Feature Refinement**

$$\min_{\Theta_V, \Theta_D} \mathbb{E}_{(O_t, L, \hat{A}_t) \sim S} \left[ \mathcal{L}\Big(\hat{A}_t, \pi\big(\mathcal{V}(O_t, L) + \mathcal{D}(\text{MaxPooling}(\mathcal{G}(I_t^{3rd})))\big)\Big) \right]$$

- Unfreeze $\Theta_V, \Theta_D$，仍冻结 $\Theta_A, \Theta_G$
- VLM 隐式 refine semantic features 以 incorporate spatial cues
- 这种 phased approach 确保稳定收敛，避免 spatial features 在初期 overwhelming semantic representations

**这种 design 灵感来自 LLaVA** (https://arxiv.org/abs/2304.08485)，先训练 projector，再 joint training，避免 catastrophic forgetting。

## 6. 实验结果深度分析

### 6.1 CALVIN Benchmark (Table 1)

CALVIN (https://github.com/mees/calvin) 是 long-horizon, language-conditioned manipulation benchmark，包含 A/B/C/D 四个 scene splits，每个 24k demos。

**ABCD→D**（in-distribution）：
- FALCON: Avg. Len. 4.53
- RoboVLM: 4.49
- UP-VLA: 4.42
- GR-1: 4.21

**ABC→D**（zero-shot，最 challenging）：
- FALCON: 4.40（Avg. Len.）
- RoboVLM: 4.25
- Seer-Large: 4.28
- 3D Diffuser Actor: 3.35（使用 GT point cloud）
- 3DDP: 0.27（使用 GT point cloud）

**关键 insight**：FALCON 仅用 RGB 就超越了使用 GT point cloud 的方法，证明 implicit spatial integration 优于 explicit 3D inputs。

### 6.2 SimplerEnv (Table 2, 3)

SimplerEnv (https://arxiv.org/abs/2405.05941) 在 simulation 中重现 real-world scenes 评测。

**WidowX Robot**（Bridge V2 setup）：
- FALCON: avg 56.3%
- SpatialVLA: 42.7%
- RoboVLM: 37.5%
- OpenVLA: 1.0%

**Google Robot**：
- FALCON: avg 62.9%
- SpatialVLA: 55.3%
- RoboVLM: 51.7%
- RT-2-X (55B): 46.3%

特别值得注意的是 **"Open Top Drawer and Place Apple"** 这个 long-horizon task：
- FALCON: 41.7%
- RT-2-X (55B): 3.7%
- SpatialVLA: 0.0%
- OpenVLA: 0.0%

这个 task 需要 spatial reasoning + sequential planning，baseline 几乎全军覆没。

### 6.3 Real-World Experiments

**Base Tasks**（9 个 task suites，90 rollouts）：
- FALCON: 70.0% avg
- SpatialVLA: 44.4%
- 提升 25.6%

**Few-shot Adaptation**（80 trajectories training）：
- Simple setting: FALCON 比 second-best 高 27.5%
- Unseen Average: 高 27%
- 在 "open drawer and place bread" Unseen Object 变种，FALCON 达 80%，其他 near-zero

**Spatial Understanding Capability Evaluations**：
四个 task 测试 spatial perception：
- Height variation（cup 下方加 3cm blocks）
- Scale variation（block 5cm vs 3cm vs 4cm training）
- Spatial prompts

RoboVLM 等在 large objects 时碰撞，small objects 时提前 release，FALCON 表现 robust。

### 6.4 Modality Transferability (Table 5)

这是 FALCON 最 impressive 的特性之一：

| Setting | Avg. Len. |
|---------|-----------|
| FALCON (w/ rgb) ABCD→D | 4.08 |
| FALCON (test w/ d) ABCD→D | 4.09 |
| FALCON (w/ rgb-d) ABCD→D | 4.09 |
| FALCON (test w/o d) ABCD→D | 4.07 |

**关键观察**：
1. 训练时只用 RGB，test 时加 depth，性能从 4.08 → 4.09（提升）
2. 训练时用 RGB-D，test 时去 depth，性能从 4.09 → 4.07（小幅下降）
3. 这种 graceful degradation 来自 stochastic conditioning strategy

在 ABC→D 更 challenging setting：
- FALCON (w/ rgb): 3.91
- FALCON (test w/ d): 3.95（test 时加 depth 反而更好）
- FALCON (w/ rgb-d): 3.97
- FALCON (test w/o d): 3.95（甚至比 RGB-only training 还好）

**这个结果非常 striking**：RGB-D trained 模型去掉 depth 后，性能竟然高于 RGB-only trained 模型。说明 ESM 在 stochastic training 中学会了 robust spatial reasoning，即使没有 depth 也能从 RGB 推断 geometry。

### 6.5 ESM Depth Estimation (Table 6)

ESM 在 CALVIN 上的 zero-shot monocular depth estimation：

| Method | Depth | Camera | δ<1.25 (%) | Abs. Rel |
|--------|-------|--------|-------------|----------|
| VGGT | - | - | 91.33 | 8.53 |
| Ours | × | × | 90.91 | 8.61 |
| Ours | ✓ | × | 99.79 | 0.91 |
| Ours | ✓ | ✓ | 99.47 | 0.87 |

- $\delta < 1.25$：relative error ≤ 25% 的预测百分比
- Abs. Rel：average absolute relative error

**Insight**：ESM 在 RGB-only 时与 VGGT 相当（90.91 vs 91.33），但加 depth 后 δ 从 90.91 飙到 99.79（error 从 8.61 降到 0.91）。这证明 stochastic conditioning 不仅没损害 RGB-only performance，还让模型学会利用 depth 时大幅提升。

### 6.6 LSTM Head Ablation (Table 9)

| Config | Avg. Len. |
|--------|-----------|
| H=16, C=10 | 4.23 |
| H=16, C=5 | 3.99 |
| H=8, C=10 | 3.90 |

- H：history length
- C：chunk size

**Insight**：history length 比 chunk size 更重要。H=8 时训练初期 oscillation 严重，convergence 慢。这符合 long-horizon 任务需要丰富历史 context 的 intuition。

## 7. Ablation Studies 深度解析 (Table 4)

### Spatial Token Injection 策略

| Method | ABCD→D Avg. Len. | ABC→D Avg. Len. |
|--------|-------------------|------------------|
| FALCON_VLM-tokens | 4.00 | 3.79 |
| Cross-Attention | 3.98 | 3.68 |
| FiLM-Gated | 4.04 | 3.76 |
| FALCON (ours) | 4.08 | 3.91 |

**FALCON_VLM-tokens**：将 spatial tokens 直接注入 VLM（像 3D-VLA、SpatialVLA 做法）。结果显示 ABC→D Avg. Len. 从 3.91 降到 3.79，证明**将 fine-grained spatial features 注入 VLM 会 disrupt pre-trained semantic representation space**，损害 generalization。

### Fusion 策略对比

| Fusion | Avg. Len. (ABC→D) |
|--------|-------------------|
| Cross-Attention | 3.68 |
| FiLM-Gated | 3.76 |
| Element-wise Addition | 3.91 |

Element-wise addition 最好，因为它：
1. 无参数，不引入 optimization 难度
2. 不改变 VLM 的 representation 分布
3. 简单线性组合让 spatial features 作为 semantic features 的 additive refinement

## 8. Implementation Details 关键点

### Hyper-parameters (Table 7)

| Experiment | Predictor | Window | Chunk | VLM View | ESM View | Batch | LR | Total |
|-----------|-----------|--------|-------|----------|----------|-------|-----|-------|
| CALVIN | LSTM | 16 | 10 | Side+Wrist | Side | 128 | 2e-5/5e-5 | 5 Ep |
| SimplerEnv | MLP | 1 | 5 | Side | Side | 128 | 2e-5 | 150K |
| Real-World | MLP | 1 | 5 | Side | Side | 512 | 2e-5 | 30 Ep |

### ESM Training

- 数据：VGGT 的 datasets
- Batch：24 images（1-12 frames from random scene）
- AdamW optimizer
- Differentiated LR：backbone 1e-6，heads 1e-5
- Bernoulli $p = 66\%$
- 16 A100 GPUs，约 2 天

### Deploy

- GPU memory: ~12.8 GB
- Inference speed: ~57 Hz on RTX 4090
- Total parameters: 2.9B（VLM 1.6B + ESM 1.0B + Action Head）

## 9. Build Intuition: 为什么 FALCON 工作？

### 9.1 Brain Analogy 的深层含义

VLM 像 cerebrum 处理 high-level reasoning（"把香蕉放到红盘子上"中的 object recognition、language understanding），action head 像 cerebellum 处理 fine-grained motor control（精确的 gripper pose、approach angle）。

如果将 spatial tokens 注入 VLM（如 SpatialVLA），相当于让 cerebrum 同时承担 spatial perception，这会 dilute 它的 semantic reasoning 能力。FALCON 让 spatial information bypass VLM，直接到 action head，preserve VLM 的 language capability。

### 9.2 Stochastic Conditioning 的妙处

公式 (5) 的 stochastic conditioning 有两个作用：

1. **Modality robustness**：训练时见到 4 种 condition 组合（RGB-only, RGB+D, RGB+P, RGB+D+P），test 时任意 condition 都能 robust
2. **Implicit distillation**：当 depth/pose 可用时，模型学到 "如果 RGB → spatial structure" 的映射，相当于 self-distillation。这就是为什么 RGB-only training + test-time depth 也能提升性能

### 9.3 为什么 RGB-only 能超越 Point Cloud 方法？

3D Diffuser Actor、3DDP 使用 GT point cloud，但 FALCON RGB-only 在 CALVIN ABC→D 上超越它们。可能原因：

1. **Point cloud sparsity**：robotic point cloud 通常很 sparse，信息密度不如 spatial foundation model 学到的 dense 3D priors
2. **Foundation model 的 prior knowledge**：VGGT 在大规模 3D reconstruction data 上 pre-trained，有 strong geometric priors
3. **Feature richness**：spatial tokens 是 high-dimensional features（$D_s$ 维），比 raw point cloud 更 expressive
4. **Alignment with language**：FALCON 保留 VLM 的 language alignment，而 point cloud 方法需要额外 alignment

### 9.4 Two-Stage Training 的稳定性

Stage 1 的 zero-initialized adapter 是关键 trick。如果直接 random init，spatial tokens 会在初期产生 random noise，disrupt VLM feature space。Zero-init 让模型从 "纯 VLM" 开始，gradual learn spatial contribution。

这种 trick 类似 LoRA 的 zero-init (https://arxiv.org/abs/2106.09685)，保证训练初期等价于 baseline，逐步引入新能力。

## 10. Limitations 与 Future Directions

### 论文承认的 limitations

1. **Single third-view for ESM**：目前 ESM 只用 third-view image，wrist image 未充分利用。Table 10 显示加入 wrist image 仅小幅提升（4.08 → 4.10）
2. **Tabletop manipulation only**：未测试 mobile manipulation 或 whole-body control
3. **Static camera assumption**：ESM 假设 camera pose 已知或可估计

### 推测的 future directions

1. **Multi-view ESM**：整合 wrist + third-view 到 ESM，可能需要 cross-view attention
2. **Tactile integration**：加入 tactile sensors，进一步 enrich spatial grounding
3. **Video-based ESM**：当前是 frame-by-frame，video extension（如 CUT3R 的 recurrent formulation）能利用 temporal consistency
4. **3D-aware VLM**：探索让 VLM 本身 3D-aware，而非完全 bypass
5. **Real-world deployment scale**：从 11 tasks 扩展到 hundreds of tasks，测试 true generalist capability

## 11. 与相关工作对比

| Method | 3D Input | Modality Transfer | Alignment Risk | Architecture |
|--------|----------|-------------------|----------------|--------------|
| RT-2 | No | High | Low | 2D VLM |
| OpenVLA | No | High | Low | 2D VLM |
| SpatialVLA | Weak (learnable embed) | Low | High (concat w/ text) | 2D VLM + spatial embeds |
| PointVLA | Point cloud | Low | Medium | 3D encoder + VLM |
| 3D-VLA | Point cloud | Low | High (3D tokens in VLM) | 3D VLM |
| Evo-0 | Weak | Low | High | 2D VLM |
| **FALCON** | **Optional (depth/pose)** | **High** | **Low (in action head)** | **2D VLM + ESM + Action Head** |

## 12. 关键 References 与 Web Links

- **Project Page**: https://falcon-vla.github.io/
- **VGGT**: https://arxiv.org/abs/2403.05171 (CVPR 2025)
- **DUSt3R**: https://arxiv.org/abs/2312.14132 (CVPR 2024)
- **MASt3R**: https://arxiv.org/abs/2406.09756 (ECCV 2024)
- **CUT3R**: https://arxiv.org/abs/2412.04403 (CVPR 2025)
- **Kosmos-2**: https://arxiv.org/abs/2306.14824
- **DINOv2**: https://arxiv.org/abs/2304.07193
- **CALVIN**: https://github.com/mees/calvin
- **SimplerEnv**: https://arxiv.org/abs/2405.05941
- **Open X-Embodiment**: https://arxiv.org/abs/2310.08864
- **SpatialVLA**: https://arxiv.org/abs/2501.15830
- **RoboVLM**: https://arxiv.org/abs/2412.14058
- **OpenVLA-OFT**: https://arxiv.org/abs/2502.19645
- **3D Diffuser Actor**: https://arxiv.org/abs/2402.10885
- **3D Diffusion Policy**: https://arxiv.org/abs/2403.03954
- **PointVLA**: https://arxiv.org/abs/2503.07511
- **GeoVLA**: https://arxiv.org/abs/2508.09071
- **3D-VLA**: https://arxiv.org/abs/2403.09631 (ICML 2024)
- **VLM-3R**: https://arxiv.org/abs/2505.20279
- **Spatial-MLLM**: https://arxiv.org/abs/2505.23747
- **Evo-0**: https://arxiv.org/abs/2507.00416
- **UP-VLA**: https://arxiv.org/abs/2501.18867
- **GR-1**: https://arxiv.org/abs/2312.13171
- **Robo-Flamingo**: https://arxiv.org/abs/2311.11080
- **RT-1**: https://arxiv.org/abs/2212.06817
- **RT-2**: https://arxiv.org/abs/2307.15818
- **TraceVLA**: https://arxiv.org/abs/2412.10345
- **Octo**: https://arxiv.org/abs/2405.12213
- **π0**: https://arxiv.org/abs/2410.24164
- **π0.5**: https://arxiv.org/abs/2504.16054
- **LLaVA**: https://arxiv.org/abs/2304.08485
- **LoRA**: https://arxiv.org/abs/2106.09685
- **Rochefort et al. (cerebellum)**: https://www.science.org/doi/10.1126/science.1207403
- **Figure AI Helix**: https://www.figure.ai/news/helix

## 13. 总结

FALCON 的核心贡献在于提出了一个 **decoupled spatial integration paradigm**：

1. **Spatial representation**：通过 spatial foundation model (VGGT-based ESM) 提供强 geometric priors
2. **Modality transferability**：通过 stochastic conditioning 实现跨 modality robust 性能
3. **Alignment preservation**：通过将 spatial tokens 注入 action head 而非 VLM，保护 pre-trained vision-language alignment

这三点共同解决了 prior 3D-VLA 方法的三个 fundamental limitations。实验结果在 CALVIN、SimplerEnv、11 个 real-world tasks 上全面 SOTA，特别是 long-horizon 和 spatially demanding tasks 上优势显著。

**最重要 takeaway**：当你想给 VLM 加新能力时，要小心是否 disrupt 它的 pre-trained alignment。FALCON 的 brain-inspired division of labor 提供了一个 principled solution——让新能力 bypass 主 backbone，直接作用于决策层。这种思路应该可以推广到其他 multimodal integration 场景，比如 tactile feedback、audio signals、proprioception 等。
