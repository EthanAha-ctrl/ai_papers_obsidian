---
source_pdf: Geometric Action Model for Robot Policy Learning.pdf
paper_sha256: 19b79e58aac413f80c3eac7d34c7c570abc921ace4ba79a42b12aeda5ff61a42
processed_at: '2026-08-19T09:29:27-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 GAM

## 一句话版本

这篇paper说：现在robot policy都在2D图里做推理，但manipulation本质是3D问题——你抓个杯子，得知道杯子离gripper多远、什么角度、有没有遮挡。这些信息在RGB像素里是隐含的，模型得自己猜。GAM说：既然我们已经有Depth Anything V3 (DA3) [13] 这种能从多视角RGB直接回归3D几何的foundation model，为什么不直接把它当成policy的backbone？让action在3D-aware的latent space里产生，而不是在2D image patch的latent space里产生。

就这么个核心想法。

---

## 问题在哪

先看现状。现在manipulation policy主要两派：

**VLA派**（OpenVLA [1]、π0.5 [6]）：拿vision-language model当backbone，输入image+language，输出action token。backbone是在web image-text上pretrain的，representation偏semantic——"这是个杯子"它懂，但"杯子相对gripper在哪、距离多少cm、从哪个角度抓"它得从2D像素里infer。Camera稍微一动，2D appearance变了，policy就懵。

**WAM派**（Cosmos Policy [3]）：拿video generation model当backbone，predict future frames + action。问题一样——video backbone也是2D pixel space的，没有显式depth、scale信息。而且diffusion要multi-step denoise，慢，Cosmos Policy一次forward要382ms。

还有人意识到这问题，搞"geometry-aware VLA"（Spatial Forcing [17]、ROCKET [18]）：把GFM当frozen feature extractor，把它的features distill进VLA backbone。但GFM只提供"static feature prior"，action decoding还是发生在VLA自己的2D-aware latent space里。相当于你请了个3D专家在旁边给建议，但最后拍板的还是那个2D思维的人。

GAM的insight：别distill了，直接把GFM本身当policy backbone。

---

## 怎么做的

DA3这种GFM的forward本来是这样：输入多视角RGB，经过40层ViT-Giant transformer，输出per-pixel depth、3D point map、camera pose。中间层的hidden states已经编码了multi-view-consistent的3D structure。

GAM干了一件事：**在第12层把GFM切开**。

- 第0-12层当observation encoder，提取每个timestep的geometric features
- 第13-39层当decoder，本来是用来decode 3D geometry的
- 中间塞一个12层的causal transformer（叫"future predictor"）

这个future predictor干三件事：
1. 接收当前和历史帧的GFM latent tokens
2. 接收language instruction（用frozen T5 [48]编码）
3. 接收proprioception和previous action

然后用block-causal attention（每个timestep只能看自己和之前的timestep，不能看未来）predict两类东西：
- 下一帧的GFM latent tokens（geometry prediction）
- 下一步的action token（action prediction）

这俩在同一个autoregressive sequence里产生，共享同一个backbone forward。Action token出来后复制V份（V是视角数），跟geometry tokens一起送进deep GFM blocks做cross-view fusion和refinement。最后deep blocks的输出分两个head：一个出action chunk（8步），一个出future depth map。

关键trick：action token必须过deep GFM blocks才能出最终action。Ablation Table 14显示，直接用predictor输出的action token vs 让它过deep blocks再decode，后者在LIBERO-Plus Object上从84.1%到89.7%。多出来的5.6个点主要来自camera perturbation鲁棒性。这说明deep blocks的global attention把action token"几何化"了——action和3D structure在同一个forward pass里互相refine。

---

## 为什么在第12层切

DA3的40层里，前12层是frame-wise attention（每个view内自己attend），第13层开始是global attention（跨view attend）。第12层正好是"per-view feature已经extract够，但还没cross-view fuse"的边界点。

在frame-wise阶段插predictor，future prediction缺cross-view信息；在global阶段太晚插，future tokens没足够deep blocks去refine。Table 3的ablation印证：$L_s=0$崩（5.4%），$L_s=12$最好（99.6%），$L_s=27$以后直接崩（1.2%）。这种sensitivity跟LLM里adapter插入位置选择是同一种道理。

---

## Training怎么监督

三个loss一起训：

**Action loss**：predict的action chunk跟expert demo的$\ell_1$距离。用$\ell_1$不用$\ell_2$是因为action distribution可能multi-modal，$\ell_2$会regress到mean导致模糊行为。

**Future-feature loss**：predictor输出的future GFM latent tokens，跟frozen GFM在ground-truth next frame上跑出的latent tokens做$\ell_1$。这是distillation-style supervision——让predictor学会"执行action后世界会变成什么样"。Teacher是frozen的，gradient只回传到predictor，不污染GFM weights。

**Future-depth loss**：把predict的future latent tokens送进GFM的DPT head decode出future depth map，跟ground-truth future depth比。用scale-invariant loss + gradient matching penalty（Eigen & Fergus那套）。仿真里ground-truth depth直接读simulator，real-world里用frozen GFM出的pseudo-depth当label。

三个loss权重 $\lambda_{\text{act}}=3, \lambda_{\text{feat}}=1, \lambda_{\text{depth}}=3$。

Ablation Table 2的关键发现：pretrain对OOD鲁棒性是决定性的。LIBERO原版分数差不多，但LIBERO-Plus从89.7%掉到73.4%（掉16个点）。future-prediction losses在没pretrain时提供strong geometric supervision，从50.0%救到73.4%。有pretrain后，去掉$\mathcal{L}_{\text{depth}}$或$\mathcal{L}_{\text{feat}}$影响不大，因为几何dynamics已经encode进backbone了。

---

## 效果

**LIBERO-Plus**（7种OOD perturbation：camera、robot、language、light、background、noise、layout）：

| Method | Size | Plus Total | Cam. Perturbation |
|--------|------|-----------|-------------------|
| OpenVLA-OFT [2] | 7B | 69.6 | 56.4 |
| π0.5 [6] | 3.3B | 84.6 | 72.0 |
| Cosmos Policy [3] | 2B | 82.4 | 73.4 |
| π0.5+ROCKET [18] | 3.3B | 47.5 | 30.9 |
| **GAM** | **1.4B** | **85.5** | **83.1** |

Camera perturbation下GAM是83.1%，第二名π0.5是72.0%，高11个点。这正是3D geometric prior的价值——外部相机平移85cm+旋转45°，2D appearance全变了，但GFM-encoded的multi-view-consistent 3D structure是camera-invariant的。

**速度**：

| Method | Latency |
|--------|---------|
| Cosmos Policy (diffusion) | 382ms |
| OpenVLA-OFT | 77.8ms |
| π0.5 | 29.2ms |
| **GAM** | **6.9ms (≈145Hz)** |

GAM快是因为single feed-forward pass，无diffusion denoising loop，无autoregressive action token decoding。55x faster than Cosmos Policy。

**Real-world**：四个task（pick-place、stack、pot-pan、insert cube），每个20 trials（10 ID + 10 OOD camera）。GAM在OOD下基本不掉点，π0.5+Spatial Forcing在OOD下严重退化。Sim-to-real transfer时geometric prior依然有效。

---

## 我的直觉

**为什么这个work**：manipulation的本质是spatial reasoning。你要知道物体在哪、gripper在哪、两者相对位姿如何。这些信息在RGB里是ill-posed的——单张2D图无法recover depth和scale。VLA逼着模型从2D appearance里猜这些3D量，所以camera一动就崩。GAM直接用GFM的3D-aware latent space，representation本身已经是metric-aware、camera-invariant的，policy不需要再学这层abstraction。

**为什么split-and-insert比distill好**：distillation（Spatial Forcing、ROCKET路线）只能传递"feature prior"，action decoding还在VLA自己的2D latent space里。GAM让action token直接过GFM的deep global attention layers，被3D structure"洗礼"。Ablation Table 14验证了这点——跳过deep blocks直接出action，LIBERO-Plus掉5.6个点。

**为什么diffusion在manipulation里overkill**：diffusion的优势在high-dim multi-modal output distribution（image generation，百万维pixel space）。Action chunk就7维×8步=56维，远没到需要diffusion去mode-cover的程度。Single-pass regression在这个dimensionality下够用，而且快55倍。Cosmos Policy的multi-step denoising在manipulation里是wasted computation。

**几个我没想明白的地方**：
- 开环执行8步action chunk后才重新观察。中间如果world state偏离prediction（比如抓取滑了、物体被碰了），policy是瞎的。这是所有action chunking policy的通病，GAM没特别处理。
- 多view是hard requirement。很多real robot只有wrist camera + 1 external，但GFM期望multi-view。单view下GFM给的latent质量如何？paper没测。
- Long-horizon task表现差（RoboCasa里PnPMicrowaveToCounter 11.3%）。Future prediction在长horizon下退化——predict下一帧容易，predict 20帧后很难。
- Frozen T5做language encoder是明显bottleneck。复杂instruction（hierarchical、conditional）可能跟不上VLA那种LLM-based language understanding。

**整体评价**：核心idea干净——repurpose rather than attach。把GFM整个forward pathway变成policy的forward pathway，action在3D-aware latent里产生。1.4B模型打7B VLA和3.3B VLA，camera perturbation下尤其显著，55倍加速。这是个architectural-level的innovation，比"给VLA加个3D module"那种incremental work高一个层次。

---

## Web References

- GAM project page: https://cvlab-kaist.github.io/Geometric-Action-Model
- Depth Anything V3: https://depth-anything-v3.github.io/
- VGGT: https://vgg-t.github.io/
- LIBERO: https://libero-project.github.io/
- LIBERO-Plus: https://arxiv.org/abs/2510.13626
- OpenVLA: https://openvla.github.io/
- π0.5: https://arxiv.org/abs/2504.16054
- Cosmos Policy: https://arxiv.org/abs/2601.16163
- Spatial Forcing: https://arxiv.org/abs/2510.12276
- ROCKET: https://arxiv.org/abs/2602.17951
- RoboCasa: https://robocasa.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- DPT: https://arxiv.org/abs/2103.13413
- T5: https://arxiv.org/abs/1910.10683

---

# GAM (Geometric Action Model) 深度技术解析

## 1. Core Motivation 与思想脉络

Andrej你好，这篇paper试图回答一个很根本的问题：**robot policy 应该在什么 representation space 里做推理？** 现有 VLA (Vision-Language-Action) models 比如 OpenVLA [1] 和 π0.5 [6] 在 2D image patch latent space 里出 action，World Action Models (WAMs) 比如 Cosmos Policy [3] 在 video diffusion latent space 里出 action，但两者都缺乏显式的 3D geometry prior — depth、scale、occlusion 都隐含在 monocular cues 里，policy 自己得从 2D 端去 disentangle 这堆 3D 物理量。

GAM 的核心 insight 非常干净：**Geometric Foundation Models (GFMs) 比如 VGGT [14] 和 Depth Anything V3 (DA3) [13] 已经学会从 multi-view RGB 直接回归 dense 3D geometry + camera pose，为什么不直接把这个 GFM 当成 policy 的 shared backbone，让 perception / temporal prediction / action decoding 三个 stage 都跑在 GFM 的 latent space 里？**

这点和 Spatial Forcing [17]、ROCKET [18] 这种 "geometry-aware VLA" 路线有本质区别：后者把 GFM 当 frozen feature extractor，把它的 features distill 进 VLA backbone。GAM 则把 GFM 在中间 layer 切开，往里塞一个 causal future predictor，让 GFM 的 deep blocks 直接承担 action decoding 的角色。

Project page: https://cvlab-kaist.github.io/Geometric-Action-Model

---

## 2. Preliminaries: Geometric Foundation Model 回顾

先把 GFM 的 forward 形式化清楚，后面 GAM 的所有 modification 都建立在这个 backbone 之上。

**输入**：V 个 view 的 RGB 图像 $\mathcal{Z} = \{I_v\}_{v=1}^{V}$，每个 $I_v \in \mathbb{R}^{3 \times h \times w}$。

**Patch tokenize**：每个 view 切成 $P$ 个 non-overlapping patch（patch size $p \times p$），通过 patch embedding 投影成 per-view token sequence：

$$\mathbf{z}_v^{(0)} = [\mathbf{c}_v, \mathbf{x}_v^1, \ldots, \mathbf{x}_v^P] \in \mathbb{R}^{(1+P) \times d}$$

变量含义：
- $\mathbf{c}_v \in \mathbb{R}^d$：per-view camera token（编码该 view 的 intrinsics/extrinsics 信息，类似 ViT 的 CLS token 角色，但专门承载 camera pose）
- $\mathbf{x}_v^j \in \mathbb{R}^d$：第 $j$ 个 patch 的 token
- $d$：hidden dimension（DA3-Giant 里 $d \approx 1024$ 量级）
- 上标 $(0)$ 表示第 0 层（embedding 层）

跨 view 拼接成完整输入：$\mathbf{Z}^{(0)} = [\mathbf{z}_1^{(0)}, \ldots, \mathbf{z}_V^{(0)}] \in \mathbb{R}^{V(1+P) \times d}$。

**M 层 transformer blocks** $\{f^{(m)}\}_{m=1}^{M}$，每个 block 用两种 attention mode 之一：
- **Frame-wise attention** $f_{\text{frame}}^{(m)}$：只在单 view 内的 $(1+P)$ 个 token 上做 self-attention，不跨 view。处理单 image 的 local geometric features。
- **Global attention** $f_{\text{global}}^{(m)}$：在所有 $V(1+P)$ token 上做 self-attention，跨 view 融合，类似 multi-view stereo 里的 cost volume 构建。

Hidden state evolution：

$$\mathbf{Z}^{(m)} = f^{(m)}(\mathbf{Z}^{(m-1)}), \quad f^{(m)} \in \{f_{\text{frame}}^{(m)}, f_{\text{global}}^{(m)}\}$$

**DPT head 解码**：从多个中间层 $\mathcal{S} = \{m_1, m_2, m_3, m_4\}$ 抽 hidden states $\mathbf{Z}^{(m^*)}$，送进 DPT [46] decoder 出 per-pixel depth $D_v \in \mathbb{R}^{h \times w}$、point maps $P_v \in \mathbb{R}^{3 \times h \times w}$，以及 camera intrinsics $K_v \in \mathbb{R}^{3\times 3}$ 和 extrinsics $\xi_v \in SE(3)$。

这里的关键是：GFM 的中间层 hidden states 已经编码了 multi-view-consistent 的 3D structure，而不只是 2D appearance。GAM 的整个设计就是要让 action 也跑在这个 latent space 里。

---

## 3. GAM 架构三段式

### 3.1 Problem Formulation

每个 timestep $t$ 输入：
- 多 view RGB：$o_t = \{I_{v,t}\}_{v=1}^{V}$
- 本体感受：$s_t \in \mathbb{R}^{d_s}$（joint config + end-effector pose，$d_s=7$）
- 自然语言指令 $\ell$（episode 内不变）
- 历史窗口 $H$：$\{o_{t-H+1}, ..., o_t\}$、$\{s_{t-H+1}, ..., s_t\}$、$\{a_{t-H}, ..., a_{t-1}\}$

输出：action chunk $\hat{\boldsymbol{a}}_t \in \mathbb{R}^{C \times d_a}$，$C=8$，$d_a=7$（delta-pose / joint commands），开环执行 $C$ 步再重新观察。

Policy 形式化：

$$\pi_\theta: \left(\{o_{t-H+1:t}\}, \{s_{t-H+1:t}\}, \{a_{t-H:t-1}\}, \ell\right) \mapsto \hat{a}_{t-H+1}, \ldots, \hat{a}_{t-H+1}$$

### 3.2 Stage 1: Observation Encoder (浅层 GFM reuse)

选一个 split layer $L_s$，把整个 GFM 切成两半：

$$E_{\le L_s} = f^{(L_s)} \circ \cdots \circ f^{(1)}, \qquad D_{>L_s} = f^{(M)} \circ \cdots \circ f^{(L_s+1)}$$

- $E_{\le L_s}$：observation encoder（浅层 + 中层）
- $D_{>L_s}$：deep decoder（深层）

**$L_s$ 的选择约束**：必须深到能提取足够 rich 的 visual features，又要浅于 DPT head 最早用的层（$L_s < m_1$），否则 predicted future tokens 无法被 DPT 解码成 future geometry。Ablation Table 3 显示 $L_s = 12$ 是 sweet spot — 这正好是 DA3 里 frame-wise attention 切换到 global attention 的边界，意味着：浅层 frame-wise 做 per-view feature extraction，深层 global 做 cross-view fusion。把 future predictor 插在 frame-wise 和 global 之间，让 predictor 输出的 future tokens 经过 global attention "被立体化"，非常巧妙。

对 context window 里每个 timestep $t'$ 独立过 $E_{\le L_s}$，得到：

$$\mathbf{Z}_{t'}^{(L_s)} \in \mathbb{R}^{V(1+P) \times d}, \quad \forall t' \in \{t-H+1, \ldots, t\}$$

### 3.3 Stage 2: Causal Future Predictor (插入的中间层)

这是 GAM 的核心组件，是一个 12-layer 的 causal transformer $g_\phi$，width $d_g = 1024$。

**Token 构造**：每个 timestep $t'$ 把三类信息打包成 block：

$$\mathbf{p}_{t'} = \psi_s(s_{t'}) \quad \text{(proprioception token)}$$
$$\mathbf{q}_{t'} = \psi_a(a_{t'-1}) \quad \text{(previous action token)}$$
$$\mathbf{U}_{t'} = [\mathbf{p}_{t'}; \mathbf{q}_{t'}; \mathbf{Z}_{t'}^{(L_s)}]$$

变量含义：
- $\psi_s, \psi_a$：轻量 MLP projection layer，把 $s_{t'} \in \mathbb{R}^{d_s}$ 和 $a_{t'-1} \in \mathbb{R}^{d_a}$ 投到 hidden dim $d$
- $\mathbf{p}_{t'}$：robot state token（一个，per timestep）
- $\mathbf{q}_{t'}$：action history token（一个，per timestep）
- $\mathbf{Z}_{t'}^{(L_s)}$：$V(1+P)$ 个 GFM latent tokens（geometry slots）

加上 frozen T5 [48] 编码的 language tokens $\mathbf{L}_\ell$，完整输入：

$$\mathbf{X} = [\mathbf{L}_\ell; \mathbf{U}_{t'-H+1}; \ldots; \mathbf{U}_{t'}]$$

**Block-causal self-attention**（Figure 3(b)）：每个 timestep block 只能看到自己以及之前 timesteps 的 block，看不到未来的 block，避免 future leakage。Language tokens 在最前面，所有 timestep 都能 attend 到。

**输出读取**：predictor 最后一层，从 sequence slots 里读出两类预测：
- **Geometry slots** → $\tilde{\mathbf{Z}}_{t'+1}^{(L_s)}$：下一个 timestep 的 GFM latent tokens（$V(1+P)$ 个）
- **Action slot**（即 $\mathbf{q}_{t'}$ 对应的 output slot）→ $\tilde{\mathbf{a}}_{t'} \in \mathbb{R}^d$：next action token（一个，per timestep）

类比 LLM 的 next-token prediction — action token 就是被预测的 "next token"。这个设计让 action 和 spatial representation 在同一个 autoregressive sequence 里 tight 交互，action 必须通过读写 geometry latent 来产生。

### 3.4 Stage 3: Feature Propagation & Action Decoding

predictor 出来的 $\tilde{\mathbf{a}}_{t'}$ 是一个 token，但 $D_{>L_s}$ 要处理 $V$ 个 views，所以**复制 V 份**：

$$\tilde{\mathbf{a}}_{v, t'} = \tilde{\mathbf{a}}_{t'}, \quad \forall v \in \{1, \ldots, V\}$$

每 view 的 geometry tokens 拼上对应 action token，送进 deep blocks：

$$\tilde{\mathbf{Z}}_{t'+1}^{(M)} = \left(f^{(M)} \circ \cdots \circ f^{(L_s+1)}\right)\left(\left[[\tilde{\mathbf{Z}}_{1,t'+1}^{(L_s)}; \tilde{\mathbf{a}}_{1,t'}], \ldots, [\tilde{\mathbf{Z}}_{V,t'+1}^{(L_s)}; \tilde{\mathbf{a}}_{V,t'}]\right]\right)$$

**Causal mask 扩展**：deep blocks 里的 global attention 层也沿用 predictor 的 causal mask，避免 future frame 信息泄漏到当前 frame 的 prediction。

**双 head 输出**：
- $h_{\text{act}}$：轻量 head，aggregate context window 里的 action tokens → 回归 action chunk $\hat{a}_{t'} \in \mathbb{R}^{C \times d_a}$
- $h_{\text{depth}}$：原始 GFM 的 DPT head，decode geometry tokens → future depth maps $\tilde{D}_{t'+1}$

**直觉**：GFM 的 deep blocks 原本就是用来把 shallow features decode 成 3D geometry 的。GAM 让这些 blocks 同时承担 action refinement — action token 进入 deep blocks 后会被 global attention "几何化"，跟 future geometry tokens 在同一个 forward pass 里互相 refine。Ablation Table 14 印证了这点：直接用 predictor 输出的 action token 做 supervision，LIBERO-Plus Object 是 84.1%；让 action token 过 deep blocks 再 decode，能到 89.7% — 多了 5.6 个百分点，主要来自 camera perturbation 鲁棒性。

---

## 4. Training Objective 详解

三任务联合 loss：

$$\mathcal{L}_{\text{total}} = \lambda_{\text{act}}\mathcal{L}_{\text{act}} + \lambda_{\text{feat}}\mathcal{L}_{\text{feat}} + \lambda_{\text{depth}}\mathcal{L}_{\text{depth}}$$

权重 $\lambda_{\text{act}} = 3$, $\lambda_{\text{feat}} = 1$, $\lambda_{\text{depth}} = 3$。

**1) Action loss**（$\ell_1$ regression）：

$$\mathcal{L}_{\text{act}} = \sum_{t' \in \mathcal{H}} \|\hat{a}_{t'} - a_{t'}\|_1$$

- $\hat{a}_{t'}$：policy 预测的 action chunk
- $a_{t'}$：expert demonstration 的 ground-truth action chunk
- $\mathcal{H} = \{t-H+1, \ldots, t\}$：context window

用 $\ell_1$ 而非 $\ell_2$ 是因为 action distribution 往往 multi-modal（同一个 observation 可能有多种合理 action），$\ell_2$ 会让模型 regression 到 mean，导致模糊行为。

**2) Future-feature loss**（让 predictor 学会几何 dynamics）：

$$\mathcal{L}_{\text{feat}} = \sum_{t' \in \mathcal{H}} \|\tilde{\mathbf{Z}}_{t'+1}^{(L_s)} - \mathbf{Z}_{t'+1}^{(L_s)}\|_1$$

- $\tilde{\mathbf{Z}}_{t'+1}^{(L_s)}$：predictor 输出的 future latent tokens
- $\mathbf{Z}_{t'+1}^{(L_s)}$：**frozen GFM** 在 ground-truth next frame 上跑出的 latent tokens（teacher）

这是 distillation-style 的 supervision — 让 predictor 学会预测下一帧的 GFM latent。Teacher 是 frozen 的，所以 $\mathcal{L}_{\text{feat}}$ 只回传到 predictor $g_\phi$，不污染 GFM 的 weights。这正是 GAM 把 GFM 切开的核心好处之一：可以在 latent space 里直接做 future prediction，而不需要 decode 成 RGB 再比较。

**3) Future-depth loss**（geometric grounding）：

$$\mathcal{L}_{\text{depth}} = \mathcal{L}_{\text{SI}}(\tilde{D}_{t'+1}, D_{t'+1}) + \mathcal{L}_{\text{grad}}(\tilde{D}_{t'+1}, D_{t'+1})$$

其中：
- $\tilde{D}_{t'+1} = h_{\text{depth}}(\tilde{\mathbf{Z}}_{t'+1}^{(m^*)})$：predicted future depth
- $D_{t'+1}$：ground-truth future depth（仿真里直接读，real-world 里用 frozen GFM 出的 pseudo-depth）
- $\mathcal{L}_{\text{SI}}$：scale-invariant loss（Eigen & Fergus 2014 那一套，对 depth 的整体 scale 不敏感）
- $\mathcal{L}_{\text{grad}}$：gradient matching penalty，让 depth edges 对齐

**Inference 阶段**：用 KV cache 维护历史 context，每 step 只 forward 一次新的 observation 和 previous action，单次 feed-forward pass 出 action chunk。这就是 6.9ms latency 的来源。

---

## 5. 实验结果深度分析

### 5.1 LIBERO / LIBERO-Plus 主表（Table 1）

LIBERO 已 saturated，大家分数都接近 100%。真正有信息量的是 LIBERO-Plus — 它引入了 7 种 OOD perturbation：camera viewpoint (Cam.)、robot embodiment (Robot)、language paraphrase (Lang.)、lighting (Light)、background (BG)、visual noise (Noise)、spatial layout (Layout)。

**GAM 关键数据**：
- Model size: **1.4B**（最小，比 OpenVLA-OFT 7B 小 5x，比 π0.5 3.3B 小 2.4x）
- LIBERO-Plus total: **85.5%**（最高）
- Cam. perturbation: **83.1%**（比第二名 π0.5 的 72.0% 高 11.1 个百分点，paper claim 是 +9.7%p，应该是和某个特定 baseline 比）
- Drop from LIBERO to Plus: **12.1**（最小，π0.5 是 12.3，OpenVLA-OFT 是 27.5）

这个 camera perturbation 鲁棒性是 GAM 整个 paper 的"signature result"，直接印证了 3D geometric prior 的价值 — 当外部相机被旋转 45°、平移 85cm 时，2D-based policy 几何上 invalid，但 GFM-encoded 的 multi-view-consistent 3D structure 是 camera-invariant 的。

### 5.2 Inference Speed（Table 4）

| Method | Size | Time |
|--------|------|------|
| OpenVLA-OFT [2] | 7B | 77.8ms |
| π0.5 [6] | 3.3B | 29.2ms |
| Cosmos-Policy [3] | 2B | 382.4ms |
| **GAM** | 1.4B | **6.9ms** |

6.9ms ≈ 145 Hz control frequency，55x faster than Cosmos Policy（diffusion-based 需要 multi-step denoising）。GAM 快的原因有二：(1) single feed-forward pass，无 diffusion denoising loop；(2) 不像 VLA 需要 autoregressively decode 多个 action token，GAM 的 action token 是一次 forward 出来的。

Table 8 里还有更细的 latency breakdown：不开 CUDA Graphs 时 GAM 17.5ms，开了 6.9ms。π0.5 在 PyTorch 实现下 29.2ms。

### 5.3 RoboCasa-Kitchen（Table 10, 11）

24 个 kitchen manipulation tasks，整体 SR 69.4%，超过 Cosmos Policy 67.1%、FLARE 66.4%。

注意几个特别差的 task：
- `PnPMicrowaveToCounter`: 11.3%
- `TurnOffStove`: 30.0%
- `CoffeeSetupMug`: 33.7%

这些 task 涉及 articulated object interaction 或者 precise placement，几何预测难度大。paper 没有针对这些做深入 failure analysis，是个 limitation。

### 5.4 Real-world（Figure 4）

四个 task，每个 20 trials（10 ID + 10 OOD）。GAM 在 OOD camera perturbation 下基本不掉点，π0.5 + Spatial Forcing 在 OOD 下严重退化。这跟 simulation 结果一致，说明 GFM-based 的几何 prior 在 sim-to-real transfer 时也保持有效。

---

## 6. Ablation Insights

### 6.1 Component Ablation（Table 2）

最关键的几行：

| Pretrain | $\mathcal{L}_{\text{depth}}$ | $\mathcal{L}_{\text{feat}}$ | H | Orig. SR | Plus SR |
|----------|------|------|---|----------|---------|
| ✓ | ✓ | ✓ | 1 | 99.6 | 89.7 |
| ✗ | ✓ | ✓ | 1 | 98.4 | 73.4 |
| ✗ | ✗ | ✗ | 1 | 93.6 | **50.0** |

**Insight 1**：Pretrain 是 OOD 鲁棒性的关键。LIBERO 原版分数差不多，但 LIBERO-Plus 从 89.7 掉到 73.4，掉 16 个点。

**Insight 2**：没 pretrain 时，future-prediction losses（$\mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{feat}}$）提供了 strong geometric supervision，从 50.0 救到 73.4。这意味着 future prediction 不只是 test-time imagination 的工具，本身就是强大的 training-time regularizer — 跟 FLARE [40]、UniVLA [33] 等 paper 的发现一致。

**Insight 3**：有 pretrain 时，去掉 $\mathcal{L}_{\text{depth}}$ 或 $\mathcal{L}_{\text{feat}}$ 影响不大（98.4/89.0 vs 99.6/89.7），因为几何 dynamics 已经 encode 进 backbone 了。但 inference 时仍用 future prediction（imagination）来出 action — paper 没单独报告 inference 时关掉 future predictor 的实验，这块没说清楚。

**Insight 4**：H=1 足够，H=2/H=4 反而略差（97.2/84.4, 98.2/85.1）。跟 Wen et al. [53]、de Haan et al. [54] 的 observation 一致：长 history 容易引入 spurious correlation，导致 causal confusion。

### 6.2 Split Layer $L_s$（Table 3）

| $L_s$ | Orig. | Plus |
|-------|-------|------|
| 0 | 5.4 | 1.8 |
| **12** | **99.6** | **70.1** |
| 19 | 95.6 | 63.4 |
| 27 | 1.2 | 1.6 |
| 33/39 | 0.0 | 0.0 |

这个表非常有意思：
- $L_s=0$：predictor 接在 patch embedding 后面，太早，没有 visual feature，崩。
- $L_s=12$：刚好是 frame-wise → global attention 的过渡点。Predictor 输出的 future tokens 经过 $D_{>L_s}$ 的 global attention 被立体化。
- $L_s=19$：还是 competitive，但略差。说明 frame-wise attention 已经足够提取 visual feature，但 7 层 global attention 不够 deep blocks 去 refine。
- $L_s \geq 27$：太晚，DPT head 用的中间层 $m_1$ 之前没有足够 layers 给 future tokens 做 refinement，崩。

**Intuition**：$L_s$ 必须卡在 "feature 已经足够 semantic，但 deep decoder 还能 refine" 的 sweet spot。这跟 LLM 里 adapter 插入位置选择、diffusion 里 timestep 选择是同一种 hyperparameter sensitivity。

注意：Table 3 这里 Plus 只有 70.1，但主表是 85.5。是因为 Table 3 的 ablation **关掉了 $\mathcal{L}_{\text{depth}}$**（paper 说 "we exclude future-depth loss in this experiment because it is not equally applicable to all split layers"），所以这是 isolated $\mathcal{L}_{\text{feat}}$ 的效果。

### 6.3 Direct Action Supervision（Table 14）

| Variant | Orig. | Plus |
|---------|-------|------|
| Direct-action (跳过 deep blocks) | 98.4 | 84.1 |
| GAM (过 deep blocks) | 99.6 | 89.7 |

让 action token 直接从 predictor 出来 vs 让它过 deep GFM blocks 再 decode，后者多了 5.6 个点（Plus）。这验证了 deep blocks 的几何 refine 作用 — action 必须被 global attention 几何化才能 robust。

### 6.4 Attention Visualization（Figure 11）

action token 在 GFM 不同层的 attention map 显示，中间层 attend 到 manipulated object 和 contact region 附近。这是 qualitative evidence 说明 action 真的 "看到" 了 3D geometry。

---

## 7. Model Size Breakdown（Table 9）

| Module | Params | Trainable |
|--------|--------|-----------|
| backbone (ViT-Giant, 40 blocks) | 1136.5M | 765M (blocks 13-39 trainable, 0-12 frozen) |
| DPT head | 50.1M | 0 (frozen) |
| **Causal Future Predictor** | **210.2M** | 210.2M |
| action head | 8.0M | 8.0M |
| **Total** | **~1404.8M** | **~983.2M** |

只有 ~70% 参数 trainable。Causal Future Predictor 210M 是新加的，占 trainable params 的 ~21%。整体 1.4B，是 OpenVLA-OFT (7B) 的 1/5、π0.5 (3.3B) 的 1/2.4。

---

## 8. Training Details

### Pre-training
- Data mixture：OpenX-Embodiment [51] 72% + MimicGen [50] 18% + RoboCasa365 [49] 10% = 784K trajectories
- 硬件：64 × NVIDIA GH200 GPU，batch size 1024，~96 hours
- Backbone：DA3-Giant [13] fine-tuned on Track4World [47]
- Optimizer：AdamW，constant LR（backbone 5.16e-5，action head + predictor 5.16e-4）
- 冻结：layers before $L_s=12$ + DPT head
- Language encoder：frozen T5 [48]
- Image augmentation：random crop, rotation, color jitter（eval 时关）

### Post-training (per benchmark)
- 16 × GH200 GPU，batch 160，~48 hours
- H=1（context window 缩到 1）
- 110k training steps 至 convergence

### Action space
- $d_a = 7$ end-effector action（位置 + 旋转 + gripper）
- $d_s = 7$ proprioceptive state（joint config + end-effector pose）
- $C = 8$ action chunk（仿真），real-world 也是 8

---

## 9. Intuition Building: 几个深层问题

### 9.1 为什么 future prediction 的 supervision 帮助 action？

这是 paper 没明说但很重要的点。Action $\hat{a}_t$ 和 future geometry $\tilde{\mathbf{Z}}_{t+1}^{(L_s)}$ 在同一个 autoregressive sequence 里被预测，共享 backbone 和 predictor。所以 $\mathcal{L}_{\text{feat}}$ 和 $\mathcal{L}_{\text{depth}}$ 的 gradient 不仅 refine future prediction，也通过 shared parameters 间接 refine action prediction。

更深层：要让 action 准确，模型必须理解 "执行 action $a_t$ 后世界会变成什么样" — 这就是 world model 的本质。Future prediction 是显式监督这个 "action → world state transition" 的方式。

### 9.2 为什么 GFM backbone 比 VLM backbone 更适合 manipulation？

VLM（OpenVLA、π0.5）的 backbone 是在 web-scale image-text pairs 上 pretrain 的，representation 偏 semantic（"这是个杯子"）。但 manipulation 真正需要的是 spatial/geometric reasoning（"杯子相对于 gripper 在哪、距离多远、抓取角度如何"）。

GFM 在 multi-view 3D reconstruction 上 pretrain，representation 本身就是 metric-aware 的，camera-invariant 的。这就是为什么 GAM 在 camera perturbation 下掉点最少 — backbone 的 representation 对 viewpoint 变化天然鲁棒。

### 9.3 为什么 single-pass 比 diffusion 好？

Cosmos Policy [3] 用 video diffusion，需要 multi-step denoising（~50 steps），latency 382ms。GAM 用 single-pass autoregressive prediction，6.9ms。但 GAM 在 LIBERO 上的 success rate 反而更高（97.6 vs 98.5 接近，LIBERO-Plus 85.5 vs 82.4 反超）。

这说明 diffusion 的 multi-step sampling 对 manipulation 这种 action space 相对低维（7-dim）的任务 overkill。Diffusion 的优势在 high-dim, multi-modal output distribution（image generation），action chunk 还没到那个 dimensionality。

### 9.4 为什么 split layer 必须在 frame-wise/global 边界？

GFM 的设计哲学是：浅层 frame-wise attention 做 per-view feature extraction（local 2D appearance → local geometry），深层 global attention 做 cross-view fusion（local geometry → consistent 3D structure）。如果在 frame-wise 阶段插入 predictor，future prediction 缺少 cross-view 信息；如果在 global 阶段太晚插入，future tokens 没有足够 deep blocks 去 refine。$L_s=12$ 正好是这两个阶段的边界，让 future prediction 在 "已经 extract 好 per-view feature 但还没 cross-view fuse" 的层次发生，然后由 deep blocks 完成 fusion。

### 9.5 与 RDT-1B、3D Diffusion Policy 的对比

RDT-1B [25] 和 3D Diffusion Policy [11] 也用 3D 信息，但它们是 task-specific encoder trained from scratch（point cloud encoder）。GAM 的优势：直接 leverage GFM 的 web-scale pretrain，无需从 scratch 学 3D encoder。代价是 GAM 依赖 multi-view RGB（至少 2 个 view），而 point-cloud policy 单 depth camera 也能用。

### 9.6 与 Spatial Forcing [17] / ROCKET [18] 的对比

这两条线把 GFM 当 frozen feature extractor，把它的 features distill 进 VLA backbone：
- Spatial Forcing：用 representation alignment loss 让 VLA 中间层 features 对齐 GFM features
- ROCKET：multi-layer residual alignment

它们的 limitation：GFM 只提供 "static feature prior"，action decoding 仍在 VLA 自己的 2D-aware latent space 里发生。GAM 直接把 GFM 当 backbone，action decoding 在 GFM 的 3D-aware latent space 里发生 — 这就是 GAM 在 LIBERO-Plus 上比 π0.5+ROCKET (47.5) 高 38 个点的原因。

### 9.7 失败模式与 limitation

Paper 自己提到：language reasoning 受限于 frozen T5 encoder。如果 task 涉及复杂 instruction 跟随（如 hierarchical、conditional instruction），T5 的小 encoder 可能跟不上 VLA 那种 LLM-based language understanding。

未明说但隐含的 limitation：
- 多 view 是 hard requirement — 单 camera setup 无法直接用
- Real-world 没有 ground-truth depth，用 pseudo-depth 监督，长期可能有 confirmation bias
- Long-horizon task（RoboCasa 里 PnPMicrowaveToCounter 11.3%）表现差，说明 future prediction 在长 horizon 下退化

---

## 10. 我的整体评价

这篇 paper 的核心 insight — **把 GFM 在中间层切开，往里塞 causal future predictor，让 action 在 GFM latent space 里产生** — 是个非常干净的 architectural innovation。它不是简单地 "把 GFM 接到 VLA 上"，而是把 GFM 的整个 forward pathway repurpose 成 policy 的 forward pathway。这种 "repurpose rather than attach" 的思路很 elegant。

实验结果也 convincing：1.4B 的 model 在 LIBERO-Plus 上超过 7B 的 OpenVLA-OFT 和 3.3B 的 π0.5，camera perturbation 下尤其显著。55x speedup over Cosmos Policy 也证明 diffusion 在 manipulation 里 overkill。

几个我想 push 的方向：
1. **Failure analysis**：RoboCasa 里那些 11-30% 的 task 到底是几何预测失败还是 action decoding 失败？需要更细的 per-task ablation。
2. **Single-view variant**：很多 real-world robot 只有 wrist camera + 1 external camera，但 GFM 期望 multi-view。能否让 GFM 在 single-view 下也给出合理 latent？
3. **Action chunk consistency**：开环执行 8 步 action chunk 后才重新观察，中间如果 world state 偏离 prediction 怎么办？这是所有 action chunking policy 的通病，GAM 没特别处理。
4. **Language understanding**：frozen T5 是明显的 bottleneck。能否用更小的 LLM（Qwen2.5-VL 3B 之类）替换 T5，让 language token 本身参与 backprop？

总的来说，这是一篇把 "geometric prior" 这个 idea 落到 architecture-level 的 paper，比那种 "add a 3D module to VLA" 的 incremental work 高一个层次。

---

## Web References

- **GAM project page**: https://cvlab-kaist.github.io/Geometric-Action-Model
- **Depth Anything V3 (DA3)**: https://depth-anything-v3.github.io/
- **VGGT (Visual Geometry Grounded Transformer)**: https://vgg-t.github.io/
- **LIBERO benchmark**: https://libero-project.github.io/
- **LIBERO-Plus**: https://arxiv.org/abs/2510.13626
- **OpenVLA**: https://openvla.github.io/
- **OpenVLA-OFT**: https://arxiv.org/abs/2502.19645
- **π0 (Physical Intelligence)**: https://www.physicalintelligence.company/blog/pi0
- **π0.5**: https://arxiv.org/abs/2504.16054
- **Cosmos Policy**: https://arxiv.org/abs/2601.16163
- **NVIDIA Cosmos**: https://www.nvidia.com/en-us/ai/cosmos/
- **Spatial Forcing**: https://arxiv.org/abs/2510.12276
- **ROCKET**: https://arxiv.org/abs/2602.17951
- **RoboCasa**: https://robocasa.github.io/
- **MimicGen**: https://mimicgen.github.io/
- **Open X-Embodiment**: https://robotics-transformer-x.github.io/
- **DPT (Dense Prediction Transformer)**: https://arxiv.org/abs/2103.13413
- **T5**: https://arxiv.org/abs/1910.10683
- **3D Diffusion Policy**: https://3d-diffusion-policy.github.io/
- **RDT-1B**: https://thu-ml.github.io/RDT/
- **Track4World**: https://arxiv.org/abs/2603.02573
- **GeoVLA**: https://geovla.github.io/
- **UniVLA**: https://arxiv.org/abs/2505.06111
- **WorldVLA**: https://arxiv.org/abs/2506.21539
- **FLARE**: https://arxiv.org/abs/2505.15659
