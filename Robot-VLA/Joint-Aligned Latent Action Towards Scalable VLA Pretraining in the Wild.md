---
source_pdf: Joint-Aligned Latent Action Towards Scalable VLA Pretraining in the Wild.pdf
paper_sha256: a9a3a5be2a6a2f435feacccf2d45544e519537151b64482573ef5af8eace946f
processed_at: '2026-08-05T10:53:24-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用大白话来拆解这篇 paper，我们直接切入核心 intuition，同时保留必要的 technical depth 来 build 你的 mental model。

### 1. 核心痛点：Robot 缺数据，Human video 不好用

我们要 train 一个 VLA (Vision-Language-Action) model，让它看图、听指令就能输出动作。最大的绊脚石是 robot data 太少。于是大家盯上了 human manipulation video。Human video 有两类，各有利弊：
- Lab data (比如 ARCTIC, DexYCB)：有精确的 3D hand tracking label (MANO 参数)，但是场景极其单调，全是在 lab 里 pick and place。
- In-the-wild data (比如 Ego4D)：场景丰富，人类自然行为多，但是 hand tracking label 极不可靠。

所以核心 problem 就是：how to mix 精确但单调的 lab data，与丰富但没 label 的 wild data，一起 pretrain VLA？

### 2. Paradigm Shift：放弃 Reconstruction，改做 Alignment

以前的 latent action 方法（比如 LAPA）怎么做呢？它们用 reconstruction 的思路。
模型看 current frame $v_t$，猜出一个 latent action $z$，然后试图用 $z$ 和 $v_t$ 去 reconstruct (重建) future frame $v_{t+\delta}$。如果重建得准，说明 $z$ 确实编码了动作信息。

这个逻辑在 wild video 里会彻底崩盘。因为 wild video 背景太乱、光线变化多、hand 经常被遮挡。模型为了把 pixel 重建对，会把大量计算力浪费在学背景、阴影、camera artifact 上，导致提取出的 $z$ 被 noise 污染，根本不是纯粹的 action 信号。

JALA 的 insight 极其精妙：我们人类学 manipulation，transfer 的是 action pattern，我们根本不需要在大脑里 memorize 每一个 pixel 的变化。所以，JALA 跳过 pixel reconstruction，直接做 alignment (对齐)。

### 3. 技术拆解：Joint Alignment 具体怎么搞

JALA 在 VLA 的 Transformer 里抽一层的 hidden state，叫 predictive embedding $h$。这个 $h$ 就是 VLA 理解了图和指令后，脑子里形成的 action concept。同时，JALA 用一个 inverse dynamics model (IDM) 看边界两帧 $(v_t, v_{t+\delta}$)，算出一个 latent action $z$。

然后核心的 loss 就极其简单：

$$\mathcal{L}_{\mathrm{Align}} = \sum_{i=1}^{N} \sum_{k=1}^{K} \| h_{i,k} - z_{i,k} \|_1$$

变量拆解：
- $N$：一个 video 被切成多少个 motion chunk
- $K$：每个 chunk 里的 token 数量（JALA 设为 128，64 wrist + 64 finger）
- $h_{i,k}$：VLA 内部对第 $i$ 个 chunk 第 $k$ 个 token 的 predictive embedding
- $z_{i,k}$：IDM 从 boundary frames 推导出的 latent action
- $\|\cdot\|_1$：L1 距离

这个 loss 的意思是：你 VLM 脑子里想的 action concept ($h$)，必须和 IDM 从视觉变化里提取的 action 动力学 ($z$) 严丝合缝。这就逼着 VLA 学到真正 action-relevant 的东西，完全不碰像素重建。

对于有 label 的 lab data，再加一个 Masked Chunk Prediction (MCP) loss，预测真实的 motion token。对于没 label 的 wild data，只跑 alignment loss。这样两套数据就完美融合在同一个 latent action space 里了。

### 4. 架构核心：Decoupled EMA Update 防止崩溃

这里有一个极度 subtle 的架构设计。为了算 $z$，JALA 引入了 Latent Action Perceiver (LAP)。但是 VLM 的 context 和 LAP 里的 visual feature 空间是不对齐的。直接硬拉对齐，训练极容易崩溃，一方 dominate 另一方就 collapse 了。

为了稳定，JALA 引入了 Latent State Perceiver (LSP)。LAP 吃 $(v_t, v_{t+\delta})$ 算 dynamics，LSP 吃 duplicate 的初始帧 $(v_0, v_0)$ 提供 context anchor。两者共享参数，但优化时做 Decoupled Update：

$$\theta_b^{\mathrm{LAP}} \leftarrow \alpha \theta_b^{\mathrm{LAP}} + (1-\alpha) \theta_b^{\mathrm{LSP}}$$
$$\theta_q^{\mathrm{LSP}} \leftarrow \alpha \theta_q^{\mathrm{LSP}} + (1-\alpha) \theta_q^{\mathrm{LAP}}$$

变量拆解：
- $\theta_b$：Perceiver 的 backbone 参数
- $\theta_q$：Perceiver 的 learnable query 参数
- $\alpha$：EMA 系数，设为 0.999，即非常缓慢的更新
- LAP / LSP 上标：分别代表这两个 module 的参数

Intuition 是这样的：LSP 的 backbone 用 gradient 优化，保证 visual feature 映射进 VLM context space；LAP 的 query 用 gradient 优化，保证 latent action 被 action 信号 anchor。然后互相用 EMA 慢慢同步 weight。这就像 MoCo 里的 momentum encoder，但双向 asymmetric。Table 2 显示，去掉这个 decoupling (JALA w/o dec.)，LIBERO 成绩直接掉到 56.6%，足见这个 stabilizer 是 critical 的。

### 5. Post-training：Flow-Matching 做精确控制

Pretraining 建立了 unified latent action space。迁移到 robot 时，用 DiT (Diffusion Transformer) 接管 predictive embedding，做 flow-matching：

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{\tau, \epsilon, A_t} \left[ \| V_\theta(\{h_{i,k}\}, A_t^\tau, q_t) - (\epsilon - A_t) \|_2^2 \right]$$

变量拆解：
- $\tau \in [0, 1]$：flow timestep
- $\epsilon$：standard Gaussian noise
- $A_t$：ground-truth robot action chunk
- $A_t^\tau = \tau A_t + (1-\tau)\epsilon$：线性插值的 noised action
- $q_t$：robot proprioceptive state
- $\{h_{i,k}\}$：pretrained VLA 输出的 predictive embeddings，作为条件输入
- $V_\theta$：DiT 需要学的 denoising vector field

Inference 时只跑 4 步 Euler integration 就能 denoise 出精确的 robot action。因为大部分 manipulation prior 已经被 pretraining 烤进 $\{h_{i,k}\}$ 里了，flow-matching head 只需要做 refinement，所以极快。

### 6. 实验 Build Intuition

在 Wild hand motion generation 上，JALA 把 MPJPE 从 Being-H0 的 16.91 降到 11.02。更重要的是，如果盲目给 wild data 但不做 alignment (JALA w/o latent)，MPJPE 反而飙到 20.34。这证明无监督 wild video 必须有 structural supervision 才能产生正向价值。

在 LIBERO benchmark 上，JALA-dino 拿下 96.9% average success rate，完爆 reconstruction-based 的 LAPA (79.5%) 和重训的 LAPA† (83.5%)。这说明 JALA 赢在 training objective，而非单纯堆数据或改 backbone。

在 Real-world robot 实验里，最震撼的是 robustness。在 unseen setting（换桌布纹理、换笔的颜色）下，baseline 都大幅掉点，JALA-dino 几乎不掉点（60.0% → 58.0%）。这说明它学到的是 action-relevant dynamics，非常 robust to visual distribution shift。甚至能涌现出抓取失败自动 retract 手腕重抓的 self-corrective behavior。

### 7. 联想与延伸

这个 work 让我想到 Yann LeCun 的 JEPA。JEPA 也是放弃 pixel-level reconstruction，在 latent space 里做 prediction。JALA 就是 VLA 领域的 JEPA，用 latent alignment 替代 world model 的 pixel prediction，因为 action-relevant 信号在 pixel space 里太稀疏，在 latent space 里更稠密。

另一个联想是 Masked Autoencoder (MAE)。MAE 证明不需要 label，只靠 reconstruct missing patch 就能学极好 visual representation。JALA 证明了在 video action space 里，甚至连 reconstruct 都不需要，只要 align latent space，就能逼出 generalizable manipulation prior。

后续如果 scale 到 10B 级别，predictive embedding $h$ 可能会演化成一个真正的 "action language" foundation，只要 wild video 不断喂，就能无监督进化。

### Reference Links
- JALA project page: https://research.beingbeyond.com/jala
- Being-H0 (predecessor): https://arxiv.org/abs/2507.15597
- LAPA (baseline): https://arxiv.org/abs/2410.11758
- V-JEPA (conceptual relative): https://ai.meta.com/research/publications/v-jepa-latent-video-prediction-for-visual-representation-learning/
- Flow Matching for generative modeling: https://arxiv.org/abs/2210.02747
- Ego4D dataset: https://ego4d-data.org/
- InternVL3 backbone: https://arxiv.org/abs/2306.14238
- DINOv3 visual encoder: https://arxiv.org/abs/2508.10104

---

# JALA: Joint-Aligned Latent Actions 详解

非常 happy 能 walk through 这篇 paper，因为是 BeingBeyond 团队 (Hao Luo 等) 在 Being-H0 之后的延续工作，刚好处于 VLA scaling 的关键路径上。我会按 motivation → method → data → experiments 的顺序展开，每个环节都尽量 build 出 intuition。

---

## 1. Motivation：为什么需要 JALA

### 1.1 VLA 的 data bottleneck

当前 VLA 的根本困境：robot data 比 vision/language data 少好几个数量级。Open X-Embodiment、DROID、Agibot World 这些加起来也就 thousands of hours 量级，跟 LLaMA 训练用的文本、LLaVA-OneVision 用的 image-text 对比完全是 different magnitude。

Human manipulation video 是天然的 scalable 替代资源，但存在一个 **quality-variety trade-off**：

| 数据类型 | 优点 | 缺点 |
|---------|------|------|
| Lab data (ARCTIC, DexYCB, HOI4D, EgoDex, Hot3D) | 精确 3D hand tracking (MANO 参数) | 场景受限、task 多样性低 |
| In-the-wild data (Ego4D, EPIC-KITCHENS, HD-EPIC) | diversity 高、natural behaviors | hand tracking label 不可靠 |

这个 trade-off 就是 JALA 要打破的。核心问题：how to combine heterogeneous human data sources?

### 1.2 Prior paradigm 的局限

之前的 latent action paradigm (LAPA, UniVLA) 是 reconstruction-based：
- **IDM (Inverse Dynamics Model)**：从 (current frame, future frame) 推 latent action $z$
- **FDM (Forward Dynamics Model)**：从 (current frame, $z$) 重建 future frame
- Latent action 通过 reconstruction loss 被 "anchored" 到 action-relevant 信息

问题在 fine-grained human manipulation 上：
- hand motion 细微、变化大，FDM 学一个准确的 next-frame predictor 很难
- FDM 的 noise 直接 degrade latent action quality
- in-the-wild video 里 hand 小或 occluded，pixel reconstruction 会放大 background shift / camera artifact 等 action-agnostic 信号

JALA 的核心 idea：**bypass full visual dynamic reconstruction**，直接 align 一个 predictive embedding $h$（来自 VLA context）与 latent action $z$（来自 IDM）。

类比人类学 manipulation：我们 transfer 的是 **action patterns**，不是 memorize every visual detail。

---

## 2. Methodology 详解

### 2.1 数据建模与 tokenization

每个 sample 是 $(v, x, \mathcal{M})$：
- $v = \{v_1, \ldots, v_T\}$：video frames
- $x$：text instruction
- $\mathcal{M} = \{m_1, \ldots, m_T\}$：hand pose sequence，每个 $m_t = (\theta_t, \mathbf{r}_t, \tau_t, \beta_t)$
  - $\theta_t$：relative joint angles (手指关节)
  - $\mathbf{r}_t$：global wrist rotation
  - $\tau_t$：wrist translation
  - $\beta_t$：hand shape（static，被排除）

**GRVQ (Group-Residual Vector Quantization)** 把 15-frame motion chunk tokenize 成 **128 tokens**：
- 64 wrist motion tokens + 64 finger motion tokens
- codebook size 各 4096
- 序列格式：`<mot> {wrist_tokens} {finger_tokens} </mot>`
- 两手场景：left/right chunks 沿时间轴 interleave

输入训练序列：$[x, v_1, A_1, A_2, \ldots, A_N]$，其中 $A_i$ 是第 i 个 motion chunk (长度 $T/N$)。

预训练目标（Eq 1）：

$$\max_{\Theta} \sum_{i=1}^{N} \log p(A_i \mid A_{<i}, v_1, x; \Theta)$$

- $\Theta$：VLA 参数
- $A_{<i}$：第 i 个 chunk 之前的 motion chunks（autoregressive context）
- $v_1$：初始帧（视觉 grounding）
- $x$：语言指令

### 2.2 Joint Alignment 的两个信号

对 chunk $A_i$ 中的每个 token $a_{i,k}$，从预设的 attention layer (第 19 层) 抽取 hidden state 作为 **predictive embedding** $h_{i,k} \in \mathbb{R}^d$。$h_{i,k}$ 被两个 complementary signal 塑造：

#### (1) Masked Chunk Prediction (MCP)

借鉴 GR-1 的 chunk-level masked token modeling。整个 chunk 用 `<placeholder>` 替换，chunk 内用 bidirectional attention：

$$\mathcal{L}_{\mathrm{MCP}} = -\sum_{i=1}^{N} \sum_{k=1}^{K} \log p_{\Theta}(a_{i,k} \mid A_{<i}, v, x)$$

- $N$：sequence 中 chunk 总数
- $K$：每 chunk 的 token 数（128）
- $a_{i,k}$：第 i chunk 第 k 个 motion token
- 双向 attention 让 chunk 内 tokens 联合建模，hidden states $h_{i,k}$ 携带 chunk-level movement pattern

#### (2) Latent Action Perceiver (LAP) 与 Alignment

LAP 是 Perceiver 架构的 inverse dynamics model：
- 输入：chunk 的 boundary frames $(v_t, v_{t+\delta})$
- 用 fixed learnable queries 抽 K 个 latent action vectors $\{z_{i,1}, \ldots, z_{i,K}\}$
- 捕获 chunk-level transition 的 dynamics

Alignment loss（Eq 3）：

$$\mathcal{L}_{\mathrm{Align}} = \sum_{i=1}^{N} \sum_{k=1}^{K} \| h_{i,k} - z_{i,k} \|_1$$

- $\|\cdot\|_1$：L1 距离（比 L2 更 robust to outlier）
- $h_{i,k}$：predictive embedding (来自 VLM context)
- $z_{i,k}$：latent action (来自 IDM)

**关键 insight**：单独 MCP 不 capture visual dynamics，单独 LAP 不保证 action-centric representation。Joint alignment 把 motion pattern (来自 hand tracking label) 和 visual dynamics (来自 IDM) fuse 到 unified latent action space。

### 2.3 总损失

$$\mathcal{L} = \mathbf{1}_{\mathrm{labeled}} \cdot \mathcal{L}_{\mathrm{MCP}} + \lambda \mathcal{L}_{\mathrm{Align}}$$

- $\mathbf{1}_{\mathrm{labeled}}$：indicator，只有 annotated data 才激活 MCP
- $\lambda = 0.5$：alignment 权重
- 对 in-the-wild video (无 label)：只 apply $\mathcal{L}_{\mathrm{Align}}$
- 对 lab data：两个 loss 都 active

### 2.4 Decoupled EMA Update（架构稳定的关键）

这是 paper 里最 subtle 也最重要的设计之一。

**问题**：LAP 和 LSP 处理异质信号
- LAP 输入：boundary frames $(v_t, v_{t+\delta})$ → dynamics
- LSP 输入：duplicate initial frame $(v_0, v_0)$ → context
- 直接用 alignment loss 耦合，可能 collapse 或一方 dominate

**架构**：LAP 和 LSP 共享 2-layer Perceiver (weight shared)
- 每层：cross-attention (query → visual features as KV) + self-attention + 2-layer MLP
- 两手视频：shared trunk + two-head MLP（channel dim 翻倍后 split left/right）

**Decoupled optimization**：
- **Backbone** $\theta_b$：用 LSP 的 gradient 优化 → visual features 映射到 predictive embedding space
- **Queries** $\theta_q$：用 LAP 的 gradient 优化 → latent action 被 explicit action cues anchor

**Asymmetric EMA update**：

$$\theta_b^{\mathrm{LAP}} \leftarrow \alpha \theta_b^{\mathrm{LAP}} + (1-\alpha) \theta_b^{\mathrm{LSP}}$$

$$\theta_q^{\mathrm{LSP}} \leftarrow \alpha \theta_q^{\mathrm{LSP}} + (1-\alpha) \theta_q^{\mathrm{LAP}}$$

- $\alpha = 0.999$：EMA coefficient
- $\theta_b^{\mathrm{LAP}}, \theta_b^{\mathrm{LSP}}$：LAP/LSP backbone 参数
- $\theta_q^{\mathrm{LAP}}, \theta_q^{\mathrm{LSP}}$：LAP/LSP query 参数
- LAP backbone 接收 LSP backbone 的更新（保持与 predictive context 一致）
- LSP query 接收 LAP query 的更新（继承 action-grounding 能力）

这个设计让 latent action 既能被 context predict，又能被 action cues anchor。Tab 2 显示去掉 decoupling（JALA w/o dec.）直接 collapse 到 56.6%（vs 96.9%）。

### 2.5 Hybrid Masking Scheme

naive masking 有 train-inference mismatch：训练时整 chunk 都 mask，inference 时 chunk 是 sequentially generated。

JALA 的 hybrid scheme：
1. 对 N 个 chunks，随机选一个作为 main prediction target
2. Target 之前的 chunks：保持完整 (no masking)
3. Target chunk 内：每个 token 以 random ratio $\in \{0.05, 0.15, \ldots, 1.0\}$ 被 mask
4. Target 之后的 chunks：固定 5% mask 概率，提供额外 supervision

对 unlabeled video：整 chunk 都替换为 `<placeholder>`，只走 alignment path。

Inference 时：current chunk 解码多次，每次解码 ~5% tokens，最后 ensemble。

### 2.6 Post-training: Flow-Matching Head

预训练建立的 unified latent action space 通过 DiT-based flow-matching head 转移到 robot action space：

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{\tau, \epsilon, A_t} \left[ \| V_\theta(\{h_{i,k}\}, A_t^\tau, q_t) - (\epsilon - A_t) \|_2^2 \right]$$

- $A_t$：ground-truth robot action chunk
- $\tau \in [0, 1]$：flow timestep
- $\epsilon$：standard Gaussian noise
- $A_t^\tau = \tau A_t + (1-\tau)\epsilon$：interpolated noised action（线性插值路径）
- $q_t$：robot proprioceptive state（关节位置、gripper config）
- $\{h_{i,k}\}$：pretrained VLA 输出的 predictive embeddings
- $V_\theta$：要学的 denoising vector field，预测 $\epsilon - A_t$（从 noise 指向 data 的方向）

DiT 结构：16 layers × 32-head attention，hidden dim 2048，alternating self-attention（处理 proprio + noised action）和 cross-attention（融合 $\{h_{i,k}\}$）。

Inference：forward Euler integration，N=4 步 denoise，输出 robot action chunk。

---

## 3. UniHand-Mix Dataset

### 3.1 规模与组成

- **总量**：7.5M instruction-video samples (>2,000 hours)
- **Lab subset**：5M+ annotated (instruction + video + MANO motion) ← 1,000 hours
- **Wild subset**：2.5M (instruction + video only) ← 1,123 hours Ego4D，其中 ~10% 有 pseudo hand-pose annotation

### 3.2 Lab Pipeline (3 步)

1. **Hand pose standardization**：统一到 MANO format
   - mocap/SLAM annotation → 直接转 MANO
   - 3D joints → optimization 拟合 MANO
   - RGB-only → HaWoR 估计 + temporal smoothing + left-right correction

2. **Hierarchical task labeling**：10s clips 分两层
   - Clip level：imperative instructions + concise summaries
   - Second level：contact states, object properties, hand-object interactions（含 bimanual）

3. **Instructional data generation**：base templates + Gemini 多样化语言，构造 motion generation / motion description / motion continuation 三类样本

### 3.3 Wild Pipeline (3 步)

1. **Visual filtering**：WiLoR 做 frame-level hand detection，丢弃无 hand 的 clip
2. **Hand-centric activity validation**：Gemini-2.5-Flash 识别 manipulation activities，丢弃 idle / distractor hand，自动生成 paired instructions
3. **Pseudo hand-pose annotation (optional)**：HaWoR 估计，confidence threshold 0.65 保留

### 3.4 统计特性

- **Task type distribution**：motion generation 占最大比例，video-only 次之，motion description/continuation 提供辅助 supervision
- **Clip length**：skewed toward shorter clips (1-10s)，short clip 学 dense local interaction，long clip 学 multi-step dependency
- **Source diversity**：8 个数据源混合，in-the-wild 占大份额但 lab subset 仍 substantial

---

## 4. Implementation Details

### 4.1 预训练超参数

| 项目 | 设置 |
|------|------|
| VLA backbone | InternVL3-2B (28 attention layers) |
| Visual encoder | DINOv3 或 V-JEPA2 |
| Motion chunk | 15 frames → 128 tokens (64+64) |
| Codebook size | 4096 (GRVQ) |
| Predictive embedding layer | 19th (out of 28) |
| λ (alignment weight) | 0.5 |
| EMA α | 0.999 |
| Optimizer | AdamW (lr=3e-5, wd=0.05, β=(0.9, 0.95)) |
| LR schedule | 5% warmup + cosine decay |
| Gradient clip | max norm 1.0 |
| Effective batch | 128 (per-GPU 16 × 8 GPU accumulation) |
| Epoch | 1 |
| Hardware | 8× A800 80GB, 68 hours |

### 4.2 Post-training 超参数

- DiT: 16 layers, 32-head, hidden 2048
- Batch size 128, lr=1e-4, 5% warmup + cosine
- LIBERO: 30k steps (~8h)
- RoboCasa: 60k steps (~16h)
- Real-world: 30k steps (~8h)
- 只 unfreeze LM 参数，vision encoder 冻结
- Inference denoising steps: N=4

### 4.3 In-the-wild data trick

in-the-wild video 时间 slowed by factor 0.5，弥补 lab data 与 wild data 的 action speed 差异。这个细节很 practical。

---

## 5. Experiments 深度解析

### 5.1 Hand Motion Generation (Table 1)

| Model | Lab MPJPE↓ | Wild MPJPE↓ | Lab PA-MPJPE↓ | Wild PA-MPJPE↓ |
|-------|------------|-------------|---------------|----------------|
| Being-H0 | 7.61 | 16.91 | 1.34 | 3.81 |
| Being-H0+dino | 7.54 | 15.14 | 0.90 | 2.78 |
| JALA w/o align | 7.72 | 15.73 | 0.89 | 2.34 |
| JALA w/o latent | 8.26 | 20.34 | 1.83 | 3.94 |
| **JALA-dino** | **7.16** | **11.02** | 0.91 | **1.12** |
| JALA-vjepa | 7.05 | 11.54 | 0.94 | 1.32 |

**关键观察**：
1. Lab split 上所有方法接近，因为 supervised signal 充足
2. Wild split 上 JALA-dino 比 Being-H0 提升 ~35% (16.91→11.02)
3. **JALA w/o latent** 反而比 w/o align 差很多（Wild MPJPE 20.34 vs 15.73），说明简单堆 unlabeled data 而无 structural supervision 反而有害
4. JALA-dino 与 JALA-vjepa 接近，说明对 visual backbone 鲁棒

Metric 含义：
- **MPJPE** (Mean Per-Joint Position Error)：3D joint 欧氏距离均值，space accuracy
- **PA-MPJPE** (Procrustes-Aligned MPJPE)：rigid alignment 后的 MPJPE，relative pose fidelity
- **MWTE** (Mean Wrist Trajectory Error)：wrist trajectory 平均偏移，global trajectory fidelity
- **MDE** (Motion Direction Error)：final displacement direction 误差，motion trend consistency

### 5.2 LIBERO Two-View (Table 2)

| Model | Spatial | Object | Goal | Long | Avg |
|-------|---------|--------|------|------|-----|
| LAPA | 83.4 | 87.6 | 78.2 | 68.8 | 79.5 |
| π0-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| GR00T N1.5 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| UniVLA | 95.4 | 98.8 | 93.6 | 94.0 | 95.5 |
| Being-H0 | 92.6 | 96.8 | 94.0 | 77.4 | 90.2 |
| JALA-act | 93.4 | 97.8 | 94.2 | 91.8 | 94.3 |
| JALA w/o dec. | 64.6 | 58.4 | 61.2 | 42.2 | 56.6 |
| LAPA† | 87.4 | 91.2 | 90.0 | 65.4 | 83.5 |
| JALA* | 95.2 | 96.4 | 97.2 | 94.0 | 95.7 |
| **JALA-dino** | 96.0 | 98.2 | 97.4 | 96.0 | **96.9** |

**关键对照**：
- **JALA vs LAPA†** (same backbone + same data)：96.9 vs 83.5，差 13.4 个点 → 训练 objective 是主因，not data/architecture
- **JALA-act vs Being-H0** (same action-available subset)：94.3 vs 90.2，Long suite 91.8 vs 77.4 → joint alignment 本身就有提升，不依赖 wild data
- **JALA w/o dec.**：56.6% → decoupled EMA 是 critical stabilizer
- **JALA***（把 wild label 当 unlabeled）：95.7 vs 96.9 → wild 的少量 motion annotation 有帮助但非必需

### 5.3 LIBERO Single-View (Table 3) - 更难 setting

| Model | Spatial | Object | Goal | Long | Avg |
|-------|---------|--------|------|------|-----|
| UniVLA-full†† | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| UniVLA-human† | 91.2 | 94.2 | 90.2 | 79.4 | 88.7 |
| GR00T N1.5 | 91.4 | 97.6 | 94.0 | 85.6 | 92.1 |
| Being-H0 | 86.6 | 92.8 | 89.6 | 70.4 | 84.9 |
| JALA w/o latent | 80.4 | 83.6 | 75.2 | 68.6 | 77.0 |
| JALA-vjepa | 91.6 | 98.2 | 94.4 | 84.2 | 92.1 |
| **JALA-dino** | 90.4 | 96.4 | 95.2 | 87.2 | **92.3** |

JALA-dino 在 ≤3B 模型里 SoTA，Long suite 上 87.2 vs GR00T N1.5 的 85.6。UniVLA-full 用 Bridge-V2 robot data 才到 95.2，JALA 只用 human video 接近这个水平。

### 5.4 RoboCasa + GR1 (Table 4)

| Model | RoboCasa Syn. | RoboCasa Human | GR1 Tabletop |
|-------|---------------|----------------|--------------|
| GR00T N1.5 | 20.83 | 35.17 | 20.41 |
| LAPA | 16.25 | 22.42 | 11.42 |
| Being-H0 | 23.83 | 31.33 | 12.91 |
| JALA-act | 24.92 | 32.42 | 20.25 |
| JALA w/o dec. | 14.25 | 19.33 | 9.25 |
| LAPA† | 20.25 | 27.33 | 13.50 |
| JALA* | 25.33 | 33.83 | 24.50 |
| **JALA** | **27.58** | **35.42** | **26.33** |

GR1 tabletop 特别有意思：用 dexterous hand（接近人手 morphology），JALA 比 Being-H0 翻倍（26.33 vs 12.91）。JALA-act 20.25 vs Being-H0 12.91 也翻倍 → embodiment shift 大时，joint alignment 尤其有用。

### 5.5 Ablation: 层选择 (Fig 6 右)

把 flow-matching head 接到不同 layer：
- Layer 14：稍弱
- **Layer 19：最优**（也是 alignment 用的层）
- Layer 24, 28：急剧退化

**Intuition**：alignment 把 generalizable cue 集中在选定的 19 层，deeper layer 过拟合数据集-specific detail，transfer 价值低。这呼应 contrastive learning 里 mid-layer 通常更 transferable 的现象。

### 5.6 Ablation: Wild Data 比例 (Fig 6 左)

lab data 固定，wild data 0% / 25% / 50% / 100% → 性能持续提升，证明 framework 真的能 leverage unlabeled video，不是 plateau 早期。

### 5.7 Reconstruction vs Joint Alignment 效率

同 backbone 同 data：LAPA† 两阶段训练 29h + 57h = 86h；JALA 68h。JALA 用 <80% wall-clock 还能跑出更高性能。Pixel reconstruction 在 in-the-wild video 上把大量 compute 浪费在 action-agnostic signal 上。

### 5.8 Real-World Experiments (Table 5)

三个 multi-step 任务：
- **Put-Three-Obj**：开抽屉 → pick&place 三水果 → 关抽屉（5 subtasks）
- **Wipe-Board**：抓布 → 擦标记区域 → 移除可见墨水（3 subtasks）
- **Water-Plant**：抓 spray bottle → 重定位 → 触发 trigger（3 subtasks）

| Model | Put-Three-Obj Seen | Unseen | Wipe-Board Seen | Unseen | Water-Plant |
|-------|--------------------|--------|------------------|--------|-------------|
| Being-H0 | 38.0 | 16.0 | 40.0 | 33.3 | 36.7 |
| GR00T N1.5 | 48.0 | 28.0 | 56.7 | 43.3 | 53.3 |
| JALA w/o align | 40.0 | 32.0 | 53.3 | 43.3 | 56.7 |
| JALA-vjepa | 38.0 | 34.0 | 66.7 | 60.0 | 66.7 |
| **JALA-dino** | **60.0** | **58.0** | **83.3** | **80.0** | **73.3** |

**最 impressive 的点**：unseen setting 下 JALA-dino 极稳定
- Put-Three-Obj Unseen 只掉 2 个点 (60.0→58.0)
- Wipe-Board Unseen 只掉 3.3 个点 (83.3→80.0)
- 其他 baseline 在 unseen setting 都大幅退化

→ JALA 依赖 action-relevant dynamics 而非 superficial appearance cue，这是 latent action alignment 的核心红利。

**Self-corrective behavior**（Fig 10）：Put-Three-Obj unseen case 下抓 banana 第一次 misalign，policy 主动 retract + re-position wrist + 二次 grasp。这个 recovery strategy 没被显式 supervised，是 latent action modeling 涌现出来的 feedback-driven 行为。

---

## 6. 几个 intuition 总结

### 6.1 为什么 joint alignment > reconstruction

reconstruction 的 loss 信号被 $p(\text{future frame} \mid \text{current}, z)$ 主导，FDM 必须同时建模 action-relevant dynamics 和 action-agnostic 的 background / lighting / camera noise。在 in-the-wild video 中后者占比更大，latent action 被污染。

joint alignment 只要求 $h \approx z$，$z$ 来自 IDM（只看 boundary frames），是纯 dynamics signal。相当于把 action-relevant 信号做了一个 "蒸馏"，跳过 appearance modeling。

### 6.2 为什么需要 LSP（不能只有 LAP）

如果只用 LAP，predictive embedding $h$ 必须从 VLM context 独立学会预测 boundary-frame dynamics。但 VLM 的 visual features 来自不同 backbone（DINOv3 / V-JEPA2），representation space 与 LAP 的输入 misaligned，直接 align 会丢信息。

LSP 用 duplicate initial frame $(v_0, v_0)$ 提供一个 "context anchor"，让 VLM context 不需要从头预测 dynamics，只需要与 context-conditioned 的 LSP 输出对齐。这是给 VLM 减负。

### 6.3 为什么 decoupled EMA

LAP 必须严格反映 dynamics（否则 $z$ 无意义），LSP 必须严格反映 VLM context（否则对齐目标无意义）。如果 joint train，两个 target 互相拉扯，容易 collapse 到 trivial solution。

Decoupled：
- backbone 用 LSP gradient 优化 → backbone 服务 VLM context
- query 用 LAP gradient 优化 → query 服务 dynamics extraction
- EMA 让信息缓慢同步，避免突然漂移

这种 "分工 + 缓慢同步" 的思路类似 MoCo 的 momentum encoder，但是是双向 asymmetric 的。

### 6.4 Layer 19 的 mid-layer 现象

深层 token 偏 output，被 task-specific (motion token prediction) 主导；浅层偏 input，visual feature 还没 abstracted；中层是 "abstraction sweet spot"。Flow-matching 接中层 embedding，相当于把 "general manipulation prior" 抽出来给 robot policy head 用。

---

## 7. 与相关工作的对比

### 7.1 LAPA / UniVLA (reconstruction-based)
- 用 FDM 重建 future frames 作为 latent action 的 supervision
- 优势：dense pixel target
- 劣势：FDM 难学 + background noise 污染 + 计算重（两阶段 86h vs 68h）

### 7.2 Being-H0 (前作)
- 用 MANO-based hand motion 作为 action token，next-token prediction
- 只能 leverage annotated data，无法用 wild video
- JALA 是其延伸，加入 alignment 机制支持 hybrid data

### 7.3 EgoVLA
- 直接从 egocentric video 学 VLA，用 hand motion 作为桥接
- 与 Being-H0 / JALA 思路相近，但 JALA 更明确地分离 "context prediction" 与 "dynamics grounding"

### 7.4 R3M / VC-1 / MVP
- 学习通用 visual representation for robot manipulation
- 间接 leverage video，没 explicit action 信号
- JALA 用 IDM 提供 explicit (虽然是 latent) action 信号

### 7.5 VPT / UniSim
- IDM 学 action，但通常需要 paired (state, action) 或 weak supervision
- JALA 用 alignment 代替 reconstruction，绕开 FDM

---

## 8. 我的几个思考与潜在 extension

1. **Perceiver 的可扩展性**：现在 LAP/LSP 是 2-layer Perceiver，如果 chunk 的 K=128 tokens 想扩展到更细粒度（比如 1024 tokens），Perceiver 的 fixed-query 设计能否 scale？或许可以试 hierarchical Perceiver。

2. **Flow matching 步数**：N=4 已经够用，因为 predictive embedding 已经 carry 了大部分 action 信息，flow matching 只做 "refinement"。如果 embedding 质量更高，是否可以 N=1（相当于 deterministic regression）？这会大幅简化 inference。

3. **Cross-embodiment 通用性**：现在 post-training 用 DiT head 适配 robot。如果换成 humanoid full-body（不只 hand），是否可以复用同样的 latent action space？MANO 只建模 hand，但人类 manipulation 涉及 arm/torso coordination。可以想象扩展到 SMPL-X (whole body) 的 latent action。

4. **Wild data 比例的极限**：现在 wild:lab = 2.5M:5M (33%)。如果继续 scale wild data 到 10x lab data，性能会持续提升还是 plateau？Fig 6 左暗示还能继续涨，但可能有上限（label noise 累积）。

5. **Action chunk size 的影响**：现在固定 15-frame chunk。如果 chunk 更长（30 frames），LAP 的 boundary frame 间隔更大，dynamics 信号更 noisy；chunk 更短（5 frames），细粒度更高但 chunk 数变多。这个 trade-off 没在 ablation 里探索。

6. **Decoupled EMA 的理论基础**：这个 asymmetric 设计很像 GAN 的 generator/discriminator 训练或 MoCo 的 momentum encoder，但方向是双向的。有没有可能用 game-theoretic formulation 来分析收敛性？

7. **Layer 选择的可学习化**：现在 hardcode 第 19 层。可以让 flow-matching head 学习一个 soft attention over multiple layers，自动决定 weighting，可能更 robust。

---

## 9. Reference Links

- Paper page: https://research.beingbeyond.com/jala
- Being-H0 (前作): https://arxiv.org/abs/2507.15597
- LAPA (reconstruction-based baseline): https://arxiv.org/abs/2410.11758
- UniVLA: https://arxiv.org/abs/2412.14846
- EgoVLA: https://arxiv.org/abs/2507.12440
- Ego4D dataset: https://ego4d-data.org/
- InternVL3: https://arxiv.org/abs/2306.14238
- DINOv3: https://arxiv.org/abs/2508.10104
- V-JEPA: https://ai.meta.com/research/publications/v-jepa-latent-video-prediction-for-visual-representation-learning/
- GRVQ (HiFi-Codec): https://arxiv.org/abs/2305.02765
- Perceiver: https://arxiv.org/abs/2103.03206
- Flow Matching: https://arxiv.org/abs/2210.02727
- LIBERO benchmark: https://arxiv.org/abs/2306.03310
- RoboCasa: https://robocasa.github.io/
- MANO model: https://mano.is.tue.mpg.de/
- HaWoR: https://arxiv.org/abs/2409.06007
- WiLoR: https://arxiv.org/abs/2503.02450
- GR00T N1.5: https://arxiv.org/abs/2503.14734
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246
- FAST tokenization: https://arxiv.org/abs/2501.09747
- DROID dataset: https://arxiv.org/abs/2403.12945
- DINOv3 GitHub (placeholder - 实际 page 待 release): https://github.com/facebookresearch/dinov3

---

## 10. 总结

JALA 的核心贡献是从 paradigm 上 shift 了 latent action 的学习方式：

- **从 reconstruction 到 alignment**：跳过 FDM 的 pixel-level 建模，直接对齐 predictive embedding 与 IDM 输出
- **从单一数据到 hybrid data**：通过 indicator loss + alignment loss 的组合，让 annotated 和 unlabeled data 在 unified latent action space 共训
- **从 unstable 到 stable**：decoupled EMA 解决 LAP/LSP 异质信号训练的 collapse 风险
- **从 next-token 到 flow-matching**：pretrain 用 token prediction 学 general pattern，post-training 用 flow-matching 做精确 robot control

效果显著：
- Hand motion generation Wild MPJPE 11.02 (Being-H0 16.91)
- LIBERO single-view SoTA in ≤3B class (92.3%)
- Real-world unseen setting 极强 robustness（掉 2-3 个点 vs baseline 大幅退化）
- 训练效率提升（68h vs LAPA 86h）

这条路 if 继续 scale（更多 wild video、更大 backbone、更长 chunk），有潜力成为 VLA pretraining 的 standard paradigm。它把 "如何从 human video 学 manipulation" 这个问题从 "如何精确重建 video" 转化为 "如何对齐 action concept"，这个 abstraction 更接近本质，也更 scalable。
