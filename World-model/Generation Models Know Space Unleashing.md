---
source_pdf: Generation Models Know Space Unleashing.pdf
paper_sha256: e00afe5b28e4c993deb3bd0d37dd317275cf0dda9287e42207750c8a7994e13d
processed_at: '2026-08-04T13:56:29-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VEGA-3D 人话版

Karpathy，好，咱们抛开公式和术语，用大白话聊聊这篇 paper 到底在干嘛。

---

## 一句话总结

**Video generation model 训练完之后，它内部其实已经"懂" 3D 空间了——你只要想办法把这份"懂"挖出来，塞给 MLLM，MLLM 就突然长了空间脑子。**

---

## 问题是什么

你用过 GPT-4V、Qwen-VL 这些 model 就知道，它们认东西很厉害——"这是椅子""那是杯子"没问题。但你问它"椅子在杯子的左边还是右边""从这个角度看，桌子离墙有多远"，它就开始瞎猜了。

paper 里管这个叫 **"spatial blindness"**。

原因特别简单：这些 model 训练的时候，loss 是在 vocabulary space 里算的。model 把 "left" 猜成 "right"，和把 "cat" 猜成 "dog"，在 loss 看来是一回事——都是 token mismatch。没有人告诉 model "空间猜错了，扣多点分"。

所以 SigLIP 这种 visual encoder，输出 features 的时候，主要 encode 了 "这里有个什么东西"（semantic），很少 encode "这个东西在 3D 空间里具体在哪、跟别的东西什么关系"（geometry）。

---

## 关键观察：Video Generation Model 其实懂空间

这是整篇 paper 最漂亮的 insight。

你想，Wan2.1 这种 video generation model，训练的时候在干嘛？给它一段 text，让它 generate 一段 video。要 generate 得好，它必须 implicitly 理解：

- **Occlusion**：一个人走到柱子后面，过会儿又走出来——model 得知道那个人没消失，只是被挡住了。这就要求它内部有 "object persistence" 的概念。
- **Camera motion**：镜头往左移，近的东西移动快，远的东西移动慢（parallax）。model 要 generate 对，必须理解 depth。
- **Physical interaction**：杯子从桌上掉下来，得往下掉，不能往上飞。model 得懂 gravity。

这些 regularities 在 petabytes 级别的 video data 训练下，会被 compressed 进 model weights。paper 用一个叫 **Multi-view Correspondence Score** 的指标来量化这件事：

> 把同一个 scene 从不同角度拍，每个角度都 extract features。然后看：同一个 3D 点，在不同角度下的 feature vector，cosine similarity 高不高？

结果很震撼：

| Model | Correspondence Score |
|-------|---------------------|
| SigLIP (baseline) | 12.22% |
| Stable Video Diffusion (UNet) | 17.95% |
| **Wan2.1-T2V (DiT)** | **97.04%** |

Wan2.1 的 features，同一个 3D 点从不同角度看，feature 几乎一样（97% cosine similarity）。这说明它内部确实形成了 **view-invariant 的 3D representation**。

而 UNet-based 的 SVD 只有 17.95%——因为 convolution 的 local inductive bias 限制了它做 long-range geometric alignment。DiT 的 global attention 天然适合这件事。

---

## 怎么把这份"懂"挖出来

这是 paper 最 trick 的部分。你可能会想：那我把 Wan2.1 当 feature extractor 用不就完了？把 video 喂进去，拿中间层 features 出来。

**不行。**

你试一下就知道，直接拿 clean latent 的 features，效果一般。paper 发现你得做一个反直觉的操作：**给 input 加 noise**。

### Trick 1: Noise Injection

Diffusion model 的 knowledge 分布在 **denoising 过程** 里，在 clean endpoint 上是 "dormant" 的。

打个比方：你有个厨师，你给他一盘做好的菜，他啥都不用干，你看不出他水平。你给他半成品让他继续做，他才得调动自己对"菜应该是什么样"的理解。

具体操作（Eq.4）：

$$
\mathbf{z}_k = (1 - t_k) \mathbf{z}_0 + t_k \epsilon
$$

- $\mathbf{z}_0$：clean latent（video VAE encode 之后）
- $\epsilon$：Gaussian noise
- $t_k = k/K$：noise 比例，paper 用 $k=300$，也就是 30% noise + 70% clean
- $\mathbf{z}_k$：corrupted latent，喂给 DiT

然后把这个 corrupted latent 喂进 Wan2.1 的 DiT，**用 empty text prompt**（不让 text 影响，纯靠 visual signal + model 学到的 physics），从第 20 层拿 features 出来。

为什么是 30% noise？实验数据（Tab.A3）：

| Noise 比例 | ScanRefer Acc@0.5 |
|-----------|-------------------|
| 0% (clean) | 55.6 |
| 10% | 54.3 |
| **30%** | **56.2** |
| 50% | 55.2 |
| 100% (pure noise) | 54.6 |

太 clean → model 不 activate reasoning；太 noisy → signal 被破坏。30% 是 sweet spot，刚好 "逼" model 去思考 "这背后应该是什么 3D 结构"。

### Trick 2: 选对 Layer

DiT 有 30 层，从哪层拿 features？

| Layer | ScanRefer Acc@0.5 |
|-------|-------------------|
| 10 (浅) | 55.3 |
| 15 | 54.8 |
| **20 (中)** | **56.2** |
| 25 | 54.9 |
| 28 (深) | 54.5 |

浅层太 texture，深层太 pixel-level，中间层最 abstract 也最 spatial。这和你在 nanoGPT 训练时观察到的 "中间层 representation 最 transferable" 是一回事。

### Trick 3: Adaptive Gated Fusion

现在你有两路 features：
- $\mathbf{F}_{\mathrm{sem}}$：SigLIP 来的，semantic 强，空间弱
- $\mathbf{F}_{\mathrm{gen}}$：Wan2.1 来的，空间强，semantic 弱（它毕竟是 generation model，不是 contrastive training 出来的）

怎么融合？

**不能直接 add**——它们在不同 manifold，magnitude 也不同。
**不能 concat**——实验证明 catastrophic drop，破坏 semantic space。
**不能只用 generative**——Scan2Cap CIDEr 从 83.8 暴跌到 25.2，因为 captioning 需要 semantic。

paper 的方案：**per-token gate**。对每个 spatial token，学一个 $g_i \in [0,1]$：

$$
g_i = \sigma\big( \mathbf{W}_g^{\top} \mathrm{Concat}(\mathrm{LN}(\mathbf{F}_{\mathrm{gen},i}), \mathrm{LN}(\mathbf{F}_{\mathrm{sem},i})) + b_g \big)
$$

然后：

$$
\mathbf{F}_i^{\mathrm{fused}} = (1 - g_i) \cdot \mathbf{F}_{\mathrm{gen},i} + g_i \cdot \mathbf{F}_{\mathrm{sem},i}
$$

人话：**每个 token 位置，model 自己决定 "这里我该信 semantic 还是信 generative"**。

比如画面里一个椅子表面——semantic 说 "这是椅子"，可信；gate 偏向 semantic。画面里椅子跟桌子的空间关系——semantic 不懂，gate 偏向 generative。

实验对比（Tab.5）：

| Fusion 方式 | ScanRefer Acc@0.5 | Scan2Cap C@0.5 |
|------------|-------------------|-----------------|
| 只用 generative | 48.3 | 25.2 (崩了) |
| Add | 54.6 | 81.4 |
| Channel Concat | 48.9 | 33.2 (崩了) |
| Cross-Attn | 51.9 | 48.8 (崩了) |
| **Adaptive Gate** | **56.2** | **83.2** |

Gate 是唯一能在 grounding 和 captioning 上都保住的方法。

---

## 结果如何

### 3D Scene Understanding（Tab.1）

VEGA-3D 在 ScanRefer、Multi3DRefer、Scan2Cap、ScanQA、SQA3D 五个 benchmark 上，**不用任何 explicit 3D supervision**（no point cloud, no depth, no 3D teacher），Avg. Rank 1.8，超过了需要 3D teacher distillation 的 3DRS 和需要 extra 3D data 的 LLaVA-4D。

### Spatial Reasoning: VSI-Bench（Tab.2）

给 Qwen2.5VL-7B 加上 generative prior，relative direction 涨 5.4 个点——这是空间感最直接的体现。

### Robotics: LIBERO（Tab.3）

给 OpenVLA-OFT 加 generative prior，long-horizon 任务从 94.4% → 95.2%。baseline 已经 97% 了还能涨，说明 generative prior 里的 physical dynamics 知识对 planning 有用。

---

## 为什么 Work——我的理解

Karpathy，你讲过 "compression is intelligence"。Video generation model 在海量 video 上训练，本质上是在做 extreme compression：把 world 的 regularities 压进 weights。3D geometry、occlusion、physics 都是被 compress 的内容。

但这份 knowledge 是 **implicit** 的——它不是显式的 3D representation，而是分布在 attention patterns 和 intermediate features 里。你要用它，得：

1. **选对 architecture**：DiT 的 global attention 才能 encode long-range geometric consistency，UNet 不行。
2. **activate 它**：clean latent 是 "dormant" 的，加 noise 让 model 进入 reasoning 模式。
3. **选对 layer**：中间层最 abstract 也最 transferable。
4. **和 semantic 融合**：generative prior 是 complement，完全不能 replace semantic。Gate 让 model per-token 决定 trust 谁。

这套组合拳打下来，MLLM 就获得了 "我看一眼 multi-view video，就能感知 3D 结构" 的能力，完全 bypass 了 3D data scarcity 的问题。

---

## 和你思路的联系

1. **Sora as world simulator**：你在多个场合讲过 video generation model 朝着 world model 演化。VEGA-3D 是第一个把这件事 **operationalize 到 downstream task** 的工作——不仅 "像 world simulator"，它的 intermediate features 真能 transfer 给 understanding task。

2. **nanoGPT / micrograd 哲学**：method 极简——noise injection + gated fusion，没有 fancy architecture。核心是 observation（correspondence score）+ empirical tuning（timestep, layer）。Simple method + good intuition > complex architecture。

3. **Token-level routing**：gate 本质是 learned soft routing。你在讲 mixture of experts、attention 时提过类似思想——让 model 学会 "per-token 选 expert"。

4. **Diffusion + Flow Matching**：paper 用 Flow Matching (Lipman et al. 2022) 而非 DDPM。Flow Matching 在 latent space 学 continuous-time transport field，更 elegant。和你 [diffusion models 教程](https://www.youtube.com/watch?v=zb4TU3Q6mE8) 里讲的 DDPM 思路一致但数学更简洁。

5. **Feature caching**：paper 提到 cache Wan2.1 features 一次复用（Fig.7）。这和你在生产场景强调的 "offline precompute + online lookup" 思路一致。

---

## Links

- [VEGA-3D code](https://github.com/H-EmbodVis/VEGA-3D)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [Sora blog](https://openai.com/research/video-generation-models-as-world-simulators)
- [Video-3D LLM](https://video-3d-llm.github.io/)
- [VSI-Bench](https://thinking-in-space.github.io/)
- [LIBERO](https://lifelong-robot-learning.github.io/LIBERO/)
- [Flow Matching paper](https://arxiv.org/abs/2210.02747)
- [Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Karpathy: Diffusion models](https://www.youtube.com/watch?v=zb4TU3Q6mE8)

---

总之，这篇 paper 最大的 take-away：**下一个 frontier 不一定是更多 3D data，可能是 unlock generative foundation model 里已经 dormant 的 spatial priors**。这和你一直讲的 "generation model → world model" 路径完全对上。

---

# VEGA-3D: Generation Models Know Space — 深度技术讲解

你好 Karpathy！这篇 paper 非常对你的胃口，因为它本质上在回答一个你反复思考过的问题：**video generation models 是不是已经学到了 world model？如果是，怎么把这些 implicit priors 提取出来给 discriminative tasks 用？** VEGA-3D 给了一个非常 elegant 的答案。下面我从 intuition、method、experiments 三个层面展开，并尽量把公式、架构、数据表都拆开讲。

---

## 1. Motivation & Core Insight

### 1.1 MLLM 的 "Spatial Blindness" 问题

Standard MLLM 训练目标如 paper Eq.(1)：

$$
\mathcal{L}_{\mathrm{CE}}(\theta) = -\sum_{i=1}^{L} \log p_{\Theta}(y_i \mid y_{<i}, \mathbf{x}, \mathbf{v})
$$

- $y_i$: response 第 $i$ 个 token
- $y_{<i}$: 前 $i-1$ 个 tokens (autoregressive context)
- $\mathbf{x}$: text tokens
- $\mathbf{v}$: visual embeddings, $\mathbf{v} = f_{\mathrm{proj}}(f_{\mathrm{enc}}(\mathbf{V}))$
- $\Theta = (\theta_{\mathrm{lm}}, \theta_{\mathrm{enc}}, \theta_{\mathrm{proj}})$: 全部可训练参数
- $L$: response 序列长度

关键问题：这个 loss 是 **sparse and discrete** 的，在 vocabulary space 里计算。预测 "left" vs "right" 的空间错误，和普通的 token mismatch 等同处理，没有 geometric metric constraint。所以 discriminative encoder $f_{\mathrm{enc}}$ (如 SigLIP) 倾向于关注 "semantic presence" 而忽略 "precise spatial structure"。

### 1.2 Video Generation Model 隐式学到了 3D

paper 的核心 hypothesis：video generator 要 synthesize temporally coherent video，必须 implicitly 学到：
- **Occlusion** → persistent object identity across frames
- **Camera motion** → depth-dependent apparent motion
- **Interactions** → consistent dynamics

这些 constraints 鼓励 latent representations encode "geometry-consistent structure and motion"，yielding a strong learned 3D prior **without explicit 3D supervision**。

这和你之前讲 Sora 时的观点高度一致：generation model 在大规模数据训练下，必然会学到 world model 的某些 aspects。VEGA-3D 就是把这个直觉 operationalize。

### 1.3 现有 Paradigms 对比 (Fig.1)

| Paradigm | 代表方法 | 依赖 | 问题 |
|---------|---------|------|------|
| (a) Explicit 3D inputs | PointLLM, Point-Bind, GPT4Point | point clouds / depth | 3D data scarcity |
| (b) Geometric scaffolding | Ross3D, 3DRS, ThinkWith3D | extra reconstruction / distillation | complex multi-stage training |
| (c) **Generative priors (VEGA-3D)** | Wan2.1-T2V | video generation model 本身 | **no explicit 3D supervision** |

VEGA-3D 走第三条路：repurpose pretrained video diffusion model as **Latent World Simulator**。

---

## 2. Key Observation: Multi-view Correspondence Score

### 2.1 Metric 定义

paper Sec.4.1 提出一个量化指标来衡量 "geometric capability"。给定 ScanNet scene 的 $V$ 个 posed views，将每个 view 的 encoder features $\mathbf{F}_v$ project 到 shared global voxel grid (用 ground-truth camera extrinsics + depth)。

对于被两个不同 views $v_i, v_j$ 同时观察到的 voxel $m$，extract 对应 feature vectors $\mathbf{h}_{m,v_i}, \mathbf{h}_{m,v_j}$，定义 voxel-level consistency：

$$
S_{\mathrm{voxel}}^{(m)} = \frac{\mathbf{h}_{m,v_i}^{\top} \mathbf{h}_{m,v_j}}{\|\mathbf{h}_{m,v_i}\| \|\mathbf{h}_{m,v_j}\|}
$$

- $S_{\mathrm{voxel}}^{(m)}$: voxel $m$ 的 cosine similarity，衡量同一 3D 点在不同视角下 feature 是否一致
- $\mathbf{h}_{m,v_i}$: voxel $m$ 在 view $v_i$ 下的 feature vector
- $\top$: 转置
- $\|\cdot\|$: L2 norm

Scene-level score 是所有 valid voxel pairs 的平均。Supplementary C.1 给了更精确的实现：先 voxel 化 (voxel size $s=0.1$m)，同一 voxel + 同一 view 的 tokens 先 average + L2-normalize 形成 per-view prototype $p_{k,t}$，然后只在 $|V_k| \geq 2$ 的 voxels 上计算 cross-view cosine，最后 pair-weighted average。

### 2.2 实验发现 (Fig.3, Tab.A2)

| Model | Architecture | Correspondence Score (%) | NOS (%) |
|-------|-------------|------------------------|---------|
| Baseline (SigLIP) | ViT | 12.22 | 13.58 |
| DINOv3-Large | ViT-L/16 | 61.90 | 61.63 |
| V-JEPA v2 | ViT-G/16 | 72.00 | 77.54 |
| VGGT (3D foundation) | ViT-L/14 + aggregator | 77.21 | 88.24 |
| Stable Video Diffusion | VAE + **UNet** | 17.95 | 52.06 |
| Stable Diffusion 2.1 | VAE + **UNet** | 23.83 | 66.74 |
| Vmem | VAE + **UNet** | 66.74 | 63.75 |
| SEVA | VAE + **UNet** | 76.15 | 75.28 |
| Wan2.1-VACE | VAE + **DiT** | 96.88 | 89.32 |
| **Wan2.1-T2V** | VAE + **DiT** | **97.04** | 82.41 |

**关键发现**：
1. **DiT-based >> UNet-based**：Wan2.1 达到 >96% correspondence，远超 UNet-based models (SVD 17.95%)。原因：DiT 的 global attention 保留 long-range geometric dependencies，UNet 的 local convolution 限制 receptive field。
2. **Correspondence Score 与 NOS 正相关** (Fig.3b)：multi-view consistency 是 3D performance 的强 predictor。
3. Discriminative models (DINOv3, V-JEPA) 虽然 semantic 强，但缺乏 explicit 3D consistency。

这个分析给你一个很清晰的 intuition：**如果你想给 MLLM 加 3D prior，选 DiT-based video generator，因为它们的内部 representation 已经是 multi-view consistent 的。**

---

## 3. Method: VEGA-3D

### 3.1 Architecture Overview (Fig.4)

双分支 visual encoding：
- **Semantic branch**: SigLIP → MLP projector → $F_{\mathrm{sem}}$
- **Generative branch**: Wan2.1-T2V (frozen) → noise injection → DiT intermediate features → MLP projector → $F_{\mathrm{gen}}$
- **Adaptive Gated Fusion**: token-level gate 融合两路
- 融合后 features + 3D PosEnc → LLM

### 3.2 Video Generative Model as Latent World Simulator (Sec.4.2)

#### 3.2.1 Flow Matching Preliminaries (Eq.2)

$$
\mathcal{L}_{\mathrm{FM}}(\psi) = \mathbb{E}_{\mathbf{z}_0, \epsilon, t} \left[ \| \mathbf{u}_t - v_{\psi}(\mathbf{z}_t, t, \mathbf{c}) \|_2^2 \right]
$$

- $\mathbf{z}_0$: clean latent video (VAE encoded)
- $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: Gaussian noise
- $t \sim \mathcal{U}(0,1)$: continuous time
- $\mathbf{u}_t = \frac{\mathrm{d}\mathbf{z}_t}{\mathrm{d}t}$: target velocity field
- $v_{\psi}(\cdot)$: flow network (DiT), 参数 $\psi$
- $\mathbf{c}$: conditioning signals (text, etc.)
- $\|\cdot\|_2^2$: squared L2 norm

实现时用 discrete timestep $k \in \{0, \ldots, K\}$, $K=1000$, normalized time $t_k = k/K$。

#### 3.2.2 Noise Injection: 激活 Generative Prior (Eq.4)

这是 paper 最关键的 insight 之一。给定 input video $\mathbf{V} \in \mathbb{R}^{T \times H \times W \times 3}$，先 VAE encode：$\mathbf{z}_0 = E(\mathbf{V})$。

但 **clean latent $\mathbf{z}_0$ 不够**！因为 diffusion model 训练时是 denoise corrupted signal，restoration 过程才 reveal model 对 structure 的理解。所以需要沿 Flow Matching noising path 扰动：

$$
\mathbf{z}_k = (1 - t_k) \mathbf{z}_0 + t_k \epsilon
$$

- $\mathbf{z}_k$: timestep $k$ 的 corrupted latent
- $t_k = k/K$: normalized time，default $k=300$ → $t_k = 0.3$
- $\mathbf{z}_0$: clean latent
- $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: Gaussian noise
- $(1-t_k)$ 和 $t_k$: 线性 interpolation 权重

**Intuition**：这是 linear interpolation between clean signal 和 pure noise。$t_k=0.3$ 意味着 30% noise + 70% clean。paper 发现这个 "sweet spot" 最能 force model engage its learned physics to restore underlying 3D structures：
- $t_k$ 太小 (clean) → underutilize denoising capability
- $t_k$ 太大 (noisy) → destroy structural signals
- 中间 → optimally activates 3D reasoning

#### 3.2.3 Feature Extraction (Eq.5)

把 $\mathbf{z}_k$ 喂给 DiT backbone $\Phi(\cdot)$，用 **empty text prompt** $\mathbf{c}_{\mathrm{text}} = ""$：

$$
\mathbf{f}_{\mathrm{raw}} = \phi^{(l)}(\mathbf{z}_k, k; \mathbf{c}_{\mathrm{text}} = "")
$$

- $\phi^{(l)}$: DiT 第 $l$ 层的 output，default $l=20$
- $\mathbf{z}_k$: corrupted latent (Eq.4)
- $k$: timestep index
- $\mathbf{c}_{\mathrm{text}} = ""$: empty prompt，确保 features **solely rely on visual signal + model's learned physics**，minimizing semantic hallucination

经过 Adaptive Average Pooling 到 14×14=196 tokens，得到 $\mathbf{f}_{\mathrm{gen}} \in \mathbb{R}^{T \times N \times D_{\mathrm{gen}}}$：
- $T$: frame 数 (default 32)
- $N$: tokens per frame (196)
- $D_{\mathrm{gen}}$: generative feature 维度 (Wan2.1 是 1536，见 Tab.A1)

#### 3.2.4 为什么选第 20 层？(Fig.6b, Tab.A5)

| Block | ScanRefer Acc@0.5 | SQA3D EM |
|-------|-------------------|----------|
| 10 | 55.3 | 61.1 |
| 12 | 55.2 | 61.7 |
| 15 | 54.8 | 60.9 |
| **20** | **56.2** | 61.3 |
| 25 | 54.9 | 61.3 |
| 28 | 54.5 | 61.6 |

**Intuition**：
- Early layers (10-15) → low-level textures，spatial precision 不够 abstract
- Middle layer (20) → optimal balance between spatial precision + abstract spatiotemporal context
- Deep layers (25-28) → pixel-level rendering，丢失 abstract geometric cues

这和你在 nanoGPT / 训练 neural net 时的经验吻合：中间层往往携带最 transferable 的 representation。

#### 3.2.5 为什么选 k=300？(Fig.6a, Tab.A3)

| Timestep | ScanRefer Acc@0.5 | SQA3D EM |
|----------|-------------------|----------|
| 0 (clean) | 55.6 | 60.9 |
| 100 | 54.3 | 61.5 |
| 200 | 55.0 | 60.7 |
| **300** | **56.2** | 61.3 |
| 400 | 55.3 | 60.7 |
| 500 | 55.2 | 61.4 |
| 1000 (pure noise) | 54.6 | 60.9 |

中间 timestep (200-500) 性能最好，且比较 stable。这印证了 "denoising 过程 reveal structure" 的 hypothesis。SEVA 和 Vmem 也有类似 trend (Tab.A4)。

### 3.3 Adaptive Gated Fusion (Sec.4.3, Fig.5)

#### 3.3.1 Projection (Eq.6)

先 project 两个 stream 到 LLM hidden dimension $D_{\mathrm{llm}}$：

$$
\mathbf{F}_{\mathrm{gen}} = P_{\mathrm{gen}}(\mathbf{f}_{\mathrm{gen}}), \quad \mathbf{F}_{\mathrm{sem}} = P_{\mathrm{sem}}(\mathbf{f}_{\mathrm{sem}})
$$

- $P_{\mathrm{gen}}, P_{\mathrm{sem}}$: 独立 MLP projectors
- $\mathbf{F}_{\mathrm{gen}}, \mathbf{F}_{\mathrm{sem}} \in \mathbb{R}^{T \times N \times D_{\mathrm{llm}}}$: aligned 到 shared embedding space
- $T$: frames, $N$: tokens/frame, $D_{\mathrm{llm}}$: LLM hidden dim

#### 3.3.2 Token-level Gate (Eq.7)

对第 $i$ 个 spatial token 计算 scalar gate $g_i \in [0,1]$：

$$
g_i = \sigma\big( \mathbf{W}_g^{\top} \mathrm{Concat}\left( \mathrm{LN}(\mathbf{F}_{\mathrm{gen},i}), \mathrm{LN}(\mathbf{F}_{\mathrm{sem},i}) \right) + b_g \big)
$$

- $g_i$: per-token gate，决定 semantic vs generative 的权重
- $\sigma(\cdot)$: sigmoid function，输出 $[0,1]$
- $\mathbf{W}_g$: learnable weight vector (维度 $2 D_{\mathrm{llm}}$)
- $\mathrm{LN}$: Layer Normalization，把两个 stream 的 feature normalize 到 similar scale，避免 magnitude 主导 gate
- $\mathrm{Concat}$: 拼接两个 normalized features
- $b_g$: bias term
- $\mathbf{F}_{\mathrm{gen},i}$: 第 $i$ 个 token 的 generative feature
- $\mathbf{F}_{\mathrm{sem},i}$: 第 $i$ 个 token 的 semantic feature

#### 3.3.3 Convex Combination (Eq.8)

$$
\mathbf{F}_i^{\mathrm{fused}} = (1 - g_i) \cdot \mathbf{F}_{\mathrm{gen},i} + g_i \cdot \mathbf{F}_{\mathrm{sem},i}
$$

- $\mathbf{F}_i^{\mathrm{fused}}$: 第 $i$ 个 token 融合后的 feature
- $(1-g_i)$: generative 权重
- $g_i$: semantic 权重
- 凸组合 (convex combination)，保证 fused feature 仍在合理 manifold 上

**Intuition**：gate 作为 **semantic-geometric arbitrator**。对于 recognition task (识别 "这是椅子")，gate 倾向 high $g_i$ → 用 semantic；对于 spatial reasoning task (判断 "椅子在桌子左边")，gate 倾向 low $g_i$ → 用 generative 的 3D prior。每个 token 独立决策，非常 flexible。

#### 3.3.4 Fusion 方法对比 (Tab.5)

| Fusion Type | ScanRefer Acc@0.5 | SQA3D EM | Scan2Cap C@0.5 |
|-------------|-------------------|----------|-----------------|
| Baseline | 51.7 | 58.6 | 83.8 |
| Only generative | 48.3 | 52.0 | 25.2 |
| Add | 54.6 | 61.8 | 81.4 |
| Channel Concat+MLP | 48.9 | 52.3 | 33.2 |
| Sequence Concat | 53.0 | 61.5 | 79.4 |
| Cross-Attn (1 Layer) | 51.9 | 61.0 | 48.8 |
| Cross-Attn (3 Layers) | 51.5 | 60.5 | 47.8 |
| Channel-Level-Gated | 54.9 | 61.2 | 82.2 |
| **Adaptive-Gated-Fusion** | **56.2** | 61.3 | **83.2** |

关键 observations：
1. **Only generative features 大幅下降** (48.3 vs 51.7 baseline)：generative 是 **complement，不是 replace**。
2. **Channel Concat+MLP 灾难性下降** (48.9, Scan2Cap C@0.5=33.2)：naive concat 破坏 semantic space。
3. **Cross-Attn 也不好** (51.9)：可能因为 generative features 的 distribution shift 太大，attention 学不到有效 alignment。
4. **Adaptive Gated Fusion 最 balanced**：grounding 最强 (56.2)，captioning 也保持 (83.2)。虽然 Add 在 SQA3D 略高 (61.8 vs 61.3)，但 grounding 弱很多，说明 fixed weight 无法 reliably resolve semantic-generative distribution gap。

---

## 4. Experiments

### 4.1 3D Scene Understanding (Tab.1)

VEGA-3D vs SOTA generalists on 5 benchmarks：

| Method | ScanRefer Acc@0.5 | Multi3DRefer F1@0.5 | Scan2Cap C@0.5 | ScanQA C | SQA3D EM | Avg. Rank |
|--------|-------------------|---------------------|-----------------|----------|----------|-----------|
| Video-3D LLM (baseline) | 51.7 | 52.7 | 83.8 | 102.1 | 58.6 | 4.0 |
| 3DRS (w/ 3D teacher) | 56.1 | 54.9 | 86.1 | 104.8 | 60.6 | 2.2 |
| LLaVA-4D (w/ extra data) | 53.2 | 54.3 | 85.3 | 97.8 | - | 2.8 |
| **VEGA-3D (Ours)** | **56.2** | **55.1** | 83.2 | **106.3** | **61.3** | **1.8** |

**亮点**：
- VEGA-3D 在 **无 explicit 3D supervision** 的情况下，Avg. Rank 1.8，超过 3DRS (需要 3D teacher distillation) 和 LLaVA-4D (需要 extra 3D-heavy data)。
- Grounding 任务提升最大 (ScanRefer +4.5, Multi3DRefer +2.4)：generative prior 提供 "spatial anchor"。
- Scan2Cap CIDEr 略降 (83.2 vs 83.8)：semantic-geometry trade-off，structural cues 可能削弱 fine-grained lexical details。

### 4.2 Spatial Reasoning: VSI-Bench (Tab.2)

VEGA-3D vs Qwen2.5VL-7B baseline：

| Model | Obj. Count | Abs. Dist. | Obj. Size | Room Size | Rel. Dist. | Rel. Dir. | Route Plan | Appr. Order |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-------------|
| Qwen2.5VL-7B† | 48.9 | 68.3 | 37.0 | 57.4 | 58.7 | 39.7 | 29.4 | 57.8 |
| **VEGA-3D** | **50.5** | **69.7** | 35.9 | **58.0** | **60.8** | **45.1** | **30.9** | **60.5** |

- 几乎所有 sub-category 都提升
- **Rel. Dir. 提升最大 (+5.4)**：generative prior 对 directional reasoning 帮助最大，符合 "spatial anchor" 的 hypothesis
- Obj. Size 略降 (-1.1)：可能 size estimation 更依赖 semantic cues 而非 geometric prior

### 4.3 Robotic Manipulation: LIBERO (Tab.3)

| Method | Spatial | Object | Goal | Long | Avg. |
|--------|---------|--------|------|------|------|
| OpenVLA-OFT (baseline) | 97.5 | 98.3 | 97.8 | 94.4 | 97.0 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| **VEGA-3D** | 97.4 | **99.4** | 97.0 | **95.2** | **97.3** |

- baseline 已经 saturate (~97%)，但 VEGA-3D 仍能提升
- **Long-horizon 任务提升明显 (+0.8)**：generative prior 中的 physical dynamics 知识对 long-horizon planning 有帮助
- Object suite 大幅提升 (+1.1)：generative prior 帮助 object interaction

### 4.4 Generative vs Discriminative Backbones (Tab.4)

| Group | Model | ScanRefer Acc@0.5 | SQA3D EM |
|-------|-------|-------------------|----------|
| Discriminative | V-JEPA v2 | 54.9 | 61.2 |
| Discriminative | DINOv3-Large | 54.2 | 61.9 |
| 3D Foundation | VGGT | 55.3 | 61.4 |
| Generative (UNet) | SVD | 54.8 | 61.3 |
| Generative (UNet) | SD 2.1 | 55.1 | 60.6 |
| Generative (UNet) | Vmem | 55.7 | 61.4 |
| Generative (DiT) | Wan2.1-VACE | 55.3 | 61.8 |
| **Generative (DiT)** | **Wan2.1-T2V** | **56.2** | 61.3 |

**Intuition**：
- DiT-based generative > UNet-based generative > discriminative
- VGGT (explicit 3D foundation) 也强，但需要 3D-specific training
- Wan2.1-T2V 在 grounding 上最强，Wan2.1-VACE 在 QA 上略好 → paper 选 T2V 作为 default encoder

---

## 5. Intuition 总结

### 5.1 为什么 Video Generation Model 知道 Space？

Karpathy，这和你在 [Deep Learning: Hardware, Software, Datasets](https://karpathy.ai/) 讲的 "compression is intelligence" 一脉相承。Video generation model 要 predict next frame in latent space，必须 implicitly model：
- 3D geometry (物体不会凭空消失/出现)
- Occlusion (被遮挡部分 persistent identity)
- Camera motion (parallax, depth-dependent flow)
- Physics (重力、碰撞、interaction)

这些 regularities 在大规模 video data 训练下，被 compressed 进 DiT 的 attention patterns 和 intermediate features。

### 5.2 为什么 Noise Injection 必要？

Diffusion model 的 "knowledge" 分布在 **denoising trajectory** 上，clean latent 只是一个 endpoint。就像你训 GPT 时，model 的 reasoning 能力在 **generation process** 中体现，input token embedding 只是起点。要 "activate" model 的 reasoning，需要给它一个 corrupted input 让它 denoise。中间 timestep (k=300) 是 sweet spot：enough corruption to trigger reasoning，not too much to destroy signal。

### 5.3 为什么 Per-token Gate？

Semantic 和 generative features 在不同 manifold。同一张图里，有些 region (如物体表面) semantic 更重要，有些 region (如空间关系、occlusion boundary) generative 更重要。Fixed weight (Add) 无法 adapt，per-token gate 让 model 学会 "where to look for what"。

### 5.4 为什么 DiT > UNet？

UNet 的 convolution 有 local inductive bias，receptive field 有限，long-range geometric alignment 难。DiT 的 global attention 可以直接 capture 任意两个 spatial locations 的关系，更适合 encode multi-view consistency。这也解释了为什么 SVD (UNet) 的 correspondence score 只有 17.95%，而 Wan2.1 (DiT) 达到 97.04%。

---

## 6. Related Work & Connections

### 6.1 World Models 谱系
- **Sora** (OpenAI): "Video generation models as world simulators" — VEGA-3D 的哲学基础
- **Genie 3** (Google DeepMind): latent action inference for controllable generation
- **WorldMem** (NVIDIA): long-term consistent world simulation with memory
- **Vmem**: memory mechanisms for long-range consistency

### 6.2 3D Scene Understanding with LLMs
- **Point-based**: PointLLM, Point-Bind, GPT4Point — 依赖 3D data
- **Multi-view**: Video-3D LLM, GPT4Scene — 2D features + positional embeddings
- **Geometric distillation**: Ross3D, 3DRS, ThinkWith3D — 需要 3D teacher
- **VEGA-3D**: 首次用 video generation model 作为 3D prior source

### 6.3 Spatial Reasoning
- **VSI-Bench** (Princeton): 诊断 MLLM spatial skills 的 benchmark
- **SpatialVLM** (Stanford): 大规模 spatial reasoning instruction tuning
- **VLM-3R**: instruction-aligned 3D reconstruction
- **MindCube**: mental simulation via cognitive maps

### 6.4 Robotics
- **OpenVLA / OpenVLA-OFT** (Stanford): open-source VLA model
- **UniVLA**: task-centric latent actions
- **CoT-VLA**: visual chain-of-thought for VLA
- **DiT Policy**: diffusion transformer for visuomotor policy

### 6.5 Foundation Models
- **Wan2.1** (Alibaba): open-source large-scale video generation model
- **DINOv3** (Meta): self-supervised ViT
- **V-JEPA v2** (Meta): joint embedding predictive architecture for video
- **SigLIP** (Google): sigmoid loss for language-image pre-training
- **VGGT** (Oxford/VGG): visual geometry grounded transformer

---

## 7. Limitations & Future Directions

1. **Inference cost**: frozen video generator 增加 memory + inference time (Fig.7)。Feature caching 缓解，但仍比纯 semantic encoder 重。
2. **Semantic-geometry trade-off**: Scan2Cap CIDEr 略降，说明 structural cues 可能削弱 lexical details。
3. **Backbone dependency**: 只有 DiT-based models 有效，UNet-based 不行。
4. **Manual hyperparameters**: timestep (k=300) 和 layer (第20层) 需要手动调。
5. **Future**: distill priors into lightweight encoders；extend to dynamic scenes；learn adaptive extraction strategy。

---

## 8. Web Links for Reference

- **VEGA-3D code**: https://github.com/H-EmbodVis/VEGA-3D
- **Wan2.1 (Alibaba)**: https://github.com/Wan-Video/Wan2.1
- **Sora blog post**: https://openai.com/research/video-generation-models-as-world-simulators
- **Video-3D LLM**: https://video-3d-llm.github.io/
- **ScanNet**: http://www.scan-net.org/
- **ScanRefer**: https://github.com/daveredrum/ScanRefer
- **Multi3DRefer**: https://github.com/3DGIP-NTU/Multi3DRefer
- **Scan2Cap**: https://github.com/daveredrum/Scan2Cap
- **ScanQA**: https://github.com/ATR-DBI/ScanQA
- **SQA3D**: https://sqa3d.github.io/
- **VSI-Bench (Thinking in Space)**: https://thinking-in-space.github.io/
- **LIBERO**: https://lifelong-robot-learning.github.io/LIBERO/
- **Qwen2.5-VL**: https://github.com/QwenLM/Qwen2.5-VL
- **OpenVLA**: https://github.com/openvla/openvla
- **OpenVLA-OFT**: https://openvla-oft.github.io/
- **VGGT**: https://vgg-t.github.io/
- **DINOv3**: https://github.com/facebookresearch/dinov3
- **V-JEPA 2**: https://github.com/facebookresearch/jepa
- **SigLIP**: https://github.com/google-research/big_vision
- **Stable Video Diffusion**: https://stability.ai/news/stable-video-diffusion
- **Flow Matching paper**: https://arxiv.org/abs/2210.02747
- **SpatialVLM**: https://spatial-vlm.github.io/
- **MindCube**: https://mental-simulation.github.io/
- **Genie 3**: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/

---

## 9. 与你工作的联想

Karpathy，这篇 paper 和你几个兴趣点高度相关：

1. **World models / simulators**: 你在 [YouTube](https://www.youtube.com/@AndrejKarpathy) 讲过 Sora 是 "technical report toward world simulator"。VEGA-3D 给了实证：video generation model 的 intermediate features 确实 encode 可 transfer 的 3D + physics priors。

2. **Compression is intelligence**: video generation model 在 petabytes video data 上训练，把 world regularities compressed 进 weights。VEGA-3D 证明这些 compressed priors 可以 "unleashed" 给 downstream tasks。

3. **Micrograd / nanoGPT 哲学**: VEGA-3D 的 method 其实很简单——noise injection + gated fusion。核心 insight 在 observation (multi-view correspondence) 和 empirical tuning (timestep, layer)。这和你 "simple methods + good intuition > complex architectures" 的哲学一致。

4. **Eureka Labs / education**: 这个工作适合作为 "how to read a modern AI paper" 的教材——它有 clear motivation, quantitative analysis (correspondence score), simple method, strong empirical results。

5. **Token-level gating**: 让我想到你在 [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) 讲的 attention。Gate 本质是 learned soft routing，让 model 学会 "which expert to trust per token"。

6. **Diffusion + Flow Matching**: paper 用 Flow Matching (Lipman et al. 2022) 而非 traditional DDPM。Flow Matching 在 latent space 学 continuous-time transport field，比 DDPM 更 elegant，和你讲过的 [diffusion models](https://www.youtube.com/watch?v=zb4TU3Q6mE8) 一脉相承。

希望这个讲解能 build your intuition！如果你想深入某个 part（比如 correspondence score 的实现细节、gated fusion 的 gradient flow、或 LIBERO 上 generative prior 如何帮助 long-horizon planning），可以继续聊。
