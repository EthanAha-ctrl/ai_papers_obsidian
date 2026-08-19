---
source_pdf: FlowWAM Optical Flow as a Unified Action.pdf
paper_sha256: a4dea6542be553e1bda65617085861650dab686b84014d6f50473f6d432a0283
processed_at: '2026-08-18T13:37:43-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FlowWAM 用人话说

## 一句话总结

**让机器人别再猜"我该往哪动"，而是先想象"整个画面会怎么流"，再照着做。**

---

## 问题是什么

想象你有一个很牛的 video generator（就是那种能生成视频的 AI）。它看过几百万小时的视频，知道东西怎么动、物理怎么运作。你想用它来控制机器人。

但有个尴尬的问题：**机器人说的是"关节角度"，video generator 听不懂这个语言**。

这就好比：你请了一个世界级的电影导演来拍片，但你只能给他发摩斯电码告诉他怎么拍。导演很厉害，但你们的沟通方式太别扭了。

之前大家试过几种方法：

1. **直接给数字**："关节1转30度，关节2转45度..." —— 精确但每个机器人说的"方言"不同，没法迁移
2. **学一套暗号**：让 AI 自己从视频里学一套压缩过的 action 编码 —— 灵活但信息丢太多，细节没了
3. **画个图告诉你**：在画面上标出"这里要动" —— 好一点，但只说了"哪里动"，没说"怎么动"

---

## FlowWAM 的 trick

**用 optical flow（光流）当 action 的语言。**

Optical flow 就是"每个像素往哪移动了多少"。你可以把它想象成：视频里每个像素都画一个小箭头，箭头方向是移动方向，长度是移动距离。

关键 trick 在于：**把光流转成一张彩色图片**。

- 方向 → 颜色（往右是红色，往上是绿色，等等）
- 大小 → 颜色深浅
- 亮度 → 拉满

这样转完之后，光流就变成了一张看起来有点像"彩色漩涡图"的 RGB 图片。**格式和普通视频帧一模一样**。

这一步看似简单，其实是整个 paper 的核心 insight：

> Video generator 的 VAE 是在 RGB 图片上训练的。你喂给它一个 raw 的 (u, v) tensor，它完全懵逼。但你喂给它一张 HSV 编码的 flow 图片，它就能正常处理了，因为 distribution 接近。

---

## 两种用法，一个模型

同一个模型，两种模式：

### Policy 模式（当 policy 用）

机器人看到当前画面 + 指令 → 模型同时生成"未来视频"和"未来光流" → 光流被 action expert 翻译成可执行的关节动作

### World model 模式（当世界模型用）

你给模型指定"我希望光流长这样" → 模型生成符合这个运动的未来视频

这就好比：同一个导演，既能"自己想怎么拍就怎么拍"（policy），也能"你给剧本他来拍"（world model）。

---

## 为什么有效

几个原因叠在一起：

**1. 信息密度高**

数字 action 是 14 个数字。光流是整张图每个像素都有信息。对于"指尖要精确移到那个杯子边缘"这种任务，光流直接告诉你"这个像素往那个方向移动"，不需要推理。

**2. 和 video prior 对齐**

Video generator 看过海量视频，知道"物体应该怎么动"。光流转成 RGB 格式后，直接复用了这个 prior。数字 action 没法做到这一点。

**3. 可以用无标注视频预训练**

光流不需要 action label，任何视频都能提取。所以可以拿人类第一视角的操作视频来预训练，学一套通用的"操作运动常识"，再迁移到机器人上。

---

## 结果如何

在 RoboTwin benchmark 上：
- FlowWAM：92.94%（带预训练）
- 之前最强的 Fast-WAM：91.88%
- 经典 VLA 方法 π0.5：42.98%

提升不算惊天动地，但 consistent。特别是 trajectory accuracy 这个指标，FlowWAM 是 64.26，第二名 54.27，**提升了将近 10 分**。这说明 dense 的光流条件确实让模型更"听话"。

---

## 我的 take

**好的部分：**

Insight 很 clean。把 action 转成 video generator 能消化的格式，这个 idea 本身就很 elegant。Ablation 也做得清楚——raw (u,v) flow 只有 72.3%，HSV-encoded flow 有 89.8%，差了 17 个点。这直接证明了"format alignment"这件事有多重要。

**存疑的部分：**

1. **提升在 saturate**。92.94% vs 91.88%，只差 1 个点。这个 benchmark 可能快到天花板了，需要更难的任务来区分方法。

2. **Action expert 太重了**。780M 参数去 cross-attend 一个 5B 的 video generator，这个比例不小。感觉用个轻量的 Q-Former 就够了，但 paper 没 ablate 这个。

3. **Wrist camera 的光流是 placeholder**。这是一个 hack。多视角场景下，手腕相机的运动信息完全丢了。

4. **依赖 simulator 提取 robot-only flow**。RoboTwin 是仿真环境，可以重放 action 去掉背景。真实世界没这个 luxury。

5. **没做 cross-embodiment transfer**。Claim 说光流是 embodiment-agnostic 的，但实验只在两个机器人上做，没展示 zero-shot 迁移。

---

## 更大的 picture

这篇 paper 让我想到一个更深的 question：

**在 generative AI for robotics 这条路上，到底什么是最好的 "interface representation"？**

VLA 用 language 当 interface —— 抽象但 information-poor
Diffusion Policy 用 raw actions —— 精确但 embodiment-specific
Latent actions —— 压缩但 bottleneck
Optical flow —— dense 但有 2D projection 限制
Point tracks —— long-horizon 但 sparse

每种 representation 都有 trade-off。FlowWAM 选了 "dense + video-native" 这个组合点，验证了它 work。但我怀疑这不是终点。

**我猜测的终极形态**：可能是某种 hierarchical 的 representation —— 高层用抽象的 task representation 做 long-horizon planning，中层用 flow 或 point tracks 做 motion planning，低层用 raw actions 做 control。每层之间有 clean 的 interface。

FlowWAM 把中层和低层打通了，但高层还是靠 language instruction。怎么把三层都 unify 起来，可能才是下一个 big thing。

---

# FlowWAM: Optical Flow as a Unified Action Representation for World Action Models - 深度技术解读

## 1. Paper 的核心 Insight 与 Motivation

Andrej，这篇 paper 触及了一个我一直觉得被低估的问题：**在 generative world models 中，action 应该如何被 represented，才能既 align with pretrained video priors，又 preserve 足够的 motion information 用于 control**。

让我先梳理一下现有 action representations 的 spectrum 和它们各自的问题：

**Numerical action tokens** (DreamZero, Cosmos Policy, UWM): 精确，但每个 robot embodiment 有不同的 action space。7-DOF single arm 和 14-DOF bimanual 的 action tokens 完全 heterogeneous，pretrained weights 无法迁移。更深层的问题是，numerical tokens 和 video generator 的 visual latent space 是**两个异构的 modality**，必须通过 action-specific heads 或 adapters 桥接，这破坏了 pretrained video prior 的 uniformity。

**Learned latent actions** (LAV, Motus): 通过 frame transitions 学习 abstract latent codes，理论上 embodiment-agnostic。但这些 latent codes 是高度 compressed 的 bottleneck representation，丢失了 dense spatial motion structure。当 robot 需要知道"指尖具体移动到哪里"时，latent code 的 information capacity 不够。

**Image-space actions** (ray maps, embodiment masks, multi-view action images): 把 action 渲染成 visual form，部分关闭了 modality gap。但这些 signals 是 **static spatial cues**——它们告诉你"where the action happens"，却没有告诉你"how each visible part moves across frames"。这是一个关键 distinction：static conditioning vs. temporally dense motion representation。

FlowWAM 的 key observation 是：**optical flow 天然地同时满足两个 property**：
1. 它是 dense per-pixel displacement field，记录了每个 pixel 如何移动
2. 通过 HSV color-wheel encoding，它可以转换为和 RGB frame format-identical 的 image，直接 compatible with frozen VAE encoder

这两个 property 的组合是独特的。Numerical actions 有 (1) 没有 (2)。Image masks 有 (2) 没有 (1)。Optical flow encoded as HSV image 同时拥有两者。

让我深入思考一下为什么这个 matters。Pretrained video generators (Wan, CogVideoX, etc.) 的 VAE 是在 RGB images 上训练的，它们的 latent space 编码的是 natural image statistics。如果你 feed 一个 raw (u, v) tensor 进去，distribution 完全 mismatch，pretrained weights 毫无用处。但如果你把 flow 转成 HSV-encoded RGB image，它的 distribution 就和 RGB image 接近——有 spatial smoothness、color coherence、edge structure——pretrained VAE 可以合理地处理它。

这个 insight 的 deep version 是：**为了让 action representation 能 leverage pretrained video priors，action 必须被 mapped 到一个 distribution 上，这个 distribution 和 pretraining data 的 distribution 足够接近**。Optical flow via HSV encoding 是一个 surprising but elegant 的 solution。

参考：[FlowWAM project page](https://flow-wam.github.io) | [Wan video generator](https://github.com/Wan-Video/Wan2.1)

---

## 2. 架构深度解析

### 2.1 Dual-Stream DiT 的设计哲学

```
RGB stream:   I_t --> VAE E --> z = E(V)   --> patch_embed_rgb --> DiT blocks (shared) --> output_head_rgb
Flow stream:  F_t --> VAE E --> z^f = E(F)  --> patch_embed_f   --> DiT blocks (shared) --> output_head_f
                                                |
                                                v
                                     Joint Self-Attention
                                     [RGB_tokens; Flow_tokens]
                                                |
                                     split back to streams
```

关键设计 decision 有几个层次：

**(a) Shared VAE Encoder**: RGB 和 flow 都通过同一个 frozen VAE。这有一个非 trivial 的 consequence——VAE 的 latent space 必须能 simultaneously represent natural images 和 HSV-encoded flow images。因为 HSV-encoded flow 在 RGB space 中看起来像一种特殊的 colorful pattern（类似 optical flow visualization），它确实落在 natural image distribution 的"tail"里，但不完全 OOD。Frozen VAE 可以处理它，但 reconstruction quality 可能不如 RGB。

**(b) Stream-specific patch embedding & output head**: 这是必须的，因为虽然 latent format 相同，但 input distribution 不同。RGB 的 patch 通常是 textured regions，flow 的 patch 通常是 smooth color fields。初始化时从 RGB weights deep copy，这是一个 warm-start trick。

**(c) Joint Self-Attention with Token Concatenation**: 这是最重要的设计。在每个 attention layer：

$$\text{Attn}([z_{\text{rgb}}; z_{\text{flow}}]) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

其中 Q, K, V 由 concatenated tokens 计算。这意味着 RGB tokens 可以 attend to flow tokens，反之亦然。这实现了 deep spatiotemporal interaction——RGB content 可以 query "这里应该有什么 motion"，flow 可以 query "这个 motion 对应什么 visual content"。

**(d) RoPE applied independently**: 每个 stream 用独立的 rotary position embedding。这意味着 RGB 和 flow 在 spatial dimension 上是 aligned 的（相同的位置编码），但在 token identity 上是 separated 的（通过 stream-specific embedding）。这是一个 subtle but important 的 design——position alignment 让 cross-stream attention 有 spatial meaning，identity separation 让 model 知道哪些 tokens 属于哪个 stream。

### 2.2 Two Operating Modes 的统一性

这是 paper 最 elegant 的部分。同一个 model，两种 inference mode：

**Policy Mode (action prediction)**:
```
Input:  I_0 (reference), τ (instruction)
        z_0^rgb = E(I_0), z_0^f = E(F_0)  [fixed, clean]
        z_{1:T}^rgb ~ N(0, I), z_{1:T}^f ~ N(0, I)  [noise init]
        
Process: Dual-stream DiT denoises both streams jointly
         Action expert reads per-layer hidden states --> action chunk

Output:  â_0 ∈ R^{N × d_c}  [executable actions]
```

**World-Model Mode (action-conditioned video generation)**:
```
Input:  I_0 (reference), τ (instruction), F_{1:T}^desired (target motion)
        z_0^rgb = E(I_0), z_0^f = E(F_0)  [fixed]
        z_{1:T}^f = E(F_{1:T}^desired)  [fixed, clean - this is the key difference!]
        z_{1:T}^rgb ~ N(0, I)  [noise init]
        
Process: Only RGB stream is denoised, flow stream provides conditioning

Output:  V_{1:T}  [future RGB video following the specified motion]
```

这个 unification 的深层含义是：**flow 是一个 dual-purpose representation**。作为 generation target，它 encodes "what motion to produce"；作为 conditioning signal，它 encodes "what motion to follow"。同一个 representation 在 input 和 output 两侧都 work，这比 numerical actions 优雅得多——numerical actions 很难作为 video generation 的 conditioning signal（你如何把一个 14D vector 注入到 video latent space？）。

### 2.3 Action Expert 的 Q-Former-like 设计

Action expert 是一个 ~780M parameter 的 AdaLN diffusion transformer：

```
Architecture:
- 30 layers (matches video DiT depth)
- hidden dim 1024, 16 heads, FFN 4096
- Per block:
  1. Self-attention over action tokens
  2. Cross-attention to [RGB hidden states; Flow hidden states] from video DiT layer i
  3. Cross-attention to T5 instruction tokens + proprioceptive token
- Output: flow-matching velocity of action chunk
```

这个设计让我想到 BLIP-2 的 Q-Former：一个 lightweight transformer 通过 cross-attention 从 frozen image encoder 中提取信息。这里 action expert 从 frozen video generator 的每一层 hidden states 中提取 information。

**为什么 per-layer cross-attention？** 不同的 DiT layer 编码不同 abstraction level 的 information。Early layers 可能编码 low-level motion patterns，late layers 可能编码 high-level task semantics。Action expert 可以从多个 abstraction level 中 aggregate 信息。

**为什么用 flow-matching objective for action prediction？** 这和 π0 的设计一致。Flow matching 比 DDPM 更 efficient，只需要 ODE integration 而不是 SDE。Action chunk 是 continuous 的，flow matching 可以 generate continuous trajectories。

参考：[π0 paper](https://www.physicalintelligence.company/blog/pi0) | [BLIP-2 Q-Former](https://arxiv.org/abs/2301.12597)

---

## 3. 关键公式的变量与含义深度讲解

### Equation 1: Flow RGB Encoding

$$F_t = \phi(\mathbf{f}_t): \quad \mathbf{H} = \frac{\mathrm{atan2}(v, u) + \pi}{2\pi}, \quad \mathbf{S} = \frac{\|\mathbf{f}_t\|}{m}, \quad \mathbf{V} = 1$$

让我详细讲一下每个变量的含义：

- $\mathbf{f}_t \in \mathbb{R}^{H \times W \times 2}$: 时间步 t 的 optical flow field。H, W 是 spatial dimensions。最后一维的 2 对应 horizontal 和 vertical displacement。
- $u, v$: flow field 的两个分量。$u$ 是 horizontal displacement (向右为正)，$v$ 是 vertical displacement (向下为正)。
- $\phi$: HSV color-wheel encoding 函数，将 2D flow vector 转换为 3D HSV color
- $\mathbf{H} \in [0, 1]$: hue channel，编码 flow 的**方向**。$\mathrm{atan2}(v, u)$ 返回 $[-\pi, \pi]$ 的角度，加 $\pi$ 后变为 $[0, 2\pi]$，除以 $2\pi$ 后归一化到 $[0, 1]$。不同方向对应不同颜色：右方向 → 红色，上方向 → 绿色，等等。
- $\mathbf{S} \in [0, 1]$: saturation channel，编码 flow 的**大小**。$\|\mathbf{f}_t\| = \sqrt{u^2 + v^2}$ 是 displacement magnitude。$m$ 是 normalization constant，paper 中用 25 px magnitude cap。
- $\mathbf{V} = 1$: value channel，固定为最大值，确保 color 是 fully saturated 的 bright color，没有 darkness variation。
- $F_t$: 转换后的 RGB image，format-identical to scene frame $I_t$

**Invertibility 的意义**：$\phi^{-1}(F_t)$ 可以恢复 numerical flow field。这意味着 flow RGB 是一个 lossless representation，没有 information bottleneck。这对于 action decoding 至关重要——action expert 需要精确的 motion 信息。

**工程细节**：paper 中用 25 px magnitude cap 来防止 large displacements saturating color encoding。0.5 px 以下的 displacement 被 threshold 到 zero，来 suppress flow noise in nearly static regions。这些 thresholds 是经验性的，但很重要。

### Equation 2: Stochastic Latent Conditioning

$$\tilde{\mathbf{z}}^f = (1-\sigma)\mathbf{z}^f + \sigma\epsilon^f, \quad \tilde{\mathbf{z}} = (1-\sigma)\mathbf{z} + \sigma\epsilon^r, \quad \sigma \sim \mathcal{U}[0,1]$$

- $\mathbf{z}^f, \mathbf{z}$: flow 和 RGB 的 clean VAE latents（从 ground-truth frames encode 得到）
- $\epsilon^f, \epsilon^r \sim \mathcal{N}(0, \mathbf{I})$: 独立采样的高斯噪声
- $\sigma \sim \mathcal{U}[0,1]$: 均匀采样的噪声水平，控制 noise injection 的强度
- $\tilde{\mathbf{z}}^f, \tilde{\mathbf{z}}$: 加噪后的 latents

**为什么需要这个 trick？** 

训练时 action expert 可以直接读取 clean VAE latents 对应的 hidden states。但 inference 时，action expert 看到的是 dual-stream DiT 生成的 latents，这些 latents 经历了 iterative denoising，可能仍然有 residual denoising error。这是典型的 train-test distribution mismatch。

Stochastic latent conditioning 通过在训练时随机加噪（50% 的 steps 加噪，50% 不加），让 action expert 见过各种 noise levels 的 latents，从而在 inference 时 robust to residual error。这和 DDIM 的 noise schedule alignment 思路类似。

**为什么 50% (p=0.5)?** 这是一个 trade-off。如果加噪太多，action expert 学不到 clean signal；如果加噪太少，action expert 在 inference 时 OOD。50% 是一个 reasonable default，但 paper 没有做 sensitivity analysis。

### Equation 3: Video Generation Loss

$$\mathcal{L}_{\mathrm{video}} = (1-\lambda_f)\mathcal{L}_{\mathrm{RGB}} + \lambda_f\mathcal{L}_{\mathrm{flow}}$$

- $\mathcal{L}_{\mathrm{RGB}}$: RGB stream 的 mean squared error，between predicted velocity field $v_\theta(\mathbf{z}_t, t)$ 和 target velocity
- $\mathcal{L}_{\mathrm{flow}}$: flow stream 的 mean squared error，between $v_\theta^f(\mathbf{z}_t^f, t)$ 和 target velocity
- $\lambda_f$: flow stream 的 weight，paper 中设为 0.1

**为什么 $\lambda_f = 0.1$ 而不是 1.0？** 

这反映了两个 stream 的 importance 不对等。RGB 是 primary output（video generation 的主要目标），flow 是 auxiliary 但 critical 的 action representation。如果 $\lambda_f$ 太大，model 可能过度 focus on flow prediction 而 sacrifice RGB quality；如果 $\lambda_f$ 太小，flow signal 学习不充分，action decoding 受影响。0.1 是一个 empirical sweet spot。

但这个 design choice 让我好奇：如果 $\lambda_f = 0.5$ 会怎样？Paper 没有 ablate 这个，这是一个 missed opportunity。

**Flow Matching Objective 的细节**：

Noisy latents 的构造：
$$\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\epsilon, \quad \mathbf{z}_t^f = (1-t)\mathbf{z}_0^f + t\epsilon'$$

- $t \sim \mathcal{U}[0,1]$: shared timestep for both streams（同一 batch 中 RGB 和 flow 用相同的 t）
- $\mathbf{z}_0, \mathbf{z}_0^f$: clean latents
- $\epsilon, \epsilon'$: 独立的高斯噪声

Model 预测 velocity fields $v_\theta(\mathbf{z}_t, t)$ 和 $v_\theta^f(\mathbf{z}_t^f, t)$，target 是 $\epsilon - \mathbf{z}_0$ 和 $\epsilon' - \mathbf{z}_0^f$（standard flow matching）。

**Shared timestep 的意义**：RGB 和 flow 用相同的 t 意味着它们在 denoising trajectory 上是 synchronized 的。这允许 cross-stream attention 在相同 noise level 下做 information exchange。

### Equation 4: Motion-Aware Reweighting

$$w_{\mathrm{motion}} = 1 + \alpha \cdot \frac{\langle|\mathbf{z}^f - \mathbf{z}_{\mathrm{ref}}^f|\rangle_c}{\max\langle|\mathbf{z}^f - \mathbf{z}_{\mathrm{ref}}^f|\rangle_c}$$

- $\mathbf{z}^f$: 当前帧的 flow latent
- $\mathbf{z}_{\mathrm{ref}}^f$: reference frame 的 flow latent（通常是第一帧）
- $\langle \cdot \rangle_c$: 对 latent channels 取 average
- $|\cdot|$: element-wise absolute value
- $\max$: spatial max（找到最大的 deviation）
- $\alpha$: boosting strength，paper 中设为 2.0
- $w_{\mathrm{motion}}$: per-location 的 loss weight

**这个公式在做什么？**

它计算每个 spatial location 的 flow deviation from reference，归一化到 [0, 1]，然后乘以 $\alpha$ 加到 1 上。所以：
- Static regions (deviation = 0): weight = 1
- Motion-rich regions (max deviation): weight = 1 + α = 3

这相当于给 motion-rich regions 3 倍的 loss weight，让 model 不要被 static background 主导。

**为什么需要这个？** Manipulation 的 flow 是 spatially sparse 的——motion 集中在 robot end-effector 和 manipulated object 周围，背景通常是 static。如果不 reweighting，loss 会被大量的 static pixels 主导，model 学到的只是"大部分 pixels 不动"，而忽略了关键的 manipulation motion。

这个 trick 在 video prediction literature 中常见，但这里的应用特别 well-motivated，因为 manipulation 的 motion sparsity 比一般 video 更极端。

### Equation 5: Total Objective

$$\mathcal{L} = \mathcal{L}_{\mathrm{video}} + \lambda_a \mathcal{L}_{\mathrm{action}}$$

- $\mathcal{L}_{\mathrm{video}}$: 上述 video generation loss
- $\mathcal{L}_{\mathrm{action}}$: action expert 的 per-chunk prediction loss（flow-matching objective on action chunks）
- $\lambda_a$: action loss weight，paper 中设为 1.0

$\lambda_a = 1.0$ 意味着 action prediction 和 video generation 同等重要。这是一个 reasonable default，但我觉得可能 action loss 应该更大一些，因为最终目标是 control。

---

## 4. 实验数据表的深度解读

### Table 1: RoboTwin 2.0 Policy Success Rates

让我提取关键的 average numbers：

| Method | Clean (%) | Random (%) | Type |
|--------|-----------|------------|------|
| π0.5 | 42.98 | 43.84 | VLA |
| X-VLA | 72.88 | 72.84 | VLA |
| Motus | 88.66 | 87.02 | WAM (latent action) |
| GigaWorld-Policy | 86.36 | 85.04 | WAM |
| X-WAM | 89.76 | 90.68 | WAM |
| Fast-WAM | 91.88 | 91.78 | WAM |
| **FlowWAM w/o PT** | 82.40 | 80.80 | WAM (flow) |
| **FlowWAM w/ PT** | **92.94** | **92.14** | WAM (flow) |

**关键 observations**：

**(a) Pretraining 的巨大影响**：没有 pretraining 的 FlowWAM 只有 82.40%，比 X-WAM (89.76%) 和 Fast-WAM (91.88%) 都低。但加了 EgoDex pretraining 后跳到 92.94%，提升了 10.54%。这说明 flow representation 本身不够，需要大规模 video pretraining 来 learn motion priors。

这是一个 important caveat：**flow 的 benefit 很大一部分来自于 action-unlabeled pretraining 的 scalability**，而不仅仅是 representation 本身的 superiority。

**(b) Clean vs Random 的差异**：FlowWAM w/ PT 在 Clean 上 92.94%，Random 上 92.14%，差距只有 0.8%。而 Motus 在 Clean 上 88.66%，Random 上 87.02%，差距 1.64%。FlowWAM 对 randomization 更 robust，这可能是因为 flow 的 dense spatial information 帮助 model 处理 visual variations。

**(c) 对 specific tasks 的分析**：

- **Hanging Mug**: 这是所有方法都表现差的任务。π0.5 只有 3%，X-VLA 23%，Fast-WAM 58%，FlowWAM w/ PT 65%。这个 task 需要精确的 spatial reasoning（把 mug handle 挂到 hook 上），flow 的 dense spatial information 帮助很大。
  
- **Pick Diverse Bottles**: 涉及 multiple objects 的 picking。π0.5 只有 5%，FlowWAM w/ PT 90%。这个 task 的 challenge 是 visual diversity，flow 帮助 model focus on correct object 的 motion。

- **Stack Blocks Three**: 需要 long-horizon planning。π0.5 15%，Motus 91%，FlowWAM 99%。这里 flow 的 advantage 明显。

### Table 2: WorldArena World Modeling

| Method | Cond. | Traj. Acc. ↑ | EWMScore ↑ |
|--------|-------|---------------|------------|
| CogVideoX | Text | 34.79 | 58.79 |
| Veo 3.1 | Text | 11.36 | 57.77 |
| Wan 2.6 | Text | 12.18 | 59.80 |
| Cosmos-Predict 2.5 (text) | Text | 11.60 | 53.06 |
| ABot-PhysWorld (text) | Text | 31.50 | 62.63 |
| Cosmos-Predict 2.5 (action) | Action | 27.49 | 54.29 |
| IRASim | Action | 35.92 | 56.15 |
| Ctrl-World | Action | 48.20 | 59.98 |
| Vidar | Img-Act | 21.26 | 51.92 |
| Genie Envisioner | Img-Act | 2.63 | 41.37 |
| GigaWorld-1 | Img-Act | 54.27 | 62.34 |
| **FlowWAM** | **Flow** | **64.26** | **63.71** |

**关键 observations**：

**(a) Trajectory Accuracy 的巨大 gap**：FlowWAM 64.26 vs 第二名 GigaWorld-1 54.27，提升了 18.4%。这是所有 metrics 中最大的 margin。Text-conditioned models 普遍 < 35%，说明 text 不足以 specify 精确 trajectory。Numerical action conditioning 也不行（IRASim 35.92，Cosmos 27.49）。Image-space actions 有改进（GigaWorld-1 54.27）。只有 dense optical flow conditioning 能达到 64+。

**(b) EWMScore 的综合性**：EWMScore 是 16 个 metrics 的 average，FlowWAM 63.71 vs 第二名 ABot-PhysWorld 62.63，提升 1.08。这个 margin 不大，但要注意 FlowWAM 在 visual quality metrics (IQ, AQ) 上不是最好的——CogVideoX 和 Veo 在 image quality 上更强。FlowWAM 的优势在 motion 和 physics metrics 上。

**(c) Action Following (AcF) 的异常**：FlowWAM 的 AcF 只有 3.50，而 Veo 3.1 是 8.52，Wan 2.6 是 9.92。这个 metric 衡量 model 对 different action instructions 的响应 diversity。FlowWAM 低可能是因为它的 flow conditioning 让 rollouts 更 deterministic，减少了 diversity。这是一个有趣的 trade-off：精确 control vs. diverse generation。

### Figure 4: Ablation Studies

**Policy Mode Ablation (RoboTwin)**:

| Variant | Succ. (%) |
|---------|-----------|
| Numerical actions | 69.8 |
| Raw (u, v) flow | 72.3 |
| w/o flow-loss reweighting | 83.9 |
| w/o stochastic AE cond. | 82.1 |
| FlowWAM (full) | 89.8 |

**Critical analysis**：

1. **Numerical actions (69.8%) vs FlowWAM (89.8%)**: 差距 20%。这是 representation 本身的 contribution。
2. **Raw (u, v) flow (72.3%)**: 比 HSV-encoded flow 差 17.5%。这验证了 format alignment 的关键性——pretrained VAE 不能处理 raw (u, v) tensors，因为 distribution 完全 mismatch。这是 paper 最强的 ablation evidence。
3. **w/o reweighting (83.9%)**: 差 6%。Motion-aware reweighting 确实重要，但不是最 critical 的。
4. **w/o stochastic cond. (82.1%)**: 差 7.7%。Train-test distribution alignment 也很重要。

**World Mode Ablation (WorldArena)**:

| Conditioning | EWMScore |
|--------------|----------|
| Text only | 49.31 |
| Numerical actions | 54.18 |
| Flow actions (u, v) | 56.72 |
| Image actions (masks) | 57.84 |
| FlowWAM (full) | 65.23 |

**这个 ablation 的 hierarchy 非常清晰**：
- Text (49.31) → Numerical (54.18): +4.87，action information 有用但 limited
- Numerical (54.18) → Raw flow (56.72): +2.54，dense motion 比 scalar 好
- Raw flow (56.72) → Image masks (57.84): +1.12，visual format 有帮助
- Image masks (57.84) → Full HSV flow (65.23): +7.39，dense + format-aligned 是关键

**最后一步的 +7.39 是最大的 jump**，这正好验证了 paper 的核心 thesis：**dense motion + format alignment with pretrained video generator 的组合是 essential**，缺一不可。

参考：[RoboTwin 2.0](https://github.com/TianxingChen/RoboTwin) | [WorldArena](https://arxiv.org/abs/2602.08971)

---

## 5. 训练 Pipeline 与工程细节

### Two-Stage Training Strategy

**Stage 1: Action-Free Motion Pretraining**
- Dataset: EgoDex (egocentric human manipulation videos, no action labels)
- Objective: $\mathcal{L}_{\mathrm{video}}$ only
- Learning rate: $5 \times 10^{-5}$
- Trainable: dual-stream DiT only
- Resolution: 320 × 256
- Frame buckets: {17, 33, 49, 65, 81}
- Effective fps: 15 (subsampled from 30)

**Stage 2: Joint Embodied Training**
- Dataset: RoboTwin 2.0 (50 tasks, Clean 50 demos + Random 500 demos each)
- Objective: $\mathcal{L}_{\mathrm{video}} + \lambda_a \mathcal{L}_{\mathrm{action}}$
- Learning rate: $1 \times 10^{-4}$
- Trainable: full DiT + flow stream + action expert
- Loss weights: $\lambda_f = 0.1, \lambda_a = 1.0, \alpha = 2.0$
- Pixel frames: 9 → action steps: 32 (temporal stride 4)
- Inference: video denoising 25 steps, action denoising 50 steps

**关键工程细节**：

**(a) Robot-Only Flow Extraction**: 对于 RoboTwin，paper 用 SAPIEN simulator 重放 robot joint actions，渲染 robot-only frames（static background），然后 RAFT 提取 flow。这避免了 object motion 和 lighting artifacts 的干扰，提供 cleaner supervision。这是一个聪明的 trick，但限制了 applicability——real-world data 没有这样的 simulator access。

**(b) Multi-View T-Shape Tiling**: head camera (320×256) 在上，两个 wrist cameras 在下，组成 320×384 tile。但 wrist camera 的 flow 用 constant placeholder，因为 wrist-view robot-only flow 不可用。这是一个 limitation——wrist camera 的 motion 信息丢失了。

**(c) Magnitude Cap & Noise Threshold**: 25 px cap 防止 large displacement saturating color encoding，0.5 px threshold suppress noise。这些 thresholds 是 empirical 的。

**(d) Ref-Aug Strength 0.1**: 对 conditioning frame 加小 noise，模拟 autoregressive rollout 中的 observation error。这是 video prediction 中的 standard trick。

参考：[EgoDex](https://egodex.github.io) | [RAFT optical flow](https://github.com/princeton-vl/RAFT) | [SAPIEN simulator](https://sapien.ucsd.edu/)

---

## 6. Critical Analysis 与相关联想

### 6.1 Optical Flow 的 Fundamental Limitations

虽然 paper 展示了 flow 的强大效果，但 optical flow 本身有一些 inherent limitations：

**Occlusion Handling**: 当 object 被遮挡时，flow 无法表示被遮挡 pixels 的 motion。在 manipulation 中，gripper 抓取 object 时会遮挡部分 object surface。Flow 只能看到 visible surface 的 motion，看不到 occluded regions。这可能是 action expert 还需要 RGB stream 的原因——RGB 提供 appearance 信息来 complement flow 的 motion 信息。

**Aperture Problem**: Optical flow 只能 measure normal component of motion（沿 edge 方向的 motion 是 ambiguous 的）。对于 textureless regions，flow estimation 不稳定。这解释了为什么 paper 用 RAFT（all-pairs field transforms）而不是 simpler methods——RAFT 通过 global reasoning 来缓解 aperture problem。

**2D Projection Loss**: Optical flow 是 3D scene motion 的 2D projection。当 robot arm 朝向 camera 方向移动时，flow magnitude 很小，无法反映真实的 3D motion。这对于需要 depth reasoning 的任务（如精确的 z-axis 插入）可能受限。Scene flow (3D motion) 可能是更好的 representation，但需要 depth input。

参考：[Aperture problem in optical flow](https://en.wikipedia.org/wiki/Motion_perception#The_aperture_problem)

### 6.2 与 Latent Action Models 的 Information-Theoretic 对比

LAV (Latent Action Pretraining from Videos) 学习一个 VQ-VAE-like 的 latent action codebook。每个 transition 被 encode 成一个 discrete code。这是高度 compressed 的 representation。

从 information theory 角度：
- **Latent actions**: high compression, low bandwidth (~log2(V) bits per transition)
- **Optical flow**: low compression, high bandwidth (H×W×2 floats per frame)

Trade-off：
- Latent actions 更 efficient 但可能 bottleneck information
- Optical flow 更 expressive 但占用更多 capacity

FlowWAM 的 evidence suggests that for manipulation, the dense representation wins。但这是否 always true？对于 simple tasks（如打开 drawer），latent action 可能足够。对于 complex bimanual tasks（如 fold towel），dense flow 的 spatial information 更 critical。

**Open question**: 能否设计一个 adaptive representation，根据 task complexity 自动选择 compression level？

参考：[Latent Action Pretraining (LAV)](https://latentactionpretraining.github.io)

### 6.3 与 Video Prediction Policies 的对比

Video Prediction Policy (VPP) 等 generate-then-infer 方法：
1. Generate future video (pixels)
2. 用 inverse dynamics model 从 pixels 恢复 actions

FlowWAM 的 action expert 不从 pixels 恢复，而是从 video generator 的 hidden states 读取。这避免了 pixel-to-action 的 information bottleneck——pixels 是 lossy compression of hidden states，直接从 hidden states 读取更 efficient。

但 VPP 的 advantage 是 modularity——video generator 和 action decoder 可以独立训练。FlowWAM 的 action expert 必须和 video generator joint train（至少 in Stage 2），这增加了 training complexity。

**Speculation**: 未来可能有 hybrid approach——先用 action-unlabeled video pretrain video generator (像 FlowWAM Stage 1)，然后用 small inverse dynamics model 从 generated video 恢复 actions，不需要 joint training。这样更 scalable。

参考：[Video Prediction Policy](https://proceedings.mlr.press/v268/hu25a.html)

### 6.4 与 JEPA 的 Representation Learning 哲学对比

Yann LeCun 的 JEPA (Joint-Embedding Predictive Architecture) 哲学是：在 latent space 做 prediction，不生成 pixels。这避免了 generative modeling 的 expense 和 mode collapse 问题。

FlowWAM 是 generative approach——它生成 pixels 和 flow。从 sample efficiency 角度，JEPA 可能更好（不需要 reconstruct pixels）。但 generative approach 的优势是：
1. **Planning**: 可以 visualize future scenarios，便于 human interpretation 和 debugging
2. **Conditioning**: 可以用 future state 作为 explicit conditioning
3. **Interpretability**: generated videos 可供 inspection

FlowWAM 的 WorldArena 评估中包含 JEPA Similarity metric，但这是用 V-JEPA encoder 计算 feature similarity，不是 JEPA 方法本身。

**Deeper question**: 能否设计一个 JEPA-like architecture，在 latent space predict flow latents (而不是 RGB pixels)，然后 decode flow 到 actions？这样可能更 efficient。Paper 没有探索这个方向。

参考：[V-JEPA](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-video-model-joint-embedding-predictive-architecture/) | [JEPA paper](https://arxiv.org/abs/2301.08243)

### 6.5 Point Tracks vs. Dense Optical Flow

ATM (Any-point Trajectory Modeling) 用 sparse point tracks 作为 representation。每个 track 是一个 pixel 在时间上的 trajectory。

对比：
- **Optical flow**: dense (per-pixel), short-term (frame-to-frame)
- **Point tracks**: sparse (selected points), long-term (entire trajectory)

Manipulation 中，我们可能需要 long-term motion planning（如 fold towel 的 multi-step motion）。Optical flow 只能 capture frame-to-frame motion，对于 long-horizon 需要多次 replanning。Point tracks 可以 represent 整个 trajectory，但 sparse。

**Hybrid idea**: 用 optical flow 做 short-term control，用 point tracks 做 long-term planning。Paper 的 future work 提到 "extending flow-based planning to longer temporal horizons"，这个方向值得探索。

参考：[ATM paper](https://arxiv.org/abs/2401.00025)

### 6.6 Action Expert 的 Parameter Efficiency Concern

Action expert 有 780M parameters，相对于 5B 的 video generator 是 ~15%。这个比例不算小。考虑到 action expert 只是 cross-attention + prediction head，为什么需要这么多参数？

可能原因：
1. 30 layers 和 video DiT 深度匹配，为了 per-layer cross-attention
2. Hidden dim 1024 + FFN 4096 是 standard transformer sizing
3. Action chunk prediction 是 high-dimensional output (32 × 14 = 448 dims)

但我觉得这个 design 可能 over-engineered。一个 lightweight Q-Former (like BLIP-2's 几十M parameters) 可能就足够。Paper 没有 ablate action expert 的 size，这是一个 missed experiment。

### 6.7 Stochastic Conditioning 的深层含义

Eq. 2 的 stochastic latent conditioning 不只是 train-test alignment trick，它还有一个 deeper implication：

**FlowWAM 的 action expert 在训练时见过各种 noise levels 的 latents，这意味着它学会了从 noisy motion plan 中 robust 地 decode actions**。这对于 real-world deployment 很重要——real sensors 有 noise，generated videos 有 artifacts，action expert 需要容忍这些 imperfections。

这个 idea 可以推广到其他 generative policies。例如，π0 的 action expert 直接从 VAE latents 读取，如果加 stochastic conditioning 可能也会 benefit。

### 6.8 关于 Embodiment Transfer 的 Potential

Paper 强调 optical flow 是 embodiment-agnostic——不同 robot 的 action 都会产生 visual motion，flow 不关心是 7-DOF 还是 14-DOF。

但 paper 的实验只在 Franka (single arm) 和 ARX (bimanual) 上做，没有跨 embodiment 的 zero-shot transfer 实验。真正测试 embodiment-agnostic claim 需要：
1. 在 robot A 上训练
2. 在 robot B 上 zero-shot evaluate（with same action expert retraining? or not?）

Flow 的 embodiment-agnostic property 可能允许：
- 用 robot A 的 flow pretrain video generator
- 用少量 robot B 的 data fine-tune action expert
- Video generator 不需要 retrain

这是一个 promising 的 future direction，paper 没有探索。

参考：[Cross-embodiment transfer in robot learning](https://robotics-transformer-x.github.io/)

### 6.9 关于 Long-Horizon Planning 的 Challenge

Paper 的 RoboTwin 评估是 single-task 的，每个 task 可能持续 10-30 seconds。对于 longer horizons（multi-task chaining），flow-based representation 可能需要：

1. **Hierarchical planning**: high-level 用 language 或 latent goals，low-level 用 flow
2. **Sub-goal decomposition**: 把 long-horizon task 分解成 flow-based sub-goals
3. **Memory mechanism**: 用 memory bank 存储 past flow trajectories

Paper 的 future work 提到 "extending flow-based planning to longer temporal horizons"，但没有具体方案。我认为 point tracks (long-term) + optical flow (short-term) 的 hybrid 是一个 promising direction。

### 6.10 与 Diffusion Policy 的关系

Diffusion Policy (Cheng Chi et al.) 用 diffusion models 直接生成 action chunks，不经过 video generation。这是 simpler 的 approach。

FlowWAM 可以看作 Diffusion Policy 的一个 augmented version：
- Diffusion Policy: noise → action chunk (直接)
- FlowWAM: noise → flow video → action chunk (经过 video prior)

中间的 flow video 是 inductive bias——它 forces action predictions to be consistent with visual motion patterns。这个 inductive bias 在 data-scarce regimes 可能有用，但在 data-rich regimes 可能是 constraint。

**Speculation**: 在 large-scale robot data (如 Open X-Embodiment) 上，Diffusion Policy 可能 catch up，因为 video prior 的 inductive bias 不再 needed。但目前在 per-task 50-500 demos 的 regime，video prior 的 benefit 明显。

参考：[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | [Open X-Embodiment](https://robotics-transformer-x.github.io/)

---

## 7. 关于这篇 Paper 的 Overall Assessment

让我从 Karpathy 的角度做一个 overall assessment：

**Strengths**:
1. **Clean insight**: optical flow 作为 unified action representation 是一个 elegant idea，well-motivated by format alignment with pretrained video generators
2. **Strong empirical evidence**: ablations 清楚地显示了 format alignment (raw vs HSV flow) 和 reweighting 的 contribution
3. **Unified framework**: 同一个 model 同时支持 policy 和 world modeling，这是少见的
4. **Scalability**: action-unlabeled pretraining 的 ability 是 practical 的 advantage

**Weaknesses**:
1. **Marginal gains over baselines**: 在 RoboTwin 上，FlowWAM w/ PT (92.94%) vs Fast-WAM (91.88%) 只提升 1.06%，已经在 saturating regime
2. **Limited cross-embodiment experiments**: 没有 zero-shot transfer 实验，embodiment-agnostic claim 没有被严格验证
3. **Action expert 的 over-engineering**: 780M parameters 可能 overkill，没有 ablate
4. **Wrist camera flow 的 placeholder**: 这是一个 hack，wrist-view motion 信息丢失了
5. **Reliance on simulator for robot-only flow**: real-world data 没有这样的 capability

**What I would have done differently**:
1. 做 cross-embodiment zero-shot transfer 实验（train on Franka, test on ARX）
2. Ablate action expert 的 size（780M vs 100M vs 50M）
3. Compare with point tracks representation
4. Test on longer-horizon multi-task chaining
5. Explore wrist camera flow extraction（maybe via neural rendering）

**Overall**: 这是一个 well-executed paper，insight clean，experiments solid。Flow 作为 unified action representation 的 idea 值得 pursue，尤其是 action-unlabeled pretraining 的 scalability。但 empirical gains 相对 marginal，需要更多 cross-embodiment 和 long-horizon 的 evidence 来 fully validate the approach。

---

## 8. Web Links 汇总

- FlowWAM project: https://flow-wam.github.io
- Wan video generator: https://github.com/Wan-Video/Wan2.1
- RoboTwin 2.0: https://github.com/TianxingChen/RoboTwin
- WorldArena: https://arxiv.org/abs/2602.08971
- EgoDex: https://egodex.github.io
- RAFT optical flow: https://github.com/princeton-vl/RAFT
- π0: https://www.physicalintelligence.company/blog/pi0
- Flow Matching paper: https://arxiv.org/abs/2210.02747
- Latent Action Pretraining (LAV): https://latentactionpretraining.github.io
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- V-JEPA: https://ai.meta.com/blog/v-jepa-yann-lecun-ai-video-model-joint-embedding-predictive-architecture/
- ATM (point tracks): https://arxiv.org/abs/2401.00025
- SAPIEN simulator: https://sapien.ucsd.edu/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- BLIP-2 Q-Former: https://arxiv.org/abs/2301.12597

---

希望这个深度解读能 build 你的 intuition about this paper。核心 takeaway 是：**action representation 的 format alignment with pretrained model 的 input distribution 是 critical 的**，optical flow via HSV encoding 是一个 elegant 的 solution，同时提供了 dense motion supervision 和 video-native format。未来的方向是探索 cross-embodiment transfer 和 long-horizon planning 的 capability。
