---
source_pdf: OmniForcing Unleashing Real-time Joint.pdf
paper_sha256: 0ca7050d400a00f97cf3c36935144dfcdf97baedab63c803e4d68b1a128e73ab
processed_at: '2026-08-05T23:13:27-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Karpathy 你好，如果用最直觉的“人话”来拆解 OmniForcing，我们可以把整个故事想象成**怎么把一个要求“全局视野”的慢性子画师，改造成一个能边画边播的快嘴直播主播**。

这里面的核心矛盾，其实是多模态物理频率的不对称在 causal attention 下引发的信息论灾难。

---

### 1. 为什么慢？为什么不能直接套用 video-only 的方法？

现有的联合音视频生成模型（比如 teacher LTX-2）像一个完美主义者：画 5 秒钟的片段，它必须同时看到第 0 秒和第 4.9 秒的所有信息才能动笔。这种 bidirectional attention 导致计算量呈平方级爆炸，生成 5 秒 480p 视频要 197 秒（Time-To-First-Chunk, TTFC）。

之前 CausVid 和 Self-Forcing 成功把纯视频模型改造成了“只看过去、不看未来”的 causal autoregressive 模型，实现了流式直播。但是，一旦你想把这套思路套到“视频+音频”双流架构上，模型直接崩溃 NaN。根本原因在于**视频和音频在 latent space 里的信息密度差距太悬殊**。

---

### 2. 核心危机：Softmax 坍塌与梯度爆炸

我们来看物理频率：经过 VAE 压缩后，视频 latent 是 $f_v = 3$ FPS，音频 latent 是 $f_a = 25$ FPS。

如果你强行按 token 做 causal mask，在一秒的 block 内，视频有 $3 \times 384 = 1152$ 个 tokens（空间 patchify 展开），而音频只有可怜的 $25$ 个 tokens。

当 audio stream 刚开始生成新 block 的第一个 token 时，它能看到的过去 history 极度稀疏。假设没有任何 stabilizer，它只能看到 Global Prefix 的 1 个 audio token。此时 Softmax 的分母只有 $e^{x_1}$，注意力分布退化成一个 one-hot 向量，信息熵 $H \to 0$。

在极端饱和区间，哪怕 latent 里有一点微小的 logit 扰动 $\delta$，经过 $\exp$ 放大后，梯度方差会直接发散：
$$ \|\nabla \mathcal{L}\| \to \infty $$
在 bf16 精度下，这就是满屏的 NaN。公式 (7) 精确描述了这个 conditional distribution shift：
$$ \underbrace{p(\mathbf{x}_i \mid \mathbf{x}_{1:N}, c)}_{\text{Bidirectional (pretrained)}} \longrightarrow \underbrace{p(\mathbf{x}_i \mid \mathbf{x}_{1:i}, c)}_{\text{Causal (target)}} $$
视频因为 token 多，还能勉强扛住这种视野截断；音频因为 token 极度稀疏，直接暴毙。

---

### 3. 物理时间的救赎：1秒 Macro-block 对齐

为了解决 25:3 这个丑陋的非整数频率比，作者发现了一个极度优雅的数学拼图：**按照物理时间 1 秒钟来切块**。

1 秒钟恰好包含 $\Delta N_v = 3$ 个 video latents 和 $\Delta N_a = 25$ 个 audio latents，没有任何分数余数。更妙的是，这完美契合了 causal VAE 的 stride 特性。对于 VAE 的时间压缩，第一帧必须独占 receptive field（stride=1），后续帧才能用大 stride 压缩。所以全序列长度 $N$ 的公式推导为：
$$ N_v = 1 + K \cdot f_v, \quad N_a = 1 + K \cdot f_a $$
这里的常数 $1$，对应 $t \approx 0$ 时刻的 anchor latent $\mathbf{V}_0, \mathbf{A}_0$。这个常数本质上是 VAE 因果卷积的产物。作者把 $\mathbf{V}_0, \mathbf{A}_0$ 显式剥离出来，组成一个 Global Prefix block $\mathcal{B}_0$。在这个 block 内部，attention 是无条件 bidirectional 的，并且对所有未来 token 永远 visible。这就如同 LLM 里的 system prompt，为长序列多模态生成提供了一个永久的跨模态语义锚点。

由此推导出的 four-way asymmetric causal mask 非常 clean（公式 4-5）：
$$ \mathbf{M}_{q,k}^{VV} = \mathbb{I}(\tau_v(k) \leq \tau_v(q)), \quad \mathbf{M}_{q,k}^{AA} = \mathbb{I}(\tau_a(k) \leq \tau_a(q)) $$
只要 token 属于同一个物理时间 $\tau$，或者过去的物理时间，mask 就是 1。这让两个模态在物理时间轴上同步向前推进。

---

### 4. 最神来之笔：Audio Sink Token + Identity RoPE

虽然 block 对齐解决了物理时间同步，但 audio 的 Softmax 坍塌还在。作者的解法极具直觉：**既然 audio token 太稀疏，我们就凭空给它塞几个“垃圾桶” token**。

借鉴 StreamingLLM 发现的 attention sink 现象，作者在 audio 序列最前面 prepend $S$ 个 learnable Sink Tokens，并永久驻留在 Global Prefix $\mathcal{B}_0$ 里。

数学上看，这强行把 early audio token 的 Softmax 分母从 $i$ 扩大到了 $i + S$。在实验中，$S \geq 4$ 就能稳住收敛，$S=16$ 时 loss 最低（0.081）。当 $S \leq 2$ 时，依然会 Softmax collapse 导致 NaN。

但这带来一个新问题：这些 sink token 是抽象的，没有真正的物理时间位置。如果用标准 RoPE 给它们注入位置信息，会干扰它们的 anchor 语义。于是作者强制给它们施加 Identity RoPE 约束（公式 8）：
$$ \cos(\theta_{\text{sink}}) = \mathbf{1}, \quad \sin(\theta_{\text{sink}}) = \mathbf{0} $$
在 RoPE 的标准变换 $x' = x \cos\theta + R(x) \sin\theta$ 中，代入上述条件，直接退化成 $x' = x$。这使得 sink token 完全免疫任何位置旋转干扰，成为纯粹的 position-agnostic semantic anchor。Ablation 证明，如果换成 incremental RoPE，虽然能收敛，但 loss 会飙升到 0.402，且输出充满噪声。

---

### 5. 让模型提前适应自己犯的错：Joint Self-Forcing

进入 Stage III，模型已经具备了 causal 生成能力，但 autoregressive 模型天生有 exposure bias：训练时看的是干净的 ground-truth history，推理时看的是自己生成的 noisy history，长 rollout 下误差迅速累积。在多模态里，这更致命——video 误差会污染 audio，audio 误差反过来污染 video，导致 cross-modal desynchronization。

解法是 Joint Self-Forcing（公式 9）：
$$ \mathcal{L}_{\text{SF}} = \sum_{k=1}^{K} \mathbb{E}_{\hat{\mathcal{B}}_{<k}}\left[\nabla_\theta \text{KL}\big(G_\theta(\mathbf{z}_k \mid \text{KV}_{<k}, c) \parallel R_\phi(\mathbf{z}_k \mid c)\big)\right] $$
变量含义：
- $G_\theta$：causal student generator
- $R_\phi$：frozen bidirectional teacher
- $\hat{\mathcal{B}}_{<k}$：student 自己 AR rollout 出来的前 $k-1$ 个 block（带 noise）
- $\text{KV}_{<k}$：把这些 noisy 输出 encode 成无噪声 KV 放进 rolling cache
- $\mathbf{z}_k$：block $k$ 的采样噪声

训练时，模型被迫 unroll 自己的 KV cache。这意味着模型在训练阶段就提前见识到了自己推理时会犯的错，并且学会在 video 和 audio 互相 drift 的情况下动态修正彼此。这就是为什么 OmniForcing 在 JavisBench 上的 DeSync 指标达到 0.392，几乎等同于 teacher 的 0.384，把所有 cascaded pipeline（通常 >0.85）远远甩在身后。

---

### 6. 推理架构：单卡 25 FPS 的秘密

LTX-2 的 dual-stream 架构有个特点：14B 的 video branch 和 5B 的 audio branch 在每个 transformer layer 内部，各自拥有独立的 FFN sub-layer，只有在 cross-modal attention（A2V 和 V2A）时才交换数据。

这意味着 video self-attention 和 audio self-attention 之间零数据依赖。作者利用这一点设计了 Modality-Independent Rolling KV-Cache。每次生成新 block，只需要保留 Global Prefix + Sink Tokens + 最近 $L$ 个 block 的 KV。单步 context 复杂度从 $O(N^2)$ 降到 $O(L)$。

由于 FFN 互不依赖，单卡上可以让 video 和 audio stream 并行算 self-attention，仅在 cross-attention 边界短暂同步。多卡扩展时，还可以做 asymmetric tensor parallelism，给沉重的 video stream 分配更多算力。

---

### 7. 实验数据的反直觉之处

看 Table 2 的 VBench 数据，你会发现 student 模型在各项 per-frame 质量指标上**全面超越了 teacher**：
- Aesthetic Quality: 0.595 vs 0.569 (+0.026)
- Imaging Quality: 0.594 vs 0.574 (+0.020)

这在 DMD 蒸馏里是个 well-known 现象。DMD 的 KL 散度 matching 倾向于 mode-seeking 行为，把概率 mass 集中在最 likely 的模式上，导致 per-sample 指标看起来更 sharp、更清晰。代价是生成多样性下降，不过在 JavisBench 这种 per-prompt 评测里看不出来。

Table 1 的核心 trade-off 也很清晰：FVD 从 125.4 微涨到 137.2（质量损失极小），但 Runtime 从 197s 暴跌到 5.7s，TTFC 从 197s 降到 0.7s，实现了 ~35× 加速和真正的 25 FPS streaming。

---

### 总结

OmniForcing 的底层直觉可以浓缩为：
1. **按物理时间切块**，用 1 秒 macro-block 绕开 25:3 的非整数帧率不对称。
2. **凭空造几个 Sink Token**，给稀疏的 audio stream 提供稳定的 Softmax 分母，并用 Identity RoPE 屏蔽位置干扰，根治梯度爆炸。
3. **训练时自我纠错**，让 video 和 audio 提前适应彼此的 drift，跨模态误差在长 rollout 下被锁死。
4. **模态解耦的 KV Cache**，让双流网络在单卡上并行奔跑，实现物理时间与计算时间的完美同步。

这种工程极其精准的工作，把 foundation model 的最后一公里落地问题解决得非常漂亮。

**参考链接：**
- OmniForcing Project Page: [https://omniforcing.com](https://omniforcing.com)
- LTX-2 Teacher Model: [https://arxiv.org/abs/2601.03233](https://arxiv.org/abs/2601.03233)
- Self-Forcing (NeurIPS 2025): [https://arxiv.org/abs/2506.07551](https://arxiv.org/abs/2506.07551)
- CausVid (CVPR 2025): [https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.pdf)
- Attention Sink (StreamingLLM): [https://arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453)
- RoPE (RoFormer): [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
- DMD2 Distillation: [https://arxiv.org/abs/2312.03258](https://arxiv.org/abs/2312.03258)

---

# OmniForcing 深度解读：把 Bidirectional 多模态 Diffusion 蒸馏成实时 Streaming 引擎

Karpathy 你好，这篇 paper 我读了几遍，整体思路非常 elegant，核心 trick 不多但每个都精确戳到 dual-stream causal 蒸馏的痛点。下面我把整个 pipeline、公式、设计直觉、实验数据全部摊开讲。

---

## 1. 大背景：这个工作站在哪条 timeline 上

要理解 OmniForcing，必须把它放在 2025–2026 这条 "video diffusion 走 causal / streaming" 的演进线上看：

1. **CausVid (CVPR 2025)** [Yin et al. 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.pdf)：第一个把 bidirectional video DiT 蒸馏成 causal AR，用 asymmetric DMD，达到 ~9.4 FPS。但只做 video。
2. **Self-Forcing (NeurIPS 2025)** [Huang et al. 2025](https://arxiv.org/abs/2506.07551)：发现 AR video diffusion 的 exposure bias 问题，训练时让模型 unroll 自己的 KV cache，解决长 rollout 误差累积。还是 video-only。
3. **Causal-Forcing / Rolling-Forcing (ICLR 2026)** [Zhu et al. 2026](https://arxiv.org/abs/2602.02214)：把 forcing 思想推广到更严格的 causal 一致性和 minute-level 长上下文。
4. **LTX-2 (2026)** [HaCohen et al. 2026](https://arxiv.org/abs/2601.03233)：dual-stream (14B video + 5B audio) bidirectional foundation model，质量 SOTA，但 TTFC 197s 完全没法 stream。
5. **OmniForcing（本文）**：第一次把上面这条 video-only 的 causal forcing 技术栈，搬到 dual-stream audio-visual 架构上，解决 modality asymmetry 带来的 training instability，最终 ~25 FPS 单卡 streaming。

直觉上，OmniForcing 的定位很清晰：它是 "Self-Forcing 的多模态延伸"，但作者发现直接 naive 移植会炸，所以核心贡献其实是 **怎么让 dual-stream causal 蒸馏稳定收敛**。

---

## 2. 问题形式化：为什么 block-level 而不是 frame-level

目标 distribution 写成：

$$p(\mathbf{V}, \mathbf{A} \mid c) = p(\mathcal{B}_0 \mid c) \prod_{k=1}^{K} p(\mathcal{B}_k \mid \mathcal{B}_{<k}, c) \tag{1}$$

变量含义：
- $\mathbf{V}, \mathbf{A}$：video / audio 全序列（在各自 VAE latent space 里）
- $c$：text prompt
- $\mathcal{B}_k$：第 $k$ 个 1 秒物理时间窗口的 macro-block，**同时包含 video chunk 和 audio chunk**
- $K$：总生成秒数
- $\mathcal{B}_0$：Global Prefix，特殊处理

**这里直觉很关键**：选 1 秒作为基本 block 单位，是因为 $f_v = 3$ FPS（video VAE）和 $f_a = 25$ FPS（audio VAE）的最小公倍数性质。1 秒正好是 $3$ video latent frames + $25$ audio latent frames，**没有分数余数**。

如果硬要 frame-by-frame causal（例如 1 video frame → 25/3 audio frame），整个 alignment 就碎掉了。Block-level 让两个模态以"一秒钟一秒钟"的节奏同步往前推，这是后面所有 attention mask 设计的物理基础。

---

## 3. Asymmetric Block-Causal Alignment：VAE stride 的数学拼图

这一节是全 paper 最美的部分，作者从 VAE 的 causal convolution stride 直接反推出 token 长度公式。

### 3.1 VAE stride 的不对称设计

标准 causal VAE（LTX-2 用的 video VAE 和 audio VAE 都是这个套路）在时间维度上对**第一帧 stride = 1**，后续帧 stride = 8（video）/ 4（audio）。这导致：

$$N_v = 1 + K \cdot f_v, \quad N_a = 1 + K \cdot f_a \tag{2}$$

变量含义：
- $N_v$：video 总 latent frame 数
- $N_a$：audio 总 latent frame 数
- $K$：生成秒数
- 常数 $1$：对应 $t \approx 0$ 时刻的 anchor latent $\mathbf{V}_0, \mathbf{A}_0$
- $f_v = 3, f_a = 25$：每秒 latent frame 速率

**直觉**：那个常数 1 不是 bug，是 VAE 因果卷积的产物。第一帧必须自己独占一个 receptive field（因为前面没有 history），后续帧才能 stride 压缩。这意味着 $\mathbf{V}_0, \mathbf{A}_0$ 在物理时间上被钉死在原点，**没法塞进任何 1 秒标准 block**。

### 3.2 Global Prefix 的设计哲学

作者把这个"塞不进去"的 VAE 副作用变成 design advantage：把 $\mathbf{V}_0, \mathbf{A}_0$ 显式组成 $\mathcal{B}_0$（Global Prefix），其中 attention **无条件 bidirectional**，并且对所有未来 token 都可见。

这玩意儿类比 LLM 里的 system prompt——永久驻留在 KV cache 里，永远 visible，作为跨模态的语义锚点。

具体 token 数：
- $\mathcal{B}_0$ 含 $1 \times H_v W_v$ 个 video token（1 frame patchified）
- $\mathcal{B}_0$ 含 $1$ 个 audio token
- 后续 video block $\mathcal{B}_k^v$ 含 $3 \times H_v W_v = 3 \times 384 = 1152$ tokens（论文配置）
- 后续 audio block $\mathcal{B}_k^a$ 含 $25$ tokens

注意这个不对称：**一个 video block 1152 tokens vs 一个 audio block 25 tokens，差距 46 倍**。这就是后面 Softmax collapse 的根源。

### 3.3 Block index 的数学映射

公式 (3)：

$$\tau_v(q) = 1 + \left\lfloor \frac{q - H_v W_v}{3 \times H_v W_v} \right\rfloor, \quad \tau_a(q) = 1 + \left\lfloor \frac{q - 1}{25} \right\rfloor$$

变量：
- $q$：token 在序列中的绝对 index
- $\tau_v(q), \tau_a(q)$：该 token 所属的物理 block 编号
- $H_v W_v$：video VAE 压缩后的空间尺寸 × patch size
- 减去 $H_v W_v$（video）或 $1$（audio）：跳过 Global Prefix 的 token 数
- $\lfloor \cdot \rfloor$：floor，得到 block 编号
- 加 $1$：因为 $\mathcal{B}_0$ 已经占了 index 0

任何 $q < H_v W_v$ (video) 或 $q < 1$ (audio) 的 token 自动归到 $\mathcal{B}_0$。

### 3.4 Four-way asymmetric causal mask

公式 (4)–(5)：

$$\mathbf{M}_{q,k}^{VV} = \mathbb{I}(\tau_v(k) \leq \tau_v(q)), \quad \mathbf{M}_{q,k}^{AA} = \mathbb{I}(\tau_a(k) \leq \tau_a(q)) \tag{4}$$

$$\mathbf{M}_{q,k}^{VA} = \mathbb{I}(\tau_a(k) \leq \tau_v(q)), \quad \mathbf{M}_{q,k}^{AV} = \mathbb{I}(\tau_v(k) \leq \tau_a(q)) \tag{5}$$

四种 attention 通路：
- $VV$：video query → video key
- $AA$：audio query → audio key
- $VA$：video query → audio key（V 看 A）
- $AV$：audio query → video key（A 看 V）

核心规则：**block 内 bidirectional，block 间严格 causal**。注意跨模态的 block index 用的是物理时间 block（即 1 秒对 1 秒），所以 $\tau_v$ 和 $\tau_a$ 在物理时间上是 1:1 对应的，跨模态 causal 就自然对齐了。

Global Prefix 的 token $\tau = 0$，所以对所有 $q \geq 0$ 都满足 $\tau(k) = 0 \leq \tau(q)$，**数学上保证永远 visible**。这是 elegant 的地方——不需要在 mask 里特判，公式自动 cover。

---

## 4. 三阶段 Distillation Pipeline：每个 stage 在干什么

这个 pipeline 我画一下结构：

```
Pretrained LTX-2 (bidirectional, ~50 step)
        ↓
Stage I: Bidirectional DMD (2000 steps)
        → Bidirectional student, few-step (~4 step)
        ↓
Stage II: Causal ODE Regression (3000 steps)
        → 装 causal mask，regress Stage I 的 ODE trajectory
        → 这里最危险，Audio Sink Token 介入
        ↓
Stage III: Joint Self-Forcing DMD (2000 steps)
        → 训练时 autoregressive unroll 自己的 KV cache
        → 跨模态同步纠错
        ↓
OmniForcing (streaming, ~25 FPS)
```

为什么这么分？因为**few-step denoising 能力** 和 **causal generation 能力** 是两个正交的 skill，同时学就会互相干扰，分阶段注入更稳定。

### Stage I: Bidirectional DMD

Loss 是加权的 video + audio score matching：

$$\mathcal{L}_{\text{Bi-DMD}} = \lambda_v \mathcal{L}_{\text{DMD}}^v + \lambda_a \mathcal{L}_{\text{DMD}}^a$$

DMD（Distribution Matching Distillation）[Yin et al. CVPR 2024](https://arxiv.org/abs/2312.03258) 的核心是 minimize student 和 teacher 之间的 KL 散度（通过 score function 近似），把 50 步的 teacher 压成 4 步 student。这一阶段不动 attention mask，保持 bidirectional，只是把步数压下来，为后面提供一条 high-quality、容易 regress 的 teacher trajectory。

### Stage II: Causal ODE Regression（核心战场）

公式 (6)：

$$\mathcal{L}_{\text{ODE}} = \mathbb{E}_{t, \mathbf{x}_t}\left[\lambda_v \|v_\theta^v(\mathbf{x}_t, c) - v_\phi^v(\mathbf{x}_t, c)\|_2^2 + \lambda_a \|v_\theta^a(\mathbf{x}_t, c) - v_\phi^a(\mathbf{x}_t, c)\|_2^2\right]$$

变量：
- $\mathbf{x}_t = [\mathbf{V}_t, \mathbf{A}_t]$：joint noisy latent at flow-matching time $t$
- $v_\theta^v, v_\theta^a$：student 在 video / audio 上的 velocity prediction（带 causal mask）
- $v_\phi^v, v_\phi^a$：Stage I teacher 的 velocity prediction（bidirectional，no mask）
- $\lambda_v, \lambda_a$：modality 权重

直觉上，这一步是把 student 的 causal prediction 直接拟合 teacher 的 bidirectional prediction。同样的输入 noisy latent $\mathbf{x}_t$，teacher 看到全局，student 只看 causal history，但 student 必须输出和 teacher 一致的 velocity。这是个 regression task，避免了完整生成的开销。

### 条件分布漂移与梯度爆炸

公式 (7) 是核心 insight：

$$\underbrace{p(\mathbf{x}_i \mid \mathbf{x}_{1:N}, c)}_{\text{Bidirectional (pretrained)}} \longrightarrow \underbrace{p(\mathbf{x}_i \mid \mathbf{x}_{1:i}, c)}_{\text{Causal (target)}} \tag{7}$$

从 globally-informed posterior 变成 truncated causal one，这是 conditional distribution shift。问题在于**这个 shift 跨模态不对称**：

- Video block：每个 block 1152 tokens，causal 历史快速增长，local context 还算丰富
- Audio block：每个 block 只有 25 tokens，新 block 的第一个 token 只能看到 Global Prefix (1 token) + Sink Tokens (S 个) + 之前 audio blocks

数学上看 Softmax collapse：假设 audio 在 block $k$ 的第一个 token，可见 key 数 $D \approx 1 + S + 25(k-1)$。当 $k=1, S=0$ 时，$D = 1$，Softmax 分母只有一个 logit，**信息熵 = 0**。

在这种 saturated regime 下，logit 微小扰动 $\delta$ 会被 exponential 放大 $e^\delta$ 倍，梯度方差爆炸 $\|\nabla \mathcal{L}\| \to \infty$，bf16 直接 NaN。

---

## 5. Audio Sink Token + Identity RoPE：最关键的创新

这是全 paper 最 trick 也是最物理直觉的部分。

### 5.1 Sink Token 的物理意义

借鉴 [Xiao et al. StreamingLLM ICLR 2024](https://arxiv.org/abs/2309.17453) 的 attention sink 现象，以及 [Darcet et al. ViT Registers ICLR 2024](https://arxiv.org/abs/2309.16588) 在视觉模型的发现：训练好的 attention 总会自然把"非语义信息"扔到前几个 token 上，让它们做"全局垃圾桶"。

OmniForcing 主动在 audio 序列前 prepend $S$ 个 learnable sink tokens，**永久驻留在 $\mathcal{B}_0$**。

数学效果：把 early audio token 的 Softmax 分母从 $i$ 强行扩到 $i + S$。当 $S=16$，分母最小也是 17，attention entropy 不再趋零，logit 扰动有 buffer 吸收，梯度 storm 平息。

物理上这些 token 是"软全局记忆 buffer"——不携带具体内容，但提供稳定的 attention 分母，让模型在 causal transition 期间有个 anchor。

### 5.2 Identity RoPE 的必要性

标准 RoPE [Su et al. 2024](https://arxiv.org/abs/2104.09864) 对每个 token 赋一个 rotary phase $\theta$，但 sink token 是"抽象"的，没有真正的物理时间位置。如果用 incremental position 给它们，会注入虚假时间偏置，干扰它们的"anchor"语义。

公式 (8)：

$$\cos(\theta_{\text{sink}}) = \mathbf{1}, \quad \sin(\theta_{\text{sink}}) = \mathbf{0} \tag{8}$$

RoPE 的标准形式是 $x' = x \cos\theta + R(x) \sin\theta$（$R$ 是 reversal operator）。当 $\cos\theta = 1, \sin\theta = 0$ 时：

$$\text{RoPE}(\mathbf{x}) = \mathbf{x} \cdot 1 + R(\mathbf{x}) \cdot 0 = \mathbf{x}$$

退化成 identity mapping，sink token 完全免疫任何位置旋转干扰，成为 **position-agnostic semantic anchor**。

### 5.3 Ablation 验证

Table 3 的 ablation 非常 clean：

| Config | Convergence | $\|\nabla\|_{\max}$ | Loss |
|---|---|---|---|
| $S=24$ + Id. RoPE | Stable | 9.15 | 0.110 |
| $S=16$ + Id. RoPE | **Stable** | 9.23 | **0.081** |
| $S=8$ + Id. RoPE | Stable | 21.95 | 0.129 |
| $S=4$ + Id. RoPE | Stable | 49.71 | 0.141 |
| $S=2$ + Id. RoPE | NaN | ∞ | — |
| $S=1$ + Id. RoPE | NaN | ∞ | — |
| $S=16$ + Incr. RoPE | Stable | 11.21 | 0.402 (noisy) |
| QK-Norm | Stable | 4.45 | 0.232 (damped) |
| Tanh-Gated Attn | Plateau | 10.61 | 1.258 (destroyed) |
| No Stabilizer | NaN | ∞ | — |

**几个直觉**：
1. $S \geq 4$ 是稳定收敛的阈值，$S = 16$ 是 loss 最低的 sweet spot。
2. $S = 16$ 用 incremental RoPE 还是能收敛但 loss 5× worse (0.402 vs 0.081)，说明 sink 的 position 信息绝对不能注入任何"虚假时间"，Identity 是必要的。
3. QK-Norm [Dehghani et al. ICML 2023](https://arxiv.org/abs/2302.05442) 也能稳定（因为它强行 normalize QK，间接避免 logit 爆炸），但它过度 damping attention 对比度，loss 仍偏高。
4. Tanh-Gated Attention（Flamingo 风格）虽然不 NaN，但 tanh 饱和把梯度直接压死，loss 卡在 1.258，输出全是 block artifacts，attention 模式被"功能上毁掉"。

**结论**：Audio Sink Token + Identity RoPE 是这个不对称架构下"既能稳定又能保质量"的唯一解。

---

## 6. Joint Self-Forcing：解决跨模态 exposure bias

### 6.1 Exposure bias 的多模态放大

AR diffusion 的经典问题 [Huang et al. NeurIPS 2025](https://arxiv.org/abs/2506.07551)：训练时 condition 在 ground-truth history 上，推理时 condition 在自己 noisy 的输出上，长 rollout 误差累积。

在 dual-stream 里这个问题被放大——video 误差会污染 audio prediction（V→A cross attention），反之亦然，跨模态 desynchronization 比单模态误差累积更糟。

### 6.2 Joint Self-Forcing loss

公式 (9)：

$$\mathcal{L}_{\text{SF}} = \sum_{k=1}^{K} \mathbb{E}_{\hat{\mathcal{B}}_{<k}}\left[\nabla_\theta \text{KL}\big(G_\theta(\mathbf{z}_k \mid \text{KV}_{<k}, c) \parallel R_\phi(\mathbf{z}_k \mid c)\big)\right] \tag{9}$$

变量：
- $G_\theta$：causal student generator
- $R_\phi$：frozen bidirectional teacher（Stage I 的产物）
- $\hat{\mathcal{B}}_{<k}$：student 自己 AR rollout 出来的前 $k-1$ 个 block（**不是 ground-truth**）
- $\text{KV}_{<k}$：$\hat{\mathcal{B}}_{<k}$ 的 KV embedding，无噪声，append 到 rolling cache
- $\mathbf{z}_k$：block $k$ 的采样噪声

训练流程：
1. 学生生成 $\hat{\mathcal{B}}_1$（condition on $\mathcal{B}_0$）
2. 把 $\hat{\mathcal{B}}_1$ encode 成 KV，加入 cache
3. 学生生成 $\hat{\mathcal{B}}_2$（condition on $\mathcal{B}_0 + \hat{\mathcal{B}}_1$）
4. ...直到 $\hat{\mathcal{B}}_K$
5. 每个 $\hat{\mathcal{B}}_k$ 和 teacher 的 $R_\phi(\mathbf{z}_k \mid c)$ 算 KL，gradient 回传 $\theta$

注意 teacher 看的是无 noise 的全局 condition，但用的是**同一个 $\mathbf{z}_k$**，所以 student 和 teacher 对应同一个去噪目标，KL 是 well-defined 的。

### 6.3 跨模态 self-correction

这里和原版 Self-Forcing 的关键区别：**video 和 audio stream 同时 unroll，互相看到对方的 drifted prediction**。这就迫使模型在训练时学会：
- "audio 即使看到 slightly-off 的 video，也要生成合理 audio"
- "video 即使看到 slightly-off 的 audio，也要生成合理 video"

这种 paired exposure 是 OmniForcing 能在长 rollout 下保持 cross-modal synchrony 的根本原因。Table 1 的 DeSync 指标验证：OmniForcing 0.392 vs teacher 0.384（几乎无损），而 JavisDiT++ 0.832（差距巨大）。

---

## 7. Modality-Independent Rolling KV-Cache：怎么实现 25 FPS

### 7.1 架构层面的 decoupling

LTX-2 的 dual-stream 在每个 transformer layer 内部：
- video 自有 FFN（14B 主体）
- audio 自有 FFN（5B 主体）
- 只在 cross-modal attention（A→V 和 V→A）处同步

这意味着 **video self-attention 和 audio self-attention 之间零数据依赖**。

### 7.2 Rolling KV-Cache

每次生成 $\mathcal{B}_k$，只需要：
- Global Prefix 的 KV（永久驻留）
- Sink Token 的 KV（永久驻留）
- 最近 $L$ 个 block 的 KV（rolling window）

Per-step context 复杂度从 $O(N^2)$（$N$ 是全序列长度）降到 $O(L)$（$L$ 是 cache window 里的 latent frame 数）。

### 7.3 Asymmetric Parallel Inference

由于两个 modality FFN 互相独立，可以：
- 单卡上 video 和 audio stream 并行算 self-attention
- 多卡上用 asymmetric tensor parallelism：给 video stream 分更多卡（14B vs 5B），平衡 compute

这就是为什么单 H100 就能跑 25 FPS。

### 7.4 TTFC 0.7s 怎么来的

TTFC = 生成 + decode $\mathcal{B}_0$ (Global Prefix) + $\mathcal{B}_1$ (第一个 streaming block) 的时间。一旦吐出，后续 block 的 decode 和生成可以 pipeline 并行，streaming 不中断。

对比 LTX-2 teacher 的 197s TTFC（必须等整个 5s 视频全 denoise 完才能 decode），这是 **280× 的 TTFC reduction**，runtime 是 5.7s vs 197s（~35×）。

---

## 8. 实验结果深度分析

### 8.1 JavisBench 主表

| Model | FVD↓ | FAD↓ | AV-IB↑ | DeSync↓ | Runtime↓ |
|---|---|---|---|---|---|
| LTX-2 (teacher) | **125.4** | **4.6** | 0.318 | **0.384** | 197s |
| OmniForcing | 137.2 | 5.7 | 0.269 | 0.392 | **5.7s** |
| JavisDiT++ | 141.5 | 5.5 | 0.198 | 0.832 | 10s |
| MMAudio (V2A) | — | 6.1 | 0.198 | 0.849 | 15s |

**几个关键直觉**：
1. **FVD/FAD 几乎无损**：137.2 vs 125.4（FVD 差 9.4%），5.7 vs 4.6（FAD 差 23.8%）。考虑到 35× 加速，这个 trade-off 非常合理。
2. **AV-IB 大幅领先所有非 teacher baseline**：0.269 vs JavisDiT++ 的 0.198，领先 36%。这是 joint streaming 训练的直接收益——模型在训练时就学会跨模态对齐。
3. **DeSync 几乎无损**：0.392 vs 0.384。这是 joint self-forcing 的功劳。所有 cascaded pipeline (V2A/A2V) 的 DeSync 都 >0.8，因为他们根本没法做 fine-grained 同步。
4. **CLIP score 反超 teacher**：0.322 vs 0.318。DMD 蒸馏常见现象，student 在 per-sample metric 上经常超过 teacher（[Yin et al. 2024](https://arxiv.org/abs/2312.03258), [Huang et al. 2025](https://arxiv.org/abs/2506.07551) 都有类似发现）。

### 8.2 VBench 单帧质量

Table 2：

| Metric | LTX-2 | OmniForcing | Δ |
|---|---|---|---|
| Aesthetic Quality | 0.569 | **0.595** | +0.026 |
| Imaging Quality | 0.574 | **0.594** | +0.020 |
| Subject Consistency | 0.945 | **0.955** | +0.010 |
| Motion Smoothness | 0.993 | **0.995** | +0.002 |
| Temporal Flickering | 0.988 | **0.989** | +0.001 |

**Student 全面超越 Teacher**，看起来反直觉，但其实是 DMD 的 well-known 效应：DMD 的 KL 散度 matching 会 prioritize mode-seeking 行为，倾向于把 mass 集中在最 likely 模式上，所以在 per-sample 指标上表现更 sharp。代价是 diversity 下降，但 JavisBench 这种 per-prompt 评测看不出来。

---

## 9. 整体直觉总结

如果让我用一句话总结这篇 paper：

**"OmniForcing 把 video-only 的 causal forcing 技术栈扩展到 dual-stream audio-visual，关键 insight 是 audio token 的极端稀疏会在 causal shift 时引发 Softmax collapse，解决方案是引入 attention sink + identity RoPE 把 attention 分母强行撑起来。"**

更深一层的 intuition：

1. **VAE stride 决定了 token distribution 的不对称**，这种不对称在 bidirectional 时被全局 attention 掩盖，一到 causal 就暴露。
2. **Sink token 是 attention 的"摆渡人"**，它本身不携带语义，但提供一个稳定的 normalization denominator，让 early tokens 有地方"借"到信息密度。
3. **Block-level 是物理时间的自然单位**，把 25:3 这个 ugly ratio 用 1 秒 macro-block 整除掉，是整个 mask 设计的奠基。
4. **Joint Self-Forcing 是跨模态 exposure bias 的疫苗**，训练时让模型提前遇到自己的 drift，推理时才能 robust。
5. **Modality-Independent KV Cache 把 dual-stream 变成 quasi-single-stream**，硬件友好度极高。

---

## 10. 我会问作者的几个问题

如果有机会讨论，我会问：

1. **Sink Token 的 $S=16$ 是怎么定的**？有没有更 principled 的选择方式，比如基于 audio block 大小 $|B^a|=25$ 的某个比例？还是纯 grid search？
2. **Global Prefix 的 $\mathbf{V}_0, \mathbf{A}_0$ 在长 rollout 下是否会成为 attention 的"信息瓶颈"**？因为它们对所有未来 token 都 visible，但容量固定，长序列下会不会变成弱 link？
3. **Rolling KV-Cache 的 window $L$ 多大**？太小会丢长程依赖，太大会爆显存。Ablation 里没看到这个 sweep。
4. **为什么不做 Stage IV 的对抗蒸馏（ADD）**？目前 Stage III 是 DMD，按 [Sauer et al. ECCV 2024](https://arxiv.org/abs/2311.17042) 的经验，ADD 在 per-step 质量上经常超过 DMD，是不是因为 dual-stream 下 discriminator 设计困难？
5. **Audio Sink Token 能不能 video 也加**？Video block 1152 tokens 看起来不需要，但极端长 sequence 时 video 早期 block 会不会也出现类似的 attention entropy 问题？

---

## 参考链接

- 论文 Project Page: [https://omniforcing.com](https://omniforcing.com)
- LTX-2 (teacher): [https://arxiv.org/abs/2601.03233](https://arxiv.org/abs/2601.03233)
- CausVid: [https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.pdf)
- Self-Forcing: [https://arxiv.org/abs/2506.07551](https://arxiv.org/abs/2506.07551)
- StreamingLLM (Attention Sink): [https://arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453)
- ViT Registers: [https://arxiv.org/abs/2309.16588](https://arxiv.org/abs/2309.16588)
- RoPE (Rotary Position Embedding): [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
- DMD2: [https://arxiv.org/abs/2312.03258](https://arxiv.org/abs/2312.03258)
- Causal-Forcing: [https://arxiv.org/abs/2602.02214](https://arxiv.org/abs/2602.02214)
- VBench: [https://arxiv.org/abs/2311.13535](https://arxiv.org/abs/2311.13535)
- JavisBench: [https://arxiv.org/abs/2407.17426](https://arxiv.org/abs/2407.17426)
- QK-Norm: [https://arxiv.org/abs/2302.05442](https://arxiv.org/abs/2302.05442)
- Flamingo (Tanh-Gated Attn): [https://arxiv.org/abs/2204.14198](https://arxiv.org/abs/2204.14198)
- Adversarial Diffusion Distillation: [https://arxiv.org/abs/2311.17042](https://arxiv.org/abs/2311.17042)

---

整体看，OmniForcing 是个非常 "engineering-aware" 的工作——它没发明 fundamentally new 的算法，但把现有技术（DMD / causal forcing / sink token / RoPE）在 dual-stream 这个 asymmetric 设定下做了一次精确的 surgical adaption，每个 trick 都对应一个具体 failure mode。这种 paper 在 2026 年的 video diffusion 社区里非常有代表性：foundation model 已经做得太大太慢，real-time streaming 是落地的最后一公里，而最后一公里的难点全在工程细节里。
