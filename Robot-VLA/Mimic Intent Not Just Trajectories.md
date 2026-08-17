---
source_pdf: Mimic Intent Not Just Trajectories.pdf
paper_sha256: afafc2399ff892312158e203cbb6694bb506aa4aa08efa5e59a473679ff2322f
processed_at: '2026-08-05T18:14:17-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 MINT

## 一句话

把一段 robot action 想象成一段音乐：bass 是节奏骨架（要干嘛），treble 是装饰细节（怎么动）。MINT 用一个老信号处理 trick（DCT）把这俩硬拆开，让一个 token 只装 intent，剩下的 token 装 execution。policy 先决定 intent 再决定 execution，自然就是先规划后执行。

---

## 用类比讲

你去 KTV 唱一首歌，别人听到的是什么？bass 和鼓点先告诉你这是哪首歌、在哪个段落，treble 和人声细节才是你的唱法。如果你只听 treble，根本听不出是哪首歌；只听 bass，能猜出歌但听不出是你唱的。

Robot action 也是这样。一段 "把香蕉放盘子" 的 trajectory：
- **low frequency** = 整段的大趋势：手伸过去 → 抓 → 抬 → 移到盘子 → 放。这是 intent，是 task 本质。
- **high frequency** = 手指微调、关节抖动、对 camera angle 的反应式修正。这是 execution，跟具体环境绑死。

JPEG 早就这么干了 — JPEG 砍高频保低频，因为人眼对低频敏感。MINT 在 action 上反其道：低频和高频都保，但强制塞进不同的 token，让模型 structure 地知道哪个是骨架、哪个是细节。

参考：[JPEG 和 DCT 的原理](https://en.wikipedia.org/wiki/JPEG#Discrete_cosine_transform)

---

## 现有方法为什么不行

π0、OpenVLA、π0.5 这些 VLA，做法都是：vision + language → 直接回归一段 continuous action chunk。问题在哪？模型把 intent 和 execution 揉在一起 regress，分不清哪个重要。结果：

- 换 camera angle → 崩，因为 execution 里有些细节是 view-specific 的（手在哪对齐），模型 overfit 了
- 换 layout → 崩，因为 trajectory 的具体形状变了
- 长 horizon → 崩，因为只靠 next-chunk prediction，没有 hierarchical planning

你给它 100 万条 "拿杯子" 的 demo，它学到的可能是：杯子在画面左边 + 手要这样动。它没学到 "拿杯子" 这个 abstract intent。

现有 action tokenizer（FAST、VQ-VAE 系列）也只把 trajectory 当成要压缩的 signal，没有 explicit 约束说 "你要把 intent 抽出来"。压缩和抽象是两回事。

---

## MINT 具体干什么

**Phase 1 — 训 tokenizer（SDAT）**：

1. 拿一段 action chunk（比如 16 步，每步 7 维）
2. 用 encoder 压成 latent
3. 用 multi-scale residual VQ 把 latent 拆成 K 层 token（LIBERO 用 3 层：1, 2, 4 个 token）
4. 每一层累积起来 decode + 做 DCT 得到 frequency spectrum
5. **关键 loss**：每一层都得能 reconstruct 出 ground-truth spectrum，L2

这个 loss 的作用，用 optimizer 的视角想：robot action 大部分能量在低频，L2 在频域里是 add 的，所以 optimizer 会优先让最粗的 token（只有 1 个，capacity 小）去抓低频（贡献 L2 最大），细 token 补高频 residual。这是 implicit 但很强的 prior。

为什么 prior 强？因为 NN 本身就有 low-frequency bias（[Rahaman et al. 2019 spectral bias](https://proceedings.mlr.press/v97/rahaman19a.html)），你给它小 capacity 让它 reconstruct，它自动选低频；你用频域 loss 显式强化，效果更稳。

Ablation 里最 striking 的对比：
- scale-wise 在 **time domain** 做 reconstruction：LIBERO-Long 82.8%（变差）
- scale-wise 在 **frequency domain** 做 reconstruction：LIBERO-Long 93.4%（+10.6%）

为什么 time domain 反而变差？因为 time domain L2 loss 对每个 timestep 平等惩罚，coarse token 会被迫去拟合时间维上的高频细节（手在第 5 步往左抖一下），disentanglement 糊掉。频域 loss 让 coarse token 看到的就是 frequency bin，它知道 "我容量小，先抓低频 bin"。

**Phase 2 — 训 policy**：

拿 SDAT tokenize 出来的 token 当 ground-truth，policy 学 next-scale autoregressive：

```
先 predict s_1 (intent token, 1个)
再 predict s_2 (2个 token, 给定 s_1)
再 predict s_3 (4个 token, 给定 s_1, s_2)
最后 decode 成 continuous trajectory
```

这就是 [VAR (Tian et al. NeurIPS 2024)](https://arxiv.org/abs/2404.02905) 在 image 上的 next-scale prediction，搬到 action。好处：
- 同 scale 内 parallel decode（快）
- 跨 scale autoregressive（保留 hierarchy）
- KV cache 兼容
- 推理结构匹配人类 planning（先想做什么，再想怎么做）

---

## 为什么这个 idea work — 直觉版

三个层面的原因：

**1. Inductive bias 对了**

Robot task 天然是 hierarchical 的。人做事也是先想 intent，再 generate motor command。MINT 把这个 hierarchy explicit 塞进 token space，模型被 forced 走这条路径。你给它 flat token，它可能学到任何奇怪的东西；你给它 structured token，它只能学你要的结构。

**2. Disentanglement 让 generalization 自然产生**

换 camera angle 时，intent（"拿杯子"）不变，execution（手怎么对齐）变。MINT 里 intent token 是 view-invariant abstraction，policy 用 visual observation 重新 generate execution token 就行。LIBERO-Plus camera shift 上 MINT-4B+ 95.6 vs OpenVLA 0.8 — 这不是 10% 提升，是 100 倍。

**3. Intent token 是比 language 更好的 task specification**

Language "pick up the cup" 对应 100 种具体动作，是 abstract 但 ambiguous。Demo trajectory 是 specific 但 overfit。Intent token 是中间产物 — 从一段 demo 抽出来，所以是 execution-aligned（它真的对应一段动作），但又是 abstract（只编码 intent，不编码细节）。

---

## One-shot transfer — 这是我觉得最 elegance 的部分

Protocol 特别简单：
1. 给 1 个 demo trajectory
2. 用 SDAT encoder 抽出它的 $s_1$ token
3. Inference 时把 $s_1$ 固定，policy conditioned on 这个 $s_1$ autoregressively 生成 $s_2, s_3, \ldots$
4. Decode 成 trajectory

完全不用 gradient update。对比实验：

| Method | New Task | New Layout | Ext. Horizon | Avg |
|---|---|---|---|---|
| Replay demo | 0.28 | 0.12 | 0.04 | 0.11 |
| Fine-tune on 1 demo | 0.42 | 0.08 | 0.00 | 0.17 |
| **Intent injection** | **0.90** | **0.68** | **0.72** | **0.77** |

Fine-tune 在 Extended Horizon 上 0.00 — 1 个 demo fine-tune 让 model catastrophic forget 训练时学的东西，退化成 replay。Intent injection 没有梯度更新，不会 forget，还能 generalize。

这本质上是把 intent token 当成 task specification 的另一种 modality，和 language instruction 平行。区别是 language 是 ambiguous 的符号，intent token 是从 demo 抽出来的、execution-aligned 的 representation。这暗示 future VLA 可能应该支持 multi-modal task conditioning：language / goal image / demo trajectory 通过 shared intent space 统一。

类似思路参考 [CMA (Dance et al.)](https://arxiv.org/abs/2106.04950)、[BC-Z (Jang et al.)](https://arxiv.org/abs/2204.06152)，但 MINT 的 intent token 是从频域 decomposition 自然来的，不是 handcrafted task embedding。

---

## 实验里最震撼的几个数字

1. **MINT-30M (from scratch) 打 7B OpenVLA**：LIBERO avg 97.1 vs 76.5。30M 参数、不预训练，靠 tokenization inductive bias，超过 7B pretrained model。这强烈说明 VLA 的瓶颈不是 model size，是 tokenization。

2. **MetaWorld Very Hard 56.0 vs π0 20.0**：3 倍。Very Hard task 需要高精度 alignment + 长时间协调，正好是 intent / execution 分离最受益的场景。

3. **LIBERO-Long 97.8 vs π0.5 92.4**：+5.4。长 horizon composition 收益最大，因为 next-scale AR 让 long-horizon planning 更稳。

4. **LIBERO-Plus Camera shift 95.6 vs OpenVLA 0.8**：120 倍。Intent token 是 view-invariant，camera 变了 intent 不变，policy 重新 generate execution 就行。

5. **One-shot transfer 0.77 vs fine-tune 0.17**：60% 提升。No gradient update，纯 conditioning。

6. **Real-world Stack Blocks 比 π0.5\* 高 29%**：20 个 demo 的 setting 下，MINT-4B 用 intent abstraction generalizes 到 "stacking" 这个 intent，不需要 see 所有 object instance。

---

## 我觉得有意思的几个联想

**1. Tokenization 决定 modeling**

你 (Andrej) 在 nanoGPT talk 里反复强调 character vs BPE vs SentencePiece 直接决定模型学到什么。MINT 在 action domain 的极致体现：用 spectral decomposition tokenize，模型就被 forced 学到 intent hierarchy。这让我觉得 VLA 下一阶段的关键 bottleneck 是 action representation，不是 model scale。

**2. Next-scale 比 next-token 更 general**

Text 上 next-token work，pixel 上要 next-scale（VAR），action 上 MINT 也是 next-scale。暗示 next-token 不是 universal solution，next-structure（scale / cluster / chunk）可能才是 general principle。这和你在 [Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNjjm) 里讲 "data 拼成 1D sequence 才能训" 的妥协有关 — 1D next-token 是为了能用 transformer，但自然界很多结构是 hierarchical 的。

**3. Frequency decomposition 作为 disentanglement prior 是 general 的**

MINT 的 trick 本质是：信号在频域天然分层 + 在频域加 L2 loss + 多层 capacity 不对称 = 自动 disentanglement。这个 idea 可以推到很多 domain：
- Video generation：低频 = scene layout，高频 = motion detail
- Audio：低频 = 旋律，高频 = 音色
- Code：低频 = 算法骨架，高频 = 命名风格
- Language：低频 = topic，高频 = syntactic variation

可能有人已经做了，参考 [Wavelet diffusion](https://arxiv.org/abs/2209.09302)、[Spectral Diffusion](https://arxiv.org/abs/2303.13383)，但 MINT 在 action domain 用 multi-scale VQ + 频域 loss 的组合是新的。

**4. Coarse-to-fine 推理匹配人类 planning**

人类做事也是先想 "我要干什么"，再想 "具体怎么做"。MINT 把这个 hierarchy explicit 化在 token space，让 transformer 自然学到 coarse-to-fine reasoning。这和 hierarchical RL（[HAC](https://arxiv.org/abs/1710.10006)、[HIRO](https://arxiv.org/abs/1805.08296)）的 subgoal / option 思路一致，但 MINT 在 IL setting 下用频域 decomposition implicit 学到 hierarchy，不需要 subgoal annotation。

**5. Wavelet vs DCT**

Wavelet 也是 multi-resolution analysis，但 wavelet 是 spatially localized（每个 coefficient 对应一个 time-frequency tile），DCT 是 globally localized（每个 basis 跨越整段）。对 robot action 来说 DCT 更合适，因为 intent 是全 chunk 的 global property，不是 local window property。Wavelet 更适合 image（局部纹理）。这解释了为什么 JPEG 用 DCT，但 JPEG2000 用 wavelet。

---

## 几个我会担心的点

1. **Intent space 被训练数据限制**：paper Limitation 章节自己提了。如果 demos 都是 "pick and place"，intent codebook 里没有 "pour water" 这种 primitive，one-shot transfer 就没法做。Future direction：从 web-scale video 学更丰富的 intent。参考 [MVP / R3M](https://arxiv.org/abs/2206.11626) 的 video pretraining 思路。

2. **Codebook size 是 handpicked**：LIBERO 512, CALVIN 512, MetaWorld 256, BridgeV2 1024，差异大但没 principled 选择方法。可能可以学 [BIG-VAE](https://arxiv.org/abs/2012.00162) 做 codebook size 自适应。

3. **New Layout 0.68 < New Task 0.90**：layout 变化要求 execution 也变，但 intent injection 只锁 intent，execution 还得 policy generate — 这部分 generalization 还是不够强。

4. **Scale 数量靠 ablation**：[1,2,4] 是 best，但没 principled 自动选择机制。理论上 frequency band 应该自动 partition，类似 [learnable wavelet](https://arxiv.org/abs/2006.05228)。

5. **MINT-4B 在 Spatial 上 97.4 略低于 π0.5 98.8**：不是全面碾压。Spatial 任务对 intent abstraction 需求小，VLM backbone 的 visual grounding 才是瓶颈。这暗示 MINT 在不同 task type 上 benefit 不均匀。

6. **Inference cost**：next-scale AR + KV cache 应该比 next-token 快，但 paper 没给 latency 数据。对 real-time control（robot 一般要 10-30Hz）这个很关键。

---

## Big picture 一句话

MINT 让我相信：**VLA 下一阶段的关键是 action representation design，不是 model scale**。30M from scratch 打 7B pretrained 这件事，和你当年讲 "inductive bias 在 small data regime 比 scale 重要" 完全一致。MINT 的 spectral disentanglement 就是精心设计的 inductive bias — 把 signal processing 的经典 prior 塞进 tokenization，让 transformer 不用从头学 hierarchy。

参考：
- [A Recipe for Training Neural Networks (Karpathy)](https://karpathy.github.io/2019/04/25/recipe/)
- [Intro to LLMs (Karpathy)](https://www.youtube.com/watch?v=zjkBMFhNjjm)
- [Let's build GPT from scratch (Karpathy)](https://www.youtube.com/watch?v=kCc8FmEb1nY)

如果想 build intuition，我建议先看 [VAR 论文](https://arxiv.org/abs/2404.02905) 的 next-scale prediction，再看 [FAST](https://arxiv.org/abs/2501.09747) 的 DCT tokenization，然后 MINT 就是这俩 + frequency domain supervision 的组合。整个 lineage 非常 clean。

---

# MINT: Mimic Intent, Not Just Trajectories — 深度讲解

Andrej, 这篇 paper 我觉得对你 build intuition 来说特别有意思，因为它本质上把你在 nanoGPT / 早期 autoregressive modeling talk 里讲的 "tokenization 决定了你建模什么" 这件事，搬到了 robot action 上 — 而且 motivated 的方式非常 clean：把一条轨迹看成 signal，在频域上做 coarse-to-fine 分解，让最粗的 token 被强制捕获 low-frequency intent。下面我把整个 pipeline 拆开讲。

---

## 1. 核心 motivation：为什么 raw trajectory mimic 不够

VLA 模型（OpenVLA, π0, π0.5, UniVLA 等）目前的范式是：把 vision + language 编码后直接回归一段 continuous action chunk。问题在于：

- action trajectory 在 time domain 是 high-frequency signal — 手指一抖就是高频。
- language instruction 是 semantic-level abstraction — "把香蕉放到盘子上" 是 intent，不是 trajectory。
- 中间缺一层：**behavioral intent**。当前的 VLA 把 intent 和 execution 都揉在一起 regress，导致 policy overfit 到 surface correlation（e.g. 抓取的角度、轨迹的 specific shape），换 layout / camera / object 就崩。

MINT 的关键 insight（也是我觉得最 elegant 的地方）：**trajectory 在频域里是天然分层的**。低频 = 全局形状 = long-horizon intent；高频 = 局部抖动 / reactive correction = execution detail。如果你能在 tokenization 阶段就强制把 low frequency 和 high frequency 分到不同的 token，那么 intent 和 execution 就被 **structural** 地 disentangled，而不是靠 hope 或 post-hoc interpretation。

这一招其实和 JPEG 一脉相承：JPEG 之所以在频域做 quantization，就是因为人眼对低频敏感、对高频不敏感。MINT 把同样的 prior 用在 action 上：低频必须保真（intent 不能丢），高频可以量化得粗一些（execution 可以容错）。

参考：
- [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2501.09747) — 第一个用 DCT tokenize action 的工作，但 MINT 在它基础上加了 multi-scale spectral supervision。
- [Discrete Cosine Transform (Ahmed, Natarajan, Rao, 1974)](https://ieeexplore.ieee.org/document/1672377)

---

## 2. SDAT: Spectrally Disentangled Action Tokenizer

这是 phase 1，独立训练的 tokenizer。本质是 **multi-scale residual VQ-VAE + frequency-domain scale-wise reconstruction loss**。架构直接借鉴 [VAR (Tian et al., NeurIPS 2024)](https://arxiv.org/abs/2404.02905)，但 VAR 在 image 上做 pixel-domain reconstruction；MINT 把 reconstruction loss 搬到 frequency domain，这是它的核心创新。

### 2.1 Encoder 与 spectrum decoder

输入：一段 action chunk $\mathbf{A} \in \mathbb{R}^{H \times D}$
- $H$ = action horizon（LIBERO 用 16，CALVIN 用 32，real-world 用 90 帧）
- $D$ = action dim（一般是 7：3 translation + 3 rotation + 1 gripper，或者 delta end-effector）

Encoder $\mathcal{E}$（1D CNN with Group CNN 早期层，把 translation / rotation / gripper 先各自走一遍再 fuse）把 $\mathbf{A}$ 压成 latent $f \in \mathbb{R}^{L \times C}$。
- $L$ = compressed temporal length（一般 $L \ll H$）
- $C$ = latent feature dim（codebook dim，32 或 64）

Spectrum decoder $\mathcal{D}_{\mathrm{spec}}$ 先用 action decoder $\mathcal{D}$ 把 latent decode 回 $\hat{\mathbf{A}} \in \mathbb{R}^{H \times D}$，再沿 temporal 轴做 **DCT-II**：

$$
\mathbf{F}_{k,d} = \sum_{h=0}^{H-1} \hat{\mathbf{A}}_{h,d} \cos\!\Bigl[\tfrac{\pi}{H}\bigl(h+\tfrac{1}{2}\bigr)k\Bigr], \quad k=0,\ldots,H-1 \tag{1}
$$

变量解释：
- $h$ = time index（0 到 H-1）
- $k$ = frequency index；$k=0$ 是 DC 分量（整段的均值 / constant offset），$k$ 越大频率越高，$k=H-1$ 是 Nyquist 频率
- $d$ = action dimension index
- $\cos[\frac{\pi}{H}(h+\tfrac{1}{2})k]$ 是 DCT-II 的 basis function；$+\tfrac{1}{2}$ 是 half-sample offset，保证 basis 在 boundary 上对称，能量集中性比 DCT-I 好 — 这就是 JPEG 选 DCT-II 的原因

输出 $\mathbf{F} \in \mathbb{R}^{H \times D}$ 就是 frequency-domain spectrum。

直觉：把一条 trajectory 想象成一段音频。DCT-II 把它拆成 H 个频率 bin，bin 0 是整段的均值（"动作的大致位置"），bin 1 是最慢的 cosine（"慢慢移动的方向"），... bin H-1 是最快震荡（"手抖"）。Robot trajectory 大部分能量集中在低频 bin（动作是 smooth 的），高频能量小但承载 reactive correction。

**工程细节（paper 附录 A.1）**：gripper 这个 binary dim 被显式排除在 DCT 之外 — 因为对 0/1 信号做频域分解没有意义（你不会说 "gripper 的低频是 0.5"），所以 gripper 单独走 time-domain 重建，continuous dim 走频域。这种 modality-aware 的处理是工程上的小细节但很重要。

### 2.2 Multi-Scale Residual Quantization

借鉴 [RQ-VAE (Lee et al. CVPR 2022)](https://arxiv.org/abs/2203.01940) 和 [VAR](https://arxiv.org/abs/2404.02905)。设 K 个 scale，每个 scale 的 token 数 $\{l_1, \ldots, l_K\}$ 是 increasing resolution，且 $l_K = L$。共享一个 codebook $\mathcal{Z} \in \mathbb{R}^{V \times C}$，V 是 codebook size。

Algorithm 1 的核心循环（我把它的过程翻译成直觉）：

```
f^(0) ← E(A)                    # 初始 latent = encoder output
for k = 1..K:
    s_k ← Q(Interpolate(f^(k-1), l_k))   # 把 residual 上采样到 l_k，再 VQ
    z_k ← Lookup(Z, s_k)                  # 查表得到 embedding
    z_k ← Interpolate(z_k, l_K)            # 上采样回最大分辨率
    f^(k) ← f^(k-1) - φ_k(z_k)            # residual = 当前 latent - 投影后的 quantized
    f_hat^(k) ← f_hat^(k-1) + φ_k(z_k)    # 累积重建
    F^(k) ← DCT(D(f_hat^(k)))             # 用累积 latent decode + DCT 得到第 k 个 spectrum
```

关键变量：
- $f^{(k)}$ = 第 k scale 之后的 residual feature（还剩什么没解释）
- $\hat{f}^{(k)}$ = 累积到第 k scale 的 latent approximation（已经解释了什么）
- $\phi_k$ = scale-specific projector，把 codebook embedding 映射回 latent space（每个 scale 一个 MLP）
- $l_k$ = 第 k scale 的 token 数（LIBERO 用 [1, 2, 4]；CALVIN 用 [1, 2, 3, 4]；real-world 用 [1, 2, 3, 4]）

为什么 $l_1 = 1$？因为最粗 scale 必须是单个 token，它才能被解释成 "this chunk's intent"。如果 $l_1 = 2$，你就有两个 token 都在解释低频，disentanglement 就糊了。

为什么所有 scale 共享一个 codebook？因为不同 scale 学到的是不同 frequency band 的 primitives，但 primitive 类型本身应该是同一套（"pick up", "move forward", "rotate" 在不同 scale 都可能出现，只是粒度不同）。共享 codebook 还能避免 codebook collapse 在不同 scale 间漂移。附录提到用 EMA（ratio 0.99）更新 codebook，这是 [VQ-VAE original](https://arxiv.org/abs/1711.00937) 的经典 trick，比 gradient-based 更新稳定得多。

### 2.3 Scale-wise Spectral Loss — 灵魂所在

定义累积 latent 近似（Eq.2）：

$$
\hat{f}^{(k)} = \sum_{i=1}^{k} \phi_i\bigl(\mathrm{Lookup}(\mathcal{Z}, \mathbf{s}_i)\bigr) \tag{2}
$$

每个 $\hat{f}^{(k)}$ 经过 spectrum decoder 得到第 k 个 partial reconstruction $\mathbf{F}^{(k)} = \mathrm{DCT}(\mathcal{D}(\hat{f}^{(k)}))$。Ground truth 是 $\mathbf{F} = \mathrm{DCT}(\mathbf{A})$。

Scale-wise spectral loss（Eq.3）：

$$
\mathcal{L}_{\mathrm{freq}} = \sum_{k=1}^{K} \lambda_k \,\bigl\| \mathbf{F} - \mathbf{F}^{(k)} \bigr\|_2 \tag{3}
$$

变量：
- $\lambda_k$ = 第 k scale 的权重（paper 没明说但默认是 1 或递减，让 coarse scale loss 权重大些）
- $\|\cdot\|_2$ 是 L2 norm（应该是 Frobenius，整个 matrix 一起算）

这是和 CARP（[Gong et al. ICCV 2025](https://arxiv.org/abs/2410.11758)）最关键的区别。CARP 也做 multi-scale residual VQ，但只用 **terminal time-domain reconstruction**，也就是只在最后一层算 $\|\mathbf{A} - \hat{\mathbf{A}}\|$。结果是：scale 之间没有 explicit 的频段分工，coarse token 可能去 memorize 高频细节（因为 terminal loss 不约束中间 scale），disentanglement 失败。

MINT 的 design：每个 scale 都必须能从自己的累积 latent 重建一个 spectrum，并且这个 spectrum 要尽量接近 ground truth spectrum。由于：
1. Cosine basis 是按 frequency 排序的
2. Robot action 的 spectrum 能量集中低频（smooth motion 假设）
3. L2 loss 在频域里是 additive 的（Parseval，严格说是 Frobenius norm 保能量）

所以 optimizer 会优先让 coarse scale 抓低频（因为低频项对 L2 的贡献大，先抓它们收益最高），细 scale 抓 residual 里的高频 — 这是 implicit 但 strong 的归纳偏置。

Ablation Table IV 印证了这点：
- Terminal Time-Domain Loss only: CALVIN 4.36 / LIBERO-Long 87.8%
- + Terminal Spectral Loss: 4.41 / 88.2% （仅加 terminal spectral 略微提升）
- + Scale-Wise Time-Domain Loss: 4.06 / 82.8% **(反而下降!)** — scale-wise 在 time domain 反而让 coarse token 去拟合 time-domain 细节，破坏 disentanglement
- + Scale-Wise Spectral Loss (本文): 4.54 / 93.4% **(+5.6% on LIBERO-Long)** — 关键 ablation

这个对比强烈说明：**不是 multi-scale 本身有效，而是 multi-scale + frequency domain 的组合**。Time domain 上做 scale-wise supervision 会让 coarse scale 也去拟合时间维上的 high-frequency residual（因为 time domain 看不到 frequency 信息，L2 loss 平等地惩罚每个 timestep），结果 coarse token 被污染，disentanglement 失败。

这让我想到 [Rahaman et al. 2019 "On the Spectral Bias of Neural Networks"](https://proceedings.mlr.press/v97/rahaman19a.html) — 神经网络天生有 low-frequency bias (spectral bias / frequency principle)。MINT 反过来利用这点：既然 NN 偏好低频，那你让 coarse scale 用 small capacity（1 个 token）去 reconstruct，它会自动选低频；细 scale 补残差。Frequency loss 是显式强化这个 prior。

### 2.4 Total training objective (Eq.4)

$$
\mathcal{L} = \mathcal{L}_{\mathrm{freq}} + \underbrace{\|\mathrm{sg}(f) - \hat{f}\|_2^2}_{\text{Codebook loss}} + \underbrace{\|f - \mathrm{sg}(\hat{f})\|_2^2}_{\text{Commitment loss}} + \alpha\underbrace{\|\mathbf{A} - \hat{\mathbf{A}}\|_1^2}_{\text{Auxiliary}}
$$

- $\mathrm{sg}(\cdot)$ = stop-gradient，阻止梯度流过
- Codebook loss 让 codebook embedding $e$ 朝 encoder output $f$ 移动
- Commitment loss 让 encoder output 别跑离 codebook 太远（commit 到一个 code）
- $\alpha$ = auxiliary L1 权重（paper 没明说数值）
- Auxiliary L1 在 time domain 上算，保证 final reconstruction（用所有 K scale）忠实

辅助 L1 是必要的因为频域 L2 loss 不直接约束 time-domain fidelity（Parseval 只在 orthogonal 变换 + Frobenius norm 下严格成立；L2 频域 loss 和 L2 时域 loss 数学上等价，但 optimize landscape 不同；L1 时域则提供 sparsity / robustness prior）。

---

## 3. MINT Policy: Next-Scale Autoregressive Modeling

Phase 2 训 policy。给定 observation $\mathbf{o}_t$ + language $\ell$ + robot proprio $\mathbf{p}_t$，policy 要 output 一串 action tokens $\mathbf{S} = (\mathbf{s}_1, \ldots, \mathbf{s}_K)$，然后 decode 成 continuous trajectory。

### 3.1 Joint distribution factorization (Eq.4 in paper, 我标 Eq.4')

$$
p(\mathbf{s}_1, \mathbf{s}_2, \ldots, \mathbf{s}_K) = \prod_{k=1}^{K} p\bigl(\mathbf{s}_k \mid \mathbf{s}_1, \ldots, \mathbf{s}_{k-1}\bigr) \tag{4'}
$$

这里 $\mathbf{s}_k$ 是一个 **token map**（包含 $l_k$ 个 token），不是单个 token。Next-scale prediction：在第 k 步，conditioned on 所有更粗的 scale $(\mathbf{s}_1, \ldots, \mathbf{s}_{k-1})$，并行预测当前 scale 的所有 $l_k$ 个 token。

这是直接从 [VAR](https://arxiv.org/abs/2404.02905) 借来的 trick，搬到 action domain 的好处：
1. **Coarse-to-fine 推理结构** — 先决定 intent（1 个 token），再决定粗 trajectory（2 个 token），再细化（4 个 token）。这符合人类 planning 直觉。
2. **Parallel decoding within scale** — 同 scale 内的 token 是 conditional independent given 更粗 scale，可以一次 forward 全出，比 next-token 快 $\sqrt{L}$ 倍左右（VAR 论文里讨论过这个 scaling）。
3. **KV cache 兼容** — 跨 scale autoregressive，每生成完一个 scale 就 cache 住，下一个 scale 的 attention 能复用。

Hybrid attention mask：scale k 的 token 只能 attend $\mathbf{s}_{\le k}$（即更粗 + 自己同 scale），不能看更细 scale — 这是 causal 在 scale 维度的体现。

Loss 是 standard cross-entropy：$\mathcal{L}_{\mathrm{CE}} = -\sum_k \log p(\mathbf{s}_k^{\mathrm{GT}} \mid \mathbf{s}_{<k}^{\mathrm{GT}})$，因为 SDAT 已经把 continuous action quantize 成 discrete token，policy 本质上是 classification（在每个 scale 上预测 V 个 code 哪个最接近）。

### 3.2 Intent-Based Action Ensemble (Eq.5, 6)

Action chunking 的常见做法（[ACT, Zhao et al. RSS 2023](https://tonyzhaozh.github.io/aloha/)）：每次预测未来 H 步 action，下一步只执行 1 步，再重新预测；多个 chunk 对同一 timestep 的 prediction 用高斯权重 ensemble（近的权重大）。

MINT 改进：用 **intent token 的相似度** 代替固定高斯权重。

$$
\mathbf{a}_t = \sum_{h=0}^{H} w_h^{\mathrm{intent}} \cdot \mathbf{a}_t \mid \mathbf{o}_{t-h} \tag{5}
$$

$$
w_h^{\mathrm{intent}} = \frac{\exp\bigl(\beta \langle \mathbf{s}_1^{(t)}, \mathbf{s}_1^{(t-h)}\rangle\bigr)}{\sum_{j=0}^{H} \exp\bigl(\beta \langle \mathbf{s}_1^{(t)}, \mathbf{s}_1^{(t-j)}\rangle\bigr)} \tag{6}
$$

变量：
- $\mathbf{a}_t \mid \mathbf{o}_{t-h}$ = 在 t-h 时刻基于观测 $\mathbf{o}_{t-h}$ 预测的、对应到当前时刻 t 的 action（rolling chunk overlap）
- $\mathbf{s}_1^{(t)}$ = 时刻 t 的 intent token embedding
- $\mathbf{s}_1^{(t-h)}$ = 时刻 t-h 的 intent token embedding
- $\langle \cdot,\cdot\rangle$ = cosine similarity
- $\beta > 0$ = temperature，越大 softmax 越尖锐
- $H$ = chunk horizon

直觉：当 behavior 在切换时（e.g. 从 "approach" 到 "grasp"），不同时刻预测的 intent token 是不同的，cosine similarity 低，旧 chunk 的 prediction 应该被迅速 down-weight；当 behavior 在 continue 时（持续 "move forward"），不同时刻的 intent 一致，相似度高，多个 chunk 的 prediction 一起 ensemble 平滑掉噪声。

Ablation (Table IV) 证实：
- No Ensemble: 4.09 / 85.8%
- Temporal-based (ACT 高斯): 4.32 / 89.2%
- Action-based (CogACT，在 action 空间算相似度): 4.10 / 90.4%
- **Intent-based (本文): 4.57 / 93.2%**

Action-based ensemble 不如 intent-based 的原因：action space 是 high-dim noisy 的，cosine 在 action 上算容易被高频细节 dominate；intent token 是 abstract 1-token representation，相似度干净。

### 3.3 两个 model variant

**MINT-30M**（from scratch，验证框架本身有效）：
- Vision: frozen SigLIP (400M) + DINOv2 (300M)，feature concat
- Language: frozen BERT
- 注入方式：FiLM ([Perez et al. AAAI 2018](https://arxiv.org/abs/1709.07871)) — $\gamma, \beta$ 由 language feature 预测，对 visual feature 做 affine modulation
- Transformer: 8 layers, 12 heads, width 1024, MLP dim 4096
- ~30M trainable params（vision/language 都 frozen）

**MINT-4B**（large-scale，对标 π0.5）：
- VLM backbone: PaliGemma-2.6B（SigLIP vision encoder + Gemma-2B LLM），用 π0.5 的预训练权重初始化
- Action expert: 300M params, decoder-only Transformer (width 1024, MLP 4096)
- **不**用 π0.5 的 DiT flow-matching action head，而是 decoder-only transformer 做 next-scale AR — 这是为了兼容 scale-wise autoregressive decoding
- Action expert 从 scratch 训（不继承 π0.5 action head 权重）

预训练数据：[BridgeDataV2 (Walke et al.)](https://arxiv.org/abs/2308.12952) — 60k trajectories, 24 envs, 13 skills
Fine-tune data: per-task 20 demos

---

## 4. 实验：把数据讲清楚

### 4.1 LIBERO (Table I)

LIBERO 有 5 个 suite：Spatial / Object / Goal / Long / 90，难度递增（Long 是长 horizon，需要 composition）。

| Method | Spatial | Object | Goal | Long | Avg | L90 |
|---|---|---|---|---|---|---|
| Diffusion Policy | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 | – |
| ACT | ~70s | – | – | – | – | – |
| OpenVLA (7B pretrained) | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | – |
| π0 (pretrained) | 90.0 | 86.0 | 95.0 | 73.0 | 86.0 | – |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 | – |
| OpenVLA-OFT | 96.9 | 98.1 | 95.6 | 91.1 | 95.4 | – |
| π0.5 (pretrained) | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 | 96.0 |
| **MINT-30M (from scratch)** | 98.6 | 99.2 | 97.4 | 93.2 | **97.1** | **97.4** |
| **MINT-4B** | 97.4 | 99.6 | 98.2 | 97.8 | **98.3** | **98.7** |

观察：
1. **MINT-30M 用 30M 参数 from scratch 打败 7B 的 OpenVLA** — 这强烈说明 tokenization 是 VLA bottleneck，不是 model size
2. MINT-4B 在 Long 上 97.8 vs π0.5 92.4 (+5.4) — 长 horizon 收益最大，因为 intent token 让 long-horizon planning 更稳
3. 在 L90（90 个 task 的 multi-task 版本）上 MINT-4B 98.7 — 几乎 saturate 这个 benchmark

### 4.2 CALVIN ABCD→D (Table I middle)

要求连续完成 5 个任务，500 chain。报告 SR@k（连续完成 k 个 task 的成功率）和 Avg Len（平均完成 task 数，max 5）。

| Method | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 | Avg Len |
|---|---|---|---|---|---|---|
| RT-1 | 84.4 | 61.7 | 43.8 | 32.3 | 22.7 | 2.45 |
| Robo-Flamingo | 96.4 | 89.6 | 82.4 | 74.0 | 66.0 | 4.08 |
| π0.5 | 94.2 | 89.3 | 82.7 | 78.5 | 70.3 | 4.15 |
| RoboVLMs | 96.7 | 93.0 | 89.9 | 86.5 | 82.6 | 4.49 |
| MINT-4B | 97.4 | 94.2 | 91.7 | 88.2 | **86.1** | **4.57** |

SR@5 上 86.1 vs π0.5 70.3 — 长序列 composition 上提升 15.8 个点。这个 gap 在 long-horizon 上特别大，符合 "intent-to-execution reasoning" 的设计目的。

### 4.3 MetaWorld (Table I right)

按 Easy/Medium/Hard/Very Hard 分。

| Method | Easy | Med | Hard | V.Hard | Avg |
|---|---|---|---|---|---|
| Diffusion Policy | 23.1 | 10.7 | 1.9 | 6.1 | 10.5 |
| TinyVLA | 77.6 | 21.5 | 11.4 | 15.8 | 31.6 |
| π0 | 77.9 | 51.8 | 53.3 | 20.0 | 50.8 |
| **MINT-4B** | 82.1 | 72.4 | 58.3 | **56.0** | **67.2** |

**Very Hard 上 56.0 vs π0 20.0，接近 3x** — 这是 paper 里最 striking 的数字之一。Very Hard task 通常需要高精度 alignment 和长时间协调，正好是 intent/execution 分离最受益的场景。

### 4.4 LIBERO-Plus (Table II / X) — Generalization

7 个 perturbation factor：Camera Viewpoints, Robot Initial States, Language Instructions, Light, Background, Sensor Noise, Object Layout。

| Method | Camera | Robot | Lang | Light | Back | Noise | Layout | Avg |
|---|---|---|---|---|---|---|---|---|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 16.3 |
| UniVLA | 1.8 | 46.2 | 69.9 | 69.0 | 81.0 | 21.2 | 31.9 | 45.9 |
| π0.5 | 53.0 | 50.3 | 65.7 | 83.1 | 77.3 | 53.2 | 72.7 | 65.0 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 71.4 |
| MINT-30M | 61.4 | 41.2 | 61.6 | 92.2 | 77.1 | 76.5 | 76.2 | 69.5 |
| MINT-4B | 72.2 | 42.4 | 85.8 | 96.6 | 88.9 | 90.1 | 84.6 | **80.1** |
| OpenVLA-OFT+ | 92.8 | 30.3 | 85.8 | 94.9 | 93.9 | 89.3 | 77.6 | 80.7 |
| π0.5+ | 67.2 | 42.4 | 59.4 | 75.8 | 74.9 | 72.6 | 64.5 | 65.3 |
| **MINT-4B+** | **95.6** | 44.6 | 84.7 | 95.1 | 94.5 | 95.2 | 78.7 | **84.1** |

观察：
- **Camera Viewpoints shift 是 VLA 的死穴** — OpenVLA 0.8, UniVLA 1.8, π0.5 53.0。MINT-4B 72.2，MINT-4B+ 95.6 — 显著领先。原因是 intent token 是 view-invariant abstraction（grab 的 intent 不应该因为 camera 角度变），execution token 才 encode view-specific 的 visual servoing。
- MINT-4B+ 在 LIBERO-Plus 上 fine-tune 后 84.1，比 OpenVLA-OFT+ 80.7 高 3.4，但 spread 更 uniform — 不像 OpenVLA-OFT+ 在 Robot Initial State 上只有 30.3（明显 overfit 到 camera 但牺牲了 robot state robustness）。
- 15% 提升 over OpenVLA-OFT 在 strong disturbance 下 — 这是 paper abstract 里说的数字。

### 4.5 One-shot Transfer (Table III) — 这是最大的 conceptual win

三种 OOD shift：New Task（完全没见过的 task semantics）、New Layout（task 见过但 layout 全新）、Extended Horizon（比训练时更长的序列）。

| Method | Task Spec | New Task | New Layout | Ext. Horizon | Avg |
|---|---|---|---|---|---|
| Replay | Replay | 0.28 | 0.12 | 0.04 | 0.11 |
| Fine-tune (MINT-30M, 1 demo) | Language | 0.42 | 0.08 | 0.00 | 0.17 |
| **Intent-injection (MINT-Zero-30M)** | Intent | **0.90** | **0.68** | **0.72** | **0.77** |

Intent-injection 的 protocol：
1. 给 1 个 demo trajectory
2. 用 SDAT encoder 抽出它的 $\mathbf{s}_1$ token（intent token）
3. 在 inference 时，把 $\mathbf{s}_1$ 固定，policy conditioned on 这个 $\mathbf{s}_1$ autoregressively 生成 $\mathbf{s}_2, \ldots, \mathbf{s}_K$，再 decode 成 trajectory

这本质上是把 intent token 当作 **task specification modality**（与 language instruction 平行）。Language 是离散的、ambiguous 的（"pick up the cup" 可能是 100 种具体动作），intent token 是 continuous、execution-aligned 的（它就是从一段真实 trajectory 抽出来的）。

为什么 fine-tune 在 Extended Horizon 上是 0.00？因为 fine-tune 1 个 demo 会让 model 灾难性 forget 训练时学到的 multi-task 能力，反而退化成 replay。Intent injection 没有梯度更新，不会 forget，反而能 generalize。

Avg 0.77 vs 0.17 — **60% 提升**，paper abstract 里的数字。这是让我觉得最 conceptual 的实验：它直接证明了 intent token 是 transferable 的 task specification，比 language 更 grounded。

### 4.6 Real-World (Fig. 5)

4 个 task：Place Banana, Stack Blocks, Insert Marker, Stack Cups (zero-shot)。20 demos/task，6-DOF Piper-X arm。

Bayesian posterior 分析显示 MINT 在 (A)(B) 上 statistically distinguishable from all baselines（ACT, π0, π0.5*）。在 unseen task (D) Stack Cups 上也好 — 它 generalize 了 (B) Stack Blocks 的 "stacking" intent。Abstract 里说 real-world 上比 π0.5 高 29%。

### 4.7 学习效率 (Table VII)

LIBERO Long 上不同 iter 的 success rate：

| Iter | 1k | 2k | 3k | 5k | 10k |
|---|---|---|---|---|---|
| ACT | 0.06 | 0.21 | 0.27 | 0.53 | 0.65 |
| MINT-30M | 0.00 | 0.43 | 0.74 | 0.87 | 0.95 |
| π0-FAST | 0.35 | 0.55 | 0.67 | 0.76 | 0.83 |
| π0.5 | 0.39 | 0.64 | 0.73 | 0.80 | 0.89 |
| MINT-4B | 0.53 | 0.76 | 0.82 | 0.94 | 0.97 |

MINT-30M 在 1k iter 还是 0，但 2k iter 就到 0.43（ACT 同期 0.21）— 快速收敛。MINT-4B 在 1k iter 就 0.53，超过 π0.5 的 0.39。Sample efficiency 来自 structured tokenization：predict 1 个 intent token 比 predict 16 个 continuous action 容易得多。

### 4.8 Scale 数量 ablation (Table VIII)

| Scales | CALVIN Avg Len | LIBERO-Long |
|---|---|---|
| [1] | 2.12 | 42.8 |
| [1,4] | 4.06 | 78.4 |
| [1,2,4] | 4.46 | **93.6** |
| [1,2,3,4] | **4.57** | 92.2 |
| [1,2,4,6,8] | 4.32 | 88.6 |

[1] 单 scale 退化成 VQ-VAE，性能很差 — 证明 multi-scale 是必要的。但太多 scale (5 个) 反而变差 — 因为每个 scale 的容量太小，optimization 不稳。**3-4 个 scale 是 sweet spot**。这和 VAR 论文里的发现一致。

### 4.9 Action chunk horizon ablation (Table VIII)

| Horizon | CALVIN | LIBERO-Long |
|---|---|---|
| 8 | 3.74 | 80.6 |
| 16 | 4.47 | **93.2** |
| 32 | 4.49 | 86.6 |
| 64 | 4.26 | 87.4 |

16-32 之间最好。太短 model 不能 plan，太长 trajectory 变 high-dim 不稳定。这和 [Diffusion Policy](https://arxiv.org/abs/2303.04137) 的发现一致。

---

## 5. T-SNE 可视化 — Intent 真的学到语义了

Fig.1 right 和 Fig.12 把 $\mathbf{s}_1$ token embedding 做 t-SNE，发现 cluster 对应语义行为：pick up, move forward, clockwise rotation 等。这个 cluster 在 LIBERO 和 CALVIN 上都出现 — cross-benchmark 语义一致，说明 intent token 学到的不是 dataset-specific artifact，而是 behavioral abstraction。

Ablation Fig.6 对比 time-domain reconstruction 和 spectral reconstruction 的 latent space：前者 fragmented（颜色乱），后者 coherent（颜色 cluster 干净）。这是 disentanglement 的直接 visual evidence。

---

## 6. 与相关工作的 positioning

### 6.1 与 CARP 的关系（最近邻）
[CARP (Gong et al. ICCV 2025)](https://arxiv.org/abs/2410.11758) 也是 multi-scale coarse-to-fine autoregressive policy，但：
1. CARP 只用 terminal time-domain reconstruction loss — coarse token 也会去拟合 high-frequency detail，disentanglement 不成立
2. CARP 没有 intent token 的概念，没有 one-shot transfer 实验
3. MINT 的 Scale-Wise Spectral Loss ablation 显示 93.4% vs CARP-style 82.8% — 10 个点 gap，证明频域监督是关键

### 6.2 与 FAST 的关系
[FAST (Pertsch et al.)](https://arxiv.org/abs/2501.09747) 用 DCT 把整段 action 压成一个 token sequence（每 freq 一个 token），但 flat — 没有 multi-scale hierarchy，没有 intent / execution 分离。MINT 借了 FAST 的 DCT insight，但加了 multi-scale 结构。

### 6.3 与 π0 / π0.5 的关系
π0 用 flow matching（continuous diffusion）on action，π0.5 加了 hierarchical reasoning。MINT-4B 直接用 π0.5 的 VLM backbone（PaliGemma）和预训练权重，但 action head 换成 decoder-only transformer 做 next-scale AR — 这是为了让 action token 的离散结构能被 autoregressive modeling 利用。Flow matching 是 continuous 的，无法直接做 next-scale AR。

### 6.4 与 BC-Z / few-shot imitation 的关系
[BC-Z (Jang et al.)](https://arxiv.org/abs/2204.06152) 和 [MAML (Finn et al.)](https://arxiv.org/abs/1703.03400) 都是 few-shot imitation 经典方法，靠 gradient update 或 meta-learning。MINT 的 intent injection 完全不用梯度更新，是 test-time conditioning，更接近 [CMA (Dance et al.)](https://arxiv.org/abs/2106.04950) 之类 task embedding 方法，但 task embedding 来自一个 demo 的最粗 scale token，比 learnable task embedding 更 interpretable。

### 6.5 与 hierarchical RL 的关系
[HAC / HIRO / fuN](https://arxiv.org/abs/1710.10006) 等 hierarchical RL 用 subgoal / option 做 abstraction，但需要 reward signal。MINT 在 IL setting 下，用 frequency-domain decomposition implicit 学到 hierarchy — 不需要 subgoal annotation。

### 6.6 与 Wavelet 的关系
Wavelet 也是 multi-resolution analysis，但 wavelet 是 spatially localized（每个 wavelet coefficient 对应一个 time-frequency tile），DCT 是 globally localized（每个 cosine basis 跨越整段）。对 robot action 来说 DCT 更合适，因为 intent 是全 chunk 的 global property，不是 local window property。Wavelet 更适合 image compression。

---

## 7. 整体架构图（文字版）

```
PHASE 1: SDAT training (offline, on demo trajectories)
─────────────────────────────────────────────────────
Action chunk A (H×D)
    │
    ▼ [1D CNN encoder, Group CNN for modalities]
Latent f (L×C)
    │
    ▼ [Multi-scale Residual VQ]
    │   ┌─ s_1 (1 token, intent) ──→ φ_1 ──→ z_1
    │   ├─ s_2 (2 tokens)         ──→ φ_2 ──→ z_2
    │   └─ s_K (l_K tokens)       ──→ φ_K ──→ z_K
    │
    ▼ [cumulative: f_hat^(k) = Σ φ_i(z_i)]
Spectrum decoder D_spec
    │
    ▼ [D → A_hat^(k), then DCT]
F^(k) (H×D spectrum per scale)
    │
    ▼ [L_freq = Σ λ_k ||F - F^(k)||_2  +  VQ losses  +  L1 aux]

PHASE 2: MINT Policy training (online, on (obs, action) pairs)
─────────────────────────────────────────────────────
(o_t, language, proprio)
    │
    ├─ Vision: SigLIP + DINOv2 (30M) / PaliGemma (4B)
    ├─ Language: BERT (30M) / Gemma (4B)
    │
    ▼ [Action expert: decoder-only Transformer]
Next-scale AR prediction:
    p(s_1) → p(s_2|s_1) → ... → p(s_K|s_<K)
    │   (within scale: parallel; across scale: AR)
    │
    ▼ [CE loss against SDAT tokenized GT]

INFERENCE
─────────────────────────────────────────────────────
Predict s_1, s_2, ..., s_K
    │
    ▼ [SDAT decoder]
Continuous trajectory
    │
    ▼ [Intent-based ensemble (Eq.5, 6)]
Final action a_t
```

---

## 8. 我觉得有启发 / 可质疑的点

**有启发：**

1. **Frequency-domain decomposition 作为 disentanglement prior** — 这是个很 general 的 idea。其实可以推到 video generation（low freq = scene layout, high freq = motion detail）、language modeling（low freq = topic, high freq = syntax variation）。MINT 在 action 上的成功暗示这条路有更广的应用。

2. **Intent token 作为 task specification modality** — 比 language 更 grounded。这暗示 future VLA 可能应该有 multi-modal task conditioning：language / goal image / demo trajectory 三者通过 shared intent space 统一。

3. **Coarse-to-fine 推理匹配 planning 的 inductive bias** — 人类做事也是先想 intent 再想 execution。MINT 把这个 hierarchy explicit 化在 token space，让 transformer 自然学到 coarse-to-fine reasoning。

**可质疑 / future direction：**

1. **Intent token 来自 demo trajectory** — 这意味着 intent space 被训练数据限制。Paper 的 Limitation 章节自己提到：从 web-scale video 学更丰富的 intent 是 future work。

2. **Codebook size 是 handpicked** — LIBERO 512, CALVIN 512, MetaWorld 256, BridgeV2 1024，差异很大但没 principled 选择方法。可能可以学 [BIG-VAE](https://arxiv.org/abs/2012.00162) 做 codebook size 自适应。

3. **Scale 数量是 ablation 来的 [1,2,4]** — 没有 principled 自动选择机制。理论上 frequency band 应该自动 partition。

4. **One-shot transfer 的 New Layout 0.68** — 比 New Task 0.90 低。可能因为 layout 变化要求 execution 也变，但 intent injection 只锁 intent，execution 仍由 policy generate — 这部分 generalize 还是不够。

5. **Codebook collapse** — paper 用 EMA 缓解，但没给 codebook utilization 数据。codebook 是否真的 100% 都被用？

6. **MINT-4B 在 LIBERO Spatial 上 97.4 比 π0.5 98.8 略低** — 不是全面碾压。可能 Spatial 任务对 intent abstraction 需求小，VLM backbone 的 visual grounding 才是瓶颈。

---

## 9. 与你 work 的潜在联系

Andrej，几个我猜你会感兴趣的 connection：

1. **Tokenization 决定 modeling** — 你在 [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) 和 nanoGPT 反复强调：character vs BPE vs SentencePiece 直接决定模型学到什么。MINT 是这个 idea 在 robot action 上的极致体现 — 把 action tokenize 成 multi-scale spectral token，模型就被 forced 学到 intent。

2. **Autoregressive 的本质** — nanoGPT 的 next-token prediction 在 text 上 work，在 pixel 上要 next-scale（VAR），在 action 上 MINT 也用 next-scale。这暗示 next-token 不是通用解，next-structure (scale / cluster / chunk) 可能才是 general principle。

3. **Inductive bias 的力量** — MINT-30M (from scratch) 打 OpenVLA (7B pretrained) 这件事让我想到你的 [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — 在 small data regime，inductive bias 比 scale 重要。MINT 的 spectral disentanglement 就是精心设计的 inductive bias。

4. **Pre-training 与 fine-tuning 的关系** — MINT-4B 用 π0.5 预训练 VLM 权重，但 action expert 从 scratch。这暗示 vision-language pretrain 跨 task transferable，但 action head 必须 retrain — 因为 action tokenization scheme 变了。这和你 LLM 里 tokenization 改了就要 re-embed 一样。

参考链接汇总：
- [MINT paper (arxiv 应该会有)](https://arxiv.org/abs/2510.13626) — LIBERO-Plus paper 引用它的 SJTU group
- [VAR: Visual Autoregressive Modeling](https://arxiv.org/abs/2404.02905)
- [VQ-VAE (van den Oord)](https://arxiv.org/abs/1711.00937)
- [RQ-VAE](https://arxiv.org/abs/2203.01940)
- [FAST](https://arxiv.org/abs/2501.09747)
- [π0](https://arxiv.org/abs/2410.24164)
- [π0.5](https://arxiv.org/abs/2504.16054)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)
- [UniVLA](https://arxiv.org/abs/2505.06111)
- [CARP](https://arxiv.org/abs/2410.11758)
- [ACT / ALOHA](https://tonyzhaozh.github.io/aloha/)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [CogACT](https://arxiv.org/abs/2411.19650)
- [LAPA](https://arxiv.org/abs/2410.11758)
- [PaliGemma](https://arxiv.org/abs/2407.07726)
- [BridgeDataV2](https://arxiv.org/abs/2308.12952)
- [LIBERO](https://arxiv.org/abs/2306.03310)
- [CALVIN](https://arxiv.org/abs/2112.03227)
- [MetaWorld](https://arxiv.org/abs/1910.10846)
- [FiLM](https://arxiv.org/abs/1709.07871)
- [Spectral Bias of NN (Rahaman)](https://proceedings.mlr.press/v97/rahaman19a.html)
- [DCT original (Ahmed et al.)](https://ieeexplore.ieee.org/document/1672377)
- [LIBERO-Plus](https://arxiv.org/abs/2510.13626)

---

## 一句话总结

MINT 的核心 elegance 在于：它把 robot action 当成 signal，用 DCT + multi-scale residual VQ + scale-wise spectral loss 这一套 classical signal processing toolbox，强制把 intent (low frequency) 和 execution (high frequency) 分到不同 token scale，然后用 next-scale autoregressive policy 让推理结构匹配 planning 的 coarse-to-fine 直觉。结果是一个 30M from-scratch 模型能打 7B pretrained VLA，4B 模型在 long-horizon / OOD generalization / one-shot transfer 上全面 SOTA。这个工作让我相信 — **tokenization 才是 VLA 下一阶段的关键瓶颈，不是 scale**。
