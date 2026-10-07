---
source_pdf: U-DiTs Downsample Tokens in U-Shaped Diffusion.pdf
paper_sha256: 400f45a827c47bc09f30769dc4f85bf8b1418dd46bc57a59422d81c0d76eca2c
processed_at: '2026-08-12T18:50:06-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 U-DiT

## 一句话版本

DiT说U-Net没用，这篇paper说"你错了，是你没把U-Net用对"——把attention里的tokens先下采样再做，省了3/4计算量，效果反而更好。

---

## 背景故事

DiT出来的时候很嚣张，说U-Net那个"先压缩再还原"的套路根本不重要，纯transformer堆叠就够强了。很多人信了，纷纷转向isotropic架构。

但这帮人心里嘀咕：U-Net在diffusion里用了这么多年，真的白用了？

作者就试了一下，把DiT的block塞进U-Net结构里，结果发现：**几乎没提升**。FID只好了4点，FLOPs还差不多。看起来DiT是对的？

但作者没放弃。他们想：U-Net的特征有个特点，**主要是低频信息**（物体的大致结构），高频部分主要是噪声。那在U-Net里做full-scale attention，等于花大量计算去让tokens互相attention噪声，纯属浪费。

---

## 核心trick

既然U-Net特征是低频主导的，那attention之前先把tokens下采样不就完了？

具体做法：
1. 输入feature通过Pixel-UnShuffle变成4个下采样版本
2. 每个版本独立做self-attention
3. 做完attention再Pixel-Shuffle拼回原尺寸

数学上很美：每个下采样attention的cost是原来的1/16，4个加起来是1/4，**省了75%的attention计算**。而且Pixel-Shuffle是lossless的，只是空间和channel的重新排列，信息一点没丢。

---

## 为什么这个能work

关键是**跟diffusion任务的频谱特性对齐了**：
- Diffusion要denoise，噪声在high-frequency
- U-Net的downsampling是天然low-pass filter，已经帮你滤掉一部分高频噪声了
- 所以U-Net backbone里的features本来就低频主导
- 在这些低频features上做dense attention是浪费
- Token downsampling强制attention只在低频上发生，跟任务特性对齐

之前有人试过只下采样KV（保留full-scale query），效果反而变差。因为query在"模糊"的key/value上做attention，但输出还是full resolution，逻辑上不自洽。U-DiT同时下采样Q/K/V，attention完全在低分辨率空间里发生，最后再upsample回来，逻辑一致。

---

## 结果有多炸

U-DiT-B用22 GFLOPs击败了DiT-XL/2的119 GFLOPs（FID 16.64 vs 20.05）。**6倍的效率差距**。

U-DiT-L训练60万步就超过DiT-XL/2训练700万步。训练compute效率高一个数量级。

在512×512分辨率、CFG=1.5的conditional生成等各种场景下优势都保持。

---

## 附加改进

除了核心的token downsampling，还加了一些trick进一步提性能：

1. **Cosine similarity attention** - Q/K先归一化再做dot product，防止attention变太sharp
2. **RoPE2D** - 用旋转矩阵编码2D位置信息，比absolute position embedding好
3. **Depthwise conv in FFN** - 在FFN的linear层之间插个轻量conv，给transformer加local spatial prior
4. **Re-parameterization** - 训练时用多分支，推理时fused成单分支，免费提升

这些trick各贡献2-5 FID，但**核心的token downsampling单独就贡献18 FID**，是最关键的。

---

## 为什么之前没人发现

几个原因凑一起：
1. DiT说U-Net不重要的结论太有影响力，大家直接信了
2. 之前KV downsampling的工作效果不好，让人觉得"下采样attention这条路走不通"
3. Token downsampling看起来很radical（连query都下采样），没人敢试
4. 这个low-frequency prior的洞察来自FreeU那个工作，比较新

---

## 我的takeaway

这篇paper给我最大的启发是：**当一个"旧技术"看起来失效时，先别急着否定它，可能是你没给它发挥空间**。

U-Net在latent DiT上看起来没用，是因为plain transformer block在U-Net backbone上做了大量redundant computation。把这些redundancy切掉（通过token downsampling），inductive bias才显现出来。

更深层的洞察：**diffusion的denoising任务在频谱上是anisotropic的**——低频是结构，高频是噪声。Standard attention是isotropic的，对所有频率一视同仁。Token downsampling相当于给attention加了个low-pass prior，跟任务特性align，所以work。

这个insight可能对video generation（temporal维度更复杂）、其他生成模型都有启发。值得好好消化。

---

# U-DiTs: Downsample Tokens in U-Shaped Diffusion Transformers 深度解析

你好 Andrej，这篇来自 PKU & Huawei Noah's Ark Lab 的 paper 直击一个很有意思的现象：**DiT 抛弃了 U-Net 的 inductive bias，但这个 bias 其实并没有真正失效，只是 naive 组合时被冗余的 computation 掩盖了**。作者用一个非常 elegant 的 token downsampling trick 把它"激活"了。下面我从直觉、架构、数学、实验四个层面拆解给你看。

---

## 1. Motivation: 为什么 rethinking U-Net in DiT?

DiT (Peebles & Xie, ICCV 2023) [https://arxiv.org/abs/2212.09748] 的核心论点是：isotropic transformer stack 已经足够强，U-Net 的 inductive bias "not crucial"。这篇 paper 直接挑战这个观点。

作者的 toy experiment 很关键 (Table 1)：

| Model | GFLOPs | FID↓ |
|---|---|---|
| DiT-S/4 | 1.41 | 97.85 |
| DiT-UNet (naive) | 1.40 | 93.48 |
| DiT-UNet + KV Downsampling | 0.91 | 94.38 |
| DiT-UNet + Token Downsampling (Ours) | 0.90 | **89.43** |

注意几个 subtle 的点：
- naive DiT-UNet 只比 isotropic DiT-S/4 好 ~4 FID，这个 margin 太小，确实支持了 DiT 作者"U-Net 不重要"的论调
- 但是，当把 self-attention 的 tokens 整体下采样之后，FID 直接降到 89.43，FLOPs 还少了 0.5G
- 只 downsample KV（之前 works 的做法）反而比 naive 更差（94.38 vs 93.48）

这里直觉上非常关键：**U-Net 的 inductive bias 本身是有效的，但是 plain transformer block 在 U-Net backbone 上有大量 redundant computation，这个 redundancy 把 inductive bias 的收益吃掉了**。

---

## 2. 核心洞察：U-Net backbone features 是 low-frequency dominated

这个洞察来源于 FreeU (Si et al., 2023) [https://arxiv.org/abs/2309.11497]，他们做了 frequency analysis，发现 latent U-Net denoiser 的中间 features 在频域上 energy 集中在 low-frequency。

为什么这跟 diffusion 有关？Williams et al. (NeurIPS 2023) [https://arxiv.org/abs/2309.14302] 在 "A unified framework for u-net design and analysis" 中理论上证明：**U-Net 的 downsampling stage transitions 是天然的 low-pass filter，自动 discard 掉 high-frequency subspaces，而这些 subspaces 恰恰是被 noise dominate 的**。

所以推理链是这样：
1. Diffusion 信号在 high-frequency 上主要是 noise
2. U-Net downsampling 本质是 low-pass filter，filter noise
3. U-Net backbone features 因此 low-frequency dominated
4. 在这些 features 上做 dense self-attention 是 wasteful 的——大量 attention 计算用在了高频 noise 上
5. Solution: 在 self-attention 之前先 downsample tokens，强制 attention 在 low-frequency 上工作

---

## 3. 架构图解析 (Figure 3)

Figure 3 展示了三种架构的 evolution：

**(a) DiT (isotropic)**: 一串 plain transformer blocks，latent spatial size 全程不变。

**(b) DiT-UNet (naive)**: 把 DiT blocks 塞进 U-Net 结构。3 stages，encoder 做 2 次 spatial 2× downsampling (feature dim 翻倍)，decoder 对称 upsample。Skip connections 在每个 stage transition 处 concatenate。**关键：每个 stage 内部的 DiT block 仍是 full-scale self-attention**。

**(c) U-DiT (proposed)**: 在 (b) 基础上，每个 DiT block 内部的 self-attention 改成 **downsampled self-attention**。Figure 3 右侧的 attention 模块里，input features 先被 split 成 4 个下采样版本（通过 Pixel-UnShuffle），分别做 self-attention，最后 merge 回去。

这里有个 subtle 但关键的差异 vs U-ViT (Bao et al., CVPR 2023) [https://arxiv.org/abs/2209.12152]：U-ViT 是 **isotropic transformer + shortcuts**，本质上 spatial size 不变，只是多了 skip connections；U-DiT 是 **true U-Net**，有真正的 multi-stage downsampling/upsampling。这个区别很容易被忽视。

---

## 4. Token Downsampling 的数学细节

### 4.1 标准 Self-Attention 复杂度

给定 input feature size $N \times N$，dimension $d$：

$$Q, K, V \in \mathbb{R}^{N^2 \times d}$$

其中 $N^2$ 是 token 数量（spatial 展平），$d$ 是 embedding dimension。

$$X = \underbrace{A V}_{\mathcal{O}(N^4 d)} \quad \text{s.t.} \quad A = \text{Softmax} \underbrace{(Q K^T)}_{\mathcal{O}(N^4 d)}$$

变量解释：
- $Q$: query matrix, shape $(N^2, d)$
- $K$: key matrix, shape $(N^2, d)$
- $V$: value matrix, shape $(N^2, d)$
- $A$: attention weight matrix, shape $(N^2, N^2)$
- $\mathcal{O}(N^4 d)$ 的来源：$Q K^T$ 是 $(N^2, d) \times (d, N^2) = (N^2, N^2)$，每个元素需要 $d$ 次乘加，所以是 $N^4 d$；同理 $A V$ 也是 $N^4 d$

### 4.2 Downsampled Self-Attention

将 input 通过 Pixel-UnShuffle split 成 4 个 $\frac{N}{2} \times \frac{N}{2}$ 的下采样版本：

$$4 \times (Q_{\downarrow 2}, K_{\downarrow 2}, V_{\downarrow 2}) \in \mathbb{R}^{(\frac{N}{2})^2 \times d}$$

下标 $\downarrow 2$ 表示 spatial 2× downsampling。每个 downsampled feature 的 token 数是 $(\frac{N}{2})^2 = \frac{N^2}{4}$。

每个 self-attention 的 cost:
$$\mathcal{O}\left(\left(\frac{N^2}{4}\right)^2 d\right) = \mathcal{O}\left(\frac{N^4}{16} d\right)$$

4 个并行 self-attention 的总 cost：
$$4 \times \mathcal{O}\left(\frac{N^4}{16} d\right) = \mathcal{O}\left(\frac{N^4}{4} d\right)$$

**节省了 3/4 的 self-attention 计算**。这个 saving 是 quadratic 的，因为 attention 是 $O(n^2)$，token 数减半意味着 cost 减 4 倍。

### 4.3 Pixel-UnShuffle / Pixel-Shuffle 的 lossless 特性

Appendix A.1 详细描述了过程：

1. 输入 $QKV$ shape: $(b, 3c, h, w)$ — $b$ 是 batch, $3c$ 是 Q/K/V 三者的 channel 合并, $h, w$ 是 spatial
2. Pixel-UnShuffle with stride $s$: 重排空间像素到 channel 维度，输出 $4 \times (b \cdot s^2, 3c, h/s, w/s)$
3. 每个 downsampled QKV 做 vanilla multi-head self-attention
4. 输出 $4 \times (b \cdot s^2, c, h/s, w/s)$
5. Pixel-Shuffle 合并: $(b, c, h, w)$

**关键点：整个过程是 lossless 的**。Pixel-Shuffle/UnShuffle 只是 spatial ↔ channel 的 rearrange，没有任何信息损失。这跟 bicubic/bilinear downsampling 不同，后者会丢失高频信息。这其实也是这个方法能 work 的原因之一——下采样只是为了 attention 计算方便，但是原始信息全都在。

---

## 5. Downsampler 的设计 (Table 7)

| Downsampler | GFLOPs | FID↓ |
|---|---|---|
| Pixel Shuffle (PS) only | 0.89 | 96.15 |
| Depthwise Conv + PS | 0.91 | 89.87 |
| DW Conv + Shortcut + PS | 0.91 | **89.43** |
| 普通 Conv + PS (参考) | 2.22 | - |

直觉分析：
- **Pixel Shuffle only** 最便宜但最差（96.15），因为纯 rearrange 没有任何 learnable mixing
- **普通 Conv** 会 double computation（2.22 GFLOPs），因为 conv 在 channel 维度上做 dense mixing，cost 跟 channel 数平方相关
- **Depthwise Conv** 只在每个 channel 内部做 spatial mixing，cost 跟 channel 数线性相关，几乎不增加 FLOPs
- **+Shortcut** 是 re-parameterizable 的：训练时 $y = \text{DWConv}(x) + x$，推理时可以 fold 成单个 conv（参考 RepVGG [https://arxiv.org/abs/2101.03697]），所以 inference 时几乎是免费的

这里的 re-parameterization trick 很聪明：shortcut 在训练阶段帮助 gradient flow 和 representation learning，inference 时被 fuse 掉，没有任何 overhead。

---

## 6. 其他架构改进 (Table 9 ablation)

从 DiT-UNet (Slim) baseline 一路叠加：

| Component | FID↓ | Δ |
|---|---|---|
| DiT-UNet (Slim, ~0.9 GFLOPs) | 107.00 | baseline |
| + Token Downsampling | 89.43 | **-17.57** ← 最大贡献 |
| + Cosine Similarity Attention | 86.96 | -2.47 |
| + RoPE2D | 84.64 | -2.32 |
| + DWConv FFN | 79.30 | -5.34 |
| + Re-param FFN | 75.71 | -3.59 |

**Token downsampling 贡献了 ~18 FID 的提升**，是所有 trick 里最 effective 的。其他 trick 各贡献 2-5 FID。

逐个讲解：

**Cosine Similarity Attention** (来自 Swin V2 [https://arxiv.org/abs/2111.09883]):
$$A = \text{Softmax}\left(\frac{Q' K'^T}{\tau}\right), \quad Q' = \frac{Q}{\|Q\|}, \quad K' = \frac{K}{\|K\|}$$
即 Q/K 先做 L2 normalize 再做 dot product。$\tau$ 是 learnable temperature。这避免了 dot product 数值爆炸导致 attention 退化，特别是训练后期 attention 变 sharp 的时候。

**RoPE2D** (Rotary Position Embedding, [https://arxiv.org/abs/2104.09864]):
将位置信息通过 rotation matrix 注入 Q 和 K：
$$q_i' = R_i q_i, \quad k_j' = R_j k_j$$
其中 $R_i, R_j$ 是取决于 token 位置 $i, j$ 的 rotation matrices。2D 版本对 spatial 两个维度分别 encode。RoPE 的好处是 relative position 自动通过 $R_i^T R_j = R_{i-j}$ 编码，且对外推（不同分辨率）友好。

**DWConv FFN**: 在 FFN 的两个 linear 层之间插入一个 depthwise conv：
$$\text{FFN}(x) = W_2 \cdot \text{DWConv}(\text{GELU}(W_1 x)) + b$$
这让 FFN 获得 local spatial inductive bias，弥补 transformer 的"global only"缺陷。

**Re-param FFN**: 训练时用 multi-branch 结构（类似 RepVGG），推理时 fuse 成单 conv。

---

## 7. 主结果分析 (Table 2, 3, 4)

### ImageNet 256×256, 400K steps, no CFG

| Model | FLOPs(G) | FID↓ | IS↑ |
|---|---|---|---|
| DiT-S/2 | 6.07 | 67.40 | 20.44 |
| **U-DiT-S** | 6.04 | **31.51** | 51.62 |
| DiT-B/2 | 23.02 | 42.84 | 33.66 |
| **U-DiT-B** | 22.22 | **16.64** | 85.15 |
| DiT-XL/2 | 118.68 | 20.05 | 66.74 |
| **U-DiT-L** | 85.00 | **10.08** | 112.44 |

几个关键观察：
1. **U-DiT-B (22.2 GFLOPs) 比 DiT-XL/2 (118.7 GFLOPs) 还好** — 1/6 FLOPs, 更好 FID (16.64 vs 20.05)
2. **U-DiT-L (85 GFLOPs) 比 DiT-XL/2 好 10 FID** (10.08 vs 20.05)
3. U-DiT-S vs DiT-S/2 同 FLOPs 下 FID 差 36 — 这是 huge gap

### With CFG=1.5 (Table 4)

| Model | FLOPs(G) | FID↓ |
|---|---|---|
| DiT-XL/2* | 118.68 | 6.24 |
| U-DiT-B | 22.22 | **4.26** |
| U-DiT-L | 85.00 | **3.37** |

U-DiT-B 在 CFG 下甚至 beat 了 5x 大的 DiT-XL/2，这非常 striking。

### Scaling to 1M steps (Table 5)

| Model | Steps | FID↓ |
|---|---|---|
| DiT-XL/2 | 7M | 9.62 |
| U-DiT-L | 600K | **8.71** |
| U-DiT-L | 1M | 7.54 |

U-DiT-L 训练 600K 步就超过 DiT-XL/2 训练 7M 步 — **训练 compute 效率 ~12x**。

### 512×512 (Table 6)

| Model | FLOPs(G) | FID↓ |
|---|---|---|
| DiT-XL/2* | 524.7 | 20.94 |
| U-DiT-B | 106.7 | **15.39** |

在更高分辨率上 advantage 依然保持，U-DiT-B 用 1/5 FLOPs 击败 DiT-XL/2。

---

## 8. 架构配置 (Table 8)

| Model | Params(M) | FLOPs(G) | Channel | Heads | Encoder-Decoder |
|---|---|---|---|---|---|
| U-DiT-S | 52.05 | 6.04 | 96 | 4 | [2,5,8,5,2] |
| U-DiT-B | 204.43 | 22.22 | 192 | 8 | [2,5,8,5,2] |
| U-DiT-L | 810.19 | 85.00 | 384 | 16 | [2,5,8,5,2] |

`[2,5,8,5,2]` 表示 encoder 有 [2,5,8] blocks（3 stages，每 stage downsampling 一次），decoder 有 [8,5,2] blocks（mirror）。最深的 stage 在最底层（most compressed），共 22 blocks。Channel 在每个 stage transition 翻倍：U-DiT-L 是 384 → 768 → 1536。

注意 latent space 是 32×32 (256×256 / 8 by VAE)，所以 3 stages 后 feature 是 8×8 — 不算太 compressed，但足够 trigger U-Net inductive bias。

---

## 9. Training overhead (Table 11)

| Model | Steps/Sec | FID↓ |
|---|---|---|
| DiT-XL/2 | 1.71 | 20.05 |
| U-DiT-L (Vanilla) | 1.55 | 12.04 |
| U-DiT-L (+All Mods) | 0.84 | 10.08 |

注意 vanilla U-DiT-L 训练速度几乎跟 DiT-XL/2 一样（1.55 vs 1.71），但 FID 已经好很多（12.04 vs 20.05）。加 all mods 后训练速度下降到 0.84 steps/sec，主要是因为 RoPE2D 和 cosine attention 的额外计算。

---

## 10. Vanilla vs Improved (Table 10)

| Model | FLOPs | FID↓ | Δ |
|---|---|---|---|
| U-DiT-S (Vanilla) | 5.91 | 41.01 | - |
| U-DiT-S (+All) | 6.04 | 31.51 | -9.5 |
| U-DiT-L (Vanilla) | 84.48 | 12.04 | - |
| U-DiT-L (+All) | 85.00 | 10.08 | -1.96 |

有意思的发现：**vanilla U-DiT-L (FID 12.04) 已经击败 DiT-XL/2 (FID 20.05) 8 FID**，说明 token downsampling 这个核心 trick 单独就足够强。其他 mods 在大模型上 contribution 变小（U-DiT-S 提升 9.5，U-DiT-L 只提升 2），可能是大模型本身 capacity 足够， tricks 的边际收益递减。

---

## 11. 跟相关工作的对比

### 跟 U-ViT 的区别
U-ViT [https://arxiv.org/abs/2209.12152] 是 isotropic transformer + skip connections，**没有真正的 spatial downsampling**。U-DiT 是 true U-Net，有 multi-stage downsampling。Table 3 显示 U-ViT-XL (113 GFLOPs) FID 18.35，远不如 U-DiT-L (85 GFLOPs, FID 10.08)。

### 跟之前 KV downsampling 的区别
之前的工作如 SRFormer [https://arxiv.org/abs/2309.17433], TODO (Smith et al., 2024), PixArt-Σ [https://arxiv.org/abs/2403.04692] 都只 downsample KV，保留 full-scale Q。这意味着：
- 输出 token 数仍然是 $N^2$（每个 query 都要计算）
- 只是 attention matrix 计算变小

U-DiT 同时下采样 Q，输出 token 数变 $(N/2)^2$，**4 个并行 attention 完全独立**。然后通过 Pixel-Shuffle merge 回 full resolution。这种 radical 设计可能更彻底地利用了 low-frequency prior。

Table 1 直接对比了这两种方案：
- KV downsampling: FID 94.38（比 baseline 还差）
- Token downsampling (Q+K+V): FID 89.43（显著好）

为什么 KV downsampling 反而变差？一个可能的解释：保留 full-scale query 但用 downsampled key/value 会让 attention 的 spatial alignment 错位，相当于 query 在"模糊"的 feature 上做 attention，但 output resolution 还是 full，这导致 information mismatch。而 token downsampling 让 attention 完全在下采样空间里发生，最后再 unshuffle 回来，逻辑上一致。

### 跟 FreeU 的联系
FreeU [https://arxiv.org/abs/2309.11497] 也是基于"U-Net backbone low-frequency dominated"的洞察，但他们的方法是 **在 inference 时 amplify low-frequency (backbone) 信号, suppress high-frequency (skip) 信号**：
$$x_{backbone}' = \alpha \cdot x_{backbone}$$
$$x_{skip}' = \beta \cdot \text{FFT-filter}(x_{skip})$$
不改架构，只是 scaling。U-DiT 则是在训练时就让 attention 自然 focus on low-frequency，更深入。

### 跟 Pyramid ViT / Hierarchical Transformer 的联系
Swin Transformer [https://arxiv.org/abs/2103.14030], PVT [https://arxiv.org/abs/2102.12122] 等也是 hierarchical 架构，但它们的 downsampling 是 **永久性的**——spatial size 真的减小。U-DiT 的 token downsampling 是 **局部的、可逆的**——只在 attention 内部 downsample，attention 结束后立刻 Pixel-Shuffle 回来，spatial size 在 block 之间保持。这是关键区别。

---

## 12. 局限与思考

作者承认 limitation 是 compute 资源不够，没训练更大模型。我看了下后续的 U-DiT 后续工作，确实可以 scale 到更大。

几个我自己思考的点：
1. **为什么不试 stride=4 downsampling？** Paper 只用了 stride=2（4 个 downsampled tokens）。如果用 stride=4 (16 个 tokens)，每个 attention cost 是 $1/256$，总 cost 是 $16/256 = 1/16$。但 16 个 attention 是否还能 cover 全部 spatial 信息？这是个有趣的 ablation 方向。

2. **跟 linear attention 的关系**：Token downsampling 本质上是把 $O(N^4 d)$ 的 attention 降到 $O(N^4 d / 4)$。Linear attention 类工作（如 Performer, Linear Transformer）用 kernel trick 降到 $O(N^2 d)$。两条路都在 attack attention 的 quadratic bottleneck，但 token downsampling 更简单且保持 exact attention semantics。

3. **跟 SORA / Video DiT 的关系**：Video generation 中 temporal dimension 让 attention cost 进一步爆炸。Token downsampling 这种 trick 在 video DiT 上可能 even more impactful。Sora [https://openai.com/research/video-generation-models-as-world-simulators] 用的 DiT 结构 details 不公开，但很可能会用类似的 efficiency trick。

4. **为什么 latent space 不需要 U-Net 但 pixel space 需要？** 这篇 paper 反过来证明 latent space 也 benefit from U-Net。之前 community 的认知（包括 DiT paper 自己 claim 的）可能是错的——naive U-Net 在 latent 上没赢是因为 redundancy 太多，不是 inductive bias 本身没用。

---

## 13. Intuition 总结

这个 paper 给我的最大启发是一个方法论：**当一个 inductive bias 看起来"没用"时，先别急着否定它，看看是不是 implementation 没有给它发挥空间**。

U-Net 在 latent DiT 上看起来没用，是因为 plain transformer block 在 U-Net backbone 上做了大量 redundant computation（在 high-frequency noise 上算 attention）。Token downsampling 把这些 redundancy 切掉，inductive bias 才真正显现。

公式背后更深的 insight：**diffusion model 的 denoising 任务在频谱上是 anisotropic 的**——low-frequency 携带 structure，high-frequency 主要是 noise。Standard attention 是 isotropic 的，对所有 frequency 一视同仁。Token downsampling 等价于在 attention 上施加了一个 low-pass prior，这个 prior 跟 diffusion task 的频谱特性 align，所以 work。

---

## References

- U-DiT GitHub: [https://github.com/YuchuanTian/U-DiT](https://github.com/YuchuanTian/U-DiT)
- DiT: [https://arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)
- U-ViT: [https://arxiv.org/abs/2209.12152](https://arxiv.org/abs/2209.12152)
- FreeU: [https://arxiv.org/abs/2309.11497](https://arxiv.org/abs/2309.11497)
- Williams U-Net framework: [https://arxiv.org/abs/2309.14302](https://arxiv.org/abs/2309.14302)
- Stable Diffusion: [https://arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)
- Swin V2: [https://arxiv.org/abs/2111.09883](https://arxiv.org/abs/2111.09883)
- RoFormer (RoPE): [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
- RepVGG: [https://arxiv.org/abs/2101.03697](https://arxiv.org/abs/2101.03697)
- SRFormer: [https://arxiv.org/abs/2309.17433](https://arxiv.org/abs/2309.17433)
- PixArt-Σ: [https://arxiv.org/abs/2403.04692](https://arxiv.org/abs/2403.04692)
- FiT: [https://arxiv.org/abs/2402.12376](https://arxiv.org/abs/2402.12376)
- SiT: [https://arxiv.org/abs/2401.08740](https://arxiv.org/abs/2401.08740)
- Sora: [https://openai.com/research/video-generation-models-as-world-simulators](https://openai.com/research/video-generation-models-as-world-simulators)
