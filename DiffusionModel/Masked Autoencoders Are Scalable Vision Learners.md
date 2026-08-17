---
source_pdf: Masked Autoencoders Are Scalable Vision Learners.pdf
paper_sha256: 1b490443925c72a2b7c770f90dd797e248729ae34a57e1abfe9ed36751c4cc5b
processed_at: '2026-08-05T16:30:11-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MAE 用人话讲 — Whiteboard Edition

## 0. 一句话先抓住魂

想象你给模型看一张照片，但把照片撕掉 75%，只剩零星几块碎片。模型要"脑补"出剩下 75% 长什么样。这个"脑补"过程逼着它真正理解这是一只狗、一辆车、一片沙滩——而不是单纯做像素级的颜色插值。这就是 MAE。

下面我把整篇 paper 的逻辑串成一条 line of thought，告诉你 Kaiming 他们为什么这么干，每个 ablation 背后的 intuition 是什么。

---

## 1. 他们到底想解决什么问题

2021 年中那个时间点，vision 自监督有点尴尬。Contrastive learning（SimCLR、MoCo、BYOL、DINO）很火，但有一堆问题：

- **依赖 heavy data augmentation**。SimCLR 没 crop+color jitter 直接 collapse，BYOL cropping-only 掉 13%。
- **Scaling 卡在 ViT-L**。DINO 和 MoCo v3 都试过 ViT-Huge，但 accuracy 不涨甚至退化。NLP 那边 GPT-3 都 175B 了，vision 这边 ViT-H 都跑不动，这就很丢人。
- **Pretext task 本身很 tricky**。Contrastive 需要 negative pairs、需要 stop-gradient、需要 predictor network、需要 EMA — 一堆工程 hacks 撑起来的。

Kaiming 的 question 很朴素：NLP 的 BERT 那么干净，mask 一段 token 然后预测就行了，凭什么 vision 不能这么干？

之前确实有人试过（iGPT、Context Encoder、BEiT），但都没"飞起来"。Kaiming 想搞清楚为什么，然后把这个 gap 填上。

---

## 2. 三个 Key Insight — 用直觉讲

### 2.1 第一个 insight：image 的 information density 跟 text 完全不一样

你想想 BERT mask 一个句子 15% 的 token，比如 "The cat sat on the [MASK]" — 模型必须懂语法、懂语义、懂世界知识才能填出 "mat"。

但 image 是 natural signal，spatial redundancy 极重。你 mask 一张狗的照片 15% 的 patches，缺的那块几乎可以从周围 patch 直接"涂"过去 — 狗毛的颜色、纹理都连续。模型走个 low-level interpolation shortcut 就完事了，根本没学什么叫"狗"。

所以 Kaiming 的策略很暴力但很对：**把 mask ratio 拉到 75%**。

75% 意味着什么？224×224 图像切成 196 个 16×16 patches，只给你看 49 个。你看 Figure 2 里那些 masked image — 你自己作为人类都看着费劲，但模型居然能填出 plausible 的狗、车、人。这时候它就**不可能**走 low-level shortcut 了，必须 holistic understand "这大概是什么东西"。

这个 75% 是 Figure 5 sweet spot，但 intuition 是：mask 太少模型偷懒，mask 太多模型直接放弃，75% 刚好在"难到必须懂语义、但不至于 ill-posed"的临界点。

### 2.2 第二个 insight：decoder 在 vision 和 NLP 里角色根本不同

BERT 的 decoder 可以 trivial（一个 linear layer），因为它预测的是 word — word 本身就是 semantic entity。Encoder 学到的 latent 已经 semantic 了，decoder 只是"翻译"出来。

但 vision 里你 reconstruct 的是 pixel。Pixel 是 low-level 的，不是 semantic entity。如果你的 decoder 太弱，那 pixel reconstruction 的 gradient 就会直接"压回"encoder，强迫 encoder 的最后几层也去学 "这个 patch 的 RGB values 大概多少" — 这就毁了，encoder latent 不再 abstract。

所以 Kaiming 说：OK，那我给一个 **足够 deep 但 narrow 的 decoder**（8 个 block，512-dim，vs encoder ViT-L 是 24 block 1024-dim）。让 decoder 把"pixel-level specialization"承担掉，encoder 的顶层就可以专心做 abstract reasoning。

你看 Table 1a 的数据：decoder 从 1 block 增加到 8 block，linear probing 从 65.5 涨到 73.5（涨 8 个点），但 fine-tuning 基本不变（84.8 → 84.9）。

Intuition：linear probing 用 encoder 最后输出，所以 decoder 深度直接影响 latent quality；fine-tuning 时 encoder 后几层会重新 tune，所以 decoder 影响小。这个 ablation 单独看好像没什么，但合起来看就解释了为什么 asymmetric 设计是必要的。

### 2.3 第三个 insight：encoder 千万别吃 mask token

这是 engineering 上最 elegant 的一招。

之前 BEiT、iGPT 都是 encoder 直接吃完整 sequence（包括 [M] token）。这有两个 problem：

**Problem A — Pretrain-test gap**：pretraining 时 encoder 输入是 75% 都是 [M] 的 sequence，但 downstream fine-tune 时你输入的是干净 image，没有 [M]。这个 distribution shift 会让 transfer performance 掉。

Table 1c 直接做了实验：让 encoder 也吃 [M] 的话，linear probing 从 73.5 掉到 59.6 — 掉 14 个点。这是巨大 gap。

**Problem B — 计算浪费**：ViT self-attention 复杂度是 $O(N^2)$。如果 encoder 处理全部 196 个 token，FLOPs 跟普通 ViT 一样。但 MAE encoder 只处理 49 个 visible token：

$$\text{FLOPs ratio} = \left(\frac{N_v}{N}\right)^2 = \left(\frac{49}{196}\right)^2 = \frac{1}{16}$$

Encoder FLOPs 直接降到 1/16。Table 1c 测出来整体训练 FLOPs 是 3.3× speedup，Table 2 测出来 wall-clock 是 2.8–4.1× speedup。对 ViT-Huge 来说 4.1× 意味着你 4 天能 train 完的 model 现在 1 天搞定 — 这就是为什么 MAE 是第一个能 scale 到 ViT-H 还跑得动的 self-supervised method。

把 [M] 推迟到 decoder 才插入，等于 encoder 始终看的是真实 patches，pretrain 和 deploy 完全 consistent。

---

## 3. Forward Pass — 完整走一遍

我用人话讲一遍数据流，你脑子里建立 mental model。

输入：$x \in \mathbb{R}^{224 \times 224 \times 3}$，一张 RGB 图。

**Step 1 — Patchify**：切成 196 个 16×16 patch，每个 patch 拉平成 768-dim vector（因为 $16 \times 16 \times 3 = 768$）。然后过一个 linear projection 把它压到 encoder dim $d_e = 1024$。加上 positional embedding（sine-cosine），告诉模型每个 patch 的空间位置。

**Step 2 — Shuffle-based masking**：这是个很简洁的 implementation trick。把 196 个 token list 做 random shuffle，然后取前 49 个作为 visible，后 147 个直接丢掉。注意：**没有 [M] token 插入**。

为什么用 shuffle 实现？因为 shuffle 是 $O(N)$ 的，比 sparse operation 快多了，而且不需要任何特殊 CUDA kernel。在 TPU/GPU 上 vectorized implementation 非常快。

**Step 3 — Encoder forward**：49 个 visible token 进 ViT encoder，经过 24 个 transformer block，输出 49 个 1024-dim latent representation。这就是 encoder 的全部工作。

**Step 4 — Unshuffle + append [M]**：准备 decoder 输入。先生成 147 个 shared learnable [M] token（每个都是同一个 512-dim vector，learnable），append 到 49 个 encoder output 后面。但是 — 这 49 + 147 是错位的，要把它们 unshuffle 回原始 196 个 patch 的空间位置。

Unshuffle 是 shuffle 的逆操作，相当于"按原顺序还原"。这步操作几乎没 cost，但让 implementation 极其干净。

再加 decoder 的 positional embedding，让 [M] token 知道自己在图像哪个位置（不然 [M] 全是同一个 vector，模型分不出来位置）。

**Step 5 — Decoder forward**：完整 196 个 token 进 decoder（8 block、512-dim），输出 196 个 512-dim representation。

**Step 6 — Pixel projection**：decoder 最后一个 linear layer 把 512-dim 投影到 768-dim（= $16 \times 16 \times 3$），reshape 回 16×16×3 patch 形状。这就是 reconstruction 出来的 patch。

**Step 7 — Loss**：只对 147 个 masked patch 算 MSE loss：

$$\mathcal{L} = \frac{1}{|\Omega_m|} \sum_{i \in \Omega_m} \| \hat{x}_i - x_i \|_2^2$$

变量解释：
- $\Omega_m$：masked patch 的 index 集合，size $|\Omega_m| = 147$
- $\hat{x}_i$：第 $i$ 个 patch 的 reconstruction，是个 768-dim vector
- $x_i$：第 $i$ 个 patch 的原始 pixel values，也是 768-dim

Visible patches 不算 loss — 这是 BERT tradition，避免模型学到"copy identity"的 trivial solution。

**Normalized pixel variant**（主结果用的）：先把每个 patch 的 pixel 减 mean 除 std：

$$\bar{x}_i = \frac{x_i - \mu_i}{\sigma_i + \epsilon}$$

其中 $\mu_i$、$\sigma_i$ 是第 $i$ 个 patch 内 256 个 pixel 的 mean 和 std（标量，跨 3 个 channel 共享或独立）。

Loss 变成：

$$\mathcal{L}_{norm} = \frac{1}{|\Omega_m|} \sum_{i \in \Omega_m} \| \hat{x}_i - \bar{x}_i \|_2^2$$

Intuition：normalize 掉每个 patch 的"整体亮度"，强迫模型学 patch 内部的 structure/contrast，不要走"输出平均颜色"的 trivial path。Table 1d 显示这给 fine-tune 涨 0.5 个点。

---

## 4. 那些反常识的 Ablation

### 4.1 Data augmentation 几乎不需要

Table 1e 最 striking 一行：**no augmentation（只有 center crop，连 flip 都没有）linear probing 65.7%**。

你对比 SimCLR 没 augmentation 直接 collapse、BYOL cropping-only 掉 13%。MAE 居然在 no aug 下都能 work，甚至 fine-tune 84.0% — 比 supervised ViT-L from scratch 还高。

Why？因为 **random masking 本身就是 per-instance augmentation**。每次 iteration 你对同一张 image 做不同 mask，相当于生成了无限多个 "views"。Masking 既 create task difficulty，又 regularize training。Color jitter 反而 hurt（Table 1e 最后一行），因为 color jitter 改变了 pixel distribution，让 reconstruction target 跟 input inconsistent。

这个发现很重要 — 它告诉你 MAE 跟 contrastive learning 在哲学上是 different species。Contrastive learning 靠 "view diversity" 学 invariant feature，MAE 靠 "input corruption" 学 generative model。

### 4.2 Mask sampling — random 完胜

Table 1f：

| 策略 | ratio | ft | lin | Intuition |
|---|---|---|---|---|
| random | 75 | 84.9 | 73.5 | missing 区域周围还有部分 context，必须 holistic reason |
| block-wise | 75 | 82.8 | 63.9 | 大块连续 missing，局部 context 完全丢，task 太难 reconstruction 模糊 |
| grid | 75 | 84.0 | 66.0 | 规律性保留，task 容易，模型走 shortcut |

Random 在 75% 这个高 ratio 下是唯一 work 的。Block-wise 适合 50% 但 75% 就崩。Grid 看起来 reconstruction sharp 但 representation 差 — 说明 sharp reconstruction 跟 good representation 不一定正相关，模型可能走 "texture completion" shortcut。

### 4.3 Reconstruction target — pixel 就够，别 tokenize

Table 1d 比较 pixel vs PCA vs dVAE token（BEiT 那种）：

| Target | ft | lin |
|---|---|---|
| pixel (no norm) | 84.9 | 73.5 |
| pixel (norm) | **85.4** | 73.9 |
| PCA 96 coeffs | 84.6 | 72.3 |
| dVAE token | 85.3 | 71.6 |

PCA 反而最差，因为 PCA 砍掉 high-frequency component。但 high-frequency 在 reconstruction 任务里是 useful signal — 强迫模型学 detailed structure，不只是 average color。

dVAE token（BEiT 那套）fine-tune 涨 0.4% 但 linear probing 掉 1.9%，且 Table 7 transfer learning 上 normalized pixel 跟 dVAE 统计等价。所以 tokenize 的额外复杂度（要先用 250M image 训 dVAE）没换来 gain。

Kaiming 的 takeaway：pixel MSE 足够好，simple is better。

---

## 5. Scaling — 这才是真正的大新闻

### 5.1 ImageNet-1K only 的 SOTA

Table 3：

| Method | ViT-B | ViT-L | ViT-H | ViT-H448 |
|---|---|---|---|---|
| supervised scratch (their impl) | 82.3 | 82.6 | 83.1 | - |
| DINO | 82.8 | - | - | - |
| MoCo v3 | 83.2 | 84.1 | - | - |
| BEiT | 83.2 | 85.2 | - | - |
| **MAE** | **83.6** | **85.9** | **86.9** | **87.8** |

注意几个事情：

1. **DINO 和 MoCo v3 在 ViT-L 之后就不 scale 了**。DINO 根本没跑 ViT-H，MoCo v3 在 ViT-L 是 84.1 跟 BEiT 84.1 持平，但都没 stretch 到 ViT-H。
2. **MAE 从 ViT-B 83.6 → ViT-L 85.9 → ViT-H 86.9 → ViT-H448 87.8**，几乎线性提升。这是 NLP 里 GPT/BERT 那种 scaling behavior 第一次在 vision self-supervised 里复现。
3. **ViT-H448 87.8% 是当时 IN1K-only 数据的 SOTA**。之前最好的 87.1% 还需要 advanced architecture（VOLO）+ 512 input size。MAE 用 vanilla ViT 就 beat 了。

### 5.2 跟 supervised pretraining 的对比

Figure 8 是这篇文章的灵魂 figure 之一。ViT-L supervised from scratch 82.6%，ViT-H supervised 83.1% — capacity 越大，supervised 越没 gain，甚至 degrade。但 MAE pretrain 让 ViT-L 85.9%、ViT-H 86.9% — capacity 越大 gain 越大。

这个 trend 几乎完美复刻了原 ViT paper 里 JFT-300M supervised pretraining 的 scaling curve。Kaiming 的 point：MAE 用 IN1K 1M image 就 match 了 300M image 的 supervised pretraining scaling behavior。这就是 self-supervised 应该有的样子。

### 5.3 Training schedule

Figure 7 显示 MAE 1600 epoch 都没饱和，linear probing 还在涨。这跟 MoCo v3 在 300 epoch 就饱和完全不同。

但 wall-clock 上 MAE ViT-L 1600 epoch 是 31 小时，MoCo v3 ViT-L 300 epoch 是 36 小时（同 128 TPU-v3）。MAE 实际上**更快**！

Why？因为 MAE encoder 每 epoch 只看 25% patches，MoCo v3 用 two-crop 每 epoch 看 200%。所以 1600 epoch 的 MAE 相当于 200 epoch 的 "full data exposure"。Masking ratio 高反而让长 training schedule 变得 affordable。

---

## 6. Partial Fine-tuning — 一个被低估的 protocol

Figure 9 这张图我觉得是 paper 里最被忽视的 ablation。看数据：

| Tuned blocks | MAE | MoCo v3 |
|---|---|---|
| 0 (linear probe) | 73.5 | 76.7 |
| 1 block | 81.0 | 79.5 |
| 4 blocks | 84.1 | 81.5 |
| 24 (full) | 84.9 | 84.1 |

故事是这样的：

- **Linear probing** 上 MoCo v3 赢 MAE 3.2 个点（76.7 vs 73.5）。如果你只看这个 metric，你会觉得 contrastive learning 比 MAE 好。
- **但 tune 一个 block 之后 MAE 立刻反超**（81.0 vs 79.5），tune 4 个 block gap 是 2.6 个点。
- **只 tune 最后一个 MLP sub-block**（sub-block of last block）MAE 就 79.1%。比 linear probe 高 5.6 个点。

Interpretation：MAE 的 representation **不是 linearly separable 的，但是高度 nonlinearly informative 的**。Linear probing 这个 metric 偏好 contrastive learning 那种 "已经 linear aligned" 的 representation。但 linear separability 不是 representation quality 的全部。

这跟 NLP 一致 — NLP BERT 都不 benchmark linear probing，只看 fine-tune。Vision 这边历史上 contrastive learning 跑 linear probing 跑得多，部分原因是 linear probing 跑得快。MAE 这个实验告诉你 linear probing 是个有 bias 的 metric。

---

## 7. Transfer Learning — Scaling 的真正证据

ImageNet fine-tune 数字可以"刷"，但 transfer learning 很难 cheat。Table 4-6 是 MAE 真正证明自己的地方。

### 7.1 COCO object detection (Table 4)

| Method | ViT-B APbox | ViT-L APbox |
|---|---|---|
| supervised | 47.9 | 49.3 |
| MoCo v3 | 47.9 | 49.3 |
| BEiT | 49.8 | 53.3 |
| **MAE** | **50.3** | **53.3** |

ViT-L 上 MAE 比 supervised 高 **4.0 AP**。MoCo v3 跟 supervised 一样，没 gain。BEiT 跟 MAE 持平但 BEiT 要 dVAE tokenizer + 250M image 预训练。

### 7.2 ADE20K semantic segmentation (Table 5)

ViT-L mIoU：supervised 49.9 → MAE 53.6，**3.7 mIoU gain**。这种 pixel-level task 上 MAE 的 gain 比 classification 还大 — 因为 MAE 的 latent 本来就是为 pixel reconstruction 训练的，feature 自然对 dense prediction task 友好。

### 7.3 iNaturalist 2017 (Table 6)

ViT-H448 MAE 83.4% vs 之前 SOTA 75.4% — **8 个点**。iNat 是 fine-grained species classification，类别 5000+。这种任务需要极强的 visual feature，MAE 大模型的 gain 在这里特别明显。

### 7.4 Places205

ViT-H448 66.8% vs 之前用 1B image pretrain 的 66.0%。MAE 用 1M image 就 beat 了 1000× 数据量的 supervised pretraining。这是 scaling efficiency 的强证据。

### 7.5 Robustness (Table 13)

IN-Adversarial 上 ViT-H448 MAE 76.7% vs supervised 33.1% — 这个 gap 是 43 个点。IN-Sketch、IN-Rendition 都有 10+ 个点的 gap。说明 MAE 学到的 feature 是更 general 的 visual concept，对 distribution shift 极其 robust。

---

## 8. 跟同期工作的关系 — 我自己的理解

### 8.1 MAE vs SimMIM

SimMIM（Microsoft，Xie et al.，2021 年 11 月同期）也是 masked image modeling + pixel MSE，但 design choices 不同：

- SimMIM encoder 吃 full sequence（包括 [M] token），mask ratio 50% 左右
- SimMIM decoder 浅但宽（2 block，dim 跟 encoder 一样）

MAE 的 asymmetric 设计让 encoder 只看 25% patches，从而 push mask ratio 到 75%，speed 和 accuracy 双赢。这两篇同期出来，互相印证了 masked pixel reconstruction 在 vision 是 viable path，但 MAE 的 engineering 更 elegant。

### 8.2 MAE vs BEiT

BEiT 的 target 是 dVAE token index，本质上把 image "语义化"成 visual word 然后 predict — 完全 BERT analogy。但 MAE Table 7 证明 normalized pixel 跟 dVAE token 在 transfer learning 上统计等价。

Kaiming 的 implicit message：你不需要"语义化" reconstruction target，**architecture + high mask ratio** 才是 key。Pixel MSE 这种最简单的 loss 就够。这跟 NLP BERT 选 token index 当 target 的原因一样 — token 是 NLP 的 native granularity；pixel 是 image 的 native granularity。

### 8.3 MAE vs DINO/MoCo v3

Contrastive learning 的哲学：构造 positive/negative pairs，让 model 学 invariant representation。问题是要 heavy augmentation、要 negative pairs、要 EMA、要 predictor。

MAE 的哲学：直接 reconstruct input，让 model 学 generative model of image。No negative pairs，no EMA，no predictor，augmentation 都可以不要。

这两个 paradigm 在 ViT-B 上 performance 接近，但在 ViT-H 上 MAE 显然更 scalable。我猜原因是 contrastive learning 的 invariance objective 在大 model 上容易 collapse，而 reconstruction objective 在大 model 上越来越强 — 因为大 model capacity 高，能学更复杂的 natural image manifold。

### 8.4 MAE 之后的发展

MAE 之后一系列工作 follow 这个 paradigm：

- **MAE-st** (Li et al. 2022) — cross-image masked modeling
- **GreenMIM** — sparse attention 加速 decoder
- **MVP** (Wei et al.) — 多模态版的 MAE
- **SAT** — scale-adaptive masking
- **MAE v2 / CAE v2** — combined with contrastive
- **iBOT** — combining MAE with DINO
- **data2vec** (Baevski et al.) — generalized to audio/vision/text
- **BEiT v2** — also moved toward CLIP-style semantic target

DINOv2（2023）实际上是 DINO + iBOT（iBOT 是 BEiT + MAE hybrid），最后证明 masked image modeling + contrastive 联合是最强组合。MAE 在这条线上是 foundational paper。

---

## 9. 一些 engineering 细节值得注意

### 9.1 Shuffle/unshuffle 的 elegance

整篇 paper 最 elegant 的实现细节是 shuffle-based masking。你看 Section 3 末尾"Simple implementation"那段：

```
1. 给每个 patch 生成 token (linear proj + pos emb)
2. Random shuffle 这个 196-list
3. 取前 49 个，丢掉后 147 个
4. Encoder 处理 49 个
5. Append 147 个 [M] token
6. Unshuffle（inverse of step 2）回到原始空间顺序
7. Decoder 处理 196 个
```

这个 trick 最大的好处是**不需要任何 sparse operation**。在 TPU/GPU 上 dense operation 永远比 sparse 快。Shuffle 是 $O(N)$ 的，几乎零 overhead。

### 9.2 Dummy class token

ViT 原版有 class token（[CLS]）。MAE pretraining 时 encoder 输入是 49 个 visible patch token + 1 个 dummy class token = 50 个。这个 dummy token 在 fine-tune 时作为分类 head 的 input。

Ablation 说 MAE 其实也可以不要 class token，直接 average pooling 也行。但为了跟 ViT 原 setup 兼容就保留了。

### 9.3 Linear probing 的 BatchNorm trick

Appendix A.1 提到 linear probing 时加一个 affine=False 的 BatchNorm 在 encoder output 和 linear classifier 之间。这是个 calibration trick — 让不同 ablation variant 的 feature magnitude normalize 到同一 scale，可以用同一套 lr 不用单独 search。

BatchNorm affine=False 等价于一个 reparameterized linear classifier，不破坏 linear probing 的 linear 性质。这种细节其实挺重要 — 没 calibrate 的话不同 variant 比较 linear probing 会有偏差。

### 9.4 Supervised ViT from scratch 的 recipe

Appendix A.2 给了一个能 train 起 ViT-L/H from scratch 的 recipe：weight decay 0.3、batch 4096、warmup 20 epoch、EMA 0.9999、β2=0.95、cosine lr。这个 recipe 把 ViT-L 从 76.5%（原 ViT paper）拉到 82.6%。这是副产品但很 useful — 之前没人是能把 ViT-L supervised 训到 82%+ 的。

### 9.5 Fine-tune 时的 layer-wise lr decay

Table 9 提到 layer-wise lr decay 0.75。意思是越靠前的 layer lr 越小：$lr_i = lr_{base} \cdot 0.75^{(L-i)}$，其中 $i$ 是 layer index，$L$ 是总层数。这是 NLP BERT fine-tune 的 standard trick，让底层 feature 不被破坏太多。

---

## 10. My personal takeaways

### 10.1 Simple thing works if you nail the details

MAE 没有任何"新东西"。Masked autoencoder 1990s 就有了（DAE），masked image modeling 2016 年 Context Encoder 就做了，ViT 2020 年论文里也试过 masked patch prediction。MAE 之所以 work 是因为把每个 design choice 都 nail 到了：

- Asymmetric encoder-decoder
- High mask ratio 75%
- Encoder skip [M] token
- Moderate depth decoder
- Pixel MSE target (with per-patch normalization)
- No color jitter
- Random sampling

任何一个 component 调差了，performance 就掉很多。这是典型的"simple but right"的工程胜利。

### 10.2 Information redundancy 这个 framing 太漂亮

Kaiming 用"image information density 低、所以需要高 mask ratio"这个 framing 解释 why vision 跟 NLP 不同。这个 framing 不仅是 MAE 的 motivation，也 generalize 到其他 vision self-supervised。比如 video temporal redundancy 极高 → 需要更 aggressive masking；multimodal 数据 redundancy 在哪 — cross-modality? 这些都是 follow-up work 探索的方向。

### 10.3 Asymmetric 是 efficiency scaling 的 general pattern

你回顾一下，NLP 那边 GPT 也是 asymmetric — decoder-only，inference 时只看 prefix。Encoder-decoder transformer 也是 asymmetric。MAE 把这个 pattern 引入 vision pretraining — encoder 处理 informative 部分，decoder 处理 "luxury" 部分（reconstruction）。这个 pattern 后来被很多 work 复用：MaskFeat、VideoMAE、MAE-st 等。

### 10.4 为什么不 collapse 是个 deep question

理论上 75% mask + pixel MSE reconstruction 是个 ill-posed problem — 同样的 visible patches 可以对应无数种 plausible missing patches。为什么模型不 collapse 到 average color？

我的猜测：natural image manifold 是高度 structured 的，模型在 1M ImageNet 上训，学到的 prior 极强。Visible 25% patches 提供的 context 足以 disambiguate 大部分 missing 区域。模型其实在学 natural image manifold 的 generative model，但用 encoder（discriminative-style）来 parameterize。这跟 VAE/diffusion 有 conceptual overlap，但用 different infrastructure 实现。

这个方向其实跟世界 model / Yann LeCun 的 JEPA 系列有联系。LeCun 批评 generative model 浪费 capacity 在重建 pixel detail 上，提倡 predict in latent space。MAE 是 pixel-space reconstruction 的 extreme case，但 empirically 它 work。JEPA 后来 follow up 这个 critique 在 latent space 做 prediction（I-JEPA, V-JEPA）。这是 vision self-supervised 的另一条 line，值得追踪。

### 10.5 Linear probing 作为 metric 是有 bias 的

Figure 9 那张 partial fine-tuning 图我反复看了很多次。它的 implication 是：linear probing 偏好 contrastive learning，但 linear separability 跟 representation quality 不是一回事。MAE representation 是 non-linear 但 informative 的，只要给一个 non-linear head（哪怕一个 MLP）就能 unlock。

这暗示 vision community 之前几年对 linear probing 的迷信是 misdirected。NLP 早就不用 linear probing benchmark BERT 了。MAE 之后 vision 也开始 more关注 fine-tune 和 transfer learning。

---

## 11. 如果你想跑 MAE

代码在 https://github.com/facebookresearch/mae，README 很清晰。

几个我踩过的坑：

1. **Pre-training memory**：ViT-H batch 4096 需要 128 TPU-v3 或 64 A100。如果你没这资源，先 reproduce ViT-B 在单 8x A100 上，大概 1-2 天。
2. **Mask ratio 别改**：75% 是 sweet spot，别觉得 50% 也行（Figure 5 显示 50% linear probe 掉 5+ 个点）。
3. **Decoder depth 别减**：默认 8 block。减到 1 block fine-tune 还行但 linear probe 砸。
4. **Color jitter 别加**：跟 contrastive learning 不同，MAE 加 color jitter 反而掉点。
5. **Fine-tune 时 layer-wise lr decay**：0.75 是 ablation 出来的 sweet spot。

---

## 12. Reference & Further Reading

- MAE 原论文: https://arxiv.org/abs/2111.06377
- 官方代码: https://github.com/facebookresearch/mae
- Kaiming He 个人页: https://kaiminghe.github.io/
- Ross Girshick CVPR 2022 talk: https://www.youtube.com/watch?v=wIPL0w-O6Sk
- Yannic Kilcher MAE 解读: https://www.youtube.com/watch?v=wIPL0w-O6Sk
- SimMIM (同期工作): https://arxiv.org/abs/2111.12798
- BEiT: https://arxiv.org/abs/2106.08254
- iBOT (MAE + DINO hybrid): https://arxiv.org/abs/2111.07832
- DINOv2 (iBOT-based, follow-up of MAE lineage): https://arxiv.org/abs/2304.07193
- I-JEPA (LeCun's latent-space follow-up): https://arxiv.org/abs/2301.08243
- data2vec (generalization to multimodal): https://arxiv.org/abs/2202.03555
- ViT 原始论文: https://arxiv.org/abs/2010.11929
- BERT 原始论文: https://arxiv.org/abs/1810.04805
- MaskFeat (MAE for video): https://arxiv.org/abs/2112.09133
- VideoMAE: https://arxiv.org/abs/2203.12602
- A survey on masked image modeling: https://arxiv.org/abs/2205.11711

---

## 13. One more thing — 如果让我重新发现 MAE

Hindsight 20/20，MAE 在 2021 年"被发明"几乎是 inevitable 的。前置条件都满足了：

- ViT 2020 年提出来了，transformer 可以处理 image patch
- BERT 2018 年证明 masked prediction 在 NLP work
- BEiT 2021 年初证明 masked image modeling 在 vision 也 work
- TPU/A100 让 ViT-H 训练成为可能

但只有 Kaiming 团队把 asymmetric + high mask ratio + skip [M] 这几个 engineering choice 组合对了。SimMIM 同期出来但 design 没这么 clean。这种"simple but right"的胜利，是工程品味和科学直觉的结合。

如果你自己想 catch 下一个 MAE 这种工作，方法上可以问：
1. 哪些 established wisdom 实际上是 overfit to current architecture?
2. 哪些 "obvious" design choices 实际上 wrong, but no one tested?
3. Different modality 之间（vision/language/audio/video）的 asymmetry 能不能 exploit?

MAE 就是这三问的完美 sample。

---

希望这版"人话讲"的版本帮你在 mental model 里把 MAE 这篇 paper 完整 install 进去。核心 take-away 不是某个公式或某个 trick，而是 Kaiming 那种"为什么 NLP work 而 vision 不 work — 好，让我一个一个把 gap 填上"的 systematic reasoning 方式。

---

# Masked Autoencoders Are Scalable Vision Learners 深度解读

## 1. 核心动机：为什么 vision 的 masked autoencoding 之前没"起飞"

Kaiming He 等人从三个角度回答了 vision 与 language 在 masked prediction 上的本质差异，这些 insight 是整个 MAE 设计的源头。

### 1.1 Architecture 之前不匹配
在 ViT 出现之前，convolution-based 网络（如 ResNet）dominant vision。Convolution 在 regular grid 上操作，把 mask token [M] 或者 positional embedding 这种"指示器"塞进 conv pipeline 是 awkward 的。iGPT 早期尝试在 pixel sequence 上做 transformer，但效率受限。ViT 的出现让 transformer-based masking 变得 natural，这个 architectural gap 消除后，masked autoencoding 在 vision 上重新可行。

### 1.2 Information density 差异 — 这是最关键的 insight
Language 是 human-generated signal，information-dense 且 highly semantic。BERT 只 mask 15% token，因为即使一小部分缺失也能强迫模型理解 syntax 和 semantics。

Image 是 natural signal，存在严重的 **spatial redundancy**。一个 missing patch 完全可以通过 neighboring patches 的 low-level interpolation 推出来，不需要 holistic understanding。如果你只 mask 15% 的 patches，模型走 shortcut path，学不到 useful representation。

**MAE 的应对策略**：把 masking ratio 拉到 75%。这样 information redundancy 大幅降低，模型被迫做"holistic understanding beyond low-level image statistics"。你可以在 Figure 2-4 看到模型其实在做"语义补全"——比如把一只狗的轮廓从只有 25% 的可见 patch 推出来。

### 1.3 Decoder 角色不同
NLP 中 BERT 的 decoder 预测 missing words，words 本身是 semantic entities，所以 decoder 可以 trivial（一个 MLP 就够）。

Vision 中 decoder 重建的是 **pixels**，pixel 是 low-level，不是 semantic。如果 decoder 太弱，encoder 的 latent representation 就会被"拖向"pixel-level，不利于 recognition 任务。所以 MAE 需要 **一个足够 deep 的 decoder**，让它承担 pixel reconstruction 的 specialization，从而"解放"encoder 的 latent space，让它更抽象、更 semantic。

---

## 2. 架构详解 — Asymmetric Encoder-Decoder

### 2.1 Forward pass 的完整 pipeline

输入图像 $x \in \mathbb{R}^{H \times W \times 3}$，patch size $p = 16$。总 patch 数为 $N = (H/p) \times (W/p) = 196$（224×224 input 时）。

**Step 1 — Patch embedding**：
$$x \to \text{patches} \to \text{linear projection} \to z_0 \in \mathbb{R}^{N \times d_e}$$
其中 $d_e$ 是 encoder embedding dim，ViT-L 是 1024。加 positional embedding $E_{pos}$：
$$z_0 = E \cdot \text{patch} + E_{pos}$$

**Step 2 — Random masking via shuffle**：
对所有 $N$ 个 token 做随机 shuffle，保留前 $N_v = N \times (1 - r)$ 个，丢弃剩余 $N \times r$（$r = 0.75$）。这里 $r$ 是 masking ratio。**关键**：masked patches 直接从 list 里 remove，**不**用 mask token 填充。这样 encoder 只处理 $N_v = 49$ 个 tokens（对 224×224 输入）。

**Step 3 — Encoder forward**：ViT transformer blocks 处理 $z_0^{visible} \in \mathbb{R}^{N_v \times d_e}$：
$$z_e = \text{TransformerEncoder}(z_0^{visible}) \in \mathbb{R}^{N_v \times d_e}$$

**Step 4 — Append mask tokens**：把 $N_m = N \times r = 147$ 个 shared learnable mask token $[\text{M}] \in \mathbb{R}^{d_e}$ append 到 $z_e$ 后面，然后 **unshuffle** 回原始位置，加上 positional embedding。得到完整的 token sequence 送入 decoder：
$$z_d^{in} = \text{unshuffle}(\text{concat}(z_e, [\text{M}]^{N_m})) + E_{pos}^{dec}$$

**Step 5 — Decoder forward**：一个 lightweight transformer 处理全部 $N$ 个 tokens：
$$z_d^{out} = \text{TransformerDecoder}(z_d^{in}) \in \mathbb{R}^{N \times d_d}$$
MAE 默认 $d_d = 512$，depth = 8 blocks（vs encoder ViT-L 是 24 blocks、$d_e = 1024$）。

**Step 6 — Linear projection to pixel space**：decoder 最后一层是一个 linear projection，输出 channel 数 = patch 内 pixel 数 = $p \times p \times 3 = 768$。
$$\hat{x}_i = W_{pix} z_{d,i}^{out} \in \mathbb{R}^{p^2 \cdot 3}$$

### 2.2 Loss function

只对 masked patches 计算 MSE：
$$\mathcal{L} = \frac{1}{N_m} \sum_{i \in \Omega_m} \| \hat{x}_i - x_i \|_2^2$$

其中 $\Omega_m$ 是 masked patch index 的集合，$x_i$ 是第 $i$ 个 patch 的原始 pixel values。

**Normalized pixel variant**（在 ViT-H/L 主结果中用）：对每个 patch 计算其 mean $\mu_i$ 和 std $\sigma_i$，把 $x_i$ normalize 成 $\bar{x}_i = (x_i - \mu_i) / \sigma_i$，loss 改为：
$$\mathcal{L}_{norm} = \frac{1}{N_m} \sum_{i \in \Omega_m} \| \hat{x}_i - \bar{x}_i \|_2^2$$

这个 trick 在 Table 1d 上给 fine-tuning 带来 0.5% 的提升。Intuition：local contrast 增强了，避免了模型只学"全 patch average color"这种 trivial solution。

### 2.3 Asymmetric 设计的 efficiency 数学

ViT 的 self-attention 复杂度是 $O(N^2 \cdot d)$。完整 sequence 处理 $N=196$ 个 token，但 MAE encoder 只处理 $N_v = 49$ 个：
$$\text{FLOPs ratio}_{encoder} = \left(\frac{49}{196}\right)^2 = \frac{1}{16}$$

对于 ViT-L encoder，每个 token FLOPs 大约是完整的 $1/16$。加上 decoder 处理全部 token 但很 shallow/narrow，整体加速 ~3×。对 ViT-H，wall-clock 加速能达到 3.5–4.1×（Table 2）。

### 2.4 关键设计权衡：encoder 不用 mask token

如果 encoder 也吃 mask token，会发生两件坏事：
1. **Pretrain-test gap**：pretraining 时 encoder 输入有大量 [M] token，但 deploy 时（fine-tune / downstream）输入是完整 clean image，没有 [M]。这个 distribution shift 会损害 transfer accuracy。Table 1c 显示 linear probing 掉了 14%（73.5% → 59.6%）。
2. **计算浪费**：encoder 处理 $N$ 个 token 而不是 $N_v$ 个，FLOPs 是 3.3×（Table 1c）。

把 [M] 推迟到 decoder 才引入，等于让 encoder 始终"看到真实数据"，并且大幅加速。

---

## 3. Masking Ratio — 最神秘的 hyperparameter

### 3.1 实验数据（Figure 5）

| Masking ratio | Fine-tune acc | Linear probe acc |
|---|---|---|
| 15% (BERT-like) | ~82.5% | ~55% |
| 40% | ~84.5% | ~67% |
| 75% (sweet spot) | **84.9%** | **73.5%** |
| 80% | ~84.7% | ~73% |
| 90% | ~83% | ~60% |

两个观察：
- **Linear probing 对 ratio 非常敏感**：从 15% → 75%，accuracy 涨了 ~20 个点。说明 latent representation 的 quality 强依赖 task difficulty。
- **Fine-tuning 相对 robust**：40%-80% 区间都差不多。因为 fine-tune 时 encoder 后几层可以重新 adjust 到 recognition task。

### 3.2 Intuition build
- 低 ratio（如 15%）：task 太 easy，模型走"局部 interpolation"shortcut。学到的 feature 是 low-level 的。
- 高 ratio（如 90%）：task 太 hard，信息几乎全部丢失，模型 collapse 到只学 average color。
- 75% 的 sweet spot：visible patches 仍然提供 sufficient context（49 个 patches 大约是图像的 1/4），但 missing 部分足够大，必须做 holistic reasoning 才能 reconstruct。

### 3.3 与 NLP 的对比
BERT 的 15% 在 language 中合适，因为 language 是 dense signal。Vision 是 redundant signal，需要更高 ratio 才能"逼迫"模型学 semantic。

---

## 4. Decoder 设计的 ablation

### 4.1 Decoder depth (Table 1a)

| Blocks | ft | lin |
|---|---|---|
| 1 | 84.8 | 65.5 |
| 2 | 84.9 | 70.0 |
| 4 | 84.9 | 71.9 |
| 8 | **84.9** | **73.5** |
| 12 | 84.4 | 73.3 |

**Intuition**：linear probing 用 encoder 最后一层 output。如果 decoder 太浅，pixel reconstruction 任务会"压"encoder 的最后几层去做 low-level pixel prediction，这些 layer 就不 semantic 了。深 decoder 可以"承担" reconstruction specialization，让 encoder 顶层保持 abstract。

但 fine-tuning 时，encoder 最后几层会重新 tune 适应 recognition，所以 decoder depth 影响小。

### 4.2 Decoder width (Table 1b)

512-d 已经够用，1024-d 没好处。所以 decoder 可以比 encoder 窄很多（ViT-L 是 1024-d）。

### 4.3 Total decoder FLOPs
Decoder 8 blocks、512-d，处理 196 tokens。FLOPs per token 是 ViT-L 的 ~9%。但 encoder 只处理 49 tokens。整体 decoder 计算只占很小一部分，所以"decoder 处理全 token"的代价被 amortize 了。

---

## 5. Reconstruction Target 讨论 — Pixels vs Tokens

Table 1d 的对比：

| Target | ft | lin |
|---|---|---|
| pixel (no norm) | 84.9 | 73.5 |
| **pixel (w/ norm)** | **85.4** | **73.9** |
| PCA (96 coeffs) | 84.6 | 72.3 |
| dVAE token (BEiT style) | 85.3 | 71.6 |

Table 7 还做了 transfer learning 比较，normalized pixels 跟 dVAE tokens 统计上无差别。

**Key takeaway**：MAE 不需要 dVAE 这种额外 tokenizer 预训练阶段（BEiT 依赖 250M images 训 dVAE）。Pixel MSE 足够好，更简单。

为什么 PCA 反而差？因为 PCA 丢掉 high-frequency component，而 high-frequency 在 reconstruction 任务里反而是 useful signal——强迫模型学到 detailed structure。

---

## 6. Data Augmentation 的"反常识"发现

Table 1e：

| Aug | ft | lin |
|---|---|---|
| none (center crop, no flip) | 84.0 | 65.7 |
| crop, fixed size | 84.7 | 73.1 |
| crop, rand size | 84.9 | 73.5 |
| crop + color jit | 84.3 | 71.9 |

**最 striking 的发现**：MAE 在 *no augmentation* 下都能 work！这跟 contrastive learning（MoCo, SimCLR, BYOL）完全相反。SimCLR 没 augmentation 直接 collapse，BYOL cropping-only 掉 13%。

**Intuition**：Random masking 本身就是一种"instance-level augmentation"。每次 iteration mask 不同，相当于同一张 image 产生无数 random "views"。Masking 既 create task difficulty，也 regularize training。

Color jitter 反而 hurt，可能是因为 color jitter 改变了 pixel distribution，而 reconstruction target 是 pixel values，distribution shift 让 task harder 但不更 informative。

---

## 7. Mask Sampling Strategy (Table 1f)

| Strategy | ratio | ft | lin |
|---|---|---|---|
| **random** | 75 | 84.9 | 73.5 |
| block | 50 | 83.9 | 72.3 |
| block | 75 | 82.8 | 63.9 |
| grid | 75 | 84.0 | 66.0 |

- **Block-wise** (BEiT 用)：在 75% 时 collapse。Intuition：大块连续 missing 意味着局部 context 完全丢失，重建极难，loss 高，reconstruction 模糊，representation 质量差。
- **Grid-wise**：保留规律性 patches，task 容易，模型走 shortcut。Loss 低，但 representation 质量低。
- **Random**：在 missing 区域周围仍有部分 visible context，但分布不规则，模型必须做真正的 holistic reasoning。同时 random sampling 让 higher masking ratio 可行，进而 enable encoder efficiency gain。

---

## 8. Training Schedule 与 Scaling Behavior

Figure 7：1600 epochs 时 linear probing 仍未饱和，跟 MoCo v3 在 300 epochs 就饱和完全不同。

**关键比较**：MAE encoder 每个 epoch 只看 25% patches（vs MoCo v3 的 200% via two-crop）。所以 1600 epochs 实际算下来，每个 patch 被看的次数相当于 MoCo v3 的 200 epochs 左右。Wall-clock 上 ViT-L 1600 epochs 31h，MoCo v3 300 epochs 36h（同硬件）——MAE 更快！

### 8.1 Scaling 表现 (Table 3)

| Method | ViT-B | ViT-L | ViT-H | ViT-H448 |
|---|---|---|---|---|
| scratch | 82.3 | 82.6 | 83.1 | - |
| DINO | 82.8 | - | - | - |
| MoCo v3 | 83.2 | 84.1 | - | - |
| BEiT | 83.2 | 85.2 | - | - |
| **MAE** | **83.6** | **85.9** | **86.9** | **87.8** |

MAE 是唯一一个 stretch 到 ViT-H 还持续提升的方法。DINO 和 MoCo v3 在 ViT-L 之后就 stop。BEiT 没跑 ViT-H。MAE 在 ViT-H448 上 87.8% 是当时 IN1K-only 的 SOTA。

### 8.2 vs supervised pretraining (Figure 8)

ViT-L supervised from scratch: 82.6% (their improved recipe)
ViT-L MAE: 85.9%

ViT-H supervised from scratch: 83.1%
ViT-H MAE: 86.9%

Capacity 越大，MAE gain 越大。这正是 NLP 中 self-supervised 的 scaling 行为——MAE 第一次在 vision 中复现了这个 trend。

---

## 9. Partial Fine-tuning — 一个被忽视的 protocol

Figure 9：ViT-L 上 partial fine-tuning 不同 block 数：

| Tuned blocks | MAE | MoCo v3 |
|---|---|---|
| 0 (linear probe) | 73.5 | 76.7 |
| 1 block | 81.0 | 79.5 |
| 4 blocks | 84.1 | 81.5 |
| last MLP sub-block only | 79.1 | - |
| 24 (full) | 84.9 | 84.1 |

**Insight**：
- Linear probing 上 MoCo v3 赢 MAE（76.7 vs 73.5）。
- 一旦 tune 一个 block，MAE 立刻反超，gap 越来越大。
- 即使只 tune 最后一个 MLP sub-block（本质是 MLP head），MAE 已经 79.1% vs linear probe 73.5%。

**Interpretation**：MAE 的 representation 不是 linearly separable 但 **是高度 nonlinearly informative**。Linear probing metric 偏好 contrastive learning，但 linear separability 不是 representation 质量的全部衡量。这呼应了 NLP 中 linear evaluation 不是 standard benchmark 的事实。

---

## 10. Transfer Learning — 真正的 scaling 证据

### 10.1 COCO (Table 4)

| Method | ViT-B box AP | ViT-L box AP |
|---|---|---|
| supervised | 47.9 | 49.3 |
| MoCo v3 | 47.9 | 49.3 |
| BEiT | 49.8 | 53.3 |
| **MAE** | **50.3** | **53.3** |

ViT-L 上 MAE 比 supervised 高 4.0 个 AP，非常显著。

### 10.2 ADE20K semantic segmentation (Table 5)

ViT-L mIoU：supervised 49.9 → MAE 53.6，**3.7 mIoU gain**。

### 10.3 iNaturalist & Places (Table 6)

iNat 2017：ViT-H448 83.4% vs 之前 SOTA 75.4%。这种细粒度分类，MAE 大模型碾压。Places205 上 ViT-H448 66.8% vs 之前用 1B image pretrain 的 66.0%。

**结论**：MAE 让 vision 第一次用 1M images 就能 match 甚至 beat 之前用 1B+ image 的 supervised pretraining。这正是 scaling law 的胜利。

---

## 11. Robustness (Table 13)

IN-Adversarial：ViT-H448 MAE 76.7% vs supervised 33.1%。这个 gap 巨大（~43 个点）。

IN-Corruption、IN-Rendition、IN-Sketch 都有显著 robustness gain。说明 MAE 学到的 feature 比 supervised 更 general，对 distribution shift 更 robust。

---

## 12. 跟 BERT 的精确类比

| 维度 | BERT | MAE |
|---|---|---|
| Mask ratio | 15% | 75% |
| Mask unit | token (word/subword) | patch |
| Reconstruct | word index (cross-entropy) | pixel values (MSE) |
| Decoder | trivial MLP | moderate depth transformer |
| Tokenize | already discrete | skip, use raw pixel |
| Pretrain-test gap | small (15% masked) | addressed by asymmetric design |

Kaiming 在 discussion 里点明：image 不像 language 有 semantic decomposition into "visual words"。MAE 选择 reconstruct random patches（多数 patch 不对应 semantic segment）和 reconstruct pixels（不是 semantic entities），但实验上 latent representation 学到了 semantics。这是一个 emergent behavior：通过 reconstruct low-level pixels，模型被迫在 latent space 中"发明"semantic abstractions。

---

## 13. 我的几个深层思考 / Intuition

### 13.1 为什么高 mask ratio 行得通，而不是 collapse
理论上 75% missing 后 reconstruction 是 ill-posed。但 image 的 manifold 是高度 structured 的（natural image prior），visible 25% patch 提供的 context 足以 disambiguate 大部分 missing 区域。模型其实是在学 natural image manifold 的 generative model——这一点上 MAE 跟 diffusion / VAE 有 conceptual overlap，但用 discriminative-style encoder 实现。

### 13.2 Asymmetric 设计是 efficiency 的关键 trick
如果不 asymmetric（encoder 也吃 mask token），计算开销是 16× for encoder attention。这意味着 ViT-H 在合理时间内根本 train 不动。Asymmetric 把"全 token 处理"推到 decoder，而 decoder 处理全 token 的代价被"depth=8, width=512"压到 encoder FLOPs 的 9% 以下。这是 elegant 的 engineering。

### 13.3 Pixel MSE 的 simplicity 是优势
不需要 dVAE tokenizer（额外 250M image pretrain stage），不需要 discrete codebook，不需要 cross-entropy loss。整个 pipeline 是 end-to-end 一次 pretrain，pixel MSE 直接可微。这种 simplicity 让 scaling 容易。

### 13.4 跟 SimMIM 的对比
SimMIM（Microsoft 同期工作）也是 pixel MSE + mask token in encoder，但 mask ratio 50% 左右更优，且 encoder 处理 full sequence。MAE 通过 asymmetric 设计让 encoder skip mask token，从而 push mask ratio 到 75%，speed 和 accuracy 双赢。

### 13.5 跟 BEiT 的本质区别
BEiT 的 target 是 dVAE token index，本质是把 reconstruction 任务"语义化"——类似把 pixel image 翻译成"visual word"，然后预测 visual word。但 MAE 实验证明（Table 7），在 transfer learning 上 normalized pixel 跟 dVAE token 统计等价。所以"语义化 target"不是必需的，**asymmetric architecture + high mask ratio** 才是关键。

---

## 14. Reference 链接

- 原论文 PDF: https://arxiv.org/abs/2111.06377
- 官方代码: https://github.com/facebookresearch/mae
- Kaiming He 项目页: https://kaiminghe.github.io/
- ViT 原始论文: https://arxiv.org/abs/2010.11929
- BEiT 论文: https://arxiv.org/abs/2106.08254
- SimMIM (Xie et al. 同期工作): https://arxiv.org/abs/2111.12798
- BERT 论文: https://arxiv.org/abs/1810.04805
- MoCo v3: https://arxiv.org/abs/2104.02057
- DINO: https://arxiv.org/abs/2104.14294
- iGPT: https://openreview.net/forum?id=XY8lq3TokDB
- Ross Girshick 在 FAIR 的 talk on MAE: https://www.youtube.com/watch?v=6E_dCkVMSVU
- Yannic Kilcher 的 MAE 视频讲解: https://www.youtube.com/watch?v=wIPL0w-O6Sk

---

## 15. 总结一句话的 intuition

**MAE 之所以 work**：通过 (1) 极高 mask ratio 消除 image redundancy 强迫 semantic reasoning，(2) asymmetric encoder-decoder 设计让 encoder 不需要处理 mask token、从而同时获得 efficiency 和 pretrain-deploy consistency，(3) moderate-depth decoder 承担 pixel-level reconstruction specialization 让 encoder latent 保持 abstract，最终在 vision 上复现了 NLP 自监督的 scaling law。

这是把 BERT 的思想"忠实但不同地"移植到 vision 的 best practice——尊重 image 的特殊性（redundancy、low-level target）而非直接照搬。
