---
source_pdf: Sapiens Foundation for Human Vision Model.pdf
paper_sha256: 0db463c4e79106a03166318f38740ad8f4dd67468b4157f8f48a16ff5d135bc9
processed_at: '2026-08-12T02:54:18-07:00'
target_folder: AI美工
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Sapiens — 用人话讲

## 一句话版本

**同样的算力, 与其在一堆杂七杂八的图片上预训练, 不如专门拿人像图片来预训练, 下游做"和人相关的任务"时效果会好一大截。**

这个道理听起来像个废话, 但在 vision 圈子里, 过去几年的共识一直是"general pretrain (ImageNet / DINOv2) 就够用了, 下游 fine-tune 会解决一切"。Sapiens 用一张 Table 7 把这个共识给锤了。

---

## 为什么这件事不 trivial

我 (Karpathy) 自己讲课时老说一句话: **pretraining 的本质是学一个 prior over data manifold**。

你给模型看什么数据, 它脑子里就建立什么世界的模型。

- 给它看 ImageNet, 它学到的是"猫狗汽车花草"
- 给它看 Instagram, 它学到的是"美食日落自拍"
- 给它看 3 亿张人像, 它学到的是 **人体结构, 骨骼, 关节, 五官, 表情, 衣服皱褶, 头发, 手指**

下游你要做 pose estimation / body parsing / depth / normal 这种 **专门关于人** 的任务, 哪个 prior 更有用? 答案很明显。

但 vision 圈子过去十年, 几乎所有人都是 "先在 ImageNet 上 pretrain, 再在 task-specific data 上 fine-tune"。这个路径依赖太强了, 以至于大家没认真想过: **如果你的下游 domain 固定, 为什么 pretrain 不也固定 domain?**

LLM 圈子反而早想明白了 — Code Llama, BioGPT, BloombergGPT, FinanceGPT, 都是这个套路。vision 里 Sapiens 把这个事给做实了, 而且做得相当 hardcore。

🔗 Code Llama: https://arxiv.org/abs/2308.12950  
🔗 BloombergGPT: https://arxiv.org/abs/2303.17530

---

## Sapiens 到底干了啥

### Step 1: 收数据

从 10 亿张 in-the-wild 图片里, 过滤出 3 亿张 "人像" 图片。过滤标准:
- 有 watermark / 是插画 / 有奇怪东西的 → 丢掉
- 用 person detector (Detectron2) 检测, score > 0.9
- person bbox 至少 300 像素, 确保 high-res

剩下 3 亿张, 其中 2.48 亿张是多人的 (Fig. 2)。这个 "多人" 很重要, 下游要 generalize 到 crowded scene。

### Step 2: MAE 预训练

MAE (Masked Autoencoder) 是 He Kaiming 2022 年的工作。简单说:
- 把图片切成 patch (16×16)
- 随机 mask 掉 75%
- 剩下 25% 过 encoder
- decoder 试图 reconstruct 被 mask 的部分

为什么选 MAE 而不是 DINOv2 / contrastive? **因为 MAE 只要一次 forward pass, DINOv2 要 teacher-student 两条路**。同等算力下 MAE 能跑更多图。Sapiens 要在 1024 分辨率上跑 2B 参数, 这个 throughput 优势是决定性的。

这里有个关键数字: **native resolution 1024×1024**, patch 16, 所以一张图被切成 64×64 = 4096 个 token。标准 ViT 在 224 下只有 14×14 = 196 token。**Sapiens 比 ViT 细粒度 16 倍**。

这对 dense prediction (keypoint / normal / depth) 是 critical 的 — 你不能在 14×14 的 grid 上预测 308 个 keypoint 的位置。

🔗 MAE 原文: https://arxiv.org/abs/2111.06377

### Step 3: 四个 fine-tune task

同一个 encoder, 接四个不同的 lightweight decoder, 各做一个任务:

| Task | Decoder 输出 | Loss | 训练数据 |
|---|---|---|---|
| 2D Pose (308 kps) | $H \times W \times K$ heatmaps | MSE | 1M studio images |
| Body Seg (28 class) | $H \times W \times C$ probs | Weighted CE | 100K studio images |
| Depth | $H \times W \times 1$ | SI log-MSE | 500K synthetic |
| Normal | $H \times W \times 3$ | L1 + (1-cos) | 500K synthetic |

注意 **depth 和 normal 完全用 synthetic data (600 个 RenderPeople 扫描渲染出来的)**, 没用任何 real depth label。这是 paper 最 bold 的 claim 之一: **纯 synthetic supervision + human pretrain → wild real generalization**。

---

## 为什么 "domain pretrain > general pretrain" 这件事是真的

Table 7 是整篇 paper 的灵魂:

```
                  Pose    Seg     Depth   Normal
Random init       30.2   40.3    0.720   35.4
General-100M      35.7   50.1    0.351   27.5
General-300M      37.3   52.8    0.347   26.8
Humans-100M       43.6   61.2    0.316   24.0
Humans-300M       47.0   66.5    0.288   21.8
```

重点对比这两行:
- **General-300M**: 3 亿张杂图
- **Humans-100M**: 1 亿张人像

同样模型 (0.3B), 同样 training schedule, **3 倍少的人像数据, 在全部四个任务上都赢**。

用我的话说: **算力是稀缺资源, 与其花在无关数据上, 不如集中花在 task-relevant 数据上**。3 倍 general data 的信息量, 抵不上 1 倍 human data 的"相关性"。

这个 insight 对所有 vertical domain 都适用: medical imaging, autonomous driving, OCR, satellite, industrial defect。过去大家都在用 ImageNet pretrain, 其实大部分情况下, 用自己 domain 的数据 pretrain 效果会更好 — 只是你得有足够多的 domain data。

🔗 这跟 "data quality > data quantity" 在 LLM instruction tuning 上的发现是同构的: LIMA paper 用 1000 条高质量数据就 tune 出 GPT-4 级对话能力。https://arxiv.org/abs/2305.06600

---

## 高分辨率这件事为什么重要

看 Table 1 的对比:

```
DINOv2:        1B params, 224 resolution, 291 GFLOPs
ViT-6.5B:      6.5B params, 224 resolution, 1657 GFLOPs
AIM-6.5B:      6.5B params, 224 resolution, 1657 GFLOPs
Sapiens-2B:    2B params,  1024 resolution, 8709 GFLOPs
```

Sapiens 参数比 ViT-6.5B 少 3 倍, 但 FLOPs 多 5 倍。**全部算力都花在分辨率上了**。

为什么值得? 你想想:
- 在 224 下, 一张人脸可能就 30×30 像素, 你要预测 243 个 facial keypoint, 根本分辨不开
- 在 1024 下, 同样人脸有 140×140, 每个 keypoint 都能落在不同像素上

**dense prediction 任务的瓶颈是 spatial resolution, 不是 model capacity**。你给 6.5B ViT 在 224 下预测 308 keypoint, 它也没办法 — 因为输入信息本身就不够。

这跟 LLM 里的 context length 类似: 你不能用 2K context 处理 100K 的文档, 不管模型多大。

---

## 四个任务的公式, 用人话讲

### Pose (公式最简单)

输入图片 $I$, 输出 $K$ 张 heatmap, 每个 heatmap 对应一个 keypoint。Loss 是 MSE:
$$\mathcal{L} = \text{MSE}(\mathbf{y}, \hat{\mathbf{y}})$$

$\mathbf{y}_{h,w,k}$ 就是 "第 k 个 keypoint 出现在 pixel (h,w) 的概率 (Gaussian 峰值)"。模型预测的 $\hat{\mathbf{y}}$ 要逼近它。没什么花哨的, 就是 regression。

### Segmentation

输出 $C$ 个 class 的 per-pixel probability, Weighted Cross-Entropy。Weighted 是因为有些 class (teeth, tongue) 很 rare, 需要 reweight。

### Depth (最复杂的公式)

设 $\mathbf{d}$ 是 GT depth, $\hat{\mathbf{d}}$ 是预测, 都取 log:
$$\Delta \mathbf{d}_i = \log d_i - \log \hat{d}_i$$

这是 **log-space 残差**, 也就是 "预测 depth 是 GT 的多少倍" 的 log。

然后:
$$\mathcal{L} = \sqrt{\mathbb{E}[\Delta^2] - \frac{1}{2}\mathbb{E}[\Delta]^2}$$

**直觉**: 这个 loss 在同时做两件事:
- $\mathbb{E}[\Delta^2]$: 残差的二阶矩, 压低它 = 让预测在 log space 精确
- $\mathbb{E}[\Delta]^2$: 残差均值的平方, 压低它 = 让预测 scale 对齐 GT

$\frac{1}{2}$ 这个系数是 Sapiens 的选择, 介于纯 scale-invariant (Eigen 原版 $\lambda=1$) 和纯 log-MSE ($\lambda=0$) 之间。因为 synthetic data 的 absolute scale 是已知的, 不需要完全 scale-invariant, 但又想容忍一点 scale shift 来让优化更稳。

🔗 Eigen depth loss 原文: https://arxiv.org/abs/1406.2283

### Normal

输出 3 维向量 (xyz), Loss 是 L1 + (1 - cosine):
$$\mathcal{L} = \|\mathbf{n} - \hat{\mathbf{n}}\|_1 + (1 - \mathbf{n} \cdot \hat{\mathbf{n}})$$

**直觉**:
- L1: per-component 准确, 避免 normal "flip" (比如把朝外的法线预测成朝内)
- $1 - \cos\theta$: angular 误差, 直接对齐 evaluation metric

两者结合 = de-facto 标配。normal 是个 3D unit vector, L2 alone 容易让模型卡在 antipodal 的 local minima (180° 反向), L1 + cosine 能避免。

---

## 结果有多炸

| Task | Metric | Prior SOTA | Sapiens-2B | 提升 |
|---|---|---|---|---|
| Pose (Humans-5K) | mAP | 53.5 (DWPose) | 61.1 | **+7.6** |
| Seg (Humans-2K) | mIoU | 64.1 (DeepLabV3+) | 81.2 | **+17.1** |
| Depth (Hi4D) | RMSE | 0.147 (DepthAnything-L) | 0.114 | **-22.4%** |
| Normal (THuman2) | Mean ° | 25.45 (ECON) | 11.84 | **-53.5%** |

normal 那个 -53.5% 我第一次看到挺震惊的。ECON 用了 4000 个 scans, Sapiens 只用 600 个, **数据少 6 倍, 误差少一半**。这就是 pretrain 的威力。

🔗 ECON: https://arxiv.org/abs/2212.07422  
🔗 Depth Anything: https://arxiv.org/abs/2401.10891

---

## 三个值得玩味的设计选择

### 1. Width over depth

Sapiens-0.3B → 2B 的 scaling: hidden 1024 → 1920 (1.9×), layers 24 → 48 (2×)。但 paper 强调 "prioritize width over depth"。

这个结论来自 LLaMA 1/2 的经验: 在固定参数下, 加宽比加深更 efficient (尤其是在 attention 为主的结构里, 深 model 的 residual path 会让信号衰减)。LLaMA 7B 用 4096 hidden / 32 layers, 而不是 2048 hidden / 64 layers, 就是这个道理。

Sapiens 把这个经验直接搬到 vision。有意思的是 vision 圈子过去更多 "deep and narrow" (ResNet-152, ViT-Huge 32 layers 1280 hidden), LLaMA 这波反向影响 vision design。

🔗 LLaMA 2: https://arxiv.org/abs/2307.09288

### 2. Layer-wise LR decay 0.85

Fine-tune 时候, **靠前的层用更小的 LR, 靠后的层用更大的 LR**。具体: 第 $l$ 层的 LR = base_lr × $0.85^{L-l}$, 其中 L 是总层数。

**直觉**: 靠前的层是 general features (edges, textures), pretrain 已经学好了, 别动太多; 靠后的层是 task-specific, 需要 adapt。

这是 ULMFiT / XLNet 时代 LLM fine-tuning 的老 trick, vision 圈子基本没用过。Sapiens 把它捡起来, 0.85 这个数也是 LLM 里的经验值。Vision fine-tune 一直用 uniform LR, 这其实是个 under-optimized 的地方。

🔗 ULMFiT: https://arxiv.org/abs/1801.06146  
🔗 XLNet: https://arxiv.org/abs/1906.08237

### 3. Synthetic data only (for depth/normal)

Depth 和 normal 完全不用 real label, 用 600 个 RenderPeople 扫描渲染 50 万张 synthetic image 来 fine-tune。

**为什么能 work?** 因为:
1. pretrain encoder 已经懂 "人长什么样"
2. synthetic data 提供 perfect, dense, pixel-aligned GT (real depth sensor有噪声, real normal 几乎没法标)
3. 渲染时 random camera + random HDRI 背景 → 让 synthetic 数据有多样性
4. pretrain prior 弥补 synthetic-real gap

这是个很 Meta (公司) 风格的思路 — Reality Labs 做 Codec Avatar 长期依赖 synthetic + photogrammetry。但把它 scale 到 foundation model 级别, Sapiens 是第一个。

这跟 NVIDIA 的 synthetic data pipeline (Omniverse Replicator) 思路同构。未来 synthetic pretrain + synthetic fine-tune 可能是 dense prediction 任务的 default 路径。

🔗 NVIDIA Omniverse Replicator: https://docs.omniverse.nvidia.com/replicator/

---

## 这篇 paper 真正的 contribution

我觉得 paper abstract 有点 undersell 自己。真正重要的不是 "我们在 4 个 benchmark 上 SOTA", 而是:

1. **验证了 domain-specific pretraining 在 vision 里 work** — 这件事 LLM 圈子知道, vision 圈子一直怀疑。Sapiens 给了 clean evidence。

2. **验证了 high-resolution native pretraining 是 viable** — 1024 native + 2B params + 8709 GFLOPs, 之前没人愿意花这个钱。Sapiens 跑通了, 而且效果显著。

3. **验证了 synthetic-only fine-tune + domain pretrain 能 generalize to real** — 这对 labeling 成本爆炸的 dense prediction 任务是个巨大信号。

4. **把 LLM 时代的几个 trick (width scaling, layer-wise LR, pretrain-finetune 分工) 搬到 vision** — 不是每个都新, 但组合起来在 human vision 上 SOTA, 是个 useful reference architecture。

---

## 我 (Karpathy) 的一些 further thoughts

### A. Sapiens 作为 "human perception tokenizer"

想象一下: 你训一个 Sora-style human video generation model。现在的 conditioning 一般是 CLIP text embedding + reference image。但 Sapiens 的输出 (308 keypoint + 28-class seg + depth + normal) 是 **结构化, 几何精确, human-aware** 的 representation。

如果把 Sapiens 的输出当作 "perception token" 喂给 generative model, 你得到的不是 "看起来像人的视频", 而是 "解剖学正确, 表情一致, 手指不乱飞" 的人。这对 avatar, telepresence, game NPC 是 game-changer。

这跟 Meta 自己的 Codec Avatar 项目其实是一脉相承的 — Codec Avatar 要精确到瞳孔微动, Sapiens 给的是 perception-side 的 infra。

### B. Sapiens → 3D

Paper conclusion 暗示下一步是 3D。可以想象:
- Sapiens-3D: image → SMPL-X 参数 + face blendshape + hand pose
- 或者 image → 3D Gaussian Splatting of a person
- 或者 video → animatable 4D human

这条路上 Sapiens 的 encoder 就是天然的 backbone, 因为它已经理解 2D human 到极致, 3D 只是 "再加一个 head" 的事。

🔗 SMPL-X: https://arxiv.org/abs/1905.03789  
🔗 HUGS (human gaussian splat): https://arxiv.org/abs/2311.17910

### C. Humanoid robot 的 vision system

Figure 02, Tesla Optimus, 1X Neo 这些 humanoid robot 的 vision system, 现在大概是用 general vision model (SAM, CLIP, DINOv2) + task-specific head。

但 humanoid robot 的 task domain 就是人 + 人造环境。如果用 Sapiens-style "human + indoor scene" pretrain, 效果应该比 general DINOv2 强。Sapiens 给的 308 keypoint + body parsing 对 grasp planning, social navigation, imitation learning 都很关键。

🔗 Figure AI: https://www.figure.ai/  
🔗 RT-2 (VLA): https://arxiv.org/abs/2310.08878

### D. 为什么 Meta 不开源 weights

这是个大问题。Sapiens 的 308 keypoint + 243 facial keypoint + depth + normal 组合, 对 deepfake detection 是利器, 对 surveillance 也是利器。Meta Reality Labs 选择只发 paper 不放 weights, 这个 decision 我理解 — 跟 LLaMA 1 当初 "research only" 的逻辑类似, 但 sensitivity 更高。

不过这也意味着 community 没法 build on top。如果 Meta 开源 Sapiens-0.3B, 整个 human vision 圈子会快速迭代, 可能半年内就有 Sapiens-derivative 在各个 vertical (medical, sports, fashion) 出现。

---

## 一个小比喻收尾

Sapiens 的故事其实挺像 LLM 圈子过去三年的微缩版:

- **GPT-2 时代**: general pretrain (ImageNet / WebText) 就够
- **GPT-3 时代**: scale general pretrain 到极致
- **Code Llama / BioGPT 时代**: 等等, 原来 domain pretrain 更香?
- **Sapiens**: vision 圈子终于也走到了这一步

Vision 比 LLM 慢了大概 2 年, 因为 vision 的 pretraining recipe (MAE / DINO) 成熟更晚, 而且 vision 的 "domain" 比 text 更模糊 (text 有明确的 code/medical/biology 分界, vision 的 "human domain" 要靠 detector 去定义)。

但 Sapiens 证明: **vision domain pretraining 一样 work, 而且 scaling law 在 domain-specific data 上同样成立**。这是 vision foundation model 时代的一个新 quadrant 被打开了。

---

## 最最后一句

如果你只记一件事: **同样的算力, 与其看 3 亿张杂图, 不如看 1 亿张人像, 下游做人的任务效果更好**。

这道理其实跟人一样 — 你想成为钢琴家, 3 小时练钢琴比 10 小时看书有用。**focus > volume**, 在 pretraining 里也成立。

---

# Sapiens: 专精于人类视觉的 Foundation Model

## TL;DR

Sapiens 是 Meta Reality Labs 的一族 human-centric foundation models，做四件事: **2D pose**, **body-part segmentation**, **depth**, **surface normal**. 核心论点很直白 — 给定相同 compute budget, 在 **curated human images** 上做 self-supervised pretraining (MAE), 比 general images 上 pretrain 显著更优. 这种效应在 fixed compute 下甚至击败了 "数据越多越好" 的常识 (Table 7: Humans-100M 在所有四个任务上都击败 General-300M). 由此 paper 的名字 Sapiens 暗合 sapiens (拉丁文 "wise"), 也呼应了人类视觉皮层在自身种族脸上 finely tuned 的现象.

🔗 Project page: https://about.meta.com/realitylabs/codecavatars/sapiens

---

## 1. 核心直觉: 为什么 domain-specific pretraining 这么 powerful?

我 (Karpathy) 自己一直在强调, pretraining 的核心是 **learn a good prior over the data manifold**. Sapiens 的关键 observation 是:

在 general images 上预训练的 ViT, 学到的是 "natural image statistics": edges, textures, objects, scenes. 在 human images 上预训练的 ViT, 学到的是 **human anatomy, body parts, faces, hands, clothing, hair, the geometric/topological regularities of humans**. 后者对 downstream human tasks 来说是 **much tighter, more useful prior**.

这与 LLM 里 "domain-adapted pretraining" (Code Llama, BioGPT, FinanceGPT) 的逻辑同构, 但是在 vision 里过去一般直接用 general pretrain (DINOv2, MAE on ImageNet) 然后期望 fine-tune 解决一切. Sapiens 在 Table 7 里给出了非常清晰的 evidence:

| Pretraining Source | #Images | Pose (↑ mAP) | Seg (↑ mIoU) | Depth (↓ RMSE) | Normal (↓ deg) |
|---|---|---|---|---|---|
| Random Init | 0 | 30.2 | 40.3 | 0.720 | 35.4 |
| General-100M | 100M | 35.7 | 50.1 | 0.351 | 27.5 |
| General-300M | 300M | 37.3 | 52.8 | 0.347 | 26.8 |
| **Humans-100M** | **100M** | **43.6** | **61.2** | **0.316** | **24.0** |
| **Humans-300M (full)** | **300M** | **47.0** | **66.5** | **0.288** | **21.8** |

注意 **General-300M vs Humans-100M** 的对比: 同样的 model (Sapiens-0.3B), 同样的 training schedule, **3× 更少的 general data 还是输给了 human data**. 这是 paper 最 strong 的 claim 之一. Intuition: 给定 fixed compute, 不如把每张图都"花在" task-relevant distribution上.

🔗 MAE paper: https://arxiv.org/abs/2111.06377  
🔗 DINOv2: https://arxiv.org/abs/2304.07193  
🔗 AIM (autoregressive scaling): https://arxiv.org/abs/2401.08541

---

## 2. 数据集: Humans-300M 的 curation

数据是从约 **1 billion in-the-wild images** 中过滤而来, 关键的 curation 步骤:

1. 丢弃 watermarks / text / artistic / unnatural
2. 用 off-the-shelf person detector (Detectron2-based, score > 0.9)
3. bounding box 维度 > 300 像素 (确保 high-res)
4. 最终得到 ~300M images
5. 其中 **248M 张包含多个人** (multi-human, Fig. 2)

这个 curation 思路与 LAION-5B / LVD-142M 类似, 但 **过滤条件是 person-centric**: 不光要有 person, 还要 visible 占图像足够大. 这与一般 COCO/ ImageNet 的 "any object anywhere" 哲学完全不同.

Fig. 3 显示了 MAE reconstruction 的 qualitative 结果 — 即使 95% mask ratio, 模型也能 reconstruct 人类 anatomy. 这就是 prior 学得很深的信号.

---

## 3. Architecture: 4 个模型族 + native 1024 resolution

遵循 LLaMA 系列的 scaling philosophy: **width over depth**.

| Model | #Params | FLOPs (T) | Hidden | Layers | Heads | Batch |
|---|---|---|---|---|---|---|
| Sapiens-0.3B | 0.336B | 1.242 | 1024 | 24 | 16 | 98,304 |
| Sapiens-0.6B | 0.664B | 2.583 | 1280 | 32 | 16 | 65,536 |
| Sapiens-1B | 1.169B | 4.647 | 1536 | 40 | 24 | 40,960 |
| Sapiens-2B | 2.163B | 8.709 | 1920 | 48 | 32 | 20,480 |

**关键 tradeoff**: native input resolution 提升到 **1024 × 1024**, patch size 16 → 64×64=4096 tokens per image. 每个 token 占图像面积 0.02%, vs 标准 ViT 在 224 下 patch 16 → 14×14=196 tokens, 每个占 0.4%. **16× 更细粒度的 token**, 这对 fine-grained keypoint/normal/depth 是 critical 的.

但是 computational cost 巨大: Sapiens-2B 在 1024 分辨率下 **8709 GFLOPs**, 比 ViT-6.5B 的 1657 GFLOPs 还多 5×. 总训练: 1.2 trillion tokens, 1024 A100, 18 days. 这是相当 hardcore 的 pretraining run.

🔗 ViT 原文: https://arxiv.org/abs/2010.11929  
🔗 LLaMA 2: https://arxiv.org/abs/2307.09288  
🔗 Scaling ViT to 22B (Dehghani): https://arxiv.org/abs/2306.04560

---

## 4. Pretraining: MAE with curated human data

Self-supervision 选了 MAE 而非 DINO/iBOT/contrastive, 理由 paper 给得很直接:

- **Single-pass inference** (vs contrastive 需要 multiple views / teacher-student)
- **Compute-efficient** for the same throughput
- **Simple to implement**

训练目标就是 reconstruct masked patches (pixel space). Masking ratio 75% (HE 沿用 MAE 默认). Pretraining 1024×1024 square, fine-tune 4:3 ratio (1024×768) — positional embeddings 通过 interpolation 适配.

Pretraining-finetuning 的 **encoder-decoder 分工**:
- **Encoder**: 用 pretrain 权重 init
- **Decoder**: lightweight task head (deconv + conv), **random init**
- End-to-end finetune

**Differential learning rate**: layer-wise LR decay 0.85, weight decay 0.1. 这是 LLM fine-tuning 的标准做法 (XLNet, ULMFiT 系), Sapiens 把它搬到了 vision — 越靠后的层 LR 越大, 越靠前 (general features) LR 越小, 保留 pretraining 知识.

🔗 XLNet (layer-wise LR): https://arxiv.org/abs/1906.08237  
🔗 ULMFiT (discriminative fine-tuning): https://arxiv.org/abs/1801.06146

---

## 5. 四个 downstream tasks, 公式详解

### 5.1 2D Pose Estimation

**Setup**: top-down paradigm (先用 detector 给 bbox, 再做单人 keypoint).

Input $I \in \mathbb{R}^{H \times W \times 3}$, bbox cropped & resized 到 $H \times W$ (4:3). 定义 pose transformer $\mathcal{P}$.

Ground truth heatmaps: $\mathbf{y} \in \mathbb{R}^{H \times W \times K}$, 其中 K = 17 / 133 / **308** (Sapiens 新设计).

预测: $\hat{\mathbf{y}} = \mathcal{P}(\mathbf{I}) \in \mathbb{R}^{H \times W \times K}$.

Loss:
$$\mathcal{L}_{pose} = \text{MSE}(\mathbf{y}, \hat{\mathbf{y}})$$

其中:
- $\mathbf{y}_{h,w,k}$: pixel (h,w) 上第 k 个 keypoint 的 gaussian-heatmap 值
- $\hat{\mathbf{y}}_{h,w,k}$: 模型预测的同位置同 keypoint heatmap 值
- $H, W$: spatial resolution of feature map (典型 256×192 或更高)
- $K$: 17 (COCO), 133 (COCO-WholeBody), 308 (Sapiens)

**308 keypoint 设计**: body + foot + 243 face (眼睛/嘴唇/鼻子/耳朵周围) + hand + surface. 这远远超过传统 68 face landmarks 的 WIDER face 标准, 专门为 capture facial expression nuances. 1M images manually annotated at 4K from indoor multi-view capture.

🔗 ViTPose: https://arxiv.org/abs/2204.09642  
🔗 ViTPose+: https://arxiv.org/abs/2212.04246  
🔗 DWPose (whole-body distillation): https://arxiv.org/abs/2307.15880  
🔗 COCO-WholeBody: https://arxiv.org/abs/2007.11858

**结果** (Table 3, Humans-5K test, flip test):

| Model | Input | Whole-body AP | Whole-body AR |
|---|---|---|---|
| DWPose-l | 384×288 | 53.5 | 60.6 |
| ViTPose+-L | 256×192 | 47.8 | 53.6 |
| ViTPose+-H | 256×192 | 53.1 | 60.6 |
| Sapiens-0.3B | 1024×768 | 53.4 (+0.3) | 60.9 |
| Sapiens-0.6B | 1024×768 | 56.2 (+2.8) | 62.4 (+2.1) |
| Sapiens-1B | 1024×768 | 59.4 (+5.9) | 65.3 (+5.1) |
| **Sapiens-2B** | 1024×768 | **61.1 (+7.6)** | **67.1 (+7.0)** |

Sapiens-0.3B 参数和 ViTPose+-L 同量级, 但 +5.6 AP — 这就是 high-res + human-centric pretrain 的纯增益.

### 5.2 Body-Part Segmentation

**Setup**: per-pixel classification to C classes (20 standard or **28** Sapiens).

Input $I$, 模型 $\mathcal{S}$, output probability map:
$$\hat{\mathbf{p}} = \mathcal{S}(\mathbf{I}) \in \mathbb{R}^{H \times W \times C}$$
$$\mathcal{L}_{seg} = \text{WeightedCE}(\mathbf{p}, \hat{\mathbf{p}})$$

变量:
- $\mathbf{p}_{h,w,c}$: GT one-hot label at pixel (h,w) for class c
- $\hat{\mathbf{p}}_{h,w,c}$: predicted softmax prob
- $C$: 20 (LIP standard) 或 **28** (Sapiens, 增加了 upper/lower lip, teeth, tongue, upper/lower limb halves, torso)

**28-class vocab** 是 paper 一个重要 contribution: 区分了 upper/lower limbs (动作理解需要), 加了 teeth/tongue (表情/对话场景), 加了 upper/lower lip (口型). 100K images manually annotated at 4K. **Multi-view capture setup** 保证 annotation quality & consistency.

🔗 LIP dataset: https://arxiv.org/abs/1703.05446  
🔗 Mask2Former: https://arxiv.org/abs/2112.01527  
🔗 DeepLabV3+: https://arxiv.org/abs/1802.02611

**结果** (Table 4, Humans-2K test):

| Model | mIoU (%) | mAcc (%) |
|---|---|---|
| FCN* | 48.2 | 57.6 |
| SegFormer* | 53.5 | 62.9 |
| Mask2Former* | 58.7 | 68.3 |
| DeepLabV3+* | 64.1 | 74.8 |
| Sapiens-0.3B | 76.7 (+12.6) | 86.1 |
| Sapiens-0.6B | 77.8 | 86.3 |
| Sapiens-1B | 79.9 | 89.1 |
| **Sapiens-2B** | **81.2** | **89.4** |

即使最小的 Sapiens-0.3B 也比 Mask2Former 高 12.6 mIoU — 这又是因为 resolution + human pretrain.

### 5.3 Depth Estimation

**Setup**: monocular depth, 用 600 RenderPeople scans 渲染 500K synthetic images @ 4K, random HDRI backgrounds, random camera focal/rotation/translation. 完全 synthetic supervision.

Ground truth depth: $\mathbf{d} \in \mathbb{R}^{H \times W}$, 归一化到 [0,1] (relative depth).

预测: $\hat{\mathbf{d}} = \mathcal{D}(\mathbf{I})$. $M$ = #human pixels.

**Loss (scale-invariant log MSE)**:
$$\Delta \mathbf{d} = \log(\mathbf{d}) - \log(\hat{\mathbf{d}}) \tag{1}$$
$$\overline{\Delta \mathbf{d}} = \frac{1}{M} \sum_{i=1}^{M} \Delta \mathbf{d}_i, \quad \overline{(\Delta \mathbf{d})^2} = \frac{1}{M} \sum_{i=1}^{M} (\Delta \mathbf{d}_i)^2 \tag{2}$$
$$\mathcal{L}_{depth} = \sqrt{\overline{(\Delta \mathbf{d})^2} - \frac{1}{2}(\overline{\Delta \mathbf{d}})^2} \tag{3}$$

变量解释:
- $\Delta \mathbf{d}_i$: pixel $i$ 处 GT depth 与 predicted depth 的 **log-ratio** (即 log-scale 残差)
- $\overline{\Delta \mathbf{d}}$: log-residual 的均值, 反映整体 scale shift
- $\overline{(\Delta \mathbf{d})^2}$: log-residual 的二阶矩
- $M$: 仅 human pixel 数 (背景 mask out)

**Intuition for the formula**: 设 $e_i = \log d_i - \log \hat{d}_i$. 公式 (3) 等价于
$$\mathcal{L} = \sqrt{\mathbb{E}[e^2] - \frac{1}{2}\mathbb{E}[e]^2}$$

注意这里 $\frac{1}{2}$ 而不是 Eigen 原始 paper 的 $\frac{1}{M^2}$ 形式. 这其实是 $Var(e) + \frac{1}{2}\mathbb{E}[e]^2 = \mathbb{E}[e^2] - \frac{1}{2}\mathbb{E}[e]^2$. 也就是说: 优化方向是 **同时最小化 log-residual 的方差 AND log-residual 的均值**. 这让预测在 log-space 上既精确 (低 variance) 又 calibrated (低 mean shift).

vs Eigen 的原始 SI loss: $\frac{1}{M}\sum e_i^2 - \frac{\lambda}{M^2}(\sum e_i)^2$, $\lambda=1$ 时纯 variance (fully scale-invariant), $\lambda=0$ 时纯 log-MSE. Sapiens 取 $\lambda = 1/2$ 是个折中: 既允许 model 学到绝对 scale (因为 synthetic data 的 scale 是 ground truth 的), 又允许一定 scale invariance.

🔗 Eigen et al. (depth multi-scale): https://arxiv.org/abs/1406.2283  
🔗 MiDaS v3.1: https://arxiv.org/abs/2307.14460  
🔗 Depth Anything: https://arxiv.org/abs/2401.10891  
🔗 ZoeDepth: https://arxiv.org/abs/2302.12288

**结果** (Table 5, 关键 row — Hi4D multi-human):

| Method | RMSE↓ | AbsRel↓ | $\delta_1$↑ |
|---|---|---|---|
| MiDaS-L | 0.261 | 0.082 | 0.975 |
| MiDaS-Swin2 | 0.209 | 0.063 | 0.997 |
| DepthAny-B | 0.143 | 0.034 | 0.997 |
| DepthAny-L | 0.147 | 0.035 | 0.997 |
| Sapiens-0.3B | 0.148 | 0.046 | 1.000 |
| Sapiens-1B | 0.125 | 0.039 | 1.000 |
| **Sapiens-2B** | **0.114** | **0.036** | **1.000** |

Hi4D RMSE 22.4% relative reduction over DepthAnything-L. 注意 Sapiens **仅用 synthetic data fine-tune**, generalize 到 real multi-person Hi4D. 这是 domain-specific pretraining + synthetic GT 的 strong 示范.

### 5.4 Surface Normal Estimation

**Setup**: synthetic data from same 600 RenderPeople scans. Decoder output channels = 3 (xyz of normal).

GT normal: $\mathbf{n}$ (unit vector at each pixel). Predicted: $\hat{\mathbf{n}} = \mathcal{N}(\mathbf{I})$.

**Loss (L1 + cosine)**:
$$\mathcal{L}_{normal} = \|\mathbf{n} - \hat{\mathbf{n}}\|_1 + (1 - \mathbf{n} \cdot \hat{\mathbf{n}}) \tag{4}$$

变量:
- $\mathbf{n} = (n_x, n_y, n_z) \in \mathbb{R}^3$, $\|\mathbf{n}\|=1$
- $\hat{\mathbf{n}} = (\hat{n}_x, \hat{n}_y, \hat{n}_z) \in \mathbb{R}^3$ (model output, ideally normalized)
- $\|\mathbf{n} - \hat{\mathbf{n}}\|_1 = |n_x - \hat{n}_x| + |n_y - \hat{n}_y| + |n_z - \hat{n}_z|$: per-component L1
- $\mathbf{n} \cdot \hat{\mathbf{n}} = \cos\theta$: cosine similarity between unit vectors
- $1 - \cos\theta$: 0 when identical, 2 when opposite

**Intuition**: L1 给 per-axis 的"分量准确"信号 (避免 normal "flip" 之类的局部最小), cosine 给 angular 信号 (本质上与 task metric 直接对齐). 两者结合是 normal estimation 的 de-facto 标配, 类似 depth 用 log + scale-invariant.

🔗 Omnidata (multi-task normals): https://arxiv.org/abs/2110.04394  
🔗 PIFuHD: https://arxiv.org/abs/2004.00452  
🔗 ICON: https://arxiv.org/abs/2206.08134  
🔗 ECON: https://arxiv.org/abs/2212.07422

**结果** (Table 6, Hi4D):

| Method | Mean Err (°) | Median (°) | % < 11.25° | % < 22.5° | % < 30° |
|---|---|---|---|---|---|
| PIFuHD | 22.39 | 19.26 | 22.98 | 60.14 | 77.02 |
| HDNet | 28.60 | 26.85 | 19.08 | 57.93 | 70.14 |
| ICON | 20.18 | 17.52 | 26.81 | 66.34 | 82.73 |
| ECON | 18.46 | 16.47 | 29.35 | 68.12 | 84.88 |
| Sapiens-0.3B | 15.04 | 12.22 | 47.07 | 81.49 | 90.70 |
| Sapiens-1B | 12.18 | 9.59 | 60.36 | 88.62 | 94.44 |
| **Sapiens-2B** | **12.14** | **9.62** | **60.22** | **89.08** | **94.74** |

THuman2 上 mean error 从 25.45 (ECON) → 11.84 (Sapiens-2B), **53.5% relative reduction**. ECON 用了 4000 scans 训练, Sapiens 只用 600 — **更少数据 + 更好 pretrain = 更好结果**.

---

## 6. Pretraining scaling curve (Fig. 10)

Fig. 10 给的是 Sapiens-0.3B 在 normal estimation (% within 30°) 上, vs unique human images seen during pretraining. 曲线**没有 saturate** — 从 ~10M 到 300M, 性能持续上升. 这是 scaling law 在 domain-specific vision pretraining 上的直接 evidence.

这跟 AIM (autoregressive image model), DINOv2 的 scaling 观察类似: vision foundation model 的 scaling 还远未到 plateau. Sapiens 的 contribution 是把这个观察搬到 **human-centric** 子域.

🔗 AIM scaling: https://arxiv.org/abs/2401.08541  
🔗 DINOv2 scaling: https://arxiv.org/abs/2304.07193  
🔗 Chinchilla scaling laws: https://arxiv.org/abs/2203.15556  
🔗 Scaling laws for neural language models (Kaplan): https://arxiv.org/abs/2001.08361

---

## 7. Generalization to in-the-wild (Fig. 11)

Sapiens finetune data 是 studio-captured, single-person, third-person view. 但模型 generalize 到:
- **Multi-human** (拥挤/互动场景)
- **Age diversity** (从婴儿到老人)
- **Egocentric views** (first-person)

这正验证了 paper 反复强调的论点: **large-scale human-centric pretraining 提供 prior, high-quality (即便小规模) finetune 提供 task alignment, 合起来 → wild generalization**.

这与 SAM (Segment Anything), Depth Anything 的 "synthetic + large-scale pretrain → wild generalization" 哲学一致, 但 Sapiens 限定在 human 子域, 且 zero-shot 到新视角/年龄/人数的能力更加 impressive.

🔗 SAM: https://arxiv.org/abs/2304.02643  
🔗 Depth Anything: https://arxiv.org/abs/2401.10891

---

## 8. Limitations & 我 (Karpathy) 的批评性思考

Paper 在 Section 4.6 诚实承认:
- **复杂/rare pose** 仍然 challenging
- **crowding / severe occlusion** 困难
- detect-and-crop 策略对 multi-person 受限

我从 model design / scaling 角度的几个 thoughts:

1. **Encoder-decoder 解耦**: Decoder 全 random init, fine-tune end-to-end. 这里可能可以更激进 — 用 task-specific pretrain decoder (e.g., 用 ViTPose 的 head). 但 paper 选择 simplicity, 合理.

2. **Width over depth**: 跟 LLaMA 一致, hidden dim 1024→1920, layers 24→48. 但 2B 的 layers 48 已经不算"shallow". 真正的 width scaling 极限还没被 push.

3. **Resolution 1024 是否足够?** 对 face keypoints (243 个) 来说, 1024 可能 marginal. 若 push 到 2048, FLOPs 4×, 但 facial micro-expression 可能需要. DPT/Real-ESRGAN 等超分辨率思路或可借鉴.

4. **Pretraining method 选择**: MAE 在 fine-tune heavy task (segmentation, depth) 上强, 在 retrieval/zero-shot classification 上弱 (vs DINOv2). Sapiens 全是 fine-tune task, MAE 选得对. 但未来若想做 open-vocab human parsing, 可能需要 DINOv2-style self-distillation.

5. **Synthetic data 依赖**: Depth/normal 完全靠 600 RenderPeople scans. 扫描 diversity 限制 wild generalization 上限. 可考虑结合 synthetic + real-with-pseudo-labels (类似 Depth Anything 的大规模 pseudo-labeling pipeline).

6. **Multi-modal / 3D extension**: Paper conclusion 提到 "future direction: 3D and multi-modal". 可以想象 Sapiens-3D: 输入 image, 直接输出 SMPL-X / 人脸 blendshape / 手部 mesh. 与 animatable NeRF / Gaussian Splatting (HUGS, Animatable Gaussians) 的 connection 也很有想象力.

🔗 HUGS: https://arxiv.org/abs/2311.17910  
🔗 Animatable Gaussians: https://arxiv.org/abs/2311.16086  
🔗 SMPL-X: https://arxiv.org/abs/1905.03789

---

## 9. 与相关工作 landscape 的关系

```
                  General Pretrain           Human Pretrain
                  ─────────────────          ──────────────
Contrastive       DINO, MoCo, SimCLR         
Self-distill      DINOv2, iBOT               
MAE               MAE, MAWS, AIM             Sapiens (this paper)
                  ─────────────────          ──────────────
                  Resolution 224             Resolution 1024 (native)
                  ~1B-6.5B params             0.3B-2B
                  1657-8709 GFLOPs           
```

Sapiens 在右上角开了个新 quadrant: **human-domain + high-res MAE**. 关键的是它把 Table 1 里那条 1024 / 8709 GFLOPs 的 row 打通了 — 之前没人愿意在 1024 native 上做 2B 参数 MAE, 因为太贵. Sapiens 投入了 1024 A100 × 18 days 的 pretraining budget, 才把这个 quadrant 跑通.

🔗 MAWS (Meta's MAE scaling): https://arxiv.org/abs/2303.13496  
🔗 iBOT: https://arxiv.org/abs/2111.07832

---

## 10. 实用 takeaways (给我自己 / community 的 notes)

- **Domain curation > raw data volume** (在 fixed compute 下). 这对所有 verticals (medical, driving, OCR) 都适用.
- **Native high-res + small patch** 对 fine-grained dense prediction (keypoint/normal/depth) 关键. 不要轻易 resize 到 224.
- **Synthetic + curated human pretrain + small real high-quality finetune** 是 viable path. 完全跳过大规模 real labeling.
- **Layer-wise LR decay 0.85** 是简单有效的 fine-tune trick, vision 里 underused.
- **Width scaling over depth** (LLaMA 经验) 在 vision ViT 上也 work.
- **MAE + domain data** 是被低估的组合 — DINOv2/iBOT 更出名, 但 fine-tune 场景 MAE 的 throughput 优势巨大.

---

## 11. 我 (Karpathy) 的几个进一步联想

1. **Sapiens 作为 "perception token" generator**: 想象把 Sapiens 的 4 个 head 输出 (pose, seg, depth, normal) concat 起来, 作为 downstream generative model (Sora-like human video model) 的 conditioning. 这是结构化、信息密集、human-aware 的 conditioning, 比 raw CLIP embedding 强很多.

2. **Encoder 作为 universal human prior**: 任何 human-centric task (motion prediction, gait recognition, emotion, sign language) 都可以 fine-tune Sapiens encoder. 这是 "foundation model" 的真正含义.

3. **与 ARC-AGI / human-centric reasoning 的联系**: Sapiens 学到的是 perception-level human prior, 跟 reasoning 不沾边. 但未来 humanoid robot (Figure 02, Tesla Optimus) 需要的 vision system, 很可能长这样: Sapiens-style perception + VLA reasoning head.

4. **Privacy / ethics**: 一个能 robust 估计 308 个 keypoint + 28-class segmentation + depth + normal 的模型, 对 deepfake detection, surveillance, biometric 都是双刃剑. Meta 选择不开源 weights, 仅发 paper, 这点值得注意.

🔗 Sora (video gen): https://openai.com/sora  
🔗 Figure AI: https://www.figure.ai/  
🔗 VLA (vision-language-action): https://arxiv.org/abs/2310.08878 (RT-2)

---

## 总结 (one-paragraph intuition)

Sapiens 的 thesis 一句话: **在 fixed compute budget 下, 把 pretraining 算力花在 task-domain data (humans) 上, 比花在 general data 上更能学到 downstream 有用的 prior**. 这 + 高分辨率 (1024 native) + 简单 MAE + 规模 (0.3B→2B) + 高质量小规模 finetune data (synthetic for depth/normal, studio for pose/seg) → 在四个 human-centric benchmark 上 SOTA. Sapiens 是 vision foundation model 时代 "domain-specific pretraining" 的 strong validation, 也直接呼应 LLM 时代 code/medical/finance domain-pretrain 的同构逻辑, 只是在 vision 里这件事过去一直被 general ImageNet/DINOv2 pretraining 掩盖了.
