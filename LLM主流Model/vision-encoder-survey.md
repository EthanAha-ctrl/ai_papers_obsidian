---
source_pdf: vision-encoder-survey.pdf
paper_sha256: 119613cb75bab042172e91b5fe7c4ab4edcd4518761f0bc1f3e50126cce0d7c3
processed_at: '2026-08-13T01:29:26-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这篇 paper 其实就在讲一个特别简单的故事，但我得用尽量直白的话把里面的门道给你全拆开。

## 核心一句话

你给 VLM 配的 vision encoder，**搞大没用，搞对才有用**。一个 400M 参数的 SigLIP 2，能把 5.9B 参数的 InternViT-6B 按在地上摩擦，前提是你 loss function 和 data curation 搞对了。

## 为什么 encoder 这么重要又这么被忽视

VLM 的架构存在一个巨大的不对称：LLM 从 2020 年到 2024 年，参数量从 billions 涨到 hundreds of billions，但 vision encoder 基本一直冻结在 300M 到 600M 的 CLIP 变体上。LLaVA 用的 CLIP ViT-L/14，2021 年的东西，到现在还有一堆 production 系统在用。

这就引出一个问题：vision encoder 到底重不重要？

答案是：**看任务**。
- 如果任务靠 language reasoning 主导（比如 VQAv2、GQA），encoder 随便选个 SigLIP-SO400M 就行，CLIP 跟 SigLIP 2 可能就差 1-2 分。
- 如果任务是 vision-centric（比如 OCR、document understanding、spatial reasoning、counting），encoder 的选择直接决定天花板，不同 encoder 之间能差几十分。

## Encoder 家族的三条主线

这篇 survey 把 vision encoder 分成三大类，每一类的 inductive bias 完全不同：

### 1. Contrastive (CLIP 系)

代表：CLIP, SigLIP, SigLIP 2

训练方式：一堆 image-text pair，让 image encoder 和 text encoder 学会把配对的 image/text 拉近，不配对的拉远。CLIP 用的是 softmax loss：

$$\mathcal{L}_{\mathrm{CLIP}} = -\frac{1}{2|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|}\left[\log\frac{e^{\tau \mathbf{x}_i^\top \mathbf{y}_i}}{\sum_{j=1}^{|\mathcal{B}|} e^{\tau \mathbf{x}_i^\top \mathbf{y}_j}} + \log\frac{e^{\tau \mathbf{x}_i^\top \mathbf{y}_i}}{\sum_{j=1}^{|\mathcal{B}|} e^{\tau \mathbf{x}_j^\top \mathbf{y}_i}}\right]$$

变量解释：
- $\mathcal{B}$：一个 batch，里面全是 image-text pair
- $\mathbf{x}_i$：第 $i$ 张图的归一化 embedding
- $\mathbf{y}_i$：第 $i$ 个 text 的归一化 embedding
- $\tau$：temperature，控制 softmax 的 sharpness
- 分母里的 $\sum_j$：把 batch 里所有的 text embedding 都拿出来比

直觉：这个 loss 本质上是把 batch 当成一个 classification 问题，batch size 就是类别数。正样本对要在 batch 里所有的配对中胜出。致命弱点是要求 batch size 极大（32K+），不然负样本不够多样，分布式训练的同步成本极高。

SigLIP 改成了 sigmoid loss：

$$\mathcal{L}_{\mathrm{SigLIP}} = -\frac{1}{|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|}\sum_{j=1}^{|\mathcal{B}|}\log\sigma(z_{ij}(\tau \mathbf{x}_i^\top \mathbf{y}_j + b))$$

变量解释：
- $z_{ij}$：正样本为 $+1$，负样本为 $-1$
- $b$：learnable bias
- $\sigma$：sigmoid

关键区别：每一对 $(i, j)$ 独立做 binary classification，不依赖 batch 内其他样本。softmax 是 "在全班选第一名"，sigmoid 是 "每对学生独立判断像不像"。后者训练更稳定，batch size 可以灵活。

SigLIP 2 更进一步，staged training，第二阶段把多个 loss 拼一起：

$$\mathcal{L}_{\mathrm{SigLIP2}} = \mathcal{L}_{\mathrm{sig}} + \mathcal{L}_{\mathrm{LocCa}} + \alpha(\mathcal{L}_{\mathrm{distill}} + \mathcal{L}_{\mathrm{mask}})$$

变量解释：
- $\mathcal{L}_{\mathrm{sig}}$：原 sigmoid contrastive loss
- $\mathcal{L}_{\mathrm{LocCa}}$：captioning + referring expression 的 decoder loss，用 lightweight cross-attention decoder
- $\mathcal{L}_{\mathrm{distill}}$：local-to-global self-distillation（partial view 匹配 full image）
- $\mathcal{L}_{\mathrm{mask}}$：masked patch prediction（student 重建 teacher 在 mask 位置的特征）
- $\alpha$：weighting factor

直觉：contrastive loss 学 image-level 语义，LocCa 强加 location-aware 语言对齐，distill + mask 注入 dense spatial features。SigLIP 2 在 OCR 和 segmentation 上比 SigLIP 强，Qwen3-VL 和 Gemma 3 都用这个。

### 2. Self-Supervised (DINOv2 系)

代表：DINOv2, DINOv3, Web-SSL

训练方式：不用 text，纯靠 image 的不同 view 之间做 self-distillation。Student 跟 Teacher 学，Teacher 参数是 Student 的 EMA：

$$\pmb{\theta}_t \gets \lambda \pmb{\theta}_t + (1-\lambda) \pmb{\theta}_s$$

变量解释：
- $\pmb{\theta}_t$：Teacher 参数
- $\pmb{\theta}_s$：Student 参数
- $\lambda$：EMA decay（接近 1）

Image-level self-distillation：

$$\mathcal{L}_{\mathrm{DINO}} = -\sum_{v \in \mathcal{V}_g}\mathbf{p}_t(v)\log\mathbf{p}_s(v)$$

变量解释：
- $\mathcal{V}_g$：global crop views
- $\mathbf{p}_t, \mathbf{p}_s$：Teacher/Student 输出的 prototype scores

Patch-level masked prediction（iBOT）：

$$\mathcal{L}_{\mathrm{iBOT}} = -\sum_{i \in \mathcal{M}}\mathbf{p}_t^i\log\mathbf{p}_s^i$$

变量解释：
- $\mathcal{M}$：masked patch indices
- Teacher 看完整图像，Student 看 masked 版本

直觉：DINOv2 学的是 visual geometry 的 invariant structure，没有语言监督。所以它在 segmentation、depth estimation、object counting 这些 "不看 label 也能定义" 的任务上强，但在 OCR 这种需要 language alignment 的任务上弱。

### 3. LLM-Aligned (InternViT 系)

代表：InternViT, SAILViT

训练方式：vision encoder 先跟 text encoder 做 contrastive pretrain，再跟 LLM 联合 generative fine-tune：

$$\mathcal{L}_{\mathrm{gen}} = -\sum_{t=1}^T \log P(w_t | w_{<t}, \mathbf{Z}_v; \theta_v, \theta_l)$$

变量解释：
- $w_t$：第 $t$ 个 text token
- $\mathbf{Z}_v$：vision encoder 输出的 visual features
- $\theta_v, \theta_l$：vision encoder / LLM 参数

直觉：encoder 被 LLM 的 next-token prediction 直接监督，学的是 "什么 visual feature 能帮 LLM 预测下一个 token"。InternViT-6B 把这个路线推到 6B 参数，但实测发现：比 300M 版本只涨 2 分，FLOPs 涨 19×。**Vision encoder 不是 VLM 能力的瓶颈**。

## 为什么 scaling law 在这里失效

LLM 的 scaling law 是 10× params 带来稳定收益。Vision encoder 这里 10× params（CLIP-L → InternViT-6B）只在 standalone benchmark（ImageNet）上显著，集成进 VLM 后收益消失。

可能的解释：
1. **Vision 信息熵有限**：自然图像的 category + spatial structure 不需要 6B 参数来表达。
2. **LLM 是 bottleneck**：visual feature 进 LLM 后，LLM 的 reasoning capacity 决定 ceiling。
3. **Connector 是 information bottleneck**：MLP / Q-Former 把 $D_v$ 维投到 $D_l$ 维，projection 限制了 information flow。
4. **Benchmark saturation**：VQAv2 / GQA 已经饱和，vision-centric benchmark 上 scale 仍有收益。

## Resolution handling 是个大问题

Fixed resolution（224/336）是 2023 的事实标准，但 document 任务暴露了它的局限。336×336 + $P=14$ 产生 576 个 token，每个 token 对应约 14×14 像素，对 10pt 字体来说每个字符只占 1-2 个 patch 的部分区域，信息密度不够。

三种 resolution strategy：

### AnyRes tiling

切 grid，让 aspect ratio 失真最小：

$$(n_h^*, n_w^*) = \arg\min_{(n_h, n_w) \in \mathcal{G}}\left|\frac{H}{W} - \frac{n_h}{n_w}\right|$$

变量解释：
- $H, W$：原图尺寸
- $n_h, n_w$：grid 行列数
- $\mathcal{G}$：预定义 grid 配置集合

Token 数：$(n_h \cdot n_w + 1) \cdot (P_{\mathrm{tile}}/p)^2$，$+1$ 是 global thumbnail。LLaVA-NeXT 用这个。问题：4K 图能产生 21K token。

### Pixel Shuffle (InternVL)

把空间维度 fold 到 channel 维度：

$$\mathbf{Z}_{i,j}' = \mathrm{concat}[\mathbf{Z}_{ri+a, rj+b}]_{a,b \in \{0, \ldots, r-1\}} \in \mathbb{R}^{r^2 D}$$

变量解释：
- $r$：downsampling factor
- $\mathbf{Z}$：原 feature map
- $\mathbf{Z}'$：shuffle 后的 feature map

直觉：spatial dimension 缩 $r^2$ 倍，channel 维度涨 $r^2$ 倍，总信息量不变。SmolVLM 用 $r=3$ 实现 9× 压缩。

### NaViT + M-RoPE (Qwen2-VL)

NaViT 打破 ViT 必须固定 input shape 的限制，用 "Patch n' Pack" 把多张不同分辨率的图打包进一个 sequence：

$$L_{\mathrm{pack}} = \sum_{i=1}^k n_i, \quad n_i = \left\lfloor\frac{H_i}{P}\right\rfloor \cdot \left\lfloor\frac{W_i}{P}\right\rfloor$$

变量解释：
- $k$：一个 packed sequence 里的图像数
- $n_i$：第 $i$ 张图的 patch 数

M-RoPE 把 position embedding 分解为 temporal, height, width 三个维度：

$$\mathbf{M}.\mathbf{RoPE}(\mathbf{x}, t, h, w) = \mathbf{x} \odot [\cos(\theta_t), \cos(\theta_h), \cos(\theta_w)] + \mathbf{x}' \odot [\sin(\theta_t), \sin(\theta_h), \sin(\theta_w)]$$

变量解释：
- $t, h, w$：temporal, height, width position ID
- $\theta_t, \theta_h, \theta_w$：三个维度的 rotation angles
- $\mathbf{x}'$：$\mathbf{x}$ 旋转 $\pi/2$ 后的版本

直觉：text token 时三个 ID 完全相同，退化为 1D-RoPE；image token 时 $t$ 恒定，$h, w$ 反映 2D 位置；video 时 $t$ 随帧递增。一套 position encoding 跨 modality 通用。

## Connector：简单粗暴最好

MM1 的 ablation 显示 connector 复杂度对最终性能影响小，simple MLP 在控制其他变量时跟复杂设计持平。

最简单的 MLP projection：

$$\mathbf{H}_v = \mathbf{W}_2 \cdot \mathbf{GELU}(\mathbf{W}_1 \cdot \mathbf{Z}_v)$$

变量解释：
- $\mathbf{Z}_v \in \mathbb{R}^{N \times D_v}$：visual tokens
- $\mathbf{W}_1 \in \mathbb{R}^{D_h \times D_v}$：升维到 hidden dim
- $\mathbf{W}_2 \in \mathbb{R}^{D_l \times D_h}$：投影到 LLM embedding dim

这反向证明：**encoder representation quality 才是 VLM 能力的主要决定因素**，connector 只是格式转换器。

## Multi-encoder：互补性假设

Cambrian-1 用四个 encoder 融合：CLIP, SigLIP, ConvNeXt, DINOv2。融合公式：

$$\mathbf{F}_{\mathrm{fused}} = \mathrm{Aggregate}(\mathbf{F}_1, \mathbf{F}_2, \ldots, \mathbf{F}_K; \mathbf{Q})$$

变量解释：
- $\mathbf{F}_k$：第 $k$ 个 encoder 输出的 feature map
- $\mathbf{Q}$：learnable query tokens

直觉：每个 encoder 的 inductive bias 不同——CLIP 偏 image-level semantic，SigLIP 偏 text-aligned，DINOv2 偏 spatial geometry，ConvNeXt 偏 multi-scale texture。融合相当于让 LLM 同时访问这些 prior。代价是 4× FLOPs，SCOPE 用 dynamic routing 把开销降到 24-49%。

更激进的是 distillation：AM-RADIO 把 CLIP + DINOv2 + SAM 蒸馏进一个 student，single-encoder cost 拿到 multi-teacher 的 diversity。

## Encoder-free：去掉 vision encoder 行不行？

Fuyu-8B 直接把 patch 投进 LLM：

$$\mathbf{h}_i = \mathbf{W}_{\mathrm{patch}} \cdot \mathrm{flatten}(\mathbf{x}_p^i) + \mathbf{e}_{\mathrm{pos}}^i$$

变量解释：
- $\mathbf{x}_p^i$：第 $i$ 个 image patch
- $\mathbf{W}_{\mathrm{patch}}$：patch-to-embedding 投影矩阵
- $\mathbf{e}_{\mathrm{pos}}^i$：position embedding

直觉：完全跳过 vision-specific pretrain，让 LLM 在 VLM 训练时自己学 visual perception。SAIL 的 scaling 分析显示：encoder-free model 在足够大时能 match modular VLM，但 training compute 显著更高。

Chameleon / Emu3 走另一极端：用 VQ-VAE 把图像 tokenize 成离散 codebook index，跟 text token 完全统一。代价是 visual fine-grained 信息受 codebook 容量限制。

## Token pruning：训练时的 attention-guided pruning

PTP 用 top-down instruction attention + bottom-up visual saliency 联合决定哪些 visual token 重要：

$$c_j = \max_{q \in \mathcal{Q}}\mathrm{Attn}_{q \to j}$$

$$s_j = \alpha c_j + (1-\alpha)b_j$$

变量解释：
- $\mathcal{Q}$：instruction token 的 index 集合
- $\mathrm{Attn}_{q \to j}$：LLM 早期层中 instruction token $q$ 对 visual token $j$ 的 attention score
- $c_j$：visual token $j$ 的 instruction relevance
- $b_j$：来自 vision encoder 中间层的 visual saliency
- $\alpha$：平衡系数

经验：$\alpha = 0.5$ 通用任务，OCR 任务偏 small $\alpha$（靠 visual saliency），open-domain reasoning 偏 large $\alpha$（靠 instruction guidance）。PTP 实现 50% token pruning 几乎不损精度。

## 实测数据告诉你的真相

| Configuration | Params | GFLOPs | VRAM | ImageNet | VQAv2 |
|---|---|---|---|---|---|
| CLIP ViT-L/14 | 304M | 81.1 | 1.2GB | 75.5 | 79.2 |
| SigLIP-SO400M | 400M | 95.8 | 1.5GB | 83.2 | 81.7 |
| SigLIP 2 S0400M | 400M | 98.3 | 1.6GB | 84.1 | 82.4 |
| InternViT-6B | 5.9B | 1,547 | 24GB | 88.2 | 82.1 |
| DINOv2-L | 304M | 81.6 | 1.2GB | 86.3 | 73.2† |
| Cambrian (4 enc) | 2.1B | 412 | 5.8GB | – | 77.8 |

关键观察：
1. **SigLIP-SO400M → SigLIP 2**：FLOPs 几乎不变，ImageNet +0.9，VQAv2 +0.7。Training methodology 的边际收益。
2. **SigLIP 2 (400M) vs InternViT-6B**：FLOPs 差 16×，ImageNet InternViT 高 4.1 分，但 VQAv2 SigLIP 2 反超 0.3 分。**ImageNet 不能预测 VLM 性能**。
3. **DINOv2**：ImageNet 86.3 极高，但 VQAv2 只有 73.2（需 text-aligned fine-tune）。
4. **Multi-encoder**：Cambrian 4-enc 比 SigLIP 单 encoder 多 4× FLOPs，VQAv2 反而低，但 vision-centric benchmark 强。

## Encoder sensitivity 的 task 维度

**High sensitivity**：Document 类、Spatial 类、Fine-grained 类、Video 类。
**Medium sensitivity**：综合类。
**Low sensitivity**：VQAv2, GQA, ScienceQA。

直觉：VQAv2 / GQA 已经 saturated，CLIP ViT-L/14 跟 SigLIP 2 之间可能只差 1-2 分。但 OCRBench、TallyQA 这些任务换 encoder 能差 100+ 分。**如果你的应用是 document understanding 或 spatial reasoning，encoder 选择是主要变量；如果是普通 VQA，随便选个 SigLIP-SO400M 就行**。

## 给 practitioner 的人话建议

1. **Default 选 SigLIP 2 S0400M**：2025 年的事实标准，多语言，dense features 强，Qwen3-VL 和 Gemma 3 都用这个。
2. **Document 任务用 native resolution encoder**：NaViT-style 或 DeepEncoder。Native resolution 保留的信息预处理损失后无法恢复。
3. **Spatial task 考虑 DINOv2 + SigLIP fusion**：互补性强，但 4× FLOPs。
4. **Resource-constrained 用 SigLIP 2 Base/Large**：86M-303M 参数，质量维持得不错。
5. **别瞎折腾 connector**：MM1 的 ablation 显示 connector 复杂度影响小，simple MLP 最稳。
6. **别一味 scale encoder**：400M 以上 scaling law 失效，把钱花在 LLM 和 data 上。

## 最深的直觉

这篇 survey 真正反直觉的是 **scaling law 在 vision encoder 这里失效**。LLM 的 scaling law 来自 next-token prediction 这个 "无限 difficulty" 的 task，模型越大越好；vision encoder 学的是 finite-entropy 的 visual representation，过了某个 threshold，extra capacity 学不到新东西。

第二个反直觉是 **encoder-free 路线 viable**。SAIL 显示 encoder-free 模型 scale 上去后能 match modular VLM，但 training compute 显著高。这暗示 pretrained vision encoder 主要价值是 "compute-efficient initialization"，如果未来 training compute 继续变便宜，encoder-free 的 unified architecture 可能成为主流。

第三个是 **document understanding 是 encoder 选择的风向标**。所有 general VLM 在 VQAv2 上都 80+，但 OCRBench 从 600 到 850 都有，差距极大。**encoder 的差异化主要体现在 "需要 dense spatial reasoning" 的任务上**，因为这些任务的信息不能靠 language prior 补救。

最后，**multi-encoder fusion 的本质是 inductive bias ensemble**。CLIP 的 inductive bias 是 "image-text alignment"、DINOv2 的是 "visual geometry invariance"、SAM 的是 "segmentation structure"——每个 encoder 在 pretrain 时注入不同的 prior。融合相当于让 LLM 同时访问这些 prior。distillation 路线（AM-RADIO）把这个 ensemble 压回 single-encoder cost，是工程上最实用的方向。

## 参考链接

- CLIP: https://arxiv.org/abs/2103.00020
- SigLIP: https://arxiv.org/abs/2303.15343
- SigLIP 2: https://arxiv.org/abs/2502.14786
- DINOv2: https://arxiv.org/abs/2304.07193
- DINOv3: https://arxiv.org/abs/2508.10104
- InternVL: https://arxiv.org/abs/2312.14238
- NaViT: https://arxiv.org/abs/2307.06304
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- LLaVA: https://arxiv.org/abs/2304.08485
- BLIP-2: https://arxiv.org/abs/2301.12597
- Flamingo: https://arxiv.org/abs/2204.14198
- Cambrian-1: https://arxiv.org/abs/2406.16860
- AM-RADIO: https://arxiv.org/abs/2312.06685
- Fuyu-8B: https://www.adept.ai/blog/fuyu-8b
- EVE: https://arxiv.org/abs/2406.11838
- SAIL: https://arxiv.org/abs/2504.10462
- Chameleon: https://arxiv.org/abs/2405.09818
- PTP: https://arxiv.org/abs/2509.15704
- MetaCLIP 2: https://arxiv.org/abs/2507.22062
- NVLM: https://arxiv.org/abs/2409.11402
- MM1: https://arxiv.org/abs/2403.18500

---

# Vision Encoders in VLMs: A Survey 深度解析

Andrej，这篇 Jina AI 出品的 survey 是 2026 年初对 vision encoder 在 VLM 中角色的系统性梳理。最有意思的发现是：**training methodology dominates parameter scale**——一个 400M 的 SigLIP 2 在大多数 VLM benchmark 上吊打 5.9B 的 InternViT-6B。这个结论本身就值得深入思考其背后的机制。

## 一、核心叙事：architecture asymmetry 的演化

2020–2024 期间 LLM 从 billions 涨到 hundreds of billions，vision encoder 却基本冻结在 300–600M 的 CLIP 变体。这篇 survey 整理了 70+ 个 VLM，把这个 asymmetry 量化，并追问 "does the vision encoder matter?" 答案是 nuanced 的：对 vision-centric 任务（document understanding、spatial reasoning、fine-grained recognition）encoder choice 极其关键，对 language reasoning 主导的任务则几乎不敏感。

参考：https://arxiv.org/abs/2507.22062 (MetaCLIP 2)
参考：https://arxiv.org/abs/2502.14786 (SigLIP 2)

## 二、Vision Transformer 基础架构回顾

ViT 是几乎所有现代 vision encoder 的底座。输入图像 $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ 被切成非重叠 patches：

$$\mathbf{z}_0 = [\mathbf{x}_{\mathrm{cls}}; \mathbf{E}\mathbf{x}_p^1; \mathbf{E}\mathbf{x}_p^2; \ldots; \mathbf{E}\mathbf{x}_p^N] + \mathbf{E}_{\mathrm{pos}} \tag{1}$$

变量解析：
- $H, W, C$：图像的高、宽、通道数
- $N = HW/P^2$：patch 数量，$P$ 为 patch size
- $\mathbf{E} \in \mathbb{R}^{D \times (P^2 \cdot C)}$：patch embedding projection matrix，把 flatten 后的 patch 像素投到 $D$ 维
- $\mathbf{x}_{\mathrm{cls}}$：learnable 的 [CLS] token
- $\mathbf{E}_{\mathrm{pos}} \in \mathbb{R}^{(N+1) \times D}$：positional embeddings，$N+1$ 是因为多一个 CLS token

关键直觉：336×336 + $P=14$ 产生 576 个 token。如果你想做 OCR，576 个 token 每个对应约 14×14 像素，对 10pt 字体来说每个字符只占 1–2 个 patch 的部分区域，信息密度不够——这就解释了为什么 document 任务对 resolution 如此敏感。

参考：https://arxiv.org/abs/2010.11929 (ViT 原始论文)

## 三、三种 training paradigm：本质区别与互补性

### 3.1 Contrastive (CLIP 系)

给定 batch $\mathcal{B} = \{(\mathbf{I}_i, \mathbf{T}_i)\}_{i=1}^{|\mathcal{B}|}$，归一化后的 image/text embedding $\mathbf{x}_i = f(\mathbf{I}_i)/\|f(\mathbf{I}_i)\|$、$\mathbf{y}_i = g(\mathbf{T}_i)/\|g(\mathbf{T}_i)\|$，CLIP 的 symmetric InfoNCE loss：

$$\mathcal{L}_{\mathrm{CLIP}} = -\frac{1}{2|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|}\left[\log\frac{e^{\tau \mathbf{x}_i^\top \mathbf{y}_i}}{\sum_{j=1}^{|\mathcal{B}|} e^{\tau \mathbf{x}_i^\top \mathbf{y}_j}} + \log\frac{e^{\tau \mathbf{x}_i^\top \mathbf{y}_i}}{\sum_{j=1}^{|\mathcal{B}|} e^{\tau \mathbf{x}_j^\top \mathbf{y}_i}}\right] \tag{2}$$

变量：
- $\tau$：learnable temperature，控制 softmax sharpness
- 第一个 log：image-to-text 方向
- 第二个 log：text-to-image 方向，对称求平均
- 分母对所有 $j$ 求和：包括正样本 $j=i$ 和负样本 $j \neq i$

直觉：这本质上是一个 batch 内的 cross-modal classification——把 batch size 当成类别数，要求正样本对的 inner product 在所有配对里最大。**致命弱点**：需要 32K+ batch size 才能让负样本足够多样，softmax 把每对样本拉进同一场竞争，分布式训练同步代价巨大。

### 3.2 SigLIP：sigmoid 替代 softmax

$$\mathcal{L}_{\mathrm{SigLIP}} = -\frac{1}{|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|}\sum_{j=1}^{|\mathcal{B}|}\log\sigma(z_{ij}(\tau \mathbf{x}_i^\top \mathbf{y}_j + b)) \tag{3}$$

变量：
- $z_{ij} = +1$ 当 $i, j$ 是 positive pair，$z_{ij} = -1$ 当 negative
- $b$：learnable bias，shift decision boundary
- $\sigma(\cdot)$：sigmoid function

关键差异：每对 $(i, j)$ 独立做 binary classification，不依赖 batch 内其他样本的相对关系。这样 batch size 可以更灵活、分布式训练无需 global gather。**直觉上**，softmax loss 是 "在全班选第一名"，sigmoid loss 是 "每对学生独立判断像不像"。后者更容易稳定 scaling，工程上更友好。

参考：https://arxiv.org/abs/2303.15343 (SigLIP)

### 3.3 SigLIP 2：staged multi-objective

SigLIP 2 第二阶段把多个 objective 合到一起：

$$\mathcal{L}_{\mathrm{SigLIP2}} = \mathcal{L}_{\mathrm{sig}} + \mathcal{L}_{\mathrm{LocCa}} + \alpha(\mathcal{L}_{\mathrm{distill}} + \mathcal{L}_{\mathrm{mask}}) \tag{4}$$

变量：
- $\mathcal{L}_{\mathrm{sig}}$：原 sigmoid contrastive loss
- $\mathcal{L}_{\mathrm{LocCa}}$：captioning + referring expression + grounded captioning 的 decoder loss（用 lightweight cross-attention decoder）
- $\mathcal{L}_{\mathrm{distill}}$：local-to-global self-distillation（SILC），partial view 匹配 full image teacher
- $\mathcal{L}_{\mathrm{mask}}$：masked patch prediction（TIPS），student 重建 teacher 在 mask 位置的特征
- $\alpha$：weighting factor，按 model size 调整

直觉：contrastive loss 学 image-level 语义、LocCa 强加 location-aware 语言对齐、distill + mask 注入 dense spatial features。这是一个 "multi-objective co-training" 范式，让 encoder 同时具备 semantic alignment 和 dense feature quality。Qwen3-VL 和 Gemma 3 都用这个，说明这个 recipe 是 2025 的事实标准。

### 3.4 Self-Supervised (DINOv2)

DINOv2 用 student-teacher 框架，teacher 参数是 student 的 EMA：

$$\pmb{\theta}_t \gets \lambda \pmb{\theta}_t + (1-\lambda) \pmb{\theta}_s \tag{5}$$

变量：
- $\pmb{\theta}_t$：teacher parameters
- $\pmb{\theta}_s$：student parameters
- $\lambda$：EMA decay rate（接近 1，慢更新）

Image-level self-distillation（global views 之间）：

$$\mathcal{L}_{\mathrm{DINO}} = -\sum_{v \in \mathcal{V}_g}\mathbf{p}_t(v)\log\mathbf{p}_s(v) \tag{6}$$

变量：
- $\mathcal{V}_g$：global crop views 集合
- $\mathbf{p}_t, \mathbf{p}_s$：teacher/student head 输出的 prototype scores（softmax 归一化）

Patch-level masked prediction（iBOT）：

$$\mathcal{L}_{\mathrm{iBOT}} = -\sum_{i \in \mathcal{M}}\mathbf{p}_t^i\log\mathbf{p}_s^i \tag{7}$$

变量：
- $\mathcal{M}$：masked patch indices 集合
- 上标 $i$：第 $i$ 个 patch 的 prototype distribution
- teacher 看完整图像，student 看 masked 版本

KoLeo regularizer（基于 Kozachenko-Leonenko 微分熵估计）：

$$\mathcal{L}_{\mathrm{KoLeo}} = -\frac{1}{n}\sum_{i=1}^n\log(d_{n,i}), \quad d_{n,i} = \min_{j \neq i}\|\mathbf{x}_i - \mathbf{x}_j\| \tag{8}$$

变量：
- $d_{n,i}$：batch 内 $\mathbf{x}_i$ 到最近邻的距离
- 直觉：让特征空间内点分布均匀（uniform spread），避免 representation collapse

总 objective：$\mathcal{L} = \mathcal{L}_{\mathrm{DINO}} + \mathcal{L}_{\mathrm{iBOT}} + \mathcal{L}_{\mathrm{KoLeo}}$

**DINOv2 vs SigLIP 的本质区别**：DINOv2 学到的是 visual geometry 自身的 invariant structure（无语言监督），所以它在 segmentation、depth estimation、object counting 这些 "不看 label 也能定义" 的任务上强；SigLIP 学到的是 "image 跟 text 的对齐"，所以它在 OCR、semantic recognition 上强，但 spatial reasoning 弱。两者互补是 multi-encoder 的根本动因。

参考：https://arxiv.org/abs/2304.07193 (DINOv2)

### 3.5 LLM-Aligned Training (InternViT)

InternViT 提出 progressive alignment：先 contrastive pretrain 对齐 text encoder，再 generative fine-tune 跟 LLM 联合训练：

$$\mathcal{L}_{\mathrm{gen}} = -\sum_{t=1}^T \log P(w_t | w_{<t}, \mathbf{Z}_v; \theta_v, \theta_l) \tag{9}$$

变量：
- $w_t$：第 $t$ 个 text token
- $w_{<t}$：前 $t-1$ 个 token 的 context
- $\mathbf{Z}_v$：vision encoder 输出的 visual features
- $\theta_v, \theta_l$：vision encoder / LLM 参数

直觉：encoder 不再独立优化 "image-text 对齐"，而是被 LLM 的 next-token prediction 直接监督。encoder 学到的是 "什么 visual feature 能帮 LLM 预测下一个 token"。InternViT-6B 这个方向把 encoder 拉到 6B 参数级别，但 survey 的实测显示：InternViT-6B 比 InternViT-300M 在 VQAv2 上只涨 2 分，FLOPs 涨 19×。**vision encoder 不是 VLM 能力的瓶颈**——这是一个反 scaling-law 现象。

参考：https://arxiv.org/abs/2312.14238 (InternVL)

## 四、Resolution handling 的演进

这是这篇 survey 的另一个核心维度。固定分辨率 224/336 是 2023 的事实标准，但 document 任务暴露了它的局限。三种策略：

### 4.1 AnyRes tiling

选 grid 让 aspect ratio 失真最小：

$$(n_h^*, n_w^*) = \arg\min_{(n_h, n_w) \in \mathcal{G}}\left|\frac{H}{W} - \frac{n_h}{n_w}\right| \tag{14}$$

变量：
- $H, W$：原图尺寸
- $n_h, n_w$：grid 的行列数
- $\mathcal{G}$：预定义 grid 配置集合（如 $\{1\times1, 1\times2, 2\times1, 2\times2, \ldots\}$）

token 数：$N_{\mathrm{tokens}} = (n_h \cdot n_w + 1) \cdot (P_{\mathrm{tile}}/p)^2$，$+1$ 是 global thumbnail。

直觉：把高分辨率图像切成多个 336×336 的 tile + 一个全局缩略图，每个 tile 独立过 encoder，最后 concat。LLaVA-NeXT 用这个。问题：token 数随分辨率平方爆炸，4K 图能产生 21K token。

### 4.2 Pixel Shuffle (InternVL)

把空间维度 fold 到 channel 维度：

$$\mathbf{Z}_{i,j}' = \mathrm{concat}[\mathbf{Z}_{ri+a, rj+b}]_{a,b \in \{0, \ldots, r-1\}} \in \mathbb{R}^{r^2 D} \tag{15}$$

变量：
- $r$：downsampling factor
- $\mathbf{Z} \in \mathbb{R}^{H' \times W' \times D}$：原 feature map
- $\mathbf{Z}'$：shuffle 后的 feature map
- $a, b$：sub-pixel offset in $r \times r$ block

效果：spatial dimension 缩 $r^2$ 倍，channel 维度涨 $r^2$ 倍，总信息量不变，但 token 数压缩 $r^2$ 倍。SmolVLM 用 $r=3$ 实现 9× 压缩。直觉：vision encoder 内部 feature map 相邻位置高度相关，这种 "space-to-depth" 等价于一种 learned downsampling，但保留了所有信息供 LLM 自己重组。

### 4.3 NaViT (Patch n' Pack)

把多张不同分辨率的图打包进一个 sequence：

$$L_{\mathrm{pack}} = \sum_{i=1}^k n_i, \quad n_i = \left\lfloor\frac{H_i}{P}\right\rfloor \cdot \left\lfloor\frac{W_i}{P}\right\rfloor \tag{10}$$

变量：
- $k$：一个 packed sequence 里的图像数
- $n_i$：第 $i$ 张图的 patch 数
- $H_i, W_i$：第 $i$ 张图的原始尺寸

直觉：打破 ViT 必须固定 input shape 的限制，让 encoder 直接吃 native resolution。Qwen2-VL、Ovis2.5、MiMo-VL 都用这个。

### 4.4 M-RoPE (Qwen2-VL)

标准 1D-RoPE 只编码序列位置，M-RoPE 分解为三个 component：

$$\mathbf{M}.\mathbf{RoPE}(\mathbf{x}, t, h, w) = \mathbf{x} \odot [\cos(\theta_t), \cos(\theta_h), \cos(\theta_w)] + \mathbf{x}' \odot [\sin(\theta_t), \sin(\theta_h), \sin(\theta_w)] \tag{11}$$

变量：
- $t$：temporal position ID（图像恒定，视频递增）
- $h, w$：spatial position ID
- $\theta_t, \theta_h, \theta_w$：三个维度的 rotation angles
- $\mathbf{x}'$：$\mathbf{x}$ 旋转 $\pi/2$ 后的版本（RoPE 标准操作）

直觉：text token 时三个 ID 完全相同，退化为 1D-RoPE；image token 时 $t$ 恒定，$h, w$ 反映 2D 位置；video 时 $t$ 随帧递增。这样一套 position encoding 跨 modality 通用，避免了 "图像用 2D-RoPE、文本用 1D-RoPE" 的不一致。

### 4.5 实测 token cost

| Strategy | 1MP tokens | 4MP tokens | Compute (4MP) |
|---|---|---|---|
| Fixed 336px | 576 | 576 | 1.0× |
| AnyRes (no compress) | 5,760 | 21,312 | 37× |
| AnyRes + PS(r=2) | 1,440 | 5,328 | 9.3× |
| AnyRes + PS(r=3) | 640 | 2,368 | 4.1× |
| Native + MLP(4×) | 1,332 | 5,329 | 9.3× |

直觉：从 1MP 到 4MP 没压缩的方案是 37× compute，加 Pixel Shuffle(r=3) 降到 4.1×——这个 ratio 决定了能不能实际部署 4K 输入。

## 五、Connector design：simplicity wins

### 5.1 MLP Projection (LLaVA)

$$\mathbf{H}_v = \mathbf{W}_2 \cdot \mathbf{GELU}(\mathbf{W}_1 \cdot \mathbf{Z}_v) \tag{12}$$

变量：
- $\mathbf{Z}_v \in \mathbb{R}^{N \times D_v}$：visual tokens，$N$ 个 token，$D_v$ encoder hidden dim
- $\mathbf{W}_1 \in \mathbb{R}^{D_h \times D_v}$：升维到 hidden dim $D_h$
- $\mathbf{W}_2 \in \mathbb{R}^{D_l \times D_h}$：投影到 LLM embedding dim $D_l$
- $\mathbf{H}_v \in \mathbb{R}^{N \times D_l}$：输出，token 数不变

直觉：两层 MLP 是个 "modality adapter"，把 encoder 的 representation 空间线性变换到 LLM 的 token embedding 空间。LLaVA 1.0 用单层 linear 就行，1.5 改两层 MLP 加 GELU。

### 5.2 Q-Former (BLIP-2)

$$\mathbf{H}_v = \mathrm{softmax}\left(\frac{\mathbf{Q}\mathbf{W}_Q(\mathbf{Z}_v\mathbf{W}_K)^\top}{\sqrt{D_k}}\right)\mathbf{Z}_v\mathbf{W}_V \tag{13}$$

变量：
- $\mathbf{Q} \in \mathbb{R}^{M \times D}$：$M$ 个 learnable query（通常 32）
- $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$：query/key/value projections
- $D_k$：key 维度，$\sqrt{D_k}$ 是 scaling factor
- 输出 $\mathbf{H}_v \in \mathbb{R}^{M \times D}$：固定 $M$ 个 token

直觉：固定 $M$ 个 query 通过 cross-attention 从变长 visual features 提取信息。优点是输出 token 数固定可控，缺点是 fine-grained detail 容易在 compression 中丢失。BLIP-2 时代 encoder 弱、LLM context 短，Q-Former 是必要的；现在 LLM context 长了，MLP projection 反而主流。

### 5.3 实证结论

MM1 的 ablation 显示 connector 复杂度对最终性能影响小，simple MLP 在控制其他变量时与复杂设计持平。这反向证明了：**encoder representation quality 才是 VLM 能力的主要决定因素**，connector 只是 "格式转换器"。

参考：https://arxiv.org/abs/2403.18500 (Eagle)
参考：https://arxiv.org/abs/2409.11402 (NVLM)

## 六、Multi-encoder fusion：互补性假设

Cambrian-1 用四个 encoder 融合：CLIP ViT-L/14@336、SigLIP-SO400M/14@384、ConvNeXt-XXL@1024、DINOv2 ViT-L/14@518。

融合公式：

$$\mathbf{F}_{\mathrm{fused}} = \mathrm{Aggregate}(\mathbf{F}_1, \mathbf{F}_2, \ldots, \mathbf{F}_K; \mathbf{Q}) \tag{18}$$

变量：
- $\mathbf{F}_k$：第 $k$ 个 encoder 输出的 feature map
- $\mathbf{Q}$：learnable query tokens（用于 cross-attention 聚合）
- $\mathrm{Aggregate}(\cdot)$：可以是静态 fusion（SVA）或动态 routing（Mixture-of-Encoders）

直觉：每个 encoder 学到的 inductive bias 不同——CLIP 偏 image-level semantic、SigLIP 偏 text-aligned、DINOv2 偏 spatial geometry、ConvNeXt 偏 multi-scale texture。融合后的 representation 比任何单一 encoder 都强。**代价**：约 4.3× FLOPs。SCOPE 用 dynamic routing 把这个开销降到 24–49%。

更激进的路线是 distillation：AM-RADIO 把 CLIP + DINOv2 + SAM 蒸馏进一个 student，single-encoder cost 拿到 multi-teacher 的 diversity。Nemotron Nano V2 用 c-RADIOv2-VLM-H 作 encoder。

参考：https://arxiv.org/abs/2406.16860 (Cambrian-1)
参考：https://arxiv.org/abs/2510.12974 (SCOPE)
参考：https://arxiv.org/abs/2312.06685 (AM-RADIO)

## 七、Encoder-free：去掉 vision encoder 行不行？

Fuyu-8B 直接把 patch 投进 LLM：

$$\mathbf{h}_i = \mathbf{W}_{\mathrm{patch}} \cdot \mathrm{flatten}(\mathbf{x}_p^i) + \mathbf{e}_{\mathrm{pos}}^i \tag{19}$$

变量：
- $\mathbf{x}_p^i$：第 $i$ 个 image patch
- $\mathbf{W}_{\mathrm{patch}}$：patch-to-embedding 投影矩阵
- $\mathbf{e}_{\mathrm{pos}}^i$：第 $i$ 个 patch 的 position embedding
- $\mathbf{h}_i$：直接进 LLM 的 embedding

直觉：完全跳过 vision-specific pretrain，让 LLM 在 VLM 训练时自己学 visual perception。EVE / EVEv2 加 vision-centric supervision 让这个路线 competitive。SAIL 的 scaling 分析显示：encoder-free model 在足够大时能 match modular MLLM，但 training compute 显著更高。

Chameleon / Emu3 走另一极端：用 VQ-VAE 把图像 tokenize 成离散 codebook index，跟 text token 完全统一处理。这套路线的代价是 visual fine-grained 信息损失——VQ codebook 大小固定，detail 受 codebook 容量限制。

参考：https://www.adept.ai/blog/fuyu-8b
参考：https://arxiv.org/abs/2406.11838 (EVE)
参考：https://arxiv.org/abs/2504.10462 (SAIL)
参考：https://arxiv.org/abs/2405.09818 (Chameleon)

## 八、PTP：instruction-guided token pruning

PTP 用 top-down instruction attention + bottom-up visual saliency 联合决定哪些 visual token 重要：

$$c_j = \max_{q \in \mathcal{Q}}\mathrm{Attn}_{q \to j} \tag{16}$$

变量：
- $\mathcal{Q}$：instruction token 的 index 集合
- $\mathrm{Attn}_{q \to j}$：LLM 早期层中 instruction token $q$ 对 visual token $j$ 的 attention score
- $c_j$：visual token $j$ 的 instruction relevance

$$s_j = \alpha c_j + (1-\alpha)b_j \tag{17}$$

变量：
- $b_j$：来自 vision encoder 中间层的 visual saliency
- $\alpha \in [0, 1]$：平衡系数
- $s_j$：综合 importance score

经验：$\alpha = 0.5$ 通用任务，OCR 任务偏 small $\alpha$（靠 visual saliency），open-domain reasoning 偏 large $\alpha$（靠 instruction guidance）。PTP 实现 50% token pruning 几乎不损精度，有时还提升（filter 掉 noisy token）。

直觉：LLM 早期 attention 已经透露 "用户关心图像哪个区域"，把这些信号反馈到 visual token 选择上，相当于在 encoder output 之后做 attention-guided pooling。

参考：https://arxiv.org/abs/2509.15704 (PTP)

## 九、Cost-performance trade-off 实测

Table 5 是这篇 survey 最 actionable 的数据点：

| Configuration | Params | GFLOPs | VRAM | ImageNet | VQAv2 |
|---|---|---|---|---|---|
| CLIP ViT-B/16 | 86M | 17.6 | 0.4GB | 68.3 | 76.8 |
| CLIP ViT-L/14 | 304M | 81.1 | 1.2GB | 75.5 | 79.2 |
| SigLIP-B/16 | 93M | 17.9 | 0.4GB | 78.4 | 78.1 |
| SigLIP-SO400M | 400M | 95.8 | 1.5GB | 83.2 | 81.7 |
| SigLIP 2 S0400M | 400M | 98.3 | 1.6GB | 84.1 | 82.4 |
| InternViT-300M | 304M | 82.6 | 1.3GB | 79.8 | 80.3 |
| InternViT-6B | 5.9B | 1,547 | 24GB | 88.2 | 82.1 |
| DINOv2-L | 304M | 81.6 | 1.2GB | 86.3 | 73.2† |
| SigLIP+DINOv2-L | 704M | 177 | 2.7GB | – | – |
| Cambrian (4 enc) | 2.1B | 412 | 5.8GB | – | 77.8 |

关键观察：
1. **SigLIP-SO400M → SigLIP 2**：FLOPs 几乎不变（95.8→98.3），ImageNet +0.9，VQAv2 +0.7。Training methodology 的边际收益
2. **SigLIP 2 (400M) vs InternViT-6B**：FLOPs 差 16×，ImageNet InternViT 高 4.1 分，但 VQAv2 SigLIP 2 反超 0.3 分。**ImageNet 不能预测 VLM 性能**
3. **DINOv2**：ImageNet 86.3 极高，但 VQAv2 只有 73.2（需 text-aligned fine-tune）
4. **Multi-encoder**：Cambrian 4-enc 比 SigLIP 单 encoder 多 4× FLOPs，VQAv2 反而低（77.8 < 82.4），但 vision-centric benchmark 强

这些数字直接回答 "training vs scale" 的问题：6B encoder 把 ImageNet 推到 88.2 但 VLM 上的边际收益几乎消失。LLM 和 connector 才是瓶颈，vision encoder 的 representation 在 400M 量级就够用。

## 十、Encoder sensitivity 的 task 维度

Table 6 给出 encoder choice 对不同 benchmark 的影响程度：

**High sensitivity**：
- Document 类：DocVQA, OCRBench, ChartQA, InfoVQA, MMLongBench-Doc
- Spatial 类：RefCOCO/+/g, RealWorldQA, TallyQA
- Fine-grained：BLINK
- Video：Video-MME

**Medium sensitivity**：
- 综合：MMBench, MME, MMMU, MathVista, AI2D, POPE, MMStar, SEED-Bench

**Low sensitivity**：
- 通用 VQA：VQAv2, GQA
- 知识问答：ScienceQA

直觉：VQAv2 / GQA 已经 saturated，CLIP ViT-L/14 跟 SigLIP 2 之间可能只差 1–2 分。但 OCRBench、TallyQA 这些任务换 encoder 能差 100+ 分。**如果你的应用是 document understanding 或 spatial reasoning，encoder 选择是主要变量；如果是普通 VQA，随便选个 SigLIP-SO400M 就行**。

## 十一、五个 Research Question 的答案

❶ **哪种 paradigm 最好**：contrastive with modern improvements（sigmoid loss + multilingual + dense objectives）的 SigLIP 2 是 general-purpose 最优；self-supervised 在 spatial task 强；LLM-aligned 在深度 VLM 整合上强但 cost 高。

❷ **scale 何时重要**：在 400M 以下 scale 有边际收益，超过 400M 后 LLM 和 connector 成为瓶颈，继续 scale encoder 几乎不提升 VLM 性能。Training methodology 改进（loss function、data curation、feature objectives）的 ROI 远高于 scale 一个 order of magnitude。

❸ **变分辨率怎么处理**：在 encoder 层处理（NaViT + M-RoPE）比预处理 resize / 后处理 tile 更好。Native resolution 保留的信息预处理损失后无法恢复——这是 document 任务的关键。

❹ **multi-encoder 何时胜出**：需要同时 strong semantic + strong spatial 时。Single encoder 在 vision-centric benchmark 上有 hard ceiling，融合 complementary encoder 能突破。代价是 4× FLOPs，需要根据 ROI 决定。

❺ **未来轨迹**：specialized encoder vs encoder-free 还在博弈。Encoder-free（Fuyu、EVE、SAIL）证明可行但 compute cost 高；specialized encoder 在 efficiency 上仍有优势。Chameleon / Emu3 的 unified tokenization 是另一条路。**这个问题的答案取决于 LLM 能不能从 raw pixel 高效学 visual perception——目前看可以，但效率不如 pretrained encoder**。

## 十二、给 practitioner 的决策表

| Application | Recommended Encoder | Rationale |
|---|---|---|
| General-purpose VLM | SigLIP 2 S0400M | Best training methodology, multilingual, dense features |
| Document understanding | NaViT-style / DeepEncoder | Native resolution preserves text details |
| Spatial reasoning | DINOv2 + SigLIP fusion | SSL features complement contrastive |
| Resource-constrained | SigLIP 2 Base/Large | 86M–303M params, 维持质量 |
| Maximum capability | Multi-encoder (Cambrian-style) | Captures complementary features |
| Research / flexibility | InternViT variants | Open weights, well-documented |

## 十三、深层 intuition

这篇 survey 真正反直觉的是 **scaling law 在 vision encoder 这里失效**。LLM 上 10× params 一般带稳定收益，vision encoder 上 10× params（CLIP-L → InternViT-6B）只在 ImageNet 这种 standalone benchmark 上显著，集成进 VLM 后收益消失。可能的解释：

1. **Vision encoder 的 representation 在 400M 量级就足够 expressive**——视觉信息熵有限，自然图像的 category + spatial structure 不需要 6B 参数来表达
2. **LLM 是 downstream bottleneck**——visual feature 通过 connector 进 LLM 后，LLM 的 reasoning capacity 决定 ceiling。Encoder 学得再好，LLM 不会用也白搭
3. **Connector 是 information bottleneck**——MLP / Q-Former 把 $D_v$ 维投到 $D_l$ 维，这个 projection 限制了 information flow。Encoder 增大但 connector 维度不变，多余 capacity 被丢失
4. **Benchmark saturation**——VQAv2 / GQA 已经饱和，encoder 改进体现不出来。Vision-centric benchmark 上 scale 仍有收益

这跟 LLM 的 scaling law 差异提示了一个更深的判断：**vision encoder 跟 LLM 不是同一种 system**。LLM 的 scaling law 来自 next-token prediction 这个 "无限 difficulty" 的 task，模型越大越好；vision encoder 学的是 finite-entropy 的 visual representation，过了某个 threshold，extra capacity 学不到新东西。

第二个反直觉点是 **encoder-free 路线的 viability**。SAIL 显示 encoder-free 模型 scale 上去后能 match modular VLM，但 training compute 显著高。这暗示 pretrained vision encoder 主要价值是 "compute-efficient initialization"，不是 "irreplaceable inductive bias"。如果未来训练 compute 继续变便宜，encoder-free 的 unified architecture 可能成为主流——vision encoder 退化成 LLM 的 "warm start" 而非独立模块。

第三个点是 **document understanding 是 encoder 选择的风向标**。所有 general VLM 在 VQAv2 上都 80+，但 OCRBench 从 600 到 850 都有，差距极大。**encoder 的差异化主要体现在 "需要 dense spatial reasoning" 的任务上**，因为这些任务的信息不能靠 language prior 补救。这跟 OCR、document AI、screen agent 等实际应用高度相关——这些场景下选错 encoder 等于选错天花板。

最后一点是 **multi-encoder fusion 的本质是 inductive bias ensemble**。CLIP 的 inductive bias 是 "image-text alignment"、DINOv2 的是 "visual geometry invariance"、SAM 的是 "segmentation structure"——每个 encoder 在 pretrain 时注入不同的 prior。融合相当于让 LLM 同时访问这些 prior。跟 LLM ensemble 不同，这是 **representation-level ensemble**，更细粒度。distillation 路线（AM-RADIO）把这个 ensemble 压回 single-encoder cost，是工程上最实用的方向。

## 参考链接汇总

- CLIP: https://arxiv.org/abs/2103.00020
- SigLIP: https://arxiv.org/abs/2303.15343
- SigLIP 2: https://arxiv.org/abs/2502.14786
- DINOv2: https://arxiv.org/abs/2304.07193
- DINOv3: https://arxiv.org/abs/2508.10104
- InternVL: https://arxiv.org/abs/2312.14238
- NaViT: https://arxiv.org/abs/2307.06304
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- LLaVA: https://arxiv.org/abs/2304.08485
- BLIP-2: https://arxiv.org/abs/2301.12597
- Flamingo: https://arxiv.org/abs/2204.14198
- Cambrian-1: https://arxiv.org/abs/2406.16860
- Eagle: https://arxiv.org/abs/2407.09437
- AM-RADIO: https://arxiv.org/abs/2312.06685
- Fuyu-8B: https://www.adept.ai/blog/fuyu-8b
- EVE: https://arxiv.org/abs/2406.11838
- SAIL: https://arxiv.org/abs/2504.10462
- Chameleon: https://arxiv.org/abs/2405.09818
- PTP: https://arxiv.org/abs/2509.15704
- MetaCLIP 2: https://arxiv.org/abs/2507.22062
- NVLM: https://arxiv.org/abs/2409.11402
- TULIP: https://arxiv.org/abs/2503.15485
- MM1: https://arxiv.org/abs/2403.18500

这篇 survey 真正的价值是建立了 vision encoder 的 **taxonomy + empirical benchmark + selection guide** 三位一体的框架。对研究者，它指明 training methodology 是高 ROI 方向；对工程师，它说 "default to SigLIP 2 + native resolution，只在 document/spatial 任务才考虑 multi-encoder"。Vision encoder 在 VLM 里看似配角，但这个配角的选择直接决定 vision-centric 任务的天花板——这是 2026 年 VLM 设计的关键 trade-off。
