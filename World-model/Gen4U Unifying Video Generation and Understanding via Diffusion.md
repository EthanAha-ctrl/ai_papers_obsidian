---
source_pdf: Gen4U Unifying Video Generation and Understanding via Diffusion.pdf
paper_sha256: fb79f24b7f8c3593bd09e2f387b953b8c7e508fb5ad025893d1669b640aa3e83
processed_at: '2026-08-19T09:05:44-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 聊聊 Gen4U 这篇 paper

## 一、一句话先说核心

它干了一件挺野的事：**直接拿一个 video diffusion 生成模型（Veo3），frozen 不动，从中间抽一层 activation 出来，发现这玩意儿当 video encoder 用，在分类、深度、位姿、captioning 上能打到 SOTA 或接近 SOTA**。一个生成模型，免费附送一个理解模型。

这事的反直觉之处在于：**以前大家都觉得 diffusion 模型只擅长"画图"，不擅长"理解"**。早期有人拿 WALT diffusion 试过 [Vélez et al. 2025]，结论是低层几何（深度、位姿）还行，高层语义一塌糊涂。Gen4U 这篇说：等一下，你们试的 diffusion 太弱了，等它生成能力变强了（Veo3 这种规模），语义理解是**自然涌现**的。

paper 链接：https://arxiv.org/abs/2507.13249（Vélez et al.）

---

## 二、为啥这事"现在"才做出来

要 build intuition，得先说清楚**为啥以前不行、现在行了**。

Diffusion 模型本质是学"从噪声还原信号"。早期 image diffusion（比如 Stable Diffusion 1.x）学的主要是"怎么把高频纹理填回去"，所以中间 feature 基本是 texture-level 信息。但 video diffusion 不一样，它得学**物体怎么动、物理约束怎么变、part-object 关系怎么演化**——因为不学这些，生成的视频就会"穿模"、"扭曲"、"动作不连贯"。

换句话说，生成 temporally consistent video 这个任务本身，**就逼着模型 implicitly 学了 world model**。这跟 Rao & Ballard 1999 的 predictive coding 假说一脉相承——大脑皮层的认知机制可能就是"预测"。你让模型预测下一帧怎么变，它就必须理解"杯子被推倒了会滚动"这种常识。

- Rao & Ballard 1999: https://www.nature.com/articles/nn0199_79
- Wiedemer et al. 2025 (Veo3 是隐式 world model): https://arxiv.org/abs/2509.20328

所以 Gen4U 的 narrative 是："生成模型变强 → 它被迫学了 world dynamics → 这些 dynamics 沉淀在中间层 activation 里 → 抽出来就是好 representation"。这是个 scaling 涌现故事，跟 LLM 的 in-context learning 涌现是同一类现象。

---

## 三、Diffusion 到底在干啥（把数学拆开）

### 3.1 Latent Diffusion 的 forward process

paper 3.1 节给的公式：
$$z_t = \sqrt{\bar{\alpha}_t}\, z_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$

逐项拆解：
- $z_0 = E(\boldsymbol{x})$：原始 video $\boldsymbol{x}$（shape 是 $F \times H \times W \times C$，F=帧数）被 VAE encoder 压成低维 latent
- $t \in [0, T]$：noise step，t 越大表示加的噪声越多
- $\bar{\alpha}_t$：累积 noise schedule 系数。t=0 时 $\bar{\alpha}_t = 1$（纯信号），t=T 时 $\bar{\alpha}_t \to 0$（纯噪声）
- $\epsilon \sim \mathcal{N}(0, I)$：标准高斯噪声
- $z_t$：被加噪后的 latent

**人话**：把一张 latent 图像和一团高斯噪声按比例混合，比例由 t 决定。t 越大噪声越多。

### 3.2 训练目标

$$\mathcal{L}_{LDM} = \mathbb{E}_{z_0, \epsilon, t}\left[\|\epsilon - f_\theta(z_t, t, c)\|_2^2\right]$$

- $f_\theta$：DiT backbone（transformer stack，L 层 block）
- $c$：condition，比如 text embedding
- 模型输入 $(z_t, t, c)$，输出对 $\epsilon$ 的预测，loss 是 MSE

**人话**：给模型看一张被加噪的图，告诉它"这是 t 时刻加噪后的状态，你猜我加了什么噪声"。模型学的是 denoising vector field。

### 3.3 Flow Matching 变体（Wan 2.2 用的）

$$z_t = (1-t)\,z_0 + t\,\epsilon$$
$$f_t = \frac{dz_t}{dt} = \epsilon - z_0$$
$$\mathcal{L}_{FM} = \mathbb{E}_{z_0, \epsilon, t}\left[\|f_\theta(z_t, t, c) - (\epsilon - z_0)\|_2^2\right]$$

变量含义同上。差别只是参数化方式：DDPM 预测"加了什么噪声"，Flow Matching 预测"从数据指向噪声的方向向量"。两者学的本质是同一个 denoising vector field，只是坐标系不同。

参考 Flow Matching 原文：https://arxiv.org/abs/2210.02747

---

## 四、主角登场：$h_t^{(l)}$

这是整篇 paper 最关键的中间变量。定义：

$$h_t^{(l)} \in \mathbb{R}^{F' \times H' \times W' \times D}$$

- 上标 $(l)$：第 $l$ 层 transformer block 的输出（深度维度）
- 下标 $t$：在 noise level $t$ 处 forward 一次
- $F', H', W'$：latent 空间的时空分辨率（已被 VAE 压过）
- $D$：channel 维度

**关键 insight**：标准 diffusion 推理要跑 T 步（30-50 步 iterative denoising）才出视频。但 paper 发现——**只要找对 (l, t) 组合，跑一次 forward 抽出来的 $h_t^{(l)}$ 已经是极好的 video representation**。这就是 Gen4U 名字（Generation for Understanding）的来历：把生成路径的中间产物复用做理解。

---

## 五、Latent Space 的三个反直觉发现

这是 paper 第 3 节的精华。作者用一堆 probe（zero-shot + trained）去系统 map 不同 (l, t) 点的 representation 性质，得出三个发现。

### 5.1 发现一：Veo3 的 Bimodal Depth Pattern

Figure 3 左是 Veo3 vs Gemma-2-9b-it 文本编码器的 mutual k-NN alignment 热图。横轴 noise level，纵轴 depth。

**出现两个峰**：
- 浅层峰（~25% depth）
- 深层峰（~70-80% depth）
- 中间有一个明显的 dip

而 Wan 2.2 是经典单峰（peak 在 25%）。这种 bimodal 现象在 diffusion 文献里**第一次报告**。

paper 把它跟 [Valeriani et al. 2023] 的工作类比——大型 transformer 的 representation manifold 几何会经历"先膨胀-再收缩-再膨胀"过程：浅层 manifold 扩张吸收信息，中层收缩做 information routing，深层再扩张做最终抽象。

- Valeriani et al. geometry of transformer reps: https://arxiv.org/abs/2302.02225

**为啥这事重要**：以前大家认为 diffusion 的 representation 是 monotonic 的——越深越接近 pixel 输出，越浅越接近 latent 抽象。这个 bimodal 模式说明大型 diffusion model 内部有更复杂的 information routing 结构，跟 discriminative transformer 越来越像。这暗示"generation model"和"understanding model"的内部计算结构正在收敛。

### 5.2 发现二：60% Noise 是 Semantic Bottleneck

无论 Veo3 还是 Wan 2.2，无论对 text encoder 还是 visual encoder（DINOv2, VideoMAEv2），alignment 峰值都稳定落在 **t ≈ 60%**。

这个一致性非常强。paper 把它叫做 **semantic bottleneck**。

**直觉解释**（这是我自己 build intuition 的方式）：
- t=90%（高 noise）：信号几乎被噪声淹没，representation 反映的是 model 的 prior（"一个视频大概长啥样"），缺乏 sample-specific 语义
- t=10%（低 noise）：representation 已经在重构 pixel-level 细节，语义被分散到局部 patch 里
- t=60%（中间）：刚好信号和噪声平衡，representation 既保留了 sample-specific 全局语义，又没沉到像素级

这跟图像频率域的"低频携带全局结构、高频携带细节"是同一个道理。Sander Dieleman 的 "Diffusion is spectral autoregression" 假说正好对应这个：denoising 过程本质是按频率从低到高 autoregressive decoding。

- Dieleman blog: https://sander.ai/2024/09/02/spectral-autoregression.html

### 5.3 发现三：Linear Probe vs Attention Probe 的 Shift

Figure 5 在 SSv2 上做了两种 probe：
- Linear probe：global pooling + linear projection（参数 ~1M）
- Attention probe：cross-attention with learnable queries（参数 ~6M）

最优位置出现 shift：
- Linear 最优：t ≈ 60%, depth ≈ 70%
- Attention 最优：t ≈ 30%, depth ≈ 80%

**核心 insight**：低 noise level 的语义信息**没有丢失**，只是被空间 scattered 到局部 patch。Linear pooling 太简单抓不到，需要 attention 这种 dynamic spatiotemporal pooling 来 aggregate。

paper 做了个 control：如果只是参数量驱动，attention 应该在所有 noise level 都 dominate。但 t=90% 时 attention 也只有 15-30%，跟 linear 差不多。这说明 attention 的优势来自**机制**（能动态聚合空间分散的信息），而不是容量。

**为啥这事 important**：它打破了"diffusion 生成质量 = pixel reconstruction quality"的迷思。低 noise 的 representation 看起来离 pixel 近，但语义信息其实**更深地嵌在 token 间的关系结构里**，要 attention 才能挖出来。

---

## 六、Mutual k-NN Alignment 公式拆解

这是 paper Appendix A 的核心 metric，zero-shot 不需要训练。完整公式：

$$\mathcal{A}_{\text{MkNN}}(X, Y) = \frac{1}{kN} \sum_{i=1}^N \sum_{j=1}^N (M^X \odot M^Y)_{ij}$$

变量：
- $N$：dataset 里 video 数量（paper 用 1024）
- $k$：邻居数（paper 用 10）
- $X \in \mathbb{R}^{N \times D}$：diffusion 模型抽出的 video embedding 矩阵，每行是一个 video 的 AvgPool($h_t^{(l)}$)
- $Y \in \mathbb{R}^{N \times D'}$：reference encoder（比如 Gemma 文本编码器）抽出的 embedding
- $M^X_{ij} = 1$ 当且仅当 $j$ 是 $i$ 在 $X$ 空间的 k-NN
- $M^Y_{ij}$ 同理
- $\odot$：element-wise 乘法

**人话翻译**：对每个 video $i$，看它在 diffusion embedding 空间的 10 个最近邻，和它在 text embedding 空间的 10 个最近邻，有多少个重叠。所有 video 取平均。

这个 metric 之所以巧妙：
- **对维度和 scale 不变**：只看 rank-order，所以 diffusion 的 D=8192 和 Gemma 的 D=3072 可以直接比
- **零训练**：不需要 learnable mapping
- **有意义**：如果两个空间把"语义相近"的 video 都聚类到一起，alignment 就高

参考 Platonic Representation Hypothesis：https://arxiv.org/abs/2405.10354

---

## 七、实验结果——为啥每个都重要

### 7.1 SSv2 Video Classification（Table 1）

| Model | Pre-training | Size (M) | Top-1 (%) |
|-------|-------------|----------|-----------|
| VideoMAEv2-g | MAE | 1013 | 65.6 |
| VideoPrism-g | MAE+Contrastive | 1113 | 65.4 |
| 4DS-j | MAE | 21495 | 68.2 |
| InternVideo2 | MAE+Contrastive+Caption | 6000 | 67.7 |
| V-JEPA-H | Masked feature pred | 635 | 72.2 |
| V-WALT | Diffusion (旧) | 1900 | 59.7 |
| **Gen4U (Veo3)** | Diffusion (新) | - | **71.3** |
| **Gen4U + aug** | Diffusion (新) | - | **72.6** |

**关键 narrative**：
- V-WALT（早期 diffusion）只有 59.7% —— 印证了 Vélez et al. 的负面结论
- Veo3（新 diffusion）直接 71.3%，超越所有 MAE 和 contrastive 方法
- 加 activation-level augmentation 后 72.6%，跟 V-JEPA 持平

这个对比的杀手锏是：**V-WALT 和 Veo3 用的是同一类方法（diffusion），结果差了 12 个点。差别只在生成能力**。这就直接论证了核心 thesis——**representation quality 跟生成能力一同涌现**。

**工程 trick**：augmentation 不在 raw video 上做（太贵），直接在 Veo3 抽出的 activation 上做。具体是：
- Temporal masking 16.7% 的 frames 设为 0
- Attention dropout 40%
- Label smoothing 0.2

这是 frozen encoder 范式的独特优势——feature 已经 cached，在 latent 空间 augment 几乎免费。

### 7.2 Captioning（Table 2）

| Model | SSv2 CIDEr | COCO CIDEr | VATEX CIDEr |
|-------|-----------|-----------|-------------|
| SigLIP-so400m/14 | 204.5 | **118.5** | **66.0** |
| SigLIP2-B/16 | 198.6 | 114.2 | 58.4 |
| Gen4U @ 30% | **289.5** | 54.9 | 44.8 |
| Gen4U + Noise Aug. | 280.4 | 69.3 | 56.7 |
| Gen4U + Noise Aug. + High res | - | 102.0 | - |

**反差很有意思**：
- SSv2（action 描述）Gen4U 大幅领先（289.5 vs 204.5，+85）
- COCO（静态图像多小物体）Gen4U 较弱（54.9 vs 118.5）
- 加 noise aug + 高分辨率后 COCO 追到 102.0，还差一点

**为啥这个反差 make sense**：
- SSv2 依赖时序运动理解——这是 video diffusion 强项，Veo3 内部就学了 dynamics
- COCO 是一堆小物体静态描述——Veo3 在低分辨率 latent 空间丢失小物体细节
- SigLIP 是 explicitly 训练做 vision-language alignment，对 captioning 有 inductive bias

**Frozen LLM 实验**（Vatex 列）：只训 cross-attention adapter，LLM 完全冻结，性能下降很小（44.8 → 40.2），且远超 VideoPrism 的 31.7。这证明 Veo3 representation 本身就蕴含足够语义，LLM 协同微调只是锦上添花。

### 7.3 Depth Estimation（Figure 6 左）

ScanNet 数据集，DPT head（23M 参数），frozen Veo3。
- 最优 (depth 80%, noise 30%) 处 AbsRel = 0.075
- 比 4DS frozen-feature baseline 提升 10.7%

AbsRel 公式：
$$\text{AbsRel} = \frac{|d^* - d|}{d + \epsilon}$$

- $d^*$：predicted depth
- $d$：ground-truth depth  
- $\epsilon$：避免除零的小常数

这是 ScanNet depth 上 frozen video model feature 的最佳结果。说明 Veo3 内部确实学了 3D 几何结构——生成视频时为了让物体不"飘"或"穿模"，必须 implicitly 学会 depth。

### 7.4 Camera Pose Estimation（Figure 6 右）

预测 6DoF 相对位姿（第一帧和最后一帧之间）。表示为 12D 向量：3×3 rotation matrix + 3×1 translation。

结果：Gen4U 1.10 EPE，与 DINOv2 baseline 1.08 EPE 持平。

**意义**：DINOv2 是专门做 geometric representation 的 SOTA 模型，Veo3 跟它打平。这说明生成视频时为了让相机运动一致，模型 implicitly 学了 camera pose 信息。

---

## 八、Appendix B 的彩蛋：跨 Block/Noise Feature 组合

paper 试了从 12 个 block × 4 个 noise level（10%/30%/60%/90%）抽 48 个 feature vector，用不同 adapter 融合。四种 adapter：

1. **Linear adapter**：$F_{combined} = \sum_{i=1}^{48} w_i \cdot F_i$，只有 48 个标量参数
2. **Shared MLP**：48→M→H→1 的 MLP 共享跨 channel 维度
3. **Self-attention**：CLS token + 48 个 feature 做自注意力
4. **Cross-attention**：Perceiver-style 单 query cross-attend

训练目标两个：
- Cross-entropy 分类
- Multi-positive InfoNCE：
$$\mathcal{L}_{InfoNCE} = -\frac{1}{B}\sum_{i=1}^B \log \frac{\sum_{j \in \mathcal{P}(i)} \exp(\sin(\boldsymbol{v}_i, \boldsymbol{v}_j)/\tau)}{\sum_{k=1}^B \exp(\sin(\boldsymbol{v}_i, \boldsymbol{v}_k)/\tau)}$$

变量：
- $B$：batch size
- $\boldsymbol{v}_i$：adapter 输出（L2 normalized）
- $\sin(\cdot, \cdot)$：cosine similarity
- $\tau$：temperature
- $\mathcal{P}(i)$：text feature 空间的 k-NN 作为多正样本

**最反直觉的发现**：

> 训练 adapter 做 classification，比训练 adapter 做 text alignment，**反而**更能提升 text alignment metric。

paper 自己的解释：discrete cross-entropy 提供更干净稳定的 gradient，比 multi-positive contrastive 的 noisy gradient 更能学到 semantic 结构。

**这跟我（Karpathy）自己常说的直觉完全吻合**：supervised discrete signal 比 self-supervised contrastive 更 sample-efficient。Gen4U 这个 finding 在 representation learning 语境下再次验证了这一点。

---

## 九、Appendix C 的另一个反直觉：Preview Decoding

paper 还试了把 $h_t^{(l)}$ 通过 linear/attention head decode 回 RGB video，看哪个 (l, t) 点 reconstruction MSE 最低。

**结果**：**这个 metric 没有清晰 sweet spot**！跟 alignment 和 classification probe 的双峰模式完全不同。MSE 随 depth 单调下降（深层离 RGB 输出近），随 noise 变化不显著。

**重要启示**：representation quality 高度依赖 probe 类型。pixel reconstruction 不能反映 semantic quality——这跟 MAE 文献的观察一致，pixel loss 容易被 low-frequency 主导，掩盖语义差异。

参考 MAE 原文：https://arxiv.org/abs/2111.06377

---

## 十、大局观——这事为啥可能改变方向

### 10.1 Generation 与 Understanding 的统一

到现在为止，社区里有几条路线：
- **MAE 路线**（VideoMAEv2, 4DS, D4RT）：pixel reconstruction，强 geometry 弱 semantic
- **Contrastive 路线**（CLIP, SigLIP, DINO）：强 semantic 弱 geometry
- **V-JEPA 路线**（latent prediction）：在 latent 空间预测未来，性能强但 pipeline 复杂
- **Hybrid 路线**（VideoPrism, InternVideo2）：MAE + contrastive，平衡但妥协

Gen4U 给出了第五条路：**直接用最强 generative 模型，理解能力免费搭车**。

这件事的深层意义在于——**它把 foundation model 的成本从"训练 N 个 specialized encoder"压到"训练 1 个 generative model，下游全 hook 轻量 decoder"**。工业上影响巨大。

### 10.2 跟 V-JEPA 的对称性

V-JEPA 在 latent 空间预测未来 [Bardes et al. 2024]，SSv2 上 72.2%。Gen4U 用 diffusion 预测噪声，72.6%。**两个数字几乎一样**。

这暗示一个深层结论：**prediction is prediction**，不管是 latent 还是 pixel 空间，只要模型被逼着预测未来的某种 aspect，就会涌现相似的 representation geometry。这跟 Rao & Ballard 1999 的 predictive coding 假说完全自洽。

- V-JEPA 2: https://arxiv.org/abs/2506.09985
- V-JEPA 2.1: https://arxiv.org/abs/2603.14482

### 10.3 跟 Transfusion 的互补

Transfusion [Zhou et al. 2025] 把 next-token prediction（语言理解）和 diffusion（生成）合到一个 model，结论是 understanding 帮 generation。
Gen4U 反过来证明：generation 已经 implicit 学了 understanding。

两个方向合起来：**generation 和 understanding 是同一枚硬币的两面，足够大的模型会自然统一它们**。这呼应了 Platonic Representation Hypothesis [Huh et al. 2024]——不同 modality、不同目标训练的大模型最终趋向相似的 representation manifold。

- Transfusion: https://openreview.net/forum?id=SI2hI0frk6
- Platonic Rep Hypothesis: https://arxiv.org/abs/2405.10354

### 10.4 跟 LLM in-context learning 的对称

Gen4U 的核心 finding——"scaling 让 implicit capability 涌现"——跟 LLM 的 in-context learning 涌现是同构现象。GPT-3 之前没人觉得 next-token prediction 能做 few-shot reasoning；Veo3 之前没人觉得 diffusion 能做 semantic understanding。**两件事的教训都是：objective function 的表面形态不重要，关键是 prediction signal 够不够强、模型够不够大**。

---

## 十一、Limitations 和开放问题

paper 自己承认：
- 主实验在 proprietary Veo3 上，reproducibility 受限
- Wan 2.2 复现了 qualitative 发现但效果弱很多——说明 finding 跟 scale 强绑定
- Bimodal depth pattern 的机制完全没解释

**我自己（站在 Karpathy 视角）会想追的方向**：

1. **训练 dynamics 中 60% noise bottleneck 是何时涌现的？** 跟 FID 下降是否同步？这能告诉我们 representation quality 和 generation quality 的因果关系——是 representation 好 → 生成好，还是反过来，还是共同由某个第三因素驱动？

2. **Bimodal pattern 的来源**：是 Veo3 特有的训练策略（比如 classifier-free guidance 的某种 side effect），还是大型 DiT 的 universal 现象？做个 ablation：用相同架构但不同训练目标，看 bimodal 是否消失。

3. **跟 LLM scaling law 类比**：representation quality 跟 generative capability 是不是 power-law 关系？这能预测下一代 diffusion model 的 understanding 性能上限。

4. **Cross-attention adapter 那 24.9% vs linear 的 21.9%**：差距虽然 3 个点，但 training 复杂度大得多（需要特殊 batching 16 anchor × 10 positive × 22 negative）。这说明 Veo3 的 representation 已经 structured 到 linear 可分，复杂 adapter 的边际收益在递减——这是 representation 极度成熟的好信号。

5. **多 noise level feature 融合**：Appendix B 只做了初步探索。如果像 Diffusion Hyperfeatures [Luo et al. 2023] 那样做 U-Net 跨尺度 fusion，能否进一步提升？特别是用 cross-attention adapter 把 48 个 feature 看作 "tokens"，让 LLM 直接处理——可能是个有趣方向。

- Diffusion Hyperfeatures: https://arxiv.org/abs/2305.16843

---

## 十二、一句话总结

Gen4U 这篇 paper 的核心叙事：**当 video diffusion model 足够强时，它内部 frozen 中间层 activation 就已经是 general-purpose video encoder，无需 fine-tuning 就能在 classification、depth、pose、captioning 上同时达到 SOTA 或接近 SOTA**。

它没改 Veo3 一行代码，只是从中间抽了一层。这事的意义不在 Gen4U 这个 framework 本身，而在于它揭示了——**generation 与 understanding 不是两件事，是同一件事的两个 readout head**。下一代 visual foundation model 可能就是一个超大 diffusion transformer，下游任务全 hook 轻量 decoder。

参考链接汇总：
- Gen4U paper (Vélez et al. 2025 prior work): https://arxiv.org/abs/2507.13249
- Veo3 tech report: https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf
- Wan 2.2: https://arxiv.org/abs/2503.20314
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- Transfusion: https://openreview.net/forum?id=SI2hI0frk6
- Platonic Representation Hypothesis: https://arxiv.org/abs/2405.10354
- Predictive coding (Rao & Ballard 1999): https://www.nature.com/articles/nn0199_79
- Diffusion is spectral autoregression (Dieleman): https://sander.ai/2024/09/02/spectral-autoregression.html
- Geometry of transformer reps (Valeriani et al.): https://arxiv.org/abs/2302.02225
- Wiedemer et al. Veo3 as world model: https://arxiv.org/abs/2509.20328
- DINOv2: https://arxiv.org/abs/2304.07193
- VideoMAEv2: https://arxiv.org/abs/2303.16755
- 4DS: https://arxiv.org/abs/2412.15212
- D4RT: https://arxiv.org/abs/2506.07758
- Track4Gen: https://arxiv.org/abs/2501.17668
- Genie 2 world model: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- Diffusion Hyperfeatures (Luo et al.): https://arxiv.org/abs/2305.16843
- MAE: https://arxiv.org/abs/2111.06377
- Flow Matching: https://arxiv.org/abs/2210.02747

---

# Gen4U 深度解析：用 Diffusion 模型做 Video Understanding

## 一、核心动机与历史脉络

这篇 paper 的核心 thesis 是一个反直觉发现：**当下足够强的 video diffusion model（如 Veo3、Wan 2.2），其内部 frozen representation 已经是一个 general-purpose video encoder**，不需要任何 fine-tuning 就能在 semantic（语义）和 geometric（几何）任务上同时达到 SOTA 或接近 SOTA 水平。

这直接挑战了之前的认知。早期工作 [Vélez et al., 2025] 研究 WALT 模型时得出结论：diffusion representation 擅长 low-level geometry（深度、位姿），但 high-level semantics 很弱。Gen4U 的关键反例是：**当 generative capability scale 上去之后，semantic alignment 自然涌现**。这其实呼应了 Platonic Representation Hypothesis [Huh et al., 2024] —— 不同模态、不同目标训练的大模型最终趋向相似的 representation geometry。

参考资料：
- Platonic Representation Hypothesis: https://proceedings.mlr.press/v235/huh24a.html
- Vélez et al. diffusion representations: https://arxiv.org/abs/2507.13249
- Predictive coding theory (Rao & Ballard 1999): https://www.nature.com/articles/nn0199_79

## 二、Latent Diffusion 的数学背景（变量详解）

paper 在 3.1 节给出两类 diffusion 形式，需要清楚区分：

### 2.1 标准 Latent Diffusion (Veo3)

输入 video $\boldsymbol{x} \in \mathbb{R}^{F \times H \times W \times C}$（F=帧数，H/W/C=高/宽/通道）。Encoder $E$ 把它压缩到 latent：
$$z_0 = E(\boldsymbol{x})$$

Forward process（加噪）按 variance schedule $\bar{\alpha}_t$（累积 noise coefficient，t 越大 $\bar{\alpha}_t$ 越小）：
$$z_t = \sqrt{\bar{\alpha}_t}\, z_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$
- $z_t$：t 时刻被污染的 latent
- $\epsilon \sim \mathcal{N}(0, I)$：标准高斯噪声
- $\bar{\alpha}_t$：控制 signal/noise 比例的累积系数

训练目标（reweighted VLB）：
$$\mathcal{L}_{LDM} = \mathbb{E}_{z_0, \epsilon, t}\left[\|\epsilon - f_\theta(z_t, t, c)\|_2^2\right]$$
- $f_\theta$：DiT backbone，预测噪声 $\epsilon$
- $c$：condition（text embedding 等）
- $t$：noise step

### 2.2 Flow Matching (Wan 2.2)

连续时间 $t \in [0,1]$，linear interpolation：
$$z_t = (1-t)\,z_0 + t\,\epsilon$$

预测 velocity field：
$$f_t = \frac{dz_t}{dt} = \epsilon - z_0$$

Loss：
$$\mathcal{L}_{FM} = \mathbb{E}_{z_0, \epsilon, t}\left[\|f_\theta(z_t, t, c) - (\epsilon - z_0)\|_2^2\right]$$

**直觉**：DDPM 预测"加了什么噪声"，Flow Matching 预测"从 data 指向 noise 的方向向量"。两者最终学的都是 denoising vector field，只是参数化不同。

参考资料：
- Flow Matching original: https://arxiv.org/abs/2210.02747
- Wan technical report: https://arxiv.org/abs/2503.20314

## 三、关键中间变量：$h_t^{(l)}$

这是整篇 paper 的主角。把 backbone $f_\theta$ 看作 L 层 transformer block 序列，定义：

$$h_t^{(l)} \in \mathbb{R}^{F' \times H' \times W' \times D}$$

- 上标 $(l)$：第 l 层（深度）
- 下标 $t$：在 noise level t 处 forward
- $F', H', W'$：latent 空间的时空分辨率（被 VAE 压缩过）
- $D$：channel 维度

**关键 insight**：生成任务虽然要跑 T 步 iterative denoising，但 paper 发现**单次 forward 在特定 (l, t) 抽出的 $h_t^{(l)}$ 已经是极好的 representation**。这就是 Gen4U 名字由来 —— Generation for Understanding，复用生成路径做理解。

## 四、Latent Space 结构分析（Section 3 核心）

### 4.1 PCA 可视化（Figure 2）

把整个 dataset 在每个 (depth, noise) 点的 token 做 PCA，前 3 个主成分映成 RGB。观察：
- **高 noise level**（t 大）：feature 是 low-frequency、粗糙 shape
- **低 noise level**（t 小）：feature 是 high-frequency、细节丰富

这与 Sander Dieleman 的 "Diffusion is spectral autoregression" 假说完全吻合 —— denoising 过程本质是频率从粗到细的 autoregressive decoding。
- Dieleman blog: https://sander.ai/2024/09/02/spectral-autoregression.html

### 4.2 Mutual k-NN Alignment（Appendix A 公式详解）

这是 [Huh et al., 2024] 提出的 zero-shot metric，核心思想：两个 encoder 的 representation manifold 是否把同一个 video 放到相似的邻居结构中。

**Step 1：从 video 抽 embedding**
$$\boldsymbol{x}_i = \text{AvgPool}\left(h_t^{(l)}(v_i)\right) \in \mathbb{R}^D$$
对 video $v_i$，取 layer $l$、noise $t$ 的 activation，做 spatiotemporal average pooling 得到单向量。堆叠成 $X \in \mathbb{R}^{N \times D}$。

**Step 2：从 reference encoder 抽 embedding**
$$\boldsymbol{y}_i = E_{\text{ref}}(c_i) \in \mathbb{R}^{D'}$$
reference 可以是 text encoder（如 Gemma-2-9b-it）或 vision encoder（DINOv2、VideoMAEv2）。堆叠成 $Y \in \mathbb{R}^{N \times D'}$。

**Step 3：特征预处理**
- 按 quantile $q=0.95$ 做 clip 抑制 outlier：$\tau = \text{Quantile}_{0.95}(|X|)$，clip 到 $[-\tau, \tau]$
- $\ell_2$-normalize 每行，这样 dot product = cosine similarity

**Step 4：构造 k-NN graph**
$$M_{ij}^X = \begin{cases} 1 & \text{if } j \in \mathcal{N}_k^X(i) \\ 0 & \text{otherwise} \end{cases}$$
$\mathcal{N}_k^X(i)$ 是 sample $i$ 在 $X$ 空间中的 k 个最近邻（k=10）。

**Step 5：Alignment score**
$$\mathcal{A}_{\text{MkNN}}(X, Y) = \frac{1}{kN} \sum_{i=1}^N \sum_{j=1}^N (M^X \odot M^Y)_{ij}$$

直觉：对每个 sample $i$，看它在 $X$ 空间的 k 个邻居和 $Y$ 空间的 k 个邻居有多少重叠，所有 sample 平均。完全重叠=1，完全无关=0。**这个 metric 不需要训练 mapping，对维度和 scale 不变，只依赖 rank-order**。

**Step 6：Layer 优化**
$$(l^\star, l'^\star) = \arg\max_{l, l'} \mathcal{A}_{\text{MkNN}}(X^{(l)}, Y^{(l')})$$
扫所有 layer pair 取最优。

### 4.3 关键发现一：Veo3 的 Bimodal Pattern

Figure 3 左是 Veo3 vs Gemma-2-9b-it 的 alignment 热图。**出现两个峰值**：
- 第一个峰在浅层 (~25% depth)
- 中间出现 dip（representation 收缩、噪声增加）
- 第二个峰在深层 (~70-80% depth)

这与 Wan 2.2 不同 —— Wan 2.2 是经典 unimodal（峰在 25%）。paper 把这种 bimodal 现象与 [Valeriani et al., 2023] 研究的大 transformer geometry 联系：浅层 manifold 先膨胀（提取信息），中层收缩路由信息，深层再膨胀做最终抽象。

参考资料：
- Valeriani et al. geometry of transformer hidden reps: https://arxiv.org/abs/2302.02225

### 4.4 关键发现二：60% Noise 是 Semantic Bottleneck

无论 Veo3 还是 Wan 2.2，无论对 text 还是 visual encoder，alignment 峰值都落在 **t ≈ 60%**。这是个很强的 empirical 一致性，暗示 diffusion 模型在 60% noise 处存在一个 semantic bottleneck —— 此处 representation 既保留了全局语义可分性，又没有沉到 pixel-level 细节。

**直觉解释**：t 太高（90%），信号被噪声淹没，只剩下 prior 结构；t 太低（10%），representation 已经偏向 pixel-level reconstruction，语义被 scattered 到局部 patch；中间 60% 是 sweet spot。

### 4.5 关键发现三：Linear Probe vs Attention Probe 的 Shift

Figure 5 在 SSv2 上做了两类 probe：
- **Linear probe**：global pooling + linear projection（参数 ~1M）
- **Attention probe**：cross-attention with learnable queries（参数 ~6M）

最优位置出现 shift：
- Linear probe 最优：t ≈ 60%, depth ≈ 70%
- Attention probe 最优：t ≈ 30%, depth ≈ 80%

**核心 insight**：低 noise level 的语义信息并未丢失，而是**空间上 scattered 到局部 patch**，需要 attention 这种 dynamic spatiotemporal pooling 来 aggregate。linear pooling 太简单，无法从分散的局部细节里 reconstruct 出全局语义。

paper 还做了 control 实验：如果只是参数量驱动，attention probe 应该在所有 noise level 都 dominate。但实际在 t ≈ 90% 时 attention probe 也只有 15-30%，与 linear 相当。这证明 attention 的优势源于**机制而非容量**。

## 五、Gen4U 框架的工程实现

### 5.1 单次 Forward Pass 的效率

经典 diffusion 推理要 T 步（通常 30-50 步）iterative denoising，极其昂贵。Gen4U 关键工程贡献：**只在固定 (l, t) 抽一次 activation**，成本与普通 ViT encoder 相当。

具体细节：
- 固定 random seed 保证 reproducibility
- text condition 用 generic embedding："A video of a scene"（避免泄漏 ground truth label）
- 也试过 empty string conditioning，差异不大

### 5.2 Decoder 设计

不同任务用不同轻量 decoder（backbone 始终 frozen）：
- **Video classification**：1-block attention decoder（仿 4DS [Carreira et al., 2025]）
- **Depth/Pose estimation**：DPT head [Ranftl et al., 2021]（~23M 参数）
- **Captioning**：cross-attention adapter + Gemma-2-2B LLM（仿 [Sajjadi et al., 2022]）

参考资料：
- 4DS scaling: https://arxiv.org/abs/2412.15212
- DPT: https://arxiv.org/abs/2103.13413

## 六、实验结果深度解读

### 6.1 SSv2 Video Classification（Table 1）

数据集：220,847 个短视频（2-6秒），174 类，12fps。任务难在 finer motion 区分（如 "pouring" vs "pretending to pour"）。

关键对比：
| Model | Pre-training | Size (M) | Top-1 (%) |
|-------|-------------|----------|-----------|
| VideoMAEv2-g | MAE | 1013 | 65.6 |
| VideoPrism-g | MAE+Contrastive | 1113 | 65.4 |
| 4DS-j | MAE | 21495 | 68.2 |
| InternVideo2 | MAE+Contrastive+Caption | 6000 | 67.7 |
| V-JEPA-H | Masked feature pred | 635 | 72.2 |
| V-WALT | Diffusion (旧) | 1900 | 59.7 |
| **Gen4U (Veo3)** | Diffusion (新) | - | **71.3** |
| **Gen4U + aug** | Diffusion (新) | - | **72.6** |

**核心叙事**：
- V-WALT（早期 diffusion）只 59.7%，证实 Vélez et al. 的负面结论
- Gen4U（Veo3）直接 71.3%，超越所有 MAE 和 contrastive 方法，仅次于 V-JEPA
- 加 activation-level augmentation（temporal masking 16.7%、attention dropout 40%、label smoothing 0.2）后 72.6%

**重要工程 trick**：不在 raw video 上做 augmentation（计算昂贵），而在 Veo3 抽出的 intermediate activation 上做。这是 frozen-encoder 范式的独特优势 —— feature 已经 cached，augmentation 在 latent 空间几乎免费。

### 6.2 Captioning（Table 2）

数据集：COCO（image）、SSv2、VATEX。Decoder 是 cross-attention adapter + Gemma-2-2B。

关键观察：
- **SSv2 上 Gen4U 大幅领先**：CIDEr 289.5 vs SigLIP-so400m 204.5（+85 分）
- **COCO/VATEX 上 Gen4U 较弱**：COCO CIDEr 54.9 vs SigLIP 118.5
- 加 noise augmentation（多 noise level 10%/30%/60% 联合训练，只在 30% 评估）后 COCO 升到 69.3
- 进一步 high-resolution 输入后 COCO 升到 102.0，仍差 SigLIP 一些

**为什么 SSv2 强 COCO 弱？**
- SSv2 是 action 描述，依赖时序运动理解 —— 这是 video diffusion 强项
- COCO 是静态图像多小物体描述 —— Veo3 在低分辨率下丢失小物体细节
- SigLIP 是 explicitly 训练做 vision-language alignment，本来就在 captioning 上有 inductive bias

**Frozen LLM 实验**（Vatex [Frozen LLM] 列）：冻结 LLM 只训 adapter，性能下降很小（CIDEr 44.8→40.2），且远超 VideoPrism（31.7）。这证明 Veo3 representation 本身就蕴含足够语义，不需要 LLM 协同微调。

### 6.3 Depth Estimation（Figure 6 左）

数据集：ScanNet。Metric：AbsRel = $|d^* - d| / (d + \epsilon)$（越小越好），$\delta_1$ threshold accuracy。

DPT head（23M 参数）+ frozen Veo3。
- 最优 (depth 80%, noise 30%) 处 AbsRel = 0.075，$\delta_1$ = 0.952
- 比 4DS frozen-feature baseline 0.084 提升 10.7%

这是 ScanNet depth 上 frozen video model feature 的最佳结果。

### 6.4 Camera Pose Estimation（Figure 6 右）

任务：从 F 帧 clip 预测第一帧和最后一帧之间的 6DoF 相对位姿（SE(3) 变换）。
- 表示成 12D 向量：3×3 rotation matrix + 3×1 translation
- Metric：EPE（rotation 和 translation 联合 end-point-error）
- 1-block attention decoder

结果：Gen4U 1.10 EPE，与 DINOv2 baseline 1.08 EPE 相当。sweet spot 同样在 ~75% depth、60% noise。

## 七、Appendix B：跨 Block/Noise 的 Feature 组合

paper 还做了一个有意思的探索：从 12 个等距 block × 4 个 noise level（10%/30%/60%/90%）抽出 48 个 feature vector，用不同 adapter 融合。

四种 adapter：
1. **Linear adapter**：$F_{combined} = \sum_{i=1}^{48} w_i \cdot F_i$，仅 48 个标量参数
2. **Shared MLP**：48→M→H→1 的 MLP 共享跨 channel
3. **Self-attention**：CLS token + 48 feature 自注意力
4. **Cross-attention**：Perceiver-style 单 query cross-attend 到 48 feature

两种训练目标：
- Cross-entropy（分类）
- Multi-positive InfoNCE：
$$\mathcal{L}_{InfoNCE} = -\frac{1}{B}\sum_{i=1}^B \log \frac{\sum_{j \in \mathcal{P}(i)} \exp(\sin(\boldsymbol{v}_i, \boldsymbol{v}_j)/\tau)}{\sum_{k=1}^B \exp(\sin(\boldsymbol{v}_i, \boldsymbol{v}_k)/\tau)}$$
- $\mathcal{P}(i)$：text feature 空间的 k 近邻作为多正样本

**反直觉发现**：
- Linear adapter 最 robust，跨 dataset 泛化最好
- Cross-attention 训练难，需要特殊 batching（16 anchor × 10 positive × 22 negative = 32 sample/anchor）
- **训练分类任务反而比训练 text alignment 更能提升 text alignment metric**

最后一个观察很深刻：discrete cross-entropy 提供更干净的 gradient，比 noisy 的 contrastive alignment 更稳定地学到 semantic 结构。这呼应了 Karpathy 你自己常说的一句话 —— " supervised signals are more sample-efficient than self-supervised"。

## 八、Appendix C：Preview Decoding 实验

paper 还做了 RGB reconstruction probe：把 $h_t^{(l)}$ 通过 linear 或 attention head decode 回 RGB video，看哪个 (l, t) 点 reconstruction MSE 最低。

结果意外：**这个 metric 没有明显的 sweet spot**！与 alignment 和 classification probe 的清晰双峰模式不同，reconstruction MSE 随 depth 单调下降（深层离 RGB 输出更近），随 noise 变化不显著。

**重要启示**：representation quality 的衡量高度依赖 probe 类型。pixel-level reconstruction 不能反映 semantic quality，这与 MAE 文献观察一致 —— pixel reconstruction loss 容易被 low-frequency 主导，丢失语义。

## 九、对社区的 Implications

### 9.1 Unification 范式

这篇工作第一次实际演示了一个 frozen video model 同时支持：
- 高质量生成（保留 Veo3 全部能力）
- SOTA-level video classification
- 强 depth/pose estimation
- 满意的 captioning

实际意义：未来可能只需要训练一个巨大的 video diffusion model，下游 understanding 任务直接 hook 一个轻量 decoder，省掉维护多个 foundation model 的成本。

### 9.2 与 V-JEPA 路线的对照

V-JEPA [Bardes et al., 2024, Assran et al., 2025] 通过 latent 空间预测未来做自监督，在 SSv2 上达到 72.2%。Gen4U 是 72.6%。两者都是 generative-style 训练（一个预测 latent 未来，一个预测 noise），都达到 SOTA，暗示 **prediction 是 representation learning 的本质**，验证了 Rao & Ballard 1999 的 predictive coding 假说。

参考资料：
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- V-JEPA 2.1 dense features: https://arxiv.org/abs/2603.14482

### 9.3 与 Transfusion 的互补

Transfusion [Zhou et al., 2025] 把 next-token prediction（语言理解）和 diffusion（生成）合到一个 model，证明 understanding 提升 generation。Gen4U 是反向 —— generation 模型已经 implicit 学到 understanding。两个方向合在一起指向同一个结论：**generation 与 understanding 是同一枚硬币的两面，足够大的模型会自然统一它们**。

参考资料：
- Transfusion: https://openreview.net/forum?id=SI2hI0frk6

### 9.4 Bimodal Pattern 的开放问题

Veo3 的 bimodal depth pattern 是个未解之谜。可能与训练目标、模型规模、conditioning 机制都有关。如果能搞清楚这个机制，可能能指导设计更好的 next-gen diffusion backbone。

## 十、Limitations 与 Future Work

paper 自己承认：
- 主实验基于 proprietary Veo3，reproducibility 受限
- Wan 2.2 复现了关键发现但效果较弱，说明 representation quality 强依赖 generative capability
- captioning 与专门训练的 VLM（如 PaliGemma、BLIP-2）仍有差距
- bimodal 现象机制未解明

未来方向：
- 多 block 多 noise feature 融合的最佳架构
- 把这个 framework 扩展到 audio、3D 等更多 modality
- 研究 diffusion 训练 dynamics 中 semantic feature 是何时涌现的（训练 curriculum 分析）

---

## 总体评价

这篇 paper 的核心贡献是把"diffusion 模型能否做 understanding"这个问题从"理论上可能"推进到"实际 SOTA"。它最大的价值是 empirical 的 —— 在一个足够强的模型（Veo3）上展示了 implicit semantic alignment 涌现，并用详尽的 probe（PCA、MkNN、linear/attention probe、reconstruction）系统 mapping 了 latent space 结构。

特别值得称赞的几个细节：
1. **零训练证明 alignment**：mutual k-NN 这种 zero-shot metric 设计精妙，避免训练引入的混淆变量
2. **Attention vs Linear probe 的 shift 解释**：揭示 semantic 信息在低 noise 时空间 scattered，需要复杂 pooling 解码
3. **Activation-level augmentation**：一个工程 trick 但实用意义大

如果你 Karpathy 要继续推这个方向，我会建议聚焦：训练 dynamics 中 representation geometry 的 evolution —— 什么时候 Veo3 开始显示 60% noise bottleneck？这与 generative capability（FID 下降）是否同步？这能告诉我们 representation 与 generation capability 的因果关系。

相关参考链接汇总：
- Veo3 tech report: https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf
- Wan 2.2: https://arxiv.org/abs/2503.20314
- DINOv2: https://arxiv.org/abs/2304.07193
- VideoMAEv2: https://arxiv.org/abs/2303.16755
- PaliGemma: https://arxiv.org/abs/2407.07726
- SigLIP 2: https://arxiv.org/abs/2502.14786
- SSv2 dataset: https://arxiv.org/abs/1706.04230
- ScanNet: https://arxiv.org/abs/1702.04405
- Latent Diffusion (Rombach): https://arxiv.org/abs/2112.10752
- Track4Gen (点跟踪提升生成): https://arxiv.org/abs/2501.17668
- Genie 2 world model: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
