---
source_pdf: Infrared and Visible Image Fusion with Hierarchical.pdf
paper_sha256: 93f52c00a9fb32b32648b98c0b7c24746488ecbaa590633cb37a761beef797d0
processed_at: '2026-08-05T09:51:46-07:00'
target_folder: Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 HPFusion

好嘞 Andrej，我换个调子，把它讲得像在 coffee chat 一样，但该有的技术细节还是不省。

---

## 一、这 paper 一句话讲什么

Infrared 和 visible 两张图 fuse 成一张，老办法都在管 "pixel 对不对、edge 锐不锐、detection 准不准"，但这篇说：**还得管 "人看着觉得对不对"**。怎么管？拿 LLaVA 当一个 "人类代理观众"，给 image pair 问 4 个由粗到细的问题，把答案塞进 fusion network 当 guidance，再让 fused image 自己被问同样问题时，答案分布跟 source image 对齐。这样 fuse 出来的图既保留 thermal 又保留 texture，同时在 semantic space 里不跑偏。

---

## 二、为什么需要 IVF（先 setup 一下场景）

想象你晚上开车，visible camera 拍出来的图黑乎乎一片，啥也看不见；infrared camera 能拍到前面有个人（热目标亮），但是脸啊、路牌啊这些 texture 全没有。你想要一张图，既把那个人高亮出来，又把路牌上的字、车的轮廓这些 detail 都留住——这就是 IVF 干的事。

典型场景：surveillance、autonomous driving、military observation、fire rescue。这任务已经火了 20 年，从 hand-crafted multi-scale transform 一直到 GAN、diffusion，方法换了一茬又一茬。

---

## 三、现有方法的痛点在哪

这块 paper 里写得比较 implicit，我帮你点破：

**Pain 1: metric 和人眼对不上。** 传统 IVF 优化的是 EN (entropy)、SD (standard deviation)、SF (spatial frequency)、VIF 这些 statistical metric。这些数字高，不代表你看着舒服。比如你把整张图都 saturate 到 max intensity，SD 上去了，但人看着像过曝的废片。

**Pain 2: task metric 和 perception 对不上。** 最近流行 fuse + detection 级联（TarDAL、SeAFusion），优化 detection mAP。但 detection 关心的是 bounding box 坐标，不关心 texture 是不是糊了。你完全可能 fuse 出一张 detection mAP 很高、但人看了觉得 "这图怎么这么脏" 的结果。

**Pain 3: 没人 explicit 管 "human semantic focus"。** 人看图是有 hierarchy 的：先扫一眼全局（这是个街景），再锁定 salient target（那边有个人），再看 detail（人手里拿着啥）。现有 IVF 方法没显式建模这个 hierarchy，靠 network 自己学，学成啥样靠运气。

这篇 paper 的切入就是 Pain 3。

---

## 四、四个问题是怎么设计的

作者设计 4 个问题，从 global 到 local 模拟人眼 attention。Paper 里 Fig.1 给了 example，大概长这样：

- **Q4（最 global）**: "What is the content of the image?" — 问整体场景是啥
- **Q3**: "What targets are significant in this image?" — 问有哪些显著目标
- **Q2**: 关注 high-contrast region 的 detail
- **Q1**: 关注另一类 rich-information region 的 detail

Q4 → Q3 → Q1/Q2 这个顺序对应认知科学里的 **coarse-to-fine attention**：先 gist，再 preattentive pop-out，最后 focal attention 到 detail。

这个 hierarchy 背后的理论是 **Treisman Feature Integration Theory** (1980) 和 **Itti-Koch saliency map** (1998)。paper 没 cite，但 design 明显踩在这个脉络上：
- https://en.wikipedia.org/wiki/Feature_integration_theory
- https://www.scholarpedia.org/article/Saliency_map

为什么这样设计能 work？因为 LLaVA 看到 IR image 和 visible image 给的答案会不一样：
- IR image 上问 "significant targets"，LLaVA 会说 "a person standing"，因为热目标最显眼
- visible image 上问同样问题，LLaVA 可能说 "cars and buildings"，因为 visible 里 texture 丰富的东西更吸引它

这种 answer 的 modality-specific bias，本身就是 complementary prior。把这两套 answer 都塞进 network，等于告诉 network "这俩 modality 各自擅长啥"。

---

## 五、架构怎么把 text 塞进 fusion network

整体 flow 大概是：

```
[IR image]   ─┬─► LLaVA ──► 4 answers ──► CLIP text encoder ──► Φ_ir^T  ─┐
[vis image]   ┘                                                              │
                                                                             ├─► 1×1 conv + concat ──► Φ_{i-s}^T
[IR image]   ─► CLIP img encoder ──► Φ_ir^V  (for loss only)                 │
[vis image]  ─► CLIP img encoder ──► Φ_vis^V (for loss only)                │
                                                                             ▼
[IR image]   ─► MDA conv block ──► M_ir   ──┐                              ┌── Cross-Attention ×2
[vis image]  ─► MDA conv block ──► M_vis  ──┘   Q = Φ_{i-s}^T              │   (text as Query)
                                                K, V = M_ir, M_vis         │   (visual as Key/Value)
                                                                             ▼
                                                                       F_ir, F_vis
                                                                             ▼
                                                                MDA multi-scale encoder + decoder
                                                                             ▼
                                                                        I_f (fused image)
                                                                             ▼
                                            LLaVA ──► 4 answers ──► CLIP text encoder ──► Φ_f^T
                                                                              │
                                                                              ▼
                                                                       hierarchical loss
```

几个关键点掰开说：

### 5.1 Human Perception Module (HPM)

HPM 由 LLaVA + CLIP 组成，**两个都 frozen**。为什么不 fine-tune？因为它们是在 web-scale 数据上 pretrain 出 align 过的 vision-text space，你 fine-tune 反而会破坏 alignment。这是 CoOp、CLIP-adapter 这些 prompt tuning 工作的共识：https://arxiv.org/abs/2108.02602

LLaVA 内部结构：CLIP ViT visual encoder + Vicuna LLM，input image + text question，output text answer。Frozen 之后 forward pass 就够了。

文本处理：4 个 answer 每个经过 CLIP text encoder 得到 `R^{77×D}` 的 embedding（77 是 CLIP 的 context length，D=512 是 embedding dim）。论文写 $T_{ir} \in \mathbb{R}^{77 \times 4}$，这里 77 是 token length，4 是 4 个 answers。

然后用 1×1 conv 把每个 answer 的 representation 压成 1 维，concat 起来形成 $\Phi_{i-s}^T$。这个 1×1 conv 是 learnable 的，是整个 pipeline 里少数 trainable 的部分。

### 5.2 Cross-Attention Block

公式 (1)：
$$F_{ir}, F_{vis} = CA(M_{ir}, M_{vis}, \Phi_{i-s}^T)$$

- $M_{ir}, M_{vis}$: MDA backbone 提取的 visual feature
- $\Phi_{i-s}^T$: 融合后的 text feature，作为 **Query (Q)**
- $M_{ir}, M_{vis}$ 同时作为 **Key (K)** 和 **Value (V)**

Cross-attention 内部计算：
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Q 来自 text，K/V 来自 visual，相当于 text 的 semantic 引导 visual feature 在哪里 attend。比如 text 说 "a person"，attention map 就会在 visual feature 上 person 位置 weight 高，相当于 "请重点关注人这个区域"。

Cascade 2 个 block 是为了 text 和 visual 充分 integrate，类似 Flamingo 的 gated cross-attention：https://arxiv.org/abs/2204.14198

### 5.3 Fusion Backbone: MDA

Backbone 来自作者前作 MDA (Neurocomputing 2024)，是 multi-scale encoder-decoder，类似 U-Net 但加了 cross-modality attention。Paper 里没细讲 MDA，因为是 baseline。

类似工作可以参考 SeAFusion (2022)：https://arxiv.org/abs/2207.02507

---

## 六、三个 loss 各自在干嘛

### 6.1 Intensity Loss（公式 2）

$$L_{int} = \| I_f - \max(I_{ir}, I_{vis}) \|_1$$

- $I_f$: fused image
- $\max(I_{ir}, I_{vis})$: pixel-wise 取最大值
- $\|\cdot\|_1$: L1 距离

直觉：每个 pixel 上，IR 和 visible 哪个 intensity 大，就保留哪个。热目标在 IR 里亮，texture 在 visible 里亮，取 max 就能两者都留住。这 trick 从 DenseFuse (2018) 就开始用：https://ieeexplore.ieee.org/document/8580578

缺点：pixel-wise max 会有 gradient artifact，所以需要下面的 detail loss 补。

### 6.2 Detail Loss（公式 3）

$$L_{detail} = \underbrace{(1 - \text{SSIM}(I_f, I_{ir})) + (1 - \text{SSIM}(I_f, I_{vis}))}_{\text{structure 保留}} + \underbrace{\|\nabla I_f - \max(\nabla I_{ir}, \nabla I_{vis})\|_1}_{\text{edge 保留}}$$

- $\text{SSIM}(\cdot, \cdot)$: structural similarity，1 表示完全相同
- $\nabla I$: image gradient (通常用 Sobel 算子)
- $\max(\nabla I_{ir}, \nabla I_{vis})$: pixel-wise 取 edge 最大

前半部分：让 fused image 在结构上像 IR 又像 visible，两边都别偏离太多。后半部分：edge 也要取 max，保留最 sharp 的边界。

### 6.3 Hierarchical Semantic Loss（公式 4 + 5，核心创新）

公式 (4) 算 image-text similarity score：
$$S(I_m) = \frac{e^{\cos(\mathcal{E}_{img}(I_m), \mathcal{E}_{text}(T_m^i))}}{\sum_{i \in \{batch\}} e^{\cos(\mathcal{E}_{img}(I_m), \mathcal{E}_{text}(T_m^i))}}$$

变量解释：
- $m \in \{ir, vis, fusion\}$: image 类型
- $i$: batch 内的 sample index
- $\mathcal{E}_{img}, \mathcal{E}_{text}$: frozen CLIP image / text encoder
- $T_m^i$: 第 $i$ 个 sample 的某一条 answer text
- 分子: 当前 image 与它自己 answer 的 cosine similarity 的 exp
- 分母: 当前 image 与 batch 内所有 sample 的同位置 answer 的 cosine similarity 的 exp 之和

**这就是 InfoNCE / CLIP contrastive loss 的 softmax 形式**。Ref: https://arxiv.org/abs/2103.00020

公式 (5) 把 fused 和 source 的 similarity distribution 对齐：
$$L_{hier} = \sum_{j \in 4} \left( \| S(I_f^j) - S(I_{ir}^j) \|_1 + \| S(I_f^j) - S(I_{vis}^j) \|_1 \right)$$

- $j$: 第 $j$ 个 question-answer set (1 到 4)
- $S(I_m^j)$: 第 $j$ 个问题下 image $I_m$ 的 similarity distribution

直觉：希望 fused image 在 CLIP space 里与 text answers 的 similarity 分布，跟 IR / visible 各自与对应 answers 的 distribution 接近。这就避免了 fused image 在 semantic space 漂移到 modality gap 中间，变成"语义四不像"。

这设计让我想到 **DreamSim** (NeurIPS 2023) 用 LVLM ensemble 做 perceptual metric：https://arxiv.org/abs/2306.09392  
还有 **DPO** 的 spirit——不显式定义 reward，直接 match reference distribution。

### 6.4 总 loss

$$L_{total} = L_{int} + \alpha \cdot L_{detail} + \beta \cdot L_{hier}, \quad \alpha=4, \beta=1$$

三个 level：
- Pixel level: $L_{int}$
- Structure level: $L_{detail}$
- Semantic level: $L_{hier}$

---

## 七、实验结果讲讲

### 7.1 数据集和训练

用 **M3FD** dataset (CVPR 2022, TarDAL 同作者发布): https://github.com/JinyuanLiu-CV/M3FD
- 4200 train pairs + 300 test pairs
- 6 个场景: daylight challenge, night challenge, smog, low illumination, low illumination challenge, no challenge
- image 都已经 spatial-aligned，还有 detection annotation

训练配置：
- Adam optimizer, batch size = 8, 100 epochs
- Image resize 到 **224×224**（因为 CLIP input 是 224）
- 2× RTX 3090: 一块跑 LLaVA inference（frozen），一块训练 fusion network

⚠️ 224×224 是个 limitation。实际 IVF 应用经常需要 640×512 甚至更高 resolution，224 会丢很多 detail。LLaVA/CLIP 只看 downsampleed 的图，fusion backbone 可以跑高 resolution，但 L_hier 计算时 high-res fused image 还得 resize 回 224 喂给 CLIP，这 mismatch 可能让 semantic guidance 在 high-resolution 上失真。

### 7.2 定量结果 (Table I)

| Method | MSE↓ | SSIM↑ | PSNR↑ | CC↑ | Q^{AB/F}↑ |
|---|---|---|---|---|---|
| RFN-Nest | 0.034 | 0.397 | 63.37 | 0.572 | 0.406 |
| GANMcC | 0.037 | 0.391 | 62.96 | 0.571 | 0.268 |
| YDTR | 0.044 | 0.471 | 62.73 | 0.554 | 0.478 |
| LRRNet | 0.039 | 0.388 | 62.95 | 0.541 | 0.498 |
| EMMA | 0.057 | 0.451 | 61.69 | 0.502 | **0.592** |
| MDA (baseline) | 0.033 | 0.438 | 62.52 | 0.585 | 0.487 |
| **HPFusion** | **0.032** | **0.500** | **63.79** | **0.595** | 0.505 |

亮点：
- 相比 baseline MDA，SSIM 从 0.438 → 0.500，**提升 14%**，相当显著
- PSNR 从 62.52 → 63.79，提升 1.27 dB
- 4 个 metric 拿 best

为啥 Q^{AB/F} 输给 EMMA？Q^{AB/F} 是 edge transfer metric，EMMA (CVPR 2024) 用了 equivariant constraint 专门优化 edge preservation，等于拿手特长打: https://arxiv.org/abs/2310.16567

### 7.3 Ablation Studies 的 insight（最有意思的部分）

| HPM | L_hier | MSE↓ | SSIM↑ | PSNR↑ | CC↑ | Q^{AB/F}↑ |
|---|---|---|---|---|---|---|
| ✗ | ✓ | 0.033 | 0.489 | 63.68 | 0.574 | 0.523 |
| ✓ | ✗ | 0.058 | 0.467 | 61.51 | 0.543 | 0.502 |
| ✓ | ✓ | **0.032** | **0.500** | **63.79** | 0.580 | 0.505 |

三个 case 对比，能拎出 3 个 insight：

**Insight 1: 只有 HPM (text injection) 没有 L_hier 约束，模型会跑偏。**
- MSE 从 0.033 飙到 0.058，几乎翻倍
- PSNR 从 63.68 掉到 61.51，掉 2 dB
- 说明 inject text feature 是 "高 variance 操作"，单纯把 LLaVA 的 answer 塞进 network 反而会扰乱 pixel fidelity
- Network 容易被 text 的 semantic bias 带跑，比如 text 说 "a person"，network 就把整个 person 区域 intensity 都拉高，破坏了和 source image 的 pixel 对应

**Insight 2: 只有 L_hier 没有 HPM，照样能涨。**
- SSIM 0.489 已经比 baseline MDA (0.438) 高 12%
- 说明 L_hier 这个 self-consistent 约束本身就有用，相当于一个 regularizer，让 fused image 的 semantic 分布别跑偏
- 这跟 Knowledge Distillation 里 teacher logit 当 soft target 的精神一致

**Insight 3: 两个一起开有协同效应。**
- HPM 提供 guidance signal，L_hier 提供 constraint，两者互补
- HPM 告诉 network "往哪儿 attend"，L_hier 告诉 network "别 attend 过头，得对齐 source"
- 这跟 ControlNet 里 condition injection + reconstruction loss 双管齐下的思路一致: https://arxiv.org/abs/2302.05543

---

## 八、我读这 paper 的几个吐槽

读下来几个觉得别扭的地方：

**吐槽 1: 4 个 question 是 ad-hoc 设计。**
Paper 没 explain 为啥是 4 个不是 3 个或 8 个，也没 ablation on question design。可能是 "global + target + 2 个 region detail" 凑出来的。如果 question 本身设计得不好，LLaVA 给的 answer 就没价值，整个 guidance 就是噪声。这个 paper 没回答 "how to design question systematically"。

**吐槽 2: LLaVA 对 IR image 的理解能力存疑。**
LLaVA 训练数据几乎都是 visible photo，对 IR image 这种 "反直觉" modality 的描述可能 unreliable。比如 LLaVA 看到 IR image 里一个白亮的人形，可能描述成 "a glowing figure" 或者干脆 hallucinate 出 "a statue"。Paper 没报告 LLaVA 在 IR image 上的 answer 质量评估。这其实是个 trust issue——如果 LLaVA 答 does not make sense，inject 进去反而有害。

**吐槽 3: batch size = 8 跑 contrastive loss 太小。**
$L_{hier}$ 里 $S(I_m)$ 是 batch 内 contrastive softmax，InfoNCE 通常要 batch ≥ 256 才能学到 robust representation。batch=8 的 contrastive 噪声很大，可能 ablation 里的提升其实更多来自 anchor 效应（让 fused image 接近 source 的 similarity pattern），而不是真正的 contrastive learning。Paper 没讨论这点。

**吐槽 4: Generalization 验证不够。**
只在 M3FD 上做实验。M3FD 是单一数据集，scene 类型有限。如果换到 RoadScene、TNO、MSRS、VFIR 这些数据集上，HPFusion 还 work 吗？特别是 LLaVA 在不同 scene 上 answer 的 stability 没验证。

**吐槽 5: 224×224 input 严重限制实际应用。**
真实 IVF 系统经常要 640×512 或 1080p，224 训练的 model 直接用会模糊，需要 patch-based inference 或者 hierarchical CLIP。Paper 没提 inference resolution 怎么处理。

---

## 九、几个 broader 联想

### 9.1 LVLM-guided low-level vision 是个 trend

最近 1-2 年集中爆发：
- **SUPIR** (CVPR 2024): LLaVA caption 指导 image restoration: https://arxiv.org/abs/2305.15036
- **DiffBIR**: 多 stage restoration with LVLM: https://arxiv.org/abs/2308.15033
- **Coser** (CVPR 2024): CLIP cognitive super-resolution: https://arxiv.org/abs/2311.17030
- **FILM** (ICML 2024): ChatGPT caption 指导 IVF: https://openreview.net/forum?id=eqY64Z1rsT
- **HPFusion** (本篇): LLaVA hierarchical caption 指导 IVF

trend 很明显：从 "metrics-driven" 到 "perception-driven" 再到 "language-driven"。

### 9.2 跟 RLHF 的精神神似

HPFusion 的 $L_{hier}$ 跟 RLHF 有个 deep parallel：
- RLHF: 不显式定义 reward function，让 model output distribution match human preference distribution
- HPFusion: 不显式定义 "什么是好的 fusion"，让 fused image 的 CLIP-text similarity distribution match source image 的 distribution

都是 "distribution matching 替代 reward engineering" 的思路。Ref: DPO paper https://arxiv.org/abs/2305.18290

### 9.3 认知科学背景

4-question hierarchy 对应 **global gist → local fixation** 的认知模型：
- Treisman & Gelade 1980 Feature Integration Theory: https://doi.org/10.1016/0010-0285(80)90005-5
- Oliva & Torralba 2007 gist of scene: https://doi.org/10.1016/j.tics.2006.11.001
- Itti & Koch 2001 saliency: https://doi.org/10.1038/73028

Vision Transformer 里 [CLS] token 的 attention rollout 也显示类似的 coarse-to-fine 迁移，跟 human gaze pattern 像。

### 9.4 如果让我 follow up

几个方向：
1. **Learnable question**: 用 LLM 自动 generate question，类似 self-ask prompting，把 question 设计也变 trainable
2. **Multi-round LLaVA dialog**: 让 LLaVA 看完 fused image 给 feedback，迭代 refine，类似 DALL-E 3 用 GPT-4 captioner 迭代 caption
3. **RLHF-style training**: 把 LLaVA 当 reward model，PPO 训 fusion network，替代 contrastive loss
4. **VLM ensemble**: 用 LLaVA + Qwen-VL + InstructBLIP 多个 VLM 投票，提高 answer robustness
5. **High-res CLIP**: 用 OpenCLIP 的 conv backbone 或 SliceCLIP 处理 high-resolution: https://arxiv.org/abs/2311.05761

---

## 十、总结

这篇 paper 的核心 insight 一句话讲：**IVF 不该只管 pixel / structure / task metric，还得管 "human-perception-aligned semantic distribution"**。做法是用 hierarchical VQA 把 human attention hierarchy 显式化，再用 CLIP contrastive distribution 在 semantic manifold 上 anchor 住 fused image。

最有意思的 ablation insight：**text feature injection 是高 variance 操作，必须配 self-consistent semantic constraint 才能稳定收益**。这跟 prompt tuning 文献里 "soft prompt 容易 overfit" 的发现神似：https://arxiv.org/abs/2104.08691

整体上这是 IVF 领域把 LVLM 嵌入 fusion 的早期尝试，框架清晰，loss design 有 principled motivation，但 question 设计偏 ad-hoc，generalization 待验证，224×224 限制实际应用。如果能解决这几个问题，会是个挺 solid 的方向。

---

## 主要 Reference

1. CLIP: https://arxiv.org/abs/2103.00020
2. LLaVA: https://arxiv.org/abs/2304.08485
3. SUPIR: https://arxiv.org/abs/2305.15036
4. FILM: https://openreview.net/forum?id=eqY64Z1rsT
5. M3FD dataset: https://github.com/JinyuanLiu-CV/M3FD
6. TarDAL: https://arxiv.org/abs/2203.08704
7. DDFM: https://arxiv.org/abs/2308.13666
8. EMMA: https://arxiv.org/abs/2310.16567
9. DenseFuse: https://ieeexplore.ieee.org/document/8580578
10. DreamSim: https://arxiv.org/abs/2306.09392
11. DPO: https://arxiv.org/abs/2305.18290
12. CoOp: https://arxiv.org/abs/2108.02602
13. Flamingo: https://arxiv.org/abs/2204.14198
14. ControlNet: https://arxiv.org/abs/2302.05543
15. Saliency map review: https://www.scholarpedia.org/article/Saliency_map

要不要我针对某个点深挖？比如 L_hier 在不同 batch size 下的稳定性、或者 LLaVA 对 IR image 究竟靠谱不靠谱，这些都可以单独展开聊。

---

# HPFusion: Hierarchical Perception Fusion 深度解析

你好 Andrej！这篇 paper 来自 Xidian University 的团队，发表在 2024 年左右，做的是 Infrared-Visible Image Fusion (IVF) 这个相对小众但很务实的方向。我把它拆开讲，同时尽量把背后的 intuition、related work 的脉络、以及我个人觉得有意思的联想都串起来。

---

## 1. 任务背景：为什么需要 IVF

Infrared (IR) image 和 visible image 是两种截然不同的 modality：
- **IR image**：捕捉 thermal radiation，能在夜间、雾天、烟雾里突出热目标（人、车、动物），但 texture 几乎没有，spatial resolution 低。
- **Visible image**：optical spectrum，texture / edge / color 丰富，但在 low-light 或恶劣天气下会失效。

IVF 的目标就是把两者 pixel-level 融合成一张图，既保留 thermal saliency，又保留 texture detail，方便 human inspection 和 downstream detection/segmentation。

传统 IVF 方法可以分为几代：
- **第一代**：sparse representation、multi-scale transform (Laplacian pyramid, wavelet)，hand-crafted features。Ref: https://www.sciencedirect.com/science/article/pii/S1566253516300185
- **第二代**：deep learning，以 DenseFuse (2018) 为起点，用 autoencoder + attention/gradient 约束。Ref: https://ieeexplore.ieee.org/document/8580578
- **第三代**：GAN-based (FusionGAN)、classification-constrained (GANMcC)、diffusion-based (DDFM)、以及 task-cascade (TarDAL, SeAFusion)。
- **第四代 (本篇)**：引入 LVLM (Large Vision-Language Model) 作为 semantic prior，让 fusion 结果对齐 human perception。

---

## 2. 核心创新点：Hierarchical Human Perception Prior

作者观察到一个被忽略的问题：**现有 IVF 方法追求 statistical metrics (EN, SD, SF, VIF) 或者 high-level task metrics (mAP)，但这两类指标并不必然对应 "人看着舒服"**。

比如，SSIM 高的融合图可能 thermal target 被弱化，detection mAP 高的融合图可能 texture 被抹平。Human visual system 实际上有一套 coarse-to-fine 的 attention 机制：先 grasp global context，再扫到 salient region，最后 fixate 到 detail。作者把这个过程显式建模成 4 个问题，用 LLaVA 回答，再用 CLIP encode 进 fusion network。

这让我想起 **SUPIR** (CVPR 2024) 用 LLaVA 生成 caption 指导 image restoration 的思路：
https://arxiv.org/abs/2305.15036  
还有 **FILM** (ICML 2024) 用 ChatGPT 生成 caption 指导 IVF：
https://openreview.net/forum?id=eqY64Z1rsT

HPFusion 的区别在于 **hierarchical**——4 个问题从 global 到 local，结构化地模拟 human attention hierarchy，而 FILM 只是 single-level caption。

---

## 3. 四个问题设计

论文 Fig.1 展示了四个问题，根据 Method 部分推断大概是：
- **Q1**: 关注 high-contrast 局部 region 的 detail（visible 擅长）
- **Q2**: 关注另一类 salient region 的 detail
- **Q3**: "What targets are significant in this image?"（IR 擅长，识别 thermal targets）
- **Q4**: "What is the content of the image?"（global context）

这种设计很有 cognitive science 的味道，对应了 **Treisman 的 Feature Integration Theory** 和 **Itti & Koch 的 saliency model**——先 preattentive global gist，再 attention focus 到 salient location：
https://www.scholarpedia.org/article/Saliency_map

LLaVA 对 IR 和 visible 给出的答案会有 modality-specific bias，比如对 IR image 描述 "a person standing in the dark"，对 visible image 描述 "a street scene with signs and buildings"。这种差异本身就是 complementary prior。

---

## 4. 架构详解

整体 pipeline（Fig.2）：

```
IR image  ─┐
           ├─► LLaVA → 4 answers → CLIP text encoder → Φ_ir^T ─┐
visible ──┘                                                     │
                                                                ├─► 1x1 conv reduce → concat → Φ_{i-s}^T
IR image  ─► CLIP img encoder → Φ_ir^V  (用于 loss)              │
visible ──► CLIP img encoder → Φ_vis^V (用于 loss)               │
                                                                ▼
IR  → conv block (MDA) → M_ir  ─────────────────────────► Cross-Attention Block × 2
vis → conv block (MDA) → M_vis ─────────────────────────►   Q = Φ_{i-s}^T
                                                            K, V = M_ir, M_vis
                                                                ▼
                                                         F_ir, F_vis
                                                                ▼
                                                    MDA multi-scale encoder + decoder
                                                                ▼
                                                            I_f (fused image)
                                                                ▼
                                            LLaVA → 4 answers → CLIP text encoder → Φ_f^T
                                            (用于 hierarchical semantic loss)
```

**几个关键设计点**：

### 4.1 Human Perception Module (HPM)

HPM 包含两个 frozen 模块：
- **LLaVA**：multimodal LLM，由 CLIP ViT visual encoder + Vicuna LLM 组成。给定 image + question，输出 natural language answer。Frozen，不参与训练。Ref: https://arxiv.org/abs/2304.08485
- **CLIP** text/image encoder：frozen，把 answer text 编码成 77×512 维度的 embedding（CLIP 默认 context length 是 77 token）。

为什么 frozen？因为 LLaVA 和 CLIP 在大规模 web data 上 pretrain 得到的是 align 过的 vision-text space，fine-tune 反而会破坏这个 alignment。这也是 CLIP-adapter、CoOp 等 prompt tuning 工作的核心 insight：
https://arxiv.org/abs/2108.02602

文本处理细节：
- 4 个 answer 每个被 CLIP 编码成 `R^{77×D}`（D=512），总共 `R^{77×4}`
- 论文写 `T_ir ∈ R^{77×4}`，这里 77 是 token length，4 是 4 个 answers
- 经过 1×1 conv 把每个 answer 压缩到 1 维 representation，再 concatenate 成 `Φ_{i-s}^T`

### 4.2 Cross-Attention Block

公式 (1)：
$$
F_{ir}, F_{vis} = CA(M_{ir}, M_{vis}, \Phi_{i-s}^T)
$$

- $M_{ir}, M_{vis}$：MDA backbone 提取的 visual feature
- $\Phi_{i-s}^T$：fused text feature，作为 **Query**
- $M_{ir}, M_{vis}$：同时作为 **Key** 和 **Value**

这是经典的 multimodal cross-attention 设计，text 引导 visual feature 在哪里 attend。这让我想起 **Flamingo** 的 gated cross-attention 和 **ControlNet** 中 condition 注入的方式：
https://arxiv.org/abs/2204.14198

具体来说，cross-attention 的输出是：
$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

这里 Q 来自 text，K/V 来自 visual，相当于用 text 的语义信息 query visual feature map，让 visual feature 在 "person"、"car" 这些 semantic location 上 weight 更高。Cascade 2 个 block 是为了充分 integrate。

### 4.3 Fusion Backbone: MDA

HPFusion 的 fusion backbone 来自作者自己前作 **MDA** (Neurocomputing 2024)，multi-scale information integration framework。它本质是 encoder-decoder 结构 + multi-scale feature fusion，类似 U-Net 思路但加了 cross-modality attention。

我没有 MDA 原文，但从描述看像 SeAFusion 和 U2Fusion 的 family：
https://arxiv.org/abs/2207.02507

---

## 5. Loss Function 详解

总 loss：
$$
L_{total} = L_{int} + \alpha \cdot L_{detail} + \beta \cdot L_{hier}
$$
其中 $\alpha = 4, \beta = 1$。

### 5.1 Intensity Loss

公式 (2)：
$$
L_{int} = \| I_f - \max(I_{ir}, I_{vis}) \|_1
$$

- $I_f$：fused image
- $\max(I_{ir}, I_{vis})$：pixel-wise maximum，相当于取每个 pixel 上更 "salient" 的那一个
- $\|\cdot\|_1$：L1 norm

这是 IVF 领域的经典约束（DenseFuse 起就用），思想是 "热目标在 IR 里 intensity 高，texture 在 visible 里 intensity 高，取 max 就能保留两者最显著的信息"。但这个 max 是 pixel-wise 的，会有 gradient artifact，所以需要后续的 detail loss 平衡。

### 5.2 Detail Loss

公式 (3)：
$$
L_{detail} = (1 - \text{SSIM}(I_f, I_{ir})) + (1 - \text{SSIM}(I_f, I_{vis})) + \|\nabla I_f - \max(\nabla I_{ir}, \nabla I_{vis})\|_1
$$

- $\text{SSIM}(\cdot, \cdot)$：structural similarity，1 表示完全相同
- $\nabla I$：image gradient (Sobel 算子)
- $\max(\nabla I_{ir}, \nabla I_{vis})$：edge max，保留两者最显著的边缘

这个 loss 对应 denseFuse 和 U2Fusion 的 "structural + gradient" 双约束思路。SSIM 保证 structural fidelity，gradient max 保证 edge sharpness。

### 5.3 Hierarchical Semantic Loss（核心创新）

公式 (4) 定义 image-text similarity score：
$$
S(I_m) = \frac{e^{\cos(\mathcal{E}_{img}(I_m), \mathcal{E}_{text}(T_m^i))}}{\sum_{i \in \{batch\}} e^{\cos(\mathcal{E}_{img}(I_m), \mathcal{E}_{text}(T_m^i))}}
$$

- $m \in \{ir, vis, fusion\}$：图像类型
- $i$：batch 内的 sample index
- $\mathcal{E}_{img}, \mathcal{E}_{text}$：frozen CLIP image / text encoder
- $T_m^i$：第 $i$ 个 sample 的第 $j$ 个 answer text（$j$ 取 1~4）
- 分子：当前 image 与它对应的 text answer 的 cosine similarity
- 分母：当前 image 与 batch 内所有 sample 的同位置 answer 的 cosine similarity 之和

**这本质是 InfoNCE / CLIP-style contrastive loss 的 softmax 形式**。Ref: https://arxiv.org/abs/2103.00020

然后公式 (5)：
$$
L_{hier} = \sum_{j \in 4} \left( \| S(I_f^j) - S(I_{ir}^j) \|_1 + \| S(I_f^j) - S(I_{vis}^j) \|_1 \right)
$$

- $j$：第 $j$ 个 question-answer set（共 4 个）
- $S(I_m^j)$：第 $j$ 个 question 下 image $I_m$ 的 similarity score 分布

Intuition：希望 fused image 在 CLIP space 里与 text answers 的 similarity 分布，**与 IR 和 visible 各自与对应 answers 的 similarity 分布尽量接近**。这就避免了 fused image 在 semantic space 里偏移到 modality gap 中间，造成 "语义漂移"。

这是个挺精妙的 design。它不要求 fused image 与 IR image 在 pixel 上接近（那本来就是融合的目标），而要求在 **CLIP contrastive distribution** 上接近。这相当于在 semantic manifold 上做了 anchor。

### 5.4 直觉总结

- $L_{int}$：保留 thermal saliency（pixel-wise max）
- $L_{detail}$：保留 texture + structure
- $L_{hier}$：保留 hierarchical semantic distribution（CLIP space）

三者结合，pixel-level、structure-level、semantic-level 都被约束。这跟 Bilevel optimization 的思路有点像：低 level 用 reconstruction loss，高 level 用 semantic loss。

---

## 6. 实验分析

### 6.1 数据集

**M3FD** dataset (CVPR 2022, TarDAL 作者发布)：
- 4200 train pairs + 300 test pairs
- 6 个场景：daylight challenge、night challenge、challenge in smog、low illumination、low illumination challenge、no challenge
- 每对图像已经 spatial-aligned
- 同时带 detection annotation
- Ref: https://github.com/JinyuanLiu-CV/M3FD

### 6.2 训练细节

- Optimizer: Adam
- Batch size: 8
- Epochs: 100
- Image resize 到 224×224（因为 CLIP input 是 224）
- 2× RTX 3090：1 块跑 LLaVA inference，1 块训练 fusion network
- 注意：LLaVA inference 也要跑（因为要给 fused image 生成 answers 算 loss），但 LLaVA frozen，所以 inference 还是 forward pass

这里的 224×224 是 CLIP 的 input size 约束，对 high-resolution fusion 来说其实是 limitation——细节会丢失。可能需要 patch-based inference 或者 resize back 的 trick。

### 6.3 定量结果 (Table I)

| Method | MSE↓ | SSIM↑ | PSNR↑ | CC↑ | Q^{AB/F}↑ |
|---|---|---|---|---|---|
| RFN-Nest | 0.034 | 0.397 | 63.373 | 0.572 | 0.406 |
| GANMcC | 0.037 | 0.391 | 62.956 | 0.571 | 0.268 |
| YDTR | 0.044 | 0.471 | 62.728 | 0.554 | 0.478 |
| LRRNet | 0.039 | 0.388 | 62.952 | 0.541 | 0.498 |
| EMMA | 0.057 | 0.451 | 61.686 | 0.502 | **0.592** |
| MDA (baseline) | 0.033 | 0.438 | 62.517 | 0.585 | 0.487 |
| **HPFusion** | **0.032** | **0.500** | **63.794** | **0.595** | 0.505 |

HPFusion 在 4 个 metric 上拿到 best，Q^{AB/F} 输给 EMMA。Q^{AB/F} 是 edge transfer metric，EMMA 用了 equivariant constraint 专门优化 edge，所以这里能理解。

值得注意：
- 相比 baseline MDA，SSIM 从 0.438 → 0.500，提升 **14%**，这是相当显著的
- PSNR 从 62.517 → 63.794，提升 1.3 dB
- L_hier 的贡献主要在 SSIM 和 PSNR 上

### 6.4 Ablation Studies

| HPM | L_hier | MSE↓ | SSIM↑ | PSNR↑ | CC↑ | Q^{AB/F}↑ |
|---|---|---|---|---|---|---|
| ✗ | ✓ | 0.033 | 0.489 | 63.677 | 0.574 | 0.523 |
| ✓ | ✗ | 0.058 | 0.467 | 61.507 | 0.543 | 0.502 |
| ✓ | ✓ | **0.032** | **0.500** | **63.794** | 0.580 | 0.505 |

观察：
1. **只有 HPM 没有 L_hier**：MSE 飙到 0.058，PSNR 掉到 61.5，说明 HPM 注入 text feature 反而会扰乱 pixel fidelity，需要 L_hier 配合约束。
2. **只有 L_hier 没有 HPM**：相当于 baseline + semantic loss，SSIM 0.489 已经比 baseline 高，说明 semantic 约束本身有用。
3. **两个都开**：协同效应最强，MSE 和 SSIM 都最佳。

这个 ablation 揭示了一个重要的 intuition：**text feature 注入是 high-risk high-reward 的设计**——单纯 inject text feature 不 constrain 的话，模型可能被文本语义带跑，pixel fidelity 反而下降。L_hier 提供了一个 self-consistent 的 constraint，强制 fused image 自己的 LLaVA answer 与 source image 的 answer 在 CLIP space 同分布，避免 inject 带来的 drift。

---

## 7. 相关联想与 broader context

### 7.1 LVLM 在 low-level vision 的 trend

最近一年 (2024-2025) LVLM-guided low-level vision 涌现：
- **SUPIR** (CVPR 2024)：LLaVA caption → image restoration。https://arxiv.org/abs/2305.15036
- **Coser** (CVPR 2024)：CLIP cognitive super-resolution。https://arxiv.org/abs/2311.17030
- **DiffBIR**：多 stage restoration with LVLM。https://arxiv.org/abs/2308.15033
- **FILM** (ICML 2024)：ChatGPT caption 指导 IVF。https://openreview.net/forum?id=eqY64Z1rsT
- **HPFusion**：本篇，hierarchical LLaVA caption 指导 IVF

trend 是：从 "metrics-driven" 到 "perception-driven"，再到 "language-driven"。

### 7.2 Hierarchical attention 的认知科学基础

作者虽然没有显式 cite，但这个 4-question 设计背后是 **global gist → local fixation** 的认知模型。相关 reading：
- Treisman & Gelade 1980 Feature Integration Theory: https://doi.org/10.1016/0010-0285(80)90005-5
- Itti & Koch 2001 saliency: https://doi.org/10.1038/73028
- Oliva & Torralba 2007 gist: https://doi.org/10.1016/j.tics.2006.11.001

更近期的，**vision transformer 的 [CLS] token attention rollout** 也显示了这种 hierarchy：CLS attention 从 global 到 local 迁移，类似 human gaze pattern。

### 7.3 Contrastive distribution matching

$L_{hier}$ 这个设计本质是 **distribution matching in CLIP space**，让我想到几个相关工作：
- **DreamSim** (NeurIPS 2023)：用 ensemble of LVLMs 做 perceptual metric。https://arxiv.org/abs/2306.09392
- **CLIP-IQA**：用 CLIP 做 image quality assessment，把 CLIP 当 perceptual metric。https://arxiv.org/abs/2307.04505
- **DPO (Direct Preference Optimization)**：直接在 reference model 的 distribution 上做 reward matching。这里 $L_{hier}$ 思路类似——直接在 source image 的 CLIP similarity distribution 上做 anchor，避免显式定义 reward function。

### 7.4 与 Complementary learning 的关联

IR 和 visible 本身是 complementary modality，从 free-energy principle 和 predictive coding 的角度，human visual system 本质就是 multi-modality complementary inference。HPFusion 用 hierarchical question 把这个 complementary 显式化：
- IR 偏 thermal salience → Q3
- Visible 偏 texture detail → Q1, Q2
- Global context → Q4

这跟 **AttnGAN**、**ControlNet** 用不同 condition type 控制 generation 的思路是一脉相承的：https://arxiv.org/abs/1711.10485

### 7.5 潜在问题

读这篇 paper 我有几个疑问：
1. **224×224 input**：对 high-res fusion 严重受限，实际 IVF 经常需要 640×512 或更高。论文没提 inference resolution，可能 LLaVA/CLIP 只看 downsampled image，但 fusion backbone 可以跑高分辨率，不过 L_hier 计算时需要 resize。这个 mismatch 可能让 semantic guidance 在高分辨率上失真。
2. **LLaVA 对 IR image 的描述能力**：LLaVA 训练数据几乎都是 visible image，对 IR image 的理解可能 unreliable。论文没有报告 LLaVA 在 IR image 上的 answer 质量评估。这是个 trust issue。
3. **batch contrastive 的稳定性**：$S(I_m)$ 是 batch 内 contrastive，batch size = 8 太小，InfoNCE 通常需要大 batch（>=256）才能学到 robust representation。论文没提这个问题，可能 metric 提升主要来自 anchor 效应而非真正的 contrastive learning。
4. **Generalization**：只在 M3FD 上做实验。M3FD 是单一数据集，分布窄。在 RoadScene、TNO、MSRS 上的 generalization 没验证。
5. **Question 设计的 principled 性**：4 个 question 是 ad-hoc 设计还是基于某个 attention model？论文没说。我猜是基于 Itti-Koch saliency 启发的，但没有 ablation on question design。

### 7.6 如果我来 follow up

几个方向：
- **Learnable question**：用 LLM 自动 generate question，类似 self-ask prompting。把 question 也变成可学习参数。
- **Multi-round LLaVA dialog**：让 LLaVA 看 fused image 后给 feedback，迭代 refine fusion，类似 DALL-E 3 用 GPT-4 captioner 迭代 caption 的思路。
- **Reward model**：把 LLaVA 当 reward model，PPO 训 fusion network，而不是用 contrastive loss。这跟 RLHF 完全 parallel。
- **VLM ensemble**：用多个 VLM (LLaVA, Qwen-VL, InstructBLIP) 投票生成 answer，提高 robustness。
- **High-res patch-based CLIP**：用 OpenCLIP 的 conv-style architecture 或者 SliceCLIP 处理高分辨率。

---

## 8. 总结性 Intuition

这篇 paper 的核心 insight 是：**IVF 不应只优化 pixel / structure / task metric，还应优化 "human-perception-aligned semantic distribution"**。它通过 hierarchical VQA 把 human attention hierarchy 显式化，再用 CLIP contrastive distribution 在 semantic manifold 上 anchor fused image。这跟 RLHF 的精神很像——不直接定义 reward，而是让 model 的 output distribution match reference distribution。

Ablation 揭示的最有趣 insight：text feature injection 是 high-variance 操作，必须配合 self-consistent semantic constraint 才能稳定收益。这跟 prompt tuning 文献里 "soft prompt 容易 overfit" 的发现一致：https://arxiv.org/abs/2104.08691

总体上这是 IVF 领域把 LVLM 嵌入 fusion 的早期尝试，框架清晰，loss design 有 principled motivation，但 question 设计偏 ad-hoc，generalization 待验证。

---

## Reference Links

1. CLIP: https://arxiv.org/abs/2103.00020
2. LLaVA: https://arxiv.org/abs/2304.08485
3. SUPIR: https://arxiv.org/abs/2305.15036
4. FILM (ICML 2024): https://openreview.net/forum?id=eqY64Z1rsT
5. M3FD dataset: https://github.com/JinyuanLiu-CV/M3FD
6. TarDAL (CVPR 2022): https://arxiv.org/abs/2203.08704
7. DDFM (ICCV 2023): https://arxiv.org/abs/2308.13666
8. EMMA (CVPR 2024): https://arxiv.org/abs/2310.16567
9. DenseFuse: https://ieeexplore.ieee.org/document/8580578
10. FusionGAN: https://arxiv.org/abs/2012.02309
11. DreamSim: https://arxiv.org/abs/2306.09392
12. ControlNet: https://arxiv.org/abs/2302.05543
13. Flamingo: https://arxiv.org/abs/2204.14198
14. IVF Survey (Information Fusion 2019): https://www.sciencedirect.com/science/article/pii/S1566253518301033
15. Saliency map (Itti & Koch): https://www.scholarpedia.org/article/Saliency_map

如果你想深入聊某个部分（比如 L_hier 的 batch size sensitivity、或者 LLaVA 在 IR image 上的失效问题），可以告诉我，我们可以一起把它挖透。
