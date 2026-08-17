---
source_pdf: Rethinking the Simulation vs. Rendering Dichotomy.pdf
paper_sha256: 0112c2e65ca3ef9e71bc6390c120efdd025586c9f13465ac1d3f42c123e95b25
processed_at: '2026-08-11T23:39:09-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话版本

现在 AI 圈有个流行信念："做空间推理不需要生成具体图像，靠抽象的 schematic representation 就够了，aphantasia 患者就是活证据"。这篇 paper 说：**这个证据被误读了，真相是 simulation 和 rendering 共享同一套 fine-grained perceptual representation，AI 想走捷径没戏——no free lunch**。

---

## 这事儿为啥值得 care

你自己跑过 nanoGPT，也教过 Eureka Labs 的学生，应该深有体会：现在的 multimodal LLM 在 spatial reasoning 上弱得离谱。给它看两张图问"哪张是从另一个视角拍的同一个场景"，它基本靠猜；让它做 mental rotation，accuracy 跟 random 差不多（[Gao et al. 2024](https://arxiv.org/abs/2410.00324)、[Cai et al. 2025](https://arxiv.org/abs/2508.13142)）。

但诡异的是，这些 model 在 general perception、object detection、high-level reasoning 上又很强。这个 dissociation 让人困惑：**到底是 representation 不够，还是 reasoning mechanism 不对**？

cognitive science 圈最近给了一个看似优雅的解释，来自 Balaban & Ullman (2025, *TICS*, [link](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(25)00048-X))：

> 人类大脑里 mental imagery 其实是两套独立系统——**simulation**（physics-based，算空间关系、物理演化）和 **rendering**（graphics-based，生成你"看见"的画面）。前者由 dorsal stream（顶叶"zombie stream"）负责，后者由 ventral stream（颞叶，识别物体）负责。aphantasia 患者只是 rendering 坏了，simulation 完好，所以 mental rotation 这种任务照做不误。

这解释听起来太顺了，AI 圈立刻照搬：**那我们做 spatial world model 也不用管 perceptual rendering，只要学个 abstract spatial representation 就够了**。

这篇 paper 说：且慢，这个 dichotomy 是错的。

---

## 主流解释为什么是错的

### 问题 1：aphantasia 这事儿没那么简单

Balaban & Ullman 的核心证据是：aphantasia 患者能做 mental rotation，所以 spatial simulation 不需要 visual imagery。

但仔细看 aphantasia 患者是怎么做任务的——他们报告说自己"imagine grasping the shape and rotating it"。等等，这明明是**招换了 embodied sensory modality**，哪来的"modality-neutral abstract representation"？Phillips (2025, *Noûs*) 想把这说成"既不是看也不是摸，是中性的"，但这话本身就矛盾——你描述它的方式已经偷偷把 sensorimotor 招换进来了。

更糟的是，Scholz et al. (2025, *Current Biology*) 用 fMRI 发现 aphantasia 患者早期 visual cortex 根本没有 shared representation，那"unconscious imagery"这个 fallback 解释也站不住。

作者的判断很干脆：**别纠结 aphantasia 用的是不是"imagery"了，承认他们用 spatial representation 完成任务，只是没有 conscious visual experience，就行了**。任务表现 ≠ phenomenal experience，这两个东西不能画等号。

### 问题 2：dorsal = zombie stream 这个老黄历早过时了

Goodale & Milner 1992 (*TINS*, [link](https://doi.org/10.1016/0166-2236(92)90388-K)) 提出 dorsal stream 是"zombie stream"，只管 action 不管 consciousness。这个 view 在 90 年代很流行，Balaban & Ullman 直接拿来用。

但近 10 年的证据狠狠打脸：

- **Bellet et al. 2022** (*Neuroscience of Consciousness*, [link](https://academic.oup.com/nc/article/doi/10.1093/nc/niac005/6585853))：在 PPC（传统"dorsal zombie"区域）记录 rapidly presented stimuli，没有 behavioral report，依然能 above-chance 解码 object identity。这说明 PPC 主动编码 perceptual content，根本不是单纯的 visuomotor transformer。

- **Lau & Passingham 2006** (*PNAS*) 和 **Anzulewicz et al. 2019**：DLPFC（dorsolateral prefrontal cortex）的活动与 visual awareness 直接相关。lesion 到 prefrontal / parietal 会破坏 visual content 在 awareness 里的整合（Szczepanski & Knight 2014, *Neuron*）。

- **Panagiotaropoulos 2024** (*Neuron*, [link](https://www.cell.com/neuron/fulltext/S0896-6273(24)00139-X))：prefrontal cortex 是 integrative hub，不是 isolated control system。

更致命的是 **Kutsche et al. 2025** (medRxiv, aphantasia lesion study) 的结果：12 个 lesion-induced aphantasia 患者，**全部损伤都连到 left fusiform gyrus**（ventral stream 的 visual imagery 区域），**没有一个的 prefrontal cortex 受损**。

这彻底改变了 aphantasia 的解释：

| 旧解释 | 新解释 |
|---|---|
| simulation/rendering 是两个独立系统 | 共享 higher-order representation |
| aphantasia = simulation 系统完好，rendering 系统坏了 | higher-order representation 完好，下游 decoding 坏了 |
| dorsal = zombie，ventral = conscious | fronto-parietal 网络共同支撑两者 |

---

## 真正发生了什么：HOT 框架

paper 用 **Higher-Order Theories of consciousness** 来统一解释。核心 idea 来自 Hakwan Lau 的 PRM (Perceptual Reality Monitoring, [Lau 2019](https://psyarxiv.com/8sg9n/)) 和 Fleming 的 HOSS (Higher-Order State Space, [Fleming 2020](https://doi.org/10.1093/nc/niz020))。

### PRM 用 GAN 来比喻意识

把意识想象成一个 GAN：

```
First-order sensory cortices (V1, V4, ...)  ←→  Fronto-parietal higher-order net
        ↑                                              ↓
   生成 perceptual content                        Discriminator D_PRM
   (像 Generator)                                 判断 "这是真的还是想象的？"
        ↑                                              ↓
        └──────────────  Pointer  ←─────────────────────┘
                        (location index)
                              ↓
                       Feedback decoding
                              ↓
                       Conscious experience
```

形式化一点：

$$
D_{\text{PRM}}(\mathbf{h}) = \sigma\Big( \mathbf{w}^\top \cdot \phi(\mathbf{h}) + b \Big)
$$

- $\mathbf{h}$：first-order perceptual representation（来自 sensory cortex）
- $\phi(\cdot)$：higher-order meta-representation function（fronto-parietal 实现）
- $D_{\text{PRM}} \in (0,1)$：输出这个 state 是"真实外界"还是"想象/噪声"的概率
- $\sigma$：sigmoid

如果 $D_{\text{PRM}} > \text{threshold}$，就 emit 一个 **pointer** $p$，携带 location index，反馈给 first-order networks 做 decoding，于是你"看见"了：

$$
\mathbf{x}_{\text{conscious}} = \text{Decode}\Big(p, \mathcal{M}_{\text{first-order}}\Big)
$$

**Aphantasia 在这个框架下是**：discriminator 工作正常（prefrontal 完好），但 downstream decoding 坏了（fusiform 受损）——所以 higher-order representation 能用于 spatial reasoning，但永远进不了 conscious visual experience。完美解释 Kutsche et al. 的 lesion pattern。

### 关键推论

simulation 和 rendering **共享同一套 higher-order fine-grained perceptual representation**，区别只在：
- simulation：higher-order representation 直接用于决策，不进入 awareness
- rendering：higher-order representation 经过 discriminator gating + pointer feedback，被 decode 进 conscious experience

这就像你训练了一个 vision encoder，latent 既喂给 downstream task head（spatial reasoning），也喂给 pixel decoder（rendering）——**encoder 是同一个**，只是 head 不同。

---

## 这对 AI 意味着什么

### 直接的 implication：no free lunch

如果人类 spatial reasoning 依赖 fine-grained perceptual encoding（因为 higher-order indices 本身就是 fine-grained 的，包含 perceived distances, object relations 这些 spatial property），那 AI 想 emulate 这种能力，**就不能用 coarse、perceptually abstract 的 approximation 走捷径**。

这点和 Fleming & Michel (2025, *BBS*) 的进化论观点吻合：**意识可能就是为了 stabilize 内部 simulation 而进化出来的 gating 机制**——让 organism 知道什么时候该 commit 到 world model 并行动。意识本身可能是 spatial reasoning 的 representational substrate 的副产品。

### AI 圈的三条路径对比

paper 给了三条路径，每条都点评了：

#### Path 1: 纯 language model 做 spatial reasoning — 失败

GPT-4V / Gemini / LLaVA 这些 MLLM 虽然有 visual prior，但在 mental rotation、perspective-taking、mechanical reasoning 上持续失败。Huh et al. 2024 ([Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)) 说 modality 之间会向 shared abstraction 收敛——但在 spatial domain 这种收敛不会发生，**因为缺一个能 encode perceptually rich representation 的 vehicle**。

#### Path 2: 纯 explicit physics engine — 不够

MuJoCo ([Todorov 2012](https://doi.org/10.1109/IROS.2012.6386109))、Isaac Gym ([Makoviychuk 2021](https://arxiv.org/abs/2108.10470))、Genesis ([Zhou 2024](https://github.com/Genesis-Embodied-AI/Genesis)) 这些 GPU physics engine 算得精确，但 sim-to-real 转移脆弱。policy 在 open-ended visual 复杂场景下 brittle，因为缺 counterfactual model 和 concept grounding。

你可以把这条路径类比成"只有 simulation 没有 perception"——physics engine 给你 state $\mathbf{s}_t$ 和 transition $\mathbf{s}_{t+1} = f(\mathbf{s}_t, \mathbf{a}_t)$，但 $\mathbf{s}$ 是手写的低维 state（关节角、位置），不是从 pixels 学出来的 fine-grained perceptual representation。

#### Path 3: Visual pretraining for manipulation — 这才是对的方向

最近 embodied AI 圈出现了一个非常成功的范式：**用 vision foundation model 的 latent 当 implicit world model**。代表工作：

**VIP (Value-Implicit Pre-training, [Ma et al. 2022](https://arxiv.org/abs/2210.00030))**：

用 contrastive value-implicit objective 学 visual embedding。给定一段 video，让 "goal frame" 和 "achieved frame" 在 embedding space 接近，与 negative 样本远离：

$$
\mathcal{L}_{\text{VIP}} = -\mathbb{E}\left[\log \frac{\exp(\text{sim}(\mathbf{z}_{\text{goal}}, \mathbf{z}_{\text{achieved}})/\tau)}{\sum_{k} \exp(\text{sim}(\mathbf{z}_{\text{goal}}, \mathbf{z}_{k}^{\text{neg}})/\tau)}\right]
$$

- $\mathbf{z}_{\text{goal}}, \mathbf{z}_{\text{achieved}}$：video clip 的 frame embedding
- $\tau$：temperature
- $\mathbf{z}_k^{\text{neg}}$：负样本 embedding

这个 latent 既 encode 了 visual content（fine-grained perceptual），又 encode 了 task-relevant structure（affordance / value），可以直接作为 reward shaping 嵌入 DRL policy。

**R3M (Nair et al. 2022, [link](https://arxiv.org/abs/2203.12601))**：用 diverse human video pretrain time-contrastive objective，学 universal visual representation for manipulation。

**LIV (Ma et al. 2023, [link](https://proceedings.mlr.press/v202/ma23b.html))**：把 CLIP-style language-image alignment 和 value function 结合，让 latent 同时 encode semantic + value。

**Majumdar et al. 2023** (*NeurIPS*, [link](https://arxiv.org/abs/2310.12968)) 的综述 "artificial visual cortex for embodied intelligence" 直接点明：**pretrained vision encoder 能极大提升 long-horizon planning 和 generalization**。

这条路径的本质：**用大规模 vision pretraining 学到的 fine-grained perceptual latent 作为 implicit spatial world model**。这恰恰呼应 paper 的主张——spatial world model 必须 perceptually rich。

#### Path 4: Video models as imagination — 新前沿

最近 video generation model 开始展现出 spatial reasoning 能力：

- **Veo 3** ([Wiedemer et al. 2025](https://arxiv.org/abs/2509.20328))：用 next-scene prediction 做 zero-shot Sudoku、maze、navigation。
- **Genex** ([Lu et al. 2024](https://arxiv.org/abs/2412.09624))：generative explorable world。
- **Genie 3** ([DeepMind blog](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/))：action-controllable world model。

paper 把这叫做 **"imagination-as-video"**，类比人类 "running movies in the mind"。

VGGT (Visual Geometry Grounded Transformer, [Wang et al. 2025](https://arxiv.org/abs/2503.11651)) 用 3D reconstruction objective 预训练 vision backbone：

$$
\mathbf{F}^{(L)} = \text{Transformer}_\theta\Big(\text{ViT}(\mathbf{I}_1), \ldots, \text{ViT}(\mathbf{I}_N)\Big)
$$

$$
(\mathbf{P}_i, \mathbf{C}_i, \mathbf{D}_i) = \text{Heads}\Big(\mathbf{F}^{(L)}_i\Big)
$$

- $\mathbf{I}_i$：第 $i$ 个输入 view
- $\mathbf{F}^{(L)}$：transformer 最后一层 feature
- $\mathbf{P}_i \in SE(3)$：camera pose
- $\mathbf{C}_i$：confidence map
- $\mathbf{D}_i$：depth / point map

这种 model 把 scene geometry、dynamics、affordances 都 internalize 进 latent，正是 paper 倡导的方向。

DINOv3 ([Siméoni et al. 2025](https://arxiv.org/abs/2508.10104)) 也是同类——self-supervised vision feature，跨任务 transfer 强，可作 implicit spatial world model 的候选 backbone。

---

## 几个"啊哈" moment

### 1. PRM 就是 attention/gating 机制的神经科学版本

你在 nanoGPT 里跑 attention，本质是 "what should the model attend to"。PRM 是大脑版的 "what should enter awareness"——同一个 idea，不同 substrate。这暗示 spatial world model 在 AI 里可能需要 explicit 的 gating module，不是单纯的 encoder。

### 2. Aphantasia 不是 "无图像"，是 "无 decoding"

这点对 AI 设计直接有启发。你 train 了一个 VAE，encoder 能产出完美 latent，但 decoder 坏了——这就像 aphantasia。latent 依然可以用于 downstream task（spatial reasoning），只是不能"看见"。所以**不要用"能不能生成图像"来判断 spatial representation 是否存在**。

### 3. Visual pretraining for manipulation 验证了 paper 主张

VIP / R3M / LIV 这一系列工作的成功，本质上证明了一件事：**用 fine-grained perceptual latent 作为 implicit world model，比手写 physics state 强得多**。这就是 no free lunch 的实证——你想要 spatial competence，就得有 perceptual richness。

### 4. Video model 可能是 AI 版的 "mental imagery"

Veo 3 / Genie 3 / Genex 这种 generative world model，可以做 action-conditioned next-scene prediction。这和人类"在脑子里放电影"高度同构。paper 大胆假设：**video model 可能就是 AI 实现 spatial world model 的正确 vehicle**，因为它天然 encode 了 fine-grained perceptual content + spatiotemporal dynamics。

### 5. 和 LeCun JEPA 的微妙关系

LeCun 的 JEPA ([LeCun 2022](https://openreview.net/pdf?id=BZ5a1r-kVsf)) 主张在 latent space 而非 pixel space 做预测，表面看像 paper 反对的 "abstract spatial representation"。但仔细想：JEPA 的 latent 来自 vision encoder（V-JEPA 用 ViT features），latent 本身就是 fine-grained perceptual。区别在于 JEPA 把"rendering"延迟到下游——这和 PRM 框架完全兼容，只是把 discriminator / decoder 留给下游 task head。

### 6. 和 Butlin et al. 2023 (AI consciousness) 的呼应

[Butlin et al. 2023](https://arxiv.org/abs/2308.08708) 讨论 AI consciousness 的指标，把 PRM / HOSS 作为 candidate computational substrate。本 paper 隐含的主张：**spatial competence 是 consciousness indicator 之一**。这给"AI 是否有 spatial world model"和"AI 是否 conscious"之间画了一条隐线。

---

## 对你（Karpathy）的几个直接启发

1. **nanoGPT 教 spatial reasoning 的瓶颈**：纯 language model 学不会 spatial，不是参数不够，是 representation vehicle 不对。你想要 spatial competence，得在 architecture 层面引入 fine-grained perceptual encoding——video / vision encoder 是候选。

2. **Eureka Labs 的 curriculum 设计**：如果 spatial reasoning 依赖 fine-grained perceptual representation，那教 AI 空间推理应该从"看"开始，不是从"读"开始。video-based curriculum 可能比 text-based curriculum 更有效。

3. **Tesla Autopilot 的 implicit world model**：你自己讲过 autopilot 本质是预测下一个 scene state。这和 paper 主张高度一致——**fine-grained perceptual latent + dynamics prediction = implicit spatial world model**。Tesla 的 vision-based approach 比 LiDAR-based approach 更接近人类认知架构。

4. **对"emergent spatial ability"的 skepticism**：现在有人 claim 大模型 scale 上去 spatial ability 会 emergent。paper 的 no free lunch 主张直接泼冷水：**没合适的 representational vehicle，scale 多大都没用**。

---

## Reference links 汇总

主论文相关：
- Balaban & Ullman 2025: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(25)00048-X
- Goodale & Milner 1992: https://doi.org/10.1016/0166-2236(92)90388-K
- Fleming 2020 HOSS: https://doi.org/10.1093/nc/niz020
- Lau 2019 PRM: https://psyarxiv.com/8sg9n/
- Fleming & Shea 2024: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(24)00156-2
- Fleming & Michel 2025 BBS: https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/
- Kutsche et al. 2025 aphantasia lesion: https://www.medrxiv.org/content/10.1101/2025.05
- Bellet et al. 2022 PPC decoding: https://academic.oup.com/nc/article/doi/10.1093/nc/niac005/6585853
- Panagiotaropoulos 2024 Neuron: https://www.cell.com/neuron/fulltext/S0896-6273(24)00139-X
- Michel et al. 2025 aphantasia blindsight: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(25)00052-1
- Phillips 2025 Noûs: https://onlinelibrary.wiley.com/doi/abs/10.1111/nous.12500

AI / embodied intelligence 相关：
- Huh et al. 2024 Platonic: https://arxiv.org/abs/2405.07987
- VIP: https://arxiv.org/abs/2210.00030
- R3M: https://arxiv.org/abs/2203.12601
- LIV: https://proceedings.mlr.press/v202/ma23b.html
- Majumdar et al. 2023 artificial visual cortex: https://arxiv.org/abs/2310.12968
- VGGT: https://arxiv.org/abs/2503.11651
- Veo 3 zero-shot: https://arxiv.org/abs/2509.20328
- DINOv3: https://arxiv.org/abs/2508.10104
- Genesis: https://github.com/Genesis-Embodied-AI/Genesis
- Genex: https://arxiv.org/abs/2412.09624
- Genie 3 blog: https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/
- LeCun JEPA: https://openreview.net/pdf?id=BZ5a1r-kVsf
- Butlin et al. 2023 AI consciousness: https://arxiv.org/abs/2308.08708

VLM spatial reasoning eval:
- Gao et al. 2024: https://arxiv.org/abs/2410.00324
- Cai et al. 2025 GPT-5 spatial: https://arxiv.org/abs/2508.13142
- Zhang et al. 2024: https://arxiv.org/abs/2410.17385
- Luo et al. 2024: https://arxiv.org/abs/2410.00332
- Li et al. 2024: https://arxiv.org/abs/2410.10855

想深挖哪条线尽管说——比如 PRM 在 AI 里怎么实现 gating module、video world model 的 latent dynamics 具体怎么 train、或者怎么把这套框架接到你 Eureka Labs 的 curriculum 设计里。

---

# Rethinking the Simulation vs. Rendering Dichotomy: No Free Lunch in Spatial World Modelling 深度解析

这篇论文是一篇 position paper，由 University of Michigan 的 Dezhi Luo、Johns Hopkins 的 Qingying Gao、CMU 的 Hokin Deng 合作完成。核心论点可一句话概括为：**spatial world modelling 在 AI 中无法绕开 fine-grained perceptual content，企图用 schematic / amodal 的抽象表示走捷径是没有 free lunch 的**。下面按论文逻辑层层展开，同时引入相关技术细节、神经科学证据、AI 实证，以及公式化的概念框架，帮你 build intuition。

---

## 1. 论文要解决的核心问题

当前 foundation models（特别是 multimodal LLMs）在 spatial reasoning 任务上表现糟糕——mental rotation、perspective-taking、mechanical reasoning、perceptual constancy 等均暴露严重缺陷（参见 [Gao et al. 2024](https://arxiv.org/abs/2410.00324)、[Sun et al. 2025](https://arxiv.org/abs/2502.10273)、[Zhang et al. 2024](https://arxiv.org/abs/2410.17385)、[Cai et al. 2025](https://arxiv.org/abs/2508.13142)）。但与此同时，这些模型在 perception 和 high-level reasoning 上却很出色。为什么会出现这种 dissociation？

目前主流的 cognitive science 解释借用了 **simulation vs. rendering dichotomy**（Balaban & Ullman, 2025, *Trends in Cognitive Sciences*，[link](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(25)00048-X)）：把 mental imagery 拆成两部分——
- **Simulation (physics-based)**：在 structured spatial representation 上做物理推理，对应 dorsal stream；
- **Rendering (graphics-based)**：生成 conscious visual imagery，对应 ventral stream。

这听起来优雅，并且能够解释 aphantasia 患者（无法生成 voluntary visual imagery 的人）依然能完成 mental rotation 等任务的现象。但是这篇论文认为这种 **linear interpretation 是错误的**，并基于神经科学 + embodied AI 的证据提出替代性框架：simulation 与 rendering 共享同一类 higher-order perceptual representation，只是 gating / decoding 路径不同。

---

## 2. 关键论点 1：Aphantasia 不能证明 simulation / rendering 分离

### 2.1 Spatial imagery framework 的内在矛盾

Balaban & Ullman 主张 aphantasia 患者保留 **spatial imagery**（一种 amodal、modality-neutral 的 schematic representation），缺失的只是 **visual/object imagery**。Phillips (2025, *Noûs*) 甚至形容 spatial imagery "neutral as to whether the location, relation, shape or structure is imagined as seen or touched"。

但论文指出这个定义自相矛盾：
- 一方面声称 imagery "modality-neutral"；
- 另一方面描述其使用时又不可避免地借助 "subjects imagine grasping the shape and rotating it"——这其实**暗中招募了 embodied sensory modalities**。

如果坚持 spatial imagery 不需要 conscious experience，那它就退化成 **unconscious mental imagery**（Michel et al. 2025, *TICS*，[link](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(25)00052-1)）；但 Scholz et al. (2025, *Current Biology*) 指出 aphantasia 患者早期 visual cortex 缺乏 shared representation，这就让 "imagery" 这个词在 aphantasia 情境下几乎无意义。

作者的结论：**与其纠结 aphantasia 患者用的是不是 "imagery"，不如承认他们依赖 spatial representations（无论是否 imagistic）来完成任务，只是缺乏相应的 visual experience**。这一区分对 AI 的启示很关键——任务表现与 phenomenal experience 不能等同。

### 2.2 Dorsal / Ventral Stream 的 Linear Interpretation 站不住脚

Balaban & Ullman 把 simulation 映射到 dorsal stream（"zombie stream"，Goodale & Milner, 1992, *TINS*，[link](https://doi.org/10.1016/0166-2236(92)90388-K)），把 rendering 映射到 ventral stream。但论文列举大量证据反对这种 **strict functional specialization**：

**证据 1：Dorsal stream 高级区域直接编码 perceptual content**
- Bellet et al. (2022, *Neuroscience of Consciousness*): 在 PPC（传统 dorsal "zombie" 区域），即使没有 behavioral report，也能 above-chance 解码 rapidly presented stimuli 的 object identity。这说明 PPC 主动编码 perceptual information，并非仅做 visuomotor transformation。
- DLPFC（dorsolateral prefrontal cortex）与 visual awareness 直接相关（Lau & Passingham, 2006, *PNAS*；Anzulewicz et al., 2019）。
- Prefrontal 与 parietal lesion 损害 visual content 的整合与维持（Szczepanski & Knight, 2014；Persaud et al., 2011）。

**证据 2：Fronto-parietal network 整合 simulation 与 rendering**
- Panagiotaropoulos (2024, *Neuron*): prefrontal cortex 在 consciousness 中起 integrative 作用，并非 isolated control system。
- Rees (2007) 和 Wu (2014, *Mind & Language*) 都强调 dorsal stream 对 visual experience 的直接贡献，特别是 VIP/LIP 这样的 intraparietal 区域维护 egocentric spatial framework。

**证据 3：Aphantasia 的 lesion pattern**
- Kutsche et al. (2025, *medRxiv*): 12 例 lesion-induced aphantasia 全部涉及 **left fusiform gyrus** 连接区域，**prefrontal cortices 完好**。这暗示 aphantasia 是 downstream decoding 失败，而非 higher-order representation 缺失。

这指向一个根本性结论：**dorsal-ventral 不是 linear 分工，而是 non-linear hierarchical 的交互网络**。

---

## 3. 关键论点 2：Higher-Order Theories (HOT) 提供统一框架

论文提出用 **Higher-Order Theories of consciousness** 重新组织 simulation 与 rendering 的关系。具体涉及两个模型：

### 3.1 Perceptual Reality Monitoring (PRM) — Lau (2019, [link](https://psyarxiv.com/8sg9n/))

PRM 把意识门控比喻为 GAN 的 discriminator。形式化地：

$$
D_{\text{PRM}}(\mathbf{h}^{(L)}) = \sigma\left(\mathbf{W}_D^\top \cdot \phi(\mathbf{h}^{(L)}) + b_D\right)
$$

其中：
- $\mathbf{h}^{(L)}$：first-order perceptual representation（例如来自 ventral stream 的 latent）；
- $\phi(\cdot)$：higher-order meta-representation function，由 fronto-parietal network 实现；
- $\mathbf{W}_D, b_D$：discriminator 的参数；
- $\sigma$：sigmoid，输出 $\Pr[\text{real} \mid \mathbf{h}^{(L)}]$。

如果 $D_{\text{PRM}}$ 判定该 state 足够 "real"，就生成一个 **pointer** $p(\mathbf{h}^{(L)})$，携带 location index，反馈给 first-order networks 做 decoding，从而 "render" 出 conscious experience：

$$
\mathbf{x}_{\text{conscious}} = \text{Decode}\left(p(\mathbf{h}^{(L)}), \mathcal{M}_{\text{first-order}}\right)
$$

这里 $\mathcal{M}_{\text{first-order}}$ 是 first-order sensory cortices 维护的 content store。

**Aphantasia 在 PRM 框架下**：discriminator 工作正常（prefrontal cortex 完好），但 downstream decoding（pointer → first-order cortex）失败，因此缺乏 conscious visual experience，但 higher-order representation 仍可用于 spatial reasoning。这与 Kutsche et al. 的 lesion evidence 吻合。

### 3.2 Higher-Order State Space (HOSS) — Fleming (2020, *Neuroscience of Consciousness*, [link](https://doi.org/10.1093/nc/niz020))

HOSS 强调 higher-order representations 编码 **perceptual states 之间的 structured quality space relations**：

$$
\mathcal{H} = \{(\mathbf{q}_i, \mathbf{q}_j, R_{ij})\}_{i,j=1}^{N}
$$

其中：
- $\mathbf{q}_i, \mathbf{q}_j$：first-order perceptual states；
- $R_{ij}$：在 quality space 中的 relation（如相似度、距离、相对位置）。

Fleming & Shea (2024, *TICS*) 强调这些 higher-order indices 本身不直接是 conscious，必须被 discriminated 为 "reality vs. imagination" 才能 enter awareness。

### 3.3 这一框架对论文主旨的意义

PRM 和 HOSS 共同指向：**simulation 与 rendering 共享 representational substrate，区别仅在于是否被 gating / decoding 进入 conscious experience**。这彻底打破了 Balaban & Ullman 的 linear dichotomy。换言之，spatial world models 的底层 representation **本质上就是 fine-grained perceptual**，因为 fronto-parietal higher-order indices 本身就是 fine-grained 的（包括 perceived distances, object relations 等 spatial properties）。

可以用一个简化的架构图（conceptual）表示：

```
   First-Order Sensory (V1/V2/V4)  +  First-Order Spatial (VIP/LIP)
                \                              /
                 \                            /
                  v                          v
              ┌─────────────────────────────────────┐
              │  Fronto-Parietal Higher-Order Net    │
              │   (HOSS / PRM meta-representations)  │
              └─────────────────────────────────────┘
                        /                     \
                       /                       \
                      v                         v
        Discriminator D_PRM            Downstream Decoder
        (reality vs imagination)        (fusiform / V1 feedback)
                      │                         │
                      v                         v
        Pointer for decision-making      Conscious visual
        (spatial reasoning intact)        experience (rendering)
```

Aphantasia = 右侧路径断；左侧路径完好 → 任务表现保留 + 视觉体验缺失。

---

## 4. 论文核心主张：No Free Lunch in Spatial World Modelling

把 HOT 框架推到 AI，作者得出一个强主张：

> **如果人类 spatial reasoning 依赖 fine-grained perceptual encodings，那么企图用 coarse、perceptually abstract 的 approximations 来 emulate 这种能力是不可能的——no free lunch。**

这条主张与 Fleming & Michel (2025, *BBS*) 的进化论观点呼应：**conscious visual experience 可能正是为了 stabilize 内部 simulation 而进化出来的一种 gating 机制**，让 organism 知道何时 commit 到 world model 并采取行动。意识本身可能就是 spatial reasoning 的 representational substrate 的副产品。

---

## 5. AI 实证：三种路径的对比

### 5.1 Language Models 不是 spatially competent

MLLMs（LLaVA, GPT-4V, Gemini）尽管 visual prior 丰富，但在 mental rotation、perspective-taking、mechanical reasoning 上持续失败（[Gao et al. 2024](https://arxiv.org/abs/2410.00324)、[Luo et al. 2024](https://arxiv.org/abs/2410.00332)、[Li et al. 2024](https://arxiv.org/abs/2410.10855)、[Wang et al. 2025a](https://arxiv.org/abs/2502.10273)）。Huh et al. (2024, [Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)) 提出 modality 之间会向 shared statistical abstraction 收敛，但作者指出 **在 spatial domain 这种 convergence 无法发生，因为缺少能 encoding perceptually rich representations 的 vehicle**。

### 5.2 Implicit Models for Embodied Control 的胜利

纯 explicit physics engines（MuJoCo [Todorov et al. 2012](https://doi.org/10.1109/IROS.2012.6386109)、Isaac Gym [Makoviychuk et al. 2021](https://arxiv.org/abs/2108.10470)、Genesis [Zhou et al. 2024b](https://github.com/Genesis-Embodied-AI/Genesis)）虽然 physics 精确，但 sim-to-real 转移脆弱，open-ended visual 复杂场景中 policy 容易 brittle。

转折点是 **visual pretraining for manipulation** 范式：
- **VIP** (Value-Implicit Pre-training, [Ma et al. 2022](https://arxiv.org/abs/2210.00030))：用 contrastive value-implicit objective 学 visual embedding $\mathbf{z} = f_\theta(\text{video clip})$，作为 reward 嵌入 DRL policy optimization。

  目标函数形式：
  $$
  \mathcal{L}_{\text{VIP}} = -\mathbb{E}_{(\tau_i, \tau_j) \sim \mathcal{D}}\left[\log \frac{\exp(\text{sim}(\mathbf{z}_i^{\text{goal}}, \mathbf{z}_j^{\text{achieved}})/\tau)}{\sum_k \exp(\text{sim}(\mathbf{z}_i^{\text{goal}}, \mathbf{z}_k^{\text{neg}})/\tau)}\right]
  $$
  其中 $\tau$ 是 temperature，$\mathbf{z}_i$ 是 video embedding，目标是让 goal state 与 achieved state 在 embedding space 中接近。

- **R3M** (Reusable Representations for Robotic Manipulation, [Nair et al. 2022](https://arxiv.org/abs/2203.12601))：用 diverse human video pretrain 时间-对比 objective 学 universal visual representation。

- **LIV** (Language-Image-Value, [Ma et al. 2023](https://proceedings.mlr.press/v202/ma23b.html))：将 CLIP-style language-image alignment 与 value function 结合。

- **Majumdar et al. 2023** (*NeurIPS*，[link](https://arxiv.org/abs/2310.12968))：综述 "artificial visual cortex for embodied intelligence"，证明 pretrained vision encoder 能极大提升 long-horizon planning 和 generalization。

这些方法实际上把 vision foundation model 的 latent 当作 **implicit world model**——它们 embed structured spatial priors，正是论文主张的 "fine-grained perceptual representation"。

### 5.3 Video Models for Action Imagination

最近的视频生成模型（Veo 3 [Wiedemer et al. 2025](https://arxiv.org/abs/2509.20328)、Genex [Lu et al. 2024](https://arxiv.org/abs/2412.09624)、Genie 3 [Parker-Holder & Fruchter blog](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)）开始 zero-shot 解决 Sudoku、maze、navigation 等 visual reasoning 任务。论文把这种 **"imagination-as-video"** 比作人类 "running movies in the mind"。

VGGT (Visual Geometry Grounded Transformer, [Wang et al. 2025b](https://arxiv.org/abs/2503.11651)) 用 3D reconstruction objective 预训练 large vision backbone，能 capture fine structural detail。它的 forward pass 大致是：

$$
\mathbf{F}^{(L)} = \text{Transformer}_\theta(\text{ViT}(\mathbf{I}_1), \dots, \text{ViT}(\mathbf{I}_N))
$$

$$
(\mathbf{P}_i, \mathbf{C}_i, \mathbf{D}_i) = \text{Heads}(\mathbf{F}^{(L)}_i)
$$

其中：
- $\mathbf{I}_i$：第 $i$ 个输入 view 的 image；
- $\mathbf{F}^{(L)}$：transformer 最后一层 feature；
- $\mathbf{P}_i$：camera pose（SE(3)）；
- $\mathbf{C}_i$：confidence；
- $\mathbf{D}_i$：depth map / point map。

这种 model 内化了 scene geometry、dynamics、affordances，正是 paper 倡导的方向。

DINOv3 (Siméoni et al. 2025, [link](https://arxiv.org/abs/2508.10104)) 也是值得一提的 vision foundation model，其 self-supervised feature 在跨任务 transfer 上表现强劲，可作 implicit spatial world model 的候选。

---

## 6. 整体论证链梳理

可以用下面这个推理链条总结全篇：

1. **Observation**: MLLMs 在 spatial reasoning 上失败，cognitive science 试图用 simulation/rendering dichotomy 解释 aphantasia；
2. **Critique**: 该 dichotomy 的 linear interpretation（dorsal=zombie / ventral=conscious）与神经科学证据冲突；
3. **Alternative**: HOT (PRM/HOSS) 提供统一框架——simulation 与 rendering 共享 higher-order fine-grained perceptual representation；
4. **Implication for AI**: 若人类如此工作，AI 也无法绕开 fine-grained perceptual encoding；
5. **AI Evidence**: Visual pretraining for manipulation（VIP/R3M/LIV）+ video world models（Veo 3、Genie 3、Genex）实证支持 fine-grained perceptual grounding 的重要性；
6. **Call to Action**: 发展能 maintain structured perceptual representations 的 architecture，作为 spatial world modeling 的基础。

---

## 7. 几个值得深挖的延伸

### 7.1 与 LeCun JEPA 的关系
LeCun (2022, [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)) 提出 JEPA (Joint Embedding Predictive Architecture)，强调在 latent space 而非 pixel space 做预测。表面看这与 paper 主张"perceptually rich"相冲突，但 JEPA 的 latent 仍来自 vision encoder，且 Hiera-V-JEPA 等确实使用 fine-grained spatiotemporal features。可以说 JEPA 是 paper 主张的一种可能 implementation，只是把"rendering"延迟到下游。

### 7.2 与 Butlin et al. (2023) Consciousness in AI 的对话
[Butlin et al. 2023](https://arxiv.org/abs/2308.08708) 讨论 AI consciousness 的指标，引用 PRM/HOSS 作为 candidate computational substrate。本论文可视为把这一讨论与 spatial reasoning 结合，提出 "spatial competence 是 consciousness indicator 之一" 的隐含主张。

### 7.3 Affordance 与 Spatial World Models
论文提到 video model 能 internalize "affordances"。这与 Gibson 的 ecological perception 直接相关。一个 affordance-aware spatial world model 形式上可写为：

$$
\mathcal{W}: (\mathbf{s}_t, \mathbf{a}_t) \mapsto (\mathbf{s}_{t+1}, \mathbf{A}_{t+1})
$$

其中 $\mathbf{A}_{t+1}$ 是 affordance field（每个 candidate action 的可执行性），$\mathbf{s}_t$ 是 structured perceptual state。这与 Genie 3 的 action-controllable video generation 高度同构。

### 7.4 对 World Models (Ha & Schmidhuber 2018) 的重新审视
经典的 World Models 用 VAE + MDN-RNN + Controller，把 perception 压缩到 latent $\mathbf{z}$，再在 $\mathbf{z}$ 上做 dynamics。从本论文视角看，VAE latent 太 coarse，丢失 fine-grained perceptual detail，因此难以支撑 robust spatial reasoning。Genex / Genie 3 这种 high-resolution video diffusion world model 才更接近"fine-grained"。

### 7.5 Hinton 的 "Imagery without Arrays" 伏笔
论文引用 Hinton (1979, *BBS*) 关于 imagery without arrays 的早期观点。这暗示 Hinton 长期主张 spatial representation 应该是 structured（如 reference frames + transformations）而非 pixel-array。这与本论文的 fine-grained perceptual representation 立场相容——只要 representation 能 capture geometric relations。

---

## 8. 一句话总结

**Spatial world models 不能用 schematic / amodal 抽象走捷径，因为人类 spatial reasoning 的 representational substrate 与 conscious visual experience 共享 fronto-parietal higher-order encoding；AI 应该向 vision foundation model + video world model 范式收敛，以 fine-grained perceptual representation 作为 implicit spatial world model 的 backbone。**

---

## 参考 web links 汇总

- 主论文: [arXiv (即将)](https://arxiv.org/)（暂未上 arXiv，作者机构页面可查询）
- Balaban & Ullman 2025: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(25)00048-X
- Goodale & Milner 1992: https://doi.org/10.1016/0166-2236(92)90388-K
- Fleming 2020 HOSS: https://doi.org/10.1093/nc/niz020
- Lau 2019 PRM: https://psyarxiv.com/8sg9n/
- Fleming & Shea 2024: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(24)00156-2
- Fleming & Michel 2025 BBS: https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/
- Kutsche et al. 2025 (aphantasia lesion): https://www.medrxiv.org/content/10.1101/2025.05
- Bellet et al. 2022: https://academic.oup.com/nc/article/doi/10.1093/nc/niac005/6585853
- Panagiotaropoulos 2024: https://www.cell.com/neuron/fulltext/S0896-6273(24)00139-X
- Butlin et al. 2023 (AI consciousness): https://arxiv.org/abs/2308.08708
- Huh et al. 2024 (Platonic): https://arxiv.org/abs/2405.07987
- VIP: https://arxiv.org/abs/2210.00030
- R3M: https://arxiv.org/abs/2203.12601
- VGGT: https://arxiv.org/abs/2503.11651
- Veo 3 zero-shot: https://arxiv.org/abs/2509.20328
- DINOv3: https://arxiv.org/abs/2508.10104
- Genesis: https://github.com/Genesis-Embodied-AI/Genesis
- Genex: https://arxiv.org/abs/2412.09624
- Genie 3 blog: https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/
- LeCun JEPA: https://openreview.net/pdf?id=BZ5a1r-kVsf
- Gao et al. 2024 (VLM spatial eval): https://arxiv.org/abs/2410.00324
- Cai et al. 2025 (GPT-5 spatial): https://arxiv.org/abs/2508.13142
- Zhang et al. 2024 (VLM space): https://arxiv.org/abs/2410.17385
- Michel et al. 2025 (aphantasia blindsight): https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(25)00052-1
- Phillips 2025: https://onlinelibrary.wiley.com/doi/abs/10.1111/nous.12500

如果你想进一步深挖某一条线（比如 PRM 的 GAN 形式化、VIP 的具体实现细节、Genie 3 的 latent dynamics、或者把 paper 主张接到你自己关于 nanoGPT / Eureka Labs 的直觉），可以告诉我，我可以继续展开。
