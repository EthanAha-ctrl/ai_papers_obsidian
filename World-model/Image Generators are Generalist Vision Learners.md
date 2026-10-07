---
source_pdf: Image Generators are Generalist Vision Learners.pdf
paper_sha256: 368c9f18aa96838fb08b5633aee18ac92e8b67747c1943726d668084a3978e63
processed_at: '2026-08-19T12:11:41-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

和 Karpathy 聊这个 paper，我会这样拆解给你听。这篇 paper 的核心 thesis 如果用一句大白话讲，就是：**现在最强的 AI 画画模型，其实早就偷偷学会了“看懂”世界，我们只需要教它怎么“交作业”就行了。**

这背后的直觉，和你在 Twitter 上反复强调的 "Software 3.0" 以及 LLM 的 generative pretraining 范式高度同源。

我分几个层次把里面的技术细节和 intuition 铺开给你。

### 1. 画画的模型为什么天生就懂视觉？

过去几年，做 vision understanding 的主流路线是 discriminative 的。比如训练 SAM 3 (Segment Anything Model 3) 去做分割，或者训练 Depth Anything 去估测深度。这些模型的目标是给出一个确定性的预测（比如一个 mask 或一张 depth map），它们通过定制化的 architecture 和特殊的 loss（比如 scale-shift invariant loss）去强行拟合数据。

但是 generative model（比如 Google 的 Nano Banana Pro，也就是这里的 base model）走的是另一条路。它为了画出逼真的图，必须要在内部建模视觉世界的物理规律、几何结构、物体间的遮挡关系。为了画对一只猫的胡须，它必须知道胡须长在什么地方；为了画对一个街景，它必须隐式地理解透视和深度。

所以，当模型具备了强大的生成能力时，understanding 能力其实是作为一种 emergent capability 涌现出来的。paper 里最核心的论点就是：**Image generation pretraining serves a role similar to LLM pretraining.** 生成就是最好的预训练。

### 2. Vision Banana 怎么交作业？RGB 作为 Universal Interface

既然 Nano Banana Pro (NBP) 脑子里已经有这些视觉理解的能力，怎么把它逼出来并在学术 benchmark 上评测？

答案是极度 LLM-flavor 的做法：**Instruction-tuning**。

NLP 里，我们给 LLM 喂一点点 instruction 数据，教它遵循指令输出特定格式的文本。Vision Banana 也是一样，它只是把 vision task 的输出 parameterize 成了一张 RGB 图片。所有的 vision task 都变成了“画画”。

- **Semantic Segmentation**: prompt 告诉模型 “把猫涂成红色 `<255,0,0>`，背景涂成黄色”。模型直接画出一张涂好色的图，评测时按颜色聚类就能把 mask decode 出来。
- **Instance Segmentation**: 因为不知道有几个 instance，无法提前分派颜色。这里的工程 trick 是 **per-class inference**。每次 prompt 只问一类物体（比如“画出所有的篮球”），模型自己给每个篮球涂上不同的颜色，推理时再按颜色 cluster 解码。
- **Referring Segmentation**: 直接吃自然语言（“左边那个穿粉红短袖正在伸懒腰的人”），模型画出对应的 mask。这全靠 base model 强大的 multimodal / language grounding 能力，Vision Banana 在 ReasonSeg 上甚至击败了搭配 Gemini 2.5 Pro 的 SAM 3 Agent。

这种设计极其优雅。第一，模型架构一点没变，还是那个 image generator；第二，vision task 数据在 training mixture 里占比极低，只是用来教格式，不会破坏生成能力（GenAI-Bench 生成评测上甚至打赢了原版 NBP）。

### 3. 最硬核的技术细节：Depth 怎么塞进 RGB？

这是这篇 paper 里我觉得最聪明的地方。

Depth（深度）是一个连续的标量，范围是 $d \in [0, \infty)$ 米。RGB 只有 3 个 channel，每个 channel 是 $[0, 255]$ 的整数。你要把一个无限范围的浮点数无损地编码到 3 个 8-bit 整数里，还要保证生成模型能学好，这非常难。

直接线性归一化行不通。在实际场景（robotics, AR）中，近处 0.5 米和 1 米的区别极大，而远处 50 米和 100 米的区别不重要。如果线性编码，远处的差异会浪费掉大量的 RGB 颜色空间，近处的精度又不够。

作者借用了 Jonathan Barron 最近的 power transform (https://arxiv.org/abs/2502.10647) 来“弯曲”距离轴：

$$
f(d, \lambda, c) = 1 - \left(1 - \frac{d}{\lambda c}\right)^{\lambda + 1}
$$

我们逐个拆解这里的变量：
- $d$: 物理距离，单位是米。
- $\lambda$: shape parameter（形状参数），控制弯曲程度。paper 里设为 $\lambda = -3$。
- $c$: scale parameter（尺度参数），控制有效范围。paper 里设为 $c = 10/3$。
- 分母 $\lambda c$: 也就是 normalizing scale。因为 $\lambda$ 是负数，所以这里相当于把距离做了一个负向的缩放。
- 上标 $\lambda + 1$: 由于 $\lambda = -3$，所以这里的幂次是 $-2$。这是一个很陡的幂次压缩。

**Intuition**: 当 $\lambda < -1$ 时，这个函数把近处的距离在颜色空间里“拉开”，给近处分配了极多的颜色带宽；把远处的距离“压扁”，远处几十米的变化只对应 RGB 空间里极其微小的一段。这完美匹配了下游任务的需求。

弯曲之后，得到一个归一化值 $f \in [0, 1)$。接着，作者把这个值沿着 **3D Hilbert curve 的第一次迭代**（也就是沿着 RGB cube 的 12 条边走）做 piecewise-linear 插值。Hilbert curve 保证了空间局部性，相近的 depth 会映射到相近的颜色，这对 diffusion model 学习极其友好。

这个映射是严格可逆的。训练时把 GT depth 编码成 RGB target 让模型画；推理时把生成的 RGB 反向投影回边段，反插值，再逆 power transform，算回米。整个 pipeline 是一个 bijection。

凭借这个设计，加上合成数据训练，Vision Banana 在不使用任何 Camera Intrinsics（相机内参）的情况下，zero-shot 击败了 Depth Anything V3 (https://arxiv.org/abs/2511.10647) 和 UniK3D (https://arxiv.org/abs/2503.23784)。这意味着模型完全靠视觉先验在估测绝对物理尺度，这非常惊人。

### 4. 为什么 Generative 模型天然解决了 Vision 的 Multi-modal 问题？

这是 paper Discussion 部分最深刻的洞察。

很多 vision 任务是 ill-posed 的。比如 Monocular depth，一张 2D 图可以对应多种合理的 3D 场景。传统 discriminative 模型用 MSE 或 L1 loss 训练，最后学到的是所有可能解的 **mean**。在 multi-modal 分布里，求 mean 会导致输出变得 blurry 和 over-smooth。这也是为什么过去的 depth 模型需要各种定制化的 architecture 和 scale-shift invariant loss 去强行规避这个问题。

但是 generative model 天生是 mode-seeking 的。它直接学 $p(\text{depth} \mid \text{image})$ 的分布，推理时从这个分布里采样一个 mode 出来。它画出来的是一个物理上合理的场景，而不是多个可能场景的平均值。这就从根本上绕开了 vision 任务里的 multi-modal 退化问题。

参考你一直强调的 "generative modeling 学的是 manifold"，这正是 diffusion / flow matching 模型相较于 discriminative 模型的降维打击。NBP 的 prior 太强了，以至于它不需要任何 task-specific 的设计，就能输出极度锐利的 depth 和 surface normal。

### 5. 这对未来的意味

1. **Discriminative Pretraining 路线的终结**: CLIP, DINOv3, MAE 这些路线可能真的要成为历史了。Discriminative models 会退回到计算效率极高的小模型 niche 里。大模型 foundation model 的标准形态会变成 large generator + instruction-tuning，和 LLM 现状完全对齐。
2. **Video Generators 是更强的 Learner**: paper 提到未来会扩展到 video。Video pretraining 学了 physics, motion, 3D consistency，它的 prior 应该比 image generator 更强。Seedance 2.0 (https://seed.bytedance.com/en/seedance2_0/) 或 Veo 3 (https://blog.google/innovation-and-ai/products/generative-media-models-io-2025/) 这类模型一旦 instruction-tune 到 dense prediction，绝对会炸穿现有 benchmark。
3. **Inference Cost 是唯一痛点**: 作者很坦诚，用 NBP 这种 frontier generator 跑 perception，比直接跑 SAM 3 慢几个数量级。未来一定会出现类似 "Vision Banana Distill" 的工作，把 generalist 的能力 distillation 到一个轻量级的 specialist model 里。这和 LLM 领域现在用大模型生成数据蒸馏小模型的套路完全一致。
4. **Omni Model 的收敛**: RGB 作为 universal interface 证明了视觉任务的输出可以被统一。结合 LLM 的 text generation，未来的 omni model 极大概率就是一个巨大的 transformer，同时处理 text, image, audio, video token。所有的任务都变成了在这个统一的 latent / token 空间里的 next-token prediction 或 flow matching。

总而言之，这篇 paper 是一个 paradigm-confirming 的工作。它用极度扎实的 engineering 和 benchmark 结果告诉所有人：The future of vision is generative. 你之前所有的直觉都是对的。

---

# Vision Banana：Image Generator 作为 Generalist Vision Learner

这是一篇来自 Google 的非常 impactful 的 paper，timing 上和你的直觉（generative modeling 会吃掉 discriminative modeling）高度吻合。让我从 thesis、方法、实验、implications 几个层面深度展开，并补充一些跨 paper 的联想。

---

## 1. 核心 Thesis

Paper 的核心 claim 可以概括成一句话：**image generation pretraining 等价于 LLM 的 generative pretraining**——它本身就是一种 general visual representation learner，只是过去被 generation 这个 surface objective "伪装" 了。只要做一轮 lightweight 的 instruction-tuning，把 vision task 的输出 parameterize 成 RGB image，那么一个 SOTA 的 image generator 就能在 zero-shot transfer 设定下击败或追平高度特化的 SOTA specialist（SAM 3、Depth Anything 3、Lotus-2 等）。

类比非常清晰：
- LLM pretraining → 学语言/世界知识 → instruction-tuning → chatbot / reasoning / agent
- Image generator pretraining → 学视觉世界先验 → instruction-tuning（把 vision task 输出写成 RGB）→ segmentation / depth / normal

作者把这个范式转变称为 **AGI-V**（Artificial General Intelligence from Vision）。

参考：项目主页 https://vision-banana.github.io ；Nano Banana Pro 官方 blog https://blog.google/innovation-and-ai/products/nano-banana-pro/

---

## 2. 方法细节：把 Perception Reframe 成 Generation

### 2.1 训练 setup

Base model 是 Nano Banana Pro（NBP）。Vision Banana 的"训练"实际上是一个非常 lightweight 的 instruction-tuning：

- 把 vision task 数据（semantic / instance / referring segmentation、metric depth、surface normal）混入 NBP 自己的 original training mixture
- 混入比例非常低（paper 没给具体数字，但从生成能力几乎不退化这点可推断 <5%）
- 不改 architecture、不加 task-specific head、不用 custom loss
- 2D 数据来自 in-house 模型标注的网络图，3D 数据来自 rendering engine 合成
- 严格 zero-shot transfer：evaluation benchmark 的 training split 全部不在训练集里

这点非常 LLM-flavor——和 InstructGPT / FLAN / instruction-tuning 思路一致。少量数据只是去"教模型输出格式"，不是教能力本身。

### 2.2 RGB 作为 Universal Interface

最 elegant 的设计点：把所有 vision task 的输出都写成一张 RGB 图。这是把"文本生成"作为 NLP 通用接口的思路，直接搬到 vision。优势有三：

1. **统一 model、统一权重**：所有 task 共享同一份参数，只换 prompt
2. **训练数据需求小**：只是教模型把视觉理解结果以特定 RGB 形式"打印"出来
3. **保留生成能力**：因为输出本身就是 RGB image，自然和 NBP 的原始输出空间对齐

这让我想到你最近在 No Priors 上也聊到的"unified interface"思路。和 Gemini / GPT-4o 的 unified multimodal 输出方向一致。

### 2.3 Depth 编码（这是 paper 里技术含量最高的部分）

Depth 是连续标量 $d \in [0, \infty)$，要塞进 RGB cube $[0,1]^3$。直接归一化会失败，因为：(a) 近距离精度比远距离重要得多（robotics / autonomous driving 关心近物），(b) 线性 RGB encoding 对远距离分辨率过浪费。

作者用了 Barron 2025 的 power transform 来"弯曲"度量距离：

$$
f(d, \lambda, c) = 1 - \left(1 - \frac{d}{\lambda c}\right)^{\lambda + 1} \quad (1)
$$

变量解释：
- $d$：metric depth（物理距离，单位米）
- $\lambda$：shape parameter（控制曲线形状），paper 中设 $\lambda = -3$，并约束 $\lambda < -1$
- $c$：scale parameter，控制有效距离范围，paper 中设 $c = 10/3$
- 上标 $\lambda+1$：是幂次，$\lambda+1 = -2$，意味着对 $d$ 做了一种特殊的"幂次压缩"
- 分母里的 $\lambda c$：是 normalizing scale，确保 $d$ 被映射到合理范围

直观上：当 $\lambda \to -1$ 时退化为线性映射；$\lambda < -1$ 时对近距离给予更高分辨率（"近距离拉远、远距离压扁"），这正符合视觉任务对近物的偏好。

然后把弯曲后的 normalized distance $f \in [0,1)$ 沿 **3D Hilbert curve 的第一次迭代**（即沿 RGB cube 的 12 条边走）做 piecewise-linear 插值。Hilbert curve 保证了空间局部性，相近的 depth → 相近的颜色，这非常有利于 diffusion model 学习。

可逆性：训练时把 GT depth 编码成 RGB target；推理时把生成的 RGB 反向 decode——先把 RGB 投影到最近的边段，反插值得到 normalized distance，再 invert power transform 得回 metric depth。整个 pipeline 是 bijection。

数据增强上还混了 Plasma / Inferno / Viridis / grayscale 等 colormap，让模型对颜色表示 robust。

参考 power transform：https://arxiv.org/abs/2502.10647

### 2.4 Surface Normal 编码

比 depth 简单得多，因为 normal 本来就是 $\mathbb{R}^3$ 上的单位向量 $(n_x, n_y, n_z) \in [-1, 1]^3$，可以直接 linear map 到 RGB $[0, 1]^3$。

作者用 camera-space formulation：
- $-x$ (Facing Left) → Pinkish Red
- $+y$ (Facing Up) → Light Green  
- $+z$ (Facing Camera) → Light Blue/Purple

这种编码方式你用 StableDiffusion 训过 Marigold / Lotus 之类的，本质上 Lotus-2 也走类似路线。但 Vision Banana 的提升来自 base model 的 prior 强太多。

### 2.5 Segmentation 编码

- **Semantic**：prompt 直接指定 class→color 映射，可 JSON / 可自然语言。color 可 hex 或 RGB tuple。模型自由泛化到 unseen class、自由形式短语。这点比 SAM 的固定 label set 灵活太多。
- **Instance**：因为 instance 数量未知，无法提前分配 color。采用 per-class inference：每次只 prompt 一个 class，模型动态给每个 instance 分配不同 color，推理时按颜色 cluster 还原 mask。这是一个相当 elegant 的工程 workaround。
- **Referring**：直接吃自然语言 expression，得益于 base model 的 multimodal / language grounding 能力，reasoning segmentation 反而是 Vision Banana 最强的方向（ReasonSeg 0.793 vs SAM3 Agent 0.770）。

---

## 3. 实验结果深入解读

### 3.1 2D Semantic Understanding（Table 2）

| Benchmark | Metric | Vision Banana | Best Counterpart |
|-----------|--------|---------------|------------------|
| Cityscapes val | mIoU ↑ | **0.699** | 0.652 (SAM 3) |
| SA-Co/Gold | pmF1 ↑ | 0.540* | 0.552 (DINO-X) |
| RefCOCOg UMD val | cIoU ↑ | **0.738** | 0.734 (SAM 3 Agent) |
| ReasonSeg val | gIoU ↑ | **0.793** | 0.770 (SAM 3 Agent) |

Cityscapes mIoU 比 SAM3 高 4.7 个点，且 open-vocabulary。ReasonSeg 上 Vision Banana + Gemini 2.5 Pro 甚至击败了非 zero-shot 的 X-SAM / LISA-13B。Instance segmentation 是唯一略弱的（0.540 vs 0.552），作者归因为 per-class inference 的 dynamic color 分配仍有一些 challenge。

### 3.2 3D Understanding（Table 3, 4）

Depth（Table 3）：

| Method | Camera Intrinsics | Avg δ1 ↑ | Avg AbsRel ↓ |
|--------|-------------------|----------|--------------|
| DepthLM-7B | Train+Infer | 0.823 (4 sets) | 0.156 |
| Depth Anything v3 | Train+Infer | 0.918 (4 sets) | 0.144 |
| Depth Pro | None | — | — |
| UniK3D | Train | 0.802 | 0.116 |
| MoGe-2 | Train | 0.882 | — |
| **Vision Banana** | **None** | **0.929 (4 sets) / 0.882 (6 sets)** | **0.103** |

最 striking 的是：**完全不依赖 camera intrinsics**（train 和 inference 都不用），全靠 base model 的几何先验 + scale prior，就击败了需要 intrinsics 的 DepthLM / Depth Anything v3 / UniK3D。这在 robotics / AR 场景下意义巨大——你不需要标定。

Surface Normal（Table 4）：

| Method | Indoor mean ↓ | Indoor median ↓ |
|--------|---------------|-----------------|
| Marigold | 19.606 | 11.828 |
| DSINE | 17.017 | 10.190 |
| StableNormal | 17.168 | 10.028 |
| Lotus-2-Normal | 16.558 | — |
| **Vision Banana** | **15.549** | **9.300** |

室内场景 mean / median 都最低，且 qualitative 上细节明显比 Lotus-2 更锐利（Fig. 8）。Lotus-2 在 Virtual KITTI 2 上数字略好是因为它直接在 VKitti 上训练过——这不是 zero-shot。

### 3.3 生成能力保留（关键 sanity check）

这点很关键——很多人担心 instruction-tuning 会破坏 generation 能力：

| Benchmark | Metric | Vision Banana | Nano Banana Pro |
|-----------|--------|---------------|-----------------|
| GenAI-Bench | Win rate | **53.5%** | 46.5% |
| ImgEdit | Win rate | 47.8% | 52.2% |

Win rate 接近 50% 意味着 instruction-tuning 几乎没破坏生成能力，甚至 GenAI-Bench 上还略胜。这强力佐证了 "vision task alignment" 是 unlock 已有能力而非 overwrite。

Fig. 9 / Fig. 10 的定性比较也验证：生成的"鬼船月光海面"、"武士樱花庭院"等图像和 NBP 基本无法肉眼分辨。

---

## 4. 我的几个 Intuition 和跨 paper 联想

### 4.1 Generative vs Discriminative 在 ambiguous task 上的本质差异

这是 paper 在 Discussion 部分最 elegant 的论点。Monocular depth 是 ill-posed——一个 2D 投影对应多个合理的 3D 场景。传统 discriminative model 训 MSE / L1 loss 会学到 **mean**，导致 blurry / over-smoothed depth（DPT、Depth Anything v1 都有这问题）。它们靠引入 scale-shift invariance、canonically normalized depth、camera intrinsics 注入等技巧绕开。

Generative model 天然学到 **mode-seeking**，直接从 $p(\text{depth} \mid \text{image})$ 里采样一个 mode 而不是 average。这就根本绕开了 multi-modal 输出分布带来的退化问题。这其实就是 Marigold 当初之所以用 diffusion 的核心理由，但 Marigold 的 base（SD 1.5 / SD 2.1）太弱，Vision Banana 用 NBP 这种 frontier generator 才把 prior 真正发挥出来。

参考 Marigold：https://arxiv.org/abs/2409.18124
参考 Lotus / Lotus-2：https://arxiv.org/abs/2512.01030

### 4.2 这和 Wiedemer et al. 2025 / Zuo et al. 2025 的差异

近期已经有人发现 video generator / Nano Banana Pro 能 zero-shot 输出 depth / segmentation 的"visualization"（https://arxiv.org/abs/2509.20328；Zuo et al. 2025 评测 NBP on 14 tasks 40 datasets）。但他们的结果是 qualitative 的——生成的 visualization 看起来对，但 decode 回数值指标差。Vision Banana 的 contribution 就是 instruction-tuning 把这种 emergent 能力"对齐到可评测的格式"，于是 paper 名字里"generalist vision learner"才成立。

### 4.3 和 LLM 的并行

| 维度 | LLM | Vision |
|------|-----|--------|
| Pretraining objective | Next-token prediction | Image generation（diffusion / AR token） |
| Emergent capability | Understanding / reasoning | Visual understanding / 3D inference |
| Universal interface | Text generation | RGB image generation |
| Instruction-tuning | Format following | RGB visualization format following |
| Cross-task transfer | CoT / multi-task | Vision Banana 自发跨 task（Fig. 2b、Fig. 3b） |

Paper 在 Discussion 里 explicit 类比"AGI-V"，我觉得这个 framing 对学术界比"practical engineering"更有冲击力——它把 generative vision 推到了 foundational model 的位置，而不是 utility tool。

### 4.4 和 SAM 系列、DINOv3、CLIP 系列的范式冲突

这是最值得讨论的点。整个 vision representation learning 主流路径是：
- Supervised (ResNet/ViT) → Contrastive (SimCLR/MoCo/CLIP) → Bootstrap (DINO/DINOv3/SigLIP) → Masked AE (MAE/BEiT)

这条线全是 discriminative。Vision Banana 实质宣告：**这条线在 scaling 角度可能已经被 generation pretraining 超过**。SAM 3 用大量 mask 数据 + heavy architecture；DINOv3 用 1B+ 参数 + 蒸馏。这些都拿到 0.65 mIoU Cityscapes zero-shot；Vision Banana 拿到 0.699，且不需要 mask 训练。

我自己的直觉：你之前在 Yann LeCun 那里聊 JEPA 时也讨论过类似问题。Generative modeling 之所以被认为是"错的"是因为 pixel-level reconstruction 浪费 capacity 在高频细节上。但 diffusion 把这个问题 solve 了——它只学信号的"流形"，不学 noise，所以 capacity 不浪费。NBP 的 prior 强，正是因为它在 latent / flow matching space 上学到了视觉世界的真正流形。

### 4.5 与 OpenAI 的路线对照

OpenAI 的 GPT-Image-1.5 / Sora 类模型显然也在内部走这条路。Wiedemer et al. 2025 实际上就是 OpenAI / Google DeepMind 内部观察的公开版本。Vision Banana 是第一个把"emergent understanding"对齐到 benchmark 并达到 SOTA 的公开工作。这等于在产业界立了一个标杆——之后所有 foundation vision model 都会被要求同时 eval 生成和理解。

参考 OpenAI 的 image gen：https://openai.com/index/new-chatgpt-images-is-here/

### 4.6 RGB encoding 是否是最优 universal interface？

我有些保留。Paper 里 depth 用 Hilbert curve + power transform 是 engineering-heavy。如果未来 task 是 6DoF pose、optical flow、3D keypoints——RGB 3 个 channel 信息容量有限。一个可能 extension：让 model 输出 multi-channel latent（如 16 channels latent），decoder 把它翻译成具体 task 的输出格式。这和 LVM / VGV 把 visual token 作为统一接口的思路更近。但 RGB 的优势是 zero-modification、和 base model 完全 compatible——这点在工程上价值极大。

参考 LVM (Bai et al. 2024)：https://arxiv.org/abs/2312.09142
参考 VGGT：https://arxiv.org/abs/2503.11651

### 4.7 Cost 问题是真正的 bottleneck

Paper Discussion 最后诚实承认：用 NBP 这种 frontier generator 跑 depth / segmentation 比 SAM3 / Depth Anything v3 慢几个数量级。如果未来 robotics 真用 Vision Banana 跑实时 perception，需要 distillation 或 speculative decoding。这点和 LLM 当年 instruction-tuning 后还需要 vLLM / quantization / speculative decoding 一致。我相信会出现 "Vision Banana Distill" 类工作，把 generalist 蒸馏成 task specialist。

### 4.8 Video 方向的联想

Paper Future Work 提到 video generator 应该有更丰富 temporally-aware prior。这点直觉上很对——video pretraining 学了 physics / motion / 3D 一致性，应该是比 image 更强的 generalist learner。Seedance 2.0、Veo 3、Sora 这类模型一旦 instruction-tune 到 dense prediction 任务，可能在 video segmentation、scene flow、monocular video depth 上炸穿现有 benchmark。

参考 Veo 3：https://blog.google/innovation-and-ai/products/generative-media-models-io-2025/
参考 Seedance 2.0：https://seed.bytedance.com/en/seedance2_0/

### 4.9 和 LLaVA / Gemini-Vision 的关系

Vision Banana 没把 LLM 接进去——只在 ReasonSeg 上借了 Gemini 2.5 Pro 做 prompt rewriting。但显然 next step 是把 NBP 之类的 generator 和 LLM backbone 接起来做 omni model。这和你看到 Gemini 2.5 / GPT-4o 走的"unified multimodal generation"完全合流。一个会同时输出 text / image / audio / video 的 omni model，背后很可能是同一个 transformer 在不同 token 上做 next-token / flow matching。

### 4.10 关于 zero-shot transfer 这个 label 的微妙

Table 2 里有些方法叫 "Zero-Shot Transfer"（如 SAM3、Vision Banana），有些叫 "Non Zero-Shot Transfer"（如 SegMan、HyperSeg、X-SAM）。但要注意 SAM3 训练用了大量 mask 数据，Vision Banana 也用了 in-house 标注的 segmentation data。这里的 "zero-shot" 沿用 SAM / CLIP 的定义——指不在 evaluation benchmark 的 train split 上训练。读者要小心，不要混淆成"完全没看过此类 task"。

---

## 5. Limitation 与未来方向

Paper 自己列了几个：
1. **Task diversity**：目前只 5 个 task，scaling 到 optical flow / pose / keypoints / panoptic 等会测试 generalist 的真正边界
2. **Multi-view / video input**：现在只 monocular image
3. **Video generator**：是否 prior 更强？
4. **和 LLM 融合**：cross-modal reasoning
5. **Cost**：deployment 障碍

我自己再加几个：
6. **Failure mode 分析**：paper 没做 error analysis。Vision Banana 在什么情况下 fail？是不是 base generator 的 hallucination 直接传染到 perception？比如生成不存在的物体 mask？
7. **Calibration**：生成的 depth 数值是否 calibrated？是否可以 uncertainty estimation？这对 robotics / safety-critical 场景重要。
8. **OOD robustness**：synthetic training data + real eval 数据没问题（NYU、KITTI 等都打过 SOTA），但 extreme OOD（medical、microscopy、satellite）会怎样？
9. **Compositionality**：referring segmentation 上 Vision Banana 强，但是否能处理"the cup that is on top of the box behind the lamp"这类复杂 referring？这测的是 reasoning 而非 grounding。

---

## 6. 总结

Vision Banana 是一个 paradigm-confirming paper——它没有发明新 architecture、新 loss、新 dataset，而是把一个大家半信半疑的猜想（"generator 是不是 secretly 也是 learner？"）通过 careful engineering 对齐到 benchmark，给出确定答案：**Yes，且已经超过 specialist**。

它对领域的影响我觉得会有三层：
1. **短期**：所有 image / video generator 的 paper 都会被要求同时跑 understanding benchmark
2. **中期**：discriminative pretraining 路线（CLIP / DINO / SAM）会被 push 到更 narrow 的 niche
3. **长期**：vision foundation model 的标准形态会变成"large generator + instruction-tuning"，和 LLM 现状对齐

从你（Karpathy）一直强调的"software 2.0/3.0"和"通过生成学习世界模型"角度，这篇 paper 是又一个强力证据。Foundation model 时代，discriminative 的"单 task 单 head"思路正在被 generative 的"统一 interface"思路替代。这正是你在多个 talk 里预测的方向。

参考链接汇总：
- 项目主页：https://vision-banana.github.io
- NBP：https://blog.google/innovation-and-ai/products/nano-banana-pro/
- Power transform (Barron)：https://arxiv.org/abs/2502.10647
- SAM 3：https://arxiv.org/abs/2511.16719
- Depth Anything 3：https://arxiv.org/abs/2511.10647
- Depth Pro：https://arxiv.org/abs/2410.02073
- UniK3D：https://arxiv.org/abs/2503.23784
- Lotus-2：https://arxiv.org/abs/2512.01030
- Video models zero-shot：https://arxiv.org/abs/2509.20328
- LVM：https://arxiv.org/abs/2312.09142
- VGGT：https://arxiv.org/abs/2503.11651
- Lotus-2 demo：https://huggingface.co/spaces/haodongli/Lotus-2_Normal
- GenAI-Bench：https://arxiv.org/abs/2406.13743
- ImgEdit：https://arxiv.org/abs/2505.20275
