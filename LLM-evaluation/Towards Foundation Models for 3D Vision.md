---
source_pdf: Towards Foundation Models for 3D Vision.pdf
paper_sha256: 016a55990f6d12df9cd446ce6e1a4f3c7c8a1621e8579041c4b1685df75ebea7
processed_at: '2026-08-12T17:11:11-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话说清楚他们在干嘛

这群 Princeton 的人想知道一件事：**现在那些吹得天花乱坠的 AI 模型，到底能不能真的"看懂" 3D 世界？**

注意"看懂"这个词。你给 GPT-4o 一张照片，问它"前面那辆车离我多远"，它能给你一个答案。但这个答案到底来自它真的理解了 3D geometry，还是来自它背诵了训练数据里"天空在上面、马路在下面、车在中间"这种统计套路？

这就是整篇 paper 的核心问题。

---

## 为什么这个问题值得问

先说一下 background。过去两年 2D vision 的 foundation model 突飞猛进，GPT-4o、Gemini、Claude 这些 VLM (Vision-Language Model) 在 image captioning、VQA、image understanding 上已经接近甚至超过 human。各种 benchmark 满天飞，leaderboard 上数字越来越漂亮。

但 3D vision 这边基本是一片荒芜。你想要一个能同时做 depth estimation、camera pose、keypoint matching 的 unified model？没有。每个任务都有 specialized model，各自为战，metric 都不统一。MiDaS 做 depth，LightGlue 做 matching，MDETR 做 VQA，互相没法比。

更糟糕的是，**你连 human 有多强都不知道**。因为 3D 任务的输出空间太怪了——depth 是 dense pixel map，pose 是 SE(3) 连续空间，你怎么让一个 MTurk worker 输出一整张 depth map？所以过去的 benchmark 都没认真测过 human baseline。

这就导致一个尴尬局面：大家都在 report model accuracy，但没人知道这个数字到底意味着什么。90% accuracy 看起来很牛，可如果 human 在同一个任务上是 99%，或者如果随便 flip 一下 image 就掉到 50%，那这 90% 就是个虚假的繁荣。

Princeton 这帮人决定把这个窟窿补上。

---

## 他们怎么做的：UniQA-3D

思路其实很朴素：**把所有 3D 任务都改写成 multiple-choice question**，这样 VLM 能答，specialized model 能答，human 也能答，三方在同一个 output space 里公平 PK。

具体四个任务：

### 1. Relative Depth（相对深度）

不让你输出 dense depth map，就给你一张图，图上点两个 marker，问你"哪个离相机更近？"。二选一。

数据来自 KITTI（自动驾驶数据集），有 Lidar 测的真值，所以 ground truth 是可靠的。他们采了 750 对点。

这个任务设计的妙处在于：它抓住了 depth estimation 的"essence"。dense depth map 当然信息更丰富，但如果你连"两点哪个更近"都答不对，那 dense map 再漂亮也是表面功夫。这是把一个 continuous regression 任务 discretize 成 binary classification，损失了信息量，但换来了可比较性。

### 2. Spatial Reasoning（空间推理）

用 CLEVR 数据集（合成场景，有各种几何体），但只挑那些包含 "left/right/front/behind/top/bottom" 这些空间词的问题。比如：

> "在灰色橡胶球左边的大的青色圆柱体有几个？"

500 个 image-question pair。这个任务测的是 model 能不能在 scene 里做 spatial reasoning，而不是单纯的 object detection。

### 3. Relative Camera Pose（相对相机位姿）

给你两张图，告诉你"相机主要往哪个轴移动了"，让你选是正向还是负向。二选一。

数据来自 DTU（机械臂采集，pose 精度极高）。他们设了个 threshold，保证两张图之间有一个明确的 dominant movement axis，不会出现"往左上的斜方向"这种模棱两可的情况。

750 对，其中 250 对是 upside-down 翻转的。

### 4. Keypoint Matching（关键点匹配）

给你两张同一场景的不同视角，让你在左图标 5 个点，在右图找到对应的 5 个点。

数据来自 MegaDepth-1500（历史建筑，宽 baseline）。450 对，每对 5 个点，总共 2250 个 keypoint pair。

这个任务最能体现 UniQA-3D 的设计哲学：它把一个本来需要 dense correspondence 的任务，简化成"人也能做的 sparse matching"，同时保留了 3D vision 的核心挑战——wide baseline 下的对应关系。

---

## 最关键的 trick：把图倒过来

这是整个 paper 最聪明的一招。

他们把一部分图片**上下翻转**（upside-down），然后再问同样的问题。

为什么这招狠？因为翻转不改变任何 semantic content。图里还是那辆车、那栋楼、那棵树，pixel 的相对关系完全没变。唯一变的是 **gravity prior 失效了**——天空不再在上面，地面不再在下面。

如果你真的理解 3D scene structure，翻转应该对你毫无影响。但如果你学到的是"上面那部分像素通常更远"这种 shortcut，翻转之后你就懵了。

Human 几乎不受影响。Model 集体崩盘。这个对比极其 stark。

---

## 实验结果：三个故事

### Story 1: VLM 在 3D 上基本是"瞎猜"

GPT-4o 在 relative depth 上 67.9%，听起来比 random（50%）好不少。但你看看 upside-down 的情况：53.5%，几乎就是 random。Gemini-1.5 在 upside-down 上直接掉到 50% 以下。

更尴尬的是他们 manually 检查了 VLM 的 output（Fig. 1c），发现三类 failure mode：

1. **Marker 定位失败**：图上明明标记在马路上，VLM 说"标记在车上"，然后开始推理车的距离
2. **把 marker 当成真实物体**：VLM 以为那两个 marker 是 scene 里的真实 object，然后用 relative size 推理"哪个更大就更近"
3. **自相矛盾**：前一句话说 A 近，后一句话说 B 近，结论瞎写

这说明 VLM 的"3D understanding"很大程度上是 **language reasoning 拼接 2D pattern matching** 的产物，根本没有真正的 geometric perception。它看到一张图，识别出"这是马路、那是车、车在中间"，然后调用"车离相机近"这种 world knowledge，跟真正从 image geometry 推 depth 是两码事。

在 camera pose 任务上更惨：所有 VLM 在 regular image 上就接近 random（50% 左右），upside-down 后直接掉到 30-40%，**比瞎猜还差**。这说明 VLM 对 camera viewpoint 的理解几乎是零。

唯一例外是 spatial reasoning 任务，Gemini-1.5 拿了 83.6%，比 specialized MDETR（74.4%）和 human（61.6%）都高。但这恰恰因为 spatial reasoning 本质是 reasoning 任务，LLM 的逻辑链能力占优。这里 human 61.6% 偏低是因为 MTurk worker 注意力不集中，长 prompt 下容易出错（Fig. 3d 显示 question 越长 human 越差，model 反而越稳）。

### Story 2: Specialized model 准但不稳

MiDaS 在 regular KITTI 上 90.4%，比 human（82.1%）还高。看起来 specialized model 已经"超越人类"了。

但 upside-down 一翻：73.9%，掉了 16 个百分点。Human 呢？82.1% → 84.1%，几乎没变甚至略升。

这个对比太说明问题了。MiDaS 的高 accuracy 来自它在大规模数据上学到了"sky is far, ground is near, objects in middle"这种 gravity-aligned prior。一旦 prior 失效，它的真实 geometric reasoning 能力就暴露了——远不如 human。

Camera pose 任务上更明显：Swin Transformer 在 regular 上 68%，upside-down 后 35.4%，**所有 specialized model 在翻转后都不如 random guess**。这说明它们学到的完全是"image-plane 的视觉 pattern"——比如"如果第二张图比第一张图偏右，那 camera 往左移了"，这种 pattern 在 gravity-aligned 训练数据上有效，翻转后完全反向。

Specialized model 的问题不是不准，是**准得虚假**。它们在 IID (in-distribution) 测试集上表现漂亮，但一旦 OOD (out-of-distribution)，哪怕只是个 trivial 的 vertical flip，就原形毕露。这是对整个 3D vision 社区的一个 warning：你可能一直在 benchmark 上 overfit 而不自知。

### Story 3: Transformer 比 CNN 更像人脑

这是最有意思的发现，也是最有 cognitive science 价值的。

在 relative depth 任务上，MiDaS 有两个版本：MiDaS-CNN（ResNet backbone）和 MiDaS-DPT（Vision Transformer backbone）。两者 accuracy 接近，但 error pattern 差别很大。

用 Cohen's κ 量化 model 和 human 的 error pattern 一致性：

- MiDaS-DPT (ViT): κ = 0.66
- MiDaS-CNN (ResNet): κ = 0.56

ViT 版本明显更像 human。

这个结论在多个维度上都成立：
- 在 sympair vs randpair 的 accuracy gap 上，ViT 更接近 human 的均匀表现
- 在不同 depth difference bin 上，ViT 的曲线形状更像 human
- 在不同 semantic class 上，ViT 的 per-class accuracy pattern 更接近 human

Keypoint matching 上也是：LightGlue（Transformer-based）比 ORB（classical）在 EPE 上和 human 更一致（19.0px vs 44.3px 差距）。

为什么这件事重要？因为 human brain 处理 2D 和 3D 走的是两条不同的神经通路：

- **Ventral stream**（腹侧通路）：负责 object recognition，从 V1 经过 V2、V4 到 IT cortex，处理"这是什么"
- **Dorsal stream**（背侧通路）：负责 spatial perception、depth、motion，从 V1 经过 MT、MST 到 posterior parietal cortex，处理"在哪里、怎么动"

之前的 cognitive science 工作（Tuli et al. 2021）发现 Transformer 在 2D object recognition 上比 CNN 更像 ventral stream。但 ventral stream 和 dorsal stream 的神经回路结构完全不同，所以这个结论能不能外推到 3D vision 是个 open question。

这篇 paper 给出了肯定答案：**在 3D vision（dorsal stream 功能）上，Transformer 依然比 CNN 更像 human**。这暗示 attention mechanism 可能捕捉到了某种更通用的视觉计算原则，不局限于腹侧通路。

但有个 caveat：在 camera pose 任务上，**所有 model 都不像 human**（κ < 0.2），包括 Transformer。说明 camera pose 这种 egocentric spatial reasoning 可能涉及更高级的认知过程，现有 architecture 都没 capture 到。这个 negative result 也很重要。

---

## 为什么人这么稳

Human 在 upside-down image 上几乎不受影响。为什么？

因为 human 的 dorsal stream 用的 depth cue 都是 **geometric invariant** 的，和 image orientation 无关：

- **Binocular disparity**（双眼视差）：两只眼睛看到的 slight offset，和 image 怎么转无关
- **Motion parallax**（运动视差）：头动的时候近物移动快、远物移动慢，也和 orientation 无关
- **Perspective convergence**（透视收敛）：平行线在远处汇聚，翻转后仍然汇聚
- **Shading & shadow**：光照分布暗示 3D 形状，翻转后 shading pattern 仍然 consistent（虽然 light-from-above prior 会失效，但 human 能快速 adapt）
- **Occlusion**（遮挡关系）：A 挡住 B 说明 A 在前面，这个关系翻转后不变

这些 cue 都是 geometric 性质的，不依赖"上=远，下=近"这种 world knowledge prior。

而 model 学到的是 image-plane 的 statistical correlation。在 gravity-aligned 训练数据上，"image 上方的 pixel 通常更远"这个 shortcut 高度有效，model 就 greedy 地学了这个。它没学到真正的 3D geometry。

这是为什么 model 在 IID 上能超 human，在 OOD 上被 human 秒杀。

Reference: human 3D vision 的 neuroscience 综述 [Welchman 2016](https://www.annualreviews.org/doi/10.1146/annurev-vision-111815-114935)

---

## Cohen's κ 是什么，为什么重要

这是 paper 的方法论核心，值得单独讲讲。

简单说，Cohen's κ 衡量两个 rater（比如 model 和 human）的 answer 一致性，**同时控制了随机一致的情况**。

举个例子：如果 model 和 human 都是 50% accuracy 的二选一任务，两人纯随机猜，他们 expected 一致率是 50%。如果实际一致率也是 50%，那 κ = 0，说明一致只是巧合。

公式：

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

- $p_o$：observed agreement，实际一致比例
- $p_e$：expected agreement by chance，随机情况下的一致期望
- $\kappa = 1$：完全一致
- $\kappa = 0$：一致性等于随机
- $\kappa < 0$：比随机还差

为什么不用 accuracy 直接比？因为 **accuracy 高不代表 error pattern 像 human**。

假设一个 model 100% accuracy，那它和 human 的 agreement 就是 human 的 accuracy，看起来很 aligned。但其实这 model 可能用完全不同的方式答对的题。Cohen's κ 通过混淆矩阵计算，会考虑 model 和 human 在**具体哪些样本上错、哪些样本上对**的一致性，而不是只看总体 accuracy。

在 paper 里，GPT-4o 在 depth 上 67.9% accuracy，但 κ 只有 0.08 左右。这说明 GPT-4o 答对的题和 human 答对的题几乎不重叠——它用的是完全不同的 reasoning path。这个 insight 单看 accuracy 是得不到的。

具体计算上，给定混淆矩阵 $M$（$M_{ij}$ = model 说 i、human 说 j 的样本数）：

$$p_o = \frac{\sum_i M_{ii}}{N}$$

$$p_e = \sum_i \left(\frac{\sum_j M_{ij}}{N}\right) \left(\frac{\sum_j M_{ji}}{N}\right)$$

$p_e$ 是 model 的 marginal distribution 和 human 的 marginal distribution 的内积，代表两人各自按自己 marginal 随机分类时的 expected 一致率。

Reference: [McHugh 2012](https://hrcak.srce.hr/file/162014)

---

## Human annotation 的工程细节别错过

Appendix A 里有个细节特别值得提。

他们在 MTurk 上发现，如果用 multiple-choice question（让 worker 选 A/B/C/D），收集到的数据只有 35% 是 valid human response，其余都是 bot 填的垃圾，accuracy 算下来只有 77%。

但他们改成让 worker **直接在 image 上 click marker 位置**，valid rate 跳到 73%，accuracy 跳到 91%，接近他们自己手动标注的 92% upper bound。

这说明 2024 年的 MTurk 已经被 LLM-based bot 大规模污染了。传统的 multiple-choice MTurk benchmark 数据可信度存疑。未来做 human evaluation 的 benchmark，click-based 或 interactive interface 是必须的，纯文字选项的 MCQ 已经不够用。

这点对未来所有 human evaluation 工作都有启示，不只是 3D vision。

---

## 把这篇 paper 放到更大的图里看

### 和 2D foundation model 的对比

2D vision 在 2012 年 AlexNet 之后经历了一个十几年渐进式突破的过程：ImageNet pretraining → transfer learning → self-supervised (SimCLR, MoCo, DINO) → CLIP → VLM (LLaVA, GPT-4o)。每个阶段都有明确的 benchmark（ImageNet、COCO、VQA）和明确的 baseline（human accuracy）。

3D vision 现在大概在哪个阶段？我觉得大概在 2D vision 的 2015 年左右——有 specialized model（MiDaS、LightGlue 相当于当时的 ResNet、Faster R-CNN），有大规模数据（ScanNet、KITTI、MegaDepth 相当于 COCO、ImageNet），但缺一个 unified foundation model，也缺一个能跨任务比较的 benchmark。

UniQA-3D 大概相当于 2D vision 的 ImageNet 早期版本——它不是 model，是 evaluation infrastructure。但它比 ImageNet 多了一个 dimension：**robustness under geometric perturbation**。这是 2D benchmark 从来没认真测过的 axis。

### 对 future 3D foundation model 的启示

如果让我赌一把，下一个 3D foundation model 的突破会来自这几个方向的组合：

1. **大规模 multi-view pretraining**：像 DINO v2 那样自监督，但用 multi-view consistency 而不是 single-view augmentation。参考 [DINO v2](https://arxiv.org/abs/2304.07193) 和 [CroCo](https://arxiv.org/abs/2210.10716)

2. **Geometric augmentation**：训练时随机旋转 image，强迫 model 学习 orientation-invariant 的 3D feature。这听起来 trivial，但目前主流 depth model 都没这么做。

3. **3D-aware tokenization**：把 image token 升级成 multi-view token 或 3D point token，让 Transformer 直接在 3D representation 上做 attention。参考 [3D-LLM](https://arxiv.org/abs/2307.12981)。

4. **Brain-aligned loss**：用 Cohen's κ 当 auxiliary loss，鼓励 model 的 error pattern align human。这需要 paired human annotation，成本高，但可能是学 human-like perception 的唯一途径。参考 [Brain-Score](https://www.biorxiv.org/content/10.1101/407007v1) 的思路。

5. **Diffusion-based 3D generation**：像 Marigold 那样用 diffusion model 做 depth estimation。Diffusion 的 iterative refinement 可能比 discriminative model 更适合 dense 3D prediction。参考 [Marigold](https://arxiv.org/abs/2312.02145)。

### 和 reinforcement learning 的联想

Paper 里没有提 RL，但我有个联想：human 在 upside-down image 上 depth 不变，这个 robustness 是怎么学来的？

Developmental psychology 告诉我们，婴儿在 6 个月大时就开始有 depth perception，但那时还不 robust。随着视觉经验积累，加上 head movement、locomotion（爬、走）、interaction with objects，dorsal stream 才逐渐学会 invariant 3D feature。这本质上是个 **embodied learning process**。

那么能不能用 RL 训练一个 agent 在 3D 环境（比如 Habitat、AI2-THOR）里 interaction，通过 multi-view observation 和 reward signal 学习 invariant 3D representation？这可能比 supervised learning on static images 更接近 human learning 机制。

这只是一个 hypothesis，但 UniQA-3D 的 robustness finding 暗示单纯 supervised learning on gravity-aligned dataset 永远学不到 robust 3D perception，必须引入某种 active 或 embodied 学习机制。这是 paper 没明说但隐含的 implication。

Reference: [Habitat](https://arxiv.org/abs/1904.01201), [AI2-THOR](https://arxiv.org/abs/1704.01265)

---

## 最后的 takeaway

这篇 paper 没有提出新 model，没有刷 SOTA，没有炫酷的 architecture diagram。它做的是一件 unglamorous 但 critical 的工作：**给 3D vision 社区建了一把尺子，然后用这把尺子量了一遍现状，发现大家都没穿衣服**。

三个核心 finding：

1. **VLM 的 3D 能力是表面功夫**：靠 language reasoning + 2D pattern matching 拼凑，upside-down 一翻就露馅
2. **Specialized model 准但不稳**：在 gravity-aligned IID 上能超 human，但在 trivial geometric perturbation 下崩盘，说明学的是 shortcut 而非真 3D understanding
3. **Transformer 在 3D 上也比 CNN 更像 human**：dorsal stream 似乎也偏好 attention-based computation，但 camera pose 这种 egocentric reasoning 上所有 model 都不像 human

给未来 3D foundation model 开发的 actionable advice：
- 必须在训练时做 geometric augmentation，不能只靠 photometric jitter
- 必须 report robustness under perturbation，不能只 report IID accuracy
- 应该把 Cohen's κ with human 纳入标准 eval suite
- 考虑引入 multi-view consistency 或 embodied learning 作为 pretext task

Reference: [Paper GitHub](https://github.com/princeton-vl/UniQA-3D)

这篇 paper 让我想起当年 ImageNet 出现之前，CV 社区也是在 Caltech-101、Pascal VOC 这种小数据集上各自为战。真正改变游戏规则的是统一 benchmark + 大规模数据 + 统一 architecture。3D vision 现在正缺这个 catalyst，UniQA-3D 也许是其中一个 piece。下一个 piece 可能是大规模 3D pretraining data，再下一个可能是某种 3D-aware Transformer architecture。等这些 piece 凑齐，3D vision 的"AlexNet moment"可能就来了。

---

# 深度讲解: Towards Foundation Models for 3D Vision: How Close Are We?

这篇paper来自Princeton VL (Jia Deng组+Tom Griffiths组)，第一作者Yiming Zuo和Karhan Kayan。核心问题是：**3D vision foundation models 离我们还有多远？** 通过构建 UniQA-3D benchmark，作者系统地比较了 VLMs、specialized models 和 humans 在四个基础 3D vision 任务上的表现，并深入分析 error pattern 与 human 的 alignment。

Reference link: https://github.com/princeton-vl/UniQA-3D

---

## 1. 核心动机与Marr视角的呼应

作者站在 David Marr 的 [Vision](https://mitpress.mit.edu/9780262514620/vision/) 计算理论框架上思考问题。Marr 把 vision 分为三个层面：**computational theory** (问题是什么)、**representation & algorithm** (如何表示与计算)、**hardware implementation** (神经实现)。过往的 cognitive science 工作大多聚焦在 implementation level (比如 grid cells、place cells、stereogram+fMRI 脑区激活)，对构建 3D foundation models 的直接启发有限。这篇 paper 想反过来从 computational + algorithm level 入手，定量比较现有 model 和 human 的 behavior。

这里有一个非常关键的生物学事实：human brain 处理 2D 和 3D 走的是**两条不同的通路**：
- **Ventral stream** ("what" pathway, V1 → V2 → V4 → IT)：负责 object recognition、2D 形状识别
- **Dorsal stream** ("where/how" pathway, V1 → V2 → MT → MST → PPC)：负责空间位置、深度、运动

Reference: [DiCarlo et al., Neuron 2012](https://www.cell.com/neuron/fulltext/S0896-6273(11)01009-X); [Welchman, Annual Review of Vision Science 2016](https://www.annualreviews.org/doi/10.1146/annurev-vision-111815-114935)

这意味着 Tuli et al. 在 2D object recognition 上发现的 "Transformer 比 CNN 更像 human" 的结论，**不能直接外推到 3D vision**——dorsal stream 与 ventral stream 的电路结构、神经元的 tuning properties 都不一样（比如 MST 神经元对 optic flow 有 selectivity，IT 神经元对 shape 有 selectivity）。所以这篇 paper 对 3D vision 重做 alignment analysis 是有信息量的。

---

## 2. UniQA-3D Benchmark 的设计哲学

### 2.1 为什么需要一个 unified output space

3D vision 任务最大的工程难题是 **output space 异构**：
- Depth estimation: pixel-wise dense map (H×W×1)
- Optical flow: pixel-wise 2D vector field (H×W×2)
- Camera pose: SE(3) 连续空间 (R∈SO(3), t∈ℝ³)
- Keypoint matching: sparse correspondences (N×4)

这导致没法用一个 metric 跨任务比较，更没法让 human 参与 dense prediction 评测（你没法让一个 MTurk worker 输出 100 万个像素的 depth）。作者的解决方案是把所有任务**统一为 multiple-choice VQA 格式**，即 binary 或 4-way classification。这是整个 benchmark 的灵魂。

### 2.2 四个子任务的数据来源

| Task | Data Source | #Images | GT 来源 | Specialized Models |
|---|---|---|---|---|
| Relative Depth | KITTI | 750 | Lidar | MiDaS (CNN/DPT) |
| Spatial Reasoning | CLEVR (子集) | 500 | Program-generated | MDETR |
| Relative Camera Pose | DTU | 750 | Robot arm 6DoF | Custom (ResNet/ViT/Swin) |
| Keypoint Matching | MegaDepth-1500 | 450 | SfM depth+pose | SIFT, FAST, SuperPoint, ORB, LightGlue |

这个设计的精妙之处：
- **KITTI**：自动驾驶场景，真实 Lidar GT，避免 DIW 那种用 human annotation 当 GT 的噪声问题
- **CLEVR**：合成数据，但作者只保留包含 spatial keywords (left/right/front/behind/top/bottom) 的 question，使得子集远比原 CLEVR 难（MDETR 在原 CLEVR 99.7%，子集只有 74.4%）
- **DTU**：机械臂采集，pose GT 精度极高，作者还设了一个 ratio threshold 保证 movement 有 dominant axis
- **MegaDepth-1500**：宽 baseline 历史建筑场景，SfM 提供 dense correspondence GT

### 2.3 Geometric Perturbation 的设计

作者引入 **upside-down flip** 作为 robustness 测试。这看起来简单，背后有深刻的考虑：

1. **Distribution shift 的最小形式**：vertical flip 不改变 image 的 semantic content，但破坏了 gravity prior
2. **训练数据偏差**：所有大规模 3D dataset (KITTI, MegaDepth, BlendedMVS, ScanNet) 都是 gravity-aligned 拍摄的，model 学到的可能是"上=远/下=近"这种 statistical shortcut
3. **Robotic application 现实需求**：robot 装在 drone、inverted camera、handheld 上时，gravity prior 经常失效

这种 perturbation 完全不同于常见的 image corruption (blur, noise, jpeg)，它是 **geometric，不是 photometric**。这也是 UniQA-3D 与 BLINK 等其他 benchmark 的最大区别（见 Table 1）。

---

## 3. 四个任务的数学定义与对应模型解析

### 3.1 Relative Depth Estimation

**Task**：给定 image $I$ 和两个 marker 像素 $p_1=(x_1,y_1)$, $p_2=(x_2,y_2)$，输出哪个像素离 camera 更近。

**MiDaS 的原理**：MiDaS 学的是 **affine-invariant depth**，即预测 $\hat{d} = a \cdot d_{\text{true}} + b$，其中 $a>0, b\in\mathbb{R}$ 是任意未知仿射参数。这是 monocular depth 的 fundamental ambiguity——单张图无法恢复 metric depth。MiDaS 用 scale-and-shift invariant loss：

$$\mathcal{L}_{\text{SSI}} = \sum_i \left| \frac{\hat{d}_i - \mu(\hat{d})}{\sigma(\hat{d})} - \frac{d_i - \mu(d)}{\sigma(d)} \right| $$

其中 $\mu(\cdot), \sigma(\cdot)$ 是对 dense depth map 计算的均值和标准差。MiDaS-DPT 用 Vision Transformer 作为 backbone，MiDaS-CNN 用 ResNet。

**Sampling 策略**：遵循 DIW [Chen et al., NeurIPS 2016](https://proceedings.neurips.cc/paper/2016/hash/d201f0db...html)，作者 50% 概率沿水平线对称采样两点（sympair），50% 概率随机采样（randpair）。Sympair 设计是为了避免 human 用 image size 作为 depth cue。

**Key result (Fig. 1b)**：

| Method | Regular | Upside-down |
|---|---|---|
| Random | 50% | 50% |
| GPT4-Turbo | ~54% | ~45% |
| GPT4-Omni | 67.9% | 53.5% |
| Gemini-1.5 | ~62% | ~50% |
| MiDaS-CNN | ~88% | ~71% |
| MiDaS-DPT | **90.4%** | 73.9% |
| Human | 82.1% | **84.1%** |

观察：**MiDaS 比 human 强（zero-shot 在 KITTI 上，但 MiDaS 没在 KITTI 上训练过）**，但 flip 后崩盘。Human 几乎不变。这是 paper 最 striking 的 robustness finding。

### 3.2 Spatial Reasoning

**Task**：CLEVR-style VQA，只保留 spatial reasoning question。例：
> "What number of things are large cyan shiny cylinders to the left of the small ball or large cyan objects that are to the left of the gray rubber ball?"

**MDETR 架构** [Kamath et al., ICCV 2021](https://arxiv.org/abs/2104.12763)：在 DETR 基础上扩展为 multi-modal，把 text token 和 image patch token 喂入同一个 Transformer，做 modulated detection。MDETR 在 CLEVR 上达到 SOTA。

**Key result (Fig. 3b)**：

| Method | Accuracy |
|---|---|
| GPT4-Turbo | ~48% |
| GPT4-Omni | 52.4% |
| Gemini-1.5 | **83.6%** |
| MDETR (specialized) | 74.4% |
| Human | 61.6% |

这里有个**反直觉现象**：Gemini-1.5 比 human 高得多，也比 specialized MDETR 高。作者解释是 spatial reasoning 本质是 reasoning 任务，LLM 的 reasoning capability 占优势。Human 61.6% 偏低是因为 MTurk 的 attention 有限。

更反直觉的是 Fig. 3d：**question 越长，model 越准，human 越差**。这是因为 LLM 把长 prompt 当 in-context learning，反而更稳定；而 human 在长 prompt 上注意力衰减。

### 3.3 Relative Camera Pose Estimation

**Task**：给定 image pair $(I_1, I_2)$ 和 dominant axis，输出 camera 移动方向 (2-way classification)。

**Formulation**：作者把 6DoF relative pose $[\mathbf{R}, \mathbf{t}]$ 简化为 dominant axis 上的方向。给定相对平移 $\mathbf{T} = [T_x, T_y]$，先取绝对值 $\mathbf{A} = [|T_x|, |T_y|]$，找 $\text{index} = \arg\max(\mathbf{A})$，然后根据符号输出 ground truth：

$$
D = \begin{cases}
0 & \text{if index}=0 \text{ and } T_x > 0 \quad (+x) \\
1 & \text{if index}=0 \text{ and } T_x < 0 \quad (-x) \\
0 & \text{if index}=1 \text{ and } T_y > 0 \quad (+y) \\
1 & \text{if index}=1 \text{ and } T_y < 0 \quad (-y)
\end{cases}
$$

变量含义：
- $T_x, T_y$：相对平移向量的 x, y 分量
- $\text{index} \in \{0, 1\}$：dominant axis 索引
- $D \in \{0, 1\}$：方向类别标签

注意：作者**只考虑 x-y 平面**（horizontal & vertical movement），忽略了 z（前进后退），因为 z 方向对人类来说更难判断，会引入 noise。

**Architecture**：作者自训 ResNet-50、ViT、Swin Transformer 三个 backbone。每张图过 backbone 得到 feature vector，两张图的 feature concatenate，再 concat 一个 axis indicator (one-hot 2D)，过 MLP head 输出 2-way logits。Cross-entropy loss 训练。Training set 是 BlendedMVS，test set 是 DTU (zero-shot)。

**Key result (Fig. 4a)**：

| Method | Regular | Upside-down |
|---|---|---|
| Random | 50% | 50% |
| GPT4-Turbo | 51.8% | 35.4% |
| GPT4-Omni | 48.6% | 31.5% |
| Gemini-1.5 | 45.7% | 46.9% |
| ResNet | 65.9% | 37.7% |
| ViT | 61.1% | 42.3% |
| Swin | **68.0%** | 35.4% |
| Human | **75.7%** | **77.7%** |

注意：**所有模型在 upside-down 上都不如 random guess**！这说明模型学到的是 gravity-aligned shortcut。Human 几乎不变甚至略升。

### 3.4 Keypoint Matching (检测+匹配两阶段)

**Detection**：给定 image $I$，找到 $N$ 个 salient keypoint。
**Matching**：给定 image pair $(I_1, I_2)$，建立 correspondence $\{(p_i, p'_i)\}_{i=1}^N$。

**比较的方法**：
- Classical detection: Harris Corner [Harris & Stephens 1988], DoG (SIFT), FAST [Rosten & Drummond 2006]
- Neural detection: SuperPoint [DeTone et al., CVPR Workshop 2018]
- Classical matching: ORB + brute-force KNN [Rublee et al., ICCV 2011]
- Neural matching: LightGlue [Lindenberger et al., ICCV 2023]

**SuperPoint 原理**：base CNN + two head (detection head 输出 pixel-wise keypoint probability, description head 输出 256-d descriptor)。Homographic Adaptation 扩充训练数据，joint loss 训练。Reference: [SuperPoint paper](https://arxiv.org/abs/1712.07629)

**LightGlue 原理**：在 SuperGlue 之上做的轻量化改进，核心是 **adaptive depth**：根据 image pair 难度动态决定 Transformer layer 数量。每个 layer 内做 self-attention + cross-attention，输出 correspondence assignment。Reference: [LightGlue paper](https://arxiv.org/abs/2306.13643)

**Evaluation metric: End-Point Error (EPE)**：

$$\text{EPE}(p, p') = \| p' - p'_{\text{GT}} \|_2 $$

其中 $p'$ 是 predicted correspondence，$p'_{\text{GT}}$ 是 GT correspondence (通过 depth + pose reprojection 计算)。

**Key result (Fig. 5a)**：
- LightGlue EPE ≈ 4 pixels (best)
- Human EPE ≈ 7 pixels
- ORB EPE ≈ 50 pixels (worst)

Human 落在 LightGlue 和 ORB 之间，但**和 LightGlue 在一致性上接近**（Fig. 5b: diff=19.0px for LightGlue, 44.3px for ORB）。

---

## 4. Cohen's Kappa 的数学讲解

整个 paper 用 Cohen's κ [McHugh 2012](https://hrcak.srce.hr/file/162014) 衡量 model-human alignment。这是 cognitive science 的标准工具。

**定义**：给定两个 rater（这里 rater A = model, rater B = human），N 个样本，C 个类别。设：

- $p_o$ (observed agreement rate)：两人实际一致的比例
- $p_e$ (expected agreement by chance)：假设两人独立随机分类时的一致期望

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

具体地，给定混淆矩阵 $M_{ij}$（rater A 说 i，rater B 说 j 的样本数）：

$$p_o = \frac{1}{N} \sum_i M_{ii}$$

$$p_e = \sum_i \left( \frac{\sum_j M_{ij}}{N} \right) \left( \frac{\sum_j M_{ji}}{N} \right)$$

**直觉**：κ=1 表示完全一致，κ=0 表示一致性等于随机，κ<0 表示比随机还差。**关键性质：κ 控制了 accuracy 的影响**——一个 model 不会因为自己 accuracy 高就和 human 自动高 κ。这正是 paper 的核心方法学贡献。

例如在 relative depth 上 (Fig. 2c)：
- MiDaS-DPT κ = 0.66 (Transformer)
- MiDaS-CNN κ = 0.56 (ResNet)
- 所有 VLMs κ < 0.1

这说明：即使 GPT4-Omni accuracy 67.9%，它的 error pattern 和 human 几乎没有 alignment。

在 camera pose 上 (Fig. 4b)：所有 model κ < 0.2，upside-down 后全部 κ < 0。这告诉我们：**没有任何现有 model 在 camera pose 任务上是 human perception 的好模型**。这是对 cognitive science 一个 negative 但重要的 finding。

---

## 5. Human Annotation 的工程细节（Appendix A）

这是 paper 最 underrated 的部分。作者在 MTurk 上做了非常严格的质量控制：

**Click-based interface vs Multiple-choice**：
- Multiple-choice interface：35% valid data (大量 bot)，accuracy 77%
- Click-based interface（让 worker 在 image 上 click marker 位置）：73% valid data，accuracy 91%
- 自标注 human upper bound ≈ 92%

这告诉我们：MTurk 上 MCQ 被 LLM-bot 污染严重，click-based 设计是 future 3D benchmark 的必备防作弊机制。

**Consensus scoring**：每个 HIT 给 3 个 worker，只有 3 人答案完全一致才算 valid。这进一步过滤 noise。

Sample size：
- Depth: 162 subjects
- Camera pose: 109 subjects
- Spatial reasoning: 449 subjects
- Keypoint matching: 143 subjects

远超 BLINK 那种只用 2 个 coauthor 当 human baseline 的做法，使得统计显著性可信。

---

## 6. 关键 Takeaway 与 Open Problems

### 6.1 VLMs 在 3D 上没 emergent capability

即使 GPT-4o 这种 multimodal SOTA，在 relative depth 上只比 random 高 17 个百分点，在 upside-down image 上**直接不如 random**。这暗示 VLMs 的"3D understanding"大多是 **2D semantic shortcut**（比如识别 car vs road，然后说 road 更远）而非真正的 geometric reasoning。Fig. 1c 生动展示了三种 failure mode：
1. **Localization failure**：把 marker on road 误认成 on car
2. **Scene understanding failure**：把 marker 当成真实 object，用 relative size 推 depth
3. **Reasoning failure**：生成自相矛盾的 response

### 6.2 Specialized models 学到的是 shortcut，不是 invariant 3D

MiDaS 在 KITTI 上 90.4% > human 82.1%，看起来"超越"了 human。但 upside-down 后掉到 73.9%，而 human 不变。这说明 MiDaS 学到的不是真正的 depth-from-shading/geometry，而是大量 "sky above, ground below" 这种 gravity-aligned prior。一旦 prior 失效，model 暴露真实能力。

这是对 foundation model 的**核心警告**：高 accuracy 不代表真正的 3D understanding。

### 6.3 Transformer 比 CNN 更像 human 在 3D vision 上也成立

这是 paper 最有 cognitive science 意义的发现。在 relative depth 任务上，MiDaS-DPT (ViT) 在所有 alignment metric 上都比 MiDaS-CNN (ResNet) 更接近 human：
- Fig. 2a: sympair vs randpair gap 更小（更接近 human 的均匀 pattern）
- Fig. 2b: depth difference 曲线形状更接近 human
- Fig. 2c: Cohen's κ 更高 (0.66 vs 0.56)
- Fig. 2d: per-class accuracy distribution 更接近 human

在 keypoint matching 上，LightGlue (Transformer) 也比 ORB (classical) 更接近 human matching pattern。

**Intuition**: 为什么 Transformer 在 3D vision 上也更像 human？可能的解释：
1. **Global attention** 捕捉 long-range geometric relationship，类似 dorsal stream 的 MT/MST 的大 receptive field
2. **Patch-based processing** 与 human vision 的 foveation + saccade 有相似性
3. **Lack of translation invariance prior**（CNN 的归纳偏置），使得 Transformer 学到的 representation 更灵活

但要注意：**camera pose 任务上 Transformer 也不像 human**（κ < 0.2）。这是 paper 一个 honesty 体现，不是 overclaim。

### 6.4 Human Vision 的秘密：Dorsal Stream 的 Robustness

为什么 human 在 upside-down image 上 depth & pose 几乎不变？这暗示 dorsal stream 不是用 gravity-aligned prior，而是真正提取 3D scene structure：
- Binocular disparity (立体视差)
- Motion parallax (运动视差)
- Perspective cues (线性透视)
- Shading & texture gradient
- Occlusion relationship

这些 cue 都是 **geometric invariance** 的，与 image orientation 无关。而 model 学到的是 image-plane statistical pattern，没真正理解 3D geometry。

Reference: [Welchman 2016 review on 3D vision in brain](https://www.annualreviews.org/doi/10.1146/annurev-vision-111815-114935); [Ohzawa et al., Science 1990 on disparity-tuned neurons in V1](https://www.science.org/doi/10.1126/science.2395799)

### 6.5 Open Problems 这篇 paper 暗示的

1. **3D foundation model 的训练范式**：单纯 2D 图文对训练不够。需要 3D-aware pretext task (multi-view consistency, stereo, SfM supervision)。
2. **Geometric augmentation 是必须的**：现有数据增强 (flip, crop, color jitter) 不足以培养 geometric invariance，需要 random rotation、arbitrary gravity direction augmentation。
3. **Unified 3D output space**：像 UniQA-3D 这种 VQA 格式可以作为 unified interface，但 dense prediction 仍需解决（reference: [Depth Anything](https://arxiv.org/abs/2401.10891) 的工作方向）。
4. **Cognitive plausibility as a training signal**: 如果我们想让 model 学到 human-like 3D vision，可以把 Cohen's κ 当作 auxiliary loss，鼓励 model 在 error pattern 上 align human。这是一个非常有意思的 future direction，类似 [Brain-Score](https://www.biorxiv.org/content/10.1101/407007v1) 的思路。

### 6.6 与其他 3D-aware foundation model 工作的关系

- **3D-LLM** [Hong et al., NeurIPS 2023](https://arxiv.org/abs/2307.12981)：把 3D point cloud 喂给 LLM，做 high-level navigation/QA。这是 high-level 3D understanding，UniQA-3D 则是 low-level geometric perception。
- **3D-VisTA** [Zhu et al., ICCV 2023](https://arxiv.org/abs/2303.11384)：3D-text pretraining Transformer。
- **Probing 3D awareness of VFM** [Banani et al., CVPR 2024](https://arxiv.org/abs/2401.02202)：linear probe on DINO, CLIP features，发现这些 model 有 implicit 3D awareness。UniQA-3D 和它互补，前者评测 generative capability，后者 probe representation。
- **Depth Anything V2** [Yang et al., 2024](https://arxiv.org/abs/2406.09414)：用大规模 unlabeled data 蒸馏 monocular depth。如果能用 UniQA-3D 评测 robustness，将是很有意思的实验。

---

## 7. Limitations 与 Personal Reflections

### 7.1 作者自陈的 limitations
- 只覆盖 4 个 task，没有 surface normal、optical flow、3D reconstruction
- 只评 closed-source VLM，没评 LLaVA、CogVLM、MiniCPM 等 open-source

### 7.2 我自己看到的 limitations

1. **Upside-down 只是 geometric perturbation 的最 trivial 形式**：in-plane rotation 90°/180°，random rotation，horizontal flip 都应该 test。Upside-down 已经让所有 specialized model 崩盘，更复杂的 perturbation 可能完全失效。
2. **VLM 输入 interface 问题**：VLM 是基于 2D image patch token 的，把 marker 用文字描述（"红色三角形 marker"）给 VLM 可能 lossy。原文也没详细说怎么 mark image for VLM——是用红圆点 overlay 还是文字坐标？这影响很大。Fig. 1c 看起来是 image overlay，所以 VLM 还要先做 marker localization 这个 non-trivial task。
3. **Human attention baseline**：MTurk 上 61.6% 的 spatial reasoning accuracy 严重低估 human 上限。如果让 paper authors 自己测，可能 90%+。建议 future work 用 lab-based 或 paid expert 标注。
4. **No fine-tuning experiment**：paper 只做 zero-shot 评测。如果在 UniQA-3D 上 fine-tune VLMs，能达到什么水平？这个 ablation 缺失。
5. **Cohen's κ 的局限**：κ 对 class imbalance 敏感，多类时偏向 majority class。Camera pose 的 2-way classification 还可以，但 keypoint matching 这种 continuous 任务用 κ 不太自然。
6. **Dorsal stream 的 hypothesis 没直接 test**：作者用 Transformer vs CNN 的对比间接说明 dorsal stream 可能更像 attention 而非 convolution，但这只是 correlation，不是 causal evidence。如果想真的 link 到 neuroscience，需要 fMRI 数据对比。

### 7.3 Paper 的真正 contribution

在 3D vision foundation model 还没出现的今天，这篇 paper 的贡献**不在提出新 model，而在建立 evaluation framework + diagnostic findings**：
- UniQA-3D benchmark 可作为 future 3D VLM 的标准评测
- 发现了 "accuracy without robustness" 这个 systematic gap
- 把 cognitive science 的 alignment methodology 引入 3D vision
- 给 "Geometric perturbation robustness" 这个被忽视的 axis 正式量化

---

## 8. 拓展阅读与 Reference

**Benchmark & VLM evaluation**:
- [BLINK](https://arxiv.org/abs/2404.12390) - Multi-modal LLM can see but not perceive
- [MMBench](https://arxiv.org/abs/2307.06281) - objective VLM evaluation
- [SeedBench](https://arxiv.org/abs/2307.16125) - generative comprehension benchmark
- [WildVision](https://arxiv.org/abs/2406.11069) - VLM eval with human preferences

**3D Foundation Models**:
- [3D-LLM](https://arxiv.org/abs/2307.12981)
- [Depth Anything](https://arxiv.org/abs/2401.10891) & [Depth Anything V2](https://arxiv.org/abs/2406.09414)
- [Marigold](https://arxiv.org/abs/2312.02145) - diffusion-based depth estimation

**Cognitive Science**:
- [Geirhos et al., arXiv 2020](https://arxiv.org/abs/2006.16736) - error consistency framework
- [Tuli et al., 2021](https://arxiv.org/abs/2105.07197) - Transformer vs CNN human alignment
- [Brain-Score](https://www.biorxiv.org/content/10.1101/407007v1)
- [Yamins et al., PNAS 2014](https://www.pnas.org/doi/10.1073/pnas.1403112111) - ventral stream modeling

**Specialized Models**:
- [MiDaS](https://arxiv.org/abs/1907.01341)
- [MDETR](https://arxiv.org/abs/2104.12763)
- [LightGlue](https://arxiv.org/abs/2306.13643)
- [SuperPoint](https://arxiv.org/abs/1712.07629)
- [LoFTR](https://arxiv.org/abs/2104.00680)
- [RAFT](https://arxiv.org/abs/2003.12039)

**Brain mechanisms**:
- [DiCarlo et al. Neuron 2012](https://www.cell.com/neuron/fulltext/S0896-6273(11)01009-X)
- [Welchman Annual Review Vision Science 2016](https://www.annualreviews.org/doi/10.1146/annurev-vision-111815-114935)

---

## 9. Final Intuition

这篇 paper 的 core message 可以浓缩为：

**"Current 3D vision models, whether VLM or specialized, solve 3D tasks via 2D statistical shortcuts. They achieve high accuracy on gravity-aligned, canonical views, but break catastrophically under trivial geometric perturbations. Human vision, by contrast, relies on geometric invariants processed by the dorsal stream."**

这给 future 3D foundation model 的开发者三个 actionable insight：
1. **数据**：需要大规模、geometrically diverse、gravity-agnostic 的 3D supervision
2. **架构**：Transformer 比 CNN 在 alignment with human 上确实有优势，但 attention 本身不够——可能需要 explicit 3D geometric inductive bias（SE(3) equivariance、epipolar constraint）
3. **评测**：永远把 geometric perturbation robustness 加入标准 eval suite，否则会被 high accuracy 误导

最终的目标应该是 build 一个 model，它在 upside-down image 上 depth accuracy 依然 84%，而 human 在它身上测 Cohen's κ = 0.85+——那才叫真正的 3D foundation model。我们离那个目标还很远，UniQA-3D 给我们一把尺子去量这个距离。
