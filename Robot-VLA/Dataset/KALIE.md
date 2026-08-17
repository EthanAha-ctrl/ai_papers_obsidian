---
source_pdf: KALIE.pdf
paper_sha256: ee9fe1f320e01fda2d9220800eb23762efe46676bba68abe75b7eece2009cfa3
processed_at: '2026-08-05T11:02:21-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# KALIE 用人话讲讲

## 一句话版本

让 AI 看 50 张人类随手标注的图片，学会用刷子扫地、关抽屉、挂毛巾、倒水、拔 USB——换完全没见过的东西也能搞定，全程不需要机器人自己练过一次。

Project page: https://kalie-vlm.github.io/

---

## 问题出在哪

现在让机器人学新 task，大概有三条路，每条都有坑：

**第一条路：让机器人自己狂练。** RT-2、OpenVLA、Octo 这些 VLA model 的思路是把 motor command 当 token，直接让 VLM 输出 action。要 work 你得有海量 teleoperation 数据，每次 trajectory 要一个熟练 operator 花几分钟盯着机械臂手柄操作。几千条 trajectory 意味着几个月的人工。BridgeData V2、DROID、Open X-Embodiment 这些数据集已经是 community 的极限了，但跟 Internet 上用来 pretrain VLM 的 image-text pair 比起来连零头都算不上。

**第二条路：直接 prompt GPT-4V 让它指挥机器人。** VoxPoser、MOKA、Code as Policies 走这条路。听起来很美，实际一跑就崩。GPT-4V 你问它 "brush 在哪",它能告诉你，但你让它精确输出 "grasp keypoint (245, 387)" 这种 2D pixel coordinate，它经常会 output 格式都搞错。Paper 里 VoxPoser 在 trowel pouring、USB unplugging 上成功率 0/15，原因就是 prompt engineering 救不回来 task-specific 的几何细节理解。

**第三条路：KALIE 走的这条路。** 把机器人控制拆成两段：VLM 只负责"看图说点"，motion planner 负责把点连成 trajectory。VLM 在这个 setup 下本质在做 image-grounded spatial reasoning——这正是它预训练时就擅长的事。然后给 50 张标注图，让 VLM 在这个 output format 上 fine-tune 一下，它就稳了。

Reference: 
- VoxPoser https://voxposer.github.io/
- MOKA https://moka-models.github.io/
- Open X-Embodiment https://robotics-transformer-x.github.io/

---

## 为什么让 VLM 只输出"点"就够了

这是 KALIE 最核心的 design choice，要 build 一下 intuition。

假设你想让机器人"用刷子扫走零食包"。你脑子里其实只闪过几个关键位置：

- 我要去抓刷子的哪里（grasp keypoint $p_g$）
- 刷子的功能端在哪（function keypoint $p_f$，bristle 那一头）
- 要扫的目标在哪（target keypoint $p_t$，零食包位置）
- 接触刷子之前从哪接近（pre-contact waypoint $p_{pre}$）
- 扫完之后从哪撤（post-contact waypoint $p_{post}$）

这 5 个点就足够 parameterize 整条 SE(3) motion trajectory 了。Low-level motion planner 拿到这 5 个 anchor，自己会做 collision-free interpolation、grasp planning、approach/retreat motion。这套 representation 来自 Manuelli et al. 的 KPAM [25] 和 MOKA [10]。

**Intuition**: VLM 在预训练时见过的 ground bounding box、keypoint estimation、image captioning，本质上都是在做 image-grounded spatial prediction。Affordance keypoints 跟这些任务同构。但 motor command（连续 6-DoF torque sequence）跟这些任务完全不同构——硬要 VLM 输出 action sequence 等于让它干一个它从来没见过的事。

公式 (1)：

$$\hat{y} = f(\rho, l, s)$$

- $f$: pretrained VLM，这里用的是 CogVLM-17B（清华 THUDM 出的）
- $\rho$: system prompt，固定不变
- $l$: language instruction，比如 "Use the brush to sweep the snack package"
- $s$: 单张 RGBD image（top-down third-person camera）
- $\hat{y}$: 输出，5 个 keypoints 的 2D 坐标 + 高度 $z$ + orientation $R$

注意 setup 是 **single-stage open-loop**——VLM 只 query 一次，然后 motion planner 一路执行到底，中间不重新观察。这听起来离谱，但因为 keypoints 已经 encode 了 task 的几何信息，open-loop 在 87-100% success rate 下居然能 work。

---

## 50 张图怎么变成 550 张——Imagined Environments 是真正的杀招

这是这篇 paper 最该被记住的部分。

### Naive 思路为什么不行

你可能会想：直接拿 SAM 分割出 object，让 Stable Diffusion inpaint 不就完了？ROSIE [44] 就是这么干的。问题是：

1. Diffusion model 不知道你想要什么 shape，你 inpaint "brush" 区域它可能给你生成一个完全不同形状的 tool，甚至是空桌面
2. 原 keypoint annotation 失配——object 形状变了，原来标的 grasp keypoint 现在指向空气
3. Task semantic 漂移——本来是扫地任务，inpaint 出一个完全无关的东西

Paper Figure 4 里 ROSIE 的 reimplementation 把 brush 区域 inpaint 成了空桌面（diffusion 觉得"刷子"这种 tool 描述模糊，干脆不生成）。这个问题对简单物体（towel）不严重，对复杂 tool 就崩了。

### KALIE 的方案：分两个问题，分别解决

**问题 1: 怎么控制 inpaint 出来的 object shape？**

用 ControlNet [47] 给 diffusion model 一个额外 condition image $c$。这个 $c$ 应该：
- 包含 affordance-relevant 的几何信息（object 轮廓、parts 边界）
- 不包含太多 appearance 信息（color、texture），留给 diffusion 自由发挥

KALIE 选了 **soft edge map** 作为 $c$。Soft edge 来自 Soria et al. [34] 的一个轻量 edge detector，跑得快、outline 清晰、保留 parts 边界（比如 brush 的 handle 和 bristle 之间的边）。

Paper 试过 depth map 和 segmentation mask 当 context（Figure 6 ablation），都不如 soft edge。原因：
- Depth map 有 general shape，但 parts 边界糊
- Segmentation mask 只有 silhouette，parts 完全丢失
- Soft edge 在 general shape 和 detailed parts 之间刚好

**Intuition**: context image 是一个 information bottleneck。它要 affordance-relevant 信息足够多（ControlNet 能据此约束形状），又要 appearance 信息足够少（diffusion model 有 free reign 做 texture diversity）。Soft edge 刚好在这两个要求之间平衡。

**问题 2: 怎么让 inpaint 出来的 object shape 多样化？**

KALIE 的 trick：不在 pixel space 变换，而是在 **soft edge space** 变换。

具体公式 (2)：

$$s' \sim g(\cdot | s, h(m) + m, h(m * c), o')$$
$$y' = h(y)$$

变量解释：
- $g$: ControlNet-conditioned diffusion model
- $s$: 原图
- $m$: object 的 segmentation mask（SAM 给的）
- $c$: soft edge context image
- $o'$: GPT-4V sample 出来的 alternative description（比如从 "blue plastic brush" 变成 "wooden kitchen brush"）
- $h(\cdot)$: 几何变换函数——random scaling、translation、rotation、还有 elastic distortion
- $h(m) + m$: 扩展 inpaint 区域——原 mask $m$ 加上变换后的 silhouette $h(m)$
- $h(m * c)$: 变换后的 context image，只在 mask 区域内取
- $y$: 原 keypoint annotation
- $y' = h(y)$: 同步变换后的 keypoint（因为 keypoint 跟着 object shape 一起走）

**为什么不在 pixel space 变换？** 因为 pixel-space rotation、scaling 会引入边界 artifact、texture 拉伸。Diffusion model 看到这些 artifact 会困惑。Soft edge 是抽象的 shape representation，对它做几何变换不引入任何 artifact。

**为什么 keypoint 能自动跟着变换？** 因为 $h(\cdot)$ 是作用在 image coordinate 上的 affine transform（+ elastic distortion）。$(x, y)$ 坐标和 soft edge map 是同一个 2D 坐标系，所以同一个 $h$ 作用在 keypoint 上就自动 align 了。这是整个 pipeline 的几何一致性核心。

### Multi-object 处理

Algorithm 1 里有个细节：scene 里的多个 object 是**逐个 inpaint** 的。第 $i$ 个 object inpaint 完，结果作为第 $i+1$ 个 object inpaint 的 background。这样 diffusion model 每次 focus 一个 object，生成质量高，object 之间不互相干扰。

### 成本

- 50 张 human annotation：每张几秒（GUI 上点几下 keypoint）
- 500 张 synthetic generation：90 分钟单 A100，平均每张 ~11 秒
- 总训练时间：8 小时单 A100
- 整体成本和"收集 50 条 teleop trajectory"差不多甚至更低，但得到 10 倍数据

参考：
- ControlNet https://github.com/lllyasviel/ControlNet
- SAM https://segment-anything.com/
- ROSIE https://rosie-robot.github.io/
- Stable Diffusion https://ommer-lab.com/research/latent-diffusion-models/

---

## VLM Fine-tune 的两个选项

### Option A: Regression Head

在 VLM 最后一个 token 的 hidden state 上接一个 linear layer，直接回归坐标。Loss 是 L2：

$$\mathcal{L}_{reg} = \|f_{reg}(h_{last}) - y\|_2^2$$

其中 $h_{last}$ 是 last token 的最后 hidden state。

### Option B: Natural Language Affordance Prediction

把 keypoints 写成格式化文字：

```
Grasp: (245, 387)
Function: (189, 412)
Target: (521, 398)
Pre-contact: (210, 350)
Post-contact: (310, 410)
```

坐标 normalize 到 [0, 999]。Loss 是 standard cross-entropy on tokens。

### 选哪个

Paper 在 Figure 6 ablation 里测出来两者 MSE 接近。但 KALIE 选了 NLAP（Option B），理由：

1. **跟 VLM 预训练目标完全对齐**——next token prediction，VLM 见过海量 "coordinate as text" 数据（OCR、charts、UI screenshot）
2. **不需要改 architecture**，直接 reuse VLM 的 LM head
3. **跟 VQA 生态兼容**，未来加 chain-of-thought reasoning 很自然
4. **LoRA fine-tune 更稳**

**Counterintuitive insight**: 给 VLM 加一个新 linear head 输出 coordinate，看起来更"直接"，实际更难 fine-tune——因为新加的 layer 没有任何 pretraining prior。让 VLM 输出 "245, 387" 这种 text 反而 sample-efficient，因为这些 token 在 VLM 的 embedding space 里已经有 well-structured representation 了。

### LoRA 配置

- Backbone: CogVLM-17B
- LoRA layers: 6
- Rank: 10
- Optimizer: Adam, weight decay $5 \times 10^{-2}$
- LR: $1 \times 10^{-5}$, cosine annealing
- Iterations: 6000
- Batch size: 4
- GPU: 单张 A100 80GB
- Training time: ~8 hours

LoRA rank 10 是相对保守的——目的是防止在 550 个样本上 overfit，同时保留 VLM 的 pretraining knowledge。

CogVLM https://github.com/THUDM/CogVLM
LoRA paper https://arxiv.org/abs/2106.09685

---

## 实验结果怎么读

### 主表（Table I）

5 个 task × 3 组 unseen objects × 15 trials：

| Method | Table Sweeping | Drawer Closing | Towel Hanging | Trowel Pouring | USB Unplugging |
|---|---|---|---|---|---|
| VoxPoser | 3/15 | 8/15 | 1/15 | 0/15 | 0/15 |
| MOKA | 9/15 | 9/15 | 5/15 | 7/15 | 2/15 |
| KALIE | **14/15** | **15/15** | **13/15** | **13/15** | **9/15** |

几个关键观察：

1. **VoxPoser 在 tool-use task 上直接报废**——USB 0/15、trowel 0/15。VoxPoser 靠 LLM 生成 code 来 compose 3D value map，缺乏 tool functional part 的概念
2. **MOKA 已经很强**，但 USB 2/15 显示 GPT-4V 对细粒度几何 reasoning 还是不稳，output 格式经常崩
3. **KALIE 用 CogVLM-17B（开源、小）击败 GPT-4V（闭源、大）**——fine-tuning 在 task-specific reasoning 上的 ROI 远高于 scale up

### USB Unplugging 是最难的 task

为什么？USB 拔插需要 grasp keypoint 精确到 mm 级（USB 头很小），orientation 要严格对齐。任何 affordance prediction 的小误差都会导致 grasp 失败。KALIE 9/15（60%）已经远好于 baseline，但仍有 40% 失败——说明 fine-grained spatial reasoning 仍是 open challenge。

### Ablation: 数据增广对比

Figure 5 在 table sweeping 上对比三种 data 策略，评估 test MSE（keypoint prediction error）：

- No augmentation: baseline
- Standard augmentation only（rotation, crop, flip, color jitter）: 主要改善 grasp keypoint（geometric invariant），对 function/target keypoint 改善有限
- Full method（standard + imagined environments）: 所有 5 个 keypoints MSE 都显著降

**关键 insight**: Standard augmentation 在 pixel space 操作，无法引入新 geometry。Imagined environments 引入新 object geometry，所以能 generalize 到 unseen objects。

### Ablation: Context Type

Figure 6 对比三种 context image：
- **Soft edge (主方法)**: 最好
- **Depth map**: 较差——parts 边界糊
- **Segmentation mask**: 最差——只有 silhouette，parts 完全丢失

### Ablation: Sample Scalability

- 50 examples + 500 synthetic (full method)
- 10 examples + 550 synthetic ("10 examples")
- 10 examples only ("10 examples w/o imagination")

结果：**10 examples + imagination ≈ 50 examples + imagination**！

**Deep insight**: 这说明 KALIE 的 bottleneck 是 imagination pipeline 的 diversity，不是 human annotation 数量。Imagination pipeline 越强，需要的人工标注越少。理论上如果 imagination pipeline 完美，人类只需标 5-10 张就能 generalize 到任意 unseen objects。

---

## 实验之外值得 build 的 intuition

### Insight 1: Affordance 是 VLM-Robot 的 "Right Interface"

VLM 擅长 image-grounded spatial reasoning，robot 需要 SE(3) motion。Affordance keypoints 在两者中间——既是 spatial（VLM 容易输出），又是 semantic（grasp、function、target 有明确 task 含义），又足够 parameterize motion。这个 abstraction 选对了，整个 system 就 work 了。

### Insight 2: Data Augmentation 应该在 Latent Geometric Space

传统 augmentation（crop、flip）在 pixel space，无法扩展 model 的 capability 边界。KALIE 把 shape 和 appearance decouple——shape 在 soft edge space 变换，appearance 在 diffusion pixel space 多样化。两个 dimension 独立 scale。

### Insight 3: Few-Shot Generalization 来自 Pretrained Prior

为什么 50 examples 能 generalize 到完全 unseen objects？因为 CogVLM 在 pretraining 时见过 Internet 上的海量 brushes、drawers、towels、USBs。Fine-tuning 只是把 output modality 从 VQA-style answer shift 到 affordance keypoints，不是从零教 VLM 认识这些 object。这呼应 Chen et al. [6] 的 "VLMs as promptable representations for RL" 观察。

### Insight 4: 人类标注成本降了 2-3 个数量级

Teleop trajectory 每个 5-10 分钟熟练操作，KALIE 的 keypoint annotation 每张几秒。如果这个 paradigm 能 scale，robotics data collection 的瓶颈可能从"收集不到足够数据"变成"VLM 不够强"。这是 game changer。

---

## Limitations 和可能的 Extension

### 1. Single-Stage Open-Loop

当前只能做 single-stage task。Multi-stage（"open drawer, take out cup, pour water"）需要 sequential decision making。

**Extension direction**: 把 KALIE 嵌入 hierarchical system——上层 LLM 做 task decomposition（SayCan 风格），下层 KALIE 做 per-stage affordance prediction。

### 2. CogVLM-17B vs GPT-4V

CogVLM-17B 跟 GPT-4V 仍有 capability gap。方法 model-agnostic，未来用 LLaVA-NeXT、Qwen-VL-Max、甚至开放 fine-tune API 的 GPT-4V 应该还能提一档。

### 3. Affordance Representation 的 Expressiveness

5 keypoints + height + orientation 表达不了：
- Bimanual manipulation（需要两个 grasp + relative pose）
- Deformable object shape control（towel folding 要多个 grasp）
- Dynamic manipulation（pouring 要 continuous tilt trajectory）

**Future direction**: affordance 从 discrete keypoints 扩到 continuous trajectory in SE(3)，或者 diffusion policy-style action representation。

### 4. Sim-to-Real Gap

Diffusion 生成 image 与真实 scene 仍有 distribution shift。Paper 没深入分析，但从 87-100% success rate 看 gap 可控——可能是因为 affordance keypoints 对 appearance 不敏感，只对 geometry 和 task semantics 敏感。

### 5. Zero-Shot Generalization Across Tasks

KALIE 当前是 few-shot per task。真正 zero-shot（在新 task 上不 fine-tune 直接 work）需要把 KALIE 在海量 task 上训练，形成 affordance generalist。

---

## 这篇 paper 的真正 legacy

如果让我押注，KALIE 这条路线在未来 robotics 里的影响可能比 VLA 路线更大：

1. **数据成本数量级下降**——teleop data 是 robotics 的 bottleneck，KALIE 把它彻底绕开
2. **方法 model-agnostic**——VLM 越强，KALIE 越强
3. **Imagination pipeline 的 idea 可 transfer**——shape-appearance decoupling 可以用到任何需要 data diversity 的 task
4. **Coordinate-as-text 比 regression head 更 sample-efficient**——这是个反直觉发现，值得在更多 spatial reasoning task 上验证

潜在风险：

1. **Open-loop 在 dynamic environment 必然失败**——任何有 moving object 或 articulated state 变化的 task 都需要 closed-loop
2. **Single-arm tabletop 的 affordance representation 可能不 scale**——bimanual、whole-body、locomotion+manipulation 都需要重新设计 affordance
3. **Diffusion 生成的 image distribution 跟 real scene 有 gap**——KALIE paper 里没显式量化这个 gap，但 success rate 数字暗示它可控

---

## 你可能想 follow up 的方向

1. 把 KALIE 跑在 GPT-4V fine-tune API 上，看 performance 上限
2. 把 affordance representation 从 5 keypoints 扩成 continuous trajectory（diffusion policy 风格）
3. 把 imagination pipeline 用在 video generation 上，直接生成 demonstration trajectory
4. 把 KALIE 加 closed-loop feedback，per-step re-plan
5. 把 KALIE 用在 bimanual task 上，重新设计 affordance representation
6. 用 LLaVA-NeXT、Qwen-VL-Max 替换 CogVLM-17B，测 VLM scale 的影响
7. 把 imagination pipeline 用在其他 robot learning task 上（navigation、whole-body control）

---

## Useful Links

- KALIE Project Page: https://kalie-vlm.github.io/
- MOKA (predecessor): https://moka-models.github.io/
- VoxPoser: https://voxposer.github.io/
- ROSIE: https://rosie-robot.github.io/
- CogVLM: https://github.com/THUDM/CogVLM
- ControlNet: https://github.com/lllyasviel/ControlNet
- SAM: https://segment-anything.com/
- Stable Diffusion: https://huggingface.co/stabilityai/stable-diffusion-2
- KPAM: http://kpam.csail.mit.edu/
- LoRA paper: https://arxiv.org/abs/2106.09685
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- BridgeData V2: https://rail-berkeley.edu/bridgedata/
- DROID: https://droid-dataset.github.io/
- Octo: https://octo-models.github.io/
- RT-2: https://robotics-transformer2.github.io/

如果你想让我深入某个具体部分（比如 motion planner 怎么从 5 个 keypoints 生成 SE(3) trajectory，或者 ControlNet 的 conditioning 机制细节，或者 soft edge detector [34] 的具体算法），告诉我。

---

# KALIE: Fine-Tuning Vision-Language Models for Open-World Manipulation without Robot Data 深度解析

## 一、Paper 的核心动机与 Big Picture

这篇 paper 来自 Sergey Levine 和 Kuan Fang 的组 (UC Berkeley + Northwestern),地址是 https://kalie-vlm.github.io/ 。要 build intuition,我们先把它的位置定位清楚。

当前 robotics + foundation models 有两条主流路线:

1. **VLA (Vision-Language-Action) 路线**: 把 motor command 当成 token,直接 fine-tune VLM 输出 action。代表: RT-2, OpenVLA, Octo。问题:需要海量 teleoperation 数据,Internet 上的视频/图像数据无法直接复用,action space 离散化损失信息。
2. **Zero-shot prompting 路线**: 直接 prompt GPT-4V 让它输出控制信号。代表: VoxPoser, MOKA, Code as Policies。问题:对 prompt engineering 依赖重,output 格式不稳定,复杂 task (tool use, articulated object) 上 generalization 差。

KALIE 选了一条中间路径——**保留 VLM 的 language-vision pretraining,但把 action 解耦成 affordance representation**,让 VLM 只负责"看图说话"式的空间预测,真正的 motion generation 交给一个 low-level motion planner。这样做有两个好处:

- VLM 的预训练分布(VQA、grounding)与 affordance prediction 同构,可以高效 fine-tune
- 不需要 robot trajectory data,只需要 (image, keypoints) pairs,人类几秒钟就能标一个

核心 insight 用一句话概括: **Affordance 是 VLM 和 robot 之间的 ideal interface,因为它是 2D spatial、language-grounded、且与 low-level dynamics 解耦的**。

---

## 二、Affordance Representation 深度讲解

KALIE 沿用了 MOKA [10] 的 point-based affordance representation。我们看一下它包含哪 5 个 keypoints:

1. **Grasp keypoint $p_g$**: gripper 应该闭合抓取的位置
2. **Function keypoint $p_f$**: tool 的 functional part,比如 brush 的 bristle 端
3. **Target keypoint $p_t$**: 任务目标位置,比如 trash package 的位置
4. **Pre-contact waypoint $p_{pre}$**: 接触前的 approach waypoint
5. **Post-contact waypoint $p_{post}$**: 接触后的 retreat waypoint

每个 keypoint 都是 2D image coordinate $(x, y)$,再加上额外的 properties (gripper height $z$, orientation $R \in SO(3)$)。

**为什么这套 representation 工作?** 它把 SE(3) 的连续 motion trajectory 压缩成几个语义 anchor points,显著降低了 VLM 的输出空间复杂度。比如"sweeping with a brush"只需要 5 个点 + 几何 properties,motion planner 就能 interpolate 出整条 trajectory。这和 keypoint-based pose estimation (KPAM, Manuelli et al. [25]) 的思路一脉相承。

**Intuition**: VLM 不擅长输出连续 control,但非常擅长 image-grounded spatial reasoning ( bounding box, segmentation, keypoint)。"不直接做 control,而是预测 affordance"是一种合理的 task decomposition。

---

## 三、Problem Formulation 细节

公式 (1):

$$\hat{y} = f(\rho, l, s)$$

变量解释:
- $f$: 预训练 VLM (这里用 CogVLM-17B)
- $\rho$: 固定的 system prompt
- $l$: free-form language instruction,比如 "Use the brush to sweep the snack package."
- $s$: 单张 RGBD image (third-person camera)
- $\hat{y}$: 预测的 affordance representation,包含 5 个 keypoints 的 2D coordinates

注意 KALIE 是 **single-stage, open-loop** task setting。这意味着 VLM 只 query 一次,motion planner 拿到 $\hat{y}$ 后生成整条 6-DoF trajectory 直接执行。这是 affordance-based 方法的优势——open-loop 也能 work,因为 keypoints 已经编码了 task 的核心几何信息。

Few-shot 设定: 每个任务只有 $|D| = 50$ 个 (image, keypoints) pairs,而且全部 collected 在 **同一组 objects** 上。Evaluation 在 3 组完全 unseen objects 上做。这是非常极端的 generalization 测试。

---

## 四、Affordance-Aware Data Synthesis:这篇 paper 的真正核心

这是 KALIE 区别于 MOKA 和 ROSIE [44] 的关键创新。我们要深度拆解。

### 4.1 Naive Inpainting 的问题

ROSIE [44] 的思路是: 用 SAM [18] 提取 object mask,然后用 Stable Diffusion [31] inpaint。问题:

- **几何不可控**: Diffusion model 不知道你想 inpaint 什么形状的 brush,可能 inpaint 出完全不同形状的 tool
- **Keypoint 失配**: Inpaint 后的 object 几何变了,原 keypoint annotation 不再 valid
- **Task semantic drift**: Inpaint 一个完全无关的 object,task context 丢失

KALIE 在 paper Figure 4 中展示: naive inpainting 经常把 brush 区域 inpaint 成 "empty table",或者把 trash inpaint 成完全不同的物体。

### 4.2 KALIE 的解决方案:ControlNet + Soft Edge + 几何变换

公式 (2) 是核心:

$$s' \sim g(\cdot | s, h(m) + m, h(m * c), o)$$
$$y' = h(y)$$

变量解释:
- $g$: ControlNet [47] conditioned diffusion model
- $s$: 原始 image
- $m$: object 的 segmentation mask (从 SAM 获得)
- $c$: **context image** (这里是 soft edge map)
- $o$: 物体的 language description
- $h(\cdot)$: 几何变换函数 (scaling, translation, rotation, elastic distortion)
- $h(m) + m$: 扩展后的 inpaint 区域——包含原 mask 区域 $m$ 加上变换后的 silhouette $h(m)$
- $h(m * c)$: 变换后的 context image (只在 mask 区域内)
- $y'$: 同步变换后的 keypoints

**关键 insight**: KALIE 不在 pixel space 上做 augmentation,而是在 **compact geometric representation (soft edge map) space** 上做。这样:

1. Soft edge 保留了物体的轮廓和 parts 信息(比如 brush 的 handle 和 bristle 的边界)
2. 但丢弃了 texture、color 等 detail,留给 diffusion model 自由发挥
3. 对 soft edge 做几何变换,等同于变换 object 的形状骨架,同时 keypoints 跟着一起变换,保持 consistency

**为什么不直接在 pixel space 变换?** 因为 pixel-space 变换会引入 artifacts(边界 artifact、texture 拉伸),而且 diffusion model 会被这些 artifact 误导。Soft edge 是一种 "shape prior",抽象程度刚好。

### 4.3 Pipeline 的 Multi-Object 处理

Algorithm 1 展示了完整 pipeline:

```
For each scene s in D:
    1. Compute context image c (soft edge)
    2. For i = 1 to M (objects):
        a. Extract description o_i (用 VLM)
        b. Get segmentation mask m_i (用 SAM)
        c. Sample alternative description o_i' (用 GPT-4V)
        d. Transform: h(m_i), h(m_i * c)
        e. Transform keypoints: h(y_i)
        f. Inpaint: s_i' ~ G(· | s_{i-1}', h(m_i) + m_i, h(m_i * c), o_i')
    3. Merge transformed keypoints -> y'
```

这里有个细节值得 build intuition: **object 是逐个 inpaint 的**,而不是一次性 inpaint 所有 mask。这样做的好处是 diffusion model 每次 focus 在一个 object 上,生成质量更高,而且不同 object 之间不会互相干扰。

另一个关键 trick: **用 GPT-4V sample alternative description $o_i'$**。比如原 description 是 "blue plastic brush",GPT-4V 可能 sample 出 "wooden kitchen brush" 或 "metal cleaning brush"。这给 diffusion model 提供了 texture/material 的 diversity,但 shape 仍然由 soft edge context 约束。

### 4.4 Computational Cost

Paper 报告: 生成 500 张合成 image 在单张 A100 上花 90 分钟,平均每张 ~11 秒。这是非常 reasonable 的成本,远低于 teleoperation 收集 robot trajectory 的时间。

---

## 五、VLM Fine-Tuning 的两个设计选项

### 5.1 Regression Head

在 VLM 的最后一个 hidden state 上加一个 linear layer,直接回归 keypoints 的 $(x, y)$ 坐标。Loss 是 L2 regression:

$$\mathcal{L}_{reg} = \|f_{reg}(h_{last}) - y\|_2^2$$

其中 $h_{last}$ 是 last token 的 last hidden state。

### 5.2 Natural Language Affordance Prediction

把 keypoints 表示为格式化的 natural language,比如:

```
Grasp: (245, 387)
Function: (189, 412)
Target: (521, 398)
Pre-contact: (210, 350)
Post-contact: (310, 410)
```

坐标 normalized 到 [0, 999] 范围内。Loss 是 standard cross-entropy on tokens。

### 5.3 为什么二者性能相近但选了 NLAP?

Paper 在 Figure 6 ablation 中显示两者 MSE 接近。但 KALIE 选 Natural Language Affordance Prediction (NLAP) 作为主方法,理由:

- 与 VLM 的 pretraining objective (next token prediction) 完全对齐
- 不需要额外修改 model architecture,直接 reuse VLM 的 LM head
- 与 VQA 生态兼容,可以轻松加 chain-of-thought reasoning
- LoRA fine-tuning 更稳定

**Intuition**: VLM 在预训练时见过了大量 "coordinate as text" 的数据 (OCR、charts、UI screenshots),所以把 coordinate 当 token 来 predict 反而比加一个新 linear head 更 sample-efficient。这个观察其实呼应了最近一系列 "LLMs are secretly good at spatial reasoning if you let them output text" 的工作 (e.g., Shikra [5])。

### 5.4 LoRA Fine-Tuning 配置

- Backbone: CogVLM-17B
- LoRA layers: 6
- Rank: 10
- Optimizer: Adam, weight decay $5 \times 10^{-2}$
- Learning rate: $1 \times 10^{-5}$, cosine annealing
- Iterations: 6000
- Batch size: 4
- Hardware: 单张 A100 80GB
- Training time: ~8 hours

LoRA 只 fine-tune 一小部分参数,极大降低了 memory footprint,使得 17B 的 VLM 可以在单卡上 fine-tune。Rank 10 是相对保守的选择,意味着 fine-tune 的 capacity 有限,但这也防止了 overfitting (因为只有 550 个 training samples)。

---

## 六、Experimental Results 详细分析

### 6.1 主实验: 5 Tasks × 3 Unseen Object Sets × 15 Trials

Table I 的 success rate:

| Method | Table Sweeping | Drawer Closing | Towel Hanging | Trowel Pouring | USB Unplugging |
|---|---|---|---|---|---|
| VoxPoser [13] | 3/15 (20%) | 8/15 (53%) | 1/15 (7%) | 0/15 (0%) | 0/15 (0%) |
| MOKA [10] | 9/15 (60%) | 9/15 (60%) | 5/15 (33%) | 7/15 (47%) | 2/15 (13%) |
| KALIE | 14/15 (93%) | 15/15 (100%) | 13/15 (87%) | 13/15 (87%) | 9/15 (60%) |

观察:

1. **VoxPoser 在 tool-use task 上几乎完全失败** (Trowel Pouring 0/15, USB Unplugging 0/15)。这是因为 VoxPoser 通过 LLM 生成 code 来 compose 3D value maps,缺乏对 tool functional part 的理解。
2. **MOKA 比 VoxPoser 强很多**,因为 GPT-4V 直接 predict affordance keypoints,但仍受限于 prompt engineering 和 GPT-4V 在 spatial reasoning 上的不稳定性。
3. **KALIE 在所有任务上都显著优于 baselines**,尤其在 USB Unplugging (从 13% → 60%) 这种精细 task 上提升最大。

注意 baselines 用 GPT-4V,KALIE 用 CogVLM-17B——KALIE 用一个**更小的开源模型**击败了更大的闭源模型。这说明 fine-tuning 在 task-specific reasoning 上的 ROI 远高于 scale up。

### 6.2 Ablation: Data Augmentation 的贡献

Figure 5 在 Table Sweeping 任务上比较三种 data 策略,以 MSE (keypoint prediction error) 评估:

- No augmentation: baseline
- Standard augmentation only: rotations, crops, flips, color jitter
- KALIE full method: standard + imagined environments

KALIE 的 imagined data 在所有 5 个 keypoints 上都显著降低 MSE,尤其是 function keypoint 和 target keypoint——这两个是 task-semantic 最关键的。Standard augmentation 主要帮助 grasp keypoint (相对几何 invariant),对 task-specific keypoints 帮助有限。

### 6.3 Ablation: Context Type

Figure 6 比较了三种 context image 给 ControlNet:

- **Soft edge (主方法)**: 最好
- **Depth map**: 较差
- **Segmentation mask**: 较差

为什么?Paper 解释:depth 和 seg mask 保留了 "general shape" 但丢失了 parts detail (比如 brush bristle 的边界)。Soft edge (用了 [34] 的 tiny efficient edge detector) 在 abstract shape 和 detailed parts 之间取得了平衡。这给 ControlNet 提供了足够的 structural guidance,同时留出 texture freedom 给 diffusion model。

**Intuition**: 这其实是一个 information bottleneck 的设计——context image 应该包含 **足够多的 affordance-relevant 几何信息**,但 **足够少的 appearance 信息**,让 diffusion model 可以 diverse 化生成。

### 6.4 Ablation: Data Scalability

Figure 6 还比较:
- 50 examples + 500 synthetic (full method)
- 10 examples + 550 synthetic ("10 examples")
- 10 examples only ("10 examples w/o imagination")

关键发现: **10 examples + imagination ≈ 50 examples + imagination**。这说明 KALIE 的 bottleneck 不是 example data 数量,而是 imagination pipeline 的 diversity。这是非常强的 scalability 信号——只要 imagination pipeline 足够好,人类 annotation 的成本可以压到极低。

---

## 七、Build Intuition: 这篇 paper 的真正贡献

让我从更高 level 总结 KALIE 的几个 deep insights:

### 7.1 Affordance 是 VLM-Robot Interface 的 "Right Abstraction"

VLM 擅长什么? Image-grounded language reasoning。Robot 需要什么? SE(3) motion trajectory。这两者之间有巨大的 semantic-geometric gap。Affordance keypoints 是这个 gap 上的桥梁——它们既是 spatial (2D coordinate, VLM 容易输出),又是 semantic (grasp, function, target 都有明确 task 含义),又足以 parameterize motion。

### 7.2 Data Augmentation 应该在 "Latent Geometric Space" 而不是 Pixel Space

传统 data augmentation (crop, flip, color jitter) 在 pixel space 操作,无法引入新的 object geometry。KALIE 的创新是:**先把 object 几何信息压缩到 soft edge map,在 edge space 做几何变换,然后让 diffusion model 在 pixel space 填充 appearance**。这相当于 decouple 了 "shape diversity" 和 "appearance diversity",两者可以独立 scale。

### 7.3 Few-Shot Generalization 来自 Pretrained VLM 的 Prior

为什么 50 个 examples 能 generalize 到 unseen objects? 因为 VLM 在 pretraining 时已经见过 Internet 上的海量 brushes, drawers, towels, USBs。Fine-tuning 只是 **shift the output modality** 从 VQA-style answer 到 affordance keypoints,而不是从 zero 教 VLM 认识这些 object。这呼应了最近 "VLMs as promptable representations for RL" [6] 的观察。

### 7.4 不需要 Robot Data 是 Game Changer

Collecting teleoperation data 是 robotics 的最大 bottleneck。每个 trajectory 需要几分钟 skilled operator 时间,而且必须 on-robot。KALIE 让 human 用 GUI 在 image 上标 keypoints,每个 sample 几秒钟。这意味着 **数据收集成本降低了 2-3 个数量级**。如果这个 paradigm 能 scale,KALIE 这类方法可能成为 robotics data collection 的新 standard。

---

## 八、Limitations 和 Future Work 的深入思考

### 8.1 Single-Stage, Open-Loop 限制

当前 KALIE 只支持 single-stage task。Multi-stage task (比如"open drawer, take out cup, pour water")需要 sequential decision making,这超出了 single forward pass 的 affordance prediction 能力。

**可能的 extension**: 把 KALIE 嵌入到一个 hierarchical system,上层 LLM 做 task decomposition,下层 KALIE 做 per-stage affordance prediction。这其实就是 SayCan / Code as Policies 的架构,只是底层 affordance predictor 换成了 fine-tuned KALIE。

### 8.2 CogVLM-17B vs GPT-4V 的差距

Paper 承认 CogVLM-17B 与 GPT-4V 之间仍有 capability gap。但 KALIE 的方法本身是 model-agnostic——只要 VLM 可以 LoRA fine-tune 就行。未来用 LLaVA-NeXT、Qwen-VL-Max、甚至 GPT-4V fine-tune API (如果开放的话),performance 应该还能再提一个台阶。

### 8.3 Affordance Representation 的 Expressiveness

5 个 keypoints + height + orientation 可以表达很多 single-arm tabletop task,但无法表达:
- Bimanual manipulation (需要两个 grasp points 和 relative pose)
- Deformable object manipulation 中的 shape control (towel folding 需要多个 grasp points)
- Dynamic manipulation (pouring 需要连续 tilt angle trajectory)

**Future direction**: 把 affordance representation 从 discrete keypoints 扩展成 continuous trajectory in SE(3),或者用 diffusion policy-style 的 action representation。

### 8.4 Sim-to-Real 的隐性 Gap

虽然 KALIE 不用 simulator,但 diffusion model 生成的 image 仍有 distribution shift 到真实 scene。Paper 没有详细分析这个 gap 的大小,但从结果看 (87-100% success rate) gap 似乎可控。这可能是因为 affordance keypoints 对 appearance 不敏感,只对 geometry 和 task semantics 敏感。

---

## 九、和 Related Work 的关系图谱

让我帮你在脑中建立这个领域的 map:

- **Affordance prediction 经典**: KPAM [25] → MOKA [10] → KALIE
- **VLM for robotics zero-shot**: Code as Policies [23] → VoxPoser [13] → MOKA [10] → KALIE (加入 fine-tuning)
- **Data synthesis for robotics**: MetaSim [14] → ROSIE [44] → KALIE (加入 affordance-aware context)
- **VLM fine-tuning for spatial output**: Shikra [5] → CogVLM [39] → KALIE
- **VLA models**: RT-2 [3] → OpenVLA → Octo [27] (不同路线,KALIE 不直接 compete 但互补)

KALIE 的位置: 它是 affordance prediction 路线上第一个 **combine VLM fine-tuning + diffusion-based data synthesis** 的工作,在 generalization 和 data efficiency 上都做到了 SOTA。

---

## 十、值得 Follow 的问题和 Personal Take

作为 deep learning 视角的 takeaways:

1. **Soft edge 作为 context 是个有趣的 information bottleneck 设计**。这个 idea 可能 transfer 到其他 conditional generation 任务 (e.g., text-to-video with motion control)。

2. **Coordinate-as-text 的输出方式比 regression head 更 sample-efficient**。这是一个反直觉但重要的观察,值得在更多 spatial reasoning 任务上验证。

3. **Imagination pipeline 的 bottleneck 是 example data 的 diversity,不是数量**。这说明未来可以进一步 reduce human annotation,甚至 zero-shot imagination。

4. **Open-source VLM + LoRA 已经可以击败 GPT-4V zero-shot**。这对开源社区是 strong positive signal。

5. **下一个突破点可能是 multi-stage task + closed-loop feedback**。KALIE 当前是 open-loop,如果加 visual feedback 后 re-plan,可能能解决 dynamic environment 中的 task。

---

## References 和 Useful Links

- KALIE Project Page: https://kalie-vlm.github.io/
- MOKA (predecessor): https://moka-models.github.io/
- VoxPoser: https://voxposer.github.io/
- ROSIE: https://rosie-robot.github.io/
- CogVLM: https://github.com/THUDM/CogVLM
- ControlNet: https://github.com/lllyasviel/ControlNet
- Segment Anything (SAM): https://segment-anything.com/
- Open X-Embodiment Dataset: https://robotics-transformer-x.github.io/
- KPAM (original keypoint affordance): http://kpam.csail.mit.edu/
- LoRA: https://arxiv.org/abs/2106.09685

如果你对其中某个细节(比如 ControlNet 的 conditioning mechanism、soft edge detector 的具体算法、或者 motion planner 怎么从 5 个 keypoints 生成 SE(3) trajectory)想深入,我可以继续展开。
