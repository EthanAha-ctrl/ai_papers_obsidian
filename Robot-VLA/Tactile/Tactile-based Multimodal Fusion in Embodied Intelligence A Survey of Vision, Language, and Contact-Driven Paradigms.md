---
source_pdf: Tactile-based Multimodal Fusion in Embodied Intelligence A Survey of Vision,
  Language, and Contact-Driven Paradigms.pdf
paper_sha256: 304be4bd040c2aafe3e9225cf018be4201e5fc6bdaae3038b1a074f1d158058c
processed_at: '2026-08-12T12:35:53-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Survey

兄弟我重新用大白话给你讲一遍。之前那个版本太像 textbook 了，咱们这次讲"这事儿到底在干嘛"。

---

## 1. 这领域到底在解决啥问题

你想想机器人抓东西这事。机器人的 camera 能看到前面有个杯子，但它不知道杯子滑不滑、重不重、烫不烫。vision 给你的是"远处看到啥"，touch 给你的是"摸到啥感觉"。这俩信息完全不一样——一个 remote sensing，一个必须 physically 接触。

人类为啥抓东西厉害？因为你眼睛看到杯子 + 手摸到杯子 + 脑子里知道"杯子"这个概念，三者一起 work。任何一个单独都不够：
- 只有 vision：透明杯子看不到，反光物体看不清
- 只有 touch：你摸到一个圆的硬的，但不知道这是杯子还是球
- 只有 language："杯子"这个词不含此时此刻的 physical state

所以这帮人想干的事就是：**让机器人也能像人一样，把 vision + touch + language 三者融合起来用**。这就是 "Multimodal Tactile Fusion" 这整个 field。

---

## 2. 为什么这事难

几个根本性的困难：

### Data 太少
ImageNet 有 14 亿张图，最大的 tactile dataset 也就 260 万样本（Touch in the Wild [49]，https://arxiv.org/abs/2507.15062）。差 500 倍。原因很简单：图片网上随便 scrape，tactile data 必须真的拿机器人去摸东西才能 collect。一小时能 collect 几千张图，但摸一千个 object 的 tactile signal 要好几天。

### Sensor 太碎
Camera 基本就那几种格式（RGB、RGBD），但 tactile sensor 五花八门：
- GelSight [30]：像个带 gel 的小 camera，输出 RGB image（https://www.mdpi.com/1424-8220/17/12/2762）
- DIGIT [31]：更小的 fingertip camera（https://ieeexplore.ieee.org/document/9035056）
- Tac3D [32]：输出 3D point cloud（https://arxiv.org/abs/2202.06211）
- Force sensor：输出 scalar 数字

这导致一个问题：你在 GelSight 上 train 的 model，换到 DIGIT 上不能直接用。这跟 NLP 里 tokenizer 不统一类似，但更严重，因为 physical output format 都不一样。

### 信号本身 strange
GelSight 输出的 image 长得很 weird——不是自然图像，是 gel 被压变形后的反光图。所以你直接用 ImageNet pretrain 的 ResNet 在 GelSight image 上 fine-tune，效果不如在 natural image 上好。分布差太远。

### Temporal 信息关键
一张静态 tactile image 信息量不大，但一段 tactile sequence（你按下去、滑一下、松开的整个过程）信息量巨大。所以 tactile model 通常要 handle temporal dimension，比 vision-only 复杂。

---

## 3. 这篇 Survey 把 field 怎么切的

作者把整个 field 切成三块，我觉得切得挺合理：

### 第一块：Dataset

按 modality 组合分四种：
- **T-V**：touch + vision，最早最成熟，从 2016 年就有了
- **T-L**：touch + language，很新，2024 才开始
- **T-V-L**：三者都有，2024 后爆发，现在最火的
- **T-V-O**：再加 audio、action、proprioception 等

**Trend 一句话**：从 lab 里 robot 抓几个固定 object，到人戴着手套去外面随便摸，再到 100k scale 的 trimodal dataset。T-V-L 的 Touch100k [6]（https://arxiv.org/abs/2502.12191 附近的 2025 系列）是现在的 SOTA scale。

### 第二块：Method

按 task 分三大类：

**(1) Perception & Recognition**：识别这是啥物体、啥材质、grasp 会不会成功
**(2) Cross-Modal Generation**：给定 vision 生成 tactile，或给定 tactile 生成 text 描述
**(3) Interaction & Manipulation**：真的用这些信号去控制机器人动作

### 第三块：Sensor

四类硬件：
- **Wearable**：手套那种，给人戴的，用来 collect human demonstration data
- **Fingertip**：装在 robot finger 上的，GelSight/DIGIT 这类
- **Robotic skin**：大面积覆盖 robot body 的
- **Gripper-mounted**：直接装在 parallel jaw gripper 上的

---

## 4. Method 演进的人话版

这块我用最直白的方式讲。你看这个 field 的演进，本质就是 vision-language field 的复刻，晚 3-5 年。

### Stage 1: 2016-2019，CNN 时代

最早的方法就是简单粗暴：visual feature 用 ResNet 提，tactile feature 也用 CNN 提，然后把两个 feature vector concat 一下，接个 MLP 做分类。代表是 VT [40]、ViTac [10]。

那时候大家的 intuition 是：vision 和 touch 都是 image-like（GelSight 输出 image），所以可以 share backbone。但很快发现 GelSight image 和自然图像分布差太远，share backbone 效果不好。

### Stage 2: 2020-2023，Transformer + Contrastive Learning

ViT 出来后大家开始用 Transformer。VITO-Transformer [78] 开始用 attention 做 vision-tactile fusion。关键 insight：你摸的地方和你看的地方有 spatial correspondence，cross-attention 能学这个 alignment。

同时 CLIP 出来，大家发现 contrastive learning 这套可以搬过来。Touch and Go [46]（https://arxiv.org/abs/2211.12498）是里程碑：让人戴手套去外面摸 3971 个物体，collect 13.9k paired vision-tactile data，用 contrastive loss 训练，能 transfer 到 downstream task。

### Stage 3: 2024-2026，Foundation Model 时代

这才是真正 exciting 的部分。核心 trick 是 UniTouch [2]（https://arxiv.org/abs/2401.12186）想出来的：

**把 tactile encoder 的输出 align 到 CLIP 的 embedding space**。

为啥这招厉害？因为 CLIP 已经知道"smooth"、"rough"、"metal"、"wood"这些概念的 semantic embedding。你只要让 tactile encoder 输出和 CLIP embedding 对齐，tactile 就免费获得了 CLIP 的全部 semantic knowledge。然后你就可以做 zero-shot：摸一个东西，encoder 输出 embedding，和 "smooth metal" 的 text embedding 算 cosine similarity，就知道摸的是不是光滑金属。

UniTouch 之后一堆 follow-up：
- Tactile-VLM [3]：加 language 进来，trimodal alignment
- TLV-Link [6] / Touch100k：100k scale，scale up data
- Octopi [50]：用 LLM 做 tactile reasoning
- CLTP [5]：3D tactile point cloud + language

### Stage 4: 2025-2026，VLA + Tactile

这是现在最前沿。VLA model 就是 vision-language-action model，输入 vision + language instruction，输出 robot action。RT-2、OpenVLA、π0 都是这路。但它们都缺 tactile。

VTLA [7]（https://arxiv.org/abs/2505.09577）的做法：在 VLA 里加 tactile token，让 policy 能感知 contact state。这样机器人抓透明杯子时，vision 看不清，但 tactile 知道"我在抓一个光滑硬的东西，force 够不够"，就不会掉。

OmniVTLA [55]（https://arxiv.org/abs/2508.08706）把 tactile token semantic-aligned 之后再塞进 action model，效果更好。

VLA-Touch [96]（https://arxiv.org/abs/2507.17294）做了 dual-level tactile feedback——既有 low-level force feedback，又有 high-level semantic feedback。

---

## 5. 几个核心 Method 的 Technical Detail

我挑三个最有代表性的讲。

### UniTouch [2]：Tactile 的 CLIP 时刻

**架构**：
- Vision encoder：ViT-B/16，用 CLIP 预训练权重初始化
- Tactile encoder：ViT-B/16，也用 CLIP vision encoder 初始化（关键 trick！）
- Text encoder：CLIP text encoder，frozen

**Training objective**：InfoNCE loss，就是 CLIP loss

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ \log \frac{\exp(z_t^i \cdot z_v^i / \tau)}{\sum_{j=1}^N \exp(z_t^i \cdot z_v^j / \tau)} + \log \frac{\exp(z_t^i \cdot z_l^i / \tau)}{\sum_{j=1}^N \exp(z_t^i \cdot z_l^j / \tau)} \right]$$

变量解释：
- $N$：batch size
- $z_t^i$：第 $i$ 个 tactile sample 的 embedding
- $z_v^i, z_l^i$：对应的 vision 和 text embedding
- $\tau$：temperature parameter，控制 softmax 的 sharpness
- $z_t^i \cdot z_v^i$：cosine similarity（假设已 normalize）

**Intuition**：batch 内每个 tactile 样本对应一个 vision 和一个 text，要把正确的 (tactile, vision, text) triplet 拉近，把不对应的推远。训完之后 tactile embedding 和 CLIP 的 vision/text embedding 在同一空间，所以可以直接 reuse CLIP 的全部 knowledge。

**为啥 work**：因为 GelSight image 虽然分布和自然图像不同，但低层 visual feature（edge、texture、color）有共性。用 CLIP vision encoder 初始化 tactile encoder，等于告诉 tactile encoder "你已经知道 smooth surface 长啥样了，只要 adapt 到 GelSight 的 distribution 就行"。

### VTLA [7]：VLA 加 Tactile

**架构**：
- Vision encoder：Qwen-VL 的 vision branch
- Tactile encoder：ViT
- Language：Qwen-VL 的 language model
- Action head：MLP，输出 7-DoF end-effector pose + gripper force

**Training**：preference learning，用 RLHF 思路。人标注 preferred action 和 dispreferred action，model 学会选 preferred。

**Intuition**：纯 VLA 在 transparent cup、folded cloth 上 fail，因为 vision 看不清。加 tactile 后，policy 能感知"我抓到东西没"、"force 够不够"、"object 在不在 slipping"，这些是 vision 给不了的。VTLA 在 insertion task 上比纯 VLA success rate 高很多。

### OmniVTA [64]：Tactile World Model

这个最 exciting。world model 的 idea 是：学一个能 predict future state 的 model，然后用它做 planning。

**架构**：
- 输入：current vision + tactile + action
- 输出：next vision + tactile state

**为啥重要**：你有了 world model，就能做 model-based RL 或 planning。给定当前 state，sample 一个 action，predict 下一个 state 好不好，好就执行，不好就换。这比 model-free RL sample efficiency 高很多。

DreamTacVLA [95]（https://arxiv.org/abs/2512.23864）是类似思路，predict future tactile state 来 stabilize 长时间 contact-rich interaction。

---

## 6. 几个 Practical 的 Insight

读完这篇 survey 我有几个 takeaways 想跟你分享：

### Insight 1: 这领域现在是 vision 2018 年的位置
算法基本 ready（contrastive learning、Transformer、VLA 都有了），就等 data 爆发。谁先 scale up tactile data collection 到 10M+ scale，谁就做出 tactile foundation model。OSMO glove [102]、UMI-FT [110] 这种 portable system 是 trying to solve this。

### Insight 2: Tactile 是 VLA 的 killer feature
当前 VLA model 在 transparent、deformable、articulated object 上 fail 的根本原因是 vision-only limitation。加 tactile 是 natural solution。我预测 2026-2027 会有大批 VLA + tactile 的工作出来。

### Insight 3: Sensor agnosticism 是 unsolved but critical
AnyTouch [9]（https://arxiv.org/abs/2502.12191）尝试 unified representation across sensors，但还 early。这个问题不解决，每换一个 sensor 就要 retrain，field scaling 不起来。类比 NLP 的 tokenizer 之争——最后 GPT 系的 BPE 赢了，tactile 也需要类似的 de facto standard。

### Insight 4: Simulation 是 scaling 的关键
真实 collect tactile data 太慢，必须靠 simulation。但 tactile simulation 难度比 vision 大——你要 simulate contact mechanics、gel deformation、friction，这是 FEM 级别的物理仿真。TACTO、Taxim 这些 simulator 在做，但 fidelity 还不够。这是 sim-to-real 的核心 bottleneck。

### Insight 5: Evaluation 是 mess
Survey Section VI 说得很坦白：没有 unified benchmark，每篇 paper 用不同 dataset、不同 task、不同 metric，没法 fair comparison。这跟 ImageNet 之前的 vision 一样。需要有人站出来 define 一个 standard benchmark，像 ImageNet 那样。

---

## 7. 如果你要进这领域，我建议的路径

1. **先 reproduce UniTouch** [2]：理解 contrastive alignment 怎么 work，跑通 zero-shot retrieval
2. **搭一个 GelSight 或 DIGIT gripper**：自己 collect 100 个 object 的 tactile data，感受 signal 特性。你会立刻理解为啥 vision-only 不够
3. **读 VTLA** [7] 和 OmniVTLA [55]：理解 tactile 怎么 integrate 进 VLA
4. **关注 world model 方向**：OmniVTA [64]、DreamTacVLA [95]、VTAM [99]，这是 next frontier
5. **关注 sensor-agnostic representation**：AnyTouch [9] 这条线，解决 scaling 问题

---

## 8. 最后一句

整个 field 一句话总结：**tactile 是 embodied AI 的 physical grounding，现在处于 vision 2018 年的位置——算法 ready，data 等待爆发，foundation model 即将出现**。

如果你 Karpathy 想做这方向，我觉得最 promising 的 angle 是：**scale up tactile data collection + VLA integration**。用 portable tactile gripper（UMI-FT 思路）collect 100k+ in-the-wild tactile manipulation trajectory，train 一个 tactile-augmented VLA，应该在 transparent/deformable manipulation 上能 beat 当前 SOTA。

你想聊哪个具体方向我都能继续展开。

---

# Multimodal Tactile Fusion: A Deep Dive for Building Intuition

这篇 survey 写得相当系统，给了一个从硬件 transduction 到 high-level semantic reasoning 的完整 pipeline 视角。我试着从你的角度——一个想 build embodied intelligence 系统、关心 representation learning 和 modality alignment 的人——来剖析这篇 paper，重点是让你对整个 field 形成 mental model。

---

## 1. High-Level Mental Model: 为什么 Tactile 是 Embodied AI 的 "Missing Modality"

人类感知世界的方式里，vision 给你 global scene context，language 给你 semantic abstraction，touch 给你 **contact-grounded physical reality**。前两者是 distal modality（remote sensing），后者是 proximal modality（必须物理接触才能产生信号）。这就是为什么 tactile 在 manipulation 里是 irreplaceable 的——vision 看到透明杯子但不知道它滑不滑，language 知道"glass"这个概念但不知道此时此刻 grasping force 够不够。

Survey 的核心 thesis 可以用一句话概括：**tactile 单独存在太 sparse，必须和 vision+language 融合才能 bridge physical interaction 和 semantic reasoning**。这就是 Multimodal Tactile Fusion 的 motivation。

Paper 里给出的 publication trend（Fig. 4）很说明问题：2015-2020 增长平缓，2020 之后突然加速。这个 timing 和 CLIP、ViT、diffusion model、VLA model 的爆发完全吻合——foundation model 让 sparse tactile data 能 leverage 预训练的 vision-language prior，这打开了整个 field。

---

## 2. Problem Formulation: 一个 Four-Stage Pipeline

Section II.A 给了一个很 clean 的 formalization，我觉得这是整篇 paper 最 worth internalizing 的部分。整个 multimodal tactile fusion 被抽象成 4 个 stage：

### Stage 1: Physical Transduction → Raw Observation

$$\boldsymbol{x}_t \in \mathbb{R}^{S_t \times T_t \times C_t}, \quad \boldsymbol{x}_v \in \mathbb{R}^{S_v \times T_v \times C_v}, \quad \boldsymbol{x}_l \in \mathcal{V}^L$$

变量解释：
- $\boldsymbol{x}_t$：tactile observation，$S_t$ = spatial configuration（sensor 的空间分辨率，比如 GelSight 的 160×120），$T_t$ = temporal length（interaction 序列长度），$C_t$ = channel（RGB 或者 force channel 数）
- $\boldsymbol{x}_v$：visual input，三个维度同义
- $\boldsymbol{x}_l$：language input，$\mathcal{V}$ = vocabulary（token 集合），$L$ = sequence length

这里有个 important insight：tactile 的 $S_t$ 通常远小于 vision 的 $S_v$（GelSight 160×120 vs RGB 224×224），但是 $T_t$ 通常更 critical，因为 tactile 是 contact-driven 的，dynamic information 比 static frame 更 informative。这就是为什么后续方法里 temporal modeling（Transformer、3D CNN）在 tactile 这边权重更高。

### Stage 2: Modality-Specific Encoding

$$z_m = \mathcal{E}_m(x_m; \theta_m), \quad m \in \{t, v, l\}$$

- $z_m$：modality $m$ 的 latent representation，通常 $z_t, z_v, z_l \in \mathbb{R}^d$（shared dimension $d$，比如 512 或 768）
- $\mathcal{E}_m$：encoder function，tactile/vision 用 ResNet/ViT，language 用 BERT/OpenCLIP
- $\theta_m$：encoder 参数

这里有个关键 design choice：**要不要 share encoder**？早期方法（ViTac [10]）share CNN backbone，因为 tactile 和 vision 都是 image-like。但 GelSight 的 image 和 natural image 分布差很远（gel 变形图 vs 自然场景），所以近期方法倾向 modality-specific encoder，但用预训练 vision encoder 初始化 tactile encoder（transfer learning）。

### Stage 3: Cross-Modal Fusion

$$z_{\text{joint}} = \Phi(\{z_m\}_{m \in \mathcal{M}}; \theta_\Phi)$$

- $z_{\text{joint}}$：fused joint representation
- $\mathcal{M} \subseteq \{t, v, l\}$：available modality set（T-V, T-L, T-V-L 三种组合）
- $\Phi$：fusion operator（concatenation / cross-attention / contrastive alignment）
- $\theta_\Phi$：fusion 参数

这是整个 field 最 active 的 research area。Survey 在 II.B.5 把 fusion strategy 分成几类：

**(a) Early fusion vs Late fusion**：
- Early fusion：在 input 或 shallow feature 层 merge，好处是 joint representation learning 强，但对 noise/missing modality 敏感
- Late fusion：各 modality 独立 encode 再 merge，保留 modality-specific structure，更灵活

**(b) Cross-attention 是目前主流**：因为 tactile patch 和 vision region 之间有 fine-grained spatial correspondence（你摸的地方就是你看到的地方），cross-attention 能学到这个 alignment。VHTformer [39]、ViTacFormer [85] 都是这条路。

**(c) Contrastive alignment**：把 (tactile, vision) 或 (tactile, language) 的 paired sample 拉近，unpaired 推远。这是 UniTouch [2]、Touch and Go [46]、TLV-Link [6] 的核心。本质上是 CLIP 思路在 tactile 上的迁移。

### Stage 4: Embodied Decoding

$$y = \mathcal{D}(z_{\text{joint}}; \theta_\mathcal{D})$$

- $y$：output，根据 task 不同含义不同
  - Perception task：classification label、attribute
  - Generation task：generated tactile image / text
  - Manipulation task：action $a_t \in \mathbb{R}^{\text{action dim}}$（end-effector pose、gripper force 等）
- $\mathcal{D}$：decoder
- $\theta_\mathcal{D}$：decoder 参数

**Intuition**：这四个 stage 其实对应了 embodied AI 的 perception-action loop。 tactile 的特殊性在于 Stage 1——它不是 passive sensing，而是 active interaction 的产物。你 grab 一个物体才有 tactile signal，没 grab 就没信号。这意味着 tactile data collection 本质上是个 exploration problem，不像 ImageNet 你可以 passively scrape。这就是为什么 tactile dataset 普遍小（survey Table I 显示最大的 Touch in the Wild [49] 也就 2.6M，远小于 ImageNet 的 1.4B），而且 sim-to-real gap 特别大。

---

## 3. Taxonomy: Three Pillars

Survey 把整个 field 分成三个 pillar，我觉得这个划分很合理：

### Pillar 1: Multimodal Datasets

按 modality 组合分四类：
- **T-V**（Tactile-Vision）：最早最成熟，从 2016 VT dataset [40] 开始，到 2025 Touch in the Wild [49] 已经 2.6M scale
- **T-L**（Tactile-Language）：很新，2024 才有 TCL3D [5]、TVL Dataset [3]，规模小（44k-50k）
- **T-V-L**（Tactile-Vision-Language）：2024 之后爆发，Touch100k [6] 到 100k scale，VTV150K [8] 到 150k
- **T-V-O**（Tactile-Vision-Other）：加入 audio/action/proprioception，ObjectFolder 系列 [56, 57, 58] 是代表，OmniViTac [64] 是 2026 最新

**关键 trend**：从 controlled lab → in-the-wild，从 static pair → temporal trajectory，从 bimodal → trimodal。这个 trend 和 VLA model 的兴起同步——VLA 需要的就是 (vision, language, action) 三元组，tactile 是这个三元的物理 grounding。

### Pillar 2: Multimodal Methods

三个 sub-paradigm，对应不同 downstream task：

**(1) Multimodal Perception and Recognition**（Section IV.A）
- Object recognition：VHTformer [39]、TVT-Transformer [54]
- Attribute/material recognition：UniTouch [2]、Surformer [12, 88]、ConViTac [91]
- Grasp success prediction：TFOS [41]、ACVTM [43]
- Cross-modal retrieval：UniTouch [2]、TLV-Link [6]

**(2) Cross-Modal Generation**（Section IV.B）
- Vision↔Tactile：VisGel [45]、BVT [82]、UVTS [48]
- Language↔Tactile：Tactile-VLM [3]、CLTP [5]、Octopi-1.5 [83]、RA-Touch [13]

**(3) Multimodal Interaction and Manipulation**（Section IV.C）
- Robot manipulation with perception：Visuotactile-RL [76]、ViTacFormer [85]、OmniVTLA [55]
- Language-guided manipulation：VTLA [7]、VTLG [92]、TLA [59]

### Pillar 3: Tactile Sensors

四类 hardware：
- **Wearable**：data glove [101]、OSMO glove [102]、FreeTacMan [60]
- **Handheld/Fingertip**：GelSight [30]、DIGIT [31]、LightTact [103]、TacThru [104]
- **Robotic skin/patch**：thin-film array [111]、hydrogel e-skin [112]、SuperTac [117]
- **Gripper-mounted**：TacUMI [109]、UMI-FT [110]

---

## 4. Method 深度剖析：三个 Paradigm 的技术演进

### 4.1 Perception & Recognition：从手工 feature 到 foundation model

**Early era (2016-2019)**：主要是 CNN-based T-V fusion
- VT [40]：visual feature + tactile sequence 直接 concat
- ViTac [10]：AlexNet (vision) + CNN (tactile) 双 backbone
- VisGel [45]：cross-modal prediction，bidirectional

**Middle era (2020-2023)**：Transformer 引入，contrastive learning 兴起
- VITO-Transformer [78]：attention-based fusion，global vision + local tactile
- Touch and Go [46]：human-collected in-the-wild，contrastive pretraining

**Foundation model era (2024-2026)**：CLIP-style alignment
- UniTouch [2]：把 tactile anchor 到 CLIP space，zero-shot retrieval
- Tactile-VLM [3]：tactile + vision + language trimodal alignment
- TLV-Link [6]：100k scale，multimodal contrastive

**Intuition**：整个演进路径就是 vision-language field 的复刻——CNN → Transformer → contrastive pretraining → foundation model。区别是 tactile data 太少，所以必须 leverage vision-language 的预训练 prior。UniTouch 的核心 trick 是：把 tactile encoder 输出 align 到 CLIP embedding space，这样 tactile 就免费获得了 CLIP 的全部 semantic knowledge。

### 4.2 Cross-Modal Generation：从 GAN 到 Diffusion

**Vision-Tactile Generation**：
- LVTR [69]：ensemble GAN，texture image → tactile vibration
- CTAV [45]：bidirectional，shared latent space
- BVT [82]：latent feature space flow model
- UniTouch [2]：通过 aligned embedding 调用预训练 generative model

**Language-Tactile Generation**：
- Tactile-VLM [3]：tactile → text caption
- CLTP [5]：fine-grained contact state decoding
- RA-Touch [13]：retrieval-augmented，用 external vision-language knowledge refine
- Octopi-1.5 [83]：real-time interactive，lightweight retrieval

**Intuition**：生成任务的核心 challenge 是 evaluation。MSE/PSNR 测 reconstruction fidelity，但不 measure perceptual realism；FID measure distribution gap，但 tactile 的 distribution 很难 define；LLM-judge 测 semantic，但 physical consistency 测不了。Survey 在 Section VI.A.2 列了 CVTP（Contrastive Visual-Tactile Pretraining Score）这种新 metric，但还很 preliminary。

### 4.3 Interaction & Manipulation：从 reactive control 到 VLA

**Early reactive control**：
- ACVTM [43]：closed-loop T-V refinement，regrasping
- Visuotactile-RL [76]：RL formulation，DrQv2 backbone

**Attention-based policy**：
- VTT [75]：tactile guide vision attention
- ViTacFormer [85]：Transformer for dexterous manipulation

**VLA era (2025-2026)**：
- OmniVTLA [55]：semantic-aligned tactile token + action generation backbone
- VTLA [7]：vision-tactile-language-action with preference learning
- TLA [59]：tactile-language-action for contact-rich manipulation
- VLA-Touch [96]：dual-level tactile feedback for VLA

**Intuition**：这是整个 field 最 exciting 的方向。VLA model（比如 RT-2、OpenVLA）目前是 vision+language→action，但缺少 tactile grounding。VTLA 这类工作的核心 insight 是：tactile 提供了 VLA 缺失的 **contact feedback**，让 policy 能 handle transparent object、deformable object、occluded object 这些 vision-fail 的 case。Survey 里 TEVG [11] 专门做了 transparent object grasping，证明 tactile 在 vision ambiguous 时是 critical 的。

---

## 5. Sensor Hardware 深度：为什么 Hardware-Software Co-Design 是 Key

Survey Fig. 3 比较了 7 个 representative sensor，我重点讲几个你可能在 VLA work 里会遇到的：

### GelSight [30]
- **原理**：camera + soft gel，gel 变形成 image
- **优势**：high-resolution（micron-level），fine surface reconstruction
- **劣势**：bulk，不能 wrap around curved surface
- **Paper**：https://www.mdpi.com/1424-8220/17/12/2762

### DIGIT [31]
- **原理**：compact fingertip，camera + LED + gel
- **优势**：small form factor，low-cost，easy mount on gripper
- **劣势**：resolution 比 GelSight 低
- **Paper**：https://ieeexplore.ieee.org/document/9035056

### Tac3D [32]
- **原理**：stereo camera，3D contact reconstruction
- **优势**：force + friction estimation，grasp analysis
- **Paper**：https://arxiv.org/abs/2202.06211

### GelSlim [34]
- **原理**：slim design，contact shape + force + slip
- **优势**：easy mount，multi-modal output
- **Paper**：https://ieeexplore.ieee.org/document/9812064

### OmniTact [35]
- **原理**：multiple camera，curved fingertip，omnidirectional
- **优势**：full coverage，no blind spot
- **Paper**：https://ieeexplore.ieee.org/document/9197055

**Intuition**：sensor 的 design 直接决定了 downstream method 能做什么。GelSight 输出 image-like signal，所以可以直接用 ViT/ResNet；DIGIT 输出也是 image-like，但因为 form factor 小，适合 gripper integration；Tac3D 输出 3D point cloud，需要不同的 encoder。这就是为什么 survey 反复强调 **sensor-agnostic representation**（AnyTouch [9]）是 open challenge——不同 sensor 输出 format 不同，cross-sensor transfer 很难。

---

## 6. Evaluation Metrics 详解

Survey Table III 给了完整的 metric 清单，我挑几个关键的解释：

### Perception Metrics

**Classification**：
$$\text{ACC} = \frac{TP + TN}{TP + TN + FP + FN}$$

- TP = True Positive，TN = True Negative，FP = False Positive，FN = False Negative
- 适用于 object recognition、grasp success prediction

**F1-Score**：
$$\text{F1} = \frac{2 \cdot \text{PREC} \cdot \text{REC}}{\text{PREC} + \text{REC}}$$

- PREC = TP / (TP + FP)，REC = TP / (TP + FN)
- 类别不平衡时（grasp success 多，failure 少）特别重要

**Retrieval**：
$$\text{R@k} = \frac{1}{Q} \sum_{q=1}^Q \mathbb{I}(\text{hit in top-}k)$$

- $Q$ = query 数量，$\mathbb{I}$ = indicator function
- 测 cross-modal alignment quality

**Cosine Similarity**：
$$\text{COS}(\mathbf{z}_1, \mathbf{z}_2) = \frac{\mathbf{z}_1 \cdot \mathbf{z}_2}{\|\mathbf{z}_1\| \|\mathbf{z}_2\|}$$

- $\mathbf{z}_1, \mathbf{z}_2$ = embedding vectors
- Zero-shot attribute recognition 的核心 metric

### Generation Metrics

**MSE**：
$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N \|\hat{\mathbf{x}}_i - \mathbf{x}_i\|_2^2$$

- $\hat{\mathbf{x}}_i$ = generated sample，$\mathbf{x}_i$ = ground truth，$N$ = sample 数
- 测 reconstruction fidelity

**PSNR**：
$$\text{PSNR} = 10 \log_{10}\left(\frac{L^2}{\text{MSE}}\right)$$

- $L$ = pixel max value（比如 255 for 8-bit）
- 高 PSNR = 低 reconstruction error

**SSIM**：
$$\text{SSIM} = \frac{(2\mu_x \mu_{\hat{x}} + C_1)(2\sigma_{x\hat{x}} + C_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + C_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + C_2)}$$

- $\mu_x, \mu_{\hat{x}}$ = mean of original and generated
- $\sigma_x^2, \sigma_{\hat{x}}^2$ = variance
- $\sigma_{x\hat{x}}$ = covariance
- $C_1, C_2$ = stability constants
- Measure structural similarity，比 MSE 更 perceptually correlated

**FID**：
$$\text{FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

- $\mu_r, \mu_g$ = mean of real and generated distribution
- $\Sigma_r, \Sigma_g$ = covariance matrix
- $\text{Tr}$ = trace
- Measure distribution-level realism

### Manipulation Metrics

**Success Rate**：
$$\text{SR} = \frac{N_{\text{success}}}{N_{\text{trials}}}$$

- 最直接，但 coarse

**Goal Convergence Rate**：
$$\text{GCR} = \frac{1}{m} \sum_{i=1}^m \mathbb{I}(|\hat{x}_i - x_i| < \epsilon_x, |\hat{y}_i - y_i| < \epsilon_y, |\hat{r}_{z,i} - r_{z,i}| < \epsilon_r)$$

- $\hat{x}_i, \hat{y}_i, \hat{r}_{z,i}$ = predicted pose (position + rotation)
- $x_i, y_i, r_{z,i}$ = target pose
- $\epsilon_x, \epsilon_y, \epsilon_r$ = tolerance
- 测 fine-grained control precision

**Human Normalized Score**：
$$\text{HNS} = \frac{\sum_{i=1}^N w_i s_i}{3 \sum_{i=1}^N w_i}$$

- $w_i$ = stage weight，$s_i$ = stage score
- 3 是 human baseline normalization constant
- Stage-aware，能 capture intermediate progress

**Intuition**：manipulation metric 的 evolution 反映了 field 的成熟度。早期只有 SR（binary），现在有 GCR（continuous tolerance）、HNS（stage-aware）。但 survey 在 VI.A.4 诚实地指出：**没有 unified benchmark**，每篇 paper 用不同 dataset、不同 task、不同 metric，没法 cross-study comparison。这是整个 field 最大的 bottleneck 之一。

---

## 7. Challenges & Future Directions

Survey Section VII 列了 4 个 challenge 和 5 个 future direction，我觉得最 critical 的是：

### Challenge 1: Data Scalability Gap
最大问题。最大的 tactile dataset 是 Touch in the Wild 2.6M [49]，但 ImageNet 是 1.4B，LAION 是 5B。差 3 个数量级。Tactile data 必须物理交互才能产生，没法 passively scrape。这导致 foundation model 在 tactile 上 pretrain 很难。

**Possible solution**：survey 提到 diffusion-based tactile hallucination。但 tactile 的物理约束比 image 强很多——你必须 satisfy contact mechanics，不能任意 generate。这是开放问题。

### Challenge 2: Modality Misalignment
Tactile 是 sparse + local + temporal，vision 是 dense + global + spatial，language 是 abstract + semantic。三者的 spatiotemporal granularity 完全不同。Cross-attention 能缓解但不能解决。

### Challenge 3: Hardware-Software Integration
没有 standard interface。GelSight 输出 RGB image，DIGIT 输出 RGB image，Tac3D 输出 3D point cloud，Force sensor 输出 scalar。一个 unified tactile tokenization（类似 image patch tokenization）是 open problem。

### Challenge 4: Benchmark Deficiency
没有 ImageNet-scale 的 unified benchmark。每篇 paper 自己 define task 和 metric。

### Future Direction 我最看好的：

**(1) VLA + Tactile**：VTLA [7]、OmniVTLA [55]、VLA-Touch [96] 这条线。当前 VLA model（RT-2、OpenVLA、π0）都缺 tactile grounding，加上 tactile 应该能解决 transparent/deformable/occluded manipulation 的 long-tail。

**(2) Tactile World Model**：OmniVTA [64]、VTAM [99]、DreamTacVLA [95]。学一个能 predict future tactile state 的 world model，让 policy 能 do model-based planning。这和 LeCun 的 JEPA 思路、Ha & Schmidhuber 的 world model 思路一致。

**(3) Bio-inspired Sensor**：SuperTac [117]、self-powered sensor [115, 116]。Neuromorphic + energy-efficient + multimodal，可能突破 current sensor 的 durability 和 power bottleneck。

**(4) Simulation Synergy**：高保真 tactile simulation（FEM-based contact、diffusion-based tactile rendering）+ sim-to-real transfer。这是 scaling data 的唯一可行路径。

---

## 8. 一些 Critical的思考

读这篇 survey 我有几个 takeaways 想分享：

**1. Tactile field 正在重走 vision-language 的路，但晚 3-5 年**
CLIP → UniTouch，ViT → ViTacFormer，VLM → Tactile-VLM，VLA → VTLA。每个 milestone 都对应。这意味着你可以用 vision-language 的经验预测 tactile 的下一步：tactile-specific foundation model、tactile instruction tuning、tactile RLHF 等。

**2. Bottleneck 不是算法，是 data**
所有 method 的 ceiling 都被 data 限制。UniTouch 用 100k scale 就能做到 zero-shot retrieval，如果 scale 到 10M，应该能有更强 emergent capability。data collection 的 cost 是关键 bottleneck——OSMO glove [102]、FreeTacMan [60]、UMI-FT [110] 这些 portable system 是 trying to solve this。

**3. Tactile 是 VLA 的 killer feature**
当前 VLA model 在 transparent cup、folded cloth、articulated object 上 fail，本质是 vision-only 的 limitation。加上 tactile 应该能解决这一类 long-tail。VTLA [7] 已经 demonstrate 这个方向。

**4. Sensor agnosticism 是 unsolved**
AnyTouch [9] 尝试 unified representation across sensors，但还 early。这个问题不解决，每个 sensor 都要 retrain model，scaling 不起来。类比 NLP 的 tokenizer，tactile 需要 unified tokenization。

---

## 9. Key References for Deep Dive

如果你要深入这个 field，我推荐按这个顺序读：

**Foundation Papers**：
- GelSight: https://www.mdpi.com/1424-8220/17/12/2762
- DIGIT: https://ieeexplore.ieee.org/document/9035056
- CLIP: https://arxiv.org/abs/2103.00020
- ViT: https://arxiv.org/abs/2010.11929

**Multimodal Tactile Core**:
- UniTouch (CVPR 2024): https://arxiv.org/abs/2401.12186
- Touch and Go: https://arxiv.org/abs/2211.12498
- Tactile-VLM: https://arxiv.org/abs/2402.13232
- TLV-Link / Touch100k: https://arxiv.org/abs/2505.xxxxx (2025)
- AnyTouch: https://arxiv.org/abs/2502.12191

**VLA + Tactile**:
- VTLA: https://arxiv.org/abs/2505.09577
- OmniVTLA: https://arxiv.org/abs/2508.08706
- TLA: https://arxiv.org/abs/2503.08548
- VLA-Touch: https://arxiv.org/abs/2507.17294

**Dataset**:
- ObjectFolder: https://objectfolder.org/
- Touch in the Wild: https://arxiv.org/abs/2507.15062
- VisGel: https://arxiv.org/abs/1903.07149

**World Model**:
- OmniVTA: https://arxiv.org/abs/2601.xxxxx (2026)
- DreamTacVLA: https://arxiv.org/abs/2512.23864

---

## 10. Final Intuition

整个 field 我用一句话总结：**tactile 是 embodied AI 的 physical grounding，目前处于 vision 2018 年的位置——算法 ready，data 等待爆发，foundation model 即将出现**。

如果你要做这个方向，我会建议：
1. 先 reproduce UniTouch [2] 的 contrastive alignment，理解 tactile-vision embedding 怎么 align
2. 再读 VTLA [7] 看 tactile 怎么 integrate 进 VLA
3. 关注 tactile world model（OmniVTA、DreamTacVLA），这可能是下一个 breakthrough
4. 自己搭一个 GelSight/DIGIT gripper，collect 一些 in-the-wild data，感受一下 tactile signal 的特性——你会立刻理解为什么 vision-only 不够

Survey 本身的 limitation 是：对每个 method 的 technical detail 不够深，更像是 map 而不是 guide。但作为 entry point 非常好，reference list 完整，taxonomy 清晰。你之后做 research 时，这篇 survey 可以当 cheatsheet 用——快速定位相关工作。

Happy to discuss more on any specific method or direction.
