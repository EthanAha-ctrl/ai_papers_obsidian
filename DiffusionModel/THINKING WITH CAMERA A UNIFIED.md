---
source_pdf: THINKING WITH CAMERA A UNIFIED.pdf
paper_sha256: d9db80416499ebb8ff4a7462f755a81e0256155a2d7b29b4b0d3ef7fd2121669
processed_at: '2026-08-12T15:36:36-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Puffin 用人话说

---

## 这 paper 到底在干嘛

一句话：**教 AI 懂相机，还能用相机参数来生成图片。**

你给 AI 一张照片，它能告诉你这照片是咋拍的——镜头啥角度、俯仰多少度、视野多宽。反过来，你告诉 AI "给我生成一张从下往上看的广角厨房图"，它真能按这个 camera 参数生成，不会瞎来。

以前这俩事儿是两拨人各干各的。研究 camera calibration 的人搞一套网络，研究 controllable generation 的人搞另一套。Puffin 说：**干嘛要分开，合在一起反而更好**。

---

## 为啥以前没人干成

核心问题：**camera 参数是一堆数字，LLM 不懂数字。**

你跟 GPT-4o 说 "roll = 0.5107 radians"，它根本不知道这是啥意思。LLM 的强项是语言，不是 precise numerical regression。你硬塞给它一个 regression task，它会懵。

而且传统 VLM 的 vision encoder 是被训练来认东西的——这是猫还是狗、桌上有几个杯子。它把图片压成 high-level semantic features，**几何信息早就丢光了**。你问它 "这张图地平线往哪边斜"，它答不上来，因为它压根没保留这种 information。

更尴尬的是，paper 做了个 ablation：直接 fine-tune InternVL3 做 camera calibration，结果 **比纯 vision-only 的网络还差**。LLM 的 semantic prior 在几何任务上不仅没帮忙，反而是拖后腿的。

---

## Puffin 的核心 trick

**把数字翻译成人话。**

Camera 参数难懂？那就先翻译成摄影师的黑话：

- roll = -30° → "大角度逆时针荷兰角"
- pitch = +25° → "大角度仰拍"  
- FoV = 25° → "特写镜头"

这些词 LLM 天天见，在 photography 论坛、图片标注、电影描述里到处都是。LLM 对这些词有 prior knowledge，它知道 "仰拍" 意味着天空占画面很大比例，"广角" 意味着近处物体被夸张放大。

**这就是 "thinking with camera" 的本质——用摄影师的语言来做几何推理。**

LLM 先看到图，用自然语言描述 "我看到大面积天空，树和屋顶在画面下方"，然后推理出 "这是大角度仰拍"，最后再 regress 出具体数值 pitch = 0.5550 radians。

这不是 skip connection 或者什么 fancy architecture，而是 **modality translation**——把 continuous numerical manifold 映射到 discrete semantic token space，让 LLM 在它擅长的地盘干活。

---

## 架构怎么搭的

四个零件拼一起：

1. **Vision encoder**: 不用普通的 CLIP，用一个专门 distill 过的 C-RADIO，同时学了 semantic（认东西）和 geometric（懂结构）的特征
2. **LLM**: Qwen2.5-1.5B，负责 reasoning 和 output token 生成
3. **Diffusion model**: SD3，负责画图
4. **Connector**: 一个小 transformer，把 LLM 的 hidden states 翻译成 diffusion model 能听懂的 conditioning signal

关键设计是 **camera 参数有两条路进去**：

- **离散路**: 数值变成 text token，跟描述文字混在一起，走 LLM
- **连续路**: 数值变成 pixel-wise 的 camera map（叫 Perspective Field），每个 pixel 都标注了 "这个点的 local up 方向" 和 "这个点的光线跟重力方向夹角"，走 diffusion model 的 conditioning

为啥要两条路？离散 token 给的是 global 信息——"这是个广角仰拍"。但 generation 需要细节，光知道 "广角" 不够，你得知道 image 左上角和右下角的 local geometry 有啥不同。Camera map 就是给这种 pixel-level 精度用的。

---

## 数据怎么来的

这是个脏活累活。

1. 收了 20 万张 **360° 全景图**，从 Google Street View、各种 HDR 网站、学术数据集里扒
2. 用 pinhole camera model 从全景图里 **裁出** 400 万张普通透视图，每张都有精确的 camera 参数
3. roll/pitch/FoV 在各自范围内均匀采样，确保各种 camera 配置都覆盖到
4. 用 Qwen2.5-VL 给每张图写 caption，再蒸馏成精简版
5. 最费劲的：**写 thinking caption**——把 camera 参数先翻译成摄影术语，再让 32B 的 VLM 把术语跟视觉线索联系起来

比如 camera 参数说 pitch = +25°，先翻译成 "大角度仰拍"，然后让 VLM 写："画面上方大面积天空，云朵分布广阔，地面元素被压缩在画面底部"。

Cross-view 数据更复杂：还要加 yaw 角，初始 view + 目标 view 配对，photographic guidance 还要请摄影专家定义 4 个审美标准，让 VLM 打分...

---

## 训练怎么训的

四阶段，很有 curriculum 的味道：

**Stage 1**: 只训 connector 和 projector，其他全冻住。先把 vision encoder、LLM、diffusion 三个零件 **对齐**，让它们知道彼此在说啥。

**Stage 2**: 全部解冻，一起训。但 vision encoder 的梯度缩小到 0.1——怕 semantic training 信号把 geometric features 冲坏。

**Stage 3**: 加入 thinking captions，让模型学会用摄影术语做 reasoning。这是 "thinking with camera" 真正注入的地方。

**Stage 4**: 专门训 cross-view 任务——spatial imagination、world exploration、photographic guidance。

64 张 A100 跑 4 天。

---

## 效果怎么样

**Understanding**: 在自己建的 Puffin-Und benchmark 上全面碾压。最亮眼的提升在 pitch 和 FoV——这俩参数需要 holistic spatial understanding，正好是 LMM world knowledge 的主场。Roll 提升不大，因为 roll 靠 local lines 就能估，vision-only 方法已经很强。

**Generation**: 对比 GPT-4o、Qwen-Image、Nano Banana 这些大模型，camera control 精度甩开一大截。GPT-4o 生成图片时基本忽略你给的 camera 参数，该水平还是水平，该正面还是正面。Puffin 能真正按照指定角度和焦距来画。

**最反直觉的发现**: 在 understanding 里 roll 最容易估，但在 generation 里 roll 最难生成。

为啥？因为训练数据里的图片基本都是水平的——摄影师不管专业不专业都倾向拍平的，人对歪了很敏感。所以 generation 模型见过的 roll variants 极少。而且强 Dutch angle 会打破物理直觉——海面跑到 horizon 上面去了，这种 "反物理" 的画面模型很难生成。

---

## 一个有意思的 ablation

把 GeoCalib（一个 29M params 的专门做 calibration 的小模型）拿过来，在 Puffin 的 400 万数据上重新训，**结果变差了**。

小模型消化不了大数据。29M 参数的 capacity 不够，在 4M 数据上 underfit，只能 fit 部分 sub-distribution，其他的都顾不上。而 Puffin 的 4.4B 大模型 + LLM backbone 恰好匹配 4M 数据规模。

这说明 **数据规模和模型容量要匹配**，不是数据越多越好，得看模型能不能吃下。

---

## Joint training 的协同效应

这是 unified model 的核心卖点，也是 paper 最想证明的：

**Understanding 和 generation 互相帮忙。**

单独训 understanding → 加上 generation supervision 后 roll error 从 0.55° 降到 0.47°。为啥？因为 diffusion loss 给了 low-level appearance 的辅助监督，逼着 LLM 学更精细的几何感知。

单独训 generation → 加上 understanding supervision 后 FoV simulation 明显改善。因为 understanding 任务让 LLM 学到的 camera representation 通过 connector 更忠实地传给 diffusion model。

之前其他 unified multimodal model（像 Janus、Show-O）没观察到这种协同效应，因为它们的 understanding 任务主要是 recognition（认猫狗、OCR），跟 generation 的 domain gap 太大。Puffin 的 understanding 任务本身就是几何相关的，所以 synergy 才显现。

---

## 这 paper 的 takeaway

1. **Modality translation 比 architecture hacking 重要**。与其设计更复杂的网络，不如把 camera 参数翻译成 LLM 懂的语言
2. **Dual representation 有必要**。Global camera tokens + pixel-wise camera map，缺一不可
3. **Unified training 有 synergy，但前提是任务相关**。Understanding 和 generation 必须共享底层 representation（这里是 camera geometry），才能互相 reinforce
4. **数据和模型要匹配**。大数据需要大模型，小模型喂大数据会 underfit
5. **摄影术语是个好 bridge**。它既保留了物理意义，又 linguistically tractable，这个 abstraction level 刚刚好

本质上这篇 paper 做的事情是：**把 3D 几何的 prior 嵌进 LMM 的 semantic space**，让模型可以用语言推理的方式处理几何问题。Photographic terms 这层 abstraction 是整个设计的 elegant 之处。

---

# Puffin: Thinking with Camera — 深度技术解析

Andrej，这篇 paper 我从直觉出发给你拆解。Puffin 的核心 contribution 在于把 **camera geometry** 作为一个 first-class modality 嵌入到 unified multimodal model 里，同时处理 **understanding**（image → camera params）和 **generation**（text + camera params → image）两个任务。这种 unification 的核心 insight 是把抽象的 numerical camera parameters "翻译" 成 LMM 已经熟悉的 **professional photographic terminology**，让 reasoning 路径变得 linguistically tractable。

Project page: https://kangliao929.github.io/projects/puffin

---

## 1. Motivation & Core Intuition

### 1.1 为什么 camera-centric 任务难以塞进 LMM

传统 LMMs (LLaVA, Qwen-VL, InternVL) 的 vision encoder 主要被训练用于 recognition 任务，输出的是 **condensed high-level semantics**——丢掉了几何细节。同时 LLM backbone 里几乎没有 spatial perception 的 prior knowledge。Table 5 的 ablation 给出了一个非常 striking 的数据：直接 fine-tune InternVL3 在 camera understanding 上 roll error = 0.91°，比纯 vision-only 的 C-RADIO encoder (0.55°) 还差。这印证了一个直觉——**naively 把几何任务塞进 semantic-heavy 的 LMM 会破坏 geometric fidelity**。

参考 InternVL3: https://arxiv.org/abs/2504.10479  
参考 LLaVA: https://arxiv.org/abs/2304.08485

### 1.2 "Camera as Language" 的核心 idea

Camera parameters 本质上是 numerical vectors——`roll = 0.5107`, `pitch = 0.5550`, `FoV = 0.7558`（radians）。LLM 对这种连续数值的 precise estimation 能力很弱。但是 LLM 对语言 token 的 reasoning 极强。Puffin 的 insight 是引入一个 **parameter-to-term mapping** $f: \mathbf{p} \mapsto \mathbf{t}$（见 Table A1）：

| Parameter | Range | Photographic Term |
|-----------|-------|-------------------|
| Roll | $[-45°, -20°)$ | Large counterclockwise Dutch angle |
| Roll | $[-5°, 5°]$ | Near level shot |
| Roll | $(20°, 45°]$ | Large clockwise Dutch angle |
| Pitch | $[-45°, -20°)$ | Large tilt-down |
| Pitch | $(20°, 45°]$ | Large tilt-up |
| FoV | $[20°, 35°)$ | Close-up |
| FoV | $[90°, 105°]$ | Ultra wide-angle |

这样做的本质是 **discretize continuous camera manifold into a semantically meaningful vocabulary**，让 LMM 可以在它熟悉的 token 空间里做 chain-of-thought reasoning，然后再 regress 具体数值。

---

## 2. Architecture Deep Dive

### 2.1 整体框架（Figure 2 解析）

Puffin 由四个核心模块组成：

1. **Geometry-aligned vision encoder**: C-RADIOv3-H (Heinrich et al., 2025)，distilled from CLIP + DINO + SAM teachers，保留了 geometric fidelity
2. **LLM backbone**: Qwen2.5-1.5B-Instruct
3. **Diffusion model**: SD3-Medium (rectified flow transformer)
4. **Connector module**: 6 transformer layers + 64 learnable queries，把 LLM hidden states 投影成 diffusion 的 conditioning signals

C-RADIO: https://nvlabs.github.io/RADIO/  
Qwen2.5: https://qwenlm.github.io/blog/qwen2.5/  
SD3: https://arxiv.org/abs/2403.03206

### 2.2 双路径 conditioning

**Understanding 路径**: 
$$\text{Image} \xrightarrow{\text{Vision Encoder}} \text{Visual Tokens} \xrightarrow{\text{LLM}} \text{Text + Camera Tokens}$$

**Generation 路径**:
$$\text{Text + Camera Tokens + Camera Map} \xrightarrow{\text{LLM}} \text{Hidden States} \xrightarrow{\text{Connector}} \text{Diffusion Conditioning} \xrightarrow{\text{SD3}} \text{Image}$$

关键设计：**camera map 作为 continuous latent**，不只是用 discrete camera tokens。这个 dual representation 是 paper 的一个 subtle 但重要的设计 choice——discrete tokens 给 global 信息，camera map 给 pixel-wise 几何细节。

### 2.3 Camera Map 公式详解

Paper 用 **Perspective Field** (Jin et al., 2023) 作为 camera map，公式 (1)：

$$\mathbf{u}_{\mathbf{x}} = \lim_{c \to 0} \frac{\mathcal{P}(\mathbf{X} - c\mathbf{g}) - \mathcal{P}(\mathbf{X})}{\|(\mathcal{P}(\mathbf{X} - c\mathbf{g}) - \mathcal{P}(\mathbf{X}))\|_2}$$

$$\varphi_{\mathbf{x}} = \arcsin\left(\frac{\mathbf{R} \cdot \mathbf{g}}{\|\mathbf{R}\|_2}\right)$$

变量解释：
- $\mathbf{x}$: image plane 上的某个 pixel
- $\mathbf{X}$: 对应的 3D world point
- $\mathcal{P}(\cdot)$: pinhole camera projection function $\mathcal{P}(\mathbf{X}) = \mathbf{x}$
- $\mathbf{g}$: gravity direction (3D vector，通常为 $(0, -1, 0)$ 在 camera coordinate 下旋转)
- $c$: infinitesimal scalar for numerical differentiation
- $\mathbf{u}_{\mathbf{x}}$: **up-vector** at pixel $\mathbf{x}$ — 2D unit vector 指向 image 中 "重力反方向" 的投影
- $\mathbf{R}$: light ray from camera center to $\mathbf{X}$
- $\varphi_{\mathbf{x}}$: **latitude angle** — ray $\mathbf{R}$ 与 gravity direction $\mathbf{g}$ 的夹角

**Intuition**: $\mathbf{u}_{\mathbf{x}}$ 告诉你 "这个 pixel 处的局部 vertical 方向"（垂直于地面在 image 中投影的方向），$\varphi_{\mathbf{x}}$ 告诉你 "这个 pixel 对应的 ray 有多倾斜"。这两个量 dense 化了 camera 参数——每个 pixel 都有自己的 local geometry。Camera map 是 3 channels（up-vector 的 x, y 分量 + latitude angle），所以可以直接复用 VAE encoder。

Perspective Fields paper: https://arxiv.org/abs/2312.03315

---

## 3. Puffin-4M Dataset 构建流水线

### 3.1 数据规模与来源

- **200K 高质量 panoramic images**，分辨率从 4K 到 10K
- 渲染出 **4M vision-language-camera triplets**
- 图像分辨率：统一 512×512
- 来源：Stanford2D3D、Google Street View (12 cities across Asia/Europe/North America)、各种 HDR 网站 (Poly Haven, HDRMaps, AmbientCG, BlenderKit)、Flickr360 等

### 3.2 透视图像生成（关键参数范围）

$$\text{Roll} \in [-45°, 45°], \quad \text{Pitch} \in [-45°, 45°], \quad \text{Vertical FoV} \in [20°, 105°]$$

Cross-view 扩展时加入 $\text{Yaw} \in [0°, 360°)$。每个 panorama 的 crop 数量自适应地由原始分辨率决定。

### 3.3 Captioning 两步法

1. **Scene caption**: Qwen2.5-VL-7B-Instruct 生成详细描述 → Qwen2.5-7B-Instruct 蒸馏成 1-2 句 vivid 描述
2. **Spatial reasoning caption**: 先 parameter → photographic term mapping，再用 Qwen2.5-VL-32B-Instruct 把 photographic term 作为 anchor 去 retrieve 相关 visual concepts（比如 "large tilt-up" → "expansive sky with clouds" 或 "pendant lights and uncluttered ceilings"）

### 3.4 Photographic Guidance 数据构造

这部分很有意思——他们咨询了摄影专家，定义了 4 个 aesthetic criteria：
1. **Viewpoint creativity**
2. **Subject emphasis**
3. **Compositional balance**
4. **Spatial harmonization**

流程：初始 view 用 random pitch $\in [-20°, 20°]$，再 sample N 个邻居 views（perturb pitch + yaw in same range），用 Qwen2.5-VL-32B-Instruct 给 4 个 criteria 打分，最高分 view 与初始 view 的 (pitch, yaw) offset 作为 label。

参考 Qwen2.5-VL: https://arxiv.org/abs/2502.13923

---

## 4. Training Recipe 细节

### 4.1 四阶段训练（Table 1）

| Stage | Steps | LR | Batch | Vision Encoder | LLM | Diffusion |
|-------|-------|-----|-------|----------------|-----|-----------|
| I: Alignment | 10K | 1e-4 | 1024 | Frozen | Frozen | Frozen |
| II: SFT | 30K | 2e-5 | 1024 | Trainable (grad scale 0.1) | Trainable | Trainable |
| III: SFT w/ Thinking | 60K | 1e-5 | 512 | Trainable | Trainable | Trainable |
| IV: Instruction Tuning | 20K | 5e-6 | 256 | Frozen | Trainable | Trainable |

总训练时间：64×A100 (80GB) 4 天。Optimizer: AdamW, betas (0.9, 0.95), weight decay 0.05, cosine schedule.

**Stage I** 只训练 MLP projector + connector，建立 vision encoder ↔ LLM ↔ diffusion 的对齐。**Stage II** 全量 fine-tune，但 vision encoder gradient scaling 0.1 防止 geometric features 被 semantic training 信号冲毁。**Stage III** 是关键——加入 thinking captions。**Stage IV** 专门处理 cross-view tasks。

### 4.2 数据采样比例的演化

- Stage I/II: 50% Gen. + 50% Und.
- Stage III: 33% Text→Text (reasoning) + 33% Gen. + 33% Und.
- Stage IV: 40% Cross-view Und. + 40% Cross-view Gen. + 5% Photography Und. + (15% 未明确)

这个 curriculum 的 insight：先学 base tasks，再注入 reasoning，最后扩展到 cross-view generalization。

### 4.3 Loss Functions

- **Understanding**: cross-entropy loss on text + camera tokens
- **Generation**: diffusion loss (rectified flow objective in SD3)
- **Joint training**: 两个 loss 同时反传到 LLM，diffusion loss 通过 connector 传回，相当于给 LLM 一个 low-level appearance 的辅助监督

---

## 5. 实验结果深度分析

### 5.1 Camera Understanding (Table 3)

在 Puffin-Und benchmark（1000 张 diverse images）上的核心数据：

| Method | Roll err ↓ | Pitch err ↓ | FoV err ↓ |
|--------|------------|-------------|-----------|
| DeepCalib | 1.90° | 3.71° | 7.43° |
| ParamNet | 2.11° | 3.40° | 6.21° |
| UVP | 0.51° | 4.59° | 10.92° |
| GeoCalib (SOTA) | 0.36° | 1.94° | 4.46° |
| **Puffin** | **0.32°** | **1.08°** | **2.42°** |

最 striking 的提升在 **Pitch 和 FoV**——这是依赖 contextual prior 的参数，而 LMM 的 world knowledge 恰好提供了这种 prior。Roll 提升相对小，因为 roll 主要靠 local geometric cues（vanishing lines），vision-only 方法已经做得很好。

### 5.2 为什么 Pitch 和 FoV 难

Paper 给出了很 intuitive 的解释：
- **Roll**: 直接对应 image 中的 vanishing lines 倾斜度，low-level feature 就能 capture
- **Pitch**: 需要理解 "天空占多少比例"、"地面/天花板的位置"、"foreground-background ratio"——这些是 holistic spatial composition
- **FoV**: 需要感知 object scale vs. scene scale、depth distribution——纯 local feature 不够

LMM 的 world knowledge 能 implicitly encode "天空多 = tilt-up"、"广角 = 物体夸张透视" 这类 prior，这是 vision-only CNN 无法捕捉的。

### 5.3 Camera-Controllable Generation (Table 4)

| Method | Up Vec err ↓ | Latitude err ↓ | Gravity err ↓ | FID ↓ |
|--------|--------------|----------------|---------------|-------|
| GPT-4o (degrees) | 24.11° | 15.87° | 28.08° | 95.92 |
| GPT-4o (terms) | 24.07° | 14.67° | 27.19° | 94.43 |
| Qwen-Image | 23.80° | 15.76° | 27.75° | 83.31 |
| Nano Banana | 24.08° | 16.66° | 28.78° | 91.66 |
| PreciseCam | 18.66° | 12.49° | 18.39° | 90.91 |
| **Puffin** | **11.94°** | **6.34°** | **6.79°** | **69.46** |

Puffin 在所有 metrics 上大幅领先，FID 也最低（69.46），说明不仅仅是 camera control 准确，visual quality 也最好。

### 5.4 一个反直觉的观察（Section 5.3 Discussion）

这是 paper 里我个人觉得最 interesting 的洞察之一：

**在 understanding 中，roll 容易，pitch/FoV 难；在 generation 中，roll 反而最难。**

Figure 7 的 scatter plot 清晰显示——所有 baseline 在 roll 维度上完全无法 align 到 ground truth（predicted values 几乎是 constant，对角线 y=x 完全 broken）。

Paper 给出两个解释：

1. **训练数据美学偏斜**：摄影师（无论专业还是业余）都倾向 near-level shots，因为人类对水平扰动很敏感 (He et al., 2013; Howard & Templeton, 1966)。所以现有 generation 模型的训练数据里 roll variants 极少。

2. **Roll 改变重力感知**：强 Dutch angle 会让海面出现在 horizon line 之上，造成 inverted spatial illusion。这种违背 physics 的 case 对 generation model 极其困难，而 pitch/FoV 只改变 viewing scope，不破坏 physical law。

参考 He et al. content-aware rotation: https://arxiv.org/abs/1311.2221

### 5.5 Ablation: Joint Training 的协同效应（Figure 9）

这是 unified model 的核心卖点——**understanding 和 generation 互相促进**：

- Understanding 单独训练 → 加入 generation supervision 后 roll err 从 0.55° → 0.47°
- Generation 单独训练 → 加入 understanding supervision 后 FoV simulation 明显改善

机制解释：diffusion loss 提供了 low-level appearance 的 auxiliary supervision，帮助 LMM 学习更精细的 geometric perception；反过来，understanding 任务让 LMM 学到的 camera geometry representation 通过 connector 更忠实地传给 diffusion model。

### 5.6 Model vs. Data（Table A4 - 一个反直觉的发现）

把 GeoCalib (29M params) 在 4M 数据上 retrain，**性能反而下降**！

| Method | Roll err | Pitch err | FoV err |
|--------|----------|-----------|---------|
| GeoCalib (原 40K) | 0.92° | 2.18° | 5.04° |
| GeoCalib (4M retrain) | 1.12° | 2.54° | 5.47° |
| Puffin (4M) | **0.41°** | **0.74°** | **1.21°** |

Paper 给出两个解释：
1. **Model capacity bottleneck**: 29M params 无法 fully model 4M 数据的整个 distribution，会 underfit，只能 fit 部分 sub-distribution
2. **Scale matching**: 小 model 适合小数据，大 model 需要大数据。LMM-based unified model 的 4.4B params 恰好匹配 4M 数据规模

---

## 6. Cross-View Applications

### 6.1 Spatial Imagination

给定初始 view + target camera params（roll, pitch, yaw, FoV），让模型想象 target view 的 **text description**。这相当于 "mental rotation"——用语言 bridge 不同 viewpoint。

### 6.2 World Exploration

给定初始 view + target camera params（含 yaw），生成 target view 的 image。关键技术：
- Initial view 通过 VAE encode，作为 condition
- Text description 50% 概率 drop，支持 text-free 和 text-conditioned 两种模式
- KV cache 机制加速 cross-view generation
- 3D reconstruction (VGGT) 验证 spatial consistency

VGGT: https://arxiv.org/abs/2503.17951

### 6.3 Photographic Guidance

给定初始 view，建议 (pitch, yaw) offset 来获得更高 aesthetics。这是把 camera understanding + aesthetic reasoning 结合的有趣应用——相当于一个 "AI 摄影教练"。

---

## 7. Limitations & Future Directions

1. **Fixed resolution 512×512**: 导致 evaluation 时 center crop + resize，对 LaMAR 这种非方形 aspect ratio 数据集 degrade 性能
2. **Evaluation metric**: 用 offline GeoCalib 评估 generated images 的 camera accuracy，但 generated images 的 subtle spatial differences 可能 calibration error 也判断不准
3. **未纳入 radial distortion**: 当前只假设理想 pinhole model，未来可以加 distortion parameter $k$
4. **Video extension**: 自然延伸是 camera-centric video understanding/generation

---

## 8. 我的 Intuition 总结

Puffin 的核心 contribution 在我看来有三层：

1. **Representation 层**: Camera map (Perspective Field) 作为 continuous latent + discrete camera tokens 作为 global descriptor，dual representation 既保留 pixel-wise 几何细节又给 global 信号
2. **Semantic bridge 层**: Professional photographic terms 作为中间 representation，把 numerical camera manifold 映射到 LMM 熟悉的 token space——这是 "thinking with camera" 的核心机制
3. **Unified training 层**: Joint optimization 让 understanding 和 generation 的 supervision 互相 reinforce，避免了 frozen module 造成的 representational bottleneck

这个工作本质上是 **把 3D vision 的几何 prior 嵌入到 LMM 的 semantic space**，让 model 可以用语言 reasoning 的方式处理几何问题。Photographic terms 这层 abstraction 是非常 elegant 的设计——既保留了 camera parameters 的物理意义，又让它变得 linguistically tractable。

---

## 相关参考链接

- **Puffin project**: https://kangliao929.github.io/projects/puffin
- **Perspective Fields (Jin et al., CVPR 2023)**: https://jinlinjian.github.io/perspectiverep/ / https://arxiv.org/abs/2312.03315
- **GeoCalib (Veicht et al., ECCV 2024)**: https://arxiv.org/abs/2409.06704
- **C-RADIO vision encoder**: https://nvlabs.github.io/RADIO/
- **Qwen2.5 technical report**: https://arxiv.org/abs/2412.15115
- **SD3 (Esser et al., ICML 2024)**: https://arxiv.org/abs/2403.03206
- **Janus (Wu et al., unified multimodal)**: https://arxiv.org/abs/2410.13848
- **Show-O (Xie et al., unified multimodal)**: https://arxiv.org/abs/2408.12528
- **DeepCalib (Bogdan et al.)**: https://arxiv.org/abs/1812.02486
- **VGGT (3D reconstruction)**: https://arxiv.org/abs/2503.17951
- **Cameras as Rays (Zhang et al.)**: https://arxiv.org/abs/2402.14817
- **Kang Liao's calibration survey**: https://arxiv.org/abs/2303.10559

如果你想 dive deeper into 任何一部分（比如 Perspective Field 的具体推导、SD3 的 rectified flow objective、connector 的 transformer 架构细节，或者 photographic guidance 的 aesthetic scoring 机制），告诉我，我可以展开讲。
