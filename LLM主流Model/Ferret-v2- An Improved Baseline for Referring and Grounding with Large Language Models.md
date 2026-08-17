---
source_pdf: Ferret-v2- An Improved Baseline for Referring and Grounding with Large
  Language Models.pdf
paper_sha256: 9ff55409ebe9f11f3203bd0c70be12bdd41b00ae7d09e700abcdbc4b722c6e7c
processed_at: '2026-08-04T08:14:48-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我换个说法，用大白话给你捋一遍。

---

## 这论文到底在干嘛

Ferret 是 Apple 之前做的一个多模态模型，能看图、能聊天、还能你画个框它告诉你框里是啥，或者你说个东西它在图上框出来。听着挺酷，但有个硬伤：**它看不清细节**。

原因很简单——CLIP-ViT 预训练就锁在 336×336 的分辨率，你喂一张 4K 风景照，它先缩成一张邮票大小的图再去看，小物体全糊了。你让它框出远处一只鸟的眼睛？做梦。

所以 Ferret-v2 要解决的问题就是：**怎么让模型既看得清全局，又看得清细节，还不把原来学的东西忘掉。**

---

## 三个核心操作

### 操作一：别暴力放大，切开来看

最直觉的办法是把图直接拉大，从 336 拉到 448 甚至更大。但问题是 ViT 在 336 上预训练了几亿张图，你突然给它 448 的输入，token 数量变了，positional embedding 也得插值，等于让它从舒适区跳到一个陌生环境。你的 fine-tuning 数据才 1.3M 张，人家预训练 400M 张，根本拧不过大腿，模型开始忘事儿。

**AnyRes 的思路就聪明多了**：你有一张大图，我把它切成几块 336×336 的小块，每块单独喂给 ViT。ViT 看到的每块图，尺寸跟它预训练时一模一样，它很舒服。然后把所有块的特征拼回来，就是一张高分辨率的特征图。

打个比方：Direct Upsampling 像把一张报纸直接拉大四倍，字是大了但模糊了；AnyRes 像拿放大镜一块一块看，每块都清楚，拼起来就全清楚了。

---

### 操作二：全局用 CLIP，局部用 DINOv2

切成块之后又有新问题。你有一张全局的低分辨率图 $I_g$ 和一堆局部的低分辨率块 $\{I_{l1}, ..., I_{lN}\}$。全局图能看到整个场景的 layout，但分辨率低；局部块分辨率高，但只看到一小块。

如果都用 CLIP 编码？不行。CLIP 的训练目标是 image-text contrastive，caption 写的是 "a dog on the beach"，它学到的是 image-level semantics，局部 texture、shape 这些它不关心。拿它去看单个局部 patch，它根本不知道那块 texture 是啥。

所以 Ferret-v2 干了件事：**CLIP 看全局，DINOv2 看局部。**

DINOv2 是 self-supervised 训练的，有 image-level 和 patch-level 两个目标，它对 local object 的 shape、texture 感知特别强。一个管 "这图整体啥意思"，一个管 "这块细节长啥样"，各司其职。

两个 encoder 后面各接一个独立的 MLP projector $MLP_g$ 和 $MLP_l$，为啥分开？因为全局语义和局部细节的 representation space 长得不一样，混在一起会互相干扰。

然后做 feature fusion。把所有局部 patch 的特征按原始空间位置拼起来 $\rightarrow H_l'$，把全局特征上采样到一样大 $\rightarrow H_g'$，然后 channel-wise 相加：

$$H_a = H_l' + H_g'$$

这个 $H_a$ 就是最终融合特征图，既带 high-res 的结构细节，又带 global 的语义信息。后面 Spatial-Aware Visual Sampler 拿这个 $H_a$ 去提取任意区域的连续特征。

---

### 操作三：三阶段训练，中间加个桥

传统 MLLM 两阶段训练：先对齐 image-caption，再 instruction tuning。Ferret-v2 在中间插了一阶段，变成三段式：

**Stage 1: Image-Caption Alignment**
低分辨率，1.4M image-text pairs，只训 projector，让 CLIP 和 LLM 先对上话。跟 Ferret、LLaVA 一模一样，没啥特别的。

**Stage 2: High-Resolution Dense Alignment**（这步是新加的）
问题在哪？Stage 1 只是让模型学会 "图 $\rightarrow$ 粗粒度描述"。但 downstream 任务需要的是精确的空间感知——你要框出每个小物体在哪。从粗粒度直接跳到 instruction tuning，gap 太大。

所以这阶段用 LVIS 数据集，每张图平均 10 个标注物体，设计两个任务：

- **Dense Referring**：给你一堆 region，你说每个 region 里是啥。"Region 1 是猫，Region 2 是狗..."
- **Dense Detection**：你列出图里所有物体的 box，但有个要求——**按 raster scan order**，从上到下从左到右排。

这个 raster scan order 设计得很妙。LLM 是自回归的，如果你不规定顺序，它生成 10 个 box 的时候会重复、遗漏、乱跳。强行让它按空间顺序输出，等于把 2D 空间拓扑 encode 成 1D 序列，LLM 天然擅长处理序列，这跟 CoT 的思路是一脉相承的。

这阶段只训两个 projector 和 visual sampler，encoder 和 LLM 都冻住。目的是让 visual sampler 先在高分辨率特征上 "练好手"。

**Stage 3: Instruction Tuning**
全解冻，用 GRIT + VQA + OCR 数据训。跟 LLaVA-1.5 的数据配方类似。但有两个额外操作：

- **Data Unification**：用 GLIPv2 给 VQA 文本里能 ground 的名词自动打 box 标签，用 OCR model 给 OCR 数据的文本打 box。这样 VQA 任务也能输出坐标了，grounding 和非 grounding 任务的格式统一了。
- **Task Generalization**：加一句 prompt "Include the coordinates for each mentioned object." 消除任务歧义，模型知道啥时候该输出坐标啥时候不该。

---

## 结果怎么样

数字说话：

- **ROC (LVIS-box)**：Ferret 79.42% $\rightarrow$ Ferret-v2 86.59%，7B 模型直接提升 7 个点
- **REC (RefCOCOg)**：Ferret 84.76% $\rightarrow$ Ferret-v2 89.27%，跟用 4B vision model 的 CogVLM-17B 持平
- **TextVQA**：Ferret 54.2 $\rightarrow$ Ferret-v2 61.7，OCR 类任务因为高分辨率大幅提升
- **Ferret-Bench**：64.5 $\rightarrow$ 75.6，复杂区域推理能力大幅提升

消融实验也清晰：
- 只加 AnyRes：Ferret-Bench 71.1 $\rightarrow$ 72.6
- 再加 DINOv2：$\rightarrow$ 75.3（multi-granularity 编码确实有用）
- 再加 Stage 2：$\rightarrow$ 75.6（dense alignment 桥梁有效）

---

## 一句话总结

Ferret-v2 的核心 insight 就三条：**别暴力放大图，切开来看；全局用 CLIP 管语义，局部用 DINOv2 管细节；训练中间加个 dense alignment 阶段当桥梁。** 三条加起来，让一个 7B 模型在 grounding 任务上打平甚至超过 17B 的模型。

---

好的, Andrej. 这篇 paper 是 Apple 联合 Columbia University 和 UCSB 推出的 Ferret-v2。作为 Ferret 的升级版，它主要解决前代模型在 fixed low-resolution 视觉 encoder 限制下，fine-grained visual comprehension 能力不足的问题。我会为你深入拆解其核心架构、公式细节、three-stage training paradigm，并补充大量相关联想来 build your intuition。

### 1. 核心动机与 Higher Resolution Scaling 的 Trade-off

现有的 Multimodal Large Language Models (MLLMs) 大多采用 pre-trained 的 CLIP-ViT 作为 vision encoder。由于 pre-training 分辨率通常锁死在 336×336，模型在处理需要细节的任务（如 small region referring, OCR）时表现挣扎。提升分辨率有两种主流路线：
*   **Direct Upsampling**: 直接放大输入图像，并通过 positional embedding interpolation 调整 ViT 输入。但这会强行改变 ViT 的 token sequence 长度，破坏 pre-training 分布。由于 fine-tuning 数据量远小于 pre-training 数据量，unfreeze encoder 反而会导致 catastrophic forgetting。
*   **Any Resolution (AnyRes)**: 摒弃暴力放大，转而将高分辨率图像切分为多个 sub-patches (例如 448×448 切分为几个 336×336 的 grid)。每个 patch 单独过 ViT。这种方式保留了 ViT pre-training 时熟悉的 token 长度，最大程度保留了预训练知识，同时获得了高分辨率的细节信息。

实验数据表明，AnyRes 策略在 ROC (Referring Object Classification), REC (Referring Expression Comprehension), TextVQA 和 Ferret-Bench 上全面碾压 Direct Upsampling。Ferret-v2 毫不犹豫地采用了 AnyRes 策略。

### 2. Multi-Granularity Visual Encoding 架构解析

引入 AnyRes 后，模型需要同时处理 global low-resolution image $I_g$ 和 local high-resolution patches $\{I_{l1}, I_{l2}, ..., I_{lN}\}$。这两种输入存在巨大的 granularity 差异。若统一用 CLIP 编码，CLIP 的 image-text contrastive objective 倾向于捕获全局语义，对 local patch 的 texture, shape 等像素级细节感知极弱。

为了 build intuition，你可以这样想：CLIP 像是一个远视眼，能概括“这是一张桌子的图”，但看不清桌子上的纹理；而 DINOv2 因为采用了 self-supervised patch-level objective，像一个显微镜，能刻画局部结构的细节。Ferret-v2 将两者结合：用 CLIP 编码全局，用 DINOv2 编码局部。

下面是论文中核心的公式拆解：

**Encoding Stage:**
$$ F_g = \text{CLIP}(I_g) ; \qquad F_{li} = \text{DINOv2}(I_{li}) , \qquad I_{li} \in \{I_{l1}, I_{l2}, ..., I_{lN}\} $$
*   $I_g$: Global low-resolution image。
*   $I_{li}$: 第 $i$ 个 local high-resolution patch。$N$ 是 patch 总数。
*   $F_g$: CLIP 提取的全局特征。
*   $F_{li}$: DINOv2 提取的局部特征。

**Projection Stage:**
$$ H_g = \text{MLP}_g(F_g) ; \qquad H_{li} = \text{MLP}_l(F_{li}) $$
*   $\text{MLP}_g$ 和 $\text{MLP}_l$ 是两个独立的 MLP projectors。分开投影是为了让模型分别学习 global 和 fine-grained 信息的 representation space，防止特征空间互相干扰。

**Feature Fusion for AnyRes Referring:**
为了让 Spatial-Aware Visual Sampler 能够提取 region feature，模型需要将局部特征拼接，并与上采样后的全局特征对齐相加。
$$ H_l' = \text{Concat}\{H_{l1}, H_{l2}, ..., H_{lN}\} \qquad (H_{li} \in \mathbb{R}^{w_l \times h_l \times c}, H_l' \in \mathbb{R}^{nw_l \times mh_l \times c}, n \times m = N) $$
*   $H_{li}$: 局部 patch 特征图，$w_l, h_l, c$ 分别为宽、高、通道数。
*   $H_l'$: 拼接后的大特征图。$n, m$ 代表 grid 的行列数，例如 $2 \times 3$ 的 grid，$N=6$。

$$ H_g' = \text{Upsample}(H_g) \qquad (H_g \in \mathbb{R}^{w_g \times h_g \times c}, H_g' \in \mathbb{R}^{nw_l \times mh_l \times c}) $$
*   $H_g$: 原始全局特征图。
*   $H_g'$: 经过插值上采样后的大特征图，spatial dimension 与 $H_l'$ 对齐。

$$ H_a = H_l' + H_g' $$
*   $H_a$: 最终融合的特征图。Channel-wise 相加类似于 ResNet 的 skip connection，既保留了 DINOv2 的 high-resolution structure details，又注入了 CLIP 的 global semantics。这个 $H_a$ 会被送入 Spatial-Aware Visual Sampler 用于提取任意形状区域的连续特征。

### 3. Three-Stage Training Paradigm

这是 Ferret-v2 设计中最精妙的部分。传统的 MLLM 两阶段训练 (Image-caption alignment -> Instruction tuning) 在引入高分辨率和复杂 grounding 任务时存在巨大的 gap。Ferret-v2 插入了一个 Dense Alignment 阶段：

**Stage I: Image-Caption Alignment**
*   **数据**: 1.4M image-text pairs。
*   **操作**: 只训练 CLIP 后面的 projector。低分辨率输入。建立基础的 vision-language 对齐。
*   **Frozen**: Vision Encoder, LLM。Visual Sampler 不参与。

**Stage II: High-resolution Dense Alignment (核心创新)**
*   **数据**: 基于 LVIS 数据集构建的 Dense Referring 和 Dense Detection 任务。每张图平均包含 10 个 object locations。
*   **操作**: 引入 DINOv2 encoder。DINOv2 的 projector 用 Stage I CLIP 的 projector 权重初始化以保证稳定。只训练两个 projectors 和 visual sampler。
*   **Task 设计**:
    *   *Dense Referring*: 输入多个 region，输出对应的 object categories。
    *   *Dense Detection*: 输入指令，要求模型按 raster scan order (从上到下，从左到右) 输出所有物体的 bounding box。这种按顺序输出的自回归方式，强迫 LLM 在生成 text token 时内部建立起 spatial awareness，这非常 brilliant，因为它用序列生成的顺序性编码了空间拓扑关系。
*   **Intuition**: 这个阶段作为一个桥梁。因为 instruction tuning 阶段的数据极其 sparse (一张图只有 1-2 个被提及的 object)，直接让模型学高分辨率 grounding 会非常困难。先在 LVIS 这种 dense annotation 上让 model 和 visual sampler 熟悉 high-resolution dense 特征的空间映射，再去做 instruction tuning，阻力就小多了。

**Stage III: Intent-Enhanced Instruction Tuning**
*   **数据**: GRIT dataset + 额外的 VQA 和 OCR datasets (LLaVA 1.5 的数据)。
*   **操作**: Unfreeze everything (encoders, projectors, samplers, LLM)。
*   **Data Unification**: 使用 GLIPv2 对 VQA 文本中 groundable 的 nouns 打上 bounding box 标签；使用 OCR model 对 OCR 数据打 text bounding box。这让模型在纯文本对话中也能自然地输出坐标，统一了 grounding 和非 grounding 任务的格式。

### 4. 实验数据与消融分析

在 Table 1 (ROC) 中，Ferret-v2-7B 在 LVIS-box 上达到 86.59%，相比 Ferret-7B 的 79.42% 有巨大飞跃。甚至在 SA-refer (高分辨率 in-the-wild 测试集) 上，7B 模型超越了此前的 13B 模型。这证明了高分辨率处理对小物体识别的决定性作用。

在 Table 3 (REC) 中，Ferret-v2-7B 在 RefCOCOg 上达到 89.42%，几乎追平了使用 4B vision model 和 6B connection module 的 CogVLM-Grounding-17B。

Table 6 的消融实验非常清晰：
*   仅用 CLIP 加 AnyRes: Ferret-Bench 为 72.6。
*   引入 DINOv2 做 multi-granularity: Ferret-Bench 提升到 75.3。这证明了多粒度编码的有效性。
*   再加上 Stage II Pre-training: Ferret-Bench 进一步提升到 75.6，并且 grounding 和 OCR 指标均有上升。这证明了 dense alignment 作为中间桥梁的必要性。

### 5. 相关联想与延展思考 (For Your Intuition)

1.  **LLaVA-NeXT/1.5 的 AnyRes 策略对比**: LLaVA-NeXT 也采用了 AnyRes 策略，但它是将所有 sub-patches 的 token sequence 拼接后直接输入 LLM。Ferret-v2 的不同之处在于，它不仅要输入 LLM，还要用这些特征图去进行 region feature extraction (给 Visual Sampler 用)。所以它必须进行 spatial feature map 的 reconstruction (公式 3 和 4 的 Concat 与 Upsample)，而 LLaVA-NeXT 只需要 1D token sequence。
2.  **CLIP vs DINOv2 的本质差异**: CLIP 的 contrastive loss 只有 image-level 的 supervision，caption 中的细粒度信息其实被浪费了，因此它的 attention map 往往非常发散且平滑。DINOv2 基于 student-teacher 的 self-distillation 和 masked image modeling，其 patch-level 特征极其 sharp，非常适合 dense prediction。Ferret-v2 将这两者相加 ($H_a = H_l' + H_g'$)，类似给高频的 structural signal 加上了一个低频的 semantic prior。参考 paper "From CLIP to DINO"。
3.  **Autoregressive Spatial Reasoning**: Stage II 中要求模型按照 raster scan order 输出 bounding boxes 是一种非常聪明的 inductive bias。传统的 detection model (如 DETR) 用 set prediction 摆脱了 NMS，但 MLLM 是自回归模型，如果在自回归过程中不给定顺序，模型生成多个 box 时会有严重的位置重复和遗漏问题。Raster scan order 利用了 LLM 处理序列的天然优势，将 2D 空间拓扑展开为 1D 有序序列，这与 Chain-of-Thought 有异曲同工之妙。

### References & Web Links
*   **Ferret-v2 Paper (arXiv)**: https://arxiv.org/abs/2404.07873
*   **Ferret (Previous Baseline) (arXiv)**: https://arxiv.org/abs/2310.07704
*   **LLaVA-NeXT (AnyRes Strategy Inspiration)**: https://arxiv.org/abs/2401.02913
*   **DINOv2 (Local Encoder)**: https://arxiv.org/abs/2304.07193
*   **CLIP (Global Encoder)**: https://arxiv.org/abs/2103.00020
*   **GLIPv2 (Used in Data Unification)**: https://arxiv.org/abs/2206.05836
