---
source_pdf: ShareGPT-4o-Image.pdf
paper_sha256: 14585e502457770beddbe9f0e2921ef20c038e4da3c06b85567b6bb2d49dd0fe
processed_at: '2026-08-12T05:37:54-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 一句话总结

OpenAI 的 GPT-4o-Image 画图很猛但不开放，这帮人用 GPT-4o-Image 生成 91K 张图（带 prompt），拿来 fine-tune 开源的 Janus-Pro，结果只花 6 小时 8 卡 A800 就让开源模型在画图上追上了 GPT-4o 一大截，还顺手解锁了"看图改图"的新能力。

## 为什么要这么干

GPT-4o-Image 出来的时候，开源圈都看傻了——让它画"一只穿宇航服的柴犬在月球上喝咖啡"，它真的能画出来，photorealistic，文字也清晰，还能根据你给的图做各种编辑。问题是 OpenAI 把它锁死在 API 后面，连 architecture 都不告诉你。

开源这边也有不少尝试，Janus-Pro、Emu3、LlamaGen 都在搞 autoregressive image generation，效果也不错，但跟 GPT-4o-Image 比就是差一截。差的不是架构，是数据——GPT-4o-Image 背后是 OpenAI 砸了大算力 + RLHF 训出来的 instruction alignment。

这帮人的 insight 很简单：**GPT-4o-Image 的输出本身就是最好的训练数据**。与其从 LAION 这种 web-crawled 噪声数据里捞，不如直接拿 GPT-4o-Image 生成干净的 (prompt, image) 配对，蒸馏它的能力到开源模型。这跟之前 NLP 领域的 Alpaca、Vicuna 思路完全一样，只是搬到图像生成上。

## 数据怎么造的

造数据有两类任务，text-to-image 和 text-and-image-to-image，各 ~45K。

**Text-to-image 那一半**用了两条 pipeline 互补：

第一条叫 **prompt-first**：先列一个六维 attribute space（Objects、Background、Style、Lighting、Camera、Composition），每个维度填一堆候选词。Objects 这维从 ImageNet 拿了 1000 类。然后随机采样 attribute 组合，让 Gemini-Pro-2.5 把这些散乱 attribute 编织成一句自然语言 prompt，再交给 GPT-4o-Image 生成图。

这里有个细节：选几个 object 放进 prompt 用 exponential decay 分布 $w(x) = \exp(-\lambda(x-1))$ 采样。直觉是大多数 prompt 只有 1-2 个 object，偶尔有几个复杂的（5+ objects），极少数极复杂（10+ objects）。这种 long-tail 采样保证了简单场景多，但复杂场景的 tail 也有覆盖——而复杂场景恰恰是 diffusion model 容易翻车的地方（比如 SDXL 画"三个人围桌打牌"经常多一只手或少一张牌）。

第二条叫 **image-first**：从 ALLaVA dataset 拿真实图，让 LLM 写一句描述当 prompt。这保证数据里也包含"描述真实场景"的语言风格，而不只是 synthetic structured prompt。

**Text-and-image-to-image 那一半**：定义了 14 种 editing task 分 5 大类，从 object 添加/删除、style transfer、background change、sketch-to-image 到 storyboard。流程是：拿一张 source image（可能来自前面 T2I 生成，也可能是 real photo），从 taxonomy 采样一个 task，LLM 根据 source image 内容写一条具体的 instruction，GPT-4o-Image 执行 instruction 生成 edited image。

这 91K 数据全由 GPT-4o-Image 生成，quality 一致性高，instruction alignment 也强。对比下 Instruct-Pix2Pix 当年用 470K 数据但 quality 参差不齐，UltraEdit 用 1M+，AnySD 用 4.1M——ShareGPT-4o-Image 用 91K 就够，背后是 GPT-4o-Image 输出的高信号密度。

## 模型怎么改的

Base 是 Janus-Pro-7B。Janus 系列的核心架构 insight 是 **decoupled visual encoding**：understanding 走 SigLIP 这种 semantic encoder（提取 high-level 含义，适合 captioning/VQA），generation 走 VQ tokenizer（把图像切成 discrete token sequence，适合 autoregressive 生成）。两条 pathway 接到同一个 LLM backbone。这样 generation 不会被 understanding encoder 的抽象 feature 拖累重建质量，understanding 也不会被 VQ 的信息损失坑到。

Janus-4o 在这基础上做了关键扩展：支持 text-and-image-to-image。这个 task Janus-Pro 原生干不了，因为它没有机制把 input image 同时作为 semantic context 和 pixel-level reference 注入。

Janus-4o 的解法是**双路注入 input image**：
- Semantic path：SigLIP 编码器把 input image 编成 global semantic embedding $\mathcal{E}(\hat{I})$，告诉模型"这是什么图"
- Pixel path：VQ tokenizer 把 input image 切成 token 序列 $\hat{X}$，告诉模型"像素怎么排布"

两路 representation 和 instruction prompt $S$ 一起拼成 LLM 输入序列，模型 autoregressively 生成 edited image tokens。

训练 loss 是标准 next-token prediction：
$$\mathcal{L} = -\sum_{i=1}^{N} \log P_\theta(x_i \mid x_{<i}, \mathcal{E}(\hat{I}), \hat{X}, S)$$

意思是给定 input image 的 semantic + pixel 双重 representation 和 instruction，模型预测每个 target image token。

**两个 mask trick** 是关键：
1. T2I 训练时 10% 的 text prompt tokens 随机 mask 成 padding
2. TI2I 训练时 50% 的 input image pixel tokens $\hat{X}$ 随机 mask

这两个 mask 让模型既学 conditional 分布又学 marginal 分布，相当于训练 side 就准备好了 classifier-free guidance 需要的 conditional + unconditional 两个 forward。

**推理时的双层 CFG** 是这个 paper 最巧妙的设计：

$$l_c' = \frac{l_c + s' \cdot l_o}{1 + s'}$$
$$l_g = l_u + s \cdot (l_c' - l_u)$$

三个 forward：
- $l_c$：full input（semantic + pixel + instruction 全给）
- $l_o$：pixel 那路被 mask 掉，只给 semantic + instruction
- $l_u$：全部 mask（null prompt）

第一层把 $l_c$ 和 $l_o$ 融合成 $l_c'$，$s'$ 控制"多强地看到 input image 像素"。$s'$ 大：淡化 pixel 信号，模型更自由发挥；$s'$ 小：忠于原图 layout。

第二层是标准 CFG，把 $l_c'$（conditional）和 $l_u$（unconditional）插值，$s$ 控制 instruction 跟随强度。

这跟 Instruct-Pix2Pix 的 dual CFG 思路一致，但实现上从 diffusion 的双 noise-image input 变成 AR 的三 forward。$s'$ 和 $s$ 都设 5。

## 训练成本

- 全参数 fine-tune Janus-Pro-7B
- 91K 数据（T2I + TI2I 混合随机采样）
- 3 epochs
- Learning rate $5 \times 10^{-6}$，batch size 128
- 8×A800，6 小时

总成本大概 50 GPU-hour，cloud cost $100 量级。任何学术组都能复现。

## 结果

三个 benchmark：

**GenEval**（text-to-image compositionality）：Janus-4o 0.80，比 Janus-Pro 0.76 提升 4 分。子项 color attribute 从 0.58 跳到 0.70，two object 从 0.85 到 0.92。超过 DALL-E 3 (0.67)、SDXL (0.55)、Emu3-Gen (0.54)。

**DPG-Bench**（dense prompt following）：Janus-4o 85.71，比 Janus-Pro 84.12 提升 1.6 分。Global prompt 子项从 80.70 到 92.59（+11.89），说明 GPT-4o 蒸馏数据对复杂多 entity prompt 的 grounding 提升很大。

**ImgEdit-Bench**（image editing）：Janus-4o 3.26（91K 数据），击败 Step1X-Edit 3.17（1M 数据）、ImgEdit-E1 3.17（1.2M）、UltraEdit 2.92（1M）、AnySD 2.62（4.1M）、Instruct-Pix2Pix 1.91（500K）。Motion Change 4.13 和 Style Transfer 4.47 是最强子项。

**Human eval**：52 个 T2I + 35 个 TI2I 真实 Twitter prompt 上对比，对 Janus-Pro win 67% lose 14%，对 UltraEdit win 60% lose 15%。

数据效率比 baseline 高 10-50 倍，这是最 shocking 的数字。

## 为什么 91K 就够

我的理解：

1. **GPT-4o-Image 输出本身已经"压缩"了 internet-scale 知识**。蒸馏它的输出相当于二级压缩，信号密度极高。

2. **Janus-Pro 已经 pre-trained**。它在大数据上学了图像分布先验，91K 只是"风格 + instruction alignment" 微调，不是 from scratch。

3. **Joint training 的正则效应**。T2I 和 TI2I 共用 backbone，互相 regularize。T2I 提供"无条件生成"能力，TI2I 提供"条件对齐"能力，比单训更好。

4. **Mask 策略做了 implicit data augmentation**。10% text mask + 50% image token mask 让每个样本给模型看了多个"视角"，有效样本数远超 91K。

5. **VQ tokenizer 让 image signal 密度高**。一张图 VQ 后变成 1K-4K tokens，每个 token 都贡献 loss，比 pixel-level diffusion 的 dense prediction 信号更集中。

## 我的几个直觉

**第一，distillation 是开源追 closed 的 standard play**。从 Alpaca 到 ShareGPT4V 到 ShareGPT-4o-Image，这个 pattern 会持续。未来 frontier model 的 moat 不在单点能力，而在 scale + RLHF + tool integration 的组合，单点能力被 distill 追上的速度会很快。

**第二，AR image generation 已经追上 diffusion**。Janus-4o 在 GenEval 80、DPG 85.71 都超过同代 diffusion baseline。AR 的优势是架构简单、跟 LLM 训练 stack 完全兼容、instruction following native 强。Emu3 "Next-token prediction is all you need" 这个 slogan 越来越像真的。

**第三，data quality > data quantity** 在 image generation 上也成立。这跟 LLM RLHF 的发现一致——少量高质量 instruction data 比大量低质量 data 有效。Instruct-Pix2Pix 用 500K 数据效果 1.91，Janus-4o 用 91K 效果 3.26，5 倍 data efficiency。

**第四，decoupled encoding 是 unified multimodal 的 key**。Janus 系列证明 understanding 和 generation 用不同 visual encoder 接到同一 LLM 比强行共用一个 encoder 好。这跟人类大脑视觉理解和视觉想象走不同回路有点像——你识别一只猫和你脑补画一只猫是两个 process。

**第五，mask-based CFG 训练 + multi-forward CFG 推理** 这个 pattern 在 AR framework 下做 image editing 很优雅。训练时 drop input，推理时多 forward 做 CFG，跟 diffusion 的 classifier-free guidance 完全平行，但实现上更自然。

## 几个潜在问题

1. **91K 能 generalize 多远**。Editing taxonomy 只有 14 类，对没见过的 multi-step complex editing 可能不稳。
2. **GPT-4o-Image 的 bias 继承**。Web-scale bias 会被蒸馏到开源模型，paper 没做 post-hoc filtering。
3. **Inference 速度**。AR 生成 image tokens 要 N 步，比 4-step distilled diffusion 可能慢。
4. **Resolution**。VQ tokenizer 通常 256×256 或 384×384，跟 SDXL/DALL-E 3 的 1024 还有 gap。
5. **Text-in-image**。GPT-4o-Image 的强项之一是文字渲染，paper 没专门测，但 Appendix D 的 document pipeline 暗示数据里包含 document 类 prompt，这能力可能 transfer 了。

## 这篇 paper 的大意义

它证明了**开源追 GPT-4o-Image 这类 closed 模型的成本比想象中低得多**。91K 数据 + 6 小时训练 + 8 卡 A800，成本 $100 量级，就能拿到 SOTA-tier 的 image editing 能力。这对开源社区是巨大鼓舞，对 closed model 厂商的 moat 是压力。

它还暗示**未来 image generation 的竞争焦点会从"架构创新"转向"数据 + 对齐"**。Janus-Pro 的架构已经够好，瓶颈是 high-quality aligned data。谁有高质量 distillation 数据（无论来自 frontier model 还是 human annotation），谁就赢。

参考链接：
- Paper GitHub: https://github.com/FreedomIntelligence/ShareGPT-4o-Image
- Janus-Pro: https://arxiv.org/abs/2501.17811
- Emu3: https://arxiv.org/abs/2409.18869
- LlamaGen: https://arxiv.org/abs/2406.06525
- InstructPix2Pix: https://arxiv.org/abs/2211.09800
- GenEval: https://arxiv.org/abs/2310.11513
- ImgEdit-Bench: https://arxiv.org/abs/2505.20275
- Classifier-Free Guidance: https://arxiv.org/abs/2207.12598

---

# ShareGPT-4o-Image 详解

## 一、整体故事线

这篇 paper 来自 The Chinese University of Hong Kong, Shenzhen，作者 Junying Chen、Zhenyang Cai、Pengcheng Chen、Shunian Chen 等，对应 GitHub repo: https://github.com/FreedomIntelligence/ShareGPT-4o-Image 。

核心 motivation 很清晰：GPT-4o-Image 在 text-to-image 和 text-and-image-to-image 上能力极强，但 OpenAI 完全封闭，连架构都不公开。开源这边（比如 Janus-Pro、LlamaGen、Emu3）虽然在 unified multimodal 生成上进展很快，但跟 GPT-4o-Image 比仍有明显 gap。这篇 paper 的贡献就是把 GPT-4o 的图像生成能力通过合成数据"搬运"到开源模型上，构建出 **ShareGPT-4o-Image** 数据集（45K text-to-image + 46K text-and-image-to-image，共 91K），并在其上 fine-tune Janus-Pro 得到 **Janus-4o**。

最 shocking 的一点是 **efficiency**：只用 91K 合成样本，8×A800 GPU 上 6 小时训练，就在 GenEval 上从 76 提到 80，DPG-Bench 上从 84.12 提到 85.71，并且**从零解锁** text-and-image-to-image 能力，在 ImgEdit-Bench 上以 91K 数据量击败需要 1M-4M 数据的 baseline。这背后是 high-fidelity distillation data 的威力。

## 二、数据集构造：双 pipeline 的精巧设计

### 2.1 Text-to-Image 数据（45K）

构造思路是 **prompt-first** + **image-first** 互补：

**Prompt-First Pipeline**（控制 attribute 覆盖）：
定义一个 6 维 attribute space：
- **Objects**：从 ImageNet 取 1000 类（Deng et al., 2009, https://arxiv.org/abs/1404.3600 实际是 CVPR 2009 paper）
- **Background**：abstract landscapes / urban / interior 三类
- **Style**：cultural & historical aesthetics / artistic movements / digital & graphic art
- **Lighting**：color temperature / direction / intensity & quality
- **Camera viewpoint**：vertical perspective / horizontal perspective / shot distance & framing
- **Composition**：basic division / balance & symmetry / framing depth & layers

从每个维度采样属性后，用 Gemini-Pro-2.5 (Team et al., 2023, https://arxiv.org/abs/2312.11805) 把这些散乱的 attribute 编织成一段 coherent 的 natural-language prompt，再交给 GPT-4o-Image 生成图像。

这里有个细节值得注意：**object 数量 $k$ 用 exponential decay 分布采样**：
$$w(x) = \exp(-\lambda (x - 1))$$
其中 $x \in [1, 100]$ 是采样到的 object 数量，$\lambda$ 控制衰减率。这个分布的目的是让"少量 object" 出现概率高（这是大多数常见 prompt 的形式），但保留"多 object 复杂场景"的 tail。这种 long-tail 采样对训练 compositional generation 很关键——GenEval 测的就是 multi-object compositionality（two obj、counting、position、color attribute 这些子项），而 multi-object 场景恰恰是大多数 diffusion model 翻车的地方。

**Image-First Pipeline**（grounding 在真实视觉分布）：
从 ALLaVA dataset (Chen et al., 2024a, https://arxiv.org/abs/2402.11684) 取真实图像，用 LLM 生成 detailed descriptive prompt。meta-prompt 在 Appendix C.1：
```
Please describe the main content of the image in one sentence. 
This sentence will be used as a prompt to regenerate the image, 
so it should clearly capture the key visual information. 
Only provide the sentence, no extra text.
```
这部分的目的是让数据集的 text distribution 同时覆盖"自然场景描述"的语言风格，不只是 synthetic prompts。这跟 DALL-E 3 (Betker et al., 2023) 用 BLIP 生成 caption、PixArt-α (Chen et al., 2023, https://arxiv.org/abs/2310.00426) 用 LLaMA 生成 detailed prompt 是一脉相承的思路。

### 2.2 Instruction-Guided Image Editing 数据（46K）

定义了 14 个 editing task taxonomy，分 5 大类（见 Table 5）：
1. **Image editing and manipulation**：inpainting/replacement, element manipulation, background modification, attribute/effect manipulation
2. **Style transfer and artistic transformation**：specific artist styles, medium/technique styles, genre/theme/era shifting
3. **Content augmentation and extension**：resolution/detail enhancement, outpainting/inpainting for extension
4. **Structured generation and conditional control**：from sketch/lineart/edges, from pose/depth/segmentation
5. **Creative ideation and iteration**：storyboarding/sequential generation, concept variation/exploration

数据格式是 triplet (source image, instruction, edited image)。source image 一部分来自前面生成的 text-to-image 输出，一部分来自 curated real-world photos。task 从 taxonomy 采样后，LLM 根据 source image content 合成具体的 natural-language instruction，最后 GPT-4o-Image 执行 instruction 得到 edited output。

值得对比的是 InstructPix2Pix (Brooks et al., 2022, https://arxiv.org/abs/2211.09800)、MagicBrush (Zhang et al., 2023)、HQ-Edit (Hui et al., 2024, https://arxiv.org/abs/2404.09990)、UltraEdit (Zhao et al., 2024)、ImgEdit (Ye et al., 2025, https://arxiv.org/abs/2505.20275)。这些 dataset 用 real image 作 target 会有 quality 不一致问题，而 ShareGPT-4o-Image 全部由 GPT-4o-Image 生成，quality 一致性更高、instruction alignment 更强。

## 三、Janus-4o 模型架构与训练

Janus-4o 建立在 Janus-Pro (Chen et al., 2025c, https://arxiv.org/abs/2501.17811) 上。Janus / Janus-Pro 的核心 architecture insight 是 **decoupled visual encoding for understanding and generation**：

- **Understanding pathway**：用 SigLIP 等 vision encoder 提取 high-level semantic feature（适合 captioning、VQA）
- **Generation pathway**：用 VQ tokenizer 把图像 flatten 成 discrete token sequence（适合 autoregressive 生成）

两条 pathway 接到同一个 LLM backbone 上。这个设计避免了 understanding encoder 的 semantic feature 直接拿来生成（这会让 reconstruction 质量下降），也避免了把 VQ tokenizer 拿来做 understanding（high-level semantics 信息损失）。

Janus-4o 的创新点是把 generation pathway 进一步扩展到 image-conditioned generation（即 text-and-image-to-image），这是 Janus-Pro 原本不支持的。

### 3.1 Text-to-Image 训练

输入：text prompt $S = (s_0, s_1, \ldots, s_M)$ 经过 text tokenizer；目标 image $I$ 经过 VQ codebook 映射成 image tokens $X = (x_0, x_1, \ldots, x_N)$。

LLM 接收 embedding 后，autoregressively 预测 image tokens，loss 是标准 next-token prediction：

$$\mathcal{L} = -\sum_{i=1}^{N} \log P_\theta(x_i \mid x_{<i}, S)$$

变量含义：
- $\theta$：所有可训练参数（全参数 fine-tune）
- $x_i$：第 $i$ 个 image token
- $x_{<i} = (x_0, x_1, \ldots, x_{i-1})$：前面已生成/ground-truth 的 image tokens（teacher forcing）
- $S$：text prompt 的 token sequence
- $P_\theta(x_i \mid x_{<i}, S)$：模型在 context $x_{<i}, S$ 下预测 token $x_i$ 的概率分布

**10% 的 prompt tokens $S$ 随机 mask 成 padding tokens**——这个 trick 是模仿 GPT-4o-style modeling，目的是鼓励模型在 pixel 层面做更深的 dependency modeling，而不是过度依赖 text prompt。这有点像 classifier-free guidance 的训练侧 drop——同一个模型既要学 conditional $P(x \mid S)$ 也要学 marginal $P(x)$，因此推理时可以做 CFG。

**Inference**：logit 计算用 CFG 形式：

$$l_g = l_u + s \cdot (l_c - l_u)$$

变量含义：
- $l_c$：conditional logit（prompt $S$ 完整输入模型，正常 forward）
- $l_u$：unconditional logit（prompt $S$ 被 mask 成 padding，即 null prompt 的 forward）
- $s$：scaling factor，论文用 $s = 5$
- $l_g$：guidance 后的最终 logit，用于 sampling
- 温度 $T = 1.0$

这正是 Classifier-Free Guidance (Ho & Salimans, 2022, https://arxiv.org/abs/2207.12598) 在 autoregressive framework 下的直接应用——和 LlamaGen (Sun et al., 2024a, https://arxiv.org/abs/2406.06525)、Emu3 (Wang et al., 2024, https://arxiv.org/abs/2409.18869) 一致。

### 3.2 Text-and-Image-to-Image 训练：新能力的解锁

这部分是 paper 的关键技术贡献。Janus-Pro 不支持这个任务，因为它没有机制把 input image 既作为 semantic conditioning 又作为 pixel-level reference 同时注入。Janus-4o 的解法是**双路注入**：

给定 input image $\hat{I}$：
1. **Semantic path**：image encoder $\mathcal{E}$ 产生 semantic embedding $\mathcal{E}(\hat{I})$（对应 understanding pathway 的 SigLIP-like encoder）
2. **Pixel path**：VQ codebook 把 $\hat{I}$ tokenize 成 $\hat{X} = (\hat{x}_0, \hat{x}_1, \ldots, \hat{x}_N)$（对应 generation pathway 的 tokenizer）

两个 representation 都和 prompt tokens $S$ 拼接成输入序列，LLM autoregressively 生成 target image tokens $X$。训练 loss：

$$\mathcal{L} = -\sum_{i=1}^{N} \log P_\theta(x_i \mid x_{<i}, \mathcal{E}(\hat{I}), \hat{X}, S)$$

新变量：
- $\hat{I}$：input source image（要被编辑的原图）
- $\mathcal{E}(\hat{I})$：input image 的 semantic embedding（global representation，捕捉"这是什么图"的语义）
- $\hat{X}$：input image 的 VQ tokens（local pixel-level representation，捕捉"像素分布是什么"）
- $S$：editing instruction 的 tokens

为什么需要两路？因为：
- 只有 $\mathcal{E}(\hat{I})$：太抽象，丢掉细节，模型很难做局部修改（比如"在帽子颜色上加个红色"）
- 只有 $\hat{X}$：太冗长且 VQ 量化损失大，全局语义不清晰
- 两者一起：模型既能"看到" image 的全局含义，又能"参照" pixel-level layout 做编辑

**50% 的 $\hat{X}$ 随机 mask**：避免模型 overfit 输入图像（即直接拷贝 $\hat{X}$）。这迫使模型既学"参照输入"模式又学"独立生成"模式，为推理时的双 CFG 提供基础。

**Inference 双层 CFG**：

$$l_c' = \frac{l_c + s' \cdot l_o}{1 + s'}$$

$$l_g = l_u + s \cdot (l_c' - l_u)$$

变量含义：
- $l_c$：full conditional logit（$\mathcal{E}(\hat{I})$、$\hat{X}$、$S$ 全部输入）
- $l_o$：partial conditional logit（$\mathcal{E}(\hat{I})$ 保留，但 $\hat{X}$ 被 mask 掉，$S$ 保留）—— 即"我有 image 的 semantic context，但不知道具体像素"
- $l_u$：unconditional logit（$\mathcal{E}(\hat{I})$、$\hat{X}$、$S$ 全部 mask）—— 即 null prompt
- $s' = 5$：image guidance scale（控制偏离 input image 的程度）
- $s = 5$：text guidance scale（控制遵循 instruction 的程度）
- $l_c'$：融合后的 conditional logit
- $l_g$：最终用于 sampling 的 guided logit
- 温度 $T = 1.0$

直觉上理解两层 CFG：
- 第一层 $l_c' = \frac{l_c + s' \cdot l_o}{1 + s'}$ 是"image-aware vs image-free"的 blend，$s'$ 控制模型多强地"看到" input image 像素。$s'$ 高（如 5）：$l_c'$ 接近 $l_o$，相当于淡化 $\hat{X}$ 像素信号，模型更自由；$s'$ 低：$l_c'$ 接近 $l_c$，模型更忠于原图 layout
- 第二层 $l_g = l_u + s(l_c' - l_u)$ 是 standard CFG 形式，只是 conditional 换成 $l_c'$，unconditional 是 $l_u$（彻底无输入）

这个 nested CFG 设计和 InstructPix2Pix 的 dual CFG（image CFG + text CFG）思路一致，但实现方式不同——InstructPix2Pix 用 two noise-image input 双 forward，这里用 autoregressive 三 forward（$l_c, l_o, l_u$）。这种设计在 autoregressive framework 下做 image editing 是 Janus-4o 的核心创新。

paper 中提到 $s'$ 参数的语义："lower values preserve more of the original, while higher values allow more creative changes" —— 这跟 InstructPix2Pix 的 image CFG scale 完全一致。

### 3.3 Joint Fine-Tuning 训练细节

- Base model: **Janus-Pro-7B**
- Dataset: ShareGPT-4o-Image（45K T2I + 46K TI2I = 91K）
- Training mode: **joint random sampling over 3 epochs**（每 step 随机抽 T2I 或 TI2I 样本）
- Fine-tuning type: **full fine-tune**（不是 LoRA）
- Learning rate: $5 \times 10^{-6}$
- Batch size: 128
- Hardware: 8×A800
- Time: 6 小时

这里有几个 Karpathy 会关心的点：
1. **full fine-tune 而不是 LoRA**：91K 样本对 7B model 来说不大，full FT 没崩，说明 data 质量高
2. **joint training 没有 catastrophic forgetting**：T2I 性能还提升了 4 个点，TI2I 也学会了——这可能得益于 mask 策略让两个任务共享 conditional probability 表达
3. **6 小时训练能拿到 SOTA-tier 编辑能力**：数据 efficiency 惊人，对比 UltraEdit 用 1M+ samples、ImgEdit-E1 用 1.2M、AnySD 用 2.5M、Step1X-Edit 用 1M+，Janus-4o 只用 91K 就跟它们打平甚至更好

## 四、实验结果深度解读

### 4.1 GenEval (Ghosh et al., 2024, https://arxiv.org/abs/2404.02608)

Table 1 关键数字：
- **Janus-4o: 0.80 overall**（+4 points over Janus-Pro 0.76）
- 子项提升：Two Obj 0.85→0.92, Counting 0.53→0.58, Color Attri. 0.58→0.70

这里 color attribute 从 0.58 跳到 0.70 是很大的提升，说明 GPT-4o-Image 蒸馏的 color grounding 比 Janus-Pro 原始训练数据强很多。Counting 提升 +0.05 看似不大，但 GenEval counting 是出了名的难（SD3-Medium 才 0.72）。

超越的 baseline：DALL-E 3 (0.67)、SDXL (0.55)、Emu3-Gen (0.54)、PixArt-α (0.48)。Janus-4o 的 0.80 已经超过所有公开 text-to-image diffusion model，仅次于 SD3-Medium (0.74) 被 Janus-Pro-7B (0.76) 已经超过——也就是说 unified autoregressive MLLM 在 GenEval 上已经全面超过专用 diffusion model。

### 4.2 DPG-Bench (Hu et al., 2024, https://arxiv.org/abs/2403.05135)

Table 2：
- **Janus-4o: 85.71 overall**（+1.59 over Janus-Pro 84.12）
- 子项 Global 大幅提升：80.70 → 92.59（+11.89）

Global prompt 是 DPG-Bench 测 multi-entity、复杂关系的子项，提升 12 点很显著。说明 GPT-4o 蒸馏数据在 dense instruction following 上有质变。

对比：DALL-E 3 (83.50)、SD3-Medium (84.08)、Emu3-Gen (80.60)。Janus-4o 85.71 是新高。

### 4.3 ImgEdit-Bench (Ye et al., 2025, https://arxiv.org/abs/2505.20275)

Table 3 是 Janus-4o 真正的"从零解锁"的展示：
- **Janus-4o: 3.26 avg (91K samples)**
- 对比：Step1X-Edit 3.17 (1M samples)、ImgEdit-E1 3.17 (1.2M samples)、UltraEdit 2.92 (1M samples)、AnySD 2.62 (4.1M samples)、Instruct-Pix2Pix 1.91 (500K samples)

子项亮点：
- Motion Change: 4.13（最强）
- Style Transfer: 4.47（最强）
- Replacement: 3.27
- Background Change: 3.32

Motion Change 和 Style Transfer 的强势很有意思——这两类要求模型既理解 input image 语义（识别物体、动作）又有 creative 生成能力，正好是 decoupled encoding + 双 CFG 设计的 sweet spot。

91K 数据量级 vs 1M-4M baseline，data efficiency ratio 大约 10-50×，这强烈印证 GPT-4o-Image 的输出作为 distillation target 的 quality 远高于 web-crawled 真实图像 + human-written instruction 的组合。

### 4.4 Human Evaluation

Figure 5 在 52 个 T2I 和 35 个 TI2I 真实 Twitter prompt 上做 pairwise：
- vs Janus-Pro (T2I)：win 67%、tie 19%、lose 14%
- vs UltraEdit (TI2I)：win 60%、tie 25%、lose 15%

Human eval 比 benchmark 数字更直接——GPT-4o-style 的 photorealism 和 instruction alignment 是真实用户能感知到的。

## 五、跟相关工作的脉络

### 5.1 Autoregressive Image Generation 系

- **LlamaGen** (Sun et al., 2024a, https://arxiv.org/abs/2406.06525)：把 Llama 直接拿来做 image generation，证明 AR 在图像上 scaling 也 work
- **Emu3** (Wang et al., 2024, https://arxiv.org/abs/2409.18869)："Next-token prediction is all you need"——纯 AR next-token 训练统一 understanding 和 generation
- **VAR** (Tian et al., 2024, https://arxiv.org/abs/2404.02905)：next-scale prediction 而不是 next-token，coarse-to-fine
- **Transfusion** (Zhou et al., 2024, https://arxiv.org/abs/2408.11039)：同一模型既做 text next-token 又做 image diffusion loss
- **Show-o** (Xie et al., 2024, https://arxiv.org/abs/2408.12528)：single transformer 统一 understanding + generation

Janus 系列（包括 Janus-4o）的差异化是 **decoupled encoding** + 纯 AR generation（不用 diffusion loss），这让模型架构极简，跟 LLM 训练 stack 完全兼容。

### 5.2 Image Editing 数据集系

- **Instruct-Pix2Pix** (Brooks et al., 2022)：用 GPT-3 生成 prompt + stable diffusion 生成配对，~470K
- **MagicBrush** (Zhang et al., 2023)：human-annotated，~10K
- **HQ-Edit** (Hui et al., 2024)：用 GPT-4V 生成 high-quality pair
- **UltraEdit** (Zhao et al., 2024)：~1M，in-house + curated
- **ImgEdit** (Ye et al., 2025)：~1.2M，大规模

ShareGPT-4o-Image 是第一个直接从 GPT-4o-Image distill 的 editing dataset，量级最小但 quality 最高。

### 5.3 Distillation from Proprietary Model 系

这个思路在 NLP 已经很成熟（Alpaca、Vicuna 等 distill from GPT-4），但在图像生成上才刚起步。ShareGPT-4o-Image 是图像版的 distillation。类似的还有：
- **ShareGPT4V** (Chen et al., 2024a, ALLaVA 团队)：GPT-4V 蒸馏图像 understanding
- 这篇 ShareGPT-4o-Image 是 GPT-4o 蒸馏图像 generation

## 六、直觉：为什么 91K 就够？

这是 paper 最 provocative 的发现。我的理解：

1. **GPT-4o-Image 输出本身已经高度"压缩"了 internet-scale 知识**。让开源模型拟合 GPT-4o-Image 输出，相当于站在巨人肩膀上做"二级 distillation"，比从头拟合 LAION-5B 这种噪声极大的 web data 高效得多。这正是 DALL-E 3 用的 caption augmentation 思路的延伸——把"高质量"信号打包进数据。

2. **Janus-Pro 已经做了大规模 pre-training**。它已经在 LAION 等大数据上学了图像分布的"先验"。Janus-4o 只是在这个先验上做"风格 + instruction alignment" 微调，91K 高质量样本足够。

3. **Joint training 的正则效应**。T2I 和 TI2I 共用 backbone 和 tokenizer，两个任务互相 regularize——T2I 提供"无条件生成"能力，TI2I 提供"条件对齐"能力，比单独训更好。这跟 multi-task learning 的迁移效应一致。

4. **Mask 策略让模型学 robust representation**。10% text mask + 50% image token mask 实际上做了 implicit data augmentation——每个样本都给模型看了多个"视角"（带 prompt / 不带 prompt / 带像素 / 不带像素），有效样本数远大于 91K。

5. **VQ tokenizer 让 image token sequence 长度受限**。一张图大概 256×256 = 65K 像素，VQ 后变 1024-4096 tokens，比 pixel-level diffusion 训练信号密度高，每个 token 都贡献 loss，sample efficiency 天然高。

## 七、潜在局限与未来方向

paper 没展开但值得讨论的：

1. **91K 数据是否能 generalize 到任意 instruction**。Editing taxonomy 只覆盖 14 个 task + 5 大类，对未见过的复杂 multi-step editing（如"先换背景再加3只猫然后改成水彩画"）可能不稳。需要更大规模、更复杂 instruction 数据。

2. **GPT-4o-Image 的 bias 继承**。Appendix E 提到 GPT-4o-Image 自身可能继承 web-scale bias（人种、性别、年龄），ShareGPT-4o-Image 不做 post-hoc filtering。下游 model 会继承这些 bias。

3. **Inference 速度**。AR 生成图像需要 N 步 token prediction（N 是 image token 数量），相比 diffusion 4-step distillation 模型（如 SD3-Turbo）可能慢。不过 MLLM framework 的优势是统一 inference pipeline。

4. **Resolution 限制**。VQ tokenizer 通常在 256×256 或 384×384 工作，photorealistic 高分辨率（1024+）需要 multi-scale AR 或后处理 super-resolution。Janus-Pro 已经在 384×384 工作，但跟 SDXL (1024)、DALL-E 3 (1024) 还有 resolution gap。

5. **Text-in-image generation**。GenEval 不测文字渲染，但 GPT-4o-Image 的强项之一是 text-in-image（比 SD3、DALL-E 3 都强）。Janus-4o 是否继承了这能力 paper 没测，但很可能 ShareGPT-4o-Image 数据中包含 document 类 prompt（Appendix D 的 Document Pipeline 就是为此设计）能 transfer。

## 八、对 MLLM 发展方向的启示

这篇 paper 在我看来印证了几个 trend：

1. **AR image generation 正在全面追上 diffusion**。Janus-4o 在 GenEval 80、DPG 85.71 都超过同代 diffusion baseline。这种 AR 框架的优势是 architecture simplicity + LLM-native instruction following。

2. **Distillation from frontier model 成为开源追赶 closed model 的 standard play**。Alpaca 之于 GPT-4、ShareGPT4V 之于 GPT-4V、ShareGPT-4o-Image 之于 GPT-4o-Image——这个 pattern 会持续。但注意 compliance 和 ethical 问题（OpenAI ToS 是否允许这种方式）。

3. **Multi-task joint training > single-task fine-tune**。Janus-4o 用同一份 91K 数据同时学 T2I 和 TI2I，两个 task 互相提升。这跟 Chameleon (Team, 2024, https://arxiv.org/abs/2405.09818)、Emu3 的"unified"思想一致。

4. **Mask-based CFG 训练 + multi-pathway CFG 推理**。这种训练时 drop input、推理时多 forward 做 CFG 的 pattern 在 AR framework 下很自然，是 image editing 在 LLM 时代的范式。

5. **Data quality > Data quantity**。91K 高质量蒸馏数据 vs 4M web data，前者胜。这跟 LLMpost-training 中"少量 high-quality instruction data > 大量低 quality data"的 RLHF finding 一致。

## 九、复现与代码

GitHub repo: https://github.com/FreedomIntelligence/ShareGPT-4o-Image

训练成本极低（6 小时 8×A800 ≈ 48 GPU-hour，约 $100 量级 cloud cost），任何学术组都能复现。这对开源社区跟进 GPT-4o-Image 能力意义重大。

## 十、总结

ShareGPT-4o-Image + Janus-4o 是一篇"小而精"的工作：
- 数据集设计上用 prompt-first + image-first 互补，editing taxonomy 14 类覆盖主要 use case
- 模型上把 Janus-Pro 的 decoupled encoding 思路扩展到 image-conditioned generation，引入双路注入 + 双层 CFG
- 训练上 91K 数据 6 小时拿到 GenEval 80 / DPG 85.71 / ImgEdit 3.26 的开源 SOTA
- 思路上继承了 distillation + AR unified multimodal + data quality > quantity 三大 trend

对我来说最有趣的 takeaway 是：**GPT-4o-Image 这种 closed 模型的输出本身已经是高质量 distillation target，开源社区可以用极少量的这种数据 + joint training 快速追上**。这暗示未来 frontier model 的 competitive moat 不在"输出来样能力"，而在"独家大规模 pre-training compute + RLHF 信号 + tool integration"——单一能力层面，开源通过 distillation 追赶的速度会比想象中快。

参考链接：
- 论文 GitHub: https://github.com/FreedomIntelligence/ShareGPT-4o-Image  
- Janus-Pro: https://arxiv.org/abs/2501.17811  
- Janus (原始): https://arxiv.org/abs/2410.13848  
- LlamaGen: https://arxiv.org/abs/2406.06525  
- Emu3: https://arxiv.org/abs/2409.18869  
- VAR: https://arxiv.org/abs/2404.02905  
- Transfusion: https://arxiv.org/abs/2408.11039  
- Show-o: https://arxiv.org/abs/2408.12528  
- Chameleon: https://arxiv.org/abs/2405.09818  
- Classifier-Free Guidance: https://arxiv.org/abs/2207.12598  
- InstructPix2Pix: https://arxiv.org/abs/2211.09800  
- GenEval: https://arxiv.org/abs/2310.11513  
- DPG-Bench (ELLA): https://arxiv.org/abs/2403.05135  
- ImgEdit-Bench: https://arxiv.org/abs/2505.20275  
- UltraEdit: https://arxiv.org/abs/2407.05282  
- HQ-Edit: https://arxiv.org/abs/2404.09990  
- ALLaVA: https://arxiv.org/abs/2402.11684  
- PixArt-α: https://arxiv.org/abs/2310.00426  
- DALL-E 3: https://cdn.openai.com/papers/dall-e-3.pdf  
- GPT-4o image generation (OpenAI blog): https://openai.com/index/introducing-4o-image-generation/
