---
source_pdf: Scaling Large Motion Models with Million-Level Human Motions.pdf
paper_sha256: c9ec1281d74bd796b3b241caa49e0193ee4a344b1b6dd6a6b934f3f91bf069cf
processed_at: '2026-08-12T03:33:54-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Being-M0 这篇 paper

## 一句话总结

**motion generation 这个领域一直被小数据卡住，这篇 paper 干了三件事: 造了个百万级的 motion dataset（MotionLib）、设计了个能扛百万数据的 tokenizer（2D-LFQ）、然后用 LLaMA-13B 把 scaling law 第一次在 motion 上完整跑通。**

参考: [Being-M0 Project Page](https://beingbeyond.github.io/Being-M0/)

---

## 为什么 motion 一直做不好

你想想 CV 的历史。ImageNet 1.2M images 直接把 AlexNet 喂出来了，然后 VGG、ResNet、ViT 一路 scaling。NLP 那边更夸张，Common Crawl 几百 B token 把 GPT 喂出来了。

motion 呢？最大的 HumanML3D 才 29K sequences，Motion-X 也才 81K。这就好比你拿 MNIST 想训 GPT-4。所有"motion model 表现不行"的结论，本质上都受制于这个 data ceiling。

paper 一上来就直击这个痛点，画了张图把 motion data 和 image-text data 的 scale 差距摆出来——差好几个数量级。所以他们做的第一件事就是**先把数据搞大**。

参考: [HumanML3D paper](https://arxiv.org/abs/2205.10839) | [Motion-X paper](https://arxiv.org/abs/2307.00818)

---

## MotionLib 怎么造出来的

### 数据来源

从 20M 视频里筛，包括 Kinetics-700、YouTube、NTU-RGBD-120、BEDLAM、GTA-Human 等等。最后搞出 1.2M sequences，1456 小时，比之前最大的 dataset 大 15×。

pipeline 大概是这样:

1. **2D keypoint 检测**: 用 ViTPose 把没人的视频筛掉
2. **3D motion 提取**: 用 WHAM（CVPR 2024，Shin et al.）从视频回归出 SMPL 参数，关键是输出在 **world coordinate** 而不是 camera coordinate，这样 motion 才能脱离相机视角独立存在
3. **物理 refine**: 训了个 RL policy $\pi_{\text{refine}}$，让 raw motion 满足物理定律（balance、no foot-sliding）。剧烈运动 refine 不动的就打个小权重，半监督那味儿
4. **occlusion / blur 处理**: 用 SAM 做 segmentation mask，配合 trajectory smoothing

参考: [ViTPose](https://arxiv.org/abs/2204.07671) | [WHAM](https://arxiv.org/abs/2404.05198) | [SAM](https://arxiv.org/abs/2304.02643) | [Perpetual Humanoid Control (RL refine)](https://arxiv.org/abs/2310.01886)

### Text annotation 是另一个亮点

之前 dataset 的 text 就是"一个人走路"这种一句话描述。MotionLib 搞了**hierarchical text**:
- **Part-level**: 每个身体部位（左臂、右腿）单独一句话
- **Body-level**: 整体 1-3 句话总结

用 Gemini-1.5-Pro 生成，GPT-4o 做 cross-check（避免自评 bias）。Table 9 显示 MotionLib 的 text 质量 score 3.837，远高于 Motion-X 的 1.703 和 HumanML3D 的 1.386。

这个 hierarchical 设计的 intuition 是: LLM 要理解 motion 就得有 part-level granularity，不然你没法说"只抬左手"这种指令。

参考: [Gemini 1.5 Report](https://arxiv.org/abs/2403.05530) | [Hierarchical HOI (Pi et al., 2023)](https://arxiv.org/abs/2307.11598)

---

## Being-M0 的架构直觉

整体哲学是 **"motion is a foreign language"**，借鉴 [UniVL](https://arxiv.org/abs/2002.06353) 和 [Pixels to Tokens](https://arxiv.org/abs/2410.02155)。

具体来说:
1. Motion tokenizer 把 $\mathcal{M} = \{m_1, ..., m_T\}$ 编码成 tokens $\mathcal{V} = \{v_1, ..., v_n\} \in \mathbb{R}^{n \times d}$
2. codebook 里的 K 个 code 直接塞进 LLM 的 vocabulary 当 additional tokens
3. 加两个 special token `<mot>` 和 `</mot>` 标记 motion 边界
4. 用 decoder-only causal transformer 自回归生成

训练 loss 就是标准 next-token prediction:

$$\mathcal{L}(\Theta) = -\sum_{j=1}^{L} \log P_\Theta(y_j | \text{desc}, \hat{y}_{1:j-1})$$

变量含义:
- $\Theta$: 模型参数
- $L$: target sequence 长度
- $y_j$: 第 $j$ 个 target token
- $\hat{y}_{1:j-1}$: 已生成的 input tokens
- $\text{desc}$: 文本描述（可以为空，纯 motion continuation 任务）

### Two-stage training

学 LLaVA 那套:
- **Stage 1 - Pretrain**: 在整个 MotionLib 上做 motion-text alignment
- **Stage 2 - Instruction tuning**: 250+ instruction template + Gemini-Pro refine，搞出 900K instruction data

参考: [LLaVA](https://arxiv.org/abs/2304.08485) | [InstructBLIP](https://arxiv.org/abs/2305.06500)

---

## MotionBook 是这篇 paper 的灵魂

tokenizer 是 bottleneck。之前 VQ 有两个根本问题:

### 问题 1: 1D embedding 表达力不够

motion state $m_i \in \mathbb{R}^D$ 里其实塞了一堆 heterogeneous 东西: joint rotation、position、velocity、foot contact。传统 VQ 把每个 timestamp 压成 1D embedding $d$ 维，相当于"一个人一句话概括全身所有关节的所有物理量"——肯定丢信息。

### 问题 2: codebook collapse

VQ 的 codebook 通常 512-1024，想扩大就 collapse（只用到其中一小撮 code）。

### 2D-LFQ 的解法

paper 提出的 2D-LFQ 两个核心 trick:

**Trick 1: 从 1D 到 2D**

把 motion $\mathcal{M} \in \mathbb{R}^{T \times D}$ 当成单通道 image $\mathcal{M} \in \mathbb{R}^{T \times D \times 1}$，encoder 输出变成 $\mathbb{R}^{\lfloor T/\alpha \rfloor \times P \times d}$。这里 $P$ 是把 motion feature 拆成几个 component（root rotation、joint rotation、foot contact 等），**每个 body part 独立 tokenize**，这就有了 part-level resolution。

**Trick 2: Lookup-Free Quantization**

codebook 不要 embedding，直接用整数集合:

$$\mathbb{C} = \times_{i=1}^{d} C_i, \quad C_i = \{-1, 1\}$$

每个维度就俩值 -1 和 1，量化函数特别简单:

$$Q(z_i) = -\mathbb{1}\{z_i \leq 0\} + \mathbb{1}\{z_i >  0\}$$

变量含义:
- $z_i$: feature vector 第 $i$ 维
- $\mathbb{1}\{\cdot\}$: indicator function
- 直觉就是: 正数 → 1，负数 → -1，一个 bit

token index 是二进制编码:

$$\text{Index}(z) = \sum_{i=1}^{d} 2^{i-1} \mathbb{1}\{z_i > 0\}$$

如果 $d=16$，codebook 大小 $= 2^{16} = 16384$，是传统 VQ 1024 的 16 倍。

### 为什么这个 work

intuition 是这样的:

1. **没有 lookup → 没有 collapse**: 传统 VQ 用 nearest neighbor 找 code，容易陷入只用一小部分 code。LFQ 每个 dim 就二值化，所有 $2^d$ 个组合理论上都会被均匀用到
2. **codebook 可以指数扩大**: 想要更大 codebook 就加 $d$，不增加任何 lookup 成本
3. **限制单 token 表达力 → 强制充分利用 codebook**: 这个跟 [Mentzer et al., FSQ](https://arxiv.org/abs/2309.15505) 的 insight 一致，单个 code embedding 维度小反而更好

### SMPL-D135 feature

paper 还顺手把 motion feature 也重新设计了。之前的 H3D-Format 通过 IK 从 position 反推 rotation，丢信息且慢。SMPL-D135 直接编码原始 SMPL rotation:

- **Root (9D)**: 6D rotation (6) + XZ velocity (2) + height (1)
- **Body joints (126D)**: 21 joints × 6D rotation

总共 135 维，比 H3D-Format 的 263 维 compact 一倍，但是 lossless。Table 7 显示 SMPL-D135 的 FPS > 100，H3D-Format 才 0.41——这意味着实时 animation 应用直接可行。

参考: [6D rotation representation (Zhou et al.)](https://arxiv.org/abs/1812.07035) | [FSQ](https://arxiv.org/abs/2309.15505) | [LlamaGen (LFQ for image)](https://arxiv.org/abs/2310.05737)

---

## Scaling law 实验的核心 take-away

### Data scaling（Table 2）

固定 LLaMA-2 backbone，变 data size:

| #Inst | MotionLib-eval R@1 | MotionLib-eval FID |
|---|---|---|
| 0.02M (HumanML3D) | 0.059 | 29.643 |
| 0.08M (Motion-X) | 0.118 | 21.593 |
| 0.5M (MotionLib-0.5) | 0.164 | 9.146 |
| 1.2M (MotionLib-full) | 0.171 | 6.632 |

FID 从 29.6 降到 6.6，R@1 从 0.059 升到 0.171。**data scaling 在 OOD 上效果非常显著**。

### Model scaling

固定 1.2M data，变 model:

| Decoder | #Param | MotionLib-eval R@1 | MotionLib-eval FID |
|---|---|---|---|
| GPT-2 | 355M | 0.166 | 6.936 |
| LLaMA-2 | 7B | 0.171 | 6.632 |
| LLaMA-3 | 8B | 0.173 | 6.029 |
| LLaMA-2 | 13B | 0.185 | 6.221 |

model scaling 也有用，但比 data scaling 温和。从 355M 到 13B，R@1 才涨 0.019。

### OOD 测试是关键

Table 4 的 UNSEEN-90K 实验:

| Train Set | R@1 | FID |
|---|---|---|
| HumanML3D | 0.034 | 82.674 |
| MotionX | 0.051 | 70.547 |
| MotionLib-#11 | 0.098 | 11.930 |

**这才是百万数据真正的价值**。在 OOD 上，HumanML3D / Motion-X 训练的模型基本废了（FID 70+），MotionLib 训练的 FID 直接降到 11.93。这跟 LLM 在 OOD task 上的表现一个道理——大规模 diverse 数据带来 generalization。

### SOTA 对比

Table 3 上 Being-M0-LFQ 在 HumanML3D 上 R@1 = 0.528，超过 T2M-GPT、MoMask、MotionGPT-v2 等。FID 0.141 不如 specialist model T2M-GPT 的 0.040，但作为 generalist 已经非常有竞争力。

---

## 几个 negative result 的 insight

这些 ablation 反而更有教育意义:

### LoRA 失败（Table 10）

| Method | R@1 | FID |
|---|---|---|
| LoRA | 0.157 | 9.287 |
| Full-param | 0.166 | 6.936 |

intuition: motion token 对 LLM 是全新的 vocabulary，需要大量参数更新来学这些 token 的 embedding 和 distribution。LoRA 的 low-rank 限制不够。这个跟 [Qwen-VL](https://arxiv.org/abs/2308.12966) 的经验一致——新模态 token 需要 full-param。

### Description masking 有害（Table 12）

| Mask | R@1 | FID |
|---|---|---|
| with desc mask | 0.388 | 0.680 |
| w/o desc mask | 0.466 | 0.101 |

不 mask 文本反而好。直觉: 保留 text 信号防止 catastrophic forgetting of language understanding，也减少 motion pattern overfit。

### Encoder-decoder 输给 decoder-only（Table 13）

| Arch | R@1 | FID |
|---|---|---|
| T2M-GPT (enc-dec) | 0.161 | 7.085 |
| GPT-2 (dec-only) | 0.166 | 6.936 |

T2M-GPT 用冻结的 CLIP text encoder，限制了对 motion-specific 语言的理解。decoder-only 联合训 text + motion tokens，对齐更好。

### Evaluator 本身就有问题（Table 16）

用 Motion-X-trained evaluator 评 HumanML3D，性能反而比 HumanML3D-trained evaluator 差。原因是 motion autoencoder 太小，trained on 20K motions，泛化不行。

这跟 [TMR (Petrovich et al.)](https://arxiv.org/abs/2309.13700) 和 [Voas et al.](https://arxiv.org/abs/2307.12028) 的发现一致: **当前 motion metric 与 human perception 没完全对齐**。这是个 open problem。

---

## 跟 LLM 历史类比

paper 自己也暗示了，MotionLib 之于 motion 就像 ImageNet 之于 CV。我觉得更准确的说法是:

- **2012 年 ImageNet**: 大数据 + AlexNet，证明 deep learning work
- **2020 年 GPT-3**: 大数据 + 大模型 + scaling law，证明 scale 是王道
- **2024 年 Being-M0**: 大数据 + LLM + tokenizer scaling，第一次在 motion 上跑通 scaling law

motion generation 正处于 image classification 在 2012 年的位置。paper 提供了三个 scaling axes:

1. **Data axis**: 1.2M sequences，15× larger
2. **Tokenizer axis**: 1024 → 16384 codes，2D part-level
3. **Model axis**: 355M → 13B

关键 insight: **这三个 axis 不是独立的**。2D-LFQ 的优势在大数据上才显现（Table 8），model scaling 的收益需要 data 才能体现。这跟 [Chinchilla](https://arxiv.org/abs/2203.15556) 说的 data-model 必须联合 scale 一个道理。

---

## 我觉得的几个问题

1. **数据 bias**: 698.5K 来自 Kinetics-700，都是 sports/fitness 类，可能 bias
2. **Refine policy 在剧烈运动上失败**: 用 small weight 是权宜之计，物理一致性打折扣
3. **Codebook 还能更大**: $2^{16}$ 相比 LLM 的 32K-128K vocabulary 还有空间
4. **没给 inference speed**: 13B 模型能不能实时？SMPL-D135 的 FPS > 100 是 feature 优势，但 model 太大可能也不行
5. **Multi-person motion 没充分用**: 数据有 multi-person，但模型还是单人生成
6. **Metric robustness**: paper 自己承认 evaluator 不行，这其实是整个领域的 bottleneck

---

## 未来方向猜测

顺着 paper 的思路往下想:

1. **Codebook 继续扩大**: $2^{18}$、$2^{20}$ + 3D / 4D quantization
2. **更好的 evaluator**: TMR-style 大规模 retrieval model 当 metric
3. **Static image 数据利用**: paper 试了 600K images 重复 60 帧，效果有限但值得探索
4. **Multi-person 场景**: 真正利用 multi-person data
5. **Physical realism**: RL refine policy 升级，处理更剧烈运动
6. **Real-time application**: 模型压缩 + SMPL-D135 高 FPS 已经铺路

---

## Final intuition

这篇 paper 给我的最大启示是: **motion 终于可以走 scaling 路线了**。之前所有人都在小数据上卷 architecture、卷 loss function，现在终于有人把 data、tokenizer、model 三个 axis 同时 scale up 跑通了一遍。

类比一下，这就像 motion generation 刚刚拿到了自己的 ImageNet + GPT-3 时刻。接下来几年，follow-up 工作大概率会沿着三轴继续 push——更大 codebook、更多 data、更大 model，同时解决 evaluator、physical realism、real-time 这些 open problem。

paper 提到的 negative results（LoRA 失败、metric 不行、masking 有害）也非常有价值，给了社区很多 follow-up 的 hook。

参考汇总:
- [Being-M0 Project Page](https://beingbeyond.github.io/Being-M0/)
- [LlamaGen (LFQ 灵感来源)](https://arxiv.org/abs/2310.05737)
- [FSQ (Mentzer et al.)](https://arxiv.org/abs/2309.15505)
- [LLaVA](https://arxiv.org/abs/2304.08485)
- [Chinchilla scaling law](https://arxiv.org/abs/2203.15556)
- [WHAM](https://arxiv.org/abs/2404.05198)
- [MotionLib 相关: HumanML3D](https://arxiv.org/abs/2205.10839) | [Motion-X](https://arxiv.org/abs/2307.00818) | [TMR](https://arxiv.org/abs/2309.13700)

希望这个版本更直觉一些，Andrej！

---

# Scaling Large Motion Models with Million-Level Human Motions 深度讲解

Andrej, 这篇 paper 是大规模 motion generation 的一个里程碑，从 scaling 的角度来系统研究 motion model，思路非常清晰。让我从多个层面来 build your intuition。

## 1. 核心问题与动机

motion generation 领域长期的瓶颈是 **data scarcity**。视觉领域有 ImageNet (1.2M images)、LAION-5B 这种规模的数据集，而 motion 领域最大的 HumanML3D 只有 29K sequences、Motion-X 也才 81K。paper 的核心 question 是: **"Can scaling the large motion model and data benefit motion generation?"** 这其实是把 LLM 的 scaling law philosophy 移植到 motion modality。

paper 同时识别了 motion representation 的两大缺陷:
1. **Information Loss**: 传统 VQ 把每个 motion state $m_i \in \mathbb{R}^D$ 压缩成 1D embedding，但 motion state 编码了 heterogeneous features (joint position、velocity、foot contact)，单 1D embedding 不足以表达
2. **Limited Codebook Size**: 小 codebook + codebook collapse 问题

相关参考:
- LLM scaling law: [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)
- VQ-VAE 原始论文: [Neural Discrete Representation Learning (van den Oord et al., 2017)](https://arxiv.org/abs/1711.00937)
- Codebook collapse 问题分析: [Addressing "Codebook Collapse" in VQ-VAE](https://arxiv.org/abs/2010.11057)

## 2. MotionLib: 百万级 motion 数据集

### 2.1 数据规模对比

| Dataset | SEQ NUM | TEXT NUM | HOURS | MOTION | TEXT | PERSON |
|---|---|---|---|---|---|---|
| KIT | 5.7K | 5.7K | 11.2 | B | body | single |
| HumanML3D | 29.2K | 89K | 28.6 | B | body | single |
| Motion-X | 81.1K | 142K | 144.2 | B,H,F | body | single |
| MotionVerse | 320K | 373K | - | B,H,F | body | single |
| **MotionLib** | **1.21M** | **2.48M** | **1456.4** | B,H | hier | single & multi |

MotionLib 是至少 15× larger than existing counterparts，1456.4 小时的 motion 数据，第一次让 motion 数据集规模接近 ImageNet 级别。

### 2.2 数据收集 Pipeline

整个 pipeline 包括 4 个关键步骤:

**Step 1: Million-Level Motion Collection**
- 从 20M+ videos 中筛选（包括 Kinetics-700、YouTube、NTU-RGBD-120、BEDLAM、GTA-Human 等）
- 使用 ViTPose (Xu et al., 2022) 检测 2D keypoints 过滤无 human 的 video
- 使用 WHAM (Shin et al., 2024) 回归 SMPL 参数，输出 world coordinate system 的 3D motion（不是 camera coordinate）

参考:
- [WHAM: Reconstructing World-Grounded Humans with Accurate 3D Motion](https://arxiv.org/abs/2404.05198)
- [ViTPose](https://arxiv.org/abs/2204.07671)
- [SMPL model](https://smpl.is.tue.mpg.de/)

**Step 2: Hierarchical Motion Descriptions**
这是 paper 的一个亮点，使用 Gemini-1.5-Pro 生成两层 text:
- **Part-level**: 为每个 body part（如 left arm）单独描述
- **Body-level**: 1-3 句话总结 whole body 运动

这种 hierarchical 结构借鉴了 [Pi et al., 2023 - Hierarchical Generation of HOI](https://arxiv.org/abs/2307.11598)，让 LLM 能更好地 part-level 控制 motion。

参考 Gemini-1.5-Pro: [Gemini 1.5 Report](https://arxiv.org/abs/2403.05530)

**Step 3: Motion and Description Refinement**
- 使用 [Sarándi et al., 2023] 的 3D keypoint estimator 做 global motion optimization
- 训练 RL-based policy $\pi_{\text{refine}}$ (基于 [Luo et al., 2023 - Perpetual Humanoid Control](https://arxiv.org/abs/2310.01886)) 让 motion 满足物理定律（balance、foot-grounding、no foot-sliding）
- 对剧烈运动的 sample 加 small weight（半监督思想）
- Description 用 GPT-4o 做 cross-level refinement

**Step 4: Short Boundary Detection + Occlusion/Blur Filtering**
- 使用 [SAM (Segment Anything)](https://arxiv.org/abs/2304.02643) 生成 segmentation mask 检测 occlusion
- 通过 trajectory smoothing 处理 motion blur

### 2.3 数据质量评估

Table 9 显示 text 质量评估:

| Eval Strategy | HumanML3D | MotionX | MotionLib |
|---|---|---|---|
| Text-only | 1.386 | 1.703 | 3.837 |
| Visual-text align | 3.081 | 2.252 | 3.823 |

MotionLib 显著优于其他 dataset。Score 由 GPT-4o (5-point scale) 给出，避免用同一个 LMM (Gemini) 既生成又评估的 bias。

## 3. Being-M0 架构

### 3.1 Overall Framework

Being-M0 的设计哲学：**把 motion 当作 "foreign language"** 来学习，借鉴 [Luo et al., 2020 - UniVL](https://arxiv.org/abs/2002.06353) 和 [Zhang et al., 2024c - Pixels to Tokens](https://arxiv.org/abs/2410.02155) 的思路。

Architecture 是经典的 **Motion Tokenizer + LLM backbone** 结构:
1. Motion tokenizer: 把 $\mathcal{M} = \{m_1, ..., m_T\}$ 编码为 token embeddings $\mathcal{V} = \{v_1, ..., v_n\} \in \mathbb{R}^{n \times d}$
2. Codebook 中 K 个 discrete codes 作为 LLM 的 **additional vocabulary**
3. 特殊 token `<mot>` 和 `</mot>` 标记 motion sequence 的起止
4. Decoder-only causal transformer 自回归生成

### 3.2 Training Objective

负 log-likelihood 优化:

$$\mathcal{L}(\Theta) = -\sum_{j=1}^{L} \log P_\Theta(y_j | \text{desc}, \hat{y}_{1:j-1})$$

变量解释:
- $\Theta$: 模型参数
- $L$: target sequence 长度
- $y_j$: 第 $j$ 个 target token
- $\hat{y}_{1:j-1}$: 之前生成的 input tokens
- $\text{desc}$: input description（可以 empty）
- $P_\Theta$: 在参数 $\Theta$ 下 token 的条件概率

### 3.3 Two-Stage Training

借鉴 [LLaVA (Liu et al., 2023)](https://arxiv.org/abs/2304.08485):
- **Stage 1 - Motion-Text Alignment Pretraining**: 在整个 MotionLib 上学习 motion-text 基础对齐
- **Stage 2 - Motion Instruction Tuning**: 用 250+ instruction templates + Gemini-Pro refinement，构建 900K instruction-following dataset

## 4. MotionBook: 核心 contribution 详解

MotionBook 是这篇 paper 的灵魂，包括两个核心组件:

### 4.1 Lossless Motion Feature: SMPL-D135

paper 对比了 5 种 feature:

| Feature | Dims | Composition |
|---|---|---|
| H3D-Format | 263 | position(63) + rotation(126, from IK) + velocity(66) + foot_contact(4) + root(4) |
| SMPL-D130 | 130 | 6D rotation(126) + root(4) |
| SMPL-D135 | 135 | 6D rotation(126) + root(9) |
| SMPL-D263 | 263 | SMPL-D130 + position(133) + foot_contact(4) |
| SMPL-D268 | 268 | SMPL-D135 + position(133) + foot_contact(4) |

**SMPL-D135** 的结构:
- **Root (9D)**:
  - 6D rotation $\mathbf{r}_{\text{rot}} \in \mathbb{R}^6$ (使用 6D rotation representation, [Zhou et al., 2019](https://arxiv.org/abs/1812.07035))
  - 2D XZ-plane velocity $\mathbf{r}_{xz}^v \in \mathbb{R}^2$
  - 1D height $r^y \in \mathbb{R}$
- **Body joints (126D)**:
  - 21 key body joints $\times$ 6D rotation vectors $\mathbf{j}^r \in \mathbb{R}^{21 \times 6}$

**关键 insight**: H3D-Format 通过 IK 从 position 反推 rotation，这有两个问题:
1. **信息丢失**: SMPL 原始的 rotation 信息被丢弃，IK 不是 invertible
2. **计算昂贵**: 实时应用（如 game animation）需要快速 recovery，IK 太慢

SMPL-D135 直接编码原始 SMPL rotation，**lossless** 同时更 compact。Table 7 显示 SMPL-D135 在 R@1 上 0.529 > H3D-Format 0.514，FPS > 100 vs 0.41。

参考:
- [On the Continuity of Rotation Representations (6D rotation)](https://arxiv.org/abs/1812.07035)
- [SMPL: A Skinned Multi-Person Linear Model](https://smpl.is.tue.mpg.de/)

### 4.2 2D-LFQ: 2D Lookup-Free Quantization

这是 paper 最 technical 的部分，借鉴 [Mentzer et al., 2023 - FSQ (Finite Scalar Quantization)](https://arxiv.org/abs/2309.15505) 和 [Yu et al., 2023 - Language Model Beats Diffusion](https://arxiv.org/abs/2310.05737)。

#### 4.2.1 从 1D 到 2D

传统 motion tokenizer:
$$\mathcal{M} \in \mathbb{R}^{T \times D} \to \text{Encoder} E \to \mathbb{R}^{\lfloor T/\alpha \rfloor \times d}$$

每个 timestamp 用 1D embedding 表示整个 motion state，信息丢失严重。

2D-LFQ 改成:
$$\mathcal{M} \in \mathbb{R}^{T \times D \times 1} \to \text{Encoder} E \to \mathbb{R}^{\lfloor T/\alpha \rfloor \times P \times d}$$

- $P$: 把 motion feature 分成 $P$ 个 components (root orientation、joint rotation、foot contact 等)
- 每个 body part 独立 tokenize，提供 **part-level resolution**

#### 4.2.2 Lookup-Free Quantization

传统 VQ codebook: $\mathbb{C} \in \mathbb{R}^{K \times d}$，需要 nearest neighbor lookup，codebook 大了易 collapse。

2D-LFQ 把 codebook 替换为 **integer set**:
$$\mathbb{C} = \times_{i=1}^{d} C_i, \quad C_i = \{-1, 1\}$$

每个 $C_i$ 只有 2 个值（-1 或 1），$d = \log_2 K$。给定 feature vector $z \in \mathbb{R}^d$，每个维度量化为:

$$Q(z_i) = \arg\min_{c_{ik}} \|z_i - c_{ik}\| = -\mathbb{1}\{z_i \leq 0\} + \mathbb{1}\{z_i > 0\}$$

变量解释:
- $z_i$: feature vector 第 $i$ 维
- $c_{ik}$: $C_i$ 中第 $k$ 个值（-1 或 1）
- $\mathbb{1}\{\cdot\}$: indicator function
- 简单地说: $z_i > 0$ 量化为 1，$z_i \leq 0$ 量化为 -1

**Token index** 计算:
$$\text{Index}(z) = \sum_{i=1}^{d} 2^{i-1} \mathbb{1}\{z_i > 0\}$$

这是一个二进制编码，每个维度贡献一个 bit。如果 $d = 16$，codebook 大小 $= 2^{16} = 16384$。

#### 4.2.3 关键 Insight

1. **No embedding lookup**: 不需要存储 codebook embedding，直接用 sign 函数量化
2. **Codebook 可以任意大**: $|C| = 2^d$，扩大 $d$ 即可
3. **Better codebook utilization**: 每个 code 被均匀使用，没有 collapse
4. **Lower embedding dim**: $d$ 通常远小于传统 VQ 的 $d$ (如 512)，限制单 token 表达力，强制充分利用 codebook

#### 4.2.4 训练 Loss

组合 loss:
- **Reconstruction loss**: 重建 motion
- **Perceptual loss**: 感知损失
- **Commitment loss**: 让 encoder output 接近 quantized value
- **Entropy penalty**: 促进 codebook 利用率（[Yu et al., 2023](https://arxiv.org/abs/2310.05737)）
- **No GAN loss**: paper 发现 GAN loss 不稳定

### 4.3 Codebook 实验对比

Table 8 显示 motion reconstruction 结果:

| Tokenizer | #Num | #Param | HumanML3D FID | HumanML3D MPJPE | Motion-X FID | Motion-X MPJPE | MotionLib FID | MotionLib MPJPE |
|---|---|---|---|---|---|---|---|---|
| VQ-VAE | 512 | 19.43M | 0.078 | 69.2 | 0.852 | 106.4 | 5.324 | 123.6 |
| RQ-VAE | 512 | 19.43M | 0.052 | 37.5 | 0.568 | 56.9 | 4.026 | 78.2 |
| 2D-LFQ | 16384 | 108.35M | 0.092 | 45.6 | 0.295 | 54.1 | 2.315 | 64.1 |

**关键观察**:
- 在小 dataset HumanML3D 上 2D-LFQ 不一定最好（FID 0.092 > RQ 0.052）
- 在大 dataset (Motion-X, MotionLib) 上 2D-LFQ **显著更好**: MotionLib FID 2.315 vs RQ 4.026
- 这意味着 **2D-LFQ 的优势在 scale up 时显现**

Figure 4 显示 codebook size 扩大时的表现:
- VQ、RQ 在 codebook > $2^{10}$ 时 performance 开始下降（codebook collapse）
- 2D-LFQ 持续 improve，codebook utilization 持续上升

Table 14 显示 2D vs 1D LFQ 的 ablation:

| Tokenizer | #Num | #Param | HumanML3D FID | MotionLib FID | MotionLib MPJPE |
|---|---|---|---|---|---|
| 1D-LFQ | 16384 | 19.43M | 3.85 | 10.358 | 80.1 |
| 2D-LFQ | 16384 | 108.35M | 1.769 | 7.853 | 64.1 |

2D 比 1D 显著好，验证了 **part-level encoding** 的必要性。

## 5. Scaling Law 实验

### 5.1 Data Scaling

Table 2 (paper 中关键 scaling 实验表):

固定 LLaMA-2 backbone, 变 data size:

| #Inst. | Motion-X-eval R@1 | Motion-X-eval FID | MotionLib-eval R@1 | MotionLib-eval FID |
|---|---|---|---|---|
| 0.02M (HumanML3D) | 0.216 | 47.538 | 0.059 | 29.643 |
| 0.08M (Motion-X) | 0.472 | 0.166 | 0.118 | 21.593 |
| 0.5M (MotionLib-0.5) | 0.468 | 0.178 | 0.164 | 9.146 |
| 1.2M (MotionLib-full) | 0.475 | 0.156 | 0.171 | 6.632 |

**Data scaling 在 MotionLib-eval 上效果显著**: FID 从 29.6 降到 6.6，R@1 从 0.059 升到 0.171。

**Cross-domain transfer**: 在 Motion-X 上 pretrain 后，Motion-X-eval 性能也 improve（小 dataset 上 pretrain 大 dataset 的优势）。

### 5.2 Model Scaling

固定 1.2M data, 变 model size:

| Decoder | #Param. | Motion-X-eval R@1 | MotionLib-eval R@1 | MotionLib-eval FID |
|---|---|---|---|---|
| GPT-2 | 355M | 0.472 | 0.166 | 6.936 |
| LLaMA-2 | 7B | 0.475 | 0.171 | 6.632 |
| LLaMA-3 | 8B | 0.486 | 0.173 | 6.029 |
| LLaMA-2 | 13B | 0.491 | 0.185 | 6.221 |

**Model scaling 效果显著但不如 data scaling 剧烈**。R@1 从 0.472 → 0.491 (Motion-X)，0.166 → 0.185 (MotionLib)。

### 5.3 SOTA Comparison on HumanML3D

Table 3:

| Method | Decoder | R@1↑ | R@3↑ | FID↓ | MMDist↓ |
|---|---|---|---|---|---|
| MLD | - | 0.481 | 0.772 | 0.473 | 3.196 |
| T2M-GPT | - | 0.525 | 0.811 | 0.040 | 2.943 |
| MoMask | - | 0.521 | 0.807 | 0.045 | 2.958 |
| MotionGPT-v2 | LLaMA3.1-8B | 0.496 | 0.782 | 0.191 | 3.080 |
| **Being-M0-VQ** | LLaMA2-13B | 0.519 | 0.803 | 0.166 | 2.964 |
| **Being-M0-LFQ** | LLaMA2-13B | **0.528** | **0.820** | 0.141 | **2.875** |

Being-M0-LFQ 在 R@1、R@3、MMDist 上达到 SOTA。FID 不如 T2M-GPT (specialist) 但作为 generalist 模型已经很有竞争力。

### 5.4 OOD Generalization

Table 4 (UNSEEN-90K 测试):

| Train Set | R@1↑ | R@3↑ | FID↓ |
|---|---|---|---|
| HumanML3D | 0.034 | 0.112 | 82.674 |
| MotionX | 0.051 | 0.141 | 70.547 |
| MotionLib-#11 | 0.098 | 0.218 | 11.930 |

**关键 insight**: 在 OOD 测试上，MotionLib 训练的模型 FID 从 70+ 降到 11.93，**这是大规模 diverse 数据的核心价值**。HumanML3D / Motion-X 在 OOD 上基本失效。

## 6. 重要 Ablations

### 6.1 LoRA vs Full-Parameter Fine-tuning (Table 10)

| Train Type | R@1 | R@3 | FID |
|---|---|---|---|
| LoRA | 0.157 | 0.354 | 9.287 |
| full-param | 0.166 | 0.375 | 6.936 |

LoRA 在 motion 任务上失败。**Paper 解释**: motion token 是 LLM vocabulary 的新 token，需要 large 参数更新来学习这些新 token 的 embedding 和 distribution，LoRA 的 low-rank 限制不足以应对。这与 [Qwen-VL](https://arxiv.org/abs/2308.12966) 等多模态模型的经验类似——多模态 token 需要更多参数变化。

### 6.2 From Scratch vs Fine-tuning (Table 11)

| #Inst | From Scratch | R@1 | R@3 | FID |
|---|---|---|---|---|
| 0.02M | Yes | 0.042 | 0.116 | 17.932 |
| 0.02M | No | 0.213 | 0.426 | 47.319 |
| 0.08M | Yes | 0.461 | 0.784 | 0.116 |
| 0.08M | No | 0.468 | 0.792 | 0.083 |

有趣的是在小数据 (0.02M) 时 fine-tune 反而 FID 差（47.3 vs 17.9），可能是 overfit 文本 prior；在大数据上 fine-tune 全面更好。

### 6.3 Description Masking (Table 12)

| Mask Strategy | R@1 | R@3 | FID |
|---|---|---|---|
| with description mask | 0.388 | 0.650 | 0.680 |
| w/o description mask | 0.466 | 0.752 | 0.101 |

**不 mask description 更好**，防止 catastrophic forgetting of text understanding + 减少 overfit。这与多模态学习中的 [FLAVA](https://arxiv.org/abs/2112.04482) 类似思路——保持 text 信号 always present。

### 6.4 Encoder-Decoder vs Decoder-Only (Table 13)

| Arch | Model | #Param. | R@1 | FID |
|---|---|---|---|---|
| enc-dec | T2M-GPT | 380M | 0.161 | 7.085 |
| dec-only | GPT-2 Medium | 355M | 0.166 | 6.936 |

参数量相近情况下 decoder-only 更好。原因: T2M-GPT 用 "CLIP + random-init decoder"，CLIP 的 text encoder 不能 fine-tune，无法理解 motion-specific 语言。Decoder-only 联合训练 text + motion tokens，对齐更好。

### 6.5 Hierarchical Description (Table 6)

| Train Text | R@1 | R@3 | FID |
|---|---|---|---|
| single-level | 0.162 | 0.371 | 7.018 |
| hierarchical | 0.166 | 0.375 | 6.936 |

Hierarchical text 效果更好，但提升不大。我推测: MotionLib-eval 上提升有限因为 evaluation text 是 body-level 的。在 part-level 控制 task 上 hierarchical 应该有更大优势。

### 6.6 Instruction Tuning (Table 5)

| Train Set | R@1 | R@3 | FID |
|---|---|---|---|
| Pretrain only | 0.471 | 0.788 | 0.103 |
| Instruction tuning | 0.488 | 0.821 | 0.093 |

Instruction tuning 提升 R@1 +0.017, FID -0.010，让模型更 user-friendly。

### 6.7 Static & Synthetic Data (Table 15)

加入 600K static images (重复 60 frames) 的 ablation。结论: 静态数据对 dynamic motion 提升有限，但可探索 future 方向。

## 7. Convergence Speed (Figure 11 LEFT)

- GPT-2、LLaMA2-7B、LLaMA3-8B 都在 ~200 epochs 收敛
- 大模型收敛更快
- 但相比 LLaVA 等视觉多模态模型，motion model 收敛慢得多
- **Paper 解释**: motion tokenizer 只有 1024 codebook，表达能力有限。这正是 2D-LFQ 的 motivation——扩大 codebook 容量加速对齐

## 8. Evaluation Metric Limitation (Table 16)

一个 deep observation: 当用 Motion-X-trained evaluator 评估 HumanML3D 测试集，性能反而下降！原因是 evaluator 的 motion autoencoder 在小数据上训练，泛化能力差。

这呼应了 [Petrovich et al., 2023 - TMR](https://arxiv.org/abs/2309.13700) 和 [Voas et al., 2023 - What is the best metric](https://arxiv.org/abs/2307.12028) 的发现: 当前 motion metric 与 human perception 不完全对齐，是 motion 领域的**重要 open problem**。

## 9. 整体直觉与关键 insights

### 9.1 Three Pillars of Scaling Motion Models

1. **Data scaling** (15× data): 解决 OOD generalization，FID 从 70+ → 11.93
2. **Model scaling** (355M → 13B): 稳健提升但回报递减
3. **Tokenizer scaling** (1024 → 16384 codes + 2D): 解决 representation bottleneck，让 model 能充分利用大 codebook

### 9.2 Why 2D-LFQ Works

类比 [LlamaGen (Yu et al., 2023)](https://arxiv.org/abs/2310.05737) 在 image 上的成功:
- Lookup-free 避免了 codebook collapse
- 2D 让每个 body part 独立编码，提供 part-level 控制
- Binary 量化简单且 stable
- Codebook 大小可指数扩展

### 9.3 与 LLM 的对比

- LLM vocabulary ~32K tokens（BPE）
- Motion 之前只有 512-1024 tokens
- 2D-LFQ 扩到 16384，但仍远小于 LLM
- 未来方向: 更大 codebook + 更长 context

### 9.4 与 ImageNet 的类比

paper 多次提到 MotionLib 是 "first large T2M dataset comparable in scale to visual benchmarks like ImageNet"。这其实暗示了 motion generation 正处于 ImageNet 之于 image classification 的阶段——足够大的 dataset + 通用模型 + scaling law。

### 9.5 多模态 LLM 的演化路径

paper 走的是 [LLaVA](https://arxiv.org/abs/2304.08485) / [BLIP-2](https://arxiv.org/abs/2301.12597) / [InstructBLIP](https://arxiv.org/abs/2305.06500) 的路径:
1. 把 motion tokenize 成 discrete tokens
2. 加入 LLM vocabulary
3. 用 instruction tuning 对齐
4. 用 scaling 推进 performance

参考:
- [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485)
- [BLIP-2](https://arxiv.org/abs/2301.12597)
- [InstructBLIP](https://arxiv.org/abs/2305.06500)

## 10. 与其他相关工作对比

### 10.1 Tokenizer 比较

- **VQ-VAE**: 最早期的 discrete token，[van den Oord 2017](https://arxiv.org/abs/1711.00937)
- **RQ-VAE**: residual quantization，[Lee et al., 2022](https://arxiv.org/abs/2203.01841)
- **HQ**: hierarchical pyramid codes, [You et al., 2022](https://arxiv.org/abs/2211.16734)
- **FSQ**: finite scalar quantization, [Mentzer et al., 2023](https://arxiv.org/abs/2309.15505)
- **LFQ**: lookup-free for image, [Yu et al., 2023](https://arxiv.org/abs/2310.05737)
- **H2VQ**: hierarchical for hand+body, [Lu et al., 2023 - HumanTomato](https://arxiv.org/abs/2310.12978)
- **2D-LFQ**: 本文，motion 专属

### 10.2 Motion Generation 演化

- **早期 deterministic**: [Fragkiadaki et al., 2015](https://arxiv.org/abs/1508.00785)
- **GAN-based**: [Wang et al., 2020](https://arxiv.org/abs/2005.05526)
- **VAE-based**: [Aliakbarian et al., 2020](https://arxiv.org/abs/2004.04993)
- **Diffusion-based**: MLD, MotionDiffuse, ReMoDiffuse, Fg-T2M++
- **Autoregressive**: T2M-GPT, DiverseMotion, MoMask
- **LLM-based**: MotionGPT, MotionLLM, AvatarGPT, MotionGPT-v2, **Being-M0**

### 10.3 多模态 LLM

- **mPLUG-Owl**: [Ye et al., 2023](https://arxiv.org/abs/2304.14178)
- **LLaVAR**: [Zhang et al., 2023c](https://arxiv.org/abs/2306.17107)
- **UniCode**: [Zheng et al., 2024](https://arxiv.org/abs/2403.09072)
- **VideoOrion**: [Feng et al., 2024](https://arxiv.org/abs/2411.16156)

## 11. 个人 critique 与 future directions

### 11.1 数据来源 Concerns

- 1456 小时 motion 来自 20M 视频，意味着大量数据可能 noisy
- RL refine policy 在剧烈运动上失败，用 small weight 是权宜之计
- 大约 698.5K 来自 Kinetics-700——这些是 sports/fitness 类，可能 bias

### 11.2 评估公平性

- 在 HumanML3D 上用 HM3D-Format feature（公平比较）
- 在 MotionLib-eval 上用 SMPL-D135（自家 metric），但 MotionLib-eval 的 evaluator 也只 trained on MotionLib，可能 underrepresent OOD performance
- FID 依赖的 motion autoencoder 本身就 trained on limited data

### 11.3 Codebook 大小的局限

- 2D-LFQ codebook 是 $2^{16} = 16384$，但 LLM vocabulary 通常 32K-128K
- 未来: $2^{18}$、$2^{20}$ codebook + 4D quantization?

### 11.4 Computational Cost

- 8×A800 GPU，100 epochs，full-parameter fine-tune
- 13B 模型训练成本不菲，限制了 research community 复现
- Paper 没给 inference speed 数据

### 11.5 Future Directions (paper 自己也提到)

1. **更鲁棒的 evaluation metric**: TMR-style retrieval-based metric
2. **更大的 codebook**: $2^{20}$ 或 beyond
3. **Static data 探索**: 600K images 提取 pose 的潜力
4. **Multi-person motion**: 数据已有，模型没充分利用
5. **Real-time generation**: SMPL-D135 的 FPS > 100 是 step forward

## 12. 总体 Intuition Summary

这篇 paper 的核心 narrative 是: **motion generation 需要跟 LLM 一样走 scaling 路线**。三个关键 components:

1. **Data scaling (MotionLib)**: 1.2M motion sequences，15× larger，让 OOD generalization 成为可能
2. **Tokenizer scaling (2D-LFQ)**: 16384 codes (32× 传统) + 2D part-level encoding + lookup-free 防 collapse
3. **Model scaling (Being-M0)**: 355M → 13B，stable but diminishing return

三个 scaling axes 不是独立的——**2D-LFQ 的优势在大数据上才显现**，**model scaling 的收益需要 data 才能体现**。这与 [Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) 的 data-vs-model scaling 关系类似: 必须联合 scale。

paper 也提出了几个深刻的 negative results:
- LoRA 在 motion token 上失败（low-rank 不足以学习新 vocab）
- Description masking 伤害 performance（应该保留 text 信号）
- Encoder-decoder (T2M-GPT) 比 decoder-only 差（CLIP 冻结限制太大）
- 当前 metric (R-Precision, FID) 不够 robust（evaluator 数据少）

参考 paper 自己的 project page: [Being-M0](https://beingbeyond.github.io/Being-M0/)

希望这个 build 了一些 intuition，Andrej！motion generation 这个领域刚进入 scaling 阶段，相当于 2012 年的 image classification——大模型 + 大数据 + 强 tokenizer 的三轴 scaling 才刚刚开始。
