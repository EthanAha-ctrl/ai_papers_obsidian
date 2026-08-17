---
source_pdf: MiniDrive More Efficient Vision-Language Models with Multi-Level 2D.pdf
paper_sha256: 6e36e18b3fd1c0daa7a614879a125bef6e0979c241923366c3a11452b93aa2b8
processed_at: '2026-08-05T18:31:29-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MiniDrive 人话版

## 一句话版本

作者把自动驾驶 VLM 从"7B 巨兽"砍到"83M 小怪兽"，靠三招：**用大卷积核 CNN 替代 ViT、把 feature map 反着 flatten 成 16 个 token、让 visual token 先看一眼 question 再进 LLM**。

---

## 为什么这 paper 值得看

现在自动驾驶 VLM 这条赛道全在卷"谁的 backbone 更大"——DriveGPT4 用 LLaMA-7B、DriveMLM 用 LLaMA-7B + ViT-g/14、DriveLM-Agent 用 3.96B。FLOPs 动辄 300B+，车端 SoC 根本跑不动。EM-VLM4AD 是第一个尝试做轻量化的（345M），但效果比大模型差一截。

MiniDrive 的定位很清晰：**能不能做到 100M 量级、单卡 4090 能训、效果还比 7B 好？** 答案是可以，但你要在 tokenization 上做文章，而不是堆参数。

参考：EM-VLM4AD https://arxiv.org/abs/2403.19838

---

## 核心架构，用大白话讲

整个 pipeline 就干一件事：**把多张图压成 96 个有语义的 token，让 T5-Small 像读一句话一样读图**。

```
6 张多视角图
    ↓
UniRepLKNet (frozen) → 每张图一张 feature map
    ↓
FE-MoE → 每张图 16 个 token (通道当 token，空间当 dim)
    ↓
DI-Adapter → token 根据 question 调整一下
    ↓
拼上 text embedding
    ↓
T5-Small → 输出回答
```

就这么简单。没有 Q-Former，没有 Perceiver Resampler，没有 256 个 patch token 灌进 LLM 的暴力做法。

---

## 三招拆解

### 第一招：用 CNN 当 vision encoder

主流 VLM 全用 ViT（CLIP-ViT、ViT-g/14），把图切成 patch 再 self-attention。但自动驾驶图像有个特点——**大量局部强相关、全局弱相关的内容**：车道线就是一条线、车边缘就是一条边、锥桶就是一个小色块。用全局 self-attention 去建模这些反而浪费。

UniRepLKNet 用 **31×31 的大卷积核**，一次卷积就能覆盖很大区域，不需要堆深，FLOPs 比 ViT 低一个量级。paper 选这个 backbone 的真实原因我猜是：**CNN 输出的 feature map 天然是 (C, H, W) 形状，方便后面做"反着 flatten"的 trick**。ViT 输出的是 (N_patch, D) 的 sequence，反而不好做这个操作。

参考 UniRepLKNet: https://arxiv.org/abs/2312.17171  
参考 RepLKNet 前作: https://arxiv.org/abs/2203.06717

---

### 第二招：FE-MoE 把 feature map "反着" flatten

这是最反直觉的设计，也是这篇 paper 最有意思的地方。

**主流做法**（LLaVA / BLIP-2）：ViT 输出 (N_patch=256, D=1024)，把 256 当 token 数，1024 当 token dim。每张图 256 个 token。

**MiniDrive 做法**：CNN 输出 (C=1280, H=7, W=7)，先用 deconv 把 spatial 放大、channel 缩小，变成 (C'=16, H'=14, W'=14)，然后 **flatten 时把 C' 当 token 数，把 H'×W'=196 当 token dim**。

公式 (Eq 3):

$$F_1 \in \mathbb{R}^{c \times h \times w} \rightarrow F_2 \in \mathbb{R}^{c' \times h' \times w'}$$

- $c=1280, h=7, w=7$：输入 feature map
- $c'=16, h'=14, w'=14$：expert 输出
- $c\downarrow$：通道减少（1280→16）
- $h\uparrow, w\uparrow$：空间增大（7→14）

结果：**每张图只有 16 个 token**，6 张多视角图总共 96 个 token，比 LLaVA 单张图的 256 个 token 还少。

**为什么这样做有道理**：在 NLP 里 token 对应"概念"，不是"像素块"。把 channel 当 token，相当于让每个 token 学到一个"语义通道"——比如一个 token 专门编码"车辆信息"、一个专门编码"车道线"、一个专门编码"交通标志"。这比 ViT 的"patch token = 图像一小块"更接近 NLP 的 token 语义。

**MoE 部分**：4 个 expert 并行跑，Gate 网络给每个 expert 一个权重，加权求和（Eq 5）：

$$V_{\text{moe}} = \sum_{i=1}^{N} W_i \cdot F_i$$

- $N=4$：expert 数
- $W_i$：Gate 给第 $i$ 个 expert 的权重（softmax 归一化）
- $F_i$：第 $i$ 个 expert 的输出

Gate 网络就是 Conv + MaxPool + Linear，输入 feature map，输出 4 维权重。soft routing 不做 top-k 选择，所有 expert 都算，避免 load balancing 问题。

直觉：不同视角（前向 vs 后向）、不同场景（高速 vs 拥挤）适合不同的"特征加工方式"，让模型自己学怎么路由。

---

### 第三招：DI-Adapter 让 visual token 先看 question

传统 VLM 有个普遍问题：**同一张图，无论你问"前方有什么车"还是"红绿灯什么颜色"，visual token 都一样**。所有 cross-modal 对齐的负担全压在 LLM 内部的 cross-attention 上。

DI-Adapter 的 idea 很简单：**在 visual token 进 LLM 之前，先用 text instruction 对它做一次 cross-attention 调整**。

公式 (Eq 6):

$$V' = \text{CrossAttn}(q=V, k=T, v=T)$$

- $V \in \mathbb{R}^{16 \times 512}$：16 个 visual token，做 **query**
- $T \in \mathbb{R}^{l_2 \times 512}$：text instruction 的 token，做 **key** 和 **value**
- $V' \in \mathbb{R}^{16 \times 512}$：融合了 text 上下文的新 visual token

**方向很关键**：用 V 做 query 意思是"每个 visual token 去问 text：你关心我哪部分？"。如果反过来用 text 做 query，就变成"把 visual 信息读到 text 里"，那是 Q-Former 的做法，visual token 仍然是 image-only 的。

然后加 residual（防止 cross-attn 把 visual 信息扭曲过度）：

$$V_{\text{input}} = V + V'$$

最后 $[V_{\text{input}}, T_{\text{input}}]$ 拼起来送进 T5-Small。

**代价**：visual token 不能 cache 了，不同 question 要重新算一遍 cross-attn。但 l1=16 很小，这个开销可以忽略。

---

## 实验数据，看几个关键点

### DriveLM 上的成绩 (Table 1)

| Method | Params | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|
| DriveLM-Agent | 3.96B | 53.09 | 36.19 | 66.79 | 2.79 |
| EM-VLM4AD_Base | 345M | 45.36 | 34.49 | 71.98 | 3.20 |
| **MiniDrive_224** | **83M** | 49.70 | 36.30 | 73.30 | 3.28 |
| **MiniDrive_384** | 137M | 50.20 | 37.40 | 73.50 | 3.32 |

- BLEU-4 输给 DriveLM-Agent（49.70 vs 53.09），但参数少 47 倍
- METEOR / ROUGE-L / CIDEr 全赢，CIDEr 最能反映 caption 质量
- 换 384 分辨率继续涨，说明 vision encoder 还有 headroom

### 计算开销 (Table 2)

| Model | Params | FLOPs | Memory |
|---|---|---|---|
| DriveMLM | 8.37B | 535B | 36 GB |
| DriveGPT4 | 7.3B | 329B | 29.2 GB |
| DriveLM-Agent | 3.96B | 439B | 14.43 GB |
| EM-VLM4AD | 345M | 9.9B | 1.97 GB |
| **MiniDrive_224** | **83M** | **5.9B** | **1.03 GB** |

83M 参数、5.9B FLOPs、1 GB 显存。单卡 4090 24GB 能同时跑好几个 instance 训练。这个量级意味着车端 SoC（Orin ~100 TOPS）能跑到 30+ FPS。

### CODA-LM 单图任务 (Table 4)

| Method | Params | Barrier | Cone | Vehicle | VRU |
|---|---|---|---|---|---|
| LLaVA-1.5 | 7B | 32.22 | 28.00 | 34.78 | 40.00 |
| Qwen-VL-Max | api | 66.00 | 56.00 | 68.17 | 69.83 |
| GPT-4o | api | 75.56 | 80.00 | 73.76 | 75.69 |
| **MiniDrive_224** | 83M | **86.67** | 36.00 | 62.15 | 62.93 |

**Barrier 识别 86.67 分，超过 GPT-4o 的 75.56**。这是 1000 倍参数差异下的超越，挺震撼的。原因我猜是：驾驶场景的 barrier 样式相对固定（锥桶、水马、护栏），CNN 的大卷积核特别适合抓这种纹理特征，而 GPT-4o 的通用 ViT 反而容易被其他语义干扰。

但 Cone 只有 36 分，paper 自己承认是训练数据分布问题。

### Ablation (Table 3)

| FE-MoE | DI-Adapter | BLEU-4 | CIDEr |
|---|---|---|---|
| ✗ | ✗ | 45.70 | 3.07 |
| ✓ | ✗ | 48.30 | 3.23 |
| ✗ | ✓ | 48.00 | 3.16 |
| ✓ | ✓ | **49.70** | **3.28** |

两个 module 各自贡献差不多，FE-MoE 对 CIDEr 贡献略大（+0.16 vs +0.09），DI-Adapter 对 BLEU-4 贡献略大（+2.3 vs +2.6）。合起来有叠加效果，没有互相抵消。

---

## 这 paper 的真正 contribution

**不是 SOTA score，是证明了一个 counter-intuitive 的事实**：

VLM 不一定要 ViT + 大 LLM 才能做 multimodal reasoning。只要 tokenization 设计得当，80M 的 tiny model 也能在专门领域 VQA 上打过 7B 模型。

三个关键 design choice 缺一不可：
1. **CNN encoder**：提供 (C, H, W) 形状的 feature map，方便后面操作
2. **Reverse flatten**：把 channel 当 token，让每个 token 是"语义通道"而不是"像素块"
3. **Instruction-conditioned visual token**：visual token 进 LLM 前先根据 question 调整

这套组合拳打下来，83M 参数就能在 CODA-LM 的 Barrier 任务上超过 GPT-4o。

---

## 局限性，paper 没细说但很重要的

1. **T5-Small 是 encoder-decoder**，生成速度比 decoder-only LLaMA 慢。换 Phi-3-mini (3.8B) 或 Qwen-0.5B 这种小 decoder-only 可能更好。
2. **DI-Adapter 的 cross-attn 是 O(l1 × l2 × d)**，text 长 l2 大时计算量会涨。paper 没测长 instruction 情况。
3. **没测端到端 planning**：DriveLM 是 VQA benchmark，真正自动驾驶要输出 trajectory / control signal，paper 没碰。
4. **Gate network 没可视化分析**：4 个 expert 各自学了什么？是否有 specialization？paper 没给 t-SNE 或激活图分析。这是 MoE 工作常见的 omission。
5. **Hallucination 严重**：tiny LLM 容量小，容易编内容。paper 自己承认这点。

---

## 对你 (Karpathy) 可能特别相关的点

你 nanoGPT / llm.c / micrograd 系列一直强调"用最少代码实现核心机制"。MiniDrive 的 FE-MoE + DI-Adapter 在结构上非常简单：一个 deconv+conv 的 expert、一个 cross-attn、一个 residual。加起来代码估计 200 行。这是一个绝佳的"教学版 reimplementation"对象——能让学生在单卡 4090 上从零训一个能跑的 VLM，而不是在 Colab 上跑 LLaVA 微调都要等半天。

另一个和 NanoFlow 相关的点：83M 参数 + 5.9B FLOPs，在 A100 上能跑 >200 FPS，4090 上 100+ FPS，车端 Orin 上 30+ FPS 可行。这把 VLM 从"research demo"拉到了"可能部署"的边界。如果你想做一个"end-to-end differentiable self-driving stack with VLM reasoning"，MiniDrive 这个量级正好能塞进 realtime loop 里。

参考链接：
- NanoGPT: https://github.com/karpathy/nanoGPT  
- llm.c: https://github.com/karpathy/llm.c  
- Phi-3-mini: https://arxiv.org/abs/2404.14219  
- T5: https://arxiv.org/abs/1910.10683  
- MoE (Shazeer): https://arxiv.org/abs/1701.06538  
- Soft MoE: https://arxiv.org/abs/2403.04634  
- Flamingo Perceiver Resampler: https://arxiv.org/abs/2204.14198  
- LLaVA: https://arxiv.org/abs/2304.08485  
- BLIP-2 Q-Former: https://arxiv.org/abs/2301.12597  
- DriveLM benchmark: https://github.com/YuanJianhao50/drivelm  
- CODA-LM: https://arxiv.org/abs/2404.10595

---

# MiniDrive: 通用化自动驾驶 VLM 的轻量化探索

## 1. 这篇 paper 要解决的核心问题

自动驾驶的 VLM 目前普遍存在三个痛点，这篇 paper 用一种"反主流"的方式逐一回应：

- **参数过大无法实时部署**：DriveGPT4、DriveMLM、LLM-Driver 这类方法都依赖 LLaMA-7B 量级的 backbone，FLOPs 在 268B–535B 之间，在车端 SoC 上根本跑不动。
- **多相机输入处理能力差**：大多通用 VLM (LLaVA, Qwen-VL, InstructBLIP) 都是单图训练的，而自动驾驶天然需要 front / front-left / front-right / rear / rear-left / rear-right 多视角。
- **视觉 token 静态固定**：传统 VLM 把 image 编码成固定的 visual token，无论用户问"前方有什么车"还是"左侧行人会怎么动"，视觉 representation 完全一样，cross-modal 对齐的负担全压在 LLM 内部 cross-attention 上。

MiniDrive 的核心 insight 是：**视觉 encoder 用大卷积核 CNN（UniRepLKNet）而非 ViT，并显式地让 visual token 在进入 LLM 之前就根据 text instruction 做一次动态调整**。

参考链接：
- arXiv: https://arxiv.org/abs/2406.06722
- UniRepLKNet: https://arxiv.org/abs/2312.17171
- DriveLM: https://arxiv.org/abs/2312.14150
- CODA-LM: https://arxiv.org/abs/2404.10595

---

## 2. 整体架构 (Figure 3 详解)

```
[Multi-view Images (n张)]
        │
        ▼
[UniRepLKNet (frozen)] ───► V_2D ∈ R^{c×h×w}   (每张图一组 2D feature map)
        │
        ▼
[FE-MoE]  ──► V_moe ∈ R^{c'×h'×w'}  (通道↓，空间↑)
        │
        ▼
[Flatten + Projection]  ──► V ∈ R^{l1×dim}
        │
        ▼
[DI-Adapter (Cross-Attn, residual)]  ──► V_input = V + V'
        │
        ▼
[Concat with Text Embedding T_input]  ──► [V_input, T_input]
        │
        ▼
[T5-Small]  ──► Text Response
```

关键设计哲学：**整个 visual pipeline 都在把"图像"压成"几个有语义的 token"**，而不是像 LLaVA 那样把 ViT 的 256 或 576 个 patch token 直接喂进 LLM。MiniDrive 每张图最终只用 **16 个 token**，6 张多视角图也就是 96 个 token，几乎和一句话长度相当。

---

## 3. Vision Encoder: UniRepLKNet

### 3.1 为什么选大卷积核 CNN 而不是 ViT

ViT 的核心机制是把图像切成 patch，通过 self-attention 建模全局关系。但自动驾驶场景的图像有大量"局部强相关、全局弱相关"的内容（车道线、车辆边缘、锥桶等），用 ViT 的全局 attention 反而是浪费。

UniRepLKNet 用的是 **large kernel convolution (例如 31×31)**，核心论点（来自 Ding et al. 2022, 2024）：
- 大卷积核一次就能覆盖很大的 receptive field，不需要堆很多层
- 对 shape / edge / texture 这种"translation-invariant"的特征更友好
- 计算量比 self-attention 低一个量级
- 已在 image/audio/video/point cloud/time-series 多个模态上验证

UniRepLKNet 由若干 Stage 串联组成，每个 Stage 内部有 **Lark Block**（large kernel）和 **Smak Block**（small kernel）交替。MiniDrive 取最后一个 Stage 的输出 feature map $F_1 \in \mathbb{R}^{c \times h \times w}$。

直觉：相当于用 CNN 把图像压缩成一张"高维语义地图"，而不是 patch sequence。后面再通过 FE-MoE 进一步压缩到 token 级。

---

## 4. FE-MoE (Feature Engineering Mixture of Experts)

这是本文最关键的模块之一，目的是**把 2D feature map 转成 text token embedding**，并且对每个图像动态选择处理路径。

### 4.1 Gate Network: 选 expert 的权重

$$ \text{Weights} = \text{Softmax}(\text{Gate}(F_1)) \tag{1}$$

- $F_1 \in \mathbb{R}^{c \times h \times w}$：单张图的 feature map
- $\text{Gate}(\cdot)$：由 Conv + MaxPool + Linear 组成，输出一个长度为 $N$（expert 数）的 logit 向量
- $\text{Softmax}$：归一化成概率权重 $W \in \mathbb{R}^{N}$

直觉：不同视角、不同场景的图像适合用不同的"特征加工方式"，Gate 网络让模型自己学怎么路由。比如前向摄像头和后向摄像头的内容分布差异大，Gate 学到给它们不同的 expert 组合。

### 4.2 Expert Network: 单个 expert 的处理

$$ F_2 = \text{Conv}(\text{ReLU}(\text{Deconv}(F_1))) \tag{2}$$

- $\text{Deconv}(\cdot)$：deconvolution (transposed conv)，作用是**上采样 spatial 维度 + 下采样 channel 维度**
  - 输入 $F_1 \in \mathbb{R}^{c \times h \times w}$
  - 输出 $\in \mathbb{R}^{c' \times h' \times w'}$，其中 $c' < c$，$h' > h$，$w' > w$
- $\text{ReLU}$：非线性激活
- $\text{Conv}$：再做一次卷积精炼特征

**这是反直觉的设计**——一般 VLM 用 Q-Former 是把 spatial 维度压扁，channel 维度对应到 token。这里 FE-MoE 反过来：**先增大 spatial 再降 channel，最后 flatten 时把 channel 当作 token 序列长度，spatial 当作 token 的 embedding 维度**。

### 4.3 形状变换的细节 (Eq 3)

$$ F_1 \in \mathbb{R}^{c \times h \times w} \rightarrow F_2 \in \mathbb{R}^{c\downarrow \times h\uparrow \times w\uparrow} = F_2 \in \mathbb{R}^{c' \times h' \times w'} \tag{3}$$

符号解读：
- $c\downarrow$：通道数减少，例如从 1280 降到 16
- $h\uparrow, w\uparrow$：空间维度增大，例如从 7×7 升到 14×14

**关键 insight**：把 $c' = 16$ 当作 token 数，把 $h' \times w' = 196$ 当作每个 token 的 dim。这正好对应 paper 4.1 节说的"每张图 16 个 token"。

### 4.4 多 expert 加权融合

$$ F_i = \text{Expert}_i(\text{VisionEncoder}(\text{Image})) \tag{4}$$

$$ V_{\text{moe}} = \sum_{i=1}^{N} W_i \cdot F_i \tag{5}$$

- $N$：expert 总数（默认 4）
- $W_i$：Gate 给第 $i$ 个 expert 的权重
- $F_i$：第 $i$ 个 expert 的输出
- $V_{\text{moe}} \in \mathbb{R}^{c' \times h' \times w'}$：加权融合后的 feature

**soft MoE 而非 hard routing**：所有 expert 都跑，再用权重加权，避免 load balancing 问题，也避免训练不稳。

### 4.5 Flatten + Projection

$V_{\text{moe}} \in \mathbb{R}^{c' \times h' \times w'}$ 沿 channel 维 flatten 成 $V \in \mathbb{R}^{l_1 \times \text{dim}_1}$，其中：
- $l_1 = c'$（token 数）
- $\text{dim}_1 = h' \times w'$（token 维度）

再过一层 linear projection 把 dim 映射到 LLM 的 hidden size $d$（T5-Small 的 $d = 512$）：

$$ V \in \mathbb{R}^{l_1 \times d} $$

---

## 5. DI-Adapter (Dynamic Instruction Adapter)

### 5.1 动机

传统 VLM 的 visual token 是"image-only"的：相同图像无论问什么问题，visual token 都一样。例如同一张图，问"前方有什么车"和问"红绿灯什么颜色"，模型用的是同一份 visual embedding，然后让 LLM 内部 cross-attention 去挑相关信息。这其实是浪费了一次"early fusion"的机会。

DI-Adapter 的想法：**让 visual token 在进入 LLM 之前，就先根据 text instruction 做一次条件调整**。

### 5.2 Cross-Attention 机制

$$ V' = \text{CrossAttn}(q = V, \, k = T, \, v = T) \tag{6}$$

- $V \in \mathbb{R}^{l_1 \times d}$：visual token 序列，做 **query**
- $T \in \mathbb{R}^{l_2 \times d}$：text instruction 的 token embedding，做 **key** 和 **value**
- $V' \in \mathbb{R}^{l_1 \times d}$：融合了 text 上下文的 visual token

**这个方向很重要**：用 $V$ 做 query，意思是"每个 visual token 去问 text token：你关心我哪部分？"。这样 visual token 被重新加权，text 提到的概念相关的 visual region 会被强化，不相关的会被弱化。

如果反过来用 text 做 query、visual 做 k/v，那就变成"把 visual 信息读到 text 里"，效果完全不同——那是 standard Q-Former 的做法，会让 visual token 仍保持 image-only 性质。

### 5.3 Residual Connection

$$ V_{\text{input}} = V + V' $$

residual 保留了原始 visual 信息，避免 cross-attn 把 visual 信息过度"扭曲"成 text 想要的样子（防止 hallucination）。

### 5.4 最终送入 LLM 的输入

$$ [V_{\text{input}}, \, T_{\text{input}}] $$

visual token 在前，text token 在后，concat 后送进 T5-Small 做自回归生成。

---

## 6. 训练 Loss

$$ \text{Loss} = -\sum_{i=1}^{n} y_i \log(p_i) \tag{3}$$

- $n$：token 序列长度
- $y_i$：第 $i$ 个 position 的 ground-truth token id（one-hot）
- $p_i$：模型预测的第 $i$ 个 position 上 vocab 分布

标准 cross-entropy，没有特殊 trick。Vision encoder frozen，其他全参数训练，6 epoch，lr=1e-4，weight decay=0.05，单卡 RTX 4090 24GB 就能跑（这是它最大的卖点之一）。

---

## 7. 实验结果分析

### 7.1 DriveLM 上的表现 (Table 1)

| Method | Params | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|
| EM-VLM4AD_Base | 345M | 45.36 | 34.49 | 71.98 | 3.20 |
| EM-VLM4AD_QLarge | 345M | 40.11 | 34.34 | 70.72 | 3.10 |
| DriveLM-Agent | 3.96B | 53.09 | 36.19 | 66.79 | 2.79 |
| **MiniDrive_224** | **83M** | 49.70 | 36.30 | 73.30 | 3.28 |
| **MiniDrive_384** | **137M** | 50.20 | 37.40 | 73.50 | 3.32 |

观察：
- BLEU-4 上 DriveLM-Agent 略胜（53.09 vs 50.20），但参数是 47 倍
- METEOR / ROUGE-L / CIDEr 三项 MiniDrive 全面领先
- CIDEr 是最能反映 caption 质量的指标，MiniDrive 最高 3.32
- 从 224 到 384 分辨率提升，4 个 metric 都小幅上涨

### 7.2 计算开销对比 (Table 2)

| Model | Params | FLOPs | Memory (GB) |
|---|---|---|---|
| DriveMLM | 8.37B | 535B | 36 |
| Drive-GPT4 | 7.3B | 329B | 29.2 |
| LLM-Driver | 7B | 268B | 28 |
| DriveLM-Agent | 3.96B | 439B | 14.43 |
| EM-VLM4AD_Base | 345M | 9.9B | 1.97 |
| **MiniDrive_224** | **83M** | **5.9B** | **1.03** |

**MiniDrive 比 EM-VLM4AD 还小 4 倍，FLOPs 小 40%，memory 小一半**。这是真正的"tiny VLM"。

### 7.3 CODA-LM 上的表现 (Table 4)

| Method | Params | General | Vehicle | VRU | Cone | Barrier | Other | Suggestion |
|---|---|---|---|---|---|---|---|---|
| LLaVA1.5 | 7B | 22.60 | 34.78 | 40.00 | 28.00 | 32.22 | 24.00 | 14.20 |
| Qwen-VL-Chat | 7B | 26.00 | 53.33 | 57.76 | 60.00 | 48.89 | 44.29 | 35.40 |
| Qwen-VL-Max | api | 34.60 | 68.17 | 69.83 | 56.00 | 66.00 | 59.29 | 47.40 |
| GPT-4o | api | 45.00 | 73.76 | 75.69 | 80.00 | 75.56 | 69.29 | 55.50 |
| **MiniDrive_224** | 83M | 21.60 | 62.15 | 62.93 | 36.00 | 86.67 | 59.29 | 45.40 |
| **MiniDrive_384** | 137M | 24.60 | 66.34 | 67.41 | 36.00 | 84.44 | 62.86 | 45.44 |

非常 interesting 的发现：
- **Barrier (锥桶/路障) 识别 86.67 分**，超过 GPT-4o 的 75.56！这是 1000 倍参数差异下的超越
- **Vehicle / VRU / Suggestion** 接近 Qwen-VL-Max（百亿参数级）
- **Cone** 偏低（36.00），paper 解释是训练集分布问题
- General 偏低（21.60），因为 T5-Small 的通用语言能力弱，回答 general question 时不如大 LLM

直觉解释：**专门任务上 tiny model 可以超过 giant model，但通用能力上 tiny model 有 ceiling**。这是 7B LLaMA 也跑不过 GPT-4o 的原因，MiniDrive 也不例外。

### 7.4 Ablation Study (Table 3)

| FE-MoE | DI-Adapter | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|
| ✗ | ✗ | 45.70 | 34.09 | 69.74 | 3.07 |
| ✓ | ✗ | 48.30 | 35.40 | 72.10 | 3.23 |
| ✗ | ✓ | 48.00 | 35.70 | 72.00 | 3.16 |
| ✓ | ✓ | **49.70** | **36.30** | **73.30** | **3.28** |

观察：
- 两个模块单独加都有提升，加起来效果最好
- FE-MoE 提升 BLEU-4 2.6 分，DI-Adapter 提升 2.3 分，差不多
- CIDEr 上 FE-MoE 贡献更大（+0.16 vs +0.09），说明 visual token 质量对 caption 相关性影响更大
- DI-Adapter 在 METEOR 上贡献更明显（+1.61 vs +1.31），说明它对"语义对齐"帮助更大

### 7.5 Token 数 & Expert 数 ablation (Figure 6)

paper 测了 tokens per image ∈ {8, 16, 32}，experts ∈ {2, 4, 6}：
- **16 token + 4 expert** 是 sweet spot
- token 数太大（32）：LLM 学长序列能力下降
- expert 数太大（6）：FE-MoE 训练难度上升，性能下降

直觉：tiny model 容量有限，过度参数化反而学不好。

---

## 8. 这篇 paper 真正的 insight

### 8.1 "Visual token 应该是 instruction-conditioned"

传统 VLM 的 image encoder 是"问题无关"的，visual token 永远一样。DI-Adapter 把这层窗户纸捅破了：**在送进 LLM 之前先做一次 text-guided visual recoding**。

这个思路其实和 Flamingo 的 Perceiver Resampler、BLIP-2 的 Q-Former 有本质区别：
- Q-Former：用 learnable query 把 visual 信息"抽"出来，仍然是 image-only
- Flamingo：在 LLM 层内做 cross-attention，visual token 仍然 image-only
- **DI-Adapter**：visual token 本身被 text 改写了

代价是 visual token 无法 cache（不同 instruction 要重新算），但好处是 cross-modal 对齐质量提升。

### 8.2 "大卷积核 CNN 可能比 ViT 更适合驾驶场景"

ViT 的 self-attention 对"密集局部 pattern"（车道线、车辆边缘）不一定最优，而且 ViT 的 patch token 数量（256+）对 tiny LLM 来说太长。大卷积核 CNN 一次 receptive field 就很大，特征更"压缩"，正好契合 FE-MoE 后面要 flatten 成 16 token 的需求。

### 8.3 "MoE 不只是 LLM 的事"

传统 MoE 用在 FFN 层做 conditional computation，FE-MoE 用在 visual feature 加工层，相当于"用不同 expert 学习不同的视觉语义加工方式"。这给 tiny VLM 提供了一个轻量级" specialization"机制。

### 8.4 "Image-to-text token mapping 的方向可以反过来"

LLaVA / MiniGPT-4 把 visual token 当成"foreign language token"塞进 LLM 的 input embedding 空间。MiniDrive 进一步把 visual feature map 的 channel 当 token、spatial 当 dim，这种"reverse flatten"配合 deconv 上采样，让一个 token 对应一个"语义区域"而不是一个"图像 patch"。这更像 NLP 中 token = 概念而不是 token = pixel block。

---

## 9. 局限性和我的思考

paper 自己承认的局限：
1. **泛化性不足**：训练数据局限于驾驶场景，CODA-LM 上 general score 只有 21.60
2. **Hallucination**：tiny LLM (T5-Small 60M) 容量小，容易编内容
3. **没有 video**：自动驾驶其实是时序问题，单帧 VLM 难以做轨迹预测

paper 没强调但我觉得重要的：
1. **DI-Adapter 的 cross-attention 是 O(l1 × l2 × d)** 的，text 长 l2 大时计算量会涨，paper 没测长 instruction 情况
2. **FE-MoE 用 soft routing**，4 个 expert 全跑，并不省 FLOPs，只是给模型更多 capacity。如果改 hard routing + top-1，理论上还能再快
3. **T5-Small 是 encoder-decoder**，比 decoder-only LLaMA 在生成上慢，未来换 Phi-3-mini 或 Qwen-0.5B 这类小 decoder-only 可能更好
4. **没测端到端 planning**：DriveLM 是 VQA benchmark，但真正 autonomous driving 需要输出 trajectory / control signal，paper 没碰这块
5. **Gate network 没分析**：4 个 expert 各自学了什么？是否有 specialization？paper 没可视化分析，这是 MoE 工作常见的 omission

---

## 10. 对你 (Karpathy) 的 intuition 可能相关的几个点

你可能熟悉的几个相关工作放一起对比：

| 维度 | LLaVA | Q-Former (BLIP-2) | Perceiver Resampler (Flamingo) | FE-MoE + DI-Adapter (MiniDrive) |
|---|---|---|---|---|
| Visual encoder | CLIP ViT-L/14 | ViT-g/14 | ViT | UniRepLKNet (CNN) |
| Visual token 数 | 576 (336px) | 32 | 64 | 16/image |
| Token 是否 text-conditioned | ✗ | ✗ | ✗ | ✓ (DI-Adapter) |
| 跨模态对齐方式 | MLP projection | learnable query + cross-attn | learnable query + cross-attn | cross-attn + residual |
| 训练 cost | 高 | 中 | 高 | 极低 |

你的 micrograd / nanoGPT / llm.c 系列一直强调"用最少的代码实现核心机制"，MiniDrive 的 FE-MoE + DI-Adapter 在结构上其实非常简单：一个 deconv + conv 的 expert、一个 cross-attn、一个 residual，加起来代码可能 200 行。这是一个值得做"教学版 reimplementation"的对象。

另一个对你 NanoFlow 课题相关的点：**83M 参数 + 5.9B FLOPs** 意味着这个模型在单张 A100 / 4090 上能跑 >100 FPS，已经满足车端实时性。如果用 quant + TensorRT 进一步优化，到 200+ FPS 可行。这把 VLM 从"research demo"拉到了"可能部署"的边界。

参考链接：
- LLaVA: https://arxiv.org/abs/2304.08485
- BLIP-2: https://arxiv.org/abs/2301.12597
- Flamingo: https://arxiv.org/abs/2204.14198
- EM-VLM4AD: https://arxiv.org/abs/2403.19838
- UniRepLKNet: https://arxiv.org/abs/2312.17171
- RepLKNet (前作): https://arxiv.org/abs/2203.06717
- DriveLM benchmark: https://github.com/YuanJianhao50/drivelm
- CODA-LM: https://arxiv.org/abs/2404.10595
- T5 (Raffel et al.): https://arxiv.org/abs/1910.10683
- MoE 原始 paper (Shazeer et al.): https://arxiv.org/abs/1701.06538
- Soft MoE: https://arxiv.org/abs/2403.04634

---

## 11. 一句话总结

MiniDrive 把 VLM 在自动驾驶的应用从 "7B+ ViT+LLaMA" 拉到 "80M CNN+T5"，靠的是三个反主流选择：**CNN encoder、reverse-flatten tokenization、instruction-conditioned visual token**。它的真正贡献不是 SOTA score，而是证明了 **VLM 不一定要 ViT + 大 LLM 才能做 multimodal reasoning**——只要 tokenization 设计得当，tiny model 也能跑得动专门领域 VQA。
