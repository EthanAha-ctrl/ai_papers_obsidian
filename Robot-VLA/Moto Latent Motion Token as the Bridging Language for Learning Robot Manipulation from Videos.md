---
source_pdf: Moto Latent Motion Token as the Bridging Language for Learning Robot Manipulation
  from Videos.pdf
paper_sha256: 836e484a30b2ce0b409807c3e4a764313a93e8f5773d6e83148cfe06a157a751
processed_at: '2026-08-05T20:52:17-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Hey Andrej，很高兴跟你接着聊。我们把这篇 paper 彻底用大白话拆解一下，顺带把里面的技术细节、公式、数据表和架构图都扒开看清楚。我会尽量多展开一些联想，希望能帮你 build 起对 robot learning 结合 LLM paradigm 的直觉。

## 核心直觉：Robot 学习的 "GPT 时刻"

Robot learning 现在最大的瓶颈是 data。Action-labeled data 太贵了，你要租机械臂、雇工程师遥操作，成本极高。然而，YouTube 之类的地方有海量视频，里面充满了物理规律和交互知识。因为 video data 极其丰富且廉价，所以研究人员一直想把 LLM 那套 "pretrain on internet text, finetune on specific task" 的成功经验搬到 robot 领域。

之前很多尝试（比如 GR-1）直接让模型预测未来的 raw pixels。但是预测 raw pixels 有个致命缺点，画面里大部分像素是静止的背景，模型浪费了大量的 capacity 去预测那些不变的墙、桌子，真正有价值的 motion 信息被淹没了。人学习新技能时，盯着的是物体怎么动、手怎么抓，也就是 frame 之间的 "delta"。Moto 的核心直觉正在于此：**我们应当把视频里的 motion 提取出来，变成一种 discrete 的 "language"，然后用 GPT 去 predict 下一个 motion word**。

## 架构图与三阶段深度解析

Moto 整体分为三个 stage，对应 Figure 2。

### Stage 1: Latent Motion Tokenizer (提取 "Motion 语言")

这个阶段目的是无监督地从视频里学一个 dictionary，把视觉变化压缩成 tokens。

**架构解析：**
这是一个 VQ-VAE 变体。输入是连续两帧 $o_{t-1}$ 和 $o_t$。Encoder 叫 M-Former，它先用一个 frozen 的 ViT 提取两帧的 patch features，然后拼接上 8 个 learnable query embeddings。M-Former 通过 self-attention 让这 8 个 query 主动去 patch features 里“抓取”运动信息。输出后经过一个 VQ Codebook 量化，得到 8 个 discrete tokens。
Decoder 接收 $o_{t-1}$ 的 patch features 和这 8 个 tokens，尝试重建 $o_t$。

**精妙的 Information Bottleneck：**
Decoder 并非把 8 个 tokens 分别加到不同 patch 上。它把这 8 个 quantized embeddings concatenate 后用一个 MLP 压成 1 个 vector，然后把这个 vector 加到 $o_{t-1}$ 的每一个 patch 上。因为 8 个 token 的信息被极度压缩成 1 个全局 vector 广播出去，模型被迫只能编码全局的、语义级别的 motion（比如“向右推”、“旋转”），而无法编码 per-pixel 的细枝末节。

**公式与变量：**
Tokenizer 的训练使用标准 VQ-VAE loss:
$$ \mathcal{L}_{tokenizer} = \mathcal{L}_{recon} + \mathcal{L}_{VQ} + \mathcal{L}_{commit} $$
其中 $\mathcal{L}_{recon}$ 是重建 $o_t$ 的 MSE loss。$\mathcal{L}_{VQ}$ 是把 encoder output 拉向 codebook entry 的 loss (stop-gradient on codebook side)。$\mathcal{L}_{commit}$ 是让 encoder output 不要偏离 codebook 太远的 commitment loss。

**实验数据表 (Table 4 & 5 细节)：**
Codebook size 只有 128，latent dimension 是 32。M-Former 只有 4 层。训练 batch size 是 256，用 AdamW 优化器，cosine decay schedule。训练时对视频做了 downsample，OXE 数据 $\Delta t = 3$，人手视频 SSV2 $\Delta t = 6$。因为如果相邻帧太近，motion 看不见，模型学不到东西。

### Stage 2: Moto-GPT Pre-training (Next Motion Token Prediction)

有了 Tokenizer，整段视频就被翻译成了一串 motion token 序列。接下来就是熟悉的 LLM 套路。

**架构解析：**
Moto-GPT 是个标准的 GPT-style transformer，12层，hidden size 768，12个 attention heads。输入序列的构造是：[Language features $l$] + [Initial frame visual features $v$] + [Motion tokens sequence $m_{1...M}$]。Language features 来自 frozen T5，Visual features 来自 frozen ViT。

**公式解析 (Equation 1)：**
Pre-training 目标是最大化 ground-truth motion token 序列的似然：
$$ \mathcal{L}_{motion} = - \sum_{i=1}^{M} \log P(m_i | l, v, m_{<i}; \Theta) $$
*   $i$: 当前预测的 token 索引。
*   $M$: 序列总长度，等于 $K \times T$。$K=8$ 是每两帧之间的 token 数量，$T$ 是视频长度。
*   $m_i$: 当前要预测的第 $i$ 个 motion token。
*   $l$: frozen T5 提取的 language instruction features。
*   $v$: frozen ViT 提取的 initial frame visual features。
*   $m_{<i}$: 第 $i$ 个 token 之前的所有 motion tokens (causal mask 保证只能看到过去)。
*   $\Theta$: Moto-GPT 中可训练的参数。

**实验数据表 (Table 8 解析)：**
衡量预训练效果用了 Top-K accuracy。Codebook size 是 128，random guess 的 Top-5 大约 3.9%。在 OXE 数据集上，Moto-GPT 的 Top-5 达到了 52.1%，Top-20 达到 85.3%。因为模型确实学到了物理和动作的 conditional distribution，所以这证明了 next-token prediction 在 motion language 上同样有效。更迷人的是，因为模型学到了 motion 的 distribution，所以它输出的 log-likelihood 可以直接当作评估 trajectory 合理性的 reward model (Figure 7)。

### Stage 3: Co-fine-tuning (桥接 Abstract Motion 与 Concrete Actions)

预训练后的 Moto-GPT 懂了“怎么动”，但它只会输出 abstract tokens，不会输出真实的电机指令。我们要把这个 motion prior 转移到 real robot 上。

**架构解析：**
在每个 time step 的 motion token chunk 后面，插入 $N$ 个 learnable **action query tokens** (比如 SIMPLER 任务 $N=3$，CALVIN 任务 $N=5$)。这些 query tokens 末端接一个 MLP action head，输出真实的 robot action：位置变化 $\Delta x$、旋转变化 $\Delta\theta$、夹爪开关 $\Delta grip$。

**Attention Mask 的精妙设计：**
这里的设计极其重要。
1. Latent motion tokens **不** attend to action query tokens。因为预训练时没见过 query tokens，这样保持了一致性，避免 distribution shift。
2. Action query tokens 有 50% 的概率 attend to motion tokens，50% 不 attend。因为我们需要 action tokens 从 motion tokens 里借取 prior knowledge，同时又要避免它过度依赖 ground-truth motion tokens。在真实推理时，我们其实可以用 padding tokens 替换 motion tokens，让 action tokens 直接输出 action，这样推理效率极高，相当于把 Moto-GPT 变成了纯 VLA model，但 weights 里保留了 motion priors。

**公式解析 (Equation 2 & 3)：**
Action loss:
$$ \mathscr{L}_{action} = \mathscr{L}(\Delta x) + \mathscr{L}(\Delta\theta) + \mathscr{L}(\Delta grip) $$
对于连续的 $\Delta x$ 和 $\Delta\theta$ 使用 Smooth-L1 loss，对于二元的 $\Delta grip$ 使用 BCE loss。

总 fine-tuning loss:
$$ \mathcal{L}_{ft} = \mathcal{L}_{motion} + \mathcal{L}_{action} $$
因为必须保留 $\mathcal{L}_{motion}$，所以模型在微调时仍在做 next-motion-token prediction。Figure 12 的 ablation 证明，如果去掉这个 loss (Moto-IML)，模型会把预训练的 motion prior 忘掉，性能下降。

## 实验数据表深度分析

我们来看看这套设计打出的战绩。

### Table 2: SIMPLER Benchmark
| Method | Overall | Pick Coke (Avg) | Move Near | Drawer (Avg) |
| :--- | :--- | :--- | :--- | :--- |
| RT-2-X (55B) | 0.607 | 0.787 | 0.779 | 0.250 |
| OpenVLA (fine-tuned) | 0.349 | 0.363 | 0.542 | 0.231 |
| **Moto (98M)** | **0.614** | 0.740 | 0.604 | 0.431 |
| Moto w/o Motion Token | 0.480 | 0.503 | 0.554 | 0.398 |

Moto 的 GPT backbone 只有 98M 参数，却打败了 55B 参数的 RT-2-X。因为 motion tokens 极度紧凑，把 model capacity 全用在了刀刃上。相比没有 pretrain 的 baseline (Moto w/o Motion Token)，overall 提升了 13.4%，直接证明了 video pretraining 带来的增益。

### Table 3: CALVIN ABC→D (Long-horizon)
| Model | Avg. Len. | Observation Space |
| :--- | :--- | :--- |
| GR-1 | 3.06 | Static + Gripper RGB + Proprio |
| **Moto** | **3.10** | **Static RGB only** |
| Moto w/o Motion Token | 2.14 | Static RGB |

CALVIN 要求连续完成 5 个任务。Moto 只用单视角的 Static RGB，连 gripper 视角和机器人本体状态都没用，就超过了用多模态输入的 GR-1。进一步证明了 model motion tokens 比 model raw pixels 有效得多。

### Figure 11: Data Efficiency
这是最 actionable 的点。用 1% 的 labeled data 微调时，Moto 成功率 52.5%，从零训练的 baseline 是 0%。用 10% data 时，Moto ~70%，baseline ~30%。这条曲线和 LLM pretraining + finetuning 的曲线完美重合，证明 motion pretraining 把模型拉到了一个极高的起点。

## 延伸思考与 Intuition Building

结合 Karpathy 你在 nanoGPT 里的思想，Moto 简直就是把 nanoGPT 搬到了物理世界上。VQ Codebook 就是 vocabulary，motion tokens 就是 words，video clips 就是 sentences。

1. **VQ 的选择与 Scaling**：Moto 用的 codebook size 只有 128，非常小。这有点像字符级别的语言模型。如果未来把 codebook 扩展到 8192，会不会像 BPE 一样涌现出更精细的 sub-motion 概念？同时，discrete representation 损失了精度，对于需要毫米级精度的组装任务，这可能是个瓶颈。未来也许需要 hierarchical VQ 或者结合 continuous tokens 的混合架构。
2. **与 LeCun JEPA 的呼应**：Moto 在 latent space 做 prediction，这点和 JEPA 思路一致。因为预测 pixel 太浪费，所以在 latent space 预测。但 Moto 用了 discrete quantization，从而能够直接套用 GPT 的 autoregressive framework 和 causal mask，这在工程上具有巨大优势。
3. **World Model 联想**：Moto 的 Decoder 实际上是一个 action-conditioned world model。输入 $o_{t-1}$ + motion tokens，输出 $o_t$。这意味着我们可以拿 Moto 去做 model-based RL。既然 log-likelihood 可以当 reward，那我们甚至可以幻想用 Moto-GPT 做 MPC (Model Predictive Control)，在 latent space 里搜索一条 log-likelihood 最高的 trajectory，然后再 decode 成 actions。
4. **Cross-embodiment 的终极奥义**：Figure 10 展示了把人手视频的 motion tokens 挪到 robot 初始帧上，能生成合理的 robot 动作。因为 motion tokens 是 hardware-agnostic 的，它只描述了“这个物体应该被这样移动”。如果这一套 scale up，我们就能用 YouTube 上所有的人类做饭、修理视频来预训练 robot，这才是通向 AGI 级别 robot 的正道。

## Web References

*   Paper Homepage: https://chenyi99.github.io/moto/
*   Open-X-Embodiment Dataset: https://robotics-transformer-x.github.io/
*   CALVIN Benchmark: https://calvinrobot.github.io/
*   SIMPLER Benchmark: https://simpler-env.github.io/
*   VQ-VAE Paper: https://arxiv.org/abs/1711.00937
*   GR-1 Paper: https://arxiv.org/abs/2312.06817
*   nanoGPT Repo: https://github.com/karpathy/nanoGPT

整体来看，Moto 这篇 paper 的直觉非常清晰，工程实现也很优雅，把 LLM 范式在 robotics 上的应用推进了一大步。Andrej，你觉得这种 discrete action token 的路线，相比 Pi 那种 continuous flow matching 的路线，哪一种更有希望 scale 到整个互联网级别的视频数据上？

---

# Moto: Latent Motion Token 深度解析

## 一、核心 Intuition 与 Motivation

Karpathy 你应该对 LLM 的 next-token prediction 范式非常熟悉。Moto 这篇 paper 的核心 question 极其优雅: **能否把 LLM 的 autoregressive pretraining 范式迁移到 robot learning？**

关键 insight 在于: robot learning 受限于 action-labeled data 的昂贵成本，而 Internet 上有海量的 video data 蕴含 interaction knowledge。问题在于 —— video data 没有 action label，怎么 pretrain 一个能 benefit robot manipulation 的 model？

Moto 的答案是: **找到一种 hardware-agnostic 的"中间表示"（bridging language）**，这种表示要 (1) 能从纯 video 中无监督学到，(2) 紧紧 bind 住 motion dynamics，(3) 能通过 fine-tuning transfer 到 real robot actions。

这里有个非常 deep 的 insight 值得 build 你的 intuition: 人学新 skill 是看 dynamic environment 的变化（motion），而不是盯着 static frame。所以 effective pretraining 应该 model **motion 而非 appearance**。这与 GR-1 那种预测 raw pixel values 形成鲜明对比 —— raw pixels 太冗余，大部分像素其实没动，浪费 model capacity。

## 二、Latent Motion Tokenizer —— 架构深度解析

### 2.1 整体设计哲学

Latent Motion Tokenizer 是一个 VQ-VAE-based 的 auto-encoder，input 是连续两帧 $(o_{t-1}, o_t)$，output 是一组 discrete tokens $m \in \mathbb{R}^{K \times D}$，其中 $K=8$ 是 token 数量，$D=32$ 是 latent dimension。

关键设计是 **conditional reconstruction**:
- Encoder 编码 $(o_{t-1}, o_t) \to$ tokens $m$
- Decoder 从 $(o_{t-1}, m) \to \hat{o}_t$ 重建下一帧

这种设计强制 tokens 必须捕获 frame 之间的 **delta**（即 motion），因为如果 tokens 只编码 static 信息，decoder 无法从 $o_{t-1}$ 重建 $o_t$ 的变化部分。

### 2.2 M-Former 细节

M-Former 的 input 是:
- Frozen pre-trained ViT (MAE trained) 提取的 $o_{t-1}$ 和 $o_t$ 的 patch features（last-layer）
- 8 个 learnable query embeddings 拼接进去

通过 self-attention 让 queries 主动从 patch features 中"拉取" motion-relevant 信息。Output 的 8 个 query features 经过 VQ codebook 量化成 discrete tokens。

```
M-Former Config:
  num_queries: 8
  num_layers: 4
  hidden_size: 768
  num_heads: 12
```

**为什么是 8 个 queries？** 这是一个 bottleneck 设计 —— 8 个 tokens 要压缩一整帧的 motion 信息，强制 model 学到 compact representation。Table 1 显示这种压缩 representation 在 video classification 上达到 79.7% accuracy，接近用所有 8 帧 ViT features 的 82.8%，证明 tokens 确实捕获了 semantic motion。

### 2.3 ViT Decoder 的 Information Bottleneck 设计

这里有一个非常巧妙的 engineering trick。Decoder 并非简单把 tokens 加到 patch embeddings 上，而是:

1. 把 8 个 quantized token embeddings concatenate 起来
2. 用 MLP project 成 **1 个** compact embedding
3. **这个 1-token embedding 加到每个 input patch embedding 上**

为什么这样设计？这是一个 extreme information bottleneck —— 8 个 tokens 的信息被压缩到 1 个 vector，然后全局广播到所有 patches。这强制 tokens 编码的是 **全局 motion 语义**（"向右移动"、"旋转 90 度"），而非 per-patch 的细节。这种 design choice 直接对应了 Figure 4 展示的 interpretability —— 同一组 tokens 在不同 initial frames 上产生 consistent motion 语义。

### 2.4 Loss Function

VQ-VAE 标准 objective:
$$\mathcal{L}_{tokenizer} = \mathcal{L}_{recon} + \mathcal{L}_{VQ} + \mathcal{L}_{commit}$$

- $\mathcal{L}_{recon}$: MSE between decoder output $\hat{o}_t$ and ground-truth $o_t$
- $\mathcal{L}_{VQ}$: 把 encoder output 拉向 codebook entry (stop-gradient on codebook side)
- $\mathcal{L}_{commit}$: 让 encoder output 不要偏离 codebook 太远

**Frame sampling rate 的细节**: 对 OXE 数据 $\Delta t = 3$（每 3 帧取 1 帧），对 human video (SSV2) $\Delta t = 6$，对 CALVIN $\Delta t = 5$。这个 downsampling 很关键 —— 让 motion 之间足够 distinct，否则相邻帧几乎一样，tokenizer 学不到有意义的 motion。

## 三、Moto-GPT Pre-training —— Next Motion Token Prediction

### 3.1 公式深度解析

Pre-training objective (Eq. 1):

$$\mathcal{L}_{motion} = -\sum_{i=1}^{M} \log P(m_i | l, v, m_{<i}; \Theta)$$

变量解释:
- $M = K \times T$，其中 $K=8$ 是 per-frame token 数，$T$ 是 video 长度（最多 3 frames in implementation）
- $l$: frozen T5 提取的 language instruction features
- $v$: frozen ViT 提取的 initial frame visual features
- $m_{<i}$: 在当前 token $m_i$ 之前的 all motion tokens（causal mask）
- $\Theta$: trainable parameters of Moto-GPT

这就是标准的 GPT-style causal language modeling，只是 "language" 换成了 motion tokens。

### 3.2 输入序列构造

对于一个 video clip $[o_0, o_1, ..., o_T]$:
1. 用 tokenizer 对每对 $(o_{t-1}, o_t)$ 提取 8 个 motion tokens
2. 按时间顺序 concatenate 成 sequence
3. Prepended 上 language features $l$ 和 initial frame visual features $v$ 作为 context

Moto-GPT backbone:
```
num_layers: 12
hidden_size: 768
num_heads: 12
total params: 98M (GPT backbone only)
```

### 3.3 Top-K Accuracy 验证

Table 8 给了 Top-K motion token prediction accuracy:

| Dataset | Top-5 | Top-10 | Top-20 |
|---------|-------|--------|--------|
| OXE | 0.521 | 0.698 | 0.853 |
| CALVIN ABC→D | 0.298 | 0.518 | 0.768 |

Codebook size 是 128，random baseline 的 Top-5 应该是 ~3.9%，Top-20 是 ~15.6%。OXE 上 Top-5 达到 52.1%，远超 random，说明 model 确实学到了 motion 的 conditional distribution。CALVIN 上低一些，可能因为 CALVIN 任务更精细（34 个 task），motion diversity 更高。

### 3.4 Log-Likelihood 作为 Trajectory Rationality Metric

Figure 7 展示了一个非常 elegant 的 emergent property: 用 Moto-GPT 的 log-likelihood 区分 successful / failed / random trajectories。这暗示 Moto-GPT 学到的 motion distribution 接近一个 **reward model** —— 越符合 natural motion 的 trajectory log-likelihood 越高。

这个 insight 很重要，因为它意味着 pre-trained Moto-GPT 可以作为:
1. Policy (生成 motion tokens → decode 成 actions)
2. Reward model (评估 trajectory rationality)
3. World model simulator (decoder 从 tokens 生成 frames)

这种 multi-functional 性质让我想到 Hafner 的 Dreamer 系列，但 Moto 的优势是 unsupervised 学到的。

## 四、Co-fine-tuning —— 桥接 Latent Motion 与 Real Actions

### 4.1 Action Query Tokens 的设计

这是我觉得全 paper 最 brilliant 的 engineering design。在 fine-tuning 阶段，每个 time step 在 motion token chunk 后面插入 $N$ 个 **action query tokens**:

- SIMPLER: $N=3$ (3 actions between two frames)
- CALVIN: $N=5$ (5 actions between two frames)

这些 action query tokens 通过 MLP action head 预测 real robot actions:
- $\Delta x$: positional displacement (Smooth-L1 loss)
- $\Delta\theta$: rotational displacement (Smooth-L1 loss)
- $\Delta grip$: gripper open/close (BCE loss)

Total action loss (Eq. 2):
$$\mathcal{L}_{action} = \mathcal{L}(\Delta x) + \mathcal{L}(\Delta\theta) + \mathcal{L}(\Delta grip)$$

### 4.2 Attention Mask 的精妙设计

这里有个 subtle 但 critical 的设计:

1. **Latent motion tokens 不 attend to action query tokens**: 保持与 pre-training 一致，避免 distribution shift
2. **Action query tokens 50% 概率 attend to motion tokens, 50% 不 attend**: 这是个 dropout-like 的 trick

为什么 50% mask？两个目的:
- (a) **Knowledge transfer**: action queries 从 motion tokens 借用 learned priors
- (b) **Reduce dependency**: 避免 action queries 过度依赖 ground-truth motion tokens (inference 时可能没有)

最 elegant 的副作用: **Inference 时可以用 padding tokens 代替 motion tokens**，让 action queries 不 attend padding，直接输出 actions。这相当于把 Moto-GPT 变成一个纯 VLA model，但保留了 pre-training 的 motion priors in its weights。

### 4.3 Co-fine-tuning Loss

Eq. 3:
$$\mathcal{L}_{ft} = \mathcal{L}_{motion} + \mathcal{L}_{action}$$

保留 $\mathcal{L}_{motion}$ 是为了 **retain pre-trained motion priors**。Figure 12 的 ablation 直接验证了这个设计:
- Moto-IML (去掉 $\mathcal{L}_{motion}$): 性能下降，因为 motion priors 在 fine-tuning 中被遗忘
- Moto-DM (完全去掉 motion tokens from input): 性能更差，因为 action queries 无法直接 attend 到 motion representations
- Moto (full co-fine-tuning): 最优

这个 ablation 给了一个 clear 的 takeaway: **representation 和 objective 都要保留 pre-training 的结构**，否则 transfer 不成功。

## 五、实验数据深度分析

### 5.1 SIMPLER Benchmark (Table 2)

| Method | Overall | Pick Coke (Avg) | Move Near | Drawer (Avg) |
|--------|---------|-----------------|-----------|--------------|
| RT-1-X | 0.534 | 0.567 | 0.317 | 0.597 |
| RT-2-X (55B) | 0.607 | 0.787 | 0.779 | 0.250 |
| OpenVLA (7B) | 0.248 | 0.163 | 0.462 | 0.356 |
| OpenVLA (fine-tuned) | 0.349 | 0.363 | 0.542 | 0.231 |
| **Moto (98M)** | **0.614** | 0.740 | 0.604 | 0.431 |
| Moto w/o Motion Token | 0.480 | 0.503 | 0.554 | 0.398 |

几个 striking observations:
1. **Moto 用 98M 参数打败了 55B 的 RT-2-X**（0.614 vs 0.607），这个 parameter efficiency 极其 impressive
2. **Motion token 带来 +13.4% 的 overall gain**（0.480 → 0.614），这是 pre-training 的直接贡献
3. OpenVLA 在 SIMPLER 上表现很差，可能因为 distribution shift，但 Moto 通过 video pretraining 展现了更好的 generalization

### 5.2 CALVIN ABC→D (Table 3)

| Model | Avg. Len. | Obs Space |
|-------|-----------|-----------|
| SuSIE | 2.69 | Static RGB |
| RoboFlamingo | 2.47 | Static + Gripper RGB |
| GR-1 | 3.06 | Static + Gripper RGB + Proprio |
| **Moto** | **3.10** | **Static RGB only** |
| Moto w/o Motion Token | 2.14 | Static RGB |

Moto 用 **更少的 input modalities**（只有 static RGB，没有 gripper view 和 proprio）超过了 GR-1。这强烈暗示 motion tokens 比 raw pixel prediction 是更好的 pre-training target。

### 5.3 Data Efficiency (Figure 11)

这是我觉得最 actionable 的结果:
- **1% labeled data**: Moto 52.5% vs baseline 0%
- **10% labeled data**: Moto ~70% vs baseline ~30%
- **100% labeled data**: Moto ~80% vs baseline ~55%

这个 trend line 的 shape 非常符合 LLM pre-training + fine-tuning 的经典曲线 —— pre-training 把 starting point 抬得很高，少量 fine-tuning data 就能 reach good performance。

### 5.4 Human Video Pre-training (Figure 9)

加上 SSV2 human video 后，Moto 在 Move Near 任务上显著提升。这说明 latent motion tokens 确实是 **cross-embodiment 的 bridging language** —— human hand motion 和 robot gripper motion 在 latent space 中可以 share 同一个 codebook。

Figure 10 的 visualization 更 striking: 把 human video 的 motion tokens 提取出来，apply 到 robot initial frame 上，能生成语义合理的 robot motion。这是 cross-embodiment transfer 的直接证据。

### 5.5 Real-world Experiments (Figure 8)

- 平均 success rate: 23.33% → 60% (Moto w/o vs Moto)
- Visual Distractor 场景: +20% 提升
- Novel Object 场景: +30% 提升

Real-world 泛化性比 simulated 还要好，这非常 promising。

## 六、与 Related Work 的深度对比

### 6.1 vs GR-1 (predict raw pixels)
- GR-1: predict next frame pixel values
- Moto: predict motion tokens
- **Key difference**: motion tokens 是 bottleneck representation，把 model capacity 集中在 motion-relevant 信息上，而非 waste 在 static background

### 6.2 vs Genie / LAPA (latent action pretraining)
- Genie/LAPA: predict one-step future latent action
- Moto: predict **trajectory** of latent motion tokens autoregressively
- **Key difference**: Moto 显式 model sequential structure，更接近真实 policy inference

### 6.3 vs IGOR (latent action as goal)
- IGOR: 用 latent actions 作为 intermediate goals
- Moto: 用 latent motion tokens 作为 pre-training target，直接 transfer 到 real actions

### 6.4 vs DynaMo (visual representation learning)
- DynaMo: focus on visual representation
- Moto: focus on policy pretraining

### 6.5 vs π₀ (flow matching VLA)
π₀ 用 flow matching 生成 continuous actions，而 Moto 用 autoregressive discrete tokens。两种 paradigm 各有优劣:
- Autoregressive: leverage LLM 成熟 stack，interpretable，可做 likelihood evaluation
- Flow matching: continuous action space，sample efficiency 高，但 harder to do pre-training on videos

## 七、延伸思考与 Intuition Building

### 7.1 Latent Motion Tokens 作为 Universal Action Language

这个 idea 让我想到 LeCun 的 JEPA (Joint-Embedding Predictive Architecture) —— 都是在 latent space 做 prediction 而非 pixel space。但 Moto 的 twist 是 **discrete quantization** (VQ)，这让 tokens 可以做 autoregressive prediction (LLM-style)，而 JEPA 通常用 continuous embedding。

潜在 implication: 如果把 codebook size 从 128 扩到很大（比如 8192，类似 LLaMA 的 vocabulary），可能捕获更 fine-grained motion semantics。

### 7.2 World Model 视角

Moto 的 tokenizer decoder 实际上是一个 **action-conditioned world model**:
- Input: $o_{t-1}$ + motion tokens $m$
- Output: $\hat{o}_t$

这与 Dreamer 的 world model 类似，但 Moto 是 unsupervised 学到的，且 actions 是 latent。一个有趣的 future direction: 用 Moto-GPT 做 model-based RL，用 log-likelihood 当 intrinsic reward。

### 7.3 Scaling Laws 推测

LLM 的 scaling laws 表明 pre-training data 越多 performance 越好。Moto 的 Figure 9 已经显示加 SSV2 human video 能进一步提升性能。如果 scale 到 Internet-scale video (YouTube, HowTo100M 等)，会有什么样的 emergent ability？

我猜测可能出现:
- Long-horizon planning (从 cooking video 学到 multi-step manipulation)
- Tool use (从 human tool usage video 学到)
- Cross-task generalization

### 7.4 与你的 work 的关联

Karpathy 你在 nanoGPT 和 makemore 中反复强调 autoregressive next-token prediction 的威力。Moto 可以理解为 **把 nanoGPT 范式应用到 robot learning 的 motion tokens 上**。Codebook 是 vocabulary，motion tokens 是 words，video clips 是 sentences。

这种 framing 让 robot learning 直接受益于 LLM 的所有 advances: scaling laws, in-context learning, chain-of-thought reasoning（可能用 motion tokens 做 planning）。

### 7.5 Limitations 和 Open Questions

1. **Codebook size 限制**: 128 太小，可能限制表达力
2. **Bi-frame tokenization**: 只 model pairwise motion，没有 longer temporal context
3. **Single camera**: 只有 static RGB，没有 3D 信息
4. **Pre-training data scale**: 109k OXE videos 太小，远不及 LLM 的 trillion tokens

## 八、Web References

- **Paper page**: https://chenyi99.github.io/moto/
- **Open-X-Embodiment**: https://robotics-transformer-x.github.io/
- **CALVIN benchmark**: https://calvinrobot.github.io/
- **SIMPLER benchmark**: https://simpler-env.github.io/
- **VQ-VAE (Van Den Oord et al.)**: https://arxiv.org/abs/1711.00937
- **MAE (He et al., ViT encoder)**: https://arxiv.org/abs/2111.06377
- **RT-1**: https://arxiv.org/abs/2212.06817
- **OpenVLA**: https://openvla.github.io/
- **GR-1**: https://arxiv.org/abs/2312.06817
- **Genie (DeepMind)**: https://arxiv.org/abs/2402.15391
- **LAPA (concurrent work)**: https://arxiv.org/abs/2410.11758
- **SSV2 dataset**: https://developer.qualcomm.com/software/ai-datasets/something-something
- **π₀ (Physical Intelligence)**: https://arxiv.org/abs/2410.24164
- **T5 (Raffel et al.)**: https://arxiv.org/abs/1910.10683
- **VIP / R3M (related video pretraining)**: https://arxiv.org/abs/2212.05218
- **nanoGPT (Karpathy)**: https://github.com/karpathy/nanoGPT

## 九、总结 Intuition

把 Moto 浓缩成一句话: **Motion tokens 是 video 的"words"，autoregressive prediction 是"grammar"，co-fine-tuning 是从 abstract grammar 到 concrete motor commands 的 translation**。

整个 pipeline 的 elegance 在于:
1. **Tokenizer**: 把 pixel-level dynamics 压缩成 discrete motion language（无监督，从 video 学）
2. **Pre-training**: 在 motion language 上做 next-token prediction（学到 motion priors）
3. **Co-fine-tuning**: 用 action-labeled data 让 model 同时输出 motion tokens 和 real actions（bridge abstract → concrete）

这种 decoupling 让 robot learning 第一次可以 leverage Internet-scale video data，类似 LLM leverage Internet text。如果这个 paradigm scale up，可能真的是 robot learning 的 "GPT moment"。

期待你的 thoughts 和 pushback，特别想听你对 (1) discrete vs continuous action representation，(2) autoregressive vs flow matching for robot policies 的看法 —— 这些都是 Moto 这个 paradigm 的核心 design choices。
