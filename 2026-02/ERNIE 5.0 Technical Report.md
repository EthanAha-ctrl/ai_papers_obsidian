---
source_pdf: ERNIE 5.0 Technical Report.pdf
paper_sha256: a1450a354beb2e94f871c41d230c11c03f319be07dd665f6f6524cc6864abaee
processed_at: '2026-08-04T05:05:07-07:00'
target_folder: 2026-02
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ERNIE 5.0 用人话讲

## 一、这篇 paper 到底在干嘛

一句话概括：**百度搞了一个万亿参数的模型，把文字、图片、视频、音频全塞进一个 autoregressive 框架里，from scratch 训练，同时能理解也能生成。**

这事为什么难？因为之前业界的做法是"拼积木"——先训一个 language model，再外挂一个 image generator、一个 video generator、一个 audio decoder。各模块用不同的训练目标（text 用 next-token，image 用 diffusion，audio 用 codec reconstruction），拼起来之后你会发现：

- 理解和生成互相打架（ability seesaw）
- 跨 modality 知识没法共享
- 部署时要维护一堆不同架构的模型

ERNIE 5.0 的核心赌注是：**把这些 modality 全部统一成 "predict next group of tokens"，一个 backbone、一套训练目标、一个 MoE expert pool，让所有 token 在同一套参数里流动。**

这个赌注的 intuition 很简单——如果你相信 "autoregressive next-token prediction is all you need" 这套 GPT 范式，那就应该把它贯彻到底。image 的 multi-scale token、video 的 frame token、audio 的 codec token，本质上都是序列，都能用 next-group prediction 来建模。难点在于怎么 tokenization、怎么处理空间/时间结构、怎么稳定训练。

---

## 二、架构的核心 intuition

### 2.1 为什么要 modality-agnostic routing

传统多模态 MoE 的做法是给每个 modality 分配专属 expert——text expert、image expert、audio expert。听起来合理，但实际有两个问题：

**第一，当 modality 超过 2 个，人工分配就是噩梦。** 你怎么决定 text 该占多少 expert？image 和 video 要不要共享？audio 和 text 在 phonetic 层面要不要 share？这些 heuristic 在 modality 数量增长时根本调不动。

**第二，modality label 其实是个 proxy。** 一个 token 真正重要的属性是它的 functional role——它是在提取 semantic、还是在编码 perceptual detail、还是在做 temporal reasoning。Text 的 semantic token 和 audio 的 semantic token（RVQ 第一个 code）在 functional role 上是 isomorphic 的，强行把它们分到不同 expert pool 反而阻碍知识共享。

所以 ERNIE 5.0 干脆 **不告诉 router token 来自哪个 modality**，让 router 只看 token representation 自己决定。实验结果（Sec 6.4.1）很 stunning：

- Expert 自然分化出 universal experts（跨所有 modality 频繁激活）和 specialized experts
- Text 和 audio 的 expert overlap 很高（因为两者都有 linguistic semantic）
- Visual understanding 和 visual generation 的 overlap 反而低（因为 functional role 不同）
- 第一层没有出现人们以为的 routing imbalance

这说明 **router 自己学到了比 modality label 更本质的东西**。Modality 信息在 unified representation space 里是 emergent 的，不需要 explicit supervision。

### 2.2 Vision 为什么要 Next-Frame-and-Scale Prediction

这是整篇 paper 最精巧的设计。先看 image generation：

传统 diffusion 的做法是从 noise 一步步 denoise 出整张图。Autoregressive image generation 的早期尝试（如 DALL-E）是把 image tokenize 成 grid，然后按 raster scan 顺序逐 token 预测——问题是 spatial 结构被拍平成 1D，建模效率极差。

VAR（[Visual Autoregressive Modeling](https://arxiv.org/abs/2404.02905)）提出了 next-scale prediction：先预测低分辨率（比如 1×1 的颜色块），再 2×2，再 4×4……每个 scale 内部 bidirectional 可见（parallel 预测），scale 之间 causal。这既保留了 autoregressive 的 sequential nature，又让 spatial 结构自然涌现。

ERNIE 5.0 的 NFSP 把这个 idea 扩展到 video：

```
Image generation:  scale 1 → scale 2 → scale 3 → ... (空间维度从粗到细)
Video generation:  frame 1 (多 scale) → frame 2 (多 scale) → ... (时间维度逐帧)
```

每个 scale 内部 tokens 互相可见（bidirectional attention），previous scales 和 previous frames 以 causal 方式可见。这就 disentangle 了 spatial 和 temporal 建模。

**Uni-RoPE** 是配套的位置编码：

$$\text{Uni-RoPE}_i = (t_i, h_i, w_i)$$

- $t_i$：temporal position（第几帧）
- $h_i$：height position（帧内高度位置）
- $w_i$：width position（帧内宽度位置）

Text 和 audio token 退化成 $t_i = h_i = w_i$（三者相等，就是 sequence index）。Visual token 则 $t$ 用于 frame index，$(h, w)$ 用于帧内 spatial location。Center-aligned 让不同 scale 的 token 在空间上对齐。

**Intuition**：RoPE 本质是给 attention 加 position-dependent rotation。1D RoPE 只有一个 frequency axis，处理不了 2D/3D 结构。Uni-RoPE 把 frequency 分解到三个正交维度，attention 就能感知 "这个 token 在哪一帧的哪个位置"。Center-alignment 保证低分辨率 scale 的中心 token 和高分辨率 scale 的中心区域在 attention 时有正确的相对位置关系。

### 2.3 Audio 的 Depth-wise Prediction

Audio tokenization 用 RVQ（Residual Vector Quantization），产生多个 codebook——第一个 code encode semantic（用 Whisper distillation 对齐），后面的 code encode 越来越细的 acoustic detail（timbre、prosody）。

问题：如果把所有 codebook flatten 成一条序列，长度爆炸（1 秒音频 × 多 codebook × token rate 12.5Hz）。

ERNIE 5.0 的解法是把 RVQ 的不同 level **分布到 transformer 的不同层**：

```
Layer N:     预测 codebook 1 (semantic)
Layer N+1:   把 codebook 1 的 embedding 加回 hidden state，预测 codebook 2
Layer N+2:   把 codebook 2 的 embedding 加回 hidden state，预测 codebook 3
...
```

Understanding 时反过来：所有 level 的 embedding 相加形成 audio token representation，送进 backbone。

**Intuition**：这就像 image 的 multi-scale，只不过 scale 换成了 codec granularity。Coarse-to-fine 的 prediction 范式在 visual 和 audio 上 unified 了。Depth-wise 的好处是序列长度不变（每个 token 还是一个 token），只是 transformer 深度被利用了。

### 2.4 Dual-Path Visual Understanding

生成任务需要 quantized token（离散），但理解任务直接用 quantized token 会丢失 fine-grained 信息。ERNIE 5.0 的解法是 **在 quantization 之前分叉一路出来专门给理解用**：

- CNN path：提 local perceptual detail（边缘、纹理）
- ViT path：提 global semantic

两者通过 **Attention-Based Patch Merger** 融合：

$$\mathbf{F}_{mrg} = \text{concat}(\text{proj}(\mathbf{F}_{cnn}), \mathbf{F}_{vit}) \in \mathbb{R}^{N \times 2K \times D_{vit}}$$

$$\mathbf{Z} = \text{Attn}(\mathbf{F}_{mrg})$$

$$\mathbf{F}_{out} = \text{MeanPool}_{patch}(\mathbf{Z}) \in \mathbb{R}^{N \times D_{vit}}$$

- $N$：visual understanding token 数
- $K$：每个 token 分组的 local patches 数（image K=4，video K=16 跨 4 帧）
- $D_{vit}$：ViT feature 维度

**Intuition**：CNN 和 ViT 看到的东西不一样。Naive MLP fusion 等于强行把它们 align，会引入 representational interference。Attention 让 CNN patch 和 ViT patch 互相 "看"，模型自己决定怎么权衡——在 document understanding 任务上 gain 特别大，因为 OCR 既需要 CNN 的 pixel-level 精度，又需要 ViT 的 layout-level 理解。

---

## 三、Elastic Training：一次训练，多个模型

### 3.1 这是什么黑魔法

传统做法：训一个大模型 → pruning/distillation 出小模型 → 部署。问题是 pruning 要单独 infra，distillation 要 teacher-student 两套，而且每出一个新 size 都要重走一遍流程。

ERNIE 5.0 的 idea：**在 pre-training 时就随机 sample sub-network，让大模型在训练时就"知道"如何在不同 capacity 下 work。**

三个 elastic 维度：

**Elastic Depth（75% full / 25% reduced）**
随机跳过一些 transformer layer。Intuition：让中间层 representation 即使在被 bypass 时仍保持 informative。类似 dropout 但作用在 layer 级别。

**Elastic Width（80% full / 20% reduced）**
随机只用 expert subset。Intuition：让每个 expert 都能独立 work，不依赖其他 expert 的协同。

**Elastic Sparsity（80% default top-k / 20% reduced k）**
随机减少 routing top-k。Intuition：让模型在 fewer activated experts 时仍能 produce 合理 output。

**关键点**：这三个 elastic 是在 **同一次 backprop** 里一起优化的。大部分时候用 full model，偶尔 sample sub-network。Sub-network 和 full model 共享参数，共享 gradient。

### 3.2 实验结果有多 stunning

在 ERNIE 5.0-Exp 上：

| 配置 | Activated Params | Total Params | AVG Score | Decoding Speed |
|------|-----------------|--------------|-----------|----------------|
| Full model | 100% | 100% | 75.55 | 1.0× |
| Elastic Sparsity (top-k 25%) | ~25% routing | 100% | 74.43 | **1.15×+** |
| Full Elastic (depth+width+sparsity) | **53.7%** | **35.8%** | **75.17** | - |

Full Elastic 只用 35.8% 总参数、53.7% activated 参数，平均分 75.17 vs 75.55——**几乎无损**。

**这个结果的 intuition**：Elastic training 让 representation capacity 在 layers 和 modalities 间重新分配。Sub-network 不是简单"剪掉"一部分参数，而是学会了用剩余参数 compensate。这比 post-hoc compression 高明得多——compression 是"事后补救"，elastic 是"天生适应"。

参考 [MatFormer](https://arxiv.org/abs/2310.07710)、[Once-For-All](https://arxiv.org/abs/1908.09791) 的类似思想，但 ERNIE 5.0 首次把它用到 trillion-scale MoE pre-training 上。

---

## 四、RL 训练怎么稳定的

RL 是这 paper 技术密度最高的部分。万亿参数 MoE + 多模态 + RL，三个 difficulty 叠加，问题爆炸。

### 4.1 Rollout 效率：U-RB

RL 训练 90% 时间花在 rollout generation。问题是 response length 长尾分布——少数极长 response 拖死整个 batch。

APRIL（[paper](https://arxiv.org/abs/2509.18521)）的方案是超额 provisioning，够了就停。但会导致 easy query 先完成、hard query 延后，data distribution 偏向 easy。

U-RB 的解法：**给每个 iteration 分配一个 data group，这个 group 必须完整（包括长尾），但其他 group 可以在等待时并行生成。**

具体来说，建两个 pool：
- Inference pool（大容量，存正在生成的 rollouts）
- Training pool（小容量，存完成的 rollouts）

Iteration $t$ 开始时，inference engine 并行生成。只有分配给 iteration $t$ 的 data group $\mathcal{D}_t$ 完成后（最长那个 rollout 到 [EOS]），才移到 training pool 更新参数。其他 group 继续在 inference pool 里生成，给后续 iteration 用。

**Intuition**：这就像餐厅厨房——APRIL 是"谁先做好谁先上"，导致简单的菜先上、难的菜堆积。U-RB 是"每桌必须上齐才能走"，但厨房同时做多桌，不 idle。

### 4.2 Entropy Collapse：MISC + WPSM

Entropy collapse 是 RL 训练的幽灵——policy entropy 突然塌缩，模型失去 exploration 能力，陷入 repetitive output。

在 MoE + 多模态下，这个问题被放大：

1. **Train-inference mismatch**：training engine 和 inference engine 分离，数值不一致。MoE 的 dynamic routing 让同一 token 可能 routed 到不同 expert，mismatch 被放大。
2. **Easy query over-fitting**：早期模型快速 fit easy query，entropy 塌缩，失去探索 hard query 的能力。

**MISC（Multi-granularity Importance Sampling Clipping）**：

先看 IcePop（[paper](https://arxiv.org/abs/2510.18855)）的 baseline，它用 double-sided masking 修正 train-inference mismatch：

$$\mathfrak{M}\left(\frac{\pi_{train}(y_{i,j} | x, y_{i,<j}; \theta_{old})}{\pi_{infer}(y_{i,j} | x, y_{i,<j}; \theta_{old})}; \alpha, \beta\right)$$

- $\pi_{train}$：training policy
- $\pi_{infer}$：inference policy
- $\theta_{old}$：旧参数
- $y_{i,j}$：第 $i$ 个 rollout 的第 $j$ 个 token
- $y_{i,<j}$：前 $j$ 个 token（causal context）
- $\alpha, \beta$：masking 的下上限

如果 train/inference ratio 超出 $[\alpha, \beta]$，这个 token 的 gradient 被 mask 掉。

GSPO（[paper](https://arxiv.org/abs/2507.18071)）把它改成 sequence-level，用 geometric mean：

$$s_i(\theta) = \left(\frac{\pi_{train}(y_i | x; \theta)}{\pi_{train}(y_i | x; \theta_{old})}\right)^{1/|y_i|}$$

- $|y_i|$：sequence length
- $s_i(\theta)$：sequence-level importance ratio

但直接用 GSPO + IcePop 在 ERNIE 5.0 上还是 collapse——sequence-level masking 会 prune 掉大量 low-entropy response。

**MISC 的改进**：masking 在 **token 粒度** 做（$\mathfrak{M}_{j \in [1, |y_i|]}$），但 importance ratio 仍用 sequence-level $s_i(\theta)$。这样既避免了 token-level ratio 的长序列爆炸，又避免了 sequence-level masking 的过度 pruning。

**Intuition**：粒度选择是个 trade-off。Token-level ratio 对长序列不稳定（乘积爆炸），sequence-level masking 太粗暴（整个序列要么留要么扔）。MISC 取中间——ratio 用 sequence-level 稳定数值，masking 用 token-level 保留信息。

**WPSM（Well-learned Positive Sample Mask）**：

对 query $x$ 的 rollout group $\mathcal{V}^x = \{y_1^x, ..., y_G^x\}$：

如果平均 accuracy $acc_t^x > \tau$（threshold），且某个 rollout $y_i^x$ 的 policy entropy $\mathcal{H}_{y_i^x}(\pi_\theta) < \eta$（stability bound），则标记为 "well-learned"，在 loss 里降权：

$$\mathbb{M}_{mask}^i = \begin{cases} \alpha & \mathcal{H}_{y_i^x}(\pi_\theta) < \eta \text{ and } acc_t^x > \tau \\ 0 & \text{otherwise} \end{cases}$$

- $\tau$：accuracy threshold
- $\eta$：entropy stability bound
- $\alpha \in [0, 1]$：well-learned response 的 supplementary learning degree

**Intuition**：模型已经掌握的 query（高 accuracy + 低 entropy），再训就是浪费 gradient budget。把这些 positive signal 的权重降低（不是完全屏蔽，保留 $\alpha$ 防止 catastrophic forgetting），把 gradient budget 让给 hard query。这类似 curriculum learning 的逆向版本——不是先学简单的，而是学会简单的就不再花时间。

### 4.3 Sparse Reward：AHRL

GRPO/DAPO 在 hard query 上的死穴：所有 rollout 都 0 reward，没 gradient 信号。

AHRL（Adaptive Hint-based RL）的解法：**给 hard query 注入 partial think sketch，把问题分解。**

Query $x$ 的 response $y = (think, solution)$。AHRL 把 $x$ augment 成 $\tilde{x}^{(p)}$，附加 think 的前 $p_{hint}$ 个 token。

Annealing schedule：

$$p_{hint}(x^t) = p_{initial} \cdot \exp(-\gamma \cdot t \cdot pass_{initial}^x)$$

- $t$：training iteration
- $\gamma$：decay rate
- $pass_{initial}^x$：query $x$ 在 SFT model 上的 pass@k score
- $p_{initial}$：初始 hint 比例

**Intuition**：难 query（低 $pass_{initial}^x$）decay 慢，hint 揭示时间长；简单 query decay 快，快速过渡到 self-exploration。随着训练推进、模型变强，hint 逐渐退场。这本质上是 **curriculum learning 的自适应版本**——hint 比例由 query 难度 + 训练进度共同决定。

参考 [STaR](https://arxiv.org/abs/2203.14465)、[rStar-Math](https://arxiv.org/abs/2501.04519) 的 rationale-augmented RL 思路。

---

## 五、实验结果的人话解读

### 5.1 Language

**Pre-trained model**：ERNIE 5.0-Base 在几乎所有 benchmark 上吊打 DeepSeek V3.2-Exp-Base 和 Kimi K2-Base。特别是 knowledge 任务——ChineseSimpleQA 90.09 vs 78.29，PreciseWikiQA 74.48 vs 61.66。说明 unified pre-training 并没削弱 text 能力，反而因为 cross-modal knowledge sharing 增强了 factual recall。

**Post-trained model**：和 Gemini 3-Pro、GPT-5 (High) 比，ERNIE 5.0 在 knowledge、instruction following、agent 上领先或持平。在极难 reasoning（AIME 2025: 89.06 vs 95.00，HMMT 2025: 79.58 vs 93.33）和 coding（LiveCodeBench: 76.21 vs 86.34）上有 gap。作者很诚实，直接承认了。

**Intuition**：ERNIE 5.0 的 design philosophy 是 "balanced capability"，没有像 o1/Gemini 3 那样 aggressively optimize 极难 reasoning。这可能是因为 unified framework 的 gradient 被多 modality 分摊了，pure text reasoning 的 specialization 程度不及 pure LLM。未来可能需要 test-time compute scaling 来补这个 gap。

### 5.2 Vision

**Image generation (GenEval)**：ERNIE 5.0 达到 90.1，和 Qwen-Image (91.0)、Nano Banana Pro (89.0) 同档。

**Video generation (VBench)**：ERNIE 5.0 的 Semantic score **83.40 超过 Veo3 的 82.49**。这是 unified architecture 的直接证据——semantic 表示从 understanding 任务迁移到 generation，让生成的视频语义更准确。

**Intuition**：传统 diffusion-based video generator 的 semantic 理解来自 text conditioning，是间接的。ERNIE 5.0 的 autoregressive backbone 直接在 unified token space 里建模 semantic，generation 时 semantic 信号是 native 的，所以 VBench-Semantic 能赢 Veo3。

### 5.3 Audio

ASR 在 AISHELL-1 (0.31)、Fleurs-zh (0.83)、LibriSpeech clean (1.16) 上都是 SOTA。

TTS 上 competitive 但不及专门的 CosyVoice 3——这合理，因为 ERNIE 5.0 没做 task-specific TTS optimization。

**Intuition**：Audio 的 depth-wise prediction + Whisper distillation 让 semantic token 质量很高，所以 ASR 强。TTS 需要 fine-grained acoustic control，这块 unified model 还比不过 specialist。

### 5.4 Expert Routing 行为

这是最 insightful 的分析。

**Expert utilization（Fig 8）**：modality-agnostic routing 下，expert 自然分化。Image/video/audio 的 activation 比 text 更集中——因为 visual/audio token 的 functional role 更 homogenous，text token 的 functional role 更 diverse（semantic、syntactic、factual、reasoning...）。

**Cross-modality IoU（Fig 9）**：
- Text-audio overlap > text-image overlap > text-video overlap
- 深层 layer 里 text 和 visual 的 overlap 增加（从 low-level modality-specific 到 high-level unified semantic）
- Visual understanding 和 visual generation 的 overlap 低（functional role 不同）

**Load balancing（Fig 10）**：Normalized Entropy $NE = \frac{-\sum p_i \log p_i}{\log N}$（$N$ = expert 数，$p_i$ = 路由到第 $i$ 个 expert 的 token 比例）。Text 几乎所有 layer 都高且稳定。**第一层没有 severe imbalance**——反驳了 "early MoE layer 需要 dense 设计" 的假设（DeepSeek-V3 的推测）。

**Big picture insight**：router 自己学到了 modality structure，无需 explicit supervision。Modality label 在 unified representation space 里是 emergent property。这暗示未来的 MoE 设计可能都应该 modality-agnostic，让 expert 按 functional role 自组织。

### 5.5 Elastic Training ablation

Small-scale MoE（64 experts, 454M activated, 3.2B total）：

- **Elastic Depth**：full-depth 性能反而略升（1.941 vs 1.945）——regularization effect。Reduced-depth (12 layers) 的 val loss 2.137，平滑 degradation。
- **Elastic Width**：full-width 几乎无损（1.964 vs 1.957）。Reduced-width (32 experts) 2.218，仍可用。
- **Elastic Sparsity**：top-k 从 8 降到 4，val loss 1.971（几乎无损）；降到 1，2.175（仍 functional）。

Scaling 到 ERNIE 5.0-Exp：full elastic（35.8% params, 53.7% activated）avg score 75.17 vs 75.55。VisualPuzzle 和 ZebraLogic 这种难 reasoning 任务上 robustness 尤其好。

**Intuition**：Elastic training 的成功说明 model 的 representation capacity 是冗余的。传统 "full capacity always on" 的训练方式让 model 依赖这种冗余。Elastic training 强制 model 学会用 subset 就能 work，等于 implicit regularization。Sub-network 继承 full model 知识后，只需少量 mid-training/post-training 就能 deploy。

---

## 六、Infrastructure 的工程亮点

### 6.1 Hybrid Parallelism

4-way TP + 12-way PP (virtual stages) + 64-way EP + ZeRO-1 DP + Context Parallelism + DeepEP。

**No-token-dropping** 是个 bold choice——MoE 训练通常会 drop 超出 capacity 的 token，ERNIE 5.0 全程不 drop，靠 dynamic adaptive offloading（OOM 时 offload activation 到 CPU）+ sub-batch computation + automatic defragmentation 来扛。

**Intuition**：Token dropping 会丢失信息，在多模态下尤其致命（一个 visual token 可能代表重要 spatial 信息）。宁可工程上麻烦点也要保住所有 token。

### 6.2 Tokenizer-Backbone Disaggregation

多 modality tokenizer 的 compute 特性和 MoE backbone 差异大，放一起会 load imbalance。解法：tokenizer 作为独立 service 部署在 dedicated nodes，backbone 通过 remote call 拿 encoded representation。

**Intuition**：这就像 microservice 架构——不同 component 的 resource profile 不同，强行耦合在 homogeneous hardware 上效率低。Disaggregation 让每个 component 用最适合自己的 parallelization 策略。

### 6.3 FlashMask

多模态下 attention mask 复杂——visual 要 bidirectional，text/audio 要 causal，同 batch 内不同 sample 的 mask pattern 可能不同。FlexAttention 支持灵活 mask 但同 batch 内变化时效率低。

FlashMask（[paper](https://arxiv.org/abs/2410.01359)）：operator-level 比 FlexAttention 快 200%，end-to-end 训练加速 20%，和 Context Parallelism 集成比 Megatron-LM 快 80%。

### 6.4 RL Infrastructure

- Disaggregated control plane：centralized controller 异步协调 training/inference/environment/reward
- Unified FP8 stack：training 和 inference 用相同 operator，最小化数值 mismatch
- Replay buffer：缓解异步 rollout 的 sequence-length bias
- Elastic CPU pooling：idle CPU capacity 给 environment interaction / result verification

---

## 七、我的一些联想和 speculation

### 7.1 为什么 text-audio overlap 高

Text 和 audio 的 RVQ 第一个 code（semantic token）在 functional role 上 isomorphic——都 encode linguistic content。Router 看到的 token representation 在 semantic 层面是 aligned 的，自然 route 到同一批 expert。

Video 缺乏这种 linguistic alignment，所以和 text 的 overlap 低。这暗示 **future modality 的加入应该优先考虑如何 align 到 linguistic semantic space**——比如 video 的 scene description token、music 的 chord progression token。

### 7.2 Visual understanding vs generation 的 low overlap

这暴露了 unified representation 的一个 tension：understanding 要 semantic abstraction（丢弃 detail），generation 要 fine-grained detail（保留 detail）。虽然 paper 声称 unified，但 expert 层面仍有分化。

可能的改进方向：adversarial training 强制 alignment，或者 explicit 的 shared subspace constraint。或者干脆接受这种分化，让 router 自己决定何时用 understanding expert、何时用 generation expert。

### 7.3 Elastic training 的理论解释

为什么 sub-network 能继承 full model 性能？我的 speculation：

Elastic sampling 相当于在 parameter space 上做了 stochastic perturbation。每个 sub-network 对应 parameter space 的一个 slice，full model 是所有 slice 的 union。训练时随机 sample slice，等于在 union 上做 SGD——每个 parameter 都被多个 slice 的 gradient 更新，representation 被迫 generalized。

这类似 dropout 的解释——dropout 是 neuron-level perturbation，elastic training 是 structure-level perturbation。两者都是 implicit regularization，让 model 不 overfit 到特定 structure。

### 7.4 MISC 的粒度选择为什么有效

Token-level ratio 在长序列下乘积爆炸（$\prod_{j=1}^{|y_i|} r_{i,j}$），sequence-level ratio（geometric mean $s_i = (\prod r_{i,j})^{1/|y_i|}$）稳定但 masking 粒度太粗。

MISC 的 hybrid 策略：ratio 用 sequence-level 稳定数值，masking 用 token-level 保留信息。这就像图像处理里的 "coarse-to-fine"——先在 coarse level 确定大方向（sequence-level ratio），再在 fine level 做局部调整（token-level masking）。

### 7.5 AHRL 和 STaR 的关系

STaR（[paper](https://arxiv.org/abs/2203.14465)）用 rationale augmentation 让 model 生成 reasoning chain，过滤正确的作为 training data。AHRL 的区别在于：

- STaR 是 offline filtering，AHRL 是 online RL
- STaR 用 full rationale，AHRL 用 partial hint + annealing
- AHRL 的 annealing schedule 由 query 难度（pass@k）自适应

两者本质上都是 **scaffolding**——给 model 一个 "脚手架" 让它能 reach 到 hard query，然后逐渐撤掉脚手架。这可能是未来 RL 训练的标准 trick。

### 7.6 第一层没有 routing imbalance 的原因

DeepSeek-V3 推测 early MoE layer 需要 dense 设计，因为早期 representation 还没分化，router 容易 collapse。但 ERNIE 5.0 的实验发现第一层 NE 很正常。

我的 speculation：modality-agnostic routing 反而帮助了早期稳定。传统 modality-specific routing 在第一层就要做 "硬决策"（分配到哪个 modality pool），容易 collapse。Modality-agnostic 让所有 token 进入 shared pool，router 有更大自由度探索，反而稳定。

### 7.7 Cascaded diffusion refiner 的 limit

Backbone 生成低分辨率 + semantic layout，refiner 做超分。这个 design 的潜在问题：refiner 不知道 backbone 的 internal reasoning，可能在超分时引入 inconsistency。

Future direction：iterative refinement 让 refiner 反馈给 backbone，或者让 backbone 直接输出 multi-resolution representation 给 refiner。甚至可能用 diffusion in token space 而非 pixel space。

### 7.8 万亿参数 + <3% activation 意味着什么

Activation rate < 3% 意味着 expert 数量极多（可能数千）。这是 fine-grained MoE 的路线（[DeepSeek-V3](https://arxiv.org/abs/2412.19437) 的 256 experts per layer 是类似思路）。

Fine-grained MoE 的优势：每个 expert 专注 narrow functional role，routing 更 precise，knowledge 更 disentangled。劣势：communication overhead 大（需要 DeepEP 这种专用库），inference 时的 expert switching 成本高。

Elastic training 正好 mitigate 了 inference 成本——latency-sensitive 时减少 top-k，throughput-sensitive 时用 full top-k。

### 7.9 为什么不 elastic hidden dimension

Paper 明确提到 elastic hidden dimension 是 future extension。我猜测原因：

Hidden dimension elasticity 涉及 weight matrix 的 sub-block sampling，比 layer/expert/k 的 elasticity 复杂得多。Layer 可以直接 skip，expert 可以直接不 route，但 hidden dimension 的 sub-block 会影响所有 dependent computation（attention、FFN、projection）。

参考 [Mixture-of-Hidden-Dimensions](https://arxiv.org/abs/2412.05644) 的方案，可能需要在 attention 和 FFN 内部做 dynamic slicing，工程复杂度高。但收益也大——hidden dimension 是 model capacity 的最直接维度。

### 7.10 RL 在 multimodal 上的 future

Paper 揭示的核心问题：ultra-sparse MoE 在 RL 中放大 train-inference mismatch。这其实是个 general problem——任何有 dynamic routing 的 architecture 在 RL 中都会遇到。

Possible solutions：
- Unified train-inference engine（paper 提到的 unified FP8 stack 是 step 1）
- Router consistency loss（强制 train/inference router 一致）
- Stochastic routing in training（让 training 也用 inference 的 sampling-based routing）

AHRL 的 hint annealing 也可能 generalize 到其他 sparse reward 场景——比如 agent task 的 sub-goal decomposition、math 的 partial solution hint。

---

## 八、总结：ERNIE 5.0 的真正贡献

抛开 marketing 话术，ERNIE 5.0 的技术贡献我认为是：

1. **证明了 unified autoregressive multimodal 在 trillion-scale 可行**——ability seesaw 不是 unification 的必然代价，关键在于 tokenization 和 training paradigm 的设计。

2. **Elastic training as pre-training paradigm**——这可能改变未来 model 开发流程。传统 "train large → compress → deploy" 的 pipeline 可能被 "train elastic → instantiate on demand" 取代。

3. **Modality-agnostic routing 的实证成功**——简化设计有时优于 explicit specialization。Router 自己学到比 human heuristic 更好的 expert allocation。

4. **MoE-specific RL stabilization 技术栈**——MISC、WPSM、U-RB、AHRL 组合解决了 ultra-sparse MoE + multimodal RL 的稳定性问题。这些技术对任何大 MoE model 的 RL 训练都有参考价值。

5. **详尽的 expert routing visualization**——对社区理解 MoE behavior 有重要价值。特别是 "第一层不需要 dense 设计"、"text-audio overlap 高"、"understanding-generation overlap 低" 这些发现，对 future MoE 设计有直接指导意义。

**诚实的 limitation**：long-horizon reasoning 上和 Gemini 3-Pro 有 gap。这可能是 unified framework 的 inherent cost——pure text reasoning 的 specialization 程度不及 pure LLM。Future work 可能需要 test-time compute scaling 或更 aggressive 的 reasoning-specific post-training。

---

参考资源：
- [ERNIE 4.5 Technical Report](https://yiyan.baidu.com/blog/publication/ERNIE_Technical_Report.pdf)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [VAR: Visual Autoregressive Modeling](https://arxiv.org/abs/2404.02905)
- [Infinity: Bit-wise Autoregressive](https://arxiv.org/abs/2412.04431)
- [FlashMask](https://arxiv.org/abs/2410.01359)
- [FlexAttention](https://arxiv.org/abs/2412.05496)
- [GSPO](https://arxiv.org/abs/2507.18071)
- [APRIL](https://arxiv.org/abs/2509.18521)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [MatFormer](https://arxiv.org/abs/2310.07710)
- [Once-For-All](https://arxiv.org/abs/1908.09791)
- [IcePop](https://arxiv.org/abs/2510.18855)
- [Whisper](https://arxiv.org/abs/2212.04356)
- [EnCodec](https://arxiv.org/abs/2210.13473)
- [SpeechTokenizer](https://arxiv.org/abs/2308.16656)
- [WSD Learning Rate (MiniCPM)](https://arxiv.org/abs/2404.06395)
- [BPE-dropout](https://arxiv.org/abs/2010.01423)
- [Ring Attention](https://arxiv.org/abs/2310.01889)
- [Echo Chamber (RL amplifies pretraining)](https://arxiv.org/abs/2504.07912)
- [ProRL](https://arxiv.org/abs/2505.24864)
- [STaR](https://arxiv.org/abs/2203.14465)
- [rStar-Math](https://arxiv.org/abs/2501.04519)
- [Mixture-of-Hidden-Dimensions](https://arxiv.org/abs/2412.05644)
- [Qwen-Image](https://arxiv.org/abs/2508.02324)
- [HunyuanVideo](https://arxiv.org/abs/2412.03603)
- [UniTok](https://arxiv.org/abs/2502.20321)
- [Veo 3](https://deepmind.google/models/veo/)
- [Gemini 3](https://blog.google/products-and-platforms/products/gemini/gemini-3/)
- [GShard](https://arxiv.org/abs/2006.16668)
- [Megatron-LM](https://arxiv.org/abs/1909.08053)
- [GPIPE](https://arxiv.org/abs/1811.06965)
- [ZeRO](https://arxiv.org/abs/1911.04460)

---

# ERNIE 5.0 Technical Report 深度解析

## 一、整体定位与设计哲学

ERNIE 5.0 是百度于 **2026年2月4日** 发布的万亿参数级 foundation model，核心定位是 **"natively autoregressive unified multimodal foundation model"**——从 scratch 训练，将 text、image、video、audio 四种 modality 统一在 **Next-Group-of-Tokens Prediction** 目标下，同时支持 understanding 与 generation。

这背后的核心 insight 是：传统 late-fusion 方案（用预训练 LM + modality-specific decoder，如 Seedream、Qwen-Image 等）存在 **"ability seesaw"** 问题——加强生成会削弱理解，加强理解会牺牲生成。ERNIE 5.0 试图通过 **formal unification** 在单一 autoregressive backbone 中同时建模两者，让 semantic 信号引导 generation，generation 训练反过来强化 fine-grained perception。

**关键 architectural facts：**
- Ultra-sparse MoE，activation rate **< 3%**
- Modality-agnostic expert routing（router 不知道 token 来自哪个 modality）
- Auxiliary-loss-free load balancing（来自 DeepSeek 的方案）
- Trillion-parameter scale，公开披露的首个 production-scale 万亿参数 unified autoregressive model

参考链接：
- [DeepSeek-V3 Auxiliary-loss-free load balancing](https://arxiv.org/abs/2408.15664)
- [DeepMoE / DeepEP 通信库](https://github.com/deepseek-ai/DeepEP)

---

## 二、架构详解

### 2.1 Unified Autoregressive Backbone

ERNIE 5.0 把所有 modality 投影到 shared token space，serialized 成统一序列。统一的优化目标在不同 modality 下有不同形式：

- **Text**：标准 Next-Token Prediction (NTP) + Multi-Token Prediction (MTP)（参考 DeepSeek-V3，[MTP paper](https://arxiv.org/abs/2404.19737)）
- **Vision**：Next-Frame-and-Scale Prediction (NFSP)
- **Audio**：Next-Codec Prediction (NCP)

这种设计的关键 intuition 是：text 是 1D causal sequence，image 是 2D spatial，video 是 3D spatiotemporal，audio 是 hierarchical codec——它们在 **token-level 的预测范式** 上可以统一为 "predict the next group of tokens"，从而避免 modality boundary 和 inconsistent optimization trajectory。

**Modality-Agnostic Routing 的核心 motivation：**

传统多模态 MoE（如 ERNIE 4.5）采用 modality-isolated routing，需要人工分配 expert 给不同 modality。当 modality 数量 ≥ 3 时这种 heuristic 不可行。ERNIE 5.0 让 router 基于 unified token representation 决策，不告诉 router token 属于哪个 modality。结果是 expert 会 **emergent specialization**——后面实验分析（Sec 6.4.1）显示 expert 自然分化出 text-heavy / visual / audio 倾向。

### 2.2 Visual Modeling

#### 2.2.1 Vision Tokenization：Next-Frame-and-Scale Prediction

这是 ERNIE 5.0 最具创新性的部分。Image 被视为 single-frame video 的特例。

**Tokenizer 训练流程：**
1. 先训练一个 **causal 2D multi-scale tokenizer**（用于 image），通过大规模 image pre-training 获得强 spatial 表示
2. 再 **inflate** 成 causal 3D convolutional tokenizer，统一 image 与 video
3. 辅助监督：
   - GAN-based discriminator 的 adversarial loss（提升 distributional fidelity，参考 [StyleGAN](https://arxiv.org/abs/1812.04948)）
   - Semantic regularization loss（来自大型 vision foundation model，保留 high-level semantic 一致性）

**Bit-wise quantization**：visual latent 表示被量化成一组 bit-codes，bit 数直接对应离散 vocabulary 大小（参考 [Infinity](https://arxiv.org/abs/2412.04431)，bit-wise autoregressive modeling）。

**Progressive tokenizer switching**：训练初期用 low-bit tokenizer（小 vocabulary），逐渐切换到 high-bit（大 vocabulary）。这背后的 intuition 是让 backbone 先学 coarse-grained 表示再学 fine-grained，类似 curriculum learning，缓解早期训练不稳定。

#### 2.2.2 NFSP 公式化与 Uni-RoPE

**NFSP 范式：**
- Image generation：Next-Scale Prediction，类似 [VAR](https://arxiv.org/abs/2404.02905)，从低分辨率 scale 到高分辨率 scale 逐级预测
- Video generation：在 NFSP 基础上加 Next-Frame Prediction，沿时间维度逐帧预测
- **Scale-wise causal attention mask**：当前 scale 内 tokens **bidirectionally 可见**（parallel 预测），previous scales 和 historical frames 以 causal（单向）方式可见

**Uni-RoPE 公式：**

$$
\text{Uni-RoPE}_i = (t_i, h_i, w_i), \quad i \in \{1, \ldots, N\}
$$

变量含义：
- $i$：token 在 unified sequence 中的 index
- $N$：unified sequence 总长度
- $t_i$：temporal position（时间维度位置）
- $h_i$：height position（空间高度位置）
- $w_i$：width position（空间宽度位置）

**Modality-specific 设置：**
- **Text 和 Audio tokens**：$t_i = h_i = w_i$，三者相等，退化成标准 1D RoPE（按 sequence index 单调递增）
- **Visual tokens**：$t_i$ 用于 frame indexing（monotonic 增加以保持 temporal ordering），$(h_i, w_i)$ 对应帧内 spatial location
- **Center-aligned coordinate**：不同 scale 的 token 基于几何中心对齐，保证跨 scale 的 spatial consistency

**Intuition**：传统 RoPE 只处理 1D，而 visual token 有 2D/3D 结构。Uni-RoPE 把 RoPE 的 frequency 分解到三个维度，让 attention 能感知 "这个 token 在哪一帧、哪个空间位置"。Center-alignment 是为了让 low-resolution scale 的 token 和 high-resolution scale 的 token 在 spatial 维度上对齐——低分辨率 scale 的中心 token 应该"看到"高分辨率 scale 的中心区域。

**抗 error accumulation 训练 trick：**

训练时随机 flip 历史 token 的 bits，让模型 self-correct 到 ground-truth。这背后的 intuition 是：autoregressive visual generation 序列极长，error 会累积，通过 corruption + self-correction 训练让模型 robust。

**Cascaded Diffusion Refiner：**

Backbone 生成低分辨率 + 精确 semantic/layout，refiner 单独训练做超分。Refiner 用 paired low-res（controlled degradation）+ high-res 训练。Decoupled 训练避免 autoregressive loss 和 diffusion loss 在共享 backbone 中的 optimization conflict。

#### 2.2.3 Dual-Path Hybrid Representation

**问题**：visual feature 在 quantization 前被 downsample 到低维，会丢失 fine-grained semantic 信息，限制 understanding 任务性能（参考 [UniTok](https://arxiv.org/abs/2502.20321)）。

**解决方案**：直接利用 quantization 前的 dual-path features。

**形式化：**

给定 image 中的 spatial token 或 video 中的 spatio-temporal token，提取两组 features：

$$
\mathbf{F}_{cnn} \in \mathbb{R}^{N \times K \times D_{cnn}}, \quad \mathbf{F}_{vit} \in \mathbb{R}^{N \times K \times D_{vit}}
$$

变量含义：
- $N$：visual understanding token 数量
- $K$：每个 token 分组的 local patches 数量（image: $K=4$ spatially adjacent；video: $K=16$，跨 4 个相邻帧）
- $D_{cnn}$：CNN feature 维度
- $D_{vit}$：ViT feature 维度

**Attention-Based Patch Merger 流程：**

1. 把 CNN feature 投影到 ViT feature 空间（维度对齐）
2. 在 patch 维度 concatenate：$\mathbf{F}_{mrg} \in \mathbb{R}^{N \times 2K \times D_{vit}}$
3. Multi-head self-attention：$\mathbf{Z} = \text{Attn}(\mathbf{F}_{mrg})$，输出 $\mathbf{Z} \in \mathbb{R}^{N \times 2K \times D_{vit}}$
4. Patch 维度 mean pooling：$\mathbf{F}_{out} \in \mathbb{R}^{N \times D_{vit}}$
5. 投影到 unified backbone 的 embedding 维度

**Intuition**：CNN 擅长 local perceptual detail（边缘、纹理），ViT 擅长 global semantic。Naive MLP fusion 会引入 representational interference。Attention 让 CNN 和 ViT patches 互相 "看"，自适应地聚合局部 patch 和 high-level semantic，同时建模 spatial/temporal 依赖。最终 visual token 数从 $N \times K$ 压缩到 $N$，得到 compact yet expressive 的表示——这对统一框架至关重要，因为 generation 任务（pixel-level editing）需要 fine-grained 信息，而 understanding 任务需要 semantic abstraction。

### 2.3 Audio Modeling

#### 2.3.1 Audio Tokenization

- Token rate：**12.5 Hz**（每秒 12.5 个 token）
- 设计：Residual Vector Quantization (RVQ)，参考 [EnCodec](https://arxiv.org/abs/2210.13473)、[SpeechTokenizer](https://arxiv.org/abs/2308.16656)
- **第 1 个 token**：encode high-level audio semantics（linguistic + phonetic cues）
- **剩余 tokens**：encode residual acoustic information，从粗到细（timbre、prosody）

**Whisper distillation**：第 1 个 audio token 的表示与 [Whisper](https://arxiv.org/abs/2212.04356) encoder output 对齐。Whisper 表示通过 average pooling 匹配 12.5 Hz 的 token rate，解决 teacher-student 时间不匹配问题。

#### 2.3.2 Next-Codec Prediction (NCP)

**核心问题**：RVQ 产生多 codebook tokens，如果 flatten 成单一序列会过长。

**Depth-wise autoregressive architecture**：

- **Understanding**：每个 audio token 包含多个 discrete codes（对应不同 residual level）。每个 level 通过 level-specific embedding matrix 映射，所有 level 的 embedding **相加** 形成 codec 表示（additive aggregation 反映 residual nature）。最终 audio token 与 text token 一起进入 unified backbone。
  
- **Generation**：NCP 在 top transformer layers 插入多个 audio heads，支持 depth-wise prediction。流程：
  1. 条件于 multimodal context，先预测第 1 个 semantic code
  2. 生成后，把 code 映射回 embedding，**加回** hidden state
  3. 加回后的 hidden state 条件化下一 level 的预测
  4. Teacher forcing 训练（用 ground-truth code 的 feedback embedding）
  5. 直到所有 level 预测完毕，audio decoder 转成 waveform
  6. Speech synthesis 时插入 speaker embedding 控制 timbre

**Intuition**：NCP 把 RVQ 的多 codebook 跨层分布到 transformer 深度上，每层负责一个 granularity level。这样既保持了 coarse-to-fine 的 prediction 范式（类似 image 的 multi-scale），又避免了序列过长。Depth-wise additive embedding 在 understanding 和 generation 之间保持结构对齐——这是 unified framework 的关键。

---

## 三、Pre-Training

### 3.1 Pre-Training Data

- **Text**：multilingual web crawls、curated corpora、books、scientific publications、code、structured knowledge
- **Tokenizer**：UTF-16BE 编码（stable byte-level fallback + compact non-Latin 表示），BPE dropout（参考 [BPE-dropout](https://arxiv.org/abs/2010.01423)）减少对 frequent pattern 的过拟合
- 中文等无空格语言：filter 长无空格 phrase（可被标准分词工具分解的），减少 vocabulary sparsity
- **Multimodal**：image-text、video-text、audio-text pairs + interleaved multimodal sequences
- 严格 preprocessing：heuristic + model-based filter、deduplication、decontamination
- 最终 corpus：trillions of text tokens + 大量 multimodal instances

### 3.2 Training Recipe

**Stage 1: 8K Pre-Training**
- Context length: 8K
- Learning rate: **WSD schedule** (Warmup-Stable-Decay，参考 [MiniCPM](https://arxiv.org/abs/2404.06395))
  - Linear warmup 2,000 steps，从 0 到 peak $1 \times 10^{-4}$
  - 保持 constant
- Batch size scheduling: 14M tokens → 56M tokens（逐渐增大）
- RoPE base = **1,000,000**（一开始就设大，避免后续 context extension 时的 reparameterization/interpolation）

**Stage 2: 32K & 128K Mid-Training**
- 逐步扩展 context length
- Cosine learning rate schedule，从 $1 \times 10^{-4}$ anneal 到 $1 \times 10^{-5}$
- Batch size 不变

**MoE-specific hyperparameters：**
- Auxiliary-loss-free load balancing bias update speed: $1 \times 10^{-4}$ (8K stage) → $1 \times 10^{-5}$ (mid-training)，抑制大 scale MoE 训练中的 iteration-level oscillation
- MTP loss weight: 0.3 (8K) → 0.1 (mid-training)
- **Posterior-based loss weighting**：把不同 modality 的 autoregressive loss rescale 到同一区间，防止 modality 间 imbalance

**Intuition**：WSD schedule 的 stable phase 让模型在 peak LR 持续吸收数据，最后 decay 精修。Batch size scheduling 类似 Pile 训练经验——前期小 batch 加速，后期大 batch 稳定。RoPE base 提前设大是为了 128K extension 时无需 NTK-aware 等技巧。

### 3.3 Once-For-All Elastic Training

这是 ERNIE 5.0 的一大亮点，源自 [Matformer](https://arxiv.org/abs/2310.07710)、[Once-For-All](https://arxiv.org/abs/1908.09791) 思想，但首次应用到 pre-training 的 MoE 架构上。

**Motivation**：传统 "train-then-compress"（pruning、distillation）有专门 infra 成本，且 compress 后架构固定。ERNIE 5.0 想在 single pre-training run 中同时优化 family of sub-networks。

**Three orthogonal elastic dimensions：**

#### Elastic Depth
- 75% 概率用 full-depth network
- 25% 概率 sample reduced-depth sub-network
- **Intuition**：让中间层 representation 即使在被 bypass 时仍保持 informative。类似 [DropPath / LayerDrop](https://arxiv.org/abs/1909.11556)，但目标是 elastic deployment

#### Elastic Width
- 80% 概率激活所有 experts（full width）
- 20% 概率 restrict routing 到随机 subset of experts
- **Intuition**：让模型在 partial experts 下仍能 work，支持 memory-constrained 部署

#### Elastic Sparsity
- 80% 概率用默认 routing top-k
- 20% 概率从预定义 range 中随机 sample 较小的 k
- **Intuition**：让模型在 latency-constrained 场景下减少 activated experts 数

**Single backprop 同时优化 full model + sampled sub-model**，所有用同一 autoregressive objective。

**Elastic sub-network 可作为后续 mid-training / fine-tuning 的起点**，避免训练多个独立模型或 post-hoc compression。

参考：[MatFormer](https://arxiv.org/abs/2310.07710)、[Flextron](https://arxiv.org/abs/2406.10260)、[Elastic MoE](https://arxiv.org/abs/2509.21892)

---

## 四、Post-Training

Post-training pipeline：SFT → Unified Multimodal RL (UMRL)。

RL 训练面临三大挑战：
1. RL 计算昂贵（>90% 时间花在 rollout）
2. Ultra-sparse MoE 放大 training-inference discrepancy
3. Multi-modality + multi-scenario 复杂度远超单 task RLVR

### 4.1 Unbiased Replay Buffer (U-RB)

**问题**：rollout response length 长尾分布——少数极长 response 拖累整个 batch，GPU idle。

**APRIL 方案的局限**（[APRIL paper](https://arxiv.org/abs/2509.18521)）：超额 provisioning，达到目标数量就停止生成，未完成的下轮续。问题是：容易的 query 先完成，难的长 horizon query 延后，导致 **non-stationary data difficulty distribution**——模型先学简单样本，后期才碰难样本，可能阻碍收敛。

**U-RB 方案：**

两个 pool：
- **Inference pool** $\mathcal{P}_{infer}$，容量 $\Omega_{RBS} = \Omega_{BS} \times N$（$\Omega_{BS}$ 训练 batch size，$N$ buffer size）
- **Training pool** $\mathcal{P}_{train}$，容量 $\Omega_{BS}$

**流程：**
1. Iteration $t$ 开始时，inference engine $\pi_{infer;\theta_t}$ 并行生成 rollouts 填充 $\mathcal{P}_{infer}$
2. **Data-ordering constraint**：只有分配给 iteration $t$ 的 data group $\mathcal{D}_t$ 才能参与后续 training
3. Inference 持续直到 $\mathcal{D}_t$ 中**最长** rollout 达到 terminal state（[EOS]）
4. $\mathcal{D}_t$ 的 rollouts 从 $\mathcal{P}_{infer}$ 移到 $\mathcal{P}_{train}$
5. Training engine $\pi_{train;\theta_t}$ 用这些 rollouts 更新参数

**Intuition**：APRIL 的 "stop early" 让 data distribution 偏向 easy query。U-RB 的 data-ordering constraint 强制每个 iteration 的 data group 必须完整（包括长尾），但同时让其他 iteration 的 data group 在等待期间并行生成。这样既不 idle GPU，又保持 unbiased data distribution。

### 4.2 MISC: Multi-granularity Importance Sampling Clipping

**Entropy Collapse 原因**（参考 [Cui et al. 2025](https://arxiv.org/abs/2505.22617)）：
1. Train/inference 分离 engine 引入数值不一致，MoE 动态 routing 放大这个 mismatch
2. Policy 早期 overfit easy queries，加速 entropy collapse，限制发现 alternative reasoning path

**IcePop / GSPO 基础**（[IcePop](https://arxiv.org/abs/2510.18855)、[GSPO](https://arxiv.org/abs/2507.18071)）：

IcePop 通过 double-sided masking calibration 修正 GRPO 的 train-inference mismatch：

$$
\widehat{\mathbf{J}}_{IcePop}^{GRPO}(\theta) = \mathbb{E}_{x \sim D, \{y_i\}_{i=1}^G \sim \pi_{infer}(\cdot | x; \theta_{old})} \frac{1}{G} \sum_{i=1}^G \mathfrak{M}\left(\prod_{j=1}^{|y_i|} \frac{\pi_{train}(y_{i,j} | x, y_{i,<j}; \theta_{old})}{\pi_{infer}(y_{i,j} | x, y_{i,<j}; \theta_{old})}; \alpha, \beta\right) \cdot \min(r_{i,j} \hat{A}_{i,j}, \text{clip}(r_{i,j}, 1-\epsilon, 1+\epsilon) \hat{A}_{i,j})
$$

变量含义：
- $x$：query
- $\{y_i\}_{i=1}^G$：group of $G$ rollouts 从 inference policy $\pi_{infer}(\cdot | x; \theta_{old})$ 采样
- $y_{i,j}$：第 $i$ 个 rollout 的第 $j$ 个 token
- $y_{i,<j}$：第 $i$ 个 rollout 的前 $j$ 个 token（causal context）
- $\pi_{train}$：training policy
- $\pi_{infer}$：inference policy
- $\theta_{old}$：旧参数
- $r_{i,j} = \frac{\pi_{train}(y_{i,j} | x, y_{i,<j}; \theta)}{\pi_{train}(y_{i,j} | x, y_{i,<j}; \theta_{old})}$：token-level importance ratio
- $\hat{A}_{i,j}$：advantage estimate
- $\epsilon$：PPO clip 范围
- $\alpha, \beta$：masking 下上限和下限
- $\mathfrak{M}(k)$：masking 函数，$k \in [\alpha, \beta]$ 时返回 $k$，否则返回 0（其他变种返回 $\delta$）

**GSPO 版本**（sequence-level）：

$$
\widehat{\mathbf{J}}_{IcePop}^{GSPO}(\theta) = \mathbb{E}_{...} \frac{1}{G} \sum_{i=1}^G \mathfrak{M}\left(\left(\frac{\pi_{train}(y_i | x; \theta_{old})}{\pi_{infer}(y_i | x; \theta_{old})}\right)^{1/|y_i|}; \alpha, \beta\right) \cdot \min(s_i(\theta) \hat{A}_i, \text{clip}(s_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i)
$$

其中：
$$
s_i(\theta) = \left(\frac{\pi_{train}(y_i | x; \theta)}{\pi_{train}(y_i | x; \theta_{old})}\right)^{1/|y_i|} = \exp\left(\frac{1}{|y_i|} \sum_{j=1}^{|y_i|} \log \frac{\pi_{train}(y_{i,j} | x, y_{i,<j}; \theta)}{\pi_{train}(y_{i,j} | x, y_{i,<j}; \theta_{old})}\right)
$$

**Intuition**：GSPO 用 sequence-level geometric mean importance ratio $s_i(\theta)$ 替代 PPO 的 token-level ratio，避免长序列 importance 爆炸。$|y_i|$ 是 sequence length。

**MISC（Mixed）改进**：

直接应用 $\widehat{\mathbf{J}}_{IcePop}^{GSPO}$ 仍导致 entropy collapse（light-blue line in Fig 6），原因是 sequence-level truncated IS 会 prune 掉大量 low-entropy responses。

$\mathfrak{I}_{IcePop}^{Mixed}(\theta)$ 把 masking 从 sequence-level 改为 token-level + sequence-level 混合：

$$
\mathfrak{I}_{IcePop}^{Mixed}(\theta) = \mathbb{E}_{...} \frac{1}{G} \sum_{i=1}^G \left[\mathfrak{M}_{j \in [1, |y_i|]}\left(\frac{\pi_{train}(y_{i,j} | x, y_{i,<j}; \theta_{old})}{\pi_{infer}(y_{i,j} | x, y_{i,<j}; \theta_{old})}; \alpha, \beta\right) \cdot \min(s_i(\theta) \hat{A}_i, \text{clip}(s_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i)\right]
$$

核心变化：masking 在 token 粒度上做（$\mathfrak{M}_{j \in [1, |y_i|]}$），但 importance ratio 仍用 sequence-level $s_i(\theta)$。根据 modality sensitivity 调整 trust region，平衡 exploration-exploitation。

### 4.3 WPSM: Well-learned Positive Sample Mask

**问题**：模型在已 mastered query 上 over-optimize，浪费 gradient budget。

**机制**：

对 query $x$ 和 rollout group $\mathcal{V}^x = \{y_1^x, y_2^x, ..., y_G^x\}$（$G$ 为 group size）：

- 计算平均 accuracy $acc_t^x$ in iteration $t$
- 如果 $acc_t^x > \tau$（threshold），且 rollout $y_i^x$ 的 policy entropy $\mathcal{H}_{y_i^x}(\pi_\theta) < \eta$（stability bound），则 flag 为 "well-learned"

**Mask 公式**：

$$
\mathfrak{I}(\theta) = \mathbb{E}_{x \sim D, \{y_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot | x)} \left[\frac{1}{G} \sum_{i=1}^G [1 - \mathbb{M}_{mask}^i] \min(s_i(\theta) \hat{A}_i, \text{clip}(s_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i)\right]
$$

$$
\mathbb{M}_{mask}^i = \begin{cases} \alpha & \mathcal{H}_{y_i^x}(\pi_\theta) < \eta \text{ and } acc_t^x > \tau \\ 0 & \text{otherwise} \end{cases}
$$

变量含义：
- $s_i(\theta)$：importance ratio
- $\hat{A}_i$：advantage
- $\epsilon$：clip 范围
- $\eta$：entropy stability bound
- $\tau$：accuracy threshold
- $\alpha \in [0, 1]$：well-learned response 的 supplementary learning degree

**Intuition**：把 gradient budget 从 easy（高 accuracy + 低 entropy）转向 hard（sparse reward / diverse reasoning paths）。Mask 不是完全屏蔽，而是降低权重（$1 - \alpha$），保留少量学习信号防止 catastrophic forgetting。

### 4.4 AHRL: Adaptive Hint-based RL

**问题**：当所有 rollouts 都 0 reward（GRPO/DAPO 在 hard query 上的局限），无 gradient 信号。

**核心思想**：注入 partial think sketches，把复杂问题分解为中间步骤。

**形式化**：

Query $x$ 的 response $y = (think, solution)$。AHRL 把 $x$ augment 成 $\tilde{x}^{(p)}$，附加 think 的前 $p_{hint}$ 个 tokens。

**Annealing schedule**：

$$
p_{hint}(x^t) = p_{initial} \cdot \exp(-\gamma \cdot t \cdot pass_{initial}^x)
$$

变量含义：
- $t$：training iteration
- $\gamma$：decay rate
- $pass_{initial}^x$：query $x$ 在 SFT model 上的 pass@k score
- $p_{initial}$：初始 hint 比例

**Intuition**：难 query（低 $pass_{initial}^x$）decay 慢，hint 揭示时间长；简单 query decay 快，快速过渡到 self-exploration。随训练推进，模型变强，hint 逐渐退场，过渡到 full self-exploration。这类似 [STaR](https://arxiv.org/abs/2203.14465)、rStar-Math 等 rationale-augmented RL 思路，但是自适应 schedule。

参考：[ProRL](https://arxiv.org/abs/2505.24864)、[Echo Chamber](https://arxiv.org/abs/2504.07912)

---

## 五、Infrastructure

### 5.1 Hybrid Parallelism

**最终配置：**
- 4-way Tensor Parallelism（[Megatron-LM](https://arxiv.org/abs/1909.08053)）
- 12-way Pipeline Parallelism + virtual stages（[GPIPE](https://arxiv.org/abs/1811.06965)）
- 64-way Expert Parallelism（[GShard](https://arxiv.org/abs/2006.16668)）
- ZeRO-1 Data Parallelism（[ZeRO](https://arxiv.org/abs/1911.04460)）
- Context Parallelism（[Ring Attention](https://arxiv.org/abs/2310.01889)）
- DeepEP for inter-node communication

**No-token-dropping**：训练全程不 drop token，早期会有 OOM 风险（routing 不均衡）。

**Memory strategies：**
1. **FP8 mixed-precision**：activation tensors 用 FP8 存储，降 peak memory
2. **Dynamic adaptive offloading**：forward 时追踪保留的 activation tensors，遇 OOM 时 adaptive offload 到 CPU。无 OOM 时不触发，min overhead
3. **Sub-batch computations**：把大 memory 请求分解为小请求，减少 fragmentation 导致的 OOM
4. **Automatic defragmentation**：基于 CUDA VMM 的 allocator，自动 defrag

### 5.2 Disaggregation Architecture

**问题**：多 modality tokenizer 的 sequence length 和 compute cost 差异大，与 backbone 一起部署在 homogeneous hardware 导致 load imbalance。

**解决方案**：Tokenizer 作为独立 horizontal scalable 服务部署在 dedicated compute nodes，data-parallel 配置。Backbone 通过 remote calls 获取 encoded representations。

### 5.3 FlashMask

**问题**：Visual input 需要 bidirectional attention，text/audio 需要 causal attention。同一 batch 内不同 sample 的 mask pattern 可能不同。FlexAttention ([FlexAttention paper](https://arxiv.org/abs/2412.05496)) 支持灵活 mask，但同 batch 内不同 sample 的 mask 变化时效率低。

**FlashMask**（[paper](https://arxiv.org/abs/2410.01359)）：
- Operator-level: 比 FlexAttention 快 200%
- End-to-end: 训练加速 20%
- 与 Context Parallelism 集成：比 Megatron-LM 方案快 80%

### 5.4 Scalable Disaggregated RL Infrastructure

1. **Disaggregated control plane**：centralized RL controller 异步协调 training、inference、environment interaction、reward evaluation
2. **Unified FP8 stack**：training 和 inference 用 identical high-perf operators，集成 Rollout Router Replay 策略，最小化数值 mismatch
3. **Replay buffer**：缓解异步 rollout 的 sequence-length bias，保持 data order
4. **Elastic CPU pooling**：隔离 idle CPU capacity 给 logic-intensive 任务（RL environment interaction、result verification），提升 TCO

---

## 六、Evaluations

### 6.1 Language Benchmarks

**Pre-trained model（ERNIE 5.0-Base）vs DS V3.2-Exp-Base vs Kimi K2-Base：**

| Category | Benchmark | DS V3.2 | Kimi K2 | ERNIE 5.0-Base |
|----------|-----------|---------|---------|----------------|
| Knowledge | PreciseWikiQA | 52.60 | 61.66 | **74.48** |
| Knowledge | ChineseSimpleQA | 74.19 | 78.29 | **90.09** |
| Knowledge | MMLU-Pro | 68.27 | 67.19 | **75.58** |
| General | MMLU | 88.60 | 88.40 | **90.58** |
| General | BBH | 73.50 | 70.07 | **75.69** |
| STEM | MATH (CoT) | 65.70 | 65.90 | **73.89** |
| STEM | GPQA-Diamond | 53.01 | 48.10 | **57.30** |
| Coding | LiveCodeBench v6 | 24.90 | 26.30 | **31.94** |
| Coding | CRUXEval-O | 71.28 | 81.61 | **84.01** |
| Multilingual | INCLUDE | 77.45 | 72.29 | **77.81** |

ERNIE 5.0-Base 在几乎所有 benchmark 上领先，特别在 knowledge-intensive 任务上 margin 巨大（ChineseSimpleQA: 90.09 vs 78.29）。

**Post-trained model vs SOTA：**

ERNIE 5.0 在 knowledge（SimpleQA: 74.01、ChineseSimpleQA: 86.03）、instruction following（MultiChallenge: 65.98、Multi-IF: 85.56）和 agent（ACEBench-zh: 89.60、BrowseComp-zh: 64.71）上表现卓越。Gemini 3-Pro 在最难的 reasoning（AIME 2025: 95.00、HMMT 2025: 93.33、GPQA-Diamond: 91.90）和 coding（LiveCodeBench v6: 86.34）上领先。ERNIE 5.0 在这些极难 benchmark 上仍有 gap，作者明确承认这一点。

### 6.2 Vision Benchmarks

**Image Generation (GenEval):**

| Model | GenEval |
|-------|---------|
| Nano Banana Pro | 89.0 |
| Seedream 4.0 | 85.4 |
| GPT-Image | 84.0 |
| Qwen-Image | 91.0 |
| ERNIE 5.0-Base | 88.4 |
| ERNIE 5.0 | 90.1 |

ERNIE 5.0（post-trained）达到 90.1，仅次于 Qwen-Image。

**Video Generation (VBench):**

| Model | Quality | Semantic | Overall |
|-------|---------|----------|---------|
| HunyuanVideo-1 | 85.07 | 76.88 | 83.43 |
| Wan2.1-14B-0725 | 85.59 | 76.11 | 83.69 |
| Veo3 | 85.70 | 82.49 | 85.06 |
| ERNIE 5.0-Base | 84.14 | 82.31 | 83.78 |
| **ERNIE 5.0** | 84.40 | **83.40** | 84.20 |

ERNIE 5.0 在 VBench-Semantic 上超越 Veo3（83.40 vs 82.49），体现 unified architecture 中 semantic 表示向生成任务的有效迁移。这印证了 paper 核心 thesis：unified training 让 semantic 信号引导 generation。

### 6.3 Audio Benchmarks

**ASR (WER, lower better)：**
- AISHELL-1：ERNIE 5.0 0.31（最佳）
- Fleurs-zh：ERNIE 5.0 0.83（最佳，超越 Kimi Audio 2.69）
- LibriSpeech clean：ERNIE 5.0 1.16（最佳）

**SEED-TTS (WER, lower better):**

| Model | test-zh | test-en |
|-------|---------|---------|
| Seed-TTS-ICL | 1.11 | 2.24 |
| CosyVoice 3 | 0.71 | 2.57 |
| Qwen2.5-Omni | 1.45 | - |
| ERNIE 5.0 | - | - |

ERNIE 5.0 在 TTS 上达到 competitive 表现，但不及专门的 CosyVoice 3。

---

## 七、关键 Discussion 分析

### 7.1 Modality-Agnostic Expert Routing 行为

**Expert Utilization（Fig 8）：**
- 不同 expert 在 modality-agnostic routing 下仍展现 distinct functional role
- Subset of experts 跨所有 modality 反复激活（universal experts）
- 其余 expert 有强 modality-specific activation pattern
- Image/video/audio 的 expert activation 比 text-only 更集中
- Visual generation 和 audio 任务的 expert activation 比 text 和 visual understanding 更集中

**Cross-Modality IoU（Fig 9）：**

报告 top 25% experts by activation frequency 的 IoU。发现：
- **Text-audio IoU > text-image IoU > text-video IoU**
- 随 layer 加深，text 与 image/video 的 overlap 增加——multimodal representation 从 low-level modality-specific 逐渐 shift 到 high-level unified semantic
- Image understanding 与 video understanding 的 expert overlap 高（符合 image = single-frame video 的设计）
- Image generation 与 video generation 同样 overlap 高
- **Visual understanding vs visual generation 的 overlap 相对低**——理解与生成分到不同 expert 子集

**Load Balancing（Fig 10）：**

Normalized Entropy (NE) 定义：

$$
\text{NE} = \frac{-\sum_{i=1}^N p_i \log(p_i)}{\log N}
$$

变量含义：
- $N$：expert 数量
- $p_i$：路由到第 $i$ 个 expert 的 token 比例
- $\text{NE} \in [0, 1]$，越大表示 expert 利用越 uniform

发现：
- **Text modality**：几乎所有 layer 都高且稳定 NE，最后层轻微下降。**第一层没有严重 imbalance**——这反驳了 "early MoE layer 需要 dense 设计" 的假设（[DeepSeek-V3 推测](https://arxiv.org/abs/2412.19437)）
- **Visual understanding**：最浅和最深 layer 不太 balanced，中间 layer 较 uniform
- **Visual generation 和 audio**：第一层 moderate balance，lower layer entropy 下降，lower-mid 层 partial recovery，higher layer fluctuating drop——指示 expert specialization 和 re-integration 的交替相位

**Insight**：modality-agnostic router 不需要 modality identifier 仍能 self-learn modality structure。这暗示未来可以探索 layer-aware expert allocation、adaptive balancing strategy、modality-shared expert 配置等。

### 7.2 Elastic Training 深度 Ablation

**Small-scale MoE model (64 experts, 454M activated, 3.2B total, 250B tokens, top-k=8)：**

#### Elastic Depth

| Training Config | Inference Config | Val Loss |
|-----------------|------------------|----------|
| Baseline (Layers=16) | Layers=16 | 1.945 |
| Elastic Depth (Layers ∈ [1,16]) | Layers=16 | 1.941 |
| Elastic Depth | Layers=12 | 2.137 |

- Full-depth 性能略有提升（1.941 vs 1.945）——elastic depth 引入 regularization 效应
- Reduced-depth sub-network 平滑可预测的 degradation

#### Elastic Width

| Training Config | Inference Config | Val Loss |
|-----------------|------------------|----------|
| Baseline (Experts=64) | Experts=64 | 1.957 |
| Elastic Width (∈{64,32}) | Experts=64 | 1.964 |
| Elastic Width | Experts=32 | 2.218 |

- Full-width 几乎无 degradation
- Reduced-width sub-network 仍可用

#### Elastic Sparsity

| Training Config | Inference Config | Val Loss |
|-----------------|------------------|----------|
| Baseline (Top-k=8) | Top-k=8 | 1.945 |
| Elastic Sparsity (Top-k ∈ [1,8]) | Top-k=8 | 1.969 |
| Elastic Sparsity | Top-k=4 | 1.971 |
| Elastic Sparsity | Top-k=2 | 2.003 |
| Elastic Sparsity | Top-k=1 | 2.175 |

- 全激活配置 modest degradation
- 显著减少 routing budget 下仍稳定有效

**Scaling to ERNIE 5.0-Exp：**

| Model | AVG | ZebraLogic | LiveCodeBench v6 | TAU2 | MMMU | MathVista | VisualPuzzle | SimpleVQA |
|-------|-----|------------|------------------|------|------|-----------|--------------|-----------|
| ERNIE 5.0-Exp | 75.55 | 95.00 | 73.35 | 79.35 | 74.11 | 83.70 | 59.93 | 63.40 |
| ERNIE 5.0-Exp-ES(25%) | 74.43 | 94.10 | 70.70 | 77.34 | 73.78 | 84.90 | 57.98 | 62.19 |
| ERNIE 5.0-Exp-EA(35.8%) | 75.17 | 95.20 | 70.93 | 77.23 | 75.11 | 84.50 | 60.39 | 62.86 |

**Key findings：**
- **ERNIE 5.0-Exp-ES**：routing top-k 降到 25%，decoding 速度提升 **>15%**，accuracy 仅轻微下降（75.55→74.43）
- **ERNIE 5.0-Exp-EA**：联合 elastic depth/width/sparsity，activated params 53.7%，total params 35.8%，平均分 75.17 vs 75.55——**几乎无损**
- VisualPuzzle 和 ZebraLogic 等难 reasoning 任务上保持强 robustness

**Intuition**：Elastic training 让 model 学会重新分配 representational capacity across layers 和 modalities，而非简单"剪枝"。Sub-network 共享 full model 的知识，继承后只需少量 mid-training/post-training 即可。

---

## 八、关键 Insights 总结

### 8.1 Unification 的代价与收益

ERNIE 5.0 论证了一个重要 thesis：**万亿参数级 unified autoregressive framework 可以同时支持 multimodal understanding 和 generation，且不牺牲 unimodal 性能**。代价是：
- 必须 from scratch 训练（不能 fine-tune 预训练 LM）
- 数据需求巨大（trillions of tokens）
- 优化复杂度高（需要 elastic training、U-RB、MISC、WPSM、AHRL 等一整套技术）

收益：
- 避免 ability seesaw
- Cross-modal knowledge generalization
- Expert emergent specialization（无需手动分配）
- 单次 pre-training 产出 family of deployable models

### 8.2 Modality-Agnostic 是 Anti-pattern 还是 Pattern？

传统 wisdom 是 modality-specific routing 让 expert 专注。ERNIE 5.0 反其道而行：**不告诉 router modality 信息**，让 expert 自组织。实验显示 expert 仍 emerge 出 specialization pattern，但跨 modality 协作更强（text-audio 高度 overlap）。

这背后的深层 insight：**modality 信息可能不是 token representation 的必要维度**。Token 的 functional role（semantic extraction vs perceptual detail encoding）比 modality label 更本质。Router 学到的是 functional specialization，modality 只是 functional role 的 correlated proxy。

### 8.3 Elastic Training 作为 Pre-Training Paradigm

传统 view：先 train 大模型，再 compress 成小模型。ERNIE 5.0 把 elastic 作为 **pre-training 时就内嵌的属性**，sub-network 与 full model 同梯度流优化。这等于让大模型在训练时就已经"知道"如何在不同 capacity 下 work。

这暗示未来的方向：**model 应该 native 支持多 deployment target**，而非事后 hacking。Elasticity 应该和 model 的 representation capacity 同步演化。

### 8.4 RL 在 Multimodal MoE 上的特殊性

Paper 揭示了一个被低估的问题：**ultra-sparse MoE 在 RL 中放大 train-inference mismatch**。Dynamic routing 让同一 token 在不同 engine 上可能 routed 到不同 expert，引入数值不一致。MISC 的 token-level masking + sequence-level importance 是针对这个问题的精巧方案。

WPSM 和 AHRL 则处理另一类问题：**easy query over-fitting** 和 **sparse reward 下无 gradient**。两者都是 GRPO/DAPO 在 hard query 上的 fundamental limitation 的补丁。

---

## 九、可能的未来方向与 Hallucination

基于 ERNIE 5.0 暴露的 limitations：

1. **Long-horizon reasoning gap**：在 AIME/HMMT/LiveCodeBench 等极难 reasoning benchmark 上仍落后 Gemini 3-Pro。可能需要：
   - Test-time compute scaling（类似 [s1](https://arxiv.org/abs/2501.19393)）
   - Hierarchical RL with planning
   - Process reward model 而非仅 outcome reward

2. **Elastic hidden dimension**：paper 没做，未来可加。结合 [MatFormer](https://arxiv.org/abs/2310.07710)、[Mixture-of-Hidden-Dimensions](https://arxiv.org/abs/2412.05644) 实现 4D elastic（depth × width × sparsity × hidden）

3. **Modality-aware routing as post-hoc analysis tool**：用 modality label 作 supervised signal fine-tune router，可能让 expert specialization 更 pronounced 但牺牲 cross-modal generalization

4. **Audio-Text 高度 overlap 的原因**：可能因为 text token 和 audio semantic token（第 1 个 RVQ code）在语义层面是 isomorphic 的（都 encode linguistic content）。Video 缺乏这种 linguistic alignment，所以 overlap 低

5. **Visual understanding vs generation 的 low overlap**：暗示 unified representation 仍有 task-specific 的分化。未来可探索 adversarial training 强制 alignment

6. **Elastic training 的理论分析**：为什么 sub-network 能继承 full model 性能？可能是 elastic sampling 引入的 implicit regularization 类似 dropout，让 representation 不依赖于特定层/expert 的组合

7. **Cascaded diffusion refiner 的 limit**：autoregressive backbone + diffusion refiner 的两阶段方案在物理一致性上可能有 gap（refiner 不知道 backbone 的内部 reasoning）。未来可探索 iterative refinement 让 refiner 反馈给 backbone

8. **MoE expert 数量 scaling law**：paper 提到 activation rate < 3%，但没给 expert 数量 vs performance 的 scaling curve。万亿参数 + <3% activation 意味着 expert 数极多（可能数千），这暗示 fine-grained MoE 是正道（参考 [DeepSeek-V3](https://arxiv.org/abs/2412.19437) 的 256 experts per layer）

9. **AHRL 与 STaR / Rationale-augmented RL 的关系**：AHRL 的 annealing schedule 基于 pass@k，自适应揭 hint 比例。这可能成为未来 RL 训练的标准 trick——把 hint 作为 curriculum signal

10. **U-RB vs importance sampling 的关系**：U-RB 解决 data distribution bias，MISC 解决 importance ratio bias。两者其实是同一问题（distribution mismatch）在不同层面的 manifestation。未来可能 unified framework

---

## 十、最终评价

ERNIE 5.0 是 2026 年初 foundation model 领域的重要 milestone。它的核心贡献：
1. **首个 trillion-scale unified autoregressive model** in production
2. **Elastic training as pre-training paradigm**（非 post-hoc compression）
3. **Modality-agnostic routing** 的实证可行性
4. **MoE-specific RL stabilization**（MISC、WPSM、U-RB、AHRL）

它的诚实之处在于承认了 long-horizon reasoning 上的 gap，且提供了详尽的 ablation 分析。Paper 的 visualization（expert utilization、cross-modality IoU、load balancing NE）对社区有重要参考价值。

作为工程师视角，paper 最值得学习的是：
- **Elastic training 的工程价值**——单次 pre-training 解决 deployment 多样性
- **MISC 在 MoE RL 中的精巧设计**——token vs sequence level granularity 的平衡
- **Modality-agnostic 的反直觉成功**——简化设计有时优于 explicit specialization

参考资源：
- [ERNIE 4.5 Technical Report](https://yiyan.baidu.com/blog/publication/ERNIE_Technical_Report.pdf)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [VAR: Visual Autoregressive Modeling](https://arxiv.org/abs/2404.02905)
- [Infinity: Bit-wise Autoregressive](https://arxiv.org/abs/2412.04431)
- [FlashMask](https://arxiv.org/abs/2410.01359)
- [FlexAttention](https://arxiv.org/abs/2412.05496)
- [GSPO](https://arxiv.org/abs/2507.18071)
- [APRIL](https://arxiv.org/abs/2509.18521)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [MatFormer](https://arxiv.org/abs/2310.07710)
- [IcePop](https://arxiv.org/abs/2510.18855)
- [Whisper](https://arxiv.org/abs/2212.04356)
- [EnCodec](https://arxiv.org/abs/2210.13473)
- [SpeechTokenizer](https://arxiv.org/abs/2308.16656)
- [WSD Learning Rate (MiniCPM)](https://arxiv.org/abs/2404.06395)
- [BPE-dropout](https://arxiv.org/abs/2010.01423)
- [Ring Attention](https://arxiv.org/abs/2310.01889)
- [Echo Chamber (RL amplifies pretraining)](https://arxiv.org/abs/2504.07912)
- [ProRL](https://arxiv.org/abs/2505.24864)
- [Qwen-Image](https://arxiv.org/abs/2508.02324)
- [HunyuanVideo](https://arxiv.org/abs/2412.03603)
- [UniTok](https://arxiv.org/abs/2502.20321)
- [Veo 3](https://deepmind.google/models/veo/)
- [Gemini 3](https://blog.google/products-and-platforms/products/gemini/gemini-3/)
