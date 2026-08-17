---
source_pdf: Intuitive physics understanding emerges from.pdf
paper_sha256: 8ec7320af6b22e8bf91dae7372a6c1c567f217f23debaf4459cbe8d39b9f620d
processed_at: '2026-08-05T10:32:27-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 一句话版本

Yann LeCun 团队拿他们之前搞的 V-JEPA (就是一个看video、猜被遮住部分的self-supervised model),拿去测它懂不懂物理。结果发现 —— 这玩意儿竟然真懂。而且 Google 的 Gemini、阿里的 Qwen2-VL 这些大模型反而基本不懂。

## 这个paper在问啥问题

人从小就有 "直觉物理" —— 你知道球不会凭空消失,知道东西会往下掉,知道两个固体不会穿过去。developmental psychology 里管这个叫 intuitive physics,Elizabeth Spelke 那帮人搞了几十年,认为这是人类天生的 core knowledge。

AI这边一直很尴尬:GPT-4能做数学题、能写代码,但 Moravec's paradox —— 你让它判断"这个球有没有穿墙",它经常搞不定。一个5个月大的婴儿都会的事,几千亿参数的模型不会。

那问题就来了:**这种物理直觉到底要不要 hardcode 进去?还是说某种 general learning principle 就能 emergent 出来?**

## 他们怎么测的

用了一个叫 Violation of Expectation 的心理学套路。原来是给婴儿看两个视频,一个正常,一个违反物理(比如球滚到墙后面就消失了)。看婴儿盯哪个更久 —— 盯得久说明ta"惊讶",说明ta心里有预期,说明ta懂这个物理规律。

他们把这个搬到AI上:让model看一段video的前几帧,预测后面的representation,然后跟真实的representation比,算个distance。这个distance就叫 **surprise**。如果video违反物理,model预测就会偏,Surprise就大。

$$S_t = \left\| p_\phi\left( f_\theta(V_{t:t+C}) \right) - f_{\theta^{EMA}}(V_{t:t+C+M}) \right\|_1$$

说人话就是:
- $V_{t:t+C}$: 给model看的前 $C$ 帧
- $V_{t:t+C+M}$: 真实发生的 $C+M$ 帧(包含未来 $M$ 帧)
- $p_\phi(f_\theta(\cdot))$: model对未来的预测
- $f_{\theta^{EMA}}(\cdot)$: 真实的representation
- L1 distance: 差多少

如果一对video里,那个不可能的video算出来的surprise更大,说明model"识破"了物理违反。

## 核心结果

在三个benchmark上测:

| 方法 | IntPhys | GRASP | InfLevel |
|------|---------|-------|----------|
| 随机初始化网络(对照组) | 50% | 55% | 50% |
| **V-JEPA** | **98%** | **66%** | **62%** |
| VideoMAEv2 (pixel prediction) | 52% | 55% | 52% |
| Qwen2-VL-72B | 50% | 55% | 50% |
| Gemini 1.5 Pro | 50% | 58% | 52% |

V-JEPA 在 IntPhys 上 98%,人类才 85%。**Gemini 和 Qwen2-VL 基本在 coin flip**。这个结果挺震撼的 —— 你以为越大的多模态LLM越懂物理,实际上完全不是。

## 关键发现:预测在哪儿预测,决定了学不学得会

三个对照:
1. **V-JEPA**:在 learned representation space 预测
2. **VideoMAEv2**:在 pixel space 预测(预测每个pixel的RGB)
3. **MLLM (Qwen/Gemini)**:在 text space 预测(输出哪个video impossible)

三个都失败了,V-JEPA 赢了。这说明了啥?

**预测的"东西"决定你学到什么。**

Pixel prediction 的问题:下一个pixel的颜色太多noise了。树叶怎么晃、光线怎么闪、压缩artifact —— 这些根本不可预测。模型为了reduce loss,把capacity都花在memorize这些noise上了,没精力去学"球在墙后面还会继续滚"这种抽象structure。

Representation prediction 的好处:encoder 学到 discard 掉不可预测的noise,只保留predictable的latent factor(物体位置、速度、身份)。predictor在这个干净的空间里predict,自然学到physics-like dynamics。

这就叫 **information bottleneck** —— 你扔掉不可预测的信息,被迫学到世界的因果结构。

## 一个让我意外的ablation

他们试了不同的masking strategy:
- Block masking(默认,mask大块)
- Causal block masking(把最后几帧也mask掉,逼model预测未来)
- Random masking(随机mask 90%的patches)

按理说 causal masking 跟 inference 时的"看过去、猜未来"最匹配,应该最好。结果 **random masking 几乎一样好**,只掉5个点。

这说明:**关键不是怎么mask,而是在representation space做prediction这件事本身**。这就像 LeCun 一直说的,JEPA是个general principle,具体怎么实现不那么critical。

## 另一个让我意外的结果

他们用 HowTo100M 的小子集训练。HowTo100M 总共15年的video。他们subsample到:
- 1289小时:还行
- **128小时(差不多5天video)**:居然还有70%+的accuracy

5天的video就能学到intuitive physics。婴儿看多久video才学会的?这是个很有意思的对照。

## V-JEPA到底是怎么训练的

```
原始video V
   │
   ├──→ mask一下 → V_C (corrupted) ──→ context encoder f_θ ──→ predictor p_φ ──┐
   │                                                                              │
   └──→ complementary mask → V_C_bar ──→ target encoder f_θ_EMA ─────────────────┘
                                                                                  │
                                                          loss = L1 distance ←──┘
```

关键点:
- target encoder 用 EMA 更新,stop-gradient。这个防止 collapse —— 如果两边都trainable,model会trivially输出constant,loss=0但啥也没学到
- EMA update rule: $\theta^{EMA}_{t+1} = (1-\alpha)\theta_t + \alpha\theta^{EMA}_t$, $\alpha$ 从 0.998 升到 1.0
- predictor是个小ViT(12 layers, embed dim 384),输入context encoder的output + mask tokens
- 用 RoPE 编码 3D positional info(height/width/time 各占feature dim的1/3)

整个training不需要negative samples,不需要contrastive loss,就是L1 regression到EMA target。几百行PyTorch能写完。

## 什么地方V-JEPA还不行

- **Color constancy**:学不会,可能encoder把color信息权重调低了
- **Solidity / Collision**:学不会,可能5.33 fps太低,碰撞瞬间(<100ms)被略过
- **需要context的task**:InfLevel里有些task要先看一段"前情提要"video,V-JEPA只能处理3-4秒,memory不够

LeCun自己也承认这些limitation。后面的 V-JEPA 2 加了action conditioning 和更大规模,部分解决了。

## 这篇paper到底告诉我们什么

1. **Intuitive physics 不需要hardwire**。不需要给model写个physics engine,不需要预先告诉它"物体是solid的"。一个general的predictive principle就够了。这直接挑战了Spelke的core knowledge hypothesis。

2. **Representation-space prediction > pixel-space prediction > text-space reasoning**,这个hierarchy很关键。你预测的东西决定了你学到的东西。

3. **当前MLLM在物理上其实很差**。Gemini 1.5 Pro、Qwen2-VL-72B 在这个task上基本是coin flip。这跟很多人以为"大模型够大就懂物理"的直觉相悖。

4. **LeCun 这条 JEPA 路线可能是对的方向**。不像 LLM 在token空间预测,JEPA在latent空间预测,这跟大脑的predictive coding理论(Rao & Ballard 1999, Friston的Free Energy Principle)高度一致。

## 我个人的take

作为Karpathy,你会appreciate这点的:

你一直讲LLM做next-token prediction学到syntactic structure。V-JEPA做next-frame *representation* prediction学到physical structure。**预测什么 = 学到什么**。这是同一个principle在不同modality上的体现。

但V-JEPA跟LLM有个本质区别:LLM预测的是discrete token(可预测性高,token有限),V-JEPA预测的是continuous representation(要决定哪些信息值得predict、哪些该discard)。这个"选择预测什么"的过程,可能就是学习abstraction的过程。

Open question:V-JEPA学到的"object permanence"是真的理解还是statistical pattern?就像LLM学到syntactic pattern但不一定真的理解semantic。这个debate会持续很久。

但有一点是清楚的:**只要你在representation space做prediction,你就比在pixel space或text space做prediction更可能学到世界的结构**。这是个deep insight,值得仔细琢磨。

参考链接:
- Paper code: https://github.com/facebookresearch/jepa-intuitive-physics
- V-JEPA原paper: https://arxiv.org/abs/2304.08471  
- LeCun的JEPA position paper: https://openreview.net/pdf?id=BZ5a1r-kVsf
- V-JEPA 2 (后续): https://ai.meta.com/blog/v-jepa-2-world-model-background-knowledge-video/

---

# Intuitive Physics Understanding Emerges from Self-Supervised Pretraining on Natural Videos — 深度技术讲解

## 1. Paper 的 Big Picture

这篇 paper 由 FAIR (Meta) 的 Quentin Garrido, Yann LeCun 等人撰写,核心 claim 是:**intuitive physics understanding 可以从 self-supervised pretraining on natural videos 中自发 emergent**,而且前提条件非常 minimal —— 只需要在 *learned representation space* 做预测,不需要 hardwired 的 core knowledge,不需要 task-specific training,甚至不需要太多 data 或 parameters。

这直接挑战了 developmental psychology 中 Spelke 的 **core knowledge hypothesis** —— 即人类婴儿天生具备针对 objects, space, number, geometry, agents 的 innate computational systems。Paper 的实验 evidence 表明:一个 general-purpose learning principle (predictive coding in latent space) 已经 sufficient。

paper code: https://github.com/facebookresearch/jepa-intuitive-physics

---

## 2. Background: JEPA 家族与 V-JEPA

### 2.1 JEPA 的动机

Yann LeCun 在 2022 提出的 **Joint Embedding Predictive Architecture (JEPA)** 是一种 non-generative, non-contrastive 的 self-supervised learning framework。参考: https://openreview.net/pdf?id=BZ5a1r-kVsf

核心 idea:学习一个 abstract representation space $\mathcal{H}$,在其中做 prediction,而不是在 pixel space 里 reconstruct。这与 Rao & Ballard 1999 的 **predictive coding** hypothesis 一致 (https://www.nature.com/articles/nn0199_79)。

JEPA 的三个 variants:
- **I-JEPA** (image): https://arxiv.org/abs/2301.08243
- **V-JEPA** (video): https://arxiv.org/abs/2304.08471
- **V-JEPA 2** (后续, scaled version with action-conditioned world model)

### 2.2 V-JEPA 架构细节

V-JEPA 包含三个核心 components:

1. **Context encoder** $f_\theta$: ViT (Vision Transformer),输入 corrupted video $V_C$ (经过 masking),输出 abstract representations
2. **Target encoder** $f_{\theta^{EMA}}$: 与 context encoder 同构,但 weights 通过 EMA 更新,处理 unmasked (或 complementary masked) video
3. **Predictor** $p_\phi$: 较小的 ViT (12 blocks, embed dim 384),输入 context encoder 的输出 + mask tokens,输出 predicted representations

**Architecture specifics (Table S1):**
- Input: 16 frames @ 5.33 fps = 3 秒 video clip
- Resolution: 224×224
- Patch size: 16×16 spatial × 2 temporal (tubelet)
- Positional encoding: **RoPE** (Rotary Position Embedding, https://arxiv.org/abs/2104.09864),3D split (H/W/T 各占 1/3 feature dim)
- Sizes: ViT-B/16 (~115M), ViT-L/16 (~300M), ViT-H/16 (~630M)
- Predictor: 12 blocks, embed dim 384 (比 encoder 小)
- Training: 90k iterations, batch 3072, ~26 years of (non-unique) video processed
- Optimizer: AdamW, lr warmup $2\times10^{-4} \to 6.25\times10^{-4}$ over 12k iters, cosine decay to $1\times10^{-6}$

---

## 3. 训练 Objective — 公式细节

### 3.1 主 loss (公式 S1)

$$\mathcal{L} = \left\| p_\phi\left( f_\theta(V_C) \right) - f_{\theta^{EMA}}(\overline{V_C}) \right\|_1$$

变量解释:
- $V$: 原始 video clip
- $V_C$: corrupted version,即对 $V$ 应用 masking 后的 video
- $\overline{V_C}$: complementary masked region,即被 mask 掉的部分 (作为 prediction target)
- $f_\theta$: context encoder,参数 $\theta$ (trainable via backprop)
- $f_{\theta^{EMA}}$: target encoder,参数 $\theta^{EMA}$ (stop-gradient,只通过 EMA 更新)
- $p_\phi$: predictor,参数 $\phi$ (trainable)
- $\|\cdot\|_1$: L1 loss (实际实现中常是 smooth L1)

**关键 design choice**: target encoder 用 EMA 更新,而不是 backprop。这是 non-contrastive SSL (BYOL, SimSiam, DINO, I-JEPA 一脉相承) 防止 representation collapse 的关键机制 —— 如果两边都 trainable 且用 MSE-like loss,模型会 trivially collapse 到 constant。

### 3.2 EMA 更新规则

$$\theta^{EMA}_{t+1} = (1-\alpha)\theta_t + \alpha \theta^{EMA}_t$$

- $t$: training iteration index
- $\alpha \in [0,1]$: EMA decay parameter (momentum)
- Paper 用 start momentum 0.998 → final 1.0 (cosine schedule)

当 $\alpha \to 1$,target encoder 几乎 frozen;当 $\alpha \to 0$,target encoder 完全跟随 context encoder。典型 SSL 实践是 $\alpha$ 从 0.99 升到 1.0。

### 3.3 为什么不直接用 contrastive loss?

V-JEPA 不需要 negative samples。这个 design 来自 I-JEPA 的 insight:在 high-dimensional representation space,predictor 如果要 minimize L1 distance 到 EMA target,而 target 是 stop-gradient 的,模型唯一 escape collapse 的途径是学习 *predictive* 的 features (i.e., features that encode scene structure so predictor can infer masked regions)。这是 LeCun 一贯的立场 (https://openreview.net/pdf?id=BZ5a1r-kVsf)。

---

## 4. 评估方法: Violation of Expectation (VoE)

### 4.1 范式起源

VoE 来自 developmental psychology (Baillargeon 1985, Spelke 1985)。给婴儿看两个 video:一个 plausible (符合物理),一个 implausible (违反物理)。如果婴儿对 implausible video 注视更久 (longer gaze time),说明 ta "感到意外",即理解了该物理 concept。

paper 把这个范式搬到 AI:用 model 的 *prediction error* 作为 "surprise" 的代理。

### 4.2 Surprise metric (公式 S2)

$$S_t = \left\| p_\phi\left( f_\theta(V_{t:t+C}) \right) - g_\psi(V_{t:t+C+M}) \right\|_1$$

- $V$: video with frames $\{1, \ldots, T\}$
- $t$: starting time index for prediction window
- $C$: context length (过去的 frames 数量)
- $M$: prediction horizon (未来要预测的 frames 数量)
- $V_{t:t+C}$: context frames,从 $t$ 到 $t+C$
- $V_{t:t+C+M}$: 包含 M 个 future frames 的完整 video segment (target)
- $f_\theta$: context encoder
- $g_\psi$: target encoder (在 V-JEPA 中 $g_\psi = f_{\theta^{EMA}}$)
- $p_\phi$: predictor

**Intuition**: 给 model 看 $C$ 帧,让它预测接下来 $M$ 帧的 *representation*;然后把实际 $M$ 帧的 representation 拿出来比 L1 distance。如果 video 违反物理 (比如 ball 突然消失),model 的 prediction 会严重失准,$S_t$ 就大。

### 4.3 Global surprise aggregation (公式 S3)

两种聚合方式:

$$\text{AvgSurprise} = \frac{1}{T} \sum_{t \in \{1, 1+s, \ldots, T-(C+M)\}} S_t$$

$$\text{MaxSurprise} = \max_{t \in \{1, 1+s, \ldots, T-(C+M)\}} S_t$$

- $s$: stride,实践用 $s=2$ (每两 frame 起一个 prediction window)
- $\{1, 1+s, \ldots, T-(C+M)\}$: 所有合法的 starting frame indices

**关键发现 (Figure S1)**: 
- **Pairwise classification** (给两个 video,哪个 impossible?):**AvgSurprise** 更好 —— 因为它捕捉了整段 video 的累积 surprise
- **Single video classification** (一个 video,plausible or not?):**MaxSurprise** 更好 —— 因为它聚焦于最 "shocking" 的瞬间,忽略 scene 复杂度带来的 noise

这个 distinction 很有意思,类似于 human 在 pair comparison 时会整合信息,而在 single judgment 时会被最 salient 的事件驱动。

### 4.4 评估的 datasets

| Dataset | 真实感 | 多样性 | 大小 | 测试 properties |
|---------|--------|--------|------|----------------|
| IntPhys | Synthetic | High | ~360 pairs | Object permanence, Shape constancy, Continuity |
| GRASP | Synthetic | Low | ~4000 | 10 properties (Object permanence, Continuity, Support, Gravity, Inertia, Color, Solidity, Collision, ...) |
| InfLevel-lab | Photorealistic | Low | ~4000 | Object permanence, Gravity, Solidity (需要 contextualization event) |

IntPhys 是最 carefully controlled 的,甚至有 private test set (3600 videos/property)。GRASP 覆盖最广。InfLevel-lab 用真实 video,但需要更多 memory (要看 pre-text event)。

测试的 intuitive physics properties:
- **Object permanence**: 物体不凭空消失 (Baillargeon & DeVos 1991)
- **Continuity**: 物体运动轨迹连续,不瞬移 (Spelke 1992)
- **Shape/color constancy**: 物体形状颜色不变 (Wilcox 1999)
- **Gravity**: 物体下落 (Kim & Spelke 1992)
- **Support**: 物体在平台上稳定 (Baillargeon 1990)
- **Solidity**: 物体不重叠不穿透 (Spelke 1992)
- **Inertia**: 无外力时运动状态不变 (Spelke 1992)
- **Collision**: 碰撞后运动状态改变 (Baillargeon 1995)

---

## 5. 主结果 — V-JEPA 完胜

### 5.1 Figure 1.A 的 headline numbers

| Method | IntPhys | GRASP | InfLevel-lab |
|--------|---------|-------|--------------|
| Untrained (random init, n=20) | ~50% | ~55% | ~50% |
| **V-JEPA (ViT-H)** | **98%** [95,99] | **66%** [64,68] | **62%** [60,63] |
| VideoMAEv2 | ~52% | ~55% | ~52% |
| Qwen2-VL-7B | ~50% | ~55% | ~50% |
| Gemini 1.5 Pro | ~50% | ~58% | ~52% |
| Human (IntPhys) | ~85% | — | — |

**核心观察**:
1. V-JEPA 是唯一在所有 datasets 上显著超过 untrained baseline 的方法
2. Pixel-prediction (VideoMAEv2) 和 MLLMs (Qwen, Gemini) 基本接近 chance
3. 在 IntPhys 上 V-JEPA 达到 98%,甚至**超过人类** (Riochet et al. 报告 human ~85%)

### 5.2 为什么 pixel prediction 失败?

VideoMAEv2 (https://arxiv.org/abs/2303.15302) 的 objective 是 reconstruct normalized pixels。问题在于:pixel space 包含大量 *不可预测* 的 details (texture noise, lighting micro-variations, compression artifacts)。模型为了 minimize pixel reconstruction loss,被迫花 capacity 去 memorize 这些 unpredictable low-level patterns,而不是学习 abstract scene structure。

这呼应了 LeCun 长期论证的 *pixel prediction is ill-posed for high-level understanding*。V-JEPA 通过在 representation space 预测,自动丢弃了 unpredictable details (encoder 学到 discard 它们,因为 predictor 无法预测它们)。

### 5.3 为什么 MLLMs 失败?

MLLMs (Qwen2-VL: https://arxiv.org/abs/2409.12191, Gemini 1.5 Pro: https://arxiv.org/abs/2403.05530) 用 text 输出。Paper 用 prompt:
> "Video 1: <video_1>, Video 2: <video_2>. ... Exactly one of the two videos has an event which breaks the laws of physics. ... which one is it?"

然后看 model 输出 "1" or "2" 的概率 (公式 S4):

$$P = \frac{P(\text{"1"})}{P(\text{"1"}) + P(\text{"2"})} \quad \text{or} \quad \frac{P(\text{"2"})}{P(\text{"1"}) + P(\text{"2"})}$$

Figure S2 显示 Qwen2-VL-72B 的 normalized probability 几乎都聚集在 0.5 附近 —— 模型基本在 coin flip。

**根本原因 (paper 的 hypothesis)**:
1. MLLMs 主要从 text 学到物理 "facts",但没有从 *sensory prediction* 中学到物理 "intuition"
2. Text-based reasoning 难以捕捉 fine-grained spatiotemporal dynamics (ball 在哪一 frame 消失的?)
3. 即便 Gemini 1.5 Pro 有 millions of tokens context,其 video processing pipeline (downsampling to 1 fps) 丢失了关键 motion 信息

### 5.4 Per-property 细分 (Figure 2)

V-JEPA 在以下 properties 上显著超过 untrained (Welch's t-test, p<0.05):

**IntPhys (ViT-L)**:
- Object permanence: M=85.7, SD=7.6 vs untrained M=51.4, SD=1.0, $t(4.0)=-8.9$, $p=4.19\times10^{-4}$, effect size $g=9.0$ [6.3, 11.7]
- Continuity: M=86.3 vs 51.2, $g=11.0$ [7.8, 14.2]
- Shape constancy: M=83.7 vs 51.7, $g=8.1$ [5.7, 10.6]

**GRASP**:
- Object permanence: 70.7 vs 54.1, $g=2.4$
- Continuity: 65.0 vs 55.0, $g=1.8$
- Support: 98.1 vs 58.4, $g=3.9$
- Gravity: 74.9 vs 55.3, $g=4.5$
- Inertia: 62.0 vs 54.3, $g=1.8$

**失败 (无显著提升)**:
- Color constancy
- Solidity (在 GRASP 和 InfLevel)
- Collision
- Gravity (在 InfLevel)

**失败的 intuition**: 
- Color constancy 失败可能因为 V-JEPA encoder 学到的 representation 偏 motion/shape,color 信息在 EMA target 中可能权重较低
- Solidity / Collision 失败可能因为 frame rate 5.33 fps 太低,碰撞瞬间 (sub-100ms) 被忽略
- InfLevel 的 gravity/solidity 需要看 *contextualization event* (前面的 pretext video),但 V-JEPA memory 只能处理 3-4 秒,看不到 pre-text

---

## 6. Ablation — Keys to Emergence

### 6.1 Masking strategy (Figure 3.A)

测试三种:
1. **Block Masking** (default): 8 个 spatial scale 0.15 的 blocks + 2 个 scale 0.7 的 blocks,aspect ratio ∈ [0.75, 1.5]
2. **Causal Block Masking**: 同上 + mask 掉最后 4 frames
3. **Random Masking**: uniform 随机 mask 90% patches

**Surprising result**: Random Masking 只掉 5 个 points on IntPhys,而它在 downstream action recognition 上掉 20 points。说明 **intuitive physics understanding 对 masking strategy 不敏感**,关键在于 "predict in representation space" 这个 general principle。

Causal Block Masking 反而比 non-causal 稍差 —— 尽管 inference 是 causal 的。这暗示 training-time causal bias 可能 overfit 到特定 prediction pattern,反而 hurt 泛化。

### 6.2 Training data (Figure 3.B, 3.C, S5)

三个 datasets 单独训:
- **SSv2** (~200k clips, 几秒): **几乎 chance** —— motion-focused 但太短太 curated
- **K710** (~650k, 10s, action classes): above chance
- **HowTo100M** (~1.2M, 6.5min avg, 15 years total): **最好**

Subsampling HowTo100M (固定 compute 30 years):
- 100% (15 years unique): high accuracy
- 1% (1289 hours unique): 仍 high
- 0.1% (128 hours unique, ~5天): **仍 >70% on all properties**

**Frame subsampling vs video subsampling (Figure S5)**: 
- Subsample videos → 减少 scene diversity
- Subsample frames → 减少 motion diversity per scene
- Video subsampling 表现更好,说明 *scene diversity* 比 *motion density* 更重要

### 6.3 Model size (Figure 3.C)

- ViT-H (~630M): best
- ViT-L (~300M): high
- ViT-B (~115M): **仍 >85% on IntPhys**

**Intuition**: representation space prediction 是 robust objective,即使小 model 也能学到 core physics structure。Scaling 主要帮助 harder tasks (single video classification, occluded scenarios)。

---

## 7. 人类对比 (Tables S4, S5)

在 IntPhys private test set (3600 videos/property):

**Pairwise (Table S4)**: V-JEPA-H 用 AvgSurprise 几乎零错误 (Object permanence: 0.28% vs human 12.5%)

**Single video AUROC (Table S5)**: V-JEPA-H 用 MaxSurprise: Object permanence 19.2% error vs human 24%; Shape constancy 21.9% vs 26%; Continuity 29.67% vs 38%

**Pattern**: 
- Visible scenarios: V-JEPA 远超 human
- Occluded scenarios: V-JEPA 接近或略低于 human (因为 model memory 有限,看不到 object 被 occluder 隐藏前的状态)

V-JEPA-H vs V-JEPA-L 差异在 single video 任务上显著 (H 匹配人类,L 落后),但在 pairwise 上差不多。Scale 帮助的是 harder generalization。

---

## 8. Limitations & Future Work

Paper 自己点出的:

1. **Object interactions 弱**: solidity, collision, color 接近 chance。Hypothesis: 这些需要更高频 temporal sampling 或 higher-order relational representations。

2. **Memory 限制**: 只能处理 3-4 秒 clip。InfLevel 的 contextualization event 看不到。需要 hierarchical memory。

3. **无 action conditioning**: V-JEPA 只能作为 *observer* 预测,不能想象 "如果我推这个球会怎样"。这是 V-JEPA 2 (后续 paper) 要解决的 —— 加入 action tokens。

4. **未测试 infant-like data**: SAYCam (https://doi.org/10.1162/opmi_a_00035), BabyView (https://arxiv.org/abs/2406.10447) 是 infant egocentric video datasets,值得测试 V-JEPA 是否能从 infant-perspective data 学到 physics。

5. **Generative video models 的对照**: Brooks et al. 2024 (Sora, https://openai.com/research/video-generation-models-as-world-simulators) claim video generation models 是 world simulators,但 Motamed et al. 2025 (https://arxiv.org/abs/2501.09038) 和 Bansal et al. 2024 (VideoPhysics, https://arxiv.org/abs/2406.03520) 显示 Sora-like models 物理理解很弱。这印证 V-JEPA 的 representation-space prediction > pixel-space generation。

---

## 9. Build Intuition — 为什么这个 work?

### 9.1 Predictive Coding 视角

从 neuroscience 角度 (Rao & Ballard 1999, Clark 2013, Friston 的 Free Energy Principle 一脉):大脑是 prediction machine, constantly generating predictions about sensory input,只对 *prediction error* 做处理。

V-JEPA 正是 computational implementation:
- Encoder = sensory cortex (extracts features)
- Predictor = generative model (predicts next-state representations)  
- Surprise = prediction error signal

婴儿的 VoE 实验中,婴儿对 impossible event 看更久 = 婴儿的 internal model 产生大 prediction error,引起 attention。V-JEPA 的 $S_t$ 大 = 同样的 mechanism。

### 9.2 为什么 representation space 关键

考虑两个预测目标:
- **Pixel prediction**: 要预测下一帧每个 pixel 的 RGB 值。但很多 pixels inherently unpredictable (e.g., leaf 在风中抖动的 exact pattern,水面反光的 micro-fluctuation)。Model 被迫分配 capacity 给这些 noise,无法集中学习 "ball 会继续滚动" 这种 abstract structure。
- **Representation prediction**: Encoder 学到一个 *predictable* 的 representation —— 即丢弃 unpredictable details,只保留 *causally relevant* 的 latent factors (object positions, velocities, identities)。Predictor 在这个 clean space 做预测,自然学到 physics-like dynamics。

这是 *information bottleneck* 的思路 —— 通过丢弃不可预测的信息,被迫学习 *causal structure* of the world。

### 9.3 为什么 EMA + stop-gradient 防 collapse

如果 encoder 和 target encoder 都 trainable 且用 L1 loss minimize 距离,trivial solution 是两者都输出 constant —— loss = 0 但 representation 无信息。

EMA + stop-gradient 切断了 target 端的 gradient flow。Target encoder 像 "slow-moving teacher",提供 stable targets。Context encoder + predictor 必须真正 *learn* 才能追上 slowly drifting teacher。这类似于 BYOL (https://arxiv.org/abs/2006.07733) 的机制。

### 9.4 Core Knowledge Hypothesis 的反驳

Spelke 的 core knowledge hypothesis 说婴儿有 innate systems 处理 objects, space, number。V-JEPA 没有 built-in object segmentation,没有 built-in 3D geometry,没有 physics engine —— 它从 raw pixels + masking + prediction 学到了这些。

**重要 caveat**: V-JEPA 学到的 "object permanence" 可能是 *statistical* 而非 *causal* 的。它知道 "video 中 ball 突然消失" 是 surprising,但不一定理解 ball 是一个 persisting entity。这类似于 LLM 学到 syntactic patterns 而非真正 semantic understanding 的 debate。

---

## 10. 与 V-JEPA 2 的联系 (后续工作)

LeCun 团队 2024 末发布了 V-JEPA 2 (https://ai.meta.com/blog/v-jepa-2-world-model-background-knowledge-video/) —— 在 V-JEPA 基础上加入 action conditioning 和 larger scale (1B+ parameters),训练在 1M+ hours of video。V-JEPA 2 展示了更强的物理 understanding,能用于 robot planning (搭配 action tokens 预测 action-conditioned futures)。

这条 research line 是 LeCun 推崇的 path to AGI:**world model + planning via latent prediction**,而非 LLM 的 next-token prediction。这篇 intuitive physics paper 是 foundation evidence —— 证明 latent prediction 即使在小 scale 也已 emergent 出 physics understanding。

---

## 11. 个人 Take-aways (写给 Karpathy)

作为 micrograd / makemore 的 author,你会 appreciate 这点:V-JEPA 的 elegance 在于它 *简单*。Loss 就是 L1 between predictor output 和 EMA target。没有 contrastive loss 的 negative sampling 工程,没有 diffusion 的 iterative denoising,没有 LLM 的 tokenization + RLHF。整个方法可以几百行 PyTorch 实现。

但这个简单 method 学到了人类婴儿级别的 physics intuition,而且 pixel-prediction 和 MLLM 都失败。这强烈暗示 **representation-space prediction is the right inductive bias for world models**。

对比你常讲的 "LLM 做 next-token prediction 学到 syntactic structure":V-JEPA 做 next-frame *representation* prediction 学到 *physical* structure。两者都是 predictive,但预测 *what* 决定了学到 *what kind of structure*。

未解决的 open question:如何把 V-JEPA-style world model 和 LLM-style reasoning 结合?LeCun 的 JEPA + LLM architecture (https://openreview.net/pdf?id=BZ5a1r-kVsf) 提出用 LLM 作为 *system 2* planner,JEPA 作为 *system 1* world model。但具体 interface 还不清晰。这是当前 ML 最 exciting 的 open problem 之一。

---

## Web Links 汇总

- **V-JEPA paper**: https://arxiv.org/abs/2304.08471
- **V-JEPA intuitive physics (this paper)**: https://github.com/facebookresearch/jepa-intuitive-physics
- **I-JEPA**: https://arxiv.org/abs/2301.08243
- **LeCun JEPA position paper**: https://openreview.net/pdf?id=BZ5a1r-kVsf
- **V-JEPA 2 (Meta blog)**: https://ai.meta.com/blog/v-jepa-2-world-model-background-knowledge-video/
- **IntPhys benchmark**: https://arxiv.org/abs/1803.07616
- **GRASP benchmark**: https://arxiv.org/abs/2406.03520 (实际是 IJCAI 2024,见 https://doi.org/10.24963/ijcai.2024/696)
- **InfLevel benchmark**: https://arxiv.org/abs/2104.05761
- **VideoMAEv2**: https://arxiv.org/abs/2303.15302
- **Qwen2-VL**: https://arxiv.org/abs/2409.12191
- **Gemini 1.5**: https://arxiv.org/abs/2403.05530
- **Rao & Ballard predictive coding**: https://www.nature.com/articles/nn0199_79
- **Spelke core knowledge**: https://doi.org/10.1037/0003-066X.55.11.1233
- **SAYCam (infant data)**: https://doi.org/10.1162/opmi_a_00035
- **BabyView**: https://arxiv.org/abs/2406.10447
- **VideoPhysics evaluation**: https://arxiv.org/abs/2406.03520
- **Sora as world simulator**: https://openai.com/research/video-generation-models-as-world-simulators
- **RoPE**: https://arxiv.org/abs/2104.09864
- **AdamW**: https://arxiv.org/abs/1711.05101
- **ViT**: https://arxiv.org/abs/2010.11929
- **BYOL**: https://arxiv.org/abs/2006.07733
- **Piloto et al. Nature Human Behaviour (developmental psych inspired DL)**: https://doi.org/10.1038/s41562-022-01394-8

---

如果要进一步 deep dive,值得追踪的方向:
1. **V-JEPA 2 + action conditioning**: 如何把 latent world model 用作 model-based RL 的 dynamics model
2. **Hierarchical JEPA**: 多时间尺度预测 (短时 dynamics + 长时 planning)
3. **JEPA + language grounding**: 把 latent representations 和 LLM 的 semantic priors 对齐
4. **Infant data training**: 用 SAYCam/BabyView 测试是否 emergence faster / different
5. **Embodied evaluation**: 把 V-JEPA 放进 robot,看是否能 zero-shot predict push/grasp outcomes
