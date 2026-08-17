---
source_pdf: Intuitive physics understanding emerges from self-supervised pretraining
  on natural videos.pdf
paper_sha256: 8ec7320af6b22e8bf91dae7372a6c1c567f217f23debaf4459cbe8d39b9f620d
processed_at: '2026-08-05T10:30:11-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇paper

好，抛开术语，咱们就当聊天讲。

---

## 这帮人到底想干嘛

Yann LeCun 这帮人一直有个信念：**AI 要真正理解世界，得学会"预测未来"**。不是预测下一个 token，不是预测下一帧的每个 pixel，而是在脑子里——在 abstract representation 里——想象"接下来会发生什么"。

这篇 paper 就是拿 V-JEPA 这个模型去做了一个实验：**我就用自然视频训练它，让它学会在 representation space 里补全被 mask 掉的视频片段，然后我去测它有没有"物理直觉"**。

结果发现：**有，而且挺强**。一个 ball 滚到遮挡物后面消失了，模型会"惊讶"——因为它预测 ball 应该从另一边出来。一个物体突然变形状了，模型也会"惊讶"。

最有意思的是：**Gemini 1.5 Pro 和 Qwen2-VL 这种巨型 MLLM 在这个任务上几乎等于瞎猜**。Sora 那种 pixel-space 的 video generation 也不行。

---

## 什么是"物理直觉"

就是你觉得理所当然的那些东西：

- 一个球滚到墙后面，应该从另一边出来（object permanence）
- 物体不能穿墙（solidity）
- 东西不会凭空消失或出现
- 球的形状不会突然变（shape constancy）
- 没支撑的东西会掉（gravity）

人类婴儿几个月大就有这感觉了。你给婴儿看一个"球穿墙"的视频，婴儿会盯着看更久——因为"不对劲"，这叫 **violation of expectation**。婴儿惊讶了，说明他脑子里有个"预期"，预期被打破了。

---

## 实验怎么做的

超级简单。你有一个训练好的 V-JEPA 模型。你给它看一段视频的前几帧，让它预测后面的 representation。然后你把它预测的 representation 和实际视频的 representation 对比，算个距离：

$$S_t = \| p_\phi(f_\theta(V_{t:t+C})) - g_\psi(V_{t:t+C+M}) \|_1$$

- $V_{t:t+C}$：从第 $t$ 帧开始的 $C$ 帧，这是 model 看到的 "过去"
- $V_{t:t+C+M}$：同样起点但多 $M$ 帧，后 $M$ 帧是 "未来" 的 ground truth
- $f_\theta$：encoder，把过去的帧编码成 representation
- $p_\phi$：predictor，从过去的 representation 预测未来的 representation
- $g_\psi$：target encoder，把真实未来帧编码成 representation 作为对比目标
- $S_t$：surprise score，就是预测的 representation 和真实 representation 的 L1 距离

距离大 = 模型预测错了 = 模型"惊讶"了。

然后你给模型看一对视频：一个 normal 的，一个 physics-breaking 的。如果模型对 physics-breaking 的那个 surprise 更高，说明模型"懂"了这个物理概念。

---

## 核心结果

| Method | IntPhys | GRASP | InfLevel |
|--------|---------|-------|----------|
| **V-JEPA** | **98%** | **66%** | **62%** |
| VideoMAEv2（pixel prediction） | ~50% | ~50% | ~50% |
| Qwen2-VL-72B | ~50% | ~50% | ~50% |
| Gemini 1.5 Pro | ~50% | ~50% | ~50% |
| 随机初始化的 network | 50% | 50% | 50% |

看到没？**VideoMAEv2、Qwen2-VL、Gemini 1.5 Pro 跟随机初始化的 network 没区别**。都是瞎猜。

而 V-JEPA 在 IntPhys 上 98%——几乎完美。

更狠的是，V-JEPA 在某些 property 上**比人类还准**（Table S4）：
- Object permanence：V-JEPA 0.28% error vs 人类 12.5%
- Shape constancy：V-JEPA 0% error vs 人类 14.5%
- Continuity：V-JEPA 0.09% error vs 人类 30%

---

## 为什么 V-JEPA 行，别人不行

### Pixel prediction 为什么不行

VideoMAEv2 也是 mask 然后预测，但它预测的是 **pixel 值**。就是说它要预测"这个位置的 RGB 值是多少"。

问题在于：pixel prediction 把大量 capacity 浪费在 irrelevant 的细节上——纹理、光照、阴影、压缩 artifact。model 学到的是"这个纹理长啥样"，而不是"这个物体应该 persist"。

你可以想象一个人画画的场景：如果你要求他精确重现每个像素的颜色，他会花大量精力在"这个阴影应该多深"上，根本没空思考"这个球应该从墙后面出来"。

### MLLM 为什么不行

Qwen2-VL 和 Gemini 1.5 Pro 是通过 text 来 reason 的。你问它"哪个视频违反物理"，它输出一个 token "1" 或 "2"。

Paper 里测了它输出的 probability distribution（Figure S2），发现 $P("1")$ 和 $P("2")$ 基本都在 0.5 附近——**model 根本没有 confidence**，就是在抛硬币。

直觉上理解：物理直觉是 **spatial + temporal + continuous** 的。你需要在脑子里 simulate 一个 3D 场景的动态演化。Text 是 discrete sequential 的，天然不 suited for 这种 simulation。

而且 MLLM 的 video processing 很 lossy——Gemini 把视频 downsampling 到 1 fps，fine-grained motion 全丢了。你 1 fps 怎么看 collision？怎么看 object continuity？

### V-JEPA 为什么行

V-JEPA 的 key insight：**在 abstract representation space 做预测**。

Training objective：
$$\mathcal{L} = \| p_\phi(f_\theta(V_C)) - f_{\theta^{EMA}}(\hat{V_C}) \|_1$$

- $V_C$：被 mask 过的视频（corrupted version）
- $\hat{V_C}$：被 mask 掉的那部分（complementary）
- $f_\theta(V_C)$：encoder 对可见部分的 representation
- $f_{\theta^{EMA}}(\hat{V_C})$：target encoder 对被 mask 部分的 representation（target）
- $p_\phi(\cdot)$：predictor，从可见部分预测被 mask 部分

这个 objective 迫使 model 做一件事：**从看到的部分推断没看到的部分**。

这跟人类认知非常像。你看到球滚到墙后面，你看不见球了，但你知道球还在——因为你的 brain 在做 inference：基于过去的视觉输入，推断当前 occluded 的状态。

V-JEPA 的 masking 训练就是在练这个 skill。Random mask 掉 90% 的 patch，model 要从剩下的 10% 推断这 90% 是什么。这个推断过程要求 model 理解 world 的 structure——物体在哪、怎么动、会不会消失。

**Predict in latent space = 被迫学习 abstract structure**。因为 latent space 没有 pixel-level 的 detail，model 不能靠"记纹理"作弊，必须 capture "物体 persist"、"形状不变"这些 invariant properties 才能 minimize loss。

---

## Ablation 的关键发现

### Masking strategy 不重要

Paper 试了三种 mask：

| Mask 策略 | 怎么 mask | IntPhys |
|----------|----------|---------|
| Block Masking | mask 大块 spatiotemporal block | ~97% |
| Causal Block Masking | 同上但额外 mask 最后 25% 帧 | ~92% |
| Random Masking | 随机 mask 90% patch | ~92% |

Random masking 几乎和精心设计的 block masking 一样好！这在 action recognition task 上是不成立的——random masking 在 classification 上掉 20 points。

**说明什么？** 物理直觉的涌现不依赖于 specific 的 pretraining task design。**关键在"在 representation space 预测"这个 framework 本身**，masking strategy 是 secondary 的。

这给了一个很强的 intuition：JEPA 的 inductive bias——predict in latent space——本身就 enough。你不需要精心设计什么 curriculum、什么 masking strategy，只要让 model 在 abstract space 做预测，它就会学会 world 的 structure。

### Data 不需要海量

HowTo100M 有 15 年的视频。Paper 发现：

| Unique video hours | IntPhys Accuracy |
|--------------------|-----------------|
| 15 years (full) | ~98% |
| 1289 hours | ~high |
| **128 hours (≈1 week)** | **>70%** |

**一周的视频就够了**。128 hours of unique video，模型就能 detect 物理违反。

而且发现 scene diversity 比 motion diversity 重要（Figure S5）。Subsample 视频（减少 scene 多样性）比 subsample 帧（减少 motion 多样性）影响更小。直觉上：物理直觉是关于物体在场景中行为的，需要 diverse scenes 来 generalize，而不是同一个人做同一个动作做很久。

### Model size 不需要巨大

| Model | Params | IntPhys |
|-------|--------|---------|
| ViT-H | ~600M+ | ~98% |
| ViT-L | ~300M | ~95% |
| ViT-B | **115M** | **>85%** |

**115M 参数就行**。对比 Qwen2-VL 7B+、Gemini 几百 B——参数量差几个数量级，但 V-JEPA 行它们不行。

---

## EMA 机制的直觉

Target encoder 权重 $\theta^{EMA}$ 是 context encoder 权重 $\theta$ 的 exponential moving average：

$$\theta_{t+1}^{EMA} = (1-\alpha)\theta_t + \alpha \theta_t^{EMA}$$

- $\theta_t$：第 $t$ 步 context encoder 的权重（通过 backprop 更新）
- $\theta_t^{EMA}$：第 $t$ 步 target encoder 的权重
- $\alpha$：smoothing 参数，start 0.998 → final 1.0
- $(1-\alpha)$：context encoder 对 target encoder 的贡献

为什么需要这个？如果 target encoder 和 context encoder 完全同步（同一个 network），predictor 可以学到 trivial identity mapping——直接把 input copy 到 output，loss 就是 0，但什么都没学到。

EMA 让 target encoder 变化更慢、更稳定，相当于一个 "teacher"——teacher 的 representation 作为 ground truth，student（context encoder + predictor）去逼近。teacher 缓慢 evolve，student 被迫学习 meaningful 的 prediction 而非 shortcut。

这跟 BYOL、SimSiam、DINO 的思路一脉相承——**避免 representation collapse 的关键 trick**。

---

## Surprise Metric 的 Avg vs Max

这个 distinction 很有 intuition。

**Pairwise classification**（两个视频比，哪个 impossible）：用 AvgSurprise 更好。因为两个视频场景匹配，只有一处物理违反不同，平均 surprise 能 capture 这个 overall signal。

**Single video classification**（单个视频判断是否 impossible）：用 MaxSurprise 更好。因为单个视频没有对照，average 会被 scene complexity 的 noise 淹没。Max surprise 能 isolate 最 suspicious 的 moment——"就是这一帧，predictor 彻底预测错了"。

直觉上：如果一段视频里有个 ball 滚来滚去 10 秒，最后一秒 ball 突然穿墙了。你问"这段视频可不可能"，你应该关注"穿墙那一瞬间"的 surprise，而不是整段视频的平均 surprise。

---

## 对你 build intuition 的帮助

### 1. Latent prediction > Pixel prediction > Text prediction（for physics）

这是 paper 最核心的 takeaway。三个层面：

- **Pixel prediction**（VideoMAEv2, Sora）：model 学 low-level detail，学不到 high-level structure。能生成漂亮视频但不"懂"物理。
- **Text prediction**（LLM, MLLM）：model 学语言 pattern，学不到 spatial-temporal simulation。能 talk about physics 但不能 simulate physics。
- **Latent prediction**（V-JEPA）：model 被迫学习 abstract representation，这个 representation 自然 encode 物体 persist、shape consistency 等 invariant。能 detect physics violation。

### 2. Predictive coding = learning

V-JEPA 的 success 支持 predictive coding hypothesis：brain 的核心机制就是"预测感官输入，用 prediction error 驱动学习"。你不需要 innate physics module，只需要一个 general 的 prediction 机制 + 足够丰富的 sensory data。

### 3. Inductive bias > Scale

115M + 128h video > 7B+ MLLM。这说明在 wrong inductive bias 下，scale 是在 compensate 而非 solve。选对 architecture 和 objective，问题可能比你想的简单。

### 4. World model 的可行性

V-JEPA 是 LeCun world model vision 的 proof of concept。它证明了：一个 passive observer，通过 latent prediction，能学会物理直觉。下一步自然是加 action conditioning——让 model 预测"如果我做 action $a$，世界会变成什么 representation"。这就是 model-based RL 和 planning 的 foundation。

### 5. 为什么 Sora 不行

Sora 能生成视觉上惊艳的视频，但物理经常崩坏（物体穿墙、突然变形）。Paper 引用了 Motamed et al. 2025 和 Bansal et al. 2024 的 evidence。原因：Sora 是 pixel-space generation，optimization target 是"看起来像真实视频"，不是"物理正确"。V-JEPA 的 representation space prediction 天然过滤掉 irrelevant 的 visual detail，focus on predictable structure。

---

## 一句话总结

**在 abstract representation space 预测视频的未来，这个简单的 learning principle 就能让模型涌现出物理直觉——不需要 innate module、不需要 text reasoning、不需要海量 scale、不需要精心设计 masking。**

这基本上是 LeCun 路线的一次重要 validation。你如果要做 world model，核心就是 latent prediction，别的都是 secondary。

---

# V-JEPA: Intuitive Physics Understanding from Self-Supervised Video Pretraining

这篇 paper 来自 FAIR Meta，作者包括 Yann LeCun, Quentin Garrido 等，发表于 2025 年 2 月。核心 claim 非常 clean：**通过在 learned representation space 做 video prediction（而非 pixel space 或 text space），V-JEPA 能 zero-shot 涌现出 intuitive physics understanding**，挑战了 core knowledge hypothesis（即物理直觉需要 innate hardwiring 的观点）。

---

## 1. 核心问题与 Motivation

### Moravec's Paradox in Physics Understanding

当前 AI 系统在 language, coding, math 上超越人类，但在 trivial 的 physical commonsense 上挣扎。比如一个 ball 滚到 occluder 后面应该 reappear，object 不应该穿过另一个 object。人类 infants（Piaget 1954, Baillargeon 1991）、primates、corvids、chicks 都有这种 intuition。

Core knowledge hypothesis (Spelke 2000) 认为这是 innate 的 computational systems。Paper 想挑战这个观点：**general learning principle 能否涌现出这种理解？**

### 三类模型的历史脉络

| Class | 代表 | 核心思想 | 问题 |
|-------|------|----------|------|
| Structured models | Battaglia 2013, Visual Interaction Networks | Hand-coded object representations in 3D Euclidean space, "mental game engine" | 需要预先定义 abstraction |
| Pixel-based generative | Lerer 2016, Finn 2016, VideoMAEv2 | Reconstruct future pixels | Representation 语义性差 |
| **JEPA (本文)** | V-JEPA | Predict in learned abstract representation space, no hand-coding | — |

JEPA 占据 middle ground：像 structured models 一样 predict abstract states（而非 pixels），但像 generative models 一样让 algorithm 自己 learn representation。这 congruent with predictive coding hypothesis (Rao & Ballard 1999, Clark 2013) in cognitive neuroscience。

参考链接：
- V-JEPA 原始 paper: https://arxiv.org/abs/2301.08243
- LeCun "A Path Towards Autonomous Machine Intelligence": https://openreview.net/pdf?id=BZ5a1r-kVsf
- Predictive coding (Rao & Ballard 1999): https://www.nature.com/articles/nn0199_79

---

## 2. V-JEPA 架构详解

### 2.1 三个核心组件

V-JEPA 由三部分组成（见 Figure 1.B）：

1. **Context encoder** $f_\theta$：处理 corrupted video $V_C$，输出 abstract representations
2. **Target encoder** $f_{\theta^{EMA}}$：处理完整 video，产生 prediction targets。权重是 context encoder 的 EMA
3. **Predictor** $p_\phi$：从 corrupted representations 预测 uncorrupted representations

### 2.2 EMA Update Rule

$$\theta_{t+1}^{EMA} = (1-\alpha)\theta_t + \alpha \theta_t^{EMA}$$

变量解释：
- $\theta_t$：context encoder 在 iteration $t$ 的权重（backprop 更新）
- $\theta_t^{EMA}$：target encoder 在 iteration $t$ 的权重
- $\alpha \in [0,1]$：EMA decay parameter，控制 target 的 "smoothing" 程度。Paper 中 start momentum 0.998 → final 1.0
- $\theta_{t+1}^{EMA}$：下一步 target encoder 权重

这个 EMA 机制是 self-supervised learning 中常见的 "teacher-student" 模式（BYOL, DINO, SimSiam 都用类似 trick），防止 representation collapse——如果 target encoder 和 context encoder 完全同步，predictor 可以 trivially 学到 identity mapping。

### 2.3 Training Objective

$$\mathcal{L} = \| p_\phi(f_\theta(V_C)) - f_{\theta^{EMA}}(\hat{V_C}) \|_1 \quad \text{(S1)}$$

变量解释：
- $V$：原始 video clip
- $V_C$：corrupted version（通过 masking 得到）
- $\hat{V_C}$：complementary masked region（即 $V$ 中被 mask 掉的部分）
- $f_\theta(V_C)$：context encoder 对可见部分的 representation
- $f_{\theta^{EMA}}(\hat{V_C})$：target encoder 对被 mask 部分的 representation（作为 ground truth target）
- $p_\phi(\cdot)$：predictor 输出
- $\|\cdot\|_1$：L1 距离

关键 insight：预测发生在 **representation space**，不是 pixel space。这就是和 VideoMAEv2 的本质区别——VideoMAEv2 在 normalized pixel space 做 reconstruction。

### 2.4 Architecture Specs

| 组件 | 架构 | 备注 |
|------|------|------|
| Context/Target encoder | ViT (B/L/H) | 16×16 patch size, tubelet size 2 |
| Predictor | 12-layer ViT, embed dim 384 | 比 encoder 小 |
| Input | 16 frames @ 5.33 fps = 3 sec | 224×224 resolution |
| Positional encoding | RoPE (3D split) | 替代 original V-JEPA 的 absolute pos embed |

Patch 化：每个 patch 是 $16 \times 16 \times 2$（spatial 16×16, temporal 2 frames），所以 16 frames → 8 temporal tokens per spatial location。

RoPE 3D 实现细节：feature dimension 分成 3 份，每份 encode 一个 spatiotemporal dimension (H, W, T)。这比 absolute positional embedding 更 generalizable 到不同 resolutions。

参考：RoPE paper https://arxiv.org/abs/2104.09864

---

## 3. Violation-of-Expectation (VoE) 评估框架

### 3.1 发展心理学根源

VoE paradigm 源自 infant 研究（Baillargeon 1985, Spelke 1985）。给 infant 看两个 scene：一个 physically possible，一个 impossible。测量 infant 的 gaze time——看 impossible scene 更久 = "surprised" = 理解了该物理概念。

### 3.2 Adaptation to AI

对 AI 模型：给定 video pair (possible, impossible)，measure model 的 "surprise" response。Surprise 越高说明 model 检测到了 violation。

### 3.3 Surprise Metric 公式

$$S_t = \| p_\phi(f_\theta(V_{t:t+C})) - g_\psi(V_{t:t+C+M}) \|_1 \quad \text{(S2)}$$

变量解释：
- $V_{t:t+C}$：从 frame $t$ 开始的 $C$ 个 context frames（过去）
- $V_{t:t+C+M}$：从 frame $t$ 开始的 $C+M$ 帧，其中后 $M$ 帧是 future targets
- $f_\theta$：context encoder，处理 $C$ 个历史帧
- $g_\psi$：target encoder，产生 future frames 的 ground truth representation
- $p_\phi$：predictor，从 context 预测 future representations
- $S_t$：time $t$ 的 surprise = predicted representation 和 actual future representation 的 L1 距离

### 3.4 Aggregation: Avg vs Max Surprise

$$\text{AvgSurprise} = \frac{1}{T} \sum_{t \in \{1, 1+s, \dots, T-(C+M)\}} S_t \quad \text{or} \quad \text{MaxSurprise} = \max_{t \in \{1, 1+s, \dots\}} S_t \quad \text{(S3)}$$

- $s$：stride，paper 用 $s=2$（从 frame 1, 3, 5, ... 开始预测）
- $T$：video 总帧数

**关键发现**（Figure S1）：
- **Pairwise classification**（possible vs impossible pair）：用 **AvgSurprise** 更好。因为 pair 内场景匹配，平均 surprise 能 capture overall physics violation signal。
- **Single video classification**（判断单个 video 是否 impossible）：用 **MaxSurprise** 更好。因为 single video 没有对照，max surprise 能 isolate 最 suspicious 的 moment，消除 scene complexity 的 noise。

这个 distinction 很重要——类似于 anomaly detection 中 "average anomaly score" vs "peak anomaly score" 的 trade-off。

---

## 4. 实验设置与 Benchmark

### 4.1 三个 Dataset

| Dataset | 类型 | 多样性 | Size | Properties 数 |
|---------|------|--------|------|---------------|
| IntPhys | Synthetic (simulator) | 高（pixel-level aligned pairs） | ~360 pairs | 3 (Object permanence, Shape constancy, Continuity) |
| GRASP | Synthetic | 低 | ~4000 | 10 |
| InfLevel-lab | Photorealistic (realistic renders) | 低 | ~4000 | 3 |

### 4.2 八个 Intuitive Physics Properties

1. **Object permanence** (Baillargeon & DeVos 1991)：物体不会凭空消失，occluded 时仍存在
2. **Continuity** (Spelke 1992)：物体路径连续，不 teleport
3. **Shape constancy** (Wilcox 1999)：形状不变
4. **Color constancy** (Wilcox & Chapa 2004)：颜色不变
5. **Gravity** (Kim & Spelke 1992)：无支撑时下落
6. **Support** (Baillargeon 1990)：平台上稳定，无支撑则掉
7. **Solidity** (Spelke 1992)：物体不能重叠/穿过
8. **Inertia** (Spelke 1992)：无外力不改变运动
9. **Collision** (Baillargeon 1995)：被撞击会动

### 4.3 Baselines

| Model | 类别 | 参数量 | Prediction space |
|-------|------|--------|------------------|
| V-JEPA-H | Latent prediction | ViT-Huge | Representation |
| VideoMAEv2 | Pixel prediction | comparable | Normalized pixels |
| Qwen2-VL-72B | MLLM | 7B+ | Text (next token) |
| Gemini 1.5 Pro | MLLM | closed | Text |

MLLM 评估 protocol 不同：因为输出是 text，不能算 surprise metric。所以给 pair 让 model 选哪个 impossible：
> "Video 1: <video_1>, Video 2: <video_2>. ... Exactly one of the two videos has an event which breaks the laws of physics. Given how objects behave on Earth, which one is it?"

还计算 normalized probability（S4）：
$$P = \frac{P("1")}{P("1") + P("2")} \quad \text{or} \quad \frac{P("2")}{P("1") + P("2")}$$

参考：
- IntPhys: https://arxiv.org/abs/2003.02707
- GRASP: https://www.ijcai.org/proceedings/2024/0696
- VideoMAEv2: https://arxiv.org/abs/2303.12002
- Qwen2-VL: https://arxiv.org/abs/2409.12191

---

## 5. 核心实验结果

### 5.1 Main Result (Figure 1.A)

| Method | IntPhys | GRASP | InfLevel-lab |
|--------|---------|-------|--------------|
| **V-JEPA** | **98%** [95,99] | **66%** [64,68] | **62%** [60,63] |
| VideoMAEv2 | ~chance | ~chance | ~chance |
| Qwen2-VL-72B | ~chance | ~chance | ~chance |
| Gemini 1.5 Pro | ~chance | ~chance | ~chance |
| Untrained networks | 50% | 50% | 50% |

V-JEPA 是唯一在所有 dataset 上显著优于 untrained network 的方法。这非常 striking——Gemini 1.5 Pro 这种 frontier MLLM 在这个任务上接近 chance，尽管参数量远大于 V-JEPA。

### 5.2 Per-Property Analysis (Figure 2)

V-JEPA 在 IntPhys 上的统计（vs untrained networks, Welch's t-test）：

| Property | V-JEPA M±SD | Untrained M±SD | Effect size g | p-value |
|----------|-------------|-----------------|---------------|---------|
| Object Permanence | 85.7±7.6 | 51.4±1.0 | 9.0 [6.3,11.7] | 4.19e-4 |
| Continuity | 86.3±6.2 | 51.2±1.2 | 11.0 [7.8,14.2] | 1.61e-4 |
| Shape Constancy | 83.7±7.8 | 51.7±1.2 | 8.1 [5.7,10.6] | 5.96e-4 |

GRASP 上显著的：Object Permanence, Continuity, Support, Gravity, Inertia  
GRASP 上**不**显著的：Color Constancy, Solidity, Collision

InfLevel 上：Object Permanence 显著，Gravity/Solidity 不显著（因为需要 contextualization event memory）

**Pattern**：V-JEPA 擅长 intrinsic object properties（permanence, shape），struggles with object-object interactions（collision, solidity）和需要 long context 的事件。Paper hypothesis：framerate 限制 + 短 memory (3-4 sec) + 无 action conditioning。

### 5.3 Human Comparison (Figure 2.B, Table S4)

在 IntPhys private test set 上：

| Method | Object Permanence (All) | Shape (All) | Continuity (All) |
|--------|-------------------------|-------------|------------------|
| V-JEPA-H (Max) | 4.4% error | 4.4% | 12.87% |
| V-JEPA-H (Avg) | 0.28% | 0.0% | 0.09% |
| Human | 12.5% | 14.5% | 30.0% |

V-JEPA-H 在所有 property 上等于或超过人类！而且 human 和 V-JEPA 的 error pattern correlated——都在 occluded settings 上更差。这说明 V-JEPA 可能 capture 了类似的 computational principle。

---

## 6. Ablations: 什么导致 Intuitive Physics 涌现？

### 6.1 Masking Strategy (Figure 3.A)

| Masking | IntPhys Score |
|---------|---------------|
| Block Masking (default) | ~97% |
| Causal Block Masking (mask last 25%) | ~92% |
| Random Masking (90% random) | ~92% |

**关键发现**：Random Masking（最简单的）几乎和 Block Masking 一样好！这和 action recognition task 上的结果形成对比——在 classification task 上 Random Masking 掉 20 points。

**Insight**：Intuitive physics understanding 不需要精心设计的 pretraining objective。**关键是在 representation space 做 prediction**，masking strategy 是 secondary 的。

这给 build intuition：JEPA 的 "predict in latent space" 这个 inductive bias 本身就足够 powerful，让 model 必须学习 world 的 structure 才能 minimize prediction loss。

### 6.2 Training Data (Figure 3.B, S5)

三个 dataset 单独训练：
- **HowTo100M**（tutorials, 15 years）：最好
- **Kinetics710**（actions, ~3 months）：above chance
- **SSv2**（motions, ~3 months）：chance level

HowTo100M 优势部分来自规模。但 sub-sampling HowTo100M 实验（Figure 3.C）：
- 1289 hours（1.3% of full）：high accuracy
- **128 hours（0.1%，相当于一周视频）**：仍然 >70% pairwise accuracy！

Subsampling 方式也有影响（Figure S5）：
- Subsample videos（reduce scene diversity）：performance 维持
- Subsample frames within videos（reduce motion diversity）：performance 略低但 still good

**Insight**：Scene diversity 比 motion diversity 更重要。这符合直觉——intuitive physics 是关于 object 在 scene 中行为的，需要 diverse scenes 来 generalize。

### 6.3 Model Size (Figure 3.C)

| Model | Params | IntPhys Accuracy |
|-------|--------|------------------|
| ViT-H | ~600M+ | ~98% |
| ViT-L | ~300M | ~95% |
| ViT-B | 115M | >85% |

**115M 参数就能 above chance**。这说明 intuitive physics 不是 emergent property of massive scale，而是 representation prediction framework 的 intrinsic consequence。

对比：Qwen2-VL 7B+ 参数，Gemini 1.5 Pro 远大于此，但 chance level。Scale 不解决问题——**架构和学习目标才解决问题**。

---

## 7. 为什么 Pixel Prediction 和 MLLM 失败？

### 7.1 Pixel Prediction (VideoMAEv2)

VideoMAEv2 在 normalized pixel space reconstruct。问题在于：
- Pixel prediction 鼓励 model 学习 low-level details（纹理、光照、compression artifacts）
- 这些 details 对 downstream classification useful（after fine-tuning），但对 intuitive physics 是 noise
- Representation space 没有 "abstract away" 不相关细节

类比：predict 下一帧的每个 pixel 值，model 会花大量 capacity 学习 "this texture should look like this"，而忽略 "this object should persist"。

### 7.2 MLLMs (Qwen, Gemini)

MLLMs 通过 text reasoning。Figure S2 显示 Qwen2-VL 输出的 normalized probability 几乎都在 0.5 附近——model 实际上在 coin flip。

可能原因：
1. **Training objective mismatch**：MLLM 训练 predict next text token，不是 predict future video states
2. **Video processing lossy**：Qwen/Gemini 对 video 做 downsampling（1 fps for Gemini），丢失 fine-grained motion
3. **Text-based reasoning 不适合 physics**：Physics 是 continuous, spatial, temporal；text 是 discrete, sequential
4. **No "mental simulation"**：MLLM 不能 "imagine" future states 来检测 violation

这呼应 LeCun 长期 argument：autoregressive LLM 不是通向 AGI 的路径，需要 world model 在 continuous latent space 推理。

参考：
- LeCun on LLM limitations: https://www.youtube.com/watch?v=5t1v2qBBgBM
- VideoMAE failures on physics: https://arxiv.org/abs/2411.02385

---

## 8. 局限性与 Future Directions

### 8.1 V-JEPA 的 Limitations

1. **Object interactions 弱**：Collision, Solidity 表现差。Hypothesis：需要 higher-order relational representations 或 hierarchical JEPA。
2. **Short memory**：3-4 second clips。InfLevel 的 gravity/solidity 需要 contextualization event memory。
3. **No action conditioning**：V-JEPA 是 passive observer，不能条件化预测于 action。未来需要 action-conditional JEPA（类似 LeCun 的 JEPA-CA architecture）。
4. **No counterfactual reasoning**：只能 detect violation，不能 reason about alternatives。

### 8.2 Future Directions（paper 提到 + 我的联想）

1. **Hierarchical JEPA**：Multiple levels of abstraction，高层次 capture object interactions
2. **Action-conditional prediction**：让 model 预测 "如果我做 X，世界会怎样"——这是 model-based RL 的 foundation
3. **Infant-perspective training data**：BabyView dataset (Long et al. 2024)，训练在婴儿视角视频上
4. **Longer context**：处理 minutes-long videos，capture contextualization events
5. **World model for planning**：V-JEPA 作为 world model，配合 actor 做 planning（Hafner et al. Dreamer 系列的 latent space 思路）

参考：
- Dreamer V3: https://arxiv.org/abs/2301.04104
- BabyView: https://arxiv.org/abs/2406.10447
- JEPA-CA (LeCun): https://openreview.net/pdf?id=BZ5a1r-kVsf

---

## 9. 更广的 Context 与 Intuition Building

### 9.1 Predictive Coding 联系

V-JEPA 的 "predict masked regions in representation space" 直接对应 predictive coding theory (Rao & Ballard 1999, Clark 2013)。Brain 被认为不断 generate predictions about sensory input，prediction error 驱动 learning。

V-JEPA 的 surprise metric $S_t$ 就是 "prediction error signal"。Violation-of-expectation 在 infants 上的 gaze time 延长，可能就是 brain 的 prediction error 触发 attention。V-JEPA 在 computational 上 instantiate 了这个 framework。

### 9.2 World Models 路线

V-JEPA 是 LeCun "world model" vision 的 instance。在 LeCun 的 H-JEPA 架构中：
- Perceptor：提取 representations
- World model：predict future representations (可能 conditioned on action)
- Actor：propose actions
- Critic：evaluate

V-JEPA 是 perceptor + world model（无 action）。Intuitive physics emergence 是 world model capability 的 evidence。

### 9.3 与 Infant Development 对比

Paper 的哲学立场：intuitive physics **不需要** innate core knowledge。General learning principle（latent prediction）+ natural video data 就够了。

这挑战 Spelke 的 core knowledge hypothesis。但要注意：
- Infants 学习更快（hours vs V-JEPA 的 weeks of video）
- Infants 有 embodied interaction（V-JEPA 是 passive）
- Infants 有 multi-modal input（touch, proprioception）

V-JEPA 证明 "latent prediction" 是 sufficient mechanism，但不否认其他 mechanism（如 innate priors）也能实现。这是 existence proof，不是 exclusivity proof。

### 9.4 Scaling Insights

Paper 显示 115M params + 128h video 就能 above chance。这和 LLM scaling laws (Kaplan 2020) 形成有趣对比：
- LLM：performance 强依赖 scale
- V-JEPA on intuitive physics：architecture/objective > scale

这暗示 intuitive physics 是 "easier" in right inductive bias regime，而 LLM 在 wrong inductive bias（text prediction）下需要 scale 来 compensate。

### 9.5 Energy-Based Model 联系

V-JEPA 的 prediction loss 可以看作 implicit energy function。Low prediction error = low energy（expected world state）；high prediction error = high energy（surprising/violating state）。

这和 LeCun 的 Energy-Based Model (EBM) framework 一致。Surprise metric $S_t$ 本质是 energy，可以用于：
- Anomaly detection
- Out-of-distribution detection
- Active learning（sample high-surprise states）

参考：LeCun EBM tutorial https://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf

---

## 10. Critical Thoughts 与 Open Questions

1. **Benchmark bias**：IntPhys, GRASP 都是 synthetic。虽然 InfLevel-lab 是 photorealistic，但仍是 rendered。Real-world physics violations（hand-crafted trick videos）能否 replicate？
2. **Pairwise vs absolute**：V-JEPA 在 pairwise（98%）远好于 single video classification（Table S5, ~80% AUROC for H）。Pairwise 控制了 scene complexity，但 real-world 没有 pairs。
3. **Causal vs correlational understanding**：V-JEPA detect violations，但能 intervene 吗？能 answer "what would happen if..." 吗？这是 true understanding 的更高 bar。
4. **Compositionality**：V-JEPA 学习的 physics 是否 compositional？能否 generalize 到 unseen object combinations？
5. **3D understanding**：V-JEPA 是否学习 3D scene structure，还是 2D appearance patterns？Paper 没有 probe 这个。

---

## 总结

这篇 paper 的核心贡献是 empirical evidence：**latent space prediction 是涌现 intuitive physics 的 sufficient condition**。不需要 hand-coded physics engine，不需要 text reasoning，不需要 massive scale。一个 115M 参数的 ViT，在 128 hours natural video 上训练，用 simplest random masking，就能 detect object permanence, shape constancy 等 violations。

这为 LeCun 的 world model 路线提供 strong support，也解释了为什么 pixel-based generative models（Sora 等 video generation models）虽然能生成 visually plausible videos 但 physics understanding incomplete——generation ability ≠ understanding。

代码：https://github.com/facebookresearch/jepa-intuitive-physics

对你的 intuition 来说：如果你想 build world models，predict in latent space，让 representation 自己 emerge，用 prediction error 作为 surprise/energy signal。这比 predict pixels 或 predict text tokens 更 sample-efficient 且语义更 rich。
