---
source_pdf: Fast-WAM Do World Action Models Need.pdf
paper_sha256: 4eea24883dcc4d8a5c0f760870f501baabb07db4eb65e5fd6c3b4b500601be8d
processed_at: '2026-08-04T07:27:52-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Fast-WAM

## 一句话版本

这帮人发现：**让 robot policy 在 training 时学 predict future video 很有用，但 test 时真的去 generate future video 基本没用，纯属浪费 latency**。

---

## 问题出在哪

现在做 robot policy 有一波人很迷一个套路叫 **WAM (World Action Model)**，思路是：

> "robot 要 pick 一个 mug，得先在脑子里'想象'一下 mug 被抓起来之后会怎样，然后根据这个想象去 act。"

听起来特别 reasonable，跟人脑的 System 2 thinking 很像。实现上就是 inference 时先 run 一个 video diffusion model 生成未来几帧，再 condition on 这个 generated video 预测 action。

问题来了——video diffusion 要 iterative denoise，10 步起步，inference 慢得要死（800ms+），real-time control 基本告别。而且你仔细想想，**这个"想象未来"到底带来多少 marginal benefit？没人真正 ablation 过**。

之前的 WAM paper 把两件事混在一起了：
1. **Training 时学 predict future video**（representation learning）
2. **Inference 时 generate future video**（foresight / imagination）

这两件事一直被 entangle，你没法说清 gain 到底来自哪个。

---

## Fast-WAM 干了什么

思路特别简单粗暴：**training 时照旧学 video prediction，inference 时把 video generation 整个砍掉**。

具体来说：
- Backbone 用 Wan2.2-5B（阿里的 video DiT）
- 训练时同时 denoise action tokens 和 future video tokens（joint flow matching）
- Inference 时 **future video branch 直接删掉**，只把当前 frame 的 clean latent 过一遍 video DiT，拿到中间层的 representation $z(o,l)$，直接喂给 action expert 出 action

结果：**190ms latency，比 imagine-then-execute 快 4× 以上，性能基本不掉**。

---

## 公式层面的"人话翻译"

Paper 的核心公式就三个：

**标准 VLA**：
$$p(a_{1:H} \mid o, l)$$
- $o$ = current image observation
- $l$ = language instruction
- $a_{1:H}$ = 未来 $H$ 步 action（这里 $H=32$）
- 含义：直接从 "看见啥" + "要干啥" → "怎么做"

**Imagine-then-execute (prior WAMs)**：
$$p(a_{1:H} \mid o, l) = \int p(v_{1:T} \mid o, l)\, p(a_{1:H} \mid o, l, v_{1:T})\, dv_{1:T}$$
- $v_{1:T}$ = 未来 $T$ 帧 video（这里 $T=9$，4× downsample 后）
- 先 sample 未来 video $v_{1:T}$，再 condition on 它出 action
- 这是个 marginalization，理论上 Bayes-optimal，实际要 run video diffusion

**Fast-WAM**：
$$p_\theta(a_{1:H} \mid o, l) = p_\theta(a_{1:H} \mid z(o, l))$$
- $z(o,l)$ = video DiT 在当前 context 上单次 forward 的 latent representation
- 不 sample 未来，直接从 encoder 的中间激活出 action
- 本质上把 marginalization 用 amortized encoder 替代了

**Training loss**：
$$\mathcal{L} = \mathcal{L}_{\mathrm{act}} + \lambda\, \mathcal{L}_{\mathrm{vid}}$$
- $\mathcal{L}_{\mathrm{act}}$ = action flow matching loss
- $\mathcal{L}_{\mathrm{vid}}$ = future video flow matching loss（auxiliary）
- $\lambda$ = balance weight

Training 时两个 loss 一起优化，inference 时只用 action path。

---

## 关键实验：ablation 说话

他们做了三个 controlled variant 来 isolate 两个 factor：

| Variant | Training video loss | Inference video gen | 测试什么 |
|---|---|---|---|
| **Fast-WAM** | ✓ | ✗ | 主方法 |
| Fast-WAM-Joint | ✓ | ✓ (joint denoise) | Paradigm (A): Joint modeling |
| Fast-WAM-IDM | ✓ | ✓ (video then action) | Paradigm (B): Causal |
| Fast-WAM w/o co-train | ✗ | ✗ | Control: 只看 video training 的作用 |

### RoboTwin 结果（50+ bimanual tasks）

| Method | Avg Success | Δ vs Fast-WAM |
|---|---|---|
| Fast-WAM | 91.8 | 0 |
| Fast-WAM-Joint | 90.6 | -1.2 |
| Fast-WAM-IDM | 91.3 | -0.5 |
| **w/o co-train** | **83.8** | **-8.0** |

读法：**test-time imagination 的贡献 < 1.5%，video co-training 的贡献 ~8%**。差了 5 倍以上。

### Real-World Towel Folding

这 task 是 deformable object manipulation，特别需要 dynamics reasoning：

- Fast-WAM w/o co-train: **10% success**（崩溃）
- Fast-WAM: 正常工作
- Fast-WAM-IDM: 810ms latency
- Fast-WAM: 190ms latency

towel 这种 deformable 物体，**video co-training 几乎是命根子**。没它直接从 80%+ 掉到 10%。但 test-time imagination 还是不重要。

---

## 为什么这个结论反直觉

你想啊，"想象未来再行动" 听起来天经地义。为什么实际上没用？

我自己的几个 hypothesis：

### 1. Future video fidelity 不够高
10 步 denoise 出来的 future 跟 GT future 差距大。Conditioning 在这个 noisy future 上的 action，可能还没 conditioning 在 "current observation 经过 video encoder" 的 representation 上好。Video encoder 在 training 时见了几百万次 "current → future" 的 mapping，已经把 future-relevant features 压进 $z(o,l)$ 了。

### 2. Mutual information 早就被吸收了
Current observation + language 已经携带了 90% 的 action-relevant 信息。剩下的 10% 是真正的 "future imagination"，但这部分 information 实际上 encode 在 $z(o,l)$ 的 representation 里了，因为 video DiT 的 self-attention 在 single forward 里已经做了一次 "implicit imagination"。

### 3. Pixel-level generation 是 redundant bottleneck
你真正需要的是 "未来会发生什么" 的 latent understanding，不是 pixel-accurate 的 future frame。生成像素级 future frame 是 overkill，还慢。这跟 **JEPA** 的哲学高度一致——LeCun 一直在讲 predict in latent space, not pixel space。

### 4. 跟 self-supervised learning 的 "input drives representation" 一脉相承
你之前在 talk 里反复讲的：predicting input as auxiliary task 是 representation learning 的强 signal。Fast-WAM 的 video co-training 就是这个套路——video prediction loss 本质上是个 **representation-shaping regularizer**，不是 inference algorithm。

---

## Attention Mask 是隐藏的精彩设计

Paper 里有个细节容易被忽略：三组 token 的 attention mask 设计。

Training 时有三组 tokens：
- **Clean first-frame latent**：visual anchor
- **Noisy future video tokens**：video prediction target
- **Action tokens**：action prediction target

Attention 规则：
- Clean first-frame: **只被 attend，不主动 attend**（read-only anchor）
- Future video tokens: 可以 bidirectional 互相 attend，可以看 first-frame
- Action tokens: 可以 bidirectional 互相 attend，可以看 first-frame
- **Action tokens 不能看 future video tokens**！

最后这条是关键。如果让 action tokens 直接 attend 到 future video tokens，training 时会出现 **future leakage shortcut**：action policy 不学 "从 current 推 latent world"，而是学 "从已经 denoise 到一半的 future video 反推 action"。这就是作弊。

Inference 时 future branch 整个移除，action 只能从 first-frame 推 $z(o,l)$。训练时 mask 保证了 information flow 跟 inference 一致，没有 distribution shift。

这个 mask 设计本质上是个 **information flow control** 的 design pattern，跟 Flamingo / Gemini 的 cross-attention mask 是一类思路。值得收藏。

---

## 跟 prior work 的谱系

把 Fast-WAM 放到 context 里看：

```
VLA (OpenVLA, RT-2)
  └─ 直接 image+text → action, 无 world modeling
       │
       ▼
WAM (imagine-then-execute)
  ├─ Du et al. 2023: text → video → action
  ├─ Vidar: video diffusion + inverse dynamics
  ├─ CausalWM / LingBot-VA: causal video then action
  └─ Motus: latent action world model
       │
       ▼
WAM (joint modeling)
  ├─ Unified World Models
  └─ WAMs-are-zero-shot-policies (NVIDIA)
       │
       ▼
Fast-WAM (this paper)
  └─ video co-training as auxiliary, no test-time gen
       │
       ▼ (下一步可能)
JEPA-style Robot Policy
  └─ predict in latent, no pixel-level video
```

VPP [34] 和 UVA [35] 已经有过类似想法——用 video diffusion 的 representation，不 explicit decode video。但 Fast-WAM 是第一个把这个 idea 做 clean ablation 的，把 "training vs inference" 的 confounding factor 真正 isolate 出来。

---

## 这给我们的 intuition

### Intuition 1: Generative training, discriminative inference
这是个大趋势。Video prediction 的真正价值不在 generation 本身，而在 **representation shaping**。训完之后，generator backbone 可以当 feature extractor 用。这跟 LLM 的 in-context learning 很像——training objective 是 next token prediction，但 emergent 能力是 reasoning。

### Intuition 2: Foundation model 时代的 world model = internet video prior
Wan2.2-5B 见过海量的 internet video，已经学到 "物体怎么动、手怎么抓、东西怎么掉" 这些物理常识。这部分 prior 顶替了 embodied pretraining 的作用——Fast-WAM 没 embodied PT 也能打 π0.5（有 embodied PT）。

这暗示：**internet video 是 robot policy 的免费世界先验**，关键是设计好的接口把它接进来。

### Intuition 3: Multi-task auxiliary loss 重新成为主流
过去几年 SSL 在 vision 里被 contrastive learning 主导。Fast-WAM 这种 "generative auxiliary loss" 的成功暗示：对 robot policy 这种 downstream task，**reconstructive / predictive auxiliary loss 可能比 contrastive 更合适**，因为它直接塑造 world-relevant representation。

### Intuition 4: Real-time 是 embodied AI 的硬约束
190ms vs 810ms 在 robot control 里是天壤之别。闭环 control 的 latency 上限大概 200-300ms，超过这个就 "看一步走一步" 都做不到。Fast-WAM 让 WAM-style 方法第一次真正能 real-time deploy。

---

## 我觉得 paper 没回答的问题

1. **Outer auto-regressive rollout 被 omit**：paper 只做 single action chunk (H=32)，没做 "predict → act → re-observe → predict" 的 long-horizon loop。Long-horizon planning 里 explicit imagination 可能真的有用。

2. **Scale 换了结论会不会变**：如果 model 10× 大、video fidelity 飞跃，imagine-then-execute 可能重新占优。现在 fidelity 不够高，所以 imagination 的 marginal benefit 被 noise 淹没了。

3. **Latent world model vs pixel world model**：Motus 走 latent action world model 路线，Fast-WAM 走 pixel/VAE-latent video prediction。两者没直接对比，谁优谁劣不清楚。

4. **Failure case 的分析**：Open Microwave 这个 task，w/o co-train 反而比 Fast-WAM 高（82 vs 62）。这种 affordance-driven task 上 video co-training 可能 over-regularize。Paper 没分析这种反例。

5. **$z(o,l)$ 的 representation 到底学了啥**：没做 probing / visualization，不知道 latent 里到底 encode 了什么。是 dynamics? affordance? geometry? 还是 language-grounded features?

---

## 一句话给你的 takeaway

如果你只记一件事：**在当前 model scale 和 data scale 下，WAM 的 gain 90% 来自 training-time 的 video prediction objective（representation shaping），不到 10% 来自 inference-time 的 explicit future generation**。所以：

- Training 时：加 video prediction auxiliary loss，cheap 且有用
- Inference 时：别浪费 latency 去 generate video，直接用 encoder 的 latent

这是个 **训练目标 vs 推理算法** 的解耦案例。Generative model 当 training objective 和当 inference algorithm 是两件事，前者廉价且高 ROI，后者昂贵且当前 ROI 低。

🔗 想深挖可以看：
- Fast-WAM project: https://yuantianyuan01.github.io/FastWAM/
- Wan2.2: https://arxiv.org/abs/2503.20314
- Motus (latent WAM 对比): https://arxiv.org/abs/2512.13030
- VPP (类似思路早期工作): https://arxiv.org/abs/2412.14803
- UVA (joint video-action, skip decode): https://arxiv.org/abs/2503.00200
- WAMs zero-shot policies (NVIDIA): https://arxiv.org/abs/2602.15922
- CausalWM / LingBot-VA: https://arxiv.org/abs/2601.21998
- DreamerV3 (RL 里的 world model): https://arxiv.org/abs/2301.04104
- JEPA (LeCun latent prediction): https://ai.facebook.com/blog/yann-lecun-ai-i2r/
- Pi0 (flow matching VLA baseline): https://arxiv.org/abs/2410.24164

---

# Fast-WAM 深度解读：WAMs 真的需要 Test-time Future Imagination 吗？

## 1. Core Question 的哲学底色

这篇 paper 提了一个非常 sharp 的问题：**WAMs 的 gain 来自哪里？是 training-time 的 video modeling objective，还是 inference-time 的 explicit future generation？**

这两件事在 prior work 里被 entangle 在一起了——同一个 model 既从 video prediction 学，又在 test-time 显式 sample future video。作者用 controlled ablation 把这两个 factor 解耦，得到一个反直觉结论：**video co-training 的 representation shaping 价值，远大于 test-time 显式 imagination 的 foresight 价值**。

这个 insight 跟你之前在 self-supervised learning 里讨论的 *competence vs comprehension*、*input drives representation, output drives behavior* 这条思路高度共振。Video prediction 在这里扮演的角色更像一个 **auxiliary representation-learning loss**，而非 test-time inference algorithm。

🔗 相关链接：
- Project page: https://yuantianyuan01.github.io/FastWAM/
- Wan2.2 base model: https://arxiv.org/abs/2503.20314
- Motus (对比对象): https://arxiv.org/abs/2512.13030
- LingBot-VA / Causal World Modeling: https://arxiv.org/abs/2601.21998
- WAMs are zero-shot policies: https://arxiv.org/abs/2602.15922

---

## 2. 三种 WAM Paradigm 的结构对比

Figure 1 给了三种 paradigm：

### (A) Joint-modeling WAMs
- Future video tokens 和 action tokens 在同一个 denoising process 里 joint denoise
- 代表：[4, 6, 5]
- 缺点：action 生成被 video denoising 的 step 数 anchor，慢

### (B) Causal WAMs (imagine-then-execute)
- 两阶段：先 sample future video，再 condition on future representation 做 action prediction
- 代表：[3, 7, 8] (Vidar, CausalWM, Du et al.)
- 缺点：test-time 需要 iterative video denoising，latency 高

### (C) Fast-WAM
- Training 时 video co-training 仍在
- Inference 时 **移除 future video branch**，只保留 clean first-frame latent，过一遍 video backbone 得到 latent world representation $z(o, l)$
- Action 直接从 $z(o, l)$ 单次 forward pass 生成

关键 insight：**video DiT 在 inference 时被 "repurpose" 成 single-pass world encoder**，而非 multi-step generator。这本质上是把 generative backbone 当 feature extractor 用——非常类似 VPP [34] 和 UVA [35] 的思路，但 Fast-WAM 更极端，连 video decoder 都跳过了。

---

## 3. Problem Formulation 的数学拆解

### Eq. (1)：标准 VLA policy
$$p(a_{1:H} \mid o, l)$$

- $o$：current observation（多视角图像被 concat 成一张图后过 VAE）
- $l$：language instruction（T5 encoding）
- $a_{1:H}$：action chunk，horizon $H = 32$
- $p$：直接从 perceptual context 到 action sequence 的条件分布

### Eq. (2)：imagine-then-execute 的 marginalization
$$p(a_{1:H} \mid o, l) = \int p(v_{1:T} \mid o, l)\, p(a_{1:H} \mid o, l, v_{1:T})\, dv_{1:T}$$

- $v_{1:T}$：future visual observations，prediction horizon $T$（论文用 9 帧，4× temporal downsample）
- 第一个 factor 是 video generator
- 第二个 factor 是 inverse dynamics / action predictor
- 这是个 **marginalization over latent future**，理论上是 Bayes-optimal，但 sample 的时候要 run video denoising

### Eq. (3)–(4)：Fast-WAM 的 direct policy
$$p_\theta(a_{1:H} \mid o, l) = p_\theta(a_{1:H} \mid z(o, l))$$

- $z(o, l)$：video backbone 在当前 context 上单次 forward 得到的 latent world representation
- 这里 $z$ 不再是 sampled future，而是 **encoder 的中间激活**
- 严格说这相当于把 Eq. (2) 的 marginalization 用一个 amortized encoder 替换掉了——所有 "future imagination" 的容量被压缩进 $z$ 的 representation 里

这个 formulation 让我想到 **JEPA (Joint Embedding Predictive Architecture)** 的哲学：predict in latent space, not in pixel space。Fast-WAM 没走那么远，它在 training 时仍做 pixel-level video prediction，但在 inference 时退回到 latent-only 的 forward path。

---

## 4. Flow Matching Objective 的细节

### Eq. (5)–(6)：标准 flow matching
$$y_t = (1-t)y + t\,\epsilon$$
$$\mathcal{L}_{\mathrm{FM}}(y) = \mathbb{E}_{y, \epsilon, t}\left[\| f_\theta(y_t, t, o, l) - (\epsilon - y) \|_2^2\right]$$

变量含义：
- $y$：clean target，可以是 action chunk $a_{1:H}$ 或 video latents $z_{1:T}$
- $\epsilon \sim \mathcal{N}(0, I)$：Gaussian noise
- $t \in (0, 1)$：time step，从 clean ($t=0$) 到 noise ($t=1$)
- $y_t$：linear interpolation（flow matching 用的是 rectified flow 形式）
- $f_\theta$：网络预测的 velocity field
- 目标 $\epsilon - y$：从 $y$ 到 $\epsilon$ 的方向向量，即 constant velocity field

注意这里用的是 **rectified flow / linear interpolation form**，跟 DDPM 的 noise-prediction $\epsilon$-parameterization 数学上等价但 trajectory 是直线。Logit-normal 分布 over $t$ 是 Wan2.2 的 default schedule，意味着 $t$ 在中间区域被 oversample，两端（clean / pure noise）频率低——这对 video 这种 high-dim data 通常更稳。

### Eq. (7)–(9)：Joint loss
$$\mathcal{L} = \mathcal{L}_{\mathrm{act}} + \lambda\, \mathcal{L}_{\mathrm{vid}}$$

- $\mathcal{L}_{\mathrm{act}} = \mathcal{L}_{\mathrm{FM}}(a_{1:H})$
- $\mathcal{L}_{\mathrm{vid}} = \mathcal{L}_{\mathrm{FM}}(z_{1:T})$，其中 $z_{1:T}$ 是 VAE 编码后的 future video latents
- $\lambda$：balance coefficient，paper 没明说具体值，但从结果看应该接近 1

这个 objective 让我想到 **multi-task auxiliary loss** 的一个特殊形式：video prediction 不是 test-time 的 inference path，而是一个 **representation-shaping regularizer**。从这个角度看，Fast-WAM 跟你说的 "predicting the input as auxiliary task shapes latent representation" 是一回事。

---

## 5. Mixture-of-Transformer (MoT) 架构的 Attention Mask

这是 paper 最有意思的设计点。三组 tokens：

| Token Group | 角色 | Training | Inference |
|---|---|---|---|
| Clean first-frame latent tokens | visual anchor，shared context | ✓ | ✓ |
| Noisy future video tokens | video modeling objective | ✓ | ✗ (removed) |
| Action tokens | action generation | ✓ | ✓ |

### Structured Attention Mask（Figure 2b）：

1. **Clean first-frame tokens**：完全不 attend 其他 token（只被 attended，不主动 attend）
   - 这保证了 anchor 是 **read-only context**，避免被污染
   
2. **Future noisy video tokens**：bidirectional attention within video branch + 可访问 clean first-frame
   - 标准 video diffusion 的结构
   
3. **Action tokens**：bidirectional attention within action branch + 可访问 clean first-frame
   - **不能 attend 到 future video tokens**！
   - 这是关键：防止 future information leak 到 action branch，否则 action 会 "cheat" 看 future

这个 mask 设计其实是 **anti-leakage 设计**。如果允许 action tokens 直接 attend 到 noisy future video tokens，那么在 training 时，video branch 在做 forward denoising，但 action branch 可以"看到"已经被部分 denoise 的 future，这会让 action policy 产生一个 shortcut：它学会的不是 *从 current observation 推 latent world*，而是 *从 noisy future 反推 action*。这跟 inverse dynamics 训练里的 "future leakage" bug 是一类问题。

Inference 时 future video branch 整个被移除，action 只能从 clean first-frame 推 $z(o,l)$，这跟训练时的信息流一致。

---

## 6. Controlled Variants 的设计哲学

为了让 ablation 真正 informative，作者保持了 backbone / tokenization / training recipe 一致，只改 inference 结构和 training objective。

### Fast-WAM-Joint（Joint generation）
- 跟 Fast-WAM 共享 training，但 inference 时 allow attention between video and action tokens
- 对应 paradigm (A)
- 10 denoising steps 联合 denoise

### Fast-WAM-IDM（Video-then-action）
- 对应 paradigm (B)
- 跟 [3] 一致，对 GT video tokens 加 noise with $p = 0.5$，做 noise augmentation
- 这是为了 mimic test-time future video 是 noisy 的情况

### Fast-WAM w/o video co-train
- **最关键的 control**：架构和 inference 完全不变，只 remove $\mathcal{L}_{\mathrm{vid}}$ 这一项
- 直接检验 video co-training objective 本身的作用

这个 ablation 设计非常 clean：**Fast-WAM 与 -Joint/-IDM 之间的差距 = test-time future imagination 的贡献；Fast-WAM 与 w/o co-train 的差距 = training-time video objective 的贡献**。

---

## 7. 实验数据深度解读

### Table 1：RoboTwin（50+ bimanual tasks）

| Method | Embodied PT | Clean | Rand | Avg | Δ vs Fast-WAM |
|---|---|---|---|---|---|
| π0 | ✓ | 65.92 | 58.40 | 62.2 | -29.6 |
| π0.5 | ✓ | 82.74 | 76.76 | 79.8 | -12.0 |
| Motus | ✓ | 88.66 | 87.02 | 87.8 | -4.0 |
| LingBot-VA | ✓ | 92.90 | 91.50 | 92.2 | +0.4 |
| LingBot-VA (no PT) | ✗ | 80.60 | – | 80.6 | -11.2 |
| **Fast-WAM** | ✗ | **91.88** | **91.78** | **91.8** | 0 |
| Fast-WAM-Joint | ✗ | 90.84 | 90.32 | 90.6 | -1.2 |
| Fast-WAM-IDM | ✗ | 91.16 | 91.34 | 91.3 | -0.5 |
| Fast-WAM w/o co-train | ✗ | 82.76 | 84.80 | **83.8** | **-8.0** |

关键观察：
- **Fast-WAM 与 -Joint / -IDM 差距 < 1.5%** → test-time imagination 贡献很小
- **w/o co-train 比 Fast-WAM 低 8%** → video co-training 贡献大 5× 以上
- 没有 embodied pretraining 也能 91.8%，跟有 PT 的 LingBot-VA 几乎打平

### Table 2：LIBERO（4 suites）

| Method | PT | Spatial | Object | Goal | Long | Avg | Δ |
|---|---|---|---|---|---|---|---|
| OpenVLA | ✓ | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | -21.1 |
| π0 | ✓ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 | -3.5 |
| π0.5 | ✓ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 | -0.7 |
| LingBot-VA | ✓ | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 | +0.9 |
| Motus | ✓ | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 | +0.1 |
| **Fast-WAM** | ✗ | 98.2 | 100.0 | 97.0 | 95.2 | **97.6** | 0 |
| Fast-WAM-Joint | ✗ | 99.6 | 99.4 | 98.2 | 96.8 | 98.5 | +0.9 |
| Fast-WAM-IDM | ✗ | 98.8 | 97.8 | 97.8 | 97.6 | 98.0 | +0.4 |
| w/o co-train | ✗ | 89.2 | 99.2 | 95.4 | 90.0 | 93.5 | **-4.1** |

注意 w/o co-train 在 **Spatial 和 Long** 上掉得最多（Spatial -9.0, Long -5.2）。这两个 subset 恰恰最需要 **spatial reasoning / long-horizon planning**，说明 video co-training 对 **物理动力学相关 representation** 的塑造最关键——这跟 paper 的核心 hypothesis 完全吻合。

Object subset 几乎没掉（100→99.2），因为 Object 任务是 "看见物体就能 pick"，对 world dynamics 依赖低。

### Real-World Towel Folding（Figure 4）

这是最能说明问题的实验：

- **Fast-WAM w/o co-train**：success rate 暴跌到 **10%**，completion time 最长
- **Fast-WAM**：190ms latency
- **Fast-WAM-IDM**：810ms latency（4.3× 慢）
- **Fast-WAM-Joint**：介于两者之间

Towel folding 是 deformable object manipulation，**需要 explicit dynamics modeling**。在这个 task 上 video co-training 的 gain 被放大到极致（10% vs 80%+），而 test-time imagination 仍然贡献有限。

---

## 8. 跟你过去直觉的连接

### 8.1 与 "input drives representation" 思想
你在很多 talk 里讲过：predicting input 是 representation learning 的强 signal。Fast-WAM 的 video co-training 本质上就是 "predict future input as auxiliary loss"，shaping 了 world encoder 的 representation。Test-time 不需要再生 future video，因为那部分 knowledge 已经被 baked 进 $z(o,l)$ 的 representation 里了。

### 8.2 与 System 1 / System 2 思考
Imagine-then-execute 类似 System 2：显式 rollout 一个 mental simulation，然后 act。
Fast-WAM 类似 System 1：直接从 perception 反射到 action，但 reflection 能力来自 training 时反复做 simulation。

Paper 的结论暗示：在当前 data scale 和 model scale 下，**System 2 的 explicit rollout 没带来多少 marginal benefit**——大部分 foresight 已经被 System 1 的 representation 内化了。

### 8.3 与 Diffusion Policy / Pi0 的关系
Pi0 也是 flow matching + action chunk，但没有 video co-training。Fast-WAM 可以看作 **Pi0 + video auxiliary loss**——video objective 是个 representation regularizer。这解释了为什么 Fast-WAM 即使没有 embodied pretraining 也能打 Pi0.5：video pretraining 提供的 world prior 顶替了 embodied pretraining 的作用。

### 8.4 与 Dreamer / World Models 的对照
Dreamer (Ha & Schmidhuber, Danijar Hafner) 在 RL 里走的是 **learn world model → rollout in latent → planning**。Fast-WAM 的发现其实在说：在 imitation learning + 大模型时代，rollout 那一步可以 skip，只要 world model 的 representation 被学好。这跟 Dreamer 在 RL 里的作用不一样——RL 里 rollout 是为了 sample 多样 trajectory 算 return，IL 里 rollout 只是为了 foresight，而 foresight 可以被 representation 内化。

🔗 相关：
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- DreamerV3: https://arxiv.org/abs/2301.04104
- Pi0: https://arxiv.org/abs/2410.24164
- VPP: https://arxiv.org/abs/2412.14803
- UVA: https://arxiv.org/abs/2503.00200

---

## 9. Per-Task RoboTwin 数据里值得挖的细节

Table 3 里有些 task 的 ablation 表现很 informative：

| Task | Fast-WAM Clean | w/o co-train Clean | Δ |
|---|---|---|---|
| Hanging Mug | 58 | 18 | **-40** |
| Open Microwave | 62 | 82 | **+20** |
| Press Stapler | 90 | 80 | -10 |
| Move Can Pot | 90 | 51 | -39 |
| Handover Block | 95 | 66 | -29 |
| Stack Bowls Three | 80 | 66 | -14 |

**Hanging Mug** 掉 40 分——这个任务需要精确的 3D 空间 reasoning 和 deformable / articulated 物体 dynamics。Video co-training 在这里贡献巨大。

奇怪的是 **Open Microwave** 反而 w/o co-train 更高（82 vs 62）。这可能因为 Microwave 任务更依赖 visual affordance，video co-training 的 representation 在这里反而是 over-regularization，让 policy 对 handle 的几何变化不够 sensitive。这是个值得深挖的反例。

---

## 10. 我对这篇 paper 的几个 critical thoughts

### 10.1 为什么 test-time imagination gain 这么小？

几个可能 hypothesis：
1. **Future video generation 的 fidelity 不够高**：10 step denoise 出来的 future 跟 GT future 差距大，conditioning 在 noisy future 上的 action 不会比 conditioning 在 latent representation 上好
2. **Action 跟 future video 的 mutual information 被压缩了**：current observation + language 已经携带了大部分 action-relevant info，future imagination 是冗余的
3. **Latent world representation $z(o,l)$ 已经编码了 future-relevant features**：video backbone 在 self-attention 里做了一次 "implicit imagination"

### 10.2 没被回答的问题

- **Outer auto-regressive rollout 被 omit**：paper 说 "we omit the outer auto-regressive loop for simplicity"，但想象一下 long-horizon task 里 rollout 的作用，可能跟 single chunk 不一样
- **Future video branch 在 inference 时真的完全没用吗？** 如果 task 需要 multi-step planning（比如先把物体推到一边再 pick），单 chunk 的 $z(o,l)$ 可能不够
- **更大 scale 下结论会不会变？** 作者在 conclusion 里也提到这是 future work。如果 model 规模 10× 大，video generation fidelity 飞跃，imagine-then-execute 可能重新占优

### 10.3 跟 latent diffusion world model 思路的潜在冲突

Motus [5] 走的是 **latent action world model** 路线，在 latent space 做 world modeling。Fast-WAM 在 pixel/VAE-latent space 做 video prediction。两者的 representation 性质不同——Motus 的 latent 更 compact，Fast-WAM 的 video latent 更 rich 但跟 action coupling 弱。Paper 没直接对比这两种 representation 的优劣。

---

## 11. Takeaway for Building Intuition

1. **Generative model training ≠ generative model inference**：training 时学一个 generative objective 的最大价值可能不是 generation 本身，而是 representation shaping
2. **Inference 时的 explicit imagination 是 expensive 的 luxury**，在当前 model scale 下没明显的 marginal benefit
3. **Video foundation model 是个免费的世界先验**：用 Wan2.2-5B 做 backbone，相当于继承了海量 internet video 的物理先验，这部分顶替了 embodied pretraining
4. **Attention mask 是 information flow control**：Fast-WAM 的 anti-leakage mask 是个值得学习的 design pattern——用 attention mask 严格控制 task 之间的 information flow，可以做 controlled multi-task training

最后一句话总结：**这篇 paper 的核心贡献是把 "video prediction is useful for robot control" 这个模糊的 statement 拆成了两个 falsifiable hypothesis，然后用 controlled ablation 证明其中一个 (training representation) 是 dominant factor**。这种 reductionist 方法论在当前 robotics + foundation model 领域特别稀缺，特别珍贵。

🔗 如果你想顺藤摸瓜：
- Unified World Models: https://arxiv.org/abs/2504.02792
- Vidar: https://arxiv.org/abs/2507.12898
- GR-2: https://arxiv.org/abs/2410.06158
- DreamGen: https://arxiv.org/abs/2505.12705
- Flare (implicit world model): https://arxiv.org/abs/2505.15659
- Cosmos Policy: https://arxiv.org/abs/2601.16163
