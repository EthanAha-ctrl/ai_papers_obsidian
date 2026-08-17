---
source_pdf: LIBERO-Plus In-depth Robustness Analysis of Vision-LanguageAction Models.pdf
paper_sha256: 661de08e0ed1bb4bb4774177cac9fafa94f06b5385fa2f4107a1927b9a4aada7
processed_at: '2026-08-05T14:37:59-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 LIBERO-Plus

Hey Andrej，我把之前那坨技术细节浓缩成"跟同事喝咖啡时聊"的版本。

---

## 一句话总结

现在这些 VLA model 在 LIBERO benchmark 上跑 95%+ 的 success rate，看起来已经 solved 了。这篇 paper 说：**你信这个数字就上当了**。稍微动一下 camera 角度，稍微挪一下 robot 起始姿态，performance 直接掉到 30% 以下。更离谱的是，把 language instruction 整个删掉，model 照样干活——它压根没在听你说话。

---

## 这篇 paper 到底干了啥

他们做的事情很简单：拿 LIBERO 的 40 个 task，在 7 个维度上做 perturbation（camera、robot 初始状态、language、light、background、noise、object layout），每个维度又细分好几个 sub-dimension，总共搞出 10,030 个 test task。然后拿 10 个 SOTA VLA model 去跑，看谁崩得最惨。

结果：**全崩**。

---

## 最 striking 的几个发现

### 1. Camera 动一下就死

这是最 universal 的弱点。几乎每个 model 在 camera viewpoint 稍微变一下就掉 40-90 个点。比如 π0 从 94.2% 掉到 15.8%，UniVLA 从 95.2% 掉到 4.3%。

**Intuition**: model 其实在 memorize "从这个角度看，物体在这个位置，该这么抓"。它没有 view-invariant 的 object representation。换个角度，整个 visual feature 就对不上了，action 就乱了。

### 2. Robot 初始姿态动一下也死

跟 camera 一样致命。robot 起始 joint angle 随机 perturb 一下（幅度 0.1-0.5 rad），performance 直接腰斩。

**Intuition**: model 学的是 "看到这个画面→执行这个 trajectory"，而不是 "理解当前 arm 状态→plan 怎么到 target"。proprioceptive reasoning 这块基本没学到。

### 3. Language 完全被忽略——这是最 disturbing 的

paper 最精彩的部分。他们做了三个 probe：

**Probe A**: 把 language instruction 换成 LLM rewrite 的版本（更长、更绕、commonsense 替换）。结果 performance 几乎不掉。表面看很 robust。

**Probe B**: 把 language instruction 直接换成空 string。结果 OpenVLA-OFT 在 object suite 上**几乎不掉点**。只有 long-horizon task 才掉。

**Probe C（最 damning）**: 把 instruction 里的 target object 换掉，比如 "pick up the alphabet soup" 改成 "pick up the butter"。结果 model **照样去抓 alphabet soup**，success rate 掉到接近 0%。

**Conclusion**: VLA 实际上是 VA。Language 是个 decorative input，model 根本没在 attend 它。你改 instruction 它当然不掉点，因为它压根没看。

这个 finding 跟 LLM 那边的 modality imbalance 是 mirror image：VLM 里 language 主导、vision 被 ignore；VLA 里 vision 主导、language 被 ignore。

### 4. Light / Background 看似 robust 其实是 cheat

OpenVLA-OFT 在 light perturbation 下只掉 11 个点，看起来很 robust。但作者做了个 clever ablation：把 third-person camera 的输入直接换成黑帧，只保留 wrist camera。结果 model 照样跑 67%。

**Intuition**: wrist camera 是个 "作弊" 通道。它提供的是 close-range geometry + contact cue，本质上 illumination-invariant。light 怎么变，wrist 看到的局部几何不变。所以 light robustness 是 hardware 的功劳，model 没那么 smart。

对比实验：只用 third-person view 的 model（OpenVLA、Nora、WorldVLA）在 light perturbation 下掉 50-60 个点。这才是 model 的真实水平。

### 5. Object layout 揭示 positional bias

Object layout 分两种 perturbation：
- 加 distractor object → 几乎不影响（model 能 ignore distractor，这倒是好事）
- 挪 target object 位置 → 直接崩

**Intuition**: model 学的是 "(object appearance, object position) → action" 的联合 mapping。它能 ignore distractor，因为它有 object-level attention。但它把 position 也当 feature memorize 了，没学到 object semantic 是 position-invariant 的。挪一下就不认识。

### 6. 组合 perturbation 是 super-linear 地 bad

这个是 paper 里最 mathematical 的部分。他们测 pairwise perturbation，发现组合起来的 success rate 比 "两个 single perturbation success rate 的乘积" 还低。用 conditional covariance 定义 compositionality gap Δ_ij，全是 negative。

**Intuition**: perturbation 不是 independent 的。camera 变了 + robot 状态变了，难度不是简单相加，而是有 synergistic 效应。model 的 representation 是 entangled 的，一个 perturbation 打到一个 feature，另一个 perturbation 打到另一个 feature，两个一起打，model 就完全 lost。

---

## 他们用更 diverse 的 data 训练了一下，能 recover 多少？

他们用同样 pipeline 生成 20,000 条 successful trajectory，覆盖 6 种 perturbation type，然后从 OpenVLA-OFT_m 开始 mixed fine-tune。

结果：total 从 67.9% → 79.5%。camera 维度从 55.6% 飙到 92.8%（+37.2 个点）。这个 huge gain 说明 camera robustness 主要是个 data coverage 问题。

但 robot state 维度只从 21.7% 涨到 30.3%。这说明 proprioception 是个更深的 representation bottleneck，光靠 data diversity 解决不了，可能需要 architecture level 的改动。

---

## 给你的 takeaway

1. **Benchmark saturation = red flag**。当所有 model 都 >95% 时，说明 benchmark 已经 saturated，区分不了 model quality。需要 harder test。LIBERO-Plus 就是这种 harder test 的范例。

2. **VLA 经常是 VA**。当前 architecture 里 language 通道是 decorative。这跟 Software 2.0 的 data-centric 思想呼应——model 学的是 data 里最 discriminative 的 cue，如果 language 不是 task 完成的必要信息，model 就懒得用它。

3. **Data diversity > architecture sophistication**。Section 6.2 实验完美验证你 Software 2.0 的思想：不动 architecture，只改 data distribution，就能大幅 recover robustness。

4. **Wrist camera 是 manipulation 的 shortcut**。它把 high-level visual reasoning 降维成 close-range geometry。这暗示未来 manipulation system 可能需要更多 camera / 更多 modality，靠 sensor diversity 来补 reasoning 的不足。

5. **Positional bias 是 deep problem**。model 记 (object, position) 而非 object semantics。这跟 transformer 在 language 里的 positional bias 有 echo——position encoding 比语义更容易学，model 就倾向于走捷径。

---

## 一句话 final

这篇 paper 的 core message：**current VLA model 的 "competence" 大部分是 benchmark overfitting 的 illusion**。真正 robust 的 manipulation agent 需要的是 view-invariant representation、真正的 language grounding、proprioceptive reasoning，以及对 perturbation 组合的 graceful degradation。光在 LIBERO 上刷分没用，得在 LIBERO-Plus 这种 stress test 上才能看出 model 的真实水平。

你要 build VLA 的 intuition，这篇是必读，因为它把 SOTA 的 facade 给拆了，让你看到底下其实是 memorization。

---

# LIBERO-Plus 深度解读 — 给 Karpathy 的 technical walkthrough

Hey Andrej, 这篇 paper 我觉得你会很有共鸣。它做的事情，本质上是用 controlled perturbation 把当前 VLA models 的 "benchmark overfitting" 现象给 expose 出来——和你之前谈 LLM 时常说的 "high accuracy ≠ true understanding" 在 robotics 这边重新走了一遍。我把核心脉络、math、experimental findings 都摊开讲一下，重点 build intuition。

---

## 1. Paper 的核心 narrative

LIBERO benchmark 上，SOTA VLA models 普遍能跑到 95%+ 的 success rate。作者的 hypothesis 是：这些数字 hide 了 fundamental 的 robustness 问题。为了 verify，他们在 7 个 axis 上做 controlled perturbation，然后跑 10 个 representative models，发现：

- Camera viewpoint / robot initial state 上的轻微 shift 能让 performance 从 95% 掉到 <30%
- Language instruction 怎么改几乎不影响 performance，进一步 probe 发现 models 根本 ignore language
- 单维度 perturbation 已经很 bad，多维度组合 perturbation 是 super-linear 地 bad
- 用更 diverse 的 training data 做 mixed fine-tuning，可以大幅 recover robustness（特别是 camera 维度）

---

## 2. 七个 Perturbation Dimensions 的具体设定

这一节是 paper 的 Appendix A，我把它展开讲一下，因为这些参数定义了整个 benchmark 的 "perturbation space"。

### 2.1 Objects Layout
- **O1 Confounding Objects**: 在场景里随机加入 n 个 unseen distractor objects（416 类的 distractor pool），通过修改 BDDL task description file 实现
- **O2 Target Object Pose**: 对 target object 的 position (x,y,z) 和 orientation (pitch, yaw, roll) 做 random perturbation，但保留 essential semantic relations（比如 "black bowl next to the cookie box" 的 spatial relation 不破坏）

### 2.2 Background Textures
- **B1 Scene Theme**: 把 wall 纹理换掉（950 个 textures pool），改 scene XML
- **B2 Surface Appearance**: 桌面/地面纹理随机化

### 2.3 Light Conditions（4 个 sub-dimensions）
- **L1 Diffuse**: diffuse color 通过 RGB channels 调（e.g., `[1,0,0]` 是红光）
- **L2 Direction**: parallel light source 方向
- **L3 Specular**: 高光强度
- **L4 Shadows**: boolean，是否投射阴影

### 2.4 Camera Viewpoints
- **C1 Camera Distance**: 沿 optical axis 移动，1.01× ~ 2.00× 原值
- **C2 Spherical Position**: 以 scene center 为球心，azimuth Δθ ∈ [15°, 75°]，elevation Δϕ ∈ [15°, 75°]
- **C3 Camera Orientation**: 固定位置扰动 orientation (yaw, pitch, roll) ∈ [2°, 10°]

### 2.5 Robot Initial States
- **Initial Joint Angle**: qpos 加 random perturbation，magnitude ∈ [0.1, 0.5]

### 2.6 Language Instructions（LLM rewrite，3 个 sub-dimensions）

| Sub-category | Example |
|---|---|
| Original | "push the plate to the front of the stove" |
| R1 Distraction | "before turning on the burner, push the plate to the front of the stove" |
| R2 Common Sense | "propel the flat surface used for holding food toward the area designated for cooking heat adjustment" |
| R3 Reasoning Chain | "make sure the plate ends up at the front of the stove" |

### 2.7 Sensor Noise（5 个 sub-dimensions，每个有 L1-L5 难度）

| ID | Type | Key Params | L1 (weak) → L5 (strong) |
|---|---|---|---|
| N1 | Motion Blur | r, σ, θ | r=5,σ=2 → r=35,σ=20 |
| N2 | Gaussian Blur | σ | σ=1 → σ=10 |
| N3 | Zoom Blur | [s_min, s_max, step] | [1,1.11,0.01] → [1,1.56,0.03] |
| N4 | Fog | density α, decay β | α=0.5,β=3.0 → α=5.0,β=1.3 |
| N5 | Glass Blur | σ, δ, iters | σ=0.5,δ=1,iters=3 → σ=2.5,δ=5,iters=1 |

**Intuition**: 这个 perturbation space 覆盖了 vision (camera, light, background, noise, layout)、language、proprioception (robot state) 三个 modality，能很 fine-grained 地 probe 哪个 modality 是 model 的 bottleneck。

---

## 3. 实验的 Models 全景

paper 测了 10 个 model，跨 architecture (autoregressive vs diffusion) 和 training paradigm (web-data co-training, world modeling, RL)。我展开讲几个关键 architecture：

### OpenVLA (Kim et al., 2024)
- Backbone: Prismatic-7B VLM
- Visual encoder: 600M dual-backbone = **SigLIP** + **DINOv2**（沿 channel 维 concat，SigLIP 提供 semantic，DINOv2 提供 spatial）
- Projector: 2-layer MLP → visual features 投到 Llama2-7B 的 input space
- Action tokenization: 把每个 action dimension 离散化成 256 bins，替换 Llama tokenizer 中 256 个最少使用的 tokens
- Loss: standard next-token prediction on action tokens
- Pretraining: Open X-Embodiment (970k trajectories)
- 关键点：vision encoder 不 frozen 而是 fine-tune

**Repo**: https://huggingface.co/openvla

### OpenVLA-OFT (Kim et al., 2025)
- 基于 OpenVLA，关键改动：
  - **Parallel decoding**: 一次 forward 同时预测所有 actions（vs autoregressive 一个一个生成）
  - **Continuous action head**: MLP 输出 continuous action（vs discrete bins）
  - **L1 regression loss**（vs cross-entropy）
  - **FiLM (Feature-wise Linear Modulation)** 增强 language grounding

**Paper**: https://arxiv.org/abs/2502.19645

### OpenVLA-OFT_w
去掉 wrist camera，只用 third-person view。这个 variant 是关键的 ablation，揭示 wrist view 的作用。

### OpenVLA-OFT_m
在 LIBERO 全部 4 个 suites 上 joint training（vs suite-specific）。

### π0 (Black et al., 2024)
- 基于 **Transfusion** framework：一个 transformer 同时训练 flow-matching loss (continuous tokens) 和 cross-entropy loss (discrete tokens)
- 双 transformer: VLM base (**PaliGemma**) + smaller **action expert**
- Input: multi-RGB images, language, proprioceptive state q_t
- 两阶段训练：大规模 diverse pretraining + 高质量小数据 post-training

**Paper**: https://arxiv.org/abs/2410.24164

### π0-fast
- 用 **FAST tokenization** 压缩 action sequence
- FAST = **DCT (Discrete Cosine Transform)** 把 action trajectory 转到 frequency domain（sparse）+ **BPE (Byte-Pair Encoding)** lossless 压缩 sparse DCT 系数

**Paper**: https://arxiv.org/abs/2501.09747

### Nora (Hung et al., 2025)
- Backbone: **Qwen-2.5-VL-3B**
- FAST+ tokenizer
- Open X-Embodiment pretraining
- 8×H100 × 3 weeks ≈ 4000 GPU hours

**Paper**: https://arxiv.org/abs/2504.19854

### WorldVLA (Cen et al., 2025)
- Autoregressive action-world model，unify VLA + world model
- 初始化自 **Chameleon**
- 3 个 tokenizers：**VQ-GAN** (image) + **BPE** (text) + **action tokenizer** (256 bins per dim)
- 关键 architecture 创新：customized attention mask 在 action generation 时阻止 current action attend to previous actions，避免 error propagation
- Joint training: action modeling loss + world modeling loss (next frame prediction)

**Paper**: https://arxiv.org/abs/2506.21539

### UniVLA (Bu et al., 2025)
- Prismatic-7B VLM
- 核心: **task-centric latent action space**（discrete codebook）
- 3 阶段训练：
  1. Latent action model 自监督训练在大规模 video 上（DINOv2 reconstruction + language conditioning）
  2. Universal policy pretraining: observation + instruction → latent action tokens
  3. Fine-tuning: latent action prediction + low-level action regression
- History-augmented input: past latent actions 反馈进 context

**Paper**: https://arxiv.org/abs/2505.06111

### RIPT-VLA (Brohan et al., 2022 / Tan et al., 2025)
- 基于 OpenVLA-OFT
- 加 **auxiliary head** 预测 scale σ_θ → 形成 **factorized Laplace distribution**
  - Action a_t ~ Laplace(μ_θ, σ_θ)
  - log π_θ(a_t | a_{<t}, c) 有 closed-form
- 第三阶段：**Reinforcement Interactive Post-Training (RIPT)**
- **Dynamic Sampling Leave-One-Out PPO (LOOP)**：每个 context c_i 采 K 条 trajectories，用 **RLOO advantage** 估 advantage，dynamic rejection 过滤掉 all-success/all-failure 的 contexts（提供 informative gradient）

**Paper**: https://arxiv.org/abs/2505.17016

---

## 4. Single-dimension Perturbation 结果（Table 1 全景）

我把 Table 1 的关键数据 reorganize 一下：

| Model | Original | Camera | Robot | Language | Light | Background | Noise | Layout |
|---|---|---|---|---|---|---|---|---|
| OpenVLA | 76.5 | 1.1 (↓75.4) | 4.1 (↓72.4) | 26.8 (↓49.7) | 4.4 (↓72.1) | 25.3 (↓51.2) | 19.3 (↓57.2) | 31.6 (↓44.9) |
| OpenVLA-OFT | 97.1 | 59.7 (↓37.4) | 37.2 (↓59.9) | 81.5 (↓15.6) | 85.8 (↓11.3) | 92.4 (↓4.7) | 76.7 (↓20.4) | 77.1 (↓20.0) |
| OpenVLA-OFT_w | 95.3 | 16.8 (↓78.5) | 43.7 (↓51.6) | 73.2 (↓22.1) | 68.2 (↓27.1) | 92.5 (↓2.8) | 51.4 (↓43.9) | 72.3 (↓23.0) |
| OpenVLA-OFT_m | 97.6 | 57.9 (↓39.7) | 30.6 (↓67.0) | 83.6 (↓14.0) | 91.6 (↓6.0) | 83.6 (↓14.0) | 76.3 (↓21.3) | 73.2 (↓24.4) |
| π0 | 94.2 | 15.8 (↓78.4) | 6.6 (↓87.6) | 61.0 (↓33.2) | 79.6 (↓14.6) | 78.5 (↓15.7) | 79.4 (↓14.8) | 70.4 (↓23.8) |
| π0-fast | 85.5 | 66.4 (↓19.1) | 24.8 (↓60.7) | 63.3 (↓22.2) | 73.0 (↓12.5) | 67.7 (↓17.8) | 75.8 (↓9.7) | 70.3 (↓15.2) |
| Nora | 87.9 | 4.0 (↓83.9) | 41.1 (↓46.8) | 67.0 (↓20.9) | 31.0 (↓56.9) | 50.5 (↓37.4) | 17.6 (↓70.3) | 63.9 (↓24.0) |
| WorldVLA | 79.1 | 0.3 (↓78.8) | 30.2 (↓48.9) | 44.2 (↓34.9) | 29.4 (↓49.7) | 14.5 (↓64.6) | 12.2 (↓66.9) | 39.4 (↓39.7) |
| UniVLA | 95.2 | 4.3 (↓90.9) | 50.3 (↓44.9) | 71.8 (↓23.4) | 59.1 (↓36.1) | 80.0 (↓15.2) | 25.3 (↓69.9) | 34.3 (↓60.9) |
| RIPT-VLA | 97.5 | 58.3 (↓39.2) | 36.7 (↓60.8) | 80.1 (↓17.4) | 87.9 (↓9.6) | 90.4 (↓7.1) | 73.8 (↓23.7) | 76.5 (↓21.0) |

**Key intuition**：

1. **Camera & Robot state 是 universal Achilles' heel**。几乎所有 model 在这两个维度掉 40-90 个点。这暗示 models 缺乏 view-invariant representation 和 deep kinematic reasoning。

2. **Light / Background 相对 robust**（OpenVLA-OFT 系列掉 <15 个点）。表面看是好事，实际原因是 wrist camera 在做大部分的活——后面 3rd-black 实验揭示了这个。

3. **Language perturbation 影响最小**（平均 ↓25.3，仅次于 background）。这个 finding 表面上 robust，实际上 deeply concerning——section 4 证明 model 根本 ignore language。

4. **Architecture & training paradigm 决定 robustness**：
   - OpenVLA-OFT（有 wrist camera） vs OpenVLA-OFT_w（无 wrist camera）：camera perturbation 下 59.7% vs 16.8%，差 42.9 个点
   - π0 / π0-fast 用了 diverse co-training，普遍比 OpenVLA base 更 robust
   - RIPT-VLA 通过 RL post-training 在多个维度都拿到 SOTA-level robustness

---

## 5. Vision 真的被 attention 到了吗？— Object Layout & Light 的 fine-grained 分析

### 5.1 Object Layout 分解

Object Layout 被拆成两个 sub-dimensions:
- **Confounding objects**: 加 distractor
- **Target object displacement**: 改 target 的 pose

结果（Figure 1）：π0、π0-fast、RIPT-VLA、UniVLA、WorldVLA 在 confounding objects 下几乎不掉点。这说明它们确实 focus 在 target object 上，能 ignore distractor。

但是 displacement 一加，performance 大幅 drop。

**Intuition**: model 学到的是 "positional bias" 而非 "object semantic understanding"。它把 (target appearance, target position) 作为 cue 一起 memorize 了。Position 一变就崩。

### 5.2 Light 的 extreme ablation

设计了两个极端测试：
- **All-black**: 两个 camera 都替换成黑帧
- **3rd-black**: 只把 third-person view 替换成黑帧，保留 wrist camera

结果：
- All-black: 所有 model 接近 0%（确认 vision 是必需的）
- 3rd-black: 三个 model 仍然达到 43.6, 43.0, 67.3

**这个实验非常 clever**。它说明：
- Light perturbation 主要影响 third-person view 的全局 appearance
- Wrist camera 提供的是 **illumination-invariant** 的 close-range 几何 / contact cues
- 所以有 wrist camera 的 model 在 light perturbation 下几乎不掉点，不是因为模型 robust，而是因为 wrist view 本身 robust

对比实验：只用 third-person view 的 OpenVLA / Nora / WorldVLA 在 light perturbation 下掉 50-60 个点。

**Insight**: "Light robustness" 在很多 model 上是个 illusion，实际是 wrist camera 的 hardware 优势。

---

## 6. Language 真的被用了吗？— Section 4 的 critical experiments

这是 paper 最 interesting 的部分。Language perturbation 影响最小，作者提出三个 hypothesis：

1. Model 有 strong language generalization，所以 robust
2. Model 只 extract 关键 keywords 做 matching
3. Model 完全 ignore language，rely on vision

### 6.1 Blank Instruction 实验（Figure 3a）

把 language input 完全替换成 empty string，看 performance 是否崩。

结果：
- **Object suite**: OpenVLA-OFT 几乎不掉点
- **Long suite**: 明显掉点

**Insight**: Object suite 是 single-step task，可以靠 vision 直接 solve，language 是 redundant。Long suite 是 long-horizon task，必须有 language 来 disambiguate steps，所以 model 不得不 attend to language。

也就是说，**VLA 实际上是 VA（Vision-Action）model**，只在 long-horizon task 才被迫变成真正的 VLA。

### 6.2 Goal Replacement 实验（Figure 3b）

这个实验最 decisive。把 instruction 里的 target object 换成 scene 中另一个 object，比如：
- 原: "pick up the alphabet soup and place it in the basket"
- 改: "pick up the butter and place it in the basket"

结果：**所有 model 几乎掉到 0%**。OpenVLA-OFT 最严重。

更 damning 的是 rollout case（Figure 10）：即使 instruction 改成抓 butter，model 还是执行抓 alphabet soup 的动作。

**这直接证明 VLA 是 visual pattern matcher**：
- 输入 scene configuration → 输出 memorized action sequence
- Language instruction 几乎是 decorative
- "Robustness to language perturbation" 是个 illusion，本质是 language 被忽略，所以怎么 perturb 都没影响

---

## 7. Compositional Generalization Gap — Section 5 的 statistical analysis

这部分是 paper 里最 mathematical 的部分。作者用 statistical 定义来 rigorously 分析 multi-dimension perturbation 下的 interaction effects。

### 7.1 数学定义

定义 random variable D_i 为 indicator：第 i 类 perturbation 是否 applied（1=是，0=否）。
定义 Y 为 success indicator（1=success，0=fail）。

**Success rate** 定义为 conditional probability:

$$s(D_i = d_i, D_j = d_j) = P(Y = 1 \mid D_i = d_i, D_j = d_j), \quad d_i, d_j \in \{0, 1\}$$

这里 $d_i, d_j$ 是 binary indicator，表示 perturbation 是否 applied。

**Joint probability conditioned on Y=1**（成功案例中某个 perturbation 组合的占比）:

$$p(D_i = d_i, D_j = d_j \mid Y = 1) = \frac{s(D_i = d_i, D_j = d_j)}{\sum_{a, b \in \{0, 1\}} s(D_i = a, D_j = b)}$$

分母是四种组合的 success rate 之和，相当于一个 normalization constant。

**Marginal probability**（在成功案例中第 i 类 perturbation 出现的概率）:

$$p(D_i = 1 \mid Y = 1) = \frac{s(D_i = 1, D_j = 0) + s(D_i = 1, D_j = 1)}{\sum_{a, b \in \{0, 1\}} s(D_i = a, D_j = b)}$$

Intuition: $p(D_i = 1 \mid Y = 1)$ 高 → 第 i 类 perturbation 经常在成功案例中 co-occur → model 对这个 perturbation robust；低 → 对这个 perturbation 敏感。

**Compositionality Gap** 定义为 conditional covariance:

$$\Delta_{ij} \triangleq \text{Cov}(D_i, D_j \mid Y = 1) = \mathbb{E}[D_i D_j \mid Y = 1] - \mathbb{E}[D_i \mid Y = 1] \mathbb{E}[D_j \mid Y = 1]$$

展开:

$$\Delta_{ij} = p(D_i = 1, D_j = 1 \mid Y = 1) - p(D_i = 1 \mid Y = 1) p(D_j = 1 \mid Y = 1)$$

- $\Delta_{ij} > 0$：正交互作用，model 能 jointly handle 两个 perturbation（synergy）
- $\Delta_{ij} < 0$：负交互作用，组合 perturbation 引入了额外难度超过 independent 效应
- $\Delta_{ij} = 0$：满足 independence assumption

### 7.2 实验结果

作者跑 **2000 次 independent repeated experiments**，用 OpenVLA-OFT 测 pairwise perturbations。Figure 4 的 heatmap 显示：
- Upper triangular A_ij (i<j): product of single-dimension probabilities (independence 假设下)
- Lower triangular A_ij (j<i): actual joint outcomes

然后 $\Delta_{ij} = A_{ij} - A_{ji}$ 算 gap。

**所有 pair 都是 negative gap**。意味着组合 perturbations 的实际成功率比 independence 假设预测的还低。Perturbations 之间是 coupled noise sources，model 学到的 representation 是 entangled 的。

### 7.3 Chi-square Test 验证 significance

用 2×2 contingency table:

|  | D_j=0 | D_j=1 | Total |
|---|---|---|---|
| D_i=0 | n_00 | n_01 | n_0. |
| D_i=1 | n_10 | n_11 | n_1. |
| Total | n_.0 | n_.1 | n |

Chi-square statistic:

$$\chi^2 = \sum_{r, c} \frac{(O_{rc} - E_{rc})^2}{E_{rc}}$$

$O_{rc}$ 是 observed count，$E_{rc} = \frac{\text{(row total)} \times \text{(column total)}}{n}$ 是 independence 假设下的 expected count。

p-value: 

$$p = P(\chi^2_{\text{dof}=1} \geq \chi^2)$$

Table 9 显示大部分 pair 的 p-value 远小于 0.05，比如 Env-Camera 的 χ²=26.1, p=3.33e-07。证明 interaction effects 是 statistically significant 的，不能忽略。

**Intuition**: 单维度 robustness 测试是 necessary but not sufficient。Multi-dimension robustness 必须 independently evaluate，因为 perturbation 之间有 non-linear interaction。

---

## 8. LIBERO-Plus Benchmark 构造

### 8.1 流程

1. 从 LIBERO 的 40 个 evaluation tasks 出发
2. 每个任务在 7 个维度 × 4 个 sub-suites (Spatial, Object, Goal, Long) 生成 500 个 instances → 14,000 candidate tasks
3. 用 baseline models 跑，删除 ceiling tasks（所有 model 都 solve 的）和 floor tasks
4. 平衡 sub-dimensions 防止 bias
5. 最终保留 **10,030 tasks**，覆盖 7 个维度 + 21 个 sub-dimensions

### 8.2 难度分级（L1-L5）

用 4 个 representative models (OpenVLA-OFT, π0, π0-fast, UniVLA) 跑每个 task，根据多少 model solve 来分级：

- **L1**: 4 个 model 都 solve (easiest)
- **L2**: 3 个 model solve
- **L3**: 2 个 model solve
- **L4**: 1 个 model solve
- **L5**: 0 个 model solve (hardest)

这种 heuristic 分级的好处是能 reveal model 在 different difficulty regime 下的 behavior pattern（Figure 5 / Figure 8 的 line plots）。

### 8.3 与其他 benchmark 对比（Table 3）

| Benchmark | Automation | Sim | Fine-grained | 维度覆盖 |
|---|---|---|---|---|
| AGNOSTOS | × | RLBench | × | 仅 Layout, Robot |
| RL4VLA | × | ManiSkill | × | Layout, Light, Camera, Robot |
| INT-ACT | × | ManiSkill | × | Layout, Background |
| GemBench | × | RLBench | × | Layout, Background, Robot |
| VLATest | ✓ | ManiSkill | × | Light, Noise |
| COLOSSEUM | ✓ | RLBench | × | Layout, Background, Light, Robot |
| **LIBERO-Plus** | ✓ | LIBERO | ✓ | 全部 7 维度 |

LIBERO-Plus 的优势：**全 7 维度覆盖 + 自动化 + fine-grained sub-dimensions + difficulty levels**。

**Project page**: https://sylvestf.github.io/LIBERO-plus/
**Code**: https://github.com/sylvestf/LIBERO-plus
**Models**: https://huggingface.co/collections/Sylvest/libero-plus

---

## 9. Training on Generalized Set — 能不能 recover robustness？

### 9.1 数据构造

用同样的 automated pipeline 生成 **20,000+ successful trajectories**，覆盖 6 种 variants:
- Objects spanning（confounding objects only，因为 pose change 的 auto trajectory 不可靠）
- Background environment sampling
- Light variations
- Camera-view shifts（角度差 5° 避开 test set）
- LLM-based language rewrites
- Sensor noise

Distributions 见 Figure 9（7 个 action dimension 的 marginal 分布）。

### 9.2 训练配置

- Starting weights: OpenVLA-OFT_m
- Hardware: 8×A100 GPUs
- LR: 5×10^-4
- Steps: 100,000
- Batch size: 2 per GPU → effective 16
- Optimizer: AdamW, weight decay 0.1
- Schedule: cosine with warmup
- Format: rlds

### 9.3 结果（Table 2）

| Model | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|---|---|---|---|---|---|---|---|---|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| OpenVLA-OFT_w | 10.4 | 38.7 | 70.5 | 76.8 | 93.6 | 49.9 | 69.9 | 55.8 |
| NORA | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 | 39.0 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| π0 | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π0-Fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| RIPT-VLA | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 68.4 |
| OpenVLA-OFT_m | 55.6 | 21.7 | 81.0 | 92.7 | 91.0 | 78.6 | 68.7 | 67.9 |
| **Ours** | **92.8** | 30.3 | 85.8 | 94.9 | 93.9 | 89.3 | 77.6 | **79.5** |
| Δ vs baseline | ↑37.2 | ↑8.6 | ↑4.8 | ↑2.2 | ↑2.9 | ↑10.7 | ↑8.9 | ↑11.6 |

**Critical observations**:

1. **Camera 维度涨 37.2 个点**（55.6% → 92.8%），这是最大的 gain。说明 camera robustness 主要是个 **data coverage** 问题，不是 architecture 问题。

2. **Robot 维度只到 30.3%**，即使训练后还是低。说明 proprioceptive understanding 是个更深的 representation 问题，单纯 data diversity 解决不了。可能需要更好的 kinematic representation learning。

3. **Language 维度提升不大**（81.0 → 85.8）。Paper 没直接 probe 训练后 model 是否真正 attend to language，但从提升幅度看，可能 model 还是 partially ignore language。这个 open question 值得 follow-up。

4. **总体 79.5%** vs baseline 67.9%，提升 11.6 个点。说明 **data diversity 是 cheap but effective 的 robustness lever**。

---

## 10. 关键 Findings 总览

把 paper 的 9 个 findings 浓缩一下：

| # | Finding | 重要性 |
|---|---|---|
| F1 | VLA 普遍 fragile，camera/robot state 最致命 | High |
| F2 | Light/background 表面 robust 实际是 wrist camera 的功劳 | Medium |
| F3 | Language perturbation 影响小，是 illusion | High |
| F4 | Architecture & training paradigm 决定 robustness | Medium |
| F5 | Models 学 positional bias 而非 object semantics | High |
| F6 | Wrist camera 是 illumination robustness 的核心来源 | Medium |
| F7 | VLA 没有跨 object 的 instruction-following generalization | High |
| F8 | VLA 依赖 fixed vision-action mapping，不 dynamic plan | High |
| F9 | Compositional generalization gap 是 negative & significant | High |

---

## 11. 与更广 context 的联系

### 11.1 跟 LLM 的 benchmark overfitting 现象 parallel

你之前多次讲过 LLM benchmark 上的 high accuracy 不一定代表 true understanding。VLA 这边一模一样：95%+ 的 LIBERO success rate 给人 false sense of competence，但稍微 perturb 一下就崩。

**类比**: 
- LLM 在 MMLU 高分 ≠ 真的 reasoning
- VLA 在 LIBERO 高分 ≠ 真的 manipulation understanding

### 11.2 跟 multimodal learning 的 modality imbalance 问题

在 VLM 里，vision 经常被 LLM backbone "ignore"（language 占主导）。VLA 里反过来：vision 占主导，language 被 ignore。这个 modality imbalance 是 bidirectional 的。

参考: https://arxiv.org/abs/2310.04191 (modality imbalance in VLMs)

### 11.3 跟 Software 2.0 的思想一致

你之前讲过 Software 2.0: data 决定 model 的 capability boundary，architecture 只是 enabler。这个 paper 的 Section 6.2 实验完美验证：单纯改 data distribution（mixed fine-tuning on generalized set）就大幅提升 robustness，architecture 不变。

### 11.4 跟 RLHF / RLAIF 类比

RIPT-VLA 用 RL post-training 在多个维度拿到 SOTA-level robustness（Table 2 中 68.4% total）。这跟 LLM 的 RLHF 思想类似：用 environment reward 来 align model behavior 而非靠 supervised imitation。

Paper: https://arxiv.org/abs/2505.17016

### 11.5 Compositional Generalization 的更广 context

Compositional generalization 是个经典问题:
- **CLEVR** (Johnson et al., 2017): visual reasoning 的 compositionality
- **SCAN** (Lake & Baroni, 2018): language 的 compositionality
- **COGS** (Kim & Linzen, 2020): semantic parsing 的 compositionality

LIBERO-Plus 把这个 framework 引到 VLA 上，发现 negative gap 是普遍现象——perturbation 不能 linearly add，组合起来 super-linear 地 bad。

---

## 12. Paper 的 Limitations 和 Open Questions

1. **只在 simulation 上做**：Real-world transfer 还需验证。Sim-to-real 的 perturbation distribution 可能不同。
2. **Compositional 只测了 pairwise**：高阶组合（3-way, 4-way）的 interaction 还未 probe。
3. **Difficulty level 是 heuristic 的**（4 model vote），不够 principled。可以用 IRT (Item Response Theory) 之类的方法。
4. **Robot state 维度训练后还是 30.3%**：proprioception 的 representation 问题更深，data diversity 不够。
5. **没 probe 训练后 model 是否真正 attend to language**：blank instruction / goal replacement 实验只在 baseline 上做。
6. **Perturbation 是 static 的**：没测 dynamic perturbation（运行中变化）的 robustness。

---

## 13. 直觉的 takeaway

这篇 paper 给我几个 deep intuition:

1. **Benchmark saturation 是个 red flag**。当所有 model 都在 benchmark 上 >95% 时，说明 benchmark 已经 saturate，需要 harder test 才能区分 model quality。LIBERO-Plus 是这种 harder test 的范例。

2. **Vision-Language-Action 经常是 Vision-Action**。当前 VLA architecture 中 language 通道经常是 decorative，模型实际靠 vision 做 decision。这个现象在 long-horizon task 才被 forced 改变。这暗示我们需要 architecture level 的改动，比如把 language 作为 hard conditioning 而非 soft hint。

3. **Robustness 来自 data diversity > architecture sophistication**。Section 6.2 实验：单纯用更多样化 data mixed fine-tune 就能 recover 大部分 robustness。这跟你 Software 2.0 思想完全一致。

4. **Wrist camera 是 manipulation task 的"作弊"**。它把很多 visual reasoning 问题转化成 close-range geometry 问题。这暗示未来的 manipulation system 可能需要 more cameras / more modalities，而非更聪明的高层 reasoning。

5. **Compositional robustness 不能从 single-dimension robustness 推断**。Perturbation 之间有 non-linear interaction，必须做 joint evaluation。这个 insight 对 future benchmark 设计有指导意义。

6. **Positional bias 是 deep problem**。Models 记 (object, position) 而非 object semantics。这跟 LLM 的 positional bias 有 echo——transformer 学 position 而非 semantic 是 known issue。

---

## 14. 相关 references

我把关键的 follow-up 和 reference 整理一下：

**VLA models**:
- OpenVLA: https://openvla.github.io/
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- π0: https://arxiv.org/abs/2410.24164
- π0-fast: https://arxiv.org/abs/2501.09747
- Nora: https://arxiv.org/abs/2504.19854
- WorldVLA: https://arxiv.org/abs/2506.21539
- UniVLA: https://arxiv.org/abs/2505.06111
- RIPT-VLA: https://arxiv.org/abs/2505.17016

**Benchmarks**:
- LIBERO: https://libero-project.github.io/
- COLOSSEUM: https://arxiv.org/abs/2402.08191
- VLATest: https://arxiv.org/abs/2501.10846
- AGNOSTOS: https://arxiv.org/abs/2505.15660
- GemBench: https://arxiv.org/abs/2410.08772

**Compositional generalization**:
- SCAN: https://arxiv.org/abs/1710.10370
- COGS: https://arxiv.org/abs/2010.04495
- CLEVR: https://arxiv.org/abs/1612.06890

**Modality imbalance in VLMs**:
- https://arxiv.org/abs/2310.04191

**Project page**: https://sylvestf.github.io/LIBERO-plus/
**Code**: https://github.com/sylvestf/LIBERO-plus
**Models**: https://huggingface.co/collections/Sylvest/libero-plus

---

## 15. 给你的几个 follow-up 思路

如果你想从这篇 paper 延伸做点什么，我猜可能有几个方向:

1. **Probe language attention mechanistically**: 用 attention map / activation patching 看 VLA 在 inference 时 language tokens 是否真被 attention 到。类似 mechanistic interpretability 在 VLA 上的应用。

2. **Hard language conditioning**: 设计 architecture 让 language 必须被用，比如把 language embedding 作为 hard constraint 注入 action head。验证能否恢复 instruction-following。

3. **Proprioceptive representation learning**: Robot state 维度即使训练后也只 30%，说明 proprioception representation 是 bottleneck。可能需要 contrastive learning 之类的 self-supervised method 学 invariant representation。

4. **Real-world LIBERO-Plus**: 把这套 perturbation framework 移到 real robot，看 sim 上的 finding 是否 hold。

5. **Higher-order compositional generalization**: 测 3-way, 4-way perturbation 的 interaction，看 gap 是否继续 negative 且 magnitude 怎么变化。

希望这个 walkthrough 给你足够的 technical depth 和 intuition。如果有 specific 部分想深挖，比如某个 architecture 的具体实现，或者 compositional gap 的更精细 statistical analysis，我可以再展开。
