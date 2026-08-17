---
source_pdf: Hunyuan-GameCraft High-dynamic Interactive Game Video Generation.pdf
paper_sha256: a14888c99c588cfd232913e4d74055eb868aa9eef0d45347808c3367d46a56dc
processed_at: '2026-08-05T08:23:03-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Hunyuan-GameCraft

## 一句话版本

他们把一个 video generation model 训练成了可以"玩"的东西——你按键盘，它生成对应的 game 画面，连续按可以生成无限长的 game video，而且画面对得上你的操作。

---

## 他们到底在解决什么问题

想象你拿 GPT 生成文字：你打一个字，它接一个字，你随时可以改方向。这种交互性在 video generation 里一直做不到。

现有的"game world model"大致几类：

**GameNGen** (Google DeepMind, 2024): 用 diffusion model 学 DOOM。很酷，但只学了一个 game，输入是离散的按键 (W/A/S/D 这种 binary signal)，分辨率 240p，没法 generalize 到新场景。链接: https://arxiv.org/abs/2408.14837

**Oasis** (Decart, 2024): Minecraft 专用，real-time 但画质糙，场景里没什么动态物体。链接: https://www.decart.ai/articles/oasis-interactive-ai-video-game-model

**Matrix / Matrix-Game** (字节, 2024-2025): 多 game 数据训练，但 action space 是 7 keys + mouse 这种离散的，long video 会 drift，scene memory 基本没有。链接: https://arxiv.org/abs/2412.03568

**Genie 2** (DeepMind, 2024): 单张图生成可交互 3D world，用 latent action (model 自己定义 action, 不是 human-readable 的)。很神秘，没开源。链接: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/

共同的痛点是：
1. **Action 是离散的**：按 W 就是 forward，按住 W 多久不知道，鼠标移动多少不知道。Model 看到的是 binary signal，丢失了所有 fine-grained 信息
2. **Long video 会崩**：autoregressive 生成 10 秒以上就开始 quality degrade，场景 drift，物体变形
3. **太慢**：diffusion model 一步要几秒，game 要 30 FPS，差了两个数量级
4. **没有 memory**：你走过去看到一个箱子，绕一圈回来，箱子不见了——model 不记得场景

Hunyuan-GameCraft 把这四件事一起处理。

---

## 核心招式一：把键盘鼠标变成连续 camera 轨迹

这是最 elegant 的 idea。传统做法是直接告诉 model "用户按了 W"，model 学到 "W → forward"。但这丢了一堆信息：

- 按住 W 0.1 秒 vs 按住 W 2 秒，应该走多远？
- 鼠标往右拖一点 vs 拖很多，应该转多少度？
- 同时按 W+A，应该是斜着走，怎么表示？

他们的做法：**把所有 keyboard/mouse 输入都映射到一个连续的 6-DoF camera trajectory space**。

数学上看公式 (1)：

$$\mathcal{A} := \left\{ \mathbf{a} = (\mathbf{d}_{\mathrm{trans}}, \mathbf{d}_{\mathrm{rot}}, \alpha, \beta) \right\}$$

- $\mathbf{d}_{\mathrm{trans}} \in \mathbb{S}^2$：translation 方向，单位向量在球面上。$\mathbb{S}^2$ 就是 3D 中所有长度为 1 的向量集合
- $\mathbf{d}_{\mathrm{rot}} \in \mathbb{S}^2$：rotation 轴方向
- $\alpha \in [0, v_{\max}]$：translation 速度（标量）
- $\beta \in [0, \omega_{\max}]$：rotation 角速度（标量）

说白了，每帧的 action 就是一个 tuple："往哪个方向走、走多快、绕哪个轴转、转多快"。

这样设计的好处：
- **可插值**：W 是 forward = $(0,0,1)$，A 是 left = $(-1,0,0)$，同时按 W+A 就自然 interpolate 成 $(-\frac{1}{\sqrt{2}}, 0, \frac{1}{\sqrt{2}})$，斜着走
- **可变速度**：按住 W 的时间长短 → 不同的 $\alpha$ 值 → 不同的移动距离
- **跨 game 通用**：不管 Assassin's Creed 还是 Cyberpunk，camera motion 的物理是一样的

他们还故意去掉了 roll 自由度（绕 z 轴旋转），因为玩家头不会侧着 roll，这是 gaming convention。所以实际是 5-DoF。

然后这个 continuous action 通过 Plücker embedding 转成 camera ray 表示，跟 video latent 在 spatial-temporal 上对齐。Plücker embedding 简单说就是用 $(\mathbf{d}, \mathbf{m})$ 表示一条 3D 线，其中 $\mathbf{d}$ 是方向，$\mathbf{m} = \mathbf{o} \times \mathbf{d}$ 是 moment（起点和方向的叉积），能 encode camera 的 intrinsic + extrinsic。

参考 CameraCtrl 用 Plücker 的原始 paper: https://arxiv.org/abs/2404.02101

---

## 核心招式二：Hybrid History Condition 让长 video 不崩

这是最巧妙的 engineering。

### 先说 naive 方案为什么不 work

**方案 A：只用第一帧做 condition**
每生成一段新 video，只看最初那一张图。问题：你走了 1 分钟后回头，model 已经忘了你后面有什么。Quality 会逐渐 collapse。

**方案 B：用上一段 video 的最后一帧做 condition**
每段新 video 都接在上一段后面。问题：model 学到"继续之前的 motion"，当你突然按 S（后退）想改变方向时，model 反应不过来，因为 strong prior 说"应该继续 forward"。

**方案 C：用上一整段 video 做 condition**
History 信息最丰富。问题更严重：model 直接复制之前的 motion，action 控制完全失效。

### 他们的解法：训练时三种 condition 混着喂

训练时随机 sample 不同类型的 condition：
- **Single image frame** (25% 概率)：只看一张图，逼 model 学会响应当前 action
- **Previous clip 的最后一帧** (70% 概率)：正常 autoregressive extend，保证 continuity
- **多个 previous clip** (5% 概率)：long-range memory，少量但关键的 long-context 训练信号

这个比例 0.25 / 0.70 / 0.05 是 ablation 试出来的。70% 的 clip condition 占主导，保证 long rollout 不崩；25% 的 single frame 让 model 学会"当 history 跟当前 action 矛盾时，听 action 的"；5% 的 multi-clip 是 long-context 的正则化。

实现上用一个 binary mask：
- Mask=1 的部分是 history frames，保持 clean latent (不加噪)
- Mask=0 的部分是要 generate 的新 frames，按 noise schedule 加噪

Denoising 时，clean history latent 通过 flow matching 引导 noisy chunk latent 生成，model 同时学到"跟随 history"和"响应 action"两个能力。

### 实验数据 (Table 4)

| Condition 方案 | FVD↓ (画质) | DA↑ (动态) | RPE trans↓ (控制误差) |
|---|---|---|---|
| Single frame only | 1655.3 | 47.6 | 0.07 |
| Clip only | 1743.5 | 55.3 | 0.16 |
| **Hybrid (theirs)** | **1554.2** | **67.2** | **0.08** |

Single frame 的 RPE trans=0.07 控制最好，但 DA 只有 47.6 (动态不够) 且 long rollout 会崩。Clip only 的 DA 高但 RPE trans 飙到 0.16 (控制差)。Hybrid 同时拿到最好的画质 (1554.2)、最好的动态 (67.2)、接近最好的控制 (0.08)。

这是典型的 "1+1+1 > 3" 现象，multi-task training 互相正则化。

---

## 核心招式三：Distillation 提速 20 倍

### 问题

原始 HunyuanVideo 是 50 step 的 flow matching，每步要跑 2 次 forward (CFG 要 conditional + unconditional)，生成 33 帧要好几秒。对应 Table 2 里 "Ours" 的 FPS 只有 0.25。Game 要 30 FPS，差 120 倍。

### 解法：PCM + CFG Distillation

**PCM (Phased Consistency Model)** [28]: 把 50-step diffusion distill 成 8-step consistency model。Consistency model 的核心 idea 是训练一个 network，让它在 trajectory 上任意点都能直接预测最终结果，而不是 step-by-step denoise。PCM 在此基础上分阶段施加 consistency constraint，避免 high-resolution 下 quality degradation。链接: https://arxiv.org/abs/2405.18481

**CFG Distillation**: 公式 (2) 的核心 idea—

$$\hat{u}_\theta = (1+w) \cdot u_\theta(\text{conditional}) - w \cdot u_\theta(\text{unconditional})$$

标准 CFG 每个 step 要跑 2 次 forward。Distillation 训一个 student network $u_\theta^s$，让它单次 forward 直接 output 上式右边的结果。

Loss 就是让 student 模仿 teacher 的 CFG 输出：

$$L_{cfg} = \mathbb{E}\left[\|\hat{u}_\theta - u_\theta^s\|_2^2\right]$$

变量说明：
- $z_t$：timestep $t$ 的 noisy latent
- $w$：guidance scale，从训练分布 $p_w$ 采样 (通常 1-10)
- $T_s$：text prompt
- $\hat{u}_\theta$：teacher 的 CFG 输出 (2 次 forward)
- $u_\theta^s$：student 的直接输出 (1 次 forward)

Inference 时只需要 student 单次 forward，CFG 的 2x 开销直接抹掉。结合 PCM 的 8x reduction，总 speedup 约 16-20x。

### 实测

Table 2:
- Original: 0.25 FPS, FVD 1554.2
- + PCM: **6.6 FPS**, FVD 1883.3

Speedup 26 倍，代价是 FVD 上升约 20%，Dynamic Average 从 67.2 降到 43.8。但 RPE 控制误差不变 (0.08/0.20)，说明 action accuracy 完全保留。

6.6 FPS 还达不到 game-ready 的 30 FPS，但比 Matrix-Game 的 0.06 FPS 已经高了 100 倍，至少 demo 级别可玩。

参考 Consistency Model 原始 paper: https://arxiv.org/abs/2303.01469
参考 LCM (Latent Consistency Model): https://arxiv.org/abs/2310.04378

---

## 数据工程：1M+ clips from 100+ AAA games

这部分是 industrial lab 的优势，academic group 很难复现。

四阶段 pipeline (Figure 3)：

**1. Partition**: 2-3 小时的 gameplay 录像，先用 PySceneDetect 切成 scene-level clip，再用 RAFT 算 optical flow gradient 检测 action boundary，最终切成 6 秒一段的 coherent clip。1M+ clips，1080p。链接: https://github.com/Breakthrough/PySceneDetect

为什么用 RAFT？因为单纯按时间切会把"突然转头瞄准"这种 motion discontinuity 切进同一段，训练时 action label 就乱了。RAFT 的 optical flow 能 detect 这种突变，保证每 6s clip 内 motion 是连续的。

**2. Filter**: 三层
- Kolors quality assessment 去低质
- OpenCV luminance filter 去 dark scene
- Qwen2-VL 做 gradient detection 综合过滤

**3. Action Annotation**: 关键步骤。用 Monst3R [35] 从 monocular video 重建 6-DoF camera trajectory。Monst3R 是 CVPR 2024 work，能在 non-rigid scene motion 下 estimate geometry，比 SLAM 鲁棒。每个 clip 每帧都标上 camera position + orientation，这就是训练时的 action label。链接: https://arxiv.org/abs/2410.03825

**4. Captioning**: Qwen2-VL 生成两级 caption (30 char summary + 100+ char detail)，训练时 random sample，让 model 对 caption granularity 鲁棒。

### Synthetic Data 的妙用

他们还额外 render 了 ~3000 个 high-quality motion sequence，从 curated 3D assets，多个 starting position × varying speed。

Ablation (Table 4 a/b/g) 揭示一个 trade-off：

| Training Data | FVD↓ | DA↑ | RPE trans↓ |
|---|---|---|---|
| Only Synthetic | 2550.7 | 34.6 | **0.07** |
| Only Live | 1937.7 | **77.2** | 0.16 |
| Hybrid 1:5 | **1554.2** | 67.2 | **0.08** |

Synthetic data 的 motion 是 ground truth 精确已知的，所以 action control 好 (RPE 0.07)。但 synthetic scene 没 NPC 没 particle，dynamic 弱 (DA 34.6)。Live data 反过来。

混合 1:5 (synthetic : live) 拿到两个 best of both worlds。这个 ratio 是 ablation 找到的 sweet spot。

### Distribution Balancing

Game video 有强烈的 forward-motion bias (玩家大部分时间在往前走)，会导致 model 只学好 forward，其他方向烂。

两个 fix：
- **Stratified sampling**：3D 方向空间均匀采样
- **Temporal inversion**：视频倒放，backward motion 数据翻倍

这个 trick 让 cross-domain RPE trans 从 Matrix-Game 的 0.18 降到 0.08，减少 55%。

---

## 跟 baseline 比怎么样

Table 2 最关键的几个数字：

| Model | FVD↓ | DA↑ | RPE trans↓ | FPS↑ |
|---|---|---|---|---|
| CameraCtrl | 1580.9 | 7.2 | 0.13 | 1.75 |
| MotionCtrl | 1902.0 | 7.8 | 0.17 | 0.67 |
| WanX-Cam | 1677.6 | 17.8 | 0.16 | 0.13 |
| Matrix-Game | 2260.7 | 31.7 | 0.18 | 0.06 |
| **Hunyuan-GameCraft** | **1554.2** | **67.2** | **0.08** | 0.25 |
| + PCM | 1883.3 | 43.8 | 0.08 | **6.6** |

几个观察：

**Dynamic Average 67.2 是 Matrix-Game 的 2 倍**。这说明 Hunyuan-GameCraft 生成的画面里物体运动幅度远大于其他。CameraCtrl/MotionCtrl 这种 camera-controlled T2V 本来就不擅长高动态，DA 只有 7-8。Matrix-Game 31.7 已经不错，但 Hybrid history condition 偏 stability 牺牲 dynamics。Hunyuan-GameCraft 的 hybrid + continuous action 让 dynamic 大幅提升。

**RPE trans 0.08 vs Matrix-Game 0.18**，减少 55%。这是 continuous action space + synthetic data 精确 annotation 的功劳。Discrete key 的信息量太低，model 很难学到 fine-grained mapping。

**FPS 6.6 with PCM**。Matrix-Game 0.06 FPS 根本不可玩，相当于每 16 秒生成一帧。Oasis 虽然能 real-time，但画质差且只支持 Minecraft。Hunyuan-GameCraft 6.6 FPS 至少 demo 级别可玩。

**FVD 1554.2 best**。视觉质量也领先，说明没 trade quality for control。

User study (Table 3) 5 个维度全是 4.4+，其他 baseline 最好才 3.23 (MotionCtrl on Temporal Consistency)。Human preference 跟 metric 一致，说明 evaluation 设计合理。

---

## 一些 Karpathy 角角的思考

### Continuous vs Discrete Action 的深层类比

把 keyboard 的 discrete input lift 到 continuous camera trajectory space，这个 abstraction 的提升跟几个经典类比很像：

- **Command line → Natural language**：从 "rm -rf" 到 "删除这个文件夹"，abstraction level 提升，表达能力指数级增长
- **Hard-coded policy → RL learned policy**：从手写规则到 reward function，flexibility 大增
- **RT-2 的 action tokenization**：把 robot action 从 discrete command 变成 language-conditioned continuous action

共同模式：**把 interface 从 designer 定义的离散空间升级成物理上有意义的连续空间**，model 的 generalization 能力就跳一个台阶。

参考 RT-2: https://arxiv.org/abs/2307.15818

### Hybrid History Condition 的 general pattern

这个 idea 本质上是 "soft attention to history" 的 training-time 实现。LLM 领域有类似 trade-off：
- Short context = responsive but forgetful
- Long context = knowledgeable but rigid

LLM 用 attention 自动学权重，diffusion model 没 attention over time 的天然机制，所以 Hunyuan-GameCraft 用 **conditioning ratio** (0.25/0.70/0.05) 手动控制。

这让我想到 *Mixture of Conditioning* 这个更 general 的 idea：训练时随机 sample 不同 strength 的 condition，让 model 学到 "何时 rely on short vs long context"。这个 pattern 在 robotic learning、agent system 里都有应用空间。

### Pixel-level World Model vs JEPA

Hunyuan-GameCraft 是 pixel-level world model，跟 LeCun 推的 JEPA 思路相反。

JEPA 学 abstract representation (joint embedding)，不学 pixel-level reconstruction。优点是 compute efficient、long horizon planning 好，缺点是需要单独 decoder 才能可视化、action grounding 难。

Pixel-level approach (Hunyuan-GameCraft, GameNGen, Oasis) 直接学 pixel 生成。优点是 visualization ready、end-to-end、action 直接对应 visual change；缺点是 compute expensive、long rollout 会 drift。

Game 这种 visual-heavy、short-horizon (每帧 action)、visual fidelity 重要的场景，pixel-level 更直接。但 long-horizon planning (玩家有 strategy)、physical reasoning (碰撞、重力) 仍然缺。未来 hybrid (pixel + abstract state) 可能是方向。

参考 JEPA: https://openreview.net/pdf?id=BvAj8DtLb2
参考 LeCun 的 world model vision: https://openreview.net/pdf?id=BZ5a1r-kVsf

### 为什么 Causal VAE 不 work

Paper 提到 causal VAE [33] 导致 streaming generation 失败。Causal VAE 设计是为了 real-time encoding (只看 past frame 编码当前)，但代价是 temporal compression 不均匀：initial frame 保留信息多，subsequent frame 压缩狠。

对 streaming generation (每生成 1-2 帧就 condition 下一段)，这种 non-uniformity 会让 condition 信号 noisy，长 video 累积 drift。

Hunyuan-GameCraft 选 **chunk-level 全 encode**，每次 condition 一整段 (33 帧)，encoding 质量均匀。代价是 latency 高 (要等整段 encode 完)，用 PCM distillation 补偿。

这是 engineering 的经典 trade-off：**consistent-but-slow vs fast-but-noisy**。他们选前者，再用 model distillation 加速。这个 pattern 在 LLM 里也有对应：长 context window (consistent) vs sliding window RAG (fast but lossy)。

参考 CausalVAE: https://arxiv.org/abs/2004.11186

### Game-specific Action 的缺失

Paper 自承认 action space 只覆盖 open-world exploration，缺 shooting、throwing、explosion。

这是真实的 limitation。要支持这些，action space $\mathcal{A}$ 需要扩展，从纯 camera motion 扩展到 general game action。可能的方向：

1. **Discrete action token + continuous camera**：类似 RT-2 的 action token，加几个 binary token (fire, jump, interact) 到 continuous camera trajectory 后面
2. **Object-centric action**：FPS 游戏要瞄准特定 object，action 需要 ground 对象 ID，这就需要 object detection + grounding module
3. **Physics-aware action**：throwing 需要物理 simulation，pure video generation 模型很难学 projectile motion

这块未来工作空间很大，可能是下一篇 paper 的方向。

---

## 这 paper 的真正意义

GameNGen 证明了 "diffusion model 可以做 game engine"。Oasis 证明了 "可以 real-time"。Matrix 证明了 "可以跨 game generalize"。

Hunyuan-GameCraft 在这三件事上各推一步：
- Continuous action 让 control 从 "按键" 升级到 "intent"
- Hybrid history condition 让 long video 从 "几十秒" 推到 "分钟级"
- PCM distillation 让 speed 从 "demo 慢动作" 推到 "近 real-time"

但距离真正 "playable game" 还差：30+ FPS、physical interaction (碰撞/射击)、多 agent、story。这条 path 还长，但 Hunyuan-GameCraft 把一个 realistic engineering baseline 立起来了。

更广泛的，这个 work 提示 "video generation model as world simulator" 这个 path 在 game 场景下已经 mature 到可以谈 productization。下一步可能是 game studio 用这种 model 做 procedural content generation、interactive cutscene、player-driven narrative 等。

参考链接汇总：
- Hunyuan-GameCraft project: https://hunyuan-gamecraft.github.io/
- HunyuanVideo base model: https://arxiv.org/abs/2412.03603
- GameNGen: https://arxiv.org/abs/2408.14837
- Oasis: https://www.decart.ai/articles/oasis-interactive-ai-video-game-model
- Matrix: https://arxiv.org/abs/2412.03568
- Genie 2: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- CameraCtrl: https://arxiv.org/abs/2404.02101
- MotionCtrl: https://arxiv.org/abs/2312.12744
- PCM: https://arxiv.org/abs/2405.18481
- Consistency Model: https://arxiv.org/abs/2303.01469
- Diffusion Forcing: https://arxiv.org/abs/2412.01013
- Monst3R: https://arxiv.org/abs/2410.03825
- RAFT: https://arxiv.org/abs/2003.12039
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- RT-2: https://arxiv.org/abs/2307.15818
- JEPA: https://openreview.net/pdf?id=BvAj8DtLb2

---

# Hunyuan-GameCraft 深度技术解读

## 一、Paper 核心定位

这篇paper 来自 Tencent Hunyuan + HUST, 目标是 build 一个 **高动态 (high-dynamic)、可交互 (interactive)、长时一致 (long-term consistent)** 的 game video generation 模型。它 build on top of HunyuanVideo (一个 MM-DiT text-to-video foundation model), 把它 turn 成一个可以接受 keyboard/mouse 输入的 "playable world model"。

与现有工作的 key differentiator 在 Table 1 中清晰呈现: Hunyuan-GameCraft 是唯一一个同时具备 **Continuous action space + Scene generalizable + Scene dynamic + Scene memory** 的工作。GameNGen [26] 只能跑 DOOM 且 action space 是 discrete key; Oasis [8] 只支持 Minecraft 且 Scene Dynamic 为 ✗; Matrix [10] 有 generalization 但 no scene memory; Genie 2 [22] 有 latent action 但 dynamics 弱。这篇 work 真正想 push 的是 "playability" 这个 axis, 即不仅要 generate video, 还要对 user input 有 low-latency、fine-grained 的 response。

参考链接:
- Project page: https://hunyuan-gamecraft.github.io/
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- GameNGen: https://arxiv.org/abs/2408.14837
- Matrix: https://arxiv.org/abs/2412.03568
- Genie 2: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/

---

## 二、Continuous Action Space 设计 (Sec 4.1)

这是这篇 paper 最 elegant 的 contribution, 也是和 Matrix-Game、Oasis 等用 discrete 7-keys+mouse 的本质区别。

### 2.1 数学定义

公式 (1) 定义 action space $\mathcal{A}$:

$$\mathcal{A} := \left\{ \mathbf{a} = (\mathbf{d}_{\mathrm{trans}}, \mathbf{d}_{\mathrm{rot}}, \alpha, \beta) : \mathbf{d}_{\mathrm{trans}} \in \mathbb{S}^2, \mathbf{d}_{\mathrm{rot}} \in \mathbb{S}^2, \alpha \in [0, v_{\max}], \beta \in [0, \omega_{\max}] \right\}$$

变量含义解析:
- $\mathbf{d}_{\mathrm{trans}} \in \mathbb{S}^2$: translation direction, 是 2-sphere 上的 unit vector, 即三维空间中位移方向的单位向量 (norm=1)。$\mathbb{S}^2 = \{\mathbf{x} \in \mathbb{R}^3 : \|\mathbf{x}\|=1\}$
- $\mathbf{d}_{\mathrm{rot}} \in \mathbb{S}^2$: rotation direction, 同样在 2-sphere 上, 表示绕哪个轴旋转
- $\alpha \in [0, v_{\max}]$: translation speed scalar, bounded by max velocity $v_{\max}$
- $\beta \in [0, \omega_{\max}]$: rotation speed scalar, bounded by max angular velocity $\omega_{\max}$

这个设计的 intuition 是: **把 keyboard/mouse 的离散输入 lift 到一个 continuous 的 6-DoF camera trajectory space**, 然后 discrete input 在这个 space 上变成可插值的 waypoint。比如按 W 键可以 map 到 $\mathbf{d}_{\mathrm{trans}} = (0, 0, 1)$ (forward), 按 A 键 map 到 $\mathbf{d}_{\mathrm{trans}} = (-1, 0, 0)$ (left), 同时按 W+A 就可以自然 interpolate 到 $\mathbf{d}_{\mathrm{trans}} = (-\frac{1}{\sqrt{2}}, 0, \frac{1}{\sqrt{2}})$, 也就是斜向前进。

注意 paper 中特别提到 **eliminate the degree of freedom in the roll dimension**, 即去掉了 roll 自由度 (绕 z 轴的旋转), 这是基于 gaming convention: 玩家 head 一般不会 roll, 只有 pitch (上下看) 和 yaw (左右看), 加上 translation 三个方向, 实际是 5-DoF。这跟 Plücker embedding 的 standard camera representation 是一致的, 但删减了冗余维度。

### 2.2 与 Plücker Embedding 的关系

$\mathcal{A}$ 可以 seamlessly 转换成 standard camera trajectory parameters 和 **Plücker embeddings**。Plücker embedding 是 line geometry 中的经典表示, 一条 3D line $L$ 由起点 $\mathbf{o}$ 和方向 $\mathbf{d}$ 决定, 其 Plücker coordinate 为 $(\mathbf{d}, \mathbf{m})$, 其中 $\mathbf{m} = \mathbf{o} \times \mathbf{d}$ 是 moment。对 camera 而言, 每个像素的 ray 都可以表示成 Plücker coordinate, 拼成一个 $(H \times W \times 6)$ 的 tensor, encode 了完整的 camera intrinsic + extrinsic。

参考 CameraCtrl [13]: https://arxiv.org/abs/2404.02101
参考 MotionCtrl [31]: https://arxiv.org/abs/2312.12744

### 2.3 Lightweight Action Encoder

Architecture 关键点 (Figure 4):

```
[Reference Image] ──patchify──┐
                              ├──token add──→ [MM-DiT Backbone (HunyuanVideo)]
[Keyboard/Mouse] ──→ Action ──┘
                  (continuous camera space)
                  ↓
            [Lightweight Encoder:
             - Conv layers (spatial downsample)
             - Pooling layers (temporal downsample)
             - Learnable scaling coefficient γ]
                  ↓
            Action tokens
```

设计上 **不用 cascaded residual blocks 或 transformer blocks** 来 build Plücker encoder, 只用 conv + pooling, 然后 token-wise add 到 patchified image latent 上。这是 ablation Table 4 (c)(d)(g) 的结论: Token Addition (DA=67.2, RPE trans=0.08) 比 Token Concat (DA=59.7, RPE trans=0.13) 和 Channel-wise Concat (DA=63.2, RPE trans=0.11) 都好。

Intuition: Plücker embedding 已经 spatially & temporally aligned with video latent 了, 所以不需要再 learn 一套 heavy transformation。简单 add 就够, 还避免 over-parameterization 导致的 training instability。Learnable scaling coefficient $\gamma$ 让 model 自己学 action token 相对于 visual token 的 weight。

这个 design philosophy 跟 ControlNet 用 zero-conv 的思路类似: 保证 base model 的 prior 不被破坏。

---

## 三、Hybrid History-Conditioned Long Video Extension (Sec 4.2)

这是 paper 中最值得深入思考的部分, 解决的是 **autoregressive video generation 的根本矛盾**。

### 3.1 三种范式的对比 (Figure 5, Figure 6)

**(i) Training-free inference**: 只用 single image as condition, 每个 chunk 独立 generate。问题: history 信息丢失, 长程 consistency 崩塌 (Figure 6a 的 quality collapse)。

**(ii) Streaming generation**: 用 non-uniform noise window, frame-level rolling。问题: causal VAE [33] 的 encoding 对 initial frame vs subsequent frame 不均匀, 架构 incompatibility。Oasis、Matrix 类工作走这条路。

**(iii) Hybrid history condition (本 paper)**: chunk-wise autoregressive, 但每个 chunk 同时条件化三种 head condition:
- **(i') Single image frame latent** (训练比例 0.25)
- **(ii') Final latent from previous clip** (0.7)
- **(iii') Longer latent clip segment** (0.05)

### 3.2 Mask Indicator 机制

这是实现 hybrid conditioning 的关键 trick。在 denoising 时构造一个 binary mask:

- Mask value **1** = head latent region (clean, noise-free)
- Mask value **0** = chunk segment (noisy, to be denoised)

在 noise schedule 中, head 部分保持 clean latent, chunk 部分 follow flow matching schedule 加噪, 通过 denoising process 让 head 引导 chunk 生成。这就是 concatenate at both condition level 和 noise level, 而 mask 明确告诉 model 哪部分是 history, 哪部分要 predict。

### 3.3 Trade-off 分析 (核心 intuition)

Table 4 (e)(f)(g) 揭示了一个 fundamental tension:

| Condition Type | FVD↓ | DA↑ | RPE trans↓ | RPE rot↓ |
|---|---|---|---|---|
| (e) Image Condition (0.25 ratio) | 1655.3 | 47.6 | 0.07 | 0.22 |
| (f) Clip Condition (0.7 ratio) | 1743.5 | 55.3 | 0.16 | 0.30 |
| (g) Hybrid (Ours) | **1554.2** | **67.2** | **0.08** | **0.20** |

Intuition:
- **Image condition only**: action 响应好 (RPE trans=0.07 低), 因为 model 只看到一个 frame, 没有 strong motion prior 来 "惯性", 所以 new action 可以主导生成。但 dynamic average 只 47.6, 因为 single frame 缺乏 motion context, model 不敢生成 high-motion 内容, 且 quality 在 long rollout 中 collapse (Figure 6a)
- **Clip condition only**: dynamic average 高 (55.3), 视觉质量稳, 但 RPE trans 飙到 0.16, 因为 strong history prior 让 model 倾向 "continue past motion", action 改变时 model 反应迟钝
- **Hybrid**: 兼顾二者。training time 随机 sample 不同 head condition, 让 model 学到 "既能跟随 strong history 保持一致性, 又能响应 single-frame 引导的 action 切换"

这个 ratio (0.7 clip + 0.05 multi-clip + 0.25 single frame) 的选择很关键。0.7 的 clip condition 占主导保证 long rollout 质量, 0.25 的 single-frame 保证 action responsiveness, 0.05 multi-clip 提供 long-range memory 的少量正则化。

### 3.4 与 Diffusion Forcing 的对比

Diffusion Forcing [6] (https://arxiv.org/abs/2412.01013, Boyuan Chen et al.) 也 combine next-token prediction 和 full-sequence diffusion, 但用的是 independent noise level per token, 更接近 classical next-token prediction。

Hunyuan-GameCraft 的 hybrid condition 思路不同: 它在 chunk level 用 full denoising, 但 condition level 是 heterogeneous 的 (image/single-frame/multi-frame)。这更像是 "soft attention to history" 而不是 "hard autoregressive"。

---

## 四、Model Distillation (Sec 4.3)

### 4.1 Phased Consistency Model (PCM)

参考 PCM [28]: https://arxiv.org/abs/2405.18481

PCM 是 Consistency Model [21] 的改进版, 把 diffusion 的 multi-step process distill 成 8-step consistency model。PCM 的核心 idea 是分阶段 (phased) 应用 consistency constraint, 解决 consistency model 在 high-resolution 上的 quality degradation 问题。

### 4.2 CFG Distillation

公式 (2) 是 Classifier-Free Guidance Distillation 的 loss:

$$L_{cfg} = \mathbb{E}_{w \sim p_w, t \sim U[0,1]} \left[ \left\| \hat{u}_\theta(z_t, t, w, T_s) - u_\theta^s(z_t, t, w, T_s) \right\|_2^2 \right]$$

$$\hat{u}_\theta(z_t, t, w, T_s) = (1+w) u_\theta(z_t, t, T_s) - w u_\theta(z_t, t, \cdot)$$

变量解析:
- $z_t$: noisy latent at timestep $t$
- $t \sim U[0,1]$: uniformly sampled timestep
- $w \sim p_w$: guidance scale sampled from training distribution (通常 $w \in [1, 10]$)
- $T_s$: text prompt condition (固定)
- $u_\theta(z_t, t, T_s)$: **conditional** velocity prediction (with prompt)
- $u_\theta(z_t, t, \cdot)$: **unconditional** velocity prediction (empty prompt, 记为 $\varnothing$)
- $\hat{u}_\theta$: standard CFG-guided output, 即 $(1+w) \cdot \text{conditional} - w \cdot \text{unconditional}$
- $u_\theta^s$: **student model** 的 single forward output (no CFG, no double forward)

Intuition: 标准 CFG 需要每个 denoising step 跑 **两次 forward** (conditional + unconditional), 然后线性外推。CFG distillation 训一个 student, 让 student 单次 forward 就直接 output CFG 后的结果。这样 inference 时只需 1 次 forward, 速度直接 2x。Combine PCM 的 8x reduction, 总 speedup 是 ~20x。

### 4.3 实测性能

Table 2 显示:
- **Ours (full)**: 0.25 FPS, FVD 1554.2, DA 67.2
- **Ours + PCM**: **6.6 FPS**, FVD 1883.3, DA 43.8

Speedup 后 FPS 从 0.25 提到 6.6 是 ~26x, 跟 paper 宣称的 10-20x 吻合。代价是 FVD 上升 21%, DA 从 67.2 降到 43.8 (下降 35%)。RPE 保持不变 (0.08/0.20), 说明 action accuracy 没损失, 主要是 visual quality 和 dynamic 稍降。

6.6 FPS 对 game interaction 仍然偏低 (现代游戏至少要 30 FPS), 但 paper 称 "near real-time", 作为 research prototype 已经可用。

---

## 五、Dataset Construction (Sec 3)

这部分是 industrial-grade 工作的标志, 工程细节值得拆解。

### 5.1 Four-Stage Pipeline (Figure 3)

1. **Scene and Action-aware Data Partition**: 用 PySceneDetect [4] 把 2-3 小时 gameplay 切成 6s coherent clip (1M+ clips at 1080p), 然后用 RAFT [24] 计算 optical flow gradient 检测 action boundary。RAFT 是经典 optical flow method, 这里用它 detect "rapid aiming" 这种 motion discontinuity, 保证 6s clip 内 motion 是连续的, 避免训练时 action 和 video 错位。

2. **Data Filtering**: 三层过滤
   - Quality assessment [17] (Kolors team) 去掉 low-fidelity clip
   - OpenCV luminance filtering 去掉 dark scene
   - VLM-based gradient detection (Qwen2-VL [29]) 多角度过滤

3. **Interaction Annotation**: 用 **Monst3R** [35] 重建 6-DoF camera trajectory。Monst3R 是 2024 年 CVPR 的 work, 能从 monocular video 估计 geometry + motion。这里它替代传统 SLAM, 因为 game video 经常 has non-rigid scene motion, SLAM 会 fail。

4. **Structured Captioning**: 用 game-specific VLM 生成两层 caption (30 char summary + 100+ char detail), 训练时 random sample。这类似 InstructPix2Pix 的做法, 让 model 对 caption granularity 鲁棒。

### 5.2 Synthetic Data

额外 render 了 ~3000 个 high-quality motion sequence 从 curated 3D assets, 多个 starting position × varying speed。这部分数据的关键作用在 ablation Table 4(a)(b):

| Training Data | FVD↓ | DA↑ | RPE trans↓ |
|---|---|---|---|
| (a) Only Synthetic | 2550.7 | 34.6 | 0.07 |
| (b) Only Live | 1937.7 | 77.2 | 0.16 |
| (g) Hybrid (Render:Live=1:5) | **1554.2** | **67.2** | **0.08** |

Intuition: Synthetic data 的 motion 是精确已知的, 所以 action control 好 (RPE trans=0.07), 但 synthetic scene 缺 dynamic objects (没 NPC, 没 particle effect), 所以 dynamic average 只有 34.6。Live data 相反。Hybrid 1:5 比例下两者优势叠加, 既保住了 dynamic (67.2), 又保住了 control accuracy (0.08)。这个 ratio 是 paper 经过 ablation 得出的 sweet spot。

### 5.3 Distribution Balancing

针对 **forward-motion bias** (game video 绝大多数时间在 forward 走) 两个策略:
1. **Stratified sampling of start-end vectors**: 在 3D 空间均匀采样方向
2. **Temporal inversion augmentation**: 把 video 倒放, backward motion 数据翻倍

这个 trick 的效果体现在 cross-domain RPE trans 从 Matrix-Game 的 0.18 降到 Ours 的 0.08, 减少 55%。倒放 augmentation 在 driving dataset 上也很常见, 这里用在 game 上很合理。

参考 Monst3R: https://arxiv.org/abs/2410.03825
参考 RAFT: https://arxiv.org/abs/2003.12039
参考 Qwen2-VL: https://arxiv.org/abs/2409.12191

---

## 六、实验结果深度分析

### 6.1 Quantitative Comparison (Table 2)

| Model | FVD↓ | IQ↑ | DA↑ | Aesthetic↑ | TC↑ | RPE_t↓ | RPE_r↓ | FPS↑ |
|---|---|---|---|---|---|---|---|---|
| CameraCtrl | 1580.9 | 0.66 | 7.2 | 0.64 | 0.92 | 0.13 | 0.25 | 1.75 |
| MotionCtrl | 1902.0 | 0.68 | 7.8 | 0.48 | 0.94 | 0.17 | 0.32 | 0.67 |
| WanX-Cam | 1677.6 | 0.70 | 17.8 | 0.67 | 0.92 | 0.16 | 0.36 | 0.13 |
| Matrix-Game | 2260.7 | 0.72 | 31.7 | 0.65 | 0.94 | 0.18 | 0.35 | 0.06 |
| **Ours** | **1554.2** | 0.69 | **67.2** | 0.67 | **0.95** | **0.08** | **0.20** | 0.25 |
| Ours+PCM | 1883.3 | 0.67 | 43.8 | 0.65 | 0.93 | 0.08 | 0.20 | **6.6** |

几个观察:
1. **Dynamic Average (67.2) 远超 baseline**: Matrix-Game 31.7, CameraCtrl 7.2。这说明 Hunyuan-GameCraft 生成的 video 中物体运动幅度远大于其他, 这是 high-dynamic 的核心指标。Matrix-Game 弱 dynamic 是因为它的 hybrid history condition 偏向 stability
2. **RPE trans 0.08 vs Matrix-Game 0.18**: 减少 55% action error, 这是 continuous action space + synthetic data 的功劳
3. **FVD 1554.2 best**: 视觉质量也领先, 说明没有 trade quality for control
4. **IQ 0.69 略低于 Matrix-Game 0.72**: 单帧 image quality 上没赢, 因为 game video 本身帧间 dynamic 大, 单帧 quality 不一定最高。但这是 acceptable trade-off
5. **FPS 6.6 with PCM**: 远超所有 baseline。Matrix-Game 0.06 FPS 几乎不可玩, WanX-Cam 0.13 同样不可用

### 6.2 User Study (Table 3)

5 个 dimension (Video Quality, Temporal Consistency, Motion Smooth, Action Accuracy, Dynamic), Hunyuan-GameCraft 全部 4.4+ 排名, 其他 baseline 最好 3.23 (MotionCtrl on TC)。Human preference 跟 quantitative metric 一致, 说明 metric 设计合理。

### 6.3 Ablation Summary (Table 4)

三个维度的 ablation 结论:
1. **Data**: Hybrid (Render:Live=1:5) 最优, single-source 都有短板
2. **Control Injection**: Token Addition > Token Concat > Channel-wise Concat (efficiency + accuracy 双赢)
3. **History Conditioning**: Hybrid > Single-condition, 解决 quality-vs-action trade-off

---

## 七、Generalization to Real World (Sec 6)

Figure 10 展示 Hunyuan-GameCraft 在 real-world image 上也能 generate reasonable interactive video, 保留 dynamics。这归功于 base model HunyuanVideo 的强大 prior。这点对 productization 很重要: 同一个 model 既能做 game scene, 又能做 real-world scene 的 camera-controlled video, 节省工程成本。

这跟 Genie 2 [22] 的 "single image → 3D world" 思路呼应, 但 Genie 2 用 latent action (无 explicit input mapping), 而 Hunyuan-GameCraft 用 explicit continuous camera action, controllability 更强。

---

## 八、Limitations 与未来方向

Paper 自承认: action space 主要针对 open-world exploration, 缺 shooting、throwing、explosion 这种 game-specific action。这指向一个更大的研究问题: **如何把 action space 从 camera control 扩展到 general game action**?

可能的 future direction:
1. **Discrete+Continuous Hybrid Action**: 把 continuous camera trajectory + discrete action token (如 "fire", "jump", "interact") 结合, 类似 RT-2 在 robotics 中的 action tokenization
2. **Physical Interaction**: 当前是 "video generation" 层面, 没有 physical engine。未来可能需要 integrate differentiable physics simulator (如 Genesis, Brax) 做 hybrid rendering
3. **Multi-agent**: 当前 single-player perspective, multi-agent game (如 MOBA) 需要不同 action representation
4. **Real-time 30+ FPS**: 当前 6.6 FPS 距离 game-ready 30 FPS 还有 5x gap, 可能需要更好的 distillation (如 LCM, DMD2) 或 dedicated inference hardware

参考 RT-2: https://arxiv.org/abs/2307.15818
参考 DMD2: https://arxiv.org/abs/2405.14867

---

## 九、Karpathy 视角的几点思考

### 9.1 为什么 Continuous Action Space 重要

Discrete keyboard (WASD+arrows+space) 听起来是 standard game input, 但实际游戏中 action 是 continuous 的: 玩家用 joystick 摇杆, 鼠标移动有速度, 键盘按住时长不同。Discrete representation 把这些信息全丢了, 导致 model 学到的是 "binary motion", 没法 interpolate, 没法 generalize 到 fine-grained control。把 input lift 到 continuous camera trajectory space, 本质上是把 "键位" 这个 interface 升级成 "intent" 这个 interface, 这跟 LLM 把 "command" 升级成 "natural language" 是同一类 abstraction 提升。

### 9.2 Hybrid History Condition 的深层意义

这个 design pattern 不仅适用于 video generation, 在 LLM agent、world model、robotic control 中都有对应:
- LLM agent 的 in-context learning = "image condition" (只有当前 input, no history)
- LLM agent 的 long-context = "multi-clip condition" (full history)
- LLM agent 的 sliding window = "clip condition"

Hunyuan-GameCraft 的 hybrid training 实际上是在 train model 学到 "什么时候 rely on short-term context, 什么时候 rely on long-term context", 这跟 attention mechanism 的本质一致, 但在 diffusion model 中用 conditioning ratio 实现, 而不是 attention weight。

### 9.3 Causal VAE 的限制

Paper 多次提到 causal VAE [33] 导致 streaming generation 不 work。Causal VAE 是为了 real-time encoding 设计的, 但它的 temporal compression 是 non-uniform 的, initial frame 保留更多信息, subsequent frame 压缩更狠。这对 single-frame conditioning OK, 但对 chunk-wise conditioning 就 inconsistent。Hunyuan-GameCraft 选择 **chunk-level 全 encode**, 避免 streaming, 用 PCM distillation 补偿 efficiency。这是 engineering trade-off 的典型 case: 选 consistent 但 slow 的方案, 再用 distillation 加速。

参考 CausalVAE: https://arxiv.org/abs/2004.11186

### 9.4 与 World Model 的关系

这篇 work 是 "video generation as world model" 的 representative。它没 explicitly 学 physics, 但 implicit 在 pixel-level generation 中。Hunyuan-GameCraft 的 "scene memory" 实际是 visual memory, 不是 symbolic state。这跟 LeCun 的 JEPA 思路相反: JEPA 学 abstract representation, video model 学 pixel-level representation。两种 approach 各有 trade-off:
- Pixel-level (Hunyuan-GameCraft): visualization ready, 但 compute expensive, long rollout 会 drift
- Abstract (JEPA): compute efficient, 但需要 decoder 才能 visualize, action grounding 难

Game 这种 visual-heavy task, pixel-level approach 更直接, 这也是为什么 GameNGen、Oasis、Matrix、GameCraft 都走这条路。

参考 JEPA: https://openreview.net/pdf?id=BvAj8DtLb2

---

## 十、总结

Hunyuan-GameCraft 在三件事上做了扎实的工作:

1. **Continuous Action Space**: 数学定义清晰, 实现简洁 (conv + pooling), 性能领先 (DA 67.2 vs Matrix-Game 31.7)
2. **Hybrid History Condition**: 解决了 long video generation 中 quality vs action responsiveness 的根本矛盾, 用 conditioning ratio (0.7/0.05/0.25) 平衡
3. **Distillation**: PCM + CFG distillation, 把 0.25 FPS 推到 6.6 FPS, 接近 real-time 边缘

这套组合拳让 "playable video game model" 从 demo 阶段向 product 阶段推进了一步。但距离真正 30+ FPS、physical interaction、multi-agent game 还需要更多 iteration。

这 paper 也是 industrial lab + academic collaboration 的典型: HUST 团队做 algorithmic innovation, Tencent 提供 compute (192× H20 GPU)、data (100+ AAA games)、engineering (distillation pipeline), 这种 combination 在前沿 AI research 中越来越必要。

相关参考链接汇总:
- Hunyuan-GameCraft: https://hunyuan-gamecraft.github.io/
- HunyuanVideo (base): https://arxiv.org/abs/2412.03603
- Matrix-Game (baseline): https://arxiv.org/abs/2506.01790
- Oasis: https://www.decart.ai/articles/oasis-interactive-ai-video-game-model
- GameNGen: https://arxiv.org/abs/2408.14837
- Genie 2: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- CameraCtrl: https://arxiv.org/abs/2404.02101
- MotionCtrl: https://arxiv.org/abs/2312.12744
- PCM (Phased Consistency Model): https://arxiv.org/abs/2405.18481
- Consistency Models: https://arxiv.org/abs/2303.01469
- Diffusion Forcing: https://arxiv.org/abs/2412.01013
- Monst3R: https://arxiv.org/abs/2410.03825
- RAFT: https://arxiv.org/abs/2003.12039
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- CausalVAE: https://arxiv.org/abs/2004.11186
- JEPA: https://openreview.net/pdf?id=BvAj8DtLb2
