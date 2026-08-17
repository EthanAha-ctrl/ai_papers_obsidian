---
source_pdf: WorldScore A Unified Evaluation Benchmark for World Generation.pdf
paper_sha256: 7ddf259570f4b302f44650329fd24ab331a19e218adf538b47019218fb235d06
processed_at: '2026-08-13T05:53:37-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话聊聊 WorldScore 这篇 paper

## 一句话总结

Stanford 的一帮人觉得现在大家都在喊 "world generation" 这个词 — Sora 说自己是 world simulator, WonderWorld 说是 3D world generation, CogVideoX 也在谈 world — 但没人能 quantitatively 说清楚到底谁更强、强在哪。于是他们造了个 ruler, 把所有这些 heterogeneous 的 model 拉到同一个坐标系里量了一下。

---

## 1. 为什么这篇 paper 存在 — 真实痛点

想象你是 Sora 的 PM, 你的老板问你: "我们 Sora 和 WonderWorld 比, 谁更会 generate world?"

你根本答不上来。

因为 Sora 输出的是一段 video, WonderWorld 输出的是一组 3D mesh / Gaussian Splatting scene, CogVideoX-I2V 又是另一回事。它们连 input format 都不一样 — WonderWorld 吃 image + camera matrix, Sora 只吃 text。你怎么比? VBench 只测 text-to-video 单 scene 质量, 根本不 care camera 走得对不对, 也不 care 你能不能从卧室走出客厅。

所以这篇 paper 的存在动机非常 simple: **造一把大家都认的尺子, 把 3D scene generation、4D generation、I2V、T2V 这四类 method 拉到同一个 testbed 上跑一遍, 给出可比的 score。**

参考: 这个 motivation 在 paper Section 1 写得很直白, Figure 1 给了一个特别好懂的例子 — VBench 觉得 Model A 和 Model B quality 差不多, 但 WorldScore 发现 Model B 根本没 generate 新 scene, 也没 follow camera 指令。这就是 "single-scene benchmark 的 blind spot"。

---

## 2. 核心设计 idea — Task decomposition

这个 idea 看起来很 academic, 但其实很接地气。

**人怎么在脑子里想象一个 "world"?** 你不是一次性想象整个 planet, 你是想象 "我现在在客厅, 往前走两步进了厨房, 左拐到了花园"。这就是一个 sequence of next-scene predictions。

WorldScore 就这么定义 world generation:

每一步给你三个东西:
- $\mathcal{C}$: 当前 scene — 一张图 + 一段 text 描述
- $\mathcal{N}$: 下一个 scene 应该有什么 — text 描述 (e.g. "下一个 scene 是花园, 有一棵大树")
- $\mathcal{L}$: layout — camera 怎么动, 是 textual description (e.g. "camera 往前推") + 一个 ground truth 的 camera matrix trajectory

然后 model 不管是什么 paradigm, 都被要求 output 一段 video。3D model 就 render 成 video, video model 直接 generate video, 4D model 也 output 成 video。然后大家都在 video 这个 format 上比。

**妙处在哪?**

妙在 $w_{\text{proc}}$ 这个 adapter 函数。WonderJourney 原生吃 camera matrix, 你直接喂 $\mathcal{T}$。CogVideoX 不懂 camera matrix, $w_{\text{proc}}$ 就把 "camera moves left" 这段 text 拼到 prompt 后面。**给什么吃什么, 把异构 model 拉平到同一 spec。**

参考: Section 3.1 公式 1 + Supp. A 的 $w_{\text{proc}}$ 详解。这个 idea 看起来简单, 但实现起来有很多 corner case, 比如 T2V model 没 image input, 他们就把 T2V 当成 "忽略 image 的 I2V"。

---

## 3. Dataset — 3000 个 test cases 怎么造的

人话讲就是:

**Static world 部分 (2000 个 case)**:
- 10 类场景: 室内 5 类 (餐厅、客厅、走廊、公共空间、办公), 室外 5 类 (城市、郊区、水景、陆地、绿地)
- 每类 100 张 photorealistic 图作为 starting image, 再用 Recraft 生成 7 种 art style 的 stylized 版本 (anime、cyberpunk、水墨、浮世绘、印象派、后印象派、minecraft)
- 然后用 GPT-4o auto-regressive 生成 next-scene text prompt — 让 LLM 想象 "你正在这间客厅里, 下一步你会看到什么 scene?"
- 80% 是 small world (2 个 scene), 20% 是 large world (4 个 scene)
- 给每个 case 随机配一个 camera movement (8 种, 电影术语里的 pan、tilt、dolly、tracking 等)

**Dynamic world 部分 (1000 个 case)**:
- 5 类 motion: articulated (关节运动, e.g. 走路)、deformable (软体)、fluid (流体)、rigid (刚体)、multi-motion
- 这里 camera 是固定的 (重点测 object 动没动, 不是 camera 动没动)
- 同样配 photorealistic 和 stylized 版本

**Image filtering 很关键** (Supp. B.1): 他们从 9 个 dataset 里 source 图 (Matterport3D、Hypersim、SUN-RGBD、DIODE、ETH3D、LHQ、EDEN、Argoverse-HD、InterriorVerse), 然后用了 5 层 filter:
1. CLIP-IQA + CLIP Aesthetic 去低质量
2. Perspective Fields 去极端视角和窄 FOV (这步特别重要, 因为窄 FOV 起步图会让 world generation 不自然)
3. CLIPSIM 去相邻帧冗余
4. 亮度阈值
5. 人工 review

最后每类只留 top 100。**这种级别的 rigor 在 benchmark paper 里算罕见的。**

---

## 4. 10 个 metrics — 论文的 meat

把 10 个 metric 分三类讲, 每个都给 intuition。

### 4.1 Controllability — 你让它干啥它干了吗?

**(a) Camera controllability**

公式 3:
$$e_{\text{camera}} = \sqrt{e_\theta \cdot e_t}$$

直觉: rotation error 和 translation error 的 geometric mean。geometric mean 而不是 arithmetic mean 是因为 — 如果 rotation 完全错, translation 再准也没用, 几何平均会狠狠惩罚这种情况。

具体公式 (S1, S2):
- $e_\theta = \arccos\left(\frac{\text{tr}(\mathbf{R}_{\text{gt}} \mathbf{R}^T) - 1}{2}\right) \cdot \frac{180}{\pi}$ — 这是标准的 rotation angular distance, 用 trace 反解 angle
- $e_t = \|\mathbf{t}_{\text{gt}} - s\mathbf{t}\|_2$ — translation error, 但有个 least-square scale $s$ 消除 video model 没有 absolute scale 的问题

实现: 用 DROID-SLAM (https://arxiv.org/abs/2104.00080) 跑 generated video, 估出 frame-wise camera pose, 和 ground truth trajectory 对。

**为什么这事难做对?** Video model 输出 video 没有显式的 camera pose, 你只能从 video 反推 camera motion。而 DROID-SLAM 在 generated (potentially OOD) video 上估 pose 本身就有误差。但这个误差是 systematic 的, 各个 model 都受影响, 所以 ranking 还是有意义。

**(b) Object controllability**

简单粗暴: 用 Grounding DINO (https://arxiv.org/abs/2303.05499) 跑 open-set detection, 看 prompt 里提到的 "yellow armchair"、"bookshelf" 在 generated frame 里 detect 出来没有。算 success rate。

**潜在问题**: Grounding DINO 在 anime、ukiyo-e 这些 stylized 图上 detection accuracy 可能掉得厉害。这点 paper 没 ablate, 算一个 limitation。

**(c) Content alignment**

CLIPScore (https://arxiv.org/abs/2104.08718) — next-scene prompt 和 generated frame 之间的 CLIP 相似度。

Object controllability 只看 prompt 里 object 那部分 (大概 1/4 prompt length), content alignment 看整个 prompt。互补。

### 4.2 Quality — 生成的 world 质量怎么样?

**(d) 3D Consistency**

公式 S3:
$$e_{\text{reproj}} = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \|\mathbf{p}_{ij}^* - \Pi(\mathbf{P}_{ij})\|_2$$

人话: 把 generated video 喂 DROID-SLAM, 让它估 dense per-pixel depth + camera pose, 然后做 dense bundle adjustment, 再算 reprojection error — 把重建的 3D 点 project 回 image plane, 看和原 pixel 对得上对不上。

**为什么用 DROID-SLAM 不用 COLMAP?** COLMAP 是 sparse 的, 只用 "good" feature matches, 丢掉大部分 pixel。DROID-SLAM 是 dense 的, 用所有 pixel, 对 appearance 变化 robust (它不要求 texture 一致, 只关心几何一致)。这正好契合我们想要的 — 只 isolate geometry inconsistency, 不要 photometric noise 干扰。

**Intuition**: 如果 video 里一个 wall 突然 warp, 那几何上 inconsistent, reprojection error 大。如果只是 grass 颜色从一帧到另一帧抖动, 几何上还是 ok 的。

**(e) Photometric Consistency — 这个 metric 设计得最 clever**

公式 S4-S6:
- $\mathbf{p}_B = \mathbf{p}_A + \mathcal{F}_{A \to B}(\mathbf{p}_A)$ — 在 frame A 中心采样点, 用 forward optical flow track 到 frame B
- $\mathbf{p}_A' = \mathbf{p}_B + \mathcal{F}_{B \to A}(\mathbf{p}_B)$ — 再用 backward flow track 回 frame A
- $e_{\text{photometric}} = \frac{1}{N} \sum_i \|\mathbf{p}_{A,i} - \mathbf{p}_{A,i}'\|_2$ — AEPE (Average End-Point Error)

**为什么要 forward-backward consistency?** 

CLIP / DINO feature-based consistency (VBench 用的) 有一个 blind spot: 它们 capture categorical identity (这帧是 mountain, 那帧也是 mountain, score 高), 但 capture 不了 fine-grained texture flickering。Figure 6 中间那个例子特别好: mountain 还是 mountain, CLIP feature 没变, 但 grass texture 在帧之间 shift、distort、warp, 这就是典型的 photometric artifact。

Forward-backward flow consistency 能 capture 这个 — 如果 texture 在帧间稳定, 你 track 过去再 track 回来应该回到原位 (AEPE 接近 0)。如果 texture 在抖动 / shift, AEPE 大。

**这个 metric 单独是这篇 paper 的一个 contribution 级别的设计。**

**(f) Style Consistency**

用 Gatys 的 Gram matrix (https://arxiv.org/abs/1508.06576) 算 first frame 和 last frame 之间的 F-norm difference。

简单但有效 — 如果 model 在 generation 过程中 style drift (从 photorealistic 慢慢变 cartoonish), Gram matrix 就变。

**Limitation**: 只看 first 和 last, miss 中间 drift。可以改成 sliding window。

**(g) Subjective Quality**

不是 single metric, 是 **CLIP-IQA+ 和 CLIP Aesthetic 的 arithmetic mean**。这个组合是通过 400 人 human study 选的 (Table S5) — 12 个 candidate metric / combination 里, 这个组合 agreement score 0.637 最高, 接近 upper bound 0.772。

**Intuition**: 主观质量这玩意儿没法用 single metric 精确 capture, 但通过 human study 校准, 找到一个 ensemble 还是可以做的。

### 4.3 Dynamics — dynamic world 测三件事

**(h) Motion Accuracy**

公式 S7:
$$s_{\text{motion-acc}} = \max(\mathbf{F} \odot \mathbf{M}) - \max(\mathbf{F} \odot \bar{\mathbf{M}})$$

- $\mathbf{F}$: optical flow magnitude (连续两帧间)
- $\mathbf{M}$: dynamic object 的 mask (dataset annotation 提供, 用 SAM2 propagate)
- $\bar{\mathbf{M}}$: 非动态区域
- $\odot$: element-wise 乘

直觉: 看 mask 内的 motion 大还是 mask 外的 motion 大。理想情况 mask 内 motion 大、mask 外 motion 小, score 高。如果反过来 (背景动了主体没动), score 低甚至为负。

Figure 6 例子: octopus 应该动, jellyfish 应该静。Bad case 是 jellyfish 动了 octopus 没动 — 这就是 motion placement 错误, motion accuracy 检测这种 failure mode。

**Potential issue**: 用 max operator 很不 robust, 一个 outlier pixel 就 dominate。改成 90th percentile 或者 mean 会 robust 很多。这是 paper 一个 minor weakness。

**(i) Motion Magnitude**

optical flow magnitude 的 median value。用 median 不用 mean 是为了 robust to outlier。

直觉: 测 model 能不能产生 large motion。有些 model 太 conservative, 输出几乎 static video, motion magnitude 低。这种 model 看起来 smooth 但 boring。

**(j) Motion Smoothness**

用 VFI-Mamba (video frame interpolation, https://arxiv.org/abs/2407.02315) 做 reference。流程: 丢掉奇数帧, 让 VFI model 重建, 然后和原帧比 (MSE + SSIM + LPIPS)。

直觉: 如果 motion smooth, VFI 能从偶数帧 interpolate 出奇数帧, 误差小。如果 motion jittery, VFI 重建不准, 误差大。

---

## 5. Score normalization 和 aggregation

每个 metric 的 raw score 都 normalize 到 [0, 1] 然后乘 100。

公式 S8:
$$s^{\text{norm}} = \begin{cases} \left\langle\frac{s - b^{\min}}{b^{\max} - b^{\min}}\right\rangle, & \text{higher better} \\ \left\langle 1 - \frac{s - b^{\min}}{b^{\max} - b^{\min}}\right\rangle, & \text{lower better} \end{cases}$$

Empirical bounds 怎么定? Supp. C.8 详述:
- Camera controllability: lower bound = 0, upper bound = "fixed camera 序列" (惩罚不动 camera 的)
- Object controllability: 0% - 100% (theoretical bound)
- 3D / photometric / style consistency: upper bound = "video frame interpolation baseline" (用 VFI 生成 inconsistent 视频做最差 baseline)
- Motion smoothness: 用 OpenVid-1M 的真实视频作 reference
- Content alignment / subjective quality / motion accuracy / magnitude: z-score rescaling 让 selected models 落在 25-75 之间 (pragmatic)

最后:
- **WorldScore-Static** = 3 controllability + 4 quality 的 arithmetic mean
- **WorldScore-Dynamic** = 上面 + 3 dynamics metric 的 arithmetic mean

3D models 不支持 dynamics, dynamics metric 全 assign 0。这就是 Table 2 里 3D models 一行最后三列都是 0.00 的原因。

---

## 6. 20 个 model 跑出来的结果 — 有什么 surprising 的发现?

Table 2 重排 (按 WorldScore-Static):

| Rank | Model | Type | WS-Static | WS-Dynamic | Cam Ctrl | 3D Cons | Photo Cons | Motion Mag |
|---|---|---|---|---|---|---|---|---|
| 1 | WonderWorld | 3D | 72.69 | 50.88 | 92.98 | 86.87 | 85.56 | 0.00 |
| 2 | LucidDreamer | 3D | 70.40 | 49.28 | 88.93 | 90.37 | 90.20 | 0.00 |
| 3 | InvisibleStitch | 3D | 63.75 | 42.78 | 93.20 | 88.51 | 89.19 | 0.00 |
| 4 | Text2Room | 3D | 62.10 | 43.47 | 94.01 | 88.71 | 88.36 | 0.00 |
| 5 | CogVideoX-I2V | I2V (open) | 62.15 | 59.12 | 38.27 | 86.21 | 88.12 | 26.42 |
| 6 | Gen-3 | I2V (closed) | 60.71 | 57.58 | 29.47 | 68.31 | 87.09 | 27.48 |
| 7 | WonderJourney | 3D | 61.12 | 44.63 | 84.60 | 80.60 | 79.03 | 0.00 |
| 8 | Hailuo | I2V (closed) | 57.55 | 56.36 | 22.39 | 67.18 | 62.82 | 27.20 |
| 9 | LTX-Video | I2V | 55.44 | 56.54 | 26.55 | 78.41 | 88.92 | 29.95 |
| 10 | Allegro | I2V | 55.31 | 51.97 | 26.72 | 67.29 | 47.35 | 40.28 |
| 11 | CogVideoX-T2V | T2V | 54.18 | 48.79 | 40.22 | 68.81 | 64.20 | 47.31 |
| ... | ... | | | | | | | |
| 20 | 4D-fy | 4D | 27.98 | 32.10 | 69.92 | 35.47 | 1.59 | 22.88 |

### 发现 1: 3D models 在 static world generation 上吊打 video models

Top 4 全是 3D models (WonderWorld 72.69, LucidDreamer 70.40, InvisibleStitch 63.75, Text2Room 62.10)。最好的 video model CogVideoX-I2V 才 62.15。

原因其实 obvious 但 paper quantitatively 证明了:
- 3D model 的 camera controllability 是 84-94 (因为它们吃 camera matrix, 当然准)
- 3D consistency 高 (本来就是 3D representation)
- Photometric consistency 高 (texture baked in)

**但 3D models 的 weakness**: dynamics 全是 0。它们天生不会动。

### 发现 2: Video models 的最大瓶颈是 camera controllability

最好的 video model (CogVideoX-T2V) 也才 40.22。Gen-3 才 29.47, Hailuo 才 22.39。

这是 video model 范式的一个 fundamental problem: 它们从 text 学 camera motion, 没显式 camera conditioning, 所以 instruction "camera moves left" 它们经常理解错、根本不动、或者瞎动。

**未来方向**: CameraCtrl (https://arxiv.org/abs/2404.02101)、MotionCtrl (https://arxiv.org/abs/2312.06021) 这类显式 camera conditioning 注入 work 是关键 direction。

### 发现 3: 最好的开源 video model 已经追上甚至超过闭源

CogVideoX-I2V (62.15) > Gen-3 (60.71) > Hailuo (57.55)。

这是开源社区一个重要 milestone。但细看发现 CogVideoX 在 camera control 上强 (38.27), 在 object control 和 content alignment 上弱。Gen-3 / Hailuo 反过来。各有所长。

### 发现 4: 4D-fy 表现惨淡 (27.98)

4D generation 还在蛮荒期。Photometric consistency 1.59 — 几乎没有。原因: 4D-fy 是 object-level 4D 生成方法, scene-level 4D 生成还没有像样的 model。

### 发现 5: T2V vs I2V 的有趣 trade-off

CogVideoX-T2V vs CogVideoX-I2V (同 architecture):
- T2V: cam ctrl 40.22 vs I2V 38.27 — T2V 更高 (这反直觉!)
- T2V: motion mag 47.31 vs I2V 26.42 — T2V 大得多
- T2V: subject qual 44.67 vs I2V 62.44 — I2V 高

**Intuition**: I2V 被 reference image anchor 死了, 不敢动 camera, 不敢 generate 大 motion, 但单帧 quality 更高。T2V 没 anchor, 更愿意 explore 大 motion, 但单帧 quality 不如 I2V (因为没 image reference 引导)。

### 发现 6: Motion smoothness vs magnitude trade-off

Table 2 一看就发现: motion magnitude 大的 model, motion smoothness 通常低。e.g., Allegro motion mag 40.28 但 smoothness 37.81; T2V-Turbo motion mag 75.00 (夸张) 但 smoothness 18.87 (差)。

这是 video model 的 fundamental challenge — 既要大动作又要 smooth。

### 发现 7: Video models 在 outdoor 和 long sequence 上明显弱

Figure 7 subdomain 分析:
- Indoor: video models 和 3D models gap 较小
- Outdoor: video models 显著弱于 3D models
- Long sequence (large world): video models struggle significantly

**Intuition**: Indoor 场景相对 structured, video model 见得多。Outdoor 太 unstructured (天空、地形、远景), video model 训练数据里 camera motion 模式可能和 3D model 用的 fly-through 不一样。Long sequence 时 video model 多次 autoregressive rollout, error 累积。

---

## 7. Human study — 验证 metrics 真的 align with human

Supp. D 做了 400 人的 human study。

设计了一个 probabilistic agreement score: 给参与者 video pair (A, B), 强制 2AFC 选 quality 更高的。设选 A 的比例 $p$:
- 若 metric 给 A 高分, agreement = $p$
- 若 metric 给 B 高分, agreement = $1-p$
- 若 metric 平局, agreement = 0.5
- 然后 average over all pairs

结果:
- Subjective quality: CLIP-IQA+ & CLIP Aesthetic 组合 0.637 agreement, upper bound 0.772
- 其他 metrics 在 2AFC test (score 差 30): 71.2%-97.3% agreement (Table S6)

还有 resolution robustness test: 把 EasyAnimate 输出从 1344×768 resize 到 256×256, 重跑 metric, 所有 metric 差 ≤ 0.83 (Table S7)。说明 metric 对 resolution/aspect ratio 变化 robust。这重要, 因为 20 个 model 的 output resolution 跨度很大。

---

## 8. 我觉得 paper 的 limitation

让我开诚布公说说我觉得 paper 哪里可以更好:

### (1) Camera controllability 的 scale 处理

$e_t = \|\mathbf{t}_{\text{gt}} - s\mathbf{t}\|_2$ 用 least-square scale $s$ 消除 scale ambiguity。但这个 $s$ 对 stationary model 会变得很小, $e_t$ 就接近 $\|\mathbf{t}_{\text{gt}}\|_2$, 惩罚很重。问题是有些场景 fixed camera 是 valid 选择 (e.g., 你让 model "look around" 但 model 选择固定 camera 让 viewer 自己 explore), 这个设计可能太一刀切。

### (2) 4D 只有一个 model

4D-fy 表现糟糕 (27.98), 但只有 1 个 4D model 不够 form ranking。需要等 VividDream、Comp4D、4Real 等 4D scene-level method 开源后再测。

### (3) Style consistency 只看第一帧和最后一帧

这会 miss 中间 style drift。应该用 sliding window 或者 all-pairs。简单但 effective 的改进点。

### (4) Motion accuracy 的 max operator

$\max(\mathbf{F} \odot \mathbf{M})$ 太不 robust, 一个 outlier pixel 就 dominate。改成 90th percentile 或 mean 更鲁棒。

### (5) Subjective quality 是 image-level

逐帧评估 CLIP-IQA / Aesthetic, 没 explicit temporal aesthetic。一个 video 可能每帧都好看但 transition 难看, 这个 metric 看不出来。

### (6) Long sequence 占比小

只有 20% 是 large world (4 个 scene)。Paper 说 video models 在 long sequence 上 struggle, 但 sample size 小。建议未来版本提到 50%。

### (7) Stylized 场景的 detection 可靠性

Grounding DINO 在 anime、ukiyo-e 上 detection 准不准? Paper 没 ablate。如果 Grounding DINO 在 stylized 场景上掉链子, object controllability metric 不可靠。

### (8) 没有 physics evaluation

World generation 真正的 essence 之一是 physical commonsense (gravity、collision、object permanence)。当前 WorldScore 没显式 physics metric。WorldModelBench (https://arxiv.org/abs/2502.20694) 有这个思路, 可以借鉴。

### (9) Open-loop evaluation

当前 spec 是预先确定的, model 一次性 generate。未来应该 evaluate closed-loop, 即 model 根据 agent action 生成下一帧 — 这是 DIAMOND、Genie、Sora "world simulator" 真正的含义。

---

## 9. 这篇 paper 在大图景里的位置

### 9.1 Sora 开启的 "world simulator" narrative

Sora (https://openai.com/research/video-generation-models-as-world-simulators) 提出 "video generation models as world simulators", 但 OpenAI 没给 quantitative eval。WorldScore 填补这个 gap。论文 [6] 显示作者明确知道 Sora 开启的 paradigm。

### 9.2 与 Cosmos 的互补

Cosmos (https://arxiv.org/abs/2501.03575, NVIDIA) 同期出现, 是 "world foundation model platform for physical AI"。Cosmos 做 training, WorldScore 做 evaluation, 二者天然互补。

### 9.3 Stanford 团队的纵向布局

WonderJourney (https://arxiv.org/abs/2312.03884) 和 WonderWorld (https://arxiv.org/abs/2406.18930) 是 Hong-Xing Yu (本论文共一) 的工作, 在 Table 2 排第 1 和第 7。说明 Stanford 团队既做 model 又做 benchmark, 横向布局。

### 9.4 SLAM、optical flow、detection 等 "古典 CV" 技术 "反哺" video generation eval

DROID-SLAM、SEA-RAFT、Grounding DINO、SAM2 都被拿来 eval video generation。这是 classical CV 在 generative AI 时代的新角色 — 不是做 task, 是做 evaluation infrastructure。

### 9.5 与 embodied AI 的桥梁

World generation 的终极 customer 之一是 embodied AI 的 simulator。如果 agent 可以在 generated world 里 interact、collect data、学 policy, 那 world generation 就从 "content creation" 变成 "training infrastructure"。WorldScore 当前 open-loop, 离 embodied AI 真正需要的 closed-loop simulator 还差一步。

---

## 10. 用人话总结

**这篇 paper 做了什么?**

造了一把尺子, 量了 20 个 "world generation" model, 发现:
- 3D models (WonderWorld、LucidDreamer) 在 static world 上完胜 video models, 因为它们天生会 camera control + 3D consistency
- Video models (CogVideoX-I2V、Gen-3、Hailuo) 在 dynamic world 上有优势, 但 camera control 是它们最大的瓶颈
- 4D-fy (唯一的 4D model) 还在蛮荒期
- CogVideoX-I2V (开源) 已经追上闭源 (Gen-3、Hailuo), 但各有 trade-off
- Video models 在 outdoor 和 long sequence 上明显弱

**这把尺子怎么造的?**

把 "world generation" 拆成 next-scene prediction sequence, 每步给 (current scene, next scene text, camera layout), 用 $w_{\text{proc}}$ adapter 让异构 model 都能跑, 然后统一 output 成 video, 在 controllability / quality / dynamics 三个 aspect 共 10 个 metric 上量。

**最 clever 的设计是什么?**

Photometric consistency metric — 用 forward-backward optical flow consistency 检测 texture flickering, 这是 CLIP/DINO feature-based consistency 的 blind spot。这个 insight 单独就值一篇 paper。

**有什么 actionable insight?**

如果你做 video generation: 优先解决 camera controllability, 这是 video model 的最大瓶颈, 参考 CameraCtrl / MotionCtrl。

如果你做 3D scene generation: 把 model 扩展到 4D, dynamics 是下一个 frontier。

如果你做 benchmark: 学习他们的 disentanglement — static 和 dynamic 分开, 避免 confounding。

如果你做 embodied AI: WorldScore 当前 open-loop, 你需要 closed-loop benchmark。

**最 deep 的 intuition?**

World generation 这件事看似模糊, 但只要你能 decompose 成 measurable subproblem (next-scene prediction + 3 个 aspect), 就能量化评估。Decomposition + disentanglement + unified output format, 这三件事让 heterogeneous model 在同一个坐标系里被比较。这是 ML benchmark 设计的经典套路, 这篇 paper 把它做对了。

参考链接合集:
- WorldScore 主页: https://haoyi-duan.github.io/WorldScore/
- Sora: https://openai.com/research/video-generation-models-as-world-simulators
- Cosmos: https://arxiv.org/abs/2501.03575
- WonderJourney: https://arxiv.org/abs/2312.03884
- WonderWorld: https://arxiv.org/abs/2406.18930
- LucidDreamer: https://arxiv.org/abs/2311.13384
- CogVideoX: https://arxiv.org/abs/2408.06072
- DROID-SLAM: https://arxiv.org/abs/2104.00080
- SEA-RAFT: https://arxiv.org/abs/2405.14714
- Grounding DINO: https://arxiv.org/abs/2303.05499
- SAM2: https://arxiv.org/abs/2408.00714
- CameraCtrl: https://arxiv.org/abs/2404.02101
- MotionCtrl: https://arxiv.org/abs/2312.06021
- 4D-fy: https://arxiv.org/abs/2311.17807
- VBench++: https://arxiv.org/abs/2411.13503
- WorldModelBench: https://arxiv.org/abs/2502.20694
- CLIPScore: https://arxiv.org/abs/2104.08718
- Gatys style: https://arxiv.org/abs/1508.06576
- VFI-Mamba: https://arxiv.org/abs/2407.02315
- CLIP-IQA+: https://arxiv.org/abs/2302.01312

希望这个版本更 "人话" 一点 — 把技术细节讲清楚, 但用更口语化、更 intuition-first 的方式组织, 配上 critical analysis。Build your intuition 才是 paper reading 的目的, technical details 只是 substrate。

---

# WorldScore: 第一个 World Generation 的 Unified Evaluation Benchmark 深度解析

## 1. 论文核心 motivation 与定位

这篇 paper 来自 Stanford 的 Haoyi Duan、Hong-Xing Yu、Sirui Chen、Li Fei-Fei、Jiajun Wu 团队。它试图解决一个非常 fundamental 的问题：**当前 visual generation 领域已经出现了 video models (Sora, Gen-3, Hailuo, CogVideoX)、3D scene generation (WonderJourney, LucidDreamer, WonderWorld)、4D generation (4D-fy) 等不同范式的 model，但是缺一个统一的 evaluation framework 来比较它们的 "world generation capability"**。

传统的 video benchmark 比如 VBench、EvalCrafter、FETV 都只 evaluate 单个 scene 的视频质量，既没有 multi-scene evaluation，也没有 camera trajectory 控制，更没有 3D consistency 评估。3D/4D scene generation 方法通常需要 image + camera matrix 作为 input，这些 benchmark 根本 incompatible。

WorldScore 的核心 design idea 是：**把 "world generation" 拆解成一个 next-scene generation 任务的序列**，每个 step 由一个 triplet $(\mathcal{C}, \mathcal{N}, \mathcal{L})$ 定义，这样可以让 video models、3D models、4D models 都在同一 framework 下比较。

参考链接:
- 项目主页: https://haoyi-duan.github.io/WorldScore/
- arXiv (与 Cosmos 同期出现的相关工作): https://arxiv.org/abs/2501.03575
- VBench (先前 video benchmark): https://arxiv.org/abs/2311.13503 (VBench++) 
- WorldModelBench: https://arxiv.org/abs/2502.20694

---

## 2. World Specification 的 formulation — 这是论文最关键的 contribution

### 2.1 Task decomposition

WorldScore 把 world generation formalize 为：

$$\mathbf{V} = g_{\text{world}}(w_{\text{proc}}(\mathcal{C}, \mathcal{N}, \mathcal{L}))$$

各变量含义:
- $\mathbf{V}$: generated video (统一输出格式, 让所有 model 都输出 video 以便直接比较)
- $g_{\text{world}}$: 任意 world generation model (可以是 video model, 3D model, 4D model)
- $w_{\text{proc}}$: model-specific preprocessing function (关键 — 它把统一规格的 input 转化为每个 model 期望的 input format)
- $\mathcal{C} = \{\mathbf{I}, \mathcal{P}\}$: current scene, 由 image $\mathbf{I}$ 和 text prompt $\mathcal{P}$ 组成
- $\mathcal{N}$: next-scene text prompt (描述下一个 scene 应该有什么内容)
- $\mathcal{L} = \{\mathcal{T}, \mathcal{Y}\}$: layout specification
  - $\mathcal{T} = (\mathbf{C}_1, \mathbf{C}_2, \cdots, \mathbf{C}_N)$: camera trajectory, 是 camera matrices 的序列
  - $\mathcal{Y}$: camera movement 的 textual description (例如 "camera moves left")

**Intuition 构建**: 这个 formulation 的精妙之处在于 $w_{\text{proc}}$ 函数。对于 3D scene generation models (WonderJourney, WonderWorld, LucidDreamer 等), 它们原生接受 camera matrices $\mathcal{T}$ 作为 input; 对于 video models (CogVideoX, Gen-3, Hailuo), 它们只接受 text, 所以 $w_{\text{proc}}$ 把 $\mathcal{Y}$ 拼接到 $\mathcal{N}$ 后面变成 augmented prompt。这种 adapter 模式让 heterogeneous models 可以在同一个 spec 上跑。

### 2.2 Static vs Dynamic disentanglement

非常关键的设计决策 — **static world 和 dynamic world 完全分离评估**:

- **Static world generation**: $\mathcal{N}$ 描述新 scene 内容 (新物体/新场景), $\mathcal{L}$ 描述 large camera movement. 评估 controllability + quality.
- **Dynamic world generation**: $\mathcal{N}$ 描述与 current scene 相同内容但 with dynamics (例如 "octopus moves"), $\mathcal{L}$ 显式指定 fixed camera position (no camera motion). 评估 dynamics.

这种 disentanglement 的好处: 如果不分离, 一个 model 既能 camera motion 又能 object motion, 你无法知道它的 dynamics metric 是来自 camera 还是来自 object。强制 fixed camera 才能纯测 object motion。这是非常 scientific 的设计。

---

## 3. Dataset Curation — 3000 examples 的构建

### 3.1 总体统计

| 类别 | Subset | # Samples |
|---|---|---|
| Static | Photorealistic Indoor | 5 categories × 100 = 500 |
| Static | Photorealistic Outdoor | 5 categories × 100 = 500 |
| Static | Stylized Indoor | 500 |
| Static | Stylized Outdoor | 500 |
| Dynamic | Photorealistic | 5 motion types × 100 = 500 |
| Dynamic | Stylized | 500 |
| **Total** | | **3000** |

Indoor 5 categories: Dining, Living, Passage, Public, Work
Outdoor 5 categories: City, Suburb, Aquatic, Terrestrial, Verdant
Dynamic 5 motion types: Articulated, Deformable, Fluid, Rigid, Multi-Motion
7 styles: anime, cyberpunk, Chinese ink painting, ukiyo-e, impressionism, post-impressionism, minecraft

### 3.2 Image filtering pipeline (Supp. B.1)

这是论文里一个非常 rigorous 的工程细节。他们 source 自多个 datasets (Table S2): Matterport3D、Hypersim、SUN-RGBD、DIODE、ETH3D、LHQ、EDEN、Argoverse-HD、InterriorVerse。然后应用 5 层 filtering:

1. **Quality filter**: CLIP-IQA + CLIP Aesthetic predictor 去掉低质量图
2. **Perspective filter**: 用 Perspective Fields [28] 估计 yaw/pitch/FOV, 过滤极端 roll/pitch 和窄 FOV 图像 (这是为了确保 starting image 适合做 world generation — 极端视角会让后续 generation 不自然)
3. **Similarity filter**: 用 CLIPSIM 去掉 sequential datasets 里冗余的相邻帧
4. **Brightness filter**: 排除过暗图像
5. **Manual judgment**: 人工 review 去掉不可行的 starting scenes (比如 "mid-air city image")

**Intuition**: 这个 filtering 说明 world generation benchmark 比 image benchmark 更挑剔 — starting image 必须有合理的 camera viewpoint 和合理的 scene composition, 才能生成连贯的 next scenes。

### 3.3 Next-scene prompt 的 auto-regressive generation (Eq. 2)

$$\mathcal{N} = \text{LLM}(\mathcal{I}, \mathcal{P})$$

这里 $\mathcal{I}$ 是 task specification (例如 "Generate a scene description different from the past scenes"), 输入是 past + current scene descriptions。**这个 design 选择很重要**: 不用 human-curated next-scene prompts, 而是用 LLM auto-regressive 生成。这意味着 benchmark 本身可以无限扩展 (生成 larger worlds 时只要重复 call LLM)。

具体来说, 20% 的 static examples 是 "large worlds" (4 个 scenes, 需要 3 次 LLM call 生成 $\mathcal{N} = \mathcal{N}_1 + \mathcal{N}_2 + \mathcal{N}_3$), 80% 是 "small worlds" (2 个 scenes, 1 次 LLM call)。

### 3.4 Layout curation — 8 种 camera movements

参考电影 industry 的 standard camera movements, 设计了 8 种:
- 覆盖所有 spatial directions
- 与 text-to-video models 的 training data (movie clips) compatible, 这样 T2V models 才能理解 textual description $\mathcal{Y}$

包括 intra-scene movement (moving into a scene) 和 inter-scene transition (pulling out camera)。

参考: 经典 cinematography 术语比如 pan、tilt、dolly、zoom、tracking 等, 参见 https://www.studiobinder.com/blog/types-of-camera-movements/

---

## 4. WorldScore Metrics — 10 个 metrics 的详细技术讲解

这是论文的核心 evaluation machinery, 分 3 个 aspect。

### 4.1 Controllability (3 metrics)

#### (1) Camera controllability (Eq. 3, S1, S2)

$$e_{\text{camera}} = \sqrt{e_\theta \cdot e_t}$$

其中:
- $e_\theta$: rotation error (degrees), 公式 S1:
  $$e_\theta = \arccos\left(\frac{\text{tr}(\mathbf{R}_{\text{gt}} \mathbf{R}^T) - 1}{2}\right) \cdot \frac{180}{\pi}$$
  - $\mathbf{R}_{\text{gt}}, \mathbf{R} \in SO(3)$: ground truth 和 estimated rotation matrices
  - 这个公式是 standard rotation angular distance, 利用 trace 性质: $\text{tr}(\mathbf{R}_a \mathbf{R}_b^T) = 1 + 2\cos(\theta)$, 反解出 $\theta$.
- $e_t$: translation error (scale-invariant), 公式 S2:
  $$e_t = \|\mathbf{t}_{\text{gt}} - s\mathbf{t}\|_2$$
  - $\mathbf{t}_{\text{gt}}, \mathbf{t} \in \mathbb{R}^3$: ground truth 和 estimated translation
  - $s$: least-square scale, 用以消除尺度不确定性 (video models 输出的 camera trajectory 没有 absolute scale)

**为什么用 geometric mean 而不是 arithmetic mean?** 几何平均对 outlier 更敏感 — 如果一个 model 的 rotation 完全错 (e.g., $e_\theta$ 很大), 即使 translation 还可以, geometric mean 会把整体 score 拉低, 这正是我们想要的 (camera 控制需要同时满足 rotation 和 translation)。

实现上用 **DROID-SLAM** (https://arxiv.org/abs/2104.00080) 估计 generated video 的 frame-wise camera poses, 然后与 ground truth trajectory 比较。

**Intuition**: 这个 metric 评估 "model 是否真的执行了指定的 camera 指令"。Table 2 显示 video models 在这一项上 score 都很低 (CogVideoX-T2V 40.22 算是 video models 里最高的, 但 3D models 都在 84-94 之间), 因为 video models 通常没有显式 camera control signal injection。

#### (2) Object controllability

- 用 **Grounding DINO** (https://arxiv.org/abs/2303.05499) 做 open-set object detection
- 从 $\mathcal{N}$ 中提取 1-2 个 object descriptions (entities, 见 Table S3 例子的 "Entities" 字段)
- 计算 detection success rate

**Intuition**: 这个 metric 评估 model 是否真的把 prompt 里指定的 object generate 出来了。例如 prompt 说 "yellow armchair + bookshelf", 那 Grounding DINO 应该 detect 到这两个 object。

#### (3) Content alignment

- 用 **CLIPScore** (https://arxiv.org/abs/2104.08718) 计算 $\mathcal{N}$ (full text) 与 generated frame 之间的相似度
- 与 Object controllability 互补: object controllability 只看 ~1/4 的 prompt length (object 部分), content alignment 看整个 prompt (包括场景描述、关系、attributes)

### 4.2 Quality (4 metrics)

#### (1) 3D Consistency (Eq. S3)

$$e_{\text{reproj}} = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \|\mathbf{p}_{ij}^* - \Pi(\mathbf{P}_{ij})\|_2$$

- $\mathcal{V}$: co-visible points 的 valid set
- $\mathbf{p}_{ij}^*$: frame $j$ 上观察到的 ground truth pixel (来自 image $i$ 的对应点)
- $\mathbf{P}_{ij}$: 由 refined depth + camera pose 重建出来的 3D 点
- $\Pi$: projection function (把 3D 点 project 回 image plane)
- $\|\cdot\|_2$: Euclidean distance

**用 DROID-SLAM 而不是 COLMAP 的原因** (Supp. C.2): DROID-SLAM 用 **Dense Bundle Adjustment (DBA)**, 估计 dense per-pixel depth, 而不是 sparse feature matching。DROID-SLAM 对 appearance changes robust (它不强求 texture 一致, 只关心几何一致), 这正好契合我们想要 isolate geometry inconsistency 的目标。

**Intuition**: 如果一个 video 在 3D 上 inconsistent (例如一个 wall 突然 warp 成别的形状), 那 reprojection error 会很大。看 Figure 6 上面的例子: bad case 里 geometry 突变, 而 good case 平滑过渡。

#### (2) Photometric Consistency (Eq. S4-S6)

$$\mathbf{p}_B = \mathbf{p}_A + \mathcal{F}_{A \to B}(\mathbf{p}_A)$$
$$\mathbf{p}_A' = \mathbf{p}_B + \mathcal{F}_{B \to A}(\mathbf{p}_B)$$
$$e_{\text{photometric}} = \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{p}_{A,i} - \mathbf{p}_{A,i}'\|_2$$

- $\mathcal{F}_{A \to B}, \mathcal{F}_{B \to A}$: forward / backward optical flow
- $\mathbf{p}_A$: 在 frame A 中心 crop 的 points
- $\mathbf{p}_B$: 用 forward flow 把 $\mathbf{p}_A$ track 到 frame B
- $\mathbf{p}_A'$: 用 backward flow 把 $\mathbf{p}_B$ track 回 frame A
- $N$: sampled points 数量
- 度量: **AEPE (Average End-Point Error)**, 即 forward-backward consistency

**关键 insight**: 这个 metric 设计用来 capture **CLIP/DINO features 无法捕获的 fine-grained texture flickering**。比如一个 mountain 在所有帧里都是 mountain (CLIP feature 几乎不变), 但 grass texture 在帧之间 shift、distort。这种 artifact 用 CLIP/DINO based consistency 检测不到, 但 forward-backward flow consistency 能检测到。

参考 optical flow estimation: SEA-RAFT (https://arxiv.org/abs/2405.14714)

#### (3) Style Consistency

- 用 **Gram matrix** (Gatys' neural style transfer, https://arxiv.org/abs/1508.06576) 的 Frobenius norm 差异
- 计算第一帧和最后一帧的 Gram matrix 之间的 F-norm difference
- Gram matrix 捕获 texture / style statistics, 而不是具体 content

**Intuition**: 如果 model 在 generation 过程中 style drift (例如从 photorealistic 慢慢变成 cartoonish), Gram matrix 会变化。这是评估 multi-frame style coherence 的 standard 技术。

#### (4) Subjective Quality

- 不是 single metric, 而是 **CLIP-IQA+ 和 CLIP Aesthetic 的 arithmetic mean** (Supp. C.4)
- 这个组合是通过 **400 人的 human study** 选出来的 (Table S5): 在 12 个 candidate metrics / combinations 中, CLIP-IQA+ & CLIP Aesthetic 的 agreement score 最高 (0.637), 接近 upper bound 0.772

参考:
- CLIP-IQA+: https://arxiv.org/abs/2302.01312
- CLIP Aesthetic: https://github.com/christophschuhmann/improved-aesthetic-predictor

### 4.3 Dynamics (3 metrics) — 只在 dynamic world generation 部分评估

#### (1) Motion Accuracy (Eq. S7)

$$s_{\text{motion-acc}} = \max(\mathbf{F} \odot \mathbf{M}) - \max(\mathbf{F} \odot \bar{\mathbf{M}})$$

- $\mathbf{F} \in \mathbb{R}^{H \times W}$: optical flow magnitude (consecutive frames 之间)
- $\mathbf{M} \in \{0, 1\}^{H \times W}$: dynamic objects 的 segmentation mask (在 former frame 上)
- $\bar{\mathbf{M}}$: $\mathbf{M}$ 的 complement (non-dynamic region)
- $\odot$: element-wise product
- $\max$: 取 matrix 中所有元素的最大值

**Intuition**: 检查 motion 是否发生在应该发生的地方。如果 mask 内 optical flow 大, mask 外 optical flow 小, 那 motion accuracy 高。如果反过来 (背景动了, 主体没动), 那 score 低甚至为负。

Figure 6 下面 row: octopus 应该动, jellyfish 应该静。Bad case 里 jellyfish 动 octopus 不动 — 这正是 motion accuracy 要检测的常见 failure mode。

实现: 用 **SEA-RAFT** 估 optical flow, **SAM2** (https://arxiv.org/abs/2408.00714) track mask across frames。第一帧的 mask 是 dataset annotation 提供的。

#### (2) Motion Magnitude

- 简单: optical flow magnitude $\mathbf{F}$ 的 median value (over all entries)
- 用 median 而不是 mean 是为了 robust to outlier (某些 region 有剧烈 motion, 但大多数 region 静止)
- 评估 model 是否能产生 large motion (有些 model 太 conservative, 输出几乎 static video)

#### (3) Motion Smoothness

- 用 **video frame interpolation** model (VFI-Mamba, https://arxiv.org/abs/2407.02315) 作为 reference
- 流程: 丢掉 odd-indexed frames $\{f_1, f_3, \cdots\}$, 用 VFI model 重建这些 frames, 然后与原始 frames 比较
- 计算 MSE + SSIM + LPIPS, 然后取 mean (after normalization)

**Intuition**: 如果 generated motion 是 smooth 的, 那么 odd frames 可以从 even frames interpolate 出来且 error 小。如果 motion jittery (常见 video artifact), VFI model 没法准确 reconstruct odd frames。

### 4.4 Score normalization (Eq. S8)

$$s^{\text{norm}} = \begin{cases} \left\langle\frac{s - b^{\min}}{b^{\max} - b^{\min}}\right\rangle, & \text{if higher better} \\ \left\langle 1 - \frac{s - b^{\min}}{b^{\max} - b^{\min}}\right\rangle, & \text{if lower better} \end{cases}$$

- $s$: raw value
- $b^{\min}, b^{\max}$: empirical bounds (Supp. C.8 详述)
- $\langle\cdot\rangle$: clip function, 限制 $s^{\text{norm}} \in [0, 1]$

Empirical bounds 的选择非常有意思:
- **Camera controllability**: max 用 "fixed camera sequence" 作为 baseline (惩罚不动 camera 的 generation), min = 0
- **Object controllability**: 直接用 0% 和 100% (theoretical bounds)
- **3D/Photometric/Style consistency**: 用 video frame interpolation baseline 作为 max (significant inconsistency), 0 作为 min
- **Motion smoothness**: 用 OpenVid-1M 的 real-world videos 作为 reference
- **Content alignment / Subjective quality / Motion accuracy / Motion magnitude**: z-score rescaling, 让 selected models 落在 25-75 之间 (这是个 pragmatic 选择, 因为这些 metric 没有 natural bounds)

**Intuition 构建**: normalization 设计反映了 metric 设计者的 prior 知识 — 哪些 baseline 是"自然下限" (fixed camera, interpolated video), 哪些需要 z-score (主观质量没有上限)。

### 4.5 Aggregation

- **WorldScore-Static** = arithmetic mean of (3 controllability + 4 quality) = 7 dimensions, 只用 static world 数据
- **WorldScore-Dynamic** = WorldScore-Static + 3 dynamics dimensions = 10 dimensions, 用 static + dynamic 数据
- 对于 3D scene generation models 不支持 dynamic, dynamics metrics 直接 assign 0 (这是 Table 2 中 3D models 一行 dynamics 列都是 0.00 的原因)

---

## 5. 20 个模型的详细 evaluation results (Table 2 解读)

让我把 Table 2 重新组织成更 readable 的形式, 按 WorldScore-Static 排序:

| Rank | Model | Type | WS-Static | WS-Dynamic | Cam Ctrl | 3D Consist | Photo Consist | Motion Mag |
|---|---|---|---|---|---|---|---|---|
| 1 | WonderWorld | 3D | 72.69 | 50.88 | 92.98 | 86.87 | 85.56 | 0.00 |
| 2 | LucidDreamer | 3D | 70.40 | 49.28 | 88.93 | 90.37 | 90.20 | 0.00 |
| 3 | InvisibleStitch | 3D | 63.75 | 42.78 | 93.20 | 88.51 | 89.19 | 0.00 |
| 4 | CogVideoX-I2V | I2V | 62.15 | 59.12 | 38.27 | 86.21 | 88.12 | 26.42 |
| 5 | Gen-3 | I2V (closed) | 60.71 | 57.58 | 29.47 | 68.31 | 87.09 | 27.48 |
| 6 | Hailuo | I2V (closed) | 57.55 | 56.36 | 22.39 | 67.18 | 62.82 | 27.20 |
| 7 | LTX-Video | I2V | 55.44 | 56.54 | 26.55 | 78.41 | 88.92 | 29.95 |
| 8 | Allegro | I2V | 55.31 | 51.97 | 26.72 | 67.29 | 47.35 | 40.28 |
| 9 | CogVideoX-T2V | T2V | 54.18 | 48.79 | 40.22 | 68.81 | 64.20 | 47.31 |
| 10 | EasyAnimate | I2V | 52.85 | 51.65 | 27.80 | 38.72 | 34.84 | 31.16 |
| 11 | WonderJourney | 3D | 61.12 | 44.63 | 84.60 | 80.60 | 79.03 | 0.00 |
| 12 | Text2Room | 3D | 62.10 | 43.47 | 94.01 | 88.71 | 88.36 | 0.00 |
| 13 | SceneScape | 3D | 50.73 | 35.51 | 84.99 | 76.54 | 62.88 | 0.00 |
| ... | 4D-fy | 4D | 27.98 | 32.10 | 69.92 | 35.47 | 1.59 | 22.88 |

### 5.1 Key insights

**Insight 1: 3D models 完胜 video models 在 static world generation**

Top-2 是 WonderWorld (72.69) 和 LucidDreamer (70.40), 远超最好的 video model CogVideoX-I2V (62.15)。原因:
- 3D models 的 camera controllability 极高 (88-94, 因为它们原生接受 camera matrix)
- 3D consistency 高 (3D model 本身就是 3D representation, 几何 naturally consistent)
- Photometric consistency 高 (一旦 texture 生成, 就是 baked-in 的)

**Insight 2: Video models 的瓶颈是 camera controllability**

最好的 video model 在 camera controllability 上也才 40.22 (CogVideoX-T2V), 远低于任何 3D/4D model。这暗示了未来 video models 的关键改进方向: **explicit camera conditioning injection** (参考 CameraCtrl https://arxiv.org/abs/2404.02101, MotionCtrl https://arxiv.org/abs/2312.06021)。

**Insight 3: Best open-source video model 已经追上 closed-source**

CogVideoX-I2V (62.15) 甚至超过了 Gen-3 (60.71) 和 Hailuo (57.55)。这是开源社区的重要 milestone。CogVideoX 在 camera controllability 上更强 (38.27 vs 29.47 和 22.39), 但在 object controllability 和 content alignment 上较弱。

**Insight 4: 4D-fy 表现糟糕 (27.98)**

4D generation 还在早期。注意 4D-fy 的 photometric consistency 只有 1.59 (极差), motion accuracy 22.22 (很低)。原因是 4D generation 任务本身就极具挑战性 — 既要 3D consistent, 又要有 dynamics。

**Insight 5: Motion smoothness vs magnitude trade-off**

观察 motion smoothness 和 motion magnitude 的关系: 大 motion 通常伴随低 smoothness。这是 video models 的 fundamental challenge — 既要大动作又要自然过渡。

**Insight 6: T2V vs I2V 的有趣 trade-off**

CogVideoX-T2V vs CogVideoX-I2V (同 architecture):
- T2V: higher controllability, larger motion magnitude (47.31 vs 26.42)
- I2V: higher quality scores

**Intuition**: I2V models 倾向于 stick to input image 的 viewpoint (因为 reference image 的 anchor 作用太强), 而 T2V models 更愿意 generate large camera motion (因为没有 image anchor)。这暗示了 I2V 的一个 fundamental challenge — 如何让 model 摆脱 input image 的 anchor, 真正 generate 新内容。

**Insight 7: Video models 在 outdoor scenes 和 long sequences 上明显弱**

Figure 7 的 subdomain analysis:
- Indoor scenes: video models 与 3D models 的 gap 较小
- Outdoor scenes: video models 显著弱于 3D models
- Long sequence (large world): video models struggle significantly

---

## 6. Human preference validation (Supp. D)

### 6.1 Agreement score 设计

非常 elegant 的 probabilistic agreement score:

给定 video pair $(\mathbf{A}, \mathbf{B})$, 让参与者 2AFC (2-alternative forced choice) 选择 quality 更高的一个。设选择 A 的比例为 $p$, 选择 B 的比例为 $1-p$。

对于 metric $m$:
- 若 $\text{score}_m(A) > \text{score}_m(B)$: agreement = $p$
- 若 $\text{score}_m(A) < \text{score}_m(B)$: agreement = $1-p$
- 若相等: agreement = 0.5

最终 metric 的 agreement = average over all human-rated pairs。

**Intuition**: 这个 score 设计的妙处在于 — 如果 metric 和 human 一致, 那 metric 偏好 A 时, $p$ 应该接近 1 (大部分人也选 A), agreement 高。如果 metric 错了 (偏好 A 但大部分人选 B), $p$ 接近 0, agreement 低。它不需要 metric 的 absolute scale 与 human 对齐, 只需要 ordinal alignment。

### 6.2 Results

- Subjective quality: CLIP-IQA+ & CLIP Aesthetic 组合 agreement = 0.637, 在 12 个 candidate 中最高 (Table S5)
- Upper bound (如果 metric 总是 agree with majority vote): 0.772
- 其他 metrics 在 2AFC test 上 (score 差 30): 71.2% - 97.3% 的 agreement (Table S6), 说明所有 metric 都与 human preference 对齐

### 6.3 Robustness to resolution (Table S7)

用 EasyAnimate 的 1344×768 输出, resize 到 256×256, 重新评估。所有 metrics 差异 ≤ 0.83, 说明 WorldScore 对 resolution/aspect ratio 变化 robust。这非常重要, 因为 Table S1 显示 models 的 resolution 从 256×256 (4D-fy) 到 1344×768 (EasyAnimate) 不等。

---

## 7. 与相关 benchmark 的对比 (Table 1)

| Benchmark | # Examples | Multi-Scene | Unified | Long Seq. | Image Cond. | Multi-Style | Camera Ctrl. | 3D Consist. |
|---|---|---|---|---|---|---|---|---|
| TC-Bench | 150 | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| EvalCrafter | 700 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| FETV | 619 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| VBench | 800 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| T2V-CompBench | 700 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Meng et al. | 160 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Wang et al. | 423 | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| ChronoMagic-Bench | 1649 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| WorldModelBench | 350 | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| **WorldScore** | **3000** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**WorldScore 是唯一一个在所有 8 个维度上都 ✓ 的 benchmark**。这是它的核心 contribution。

---

## 8. 个人 critical analysis 与未来方向

### 8.1 强项

1. **Task decomposition 是 elegant 的**: $(\mathcal{C}, \mathcal{N}, \mathcal{L})$ triplet 既 universal 又 interpretable
2. **Static/dynamic disentanglement 是科学的**: 避免 confounding, 让 metrics 可以 isolate 不同 aspect
3. **DROID-SLAM based 3D consistency 是正确的选择**: dense, robust to appearance changes
4. **Photometric consistency 设计非常 insightful**: CLIP/DINO consistency 确实有 blind spot, forward-backward flow consistency 填补了这个 gap
5. **20 个 model 大规模 evaluation 提供了 actionable insights**

### 8.2 潜在 limitations 与开放问题

1. **Camera controllability 的 scale ambiguity**: $e_t$ 用 least-square scale $s$ 消除 scale, 但 video model 的 absolute depth 本就 undefined, 这个 scale 选择是否对 evaluation 公平? 如果 video model 完全 stationary, $s$ 会变成 0, $e_t = \|\mathbf{t}_{\text{gt}}\|_2$, 但 paper 用 fixed camera 作为 baseline, 这意味着 stationary 模型会被严厉惩罚 — 这个设计是否太苛刻? 一些场景 fixed camera 是 valid 选择。

2. **4D generation 的 inclusion 不够 robust**: 只有一个 4D model (4D-fy), 而且 4D-fy 的 photometric consistency 是 1.59 (接近 0), 这意味着 metric 对 4D generation 可能过于苛刻, 或者 4D-fy 本身就 broken。需要更多 4D models 才能给出有意义的 ranking。

3. **Style consistency 只看第一帧和最后一帧**: 这个简化可能 miss 中间的 style drift。考虑 sliding window 或者 all-pairs comparison 更 robust。

4. **Subjective quality 是 image-level metric**: 它逐帧评估, 但没有 explicit temporal aesthetic。一个视频可能每帧都好看但 transitions 难看。

5. **Motion accuracy 的 $\max$ operator**: $\max(\mathbf{F} \odot \mathbf{M})$ 是 non-robust 的 — 一个 outlier pixel 就能 dominate。考虑 90th percentile 或者 mean 更 robust。

6. **Long sequence evaluation (large worlds) 占比小**: 只有 20% 的 static examples 是 large worlds, 这可能不足以给出 statistically significant 的 long sequence 评估。

### 8.3 未来方向 (从论文 Section 5 + 个人分析)

1. **Bridging 3D/4D**: 当前 3D models 在 static worlds 强, 4D models 弱, video models 在 dynamics 上有优势但 controllability 弱。未来 ideal model 应该 combine 3D 的 camera control + video 的 dynamics + 4D 的 spatiotemporal coherence。

2. **Explicit camera conditioning for video models**: 这是 video model 提升的最关键 direction。参考 CameraCtrl、MotionCtrl。

3. **Long sequence generation**: 当前 20% large worlds 配比应该在未来版本提升到 50%+, 这样才能 push 模型做真正的 world generation 而不是 single scene extrapolation。

4. **Embodied AI 视角**: WorldScore 与 embodied AI 的联系 — world generation 是 embodied agent 的 simulator 的关键, 参考 Sora's world simulator vision (https://openai.com/research/video-generation-models-as-world-simulators)。

5. **Real-time world generation**: Table S1 显示 WonderWorld 10s 一个 generation, 但 4D-fy 需要 3 小时 (even with reduced steps)。Real-time constraint 应该被加入 future benchmark。

6. **Physics-aware evaluation**: 当前 benchmark 没有显式 physics evaluation (e.g., object permanence, gravity, collision)。参考 WorldModelBench 的 physical commonsense 评估思路。

7. **Interactive world generation**: 当前 benchmark 是 open-loop的, 给定 spec 让 model 一次性 generate。未来应该 evaluate closed-loop, 即 model 根据 agent action 生成下一帧 (这与 DIAMOND, Genie 等 world model 方向相关)。

---

## 9. 相关工作的关联 web (扩展联想)

### 9.1 与 Sora 的关系

Sora (https://openai.com/research/video-generation-models-as-world-simulators) 提出 "video generation models as world simulators" 的 vision, 但 OpenAI 没有给出 quantitative evaluation。WorldScore 正好填补了这个 evaluation gap。论文引用 [6] 显示作者明确意识到 Sora 开启的 world generation paradigm。

### 9.2 与 Cosmos 的关系

Cosmos (https://arxiv.org/abs/2501.03575, NVIDIA) 是同期工作, 提出 "world foundation model platform for physical AI"。Cosmos 关注 training, WorldScore 关注 evaluation, 二者互补。论文引用 [1] 把它列为 video generation model。

### 9.3 与 3D scene generation 的最新进展

LucidDreamer (https://arxiv.org/abs/2311.13384) 用 3D Gaussian Splatting + outpainting, 在 Table 2 排名第 2。WonderJourney (https://arxiv.org/abs/2312.03884) 和 WonderWorld (https://arxiv.org/abs/2406.18930) 是同一作者 (Hong-Xing Yu 也是本论文作者), 用 sequential generation, 排名第 1。这暗示了 Stanford 团队的纵向布局 — 既做 model 又做 benchmark。

### 9.4 与 SLAM 社区的关系

DROID-SLAM (https://arxiv.org/abs/2104.00080) 被 WorldScore 用来 evaluate 3D consistency 和 camera controllability。这是 SLAM 技术"反哺" video generation evaluation 的有趣案例。

### 9.5 与 optical flow 社区的关系

SEA-RAFT (https://arxiv.org/abs/2405.14714) 用来估 optical flow for photometric consistency 和 motion accuracy。这暗示 optical flow estimator 的质量直接影响 WorldScore 的 reliability — 如果 flow estimator 在 generated (potentially out-of-distribution) 视频上不 robust, 整个 evaluation 不可信。

### 9.6 与 open-set detection 的关系

Grounding DINO (https://arxiv.org/abs/2303.05499) 用于 object controllability 评估。但 Grounding DINO 在 stylized (e.g., ukiyo-e) 图像上的 detection accuracy 如何? 这是 benchmark reliability 的潜在隐患。

---

## 10. 总结性 intuition

这篇 paper 的核心 contribution 可以浓缩为:

> **World generation 作为一个 AI capability, 一直没有统一衡量方法。WorldScore 把它分解为 (current, next, layout) triplet 序列, 在 3 个 aspect (controllability, quality, dynamics) 上用 10 个 metrics 评估, 通过 $w_{\text{proc}}$ adapter 让 3D/4D/video models 可以在同一 spec 上比较。3000 个测试样本覆盖 static/dynamic × photorealistic/stylized × indoor/outdoor × multi-scale sequence, 20 个 model 的 evaluation 揭示了 "3D models 完胜 video models 在 static, video models 在 dynamics 有优势, 但所有模型都 struggle 在 long sequence generation 和 outdoor scenes" 的 key insights。**

更深层 intuition:
- **Decomposition**: 把模糊的 "world generation" 变成 concrete 的 next-scene prediction, 这是经典 ML 方法论 — 把 hard problem 分解成 measurable subproblems
- **Disentanglement**: static 和 dynamic 分开评估, 避免 confounding (motion accuracy 不会被 camera motion 污染)
- **Dense vs Sparse**: 用 DROID-SLAM (dense) 而不是 COLMAP (sparse), 用 forward-backward flow consistency (dense) 而不是 CLIP feature (sparse semantic), 都是想要 capture fine-grained artifact 而不是 coarse categorical correctness
- **Unified output format**: 强制所有 model 输出 video, 这是 evaluation 公平性的基础
- **Human alignment**: 400 人 study 验证 metric 与 human preference 对齐, 而不是 arbitrary automatic score

参考与扩展阅读:
- WorldScore 主页: https://haoyi-duan.github.io/WorldScore/
- 论文 (本 markdown 描述): https://arxiv.org/abs/2501.XXXXX (推测)
- VBench: https://vchitect.github.io/VBench-project/
- Sora: https://openai.com/research/video-generation-models-as-world-simulators
- Cosmos: https://arxiv.org/abs/2501.03575
- WonderJourney: https://arxiv.org/abs/2312.03884
- WonderWorld: https://arxiv.org/abs/2406.18930
- LucidDreamer: https://arxiv.org/abs/2311.13384
- CogVideoX: https://arxiv.org/abs/2408.06072
- CameraCtrl: https://arxiv.org/abs/2404.02101
- DROID-SLAM: https://arxiv.org/abs/2104.00080
- SEA-RAFT: https://arxiv.org/abs/2405.14714
- Grounding DINO: https://arxiv.org/abs/2303.05499
- SAM2: https://arxiv.org/abs/2408.00714
- 4D-fy: https://arxiv.org/abs/2311.17807
- CLIPScore: https://arxiv.org/abs/2104.08718
- Gatys style transfer: https://arxiv.org/abs/1508.06576

这就是对 WorldScore 的一个 comprehensive technical walkthrough。希望这能 build 起你的 intuition — 不仅是 paper 说了什么, 更是为什么这么 design, 每个 metric 的 deep rationale, 以及它在我们 trajectory towards real world generation models 中的位置。
