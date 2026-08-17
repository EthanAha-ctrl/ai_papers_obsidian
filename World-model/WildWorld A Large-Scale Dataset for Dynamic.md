---
source_pdf: WildWorld A Large-Scale Dataset for Dynamic.pdf
paper_sha256: 63a80641653ae93a2a31e0ec7a48b28148ade99f2bfa639f4deb6ea1146c3850
processed_at: '2026-08-13T04:38:07-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 WildWorld

## 1. 一句话说清楚这篇 paper 在干啥

哥们你知道现在 video generation model 厉害，Sora 2、Wan 这些都能生成像样的视频了。但是你想要 **interactive** 的视频——就是你给个 action，视频按 action 演化——这事还很烂。

为啥烂？因为之前的数据集 action 跟像素变化是**直接绑死**的。比如 "move left" 在视频里就是镜头往左 pan，模型学到的是个 surface shortcut，根本没学到 world dynamics。

WildWorld 这篇 paper 做的事：从 **Monster Hunter Wilds**（Capcom 的 AAA 级 ARPG）里自动采集了一个 108M 帧的数据集，关键创新是每帧都带 **explicit state annotation**——角色 skeleton、HP、位置、camera pose、depth 全有。这样模型就得学真正的 state transition，不能靠 pixel shortcut 作弊。

项目页：https://shandaai.github.io/wildworld-project/
代码：https://github.com/ShandaAI/WildWorld

---

## 2. 核心 motivation：为什么 action-conditioned video model 现在学不到 dynamics

paper 用 POMDP 的视角切入，这个视角对咱们搞 RL 的人很亲切。世界的真实演化是：

$$s_{t+1} = \mathcal{T}(s_t, a_t)$$

state $s_t$ 通过 transition function $\mathcal{T}$ 在 action $a_t$ 驱动下演化到 $s_{t+1}$。我们能观察到的 video frame $o_t$ 只是 state 的 partial projection：

$$o_t = \mathcal{O}(s_t)$$

video world model 想学的就是从 $(o_{\le t}, a_{\le t})$ 推出 $o_{t+1}, o_{t+2}, \dots$。但现在的数据集有个致命问题：

**大部分数据集里 action 直接对应 pixel 变化**。举个 paper 里的例子："shoot" 这个 action，如果弹药还有，视频里有开火效果；弹药打光了，再 "shoot" 视频里啥也不发生。但数据集没标弹药数这个 state，模型只看到 "shoot action → 有时画面有火光有时没有"，它学不到 "弹药" 这个 latent state，所以在 long-horizon 生成时会崩。

这就跟 AlphaGo 学棋一样：只看棋盘图像学，模型学到一堆视觉 pattern；给 board state 的 structured representation，模型才能学到真正的 game tree。

WildWorld 的设计哲学就是：**给 video world model 提供 board state**。

---

## 3. 数据从哪来：给 game engine 插桩

Monster Hunter Wilds 用的是 Capcom 的 RE Engine。现代 game engine 的执行流程是两段：
1. **game logic tick**：处理 player input，更新 world state
2. **rendering pipeline**：消费 state，渲染图像

paper 的骚操作是**两段都插桩**：
- engine tick 阶段：把 action/state 序列化成 JSON（HP、位置、rotation quaternion、animation ID 等等）
- rendering 阶段：用 **Reshade shader hook** 拿 RGB buffer 和 depth buffer

HUD 移除很巧妙——直接 disable late-stage shaders 拿到无 HUD 干净帧，省掉后处理 inpainting 的麻烦。

录制用魔改的 OBS Studio，2K 全屏用 Reshade 分成 4 个 sub-window，RGB 和 depth 各占一个。RGB 用 lossy HEVC（16-20 Mbps VBR），depth 用 HEVC lossless（约 20 Mbps）。所有 stream 都嵌 timestamp，后处理时用 timestamp 对齐。

**自动化 gameplay**：用 game 自带的 rule-based companion AI 当 bot。这个 AI 是 Capcom 调过的 behavior tree，零训练成本。多样性怎么保证？behavior tree 在高维 state space 下产生 divergent trajectories，加上 monster 反应本身 stochastic，每个 session 的 trajectory 差异很大。camera 用 game 的 target-lock 系统，自动跟 monster。

---

## 4. 数据规模：量级很吓人

| 维度 | 数值 |
|---|---|
| 总帧数 | **108M** |
| 每帧 annotation 列数 | 119 |
| Player action triplets | 5,960 种 |
| Monster action pairs | 2,132 种 |
| Player motion IDs | 455 (跨 24 个 bank) |
| Monster motion IDs | 527 (跨 13 个 bank) |
| Monster 物种 | 29 |
| 武器类型 | 4（Great Sword, Long Sword, Bow, Dual Blades）|
| 地图 | 5 张 |
| Combat vs Travel | 66% : 34% |

Player action 用三元组编码：$a = (\text{weapon\_id}, \text{bank\_id}, \text{motion\_id})$。weapon 选大类，bank 选动作类别（站立/攻击/回避），motion 选具体动画。这种 hierarchical action space 的 cardinality 是 $\sum_{w,b} |\mathcal{M}_{w,b}|$，远比 flat discrete action space 丰富。

Top-150 action 占 58.49% 样本，长尾分布——说明 action space 真的很大很丰富。

---

## 5. Data filtering：五个维度砍低质量样本

这部分细节很值得看，因为它揭示了 interactive video model 训练的实战痛点：

| Filter | 阈值 | 为啥这么设 |
|---|---|---|
| Duration | $T \ge 81$ frames | 81 帧 @16FPS ≈ 5 秒，短于这个对 long-horizon 建模没用 |
| Temporal continuity | $\Delta t \le 1.5 \Delta t_{target}$ | 30FPS 下目标帧间隔 33ms，阈值约 50ms，超过说明卡顿或进 cutscene |
| Luminance | 不连续 15 帧极亮/极暗 | 极端亮度破坏训练稳定性 |
| Camera occlusion | spring-arm 不持续收缩 | 第三人称 camera 被遮挡时 spring-arm 自动收缩，camera-character 距离异常小 |
| Character occlusion | 首帧 2D skeleton overlap < 30% | 避免首帧角色重叠导致 i2v 起始条件歧义 |

Character occlusion filter 的计算挺聪明的：把 3D skeleton 用 GT camera 投影到 2D screen coordinates：

$$\mathbf{p}_{2D}^{(j)} = \pi\left(K[R|t] \mathbf{p}_{3D}^{(j)}\right)$$

变量含义：
- $\mathbf{p}_{3D}^{(j)} \in \mathbb{R}^3$：第 $j$ 个 keypoint 的 3D 位置
- $K \in \mathbb{R}^{3\times 3}$：camera intrinsic matrix
- $[R|t] \in \mathbb{R}^{3\times 4}$：camera extrinsic（rotation $R$ + translation $t$）
- $\pi$：perspective projection（齐次坐标除以 z）

然后算两个角色 projected 2D area 的 overlap。用 skeleton projection 是 deterministic 的，比 image segmentation 可靠（角色互相遮挡时分割会失败）。

---

## 6. Caption 标注：两层 hierarchy

**Action-level caption**：按 action ID 切分 sample，每段用 **Qwen3-VL-235B-A22B-Instruct**（MoE，A22B 表示 22B activated parameters，total 235B）以 1 FPS 采样 480p 帧生成 caption。关键 trick 是把 action/state ground-truth 注入 prompt context，弥补 VLM 对 game-specific 场景的不熟悉。

**Sample-level caption**：用 **Gemini 3 Flash** 汇总所有 action-level caption。

这种 hierarchical 设计支持 **prompt switching**——video generation 中途切换 prompt 来切换 action，这是 long video generation 的关键技术（参考 MemFlow https://arxiv.org/abs/2512.14699）。

---

## 7. WildBench：四个维度的评估

这是 paper 的另一个核心贡献。现有的 VBench 在这数据集上**饱和**了（MS、DD 都 95%+），根本区分不出 interactive 能力差异。WildBench 设计了四个维度：

### 7.1 Video Quality

复用 VBench 的 4 个 sub-metric：MS (Motion Smoothness), DD (Dynamic Degree), AQ (Aesthetic Quality), IQ (Image Quality)。这部分 paper 发现它们饱和，所以才需要新 metric。

### 7.2 Camera Control

用 **ViPE** (https://arxiv.org/abs/2508.10934) 从生成视频估计 camera trajectory，然后用 scalar alignment factor $s$ 对齐尺度（因为 game engine 的 scale 和视频估计的 scale 不一样），计算两个指标：

**Absolute Trajectory Error (translation)**：
$$\text{ATE}_{trans} = \frac{1}{T}\sum_{t=1}^{T} \| \hat{\mathbf{T}}_t - s \cdot \mathbf{T}_t \|_2$$

变量：
- $\hat{\mathbf{T}}_t \in \mathbb{R}^3$：估计的 frame $t$ 的 camera translation
- $\mathbf{T}_t \in \mathbb{R}^3$：GT translation
- $s \in \mathbb{R}^+$：scale alignment factor
- $T$：总帧数

**Relative Pose Error**：
$$\text{RPE}_t(\Delta) = \| (\hat{\mathbf{T}}_t^{-1} \hat{\mathbf{T}}_{t+\Delta}) \boxminus (\mathbf{T}_t^{-1} \mathbf{T}_{t+\Delta}) \|$$

变量：
- $\hat{\mathbf{T}}_t$：估计的 frame $t$ 位姿
- $\Delta$：frame 间隔
- $\boxminus$：SE(3) 上的相对位姿差算子

ATE 看整体轨迹准确性，RPE 对 local consistency 和 drift 更敏感。

### 7.3 Action Following

按 frame-wise action ID 把 sample 切成 segments，每段用 **Gemini 3 Flash** 做 VLM judge，prompt 按 movement / fast displacement / attack 三类定制。每段打分 1/0：

$$\text{AF} = \frac{1}{|\mathcal{S}|}\sum_{s \in \mathcal{S}} \mathbb{1}[\text{VLM judge}(v_s^{gen}, v_s^{gt}) = \text{same}]$$

变量：
- $\mathcal{S}$：所有 action segment 集合
- $v_s^{gen}, v_s^{gt}$：生成视频和 GT 视频的 segment $s$
- $\mathbb{1}[\cdot]$：指示函数

**Human-machine agreement 实测 85%**——10 个志愿者每人标 3 次，不一致 segment 弃用（约 5%）。这个 85% 说明 VLM judge 还算靠谱。

### 7.4 State Alignment — 最 novel 的 metric

用 skeleton pose 作为 state proxy。理由：pose 直接反映很多 state（比如 death pose ↔ HP=0），间接反映其他 state。

流程：
1. 从 GT 3D skeleton 投影到 2D 拿 ground-truth 2D trajectory
2. 对生成视频，用 **TAPNext** (https://tapnext.github.io/) 从第一帧（GT）初始化 keypoints 然后 track
3. 对每个 keypoint 计算 multi-threshold accuracy：

$$\text{SA} = \frac{1}{K}\sum_{k=1}^{K} \frac{1}{|\Theta|}\sum_{\tau \in \Theta} \frac{1}{T}\sum_{t=1}^{T} \mathbb{1}\left[\|\hat{\mathbf{p}}_{k,t} - \mathbf{p}_{k,t}\|_2 < \tau\right]$$

变量：
- $K$：keypoint 总数
- $\Theta = \{4, 8, 16, 32\}$：pixel 阈值集合
- $\hat{\mathbf{p}}_{k,t} \in \mathbb{R}^2$：TAPNext 预测的 keypoint $k$ 在 frame $t$ 的 2D 位置
- $\mathbf{p}_{k,t} \in \mathbb{R}^2$：GT 2D 位置
- $\tau$：距离阈值

在 GT video 上自验证：TAPNext + 这个 metric 的 SA = **43.23%**。这数字看起来不高，paper 解释说 state evolution 本身有 stochasticity（random events 等），单 sample 严格 alignment 不该 100%，但统计上有意义。这个 43% baseline 让后续 model 评估有 reference。

---

## 8. 三个 baseline 模型

### 8.1 CamCtrl — Camera-Conditioned

基于 **Wan2.2-Fun-5B-Control-Camera** (https://arxiv.org/abs/2503.20314)。输入：camera trajectory + initial image + text prompt。

Baseline 用 rule-based 把离散 camera action 转成 camera poses，再用 **Plücker embeddings** (https://arxiv.org/abs/2106.08240) 注入：

$$\mathcal{P}_t = \{(\mathbf{o}_t, \mathbf{d}_t^{(h,w)})\}_{h,w}$$

变量：
- $\mathbf{o}_t \in \mathbb{R}^3$：frame $t$ 的 camera origin
- $\mathbf{d}_t^{(h,w)} \in \mathbb{R}^3$：从 camera 指向 pixel $(h,w)$ 的单位 ray direction
- $\mathcal{P}_t$：frame $t$ 的 Plücker embedding 集合

Plücker 表示同时编码 origin 和 direction，比单纯 6DoF camera pose 更适合注入 DiT。

CamCtrl 的改进：**直接用 WildWorld 的 GT per-frame camera poses fine-tune**，跳过 rule-based 转换的损失。

### 8.2 SkelCtrl — Skeleton-Conditioned

基于 **Wan2.2-Fun-5B-Control**（video-to-video 模型）。输入：first frame + skeleton video。

skeleton video 构造：用 3D skeleton keypoints 和 joint tree，用 GT camera 投影到 2D，render 成 colored-skeleton video（colored 是为了区分不同 character 和不同 joint）。

这个 setting 给模型最强的 motion 控制信号——相当于直接告诉模型"下一帧 pose 应该长这样"。

### 8.3 StateCtrl — State-Conditioned（paper 的核心贡献）

架构拆解：

```
Discrete states (weapon_id, bank_id, ...)   Continuous states (HP, position, Atk, ...)
        │                                            │
        ▼                                            ▼
   Embedding lookup f_θ                       MLP g_φ
        │                                            │
        └──────────────┬─────────────────────────────┘
                       ▼
            Entity-level embedding e_i
                       │
                       ▼  Transformer over all entities + global
            E = Transformer([e_player, e_monster, e_npc..., e_global])
                       │
        ┌──────────────┼──────────────────────────┐
        ▼              ▼                           ▼
   Inject into    State decoder D         State predictor P
   DiT middle     (decoder loss)         (predictor loss)
   layers
        │
        ▼
   Video diffusion denoising → generated frame
```

**Key 设计要点**：

1. **Hierarchical state modeling**：entity-level 每个 entity（player/monster/NPC）独立编码自己的 state，global-level 编码 recording time 等全局 state。Transformer 处理 entity 间关系，输出 unified state embedding。

2. **Discrete vs Continuous 解耦**：
   - Discrete：$s^{disc} \in \mathbb{Z}^d$（monster type, weapon category）→ trainable embedding table lookup
   - Continuous：$s^{cont} \in \mathbb{R}^d$（coordinates, HP）→ MLP encoding 到同维度 feature space
   - 二者相加得到 entity embedding

3. **DiT 中间层注入**：state embedding 对齐到 video frame 数（per-frame injection），加到 DiT 的中间层作为 conditioning signal，类似 ControlNet / T2I-Adapter 的思路但作用在 DiT 的 cross-attention 或 AdaLN。

4. **State decoder**：从 embedding $\mathbf{E}$ recover 出 state：
$$\hat{\mathbf{s}} = D(\mathbf{E}), \quad \mathcal{L}_{dec} = \|\hat{\mathbf{s}} - \mathbf{s}\|^2$$
保证 embedding 不丢失 state 信息（类似 VAE 的 reconstruction loss）。

5. **State predictor**：预测下一帧的 state：
$$\hat{\mathbf{s}}_{t+1} = P(\mathbf{E}_t), \quad \mathcal{L}_{pred} = \|\hat{\mathbf{s}}_{t+1} - \mathbf{s}_{t+1}\|^2$$
让 embedding 学到 dynamics——从当前 state 应该能推出下一帧 state。**这就是 world model 的本质**。

6. **StateCtrl-AR**：inference 时只用 first frame 的 GT state，后续 state 由 predictor autoregressive 预测：
$$\hat{\mathbf{s}}_{t+1} = P(\mathbf{E}_t), \quad \mathbf{E}_{t+1} = \text{Encode}(\hat{\mathbf{s}}_{t+1}), \quad \hat{\mathbf{s}}_{t+2} = P(\mathbf{E}_{t+1}), \dots$$
这就形成了 latent state dynamics 的 autoregressive rollout，与 video diffusion 的 autoregressive rollout 解耦但协同。

---

## 9. 训练超参数

| 参数 | 值 |
|---|---|
| Resolution | 544 × 960 |
| Frames per sample | 81 |
| Frame rate | 16 FPS（约 5 秒视频）|
| Batch size | 1 per GPU / 8 total |
| Learning rate | $1 \times 10^{-5}$ |
| Optimizer | Adam |
| Iterations | 250,000 |
| Inference sampling steps | 50 |

81 帧 @ 16FPS = 5.06 秒，刚好覆盖一个完整 action combo（Monster Hunter 一个 motion 通常 60-90 帧 @ 30FPS，即 30-45 帧 @ 16FPS）。

---

## 10. 实验结果里有意思的点

完整结果表：

| Method | MS | DD | AQ | IQ | ATE(↓) | RPE(↓) | AF | SA |
|---|---|---|---|---|---|---|---|---|
| Baseline (Wan2.2-TI2V-5B) | 96.38 | 99.00 | 50.81 | 65.62 | 4.63 | 0.18 | 53.77 | 11.29 |
| CamCtrl | 97.85 | 97.00 | 48.29 | 62.88 | **2.02** | 0.13 | 83.46 | 15.18 |
| SkelCtrl | 97.85 | 95.00 | **47.92** | **62.43** | 2.55 | 0.10 | **92.81** | **22.03** |
| StateCtrl | 97.45 | 99.00 | **50.86** | **67.78** | **0.94** | **0.07** | 85.66 | 16.06 |
| StateCtrl-AR | 97.43 | 99.00 | 50.90 | 67.76 | 1.01 | 0.08 | 74.66 | 16.13 |

### 10.1 VBench saturation

MS 在 96-98，DD 在 95-99，几乎区分不出模型。但 AF 从 53 到 92，SA 从 11 到 22，差异巨大。**现有 video quality metric 对 interactive world model 几乎无区分度**——这是 WildBench 设计的核心 motivation。

### 10.2 Visual control vs Soft embedding 的 trade-off

SkelCtrl 在 AF (92.81) 和 SA (22.03) 上最强，但 AQ (47.92) 和 IQ (62.43) 最差。StateCtrl 在 AQ (50.86) 和 IQ (67.78) 上最强，AF (85.66) 和 SA (16.06) 中等。

直觉解释：SkelCtrl 直接告诉模型每帧 skeleton pose，模型必须严格遵循 → AF/SA 高，但渲染时受 skeleton 约束 → 像素质量被牺牲（pose stick figure 太 rigid，破坏自然 motion blur 和 detail）。StateCtrl 学 state → soft embedding，模型有自由度渲染 → 视觉质量高，但 AF/SA 不如硬约束。

这个 trade-off 提示未来方向：**hybrid control**（state 提供 coarse guide，skeleton 提供 fine guide，分层注入）可能打破这个 trade-off。

### 10.3 StateCtrl vs StateCtrl-AR

StateCtrl-AR 的 AF 从 85.66 掉到 74.66，但 SA 几乎不变（16.06 vs 16.13），Camera Control 也几乎不变（ATE 0.94 vs 1.01）。

**autoregressive state prediction 的误差主要伤害 action 表达，而非 state 演化本身**。state 是 smooth continuous variable（HP、position），AR 误差累积慢；action 是 discrete motion ID，AR 一旦预测错就切换到完全不同的 motion，错误成本高。这与 LLM AR generation 中 "early token error → whole sentence wrong" 同构。

paper 引用 **Self-Forcing** (https://arxiv.org/abs/2506.08009) 暗示这个 train-test gap 可以通过 teacher forcing 变体缓解。

### 10.4 Camera Control 的反直觉结果

StateCtrl 的 ATE 0.94 是所有方法中最低的，比 CamCtrl (2.02) 还低。StateCtrl 没有直接 condition camera，但 Camera Control 表现最好。

可能解释：StateCtrl 注入了 camera 相关的 state（位置、旋转），模型 implicit 学到了 camera 跟随的 dynamics。Monster Hunter 的 target-lock 系统让 camera 严格跟随 monster，所以 monster state 几乎 deterministic 地决定了 camera pose。这意味着**state-conditioned video generation 有可能 subsume camera control**，不需要单独的 camera condition。

---

## 11. 我的几个直觉

### 11.1 为什么 game data 是 world model 的最佳训练场

game engine 本质是一个完全可观测、可控制、可重复的 world simulator。WildWorld 把 game 的内部 state（设计上就是为 game logic 服务的）作为 supervision signal，等于把一个 RL environment 的 transition function $(s_t, a_t) \to s_{t+1}$ 的数据倒出来给 generative model 学。这是 RL 与 generative modeling 融合的最佳接口。

对比 Sora 这类从 Internet video 学的 world model：Internet video 只有 observation $o_t$，没有 state $s_t$，所以模型必须 implicit 推 state（这就是 **Latent World Model** 方向，参考 https://arxiv.org/abs/2601.17067）。但 explicit state supervision 让模型可以学到 sharper dynamics，因为不需要从 $o$ 反推 $s$。

### 11.2 State predictor 就是个 mini world model

StateCtrl 的 state predictor $P: \mathbf{E}_t \to \hat{\mathbf{s}}_{t+1}$ 实际上就是一个 latent world model（参考 Ha & Schmidhuber "World Models" https://arxiv.org/abs/1803.10122、DreamerV3 https://arxiv.org/abs/2301.04104）。整个 StateCtrl 可以看作：

- **Latent dynamics model**：$P$，在 embedding space 预测下一帧 state
- **Observation model**：DiT，从 state embedding 生成 video frame
- **Encoder**：$f_\theta + g_\phi$ + Transformer，把 raw state 编码到 embedding

这与 Dreamer 的 RSSM 结构有同构性，只是 observation model 从 VAE 换成了 diffusion model。这提示：**未来 RL 的 world model 和 video generation 的 world model 可能完全融合**——前者用作 policy training 的 simulator，后者用作 interactive video 生成。WildWorld 提供的 state annotation 让这种融合成为可能。

### 11.3 Action following 的瓶颈在 motion ID，不在 pixel

SkelCtrl 把 AF 推到 92.81，但没到 100。剩余 7.19% gap 可能来自：
- 长尾 action（top-150 占 58%，剩下 42% 数据 action 频次低，模型 under-trained）
- Skill casting 类 action 视觉效果高度依赖 particle effect，skeleton 控制不到
- Stochastic action outcome（critical hit vs normal hit）模型无法从 skeleton 区分

### 11.4 State alignment 43.23% baseline 的含义

GT video 上 TAPNext 自身 SA 只有 43.23%，说明这个 metric 设计上是"宽松"的——它衡量 tracked point 是否在阈值内，而非 skeleton 是否完全对齐。43% 意味着 43% 的 (keypoint, frame) 落在至少一个阈值内。反映：
- TAPNext 在 fast motion 下 tracking 不稳
- Multi-threshold (4, 8, 16, 32) 平均让小阈值拉低分数
- 即使 GT skeleton 也有标注噪声

作为相对 metric（比较 model 间 SA 差异）仍有意义。

### 11.5 Autoregressive state rollout 的根本困难

StateCtrl-AR 的 AF 掉 11 个点，但 SA 几乎不变。这联系到 LLM 的 "exposure bias" / "teacher forcing train-test gap"。train 时 state 是 GT，inference 时 state 是模型自己 prediction。一旦 AR 累积到某 frame state 越界，后续 action ID 切换时机就会错（比如 GT 是 frame 60 切下一 motion，AR 因 state drift 在 frame 65 才切）。

Self-Forcing 的解决方案是在 train 时也用 model 自己 prediction 作为 next input，让 train 和 inference 一致。把这个思路搬到 state predictor 上是直接改进方向。

---

## 12. 相关 reference links

- **Project page**: https://shandaai.github.io/wildworld-project/
- **Code**: https://github.com/ShandaAI/WildWorld
- **Monster Hunter Wilds 官方**: https://www.monsterhunter.com/wilds/
- **Wan video model (baseline)**: https://arxiv.org/abs/2503.20314
- **CameraCtrl**: https://arxiv.org/abs/2404.02101
- **VBench**: https://github.com/Vchitect/VBench
- **TAPNext**: https://tapnext.github.io/
- **ViPE (camera trajectory estimation)**: https://arxiv.org/abs/2508.10934
- **Plücker embeddings (Light Field Networks)**: https://arxiv.org/abs/2106.08240
- **Self-Forcing (AR video diffusion)**: https://arxiv.org/abs/2506.08009
- **World Models (Ha & Schmidhuber)**: https://arxiv.org/abs/1803.10122
- **DreamerV3**: https://arxiv.org/abs/2301.04104
- **MIND benchmark**: https://arxiv.org/abs/2602.08025
- **OmniWorld**: https://arxiv.org/abs/2509.12201
- **Sekai**: https://arxiv.org/abs/2506.15675
- **GameGen-X**: https://arxiv.org/abs/2411.00769
- **PLAICraft**: https://arxiv.org/abs/2505.12707
- **GF-Minecraft (GameFactory)**: https://arxiv.org/abs/2501.08525
- **Latent World Model view (Wang et al. 2026)**: https://arxiv.org/abs/2601.17067
- **Learning latent action world models in the wild (Garrido et al.)**: https://arxiv.org/abs/2601.05230
- **Genie (DeepMind)**: https://deepmind.google/models/genie/
- **Cosmos World Foundation Model (NVIDIA)**: https://arxiv.org/abs/2501.03575
- **MemFlow**: https://arxiv.org/abs/2512.14699

---

## 13. 一句话总结

WildWorld 把 game engine 当成一个完全可观测的 world simulator，把它的 internal state dump 出来作为 video world model 的 supervision signal，让模型从 "学 pixel shortcut" 转向 "学 state dynamics"。它揭示的几个 finding 都很有信息量：VBench 对 interactive model 饱和、visual control 和 soft embedding 之间有 trade-off、autoregressive state prediction 主要伤害 action 而非 state、state-conditioned 可能 subsume camera control。

剩下的开放问题也很清晰：能不能把 StateCtrl 的 state embedding 蒸馏成可用于 RL policy training 的 world model？hybrid control（skeleton + state + camera）能不能打破 trade-off？Self-Forcing 类方法能不能解决 AR state rollout 的 train-test gap？这些问题的答案可能决定下一代 interactive world model 的走向。

---

# WildWorld 深度解析：面向 Generative ARPG 的动态世界建模数据集

## 1. 核心动机：为什么需要 explicit state annotations

这篇 paper 的核心 insight 来自于 dynamical systems theory 和 reinforcement learning 的视角。在 POMDP (Partially Observable Markov Decision Process) 框架下，世界的演化被建模为：

$$s_{t+1} = \mathcal{T}(s_t, a_t), \quad o_t = \mathcal{O}(s_t)$$

其中 $s_t \in \mathcal{S}$ 是 latent state，$a_t \in \mathcal{A}$ 是 action，$o_t \in \mathcal{O}$ 是 observation，$\mathcal{T}$ 是 transition function，$\mathcal{O}$ 是 emission/observation function。video world model 试图从 $(o_{\le t}, a_{\le t})$ 学到 $\mathcal{T}$ 和 $\mathcal{O}$。

paper 指出现有 dataset 的根本缺陷：**action 直接绑定到 pixel-level variation**。例如 "move left" 在 video 中体现为 viewpoint pan，这种 action-observation 的强耦合让模型不需要建模 latent state 就能"作弊"地预测未来。但当 action 通过 implicit state transition 体现时（例如 "shoot" 消耗 ammunition，当 ammo=0 时再执行 "shoot" 视觉上完全不同），没有 explicit state 标注的模型就会崩溃。这种 action 与 state 解耦、state 再驱动 observation 的两层结构，正是 ARPG（如 Monster Hunter）天然具备的。

直觉上，这就好比 AlphaGo 学棋：如果只让模型看棋盘图像学下一步，它会学到很多视觉捷径；但如果给它 board state 的 structured representation，它就能学到真正的 game dynamics。WildWorld 的设计哲学就是给 video world model 提供 board state。

---

## 2. WildWorld Dataset 规模与结构

### 2.1 核心统计

| 维度 | 数值 |
|---|---|
| Total frames | 108M |
| Annotation columns/frame | 119 |
| Character actions (triplets) | 5,960 |
| Monster action pairs | 2,132 |
| Motion IDs (player) | 455 (across 24 banks) |
| Motion IDs (monster) | 527 (across 13 banks) |
| Monster species | 29 |
| Player characters | 4 |
| Weapon types | 4 (Great Sword, Long Sword, Bow, Dual Blades) |
| Stages | 5 |
| Top-150 action coverage | 58.49% (long-tail) |
| Combat vs Travel ratio | 66% : 34% |

Character action 用一个 triplet 编码：

$$a_t^{char} = (\text{weapon\_id}, \text{bank\_id}, \text{motion\_id}) \in \mathbb{Z}^3$$

这种结构化编码方式让 action space 是一个 hierarchy：weapon 选定后 bank 决定动作类别（如站立、攻击、回避），motion 决定具体动画。这种 hierarchical action space 的 cardinality 是 $\prod_i |\mathcal{W}_i| \times |\mathcal{B}_i| \times |\mathcal{M}_i|$，远比 flat discrete action space 信息丰富。

### 2.2 Per-frame Annotation 内容

每一帧记录 119 列结构化数据，包括：
- **Actions**: 执行的 action ID、animation frame 进度
- **States (player)**: type_id, weapon_id, motion_id, animation_frame, Atk (attack), Wp (weapon power), Def (defense), HP, 位置 $(x, y, z)$，旋转 quaternion $(q_w, q_x, q_y, q_z)$，looking point
- **States (monster)**: type_id, motion_id, HP, animation_frame, 位置、旋转
- **States (NPC)**: 同上
- **Skeleton**: player 和 monster 的 3D keypoints 与 joint tree
- **Camera**: intrinsics $K \in \mathbb{R}^{3\times 3}$，extrinsics $[R|t] \in \mathbb{R}^{3\times 4}$
- **Observations**: RGB frame (720p sub-window, HEVC lossy), depth map (lossless)

注意 HP 在 caption 示例中是 `HP=14131.85/42336`，这种 continuous fraction 形式特别适合训练 world model 学 "damage accumulation" 的 dynamics。

---

## 3. Data Acquisition Pipeline 技术细节

### 3.1 渲染管线插桩

Monster Hunter: Wilds 使用现代 game engine（推测为 RE Engine，Capcom 自研）。game engine 把 player input 经过 game logic 更新 world state，rendering pipeline 消费 state 生成 imagery。paper 的关键设计是**在这两个 stage 都插桩**：
- engine tick 阶段：序列化 action/state 到 JSON
- rendering 阶段：通过 Reshade shader hook 拿到 RGB buffer 和 depth buffer

HUD 移除是关键技巧：通过 disable late-stage shaders 直接拿到无 HUD 的干净帧，避免了后处理需要 inpainting 去除 HUD 的麻烦。

### 3.2 多流同步录制

Reshade shader 把 2K 全屏分成 4 个 sub-window，其中 2 个分别显示 RGB 和 depth，再用魔改的 OBS Studio 分别录制。RGB 用 lossy HEVC VBR (16 Mbps target, 20 Mbps max)，depth 用 HEVC lossless with B-frames（实际码率约 20 Mbps）。

帧间同步靠 **embedded timestamps**：所有 stream 都嵌入 timestamp，post-processing 阶段用 timestamp 对齐。这是处理 multi-modal time series 的经典做法，避免了硬件级 genlock 的成本。

### 3.3 自动化 gameplay

paper 利用 game 自带的 rule-based companion AI 作为 bot。这有几个优势：
1. **零训练成本**：直接复用 Capcom 调优过的 behavior tree
2. **多样性足够**：behavior tree 在高维 state space 下产生 divergent trajectories，加上 monster 本身 stochastic 的反应，session 间 trajectory 差异大
3. **Camera 自动管理**：用 game 的 target-lock 系统，camera 动态调整以保持 monster 在视野内

quest 选择通过 UI 程序化导航完成，随机采样 quest-NPC 组合保证 map/monster/team 多样性。

---

## 4. Data Filtering 的五个维度

过滤规则值得仔细看，因为它们揭示了 interactive video model 训练的实际痛点：

| Filter | 阈值 | 物理意义 |
|---|---|---|
| Duration | $T \ge 81$ frames | 81 frames ≈ 5s at 16 FPS，短于这个长度的 sample 对 long-horizon 建模价值低 |
| Temporal continuity | $\Delta t \le 1.5 \times \Delta t_{target}$ | 在 30 FPS 下 $\Delta t_{target} \approx 33$ms，阈值约 50ms，超过则可能是 stutter 或 cutscene |
| Luminance | 15 frames 内不持续极亮/极暗 | 极端亮度对训练稳定性有害，且 game nighttime 场景 contrast 不足 |
| Camera occlusion | spring-arm 不持续 contract | 第三人称 camera 的 spring-arm 在被遮挡时收缩，camera-character 距离异常小 |
| Character occlusion | 首 frame 2D skeleton overlap $< 30\%$ | 避免首帧 character 重叠导致 image-to-video 起始条件歧义 |

Character occlusion filter 的具体计算是把 3D skeleton 用 ground-truth camera 投影到 2D：

$$\mathbf{p}_{2D}^{(j)} = \pi(K[R|t] \mathbf{p}_{3D}^{(j)})$$

其中 $\pi$ 是 perspective projection，然后计算两个 character 的 projected 2D area overlap。这是一个聪明的设计，因为直接用 image segmentation 算 overlap 不可靠（character 互相遮挡时分割会失败），但 skeleton projection 是 deterministic 的。

---

## 5. Hierarchical Caption Annotation

caption 分两层：

**Action-level caption**：以 action ID 切分 sample，每段用 Qwen3-VL-235B-A22B-Instruct（MoE，A22B 表示 22B activated parameters，total 235B）以 1 FPS 采样 480p 帧生成 caption。关键是**把 action/state ground-truth 注入 prompt context**，弥补 VLM 对 game-specific 场景的不熟悉。

**Sample-level caption**：用 Gemini 3 Flash 汇总所有 action-level caption。

这种 hierarchical 设计支持 prompt switching（参考 MemFlow [24, 50]）— 即在 video generation 中途切换 prompt 来切换 action，这是 long video generation 的关键技术。

---

## 6. WildBench：四个维度的评估

### 6.1 Video Quality (VBench metrics)

来自 VBench [22]，包含 4 个 sub-metric：
- **MS (Motion Smoothness)**：检测 frame 间的运动平滑度和物理合理性
- **DD (Dynamic Degree)**：motion magnitude，惩罚过 static 的 video
- **AQ (Aesthetic Quality)**：艺术和视觉吸引力
- **IQ (Image Quality)**：low-level distortions 如 over-exposure, noise, blur

paper 发现 MS 和 DD 在 WildWorld 上 **饱和**（>95%），这意味着 VBench 无法区分 interaction 能力强的模型，是 WildBench 设计的 motivation。

### 6.2 Camera Control

用 ViPE [19] 从生成 video 估计 camera trajectory，再用 scalar alignment factor $s$ 对齐到 game engine 的尺度，然后计算：

**Absolute Trajectory Error (translation)**：
$$\text{ATE}_{trans} = \frac{1}{T}\sum_{t=1}^{T} \| \hat{\mathbf{T}}_t - s \cdot \mathbf{T}_t \|_2$$

其中 $\hat{\mathbf{T}}_t \in \mathbb{R}^3$ 是 estimated translation at frame $t$，$\mathbf{T}_t \in \mathbb{R}^3$ 是 ground truth，$s \in \mathbb{R}^+$ 是 scale alignment factor。

**Relative Pose Error**：
$$\text{RPE}_t(\Delta) = \| (\hat{\mathbf{T}}_t^{-1} \hat{\mathbf{T}}_{t+\Delta}) \boxminus (\mathbf{T}_t^{-1} \mathbf{T}_{t+\Delta}) \|$$

这里 $\boxminus$ 表示 SE(3) 上的相对位姿差，RPE 对 local consistency 和 drift 更敏感。RPE 也分 translation 和 rotation 两个 component。

### 6.3 Action Following

按 frame-wise action ID 把 sample 切成 segments（同 action ID 的连续帧为一段），对每段用 Gemini 3 Flash 做 VLM judge，prompt 按三类 action 定制：
- **Movement**：移动类
- **Fast displacement**：突进/位移类
- **Attack**：攻击类

每段打分 1/0，最终 score 是所有 segment 的平均：

$$\text{AF} = \frac{1}{|\mathcal{S}|}\sum_{s \in \mathcal{S}} \mathbb{1}[\text{VLM judge}(v_s^{gen}, v_s^{gt}) = \text{same}]$$

**Human-machine agreement 实测 85%**，10 个志愿者每人 3 个 annotation per segment，不一致 segment 弃用（约 5%）。

### 6.4 State Alignment

这是 paper 最 novel 的 metric。用 skeleton pose 作为 state proxy，理由是 pose 直接反映许多 state（如 death pose ↔ HP=0），间接反映其他 state。

流程：
1. 从 ground truth 3D skeleton 投影到 2D screen coordinates 拿到 ground-truth 2D trajectory
2. 对生成 video，用 **TAPNext** [56] 从第一帧（GT）初始化 keypoints 然后 track
3. 对每个 keypoint 计算 multi-threshold accuracy：

$$\text{SA} = \frac{1}{K}\sum_{k=1}^{K} \frac{1}{|\Theta|}\sum_{\tau \in \Theta} \frac{1}{T}\sum_{t=1}^{T} \mathbb{1}\left[\|\hat{\mathbf{p}}_{k,t} - \mathbf{p}_{k,t}\|_2 < \tau\right]$$

其中 $K$ 是 keypoint 数，$\Theta = \{4, 8, 16, 32\}$ pixels 是阈值集合，$\hat{\mathbf{p}}_{k,t}$ 是 TAPNext 预测的 2D 位置，$\mathbf{p}_{k,t}$ 是 GT 2D 位置。

在 GT video 上自验证：TAPNext + 这个 metric 的 SA = 43.23%，这个数字看起来不高，但 paper 解释是 state evolution 本身有 stochasticity（random events 等），单 sample 上严格 alignment 不该 100%，但统计上有意义。这个 baseline 数字让后续 model 评估有 reference。

---

## 7. 三类 Baseline 模型架构

### 7.1 CamCtrl (Camera-Conditioned)

基于 **Wan2.2-Fun-5B-Control-Camera** [44]（5B 参数 video diffusion model）。输入：camera trajectory + initial image + text prompt。

Baseline 用 rule-based 把离散 camera action 转成 camera poses，再用 **Plücker embeddings** [39] 注入：

$$\mathcal{P}_t = \{(\mathbf{o}_t, \mathbf{d}_t^{(h,w)})\}_{h,w}$$

其中 $\mathbf{o}_t \in \mathbb{R}^3$ 是 frame $t$ 的 camera origin，$\mathbf{d}_t^{(h,w)} \in \mathbb{R}^3$ 是从 camera 指向 pixel $(h,w)$ 的单位 ray direction。Plücker 表示同时编码 origin 和 direction，比单纯用 6DoF camera pose 更适合注入 DiT。

CamCtrl 的改进：**直接用 WildWorld 的 ground-truth per-frame camera poses fine-tune**，跳过 rule-based 转换的损失。

### 7.2 SkelCtrl (Skeleton-Conditioned)

基于 **Wan2.2-Fun-5B-Control** [44]（video-to-video 模型）。输入：first frame + skeleton video。

skeleton video 构造：用 3D skeleton keypoints 和 joint tree，用 GT camera 投影到 2D，render 成 colored-skeleton video（colored 是为了区分不同 character 和不同 joint）。

这个 setting 给模型最强的 motion 控制信号，相当于直接告诉模型 "下一帧 pose 应该长这样"。

### 7.3 StateCtrl (State-Conditioned) — paper 的核心贡献

架构图（文字描述）：

```
                    ┌─────────────────────────────────────────────┐
                    │  Discrete states                            │
                    │   s_disc_i ∈ {weapon_id, bank_id, ...}      │
                    │   → Embedding lookup f_θ                    │
                    └─────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────────────────────┐
                    │  Continuous states                          │
                    │   s_cont_i ∈ R^d (HP, position, Atk, ...)   │
                    │   → MLP g_φ                                 │
                    └─────────────────────────────────────────────┘
                                  │
                                  ▼
              Entity-level embedding e_i = f_θ(s_disc_i) + g_φ(s_cont_i)
                                  │
                                  ▼  Transformer over all entities + global
              ┌──────────────────────────────────────────────────────┐
              │  E = Transformer([e_player, e_monster, e_npc..., e_global]) │
              └──────────────────────────────────────────────────────┘
                                  │
              ┌───────────────────┼──────────────────────────────┐
              ▼                   ▼                              ▼
        Inject into DiT     State decoder D      State predictor P
        intermediate layers (decoder loss)    (predictor loss)
              │
              ▼
        Video diffusion denoising → generated frame
```

**Key 设计要点**：

1. **Hierarchical state modeling**：entity-level 每个 entity（player/monster/NPC）独立编码自己的 state，global-level 编码 recording time 等全局 state。Transformer 处理 entity 间关系，输出 unified state embedding。

2. **Discrete vs Continuous 解耦**：
   - Discrete：$s^{disc} \in \mathbb{Z}^d$（monster type, weapon category 等）→ trainable embedding table lookup
   - Continuous：$s^{cont} \in \mathbb{R}^d$（coordinates, HP 等）→ MLP encoding 到同维度 feature space
   - 二者相加得到 entity embedding

3. **DiT 中间层注入**：state embedding 对齐到 video frame 数（per-frame injection），加到 DiT 的中间层作为 conditioning signal，类似 ControlNet / T2I-Adapter 的思路但作用在 DiT 的 cross-attention 或 AdaLN。

4. **State decoder**：从 embedding $\mathbf{E}$ recover 出 state：
$$\hat{\mathbf{s}} = D(\mathbf{E}), \quad \mathcal{L}_{dec} = \|\hat{\mathbf{s}} - \mathbf{s}\|^2$$

   这个 loss 保证 embedding 不丢失 state 信息（类似 VAE 的 reconstruction loss）。

5. **State predictor**：预测下一帧的 state：
$$\hat{\mathbf{s}}_{t+1} = P(\mathbf{E}_t), \quad \mathcal{L}_{pred} = \|\hat{\mathbf{s}}_{t+1} - \mathbf{s}_{t+1}\|^2$$

   这个 loss 让 embedding 学到 dynamics——即从当前 state 应该能推出下一帧 state。这是 world model 的本质。

6. **StateCtrl-AR**：inference 时只用 first frame 的 GT state，后续 state 由 predictor autoregressive 预测：
$$\hat{\mathbf{s}}_{t+1} = P(\mathbf{E}_t), \quad \mathbf{E}_{t+1} = \text{Encode}(\hat{\mathbf{s}}_{t+1}), \quad \hat{\mathbf{s}}_{t+2} = P(\mathbf{E}_{t+1}), \dots$$

   这就形成了 latent state dynamics 的 autoregressive rollout，与 video diffusion 的 autoregressive rollout 解耦但协同。

---

## 8. 训练细节

| Hyperparameter | Value |
|---|---|
| Resolution | 544 × 960 |
| Frames per sample | 81 |
| Frame rate | 16 FPS (≈5s video) |
| Batch size | 1 (per GPU) / 8 (total, gradient accumulation 推测) |
| Learning rate | $1 \times 10^{-5}$ |
| Optimizer | Adam |
| Iterations | 250,000 |
| Inference sampling steps | 50 |

81 frames at 16 FPS = 5.06 秒。这个长度刚好能覆盖一个完整 action combo（Monster Hunter 中一个 motion 通常 60-90 frames at 30 FPS，即 30-45 frames at 16 FPS）。

---

## 9. 实验结果深度解读

完整结果表：

| Method | MS | DD | AQ | IQ | ATE(↓) | RPE(↓) | AF | SA |
|---|---|---|---|---|---|---|---|---|
| Baseline (Wan2.2-TI2V-5B) | 96.38 | 99.00 | 50.81 | 65.62 | 4.63 | 0.18 | 53.77 | 11.29 |
| CamCtrl | 97.85 | 97.00 | 48.29 | 62.88 | **2.02** | 0.13 | 83.46 | 15.18 |
| SkelCtrl | 97.85 | 95.00 | **47.92** | **62.43** | 2.55 | 0.10 | **92.81** | **22.03** |
| StateCtrl | 97.45 | 99.00 | **50.86** | **67.78** | **0.94** | **0.07** | 85.66 | 16.06 |
| StateCtrl-AR | 97.43 | 99.00 | 50.90 | 67.76 | 1.01 | 0.08 | 74.66 | 16.13 |

### 9.1 几个关键 takeaways

**(1) VBench saturation 问题**

MS 在 96-98 区间，DD 在 95-99 区间，几乎无法区分模型。但 AF 从 53 到 92，SA 从 11 到 22，差异巨大。这是 paper 强调 WildBench 必要性的核心证据：**现有的 video quality metric 对 interactive world model 几乎无区分度**。

**(2) Visual control 信号 vs Soft embedding 的 trade-off**

SkelCtrl 在 AF (92.81) 和 SA (22.03) 上最强，但 AQ (47.92) 和 IQ (62.43) 最差。StateCtrl 在 AQ (50.86) 和 IQ (67.78) 上最强，AF (85.66) 和 SA (16.06) 中等。

直觉解释：SkelCtrl 直接告诉模型每帧的 skeleton pose，模型必须严格遵循 → AF/SA 高，但渲染时受 skeleton 约束 → 像素质量被牺牲（pose stick figure 太 rigid，破坏自然 motion blur 和 detail）。StateCtrl 学的是 state → soft embedding，模型有自由度渲染 → 视觉质量高，但 AF/SA 不如硬约束。

这就是 paper 中"directly using visual signals as conditional input yield a trade-off"的核心。这个 trade-off 提示了未来方向：**如何在保持 pixel quality 的同时获得 strong control**——可能是 hybrid（state 提供 coarse guide，skeleton 提供 fine guide，分层注入）。

**(3) StateCtrl vs StateCtrl-AR**

StateCtrl-AR 的 AF 从 85.66 掉到 74.66，但 SA 几乎不变（16.06 vs 16.13），Camera Control 也几乎不变（ATE 0.94 vs 1.01）。

这个发现非常有意思：**autoregressive state prediction 的误差主要伤害 action 表达，而非 state 演化本身**。直觉上，state 是 smooth 的 continuous variable（HP、position），AR 误差累积慢；但 action 是 discrete motion ID，AR 一旦预测错就会切换到完全不同的 motion，错误成本高。这与 LLM AR generation 中 "early token error → whole sentence wrong" 的现象同构。

paper 引用 Self-Forcing [21] 暗示这个 train-test gap 问题可以通过 teacher forcing 的变体缓解。

**(4) Camera Control 的提升**

StateCtrl 的 ATE 0.94 是所有方法中最低的，比 CamCtrl (2.02) 还低。这个结果有点反直觉——StateCtrl 没有直接 condition camera，但 Camera Control 表现最好？

可能解释：StateCtrl 注入了 camera 相关的 state（位置、旋转），模型 implicit 学到了 camera 跟随的 dynamics。Monster Hunter 的 target-lock 系统让 camera 严格跟随 monster，所以 monster state 几乎 deterministic 地决定了 camera pose。这意味着**state-conditioned video generation 有可能 subsume camera control**，不需要单独的 camera condition。

---

## 10. 与现有 dataset 的对比

paper Section 2.2 列举了相关 dataset：

| Dataset | Action | State | 来源 | 规模 |
|---|---|---|---|---|
| OpenVid-1M [36] | ❌ | ❌ | Internet video | 1M |
| MiraData [25] | ❌ | ❌ | Internet video | - |
| Open-Sora [33] | ❌ | ❌ | Internet video | - |
| SpatialVID [45] | Camera | Partial spatial | - | - |
| Sekai [31] | Limited movement | ❌ | - | - |
| GF-Minecraft [53] | Game action | ❌ | Minecraft | - |
| PLAICraft [18] | Keyboard/mouse | ❌ | GTA V | - |
| GameGen-X [4] | Game action | ❌ | Game | - |
| OmniWorld [57] | Multi-modal | ❌ | Multi-domain | - |
| MIND [52] | Action | Memory consistency | Benchmark | - |
| **WildWorld** | **450+ actions** | **Skeleton, world state, camera, depth** | **Monster Hunter Wilds** | **108M frames** |

WildWorld 的差异化在于：**explicit state 而非 implicit latent state**。MIND [52] 也强调 memory consistency，但 WildWorld 把 state 完全 explicit 化，使得可以定量 evaluate state alignment。

---

## 11. 我对这篇 paper 的几点 intuition

### 11.1 为什么 game data 是 world model 的最佳训练场

game engine 本质是一个完全可观测、可控制、可重复的 world simulator。WildWorld 把 game 的内部 state（设计上就是为 game logic 服务的）作为 supervision signal，等于把一个 RL environment 的 transition function $(s_t, a_t) \to s_{t+1}$ 的数据倒出来给 generative model 学。这是 RL 与 generative modeling 融合的最佳接口。

对比 Sora 这类从 Internet video 学的 world model：Internet video 只有 observation $o_t$，没有 state $s_t$，所以模型必须 implicit 推 state（这就是 Latent World Model [46, 54, 10, 32, 27] 的方向）。但 explicit state supervision 让模型可以学到 sharper dynamics，因为不需要从 $o$ 反推 $s$。

### 11.2 State predictor 是 mini world model

StateCtrl 的 state predictor $P: \mathbf{E}_t \to \hat{\mathbf{s}}_{t+1}$ 实际上就是一个 latent world model（参考 Ha & Schmidhuber "World Models" [11]、DreamerV3 [13]）。整个 StateCtrl 可以看作：

- **Latent dynamics model**: $P$，在 embedding space 预测下一帧 state
- **Observation model**: DiT，从 state embedding 生成 video frame
- **Encoder**: $f_\theta + g_\phi$ + Transformer，把 raw state 编码到 embedding

这与 Dreamer 的 RSSM 结构有同构性，只是 observation model 从 VAE 换成了 diffusion model。这种架构上的同构性提示：**未来 RL 的 world model 和 video generation 的 world model 可能完全融合**——前者用作 policy training 的 simulator，后者用作 interactive video 生成。WildWorld 提供的 state annotation 让这种融合成为可能。

### 11.3 Action following 的瓶颈在 motion ID，不在 pixel

SkelCtrl 把 AF 推到 92.81，但没到 100。剩余的 7.19% gap 是什么？看 paper Section 5.3 的 qualitative 分析：SkelCtrl 在 ground truth 有 sand/gravel splash occlusion 时反而能复现，说明 skeleton 控制了 character motion 但模型有空间重渲 detail。剩下 gap 可能来自：
- 长尾 action（top-150 占 58%，剩下 42% 数据 action 频次低，模型 under-trained）
- Skill casting 类 action 视觉效果高度依赖 particle effect，skeleton 控制不到
- Stochastic action outcome（critical hit vs normal hit）模型无法从 skeleton 区分

### 11.4 State alignment 43.23% baseline 的含义

GT video 上 TAPNext 自身 SA 只有 43.23%，说明这个 metric 设计上是"宽松"的——它衡量的是 "tracked point 是否在阈值内"，而不是 "skeleton 是否完全对齐"。43% 意味着 43% 的 (keypoint, frame) 落在至少一个阈值内。这个数字反映了：
- TAPNext 在 fast motion 下 tracking 不稳
- Multi-threshold (4, 8, 16, 32) 平均让小阈值拉低分数
- 即使 GT skeleton 也有标注噪声

但作为相对 metric（比较 model 间的 SA 差异）仍有意义，paper 用它做的是相对比较。

### 11.5 Autoregressive state rollout 的根本困难

StateCtrl-AR 的 AF 掉 11 个点，但 SA 几乎不变。这个现象可以联系到 LLM 的 "exposure bias" / "teacher forcing train-test gap"。在 train 时 state 是 GT，inference 时 state 是模型自己的 prediction。一旦 AR 累积到某个 frame 的 state 越界，后续的 action ID 切换时机就会错（比如 GT 是 frame 60 切到下一个 motion，AR 因 state drift 在 frame 65 才切）。

Self-Forcing [21] 的解决方案是在 train 时也用 model 自己的 prediction 作为 next input，让 train 和 inference 一致。把这个思路搬到 state predictor 上是直接的改进方向。

---

## 12. 相关 reference links

- **Project page**: https://shandaai.github.io/wildworld-project/
- **Code**: https://github.com/ShandaAI/WildWorld
- **Monster Hunter Wilds 官方**: https://www.monsterhunter.com/wilds/
- **Wan video model (baseline)**: https://arxiv.org/abs/2503.20314
- **CameraCtrl**: https://arxiv.org/abs/2404.02101
- **VBench**: https://github.com/Vchitect/VBench
- **TAPNext**: https://tapnext.github.io/
- **ViPE (camera trajectory estimation)**: https://arxiv.org/abs/2508.10934
- **Plücker embeddings (Light Field Networks)**: https://arxiv.org/abs/2106.08240
- **Self-Forcing (AR video diffusion)**: https://arxiv.org/abs/2506.08009
- **World Models (Ha & Schmidhuber)**: https://arxiv.org/abs/1803.10122
- **DreamerV3**: https://arxiv.org/abs/2301.04104
- **MIND benchmark**: https://arxiv.org/abs/2602.08025
- **OmniWorld**: https://arxiv.org/abs/2509.12201
- **Sekai**: https://arxiv.org/abs/2506.15675
- **GameGen-X**: https://arxiv.org/abs/2411.00769
- **PLAICraft**: https://arxiv.org/abs/2505.12707
- **GF-Minecraft**: https://arxiv.org/abs/2501.08525 (GameFactory paper)
- **Latent World Model view (Wang et al. 2026)**: https://arxiv.org/abs/2601.17067
- **Learning latent action world models in the wild (Garrido et al.)**: https://arxiv.org/abs/2601.05230
- **Genie (DeepMind)**: https://deepmind.google/models/genie/
- **Cosmos World Foundation Model (NVIDIA)**: https://arxiv.org/abs/2501.03575

---

## 13. 总结：这篇 paper 的位置

WildWorld 不是单点突破，而是把几条 line of work 拼接起来：
1. **Game data → world model training**（GameGen-X, GF-Minecraft, PLAICraft 已经开始）
2. **Explicit state annotation**（MIND 强调 memory，但没有 explicit state）
3. **State-aware video generation**（Latent World Models 隐式，WildWorld 显式）
4. **ARPG 作为 benchmark domain**（之前主要 Minecraft / GTA，WildWorld 第一次用 photorealistic AAA）

它的真正价值在于**为下一代 interactive world model 提供了清晰的 supervision signal 和 evaluation metric**，让 "video diffusion 是否真的学到了 world dynamics" 这个问题从定性讨论变成定量 evaluate。

剩下最值得关注的开放问题：
- 能否把 StateCtrl 的 state embedding 蒸馏成一个真正可用于 RL policy training 的 world model？
- State annotation 的 collection 成本（依赖 game engine 插桩）如何泛化到非 game 场景（如 robotics）？
- Hybrid control（skeleton + state + camera）能否打破 paper 中揭示的 trade-off？
- Autoregressive state prediction 的 train-test gap 能否用 Self-Forcing 类方法解决？

paper 的结论相当 honest：current models 在 long-horizon state consistency 上仍有持续挑战，这是后续工作的明确方向。
