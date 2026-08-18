---
source_pdf: Decoupled Diffusion Sparks Adaptive Scene Generation.pdf
paper_sha256: b42c52f4d2a6643837fbbb34fd61b8755f36d3e19bdf1aa8c47069956d2e94fe
processed_at: '2026-08-18T04:38:44-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Nexus 论文的大白话解读

## 一句话概括

**让 AI 导演交通场景**: 你给它一个起点和终点, 它能编出一个合理的交通故事; 如果中途某个车突然变道, 它能马上改剧本, 而不用从头重写。

---

## 1. 这篇 paper 在解决什么问题?

想象你在训练一个自动驾驶系统。你需要给它看各种各样的交通场景 — 正常行驶、加塞、急刹、碰撞。但现实里你不可能真去撞车收集数据, 所以需要一个 "场景生成器" 来编故事。

现有的生成器有两种路子, 各有各的毛病:

### 路子 A: 一次性生成整段未来 (Full-sequence Diffusion)

就像让 AI 一次性写完一整集电视剧。好处是你可以说 "最后要让 A 车撞上 B 车", AI 就往这个结局编。坏处是, 如果拍到一半导演突然说 "C 车改左转", AI 得把后面整集剧本撕了重写, 来不及反应。

代表: SceneDiffuser, Diffusion Policy

### 路子 B: 一帧一帧往后推 (Next-token Prediction)

就像写连环画, 画一页看一页。好处是随时能根据新情况调整下一页。坏处是 AI 根本不知道结局是什么, 你没法指挥它 "最后撞车", 因为它只看眼前。

代表: GUMP, MotionLM, Trajeglish

**Nexus 说: 我全都要** — 既能看结局, 又能随时改。

---

## 2. 核心 Idea: 噪声就是面具

这是整篇论文最聪明的洞察, 我换个方式讲。

Diffusion model 的工作原理是: 把一张干净的图逐步加噪声变成糊图, 然后训练 AI 反向去噪还原。传统做法是整张图加一样的噪声 — 要么全糊, 要么全干净。

Nexus 说: 凭什么要一样? **每个 token (每个车在每一时刻的状态) 加不同级别的噪声**:
- 噪声 = 0: 这帧已经定了, 是 "已知条件" (比如历史、终点)
- 噪声 = 1: 纯随机, 这帧待生成
- 噪声 = 0.5: 半确定, AI 可以微调

这就像给每个时间点戴不同透明度的面具。透明的部分是确定的, 不透明的部分让 AI 发挥。

**为什么这样就能两全?**

训练时让 AI 见各种 "部分确定 + 部分待填" 的组合, 它就学会了两件事:
1. 从已知推未知 (goal orientation)
2. 随时把某个 token 的噪声调高, 就能 "擦掉" 那帧重新画 (reactivity)

这个想法跟 MIT 的 [Diffusion Forcing](https://arxiv.org/abs/2407.01392) 不谋而合, 但 Nexus 是第一次把它做到 multi-agent 交通仿真里, 还加了 scheduling 机制让 online 交互真正 work。

---

## 3. 具体怎么训练?

### 数据表示

先把交通场景编码成两个 tensor:

**Agent tensor** $\mathbf{x} \in \mathbb{R}^{A \times T \times D}$
- $A$: 最多多少辆车 (比如 32)
- $T$: 多少个时间步 (比如 21 帧, 10 秒 @ 2Hz)
- $D$: 每辆车的属性维度, paper 里是 8 维: $(x, y, \sin\alpha, \cos\alpha, v_x, v_y, l, w)$
  - $x, y$: 位置坐标
  - $\sin\alpha, \cos\alpha$: 朝向 (用 sin cos 而不是角度, 避免 0° 和 360° 不连续)
  - $v_x, v_y$: 速度分量
  - $l, w$: 车长车宽

**Map tensor** $\mathbf{c} \in \mathbb{R}^{L \times N \times D'}$
- $L$: 多少条车道
- $N$: 每条车道多少个点
- $D'$: lane 的属性 (坐标、类型)

### 训练目标

传统 diffusion 训练 (Eq 2):
$$\min_\theta \mathbb{E}\|(\epsilon - \epsilon_\theta(\mathbf{x}^t; \mathbf{c}, t)) \odot \mathbf{m}\|_2^2$$

- $t$: 全局 denoising step (标量, 所有 token 共享)
- $\mathbf{m}$: valid mask (二值, 标记哪些 agent 有效)
- 问题: 所有 token 同一时刻 $t$ 一样

Nexus 训练 (Eq 4):
$$\min_\theta \mathbb{E}\|(\epsilon - \epsilon_\theta(g(\mathbf{x}^0, \mathbf{k}); \mathbf{c}, \mathbf{k}))\|_2^2$$

- $\mathbf{k} = [k_{a,\tau}] \in (0,1]^{A \times T}$: **每个 agent 每帧独立的噪声级别**, 是矩阵
- $g$: 根据 $\mathbf{k}$ 给每个 token 独立加噪的函数
- $\mathbf{k}$ 本身作为模型的**输入**, 让模型知道 "每个 token 现在有多确定"

**训练数据怎么造?**
- 每个 scenario 随机采样一个 $\mathbf{k}$ 矩阵
- 每个 $k_{a,\tau}$ 独立从 (0, 1] 均匀采
- 模型学会从任意 "部分确定" 的状态重建完整 sequence

**直觉**: 这本质上就是 [MAE (Masked Autoencoder)](https://arxiv.org/abs/2111.06377) 的思路, 只是用 noise level 替代 binary mask。低噪声 token 提供 "context", 高噪声 token 是 "query", 模型通过 attention 把 context 信息传递给 query 来重建。

---

## 4. 架构: 怎么实现?

参考 [paper 的 Tab. 7](https://opendrivelab.com/Nexus), 架构基于 [DiT (Diffusion Transformer)](https://arxiv.org/abs/2212.09748):

```
输入:
  Agent tensor (A×T×25) → Linear → (A×T×256) token embeddings
  Map tensor → Perceiver IO → 固定长度 map tokens

主干 (4 层, 每层):
  1. Map Cross-Attention: agent tokens 查询 map (车道信息)
  2. TemporalBlock: 沿时间轴 self-attention (同一辆车不同时刻的关系)
  3. SpatialBlock: 沿 agent 轴 self-attention (同一时刻不同车的关系)
  4. FeedForward MLP (256→1024→256)

位置编码: 2D Rotary (物理时间 τ + denoising step k)
输出: Linear(256, 8) → 预测的噪声
```

几个关键设计点:

### 4.1 为什么用 2D Rotary Positional Embedding?

模型需要知道两件事:
- 这个 token 是哪辆车的哪个时刻 (physical time)
- 这个 token 现在有多 "确定" (denoising step $k$)

两者都要编码进 positional embedding, 否则模型分不清 "噪声级别 0.3 的第 5 帧" 和 "噪声级别 0.7 的第 3 帧"。

### 4.2 为什么分开 Temporal 和 Spatial attention?

如果用单个 3D attention, cost 是 $O((AT)^2)$ — 假设 32 车 × 21 帧 = 672 token, attention matrix 是 672×672。

拆成 temporal (沿时间) + spatial (沿 agent):
- Temporal: $O(A \times T^2)$ = 32 × 441 = 14K
- Spatial: $O(T \times A^2)$ = 21 × 1024 = 21K
- 总和 ~35K vs 672² = 451K, **节省 13×**

这是 [MotionLM](https://arxiv.org/abs/2309.16534) 的做法, Nexus 沿用了。

### 4.3 为什么用 Perceiver IO 处理 map?

Map 的大小不固定 (lane 数量、点数都变), 直接 cross-attention 会 cost 爆炸。Perceiver IO 用一组 learnable queries 把 map 压成固定长度的 latent, 然后 agent tokens 再 attend 这个 latent。

参考 [Perceiver IO paper](https://arxiv.org/abs/2107.14795)。

---

## 5. 采样: 怎么做到实时反应?

训练搞定了 "给条件", 采样要搞定 "反应快"。这里 Nexus 发明了 **chunk-based pipelined sampling**。

### 5.1 什么是 Chunk?

一个 chunk 是场景的一个局部子集, 包含三类 token:
- **History**: 噪声 = 0, 已经确定的历史帧
- **Active**: 噪声在 (0, 1) 之间, 正在被 denoise 的未来帧
- **Goal**: 噪声 = 0, 终点 (可选)

每次 denoising step 做三件事:
1. Active tokens 的噪声都降一点
2. 噪声降到 0 的 token 被 **pop** (出 chunk, 成为确定的)
3. 新的高噪声 token 被 **push** (进 chunk, 开始被 denoise)

整个 chunk 沿时间轴滑动, 像流水线一样。

### 5.2 两种 Scheduling 策略

paper 提了两种:

**Pyramidal (金字塔形)** — 单向流动
- Chunk 从小到大增长, 每步加一个新帧
- 只往 future 方向走
- 适合 free exploration

**Trapezoidal (梯形)** — 双向流动
- Tokens 从两端进入和退出
- Goal 在右端, History 在左端, 中间是 active
- Goal 的信息可以通过 attention 反向传给所有 agent
- 适合 goal-conditioned generation

### 5.3 为什么这样能 react?

传统 full-sequence diffusion 一旦 agent 改决策, 整个 future 都要重新生成, 等几十秒。

Nexus 的做法: 检测到 agent 改变, **直接把对应 token 的噪声调高** (相当于 "擦掉" 这帧的确定状态), 下一个 denoising step 自然会重新生成它, 不影响其他已确定的帧。

实测反应时间: **0.16 秒** (Tab. 2), 而传统方法要 4.96 秒。

### 5.4 Tab. 2 的数据解读

| Scheduling | Steps | React Time | Overall Time | ADE |
|------------|-------|------------|--------------|-----|
| Autoregressive | 512 | 4.96s | 79.36s | 1.48 |
| Full-sequence | 32 | 4.96s | 4.96s | 1.28 |
| Pyramidal | 48 | 0.16s | 7.68s | 1.53 |
| Trapezoidal | 40 | 0.16s | 6.20s | 1.39 |
| Trapezoidal + feedback | 40 | 0.16s | 6.20s | **1.17** |

关键观察:
- **React time 从 4.96s 降到 0.16s** — 30 倍提升, 这是 closed-loop 的关键
- Overall time 比 full-sequence 慢一点 (6.20 vs 4.96), 但换来 reactivity 值得
- Trapezoidal + feedback 比 full-sequence ADE 还低 (1.17 vs 1.28), 说明 reactivity 不仅不 hurt 质量, 还能 help

---

## 6. Classifier Guidance: 让生成别太离谱

Diffusion model 有时会生成不合常理的场景 — 两车重叠、开出路面、轨迹抖动。Nexus 在每个 denoising step 加了三个 "纠偏" 函数。

### 6.1 防碰撞 (Eq 5-6)

如果两车的 bounding box 重叠, 沿着两车中心连线方向把两车推开。

$$\mathbf{x}_{\text{loc}}^t \leftarrow \mathbf{x}_{\text{loc}}^t + \lambda_t \sum_{i \neq j} \mathbb{I}\{\text{overlap}\} \cdot \frac{\mathbf{x}_i - \mathbf{x}_j}{\|\mathbf{x}_i - \mathbf{x}_j\|}$$

- $\lambda_t$: 推力强度
- $\mathbb{I}\{\cdot\}$: 重叠时为 1, 否则 0
- 分数: 从 $j$ 指向远离 $i$ 的单位向量

本质就是简化的 contact force。

### 6.2 防抖动 (Eq 7-9)

用二阶差分算加速度, 减去加速度的一部分:

$$\mathbf{a}^t = \frac{1}{2}(\mathbf{x}_{\tau-1} - 2\mathbf{x}_\tau + \mathbf{x}_{\tau+1})$$

$$\mathbf{x}_{\text{loc}}^t \leftarrow \mathbf{x}_{\text{loc}}^t - \lambda_t \mathbf{a}^t$$

相当于 low-pass filter, 让轨迹更平滑。

### 6.3 防出轨 (Eq 10-12)

如果车偏离最近车道点超过阈值 $d_{\text{th}}$, 把车拉回那个点:

$$\mathbf{x}_{i,\text{loc}}^t \leftarrow \mathbf{x}_{i,\text{loc}}^t + \lambda_t \mathbb{I}\{\text{off-road}\} \cdot (\mathbf{c}_i^t - \mathbf{x}_{i,\text{loc}}^t)$$

相当于 spring force, 把车拴在车道上。

**注意**: 这三个 guidance 是 deterministic 的, 在每个 sampling step 后直接改 $\mathbf{x}^t$, 不像 [classifier guidance](https://arxiv.org/abs/2105.05233) 要算梯度。更像是 [Imagen 的 dynamic thresholding](https://arxiv.org/abs/2205.11487) 的简化版, 简单粗暴但有效。

paper 设 $\lambda = 0.2$ 总和, 多个 constraint 同时激活时均分。

---

## 7. 数据集: Nexus-Data

### 7.1 为什么要自己造数据?

公开数据集 (nuPlan, Waymo) 95% 以上是正常行驶, 很少有加塞、急刹、碰撞。模型见得少就学不好, 生成不出 risky scenario。

Nexus 团队用 [MetaDrive](https://github.com/metadriverse/metadrive) simulator + [CAT (Closed-loop Adversarial Training)](https://arxiv.org/abs/2306.08312) 造了 **540 小时** 的 safety-critical data:

流程 (Fig. 5):
1. 从 nuPlan 拿真实场景做初始化
2. 在 MetaDrive 里重建 digital twin
3. 选一辆车当 "攻击车", 用对抗学习找最容易撞车的轨迹
4. 自动过滤无效场景 (只有 36.9% 是真的产生了碰撞)
5. 最终得到 540 小时高质量 risky data

### 7.2 数据分布对比 (Tab. 6)

关键指标是 lane change 比例:
- nuScenes: 左变道 5.0% + 右变道 2.5% = 7.5%
- nuPlan: 14.4% + 14.6% = 29.0%
- **Nexus-Data: 22.2% + 23.3% = 45.5%**

Nexus-Data 的变道场景是 nuPlan 的 1.5 倍, nuScenes 的 6 倍。这对训练 planner 应对复杂交互非常重要。

---

## 8. 实验结果讲人话

### 8.1 主实验 (Tab. 1)

跟 baseline 比:
- vs SceneDiffuser: 位移误差 (ADE) 从 5.99 → 1.28, **降了 78.6%**, 同时还更快 (2.79s vs 5.34s)
- vs GUMP (transformer 路子): ADE 1.93 → 1.28, 碰撞率 7.85% → 1.62%, **碰撞率降了 79.4%**
- Nexus-Full (加上 Nexus-Data 和 classifier guidance) 进一步把 ADE 降到 1.12, 碰撞率 1.56%

**翻译**: Nexus 生成的场景更准 (ADE 低)、更安全 (碰撞率低)、更平滑 (kinematic metric 低), 而且生成得更快。

### 8.2 Ablation (Tab. 3)

把 Nexus 的几个部件逐个加上去, 看 ADE 怎么变:

| 加什么 | ADE |
|--------|-----|
| Baseline (Diffusion Policy) | 7.53 |
| + Noise Masking (核心 idea) | 3.42 (**-4.11**) |
| + 2D Positional Encoding | 2.52 (-0.90) |
| + Nexus-Data | 1.92 (-0.60) |
| + Classifier Guidance | 1.25 (-0.67) |

**结论**: Noise masking 是大头, 贡献了 -4.11 的 ADE 提升。这验证了 decoupled noise 训练的有效性 — 这是 paper 的核心 claim 的直接证据。

### 8.3 当 World Generator (Tab. 4)

把 Nexus 当作环境, 让一个 planner agent 在里面跑 closed-loop:

| 环境 | Reactive Score |
|------|----------------|
| 真实环境 (Oracle) | 82.8 |
| Diffusion Policy 当环境 | 61.6 (-21.2) |
| SceneDiffuser 当环境 | 57.2 (-25.6) |
| **Nexus 当环境** | **73.0 (-9.8)** |

**翻译**: 当 agent 改变决策时, Nexus 环境能正确响应, 跟真实环境差距只有 9.8 分; 而 SceneDiffuser 差 25.6 分。这就是 reactivity 的直接度量。

### 8.4 当 Data Engine (Tab. 5)

用 Nexus 生成 synthetic data, 混进 real data 训练 planner:

| 训练数据 | Reactive Score |
|----------|----------------|
| 只用 real | 48.11 |
| + 3× 合成 | 46.61 (反而降!) |
| + 30× 合成 | 56.46 (+8.35) |
| + 60× 合成 | 57.86 (**+9.75**) |

**翻译**: 少量合成数据反而有害 (因为 noise 干扰), 但量大了 (60×) 能把 planner 性能提升 20%。这验证了 Nexus 作为 "data engine" 的价值 — 可以批量生产训练数据。

继续加到 100× 会饱和, 这符合 [Chinchilla scaling law](https://arxiv.org/abs/2203.15571) 的规律: 数据和模型要按比例 scale。

---

## 9. 这篇 paper 为什么重要

从 Karpathy 的视角看, 这篇 paper 有几个重要的信号:

### 9.1 Diffusion 正在从 "图像生成" 走向 "交互式仿真"

最初 [DDPM](https://arxiv.org/abs/2006.11239) 只能生成静态图, 后来 [Sora](https://openai.com/sora) 能生成 video, 但都是 offline 的 — 生成完了就完了, 不能交互。

Nexus 代表了新趋势: **diffusion 作为 interactive environment**。这跟 [Genie](https://arxiv.org/abs/2402.05321) (DeepMind 的可控游戏生成) 是同一个方向。未来 RL training 可能不再依赖 CARLA 这种 rule-based simulator, 而是用 diffusion-based world model。

### 9.2 "Noise as Mask" 是一个 deep insight

Nexus 的核心数学 (Eq 4) 把 diffusion 和 next-token prediction 统一了。这跟 [Diffusion Forcing](https://arxiv.org/abs/2407.01392) 的思想一致, 说明学术界正在形成共识: **generative modeling 的本质是 masked prediction**, 不管是 BERT 式的 hard mask 还是 diffusion 式的 soft mask。

未来可能看到更多 "noise + mask" 混合的架构, 比如 [MD4](https://arxiv.org/abs/2406.05509) 那种把 diffusion 用在 masked language modeling 上的尝试。

### 9.3 自动驾驶数据飞轮的雏形

Tab. 5 的 data augmentation 实验很有意思: 60× 合成数据能提升 20% 性能。这暗示了一个 future:

1. 真实数据训练 world model
2. World model 大量生成 synthetic data
3. Synthetic data 训练 planner
4. Planner 上路收集更多真实数据
5. Goto 1

这就是所谓的 "data engine" 或 "self-improving loop"。Nexus 提供了 step 2 的可行性验证。参考 Tesla 的 [Dojo + AutoLabeling](https://www.tesla.com/AI) 思路。

### 9.4 局限性

Paper 在 Appendix Q4 自己承认:
1. **只生成 layout, 不生成 visual** — 不能直接训练 perception model。未来要结合 [NeRF](https://arxiv.org/abs/2003.08934) 或 [Video Diffusion](https://arxiv.org/abs/2311.15127)。
2. **Hallucination 风险** — Diffusion 可能生成不合理的场景, 需要 rule-based validation。
3. **Closed-loop 仿真保真度还没到 Oracle 水平** — 73.0 vs 82.8, 还有 9.8 分差距。

从我的角度看, 还有几个开放问题:
- **Chunk length 怎么动态调?** 现在固定 40 steps, 但简单场景 (高速巡航) 可以短, 复杂场景 (路口交互) 应该长。
- **Goal guidance 强度怎么 schedule?** 噪声 = 0 太 hard, 也许 0.01 更平滑, 允许 model 有小幅度偏离。
- **能否 multi-modal?** 现在 Nexus 是 deterministic 的, 但 driving 本质是 multi-modal (同一时刻可能左转也可能直行)。可以参考 [Diffusion-LM](https://arxiv.org/abs/2205.15025) 在 text 上的做法。

---

## 10. 我的直觉总结

这篇 paper 的核心贡献是一个概念升级: **把 noise 从 "全局调度参数" 变成 "per-token 的局部 mask"**。

这个升级让 diffusion model 从 "画一幅画" 变成了 "一边画一边改", 从 offline generation 变成了 online interaction。

具体到 driving 场景:
- 传统 diffusion: 给我起点终点, 我一次性画出完整轨迹, 中途不能改
- Nexus: 给我起点终点, 我边画边跟你确认, 你随时可以喊 "这里不对, 改一下", 我就改那一帧, 其他不动

这种 "incremental + interactive" 的生成范式, 对 closed-loop RL training 至关重要。因为 RL agent 需要 environment 能实时响应自己的 action, 否则训练出来的 policy 在真实世界会失效。

未来如果 Nexus 能扩展到 visual generation (结合 [Sora-like video diffusion](https://arxiv.org/abs/2311.15127) 或 [NeuRAD](https://arxiv.org/abs/2312.06045)), 那就是真正的 "generative driving simulator" 了 — 既能生成结构化轨迹, 又能渲染逼真视觉, 还能实时交互。这会是 autonomous driving 的 data engine 关键拼图。

参考链接:
- [Nexus 项目主页](https://opendrivelab.com/Nexus)
- [Diffusion Forcing (概念启发)](https://arxiv.org/abs/2407.01392)
- [Diffusion Policy (baseline)](https://arxiv.org/abs/2303.04137)
- [SceneDiffuser (baseline)](https://arxiv.org/abs/2412.12129)
- [GUMP (transformer baseline)](https://arxiv.org/abs/2412.04675)
- [DiT (架构基础)](https://arxiv.org/abs/2212.09748)
- [Perceiver IO (map encoder)](https://arxiv.org/abs/2107.14795)
- [MAE (mask modeling 启发)](https://arxiv.org/abs/2111.06377)
- [CAT (对抗数据生成)](https://arxiv.org/abs/2306.08312)
- [MetaDrive (simulator)](https://github.com/metadriverse/metadrive)
- [Vista (visual world model)](https://arxiv.org/abs/2405.17398)
- [NeuRAD (神经渲染)](https://arxiv.org/abs/2312.06045)
- [nuPlan (训练数据)](https://www.nuscenes.org/nuplan)
- [Genie (DeepMind 交互生成)](https://arxiv.org/abs/2402.05321)
- [DDPM (diffusion 基础)](https://arxiv.org/abs/2006.11239)
- [Chinchilla (scaling law)](https://arxiv.org/abs/2203.15571)

---

# Nexus: Decoupled Diffusion for Adaptive Scene Generation 深度解析

## 1. Problem Framing: Reactivity vs Goal Orientation 的 Trade-off

这篇 paper 直击 autonomous driving scene generation 中的一个核心矛盾。现有方法都陷入 two extremes:

**Extreme 1: Full-sequence Diffusion** (如 [SceneDiffuser](https://arxiv.org/abs/2412.12129), [Diffusion Policy](https://arxiv.org/abs/2303.04137))
- 把整个 future sequence 当作一张 image 来 denoise
- 优点: 可以通过 inpainting 实现 goal conditioning (用 hard mask 固定 goal tokens)
- 缺点: 一旦 agent decision 变化, 整个 sequence 要全部丢弃重新生成, **无法 online react**

**Extreme 2: Next-token Prediction** (如 [GUMP](https://arxiv.org/abs/2412.04675), [MotionLM](https://arxiv.org/abs/2309.16534), [Trajeglish](https://philion.github.io/trajeglish/))
- 用 autoregressive transformer 滚动预测下一帧
- 优点: 可以实时 incorporate environmental feedback
- 缺点: **causal masking 让 model 看不到 future state**, 想要 "导演" 一个 collision 场景几乎不可能

Nexus 的核心洞察: **noise 本身就是 mask**, 而且是 soft mask。给不同 token 分配**独立的 noise level**, 就能同时获得两种能力:
- Low-noise tokens $\rightarrow$ 起到 "已知条件" 的作用 (goal/past)
- High-noise tokens $\rightarrow$ 起到 "待生成" 的作用 (future)
- 中间状态的 noise $\rightarrow$ 平滑过渡, 既不完全确定也不完全自由

这个 idea 在概念上跟 [Diffusion Forcing](https://arxiv.org/abs/2407.01392) (Boyuan Chen et al., MIT) 很接近 — 把 next-token prediction 和 full-sequence diffusion 统一到 "mask modeling" 框架下, 但 Nexus 是把这个思想具体应用到了 multi-agent traffic simulation, 并加入了 noise-aware scheduling 来解决 online reactivity 问题。

---

## 2. 数学细节: 从 Tri-axial Mask 到 Decoupled Noise

### 2.1 传统的 Full-sequence Diffusion

先看 standard DDPM 的训练目标。给定原始 trajectory $\mathbf{x}^0 \sim p(\mathbf{x})$, 前向加噪:

$$\mathbf{x}^t = \alpha_t \mathbf{x}^0 + \sigma_t \epsilon, \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \tag{1}$$

变量解释:
- $\mathbf{x}^0 \in \mathbb{R}^{A \times T \times D}$: 原始 agent tensor
  - $A$: 最大 agent 数量
  - $T$: timesteps 总数
  - $D$: 每个 agent 的 attribute 维度 (paper 中是 8: $x, y, \sin\alpha, \cos\alpha, v_x, v_y, l, w$)
- $t \in (0, 1]$: denoising step (归一化到 (0, 1])
- $\alpha_t, \sigma_t$: signal/noise schedule (标量函数), 满足 $\alpha_t$ 随 $t$ 增大而减小, $\sigma_t$ 相反
- $\epsilon$: 标准高斯噪声

训练目标 (Eq 2):

$$\forall t, \min_\theta \mathbb{E}\|(\epsilon - \epsilon_\theta(\mathbf{x}^t; \mathbf{c}, t)) \odot \mathbf{m}\|_2^2 \tag{2}$$

- $\theta$: DiT 的参数
- $\epsilon_\theta$: 神经网络预测的噪声
- $\mathbf{c} \in \mathbb{R}^{L \times N \times D'}$: map tensor (条件输入)
  - $L$: lanes 数量
  - $N$: 每个 lane 的 points 数
  - $D'$: lane attribute 维度
- $\mathbf{m} \in \mathbb{B}^{A \times T}$: valid mask, 标记哪些 agent-timestep 是有效的
- $\odot$: element-wise 乘法, 用 mask 排除 invalid tokens

关键点: **这里所有 token 共享同一个 $t$**, 这就是 "uniform noise" 的本意。

### 2.2 Inpainting Sampling (Hard Mask Conditioning)

Eq (3) 描述传统的 conditioning 方式:

$$p(\mathbf{x}^s \vert \mathbf{x}^t) = \mathcal{N}(\mathbf{x}^s \vert \mu(\mathbf{x}^t, t), \Sigma(\mathbf{x}^t, t)) \odot \bar{\mathbf{m}}_c + \mathbf{x}^t \odot \mathbf{m}_c \tag{3}$$

- $s$: 下一个 denoising step ($s < t$)
- $\mu, \Sigma$: 由 DiT 输出的 mean 和 covariance
- $\mathbf{m}_c$: keep mask (binary), 标记要固定的 tokens (goal, past history)
- $\bar{\mathbf{m}}_c = 1 - \mathbf{m}_c$: complement mask

**问题**: 这个机制要求整个 sequence 都参与每一步 denoising, 即使大部分内容已经确定了, 也要继续算 attention, 不能 "pop" 出来。

### 2.3 核心: Tri-axial Mask Modeling (Eq 4)

这是 Nexus 最关键的改动。定义每个 token 的独立 noise level:

$$\pmb{x}_{a, \tau}^{k_{a, \tau}}: \text{ agent } a \text{ 在 timestep } \tau \text{ 的 token, 噪声水平 } k_{a, \tau}$$

- $a$: agent 索引 ($1 \le a \le A$)
- $\tau$: 物理 timestep 索引
- $k_{a, \tau} \in (0, 1]$: 该 token 的 noise level
- $k_{a, \tau} = 0$: 纯净 token (ground truth)
- $k_{a, \tau} = 1$: 纯噪声

整个 sequence 的 noise level 用矩阵 $\mathbf{k} = [k_{a, \tau}] \in (0, 1]^{A \times T}$ 表示。

新的训练目标 (Eq 4):

$$\forall \mathbf{k} \in (0, 1]^{A \times T}, \min_\theta \mathbb{E}\|(\epsilon - \epsilon_\theta(g(\mathbf{x}^0, \mathbf{k}); \mathbf{c}, \mathbf{k}))\|_2^2 \tag{4}$$

- $g$: 根据 noise matrix $\mathbf{k}$ 给每个 token 独立加噪的函数
- $\mathbf{k}$: 现在是 model 的**输入之一**, 替代了原来的标量 $t$
- **没有** $\mathbf{m}$ 的 hard mask 了! 取而代之的是 soft mask (通过 noise level 控制)

**直觉解释**: 训练时, 每个 agent-timestep 组合随机抽一个 noise level。模型学到: 给定一些 "已知" 的 tokens (low noise) 和一些 "未知" 的 tokens (high noise), 重建完整 sequence。这本质上是把 [MAE](https://arxiv.org/abs/2111.06377) 的 masked prediction 思想和 diffusion 结合起来, 用 noise level 代替 binary mask。

好处在 inference 时显现:
- 想做 goal-conditioned generation $\rightarrow$ 把 goal token 的 noise 设为 0, 其余设为 1
- 想做 free exploration $\rightarrow$ 全部设为 1
- 想做 "partial rollout" (类似 Sora 的 video extension) $\rightarrow$ past tokens noise=0, future tokens noise 渐增

### 2.4 Per-token 加噪的实现

根据 [SimpleDiffusion](https://arxiv.org/abs/2301.00927) 的框架, 单个 token 的加噪:

$$\pmb{x}_{a, \tau}^{k_{a, \tau}} = \alpha_{k_{a, \tau}} \cdot \pmb{x}_{a, \tau}^0 + \sigma_{k_{a, \tau}} \cdot \epsilon_{a, \tau}$$

每个 token 独立采样自己的 $\epsilon_{a, \tau} \sim \mathcal{N}(0, I)$。

---

## 3. Architecture: Nexus 的 DiT 设计

参考 [Tab. 7](https://opendrivelab.com/Nexus) 的架构细节:

```
Input: 
  Agent tensor x ∈ R^(A×T×25) → Linear(25, 256) → token embeddings
  Map tensor c → Perceiver IO → fixed-length map tokens

Backbone (4 layers):
  For each layer:
    Map Cross-Attention: agent tokens query map tokens
    TemporalBlock: LayerNorm → MultiHeadAttention over time axis → AdaLN modulation
    SpatialBlock: LayerNorm → MultiHeadAttention over agent axis → AdaLN modulation  
    FeedForward: Linear(256, 1024) → GELU → Linear(1024, 256)

Positional Encoding: 2D Rotary (physical time + denoising step)
Output: Linear(256, 8) → reconstructed noise
```

关键设计选择:

1. **Rotary Positional Embedding (2D)**: 同时编码 physical time $\tau$ 和 denoising step $k$。这是 Eq (4) 能 work 的前提 — 模型必须知道每个 token 处在哪个 noise level, 才能 decide 如何 attend。

2. **Perceiver IO for Map**: [Perceiver IO](https://arxiv.org/abs/2107.14795) 把变长的 map tokens 压缩成固定长度, 避免 map 规模变化导致 attention cost 爆炸。

3. **AdaLN Modulation**: 来自 [DiT paper](https://arxiv.org/abs/2212.09748), 用 timestep embedding 通过 MLP 调制 attention 和 FFN 的 scale/shift 参数。在 Nexus 中, 这个 modulation 是 per-token 的 (因为每个 token 有自己的 $k_{a,\tau}$), 比 standard DiT 的 per-sequence modulation 更细粒度。

4. **TemporalBlock + SpatialBlock 交替**: 类似 [MotionLM](https://arxiv.org/abs/2309.16534) 的设计, 分开处理 time 和 agent 维度的 attention, 而非用单个 3D attention。这降低计算量从 $O((AT)^2)$ 到 $O(A^2T + AT^2)$。

---

## 4. Noise-aware Scheduling: 实现 Reactivity

训练解决了 "如何 condition" 的问题, 采样要解决 "如何 react" 的问题。这里 Nexus 提出 **chunk-based pipelined sampling**。

### 4.1 Chunk 的概念

定义一个 chunk = 一个 localized subset of scenario, 包含:
- Historical context (low noise, 已确定)
- Active future frames (varying noise, 正在 denoise)
- Optional goal tokens (noise = 0, 终点)

每次 denoising step:
1. 所有 active tokens 的 noise level 下降
2. 噪声降到 0 的 token 被 **pop** 出来 (成为确定状态)
3. 新的高噪声 token 被 **push** 进来 (开始 denoise)
4. chunk 整体沿时间轴滑动

形式化: 定义三维 scheduling matrix $\boldsymbol{\mathcal{K}} \in [\mathbf{k}]^M$, 每个元素 $\mathcal{K}_{a, \tau, m}$ 表示 agent $a$ 在 timestep $\tau$ 在 sampling step $m$ 的 noise level。

### 4.2 三种 Scheduling 策略对比 (Tab. 2)

| Strategy | Steps | React Time | Overall Time | ADE |
|----------|-------|------------|--------------|-----|
| Autoregressive | 512 | 4.96s | 79.36s | 1.48 |
| Full-sequence | 32 | 4.96s | 4.96s | 1.28 |
| **Pyramidal** | 48 | **0.16s** | 7.68s | 1.53 |
| **Trapezoidal** | 40 | **0.16s** | **6.20s** | 1.39 |
| Trapezoidal + feedback | 40 | 0.16s | 6.20s | **1.17** |

**Pyramidal Scheduling** (Fig. 4c):
- Chunk length 随 denoising steps 增长
- 每个 step 加入 1 个新 frame token, 弹出 1 个完全 denoise 的 token
- 单向流动 (past → future)
- React time 极短 (0.16s), 因为环境变化只需 overwrite 当前 active chunk 中的对应 token

**Trapezoidal Scheduling** (Fig. 4d):
- 双向更新: tokens 从 chunk 两端进入和退出
- Goal 的 guidance 可以通过 spatial-temporal attention 反向传播给所有 agents
- 比 pyramidal 更适合 goal-conditioned generation
- 类似 [Bidirectional Mamba](https://arxiv.org/abs/2312.00751) 的思想, 利用未来信息辅助当前生成

**Trapezoidal + feedback**: 进一步把 agent 的实际 action 反馈到 history tokens, ADE 从 1.39 降到 1.17。这是 closed-loop interaction 的关键。

### 4.3 Why this works?

对比 naive autoregressive:
- Autoregressive 每生成 1 帧需要完整 denoise 32 步 $\times$ 单帧 cost $\approx$ 16s/frame, 慢
- Nexus 在 chunk 内并行 denoise 多帧, 摊薄了 cost
- 而且因为 noise level 是 decoupled 的, "已确定" 的帧不会被重复 denoise, 直接 pop

对比 full-sequence:
- Full-sequence 虽然只 denoise 32 步, 但每步 cost 与 $A \times T$ 成正比
- 一旦 agent 改变决策, 整个 future 都要重新 denoise
- Nexus 只需修改对应 token 的 noise state, 在下一个 step 自然被处理

---

## 5. Classifier Guidance: 行为合理性约束

Diffusion model 容易生成不合理的轨迹 (车辆重叠, 离开道路, 抖动)。Nexus 借鉴 [Imagen](https://arxiv.org/abs/2205.11487) 的 dynamic thresholding, 在每个 sampling step 加入 deterministic correction。

### 5.1 Collision Avoidance (Eq 5-6)

$$f_{\text{collision}}(\mathbf{x}^t, t) = [\mathbf{x}_{\text{loc}}^t, \mathbf{x}^{t, 3:d}]$$

其中位置更新:

$$\mathbf{x}_{\text{loc}}^t \leftarrow \mathbf{x}_{\text{loc}}^t + \lambda_t \sum_{i \neq j} \mathbb{I}\{B(\mathbf{x}_i^t) \cap B(\mathbf{x}_j^t) \neq \emptyset\} \cdot \frac{\mathbf{x}_{i, \text{loc}}^t - \mathbf{x}_{j, \text{loc}}^t}{\|\mathbf{x}_{i, \text{loc}}^t - \mathbf{x}_{j, \text{loc}}^t\|}$$

变量:
- $\mathbf{x}_{\text{loc}}^t \in \mathbb{R}^{A \times 2}$: 所有 agent 的 位置 (从 8-d attribute 中切出前 2 维)
- $\mathbf{x}^{t, 3:d}$: 其余 attribute (heading, velocity, size), 保持不变
- $\lambda_t$: scalar, 控制 separation 力度 (随 $t$ 变化)
- $\mathbb{I}\{\cdot\}$: 指示函数, bounding box 重叠时为 1
- $B(\mathbf{x}_i^t)$: agent $i$ 的 bounding box (基于位置和 size 算出)
- 分数项: agent $i$ 指向远离 agent $j$ 的单位向量

直觉: 检测到 overlap 就把两车沿连线方向推开。等价于一个简化的 contact force。

### 5.2 Comfort (Smoothness, Eq 7-9)

$$\mathbf{a}^t = \frac{1}{2}(\mathbf{x}_{\tau-1, \text{loc}}^t - 2\mathbf{x}_{\tau, \text{loc}}^t + \mathbf{x}_{\tau+1, \text{loc}}^t)$$

二阶差分近似加速度 (注意: 标准 central difference 系数是 1, 这里 1/2 是某种 normalization)。

$$\mathbf{x}_{\text{loc}}^t \leftarrow \mathbf{x}_{\text{loc}}^t - \lambda_t \mathbf{a}^t$$

减去加速度的比例, 抑制急停急转。等价于一个 low-pass filter。

### 5.3 On-road (Eq 10-12)

$$\mathbf{c}_i^t = \arg\min_{l, n} \|\mathbf{x}_{i, \text{loc}}^t - \mathbf{c}_{l, n, \text{loc}}\|$$

在所有 lane points 中找最近的 $\mathbf{c}_i^t$。

$$\mathbf{x}_{i, \text{loc}}^t \leftarrow \mathbf{x}_{i, \text{loc}}^t + \lambda_t \mathbb{I}\{\|\mathbf{x}_{i, \text{loc}}^t - \mathbf{c}_i^t\| > d_{\text{th}}\} \cdot (\mathbf{c}_i^t - \mathbf{x}_{i, \text{loc}}^t)$$

偏离阈值 $d_{\text{th}}$ 时, 拉向最近 lane point。相当于 spring force。

**注**: paper 中提到 total $\lambda = 0.2$, 多个 constraint 同时激活时均分。这种 hybrid guidance 的方式跟 [classifier-free guidance](https://arxiv.org/abs/2207.12598) 的思路类似, 都是手动加 structural prior。

---

## 6. Nexus-Data: Safety-critical Scenarios 数据集

### 6.1 动机

公开数据集 ([nuPlan](https://www.nuscenes.org/nuplan), [Waymo Open](https://waymo.com/open/)) 主要是 ordinal driving, 缺少 risky behaviors。从 Tab. 6 可看到 lane change 的占比:
- nuScenes: 5.0% left + 2.5% right
- nuPlan: 14.4% + 14.6%
- **Nexus-Data: 22.2% + 23.3%**

### 6.2 构造流程 (Fig. 5)

1. **Scene Record 提取**: 用 [ScenarioNet](https://arxiv.org/abs/2306.12241) 把 nuPlan scenes 转换成统一格式
2. **Simulator 重建**: 用 [MetaDrive](https://github.com/metadriverse/metadrive) 重建 digital twin
3. **Adversarial Attack**: 用 [CAT](https://arxiv.org/abs/2306.08312) 选择一辆 attack vehicle, 通过对抗学习找最容易造成 collision 的轨迹
4. **Filtering**: 只有 36.9% 的对抗 sample 真的产生 collision; 用 checklist 过滤 off-road, invalid trajectory 等
5. 最终: **540 hours** 的 safety-critical data

### 6.3 数据集统计 (Tab. 6)

| Dataset | Hours | Inter. Passing | L. Turn | R. Turn | L. Change | R. Change | U-Turn | Stop |
|---------|-------|----------------|---------|---------|------------|------------|--------|------|
| nuScenes | 5.5 | 13.1 | 18.0 | 10.2 | 5.0 | 2.5 | 0.0 | 4.1 |
| nuPlan | 1.2K | 13.8 | 1.5 | 1.6 | 14.4 | 14.6 | 0.9 | 46.8 |
| **Nexus-Data** | 540 | 35.3 | 1.7 | 2.5 | 22.2 | 23.3 | 1.2 | 10.0 |

Nexus-Data 显著提升了 lane change 比例, 这对训练 robust planner 极其重要。

---

## 7. 实验结果深度分析

### 7.1 主实验 (Tab. 1)

| Method | ADE↓ | R_road↓ | R_col↓ | M_k↓ | Time (s) |
|--------|------|---------|--------|------|----------|
| IDM (rule-based) | 10.52 | 9.85 | 10.17 | 6.30 | 12.16 |
| Diffusion Policy | 7.80 | 13.9 | 14.92 | 12.71 | 6.59 |
| SceneDiffuser | 5.99 | 8.53 | 11.78 | 9.64 | 5.34 |
| GUMP (transformer) | 1.93 | 7.73 | 7.85 | 16.18 | 5.59 |
| **Nexus** | **1.28** | 6.89 | 1.62 | 4.63 | 2.79 |
| **Nexus-Full** | **1.12** | **6.25** | **1.56** | **3.17** | 2.93 |

- vs SceneDiffuser: ADE 从 5.99 → 1.28 (**-78.6%**), 同时 speed 提升 2.55s
- vs GUMP: ADE 从 1.93 → 1.28 (**-33.7%**), 关键是 collision rate 从 7.85% → 1.62% (**-79.4%**)
- Nexus-Full 加上 Nexus-Data + classifier guidance, 进一步把 collision 降到 1.56%

### 7.2 Ablation Study (Tab. 3)

| Method | ADE↓ |
|--------|------|
| Baseline (Diffusion Policy) | 7.53 |
| + Noise Masking (Eq 4) | 3.42 (**-4.11**) |
| + Positional Embedding (time + denoise) | 2.52 |
| + Nexus-Data | 1.92 |
| + Classifier Guidance | **1.25** |

Noise masking 是最关键的 contribution (-4.11 ADE), 这验证了 decoupled noise 训练的有效性。

### 7.3 Closed-loop Evaluation (Tab. 4)

把 Nexus 当作 world generator, 用 [Diffusion Planner](https://arxiv.org/abs/2501.15564) 作为 ego agent 在 Nexus 生成的环境里跑 closed-loop:

| Method | Reactive Score | Non-reactive Score |
|--------|----------------|---------------------|
| Oracle (real env) | 82.8 | 89.2 |
| Diffusion Policy (as world) | 61.6 (-21.2) | 47.2 (-42.0) |
| SceneDiffuser (as world) | 57.2 (-25.6) | 50.1 (-39.1) |
| **Nexus** (as world) | **73.0 (-9.8)** | **68.1 (-21.1)** |

Reactive eval gap: Nexus 比 SceneDiffuser 高 15.8 分。这直接证明 Nexus 的 reactivity 优势 — 当 agent 改变决策时, Nexus 能正确响应, 而 SceneDiffuser 会卡住。

### 7.4 Data Augmentation (Tab. 5)

用 Nexus 生成 synthetic data 训练 lightweight planner:

| Training Data | Reactive Score |
|---------------|----------------|
| Real only | 48.11 |
| + 3× Synth. | 46.61 (down!) |
| + 30× Synth. | 56.46 (+8.35) |
| + 60× Synth. | **57.86 (+9.75)** |

- 3× 反而下降: 说明少量 synthetic data 的 noise 会 hurt learning
- 60× 提升 20%: 验证 Nexus 作为 data engine 的潜力
- 继续增加开始饱和: 典型的 data scaling law 行为

---

## 8. 与 Related Work 的对比

### 8.1 vs Diffusion Forcing ([Boyuan Chen et al.](https://arxiv.org/abs/2407.01392))

Diffusion Forcing 是 Nexus 在概念上最接近的工作, 都主张 "noise as mask"。区别:
- Diffusion Forcing: 通用框架, 理论分析为主
- Nexus: 具体应用到 multi-agent traffic, 加了 scheduling strategy 和 classifier guidance

### 8.2 vs Sora ([Video Diffusion Models](https://arxiv.org/abs/2311.15127))

Sora 也是用 diffusion 做 video generation, 但:
- Sora: pixel-space, joint space-time attention, fixed sequence length
- Nexus: structured token-space (vectorized agent states), decoupled noise, variable-length via chunking

### 8.3 vs GenAD / Vista ([World Models](https://arxiv.org/abs/2405.17398))

[Vista](https://arxiv.org/abs/2405.17398) 和 [GenAD](https://arxiv.org/abs/2406.01349) 做的是 visual world model, 直接生成 video。Nexus 只生成 layout (vectorized trajectories), 然后用 [NeuRAD](https://arxiv.org/abs/2312.06045) 渲染 visual。这是分工: structural generation vs appearance generation。

### 8.4 vs SceneDiffuser ([Waymo's scene gen](https://arxiv.org/abs/2412.12129))

SceneDiffuser 是 full-sequence diffusion + LLM-driven conditioning。Nexus 的优势:
- Decoupled noise → online reactivity
- 不需要 LLM prompt → 更直接
- 实测 closed-loop 表现更好 (73.0 vs 57.2)

---

## 9. Limitations & Future Directions

Paper 在 Appendix Q4 提到:
1. **没有 visual synthesis**: 只生成 trajectory layout, 不能直接用于 perception training
2. **未来工作**: 集成 [NeRF](https://arxiv.org/abs/2003.08934) 或 [Video Diffusion](https://arxiv.org/abs/2311.15127) 做 full visual simulation
3. **Hallucination 风险**: Diffusion 可能生成不现实场景, 需要 rule-based validation

从我 (Karpathy) 的视角看, 还有几个开放问题:
- Chunk length 如何自适应? paper 是固定 40 steps, 是否可以根据场景复杂度动态调整
- Goal guidance 强度如何 schedule? 现在 noise=0 太 hard, 是否用 low but nonzero noise 更平滑
- 能否扩展到 [Behavior Cloning from Observation](https://arxiv.org/abs/2305.19116) 的 setting, 让 model 自己生成 training data?

---

## 10. 总结: 这篇 paper 的核心贡献

**核心 Insight**: Noise level 本质上是 soft mask。给每个 token 独立的 noise level, 就把 "uniform noise diffusion" 和 "next-token prediction" 统一了 — 这就是 **decoupled diffusion** 的数学意义。

**三个 Engineering Contribution**:
1. **Noise-masking Training (Eq 4)**: 让模型从 soft-masked tokens 学会 sequence completion
2. **Noise-aware Scheduling (Fig. 4)**: pyramidal / trapezoidal 策略实现 pipelined denoising, 兼顾 speed 和 reactivity
3. **Nexus-Data (540h)**: 用 adversarial learning 大规模合成 safety-critical 场景

**为什么重要**: 这是 closed-loop driving simulation 的重要一步。Rule-based simulators (CARLA, MetaDrive) 太 rigid, 数据集 replay 不能 react, 而 Nexus 提供了一个 data-driven, reactive, goal-controllable 的第三选择。论文展示的 60× synthetic data 把 planner 性能提升 20%, 直接证明这种生成模型可以作为 autonomous driving 的 "data engine"。

参考资源:
- Project page: https://opendrivelab.com/Nexus
- [Diffusion Forcing (conceptual inspiration)](https://arxiv.org/abs/2407.01392)
- [DiT (architecture base)](https://arxiv.org/abs/2212.09748)
- [Perceiver IO (map encoder)](https://arxiv.org/abs/2107.14795)
- [MetaDrive (simulator)](https://github.com/metadriverse/metadrive)
- [CAT (adversarial data)](https://arxiv.org/abs/2306.08312)
- [SceneDiffuser (baseline)](https://arxiv.org/abs/2412.12129)
- [GUMP (transformer baseline)](https://arxiv.org/abs/2412.04675)
- [Vista (visual world model)](https://arxiv.org/abs/2405.17398)
- [NeuRAD (visual renderer)](https://arxiv.org/abs/2312.06045)
- [nuPlan (training data)](https://www.nuscenes.org/nuplan)
