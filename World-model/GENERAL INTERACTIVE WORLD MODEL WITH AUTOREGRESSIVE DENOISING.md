---
source_pdf: GENERAL INTERACTIVE WORLD MODEL WITH AUTOREGRESSIVE DENOISING.pdf
paper_sha256: a98e398b2303026ea7dc7691af1fb2a9835a9801a014900c0db08bf4dc0b969e
processed_at: '2026-08-04T13:28:48-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Astra 用人话讲

## 一句话

把一个已经会生成视频的大模型 (Wan-2.1) 稍微改一改，让它变成一个**能听你指挥、一段一段往外吐视频**的 world model。改的地方很小 (366M params)，但效果吊打一票 full-tune 的 baseline (YUME 14B)。

Project: https://eternalevan.github.io/Astra-project/
Code: https://github.com/EternalEvan/Astra

---

## 问题是什么

你手里有个 Wan-2.1，给它一句话 "一只猫在跑步"，它给你一段 5 秒视频。挺漂亮。

但你想要的是：**给一张图 + 一串 action "向前走 → 左转 → 停下看右边"**，它得给你一段 30 秒视频，每一步都乖乖听话。

Wan-2.1 做不到，因为：
- 它的 conditioning 是 global text prompt，静态的，没法 per-step 注入
- 它生成 short clip 一次搞定，没法 streaming 往后接
- 它没有 action interface 这个概念

所以核心问题：**怎么把一个 passive renderer 变成 interactive simulator？**

---

## Astra 的思路：三步走

### 第 1 步：把生成拆成 chunk-by-chunk

别一次性生成整段视频。切成 chunk，每次生成一小段 (33 frames)，生成完 append 到 history，下一段基于新 history 再生成。

这就是 autoregressive (AR)：

$$
p(z^{1:N}) = \prod_{i=1}^{N} p(z^i \mid z^{<i})
$$

- $z^i$: 第 $i$ 个 chunk 的 VAE latent
- $z^{<i}$: 之前所有 chunk 的 latent
- $N$: 总 chunk 数

每个 $p(z^i \mid z^{<i})$ 用一个 flow matching model 建模。Flow matching 就是：从 noise $z_1$ 到 clean $z_0$ 画一条直线路径，学一个 velocity field 沿着走。

$$
z_t^i = (1-t) z_0^i + t \epsilon, \quad t \in [0,1], \quad \epsilon \sim \mathcal{N}(0,I)
$$

- $t$: flow 时间，0 = clean，1 = noise
- $z_0^i$: ground truth clean latent
- $\epsilon$: Gaussian noise
- $z_t^i$: 插值点

训练 loss：

$$
\mathcal{L} = \mathbb{E} \| v_\theta(z_t^i, t, z^{<i}) - (\epsilon - z_0^i) \|^2
$$

$v_\theta$ 是要学的 velocity network，$(\epsilon - z_0^i)$ 是 ground-truth 方向。

推理就是：从 noise 出发，ODE 积分到 clean，decode 成视频帧，append 到 history，循环。

这样你就有了 **streaming + interactivity** 的骨架。每生成完一个 chunk，可以注入新的 action，模型即时响应。

参考 MAGI-1 (https://arxiv.org/abs/2505.13211) 也是 AR + diffusion 的思路。

---

### 第 2 步：怎么把 action 塞进模型

这里有个非常漂亮的 intuition，paper Section 3.2 说的：

**Action 就像 optical flow。**

Optical flow 说：$I(x,t) = I(x + \delta x, t + \delta t)$，像素移动 $\delta x$。Action 说的就是：latent feature 整体 shift 一个 $\delta z$。

既然是 shift，最自然的注入方式是 **element-wise addition**——把 action encode 成一个向量，直接加到 transformer block 的 output 上。

这就是 ACT-Adapter。

Wan-2.1 是 30 个 DiT block，每个 block 原来长这样：

```
x → Self-Attn (frozen) → FFN (frozen) → x'
```

Astra 改成：

```
x → Self-Attn (trainable) → + action_emb → ACT-Adapter (linear, init=I) → FFN (frozen) → x'
```

几个关键工程决定：

1. **Backbone 基本冻结**：只解冻 self-attention，其他全冻。最大化复用 Wan-2.1 学到的世界知识。
2. **ACT-Adapter 是单层 linear，初始化成 identity matrix**：训练第 0 步 adapter 是恒等映射，模型行为 = 原始 Wan-2.1。随着训练慢慢偏离，stable。
3. **Element-wise add，不是 cross-attention**：MatrixGame (https://arxiv.org/abs/2508.13009) 用 cross-attn 注 action，Astra ablation (Table 3) 显示 cross-attn 比 element-wise add 差 (0.642 vs 0.669)。原因就是 action 是 per-token shift，cross-attn 把它当 global condition 来 attend，信息稀释了。

**Action-Free Guidance (AFG)**：

训练时随机 drop action (换成 null token $\varnothing$)，让模型同时学 "有 action" 和 "无 action" 两个 mode。推理时放大两者差：

$$
v_{\text{guided}} = v_\theta(z_t, t, \varnothing) + s \cdot \big( v_\theta(z_t, t, a) - v_\theta(z_t, t, \varnothing) \big)
$$

- $v_\theta(z_t, t, \varnothing)$: 无 action 的 baseline velocity
- $v_\theta(z_t, t, a)$: 有 action 的 velocity
- $s$: guidance scale，paper 用 3.0
- 差值就是 action 的纯效应，乘 $s$ 放大

跟 classifier-free guidance (https://arxiv.org/abs/2207.12598) 同构，只不过 condition 从 class label 换成 action。

Ablation (Table 3): 去掉 AFG，Instruction Following 从 0.669 掉到 0.545。证明 action 信号太弱，得靠 guidance 放大才能盖过 visual context。

---

### 第 3 步：解决 visual inertia 这个大坑

**Visual inertia 是这篇 paper 最有意思的发现。**

现象：你给模型越多 history frames，视频质量越好 (consistency 高)，但 action 跟随度急剧下降。

Figure C 的曲线非常戏剧化：history length 从 1 涨到 128，quality 一路升，action-following 一路跌。

为什么？因为真实视频 95% 是 smooth motion——相机缓慢平移、人慢慢走。模型学到的 prior 是 "下一帧 ≈ 上一帧 + 微小扰动"。给它一堆干净 history，它直接外推过去，action 信号完全被淹没了。

这就是 generative world model 的根本张力：**history 给一致性，action 给响应性，二者此消彼长。**

Astra 的解法极简：**训练时给 history 加噪声**。

$$
z_c^{\text{noisy}} = z_c + \sigma \cdot \eta, \quad \eta \sim \mathcal{N}(0,I)
$$

- $z_c$: history latent
- $\eta$: 跟 diffusion noise 独立的 Gaussian noise
- $\sigma$: 噪声强度 (paper 没给具体值)

核心 trick：
- **训练时 history 是 noisy 的**：模型不能直接 copy history，被迫同时用 action 和 visual cue。
- **推理时 history 是 clean 的**：模型已经学会平衡两个信号，给 clean history 也不会只靠外推。
- **不需要加任何参数**：纯 data augmentation。

Ablation (Table 3): 去掉 noise，Instruction Following 从 0.669 暴跌到 **0.359**。这是所有 ablation 里跌幅最大的。Visual inertia 是头号敌人，noise-as-mask 是头号解药。

对比 YUME (https://arxiv.org/abs/2507.17744) 用 mask-token 策略——随机 mask 掉 visual token。YUME 要改 architecture，Astra 只动 data，更轻量。

---

### 第 4 步：怎么处理不同类型的 action

不同场景 action 长得完全不一样：
- 自动驾驶：7-dim camera pose (nuScenes, https://www.nuscenes.org/)
- 走路探索：12-dim camera pose (Sekai, https://arxiv.org/abs/2506.15675)
- 机器人：7-dim end-effector pose (RT-1, https://robotics-transformer-x.github.io/)
- 键盘鼠标：discrete commands (SpatialVID, https://arxiv.org/abs/2509.09676)

一个 encoder 搞不定所有。于是有 Mixture of Action Experts (MoAE)：

1. 每个模态有自己的 projector $\mathcal{R}_m$，把 raw action 投到 shared space：

$$
\tilde{a}^i = \mathcal{R}_m(a_m^i), \quad m \in \{\text{cam}, \text{rob}, \text{cmd}\}
$$

2. Router 算 gating score：

$$
g^i = \text{Router}(\tilde{a}^i)
$$

3. Top-K expert 聚合：

$$
e^i = \sum_{k=1}^{K} g_k^i \cdot E_k(\tilde{a}^i)
$$

- $E_k$: 第 $k$ 个 expert (独立 MLP)
- $g_k^i$: 第 $k$ 个 expert 在 step $i$ 的 gating weight
- $e^i$: 最终 unified action embedding，喂给 ACT-Adapter

Paper 说实际只激活 top-1 expert (每步一个 expert)。还在 $\tilde{a}^i$ 上拼一个 binary indicator 标记是 past action 还是 current action，让 router 知道这个 action 是已经发生还是待执行。

MoAE 对单域性能提升不大 (Table 3: 0.651 → 0.669)，但它的价值是 **versatility**——一个模型同时能开车、能操作机器人、能控制 camera。

---

## 数据和训练

5 个数据集混合训：

| Dataset | Action | 场景 | 量 |
|---|---|---|---|
| nuScenes | Camera (7-dim) | 自动驾驶 | 850 |
| Sekai | Camera (12-dim) | 走路/drone | 50K |
| SpatialVID | Camera + KB/mouse | in-the-wild | 200K |
| RT-1 | Robot pose (7-dim) | 机器人抓取 | 9978 |
| Multi-Cam Video | Camera (12-dim) | 人体运动 | 136K |
| **Total** | | | **~397K (360 hours)** |

- Base model: Wan-2.1 (1.3B 版本)
- Resolution: 480×832, 20 FPS
- Target frames per chunk: 33
- Condition frames: 随机采样 [1, 128]
- Optimizer: AdamW, lr=1e-5
- 8 GPUs (80G), batch=1 per GPU
- 30 epochs, ~24 hours

参考 Wan-2.1: https://github.com/Wan-Video/Wan2.1

---

## 效果怎么样

### 主表 (Table 2)

| Method | Instr. Follow ↑ | Subj. Cons. ↑ | BG Cons. ↑ | Motion Smooth. ↑ | Aesthetic ↑ | Imaging ↑ |
|---|---|---|---|---|---|---|
| Wan-2.1 | 0.061 | 0.854 | 0.903 | 0.958 | 0.489 | 0.691 |
| MatrixGame | 0.268 | 0.916 | 0.928 | 0.981 | 0.441 | 0.748 |
| YUME | 0.652 | 0.936 | 0.938 | 0.985 | 0.523 | 0.741 |
| **Astra** | **0.669** | **0.939** | **0.945** | **0.989** | **0.531** | **0.747** |

Wan-2.1 的 Instr. Follow 0.061 基本是随机——它没有 action interface，根本听不懂指令。Astra 在所有 6 个 metric 全第一，说明 action conditioning 没牺牲 generation quality。

### Action alignment 硬指标 (Table A)

用 MegaSaM (https://arxiv.org/abs/2412.16891) 估生成视频的 camera pose，跟 ground truth 比：

| Method | RotErr ↓ | TransErr ↓ |
|---|---|---|
| Wan-2.1 | 2.96 | 7.37 |
| YUME | 2.20 | 5.80 |
| MatrixGame | 2.25 | 5.63 |
| NWM | 2.47 | 6.13 |
| **Astra** | **1.23** | **4.86** |

RotErr 几乎是 baseline 的一半。这是 action 跟随度的客观指标，比 human eval 可信。

### Parameter efficiency (Table B)

| Method | Trainable Params |
|---|---|
| NWM | ~1B (full tune) |
| YUME | ~14B (full tune) |
| MatrixGame | ~1.8B (full tune + cross-attn) |
| **Astra** | **366.8M** (adapter + self-attn) |

比 YUME 少 38 倍参数，性能还更好。这是 "freeze backbone + tiny adapter" 路线的强力证据。

---

## 直觉总结

把整件事翻译成大白话：

1. **Wan-2.1 已经懂很多世界知识** (3D、运动、物理直觉)，只是它生成方式是 one-shot 的，没有 action 入口。
2. **AR + flow matching** 把生成拆成 chunk 循环，就有了 per-step 注入 action 的入口。
3. **ACT-Adapter** 用 element-wise add 把 action 作为 latent shift 注入，模拟 optical flow 的效果。比 cross-attn 更自然。
4. **AFG** 放大 action 信号，让它不被视觉信号淹没。
5. **Noise-as-mask** 是最关键的 insight：训练时给 history 加噪声，打破模型对 history 的惰性依赖，逼它同时用 action 和 visual cue。推理时给 clean history，模型已经学会平衡。
6. **MoAE** 用 modality-specific experts 处理异构 action，一个模型通吃多场景。

整篇 paper 的味道：**不发明新技术，老技术 (AR + flow matching + adapter + MoE + CFG) 组合得恰到好处，每个组件都有清晰 intuition，ablation 干净利落**。这种工作我非常喜欢——solve 真问题，intuition 可迁移。

---

## 我会怎么延伸

如果让我接着做：

1. **Inference 加速**：50 步 denoising 太慢。用 consistency model (https://arxiv.org/abs/2303.01469) 或 LCM (https://arxiv.org/abs/2402.05608) 蒸馏到 4 步，real-time 才有意义。
2. **Action space 扩展**：现在只测了低维 pose (7-12 dim)。语言指令 ("把红色杯子拿起来")、long-horizon plan、tool use skill 这些复杂 action 没测。可以接 LLM 做 action decomposition。
3. **闭环 with VLA**：world model 做 imagination rollout，VLA policy (比如 π0, https://arxiv.org/abs/2410.24164) 在 rollout 里训 policy。这是 model-based RL 的经典套路，但用 generative world model 做 imagination 是新维度。
4. **Visual inertia 根因**：paper 说根因是 data bias (smooth motion dominant)。Noise-as-mask 是 symptom treatment。根因解法应该是收集 high-action-diversity data，或者用 synthetic data (Unity/Unreal 渲染) 平衡 action 分布。
5. **Multi-agent 量化**：Figure 8 只展示了 ego-car 超两车，没量化 multi-agent 物理 fidelity。跟 Genie (https://arxiv.org/abs/2401.04024) / Cosmos (https://arxiv.org/abs/2501.03575) 的 multi-agent benchmark 比还浅。
6. **Reward model**：训一个 reward model 评估 rollout 质量，做 RL fine-tuning。RLHF for world models。
7. **Long-horizon drift**：虽然 paper 说 Astra 长程稳定，但 8-10 秒还是短。真要 sim real world 得几分钟几小时，error accumulation 一定会出来。可能要加 explicit memory (retrieval) 而不只是 implicit latent memory。

---

## 参考

- Astra: https://eternalevan.github.io/Astra-project/ / https://github.com/EternalEvan/Astra
- Wan-2.1: https://github.com/Wan-Video/Wan2.1 / https://arxiv.org/abs/2503.20314
- Flow Matching: https://arxiv.org/abs/2210.02747
- Rectified Flow: https://arxiv.org/abs/2209.03003
- CFG: https://arxiv.org/abs/2207.12598
- YUME: https://arxiv.org/abs/2507.17744
- MatrixGame: https://arxiv.org/abs/2508.13009
- NWM: https://arxiv.org/abs/2502.00909
- MAGI-1: https://arxiv.org/abs/2505.13211
- Packing context (Zhang & Agrawala): https://arxiv.org/abs/2504.12626
- Genie: https://arxiv.org/abs/2401.04024
- Cosmos: https://arxiv.org/abs/2501.03575
- Sora: https://openai.com/research/video-generation-models-as-world-simulators
- iVideoGPT: https://arxiv.org/abs/2405.15211
- WorldDreamer: https://arxiv.org/abs/2401.09985
- WorldVLA: https://arxiv.org/abs/2506.21539
- MegaSaM: https://arxiv.org/abs/2412.16891
- Consistency Model: https://arxiv.org/abs/2303.01469
- LCM: https://arxiv.org/abs/2402.05608
- π0: https://arxiv.org/abs/2410.24164
- nuScenes: https://www.nuscenes.org/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- Sekai: https://arxiv.org/abs/2506.15675
- SpatialVID: https://arxiv.org/abs/2509.09676
- VBench: https://arxiv.org/abs/2311.13513
- DiT: https://arxiv.org/abs/2212.09748

---

# Astra: General Interactive World Model with Autoregressive Denoising

## 1. 一句话核心

Astra 把一个 pre-trained video diffusion backbone (Wan-2.1) 改造成一个**真正的交互式 world model**——能 chunk-by-chunk 自回归地吐视频，每一步都对 action 作即时响应。这个改造用了三件小事：ACT-Adapter (把 action 作为 latent feature shift 注入)、Noise-as-mask (打破 visual inertia)、MoAE (异构 action 模态统一)。计算开销几乎为零 (366.8M trainable params vs YUME 的 14B)。

Paper: https://arxiv.org/abs/2505. (实际是 https://eternalevan.github.io/Astra-project/)
Code: https://github.com/EternalEvan/Astra
Wan-2.1 base: https://github.com/Wan-Video/Wan2.1

---

## 2. 大背景：为什么 video generation ≠ world model

Section 3.2 抛出那个尖锐问题：**"Are T2V models really world models?"**

我的理解是——Sora / Wan / CogVideoX 这一类 T2V 模型虽然能生成令人惊艳的 clip，但它们生成的是 *self-contained short clips*，本质上是 *passive renderers*。你给一个 prompt，它给你一段视频，结束。

World model 的 defining property 是 **interactivity**: 在任意时刻注入任意 action，模型必须即时响应。这种"在线因果响应"是 T2V 的 cross-attention prompt 机制天然给不了的——prompt 是 global static signal，action 是 per-step causal signal。

Astra 的解法是把 diffusion 拆成 chunk-wise autoregressive loop：

$$
p(z^{1:N}) = \prod_{i=1}^{N} p(z^i \mid z^{<i}) \tag{1}
$$

变量解释：
- $z^{1:N}$: 整段视频被切成 $N$ 个 chunk，每个 chunk 是 VAE latent
- $z^i$: 第 $i$ 个 chunk 的 latent
- $z^{<i}$: 所有过去 chunk 的 latent 集合 (history)
- $p(\cdot \mid \cdot)$: 每个 chunk 的条件分布由一个 flow matching model 建模

这就把 *one-shot generation* 转成 *streaming interactive generation*。每生成完一个 chunk，就 append 到 history，下一个 chunk 的生成就基于最新 history + 最新 action——这就是 interactivity 的来源。

参考类似思路的 MAGI-1 (https://arxiv.org/abs/2505.13211) 和 StreamingT2V (https://arxiv.org/abs/2504.07357)，但它们没有 action interface。

---

## 3. 数学层面：Flow Matching + Autoregression

### 3.1 Flow matching 训练目标

公式 (2) 定义 noisy interpolation:

$$
z_t^i = (1 - t) \, z_0^i + t \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I), \quad t \in [0, 1] \tag{2}
$$

变量解释：
- $z_0^i$: 第 $i$ 个 chunk 的 clean latent (ground truth)
- $\epsilon$: 标准 Gaussian noise
- $t$: flow matching 时间参数。$t=0$ 时是 clean，$t=1$ 时是 pure noise
- $z_t^i$: 在时间 $t$ 上的 noisy interpolation

这是 rectified flow 的标准线性插值路径 (Lipman et al. 2023, https://arxiv.org/abs/2210.02747)。比起 DDPM 的 forward process，rectified flow 的 path 更直，few-step sampling 友好。

训练目标公式 (3):

$$
\mathcal{L}(\theta) = \mathbb{E}_{i, t, \epsilon} \Big[ \| v_\theta(z_t^i, t \mid z^{<i}) - v^*(z_t^i, t \mid z^{<i}) \|_2^2 \Big] \tag{3}
$$

变量解释：
- $v_\theta$: 可学习的 velocity field，参数 $\theta$
- $v^*$: ground-truth velocity。对于 rectified flow, $v^* = \epsilon - z_0^i$ (从 noise 指向 clean 的方向)
- $z^{<i}$: history condition 通过 causal attention 注入
- 期望对 $i$ (chunk index), $t$ (flow time), $\epsilon$ (noise) 三者取

**Intuition**: 训练时模型学一个 vector field——给定当前 noisy state $z_t^i$、flow time $t$、history $z^{<i}$，预测从 noise 走向 clean 的方向。推理时从 $\epsilon$ 出发沿 $v_\theta$ 积分到 $t=0$ 得到 clean chunk。

### 3.2 推理 loop

推理时:
1. 从 $z_1^{i+1} \sim \mathcal{N}(0, I)$ 出发
2. 用 ODE solver (Euler / DPMSolver) 沿 $v_\theta$ 从 $t=1$ 积分到 $t=0$
3. 得到 $z_0^{i+1}$ → VAE decode → 视频帧
4. Append $z_0^{i+1}$ 到 history → 下一个 chunk

这个 AR-denoising loop 是核心架构选择。Diffusion 提供 *local fidelity*，autoregression 提供 *long-horizon coherence* 和 *online interactivity*。

---

## 4. ACT-Adapter：action 作为 latent shift

### 4.1 设计哲学

Section 3.2 有个非常漂亮的 insight——把 action 类比为 **optical flow**。

Optical flow 公式：$I(x, t) = I(x + \delta x, t + \delta t)$。Action 就是在 latent space 上的位移 $\delta z$。

所以 action 不应该走 cross-attention (那是给 text prompt 的，全局静态条件)，而应该走 **element-wise addition** 直接调制 latent feature。这就是 ACT-Adapter。

### 4.2 架构细节

看 Figure 3(b)。Wan-2.1 是 30 个 stacked DiT block。每个 block 结构:
```
x → Self-Attn → FFN → x'
```

Astra 改成:
```
x → [frozen] Self-Attn → [trainable] ACT-Adapter(linear, init=identity) → +a_emb → FFN → x'
```

关键设计点：
- **Backbone 冻结**：除了 self-attention 层，所有参数冻结，最大化复用 pre-trained 知识
- **Adapter 初始化为 identity matrix**：训练开始时 adapter 是恒等映射，模型行为 = 原始 Wan-2.1，gradual deviation
- **Element-wise addition**：action embedding $a^i$ 加到每个 block 的 output，模拟 latent shift

这跟 LoRA 哲学类似但更轻——LoRA 是 low-rank decomposition，ACT-Adapter 就是 single linear layer。Table B 显示 Astra 只加 366.8M trainable params，YUME 全调 14B，MatrixGame 加 cross-attn adapters 共 1.8B。

### 4.3 Action-Free Guidance (AFG)

公式 (4) 是 classifier-free guidance 的 action 版本：

$$
v_{\text{guided}} = v_\theta(z_t, t, \varnothing) + s \cdot \big( v_\theta(z_t, t, a) - v_\theta(z_t, t, \varnothing) \big) \tag{4}
$$

变量解释：
- $v_\theta(z_t, t, \varnothing)$: action 被替换成 null token 时的 velocity (unconditional)
- $v_\theta(z_t, t, a)$: 给定真实 action 时的 velocity (conditional)
- $s$: guidance scale，inference 时设为 3.0 (见 Section A.3)
- $v_{\text{guided}}$: 最终用的 velocity field

**Intuition**: 训练时随机 drop action (10% 概率通常)，模型同时学到 "有 action" 和 "无 action" 两种 mode。推理时把这两个 mode 的差 *放大*——"无 action" 是 baseline trajectory，"有 action" 是 desired trajectory，差就是 action 的纯效应，乘 $s$ 放大，强制模型 follow action。

这跟 CFG (https://arxiv.org/abs/2207.12598) 完全同构，只不过 conditioning signal 从 class/text 换成 action。

Table 3 ablation: w/o AFG → Instruction Following 从 0.669 跌到 0.545，说明 action 信号被 visual context 淹没了，需要 guidance 把它"喊出来"。

---

## 5. Visual Inertia 与 Noise-as-Mask

### 5.1 现象

Section 3.3 + Appendix D 描述的现象非常有意思——**visual inertia**。

Figure C 的曲线：随着 history length 增加，video quality (subject/background consistency) 上升，但 action-following score **断崖式下降**。

为什么？因为真实世界视频里 95% 都是 smooth motion (相机平稳、物体缓动)。模型学到的 prior 是 "下一帧 ≈ 上一帧 + 小扰动"。给定一堆 clean history frames，模型直接 extrapolate 它们，action 信号被 drown out。

这是 generative world model 的根本张力：**history 给 consistency，action 给 responsiveness，两者此消彼长**。

### 5.2 Noise-as-Mask 解法

Astra 的解法极其简洁：训练时给 history latent 注入 *独立* 的 Gaussian noise:

$$
z_c^{\text{noisy}} = z_c + \sigma \cdot \eta, \quad \eta \sim \mathcal{N}(0, I)
$$

(注意：paper 没显式写这个公式，但 Section 3.3 描述清楚了——"injecting random noise into the conditioning video to degrade and blur its information content"。)

关键点：
1. **训练时 noisy，推理时 clean**：训练时模型见到的是 corrupted history，被迫 *同时* 用 action 和 visual cue；推理时给 clean history，模型已经学会平衡两个信号。
2. **Noise 独立于 diffusion noise**：flow matching 自己有 $\epsilon$，history noise 是另一个独立的 $\eta$。两者正交，不会干扰。
3. **无需新增参数**：跟 Mao et al. YUME (https://arxiv.org/abs/2507.17744) 的 mask-token 策略不同，YUME 要改 architecture，Astra 只是 data augmentation。

Table 3 ablation: w/o noise → Instruction Following 从 0.669 暴跌到 0.359！这是所有 ablation 里跌幅最大的。说明 visual inertia 是真问题，noise-as-mask 是真解药。

### 5.3 长程 history 压缩

为了进一步 extend history horizon 而不淹没 action，Astra 借用 Zhang & Agrawala (https://arxiv.org/abs/2504.12626) 的 packing trick：
- 保留第一帧 (anchor)
- 中间 history 压缩成 compact visual tokens
- 这样有效 history 可以到 128 frames (pixel space)

Table C: Astra interaction horizon 8-10 秒，Wan-2.1 / MatrixGame 只能几秒。

---

## 6. MoAE: Mixture of Action Experts

### 6.1 问题

异构 action 模态：
- Camera pose: 7-dim (nuScenes) 或 12-dim (Sekai, Multi-Cam)
- Robot end-effector pose: 7-dim (RT-1)
- Keyboard/mouse: discrete commands (SpatialVID)

单一 encoder 难以同时建模 continuous + discrete + 不同 dimensionality。

### 6.2 架构 (Figure 4)

Step 1: Modality-specific projector
$$
\tilde{a}^i = \mathcal{R}_m(a_m^i), \quad m \in \{\text{cam}, \text{rob}, \text{cmd}\}
$$

变量：
- $a_m^i$: 第 $i$ 步、模态 $m$ 的原始 action
- $\mathcal{R}_m$: 模态特定的线性 projector
- $\tilde{a}^i$: 投影到 shared action space 的特征

Step 2: Router 计算 gating
$$
g^i = \text{Router}(\tilde{a}^i)
$$

Router 是个 linear layer，输出 $K$ 个 gating score。

Step 3: Top-K 专家选择 + 聚合
$$
e^i = \sum_{k=1}^{K} g_k^i \cdot E_k(\tilde{a}^i) \tag{MoAE}
$$

变量：
- $E_k$: 第 $k$ 个 expert，独立 MLP
- $g_k^i$: 第 $k$ 个 expert 在 step $i$ 的 gating weight (softmax 后)
- $e^i$: 最终 unified action embedding，喂给 ACT-Adapter

每个 step 只激活 top-K 个 expert (paper 说"only one expert active per step"，意味着 top-1 routing)。

### 6.3 History-aware routing

Section 3.4 最后一句很关键：在 $\tilde{a}^i$ 上 augment 一个 binary indicator，标记是 *past* action 还是 *current* action。这样 router 能区分 "history action 已发生" vs "current action 待执行"，给不同 expert。

Table 3 ablation (w/o MoAE)：只训练 camera action 数据 → Instruction Following 0.651 vs 0.669，跌幅不大但丧失了 versatility。MoAE 的真正价值在 generalization——同一模型能开车、能操作机器人、能控制 camera。

---

## 7. Architecture 整体走查

把 Figure 3(a) 和 (b) 串起来：

**Training**:
1. Sample chunk $z^i$ + history $z^{<i}$ + action $a^{1:i}$ + prompt $c$
2. Encode to VAE latent
3. 对 history latent 加 noise (noise-as-mask)
4. Concat history + target along temporal dim → input to DiT
5. Sample flow time $t$, noise $\epsilon$, compute $z_t^i$
6. DiT forward (30 blocks, each with frozen self-attn + trainable ACT-Adapter + frozen FFN)
7. ACT-Adapter 接收 MoAE 输出的 $e^{1:i}$，element-wise add
8. 预测 velocity $\hat{v}$
9. Loss = MSE($\hat{v}$, $\epsilon - z_0^i$)
10. 随机 drop action (10%?) 训练 null-action branch

**Inference**:
1. 给 initial image + action stream
2. For chunk $i = 1, 2, \ldots$:
   a. 取 history (clean, no noise) → encode
   b. MoAE 处理 action $a^i$
   c. 采样 $z_1^i \sim \mathcal{N}(0, I)$
   d. 50 步 ODE 求解，每步用 AFG (公式 4, $s=3.0$)
   e. 得 $z_0^i$ → decode → video chunk
   f. Append to history
3. 输出 long-horizon video

---

## 8. 实验数据深度解读

### 8.1 Main Results (Table 2)

Astra vs SOTA:

| Method | Instr. Follow ↑ | Subj. Cons. ↑ | BG Cons. ↑ | Motion Smooth. ↑ | Aesthetic ↑ | Imaging ↑ |
|---|---|---|---|---|---|---|
| Wan-2.1 | 0.061 | 0.854 | 0.903 | 0.958 | 0.489 | 0.691 |
| MatrixGame | 0.268 | 0.916 | 0.928 | 0.981 | 0.441 | 0.748 |
| YUME | 0.652 | 0.936 | 0.938 | 0.985 | 0.523 | 0.741 |
| **Astra** | **0.669** | **0.939** | **0.945** | **0.989** | **0.531** | **0.747** |

Wan-2.1 的 Instr. Follow 0.061 几乎是 random——它根本不是 world model，没有 action interface。MatrixGame 0.268 略好但 game-specific。YUME 0.652 是最强 baseline，Astra 0.669 略胜。

值得注意：Astra 在所有 6 个 metric 上都第一，包括 visual quality。这说明 action conditioning 没有牺牲 generation quality——这正是 ACT-Adapter + frozen backbone 设计的成功。

### 8.2 Action Alignment (Table A)

这是更客观的 metric——用 MegaSaM (https://arxiv.org/abs/2412.16891) 估计生成视频的 camera pose，跟 ground truth 比：

| Method | RotErr ↓ | TransErr ↓ |
|---|---|---|
| Wan-2.1 | 2.96 | 7.37 |
| YUME | 2.20 | 5.80 |
| MatrixGame | 2.25 | 5.63 |
| NWM | 2.47 | 6.13 |
| **Astra** | **1.23** | **4.86** |

Astra 的 RotErr 几乎是 baseline 的一半！这是 action alignment 的硬指标，证明 ACT-Adapter + AFG + noise-as-mask 三件套合力把 action responsiveness 推到了 SOTA。

### 8.3 CityWalker 泛化 (Table D)

100 个 held-out scenes (比 Astra-Bench 的 20 大 5 倍)：

| Method | Instr. Follow ↑ |
|---|---|
| YUME | 0.619 |
| **Astra** | **0.641** |

差距 maintained，说明 Astra-Bench 不是 cherry-picked。

### 8.4 Ablation (Table 3) 排序贡献

按 Instruction Following 跌幅排序：
1. w/o noise: 0.669 → 0.359 (Δ=0.310) ← **最大**，visual inertia 是头号敌人
2. w/o AFG: 0.669 → 0.545 (Δ=0.124)
3. w/o MoAE: 0.669 → 0.651 (Δ=0.018)
4. cross-attn adapter: 0.669 → 0.642 (Δ=0.027)

insight: 
- Noise-as-mask 是 single highest-impact component
- AFG 第二重要
- MoAE 对 single-domain 性能影响小，价值在 versatility
- Cross-attn adapter (MatrixGame 用法) 比 ACT-Adapter 差，证明 element-wise add 是对的设计

### 8.5 Parameter Efficiency (Table B)

| Method | Trainable Params |
|---|---|
| NWM | ~1B (full tune) |
| YUME | ~14B (full tune) |
| MatrixGame | ~1.8B (full tune + cross-attn) |
| **Astra** | **366.8M** (adapters + self-attn) |

Astra 是 parameter-efficient 的极致——比 YUME 少 38×，比 MatrixGame 少 5×，性能还更好。这是 strong evidence 支持 "freeze backbone + tiny adapters" 路线。

---

## 9. Limitations 与我的看法

Section E 自陈 limitation: inference efficiency。每 chunk 要 50 步 denoising，无法实时。

我的补充看法：

**优点**:
1. ACT-Adapter 的 optical-flow 类比是漂亮的 intuition——action 是 latent shift，所以 element-wise add 比 cross-attn 自然。
2. Noise-as-mask 的 *training-noisy / inference-clean* asymmetry 是巧妙设计，本质上是一种 *dropout for visual modality*，强迫模型用 action。
3. 三件套互相 *正交*，可以独立 ablation，engineering 友好。
4. Parameter efficiency 令人印象深刻，证明 pre-trained video diffusion 已经蕴含大量 world knowledge，只需轻调即可变 world model。

**潜在疑虑**:
1. **Instruction Following 主要靠 human eval** (Section A.4)——20 个 user 看视频打分，subjective。MegaSaM-based RotErr/TransErr 是补充，但 MegaSaM 本身在 generated video 上可靠性也存疑 (paper 自己承认)。
2. **Action space 局限**: 都是低维 (7-12 dim) camera/robot pose。真正复杂的 action (语言指令、long-horizon plan、tool use skill) 没测。
3. **Visual inertia 的根因**: paper 把它归为 "real-world datasets contain predominantly smooth motions"。这暗示 *data bias* 是根因。那 noise-as-mask 只是 symptom treatment，根因解法应该是 *收集更多 high-action-diversity data*。
4. **Multi-agent interaction** (Figure 8) 只展示了 ego-car 超两车，没量化 multi-agent 物理 fidelity。这跟 Genie (https://arxiv.org/abs/2401.04024) / Cosmos (https://arxiv.org/abs/2501.03575) 的 multi-agent benchmark 比还浅。
5. **Ood generalization** (Figure A, Minecraft / anime / indoor) 是 qualitative only，没数值。Minecraft 这种 blocky 低复杂度场景容易 overfit 视觉风格，但物理交互是否正确没说。

**未来方向**:
- 蒸馏到 few-step (consistency model / LCM) 解决 latency
- 加入 reward model 做 RL fine-tuning (RLHF for world models)
- 跟 VLA models (π0, OpenVLA) 闭环: world model 做 imagination, VLA 做 policy
- Multi-agent benchmark 量化物理 fidelity

---

## 10. 在 world model landscape 中的定位

把 Astra 放到 2025 年 world model 全景看：

- **Genie 2/3** (DeepMind, https://arxiv.org/abs/2401.04024): closed, latent action space, 通用但不可控 action
- **Cosmos** (NVIDIA, https://arxiv.org/abs/2501.03575): 大规模 physical AI foundation model, 多模态
- **Sora** (OpenAI, https://openai.com/research/video-generation-models-as-world-simulators): T2V, 声称 world simulator 但无 action interface
- **YUME** (https://arxiv.org/abs/2507.17744): walking-specific, mask-token 策略
- **MatrixGame** (https://arxiv.org/abs/2508.13009): game-specific, cross-attn adapter
- **NWM** (Bar et al., https://arxiv.org/abs/2502.00909): navigation-focused
- **iVideoGPT** (https://arxiv.org/abs/2405.15211): interactive VideoGPT, masked-token prediction in discrete latent
- **WorldDreamer** (https://arxiv.org/abs/2401.09985): masked-token in discrete latent
- **WorldVLA** (https://arxiv.org/abs/2506.21539): joint VLA + world model
- **Astra**: general-purpose, action-conditioned, AR-denoising, parameter-efficient

Astra 的独特价值：**用最少的参数实现了最广的 action modality 覆盖**。它在 versatility 和 efficiency 两个轴上同时领先，这是 engineering 上的 sweet spot。

理论 novelty 不算高——每个组件 (AR + diffusion, adapter, MoE, CFG) 都是已有技术。但 *组合方式* 和 *intuition* (action as latent shift, noise-as-mask vs visual inertia, modality-specialized experts) 是真正 contribute 的 insight。这种 paper 我很喜欢——不炫技，solve 真问题，intuition 清晰可复用。

---

## 参考

- Astra project: https://eternalevan.github.io/Astra-project/
- Astra code: https://github.com/EternalEvan/Astra
- Wan-2.1: https://github.com/Wan-Video/Wan2.1 / https://arxiv.org/abs/2503.20314
- Flow Matching: https://arxiv.org/abs/2210.02747
- Rectified Flow: https://arxiv.org/abs/2209.03003
- CFG: https://arxiv.org/abs/2207.12598
- YUME: https://arxiv.org/abs/2507.17744
- MatrixGame: https://arxiv.org/abs/2508.13009
- NWM: https://arxiv.org/abs/2502.00909
- MAGI-1: https://arxiv.org/abs/2505.13211
- StreamingT2V: https://arxiv.org/abs/2403.14795
- Packing context (Zhang & Agrawala): https://arxiv.org/abs/2504.12626
- Genie: https://arxiv.org/abs/2401.04024
- Cosmos: https://arxiv.org/abs/2501.03575
- Sora: https://openai.com/research/video-generation-models-as-world-simulators
- iVideoGPT: https://arxiv.org/abs/2405.15211
- WorldDreamer: https://arxiv.org/abs/2401.09985
- WorldVLA: https://arxiv.org/abs/2506.21539
- MegaSaM: https://arxiv.org/abs/2412.16891
- nuScenes: https://www.nuscenes.org/
- RT-1 / Open X-Embodiment: https://robotics-transformer-x.github.io/
- Sekai: https://arxiv.org/abs/2506.15675
- SpatialVID: https://arxiv.org/abs/2509.09676
- VBench: https://arxiv.org/abs/2311.13513
- DiT: https://arxiv.org/abs/2212.09748
