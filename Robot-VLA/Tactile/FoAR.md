---
source_pdf: FoAR.pdf
paper_sha256: bc6061404a8d41d1cf23288385bb6604b87494d329c7a572074d37e6516c53fd
processed_at: '2026-08-04T09:48:48-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Karpathy，如果用大白话和直觉来聊这篇 paper，核心其实就讲了一个道理：**多模态融合，不能无脑 concat，得看场合下菜碟。**

我们把这个事情拆开来看。

### 1. 痛点：视觉是“瞎子”，力觉是“噪音”

做机器人接触丰富的任务（contact-rich manipulation，比如擦白板、削黄瓜皮、切辣椒），光靠 RGB-D 相机是不行的。视觉能看见“刀在辣椒上方”，但刀到底碰没碰到辣椒、使了多大劲，视觉是看不出来的。Fig. 1 里面展示得很清楚，接触前后的 point cloud 几乎一模一样，纯视觉 policy $\pi(p_t) = a_t$ 根本不知道当前处于什么状态。

那直接加 force/torque (F/T) 传感器不就行了？以前的研究确实这么干，比如直接把 F/T 数据和 vision feature 拼一起（concat）或者丢进同一个 Transformer。但这会引发一个致命问题：**F/T 数据是稀疏激活的**。
一个任务分为非接触阶段（去拿刀、空中移动）和接触阶段（切辣椒）。在非接触阶段，F/T 传感器读到的全是底噪（millinewton 级），这些没用的噪声一旦进到 Transformer 里，会干扰 attention head 的判断。Table I 里有个 baseline 叫 `RISE (force-token)`，它在擦白板任务中，连“抓取橡皮”这个非接触动作的成功率都从 100% 掉到了 85%。这就是被噪声坑了。

### 2. FoAR 的绝招：Future Contact Predictor 门控

既然 F/T 数据在非接触阶段是毒药，在接触阶段是解药，那我们就需要一个“门”来控制它。FoAR 引入了一个 **future contact predictor**，输出一个概率 $\phi(t) \in [0,1]$，表示“未来几秒内会不会发生接触”。

这个 $\phi(t)$ 怎么用？看融合公式：

$$ h_t = \big[\, h_t^s \,;\, \phi(t) \cdot h_t^f + (1-\phi(t)) \cdot h^* \,\big] $$

变量解释：
- $h_t^s$：视觉 scene feature
- $h_t^f$：力觉 force feature
- $h^*$：一个 learnable embedding，你可以把它理解为“我不知道力觉是啥”的默认占位符
- $\phi(t)$：门控概率

**直觉解释**：
如果预测接下来要接触了（$\phi \to 1$），公式就变成 $[h_t^s; h_t^f]$，把力觉 feature 全力放进来；
如果预测不接触（$\phi \to 0$），公式就变成 $[h_t^s; h^*]$，把力觉 feature 替换成那个无害的占位符 $h^*$，把噪声彻底屏蔽掉。

这就好比自动驾驶里的 Mixture of Experts (MoE) 或者 Switch Transformer，不同模态的可靠性随时间变化，我们不能让 model 自己硬学这个 gate，因为接触阶段太短，loss 被非接触阶段主导，gate 学不出来。必须用显式的监督信号（通过判断未来 F/T 是否超阈值来打标签）把这个 gate 训出来。

### 3. Reactive Control：简单粗暴的“再往前凑凑”

Policy 输出的是一个 2 秒的 action chunk，这是开环的。但现实中，抓刀的位置稍微偏了几毫米，或者白板倾斜了一点，policy 预测的“接触时刻”就不准了。

FoAR 在部署时加了一个 Reactive Control 策略。逻辑很简单（见 Algorithm 1）：
1. Policy 说“接下来要接触了”（$\phi > 0.9$）；
2. 控制器看一眼当前的 F/T 传感器；
3. 如果发现没读数（力 < 8N，力矩 < 5N·m），说明“预测接触了但其实还没碰上”；
4. 怎么办？顺着 policy 预测的方向，再往前硬走一个小步长 $\epsilon = 0.006$ m (6mm)；
5. 一旦 F/T 超过阈值，立刻停止微调，让 policy 接管。

这本质上就是工业界用了几十年的“search-and-insert”策略，只不过以前是随机螺旋搜索，现在是 learned policy 提供搜索方向。它的好处是：不需要调那些恶心的 admittance control 或 impedance control 参数（比如 stiffness matrix），全任务共用一套阈值，简单 position control 就能把活干了。

### 4. 实验效果：降维打击

看 Table I 的数据，FoAR 在擦白板任务上 score 是 0.875，而纯视觉 RISE 是 0.500，无脑融合的 `RISE (force-concat)` 是 0.475。FoAR 几乎是翻倍的提升，而且接触和非接触阶段的成功率都是 100%。

更有意思的是 Chopping 任务（Table II），切辣椒。纯视觉 RISE 平均只能切出 1.8 段，而且段长方差极大（std 0.411）；FoAR 能切出 3.9 段，段长方差只有 0.094。切菜这种瞬间冲击力的任务，没有 force feedback 根本不可能控制好深度，视觉是完全瞎的。

### 5. 深层联想：多模态的未来在于 Scheduling

Karpathy，聊到这儿，我觉得这篇 paper 给了我一个很强的 meta intuition。

大家都爱搞多模态，音频、视频、触觉、力觉全塞进去。但这篇 paper 指出：**模态不是越多越好，而是要在对的 phase 用对的模态。** 

这跟 Tesla Autopilot 的传感器融合逻辑很像：雷达在远距离测距是金子，但在近距离 bumper 里全是噪点，所以得有逻辑去 gate 它。FoAR 把这个思想用 end-to-end learning 实现了，用一个轻量级的 future contact predictor 去动态调度 modality 的权重。

我觉得这个思路极其 generalizable。比如：
- **Audio modality**: 只有在 rubbing/scratching phase 才有用，其他时候是环境噪音。
- **Language instruction**: 只有在 task transition phase 才需要 attend，执行精细动作时不需要一直听 LLM 唠叨。

未来的多模态 policy，与其去卷更大的 backbone，不如去研究怎么设计更精细的 phase-aware gating 机制。把“什么时候用什么模态”这个决策本身，也变成 policy 学习的一部分。

相关参考链接：
- [FoAR Project Page](https://tonyfang.net/FoAR/)
- [RISE Policy (Backbone)](https://arxiv.org/abs/2404.12281)
- [Diffusion Policy (Action Head)](https://diffusion-policy.cs.columbia.edu/)
- [MoE / Switch Transformer 思想类比](https://arxiv.org/abs/2101.03961)

---

# FoAR: Force-Aware Reactive Policy — 深度技术解读

## 1. Paper 全景

FoAR 来自 SJTU 的 MVIG / SJTU-VARSYS 实验室（方浩树、卢策吾团队），是 RISE policy [Wang et al. 2024] 的"force-aware"演化版本。核心 thesis 一句话：

> Force/Torque 是 **sparse-activated modality**——在 contact phase 它是金子，在 non-contact phase 它是噪声。直接 concat 进 policy 会污染决策；用一个可微分的 future contact probability φ(t) 去 gate force feature 的融合，配合 reactive control，就能用 **简单 position control** 完成 contact-rich 任务。

Project page: https://tonyfang.net/FoAR/  
RISE backbone: https://arxiv.org/abs/2404.12281  
Diffusion Policy (action head): https://diffusion-policy.cs.columbia.edu/  

---

## 2. Motivation — 为什么 naive 多模态融合会失败

### 2.1 Vision-only policy 的盲点

Contact-rich 任务（wiping、peeling、chopping）的本质是 **sustained dynamic interaction with environment**。Karpathy 你应该会立刻联想到：纯视觉 policy 学到的是一个 $\pi(o_t) \to a_t$ 的 map，但 contact state 是一个 **hidden variable**——RGB image 在 contact 前后几乎不可区分（Fig.1 中 point cloud 几乎一致），信息完全藏在 force/torque 信号里。

这点在 sim-to-real 里其实早就被很多人发现了——AnyTeleop、Diffusion Policy 在 opening drawer、insertion 这种任务上一旦有 mm 级 misalignment 就崩，因为 vision 看不出"我其实还没碰到"。

### 2.2 Naive multimodal fusion 的问题

已有工作（[Liu et al. ForceMimic 2024](https://arxiv.org/abs/2410.07554)、[Wu et al. TacDiffusion 2024](https://arxiv.org/abs/2409.11047)、[Hou et al. ACP 2024](https://arxiv.org/abs/2410.09309)）基本都假设 object 已经 grasp 在 gripper 里——这等于 **bypass 掉了 non-contact phase**。但真实任务是多 phase 的：

```
Phase 1: approach (non-contact, F/T = noise floor ~ mN)
Phase 2: grasp   (transient contact)
Phase 3: wipe    (sustained contact, F/T = N 级)
Phase 4: retreat (non-contact)
```

如果在 phase 1/4 把 F/T 噪声塞进 Transformer，attention head 会被误导。Table I 里 `RISE (force-token)` 在 wiping 的 grasp ASR 只有 85%，比纯 vision RISE 的 100% 还低——这就是直接证据。force noise 在非接触阶段 **actively hurt** 了 policy。

### 2.3 Audio / Tactile 为什么不是首选

Paper 里花了一段讨论：
- **Audio** (ManiWAV [Liu 2024](https://arxiv.org/abs/2410.13062)): susceptible to background noise，indirect
- **Tactile** (GelSight/DIGIT [Yuan 2017](https://arxiv.org/abs/1706.04826); ReSkin [Bhirangi 2021](https://arxiv.org/abs/2107.08059); AnySkin [2024](https://arxiv.org/abs/2409.08276)): sensor heterogeneity 让 representation 难以标准化，更换 sensor 还有 inconsistency
- **Force/Torque**: 直接、standardized (6-axis F/T sensor 几乎是工业标准)、便宜、calibration 稳定

这个 trade-off 很关键——选择 F/T 是 **engineering pragmatism** 的胜利。

---

## 3. Architecture 详解

整体结构（Fig.2）：

```
RGB-D → Point Cloud ──→ Sparse 3D Encoder ──→ Transformer ──→ h_t^s (scene)
                                                              │
F/T history ─→ MLP ─→ Force tokens ─→ Transformer ─→ h_t^f    │
                                                              │
RGB + F/T ──→ ResNet18 + MLP ──→ Linear ──→ φ(t) ─────────────┤
                                                              ▼
                                            h_t = [h_t^s ; φ·h_t^f + (1-φ)·h*]
                                                              │
                                                              ▼
                                            Diffusion Action Head ─→ a_{t:t+T_a}
```

### 3.1 Point Cloud Encoder（继承 RISE）

输入：$p_t \in \mathbb{R}^{N_t \times 6}$（XYZ + RGB，单视角 RGB-D 反投影出来的 point cloud）。

- **MinkowskiEngine** [Choy 2019](https://github.com/NVIDIA/MinkowskiEngine)，voxel size = **5mm**，sparse 3D convolution + shallow ResNet → sparse point tokens $\bar{P}_t \in \mathbb{R}^{N_p \times 512}$。$N_p$ 是 sparse voxel 数量。
- 然后 **4 encoder blocks + 1 decoder block + readout token** 的 Transformer，$d_{model}=512$, $d_{ff}=2048$ → scene feature $h_t^s \in \mathbb{R}^{512}$。

这里 RISE 的精髓是 **point cloud 在 camera frame 内不 cropped 到 gripper 周围**——而是整个 workspace 的 sparse 表示，让 Transformer 自己学会 attend。FoAR 完全继承。

### 3.2 Force/Torque Encoder

输入：$f_{t-T_o:t} \in \mathbb{R}^{T_o \times 6}$，$T_o = 200$ step @ 100Hz ≈ 2 秒历史窗口，每 step 是 $(F_x, F_y, F_z, \tau_x, \tau_y, \tau_z)$ 6 维。

- 每个 $f_t$ 先过 3-layer MLP (64 → 128 → 512) → force token $\bar{F}_t \in \mathbb{R}^{512}$
- 200 个 token 喂给 **Transformer + sinusoidal positional encoding（temporal axis）** → readout token 输出 $h_t^f \in \mathbb{R}^{512}$

为什么用 Transformer 不用 MLP？Table V 的 ablation 给出答案——MLP 版本 Peeling score 从 0.588 掉到 0.426。F/T 是 **时序信号**，contact transition 在 100ms 量级，必须 capture temporal pattern。MLP 只看 200×6 的 flat vector，丢失了 "force 在 t-50ms 时刻突然上升" 这种 event structure。

Karpathy 你会立刻想到——这跟 audio waveform 需要 Transformer/conv 而非 MLP 是一个道理，time-locality 必须由 architecture inductive bias 进去。

### 3.3 Future Contact Predictor

这是 paper 的 **key contribution**。

- Input: RGB image $I_t$（**不是** point cloud，理由是 lightweight + contact state 判断 RGB 就够了）+ F/T history $f_{t-T_o:t}$
- Encoder: **ResNet18** for RGB + 2-layer MLP (128→512) for F/T
- Concat → Linear → sigmoid → $\phi(t) \in [0,1]$

**Ground truth 怎么构造？**
看 demo 中 $[t-2\text{s}, t+2\text{s}]$ 时间窗内 F/T 是否超过阈值 $\delta_{demo}$——超过就 label = 1，否则 0。注意这是 **未来** 窗口，预测的是"接下来会不会 contact"，而非"现在是否 contact"。

这里有个微妙的设计选择：为什么 future 不是 current？因为 action 是 future-oriented（$T_a=20$ step @ 10Hz = 2 秒未来动作）。如果用 current contact state，policy 还没来得及反应；用 future contact state，policy 可以 **预先** 把 force modality 的 weight 调高，等真的 contact 时 feature 已经准备好了。

这与 MPC 的 prediction horizon 思想一致，也与 Trajectory Prediction 中"先 predict intent 再 predict trajectory"的两阶段思路同源。

### 3.4 Feature Fusion — Gating Mechanism

公式：

$$
h_t = \big[\, h_t^s \,;\, \phi(t) \cdot h_t^f + (1-\phi(t)) \cdot h^* \,\big]
$$

变量解释：
- $h_t$：fused feature，输入到 diffusion action head
- $h_t^s \in \mathbb{R}^{512}$：scene feature（vision）
- $h_t^f \in \mathbb{R}^{512}$：force feature
- $\phi(t) \in [0,1]$：future contact probability
- $h^* \in \mathbb{R}^{512}$：**learnable neutral embedding**，初始化随机，与 policy 一起 end-to-end 训练
- $[\cdot;\cdot]$：concatenation（结果维度 $\mathbb{R}^{1024}$）

**直觉**：
- 当 $\phi \to 1$（即将 contact）：fusion 几乎等于 $[h_t^s; h_t^f]$，force 信息 full strength
- 当 $\phi \to 0$（不会 contact）：fusion 等于 $[h_t^s; h^*]$，force 被替换成 "null token" $h^*$
- 中间状态：linear interpolation，可微分，端到端训练

这个 $h^*$ 类比一下 Karpathy 你应该会很熟：
- BERT 的 `[CLS]` token 也是一种 "summary slot"
- Switch Transformer / MoE 的 router 决定 expert 权重
- LSTM 的 forget gate 决定 hidden state 的更新

更准确地说，这个机制类似于 **learnable default + soft gate**——当某 modality 不可靠时，用一个 learned "I don't know" embedding 填位，避免 noise 注入。这个思想在 Perceiver IO、DINOv2 的 register token、甚至 VAE 的 prior 中都有 echo。

### 3.5 Action Head — Diffusion Policy

继承 [Diffusion Policy Chi 2023](https://diffusion-policy.cs.columbia.edu/)：
- CNN-based 1D diffusion head（temporal convolution over action sequence）
- 100 denoising iterations for training，20 DDIM [Song 2021](https://arxiv.org/abs/2010.02502) iterations for inference
- Predict $T_a = 20$ step future actions @ 10Hz → 2 秒 action chunk
- Conditioning input 是 fused feature $h_t$

### 3.6 Loss

$$
\mathcal{L} = \mathcal{L}_{action} + \alpha \, \mathcal{L}_{predictor}
$$

- $\mathcal{L}_{action}$：diffusion L2 reconstruction loss on ground-truth actions
- $\mathcal{L}_{predictor}$：binary cross-entropy on future contact label
- $\alpha = 0.1$：predictor loss 权重，小一点防止 dominate 主任务

---

## 4. Reactive Control — Algorithm 1 拆解

部署时不仅跑 policy，还跑一个 **force-aware reactive loop**：

```python
for t in range(N_max):
    if t % N_inference == 0:   # policy inference step
        phi, a_chunk = FoAR(p_t, f_hist, I_t)
        if phi < delta_phi:   # NON-CONTACT phase
            buffer.add(a_chunk)
        else:                 # CONTACT phase
            if force(f_t) < delta_f AND torque(f_t) < delta_t:
                # 预测说要 contact，但实测还没接触 → 往预测方向推一小步
                d = a_chunk[:T_f].pos - q_t.pos
                a_chunk.pos += epsilon * d / ||d||_2
            contact_buffer.add(a_chunk)
    a_t = buffer.get(t) if phi < delta_phi else contact_buffer.get(t)
    robot.execute(a_t)
```

参数：
- $\delta_\phi = 0.9$（high threshold 防误触发）
- $\delta_f = 8\text{N}$, $\delta_t = 5\text{N·m}$
- $\epsilon = 0.006\text{m} = 6\text{mm}$
- $T_f = 5$ step 用于估计 future direction（防 noise，用 5 步平均）

### 4.1 这个 reactive control 的 intuition

Policy 预测的 action chunk 是开环的——它预测"接下来 2 秒这么走"。但 demo 和 deployment 之间有 sim-to-real gap：
- Tool grasp 位置变了 → end-effector 相对 tool 的几何关系变了
- Whiteboard 倾斜了 → wiping normal direction 变了
- Cucumber 弯曲了 → peeling path 不再是直线

Policy 学到的 contact onset 时机不会 100% 准。Reactive control 干一件事：**如果 policy 说"接下来要 contact"，但 force sensor 显示"还没 contact"，那就朝 policy 预测的方向再多走 6mm**。一旦 force 超过 8N，立刻停止 correction，让 policy 接管。

这本质上是 **force-triggered position creep**，类似 industrial assembly 里的 search-and-insert strategy，但用 learned policy 提供搜索方向，而不是 random spiral。

### 4.2 为什么不用 admittance / impedance / hybrid force-position control？

Paper 在 §III-C 明确对比：
- [Admittance Zhou 2024](https://arxiv.org/abs/2409.14440): 需要预测 desired force，需要 stiffness parameter
- [Compliance Hou 2024](https://arxiv.org/abs/2410.09309): 需要预设 stiffness matrix
- [Hybrid Liu 2024 ForceMimic](https://arxiv.org/abs/2410.07554): 需要 contact wrench direction 信息

FoAR 的 reactive control **零参数调优**——只有 4 个 threshold / step size 常数，所有任务都用同一组。这是工程上的胜利，因为 contact-rich task 每个 task 调 stiffness 是很痛的。

### 4.3 Dual temporal ensemble buffer

注意 algorithm 里 `buffer` 和 `contact_buffer` 是分开的。这是因为 contact phase 和 non-contact phase 的 action distribution 差异很大——如果共享 buffer，temporal ensemble [Zhao 2023 ACT](https://tonyzhaozh.github.io/aloha/) 会把两类 action 平均掉，得到不伦不类的 trajectory。

Karpathy 你应该记得 ACT 的 temporal ensemble 是把 overlapping chunks 加权平均，这里两套 buffer 等于 **phase-aware temporal ensemble**。

---

## 5. Experiments — 数据说话

### 5.1 Setup

- Robot: Flexiv Rizon（impedance controlled 7-DOF arm）
- Gripper: Dahuan AG-95
- F/T sensor: OptoForce（mounted flange↔gripper，6-axis，100Hz）
- Camera: Intel RealSense D435（global view，RGB-D）
- GPU: RTX 3090 + i9-10900K
- Workspace: 45×60×40 cm
- Demos: 50/task（Chopping 40），haptic teleoperation 收集

### 5.2 Tasks（Fig.3）

三个 task 覆盖 contact 的两种形态：
1. **Wiping**（surface force control, sustained contact）：擦白板，grasp 位置可变
2. **Wiping (General)**：白板方向任意
3. **Peeling**（surface force control, precision）：削黄瓜皮，需要 mm 级力控制
4. **Chopping**（instantaneous force impact）：切辣椒，瞬态冲击力

### 5.3 Main Results — Table I

| Method | Wiping Score | Wiping-Gen Score | Peeling Score |
|---|---|---|---|
| ACT | 0.275 | 0.250 | 0.120 |
| Diffusion Policy | 0.400 | 0.350 | 0.386 |
| RISE | 0.500 | 0.500 | 0.377 |
| RISE (force-token) | 0.575 | 0.600 | 0.487 |
| RISE (force-concat) | 0.475 | 0.675 | 0.524 |
| FoAR (3D-cls) | 0.175 | 0.200 | 0.270 |
| **FoAR** | **0.875** | **0.850** | **0.756** |

观察：
1. FoAR 比 RISE 提升 **+0.375 / +0.350 / +0.379**——非常大幅度的提升
2. `force-token` / `force-concat` 这种 naive 融合只有 marginal 提升（甚至某些 grasp ASR 还掉）——**证明 gating 是必须的**
3. `FoAR (3D-cls)` 用 point cloud 替代 RGB 给 contact predictor，性能暴跌到 0.175——**vision encoder 共享会互相干扰**。Contact predictor 关心"是否在白板上方"，policy 关心"eraser 精确位置 + ee pose"，两类 feature 完全不同，share backbone 会 conflict

### 5.4 Chopping — Table II

| Method | # Segments | Norm. Length Avg | Norm. Length Std |
|---|---|---|---|
| RISE | 1.8±0.6 | 0.727 | 0.411 |
| **FoAR** | **3.9±0.9** | **0.353** | **0.094** |
| Oracle | 5.0±0.0 | 0.200 | 0.056 |

FoAR 切出 ~4 段均匀辣椒，segment length std 0.094 vs RISE 0.411——**4 倍精度提升**。Chopping 是瞬态冲击任务，对 force feedback 的需求最刚性，结果最 dramatic。

### 5.5 Ablation — Table III (Wiping)

| F/T Freq | w/ Predictor | w/ Reactive | Score |
|---|---|---|---|
| 100Hz | ✗ | ✓ | 0.650 |
| 100Hz | ✓ | ✗ | 0.650 |
| 2Hz | ✓ | ✓ | 0.625 |
| 10Hz | ✓ | ✓ | 0.800 |
| **100Hz** | **✓** | **✓** | **0.875** |

三个 component 都是必要的：
- 缺 predictor：0.875 → 0.650（-0.225）
- 缺 reactive：0.875 → 0.650（-0.225）
- 缺高频 F/T：100Hz → 10Hz 掉 0.075，→ 2Hz 掉 0.250

**Frequency 的 effect 是 non-linear 的**——2Hz→10Hz 提升 0.175，10Hz→100Hz 提升 0.075。Karpathy 你可以推断：高频主要为了捕捉 contact onset event（< 100ms 量级），10Hz 已经能抓到主要事件，100Hz 是 marginal 改进但值得。

### 5.6 Robustness — Table IV

三种 dynamic disturbance：Rewrite（擦完再写）、Move（擦完移动白板）、Rewrite+Move：

| Method | Original | Rewrite | Move | Rewrite+Move |
|---|---|---|---|---|
| RISE | 0.500 | 0.500 | 0.600 | 0.500 |
| RISE (force-token) | 0.600 | 0.450 | 0.500 | 0.600 |
| **FoAR** | **0.850** | **0.800** | **0.850** | **0.800** |

FoAR 在所有 disturbance 下保持 0.80+，**没有 collapse**。这归功于：
1. Force-aware gating 让 policy 在 phase transition 时 robust
2. Reactive control 自适应 contact onset 偏差
3. 继承 RISE 的 generalization（RISE 本身也很 robust）

### 5.7 Force Encoder Ablation — Table V (Peeling)

| Method | Score | Peel ASR |
|---|---|---|
| RISE | 0.293 | 50% |
| FoAR (MLP) | 0.426 | 75% |
| **FoAR (Transformer)** | **0.588** | **100%** |

证实 Transformer 对时序 F/T 的必要性。MLP 失去 temporal attention，捕捉不到"force gradient over 100ms"这种 dynamic pattern。

---

## 6. Intuitive Summary — 这篇 paper 的关键 insight

### 6.1 Sparse modalities 需要 gating，不是 concat

这是 paper 最 deep 的贡献。Karpathy 你应该会联想到：
- Mixture-of-Experts：sparse activation + gating
- Switch Transformer：top-1 routing
- Early-exit network：confidence-gated
- DropConnect / Dropout：random gating

FoAR 的 gating 是 **input-conditioned soft gate**，由 future contact predictor 决定。这是 multimodal fusion 的一个 generalizable insight：**不同模态在不同 phase 的 reliability 不同**，naive concat 等于强制 policy 自己学 gate——但 contact phase 比例小，loss 被 non-contact phase 主导，gate 学不出来。Explicit gate + supervised signal（BCE on future contact）是 sample-efficient 的解。

### 6.2 Future > Current

预测 "接下来会不会 contact" 而非"现在是否 contact"——这是 **predictive state** 思想，与 model-based RL 中的 world model、Trajectory Forecasting 中的 intention prediction一脉相承。Policy 需要前瞻，因为 action 是 future-oriented chunk。

### 6.3 Reactive Control 是 Sim-to-Real 的补丁

Learned policy 永远会有 distribution shift。Force-triggered reactive correction 是一个 model-free 的 sim-to-real bridge——不需要精确的 dynamics model，只需要一个 threshold 和一个小步长。这跟工业装配里的 "search and insert" 同源，但用 learned policy 提供搜索方向。

### 6.4 Architecture 即 Inductive Bias

- Point cloud + sparse 3D conv：保留 spatial geometry
- F/T + Transformer：保留 temporal pattern
- Future contact predictor 单独 RGB encoder：避免 backbone 共享 conflict
- Diffusion head：multi-modal action distribution

每个组件都对应一个特定的 inductive bias，没有 over-engineering。

---

## 7. 我会提的 Critical Thoughts

虽然 paper 写得很 solid，但有一些值得 push 的点：

1. **Static threshold limitation**：$\delta_f = 8\text{N}$, $\delta_t = 5\text{N·m}$ 是手调的，每个 task 可能需要不同阈值。Paper 自己在 conclusion 提到了这点。可以想象一个 **learned adaptive threshold** 或 task-conditional threshold。

2. **Future contact prediction 是 binary**：用 sigmoid 输出 single probability。更精细的版本可以预测 **contact onset time**（regression）或 **contact duration**，给 reactive control 更精细的 scheduling。

3. **Reactive control 只能 "push harder"**：当 force 不足时往前推 6mm。但如果 force 过大（over-contact），policy 没有 "pull back" 机制。这对 fragile object（鸡蛋、莓果）可能危险。一个对称的 "if force > upper_threshold, retreat" 会让它更 universal。

4. **Single-arm only**：作者在 conclusion 提到 dual-arm / dexterous hand 是 future work。Bimanual contact-rich（比如双手剥香蕉）会引入新的 phase coupling 问题。

5. **Point cloud 还是 single-view**：从 single RGB-D 反投影，occlusion 严重。Multi-view 或 in-hand vision 会是自然扩展。

6. **与 Diffusion Policy 的对比维度**：FoAR 用的是 CNN-based 1D diffusion（Diffusion Policy 默认），最近 [3D Diffusion Policy](https://3d-diffusion-policy.github.io/)、$\pi_0$、OpenVLA 等都有新 action head 设计，可以替换测试。

7. **Future contact predictor 只用 RGB**：作者 argue 是 lightweight，但 RGB 的 depth 信息丢失了。一个 lightweight 的 depth-aware encoder 可能更好。

8. **类比 VLA**：FoAR 没有 language conditioning。把它和 OpenVLA / RT-2 / $\pi_0$ 结合，做 "language-conditioned force-aware policy"，是一个很自然的下一步——你说 "wipe the board gently"，policy 应该输出 low force threshold。

---

## 8. 相关联想与延伸阅读

### 8.1 Force/Torque in Manipulation
- [ForceMimic (Liu 2024)](https://arxiv.org/abs/2410.07554) — force-centric imitation
- [TacDiffusion (Wu 2024)](https://arxiv.org/abs/2409.11047) — force-domain diffusion
- [Adaptive Compliance Policy (Hou 2024)](https://arxiv.org/abs/2410.09309) — diffusion + compliance
- [Admittance Visuomotor Policy (Zhou 2024)](https://arxiv.org/abs/2409.14440) — admittance control + visuomotor

### 8.2 Backbone policies
- [RISE (Wang 2024)](https://arxiv.org/abs/2404.12281) — FoAR 的 backbone
- [Diffusion Policy (Chi 2023)](https://diffusion-policy.cs.columbia.edu/) — action head
- [ACT (Zhao 2023)](https://tonyzhaozh.github.io/aloha/) — temporal ensemble
- [3D Diffusion Policy](https://3d-diffusion-policy.github.io/)
- [OpenVLA (Kim 2024)](https://arxiv.org/abs/2406.09246)
- [Octo](https://octo-models.github.io/)
- [$\pi_0$ (Physical Intelligence)](https://www.physicalintelligence.company/blog/pi0)

### 8.3 Tactile / Multimodal
- [ManiWAV (Liu 2024)](https://arxiv.org/abs/2410.13062) — audio-visual
- [3D-ViTac (Huang 2024)](https://arxiv.org/abs/2410.24091) — visuo-tactile
- [AnySkin (Bhirangi 2024)](https://arxiv.org/abs/2409.08276) — plug-and-play tactile
- [GelSight (Yuan 2017)](https://arxiv.org/abs/1706.04826)
- [ReSkin (Bhirangi 2021)](https://arxiv.org/abs/2107.08059)

### 8.4 Classical force control
- [Impedance Control (Hogan 1985)](https://asmedigitalcollection.asme.org/dsc/article-abstract/107/1/1/112098)
- [Hybrid Position/Force Control (Raibert & Craig 1981)](https://asmedigitalcollection.asme.org/dsc/article-abstract/103/2/126/112330)

### 8.5 Gating / Sparse MoE
- [Switch Transformer (Fedus 2021)](https://arxiv.org/abs/2101.03961)
- [Mixture of Experts survey](https://arxiv.org/abs/2202.07543)
- [Perceiver IO (Jaegle 2022)](https://arxiv.org/abs/2107.14795)

### 8.6 Reactive / Closed-loop Control in Learning
- [FORGE (Noseworthy 2024)](https://arxiv.org/abs/2408.04587) — force-guided exploration
- [MimicTouch (Yu 2024)](https://arxiv.org/abs/2406.14990)
- [Bi-ACT (Buamanee 2024)](https://arxiv.org/abs/2401.17698) — bilateral control + ACT

---

## 9. Final Intuition — 一句话总结

> **FoAR 把 "force/torque 是 sparse modality" 这个事实变成了 architecture design choice**：用一个 learned gate（future contact predictor）控制 force feature 的注入，配合 force-triggered reactive correction，让简单 position control 也能做 contact-rich 任务。这本质上是把工业界几十年累积的 "force control needs phase-awareness" 经验蒸馏进了 end-to-end learning framework。

Karpathy，如果让我用一个 analogy：FoAR 之于 force-aware manipulation，**类似于 MoE 之于 dense transformer**——把"什么时候用哪个 modality / expert"显式化为一个可学习的 gate，而不是让一个大 model 隐式学会。这个 insight 应该会 generalize 到更多 sparse-activated modality 的场景，比如 audio（speech-only phase）、tactile（contact-only phase）、甚至 language instruction（only relevant phase）。

特别 worth watch 的后续方向：
1. **Multi-phase gate**：φ 从 binary → multi-class phase（approach / align / contact / retreat），每个 phase 对应不同 modality 权重
2. **End-to-end learned reactive control**：把 Alg.1 的 threshold 也学进来
3. **Language-conditioned force**：natural language 指定 force profile（"gently" / "firmly"）
4. **Cross-embodiment generalization**：FoAR + Open X-Embodiment，让 force-aware policy 跨 robot transfer

Paper 的开源代码（如果 release）：建议先看 future contact predictor 的 label 生成脚本，那是整个 pipeline 的 labeling 关键。

最后，这个 paper 给我一个 meta 启发：**多模态融合的下一步不是更多 modality，而是更精细的 modality scheduling**——什么时候用、用多少、不用时填什么 default。这条路线还有非常多的 design space 没探索。
