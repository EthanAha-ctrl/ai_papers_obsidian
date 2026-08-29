---
source_pdf: FutureSightDrive Thinking Visually with Spatio-Temporal CoT for Autonomous
  Driving.pdf
paper_sha256: f8afdb8f41bff28c665830bfc7d6621334fbf192330daa87a984d8c8a32d17d1
processed_at: '2026-08-04T11:21:40-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---
直接让模型画一张未来的图出来当 CoT
图本身就是最 rich 的 representation，lane 在哪、车在哪、怎么动，一目了然。

Qwen2-VL 本来只会看图说话 (understanding)，不会画图 (generation)。怎么让它画?

经典做法 (Show-o, VILA-U, Janus) 是用 VQ-VAE 把 image tokenize 成离散 token，从头训一个 billion-scale 的 unified model。太贵。

FSDrive 的 trick: Qwen2-VL 内部已经有一个 ViT encoder 把 image 变成 continuous feature 喂给 LLM。LLM 的 vocabulary 本来只有 text token (比如 32K 个 BPE token)。现在直接把 MoVQGAN 的 visual codebook (~16K-65K 个 entry) **塞进 vocabulary 里**，text token 和 visual token 共用同一张 embedding table 和同一个 softmax。

这样 LLM 的 next-token prediction 公式:

$$\mathcal{L} = -\sum_{i=1} \log P_{\theta}(q_i | q_{<i})$$

- $q_i$: 第 $i$ 个 token，可能是 text token，也可能是 visual token
- $q_{<i}$: 之前所有 token (text + visual 混在一起)
- $\theta$: LLM 参数 (只 fine-tune LLM，ViT 和 VQ-VAE 都 frozen)

训练目标完全没变，还是标准的 next-token prediction。只是 vocabulary 变大了，LLM 既能吐文字也能吐 image token。image token 出来后，用 MoVQGAN 的 detokenizer 解码回 pixel。

数据成本: 只要 ~200K pairs (之前 Janus 要 100M，FSDrive 是它的 0.3%)。因为 LLM 的 world knowledge 已经 pre-train 好了，这里只是 **激活**，不是从头学。

### Step 2: Progressive 生成 - 先骨架后细节

直接让模型画 future frame 会翻车 - 车飞到天上、lane 突然消失，物理定律全乱。因为 autoregressive 从左上角逐 token 生成，没有全局物理约束。

FSDrive 的 curriculum:

```
先画 lane dividers (红色车道线)
    → 定义 "哪里能开" (static constraint)
        ↓
再画 3D detection boxes
    → 定义 "哪里有障碍物" (dynamic constraint)
        ↓
最后在 lane + box 约束下画完整 future frame
    → 补充 texture, lighting, fine details
```

公式:

$$P(Q_f | Q_l, Q_d) = \Pi_{t=1}^{h \cdot w} P_{\theta}(q_i | q_{<i}, Q_l, Q_d)$$

- $Q_f$: future frame 的 visual tokens (要生成的)
- $Q_l$: lane divider tokens (已经生成好的，作为 condition)
- $Q_d$: 3D box tokens (已经生成好的，作为 condition)
- $q_i$: future frame 第 $i$ 个 token
- $h \cdot w$: token grid 总数 ($h=H/p$, $w=W/p$, $p$ 是 VQ-VAE 的 stride)

这就像画画 - 先画线稿 (lane)，再画 silhouette (box)，最后上色 (full frame)。每一步都给下一步提供 strong prior。

### Step 3: 推理时合并成一张 unified frame

训练时分三步生成 $Q_l$, $Q_d$, $Q_f$。推理时为了速度，**合并成一张图**: future front-view image 上直接叠加红色 lane dividers + 3D boxes。这张图就是 spatio-temporal CoT $Q_{CoT}$。

- **Spatial 信息**: lane dividers (drivable area) + 3D boxes (obstacle location)
- **Temporal 信息**: future frame 内容本身 - 动态演化的 visual content 直接编码了时间推进

### Step 4: Inverse Dynamics 规划 trajectory

有了当前 observation $I_t$ 和想象的未来 $Q_{CoT}$，VLA 作为 **inverse dynamics model** 输出 trajectory:

$$P(W_t | I_t, Q_{CoT}, opt(T_{com}, T_{ego})) = \Pi_{i=1}^{n} P_{\theta}(w_i | w_{<i}, I_t, Q_{CoT}, opt(T_{com}, T_{ego}))$$

- $W_t = \{w_t^1, w_t^2, \ldots, w_t^n\}$: 未来 $n$ 个 waypoints (通常 6 个，对应 1s/2s/3s，间隔 0.5s)
- 每个 $w_t^i = (x_t^i, y_t^i)$: BEV 平面 2D 坐标
- $I_t$: 当前 6 个 surround-view 图像
- $Q_{CoT}$: 视觉 CoT (那张 future unified frame)
- $opt(T_{com}, T_{ego})$: navigation command (左转/直行/右转) + ego status (velocity, acceleration)

为什么叫 inverse dynamics?

- **Forward dynamics**: $s_{t+1} = f(s_t, a_t)$ - 给当前 state + action，预测 next state
- **Inverse dynamics**: $a_t = g(s_t, s_{t+1})$ - 给当前 state + 想要的 next state，反推 action

FSDrive 里:
- $s_t$ = 当前 observation $I_t$
- $s_{t+1}$ = 想象的未来 $Q_{CoT}$
- $a_t$ = trajectory $W_t$

模型看到"我现在在哪"和"我想到达哪个安全未来"，然后反推"该怎么开过去"。这比 forward planning 直接 regressive trajectory 容易得多 - 因为 future frame 是 high-dim supervision signal (128×192×3 ≈ 73728 个 pixel 值都来约束 trajectory)，比 waypoint 那 6 个 2D 坐标 (12 个 number) 信息量大几千倍。

---

## 架构图白话版

```
┌─────────────────────────────────────────────┐
│ 输入: 6 张 surround-view 图 + 文字指令        │
│         (e.g. "请在路口左转")                  │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  Qwen2-VL-2B (LLM)  │   ← 只 fine-tune LLM
        │  ViT encoder frozen │     ViT + MoVQGAN frozen
        │  LLM full fine-tune │
        └──────────┬──────────┘
                   │
         autoregressive
         next-token prediction
                   │
        ┌──────────▼──────────────────┐
        │ Token stream 输出:           │
        │                             │
        │ [VIS] #### #### #### ...    │  ← visual tokens
        │  (future unified frame,     │     画出来 = $Q_{CoT}$
        │   含 lane + box + RGB)      │
        │                             │
        │ [TXT] (x1,y1) (x2,y2) ...   │  ← text tokens
        │  (6 个 BEV waypoints)       │     = trajectory $W_t$
        └─────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │ MoVQGAN detokenizer  │  ← 只对 visual token
        │ → 还原成 128×192 图   │
        └─────────────────────┘
```

整个 model 就是一个 next-token predictor。它吐 visual token，detokenizer 还原成图; 它吐 text token (waypoint 坐标)，直接 parse 成 BEV trajectory。**一个 Transformer 统一所有任务**。

---

## 实验数据 - 哪些数字最 punch

### Trajectory Planning (nuScenes, no ego status)

最关键的对比:

| Method | L2 Avg ↓ | Collision Avg ↓ | Params |
|---|---|---|---|
| UniAD | 1.03 | 0.77 | - |
| Doe-1 | 0.70 | 0.21 | 7B (Lumina-mGPT) |
| OmniDrive | 0.84 | 0.94 | 7B (LLaVA) |
| **FSDrive** | **0.53** | **0.17** | **2B** (Qwen2-VL) |

2B 模型打赢 7B 模型。visual CoT 的信息密度确实比 textual CoT 或 VQ-only token 高得多。

### CoT 类型 ablation - 最 enlightening 的表

| CoT Type | L2 Avg ↓ | Collision Avg ↓ |
|---|---|---|
| None | 0.98 | 0.58 |
| Text CoT | 0.97 | 0.53 (-8.6%) |
| Image-text CoT | 0.98 | 0.50 (-13.8%) |
| **Spatio-temporal CoT** | **0.96** | **0.40 (-31%)** |

注意 L2 几乎没变 (~0.97)，但 collision 暴跌 31%。这说明:

**visual CoT 主要帮模型避免撞车，对 trajectory 几何精度帮助不大。**

直觉解释: L2 衡量的是 trajectory 和 ground truth 的几何距离。大部分时候直行跟车，trajectory 差不多就那样。collision 衡量的是有没有撞到别人 - 这需要模型 **理解未来 spatial relationship**。文字 "前方有车" 帮助有限，但看到 future frame 里那辆车 relative position + heading，模型就知道该让。

### Future Frame FID

| Method | Resolution | FID ↓ |
|---|---|---|
| GEM (diffusion) | 576×1024 | 10.5 |
| Doe-1 (autoregressive, Lumina-mGPT) | 384×672 | 15.9 |
| **FSDrive** (autoregressive, Qwen2-VL) | **128×192** | **10.1** |

FSDrive 用 **最低分辨率** 拿了 **最低 FID**。这有点反直觉 - 通常分辨率越低 FID 越好 (细节少，容易骗过 Inception-V3)。但 Doe-1 也是 autoregressive 且分辨率更高，FID 反而更差。

原因: FSDrive 继承了 Qwen2-VL 的 world knowledge (它在 internet 图像上 pre-train 过)，知道 "路长什么样、车长什么样"。Doe-1 用的 Lumina-mGPT 的 VQ token 丢失了 semantic，生成出来的图虽然在像素层面 OK，但语义上不连贯。

### NAVSIM closed-loop

PDMS 85.1, 超过 LAW (84.6) 和 DiffusionDrive-Cam (83.6)。NAVSIM 是 pseudo closed-loop - 会根据 model 的 trajectory 模拟其他 agent 反应。这比 nuScenes open-loop 更接近真实部署。FSDrive 能赢说明 visual CoT 在 closed-loop 下也 robust，不只是 open-loop overfitting。

---

## 几个 build intuition 的角度

### 1. Information Bottleneck

Textual CoT:

```
Image (~10^6 bit) → Text (~10^3 bit) → Trajectory (~10^2 bit)
         瓶颈             瓶颈
```

Visual CoT:

```
Image (~10^6 bit) → Future Image (~10^6 bit) → Trajectory (~10^2 bit)
         保持信息量         适度压缩
```

全程 visual modality，信息密度保持一致，没有 cross-modal conversion loss。

### 2. 人类 driver 的认知

你有没有注意到，老司机开车的时候嘴上不会念叨 "前方有车，距离 30 米，速度 15 m/s"。他们直接 **mental simulate** 3 秒后场景，然后本能调整方向盘和油门。

这就是 System 2 thinking 的真实形态 - 不是语言推理，是 **mental imagery + simulation**。FSDrive 把这个机制塞进了 VLA。

### 3. 为什么 Inverse Dynamics 比 Forward Planning 好

Forward planning 的困境:

```
当前 state → 搜索 action space (连续, infinite) → 评估每个 action 的 future
```

Inverse dynamics 的优势:

```
当前 state + 想要的 future state → 直接 regress action
```

目标 state (future frame) 已经定义了 "安全" 的 visual 条件，模型只需要找一条 trajectory 把现在带到那个未来。约束明确，搜索空间大幅缩小。

这跟 [Decision Transformer](https://arxiv.org/abs/2106.01345) 的哲学一致 - 把 RL 转成 conditional sequence modeling，给定 return，regress action。

### 4. World Model 的双重身份

FSDrive 让 VLA 一人分饰两角:

- **World Model**: $P(Q_{CoT} | I_t)$ - "我会想象未来"
- **Inverse Dynamics Model**: $P(W_t | I_t, Q_{CoT})$ - "我知道该怎么开到那个未来"

两个角色共享同一个 Transformer，共享同一套 weights，共享同一个 next-token prediction loss。训练时同时学，推理时串行执行。

这跟 [Ha & Schmidhuber 2018](https://arxiv.org/abs/1803.10122) 的 World Models 哲学一致，但 Ha 的实现是 VAE + RNN + controller 三段式，FSDrive 全塞进一个 LLM。

### 5. 和 LeCun JEPA 的分歧

LeCun 一直推 [JEPA](https://openreview.net/forum?id=JMF4iCHrJt) - 在 latent space 做 prediction，不生成 pixel，理由是 pixel-level 生成太 expensive 且容易 overfit detail。

FSDrive 走了相反路 - 坚持 pixel-level generation。理由很 practical: **pixel 是 VLA 和 perception 的 common language**。lane detection 输出是 pixel-overlay 形式，3D box 可以画在图上，future frame 本身就是 pixel。统一在 pixel modality 消除 cross-modal gap。

谁对? 取决于下游 task。如果只做 planning，latent CoT 可能够用 (像 [LAW](https://arxiv.org/abs/2412.10547))。如果还要做 scene understanding, VQA, 可解释性，pixel CoT 更 interpretable - 人可以直接看模型 "想了什么"。

### 6. 为什么 Progressive 生成能 enforce 物理定律

Autoregressive 从左上到右下逐 token 生成，每个 token 只 conditioned on 之前 token。到图像中间某处，模型可能已经忘了开头画的 lane 在哪，于是车画飞了。

Progressive 方法先把 lane 和 box 作为 explicit condition 放在 context 里:

```
[lane tokens] [box tokens] [future frame tokens]
     ↑              ↑              ↑
   static        dynamic      conditioned on both
   prior         prior
```

模型生成 future frame 每个位置时，attention 可以 attend 到 lane 和 box tokens，确保"这里画 road surface" "那里画 car"，物理一致。

这相当于 [ControlNet](https://arxiv.org/abs/2302.05543) 用 spatial condition 引导 diffusion 的思路，但用在 autoregressive generation 上。

---

## 几个有意思的细节

### 1. 数据效率为什么这么高

FSDrive Stage 1 pre-training 只用 ~200K pairs。对比:

| Method | Pre-training Data |
|---|---|
| Janus | ~100M |
| Show-o | ~similar scale |
| VILA-U | billion-scale |
| **FSDrive** | **~200K** |

200K vs 100M，差 500 倍。FSDrive 不需要从头学 visual generation 的所有 primitives (边缘、纹理、object shape)，这些 Qwen2-VL 已经在 internet image-text pairs 上学过了。FSDrive 只需要:

1. 学会 mapping: ViT feature → VQ token (codebook mapping)
2. 学会 driving domain 的 specific visual patterns (lane, car, road)
3. 学会 temporal evolution (future 和 current 的关系)

这三个任务 200K 数据足够。

### 2. 为什么 ViT frozen 但 LLM fine-tune

ViT 已经在 internet 图像上学会通用 visual feature，不需要再学。LLM 需要学会:
- 输出 visual token (新 vocabulary)
- 理解 driving-specific instruction
- 学会 spatio-temporal CoT 的 reasoning pattern

Fine-tune LLM 参数量 ~2B，训练成本相对低。8 张 A6000 训 12 epochs，个人 lab 也能跑。

### 3. 错误指令的鲁棒性 (Figure 3)

Paper 里最 striking 的 qualitative case:

给模型 **错误 navigation command** (告诉它直行，但实际该左转)。
- Baseline (无 CoT): 跟着错误指令走 → 撞车
- FSDrive: 通过 future frame prediction 看到前方 obstacle → 自动修正 trajectory → 安全通过

这说明 visual CoT 让模型 **不再盲信语言指令**，而是基于 observation 和 imagination 做决策。这很像 [CoT-VLA](https://arxiv.org/abs/2502.03491) 在 robotics 上发现的: visual CoT 让 VLA 具备 instruction correction 能力。

对 deployment 很重要 - GPS 导航偶尔会错，但 visual reasoning 能兜底。

### 4. FID 数字的小 caveat

FSDrive FID=10.1，GEM FID=10.5，差距很小。但要注意:

- GEM 用 576×1024 高分辨率，FSDrive 只 128×192
- FID 对低分辨率图通常更有利 (Inception-V3 在低 res 上 less discriminative)
- 所以 FID 数字不能直接说 "FSDrive 生成质量比 GEM 好"

更准确的解读: FSDrive 在 driving 场景下，**用极低 resolution autoregressive** 达到了和 diffusion 高分辨率 **comparable** 的质量。这对 real-time deployment 有意义 - 128×192 推理快得多。

### 5. Limitation 的真实影响

Paper 自陈只生成 front-view。问题:

- Intersection 场景下，side-approaching vehicle 可能不在 front-view 里
- Future frame 看不到右侧来车，CoT 信息不完整
- 实际上 nuScenes 的 collision cases 很多发生在 intersection

改进方向 - multi-view future generation:

$$Q_{CoT} = \{Q_{CoT}^{front}, Q_{CoT}^{back}, Q_{CoT}^{left}, Q_{CoT}^{right}, Q_{CoT}^{front-left}, Q_{CoT}^{front-right}\}$$

但这会让 token 数量 × 6，autoregressive generation 时间也 × 6，real-time 部署挑战大。

---

## 一句话总结

FSDrive 把 VLA 的 CoT 从 "先说话再开车" 改成 "先想象未来再开车"。用 2B 参数 + 200K 数据 + progressive 生成物理约束 + inverse dynamics 规划，在 nuScenes 和 NAVSIM 上打赢 7B 的 textual CoT 方法。collision rate 降 31% - visual 想象确实帮模型更早识别碰撞风险。

---

**参考链接**:

- FSDrive GitHub: [https://github.com/MIV-XJTU/FSDrive](https://github.com/MIV-XJTU/FSDrive)
- Qwen2-VL: [https://arxiv.org/abs/2409.12191](https://arxiv.org/abs/2409.12191)
- Janus (unified MLLM baseline): [https://arxiv.org/abs/2410.13848](https://arxiv.org/abs/2410.13848)
- EMMA (text-only VLA): [https://arxiv.org/abs/2410.23262](https://arxiv.org/abs/2410.23262)
- OmniDrive (textual CoT): [https://arxiv.org/abs/2411.15337](https://arxiv.org/abs/2411.15337)
- Doe-1 (VQ-based generation): [https://arxiv.org/abs/2412.09627](https://arxiv.org/abs/2412.09627)
- Decision Transformer: [https://arxiv.org/abs/2106.01345](https://arxiv.org/abs/2106.01345)
- Ha & Schmidhuber World Models: [https://arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)
- JEPA (LeCun latent prediction): [https://openreview.net/forum?id=JMF4iCHrJt](https://openreview.net/forum?id=JMF4iCHrJt)
- LAW (latent world model): [https://arxiv.org/abs/2412.10547](https://arxiv.org/abs/2412.10547)
- DreamerV3: [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)
- MoVQGAN: [https://arxiv.org/abs/2209.09002](https://arxiv.org/abs/2209.09002)
- CoT-VLA (visual CoT for robotics): [https://arxiv.org/abs/2502.03491](https://arxiv.org/abs/2502.03491)
- VLIPP (physical prior for video gen): [https://arxiv.org/abs/2310.08785](https://arxiv.org/abs/2310.08785)
- LlamaGen (autoregressive image gen): [https://arxiv.org/abs/2406.06525](https://arxiv.org/abs/2406.06525)
- BEV-Planner (ego status critique): [https://arxiv.org/abs/2406.08456](https://arxiv.org/abs/2406.08456)
- GAIA-1 (Wayve world model): [https://arxiv.org/abs/2309.17080](https://arxiv.org/abs/2309.17080)
- Chain-of-Thought (Wei et al.): [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)
- nuScenes: [https://www.nuscenes.org/](https://www.nuscenes.org/)
- NAVSIM: [https://github.com/autonomousvision/navsim](https://github.com/autonomousvision/navsim)
- DriveLM: [https://github.com/OpenDriveLab/DriveLM](https://github.com/OpenDriveLab/DriveLM)

---

# FSDrive: Spatio-Temporal CoT 让 VLA 进行"视觉思维"

## 1. 核心动机 - 从符号思维到视觉想象

这篇 paper 解决的是 VLA (Vision-Language-Action) 模型在 autonomous driving 中 reasoning 的根本性问题。当前主流做法 (EMMA, OmniDrive, DriveVLM) 基本上沿用 textual CoT - 让模型先用语言描述场景 ("前方有红色车辆在 30 米处", "右侧车道线弯曲"), 再基于这些文字 token 做 planning。Karpathy 你应该能立刻感受到这里的问题: **visual 信息经过 language 符号压缩后, 丢失了大量 spatio-temporal 关系**。一个 3D box 的 (x, y, z, l, w, h, yaw) 七元组写成文字"前方 5 米处有一辆车", 这种 lossy compression 丢掉了 fine-grained geometric cues, 也丢掉了 temporal dynamics 的连续演化。

FSDrive 的 insight 借鉴人类 driver 的认知机制 - 司机在脑海里构建的是 **visual imagery of future scene**, 而不是 verbal description。所以 CoT 的中间表示应该直接是 image, 把 spatial structure (lane dividers + 3D boxes) 和 temporal evolution (future frame 内容) 编码进单一 unified frame。

这里可以联想 LeCun 一直在推的 [JEPA (Joint Embedding Predictive Architecture)](https://openreview.net/forum?id=JMF4iCHrJt) - 在 latent space 做 prediction, 避免 pixel-level 生成的开销, 同时保留抽象 reasoning。但 FSDrive 走了相反方向 - 它 **坚持在 pixel space 做 prediction**, 理由是 pixel-level 的 visual CoT 可以和 perception output 统一在同一 modality, 消除 cross-modal gap。这个 trade-off 很有意思。

## 2. 方法论详解

### 2.1 Preliminary - End-to-end trajectory planning

输入: $N$ 个 surround-view images $I_t = \{I_t^1, I_t^2, \ldots, I_t^N\}$ at timestep $t$

- 上标 $1, 2, \ldots, N$ 表示第几个 camera (nuScenes 有 6 个 camera, 通常 $N=6$)
- 下标 $t$ 表示当前时刻

输出: BEV trajectory $W_t = \{w_t^1, w_t^2, \ldots, w_t^{\bar{n}}\}$

- 每个 waypoint $\vec{w_t^i} = (x_t^i, \bar{y_t^i})$ - 注意这里有个 typo, 应该是 $(x_t^i, y_t^i)$, 表示在 BEV 平面上的 2D 坐标
- $\bar{n}$ 表示预测未来多少个 waypoints (通常对应未来 1s, 2s, 3s, 间隔 0.5s, 所以 $\bar{n}=6$)
- $opt(T_{com}, T_{ego})$ 是 optional 的 navigation command (左转/右转/直行) 和 ego status (velocity, acceleration)

planning 公式 (Eq.1):

$$W_t = \mathcal{M}(I_t, opt(T_{com}, T_{ego}))$$

这里 $\mathcal{M}$ 就是整个 VLA model。

### 2.2 Unified visual generation and understanding - 核心创新

传统做法 ([Show-o](https://arxiv.org/abs/2408.12528), [VILA-U](https://arxiv.org/abs/2409.04429)) 用 VQ-VAE 把 image tokenize 成离散 token, 但这些 token 缺乏 semantic 信息, 伤害 downstream understanding 任务。而且要从头在 billion-scale 数据上 train, 成本巨大。

FSDrive 的巧妙之处: **直接复用现有 MLLM (Qwen2-VL-2B)**, 把 VQ-VAE 的 image codebook 加到 LLM 的 text vocabulary 中, 让 text 和 visual token 共享同一 vocabulary space。

image tokenize 过程:

$$x \in \mathbb{R}^{H \times W \times 3} \xrightarrow{\text{VQ-VAE}} q \in \mathcal{Q}^{h \times w}$$

- $H, W$: 原始图像高宽
- $h = H/p, w = W/p$: token grid 大小
- $p$: downsampling factor (stride)
- $\mathcal{Q}$: codebook (通常是 16384 或 65536 个 entry)
- $q(i, j)$: 位置 $(i, j)$ 处对应的 codebook index

训练 image generation 用 next-token prediction (Eq.2):

$$\mathcal{L} = -\sum_{i=1} \log P_{\theta}(q_i | q_{<i})$$

- $q_i$: 第 $i$ 个 visual token (按 raster order 排列)
- $q_{<i}$: 之前所有 token (包括 text token 和 visual token)
- $\theta$: LLM 参数

这其实是把 image generation 当成 language modeling 的扩展 - 同一个 Transformer, 同一个 loss, 同一个 softmax over vocabulary。这让我想起 [LlamaGen](https://arxiv.org/abs/2406.06525) 的 "Autoregressive model beats diffusion" 论断。

### 2.3 Progressive generation - 物理约束的 curriculum

直接生成完整 future frame 容易违反物理定律 (车飞起来、lane 突然消失)。FSDrive 用 easy-to-hard 的两阶段:

**Stage 1**: 先生成 lane dividers $Q_l$ (静态物理约束 - drivable area)
**Stage 2**: 再生成 3D bounding boxes $Q_d$ (动态物理约束 - 关键 object 的运动模式)
**Stage 3**: 最后在 $Q_l, Q_d$ 条件下生成 full future frame $Q_f$

公式 (Eq.5):

$$P(Q_f | Q_l, Q_d) = \Pi_{t=1}^{h \cdot w} P_{\theta}(q_i | q_{<i}, Q_l, Q_d)$$

- $Q_f$: future frame 的 visual tokens
- $Q_l$: lane divider 的 visual tokens (画在一张图上, 红色 lane 线)
- $Q_d$: 3D detection 的 visual tokens (画在图上的 3D box)
- $q_i$: future frame 的第 $i$ 个 token

这个 curriculum 的 intuition 是: lane 定义了静态 spatial layout (哪里能开), 3D box 定义了 dynamic obstacle (哪里不能开), 两者结合后, future frame 的生成被 strong constrained 在物理合理区域内。这类似于 [VLIPP](https://arxiv.org/abs/2310.08785) 用 physical prior 引导 video generation 的思路。

### 2.4 Spatio-temporal CoT - 推理时的 unified frame

推理阶段, 为了 efficiency, 不再分开生成 $Q_l, Q_d, Q_f$, 而是 **整合成 single unified frame** - 一张 future front-view image, 上面叠加红色 lane dividers 和 3D boxes。

这个 unified frame 就是 spatio-temporal CoT $Q_{CoT}$, 然后 VLA 作为 **inverse dynamics model** 输出 trajectory:

公式 (Eq.6):

$$P(W_t | I_t, Q_{CoT}, opt(T_{com}, T_{ego})) = \Pi_{i=1}^{n} P_{\theta}(w_i | w_{<i}, I_t, Q_{CoT}, opt(T_{com}, T_{ego}))$$

- $W_t$: 输出 trajectory
- $I_t$: 当前 observation (6 cameras)
- $Q_{CoT}$: 视觉思维链 (predicted unified future frame)
- $w_i$: trajectory 的第 $i$ 个 waypoint
- $w_{<i}$: 之前 waypoints (autoregressive)

**关键 insight: inverse dynamics model**

这里为什么叫 "inverse dynamics"? 在 RL / world model 文献里, forward dynamics 是 $s_{t+1} = f(s_t, a_t)$ - 给 state 和 action 预测 next state。Inverse dynamics 则是 $a_t = g(s_t, s_{t+1})$ - 给当前 state 和 target state, 反推 action。

FSDrive 的设定正是 inverse: VLA 看到 $I_t$ (当前) 和 $Q_{CoT}$ (想象的未来), 然后输出 trajectory (action sequence) - 这个 trajectory 正是把当前状态"驱动"到想象状态的 action。这种 formulation 比 forward model 直接 regressive trajectory 更 sample-efficient, 因为 future frame 是 high-dimensional 但 information-rich 的 supervision signal, 比 waypoint 这种低维信号蕴含更多约束。

这与 [DreamerV3](https://arxiv.org/abs/2301.04104) 的 actor-critic in imagination 有哲学相似性, 但 Dreamer 在 latent space 做 rollout, FSDrive 在 pixel space 直接 imagine 一帧。

## 3. 架构解析

```
Input:
  - 6 surround-view images (current, $I_t$)
  - Task instruction (e.g., "turn left at intersection")
  
MLLM (Qwen2-VL-2B, frozen ViT encoder, fine-tuned LLM):
  ├── ViT encodes 6 images → continuous features
  ├── Text tokens + visual features → unified sequence
  └── LLM autoregressively predicts:
      ├── [Stage 1 output] Spatio-temporal CoT tokens ($Q_{CoT}$)
      │     = unified future front-view frame
      │       containing:
      │         - future RGB pixels
      │         - red lane dividers
      │         - 3D detection boxes
      └── [Stage 2 output] Trajectory waypoints ($W_t$)

Decoding:
  - $Q_{CoT}$ tokens → MoVQGAN detokenizer → future image
  - $W_t$ tokens → text parsing → BEV coordinates
```

值得一提的几个细节:

1. **Encoder frozen, LLM fine-tuned**: ViT 和 VQ-VAE 都 frozen, 只 fine-tune LLM, 大幅降低 training cost
2. **MoVQGAN 作为 detokenizer**: MoVQGAN 是 VQGAN 的改进版, 引入 spatial conditional modulation, 生成质量比原版 VQ-VAE 更高 ([MoVQ paper](https://arxiv.org/abs/2209.09002))
3. **数据效率**: 只用了 ~200K image-instruction pairs, 相比 [Janus](https://arxiv.org/abs/2410.13848) 等方法的 100M scale, 只有 0.3%

## 4. 实验数据深度解析

### 4.1 Trajectory Planning on nuScenes (Table 1)

关键对比 (ST-P3 metrics, no ego status):

| Method | L2 Avg (m) ↓ | Collision Avg (%) ↓ | Backbone |
|---|---|---|---|
| UniAD | 1.03 | 0.77 | - |
| Doe-1 | 0.70 | 0.21 | Lumina-mGPT-7B |
| OmniDrive | 0.84 | 0.94 | LLaVA-7B |
| **FSDrive** | **0.53** | **0.17** | Qwen2-VL-2B |
| **FSDrive*** | **0.28** | **0.10** | Qwen2-VL-2B |

注意: FSDrive 用 **2B 参数** 的 Qwen2-VL, 比 Doe-1 (7B) 和 OmniDrive (7B) 都小, 但 L2 和 collision 都更优。这验证了 visual CoT 比纯 textual 或 VQ-based 方法更高效。

带 ego status (*版) 的 FSDrive 在 ST-P3 metrics 下达到 L2=0.28, collision=0.10 - 接近 [EMMA](https://arxiv.org/abs/2410.23262) (Gemini 1.6, L2=0.32, 但 EMMA 是 Google 内部 Gemini, 别人无法复现)。这印证了 [BEV-Planner](https://arxiv.org/abs/2406.08456) 揭示的 ego status 主导 open-loop metric 的问题。

### 4.2 NAVSIM closed-loop (Table 2)

| Method | PDMS ↑ |
|---|---|
| UniAD | 83.4 |
| DiffusionDrive-Cam | 83.6 |
| LAW | 84.6 |
| **FSDrive** | **85.1** |

PDMS (Predictive Driver Model Score) 是 NAVSIM 的综合指标, 包含 NC (no collision), DAC (drivable area compliance), TTC (time to collision), Comfort, EP (effective progress)。FSDrive 在 camera-only 设定下达到 85.1, 超越 [LAW](https://arxiv.org/abs/2412.10547) 的 latent world model 方法, 说明 visual CoT 在 pseudo closed-loop 下也 robust。

### 4.3 Future Frame Generation FID (Table 3)

| Method | Type | FID ↓ |
|---|---|---|
| DriveDreamer (diffusion) | 192×384 | 15.8 |
| GenAD (diffusion) | 256×448 | 15.4 |
| GEM (diffusion) | 576×1024 | 10.5 |
| Doe-1 (autoregressive) | 384×672 | 15.9 |
| **FSDrive** (autoregressive) | **128×192** | **10.1** |

最 remarkable 的数字: FSDrive 用 **128×192 低分辨率 autoregressive** 模型, FID=10.1, 超越了用 576×1024 高分辨率的 GEM diffusion 方法。这暗示 visual generation 的质量瓶颈不在分辨率, 而在 semantic understanding - 因为 FSDrive 继承了 MLLM 的 world knowledge, 而 Doe-1 用 Lumina-mGPT 的 VQ token 丢失了 semantic 信息。

### 4.4 DriveLM GVQA (Table 4)

Final Score: FSDrive 0.57 vs OmniDrive 0.56 vs SimpleLLM4AD 0.53。说明 unified pre-training 同时提升了 understanding 和 generation, 没有发生 catastrophic forgetting。

### 4.5 Ablation 关键发现

**Pre-training ablation (Table 5)**:
- Pure VQA: L2=1.19, 几乎无提升
- +Future frame generation: L2=1.02 (↓16.4%), collision=0.60 (↓15.8%)
- +3D detection + lane divider: L2=0.98, collision=0.58
- Full combination: 最佳

**CoT 类型 ablation (Table 6)** - 这是最关键的:
- None: collision avg = 0.58
- Text CoT: 0.53 (↓8.6%)
- Image-text CoT: 0.50 (再降, 但提升有限 - modality gap 问题)
- **Spatio-temporal CoT: 0.40 (↓31%)**

L2 在不同 CoT 下变化不大 (~0.96-0.98), 但 collision 大幅下降 - 这说明 visual CoT 主要帮助 model 理解 **未来 collision risk**, 而不是改进 trajectory 的几何精度。这非常符合直觉 - 文字描述 "前方有车" 不够, model 需要看到 future frame 里那辆车 relative position 才能避免 collision。

**Progressive generation ablation (Table 7)**:
- No pre-training: FID=29.4
- 100K data: 16.2
- 200K data: 12.7
- 200K + progressive: **10.1**

数据从 0 → 200K 让 FID 从 29.4 → 12.7, progressive method 再降到 10.1 - 物理 prior 的作用相当于 data augmentation 的效果。

### 4.6 Qualitative (Figure 3)

最有说服力的 case: 错误的 navigation instruction (e.g., 模型被告诉"直行"但实际应该"左转")。没有 visual CoT 时, 模型跟着错误 instruction 走, 撞车。FSDrive 即使 instruction 错, 也能通过 future frame prediction 看到前方 obstacle, 自主修正 trajectory - 这是 inverse dynamics model 的精髓, 类似于 [CoT-VLA](https://arxiv.org/abs/2502.03491) 提出的 visual CoT for robotics。

## 5. Intuition Building - 为什么 Visual CoT 有效

让我从几个角度构建 intuition:

### 5.1 Information Bottleneck 视角

Textual CoT 经过 language 这个 bottleneck:

$$\text{Visual}(H \times W \times 3) \xrightarrow{\text{lossy}} \text{Text}(L \text{ tokens}) \xrightarrow{\text{planning}} \text{Trajectory}$$

每个文字 token 大约对应几百 bit 信息, 而每张图 ~10^6 bit。压缩比 ~1000x, 必然丢失 spatio-temporal 关系。

Visual CoT 消除 bottleneck:

$$\text{Visual}(I_t) \xrightarrow{\text{imagine}} \text{Visual}(Q_{CoT}) \xrightarrow{\text{inverse dynamics}} \text{Trajectory}$$

全程同 modality, 信息密度保持。

### 5.2 Causal 视角

Driving 本质是 causal - 当前 action 导致未来 state 变化。Forward model 学 $P(\text{future} | \text{current}, \text{action})$, inverse model 学 $P(\text{action} | \text{current}, \text{future})$。

Inverse model 的优势: 给定 target state (想象中的 safe future), 推 action 比较容易, 因为约束明确。Forward model 需要在 exponential action space 中搜索, 而 inverse 直接 regress。

这与 [Decision Transformer](https://arxiv.org/abs/2106.01345) 的哲学一致 - 把 RL 转成 conditional sequence modeling。

### 5.3 System 1 vs System 2 思维

Kahneman 的双系统理论在 driving 上特别适用:
- System 1: 快速直觉反应 (跟车、保持车道)
- System 2: 慢思考, 想象未来场景, 评估风险

Textual CoT 是 System 2 的语言化版本。Visual CoT 更接近人类 System 2 的实际运作 - mental simulation。这也呼应了 [HUME](https://arxiv.org/abs/2505.21432) 把 System 2 thinking 引入 VLA 的方向。

### 5.4 World Model 的双重角色

FSDrive 让 VLA 同时扮演:
1. **World model**: $P(Q_{CoT} | I_t)$ - 想象 future
2. **Inverse dynamics model**: $P(W_t | I_t, Q_{CoT})$ - 反推 action

这与 [Ha & Schmidhuber 的 World Models](https://arxiv.org/abs/1803.10122) 类似, 但 Ha 的做法是 VAE + RNN + controller 三段式, FSDrive 全部统一在 single Transformer 中, 用 next-token prediction 统一所有任务。

## 6. 与 Related Work 的精细对比

### 6.1 vs Doe-1 ([arxiv 2412.09627](https://arxiv.org/abs/2412.09627))

Doe-1 也基于 Lumina-mGPT 做视觉生成, 但用 VQ-VAE 的 discrete token, 丢失 semantic 信息。FSDrive 直接复用 MLLM 已有的 continuous ViT feature + 扩展 vocabulary, 保留 semantic understanding。Table 1 显示 FSDrive 在 L2 上比 Doe-1 好 24% (0.53 vs 0.70), 在 collision 上好 19% (0.17 vs 0.21)。

### 6.2 vs EMMA ([arxiv 2410.23262](https://arxiv.org/abs/2410.23262))

EMMA 用 Gemini 把所有 input/output 都转成 text - 包括 trajectory waypoints 也写成文字 "waypoint 1: (x, y), waypoint 2: (x, y)"。FSDrive 保留 trajectory 输出但用 visual CoT 替代 text reasoning。EMMA 依赖 Google 内部 Gemini, 复现困难; FSDrive 用开源 Qwen2-VL-2B, 可复现。

### 6.3 vs OmniDrive ([CVPR 2025](https://arxiv.org/abs/2411.15337))

OmniDrive 也是 LLaVA-based, 用 textual CoT + 3D perception。FSDrive 在相同 backbone (LLaVA-7B) 下也优于 OmniDrive, 证明 visual CoT 比 text CoT 更有效。

### 6.4 vs GAIA-1 ([Wayve](https://arxiv.org/abs/2309.17080))

GAIA-1 是早期 driving world model, 用 next-token predictor + diffusion decoder。FSDrive 全 autoregressive, 没有 diffusion component, 推理更快。

### 6.5 vs DrivingGPT ([ICCV 2025](https://arxiv.org/abs/2505.15745))

DrivingGPT 用 LlamaGen 的 visual tokenizer, 同时输出 future state 和 action。但 LlamaGen 的 VQ token 同样缺 semantic, 而 FSDrive 继承 MLLM 的 semantic understanding 能力。

## 7. 局限性与可改进方向

Paper 自陈局限:
1. 只生成 front-view future frame, 缺 surround awareness - 在复杂 intersection 容易漏 side-approaching vehicle
2. FID=10.1 虽 competitive, 但 resolution 只有 128×192, 远低于 production 需求
3. Real-time efficiency 没有详细 benchmark - autoregressive generation 一帧可能需要几百 ms

可以想到的改进:

**Multi-view generation**: 把 $Q_{CoT}$ 扩展成 6 个 surround future frames, 用 BEV-aligned representation (类似 [BEVFormer](https://arxiv.org/abs/2203.17270)) 做 spatial alignment。

**Latent CoT**: 借鉴 [LAW](https://arxiv.org/abs/2412.10547) 的 latent world model, 在 latent space 做 CoT, 避免像素生成的计算开销, 同时保留 visual 的 high-dim 信息。

**Hierarchical CoT**: 短期 (1s) 用 pixel-level visual CoT, 长期 (3s+) 用更抽象的 latent CoT - 类似 [DreamerV3](https://arxiv.org/abs/2301.04104) 的 imagination horizon 设计。

**Closed-loop training**: 当前是 open-loop (next-token prediction on recorded data)。要真正部署, 需要 [CARLA](https://carla.org/) 或 NAISYS 这样的 closed-loop simulator, 用 visual CoT 在 sim 里 rollout。

**Diffusion hybrid**: 用 diffusion 做 final frame decoding (像 GAIA-1), autoregressive 做语义 planning, 兼顾质量与速度。

## 8. 对 VLA 范式的更广启示

FSDrive 对 [VLA](https://arxiv.org/abs/2406.05404) 范式 (robotics + driving) 有普适启示:

1. **CoT modality 应该匹配 task modality**: 语言任务用 text CoT, 视觉/物理任务用 visual CoT
2. **World model + Inverse dynamics** 是强大的 dual formulation - 比纯 forward planning 更 sample efficient
3. **Unified vocabulary** 是低成本激活 generation 能力的关键 - 不需要从头 train, 复用 MLLM 已有 world knowledge
4. **Progressive curriculum** with physical priors 是让 generation 遵守物理的有效手段

联想到 robotics 领域的 [π0](https://arxiv.org/abs/2410.24164) 和 [OpenVLA](https://arxiv.org/abs/2406.09246) - 它们也在探索 visual CoT, 但通常用 image generation 作为 auxiliary, 不像 FSDrive 把它放在 planning 的核心位置。FSDrive 的实验数据 (collision ↓31%) 强烈支持 visual CoT 是 VLA 的关键 missing piece。

## 9. 总结

FSDrive 的核心贡献可以浓缩为:

> **把 CoT 从 language 提升到 pixel, 让 VLA 在 driving 上 "think visually" - 想象未来场景, 反推当前 action, 用 unified pre-training 激活 generation, 用 progressive curriculum 注入物理 prior。**

工程上, 它用 2B 模型 + 200K 数据达到了 SOTA, 证明了 **visual reasoning 不需要 billion-scale from-scratch training**, 关键是正确的 formulation。

哲学上, 它呼应了 Yann LeCun 一直强调的 "machine needs world model for reasoning" - 但走了一条与 LeCun 的 JEPA 相反的路: 在 pixel space 直接 imagine, 而非在 latent space 做 abstract prediction。孰优孰劣, 还需要更多 closed-loop benchmark 验证。

---

**参考链接**:

- Paper GitHub: [https://github.com/MIV-XJTU/FSDrive](https://github.com/MIV-XJTU/FSDrive)
- nuScenes: [https://www.nuscenes.org/](https://www.nuscenes.org/)
- NAVSIM: [https://github.com/autonomousvision/navsim](https://github.com/autonomousvision/navsim)
- DriveLM: [https://github.com/OpenDriveLab/DriveLM](https://github.com/OpenDriveLab/DriveLM)
- Qwen2-VL: [https://arxiv.org/abs/2409.12191](https://arxiv.org/abs/2409.12191)
- Janus (unified MLLM): [https://arxiv.org/abs/2410.13848](https://arxiv.org/abs/2410.13848)
- JEPA (LeCun): [https://openreview.net/forum?id=JMF4iCHrJt](https://openreview.net/forum?id=JMF4iCHrJt)
- Decision Transformer: [https://arxiv.org/abs/2106.01345](https://arxiv.org/abs/2106.01345)
- DreamerV3: [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)
- Ha & Schmidhuber World Models: [https://arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)
- EMMA: [https://arxiv.org/abs/2410.23262](https://arxiv.org/abs/2410.23262)
- Doe-1: [https://arxiv.org/abs/2412.09627](https://arxiv.org/abs/2412.09627)
- OmniDrive: [https://arxiv.org/abs/2411.15337](https://arxiv.org/abs/2411.15337)
- GAIA-1: [https://arxiv.org/abs/2309.17080](https://arxiv.org/abs/2309.17080)
- VLIPP: [https://arxiv.org/abs/2310.08785](https://arxiv.org/abs/2310.08785)
- MoVQGAN: [https://arxiv.org/abs/2209.09002](https://arxiv.org/abs/2209.09002)
- CoT-VLA: [https://arxiv.org/abs/2502.03491](https://arxiv.org/abs/2502.03491)
- Chain-of-Thought original: [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)
- BEV-Planner: [https://arxiv.org/abs/2406.08456](https://arxiv.org/abs/2406.08456)
- LAW: [https://arxiv.org/abs/2412.10547](https://arxiv.org/abs/2412.10547)
- LlamaGen: [https://arxiv.org/abs/2406.06525](https://arxiv.org/abs/2406.06525)
