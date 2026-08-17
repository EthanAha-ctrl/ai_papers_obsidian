---
source_pdf: DO WORLD ACTION MODELS GENERALIZE BETTER THAN.pdf
paper_sha256: e94e3903c27d985587460acd39cfae58f9c237416284fba2f5d7882e00f22bcd
processed_at: '2026-08-03T22:58:56-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 1. 这篇 paper 在搞什么

就是一场擂台赛。

左边是 **VLA**，右边是 **WAM**。两边都想当 robot 的 policy backbone，但出身完全不一样。

**VLA** 这边的逻辑是：我拿一个 vision-language model（比如 PaliGemma, Qwen-VL），它在互联网上看了几十亿张 image-text pair，学会了 "这是红色方块"、"杯子在碗左边" 这种 semantic understanding。然后我给它接一个 action head，用 robot demonstration 数据 finetune 一下，让它学会从 image+language 输出 robot action。代表选手：$\pi_0$, $\pi_{0.5}$, OpenVLA, X-VLA。

**WAM** 这边的逻辑是：我拿一个 video diffusion model（比如 Cosmos-Predict2, Wan2.1/2.2, LTX-Video），它在互联网上看了几千万段 video，学会了 "锤子敲下去方块会飞"、"手抓杯子杯子会跟着动" 这种 **physical dynamics**。然后我做点小改动让它输出 robot action。代表选手：Cosmos-Policy, LingBot-VA, GE-Act, DreamZero。

两边都在说自己的方法 generalize 更好。VLA 说 "我有 semantic understanding，能听懂指令"；WAM 说 "我有 dynamics prior，能预测未来会怎样"。paper 作者就想：**别吵了，上擂台打一场，用数据说话。**

## 2. 两边选手的 "基因" 差在哪

这是理解后面所有结果的 key。

### VLA 的训练目标

VLA backbone 学的是：

$$\mathcal{L}_{VLM} = -\sum_t \log P(x_t \mid x_{<t}, I, L)$$

- $x_t$: 第 $t$ 个 text token
- $I$: input image
- $L$: language instruction
- $P$: VLM 预测下一个 token 的概率

这个训练 **完全静态**。image 是单张的，没有时间维度。模型学到的是 "image 里有啥 + 文字描述啥 → 接下来文字说啥"。它不知道 "如果我手臂往左推，方块会往右滚" 这种 causal physics。

### WAM 的训练目标

Video diffusion model 学的是：

$$\mathcal{L}_{video} = \mathbb{E}_{\epsilon, x_0}\left[\|\epsilon - \epsilon_\phi(x_t, t, \text{cond})\|^2\right]$$

- $x_0$: clean future frame
- $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$: 加了噪声的 frame
- $\epsilon \sim \mathcal{N}(0,I)$: Gaussian noise
- $\bar{\alpha}_t$: noise schedule 累积系数
- $\epsilon_\phi$: 要学的 noise prediction network
- cond: 通常是过去几帧 + text

这个训练 **本质是 temporal 的**。模型必须理解 "frame A 之后 frame B 会怎样"——也就是 dynamics。而且 video 涵盖了 hand manipulation, object interaction, driving, natural scene motion 等，physics prior 非常 rich。

### 一句话总结基因差异

**VLA backbone 是 "看图说话的文科生"，WAM backbone 是 "看 video 预测未来的理科生"。**

前者懂语义但不懂物理，后者懂物理但语义理解弱。Paper 想验证的就是：**做 robot policy 时，物理 prior 比 semantic prior 更值钱吗？尤其在面对各种干扰时。**

## 3. 擂台规则怎么设的

### 两个互补的 benchmark

作者用了两个 benchmark，互补性设计得很好：

**LIBERO-Plus**：
- Franka Panda 单臂 7-DoF
- MuJoCo simulator
- 256×256 图像
- 7-dim delta end-effector action
- 10Hz 控制
- 40 个 base task

**RoboTwin 2.0-Plus**（这篇 paper 自己搭的）：
- Aloha-AgileX 双臂 14-DoF
- SAPIEN simulator
- 320×240 图像，3 个 camera
- 14-dim joint position action
- 25-30Hz 控制
- 50 个 bimanual 协作 task

一个测单臂精细操作，一个测双臂协调。embodiment, action space, control frequency 全不一样。如果两边结论一致，说明结论是 robust 的，不是某个 benchmark 的 artifact。

### 七种 perturbation - 这才是精华

作者没有简单 "加 random noise 测一下"，而是把 robustness 拆成 7 个维度，每个维度单独测试。这样能看出 **每种方法到底怕什么**。

| 维度 | 代号 | 测的是啥 |
|------|------|---------|
| Camera | C1-C3 | 第三视角 camera 位置和角度变化 |
| Robot Init | R | 机器人初始关节角度扰动 ±0.225 rad |
| Language | R1-R3 | 指令重写（distract / reword / reasoning chain） |
| Light | L1-L4 | 光照强度、方向、阴影、specular 全变 |
| Background | B1-B2 | 桌面材质、墙面地板颜色全换 |
| Noise | N1-N5 | 5 种 image corruption（motion blur, Gaussian blur, zoom blur, fog, glass blur） |
| Layout | O1-O2 | 加 3-15 个 task-irrelevant distractor，目标物体位置扰动 |

每种 perturbation 单独激活，其他保持 clean。这样能 isolate 每个维度的影响。

这个设计很 important，因为 **不同 perturbation 测的是不同的能力**：
- Camera, Robot Init 测 **geometric invariance**
- Light, Noise, Background 测 **visual appearance robustness**
- Layout 测 **attention focus**（能不能不被 distractor 带跑）
- Language 测 **semantic understanding**

## 4. 擂台结果 - 关键数据

### RoboTwin 2.0-Plus（双臂）

| Model | Clean | Camera | Robot | Lang | Light | BG | Noise | Layout | **Total** |
|-------|-------|--------|-------|------|-------|-----|--------|--------|---------|
| $\pi_{0.5}$ | 78.4 | 45.6 | 27.6 | 74.4 | 49.6 | 71.7 | 64.9 | 56.8 | 58.6 |
| X-VLA | 65.6 | 23.2 | 65.2 | 64.4 | 63.1 | 58.6 | 49.7 | 34.8 | 53.1 |
| MOTUS | 87.0 | 21.6 | 85.0 | 83.2 | 84.6 | 84.4 | 43.1 | 82.8 | 71.5 |
| **LingBot-VA** | **92.1** | 28.9 | 36.2 | 87.3 | **89.0** | **91.3** | **80.9** | **87.9** | **74.2** |

**LingBot-VA（WAM）总成绩 74.2% 完胜**，比第二名 MOTUS 71.5% 高，比 $\pi_{0.5}$ 58.6% 高 15.6 个百分点。

### LIBERO-Plus（单臂）

| Model | Clean | Total |
|-------|-------|-------|
| $\pi_0$ | 94.2 | 53.6 |
| $\pi_{0.5}$ | 96.9 | **85.7** |
| OpenVLA-OFT | 97.6 | 67.9 |
| X-VLA | 98.1 | 71.4 |
| ABot-M0 | 98.6 | 80.5 |
| VLA-JEPA | 97.2 | 77.9 |
| GE-Act | 94.4 | 80.3 |
| **Cosmos-Policy** | **98.5** | 82.2 |

**这就有意思了：$\pi_{0.5}$ 在 LIBERO 上反超 WAM（85.7 vs 82.2）**。

## 5. 结果该怎么解读

### Insight 1: WAM 在 "视觉干扰类" perturbation 上碾压

看 RoboTwin 数据：

| Perturbation | LingBot-VA (WAM) | $\pi_{0.5}$ (VLA) | 差距 |
|--------------|------------------|--------------------|------|
| Light | 89.0 | 49.6 | +39.4 |
| Background | 91.3 | 71.7 | +19.6 |
| Noise | 80.9 | 64.9 | +16.0 |
| Layout | 87.9 | 56.8 | +31.1 |

这四类都是 **视觉外观变化但 task 结构不变**。WAM 强是合理的：

**Video backbone 在 pre-training 时见过几千万段 video，什么光照、什么背景、什么 noise 都见过**。它学会的是 "关注 task-relevant 的 motion pattern，忽略 appearance variation"。这个能力 transfer 到 policy 上。

特别 striking 的是 **Layout** 这一项：WAM 87.9 vs VLA 56.8，差距 31 个点。Layout 是加 3-15 个 distractor 物体。VLA 容易被 distractor 吸引注意力，WAM 因为预测的 future frame 主要关注 task-relevant motion（被 instruction 条件化），distractor 在 predicted future 里 "不动"，自然不影响 action 生成。

**Paper 里 Figure 3 还展示了一个 remarkable 现象**：Cosmos-Policy 输入图像被严重 noise corrupt，但 predicted future frame 里的 robot arm 是 clean 的。这就是 video diffusion 自带的 denoising 能力——它在训练时就是从 noisy latent denoise 出 clean frame，这个能力 transfer 到 policy 上，相当于自带 image restoration。

### Insight 2: WAM 在 "几何结构类" perturbation 上崩盘

| Perturbation | LingBot-VA (WAM) | $\pi_{0.5}$ (VLA) |
|--------------|------------------|--------------------|
| Camera | 28.9 | 45.6 |
| Robot Init | 36.2 | 27.6 |

**Camera perturbation 是所有方法的死穴**。SOTA 的 LingBot-VA 只有 28.9%。

这看似矛盾：video 不是见过各种视角吗？但要注意 video 见的是 **camera ego-motion**（手持拍摄、车载），scene 内容不变。LIBERO 的 camera perturbation 是 **固定 camera 的 extrinsic 变化**，scene 中 robot 和 object 的 pixel projection 变了，但 robot proprioception 不变。

WAM 的 future frame prediction 依赖 "current pixel pattern → future pixel pattern" 的 mapping。camera 角度一变，这个 mapping 就 out of distribution 了。

**VLA 在 camera 上反而稍强（$\pi_{0.5}$ 45.6 vs 28.9）**，可能因为 $\pi_{0.5}$ 训练数据包含 multi-camera mobile manipulation，对 viewpoint variation 有 implicit robustness。

**Robot Init 上 VLA 稍弱，WAM 也弱**。两者对 "robot 初始姿态变化" 都不够 robust。MOTUS 例外（85.0），因为它用 optical-flow based latent action，embodiment-agnostic，只关心 "视觉上物体怎么动"，robot 姿态变化不影响 latent action。

### Insight 3: $\pi_{0.5}$ 在 LIBERO 上反超 WAM

这个结果非常 important，需要仔细解读。

$\pi_{0.5}$ 在 LIBERO-Plus 总分 85.7，超过 Cosmos-Policy 82.2。但在 RoboTwin 上只有 58.6，远低于 LingBot-VA 74.2。为啥？

看 Table 2 的 training data：

$\pi_{0.5}$ 的训练数据极其庞大：
- Cross-embodiment robot data (>10k hours)
- Mobile manipulation (400h)
- Multi-env tabletop
- Web data-VQA, captioning, grounding
- High-level planning post-training

也就是说，$\pi_{0.5}$ 用 **海量 diverse data** 硬生生把 dynamics prior 给 implicit 学出来了。这能 work，但代价是数据需求巨大。

而 WAM（Cosmos-Policy）的 task-specific finetuning 只需要 **185 个 trajectory**，因为 forward dynamics 已经在 video pre-training 学好了，只需要学 inverse dynamics。

**这是两种哲学的对比**：
- VLA: "我没有好 prior，但我有海量 data，可以 brute-force 学"
- WAM: "我有好 prior，所以只需要少量 task-specific data"

### Insight 4: Hybrid 方法（MOTUS, VLA-JEPA）夹在中间

MOTUS 和 VLA-JEPA 都是 "VLA + video auxiliary" 的 hybrid：

- **MOTUS**: Wan2.2-5B video backbone + 独立 VLM action expert。video backbone 只负责 video generation，action 由 VLM 出。结果 RoboTwin 上 71.5%，比纯 VLA 强，比纯 WAM 弱。
- **VLA-JEPA**: Qwen3-VL-2B + future state prediction auxiliary loss 在 human ego video 上训练。LIBERO 上 77.9%，介于纯 VLA 和 WAM 之间。

**这说明 video prior 可以通过 auxiliary task 注入 VLM，但效果不如 native video backbone**。VLM 的 attention pattern 没有为 temporal prediction 优化，auxiliary loss 只能在 latent space 注入部分 dynamics 知识。

换句话说：**Video prior 怎么集成很重要，不只是 "有没有" 的问题**。Native WAM > Hybrid VLA+WM > 纯 VLA。

### Insight 5: WAM 慢得离谱

| Model | Inference time | vs. $\pi_{0.5}$ |
|-------|----------------|-----------------|
| $\pi_{0.5}$ | 63 ms | 1.0× |
| GE-Act | 300 ms | 4.8× |
| Cosmos-Policy | 390 ms | 6.2× |
| LingBot-VA(RW) | 480 ms | 7.6× |
| MOTUS | 1175 ms | 18.6× |
| LingBot-VA(RT) | 5230 ms | 83.0× |

WAM 慢的核心原因是 **video diffusion 需要多步 denoising**。每一步要 full backbone forward 一遍。LingBot-VA 在 RoboTwin 设置下用 25 步 state denoising + 50 步 action denoising，所以慢 83 倍。Real-world 设置减到 3+5 步，但仍慢 7.6 倍。

Fast-WAM 和 GigaWorld-Policy 试图通过 "训练时联合 denoise state+action，推理时只 denoise action" 来加速，报告 190ms。但这仍然比 $\pi_{0.5}$ 慢 3 倍，而且性能略降。**WAM 的速度问题还没真正解决**。

## 6. 我的几点 take

### Take 1: Backbone prior 决定上限

2B 参数的 Cosmos-Policy 在 robustness 上能 beat 5.3B 的 LingBot-VA 在部分维度，因为 Cosmos-Predict2 的 video pre-training data 更 diverse。**Video foundation model 的 quality 是 WAM 上限的决定因素**，policy finetuning 只是 "激活" 这个 prior。

### Take 2: Spatiotemporal prior 确实 transferable

从 web video 到 robot policy 的 transfer verified 了。但 transfer 的是 **dynamics understanding**，不是 task knowledge。Robot 还是要 finetune 才能做具体 task。

### Take 3: Inverse dynamics decomposition 是 data efficiency 的 key

WAM 的 IDM 方案：

$$\underbrace{p_\phi(h_{t+1} \mid h_t)}_{\text{video backbone 搞定}} \cdot \underbrace{g_\psi(a_t \mid h_t, h_{t+1})}_{\text{finetune 学这个}}$$

Forward dynamics 外包给 video pre-training，finetune 只学 inverse dynamics，所以 task-specific data 需求降到 50-185 trajectory。VLA 想达到同样 robustness 需要几千小时 diverse data。

### Take 4: Camera robustness 需要 3D

纯 2D video prior 解决不了 viewpoint invariance。这是 WAM 下一个 frontier。可能方向：
- Multi-view video pre-training
- 3D Gaussian Splatting 作为 state representation
- Depth-aware video diffusion
- Voxel features 补充 2D image

### Take 5: 不同 robustness 维度需要不同 inductive bias

这篇 paper 最有价值的 insight 是：

| Robustness 维度 | 谁强 | 需要啥 prior |
|----------------|------|-------------|
| Visual appearance (light, noise, BG) | WAM | Video temporal prior |
| Layout (distractor) | WAM | Temporal attention 过滤 static |
| Camera viewpoint | 都弱 | 3D geometric prior |
| Robot init state | 都弱 | Control theory prior |
| Language | VLA | Semantic understanding |

**单一 backbone 难以覆盖所有维度**。未来 robot foundation model 可能需要 multi-backbone architecture：VLM 搞语义，video model 搞 dynamics，3D encoder 搞 geometric invariance。

### Take 6: 速度不解决 WAM 只能上 quasi-static task

480ms-5230ms 的 inference time 意味着 robot 在 action chunk 执行期间基本是 "盲动"。对 dynamic environment（物体在动、人在动）完全不行。要上 real-time dynamic task，WAM 需要架构级加速：
- Consistency distillation 把 multi-step diffusion 蒸馏成 1-2 步
- Autoregressive action token（像 $\pi_0$-FAST 那样）
- Hybrid: VLA 做 high-level planning（慢），WAM 做 low-level refinement（按需触发）

## 7. 一句话总结

**WAM 用 video diffusion backbone 自带的 spatiotemporal prior，在 visual perturbation (light, noise, background, layout) 上碾压 VLA，而且 data efficiency 高得多。但 camera viewpoint 和 robot initial state 上崩盘，inference speed 慢 5-80 倍。VLA 通过海量 diverse data（如 $\pi_{0.5}$）可以追上 WAM 的 robustness，但代价是数据需求巨大。未来方向是 multi-backbone hybrid + 3D-aware representation + 架构级加速。**

---

**相关 reference 链接**:

- 主 paper (LIBERO-Plus): https://arxiv.org/abs/2510.13626
- Cosmos world foundation: https://arxiv.org/abs/2501.03575
- Cosmos-Policy: https://openreview.net/forum?id=wPEIStHxYH
- $\pi_0$: https://arxiv.org/abs/2410.24164
- $\pi_{0.5}$: https://arxiv.org/abs/2504.16054
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- DreamerV3: https://arxiv.org/abs/2301.04104
- OpenVLA: https://arxiv.org/abs/2406.09246
- X-VLA: https://openreview.net/forum?id=kt51kZH4aG
- DreamVLA: https://arxiv.org/abs/2507.01005
- WorldVLA: https://arxiv.org/abs/2506.21539
- RT-2: https://arxiv.org/abs/2307.15818
- PaLM-E: https://arxiv.org/abs/2303.03378
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- TD-MPC2: https://arxiv.org/abs/2310.16828
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- GigaWorld-Policy: https://arxiv.org/abs/2603.17240
- Fast-WAM: https://arxiv.org/abs/2603.16666
- DreamZero: https://arxiv.org/abs/2602.15922
- MOTUS: https://arxiv.org/abs/2512.13030

---

# World Action Models vs. VLAs: 一场关于 "机器人 policy 的 backbone 该选什么" 的系统性对比实验

## 1. 这篇 paper 在问什么问题

这本质上是一个 **backbone prior transfer** 的问题。当前 robot policy 领域存在两条主线：

- **VLA (Vision-Language-Action)**：基于 VLM backbone（如 PaliGemma, Qwen-VL），这些 backbone 通过 next-token prediction 在 image-text pairs 上训练，擅长 semantic understanding，但对 dynamic evolution 基本没有 explicit 建模
- **WAM (World Action Model)**：基于 video diffusion backbone（如 Cosmos-Predict2, Wan2.1/2.2, LTX-Video, SVD），这些 backbone 通过 action-conditioned future frame prediction 训练，天然携带 spatiotemporal dynamics 的 prior

核心 hypothesis 是：**video pre-training 阶段获得的对 physical dynamics 的 prior，能否在下游 robot task 上以更少的 task-specific data 换取更强的 robustness？** 也就是说，WAM 是否真的 "generalize better"？

Reference:
- Cosmos world foundation model: https://arxiv.org/abs/2501.03575
- Wan2.1 video generation: https://arxiv.org/abs/2503.20314
- π0 flow matching VLA: https://arxiv.org/abs/2410.24164

## 2. VLA vs. WAM 的数学本质区别

### 2.1 Prediction scheme 的形式化

**VLA 的 prediction**:

$$p_\theta(a_t \mid h_t)$$

- $\theta$: VLA policy 的参数
- $a_t$: 时刻 $t$ 的 action（如 7-DoF delta EEF 或 14-DoF joint positions）
- $h_t$: 时刻 $t$ 的 observation history encoding（通常是当前 image + language instruction + proprioceptive state 经 VLM encoder 后的 latent）

这是一个 **direct state→action mapping**，模型必须 implicit 地推断 "如果我做 action $a_t$，世界会变成什么样"。

**WAM 的 prediction** 有两种变体：

**(a) Joint prediction**:

$$p_\phi(h_{t+1}, a_t \mid h_t)$$

- $\phi$: WAM 的参数（包含 video diffusion backbone）
- $h_{t+1}$: 预测的下一时刻 visual state（在 latent space 或 pixel space）
- $a_t$: 同时输出的 action

这是 Cosmos-Policy 和 DreamZero 采用的方案。两者通过 joint denoising 同时生成 future frame 和 action。

**(b) Inverse Dynamic Model (IDM) 分解**:

$$p_\phi(h_{t+1} \mid h_t) \cdot g_\psi(a_t \mid h_t, h_{t+1})$$

- $p_\phi$: video backbone 负责 forward dynamics $\rightarrow$ 预测 future visual state
- $g_\psi$: 轻量 action decoder 负责 inverse dynamics $\rightarrow$ 给定当前 state 和 predicted future state，推断实现该 transition 所需的 action
- $\psi$: action decoder 的参数（通常是一个 flow matching head，参数量远小于 $\phi$）

这是 LingBot-VA, GE-Act, mimic-video 采用的方案。核心 insight 是：**forward dynamics $p_\phi(h_{t+1}|h_t)$ 已经被 web-scale video pre-training 很好地学习了，embodied pre-training 只需要学 inverse dynamics $g_\psi$，这是一个容易得多的问题。**

**(c) Causal conditioning（GigaWorld-Policy）**:

$$p_\phi(h_{t+1} \mid h_t, a_t) \cdot g_\psi(a_t \mid h_t)$$

先预测 action，再以 action 为 condition 生成 future state。在 test time 可以跳过 future state 生成，直接出 action，加速推理。

### 2.2 为什么 IDM 分解有意义

这里可以引入一个 control theory 视角。在 robotics 中：

- **Forward model**: $s_{t+1} = f(s_t, a_t)$ — 预测 action 的后果
- **Inverse model**: $a_t = f^{-1}(s_t, s_{t+1})$ — 给定起止 state，推断 action

Inverse model 通常比 forward model 容易学，因为：
1. Forward model 需要 modeling 环境的全部 dynamic（包括 distractor 物体的运动、光照变化等 task-irrelevant 因素）
2. Inverse model 只需要 modeling "从 $s_t$ 到 $s_{t+1}$ 需要 what action"，task-irrelevant 因素可以 marginal 掉

WAM 的 IDM 方案巧妙地利用了这一点：把 forward model 的学习 **外包给 web-scale video pre-training**，robot-specific finetuning 只负责 inverse model。这解释了为什么 Cosmos-Policy 可以 "pretrain-free"（Table 1），只需要 185 个 trajectory 的 task-specific finetuning。

Reference:
- Forward vs inverse dynamics in robot learning: https://arxiv.org/abs/1705.05420
- Inverse dynamics in video policy: https://arxiv.org/abs/2512.15692

## 3. Spatiotemporal Prior 的传递机制

这是理解整篇 paper 的关键 intuition。

### 3.1 VLM backbone 学到了什么

VLM（如 Qwen3-VL-2B, PaliGemma）的训练目标是：

$$\mathcal{L}_{VLM} = -\sum_{t} \log P(x_t \mid x_{<t}, \text{image}, \text{instruction})$$

即 next-token prediction。image 通过 vision encoder 编码为 patch tokens，参与 cross-attention。学到的 prior 主要是：
- **Object semantics**: "这是一个红色方块"
- **Spatial relations**: "方块在杯子左边"
- **Visual grounding**: "指令中的 hammer 对应图像中那个金属物体"

但 **完全没有时间维度**。VLM 不知道 "如果锤子敲下去，方块会飞出去" 这种 causal physics。

### 3.2 Video diffusion backbone 学到了什么

Video diffusion model（如 Cosmos-Predict2）训练目标是去噪 future frame:

$$\mathcal{L}_{video} = \mathbb{E}_{t, \epsilon, x_0, x_t}\left[\| \epsilon - \epsilon_\phi(x_t, t, \text{condition}) \|^2\right]$$

- $x_0$: clean future frame
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$: 加噪后的 frame
- $\epsilon \sim \mathcal{N}(0, I)$: Gaussian noise
- $\bar{\alpha}_t$: noise schedule 的累积系数
- $\epsilon_\phi$: noise prediction network（即 video diffusion backbone）

这个训练过程在千万级 video 上进行，涵盖自然 dynamics、手部动作、驾驶、物体交互等。学到的 prior 包括：
- **Physical dynamics**: 物体下落、碰撞、摩擦
- **Hand-object interaction**: 手指如何抓取、推动
- **Temporal consistency**: 同一物体跨帧的 identity 保持
- **Camera ego-motion**: 视角变化引起的 pixel flow
- **Fine-grained motion**: 比 VLM 的 frame-level 理解细粒度得多

### 3.3 Prior 如何在 finetuning 阶段被激活

WAM 的 embodied pre-training 阶段，video backbone 接收的输入变成了：

$$\text{input} = [\text{image}_t, \text{robot\_state}_t, \text{instruction}]$$

其中 robot_state 被编码为额外的 latent frame（Cosmos-Policy 的做法）或 cross-attention token。Output 变成 future frame + action。

关键在于：**backbone 的 spatiotemporal prior 并没有被 "覆盖"，而是被 "条件化" 为 "如果机器人手臂这样动，场景会这样演化"**。这是参数高效的，因为 physics prior 本身是 task-agnostic 的。

VLA 想达到同样效果，必须在 embodied pre-training 阶段喂入大量 diverse dynamic data（如 $\pi_{0.5}$ 用了 400h mobile manipulation + cross-embodiment + web data），让 VLM 从零开始 implicit 学习 dynamics。这正是 Table 2 中 $\pi_{0.5}$ 训练数据如此复杂的原因。

Reference:
- Cosmos-Policy: https://openreview.net/forum?id=wPEIStHxYH
- $\pi_{0.5}$: https://arxiv.org/abs/2410.24164 (Pi0)
- JEPA pre-training for policy: https://arxiv.org/abs/2506.09985

## 4. 实验设置的深度解读

### 4.1 两个 benchmark 的互补性

| Aspect | LIBERO-Plus | RoboTwin 2.0-Plus |
|--------|-------------|-------------------|
| Simulator | MuJoCo (robosuite) | SAPIEN (ManiSkill3) |
| Robot | Franka Panda (7-DoF) | Aloha-AgileX (14-DoF) |
| Arms | Single | Dual |
| Cameras | 2 (third + wrist) | 3 (head + 2 wrist) |
| Action space | 7-dim delta EEF | 14-dim joint positions |
| Control freq | 10 Hz | 25-30 Hz |
| Total trajectories | 22,400 | 27,500 |

**LIBERO-Plus 评估 single-arm dexterity under perturbation，RoboTwin 2.0-Plus 评估 bimanual coordination under perturbation**。两者在 embodiment、observation 配置、action space 上完全不同，结果一致性增强了结论的可信度。

### 4.2 七维 perturbation taxonomy

这是这篇 paper 最有价值的设计之一。不是简单地加 random noise，而是结构化地分解 robustness：

1. **Camera (C1-C3)**: viewpoint pose 变化 — 测试 geometric invariance
2. **Robot initial state**: joint configuration 变化 — 测试 motor control 的 robustness
3. **Language (R1-R3)**: instruction paraphrase — 测试 semantic robustness
   - R1: distraction wrapping (~30%)
   - R2: common-sense rewording (~50%)
   - R3: reasoning chain (~20%)
4. **Light (L1-L4)**: illumination intensity, direction, shadow, specular
5. **Background (B1-B2)**: texture swap, material variation
6. **Noise (N1-N5)**: motion blur, Gaussian blur, zoom blur, fog, glass blur
7. **Layout (O1-O2)**: distractor objects (3-15), target pose perturbation

每种 perturbation 都是单独激活的（除了 light 中的 L2/L4 联动），这样可以 isolate 每个维度对 performance 的影响。这是 standard robustness evaluation 的做法，比 aggregate score 信息量大得多。

每个 perturbation 的具体参数化很有讲究，例如 Table 6 中的：
- L1 diffuse color: per-channel RGB tint ∈ [0.0, 3.5] — 注意这里 0 表示完全关掉某个 channel，3.5 表示 saturate，测试 extreme lighting
- N4 fog: transmittance $e^{-\alpha d}$, $\alpha \in [0.3, 1.5]$, $d=3$ — 物理 accurate 的 fog model
- O2 target pose: $\sigma = 2$ cm Gaussian + $\pm 15°$ yaw — 测试 grasping 对 object pose 的 tolerance

Reference:
- LIBERO-Plus: https://arxiv.org/abs/2510.13626
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088

## 5. 关键实验数据的深度解读

### 5.1 RoboTwin 2.0-Plus 结果 (Table 3)

| Model | Original | Camera | Robot | Lang | Light | BG | Noise | Layout | Total |
|-------|----------|--------|-------|------|-------|-----|--------|--------|-------|
| $\pi_{0.5}$ | 78.4 | 45.6 | 27.6 | 74.4 | 49.6 | 71.7 | 64.9 | 56.8 | 58.6 |
| X-VLA | 65.6 | 23.2 | 65.2 | 64.4 | 63.1 | 58.6 | 49.7 | 34.8 | 53.1 |
| MOTUS | 87.0 | 21.6 | 85.0 | 83.2 | 84.6 | 84.4 | 43.1 | 82.8 | 71.5 |
| **LingBot-VA** | **92.1** | 28.9 | 36.2 | 87.3 | **89.0** | **91.3** | **80.9** | **87.9** | **74.2** |

**关键观察**：

**(a) Camera perturbation 是所有方法的弱点**

即使是 SOTA 的 LingBot-VA 也只有 28.9% success rate（vs. 92.1% clean）。这非常合理：camera viewpoint 改变意味着 pixel-level observation 完全变化，而 video backbone 的 prior 是基于训练时的视角分布学的。要解决这个需要 **3D-aware representation**（如 NeRF, 3D Gaussian Splatting, 或 voxel features），而非 2D image generation。

$\pi_{0.5}$ 在 camera perturbation 上反而相对强（45.6），可能因为它的训练数据包含了 multi-camera mobile manipulation，对 viewpoint variation 有 implicit robustness。

**(b) WAM 对 visual perturbation (Light, Noise, Layout) 显著强**

LingBot-VA 在 Light (89.0), Noise (80.9), Layout (87.9) 上都接近 clean performance。这强烈支持 spatiotemporal prior 假设：video backbone 见过各种光照、各种 noise、各种 distractor，学会了 "关注 task-relevant 的 motion 而忽略 distractor"。

特别值得注意的是 **Noise 鲁棒性**：LingBot-VA 80.9 vs. MOTUS 43.1 vs. $\pi_{0.5}$ 64.9。这是因为 video diffusion 训练中，模型必须从 noisy latent 中 denoise 出 clean frame，本质上就是在做 noise robustness training。这个能力 transfer 到了 policy 上。

**(c) Layout perturbation 上 WAM 优势最明显**

LingBot-VA 87.9 vs. $\pi_{0.5}$ 56.8。Layout 是加入 3-15 个 task-irrelevant distractor objects。VLA 容易被 distractor 吸引（attention 分散），而 WAM 因为预测的 future frame 主要关注 task-relevant motion（被 instruction 条件化），distractor 在 predicted future 中 "不动"，不影响 action 生成。

这是 paper 中 Figure 2(b) 的 case study 展示的：$\pi_{0.5}$ 被 distractor block 干扰，LingBot-VA 正常完成。

### 5.2 LIBERO-Plus 结果 (Table 4)

| Model | Original | Total |
|-------|----------|-------|
| $\pi_0$ | 94.2 | 53.6 |
| $\pi_0$ (rerun) | 91.3 | 69.4 |
| $\pi_{0.5}$ | 96.9 | **85.7** |
| OpenVLA-OFT | 97.6 | 67.9 |
| X-VLA | 98.1 | 71.4 |
| ABot-M0 | 98.6 | 80.5 |
| VLA-JEPA | 97.2 | 77.9 |
| GE-Act | 94.4 | 80.3 |
| **Cosmos-Policy** | **98.5** | 82.2 |

**关键观察**：

**(a) $\pi_{0.5}$ 在 LIBERO-Plus 上反超 Cosmos-Policy**

这与 RoboTwin 上的结果相反。解释：
1. LIBERO 是 $\pi$-series 的 "主场"，robosuite/MuJoCo 环境与 $\pi$ 的训练分布接近
2. $\pi_{0.5}$ 的 training data（Table 2）包含 multi-env tabletop + cross-embodiment + mobile manipulation，对 LIBERO 这种 tabletop manipulation 覆盖很好
3. $\pi_{0.5}$ 是 post-trained 模型，有专门的 robustness post-training

**(b) VLA-JEPA 的中间表现验证了 "partial video prior" 的价值**

VLA-JEPA 用 VLM backbone (Qwen3-VL-2B) + future-state prediction auxiliary loss 在 human ego video (220k) 上训练。结果 77.9% total，介于纯 VLA 和 WAM 之间。

这说明：**spatiotemporal prior 可以通过 auxiliary task 注入 VLM，但效果不如 native video backbone**。原因可能是 VLM 的 attention pattern 没有原生为 temporal prediction 优化，auxiliary loss 只能在 latent space 注入部分 dynamics 知识。

**(c) GE-Act 和 Cosmos-Policy 的对比**

两者都是 WAM，但 GE-Act 80.3 vs. Cosmos-Policy 82.2。差异来源：
- GE-Act: LTX-Video-2B backbone + 独立 flow matching action decoder（MOT 架构）
- Cosmos-Policy: Cosmos-Predict2-2B backbone + 统一 latent frame 编码 action（unified transformer）

Cosmos-Policy 的优势在于 **不破坏 video backbone 的 latent space**——action 被编码为 latent frame，参与同一个 diffusion process，因此 action 和 future state 在同一 representation space 中联合生成。GE-Act 的独立 decoder 可能引入 representation gap。

### 5.3 Inference speed (Table 5)

| Model | Action chunk | Inference t | vs. $\pi_{0.5}$ |
|-------|--------------|-------------|-----------------|
| $\pi_{0.5}$ | 50 | 63 ms | 1.0× |
| X-VLA | 30 | 195 ms | 3.1× |
| GE-Act | 36 | 300 ms | 4.8× |
| Cosmos-Policy | 16 | 390 ms | 6.2× |
| LingBot-VA(RW) | 32 | 480 ms | 7.6× |
| MOTUS | 16 | 1175 ms | 18.6× |
| LingBot-VA(RT) | 32 | 5230 ms | 83.0× |

**WAM 慢的原因**：

Video diffusion 需要 $T$ 步 denoising，每步需要 full backbone forward。对于 action chunk size $K$，总计算量约为：

$$\text{FLOPs} \propto T_{state} \cdot K \cdot N_{backbone} + T_{action} \cdot K \cdot N_{decoder}$$

- $T_{state}$: future state denoising steps
- $T_{action}$: action denoising steps
- $N_{backbone}$: video backbone 参数量
- $N_{decoder}$: action decoder 参数量

LingBot-VA(RT) 用 25 state steps + 50 action steps，所以慢了 83 倍。LingBot-VA(RW) 减到 3+5 步，降到 7.6 倍。但即便如此，对 50Hz 控制频率，480ms 意味着 24 个 control step 内 robot 在 "盲动"，必须靠 open-loop chunk execution 补偿。

**Fast-WAM 的思路**：训练时 joint denoising state+action，推理时只 denoise action（跳过 state 生成），报告 190ms。但仍然比 $\pi_{0.5}$ 慢 3 倍，且性能略降。这引出一个开放问题：**能否在训练时让 action 生成 path 成为 self-sufficient，state generation 只作为 auxiliary regularization？** 这本质上回到了 VLA-JEPA 的思路。

Reference:
- Fast-WAM: https://arxiv.org/abs/2603.16666
- GigaWorld-Policy: https://arxiv.org/abs/2603.17240
- DreamZero: https://arxiv.org/abs/2602.15922

## 6. 对几个反直觉现象的解读

### 6.1 为什么 WAM 在 camera perturbation 上崩盘

这看似矛盾——video 不是应该见过各种视角吗？但要注意：

Video pre-training 见的是 **natural camera motion**（如手持拍摄、车载），而 LIBERO-Plus 的 camera perturbation 是 **固定 camera 的 extrinsic 变化**。这是两种不同的 distribution shift：

- Natural camera motion: scene 内容不变，视角 ego-motion
- Camera extrinsic perturbation: scene 中 robot 和 object 的 pixel projection 变化，但 robot proprioception 不变

WAM 的 future frame prediction 依赖 "current pixel pattern → future pixel pattern" 的 mapping。当 camera 角度变了，这个 mapping 在 training distribution 外。而 VLA 的 direct $h_t \rightarrow a_t$ mapping 对视角变化的依赖可能更间接（通过 VLM 的 semantic understanding）。

可能的解决方向：**camera-conditioned WAM**，将 camera extrinsic 作为 explicit condition，或者用 3D representation（如 depth, point cloud）补充 2D image。

### 6.2 为什么 MOTUS 在 robot initial state perturbation 上最强 (85.0)

MOTUS 用 latent action space（optical flow based, 14-dim），这个 representation 是 **embodiment-agnostic** 的——它学的不是 "joint angle" 而是 "视觉上物体如何移动"。当 robot initial state 变化时，task 所需的 visual outcome 不变（block 还是要放到 bowl 里），所以 latent action 不变，只是 mapping 到具体 joint command 时调整。

这暗示：**action representation 的抽象层级 影响 robustness**。越抽象（latent, optical flow）越 robust to embodiment variation，越具体（joint angles）越 brittle 但 control 精度越高。

### 6.3 为什么 Cosmos-Policy 对 noise 鲁棒性比 LingBot-VA 更好

Cosmos-Policy 在 LIBERO-Plus Noise 上 92.7%，LingBot-VA 在 RoboTwin Noise 上 80.9%。虽然 benchmark 不同，但 Cosmos-Policy 在 Figure 3 中展示了一个 remarkable 能力：**当输入图像被 noise 严重 corrupt 时，predicted future frame 中的 robot arm 是 clean 的**。

这是 **video diffusion 的 implicit denoising 能力**。Cosmos-Predict2 在 training 时见过 noisy video（web video 本身就有 compression artifact、motion blur），学会了 "想象 clean future" 的能力。这个能力直接 transfer 到 policy：即便 input noise，predicted future 仍然清晰，action 从 clean future 解码，自然 robust。

LingBot-VA 用 Wan2.2-5B backbone，可能训练数据中 noise distribution 不同，或者 IDM 架构下 action 直接 condition on noisy input 而非 predicted future，导致 denoising 能力没充分利用。

## 7. 对未来研究方向的思考

### 7.1 WAM 的速度问题需要架构级创新

现有优化（减少 denoising steps、跳过 state 生成）都是在现有框架内压榨。根本性方案可能包括：

- **Consistency distillation for action**: 把 multi-step diffusion 蒸馏成 1-2 步
- **Autoregressive action token**: 像 LLM 一样 AR 生成 action token（已有 $\pi_0$-FAST 验证）
- **Hybrid backbone**: VLM 做 high-level planning（慢），WAM 做 low-level refinement（按需触发）

### 7.2 3D-aware WAM 是 camera robustness 的关键

当前 WAM 是 2D video generation，本质对 viewpoint 不 invariant。引入：
- Multi-view video pre-training
- 3D Gaussian Splatting 作为 state representation
- Depth-aware video diffusion

可能解决 camera perturbation 问题。

### 7.3 Hierarchical WAM

H-WM (Chen et al., 2026) 探索了 hierarchical planning：symbolic task-level + visual state-level。这可能是 long-horizon task 的方向。关键在于 **如何让 hierarchy 的不同层共享 spatiotemporal prior**。

### 7.4 VLA + WAM 的真正 hybrid

当前 "hybrid" 方法（MOTUS, VLA-JEPA）只是把 video backbone 作为 auxiliary。真正的 hybrid 可能需要：
- WAM 提供 "imagination"（未来视觉预测）
- VLA 提供 "reasoning"（语言理解和任务规划）
- 两者通过 cross-attention 在 latent space 交互

这有点像 System 1 (WAM, fast intuitive dynamics) + System 2 (VLA, slow deliberate reasoning) 的 cognitive architecture。

Reference:
- H-WM: https://arxiv.org/abs/2602.11291
- CoT-VLA: https://arxiv.org/abs/2503.22020

## 8. 总结：这篇 paper 真正告诉我们什么

1. **Backbone prior matters more than model size**: 2B 的 Cosmos-Policy 在 robustness 上 beat 5.3B 的 LingBot-VA 在部分维度（因为 Cosmos-Predict2 的 video pre-training data 更 diverse）。这暗示 video foundation model 的 quality 是 WAM 上限的决定因素。

2. **Spatiotemporal prior is transferable**: 从 web video 到 robot policy 的 transfer 是 verified 的，且 transfer 的是 "dynamics understanding" 而非 "task knowledge"。

3. **Inverse dynamics decomposition is parameter-efficient**: IDM 方案让 robot-specific data 需求降到 50-185 trajectories，这是 VLA 难以匹敌的 data efficiency。

4. **Speed is the bottleneck**: WAM 离 real-time deployment 还差 5-80 倍速度。这个问题不解决，WAM 只能用于 quasi-static manipulation。

5. **Camera robustness 需要 3D**: 纯 2D video prior 无法解决 viewpoint invariance，这是 WAM 的下一个 frontier。

6. **VLA 通过 data diversity 可以补偿**: $\pi_{0.5}$ 用海量 diverse data 达到可比 robustness，证明 "data scale + diversity" 仍然是一个可行路径，与 "backbone prior" 是 alternative strategies。

整体来看，这篇 paper 的最大贡献不在于 ranking WAM vs. VLA，而在于 **结构化地揭示了 robustness 的不同维度需要不同的 prior**：
- Visual robustness (noise, light, background) → video prior 帮助大
- Geometric robustness (camera, robot state) → 需要 3D + control prior
- Semantic robustness (language) → VLM prior 帮助大
- Layout robustness (distractor) → video prior 帮助大，因为 temporal attention 自然过滤 static distractor

这给未来的 robot foundation model 设计提供了清晰的 intuition：**不同 robustness 维度需要不同 inductive bias，单一 backbone 难以全部覆盖**。

---

**相关 reference 链接汇总**:

- 主 paper: https://arxiv.org/abs/2510.13626 (LIBERO-Plus)
- Cosmos world foundation: https://arxiv.org/abs/2501.03575
- Cosmos-Policy: https://openreview.net/forum?id=wPEIStHxYH
- $\pi_0$: https://arxiv.org/abs/2410.24164
- $\pi_{0.5}$: https://arxiv.org/abs/2504.16054
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- DreamerV3: https://arxiv.org/abs/2301.04104
- OpenVLA: https://arxiv.org/abs/2406.09246
- X-VLA: https://openreview.net/forum?id=kt51kZH4aG
- DreamVLA: https://arxiv.org/abs/2507.01005
- WorldVLA: https://arxiv.org/abs/2506.21539
- RT-2: https://arxiv.org/abs/2307.15818
- PaLM-E: https://arxiv.org/abs/2303.03378
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- TD-MPC2: https://arxiv.org/abs/2310.16828
- Navigation World Model: https://arxiv.org/abs/2411.15946
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
