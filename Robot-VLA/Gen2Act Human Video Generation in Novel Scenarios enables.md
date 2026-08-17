---
source_pdf: Gen2Act Human Video Generation in Novel Scenarios enables.pdf
paper_sha256: 0daa63e7f9212f2da910a0a09fbb4504f458e45e9323c17718b35e9b4a23e327
processed_at: '2026-08-04T13:16:07-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Gen2Act 人话版

Andrej，说白了就这么个事儿：

## 核心idea

机器人数据贵，网上人类视频多到爆炸。所以——

**让 video generation model 先"想象"人类怎么做这个任务，然后机器人照着想象出来的视频执行。**

就这一句话。

## 具体怎么干的

你给机器人一个场景 + 一句话任务描述，比如 "打开微波炉"。

1. VideoPoet（Google 的视频生成模型，在 2.7 亿个视频上训练的）拿到场景图 + 文字，**零样本**生成一段人类开微波炉的视频
2. 机器人 policy 看着这段视频，输出 end-effector 动作去执行

就这么简单。没有 fine-tuning video model，没有收集 paired human-robot data。

## 为什么这样做有效

关键 insight 是把问题**劈成两半**：

- **"做什么 + 怎么做"** → video generation model 负责。这东西在 web-scale data 上训练过，见过无数物体和动作，generalization 能力极强
- **"翻译成机器人动作"** → 小 policy 负责。这个任务简单多了，只需要学"人手的运动 → 机器人 arm 的运动"这个 mapping

所以 generalization 的 burden 转移到了 video model 上。而 video generation 恰好是目前 AI 最火的领域，进步最快。**视频模型变强，Gen2Act 自动变强。**

## track prediction 是干嘛的

这是唯一的 technical trick。

光看视频的 visual features 可能不够——policy 可能只关注了"长什么样"而忽略了"怎么动"。所以作者加了一个 auxiliary loss：

在生成的视频上跑一个 point tracker，得到一堆点的运动轨迹。然后逼 policy 的内部 representation 也能预测这些轨迹。

**效果**：policy 的 latent tokens 被迫编码 motion 信息，inference 时 tracker 不用，但 representation 已经被 shape 成 motion-aware 的了。

跟你的直觉一致——**auxiliary task 的 gradient 会塑造中间表示**。

## 结果有多 convincing

| 方法 | 没见过的物体 (OTG) | 没见过的动作 (MTG) |
|------|---------|---------|
| RT1 (纯机器人数据) | 0% | 0% |
| RT1-GC (goal image) | 5% | 0% |
| Vid2Robot (真实人类视频配对) | 25% | 0% |
| Gen2Act | **58%** | **30%** |

RT1 在没见过的物体和动作上直接归零。这说明**纯靠堆机器人数据这条路有天花板**。

Goal image conditioning 也基本没用——一张静态图只能告诉你"做到什么样"，没法告诉你"怎么动"。

Gen2Act 把 MTG 从 0% 拉到 30%，track prediction 贡献了 +25%。novel motion 最需要 explicit motion encoding，验证了 auxiliary loss 的价值。

## 还能串联做长任务

比如"做咖啡"，Gemini 把它拆成三步：

1. 打开盖子 → 生成视频 → 机器人执行
2. 放入 K-Cup → 拿上一步执行后的最后一帧作为新场景 → 生成视频 → 执行
3. 关上盖子 → 同理

成功率会逐步衰减（40% → 20% → 20%），因为 error accumulation，但整个 chain 能跑通。

## 机器人数据要多少

**就 400 条** diverse teleop demos。

对比视频生成模型训练用的 2.7 亿个视频，差了六个数量级。这就是 factorization 的威力——translation policy 的任务足够简单，不需要海量数据。

## 局限

- 视频模型生成的手还是不够真实，dexterous task 做不了
- 如果生成的视频就是错的（尤其在没见过的场景），policy 就跟着错
- chaining 会累积误差，没有 recovery 机制
- point track 是 sparse motion representation，复杂交互可能信息不够

## 一句话总结

**把 robot manipulation 的 generalization 问题，reduce 成 video generation 问题。** 视频模型负责"想象怎么做"，policy 负责"照着做"。两半各自用不同 scale 的数据，各司其职。

Reference: https://homangab.github.io/gen2act/

---

# Gen2Act: 深度技术解析

Andrej, 这篇 paper 触及了一个我认为非常重要的方向——如何将 video generation models 的 web-scale prior 注入到 robot manipulation 中。让我从 intuition 出发，逐层拆解。

## 1. Core Thesis: Factorization 作为 Generalization 的关键

这篇 paper 的核心论点可以概括为一个 factorization principle:

**将 manipulation task 分解为两个 sub-problems，各自由不同的 data scale 驱动：**

| Component | Data Source | Scale | Function |
|-----------|------------|-------|----------|
| Video Generation V(I₀, G) | Web videos (270M+) | Massive | "What & How" — motion prediction |
| Translation Policy π_θ(I_{t-k:t}, V_g) | Robot demos (~400 traj) | Small | "Execution" — embodiment mapping |

这个 factorization 的关键 insight 在于：**generalization 的 burden 转移到了 video generation model 上**，而 video generation model 恰好是当前 AI 领域投入最大、进步最快的方向之一。Translation policy 只需要学习一个相对简单的 mapping——从 human video 的 motion cues 到 robot end-effector actions。

这个思路与 RT-2 [1] 的 "web knowledge → robotic control" 有哲学上的相似性，但 Gen2Act 选择在 **motion level** 而非 semantic level 进行 transfer，这是一个更直接的路径。

## 2. Architecture 深度解析

### 2.1 整体 Pipeline

```
Input: I₀ (scene image), G (language goal)
    ↓
VideoPoet V(I₀, G) → V_g (human video, ~16 frames)
    ↓
Off-the-shelf Tracker [21] → τ_g (point tracks on V_g)
    ↓
Policy π_θ(I_{t-k:t}, V_g) → a_{t:t+h} (discretized actions)
    ↓
Robot execution at 3Hz
```

### 2.2 为什么选择 Human Video 而非 Robot Video？

这是一个关键的 design choice。Paper 给出的 reasoning:

1. **Video generation models trained on web data 主要 contain human videos** — zero-shot generation quality 高
2. **Robot video generation 需要 fine-tuning** with robot-specific data — 这会 "subtract the benefits of generalization"
3. Human hands 的 dexterity 和 motion patterns 足够 informative 给 robot policy

这里有一个 implicit assumption：**human motion 和 robot motion 之间存在一个 learnable mapping**，即使 embodiment 不同（human arm vs. robot arm with 2-finger gripper）。这个 assumption 在 manipulation 领域是 reasonable 的，因为很多 tasks 的 motion essence（approach, grasp, pull, push）是 embodiment-agnostic 的。

### 2.3 Translation Policy 架构细节

这是 paper 的 technical core。让我逐层拆解：

#### Visual Feature Extraction

```
V_g (16 frames) → ViT encoder χ → i_g (high-dim tokens)
                                    ↓
                Perceiver-Resampler Φ_g (2 layers, gated cross-attention)
                                    ↓
                              z_g (64 tokens)

I_{t-k:t} (8 frames) → ViT encoder χ → i_r
                                    ↓
                Perceiver-Resampler Φ_r (2 layers)
                                    ↓
                              z_r (64 tokens)
```

**Perceiver-Resampler** 来自 Flamingo [60]，其核心作用是 **token compression**：从 ViT 提取的大量 temporally uncorrelated tokens 压缩到固定数量（64）的 tokens。这里用 gated cross-attention layers，gating 机制允许 model 学习何时 attend to visual features vs. 何时 rely on prior information。

公式表示：

$$z_g = \Phi_g(i_g), \quad z_r = \Phi_r(i_r)$$

其中：
- $i_g = \chi(\mathbf{V}_g)$: ViT encoder $\chi$ 对 generated human video 提取的 visual features
- $i_r = \chi(\mathbf{I}_{t-k:t})$: ViT encoder 对 robot observation history 提取的 features
- $\Phi_g, \Phi_r$: 各自的 Perceiver-Resampler transformer encoders
- $z_g, z_r \in \mathbb{R}^{64 \times d}$: 压缩后的 64 个 tokens，$d$ 是 hidden dimension

#### Point Track Prediction (Auxiliary Loss)

这是 paper 最 interesting 的 technical contribution。让我详细讲。

**Motivation**: 仅靠 visual features $z_g$ 可能 **implicitly insufficient** 来 capture motion information。Video generation model 产生的 visual features 主要是 appearance-oriented 的，而 manipulation 需要 **explicit motion understanding**。

**Setup**:
1. 在 generated human video $\mathbf{V}_g$ 上运行 tracker [21] 得到 ground-truth tracks $\tau_g$
2. 在 first frame 随机采样一组 points $P^0$
3. 定义 track prediction transformer $\psi_g$:

$$\hat{\tau}_g = \psi_g(P^0, i_q^0, z_g)$$

其中：
- $P^0$: first frame 中随机采样的 points 集合，$P^0 \in \mathbb{R}^{N_p \times 2}$（$N_p$ 个点，每个点 2D 坐标）
- $i_q^0$: query image features at first frame，提供 point localization 的 visual context
- $z_g$: compressed video tokens（64 个），提供 motion information
- $\hat{\tau}_g$: predicted tracks, $\hat{\tau}_g \in \mathbb{R}^{N_p \times T \times 2}$（$T$ 个 timesteps 的 2D 坐标）

**Loss**:
$$\mathcal{L}_{\tau}^{g} = \|\tau_g - \hat{\tau}_g\|_2$$

同样地，对 robot observation history 也有一个 track prediction loss:

$$\hat{\tau}_r = \psi_r(P^{t-k}, i^{t-k}, z_r)$$
$$\mathcal{L}_{\tau}^{r} = \|\tau_r - \hat{\tau}_r\|_2$$

其中：
- $P^{t-k}$: chunk 起始帧的随机 points
- $i^{t-k}$: 起始帧的 image features
- $z_r$: robot observation 的 compressed tokens

**Track prediction transformer** 有 6 self-attention layers, 8 heads。

**Intuition**: 这个 auxiliary loss 的作用是 **force latent tokens $z_g, z_r$ to encode motion information**。虽然 track prediction transformer $\psi$ 在 inference 时不用，但它 training 时的 gradient 会 backpropagate 到 $z_g, z_r$，使得这些 tokens 变得 motion-informative。这是一种 **information bottleneck** 的设计——通过 prediction task 来 shape representation。

这个思路让我想到你的 "softmax classifier as representation learning" 的 insight——**auxiliary task 的 gradient 会 shape intermediate representations**。这里 track prediction 就是在 shape video tokens 使其 motion-aware。

#### BC Loss

Action space 被离散化为 256 bins per dimension：

$$\mathcal{L}_{BC} = \text{CrossEntropy}(\hat{a}_{t:t+h}, a_{t:t+h})$$

其中：
- $\hat{a}_{t:t+h}$: predicted action chunk（$h$ 是 prediction horizon）
- $a_{t:t+h}$: ground-truth action chunk from demonstrations
- 每个 action dimension 被离散化为 [0, 255] 的 bin
- Action space: end-effector pose + gripper open/close + terminate signal

**Total training loss**:
$$\mathcal{L} = \mathcal{L}_{BC} + \lambda_g \mathcal{L}_{\tau}^{g} + \lambda_r \mathcal{L}_{\tau}^{r}$$

（Paper 没有明确给出 $\lambda$ 值，可能是 1.0）

## 3. 实验数据深度分析

### 3.1 Generalization Hierarchy (Table I)

| Method | MG | G | OTG | MTG | Avg |
|--------|-----|-----|-----|-----|-----|
| RT1 | 68 | 18 | 0 | 0 | 22 |
| RT1-GC | 75 | 24 | 5 | 0 | 26 |
| Vid2Robot | 83 | 38 | 25 | 0 | 37 |
| Gen2Act (w/o track) | 83 | 58 | 50 | 5 | 49 |
| **Gen2Act** | **83** | **67** | **58** | **30** | **60** |

**关键 observations**:

1. **RT1 在 OTG/MTG 上完全失败 (0%)**: 纯 robot data 训练的 policy 无法 generalize 到 unseen object/motion types。这证实了 robot data scaling 的 fundamental limitation。

2. **RT1-GC (goal image conditioning) 只有 marginal improvement**: Goal image 只传达 "what"，缺少 "how" 的 motion information。这验证了 paper 的 thesis——**video 比 goal image 更 informative**。

3. **Vid2Robot 在 MTG 上也是 0%**: 即使用 real paired human-robot data，如果没有 web-scale prior，仍然无法 generalize 到 novel motions。

4. **Track prediction 的贡献**:
   - G: 58% → 67% (+9%)
   - OTG: 50% → 58% (+8%)
   - MTG: 5% → 30% (+25%!)
   
   **Track prediction 对 MTG 的巨大提升 (+25%) 特别 telling**——novel motions 最需要 explicit motion encoding，visual features alone 不足以 distinguish motion patterns。

5. **Gen2Act 在 MG 上与 Vid2Robot 持平 (83%)**: 在 seen scenarios 下，generated video 的优势不明显，因为 robot data 已经 sufficient。

### 3.2 Long-Horizon Chaining (Table II)

| Activity | Stage 1 | Stage 2 | Stage 3 |
|----------|---------|---------|---------|
| Stowing Apple | 80% | 60% | 60% |
| Making Coffee | 40% | 20% | 20% |
| Cleaning Table | 60% | 40% | 40% |
| Heating Food | 40% | 20% | 20% |

**Success rate 衰减 pattern**: Stage 1 → Stage 2 下降约 20-40%，Stage 2 → Stage 3 基本持平。

**Intuition**: 衰减主要来自 **error accumulation**——每个 stage 的 robot execution 可能偏离 ideal state，导致 next stage 的 video generation conditioned on a "wrong" scene image。这 是 chaining 的 fundamental challenge。

**Chaining 的关键 design choice**: 用 **previous rollout 的 last frame** 作为 next video generation 的 conditioning image，而不是用 initial image 生成所有 videos。这是因为 robot execution 后 scene state 会改变。这个 sequential chaining 要求 video generation 快速（VideoPoet < 10s/video）。

### 3.3 Co-Training Effect (Table III)

| Variant | MG | G | OTG | MTG | Avg |
|---------|-----|-----|-----|-----|-----|
| w/o co-train | 83 | 67 | 58 | 30 | 60 |
| w/ co-train (+400 traj) | 85 | 75 | 62 | 35 | 64 |

**仅 400 条 diverse teleop demos 就带来 +4% average improvement**。这表明 translation policy 对 data efficiency 很高——少量 diverse data 就能 improve generalization。

## 4. Failure Analysis 的 Insight

Paper Section IV-H 和 Appendix D 的 failure analysis 特别有启发性：

**Key finding**: 在 MG/G levels，video generation 的 inaccuracies 与 policy failure **弱相关**；但在 OTG/MTG levels，video generation failure **强相关** with policy failure。

这说明：
1. Policy 确实在 **using** generated video 的 motion cues（otherwise 不会 correlate）
2. 在 robot data support 充足时 (MG/G)，policy 可以 compensate for video generation 的 imperfections
3. 在 robot data support 不足时 (OTG/MTG)，policy 完全依赖 video generation 的质量

**Failure types** (Fig. 6):
- Type 1: Video generation implausible → policy fails (前 3 行)
- Type 2: Video plausible but policy fails (第 4 行) — grasping 或 trajectory following 出错

Type 2 failure 暗示了一个 limitation：**point tracks 可能 insufficient 来 capture 全部 motion information**。Paper 提到 future work 可以 explore object meshes 等 denser motion representations。

## 5. 与 Related Work 的 Positioning

### 5.1 vs. Track2Act [17]

Track2Act (同一作者群的前作) 直接 predict point tracks from web videos 然后 condition policy on tracks。Gen2Act 的区别：
- **Gen2Act**: Video generation (V_g) + track prediction (auxiliary)
- **Track2Act**: Direct track prediction from video (no generation step)

Gen2Act 的优势：video generation model (VideoPoet) 在 270M+ videos 上训练，比单独训练 track prediction model 更 scalable。且 video generation 领域的 progress 会自动 benefit Gen2Act。

### 5.2 vs. RT-2 [10]

RT-2 用 VLM 的 semantic understanding 来 condition policy。Gen2Act 用 video generation 的 **motion understanding**。两者是 complementary 的——RT-2 擅长 task understanding，Gen2Act 擅长 motion execution。

### 5.3 vs. Vid2Robot [46]

Vid2Robot 需要 **paired** human-robot data。Gen2Act 通过 **generated** human videos 自动 create pairs，无需 manual collection。这是 scalability 的关键差异。

## 6. Broader Implications & Intuition Building

### 6.1 为什么这个 approach 可能有效？

我认为 Gen2Act 的有效性源于几个 factors 的 alignment:

1. **Video generation models 的 emergent capability**: 经过 web-scale training，这些 models 实际上 learned 了 physics priors, object affordances, 和 motion patterns。即使它们不是为 robotics 设计的，这些 capabilities 是 manipulation 所需的。

2. **Embodiment gap 的 bridgeability**: Human hand 和 robot gripper 虽然 morphologically 不同，但 manipulation 的 **spatial-temporal motion structure** 有共性。Policy 只需要 learn 这个 mapping。

3. **Auxiliary loss as representation shaping**: Track prediction loss 确保 policy 的 internal representation 是 motion-aware 的，这是一个 well-designed inductive bias。

### 6.2 Scalability Outlook

Paper 的 approach 有一个 **positive feedback loop** 的 potential:

- Video generation models 在快速进步 (Sora, Emu Video, etc.)
- 这些进步会 **automatically** 提升 Gen2Act 的 performance
- Translation policy 只需要少量 robot data，且可以 incremental improve

这意味着 **Gen2Act 的 performance ceiling 会随着 video generation 领域的进步而提升**，这是一个非常 attractive 的 property。

### 6.3 Limitations & Open Questions

1. **Dexterous tasks**: Current video models 对 realistic hand generation 仍有困难，限制 dexterous manipulation
2. **Video generation latency**: 10s/video 虽然可接受，但对 reactive tasks 可能太慢
3. **Track prediction 的 sufficiency**: Point tracks 是 sparse motion representation，可能 insufficient for complex interactions
4. **Chaining robustness**: Error accumulation 问题需要 recovery policies
5. **Video-reality gap**: Generated videos 可能有 artifacts，policy 需要 robustness

### 6.4 更深层的相关联想

这个 paper 让我想到几个 broader directions:

**A. World models for manipulation**: Video generation model 实际上是一个 **implicit world model**。Gen2Act 在用这个 world model 来 "imagine" task execution，然后让 policy 跟随。这与 Dreamer [3] 系列 的 model-based RL 有哲学相似性，但 Gen2Act 用的是 web-trained generative model 而非 task-specific learned model。

**B. Diffusion policies + video generation**: 最近 diffusion policies [4] 在 manipulation 上表现很好。一个 natural extension 是用 video diffusion model 直接 generate robot trajectories（如 [57, 58]），但 paper 指出这需要 robot-specific fine-tuning。Gen2Act 的 factorization 避免了这个问题。

**C. Foundation models for robotics**: Gen2Act 是 "foundation model for robotics" 的一个 specific instantiation——用 web-trained generative model 作为 manipulation 的 foundation。这与 RT-X, OpenVLA [12] 等方向 complementary。

**D. Cross-embodiment transfer**: Human-to-robot transfer 是 cross-embodiment transfer 的一个 instance。更广义地，这个 framework 可以 extend 到 different robot embodiments（不同 arm, gripper, mobile base）。

## 7. Technical Details 容易被忽略的点

1. **Video frame sampling**: Training 时从 generated video 采样 16 frames（确保 first 和 last frame 都被采样），robot history 用 last 8 frames。这个 asymmetry (16 vs 8) 可能因为 generated video 包含完整 task trajectory，而 robot history 只需 recent context。

2. **Image preprocessing**: 所有 images resize 到 224×224，square format。VideoPoet 被适配为接受 square images。

3. **Robot reset position**: 每次任务前 reset 到 predefined position，确保 scene 不被 robot arm occlude。这是为了 video generation 的 input quality。

4. **Action representation**: End-effector space control，包含 terminate signal 和 gripper open/close。256 bins discretization per dimension。

5. **VideoPoet 的 prompt format**: "A person [task-name], static camera"。这个 "static camera" instruction 很重要——避免 generated video 有 camera motion，否则 track extraction 会 noisy。

## References

- [Gen2Act Project Page](https://homangab.github.io/gen2act/)
- [VideoPoet](https://arxiv.org/abs/2312.14125)
- [Bootstap Tracker](https://arxiv.org/abs/2402.00847)
- [RT-1](https://arxiv.org/abs/2212.06817)
- [RT-2](https://arxiv.org/abs/2307.15818)
- [Vid2Robot](https://arxiv.org/abs/2403.12943)
- [Track2Act](https://arxiv.org/abs/2405.01527)
- [Flamingo (Perceiver-Resampler)](https://arxiv.org/abs/2204.14198)
- [TAP-VID](https://arxiv.org/abs/2111.09576)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [Gemini](https://arxiv.org/abs/2312.11805)
- [CoTracker](https://arxiv.org/abs/2307.07635)
- [Dreamer V3](https://arxiv.org/abs/2301.04104)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [Learning Universal Policies via Text-Guided Video Generation](https://arxiv.org/abs/2310.08584)

---

**Final intuition**: Gen2Act 的核心贡献是一个 **scalable factorization**——让 web-scale video generation 承担 generalization 的 heavy lifting，让 small-scale robot data 只学习 embodiment-specific translation。Track prediction auxiliary loss 是连接这两个 worlds 的桥梁。这个 approach 的 beauty 在于它会 **自动 benefit from** video generation 领域的 ongoing progress，使得 robot manipulation 的 generalization 问题部分 reduce 为 video generation 问题。
