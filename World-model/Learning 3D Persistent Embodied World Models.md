---
source_pdf: Learning 3D Persistent Embodied World Models.pdf
paper_sha256: 30ca22fca365c88730098d05f590c83e4b9109f5e8b660b5abd814031ee5ab4f
processed_at: '2026-08-05T12:37:15-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话版本

**让 AI 在脑子里 "想象" 自己在房间里走来走去时，能记住刚才看到了什么，不会转头就忘。**

---

## 它要解决什么问题

想象你站在客厅里，面前有张沙发。你转身看向窗户——沙发从你视野里消失了，但你脑子里还知道沙发在你背后。

现在的 video world model 没这个能力。它就像一个**只有鱼记忆的人**：转身之后，沙发就真没了。再转回来，它可能凭空 "编" 出一个完全不同的东西放在那里。

这在看短视频时无所谓（你看 TikTok 也不需要记住上一条视频），但对机器人来说致命——机器人需要规划 "我先走到厨房拿杯子，再走回客厅倒水"，如果它走到一半就忘了客厅长什么样，计划全乱套。

paper 里 Figure 1 给了个直观例子：给定一个房间的初始画面，baseline model（NWM）生成几帧后，画框消失了、桌子不见了、房间结构都变了。而他们的方法保留了之前看到的所有东西。

---

## 他们怎么做的

核心 idea 特别朴素，朴素到你会想 "为什么之前没人做"——

**给 video model 配一个 3D 记事本。**

具体来说：

### 1. 建一个 3D 记事本

model 每看到一帧画面，就把它 "画" 到一个 3D voxel grid 上。就像你在 Minecraft 里一边探索一边把看到的方块摆下来。

这个 grid 大概是 64 米 × 32 米 × 64 米的空间，每格 0.25 米见方。每个格子里存的不是颜色，是 DINO feature——一种压缩过的 "语义指纹"。

为什么用 DINO feature 不用 RGB？因为同一个格子可能被多次观察到（你来回走过同一个地方），需要某种方式 "合并" 这些观察。RGB 没法 max-pool（取最大值没意义），但 semantic feature 可以——"这个格子最显著的语义是什么"。

### 2. 生成未来时，查这个记事本

当 model 要生成 "如果我往前走 3 步会看到什么" 时，它不凭空瞎编，而是去 3D 记事本里查：

"我现在在这个位置、朝这个方向，那我前方那些 voxel 里存了什么 feature？"

通过 cross-attention 把 memory 信息 "拉" 到 video latent 里。就像你查地图导航——地图是固定的世界坐标系，你的位置变了，查到的信息就变了，但地图本身不变。

### 3. 生成完，更新记事本

新生成的 frame 又被反投影回 3D grid，更新记事本。这样 memory 越滚越大，探索的区域越来越多。

**这是一个 "观察 → 记忆 → 查询 → 生成 → 再观察" 的闭环。**

---

## 几个技术细节的人话翻译

### Plücker embedding 是什么

你告诉 model "相机往左转 30 度"，它听不懂。因为 "30 度" 这个数字和 "画面里那个像素应该怎么变" 之间没有直接关系。

Plücker embedding 的 trick 是：**给每个像素发一张 "身份证"，写清楚 "我是从这根 ray 来的"**。

ray 从哪出发？相机中心。ray 朝哪走？穿过这个像素射向世界。这样每个像素都自带几何信息，model 不用猜 "相机姿态和我的关系"，直接看身份证就行。

### 为什么要生成 RGB-D 而不是只 RGB

因为 video latent 里如果有 depth 信号，memory block 做 cross-attention 时才能 "对齐"——"当前画面这个像素，对应 3D grid 里哪个 voxel？" 没 depth 这个对齐就没法做。

就像你看一张照片，如果不知道拍照时相机在哪、距离多远，你没法把照片里的东西精确放回 3D 世界。

### 两阶段训练是什么意思

Stage 1：先让 CogVideoX 学会 "在这个 navigation domain 里，给定 action，生成 RGB-D 视频"。这时候没有 memory，纯靠 action condition。

Stage 2：freeze 住 Stage 1 的 model，只训新加的 memory block。

为什么要这样？因为 memory block 一开始是随机初始化的。如果和 DiT 一起训，memory block 会给 DiT 喂噪声 gradient，把 pre-trained 的生成能力搞坏。

就像你雇了个新员工（memory block），先别让他碰核心业务（DiT），让他先学会怎么用公司的数据库（3D map），学会了再上手。

---

## 实验说明了什么

### Table 1 的核心 takeaway

- NWM（无 memory）：FVD 194，SRC 63.4
- 加任何 memory 都显著提升
- 他们的完整方法：FVD 92，SRC 81.7

FVD 降一半意味着视频质量+连贯性大幅提升。SRC（Scene Revisit Consistency）从 63 到 82，意味着 "走回原地看到的东西和之前一致" 的概率大幅提高。

### 消融实验的关键发现

1. **2D memory 不够，要 3D**：Image Memory（用几张 snapshot 图）SRC 69.4，2D feature map 79.2，3D voxel 81.7。空间维度逐级升级，收益逐级来。
2. **depth 信号重要**：没 depth 的 3D memory PSNR 20.6，有 depth 的 22.5。depth 是连接 2D video 和 3D memory 的桥梁。

### 下游应用的亮点

- **Ranking trajectories**：NoMaD policy sample 16 条路径，用 world model 模拟每条路径的未来，选最接近目标的。有 memory 的 model 让成功率从 36.8 翻到 70.8。
- **MPC**：纯靠 world model rollout + cross-entropy method 搜 action，也能达到不错的效果。这意味着 world model 可以当 "simulator" 用，不用真在环境里 interact。
- **Policy learning in new scene**：给几张新场景的照片初始化 memory，就能在新场景里 generate 训练数据 fine-tune policy。

---

## 它的 limitation

### 作者承认的

1. **需要 depth 数据**：真实世界数据集大多没 depth。可以用 Depth Anything 估计，但估计的 depth 不如真值精确。
2. **memory 是 static 的**：不建模动态物体。如果房间里有人走来走去、有车开过，memory 就失效了——因为 memory 是 "一次写入" 的 snapshot，不会更新已有 voxel 的内容。

### 我觉得还没说的

1. **memory 会越来越大**：探索越大空间，voxel token 越多，cross-attention 越慢。256×32×256 已经 800M floats，再大就爆显存。SLAM 领域有 submap、pose graph 这些成熟方案，这里还没借鉴。
2. **只适合 navigation，不适合 manipulation**：机器人拿起一个杯子，杯子位置变了，但 memory 里杯子的 voxel feature 还在原地。需要 object-level 的 affordance 和 dynamics，这是另一个维度的问题。
3. **inference 慢**：50 步 DDIM + memory cross-attention，实时控制做不到。目前适合 offline planning，不适合 online reactive control。

---

## 我觉得这 paper 真正的 insight

**生成模型做长 horizon simulation，必须有 explicit 的、和世界坐标系对齐的 memory。不能指望 latent space implicit 学到 persistence。**

这不是 "video model 不够大所以记不住"，而是 "2D latent 这个 representation 本身就不适合 encode 3D persistence"。你再 scale up Sora，它也不会有这个能力，因为 paradigm 不对。

这和几个老问题是同构的：
- LLM 为什么需要 KV cache？因为重新算 attention 太贵，需要 explicit 存储。
- SLAM 为什么需要 occupancy map？因为 camera frame 每帧变，需要 world frame 锚定。
- 人脑为什么需要 hippocampus 的 place cells？因为 spatial memory 需要专门的 neural substrate。

**Persistence 需要专门的 substrate。** 这是这 paper 最深的 insight。

它选的 substrate 是 3D voxel grid of DINO features，很朴素，甚至有点 "old school"（SLAM 二十年前就这么干了）。但把它和 modern video diffusion model 结合起来，加上 cross-attention memory block、Plücker camera control、RGB-D generation，就构成了一个完整的、可用的 embodied world model。

---

## 和大趋势的关系

现在 embodied AI 有两条路线在赛跑：

**路线 A：Scale up foundation model。** Sora、Genie 2、π0、RT-2——把 model 做大，data 喂多，指望 emergent ability。优点是通用、视觉质量好，缺点是 long horizon drift、没法 inspect、control 不精确。

**路线 B：Structured inductive bias。** 这篇 paper、DreamerV3、NeRF-based methods——把领域知识（3D geometry、memory、dynamics）encode 进架构。优点是 sample efficient、interpretable、long horizon stable，缺点是通用性差、需要 hand-crafted structure。

这篇 paper 是路线 B 的一个漂亮 case study。它告诉我们：**至少在 embodied navigation 这个问题上，路线 B 的 structured approach 显著优于路线 A 的 pure scale**（FVD 194 vs 92）。

未来最可能的赢家是两者的融合——large video model 做 "生成器"，structured 3D memory 做 "记忆体"，policy model 做 "决策者"。这 paper 展示了前两者的集成是可行的、有效的。第三步（policy）他们用 NoMaD + ranking 简单 demo 了，但还没 fully integrate。

如果 Yilun Du 的下一步工作把 manipulation dynamics 也 encode 进 memory（比如 object-level 3D scene graph + action-conditioned update），那就是完整的 embodied world model 了。值得期待。

---

## 给你的 actionable takeaway

如果你要从这 paper 里 steal 一个 idea 用到自己的工作里，我觉得最值得偷的是：

**"显式 memory substrate + cross-attention injection" 这个 pattern。**

不管你做的是 video generation、robot learning、还是任何 long-horizon sequence prediction，只要遇到 "model 在长 rollout 时 drift" 的问题，都可以套这个 pattern：

1. 定义一个和你的 task 对齐的 memory substrate（3D voxel、scene graph、knowledge graph、object-centric state…）
2. 设计一个 incremental update rule（observation → memory）
3. 用 cross-attention 把 memory 注入到你的 backbone（freeze backbone，只训 memory block）

这个 pattern 是 parameter-efficient、interpretable、incrementally updatable 的。它不是银弹，但在很多场景下比 "把 model 做大 10 倍" 更划算。

---

# Learning 3D Persistent Embodied World Models 深度解读

这篇 paper 由 MIT 的 Yilun Du 团队、UST 的 Dit-Yan Yeung 团队、以及 UMass 的 Chuang Gan 团队合作完成，第一作者 Siyuan Zhou。核心 idea 非常清晰：**给 video diffusion world model 装一个显式的 3D voxel memory，让 embodied agent 在长 horizon 的 imagination 里保持 3D 空间一致性**。

paper 链接：
- arXiv: https://arxiv.org/abs/2501.01890 (or 同名)
- 项目页 (UMass/MIT Embodied Intelligence): http://3d-persistent-world-model.github.io
- Yilun Du 主页: https://yilundu.github.io
- Chuang Gan 主页: https://chaang99.github.io

---

## 1. 问题动机：为什么需要 3D Persistent Memory

现有的 video world model（Sora [5]、Genie [6]、NWM [2]、UniPi [11]、Uni-Sim [35]、RoboDreamer [43]）有一个本质缺陷：**它们把 world state 编码在一个 2D latent 里**，没有任何 "scene 之外发生了什么" 的概念。对于通用 video generation 这种 "流式" 任务还 OK，但是对于 embodied agent 来说，环境是 partially observable 的——agent 转身之后，沙发就不在视野里了，但沙发并没有消失。

作者用 Figure 1 给了一个非常直观的例子：给定一段 context video 描述了左上角房间的布局，baseline（NWM）在自回归生成时，画、桌子都消失了，房间结构甚至变成了另一个空间。而带 3D memory 的方法保留了之前看到的结构。

这本质上是在问：**video diffusion model 的 2D latent space 是否能够 implicit 地 encode 一个完整的 3D scene graph？答案是 NO，至少在长 horizon 下是 NO**。作者把它做成 explicit 的 3D voxel memory，这非常接近 classical SLAM 里 occupancy grid mapping 的思想，但用 DINO feature 代替了 occupancy probability。

相关工作的谱系：
- **Classical SLAM**：ORB-SLAM, RTAB-Map — 显式 3D map 但不是 generative
- **NeRF / 3DGS** ([20]): 可以做 view synthesis 但不是 generative world model，也没法 imagine 未观察区域
- **Persistent Nature** [7]: NeRF + generative 但只做 terrain
- **InfiniCube** [19]: 用 HD map 强制 geometric consistency，但 map 是 static 的、只限自动驾驶
- **SlowFast-VGen** [17]: 用 LoRA 做 episodic memory，但需要 inference-time training，且 2D latent 没用 3D prior
- **3D-Mem** [37]: 同作者组的 prior 工作，用 snapshot images 做 VQA memory，本 paper 用它做 baseline (Image Memory)

---

## 2. 方法论

### 2.1 Formulation

World model with memory 的形式化：
$$
\{O_{t+1}, \dots, O_{t+H}\} = p_\theta(O_t, A_t, M)
$$

其中：
- $O_t$: 当前 RGB-D observation
- $A_t$: action chunk（navigation commands: move forward/backward, turn left/right；interaction commands: pick/move objects）
- $M$: 3D feature map memory，初始化为全零，增量更新
- $H$: prediction horizon

这里 $M$ 是关键，它让 model 既能 imagine 未知区域，也能在已观察区域保持一致。

### 2.2 Action Representation: Plücker Embedding

直接用 raw camera pose $[R|t]$ 来 condition video generation，pixel 和 pose 值之间的相关性很难学（[15] He et al. 的发现）。作者用 Plücker embedding 把 intrinsic $K$ + extrinsic $E=[R|t]$ 编码成一张 $h \times w \times 6$ 的 "ray image"：

对于 pixel $(u,v)$：
$$
d_{(u,v)} = R \cdot K^{-1} (u, v, 1)^T + t
$$

每个 pixel 的 6 维向量是 $(o \times d, d)$：
- $o$: ray origin = camera center in world coordinate = $\mathbf{t}$（平移向量）
- $d$: ray direction（这根 ray 从相机出发穿过 pixel $(u,v)$ 在世界中走的方向）
- $o \times d$: ray origin 和 direction 的叉积，给出 ray 的 moment（line geometry 里的概念，约束 ray 到原点的最短距离向量）

**intuition**：这种表示让每个像素都"知道"自己是从哪根 ray 来的，video model 不需要去 infer "相机姿态这个 4×4 矩阵到底意味着像素怎么变"，而是直接每个像素有自己的 ray 信息。这比 cross-attention 把 pose 作为一个 token 喂进去更适合 CogVideoX 的 DiT 架构——作者发现直接 concat 到 input image channels 比 cross-attention 效果好。

Plücker 嵌入出自 Sitzmann et al. Light Field Networks [28]。

### 2.3 3D Memory Map 构造

3D grid map 形状：$256 \times 32 \times 256 \times 384$（x, y, z, feature_dim），每个 grid 物理尺寸 $0.25\text{m} \times 1\text{m} \times 0.25\text{m}$，所以总尺寸约 $64\text{m} \times 32\text{m} \times 64\text{m}$，足够覆盖一个多层建筑。

构造步骤：
1. **Feature extraction**: DINO-v2 [21] 提取 image feature $F$，bilinear upsample 到像素级
2. **Feature unprojection**: 对每个 pixel，用 depth + intrinsic $K$ + extrinsic $E$ 反投影到 3D world grid
   - 反投影公式：3D 点 $P_{world} = R^{-1}(D_{u,v} \cdot K^{-1}(u,v,1)^T - t)$，其中 $D_{u,v}$ 是该 pixel 的深度
3. **Aggregation**: 每个 grid 通过 max-pooling 聚合所有落到它里面的 features（max-pooling 比 mean 好，因为稀疏覆盖下 mean 会被零稀释）
4. **Position encoding**: 每个 grid concat 上 3D sinusoidal absolute position embedding，保留 3D 空间关系

只取 "meaningful grids"（非零 feature 的 grid）来节省计算。

**intuition**：这等价于把 DINO feature 当作 semantic label "渲染" 到一个 3D voxel grid 上，本质是 voxelized feature field。和 NeRF 区别在于：NeRF 是 continuous MLP，这里是 discrete voxel + 显式存储；和 Occupancy Network 区别在于：这里 voxel 存的是 DINO semantic feature 而不是 occupancy probability，所以可以做 generation conditioning 而不仅是 query。

### 2.4 RGB-D Video Generation

为了把 3D 几何信息引入 video latent，作者让 model 生成 RGB-D video 而不只是 RGB。借鉴 Zhen et al. [41]（Learning 4D Embodied World Models，同组 prior work）：

- RGB 和 depth 分别用 3D VAE encode（共享 weights？paper 没明说，应该是 separate encoder）
- 3D VAE 把 $9 \times 512 \times 512$ frames 压缩到 $3 \times 64 \times 64$ latents（时间压缩 3×，空间压缩 8×）
- 输入输出层扩展以接受/输出 depth channels

为什么需要 depth？因为 video latent 里有 depth 信号后，memory block 的 cross-attention 才能 "对齐" 3D voxel grid 和当前 video frame 的几何关系。消融实验显示 depth 让 PSNR 从 20.627 涨到 22.458，SSIM 从 0.696 到 0.759。

### 2.5 Memory Block Architecture

Memory block 是 cross-attention expert block，注入到每个 CogVideoX transformer block 之后（Fig 2, 8）。

把 camera embeddings $C$ concat 到 video hidden states $H$ 上，让 model 同时知道 camera pose 和 depth。Memory block 用 expert adaptive layernorm（CogVideoX 的设计）分别从 time embedding $t$ regress video latent 和 map latent 各自的 scale/shift 参数 $\alpha, \beta, \gamma$。

公式 (2)：
$$
H_{norm}, M_{norm}, \alpha_H, \alpha_M = \text{norm}_1(H, M, t)
$$
$$
H = H + \alpha_M \cdot \text{Attn}(H_{norm}, M_{norm})
$$
$$
H = H + \alpha_H \cdot \text{ff}(\text{norm}_2(H))
$$

变量含义：
- $H$: video hidden states（query side）
- $M$: 3D map features（key/value side）
- $t$: diffusion time embedding
- $\alpha_H, \alpha_M$: expert adaptive layernorm 输出的 scale，分别控制 attention 和 feedforward 的贡献
- $\text{Attn}(H_{norm}, M_{norm})$: cross-attention，$H$ 做 query，$M$ 做 key/value
- $\text{ff}$: feedforward network
- $\text{norm}_1$: expert adaptive layernorm（输入 H, M, t 联合 conditioning）
- $\text{norm}_2$: 普通 layernorm

**intuition**：这是 Perceiver / Flamingo 风格的 cross-attention——把可变长度的 memory token 注入到 fixed-shape video latent 里。expert adaptive layernorm 让模型在不同 diffusion timestep 下动态调节 memory 注入的强度（早期 denoising 时 memory 注入可能更弱，后期 sharpen detail 时更强）。Cross-attention 而不是 full self-attention 是为了 efficiency——3D map 可能有很多 voxel token。

### 2.6 两阶段训练

直接从头训 memory block + DiT 在 3D feature map 上太贵。两阶段：

**Stage 1**: fine-tune CogVideoX，无 memory block
$$
\mathcal{L} = \mathbb{E}_{z^i, i, \epsilon} \left[ \| \epsilon_\theta(z^i, i | o_t, a_t, c) - \epsilon \|^2 \right] \quad (3)
$$

这里：
- $z^i$: video latents at diffusion step $i$（上标 $i \in [0, N]$ 表示 diffusion timestep）
- $i$: diffusion timestep
- $\epsilon$: 加的 Gaussian noise
- $\epsilon_\theta$: video model 预测的 noise
- $o_t, a_t, c$: observation, action, camera embedding

Stage 1 目的：
1. Adapt CogVideoX 到 embodied navigation domain
2. 让 action-conditioned control 起作用
3. 学会输出 RGB-D 而非 RGB

**Stage 2**: freeze DiT，只训 memory block
$$
\mathcal{L} = \mathbb{E}_{z^i, i, \epsilon} \left[ \| \epsilon_\theta(z^i, i | o_t, a_t, c, M) - \epsilon \|^2 \right] \quad (4)
$$

$M$ 是 3D feature map。这个 stage 让 model 学会利用 map 信息而不破坏原始生成能力——典型的 "additive adapter" 思路，类似 LoRA 但用的是 cross-attention block。

**训练细节**：12× H100 GPU, ~3 days, Adam optimizer, lr=1e-4, bf16, gradient clip norm=1.0, v-prediction [26], DDIM sampler 50 steps, frame stride 1-3 随机采样（让 model 学多种 camera speed）。

---

## 3. 应用

### 3.1 Model Predictive Control (Algorithm 1c)

给定 reward function $r(s)$，优化：
$$
a^* = \arg\max_a R(V) \quad (5)
$$

$V$ 是用 world model 生成的 video，$R(V) = \sum_t r(s_t)$ 是累计 reward。这里 $V$ 是 image observation 序列，所以 $r$ 是定义在 image 上的 reward（比如 "去到目标物体"）。

实现：cross-entropy method [9]
1. 初始化 action space 上 uniform distribution
2. 采样 $N=60$ 个 action chunks
3. 每个 chunk 生成 future video
4. 算 cost，保留 top $k=30\%$
5. 更新 distribution，迭代

实验结果（Fig 6a）：~350 iterations 后达到和 NoMaD ranking 相似的 SIM 分；~720 iterations 后 SIM=87.5，比 ranking action trajectories 提升 17%。

### 3.2 Policy Learning in New Environment

- 用 few-shot images 初始化 map $M$
- 反复：policy 生成 action → video model 生成 future obs → 更新 map $M$
- Hindsight relabeling [35] 给 trajectories 打 goal/reward label
- 用这些数据 fine-tune policy

这是 "world model as simulator" 范式：不需要真的在 environment 里 interact，纯用 generative rollout 训 policy。

Fig 6b 显示显著 boost policy learning。

### 3.3 Ranking Trajectories (Algorithm 1c + NoMaD)

用 NoMaD [29]（SOTA diffusion policy for navigation）sample $N=8$ 或 $16$ 个 action trajectories，用 world model 生成 future video，rank by LPIPS similarity between goal observation 和最后 frame。

Table 2 结果：

| Method | ATE↓ | RPE↓ | SIM↑ |
|---|---|---|---|
| NoMaD | 4.94 | 4.86 | 36.8 |
| NWM (×8) | 4.82 | 3.62 | 59.7 |
| Ours (×8) | 4.80 | 3.58 | 62.5 |
| NWM (×16) | 4.54 | 3.51 | 68.7 |
| **Ours (×16)** | **4.47** | **3.28** | **70.8** |

SIM 从 NoMaD 的 36.8 提升到 70.8，几乎翻倍。memory 让 ranking 更可靠，因为 model 知道 environment 长什么样，不会因为 unseen area 就 hallucinate 出无关内容。

---

## 4. 实验结果分析

### 4.1 Video Generation (Table 1)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ | DreamSim↓ | SRC↑ |
|---|---|---|---|---|---|---|
| NWM | 17.48 | 0.657 | 0.308 | 194.04 | 0.247 | 63.4 |
| Image Memory | 18.98 | 0.678 | 0.271 | 124.04 | 0.184 | 69.4 |
| Ours (w/o depth) | 20.63 | 0.696 | 0.218 | 114.42 | 0.118 | 77.8 |
| Ours (w/ 2D-Map) | 21.61 | 0.752 | 0.172 | 98.15 | 0.097 | 79.2 |
| **Ours** | **22.46** | **0.759** | **0.157** | **91.89** | **0.086** | **81.7** |

分析 ablation 链：
- **NWM → Image Memory**：FVD 194→124，SRC 63.4→69.4。加任何 memory 都显著提升。
- **Image Memory → Ours (w/ 2D-Map)**：FVD 124→98，SRC 69.4→79.2。从 2D snapshot 升级到 2D feature map，依然有效，因为 feature 表达更密集。
- **2D-Map → Ours (w/o depth) → Ours**：两个 axes 起作用——3D voxel（vs 2D map）和 depth-in-latent。3D voxel 把 SRC 从 79.2 推到 81.7（depth 加进来）；depth 单独（w/o depth 行 vs Ours 行）从 PSNR 20.63→22.46 贡献巨大。

SRC（Scene Revisit Consistency）是 Slow-Fast [17] 的指标，测 "agent 反向走回原位置时，看到的画面和之前一致吗"。这个 metric 直接对应 paper 的核心 claim——persistent memory。NWM 的 63.4 vs Ours 的 81.7 是 ~30% 相对提升。

FVD 91.89 vs NWM 194.04，几乎减半，是 visual quality + temporal consistency 的综合衡量。

### 4.2 Long Video Generation

Fig 5 展示 112 frames 的 autoregressive generation（5 次自回归，每次 9 frames × frame stride）。Memory 持续累积，不退化。这是关键 demonstration：**没有 memory 的 model 在 long rollout 时一定 drift，而带 memory 的可以稳态运行**。

### 4.3 Generalization to Unseen Scenes (Fig 7)

用 few-shot images init map，model 在没见过的 scene 里也能 persistent generate。这验证了 model 学到的是 "如何用 memory" 而不是 "记住训练 scene"。

---

## 5. Limitations 和未来方向

作者承认：
1. **需要 depth data**：真实数据集要么没 depth（RealEstate10K [44]），要么 trajectory 多样性差（ScanNet++ [39]、ARKitScenes [3]）。Table 4 对比了这些。建议用 Depth Anything [34] 估计 depth + 混合 sim/real data。
2. **3D map 不建模动态环境**：车流、人走动这些 dynamic content 没法 handle，因为 map 是 static snapshot。建议在 3D memory 之上加一个 dynamics model。

我觉得还有一个**未被讨论的 limitation**：3D map 的分辨率和 memory 增长。$256 \times 32 \times 256 \times 384$ 已经很大（~800M floats），更大 scene 怎么办？是否有压缩、submap、importance sampling 的可能？这其实是 SLAM 领域几十年的老问题（submap-based mapping, pose graph）。可以借鉴。

---

## 6. Build Intuition

我帮你梳理一下这 paper 的 "insight skeleton"：

### 6.1 为什么 2D latent 不够

video diffusion 的 2D latent space 本质是 "以相机为中心" 的表示，每帧一个 latent。当 agent 转身时，camera frame 完全变了，没有任何机制让 model 知道 "之前看到的沙发在背后哪里"。这就像一个人没有空间记忆，每次转头都以为是个新房间。

3D voxel memory 提供的是 "以世界为坐标系" 的 representation，每个 voxel 在 world frame 里有 fixed 位置。这就把 "记忆" 从 "怎么从相机坐标系变换" 这个难题，转化成了 "把 3D voxel features 拉到当前 view" 这个相对简单的几何查询问题。这是经典 SLAM/robotics 思想在 generative model 里的复现。

### 6.2 为什么 DINO feature 而不是 RGB voxel

如果 voxel 存原始 RGB，aggregation 没法做（max-pool RGB 没意义），且 semantic information 弱。DINO feature 是 self-supervised 出来的 dense semantic representation，max-pool 是合理的（同一个 voxel 的多个观察，取最显著的 semantic signal）。同时 DINO feature 和 CogVideoX 的 latent space 在某种程度 compatible（都是 transformer-based visual feature）。

### 6.3 为什么 Plücker embedding 而不是 raw pose

视频 model 的 input 是像素级，pixel 和 pose 矩阵的 entry 之间没有直接的 spatial correspondence。Plücker 把 pose 转成 "每像素一根 ray"，pixel-level 的 spatial alignment 直接对齐，conv/attention 可以自然处理。这是把相机几何"翻译"成 CNN/DiT 友好形式的经典 trick。

### 6.4 为什么两阶段训练

memory block 是新加的模块，初始随机初始化。如果同时训 DiT + memory block，random memory block 会给 DiT 噪声 gradient，破坏 pre-trained generative 能力。Stage 1 先让 DiT 学会 domain + action + depth；Stage 2 freeze DiT，让 memory block 学 "如何在已有生成能力上加 memory 条件"。这是 parameter-efficient fine-tuning 的 standard trick。

### 6.5 为什么 cross-attention 而不是 concat memory

memory 的 token 数量可变（取决于 explored region 大小），且 spatial layout 不规则。concat 进 latent 没法对齐 video frame 的 spatial grid。cross-attention 是 Perceiver/Flamingo 的标准做法：query 是 fixed-shape video latent，key/value 是可变长度 memory tokens。attention 自己学会 "我当前看这个方向时，需要 fetch 哪些 voxel 的信息"。

### 6.6 这个 paradigm 适合什么、不适合什么

**适合**：
- Indoor navigation（Habitat, HM3D）
- Static environment（家具不会动）
- Long horizon planning（revisit 同一位置）
- Few-shot adaptation to new scene

**不适合**（paper 没强调但应该注意）：
- Highly dynamic scene（自动驾驶里车流变化）—— InfiniCube [19] 那种用 HD map 的方案在 dynamic 上更弱
- Manipulation（物体被拿起后位置变化）—— 3D map 没有物体-level 的 affordance/dynamics
- Outdoor large-scale（256m × 256m 还行，更大就爆显存）
- Real-time inference（50 steps DDIM + memory cross-attention 很慢）

---

## 7. 相关联想

### 7.1 与 Sora、Genie 2 的对比

OpenAI 的 Sora [5] 据传内部有某种 "spatiotemporal patch" tokenizer，可能 implicit 学到 3D consistency，但没有 explicit memory，无法长 horizon。Genie 2（DeepMind 2024 末）也是 interactive world model，模型大、视觉好，但同样没 explicit memory，长 horizon 会有 drift。这篇 paper 的 3D memory 是 explicit、structured、可以 inspect 的，是和 foundation video model 不同的 paradigm：foundation model 走 scale，这篇走 structure。

### 7.2 与 3DGS-World-Model 的潜在融合

3D Gaussian Splatting 是 explicit 3D 表示，比 voxel grid 更 compact。如果用 3DGS 作为 memory（每个 gaussian 存 DINO feature），可能比 voxel 更 scalable。这是合理 future work 方向。

### 7.3 与 LLM-style KV cache 的类比

LLM 的 KV cache 是 "显式存储历史 attention 用的 key/value"，避免重复计算。这里的 3D memory block 类似——存储历史观察的 3D feature，generation 时通过 cross-attention retrieve。这是把 LLM 的 "memory mechanism" 思想迁移到 3D embodied world model，载体从 text token 变成 3D voxel token。

### 7.4 与 DreamerV3 的对比

DreamerV3 (Hafner et al.) 是 RSSM-based latent world model，state 是 latent vector，有 recurrent memory。但 latent space 是 abstract 的，没有显式 3D 结构。Dreamer 适合 control 但不适合 visual generation。这篇 paper 反过来——visual generation 强但 control 弱（需要 MPC 在外层 wrap）。

### 7.5 与 robotics foundation model 的关系

π0 (Physical Intelligence)、RT-2、Octo 这些 VLA model 都是 "policy = VLM"。如果用 world model 做 "look-ahead planner" 包在 VLA 外面，就像这篇 paper 的 ranking trajectories 实验——policy 提议多个 action，world model 模拟未来，reward 选最好的。这是 model-based RL 在 VLA 时代的复兴。Yilun Du 之前很多工作都在这条线上（UniPi [11], Video Language Planning [10], RoboDreamer [43], Adaptive Online Replanning [42]）。

### 7.6 与 4D Embodied World Model 的关系

Zhen et al. [41] (Learning 4D Embodied World Models) 是同组 prior work，做 4D（3D space + time）embodied world model。本 paper 的 RGB-D generation 部分借鉴了它。但 Zhen 那篇没显式 persistent memory，所以长 horizon 同样 drift。本 paper 可以看作 [41] 的 "memory-augmented" 升级版。

---

## 8. 总结性直觉

这篇 paper 的核心 intellectual move 是：**把 embodied world model 从 "以相机为中心的 2D latent sequence" 升级为 "以世界为坐标系的 3D voxel memory + 2D view rendering"**。

它告诉我们一个 deep lesson：**生成模型在做 long-horizon simulation 时，必须有 explicit 的、可以增量更新的、和世界坐标系对齐的 memory，不能指望 2D latent implicit 学到 3D persistence**。这和 LLM 需要 KV cache、SLAM 需要 occupancy map、人脑需要 hippocampal place cells 是同一个道理——persistence 需要专门的 substrate。

未来 5 年 embodied world model 的发展，我认为就是这条路线的扩展：3D memory → 4D memory（动态物体）→ multi-agent memory（其他 agent 的轨迹）→ semantic+geometric joint memory（scene graph + voxel）→ real-time differentiable memory（可以 gradient-based planning）。这篇 paper 是这条路线的早期 milestone。

---

## 参考链接

- Paper arXiv: https://arxiv.org/abs/2501.01890
- 项目页: https://3d-persistent-world-model.github.io
- Yilun Du group: https://yilundu.github.io
- CogVideoX: https://arxiv.org/abs/2408.06072
- DINO-v2: https://arxiv.org/abs/2304.07193
- NWM (Navigation World Models): https://arxiv.org/abs/2406.09494
- SlowFast-VGen: https://arxiv.org/abs/2410.23277
- 3D-Mem: https://arxiv.org/abs/2408.09685
- CameraCtrl (Plücker): https://arxiv.org/abs/2404.02101
- Light Field Networks (Plücker origin): https://arxiv.org/abs/2106.10634
- Habitat: https://arxiv.org/abs/2106.14405
- HM3D: https://arxiv.org/abs/2109.08238
- NoMaD: https://arxiv.org/abs/2310.07896
- UniPi: https://arxiv.org/abs/2302.09545
- Uni-Sim: https://arxiv.org/abs/2310.06114
- RoboDreamer: https://arxiv.org/abs/2404.12377
- Learning 4D Embodied World Models: https://arxiv.org/abs/2502.09491 (Zhen et al.)
- Genie: https://arxiv.org/abs/2402.15391
- InfiniCube: https://arxiv.org/abs/2412.03934
- Persistent Nature: https://persistentnature.github.io
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- ARKitScenes: https://github.com/apple/ARKitScenes
- ScanNet++: https://markus-ye.github.io/scannet_pp/
- DreamSim: https://dreamsim-nature.github.io
- DDIM: https://arxiv.org/abs/2010.02502
- V-prediction (Progressive Distillation): https://arxiv.org/abs/2202.00512
- Cross-Entropy Method tutorial: https://link.springer.com/article/10.1007/s10479-005-5723-z
- Sora (technical report): https://openai.com/research/video-generation-models-as-world-simulators
- NeRF: https://arxiv.org/abs/2003.08934
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.14737
- DreamerV3: https://arxiv.org/abs/2301.04104
- Flamingo (cross-attention memory): https://arxiv.org/abs/2204.14198
- Perceiver: https://arxiv.org/abs/2103.03206

如果你想深入聊任何一块（比如 Plücker embedding 的几何推导、3D map aggregation 的实现细节、MPC cross-entropy method 的调参、或者怎么把这个 paradigm 推广到 manipulation），可以指方向，我再展开讲。
