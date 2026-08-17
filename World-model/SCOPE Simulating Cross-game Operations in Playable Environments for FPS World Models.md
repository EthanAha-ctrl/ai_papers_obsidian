---
source_pdf: SCOPE Simulating Cross-game Operations in Playable Environments for FPS
  World Models.pdf
paper_sha256: 916d7e9dca18659c409b524681abe406096e7119e2537813314505cf0ae87aea
processed_at: '2026-08-12T04:14:51-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲一遍 SCOPE

## 这篇 paper 在解决什么问题

你想用 video diffusion model 当游戏引擎——给它一个画面，再给它玩家的按键操作，它生成接下来的画面。

这个事在 Minecraft、DOOM、Atari 上已经有人做出来了。但一到 FPS 就崩。

为什么崩？因为 FPS 的操作太密了。打 FPS 的时候，你一秒钟能做一堆事：mouse 甩 180 度、同时按 fire、还在走路、中间还 reload 一下。这种高频密集的 action signal，现有的 world model 处理不了。

现有模型怎么接 action？把 action 编成一个向量，用 AdaLN 或者 cross-attention 广播到整个画面所有 pixel。听起来 OK，但有个致命问题：你按 fire，本意只是枪口附近亮一下 muzzle flash，结果模型把"fire"这个信号打到了画面每一个 pixel 上——天空在 fire，远处的墙在 fire，地上的草也在 fire。然后下一帧又 fire，下下帧 reload，每一帧整个画面都被扰动，几帧之后画面就糊了。

这就是为什么 baseline 在 FPS 场景下要么画面基本不动（suppress 太狠），要么动一下就乱。

## 作者的 key insight

作者观察到一件事：FPS 里的 action 其实是分区域的。

你开火，只有枪口和准星附近那块区域有变化。你 reload，只有手和枪在动。这些叫 **discrete event**，影响的是一个 localized 的 "scope"。

但你转视角、走路，整个画面的场景都在流动——墙、天空、远处的山都在做 scene flow。这叫 **continuous control**，影响的是 scope 外面的区域。

所以天然有一个分解：
- scope 里面：discrete event，局部动画，需要精确控制
- scope 外面：continuous control，整体场景流动，需要稳定

两边的需求完全不同。你 fire 的时候不应该 perturb 天空，你转视角的时候也不应该把枪口弄乱。

问题是：怎么让模型自己知道哪个 pixel 是 in-scope、哪个是 out-of-scope？又没有 segmentation label 标出"这是枪"。

## SCOPE 的做法

核心 idea 很简洁：**让每个 pixel 根据自己的 visual content 决定要不要响应这个 action**。

具体实现分两步。

### 第一步：reshape

正常 video DiT 里，token 是按"所有 frame 的所有 spatial position"排成一个长序列，attention 是在 spatial 上跑的。SCOPE 在每个 transformer block 里加一个 module，先把 token reshape 成"每个 spatial position 持有自己的一条 temporal 序列"。

什么意思？原来 attention 是"这个 frame 的这个 pixel 关注同 frame 的其他 pixel"。reshape 之后变成"这个 spatial 位置的第 1 帧关注自己的第 2 帧、第 3 帧……"。

这一下就把 attention 的轴从空间转到了时间。每个 pixel 独立地、跨时间地处理自己那条 action 序列。

### 第二步：两条 pathway

**Discrete action 走 cross-attention**：query 是 visual feature（这个 pixel 长什么样），key 和 value 是 action embedding（fire、reload 这些）。

这里有个特别优雅的直觉。如果这个 pixel 是枪口附近，它的 visual feature 在训练中会和 fire action 学到强 alignment，attention 权重就大，fire 就在这里生效。如果这个 pixel 是天空，它的 visual feature 和 fire 毫无关系，attention 几乎为零，fire 就不在这里生效。

**分离是从 visual content 自然 emerge 的，不需要任何 segmentation label**。这是整篇 paper 最聪明的地方。

**Continuous action 走 temporal self-attention**：把 camera/movement 信号和 pixel feature 拼起来，在时间轴上做 self-attention，加 RoPE 给时间顺序信息。因为 discrete pathway 已经处理了 in-scope 动态，continuous pathway 专注稳定的整体场景流动，不会被 fire 这种局部事件污染。

两条 pathway 的输出加回原 feature，过 FFN，进下一个 block。

### 一个工程细节

所有新加的 output projection 是 zero-init 的。意思是训练开始时，SCOPE module 输出严格为零，模型就等于原始 Wan2.2，生成质量不受影响。然后随着训练逐步学到 action conditioning。这保证了稳定性——你不会因为加了一堆新 module 把预训练好的 video model 破坏掉。

## 训练数据：CrossFPS

现有 game world model 都是单游戏训练的。但 FPS 游戏其实共享很多 visual-action 物理：开火都有 muzzle flash，右转视角场景都往左流。作者想证明，只要数据 curated 好，模型能学到通用 mapping，泛化到没见过的游戏。

所以做了 CrossFPS：69K 个 5 秒 clip，来自 7 个 FPS 游戏，每个 clip 配 frame-aligned 的 10-DoF 手柄信号（4 个连续轴：左右/前后移动、左右/上下视角；6 个离散按钮：开火、ADS、reload、jump、melee、switch）。

但光收集数据不够，还有三个 curation 步骤很关键：

**1. 平衡 action 分布**。原始 gameplay 大部分时间在 idle 或直线走，高强度片段是 long-tail。先过滤掉 idle，再对 top 15% 高强度 clip（180 度甩枪、连续跳）oversample 3 倍。否则模型只会生成"缓慢散步"的视频。

**2. 去掉玩家策略 bias**。这是最巧的一步。高水平玩家有固定 pattern——总是对着敌人开火、总是躲掩体。如果模型学这些，它学的是"游戏策略"而不是"action 对应什么视觉变化"。

怎么去？用预训练 scene classifier 提 visual feature，算它和 action sequence 的互信息。**保留互信息最低的 20% clip**——对着空地开火、对着墙跑这种"没效率"的操作。强制模型学：不管这操作有没有意义，fire 就是会产生 muzzle flash。

这把 action entropy 从 1.85 bit 推到 2.94 bit，接近 10-DoF 空间理论最大值。

**3. 跨游戏动力学归一化**。不同游戏引擎，同样的摇杆位移对应的视角转速差很多——Halo 里推一下转 10 度，CoD 里转 30 度。多游戏联合训练时这会导致 gradient conflict。

做法是用 optical flow 估计每个 clip 的实际 pixel 位移，fit 一个 gain model，然后 rescale 让所有游戏的 action-to-pixel-displacement ratio 统一。归一化后 inter-game gain variance 从 0.8+ 降到 0.034。

## 结果怎么样

### 量化

8 个 metric，SCOPE 在 7 个上 SOTA。

最有说服力的是 **Photometric Smoothness 0.198**，比第二名 LingBot 的 0.626 好 3.2 倍，比 HY-World 的 2.523 好 12.7 倍。这个 metric 衡量相邻帧之间 pixel 级颜色一致性——数值低意味着画面稳定不抖。

这说明 scope separation 真的 work：你 fire 的时候，out-of-scope 的天空和墙没有跟着 fire 一起抖。

同时 Dynamic Degree 0.910 和 Flow Score 18.24 都是最高——动作响应强。动作强 + 画面稳，这两个同时领先才是 FPS 需要的。baseline 要么动作强但画面乱（HY-World），要么画面稳但动作弱（Matrix-Game 靠 suppress action 换 smoothness）。

唯一输的是 Motion Smoothness，Matrix-Game 第一。但 Matrix-Game 是靠压 action 响应换来的，它的 Dynamic Degree 只有 0.661，这是预期内的 trade-off。

### Ablation

每个 ablation 揭示一个机制：

- **去掉 spatial selectivity**（换成 global injection，但 input 仍用 native telemetry）：Photo. 0.198→0.745，崩了。这证明性能提升来自 architecture，不是来自 input modality 差异（baseline 用 Gemini 把 action 翻译成 text 是有 bottleneck 的，但这个 ablation 用原生 telemetry 也崩，说明问题在 global conditioning 本身）。

- **去掉 temporal self-attention**：Flow Score 18.24→11.60，continuous control 崩了。说明专门的时间建模对 camera/movement 必不可少。

- **去掉 discrete cross-attention**：Photo. 0.198→0.234，小幅恶化，fire effect 漏到 out-of-scope。但 Dynamic Degree 保持 0.846，说明动作还在，只是不 confined 了。这证实 visual querying 的空间约束作用。

- **训练策略**：Frozen backbone → Two-stage → End-to-end，FVD 单调下降 775→732→690。End-to-end 让 backbone 和 SCOPE module 深度协同适应。但 Frozen variant 的 Photo. 0.264 仍远优于"去掉 spatial selectivity"的 0.745——说明 per-pixel conditioning 设计本身（不需要 backbone 适应）就驱动 out-of-scope 稳定性。这给了 plug-and-play adapter 的可能性。

### Zero-shot 泛化

这是最有意思的实验。用 GPT-image-2 合成 4 种训练集里没有的风格的初始帧：stylized open-world、cooperative adventure、mythological action、sci-fi corridor。然后给 action，看模型能不能生成对的视频。

视觉质量：JEPA 0.777 vs in-distribution 0.806，退化很小。Photo. Smoothness 跨所有风格都低于 0.251，说明 scope separation 泛化到新 visual domain——模型学到的是"weapon-like appearance"而不是具体某把枪。

动作可控性（50 clip per task）：
- Single action：SCOPE 92% vs LingBot 78%
- Multi-action composition：75% vs 29%
- Action-environment interaction：54% vs 21%

复杂度越高，SCOPE 优势越大。这完全符合预期——global conditioning 在简单任务上还能竞争，一旦 action 组合复杂就彻底崩。Environment interaction 62% > object deformation 46%，反映 diffusion backbone 擅长 texture 不擅长 geometry。

### Scalability

1K → 5K → 10K → 30K → 65K，FVD 不是单调下降。1K 单游戏最好（478），10K 跨系列最差（1018），65K 恢复（690）。

解释：10K 时跨系列游戏 asset 差异大，gradient conflict 严重，需要 progressive warm-up。30K 开始多源 variety 提供自然 regularization。65K 多样性最大，single-stage 反超 progressive。

这给了一个 scaling law 的雏形：**小规模同质 single-stage，中等规模跨域要 progressive，大规模多源 single-stage 最好**。

## Limitation

作者诚实地讲了几点：

- 复杂 in-scope 行为还做不好——multi-step weapon mechanic、物品使用、精细物体操作。训练数据 interaction diversity 不够。
- 几何变换弱于外观——fire/smoke/lighting 处理好，structural deformation/physics 反应弱。这是 diffusion backbone 的 texture bias。
- 退化初始帧（极端模糊）会让生成回归训练集平均 appearance。
- Long-horizon 未解决——目前是 single-clip 生成，跨 clip 状态一致性是下一步。

## 我的整体判断

**最 sharp 的点**：观察到 FPS action 的 spatial selectivity。这个现象被所有 prior work 忽视，作者把它显式化并设计了匹配的 architecture。Visually-queried cross-attention 是干净可解释的——你能在脑子里看见为什么枪口附近的 pixel attend 强、天空的 pixel attend 弱。

**和 ControlNet 的类比**：ControlNet 把 text-to-image 的 condition 从 global prompt 推到 spatial control map。SCOPE 把 game world model 的 action condition 从 global injection 推到 per-pixel visual-content-gated injection。这是 conditioning mechanism 的 architectural shift，方向一致。

**可追问的点**：

1. 为什么不直接用 mask？既然 in-scope 大致是枪口+准星，理论上可以 heuristic 或 segmentation 强制分离。Paper 暗示这会 game-specific 不泛化，但 zero-shot 实验证明 visual query 学到的是 "weapon-like appearance" 而非具体武器——这本身是 generalizable 的。加少量 mask 监督会不会加速收敛？没 ablation。

2. Spatial reshape 的代价。把 attention 轴从空间转到时间，是对 backbone 预训练 representation 的某种 misuse。Frozen variant 性能下降证实了这点。可能需要 spatial-temporal hybrid attention。

3. Flow Score vs Motion Smoothness 的张力。SCOPE 同时拿到大运动 + 高 photometric stability，但 motion smoothness 输给 Matrix-Game。Action-CFG 的 λ 可能可以在 smoothness 上 tune，但 paper 没探这个 trade-off 曲线。

4. Cross-game 数据 scaling 的 sweet spot。10K 跨系列崩溃提示：多游戏训练不是单纯加数据，需要 diversity vs interference 的平衡。类似 LLM 多语言训练——少量语言 OK，多语言要 curriculum。Paper 的 single-stage on 65K 是 empirical 选择，缺理论分析。

5. FPS 之外的泛化。Per-pixel conditioning 对任何 egocentric interactive scene 都该有用——机器人操作（gripper 影响 object 局部）、驾驶（steering 影响 scene flow）、AR/VR。Paper 提了没实验，明显的下一步。

**对 world model 范式的影响**：把"video diffusion as world model"从"global action injection"推到"spatially-selective action injection"。69K 数据训练 5B 模型 18 小时就 zero-shot 泛化，证明 architecture 的 inductive bias 比 data volume 重要。这条路值得 follow。

参考：
- [Wan2.2 模型](https://github.com/Wan-Video/Wan2.2)
- [Flow Matching (Lipman et al.)](https://arxiv.org/abs/2210.02747)
- [DiT (Peebles & Xie)](https://arxiv.org/abs/2212.09748)
- [V-JEPA](https://arxiv.org/abs/2301.07843)
- [ControlNet](https://arxiv.org/abs/2302.05543)
- [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model)
- [Oasis (Decart)](https://oasis-model.github.io/)
- [GameGen-X](https://arxiv.org/abs/2411.00769)
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)
- [GameGAN](https://arxiv.org/abs/2005.12326)

---

# SCOPE: 为 FPS world model 设计的 per-pixel action conditioning

## 1. 核心问题的精确定义

这篇 paper 抓住了 video diffusion 当作 world model 时一个特别 sharp 的 failure mode。在 Atari / Minecraft / DOOM 这种 task 里，action signal 是稀疏的（按一下键、走一步），global conditioning（一个 embedding broadcast 到所有 spatial position）就够了。但 FPS 游戏 action 的密度是另一个量级：

- Camera sweep 超过 180°/s，相当于每帧 ~9° 的视角变化
- 在一个 generation window（5 秒、100 帧）内可能 fire + move + reload 同时发生
- Discrete event（开火、换弹）和 continuous control（camera、movement）在同一个 frame 交织

Global conditioning 在这种 regime 下崩溃的机制是：**一个 fire 命令本意只影响 weapon 周围几个 pixel，但 AdaLN 或 global cross-attention 会把它 broadcast 到所有 H×W 个 position**。开火时整张图都被 perturb，连续几个 fire 事件后 distortion 累积，frame 结构就崩了。这就是为什么 HY-World 1.5 在测试集上 Dynamic Degree 只有 0.225（基本是 static output）——它的 global normalization 把 dense FPS signal dilute 到 effective threshold 以下。

## 2. 核心 insight: FPS action 的 spatial selectivity

观察 FPS gameplay 的一个 key 性质：action 的 visual effect 是 spatially localized 的。

- **In-scope region**: weapon 周围、crosshair 附近、muzzle flash 出现的位置——discrete event（fire, reload, ADS, melee）只在这里产生 animation
- **Out-of-scope region**: 墙壁、天空、远处环境——受 continuous camera/movement 控制，需要 smooth scene flow 但不能被 fire 这种 local event 污染

这给了一个自然的 decomposition：

| Region | Action type | 适合的建模方式 |
|---|---|---|
| in-scope | discrete event | confined spatial context 内学 action→visual 对应 |
| out-of-scope | continuous ego-motion | 排除 in-scope 动态后的稳定生成 |

两边都需要同一个 primitive：**per-pixel conditioning，让每个 position 从 local visual content 决定自己是 in-scope 还是 out-of-scope**。这是无监督的——不需要 segmentation label，分离会从 visual content 自然 emerge。

## 3. Architecture 深度解析

### 3.1 Backbone

Base model 是 **Wan2.2-TI2V-5B**（[Wan repo](https://github.com/Wan-Video/Wan2.2)），5B 参数 video DiT：

- 30 transformer layer
- hidden dim D = 3072
- 24 attention head
- patch size [1, 2, 2]（temporal 不 patch，spatial 2×2）
- FFN dim 14336
- 3D VAE: spatial compression 8×, temporal compression 4×，所以输入 video V ∈ R^{3×T×H×W} 编码到 latent z ∈ R^{C×f×h×w}，其中 f = T/4, h = H/8, w = W/8
- Text encoder UMT5-XXL，4096-dim embedding
- 3D RoPE for positional encoding

Token 序列 x ∈ R^{B×N×D}, N = f × h × w。在 480×832、81 frame 的设置下，f = 81/4 ≈ 20, h = 480/8 = 60, w = 832/8 = 104，N ≈ 124800 token。这个 token 数量是 attention 计算的关键。

### 3.2 Flow matching 训练目标

不直接用 DDPM，而是用 flow matching（[Lipman et al. 2022](https://arxiv.org/abs/2210.02747)）。给定 clean latent z_0 和 Gaussian noise ε ~ N(0, I)，noisy latent 构造为线性插值：

$$z_t = (1-t) z_0 + t \epsilon, \quad t \in [0, 1]$$

模型学 velocity field v_θ(z_t, t, c)，loss 为：

$$\mathcal{L} = \mathbb{E}_{t, z_0, \epsilon}\left[ w(t) \left\| v_\theta(z_t, t, c) - (\epsilon - z_0) \right\|^2 \right]$$

变量解释：
- t: flow matching timestep，0 是 clean data 端，1 是 pure noise 端
- z_0: clean video latent
- ε: 一次性采样的 Gaussian noise
- c: conditioning（text、first frame，扩展后包含 action）
- w(t): timestep-dependent weight，给中间 timestep 更高权重（中间 step 学起来最难）
- v_θ: 神经网络预测的 velocity，目标是 (ε - z_0)，即从 clean 到 noise 的方向

**Image-to-video paradigm**: first frame latent 替换掉 noisy latent 的第一个 temporal position，loss 只算后续 frame。这是关键——给一个 anchor 让模型知道 scene 从哪开始。

### 3.3 SCOPE module 的位置和结构

SCOPE 插在 **text cross-attention 和 FFN 之间**，每个 block 一个，共 30 个。

#### Spatial reshape 的精确数学

标准 token layout x ∈ R^{B × (f·h·w) × D}。SCOPE 第一步是 reshape 到 per-pixel temporal sequence：

$$x \in \mathbb{R}^{B \times (f \cdot h \cdot w) \times D} \longrightarrow \hat{x} \in \mathbb{R}^{(B \cdot h \cdot w) \times f \times D}$$

变量含义：
- B: batch size
- f: temporal token 数（latent frame 数）
- h, w: spatial latent dimension
- D: hidden dim
- reshape 后，每个 spatial position (h_i, w_j) 持有长度 f 的 temporal sequence

这个 reshape 是 spatial selectivity 的物理基础——它把 attention 的"轴"从 spatial 转到 temporal，让每个 pixel 独立地、跨时间地处理自己的 action response。

#### Action representation

10-DoF controller signal 分两类：

| 类别 | 维度 | 具体 axis |
|---|---|---|
| Continuous a_c ∈ R^{T_raw × 4} | d_c = 4 | LX, LY (movement), RX, RY (camera) |
| Discrete a_d ∈ R^{T_raw × 6} | d_d = 6 | Fire (RT), ADS (LT), Melee (R3), Jump (A), Reload (X), Switch (Y) |

T_raw 是原始 gameplay frame 数（20fps × 5s = 100 frame），latent frame f = T_raw / 4 = 25。

#### Discrete pathway: visually-queried cross-attention

$$\Delta x_d = \text{CrossAttn}(Q = \hat{x}, K = V = \text{MLP}_{\text{embed}}(a_d))$$

变量：
- Q (query): 来自 per-pixel visual feature x̂，所以 query 是"这个 pixel 的 visual content"
- K, V (key, value): 来自 discrete action 的 MLP embedding，所以是"action 信号"
- 输出 Δx_d: per-pixel discrete action residual

关键直觉：query 来自 local visual content。如果某个 pixel 是 muzzle flash 应该出现的位置（weapon tip、crosshair 附近），它的 visual feature 与 fire action embedding 在训练中学到了强 alignment，attention weight 大；如果 pixel 是天空或远处墙壁，visual feature 与 fire action 无关，attention 趋近零。**分离 emerge 自 visual content 本身，不需要任何 segmentation label**——这是 paper 最优雅的设计。

#### Continuous pathway: temporal self-attention with RoPE

对每个 latent frame i，提取 raw-frame action 的 temporal window：

$$w_i = a_c[i \cdot r : i \cdot r + r \cdot s]$$

变量：
- r = 4: temporal compression ratio（一个 latent frame 对应 4 个 raw frame）
- s: window size（看前后的 raw action）
- w_i: 长度 r·s 的 action window，覆盖 latent frame i 周围的连续控制信号

然后 flatten + concat + MLP fuse + temporal self-attention：

$$\tilde{x} = \text{MLP}_{\text{fuse}}([\hat{x}; \text{flatten}(w)]), \quad \Delta x_c = \text{SelfAttn}(\tilde{x}, \text{RoPE}_t)$$

变量：
- [·; ·]: concatenation
- RoPE_t: temporal axis 上的 rotary position embedding，给 self-attention 提供时间顺序信息
- 输出 Δx_c: per-pixel continuous action residual

直觉：continuous control 影响所有 pixel 的 scene flow（camera 转，所有东西都流），所以用 self-attention（每个 pixel 都参与），但通过 RoPE 强制时间平滑。discrete pathway 已经处理 in-scope 动态，continuous pathway 不被 local effect 污染，专注 stable ego-motion。

#### Residual 组合

$$(\hat{x} + \Delta x_c + \Delta x_d) \to \text{reshape 回 token layout} \to \text{FFN}$$

所有 output projection **zero-initialized**——训练 step 0 时 module 输出严格为零，模型退化为原始 Wan2.2。然后逐步学到 action conditioning。这保证训练稳定性，并让 backbone 与 action pathway co-adapt。

## 4. Action-CFG: 可调的 action 强度

训练时 stochastic action dropout：以概率 p_drop = 0.1，所有 action 替换成 learnable null embedding a_null。推理时做 classifier-free guidance：

$$\hat{v} = v_\theta(z_t, a_{\text{null}}) + \lambda \left[ v_\theta(z_t, a_c, a_d) - v_\theta(z_t, a_{\text{null}}) \right]$$

变量：
- v_θ(z_t, a_null): unconditional velocity prediction（"没有 action 时模型想生成的"）
- v_θ(z_t, a_c, a_d): conditional velocity（"给定 action 时想生成的"）
- λ: guidance scale，控制 action 强度
  - λ = 1: 标准 conditioning
  - λ > 1: 放大 action 响应（更夸张的 muzzle flash、更快的 camera turn）
  - λ < 1: 衰减 action 响应（更 subtle）

这与 standard CFG（[Ho & Salimans 2022](https://arxiv.org/abs/2207.12598)）思想一致，但作用在 action 而非 text 上。代价是每个 denoising step 要跑两次 forward pass。

## 5. CrossFPS dataset 的 curation

69K 5-second clip, 7 个 FPS title（Halo Infinite, Xonotic, CoD: MW, Halo MCC, CoD: Warzone, CoD: MW3, CoD），来自 NitroGen（[Magne et al. 2026](https://arxiv.org/abs/2601.02427)）和 WorldCam（[Nam et al. 2026](https://arxiv.org/abs/2603.16871)）。

三个 curation stage 是这份 dataset 真正有贡献的地方：

### 5.1 Action Distribution Balancing

原始 gameplay 是 long-tail——大部分时间在直线走或 idle。先做 activity filter（left stick active ≥ 70%）扔掉 idle clip，再对 top 15% 高强度 clip（180° flick、连续 jump）oversample 3×。否则模型会 collapse 到只生成 smooth low-motion sequence。

### 5.2 Visual-Action De-biasing

最聪明的一步。skilled player 有 stereotyped pattern——总是对着 highlighted enemy 开火、总是躲在 cover 后。如果模型学这些，它学的是 game strategy 而不是物理 action→visual mapping。

方法：用预训练 scene classifier 提 visual feature，计算 visual feature 与 discrete action sequence 的 mutual information。**保留 bottom 20% 互信息最低的 clip**（fire at empty sky、sprint into wall）作为 "de-biased" sample 强制加入训练集。

效果：action entropy 从 1.85 bit 升到 2.94 bit，接近 10-DoF 离散空间的理论最大值。这意味着模型不能靠 temporal prior 偷懒，必须学真实物理对应。

### 5.3 Kinetic Normalization

不同 engine 的 stick→rotation gain 差异巨大（Halo 1° per unit vs CoD 3° per unit）。多游戏联合训练时这会导致 gradient conflict。方法：对每个 clip，提取 camera rotation 引起的 mean pixel displacement (Δu, Δv)，fit linear gain model Δu ≈ g_x · RX，然后 rescale：

$$RX_{\text{norm}} = RX \cdot (\bar{g}_x / g_x)$$

变量：
- g_x: 该 clip 的实测 gain
- ḡ_x: dataset-wide mean gain
- RX_norm: normalize 后的 camera input

Normalization 后 inter-game gain variance 从 >0.8 降到 0.034，optical flow-action correlation 达到 r = 0.91 ± 0.03。这是为什么模型能在多游戏上学到通用 mapping。

## 6. 实验结果深度分析

### 6.1 主表（Table 1）

8 个 metric，SCOPE 在 7 个上 SOTA：

| Metric | SCOPE | Best baseline (LingBot) | 差距 |
|---|---|---|---|
| JEPA↑ | 0.806 | 0.615 | +31% |
| FVD↓ | 690.3 | 954.4 | -28% |
| LPIPS↓ | 0.601 | 0.611 | -1.6% |
| Dynamic Degree↑ | 0.910 | 0.868 | +4.8% |
| Flow Score↑ | 18.24 | 15.50 | +17.7% |
| Photometric Smoothness↓ | 0.198 | 0.626 | **-68% (3.2×)** |
| Depth↓ | 1.299 | 1.454 | -10.7% |

唯一输的是 Motion Smoothness（Matrix-Game 3.0 第一 2.502 vs SCOPE 2.383）。但 paper 指出 Matrix-Game 是靠 **suppressing action response** 换 smoothness——它的 Dynamic Degree 只有 0.661，trade-off 是预期的。HY-World 1.5 最极端：Dyn.Deg. 0.225（基本 static），Photo. 2.523（极其不稳定，矛盾吗？不，是结构 artifact 严重）。

**Photometric Smoothness 0.198** 是最重要的数字。它衡量 adjacent frame 间 pixel-level color consistency（用 depth + flow backward warp）。0.198 比 LingBot 好 3.2×，比 HY-World 好 12.7×——这是 scope separation 直接证据：out-of-scope region 真的稳定，没有 in-scope event 的污染。

### 6.2 Ablation（Table 2）

每个 ablation 都揭示一个机制：

**w/o Spatial Selectivity**: 把 per-pixel conditioning 换回 global injection（但保留 native telemetry input）。结果 Photo. 0.198→0.745（3.8× 恶化），Dyn.Deg. 0.910→0.521。这证明 contribution 是 architectural 的，**不是 input modality 驱动的**——这是对 baseline 比较公平性的关键回应（baseline 用 Gemini text translation 接 action 是有 information bottleneck 的，但这个 ablation 用 native telemetry 都崩，说明问题不在 input）。

**w/o Temporal Self-Attn**: 去掉 continuous pathway 的 self-attention。Flow Score 18.24→11.60（-36%）。证实 dedicated temporal modeling 对 continuous control 必不可少。

**w/o Discrete Cross-Attn**: 去掉 discrete pathway 的 cross-attention。Photo. 0.198→0.234（小幅恶化，effects leak 到 out-of-scope），但 Dyn.Deg. 保持 0.846。证实 visual querying 的 spatial confinement 作用——没有它，fire effect 会 leak；有它，effect 精确 confined。

**w/o Action-CFG**: Dyn.Deg. 0.910→0.820, Flow 18.24→15.90。CFG 防止 regression-to-mean。

**Training strategy**:
- Frozen backbone: FVD 775.4
- Two-stage (FT → freeze): FVD 732.1
- End-to-end: FVD 690.3

End-to-end 让 backbone 与 SCOPE module 深 co-adapt，Flow Score 单调上升 15.57 → 17.13 → 18.24。值得注意的是，**Frozen variant 的 Photo. Smoothness 0.264 仍远优于 "w/o Spatial Selectivity" 的 0.745**——证明 per-pixel conditioning 设计本身（独立于 backbone adaptation）就驱动 out-of-scope stability。这给了 plug-and-play adapter 的可能性。

### 6.3 Zero-shot generalization（Table 3, 4）

用 **GPT-image-2**（[OpenAI 2026](https://openai.com/index/introducing-chatgpt-images-2-0/)）合成 first-person frame，覆盖 4 个训练集没有的 aesthetic：stylized open-world, cooperative adventure, mythological action, sci-fi corridor。

视觉质量（Table 3）：JEPA 0.777 vs in-distribution 0.806，Photo. 0.231 vs 0.198——modest 退化。Sci-fi corridor（结构上最像 FPS）几乎 parity。Photo. ≤ 0.251 跨所有类别，证明 scope separation 泛化到新 visual domain。

Action controllability（Table 4，50 clip per task）：

| Method | Single Action | Multi-Action Comp | Action-Env Interaction | Average |
|---|---|---|---|---|
| Matrix-Game 3.0 | 0% | 1.3% | 0% | 0.5% |
| HY-World 1.5 | 8% | 13.3% | 2.7% | 8.0% |
| LingBot-World | 78% | 28.7% | 21.3% | 38.3% |
| **SCOPE** | **92%** | **75.3%** | **54%** | **71.5%** |

SCOPE 比 LingBot 高 1.9×。Gap 随 complexity 扩大：single 92 vs 78，composition 75 vs 29，interaction 54 vs 21。这显示 global conditioning 在简单 task 上还能竞争，复杂 composition 下彻底崩溃。Environment interaction 62% > object deformation 46%，反映 diffusion backbone 在 texture 上强、geometry 上弱。

### 6.4 Scalability（Table 9, 10）

5 个 scale：1K → 5K → 10K → 30K → 65K

| Scale | Titles | Series | FVD↓ |
|---|---|---|---|
| 1K | 1 | 1 | 478.20 |
| 5K | 2 | 1 | 603.91 |
| 10K | 3 | 2 | 1017.82 |
| 30K | 6 | 3 | 799.70 |
| 65K | 7 | 3 | 690.30 |

非单调！1K 最好，10K 最差。解释：1K 单游戏同质，5K 同系列还 OK，**10K 跨系列时 visually distinct game asset 创造 conflicting gradient**，需要 progressive warm-up 才稳。30K 多源 variety 提供自然 regularization，65K 全数据 + maximum diversity 时 single-stage 反超 progressive（FVD 690.30 vs 756.28）。

这给了一个 scaling law：**小规模同质用 single-stage，中等规模跨域用 progressive，大规模多源 single-stage 最好**。最终设计选 single-stage on full 65K。

## 7. Limitation 和未来方向

诚实指出的局限：

- **复杂 in-scope 行为**：multi-step weapon mechanic、item usage、fine-grained object manipulation 仍难。训练数据 interaction diversity 不够。
- **几何变换弱于外观**：fire/smoke/lighting 处理好，structural deformation/physics-driven reaction 弱。这是 diffusion backbone 的 texture bias——V-JEPA 这种 JEPA-style 训练可能更适合几何，参见 [V-JEPA](https://arxiv.org/abs/2301.07843)。
- **退化 first frame**：极端模糊的初始 frame 会回归训练平均 appearance。
- **Long-horizon**: 目前是 single-clip 生成，跨 clip 状态一致性未解决。WorldMem（[Xiao et al. 2025](https://arxiv.org/abs/2505.xxxxx)）的 memory retrieval 思路可能是答案。

## 8. 与 related work 的定位

Game world model 谱系：
- **GAN-based**: GameGAN（[Kim et al. 2020](https://arxiv.org/abs/2005.12326)）早期有限生成能力
- **Diffusion + global cond**: GameGen-X（[Che et al. 2024](https://arxiv.org/abs/2411.00769)）、Oasis（[Decart 2024](https://oasis-model.github.io/)）、GameCraft-2（[Tang et al. 2025](https://arxiv.org/abs/2511.23429)）——单游戏、sparse action、global injection
- **Scale-oriented cross-game**: Genie 2（[Parker-Holder et al. 2024](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model)）、Genie 3（[Ball et al. 2025](https://arxiv.org/abs/2507.xxxxx)）——需要海量 proprietary data
- **SCOPE**: compact 69K dataset + architectural innovation（per-pixel conditioning）实现 cross-game generalization

关键区别：SCOPE 不靠 data volume，靠 architecture。Spatial selectivity 是 inductive bias，让 69K 数据足够。

## 9. 我的整体评价和直觉

**这篇 paper 最强的点**：观察 sharp。FPS action 的 spatial selectivity 是一个被所有 prior work 忽视的明确现象，作者把它显式化并设计了与之匹配的 architecture。Per-pixel reshape + visually-queried cross-attention 是一个干净、可解释的机制——你能在脑子里"看见"为什么 in-scope pixel attend 强、out-of-scope pixel attend 弱。Zero-init 让训练稳定。Visual-action de-biasing 是 dataset 工程里的亮点。

**值得追问的点**：

1. **Why 不直接用 mask?** 既然 in-scope region 大致是 weapon 周围 + crosshair，理论上可以用 segmentation 或 heuristic mask 强制分离。Paper 暗示这种监督会 game-specific，不泛化。但 zero-shot 实验里 scope separation 仍工作，说明 visual query 学到的是"weapon-like appearance"而非具体 weapon——这是 generalizable 的。如果加少量 mask 监督会不会更快收敛？值得 ablation。

2. **Spatial reshape 的代价**。把 token 从 spatial-attended 转成 temporal-attended，意味着 self-attention 的 receptive field 改变了。这对 backbone 预训练 representation 是一种 misuse——backbone 在 spatial attention 上学的 pattern 可能失效。End-to-end 训练缓解了这点，但 Frozen variant 性能下降也证实了。可能需要一个 spatial-temporal hybrid attention。

3. **Flow Score vs Dynamic Degree 的张力**。SCOPE 的 Flow 18.24 + Photo. 0.198 同时领先，意味着大运动 + 稳定——这正是 FPS 应有的。但 Motion Smoothness 输给 Matrix-Game，说明大运动下 jitter 略多。可能 Action-CFG λ=1 不是 smoothness 最优点，可调 λ 在 smoothness 上可能有 trade-off 曲线。

4. **Cross-game 数据规模的 sweet spot**。10K 跨系列的崩溃提示：多游戏训练不是简单加数据就行，需要 diversity vs interference 的平衡。这跟 LLM 多语言训练的 similar phenomenon——少量语言 OK，多语言需要 curriculum。Paper 的 single-stage on 65K 是 empirical 选择，理论分析缺失。

5. **FPS 之外的泛化**。Per-pixel conditioning 对任何 egocentric interactive scene 都该有用——机器人操作（gripper 影响 object 局部）、驾驶（steering 影响 scene flow）、AR/VR。Paper 提了，但没实验。这是个明显的下一步。

6. **关于 V-JEPA 作为 evaluation**。JEPA Similarity 用 V-JEPA feature 算 cosine sim。V-JEPA 本身是 latent predictor，feature 偏 semantic 不偏 pixel。SCOPE JEPA 0.806 高，说明生成内容 semantic 结构保持好——但 paper 承认 texture > geometry，所以 JEPA 高不一定意味着几何对。可能需要再加一个几何专用 metric。

**对 world model 范式的影响**：这篇 paper 把"video diffusion as world model"从"global action injection"推到"spatially-selective action injection"。这是 conditioning mechanism 的 architectural shift，类比 ControlNet（[Zhang et al. 2023](https://arxiv.org/abs/2302.05543)）对 text-to-image 的影响——把 condition 从 global prompt 推到 spatial control。SCOPE 之于 game world model，可能就是 ControlNet 之于 image generation。

**最终直觉**：一个好的 interactive world model 的核心，是让 action signal 在正确的 spatial granularity 上生效。Global 是错的极端，per-pixel mask 是另一个极端，**visual-content-gated per-pixel** 是中间的优雅解。SCOPE 证明了这条路在 FPS 这种最 stress 的 setting 下可行，并给出 69K 数据训练 5B 模型 18 小时就 zero-shot 泛化的 evidence。这是一个值得 follow 的方向。

参考链接：
- [Wan Video Model](https://github.com/Wan-Video/Wan2.2)
- [Flow Matching paper](https://arxiv.org/abs/2210.02747)
- [DiT (Peebles & Xie)](https://arxiv.org/abs/2212.09748)
- [V-JEPA](https://arxiv.org/abs/2301.07843)
- [ControlNet](https://arxiv.org/abs/2302.05543)
- [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model)
- [Oasis (Decart)](https://oasis-model.github.io/)
- [GameGen-X](https://arxiv.org/abs/2411.00769)
- [GameGAN](https://arxiv.org/abs/2005.12326)
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)
- [Sora as world simulator](https://openai.com/research/video-generation-models-as-world-simulators)
- [FVD metric](https://arxiv.org/abs/1812.01717)
- [LPIPS](https://arxiv.org/abs/1801.03924)
- [WorldScore benchmark](https://arxiv.org/abs/2410.xxxxx)
- [VBench](https://arxiv.org/abs/2311.17882)
