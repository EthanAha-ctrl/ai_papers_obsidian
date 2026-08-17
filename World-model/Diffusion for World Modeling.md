---
source_pdf: Diffusion for World Modeling.pdf
paper_sha256: 842ff972404c74ca4bac9d2f37a6d309506c19eb3f2efa3e0daf75d6abc19707
processed_at: '2026-08-03T21:26:40-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 DIAMOND

## 一句话版本

**让 diffusion model 直接在像素空间当 world model, 训出来的 RL agent 在 Atari 100k 上打败了所有用 latent space 的方法, 还顺手做了一个能玩的 CS:GO neural game engine。**

---

## 背景到底在吵什么

先说整个 field 的两大 camp 在打什么仗。

### Camp A: Latent-based world model

代表: DreamerV3, IRIS, STORM, TWM

思路很自然: 64×64×3 的图像有 12288 个数, 直接在像素空间 rollout 太贵了, 先用一个 encoder 把图像压成几十个 latent variable, 在这个小空间里 rollout dynamics, 要用的时候再 decode 回图像。

为了 long-horizon 稳定, 大多数 latent 方法还把 latent 离散化 (比如 IRIS 把图像压成 16 个 token, 每个 token 是 codebook 里某个 vector 的 index)。离散化的好处是 rollout 时候不会慢慢 drift 到没见过的空间去, 因为 quantization 把 latent 锁死在 codebook 里。

**问题**: 压缩就有信息损失。你要是 codebook 只有 16 个 token, 那么一幅图里远处那个红绿灯、那个小敌人, 跟别的什么长得差不多的东西, 可能被 quantize 成同一个 token。然后 RL agent 看 decode 出来的图, 根本分辨不出红绿灯到底亮没亮。你可以说, 那我把 codebook 搞大点, 用 64 个 token, 256 个 token... 行, 但每多一个 token, autoregressive transformer 的推理就慢一截, IRIS 从 K=16 涨到 K=64 速度降了 2.8 倍 (paper Table 8 实测)。

### Camp B: Image-space world model

代表: SimPLe (2019, 老古董), DIAMOND (这篇)

思路: 别压缩, 直接在像素空间预测下一帧。这样 visual detail 一个都不丢。而且 world model 本身就能当 "neural game engine" 用——你不用 decode, 直接 render 就是图像。

问题是以前没有好工具。SimPLe 用的是 GAN, 有 mode collapse, 长 rollout 会崩。后来有了 diffusion, image generation 已经被 diffusion 统治了 (Stable Diffusion, SDXL, Midjourney, Sora...), diffusion 还天然能处理多模态分布 (给定 history 下一帧有多种可能), 还很容易加 conditioning。那能不能用 diffusion 做 world model? 这就是 DIAMOND 要回答的问题。

---

## 为什么这事不容易: compounding error

你想象一下, world model 是怎么用的: RL agent 不接触真环境, 它在 world model 的 imagination 里 rollout 1000 步来训练自己。每一步 world model 都要从 noise 生成一帧, 然后 RL agent 看这帧决定下一步 action, 再生成下一帧。

每帧哪怕只有一点点小偏差, 1000 步累积下来就是灾难。SimPLe 当年就栽在这上面, rollout 几十帧后就糊成噪声了。

diffusion 本身看着挺适合: 生成质量高, 能 condition。但 diffusion 有个大麻烦: **采样贵**。传统 DDPM 要 1000 步 denoising 才能从 noise 变成图像, Stable Diffusion 减到几十步。你 world model rollout 一步要 1000 步网络 forward, 那 RL 训练根本跑不动。

所以你必须把 denoising steps 砍到个位数, 最好 1 步。1 步 diffusion 采样听起来就离谱, 1 步能有什么质量?

---

## 关键 insight: 为什么用 EDM 不用 DDPM

这是 paper 最 beautiful 的 part。

作者试了两个方案:
- **DDPM** (Ho et al. 2020): 经典选择, Stable Diffusion 早期就基于 DDPM
- **EDM** (Karras et al. 2022, NVIDIA 那篇 design space 分析的 paper)

两个都拿来在 Breakout 上训 world model, 然后用同样初始帧 autoregressive rollout 1000 步。结果 (Figure 3):
- DDPM 用 10 步 denoising, 几十帧后开始颜色失真, 100 帧彻底糊
- DDPM 用 3 步, 更快崩
- DDPM 用 1 步, 第一帧就崩
- **EDM 用 1 步, 1000 帧后还稳**

这个对比非常 striking。为什么差距这么大?

回到训练目标的本质。DDPM 训练时让网络预测 "加进去的 noise"。你想想极端情况: 当 noise 加得特别多, $\sigma \to \infty$, 输入图像 $\mathbf{x}^\tau$ 本身就约等于纯 noise 了。这时候要预测的 noise 就是 $\mathbf{x}^\tau$ 本身, 所以网络学会的就是 **identity function**: 输出等于输入。

在 sampling 开始时 (从纯 noise 出发), 这个 identity 给的 score function 估计很差, reverse process 第一步就走错, 后面步步错。在图像生成里这没事, 你 1000 步慢慢修正回来; 在 world model 里 1 步采样, 第一步错就完了, 下一帧输入就是错的, 一帧错一帧, 1000 帧后画面成了噪声色块。

EDM 怎么解决? 它的训练 target 不固定, **根据 noise level 自适应**:
- 当 noise 极大: target 是 clean image 本身。网络学会在高 noise 下直接预测 "图像应该长什么样"
- 当 noise 极小: target 是 "加的 noise", 跟 DDPM 一样
- 中间 noise: target 是 clean 和 noise 的加权混合

这样在高 noise 区间, 网络给出一个有意义的 score estimate, reverse process 起步就准。1 步采样也稳定, 因为那一步大致就是 "从 noise 直接跳到 clean image"。

这个 insight 对所有 "autoregressive diffusion" 都适用, 不只是 world model。你想想看, image-to-image, video generation, 任何要长程 rollout 的 diffusion 应用, 都会遇到 high-noise 区间 score 估计差的问题。EDM 的 adaptive target 是个普适解。

---

## 那 1 步够用吗? 不一定

EDM 在 Breakout 这种 deterministic game 上 1 步完美, 但在 Boxing 这种 partial observable game 上 1 步不够。

Boxing 里黑色对手玩家的移动是不可预测的。给定同样历史, 下一帧黑人可能在左边, 也可能在右边, 这是个 **multimodal 分布**。

1 步 denoising 的最优解是 posterior mean, 也就是 "所有可能位置的平均"。结果就是一张糊图, 黑人变成一团模糊的影子, 同时出现在多个位置附近。RL agent 看这种糊图根本学不到东西。

多步 denoising 能让 reverse process "选一个 mode", 收敛到 "黑人在左边" 或 "黑人在右边" 中的一个, 给出 crisp 图像。

有趣的是白色玩家 (policy 控制的那个) 的位置, 1 步和多步都正确预测, 因为 action 已知没有 ambiguity。

所以作者折中选了 **3 步** denoising。3 步在大部分 game 上质量够好, 又比 IRIS 的 16 步便宜得多。

Table 7 给量化对比: Boxing 3-step 得 86.9 分, 1-step 只有 41.9; Breakout 132.5 vs 50.8; RoadRunner 20673 vs 5084。但 Asterix 这种 deterministic game 1-step 反而更好 (6687 vs 3698), 因为没有 multimodal 问题, 1 步直接预测 mean 反而更稳。

---

## 视觉细节怎么影响 RL 性能

这是 paper 最有说服力的实验 (Figure 5)。在 Asterix, Breakout, RoadRunner 上, 拿同样 100k expert frames 训 DIAMOND 和 IRIS 两个 world model, 对比生成帧的视觉一致性。

**IRIS 生成的问题**:
- Asterix 里: 第一帧是敌人 (橙色), 第二帧突然变成 reward (红色), 第三帧变回敌人, 第四帧又变 reward。一个东西帧间在 "敌人" 和 "奖励" 之间反复横跳
- Breakout 里: 砖块颜色和位置帧间不一致, 分数会跳变
- RoadRunner 里: 路上的 reward dot 时隐时现

这些 "小" 错误就几个像素, 人眼可能扫一眼还不一定注意, 但对 RL 是致命的。RL agent 学的是 "看到 reward 就去吃, 看到敌人就躲"。现在 world model 告诉它 "这帧是敌人", 下一帧说 "其实刚才是 reward", agent 直接裂开, credit assignment 全乱套。

**DIAMOND 完全没这问题**。而且 Breakout 里, 打掉红砖后分数自动 +7 更新, 说明 world model 真学到了 game logic, 不是 pattern match。

这种一致性直接反映到 RL 性能上: 这三个 game DIAMOND 都远超 IRIS (Asterix 3698 vs 854, Breakout 132 vs 84, RoadRunner 20673 vs 9614)。

而且 DIAMOND 计算还更便宜: 3 NFE/frame vs IRIS 16 NFE/frame, 参数 13M vs 30M, 训练 2.9 天 vs 4.1 天。

---

## 整体架构长什么样

把所有部件拼起来:

**Diffusion world model $\mathbf{D}_\theta$**:
- U-Net 2D, 不是 video diffusion 用的 U-Net 3D
- 原因: autoregressive 一次生一帧, U-Net 2D 更合适; U-Net 3D 是 joint diffuse 一整块帧, 不适合 autoregressive
- Conditioning: 过去 4 帧 image channel-wise concat (frame stacking), action 通过 adaptive group normalization 注入, diffusion time $\tau$ 也通过 adaptive group normalization

**Reward/termination model $\mathcal{R}_\psi$** (单独网络):
- CNN + LSTM, 两个 head 预测 reward 和 done
- 为什么不集成进 diffusion? 作者说提取 diffusion representation 不容易, 留给 future work

**Actor-critic** (RL agent):
- 共享 CNN-LSTM backbone, policy head 和 value head
- REINFORCE + value baseline + entropy regularization
- λ-returns 训 value (类似 Dreamer 系列)

**训练循环**: 每 epoch 在真环境收集 100 步, 然后分别 update diffusion model, reward model, actor-critic 各 400 步。共 1000 epochs。

整个 pipeline 和 IRIS 很像 (IRIS 也是 actor-critic + REINFORCE + λ-returns), 关键区别就是 world model 从 "discrete tokens + transformer" 换成 "diffusion in image space"。

---

## 主结果

Atari 100k, 26 个 game, 5 个 seed, 总共 1.03 GPU-years:

| 方法 | Mean HNS | IQM | #Superhuman |
|---|---|---|---|
| SimPLe | 0.332 | 0.130 | 1 |
| TWM | 0.956 | 0.459 | 8 |
| IRIS | 1.046 | 0.501 | 10 |
| DreamerV3 | 1.097 | 0.497 | 9 |
| STORM | 1.266 | 0.636 | 10 |
| **DIAMOND** | **1.46** | **0.64** | **11** |

mean HNS 1.46 是 world model trained agents 里新 SOTA。IQM 跟 STORM 持平但其他指标更强。

特别强: Asterix, Breakout, RoadRunner, Boxing, CrazyClimber, Kangaroo——都是 visual detail 关键的 game。

弱项: BankHeist (19.7), Seaquest (551), Frostbite (274), PrivateEye (114)——这些是 partial observability 重的 game, 需要长 memory, frame stacking 只有 4 帧不够。

跟 model-free / search 方法比:
- BBF 2.247 (用 periodic network reset + hyperparameter schedule)
- EfficientZero 1.943 (用 MCTS lookahead)
- DIAMOND 1.459

作者说这些 trick 和 DIAMOND 是 orthogonal 的, 加上来应该还能涨。

---

## CS:GO neural game engine

为了证明能 scale 到 3D, 作者在 CS:GO Dust II 地图上用 87 小时 human gameplay 数据训了个 world model, 可以用键鼠实时玩 (10 Hz on RTX 3090)。

做法:
- 主 model 在 56×30 低分辨率生成 dynamics
- 加第二个 small diffusion model 做 super-resolution 升到 280×150 (类似 SR3)
- 总参数 381M (Atari 只 13M), 训练 12 天

观察到的有趣 failure modes:
- 几百帧后可能在少见地图区域 drift 出分布
- 靠近墙时 memory 不够会 "忘" 当前状态, 生成新 weapon 或地图区域
- 错误地允许 mid-air 连跳 (训练数据里这种 action 太少, model 没学到 mid-air jump 应该被 ignore)

这是 offline data 训 world model 的固有问题, 不是 diffusion 的锅。同期 GameNGen (Google, 做 DOOM 的) 用 RL agent 收 data 来缓解, 但规模大得多。

---

## 几个值得记住的 intuition

**1. Autoregressive long-horizon 生成对 diffusion 训练目标敏感**

DDPM 的固定 noise-prediction target 在高 noise 区崩成 identity, long rollout 必死。EDM 的 adaptive target (高 noise 预测 clean, 低 noise 预测 noise) 给出全 noise 区间合理的 score estimate, 1 步采样都能稳定。这个 lesson 超出 world model 范畴, 任何 autoregressive diffusion (image-to-image, video gen, audio gen) 都用得上。

**2. Visual detail 直接影响 RL credit assignment**

世界模型不是 "生成质量好一点" 这种锦上添花, 而是几个像素的 enemy/reward 混淆直接破坏 RL learning。这跟自动驾驶里远处红绿灯必须清晰是同一个道理。Latent compression 有个根本的 detail 保真度上限, image-space 没有。

**3. World model 不等于 video generation**

Video generation 关心秒级 fidelity, 用 U-Net 3D joint diffuse 多帧; world model 要 long-horizon autoregressive 稳定, 用 U-Net 2D 单帧 + frame stacking 就够, 3D 时空 attention 反而不合适。简单架构在 autoregressive setting 下可能更好。

**4. NFE 是 world model 的核心指标**

imagination rollout 时每帧都要网络 forward, NFE 直接决定 RL 训练速度。3 NFE/frame 让 DIAMOND 比 IRIS 16 NFE/frame 在计算上更有竞争力, 参数还少一半。

**5. Partial observability 需要 multimodal 采样**

deterministic transition 1 步预测就够; partial observable + multimodal 分布必须多步采样让 reverse process 选 mode。这个 trade-off 决定了 denoising steps 不能一味求少。

---

## 我的看法

这篇 paper 巧妙地站在两个趋势的交叉点:
- 生成模型这边, diffusion 已经 dominate, 越来越多人意识到 DDPM 不一定是最优 choice, EDM 的 design space 分析提供了更好的工具
- World model 这边, discrete latent 路线 (Dreamer/IRIS) 已经到瓶颈, visual detail 的天花板卡住 RL 性能

DIAMOND 把 EDM 拿过来做 image-space world model, 解决了 compounding error 和 NFE 两大痛点, 一举打破 world-model-trained agent 在 Atari 100k 的纪录。CS:GO demo 更是证明 image-space world model 能当 neural game engine 用, 这是个很 exciting 的方向。

潜在问题: continuous control 没验证, long memory 依赖 frame stacking 比较弱 (长 POMDP 上掉链子, BankHeist/Seaquest 表现弱就是证据), reward model 还没集成进 diffusion。这些都是 future work 的明确方向。scale 上去之后能不能跟 GameNGen 拼, 也很值得看。

---

## References

- DIAMOND paper: https://arxiv.org/abs/2405.12399
- Project page (有 demo 视频和 playable CS:GO): https://diamond-wm.github.io
- EDM (Karras et al. 2022): https://arxiv.org/abs/2206.00364
- DDPM (Ho et al. 2020): https://arxiv.org/abs/2006.11239
- IRIS (Micheli et al. 2023): https://arxiv.org/abs/2303.07576
- DreamerV3 (Hafner et al. 2023): https://arxiv.org/abs/2301.04104
- SimPLe / Atari 100k (Kaiser et al. 2019): https://arxiv.org/abs/1903.00374
- World Models (Ha and Schmidhuber 2018): https://arxiv.org/abs/1803.10122
- GameNGen (Valevski et al. 2024, 同期 DOOM 工作): https://arxiv.org/abs/2408.14837
- Genie (Bruce et al. 2024, 同期 platformer 工作): https://arxiv.org/abs/2402.15391
- Score SDE (Song et al. 2020): https://arxiv.org/abs/2011.13456
- SR3 (Saharia et al. 2022, super-resolution diffusion): https://arxiv.org/abs/2104.07636
- EfficientZero (Ye et al. 2021): https://arxiv.org/abs/2111.00210
- BBF (Schwarzer et al. 2023): https://arxiv.org/abs/2305.19407
- CS:GO dataset (Pearce and Zhu 2022): https://arxiv.org/abs/2104.11068

---

# Diffusion for World Modeling (DIAMOND) 详细讲解

这篇 paper 是 NeurIPS 2024 的工作, 来自 University of Geneva 和 Microsoft Research Cambridge, 由 Eloi Alonso 和 Adam Jelley 共同一作, François Fleuret 和 Tim Pearce 共同通讯。核心 idea 是用 **diffusion model 直接在 image space 做 world model**, 而不是用主流的 discrete latent representation (如 DreamerV3, IRIS, STORM)。代码和可视化在 https://diamond-wm.github.io, arxiv 链接 https://arxiv.org/abs/2405.12399 。

---

## 1. Motivation: 为什么放弃 discrete latent 转向 diffusion

### 1.1 World model 的两种范式

当前 model-based RL 的 world model 大致分两类:

**Latent-based world model** (DreamerV2/V3, IRIS, TWM, STORM):
- 把 image $\mathbf{x}_t$ 通过 encoder 压成 latent $\mathbf{z}_t$, 在 latent 空间里 rollout dynamics
- DreamerV3 用 continuous + discrete latent 的 RSSM; IRIS 用 VQ-VAE 离散成 tokens 后用 autoregressive transformer; STORM 用不同的 tokenization
- Discrete latent 的好处是 long-horizon rollout 时 compounding error 小, 因为 quantization 限制了 drift

**Image-space world model** (SimPLe, 还有这篇 DIAMOND):
- 直接在 image space 预测下一帧 $\mathbf{x}_{t+1}$
- 好处是没有 encoder bottleneck, 不会丢 visual detail; world model 还能直接当 "neural game engine" 用

### 1.2 Discrete latent 的痛点

关键 insight 在 abstract 里: *"this compression into a compact discrete representation may ignore visual details that are important for reinforcement learning"*

举个具体例子: 自动驾驶场景里, 远处的红绿灯或行人只有几个像素, 如果离散 latent 的 codebook 不够大, 这些 pixels 就被 quantize 成同一个 token, 然后下游 RL agent 看不到红绿灯。要补救就要增加 codebook size (IRIS 用 K=16 tokens, 也试过 K=64, 但推理时间 2.8× slower, 见 paper Table 8)。

### 1.3 为什么 diffusion 适合做 world model

作者列出三个 diffusion 的性质:
1. **High-fidelity**: 现在 diffusion 已经 dominate 高分辨率 image generation (Stable Diffusion, SDXL, Imagen, Midjourney 等都是 diffusion)
2. **Easily conditionable**: diffusion 的 conditioning 很自然, classifier-free guidance 之类很成熟, 在 world model 里就是要 condition 在 past observations + actions 上
3. **Multi-modal modeling without mode collapse**: GAN-based 方法 (如 GameGAN) 有 mode collapse 问题, diffusion 能自然建模多模态分布。这在 world model 里很重要, 因为部分可观测环境下, 给定同样的 history, 下一帧可能有多种合法可能

---

## 2. Background: Score-based Diffusion

要把后面的 EDM choice 讲清楚, 必须先建立 score-based diffusion 的数学框架。

### 2.1 Forward process (noising)

定义 diffusion process $\{\mathbf{x}^\tau\}_{\tau \in [0, T]}$, 其中 $\tau$ 是 diffusion time (注意: paper 里特意用 $\tau$ 和上标表示 diffusion time, 而 $t$ 和下标表示 environment time)。边界条件:
- $\tau = 0$: $p^0 = p^{data}$ (clean data)
- $\tau = T$: $p^T = p^{prior}$ (Gaussian prior)

Forward SDE (公式 1):
$$
d\mathbf{x} = \mathbf{f}(\mathbf{x}, \tau) d\tau + g(\tau) d\mathbf{w}
$$

变量解释:
- $\mathbf{x}$: 数据点 (这里就是图像像素, Atari 上是 64×64×3)
- $\tau$: diffusion 时间, $[0, T]$ 区间
- $\mathbf{w}$: Wiener process, 即标准布朗运动
- $\mathbf{f}(\mathbf{x}, \tau)$: drift coefficient, 决定平均运动方向
- $g(\tau)$: diffusion coefficient, 决定噪声强度
- $d\tau$: 时间微分元

### 2.2 Reverse process (denoising)

Anderson (1982) 的关键结果: reverse process 也是 diffusion process, 由公式 2 描述:
$$
d\mathbf{x} = [\mathbf{f}(\mathbf{x}, \tau) - g(\tau)^2 \nabla_{\mathbf{x}} \log p^\tau(\mathbf{x})] d\tau + g(\tau) d\bar{\mathbf{w}}
$$

变量:
- $\nabla_{\mathbf{x}} \log p^\tau(\mathbf{x})$: **Stein score function**, 即 log-marginal 对 $\mathbf{x}$ 的梯度。注意它不是 log-likelihood 的梯度, 而是 log-density 对 *support* 的梯度
- $\bar{\mathbf{w}}$: reverse-time Wiener process

**核心难点**: 我们不知道真 score function, 要用神经网络 $\mathbf{S}_\theta(\mathbf{x}, \tau)$ 去估计它。

### 2.3 Denoising Score Matching

Hyvärinen (2005) 的 score matching 让我们不依赖真 score 也能训练。如果 forward kernel 是 Gaussian (即 $\mathbf{f}$ 是 affine), 可以解析地从 $\mathbf{x}^0$ 一步到 $\mathbf{x}^\tau$。

Denoising score matching objective (公式 3):
$$
\mathcal{L}(\theta) = \mathbb{E}\left[\|\mathbf{S}_\theta(\mathbf{x}^\tau, \tau) - \nabla_{\mathbf{x}^\tau} \log p^{0\tau}(\mathbf{x}^\tau \mid \mathbf{x}^0)\|^2\right]
$$

其中:
- 期望 over: diffusion time $\tau$, noised sample $\mathbf{x}^\tau \sim p^{0\tau}(\mathbf{x}^\tau \mid \mathbf{x}^0)$, clean sample $\mathbf{x}^0 \sim p^{data}$
- $p^{0\tau}$: τ-level perturbation kernel (已知 Gaussian)

因为 Gaussian kernel 可微, 这简化成 L2 reconstruction loss (公式 4):
$$
\mathcal{L}(\theta) = \mathbb{E}\left[\|\mathbf{D}_\theta(\mathbf{x}^\tau, \tau) - \mathbf{x}^0\|^2\right]
$$

通过 reparameterization $\mathbf{D}_\theta(\mathbf{x}^\tau, \tau) = \mathbf{S}_\theta(\mathbf{x}^\tau, \tau) \sigma^2(\tau) + \mathbf{x}^\tau$ 把 score model 变 denoiser, $\sigma(\tau)$ 是 perturbation kernel 的标准差。

### 2.4 Condition 到 world model

普通 diffusion 学的是 $p_{data}(\mathbf{x})$, world model 要学的是 $p(\mathbf{x}_{t+1} \mid \mathbf{x}_{\leq t}, a_{\leq t})$。在 POMDP 假设下, 用 history 估计 Markov state, 把 diffusion 条件化在 history 上 (公式 5):

$$
\mathcal{L}(\theta) = \mathbb{E}\left[\|\mathbf{D}_\theta(\mathbf{x}_{t+1}^\tau, \tau, \mathbf{x}_{\leq t}^0, a_{\leq t}) - \mathbf{x}_{t+1}^0\|^2\right]
$$

变量解释:
- $\mathbf{x}_{t+1}^\tau$: 下一帧加噪版本
- $\tau$: diffusion time
- $\mathbf{x}_{\leq t}^0$: 过去的 clean observations
- $a_{\leq t}$: 过去的 actions
- $\mathbf{x}_{t+1}^0$: clean target 下一帧

图 1 很关键, 描绘了 imagination 过程: 横轴是 environment time $t$, 纵轴是 diffusion time $\tau$ 从 $T$ 流向 $0$。给定 (clean) past observations $\mathbf{x}_{<t}^0$, actions $a_{<t}$, 从 noise $\mathbf{x}_t^T$ 出发, 反复调用 $\mathbf{D}_\theta$ 做 reverse process, 得到 clean next observation $\mathbf{x}_t^0$。然后这帧变成下一步的 conditioning, 自回归展开。https://diamond-wm.github.io 上有动画。

---

## 3. Method: DIAMOND 的具体实现

### 3.1 为什么选 EDM 而不是 DDPM

这是 paper 最有意思的设计选择之一 (Section 5.1)。历史上 DDPM (Ho et al. 2020) 是 diffusion 的经典选择, Stable Diffusion 等很多应用都用 DDPM 系。但作者发现 DDPM 在 world model 上严重失败, 选了 Karras et al. (2022) 的 **EDM** (Elucidating the Design Space of Diffusion Models, https://arxiv.org/abs/2206.00364)。

EDM 的 forward kernel 选 Gaussian:
$$
p^{0\tau}(\mathbf{x}_{t+1}^\tau \mid \mathbf{x}_{t+1}^0) = \mathcal{N}(\mathbf{x}_{t+1}^\tau; \mathbf{x}_{t+1}^0, \sigma^2(\tau) \mathbf{I})
$$

对应 drift 和 diffusion coefficient:
- $\mathbf{f}(\mathbf{x}, \tau) = \mathbf{0}$ (VE, variance exploding 形式)
- $g(\tau) = \sqrt{2\dot{\sigma}(\tau) \sigma(\tau)}$

这里 $\dot{\sigma}(\tau) = d\sigma/d\tau$, $\sigma(\tau)$ 是 noise schedule, 称为 noise level (注意: 这里 $\sigma$ 既是 noise level 又是 perturbation kernel 的 std, 在 VE 形式下是一回事)。

### 3.2 EDM 的网络 preconditioning

EDM 的精髓是 network preconditioning (公式 6):
$$
\mathbf{D}_\theta(\mathbf{x}_{t+1}^\tau, y_t^\tau) = c_{\mathrm{skip}}^\tau \mathbf{x}_{t+1}^\tau + c_{\mathrm{out}}^\tau \mathbf{F}_\theta(c_{\mathrm{in}}^\tau \mathbf{x}_{t+1}^\tau, y_t^\tau)
$$

变量:
- $y_t^\tau := (c_{\mathrm{noise}}^\tau, \mathbf{x}_{<t}^0, a_{\leq t})$: 所有 conditioning 变量打包
- $c_{\mathrm{in}}^\tau = 1/\sqrt{\sigma(\tau)^2 + \sigma_{data}^2}$: 网络输入的归一化系数, 保证网络输入单位方差
- $c_{\mathrm{out}}^\tau = \sigma(\tau) \sigma_{data} / \sqrt{\sigma(\tau)^2 + \sigma_{data}^2}$: 网络输出的归一化系数
- $c_{\mathrm{noise}}^\tau = \frac{1}{4} \log(\sigma(\tau))$: noise level 的经验变换, 喂给网络做 conditioning
- $c_{\mathrm{skip}}^\tau = \sigma_{data}^2 / (\sigma_{data}^2 + \sigma^2(\tau))$: skip connection 的权重
- $\sigma_{data} = 0.5$: data 分布的标准差 (EDM 经验值, 假设 image data 大致 normalize 到 [-1, 1] 后 std ≈ 0.5)

代入训练目标 (公式 7):
$$
\mathcal{L}(\theta) = \mathbb{E}\left[\left\|\underbrace{\mathbf{F}_\theta(c_{\mathrm{in}}^\tau \mathbf{x}_{t+1}^\tau, y_t^\tau)}_{\text{Network prediction}} - \underbrace{\frac{1}{c_{\mathrm{out}}^\tau}(\mathbf{x}_{t+1}^0 - c_{\mathrm{skip}}^\tau \mathbf{x}_{t+1}^\tau)}_{\text{Network training target}}\right\|^2\right]
$$

**关键 intuition**: 训练 target 是 $\frac{1}{c_{\mathrm{out}}^\tau}(\mathbf{x}_{t+1}^0 - c_{\mathrm{skip}}^\tau \mathbf{x}_{t+1}^\tau)$, 这个 target 根据 noise level $\sigma(\tau)$ 自适应地混合 clean signal 和 noise:

- **当 noise 大, $\sigma(\tau) \gg \sigma_{data}$**: $c_{\mathrm{skip}}^\tau \to 0$, target 变成 $\frac{1}{c_{\mathrm{out}}^\tau} \mathbf{x}_{t+1}^0$, 也就是网络要直接预测 clean image
- **当 noise 小, $\sigma(\tau) \to 0$**: $c_{\mathrm{skip}}^\tau \to 1$, target 变成 $\frac{1}{c_{\mathrm{out}}^\tau}(\mathbf{x}_{t+1}^0 - \mathbf{x}_{t+1}^\tau) \approx \frac{1}{c_{\mathrm{out}}^\tau} \cdot \text{noise}$, 网络要预测 "加的 noise"

**为什么这比 DDPM 好**: DDPM 用 noise prediction 目标 $\xi_\theta(\mathbf{x}^\tau) \to \text{noise}$。当 noise 极大时, $\mathbf{x}^\tau \approx \text{noise}$, 所以网络学会的近似是 $\xi_\theta(\mathbf{x}^\tau) \to \mathbf{x}^\tau$ (identity function!)。这给 score function 一个非常差的估计, 导致 sampling 一开始就跑偏, 几步后就 drift 出分布外。

EDM 在 noise 大时让网络直接预测 clean image, 给出靠谱的 score estimate, 所以即使只用 **1 个 denoising step** 都很稳定 (Figure 3 右下角)。DDPM 即使用 10 个 denoising steps 都 drift 严重 (Figure 3a)。

### 3.3 Noise level 采样

Karras et al. (2022) 发现 objective 在 noise 极大或极小时方差很大, 训练效率低。解法是从 log-normal 分布采样 $\sigma$:
$$
\log(\sigma(\tau)) \sim \mathcal{N}(P_{mean}, P_{std}^2)
$$
其中 $P_{mean} = -0.4$, $P_{std} = 1.2$。这把训练 mass 集中在 medium-noise region, 让网络重点学好 "中段" 的 denoising, 这是 sampling 过程中实际经过的轨迹。

### 3.4 网络架构

$\mathbf{F}_\theta$ 用 standard **U-Net 2D** (Ronneberger et al. 2015), 不是 video diffusion 里常用的 U-Net 3D (Çiçek et al. 2016, 如 Make-A-Video, Lumiere)。原因: U-Net 3D 是把一整块 frames 联合 diffuse, 而这里我们是 **autoregressive** 生成, 一次只生成一帧, 用 U-Net 2D 更合适。

Conditioning 机制:
- **Frame stacking**: 保留过去 $L=4$ 帧 observation 和 action, channel-wise concat 到 noised frame
- **Action conditioning**: 通过 adaptive group normalization (Zheng et al. 2020) 在 residual blocks 里注入
- **Diffusion time $\tau$ conditioning**: 同样通过 adaptive group normalization

Appendix M 里对比了 frame-stacking 和 cross-attention 两种架构 (Figure 9), 发现 frame-stacking 反而更好, 即使 cross-attention 在 video generation 里更主流。作者推测是 autoregressive generation 里直接喂 input 的 inductive bias 更合适。

架构参数 (Table 2):
- Residual blocks layers: [2, 2, 2, 2]
- Residual blocks channels: [64, 64, 64, 64] (Atari), 256 conditioning dim
- DIAMOND 总参数 13M (vs IRIS 30M, DreamerV3 18M, Table 4)

### 3.5 Sampler 选择

理论上任何 ODE/SDE solver 都行。Song et al. 2020 还提了 "probability flow ODE", deterministic 的, 但 marginal 等价于 SDE。

DIAMOND 用最简单的 **Euler's method** (一阶 deterministic)。理由:
- 高阶 solver (Heun, Runge-Kutta) 减少 truncation error 但增加 NFE (Number of Function Evaluations), 推理贵
- Stochastic samplers (Euler-Maruyama) 增加复杂度没明显好处

每个 denoising step 是一次网络 forward, 所以 NFE = denoising steps $n$。**默认 $n=3$**。

---

## 4. RL in Imagination

完整 world model 还需要 reward 和 termination 预测 (DIAMOND 这里没把它们集成进 diffusion, 留给 future work)。

### 4.1 Reward/Termination model $\mathcal{R}_\psi$

单独的 CNN + LSTM 网络:
- CNN: residual blocks, [32, 32, 32, 32] channels
- LSTM: 512 dim
- 两个 head: 一个预测 reward, 一个预测 done
- 用 burn-in (Kapturowski et al. 2018) 初始化 LSTM hidden state

训练目标:
$$
\mathcal{L}(\psi) = \sum_i \mathrm{CE}(\hat{r}_i, \mathrm{sign}(r_i)) + \mathrm{CE}(\hat{d}_i, d_i)
$$

reward 用 sign (即 {-1, 0, 1} 三个 class), Atari 里 reward clip 后就这三种, 当分类问题做。

### 4.2 Actor-Critic

$\pi_\phi$ 和 $V_\phi$ 共享 CNN-LSTM backbone:
- 4 个 residual blocks, [32, 32, 64, 64] channels
- LSTM 512 dim
- 两个 head: policy 和 value

### 4.3 RL objectives

**λ-returns** (公式 14):
$$
\Lambda_t = \begin{cases} r_t + \gamma(1-d_t)[(1-\lambda)V_\phi(\mathbf{x}_{t+1}) + \lambda \Lambda_{t+1}] & \text{if } t < H \\ V_\phi(\mathbf{x}_H) & \text{if } t = H \end{cases}
$$

变量:
- $r_t, d_t$: world model 预测的 reward 和 termination
- $H = 15$: imagination horizon
- $\gamma = 0.985$: discount factor (Atari 100k 常用)
- $\lambda = 0.95$: λ-returns 的 bias-variance tradeoff

**Value loss** (公式 15):
$$
\mathcal{L}_V(\phi) = \mathbb{E}_{\pi_\phi}\left[\sum_{t=0}^{H-1}(V_\phi(\mathbf{x}_t) - \mathrm{sg}(\Lambda_t))^2\right]
$$
$\mathrm{sg}$ 是 stop-gradient, target 不参与反传 (经典做法, 见 Mnih et al. 2015, DreamerV2)。

**Policy loss** (公式 16) 用 REINFORCE + value baseline + entropy regularization:
$$
\mathcal{L}_\pi(\phi) = -\mathbb{E}_{\pi_\phi}\left[\sum_{t=0}^{H-1}\log(\pi_\phi(a_t \mid \mathbf{x}_{\leq t})) \mathrm{sg}(\Lambda_t - V_\phi(\mathbf{x}_t)) + \eta \mathcal{H}(\pi_\phi(a_t \mid \mathbf{x}_{\leq t}))\right]
$$

变量:
- $\eta = 0.001$: entropy weight
- $\mathcal{H}$: entropy
- $\Lambda_t - V_\phi(\mathbf{x}_t)$: advantage estimate, stop-gradient

不用 PPO 等 on-policy 优化, 因为 imagination 里可以大量 on-policy rollout, 简单 REINFORCE 就够 (类似 IRIS 的设计)。

### 4.4 整体训练循环 (Algorithm 1)

每个 epoch:
1. `collect_experience(100 steps)`: 在真环境用 ε-greedy ($\epsilon=0.01$) 收集 100 步
2. `update_diffusion_model × 400 steps`
3. `update_reward_end_model × 400 steps`
4. `update_actor_critic × 400 steps`

共 1000 epochs, batch size 32。2.9 天 / game / seed on RTX 4090, 12GB VRAM。

---

## 5. Atari 100k 实验

### 5.1 Benchmark 设置

Atari 100k (Kaiser et al. 2019, SimPLe paper): 26 个 game, agent 只能在真环境 interaction 100k 步 (~2 小时 human gameplay), 远少于标准 50M 步 ALE 设置。

DIAMOND 5 个 seed, 1.03 GPU years 总计。

### 5.2 主结果 (Table 1)

mean Human Normalized Score (HNS):
- DIAMOND: **1.46** (新 SOTA among world-model trained agents)
- STORM: 1.266
- DreamerV3: 1.097
- IRIS: 1.046
- TWM: 0.956
- SimPLe: 0.332

IQM (interquartile mean, Agarwal et al. 2021 推荐):
- DIAMOND: 0.641 (与 STORM 的 0.636 持平)
- 其他都更低

11/26 games superhuman, 也是 world model 类方法里最多。

特别强的 game: Asterix (3698.5 vs IRIS 853.6), Breakout (132.5 vs IRIS 83.7), RoadRunner (20673.2 vs STORM 17564.0), Boxing (86.9), CrazyClimber (99167.8)。

弱的 game: BankHeist (19.7), Seaquest (551.2), Frostbite (274.1), PrivateEye (114.3) - 这些是 partial observability 重的 game, 可能需要更长 memory。

### 5.3 和 model-free / search 方法对比 (Table 6, Appendix J)

- BBF (Schwarzer et al. 2023): 2.247 mean HNS (但用 periodic network reset + hyperparameter scheduling)
- EfficientZero (Ye et al. 2021): 1.943 (用 MCTS lookahead, 计算贵)
- DIAMOND: 1.459

作者强调这些方法和 DIAMOND 是 orthogonal 的 (network reset, MCTS 都可以叠加到 DIAMOND 上)。

---

## 6. 关键分析

### 6.1 DDPM vs EDM 的 compounding error (Section 5.1, Figure 3)

这是 paper 最有教学意义的实验。在 Breakout 上用 expert 收集 100k frames 的 static dataset, 分别训 DDPM 和 EDM world model, 然后 autoregressive rollout 1000 步看漂移。

**DDPM (Figure 3a)**:
- n=10 步: 几十帧后开始 drift, 颜色失真
- n=3 步: 更快 drift
- n=1 步: 几乎立刻坏掉, 变成纯噪声色块

**EDM (Figure 3b)**:
- n=10, 3, 1 步: 都很稳定, 即使 1000 步后画面还是 reasonable

为什么? 回到 Section 3.2 的分析:
- DDPM 训练目标固定是 "预测 noise"。当 noise 极大时 $\mathbf{x}^\tau \approx \text{noise}$, 网络学会 identity $\xi_\theta(\mathbf{x}^\tau) \to \mathbf{x}^\tau$
- 在 sampling 开始时 ($\tau = T$, noise 极大), 这个 identity 给出非常差的 score 估计, 整个 reverse trajectory 起步就错
- 在 autoregressive world model 里, 每一帧都从 noise 开始, 每帧都错一点, compounding error 1000 帧后崩溃
- EDM 的 adaptive target 让网络在 noise 大时直接预测 clean image, 给出靠谱 score estimate, 一步 denoising 都稳定

Appendix K (Figure 8) 给定量证据: 用 400 条 reference trajectory, 量 average pixel drift。DDPM 即使 10 步都不如 EDM 1 步稳定。

### 6.2 Denoising steps 的取舍 (Section 5.2, Figure 4, Table 7)

为什么默认 $n=3$ 而不是 $n=1$?

$n=1$ 在 deterministic transitions (Breakout) 上 OK, 但在 partial observable / multimodal 环境 (Boxing) 上有问题。

Boxing 里黑色对手玩家移动是不可预测的, 给定 history 下一帧他的位置有多个可能。单步 denoising 的最优预测是 **posterior mean** (即所有 modes 的平均), 这给出一张模糊图 (Figure 4 上排)。多步 sampling 让反向过程 "选择" 一个 mode, 给出 crisp image (Figure 4 下排)。

有趣的是白色玩家由 policy 控制, action 已知, 没有 ambiguity, 所以单步和多步都正确预测白人位置。

Table 7 给 1-step vs 3-step 量化对比 (top-10 games):
- Boxing: 86.9 (n=3) vs 41.9 (n=1)
- Breakout: 132.5 vs 50.8
- RoadRunner: 20673.2 vs 5084.0
- Asterix: 3698.5 vs 6687.0 (这个 1-step 反而更好, 单模态游戏)
- Kangaroo: 5382.2 vs 1710.0

Mean HNS (这 10 个 game): 3.052 (n=3) vs 1.962 (n=1)

### 6.3 Visual quality vs IRIS (Section 5.3, Figure 5)

在 Asterix, Breakout, RoadRunner 上和 IRIS 对比生成帧。IRIS 用 VQ-VAE 离散 tokens + transformer, 容易出 "视觉不一致":
- Asterix 里敌人 (orange) 在第二帧变成 reward (red), 第三帧变回 enemy
- Breakout 里砖块和分数帧间不一致
- RoadRunner 里 reward dots 帧间消失

DIAMOND 没有这类问题, 而且 Breakout 里分数能在打掉红砖后自动 +7 更新, 说明 world model 真的学到了 game logic, 不是 pattern matching。

这些 visual 不一致对 RL 伤害很大: agent 要 target reward avoid enemy, 但 IRIS 给的帧里 enemy/reward 混淆, credit assignment 困难。

而且 DIAMOND 计算更便宜: 3 NFE vs IRIS 16 NFE per frame, 参数更少 (13M vs 30M), 训练更快 (2.9 vs 4.1 days)。

---

## 7. CS:GO: Scalable 到 3D 游戏

为展示 diffusion world model 能 scale 到更复杂 3D 环境, 在 CS:GO 的 Dust II 地图上训练 interactive neural game engine。

### 7.1 数据

用 Pearce and Zhu (2022) 的 CS:GO 数据集:
- 5.5M frames, 16 Hz, 95 小时 human gameplay
- 5M frames (87 小时) 训练, 0.5M (8 小时) 测试
- 无 RL agent, 纯 static dataset 训练 world model

### 7.2 架构调整

- 主 model 分辨率: 280×150 → 56×30 (降低 25×)
- 加第二个 small diffusion model 做 **upsampler** (类似 SR3, Saharia et al. 2022), 输出 280×150
- U-Net channels 大幅扩张, 4M (Atari) → 381M (CS:GO, 含 51M upsampler)
- 12 天训练 on RTX 4090
- 推理: dynamics model 3 denoising steps, upsampler 10 denoising steps + stochastic sampling
- 在 RTX 3090 上 10 Hz (人可以实时玩)

https://diamond-wm.github.io 有 demo 视频。

### 7.3 观察到的 failure modes

- Long rollout 几百帧后可能在少见地图区域 drift out-of-distribution
- 接近墙或失去视线时, model 记忆有限可能 "忘" 当前状态, 生成新 weapon 或地图区域
- 错误地允许 mid-air 连跳 (训练数据里这种 action 太少, model 没学到 mid-air jump 应该被 ignore)
- 这些是 offline data 训练 world model 的固有问题, 不是 diffusion 特有

### 7.4 和 GameNGen 的对比

GameNGen (Valevski et al. 2024, https://arxiv.org/abs/2408.14837) 是同期工作, 也用 diffusion 做 DOOM 的 game engine。区别: GameNGen 用两阶段 (先 RL agent 收集 data, 再训 diffusion world model), scale 更大。DIAMOND 用 static human data。

---

## 8. Appendix M: 3D 环境定量对比

Appendix M 是被审稿人塞进 appendix 的实质内容, 给了 CS:GO 和 motorway driving (Santana and Hotz 2016) 上的定量对比。

### 8.1 两种架构对比 (Figure 9)

- **Frame-stacking**: concat $[\mathbf{x}_t^\tau, \mathbf{x}_{t-1}^0, \ldots, \mathbf{x}_{t-L}^0]$, U-Net 2D 处理
- **Cross-attention**: U-Net 2D 只收 noised frame, 通过 cross-attention 关注 history encoder (类似 video diffusion 的设计)

### 8.2 量化结果 (Table 8)

CS:GO 上:
| Method | FID ↓ | FVD ↓ | LPIPS ↓ | Sample rate (Hz) ↑ | #Params |
|---|---|---|---|---|---|
| DreamerV3 | 106.8 | 509.1 | 0.173 | 266.7 | 181M |
| IRIS (K=16) | 24.5 | 110.1 | 0.129 | 4.2 | 123M |
| IRIS (K=64) | 22.8 | 85.7 | 0.116 | 1.5 | 111M |
| DIAMOND frame-stack | **9.6** | **81.4** | **0.107** | 7.4 | 122M |
| DIAMOND cross-attention | 11.6 | 34.8 | 0.125 | 2.5 | 184M |

(注意: cross-attention 在 driving 上 FVD 反而更好 299.9 vs 80.3, 两个架构各有强项)

Driving (motorway) 类似, DIAMOND frame-stack 全面最好。

**结论**: DIAMOND frame-stack > DIAMOND cross-attention ≈ IRIS 64 > IRIS 16 > DreamerV3

**DreamerV3 的 speed/quality trade-off**: 快 40 倍但质量差, 因为它 latent-space 独立采样 (不 joint decode)。

**IRIS 的 K 瓶颈**: K=16 → K=64 速度降 2.8×, 量化 fidelity 上限就是 K 个 token。

---

## 9. Limitations 和 Future Work

1. **Continuous control**: 只在 discrete action (Atari, CS:GO 的离散 action) 上验证, continuous domain 没试
2. **Memory**: frame-stacking 是 minimal memory 机制, 长 horizon 会忘。可以引入 autoregressive transformer over environment time (类似 DiT, Peebles and Xie 2023) 提供长记忆。早期 cross-attention 实验不如 frame-stacking, 但值得继续探索
3. **Reward/termination 集成**: 当前 $\mathcal{R}_\psi$ 是独立模型。集成进 diffusion (类似 DiffusionClassifier) 会让 world model 更紧凑, 但从 diffusion 提 representation 不容易 (Luo et al. 2023, Xu et al. 2023)

---

## 10. 个人 Intuition 总结

1. **Diffusion 作 world model 的本质优势**: 不压缩信息, 没有 encoder bottleneck, 不丢 visual detail。代价是推理贵 (要做 reverse process), 但 EDM 的 adaptive target 让 1-step 都可用, 大幅缓解
2. **EDM > DDPM 的关键 insight**: 训练目标的自适应性。DDPM 固定预测 noise, 在高 noise 区崩成 identity; EDM 自适应预测 clean/noise/diff, 全 noise 区间都 work。这个 lesson 对所有 autoregressive diffusion 应用都适用
3. **World model vs video generation 的差异**: Video generation 关心短期 fidelity (秒级), 用 U-Net 3D joint diffuse 多帧; world model 要 long-horizon autoregressive 稳定, 用 U-Net 2D 单帧生成 + frame-stacking, 不需要 3D 时空 attention
4. **Visual detail → RL performance 的因果链**: 越细节的 visual fidelity 越能让 RL agent 学到对 credit assignment 关键的 signal。IRIS 的 enemy/reward 混淆直接破坏 RL learning, DIAMOND 的一致性带来 RL 性能提升
5. **NFE 是 world model 的核心指标**: 因为 imagination rollout 时每步都要 network forward, NFE 决定训练速度。3 NFE/frame 让 DIAMOND 在计算上比 IRIS (16 NFE) 更有竞争力

---

## References

- DIAMOND paper: https://arxiv.org/abs/2405.12399
- Project page: https://diamond-wm.github.io
- EDM (Karras et al. 2022): https://arxiv.org/abs/2206.00364
- DDPM (Ho et al. 2020): https://arxiv.org/abs/2006.11239
- IRIS (Micheli et al. 2023): https://arxiv.org/abs/2303.07576
- DreamerV3 (Hafner et al. 2023): https://arxiv.org/abs/2301.04104
- STORM (Zhang et al. 2023): https://arxiv.org/abs/2307.07959
- Atari 100k benchmark / SimPLe (Kaiser et al. 2019): https://arxiv.org/abs/1903.00374
- World Models (Ha and Schmidhuber 2018): https://arxiv.org/abs/1803.10122
- GameGAN (Kim et al. 2020): https://arxiv.org/abs/2005.12326
- GameNGen (Valevski et al. 2024): https://arxiv.org/abs/2408.14837
- Genie (Bruce et al. 2024): https://arxiv.org/abs/2402.15391
- EfficientZero (Ye et al. 2021): https://arxiv.org/abs/2111.00210
- BBF (Schwarzer et al. 2023): https://arxiv.org/abs/2305.19407
- Score SDE (Song et al. 2020): https://arxiv.org/abs/2011.13456
- Anderson reverse-time diffusion (1982): https://www.sciencedirect.com/science/article/pii/0304414982900365
- CS:GO dataset (Pearce and Zhu 2022): https://arxiv.org/abs/2104.11068
- Adaptive Group Normalization (Zheng et al. 2020): https://arxiv.org/abs/2010.12903
- U-Net (Ronneberger et al. 2015): https://arxiv.org/abs/1505.04597
- VQ-VAE (Van Den Oord et al. 2017): https://arxiv.org/abs/1711.00937
- DiT (Peebles and Xie 2023): https://arxiv.org/abs/2212.09748
- Sora (Brooks et al. 2024): https://openai.com/research/video-generation-models-as-world-simulators
- SR3 (Saharia et al. 2022): https://arxiv.org/abs/2104.07636
- Agarwal et al. 2021 (RL statistical evaluation): https://arxiv.org/abs/2108.13264
