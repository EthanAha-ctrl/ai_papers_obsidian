---
source_pdf: AdaWorld Learning Adaptable World Models with Latent Actions.pdf
paper_sha256: ebaded596ac3941608f0faebfbb8e25e72baf2f8c758f7e08f464d31250ce26e
processed_at: '2026-08-17T23:56:37-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# AdaWorld 人话版

好，我换个说法。刚才那版太"论文解读"了，我重来一遍，讲人话。

---

## 一句话版本

AdaWorld 干的事：**先从一堆没人标注的视频里，让模型自己猜"这两帧之间发生了什么动作"，把这个"动作猜测"压缩成一个 32 维的小向量。然后用这玩意儿当遥控器，预训练一个能预测下一帧的 world model。等到了新环境，只要给一两个动作示范，模型就知道该按哪个"按钮"了。**

就这么个事。

---

## 为什么要这么干？先讲个故事

假设你要训练一个机器人。传统做法两条路：

**路线 A**：给它看一百万段 YouTube 视频，让它学"世界是怎么运转的"。但它只学会了画面怎么变化，**根本不知道什么是"动作"**——给它一个新手柄，它傻眼，得从零学哪个键是跳。

**路线 B**：雇人给视频标动作（"这一帧按了 A，下一帧按了左"）。问题：一个视频标动作要好几块钱，一百万视频几百万刀；而且"按 A"在马里奥和魂斗罗里含义完全不同，**action space 跨环境不统一**，标注格式定不下来。

AdaWorld 选了路线 C：**让模型自己从画面变化里"猜"动作，猜完之后拿这个猜出来的动作当统一遥控器用。**

这就像你小时候看别人玩红白机——没人告诉你"上键是跳"，但你看着看着就懂了：**哦，画面里小人突然腾空那一下，就是"跳"这个动作。** 然后你换到另一个游戏，你脑子里"跳"这个概念是不变的，你按上键就知道会发生类似的事。

AdaWorld 就是把这个"看会的动作概念"做成模型的一个 32 维向量。

---

## 这个 32 维向量怎么来的？信息瓶颈的艺术

### 核心机制

给模型两帧连续画面 $f_t$ 和 $f_{t+1}$，让它回答一个问题：

> "这两帧之间，**最少**需要多少信息，我才能从 $f_t$ 重建出 $f_{t+1}$？"

把这个"最少必要信息"压缩到 32 维。就这么简单。

数学上就是个 VAE：

$$
\underbrace{\mathbb{E}_{q_\phi(\tilde a | f_{t:t+1})} \log p_\theta(f_{t+1} | \tilde a, f_t)}_{\text{重建得越准越好}} - \beta \underbrace{D_{KL}(q_\phi(\tilde a | f_{t:t+1}) \| \mathcal{N}(0, I))}_{\text{但 latent 分布要规整}}
$$

变量含义：
- $\tilde a \in \mathbb{R}^{32}$：latent action，那个 32 维向量
- $q_\phi(\tilde a | f_{t:t+1})$：encoder 输出的高斯分布，参数是 $(\mu_{\tilde a}, \sigma_{\tilde a})$
- $p_\theta(f_{t+1} | \tilde a, f_t)$：decoder 用 $f_t$ + $\tilde a$ 重建 $f_{t+1}$ 的 likelihood
- $\beta = 2 \times 10^{-4}$：KL 项的权重，**故意调得极小**

### 为什么要 32 维这么小？

一张 256×256 的 RGB 图有 19 万 6 千多个数。32 维连 0.02% 都不到。

这是个**故意的饥饿营销**。你给 encoder 32 个槽位，它装不下整张下一帧。它必须挑：**哪些信息最值得保留？**

在交互式视频里，相邻两帧最大的变化通常是**有人/agent 在做事**——比如手把杯子往左推、马里奥跳起来、车辆转弯。背景的云朵飘动、树叶摇晃那种是"噪声"，不重要。

所以 encoder 会被迫优先编码"动作意图"那种 dominant signal，把 context 细节（颜色、纹理）丢掉。

这就是 paper Figure 2 那个 "information bottleneck" 图的含义。

### 为什么 β 要调这么小？

经典 β-VAE 用大 β 是为了强行 disentangle。但这里反过来——**β 太大的话 latent 被压得太死，连"前进"和"后退"都区分不开**，因为高斯 prior 把所有方向都抹平了。

paper 原话（Section 2.1）：

> 标准 VAE imposes a strong constraint on posterior distributions ... struggles to express diverse transitions. 但完全去掉 KL 又会 compromise the disentanglement ability.

所以 β 是个 dial：
- β 太大 → 所有 latent 都长一样，区分不出不同动作
- β 太小 → latent 偷偷塞进 context 信息（比如"红色物体在左上角"），跨环境 transfer 时崩
- $\beta = 2\times10^{-4}$ 是 sweet spot

Figure 7 的 UMAP 可视化很直观——这个值下不同 action 自然聚类，相同 action 跨环境还能 overlap，transfer 就 work。

---

## 为什么不用离散 codebook？Genie 那条路的死结

DeepMind 的 Genie (Bruce et al., 2024, <https://sites.google.com/view/genie-2024>) 用的就是 8 个离散 code 的 VQ-VAE。AdaWorld 偏偏用 continuous 32 维。这是有意为之的对抗。

离散 codebook 的三个死结：

**1. 表达力天花板。** 8 个 code 怎么覆盖连续动作？开车方向盘转 30 度和 35 度，是同一个 code 还是两个？没法精确表达。Table 1 里 discrete cond. baseline 在 LIBERO Human eval 只有 3.5%，AdaWorld 70.5%。这 67 个百分点的差距就是离散化的代价。

**2. 不能 average。** AdaWorld 适应新环境时一个关键操作：把同 action 的 100 个 latent 平均，得到一个稳定的 action embedding。连续空间里 average 是合法的——两个 "前进" 的 latent 求平均还是 "前进"。离散 codebook 你把 code 5 和 code 7 average 出什么？code 6 吗？没意义。

**3. 不能 compose。** Figure 5 演示把两个 latent action 平均一下，得到一个新 action 是两者功能的融合。比如"伸手" + "抓" 平均成"伸手抓"。离散 codebook 没法做这种插值。

这就解释了为什么 AdaWorld 能做 paper 里 Section 2.3 那几个 trick：
- Action transfer：直接拿源 video 的 latent 接到新 context
- Action composition：两个 action latent 平均出新 action
- Action creation：聚类所有 latent 得到任意数量的控制选项（Table 9）

这些都是 continuous 空间的福利。

---

## World Model：把 SVD 改成"下一帧预测器"

### 为什么不能直接用 latent action decoder？

理论上 decoder 已经会 $f_t + \tilde a \to f_{t+1}$ 了，直接拿来当 world model 不行吗？

不行。Paper Section 2.2 明说：decoder 单步预测是 coarse 的，连续 rollout 几步质量就崩。原因——它是从 VAE 训出来的，优化的是"平均重建误差"，不是"高保真预测"。

所以 AdaWorld 在 decoder 之外又架了一个真正的 world model，基于 **Stable Video Diffusion (SVD)**（Blattmann et al., 2023, <https://stability.ai/research/stable-video-diffusion>）。

### SVD 是什么？

Stability AI 的视频生成模型，latent diffusion 架构，1.5B 参数。原本设计是给一段条件图生成 25 帧短视频。

AdaWorld 把它改造成 **autoregressive 单帧预测器**：
1. 原本一次 denoise 整段视频，改成**一次只 denoise 一帧**
2. 把 latent action $\tilde a$ 拼到 timestep embedding 和 CLIP image embedding 上（一个 global condition）
3. 历史最多 6 帧通过 SVD 的 image encoder 编码，concatenate 到当前要预测的 noise latent 上
4. 训练时随机采样 1-6 帧历史 + 加噪声增强

Diffusion loss 很标准（Eq. 3）：

$$
\mathcal{L}_{\text{pretrain}} = \mathbb{E}_{x_0, \epsilon, t}\left[\|x_0 - \hat{x}_0(x_t, t, c)\|^2\right]
$$

变量：
- $x_0$：干净的目标帧
- $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$：加噪版本
- $\hat{x}_0(x_t, t, c)$：模型直接预测 $x_0$（EDM 风格，Karras et al., 2022）
- $c$：所有 condition 的合集，包括历史帧 + latent action $\tilde a$

### Noise augmentation 这招很妙

训练时给历史帧加 0.0~0.7 随机噪声，推理时不加。

为什么？因为 autoregressive rollout 时，**第 5 步用的"历史"是第 1-4 步模型自己预测出来的**，不是 clean 的。如果训练时只见过 clean history，推理时一遇到自己生成的 imperfect history 就崩。训练时见过噪声，模型就皮实了。

这个 trick 来自 GameNGen (Valevski et al., 2025, <https://gamengen.github.io/>)，是 world model 工程化的关键小细节。

---

## 三种用法：transfer、adapt、compose

预训练完，AdaWorld 三个杀手锏：

### 用法 1：零训练 Action Transfer

给一段 demo video（比如某人推杯子）：
1. 用 latent action encoder 抽出 latent action 序列 $\tilde a_{1:T}$
2. 给一个新场景的初始帧
3. 用 $\tilde a_{1:T}$ 当 condition，autoregressive 生成新场景下的 rollout

**完全不更新任何权重。** 你给它看一遍推杯子，它换到厨房、客厅、办公室都能演同样的"推"动作。

Table 1 的 LIBERO Human eval：AdaWorld 70.5%，action-agnostic baseline 0%。这个 0 vs 70 是质变。

### 用法 2：小样本 Adaptation

新环境有具体 action label 怎么办？比如 Habitat 有 4 个离散 action（前进、后退、左转、右转）。

做法：
1. 每个 action 收集 100 个 transition 样本
2. 用 encoder 提取每个样本的 latent action
3. **同 action 的 100 个 latent 直接求平均**，得到 4 个 averaged latent
4. 用这 4 个 averaged latent 初始化 action embedding table
5. Finetune 整个 model 800 步

对于连续 action（nuScenes 方向盘），加个 2 层 MLP 把 raw action 映射到 latent action interface，finetune 3K 步，30 秒搞定。

Table 2 显示 finetune 800 步后，AdaWorld 在 4 个 unseen environment 上 PSNR 全胜。Figure 6 的曲线更直观——AdaWorld 一开始就比 baseline 高，finetune 几步就把 baseline 甩开。

最骚的是 Table 3 这一行：**AdaWorld w/o finetune**（不更新权重，只用 averaged latent）success rate 44.83%，已经超过 action-agnostic **w/ finetune**（26%）。这是"latent action 本身就是个能用的 control interface"的硬证据。

### 用法 3：Action Composition

两个 latent action 求平均，得到一个新 action 是两者融合。Figure 5 的 demo 很直观。

更进一步——把训练集所有 latent action 拿来 K-means 聚类，聚成 5 个、6 个、10 个 cluster 都行，每个 cluster 中心就是一个"控制选项"。Table 9 的 ∆PSNR 显示这些自动发现的 action 都有清晰不同的效果。

这意味着你**可以自定义 action 数量**，不绑定固定 codebook。Genie 训完就锁死 8 个 action，AdaWorld 可以根据任务需要任意切分 latent 空间。

---

## 一个被低估的工程 trick：Biased Action Sampling

Appendix A.1 里藏了个不起眼但很聪明的 trick。

采集训练数据时，如果用 uniform random policy（每个 action 等概率采样），agent 在原地反复横跳，探索的 scene 多样性很差。Genie 就是这么干的。

AdaWorld 的做法：**短期内偏重某个 action，过一段再切换**。比如前 50 步主要按"前进"，后 50 步主要按"右转"。

Figure 8 一目了然——uniform 采样下 agent 局限在小区域，biased 采样下 agent 走出长路径，看到各种 scene。

这是个很老的 exploration heuristic（curiosity-driven exploration 那一脉都用），但在 world model 数据采集场景里用得少。简单粗暴但有效。

---

## 它最让我"哇"的几个发现

### 发现 1：Robot 视频帮 Game 视频 generalize

Table 5：训练数据只用 OpenX（真实机器人）→ Procgen PSNR 25.51；只用 Retro（2D 游戏）→ 26.43；**两者混合 → 26.62**。

也就是说，**真实世界的机器人操作视频**居然帮助模型在 **2D 像素游戏**上表现更好。这暗示 latent action 学到的不是"像素级别的 transition pattern"，而是更抽象的"动作-效果"因果结构，这种结构跨 modality 共享。

这是 emergent cross-domain transfer 的证据。如果你相信 world model 路线，这是个鼓舞人心的信号。

### 发现 2：Averaging 为什么 work？

最让我困惑也最让我兴奋的点：同一 action label 下 100 个不同 transition 的 latent，平均一下居然得到稳定的 action 表示。

按理说"前进"撞墙和"前进"在空地，next frame 完全不一样，对应的 latent 应该差异很大。但平均居然 work。

我的解释：**encoder 学到的不是"下一帧长什么样"，而是"我想做什么"**。它学到的是 action intent，类似 RL 里的 action embedding。Decoder 在推理时根据 current state + intent 合成具体 pixel delta。

如果这个解释对，那 latent action 实际上是 **action 的 semantic embedding**。这是个很深的 insight——意味着从纯视觉观察里，能浮现出类似语言级别的 action 概念。

### 发现 3：零训练已经超过有训练的 baseline

Table 3 那一行 AdaWorld w/o finetune（44.83%）vs Act-agnostic w/ finetune（26%）。这不是量变是质变——**说明 action-aware pretraining 学到的 control interface 本身就是个好东西，不需要权重更新就能用**。

---

## 我觉得可以 push 的方向

### 1. Latent action 的信息论分析

32 维连续高斯，理论上 mutual information 上限是 512 bit 量级。复杂 3D 环境（多物体物理交互）一个 transition 可能需要更多信息。Paper 没做 information-theoretic upper bound 分析。

一个简单实验：训一个 classifier 从 latent action 反推 context（比如 frame 的颜色直方图），看 accuracy 接不接近 chance。如果接近 chance，说明 latent 真的 context-invariant；如果远高于 chance，说明 latent 仍然 leak context，transfer 早晚出问题。

### 2. Condition injection 机制太粗

Paper 说 latent action "concatenated with timestep embedding and CLIP image embedding"。这是 **global** condition。但"左转"这个动作在画面上应该 spatially 不均匀——左侧变化大、右侧变化小。Global condition 区分不出空间。

更精细的做法：用 cross-attention 把 latent action 注入到 UNet 各层，让它 spatially 影响 generation。Paper 没做这个 ablation。

### 3. Long-horizon 是真硬伤

Memory 只有 6 帧，rollout 超过 6 帧就要吃自己的 prediction 当 history。Paper Section 4 自己承认这是 limitation。

可能的解法我列几个：
- **Diffusion Forcing** (Chen et al., 2024, <https://boyuan.space/diffusion-forcing/>)：把 next-token prediction 和 full-sequence diffusion 统一，variable horizon
- **Rolling Diffusion** (Ruhe et al., 2024)：滑动窗口式 denoising
- **Latent state compression**：用 perceiver / tokenizer 压缩 long history 到固定大小

### 4. Real-time 还没解决

5 步 DDIM + CFG 1.05，单帧几百 ms。10 Hz 控制跟不上。

The Matrix (Feng et al., 2024, <https://thematrix-2.github.io/>) 用 consistency model 做实时控制是个方向，AdaWorld 还没集成 distillation。

### 5. Distractor 的影响

Nikulin et al., 2025（<https://arxiv.org/abs/2502.00379>）指出 latent action learning 在有 distractor（背景独立运动的物体）时会出问题。AdaWorld 的 bottleneck 设计理论上能 mitigate（distractor 不可预测，平均下来不进 latent），但没显式 test。

一个 stress test：在视频背景里加 random 噪声物体乱动，看 latent action 的 transfer PSNR 下降多少。

---

## 在更大的 landscape 里看

AdaWorld 在我脑子里跟几条线连在一起：

**线 1：Latent Action 谱系**
- Rybkin 2019：最早从 video 学 action rep，但 single env
- Genie 2024：scale 到 2D platformer，但离散 codebook
- LAPA 2025（<https://latentactionpretraining.github.io/>）：scale latent action pretraining，但用于 policy
- PreLAR 2024（<https://arxiv.org/abs/2410.13917>）：world model pretraining with learnable action，跟 AdaWorld 最像
- AdaWorld：把 latent action 当 world model 的 transferable control interface

**线 2：World Model 谱系**
- DayDreamer 2022（<https://daydreamer-tools.github.io/>）：real robot dream-based RL
- IRIS 2023、DIAMOND 2024（<https://diamond-wm.github.io/>）、GameNGen 2025（<https://gamengen.github.io/>）：游戏 world model
- Cosmos 2025（<https://research.nvidia.com/labs/toronto-ai/cosmos/>）：NVIDIA 物理 AI foundation model
- Vista 2024（<https://vista-web.github.io/>）、GAIA-1 2023：driving world model

**线 3：VLA / Robotics Foundation Model**
- OpenVLA 2024（<https://openvla.github.io/>）
- 3D-VLA 2024（<https://3d-vla.github.io/>）
- Open X-Embodiment 2024（<https://robotics-transformer-x.github.io/>）：dataset
- AgiBot World 2025（<https://agibot-world.com/>）

AdaWorld 站在 latent action 和 world model 两条线的交叉点。它的核心 contribution 是 framing——把 latent action 从 "policy 的中间产物" 重新定位成 "world model 的 transferable interface"。

---

## 一句话总结

AdaWorld 让我想到你（Karpathy）讲过的一个观点：**world model 是真正的 imagination engine**——agent 在 mind 里 simulate "如果做 X 会怎样" 再决定。

AdaWorld 的 latent action + autoregressive diffusion，是在显式建模 **counterfactual imagination**：

$$
p(f_{t+1} | f_t, \text{do}(\tilde a))
$$

注意是 $\text{do}(\tilde a)$ 而不是 conditioning。如果 latent action 真的捕获 "action intent" 而非 "observed transition"，那 AdaWorld 实际上在做 do-calculus 的近似——这就是它能 transfer 到新 context 的根本原因：action intent 不依赖 context。

这个 framing 比纸面上的 SOTA 数字更让我兴奋。它指向一个 future：**从观察中浮现出可组合、可迁移、可插值的 action 概念**，本质上是在学一种 emergent action language。

如果 2026 年有人把这条路线 scale 到 10B+ 参数，加 1B 小时跨域视频，我赌会出现一些 qualitatively new 的东西——比如模型自发学会"先抓再放"这种组合 action，或者跨 modality 的 zero-shot action transfer（看人推杯子就会指挥机器人推）。

Paper 主页：<https://adaptable-world-model.github.io/>

要不要我再展开讲：
- (a) latent action averaging 为什么 work 的理论猜想
- (b) AdaWorld vs PreLAR vs LAPA 的细节对比
- (c) 把 AdaWorld 推到机器人 manipulation 的工程挑战
- (d) 10B 参数 scale 上去会 emerge 什么

挑一个？

---

# AdaWorld 深度讲解

Andrej，这篇 paper 我读完之后挺兴奋的，因为它触及了 world model pretraining 里一个非常本质的 tension：**video data 里到底有没有 action 信息，以及怎么把它"挤"出来变成可迁移的控制接口**。下面我会把整个方法拆开讲，包括一些 paper 没明说但你能从公式里读出来的设计意图。

Paper link: <https://adaptable-world-model.github.io/>  
arXiv (通过项目页可找): AdaWorld: Learning Adaptable World Models with Latent Actions  
相关 baseline 参考：
- Genie (Bruce et al., 2024): <https://sites.google.com/view/genie-2024>
- LAPA (Ye et al., 2025): <https://latentactionpretraining.github.io/>
- SVD (Blattmann et al., 2023): <https://stability.ai/research/stable-video-diffusion>
- iVideoGPT (Wu et al., 2024): <https://thuml.github.io/iVideoGPT/>
- DIAMOND (Alonso et al., 2024): <https://diamond-wm.github.io/>
- PreLAR (Zhang et al., 2024): <https://arxiv.org/abs/2410.13917>

---

## 1. Motivation：把"观察学习"建模化

Paper 的核心 motivation 在 intro 里讲得很清楚：人类能从有限交互中估计 action 效果（Ha & Schmidhuber, 2018; Poggio & Bizzi, 2004），是因为我们有从大量观察中学到的 **context-invariant action representation**。这个想法对应到 neuroscience 里的 mirror neuron system (Rizzolatti et al., 1996)。

把这个 intuition 翻译到 ML：

- 现有 paradigm 1：action-agnostic video pretraining（Seo et al., 2022; Mendonca et al., 2023; Wu et al., 2023; Agarwal et al., 2025; He et al., 2025）—— 学到 visual dynamics，但 **control interface 是空的**，到新环境要重训。
- 现有 paradigm 2：action-labeled pretraining（VPT, Baker et al., 2022）—— 需要昂贵标注，且 action format 跨环境不统一。

AdaWorld 的 paradigm 3：**从 video 自监督提取 latent action → 用它做 condition 预训练 world model → 新环境只需"对齐 latent action interface"**。这是把"观察获得 action 概念"这件事显式建模出来。

我觉得这个 framing 本身就是 paper 的贡献。它把 latent action 从 "behavior cloning 的中间产物"（LAPA, Moto, IGOR）重新定位成 **"world model 的可迁移 control interface"**。

---

## 2. Latent Action Autoencoder：information bottleneck 的双重作用

这是整个方法最精妙的部分。让我把公式拆开。

### 2.1 信息流

```
f_t, f_{t+1}  ──[Encoder φ]──>  q_φ(ã | f_{t:t+1}) = N(μ_ã, σ_ã)
                                          │  sample ã
                                          ▼
                            f_t, ã  ──[Decoder θ]──>  f̂_{t+1}
```

Loss (Eq. 2)：

$$
\mathcal{L}_{\theta,\phi}^{pred}(f_{t+1}) = \underbrace{\mathbb{E}_{q_\phi(\tilde a \mid f_{t:t+1})} \log p_\theta(f_{t+1}\mid \tilde a, f_t)}_{\text{reconstruction}} - \beta \underbrace{D_{KL}\big(q_\phi(\tilde a \mid f_{t:t+1})\,\Vert\, p(\tilde a)\big)}_{\text{KL to prior}}
$$

变量含义：
- $\theta, \phi$：decoder 和 encoder 参数
- $q_\phi(\tilde a \mid f_{t:t+1})$：encoder 输出的 posterior，参数化为 $(\mu_{\tilde a}, \sigma_{\tilde a})$，对角高斯
- $p_\theta(f_{t+1} \mid \tilde a, f_t)$：decoder 给出 frame 的 likelihood（pixel space 重建）
- $p(\tilde a)$：prior，标准正态 $\mathcal{N}(0, I)$
- $\tilde a \in \mathbb{R}^{32}$：latent action，**只有 32 维**（这个数字关键）
- $\beta = 2 \times 10^{-4}$：非常小的 KL weight

### 2.2 Bottleneck 在哪里？为什么这样设？

这里有两层 bottleneck 同时作用，我觉得是 paper 最 subtle 的地方：

**第一层：dimension bottleneck。** Latent action 只有 32 维 continuous。对比一张 256×256×3 的 frame（约 19.6 万维），32 维根本塞不下整张图。所以 encoder 必须选择"哪些信息最值得保留"。在 interactive video 里，相邻 frame 的 dominant variation 通常是 agent 的 action（Rybkin et al., 2019; Menapace et al., 2021），所以 latent 会优先编码 action-related transition。

**第二层：KL regularization。** 这一层防止 latent 偷偷塞入 context-specific 的 spurious pattern。比如 encoder 可能学到"如果 frame 里有红色物体，就编码红色位置"——这种 shortcut 在同 context 内有用，但跨 context 就崩了。KL term 把 posterior 拉向 isotropic prior，强制 latent 各维独立、分布平滑。

这两层配合，导致 latent action 既有一定表达力（能区分不同 transition），又有 context-invariance（不绑定具体 pixel pattern）。

### 2.3 为什么用小 β 而不是标准 β-VAE？

这点 paper 没有展开但很关键。经典 β-VAE (Higgins et al., 2017) 是 **大 β** 用来强化 disentanglement。AdaWorld 反过来用 **极小 β** ($2\times10^{-4}$)，目的是放松 posterior 约束，让 latent 有足够表达力覆盖 diverse transitions。

Paper Section 2.1 末段说：标准 VAE "imposes a strong constraint on posterior distributions"，会导致 latent action "struggle to express diverse transitions"；但完全去掉 KL 又会 "compromise the disentanglement ability"。所以 β 是 expressiveness vs disentanglement 的 trade-off knob。

Figure 7 的 UMAP 可视化很直观地展示了这个 trade-off：
- 大 β（左图 mindset）：不同 action 聚成一团，但跨 environment 同 action overlap 多 → transferable
- 小 β（右图 mindset）：action 间 separable，但跨 environment 同 action 分散 → expressive 但不 transferable

默认 $\beta = 2 \times 10^{-4}$ 是 sweet spot。这种 sensitivity 在实际工程里要小心的，β 调大一点 transfer 就崩，调小一点就 leak context。

### 2.4 为什么不用 VQ-VAE 离散 latent action？

这是 AdaWorld 对比 Genie 的核心 architectural choice。Paper 给出三个理由：

1. **Expressiveness**：8 个离散 code（Genie 的设定）覆盖不了连续 action space，比如驾驶的微小方向盘变化。
2. **Compositionality**：连续空间可以平均、插值，得到新 action（Figure 5）。离散 codebook 平均没意义。
3. **Adaptability**：连续 latent 可以用 MLP 从 raw action 拟合；离散 code 需要查找表，新 action 要新 code，不灵活。

实验 Table 1 里 discrete cond. baseline 在 LIBERO 上 Human eval 只有 3.5%，AdaWorld 是 70.5%，差距巨大。这印证了离散化丢信息严重。

---

## 3. 架构细节：Spatiotemporal Transformer

Encoder 是 spatiotemporal Transformer（参考 Bruce et al., 2024），关键设计：

- 两帧 $f_t, f_{t+1}$ 分成 16×16 patches，project 成 embeddings
- 两个 learnable tokens $a_{t:t+1}$ concatenated 进去（类似 [CLS] token，但有两个，分别对应两帧）
- L 个 block，每个 block：
  - **Spatial attention**：单帧内所有 patch 互相 attend（建模 within-frame context）
  - **Temporal attention**：跨帧 **同位置 patch** attend，加 **rotary embedding**（Su et al., 2024）指示 $t \to t+1$ 因果方向
  - FFN
- 最后丢掉所有 patch token，只把 $a_{t+1}$ project 成 $(\mu_{\tilde a}, \sigma_{\tilde a})$

这里 temporal attention 用 RoPE 表示因果是巧妙的小技巧。Patch-level 的 temporal attention 让 model 显式 track "同一个 spatial location 在两帧间怎么变"，这正是 action 最容易在 pixel space 留下痕迹的地方。

Decoder 是纯 spatial Transformer，输入 $\tilde a$ 和 $f_t$ 的 patches，输出 $f_{t+1}$。

总参数：500M（16 encoder + 16 decoder blocks, 1024 channels, 16 heads）。

---

## 4. World Model：基于 SVD 的 latent diffusion

这里 paper 的设计稍微 hacky 但合理。

### 4.1 为什么不用 latent action decoder 直接当 world model？

Paper Section 2.2 明说：latent action decoder 只做"单步 coarse prediction"，几步 rollout 后 quality 严重退化。所以独立建一个 **diffusion-based** world model 保证生成质量。

这是 world model 的老问题：autoregressive pixel prediction 会 compound error。Diffusion model 的 iterative refinement 缓解这个问题，但代价是慢。

### 4.2 SVD 改造细节

- Base model: Stable Video Diffusion (Blattmann et al., 2023)，latent diffusion + EDM framework (Karras et al., 2022)
- **改造1**：每次只 denoise **一帧** 而不是整段 video（这是 autoregressive world model 必须的）
- **改造2**：latent action $\tilde a$ 拼到 timestep embedding 和 CLIP image embedding 上（c 拼接的方式）
- **改造3**：history frames（最多 6 帧）通过 SVD image encoder 编码，concatenate 到 noise latent 上
- **改造4**：训练时随机选 1-6 帧 history + 随机 noise augmentation level（0.0-0.7）

Diffusion loss (Eq. 3)：

$$
\mathcal{L}_{\text{pretrain}} = \mathbb{E}_{x_0, \epsilon, t}\big[\|x_0 - \hat{x}_0(x_t, t, c)\|^2\big]
$$

变量：
- $x_0$：clean target frame
- $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$：noised version，$\bar\alpha_t$ 是 noise schedule
- $t$：diffusion timestep
- $\epsilon \sim \mathcal{N}(0, I)$：noise
- $\hat{x}_0(x_t, t, c)$：model 直接预测 $x_0$（EDM 风格，不是预测 noise）
- $c$：condition = (history frames, $\tilde a$)

参数量 1.5B trainable。

### 4.3 Noise augmentation 的妙用

训练时给 history frames 加噪声，推理时不加。这看起来矛盾，其实是为了 **缓解 train-test mismatch**：autoregressive rollout 时，之前预测的 frame 不是 perfect clean 的，有 compounding error。训练时见过 noisy history，模型就 robust 到 imperfect input。这个 trick 来自 He et al., 2022; Valevski et al., 2025（GameNGen）。

---

## 5. Adaptation：三种使用模式

这是 AdaWorld 的 payoff。Action-aware pretraining 的好处全在这里。

### 5.1 Action Transfer（零训练）

给一段 demo video：
1. 用 latent action encoder 提取 latent action sequence $\tilde a_{1:T}$
2. 给 new context 的初始帧 $f_0^{new}$
3. 用 $\tilde a_{1:T}$ 作为 condition，autoregressive 生成 new context 下的 rollout

**完全不需要训练。** 这是 Figure 4 的核心展示。

Intuition：latent action encoder 学到的是"action 概念"（如"把东西往左推"），context-invariant。所以从源 video 提取的"左推 action"可以直接接到新 scene 的初始帧上。

### 5.2 World Model Adaptation（小样本微调）

两种情况：

**Discrete action**：N 个 action，每个收集 100 个 transition samples。对每个 action $a_i$，用 encoder 提取所有 sample 的 latent actions $\{\tilde a_i^{(k)}\}$，**直接平均** $\bar{\tilde a}_i = \frac{1}{100}\sum_k \tilde a_i^{(k)}$。用这 N 个 averaged embedding 初始化 action embedding table，然后 finetune 整个 model 800 步。

**Continuous action**（如 nuScenes 的方向盘位移）：加 2-layer MLP 把 raw action 映射到 latent action interface。MLP 用 action-latent action pairs finetune 3K 步，30 秒。

这里 latent action 的 continuity 是关键：连续空间下，"averaging" 是合法操作（同 action 的 latent 在空间里应该 cluster 在一起，中心代表 action intent）。离散 codebook 不能 average。

### 5.3 Action Composition（创造新 action）

Figure 5：两个 latent action 平均 → 新 action，融合两者功能。这说明 latent space 是 semantic continuous 的。

更高级：clustering 所有训练 video 的 latent actions，按需生成 K 个控制选项（Table 9）。这种 flexibility 是 discrete codebook 给不了的。

---

## 6. 实验：哪些数据点最有说服力

### 6.1 Table 1 Action Transfer

| Method | LIBERO FVD↓ | LIBERO Human↑ | SSv2 FVD↓ | SSv2 Human↑ |
|---|---|---|---|---|
| Act-agnostic | 1545.2 | 0% | 847.2 | 1% |
| Flow cond. | 1409.5 | 2% | 702.8 | 10.5% |
| Discrete cond. | 1504.5 | 3.5% | 726.8 | 21.5% |
| **AdaWorld** | **767.0** | **70.5%** | **473.4** | **61.5%** |

Human eval 0% → 70.5% 是 qualitative jump。Act-agnostic 0% 完全符合预期：它根本没学过 action condition，所以"transfer"等于乱生成。但 flow cond. 只有 2% 反而让我意外——optical flow 理论上带 action info，但 flow 包含太多 context-dependent pixel motion（相机抖动、物体自然摆动），所以 transfer 不干净。这反向验证了 AdaWorld bottleneck 的 disentanglement 价值。

### 6.2 Table 3 Procgen Planning

| Method | Heist | Jumper | Maze | CaveFlyer | Avg |
|---|---|---|---|---|---|
| Random | 19.33 | 22.00 | 41.33 | 22.00 | 26.17 |
| Act-agnostic | 20.67 | 20.67 | 39.33 | 23.33 | 26.00 |
| AdaWorld w/o finetune | 38.67 | 68.00 | 41.33 | 31.33 | 44.83 |
| AdaWorld w/ finetune | **66.67** | 58.67 | **68.00** | 33.33 | **56.67** |
| Q-learning | 22.67 | 47.33 | 4.67 | 34.00 | 27.17 |
| Oracle | 86.67 | 77.33 | 84.67 | 74.00 | 80.67 |

两个观察：
1. **AdaWorld w/o finetune** (44.83%) 已经超过 act-agnostic + finetune (26%)，证明 latent action 本身就能初始化出可用的 control interface，无需更新权重。这是 zero-shot transfer 的强证据。
2. Q-learning 在 Maze 上崩到 4.67%——state space 太大，quantized image Q-table 根本覆盖不过来。Model-based planning（MPC）优势就出来了。

### 6.3 Table 5 Data Diversity

| Training Data | Procgen PSNR↑ | Procgen LPIPS↓ |
|---|---|---|
| OpenX only | 25.51 | 0.318 |
| Retro only | 26.43 | 0.250 |
| Retro+OpenX | **26.62** | **0.234** |

OpenX 是 real-world robot 视频，Procgen 是 2D 像素 game。**Robot video 帮 game video generalize**。这是 emergent cross-domain transfer，说明 latent action 学的是真正 abstract 的 transition 概念，跨 modality 仍然 share structure。

### 6.4 Table 6 Generality

iVideoGPT + AdaWorld 在 BAIR 上 PSNR 16.59 → 17.40，LPIPS 0.220 → 0.204。证明 action-aware pretraining 范式可以 plug-in 到其他 world model 架构。

---

## 7. Biased Action Sampling：被低估的小 trick

Appendix A.1 提到一个数据生成的 trick 值得单独说。

标准做法：random policy 每 step 均匀采样 action（Genie 的做法）。问题：environment state 短期内反复横跳，scene exploration 浅。

AdaWorld 做法：**短期内偏向某个 action，过一段再切换**。这模拟"intentional behavior"，让 agent 在一个方向上探索更远。

Figure 8 可视化明显——biased sampling 探索到的 scene 远比 uniform 多。这种 exploration heuristic 在 RL 文献里其实很经典（如 episodic curiosity, RND 等），但 world model 数据收集里用得少。这是个值得复用的小 idea。

---

## 8. 我的几个 Critique 和 Open Questions

这部分是我以你（Karpathy）视角的思考：

### 8.1 Latent dim 32 + 小 β，bottleneck 真的够吗？

32 维 continuous Gaussian，理论上 mutual information 上界是 $\frac{1}{2}\log(1 + \text{SNR})$ 量级。32 维 × 16 bit ≈ 512 bit per transition。对于复杂 3D 环境，一个 transition 可能涉及多物体运动、相机变化、物理交互。512 bit 够吗？

Paper 没做 information-theoretic analysis。UMAP 显示同 action 聚类，但聚类 ≠ 信息没有 leak。一个更严格的检验：训练一个 classifier 从 latent action 反推 context（比如 frame 的颜色统计），看能达到什么 accuracy。如果接近 chance，才算真 disentangled。

### 8.2 Latent Action 平均为什么 work？

这其实最 surprising。同一 action label（比如"前进"）在不同 current state 下产生的 transition 是不同的——前进撞墙和前进在空地，next frame 完全不一样。但 paper 说 averaging latent actions across samples 仍然得到 consistent action embedding。

我的解释：encoder 学到的可能不是 "exact next frame delta"，而是 **"action intent"**——一个抽象的 "想往这个方向移动" 表示。Decoder 在 inference 时根据 current state + intent 合成具体 pixel delta。

如果是这样，latent action 实际上是 **action 的 semantic embedding**，不是 transition 的 literal encoding。这跟 action embedding 在 RL 里的角色其实是一样的——只是从 video 自监督学出来了。

但这也意味着：如果 action 的"意图"在不同 environment 下定义不同（比如"jump"在 platformer 和 FPS 含义不同），transfer 可能失败。Paper 没显式 test 这种 cross-domain action semantic shift。

### 8.3 SVD 的 condition 注入方式

Paper 说 latent action "concatenated with both the timestep embedding and the CLIP image embedding"。但 SVD 的 condition 机制原本是 cross-attention 到 text/CLIP embedding。Latent action 直接 concat 到 embedding 上，是改了 condition vector，还是改了 cross-attention 的 K/V？

如果是简单 concat 到 global embedding，那 action info 通过 global token 影响整个 UNet，可能不够 spatially precise。比如"左转"应该影响画面左侧更多，但 global condition 不区分空间。Cross-attention to action 可能更好。

Paper 没给 ablation。这是个潜在改进点。

### 8.4 Long-term Rollout

Paper 自己 admit 这是 limitation。Memory 只有 6 帧，rollout 超过 6 帧就要靠自己的 prediction 当 history。这是 world model 通用问题。

可能的解法（paper 提到但没做）：
- **Diffusion Forcing** (Chen et al., 2024a): <https://boyuan.space/diffusion-forcing/> 把 next-token prediction 和 full-sequence diffusion 统一，可以处理 variable horizon
- **Rolling Diffusion Models** (Ruhe et al., 2024)
- **Savta-ish memory**：longer memory with compression

### 8.5 Action vs Environment Dynamics 的 disentanglement

Latent action 提取的是 "frame 间 dominant variation"。但 variation 可能来自：
1. Agent action（我们想要的）
2. Environment dynamics（重力、对手 AI、NPC）
3. Camera motion
4. Distractor（背景里独立运动的物体）

Nikulin et al., 2025 (<https://arxiv.org/abs/2502.00379>) 指出 latent action learning 在有 distractor 时需要 supervision。AdaWorld 在 OpenX + Retro + MiraData 等混合数据上训练，distractor 不可避免。

但 AdaWorld 的 bottleneck 设计可能恰好 mitigate 这点：32 维只能编码最 systematic 的 transition source，random distractor 因为不可预测，平均下来不会进 latent。这是个 emergent robustness，但没显式验证。

### 8.6 Real-time Inference

5 步 DDIM + CFG 1.05，单帧 generation 估计几百 ms。10 Hz 控制根本跟不上。

Paper 提到 distillation (Yin et al., 2025: <https://lfyin2024.github.io/projects/cogvideox-fast/>) 和 sampling acceleration 是 future work。但这是个硬伤——world model 必须快到能 rollout 数千次做 planning。The Matrix (Feng et al., 2024) 用 consistency model 试图解决，AdaWorld 还没集成。

---

## 9. 相关 Work 联想图谱

让我把 AdaWorld 放在更大的 landscape 里：

### 9.1 Latent Action 谱系

- **Rybkin et al., 2019** ("Learning What You Can Do before Doing Anything", ICLR): <https://orybkin.github.io/animal/> — 最早从 video 学 action representation 的一批，但 single environment
- **Schmeckpeper et al., 2020** (Learning Predictive Models From Observation and Interaction, ECCV): <https://schmeckpeper.github.io/> — 类似 idea
- **Edwards et al., 2019** (Imitating Latent Policies from Observation, ICML) — latent policy from observation
- **Genie (Bruce et al., 2024)**: 离散 8 codes，scale 到 2D platformer
- **LAPA (Ye et al., 2025)**: <https://latentactionpretraining.github.io/> — latent action pretraining at scale，但用于 policy
- **PreLAR (Zhang et al., 2024)**: <https://arxiv.org/abs/2410.13917> — World Model Pre-Training with Learnable Action Representation，与 AdaWorld 最像
- **Moto (Chen et al., 2024c)**: <https://arxiv.org/abs/2412.04445> — Latent Motion Token
- **IGOR (Chen et al., 2024b)**: <https://arxiv.org/abs/2411.00785> — Image-Goal Representation as atomic control unit

AdaWorld 与 PreLAR 的区别可能是 AdaWorld 用 continuous + SVD base + 显式 action transfer 评测；PreLAR 更偏 RL 内的 pretraining。

### 9.2 World Model 谱系

- **DayDreamer (Wu et al., 2022)**: <https://daydreamer-tools.github.io/> — dream-based RL on real robot
- **IRIS (Micheli et al., 2023)**: <https://arxiv.org/abs/2210.15002> — Transformer world model on Atari
- **DIAMOND (Alonso et al., 2024)**: <https://diamond-wm.github.io/> — Diffusion world model on Atari
- **GameNGen (Valevski et al., 2025)**: <https://gamengen.github.io/> — DOOM diffusion world model
- **GameGen-X (Che et al., 2025)**: <https://arxiv.org/abs/2410.13734>
- **The Matrix (Feng et al., 2024)**: <https://thematrix-2.github.io/> — real-time moving control
- **Cosmos (Agarwal et al., 2025)**: <https://research.nvidia.com/labs/toronto-ai/cosmos/> — NVIDIA 物理 AI world foundation model
- **GAIA-1 (Hu et al., 2023)**: <https://arxiv.org/abs/2309.17080> — driving world model
- **Vista (Gao et al., 2024)**: <https://vista-web.github.io/> — driving world model
- **IRASim (Zhu et al., 2024)**: <https://irsim.github.io/> — robot action simulator
- **iVideoGPT (Wu et al., 2024)**: <https://thuml.github.io/iVideoGPT/> — interactive VideoGPT，AdaWorld 用来做 generality 实验

### 9.3 VLA / Action Foundation Model 谱系

- **OpenVLA (Kim et al., 2024)**: <https://openvla.github.io/> — open-source VLA
- **3D-VLA (Zhen et al., 2024)**: <https://3d-vla.github.io/> — 3D vision-language-action generative world model
- **AgiBot World (Bu et al., 2025)**: <https://agibot-world.com/> — large manipulation dataset
- **Open X-Embodiment (O'Neill et al., 2024)**: <https://robotics-transformer-x.github.io/> — robot dataset，AdaWorld 用作 training data

### 9.4 Video Pretraining → Action

- **VPT (Baker et al., 2022)**: <https://openai.com/research/video-pre-training> — pseudo-label Minecraft
- **SlowFast-VGen (Hong et al., 2025)**: <https://slowfast-vgen.github.io/> — long video generation
- **VideoWorld (Ren et al., 2025)**: knowledge from unlabeled video

---

## 10. 一些更深的思考：从 World Model 到 "Imagination Engine"

Andrej，从你的视角我想多谈一点。AdaWorld 让我联想到你之前在 Eureka Labs / world model 讨论里讲的一个观点：**world model 是真正的 "imagination engine"**——agent 能在 mind 里 simulate 一下"如果我做 X 会怎样"，再决定行动。

AdaWorld 的 latent action + autoregressive diffusion 框架，其实是在尝试建模 **counterfactual imagination**：

$$
p(f_{t+1} \mid f_t, \text{do}(\tilde a))
$$

注意是 $\text{do}(\tilde a)$ 而不是 conditioning on $\tilde a$。区别在于：
- Conditioning: 假设观测到 $\tilde a$，frame 会怎么变
- Do: 强制干预执行 $\tilde a$，frame 会怎么变

如果 latent action 学的是 "action intent"（而非 "observed transition"），那 AdaWorld 实际上是在做 do-calculus 的近似。这让它能 transfer 到新 context——因为 action intent 不依赖 context，只依赖 "我想做什么"。

这个 framing 让我想起 Tassa et al. 的 MuZero / Dreamer 路线，但那些是在 latent state space 做 planning，AdaWorld 是在 pixel/visual space 做。后者更通用，但更贵。

### 10.1 "Latent Action 是 emergent language 吗？"

如果 latent action 真的捕获了 "推、抓、转、跳" 这种 abstract 概念，那它实际上是一种 **emergent action language**——从 visual observation 中自监督浮现出来的。

这和 LLM 里 token 自然 emerge 出 semantic 有点像。但 LLM 的 token 是离散的，AdaWorld 的 latent action 是 continuous，更适合控制（连续 action space 天然 fit）。

可能 future work 是把 latent action **量化**但**保留 continuous 优势**——比如 product quantization 或 sparse coding，既可组合又可解释。

### 10.2 Transferability vs Expressiveness 的 fundamental trade-off

AdaWorld 的 UMAP 显示：β 调大 → transferable 但不 expressive；β 调小 → expressive 但不 transferable。这其实是 representation learning 的一个 fundamental tension：

- **要 transferable**：representation 必须丢掉 context-specific info
- **要 expressive**：representation 必须保留足够 info 区分细节

理论上这两者可同时达到，如果存在一个 "disentangled representation" 把 context 和 action 在 latent space 显式分开。但 β-VAE 只能"软"约束，不能完全 disentangle。

可能的改进：
- **Contrastive learning**：同 action 跨 context 的 latent 拉近，不同 action 推远
- **Adversarial disentanglement**：训练一个 classifier 从 latent 推 context，gradient reversal 让 latent 抗 context
- **Multi-task latent**：latent 显式分成 action subspace 和 context subspace

---

## 11. 复现 & 工程建议

如果你想实验这个方向：

1. **Latent action autoencoder 先小规模验证**：在 BAIR / RoboNet 上训练一个小 ViT + 32 dim latent，先看 UMAP 是否能 cluster action。这个 ablation 比较便宜。
2. **β 敏感性扫描**：$\beta \in \{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$，找 transfer PSNR vs expressiveness 的 Pareto front。
3. **SVD base 可替换**：可以试 CogVideoX (Yang et al., 2025: <https://CogVideoX.github.io/>) 或 HunyuanVideo (Kong et al., 2024: <https://arxiv.org/abs/2412.03603>) 作为 base，看 condition injection 方式对效果影响。
4. **Condition injection ablation**：global concat vs cross-attention，这是 paper 没做的实验。
5. **Memory length 影响**：6 帧 → 12/24 帧，看 long-rollout 是否改善。但要注意 noise augmentation 范围。

---

## 12. 我对这篇 paper 的总体评价

**Strengths:**
1. Action-aware pretraining 的 paradigm 提得很 clear，把 latent action 从 "policy 中间产物" 重定位为 "world model control interface"，是有 conceptual 价值的。
2. Information bottleneck 双层设计（dim + KL）是 elegant 的，β trade-off 实验扎实。
3. Action transfer 零训练（Table 1 Human 70%）是真的 impressive。
4. w/o finetune 也 beat act-agnostic w/ finetune（Table 3），是 strong claim。
5. Generality 实验在 iVideoGPT 上验证，证明 paradigm 可移植。
6. Data diversity 实验显示 emergent cross-domain transfer，是 positive surprise。

**Weaknesses:**
1. Condition injection 细节含糊（SVD 改造那部分）。
2. 没有显式 disentanglement metric（只能看 UMAP）。
3. Long-rollout limitation 没 solve。
4. Real-time 没解决。
5. Latent action averaging 为什么 work 缺少理论解释。
6. Action vs distractor disentanglement 没显式 test。

**Bottom line:** 这是一篇 paradigm paper，不是 SOTA paper。它最大的贡献是 framing——把 action-aware pretraining 提成 world model 的第一性方法。后续工作可以在 condition 注入、disentanglement metric、long-horizon、real-time 上改进。我预期 2026 年会有几篇 follow-up 沿着这个路线 scale 上去。

---

## 相关链接汇总

| 类别 | 资源 |
|---|---|
| Paper 主页 | <https://adaptable-world-model.github.io/> |
| Genie | <https://sites.google.com/view/genie-2024> |
| LAPA | <https://latentactionpretraining.github.io/> |
| SVD | <https://stability.ai/research/stable-video-diffusion> |
| iVideoGPT | <https://thuml.github.io/iVideoGPT/> |
| GameNGen | <https://gamengen.github.io/> |
| DIAMOND | <https://diamond-wm.github.io/> |
| The Matrix | <https://thematrix-2.github.io/> |
| Cosmos | <https://research.nvidia.com/labs/toronto-ai/cosmos/> |
| OpenVLA | <https://openvla.github.io/> |
| 3D-VLA | <https://3d-vla.github.io/> |
| Vista | <https://vista-web.github.io/> |
| VPT | <https://openai.com/research/video-pre-training> |
| Diffusion Forcing | <https://boyuan.space/diffusion-forcing/> |
| DayDreamer | <https://daydreamer-tools.github.io/> |
| IRIS | <https://arxiv.org/abs/2210.15002> |
| Nikulin et al. (distractor) | <https://arxiv.org/abs/2502.00379> |
| PreLAR | <https://arxiv.org/abs/2410.13917> |
| Moto | <https://arxiv.org/abs/2412.04445> |
| IGOR | <https://arxiv.org/abs/2411.00785> |
| AgiBot World | <https://agibot-world.com/> |
| Open X-Embodiment | <https://robotics-transformer-x.github.io/> |
| RoboDreamer | <https://robovideo.github.io/> |
| Navigation World Models | <https://nav-world-models.github.io/> |

---

如果你想，我可以下一步：
- (a) 深入推一下 latent action 的 information theory 分析，看 32 维理论上限
- (b) 把 AdaWorld vs PreLAR vs LAPA 三家的差异列一个对照表
- (c) 谈谈 latent action 在机器人 manipulation 上的实际部署瓶颈
- (d) 推测一下 AdaWorld 路线 scale 到 10B+ 参数会出什么 emergent 行为

你想往哪个方向钻？
