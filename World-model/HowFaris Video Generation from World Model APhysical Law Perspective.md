---
source_pdf: HowFaris Video Generation from World Model APhysical Law Perspective.pdf
paper_sha256: 164afde7e4ba52d3e9243eb14076350bee3940618c25e2f3b00013def5744486
processed_at: '2026-08-05T07:31:35-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话总结

Sora team 说"video generation 是 world simulator", 这篇 paper 基本上就是用实验打脸说: **scaling 再大也没用, video model 根本没学到物理 law, 它在背 cases**。

## 为什么要写这篇 paper

2024 年 OpenAI 发 Sora tech report, 里面有一句话特别激动人心, 说 video generation models 是 "world simulators", 仿佛只要 scale 够大, model 就能从 video 里 discover $F=ma$、energy conservation 这些东西。整个 field 都很兴奋, robotics、autonomous driving 的人都觉得有救了 - 不用手工写 simulator, 直接 learn 一个就行。

但 Karpathy 你肯定有感觉, 这种 claim 很难 verify。你怎么知道 model 真的 "理解" 了惯性定律, 还是因为 training data 里见过类似的就 mimic 出来了？黑箱里面是啥, 你看不见。你只能看 generalization behavior - 给它没见过的场景, 看它 predict 得对不对。

所以 ByteDance 这帮人做了一件很 Karpathy-style 的事: **从 scratch 训练小模型, 完全控制 data, 用最简单的物理场景, 看 model 到底学到了什么**。这跟你教 micrograd、makemore 的哲学一样 - 想理解 model 行为, 就得能掌控全部输入。

## 他们具体怎么做的

### 先搞一个 super 简单的 testbed

他们用 Box2D (那个 2D 物理 engine) 搞了三个最经典的物理场景:

1. **Uniform motion**: 一个球在水平方向匀速运动。对应 Newton 第一定律 - 惯性。这个超级简单, 球就一路直走, velocity 不变。

2. **Elastic collision**: 两个球对撞。对应 energy 和 momentum conservation。这个就 nonlinear 了, 撞完之后两个球速度根据质量重新分配。

3. **Parabolic motion**: 一个球有水平初速度, 然后受重力下落。对应 $F=ma$。这个是抛物线。

每个场景就 2-4 个 degree of freedom, 比如球的 radius、initial velocity、mass (由 radius 算出来, 同密度)。这些 DoF 决定了整个 video 的 evolution, 因为物理是 deterministic 的。

视频格式: $128 \times 128$ resolution, 32 frames, 每帧间隔 0.1s, 总共 3.2s。

### 训练 setup - Sora 的 mini 版

Architecture 直接 follow Sora:
- **(2+1)D VAE**: 把 SD1.5-VAE 改造, 最后几层换 3D causal conv, 加 1D temporal layers。这个 VAE 预训练好就 freeze, 当 compressor 用。
- **DiT (Diffusion Transformer)**: 在 VAE latent space 上做 denoising。用 3D RoPE 做 position embedding。Conditioning 是前 $c$ 帧 (uniform/collision 用 3 帧, PHYRE 用 1 帧), zero-pad 到 full length, 再拼一个 binary mask。

Model size 从 22M 到 456M (DiT-S/B/L/XL), 数据从 30K 到 3M videos。

训练用 velocity prediction (Salimans & Ho 2022 那个):
$$\mathbf{y} = \sqrt{1 - \gamma_t}\,\epsilon - \sqrt{\gamma_t}\,V$$

这里 $\gamma_t$ 是 noise schedule, $t=1$ 时强制 $\gamma_t = 1$ 消除 train/inference gap。$\epsilon$ 是 Gaussian noise, $V$ 是 clean video。这个 target 其实是 noise-prediction 和 data-prediction 的 interpolation, 训练更稳定。

### 评估 - 这部分很 clever

他们不用 FVD 或者 human eval 之类模糊的指标, 而是**从 generated video 里把球的 position 解析出来, 算 velocity, 跟 ground truth 比**。

具体: 用 color-based heuristic 找球中心 $x_t^i$, 然后 $v_t^i = \frac{dx_t^i}{dt}$, 误差:
$$e = \frac{1}{N|T|} \sum_{i=1}^{N} \sum_{t \in T} |v_t^i - \hat{v}_t^i|$$

- $N$ 是球的数量
- $|T|$ 是 valid frame 数 (排除出界的 frame, collision 排除碰撞前的 frame)
- $v_t^i$ 是从 generated video parse 出来的速度
- $\hat{v}_t^i$ 是 simulator ground truth

这个指标直接量化 "model 学没学到物理 law", 因为你 prediction 速度对不对就是物理对不对。Baseline 是 GT error - 从 ground truth video parse 出来的速度误差, 这是 parsing noise 下限。

## 主要发现 - 一锤一锤打脸

### Finding 1: ID generalization 完美, OOD 灾难

In-distribution 测试 (训练范围内采样没见过的点):
- DiT-S @ 30K data: error 0.022
- DiT-L @ 3M data: error 0.012
- GT baseline: 0.010

Scaling 数据和模型 size, ID error 稳步下降, 接近 parsing 极限。完美。你看到这个会想 "啊, scaling work 了"。

OOD 测试 (radius 和 velocity 超出 training range):
- DiT-L uniform motion @ 3M: **OOD error 0.427, 比 ID 高 35 倍**
- scaling 几乎没用: DiT-B 在 30K/300K/3M 上的 OOD error 是 0.433/0.328/0.358, 纯随机波动
- 上 DiT-XL 也没改善

这就是 paper 第一个核心 finding: **naive scaling 不能让 video model 学到 physical law**。模型在 training range 内能 fit 得很好, 出去就崩。

### Finding 2: Combinatorial generalization 反而能 scale

这个 finding 很有意思, 他们用 PHYRE 这个 2D 物理模拟器, 设计了一个组合场景:

- 8 类 objects (gray ball, black bar, jar, standing stick 等等)
- 每次挑 4 个 + 1 个 red ball, $C_8^4 = 70$ 种 combinations
- 60 个 templates 训练, 10 个测试
- 训练数据分三档: 6 templates / 30 templates / 60 templates

结果:

| Templates | Out-of-template abnormal rate |
|-----------|-------------------------------|
| 6 | 67% |
| 30 | 18% |
| 60 | 10% |

**Combination diversity 从 6 涨到 60, abnormal rate 从 67% 降到 10%**。但同样 60 templates, 把 DiT-XL 换成 DiT-B, abnormal rate 涨到 24% - model capacity 也重要。

这个 finding 很 actionable: video generation scaling 的关键 axis 不是 data volume, 是 **combination diversity**。这跟你之前讲 data quality 比 data quantity 重要的直觉是一致的。

### Finding 3: 模型在 case-based 检索, 不是 rule-based 推理

这个 finding 最深刻, 也是 paper 的核心 insight。他们设计了几个 clever 的实验。

**实验 A: 验证 case-based behavior**

训练 uniform motion, $v \in [2.5, 4.0]$。两组:
- Set-1: 只有左→右运动
- Set-2: 加入水平翻转 (双向都有)

测试 $v \in [1.0, 2.5]$ (训练外低速度):
- Set-1 model: 只生成正速度, 偏向 high-speed
- Set-2 model: **偶尔生成负速度**, 即使 conditioning frame 显示球向右低速运动, 几帧后突然反向

这个反向行为物理上完全错 - 惯性定律说没有外力速度不变。但 model 不管, 它在 matching closest training case。Set-2 因为有翻转数据, 低速向右的球在它的 "记忆" 里 closest 的就是低速向左的 case, 所以它就生成了反向运动。

这跟 Hu et al. ICML 2024 在 LLM 加法上发现的现象一模一样 - Transformer 做算术也是 case-based, 不是 rule-based。Paper: https://arxiv.org/abs/2402.09996

**实验 B: 揭示 attribute 优先级 - 这部分超有意思**

他们用 pairwise comparison 测试模型在 case matching 时更看重哪个 attribute。设计: 两个 attribute 各有两个 value set, 训练用 4 种组合里的 2 种, 测试另外 2 种, 看 model preserve 哪个 attribute、改变哪个。

举例: 训练 data 是 red ball 和 blue square。测试 red square:
- 如果 model 学到的是 "保持 color", red square 应该变成 red ball
- 如果 model 学到的是 "保持 shape", red square 应该变成 blue square

实验结果: **red square 立刻变 red ball**, color 赢了。1400 个测试 case 无一例外。

跑了所有 pairwise 组合, 得到优先级:
> **color > size > velocity > shape**

具体证据:
- Color vs Shape: color 总是赢 (1400/1400)
- Color vs Size: color 赢
- Color vs Velocity: color 赢
- Size vs Velocity: size 大部分赢, 极端值时 velocity 略占
- Size vs Shape: size 赢
- Velocity vs Shape: velocity 赢

### Finding 4: 为什么是这个优先级? Pixel variation hypothesis

他们给了一个很优雅的解释。优先级反映了 **attribute 改变时 pixel variation 的大小**:

- **Color change**: 几乎每个 pixel 都变 (整个物体换色) → pixel variation 最大
- **Size change**: pixel 数量变, 但单个 pixel 值不变 → 中等
- **Velocity change**: 位置随时间 shift, 局部 pixel 变 → 中等
- **Shape change**: 只在 edges/corners 变 (圆变方只动四个角) → pixel variation 最小

Diffusion model 训练目标就是 minimize pixel/latent space 的 reconstruction error, 所以它在检索最近邻 training case 时, pixel variation 大的 attribute "anchor" 作用强 - 改这个 attribute 代价高, model 就倾向保持它。

**反例验证**: 当 shape 是 ring vs ball 时 (ring 变 ball 需要把中间空洞填上, pixel variation 大), color > shape 的规律就**不成立了** - ring 可以变 blue ring 也可以变 red ball。这进一步支持 pixel variation hypothesis。

这个 finding 直接解释了为什么 Sora-style model 难以保持 shape consistency - shape 在 model 的检索优先级最低, 一遇到冲突就牺牲 shape。

### Finding 5: 视觉表征本身不够

Section 5.5 给了一个很 punch 的例子: 球能不能穿过一个 gap, 当 size 差异在 pixel level 时, **视觉上根本看不出来**, 但物理结果完全不同。模型生成出来 "看起来对" 但物理错。

这暗示: **光看 pixels 是不够的**, 视频 alone 不够 build 完整 world model。需要额外 modalities 或者 explicit state representation。

## 更深的实验 - Interpolation vs Extrapolation

Section 5.1 的实验我觉得是 paper 最 deep 的部分。他们故意在 uniform motion 的 velocity training range 中间挖掉一段, 比如训练 $[1.0, 1.25] \cup [3.75, 4.0]$, 中间 $[1.25, 3.75]$ 不训练。

测试 $v = 2.5$ (中间 OOD):
- 模型生成的球 velocity 不保持在 2.5, 而是 "snap" 到 high 或 low, 模仿最近的 training case
- Gap 越小, 模型开始能 interpolate
- 把 missing range 的 subset 重新加回去 (不增加总数据量), interpolation 能力提升

**Collision 的 2D 实验**更直观: 在 $(v_1, v_2)$ 空间挖掉一些 square region。测试时:
- OOD 点在训练数据的 **convex hull 内** (被训练数据包围): generalization 还行
- OOD 点在 **convex hull 外**: 误差爆炸

这跟 Balestriero, LeCun 2021 的 paper "Learning in high dimension always amounts to extrapolation" 呼应 - 高维空间几乎所有 test point 都是 OOD, 所谓的 generalization 本质都是 interpolation。Paper: https://arxiv.org/abs/2110.09485

**video model 本质是个 interpolator**, 不是 rule learner。这跟 deep learning 的 fundamental limitation 有关。

## 跟你直觉的连接

Karpathy, 这篇 paper 的几个 finding 跟你之前讲的好多东西都 resonant:

### 1. "Software 2.0" 的边界

你 2017 年写 Software 2.0, 说 neural network 在 replacing explicit code。这篇 paper 暗示 Software 2.0 在 **physical world modeling** 这个 task 上有 fundamental limit - scaling 到 456M params + 3M data 都 discover 不了 $v = v_0$ 这种 trivial law 的 universal form。

可能需要的是 Software 2.0 + symbolic component 的 hybrid (像 AlphaGeometry 那种)。

### 2. Training from scratch 的方法论

你教 micrograd、makemore、nanoGPT 一直强调: 想理解 model, 就 from scratch 训练小模型, 完全控制 setup。这篇 paper 完全用这种 methodology, Section D.2 还专门论证了为什么不用 SVD 之类的 pretrained model - 因为 pretraining data 不可控, evaluation contamination 风险大。

他们 fine-tune SVD 的结果 (Table 4): OOD error 0.9081, 比 from-scratch DiT-B 的 0.3583 还差, 而且依然是 ID 的一个数量级。预训练不解决 OOD 问题。

### 3. "Grokking" 现象的缺席

Power et al. 2022 发现 grokking - 模型 overfit 之后突然 generalization 跳上来。这篇 paper 训练了 100K-1000K steps, training loss 早就 plateau (Figure 18), 但没看到 grokking 发生。说明这不是 optimization 时间问题, 是 model expressivity / inductive bias 的根本问题。

### 4. 跟 LeCun JEPA 的对比

LeCun 一直 diss diffusion / pixel-level prediction 这类生成模型, 主张 JEPA - 在 abstract latent space 做预测, 不做 pixel reconstruction。这篇 paper 的结论 (Section 5.5 视觉信息不足、整个 case-based retrieval 机制) 实际上支持了 LeCun 的论点:

- Pixel space 信息冗余度高, 模型容易走捷径做 case matching
- 在 abstract latent space (object-centric、physical state) 做 prediction 可能更 fundamental

LeCun JEPA paper: https://openreview.net/pdf?id=BZ5a1r-kVsf

### 5. 跟 LLM "幻觉" 的对比

你之前讲过 "hallucination 不是 bug, 是 feature" - 指 LLM 创造力的来源。但在 world model 场景下, 这种 case-based 的 "hallucination" 是危险的。物理世界不允许 hallucination - 你让 autonomous driving 的 world model hallucinate 一下试试?

paper 揭示 video model 的 hallucination 跟 LLM 同源 - 都是 case-based retrieval 而非 rule-based reasoning。要在物理 world 上 deploy, 必须解决这个问题。

### 6. Information Bottleneck 视角

Tishby 的 Information Bottleneck 理论说 deep network 训练分两阶段: 先 memorize (fit training data), 再 compress (forget noise, retain structure)。

这篇 paper 的 model 看起来停在 memorize 阶段没 compress 出 abstract rule。可能因为 visual data information 太冗余, compression 太难。对比 LLM 学算术 - 至少加法表是 discrete、low-dim, 还能 grokking 出 rule; video 的连续 high-dim state space 困难得多。

### 7. Compositional Generalization 和 LLM 的连接

Du & Kaelbling 2024 "A single model is not all you need" (https://arxiv.org/abs/2402.01103) 讲 compositional generative modeling - 你需要 multiple models 组合, 不是 single monolithic model。这篇 paper 的 combinatorial generalization 实验呼应: 6 templates 的 DiT-XL 不如 60 templates, 但 60 templates 还是有 10% abnormal。要真正做到 combinatorial generalization, 可能需要 explicit decomposition。

Riveland & Pouget 2024 Nature Neuroscience (https://www.nature.com/articles/s41593-024-01614-z) 显示 language instructions 能 induce compositional generalization in neural networks - 跟 paper 的 combinatorial generalization 主题 resonant。

## 一些可能被忽略的细节

### Multimodal conditioning 反而更差 (Section E.1)

他们试了在 collision 场景加 numeric state condition 和 text condition:
- ID: vision alone ≈ vision+numeric ≈ vision+text, 加额外 modality 没好处 (视觉信息已 sufficient)
- OOD: vision+numeric 略差, **vision+text 显著差**

解释: text embedding 是 discrete token, 变异性大, 容易让模型 overfit training pattern, 损害 OOD。这跟 Karpathy 你讲过 prompt engineering 时说的 "discrete space 难以 generalize" 一致。

这个 finding 对 VLA (Vision-Language-Action) model 是个警示 - 直接把 state 转成 text 塞进去可能损害 OOD generalization。

### 模型能做 spatial 和 temporal composition

Section 5.4 显示模型有一定 combinatorial 能力:
- **Spatial composition**: 训练有 "blue square moving + red ball static" 和 "red ball bouncing + blue square static", 测试两者同时运动能正确生成
- **Temporal composition**: 训练有 "two balls collide without bounce" 和 "red ball bounces off wall", 测试 "ball collides near wall" 能正确生成 collision + bounce

但 failure case (Figure 17): 当训练集没 red ball bounce 场景, collision 后 red ball 可能消失。模型 retrieve 了不含 red ball 的 collision case 来 stitch, 结果 red ball 凭空消失。这进一步证实 case-based retrieval 机制。

### CogVideo VAE 一致性

他们把 VAE 换成 CogVideo 的 VAE 重做 attribute priority 实验, 结论完全一致: color > size > velocity > shape。说明这个 priority 是 **diffusion model 本身的特性**, 跟具体 VAE 选择无关。

## 这篇 paper 的局限

诚实说几个局限:

1. **场景太简单**: Box2D 2D 场景, 没有真实 world 的 lighting、occlusion、non-rigid deformation。Sora 那种 hyper-realistic 的 video 上行为可能不同 (虽然我猜结论类似)。

2. **只测了 classical mechanics**: 没 quantum、fluid、EM 等其他 physics。

3. **VAE 固定**: 没探索 VAE 设计对 generalization 的影响。如果用 object-centric VAE (比如 Slot Attention 风格) 可能不同。

4. **只测 diffusion**: 没对比 autoregressive model (VideoPoet、MAGVIT 之类)。可能 AR model 的 inductive bias 不同。

5. **Combinatorial 部分用 human eval**: 10 个人标 abnormal rate, 主观性存在。但他们也用了 FVD/SSIM/PSNR/LPIPS, 这些 metric 跟 human eval 的 correlation 也只是 decent, 不是 perfect。

## 对未来的启示

如果让我从中提炼 actionable insight:

1. **数据策略**: 优先 combinatorial diversity, 别只堆 volume。Active learning 选 high-information combinations。

2. **架构方向**:
   - **Object-centric representations**: 显式建模 entities, 避免 pixel-level retrieval (Slot Attention, OC-VAE)
   - **Hierarchical dynamics**: 分离 per-object dynamics 和 interactions
   - **Causal inductive bias**: 引入 causal structure

3. **Training objective**: 仅 next-frame prediction 不足以学 rule, 可能需要 counterfactual training、contrastive learning on physical plausibility。

4. **Evaluation**: ID 评估完全不够, 必须有 OOD 和 combinatorial benchmark。Long-horizon rollout 评估也很重要 (paper 只测 32 frames)。

5. **Hybrid system**: 纯 neural 可能不行, 加 symbolic physics module (像 AlphaGeometry 那样) 可能是出路。

## 我自己的看法

这篇 paper 写得很扎实, 实验设计很 clever。但它最大的价值不是结论 (结论其实有点 obvious, 资深研究者早就怀疑 scaling 不能学 physics), 而是 **把怀疑变成了证据**。

之前大家口头说 "Sora 不真懂物理", 现在 paper 拿出 35 倍 OOD error、color>size>velocity>shape 优先级、case-based reverse direction 这些具体证据。这个从 "直觉" 到 "证据" 的过程是科学进步。

对 field 的影响: 接下来几年 world model 研究会往 object-centric、neuro-symbolic、hierarchical latent dynamics 这些方向走。Sora-style 的 raw pixel diffusion 路线在 physics modeling 上可能撞墙。

Karpathy, 你之前讲过 "supervised learning is all you need" 对 LLM 的 limitation, 这篇 paper 在 video generation 上 echo 了类似的论点 - supervised next-frame prediction 不够学 physics, 需要 different paradigm。

---

### 关键参考链接

- Paper 主页: https://phyworld.github.io
- arXiv: https://arxiv.org/abs/2503.01843
- Code: https://github.com/phyworld/phyworld
- Sora tech report: https://openai.com/research/video-generation-models-as-world-simulators
- PHYRE: https://phyre.ai
- Box2D: https://box2d.org
- DiT: https://arxiv.org/abs/2212.09748
- Hu et al. "Case-based or rule-based": https://arxiv.org/abs/2402.09996
- LeCun JEPA: https://openreview.net/pdf?id=BZ5a1r-kVsf
- Balestriero & LeCun extrapolation: https://arxiv.org/abs/2110.09485
- Du & Kaelbling compositional: https://arxiv.org/abs/2402.01103
- Riveland & Pouget: https://www.nature.com/articles/s41593-024-01614-z
- Karpathy Software 2.0: https://karpathy.medium.com/software-2-0-a6c52b88c4d8
- Slot Attention: https://arxiv.org/abs/2006.15055
- VideoPhysics benchmark: https://arxiv.org/abs/2406.03520
- Schott et al. OOD generalization: https://arxiv.org/abs/2107.08221
- Dreamer V3: https://arxiv.org/abs/2301.04104
- Ha & Schmidhuber World Models: https://worldmodels.github.io
- Genie (DeepMind): https://arxiv.org/abs/2402.19427
- Stable Video Diffusion: https://arxiv.org/abs/2311.15127
- MAGVIT: https://arxiv.org/abs/2212.05199
- VideoPoet: https://arxiv.org/abs/2312.14125
- CogVideo: https://arxiv.org/abs/2205.15868

---

# PhyWorld: Video Generation 离 World Model 还有多远? - 深度技术解析

## 1. Paper 核心问题与动机

这篇 paper 由 ByteDance Seed 团队(Bingyi Kang, Yang Yue 等)撰写, 直接质疑 Sora technical report 中的核心 assumption: video generation models 通过 scaling 是否能自动 discover physical laws 并成为 world simulator? OpenAI 在 Sora tech report 中声称 video generation 是 "world simulators" 的 promising path, 但该 paper 通过 controlled experiments 给出了 negative answer.

核心 motivation: 我们无法从 black-box model 内部判断它是否 "理解" 了 law, 只能通过 generalization behavior 来 infer. 因此 paper 提出三类 generalization 评估:
- **In-distribution (ID)**: 训练/测试数据 i.i.d.
- **Out-of-distribution (OOD)**: latent parameters 超出 training range
- **Combinatorial generalization**: 单个 concept 见过, 但组合没见过

paper 链接: https://phyworld.github.io
arXiv: https://arxiv.org/abs/2407.04241 (实际编号)

## 2. 问题形式化

### 2.1 Physical law discovery 的数学定义

考虑一个 physical process, latent variables $\mathbf{z} = (z_1, z_2, \ldots, z_k) \in \mathcal{Z} \subseteq \mathbb{R}^k$, 每个 $z_i$ 表示一个物理参数 (e.g., velocity, position, mass).

经典力学下 evolution 由 ODE 给出:
$$\dot{\mathbf{z}} = F(\mathbf{z})$$

离散化(frame 间隔 $\delta$):
$$\mathbf{z}_{t+1} \approx \mathbf{z}_t + \delta F(\mathbf{z}_t)$$

- $\mathbf{z}_t$: 时间 $t$ 的 latent state vector
- $\delta$: 相邻 frame 间的时间间隔 (paper 中为 0.1s)
- $F(\cdot)$: 动力学函数, 通常是 nonlinear

Rendering function $R: \mathcal{Z} \mapsto \mathbb{R}^{3 \times H \times W}$, 将 latent state render 成 $H \times W$ RGB image.

Video $V = \{I_1, I_2, \ldots, I_L\}$ 共 $L$ frames, 物理一致性要求:
1. $\mathbf{z}_{t+1} = \mathbf{z}_t + \delta F(\mathbf{z}_t)$, $t = 1, \ldots, L-1$
2. $I_t = R(\mathbf{z}_t)$, $t = 1, \ldots, L$

Video generation model $p_\theta$ 参数化为 $\theta$, 训练目标为 physical coherence loss:
$$\mathcal{L}_{\text{phys}} = -\log p_\theta(I_{c+1}, \ldots, I_L \mid I_1, \ldots, I_c)$$

- $c$: conditioning frames 数量 (uniform/collision 用 3, PHYRE 用 1)
- 这个 loss 形式上是 conditional likelihood, 模型必须理解 underlying dynamics 才能 minimize

### 2.2 Architecture: (2+1)D VAE + DiT

**(2+1)D VAE**: 基于 SD1.5-VAE 结构, 将最后几个 2D downsample block 替换为 3D causal conv block, 并增加 1D temporal layers. 这种设计保留 appearance modeling 能力的同时 enable motion modeling. Causal padding 保证只看 past frames, 避免未来信息泄漏.

**Diffusion Transformer (DiT)**: 跟随 Sora 架构, latent 表示 flatten 成 spatio-temporal patch sequence, self-attention 跨越 spatial+temporal 全部 tokens. Position embedding 用 3D variant of RoPE (Rotary Position Embedding, Su et al. 2024).

Conditioning 处理: 前 $c$ 帧 zero-pad 到 full length, 同时引入 binary mask $M \in \{0,1\}^{L \times H \times W}$, 前 $c$ 帧为 1. 最终 input 沿 channel 维拼接 noise + condition + mask.

## 3. Diffusion 训练细节

### 3.1 Forward process

DDPM formulation:
$$V_t = \alpha_t V + \beta_t \epsilon$$

- $V_t$: corrupted video at diffusion step $t$
- $V$: clean video
- $\epsilon \sim \mathcal{N}(0, I)$: Gaussian noise
- $\alpha_t = \sqrt{\gamma_t}$, $\beta_t = \sqrt{1 - \gamma_t}$
- $\gamma$: monotonically decreasing scheduler from 1 to 0
- $t \sim \mathcal{U}(0, 1)$: uniform sampling over diffusion timesteps

### 3.2 Velocity prediction target

跟随 Salimans & Ho 2022 (progressive distillation), 训练 target 为 velocity:
$$\mathbf{y} = \sqrt{1 - \gamma_t}\,\epsilon - \sqrt{\gamma_t}\,V$$

- 第一项: noise contribution scaled by $\sqrt{1-\gamma_t}$
- 第二项: data contribution scaled by $-\sqrt{\gamma_t}$
- 这相当于在 data-noise 之间定义 "velocity", 是 $\epsilon$-prediction 和 $V$-prediction 的 interpolation

Training loss:
$$\mathbb{E}_{V \sim p(x), t \sim \mathcal{U}(0,1), \epsilon \sim \mathcal{N}(0,I)} \left[ \| \mathbf{y} - p_\theta(V_t, \mathbf{c}, t) \|^2 \right]$$

- $\mathbf{c}$: conditioning signal (前 $c$ 帧 + mask)
- $p_\theta$: DiT network

特殊处理: $t = 1$ 时强制 $\gamma_t = 1$, 确保 final timestep 的 SNR = 0, 消除 train/inference gap (Lin et al. 2024).

## 4. 三个基本物理场景

### 4.1 Scenarios 设计

1. **Uniform Linear Motion**: 球水平匀速运动, 对应 **Law of Inertia** (Newton's first law). 2 DoF: radius, velocity.

2. **Perfectly Elastic Collision**: 两球对撞, 对应 **Conservation of Energy & Momentum**. 4 DoF: 两个球的质量 (由 radius 推断, 同密度) 和初速度.

3. **Parabolic Motion**: 球水平初速度 + 重力下落, 对应 **Newton's Second Law** $F = ma$. 2 DoF: radius, initial velocity.

### 4.2 数据范围

In-distribution:
- $r \in [0.7, 1.5]$
- $v \in [1, 4]$

Out-of-distribution:
- $r \in [0.3, 0.6] \cup [1.5, 2.0]$
- $v \in [0, 0.8] \cup [4.5, 6.0]$

Box2D simulator, $10 \times 10$ grid, $\delta = 0.1$s, 总时长 3.2s = 32 frames, resolution $128 \times 128$.

### 4.3 评估指标

从 generated video 中解析 ball 中心位置 $x_t^i$ (用 color-based heuristic, pixel mean). 计算速度 $v_t^i = \frac{dx_t^i}{dt}$. 误差:

$$e = \frac{1}{N|T|} \sum_{i=1}^{N} \sum_{t \in T} |v_t^i - \hat{v}_t^i|$$

- $N$: 球的数量 (uniform/parabola=1, collision=2)
- $|T|$: valid frames (排除球出界的 frame; collision 排除 collision 之前 frame)
- $v_t^i$: 解析出来的 ball $i$ 在 time $t$ 的速度
- $\hat{v}_t^i$: simulator ground truth 速度

Baseline: GT error - 即从 ground truth video 解析的速度误差, 代表系统下限 (parsing noise).

## 5. ID vs OOD Generalization 结果

### 5.1 Model scaling table

| Model | Layers | Hidden | Heads | #Param |
|-------|--------|--------|-------|--------|
| DiT-S | 12 | 384 | 6 | 22.5M |
| DiT-B | 12 | 768 | 12 | 89.5M |
| DiT-L | 24 | 1024 | 16 | 310M |
| DiT-XL | 28 | 1152 | 16 | 456M |

### 5.2 ID generalization - 完美

DiT-S @ 30K: 0.022 → DiT-L @ 3M: 0.012 → GT: 0.010 (uniform motion)

Scaling data + model 显著降低 ID error, 接近 GT parsing 极限.

### 5.3 OOD generalization - 失败

Uniform motion DiT-L @ 3M:
- ID error: 0.012
- OOD error: 0.427
- **OOD error 比 ID 高 ~35x**

更糟糕的是, scaling 数据和模型 size 对 OOD 几乎无改善:
- DiT-B uniform motion OOD: 30K→0.433, 300K→0.328, 3M→0.358 (随机波动)
- DiT-XL 在 3M 数据上 OOD 不优于 DiT-L

这直接说明: **naive scaling 不能让 video model discover physical laws**.

## 6. Combinatorial Generalization - 关键发现

### 6.1 PHYRE 设置

8 类 objects: 2 dynamic gray balls, fixed black balls, fixed black bar, dynamic bar, dynamic standing bars, dynamic jar, dynamic standing stick. 每次 4 个 + 1 个 red ball, $C_8^4 = 70$ templates.

Training data 三档: 6 templates (0.6M), 30 templates (3M), 60 templates (6M). 6 templates 包含所有 pairwise interactions.

### 6.2 结果 (Table 2)

| Model | #Templates | FVD (in/out) | SSIM | PSNR | LPIPS | Abnormal |
|-------|-----------|--------------|------|------|-------|----------|
| DiT-XL | 6 | 18.2/22.1 | 0.973/0.943 | 32.8/25.5 | 0.028/0.082 | 3%/67% |
| DiT-XL | 30 | 19.5/19.7 | 0.973/0.950 | 32.7/27.1 | 0.028/0.065 | 3%/18% |
| DiT-XL | 60 | 17.6/18.7 | 0.972/0.951 | 32.4/27.3 | 0.030/0.062 | 2%/10% |
| DiT-B | 60 | 18.4/21.4 | 0.967/0.949 | 30.9/27.0 | 0.035/0.066 | 3%/24% |

**关键洞察**: 
- Combinatorial 多样性 (6→60 templates) 将 abnormal rate 从 67% 降到 10%
- 同样 60 templates, DiT-B (24%) vs DiT-XL (10%) - model capacity 也重要
- 6 templates 在 in-template 最好 (SSIM/PSNR), 因为每个 example 训练频率高 10x

**Scaling law 的真正含义**: 视频生成 scaling 应关注 **combination diversity**, 单纯 volume 无效. 这呼应 Du & Kaelbling 2024 的 "A single model is not all you need".

## 7. 深入分析 - Generalization 机制

### 7.1 Interpolation vs Extrapolation (Section 5.1)

Uniform motion 实验中, 故意留出 velocity 中间区域 $[1.25, 3.75]$ 不训练, 仅训练 $[1.0, 1.25] \cup [3.75, 4.0]$. 

发现:
- 大 gap 时, 模型对中间 OOD 测试倾向于 "snap" 到 high 或 low velocity, 模仿 nearest training case
- Gap 缩小时, 模型开始正确 interpolate
- 重新加入 missing range 的 subset (总数据量不变), interpolation 能力提升

**Collision 实验的关键图 (Figure 6)**: 在 2D velocity 空间挖掉某些 square regions, 测试 OOD 时:
- OOD 点 **落在训练数据 convex hull 内** (内部 red squares): 泛化较好
- OOD 点 **落在 convex hull 外**: 误差极大

这是非常强的证据, 表明 video model 本质在做 **interpolation**, 而非 rule abstraction. 联想到 Balestriero, Pesenti, LeCun 2021 "Learning in high dimension always amounts to extrapolation" - 高维下 OOD 本质上都是 extrapolation, 这就是 deep learning 的 fundamental limitation.

### 7.2 Memorization vs Generalization (Section 5.2)

类似 Hu et al. ICML 2024 "Case-based or rule-based: How do transformers do the math?" 对 LLM 算术的研究.

实验设计:
- **Set-1**: 仅左→右运动训练, $v \in [2.5, 4.0]$
- **Set-2**: 加入水平翻转 (双向运动)
- 测试: $v \in [1.0, 2.5]$ (训练外)

结果:
- Set-1 model: 仅生成正速度 (向 high-speed range 偏)
- Set-2 model: 偶尔生成 **负速度**, 即使条件帧显示低速度球向右, 几帧后突然反向

**结论**: Model 没有抽象出 "uniform motion 保持速度方向" 的 rule, 它在 matching closest training case. 这就是 "case-based generalization".

### 7.3 Attribute Priority - 最重要的发现

Section 5.3 通过 pairwise 实验揭示 diffusion model 在 case matching 时的 attribute 优先级. 设计: 两个 attributes 各两个 disjoint value set, 训练用 4 种组合中的 2 种, 测试另外 2 种.

**结果 (Figure 8, 9, 14)**:

> **color > size > velocity > shape**

具体证据:
- **Color vs Shape**: 训练 red ball + blue square. 测试 red square → 立刻变 ball; blue ball → 变 square. 1400 测试无例外.
- **Size vs Shape**: shape 总是向 size 匹配的 case 靠拢.
- **Velocity vs Shape**: shape 总是让位于 velocity 匹配.
- **Color vs Size**: 训练 small red + large blue. 测试保持 color 不变, size 可以 shift.
- **Color vs Velocity**: color 不变, velocity 大幅 shift.
- **Size vs Velocity**: size 优先, 但极端值时 velocity 略占优.

**理论解释 (Section E.3)**: 优先级反映了 **pixel variation magnitude**:
- Color change: 几乎每个 pixel 都变 → 巨大 variation
- Size change: 改变 pixel 数量, 但每个 pixel 值不变 → 中等
- Velocity change: 位置随时间 shift, 局部 pixel 变化 → 中等
- Shape change: 局部 edges/corners 变化 → 最小

模型在 latent space 中 retrieve 最近邻 training case, 因此 pixel 变化大的 attribute "anchor" 更强. 

**反例验证 (Figure 16)**: 当 shape 是 ring vs ball 时 (transform 需要 fill in 中心, pixel variation 大), color > shape 不再成立 - 进一步支持 pixel variation hypothesis.

这也解释了为什么 Sora-style model 难以保持 shape consistency - shape 在 model 的检索优先级最低.

## 8. Combinatorial Generalization 的 Patterns (Section 5.4)

三种组合模式:

1. **Attribute composition**: velocity-size, color-size 等属性可组合 (Figure 14 1-2)
2. **Spatial composition**: 训练集中有 "blue square moving + red ball static" 和 "red ball bouncing + blue square static", 测试两者同时运动可生成 (Figure 11 left)
3. **Temporal composition**: 训练集中有 "two balls collide without bounce" 和 "red ball bounces off wall", 测试 "ball collides near wall" 能正确生成 collision + bounce (Figure 11 right)

**Failure case (Figure 17)**: 当 training set 没有 red ball bounce 场景, collision 后 red ball 可能消失 - 模型 retrieve 了不含 red ball 的 collision case.

这进一步证实 case-based retrieval 机制: 模型可以 stitch training segments, 但不理解 underlying entities/rules.

## 9. 视觉表征的 insufficiency (Section 5.5)

**视觉歧义问题**: 例如球能否穿过一个 gap, 像素级差异无法区分 (Figure 10). 视觉表示对 fine-grained physics 建模不足.

这暗示: **video 单独不足以构建完整 world model**, 需要额外 modalities 或 explicit state representation.

## 10. 与 LLM, SVD 等对比

### 10.1 Multimodal conditioning 实验 (Section E.1)

测试 vision+numeric vs vision+text 在 collision scenario 上的效果:
- ID: 三种 condition 表现相当 → 视觉信息已 sufficient
- OOD: vision+numeric 略差, vision+text 显著差

**解释**: numeric/text embedding 让模型 overfit 训练 pattern, 反而损害 OOD. Discrete token (text) 比 continuous numeric 更易 overfit.

这呼应 LLM 的 in-context learning 也存在类似 case-based 现象.

### 10.2 SVD fine-tune (Table 4)

| Model | ID Error | OOD Error |
|-------|----------|-----------|
| GT | 0.0099 | 0.0104 |
| DiT-B (ours, from scratch) | 0.0138 | 0.3583 |
| SVD-VAE-Recon | 0.0103 | 0.0107 |
| SVD-Finetune | 0.0505 | 0.9081 |

SVD (预训练 on internet videos) fine-tune 后 ID/OOD 表现都差于 from-scratch DiT-B - domain gap 大. OOD error 仍比 ID 高一个数量级, 与 from-scratch 趋势一致.

**结论**: 预训练不能解决 OOD 问题, 与 architecture 选择无关.

### 10.3 CogVideo VAE 实验 (Section D.1)

替换为 CogVideo VAE 重做 attribute priority 实验, 结论一致: **color > size > velocity > shape** 与 VAE 选择无关.

## 11. 与相关工作的连接

### 11.1 World Model 历史

- **Ha & Schmidhuber 2018**: "Recurrent world models facilitate policy evolution" - 经典 world model 工作, 在 abstract latent space 操作
- **Dreamer (Hafner et al. 2019, 2020, 2023)**: RSSM 架构, explicit latent dynamics model, 用于 RL planning
- **Genie (DeepMind, Bruce et al. 2024)**: unsupervised recovery of latent action from game videos
- **GAIA-1 (Hu et al. 2023)**: autonomous driving world model
- **1x World Model 2024**: robot control world model

这些传统 world model 工作在 **abstract latent space** 操作, 而 Sora-style 工作 **raw pixel/VAE space**. paper 暗示前者可能更 fundamental - 直接在 pixels 上建模难以 capture 真正 dynamics.

### 11.2 LLM 的 case-based reasoning

- **Hu et al. ICML 2024**: Transformer 做加法是 case-based memorization, 不是 rule learning
- **Riveland & Pouget 2024 Nature Neuroscience**: Language instructions induce compositional generalization - 与 paper 的 combinatorial generalization 呼应

### 11.3 Scaling laws

- **Kaplan et al. 2020**: Neural language model scaling laws - 经典 scaling law
- **Hoffmann et al. 2022 (Chinchilla)**: compute-optimal scaling

paper 揭示: video generation 的 scaling law 在 ID 上有效, 但在 OOD 上失效. Combinatorial scaling 是新的 scaling dimension.

### 11.4 OOD 文献

- **Schott et al. 2021**: "Visual representation learning does not generalize strongly within the same domain" - 类似 OOD 困境
- **Balestriero, Pesenti, LeCun 2021**: "Learning in high dimension always amounts to extrapolation" - 高维空间 OOD 的根本困难

### 11.5 Physical reasoning AI

- **PHYRE (Bakhtin et al. 2019)**: 2D 物理 reasoning benchmark - paper 直接用作 combinatorial testbed
- **CLEVRER (Yi et al. 2019)**: collision events video reasoning
- **CRAFT (Ates et al. 2020)**: causal reasoning about forces
- **Phy-Q (Xue et al. 2023)**: physical reasoning intelligence measure
- **VideoPhysics (Bansal et al. 2024)**: evaluating physical commonsense for video generation - 直接相关

## 12. 对 Karpathy 直觉的连接

### 12.1 与 micrograd/makemore/nanoGPT 教学哲学

Karpathy 一直强调 **从 scratch 训练小模型** 来理解 model behavior. 这篇 paper 完全采用这种 methodology - 训练 22M-456M DiT from scratch, 完全控制 training data. Section D.2 明确论证这种 methodology 的必要性: 避免 pretraining data contamination.

### 12.2 与 "Software 2.0", "Software 3.0" 思想

Karpathy 提出的 Software 2.0 (gradient-based) 和 Software 3.0 (prompt-based, LLM) 框架. 这篇 paper 揭示: 即使 scaling 到 billion params, video diffusion 仍是 "case-based" 检索系统, 没有真正 acquire "rules". 这对 world model 的 Software 2.0 路径提出质疑 - 可能需要 explicit symbolic/rule component (类似 Neuro-symbolic).

### 12.3 与 "grokking" 现象

Power et al. 2022 发现 grokking - 模型在 overfit 后突然 generalization. 这篇 paper 的训练 100K-1000K steps 是否足够长? 但 Figure 18 显示 training loss 已经 plateau, 300K vs 100K steps 表现一致 - 看起来不是 grokking 能解决的问题.

### 12.4 与 "信息瓶颈" 理论

Tishby 等人的 Information Bottleneck 理论: deep network 先 memorize 再 compress. 这篇 paper 显示 video model 停留在 memorization 阶段, 没有 compress 出抽象 rule. 可能因为 visual data 的 information redundancy 太高, compression 困难.

### 12.5 与 LeCun JEPA 的对比

LeCun 的 JEPA (Joint Embedding Predictive Architecture) 主张在 abstract latent space 做预测, 避免 pixel-level 重建. 这篇 paper 的结论 (Section 5.5 视觉信息不足) 倾向支持 JEPA 路线 - 在 latent space 中建模 dynamics 可能更适合 world model.

参考: LeCun 2022 "A Path Towards Autonomous Machine Intelligence" https://openreview.net/pdf?id=BZ5a1r-kVsf

## 13. 局限与开放问题

### 13.1 Paper 局限

1. **仅 2D 简单场景**: Box2D 模拟器, 缺乏 3D 真实世界的 complexity (lighting, occlusion, non-rigid)
2. **仅 classical mechanics**: 未测试 quantum, fluid, EM 等其他 physical laws
3. **VAE 已固定**: 未探索 VAE 设计对 generalization 的影响
4. **Diffusion only**: 未对比 autoregressive models (如 VideoPoet, MAGVIT)
5. **评估指标 reliance on human**: combinatorial 部分用人工评估, 主观性

### 13.2 开放问题

1. **如何 inject inductive bias** 让 model 学习 rule 而非 case matching?
   - Symbolic module? Physics-aware loss?
   - 类似 AlphaGeometry 的 neuro-symbolic 结合

2. **Cross-modal grounding**: video + state + action 是否足以学习 rule?
   - Section E.1 实验显示 numeric/text 反而损害 OOD, 可能需要更好的 integration 方式

3. **Object permanence / entity tracking**: case-based 检索的根本原因是缺乏 entity representation. 如何在 video model 中引入 object-centric representation?
   - Slot Attention (Locatello et al. 2020)
   - Object-centric world models (Večerík et al. 2023)

4. **Long-horizon extrapolation**: paper 测试 32 frames, 长视频下误差如何累积?

5. **Real-world transfer**: 从 2D synthetic 到 3D real video 的 gap 多大?

## 14. 对未来 video model 设计的启示

### 14.1 数据策略

- 优先 **combinatorial diversity** 而非单纯 volume
- Active learning 选择 high-information combinations
- Curriculum learning: 从简单组合到复杂组合

### 14.2 架构设计

- **Object-centric representations**: 显式建模 entities, 避免 pixel-level retrieval
- **Hierarchical dynamics**: 分离 per-object dynamics 与 interactions
- **Causal structure**: 引入 causal inductive bias

### 14.3 Training objective

- 仅 next-frame prediction 不足以学习 rule
- 可能需要 **counterfactual** training (what-if scenarios)
- Contrastive learning on physical plausibility

### 14.4 Evaluation

- 必须包含 OOD evaluation
- Combinatorial generalization benchmark 应成为 standard
- Long-horizon rollout 评估

## 15. 总结

这篇 paper 是对 Sora hype 的一剂清醒剂. 通过精细的 controlled experiments, 揭示:

1. **Video generation scaling ≠ world model**: naive scaling 在 OOD 失效
2. **Combinatorial diversity 是关键 scaling axis**: 60 templates 显著优于 6
3. **Case-based generalization 是 model 的真实行为**: 不是 rule abstraction
4. **Attribute priority: color > size > velocity > shape**: 反映 pixel variation magnitude
5. **Video 单独不足以构建完整 world model**: 视觉歧义问题

Karpathy 在多个场合提到 "幻觉不是 bug, 是 feature" (指 LLM creativity), 但在 world model 场景下, 这种 case-based 的 hallucination 是危险的 - 物理 world 不容 hallucination. 

paper 的核心 message: **要构建真正 world model, 我们需要的不仅是 bigger model 和 more data, 而是 fundamentally different 的归纳偏置和架构设计**. 这是一个 deep learning 时代的 fundamental challenge.

---

### 参考链接

- Paper project page: https://phyworld.github.io
- arXiv (paper 实际编号): https://arxiv.org/abs/2503.01843 (v2)
- Sora tech report: https://openai.com/research/video-generation-models-as-world-simulators
- PhyWorld GitHub (代码): https://github.com/phyworld/phyworld
- Box2D simulator: https://box2d.org
- PHYRE benchmark: https://phyre.ai
- DiT paper (Peebles & Xie 2023): https://arxiv.org/abs/2212.09748
- DDPM (Ho et al. 2020): https://arxiv.org/abs/2006.11239
- Progressive distillation (Salimans & Ho): https://arxiv.org/abs/2202.00512
- RoPE (Su et al.): https://arxiv.org/abs/2104.09864
- Ha & Schmidhuber World Models: https://worldmodels.github.io
- Dreamer V3 (Hafner et al.): https://arxiv.org/abs/2301.04104
- Genie (DeepMind): https://arxiv.org/abs/2402.19427
- Hu et al. "Case-based or rule-based" ICML 2024: https://arxiv.org/abs/2402.09996
- LeCun JEPA paper: https://openreview.net/pdf?id=BZ5a1r-kVsf
- Balestriero, LeCun "Learning in high dimension is extrapolation": https://arxiv.org/abs/2110.09485
- Schott et al. "Visual representation learning does not generalize": https://arxiv.org/abs/2107.08221
- VideoPhysics benchmark: https://arxiv.org/abs/2406.03520
- Riveland & Pouget Nature Neuro: https://www.nature.com/articles/s41593-024-01614-z
- Du & Kaelbling compositional gen: https://arxiv.org/abs/2402.01103
- Karpathy Software 2.0: https://karpathy.medium.com/software-2-0-a6c52b88c4d8
- Slot Attention: https://arxiv.org/abs/2006.15055
