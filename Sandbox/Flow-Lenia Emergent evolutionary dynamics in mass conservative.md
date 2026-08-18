---
source_pdf: Flow-Lenia Emergent evolutionary dynamics in mass conservative.pdf
paper_sha256: 1d94c1f34cfcb9294b9d83846b251090291ef3a20e5e1f70f3cd7b447929de54
processed_at: '2026-08-18T13:29:22-07:00'
target_folder: Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 Flow-Lenia

## 一句话 version

Lenia 是个出 life-like 动画的连续 CA，但找好看的 pattern 特难，大部分参数要么让世界炸掉、要么让世界空掉。这篇 paper 给 Lenia 加了**质量守恒**这个物理约束 —— matter 不能凭空产生也不能凭空消失，只能流动 —— 于是 random 参数就能稳定跑出"小虫子"了，还能让多个 species 在同一个 world 里共存 + 竞争 + 演化。

## 为什么原始 Lenia 难搞

原始 Lenia 的 update rule 是：

```
A_next = clip(A + dt * U, 0, 1)
```

这个 `clip(·, 0, 1)` 是罪魁祸首。它做两件事：
1. **创造 matter**：当 A + dt*U > 1 时，多余的部分被 clip 掉，相当于凭空消失了
2. **湮灭 matter**：当 A + dt*U < 0 时，负值被 clip 成 0，相当于凭空消失了

所以 matter 不守恒。参数稍微偏一点，要么全 grid 爆炸成 Turing pattern，要么全死了。所以绝大部分参数空间是"无用的"，你必须用 IMGEP + curriculum learning + gradient descent（Hamon et al. 2024, https://arxiv.org/abs/2402.10236）才能找到稳定的小生物。

类比：就像一个 RL environment，reward landscape 全是 cliff —— 一脚踩错就 game over。

## Flow-Lenia 怎么 fix 的

核心 idea 很简单：**让 matter 沿着 affinity gradient 流动，别 clip 了**。

```
1. 算 affinity map U (跟 Lenia 一样)
2. 算 flow F = 沿 ∇U 流 - 防爆聚的 diffusion term
3. 用 reintegration tracking 把 matter 按流移动，积分保证守恒
```

**关键技巧是 reintegration tracking**（Moroz 2020, https://observablehq.com/@moroz/reintegration-tracking）。

直觉：假设每个 cell 装着无限多个 infinitesimal 粒子，按 flow field F 流动后，你要把这些粒子重新 bucket 到 grid cells 里。具体做法是把每个 source cell 的 mass 当作一个 uniform square distribution，center 在 `x'' = x' + dt * F(x')`，然后算这个 square 跟 target cell 的 1×1 square 的 intersection area，按比例分配 mass。

```
mass_to_target(x) = sum over source cells x' of:
    mass(x') * area(D(x'', s) ∩ Ω(x)) / (2s)^2
```

因为每个 source cell 的 distribution 积分为 1，所以总和守恒。这就是一个 grid-based particle system 的近似。

## 为什么守恒这件事这么 magic

paper 给了三个层面的 benefit：

### 1. Random search 就能出好 pattern

105 个 random 参数跑下来：
- Lenia: 大部分要么空、要么炸
- Flow-Lenia: 大部分是 spatially localized pattern（SLP）

Mass conservation 是一个**物理 inductive bias**，它把 parameter space 的 useless region 大幅压缩了。你不用再做"它会不会爆"这件事的 search 了。

直觉：就像给 GAN 加 spectral normalization 一下稳定多了 —— 一个简单的物理约束就能 reshape 掉整个 loss landscape。

### 2. Vanilla ES 就能 optimize 行为

四个 task：直线运动、转向、穿越障碍、chemotaxis。用 OpenES（Salimans 2017, https://arxiv.org/abs/1703.03864），population 16，Adam lr 0.01，就 work 了。

对比：原始 Lenia 上跑一样的 ES，curve 抖得要死，而且发现的 pattern 全是 exploding 的。

为什么 stable？因为 mass conservation 给了一个 intrinsic robustness —— creature 不会因为 perturbation 消失，因为 mass 不会蒸发。ES 的 noisy perturbation 不会 kill 掉 creature，creature 只是 slightly 改变形状，mass 还在，可以 self-repair。

这跟 RL 里 reward shaping 很像 —— 一个物理 prior 把 reward landscape 磨平了。

### 3. Parameter embedding → multi-species 共存

这是最 cool 的部分。把 update rule 的参数（kernel weight vector $h$）当作 genome，attach 到每个 cell 的 matter 上，随 matter 流动。

直觉：每个 cell 不光有 matter concentration，还有自己的 local DNA。Update rule 用 local DNA 计算 affinity。

当一个 cell 从多个 source cells 收到 matter 时，按 incoming mass 做 softmax sampling 选一个 source 的 DNA 继承下来（不是 average —— average 会让每次碰撞都产生 hybrid，破坏 species identity）。

这个 stochastic sampling rule 创造了**竞争 dynamics**：species A 可以把 species B 的 matter "convert" 过来变成 A 自己的 matter。这是 intrinsic evolution 的 driver。

## 实际跑出来什么

500k steps 的 simulation，初始化 64 个 creatures，加 mutation beams（10×10 patch 给 parameter map 加 N(0,1) perturbation）。

跑完后用 PCA 把高维 parameter space 投到 2D，按时间染色 plot trajectory —— 你会看到一棵**演化树**！有 branching、有 extinction、有 speciation。这说明 parameter space 里存在一个 **intrinsic fitness landscape** —— 不是 random walk，是有方向性的演化。

这跟 biology 直觉相反的发现：dissipative model（不断注入 new creatures）和 food model（creature 必须进食才能活）的 normalized evolutionary activity 反而**低于** vanilla model。

为什么？dissipative model 注入的新 parameters 是从 N(0,1) 采样的，但经过演化后适应了环境的 parameters 已经 drift 出这个分布了，所以新来的"移民"打不过"本地物种"，diversity 反而最低。

这正好体现了 quantitative model 的价值 —— 你以为 dissipation 会驱动 OEE，但 formalize 跑一下，发现取决于你怎么 instantiate。直觉需要 measurement 来校准。

## 几个让你（Karpathy）会眼前一亮的点

### Reintegration tracking 可微吗

是的，可以。$I(x', x)$ 是两个 square 的 intersection area / (2s)^2，intersection 是可微的（或者可以用 soft approximation）。所以整个 Flow-Lenia 可以 backprop。Paper 用 ES 是图省事，但其实你可以直接 gradient descent through reintegration tracking —— 这就是一个 differentiable particle system。

跟 Neural CA（Mordvintsev 2020, https://distill.pub/2020/growing-ca/）对比：Neural CA 用 small ConvNet 当 update rule，参数是 global 的、learned 的、fixed 的。Flow-Lenia 用 Gaussian kernel family 当 update rule，参数是 local 的、embedded 的、evolving 的。

两者结合的 obvious next step：用 small NN 当 local growth function $G_i$，把 NN weights 当 genome embed 进去。你就得到了一个 mass-conservative multi-species Neural CA。这个 hybrid 我没看到有人做过。

### Critical regime

Fig 4j 超有意思：linearly 增加 temperature $s$，你会发现 Turing-like phase 和 equilibrium phase 之间的 boundary 最 dynamic、最 life-like。这跟 Langton 1990 的 "life at the edge of chaos"（https://doi.org/10.1016/0167-2789(90)90057-V）和 Bak 的 self-organized criticality 完全呼应。

Flow-Lenia 给了一个**干净的 criticality tuning knob**（temperature $s$），可以系统地研究 SOC 与 life 的关系。

### Parameter embedding 就是 weight-agnostic brains

把 $h$ 当 genome attach 到 matter 上，这非常像 Stanley 的 Weight Agentic Neural Networks（https://weightagnostic.github.io/）。区别是 WANN 的 genome 编码一个 network 的 weight，Flow-Lenia 的 genome 编码一个**局部物理定律**（update rule）。这是 physics-as-computation 的视角 —— genome 不编码 brain，genome 编码"局部物质如何流动"。

### Mass conservation 作为 intrinsic regularizer

ES 在 Lenia 上 unstable、在 Flow-Lenia 上 stable。这跟 RL 里 reward shaping、constrained optimization 的 intuition 一致 —— 一个物理 prior（守恒律）把 hypothesis space 压缩到了"合理"的 sub-space。

类比：给 language model 加 next-token prediction 这个 simple prior，整个 representation 就自发涌现了。物理定律作为 prior 在 ALife 里扮演类似角色。

## 一张图总结整个 paper

```
原始 Lenia: A_next = clip(A + dt*U, 0, 1)
            ↑ 不守恒，参数空间难搜，单 species

Flow-Lenia: 1. U = affinity map (跟 Lenia 一样)
            2. F = (1-α)∇U - α∇A_Σ  (flow field)
            3. A_next = reintegration_tracking(A, F)  (守恒!)
            
            ↓ 加上 parameter embedding
            
            P: L → Θ (genome map)
            U(x) uses local P(x)
            reintegration moves P with matter
            softmax sampling on incoming mass
            → multi-species competition
            → intrinsic fitness landscape
            → evolutionary tree emerges
```

## 给你的 actionable suggestions

1. **直接玩**：companion website https://sites.google.com/view/flow-lenia 有 videos，code 是 JAX 跑在 Colab 上
2. **复现 cost 极低**：128×128 grid + 10 kernels，Tesla T4 单步 255μs，一个 500k steps 的演化 sim 几分钟跑完
3. **下一步 obvious idea**：用 small MLP 替换 Gaussian growth function $G_i$，embed MLP weights 作为 genome，跟 Neural CA 合流。这是一个 potential "Neural Flow-Lenia" 的低 hanging fruit
4. **加 differentiable reintegration tracking**，直接 backprop through 演化 —— 让 creature 通过 gradient 学到更好的"genome"，跟 ES 对比
5. **研究 critical regime**：系统地 sweep temperature $s$，看 evolutionary activity 是不是在 phase boundary 处 peak —— 如果是，这就是 ALife 版的 "scaling law at criticality"

## 参考链接汇总

- Paper: https://doi.org/10.1162/artl_a_00471
- Website: https://sites.google.com/view/flow-lenia
- Reintegration tracking: https://observablehq.com/@moroz/reintegration-tracking
- Lenia: https://doi.org/10.25088/ComplexSystems.28.3.251
- Particle Lenia: https://google-research.github.io/self-organising-systems/particle-lenia.html
- Neural CA: https://distill.pub/2020/growing-ca/
- OpenES: https://arxiv.org/abs/1703.03864
- WANN: https://weightagnostic.github.io/
- Hamon et al. sensorimotor: https://arxiv.org/abs/2402.10236
- Langton edge of chaos: https://doi.org/10.1016/0167-2789(90)90057-V

---

# Flow-Lenia: Mass Conservative Continuous Cellular Automata 中的 Emergent Evolutionary Dynamics

## 1. 大局观:这篇 paper 在 ALife 谱系中的位置

这篇 paper 由 IT University of Copenhagen 的 REAL lab 与 Inria 的 Flowers AI & CogSci Lab 联合 Google DeepMind 的 Bert Wang-Chak Chan 共同完成,是 Lenia 系列工作的延续。核心贡献在于把 Lenia (Chan, 2019/2020) 从一个 **non-conservative continuous CA** 改造成一个 **mass-conservative** 的 system,使得三件之前在 CA 中很难的事情变得容易:

1. **Spatially-localized patterns (SLPs) 的自发涌现**: 在原始 Lenia 中,需要复杂的 IMGEP + gradient descent (Hamon et al., 2024, https://arxiv.org/abs/2402.10236) 才能稳定找到 SLPs;Flow-Lenia 中,random search 就能大量产生。
2. **Directed optimization 的稳定性**: vanilla ES (Salimans et al., 2017, https://arxiv.org/abs/1703.03864) 直接能优化出 directed motion、angular motion、chemotaxis 等行为。
3. **Multi-species 共存 + intrinsic evolution**: 通过 parameter embedding,把 update rule 的 parameters 当作 genome 搭载在 matter 上,使不同 species 在同一 simulation 中竞争/共生。

这个工作可以看作朝着 **emergent microcosms** (Arbesman, 2022) 与 **open-ended evolution** (Stanley, 2019, https://direct.mit.edu/artl/article/25/3/232/99_Why-Open-Endedness-Matters) 迈进的实验性一步。参考 companion website: https://sites.google.com/view/flow-lenia 。

---

## 2. 背景:Lenia 的更新规则回顾

Lenia 把 Conway's Game of Life (Adamatzky, 2010) 推广到 continuous space / continuous time / continuous state。State 是一个 map $A^t: \mathcal{L} \to [0,1]^C$,其中 $\mathcal{L}$ 是 2D grid,$C$ 是 channel 数。

Update rule 由 tuple $\langle K, G, c_1, c_0, A^0 \rangle$ 定义:

- $K = \{K_i\}$: 一组 convolution kernel,每个 $K_i: \mathcal{L} \to [0,1]$,且 $\int_{\mathcal{L}} K_i = 1$(normalized)
- $G = \{G_i\}$: 一组 growth function $G_i: [0,1] \to [-1,1]$
- $(c_0^i, c_1^i)$: 第 $i$ 个 kernel-growth pair 感应 source channel $c_0^i$、更新 target channel $c_1^i$

Kernel 用 radial symmetrical 形式 (sum of concentric Gaussian bumps):

$$
K_i(x) = \sum_{j=1}^{k} b_{i,j} \exp\Big( -\frac{(\frac{x}{r_i R} - a_{i,j})^2}{2 w_{i,j}^2} \Big) \tag{1}
$$

变量含义:
- $a_{i,j} \in [0,1]$: 第 $i$ 个 kernel 第 $j$ 个 ring 的 normalized center (相对 $r_i R$)
- $b_{i,j}$: 该 ring 的 amplitude
- $w_{i,j}$: 该 ring 的 width
- $r_i \in [0.2, 1]$: kernel 半径的 scale factor
- $R \in [2, 25]$: global neighborhood radius
- $k=3$: 每个 kernel 的 ring 数量

每个 kernel 共 $3k+1 = 10$ 个参数。

Growth function:
$$
G_i(x) = 2\exp\Big(-\frac{(\mu_i - x)^2}{2\sigma_i^2}\Big) - 1 \tag{2}
$$
- $\mu_i \in [0.05, 0.5]$: growth peak 的位置
- $\sigma_i \in [0.001, 0.2]$: growth 的 sharpness
- 输出范围 $[-1, 1]$, $\mu_i$ 处取最大值 $+1$

Lenia 的一步更新:

$$
U_j^t = \sum_{i=1}^{|K|} h_i \cdot G_i\big(K_i * A_{c_0^i}^t\big) \cdot [c_1^i = j] \tag{3}
$$

$$
A_i^{t+dt} = \big[ A_i^t + dt\, U_i^t \big]_0^1 \tag{4}
$$

其中:
- $h \in \mathbb{R}^{|K|}$: kernel-growth pair 的 weighting vector
- $[\cdot]$ 是 Iverson bracket,条件满足取 1 否则 0
- $[\cdot]_0^1$ 是 clip 到 $[0,1]$
- $dt$ 是 time step

**关键问题**:公式 (4) 的 clip 是一个 hard nonlinearity,它破坏了 mass conservation —— mass 可以被凭空创造或湮灭。这导致 Lenia 中要么 patterns 消失,要么 patterns 爆炸性增长占据整个 grid (Turing-like patterns)。所以大部分 random parameter 都不可用,必须用复杂 search algorithm 才能找到 SLPs。

---

## 3. Flow-Lenia 的核心创新:Mass Conservation via Reintegration Tracking

### 3.1 把 growth 重新解释为 affinity map

Flow-Lenia 复用 Lenia 的所有 components (kernel、growth function),但把 $U^t$ 从 "growth" 重新解释为 **affinity map** —— 表示 matter 应该向哪里聚集。Activations $A$ 重新解释为 matter concentration,不再 clip 到 $[0,1]$,而是取 $\mathbb{R}_{>0}$ (任意正实数)。

### 3.2 Flow 的定义

matter 沿着 affinity map 的 gradient $\nabla U$ 流动,但需要一个 diffusion term 来防止所有 matter 聚集到无穷小的点上 (类似于 Lenia 中 clip 的作用)。Flow 定义为:

$$
\boxed{
\begin{aligned}
F_i^t &= (1 - \alpha^t)\, \nabla U_i^t \;-\; \alpha^t\, \nabla A_\Sigma^t \\
\alpha^t(x) &= \Big[\big(A_\Sigma^t(x) / \beta_A\big)^n\Big]_0^1
\end{aligned}
} \tag{5}
$$

变量和 intuition:
- $F_i^t: \mathcal{L} \to \mathbb{R}^2$: channel $i$ 在每个 cell 上的 instantaneous flow velocity (vector field)
- $A_\Sigma^t(x) = \sum_{i=1}^C A_i^t(x)$: 该 cell 的 total mass (跨所有 channel 求和)
- $\nabla U_i^t$: affinity gradient,推动 matter向 affinity 高的地方聚集
- $-\nabla A_\Sigma^t$: 负的 concentration gradient,纯 diffusion term,推动 matter 从高浓度向低浓度扩散
- $\alpha^t(x) \in [0,1]$: 一个 spatial weighting map
- $\beta_A$: critical mass threshold
- $n > 1$ (通常取 2): 控制 switching 的 sharpness

**直觉**: 当某处 total mass 远低于 $\beta_A$ 时,$\alpha \to 0$,flow 由 affinity gradient 主导 —— matter 自由地朝 affinity 高的地方聚集;当 mass 接近 $\beta_A$ 时,$\alpha \to 1$,flow 由 diffusion 主导 —— matter 被强制从密集区扩散出来。这就形成一个**自我调节的 density cap**,无需硬性 clip。

实际实现中,$\nabla U_i^t$ 和 $\nabla A_\Sigma^t$ 通过 Sobel filter 估计。

### 3.3 Reintegration Tracking

下一步是按 flow $F^t$ 移动 matter,要求 **mass conservation**。Paper 采用 Moroz (2020) 提出的 reintegration tracking (https://observablehq.com/@moroz/reintegration-tracking)。其核心 idea:

把每个 cell 中的 mass distribution 看作"无限多个粒子",沿 flow 流动后,落到目标位置的 distribution 上,然后积分到目标 cell。

$$
\boxed{
\begin{aligned}
A_i^{t+dt}(x) &= \sum_{x' \in \mathcal{L}} A_i^t(x')\, I_i(x', x) \\
I_i(x', x) &= \int_{\Omega(x)} \mathcal{D}(x_i'', s) \\
x_i'' &= x' + dt \cdot F_i^t(x')
\end{aligned}
} \tag{6}
$$

变量含义:
- $x_i''$: 从 source cell $x'$ 在 channel $i$ 上沿 flow $F_i^t$ 移动 $dt$ 时间后的目标位置
- $\Omega(x)$: 目标 cell $x$ 的 spatial domain (1×1 square)
- $\mathcal{D}(m, s)$: 以 $m$ 为 mean、$s$ 为 variance 的 distribution (实践中是 side length $2s$ 的 uniform square distribution,centered at $m$),且 $\int_{\mathcal{L}} \mathcal{D} = 1$
- $s$: 一个 hyperparameter,扮演 "temperature" 角色 —— 模拟 Brownian motion
- $I_i(x', x)$: 从 $x'$ 流出的 mass 中落到 $x$ 的比例

**为何 conserves mass**: 因为 $\mathcal{D}$ 积分为 1,从任一 source cell $x'$ 流出的总 mass 比例是 1(对 $x$ 求和为 1),所以 cell 既不会凭空创造 mass,也不会丢失 mass。

**计算 trick**: 不遍历所有 cell,只看 Chebyshev distance ≤ 5 的 extended Moore neighborhood,大幅加速。

实现用 JAX (https://github.com/google/jax),在 Tesla T4 GPU 上 1 channel / 10 kernels / 128×128 world 单步 $255 \mu s \pm 3.11 \mu s$。

### 3.4 与 Lenia / Particle Lenia 的关系

Paper 明确说 Flow-Lenia 是 "frontier between continuous CA and particle systems"。Mordvintsev 等人后来直接提出了一个 particle-based model "Particle Lenia" (https://google-research.github.io/self-organising-systems/particle-lenia.html),受 Flow-Lenia formulation 启发。可以理解成 Flow-Lenia 是 reintegration tracking 给出的 particle system 的连续极限 / grid approximation。

---

## 4. Parameter Embedding: 让 update rule 随 matter 流动

### 4.1 动机

原始 Lenia 中,update rule parameters (kernel weights $h$, kernel shape $a/b/w/r$, growth $\mu/\sigma$) 是 global 的。如果想做多 species simulation,必须让 parameters 变成 local 的 —— 像 genome 一样附着在 matter 上,随 matter 流动、随 matter 复制。

### 4.2 形式化

定义 parameter map $P: \mathcal{L} \to \Theta$,此处 $\Theta \equiv \mathbb{R}^{|K|}$ (只 embed kernel weighting vector $h$,因为 dynamically 改 kernel shape 会让 fast-Fourier convolution 失效)。

修改 affinity 计算 (公式 3):
$$
\boxed{
U_j^t(x) = \sum_{i=1}^{|K|} P_i^t(x) \cdot G_i\big(K_i * A_{c_0^i}^t\big)(x) \cdot [c_1^i = j]
} \tag{7}
$$

即每个 cell 用自己 local 的 $P(x)$ 加权 kernel-growth pairs。

### 4.3 Mixing rule

当 reintegration tracking 把多个 source cells 的 matter 送入同一个 target cell 时,要决定 target cell 的 parameter 取什么。Paper 比较了两种:

1. **Average rule** (Plantec et al., 2023): 取加权平均 —— 但这会让每次 interaction 都产生 hybrid parameters,破坏 species identity。
2. **Stochastic sampling (softmax)**: 论文采用。
   
$$
\boxed{
\mathbb{P}\big[P^{t+dt}(x) = P^t(x')\big] = \frac{e^{A^t(x')\, I(x', x)}}{\sum_{x'' \in \mathcal{L}} e^{A^t(x'')\, I(x'', x)}}
} \tag{8}
$$

直觉: 落入 $x$ 的 mass 越多的 source cell,其 parameter 被选中的概率越高 (softmax over incoming mass)。这创造**竞争 dynamics**:一种 species 可以 "convert" 另一种 species 的 mass,把它的 mass 拉过来变成自己 parameter 下的 matter。这是 intrinsic evolution 的关键 driver。

也提供 deterministic 版本 (用 argmax 代替 softmax sampling),会让 evolutionary tree 的 branches 更细、trajectory 更清晰。

---

## 5. 实验设计

### 5.1 Random search

参数空间 (Table 1):
- $R \in [2, 25]$, $r \in [0.2, 1]$
- 每个 kernel: $h \in [0,1]$, $a, b, w \in \mathbb{R}^3$ (3 个 ring)
- 每个 growth function: $\mu \in [0.05, 0.5]$, $\sigma \in [0.001, 0.2]$
- Flow hyperparameters: $S = 0.65$, $n = 2$, $dt = 0.2$

用 105 个 random parameter sets,在 Lenia 和 Flow-Lenia 上跑同样的 150 steps。

### 5.2 Directed search (4 个 task)

用 OpenES (Salimans et al., 2017, https://arxiv.org/abs/1703.03864) via EvoSax (https://arxiv.org/abs/2212.04180),population size 16,Adam (https://arxiv.org/abs/1412.6980) lr 0.01。

Tasks:
- **Directed motion**: 单方向最大位移
- **Angular motion**: 直线 + 转向
- **Navigation through obstacles**: 在 "forest" 中保持 integrity
- **Chemotaxis**: 沿 concentration gradient 爬升

对 Lenia 做同样 directed motion 优化作为 baseline。

### 5.3 Intrinsic evolution experiments

500k steps,3 channels + 5 kernels per channel pair = 45 kernels total。初始 64 个 creatures (20×20 patch)。

**Mutation beams**: 随机 10×10 patch,perturbation $\sim \mathcal{N}(0,1)$ 加到 $P$ 上。Mutation rate $p_{mut}$。用 beams 而非 single-cell mutation 是因为 single-cell mutation 会被邻居迅速 overwrite。

**三种 model 变体**:

1. **Vanilla**: 仅初始 + mutation beams
2. **Dissipative**: 加 dissipative beams —— 一类移除 matter+parameters,另一类在 corner 区域 (100×100 input zone) 注入新 creatures (random parameters)。Rate $p_{diss}$
3. **Food**: 引入 food map $\Psi: \mathcal{L} \to [0, \infty)$; creatures 的 mass 以 $r_{decay}$ 衰减;matter 与 food 同 cell 时以 $r_{digest}$ 转化;初始 32 个 5×5 food patches;每步以 $p_{food}$ 概率加新 patch;creatures 通过 extra kernels 感知 food

Food model 引入了 **minimal criterion** (Soros & Stanley, 2014, https://doi.org/10.1162/978-0-262-32621-6-ch128; Taylor, 2015, https://arxiv.org/abs/1507.07403) —— creatures 必须不断进食才能维持 mass,这是 OEE 的 candidate necessary condition。

### 5.4 Diversity 与 Evolutionary Activity Metrics

**Diversity** $D(t)$: parameter space 中所有 present parameters 之间的平均 Euclidean 距离。

$$
D(t) = \frac{1}{|\mathcal{P}^t|} \sum_{p \in \mathcal{P}^t} \sum_{p' \in \mathcal{P}^t} \|p - p'\|_2 \tag{9}
$$

**Count-based evolutionary activity** ($EA^C$):
$$
a_p^C(t) = \big(a_p^C(t-1) + M(p,t)\big) \cdot [p \in \mathcal{P}^t] \tag{10}
$$
即每步累加 species $p$ 的 total mass,只要它还活着。Intuition: mass 大 + 存活长 = activity 高。

**Non-neutral evolutionary activity** ($EA^N$): 惩罚 stasis。
$$
\boxed{
\begin{aligned}
a_p^N(t) &= \big(a_p^N(t-1) + \Delta_p^N(p,t)\big) \cdot [p \in \mathcal{P}^t] \\
\Delta_p^N(t) &= \Big(\sum_{p'} M(p', t)\Big) \cdot \big(\rho(p,t) - \rho(p,t-1)\big)^2 \cdot [\rho(p,t) > \rho(p,t-1)] \\
\rho(p,t) &= \frac{M(p,t)}{\sum_{p'} M(p', t)}
\end{aligned}
} \tag{11}
$$

变量:
- $M(p,t)$: species $p$ 在时刻 $t$ 的 total mass
- $\rho(p,t)$: $p$ 的 mass proportion in total population
- $\Delta_p^N$: 只在 proportion **上升** 时累加,且累加量为 squared proportion change × total mass

Intuition: 静止不变的 species (proportion 稳定) 不累加 activity;只有真正 "扩张中" 的 species 才累加,这避免了 "stasis 充数" 的 artifact。参考 Bedau & Packard (1996) 与 Droop & Hickinbotham (2012, https://doi.org/10.1162/978-0-262-31050-5-ch007)。

Global activity: $EA^*(t) = \sum_p a_p^*(t)$, $* \in \{C, N\}$。

---

## 6. 主要结果与解读

### 6.1 Random search (5.1)

105 个 random parameter sets:
- **Flow-Lenia**: 大部分产生 SLPs (Fig 3a)
- **Lenia**: 大部分要么 vanishing 要么 exploding (Fig 3b)

直观对比图非常 striking —— Flow-Lenia 把搜索空间的有效 region 大幅扩张。Mass conservation 作为 **regularizer**,把 dynamics 推向 SLP-attractor。

还观察到几类典型 patterns:
- Gyrating SLPs (Fig 4a)
- Snake-like with attraction/repulsion (Fig 4b)
- Dividing/merging dots (reaction-diffusion-like, Fig 4c)
- 复杂 membrane + organoid + central nuclei (Fig 4d-f),最终 phase transition
- 2-channel modular creatures (Fig 4g-i)
- **Temperature critical regime** (Fig 4j): linearly 增加 $s$,观察到 Turing-like phase (中) 和 equilibrium phase (右) 的边界最 dynamic

这最后一个观察非常有意思 —— 暗示 Flow-Lenia 在 critical regime 附近最 life-like,与 criticality 与 life 的关联猜想 (Kauffman, Mora 等) 呼应。

### 6.2 Directed search (5.2)

Fig 5a 显示:
- Flow-Lenia directed motion 在 2 channel / 20 kernels 下快速收敛到 high fitness
- 1 channel 难,但 5000 generations 后也能达到相近 fitness
- Kernels 多 → 收敛快
- **Lenia (黄线) 优化不稳定,且只发现 exploding patterns**

四个 task 都成功:
- **Directed motion** (Fig 5b): 2 channels 间 attraction/repulsion 推进
- **Angular motion** (Fig 5c): 周期性 180° turn,内部 dynamics 复杂
- **Navigation through obstacles** (Fig 5d): "forest" 中保持 integrity
- **Chemotaxis** (Fig 5e): 完美沿 gradient 爬升

对比 Hamon et al. 2024 (https://arxiv.org/abs/2402.10236) 在 Lenia 上做类似 task 需要 IMGEP + curriculum learning + differentiable CA + gradient descent,而 Flow-Lenia 仅用 vanilla ES。Mass conservation 提供 intrinsic robustness —— creature 不会因 perturbation 消失,因为 mass 不会凭空蒸发。

### 6.3 Intrinsic evolutionary dynamics (5.3)

**Evolutionary tree visualization**:
- 用 PCA 把高维 parameter space $\mathcal{P} = \bigcup_t \mathcal{P}^t$ 投影到 2D / 3D
- Plot $\mathcal{P}^t$ over time, color-coded by time
- 得到树状 trajectory (Fig 7a, 7b) —— 明确的 speciation branching
- Deterministic sampling rule → 更细更清晰的 branches (Fig 7b)
- Stochastic sampling → 更 noisy 但 tree structure 仍在

这暗示 **intrinsic fitness landscape** 的存在 —— 不是 random walk。

**Parameter count vs mutation rate**:
- $|\mathcal{P}|$ 与 $p_{mut}$ 呈 **sub-linear** 关系 (Fig 8) → competitive dynamics 在限制 species 数量
- 若无 competition,应是线性
- 参数数量 initial rapid growth 后趋于稳定 → intrinsic regulation

**Evolutionary activity vs mutation rate** (Fig 9):
- 高 $p_{mut}$ 反而降低 EA
- Power law fit:
  - $EA^N$: $\gamma = -0.5$, $R^2 = 0.75$
  - $EA^C$: $\gamma = -0.71$, $R^2 = 0.71$
- EA vs time 为 linear growth,slope 随 $p_{mut}$ 衰减 (power function)
- 这与 Droop & Hickinbotham (2012) 在 ALife 系统中观察到的 "too high mutation rate kills EA" 一致

**三 model 对比** (Fig 10):
- Raw $EA^C$ / $EA^N$: dissipative > food > vanilla ($p < 10^{-5}$, Mann-Whitney)
- **Corrected by total mass**: 关系反转 —— dissipative 最低!
  - 因为 dissipative model 不断注入新 mass,raw EA 被 mass inflation 抬高
  - Corrected 后 dissipative 的 normalized EA 反而最低
- Diversity: dissipative **最低** (counter-intuitive —— 它唯一注入新 parameters)
  - 原因: input zone 注入的 parameters 从 $\mathcal{N}(0,1)$ 采样,但 evolution 中 parameters 已 drift 出这个 distribution;新 parameters 与 adapted 的旧 parameters 竞争时劣势
- Food: diversity 上升快、稳定快
- Vanilla: diversity 稳定线性增长

**重要 finding**: 实验结果与 "dissipation + limited resources 驱动 OEE" 的生物学直觉**相反** (Bartlett & Wong, 2020, https://doi.org/10.3390/life10040042)。这正好体现了 quantitative model 的价值 —— 提醒我们 origins-of-life 的直觉需要 formalize + 测量,不能直接 take for granted。

---

## 7. 讨论:Limitations 与 Future Work

### 7.1 Species definition 的局限

当前把 "species" 定义为 parameter space 中的一个 **point** —— 两个 parameter sets 即使差别无穷小也算不同 species。这与生物 species 概念差距大。作者注意到 parameters 在 high-dim space 中**形成 clusters**,cluster 间偶尔 branching。更合理的 species definition 可能是这些 coherent cluster (类似生态学中的 cloud-based species concept)。

### 7.2 Individual 的定义

观察到一个 stable "individual creature" 在 simulation 中可能由多种不同 parameters 的 matter 组成。Levin (2023, https://doi.org/10.1007/s00018-023-04790-z) 等提出 agent / individual 才是 unit of selection,而非 gene。Flow-Lenia 是测试这一想法的理想 testbed。Krakauer et al. (2020, https://doi.org/10.1007/s12064-020-00313-7) 的 information-theoretic individuality measures 可以引入。

### 7.3 其他 open-endedness metrics

- Assembly theory (Sharma et al., 2023, https://www.nature.com/articles/s41586-023-06600-9) 在 CA 上的版本 AssemblyCA (Patarroyo et al., 2023)
- 这些可以补充 EA metrics

### 7.4 Phenotypic species definition

未来可把 phenotype (实际 creature shape / behavior) 纳入 species 定义,而不只是 parameter set。

---

## 8. 对 Karpathy 可能感兴趣的几个 angle

基于您过去 work (nanoGPT, micrograd, "Software 2.0" 等) 与对 ALife / artificial life long-standing interest (您曾经多次 tweet Lenia / Game of Life / Neural CA),以下 angle 可能特别 resonant:

### 8.1 Reintegration Tracking 是 differentiable 的吗?

公式 (6) 是一个 grid-based approximation of particle system。$I(x', x)$ 含一个 integral over $\Omega(x)$,实际是 distribution $\mathcal{D}$ 在 square 上的积分。$\mathcal{D}$ 是 uniform square centered at $x'' = x' + dt \cdot F$,所以 $I$ 实际是 "square D 与 square Ω(x) 的 intersection area / (2s)^2"。这 **可微** (soft intersection) 也**可 hard binarize**。JAX 实现里大概率用了某种 soft intersection。可微性意味着可以 backprop 通过 reintegration tracking,做 fully differentiable evolution / meta-learning。

### 8.2 Parameter embedding 与 "weight-agnostic brains"

Parameter embedding 把 $h$ 变成 spatial map $P(x)$ 随 matter 流动 —— 这非常像 GA-based weight-agnostic networks (Stanley's Weight Agnostic Neural Networks, https://weightagnostic.github.io/) 或者更广义的 "morphology + genome co-evolution"。区别在于:Flow-Lenia 的 "genome" 不直接 encode 一个 network,而是 encode 一个**局部 update rule** —— 它决定 matter 如何流动、聚集、组织。这其实是一个 **physics-as-computation** 视角:genome 编码"局部物理定律"。

### 8.3 Critical regime 与 life

Fig 4j 显示在 Turing-like phase 和 equilibrium phase 之间的边界最 dynamic。这非常 reminiscent of self-organized criticality (Bak, Tang, Wiesenfeld) 与 "life at the edge of chaos" (Langton, 1990, https://doi.org/10.1016/0167-2789(90)90057-V)。Flow-Lenia 可作为研究 SOC 与 life 关系的实验台 —— temperature $s$ 是一个干净的 tuning knob。

### 8.4 与 Neural Cellular Automata 的对比

Mordvintsev et al. (2020, https://distill.pub/2020/growing-ca/) 的 Growing Neural CA 用 learned update rule (small ConvNet) 来 grow/morph images。Flow-Lenia 的 update rule 是 hand-defined family (Gaussian kernel + Gaussian growth),但 parameter 可以 locally vary。两者可以 merge —— 用一个小 NN 作为 local growth function $G_i$ 并 embed 其 weights 作为 genome。这会是一个 beautiful hybrid:Neural CA 的 expressive power + Flow-Lenia 的 mass conservation + multi-species 共存。

### 8.5 Open-endedness 的 measurable proxy

Evolutionary activity metrics (Bedau, 1996) 是 ALife 社区长期用的 OEE proxy。Flow-Lenia 给出了一个 clean testbed 来 stress-test 这些 metrics —— 例如 dissipative model 中 raw EA 与 mass-corrected EA 关系反转,正好暴露了 EA metric 的 sensitivity to total mass 的 confound。这与 ML 中 "metric gaming" 问题相通 —— reward hacking 在 ALife 中以 EA-hacking 形式出现。

### 8.6 优化稳定性

ES (OpenES) 在 Lenia 上 unstable,在 Flow-Lenia 上 stable。这非常像 RL 中 reward shaping / constraint 的作用 —— mass conservation 充当了 intrinsic regularizer,使得 reward landscape 更 smooth。其实可以把它看作一种 "physical inductive bias" —— 物理定律 (mass conservation) 作为 prior 限制了 hypothesis space。

### 8.7 JAX 的角色

整个系统用 JAX 实现并跑在 Tesla T4 上,$255 \mu s$ per step。这与 Growing Neural CA、Particle Lenia 等都用 JAX 趋势一致 —— JAX 的 functional + XLA + vmap 让 CA 这类 grid-based SIMD 计算非常高效。这与您自己对 JAX / Equinox 的偏好一致。

### 8.8 Mutation beams 与 curriculum

Mutation 以 10×10 beam 形式 perturb parameter map。这其实是一种 spatially-structured noise injection —— 类似于 SDE 中的 stochastic perturbation,但 spatially correlated。Beam size 是一个 key hyperparameter:太大 → 行为 disruptive;太小 → 立刻被邻居 overwrite。这呼应了 speciation 的 allopatric / parapatric 模式 —— geographical isolation 才能让 mutation 萌芽。

---

## 9. 公式变量速查表

为了 build intuition,把关键变量 / superscript / subscript 总结:

| Symbol | Meaning |
|---|---|
| $A^t$ | state map at time $t$, $A: \mathcal{L} \to \mathbb{R}_{>0}^C$ |
| $A_i^t$ | channel $i$ 的 activation |
| $A_\Sigma^t$ | total mass per cell, $\sum_i A_i^t$ |
| $U^t$ | affinity map (Lenia 中叫 growth) |
| $U_i^t$ | channel $i$ 的 affinity |
| $K_i$ | 第 $i$ 个 kernel |
| $G_i$ | 第 $i$ 个 growth function |
| $h_i$ | 第 $i$ 个 kernel-growth pair 的 weight (参数) |
| $c_0^i, c_1^i$ | source / target channel index |
| $R$ | global neighborhood radius |
| $r_i$ | kernel $i$ 的 scale |
| $a_{i,j}, b_{i,j}, w_{i,j}$ | kernel $i$ 的 ring $j$ 的 center / amplitude / width |
| $\mu_i, \sigma_i$ | growth function $i$ 的 peak / sharpness |
| $F_i^t$ | channel $i$ 的 flow field, $F: \mathcal{L} \to \mathbb{R}^2$ |
| $\alpha^t$ | spatial weighting map, $[0,1]$ |
| $\beta_A$ | critical mass threshold |
| $n$ | switching sharpness exponent (默认 2) |
| $s$ | reintegration distribution 的 "temperature" / variance |
| $dt$ | time step (默认 0.2) |
| $\Omega(x)$ | cell $x$ 的 spatial domain (1×1 square) |
| $\mathcal{D}(m, s)$ | uniform square distribution, center $m$, side $2s$ |
| $I_i(x', x)$ | 从 $x'$ 到 $x$ (channel $i$) 的 mass transfer proportion |
| $P^t$ | parameter map at time $t$, $P: \mathcal{L} \to \Theta$ |
| $\mathcal{P}^t$ | set of unique parameters present at time $t$ |
| $M(p, t)$ | species $p$ 在时刻 $t$ 的 total mass |
| $\rho(p, t)$ | species $p$ 的 mass proportion |
| $a_p^C, a_p^N$ | count-based / non-neutral activity of species $p$ |
| $EA^C, EA^N$ | global evolutionary activity (sum over species) |
| $D(t)$ | diversity (mean pairwise L2 distance of parameters) |
| $p_{mut}$ | mutation beam probability per step |
| $p_{diss}$ | dissipative beam probability |
| $r_{decay}, r_{digest}$ | food model 中的 decay 和 digestion rate |
| $\Psi$ | food map |
| $\nabla U_i^t$ | affinity gradient, Sobel 估计 |
| $-\nabla A_\Sigma^t$ | diffusion term (negative concentration gradient) |

---

## 10. 关键 Web References

- **Paper (Artificial Life journal version)**: https://doi.org/10.1162/artl_a_00471
- **Companion website (videos!)**: https://sites.google.com/view/flow-lenia
- **Earlier ALIFE 2023 version**: https://doi.org/10.1162/isal_a_00651
- **Lenia original**: https://doi.org/10.25088/ComplexSystems.28.3.251
- **Lenia and Expanded Universe**: https://doi.org/10.1162/isal_a_00297
- **Hamon et al. 2024 (sensorimotor agency in CA)**: https://arxiv.org/abs/2402.10236
- **Reintegration tracking (Moroz)**: https://observablehq.com/@moroz/reintegration-tracking
- **Particle Lenia (Mordvintsev et al.)**: https://google-research.github.io/self-organising-systems/particle-lenia.html
- **Growing Neural Cellular Automata (Mordvintsev et al. 2020)**: https://distill.pub/2020/growing-ca/
- **OpenES (Salimans et al. 2017)**: https://arxiv.org/abs/1703.03864
- **EvoSax**: https://arxiv.org/abs/2212.04180
- **Adam optimizer**: https://arxiv.org/abs/1412.6980
- **JAX**: https://github.com/google/jax
- **Evolutionary activity (Bedau & Packard)**: https://doi.org/10.1162/978-0-262-31050-5-ch007
- **Conservation increases EA (Hickinbotham & Stepney)**: https://doi.org/10.1162/978-0-262-33027-5-ch024
- **Minimal criterion for OEE (Soros & Stanley)**: https://doi.org/10.1162/978-0-262-32621-6-ch128
- **Taylor 2015 OEE requirements**: https://arxiv.org/abs/1507.07403
- **Lyfe (Bartlett & Wong)**: https://doi.org/10.3390/life10040042
- **Levin 2023 (agential materials)**: https://doi.org/10.1007/s00018-023-04790-z
- **Krakauer et al. info-theoretic individuality**: https://doi.org/10.1007/s12064-020-00313-7
- **Assembly theory (Sharma et al.)**: https://www.nature.com/articles/s41586-023-06600-9
- **Evoloops (Sayama)**: https://doi.org/10.48550/arXiv.2402.03961
- **Weight Agnostic NN (Stanley)**: https://weightagnostic.github.io/
- **Sayama evolutionary CA survey**: https://doi.org/10.48550/arXiv.2402.03961

---

## 11. 一句话总结

Flow-Lenia 把 Lenia 的 hard clip 替换为 mass-conserving flow + reintegration tracking,这一物理约束让 SLPs 从"难找的例外"变成"random search 的默认产物",让 directed optimization 从"需要 curriculum + IMGEP"降级到"vanilla ES",并 enable parameter embedding 进而 enable multi-species simulation。Evolutionary tree 在 PCA 子空间中自发形成,demonstrating intrinsic fitness landscape —— 但 dissipation 与 food constraints 并未如生物学直觉预测的那样提升 normalized EA,反而 vanilla model 表现最好,这本身就是 quantitative ALife model 的价值:它迫使我们 formalize intuition。下一步大方向是更合理的 species / individual 定义,以及引入 assembly-theoretic 或 information-theoretic 的 open-endedness metrics。
