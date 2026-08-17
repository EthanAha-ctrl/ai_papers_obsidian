---
source_pdf: DCARL- A Divide-and-Conquer Framework for.pdf
paper_sha256: f85486f915cdf469c7d7641bc7c7cbeb21fc167bbd73c6e7fd8f78e94ff8430c
processed_at: '2026-08-03T18:27:39-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DCARL 人话版

## 一句话总结

**别一帧一帧往后推（会越推越歪），先在整条线上稀疏地钉几个锚点，再在锚点之间填空。**

---

## 问题在哪

你让模型生成 32 秒视频（320 帧），它一帧一帧往后推：第 1 帧→第 2 帧→…→第 320 帧。

问题是每推一步都会犯点小错。第 10 帧的小错会传给第 11 帧，第 11 帧的小错会传给第 12 帧……到第 320 帧时错误已经累积成灾难。

更糟的是：模型训练时学的是"ground-truth 上下文 → 下一帧"，但推理时上下文是自己生成的烂帧。这种 train-test mismatch 让误差更严重。这就是 **exposure bias**。

具体表现：
- 视频越往后越糊、越跑偏
- Camera 指令一开始还跟得住，到后面完全失控（因为生成内容已经脱离 valid manifold，模型没法再正确解读 pose 条件）

之前的修补办法（Diffusion Forcing 给条件加噪、Self Forcing 用 self-rollout 训练）只是缓解局部 drift，没给一个 explicit error bound。到长 horizon 还是崩。

---

## DCARL 的核心思路

**分而治之：把"一条长线"切成"一堆短线段"，每段两端有锚点钉死。**

### 第一阶段：生成 keyframes（全局锚点）

一次性 jointly 生成 21 个稀疏 keyframes（间隔约 8 帧）。

关键：**这些 keyframes 是一起生成的，不是一步步推出来的**。所以它们之间没有 causal dependency，没有 AR 误差累积。

每两个 keyframe 之间间隔 8 帧，32 秒视频总共 ~320 帧，21 个 keyframe 刚好覆盖。每个 keyframe 带对应时刻的 camera pose。

### 第二阶段：interpolation（局部填空）

每两个 keyframe 之间是一个 segment，用 AR 方式生成中间的 dense frames。

每个 segment 生成时，condition 三样东西：
1. **前一段的末尾几帧**（local momentum，保证段间连续）
2. **当前 segment 前后的 keyframes**（global anchor，防止跑偏）
3. **当前 segment 的 camera poses**

---

## 为什么这样不会 drift

这是 paper 的灵魂，用控制论思路解释。

### Pure AR：无界误差累积

每步误差 $e_t$ 递归传递：$e_{t+1} \approx J \cdot e_t + \eta_{t+1}$

- $J$ 是 Jacobian（误差传递矩阵）
- $\eta_{t+1}$ 是单步新引入的误差

展开到 N 步：$\|e_N\| \le \sum L^{N-t}\|\eta_t\|$

- $L=1$ 时线性发散 $\mathcal{O}(N)$
- $L>1$ 时指数发散

更狠的是**下界**：只要模型不主动 shrink 信号（$\sigma_{\min}(J) \ge 1$），且每步有非零 systematic drift（$\|\mathbb{E}[\eta_t]\| \ge \mu$），误差至少 $N\mu$ 线性增长。**MSE 至少 $\Omega(N)$，无药可救。**

### DCARL：有界误差

每个 segment 建模成 **Brownian Bridge**——两端钉死在 keyframes 上的随机过程。

中间任意帧的误差拆成三部分：

1. **Anchor error** $E_{\text{anch}}$: keyframe 自己的误差线性插值。如果 keyframes 是 globally generated，这部分是 $\mathcal{O}(1)$
2. **Momentum leakage** $E_{\text{leak}}$: 前一段传过来的"动量误差"
3. **Local noise** $E_{\text{noise}}$: 当前 segment 内的随机噪声

核心要证 $E_{\text{leak}}$ 不发散。

### Lemma 1: 动量误差每段衰减一半

模型为了既保持连续又 adhere keyframes，隐式最小化误差轨迹的"加速度能量"：

$$L = \int_0^T (\delta''(\tau))^2 d\tau$$

这相当于让误差轨迹在 segment 内尽量"平直"。

边界条件：
- 两端位置误差 = 0（被 keyframes 钉死）
- 初速度 = 前一段末速度（连续性）
- 末端加速度 = 0（自然边界）

解出来：**每段末尾的动量误差 = 前一段末尾的 $-1/2$**

$$\Delta v_i = -\frac{1}{2}\Delta v_{i-1}$$

几何级数衰减，累积上界 $0.384 \cdot T \cdot \|\Delta v_0\|$。**只依赖 keyframe 间隔 $T$，与总长度 $N$ 无关。**

### 最终 bound

$$\|e_t\| \le \underbrace{\max_j\|e(K_j)\|}_{\text{keyframe 质量}} + \underbrace{0.384\,T\|\Delta v_0\|}_{\text{有界动量泄漏}} + \underbrace{\frac{\sqrt{T}}{2}\sigma_{\text{int}}}_{\text{局部噪声}}$$

- Keyframes globally generated: $\mathcal{O}(1)$，全局稳定
- Keyframes AR generated with step $T$: $\mathcal{O}(N/T)$，比 pure AR 抑制 $T$ 倍（MSE 上 $T^2$ 倍）

**人话：钉了锚点之后，误差被关在每个 segment 里，不会跨段滚雪球。前一段的"速度误差"每过一段减半，所以不管视频多长，都不会发散。**

---

## 几个关键的工程细节

### 1. Keyframe 用 image-level 编码，不用 temporal VAE

标准 temporal VAE 有 4:1 temporal compression。对 sparse keyframes（间隔大、空间位移大）这种 compression 是 lossy 的，fine-grained geometric cues 丢失，camera encoder 学不准。

DCARL 把 keyframes 的 frame dimension reshape 成 batch dimension，当独立 images 编码。每个 keyframe 保留 full spatial fidelity。

Ablation 证实：ATE 从 0.155 → 0.100，ARE 从 7.276 → 3.999。空间细节对 camera control 至关重要。

### 2. 给 keyframe 加噪声，防"偷懒复制"

如果不加噪声，模型发现直接 copy keyframe 像素就能 minimize training loss——这是 identity mapping shortcut。结果：keyframe 附近画面"停住不动"，motion stagnation。

解决：给 keyframe latents 加固定 noise level：

$$\tilde{\mathbf{z}}_K = 0.7 \cdot \mathbf{z}_K + 0.3 \cdot \epsilon$$

打破 pixel-level shortcut，强迫模型学 underlying motion dynamics 而非 anchor 到 visual guide。

Ablation 证实：FID 几乎不变（单帧看还行），但 FVD 从 246.5 → 201.7（temporal dynamics 显著改善）。**这是个 temporal artifact 不是 spatial artifact。**

### 3. Segment 边界做 overlap + latent substitution

相邻 segment 之间 overlap 1 帧，denoising 每步把当前段第 1 帧的 latent 替换成前一段末尾的 latent，且这个 historical latent 保持 noise-free。

关键：**训练时就要让模型适应这种 deterministic boundary conditioning**。如果只在 inference 时做 substitution，模型期待 noisy input 却拿到 clean input，会出 artifact。

### 4. Keyframe selection policy

每个 segment 拿到的 keyframes 不只是它"内部"的，还要包括**前一段末尾的 keyframe 和后一段开头的 keyframe**（look-back + look-ahead）。

只 look-back 的话，segment 末尾离 anchor 越远 drift 越大。Look-ahead 把末尾拉回 anchor。

---

## 实验结果怎么看

### 32s on ODV-YT (Table 1)

关键看 **24-32s 的 FID**（最考验长程稳定性）：
- DiffF: 25.3 → 54.1（涨 114%）
- SelfF: 26.2 → 119.4（涨 356%）
- DCARL: 19.6 → 28.6（涨 46%）

**纯 AR 方法后期 FID 暴涨 2-4 倍，DCARL 涨不到 50%。这就是理论 $\mathcal{O}(N)$ vs $\mathcal{O}(1)$ 的实验体现。**

Camera 跟踪（ATE）：
- DiffF: 0.469
- DCARL: 0.237

### Zero-shot on nuScenes

完全没见过的数据集，DCARL 的 ATE 0.045 vs SEVA 0.117，好 2.6 倍。Keyframe anchoring 让 camera control 泛化得很好。

### Ablation: 去掉 keyframes 会怎样

Table 7：去掉 keyframe conditioning，32s 视频：
- FID 从 19.2 → 25.2
- 24-32s FID 从 28.6 → 45.2
- ATE 从 0.237 → 0.387

**一去掉 keyframe，立刻退化成 pure AR 的 drift 模式。这就是 Proposition 1 的实验验证。**

---

## 直觉总结

把 DCARL 想成**盖房子**：
- Pure AR 像一砖一砖往上摞，底下歪了上面全歪
- DCARL 像先立钢筋骨架（keyframes），再往里填混凝土（interpolation）。骨架正了，填的再差也是局部问题

或者用**导航**类比：
- Pure AR 像只看眼前 1 米往前走，走 320 步可能歪到哪都不知道
- DCARL 像先在地图上标 21 个 waypoint（全局一起看，不累积误差），然后每两个 waypoint 之间闭着眼走 8 步，走歪了下一个 waypoint 会把你拉回来

理论上的核心 elegant 之处：**$\gamma = -1/2$ 这个 damping coefficient**。它来自"最小化误差轨迹加速度"这个 energy functional。每过一段动量误差减半，几何衰减，所以无论视频多长都不会发散。

这 paper 本质上把**控制论的 Lyapunov 稳定性思想**引进了 video diffusion：pure AR 是不稳定系统（误差无界），加 keyframe anchor 之后变成有界误差系统。这个视角比单纯"加 noise 缩小 train-test gap"要深刻得多。

---

# DCARL: Autoregressive Long-Trajectory Video Generation via Divide-and-Conquer

来给个 deep dive。这篇 paper 解决的问题本质上是 **"如何在 long horizon 下让 autoregressive video diffusion 不 drift"**，给出的答案是 **"用 keyframe anchor 把无界累积误差变成有界局部误差"**。下面拆开讲。

---

## 1. The Core Problem: Why Pure AR Fails on Long Trajectories

Long-trajectory video generation 的核心难点是 **exposure bias × error compounding**。给定条件 $\mathcal{C}$，AR 视频生成分解为：

$$p(\mathcal{V}|\mathcal{C}) = \prod_{i=1}^{S} p(S_i | S_{<i}, \mathcal{C})$$

- $\mathcal{V}$: 整个 video sequence，被切成 $S$ 个 block $\{S_1, \ldots, S_S\}$
- $B$: block size，token-level / frame-level / segment-level AR 由 $B$ 决定
- $S_i$: 第 $i$ 个 block（包含 $B$ 帧）

训练时模型在 ground-truth context 上学习，推理时用 self-generated frames 当 context。这个 train-test mismatch 在每个 step 注入一个小误差 $\eta_t$，误差逐步累积。当 trajectory 长度上去（例如 32s @ 10fps = 320 frames），即使每步误差很小，累积下来也会让 scene 跑出 valid manifold，进而让 camera pose 条件失效。

之前的工作（[Diffusion Forcing](https://boyuan.space/diffusion-forcing/)、[Self Forcing](https://self-forcing.github.io/)）想用 noisy conditioning / self-rollout 训练来缩小 distribution gap，但只是 **mitigate local drift**，没有 explicit error bound。Attention sink 类方法（[Deep Forcing](https://arxiv.org/abs/2512.05081)）又会让 model 对动态 camera 指令 responsiveness 变差。

---

## 2. The Key Insight: Bound Error by Structural Anchors

DCARL 的 insight 可以用一句话概括：**"把 unbounded recursive drift 替换成 bounded interpolation error"**。

具体做法分两阶段：

### Stage 1: Keyframe Generator $G_K$（全局结构锚点）
联合生成稀疏 keyframe set $\mathcal{K} = \{I_{k_1}, \ldots, I_{k_M}\}$，通过 DiT-based flow matching：

$$\mathcal{K} = \mathcal{D}\big(\Phi_K(\epsilon \mid \mathcal{E}(I_0), \mathcal{P}, \mathcal{C})\big)$$

- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: 初始 Gaussian noise
- $\mathcal{E}, \mathcal{D}$: VAE encoder / decoder
- $\mathcal{P} = \{c_{k_1}, \ldots, c_{k_M}\}$: 从全局 trajectory $\tau$ 在 keyframe 时刻采样的 camera poses
- $\Phi_K$: keyframe flow matching network

**关键设计：所有 keyframes 是 jointly generated（一次 forward 出全部 21 个 keyframe）而非 sequentially**。这样 keyframe 之间没有 causal dependency → 没有 AR 误差累积。

### Stage 2: Interpolation Generator $G_I$（局部 dense frame 生成）
对每个 segment $S_i$：

$$S_i = G_I(\mathcal{H}_i, \mathcal{K}_i, \mathcal{T}_i, \mathcal{C}) = \mathcal{D}\big(\Phi_I(\mathbf{z}_{in}, \mathcal{T}_i, \mathcal{C})\big)$$

- $\mathcal{H}_i$: 过去 $p$ 帧 history
- $\mathcal{K}_i \subset \mathcal{K}$: 当前 segment 周围的局部 keyframe subset（look-back + look-ahead）
- $\mathcal{T}_i$: 当前 segment 的 camera poses
- $\mathbf{z}_{in} = [\epsilon, \mathbf{z}_{\mathcal{H}_i}, \mathbf{z}_{\mathcal{K}_i}]$: noise + history latents + keyframe latents 沿时间维 concat

**这是 divide-and-conquer 的精髓：global keyframes 限定全局结构，local interpolation 只在两个 anchor 之间做有界 perturbation。**

---

## 3. Theoretical Analysis: Proposition 1 (这是 paper 的灵魂)

附录 A 给的证明非常 clean，值得仔细读。我把它分两块拆。

### Part I — Pure AR 的发散下界

把 generation 写成 $I_{t+1} = \mathcal{F}(I_t, \epsilon_t)$。误差 $e_t = I_t - I_t^*$。一阶 perturbation：

$$e_{t+1} \approx \nabla_{I_t^*}\mathcal{F} \cdot e_t + \eta_{t+1}$$

- $\nabla_{I_t^*}\mathcal{F}$: 在 ground-truth 处的 Jacobian，描述误差如何被传递
- $\eta_{t+1} = \mathcal{F}(I_t^*, \epsilon_t) - I_{t+1}^*$: 在 ground-truth context 下的 single-step drift

假设 $\mathcal{F}$ 是 $L$-Lipschitz，递归展开：

$$\|e_N^{\text{base}}\| \le \sum_{t=1}^{N} L^{N-t} \|\eta_t\|$$

- $L=1$ 且 $\|\eta_t\| \le \eta$: linear divergence $\mathcal{O}(N)$
- $L>1$: exponential divergence $\mathcal{O}(L^N)$

**但这是上界。** 真正狠的是 **下界**，因为实际系统不会让 $L<1$（否则 video 信号会 decay）。

下界推导用了两个假设：
1. **Exposure bias 假设**: $\|\mathbb{E}[\eta_t]\| \ge \mu > 0$，即每步的 systematic drift 不会消失
2. **Non-contractive 假设**: $\sigma_{\min}(\mathbf{J}_t) \ge 1$，即 Jacobian 的最小奇异值 $\ge 1$（model 不会主动 shrink latent state space，否则视觉信号会衰减）

在这两个假设下：

$$\|\mathbb{E}[e_N^{\text{base}}]\| \ge \sum_{t=1}^{N}\Big(\prod_{k=t}^{N-1}\sigma_{\min}(\mathbf{J}_k)\Big)\mu \ge N\mu$$

- $\prod_{k=t}^{N-1}\mathbf{J}_k$: 从 step $t$ 到 $N$ 的误差 transition 矩阵
- 因为 $\sigma_{\min} \ge 1$，乘积 $\ge 1$，所以总 systematic bias 至少 $N\mu$

MSE = Bias$^2$ + Variance：
- Bias 平方: $\Omega(N^2)$
- Variance: 假设 $\eta_t$ 是 i.i.d. 方差 $\sigma^2$，则 $\text{Tr}(\text{Cov}(e_N)) = N\sigma^2 = \Omega(N)$

**所以 pure AR 的 MSE 至少是 $\Omega(N)$，物理意义是没有 corrective mechanism，drift 不可逆。**

### Part II — DCARL 的误差 bound（Brownian Bridge + Damping）

把每个 segment 建模成 **Brownian Bridge pinned at keyframes** + AR residual：

$$I_{iT+\tau} = B_\tau + \Phi_i(\tau)$$

- $T$: keyframe 间隔
- $\tau \in (0, T)$: segment 内的时间 offset
- $B_\tau$: Brownian Bridge，钉在 $K_i$ 和 $K_{i+1}$ 上
- $\Phi_i(\tau)$: 来自前一段 $S_{i-1}$ 的 momentum 残差

Brownian Bridge 的统计量：

$$\mathbb{E}[B_\tau] = \Big(1-\frac{\tau}{T}\Big)K_i + \frac{\tau}{T}K_{i+1}$$

$$\text{Var}(B_\tau) = \frac{\tau(T-\tau)}{T}\sigma_{\text{int}}^2$$

- $\sigma_{\text{int}}^2$: interpolation noise intensity

误差分解成三项：

$$e_{iT+\tau}^{\text{ours}} = \underbrace{\Big(1-\frac{\tau}{T}\Big)e(K_i) + \frac{\tau}{T}e(K_{i+1})}_{E_{\text{anch}}} + \underbrace{\delta_i(\tau)}_{E_{\text{leak}}} + \underbrace{w_\tau}_{E_{\text{noise}}}$$

- $E_{\text{anch}}$: keyframe 误差的线性插值
- $E_{\text{leak}}$: momentum leakage from previous segment
- $E_{\text{noise}}$: local stochastic hallucination

**关键：要证 $E_{\text{leak}}$ 不发散。**

### Lemma 1 — Momentum Damping Coefficient $\gamma = -1/2$

模型为了维持 temporal continuity 同时 adhere keyframes，隐式最小化 acceleration 的能量泛函：

$$L(\delta_i) = \int_0^T (\delta_i''(\tau))^2 \, d\tau$$

- $\delta_i''(\tau)$: 误差轨迹的二阶导（即 acceleration）
- 这相当于让 error trajectory 在 segment 内尽量"直"

Euler-Lagrange 方程（对 $f = (\delta_i'')^2$）：

$$\frac{d^2}{d\tau^2}\Big(2\delta_i''(\tau)\Big) = 0 \implies \delta_i^{(4)}(\tau) = 0$$

通解是 cubic polynomial：$\delta_i(\tau) = a\tau^3 + b\tau^2 + c\tau + d$

边界条件：
- $\delta_i(0) = 0, \delta_i(T) = 0$: keyframes 是 rigid anchors（位置误差被强制 reset）
- $\delta_i'(0) = \Delta v_{i-1}$: velocity continuity（前一段末尾的误差速度作为初速度）
- $\delta_i''(T) = 0$: natural boundary condition（末端自由）

解这个代数系统得：

$$\Delta v_i = \delta_i'(T) = -\frac{1}{2}\Delta v_{i-1}$$

**这是 paper 最 elegant 的结果：momentum error 每过一个 segment 减半，几何级数衰减！**

Intra-segment peak error（对 $\delta_i'(\tau)=0$ 求极值，得 $\tau^* = T(1-\sqrt{3}/3)$）：

$$\max_{\tau \in [0,T]}\|\delta_i(\tau)\| = \frac{\sqrt{3}}{9}T\|\Delta v_{i-1}\| \approx 0.192 \, T \|\Delta v_{i-1}\|$$

累积（geometric series）：

$$\sup_{i,\tau}\|\delta_i(\tau)\| \le 0.192 T \cdot \sum_{k=0}^{\infty}|\gamma|^k \|\Delta v_0\| = 0.192 T \cdot 2\|\Delta v_0\| = 0.384\,T\|\Delta v_0\|$$

**这个 bound 只依赖 $T$ 和初始 momentum error，与总长度 $N$ 无关 → 全局 stable。**

Interpolation noise 的上界（对 $\tau$ 求二次函数极值）：

$$\max_\tau \text{Var}(B_\tau) = \frac{T}{4}\sigma_{\text{int}}^2 \implies \|E_{\text{noise}}\| \le \frac{\sqrt{T}}{2}\sigma_{\text{int}}$$

### Final Unified Bound

$$\|e_t^{\text{ours}}\| \le \max_j \|e(K_j)\| + 0.384\,T\|\Delta v_0\| + \frac{\sqrt{T}}{2}\sigma_{\text{int}}$$

两种 keyframe 生成策略：
- **Globally generated keyframes**: $\|e(K_j)\| = \mathcal{O}(1)$ → 总误差 $\mathcal{O}(1)$
- **AR generated keyframes with step $T$**: $\|e(K_j)\| = \mathcal{O}(N/T)$ → 总误差 $\mathcal{O}(N/T)$，相比 pure AR 的 $\mathcal{O}(N)$ 抑制了 $T$ 倍（MSE 上 $T^2$ 倍）

**这就是 DCARL 的核心理论保证：divide-and-conquer 把 unbounded 误差换成了 bounded 误差。**

---

## 4. Architecture Details

### 4.1 Spatial-Structural Preservation（避免 temporal VAE 损失）

标准 temporal VAE（如 Wan2.1 用的）有 4:1 temporal compression。对于 sparse keyframes（间隔大、空间位移大），temporal compression 会 lossy，fine-grained geometric cues 丢失，camera encoder 学不准。

DCARL 的做法：**把 keyframes 的 frame dimension reshape 成 batch dimension**，当成独立 images 编码。每个 keyframe 保持 full spatial fidelity。

Ablation（Table 3，16s ODV-YT）：
| Model Design | FID↓ | ATE↓ | ARE↓ |
|---|---|---|---|
| Temporal compression | 19.6 | 0.155 | 7.276 |
| Ours (image-level) | 16.3 | 0.100 | 3.999 |

ATE 从 0.155 降到 0.100，ARE 从 7.276 降到 3.999，证实 spatial detail 对 camera control 很关键。

### 4.2 Keyframe Sampling Strategy

- 训练时 $\Delta k \in \{4, 8, 16\}$ 随机采样（robustness across motion scales）
- 测试时固定 $\Delta k_{\text{test}} = 8$
- $|K| = 21$ keyframes

Ablation（Table 4，$\Delta k_{\text{gen}}$ = keyframe generation stride, $\Delta k_{\text{int}}$ = interpolation conditioning stride）：

| $(\Delta k_{\text{gen}}, \Delta k_{\text{int}})$ | (4,4) | (4,8) | (4,16) | (8,8) | (8,16) | (16,16) |
|---|---|---|---|---|---|---|
| FID↓ | 21.4 | 21.0 | 20.9 | **19.3** | 19.0 | 18.8 |
| FVD↓ | 201.8 | 199.9 | 204.6 | **196.0** | 201.5 | 187.8 |
| ATE↓ | 0.148 | 0.147 | 0.131 | **0.096** | 0.098 | 0.104 |

**Insight**: $\Delta k_{\text{gen}}$ 是主要 determinant（越大 keyframe 质量越好），$\Delta k_{\text{int}}$ 影响小。但 $\Delta k_{\text{gen}}$ 太大时 keyframe 跨度过大，keyframe generator 本身难学，所以 (8,8) 是 sweet spot。

### 4.3 Motion-Inductive Noisy Conditioning（防 keyframe duplication）

**Failure mode**: 给 interpolation model 喂 clean keyframes 时，model 会学 "copy-paste shortcut"——直接复制 keyframe 像素而非学 motion dynamics，导致 stuttering。

DCARL 的 trick：**在 keyframe latents 上注入固定 noise level**：

$$\tilde{\mathbf{z}}_{K_i} = \alpha_c \mathbf{z}_{K_i} + \sigma_c \epsilon, \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

- $\alpha_c = 0.7$: keyframe guidance 强度
- $\sigma_c = 0.3$: noise 强度
- 训练时 $\alpha_c \in [0.1, 0.5]$ 随机，inference 时固定 0.7

物理意义：noise 打破 pixel-level shortcut，强制 model 去学 underlying motion dynamics 而非 anchor 到 visual guide。

Ablation（Table 5）：
| Training Design | FID↓ | FVD↓ |
|---|---|---|
| W/o noisy keyframe | 19.0 | 246.5 |
| Train w/ noisy keyframe | 19.9 | 240.9 |
| Train/infer w/ noisy keyframe | 19.3 | **201.7** |

**Insight**: FID 几乎不变（因为 frame 单独看都还行），但 FVD 暴跌（temporal dynamics 显著改善）——证实 keyframe duplication 是 temporal artifact 而非 spatial artifact。Fig. 5 显示不加 noise 时有 fade-like blending 和 motion stagnation 两种 artifact。

### 4.4 Seamless Boundary Consistency（防 segment 边界 flickering）

设计：相邻 segment 之间 $p$-frame overlap（实验中 $p=1$），denoising 每一步把 segment $S_i$ 的前 $p$ 个 latents 替换成前一段的 historical latents $\mathbf{z}_{\mathcal{H}_i}$，且 $\mathbf{z}_{\mathcal{H}_i}$ 保持 noise-free。

Ablation（Table 6）：
| Design | MS↑ | PSNR↑ | SSIM↑ |
|---|---|---|---|
| Overlap only | 0.969 | 18.15 | 0.534 |
| + Sub | 0.968 | 17.95 | 0.524 |
| + Sub + Train | **0.972** | **18.75** | **0.562** |

**Insight**: 仅 overlap 不够（前段最后 frame 偏离后段期望起始 pose）；naive latent substitution 引入 distribution mismatch（model 期待 noisy input，给 clean input 会出 artifact）；只有 train 时就让 model 适应 deterministic boundary conditioning 才能解决。

### 4.5 Keyframe Selection Policy

对 segment $S_i = \{I_{t_i^*}, \ldots, I_{t_i^*}\}$，选局部 keyframe：

$$\mathcal{K}_i = \{I_k \in \mathcal{K} \mid k_{\text{pre}} \le k \le k_{\text{next}}\}$$

- $k_{\text{pre}} = \max_{k: I_k \in \mathcal{K}, k < t_i^*} k$: segment 起点之前最近的 keyframe
- $k_{\text{next}} = \min_{k: I_k \in \mathcal{K}, k > t_i^*} k$: segment 终点之后最近的 keyframe

**这个 look-back + look-ahead 设计很重要**：纯 look-back 会让 segment 末尾 drift（因为离 anchor 越远 error 越大），look-ahead 把 segment 末尾拉回 anchor。

训练时 $|\mathcal{K}|$ 从 [1, 10] 随机采样，提升 robustness。

---

## 5. Implementation Details

| Component | Detail |
|---|---|
| Base model | [Wan2.1-T2V-1.3B](https://github.com/Wan-Video/Wan2.1) |
| Camera control | [ReCamMaster](https://arxiv.org/abs/2506.15570) 的 camera feature map 加到 DiT block 的 visual tokens |
| Optimizer | AdamW (LR 5e-5, weight decay 0.01, $\beta=(0.9, 0.95)$) |
| Effective batch size | 16 |
| Hardware | 8× NVIDIA H100 |
| Steps | 30,000 |
| Dataset | [OpenDV-YouTube](https://github.com/OpenDriveLab/OpenDV) 480h 子集，1min clips @ 10fps |
| Camera poses | [$\pi_3$](https://arxiv.org/abs/2507.13347) 每 0.5s 重建，中间 slerp + linear interp |
| Captions | [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215) 每 20s 采样 frame 生成 |
| Keyframe $|K|$ | 21 (训练), stride $\{4,8,16\}$; test stride 8 |
| Noisy keyframe | $\alpha_c=0.7, \sigma_c=0.3$ |
| Boundary overlap | $p=1$ |

---

## 6. Experimental Results

### 6.1 Main Comparison: 32s on ODV-YT (Table 1)

| Method | Overall FID | Overall FVD | 0-8s FID | 24-32s FID | ATE↓ | ARE↓ |
|---|---|---|---|---|---|---|
| DiffF | 35.0 | 664.1 | 25.3 | 54.1 | 0.469 | 19.448 |
| SelfF | 58.0 | 2113.6 | 26.2 | 119.4 | 0.610 | 14.386 |
| DeepF | 42.3 | 1558.5 | 26.1 | 77.2 | 0.571 | 15.144 |
| Vista | 66.7 | 1550.0 | 29.1 | 134.1 | 0.641 | 19.332 |
| SEVA | 22.2 | 548.0 | 27.2 | 33.1 | 0.294 | 8.527 |
| **Ours** | **19.2** | **203.7** | **19.6** | **28.6** | **0.237** | **7.669** |

**关键观察**：
- 纯 AR 方法（DiffF, SelfF, DeepF, Vista）从 0-8s 到 24-32s 的 FID 暴涨 2-4 倍，证实 theoretical $\mathcal{O}(N)$ drift
- SEVA 是 non-AR baseline，相对稳定但 quality 不够高
- DCARL 从 19.6 → 28.6，相对 drift 很小（49% 增长 vs DiffF 的 114%）

### 6.2 Zero-shot on nuScenes (Table 2)

| Method | FID | FVD | ATE↓ | ARE↓ |
|---|---|---|---|---|
| SEVA | 35.9 | 487.9 | 0.117 | 6.289 |
| **Ours** | **19.6** | **225.4** | **0.045** | **5.274** |

DCARL 在 nuScenes 上 zero-shot 泛化优秀，ATE 比 SEVA 好 2.6 倍。

### 6.3 DL3DV-10K (Table A)

DL3DV 是 small-scene dataset。DCARL FID 22.8 vs SEVA 24.8，FVD 220.1 vs 275.0，ARE 19.646 vs 68.129（差距巨大）。

---

## 7. Ablation: Necessity of Keyframe Anchoring (Table 7)

去掉 keyframe conditioning，32s 视频结果：

| Method | Overall FID/FVD | 0-8s | 24-32s | ATE | ARE |
|---|---|---|---|---|---|
| w/o Keyframe | 25.2/376.7 | 21.1/247.1 | 45.2/665.3 | 0.387 | 12.184 |
| Ours | 19.2/203.7 | 19.6/191.4 | 28.6/313.8 | 0.237 | 7.669 |

**没有 keyframe，又退化成 pure AR 的 drift 模式**。这是 Proposition 1 的实验验证。

---

## 8. Failure Modes & Limitations

Fig. D 展示三类 failure：
1. **Long-distance perception**: roundabout 中心植被位置错乱； pedestrian island 被穿过 → 缺 fine-grained long-range geometric understanding
2. **Corner cases**: 从 overpass 下出来的 transition 异常 → 训练分布不够 diverse
3. **Real-time generation**: 当前 non-real-time，但作者提出 dual-stream 蓝图：causal model 低延迟生成 + sparse model 维护 global anchor

---

## 9. My Take (Build Intuition)

**这套 framework 的 elegance 在于把控制论/动力系统的稳定性思想引进了 video diffusion**：
- Pure AR 是 unbounded recursive system，误差是 Lyapunov 不稳定的
- DCARL 用 keyframe 当 "reset boundary"，每个 segment 内误差被强制 reset 到 0，momentum 以 $\gamma=-1/2$ 几何衰减
- Brownian Bridge 是个非常好的 abstraction：interpolation 在 anchor 之间，variance 在中点最大（$T/4 \sigma_{\text{int}}^2$），这个 bound 解释了为什么 stride 不能太大（$T$ 大了 interpolation variance 大）

**值得深挖的方向**：
- Lemma 1 的 $\gamma = -1/2$ 这个 magic number 是怎么来的？它依赖于 $L = \int (\delta'')^2 d\tau$ 这个特定的 energy functional。如果换成 $L = \int (\delta')^2 d\tau$（一阶）或者 $L = \int (\delta'''')^2 d\tau$（四阶），$\gamma$ 会变。这是个值得 explore 的 design space
- Keyframe generator 现在是 jointly generated，理论上 $\|e(K_j)\| = \mathcal{O}(1)$。但实际 21 个 keyframe 已经是 AR 性质（不同 keyframe 之间有 quality variance）。一个 unified architecture 把 keyframe + interpolation 合在一个 latent space 是作者指出的 future direction
- 整个 framework 假设 keyframe 间隔 $T$ 固定。Adaptive keyframe sampling（基于 motion magnitude / scene complexity 动态调整 $T$）应该能进一步改善 trade-off

**相关的潜在 baseline / 后续工作**（基于 reference list，给些猜测方向）：
- [LIVE](https://arxiv.org/abs/2602.03747): consistency-style training bound long-term deviation，跟 DCARL 的 keyframe anchoring 是不同思路的 drift-bounding
- [Epona](https://arxiv.org/abs/2506.15570): 把 long-horizon synthesis 显式 formulates 成 AR diffusion
- [LoL](https://arxiv.org/abs/2601.16914): scaling 到小时级，用 attention sink 类方法
- [Causal Forcing](https://arxiv.org/abs/2602.02214): AR diffusion distillation for real-time
- [Rolling Forcing](https://arxiv.org/abs/2509.25161): real-time AR long video

---

## References

- Project page: https://junyiouy.github.io/projects/dcarl/
- Wan2.1: https://github.com/Wan-Video/Wan2.1
- $\pi_3$: https://arxiv.org/abs/2507.13347
- ReCamMaster: https://arxiv.org/abs/2506.15570
- Diffusion Forcing: https://boyuan.space/diffusion-forcing/
- Self Forcing: https://self-forcing.github.io/
- Deep Forcing: https://arxiv.org/abs/2512.05081
- SEVA: https://stablevirtualcamera.github.io/
- VISTA: https://github.com/OpenDriveLab/VISTA
- OpenDV: https://github.com/OpenDriveLab/OpenDV
- nuScenes: https://www.nuscenes.org
- Qwen2.5-Omni: https://arxiv.org/abs/2503.20215
- DL3DV-10K: https://github.com/DL3DV-10K/DL3DV
- Brownian Bridge (背景知识): https://en.wikipedia.org/wiki/Brownian_bridge
- Euler-Lagrange equation: https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation
- Lipschitz continuity: https://en.wikipedia.org/wiki/Lipschitz_continuity
