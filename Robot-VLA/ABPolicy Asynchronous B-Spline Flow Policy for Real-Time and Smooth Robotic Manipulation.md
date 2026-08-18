---
source_pdf: ABPolicy Asynchronous B-Spline Flow Policy for Real-Time and Smooth Robotic
  Manipulation.pdf
paper_sha256: 1c83a9ec205bc8ee0769ca66fa9f1090ca0e3449dd70ab6a8936842cec0002df
processed_at: '2026-08-17T23:43:28-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 ABPolicy

好，剥掉学术黑话，咱就当聊天讲。

## 问题长什么样

你训 imitation learning policy，给它相机画面，它输出 robot 下一步怎么动。现在主流做法（Diffusion Policy、ACT、π0 这些）不是一步一步输出，而是一次吐一整段动作，比如未来 32 步，这叫 action chunk。这么做是因为一段动作整体建模比一步一步预测更稳，能学到 multimodal 分布。

但实操起来 chunk 有三个毛病。

**第一个毛病，chunk 内部发抖。** 模型从 noise 一路 denoise 出来 32 个 action，每一步都是采样采出来的。相邻两步之间没有"必须平滑过渡"的约束，可能第 5 步 joint 角度往左 10 度，第 6 步往右 12 度，robot 走起来就 zig-zag。你看 Fig. 5 那张图，raw action 的 acceleration 在 chunk 内部到处都是毛刺。

**第二个毛病，chunk 接缝处会跳一下。** 第一个 chunk 是模型这一刻采的样，第二个 chunk 是下一刻独立采的样。它们俩互不知道对方。所以 chunk 1 的最后一步和 chunk 2 的第一步之间 velocity 可能直接跳变，robot 在接缝处猛地一抖。这一抖还连锁反应——robot 抖一下，下一步的 camera observation 就偏离 training distribution 了，policy 越走越偏。

**第三个毛病，stop-and-go。** 模型算下一 chunk 要时间（diffusion 模型几十步 denoise，几百毫秒）。算的时候 robot 在干嘛？干等着。环境不等你——譬如桌上有个旋转平台在转，等你算完物体已经转了 30 度，你算出来的 action 是针对 30 度之前的物体，全废了。Dynamic task 上这问题最致命。

三个毛病的根：action 表示和 inference 调度没分开想。如果 action 表示本身就平滑，chunk 之间天然连续，inference 能异步而不破坏连续性，三个毛病一起没了。这就是这篇 paper 的核心 insight。

## 思路：换一种 action 表示

raw action 是什么？是 $T$ 个离散点 $[a_0, a_1, \ldots, a_{T-1}]$，每点都是高维（7 维：6 个 joint + 1 个 gripper）。模型要在这个 $T \times 7$ 的高维空间里做生成。

ABPolicy 说：咱们别这么干。把这 $T$ 个点拟合成一条平滑曲线（B-spline），曲线由 $N$ 个 control point 决定（$N=8$，比 $T=40$ 少很多）。模型只预测这 8 个 control point，曲线本身保证平滑。

好处是一连串的：

第一，曲线天然平滑。cubic B-spline 保证 $C^2$ 连续，意思是 position、velocity、acceleration 三阶都连续。chunk 内部 jitter 这个毛病，在 representation 层面就消灭了，模型根本没机会采出 zig-zag 的样。

第二，维度暴降。从 40×7=280 维降到 8×7=56 维。Flow matching 在这种低维流形上 10 步 denoise 就够，raw action 空间可能要 50 步以上。

第三，local support。这条是后面 CCR 的命根子。B-spline 的 basis function 只在局部非零，改一个 control point 只影响曲线一小段。这意味着如果你想 patch 曲线开头对接上一段结尾，只动前面几个 control point 就行，后面模型预测的部分保持原样。

## 怎么生成 control point

用 flow matching。简单说就是训一个网络，输入 observation，输出从 noise 到 data 的 vector field。推理时从 noise 出发，按 vector field 走 10 步，落到 data 上，就是预测的 control point。

为什么不用 regression 直接预测 control point？因为 manipulation 的 action 分布经常是 multimodal 的——同一个 task 可能从左边绕，也可能从右边绕。regression 会把两个 mode 平均，得到中间穿过去那条（撞物体的那条）。Flow matching 采样能正确选一个 mode。

训练 loss 长这样：

$$\mathcal{L} = \mathbb{E}\left[ \| \pi_\theta(C_t^\tau | o_t) - (C_t^* - z) \|^2 \right]$$

$z$ 是 noise，$C_t^*$ 是 ground truth control point，$\tau \in [0,1]$ 是从 noise 到 data 路径上的位置，$C_t^\tau = (1-\tau)z + \tau C_t^*$ 是这个路径上的点。网络要预测的是从当前点指向 data 的方向 $(C_t^* - z)$。这就是 flow matching 的标准形式，path 是直线，target 是常量，比 diffusion 的 noise schedule 简单很多。

## BiAP：为什么要预测过去

paper 有个细节，target chunk 不只是 future，还包含 past：

$$A_t = [a_{t-P}, \ldots, a_{t-1}, a_t, a_{t+1}, \ldots, a_{t+H-1}]$$

$P=8$ 是过去，$H=32$ 是未来。整段一起拟合成 B-spline，control point 作为 regression target。

为什么要预测过去？robot 又不执行过去。但预测过去等于把 history 作为 condition 喂进 policy。考虑这个场景：training data 里在 observation $o_t$ 处有两条轨迹，一条是从左边 approach 过来的，一条是从右边。只看 $o_t$ 分不出来，policy 会把两种 mode 混。加 history 进来，policy 看到"我刚才是从左边过来的"，就知道 future 应该接着从左边走。分布更 unimodal，更容易学。

Ablation 数据（Table IV）：不加 BiAP，success 60%，boundary jitter 0.022；加了 BiAP，success 85%，boundary jitter 0.0097。差距 25 个百分点，非常大。

## CCR：接缝怎么补

这是 paper 最巧妙的地方。

异步 inference 的设定：robot 一边执行 chunk $i$ 的动作，一边后台跑模型算 chunk $i+1$。Inference 大概 90ms。等算完，robot 已经执行了 chunk $i$ 最后 2-3 步。

问题：新算出来的 trajectory 是基于算之前的 observation $o_t$ 算的，它的起点对应 $o_t$ 时刻。但现在 robot 已经走到 $o_{t+P}$ 时刻了。如果直接把新 trajectory 接到 robot 当前位置，新 trajectory 起点和刚才执行的几步 action 不连续，接缝处会跳。

CCR 的做法：用 B-spline 的 local support，只调整新 trajectory 前 $N_{\text{free}}$ 个 control point，让曲线开头拟合刚才执行的几步 action，后面 control point 保持模型预测的原样不变。

数学上是个标准 least-squares：

$$\min_{\{c_i\}_{i=0}^{N_{\text{free}}-1}} \sum_{t=0}^{P-1} \left( a_t^{\text{exec}} - \hat{s}_{\text{new}}(u_t) \right)^2$$

$\hat{s}_{\text{new}}$ 是部分更新的曲线，free + fixed 两段拼起来。目标是最小化曲线开头和已执行 action 的误差。

关键：这玩意是线性的，closed-form 解 $c = (A^T A)^{-1} A^T b$，微秒级。没有 gradient descent，没有超参，不扰动 model 学到的 dynamics。

对比一下 RTC 那条路：RTC 在 raw action 空间用 inpainting + gradient guidance + weighted fusion 补接缝。Gradient-based 方法有 step size、guidance scale 这些 hyperparam，还会扰动 diffusion 的 denoise 过程，把 model 预测弄坏。ABPolicy 的 CCR 在 control point 空间用 closed-form 解决，干净得多。

为什么 DCT 系数做不了 CCR 这一步？DCT 是 global basis，改一个 coefficient 整条曲线都动。要 patch 曲线开头，没办法只动局部。这就是 paper 选 B-spline 不选 DCT 的根本理由——DCT 也能保证 intra-chunk 平滑，但 CCR 这步做不了，inter-chunk discontinuity 解决不掉。

## Async：隐藏 latency

两 thread 并行：inference thread 跑模型算下一 chunk；control thread 30Hz 从 action queue 取 action 发给 robot。模型算的时候 robot 还在执行上一 chunk 的动作，没停。算完 apply CCR 接上去，立即触发下一次 inference。

Dynamic task 上效果特别明显（Table I）：旋转平台 stack block，sync 30% success，async 55%，差 25 个点。因为 sync 模式 robot 等 inference 的时候旋转平台物体在动，observation 过期，policy 抓瞎。Async 不等，抓到的是最新 observation，反应快。

Static task 上 async 主要省时间（14.2%），success rate 差不多，因为 static 不需要 responsive。

## 把整个 loop 串起来

推理时刻 t 的 pipeline：

1. Control thread 拿到 observation $o_t$，传给 inference thread
2. Inference thread 用 flow matching 从 noise denoise 10 步，得到 control point $\{c_{\text{pred}, i}\}$
3. 与此同时 control thread 还在执行上一 chunk 末尾的 action，这些 action 是 $\{a_t^{\text{exec}}\}_{t=0}^{P-1}$
4. Inference 完，跑 CCR：以 $\{a_t^{\text{exec}}\}$ 为 anchor，refit 前 $N_{\text{free}}$ 个 control point
5. 把 refit 后的 control point 转成 continuous trajectory，push 进 action queue
6. Control thread 接着执行新 trajectory，同时 inference thread 拿下一 observation 算下一 chunk

每一步都 essential：B-spline 保证 intra 平滑；BiAP 让 prediction 起点更稳；CCR 修 inter 接缝；async 隐藏 latency。少一个都不行。

## 你大概会关心的几个点

**representation 比 architecture 重要。** 这篇 paper 的 thesis 是这个。换 action 表示从 raw 换到 B-spline control point，三个毛病一起解决，解决方式还都是 closed-form。这跟你一直强调的 representation 决定 difficulty 思路一致。自动驾驶 trajectory prediction 里也是，直接预测 $(x_t, y_t)$ 序列 vs 预测 polynomial coefficient，难度差很多。

**Flow matching 在低维空间的优势。** Control point 空间 56 维比 raw action 280 维低 5 倍。10 步 denoise 够。这呼应 consistency model、mean flow、shortcut model 那条线——降 sampling cost 的另一个路径是改 representation。

**Async = model as a service。** Policy 当 async service，control loop 当 client，不阻塞。这个 pattern LLM serving 里到处都是（continuous batching、speculative decoding）。Robot policy 这边刚起步，SmolVLA / RTC / ABPolicy 都往这方向走。

## 一个我自己想追问的点

CCR 是 deterministic least-squares，但 flow matching 是 multimodal 采样。如果 policy 采样出 mode A，executed action 偏 mode B 路径，CCR 会强行把 mode A 头部拉到 mode B，可能产生不自然 trajectory。Async + multimodal 之间有潜在矛盾，paper 没讨论这块。Delay 短的时候没事（90ms，executed action 还没偏太多），delay 长了（cloud inference 几百 ms）可能要重新想。

---

Reference:
- Project: https://teee000.github.io/ABPolicy/
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ACT: https://arxiv.org/abs/2304.13705
- Flow Matching: https://arxiv.org/abs/2210.02747
- π0: https://arxiv.org/abs/2410.24164
- BEAST: https://arxiv.org/abs/2506.06072
- RTC: https://arxiv.org/abs/2506.07339
- SmolVLA: https://arxiv.org/abs/2506.01844
- PTP: https://arxiv.org/abs/2505.09561
- FAST: https://arxiv.org/abs/2501.09747
- DiT: https://arxiv.org/abs/2210.02794
- DINO-V2: https://arxiv.org/abs/2304.07193
- Mean Flows: https://arxiv.org/abs/2505.13447
- Shortcut Models: https://arxiv.org/abs/2410.12557

---

# ABPolicy 深度解读

Andrej，这篇 paper 处理的是 imitation learning 中一个相当根本的问题：当你用 action chunking + diffusion/flow policy 时，robot 实际执行的时候会出现三类病。我先把 intuition 讲清楚，再深入到每块的数学。

Paper link: https://teee000.github.io/ABPolicy/
arXiv: https://arxiv.org/abs/2506.06072 (BEAST), https://arxiv.org/abs/2506.07339 (RTC), https://arxiv.org/abs/2410.24164 (π0)

---

## 1. 问题本身的三层病

这块 paper 抓住了三个互相纠缠的现象：

**Intra-chunk jitter**：每个 action chunk 由 diffusion/flow 模型独立 denoise 出来，本质是从一个 multimodal distribution 采一个样本。哪怕在 chunk 内部，相邻 timesteps 的 action 也可能因为高频采样出现 zig-zag。原因是网络预测的是 raw action $a_t$，每一步都是独立采样出来的，没有"轨迹必须平滑"这个 inductive bias。

**Inter-chunk discontinuity**：chunk 1 的最后一个 action 和 chunk 2 的第一个 action 没有共享信息。即使每个 chunk 内部 smooth，在 boundary 处 velocity/acceleration 会突变，因为它们是两次独立的 denoise 结果。这会导致 robot 在 boundary 抖一下，而抖一下又会让下一个 observation 偏离 training distribution → distribution shift → 后续越走越偏。

**Stop-and-go execution**：sync inference 模式下 robot 等模型算完再走。dynamic environment（譬如旋转平台）在等的时候物体继续动，observation 过期，policy 失效。Latency 越长这个问题越严重，尤其是 VLA 这类大模型。

这三层病其实是同一个根因的不同表现：**action 表征和 inference 调度没有 decouple**。如果 action 表征本身 smooth，chunk 之间天然连续，inference 可以异步而不破坏连续性，三个问题就一起解决了。这就是 ABPolicy 的设计思路。

---

## 2. B-Spline Parameterization —— 为什么这是关键直觉

### 2.1 数学定义

给定离散 action sequence $\{a_t\}_{t=0}^{T-1}$，要找 $N$ 个 control points $\{c_i\}_{i=0}^{N-1}$，使得 B-spline curve $s(t)$：

$$s(t) = \sum_{i=0}^{N-1} c_i \, N_{i,p}(t) \tag{1}$$

最优拟合通过 least-squares：

$$\{c_i^*\}_{i=0}^{N-1} = \arg\min_{\{c_i\}} \sum_{t=0}^{T-1} \left(a_t - s(t)\right)^2 \tag{2}$$

变量含义：
- $a_t$：第 $t$ 步的原始 action（某维度的 joint angle 或 gripper aperture）
- $c_i$：第 $i$ 个 control point（待优化的低维参数）
- $N_{i,p}(t)$：degree $p$ 的第 $i$ 个 B-spline basis function，由 Cox-de Boor recursion 递归定义，依赖于 knot vector
- $p=3$：cubic，保证 $C^2$ continuity（position、velocity、acceleration 都连续）

### 2.2 为什么 cubic B-spline 是直觉上对的选择

B-spline 的几个性质恰好像是为这个问题定制的：

**Local support**：$N_{i,p}(t)$ 只在 $[u_i, u_{i+p+1}]$ 区间非零。这意味着改一个 control point $c_i$ 只影响曲线局部一段。这个性质后面 CCR 要用——可以只 refit 前 $N_{\text{free}}$ 个 control points 而不破坏后面 policy 预测的部分。

**$C^2$ continuity**：cubic B-spline 在 knot 处自动满足 position、velocity、acceleration 连续。所以 intra-chunk jitter 这个问题在表征层面就被消灭了——你采样出的不是一堆离散 action，而是连续曲线上的 control points，曲线本身平滑。

**Compact representation**：$T=40$ 步 action 用 $N=8$ 个 control points 编码。压缩比 5×。这意味着 flow matching 模型不用在 40 维空间生成，只要在 8 维空间生成。学习更稳定，推理更快，模型更容易训练。

**Convex hull property**：B-spline curve 必定落在 control points 的 convex hull 内。这是 stable generation 的隐含保证——control points 不奇怪，curve 就不奇怪。

### 2.3 与其他 action representation 的对比（Table II）

| Method | Mean Error ↓ | SNR (dB) ↑ |
|---|---|---|
| 256 Discrete Bins (RT-1, OpenVLA) | 0.0020 | 41.9 |
| DCT Low-Frequency Coeffs (HumanMAC, FAST) | 0.0010 | 44.6 |
| B-spline (Discrete) (BEAST) | 0.0025 | 37.7 |
| **B-spline (Continuous) (Ours)** | **0.00031** | **50.7** |

这里有两个 intuition：

**为什么 DCT 不如 continuous B-spline**：DCT 是 global basis，每个 coefficient 影响整条曲线。要 local 调整 boundary 的时候，改一个 DCT coefficient 整条曲线都会跟着动，无法做 CCR 那种"只改前几个 control points"的 local surgery。B-spline 是 local basis，恰好相反。

**为什么 BEAST 的 discrete B-spline 反而最差**：BEAST 把 control points 量化成 256 bins 再 tokenize 给 autoregressive model。量化误差 0.0025 比 continuous 直接拟合 0.00031 高一个数量级。量化把 B-spline 的精度优势全部抵消。这告诉我们：B-spline 一定要在 continuous domain 上做生成，离散化会丢掉它的核心价值。

---

## 3. Flow Matching on Control Points

### 3.1 Formulation

Policy 是条件生成模型 $\pi_\theta(C_t^* | o_t)$，$o_t$ 是 observation，$C_t^* = [c_0^*, c_1^*, ..., c_{N-1}^*]$ 是 ground-truth control points。

训练 loss 是 flow matching 的标准形式：

$$\mathcal{L}_{\text{FM}}^\tau(\theta) = \mathbb{E}_{\tau, o_t, z, C_t^*} \left[ \| \pi_\theta(C_t^\tau | o_t) - (C_t^* - z) \|^2 \right] \tag{4}$$

变量含义：
- $z \sim \mathcal{N}(0, \mathbf{I})$：prior 样本，和 $C_t^*$ 同维（$N \times \dim(a)$）
- $\tau \sim U[0,1]$：flow time，沿路径的位置
- $C_t^\tau = (1-\tau) z + \tau C_t^*$：从 noise 到 data 的线性插值路径上一点
- $\pi_\theta(C_t^\tau | o_t)$：网络输出，预测从当前点到 data 的 vector field
- $(C_t^* - z)$：target vector，从当前 noise 指向 data

### 3.2 Flow Matching vs Diffusion 的直觉

这里 flow matching 比 diffusion 更合适，原因有几层：

**Linear path**：$C_t^\tau = (1-\tau)z + \tau C_t^*$ 是直线路径，不像 DDPM 那种需要 noise schedule。vector field target $(C_t^* - z)$ 是常量，训练 signal 非常稳定。

**Few-step inference**：paper 用 10 步 denoising。在 8 维 control point 空间上 10 步足够，因为目标分布是低维 smooth manifold。要在 40 维 raw action 空间达到同质量，需要更多步。这是 B-spline 压缩的额外红利。

**Multimodality**：flow matching 可以建模 multimodal distribution。Robot manipulation 的 action distribution 经常是 multimodal——同一个 task 可以从左边绕过去也可以从右边绕过去。regression 模型会把这两种 mode 平均掉，得到一个都不靠的中间值。generative 模型采样才能正确处理 multimodality。

### 3.3 为什么不直接 regression

如果只做 MSE regression $C_t^*$，loss 是 $\|f_\theta(o_t) - C_t^*\|^2$。这种情况下模型会预测所有 mode 的均值。对 bimodal action（左/右两条轨迹）会预测中间穿过去的轨迹，这恰好是最危险的（撞到物体）。

Flow matching 通过采样，每次选一个 mode，符合 imitation learning 的需求。

---

## 4. Bidirectional Action Prediction (BiAP)

### 4.1 Formulation

每个 timestep $t$ 的 target chunk 跨过去和未来：

$$A_t = [a_{t-P}, ..., a_{t-1}, a_t, a_{t+1}, ..., a_{t+H-1}] \tag{3}$$

paper 用 $P=8$（历史）、$H=32$（未来）。整段 $A_t$ 拟合成 B-spline 得到 $C_t^*$，flow matching 在 $C_t^*$ 上训练。

### 4.2 直觉

BiAP 看起来奇怪——为什么预测过去？robot 又不执行过去。但它的作用是 **temporal context regularization**。

考虑一个 chunk $[a_t, ..., a_{t+H-1}]$ 单独训练。模型看到 $o_t$，要预测未来。它对"过去发生了什么"没概念。如果 training data 里有两条轨迹在 $o_t$ 处 observation 看起来差不多，但 history 不同（一条是 approaching from left，一条 from right），那 future 也应该不同。只看 $o_t$ 的 policy 会把这两种 mode 混在一起。

加 history $[a_{t-P}, ..., a_{t-1}]$ 进来，等于把"我刚才在做什么"作为 condition。flow matching 在 $C_t^*$ 上建模 $p(C_t^* | o_t, a_{t-P:t-1})$，分布更 unimodal，更容易学。

### 4.3 和 PTP 的对比

PTP (Past-Token Prediction, https://arxiv.org/abs/2505.09561) 也用 past tokens，但目的是 test-time 选 consistency 最好的 trajectory，要重复 sample 多次。BiAP 把 past 直接焊进 target，一次 inference 就够。这是 efficiency 上的本质区别。

### 4.4 Ablation 数据

Table IV 在 static block stacking 上：

| | Success Rate ↑ | Initial Jitter ↓ | Refitted Jitter ↓ |
|---|---|---|---|
| w/o BiAP | 60% | 0.0220 | 0.0180 |
| w/ BiAP | 85% | 0.0170 | 0.0097 |

Success rate 从 60% 到 85% 是 25 个百分点。Initial jitter 降 23%，refitted jitter 降 46%。这说明 BiAP 不仅自己减少 jitter，还给 CCR 留下更干净的初始状态，让 CCR 更有效。两步是协同的。

---

## 5. Continuity-Constrained Refitting (CCR) —— 这块是 paper 的精髓

### 5.1 异步 inference 的根本矛盾

Asynchronous inference 的设定：robot 在执行 chunk $i$ 的同时，后台 thread 算 chunk $i+1$。Inference 大约 90ms（paper 用 RTX 4070 Ti Super，10 step flow matching）。

矛盾在哪：开始 inference 时 observation 是 $o_t$，等 inference 完，robot 已经执行了 chunk $i$ 的最后 $P$ 步 actions $\{a_t^{\text{exec}}\}_{t=0}^{P-1}$。新预测出来的 trajectory 是基于 $o_t$ 算的，它的"起点"对应 $o_t$ 时刻，但现在 robot 已经走到 $o_{t+P}$ 时刻了。如果直接接上去，新 trajectory 的初始 actions 和刚才执行的 actions 之间不连续。

### 5.2 数学 formulation

Policy 预测出 control points $\{c_{\text{pred},i}\}_{i=0}^{N-1}$。要找新的 $\{c_{\text{new},i}\}_{i=0}^{N-1}$，使得新 trajectory 起点拟合已执行 actions，同时尽量保留 policy 预测的形状。

利用 B-spline local support：只优化前 $N_{\text{free}}$ 个 control points，后面的 $\{c_{\text{pred},i}\}_{i=N_{\text{free}}}^{N-1}$ 不动。

$$\{c_{\text{new},i}\}_{i=0}^{N_{\text{free}}-1} = \arg\min_{\{c_i\}_{i=0}^{N_{\text{free}}-1}} \sum_{t=0}^{P-1} \left( a_t^{\text{exec}} - \hat{s}_{\text{new}}(u_t) \right)^2 \tag{5}$$

其中：

$$\hat{s}_{\text{new}}(u) = \sum_{i=0}^{N_{\text{free}}-1} c_i \, N_{i,p}(u) + \sum_{i=N_{\text{free}}}^{N-1} c_{\text{pred},i} \, N_{i,p}(u) \tag{6}$$

变量含义：
- $u_t$：把 timestep $t$ 映射到 B-spline parameter domain
- $u$：B-spline 的 parameter（通常 $[0,1]$ 或 knot span）
- $N_{\text{free}}$：可优化的 control points 数量，决定 local adjustment 范围
- $\hat{s}_{\text{new}}(u)$：部分更新的 trajectory，由 free + fixed 两部分组成

### 5.3 为什么这是个 linear least-squares

Eq.(6) 对 $c_i$ 是线性的（$c_i$ 只乘以已知的 $N_{i,p}(u)$）。Eq.(5) 的 objective 是二次的。所以这是个标准 linear least-squares：

$$\min_c \| A c - b \|^2$$

其中 $A$ 是 design matrix，元素是 $N_{i,p}(u_t)$；$b$ 是 $a_t^{\text{exec}} - \sum_{i=N_{\text{free}}}^{N-1} c_{\text{pred},i} N_{i,p}(u_t)$，把 fixed 部分移到右边。Closed-form 解 $c = (A^T A)^{-1} A^T b$，几个 microsecond 就能算完。

这是关键：**CCR 不需要 gradient descent，没有 sensitive hyperparameter，不扰动 learned dynamics**。这正好是 RTC 的问题——RTC 用 inpainting + gradient guidance + weighted fusion，gradient-based 方法有 step size、guidance scale 等超参，还要扰动 diffusion 的去噪过程，容易把 model 的 prediction 弄坏。

### 5.4 直觉：B-spline local support 是 CCR 的 enabling property

如果用 DCT 系数做 action representation，要 refit 起点和已执行 actions 对齐，得改所有 DCT coefficients（global basis），整条曲线都会动，policy 的 prediction 就被废了。

B-spline 的 local support 让 CCR 变成"只 patch 头部，body 不动"。这是 paper 选择 B-spline 而非 DCT 的根本理由——DCT 也能保证 intra-chunk smooth，但 CCR 这步做不了。

### 5.5 实验：boundary smoothing 对比

Fig. 6 比较 raw actions / weighted fusion (SmolVLA, RTC 风格) / B-spline refitting 在 chunk boundary 的 acceleration magnitude。Raw actions 在 boundary 处有巨大 spike；weighted fusion 把 spike 摊平但还有 residual；B-spline refitting 几乎 flat。

Table III 给 quantitative 数据，average over 6 joints：

| | Avg ZCR of Velocity ↓ | Acc p95 ↓ |
|---|---|---|
| B-spline | 0.1301 | 3.06 |
| Raw | 0.1837 | 7.13 |

- ZCR (Zero-Crossing Rate) of velocity：velocity 改变符号的频率，越低越 smooth。降 29.2%。
- Acc p95：acceleration 95 分位。降 57.1%。

57% 的 acceleration p95 降低是非常显著的。Acc p95 直接对应 robot 的 jerk 和 motor wear，对 real deployment 很关键。

---

## 6. Asynchronous Inference 调度

### 6.1 设计

两个并行 threads：
- **Inference thread**：观测 → flow matching → 得到 control points → CCR → push 到 action queue
- **Control thread**：30Hz 从 action queue 取 action 发给 robot

Inference delay ~90ms 时，robot 在 90ms 内执行了大约 2-3 步 action。这些就是 CCR 要 anchor 的 $\{a_t^{\text{exec}}\}$。

### 6.2 异步 vs 同步的实验对比

Table I：dynamic tasks（旋转平台）：

| Method | Stack Block | Push Block | Hang Cup |
|---|---|---|---|
| DP (sync) | 40 | 75 | 35 |
| Sync | 30 | 75 | 40 |
| **Async** | **55** | **85** | **60** |

Async 比 sync 在 dynamic tasks 平均 +18.3% success rate。这个 gap 在 dynamic 任务上特别大，因为 sync inference 的 stop-and-go 让 observation 过期，旋转平台上的物体在等 inference 期间已经转了 30 度。

Static tasks：async 平均减 14.2% completion time，效率提升但 success rate 基本持平（因为 static 不需要 responsive）。

### 6.3 和 SmolVLA / RTC 的对比

SmolVLA (https://arxiv.org/abs/2506.01844) 也是 async，但把 inference 放 cloud GPU，有 data transmission latency。ABPolicy 是 local GPU，没 transmission overhead。

RTC (https://arxiv.org/abs/2506.07339) 也是 async + 修复 continuity，但用 inpainting + gradient guidance + weighted fusion。Inpainting 是在 raw action space 用 gradient 引导 denoise 过程满足 boundary constraint，这会扰动 model 的 learned distribution；weighted fusion 是把新旧 chunk 在 overlap 区做加权平均，averaged action 不保证 valid。ABPolicy 用 closed-form least-squares 在 control point 空间解决，orthogonal 且更干净。

---

## 7. 整体架构

Policy network 是 DiT (Diffusion Transformer, https://arxiv.org/abs/2210.02794)：
- Observation encoder：DINO-V2 (frozen, https://arxiv.org/abs/2304.07193) 处理当前 RGB；MLP 处理过去 8 帧的 robot state
- DiT blocks 通过 cross-attention 把 observation 注入 denoising 过程
- 10 step flow matching sampling

Action space：6-DoF AgileX Piper manipulator + parallel gripper。Joint angles + continuous gripper aperture。30Hz control。每个 dimension 独立拟合 cubic B-spline，open-uniform (clamped) knot vector。

Clamped knot vector 的作用：让 curve 穿过第一个和最后一个 control point，boundary 处可以精确控制起点终点位置。

---

## 8. 对你（Karpathy）可能感兴趣的几个 angle

### 8.1 Action representation 作为 inductive bias

整个 paper 的核心 thesis 是：**action representation 比 model architecture 更重要**。把 action 从 raw space 换到 B-spline control point space，三个问题（jitter / discontinuity / latency）一起解决，且解决方式是 closed-form 而非 gradient-based。这呼应你在 nanoGPT / Tesla 自动驾驶一直强调的：representation 决定 difficulty。

类比：在自动驾驶 trajectory prediction 里，直接预测 $(x_t, y_t)$ 序列 vs 预测 polynomial coefficients 或 spline control points，也是这个 trade-off。后者压缩 + smoothness inductive bias 让 model 学得更快、inference 更稳。

### 8.2 Flow matching 在低维空间的优势

Control point 维度 8×7（8 control points × 7 action dims）≈ 56 维。比 raw action chunk 40×7=280 维低 5×。Flow matching 在 56 维上 10 步就够，280 维可能需要 50 步以上。这是 representation 改变 sampling efficiency 的直接体现。

也呼应 consistency models / mean flows (https://arxiv.org/abs/2505.13447) / shortcut models (https://arxiv.org/abs/2410.12557) 这条线——降低 sampling cost 的另一个路径是改 representation。

### 8.3 Asynchronous inference 和 "model as a service"

把 policy 想成一个 async service，robot control loop 是 client。Inference 不阻塞 control。这个 pattern 在 LLM serving 里也常见——continuous batching、speculative decoding 都是把 model inference 和 downstream consumption decouple。Robot policy 这边刚意识到这点不久，SmolVLA / RTC / ABPolicy 都在这方向。

### 8.4 Bidirectional prediction 和 LLM 的双向 context

BiAP 在 spirit 上类似 LLM 的 bidirectional context（譬如 BERT vs GPT）。LLM 里 causal 预测 future token，但 LM objective 训出来的 representation 不一定 capture 全局 context。这里也是——只预测 future 的 policy 看不到过去，加 future+past 让 policy 隐式学到 action 的 temporal structure。

不过 paper 没明确说是不是用 masking（像 BERT）还是 autoregressive over the whole chunk（像 decoder）。从 DiT 架构 + flow matching 看，应该是 non-autoregressive，整段 $C_t^*$ 一次性 denoise 出来。所以 BiAP 的"bidirectional"是 target 层面的 bidirectional，不是 inference 层面的。

---

## 9. 几个可以追问的点

**$N_{\text{free}}$ 怎么选？** Paper 没明确给数值。Intuition：$N_{\text{free}}$ 要足够大以 fit $\{a_t^{\text{exec}}\}_{t=0}^{P-1}$（8 个 actions），但又不能太大否则破坏 policy prediction。$P=8$, $N=8$（推测），可能 $N_{\text{free}}=4$ 左右。

**CCR 是不是 marginalize 掉 policy 的预测？** 如果 inference delay 很长，executed actions 跨越了 $N_{\text{free}}$ 个 control points 对应的 spline 段，那 fixed 部分也需要 anchor，CCR 就失效。Paper 的 90ms delay 比较短，所以没事。Delay 长到几百 ms 时（譬如 cloud inference）需要重新设计。

**Multimodality 在 CCR 后还保留吗？** Flow matching 在 control point 空间建模 multimodal distribution，但 CCR 是 deterministic least-squares。如果 policy 采样出 mode A，但 executed actions 偏向 mode B 路径，CCR 会强行把 mode A 的 head 拉到 mode B 路径，可能产生不自然的 trajectory。这是 async + multimodal 的一个潜在矛盾，paper 没讨论。

**为什么不用 quintic B-spline（$p=5$）保证 $C^4$ continuity？** Cubic 已经够 jerk limited（jerk 是三阶导，$C^2$ 保证 acceleration 连续，jerk 有界但不连续）。Quintic 保证 jerk 连续，但 control points 要更多，fitting 更难。Trade-off 不划算。

---

## 10. 总结：method 的 elegance

ABPolicy 的优雅之处在于三个机制互相 enabling：

1. **B-spline** 提供 local support → CCR 可行
2. **B-spline** 提供 $C^2$ continuity → intra-chunk jitter 消失
3. **Flow matching on control points** 提供 multimodality + 低维 sampling
4. **BiAP** 给 policy temporal context → 让 prediction 起点更稳 → CCR 起点更准
5. **CCR** closed-form refit → 不扰动 learned dynamics、无 hyperparameter
6. **Async inference** 隐藏 latency → dynamic task responsiveness
7. **CCR** 修复 async 引入的 discontinuity → async + smoothness 同时成立

任何一个机制单独拿出来都不够：B-spline 没 CCR 就 chunk 边界断；CCR 没 B-spline local support 就动整条曲线；async 没 CCR 就 stop-and-go 或 discontinuity；sync 没 async 就 dynamic task 失败。组合起来才是完整方案。

Reference links：
- Project: https://teee000.github.io/ABPolicy/
- Flow Matching: https://arxiv.org/abs/2210.02747
- ACT: https://arxiv.org/abs/2304.13705
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- BEAST: https://arxiv.org/abs/2506.06072
- RTC: https://arxiv.org/abs/2506.07339
- π0: https://arxiv.org/abs/2410.24164
- DiT: https://arxiv.org/abs/2210.02794 (Peebles & Xie, scalable diffusion with transformers)
- DINO-V2: https://arxiv.org/abs/2304.07193
- OpenVLA: https://arxiv.org/abs/2406.09246
- PTP: https://arxiv.org/abs/2505.09561
- FAST: https://arxiv.org/abs/2501.09747
- SmolVLA: https://arxiv.org/abs/2506.01844
- Mean Flows: https://arxiv.org/abs/2505.13447
- Shortcut Models: https://arxiv.org/abs/2410.12557
