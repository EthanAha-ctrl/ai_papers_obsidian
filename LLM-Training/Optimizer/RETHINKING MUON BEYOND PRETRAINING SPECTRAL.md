---
source_pdf: RETHINKING MUON BEYOND PRETRAINING SPECTRAL.pdf
paper_sha256: 2016d07cca61aaa9be936f5a5a701ff033a89e6022f03a9795f639c4d1a2428f
processed_at: '2026-08-11T23:17:50-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Pion 用人话讲

## 一句话版本

Muon 把 momentum 的所有 singular value 都拉成 1，pretraining 时这是 feature，post-training 时这是 bug；Pion 干了同一件事，但只拉大 singular value，把小的压到 0——一个 spectral high-pass filter，per-step cost 一模一样。

---

## Muon 到底在干嘛

要理解 Pion，先得真的理解 Muon 在干嘛，不是表面上的"用 NS 代替 SVD"。

SGD 在 Frobenius norm 下是 steepest descent——你沿着 gradient 本身走。但 spectral norm 下的 steepest descent 长得不一样：你需要把所有 singular direction 等权化。为什么？因为 spectral norm 只 care 最大的那个 direction，你想最小化 $\|\mathbf{G}\mathbf{W}\|_2$ 这种东西，等比例 scale 所有方向就是最自然的解。

具体说，momentum $\mathbf{M} = \mathbf{U}\Sigma\mathbf{V}^\top$，Muon 把 $\Sigma$ 换成 $\mathbf{I}$——所有 singular value 变 1，singular vectors 不动。Update 变成 $\Theta_t = \Theta_{t-1} - \eta \mathbf{U}\mathbf{V}^\top$。

这是一个**平等主义者**：每个 singular direction 不论大小，都被赋予同样的 magnitude 1。

在 LLM pretraining 里，gradient 是 high-rank 的——每一个 singular direction 都携带着真实的学习信号，因为你在 fit 整个 internet 的数据分布。把小的 singular value 也撑到 1 是一种**exploration**——你不想让 update 只 dominate 在几个 principal direction 上，你想要 spectral 上的 diversity。

这就是 Muon 的核心 inductive bias：**gradient 的所有方向都同等重要**。

Reference: [Muon blog](https://kellerjordan.github.io/posts/muon/), [Shampoo](https://arxiv.org/abs/1802.09568), [SOAP](https://arxiv.org/abs/2409.11321)

---

## 为什么 pretraining 是 feature，post-training 是 bug

### VLA：action head 的 gradient 是低秩的

VLA 模型有三个 module：vision encoder、language backbone、action head。Action head 输出的是 7-dim vector（end-effector translation + rotation + gripper binary），而 vision 是 pixel 级统计、language 是 high-dim embedding。

作者用 effective rank 测量：

$$\mathrm{erank}(\mathbf{G}) = \exp\left(-\sum_i p_i \log p_i\right), \quad p_i = \frac{\sigma_i(\mathbf{G})}{\sum_j \sigma_j(\mathbf{G})}$$

变量解释：
- $\mathbf{G}$：gradient matrix
- $\sigma_i(\mathbf{G})$：第 $i$ 个 singular value
- $p_i$：归一化后的"概率"
- 整个公式就是 Shannon entropy 的指数——perplexity 的概念

**Intuition**：如果 gradient 的能量都集中在一个方向上，erank = 1；如果均匀分散在 $n$ 个方向，erank = $n$。

实验结果（Fig. 1a）：vision 的 erank 远高于 language，language 远高于 action。Action module 的 gradient 几乎是 low-rank 的——少数 leading singular direction 携带真实 signal，剩下都是 spectral floor（数值噪声、量化噪声、batch 采样噪声）。

现在你想：Muon 看到 action gradient，把每个 singular value 都拉到 1。**包括那些 noise floor 上的方向**。这就把噪声放大到和真实 signal 同等量级，update 被 corrupt。

Fig. 1(b) 印证：在 action module 上，Muon 的 success rate 比 AdamW 还低（32.2% vs 97.0% on LIBERO Object at 4.5k steps）。这不是偶然——是结构性失败。

### RLVR：gradient 的 SNR 极低

RLVR（GRPO、GMPO）的 gradient 长得跟 SFT 完全不一样。SFT 是 token-level 监督——每个 token 都有 teacher signal，gradient 信号干净。RLVR 是 trajectory-level reward——一整条 trajectory 跑完才有一个 binary reward，然后这个 reward 要 back-propagate 到每个 token 的 gradient 上。

作者推导了 closed-form SNR：

SFT 的 SNR：
$$\mathrm{SNR}_{\mathrm{SFT}} = gT \frac{\|\bar{\mathbf{s}}\|^2}{\sigma_s^2}$$

变量：
- $g$：batch size
- $T$：sequence length
- $\bar{\mathbf{s}} := \mathbb{E}[\nabla_\Theta \ell_{i,t}]$：per-token score 期望（确定性 signal）
- $\sigma_s^2$：per-token score variance（噪声）

GRPO 的 SNR（on-policy 近似）：
$$\mathrm{SNR}_{\mathrm{GRPO}} \approx gT \frac{\kappa_g(p)\|\Delta\|^2}{\sigma_s^2}$$

变量：
- $p$：单 prompt 的 success probability
- $\kappa_g(p) \approx p(1-p)$：reward 信号的有效强度
- $\Delta := \mu_S^+ - \mu_S^-$：成功 trajectory 与失败 trajectory 的 expected score gap——**这才是 RLVR 的真正 signal**

加上 off-policy 的额外损失：
$$\frac{\mathrm{SNR}_{\mathrm{SFT}}}{\mathrm{SNR}_{\mathrm{GRPO}}^{\mathrm{full}}} \gtrsim \frac{\|\bar{\mathbf{s}}\|^2}{\kappa_g(p)\|\Delta\|^2} \cdot (1+\chi^2) \cdot \frac{1}{1-\alpha}$$

变量：
- $\chi^2$：per-token chi-squared divergence，importance sampling 引入的方差放大
- $\alpha$：clip fraction，PPO-style clipping 又损失一部分 signal

**Intuition**：RLVR 的 signal 是 $\Delta$（成功 vs 失败的差距），但大部分时候 $\|\Delta\| \ll \|\bar{\mathbf{s}}\|$（成功和失败的 trajectory 看起来很像），加上 $p(1-p)$ 这个因子在任务难或易时都趋 0——RLVR 的 gradient SNR 比 SFT 低一到两个数量级。

Muon 在这个低 SNR regime 下做什么？把所有方向都拉到 1，**包括噪声方向**。Noise 被放大到和 signal 同等量级，policy 迅速 collapse。Fig. 2(b) 实验：Muon 训 GRPO on Qwen3-1.7B，MATH500 accuracy 从初始 checkpoint 一路跌到 0。

这是 paper 最 striking 的 negative result：**Muon 在 RLVR 上完全不工作**。

Reference: [DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300), [DeepSeek-R1](https://arxiv.org/abs/2501.12948), [GMPO](https://arxiv.org/abs/2507.20673), [DAPO](https://arxiv.org/abs/2503.14476)

---

## Pion 的核心 trick

### 观察：两个 failure 共享一个 spectral signature

VLA 的 action gradient 是 low-rank（少数 leading singular value 携带 signal，tail 是 spectral floor）。
RLVR 的 gradient 是 low-SNR（少数 leading singular value 携带 $\Delta$ signal，tail 是 stochastic noise）。

**两种情况下，informative signal 都集中在 leading singular values，tail 是 noise**。Muon 的问题是把 tail 也撑到 1。

**Pion 的 fix**：设计一个 spectral filter，大 singular value 锚定在 1，小 singular value 推到 0。一个 high-pass。

### 数学上的 elegance

NS iteration 每步是 quintic matrix polynomial：
$$\mathcal{P}(\mathbf{X}; a, b, c) = a\mathbf{X} + b\mathbf{X}\mathbf{X}^\top\mathbf{X} + c\mathbf{X}(\mathbf{X}^\top\mathbf{X})^2$$

变量：
- $\mathbf{X} \in \mathbb{R}^{m \times n}$：输入 matrix
- $a, b, c$：三个标量系数
- 三项分别是 $\mathbf{X}$、$\mathbf{X}\mathbf{X}^\top\mathbf{X}$、$\mathbf{X}(\mathbf{X}^\top\mathbf{X})^2$——这是 odd function in $\mathbf{X}$

代入 SVD $\mathbf{X} = \mathbf{U}\Sigma\mathbf{V}^\top$，用 Gram-power identity：
$$\mathbf{X}(\mathbf{X}^\top\mathbf{X})^k = \mathbf{U}\Sigma^{2k+1}\mathbf{V}^\top$$

得到：
$$\mathcal{P}(\mathbf{X}; a, b, c) = \mathbf{U}(a\Sigma + b\Sigma^3 + c\Sigma^5)\mathbf{V}^\top = \mathbf{U}f(\Sigma)\mathbf{V}^\top$$

其中 $f(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$ 是标量 polynomial。

**三个推论**：
1. **Per-singular-value control**：matrix filter 等价于 scalar filter 独立作用在每个 singular value 上
2. **Singular vector invariance**：$\mathbf{U}, \mathbf{V}$ 完全不变
3. **3 维 coefficient design**：设计 matrix filter = 设计 3 个标量

这是整篇 paper 最 elegant 的步骤。Muon 的 iteration 看起来是矩阵运算，其实在每个 singular value 上独立作用——你只需要设计一个 scalar function $f: [0, 1] \to [0, 1]$ 的形状。

Composition of steps 也 compose：
$$\mathcal{P}_t \circ \dots \circ \mathcal{P}_1(\mathbf{X}) = \mathbf{U}(f_t \circ \dots \circ f_1)(\Sigma)\mathbf{V}^\top$$

### 两阶段：Promotion + Suppression

单个 quintic polynomial 只有 3 个自由度，无法做到 sharp transition。作者的 trick：5 步 NS 分两阶段——$k_p$ 步 Promotion + $k_s = 5 - k_p$ 步 Suppression。

**Promotion**（放大阶段）：
$$f_p(\sigma) = 1.875\sigma - 1.25\sigma^3 + 0.375\sigma^5$$

设计约束：
- (P1) $f_p(1) = 1$：已经在 1 的不动
- (P2) $f_p'(1) = 0$：$\sigma = 1$ 处一阶平稳，小扰动不放大
- (P3) $f_p''(1) \leq 0$：边界处凹，防止向上弯把值推出 $[0, 1]$
- 额外要求 $f_p$ 在 $[0, 1]$ 单调——singular value 的相对顺序保持

在 (P1)(P2) 约束下剩 1 个自由度，论文取 $a_p = 1.875$（最大可行 slope at origin）使导数成为 perfect square：
$$f_p'(\sigma) = 1.875(1-\sigma^2)^2 \geq 0$$

**Intuition**：Promotion 阶段把小的 singular value 推大，让它们越过后续 Suppression 的 threshold；同时保持单调性，让 Suppression 能识别哪些是 tail。

**Suppression**（抑制阶段）：
$$f_s(\sigma) = 2.5\sigma^3 - 1.5\sigma^5$$

设计约束：
- (S1) $f_s(1) = 1$
- (S2) $f_s'(1) = 0$
- (S3) $f_s'(0) = 0$：在原点处一阶导为 0，去掉线性项——小的 singular value 被高阶项推向 0

三个约束唯一确定三个系数。导数：
$$f_s'(\sigma) = 7.5\sigma^2(1-\sigma^2) \geq 0$$

**Intuition**：Suppression 把已经在 1 附近的钉死，把接近 0 的继续压到 0——这就是 high-pass filter 的核心。

**Composite**：
$$f_{\mathrm{Pion}} = f_s^{\circ k_s} \circ f_p^{\circ k_p}$$

$k_p \in \{0, 1, \dots, 5\}$ 是唯一超参，控制 cutoff。Empirically $k_s \geq 3$（即 $k_p \leq 2$）对 VLA 和 RLVR 都最好——Suppression 主导才能激进砍掉 noisy tail。

Fig. 3(d) 可视化：composite map 在 $[0, 1]$ 上有一个 sharp transition band——singular value 大于某个阈值被锚定在 1，小于阈值被压到 0。这就是 high-pass。

---

## Per-Head Mode：为什么 RLVR 还需要额外步骤

### Attention head 的 heterogeneity

Pretrained 模型的 attention heads 有显著 norm 差异。Proposition G.1 给出 forward factorization：

$$\mathbf{S}^h = \frac{\|\mathbf{W}_Q^h\|_F \|\mathbf{W}_K^h\|_F}{\sqrt{d_k}} \cdot \mathbf{X}\widetilde{\mathbf{W}}^h\mathbf{X}^\top$$

变量：
- $\mathbf{W}_Q^h, \mathbf{W}_K^h$：head $h$ 的 Q/K projection
- $d_k$：head dimension
- $\beta_h := \frac{\|\mathbf{W}_Q^h\|_F \|\mathbf{W}_K^h\|_F}{\sqrt{d_k}}$：effective inverse temperature
- $\widetilde{\mathbf{W}}^h$：归一化 shape

**Intuition**：head 的 Frobenius norm 大 → attention pattern 锐化；norm 小 → attention 平坦。不同 head 的 norm 不同 = 不同 head 的 attention sharpness 不同 = 功能特化。

Backward 的 gradient bound 也跟 head norm 耦合：
$$\left\|\frac{\partial \mathcal{L}}{\partial \mathbf{W}_Q^h}\right\|_F \leq C_X \|\mathbf{W}_K^h\|_2 \|\mathbf{W}_V^h\|_2 \|\mathbf{W}_O^h\|_2 \|\mathbf{G}\|_F$$

每个 head 的 gradient magnitude 被它自己的 weight norm 调制——所以 pretrained 模型每个 head 自然需要不同尺度的 update。

### Default mode 的问题

Default Pion 把整个 projection matrix 当一个 block 应用 high-pass NS——这**抹平了 head 之间的差异**。Fig. 4(b) 测量 Q projection 的 cross-head variance：
- 训练前 $\mathrm{Var}(\|\mathbf{W}_{0,Q}^h\|_F)$：跨 28 层都有 non-trivial variance
- Default Pion 训练后 update variance $\mathrm{Var}(\|\mathbf{W}_{*,Q}^h - \mathbf{W}_{0,Q}^h\|_F)$：几乎 flat——heads 被均匀 update

**Per-head mode**：先 reshape attention projection 沿 head dimension 拆成 $H$ 个 sub-block，每个 head 独立做 high-pass NS，最后 reshape 回去。**只多一个 reshape，per-step cost 完全相同**。

Fig. 4(a) 实验对比：
- Muon (default): collapse
- Muon (per-head): 仍 collapse——lack of noise adaptiveness 是主因
- Pion (default): underperform AdamW
- **Pion (per-head)**: 超越 AdamW

VLA 不需要 per-head mode（Table A5）——action head 是 from-scratch training，没有 pretrained heterogeneity 可言。Default mode 略好（97.25 vs 96.85）。

Reference: [Induction Heads (Anthropic)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html), [QK-Norm](https://arxiv.org/abs/2306.17284)

---

## 实验数据

### VLA on LIBERO (VLA-Adapter, ℓ1-regression)

固定 budget（Object 1500 steps，其他 15000 steps）：

| Task | AdamW | Muon | Pion |
|------|-------|------|------|
| Object | 32.2 | 97.0 | **100.0** |
| Spatial | 97.0 | 99.0 | 99.4 |
| Goal | 89.2 | 95.8 | 97.2 |
| Long | 69.6 | 88.0 | 92.4 |

Convergence speed：Pion 在 500 steps 达到 95.4%，1500 steps 饱和到 100%。AdamW 远未收敛，Muon 滞后。

### VLANeXt (flow-matching) on LIBERO-Plus

Perturbation robustness：

| Perturbation | AdamW | Muon | Pion |
|---|---|---|---|
| Language | 54.5 | 77.5 | **86.9** |
| Noise | 66.4 | 70.0 | **76.1** |
| Robot | 47.0 | 57.4 | **63.2** |

**Intuition**：Muon 的 uniform whitening 在 training set 上 over-amplify "non-generalizable directions"——这些方向在 perturbed test 上是 noise。Pion 的 high-pass 天然 select "generalizable" 的 leading directions。

### Real Robot (Franka Research 3 + π0.5 + DROID)

30 randomized trials per task，20000 training steps：

| Optimizer | Cucumber→Plate | Cube→Plate | Cube→Bowl | Average |
|-----------|---------------|-----------|----------|---------|
| AdamW | 40.0 | 33.3 | 20.0 | 31.1 |
| Muon | 56.7 | 33.3 | 26.7 | 38.9 |
| **Pion** | **93.3** | **83.3** | **80.0** | **85.6** |

物理机器人 tolerance 紧，Pion 的收益被放大。

### RLVR on Qwen3-1.7B/4B with GRPO/GMPO

8 个 setting（2 algorithms × 2 model sizes × 2 benchmarks）：
- **Muon 全部 collapse 到 0**——甚至低于 initial checkpoint
- Pion 全部超过 AdamW，且收敛更快

Fig. 7：Pion 训练全程 gradient SNR 始终高于 AdamW——high-pass 不仅避免 noise 放大，还**主动提高 SNR**（noise 被 contract 到 0）。

### Reverse Ablation: LPMuon

作者设计 Low-pass Muon 作为 mirror——保留小 singular values，砍掉大的。Appendix L 用 L-BFGS-B 多起点拟合 15 个 coefficient。

Fig. 8(b) 结果：**LPMuon fails to train at all**——accuracy 一直停在 initial checkpoint。这隔离出 Pion 的 gain 确实来自"high-pass 方向"，而非"任何 spectral shaping"或"per-head reshape"。

### Modality-wise ablation (Table A6)

| Setting | Vision | Language | Action | Success Rate |
|---|---|---|---|---|
| S1 | AdamW | AdamW | AdamW | 43.6 |
| S2 | AdamW | AdamW | Muon | 40.0 |
| S3 | AdamW | AdamW | Pion | 73.6 |
| S5 | AdamW | Pion | AdamW | 73.8 |
| S7 | Pion | AdamW | AdamW | 17.8 |
| S8 | Muon | Muon | Muon | 97.0 |
| S9 | Muon | Muon | Pion | **100.0** |

**Intuition**：每个 module 的 spectral 结构决定最适合的 optimizer。Vision/Language 是 high-rank（Muon 的 uniform whitening 利于 exploration），Action 是 low-rank（需要 Pion 的高通）。S7 Pion on Vision collapse 到 17.8%——Pion 砍掉了 vision 的 informative tail，证明 high-pass 不是 universal good。

---

## 直觉总结与联想

### Muon vs. Pion 的对立

Muon 是**平等主义者**：所有 singular direction 同等对待。这是 pretraining 的正确 prior——gradient 是 high-rank，每个方向都 informative，uniform whitening = spectral exploration。

Pion 是**精英主义者**：大 singular value 锚定，小的砍掉。这是 post-training 的正确 prior——gradient 是 low-rank 或 low-SNR，signal 集中在少数 leading directions，tail 是 noise。

两者的本质区别：**对 gradient spectral 结构的假设**。Pretraining 和 post-training 的 gradient 结构不同，所以需要不同的 inductive bias。

### 与 signal processing 的连接

Pion 的 Promotion + Suppression 设计简直就是 digital filter design：
- Promotion 像 pre-emphasis filter（放大弱信号）
- Suppression 像 low-pass（在 frequency domain 但这里是 singular value domain）
- Composite 是 high-pass

可以联想：
- **Adaptive filter**：每层每步根据 gradient 谱自动选 $k_p$——erank 高就多用 Promotion，erank 低就多用 Suppression
- **Multi-band filter**：不止 head/tail 两段，而是多 band 不同增益
- **Wavelet-like decomposition**：不同 scale 不同处理

### 与 LoRA / low-rank adaptation 的对照

LRMuon 用 SVD project 到 top-k 子空间——本质上等价于"先 LoRA 化 momentum 再 msign"。但 fixed rank $k$ 不能跨层、跨步自适应，且每步 SVD 代价高（Fig. 1c 显示 15× slowdown）。

Pion 用 soft high-pass 替代 hard top-k truncation——这是 sparse recovery 里 **soft thresholding vs. hard thresholding** 的对比。Soft threshold 更鲁棒、更可微、更易 tune。

更进一步：Pion 的成功可以 reinterpret 为"spectral domain 的 credit assignment"。RLVR 的难题是 trajectory-level reward 如何 back-propagate 到 per-token gradient——制造了大量 noise direction。Pion 假设"informative signal 在 leading singular directions" = "policy update 的有效自由度远小于 weight matrix 维度"——和 LoRA、adapters、prefix tuning 的 low-rank assumption 异曲同工。

### Per-head heterogeneity 的更广含义

Per-head norm heterogeneity 呼应了一批工作：
- Attention heads 有 functional specialization（induction heads, positional heads, ...）
- Pruning 文献显示 head norms 长尾分布
- QK-norm 显示 head norm 影响 attention entropy

Pion 的 per-head mode 实质是"每个 head 用自己的 spectral geometry"——尊重 pretrained 模型的 functional specialization。

### Information theory 的角度

erank 用 Shannon entropy 定义——singular value 谱的"有效维度"。如果 gradient 的 erank 是 $r_{\mathrm{eff}}$，signal 在 $r_{\mathrm{eff}}$ 维子空间里，剩下 $n - r_{\mathrm{eff}}$ 维是 noise。Muon 的 uniform whitening 把 $n$ 维全部等权，noise/signal ratio 是 $\sqrt{(n - r_{\mathrm{eff}})/r_{\mathrm{eff}}}$——$r_{\mathrm{eff}}$ 越小，noise 比例越大，Muon 越糟。Pion 的高通把这个 ratio 推回 signal-only regime。

### RLVR 中 Muon 失败的更广含义

Muon 在 RLVR 上 collapse 是非常重要的 negative result——matrix-aware optimizer 在 post-training 不能直接 drop-in 替换 AdamW。给所有 RLVR infrastructure 一个 caution：如果你切到 Muon，你可能在 noise 上 faster collapse。**Reference**: [Muon is scalable for LLM training](https://arxiv.org/abs/2502.16982), [Normuon](https://arxiv.org/abs/2510.05491)

### Annealing 的联想

$k_p$ 是 single hyperparameter 控制 cutoff——可以联想 annealing schedule：
- 早期 RLVR：noise 还不严重，用大 $k_p$（接近 Muon）保留 exploration
- 后期 RLVR：policy sharpening，需要 fine adjustment，用小 $k_p$（强 high-pass）抑制 noise

类似 simulated annealing 中 temperature 退火。

### Polynomial 自由度的扩展

NS 用 quintic polynomial $f(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$。可以联想：
- 用更高阶 polynomial（septic, 7th degree）会有更多自由度，可能 sharper transition
- 但每步 matmul 更多——scalability tradeoff
- 不同 stage 用不同 degree 也可以

Appendix L 的 LPMuon 直接 fit 15 个 coefficient——但 Pion 用 closed-form constraint 推出 6 个 coefficient，可解释性更强，对 hyperparameter $k_p$ 更鲁棒。

---

## 一些小批评

1. **只测了 Qwen3**：没测 Llama、Mistral、DeepSeek 系列。Muon 在不同 family 的行为可能不同。
2. **Per-head mode 只在 attention**：FFN 的两个 matrix 没有类似的"sub-block"概念——但 FFN 的 gradient 也可能 low-rank。Pion 在 FFN 上用 default mode，可能次优。
3. **$k_p$ 选择**：empirical 用 $k_s \geq 3$，但没有系统的 hyperparameter sweep 显示 cutoff frequency 与 erank 的关系。
4. **LPMuon 的 cutoff $\tau = 0.5$**：选 0.5 比较任意，可能 $\tau$ 调小一点 LPMuon 也不那么崩——不过作者意图是 reverse ablation，不需要 fair comparison。
5. **Real robot 只 30 trials**：方差大，但 85.6% vs 38.9% 的 gap 显著。
6. **没测 LLM pretraining 上 Pion 是否真的 underperform Muon**——这是 limitation 里提到但没实验验证的。如果 Pion 在 pretraining 上也不输 Muon，那 high-pass 可能是更通用的选择。

---

## 最终直觉

Pion 的 elegant 之处在于：通过 SVD factorization 把 matrix filter design 完全 reduced 到 scalar polynomial design；用 closed-form constraint 推出 Promotion + Suppression 两个 polynomial；保持 Muon 的 per-step cost；用 reverse ablation (LPMuon) 隔离出 high-pass 方向的关键性。

更深层的 message：**matrix-aware optimizer 的 inductive bias 不是 universal**。Uniform spectral whitening 是 pretraining 的正确 prior，但在 cross-modal / low-SNR / post-training 场景，prior 应该是 "informative signal 在 leading singular directions，tail 是 noise"。

这给了 framework：根据 gradient 的 spectral 结构（rank、SNR）选合适的 spectral filter 形状——uniform（Muon）、high-pass（Pion）、low-pass（LPMuon）、甚至 multi-band。Optimizer 设计从"element-wise adaptive"（Adam 系列）走向"spectral-shape adaptive"（Pion 系列），这是一个 exciting 的方向。

**Reference 链接汇总**：
- Paper GitHub: 见原文顶端链接
- Muon: https://kellerjordan.github.io/posts/muon/
- Muon scalable: https://arxiv.org/abs/2502.16982
- Shampoo: https://arxiv.org/abs/1802.09568
- SOAP: https://arxiv.org/abs/2409.11321
- Polar Express: https://arxiv.org/abs/2505.16932
- Low-Rank Muon: https://arxiv.org/abs/2509.11983
- UBLR: https://arxiv.org/abs/2510.17802
- LIBERO: https://arxiv.org/abs/2306.03310
- LIBERO-Plus: https://arxiv.org/abs/2510.13626
- OpenVLA: https://arxiv.org/abs/2406.09246
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- DROID: https://arxiv.org/abs/2403.12945
- GRPO / DeepSeekMath: https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- GMPO: https://arxiv.org/abs/2507.20673
- DAPO: https://arxiv.org/abs/2503.14476
- Qwen3: https://arxiv.org/abs/2505.09388
- MATH: https://arxiv.org/abs/2103.03874
- GSM8K: https://arxiv.org/abs/2110.14168
- Flow Matching: https://arxiv.org/abs/2210.02747
- Polar Decomposition: https://en.wikipedia.org/wiki/Polar_decomposition
- Matrix Sign Function (Higham): https://epubs.siam.org/doi/10.1137/1.9780898717778
- Induction Heads: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- QK-Norm: https://arxiv.org/abs/2306.17284
- Effective Rank (Roy & Vetterli): https://ieeexplore.ieee.org/document/7099090

---

# Rethinking Muon Beyond Pretraining: Pion 的 Spectral High-pass 设计

Andrej，这篇 paper 我读得很兴奋——它精确地戳中了 Muon 这类 matrix-aware optimizer 的一个隐藏假设：**uniform spectral whitening 在 pretraining 之外的场景会反向操作**。下面我从 intuition → 数学 → 实验 → 联想四个层次展开。

---

## 1. Muon 的本质与"隐藏假设"

Muon 的核心 update rule（Eq. 1）：

$$\Theta_t = \Theta_{t-1} - \eta \, \mathrm{msign}(\mathbf{M}_t)$$

变量解释：
- $\Theta_t \in \mathbb{R}^{m \times n}$：第 $t$ 步的 weight matrix
- $\mathbf{M}_t = \mu \mathbf{M}_{t-1} + \mathbf{G}_t$：momentum buffer，$\mu$ 是 momentum coefficient，$\mathbf{G}_t$ 是 stochastic gradient
- $\eta > 0$：step size
- $\mathrm{msign}(\cdot)$：matrix sign operator

而 $\mathrm{msign}$ 通过 SVD 定义（Eq. 2）：

$$\mathrm{msign}(\mathbf{M}) = \mathbf{U} \, \mathrm{sign}(\Sigma) \, \mathbf{V}^\top = \mathbf{U}\mathbf{V}^\top$$

变量解释：
- $\mathbf{M} = \mathbf{U}\Sigma\mathbf{V}^\top$：compact SVD
- $\mathbf{U} \in \mathbb{R}^{m \times r}$, $\mathbf{V} \in \mathbb{R}^{n \times r}$：左右 singular vector matrices
- $\Sigma \in \mathbb{R}^{r \times r}$：对角阵，对角元素是 $r = \mathrm{rank}(\mathbf{M})$ 个严格正的 singular values $\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_r > 0$
- $\mathrm{sign}(\Sigma) = \mathbf{I}_r$：把每个 singular value 变成 1

**关键 intuition**：Muon 在 spectral domain 做的是"全频段白化"——所有 singular direction 被赋予相同的 magnitude 1，singular vectors 保持不变。这是 spectral norm 下的 steepest descent。

Newton-Schulz (NS) iteration（Eq. 3）用来近似 msign，避免显式 SVD：

$$\mathbf{X} \gets a\mathbf{X} + b\mathbf{X}\mathbf{X}^\top\mathbf{X} + c\mathbf{X}(\mathbf{X}^\top\mathbf{X})^2, \quad (a,b,c) = (3.4445, -4.7750, 2.0315)$$

变量解释：
- 输入预先归一化为 $\mathbf{X} \leftarrow \mathbf{X}/(\|\mathbf{X}\|_F + \epsilon)$，使所有 singular values 落在 $[0, 1]$
- $\|\cdot\|_F$：Frobenius norm
- 三次 quintic polynomial iteration 收敛到 matrix sign

**隐藏假设**：Muon 假设 gradient 的所有 singular direction 都"同等 informative"。在 LLM pretraining 这个假设成立——gradient 是 high-rank 的，每一步都在全谱上探索。

---

## 2. 为什么 Beyond Pretraining 会 Fail：两个 Spectral Mismatch

### 2.1 VLA 的 rank heterogeneity

VLA policy 分解为 $\Theta = \{\Theta_{\mathrm{VLM}}, \Theta_{\mathrm{action}}\}$，三个 module（vision、language、action）的 gradient intrinsic dimensionality 差异巨大。

作者用 **effective rank (erank)** 量化（Eq. 4）：

$$\mathrm{erank}(\mathbf{G}) := \exp(H(\mathbf{p})), \quad H(\mathbf{p}) = -\sum_{i=1}^n p_i \log p_i, \quad p_i = \frac{\sigma_i(\mathbf{G})}{\sum_{j=1}^n \sigma_j(\mathbf{G})}$$

变量解释：
- $\mathbf{G} \in \mathbb{R}^{m \times n}$：gradient matrix，假设 $n \leq m$
- $\sigma_i(\mathbf{G})$：第 $i$ 个 singular value
- $\mathbf{p} = [p_1, \dots, p_n]^\top$：singular value 的归一化分布（像一个 probability distribution）
- $H(\mathbf{p})$：Shannon entropy
- $\exp(H(\mathbf{p}))$：entropy 的指数，perplexity 的概念——singular value 谱的"有效支撑宽度"

**Intuition**：如果只有一个非零 singular value，$H=0$，erank=1；如果所有 singular value 相等，$H = \log n$，erank=$n$。

Fig. 1(a) 的实验观察：
- Vision module：erank 最高（pixel-level 统计丰富）
- Language module：中等
- **Action module：erank 最低**——因为 action 只是 7-dim vector（end-effector translation + rotation + binary gripper）

对 action module 应用 Muon：singular value 谱尾部的小值（noise floor）被拉到 1，noise 被放大到和 informative direction 同等量级。这就 corrupt 了 update。

### 2.2 RLVR 的 low-SNR 问题

定义 gradient SNR（Eq. 5）：

$$\mathrm{SNR}(\mathbf{G}) := \frac{\|\mathbb{E}[\mathbf{G}]\|_F^2}{\mathbb{E}[\|\mathbf{G} - \mathbb{E}[\mathbf{G}]\|_F^2]}$$

变量解释：
- 分子：gradient 期望的 Frobenius norm 平方（signal power）
- 分母：gradient 的 variance（noise power）
- 期望是 over batch

**Appendix C 的 closed-form SNR 推导**（这部分非常漂亮）：

SFT 的 gradient estimator（Eq. A7）：
$$\hat{\mathbf{g}}_{\mathrm{SFT}} = -\frac{1}{g}\sum_{j=1}^g \sum_{t=1}^T \nabla_\Theta \ell_{j,t}(\Theta)$$

变量：
- $g$：batch size
- $T$：序列长度
- $\ell_{j,t}(\Theta) = \log \pi_\Theta(o_{j,t}^\star | \mathbf{q}_j, \mathbf{o}_{j,<t}^\star)$：第 $j$ 个 sample 第 $t$ 个 token 的 log-likelihood

SFT 的 SNR（Eq. A12）：
$$\mathrm{SNR}_{\mathrm{SFT}} = gT \frac{\|\bar{\mathbf{s}}\|^2}{\sigma_s^2}$$

变量：
- $\bar{\mathbf{s}} := \mathbb{E}[\nabla_\Theta \ell_{i,t}]$：per-token score 的期望（deterministic signal）
- $\sigma_s^2$：per-token score variance

GRPO 的 on-policy gradient estimator（Eq. A9）：
$$\hat{\mathbf{g}}_{\mathrm{GRPO}} = \frac{1}{g}\sum_{i=1}^g \hat{a}_i \bar{\mathbf{S}}_i, \quad \bar{\mathbf{S}}_i := \frac{1}{|\mathbf{o}_i|}\sum_{t=1}^{|\mathbf{o}_i|} \nabla_\Theta \ell_{i,t}(\Theta)$$

变量：
- $\hat{a}_i = (R_i - \bar{R})/(\mathrm{std}(R) + \epsilon)$：group-relative advantage
- $R_i \in \{0, 1\}$：binary reward
- $\bar{R} = \frac{1}{g}\sum_j R_j$：group reward mean

关键的 decomposition（Eq. A13）：
$$\bar{\mathbf{S}}_i = \underbrace{\mathbb{E}[\bar{\mathbf{S}}_i | R_i]}_{\mathbf{u}_i} + \underbrace{\bar{\mathbf{S}}_i - \mathbb{E}[\bar{\mathbf{S}}_i | R_i]}_{\mathbf{v}_i}, \quad \mathbf{u}_i = \mu_S^- + R_i \Delta$$

变量：
- $\mu_S^+ := \mathbb{E}[\bar{\mathbf{S}}_i | R_i = 1]$：成功 trajectory 的 expected score
- $\mu_S^- := \mathbb{E}[\bar{\mathbf{S}}_i | R_i = 0]$：失败 trajectory 的 expected score
- $\Delta := \mu_S^+ - \mu_S^-$：**成功与失败的 score gap**——这是 RLVR 的真正 signal

GRPO 的 SNR（Eq. A20）：
$$\mathrm{SNR}_{\mathrm{GRPO}} \approx gT \frac{\kappa_g(p)\|\Delta\|^2}{\sigma_s^2}, \quad \kappa_g(p) := q_{\mathrm{nd}}\rho_g(p)^2$$

变量：
- $p = p(\mathbf{q})$：单 prompt 的 success probability
- $q_{\mathrm{nd}} = 1 - p^g - (1-p)^g$：non-degenerate group probability（组里既有成功也有失败）
- $\rho_g(p) := \mathbb{E}[\sqrt{(K/g)(1 - K/g)} | 0 < K < g]$，$K \sim \mathrm{Binomial}(g, p)$
- 大 $g$ 近似下 $\kappa_g(p) \approx p(1-p)$

**SNR ratio**（Eq. A21）：
$$\frac{\mathrm{SNR}_{\mathrm{SFT}}}{\mathrm{SNR}_{\mathrm{GRPO}}} \approx \frac{\|\bar{\mathbf{s}}\|^2}{\kappa_g(p)\|\Delta\|^2}$$

两种 regime 让这个 ratio 爆炸：
1. **Extreme difficulty** $p \to 0$ 或 $p \to 1$：$\kappa_g(p) \to 0$，group 大部分 degenerate，advantage 消失
2. **Low distinctiveness** $\|\Delta\| \ll \|\bar{\mathbf{s}}\|$：成功和失败的 trajectory 看起来很像

加上 off-policy 的额外 degradation（Eq. A26）：
$$\frac{\mathrm{SNR}_{\mathrm{SFT}}}{\mathrm{SNR}_{\mathrm{GRPO}}^{\mathrm{full}}} \gtrsim \frac{\|\bar{\mathbf{s}}\|^2}{\kappa_g(p)\|\Delta\|^2} \cdot (1+\chi^2) \cdot \frac{1}{1-\alpha}$$

变量：
- $\chi^2 := \mathbb{E}_{\pi_{\mathrm{old}}}[r_{i,t}^2] - 1$：per-token chi-squared divergence between $\pi_\Theta$ and $\pi_{\mathrm{old}}$（importance sampling 引入）
- $\alpha$：clip fraction（PPO-style clipping 抑制部分 token 的 gradient）

**结论**：RLVR 的 gradient SNR 远低于 SFT，Muon 的 uniform whitening 把 noise directions 提到和 signal directions 同等量级 → policy collapse。Fig. 2(b) 实验验证：Muon 训练 Qwen3-1.7B GRPO 在 MATH500 上 accuracy 从初始 checkpoint 跌到 0。

---

## 3. Pion 的核心设计：把 Matrix Filter 降成 Scalar Polynomial

### 3.1 SVD Factorization（Appendix D 的关键 lemma）

这是整篇 paper 最 elegant 的数学步骤。考虑单个 NS step 的 quintic matrix polynomial（Eq. A28）：

$$\mathcal{P}(\mathbf{X}; a, b, c) := a\mathbf{X} + b\mathbf{X}\mathbf{X}^\top\mathbf{X} + c\mathbf{X}(\mathbf{X}^\top\mathbf{X})^2$$

代入 SVD $\mathbf{X} = \mathbf{U}\Sigma\mathbf{V}^\top$，用 Gram-power identity（Eq. A30）：

$$\mathbf{X}(\mathbf{X}^\top\mathbf{X})^k = \mathbf{U}\Sigma^{2k+1}\mathbf{V}^\top$$

得到（Eq. A31）：

$$\mathcal{P}(\mathbf{X}; a, b, c) = \mathbf{U}\underbrace{(a\Sigma + b\Sigma^3 + c\Sigma^5)}_{f(\Sigma; a,b,c)}\mathbf{V}^\top = \mathbf{U} f(\Sigma) \mathbf{V}^\top$$

**三个推论**：
1. **Per-singular-value control**：矩阵层 filter 等价于 scalar map $\sigma_i \mapsto f(\sigma_i)$ 独立作用在每个 singular value 上
2. **Singular vector invariance**：$\mathbf{U}$ 和 $\mathbf{V}$ 完全不变
3. **3-dim coefficient design**：设计 matrix filter = 设计 3 个 scalar 系数

Composition of NS steps（Eq. A32）：
$$\mathcal{P}_t \circ \dots \circ \mathcal{P}_1 (\mathbf{X}) = \mathbf{U}(f_t \circ \dots \circ f_1)(\Sigma)\mathbf{V}^\top$$

所以 Pion 把"设计 matrix-level spectral high-pass filter"完全 reduced 到"设计 scalar polynomial 在 $[0, 1]$ 上的 shape"。

### 3.2 为什么单个 polynomial 不够

一个 quintic polynomial $f(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$ 只有 3 个自由度，无法同时做到：
- 在 $\sigma = 1$ 处 pinning（fixed point + first-order stationarity）
- 在 $\sigma = 0$ 处抑制（zero first derivative，让高阶项 dominate）
- 中间有 sharp transition

作者的天才想法：**分两阶段**，用 5 步 NS（保持 Muon 的 per-step cost）分配为 $k_p$ 步 Promotion + $k_s = 5 - k_p$ 步 Suppression。

### 3.3 Promotion polynomial（Eq. 7）

$$f_p(\sigma) = a_p \sigma + b_p \sigma^3 + c_p \sigma^5, \quad (a_p, b_p, c_p) = (1.875, -1.25, 0.375)$$

**设计约束**（Appendix E.1）：
- **(P1) Fixed point**：$f_p(1) = 1$——已经在 1 的 singular value 不动
- **(P2) First-order stationarity**：$f_p'(1) = 0$——$\sigma = 1$ 周围的小扰动不被放大
- **(P3) Boundary concavity**：$f_p''(1) \leq 0$——防止 Promotion 在 $\sigma = 1$ 附近向上弯，把附近值推出 $[0, 1]$

(P1) 和 (P2) 给出 2 个等式约束，留下 1 维 family（Eq. A35）：
$$b_p = -\frac{1 + 4c_p}{2}, \quad a_p = \frac{3 + 2c_p}{2}$$

Global monotonicity 推导（Eq. A38–A40）：令 $u = \sigma^2$，则 $g(u) = f_p'(\sigma) = a_p + 3b_p u + 5c_p u^2$ 是 $u$ 的二次式，$g(1) = 0$，因式分解：
$$g(u) = 5c_p(u-1)(u-r), \quad r = \frac{a_p}{5c_p} = \frac{3 + 2c_p}{10c_p}$$

要 $g(u) \geq 0$ 在 $u \in [0, 1]$，得到 $c_p \in [-1.5, 0.375]$，对应 $a_p \in [0, 1.875]$。

**关键选择**：$a_p = 1.875$（取最大值），最大化 origin 处的 slope $f_p'(0) = a_p$。直觉：Promotion 步骤数有限（$k_p \leq 5$），slopes 越大，小 singular value 被推到 suppression threshold 之上的速度越快。

代入得到 $c_p = 0.375, b_p = -1.25$。导数变成 perfect square（Eq. A43）：
$$f_p'(\sigma) = 1.875(1 - \sigma^2)^2 \geq 0 \quad \forall \sigma \in [0, 1]$$

单调性保证：singular value 的相对顺序在每一步 Promotion 后保持，suppression 阶段才能"识别"哪些是 tail。

### 3.4 Suppression polynomial（Eq. 8）

$$f_s(\sigma) = a_s \sigma + b_s \sigma^3 + c_s \sigma^5, \quad (a_s, b_s, c_s) = (0, 2.5, -1.5)$$

**设计约束**（Appendix E.2）：
- **(S1) Fixed point**：$f_s(1) = 1$——继承自 Promotion
- **(S2) First-order stationarity**：$f_s'(1) = 0$——pinning 大 singular value
- **(S3) Spectral filtering at origin**：$f_s'(0) = 0$——去掉 linear term，使小 singular value 被高阶项（$\sigma^3, \sigma^5$）推向 0

(S3) 直接给 $a_s = 0$，(S1)(S2) 解 $2 \times 2$ 线性系统（Eq. A44）：
$$b_s + c_s = 1, \quad 3b_s + 5c_s = 0 \Rightarrow b_s = 2.5, c_s = -1.5$$

唯一解，无 residual freedom。导数（Eq. A46）：
$$f_s'(\sigma) = 7.5\sigma^2(1 - \sigma^2) \geq 0 \quad \forall \sigma \in [0, 1]$$

vanish 在 $\sigma \in \{0, 1\}$ 两个端点，保证 monotonicity。

### 3.5 Composite high-pass

$$f_{\mathrm{Pion}} = f_s^{\circ k_s} \circ f_p^{\circ k_p}$$

- $k_p \in \{0, 1, \dots, 5\}$：唯一 hyperparameter，控制 cutoff
- Empirically $k_s \geq 3$（即 $k_p \leq 2$）对 VLA 和 RLVR 都最好——suppression dominant 才能激进地砍掉 noisy tail

Fig. 3(d) 可视化：composite map 在 $[0, 1]$ 上呈现 sharp transition——大 singular value 锚定在 1，小 singular value 收缩到 0，中间有一个可控的 transition band。

---

## 4. Per-Head Mode：保留 Pretrained Heterogeneity

为什么 RLVR 需要 per-head mode 而 VLA 不需要？

### 4.1 Per-Head Norm Heterogeneity（Appendix G 的 Proposition G.1）

标准 multi-head attention 的 forward logit factorization（Eq. A47）：
$$\mathbf{S}^h = \underbrace{\frac{\|\mathbf{W}_Q^h\|_F \|\mathbf{W}_K^h\|_F}{\sqrt{d_k}}}_{\beta_h} \cdot \mathbf{X}\widetilde{\mathbf{W}}^h \mathbf{X}^\top, \quad \widetilde{\mathbf{W}}^h := \frac{\mathbf{W}_Q^h (\mathbf{W}_K^h)^\top}{\|\mathbf{W}_Q^h\|_F \|\mathbf{W}_K^h\|_F}$$

变量：
- $\mathbf{W}_Q^h, \mathbf{W}_K^h, \mathbf{W}_V^h \in \mathbb{R}^{d \times d_k}$：head $h$ 的 Q/K/V projection
- $\mathbf{W}_O^h \in \mathbb{R}^{d_k \times d}$：head $h$ 的 output projection
- $d_k$：head dimension
- $\beta_h$：**effective inverse temperature**——Q/K 的 Frobenius norm 乘积决定 attention sharpness
- $\widetilde{\mathbf{W}}^h$：normalized shape

**直觉**：head 的 norm 大 → attention pattern 锐化（更 peaked）；norm 小 → attention 平坦。

Backward 的 gradient norm bound（Eq. A49–A52）：
$$\left\|\frac{\partial \mathcal{L}}{\partial \mathbf{W}_Q^h}\right\|_F \leq C_X \|\mathbf{W}_K^h\|_2 \|\mathbf{W}_V^h\|_2 \|\mathbf{W}_O^h\|_F \|\mathbf{G}\|_F$$

其中 $C_X := 2\|\mathbf{X}\|_2^3/\sqrt{d_k}$，$\mathbf{G} = \partial \mathcal{L}/\partial \mathbf{Z}$。

**直觉**：每个 head 的 gradient magnitude 被 its own weight norms 调制——pretrained 模型每个 head 的 norm 各异（Fig. 4(b) 顶部），所以每个 head 自然需要不同尺度的 update。

### 4.2 Default vs. Per-Head Mode

- **Default mode**（Alg. 2）：对整个 weight matrix $\mathbf{M}_t \in \mathbb{R}^{m \times n}$ 应用 high-pass NS，用于 VLA
- **Per-head mode**（Alg. 3）：先 reshape attention projection 沿 head dimension 拆成 $H$ 个 sub-block $\{\mathbf{M}_t^h\}_{h=1}^H$，每个 head 独立做 high-pass NS，最后 reshape 回去——**只多一个 reshape，per-step cost 完全相同**

Fig. 4(a) 实验对比：
- Muon（default）：collapse
- Muon（per-head）：仍然 collapse——因为 lack of noise adaptiveness 是主因
- Pion（default）：underperform AdamW
- **Pion（per-head）**：超越 AdamW——spectral high-pass 是主因，per-head 是辅助

Fig. 4(b) 测量 cross-head Q projection variance：
- 训练前 $\mathrm{Var}(\|\mathbf{W}_{0,Q}^h\|_F)$：跨 28 层都有 non-trivial variance
- Default Pion 训练后 $\mathrm{Var}(\|\mathbf{W}_{\*,Q}^h - \mathbf{W}_{0,Q}^h\|_F)$：几乎 flat——heads 被均匀 update，丢掉 heterogeneity
- Per-head Pion：layer-dependent heterogeneous updates

### 4.3 VLA 为什么不需要 per-head

VLA 的 action head 是 from-scratch training，没有 pretrained per-head heterogeneity 可言。Table A5 印证：default mode 略好于 per-head mode（平均 97.25 vs. 96.85）。

---

## 5. 实验数据详解

### 5.1 VLA on LIBERO（VLA-Adapter）

Fig. 5(a) 在四个 task suite 上的 success rate（固定 budget：Object 1500 steps，其他 15000 steps）：

| Task | AdamW | Muon | Pion |
|------|-------|------|------|
| Object | 32.2 | 97.0 | **100.0** |
| Spatial | 97.0 | 99.0 | 99.4 |
| Goal | 89.2 | 95.8 | 97.2 |
| Long | 69.6 | 88.0 | 92.4 |

Fig. 5(b) 的 convergence speed：Pion 在 500 steps 达到 95.4%，1500 steps 饱和到 100%。AdamW 远未收敛，Muon 滞后。

### 5.2 VLANeXt (flow-matching) on LIBERO-Plus

Table 1 显示 Pion 在所有 perturbation 下都领先，尤其 robust 的：
- Language perturbation：+9% over Muon
- Noise perturbation：+6%
- Robot perturbation：+6%

**直觉解释**：Muon 的 uniform whitening 容易 over-amplify "non-generalizable noise directions"——这些方向在 training set 上有 signal，但在 perturbed test 上是 noise。Pion 的高通过滤天然 select "generalizable" 的 leading directions。

### 5.3 Real Robot (Franka Research 3 + π0.5 + DROID)

Table 3 在三个 grasp-and-place 任务上（每个 30 randomized trials，20000 training steps）：

| Optimizer | Cucumber→Plate | Cube→Plate | Cube→Bowl | Average |
|-----------|---------------|-----------|----------|---------|
| AdamW | 40.0 | 33.3 | 20.0 | 31.1 |
| Muon | 56.7 | 33.3 | 26.7 | 38.9 |
| **Pion** | **93.3** | **83.3** | **80.0** | **85.6** |

**直觉**：物理机器人的 tolerance 极紧，Pion 的 high-pass 在 action module 上的收益被放大。

### 5.4 RLVR on Qwen3-1.7B/4B with GRPO/GMPO

Fig. 6 八个 setting 全部显示：
- **Muon collapse 到 0**（甚至低于 initial checkpoint）
- Pion 超过 AdamW，且收敛更快
- 跨 algorithm（GRPO/GMPO）和 model size（1.7B/4B）一致

Fig. 7：Pion 训练全过程的 gradient SNR 始终高于 AdamW——high-pass 不只避免 noise 放大，还**主动提高 SNR**（因为 noise 被 contract 到 0，signal 被保留）。

### 5.5 Reverse Ablation：LPMuon

作者设计 Low-pass Muon 作为 mirror：保留小 singular values，砍掉大 singular values。Appendix L 详细讲 LPMuon 的 coefficient fitting——用 5 个 quintic polynomial 组合拟合一个 band indicator：
$$\tilde{f}_\theta(\sigma) \approx \mathrm{sign}(\sigma) \cdot \mathbb{1}[|\sigma| \leq \tau]$$

用 L-BFGS-B 多起点拟合 15 个 coefficient（Eq. A56 的 loss 含 pass-band、stop-band、overshoot、non-negativity 四项）。

Fig. 8(b) 结果：**LPMuon fails to train at all**——accuracy 一直停在 initial checkpoint。这隔离出 Pion 的 gain 确实来自"high-pass 方向"，而非"任何 spectral shaping"或"per-head reshape"或"NS 形式"。

### 5.6 Modality-wise Ablation（Table A6）

九个配置 S1–S9 测试不同 module 用不同 optimizer：
- S2（V/L=AdamW, A=Muon）：40.0%，比 S1（all AdamW）43.6% 还低——证明 Muon 在 action module 反效果
- S3（V/L=AdamW, A=Pion）：73.6%——Pion 在 action 上有用
- S5（V/L=Pion, L=AdamW, A=AdamW）：73.8%——Pion 在 language 上不如 Muon
- S7（V=Pion, L/A=AdamW）：17.8% collapse——Pion 在 vision 上反而砍掉 informative tail
- S9（V/L=Muon, A=Pion）：**100.0%**——最优 assignment

**直觉**：每个 module 的 spectral 结构决定最适合的 optimizer。Vision/Language 是 high-rank（Muon 的 uniform whitening 利于 exploration），Action 是 low-rank（需要 Pion 的高通）。

---

## 6. Limitation 与 Open Question

Appendix M 老实承认：Pion 在 LLM pretraining 上**应该** underperform Muon——因为 pretraining 的 gradient 是 high-rank 的，所有 singular direction 都 informative，Pion 的高通会砍掉 useful exploration direction。

这给了一个很自然的 follow-up：**能不能设计一个 adaptive cutoff**，让 optimizer 在 pretraining 时退化为 Muon（uniform），在 post-training 时变成 Pion（high-pass）？比如根据 gradient 的 erank 动态调整 $k_p$——erank 高就多用 Promotion，erank 低就多用 Suppression。

---

## 7. 更深的 Intuition 与联想

### 7.1 Muon 与 Spectral Norm Steepest Descent

Muon 本质是 spectral norm 下的 steepest descent。Frobenius norm 下的 steepest descent 就是 SGD（梯度本身），spectral norm 下需要把所有 singular direction 等权化——这就是 msign 的几何意义。**Reference**: Muon blog (https://kellerjordan.github.io/posts/muon/), Shampoo (https://arxiv.org/abs/1802.09568), SOAP (https://arxiv.org/abs/2409.11321)。

### 7.2 Newton-Schulz 与 Matrix Sign Function

NS iteration 是经典的 matrix sign function 计算——可以追到 Higham's "Functions of Matrices" book。matrix sign function 在 numerical linear algebra 用于 polar decomposition $\mathbf{M} = \mathbf{U}\mathbf{H}$，sign 给的就是 $\mathbf{U}\mathbf{V}^\top$（unitary factor）。**Reference**: https://en.wikipedia.org/wiki/Polar_decomposition, Higham (2008)。

### 7.3 与 Spectral Filtering / Signal Processing 的连接

Pion 的设计简直就是 digital filter design——Promotion 像 pre-emphasis filter，Suppression 像 low-pass 在 frequency domain 但这里是在 singular value domain。整个论文的 framing 是"spectral high-pass"，但完全可以联想：

- **Adaptive filter**：每层、每步根据 gradient 谱自动选 $k_p$
- **Multi-band filter**：不只 head/tail 两段，而是多 band 不同增益（用更高阶 polynomial 或 cascade）
- **Wavelet-like**：在不同 scale 上做不同处理

### 7.4 与 LoRA / Low-Rank Adaptation 的对照

LRMuon 用 SVD/project 到 top-k 子空间，本质上等价于"先 LoRA 化 momentum 再 msign"——但 fixed rank $k$ 不能跨层、跨步自适应。Pion 用 soft high-pass 替代 hard top-k truncation，类似**soft thresholding vs. hard thresholding** 在 sparse recovery 中的对比。**Reference**: Low-Rank Muon (https://arxiv.org/abs/2509.11983), UBLR (https://arxiv.org/abs/2510.17802)。

### 7.5 与 RL 中 Credit Assignment 的连接

Pion 在 RLVR 上的成功可以 reinterpret 为：**spectral domain 的 credit assignment**。RLVR 的 credit assignment 难题是 trajectory-level reward 如何 back-propagate 到 per-token gradient——这制造了大量 noise direction。Pion 的高通假设 "informative signal 在 leading singular directions"，这其实是说"policy update 的有效自由度远小于 weight matrix 的维度"——和 LoRA、adapters、prefix tuning 的 low-rank assumption 异曲同工。**Reference**: GRPO (https://arxiv.org/abs/2402.03300), DeepSeek-R1 (https://arxiv.org/abs/2501.12948), DAPO (https://arxiv.org/abs/2503.14476)。

### 7.6 与 Attention Head Heterogeneity 的连接

Per-head norm heterogeneity 这个观察其实呼应了最近一批 work：
- Attention heads 有 functional specialization（induction heads, positional heads, ...）
- Pruning work 显示 head norms 长尾分布
- QK-norm 论文显示 head norm 影响 attention entropy

Pion 的 per-head mode 实质是"每个 head 用自己的 spectral geometry"。**Reference**: Induction Heads (Anthropic, https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html), QK-Norm (https://arxiv.org/abs/2306.17284)。

### 7.7 与 Information Theory 的连接

Effective rank 用 Shannon entropy 定义——这让人联想 sample complexity 的 effective dimension 概念。如果 gradient 的 erank 是 $r_{\mathrm{eff}}$，那么 signal 在 $r_{\mathrm{eff}}$ 维子空间里，剩下来都是 noise。Muon 的 uniform whitening 把 $n$ 维全部等权，noise/signal ratio 是 $\sqrt{(n - r_{\mathrm{eff}})/r_{\mathrm{eff}}}$——$r_{\mathrm{eff}}$ 越小，noise 比例越大。Pion 的高通把这个 ratio 推回 signal-only 的 regime。

### 7.8 Pion 与 Polynomial Iteration 设计的自由度

Newton-Schulz 用 quintic polynomial $f(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$。可以联想：
- 用更高阶 polynomial（septic, 7th degree）会有更多自由度，可能 sharper transition
- 但每步 matmul 更多——scalability tradeoff
- 不同 stage 用不同 degree 也可以

Appendix L 的 LPMuon 直接 fit 15 个 coefficient——但 Pion 用 closed-form constraint 推出 6 个 coefficient，可解释性更强，也对 hyperparameter $k_p$ 更鲁棒。

### 7.9 RLVR 中 Muon 失败的更广含义

Muon 在 RLVR 上 collapse 是个非常重要的 negative result——意味着 matrix-aware optimizer 在 post-training 不能直接 drop-in 替换 AdamW。这给所有 RLVR infrastructure 一个 caution：如果你切到 Muon，你可能在 noise 上 faster collapse。

### 7.10 与 Curriculum / Annealing 的联想

Pion 的 $k_p$ 是 single hyperparameter 控制 cutoff——可以联想 annealing schedule：
- 早期 RLVR：noise 还不严重，用大 $k_p$（接近 Muon）保留 exploration
- 后期 RLVR：policy sharpening，需要 fine adjustment，用小 $k_p$（强 high-pass）抑制 noise

类似 simulated annealing 中 temperature 的退火。

---

## 8. 一些小批评

1. **只测了 Qwen3**：没测 Llama、Mistral、DeepSeek 系列。Muon 在 Llama family 的行为可能不同。
2. **Per-head mode 只在 attention**：FFN 的两个 matrix 没有类似的"sub-block"概念——但 FFN 的 gradient 也可能 low-rank。Pion 在 FFN 上用的是 default mode，可能次优。
3. **$k_p$ 选择**：empirical 用 $k_s \geq 3$，但没有系统的 hyperparameter sweep 显示 cutoff frequency 与 erank 的关系。
4. **LPMuon 的 cutoff $\tau = 0.5$**：选 0.5 比较任意，可能 $\tau$ 调小一点 LPMuon 也不那么崩——不过作者意图是 reverse ablation，不需要 fair comparison。
5. **Real robot 只 30 trials**：方差可能大，但 85.6% vs 38.9% 的 gap 还是显著。

---

## 9. 总结

Pion 的 elegant 之处在于：通过 SVD factorization 把 matrix filter design 完全 reduced 到 scalar polynomial design；用 closed-form constraint 推出 Promotion + Suppression 两个 polynomial；保持 Muon 的 per-step cost；用 reverse ablation (LPMuon) 隔离出 high-pass 方向的关键性。

更深层的 message：**matrix-aware optimizer 的 inductive bias 不是 universal**。Uniform spectral whitening 是 pretraining 的正确 prior，但在 cross-modal / low-SNR / post-training 场景，prior 应该是 "informative signal 在 leading singular directions，tail 是 noise"。

这给了我们一个 framework：根据 gradient 的 spectral 结构（rank、SNR）选合适的 spectral filter 形状——可以是 uniform（Muon）、high-pass（Pion）、low-pass（LPMuon）、甚至 multi-band。Optimizer 设计从"element-wise adaptive"（Adam 系列）走向"spectral-shape adaptive"（Pion 系列），是一个很 exciting 的方向。

**Reference 链接汇总**：
- Paper GitHub: 见原文顶端链接
- Muon: https://kellerjordan.github.io/posts/muon/
- Muon scalable: https://arxiv.org/abs/2502.16982
- Shampoo: https://arxiv.org/abs/1802.09568
- SOAP: https://arxiv.org/abs/2409.11321
- Polar Express: https://arxiv.org/abs/2505.16932
- Low-Rank Muon: https://arxiv.org/abs/2509.11983
- UBLR: https://arxiv.org/abs/2510.17802
- LIBERO: https://arxiv.org/abs/2306.03310
- LIBERO-Plus: https://arxiv.org/abs/2510.13626
- OpenVLA: https://arxiv.org/abs/2406.09246
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- DROID: https://arxiv.org/abs/2403.12945
- GRPO / DeepSeekMath: https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- GMPO: https://arxiv.org/abs/2507.20673
- DAPO: https://arxiv.org/abs/2503.14476
- Qwen3: https://arxiv.org/abs/2505.09388
- MATH: https://arxiv.org/abs/2103.03874
- GSM8K: https://arxiv.org/abs/2110.14168
- Flow Matching: https://arxiv.org/abs/2210.02747
- Polar Decomposition: https://en.wikipedia.org/wiki/Polar_decomposition
- Matrix Sign Function (Higham): https://epubs.siam.org/doi/10.1137/1.9780898717778
- Induction Heads: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- QK-Norm: https://arxiv.org/abs/2306.17284
- Effective Rank (Roy & Vetterli): https://ieeexplore.ieee.org/document/7099090
