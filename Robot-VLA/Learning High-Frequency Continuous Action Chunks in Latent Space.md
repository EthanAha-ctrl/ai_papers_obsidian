---
source_pdf: Learning High-Frequency Continuous Action Chunks in Latent Space.pdf
paper_sha256: 5d3e8cc614b1845d70d951013d8784294c562fdf005859577e111f7aeb1c0aa0
processed_at: '2026-08-05T13:14:54-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话总结

机器人控制频率从 15 Hz 提到 60 Hz，听起来是好事——动作更丝滑、不再走走停停。但直接拿现有 policy 在 60 Hz 上训练，效果反而崩了。这篇 paper 的核心发现：**别在 action space 学高频动作，先压到 latent space 再学，执行时再用一个叫 RTR 的小技巧把相邻动作块"缝合"起来。**

---

## 为什么要高频？——走走停停问题

想象你开车。如果每秒只给你一个指令"往前开 10 米"，你会猛踩油门、猛刹车、再猛踩油门——这就是 15 Hz 机器人控制的样子。每个 action 之间隔太远，controller 把每个 action 当成"远端目标"，到了就停一下，下一个 action 来了再加速。

60 Hz 就像每秒给你 60 个指令，每个指令只让你挪 2 毫米。你几乎不需要刹车，一路滑过去。**trajectory 自然就丝滑了**，速度也不会反复掉到零。

这点 paper 里 Figure 1 画得很清楚：低频是锯齿状速度曲线，高频是平稳速度曲线。

---

## 高频的坑：jerk 会爆炸

但问题是，60 Hz 下 policy 学不动。为什么？

关键在 **jerk**——动作的三阶导数（加速度的变化率）。Paper 里给的公式：

$$
j_t = \frac{x_{t+3} - 3x_{t+2} + 3x_{t+1} - x_t}{\Delta t^3}
$$

变量含义：$x_t$ 是 t 时刻机器人末端位置，$\Delta t$ 是相邻 action 时间间隔。

人话翻译：jerk 衡量"加加速度"，对 trajectory 是否 smooth 极其敏感。机器人如果 jerk 大，会抖、会震荡、会伤硬件。

**为什么高频下 jerk 爆炸？看分母 $\Delta t^3$。**

15 Hz 时 $\Delta t \approx 0.067$ 秒，60 Hz 时 $\Delta t \approx 0.017$ 秒。$\Delta t$ 缩小 4 倍，分母 $\Delta t^3$ 缩小 64 倍。也就是说，**同样的 prediction noise，在 60 Hz 下被放大 64 倍变成 jerk**。

Paper 里 Figure 4 实测就是这个现象：把 DP、OFT、PI0.5 三个 policy 都从 15 Hz 切到 60 Hz 训练，OFT 的 jerk 从还算能看直接飙升到不可用。OFT 尤其惨，因为它用 discrete token 量化动作，高频下每个 token 的步长小，量化误差占比就大，噪声放大效应更明显。

---

## 那能不能低频训练、高频插值？

直觉思路：15 Hz 训练 policy，推理出低频 action chunk，再用插值（cubic spline 之类）补到 60 Hz 执行。

Paper 里 Table 8 直接打脸：DP 用插值，jerk 是 2.874，直接 latent 是 0.412。Exceed count（超过安全速度 120 mm/s 的步数）从 1.8 飙到 16.9。

为什么插值不行？因为插值只管 chunk 内部 smooth，但 chunk 之间的 boundary 没法保证。低频 prediction 的微小误差被插值"放大"成高频执行时的速度 violation。**插值不是 learned prior，没有 motion 的物理直觉，只是数学上的平滑。**

---

## 核心方法一：Latent Space Policy

作者的解法：**用 VAE 把高频 action chunk 压到低频 latent，policy 在 latent 里学。**

### 具体怎么做

1. 训一个 VAE：
   - 输入：$A_t \in \mathbb{R}^{48 \times 10}$（48 个 timestep × 10 维动作：xyz + rpy + gripper）
   - Encoder：2 层 1D Conv，stride=2，把 48 压到 12（downsampling ratio $f=4$）
   - Latent：$z \in \mathbb{R}^{12 \times 10}$，diagonal Gaussian
   - Decoder：2 层 MLP，把 12 还原成 48
   - KL weight $\beta = 10^{-6}$（极小，几乎是纯 autoencoder）

2. 冻结 VAE，把数据集里所有 action chunk 编码成 latent
3. Policy 还是 DP / OFT / PI0.5，但 prediction target 从 $A_t$ 换成 $z_t$
4. 推理：policy 预测 $\hat{z}$ → VAE decoder → $\hat{A}$

### 为什么这招 work？两个直觉

**直觉一：latent 是"learned low-pass filter"**

每个 latent step $z_i$ 对应 4 个原始 action step。VAE encoder 的 1D Conv stride=2 两层，相当于让 $z_i$ summarize 4 个相邻 timestep 的 dominant motion trend，而不是每一步的小 fluctuation。

Policy 的任务从"预测 48 个高频命令"变成"预测 12 个 local motion pattern"。**prediction noise 被分摊到 4 个 step 上，jerk 放大效应直接降一个数量级。**

**直觉二：VAE manifold 自带 smoothness prior**

KL 压着 latent 落在 smooth manifold 上。即使 policy 预测的 $\hat{z}$ 有小偏差，decode 出来的 $\hat{A}$ 仍会是 smooth motion——因为 VAE 训练时只见过 smooth demo，decode 出 OOD 的非 smooth 序列需要走出 manifold，概率低。

这就像 Stable Diffusion 为什么在 latent 里 diffusion 比在 pixel 里 diffusion 效果好：latent space 的 geometry 自带 image 的 structural prior。这里同理，latent 自带 motion 的 smoothness prior。

### 实测效果

Paper Table 2 的 dataset-based 对比（Write Board 任务）：

| Policy | xyz deviation | xyz jerk |
|---|---|---|
| DP original | 0.34 | 0.35 |
| DP latent | 0.26 | 0.01 |
| OFT original | 7.59 | 3.50 |
| OFT latent | 1.47 | 0.02 |
| PI0.5 original | 1.24 | 2.13 |
| PI0.5 latent | 1.32 | 0.01 |

OFT 的 jerk 从 3.50 降到 0.02，**降了 175 倍**。这就是 latent compression 的威力。

VAE reconstruction error 实测亚毫米级（Appendix C.5），所以 smoothness 不是靠丢信息换来的，是 representation 本身带来的。

---

## 核心方法二：Reuse-then-Refine (RTR)

### 问题：asynchronous inference 下 chunk 边界会断

实际部署时，policy 推理慢（DP 215ms, PI0.5 274ms），不能等推理完再执行。标准做法是 **asynchronous inference**：执行当前 chunk 的同时，提前开始推理下一个 chunk。

设置：chunk horizon = 48（0.8 秒），latency window = 24（0.4 秒）。执行到第 24 个 action 时触发新 chunk 推理，推理完时新 chunk 的前几个 action 已经过时（因为推理花了 0.4 秒，期间机器人已经在动）。

**Naive async**：丢弃过时 action，直接执行新 chunk 剩余部分。问题：新旧 chunk 在边界处不连续，机器人会 stall 甚至 rollback（位置回退）。

### RTR 怎么做

两步：

**Step 1: Reuse**
- 新 chunk 推理完成时，它前几个 action 已经过时
- 但这几个过时 action 对应的时间段，机器人实际执行的是旧 chunk 的某些 action
- 把这些"已执行的旧 action"拿来，和新 chunk 的"未过时部分"拼起来，形成一个 misaligned 的中间序列 $\tilde{A}$

**Step 2: Refine**
- 把 $\tilde{A}$ 喂进 VAE encoder → 得 $\tilde{z}$
- 再 decode → $\hat{A}_{\text{refined}}$

**为什么这能 fix discontinuity？**

因为 VAE encode-decode 是一次"manifold projection"。$\tilde{A}$ 虽然 temporally misaligned，但 VAE 会把它拉回训练时见过的 smooth motion manifold。Refined 输出在 overlap 区域自然与已执行部分对齐，boundary 处 continuity 恢复。

VAE encode-decode 只要 2.3 ms（Table 6），几乎不增加 latency。

### 为什么 RT-C 在 latent 里不 work

RT-C 是 PI0 团队提出的方法（https://arxiv.org/abs/2506.07339），思路是 inpainting：生成时强制新 chunk 的前几个 action 等于旧 chunk 的后几个 action，让 flow matching 在约束下 sample 剩余部分。

在 action space 里这招 work。但 paper Table 4 显示，**直接把 RT-C 搬到 latent space，continuity 反而变差**：

| PI0.5 method | Overlap ∆xyz | Bound ∆xyz |
|---|---|---|
| Latent alone | 1.778 | 6.842 |
| Latent + RT-C | 1.979 | 8.478 ← 更差 |
| Latent + RTR | 0.331 | 4.069 ← 好 5 倍 |

直觉：latent space 每个 dim 是 entangled representation，编码的是 4-step motion pattern。硬约束某些 dim 等于"破坏 pattern"，其他 dim 的 conditional 分布会偏离训练数据，反而恶化。

RTR 反过来：生成完再 refine，用 VAE 自己的 prior 做 projection，不干预 policy 本身。**这是 manifold projection 和 inpainting 的本质区别。**

---

## 实验结果亮点

### Synchronous（Table 1）

最显著的是 OFT 在 Peel Cucumber 任务：
- Original: 28% 成功，jerk 4.367
- Latent: 74% 成功，jerk 0.486

**成功率从 28% 到 74%，jerk 降 9 倍。** 这说明 high-frequency 下 OFT 的 discrete tokenization 真的是被 quantization error 拖死的，latent 救命。

### Asynchronous（Table 3）

PI0.5 完整对比：
- Original: 72% / 4.124 jerk
- Original + RT-C: 74% / 4.697
- Latent alone: 68% / 3.608 ← **比 original 还低**
- Latent + RTR: 80% / 1.601 ← 最佳

注意 Latent alone 在 async 下 success 比 original 还低。因为 latent 让单 chunk 内 smooth，但 chunk 边界 prediction 偏差被 VAE decoder "放大为 smooth 但 misaligned" 的 trajectory。没有 RTR 时这种 misalignment 导致 boundary gap，产生 stall。

**RTR 不是锦上添花，是 latent policy 在 async 下能用的必要条件。**

### End-to-end latency（Table 5）

| Method | Peel | Wipe | Write |
|---|---|---|---|
| Low freq original | 39.65s | 39.57s | 41.30s |
| High freq original | 20.38s | 11.68s | 17.93s |
| High freq latent+RTR | 14.59s | 9.49s | 15.11s |

**Wipe Vase 从 39.57 秒降到 9.49 秒，快 4 倍。** 主要原因：高频消除走走停停，RTR 消除 chunk 边界 stall。Smooth motion 不只是好看，直接转化为执行效率。

---

## Ablation 里的关键 intuition

### Downsampling ratio $f$ 的 sweet spot

Figure 9/10：$f$ 从 1 增到 8，deviation 降；$f=16$ 时 deviation 突然升。

直觉：$f$ 是 temporal compression 强度。适度压缩让 policy 学得更容易（target 变简单），但过度压缩让每个 latent step 影响太多 action step，policy 一旦预测错，error 被 decode 放大到更大区间。

**$f=4$（48→12）是 sweet spot**，和 Stable Diffusion 常用的 $f=8$ 量级一致。这可能反映 universal scaling：latent token 数大约是 original 的 1/4 到 1/8 最优。

### VAE vs VQ-VAE

Figure 8：continuous VAE 全面优于 VQ-VAE。

VQ 的 codebook 在高频 action 上需要极细的 code 分辨率，否则 quantization error 主导。Continuous latent 可以任意精度表征 motion。**离散化在高频下是天然劣势。**

### LIBERO 泛化（Table 14）

ACT-Latent 与 ACT 成功率持平（83.8% vs 82.3%），PI0.5-Latent 与 PI0.5 持平（90.65% vs 89.85%）。**Latent representation 不损害泛化。**

---

## 这篇 paper 的更大意义

### 1. Representation 与 frequency 必须 co-design

VLA 社区一直在搞 model scaling、data scaling，但很少人认真想 **action representation 和 control frequency 的关系**。这篇 paper 指出：直接拿 15 Hz 的方法搬到 60 Hz 会 break，representation 必须跟着 frequency 一起设计。

### 2. Latent action 是 robotic 的"Stable Diffusion 时刻"

Stable Diffusion 之前，diffusion 在 pixel space 算，贵且效果一般。压到 latent 后，质量提升、计算降低。这篇 paper 在 robotic action 上做了类似的事：**high-frequency action learning 的关键瓶颈是 representation，不是 model size。**

### 3. Manifold projection > inpainting for continuity

RT-C 的 inpainting 思路在 action space work，但在 latent space 失效。RTR 用 VAE 自己的 prior 做 projection，反而更优雅。这个 insight 对未来 latent policy 的 execution 策略有指导意义。

### 4. Smoothness 直接转化为 latency 优势

很多人以为 async inference 只是隐藏 latency，但 paper 显示 chunk 边界 discontinuity 才是 success killer。RTR 把 boundary gap 从 6.842 降到 4.069，直接让 Wipe Vase 快 4 倍。**Smoothness 不是 aesthetic，是 efficiency。**

---

## 如果让我 push 这工作

### Hierarchical latent

当前 latent 是单层 48→12。如果做 hierarchical：macro latent（48→4，覆盖 0.8s 的 long-horizon intent）+ micro latent（48→12，覆盖 0.8s 的 smooth motion），policy 同时学 planning 和 control。这可能让 long-horizon task 表现更好。

### RTR + force feedback

RTR 现在是 open-loop refine，没用 environment feedback。Peel Cucumber 这种 contact-rich 任务，接触瞬间 force/torque 突变，VAE manifold projection 可能 fix 不了。如果结合 RDP（https://arxiv.org/abs/2503.02881）的 tactile closed-loop，可能能处理 contact transition。

### Jerk-aware training

现在 VAE loss 是 reconstruction + KL。如果直接加 jerk penalty on decoded action，让 VAE 学到的 manifold 专门 favor low-jerk motion，可能比纯 reconstruction 更适合 high-frequency control。但风险是 over-smoothing 丢失 task-critical 快速变化。

### 更高频率：120 Hz / 200 Hz

Paper 承认 60 Hz 是 sensor 限制。120 Hz 下 jerk 放大因子再翻 8 倍，$f=4$ 可能不够，可能需要 $f=8$ 或 multi-scale VAE。

### RTR 的理论分析

现在 RTR 是 empirical work。如果用 score-based generative model 的视角，VAE encode-decode 近似于一步 Langevin dynamics 的 denoising step。是否可以证明 RTR 的 convergence 和 continuity guarantee？这会让方法更有理论基础。

---

## 一句话给你的 take-away

**这篇 paper 抓住了一个被 VLA 社区忽略的细节——高频控制下，action representation 必须重新设计。用 VAE 压 latent 解决学不动的問題，用 RTR 解决 chunk 边界断开的问题，两者配合让 60 Hz 真正可用。核心数学直觉是 jerk 的 $\Delta t^{-3}$ 放大效应，核心方法直觉是 learned low-pass filter + manifold projection。**

参考链接：
- Paper code: https://github.com/tars-robotics/RTR
- RT-C: https://arxiv.org/abs/2506.07339
- Stable Diffusion (LDM): https://arxiv.org/abs/2112.10752
- VAE: https://arxiv.org/abs/1312.6114
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- PI0.5: https://arxiv.org/abs/2504.16054
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- RDP: https://arxiv.org/abs/2503.02881

---

# Paper 解读：Learning High-Frequency Continuous Action Chunks in Latent Space

## 1. 总览与核心 insight

这篇 paper 来自 Shanghai Jiao Tong University TARS 团队，针对 robotic imitation learning 中的一个被忽略但实际影响巨大的问题：**当 action frequency 从常规的 15 Hz 提升到 60 Hz 时，action chunking 的 representation learning 会 break down**。作者给出两条耦合的贡献：

1. **Latent-space policy**：用 VAE 把 high-frequency action chunk 压缩到 temporally downsampled 的 continuous latent space，policy 在 latent 中预测，再 decode 回 high-frequency action。
2. **Reuse-then-Refine (RTR)**：一个 training-free 的执行时策略，专门处理 asynchronous inference 下相邻 chunk 之间的 discontinuity，且专为 latent space 设计（与 RT-C 不同，RTR 利用已有 VAE 做 refine，无需额外训练）。

代码：https://github.com/tars-robotics/RTR
ArXiv 上下文相关的参考 paper：
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- PI0.5: https://arxiv.org/abs/2504.16054
- RT-C (Real-time execution of action chunking flow policies): https://arxiv.org/abs/2506.07339
- SmolVLA: https://arxiv.org/abs/2506.01844
- VQ-VLA: https://arxiv.org/abs/2507.01016
- LatentVLA: https://arxiv.org/abs/2601.05611
- VAE (Kingma & Welling): https://arxiv.org/abs/1312.6114
- VQ-VAE: https://arxiv.org/abs/1711.00937

---

## 2. Motivation：为什么 high frequency 既 desirable 又难学

### 2.1 Low-frequency 会导致 stop-and-go

Figure 1 给出直观对比：
- **15 Hz**：相邻 action 之间 spatial step 大（mm 量级 → cm 量级），controller 把每个 action 视为远端 target，等价于 implicit zero-velocity boundary，每个 chunk 边界产生 deceleration→停顿→acceleration 循环。
- **60 Hz**：spatial step 仅 ~2mm，controller 跨 action 维持 nonzero velocity，避免反复加减速。

物理上：当 control frequency 与 action frequency 匹配、且 action frequency 足够高时，trajectory 在时间上"接得住"速度信息。

### 2.2 High frequency 在 action space 中难学

作者训练 DP、OFT、PI0.5 三种代表性 policy 在 60 Hz 和 15 Hz 两套数据上，并在原始 60 Hz trajectory 上评估，得到 Figure 4 的关键现象：

- **DP** 在 Cartesian space 上还勉强扛得住（diffusion 是 continuous generation，stride 误差不放大）
- **OFT** 在 60 Hz 下 jerk 暴涨：原因是 OFT 用 discrete action tokenization，stride 越小 quantization error 相对占比越大
- **PI0.5** 也出现 jerk 增加，因为 flow matching 在细粒度 target 上误差累积

这里有一个非常微妙的点值得 build intuition：

> **Jerk 是 trajectory 的三阶差分，对 prediction error 具有"高通滤波"放大效应。**

下面这个公式是关键：

$$
\mathbf{j}_t = \frac{\mathbf{x}_{t+3} - 3\mathbf{x}_{t+2} + 3\mathbf{x}_{t+1} - \mathbf{x}_t}{\Delta t^3}
$$

变量含义：
- $\mathbf{x}_t$：t 时刻的 end-effector pose（位置或姿态）
- $\Delta t$：相邻 action 的时间间隔，60 Hz 下 $\Delta t = 1/60 \approx 16.7\,\text{ms}$；15 Hz 下 $\Delta t = 1/15 \approx 66.7\,\text{ms}$
- 系数 $1, -3, 3, -1$ 是二项式系数 $(1-1)^3$ 展开前三项，对应三阶后向差分

**关键直觉**：分母是 $\Delta t^3$。如果 policy 在每个 timestep 上引入一个独立的高频 prediction noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$，那么 jerk 期望幅度近似为 $\sqrt{1+9+9+1}\,\sigma / \Delta t^3 = \sqrt{20}\,\sigma / \Delta t^3$。从 15 Hz 升到 60 Hz，$\Delta t$ 缩小 4 倍，jerk 放大 $4^3 = 64$ 倍！这是 high frequency 下 jerk 爆炸的数学根源。

同样对 acceleration (Eq.1)：
$$
\mathbf{a}_t = \frac{\mathbf{x}_{t+2} - 2\mathbf{x}_{t+1} + \mathbf{x}_t}{\Delta t^2}
$$
- 二阶差分，对应二阶导数
- noise 放大因子 $\Delta t^{-2}$，从 15 到 60 Hz 放大 16 倍

这就是为什么 high frequency 下"做平滑"在数学上极其困难——任何 per-timestep prediction error 在 jerk 上都会被 cubic 放大。

### 2.3 Interpolation 不行

一个 naive 思路：低频训练 + 时间插值到高频。Table 8 给出 DP 上对比：
- Latent (60Hz)：jerk 0.412, exceed 1.8, latency 15.83s
- Interpolate (15Hz→60Hz)：jerk 2.874, exceed 16.9, latency 22.96s

为什么 interpolate 失败？因为低频 prediction 误差在 cubic spline 等插值器下并不能产生 physically-consistent 高频 motion；插值在 chunk 内部 smooth，但 chunk 之间的"放大速度"会违反 actuator limit。Exceed count（每步位移超过 2mm 即 120 mm/s @ 60 Hz 的步数）从 1.8 飙到 16.9 就直接量化了这个 violation。

---

## 3. Method 第一部分：Latent-Space Policy Learning

### 3.1 VAE 架构细节

VAE 设计选择（Table 7）：
- **Input**：$A_t \in \mathbb{R}^{H \times c}$，其中 $H = 48$（60 Hz 下覆盖 0.8s），$c = 10$（xyz + rpy + gripper width = 7 + 1 + ... 实际看应该是 3+3+1+3 之类，paper 中说 c=10）
- **Encoder**：1D Conv，2 层，kernel size 5，stride 2，hidden channel 32
- **Temporal compression ratio**：$f = 4$，即 48 → 12
- **Latent**：$z \in \mathbb{R}^{h \times d}$，$h = 12$，$d = 10$（diagonal Gaussian，mean + log-var 参数化）
- **Decoder**：2 层 MLP
- **KL weight $\beta$**：$1 \times 10^{-6}$（极小，几乎是 pure autoencoder，只保留一点 manifold 平滑约束）

这里有几个值得 build intuition 的设计：

**(a) 为什么用 1D Conv 而不是 Transformer？**
- Action chunk 是时间序列，local temporal pattern 比长程依赖更重要
- Conv 的参数效率高，且天然带 temporal inductive bias
- 2 层 stride-2 conv 等价于 factor-4 downsampling，无重叠 receptive field

**(b) 为什么 decoder 用 MLP 而不是 Conv？**
- Decoder 输出要重建每个 timestep 的 action，MLP 在 latent $z \in \mathbb{R}^{12 \times 10}$ 上 broadcast 到 $\hat{A}_t \in \mathbb{R}^{48 \times 10}$，相当于每个 latent step 直接映射到对应 4 个 action step
- Conv decoder 会引入跨 step 耦合，可能反而损害精度
- 这里 MLP 起到 "per-step upsample" 的角色，类似 nearest-neighbor + nonlinear projection

**(c) 为什么 $\beta$ 那么小？**
- KL 太大会把 latent 压成 isotropic Gaussian，丢失 motion 细节
- $\beta = 10^{-6}$ 几乎是 deterministic autoencoder，但保留极弱的 posterior regularization，避免 latent manifold 出现 sharp discontinuity
- 这与 $\beta$-VAE 中的 trade-off 一致，但这里重点在 reconstruction fidelity，KL 只作为 manifold smoothness 的"轻微润滑剂"

### 3.2 Latent 的物理意义

paper 在 4.1 节末尾给出一个很有 insight 的解读：

> 每个 latent step $z_i$ ($i=1,\dots,12$) summarizes the dominant motion trend over 4 neighboring timesteps，而非每一步的微小 fluctuation。

这相当于 **learning-based low-pass filter**。policy 的 prediction target 从"高频命令级 variation"切换为"局部 motion structure"。同时 KL 把这些 motion pattern 压在 smooth manifold 上，让 policy function approximation 更容易拟合。

Reconstruction error 实测（Appendix C.5）：
- Peel Cucumber: $\Delta x = 0.37$ mm, $\Delta y = 0.11$ mm, $\Delta z = 0.17$ mm
- Wipe Vase: $\Delta x = 0.38$ mm, $\Delta y = 0.17$ mm, $\Delta z = 0.24$ mm
- Write Board: $\Delta x = 0.50$ mm, $\Delta y = 0.28$ mm, $\Delta z = 0.12$ mm

亚毫米级，说明 VAE 信息损失几乎可忽略，**smoothness 来自 representation 本身，而不来自压缩信息丢弃**。

### 3.3 Policy 训练流程

1. 独立训练 VAE，冻结
2. 用 VAE encoder 把 dataset 中每个 $A_t$ 编码为 $z_t$，得到 latent dataset $\{(o_t, z_t)\}$
3. 在 latent dataset 上训练 policy（与原 policy 完全相同的 hyperparameter、网络结构、训练步数）
4. 推理：policy 预测 $\hat{z}$ → VAE decoder → $\hat{A}$

**Latency 实测**（Table 6）：
- VAE encode/decode: 2.30 ms（可忽略）
- DP original: 215.72 ms, DP latent: 216.64 ms（VAE 加约 1 ms）
- OFT original: 154.38 ms, OFT latent: 123.99 ms（latent 反而更快，因为 token 数变少）
- PI0.5 original: 274.51 ms, PI0.5 latent: 271.43 ms

OF T 在 latent 下变快很有意思：原 OFT 在 48 step 上做 token prediction，latent 下变成 12 step token prediction，infer cost 直接降 4 倍。

---

## 4. Method 第二部分：Reuse-then-Refine (RTR)

### 4.1 Asynchronous inference 的问题

设置（Appendix B.5）：
- prediction horizon $H = 48$ @ 60 Hz = 0.8s
- latency window = 24 actions = 0.4s
- 执行 24 个 action 后触发新 chunk inference，期间继续执行剩余 24 个 action
- inference 完成时，新 chunk 中前 2 个 action 已经过时（outdated）

如图 5(a) Naive asynchronous：丢弃 outdated，直接执行剩余，会出现 chunk 边界的 spatial gap，导致 visible stalls 甚至 rollback（机器人位置回退）。

### 4.2 RTR 算法

RTR 流程（图 5(b)）：

1. **Reuse 阶段**：取已执行的、与新 chunk 推理窗口重叠的 actions（如图中 t+3, t+4 这两个），与新生成 chunk 的非 outdated 部分（t+5, ..., t+11）拼接，形成 temporally misaligned 的中间 chunk $\tilde{A}$
2. **Refine 阶段**：把 $\tilde{A}$ 喂入 VAE encoder 得 $\tilde{z} = \mathcal{E}(\tilde{A})$，再 decode 得 $\hat{A}_{\text{refined}} = \mathcal{D}(\tilde{z})$

为什么这样做能 fix discontinuity？关键 insight：
- **VAE encode-decode 等价于一次 projection 到 learned manifold 的操作**
- 输入 $\tilde{A}$ 即使 temporally misaligned，VAE 会把它"拉回"到训练时见过的 smooth motion manifold
- 因为 reused actions 本身来自上一 chunk 已执行部分，refine 后的输出在 overlap 区域与上一 chunk 几乎重合，从而保证 boundary continuity

这与 RT-C 的根本差异在于：
- **RT-C** 是 inpainting（conditioning on previous chunk during generation），适用于 flow matching / diffusion，需要重新训练或修改 sampling 过程
- **RTR** 是 post-hoc refine，无需 retrain policy，复用已训练好的 VAE，2ms 开销

### 4.3 为什么 RT-C 在 latent space 失效？

Table 4 给出关键对比（PI0.5）：
| Method | Overlap ∆xyz ↓ | Bound ∆xyz ↓ |
|---|---|---|
| Original | 1.575 | 5.636 |
| Original + RT-C | 1.242 | 4.640 |
| Latent | 1.778 | 6.842 |
| **Latent + RT-C** | **1.979** | **8.478** ← 比 Latent alone 还差！ |
| **Latent + RTR** | **0.331** | **4.069** |

直觉解释：
- Latent space 中 inpainting（fix 部分 latent dim，sample 剩余）会破坏 latent VAE 学到的 marginal 分布
- latent 每个 dim 编码的是 4-step motion pattern，硬约束某些 dim 会让其他 dim 的 conditional 分布偏离训练数据
- 结果 RT-C + Latent 反而把 boundary gap 拉大

RTR 反过来：不在 generation 时强制 inpainting，而是 generation 完成后再 project 回 manifold，避免对 policy 本身的干预。

### 4.4 RTR 的 OOD 鲁棒性

Appendix C.6 专门讨论 RTR 输入是否 OOD：
- Reused 部分与新 chunk prefix 对应同一 timesteps 的同一 target motion，所以 misalignment 有限
- Open-loop 测试 DP-Latent：no RTR $\Delta x = 0.48$ mm → RTR $\Delta x = 0.62$ mm
- 误差增加约 0.14 mm，仍在 sub-millimeter 范围
- 闭环执行成功率不降反升（Table 3），说明 continuity 改善抵消了精度微小损失

---

## 5. 实验：三个 contact-rich 任务 + 三种 policy

### 5.1 Setup

- **Robot**: UFACTORY xArm 7 + Robotiq 2F-85 gripper
- **GPU**: RTX 3060 数据采集，RTX 4090 推理
- **Tasks**:
  - Peel Cucumber（用 peeler 削皮）
  - Wipe Vase（用 eraser 擦花瓶污渍）
  - Write Board（白板上画直线）
- **Trials**: 每方法每任务 50 次
- **Success criteria**:
  - Peel: >50% 黄瓜皮被削下
  - Wipe: 污渍擦净
  - Write: 完整直线
- **Safe speed threshold**: 120 mm/s（@ 60 Hz 即 2 mm/step）

### 5.2 Synchronous inference（Table 1）

| Policy | Method | Peel Succ/Jerk/Exc | Wipe Succ/Jerk/Exc | Write Succ/Jerk/Exc |
|---|---|---|---|---|
| DP | Original | 90% / 2.057 / 4.0 | 100% / 1.433 / 5.1 | 100% / 1.140 / 1.6 |
| DP | Latent | 90% / 0.412 / 1.8 | 100% / 0.645 / 2.0 | 100% / 0.511 / 0.8 |
| OFT | Original | 28% / 4.367 / 32.7 | 94% / 3.131 / 12.3 | 74% / 5.238 / 50.5 |
| OFT | Latent | 74% / 0.486 / 3.1 | 100% / 1.055 / 3.0 | 100% / 0.558 / 2.2 |
| PI0.5 | Original | 78% / 2.790 / 9.0 | 100% / 2.661 / 4.7 | 100% / 2.509 / 5.3 |
| PI0.5 | Latent | 84% / 0.678 / 2.5 | 100% / 0.697 / 2.3 | 100% / 0.673 / 2.6 |

**关键观察**：
- **OFT 提升最大**：Peel 成功率 28%→74%，jerk 4.367→0.486，这是数量级的提升。验证 paper 核心论点：discrete tokenization 在 high frequency 下 break 最严重，latent 补救最显著。
- **DP 提升较小但 jerk 大幅下降**：DP 本身在 action space 已较稳，但 latent 把 jerk 从 2.057 降到 0.412，对 contact-rich 任务安全性提升明显。
- **PI0.5 中等提升**：Peel 78%→84%，jerk 2.790→0.678。

### 5.3 Asynchronous inference（Table 3）

PI0.5 完整对比（最能体现 RTR 价值）：
- Original: 72% / 4.124 jerk / 21.0 exceed
- Original + RT-C: 74% / 4.697 / 18.5
- Latent alone: 68% / 3.608 / 15.2 ← 注意 success 反而降了！
- **Latent + RTR**: 80% / 1.601 / 10.3 ← 全面最佳

**这里有个非常重要的现象**：Latent alone 在异步下 success 反而比 Original 还低（68% vs 72%）！为什么？

直觉：latent 让单 chunk 内 smooth，但 latent policy 在 chunk 边界的 prediction 一旦有偏差，VAE decoder 会把这个偏差"放大为 smooth 但 misaligned"的 trajectory。没有 RTR 时这种 misalignment 导致 boundary gap，asynchronous 执行产生 stall/rollback。

加 RTR 后，refine 把 misalignment "拉回" executed trajectory，恢复 continuity，success 反超 Original + RT-C。

### 5.4 End-to-end latency（Table 5）

| Freq | Method | Peel | Wipe | Write |
|---|---|---|---|---|
| Low | Original | 39.65 | 39.57 | 41.30 |
| High | Original | 20.38 | 11.68 | 17.93 |
| High | Latent | 18.07 | 11.66 | 19.09 |
| High | **Latent + RTR** | **14.59** | **9.49** | **15.11** |

**Latent + RTR 在 Wipe Vase 上比 Low freq 快 4 倍！** 主要原因：
- High freq 消除 stop-and-go（每步不需重新加速）
- RTR 消除 chunk 边界 stall，进一步压缩 stall 时间
- 这证明 smooth motion 不只是"好看"，直接转化为执行效率

### 5.5 Ablation：downsampling ratio (Figure 9, 10)

PI0.5 Latent 不同 $f$：
- $f=1$：no compression，deviation 较高
- $f=2, 4, 8$：deviation 递减（f=8 最优）
- $f=16$：deviation 突然上升 ← over-compression 丢失信息

OFT Latent：
- 转折点在 $f=2$ ← 量化误差让 OFT 容忍压缩能力差

Jerk：随 $f$ 单调上升，因为相邻 reconstructed action 之间 temporal gap 增大，三阶差分放大。

**这个 trade-off 可以这样理解**：temporal compression 是 low-pass filter，但 filter 过强会让 latent 对 prediction error 敏感（因为每个 latent step 影响 16 个 action step），所以 f 太大时反而恶化。$f = 4$（48→12）是 sweet spot。

### 5.6 LIBERO 泛化性（Table 14）

ACT-Latent 与 ACT 相当（83.8% vs 82.3% average）；PI0.5-Latent 与 PI0.5 相当（90.65% vs 89.85%）。证明 latent representation 没有损害 generalization。

### 5.7 VAE vs VQ-VAE (Figure 8)

Continuous VAE 全面优于 VQ-VAE。直觉：VQ 的 codebook 在 high-frequency action 上需要更细的 code 分辨率，否则 quantization error 主导。Continuous latent 可以以任意精度表征 motion。

---

## 6. Intuition Summary（构建心智模型）

### 6.1 三阶差分放大效应

整篇 paper 的核心数学动机是公式 (2) 的 jerk。在时间维度上，high frequency 等于 high-pass，对 prediction noise 极度敏感。Latent compression 起到 temporal low-pass 的作用，把高频 noise "吸收"到 latent manifold 中。这相当于在 representation 层面引入了一个 **learned smoothing prior**。

### 6.2 Latent = learned motion primitives

每个 latent step $z_i \in \mathbb{R}^{10}$ 不是单纯降维，而是"4 个相邻 action 的 motion summary"。policy 学到的是 "在当前 observation 下，下一步局部 motion 是哪种模式"。这与 LAPA / Moto (https://arxiv.org/abs/2412.04445) 的 latent action pretraining 思想类似，但本文是 chunk-level 而非 step-level。

### 6.3 RTR = manifold projection

把 outdated + new chunk concat 后过 VAE encode-decode，等价于把 trajectory 投影回 VAE 学到的 motion manifold。manifold 的性质是：训练分布内的输入会被"修整"为最接近的合法 smooth motion。这与 image diffusion 中的 DDIM inversion + denoise 异曲同工，但用了 VAE 而非 diffusion（效率高得多）。

### 6.4 RT-C 在 latent 失效的本质

RT-C 是 conditioning-based inpainting，假设 generation process 可以被 partial observation 约束。这在 action space 中成立（因为 action 是显式数值），但 latent space 的每个 dim 是 entangled representation，强制约束部分 dim 会破坏 VAE 学到的 latent distribution 结构。RTR 反过来用 VAE 自己的 prior 做 refine，自然得多。

### 6.5 与 image/video latent diffusion 的联系

paper 在 2.3 节明确提到 inspiration 来自 Stable Diffusion (Rombach et al., 2022, https://arxiv.org/abs/2112.10752) 与 Stable Video Diffusion (https://arxiv.org/abs/2311.15127)。Latent compression 在视觉生成中已成为 standard practice，这篇 paper 把这个思想移植到 robotic action learning，并且发现 downsampling ratio $f=4$ 与 Stable Diffusion 的常用 $f=8$ 量级一致——可能反映了一个 universal scaling law：latent token 数大约是 original 的 1/4 到 1/8 时最优。

---

## 7. Critical analysis（值得 push back 的点）

### 7.1 Contact-rich 任务的局限

三个任务都是 quasi-static contact（peel、wipe、write），都是 end-effector 与环境保持持续接触，slow motion。是否适用于 dynamic task（如抛接、打击）？paper 没回答。dynamic 任务中 velocity 本身是 task 一部分，latent compression 可能直接丢掉 task-critical 速度信息。

### 7.2 60 Hz 上限

作者在 Appendix A 承认 60 Hz 受限于 sensor sampling rate。120 Hz、200 Hz 下 jerk 放大因子会再翻 8-30 倍，latent 是否还够用？我相信需要更强 representation（hierarchical latent 或 multi-scale VAE）。

### 7.3 VAE 的 reconstruction 极限

sub-millimeter reconstruction 看起来很好，但这是因为 action space 的局部 linear 性。如果 action 中包含突然变向（如 peg-in-hole 中的 contact transition），VAE 这种 L2 reconstruction loss 是否能保持细节？这值得专门 ablation。

### 7.4 RTR 与 receding horizon 的关系

RTR 其实是 MPC 中"hot start"思想的变种：用上一次解作为新解的初值。但 RTR 是 open-loop refine，没用 environment feedback。如果加入 force/torque feedback（如 RDP: https://arxiv.org/abs/2503.02881），可能能进一步 fix 接触瞬间的 deviation。

### 7.5 KL weight 太小

$\beta = 10^{-6}$ 几乎等于 autoencoder。这意味着 latent manifold 可能并不严格 Gaussian，所以 RTR 做 manifold projection 时，"投影"可能更接近 nearest-neighbor 而非真正的 posterior projection。如果换用 normalizing flow 或 diffusion-based latent prior，RTR 可能更鲁棒。

### 7.6 与 ACT 的关系

ACT (Zhao et al., 2023, https://tonyzhaozh.github.io/aloha/) 本身就是 CVAE chunking policy，与本文的"先训 VAE 再训 policy"思路接近。paper 在 LIBERO 上做了 ACT-Latent 对比，但没有 ACT 在 real-world 上的 high-frequency 对比。如果 ACT 本身就是 CVAE，是否还存在 "加 latent VAE" 的边际收益？这是个有意思的问题。

### 7.7 RTR 推理成本

VAE encode-decode 2 ms 看似可忽略，但 RTR 需要每 chunk 切换时跑一次，60 Hz @ H=48, window=24 意味着每 24 step 跑一次，即 0.4s 间隔，相对 200 ms 的 PI0.5 推理也算可观。如果未来降到 mobile robot 上推理，这 2 ms 累积起来不可忽略。

---

## 8. 与同期工作的关系图谱

| Work | 核心 idea | 与本文关系 |
|---|---|---|
| DP (Chi 2025) | action chunking via diffusion | baseline + 加 latent 提升 jerk |
| ACT (Zhao 2023) | CVAE chunking | 思路相近，本文 chunk 更细 + 单独 VAE |
| OpenVLA-OFT (Kim 2025) | discrete action tokenization | baseline，latent 救 quantization |
| PI0 / PI0.5 (Black 2024, 2025) | flow matching VLA | baseline，chunk + flow |
| RT-C (Black 2025) | async + inpainting | 主要对比对象，latent 下失效 |
| SmolVLA (Shukor 2025) | async 切换 | 无 continuity 处理，naive baseline |
| VLASH (Tang 2025) | future-state aware async | concurrent work |
| RDP (Xue 2025) | tactile closed-loop + latent tokenizer | latent action 思想相近 |
| LAPA (Ye 2024) | latent action pretraining from video | latent action 来源不同（video 而非演示） |
| VQ-VLA (Wang 2025) | vector-quantized action tokenizer | ablation 对比，本文证明 continuous 更优 |
| LatentVLA (Xie 2026) | latent for autonomous driving | 概念类似，不同 domain |
| Stable Diffusion (Rombach 2022) | image latent diffusion | 主要 inspiration |

---

## 9. 给你的 take-away（Karpathy 视角）

如果我来看这篇 paper，三个最值得记住的 intuition：

1. **三阶差分是 high-frequency policy 的真正瓶颈**：jerk 的 $\Delta t^{-3}$ 放大因子让 per-step error 在 evaluation metric 上 cubic 增长。这是为什么 naive "train at high freq" 不 work。任何 representation design 都要先问"是否降低 effective derivative order"。

2. **Latent compression = learned low-pass filter**：与其在 policy loss 上加 smoothness regularization（容易被 prediction error 抵消），不如直接换 representation，让 "smoothness" 在 representation 本身。这种 "把约束 baked into representation" 的思路与 ImageNet pretraining、contrastive learning 一脉相承。

3. **Asynchronous chunk switching 的真实瓶颈是 boundary，不是 inference latency**：很多人以为 async 推理只是隐藏 latency，但实际部署中 chunk 边界的 discontinuity 才是 success killer。RTR 用 manifold projection 解决这个，比 inpainting 优雅得多，且 training-free。这与 diffusion sampling 中的 DDIM inversion 思想非常类似。

如果让我把这个工作 push 一步，我会想知道：
- latent chunk 是否可以做 hierarchical（macro latent + micro latent），让 policy 同时学 long-horizon planning 与 short-horizon smoothness？
- RTR 是否可以换成 conditional diffusion refine，让 refine 不是 VAE manifold 而是真正的 motion prior？
- 三阶差分敏感度是否可以转化为直接 training loss（minimize jerk in latent decode）？

整体来说，这篇 paper 抓住了一个长期被 VLA 社区忽略的细节——**frequency 与 representation 必须 co-design**——并给出了一个简洁有效、engineering-friendly 的方案。值得在自己的 robot stack 上 reproduce。

---

参考 web 链接：
- Paper code: https://github.com/tars-robotics/RTR
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- PI0.5: https://arxiv.org/abs/2504.16054
- RT-C: https://arxiv.org/abs/2506.07339
- SmolVLA: https://arxiv.org/abs/2506.01844
- VLASH: https://arxiv.org/abs/2512.01031
- RDP: https://arxiv.org/abs/2503.02881
- LAPA: https://arxiv.org/abs/2410.11758
- VQ-VLA: https://arxiv.org/abs/2507.01016
- LatentVLA: https://arxiv.org/abs/2601.05611
- ACT / ALOHA: https://tonyzhaozh.github.io/aloha/
- VAE: https://arxiv.org/abs/1312.6114
- VQ-VAE: https://arxiv.org/abs/1711.00937
- Stable Diffusion (LDM): https://arxiv.org/abs/2112.10752
- Stable Video Diffusion: https://arxiv.org/abs/2311.15127
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- DROID dataset: https://arxiv.org/abs/2403.12945
- Open X-Embodiment: https://robotics-transformer-x.github.io/
