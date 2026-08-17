---
source_pdf: ONE-STEP DIFFUSION POLICY.pdf
paper_sha256: 8de440c97f946a1080ada27849728358304776cdbea306cf2e4a292c0926f858
processed_at: '2026-08-05T23:50:41-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 OneDP

## 先说它要解决什么麻烦

Diffusion Policy（DP）在机器人模仿学习里效果好，但**慢**。慢到什么程度？生成一个 action chunk 要在同一个 U-Net 里来回跑 100 次（DDPM）或 10 次（DDIM）。在 Franka 上实测一次决策要 66~660ms，policy frequency 只有 1.5~13 Hz。

这个速度在静态 pick-and-place 勉强够用，但只要环境稍微动态一点就崩——比如人突然把抓取目标挪走，机械臂还在用 0.5 秒前的观测算 action，gripper 自然抓空。论文里 pnp-milk-move 任务就是这个场景，原 DP 成功率只有 80%。

**根本症结**：diffusion 生成时必须沿着 reverse chain 一步步走，因为 $\epsilon_\theta$ 只在带噪样本（训练分布）附近才准。你想一步从纯噪声 $\mathbf{A}^K$ 直接跳到 $\mathbf{A}^0$，会落到网络没见过的 OOD 区域，Table 1 里 DP 1-step 全 0% 就是这么来的。

## OneDP 的招数

一句话：**别让 student 也去"去噪"，直接让它学一个从噪声到动作的单步映射，只要输出分布跟 teacher 像就行**。

具体讲，定义一个 generator $G_\theta$：

$$z \sim \mathcal{N}(0, I), \quad \mathbf{A}_\theta = G_\theta(z, \mathbf{O})$$

- $z$：高斯噪声向量，输入端"种子"
- $\mathbf{O}$：视觉观测（ResNet18 编的 feature）
- $\mathbf{A}_\theta$：一步出来的 action chunk（未来 16 步动作）

这个 $G_\theta$ 不再是 denoising network，就是个普通的 conditional generator。问题变成：怎么训 $\theta$ 让 $G_\theta$ 诱导的分布 $p_{G_\theta}$ 跟原 DP 的分布 $p_{\pi_\phi}$ 对齐？

## 用 reverse KL 衡量"像不像"

作者选 reverse KL：

$$\mathcal{D}_{KL}(p_{G_\theta} \| p_{\pi_\phi}) = \mathbb{E}\left[\log p_{G_\theta}(\mathbf{A}_\theta|\mathbf{O}) - \log p_{\pi_\phi}(\mathbf{A}_\theta|\mathbf{O})\right]$$

- $p_{G_\theta}$: generator 输出分布
- $p_{\pi_\phi}$: teacher（预训练 DP）输出分布
- $\mathbf{A}_\theta = G_\theta(z, \mathbf{O})$: 从 generator 采的样本

reverse KL 是 **mode-seeking**：它会惩罚 generator 跑到 teacher 低概率区，但不会惩罚 teacher 有而 generator 没覆盖的 mode。这正好适合机器人——示范数据里同一观测常有多模态动作（比如杯子可以从左边抓也可以从右边抓），forward KL 会逼 student 把两个 mode 平均成一个无意义的中间动作，reverse KL 不会。

这跟 [Diffusion-QL](https://arxiv.org/abs/2208.06193) 在 offline RL 用 reverse KL 的逻辑一脉相承。

## 关键 trick 1：在加噪样本上算，不在 clean 上算

直接优化上面 KL 有两个死结：

1. **score 爆炸**：当 generator 输出 $\mathbf{A}_\theta$ 落在 teacher 低概率区时，$\log p_{\pi_\phi}(\mathbf{A}_\theta) \to -\infty$，梯度炸。SDS 也这毛病，所以 3D 生成容易过饱和。
2. **diffusion 只会算带噪样本的 score**：clean sample 的 score 它没训过。

作者的解法（思路来自 [Diffusion-GAN](https://diffusion-gan.github.io/)）：**先把 generator 的输出加一次噪声再算 KL**。

$$\mathbf{A}_\theta^k = \alpha_k \mathbf{A}_\theta + \sigma_k \epsilon_k, \quad \epsilon_k \sim \mathcal{N}(0, I)$$

- $\alpha_k, \sigma_k$: 第 $k$ 步的噪声调度系数
- $\epsilon_k$: 这次实际采的高斯噪声
- $k \sim \mathcal{U}$: 每次迭代随机采一个 diffusion timestep

然后对加噪后的样本算 score 差：

$$\nabla_\theta \mathcal{L} = \mathbb{E}\left[ w(k) \big( s_{p_{G_\theta}}(\mathbf{A}_\theta^k) - s_{p_{\pi_\phi}}(\mathbf{A}_\theta^k) \big) \nabla_\theta \mathbf{A}_\theta^k \right]$$

- $s_{p_{G_\theta}}(\mathbf{A}_\theta^k) = \nabla_{\mathbf{A}_\theta^k} \log p_{G_\theta}$: generator 分布在加噪样本处的 score
- $s_{p_{\pi_\phi}}(\mathbf{A}_\theta^k)$: teacher 分布在加噪样本处的 score，这个直接用预训练 $\epsilon_\phi$ 算：$s = -\epsilon_\phi/\sigma_k$
- $w(k) = \sigma_k^2$: 重加权，跟 DreamFusion 一样

加噪之后样本落在 teacher 训练过的分布里，score 估得准；$k$ 随机采相当于在整条 diffusion chain 上都做分布对齐，监督信号密度比 trajectory 一致性强很多。

## 关键 trick 2：怎么估 generator 自己的 score

这里分化出两个版本。

### OneDP-S（stochastic）：训一个辅助 score network

$s_{p_{G_\theta}}(\mathbf{A}_\theta^k)$ 没法直接算（$p_{G_\theta}$ 是 implicit 分布）。作者引入第二个网络 $\pi_\psi$，用标准 denoising loss 训，但训练数据是 generator 的输出：

$$\min_\psi \mathbb{E}\left[ \lambda(k) \| \epsilon_\psi(\mathbf{x}^k, k) - \epsilon_k \|^2 \right]$$

- $\epsilon_\psi$: 要训的 score network 参数
- $\mathbf{x}^0 = \text{stop-grad}(G_\theta(z))$: generator 输出，detach 掉不反传
- $\mathbf{x}^k = \alpha_k \mathbf{x}^0 + \sigma_k \epsilon_k$: 对 generator 输出加噪
- $\epsilon_k$: 实际加的噪声，作为监督标签
- $\lambda(k)$: EDM 里的加权函数

这就是 [VSD (ProlificDreamer)](https://ml.cs.tsinghua.edu.cn/prolicdreamer/) 的做法。$\pi_\psi$ 学会 generator 分布的 score，然后回代到主 loss 里替换 $s_{p_{G_\theta}}$。

训练时交替更新 $\psi$ 和 $\theta$。这是 GAN 的节奏——一个网络追另一个网络的分布。

### OneDP-D（deterministic）：闭式解，省一个网络

如果 generator 不输入 $z$，即 $\mathbf{A}_\theta = G_\theta(\mathbf{O})$，那 $p_{G_\theta}$ 退化成 Dirac delta $\delta_{G_\theta(\mathbf{O})}$。Dirac delta 卷高斯 = 高斯，高斯的 score 是闭式的：

$$s_{p_{G_\theta}}(\mathbf{A}_\theta^k) = -\frac{\epsilon_k}{\sigma_k}$$

- $\epsilon_k$: 你前向加噪时实际采的那个噪声，已知
- $\sigma_k$: 调度系数

代入主 loss 化简：

$$\nabla_\theta \mathcal{L} = \mathbb{E}\left[ \frac{w(k)}{\sigma_k} \big( \epsilon_\phi(\mathbf{A}_\theta^k, k) - \epsilon_k \big) \nabla_\theta \mathbf{A}_\theta^k \right]$$

这就是标准的 [SDS loss](https://dreamfusion3d.github.io/) 形式！$\epsilon_\phi$ 是 teacher 预测的噪声，$\epsilon_k$ 是实际噪声，差值乘 $\nabla_\theta \mathbf{A}_\theta^k$ 反传给 generator。

OneDP-D 不需要训 $\pi_\psi$，计算和显存减半，但 generator 是确定性的，丢掉了多模态表达能力。

## 两个工程细节决定成败

### Warm-start 是 must

$G_\theta$ 和 $\pi_\psi$ 都从预训练 DP checkpoint 复制权重。这一点至关重要：distillation 不是从零学一个 generator，是让一个已经在 teacher 解空间里的网络做"微调"。如果 cold start，generator 早期输出全是垃圾，$\pi_\psi$ 学不到有意义的 score，整个 KL 优化方向乱跑。

这也是 [DMD](https://tianweiy.github.io/dmd/)、[SiD](https://score-id.github.io/) 的标配。

### Score network 学习率必须比 generator 快 20 倍

generator lr = $10^{-6}$，score network lr = $2 \times 10^{-5}$。

直觉：generator 每次更新都改变分布，$\pi_\psi$ 必须跟上这个变化才能给出准确的 $s_{p_{G_\theta}}$。如果 $\psi$ 跟 $\theta$ 同样慢，$\pi_\psi$ 估的是"过时的"generator 分布的 score，修正方向偏。这跟 GAN 里 discriminator 学习率通常要大于 generator 是一个道理。

作者还把两个网络的 Adam $\beta_1$ 都设成 0（不用动量），让分布追踪更敏感，这也是 [BigGAN](https://arxiv.org/abs/1909.02100) 训练 GAN 的 trick。

## 实验数据讲什么

### 模拟：20 epoch 就收敛，还略超 teacher

Robomimic + PushT 6 个任务平均成功率：

| | epochs | NFE | avg |
|---|---|---|---|
| DP (100步) | 1000 | 100 | 0.829 |
| DP (DDIM 10步) | 1000 | 10 | 0.836 |
| DP (1步) | 1000 | 1 | 0.000 |
| CP | 450 | 1 | 0.672 |
| CP | 450 | 3 | 0.712 |
| **OneDP-S** | **20** | **1** | **0.843** |

几个反直觉的点：
1. **OneDP-S 略超 DP**：0.843 > 0.829。distillation 后 student 比 teacher 还好。作者的猜想是迭代去噪会累积微小误差，single-step 一步跳过去反而干净。这跟 DMD 在图像蒸馏上的发现一致。
2. **CP 20 epoch 只有 0.251**，要 450 epoch 才到 0.672。consistency model 没有 adversarial auxiliary loss 收敛巨慢。OneDP 比 CP 收敛快 20× 以上。
3. **DP 1 步全 0**：直接跳过 denoise chain 不行，必须 distillation。

### 真实机器人：62 Hz 让动态任务变可行

[pnp-milk-move](https://research.nvidia.com/labs/dir/onedp/) 任务最能说明问题。人把抓取目标挪走时：

- DP (DDIM 10步)：66ms/决策，13 Hz，跟不上 box 移动，成功率 80%
- OneDP：16ms/决策（9ms 观测编码 + 7ms 生成），62 Hz，实时跟踪，成功率 **100%**

完成时间也快 10 秒以上——因为机械臂不再有"等 policy 想一下"的停顿，运动连续。

## 更广的联想

### 跟 LLM 里的 speculative decoding 像
LLM 里要"一步出多个 token"也是 distillation 思路：用一个强模型当 teacher 训一个能并行出多 token 的 student。OneDP 在 action space 做类似的事：一步出整个 action chunk（16 步动作）。差别是 action chunk 维度低（7 dof × 16 ≈ 112 维），distillation 容易得多，甚至 deterministic 闭式版都能 work；image generation 是 200K 维，必须 stochastic + score network。

### 跟 flow matching / rectified flow 的关系
[Flow Matching](https://arxiv.org/abs/2210.02747)、[Rectified Flow](https://arxiv.org/abs/2209.03003) 把 diffusion 路径拉直，理论上可以少步采样。OneDP 不走这条路——它不修改 teacher 的概率路径，直接学一个 student 把整条路径塌缩成一步。两者可以叠加：先 flow matching 训 teacher，再 OneDP 蒸馏。

### 跟 policy distillation 经典工作的区别
经典 [policy distillation (Rusu et al. 2015)](https://arxiv.org/abs/1511.06295) 是 RL 里把 Q-network 蒸成小 network，对齐的是 logit 或 action distribution。OneDP 蒸的是生成过程本身——teacher 是个采样器（diffusion），student 是个一步映射，对齐的是 marginal 分布而非单点 action。本质更接近 generative model distillation。

### 跟 BC 的多模态痛点
经典 behavior cloning 用 MSE 训 deterministic policy，在多模态示范下会 mode average。Diffusion Policy 之所以 SOTA 就是因为它能表达多模态。OneDP-S 保留了这点（通过 $z$ 注入随机性），OneDP-D 又退化回 deterministic 但靠 warm-start 锚定到一个具体 mode。Transport-ph 上 OneDP-S 比 OneDP-D 高 1.2 个点，可能就是多模态表达的差距。

### 为什么不直接用 consistency model？
Consistency Model ([Song et al. 2023](https://arxiv.org/abs/2303.01469)) 学的是 ODE 轨迹的自一致性：$f(x_t, t) = f(x_{t'}, t')$。约束是 trajectory 上的点映射同一，监督信号稀疏。CTM 加了 adversarial loss 增强，CP 简化掉了所以收敛慢。OneDP 用 KL 直接对齐分布，每个 $k$ 上都有监督信号，密度高，所以 20 epoch 就够。

### 可能的下一步
作者自己提了 KL 蒸馏可能不是最优，引入 discriminator（像 CTM 原版）可能更好。我额外想到几个方向：
1. **多 teacher 蒸馏**：把不同 noise schedule (DDPM/EDM/flow matching) 的 DP 蒸成一个 student，可能更鲁棒
2. **Long-horizon**：论文没测，但 action chunk 长度 16 是不是瓶颈？是否可以 distill 出 hierarchical policy（chunk 内一步 + chunk 间规划）
3. **在线适应**：62 Hz 留出了计算预算，能否把 distillation 搬到 online（demo 采集后实时蒸）
4. **跟 [3D Diffuser Actor](https://3d-diffuser-actor.github.io/) 结合**：3D 表示更强的 policy + 一步蒸馏 = 又快又准

### 局限的诚实评估
- Reverse KL 的 mode-seeking 在 deterministic 版本下确实可能丢 mode，论文靠 stochastic $z$ 缓解但没系统验证 mode 覆盖率
- 真实实验把控制频率限在 20 Hz（为了安全），其实没榨干 62 Hz 的潜力
- 只测了 manipulation，locomotion 或 whole-body control 没测
- 蒸馏成本 2-10% 这个数字漂亮，但前提是已经有预训练 DP；如果从零算总账，成本还是要付

## 一句话总结

OneDP 把 SDS/VSD 的 score distillation 范式从 text-to-3D 搬到 robot policy，用 reverse KL 对齐分布、在加噪样本上算 score 避免爆炸、warm-start 让 distillation 只需 20 epoch。结果是一个 62 Hz 的一步 policy，在动态任务上把 DP 按在地上摩擦，还顺便比 teacher 涨了点。核心 insight：**低维 action space 让 generative distillation 比图像领域容易一个数量级，连 deterministic 闭式解都能 work**。

参考链接：
- [OneDP project](https://research.nvidia.com/labs/dir/onedp/)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [DreamFusion (SDS)](https://dreamfusion3d.github.io/)
- [ProlificDreamer (VSD)](https://ml.cs.tsinghua.edu.cn/prolicdreamer/)
- [DMD](https://tianweiy.github.io/dmd/)
- [SiD](https://score-id.github.io/)
- [Diffusion-GAN](https://diffusion-gan.github.io/)
- [Consistency Policy](https://consistency-policy.github.io/)
- [Consistency Models](https://arxiv.org/abs/2303.01469)
- [CTM](https://arxiv.org/abs/2310.02279)
- [EDM](https://nvlabs.github.io/edm/)
- [Robomimic](https://robomimic.github.io/)
- [Diffusion-QL](https://arxiv.org/abs/2208.06193)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
- [3D Diffuser Actor](https://3d-diffuser-actor.github.io/)
- [ACT (action chunking)](https://tonyzhaozh.github.io/aloha/)

---

# One-Step Diffusion Policy (OneDP) 深度解析

Andrej，这篇 paper 是 NVIDIA + UT Austin 团队的工作，核心贡献是把 Diffusion Policy (Chi et al., 2023) 这种迭代式的 visuomotor policy 通过 distillation 压成一个 single-step generator。我会把数学推导、架构、和实验直觉都拆开来聊。

---

## 1. 为什么需要这个工作？背景与动机

### 1.1 Diffusion Policy 的痛点

[Diffusion Policy (DP)](https://diffusion-policy.cs.columbia.edu/) 在 [Robomimic](https://robomimic.github.io/) 等机器人 manipulation 任务上 SOTA，但它的本质是 conditional diffusion model 在 action space 上 denoise：

$$
\pi_\theta(\mathbf{A}^0 | \mathbf{O}) := \int \cdots \int \mathcal{N}(\mathbf{A}^K; \mathbf{0}, I) \prod_{k=K}^{k=1} p_\theta(\mathbf{A}^{k-1} | \mathbf{A}^k, \mathbf{O}) \, d\mathbf{A}^K \cdots d\mathbf{A}^1
$$

变量说明：
- $\mathbf{O}$: 过去几帧的视觉观测（observation images，ResNet18 编码后变成 feature）
- $\mathbf{A}^0$: 干净的 action chunk，即未来 16 步连续动作序列（action chunking idea 来自 [ACT](https://tonyzhaozh.github.io/aloha/)）
- $\mathbf{A}^K$: 纯 Gaussian 噪声（最嘈杂的 action）
- $k$: diffusion step index，从 $K$ 降到 $0$，$K=100$ (DDPM) 或连续 (EDM)
- $p_\theta(\mathbf{A}^{k-1}|\mathbf{A}^k, \mathbf{O})$: 由 1D temporal CNN U-Net 参数化的反向扩散核

**问题**：要跑 100 步 DDPM 或 10 步 DDIM 才能出一个 action chunk。在 Franka 机器人上，整条 pipeline（observation encoding + action prediction）实测 660ms（DDPM 100 步）或 66ms（DDIM 10 步），policy frequency 仅 1.5 Hz。这在人类干扰或动态环境下根本来不及反应——比如 pnp-milk-move 任务里，box 被人挪走时 DP 还在用上一帧观测算 action，必然抓空。

### 1.2 现有加速方案的不足

- **[DPM-Solver / EDM 2nd-order solver](https://nvlabs.github.io/edm/)**：10 步已经接近极限，1 步完全崩（Table 1 中 DP 1-step 全 0%）。
- **[Consistency Policy (CP)](https://consistency-policy.github.io/)**：基于 [Consistency Trajectory Model (CTM)](https://arxiv.org/abs/2310.02279)，需要 EDM 调度，且即使跑 450 epoch 也只到 0.672 avg success rate，比 DP 的 0.829 差一大截。而且 CP 还需要 3 步采样才达到 0.712。

OneDP 要做到：**1 步生成 + 不掉点（甚至略涨点）+ 收敛快**。

---

## 2. 方法论：从 SDS/VSD 借鉴的 score distillation 思路

### 2.1 关键 inspiration

这个工作直接受 [DreamFusion (SDS)](https://dreamfusion3d.github.io/) 和 [ProlificDreamer (VSD)](https://ml.cs.tsinghua.edu.cn/prolicdreamer/) 启发。SDS 把 2D diffusion 当 prior 优化 3D NeRF 的参数 $\theta$，梯度形式是：

$$
\nabla_\theta \mathcal{L}_{SDS}(\theta) = \mathbb{E}_{t, \epsilon}\left[ w(t) \left( \epsilon_\phi(\mathbf{x}_t; t) - \epsilon \right) \frac{\partial \mathbf{x}_t}{\partial \theta} \right]
$$

VSD 进一步把 SDS 看作 KL 散度优化，并引入一个 student score network 来逼近 generator 自己的 score，避免 SDS 的过饱和问题。OneDP 把这套搬到 action space：**3D asset → robot action chunk**，**text-to-3D → visuomotor policy distillation**。

### 2.2 Generator 定义

定义一个 one-step implicit generator $G_\theta$：

$$
z \sim \mathcal{N}(\mathbf{0}, I), \quad \mathbf{A}_\theta = G_\theta(z, \mathbf{O})
$$

变量：
- $z$: 标准高斯噪声向量（OneDP-S 使用，OneDP-D 省略）
- $\mathbf{O}$: 视觉观测 feature
- $\mathbf{A}_\theta$: 一步生成的 action chunk

$p_{G_\theta}$ 是 $G_\theta$ 诱导的隐式分布；预训练 diffusion policy $\pi_\phi$ 诱导分布 $p_{\pi_\phi}$，$\phi$ 全程 frozen。

### 2.3 Reverse KL 目标

作者选 **reverse KL**（$p_{G_\theta} \| p_{\pi_\phi}$）而非 forward KL，原因是 **mode-seeking**：

$$
\mathcal{D}_{KL}(p_{G_\theta} \| p_{\pi_\phi}) = \mathbb{E}_{z \sim \mathcal{N}(0,I),\, \mathbf{A}_\theta = G_\theta(z,\mathbf{O})}\left[ \log p_{G_\theta}(\mathbf{A}_\theta|\mathbf{O}) - \log p_{\pi_\phi}(\mathbf{A}_\theta|\mathbf{O}) \right]
$$

直觉：reverse KL 会惩罚 generator 采样到 teacher 低概率区域（$\log p_{\pi_\phi} \to -\infty$），但不惩罚 teacher 有而 generator 没覆盖的 mode。在 offline RL / 机器人示范数据里，multi-modal action 分布很常见（同一个观测下有多种合理动作），mode-seeking 行为让 student 不会"平均"出无意义动作（这正是 vanilla BC 在 multi-modal 下崩的原因）。这一点和 [Diffusion-QL](https://arxiv.org/abs/2208.06193) 在 offline RL 里用 reverse KL 的逻辑一致。

### 2.4 对 $\theta$ 求梯度 → score difference

对 $\theta$ 求梯度：

$$
\nabla_\theta \mathcal{D}_{KL}(p_{G_\theta} \| p_{\pi_\phi}) = \mathbb{E}\left[ \big( \nabla_{\mathbf{A}_\theta} \log p_{G_\theta}(\mathbf{A}_\theta|\mathbf{O}) - \nabla_{\mathbf{A}_\theta} \log p_{\pi_\phi}(\mathbf{A}_\theta|\mathbf{O}) \big) \nabla_\theta \mathbf{A}_\theta \right]
$$

这里出现了两个 **score**：
- $s_{p_{G_\theta}}(\mathbf{A}_\theta) = \nabla_{\mathbf{A}_\theta} \log p_{G_\theta}(\mathbf{A}_\theta|\mathbf{O})$: generator 自己分布的 score
- $s_{p_{\pi_\phi}}(\mathbf{A}_\theta) = \nabla_{\mathbf{A}_\theta} \log p_{\pi_\phi}(\mathbf{A}_\theta|\mathbf{O})$: teacher 分布的 score

### 2.5 两个困难 → Diffusion-GAN 的 trick

直接用上面公式有两个麻烦：

**困难 1**: 当 $\mathbf{A}_\theta \sim p_{G_\theta}$ 落在 $p_{\pi_\phi}$ 的低概率区时，$\log p_{\pi_\phi} \to -\infty$，score 爆炸。SDS 也有这问题（"过饱和、过平滑"）。

**困难 2**: Diffusion model 只能算 diffused sample 的 score（即 $\mathbf{x}^k$ 处的 score），不能算 clean sample 的 score（因为 denoising score matching 只训了带噪样本）。

**关键 trick**: 不要在 clean action $\mathbf{A}_\theta$ 上算 KL，而是先做一次 forward diffusion 再算（思路来自 [Diffusion-GAN](https://diffusion-gan.github.io/)）：

$$
\nabla_\theta \mathbb{E}_{k\sim \mathcal{U}}[\mathcal{D}_{KL}(p_{G_\theta,k} \| p_{\pi_\phi,k})] = \mathbb{E}\left[ w(k) \big( s_{p_{G_\theta}}(\mathbf{A}_\theta^k) - s_{p_{\pi_\phi}}(\mathbf{A}_\theta^k) \big) \nabla_\theta \mathbf{A}_\theta^k \right]
$$

变量：
- $k \sim \mathcal{U}$: 在 diffusion 时间轴上均匀采样
- $\mathbf{A}_\theta^k = \alpha_k \mathbf{A}_\theta + \sigma_k \epsilon_k$, $\epsilon_k \sim \mathcal{N}(0,I)$: 对 generator 输出做一步前向加噪
- $p_{G_\theta,k}$ 和 $p_{\pi_\phi,k}$: $p_{G_\theta}$ 和 $p_{\pi_\phi}$ 在 $k$ 步加噪后的 marginal 分布
- $w(k)$: 重加权函数，作者沿用 DreamFusion 设 $w(k) = \sigma_k^2$
- $s_{p_{\pi_\phi}}(\mathbf{A}_\theta^k)$: 可以直接用预训练 $\epsilon_\phi$ 算，$s = -\epsilon_\phi/\sigma_k$
- $s_{p_{G_\theta}}(\mathbf{A}_\theta^k)$: generator 自己分布的 score —— 这一项**不能**直接算，因为 $p_{G_\theta}$ 是 implicit 分布

### 2.6 Generator score network $\pi_\psi$ (OneDP-S)

为了估 $s_{p_{G_\theta}}$，作者引入辅助 diffusion network $\pi_\psi$，参数为 $\epsilon_\psi$，用标准 denoising score matching 训练，但目标数据集是 generator 的输出：

$$
\min_\psi \mathbb{E}_{\mathbf{x}^k \sim q(\mathbf{x}^k | \mathbf{x}^0),\, \mathbf{x}^0 = \text{stop-grad}(G_\theta(z)),\, z \sim \mathcal{N}(0,I),\, k \sim \mathcal{U}}\left[ \lambda(k) \cdot \| \epsilon_\psi(\mathbf{x}^k, k) - \epsilon_k \|^2 \right]
$$

关键点：
- **stop-grad**: generator 输出 detach 掉，不让 score network 的训练影响 generator
- $\epsilon_\psi$ 学到 generator 分布的 score，然后代入主 loss 的 $s_{p_{G_\theta}}$ 项
- 主 loop：交替更新 $\psi$（式 6）和 $\theta$（式 5）

这是 VSD 的标准做法，把 SDS 里那项 $\epsilon$（先验噪声）换成 $\epsilon_\psi$（generator 自己的 score），消除 SDS 的 mode-seeking 过强导致的 mode collapse / 过平滑。

### 2.7 Deterministic 简化 (OneDP-D)

如果 generator 是确定的（去掉 $z$，$\mathbf{A}_\theta = G_\theta(\mathbf{O})$），则 $p_{G_\theta} = \delta_{G_\theta(\mathbf{O})}$ 是 Dirac delta。给 Dirac delta 加高斯噪声 $q(\mathbf{A}^k | \mathbf{A}_\theta) = \mathcal{N}(\alpha_k \mathbf{A}_\theta, \sigma_k^2 I)$，score 可以解析地写出来：

$$
s_{p_{G_\theta}}(\mathbf{A}_\theta^k) = \nabla_{\mathbf{A}_\theta^k} \log q(\mathbf{A}_\theta^k | \mathbf{A}_\theta) = -\frac{\epsilon_k}{\sigma_k}, \quad \text{where } \mathbf{A}_\theta^k = \alpha_k \mathbf{A}_\theta + \sigma_k \epsilon_k
$$

**直觉**：Dirac delta 卷 Gaussian = Gaussian，Gaussian 的 log-density 对 $\mathbf{A}^k$ 求梯度恰好就是 $-(\mathbf{A}^k - \alpha_k \mathbf{A}_\theta)/\sigma_k^2 = -\epsilon_k/\sigma_k$。这里 $\epsilon_k$ 就是前向加噪时实际采样的噪声，是已知量。

代入主式 5 得简化 loss：

$$
\nabla_\theta \mathbb{E}_{k \sim \mathcal{U}}[\mathcal{D}_{KL}(p_{G_\theta,k} \| p_{\pi_\phi,k})] = \mathbb{E}\left[ \frac{w(k)}{\sigma_k} \big( \epsilon_\phi(\mathbf{A}_\theta^k, k) - \epsilon_k \big) \nabla_\theta \mathbf{A}_\theta^k \right]
$$

这正是 **SDS loss 形式**！$\epsilon_\phi$ 是 teacher 的噪声预测，$\epsilon_k$ 是实际加噪，$\nabla_\theta \mathbf{A}_\theta^k$ 通过 chain rule 传回 generator。OneDP-D 不需要训 $\pi_\psi$，参数和计算量减半，但失去了 multi-modal 采样的能力。

### 2.8 算法整体

```
Initialize: G_θ ← π_φ (warm-start from pretrained DP), π_ψ ← π_φ
while not converged:
  Sample z ~ N(0, I), A_θ = G_θ(z, O)
  Diffuse: A_θ^k = α_k A_θ + σ_k ε_k
  if OneDP-S:
    Update ψ by Eq.(6)  # train generator score net
    Update θ by Eq.(5)  # use s_{p_ψ} as s_{p_G}
  elif OneDP-D:
    Update θ by Eq.(8)  # closed-form s_{p_G} = -ε_k/σ_k
```

---

## 3. 架构细节

### 3.1 网络结构

- **Vision encoder**: ResNet18 (从 scratch 训，no pretrain)，输入 120×160 RGB
- **Denoising backbone**: 1D temporal CNN U-Net（[Janner et al. 2022](https://diffusion-planning.github.io/) 风格）
- **DDPM 版本**: 256M 参数，100 step
- **EDM 版本**: 67M 参数（小一点反而更好，作者实测）
- **Real-world**: 67M 参数

### 3.2 重要 implementation 细节

- **Warm-start**: $G_\theta$ 和 $\pi_\psi$ 都从预训练 DP checkpoint 复制结构和权重。这是 [DMD](https://tianweiy.github.io/dmd/) / [SiD](https://score-id.github.io/) 等蒸馏工作的标配，避免 cold start。
- **Generator timestep 输入**: 因为 backbone 原本是 denoising 网络，需要 timestep 输入。作者把 timestep 固定为 $t_{init}=65$ (DDPM) 或 $\sigma=2.5$ (EDM)，相当于"假装"在一个固定噪声水平上做单步 denoise。
- **Distillation 时间范围**: $k \in [2, 95]$（DDPM），避开两端边缘 case；EDM 用 log-normal schedule。
- **学习率**: generator $1\times 10^{-6}$（慢），score network $2\times 10^{-5}$（快 20×）。让 score network 跟得上 generator 分布的快速变化。
- **Optimizer**: Adam $\beta_1=0$，借鉴 GAN 训练 trick，让两个网络协同演化更快。

### 3.3 数据流图

```
       ┌──────────────┐
   O → │  ResNet18    │ ──── obs feature ────────┐
       └──────────────┘                          │
                                                 ↓
   z ~ N(0,I) ──→ G_θ(z, O) ──→ A_θ ──→ forward diffuse ──→ A_θ^k
                                                                │
                ┌────────────────────────────────────────────────┤
                ↓                                                ↓
       ε_ψ(A_θ^k, k) ←──── π_ψ ────        ε_φ(A_θ^k, k) ←── π_φ (frozen)
                │                                                │
                └── s_{p_G} ──── score diff ─── s_{p_π} ──────────┘
                                  │
                                  ↓
                            w(k)/σ_k × (ε_φ - ε_ψ 或 ε_k)
                                  │
                                  ↓ × ∇_θ A_θ^k
                            ∇_θ L → update G_θ
```

---

## 4. 实验结果深度解析

### 4.1 模拟实验：Robomimic + PushT

**Table 1 (DDPM setting)**：

| Method | Epochs | NFE | PushT | Square-mh | Square-ph | ToolHang-ph | Transport-mh | Transport-ph | Avg |
|---|---|---|---|---|---|---|---|---|---|
| DP (DDPM) | 1000 | 100 | 0.863 | 0.846 | **0.926** | 0.822 | 0.620 | 0.896 | 0.829 |
| DP (DDIM) | 1000 | 10 | 0.823 | 0.850 | 0.918 | 0.828 | 0.688 | 0.908 | 0.836 |
| DP (DDIM) | 1000 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **OneDP-D** | 20 | 1 | 0.802 | 0.846 | 0.926 | 0.808 | 0.676 | 0.896 | 0.826 |
| **OneDP-S** | 20 | 1 | 0.816 | **0.864** | **0.926** | **0.850** | **0.690** | **0.914** | **0.843** |

几个观察：
1. **DP 1-step 全 0**：直接跳过 denoise chain 完全失败，说明 100 step 训出来的 $\epsilon_\phi$ 在 $k=0$ 邻域不能直接一步外推。
2. **OneDP-S (20 epoch) > DP (1000 epoch)**: 0.843 vs 0.829，蒸馏后 student 反而略涨！作者归因为 iterative denoising 会累积微小误差，而 single-step 跳过这种累积（和 [DMD](https://tianweiy.github.io/dmd/) 在图像蒸馏上的发现一致）。
3. **OneDP-S > OneDP-D**: stochastic policy 在 multi-modal 任务（Transport、ToolHang）更鲁棒，因为可以表示分布；deterministic 退化为 Dirac delta。
4. **Distillation 成本**: 20 epoch vs DP 的 1000 epoch = 2% 预训练成本。

**Table 2 (EDM setting, 含 CP 比较)**：

| Method | Epochs | NFE | Avg |
|---|---|---|---|
| DP (EDM) | 1000 | 35 | 0.829 |
| DP (EDM) | 1000 | 19 | 0.818 |
| DP (EDM) | 1000 | 1 | 0.000 |
| CP | 20 | 1 | 0.251 |
| CP | 450 | 1 | 0.672 |
| CP | 450 | 3 | 0.712 |
| **OneDP-D** | 20 | 1 | 0.812 |
| **OneDP-S** | 20 | 1 | **0.830** |

观察：
- **CP 20 epoch 只有 0.251**: consistency model 收敛超慢，因为它没有 CTM 原版的 adversarial auxiliary loss（CTM 用 discriminator 增强 distillation）。
- **CP 450 epoch 才到 0.672**，3-step 才到 0.712，仍远低于 OneDP。
- **OneDP 20 epoch 就到 0.83**: 收敛快 **20×** 以上（Figure 4 画的曲线对比很直观）。

为什么 OneDP 收敛这么快？我猜测：warm-start 让 generator 一开始就在 teacher 的"解空间"附近；KL-score 形式梯度信号直接对齐分布而不是 trajectory，比 self-consistency 约束更"密集"的监督。

### 4.2 真实机器人实验

**Setup**: Franka Panda + RealSense D415 (front) + D435 (wrist)，120×160 RGB，phone-based teleoperation 收 100 demo/task。

**4 个任务**（难度递增）：
1. **pnp-milk**: 抓固定 milk box 放到 box
2. **pnp-anything**: 11 种不同物体随机选
3. **pnp-milk-move**: 动态干扰，人把 milk box 移走（10 条预设轨迹）
4. **coffee**: 抓 coffee pod → 放入 holder → 关 lid，多阶段高精度

**Table 3 (success rate)**：

| Method | Epochs | NFE | pnp-milk | pnp-anything | pnp-milk-move | coffee | Avg |
|---|---|---|---|---|---|---|---|
| DP (DDIM) | 1000 | 10 | 1.00 | 0.95 | 0.80 | 0.80 | 0.83 |
| OneDP-D | 100 | 1 | 1.00 | 1.00 | 1.00 | 0.80 | 0.95 |
| OneDP-S | 100 | 1 | 1.00 | 1.00 | 1.00 | 0.90 | **0.98** |

最亮眼的是 **pnp-milk-move**：DP 只有 80%，OneDP 全 100%。原因在 Table 5：

**Table 5 (inference speed)**：

| | OE | DDPM (100) | DDIM (10) | OneDP (1) |
|---|---|---|---|---|
| Time (ms) | 9 | 660 | 66 | 7 |
| NFE | 1 | 100 | 10 | 1 |

OneDP 端到端 16ms（9 OE + 7 pred），policy frequency 62 Hz；DP(DDIM) 是 75ms，13 Hz；DP(DDPM) 是 669ms，1.5 Hz。

pnp-milk-move 任务里，box 被移走时机械臂 camera 视野变化快。DP 慢到跟不上，预测的 action 还基于 0.5 秒前的观测，gripper 落空。OneDP 实时跟踪，所以 100% 成功。

**Table 4 (completion time)**：

| Method | pnp-milk | pnp-anything | pnp-milk-move | coffee | Avg |
|---|---|---|---|---|---|
| DP (DDIM) | 29.74 | 26.03 | 34.75 | 54.92 | 36.36 |
| OneDP-D | 23.21 | 22.93 | 28.73 | 33.13 | 27.00 |
| OneDP-S | 22.69 | 22.62 | 28.15 | 29.78 | **25.81** |

OneDP 平均快 10 秒以上。原因：快速 action prediction 让机械臂移动更流畅，没有"思考停顿"导致的犹豫。

---

## 5. 关键 ablation

### 5.1 Generator score network 学习率

作者扫了 $[10^{-6}, 10^{-5}, 2\times 10^{-5}, 3\times 10^{-5}, 4\times 10^{-5}]$，发现 $2\times 10^{-5}$ 最优。逻辑：score network lr 必须比 generator lr 大（20×），否则 score network 滞后于 generator 分布，估计的 $s_{p_{G_\theta}}$ 不准，distillation 信号偏。

### 5.2 Optimizer $\beta_1 = 0$

Adam 的 $\beta_1$ 控制一阶矩的动量。设为 0 等于不用动量，让 GAN-style 双网络对抗时更稳。这是 [BigGAN](https://arxiv.org/abs/1909.02100)、[StyleGAN](https://nvlabs.github.io/stylegan/) 等的常见 trick。

---

## 6. 与相关工作的关系

### 6.1 SDS / VSD 谱系

| 方法 | 应用 | Student score | 优化空间 |
|---|---|---|---|
| [SDS (DreamFusion)](https://dreamfusion3d.github.io/) | text-to-3D | 用 $\epsilon$ 近似 | 3D scene param |
| [VSD (ProlificDreamer)](https://ml.cs.tsinghua.edu.cn/prolicdreamer/) | text-to-3D | 训 $\epsilon_\psi$ | 3D scene param |
| [DMD](https://tianweiy.github.io/dmd/) | image gen | 训 $\epsilon_\psi$ | image generator |
| [SiD](https://score-id.github.io/) | image gen | Fisher divergence | image generator |
| **OneDP** | robot policy | 训 $\epsilon_\psi$ (S) 或 closed-form (D) | action generator |

OneDP 是 VSD/DMD 思路在 robot policy 上的应用，差异在于：robot 任务输出是 low-dim action chunk（不是高维 image/3D），所以 deterministic 变体 OneDP-D 可行（Dirac delta 加噪后 score 解析）。

### 6.2 与 Consistency Policy 比较

| 维度 | OneDP | CP |
|---|---|---|
| 基础 | VSD-style score distillation | CTM (consistency trajectory) |
| 需要 EDM 调度 | 否 | 是 |
| Adversarial aux loss | 否 | 否（CTM 简化版） |
| 一步性能 | 0.83 (sim) | 0.67 (sim, 450ep) |
| 收敛 | 20 epoch | 450 epoch |
| 步数 | 1 | 1-3 |

### 6.3 与 Diffusion Policy 蒸馏谱系

Diffusion Policy 加速目前有三条路：
1. **Solver-based**: DDIM, [DPM-Solver](https://arxiv.org/abs/2206.00927), [EDM](https://nvlabs.github.io/edm/)。10 步左右。
2. **Consistency-based**: [Consistency Models](https://arxiv.org/abs/2303.01469), CP。1-3 步，但训练复杂。
3. **Score distillation**: OneDP。1 步，warm-start + KL。

---

## 7. 直觉与思考

### 7.1 为什么 single-step 能 work？

这是最反直觉的地方：100 步迭代去噪怎么压成 1 步？我理解的关键：
- Diffusion 在生成时之所以需要多步，是因为 $\epsilon_\theta$ 只在 training 分布（diffused samples）附近被训练好。从纯噪声 $\mathbf{A}^K$ 一步跳到 $\mathbf{A}^0$ 会落到 OOD 区域。
- 但 distillation 时，generator $G_\theta$ 是**自由形态**的，它不再受限去预测噪声，而是直接学一个映射 $z \to \mathbf{A}$，让输出的 marginal 分布匹配 teacher。这是 functional 空间的直接对齐，比 trajectory 对齐容易得多。
- Warm-start 让 generator 一开始就在 teacher 的输出 space 附近，distillation 只需做"局部细化"。

### 7.2 Reverse KL 的代价

Reverse KL 是 mode-seeking，会"挑"teacher 的某个 mode 而忽略其他。这对 deterministic generator (OneDP-D) 严重——它只能输出一个 mode。Stochastic (OneDP-S) 通过 $z$ 注入噪声，可以在不同 $z$ 下触发不同 mode，所以 multi-modal 表现更好。Transport-ph 上 OneDP-S 比 OneDP-D 高 1.2 个点，可能就是这点差距。

### 7.3 为什么 score network 必须快？

如果把 $\psi$ 和 $\theta$ 设同样的慢 lr，$\pi_\psi$ 估的 $s_{p_{G_\theta}}$ 永远滞后 generator 当前分布，相当于在用"过时的"score 修正 generator，方向偏。设 20× lr 让 $\pi_\psi$ 准实时追踪 generator 分布。这是 GAN 训练里"D 跟得上 G"的标准 issue。

### 7.4 为什么 OneDP 不会 mode collapse？

理论上 reverse KL 有 mode collapse 风险。这里没观察到的原因我猜：
1. Warm-start 让 generator 从 teacher 的 multi-modal 分布出发
2. Observation conditioning $\mathbf{O}$ 已经把 multi-modality 大部分编码到 condition 里（同一观测下可能只有单一合理动作）
3. Stochastic generator 的 $z$ 提供了隐式 mode selector

### 7.5 局限

作者在 conclusion 提到：
- 没测 long-horizon 任务
- 机器人控制频率限在 20 Hz，没充分发挥 OneDP 的 62 Hz 潜力
- KL 蒸馏可能不是最优，引入 discriminator（像 CTM 原版）可能更好

---

## 8. 总结

OneDP 给我最大的启发是：**robot policy 的 diffusion distillation 远比 image diffusion distillation 容易**。原因是 action 是 low-dim 且结构化（chunk of 16 actions × 7 dof ≈ 112 维），而 image 是 $256 \times 256 \times 3 \approx 200K$ 维。low-dim 让 Dirac delta 解析 score (OneDP-D) 可行，让 warm-start 后的 distillation 几乎不用动太多参数。

数学上的核心：把 SDS 的 $\epsilon - \epsilon_\phi$ 形式推广到 student score network（VSD 思路）+ 在 diffused samples 上做（Diffusion-GAN 思路）= 既能避免 score 爆炸又能匹配 implicit 分布。Deterministic 变体进一步把 student score 项闭式化，省一半计算。

工程上：62 Hz policy frequency 让 dynamic manipulation（人干扰、移动目标）真正可行。这是 Diffusion Policy 走向 real-time 的关键一步。

**Reference 链接**：
- Paper project page: https://research.nvidia.com/labs/dir/onedp/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Robomimic: https://robomimic.github.io/
- DreamFusion (SDS): https://dreamfusion3d.github.io/
- ProlificDreamer (VSD): https://ml.cs.tsinghua.edu.cn/prolicdreamer/
- DMD: https://tianweiy.github.io/dmd/
- SiD: https://score-id.github.io/
- Consistency Policy: https://consistency-policy.github.io/
- EDM: https://nvlabs.github.io/edm/
- Diffusion-GAN: https://diffusion-gan.github.io/
- DDPM: https://arxiv.org/abs/2006.11239
- Score SDE: https://arxiv.org/abs/2011.13456
- CTM: https://arxiv.org/abs/2310.02279
- Consistency Models: https://arxiv.org/abs/2303.01469
- ACT (action chunking): https://tonyzhaozh.github.io/aloha/
- 3D Diffuser Actor: https://3d-diffuser-actor.github.io/
- Diffusion-QL: https://arxiv.org/abs/2208.06193
