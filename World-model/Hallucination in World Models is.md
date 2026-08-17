---
source_pdf: Hallucination in World Models is.pdf
paper_sha256: 2f10358cff583fc00df6a434c31a01872748bd9ed767c5874ff23eaecf823beb
processed_at: '2026-08-04T23:21:46-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 paper

## 一句话版本

world model 会"幻觉"（生成得挺好看但其实不对），根因是**数据没盖到那块地方**，所以能预测、能治。

---

## 先说 world model 是个啥

你给它一帧画面 + 一个 action，它预测下一帧。串起来就是"想象未来"。然后你拿这个想象去跑 planner 做决策。所以它要是偷偷错了，你的决策就跟着错——而且你看不出来，因为画面挺流畅的。

这跟 LM 的 hallucination 不一样：LM 错一个 token 就是错一个 token，world model 错一帧，后面整条轨迹全跟着歪，因为它是 sequential rollout，error 会 compound。

---

## 三种"幻觉"，对应三个零件

world model 想象未来要过三道关，每道关都可能出问题：

**第一种：看不清（perceptual）** —— tokenizer 的锅。你给它一帧没见过的场景，它重建出来是另一个见过的场景。agent 和 goal 位置对，但墙全是另一个 maze 的墙。然后 dynamics 拿着这个错的场景往下算，还以为是真的。

这个最阴险：你 horizon = 0 就能看见它，单帧重建就错了。

**第二种：不听话（action-marginalized）** —— dynamics 的锅。你给 action a 它往左走，给 action b 它也往左走，给 action c 它还往左走。画面流畅，但 action 根本没起作用，它退化成一个无条件的 video generator 了。

这个最坑：PSNR 抓不住，因为画面本身没问题。你得 shuffle action 看预测动不动才知道。

**第三种：跑飞了（scene-diverging）** —— multi-step rollout 的锅。Pong 里球得分了又瞬移回场地中间。物理上不可能的事它给你编出来了。通常发生在数据覆盖差的区域。

---

## 三个检测信号，都不要 label 不要额外训练

**信号一 $u_r$（round-trip residual）**：你预测出下一帧的 latent $\hat{z}$，decode 回 pixel，再 encode 回 latent，看跟原来的 $\hat{z}$ 差多少。差得多 = 这帧 decode 出了 tokenizer manifold，肯定有问题。对应"看不清"。

**信号二 $u_f$（flow instability）**：diffusion 采样时看 clean-target prediction 在 substep 之间抖不抖。抖 = dynamics head 没拿到有效 conditioning signal。对应"不听话"。

**信号三 $u_s$（inter-seed variance）**：同个 context 同个 action，换几个 noise seed 多跑几次，看结果分叉不分叉。分叉 = epistemic uncertainty 高，multi-step 会 fan-out。对应"跑飞了"。

三个信号机制完全不同，但都跟 rollout ∆PSNR 相关性 $\rho \approx 0.80$。作者说这恰好证明信号是真的——三条独立路径都指向同一个地方。

**一个关键 trick**：三个信号都要除以"这帧场景动多少"（latent 的 RMS change），不然 high-motion transition 会把信号搞混。normalized 之后 AUROC 从 0.85 跳到 0.93 左右。

---

## 两种治法

### 治法一：重新配数据（免费）

原来 sampling 是 uniform across frames，Atari 那 1000-step episode 就会淹没 ManiSkill3 那 25-step 的。改成 uniform across tasks，每个 task 平等被采样。

结果：所有三个信号同时变好，rollout ∆PSNR +0.88 dB。一个改动打三个目标，因为三个 failure mode 的根都是 coverage gap。

### 治法二：拿信号当 curiosity 去主动找数据（花点钱但省人工）

跟 live env 交互时，先在 world model 里 rollout 几条候选 trajectory，用 $u_r^{\text{norm}}$ 打分，挑分最高的去真实 env 执行。by construction，收集的就是之前让 model 幻觉的 transition。

结果：10 个完全没见过的 task，每个只采 50 条 curiosity trajectory，finetune 后 task performance 0.325。对比 expert data finetune 是 0.362，human play 也是 0.362。**curiosity 几乎追平 expert，而且不用任何 expert 知识**。

---

## 最实用的几个 takeaways

1. **unseen task 上，tokenizer 是第一堵墙**。没 fine-tune tokenizer 前，Recon PSNR 17 dB；fine-tune 后跳到 35+。你单帧都重建不对，谈 dynamics 没意义。所以 transfer 到新 task，先把 tokenizer 在新 domain 上 fine-tune。

2. **uniform-task sampling 几乎免费就能涨**。就改个采样策略，不用新数据、不用改架构。

3. **curiosity 收数据 ≈ expert 收数据**。这意味着在没 expert 的 domain（real robot、新游戏），你不用找人示教，让 world model 自己找自己怕的地方去采数据就行。

4. **pixel PSNR 会骗你**。画面好看不等于 action 起作用了。必须额外测 action sensitivity（shuffle ratio）。

5. **off-the-shelf tokenizer 在 unseen 上更强，但 in-domain fine-tune 后反超**。Wan 2.1 VAE 在 unseen task 36.62 dB，但你的 in-domain tokenizer fine-tune 后能到 38.04。所以最佳策略：先用 off-the-shelf 起步，再 in-domain fine-tune。

---

## 对你的 build intuition 价值

这篇 paper 的 framing 很 clean：**别一看到 model 出问题就想着把 backbone 加大。先问数据盖到那了没。**

这个思路跟 Chinchilla、DataComp 一脉相承，但专门化到了 world model 的 sequential pipeline 上。核心 insight 是：**world model 的 hallucination 会跨 stage 放大**——tokenizer 错一点，dynamics 跑多步就放大成 scene divergence。所以早期 stage 的 coverage gap 影响远大于后期 stage。

这跟 LM 本质不同：LM 的 hallucination 是同类型操作（token → token）autoregressive 累积；world model 的 hallucination 是 cross-stage（latent → pixel → latent），stage 之间 representation 都不一样。所以"哪个 stage 出问题"在 world model 是 first-class question。

可迁移的方法学：任何 action-conditioned generative model——diffusion policy、video generation with controls、甚至 LM 的 tool-use（tool call 有没有被 marginalized）——都能用这套"intervention + 内部信号"的框架去诊断。

paper project page 有 video demo：https://nicklashansen.com/mmbench2

---

# Hallucination in World Models is Predictable and Preventable 深度解读

## 核心论点与 motivation

这篇文章来自 Nicklas Hansen 和 Xiaolong Wang（UCSD），project page: https://nicklashansen.com/mmbench2 。核心论点非常清晰：**generative world model 的 hallucination 本质上是 data coverage 问题**，因此可预测、可预防。作者反对把 hallucination 当作纯 architecture 问题用 scale 硬怼的思路（对应 Hoffmann et al. 的 Chinchilla scaling law 思路，https://arxiv.org/abs/2203.15555 ）。

这个 framing 很关键。在 LM 文献里，"hallucination" 指生成 factually incorrect text（Ji et al. survey: https://arxiv.org/abs/2202.03629 ）；image/video 生成里也有 object hallucination 问题（Li et al.: https://arxiv.org/abs/2310.00754 ）。但 world model 的 hallucination 后果更严重——因为 rollout 结果会直接喂给下游 planner/policy（如 MuZero 风格的 MCTS，Schrittwieser et al.: https://www.nature.com/articles/s41586-020-03051-4 ；TD-MPC2，Hansen et al.: https://arxiv.org/abs/2310.16828 ），silent hallucination 等于 silent wrong decision。

作者的赌注是：一个单一的底层 cause——**coverage gaps**——能解释 pipeline 每个阶段的 failure。这是一个 strong claim，因为它把三个看似不相关的 failure mode 统一到了一个 data-centric lens 下面。

---

## MMBench2：为研究 hallucination 专门造的数据集

要研究 "hallucination 在哪里、为什么发生"，需要三个资源同时具备：(1) 训练 corpus 完全可控；(2) 行为多样的数据跨多个 task/domain；(3) 有 live environment 能 online interaction 探测 coverage gap。现有 dataset 都缺至少一项。

### 数据规模与组成

- **65.6k trajectories / 427 小时 / 23M frames**，全部 224×224 @ 15fps
- **210 个 task 跨 10 个 domain**：DMControl (Tassa et al., https://arxiv.org/abs/1801.00690 ), DMControl Extended, Meta-World (Yu et al., https://arxiv.org/abs/1910.10897 ), ManiSkill3 (Tao et al., https://arxiv.org/abs/2410.00425 ), MuJoCo (Todorov et al.), MiniArcade, Box2D, RoboDesk, OGBench (Park et al., https://arxiv.org/abs/2410.20092 ), Continuous Atari (Farebrother & Castro, https://arxiv.org/abs/2405.14960 )
- **200 task 用于 pretraining，10 task 完全 held-out 作 unseen transfer**
- Action dim 1–16，全部 zero-pad 到 $d_a = 16$，配 per-dimension validity mask
- Episode 长度 25–1000，导致 per-task frame 数严重不均（Figure 3 heavy-tailed：top 20 task 占 26% frame，bottom 20 只占 0.7%）

### Behavior diversity（这是 MMBench2 相对 MMBench 的关键升级）

这是我最欣赏的设计之一。作者刻意 mixed-quality：
- **Random policy**：action $\sim U([-1,1])$，coverage 广但 task 表现差
- **No-op**：$\mathbf{a} = \mathbf{0}$，纯 passive dynamics
- **Expert** $\pi^\star$：high performance, low diversity
- **Transformed expert**：5 种 transform（scale by $\varepsilon \in [0,1]$、dropout、flip sign、zero、repeat prev），构造 counterfactual transition
- **Structured noise**：Gaussian / Ornstein-Uhlenbeck / mixture with random
- **Curiosity-driven**：用本文提出的 $u_r^{\text{norm}}$ 作 reward，CEM planning
- **Human play**：keyboard 接口（Figure 9），共 1400 条 trajectory

这个设计直击 expert-only dataset 的盲区——一个只看过 expert 轨迹的 world model 在偏离 expert manifold 时就会 hallucinate，而 RL 控制、recovery 行为恰恰需要这种 off-manifold 的能力。

与现有数据集对比（Table 4）：MMBench2 是唯一同时提供 ground-truth action + reward + live env + mixed-quality behavior + 跨 domain 的 corpus。VPT (https://arxiv.org/abs/2210.01537 ) 和 NitroGen 是 pseudo-label action；DROID (https://arxiv.org/abs/2403.12945 ) 和 Open X-Embodiment (https://arxiv.org/abs/2310.08864 ) 没 reward 和 live env。

---

## 架构细节：Dreamer 4 reproduction（350M params）

作者 reproduce 了 Dreamer 4（Hafner et al., https://arxiv.org/abs/2509.24527 ），total 350M：tokenizer ~100M + dynamics ~230M + reward/BC head ~20M。训练 8×H100，tokenizer 300k step（14 GPU day），dynamics 180k step（24 GPU day）。

### Tokenizer（symmetric encoder-decoder MAE）

输入：224×224 RGB frame，patchify stride=14 → **256 patch token**。前面 prepend **64 learnable latent query**，project 到 **64-dim bottleneck** + tanh，输出 per-frame code $z \in [-1, 1]^{64 \times 64}$（即 $n_L = 64$ latent，每个 64 dim，total 4096 dim/frame）。

- Encoder/decoder 各 50M params，depth 12，$d_{\text{model}} = 512$，8 head，MLP ratio 4
- **Masked reconstruction**（MAE, He et al. https://arxiv.org/abs/2111.06377 ）：per-frame 随机 mask fraction $\sim U(0, 0.9)$（注意是 keep range [0.1, 1.0]，即最多 mask 掉 90%），loss 只在 masked position 上算 pixel MSE + LPIPS（Zhang et al., https://arxiv.org/abs/1801.03924 ，weight 0.2）
- **Loss 归一化**：每个 loss term 各自除以 running RMS 再加权——这是个很实用的 trick，让 loss weight 不再依赖 dataset/resolution/backbone 的绝对 scale

**Modality-aware mask**：encoder 里 latent query attend 所有 token，patch query 只 attend image modality（防止 patch 在 bottleneck 之前跨 modality mix）；decoder 反过来——patch 能 read latent bottleneck，但 patch 之间不直接互看。这个 asymmetric 设计强制信息流过 latent bottleneck，否则 decoder 可能绕过 bottleneck 直接从 patch token 重建，让 latent 失去信息压缩的作用。

### Dynamics model（block-causal Transformer + shortcut flow matching）

250M params，depth 16，$d_{\text{model}} = 1024$，8 head。每个 timestep 的 token layout：

$$[\text{ACTION} \times 1,\ \text{SHORTCUT} \times 1,\ \text{SPATIAL} \times 32,\ \text{REGISTER} \times 4,\ \text{AGENT} \times 4]$$

- **ACTION token**：2-layer MLP over 16-dim padded action
- **SHORTCUT token**：concatenate 两个 embedding（noise level $\sigma$ 和 step size $d = 1/2^{\text{step}}$ 的离散化 embedding）
- **SPATIAL token**：tokenizer 输出 $(64, 64)$ 被 spatial packing factor $k = 2$ 压成 $(32, 128)$，省一半 spatial attention cost
- **REGISTER token**（Darcet et al., https://arxiv.org/abs/2309.16588 ）：4 个 learnable sink token，吸收 attention 残余
- **AGENT token**：4 个，初始化自 per-task CLIP embedding（https://arxiv.org/abs/2103.00020 ）沿时间 broadcast；reward 和 BC head 通过 attention pooling 读取 agent token

**Block-causal structure**：(i) space self-attention over per-frame token；(ii) causal time self-attention along temporal axis；(iii) SiLU-gated MLP ratio 4。RoPE（Su et al., https://arxiv.org/abs/2104.09864 ）在 Q/K 上，KV-cache-aware offset；QK-norm（Henry et al.）；RMSNorm（Zhang & Sennrich, https://arxiv.org/abs/1910.07467 ）pre-norm，no bias。

**Modality-aware mask in dynamics**：action/shortcut/spatial/register 互相可见；agent token asymmetric——agent query 能看所有 token，但非 agent query 看不到 agent key。这避免了 reward head 的梯度污染 dynamics 的 representation。

### Shortcut flow matching（Frans et al., https://arxiv.org/abs/2410.12557 ）

这是 Dreamer 4 的核心训练 objective。Noise level 离散化 $\sigma \in \{0, \ldots, k_{\max}\}$，$k_{\max} = 64$（0 是纯噪声，64 是 clean）；step 索引 $\{0, \ldots, \log_2(k_{\max})\}$ 对应 $d = 1/2^{\text{step}}$。

Batch 中 **25% 走 self-consistency bootstrap**：在 $(\sigma, \text{step})$ 处，跑两个 no-grad 的 coarser half-step（step+1），它们的 averaged velocity 作为 stop-gradient target 训当前 step 的 velocity。剩下 75% 用 empirical one-step regression at finest step。

这个 trick 让 inference 时只需少量 Euler substep——paper 里用 $K = 8$ substep（$d = 0.125$），从 $z \sim \mathcal{N}(0, I)$ 出发，$b = (\hat{x}_1 - z)/(1 - \sigma)$，$z \leftarrow z + b \cdot d$。

### Reward head & BC policy

- Reward head：predict $L = 8$ multi-step symlog two-hot 分布，255 bin on $[-10, 10]$，梯度回传进 dynamics model
- BC policy：deterministic Gaussian，diagonal cov，MSE loss 拟合 ground-truth 16-dim padded action（这是与原 Dreamer 4 的偏离，后者只考虑 discrete action）

---

## Hallucination 的三分法：把抽象失败钉死在 pipeline 的具体 stage

这是 paper 最有教学价值的部分。一个 generative world model "想象"未来要串联三步：(1) encoder 把 observation 映到 latent；(2) dynamics head 在 action 条件下预测 next latent；(3) decoder 渲回 pixel。每个 stage 都是有限数据训练的 learned function，所以都可能 extrapolate 失败。**关键洞见：因为三步 sequential compose，早期 stage 引入的 hallucination 会被后续 stage propagate 和 amplify**——所以必须先定位是哪个 stage 的问题。

### (i) Perceptual hallucination（tokenizer 层）

定义：tokenizer 对一个 observation 的 reconstruction 在 dynamics rollout 之前就已经偏离了 observation 本身。具体机制：encoder-decoder 把 OOD scene structure project 到最近 in-distribution exemplar。例子：unseen maze layout 被重建时 agent/goal 位置正确，但 walls 来自训练时见过的完全不同 layout。然后 dynamics head 拿着这个 corrupted scene rollout，还以为它是 ground truth。

**这个 failure 是 frozen encoder-decoder pair 的固有属性，即使 horizon $H = 0$ 也存在**。这是个很 sharp 的判据——你只需检查单帧 reconstruction 就能 catch 它，不需要 rollout。

### (ii) Action-marginalized hallucination（dynamics 层）

定义：给定 context，predicted next latent 对 input action **几乎不敏感**。rollout 视觉上 plausible，但 collapse 到 action-marginalized future——model 表现得像 video generator 而非 controllable world model。

操作化检测：evaluation 时对 action stream 做 intervention，比如 batch 内 shuffle action，测 flow MSE 变化。如果 shuffle 后 flow MSE 几乎不动，说明 action 被 ignored。

这个 mode 的危险在于它**视觉上完全 plausible**，所以 PSNR、LPIPS 这类 pixel-level metric 抓不住它。你需要 explicitly probe action sensitivity。

### (iii) Scene-diverging hallucination（multi-step rollout 层）

定义：autoregressive rollout 累积 compounding error 是众所周知的（Janner et al. MBPO, https://arxiv.org/abs/1906.08253 ），但 scene-diverging 是个**特定的极端 mode**——预测出物理上不合理的事件，比如 Pong 里球得分后 teleport 回场地。这种 hallucination 在 data coverage 差的状态最频繁。

三个 mode 探查 pipeline 的 disjoint 部分：tokenizer / action-conditioning / multi-step error accumulation。这个分解的优雅之处在于，它把"hallucination"这个模糊概念拆成了可操作的诊断对象。

---

## 三个 Predictor：从 model 内部信号检测 hallucination

这是 paper 的核心贡献之一。三个 predictor 机制上不同，但都强预测 hallucination——作者把这个 convergence 当作 feature：三种独立机制同时指向同一信号，本身就是该信号真实存在的证据。

### Predictor 1: Tokenizer round-trip residual $u_r$

$$u_r := \|\hat{z} - \text{Encode}(\text{Decode}(\hat{z}))\|$$

变量含义：
- $\hat{z}$：dynamics 预测的 next latent
- $\text{Decode}(\hat{z})$：把预测 latent 渲回 pixel
- $\text{Encode}(\cdot)$：再 encode 回 latent
- $u_r$：单次 decode-encode round-trip 的 latent-space 残差

**直觉**：如果 $\hat{z}$ decode 后落在 tokenizer manifold 外（corrupted layout、fabricated object），那再 encode 回来跟 $\hat{z}$ 对不上，残差大。这直接对应 perceptual hallucination 的定义症状。注意，这是个**无需 label、无需额外训练**的 signal——你只需要在推理时多跑一次 encode-decode round-trip。

### Predictor 2: Flow instability $u_f$

衡量 dynamics head 在给定 (context, action) 下，denoiser 的 clean-target prediction $\hat{x}_1$ 在 successive Euler integration substep 之间的移动幅度，取后半段 substep 的平均。

**直觉**：sharp、well-conditioned 的 dynamics head 会快速收敛到稳定 $\hat{x}_1$（低 $u_f$）；conditioning 信号弱的 head 会在 substep 间持续 oscillate（高 $u_f$）。这捕捉的是 dynamics head 内部的不确定性，直接对应 action-marginalized hallucination。

### Predictor 3: Inter-seed variance $u_s$

固定 (past, action)，跑 $N$ 条 independent denoising trajectory（不同 noise seed），测 next-latent prediction 的 inter-seed variance。

**直觉**：这是 epistemic uncertainty 的经典度量（deep ensemble 思路，Lakshminarayanan et al. https://arxiv.org/abs/1612.01474 ；MC dropout 思路，Gal & Ghahramani https://arxiv.org/abs/1506.02142 ）。seed 间不一致的区域正是 multi-step rollout 会 fan-out 的区域，直接对应 scene-diverging hallucination。

### 关键工程细节：dynamism normalization

naive 用这三个 signal 会被 scene activity confound——high-motion transition 会同时拉高 $u_r$、$u_f$、$u_s$。所以作者用 normalized 版本：

$$u^{\text{norm}} := \frac{u}{m}$$

其中 $m$ 是 per-step latent 表示的 RMS change（per-task average over dataset，或 online 时用 running estimate）。这个 normalization 让每个 predictor 追踪的是**相对于 scene 中发生了多少事的不确定性**。

Table 7 验证了 normalization 的价值：$u_f^{\text{norm}}$ 在 scene-divergent 上的 AUROC 是 0.939，而 raw $u_f$ 只有 0.854；scene motion baseline $m$ 单独是 0.927（看似不错但其实混淆了 motion 和 uncertainty）。

### 整体检测性能

Figure 5：在 9k held-out 24-frame sequence 上，三个 predictor 与 open-loop rollout ∆PSNR 的 Spearman 相关 $\rho \approx -0.80$（强负相关，predictor 高 → rollout 质量差）。Table 7：对两个 binary hallucination label（action ignored: shuffle ratio $\leq 1.1$；scene divergent: rollout ∆PSNR $\leq 0$ vs frame-repeating baseline）的 AUROC：

| Predictor | Action ignored | Scene divergent |
|---|---|---|
| $u_r^{\text{norm}}$ | 0.887 | 0.919 |
| $u_f^{\text{norm}}$ | 0.868 | 0.939 |
| $u_s^{\text{norm}}$ | 0.873 | 0.934 |
| latent motion $m$ | 0.803 | 0.927 |
| kNN distance (global) | 0.814 | 0.731 |
| raw $u_f$ | 0.752 | 0.854 |
| $n_{\text{frames}}$ baseline | 0.596 | 0.534 |

---

## 两种 Mitigation：把 coverage lens 转化为行动

### Coverage-aware training（offline，零额外数据成本）

逻辑很直接：既然三个 failure mode 机制上都是 coverage gap，那一个 reweighting 就应该同时改善所有三个 signal。具体做法：把 sampling 从 uniform across **frames** 改成 uniform across **tasks**。也试过 loss reweighting 但 sampling 更好。

Table 1 显示，在 200 个 training task 上的 mean 改善（vs base model）：

| Metric | Tok FT | Dyn FT | Both |
|---|---|---|---|
| Recon PSNR (dB) ↑ | +0.46 | -0.01 | +0.44 |
| Action-shuffle ratio ↑ | +0.02 | +0.27 | +0.29 |
| Rollout ∆PSNR (dB) ↑ | +0.42 | +0.68 | +0.88 |
| $u_r^{\text{norm}}$ ↓ | -0.07 | -0.16 | -0.20 |
| $u_f^{\text{norm}}$ ↓ | -0.03 | -0.06 | -0.07 |
| $u_s^{\text{norm}}$ ↓ | -0.06 | -0.13 | -0.14 |

观察：tokenizer 和 dynamics 都能从 coverage-aware training 受益，且作用在不同 metric 上——Tok FT 主要改 Recon PSNR 和 $u_r$，Dyn FT 主要改 action-shuffle 和 rollout ∆PSNR。Both 同时 FT 30k step 后所有 metric 都改善。

### Targeted data collection（online，用 predictor 作 curiosity reward）

当现有 dataset 在某区域 coverage 不够时，reweighting 也救不了——需要新数据。这时 predictor 本身就能当 objective。思路传承自 Plan2Explore（Sekar et al., https://arxiv.org/abs/2007.00154 ）和 RND（Burda et al., https://arxiv.org/abs/1810.12894 ），但区别是：prior work 用 ensemble disagreement 或辅助网络做 single-task exploration 的 reward，本文直接复用 world model 内部 signal 跨 multi-task 数据收集。

具体流程：与 live env 交互时，候选 trajectory 先在 world model 里 rollout（horizon $H = 32$，replan every $K = 16$），按 predicted hallucination 打分，最高分 trajectory 在真实 env 执行，于是**by construction** 收集的就是之前让 model hallucinate 的 transition。

### Table 2：unseen task 上的 finetuning（10 task × 50 traj）

| Method | Recon PSNR ↑ | Rollout ∆PSNR ↑ | Action shuf. ↑ | $u_r^{\text{norm}}$ ↓ | Task perf. (MPC) ↑ |
|---|---|---|---|---|---|
| Random policy (lower bound) | - | - | - | - | 0.118 |
| Base (zero-shot) | 17.37 | -12.44 | 1.12 | 3.860 | 0.276 |
| Coverage-aware (zero-shot) | 17.21 | -12.52 | 1.29 | 3.769 | 0.276 |
| No-op, Tok FT | 34.74 | +0.66 | 1.55 | 1.486 | 0.163 |
| Random policy, Tok+Dyn FT | 35.81 | +2.66 | 2.00 | 1.201 | 0.228 |
| Expert policy, Tok+Dyn FT | 35.86 | +2.84 | 2.04 | 1.131 | 0.362 |
| Human play, Tok+Dyn FT | 37.11 | +3.89 | 2.42 | 1.002 | 0.362 |
| **Curiosity ($u_r^{\text{norm}}$), Tok+Dyn FT** | 36.05 | +3.00 | 2.00 | 1.144 | 0.325 |
| All (combined) | 37.91 | +4.02 | 2.34 | 0.975 | 0.390 |

几个关键 takeaways：

1. **Zero-shot 转移存在但有限**：base model 在 unseen task 上 0.276，是 random policy baseline (0.118) 的 2.3×——预训练确实学到了某种通用 dynamics prior，但 rollout ∆PSNR 是 -12.44 dB，说明 scene 严重 divergent。

2. **Tokenizer finetuning 是 unseen task 的关键瓶颈**：注意 "No-op, no Tok FT" 的 Recon PSNR 只有 17.21，几乎没提升；一旦 Tok FT，Recon PSNR 跳到 34.74。这印证了 perceptual hallucination 是 unseen task 的 first wall——你连单帧都重建不对，谈 dynamics 没意义。

3. **Curiosity 接近 expert/human oracle**：50 条 $u_r^{\text{norm}}$-driven curiosity trajectory 拿到 0.325，是 expert/human (0.362) 的约 90%，**且不使用任何 privileged behavior**。这是个很有说服力的结果——curiosity 自己找的 data 几乎跟 expert data 一样好。

4. **Combined (all data source) 最好**：0.390，说明不同 data source 覆盖的 coverage gap 互补。这与 author 的 hypothesis 一致：no single policy 覆盖所有 gap。

### Table 3：与 off-the-shelf tokenizer 对比

这是很实际的 ablation：能不能直接用大数据集预训练的现成 tokenizer 解决 perceptual hallucination？

| Tokenizer | Params | Latent/frame | Seen PSNR | Unseen PSNR | $\Delta_{S-U}$ | Seen LPIPS | Unseen LPIPS |
|---|---|---|---|---|---|---|---|
| Ours (base) | 102M | 4096 | 38.29 | 17.34 | +20.95 | 0.011 | 0.389 |
| Ours (coverage-aware) | 102M | 4096 | 38.93 | 17.12 | +21.81 | 0.008 | 0.348 |
| Ours (post-FT) | 102M | 4096 | 39.66 | 38.04 | +1.62 | 0.007 | 0.010 |
| SD-VAE-MSE | 84M | 3136 | 33.32 | 32.39 | +0.93 | 0.031 | 0.030 |
| Cosmos-CV8x8x8 | 106M | 2048 | 32.80 | 32.72 | +0.08 | 0.050 | 0.042 |
| Wan 2.1 VAE (https://arxiv.org/abs/2503.20314 ) | 127M | 4096 | 36.45 | 36.62 | -0.17 | 0.010 | 0.010 |
| DC-AE-f32c32 | 323M | 2048 | 31.49 | 32.15 | -0.66 | 0.035 | 0.031 |

观察：
- In-domain tokenizer 在 seen task 上吊打所有 off-the-shelf（38.29 vs Wan 2.1 的 36.45）
- 但 unseen task 上 in-domain base 只有 17.34，Wan 2.1 是 36.62——off-the-shelf 的 generalization 来自训练数据规模几个数量级更大
- **关键转折**：in-domain post-FT 后 unseen PSNR 跳到 38.04，反超 Wan 2.1。这证明 in-domain finetune 仍有 tangible benefit，但你需要先有一份 in-domain pretrain 才能 finetune

$\Delta_{S-U}$ 这一列很说明问题：in-domain base 的 seen-unseen gap 是 +20.95 dB（严重 overfit），off-the-shelf 是 ~0（uniform generalization），post-FT 降到 +1.62（gap 几乎弥合）。

---

## 评估指标的设计哲学

paper 的 evaluation 设计值得单独拎出来讲，因为它直接对应三种 hallucination mode：

1. **Reconstruction PSNR ↑**：纯 tokenizer 质量，无 dynamics。对应 perceptual hallucination 的"上界"——你重建不好肯定 hallucinate。

2. **Rollout PSNR gain (dB) ↑**：生成的 rollout 相对于 "repeat last frame" baseline 的 PSNR gain。这个 baseline 看似 naive 但在某些 task（静止场景）惊人地强。判据：$\Delta\text{PSNR} \leq 0$ 即 scene divergent。这个 metric 巧妙地避免了"绝对 PSNR 高但其实是静止"的 false positive。

3. **Action shuffle ratio ↑**：one-step teacher-forced flow MSE（正常 action）vs batch-shuffled action 的 MSE 比值。判据：$\leq 1.1$ 即 action ignored。这是个非常聪明的 intervention-based metric——它不直接测"预测对不对"，而测"预测对 action 敏不敏感"。

4. **Downstream task performance (normalized score)**：用 CEM (https://www.cs.cmu.edu/~m81/courses/mlspr2014/reading/CEM-and-CMAES.pdf ) 做闭环 MPC，horizon $H = 32$，replan every $K = 16$。这是终极 test——world model 好不好最终看它能不能 drive 一个 planner 完成任务。normalized 到 $[0, 1]$ 是因为 reward scale 跨 task 差几个数量级。

CEM 配置：3 iteration，population 32，每个 candidate 2 rollout，warm start mean 用 BC prior。

---

## 几个值得深挖的细节

### 1. Reward finetune 的副作用（Table 11）

作者做了一个对照：pretrained base + 30k dynamics step，一个 joint train reward head（梯度回传进 dynamics），一个 reward-free control。结果几乎所有 metric 无显著差异（Recon PSNR +0.01, action-shuffle -0.05, rollout ∆PSNR +0.13）。这暗示：**在已有强 action-conditioning 的 base 上，reward signal 不额外塑形 dynamics representation**。这与 Dreamer 系列一贯主张 "reward 是 dynamics 上的 readout" 一致，但 paper 没深入展开，是个潜在 future work 方向。

### 2. Context corruption $\tau_{\text{ctx}} = 0.1$

训练时对 past token 加 10% corruption——这是 regularization，防 model 过度依赖 context 的精确值，相当于 action robustness training 的时间版。这对 rollout 长期稳定性应该有贡献，但 paper 没单独 ablate。

### 3. Spatial packing 的 trade-off

$(64, 64) \to (32, 128)$ 把 spatial attention cost 砍半，channel 翻倍。这是个工程 trade-off：vision transformer 里 spatial locality 强，packing 损失小；但若 task 有强 long-range spatial 依赖（比如迷宫两端的目标），packing 可能伤精度。paper 没在 maze 类 task 单独评估 packing 影响，值得后续探。

### 4. Heavy-tail 数据分布的影响

Figure 3 揭示的 per-task frame 分布严重 heavy-tail。Coverage-aware training 用 uniform-task sampling 直接对抗这个 skew。但更深的问题：**top 20 task（多为 Atari，1000 step/ep）的 visual diversity 本身就高**，所以即使 frame 多，coverage 也未必饱和；而 bottom 20 task（ManiSkill3 短 manipulation）frame 少但 visual diversity 低，可能 coverage 反而够。Paper 的 uniform-task reweight 没考虑 task 内部 diversity，是个可能的改进方向——可以按 predictor 在 task 内的 variance 来加权。

### 5. AUROC 的不对称性

Table 7 里 $u_r^{\text{norm}}$ 在 action-ignored 上 0.887，比 $u_s^{\text{norm}}$ 的 0.873 略高；但 scene-divergent 上 $u_f^{\text{norm}}$ 0.939 略胜 $u_r^{\text{norm}}$ 0.919。这暗示每个 predictor 虽然都能用，但有"主场"：$u_r$ 更擅长 perceptual，$u_f$ 更擅长 dynamics instability。author 选 $u_r^{\text{norm}}$ 做 curiosity 是因为 perceptual 是 unseen task 的 first wall（Table 2 验证）。

### 6. Curiosity 的规划 horizon

H = 32, replan K = 16，意味着 curiosity agent 在 world model 里想象 32 步未来，每 16 步重新决策。这个 horizon 足够长以发现远端 hallucination，又足够短避免世界模型自己的 hallucination 把 curiosity agent 带偏（想象一下：如果 model hallucinate 出一个假的高 reward 区域，curiosity agent 会一头扎进去收集无意义 data）。Paper 没讨论这个 "world model hallucinate 引导 curiosity 走偏" 的 failure mode，是个潜在 circular dependency。

---

## 局限与 open question

作者自己 acknowledge：
- **Scale**：350M 参数、210 个 simulated task。能否 translate 到 billion-parameter model 和 real robot data（sensor noise、partial observability）是 open empirical question。DROID（https://arxiv.org/abs/2403.12945 ）和 Open X-Embodiment（https://arxiv.org/abs/2310.08864 ）上的验证会很有说服力。
- **Compute cost**：58 GPU day for final checkpoint，不算 extreme 但也不轻。
- **Data diversity**：210 task 跨 10 domain 在 RL benchmark 里算广，但与 LAION-5B（https://arxiv.org/abs/2210.08414 ）这种 vision pretrain scale 比还差好几个数量级。

我自己想补充几个：
1. **Hallucination 的下游传播**：paper 只测了 MPC 的 task performance，但没量化 "hallucination 频率 → planner 决策错误率" 的因果链。如果 hallucination 集中在低 coverage 区域，而 planner 主要在高 coverage 区域决策，那 hallucination 对下游的实际影响可能被高估。反之亦然。一个 "hallucination-weighted downstream risk" metric 会更有说服力。
2. **Action-marginalization 与 representation collapse 的关系**：action 被 ignore 可能是 representation 把 action 信息丢了，也可能是 dynamics head 没用上。区分这两者需要看 latent 的 action-information content，比如 mutual information $I(z; a | s)$。Paper 没做这个分析。
3. **Perceptual hallucination 的 manifold 几何**：作者说 OOD scene 被 project 到最近 in-distribution exemplar，这听起来像 VAE 的 posterior collapse 或 mode covering vs mode seeking 的经典问题（见 Oord 的 VQ-VAE, https://arxiv.org/abs/1711.00937 ）。 tokenizer 用 continuous bottleneck + tanh 而非 VQ，是否影响 hallucination 模式？VQ 的 codebook 是离散的，project 行为会更尖锐；continuous 的 project 更平滑。这值得 ablate。
4. **Shortcut flow matching 的 stability metric 是否最优**：$u_f$ 取后半段 substep 的平均 velocity change，但为什么是后半段？前半段是高噪声区，velocity 本来就大且不稳定；后半段接近 clean，velocity 应该收敛。如果在前半段都 instability，说明 conditioning 完全没起作用；后半段 instability 说明 fine-grain 不确定性。Paper 选后半段是合理，但 ablate 不同 substep window 会有趣。

---

## 与相关工作的 positioning

- **vs Dreamer 3/4 (Hafner et al.)**：本文直接 reproduce Dreamer 4，但 Dreamer 4 没系统分析 hallucination mode，也没提 predictor。本文的 contribution 是"诊断 + 治疗框架"，Dreamer 4 是"架构 + 训练 recipe"。
- **vs DIAMOND (Alonso et al., https://arxiv.org/abs/2405.05951 )**：DIAMOND 用 diffusion world model 打 Atari，发现 visual detail matter。本文 hallucination 概念涵盖 DIAMOND 关注的细节失真，但 framework 更广。
- **vs GameNGen (Valevski et al., https://arxiv.org/abs/2408.14837 )**：GameNGen 是 playable DOOM，重点是 fidelity 和 latency。本文重点是可控性 + generalization 失败模式。
- **vs Genie (Bruce et al., https://arxiv.org/abs/2402.19409 ) / Genie 3**：Genie 用 latent action model 处理 unlabeled video，本文用 ground-truth action，研究的是 action-controllable 而非 action-discoverable 的 setting。
- **vs Plan2Explore (Sekar et al., https://arxiv.org/abs/2007.00154 )**：curiosity-driven exploration 的鼻祖，用 ensemble disagreement。本文用 world model 内部 predictor 替代 ensemble，更高效，且目标是数据收集而非 single-task exploration。
- **vs NitroGen (Magne et al., https://arxiv.org/abs/2601.02427 )**：generalist gaming agent，1000+ game，4B frame，但 action 是 pseudo-label。本文 MMBench2 是 ground-truth action + reward + live env，是 NitroGen 想要但还没提供的。
- **vs Cosmos (NVIDIA, https://arxiv.org/abs/2501.03575 )**：physical AI foundation model，scale 大但 hallucination 分析少。本文框架原则上适用于 Cosmos 类大模型。

---

## 对你的 intuition building 价值

如果你（Karpathy）从 training large neural network 的角度想，这篇 paper 的核心 insight 可以总结成一句话：**hallucination 不是 model 不够大，是 data 不够广**。这个 framing 跟 Chinchilla（https://arxiv.org/abs/2203.15555 ）和 DataComp（https://arxiv.org/abs/2304.14108 ）的"数据是第一性杠杆"一脉相承，但把它专门化到了 world model 的 sequential pipeline 上。

更深一层：paper 揭示了 **sequential composition 是 hallucination propagation 的放大器**。一个 tokenizer 的小 perceptual error 会经过 dynamics head 的多步 rollout 被放大成 scene divergence。这意味着，与 LM 不同，world model 的 hallucination 不只是"生成错了一个 token"，而是"前面错了一帧，后面整条轨迹都跟着错"。这种 amplification 让早期 stage 的 coverage gap 影响远大于后期 stage。

这跟 LM 中的 hallucination 有本质区别——LM 的 hallucination 是 autoregressive 在 token 空间累积，每个 token 都是同一类操作；world model 的 hallucination 是 cross-stage（tokenizer → dynamics → decoder），stage 之间 representation 完全不同。所以"哪一个 stage 出问题"在 world model 里是 first-class question，在 LM 里相对没那么 sharp。

对实际 build agent / world model 的人，paper 的可操作建议：
1. **先确保 tokenizer 在 target domain 重建好**，否则下游全白搭
2. **用 uniform-task 而非 uniform-frame sampling**，这个改动几乎免费就能拉所有 metric
3. **用 $u_r^{\text{norm}}$ 做 online curiosity**，能在没有 expert demo 的情况下逼近 expert-data finetune 效果
4. **用 action shuffle ratio 监控 dynamics 是否真用上了 action**，pixel PSNR 会骗你

这套方法学可以 transfer 到任何 action-conditioned generative model——robotics 的 diffusion policy（https://arxiv.org/abs/2303.04137 ）、video generation 的 action-conditioned model（https://arxiv.org/abs/2406.10931 ），甚至 LM 的 tool-use 场景（tool call 是否真被 marginalized？可用类似 shuffle intervention 测）。

---

## 参考链接汇总

- Paper project page: https://nicklashansen.com/mmbench2
- Dreamer 4: https://arxiv.org/abs/2509.24527
- TD-MPC2: https://arxiv.org/abs/2310.16828
- DIAMOND: https://arxiv.org/abs/2405.05951
- GameNGen: https://arxiv.org/abs/2408.14837
- Genie: https://arxiv.org/abs/2402.19409
- Plan2Explore: https://arxiv.org/abs/2007.00154
- RND: https://arxiv.org/abs/1810.12894
- Shortcut models: https://arxiv.org/abs/2410.12557
- MAE: https://arxiv.org/abs/2111.06377
- LPIPS: https://arxiv.org/abs/1801.03924
- Deep Ensembles: https://arxiv.org/abs/1612.01474
- MC Dropout: https://arxiv.org/abs/1506.02142
- CLIP: https://arxiv.org/abs/2103.00020
- RoPE: https://arxiv.org/abs/2104.09864
- RMSNorm: https://arxiv.org/abs/1910.07467
- Register tokens: https://arxiv.org/abs/2309.16588
- Chinchilla scaling: https://arxiv.org/abs/2203.15555
- DataComp: https://arxiv.org/abs/2304.14108
- OGBench: https://arxiv.org/abs/2410.20092
- DROID: https://arxiv.org/abs/2403.12945
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- VPT: https://arxiv.org/abs/2210.01537
- Cosmos: https://arxiv.org/abs/2501.03575
- Wan 2.1: https://arxiv.org/abs/2503.20314
- MBPO: https://arxiv.org/abs/1906.08253
- MuZero: https://www.nature.com/articles/s41586-020-03051-4
- Continuous Atari: https://arxiv.org/abs/2405.14960
- ManiSkill3: https://arxiv.org/abs/2410.00425
- DMControl: https://arxiv.org/abs/1801.00690
- Meta-World: https://arxiv.org/abs/1910.10897
- Hallucination survey (LM): https://arxiv.org/abs/2202.03629
- Object hallucination (VLM): https://arxiv.org/abs/2310.00754
- VBench (video): https://arxiv.org/abs/2311.13535
