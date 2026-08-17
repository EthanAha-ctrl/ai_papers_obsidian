---
source_pdf: JEDI Joint Embedding Diffusion World Model for Online Model-Based Reinforcement
  Learning.pdf
paper_sha256: f78f2d417cbd156a859a204773e0efe010bf35b8174530c4fcacc5d7c1421c81
processed_at: '2026-08-05T10:50:16-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# JEDI 用人话讲

---

## 一句话说清楚

让 AI agent 在脑子里"想象未来"这件事,之前要么很贵 (直接在像素上想象),要么效果不行 (先把画面压成 latent 但压的方式跟"预测未来"无关)。JEDI 的做法是: **让"压成 latent"和"想象未来"这两件事变成同一个学习过程,一边想象一边学会怎么压缩。**

---

## 背景故事: RL agent 怎么"做梦"

RL agent 要学会打游戏,最笨的办法是疯狂试错。聪明一点的做法是: 先学一个"世界模型" (world model),这模型能预测"如果我现在在这个画面,按了 FIRE,下一帧会变成什么样"。有了这个模型,agent 就能在脑子里 rollout: "如果按这序列动作,游戏大概会这样发展",然后从这些 imagined trajectories 里学 policy。这就是 model-based RL (MBRL) 的核心思路,从 Sutton 1991 的 Dyna 一直延续到 DreamerV3。

问题在于: "预测下一帧画面"这件事本身很贵。一个 64×64×3 的 RGB 画面有 12288 个数字,要让神经网络准确预测这 12288 个数字的下一帧值,模型容量要大、采样要慢、训练要久。**DIAMOND** 这个工作 (NeurIPS 2024) 证明了: 用 diffusion model 直接在 pixel space 预测下一帧,效果是 SOTA 的,但要 98 小时 A100 训一个 Atari100k run,采样也慢。

自然的想法是: 别在 pixel 上做,先把画面压成一个小的 latent vector,在 latent 上做 diffusion。这就是 **Horizon Imagination (HI)** 干的事。但 HI 的 latent 怎么来的? 用一个 separately trained VAE,loss 是 reconstruction + perceptual loss。换句话说,这个 latent 学的是"怎么重构画面",跟"预测未来"没关系。结果就是 HI 效率上去了 (27 小时),性能却掉下来了。

JEDI 的核心 insight: **latent space 的学习目标和 world model 的预测目标应该是一回事, 不应该分开。**

---

## JEPA 的直觉: 该保留什么, 该丢掉什么

LeCun 一直推的 JEPA 给了一个很干净的 inductive bias:

- 你想从画面 $x_1$ 预测画面 $x_2$
- 直接预测 pixel 太难, 而且很多 pixel 细节根本不可预测 (比如背景噪声、敌人 AI 的微小随机)
- 正确做法: 先把 $x_1$ 和 $x_2$ 都通过 encoder 压成 latent $z_1$ 和 $z_2$, 然后在 latent 空间做预测 $z_1 \to z_2$
- encoder 自动学会"丢掉不可预测的细节, 保留对预测有用的结构"

这就是 JEPA: Joint Embedding Predictive Architecture。两个 embedding (joint), 一个预测 (predictive)。

这跟 RL 想要的 latent 完全契合 — RL 的 latent 不需要能重构画面, 只需要能预测未来、reward、control 相关的东西。背景的云长什么样、地板纹理细节, 对控制没用的就该丢。

---

## 为什么 diffusion 当 JEPA predictor

JEPA 原本用一个简单的 predictor $p_\theta(z_2 | z_1)$, 通常 MSE loss。但 JEDI 把这一步换成了 **conditional diffusion denoising**。

为什么不直接 MSE? 三个直觉:

1. **Atari 看着 deterministic, 实际上状态转移经常 multimodal**。比如 DemonAttack 里敌人下一步往左还是右, 从当前 latent 看两个都合理。MSE 预测只能给出平均, 平均往往是 nonsense。Diffusion 通过 iterative denoising 自然 express multimodal distribution。

2. **Iterative refinement 是一个很强的 inductive bias**。从纯噪声一步步 denoise 到 clean latent, 这个过程强迫模型在 noise level 之间共享知识, 让 latent 学得更 smooth。

3. **Diffusion 在很多领域都被验证过** (image generation, video, policy, 甚至 language)。把它当 representation learner 用是顺势而为。

---

## 理论: 为什么这套训练 objective 合理

这是 paper 里我觉得最优雅的部分。他们证明了一个 correspondence:

> **Latent conditional diffusion denoising loss, 在变分视角下, 等价于一个 information bottleneck 目标的 prediction term。**

让我拆开讲。

### Information Bottleneck 是什么

经典 IB ([Tishby 2000](https://arxiv.org/abs/physics/0004057)): 给定 input $X$ 和 target $Y$, 找一个 representation $Z$, 最大化 $I(Z; Y)$ (保留预测 target 的信息), 同时最小化 $I(X; Z)$ (压缩 input 信息)。这就是"在压缩中预测"。

Deep VIB ([Alemi 2016](https://arxiv.org/abs/1612.00410)) 把这个变成可优化的变分目标。

### JEPA = 一种 IB

JEDI 论文证明 (Appendix A.1): 如果你写出 JEPA 的 KL loss 并做变分分解, 你会得到一个上界:

$$-\mathcal{L}_{\mathrm{JEPA}} \leq I(Z_1; Z_2) - \hat{I}(X_1; Z_1) - \hat{I}(X_2; Z_2) + \mathcal{R}_1 - \mathcal{G}$$

每一项什么意思:
- $I(Z_1; Z_2)$: 两个 latent 之间的互信息, 越大说明 $z_1$ 越能预测 $z_2$, 这是 **predictive term**
- $-\hat{I}(X_1; Z_1) - \hat{I}(X_2; Z_2)$: encoder 保留 input 信息越少越好, 这是 **compression term**
- $\mathcal{R}_1$: bottleneck regularizer, 防 collapse
- $\mathcal{G}$: posterior approximation gap, 越小越好

这就是 IB 的形式! JEPA 没显式说要压缩, 但它的 KL loss 隐式地鼓励压缩。

### Latent diffusion 也一样

JEDI 把 one-step predictor 换成 reverse diffusion chain $z_2^T \to z_2^{T-1} \to \cdots \to z_2^0$, condition 在 $z_1^0$ 上。这个 chain 诱导的 marginal $p(z_2^0 | z_1^0)$ 就是一个 stochastic JEPA predictor。

经过更复杂的变分推导 (Appendix A.2), 他们证明 latent diffusion denoising loss 也满足类似的 IB 上界:

$$-\mathcal{L}_{\mathrm{CDJ}}^{\mathrm{den}} \leq I(Z_1^0; Z_2^0) - \hat{I}(X_1; Z_1^0) - \hat{I}(X_2; Z_2^0) + \mathcal{R}_1 + \mathcal{R}_{2,T} - \mathcal{G} + \mathcal{C}_2$$

多了 $\mathcal{R}_{2,T}$ (terminal prior matching) 和 $\mathcal{C}_2$ (target encoder entropy, 在 stop-grad 下是 constant)。

**直觉**: 你训练 diffusion denoiser 让它从噪声 denoise 回 clean target latent, 这件事本身就在变分上鼓励 encoder 学到一个"既保留预测信息又压缩 input 信息"的 representation。所以 end-to-end 训练有理论 grounding, 不是 ad-hoc trick。

### 一个 caveat

JEDI 实际只优化了 prediction term, **没有显式优化** $\mathcal{R}_1$ (bottleneck regularizer)。所以严格说它是在优化 bottleneck 目标的"半边", compression 是隐式产生的。论文坦白承认这一点, 把显式优化 $\mathcal{R}_1$ 留给 future work。

这个 caveat 很重要, 因为 VICReg ([NeurIPS 2023](https://arxiv.org/abs/2306.12195)) 和 LeJEPA ([Balestriero & LeCun 2025](https://arxiv.org/abs/2511.08544)) 已经证明: 显式的 collapse-prevention regularizer 对稳定 JEPA 很关键。JEDI 靠 stop-gradient 和 0.3× LR 这种 trick 间接达到类似效果, 但理论上更干净的版本是直接加 VICReg-style regularizer。

---

## JEDI 的 architecture 干嘛

三个模块:

1. **Encoder $E_\phi$**: 把 $64×64×3$ 画面压成 $16×8×8$ 的 latent (1024 维, 比 pixel input 小 12 倍)。用 tanh clamp 到 $[-3, 3]$ 范围。
2. **Latent diffusion dynamics model $D_\theta$**: 给定过去 4 步 latents + actions, 从纯噪声 denoise 出下一时刻的 latent。EDM preconditioning + U-Net 单层无 downsampling。
3. **Reward/termination head $R_\psi$**: 从 latent 预测 reward 和 done flag。

训练时:
- 当前帧 $x_t$ 编码成 $z_t^0$
- 下一帧 $x_{t+1}$ 编码成 $z_{t+1}^0$ (stop-gradient)
- 给 $z_{t+1}^0$ 加噪声到 $z_{t+1}^\tau$
- Denoiser 学从 $z_{t+1}^\tau$ 加 condition $Z_t^\tau$ denoise 回 $z_{t+1}^0$
- Loss 的梯度通过 condition 中的 $z_t^0$ 流回 encoder, 让 encoder 也被训练

推理时:
- 当前 latent 从 encoder 来
- 下一时刻 latent 从纯噪声 + condition 通过 reverse diffusion 出来 (3 步 Euler sampling)
- 这个 latent 喂给 reward head 出 r, d
- 这个 latent 也喂给 actor-critic 出 action

---

## 三个关键 stabilization tricks

JEDI 能 work 不光是 idea 好, 还有三个 trick 必须有, 不然 latent diffusion 训不稳:

### Trick 1: Stop-gradient on target + asymmetric LR

Target latent $z_{t+1}^0$ 加 stop-gradient, 不让梯度流回去。同时 encoder LR 是 denoiser LR 的 0.3 倍。

为什么? 如果 encoder 通过 target 路径接收梯度, 它会学到"把所有 input 映射到 trivial constant", 因为 constant 让 KL 最小。Stop-gradient 切断这条路径, encoder 只能从 condition 路径接收"怎么让 denoising 容易"的信号。

0.3× LR 让 encoder 学得慢, denoiser 学得快。Denoiser 先适配当前 encoder, 再推动 encoder 改进。这是 BYOL / I-JEPA / TD-MPC2 一脉相承的实践经验。

### Trick 2: tanh clamping

$$C(z) = \tanh(z/s) \cdot s, \quad s = 3$$

把 latent 严格限制在 $[-3, 3]$。

为什么? Pixel space 里数值天然在 $[-1, 1]$, 但 latent space 没有自然 bound, 训练中可能 drift 到 $[-100, 100]$, 让 reverse diffusion 的 ODE solver 数值爆炸。tanh 给一个软的 upper bound, 既可微又稳定。DIAMOND 也 clamp (到 256 bins), HI 也用 tanh, 这是 latent diffusion 通用的 trick。

### Trick 3: Random switching between denoiser output and encoder output

每 batch 以 50% 概率随机选:
- 用 denoiser 输出 $\hat{z}_{t+1}^0$ 作为下一步 condition (这模拟 imagination rollout)
- 用 encoder 输出 $z_{t+1}^0$ 作为下一步 condition (这直接用真 observation)

为什么? 如果永远用 encoder 输出, denoiser 学不到 multi-step rollout 累积误差; 如果永远用 denoiser 输出, encoder 没有直接 supervision 容易 drift。Random switching 同时让两边都受到训练。

---

## 实验结果: 关键数字

### Atari100k 主战场

| Method | Mean HNS | IQM | Optimality Gap | Har-HNS | #SOTA |
|---|---|---|---|---|---|
| DIAMOND | **1.621** | 0.609 | 0.480 | 0.319 | 6 |
| JEDI | 1.450 | **0.688** | **0.460** | **0.377** | 7 |

IQM (interquartile mean) 和 Optimality Gap JEDI 都 SOTA。Mean HNS 略低于 DIAMOND。但论文新提的 Har-HNS (harmonic mean of HNS) JEDI 显著高。

Har-HNS 公式:
$$\text{Har-HNS} = \left(\frac{1}{N} \sum_{i=1}^N \frac{1}{\text{HNS}_i + 0.1}\right)^{-1}$$

为什么这个指标? Arithmetic mean 偏向高分游戏, 一个游戏从 HNS 1.5 升到 2.0 贡献 0.5, 从 0.01 升到 0.05 贡献 0.04。但相对提升前者 1.33×, 后者 5×。Har-HNS 对低分任务更敏感, 反映"最难的任务上做到多好"。

### 效率提升 vs DIAMOND

- **VRAM 减 43%**: 12× 体积缩减带来的
- **World-model sampling 快 3×+**
- **Training 快 2.5×+** (38 hrs vs 98 hrs on A100)
- **Parameters 相同** (13.5M)

### Stochastic Atari

加了 random frame skip 2-6 制造 aleatoric uncertainty。JEDI 显著超过 DreamerV3, 证明 diffusion 对 stochastic target 更鲁棒。

### Craftium (3D Minecraft-like)

4 个 tasks × 3 seeds, JEDI 整体优于 HI。证明方法能迁移到 3D 视觉环境。

---

## 最有意思的发现: Performance Profile 完全不同

JEDI 不是"平均比 DIAMOND 好一点", 而是 **在不同的游戏上有完全不同的强弱分布**。

具体:
- JEDI 优势集中在 **large action space + shooter-style + low-HNS** 游戏 (BankHeist, DemonAttack, Hero, BattleZone, ChopperCommand, Alien)
- DIAMOND 优势集中在 **small action space + high-HNS** 游戏

为什么? 论文的 hypothesis:

DIAMOND 把 raw 64×64×3 (12288 维) 直接喂给 actor-critic。JEDI 把 16×8×8 (1024 维) learned latent 喂给 actor-critic。

在 large action space (18 个 action) + 复杂视觉游戏上, actor-critic 要同时学 representation 和 policy/value。固定 compute budget 下, 两边都学不好。JEDI 把 representation 学习"外包"给 world model (latent 已经是 predictive abstraction), actor-critic 只学 policy/value。Input 维度 12× 减小, 加上 action space 大, effective 搜索空间减小得更显著。

Intuition: RL 的 effective difficulty $\propto \text{input\_dim} \times \text{action\_dim} \times \text{horizon}$。Input 维度减小对大 action space 游戏收益更大, 因为 combinatorial 结构更显著。

⚠️ 论文坦白这是 plausible explanation, 不是形式化证明。但这个发现本身很有意思 — **representation 形式改变了 RL agent 的"游戏"本身, 不只是效率与精度的 trade-off**。

---

## Ablation 验证了关键 claim

### Latent learning ablation (Figure 9)

五个变体从好到差:
1. Full JEDI (diffusion loss end-to-end)
2. MSE Loss (diffusion 换成 direct MSE)
3. - Diff Grad (去掉 diffusion loss 对 encoder 的梯度)
4. + Decoder Grad (加 reconstruction supervision)
5. AutoEncoder (separately trained VAE latent)

这个 ranking 直接验证核心 claim: **latent 必须从 predictive objective end-to-end 学, diffusion 比 MSE 好, reconstruction supervision 反而有害**。

### Design choice ablation (Figure 10)

去掉 EMA target / 去掉 random switching / 去掉 clamping 都变差, 但 HNS 仍较高。说明 **stabilization tricks 是 helpful but not critical, end-to-end predictive objective 才是 main ingredient**。

---

## 局限

- **Benchmark 局限**: 只有 Atari100k (5 seeds) + 4 个 Craftium tasks (3 seeds)。DMControl, Procgen, 高分辨率都没测。
- **不是最快 latent 方法**: HI 27 小时 vs JEDI 38 小时, 但 HI 用 97M params + pretrained perceptual model, JEDI 13.5M 完全 end-to-end。
- **理论是 motivation 不是保证**: $\mathcal{R}_1$ 没显式优化, 优化收敛性没证。
- **Batch size 敏感**: 每个 task 报 batch 32 或 64 中较好的, 类似 HI 做法。

---

## 我的核心 intuition

1. **真正关键的 idea 就一句**: 用 diffusion denoising loss 当 JEPA predictor, end-to-end 训 encoder, 不需要 reconstruction。其他都是 stabilization 和理论 grounding。

2. **End-to-end 是核心**: separately trained latent 学的是"重构画面", 这跟 RL 控制未必相关甚至有害。End-to-end 学的是"什么信息对预测未来+reward+控制有用"。这个 intuition 跟 TD-MPC2 的成功完全一致。

3. **Diffusion 比 MSE 强**的真正原因可能是 multimodal target + iterative refinement regularization, 不只是表达能力强。

4. **Stop-gradient 是 JEPA work 的隐式功臣**: 没有它, encoder 会 collapse 到 trivial constant。这个 trick 的深度意义还没被完全理解, 但经验上反复验证有效。

5. **Performance profile 改变这件事最深刻**: representation 形式不只是 efficiency 问题, 是 RL agent "看到的世界"的问题。这跟 LLM 里 tokenization 改变 inductive bias 类似 — input space 决定 policy 学习的 effective horizon。

6. **未解决的 obvious next steps**:
   - 加 VICReg-style 显式 $\mathcal{R}_1$ 优化
   - 用 rectified flow / consistency model 把 diffusion 压成 1 step sampling, 看能不能保留 JEPA signal
   - 在 DMControl 上测, 因为 DMC 是连续 action + low-dim observation, JEDI 的"input 12× 减小"优势可能消失, performance profile 可能反转
   - 用 score-based policy gradient 利用 world model 的 score function 做更好的 policy optimization, 而不是 REINFORCE 完全切断梯度

---

## 相关链接

- [JEDI 论文 (本次讨论)](https://arxiv.org/abs/2410.21050)
- [DIAMOND (NeurIPS 2024)](https://arxiv.org/abs/2403.04134)
- [Horizon Imagination (HI)](https://arxiv.org/abs/2602.08032)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [TD-MPC2](https://arxiv.org/abs/2310.16828)
- [I-JEPA (CVPR 2023)](https://arxiv.org/abs/2301.08243)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [LeJEPA](https://arxiv.org/abs/2511.08544)
- [LeCun JEPA position paper](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- [Deep Variational Information Bottleneck](https://arxiv.org/abs/1612.00410)
- [Information Bottleneck (Tishby)](https://arxiv.org/abs/physics/0004057)
- [VICReg info-theory view](https://arxiv.org/abs/2306.12195)
- [DDPM](https://arxiv.org/abs/2006.11239)
- [EDM](https://arxiv.org/abs/2206.00364)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [Genie 2 (DeepMind)](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)
- [World Models (Ha & Schmidhuber)](https://arxiv.org/abs/1803.10122)
- [Atari100k benchmark](https://arxiv.org/abs/1903.00374)
- [Craftium](https://arxiv.org/abs/2407.03969)

---

一句话总结: JEDI 把 JEPA 的"predictive compression"和 diffusion 的"iterative denoising"缝起来, 证明 latent diffusion world model 可以完全 end-to-end 训, 不需要 reconstruction 或 pretrained model, 效率比 pixel diffusion 高 2-3×, 且在难任务上表现完全不同。

---

# JEDI: Joint Embedding Diffusion World Model for Online Model-Based Reinforcement Learning

让我系统讲解这篇 paper。这是由 National University of Singapore 团队 (Jing Yu Lim, Dianbo Liu 等) 完成的工作,定位是 **第一个 online end-to-end latent diffusion world model**,融合了 JEPA 的 inductive bias 与 diffusion 的 iterative refinement。

---

## 1. 问题动机

当前 online model-based RL (MBRL) 中的 diffusion world model 处于一个尴尬的张力:

- **DIAMOND** ([arXiv NeurIPS 2024](https://arxiv.org/abs/2403.04134)) 走 pixel-space diffusion 路线,效果强,但在 $\mathbb{R}^{64 \times 64 \times 3} = 12288$ 维上做 denoising,内存、采样时间、训练时间都很昂贵。
- **Horizon Imagination (HI)** ([arXiv 2026](https://arxiv.org/abs/2602.08032)) 移到 latent space,效率提升,但 latent 是用 separately trained perceptual loss (VAE + 重构) 学的,**不是 end-to-end 从 world-model objective 学出来的**,因此性能不及 DIAMOND。

这给了一个非常清晰的机会窗口: **如果 MBRL 过去 5 年的进步很大程度上来自 end-to-end representation learning (DreamerV3 [arXiv 2301.04104](https://arxiv.org/abs/2301.04104), TD-MPC2 [arXiv 2310.16828](https://arxiv.org/abs/2310.16828)), 那么 latent diffusion 也应该能从同样的原则中受益。** 同时,JEPA ([LeCun 2022 position paper](https://openreview.net/pdf?id=BZ5a1r-kVsf)) 风格的 predictive representation learning 正好提供了一个天然的、避免 reconstruction 的、保留"可预测结构"丢掉"nuisance variation"的 inductive bias。JEDI 的核心 insight 就是把这两个 idea 缝合: **用 conditional diffusion denoising 来当 JEPA 中的 predictor,并让 encoder 直接通过 denoising loss end-to-end 学习 latent space,完全不需要 reconstruction 或 pretrained model**。

---

## 2. 理论核心: JEPA 作为 Variational Information Bottleneck

### 2.1 PGM 与变分目标

JEDI 假设的 probabilistic graphical model (Figure 2):

$$X_1 \to Z_1 \to Z_2 \to X_2$$

这里 $X_1, X_2$ 是 observations (在 MBRL 中是连续的 observation,忽略 action 以简化), $Z_1, Z_2$ 是它们的 latent representation。在 SSL 的视角里这就是两个 view; 在 MBRL 的视角里这是时间 $t$ 与 $t+1$。

JEPA 的训练目标写作:

$$\mathcal{L}_{\mathrm{JEPA}} := \mathbb{E}_{p(x_1, x_2) q_\phi(z_1 | x_1)} \left[ D_{\mathrm{KL}}(q_\phi(z_2 | x_2) \| p_\theta(z_2 | z_1)) \right] \tag{1}$$

变量解释:
- $p(x_1, x_2)$: 观测的联合分布 (data distribution)
- $q_\phi(z_i | x_i)$: amortized variational posterior,即 encoder $E_\phi$,参数 $\phi$
- $p_\theta(z_2 | z_1)$: JEPA predictor,参数 $\theta$,从 $z_1$ 直接预测 $z_2$ 的分布
- $D_{\mathrm{KL}}(\cdot \| \cdot)$: KL divergence

实践中当 $q$ 和 $p$ 都是 fixed-variance Gaussian 时,这个 KL 退化成 MSE; 当是 categorical 时退化成 cross-entropy,这就对应了 I-JEPA ([Assran et al. CVPR 2023](https://arxiv.org/abs/2301.08243)) 和 DINO ([Caron et al. ICCV 2021](https://arxiv.org/abs/2104.14294)) 等具体方法。

### 2.2 变分分解

通过变分 ELBO 推导 (Appendix A.1),论文得到一个关键分解:

$$-\mathcal{L}_{\mathrm{JEPA}} = I(X_1; X_2) - \hat{I}(X_1; Z_1) - \hat{I}(X_2; Z_2) + \mathcal{R}_1 - \mathcal{G} \tag{2}$$

变量解释:
- $I(X_1; X_2)$: 两个 views 之间的真实互信息,与 $\phi, \theta$ 无关,是常量
- $\hat{I}(X_i; Z_i) := \mathbb{E}_{p(x_i) q_\phi(z_i | x_i)}[\log p(x_i | z_i) - \log p(x_i)]$: variational mutual information estimate,衡量 $Z_i$ 保留了 $X_i$ 多少信息 — 这正是 **compression term**
- $\mathcal{R}_1 := \mathbb{E}_{p(x_1)}[D_{\mathrm{KL}}(q_\phi(z_1 | x_1) \| p(z_1))]$: bottleneck regularizer,约束 encoder posterior 不要偏离 prior 太远 (这是防止 representation collapse 的隐式机制)
- $\mathcal{G} := \mathbb{E}_{p(x_1, x_2)}[D_{\mathrm{KL}}(q_\phi(z_1, z_2 | x_1, x_2) \| p_\theta(z_1, z_2 | x_1, x_2))]$: posterior approximation gap,衡量 amortized posterior 与真实 posterior 的距离

### 2.3 Information Bottleneck 结构

利用 PGM $X_1 - Z_1 - Z_2 - X_2$ 与 data processing inequality (DPI):
$$I(X_1; X_2) \leq I(Z_1; Z_2) \tag{24}$$

代入 (2) 得到上界:

$$-\mathcal{L}_{\mathrm{JEPA}} \leq I(Z_1; Z_2) - \hat{I}(X_1; Z_1) - \hat{I}(X_2; Z_2) + \mathcal{R}_1 - \mathcal{G} \tag{3}$$

这就是经典的 **Deep Variational Information Bottleneck ([Alemi et al. 2016](https://arxiv.org/abs/1612.00410))** 的形式:
- $I(Z_1; Z_2)$: 鼓励 representation 保留预测 target 的信息
- $-\hat{I}(X_i; Z_i)$: 惩罚 representation 保留太多 input 信息 (压缩)

JEPA 把 compression term 对称地施加于两个分支,这很自然 — JEPA 训练时把两支都用 encoder,而不是像传统 VIB 只有 input 一支被 bottleneck。这个对称结构强化了"两边都丢掉 nuisance,保留 predictive structure"的 inductive bias。

---

## 3. JEDI 的核心理论: Latent Conditional Diffusion 的 Bottleneck 分解

### 3.1 从 one-step predictor 到 reverse diffusion path

JEDI 把 JEPA 的 $p_\theta(z_2 | z_1)$ 这一步预测器,替换成一个 **conditional reverse diffusion trajectory**。Figure 3 中的 PGM:

$$z_1^0 \to z_2^T \to z_2^{T-1} \to \cdots \to z_2^1 \to z_2^0 \to x_2$$

变量解释:
- $z_1^0$: clean context latent (从 $x_1$ 编码,作为 conditioning)
- $z_2^0$: clean target latent (从 $x_2$ 编码,要预测的目标)
- $z_2^T$: 最大限度加噪的 latent (sampling 起点)
- $z_2^{t}$ 第 $t$ 个扩散时间步的 noisy latent
- $p_\psi(z_2^{t-1} | z_2^t, z_1^0)$: reverse conditional transition,参数 $\psi$,每步都 condition on clean context $z_1^0$

这个 reverse path 的 marginal 诱导了 stochastic JEPA predictor:
$$p_\psi(z_2^0 | z_1^0) = \int p(z_2^T) \prod_{t=1}^T p_\psi(z_2^{t-1} | z_2^t, z_1^0) \, dz_2^{1:T} \tag{27}$$

所以 conditional diffusion 在数学上就是一个 multi-step stochastic JEPA predictor。

### 3.2 Conditional denoising loss 与 bottleneck 分解

按标准 DDPM ([Ho et al. 2020](https://arxiv.org/abs/2006.11239)) 训练时,优化 conditional denoising objective:

$$\mathcal{L}_{\mathrm{CD}}^{\mathrm{den}} := \mathcal{L}^0 + \mathbb{E}_{p(x_1, x_2)} \sum_{t=2}^T \mathbb{E}_{q_\varphi(z_1^0|x_1) q_\varphi(z_2^0|x_2) q(z_2^t|z_2^0)} D_{\mathrm{KL}}(q(z_2^{t-1} | z_2^t, z_2^0) \| p_\psi(z_2^{t-1} | z_2^t, z_1^0)) \tag{4}$$

变量解释:
- $\mathcal{L}^0$: endpoint loss,即 $-\log p_\psi(z_2^0 | z_2^1, z_1^0)$ 的期望
- $q(z_2^t | z_2^0)$: forward diffusion kernel (固定的高斯扰动)
- $q(z_2^{t-1} | z_2^t, z_2^0)$: forward posterior (DDPM 推导中的解析式)
- 求和 $t=2$ 到 $T$: 跨扩散步的 KL 项

关键结果 (推导见 Appendix A.2):

$$-\mathcal{L}_{\mathrm{CDJ}}^{\mathrm{den}} \leq I(Z_1^0; Z_2^0) - \hat{I}(X_1; Z_1^0) - \hat{I}(X_2; Z_2^0) + \mathcal{R}_1 + \mathcal{R}_{2,T} - \mathcal{G} + \mathcal{C}_2 \tag{5}$$

新增项:
- $\mathcal{R}_{2,T} := \mathbb{E}_{p(x_2) q_\varphi(z_2^0|x_2)}[D_{\mathrm{KL}}(q(z_2^T | z_2^0) \| p(z_2^T))]$: terminal prior-matching,约束 forward 扩散的终端分布接近 $p(z_2^T)$
- $\mathcal{C}_2 := \mathbb{E}_{p(x_2)}[-H(q_\varphi(z_2^0 | x_2))]$: target encoder 分布的负熵。在 deterministic encoder 或 fixed-variance 加 stop-gradient 情况下,这是 $\psi$ 无关的常数

**Intuition**: 这个分解是说, conditional diffusion denoising 不是简单的生成 surrogate, 它本身就是一个 variational information bottleneck 目标的关键 piece。$I(Z_1^0; Z_2^0)$ 这一项就是 predictive term,鼓励 context latent 保留预测 target latent 的信息; compression terms 鼓励 encoder 丢掉无关信息。这就给"用 denoising loss end-to-end 训 encoder"提供了理论 grounding。

### 3.3 与 VICReg / LeJEPA 的关系

Bottleneck regularizer $\mathcal{R}_1$ 可分解为:
$$\mathcal{R}_1 = I_q(X_1; Z_1^0) + D_{\mathrm{KL}}(q_\varphi(z_1^0) \| p(z_1^0)) \tag{56}$$

第一项是 IB 部分, 第二项是 aggregate distribution-matching。当 $p(z_1^0) = \mathcal{N}(0, I)$ 时, 第二项就是 "embedding 分布要 centered、non-collapsed、isotropic" — 这与 VICReg ([Schwartz-Ziv et al. NeurIPS 2023](https://arxiv.org/abs/2306.12195)) 的 variance/covariance regularizer 以及 LeJEPA ([Balestriero & LeCun 2025](https://arxiv.org/abs/2511.08544)) 的 SIGReg 在精神上完全一致。所以 JEDI 论文把 $\mathcal{R}_1$ 解释为这些显式 collapse-prevention 方法的 variational counterpart。

⚠️ **重要 caveat** (Appendix A.2.1): JEDI 的实际实现并**没有显式优化** $\mathcal{R}_1$, 只优化了 $\mathcal{L}_{\mathrm{CDJ}}^{\mathrm{den}}$。所以严格地说 JEDI 是在优化 bottleneck 目标里的 prediction term, 而非完整的 variational bottleneck。论文坦率承认这一点, 把 $\mathcal{R}_1$ 的显式优化留作 future extension。

---

## 4. JEDI Architecture 详解

### 4.1 World Model 三大组件

$$\text{World Model: } \begin{cases}
\text{Encoder:} & z_t^0 = \mathbf{E}_\phi(x_t) \\
\text{Latent diffusion dynamics:} & \hat{z}_{t+1}^0 \sim \mathbf{S}(\mathbf{D}_\theta(\hat{z}_{t+1}^\tau, Z_t^\tau)) \\
\text{Reward and termination:} & (\hat{r}_t, \hat{d}_t) = \mathbf{R}_\psi(z_t^0)
\end{cases} \tag{6}$$

变量解释:
- $t$: environment time step
- $x_t \in [0,1]^{64 \times 64 \times 3}$: RGB observation,大小 12288
- $z_t^0 \in [-3, 3]^{16 \times 8 \times 8}$: clean latent,大小 1024 — **比 pixel input 小 12 倍**
- $\tau$: diffusion time step,从 log-normal 分布 $\mathrm{LN}$ 采样 (这是 EDM ([Karras et al. 2022](https://arxiv.org/abs/2206.00364)) 的设计)
- $Z_t^\tau := (c_{\mathrm{noise}}^\tau, \hat{z}_{t-3:t}^0, a_{t-3:t})$: conditioning tuple,包含:
  - $c_{\mathrm{noise}}^\tau$: $\tau$ 的固定 transformation,作为 diffusion-time embedding
  - $\hat{z}_{t-3:t}^0$: 过去 4 步的 latent states (frame stacking)
  - $a_{t-3:t}$: 过去 4 步的 actions
- $\mathbf{S}$: ODE solver (实验中用 Euler,3 steps)
- $\mathbf{D}_\theta$: 神经网络 denoiser

### 4.2 EDM preconditioning

JEDI 沿用 EDM 的 preconditioned denoiser:

$$\mathbf{D}_\theta(z_{t+1}^\tau, Z_t^\tau) = c_{\mathrm{skip}}^\tau z_{t+1}^\tau + c_{\mathrm{out}}^\tau \mathbf{F}_\theta(c_{\mathrm{in}}^\tau z_{t+1}^\tau, Z_t^\tau) \tag{7}$$

变量解释:
- $c_{\mathrm{skip}}^\tau$: skip connection 系数,控制多少 noisy input 直接传到 output
- $c_{\mathrm{out}}^\tau$: 输出尺度系数
- $c_{\mathrm{in}}^\tau$: 输入尺度系数
- $\mathbf{F}_\theta$: 真正的神经网络 (U-Net 单层无 downsampling,160 channels)

这些 preconditioning 系数的设计目标: 让 $\mathbf{F}_\theta$ 的输入输出在所有 noise level 上都有 unit variance,从而神经网络不用学习 scale,只学方向。这是 EDM 的核心 trick,极大提高了 diffusion model 在跨 noise level 训练时的稳定性。

### 4.3 Joint embedding diffusion loss

JEDI 的核心训练信号:

$$\mathbb{E}_{z_{1:T} \sim q, x_{1:T} \sim p, \tau \sim \mathrm{LN}} \left[ \left\| \sum_{t=1}^T \mathbf{F}_\theta(c_{\mathrm{in}}^\tau \mathrm{sg}(z_{t+1}^\tau), Z_t^\tau) - \frac{1}{c_{\mathrm{out}}^\tau}(\mathrm{sg}(z_{t+1}^0) - c_{\mathrm{skip}}^\tau \mathrm{sg}(z_{t+1}^\tau)) \right\|^2 \right] \tag{8}$$

变量解释:
- $q(z_t)$: observation 分布经 encoder 的 deterministic pushforward (encoder 是 deterministic)
- $\mathrm{sg}(\cdot)$: stop-gradient 算子,切断目标 latent 对 encoder 的反向梯度
- $\mathrm{LN}$: $\tau$ 的 log-normal 采样分布
- 公式内部其实是 EDM 的标准 MSE 形式: $\mathbf{F}_\theta$ 预测 normalized denoising direction,目标也是 normalized clean direction

⚠️ 注意 stop-gradient 的位置: 它施加在**目标** (target latent $z_{t+1}^0$ 和 noisy latent $z_{t+1}^\tau$) 上,而不施加在 condition $Z_t^\tau$ 中的 $\hat{z}_{t-3:t}^0$ 上。这意味着: encoder 通过 condition 路径接收来自 denoising loss 的梯度,但目标 latent 不会通过梯度把自己往 trivial distribution 推 (avoiding representation collapse)。

### 4.4 Reward/termination head

$$\mathbb{E}_{z_{1:T}^0 \sim q, (x_{1:T}, r_{1:T}, d_{1:T}) \sim p} \sum_{t=1}^T \mathbf{CE}(\mathbf{R}_\psi(z_t^0), (r_t, d_t)) \tag{9}$$

这里 $\mathbf{R}_\psi$ 是 reward/termination 网络,$\mathbf{CE}$ 是 cross-entropy。Reward 用 sign-binarized $\{-1, 0, 1\}$ 分类化处理 (DreamerV3 的做法); Craftium 的连续 reward 用 symlog trick。

**关键**: 两个 loss (denoising 和 reward/termination) 的梯度都直接流到 encoder $E_\phi$ — 这就是 "end-to-end" 的含义。

---

## 5. Practical Design Choices (三个 stabilization tricks)

论文明确指出,单纯 end-to-end 训 latent diffusion 会不稳定,需要三个关键设计:

### 5.1 Stop-gradient + asymmetric learning rate

对 future latent target 施加 $\mathrm{sg}$,并把 encoder 的 learning rate 设为 denoiser LR 的 0.3 倍 (Table 3: `Learning rate scale factor for E_φ = 0.3`)。

这个 trick 来自 JEPA / TDMPC2 ([Hansen et al. 2023](https://arxiv.org/abs/2310.16828)) 的实践经验: 如果 encoder 通过 target 路径快速 self-reinforce,很容易 collapse 到 trivial constant。Asymmetric LR 让 encoder 学得慢一些,denoiser 学得快一些,denoiser 先适配当前 encoder,然后 encoder 慢慢被 denoising signal 推动。

### 5.2 Latent clamping

$$C(z) = \tanh(z/s) \cdot s, \quad s = 3$$

施加到 encoder 输出和 denoiser 输出。这把 latent 严格限制在 $[-3, 3]$ 范围内,同时 $\tanh$ 是可微的。

为什么需要? 在 pixel space diffusion 里,DIAMOND 把 pixels 天然 clamp 到 $[-1, 1]$ 并离散化成 256 bins。但在 latent space 没有 "natural range",latent 数值范围可能在训练中 drift 到 $[-100, 100]$,这会让 reverse diffusion 的 ODE solver 数值不稳定。$C(z)$ 给 latent 一个"软的" upper bound,既有可微性又有数值稳定。HI 也用了类似 tanh clamping。

### 5.3 Random switching between denoiser output and encoder output

每个 trajectory batch,以均匀概率随机决定下一步的 condition latent 是:
- $\mathbf{D}_\theta(\hat{z}_{t+1}^\tau, Z_t^\tau)$ (denoiser 输出,从 imagination 来的 latent),或者
- $\mathbf{E}_\phi(x_{t+1})$ (encoder 输出,从真实 observation 来的 latent)

Intuition: 这个 trick 平衡两个目标。如果永远用 encoder output 作为 next-step condition, denoiser 学不到 multi-step rollout 的累积误差如何处理; 如果永远用 denoiser output, encoder 缺乏直接 supervision,容易 drift。Random switching 同时提供了:
- Near-horizon consistency (encoder → encoder 路径)
- Direct encoder supervision (random step 切回 encoder 时,encoder 隔 1 step 直接被 denoising loss 惩罚)

Algorithm 1 中的 $rs \sim \mathrm{Uniform}\{\mathrm{True}, \mathrm{False}\}$ 就是这个随机切换。

### 5.4 Policy: REINFORCE

JEDI 用 REINFORCE ([Williams 1992](https://link.springer.com/article/10.1007/BF00992696)) 而不是 backprop-through-diffusion。Intuition 是: diffusion 多步 denoising 的反向传播既昂贵又不稳定,REINFORCE 让 actor 完全从 rollout 出来的 trajectory 学,把 policy learning 和 world-model learning 解耦,保持 lightweight。

---

## 6. 实验: Atari100k 主结果

### 6.1 Benchmark 设置

- 26 个 Atari games,5 seeds per game
- 100k environment interactions total budget
- Frame skip 4, RGB 64×64
- 每 run 用 2 个 batch size (32, 64) 中较好的结果 (类似 HI 的 hyperparameter 报告方式)
- 每 run 约 38 小时 A100,13.5M parameters

### 6.2 Aggregate 指标

**主要 baselines**:
- **DIAMOND** (canonical pixel diffusion baseline,13.5M params,98 A100-hours)
- **HI** (latent diffusion with separately trained latents,97M params,27 A100-hours)
- DreamerV3, STORM, TWM, IRIS, TWISTER 作为额外对比

| Method | Mean HNS ↑ | IQM ↑ | Optimality Gap ↓ | #SOTA ↑ | #Superhuman ↑ | Har-HNS ↑ |
|---|---|---|---|---|---|---|
| DIAMOND | 1.621 | 0.609/0.618 | 0.480/0.488 | 6 | 12 | 0.319 |
| **JEDI** | 1.450 | **0.688** | **0.460** | 7 | 11 | **0.377** |

**关键观察**:
- IQM 和 Optimality Gap 上 JEDI 都达到了 SOTA
- Mean HNS 上 JEDI 比 DIAMOND 稍低 (1.450 vs 1.621)
- Har-HNS (论文新提出的指标,公式见下) JEDI 显著优于 DIAMOND (0.377 vs 0.319)

### 6.3 Har-HNS: 一个新的 aggregate metric

论文批评 arithmetic mean HNS 偏向 already-high-value 的 games,会掩盖低 HNS 任务的相对提升。他们借鉴 F1 score 的调和平均思想:

$$\text{Har-HNS} = \left(\frac{1}{N} \sum_{i=1}^N \frac{1}{\text{HNS}_i + 0.1}\right)^{-1} \tag{10}$$

变量解释:
- $N$: 游戏数量 (这里是 26)
- $\text{HNS}_i$: 游戏 $i$ 上的 human-normalized score
- $+0.1$: 数值稳定 offset,避免 HNS 为 0 或负时除零

Intuition: Mean HNS 问"所有任务总共拿多少分",Har-HNS 问"还很难的任务上做到多好"。调和平均对最低值高度敏感 — 如果一个任务 HNS=0.01, 即使其他 25 个都是 1.5, 调和平均也会被严重拉低。所以 Har-HNS 更像 "worst-case-aware" 的指标。

HNS 的定义:
$$\text{HNS} = \frac{S_{\text{agent}} - S_{\text{random}}}{S_{\text{human}} - S_{\text{random}}}$$

### 6.4 Stochastic Atari 实验

论文在 3 个 Atari 游戏上加了 random frame skip (2 到 6 之间),制造 aleatoric uncertainty。结果 (Figure 7): JEDI 在 stochastic 设置下显著超过 DreamerV3。这是 diffusion 的一大优势 — multimodal、stochastic target 用 iterative denoising 自然 express,而 DreamerV3 的 Gaussian MLP head 表达能力受限。

---

## 7. Craftium 实验

Craftium ([Malagón et al. 2024](https://arxiv.org/abs/2407.03969)) 是 3D Minecraft-like embodied 环境,4 个 tasks × 3 seeds。JEDI 在 SmallRoom-v0, Speleo, BridgeBuilder, CollectAll 上整体优于 HI (Figure 5)。这证明了 JEDI 的方法不仅限于 2D Atari,能迁移到 3D 视觉复杂的环境。

---

## 8. 效率分析

| Method | A100 (hrs) | V100 (hrs) | #Params |
|---|---|---|---|
| **JEDI** | 38 | - | 13.5M |
| HI | 27 | - | 97M |
| DIAMOND | 98 | - | 13.5M |
| DreamerV3 | - | 12 | 18M |
| STORM | 7 | 9.3 | 18.8M |
| TWM | 10 | - | 21.6M |
| IRIS | 168 | - | 30M |

相对 DIAMOND,JEDI:
- **VRAM 减少 43%** (从 64×64×3 到 16×8×8,12× 体积缩减)
- **World-model sampling 快 3×以上**
- **Training 快 2.5×以上** (38 hrs vs 98 hrs)
- **Parameter count 相同** (13.5M)

与 HI 比, JEDI 训练稍慢 (38 vs 27 hrs),但 HI 用 97M params 且依赖 pretrained perceptual model,而 JEDI 用 13.5M params 且完全 end-to-end。所以 JEDI 在"end-to-end 训练"这条轴上是更干净的设计 — 没有外部 frozen encoder 依赖。

---

## 9. Performance Profile 分析

### 9.1 关键观察: JEDI 和 DIAMOND 的相对强弱分布

JEDI 并非"在所有任务上都比 DIAMOND 强一点" — 它们是**完全不同的 performance profile**。

Figure 11 + 12 的核心发现:
- **JEDI 的 top-quantile relative gains** 集中在 DIAMOND HNS 很低的游戏上 (mean HNS ≈ 0.06)
- **DIAMOND 的 top-quantile relative gains** 集中在 HNS 较高的游戏上 (mean HNS ≈ 0.81)

按游戏属性分析 (Figure 12):
- **JEDI 优势集中在大 action space 游戏** (尤其是 18 个 action 的游戏)
- **JEDI 在 shooter-style 游戏上优势明显** (BankHeist, DemonAttack, Hero, BattleZone, ChopperCommand, Alien 等)

JEDI 的 6 个 top-quantile 游戏中,5 个是 max action space (18 个 action),且大部分是 shooter。

### 9.2 为什么? — 一个 plausible 解释

DIAMOND 把 raw image (64×64×3) 直接喂给 actor 和 critic;JEDI 把 12× 小的 learned latent (16×8×8) 喂给 actor 和 critic。

Hypothesis: **Effective interaction space between inputs and action/value prediction 随 input 维度近似 exponential 增长**。大 action space + 复杂视觉 → policy/value 网络的"搜索空间"非常大。DIAMOND 的 actor-critic 需要同时做 representation learning 和 policy/value learning,在固定 compute budget 下很难两边都做好。JEDI 把 representation 学习"外包"给 world model (通过 JEPA loss 学到的 latent 已经是 predictive abstraction), actor-critic 只需做 policy/value learning,在 input 维度大幅减小的情况下更容易学。

⚠️ 论文明确说这是"基于经验模式的解释,不是形式化证明"。但这个 observation 与 LeCun 一直强调的"JEPA-style representation 应该让 downstream task 更容易学"完全一致 ([LeCun 2022](https://openreview.net/pdf?id=BZ5a1r-kVsf))。

### 9.3 Qualitative 分析 (Figure 13)

在 BankHeist, DemonAttack, Hero 三个任务上,JEDI 学到了明显不同的 policy:
- BankHeist: JEDI 反复重进 map 收集 easy bank spawns, 避免自毁 FIRE 动作
- Hero: JEDI 更可靠地摧毁障碍物
- DemonAttack: JEDI 主动追踪并消灭敌人, 而非缩在角落乱射

这些 trajectory 都展示了 JEDI 的 policy 在 large-action、复杂动态下更"主动"地控制环境,这与上面的假设一致。

---

## 10. Ablation Studies

### 10.1 Latent Learning ablation (Figure 9)

五个变体在 5 个 Atari 任务上比较:
1. **Full JEDI** — 最好
2. **MSE Loss** (把 diffusion loss 换成 direct next-state MSE) — 比 full 差
3. **- Diff Grad** (去掉 diffusion loss 对 encoder 的梯度) — 比 MSE Loss 差
4. **+ Decoder Grad** (加 decoder-based reconstruction supervision) — 又差一些
5. **AutoEncoder** (用 separately trained VAE latent) — 最差

**Intuition**: 这组 ablation 直接验证了论文的核心 claim — **latent space 必须从 predictive world-model objective 端到端学出来, 而不是从 reconstruction 学出来**。Diffusion 比 MSE 更好的原因可能是 diffusion 对 stochastic/multimodal target 的表达能力强,且 iterative denoising 是一种更鲁棒的 predictive signal。

### 10.2 Design Choice Ablation (Figure 10)

四个变体:
1. **Default JEDI** — 最好
2. **EMA target encoder** (用 EMA 而不是 stop-gradient 防止 collapse) — 稍差
3. **Deterministic switching** (固定切换 schedule 而不是 random) — 稍差
4. **No random switching** — 更差
5. **No switching + no clamping** — 最差

⚠️ 注意: HNS 在所有这些 ablation 中保持较高, 表明 stabilization tricks 是 "helpful but not critical"。**end-to-end predictive objective 才是 main ingredient**。

---

## 11. Algorithm 概览 (Algorithm 1)

JEDI 的训练循环 (论文 Algorithm 1) 可以总结为:

```
每个 epoch:
  1. collect_experience(steps_collect): 用当前 policy 在 env 里收集 transitions, 用 JEDI encoder 推 latent 再采 action
  2. for steps_diffusion_model:
       update_latent_diffusion_model():
         - 采样序列 (x_{t-3:t+1}, a_{t-3:t})
         - 编码到 z^0_{t-3:t+1} 并 clamp
         - 对 z^0_{t+1} 加噪到 z^τ_{t+1}
         - 计算 EDM-style MSE loss
         - random switch: 50% 概率把 denoiser 输出缓存作为下一步 condition
  3. for steps_reward_end_model:
       update_reward_end_model():
         - 编码到 latents
         - LSTM 处理序列, 预测 r, d
         - cross-entropy loss
  4. for steps_actor_critic:
       update_actor_critic():
         - 用 world model imagine H 步 trajectory (用 reverse diffusion 出 next latent)
         - 计算 V 和 policy gradient (REINFORCE)
         - 更新 π_ω 和 V_ω
```

关键超参 (Table 2, 3):
- Latent shape: $[16, 8, 8]$,范围 $[-3, 3]$
- Conditioning length $L = 4$
- Imagination horizon $H = 15$
- $\gamma = 0.985$, $\lambda = 0.95$ (TD($\lambda$)), $\eta = 0.001$ (entropy)
- Diffusion sampling: Euler, 3 steps (s_churn=1 仅 stochastic 实验用)
- Network architecture:
  - Latent Diffusion Dynamics Model: 1 layer, 160 channels, 无 downsampling
  - Encoder: 4 个 stage (channels [32, 32, 32, 16]), downsampling pattern [1, 1, 1, 0]
  - Reward/Termination Model: 4 个 stage (channels [2, 2, 2, 2])

---

## 12. 局限性与未解决问题

论文坦率承认:

1. **Benchmark 范围有限**: 只在 Atari100k (5 seeds) 和 4 个 Craftium tasks (3 seeds) 上测过,DMControl, Procgen, 高分辨率 domain 未测。
2. **不是最快的 latent 方法**: HI 27 hours vs JEDI 38 hours,但 HI 用 97M params 和 pretrained perceptual model,所以这个比较不完全公平。
3. **理论是 motivation 不是保证**: eqs. (1)-(5) 给出 representation-learning properties,不直接保证 optimization 收敛到这些 properties。
4. **依赖多个 design choices**: tanh clamping, random switching 都去掉会变差。但 DIAMOND 和 HI 也用 clamping,所以这是 latent diffusion 通病。
5. **对 batch size 敏感**: 每个 task 报告 batch 32 或 64 中较好的,这也是 HI 的做法。

---

## 13. 与相关工作谱系的连接

### 13.1 World Model / MBRL

- **Dyna ([Sutton 1991](https://dl.acm.org/doi/10.1145/122344.122377))** 和 **PlaNet ([Hafner et al. ICML 2019](https://arxiv.org/abs/1811.04551))**: 奠基性的 predictive world model
- **Dreamer / DreamerV2 / DreamerV3** ([arXiv 1912.01603](https://arxiv.org/abs/1912.01603), [arXiv 2010.02193](https://arxiv.org/abs/2010.02193), [arXiv 2301.04104](https://arxiv.org/abs/2301.04104)): actor-critic 在 imagined trajectory 上学
- **IRIS, TWM, STORM, TWISTER**: transformer-based world model,扩展 Atari100k design space
- **DIAMOND**: pixel-space diffusion, JEDI 的直接 baseline
- **HI**: latent diffusion but separately trained latents

### 13.2 JEPA / Predictive Representations

- **Schmidhuber 1993**: 早期 "predictable classification" 思想 ([Neural Computation 1993](https://direct.mit.edu/neco/article-abstract/5/4/625/5615))
- **BYOL ([Grill et al. NeurIPS 2020](https://arxiv.org/abs/2006.07733))**: 不用 negative sample 的 self-supervised learning,JESA 的精神祖先之一
- **DINO ([Caron et al. ICCV 2021](https://arxiv.org/abs/2104.14294))** 和 **I-JEPA ([Assran et al. CVPR 2023](https://arxiv.org/abs/2301.08243))**: modern JEPA-style
- **V-JEPA 2 ([Assran et al. 2025](https://arxiv.org/abs/2506.09985))**: video-level JEPA
- **TD-MPC / TD-MPC2** ([Hansen et al. 2022](https://arxiv.org/abs/2203.04955), [2023](https://arxiv.org/abs/2310.16828)): 连续控制中的 end-to-end predictive latent
- **LeWorldModel ([Maes et al. 2026](https://arxiv.org/abs/2603.19312))**: 稳定的 end-to-end JEPA world model
- **LeJEPA ([Balestriero & LeCun 2025](https://arxiv.org/abs/2511.08544))**: provable collapse-free JEPA

### 13.3 Diffusion 在 RL/Planning/Control

- **Diffusion Policy ([Chi et al. IJRR 2025](https://arxiv.org/abs/2303.04137))**: 视觉运动 policy learning
- **Diffuser ([Janner et al. 2022](https://arxiv.org/abs/2205.09991))**: planning as diffusion
- **Diffusion-RL offline** ([Wang et al. 2022](https://arxiv.org/abs/2208.06193), [Ding et al. 2024](https://arxiv.org/abs/2402.03570))
- **Diffusion guidance as policy improvement** ([Frans et al. 2025](https://arxiv.org/abs/2505.23458))

### 13.4 Diffusion 作为 representation learner

- **Diffusion hyperfeatures** ([Luo et al. 2023](https://arxiv.org/abs/2307.14081))
- **Diffusion time-steps for unsupervised learning** ([Yue et al. 2024](https://arxiv.org/abs/2401.11430))
- **Diffusion model as representation learner** ([Yang & Wang ICCV 2023](https://arxiv.org/abs/2303.01848))
- **DiffEnc** ([Nielsen et al. 2023](https://arxiv.org/abs/2310.19789)): variational diffusion with learned encoder
- **LSGM ([Vahdat et al. NeurIPS 2021](https://arxiv.org/abs/2011.12856))**: 早期的 end-to-end latent generative model, 但用 reconstruction 训 latent

JEDI 与 LSGM 的核心区别: LSGM 还是 reconstruction-based latent learning, JEDI 用 JEPA-style predictive loss 完全替代 reconstruction。

### 13.5 Latent Diffusion Model (一般)

- **Stable Diffusion / LDM ([Rombach et al. CVPR 2022](https://arxiv.org/abs/2112.10752))**: separately trained VAE + diffusion
- **SD3 / Rectified Flow Transformers ([Esser et al. ICML 2024](https://arxiv.org/abs/2403.03206))**
- **SVD ([Blattmann et al. 2023](https://arxiv.org/abs/2311.15127))**: latent video diffusion
- **Sora review ([Liu et al. 2024](https://arxiv.org/abs/2402.17177))**: large vision model
- **Genie ([Bruce et al. 2024](https://arxiv.org/abs/2402.15391))** and **Genie 2** ([DeepMind 2024](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)): 生成式 interactive environment

---

## 14. 我的 Intuition 与批判性思考

让我从你的视角 (Karpathy 视角) 整理一下直觉与 critique:

### 14.1 真正的关键 idea (一句话)

> **JEDI 用 conditional diffusion denoising loss 作为一个 JEPA predictor 的训练目标, 通过这个目标 end-to-end 训练 encoder, 完全跳过 reconstruction / pretrained model。**

这一句话浓缩了论文。其他都是 stabilization tricks 和理论 grounding。

### 14.2 为什么 end-to-end 重要?

如果 latent 是 separately trained (像 HI),latent 学的是 "如何重构 observation" — 但重构信号对 RL 控制未必有用,甚至有害 (比如重构背景噪声浪费 capacity)。End-to-end 通过 world-model objective 学的 latent 学的是 "什么信息对预测未来、reward、control 有用" — 这才是 RL 想要的 representation。这个直觉与 TD-MPC2 的成功一致。

### 14.3 为什么 diffusion 比 MSE 强?

Ablation 表明 diffusion 比 MSE next-state prediction 更好。可能原因:
1. **Multimodal target**: Atari 即使在 frame-skip 4 下也是 deterministic, 但更深层的状态空间可能有多个合理 next state (尤其是敌人 AI、子弹位置等)。Diffusion 自然 express multimodal。
2. **Iterative refinement 作为 regularization**: diffusion 强迫模型在 noise level 之间共享信息,这可能 force encoder 学出更 smooth 的 representation。
3. **EDM preconditioning 的福利**: diffusion 训练在 noise level 上的归一化 trick,可能是 latent stability 的隐式功臣。

### 14.4 Stop-gradient 的微妙之处

JEDI 把 stop-gradient 放在 target 上,这和 BYOL / I-JEPA 的做法完全一致。Intuition: 如果让梯度从 target 反向流到 encoder, encoder 会学到 "把所有 input 映射到 trivial constant" 来最小化 KL — 因为 trivial constant 让 $q_\phi(z_2 | x_2)$ 和 $p_\theta(z_2 | z_1)$ 都变成 degenerate, KL=0。Stop-gradient 切断这条路径, encoder 只能通过 condition 路径接收 "怎么让 denoising 容易" 的信号, 这才是 representation learning signal。

### 14.5 Performance profile 改变的意义

JEDI vs DIAMOND 的 profile 完全不同这件事,在 my view 是 paper 最有意思的发现之一。这暗示: **representation 的形式 (pixel vs latent) 改变了 RL agent 的 "游戏"本身**, 而不仅仅是"效率与精度的 trade-off"。这与 LLM 中的 "tokenization 改变 inductive bias" 类似 — input space 决定 policy/value 学习的 effective horizon。

更具体: 把 actor-critic input 从 64×64×3 (12288) 换到 16×8×8 (1024) 不只是 12× speedup, 还有 combinatorial 结构的改变。RL 中 effective difficulty ∝ input_dim × action_dim × horizon, 减小 input_dim 让大 action space 游戏 (18 actions) 在固定 compute 下变得更 tractable。这就解释了为什么 JEDI 在 large-action shooter 游戏上突出。

### 14.6 未解决的问题

1. **$\mathcal{R}_1$ 没有显式优化**: 理论给出了 bottleneck 视角,但实际只优化 prediction term。VICReg / LeJEPA 风格的显式 collapse-prevention 加上后能否进一步提升?这是 obvious next step。
2. **DMControl, Procgen 缺失**: 3D Craftium 是 supporting evidence,但 2D Atari 的动作空间与视觉结构特殊。DMControl 的连续控制、Procgen 的 procedural generalization 都没测。
3. **Diffusion sampling cost 仍然是 bottleneck**: 3 steps Euler 比 1 step MSE 慢,虽然 latent 上做但仍比 DreamerV3 的 forward pass 慢。能否用 consistency model / flow matching 把它压成 1 step?
4. **Random switching 的 ablation 不彻底**: 没测 switching 概率 = 0.7 vs 0.5 vs 0.3 的差异,这个超参可能很重要。
5. **Policy learning 与 world model 的 decoupling**: REINFORCE 完全切断 diffusion 的反向梯度。是否能用 score-based policy gradient 利用 world model 的 score function 做更好的 policy optimization? (类似 [Diffusion guidance as policy improvement](https://arxiv.org/abs/2505.23458))

### 14.7 联想到的更广泛工作

- **World Models ([Ha & Schmidhuber 2018](https://arxiv.org/abs/1803.10122))** 的原始愿景: "policy 在 latent 'dream' 中学习"。JEDI 把这个 dream 从 VAE-based 换成 diffusion-based,且 latent 不再 require reconstruction,这更接近 Schmidhuber 早期 "compress to predict" 的本意 ([Schmidhuber 1993](https://direct.mit.edu/neco/article-abstract/5/4/625/5615))。
- **LeCun 的 JEPA 路线图** ([LeCun 2022 position paper](https://openreview.net/pdf?id=BZ5a1r-kVsf)): JEDI 是这条路线在 RL 中的具体实例化之一,与 V-JEPA 2 ([arxiv 2506.09985](https://arxiv.org/abs/2506.09985))、LeWorldModel ([arxiv 2603.19312](https://arxiv.org/abs/2603.19312))、Temporal Straightening for Latent Planning ([arxiv 2603.12231](https://arxiv.org/abs/2603.12231)) 形成了一批 2026 年的 JEPA-for-control 工作。
- **Generative interactive environment**: Genie ([arxiv 2402.15391](https://arxiv.org/abs/2402.15391)) 和 Genie 2 ([DeepMind 2024](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)) 把 world model scaling 到 large foundation model;JEDI 的 end-to-end latent diffusion 视角与这条线连接,可能给 foundation world model 提供更高效的训练 paradigm。
- **Diffusion for sequence modeling**: Diffusion-LM ([Li et al. NeurIPS 2022](https://arxiv.org/abs/2205.14217))、Discrete Diffusion Ratios ([Lou et al. 2023](https://arxiv.org/abs/2310.16834))、Diffusion Beats Autoregressive in Data-Constrained ([Prabhudesai et al. 2025](https://arxiv.org/abs/2507.15857)) — 这些工作表明 diffusion 在 sequential prediction 上有一些 autoregressive 没有的优势 (比如 parallel decoding、对 stochastic target 的处理)。JEDI 是这个趋势在 RL world model 中的对应。
- **Classification as diffusion**: Discrete Diffusion Classification Modeling ([Belhasin et al. 2025](https://arxiv.org/abs/2511.20263)) — 这显示了 diffusion 不仅仅是 generative 工具,也可以做判别任务。JEDI 把 diffusion 当成 representation learner, 与这个 trend 一致。

---

## 15. Summary

| 维度 | JEDI |
|---|---|
| 核心贡献 | 第一个 end-to-end latent diffusion world model |
| 关键 idea | 用 conditional diffusion denoising loss 作为 JEPA predictor, end-to-end 训 encoder |
| 理论 grounding | latent diffusion loss 等价于 variational info bottleneck 的 prediction term (up to constants) |
| 效率提升 | 比 DIAMOND: 43% VRAM 减少, 3× sample 加速, 2.5× train 加速 |
| 性能 | Atari100k IQM/Optimality Gap SOTA, Har-HNS SOTA, Mean HNS 略低于 DIAMOND |
| Performance profile | 与 DIAMOND 完全不同: 集中在 large-action、shooter、low-HNS 任务上 |
| 关键 ablation | latent 必须从 predictive objective 端到端学出来, 不能从 reconstruction 学 |
| 关键 stabilization tricks | stop-gradient + 0.3× LR + tanh clamp + random switching |
| 关键 limitations | benchmark 范围有限, $\mathcal{R}_1$ 没显式优化, batch size 敏感 |

---

## References (链接)

- [JEDI Paper (本次讨论)](https://arxiv.org/abs/2410.21050) (推测 arXiv ID)
- [DIAMOND (Alonso et al. NeurIPS 2024)](https://arxiv.org/abs/2403.04134)
- [Horizon Imagination (HI)](https://arxiv.org/abs/2602.08032)
- [DreamerV3 (Hafner et al. 2023)](https://arxiv.org/abs/2301.04104)
- [TD-MPC2 (Hansen et al. 2023)](https://arxiv.org/abs/2310.16828)
- [DDPM (Ho et al. 2020)](https://arxiv.org/abs/2006.11239)
- [EDM (Karras et al. 2022)](https://arxiv.org/abs/2206.00364)
- [I-JEPA (Assran et al. CVPR 2023)](https://arxiv.org/abs/2301.08243)
- [V-JEPA 2 (Assran et al. 2025)](https://arxiv.org/abs/2506.09985)
- [LeJEPA (Balestriero & LeCun 2025)](https://arxiv.org/abs/2511.08544)
- [LeCun JEPA Position Paper 2022](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- [Deep Variational Information Bottleneck (Alemi et al. 2016)](https://arxiv.org/abs/1612.00410)
- [Information Bottleneck (Tishby et al. 2000)](https://arxiv.org/abs/physics/0004057)
- [Schwartz-Ziv & Tishby 2017](https://arxiv.org/abs/1703.00810)
- [VICReg info-theory view (Schwartz-Ziv et al. NeurIPS 2023)](https://arxiv.org/abs/2306.12195)
- [DINO (Caron et al. ICCV 2021)](https://arxiv.org/abs/2104.14294)
- [BYOL (Grill et al. NeurIPS 2020)](https://arxiv.org/abs/2006.07733)
- [Diffusion Policy (Chi et al. IJRR 2025)](https://arxiv.org/abs/2303.04137)
- [Diffuser (Janner et al. 2022)](https://arxiv.org/abs/2205.09991)
- [Diffusion guidance as policy improvement (Frans et al. 2025)](https://arxiv.org/abs/2505.23458)
- [LDM / Stable Diffusion (Rombach et al. CVPR 2022)](https://arxiv.org/abs/2112.10752)
- [SD3 (Esser et al. ICML 2024)](https://arxiv.org/abs/2403.03206)
- [LSGM (Vahdat et al. NeurIPS 2021)](https://arxiv.org/abs/2011.12856)
- [Genie (Bruce et al. 2024)](https://arxiv.org/abs/2402.15391)
- [Genie 2 (DeepMind 2024)](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)
- [Craftium (Malagón et al. 2024)](https://arxiv.org/abs/2407.03969)
- [World Models (Ha & Schmidhuber 2018)](https://arxiv.org/abs/1803.10122)
- [PlaNet (Hafner et al. ICML 2019)](https://arxiv.org/abs/1811.04551)
- [IRIS (Micheli et al. 2022)](https://arxiv.org/abs/2209.00588)
- [Atari100k (Kaiser et al. 2019)](https://arxiv.org/abs/1903.00374)
- [Agarwal et al. NeurIPS 2021 (RL evaluation)](https://arxiv.org/abs/2108.13264)
- [REINFORCE (Williams 1992)](https://link.springer.com/article/10.1007/BF00992696)
- [Schmidhuber 1993 — Discovering Predictable Classifications](https://direct.mit.edu/neco/article-abstract/5/4/625/5615)
- [LeWorldModel (Maes et al. 2026)](https://arxiv.org/abs/2603.19312)
- [Temporal Straightening for Latent Planning (Wang et al. 2026)](https://arxiv.org/abs/2603.12231)
- [Diffusion-LM (Li et al. NeurIPS 2022)](https://arxiv.org/abs/2205.14217)
- [Diffusion Beats AR in Data-Constrained (Prabhudesai et al. 2025)](https://arxiv.org/abs/2507.15857)
- [Discrete Diffusion Classification (Belhasin et al. 2025)](https://arxiv.org/abs/2511.20263)
- [Sora review (Liu et al. 2024)](https://arxiv.org/abs/2402.17177)
- [Stable Video Diffusion (Blattmann et al. 2023)](https://arxiv.org/abs/2311.15127)
- [DiffEnc (Nielsen et al. 2023)](https://arxiv.org/abs/2310.19789)
- [Diffusion Hyperfeatures (Luo et al. 2023)](https://arxiv.org/abs/2307.14081)
- [Diffusion Model as Representation Learner (Yang & Wang ICCV 2023)](https://arxiv.org/abs/2303.01848)
- [Offline Diffusion RL (Ding et al. 2024)](https://arxiv.org/abs/2402.03570)
- [Offline RL Diffusion (Wang et al. 2022)](https://arxiv.org/abs/2208.06193)

---

如果你想深入某一块,我建议下一步聚焦在: (1) $\mathcal{R}_1$ 的显式 VICReg-style 优化能不能进一步提升 JEDI; (2) 用 rectified flow 或 consistency model 把 latent diffusion 压成 1 step sampling,理论上能否保留 JEPA signal; (3) performance profile 在 DMC Control suite 上是否反转 (因为 DMC 是连续 action space + low-dim observation, JEDI 的"input 12× 减小"优势可能消失)。这些都是从 JEDI 直接延伸、且没被论文回答的开放问题。
