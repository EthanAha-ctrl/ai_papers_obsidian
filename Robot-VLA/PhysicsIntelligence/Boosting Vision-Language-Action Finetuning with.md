---
source_pdf: Boosting Vision-Language-Action Finetuning with.pdf
paper_sha256: 55fbac15abbf286d2253a1eab85518070251b5677494b882457e754be78278c8
processed_at: '2026-07-20T18:41:51-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# FAN: Feasible Action Neighborhood — VLA Finetuning 的新范式

这篇 paper 来自 Shanghai Jiao Tong University 和 Huawei，核心 idea 非常 clean：**物理 action 具有 intrinsic tolerance，而 language-style training objective 忽略了这一点**。作者用 Gaussian prior 显式 shape policy distribution 的几何结构，得到一个 simple 但 effective 的 regularizer。我从 motivation、math、experiment 三个层次详细 build intuition。

---

## 1. Motivation: 为什么 VLA 需要 FAN

### 1.1 现有 VLA training 的问题

主流 VLA model（OpenVLA、RT-2、π0 等）都把 action 离散化为 token，然后用 next-token prediction + cross-entropy loss 来训。这个 paradigm 直接继承自 LLM。但是 language token 和 physical action 有一个 fundamental difference：

- **Language**: 每个 token 基本是 uniquely correct。比如翻译 "猫" 到 "cat"，"dog" 就是错的，没有 neighborhood。
- **Robotics action**: 当 gripper 要往左移 1.0 cm，移 0.9 cm 或 1.1 cm 几乎产生 indistinguishable progress。这就是 **action tolerance**。

但 SFT 用 one-hot cross-entropy，强迫 model 把所有 probability mass 压在 demonstration 那一个 bin 上，导致 **spiky distribution**，generalization 很差（特别是 OOD）。RFT (PPO/GRPO) 虽然 eventually 能学到 broader distribution，但 sample efficiency 低，因为 agent 要 implicit 探索很久才能 discover 这个 beneficial property。

### 1.2 Figure 1 给出的核心 insight

Figure 1 是这篇 paper 最关键的 visualization。在 ManiSkill task 上，作者画了 policy distribution 的几何结构：

| Stage | Distribution shape | Success rate |
|---|---|---|
| (a) SFT warm-up | Spiky, peaked, minimal FAN | 48.4% |
| (b) SFT + PPO | Broader distribution | 93.8% |
| (c) SFT + FAN-PPO | Approximate Gaussian, robust | 97.4% |

**Intuition**: Policy distribution 的 "broadness" 直接对应 implicit FAN 的大小，broadness ↑ ⟹ generalization ↑。这是整篇 paper 的核心经验观察，也是 regularizer 设计的 motivation。

---

## 2. FAN 的形式化定义

### 2.1 Definition 1 (Feasible Action Neighborhood)

给定 state $s$，optimal action $a^*(s) = \arg\max_{a' \in A} Q(s, a')$。对 tolerance $\delta > 0$：

$$\mathbb{N}_\delta(s) \subseteq \{a \in A : Q(s, a^*(s)) - Q(s, a) \leq \delta\}$$

变量含义：
- $Q(s, a)$: state-action value function，expected discounted return
- $a^*(s)$: 当前 state 下的最优 action
- $\delta$: tolerance threshold，控制 FAN 的大小
- $\mathbb{N}_\delta(s)$: 包含 $a^*(s)$ 的 connected component

**关键 assumption**: 对 well-posed physical task，$\mathbb{N}_\delta(s)$ 是 non-trivial 的，即 around $a^*(s)$ 形成 non-zero volume 的 region。这就是 robotics 与 NLP 的本质区别。

### 2.2 Policy distribution 作为 FAN 的 proxy

直接 access $Q(s, a)$ 很难，但 policy $\pi(a|s)$ implicit encode 这个信息。对 softmax-over-Q 这类 policy，相似 Q-value 的 action 有相似 probability。因此：

- **Spiky distribution** ⟹ trivial FAN ⟹ poor generalization
- **Broad, smooth distribution** ⟹ large FAN ⟹ robust generalization

这个观察让作者把 "shape policy distribution" 当作训练目标，避开了直接估计 Q-function 的困难。

---

## 3. FAN-guided Regularizer 的设计

### 3.1 核心公式 (Eq. 5)

$$\mathcal{L}_{\mathrm{FAN}} = \mathbb{E}_s \left[ D_{\mathrm{KL}}(\pi(\cdot|s) \| \mathcal{N}(\cdot | \mu(s), \Sigma(s))) \right]$$

变量含义：
- $\mu(s) = \arg\max_a \pi(a|s)$: policy 自己预测的 optimal action（mode of current policy）
- $\Sigma(s)$: covariance matrix，控制 FAN 的 size
- $\mathcal{N}(\cdot | \mu(s), \Sigma(s))$: target Gaussian distribution
- $D_{\mathrm{KL}}$: KL divergence

**为什么选 Gaussian?** 物理 FAN 的三个核心 property：unimodality、smoothness、local contiguity，Gaussian 全都满足。这比 entropy maximization 更 structured——entropy max 只 encourage broad，但不约束 shape，可能是 multi-modal 或 irregular。

### 3.2 SFT 版本: FAN-SFT (Eq. 6)

$$\mathcal{L}_{\mathrm{FAN-SFT}}(\theta) = -\frac{1}{n} \sum_{i=1}^{n} \sum_{t=0}^{K^i-1} \left[ \log \pi_\theta(a_t^i | s_t^i, l^i) + \alpha D_{\mathrm{KL}}(\pi_\theta(\cdot|s_t^i, l^i) \| \mathcal{N}(\cdot | \mu(s_t^i), \Sigma(s_t^i))) \right]$$

变量含义：
- $n$: demonstration 轨迹数量
- $K^i$: 第 $i$ 条 trajectory 的长度
- $s_t^i, a_t^i, l^i$: 第 $i$ 条 trajectory 在 step $t$ 的 state、action、language instruction
- $\alpha$: regularization coefficient

**SFT 的关键设计**: covariance 是 dynamic 的：

$$\Sigma(s) = \mathrm{diag}\left(\sum_{a \in A} \pi(a|s, l)(a - \mu(s))^2\right)$$

即用 policy 自己当前的 variance 作为 target variance。**Intuition**: SFT objective 本身 stable，可以 accommodate variable target 而不失稳定性；让 policy 自己决定 FAN 的 size 更 flexible。

### 3.3 RFT 版本: FAN-PPO (Eq. 7)

RFT 用 trust-region optimization 框架，加入 FAN regularizer：

$$\max_\pi \mathbb{E}_{s \sim d_\mu^{\pi_t}, a \sim \pi_t}\left[\frac{\pi(a|s,l)}{\pi_t(a|s,l)} A^{\pi_t}(s,a,l)\right] - \alpha \mathbb{E}_s \left[D_{\mathrm{KL}}(\pi(\cdot|s,l) \| \mathcal{N}(\cdot | \mu(s), \Sigma))\right]$$

$$\text{s.t.} \quad \mathbb{E}_s[D_{\mathrm{KL}}(\pi(\cdot|s,l) \| \pi_t(\cdot|s,l))] \leq \epsilon$$

变量含义：
- $d_\mu^{\pi_t}$: discounted state visitation distribution under policy $\pi_t$
- $A^{\pi_t}(s,a,l)$: advantage function under $\pi_t$
- $\pi_t$: 当前 iteration 的 policy
- $\epsilon$: trust-region size
- $\Sigma := \sigma^2 I$: **fixed** covariance（与 SFT 不同！）

**RFT 用 fixed covariance 的原因**: RL training 本身 noisy，如果 target 还在动会 destabilize training。$\sigma$ 作为 hyperparameter 控制 target FAN size。

### 3.4 Proposition 1: Optimal policy 的 closed form

作者证明了这个 constrained optimization 有 elegant closed-form solution：

$$\pi_{t+1}(a|s,l) \propto \mathcal{N}(a | \mu(s), \Sigma)^{\frac{\alpha}{\alpha + \beta^*}} \cdot \pi_t(a|s,l)^{\frac{\beta^*}{\alpha + \beta^*}} \cdot \exp\left(\frac{Q^{\pi_t}(s,a,l)}{\alpha + \beta^*}\right)$$

变量含义：
- $\beta^* \geq 0$: trust-region constraint 的 optimal Lagrange multiplier
- $\alpha$: FAN regularization coefficient
- $Q^{\pi_t}(s,a,l)$: state-action value under $\pi_t$

**这个公式非常 illuminating**，可以拆解为三部分：

1. **$\mathcal{N}(a|\mu(s), \Sigma)^{\alpha/(\alpha+\beta^*)}$**: Gaussian prior 的 pull，强度由 $\alpha$ 控制
2. **$\pi_t(a|s,l)^{\beta^*/(\alpha+\beta^*)}$**: 上一轮 policy 的 pull（trust region），强度由 $\beta^*$ 控制
3. **$\exp(Q^{\pi_t}/(\alpha+\beta^*))$**: Q-value 的 re-weighting，鼓励 high-return action

这是 **geometric interpolation** between $\pi_t$ 和 Gaussian，再用 $Q$-value re-weight。$\alpha$ 和 $\beta^*$ 之间有 competition：

- $\alpha$ 大 ⟹ 更强 pull 向 Gaussian shape
- $\epsilon$ 小 ⟹ $\beta^*$ 大（KKT 条件）⟹ 更强 pull 向 $\pi_t$（更保守 update）

**Proof intuition**: Lagrangian 是关于 $\pi$ 的 convex function，对 $\pi(a|s,l)$ 求偏导等于 0，再 normalize 就得到 closed form。Supplementary material Section 8 有完整推导。

### 3.5 Practical FAN-PPO loss (Eq. 8)

把 Proposition 1 的思想嵌入 PPO clip loss：

$$\mathcal{L}_{\mathrm{FAN-PPO}}(\theta) = -\frac{1}{K} \sum_{k=0}^{K-1} \left[\min(I_t^k \hat{A}, \mathrm{Clip}(I_t^k, 1-\epsilon, 1+\epsilon)\hat{A}) - \alpha D_{\mathrm{KL}}(\pi_\theta(\cdot|s_k,l) \| \mathcal{N}(\cdot | \mu(s_k), \Sigma))\right]$$

变量含义：
- $I_t^k = \pi_\theta(a_k|s_k,l)/\pi_{\theta_t}(a_k|s_k,l)$: importance ratio
- $\hat{A}(s_k, a_k, l)$: GAE-estimated advantage
- $\epsilon$: PPO clip ratio（通常 0.2）

注意：FAN regularizer 直接加到 clip loss 内部，而不是 external penalty。这种 integration 让 regularization 与 policy improvement 同步进行。

---

## 4. 与 Entropy Maximization 的区别

这是 paper 里强调的关键 distinction。Entropy maximization（常用于 RL exploration）鼓励 $\pi$ 接近 uniform，是 **unstructured** spreading。FAN 则是 **structured** prior：

| Property | Entropy Max | FAN |
|---|---|---|
| Target distribution | Uniform | Gaussian centered at $\mu(s)$ |
| Shape constraint | None | Unimodal, smooth |
| Goal | Exploration | Generalization via tolerance |
| Mode | Multi-modal allowed | Single mode at optimal action |

实验上 Figure 28a 直接对比了两者：entropy max 训练更慢，且对 $\alpha$ 更敏感。

---

## 5. Experiments 详解

### 5.1 Setup

- **Backbones**: OpenVLA (single action prediction) 和 OpenVLA-OFT (action chunks, 8 步)
- **Benchmarks**: 
  - ManiSkill: PutOnPlateInScene25Main-v3，25 个 pick-and-place primitives
  - LIBERO: LIBERO-Spatial suite，10 tasks，每 task 50 demonstrations
- **OOD evaluation**: 15 种 perturbation 分三类
  - Vision: unseen table, dynamic texture, dynamic noise
  - Semantic: unseen objects/receptacles/instructions, multi-object, distractors
  - Execution: unseen robot pose, mid-episode object reposition
- **Hardware**: NVIDIA A100 80GB

### 5.2 SFT 主结果 (Table 1)

| Method | In-Dist | Vision | Semantic | Execution | OOD Avg |
|---|---|---|---|---|---|
| RL4VLA | 88.5 | 74.0 | 61.8 | 46.2 | 60.7 |
| OpenVLA + SFT | 78.1 ± 3.1 | 76.6 ± 1.9 | 57.4 ± 0.9 | 40.4 ± 0.8 | 58.1 |
| OpenVLA + FAN-SFT | **89.8 ± 0.8** | **81.7 ± 1.1** | **63.5 ± 1.5** | **44.8 ± 0.5** | **63.3** |
| Δ | +11.7 | +5.1 | +6.1 | +4.4 | +5.2 |

In-distribution 提升 +11.7% 是惊人的，说明 baseline SFT 严重 overfitting；OOD +5.2% 说明 regularizer 真的帮助了 generalization，而不只是 in-dist 拟合。

### 5.3 RFT 主结果 (Table 2)

| Method | In-Dist | Vision | Semantic | Execution | OOD Avg |
|---|---|---|---|---|---|
| OpenVLA + PPO | 95.9 ± 3.2 | 80.1 ± 0.1 | 79.7 ± 2.0 | 85.8 ± 1.8 | 81.9 |
| OpenVLA + FAN-PPO | **97.4 ± 0.7** | **85.0 ± 4.0** | **86.7 ± 1.3** | **92.6 ± 1.5** | **88.1** |
| OpenVLA-OFT + PPO | 92.3 ± 2.5 | 84.9 ± 1.1 | 49.0 ± 0.6 | 55.9 ± 1.2 | 63.3 |
| OpenVLA-OFT + FAN-PPO | **97.3 ± 1.3** | **88.1 ± 2.2** | **58.6 ± 1.0** | **67.0 ± 2.2** | **71.2** |

Execution OOD 提升最显著（OpenVLA +6.9%，OpenVLA-OFT +11.1%），因为 execution perturbation（robot pose 变化、object 重定位）最依赖 action tolerance。

### 5.4 Sample efficiency (Figure 7, 8; Table 15-18)

OpenVLA + FAN-PPO 达到 90% success rate 只需 98 steps，而 baseline PPO 需要 249 steps——**约 2.5x 加速**。这印证了作者的核心 claim：explicit regularization 让 agent 不必 implicit discover tolerance structure。

| Method | Steps to 60% SR | Steps to 70% SR | Steps to 80% SR | Steps to 90% SR |
|---|---|---|---|---|
| OpenVLA + PPO | 18 | 62 | 133 | 249 |
| OpenVLA + FAN-PPO | 18 | 37 | 56 | 98 |

### 5.5 Real-world 实验 (Table 3)

JAKA 7-DoF manipulator + Intel RealSense D455。4 个 task，每个 30 trials：

| Method | Task-1 (IND) | Task-2 (obj pose) | Task-3 (robot pose) | Task-4 (box pos) |
|---|---|---|---|---|
| OpenVLA + SFT | 19/30 | 7/30 | 7/30 | 1/30 |
| OpenVLA + FAN-SFT | 22/30 | 12/30 | 17/30 | 7/30 |

Task-4 (box position 扰动) baseline 几乎完全失败（1/30），FAN-SFT 提升 7x。这非常符合 FAN 的 intuition——spatial shift 最需要 action tolerance。

### 5.6 Hyperparameter sensitivity

**$\alpha$ (regularization coefficient)**: ManiSkill SFT 最优 $\alpha = 0.05$；RFT OpenVLA 最优 $\alpha = 1.0$，OFT 用 $\alpha = 0.1$。过大（如 $\alpha = 2.0$）会 destabilize training，甚至 collapse（$\alpha = 5.0, 10.0$）。

**$\sigma$ (Gaussian std)**: OpenVLA 用 $\sigma = 0.3$，OFT 用 $\sigma = 0.2$。$\sigma$ 太小（如 0.05）退化为 single bin；$\sigma \in [0.1, 2.0]$ 性能相对 stable。

### 5.7 Ablation: Label Smoothing vs FAN (Table 12)

| Method | In-Dist | OOD Avg |
|---|---|---|
| Original | 78.1 | 58.1 |
| + Label smoothing $\epsilon=0.05$ | 82.8 | 60.1 |
| + Label smoothing $\epsilon=0.1$ | 81.3 | 56.3 |
| + FAN-SFT (ours) | **89.8** | **63.3** |

Label smoothing 也有 modest benefit，但远不如 FAN。这印证了 **structured > unstructured** regularization 的核心论点。

### 5.8 Ablation: Multi-modal target (Figure 28b)

作者还试了 Gaussian-kernel-smoothed multi-modal target $q_\kappa = \mathrm{Normalize}(K_\kappa \pi)$，结果有提升但不如 single-modal Gaussian。这说明 unimodality 是 FAN 的关键 property。

---

## 6. 与其他 VLA finetuning 工作的关系

### 6.1 vs RL4VLA / SimpleVLA-RL / RLAIF-VLA

这些工作直接把 PPO/GRPO 搬到 VLA 上，没有 exploit action tolerance。FAN 是 orthogonal contribution，可以 plug-in 到这些方法里。

### 6.2 vs VLA-RL (RPRM) / GRAPE / TGRPO

这些工作改进 reward design（dense reward、VLM-based reward），FAN 改进的是 policy optimization objective 本身。理论上可以组合。

### 6.3 vs $\pi_0$ / HybridVLA / VQ-VLA

这些工作改 action representation（flow matching、diffusion、VQ-VAE）。FAN 假设 discrete bin tokenization，但 idea 可以推广——只要 policy 是 distributional 的，就可以加 KL regularizer。

### 6.4 vs LIBERO-PRO / LIBERO-Plus

这些 benchmark 专门测 VLA 的 robustness，揭示 vanilla SFT 在 OOD 下 catastrophic degradation。FAN 直接 attack 这个问题。

---

## 7. 我的思考与延伸

### 7.1 为什么 dynamic Σ 在 SFT work，fixed Σ 在 RFT 必须？

SFT 的 supervised signal 是稳定的 ground truth action，policy 不会被 noisy reward 推得很远，所以可以让 target covariance 跟着 policy variance 走。RFT 的 advantage 估计本身有 high variance，如果 target 也在动，相当于 double noise，training 容易 collapse。这是一个值得 generalize 的 design principle：**regularizer 的 stability 应该 match training objective 的 stability**。

### 7.2 FAN 和 consistency policy / diffusion policy 的关系

Diffusion policy (Chi et al.) 和 $\pi_0$ 的 flow matching 本质上也是建模 multi-modal action distribution。它们 implicit 允许 action tolerance。但 FAN 的角度不同——它不改变 action representation，而是改 training objective。理论上 FAN 的思想可以加到 diffusion policy training 里（比如 KL regularize score function 的 mode）。

### 7.3 FAN 和 Q-function 的 connection

FAN 定义在 $Q(s, a)$ 上，但实际只用 policy distribution。这里有个 implicit assumption: policy 的 mode 与 $a^*(s)$ 对齐。如果 policy 学错了 mode（比如 mode 在 suboptimal action），FAN regularizer 会强化错误。SFT warm-up 阶段保证了 mode 大致正确，这是为什么 FAN-PPO 需要 SFT warm-up 的原因。

### 7.4 Extension: 学习 $\sigma(s)$ 而不是固定

当前 RFT 用 fixed $\sigma$。更 principled 的做法是让 $\sigma(s)$ 是 state-dependent 的——简单 task tolerance 大，precision task tolerance 小。这可以做成 auxiliary head 预测 $\sigma(s)$，或者从 environment dynamics 推断。但会增加 hyperparameter 和 training complexity。

### 7.5 Connection to Bayesian policy / KL-regularized RL

FAN-PPO 的 objective 实际上是 KL-regularized policy optimization 的特例。一般 KL-regularized RL 用 reference policy $\pi_{\mathrm{ref}}$ 作为 prior：

$$\max_\pi \mathbb{E}[Q(s,a)] - \beta D_{\mathrm{KL}}(\pi \| \pi_{\mathrm{ref}})$$

FAN 把 $\pi_{\mathrm{ref}}$ 换成 Gaussian $\mathcal{N}(\mu(s), \Sigma)$，而 $\mu(s)$ 是 policy 自己的 mode。这是一种 **self-referential prior**——policy 用自己的 mode 作为 anchor，但 shape 被 Gaussian 约束。这个角度可以连接到 mirror descent、REPS (Relative Entropy Policy Search) 等经典 RL 工作。

### 7.6 Limitations 没明说但值得注意

1. **Discrete bin 假设**: FAN 假设 action 是 discretized bin。对 continuous action（如 $\pi_0$ 的 flow matching），需要重新设计 regularizer。
2. **Single mode 假设**: FAN 的 Gaussian 是 unimodal。但有些 task 真正的 FAN 可能是 multi-modal（比如可以绕左或绕右避障）。作者用 Gaussian-kernel smoothing 试了 multi-modal，效果不如 Gaussian，但这可能是因为 smoothing 的方式不够好。
3. **$\mu(s)$ 的估计**: 用 $\arg\max_a \pi(a|s)$ 估计 mode 在 discrete bin 下是 well-defined，但需要 policy 已经 reasonably trained。Cold start 时 mode 可能 noisy。
4. **Covariance 形式**: 当前用 diagonal covariance $\sigma^2 I$，没考虑 action dimension 之间的 correlation。真实 FAN 可能是 anisotropic 的。

---

## 8. 总结

这篇 paper 的核心 contribution 是把 **physical action tolerance** 这个 robotics 特有的 inductive bias 显式 encode 到 training objective 里。技术实现非常 simple——一个 KL regularizer——但 motivation 充分，理论 closed form 优雅，实验全面。它揭示了 VLA training 的一个 fundamental mismatch：**language-style "exclusive correctness" 与 physical "weighted tolerance" 的对立**。

更深层地看，这是 **embodied AI 摆脱 language model recipe** 的一个 signal work。VLA 不只是 VLM + action head，physical world 有自己的 structure（tolerance、smoothness、contiguity），training objective 应该 reflect 这些 structure。FAN 是这个方向的一个 concrete step，未来可能延伸到 temporal smoothness、kinematic constraint、physical dynamics consistency等更丰富的 structure。

---

## References

- [OpenVLA](https://openvla.github.io/) — Kim et al., 2024
- [OpenVLA-OFT](https://openvla-oft.github.io/) — Kim et al., 2025
- [RT-2](https://robotics-transformer2.github.io/) — Zitkovich et al., CoRL 2023
- [$\pi_0$](https://arxiv.org/abs/2410.24164) — Black et al., 2024
- [ManiSkill3](https://maniskill.readthedocs.io/) — Tao et al., 2025
- [LIBERO](https://libero-project.github.io/) — Liu et al., NeurIPS 2023
- [LIBERO-PRO](https://arxiv.org/abs/2510.03827) — Zhou et al., 2025
- [PPO](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017
- [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300) — Shao et al., 2024
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — RL finetuning for LLMs
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) — Chi et al., RSS 2023
- [Open X-Embodiment](https://robotics-transformer-x.github.io/) — collaboration, ICRA 2024
- [RL4VLA](https://arxiv.org/abs/2509.09674) — Liu et al., NeurIPS 2025
- [VLA-RL](https://arxiv.org/abs/2505.18719) — Lu et al., 2025
- [SayCan](https://say-can.github.io/) — Brohan et al., CoRL 2023
- [VoxPoser](https://voxposer.github.io/) — Huang et al., 2023
- [Code as Policies](https://code-as-policies.github.io/) — Liang et al., ICRA 2023
- [DINOv2](https://dinov2.metademolab.com/) — Oquab et al., TMLR 2024
- [SigLIP](https://arxiv.org/abs/2303.15343) — Zhai et al., ICCV 2023
- [Llama 2](https://arxiv.org/abs/2307.09288) — Touvron et al., 2023
- [GAE](https://arxiv.org/abs/1506.02438) — Schulman et al., 2015
- [Label Smoothing / Inception](https://arxiv.org/abs/1512.00567) — Szegedy et al., CVPR 2016
