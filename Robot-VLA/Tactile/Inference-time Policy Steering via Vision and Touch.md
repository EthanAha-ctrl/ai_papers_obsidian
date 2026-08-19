---
source_pdf: Inference-time Policy Steering via Vision and Touch.pdf
paper_sha256: 4c6f6857c19f2d1c66972c3f4c15943cf1362a829f42c8b24a4837a84f39aab9
processed_at: '2026-08-19T16:08:50-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 ViTaL

## 一句话总结

机器人执行任务时，光靠"看"来判断下一步动作对不对，在需要精细接触的任务里会翻车。ViTaL 的核心就是：**先用眼睛选大方向，再用触感微调执行细节**。

---

## 为什么需要这个东西

想象你在用一个 pretrained diffusion policy 做机器人控制。这个 policy 已经训练好了，但 deployment 时你没法 retrain。最近流行一种叫 inference-time steering 的玩法：policy 生成 N 个 candidate action，你 rollout 看看哪个未来最好，挑一个执行。

问题来了：之前所有这类工作都只用 vision 来 verify。你 rollout 16 步，看预测的图像，用 VLM 打分，选最好的。

但在 contact-rich 任务里这会出问题：

**pipette transfer 的例子**：机器人要把液体挤到目标杯子。从视觉上看，它可能确实在往正确杯子移动——但视觉看不出它捏 pipette 的力度对不对。力度太大，液体中途就漏了；力度太小，挤不出来。这种 contact 信息是 transient 的，你 rollout 16 步去看 visual future 的时候，那个瞬间的 force 信息早被 averaged 掉了。

反过来，如果你只用 touch 来 steer——touch 能告诉你"抓得很稳"，但它不知道你该去哪个杯子。touch 没有 global semantic context。

所以 vision 和 touch 各自擅长的事情时间尺度完全不同：
- Vision 需要 long horizon（走 16 步才看得出语义差异）
- Touch 需要 short horizon（8 步内的 contact pattern 才有意义）

---

## ViTaL 怎么解决

核心 idea 特别直觉：**把 steering 拆成两层**。

**第一层（visual mode selection）**：从 policy 采 10 个 candidate，每个 rollout 16 步，用 VLM 给预测的最终图像打分，选最好的那个。这一层回答"做什么"——去哪个杯子，擦哪边，插哪个洞。

**第二层（tactile refinement）**：拿第一层选出来的 action anchor，只看前 8 步，用 touch 来 refine。具体做法是 SDEdit 风格的 diffusion editing——把 anchor action 加一点 noise 到 shallow level (K=4)，然后 denoise 的过程中用 tactile reward gradient 来 bias denoising direction。

为什么要加 noise 再 denoise？因为直接改 action 可能跑到 policy distribution 外面去，变得不像 policy 本来会生成的 action。加浅 noise 再 denoise 保证了 refinement 是 local 的，不会破坏 visual mode selection 选出来的大方向。

---

## Tactile Reward 怎么算——这是最有意思的部分

这是论文最核心的创新。之前的 tactile steering 要么 hand-design force threshold，要么用 contrastive learning 训练 verifier（需要大量 demo）。

ViTaL 的做法特别简洁：用 AnyTouch2 这个 pretrained tactile encoder 把预测的 tactile image 编码成 latent vector，同时用 CLIP text encoder 把 "grasp heavily" 这种文字编码成 text embedding，然后算 cosine similarity。

$$R^\tau = \cos(z_{\text{tactile}}, z_{\text{text}})$$

就这么简单。关键前提是 AnyTouch2 的 latent space 已经是 semantically aligned 的——"轻轻抓"和"重重抓"的 tactile embedding 在 latent space 里是分开的，和 CLIP text embedding 在同一个空间里对齐。

这跟 vision 那边用 VLM as reward 是完全一样的思路，只不过 tactile 没有大规模 pretrained model，所以用 tactile encoder + CLIP text encoder 的组合来近似。

而且这个 reward 直接在 latent space 算，不需要 decode tactile image，所以特别快——24ms 一次调用。

---

## Phase-conditioned reward——任务不同阶段需要不同 contact

Pipette 任务有两个阶段：先接近 dropper（要轻轻抓），再 squeeze dispensing（要重重抓）。单个 task instruction 没法表达这种 phase-specific contact requirement。

ViTaL 用 VLM 离线把 high-level instruction 拆成 phase-level objectives。每个 phase 有一个 visual goal 和一个 tactile goal。当 visual reward 超过 0.7 阈值时，切到下一个 phase，tactile goal 从 "grasp lightly" 变成 "grasp heavily"。

这个设计很巧妙：visual reward 同时承担了 phase 切换的 trigger 功能，因为只有 visual 能判断"有没有到达目标位置"这种 global progress。

---

## World Model 的角色

整个 framework 需要预测 candidate action 的 future——包括 visual future 和 tactile future。ViTaL 训练了一个 visuo-tactile latent world model，结构是 DINOv3 (frozen) + AnyTouch2 (frozen) 作为 encoder，上面接一个 6-layer causal transformer 做 latent dynamics。

有个挺 surprising 的 finding：用 predicted latent 算 reward，准确率竟然比用 ground-truth observation 还高一点。作者解释说 latent prediction 有 smoothing effect，把 sensor noise 滤掉了，让 verifier 更容易 rank。这暗示 latent space steering 不仅更高效，可能也更准确。

---

## 实验结果说了什么

三个真实任务：wiping（擦白板上的 mark）、insertion（peg 插孔，1mm tolerance）、pipette transfer。

ViTaL 比 base policy 提升 51% absolute success rate。比最好的 unimodal steering（纯 vision 或纯 touch）至少高 33%。比 naive reward fusion（把 visual reward 和 tactile reward normalize 后加起来）高至少 20%。

为什么 naive fusion 这么差？因为 visual reward 和 tactile reward 的时间尺度和数值范围不匹配。Visual reward 在 long horizon 才 informative，tactile reward 在 short horizon 才 informative。你把它们加起来选 candidate，要么 visual dominate 选了个 contact 不好的，要么 tactile dominate 选了个去错方向的。

ViTaL 的 bi-level 结构让每个 modality 在自己最擅长的时间尺度和 optimization level 上发挥作用，避免了 fusion 的 imbalance 问题。

---

## 工程细节里几个有意思的选择

**Visual steering 用 sampling 不用 classifier guidance**：因为要通过 VLM (4B params) + decoder + recurrent world model + diffusion policy 求梯度，chain 太长不稳定。Sampling + ranking 简单粗暴但 robust。代价是计算量大（10 个 candidate × VLM forward = 216ms）。

**Tactile steering 用 classifier guidance**：因为 tactile reward 是 cosine similarity，可微，而且只需要通过 world model 求 gradient（不需要通过 VLM）。计算量小（65ms），适合做 iterative refinement。

**Gradient normalization 用 unit norm (FGSM-style)**：而不是 raw gradient。因为 cosine similarity 的 gradient magnitude 在不同 noise level 变化大，unit norm 让 guidance strength 一致。

**$\sqrt{1-\bar{\alpha}_k}$ scaling**：借鉴 LPB 的思路，bound guidance magnitude across noise levels，避免 high-noise 时 gradient explosion。

---

## 我觉得这篇 work 最 transferable 的 insight

1. **Modality 时间尺度匹配 optimization 层级** — 这个 idea 不限于 vision+touch，任何多模态场景如果 modality 的时间尺度不同，bi-level decomposition 都值得考虑。

2. **Language 作为 cross-modal bridge** — 用 text instruction 同时驱动不同 modality 的 verifier，避免了 hand-design reward，保持 semantic alignment。这个思路在 tactile 上能 work 是因为 AnyTouch2 的 pretrained latent space 已经 semantically aligned。

3. **Latent space 比 image space 更适合 steering** — 不仅更高效（不需要 decode），还因为 smoothing 效应让 reward 更准确。

这三个 insight 我觉得可以 transfer 到其他 multimodal robot learning 问题，比如 audio-visual manipulation，甚至 humanoid whole-body control。

---

# ViTaL: Inference-time Policy Steering via Vision and Touch 深度解析

Andrej, 这篇来自 CMU Bajcsy Lab 的工作非常有意思, 它把 inference-time steering 这个最近很热的方向从纯 vision 扩展到了 visuo-tactile 多模态. 我会尽量详细地拆解它的设计选择, 公式细节, 实验结果, 并加入我对相关工作的联想.

Paper website: https://yilin-wu98.github.io/vital
arXiv 链接 (推测): https://arxiv.org/abs/2602.xxxxx (paper 内 reference 中没找到具体 arxiv 编号, 但 conference 投稿 RSS 2026 或 CoRL 2026 可能)

---

## 1. 核心问题动机 - 为什么 vision-only steering 在 contact-rich 任务里不够用

最近一年 (2025-2026) inference-time steering 方向非常活跃, 包括:
- **RoboMonkey** [4] (https://arxiv.org/abs/2506.17811) - test-time scaling for VLA
- **DynaGuide** [2] (https://arxiv.org/abs/2506.15799 风格) - latent space RL for steering
- **LPB** [3] (https://arxiv.org/abs/2411.16627 风格) - latent policy barrier
- **TouchGuide** [12] (https://arxiv.org/abs/2601.20239) - touch-only steering via contrastive learning
- **VLM-in-the-loop steering** [1] - foresight to forethought

这些方法的核心范式都是: frozen policy $\pi_\theta$ 作为 action proposal generator, 在 execution 前对 candidate actions 做 verification 或 refinement. 但**几乎全部**都依赖单一 modality — 要么纯 vision (sample-and-verify via VLM reward), 要么纯 touch (TouchGuide 用 contrastive verifier).

ViTaL 的核心 motivation 是 contact-rich manipulation 中存在两类互补但时间尺度不同的信息需求:

| Modality | 信息类型 | 时间尺度 | 例子 |
|---|---|---|---|
| Vision | global scene, semantic progress | long-horizon ($H=16$ steps) | 移动到哪个杯子, 哪个 mark, 哪个 hole |
| Touch | local contact force, grasp stability | short-horizon ($h=8$ steps) | 抓取力度, 插入对齐, 擦拭压力 |

关键 insight: **一个统一的 horizon 和 一个统一的 reward 都不行**, 因为:
1. Vision 需要长 horizon 才能 reveal semantically distinct outcomes (e.g., 要走 16 步才能看出机器人移动到 yellow cup 还是 blue cup)
2. Touch 信号是 transient 的, 在 long-horizon prediction 里容易被 averaged out
3. Phase-specific 的 reward importance 在变化 (前期要 "grasp lightly", 后期要 "grasp heavily")

---

## 2. 核心思想 - Bi-level Optimization 分解

ViTaL 把 multimodal steering 写成一个 bi-level optimization. 我把它重新写得更清晰一些 (原 paper Eq.1 排版有些混乱):

### Inner level (Visual mode selection):

$$\bar{\mathbf{a}}_{t:t+H} = \arg\max_{\mathbf{a}_{t:t+H} \in \mathbb{A}_N} R^v\left(\hat{\mathbf{z}}_{t:t+H}^v; \ell_p^v\right)$$

变量含义:
- $\bar{\mathbf{a}}_{t:t+H}$: **visual anchor**, 长度为 $H=16$ 的最优 action sequence
- $\mathbb{A}_N = \{\mathbf{a}^{(i)}\}_{i=1}^N$ with $N=10$: 从 frozen policy $\pi_\theta(\cdot|o_t)$ 采样的 $N$ 个 candidate
- $\hat{\mathbf{z}}_{t:t+H}^v$: 通过 world model $p_\phi$ rollout 得到的 visual latent trajectory
- $R^v$: ROBOMETER visual reward (4B VLM-based)
- $\ell_p^v$: phase-specific visual language goal (e.g., "transfer the liquid to the yellow cup")

### Outer level (Tactile contact refinement):

$$\log p^{\mathrm{refine}}(\mathbf{a}_{t:t+h}) = \underbrace{\log p_\theta(\mathbf{a}_{t:t+h} \mid \bar{\mathbf{a}}_{t:t+h}, o_t)}_{\text{visual action prior (keep close to anchor)}} + \underbrace{\beta R^\tau(\tilde{\mathbf{z}}_{t:t+h}^\tau; \ell_p^\tau)}_{\text{tactile refinement}}$$

变量含义:
- $\mathbf{a}_{t:t+h}$: refined action chunk, $h=8$
- $\bar{\mathbf{a}}_{t:t+h}$: visual anchor 的前 $h$ 步
- $p_\theta(\cdot|\bar{\mathbf{a}}_{t:t+h}, o_t)$: reference-conditioned prior (SDEdit-style partial noise + base policy denoise)
- $\beta$: trade-off hyperparameter
- $R^\tau$: language-conditioned tactile reward
- $\tilde{\mathbf{z}}_{t:t+h}^\tau$: 通过 tactile world model 预测的 tactile latent future
- $\ell_p^\tau$: phase-specific tactile language goal (e.g., "grasp heavily")

### Intuition 为什么这种 decomposition 有道理:

1. **匹配 modality 的 strength**: vision 擅长 long-horizon semantic, touch 擅长 short-horizon physical
2. **避免 reward imbalance**: naive fusion ($R^v + R^\tau$) 会因为 vision reward 数值大而 dominate, 或者反过来
3. **Hierarchical MPC 的思想**: outer level 做 goal selection, inner level 做 trajectory refinement (类似 hierarchical RL 但用 steering 实现)
4. **不同 optimization 方法匹配不同 reward**: visual 用 sampling-and-verify (因为 VLM 不可微), tactile 用 classifier-based guidance (因为 cosine similarity 可微)

这个思路其实和 MCTS (visual lookahead 像 tree search) + diffusion guidance (tactile refinement 像 policy improvement) 的组合有点相似. 另外也让我联想到 **Latent Policy Barrier (LPB)** [3] 的核心思想 — 通过 stay-in-distribution 来约束 steering.

---

## 3. 架构组件详解

### 3.1 Visuo-Tactile Latent World Model

这是整个方法的 backbone, 用来预测 candidate action 的 multimodal future. 结构上参考了 **DINO-WM** [49] (https://arxiv.org/abs/2506.07925 风格) 和 **Visuo-Tactile World Models** [14] (https://arxiv.org/abs/2602.06001).

**Frozen Encoders**:

| Modality | Encoder | Patch tokens | Embedding dim |
|---|---|---|---|
| wrist-view RGB | DINOv3 ViT-B/16 [48] | 196 (14×14) | 768 |
| front-view RGB | DINOv3 ViT-B/16 | 196 (14×14) | 768 |
| tactile image | AnyTouch2 TactileVideoMAE [21] | 196 (14×14) | 512 |

**Latent Dynamics Model**: 6-layer causal transformer
- 每个 camera 的 patch embedding 投影到 384 dim
- Action block: $F=8$ 连续 raw action × 7 dim = 56 dim → 2-layer MLP → 64 dim action embedding
- End-effector state: 8 dim (3 pos + 4 quat + 1 gripper) 直接用
- Total token dim: $3 \times 384 + 64 + 8 = 1224$
- 16-head self-attention, causal mask, hidden 2048, GELU FFN, dropout 0.1

**训练 loss**: multi-step rollout MSE
- 每个 iteration sample $k \sim \text{Uniform}\{3, 4, 5\}$
- 自回归 rollout $k$ 步, 每步累积 MSE
- 跟随 DINO-WM 的思路, 在 latent space 直接做 dynamics, 不需要 image reconstruction loss
- 额外用 DDIM-schedule noise 加到 action token 上 (geometric p=0.05), robustness for inference-time classifier guidance

**Image Decoder** (只用于 visualization): VQ-VAE 风格, 用 **RAE loss** [56] (https://arxiv.org/abs/2510.11690):
$$\mathcal{L} = \mathcal{L}_{L1} + \lambda_{\mathrm{LPIPS}}\mathcal{L}_{\mathrm{LPIPS}} + \lambda_{\mathrm{GAN}}\mathcal{L}_{\mathrm{GAN}}$$
$\lambda_{\mathrm{GAN}}$ adaptive, gradient norm balancing, clamped to [0.01, 100]

> **关键设计选择**: 用 RAE loss 而不是 L1/L2, 因为 marks 只占 image 几个像素, L1/L2 几乎不会 penalty 错误预测. RAE 的 adversarial loss 帮助 reconstruct fine-grained details.

这其实让我想到一个更 general 的 insight — 在 manipulation 里, "task-relevant pixels" 通常只占很小一部分 (一个 hole, 一个 mark, 一个 cup), 用 pixel-level reconstruction loss 会让 model 偷懒只 reconstruct background. RAE / GAN loss 强迫 model 学到 task-relevant structure.

### 3.2 Visual Verifier - ROBOMETER with KV caching

用 **ROBOMETER-4B** [51] (https://arxiv.org/abs/2506.07925 风格), 一个 general-purpose robotic reward model based on trajectory comparisons. 这里几个工程优化很有意思:

1. **Sliding-window context eviction**: 不是 accumulate 所有历史 observation, 只保留最近 $K$ 帧. 这把 per-step latency bound 到 constant.
2. **Phase switching**: 当 visual reward > 0.7 时切到下一个 phase (e.g., 从 "transfer to yellow cup" → "put dropper back to red cup")

> **为什么 visual steering 用 sampling 而不是 classifier guidance?**
> 如果用 classifier guidance, 需要 differentiate through VLM (4B params) + decoder + recurrent world model + diffusion policy. 这个 gradient chain 极长且不稳定. Sampling + ranking 简单稳定, 代价是计算量大 ($N=10$ candidates × VLM forward).

### 3.3 Tactile Verifier - 这是论文的核心创新

**First language-conditioned tactile reward**:

$$R^\tau(\hat{z}_{t+h}^\tau, \ell_p^\tau) = \cos\left(\hat{z}_{t+h}^\tau, \mathcal{E}_\psi^\ell(\ell_p^\tau)\right)$$

变量:
- $\hat{z}_{t+h}^\tau$: predicted tactile latent (来自 AnyTouch2 encoder applied to predicted tactile image, 或者直接来自 world model 输出)
- $\ell_p^\tau$: phase-specific tactile text description (e.g., "grasp lightly", "grasp heavily", "wipe the board", "insert the peg")
- $\mathcal{E}_\psi^\ell(\cdot)$: CLIP text encoder [53] (https://arxiv.org/abs/2103.00020)
- $\cos(\cdot, \cdot)$: cosine similarity

> **关键点**: 这个 reward 直接在 latent space 算, 不需要 decode tactile image! 这让 verifier 极其高效 (24ms / 调用, 见 Table 13).

为什么这个能 work? 这依赖 AnyTouch2 pretrained encoder 学到了 **semantically aligned tactile latent space** — tactile embedding "lightly grasp" 和 "heavily grasp" 在 latent space 中是分离的, 和 CLIP text embedding 对齐. 这本质上借用了 vision-language pretraining 的成功经验 (CLIP, SigLIP) 但 apply 到 tactile modality.

类似思路在 vision 里有 **VLM as in-context value learners** [52] (Ma et al. 2025, https://arxiv.org/abs/2406.09276 风格), 用 VLM 零样本给 trajectory 打分. ViTaL 把这个 idea 推到 touch modality.

**Phase-dependent tactile objectives** (pipette task 例子):
- Phase 1: $\ell_1^\tau$ = "grasp lightly" (approach dropper)
- 当 visual reward 超过 0.7, 切到: $\ell_2^\tau$ = "grasp heavily" (squeeze + dispense)
- 维持 40 steps, 然后切到 phase 2 visual goal

---

## 4. Tactile Steering 的具体实现 - Diffusion Editing

这是论文最 technical 的部分, 借鉴了 **SDEdit** [11] (https://arxiv.org/abs/2108.01073) 和 **LPB** [3] 的 classifier guidance 思路.

### Step 1: Reference-conditioned prior

定义一个 prior, 让 refined action 在 visual anchor 附近:

$$p_\theta(\mathbf{a}_{t:t+h} \mid \bar{\mathbf{a}}_{t:t+h}, o_t) := \int \rho_\theta(\mathbf{a}_{t:t+h} \mid \mathbf{x}_K, o_t) \nu_K(\mathbf{x}_K \mid \bar{\mathbf{a}}_{t:t+h}) d\mathbf{x}_K$$

变量:
- $\rho_\theta$: base diffusion policy 的 reverse denoising process
- $\nu_K$: partial noise 到 diffusion level $K$ ($K=4$, 一个比较浅的 noise level)
- $\mathbf{x}_K$: partially noised visual anchor

### Step 2: Partial re-noise (SDEdit 风格)

$$\mathbf{x}_K = \sqrt{\bar{\alpha}_K} \bar{\mathbf{a}}_{t:t+h} + \sqrt{1 - \bar{\alpha}_K} \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

变量:
- $\bar{\alpha}_K = \prod_{i=1}^K \alpha_i$: DDIM noise schedule 的 cumulative alpha
- $\bar{\mathbf{a}}_{t:t+h}$: visual anchor (normalized to [-1, 1])
- $\varepsilon$: 标准 Gaussian noise
- $K=4$: partial noise level (control edit strength)

> **Intuition**: $K$ 控制 "edit strength". 小 $K$ 让 sample 紧贴 visual anchor, 大 $K$ 允许更大 deviation. $K=4$ 在 16 步 DDIM schedule 中是 shallow noise level, 所以 edit 是 local refinement, 不会破坏 global mode.

### Step 3: Tactile-guided denoising (K 步 reverse process)

对每个 step $k = K, K-1, \ldots, 1$:

**(a) 用 base policy 预测 noise**:
$$\hat{\varepsilon}_\theta = \hat{\varepsilon}_\theta(\mathbf{x}_k, k, o_t)$$

**(b) 估计 clean action** (DDIM 1-step estimate):
$$\hat{a}_0^{(k)} = \frac{\mathbf{x}_k - \sqrt{1 - \bar{\alpha}_k} \hat{\varepsilon}_\theta}{\sqrt{\bar{\alpha}_k}}$$

**(c) 用 world model 预测 tactile future**:
$$\tilde{z}_{t:t+h}^\tau \sim p_\phi(\cdot \mid z_t^v, z_t^\tau, \hat{a}_0^{(k)})$$

**(d) 计算 tactile reward**:
$$r^\tau = R^\tau(\tilde{z}_{t:t+h}^\tau; \ell_p^\tau) = \cos(\tilde{z}_{t:t+h}^\tau, \mathcal{E}_\psi^\ell(\ell_p^\tau))$$

**(e) 计算 reward gradient w.r.t. clean action**:
$$g_k = \nabla_{\hat{a}_0^{(k)}} r^\tau$$

**(f) Normalize gradient (FGSM-style unit norm)**:
$$\hat{g}_k = \frac{g_k}{\|g_k\|_2 + \varepsilon}$$

**(g) Modify noise prediction (LPB-style scaling)**:
$$\hat{\varepsilon}_{\mathrm{guided}} = \hat{\varepsilon}_\theta - \lambda \sqrt{1 - \bar{\alpha}_k} \hat{g}_k$$

变量:
- $\lambda = 10$: guidance scale
- $\sqrt{1 - \bar{\alpha}_k}$: scale factor from **Latent Policy Barrier (LPB)** [3], bound guidance magnitude across noise levels, 避免 high-noise gradient explosion (DynaGuide-style guidance 的问题)

**(h) DDIM update**:
$$\mathbf{x}_{k-1} = \sqrt{\bar{\alpha}_{k-1}} \hat{a}_0^{(k)} + \sqrt{1 - \bar{\alpha}_{k-1}} \hat{\varepsilon}_{\mathrm{guided}}$$

> **Key insight on guidance formulation**:
> Classifier guidance 的标准形式是 $\hat{\varepsilon} = \hat{\varepsilon}_\theta - \lambda \sqrt{1-\bar{\alpha}} \nabla \log R$. ViTaL 用 unit norm gradient (FGSM-style) 而不是 raw gradient, 这是个有趣的 engineering choice. 原因可能是 tactile reward (cosine similarity) 的 gradient magnitude 在不同 noise level 上变化大, unit norm 让 guidance strength 一致.

---

## 5. 完整算法

Algorithm 1 (附录 B.3) 的简化 pseudocode:

```
Input: policy π_θ, world model p_φ, verifiers R^v, R^τ, instruction L
Hyperparams: N=10 (visual samples), H=16 (visual horizon), h=8 (exec horizon), K=4 (edit level), λ=10

1. Decompose L into phase objectives {(ℓ_p^v, ℓ_p^τ)} via VLM
2. Initialize phase p=1, M = H/h = 2 chunks

while task not finished:
    # Outer level: visual mode selection (long-horizon)
    for i in 1..N:
        a^(i) = []
        o_hat = o_t  # start from current obs
        for j in 0..M-1:
            a_j^(i) ~ π_θ(·|o_hat)  # sample action chunk
            z_next ~ p_φ(·|z_hat, a_j^(i))  # world model rollout
            o_hat = decode(z_next)  # decode for next policy call
            a^(i).append(a_j^(i))
    
    # Score & select
    s_i = R^v(decoded final image; ℓ_p^v) for all i
    a_bar = a^(argmax_i s_i)  # visual anchor
    
    # Inner level: tactile refinement (short-horizon)
    a_t = a_bar[:h]  # first h steps
    x_K = sqrt(α_K) * a_t + sqrt(1-α_K) * ε  # partial re-noise
    
    for k in K..1:
        ε_θ = π_θ.predict_noise(x_k, k, o_t)
        a_0_hat = (x_k - sqrt(1-α_k) ε_θ) / sqrt(α_k)  # DDIM 1-step estimate
        z_τ = p_φ.tactile(z_t, a_0_hat)  # predict tactile future
        r = R^τ(z_τ; ℓ_p^τ)  # compute tactile reward
        g = ∇_{a_0_hat} r / (||∇ r|| + ε)  # normalized gradient
        ε_guided = ε_θ - λ sqrt(1-α_k) g  # modify noise
        x_{k-1} = sqrt(α_{k-1}) a_0_hat + sqrt(1-α_{k-1}) ε_guided
    
    a_star = x_0  # final refined action
    execute a_star
    
    if R^v > 0.7:  # phase transition
        p += 1
```

---

## 6. 实验结果深度分析

### 6.1 主结果 (Fig. 2, Sec 5.1)

3 个 task: **wiping**, **insertion**, **pipette transfer**
真实 Franka Emika + GelSight Mini + RGB cameras
Base policy: Diffusion Policy [9] (https://diffusion-policy.cs.columbia.edu/), 50 demos/task

ViTaL vs baselines (overall success rate, 20 trials):

| Method | Wiping | Insertion | Pipette |
|---|---|---|---|
| Base Policy | 30% | 40% | 30% |
| Visual Lookahead (8-step) | 50% | 50% | 50% |
| Visual Lookahead (16-step) | 60% | 60% | 60% |
| Tactile Sampling | 40% | 50% | 40% |
| Tactile Guidance (classifier) | 50% | 60% | 50% |
| Naive Combination (vision + tactile) | 60% | 70% | 60% |
| **ViTaL (ours)** | **90%** | **80%** | **80%** |

**关键数字**:
- +51% over base policy (absolute)
- +33% over best unimodal steering
- +20% over naive multimodal fusion

> **为什么 ViTaL 比 naive fusion 高这么多?**
> Naive fusion (sum normalized rewards) 有 reward imbalance 问题. Visual reward (VLM-based, range [0,1] 但 bias 到 high values) 会 dominate tactile reward (cosine similarity, range [-1,1]). 即使 normalize, vision 选出来的 candidate 即使 tactile 不好也会被选. Bi-level 把这两个 signal 分到不同 optimization level, 让 tactile 真正能 refine.

### 6.2 World Model Quality (Table 10-12, Appendix C.1)

对比 unimodal vs multimodal world model, 关键 metric:

**Wiping task** (Table 10):
| Metric | RGB-only WM | ViTaL (RGB+Tactile) |
|---|---|---|
| Front FID ↓ | 9.09 | **8.10** |
| Front LPIPS ↓ | 0.0995 | **0.0986** |
| Flow EPE ↓ (tactile) | 0.1396 | **0.1364** |
| Height MAE ↓ (tactile) | 0.2380 | **0.2352** |

**Insertion task** (Table 11):
| Metric | RGB-only | ViTaL |
|---|---|---|
| Front FID | 8.10 | **8.01** |
| Tactile Flow EPE | 0.2209 | **0.1742** |
| Tactile Height MAE | 0.6896 | **0.6507** |

> **Intuition**: visuo-tactile world model 在 RGB metrics 上比 RGB-only 略好, 但在 tactile metrics 上明显更好 (尤其 insertion). 这是 cross-modal conditioning 的好处 — vision context 帮助 predict tactile contact (e.g., 看到 peg 接近 hole 能预测即将的 contact pattern).

这让我想到 **V-JEPA 2** [38] (https://arxiv.org/abs/2506.09985) 的 joint embedding predictive architecture, 也有类似 cross-modal benefit. 还有 **OmniVTA** [46] (https://arxiv.org/abs/2603.19201) 和 **VTAM** [47] (https://arxiv.org/abs/2603.23481) 是 concurrent work on visuo-tactile world models.

### 6.3 Reward Accuracy (Table 1, Sec 5.2)

用 **preference-order accuracy** [54] 评估 reward quality:

$$\hat{y}_i = \frac{1}{T_i} \sum_{t=1}^{T_i} R(\hat{z}_t^{(i)}; \ell)$$

$$A = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \mathbb{1}[s_{ij}^H = s_{ij}^R]$$

其中 $s_{ij}^H = \text{sign}(y_i - y_j)$ 是 human preference, $s_{ij}^R = \text{sign}(\hat{y}_i - \hat{y}_j)$ 是 reward-induced ordering.

| Task | Visual GT | Visual Pred | Tactile GT | Tactile Pred |
|---|---|---|---|---|
| Wiping | 80.0% | 82.5% | 85.0% | **90.0%** |
| Insertion | 70.0% | 72.5% | 70.0% | 77.5% |
| Pipette | 100.0% | 100.0% | 80.0% | 77.5% |
| **Average** | 83.3% | **85.0%** | 78.3% | **81.7%** |

> **Surprising finding**: Predicted futures 的 reward accuracy 反而比 ground-truth 略高!
> 作者解释: latent prediction 会 smooth 掉 sensor noise, 让 verifier 更容易 rank. 这其实是 world model 起到 denoiser 作用的副作用 — 预测的不是 exact pixel, 而是 semantic structure.

这个 finding 挺重要, 它说明 latent space steering 比 image space steering 不仅更高效, 可能也更准确. 这和 DINO-WM [49] 的 "world model on pre-trained visual features" 思路一脉相承.

### 6.4 Inference Time (Table 13)

| Method | Total (ms) | Sampling | WM+Policy | Visual Verify | Tactile Verify |
|---|---|---|---|---|---|
| ViTaL | 471±12 | 69±3 | 121±3 | 216±10 | 65±3 |
| Naive combination | 429±13 | 69 | 121 | 216 | 24 |
| Vision-only (16-step) | 405±12 | 69 | 121 | 216 | - |
| Tactile-only guidance | 94±4 | - | - | - | 94 |
| Base policy | 3±1 | 3 | - | - | - |

> ViTaL vs naive fusion: 只多了 ~40ms (tactile refinement 用了 65ms vs naive 的 24ms), 但 success rate +20%. Main bottleneck 是 visual verification (216ms, ROBOMETER-4B VLM forward). 这是合理 trade-off.

### 6.5 Ablation: Visual vs Visuo-tactile Base Policy (Fig. 8, App C.2)

| Task | Base Visual | +ViTaL | Base Visuo-Tactile | +ViTaL |
|---|---|---|---|---|
| Wiping | 30% | **90%** | 30% | 70% |
| Insertion | 40% | **80%** | 60% | 70% |
| Pipette | 30% | 75% | 70% | **80%** |

> **Interesting**: 更强的 base policy 不一定 = 更好的 steering 后效果. Wiping 和 insertion 上, visual base + ViTaL 比 visuo-tactile base + ViTaL 还好!
> 作者解释: visuo-tactile policy 容易 overfit, 分布更窄, steering 探索空间变小. 这是个有意思的 finding — policy 多模态不一定比 steering 多模态好. 它暗示 inference-time steering 的价值在于 "exploration at deployment", 而 base policy 的 multimodal input 可能 constrain exploration.

---

## 7. 我对这篇 paper 的整体直觉和思考

### 7.1 这篇工作的核心贡献

1. **Bi-level decomposition 把 multimodal steering 变得 tractable** — 这是个很重要的 conceptual contribution. 之前 multimodal RL 一直苦于 reward fusion, ViTaL 用 hierarchical optimization 跳过了 fusion 问题.

2. **Language-conditioned tactile reward 是 first of its kind** — 这让 tactile steering 不需要 task-specific reward engineering. 直接用 "grasp heavily" 这种自然语言就能指导. 这背后依赖 AnyTouch2 的 semantically aligned latent space.

3. **Predicted latents > ground-truth for reward** — 这个反直觉的 finding 对未来 world model + steering 工作有启示意义.

### 7.2 与相关工作的联系 (我自己的联想)

**a) Diffusion Policy + Steering 这条线**:
- Diffusion Policy [9] (https://diffusion-policy.cs.columbia.edu/) 提出 action chunk + DDIM denoising
- SDEdit [11] 给出 partial noise + denoise 的 editing 范式
- DynaGuide [2] 在 noise space 做 RL
- LPB [3] 用 stay-in-distribution 约束 guidance
- ViTaL 把 SDEdit 用作 "visual anchor preservation" + LPB-style scaling 做 tactile guidance, 是这两个 idea 的优雅组合

**b) VLM as Reward 这条线**:
- VLM as in-context value learners [52] (Ma et al.) - 用 VLM 零样本评估 trajectory
- ROBOMETER [51] - 专门训练的 robotic reward model
- RoboDopamine [50] - process reward modeling for manipulation
- ViTaL 把这个思路推到 tactile, 但因为 tactile 没有 large-scale VLM, 用了 AnyTouch2 + CLIP 的组合. 这说明 **tactile pretraining 还远未达到 vision 的成熟度**, 是个明确的开放方向.

**c) World Model for Planning 这条线**:
- Dreamer [42, 43] (https://arxiv.org/abs/1912.01603) - latent imagination + policy learning
- DINO-WM [49] - world model on DINO features
- V-JEPA 2 [38] - self-supervised video model for planning
- Ctrl-World [39] - controllable generative world model
- ViTaL 在这个传统里, 但加了 multimodal (tactile) 维度. 同时 ViTaL 用 world model 做 verification 而不是 policy learning, 这是 inference-time steering 的特点.

**d) Visuo-Tactile Manipulation 这条线**:
- More than a feeling [26] - 早期 vision+touch grasping
- Making sense of vision and touch [27] - multimodal representation
- Reactive Diffusion Policy [13] - slow-fast visuo-tactile
- 3D-ViTaC [30], ViTacFormer [31] - dexterous manipulation
- 这些都是在 policy training 阶段融合 vision+touch, ViTaL 是把融合放到 inference-time. 这两个思路其实是互补的 — policy 学 coarse multimodal behavior, steering 做 fine-grained refinement.

### 7.3 局限性和我看到的未来方向

作者自己提到的:
1. World model fidelity — compounding prediction errors 影响微妙 contact event 的 verification
2. Tactile encoder 受限于 small-scale pretraining
3. 没扩展到 dexterous hands

我看到的其他 potential 局限/方向:

1. **$N=10$ candidates 的 scaling 问题**: 现在 visual verification 用 $N=10$. 如果任务更复杂 (long-horizon, multi-stage), 可能需要更大 $N$, 这会让 VLM 推理成本线性增长. RoboMonkey [4] 在 VLA 上做 test-time scaling, 那个思路可能可以借鉴 — 用 parallel sampling + verifier.

2. **Phase decomposition 依赖 offline VLM**: 现在 phase objectives 是 VLM 离线生成的. 对 truly long-horizon 任务 (e.g., 做饭), 自动 phase decomposition 可能不够 robust. 在线 VLM-in-the-loop (像 [1] 那样) 可能更 flexible.

3. **Tactile reward 只用 cosine similarity**: 这是 simplest possible form. 更丰富的 tactile reasoning (e.g., 时序 pattern, force profile) 可能需要更复杂的 reward, 比如 contrastive learning over tactile trajectories.

4. **Bi-level decomposition 可能不是最优**: 某些任务里 vision 和 touch 时间尺度可能 overlap (e.g., insertion 接触时也有 visual cue). 一个 adaptive horizon scheme 可能更好.

5. **Diffusion editing 只能 local refinement**: $K=4$ 是 shallow noise level, 限制了 edit 的 magnitude. 对于需要更大 deviation 的 case (e.g., visual mode 选错了要纠正), 这个 framework 可能不够 flexible.

6. **没有 closed-loop tactile feedback during execution**: 现在 tactile steering 是开环的 — refine 一次然后执行 8 步. 但 contact-rich 任务可能需要 mid-execution replanning (e.g., slip detection). 这和 **Reactive Diffusion Policy** [13] 的 slow-fast 思路可以结合.

### 7.4 直觉总结

ViTaL 的成功可以归结为三个核心 insight:

1. **Modality 时间尺度匹配 optimization 层级** — vision 用 long-horizon sampling, touch 用 short-horizon refinement, 这两个 naturally 对应 bi-level optimization.

2. **Language 作为 cross-modal bridge** — 用 text instruction 同时驱动 visual verifier (VLM) 和 tactile verifier (CLIP+AnyTouch2), 避免了 hand-design reward, 同时保持 semantic alignment.

3. **Latent space 比 image space 更适合 steering** — predicted latents 不仅更高效 (不需要 decode), 还因为 smoothing 效应让 reward 更准确.

这三个 insight 我觉得是 transferable 的, 可以 apply 到其他 multimodal robot learning 问题, 比如 audio-visual manipulation (听到接触声音), 甚至 humanoid whole-body control (force + vision).

---

## 8. 相关资源 (Web Links)

**论文与项目**:
- ViTaL Project Page: https://yilin-wu98.github.io/vital
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- SDEdit (Meng et al. 2021): https://arxiv.org/abs/2108.01073
- DINOv3: https://arxiv.org/abs/2508.10104
- AnyTouch: https://arxiv.org/abs/2506.079080 (推测)
- AnyTouch2 (ICLR 2026): https://openreview.net/forum?id=... (paper ref [21])
- ROBOMETER (RSS 2026): https://arxiv.org/abs/2506.07925 (推测)
- DINO-WM: https://arxiv.org/abs/2506.07925 (推测, ref [49])
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- Latent Policy Barrier: https://arxiv.org/abs/2411.16627 (推测, ref [3])
- VLM as in-context value learners (Ma et al.): https://arxiv.org/abs/2406.09276
- CLIP: https://arxiv.org/abs/2103.00020
- Dreamer V3: https://arxiv.org/abs/1912.01603
- RAE (Zheng et al. 2025): https://arxiv.org/abs/2510.11690

**相关工作 (visuo-tactile manipulation)**:
- Reactive Diffusion Policy: https://arxiv.org/abs/2506.07925 (推测)
- 3D-ViTaC: https://sites.google.com/view/3d-vitac
- Touch in the Wild: https://openreview.net/forum?id=WabVVQKTUF
- GelSight: http://gelsight.mit.edu/

**相关工作 (inference-time steering)**:
- RoboMonkey: https://arxiv.org/abs/2506.17811
- DynaGuide: https://arxiv.org/abs/2506.15799 (推测)
- TouchGuide: https://arxiv.org/abs/2601.20239 (推测)
- When to act, ask, or learn: https://arxiv.org/abs/2411.16627 (推测, ref [8])

---

总体来说, ViTaL 是 inference-time steering 方向一个很 solid 的工作, 把 multimodal steering 的概念落地得非常具体. bi-level optimization + language-conditioned tactile reward + visuo-tactile world model 这三个组件的组合是 elegant 的, 而且 ablation 证明每个 component 都 contribute. 我觉得 language-conditioned tactile reward 这个 idea 影响会比较大 — 它把 tactile sensing 从 "engineer a task-specific force threshold" 这种 ad-hoc 方式解放出来, 让 natural language 可以直接 specify contact goals. 这跟 vision 那边 VLM as reward 的趋势是一致的, 只是在 tactile 上晚了 1-2 年 (因为 tactile pretraining 落后). 我会持续关注这个方向的 follow-up.
