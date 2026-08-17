---
source_pdf: Learning When to See and When to Feel Adaptive Vision-Torque.pdf
paper_sha256: aeca90ad9bc01b3b1a2607a9139513721696efce2a1f8c368a47604dcb8e8858
processed_at: '2026-08-05T14:16:23-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版

Andrej，用一句话总结这篇 paper：**在 free-space 时把 torque 通道关掉，在 contact 时让 torque expert 去修正 vision expert 的预测**。

---

## 问题是什么

做 manipulation 的 imitation learning，现在主流是 diffusion policy，主要靠 vision。但有一类任务 vision 根本看不出名堂——比如插拔 connector、拧盖子、靠重量区分两个一样的瓶子。视觉上"插进去了"和"没插进去"长得一模一样，只有 torque 突变能告诉你真相。

但 torque 自己也不干净。Franka 这种 robot 给的是 joint external torque（用 motor current 反推的），free-space 运动时会被 noise、inertia、gravity compensation 残差污染，pattern 完全是乱的（paper 的 Fig. 5 直接画出来给你看，free-space 阶段 torque 瞎跳）。

所以 naive 的 `concat(vision_feat, torque_feat)` 反而把 vision 也带崩了——Table I 里 feature concat 在 bottle 任务只有 3/20，比 vision-only 的 8/20 还差。这就是"加了反而更烂"的典型 case。

---

## 他们做了什么

两件事，叠起来用。

**第一件：Contact Gating。** 拿一个 binary flag φ——只要 7 个关节里任何一个 external torque 超过阈值，就 φ=1。然后 free-space（φ=0）时把 torque feature 换成一个 learnable 的 placeholder vector，contact（φ=1）时才用真实 torque encoder 输出。

这一步单独就能拿 68% success rate，是所有 baseline 里最强的。说明"free-space 时别让 torque 捣乱"这件事的价值远大于"contact 时怎么用 torque"。

**第二件：CFG-style Fusion。** 训两个 U-Net，一个只看 vision，一个只看 torque。最终 noise prediction 是：

```
ε_final = ε_vision + w_torque * (ε_torque - ε_vision)
```

这就是 classifier-free guidance 的形式。几何上，`(ε_torque - ε_vision)` 是 torque expert 相对 vision expert 的"修正方向"，`w_torque` 控制修正强度。

关键 trick：`w_torque` 被 contact flag 硬切——free-space 时 = 0（退化成纯 vision policy），contact 时 = softplus(learned_scale)（可以放大到任意倍）。Scale 由一个小 MLP 根据当前 image + torque feature 自适应预测。

---

## 为什么这套设计 work

对比一下其他方法就清楚了。

**Auxiliary Goals（TA-VLA 思路）**：让 U-Net 额外预测 future torque 作为 regularizer。结果 28%，比 vision-only 还差。原因：joint external torque 在 free-space 是 patternless 的，预测它等于让模型学 noise，regularizer 反而添乱。

**MoE**：两个 expert + router network，softmax 出权重做 convex blend。结果 24%，更差。Ablation 里给 MoE 也加上 contact-gated torque feature，涨到 54%，但还是输给 Ours 的 80%。

核心差别在 Fig. 7：MoE 的 router 在 contact 和 free-space 时输出的权重几乎一样（0.77/0.23 vs 0.77/0.23），完全学不出 contact-aware routing。而 Ours 的 `w_torque` 在 free-space 严格为 0，contact 时明显跳起来。

为什么 MoE 学不出来？我猜两个原因：
- Softmax 的输出被约束在 [0,1] 且和为 1，contact 时即使想"让 torque dominate"也只能给 0.6/0.4 这种温吞的权重。
- Router 的梯度在两个 logit 接近时很 flat，学不出 decisive switching。

CFG 的设计正好绕开这两点：硬切（φ=0 直接归零）+ unbounded（softplus 无上界）。**Bimodal regime 就该用 hard switching，不该用 soft interpolation。**

---

## 一个我特别喜欢的细节

Table III 的 single-shot success rate：Torque Gating 20%，Ours 60%。

Torque Gating 虽然 average success rate 68%，但经常要"试两次"——第一次 grasp pose 不准撞一下，第二次调整才成功。Ours 在 contact phase 的精度高得多，一次就成。这意味着 CFG 的 adaptive scaling 不光提升"最终成功率"，更提升"每次操作的效率"——这对 real-world deployment 意义很大，因为重试是有时间成本和碰撞风险的。

---

## 一句话 takeaway

Multimodal fusion 的关键不在"怎么加"，在"什么时候加、加多少"。Contact state 是个天然的开关信号，应该被显式 hard-encoded 进 fusion mechanism，而不是指望 network 自己从 data 里 soft-learn 出来。这个原则应该能推广到 tactile、audio 这些 modality 的融合上。

如果你想 build 更深的 intuition，强烈建议看 paper 的 Fig. 5 和 Fig. 7——前者是 free-space torque 瞎跳的视觉证据，后者是 CFG vs MoE 的 routing weight 对比，这两张图比所有公式都更能说明问题。

---

# Learning When to See and When to Feel 深度讲解

Andrej 你好，这篇 paper 来自 Rutgers / Penn State / Brown 的合作团队，第一作者 Jiuzhou Lei，发表在 2026 年 ICRA 周期（arXiv 时间戳显示是 2512.xxx 系列）。它属于 imitation learning + diffusion policy + multimodal fusion 的脉络，跟 FACTR、FoAR、ForceVLA、TA-VLA 是同一波潮流。核心卖点是做一个 contact-aware 的 vision-torque 融合，并且做了横向 benchmark。

参考链接：
- Diffusion Policy 官方: https://diffusion-policy.cs.columbia.edu/
- Classifier-Free Guidance: https://arxiv.org/abs/2207.12598
- FACTR: https://arxiv.org/abs/2502.17432
- FoAR: https://arxiv.org/abs/2501.14990
- Robomimic: https://robomimic.github.io/
- Franka Research 3: https://www.franka.de/
- ResNet 原文: https://arxiv.org/abs/1512.03385
- U-Net 原文: https://arxiv.org/abs/1505.04597

---

## 1. 这个问题为什么值得做：直觉层面的动机

纯粹用 vision 做 manipulation policy（diffusion policy、Octo、π0、GR00t N1 这些）在 free-space pick-and-place 这种任务上效果已经很好。但有一类 contact-rich 任务，视觉信号本身的"信息增益"非常低：

- **Connector plug-in / pull-out**：插拔前后的视觉几乎不变，但内部的 contact state 完全不同。
- **Twist-lock rotation**：旋转到机械限位的瞬间，视觉上跟"还没到限位"长得一模一样，只有 torque 飙升能告诉你"该停止旋转改为拉拔了"。
- **Weight discrimination**：两个外观完全一样的瓶子，必须靠 lifting 时的 joint torque 才能区分轻重。
- **Self-occlusion**：gripper 包住 workpiece 后，wrist camera 看不到关键信息，只剩 F/T 还能持续提供 feedback。

这就是 paper 的核心 motivation：**vision 在 contact phase 会"瞎"掉，必须靠 F/T 把 policy 唤醒**。

但是 F/T 自己也有坑——它**不是干净信号**。Franka 这种 robot 给的是 joint external torque（用 motor current + 动力学模型反推的外部扰动 torque），而不是 wrist-mounted F/T sensor。在 free-space motion 时，这部分读数会被：
- Sensor noise（电流测量噪声、摩擦建模误差）
- Inertial effects（加速度本身会让 external torque 估计跳变）
- Gravity compensation 残差

污染掉。所以一个 naive 的"vision feature ⊕ torque feature concat"会反过来让 policy 在 free-space 阶段也被 noise 干扰，论文 Table I 里 Feature Concatenation 在 bottle 任务上只有 3/20，比 vision-only 的 8/20 还差，就是这个原因。这就是 **modality collapse 的反面**：不是 policy 忽略 torque，是 noise 把 vision 也带崩了。

---

## 2. 三类既有 fusion 策略的 taxonomy

作者把已有工作归成三类，这是这篇 paper 的 literature 组织方式：

### (a) Auxiliary prediction methods
代表作：TA-VLA、ImplicitRDP、Lee et al. ICRA 2019。在主任务（predict action noise）之外，加一个"predict future torque"的 auxiliary head，作为 representation 的 regularizer。

### (b) Mixture-of-Experts / routing
代表作：Chen et al. 2025、ForceVLA。每个 modality 训一个 expert，用一个 router network 输出 softmax 权重做 element-wise blend。

### (c) Gating mechanisms
代表作：FoAR。用一个 future contact predictor 来 modulate force feature 的贡献。

paper 的核心 contribution 之一就是：**这三类方法过去只跟 vision-only 或 naive concat 比较，从来没在同一个 benchmark 上互相比过**。这篇就是来填这个坑的。

---

## 3. 方法详解：Contact Gating + CFG-style Fusion

整个架构 build 在 diffusion policy 之上。Conditioning 包含：
- 2 路 RGB（agent-view + wrist）：224×224×3，各自过 ResNet-18
- Joint torque history：7 维 × 10 步（7 个关节，过去 10 个 observation step），过 MLP 输出 64-d
- Joint position：7 维，过 MLP 输出 64-d

为什么用 10 步 window？因为 task horizon 大概是这个量级，10 步足够感知 torque 的 magnitude 和 rate-of-change。这跟 Diffusion Policy 默认的 obs horizon=2、pred horizon=16 是一致的设定。

### 3.1 Contact Gating

这部分直接借鉴 FoAR 的 gating。设 φ ∈ {0, 1} 是 contact flag——只要 7 个关节里**任何一个** external torque 超过预设阈值，就 φ=1。最终的 force feature：

$$
f_{\text{torque-gated}} = \phi \cdot f_{\text{torque}} + (1-\phi) \cdot f^{*}
$$

变量解释：
- $f_{\text{torque}}$：MLP torque encoder 的输出（64-d vector），代表"当前真实感知到的 torque 状态"
- $f^{*}$：一个 learnable parameter（不是 encoder 输出，是直接 trainable 的 64-d vector），代表"自由空间下 torque feature 的占位符"
- $\phi$：binary contact indicator

**Intuition**：$f^{*}$ 相当于给 network 一个"free-space 时 torque channel 该长什么样"的稳定 anchor。如果直接把 encoder 输出 zero-mask 掉，network 会学到"哦，只要这一路是 0 就是 free space"——但这种 hard zero 会跟"encoder 真的输出接近 0 但有 noise"的情况混淆。Trainable anchor 让 free-space 的 representation 是一个 learnable、跟训练分布对齐的 vector，更稳定。

这一步**单独**就能拿到 68% 的 average success rate（Table I 的 Torque Gating 行），是所有 baseline 里最强的。说明 free-space 的 torque 抑制是关键。

### 3.2 CFG-Style Adaptive Vision-Torque Fusion

这是 paper 的真正创新点。不是训一个 monolithic denoiser，而是训**两个 modality-specialized U-Net**：

- $\hat{\epsilon}_{\text{vision}}$：condition 在 image features + proprioception
- $\hat{\epsilon}_{\text{torque}}$：condition 在 gated torque features + proprioception

然后用 classifier-free guidance 的形式融合：

$$
\hat{\epsilon}_{\text{final}} = \hat{\epsilon}_{\text{vision}} + w_{\text{torque}} \left( \hat{\epsilon}_{\text{torque}} - \hat{\epsilon}_{\text{vision}} \right)
$$

变量含义：
- $\hat{\epsilon}_{\text{vision}}$：vision-specialized U-Net 在当前 noisy action $a_t$ 和 vision conditioning 下预测的 noise
- $\hat{\epsilon}_{\text{torque}}$：torque-specialized U-Net 在同一个 noisy action 和 torque conditioning 下预测的 noise
- $w_{\text{torque}}$：guidance scale，决定 torque expert 相对于 vision expert 的"修正强度"
- $\hat{\epsilon}_{\text{final}}$：最终送进 DDPM denoising update 的 noise estimate

这个公式直接对应 classifier-free guidance 的 $\epsilon_{\theta}(x_t, c) = (1+w)\epsilon_{\theta}(x_t, c) - w\cdot\epsilon_{\theta}(x_t, \emptyset)$，只是这里把"conditional vs unconditional"换成了"vision-expert vs torque-expert"。**几何上**，$\hat{\epsilon}_{\text{torque}} - \hat{\epsilon}_{\text{vision}}$ 是一个"修正向量"，它指向"torque expert 认为该走的方向相对于 vision expert 的偏移"。

**直觉**：vision expert 给一个 base 方向，torque expert 给一个偏置。当 contact 发生时，torque 信号 informative，偏置应该被采纳；free-space 时，偏置应该归零。这个 $w_{\text{torque}}$ 就是控制偏置采纳强度的旋钮。

而 $w_{\text{torque}}$ 本身被 contact gating 调制：

$$
w_{\text{torque}} = \phi \cdot \sigma(w_{\text{scale}})
$$

- $w_{\text{scale}}$：scale predictor（3-layer MLP）的 raw output，输入是 gated torque feature + image feature
- $\sigma(\cdot)$：softplus 函数 $\sigma(x) = \log(1+e^x)$，保证非负
- $\phi$：contact flag，跟前面一样

注意 $\sigma$ 用 softplus 而不是 sigmoid，意味着 $w_{\text{torque}} \in [0, +\infty)$——理论上允许 torque expert 完全主导甚至"过度修正"。这跟 CFG 在 image generation 里 $w$ 可以 >1 的超参设定是一致的。

**两个 regime 的行为**：
- Free-space（$\phi=0$）：$w_{\text{torque}}=0$，$\hat{\epsilon}_{\text{final}} = \hat{\epsilon}_{\text{vision}}$，policy 完全退化成 vision-only diffusion policy，干净。
- Contact（$\phi=1$）：$w_{\text{torque}} = \sigma(w_{\text{scale}})$，scale predictor 根据当前 image + torque 决定融合强度，是 learned adaptive behavior。

### 3.3 训练目标

$$
\mathcal{L} = \text{MSE}(\hat{\epsilon}_{\text{final}}, \epsilon)
$$

- $\epsilon$：从 Gaussian 采样的 ground-truth noise
- $\hat{\epsilon}_{\text{final}}$：上面 CFG 公式得到的 final prediction

两个 U-Net 和 scale predictor **jointly optimized**。这里有个微妙的点：因为 $w_{\text{torque}}$ 是 learned 且 contact-dependent，梯度会同时回传给两个 U-Net 和 scale predictor，强迫它们**协同**学到"vision expert 主导 free-space，torque expert 提供 contact-phase 修正"的分工。

### 3.4 Baseline 实现细节

**Auxiliary Goals**（对应 TA-VLA 思路）：

$$
\mathcal{L} = \text{MSE}(\hat{\epsilon}_{\text{action}}, \epsilon_{\text{action}}) + \alpha \text{MSE}(\hat{\epsilon}_{\text{torque}}, \epsilon_{\text{torque}})
$$

- $\epsilon_{\text{action}}$：action 维度对应的 noise slice
- $\epsilon_{\text{torque}}$：future torque 维度对应的 noise slice
- $\alpha$：loss 权重

Inference 时只取 action slice，torque 预测丢弃。本质是 representation regularizer。

**MoE**（对应 Chen et al. 2025）：

$$
\hat{\epsilon} = w_{\text{vision}} \cdot \hat{\epsilon}_{\text{vision}} + w_{\text{tor}} \cdot \hat{\epsilon}_{\text{torque}}
$$

- $w_{\text{vision}}, w_{\text{tor}}$：router MLP 输出的 softmax 权重，$w_{\text{vision}} + w_{\text{tor}} = 1$

跟 CFG 的核心区别：MoE 是 **convex combination**（权重和为 1，只能在两个 expert 之间插值），CFG 是 **affine combination**（允许外推，$w$ 可以 >1）。更重要的区别在 Section 4 的 ablation 里揭示。

---

## 4. 实验：三个 task 的设计哲学

作者精心选了三个 task，每个考察不同的 force reasoning 能力：

| Task | Demos | Force reasoning 需求 |
|------|-------|---------------------|
| Egg Boiler Lid Opening | 150 | 检测 resistance 变化，避免提前 lift |
| Weight-Based Bottle Placement | 110 | 用 lifting torque 区分 empty/full |
| Twisty Connector Pull Out | 250 | 检测 rotation limit，切换到 pull 阶段 |

这三个 task 覆盖了"transition detection"、"classification"、"phase switching"三类典型 force-aware 场景，设计得相当有代表性。

### 4.1 主结果（Table I）

| Method | Bottle | Connector | Lid | Avg |
|--------|--------|-----------|-----|-----|
| Vision-only | 8/20 | 0/10 | 7/20 | 30.0% |
| Feature Concatenation | 3/20 | 0/10 | 12/20 | 30.0% |
| Torque Gating | 14/20 | 5/10 | 15/20 | 68.0% |
| Auxiliary Goals | 6/20 | 1/10 | 7/20 | 28.0% |
| MoE | 5/20 | 1/10 | 7/20 | 24.0% |
| MoE w/o torque encoding | 15/20 | 1/10 | 11/20 | 54.0% |
| **Ours** | **16/20** | **7/10** | **18/20** | **82.0%** |

几个关键观察：

1. **Vision-only 和 Feature Concatenation 持平**（都 30%）：说明 naive concat 不仅没帮助，反而被 noise 拖累。
2. **Auxiliary Goals 和 MoE 都比 vision-only 差**（28% / 24%）：这两个"看起来更高级"的方法在这个 setting 下反而退步，因为 joint external torque 在 free-space 时 patternless，auxiliary prediction 学不到有用信号；MoE 的 router 学不出 contact-aware 的 routing。
3. **MoE w/o torque encoding（54%）比 MoE（24%）高很多**：这个反直觉结果非常 informative——当 torque encoder 学出来的 feature 是 noise 时，干脆把 raw torque 喂给 router 反而更好。这暗示 "torque encoder 学坏了" 比 "不 encode" 还糟。
4. **Torque Gating（68%）是 strongest baseline**：印证了 free-space filtering 的核心价值。
5. **Ours（82%）比 Torque Gating 高 14%**：CFG-style adaptive scaling 在 contact phase 的精度提升贡献了这 14%。

### 4.2 Bottle 任务的特殊性（Table II）

Vision-only 在 bottle 任务上 8/20 看起来还行，但 Table II 拆开看：empty 8/10、full 0/10。**Policy 学到了一个 bias**：永远放红盘（empty 的目标），因为训练数据虽然平衡，但 vision 无法区分两种瓶子，policy 退化成"默认走红盘"。

这正是 force-aware 任务里 vision-only 的典型 failure mode——表面 success rate 看起来合理，实际是 exploitable bias。

### 4.3 Single-shot success（Table III）

| Method | Single-shot Success | Avg Horizon |
|--------|---------------------|------------|
| Torque Gating | 20.0% | 110.7 |
| Ours | 60.0% | 87.1 |

Torque Gating 经常需要"试两次"才成功——第一次 grasp pose 不准，撞一下重新调整。Ours 在 contact phase 的精度更高，第一次就成的概率翻了 3 倍，平均 horizon 也短 23 步。这是 CFG adaptive scaling 的直接 payoff。

### 4.4 Ablation: CFG vs MoE（Table IV + Fig. 7）

这是 paper 最 insightful 的分析。把 MoE 的 router 也接上 contact-gated torque feature（消除 gating 这个变量），单纯比较 fusion 公式：

| Method | Bottle Success |
|--------|----------------|
| Torque-Gated MoE | 12/20 |
| Ours (CFG) | 16/20 |

差 4/20。原因在 Fig. 7 的 routing weight 分析：

**MoE 的 router 学不出 contact-aware routing**：
- Free-space: $w_{\text{vision}} = 0.7634$, $w_{\text{torque}} = 0.2366$
- Contact: $w_{\text{vision}} = 0.7714$, $w_{\text{torque}} = 0.2286$

两者**几乎一样**！router 把 contact 状态"抹平"了，给一个固定权重。

**Ours 的 $w_{\text{torque}}$**：
- Free-space: $w_{\text{torque}} = 0$（因为 $\phi=0$ 硬切）
- Contact: $w_{\text{torque}}$ 动态变化，明显非零

为什么 MoE 学不出 contact-aware routing？我个人的理解（paper 没明说但符合直觉）：MoE 的 router 输出要过 softmax，且权重和为 1。即使 contact 时 torque feature 有信号，router 也只能稍微调高 $w_{\text{tor}}$ 一点点，因为 softmax 的梯度在两个 logit 接近时很 flat。而 CFG 的 $w_{\text{torque}}$ 是 softplus 输出，无上界，contact 时可以飙到很大，free-space 时被 $\phi$ 硬切到 0。**Hard switching + unbounded amplification** 比 **soft softmax routing** 更适合这种 bimodal（contact / no-contact）的 regime。

这也呼应了 control theory 里 hybrid system 的设计哲学：在 mode 切换时，硬切换比软插值更可靠。

---

## 5. 跟相关工作的联想与延伸

### 5.1 跟 classifier-free guidance 的关系
CFG 在 image generation 里是解决 "conditional generation 信号太弱" 的问题，通过 $(1+w)\epsilon_c - w\epsilon_{\emptyset}$ 放大 conditioning 的影响。这里作者做了一个很优雅的迁移：把 "unconditional" 替换成 "vision-only expert"，把 "conditional" 替换成 "torque-conditioned expert"。**$\hat{\epsilon}_{\text{torque}} - \hat{\epsilon}_{\text{vision}}$ 就是 torque modality 带来的"额外信息方向"**。

但跟标准 CFG 的关键区别是：标准 CFG 的 $w$ 是手动设定的 hyperparameter，这里是 **learned + contact-gated**。这其实更像 Dynamic Thresholding 或者 Guidance Distillation 的思路——让模型自己决定 guidance strength。

参考：https://arxiv.org/abs/2207.12598

### 5.2 跟 Hybrid Force-Position Control 的关系
Classic robotics 里 hybrid force-position control（Raibert-Craig formulation）是把 task space 分成 position-controlled subspace 和 force-controlled subspace，正交分解。这篇 paper 的 contact-gating 是一种**时间维度上的 hybrid**：free-space 时纯 position（vision-driven），contact 时混合 force/torque（vision + torque blended）。这是 learning-based 对 classic hybrid control 的 re-instantiation。

参考 Raibert-Craig: https://ieeexplore.ieee.org/document/6316257

### 5.3 跟 Adaptive Compliance Policy 的关系
Hou et al. 2025（Adaptive Compliance Policy）学一个 stiffness scalar + virtual target pose，本质是在 action space 之外预测 controller 参数。这篇 paper 走的是另一条路：在 noise-prediction space 做融合，不显式预测 stiffness。两条路殊途同归——都试图让 policy 在 contact phase "变软" / "变 force-aware"。

参考：https://diffusion-policy.cs.columbia.edu/data/adaptive_compliance_policy.pdf

### 5.4 跟 Tactile-based methods 的关系
这篇用 joint external torque，**没有用 tactile sensor**（GelSight、DIGT 等）。Joint torque 的空间分辨率低（只有 7 维），但 hardware cost 几乎为零（Franka 自带）。跟 Qi et al. 2023（General In-Hand Object Rotation with Vision and Touch, https://arxiv.org/abs/2210.09486）相比，这篇的 torque 信号更"粗"，但通过 contact-gating + CFG 放大，依然能拿到 82% success rate。这暗示了一个有意思的方向：**粗粒度 F/T + 精细 fusion 策略** 可能比 **细粒度 tactile + naive fusion** 更 cost-effective。

### 5.5 跟 ForceVLA / TA-VLA 这些 VLA 方向的关系
ForceVLA（https://arxiv.org/abs/2505.22159）和 TA-VLA（https://arxiv.org/abs/2509.07962）都是在 VLA 框架里加 force，但它们 scale 大得多（VLM backbone + MoE），数据需求也大。这篇 paper 的 setting 更"实验室化"——小数据（110-250 demos）、小模型（ResNet-18 + U-Net）、专门 task。它的价值在于**提供了一个干净的 ablation playground**，验证了 fusion strategy 的设计原则，这些原则理论上可以迁移到 VLA scale。

### 5.6 跟 Reactive Diffusion Policy / ImplicitRDP 的 slow-fast 思路
ImplicitRDP（https://arxiv.org/abs/2512.10946）和 Reactive Diffusion Policy（https://arxiv.org/abs/2503.02881）都用 slow-fast 双分支结构——slow 负责视觉规划，fast 负责触觉反应。Ours 的 vision-specialized U-Net + torque-specialized U-Net 在结构上有相似性，但融合方式不同：slow-fast 通常在 action level 融合（fast 修正 slow 的 action），Ours 在 noise-prediction level 融合（CFG-style）。两者的 trade-off 还没有充分比较，是个 open direction。

### 5.7 跟 Multi-modal Policy Consensus（Chen et al. 2025）的关系
这篇的 MoE baseline 就是简化版的 Chen et al. 2025（https://arxiv.org/abs/2509.23468）。原版用多个 sub-policy 做 consensus，这篇简化成两个 expert + 一个 router。简化版都跑出 24%，说明原版的复杂 routing 在 contact-rich setting 下可能也不 work——除非 router 设计得更 contact-aware。

---

## 6. 局限性与 Open Questions

作者自己承认：
1. **只覆盖 representative subset**：没比较 FACTR 的 visual corruption curriculum、没比较 Adaptive Compliance Policy 的 stiffness prediction。
2. **Passive force sensing**：policy 被动感知 F/T，没有主动施力（比如 surface finishing 需要持续施压）。
3. **没探索 active interaction**：precision assembly / disassembly 这种需要闭环 force control 的任务没测。

我补充几个**潜在问题**：

**a. Contact threshold 的 sensitivity**。$\phi$ 是 binary，依赖预设的 torque 阈值。如果阈值设错（比如不同物体 contact 时 torque 幅度差异大），$\phi$ 会误判。一个更 robust 的设计是 learned soft gating（比如 sigmoid(threshold) 替代 hard threshold），但作者没试。

**b. Two U-Net 的训练成本翻倍**。Vision-specialized + torque-specialized 两个 U-Net，参数量和训练时间翻倍。是否能用 shared backbone + modality-specific head 降低成本？类似 LoRA-style 的 modality adapter？

**c. Contact phase 内的 sub-regime 没区分**。Contact 也有"刚接触"、"稳定接触"、"脱离接触"几个 sub-phase，paper 的 $\phi$ 只看"有没有 contact"。更细粒度的 contact state（比如 normal force vs friction force）可能进一步提升精度。

**d. Generalization 到新物体**。三个 task 都是固定物体，没测新物体的 zero-shot。FACTR 报告了 40% improvement on unseen objects，这篇没做这个 comparison，是 gap。

**e. $\hat{\epsilon}_{\text{torque}} - \hat{\epsilon}_{\text{vision}}$ 的几何含义没可视化**。作者给了 $w_{\text{torque}}$ 的时序曲线（Fig. 7），但没给"修正向量"本身的可视化。如果能把 $\hat{\epsilon}_{\text{torque}} - \hat{\epsilon}_{\text{vision}}$ 在 action space 投影出来看它指向哪里，会极大增强 intuition。比如在 twisty connector 任务里，这个差值是否真的指向"停止旋转开始拉拔"的方向？

**f. 跟 RL-based force control 的比较缺失**。经典 RL 方法（比如 SAC + force reward）在 contact-rich 任务上也有 work，这篇没纳入比较。虽然 focus 是 imitation learning，但读者会想知道 IL-based fusion 跟 RL-based force control 的 trade-off。

---

## 7. 给你的 Take-away

如果让我提炼这篇 paper 对 manipulation learning 社区的贡献：

1. **Empirical finding**：在 joint external torque 这种 noisy F/T 信号下，free-space filtering 是第一性的。不做 gating 的方法（auxiliary goal、naive MoE）都会被 noise 拖累，甚至比 vision-only 还差。
2. **Methodological contribution**：CFG-style fusion 在 noise-prediction level 做 modality blend，比 MoE 的 convex combination 更适合 bimodal regime——因为 hard switching + unbounded amplification 能表达"contact 时 torque 应该 dominate"这种 non-convex 的融合。
3. **Benchmarking value**：第一次在同一 framework 下比较 auxiliary goal / MoE / gating / CFG 四类策略，给出明确的 ranking。

对未来的启发：**multimodal fusion 不该是"无脑 concat"或"无脑 attention"，而该是 contact-state-conditioned 的 dynamic routing**。这个原则应该可以迁移到 tactile、audio、proprioception 等其他 modality 的融合上。

如果你想 build 更深的 intuition，建议看 Fig. 5 的 torque 时间序列——它直观展示了 free-space 阶段 torque 的"patternless 抖动"，这是整个 paper motivation 的视觉证据。然后看 Fig. 7 的 $w_{\text{torque}}$ 曲线对比，这是 CFG vs MoE 区别的核心可视化。

参考链接汇总：
- Paper arXiv（推测）: https://arxiv.org/abs/2512.xxxxx
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Classifier-Free Guidance: https://arxiv.org/abs/2207.12598
- FACTR: https://arxiv.org/abs/2502.17432
- FoAR: https://arxiv.org/abs/2501.14990
- Robomimic: https://robomimic.github.io/
- Franka FR3: https://www.franka.de/product/franka-research-3/
- Adaptive Compliance Policy: https://diffusion-policy.cs.columbia.edu/data/adaptive_compliance_policy.pdf
- General In-Hand Rotation (Qi et al.): https://arxiv.org/abs/2210.09486
- ForceVLA: https://arxiv.org/abs/2505.22159
- TA-VLA: https://arxiv.org/abs/2509.07962
- ImplicitRDP: https://arxiv.org/abs/2512.10946
- Reactive Diffusion Policy: https://arxiv.org/abs/2503.02881
- Multi-modal Policy Consensus: https://arxiv.org/abs/2509.23468
