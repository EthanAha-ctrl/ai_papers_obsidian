---
source_pdf: FLOW STRAIGHT AND FAST LEARNING TO GENERATE AND TRANSFER DATA WITH RECTIFIED
  FLOW.pdf
paper_sha256: 03840fbd74dbed55732bc65ec2b478cf73573b10bd2aef3bdf8d59017a828f48
processed_at: '2026-08-18T13:25:37-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Rectified Flow 的人话版本

## 一句话

两点之间直线最短, 所以你把每一个 noise 点 X₀ 和随便配的一个 cat image X₁ 用直线连起来, 训一个神经网络去学"在这条直线上每一点该往哪个方向走", 这个网络就是一个 ODE, 一次 Euler 步就能生成图像。问题是直线之间会交叉, 所以第一次训完轨迹还是弯的, 但你把生成的样本拿来再训一次, 再训一次, 轨迹就越来越直 (理论上是 O(1/k) 速率变直)。这就是整篇 paper。

下面拆开讲。

---

## 1. 你想干啥

你手上有两个分布:
- π₀ = 标准 Gaussian 噪声 𝒩(0, I) 
- π₁ = cat images (CIFAR10, AFHQ Cat, etc)

你想要一个映射 T, 把 X₀ ~ π₀ 变成 X₁ ~ π₁。

GAN 用 minimax 学这个 T, DDPM 用 SDE + score matching 学, normalizing flow 用 invertible 架构 + MLE 学。Rectified Flow 说: 都别整这些花活, 就用最朴素的 supervised regression。

---

## 2. 先画一堆直线

你随机抽一对 (X₀, X₁), X₀ 是噪声, X₁ 是 cat。把它们之间连一条直线:

$$ X_t = t \cdot X_1 + (1-t) \cdot X_0, \quad t \in [0, 1] $$

变量解释:
- X_t 是中间点 (一张 image-shaped 的向量, 比如 3×32×32)
- t 是时间, 0 到 1 之间
- t=0 时 X_t = X₀ (噪声), t=1 时 X_t = X₁ (cat)
- 系数 t 和 (1-t) 就是线性插值权重

这条直线本身满足一个 ODE:

$$ \frac{dX_t}{dt} = X_1 - X_0 $$

速度 (X₁ - X₀) 是个常向量, 从 X₀ 指向 X₁。

但这个 ODE 没法用来生成图像! 因为想算 dX_t/dt 你必须知道终点 X₁ — 生成的时候你哪知道 X₁ 是啥, 你就是要生成它。这叫 non-causal / anticipating。

---

## 3. "因果化": 用回归把直线改造成可模拟的 ODE

idea 很简单: 训一个神经网络 v(z, t), 输入是当前位置 z 和时刻 t, 输出是该往哪走。target 就是 (X₁ - X₀)。

loss (Eq. 1):

$$ \min_v \int_0^1 \mathbb{E}\left[\left\| (X_1 - X_0) - v(X_t, t) \right\|^2 \right] dt $$

变量:
- v 是要学的 velocity field (一个 U-Net, 比如 DDPM++ 架构)
- X_t = t·X₁ + (1-t)·X₀ 是网络输入
- (X₁ - X₀) 是 regression target
- ‖·‖² 是 L2 范数平方
- 𝔼 是对 (X₀, X₁) 这个随机配对取期望
- 积分 ∫₀¹ dt 在实现里就是 t ~ Uniform[0,1] 采样

训练伪代码 (Algorithm 2):

```python
for x0, x1 in dataloader:      # x0 ~ π₀ (noise), x1 ~ π₁ (cat)
    t = torch.rand(B)            # t ~ U[0,1]
    x_t = t * x1 + (1 - t) * x0  # 直线上某点
    target = x1 - x0             # 直线方向
    pred = model(x_t, t)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    optimizer.step()
```

就这样。没有 minimax, 没有 ELBO, 没有 KL divergence, 没有 score function 估计, 没有 Langevin dynamics, 没有 reverse-time SDE。就是 standard supervised learning, scale 到 billion parameters 没障碍。

闭式解 (Eq. 2):

$$ v^X(x, t) = \mathbb{E}[X_1 - X_0 \mid X_t = x] $$

意思: 在位置 x、时刻 t 处, 所有穿过这点的直线方向 (X₁ - X₀) 的**条件期望**。如果有多条直线穿过 x, 你取平均方向。这就解决了"不知道往哪走"的问题 — 多条路交叉时, 走平均方向, 这样得到的轨迹不会交叉。

---

## 4. 为什么"不交叉"这么关键

ODE dZ_t = v(Z_t, t) dt 的一个数学事实: 如果 v 足够光滑, ODE 解唯一。**唯一性 → 轨迹不交叉**。因为如果两条轨迹在 (z, t) 处相遇, 从那个时刻起它们满足同样的 ODE、同样的初值, 必须重合, 矛盾。

paper Figure 2 给了个超棒的图:
- (a) linear interpolation 的所有直线, 互相交叉
- (b) 训完 v 之后的 ODE 轨迹, 在交叉点被"重排"成不交叉的网络
- (c)(d) 把不交叉的轨迹端点再连直线, 已经不交叉了

直觉: linear interpolation 是建了一堆"公路", 公路会交叉。Rectified flow 是"在这公路网上开车", 车流在交叉路口不真的过叉, 而是平均化方向走过去, 自然就把交叉消除了。

---

## 5. 三大理论保证 (Appendix D)

### 5.1 Marginal 不变 (Theorem D.3)

对任意时刻 t:

$$ \text{Law}(Z_t) = \text{Law}(X_t) $$

Z_t 是 ODE 解, X_t 是 linear interpolation。它们的边缘分布处处相同。意思是: 你并没有改变"任何时刻看到的图像分布", 你只是改了"哪个噪声配哪只猫"的对应关系。

证明 idea (这是 paper Appendix D.1 的核心): 对任意 test function h, 用 chain rule:

$$ \frac{d}{dt}\mathbb{E}[h(X_t)] = \mathbb{E}[\nabla h(X_t)^\top \dot X_t] = \mathbb{E}[\nabla h(X_t)^\top v^X(X_t, t)] $$

最后一步用 tower property: 𝔼[Ẋ_t | X_t] = v^X(X_t, t)。这就是 continuity equation ∂_t ρ_t + ∇·(v^X ρ_t) = 0 的弱形式。Z_t 满足同样的 PDE, 同样的初值, 所以 marginal 相同。

### 5.2 Transport cost 单调下降 (Theorem D.5)

对任意凸函数 c (比如 c(x) = ‖x‖² 或 ‖x‖):

$$ \mathbb{E}[c(Z_1 - Z_0)] \le \mathbb{E}[c(X_1 - X_0)] $$

证明用两次 Jensen:
1. c 是凸的, 所以 c(∫v dt) ≤ ∫c(v) dt
2. c 是凸的 + 条件期望, 所以 c(𝔼[Y|X]) ≤ 𝔼[c(Y)|X]

含义: rectified flow 对**所有凸 cost** 同时下降, 不针对特定 c 优化。这区别于 OT, OT 是针对一个 c 求最优。RF 是"对所有凸 cost 做 Pareto descent"。

直观 (c = ‖·‖, 即路径长度): linear interpolation 的总长度 = ∑‖X₁ - X₀‖, 有交叉意味着三角不等式严格成立, rewire 后的 Z 网络总长度更短。

### 5.3 Reflow 让 flow 越来越直 (Theorem D.7)

定义 straightness (Eq. 3):

$$ S(Z) = \int_0^1 \mathbb{E}\left[\left\| (Z_1 - Z_0) - \dot Z_t \right\|^2 \right] dt $$

- Z₁ - Z₀ 是从起点到终点的常向量
- Ż_t = v(Z_t, t) 是 ODE 在时刻 t 的实际速度
- S(Z) = 0 当且仅当 Ż_t = Z₁ - Z₀ 恒定 → 轨迹是匀速直线

Reflow: Z^(k+1) = RectFlow((Z₀^k, Z₁^k)), 起始 (Z₀^0, Z₁^0) = (X₀, X₁)

收敛速率:

$$ \min_{k \in \{0, \dots, K\}} S(Z^k) \le \frac{\mathbb{E}[\|X_1 - X_0\|^2]}{K} $$

O(1/K) 速率。证明是 telescoping sum, 每次 rectify 把 transport cost 下降的量分配给 S 和 V (交叉度)。

为什么 straight 这么重要: 完美直线的 ODE, 一次 Euler 步就精确求解:
$$ Z_1 = Z_0 + v(Z_0, 0) \cdot 1 $$
这就是 one-step generation! GAN 一步生成, RF 也能一步生成, 还不用 minimax。

---

## 6. 跟 DDPM / DDIM / PF-ODE 啥关系 (Appendix C)

这是 paper 最 deep 的部分。Qiang Liu 把 DDIM, VP ODE, sub-VP ODE, VE ODE 全部装进一个统一框架:

$$ \min_v \int_0^1 \mathbb{E}\left[\| \dot X_t - v(X_t, t) \|^2 \right] dt $$

其中 X_t = α_t · X₁ + β_t · X₀ 是任意 smooth interpolation, Ẋ_t = α̇_t · X₁ + β̇_t · X₀。

各方法的 α_t, β_t 对照表:

| Method | α_t | β_t |
|---|---|---|
| VP ODE / DDIM | exp(-a(1-t)²/4 - b(1-t)/2) | √(1 - α_t²) |
| sub-VP ODE | exp(-a(1-t)²/4 - b(1-t)/2) | 1 - α_t² |
| VE ODE | 1 | σ_min √(r^{2(1-t)} - 1) |
| Linear Rectified Flow | t | 1 - t |

(其中 a=19.9, b=0.1 是 DDPM 默认超参, r 是数据集最大距离/σ_min)

关键观察:
- DDIM 的 β_t ≠ 1 - α_t, 所以轨迹**天生就是弯的**, reflow 救不回来
- DDIM 的 α_t 是指数衰减, 前期变化慢 (大部分时间停在噪声附近), 后期才急速 denoise, **速度不均匀**
- Rectified Flow 用最朴素的 α_t = t, β_t = 1 - t, 路径笔直, 速度恒定

paper Figure 11/12 用 2D toy 演示: DDIM/VP ODE 在 Gaussian mixture 上轨迹弯弯曲曲, reflow 也没用; RF 一次 reflow 就接近直线。

---

## 7. 实验数据 (Table 1)

CIFAR10 无条件生成 (DDPM++ 架构, 同等 capacity 公平比较):

| Method | NFE | FID ↓ | Recall ↑ |
|---|---|---|---|
| 1-Rectified Flow (RK45) | 127 | **2.58** | **0.57** |
| 2-Rectified Flow (RK45) | 110 | 3.36 | 0.54 |
| VP ODE (RK45) | 140 | 3.93 | 0.51 |
| sub-VP ODE (RK45) | 146 | 3.16 | 0.55 |
| VP SDE (2000 Euler steps) | 2000 | 2.55 | 0.58 |
| **2-RF + Distill (1 step)** | **1** | **4.85** | **0.50** |
| 3-RF + Distill (1 step) | 1 | 5.21 | 0.51 |
| VP ODE + Distill (1 step) | 1 | 16.23 | 0.29 |
| sub-VP ODE + Distill (1 step) | 1 | 14.32 | 0.35 |

(其中 NFE = number of function evaluations, 即前向神经网络的次数)

读这张表的姿势:
- **Full simulation (RK45)**: 1-RF FID 2.58 是 ODE 类 SOTA, 比 VP/sub-VP ODE 都好, 而且用更少 NFE
- **One-step**: 2-RF+Distill 用 1 步达到 FID 4.85, 当时 (2022) 是 U-Net 架构 one-step SOTA
- DDIM 即使 distill 到 one-step 还是 FID 14-16, 因为路径不直, distill 救不了
- Recall 指标 (diversity) RF 全面碾压 GAN 类 (StyleGAN2+ADA 是 0.49)

---

## 8. 一个超漂亮的诊断 trick (Figure 4, 18)

paper 想可视化 flow 直不直, 用了这招: 对轨迹上任意点 z_t, 外推到终点:

$$ \hat z_1^t = z_t + (1-t) \cdot v(z_t, t) $$

变量:
- z_t 是当前 ODE 状态
- (1-t) 是剩余时间长度
- v(z_t, t) 是当前速度

直觉: 如果轨迹是直线且匀速, 那从任何中间点 z_t 用速度 v 外推 (1-t) 时间, 应该到同一个终点 z_1。所以 $\hat z_1^t$ 应该和 t 无关。

paper 显示: 1-Rectified Flow 的 $\hat z_1^t$ 随 t 变化较大 (弯), 2-Rectified Flow 的 $\hat z_1^t$ 几乎不依赖 t (直)。Figure 18 在 CIFAR10 上验证, 非常直观。这个 diagnostic 后来在 SD3 / FLUX 的 ablation 里也被大量使用。

---

## 9. Image-to-image translation (Section 3.2)

把 π₀ 设成 source domain (人脸), π₁ 设成 target domain (猫脸), 同样的 algorithm 直接做 image-to-image translation。完全不需要 CycleGAN 的 adversarial loss 和 cycle consistency, 因为 ODE 天然 time-reversible (反向积分就是逆映射)。

但有个细节: 直接学 v 会让 identity 也变 (人变成完全的猫)。paper 提出用 saliency-weighted loss (Eq. 4):

$$ \min_v \int_0^1 \mathbb{E}\left[\left\| \nabla h(X_t)^\top (X_1 - X_0 - v(X_t, t)) \right\|^2 \right] dt $$

- h(x) 是一个预训练 classifier (ImageNet pretrained, fine-tuned 在两 domain 之间)的 latent
- ∇h(x) 是 h 对 x 的 Jacobian, 当 saliency mask 用
- 直觉: identity 相关的像素 (classifier 关心的) 严格拟合, style 相关的像素允许 network 自由发挥

结果 (Figure 8, 9): AFHQ cat↔wild, MetFace↔CelebA 互相 transfer, 2-Rectified Flow 用 N=1 Euler step 就能输出 style 正确的结果。

---

## 10. Reflow 算法 (Algorithm 4)

完整 pipeline:

```python
# Stage 1: 训第一版 flow
model_1 = train(pairs=[(x0_noise, x1_cat) for ...])

# Stage 2: 用 model_1 生成新配对, 训第二版
new_pairs = []
for x0 in noise_loader:
    x1 = ode_solve(model_1, x0, n_steps=100)  # forward simulate
    new_pairs.append((x0, x1))
model_2 = train(pairs=new_pairs)

# Stage 3 (optional): distill 到 one-step
# 对 k-step distillation, 采样 t in {0, 1/k, ..., (k-1)/k}
# k=1 时换 LPIPS loss (perceptual) 比 L2 更好
```

工程细节 (Appendix E):
- DDPM++ U-Net 架构, Adam, lr=2e-4, dropout=0.15
- EMA decay 0.999999
- 每次 reflow 生成 4M pairs, fine-tune 300k steps
- 256×256 用 NCSN++ 架构
- One-step distill 用 LPIPS 替 L2

---

## 11. 后续发展 (这是我补充的, paper 写在 2022 年 9 月)

Rectified Flow 真正的爆发是 2023-2025:

**InstaFlow** (Yan Liu 等, 2023, arXiv:2309.06370, https://arxiv.org/abs/2309.06370)
Qiang Liu 同组, 把 RF 应用到 Stable Diffusion, 1-step 接近 50-step SD 的 FID, 速度 10-100×。这是 RF 走向大规模文生图的桥梁。

**Stable Diffusion 3** (Esser et al., 2024, arXiv:2403.03206, https://arxiv.org/abs/2403.03206)
Stability AI 官方采用 Rectified Flow 作为核心, 取代 DDPM。SD3 paper 里有大量 ablation 显示 RF > DDPM/EDM。α_t = t 是 SD3 默认。

**FLUX** (Black Forest Labs, 2024, https://blackforestlabs.ai)
SD 原班人马离开 Stability 后建立, 采用 flow matching。当前最好的开源文生图模型之一。

**Latent Consistency Models / SDXL Turbo** (Luo et al., 2023, https://arxiv.org/abs/2310.04378; https://arxiv.org/abs/2311.17042)
LCM 借鉴 Consistency Model 的 self-consistency 思路, 但大量吸收 RF 的 reflow+distill 范式。SDXL Turbo 用 adversarial + distill 但底层 flow-based。

**Consistency Models** (Song et al., 2023, https://arxiv.org/abs/2303.01969)
OpenAI 另一条 one-step 路线, 用 self-consistency loss, 不依赖 reflow。是 RF 的"竞争者", 经常被 combine (CTM, MeanFlow)。

**MeanFlow** (Geng et al., 2025, https://arxiv.org/abs/2505.13479)
从 mean (时间平均) 角度建模 one-step learning, RF 的理论扩展。

**Shortcut Models** (Boots et al., 2024, https://arxiv.org/abs/2410.12557)
Facebook AI, 让一步生成支持 multistep 一致性, 基于 flow / RF 思路。

**Meta MovieGen** / Sora 类视频模型
视频生成基本都迁移到 flow matching, 因为高维数据上 step 数减少更关键。Meta MovieGen 技术报告里明确用 flow matching。

**Voicebox / Audiobox**
语音生成 (https://arxiv.org/abs/2306.03926, https://arxiv.org/abs/2310.15913) 都基于 flow matching。

**生物 / 蛋白**
Boltzmann generator, AlphaFold-ligand 类工作开始用 flow matching, 因为 RF 给的 deterministic coupling 适合做 latent representation (DDPM 的 (Z₀, Z₁) 是 random 的, 不适合)。

---

## 12. 一个隐藏的几何 / 物理联系

完美直线的 flow v 满足 inviscid Burgers equation:

$$ \partial_t v + (\partial_z v) \cdot v = 0 $$

证明: 沿 ODE 轨迹, dv/dt = ∂_z v · Ż_t + ∂_t v = ∂_z v · v + ∂_t v = 0 (因为 v 沿直线是常数)。

Burgers equation 在流体力学里以"shock formation"著名 — 平滑初值会发展出 discontinuity (交叉)。Reflow 在数值上就是在消去这些 shock。这个联系在 Albergo & Vanden-Eijnden (https://arxiv.org/abs/2209.15571) 后续 work 里被进一步发挥。

---

## 13. 跟你 (Karpathy) 的潜在关联

Andrej, 你最近对 nanoGPT, llama2.c, Eureka Labs 投入很大。RF 角度看, 几个联想:

**离散 token 的 rectified flow**
GPT 是 autoregressive, 但 continuous relaxation 下能否做 RF? Diffusion-LM (https://arxiv.org/abs/2205.14217), Plaid, MDLM 都尝试过。RF 比 DDPM 更适合做这个, 因为 straight path → 可以一步生成一个 token, 可能是 parallel decoding 的桥梁。MAR (https://arxiv.org/abs/2406.11837), Masked Diffusion (https://arxiv.org/abs/2406.07324), Squar (https://arxiv.org/abs/2410.18972) 试图统一 AR 和 diffusion, RF 提供 natural 几何骨架。

**视频生成**
你对 world simulator 有 commentary, Meta MovieGen 和 Sora 都用 flow matching, RF 的 straight trajectory 在视频高维数据上更有意义。

**教育角度**
RF 的"直线 + 回归"比 DDPM 的"SDE + reverse-time SDE + score function"在教学上简洁太多。Eureka Labs 入门课用 RF 讲比 DDPM 友好得多。Qiang Liu 在 UT 的课 (https://www.cs.utexas.edu/~lqili/) 把 RF 讲得很清晰。

**Latent space + LLM 集成**
SD3 用 3-text-encoder + flow matching, 把 LLM embedding 直接作为 conditioning, 这条路线跟你对 multimodal 的兴趣相关。

---

## 14. 一图胜千言的 mental model

如果只记一个 picture, 记这个:

```
π₀ (noise) ────────直线─────── π₁ (cat)
       \         /
        \  X    /     ← linear interpolation, 直线交叉
         \  / 
          X         
          ↑
     v^X = E[X₁-X₀ | X_t]   ← 在交叉点取条件期望, 平均方向
          ↓
π₀ (noise) ────────不交叉─────── π₁ (cat)  ← rectified flow, ODE 轨迹不交叉
```

Reflow 就是把右图再当数据画一次直线, 第二次直线就更直了, 因为端点配对已经被 rewire 过一次。O(1/k) 速率越来越直。

---

## 参考链接

- Rectified Flow 原文: https://arxiv.org/abs/2209.03003
- 代码: https://github.com/gnobitab/RectifiedFlow
- Qiang Liu 课程: https://www.cs.utexas.edu/~lqili/
- Qiang Liu follow-up OT paper: https://arxiv.org/abs/2209.14577
- Flow Matching (Lipman et al.): https://arxiv.org/abs/2210.02747
- Stochastic Interpolants (Albergo & Vanden-Eijnden): https://arxiv.org/abs/2209.15571
- Action Matching (Neklyudov et al.): https://arxiv.org/abs/2210.06662
- InstaFlow: https://arxiv.org/abs/2309.06370
- Consistency Models: https://arxiv.org/abs/2303.01969
- LCM: https://arxiv.org/abs/2310.04378
- SDXL Turbo: https://arxiv.org/abs/2311.17042
- Stable Diffusion 3: https://arxiv.org/abs/2403.03206
- Voicebox: https://arxiv.org/abs/2306.03926
- Meta Audiobox: https://arxiv.org/abs/2310.15913
- MAR: https://arxiv.org/abs/2406.11837
- Diffusion-LM: https://arxiv.org/abs/2205.14217
- Schrödinger Bridge survey: https://arxiv.org/abs/2405.21098

如果你想要我针对某个具体部分再展开 (比如 Theorem D.5 在 c=‖·‖² 特例下的几何 proof, 或者 Burgers equation shock 与 reflow 的物理对应, 或者 InstaFlow 的 distill 配方细节), 告诉我哪个, 我再钻深。

---

# Rectified Flow: 一篇重新定义生成模型 inference 范式的 paper

你好 Andrej, 这篇 paper 在我看来是 2022 年生成模型领域被严重低估的"分水岭"工作, 它和 Flow Matching (Lipman et al.), Stochastic Interpolants (Albergo & Vanden-Eijnden) 几乎同时被独立提出, 共同把 diffusion model 的复杂 SDE 框架回归到了最朴素的"两点之间直线最短"的几何直觉。Stable Diffusion 3 / SD3.5, FLUX, Meta MovieGen, Microsoft 的视频生成以及最新的 SDXL Turbo / LCM 路线背后都有它的影子。下面我尽量把直觉、数学、实验和后续发展都讲透。

arXiv: https://arxiv.org/abs/2209.03003
Code: https://github.com/gnobitab/RectifiedFlow
Qiang Liu 的 tutorial: https://www.cs.utexas.edu/~lqili/teaching.html

---

## 1. 问题设定: 统一 generative modeling 与 domain transfer

paper 把一个非常普遍的问题抽象出来: 给定两个经验分布 π₀ 与 π₁ 的样本 (在 ℝ^d 上), 求一个 transport map T: ℝ^d → ℝ^d, 使得若 Z₀ ~ π₀, 则 Z₁ = T(Z₀) ~ π₁。

- 生成模型: π₀ = 𝒩(0, I), π₁ = data distribution (图像)
- domain transfer: π₀ = source domain (人脸), π₁ = target domain (猫脸)
- Schrödinger bridge / diffusion bridge: 更一般的两端固定问题

这一定义就把 GAN, VAE, normalizing flow, DDPM, CycleGAN, SDEdit 全部装进同一个框架里。OT (Optimal Transport) 给出的 c-optimal coupling 是这个问题的"最强解", 但在高维下计算几乎不可行, 而且 transport cost 本身并不是 ML 任务真正关心的指标。Rectified Flow 的关键 insight: 找一个**足够直 (straight) 的 coupling**就够了, 不需要找 c-optimal coupling。直线的耦合 → ODE 轨迹是直线 → 一次 Euler 步就能精确求解 → 推断 zero-cost。

---

## 2. 核心方法: 用直线插值"因果化"任意 coupling

### 2.1 Linear interpolation 与它"不因果"的病

给定样本对 (X₀, X₁), 它们可能 independent (X₀ ⊥ X₁, 即 (X₀, X₁) ~ π₀ × π₁) 或者任何其他 coupling。定义 linear interpolation:

$$ X_t = t X_1 + (1-t) X_0, \quad t \in [0,1] $$

这里下标 0/1 表示起点/终点, t 是连续时间 ∈ [0,1]。这条路径满足 ODE:

$$ \mathrm{d}X_t = (X_1 - X_0)\,\mathrm{d}t $$

(X₁ - X₀) 是从起点指向终点的常向量。问题在于: 这个 ODE 是 **anticipating / non-causal** 的 — 要计算 X_t 的导数, 必须提前知道终点 X₁, 所以没法用它做生成。

### 2.2 Rectified flow 的训练目标: 把 non-causal 拟合成 causal

核心 loss 就是把它当一个标准回归:

$$ \min_v \int_0^1 \mathbb{E}\left[\left\| (X_1 - X_0) - v(X_t, t) \right\|^2\right] \mathrm{d}t \tag{1} $$

- v: ℝ^d × [0,1] → ℝ^d, 是要学习的 drift (velocity field), 通常用 U-Net (DDPM++ 结构) 或 NCSN++ 参数化
- X_t = t X₁ + (1-t) X₀ 是输入, 给网络看的"位置 + 时刻"
- (X₁ - X₀) 是回归 target
- 期望 𝔼[·] 关于 (X₀, X₁) 的随机性 (训练时 t ~ Uniform[0,1])
- L² 范数 ‖·‖² 是欧氏距离平方

求解最小化得到闭式解 (Theorem 等价):

$$ v^X(x, t) = \mathbb{E}[X_1 - X_0 \mid X_t = x] \tag{2} $$

也就是: 在"位置 x、时刻 t"处, 所有穿过这个 (x,t) 的 linear interpolation 路径的方向 X₁ - X₀ 的**条件期望**。这是一个 "myopic / memoryless" 的速度, 不需要知道未来。

### 2.3 训练 algorithm 极简 (Algorithm 1-4)

```python
# Algorithm 2 (训练)
for x0, x1 in Data:                       # x0 ~ π₀, x1 ~ π₁
    t = torch.rand(batch_size)             # t ~ U[0,1]
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    loss = (model(x_t, t) - target).pow(2).mean()
    loss.backward(); optimizer.step()
```

完全标准的 supervised regression, 无 minimax, 无 KL, 无 ELBO, 无 score function 估计, 无 Langevin dynamics。这正是它能够 scale 到大模型的原因。

---

## 3. 三大理论性质 (Appendix D)

### 3.1 Marginal preserving (Theorem D.3)

定义 v^X(x, t) = 𝔼[Ẋ_t | X_t = x] (Definition D.1, 这里 Ẋ_t = ∂_t X_t 是 X 的时间导数; 在 linear case 下 Ẋ_t = X₁ - X₀)。如果 ODE dZ_t = v^X(Z_t, t) dt 有唯一解, 那么对任意 t ∈ [0,1]:

$$ \mathrm{Law}(Z_t) = \mathrm{Law}(X_t) $$

**直觉**: 对任意 test function h, 用 chain rule:
$$ \frac{\mathrm{d}}{\mathrm{d}t}\mathbb{E}[h(X_t)] = \mathbb{E}[\nabla h(X_t)^\top \dot X_t] = \mathbb{E}[\nabla h(X_t)^\top v^X(X_t, t)] $$
最后一步用了 tower property: 𝔼[Ẋ_t | X_t] = v^X(X_t, t)。这就是 continuity equation ∂_t ρ_t + ∇·(v^X_t ρ_t) = 0 的弱形式。Z_t 由同一个 v^X 驱动, 起点相同 (Z₀ = X₀ ~ π₀), 所以 marginal 分布处处相同。

含义: 我们并没"破坏" X_t 的 marginals, 只是把 X_t 的"随机的、跨时间的"耦合改写成一个"确定的、Markov 的、causal"的耦合, 同时保留每个时刻的边缘分布。**X_t 是 non-Markov (路径交叉) 的随机过程, Z_t 是 Markov 且确定性的 ODE**。

### 3.2 Non-crossing 与降低 transport cost (Theorem D.5)

ODE 轨迹不能交叉 — 如果两条轨迹在 (z, t) 处相遇, 那么从该时刻起 ODE 的解唯一性要求它们重合。这是 well-posed ODE 的本质。

**降低凸 cost 定理**: 对任意凸函数 c: ℝ^d → ℝ:

$$ \mathbb{E}[c(Z_1 - Z_0)] \le \mathbb{E}[c(X_1 - X_0)] $$

证明 (Appendix D.2) 用两次 Jensen 不等式 + 一次 marginal preserving:

$$
\begin{aligned}
\mathbb{E}[c(Z_1 - Z_0)] &= \mathbb{E}\left[c\left(\int_0^1 v^X(Z_t, t)\,\mathrm{d}t\right)\right] \\
&\le \mathbb{E}\left[\int_0^1 c(v^X(Z_t, t))\,\mathrm{d}t\right] \quad \text{(Jensen on } c, \int \mathrm{d}t = 1) \\
&= \mathbb{E}\left[\int_0^1 c(v^X(X_t, t))\,\mathrm{d}t\right] \quad \text{(Law}(Z_t)=\text{Law}(X_t)) \\
&= \mathbb{E}\left[\int_0^1 c(\mathbb{E}[X_1 - X_0 \mid X_t])\,\mathrm{d}t\right] \\
&\le \mathbb{E}\left[\int_0^1 \mathbb{E}[c(X_1 - X_0) \mid X_t]\,\mathrm{d}t\right] \quad \text{(Jensen on } c) \\
&= \mathbb{E}[c(X_1 - X_0)]
\end{aligned}
$$

**几何直觉 (paper 里一个非常漂亮的图示)**: 想象 X₀ → X₁ 是一根绷紧的弦。在弦交叉的地方, Z_t 把弦"重排"成不交叉的网络。对 c(x) = ‖x‖ (路径长度), 这就是"两点之间直线最短 + 三角不等式": Z₀-Z₁ 的总长度 ≤ X₀-X₁ 直线长度总和 (因为 Z 是不交叉的网络, X 是有交叉的直线, 重新配对可以缩短)。这一点也是为什么 paper 把 RectFlow 叫做"对凸 transport cost 的 Pareto descent" — 它不针对特定 c 优化, 而是同时降低所有凸 c。

### 3.3 Reflow 与 straightening (Theorem D.7)

定义 straightness 度量 (Eq. 3):

$$ S(Z) = \int_0^1 \mathbb{E}\left[\left\| (Z_1 - Z_0) - \dot Z_t \right\|^2\right] \mathrm{d}t $$

S(Z) = 0 当且仅当轨迹 a.s. 是直线且速度恒定 (即 Z_t = tZ₁ + (1-t)Z₀, v(Z_t, t) = Z₁ - Z₀ = const 沿每条路径)。**完美直线的 flow, 一次 Euler 步就能精确求解**: Z₁ = Z₀ + v(Z₀, 0)。这就是 paper 标题 "Flow Straight and Fast" 的来源。

Reflow 算法 (Algorithm 1, 4):
- Z^(k+1) = RectFlow((Z₀^k, Z₁^k))
- 起始 (Z₀^0, Z₁^0) = (X₀, X₁)

**收敛速率** (Theorem D.7):

$$ \min_{k \in \{0,\dots,K\}} S(Z^k) \le \frac{\mathbb{E}[\|X_1 - X_0\|^2]}{K} $$

即 O(1/K) 速率。证明基于 telescoping sum: 𝔼[‖Z₁^k - Z₀^k‖²] - 𝔼[‖Z₁^{k+1} - Z₀^{k+1}‖²] = S(Z^{k+1}) + V((Z₀^k, Z₁^k)), 其中 V 是路径交叉度量 (Eq. 14)。

**Burgers 方程的隐藏链接**: 若 Z_t 是直线, 则 v 满足 inviscid Burgers equation:
$$ \partial_t v + (\partial_z v) v = 0 $$
因为 dv/dt = ∂_z v · v + ∂_t v = 0 (沿特征线 v 是常数)。Burgers equation 的 shock 形成正是"交叉"的物理表现, reflow 就是在数值上消去这些 shock。这一联系在后续 Albergo-Vanden-Eijnden 等的工作里被进一步发挥。

---

## 4. 与 PF-ODE / DDIM / diffusion 的统一视角 (Appendix C)

这是 paper 最 deep 的部分。Qiang Liu 用一个 general nonlinear rectified flow 把 PF-ODE、DDIM 全部装进去:

$$ \min_v \int_0^1 \mathbb{E}\left[w_t \| \dot X_t - v(X_t, t) \|^2\right] \mathrm{d}t $$

其中 X_t = α_t X₁ + β_t X₀ 是任意 smooth interpolation, Ẋ_t = α̇_t X₁ + β̇_t X₀ 是它的时间导数, w_t 是权重。

**Proposition C.1**: 所有 PF-ODE 都是 X_t = α_t X₁ + β_t ξ, ξ ~ 𝒩(0, I) 的特例。具体:

| Method | α_t | β_t |
|---|---|---|
| VP ODE / DDIM | exp(-a(1-t)²/4 - b(1-t)/2) | √(1 - α_t²) |
| sub-VP ODE | exp(-a(1-t)²/4 - b(1-t)/2) | 1 - α_t² |
| VE ODE | 1 | σ_min √(r^{2(1-t)} - 1) |
| **Linear Rectified Flow** | **t** | **1 - t** |

paper 在 Figure 11/12 用 2D Gaussian mixture 演示: VP/sub-VP ODE 的轨迹是**弯曲**的, 而且**速度不均匀** (前期慢, 后期快, 大部分更新集中在 t≈0.5 附近)。原因是 (a) β_t ≠ 1-α_t 让路径弯, (b) 指数 α_t 让速度不均匀。Linear α_t = t 才是 canonical。这解释了 DDIM 为何需要 10-100 步, 而且难以加速 — 它根本不是为 fast sampling 设计的。

paper 用 Figure 13 给了一个非常清晰的对比: VP ODE 的 α_t 曲线在 t<0.5 时几乎平, t>0.5 时陡降 — 直观上"先停在噪声附近, 后期才大量 denoise"。

---

## 5. 实验: 一个简单的 idea 击败了复杂 SDE 框架

### 5.1 CIFAR10 (Table 1a)

| Method | NFE | IS ↑ | FID ↓ | Recall ↑ |
|---|---|---|---|---|
| 1-Rectified Flow (RK45) | 127 | 9.60 | **2.58** | **0.57** |
| 2-Rectified Flow (RK45) | 110 | 9.24 | 3.36 | 0.54 |
| 3-Rectified Flow (RK45) | 104 | 9.01 | 3.96 | 0.53 |
| VP ODE (RK45) | 140 | 9.37 | 3.93 | 0.51 |
| sub-VP ODE (RK45) | 146 | 9.46 | 3.16 | 0.55 |
| VP SDE (Euler, N=2000) | 2000 | 9.58 | 2.55 | 0.58 |
| 2-Rectified Flow + Distill (1 step) | 1 | 9.01 | **4.85** | **0.50** |
| 3-Rectified Flow + Distill (1 step) | 1 | 8.79 | 5.21 | 0.51 |
| VP ODE + Distill (1 step) | 1 | 8.73 | 16.23 | 0.29 |
| sub-VP ODE + Distill (1 step) | 1 | 8.80 | 14.32 | 0.35 |

关键观察:
- **Full simulation**: 1-Rectified Flow FID 2.58 已是 ODE 类 SOTA, Recall 0.57 大幅领先所有 ODE 和 GAN
- **One-step**: 2-RF + Distill FID 4.85 击败所有 prior one-step diffusion/flow 模型 (此前 DDIM distill 是 9.36, TDPM T=1 是 8.91)
- VP/sub-VP ODE 即使 distill 到 one-step 也只有 FID 14-16, 因为路径根本不直, distill 救不了

### 5.2 Reflow 直觉图 (Figure 4, 6, 18)

paper 给了一个超漂亮的诊断: 对 ODE 轨迹上的每个 z_t, 计算"外推终点"
$$ \hat z_1^t = z_t + (1-t) v(z_t, t) $$

如果轨迹是直线, $\hat z_1^t$ 应该与 t 无关 (一条直线, 任何点外推都到同一个终点)。Figure 4 显示 1-Rectified Flow 的 $\hat z_1^t$ 随 t 变化较大, 2-Rectified Flow 几乎不依赖 t — 直观证实 straightening。Figure 18 在 CIFAR10 上同样验证。

### 5.3 High-res 256×256 (Figure 7)

LSUN Bedroom, LSUN Church, CelebA HQ, AFHQ Cat 四个数据集都用 1-Rectified Flow 训练, 都生成高质量图像。RF 训练成本和 DDPM 训练成本几乎一样 (因为是同一个 U-Net, 同一个 regression loss), 但 inference 时间可以低一个数量级。

### 5.4 Image-to-image translation (Figure 8, 9)

domain transfer loss (Eq. 4):
$$ \min_v \int_0^1 \mathbb{E}\left[\left\| \nabla h(X_t)^\top (X_1 - X_0 - v(X_t, t)) \right\|_2^2\right] \mathrm{d}t $$

- h(x) 是一个预训练 classifier (ImageNet pretrained, fine-tuned 在两 domain 之间)的 latent feature
- ∇h(x) 是 h 对 x 的 Jacobian, 用作 saliency mask
- 直觉: 对图像 identity (主要 object) 影响大的像素, 我们要求 v 在这些方向上严格拟合 (X₁-X₀); 对 style-only 像素, v 可以"自由发挥"。这避免了 CycleGAN 的 adversarial 训练和 cycle consistency, ODE reversibility 天然保证 cycle consistency。

结果: AFHQ cat↔wild, MetFace↔CelebA 互相 transfer, 2-Rectified Flow 一步 Euler 就能输出风格正确的结果。

### 5.5 Domain adaptation (Table 3, Appendix E.1)

把测试 domain 的 latent feature 通过 rectified flow 映回训练 domain, 再用训练 domain 的分类器预测。OfficeHome 69.2%, DomainNet 41.4%, 与 Deep CORAL 持平甚至更好, 大幅超过 ERM, IRM, ARM, Mixup, MLDG 等基线。这是一个 RF 在 discriminative ML 任务的副产品应用。

---

## 6. 工程细节 (Appendix E)

- **网络架构**: DDPM++ (U-Net with attention) for CIFAR10, NCSN++ for 256×256
- **Optimizer**: Adam, lr 2e-4, dropout 0.15, EMA decay 0.999999
- **Reflow 数据**: 一次 reflow 生成 4M pairs (z₀, z₁), 然后 fine-tune 300k steps
- **Distillation trick**: 对 k-step distillation, 采样 t ∈ {0, 1/k, ..., (k-1)/k} 而不是 U[0,1]
- **One-step distillation**: k=1 时把 L2 loss 换成 LPIPS (perceptual loss), 经验上更好
- **Sampling**: 默认 Euler step size 1/N, 或者 RK45 (Scipy, adaptive)

---

## 7. 后续发展 (这部分是我补充的, 原文没有)

Rectified Flow 这篇文章的影响力在 2023-2025 才真正爆发:

1. **InstaFlow** (Liu et al., 2023, arXiv:2309.06370, https://arxiv.org/abs/2309.06370) — 同一组 (Liu, Wu, Ye, Qiang Liu) 把 RF 应用到 Stable Diffusion, 用 1-step rectified flow + reflow + distill 实现了接近 50-step Stable Diffusion 的 FID, 速度快 10-100×。这是 RF 走向大规模文生图的关键工作。

2. **Stable Diffusion 3 / SD3.5** (Esser et al., 2024, arXiv:2403.03206, https://arxiv.org/abs/2403.03206) — Stability AI 官方采用了 Rectified Flow (他们叫 "rectified flow" 或 "flow matching")作为 SD3 的核心采样方法, 取代 DDPM。SD3 paper 里有大量 ablation 显示 RF > DDPM/EDM。

3. **FLUX** (Black Forest Labs, 2024) — 由 SD 原班人马离开 Stability 后建立, 采用 flow matching (Rectified Flow)。当前最好的开源文生图模型之一。

4. **Latent Consistency Models / SDXL Turbo / LCM-LoRA** (Luo et al., 2023, arXiv:2310.04378, https://arxiv.org/abs/2310.04378) — 虽然 LCM 借鉴 Consistency Model 的 self-consistency 思路, 但同时大量吸收了 RF 的 reflow+distillation 范式。SDXL Turbo (https://arxiv.org/abs/2311.17042) 用 adversarial + distill 路线但底层仍是 flow-based。

5. **Consistency Models** (Song et al., 2023, arXiv:2303.01969, https://arxiv.org/abs/2303.01969) — OpenAI 的另一条 one-step 路线, 用 self-consistency loss, 不依赖 reflow。这是 RF 的"竞争者", 但两者经常被 combine (Consistency Trajectory Model, MeanFlow)。

6. **MeanFlow** (Geng et al., 2025, arXiv:2505.13479) — 把 one-step learning 直接从 mean (时间平均) 角度建模, 是 RF 的进一步理论扩展。

7. **Shortcut Models** (Boots et al., 2024, arXiv:2410.12557) — Facebook AI 的工作, 让一步生成支持 multistep 一致性, 也是基于 flow / RF 思路。

8. **Stable Diffusion 3.5 / Meta MovieGen / Sora 类** — 视频生成模型基本都迁移到 flow matching 框架, Sora 技术报告里有相关线索。

9. **Flow Matching in biology / protein** — Boltzmann generator, AlphaFold-ligand 等都开始用 flow matching, 因为 RF 给出的 deterministic coupling 适合做 latent representation, 而 DDPM 给的 (Z₀, Z₁) 是 random 的 (paper Appendix A 明确指出了这一点)。

10. **Cond-OT 和 Schrödinger bridge** — RF 框架让 SB 也有了 "straight" 的对应版本 (SF²M, Bridge Matching, https://arxiv.org/abs/2305.15010)。

11. **语音 / 音乐** — Voicebox (https://arxiv.org/abs/2306.03926), Meta Audiobox (https://arxiv.org/abs/2310.15913) 都基于 flow matching。

---

## 8. 直觉总结 (build your intuition)

如果让我用一句话讲 rectified flow 的精髓:

> **两点之间直线最短。diffusion model 学了一堆弯曲的轨迹是因为它从 SDE 反推 ODE; rectified flow 直接从 (X₀, X₁) 之间画直线, 然后用回归把这条直线"因果化"成可模拟的 ODE。Reflow 就是反复画直线 — 每画一次, ODE 都更直一点 (O(1/k) 直化), 直到一条 Euler 步就能精确求解。**

几个关键的 mental model 值得反复在脑子里跑:
- **公路 vs 交通**: X_t 是连接 π₀ 与 π₁ 的"所有可能公路" (含交叉); rectified flow 是"在这公路网上跑车的交通流", 在交叉点用条件期望把车流 rewire 到不交叉。两次 Jensen 不等式就是"重排车辆以缩短总路程"的数学形式。
- **Marginal 不变, joint 变**: rectified flow 不改变任何时刻的分布密度, 只是把"哪个 X₀ 配哪个 X₁"重新配对。这就像 OT 但弱化 — OT 是找最优配对, RF 是找"直且不交叉"的配对。
- **ODE vs SDE 之争**: paper Appendix A 是一份非常清晰的 ODE 优于 SDE 的论证清单 (conceptual simplicity, time reversibility, latent space quality, training difficulty, expressive power, manifold data)。SDE 唯一的优势是建模时间相关噪声。这是后续 SD3 / FLUX 全面转 ODE 的理论依据。
- **distillation vs reflow 的区别**: distillation 是"忠实复制" (Z₀^k, Z₁^k) 的关系; reflow 是"再造一个更直的 coupling"。distillation 只在最后阶段做 fine-tune, reflow 可以重复多次。

---

## 9. 可联想的其他理论分支 (Hallucination-friendly section)

- **Bregman divergence + convex cost**: Theorem D.5 的 Jensen 证明结构暗示 RF 在 Bregman 几何下可能有更一般的版本。
- **Schrödinger bridge**: 给定 (X₀, X₁) coupling, 最可能 path 是 entropic OT。RF 是它的 zero-temperature limit (https://arxiv.org/abs/2405.21098, De Bortoli 等后续)。
- **Lipschitz regularity**: Appendix B 给出 v^X 在 t < 1 上 Lipschitz 的条件 (1/(1-t) 的奇异)。这暗示 t=1 处有奇点, 工程上人们用 t ∈ [0, 1-ε] 或 σ-噪声平滑处理。
- **Mean Field Game**: RF 的 reflow 在 MFG 角度是"agent 重新决定策略以减少拥挤"。
- **Action Matching** (Neklyudov et al., https://arxiv.org/abs/2210.06662): 同期工作, 用变分方法学习 stochastic dynamics, 与 RF 互补。
- **OT-FLOW / RGF** (Onken, Ruthotto 等, https://arxiv.org/abs/2002.02589): 早期用 OT regularizer 训练 neural ODE 的工作, RF 把 regularizer 移除了, 直接 loss 即可。

---

## 10. 对你的 work 可能的关联点

Andrej, 你最近对 nanoGPT, llama2.c, Eureka Labs / 教育内容投入很大。从 RF 角度看, 我有几个联想:
- **离散 token 的 rectified flow**: 文本生成 GPT 是 autoregressive, 但在 continuous relaxation 下能否做 RF? Diffusion-LM (Li et al., https://arxiv.org/abs/2205.14217), Plaid, MDLM 都尝试过。RF 可能比 DDPM 更适合做 this 因为 straight path → 可以一步生成一个 token (并行 decoding 的桥梁)。
- **Sora / 视频生成**: 你对 world simulator / video 有不少 commentary, Meta MovieGen 和 OpenAI Sora 都用 flow matching, RF 的 straight trajectory 在视频高维数据上更有意义 (减少 steps)。
- **教育角度**: RF 的"直线 + 回归"比 DDPM 的"SDE + reverse-time SDE + score function"在教学上简洁太多, Eureka Labs 的入门课用 RF 比 DDPM 友好。Qiang Liu 在 UT 的课 (https://www.cs.utexas.edu/~lqili/) 把 RF 讲得很清晰。
- **Latent space + LLM 集成**: SD3 用 3-text-encoder + flow matching, 把 LLM embedding 直接作为 conditioning, 这条路线和你对 multimodal 的兴趣相关。
- **Autoregressive ↔ Flow 统一**: 最近一些工作 (MAR, https://arxiv.org/abs/2406.11837; Masked Diffusion, https://arxiv.org/abs/2406.07324; Squar, https://arxiv.org/abs/2410.18972) 试图把 AR 与 diffusion 统一, RF 提供了一个 natural 几何骨架。

---

## 参考链接汇总

- 原文: https://arxiv.org/abs/2209.03003
- 代码: https://github.com/gnobitab/RectifiedFlow
- Qiang Liu 课程: https://www.cs.utexas.edu/~lqili/
- Flow Matching (Lipman et al.): https://arxiv.org/abs/2210.02747
- Stochastic Interpolants (Albergo & Vanden-Eijnden): https://arxiv.org/abs/2209.15571
- Action Matching (Neklyudov et al.): https://arxiv.org/abs/2210.06662
- InstaFlow: https://arxiv.org/abs/2309.06370
- Consistency Models: https://arxiv.org/abs/2303.01969
- LCM: https://arxiv.org/abs/2310.04378
- SDXL Turbo: https://arxiv.org/abs/2311.17042
- Stable Diffusion 3: https://arxiv.org/abs/2403.03206
- Voicebox: https://arxiv.org/abs/2306.03926
- Meta Audiobox: https://arxiv.org/abs/2310.15913
- MAR (Autoregressive + flow): https://arxiv.org/abs/2406.11837
- Diffusion-LM: https://arxiv.org/abs/2205.14217
- Liu 的 follow-up OT paper: https://arxiv.org/abs/2209.14577
- Schrödinger Bridge survey: https://arxiv.org/abs/2405.21098

如果你想要我把某一个细节再钻得更深 (例如 Theorem D.5 在 c(·)=‖·‖² 下的特例, 或者 Burgers equation shock 与 reflow 的几何对应, 或者 InstaFlow 的具体 distill 配方), 告诉我具体哪一块, 我再展开。
