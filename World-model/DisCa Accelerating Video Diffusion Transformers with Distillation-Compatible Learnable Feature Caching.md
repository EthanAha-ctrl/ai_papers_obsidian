---
source_pdf: DisCa Accelerating Video Diffusion Transformers with Distillation-Compatible
  Learnable Feature Caching.pdf
paper_sha256: e72175121cb043571597ecb63da52ec6440d0f5fa4cf18d2e04dd73d8c248ba3
processed_at: '2026-08-03T22:07:25-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 DisCa

---

## 一句话版本

**把大视频生成模型加速 12 倍，质量几乎不掉。**

---

## 故事背景

HunyuanVideo 是目前最强的开源视频生成模型之一。生成一个 5 秒、704p 的视频，需要跑 50 步推理，单次 ~1155 秒。太慢了，根本没法部署。

业界有两条加速路线：

**路线 A：蒸馏** — 训练模型用更少步数出图。比如 MeanFlow 能把 50 步压到 1 步。图像上效果很好，视频上崩了。

**路线 B：缓存** — 推理时不每步都跑完整网络，复用上一步的中间特征。比如 DeepCache、TaylorSeer。免训练，但加速比一高就模糊、变形。

**两条路线各自用还行，合在一起用反而崩得更厉害。** 这篇 paper 就在解决这个"合体"问题。

---

## 为什么"蒸馏 + 缓存"会崩

用个比喻。

想象你从北京走到上海，走 50 步 vs 走 10 步。

走 50 步时，每一步只挪几厘米，相邻两步位置几乎一样。这时候你说"第 31 步的位置大概跟第 30 步差不多"——这个假设很靠谱。**这就是缓存的工作原理：相邻 timestep 特征相似，直接复用。**

走 10 步时，每步跨几百公里。第 3 步在济南，第 4 步就到南京了。你还说"第 4 步跟第 3 步差不多"——完全错。**蒸馏后步数少了，每步跨的距离大，相邻特征差异巨大，缓存复用就废了。**

Figure 1 画的就是这个：
- (a) 未蒸馏：trajectory 密集，相邻点近，缓存有效
- (b) 蒸馏后：trajectory 稀疏，相邻点远，缓存失效

传统缓存方法（TaylorSeer）用 Taylor 多项式去"预测"下一步特征。在密集 trajectory 上 Taylor 一阶展开够用，在稀疏 trajectory 上 Taylor 完全 capture 不了高维特征的非线性演化。

---

## DisCa 的两个核心动作

### 动作 1：让蒸馏本身别太激进 — Restricted MeanFlow

MeanFlow 是 Kaiming He 组今年 5 月的工作，核心 idea 是把训练目标从"瞬时速度"换成"平均速度"。原本设计目标是 one-step 生成（从 noise 一步到 data）。

在图像上 one-step 没问题。在视频大模型上 one-step 太激进，训练发散，生成出来全是 artifact 和变形。

**为什么？** MeanFlow 训练目标里有 Jacobian-vector product (JVP) 项，长序列上数值误差大。而且要求模型一步从 $t=0$ 跳到 $t=1$，这个跨度对视频大模型太大。

**解法很简单粗暴：把训练时"跨度大"的样本直接砍掉。**

原 MeanFlow 训练时 sample 区间 $\mathcal{T} = t - r \in [0, 1]$（0 到 1 全范围）。

Restricted MeanFlow 改成 $\mathcal{T} \in [0, \mathcal{R}]$，$\mathcal{R} = 0.2$。

意思是：训练时只让模型学"跨 0.2 的小区间平均速度"，不学"跨 1.0 的大区间"。相当于只教模型走小步，不让它学跑。

**效果**：在 10 步生成场景，比原 MeanFlow semantic score 高 12 个百分点。Table 1 数据：

| 方法 | 10 步 semantic |
|---|---|
| MeanFlow | 60.9 |
| Restricted MF (R=0.2) | 68.2 (+12.0) |

Figure 3 可视化也能看到 MeanFlow 生成变形严重，Restricted MF 保持正常。

这个 trick 简单到有点可疑，但 ablation (Table 3) 显示这是三个 component 里贡献最大的（去掉它 semantic 掉 5.9%）。

### 动作 2：用神经网络替代 Taylor 公式 — Learnable Feature Caching

既然手工 Taylor 公式在稀疏 trajectory 上 capture 不了特征演化，那就让神经网络来学。

**TaylorSeer 的公式**：
$$\mathcal{F}_{pred} = \mathcal{F}(x_t) + \sum_{i=1}^m \frac{\Delta^i \mathcal{F}}{i! \cdot N^i} (-k)^i$$

这是个手工设计的低阶多项式，靠特征的多阶差分做预测。

**DisCa 的方案**：训一个小神经网络 $\mathcal{P}$，输入是 cache tensor $\mathcal{C}$ + 当前 $x_{t'}$ + timestep embedding，输出预测的 mean velocity。

**Predictor 架构**：就 2 个 DiT block，参数量 < 大模型的 4%。Figure 2c。

**推理流程**（Figure 2a）：
- 第 1 步：完整跑一遍大模型，得到 cache $\mathcal{C}$
- 第 2-N 步：只用小 predictor 跑，复用 $\mathcal{C}$ 作为输入
- 第 N+1 步：再跑一次大模型，刷新 cache
- 循环

比如 N=4：20 步里只有 5 次跑大模型，其他 15 次跑小 predictor。大模型慢，小 predictor 几乎不要时间，所以加速明显。

**训练流程**（Figure 2b, Algorithm 3）：
- 采样一个 $t$ 和一个 $t' = t - \Delta$（$\Delta \in [0, 0.2]$ 随机）
- 大模型跑两次：$\mathcal{M}(x_t, r, t)$ 得到 cache $\mathcal{C}$，$\mathcal{M}(x_{t'}, r', t')$ 得到 ground truth $u_{tar}$
- Predictor 跑一次：$\mathcal{P}(\mathcal{C}, x_{t'}, r', t')$ 得到 $u_{pred}$
- MSE loss：$||u_{pred} - u_{tar}||^2$
- 再加 GAN loss：判别器看 predictor 和大模型各自"走一步"后的 latent state，区分真假

**为什么加 GAN**：单用 MSE 训 predictor 会输出模糊、缺高频细节。GAN 强迫 predictor 在 perceptual space 输出 sharp 结果。Ablation 显示 GAN 贡献 1.2% semantic。

**GAN 细节**：
- Discriminator 用 spectral norm + hinge loss
- Feature extractor 直接用大模型 backbone，不另训 VGG
- 判别在 latent space 一步 denoise 后做，不在 velocity space

---

## 一个容易被忽略但很关键的设计：单 tensor cache

传统缓存方法每个 layer 都存 cache。TaylorSeer 还要存多阶导数。VRAM 爆炸。

**TaylorSeer 在 HunyuanVideo 上多占 33.5 GB VRAM。PAB 多占 22 GB。**

DisCa 只存一个 cache tensor，多占 **0.43 GB**。

**为什么能只存一个？** Predictor 是 learnable 的，有足够 capacity 从单 tensor + 当前状态解出所有需要的信息。传统方法是 training-free 的，必须靠"信息冗余"补偿。Learnable 的不需要。

**为什么这在实际部署时更重要？** HunyuanVideo 704p × 129 frames 需要开 sequence parallel（4 卡）。多卡并行下，多层 cache 会产生大量稀疏 memory access。底层 CUDA library 在并行场景下不优化稀疏访问，导致实际加速远低于理论。

Table 4 数据：
- TaylorSeer (N=6)：理论 9.09× vs 实际 6.96×，掉了 23%
- FORA (N=6)：理论 11.1× vs 实际 8.01×，掉了 28%
- **DisCa (N=4)：理论 11.9× vs 实际 11.8×，几乎完美**

DisCa 的单 tensor cache + dense predictor compute 完美匹配 GPU 并行范式。

---

## 整体 pipeline（三段式）

**Stage 1**：CFG distillation。原本推理要跑"有条件"和"无条件"两条分支（CFG），现在用小 FFN 把 CFG scale 编进 condition，单次前向搞定。2× 加速。

**Stage 2**：Restricted MeanFlow distillation。在 CFG distilled 模型上继续蒸馏，把 50 步压到 10-20 步。$\mathcal{R} = 0.2$。

**Stage 3**：Predictor 训练。在 Restricted MeanFlow 模型上训小 predictor + GAN。500 iter MSE warmup + 1000 iter GAN。

推理时：CFG distilled + Restricted MeanFlow + cache predictor，叠加三重加速。

---

## 最终效果

Table 2 主结果：

| 方法 | 加速 | Semantic | Quality | Total |
|---|---|---|---|---|
| 原始 50 步 | 1.00× | 73.5 | 81.5 | 79.9 |
| TaylorSeer (最强 baseline) | 6.96× | 63.7 (-13.3) | 79.9 (-2.0) | 76.7 (-4.0) |
| **DisCa (N=4)** | **11.8×** | 69.3 (-5.7) | 81.1 (-0.5) | 78.8 (-1.4) |

11.8× 加速下，semantic 只掉 5.7%，quality 几乎无损，total 只掉 1.4%。

对比一下：PAB 在 5.34× 时 semantic 就掉 27.3%，完全崩了。

**甚至 DisCa 在 7.56× 加速时 quality score 还比原始 50 步模型高（81.9 vs 81.5）。**

---

## 灵魂拷问几个点

**1. Restricted MeanFlow 的 $\mathcal{R} = 0.2$ 怎么定的？**

Paper 没给清晰的 selection criterion。试了 0.2 和 0.4，0.2 更稳就用 0.2。这个是不是 model-specific？换模型要不要重新调？不知道。

**2. Predictor 4% 大小，是不是还可以更小？**

Paper 没探索 predictor size 的 ablation。4% 是拍脑袋还是有理论依据？能否做到 1% 用 MoE 或 pruning？

**3. GAN 训练稳定性？**

Figure 5 只画了 1000 iter 的 loss curve。更长训练会不会 mode collapse？Predictor 这么小，判别器这么强，动态平衡能维持多久？

**4. 跨模型迁移性？**

Predictor 在 HunyuanVideo 上训的，换 Wan / CogVideoX 还能用吗？还是每个模型都要重新训？如果每个模型都要重训，部署成本其实不低。

**5. 为什么 GAN 在 latent space 而不是 pixel space？**

Paper 没解释清楚。ADD 在 image domain 是 pixel space。Video domain 用 latent space 是因为计算成本还是效果考量？

**6. 推理时间 breakdown？**

Predictor 占总推理时间多少？Cache hit ratio 多少？Paper 没给。说"几乎可忽略"但没数字。

---

## 我的直觉判断

**Strong 的点**：
- "蒸馏让 trajectory 稀疏，稀疏 trajectory 上 training-free heuristic 失效，所以要 learnable" — 这个 insight 很 sharp，直击问题本质
- Restricted MeanFlow 简单有效，pruning 大跨度训练样本 = implicit curriculum，思路干净
- Single-tensor cache 在 sequence parallel 下的优势分析，是之前 caching 工作都忽略的工程细节，很 practical
- 11.8× 实际加速 + 理论完美匹配，工程执行到位

**Weak 的点**：
- 整体 pipeline 三段式，每段都要训练，总训练成本不低。论文没报总训练时间/GPU 小时
- "Learnable" 这个词有点 overclaim — predictor 还是得训练，不是 zero-shot 的
- Generalization 没验证（只 HunyuanVideo）
- GAN 训练虽然 stable 但没长跑实验
- $\mathcal{R}$ 超参敏感性没充分 ablation

**最值得 follow 的方向**：
- 把 Restricted MeanFlow 的 $\mathcal{R}$ 做成 curriculum schedule（从大到小 anneal），可能能推到更少步
- Predictor 跨模型迁移（meta-learning 或 architecture-agnostic design）
- End-to-end joint training（MeanFlow + predictor 同时训）
- 推到 image domain 验证（Flux / SD3）

---

## Reference Links

主 paper: https://arxiv.org/abs/2507.02705
MeanFlow: https://arxiv.org/abs/2505.13447
HunyuanVideo: https://arxiv.org/abs/2412.03603
TaylorSeer: https://arxiv.org/abs/2503.06123
ADD (Adversarial Diffusion Distillation): https://arxiv.org/abs/2311.17042
DeepCache: https://arxiv.org/abs/2312.00858
ToCa: https://arxiv.org/abs/2410.05340
TeaCache: https://arxiv.org/abs/2411.17783
PAB: https://arxiv.org/abs/2408.12588
FORA: https://arxiv.org/abs/2407.01425
Δ-DiT: https://arxiv.org/abs/2406.01125
FasterCache: https://arxiv.org/abs/2410.19355
AdaCache: https://arxiv.org/abs/2411.02397
L2C: https://arxiv.org/abs/2406.01733
Shortcut Models: https://arxiv.org/abs/2410.12557
Consistency Models: https://arxiv.org/abs/2303.01469
Flow Matching: https://arxiv.org/abs/2210.02747
VBench: https://arxiv.org/abs/2311.17982
Wan: https://arxiv.org/abs/2503.20314
CogVideoX: https://arxiv.org/abs/2408.07757

---

核心就一句话：**蒸馏让步子变大，步子大了缓存复用就失灵，所以用小神经网络学怎么从上一步预测下一步；同时蒸馏本身别太狠，只学小步走，别学一步登天。两者配合，12 倍加速质量几乎不掉。**

---

# DisCa: Distillation-Compatible Learnable Feature Caching for Video DiTs

你好 Andrej。这篇 paper 我来给你做一个 deep dive，重点放在 intuition building、公式细节和架构图解析上。

---

## 1. The Big Picture: 这篇文章在解决什么问题

Video Diffusion Transformer (DiT) 如 HunyuanVideo (540p, 129 frames, 5 sec) 单次推理 ~1155s，部署成本极高。加速有两条主流路径：

**Path A — Training-aware Step Distillation**: MeanFlow [14], consistency models, progressive distillation, DMD2。MeanFlow 是 Kaiming He 组的工作，把训练目标从 instant velocity $\vec v$ 换成 mean velocity $\vec u$，能做 one-step 生成。

**Path B — Training-free Feature Caching**: DeepCache, ToCa, TaylorSeer, TeaCache, FORA, PAB, Δ-DiT。这些方法 cache 上一 step 的 feature 直接 reuse 或者用 Taylor expansion forecast。

**核心 insight**: 两条路径独立工作都还行，**但二者结合时严重退化**。原因如 Figure 1 所示：
- 未 distill 模型相邻 timestep feature 高度相似（trajectory 密集），caching 有效。
- Distill 后 timestep 从 50 → 10，sampling trajectory 上点变得稀疏，velocity prediction 在 step 间差异巨大，简单的 linear / Taylor interpolation 无法 capture high-dim feature evolution。

DisCa 的两个 contribution 正是为此设计：
1. **Restricted MeanFlow** — 让 distillation 在 large video model 上稳定。
2. **Learnable Feature Caching** — 用 lightweight learnable predictor 替代手工 Taylor 公式，专门适配 distilled model。

最终在 HunyuanVideo 上做到 **11.8× 加速，semantic score 仅掉 5.7%，quality 几乎无损**。

---

## 2. Preliminary: 公式回顾与变量解释

### 2.1 Diffusion Forward (Eq. 1)

$$\mathcal{N}\left(x_{t-1}; \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right), \beta_t \mathbf{I}\right)$$

- $t$：timestep index
- $\beta_t$：noise variance schedule
- $\alpha_t = 1 - \beta_t$
- $\bar\alpha_t = \prod_{i=1}^{T} \alpha_i$：累积 product
- $\epsilon_\theta$：parameterized denoising network, input $x_t$ 和 $t$, 预测 noise
- $T$：总 timesteps（HunyuanVideo 默认 50）

### 2.2 Flow Matching

$$\mathbb{E}_{t, p_t(x)} ||v_t(x;\theta) - u_t(x)||^2$$

- $v_t(x;\theta)$：参数化 vector field (CNF)
- $u_t(x)$：target vector field 生成 probability path $p_t$
- 边界条件 $p_{t=0} = q_0$ (data), $p_{t=1} = q_1$ (noise)

### 2.3 MeanFlow (Eq. 2)

MeanFlow 核心公式：
$$(t-r)\vec u(r, t, x_t) = \int_r^t \vec v(\tau, x_\tau) \, d\tau$$

- $\vec u$：mean velocity（从 $r$ 到 $t$ 的平均）
- $\vec v$：instant velocity（即 model 预测的速度场）
- $r$：sampling 区间起点
- $t$：sampling 区间终点

经过 partial derivative 变换后优化目标：
$$\mathcal{L}(\theta) = \mathbb{E}\left\|u_\theta(x_t, r, t) - \text{sg}(u_{tgt})\right\|_2^2$$

$$u_{tgt} = v(x_t, t) - (t-r)\left(v(x_t,t)\partial_x u_\theta + \partial_t u_\theta\right)$$

- $u_\theta(x_t, r, t)$：MeanFlow model 预测的 mean velocity
- $u_{tgt}$：target mean velocity，包含 JVP (Jacobian-vector product) 项
- sg：stop-gradient
- $\partial_x u_\theta$：mean velocity 对 $x$ 的 Jacobian
- $\partial_t u_\theta$：mean velocity 对 $t$ 的偏导

**问题**: JVP 在 long sequence + 大 video model 上数值误差严重累积，导致 training divergence 和 generation artifacts。MeanFlow 原生目标是 one-step（$\mathcal{T} = t - r \in [0, 1]$），对 high-quality video generation 过于激进。

---

## 3. Restricted MeanFlow: 核心改进

### 3.1 Intuition

MeanFlow 训练时 sample $\mathcal{T} = t - r \in [0, 1]$。当 $\mathcal{T}$ 接近 1 时，相当于要求 model 一步从 noise 直接跳到 data — 这对 large video model 是过于 aggressive 的目标，会导致 catastrophic distortion。

**关键 idea**: 如果不再追求 one-step，只追求 multi-step（如 10 步），那么可以把 $\mathcal{T}$ 较大的部分 prune 掉，只学 local mean velocity。

### 3.2 公式 (Eq. 3)

$$\mathcal{T} = (t - r) \in [0, \mathcal{R}], \quad \mathcal{R} \in (0, 1)$$

- $\mathcal{T}$：sampling 区间长度
- $\mathcal{R}$：restrict factor，论文用 $\mathcal{R} = 0.2$ 和 $\mathcal{R} = 0.4$

具体采样（Algorithm 2）：
```
Sample T ~ U(0, R)
Sample t ~ U(0, 1)
Compute r = max(0, t - T)
```

### 3.3 训练流程 (Algorithm 2)

```
1. Sample T from U(0, R), t from U(0, 1), r = max(0, t-T)
2. Sample x_t = (1-t)·x_0 + t·ε  (noise-data interpolation)
3. v = M*(x_t, t)  // CFG distilled model 给 instant velocity
4. u, du/dt = jvp(M_θ, (x_t, r, t), (v, 0, 1))
5. u_tgt = v - (t-r)·du/dt
6. loss = ||u - stopgrad(u_tgt)||²
```

JVP 计算时 input tangent vector 是 $(v, 0, 1)$，对应 $(x_t, r, t)$ 三个输入的扰动方向。

### 3.4 实验数据 (Table 1)

| Method | Steps | Speed | Semantic (%) | Quality (%) | Total (%) |
|--------|-------|-------|--------------|-------------|-----------|
| Original (CFG) | 50 | 1.00× | 73.5 | 81.5 | 79.9 |
| CFG Distilled | 50 | 1.99× | 66.7 (-9.3) | 80.6 (-1.1) | 77.9 (-2.5) |
| MeanFlow | 20 | 4.96× | 66.6 (0.0) | 81.8 (0.0) | 78.8 (0.0) |
| **Restricted MF (R=0.4)** | 20 | 4.97× | 70.2 (+4.5) | 82.0 (+0.2) | 79.7 (+1.1) |
| **Restricted MF (R=0.2)** | 20 | 4.97× | 70.4 (+5.7) | 81.8 (0.0) | 79.5 (+0.9) |
| MeanFlow | 10 | 9.68× | 60.9 (0.0) | 80.6 (0.0) | 76.7 (0.0) |
| **Restricted MF (R=0.4)** | 10 | 9.69× | 67.6 (+11.0) | 81.3 (+0.9) | 78.6 (+2.5) |
| **Restricted MF (R=0.2)** | 10 | 9.68× | 68.2 (+12.0) | 81.3 (+0.9) | 78.7 (+2.9) |

**Key takeaway**: 在 10 steps（aggressive）场景下 Restricted MeanFlow 相对原 MeanFlow **semantic 提升 12%**。R=0.2 比 R=0.4 更保守更稳定。最终论文选 R=0.2 作为 DisCa base。

---

## 4. Learnable Feature Caching (DisCa): 核心创新

### 4.1 与 TaylorSeer 对比

**TaylorSeer** (cache-then-forecast):
$$\mathcal{F}_{pred, m}(x_{t-k}^l) = \mathcal{F}(x_t^l) + \sum_{i=1}^m \frac{\Delta^i \mathcal{F}(x_t^l)}{i! \cdot N^i} (-k)^i$$

- $\mathcal{F}(x_t^l)$：layer $l$ 在 timestep $t$ 的 feature
- $\Delta^i \mathcal{F}$：feature 的 $i$ 阶差分
- $N$：cache interval
- $k$：距离 cache 点的 step 数

**问题**: Taylor 是手工设计的低阶 polynomial，无法 capture high-dim feature evolution，尤其在 distilled model 上 trajectory 稀疏时完全失效。

**DisCa 方案**: 用 learnable neural predictor 替代 Taylor 公式，data-driven 学习 evolution。

### 4.2 Inference (Eq. 4, 5)

**Cache 初始化（完整 DiT 推理）**:
$$\mathcal{C}(x_{t_i}) = u(x_{t_i}, r_i, t_i) = \mathcal{M}_{\theta_M}(x_{t_i}, r_i, t_i, c_{t_i})$$

- $\mathcal{M}$：large-scale DM (Restricted MeanFlow distilled HunyuanVideo)
- $\theta_M$：DM 参数
- $c_{t_i}$：condition vector (text embedding)
- $u$：mean velocity 预测

**Cache step（后续 N-1 步用 predictor）**:
$$u(x_{t'}, t', r') = \mathcal{P}_{\theta_p}(\mathcal{C}, x_{t'}, r', t', c_{t'})$$

- $\mathcal{P}$：lightweight predictor
- $\theta_p$：predictor 参数
- $\mathcal{C}$：cache（单 tensor，不分层）
- $(t', r') \in \{(t_{i-1}, r_{i-1}), \ldots, (t_{i-(N-1)}, r_{i-(N-1)})\}$

### 4.3 Predictor 训练 (Eq. 6 + GAN)

**主目标 MSE**:
$$\mathcal{L}(\theta_p) = \mathbb{E}\left\|\mathcal{M}_{\theta_M}(x_{t'}, r', t') - \mathcal{P}_{\theta_p}(\mathcal{C}, x_{t'}, r', t')\right\|_2^2$$

- $x_{t'} = (1-t')x_0 + t'\epsilon$：从 noise-data 插值采样
- $(t', r') = (t - \Delta, r - \Delta)$，$\Delta$ 是小 timestep bias，表示 cache step 距 full computation step 的距离

**GAN loss (Eq. 7, 8)**:

Discriminator:
$$\mathcal{L}_D = \mathbb{E}\left[\max(0, 1 - \mathcal{D} \circ \mathcal{F} \circ \mathcal{M}_{\theta_M}(x_{t'}, r', t')) + \max(0, 1 + \mathcal{D} \circ \mathcal{F} \circ \mathcal{P}_{\theta_p}(\mathcal{C}, x_{t'}, r', t'))\right]$$

Predictor:
$$\mathcal{L}_P = \mathbb{E}\left[\|\mathcal{M}_{\theta_M} - \mathcal{P}_{\theta_p}\|_2^2 + \lambda \cdot \max(0, 1 - \mathcal{D} \circ \mathcal{F} \circ \mathcal{P}_{\theta_p})\right]$$

- $\mathcal{D}$：multi-scale discriminator (Spectral Normalization + Hinge Loss)
- $\mathcal{F}$：feature extractor（直接用 large DM backbone 作为 feature extractor）
- $\lambda = 1.0$：adversarial loss 权重
- Hinge loss 形式：real samples 鼓励 $\mathcal{D} \circ \mathcal{F}(real) > 1$，fake 鼓励 $< -1$

**Intuition**: MSE 单独训练 predictor 容易输出模糊、缺 high-freq detail。GAN 强迫 predictor 在 perceptual feature space 生成 sharp、semantically rich 的输出。这和 ADD (Adversarial Diffusion Distillation, Sauer et al. [55]) 的思路一致。

### 4.4 训练流程 (Algorithm 3)

```
1. Sample Δ ~ U(0, Δ_max)  // Δ_max = 0.2
2. (t', r') = max((0,0), (t-Δ, r-Δ))
3. x_t = (1-t)·x_0 + t·ε
4. C = M(x_t, r, t)  // 大 model 算 cache
5. x_{t'} = (1-t')·x_0 + t'·ε
6. u_pred = P_θp(C, x_{t'}, r', t')  // predictor 预测
7. u_tar = M(x_{t'}, r', t')  // 大 model 给 ground truth
8. x_{t''}^{pred} = x_{t'} - (t'-r')·u_pred  // 一步 denoise 后 latent
9. x_{t''}^{tar} = x_{t'} - (t'-r')·u_tar
10. L_P = ||u_pred - u_tar||² + λ·max(0, 1 - D∘F(x_{t''}^{pred}))
11. L_P.backward(); optimizer_P.step()
12. L_D = max(0, 1 - D∘F(x_{t''}^{pred})) + max(0, 1 + D∘F(x_{t''}^{tar}))
13. L_D.backward(); optimizer_D.step()
```

注意：GAN 在 latent space $x_{t''}$ 上做，不在 pixel space。先一步 denoise 再过 discriminator，让 discriminator 判断真实/虚假的"下一步 denoise 结果"。

### 4.5 Predictor 架构 (Figure 2c)

- **2 个 DiT Blocks 堆叠**
- 大小 < 4% 的 large model（ HunyuanVideo 是 20 Double-Stream + 40 Single-Stream layers，predictor 只有 2 个 block，参数量 < 4%）
- Decoder-only 架构（保持 DiT 的 robust processing capability）
- Input：cache tensor $\mathcal{C}$ + 当前 $x_{t'}$ + timestep embedding $(r', t')$ + condition $c_{t'}$
- Output：predicted mean velocity $u(x_{t'}, t', r')$

### 4.6 Memory-Efficient Cache: 关键设计

**传统 multi-layer cache (TaylorSeer, FORA, PAB)**:
- 每 layer 维护一个 cache tensor
- TaylorSeer 还要存 multi-order derivative tensors
- VRAM 占用爆炸式增长

**DisCa single-tensor cache**:
- 整个 model 只存 **一个 cache tensor** $\mathcal{C}$
- predictor 自身有足够 learning capacity，不需要结构化 cache 提供信息

Table 4 + Figure 6 数据：
| Method | Extra VRAM |
|--------|-----------|
| Original | 0 |
| TaylorSeer (N=6) | +33.49 GB |
| FORA | +27 GB |
| PAB | +22 GB |
| TeaCache | +0.49 GB |
| Δ-DiT | +0.45 GB |
| **DisCa** | **+0.43 GB** |

**为什么 single-layer cache 在 distributed parallel 下关键**: 多层 cache 在 sequence parallel 下产生 sparse memory access，底层 hardware library 在并行场景下不优化 sparse 访问，导致实际加速远低于理论。DisCa 的 single-tensor cache 几乎无 overhead，理论/实际加速完美匹配（Table 4：DisCa 11.9× 理论 vs 11.8× 实际）。

---

## 5. 整体 pipeline (Figure 2 全景)

DisCa 完整 pipeline 三阶段：

**Stage 1: CFG Distillation** (Algorithm 1)
- 用小 FFN 把 CFG scale 编码进 condition vector
- 学习 $v_{target} = g \cdot v_c + (1-g) \cdot v_{uc}$，$g \in [1.0, 8.0]$ random
- 2× 加速（无需单独 inference CFG/No-CFG 两条分支）

**Stage 2: Restricted MeanFlow Distillation** (Algorithm 2)
- 基于 CFG distilled model 继续 distill
- $\mathcal{R} = 0.2$
- 学习 local mean velocity
- 50 → 20 steps 或 50 → 10 steps

**Stage 3: DisCa Predictor Training** (Algorithm 3)
- 基于 Restricted MeanFlow distilled model
- 500 iter MSE warmup
- 1000 iter GAN training
- 学习率：predictor $10^{-4}$, discriminator $10^{-2}$
- $\Delta_{max} = 0.2$

**Inference 时序** (Figure 2a, Table 4):
- N=2: 13 步 → 7 DiT + 7 predictor
- N=3: 12 步 → 4 DiT + 8 predictor (实际 latency 130.7s)
- N=4: 12 步 → 3 DiT + 9 predictor (实际 latency 97.7s, 11.8× 加速)

具体地，N=4 时是 8 次 DiT inference + 12 次 predictor inference（从 supplementary 6.3 看到 N=4 是 "8, 11, 13 DiT inferences and corresponding 12, 9, 7 predictor inferences"，这个对应 N=4, 3, 2。等等，让我重新看：原句 "For N=4, 3, 2, the inference alternates between 8,11,13 DiT inferences and corresponding 12,9,7 predictor inferences" — 即 N=4 → 8 DiT + 12 predictor; N=3 → 11 DiT + 9 predictor; N=2 → 13 DiT + 7 predictor。总共 20 步 inference cycle。）

---

## 6. 实验：SOTA 性能数据

### 6.1 主结果 (Table 2)

| Method | Speed | VRAM | Semantic | Quality | Total |
|--------|-------|------|----------|---------|-------|
| Original (50 step) | 1.00× | 99.23GB | 73.5 | 81.5 | 79.9 |
| CFG Distilled (50 step) | 1.99× | 97.21GB | 66.7 (-9.3) | 80.6 (-1.1) | 77.9 (-2.5) |
| Δ-DiT (N=5) | 3.77× | 97.68GB | 60.0 (-18.4) | 76.7 (-5.9) | 73.3 (-8.3) |
| PAB (N=5) | 5.34× | 121.3GB | 53.4 (-27.3) | 73.1 (-10.3) | 69.2 (-13.4) |
| TeaCache (ℓ=0.15) | 5.00× | 97.70GB | 65.5 (-10.9) | 80.3 (-1.5) | 77.4 (-3.1) |
| FORA (N=3) | 4.35× | 124.6GB | 63.9 (-13.1) | 79.7 (-2.2) | 76.6 (-4.1) |
| TaylorSeer (N=3, O=1) | 4.31× | 130.7GB | 65.2 (-11.3) | 80.6 (-1.1) | 77.5 (-3.0) |
| Restricted MF (20 step) | 4.97× | 97.21GB | 70.4 (-4.2) | 81.8 (+0.4) | 79.5 (-0.5) |
| **DisCa (R=0.2, N=2)** | **7.56×** | 97.64GB | 70.8 (-3.7) | 81.9 (+0.5) | 79.7 (-0.3) |
| TaylorSeer (N=6, O=1) | 6.96× | 130.7GB | 63.7 (-13.3) | 79.9 (-2.0) | 76.7 (-4.0) |
| Restricted MF (9 step) | 10.7× | 97.21GB | 67.8 (-7.8) | 81.0 (-0.6) | 78.4 (-1.9) |
| **DisCa (R=0.2, N=3)** | **8.84×** | 97.64GB | 70.3 (-4.4) | 81.8 (+0.4) | 79.5 (-0.5) |
| **DisCa (R=0.2, N=4)** | **11.8×** | 97.64GB | 69.3 (-5.7) | 81.1 (-0.5) | 78.8 (-1.4) |

**Key observations**:
1. DisCa 在 7.56× 时甚至 **quality score 超过原始 50-step model** (+0.5%)，semantic 也仅 -3.7%
2. 11.8× 时仍优于所有 baseline 在 4-5× 时的表现
3. PAB 在 5.34× 时 semantic -27.3%，几乎完全崩溃
4. DisCa VRAM 仅比原始多 0.43GB，TaylorSeer 多 33GB

### 6.2 Ablation (Table 3)

| Restricted MF | Learnable Pred | GAN | Semantic | Quality | Total |
|---|---|---|---|---|---|
| ✓ | ✓ | ✓ | 69.3 | 81.1 | 78.7 |
| ✗ | ✓ | ✓ | 65.2 (-5.9) | 80.3 (-1.0) | 77.3 (-1.8) |
| ✓ | ✗ | ✓ | 67.3 (-2.9) | 80.5 (-0.7) | 77.9 (-1.0) |
| ✓ | ✓ | ✗ | 68.5 (-1.2) | 81.0 (-0.1) | 78.5 (-0.3) |

**Takeaway**:
- Restricted MeanFlow 贡献最大：5.9% semantic（说明 MeanFlow 在 video 上确实有 catastrophic 问题）
- Learnable predictor vs training-free caching：2.9% semantic
- GAN：1.2% semantic（但带来 sharpness）

### 6.3 Theory vs Practice (Table 4)

**DisCa (N=4)**: 理论 11.9× vs 实际 11.8× — 几乎完美匹配
**TaylorSeer (N=6)**: 理论 9.09× vs 实际 6.96× — 实际掉了 23%
**FORA (N=6)**: 理论 11.1× vs 实际 8.01× — 实际掉了 28%
**PAB (N=8)**: 理论 6.80× vs 实际 6.46× — 掉了 5%

**根因**: multi-layer cache 产生 sparse memory access，在 sequence parallel 下底层库不优化，导致大量隐藏 latency。DisCa 单 cache + dense predictor computation 完美匹配 GPU parallelism。

---

## 7. Key Intuitions 总结

### 7.1 为什么 distill + cache 之前不兼容

Trajectory sparsity problem。在 dense trajectory (50 steps) 下，相邻 timestep 间 feature $\mathcal{F}(x_t^l)$ 和 $\mathcal{F}(x_{t-1}^l)$ 几乎线性可插值，$\Delta \mathcal{F}$ 小。Distill 到 10 steps 后，相邻点距离扩大 5 倍，且 model 行为本身在 distilled 后更 "decisive"（一步跨更大距离），feature 不再 smooth 演化。Taylor 一阶/二阶展开不够，需要 learnable function approximator。

### 7.2 为什么 Restricted MeanFlow 必要

MeanFlow 的 JVP term $(t-r)(v \partial_x u_\theta + \partial_t u_\theta)$ 在 $\mathcal{T} \to 1$ 时被放大。在 large video model 上，单 sample 的 $x$ 维度极大（704×704×129 frames latent），Jacobian 数值误差累积巨大。Prune 掉 $\mathcal{T} > \mathcal{R}$ 的训练样本，等价于只在"容易学"的 local regime 训练，避开 numerical instability。这本质上是个 **curriculum restriction**。

### 7.3 为什么 cache 只存一个 tensor

Predictor 是 learnable 的，有能力从单 tensor + current $x_{t'}$ + timestep embedding 解出所有需要的演化信息。Multi-layer cache 是为 training-free heuristic 准备的"信息冗余"，learnable predictor 不需要。这也带来 sequence parallel 下的 dense compute 优势。

### 7.4 为什么 GAN 在 latent space 一步 denoise 后做

直接在 $u_{pred}$ 上判别意义有限。论文先一步 denoise 得到 $x_{t''}^{pred} = x_{t'} - (t'-r') u_{pred}$，再过 discriminator，相当于让 discriminator 看真实/虚假的"下一步 latent state"。这避免了在 velocity space 判别的 difficulty，更接近 image-space perceptual loss 的 spirit。Feature extractor $\mathcal{F}$ 直接复用 pretrained large DM backbone（不用额外 VGG 或 CLIP），节省训练成本且 feature 分布自然匹配。

---

## 8. 相关工作和延伸联想

### 8.1 MeanFlow 家族
- MeanFlow [Geng, Deng, Bai, Kolter, He. ArXiv:2505.13447, 2025] https://arxiv.org/abs/2505.13447
- Shortcut Models [Frans, Hafner, Levine, Abbeel. ArXiv:2410.12557] https://arxiv.org/abs/2410.12557
- Consistency Models [Song, Dhariwal, Chen, Sutskever. ICML 2023] https://arxiv.org/abs/2303.01469
- iCT (improved Consistency Training) [Lu, Song. ICLR 2025] https://arxiv.org/abs/2310.14189
- DMD2 [Yin et al. CVPR 2024] https://arxiv.org/abs/2405.14867
- Distribution Matching Distillation [Yin et al. CVPR 2024]

### 8.2 Feature Caching 家族
- DeepCache [Ma, Fang, Wang. CVPR 2024] https://arxiv.org/abs/2312.00858
- FORA [Selvaraju et al. ArXiv:2407.01425] https://arxiv.org/abs/2407.01425
- Δ-DiT [Chen et al. ArXiv:2406.01125] https://arxiv.org/abs/2406.01125
- TaylorSeer [Liu, Zou, Lyu, Chen, Zhang. ICCV 2025] https://arxiv.org/abs/2503.06123 (估)
- ToCa [Zou et al. ICLR 2025] https://arxiv.org/abs/2410.05340
- TeaCache [Liu et al. ArXiv:2411.17783] https://arxiv.org/abs/2411.17783
- PAB [Zhao et al. ArXiv:2408.12588] https://arxiv.org/abs/2408.12588
- FasterCache [Lv et al. ICLR 2025] https://arxiv.org/abs/2410.19355
- AdaCache [Kahatapitiya et al. ArXiv:2411.02397] https://arxiv.org/abs/2411.02397
- L2C [Ma et al. ArXiv:2406.01733] https://arxiv.org/abs/2406.01733
- MagCache [Ma et al. ArXiv:2506.09045] https://arxiv.org/abs/2506.09045
- SpeCa [Liu et al. ACM MM 2025]

### 8.3 Video Diffusion 大模型
- HunyuanVideo [Kong et al. ArXiv:2412.03603] https://arxiv.org/abs/2412.03603
- Wan [Alibaba Wan Team] https://arxiv.org/abs/2503.20314
- CogVideoX [Yang et al. ICLR 2025] https://arxiv.org/abs/2408.07757
- Sora (OpenAI technical report)
- Stable Video Diffusion [Blattmann et al. ArXiv:2311.15127] https://arxiv.org/abs/2311.15127

### 8.4 Adversarial Distillation
- ADD (Adversarial Diffusion Distillation) [Sauer et al. ECCV 2023] https://arxiv.org/abs/2311.17042
- ADD 主要在 image domain，DisCa 借鉴到 video + caching 场景

### 8.5 Flow Matching
- Flow Matching [Lipman et al. ICLR 2023] https://arxiv.org/abs/2210.02747
- Rectified Flow [Liu, Gong, Liu. ICLR 2023] https://arxiv.org/abs/2209.03003
- SD3 / Flux 用 flow matching backbone

### 8.6 VBench
- VBench [Huang et al. ArXiv:2311.17982] https://arxiv.org/abs/2311.17982
- 16 sub-dimensions，9 个 semantic + 7 个 quality
- Semantic : Quality = 1 : 4 weighting
- 论文批判 Quality score 对 malformation/blurring 不敏感，主张 Semantic 作为主指标

---

## 9. 可能的 follow-up 方向

1. **Predictor 进一步缩小**: 目前 4% 大小，能否做到 <1%？用 transformer 蒸馏 + pruning？MoE predictor？
2. **One-step DisCa**: 当前 Restricted MF 仍需 9-20 步。能否逐步 reduce $\mathcal{R}$ schedule（curriculum），最终 push 到 1-2 step？
3. **跨模型迁移**: predictor 能否在 HunyuanVideo 上训完后迁移到 Wan / CogVideoX？需要 architectural adaptation。
4. **Cache predictor 联合优化**: 当前是 sequential（先 distill MeanFlow，再训 predictor），能否 end-to-end joint training？
5. **High-order learnable predictor**: 是否需要 multi-step predictor（类似 multi-step Taylor）？目前 N=4 时是单步预测，会不会有累积误差？
6. **CFG distillation 改进**: 用更 sophisticated 的 CFG distillation（如 CFG-distill + adversarial）能否推到更小 steps？
7. **Long video generation**: paper 用 129 frames / 5 sec，能否扩展到 minute-level？sequence parallel + DisCa 可能是关键。
8. **Audio/Text diffusion**: DisCa 思路在 F5-TTS / LLADA 上能否 work？discrete token diffusion 的 trajectory 是否也 sparse？
9. **Memory bandwidth bottleneck 分析**: paper 提到 sparse memory access 在 sequence parallel 下不被优化，能否设计 hardware-aware cache 架构？
10. **Adversarial 在 latent space 的理论分析**: 为什么不在 pixel space 做？是否会有 mode collapse 风险？predictor 4% size 时 GAN 训练动态稳定性如何？

---

## 10. Critical 评价

**优点**:
- Insight 很 sharp：trajectory sparsity → caching 失效，learnable predictor 是 natural solution
- Restricted MeanFlow 是简单但有效的 trick，pruning $\mathcal{T} > \mathcal{R}$ 等价于 implicit curriculum
- Single-tensor cache 在 sequence parallel 下的优势分析到位
- 实验完整，对比 baseline 充分，包括 VRAM 和 theory-practice gap 分析
- 11.8× 实际加速且 semantic 仅 -5.7% 是 SOTA

**潜在 concern**:
- Predictor 仍需训练（非 training-free），增加了部署 pipeline 复杂度
- Restricted MeanFlow 选 $\mathcal{R} = 0.2$ 是怎么定的？是否 model-specific？
- Predictor 训练用 GAN，是否引入 mode collapse / 训练不稳风险？Figure 5 只显示 1000 iter，更长训练如何？
- 没有给 inference time breakdown：predictor 占多少？cache hit 占多少？
- HunyuanVideo 上验证，能否 generalize 到 image generation（如 Flux, SD3）？
- Ablation 没拆 GAN 各 component（spectral norm vs hinge vs multi-scale D）的贡献
- $\Delta_{max} = 0.2$ 与 inference 时 N=4 (cache 4 步) 是否匹配？需要更细的 cache interval ablation

---

## 11. 相关 reference links

主 paper: https://arxiv.org/abs/2507.02705 (DisCa)

MeanFlow: https://arxiv.org/abs/2505.13447
HunyuanVideo: https://arxiv.org/abs/2412.03603
TaylorSeer: https://arxiv.org/abs/2503.06123
ADD: https://arxiv.org/abs/2311.17042
DeepCache: https://arxiv.org/abs/2312.00858
ToCa: https://arxiv.org/abs/2410.05340
TeaCache: https://arxiv.org/abs/2411.17783
PAB: https://arxiv.org/abs/2408.12588
FORA: https://arxiv.org/abs/2407.01425
Δ-DiT: https://arxiv.org/abs/2406.01125
FasterCache: https://arxiv.org/abs/2410.19355
AdaCache: https://arxiv.org/abs/2411.02397
L2C: https://arxiv.org/abs/2406.01733
Flow Matching: https://arxiv.org/abs/2210.02747
Shortcut Models: https://arxiv.org/abs/2410.12557
Consistency Models: https://arxiv.org/abs/2303.01469
iCT: https://arxiv.org/abs/2310.14189
DMD2: https://arxiv.org/abs/2405.14867
VBench: https://arxiv.org/abs/2311.17982
Wan: https://arxiv.org/abs/2503.20314
CogVideoX: https://arxiv.org/abs/2408.07757
Stable Video Diffusion: https://arxiv.org/abs/2311.15127
PixArt-α: https://arxiv.org/abs/2310.00626
PixArt-Σ: https://arxiv.org/abs/2403.04692
F5-TTS: https://arxiv.org/abs/2410.06885
LLADA: https://arxiv.org/abs/2502.09992
DiTFastAttn: https://arxiv.org/abs/2403.11107
DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
MagCache: https://arxiv.org/abs/2506.09045
EEdit: https://arxiv.org/abs/2506.09181 (估)
SnapFusion: https://arxiv.org/abs/2304.14501 (估)
Q-Diffusion: https://arxiv.org/abs/2302.04304

---

希望这个 deep dive 给你足够 detail 来 build intuition。核心 takeaway: **distillation 让 trajectory 变稀疏，training-free heuristic 在稀疏 trajectory 上失效，learnable predictor 是 natural upgrade；同时 distillation 本身在 large video model 上要 conservative（Restricted MeanFlow），二者协同 push 到 11.8×**。single-tensor cache 是 sequence parallel 下的关键 enabler，这点常被忽视。
