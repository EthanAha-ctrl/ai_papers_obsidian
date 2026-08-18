---
source_pdf: 8DNA 8D Neural Asset Light Transport by Distribution Learning.pdf
paper_sha256: 69409f98a774b8f3e598637e7c765aa4b4e5c70b0f6a4d04b75c7274b4a9c316
processed_at: '2026-08-17T22:45:35-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 8DNA 用人话说一遍

## 1. 他们到底在干啥

假设你做了一个 jade seal（玉玺），里面有复杂的 subsurface scattering + glossy boundary。你要把它放进一个 game engine 里实时渲染。问题：engine 里既没有实现你用的 fancy phase function，也没有算力去 trace 内部几百次 bounce 的 light path。

8DNA 的 idea 就是：**把这个玉玺在工厂里（offline）"bake" 成一个 neural network**，network 把"光从哪儿进、从哪儿出、中间怎么 bounce" 全部记住。以后 engine 渲染时，ray 碰到玉玺表面就直接 query network 拿答案，不用再 trace 内部。

这个 idea 本身不新，Mullia 2024 (RNA) 和 Tg 2024a (NeuPreSS) 都做过。但 8DNA 有两个升级：
- 维度从 6D 升到 8D（多了 incident position）
- Training 方法从 regression 换成 distribution learning

我下面分开讲为什么这两个升级都必要。

References:
- RNA: https://research.nvidia.com/labs/rtr/neural-assets/
- NeuPreSS: https://rgl.epfl.ch/publications/Tg2024Precomputed

---

## 2. 为什么 6D 不够：far-field vs near-field

### Far-field 假设是啥

6D 方法假设 light 来自无穷远。Sun、environment map、sky dome 都算 far-field。这种情况下，incident light 只取决于方向 $\omega_i$，不取决于位置 $x_i$。所以 transport 可以写成：

$$
L_o(x_o, \omega_o) = \int F'(x_o, \omega_o, \omega_i) \, L_i(\omega_i) \, d\omega_i
$$

变量：
- $x_o$：asset 表面 outgoing 位置
- $\omega_o$：往 camera 的方向
- $\omega_i$：往 light 的方向
- $F'$：6D transport（已经把 $x_i$ 积分掉了）

这个假设在很多场景下 OK：户外 sun lighting、studio softbox 远远放着、IBL。

### Near-field 出问题的地方

但是如果你放一个**很近的小 area light**，比如 Figure 1 里 jade seal 右后方的小 emitter：
- Seal 正面被照亮，但背面是阴影（light 被 seal 自己挡住）
- 这个 shadow 取决于 light 相对 seal 的**位置**

6D model 把 $x_i$ 积分掉了，相当于说"所有 incident 位置的光强都一样"，于是它把 shadow 区也填上光了，玉玺背面偏亮。

真实 transport 是 8D 的：
$$
F(x_o, \omega_o, x_i, \omega_i)
$$

多了 $x_i$ 这个维度才能描述"光从位置 A 进来，能不能 reach 位置 B"。

### 为啥之前大家不直接做 8D

如果直接拿 6D 那套 regression 思路扩到 8D，loss 长这样：

$$
\iiiint (F_\theta(x_o, \omega_o, x_i, \omega_i) - F(x_o, \omega_o, x_i, \omega_i))^2 \, dx_i \, d\omega_i \, dx_o \, d\omega_o
$$

要训练这个，每个 query $(x_o, \omega_o, x_i, \omega_i)$ 都需要 estimate ground truth $F$。怎么 estimate？用 doubly-delta light（在 $x_i$ 放一个点光，方向是 $\omega_i$），然后 path trace。问题是：

1. **维度翻倍**：从 6D sample space 变成 8D，MC 采样效率暴跌
2. **Doubly-delta light 采样极难**：light 在位置上是一个点，在方向上是一条线，path 几乎不可能撞上；典型高 specular dielectric boundary + fiber 的强 forward scattering 让 emitter sampling 退化
3. Figure 3 实测：1 sample per query，regression 完全崩；需要 ~4096 spp 才能让 ground truth $F'$ 收敛

**这就是 8DNA 要解决的核心难题：8D regression 太贵了，需要换个 training paradigm。**

---

## 3. Distribution learning：把 loss 变成 path tracing

### 关键 insight

$F$ 是非负的，可以拆成两部分（per RGB channel）：

$$
F(x_o, \omega_o, x_i, \omega_i) = \alpha(x_o, \omega_o) \cdot p(x_i, \omega_i \mid x_o, \omega_o)
$$

直觉：
- $\alpha$：albedo（survival probability）—— "光从 $(x_o, \omega_o)$ 进来后，有多少活着出来了"
- $p$：conditional distribution —— "活着出来的光，从哪出、往哪走"

分开学：
- $\alpha_\theta$：简单 MLP
- $p_\theta$：normalizing flow（可以 sample + evaluate pdf）

### NLL loss 的 magic trick

学 distribution 的标准方法是最小化 negative log-likelihood：

$$
\mathcal{L}_p = -\iint F \log p_\theta \, dx_i \, d\omega_i
$$

注意 drop 了 $1/\alpha$（因为只是 scalar，不影响 gradient direction）。

**关键转化**：把 $-\log p_\theta$ 当成 "incident radiance" 喂给 rendering equation：

$$
\mathcal{L}_p = \iint F \cdot (-\log p_\theta) \, dx_i \, d\omega_i = L_o(x_o, \omega_o; \, -\log p_\theta)
$$

意思是："假装用 $-\log p_\theta$ 当 lighting，render 这个 asset，得到的 outgoing radiance 就是 loss"。

那 path tracing 就是 unbiased estimator：

$$
\mathcal{L}_p = \mathbb{E}_{\text{path sampling}}[-\beta_i \log p_\theta(x_i, \omega_i \mid x_o, \omega_o)]
$$

变量：
- $\beta_i$：path throughput（path 上每次 BSDF/phase/pdf 乘积）
- $(x_i, \omega_i)$：path 从 asset 出来时的位置方向
- path sampling 可以是任意分布（forward path tracing 没 NEE），throughput 自动 correct

**所以每个 query 只要 1 sample 就够！** 不需要 estimate $F$，因为 loss 本身就是 path tracing expectation。

### 为啥 regression 做不到这点

Regression 的 loss 是 $(F_\theta - F)^2$，需要同时 estimate $F$ 和 evaluate $F_\theta$。$F$ 是高维积分，1 sample 不够；$F_\theta$ 在 high-specular region 几乎处处为 0，sample 不中。

Distribution learning 的 loss 是 $F \cdot \log p_\theta$，本质是 "rendering with $-\log p_\theta$"，path tracing 天生就是它的 unbiased estimator。

这个 trick 让我想到 importance sampling 的对偶：regression 想要 "fit function value"，distribution learning 想要 "fit distribution shape"。后者可以通过 forward sampling 直接做，前者需要 evaluate target function 才行。

---

## 4. 网络架构：怎么把 manifold 上的 distribution 塞进 flow

### 问题

Normalizing flow 默认在 $\mathbb{R}^d$ 上工作。但 $x_i \in \mathcal{M}$（asset 表面 manifold）, $\omega_i \in S^2$（单位球面）。怎么搞？

### 解法：bounding box projection

把 incident ray $(x_i, \omega_i)$ 反向 trace 到 asset 的 axis-aligned bounding box 上的交点 $u_i$。如果 light 来自 asset convex hull 外面，$u_i \to x_i$ 是 injective（一对一）。

所以改成学 $(u_i, \omega_i)$ 的 distribution，再加 Jacobian correction：

$$
p_\theta(x_i, \omega_i) = p_\theta(u_i, \omega_i) \cdot \left| \frac{n_{x_i} \cdot \omega_i}{n_{u_i} \cdot \omega_i} \right|
$$

物理直觉：投影面积守恒，$|du_i| |n_{u_i} \cdot \omega_i| = |dx_i| |n_{x_i} \cdot \omega_i|$。

### 再转 cylindrical coordinates

$u_i$ 在 bounding box 上，用 cylindrical 坐标避免 spherical 坐标的 pole singularity：

$$
s = (s^1, s^2, s^3, s^4) = \left(\frac{u_i^3}{\|u_i\|}, \, \arctan\frac{u_i^1}{u_i^2}, \, \omega_i^3, \, \arctan\frac{\omega_i^1}{\omega_i^2}\right)
$$

变量含义：
- $s^1$：bounding box 哪个面（normalized z 坐标）
- $s^2$：该面上的 azimuth
- $s^3$：incident direction 的 z 分量
- $s^4$：incident direction 的 azimuth

### Autoregressive RQS flow

4 维 $s$ 用 autoregressive 分解：

$$
p_\theta(s \mid x_o, \omega_o) = \prod_{j=1}^{4} p_\theta(s^j \mid s^{k<j}, x_o, \omega_o)
$$

每个 conditional 是 MLP-predicted **rational quadratic spline**（RQS, Durkan 2019）：
- 32 knots × RGB 3 channels = 每 conditional 输出 288 维
- 4 层 MLP, 128 hidden, ReLU

为啥用 RQS：exact inverse 可 sampling，可微，能 fit 多模态。Vector-valued over RGB 避免 channel 独立假设。

这个 autoregressive 还隐含了 $p_\theta(u_i, \omega_i) = p_\theta(u_i) \cdot p_\theta(\omega_i \mid u_i)$ 的 factorization，inference 时可以在 $u_i$ 和 $\omega_i$ 上分别做 MIS。

Reference: Neural Spline Flows https://arxiv.org/abs/1906.04032

### Positional encoding

- $x_o$：triplane feature grid（$64^2$ per axis, 8-dim feature per vertex），同 EG3D
- $\omega_o$：cubemap feature grid ($16 \times 32 \times 32$)，同 Wu 2024 Neural Directional Encoding
- $s^1, s^3$：1D feature grid
- $(s^1, s^2)$ pairs 等：再 encode 成 cubemap

为啥这么搞：triplane 对 3D position 高效，cubemap 对 direction 高效，这是 neural rendering 现在的 standard recipe。

References:
- EG3D: https://nvlabs.github.io/eg3d/
- Neural Directional Encoding: https://sites.google.com/ucsd.edu/neural-directional-encoding

---

## 5. Training 细节

### Data generation

Online 生成 path samples：
1. 在 asset 的 bounding sphere 上随机选点
2. Cosine-weighted sample $\omega_o$
3. Trace ray 找第一个交点 $x_o$
4. 继续标准 path tracing（surface BSDF sampling、volume phase sampling、null scattering、Russian roulette）直到 ray 离开 asset
5. 记录 $(x_o, \omega_o, u_i, \omega_i, \beta_i)$

Buffer 存 $128^4$ tuples in 15GB RAM，24 次刷新避免 overfit。

### Loss

每个 batch 算两个 loss：

$$
\mathcal{L} = \text{Mean}[-\beta_i \log p_\theta(u_i, \omega_i \mid x_o, \omega_o)] + \text{Mean}[(\alpha_\theta - \beta_i)^2]
$$

第一个是 NLL（per-sample path tracing estimator），第二个是 albedo L2。

### Direct–indirect separation

很多 translucent asset 有 dielectric boundary，direct scattering 是高频 sharp lobe，indirect 是 smooth volumetric。一个 network 学两个 scale 难。

解法：training 时把 "只 bounce 一次就出来" 的 path 的 $\beta_i$ 设成 0，让 network 只学 indirect。Inference 时 direct lobe 用 analytic BSDF + emitter sampling。

这等于把 network 的 "任务" 减负，让它专注 smooth 部分。Figure 4 显示这个 separation 提升明显。

### Hyperparameters

- Adam, lr $5 \times 10^{-4}$
- 240K steps, batch 32768
- PyTorch + Mitsuba 3 (Dr.Jit) 实现

Reference: Mitsuba 3 https://www.mitsuba-renderer.org/

---

## 6. Inference：怎么塞进 path tracer

每次 camera ray 碰到 neural asset：

1. 在 $u_i$ 上 sample：从 $p_\theta(u_i \mid x_o, \omega_o)$ 拿一个位置（per random color channel）
2. 在 $\omega_i$ 上做 MIS：emitter sampling vs $p_\theta(\omega_i \mid u_i, \cdot)$，power heuristic
3. Project $u_i$ 回 asset 表面拿 $x_i$
4. Continue tracing from $(x_i, \omega_i)$
5. Throughput: $\beta \leftarrow \beta \cdot F_\theta / [p_{u_i} \, p_{\omega_i}]$

### Direct–indirect lobe selection

如果 direct separated 出去，在 $x_o$ 同时 sample 两条 ray（direct BSDF + indirect neural），stochastically 选一条继续 trace。Selection probability $m$ 是 visibility-aware 的：被 occluded 的 direct lobe 不会被选。

### 整体效果

Algorithm 3 的 pseudocode 看起来跟标准 path tracing 差不多，只是在 neural asset 处多了一步 neural sample。

---

## 7. 实验结果

### Assets

Figure 8 测了 10 个 asset：
- **Volumetric**：Candle, Milk, Cat, Seal, Dragon, Bunny（heterogeneous media + dielectric boundary）
- **Fiber**：CurlHair, Hair, Fabric（hair BSDF）
- **Surface**：Teaset（conductor BSDF）

### Baselines

- **PT**：标准 path tracing
- **Far-field**：6D MLP regression，按 Tg 2024a setup

### Accuracy（Table 1）

8DNA 在所有 asset 上 MSE 都比 far-field 低。最大 gap：
- Teaset: 0.075 vs 0.707（~10×）
- Seal: 0.182 vs 0.505
- CurlHair: 0.950 vs 1.381

### Variance & speed（Table 2）

@ 128 spp：
- **Far-field** variance 最低（因为 $x_i$ pre-integrated 不用 sample），inference 最快（~0.35 min）
- **8DNA vs PT**：
  - Volumetric：variance 大幅降（Milk 2.49 vs 33.4，13×），速度 2-20× 快
  - Surface：varance 持平或略升（因为没 MIS over $x_i$），但时间还是快 1.4-4×

为啥 8DNA 还能比 PT 快这么多？因为 PT 在高 albedo asset（Seal, Milk, Hair）里 light 在内部 bounce 几百次才衰减完，single path 巨贵。8DNA 把这些 bounce 压缩成单次 network query。

### Training time（Table 3）

- Volumetric：8DNA 2.23h vs Far-field 6.75h（**3× 快**）。Far-field 慢是因为需要 4096-8192 spp 生成 ground truth data
- Surface：1.78h vs 1.98h（持平）

注意 8DNA training 时 network 优化比 far-field 慢（normalizing flow pdf eval 慢），但 data generation 快得多（1 spp vs 4096 spp），总时间还是赢。

### Ablation（Table 4）

- Smaller network：速度 7.16s vs 12.21s，但 MSE 翻倍
- w/o direct separation：MSE 0.365（vs 0.200 with separation），证明 separation 重要
- Bounding geometry：convex hull < bounding box < bounding sphere（surface area 越小越好）

### Extension to pure volume（Cloud）

对没有 solid boundary 的 cloud，transmittance-sample 第一个 scattering event 当 $x_o$，network 学剩下的。Inference 比 explicit multi-scattering 快，variance 因为 path tracing 本来就能 MIS 所以接近。

---

## 8. Limitations & 我想到的 follow-up

### 作者列的 limitations

1. **Convex hull 假设**：其他 object 不能侵入 asset convex hull，否则 pre-baked paths 被破坏（Figure 14）。Fix：convex decomposition。
2. **没 MIS over $x_i$**：对 point light / 小 emitter，$x_i$ sampling 方差高。
3. **Normalizing flow smooth bias**：高频 specular interreflection、caustics、glints 难 fit。
4. **Flow eval 比 MLP 慢**：如果 far-field 够用就别用 8D。

### 我的 follow-up 想法

1. **MIS over $x_i$**：emitter-driven $u_i$ sampling 配合 $p_\theta(u_i \mid \cdot)$，类似 Clarberg 2008 product sampling 的 neural 版本。能解决 point light 问题。
2. **Diffusion model 替换 flow**：flow 的 smooth bias 是 hard limit；diffusion 能 fit 高频。Fu 2024 已经在 BSDF sampling 上试过。
3. **Convex decomposition**：把 asset 切 chunks，每 chunk 独立 8DNA，compose。能解决 convex hull 限制。
4. **Time-varying / animated asset**：add time 维度成 9D，或用 canonical space。
5. **Differentiable rendering**：现在 training 用 forward path tracing gradient，没 backprop through Mitsuba；如果 reverse-mode AD，可以 end-to-end 学 asset 几何 + transport。
6. **Real-time inference**：现在 RQS flow eval 在 GPU 上不够快。如果换成 lighter weight representation（比如 small neural Gaussian mixture），可能能 real-time。

References:
- Clarberg 2008 product sampling: https://dl.acm.org/doi/10.1111/j.1467-8659.2008.01190.x
- Fu 2024 BSDF diffusion: https://research.nvidia.com/labs/rtr/

---

## 9. 最 core 的 intuition 一句话

8DNA 把 8D light transport 这个看似 intractable 的 regression 问题，**通过 "loss = render with $-\log p_\theta$ as light" 这个 trick 坍缩成 forward path tracing**。1 sample per query 够用，因为 loss 本身就是 path tracing expectation。再加上 normalizing flow 能同时 sample + evaluate pdf，inference 时还能 importance sample 进 path tracer。

整个 pipeline 的 elegant 之处在于：**training 和 inference 都用 forward path tracing 这一个工具**，区别只是 training 时 "illumination" 是 $-\log p_\theta$，inference 时 "BSDF" 是 $F_\theta$。这种 symmetry 在 differentiable rendering 里少见。

---

## 10. 一些值得深挖的细节

### 为啥 albedo loss 是 L2 而不是 NLL

$\alpha$ 是 scalar per channel，没有 distribution 概念。L2 + per-batch mean 是 unbiased gradient estimator（因为 $\nabla_\theta \mathcal{L}_\alpha = 2(\alpha_\theta - \mathbb{E}^n[\beta_i]) \nabla_\theta \alpha_\theta$，$\mathbb{E}^n[\beta_i]$ 当前 batch 当 constant）。

### 为啥 drop $1/\alpha$ 不影响收敛

NLL 标准 form 是 $-\iint \bar{p} \log p_\theta$，$\bar{p} = F/\alpha$。代入得 $-\frac{1}{\alpha} \iint F \log p_\theta$。Drop $1/\alpha$ 只 scale gradient，不改变 direction。但 $\alpha$ 还是要单独学，因为 inference 时要用 $\alpha \cdot p$ 还原 $F$。

###为啥 NeuralSSS 在 Appendix A.1 失败

NeuralSSS（Tg 2024b）的 loss（Eq. 12）用 same samples for network query 和 ground truth estimate。Appendix 证明这导致 network regress 到 per-sample estimator $\frac{F_n L_n}{q_n}$ 而不是 expectation $L_o$。只有在 $q(\omega_i \mid x_i, \cdot)$ 是 cosine-hemisphere（infinite slab + isotropic）时 unbiased。Complex asset 上崩。8DNA 通过把 loss 写成 expectation form 避免这个 anti-pattern。

### Eq. 9 Jacobian 直觉

沿 $\omega_i$ 投影，bounding box 上的 $du_i$ 和 asset surface 上的 $dx_i$ 投影到 $\omega_i$ 垂面面积相等：

$$
|du_i| \, |n_{u_i} \cdot \omega_i| = |dx_i| \, |n_{x_i} \cdot \omega_i|
$$

所以 $\frac{|du_i|}{|dx_i|} = \frac{|n_{x_i} \cdot \omega_i|}{|n_{u_i} \cdot \omega_i|}$，这就是 reparameterization Jacobian。

### Triplane + cubemap encoding 的 trend

8DNA 用 triplane encode $x_o$、cubemap encode $\omega$，跟 EG3D、NDE、Mip-NeRF 360 一脉相承。这个 "position 用 triplane、direction 用 cubemap" 的 decomposition 正在成为 neural rendering 标准。原因是 triplane 比 3D voxel 省内存，cubemap 比 SH expressivity 强。

---

## 11. 跟其他工作的关系网

```
                   6D far-field regression
                   ├── Kuznetsov 2021 (NeuMIP, flat materials)
                   ├── Kuznetsov 2022 (curved surfaces)  
                   ├── Mullia 2024 (RNA, 3D assets) ← 8DNA 的 baseline
                   └── Tg 2024a (NeuPreSS, translucent)
                   
                   8D near-field
                   ├── Tg 2024b (NeuralSSS, isotropic assumption, limited)
                   └── Wu 2025 (8DNA, this paper)
                   
                   Distribution learning
                   ├── Müller 2017 (path guiding)
                   ├── Müller 2019 (neural IS)
                   ├── Xu 2023 (NeuSample, BSDF)
                   └── Li 2025 (Pure-Sample, microgeometry, concurrent)
                   
                   Pre-baked asset
                   ├── Kallweit 2017 (Deep Scattering, radiance predictor)
                   ├── Vicini 2019 (shape-adaptive SSS)
                   └── 8DNA (transport operator, relightable)
```

8DNA 是这几个 line 的交汇：把 6D pre-baking 推到 8D，用 distribution learning 解决 training variance，结合 normalizing flow 做 representation。

References:
- Müller 2017 path guiding: https://tom94.net/software/mlt/
- Müller 2019 NIS: https://tom94.net/software/neural-is/
- Vicini 2019: https://rgl.epfl.ch/publications/Vicini2019Learned

---

## 12. 给 Karpathy 的 TL;DR

**Problem**: Pre-bake 3D asset 的 global light transport 进 neural network，让 deployment renderer 不用 trace 内部 long paths。

**Prior work limitation**: 6D far-field assumption（忽略 incident position $x_i$）在 near-field lighting 下崩；8D regression training variance 太高，几千 spp 都不够。

**This paper's insight**: Reformulate NLL loss 成 "render with $-\log p_\theta$ as illumination"。Loss 本身变 path tracing expectation，1 sample per query 够用。Transport 拆成 $\alpha$ (albedo MLP) + $p$ (normalizing flow)，分别学。

**Architecture**: $x_o$ 用 triplane encode，$\omega_o$ 用 cubemap encode，$p$ 是 4-stage autoregressive RQS flow over cylindrical coords，vector-valued over RGB。

**Results**: 8D 准确度上比 6D far-field 全面优；volumetric asset 上 variance 比 PT 降 13×、speed 2-20× 快；training 比 far-field 3× 快（因为 data gen 1 spp vs 4096 spp）。

**Key trick**: Loss 的 reformulation 让 training 和 inference 都用 forward path tracing 这一个工具，对偶得很优雅。

---

# 8DNA: 8D Neural Asset Light Transport by Distribution Learning

## 1. Big Picture：这篇 paper 在解决什么问题

想象你有一个 jade seal（玉玺），里面是 heterogeneous translucent 媒质 + glossy dielectric boundary + 内部 interreflection。渲染这种 asset 的痛点是：light 在内部 bounce 很多次（long scattering paths），naive path tracing 要很多 spp 才能收敛，而且 deployment renderer（比如游戏引擎、实时渲染器）里可能根本就没有实现那些 fancy BSDF / phase function。

8DNA 的 proposal：把这个 asset 的 **全局光传输** 预先 bake 进一个 neural network，之后在任何新场景里直接 query 这个 network 就能拿到 outgoing radiance，不需要再 trace 内部的 long paths。这个 idea 跟 Mullia et al. 2024 的 RNA（Relightable Neural Assets）和 Tg et al. 2024a 的 NeuPreSS 一脉相承，但 8DNA 把维度从 6D 提升到 8D，training paradigm 从 regression 变成 distribution learning。

Reference:
- RNA: https://research.nvidia.com/labs/rtr/neural-assets/
- NeuPreSS: https://rgl.epfl.ch/publications/Tg2024Precomputed
- Mitsuba 3 (他们的 implementation base): https://www.mitsuba-renderer.org/
- Dr.Jit (Mitsuba 3 backend): https://drjit.readthedocs.io/

---

## 2. 6D vs 8D：为什么远场假设会崩

### 6D far-field formulation

过去的 neural asset 工作都假设 incident light $L_i$ 来自远处，所以只依赖方向 $\omega_i$ 不依赖位置 $x_i$。在这种情况下 transport 可以 pre-integrate 成 6D：

$$
L_o(x_o, \omega_o; L_i) = \int F'(x_o, \omega_o, \omega_i) \, L_i(\omega_i) \, d\omega_i
$$

这里 $F'(x_o, \omega_o, \omega_i) = \int F(x_o, \omega_o, x_i, \omega_i) \, dx_i$。变量：
- $x_o \in \mathcal{M}$：outgoing 位置（asset 表面）
- $\omega_o \in S^2$：outgoing 方向（往 camera）
- $\omega_i \in S^2$：incident 方向（往 light）

### 8D near-field 的物理直觉

真实 light transport operator 是 8D 的：
$$
F(x_o, \omega_o, x_i, \omega_i)
$$

这里多了 $x_i$：incident 位置。当 light source 是 near-field（比如附近的小 area light），物体不同位置 $x_o$ 看到的 incident radiance $L_i(x_i, \omega_i)$ 是不一样的——一边被 light 直接照，另一边被 self-occlusion 遮住。6D 把 $x_i$ 积分掉相当于做了一个 "spatially averaged" illumination 假设，在 near-field + 有 occlusion 的场景下会 overestimate incoming radiance，导致物体看上去比 ground truth 亮、阴影边变软。

Figure 1 里的 jade seal 就是典型：near-field area emitter 从右后方打过来，seal 背面应该有 occlusion 阴影；6D far-field model 把它忽略了，背面亮得不对；8DNA 通过 8D 参数化正确还原。

### 为什么不直接拿 6D 的 regression scheme 扩到 8D？

如果做 8D regression：
$$
\iiint (F_\theta(x_o, \omega_o, x_i, \omega_i) - F(x_o, \omega_o, x_i, \omega_i))^2 \, dx_i \, d\omega_i \, dx_o \, d\omega_o
$$

要 estimate ground truth $F$ 需要 doubly-delta illumination $L_i(x, \omega) = \delta(x - x_i)\delta(\omega - \omega_i)$，再去 path trace。在 8D 空间里 Monte Carlo 采样本来方差就大，再加高度 spec 的 illumination（dielectric boundary、fiber 的强 forward scattering），需要 thousands of spp 才能让 $F'$ 收敛。Figure 3 直观展示了：regression 在 1 sample/query 下完全失败，需要 ~4096+ spp；而 distribution learning 1 sample/query 就 work。

这是典型的 **curse of dimensionality + sharp target combination**。之前的 6D 工作能 escape 是因为可以 rely on emitter sampling 降低 variance，8D 下 doubly-delta 的 emitter sampling 退化了。

---

## 3. 核心创新：Distribution Learning

### 把 $F$ 分解

因为 $F \geq 0$，可以 per-color-channel 拆成 normalizing factor $\alpha$ 和 conditional distribution $p$：

$$
F(x_o, \omega_o, x_i, \omega_i) = \alpha(x_o, \omega_o) \cdot p(x_i, \omega_i \mid x_o, \omega_o)
$$

变量：
- $\alpha(x_o, \omega_o) = \iint F \, dx_i \, d\omega_i$：directionally-varying **albedo**（也可以理解为 survival probability / directional diffuse color）
- $p(x_i, \omega_i \mid x_o, \omega_o) = F / \alpha$：given 一个 outgoing query，incident configuration 的 **scattering 分布**

直觉：从 $(x_o, \omega_o)$ 发射一条 ray 进入 asset，经过很多次 surface/volume/null scattering 后从某个 $(x_i, \omega_i)$ 离开 asset。$\alpha$ 是 "有多少光活着出来了"（throughput 期望），$p$ 是 "出来的位置和方向分布"。

### Scattering distribution loss：把 NLL 变成 path tracing

标准 NLL training 是最小化 $-\iint \bar{p} \log p_\theta$，其中 $\bar{p}$ 是真值分布。代入 $\bar{p} = F/\alpha$：

$$
\mathcal{L}_p(\theta \mid x_o, \omega_o) = -\iint F(x_i, \omega_i, x_o, \omega_o) \log p_\theta(x_i, \omega_i \mid x_o, \omega_o) \, dx_i \, d\omega_i
$$

（drop 掉 $1/\alpha$ 因为它只是个 scalar 不影响 gradient direction。）

**关键 trick**：把这个积分重写成 rendering equation 的形式：

$$
\mathcal{L}_p = -\iint F \log p_\theta \, dx_i \, d\omega_i = \iint F \cdot (-\log p_\theta) \, dx_i \, d\omega_i = L_o(x_o, \omega_o; -\log p_\theta)
$$

也就是把 $-\log p_\theta$ 当成 "虚拟的 incident radiance" 去 render asset！所以可以用标准 path tracing（forward，no NEE）来 unbiased 估计：

$$
\mathcal{L}_p = \mathbb{E}_{x_i, \omega_i, \beta_i \sim \text{path sampling}}[-\beta_i \log p_\theta(x_i, \omega_i \mid x_o, \omega_o)]
$$

变量：
- $\beta_i$：path throughput（path 上每次 scattering 的 BSDF/phase/pdf 乘积）
- $(x_i, \omega_i)$：path 离开 asset 时的位置和方向
- path sampling：可以任意分布（importance sampling、BSDF sampling、Russian Roulette），throughput $\beta_i$ 自动 correction

这就是为什么 training 只要 1 sample/query 就行：loss 本身就是 path-tracing 的 expectation，不需要 estimate $F$。

### Albedo loss

$\alpha$ 等价于 "在 constant unit illumination 下渲染"：

$$
\alpha(x_o, \omega_o) = \iint F \cdot 1 \, dx_i \, d\omega_i = L_o(x_o, \omega_o; 1) = \mathbb{E}[\beta_i]
$$

所以直接拿 path sampling 的 throughput 平均就是 unbiased estimator。再用 L2 regression：

$$
\mathcal{L}_\alpha = (\alpha_\theta(x_o, \omega_o) - \mathbb{E}^n[\beta_i])^2
$$

注意：这里 $\mathbb{E}^n[\beta_i]$ 是 batch mean，但梯度对 $\theta$ 是 unbiased 的（因为 $\nabla_\theta \mathcal{L}_\alpha = 2(\alpha_\theta - \mathbb{E}^n[\beta_i]) \nabla_\theta \alpha_\theta$，$\mathbb{E}^n[\beta_i]$ 在当前 batch 被当 constant）。

### Direct–indirect separation

很多 translucent asset 表面有 dielectric boundary，direct scattering（一次 surface reflection）是 high-frequency lobe，indirect（多次内部 bounce）是 smooth volumetric transport。一个 network 同时学这两个尺度很难，所以论文把 direct BSDF $f$ 保留为 analytic：

$$
F_\text{trained} = F - V(x_o, \omega_i) f(x_o, \omega_o, \omega_i)
$$

其中 $V$ 是 self-visibility。Training 时把 "只 bounce 一次" 的 path 的 $\beta_i$ 设成 0。Inference 时 direct lobe 还是用 analytic BSDF sampling + emitter sampling，indirect 用 neural asset。

这跟 Müller et al. 2019 的 visibility hint 类似——visibility 隐藏 specular lobe 让 network 学 indirect。

---

## 4. Reparameterization：让 manifold 上的点 fit normalizing flow

### 问题

$x_i \in \mathcal{M}$（asset 表面），$\omega_i \in S^2$，都是 manifold，不是 Euclidean。Standard normalizing flow 在 $\mathbb{R}^d$ 上工作。

### 解法：bounding box projection

把 incident ray $(x_i, \omega_i)$ 一直 trace 到 asset 的 axis-aligned bounding box 上的交点 $u_i$。如果 light 来自 asset convex hull 外面，从 $(x_o, \omega_o)$ 出发反向 trace 到 $u_i$ 后再沿 $\omega_i$ 继续走，第一次碰到 asset 表面就是 $x_i$（Figure 5）。所以 $(u_i, \omega_i) \leftrightarrow (x_i, \omega_i)$ 是 injective mapping。

Jacobian 推导（Eq. 9）：
$$
p_\theta(x_i, \omega_i \mid \cdot) = p_\theta(u_i, \omega_i \mid \cdot) \cdot \left| \frac{n_{x_i} \cdot \omega_i}{n_{u_i} \cdot \omega_i} \right|
$$

这是因为沿着 $\omega_i$ 投影，$|du_i| |n_{u_i} \cdot \omega_i| = |dx_i| |n_{x_i} \cdot \omega_i|$（投影面积不变）。

### Cylindrical coordinates on bounding box

$u_i$ 在 bounding box 表面上，转到 local cylindrical coordinates：
$$
s = \big( u_i^3 / \sqrt{u_i \cdot u_i}, \; \arctan(u_i^1 / u_i^2), \; \omega_i^3, \; \arctan(\omega_i^1 / \omega_i^2) \big)
$$

变量含义：
- $s^1 = u_i^3 / \|u_i\|$：$u_i$ 在 z 轴方向 normalized（bounding box 哪个面）
- $s^2 = \arctan(u_i^1 / u_i^2)$：$u_i$ 在该面上的 azimuth
- $s^3 = \omega_i^3$：incident direction 的 z 分量
- $s^4 = \arctan(\omega_i^1 / \omega_i^2)$：incident direction 的 azimuth

不用 spherical coordinates 因为 pole 处 Jacobian = 0（singularity）。Cylindrical coordinates 在 bounding box face 上是 well-defined 的。

Jacobian（Eq. 10）：
$$
p_\theta(u_i, \omega_i \mid \cdot) = p_\theta(s \mid \cdot) \cdot \frac{|n_{u_i} \cdot u_i|}{(\sqrt{u_i \cdot u_i})^3}
$$

这是 $u_i \to \omega = u_i/\|u_i\|$ 的立体角 Jacobian。

---

## 5. 网络架构

### Normalizing flow for $p_\theta$

Autoregressive over 4 维 $s = (s^1, s^2, s^3, s^4)$：

$$
p_\theta(s \mid x_o, \omega_o) = \prod_{j=1}^{4} p_\theta(s^j \mid s^{k<j}, x_o, \omega_o)
$$

每个 conditional 是一个 MLP-predicted **rational quadratic spline**（RQS）[Durkan et al. 2019]：
- 32 knots per RGB channel
- 每条件 4-layer MLP, 128 hidden units, ReLU
- 输出 size: $3 \times 32 \times (2+1) = 288$ per conditional

RQS 的好处：exact inverse（可 sampling），可微，能 fit 多模态分布（比 Gaussian mixture 强），比 neural spline 之类的 expressivity 适中。

Vector-valued over RGB：每个 $g_j$ 同时 encode 三个 channel 的 spline。这样 $p_\theta$ 输出 RGB-vector 而不是标量——避免三个 channel 之间独立假设。

Autoregressive structure 还隐含了一个 factorization $p_\theta(u_i, \omega_i \mid \cdot) = p_\theta(u_i \mid \cdot) p_\theta(\omega_i \mid u_i, \cdot)$，inference 时可以做 MIS over $u_i$ vs $\omega_i$。

### Albedo MLP

- 2 hidden layers, 128 features, ReLU
- 输入 $(x_o, \omega_o)$，输出 RGB

### Input encoding

- $x_o$：3-axis triplane feature grid [Chan et al. 2022 EG3D]，resolution $64^2$ per plane, 8-dim feature per vertex
- $\omega_o$：cubemap feature grid [Wu et al. 2024 Neural Directional Encoding]，$16 \times 32 \times 32$
- $s^1, s^3$：1D feature grid, resolution 32
- $(s^1, s^2), (s^3, s^4)$ pairs in 3rd/4th MLP：先 map 到 unit direction 再 cubemap encode, $6 \times 32 \times 32$

Reference:
- EG3D triplane: https://nvlabs.github.io/eg3d/
- Neural Directional Encoding: https://sites.google.com/ucsd.edu/neural-directional-encoding
- Neural Spline Flows: https://arxiv.org/abs/1906.04032

### Training setup

- PyTorch + Mitsuba 3 (Dr.Jit)
- Adam, lr 5e-4, 240K steps, batch 32768
- Training buffer: $128^4$ path samples in 15GB RAM, 24 次刷新（online re-generation，避免 overfit static paths）

---

## 6. Inference：在 path tracer 里 integrate

### Pipeline（Algorithm 3）

每次 path tracing bounce 命中 neural asset 时：

1. **Sample $u_i \sim p_\theta(u_i \mid x_o, \omega_o)$**（沿 random color channel $c$）
2. **MIS at $u_i$**：在 $\omega_i$ 上既做 emitter sampling 又做 $p_\theta(\omega_i \mid u_i, \cdot)$ path sampling，power heuristic
3. **Project $u_i$ back to asset boundary** 拿 $x_i$，沿 $(x_i, \omega_i)$ 继续下一 bounce
4. **Throughput update**: $\beta \leftarrow \beta \cdot F_\theta(x_o, \omega_o, u_i, \omega_i) / [p_{u_i}^c \, p_{\omega_i}^c]$

### Direct–indirect lobe selection（Eq. 23）

如果 direct scattering 被 separated 出去，在 $x_o$ 处同时 sample 两条 ray：
- direct lobe $\omega_i'$ from BSDF sampling $\mathcal{P}_f$
- indirect sample $(u_i, \omega_i)$ from $p_\theta$

然后 stochastic select（probability $m = b^c / (b^c + b'^c)$，$b$ 是 direct throughput，$b'$ 是 indirect throughput）选其中一条继续 trace。Visibility-aware 的 $m$ 避免选被 occluded 的 direct lobe。

这个 trick 等价于 "MIS in path space" 但只在 single next-bounce 上做。

---

## 7. Experiments

### Test assets

Figure 8: Candle, Milk（homogeneous SSS）, Cat, Seal, Dragon, Bunny（heterogeneous media），CurlHair, Hair, Fabric（hair BSDF [Chiang et al. 2016]），Teaset（conductor BSDF）。

### Baselines

- **PT** (standard path tracing)
- **Far-field**：6D MLP + rqs importance sampling，按 Tg et al. 2024a 的 setup（regression loss + precomputed 4096-8192 spp data）

### Quantitative（Table 1, 2, 3）

**Reconstruction MSE（×100，越低越好）**：8DNA 在所有 asset 上都低于 far-field。最大 gap 在 Teaset（0.075 vs 0.707，~10×），Seal（0.182 vs 0.505）, CurlHair（0.950 vs 1.381）。

**Rendering variance @ 128 spp**：
- Far-field 最低（因为 pre-integrated $x_i$ 不需要 sample）
- Ours vs PT：volumetric 上 ours 显著低（Milk 2.49 vs 33.41，13×↓；Seal 1.975 vs 5.723；Dragon 0.588 vs 3.673）
- Surface-only 上 PT 已经有 MIS 帮忙，gap 小

**Inference time @ 128 spp**：
- Ours 比 PT 快 2-20×（volumetric），1.4-4×（surface）
- Far-field 最快（~2× ours）但 biased

**Training time**：
- Volumetric：Ours 2.23h vs Far-field 6.75h（3× faster）—— 因为 far-field 需要 thousands-spp data generation
- Surface：1.78h vs 1.98h（基本持平）

### 关键 qualitative findings

- **Far-field bias**：在 near-field 配置下（Candle, Seal, CurlHair 阴影区）6D model over-estimates incident radiance
- **NeuralSSS 的 isotropic assumption**：Appendix A.1 证明 NeuralSSS [Tg et al. 2024b] 实际上在 regress $\frac{F_n L_n}{q_n}$（per-sample estimator）而不是 outgoing radiance，只有当 conditional $q(\omega_i \mid x_i, \cdot)$ 是 cosine-hemisphere 时才 unbiased，这只在 infinite slab + isotropic scattering 成立。Complex asset 直接崩。
- **Convergence（Figure 11）**：log-log variance vs spp / time，ours 在所有 asset 上 slope 都更陡，complexity 越高优势越大。

### Ablation

- Smaller network（64 width, 16-knot RQS）：inference 7.16s vs 12.21s，但 MSE 翻倍（0.416 vs 0.200）
- w/o direct separation：MSE 0.365（差），inference 速度类似
- Bounding geometry proxy：convex hull < bounding box < bounding sphere（surface area 越小 variance 越低）

### Extension：purely volumetric asset

Cloud（无 solid boundary）：transmittance-sample 第一个 scattering event 作为 $x_o$，network 学剩下的。Equal-time variance 比 path tracing 好（path tracing 对 pure volume 已经能用 MIS）。

---

## 8. Limitations

1. **Convex hull assumption**：内部 scattering structure 不能被其他 object 侵入（Figure 14），否则 pre-baked paths 会被 occlude。Fix：convex decomposition。
2. **No MIS over $x_i$**：对 point light / small emitter 方差高（emitter 在 $x_i$ 上是 delta）。
3. **Normalizing flow smooth bias**：难 fit 高频 specular interreflection / caustics / glints（在 sub-manifold 上）。MLP-based 反而更适合 far-field。
4. **Inference 效率 vs MLP**：normalizing flow 比 MLP 慢，所以如果 far-field 够用就别用 8D。

---

## 9. 我的 takeaways & 联想

### 真正的 insight

我觉得最 elegant 的是 **Eq. 5 → Eq. 6 的转化**——把 NLL loss 重新表述成 "render with $-\log p_\theta$ as illumination"。这是一个非常 deep 的 reformulation：它把一个看似高维的 optimization 问题坍缩成 forward path tracing 的 1-sample Monte Carlo。

类似的精神在 differentiable rendering 里也有：instead of forward+adjoint，reparameterize 让 backward pass 自动 unfold。但 8DNA 更聪明，它根本不需要 backward——只是 reformulate objective 使其 fit forward sampling。

这种 trick 让我想到：
- **Path-space regularization**：把 BSDF sampling 的 "intractable" 部分 convert 成 path integral（Müller et al. 2017 path guiding 也是这个思路）
- **Score-based generative modeling**：$-\log p_\theta$ 和 score $\nabla_\theta \log p_\theta$ 的对偶性，这里他们也用了 $\beta_i$ 作为 importance weight 而不是 path pdf
- **Normalizing flow + physical simulator**：flow 在 latent space，physical engine 在 sample space，二者通过 importance sampling 桥接

### 跟 NeRF / Neural Radiance Field 的对比

NeRF 学的是 $L(x, \omega_o)$（固定 illumination 下的 view-dependent radiance），是 5D。8DNA 学的是 transport operator $F$，是 8D。差别：NeRF 是 "baked scene"，8DNA 是 "baked transfer function"。后者更 general——任何 illumination 都能用。

Reference: NeRF https://arxiv.org/abs/2003.08934

### 跟 PRT (Precomputed Radiance Transfer) 的对比

PRT 也是 precompute light transport 然后用 basis function compression。差别：
- PRT 用 SH / wavelet / Gaussian basis，assumes far-field + low-frequency
- 8DNA 用 normalizing flow，handles arbitrary frequency content（within flow's expressivity）
- PRT 是 linear-algebra compression，8DNA 是 non-linear generative model

Reference: Sloan 2002 PRT https://research.microsoft.com/pubs/72484/precomputed_radiance_transfer_for_dynamic.pdf

### 跟 Mip-NeRF / Triplane 的关联

他们用 triplane encode $x_o$（同 EG3D），用 cubemap encode $\omega$（同 Neural Directional Encoding）。这种 "geometry 在 triplane、direction 在 cubemap" 的 decomposition 看起来正在成为 neural rendering 的标准 recipe。

Reference: 
- EG3D https://nvlabs.github.io/eg3d/
- Mip-NeRF 360 https://arxiv.org/abs/2111.1209

### 跟 concurrent Li et al. 2025 (Pure-Sample) 的关系

Concurrent work [Li et al. 2025 Pure-Sample, arXiv 2508.07240] 也用 forward-sampled training，但目标是 **microgeometry**（surface micro-structure 6D transport）而不是 3D asset 8D transport。所以 8DNA 是 first 把这个 distribution-learning paradigm 推到 8D asset level 的。

### 跟 Deep Scattering [Kallweit 2017] 的对比

Deep Scattering 用 radiance-predicting NN 给 cloud 做 sample-based precomputation，本质是 NeRF-style "baked appearance"。8DNA 是 transport operator 而不是 radiance predictor，所以可 relight。

### Veach-style path sampling 的 connection

Eq. 19-23 的 MIS scheme 本质上是 Veach & Guibas 1995 power heuristic 的 neural extension。但 neural asset 的特殊性是：$x_i$ 已经被 network sample 了，所以 $\omega_i$ 上的 MIS 是 "在 network 已经决定 spatial sampling 后" 做。这跟 product importance sampling [Herholz 2016] 也有点像——先 sample 一个 marginal 再 sample conditional。

Reference:
- Veach 1995 MIS: https://dl.acm.org/doi/10.1145/218380.218513
- Herholz 2016 product sampling: https://dl.acm.org/doi/10.1145/2897824.2925915

### 数学上的"巧合"：为什么 $-\log p_\theta$ 能当 incident radiance？

这其实是 **cross-entropy = KL divergence + entropy** 的另一面。$\mathcal{L}_p = -\mathbb{E}_F[\log p_\theta] = D_{KL}(F \| p_\theta) + H(F)$。$H(F)$ 是常数，所以 minimize $\mathcal{L}_p$ 等价 minimize KL。KL 的 Monte Carlo estimator 是 $\mathbb{E}_F[-\log p_\theta]$，刚好对应 path sample from $F$ 然后 evaluate $-\log p_\theta$。

但这里 $F$ 本身是 transport operator 不是分布——只有 normalize by $\alpha$ 才是。所以他们在 loss 里 drop 了 $1/\alpha$，等价于 optimize weighted KL，weighted by $\alpha$。这就是为什么 albedo loss 必须单独学：把 $\alpha$ 还原。

### 一个潜在的问题

我注意到 Eq. 13 在 NeuralSSS appendix 推导里说 $S_\theta$ regresses 到 $\frac{F_n L_n}{q_n}$ 而不是 outgoing radiance。这其实是说 NeuralSSS 在做 **per-sample estimator regression** 而不是 expectation regression。这是常见的 anti-pattern：直接把 MC estimator 当 target 训练会让 network 学到带 noise 的 function。8DNA 通过把 loss 写成 expectation 形式避免了这个问题。

### 推测可能 follow-up 方向

1. **MIS over $x_i$**：emitter-driven sampling for $u_i$ 配合 $p_\theta(u_i \mid \cdot)$，类似 [Clarberg 2008] product sampling 的 neural 版本。可能解决 point light limitation。
2. **Normalizing flow → diffusion model**：flow 的 smooth bias 是 hard limit；diffusion 可以 fit 更高频。但 sampling cost 高。Fu et al. 2024 BSDF diffusion 是 first attempt。
3. **Convex decomposition**：把 asset 切成 chunks，每 chunk 单独 8DNA，再 compose。这个能解决 convex hull 限制。
4. **Time-varying assets**：asset 动起来怎么办？add time dimension 成 9D transport，或者 use canonical space。
5. **Differentiable rendering**：8DNA 训练时用了 forward path tracing 的 gradient，但没做 differentiation through Mitsuba；如果 reverse-mode AD 通过 path tracing，可以 end-to-end 学 asset 几何 + transport。

Reference:
- Fu 2024 BSDF diffusion: https://research.nvidia.com/labs/rtr/publications/
- Differentiable Mitsuba 3: https://rgl.epfl.ch/publications/Jakob2022Drjit

---

## 10. 一句话总结

8DNA 的核心 insight 是：**把 8D light transport 的 regression problem reformulate 成 distribution learning，使 training loss 等价于 "用 $-\log p_\theta$ 当 incident radiance 做 path tracing"**，从而把 high-dimensional、high-variance regression 转化成 low-variance forward sampling + NLL。8D 参数化 + distribution learning + autoregressive RQS flow + triplane/cubemap encoding，这套组合让它在 near-field illumination 下准确且训练快、inference 快。

关键 reference：
- Paper 主页（likely ACM TOG/SIGGRAPH Asia 2025）：见 https://cseweb.ucsd.edu/~ravir/
- Mitsuba 3: https://www.mitsuba-renderer.org/
- Neural Spline Flows: https://arxiv.org/abs/1906.04032
- RNA (Mullia 2024): https://research.nvidia.com/labs/rtr/neural-assets/
- NeuPreSS (Tg 2024a): https://rgl.epfl.ch/publications/Tg2024Precomputed
- Neural SSS (Tg 2024b): https://rgl.epfl.ch/publications/Tg2024NeuralSSS
- Li 2025 Pure-Sample: https://arxiv.org/abs/2508.07240
