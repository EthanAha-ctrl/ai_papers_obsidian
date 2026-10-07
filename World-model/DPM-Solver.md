---
source_pdf: DPM-Solver.pdf
paper_sha256: 4a14cb652bb4996c6bfdd717b0bf71d6a5c92cf9e09abdc3173106213f4fd671
processed_at: '2026-08-03T23:11:57-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 DPM-Solver

## 痛点在哪

Diffusion model sample 慢这件事大家都知道了。慢的本质其实挺简单的——你要从一团纯噪声出发，一步步往回走，走到 clean image。传统做法是 1000 步，每步都要跑一遍那个庞大的 U-Net，所以慢得离谱。

为什么非得 1000 步？因为大家一直把它当 SDE 来解。SDE 里有 Wiener process 的随机性，步子迈大了随机扰动就爆掉，高维空间里尤其严重。你可以想象成在浓雾里走山路，每步只能挪一小点，走快了就掉沟里。

后来 Score-SDE (Song et al., 2021, https://arxiv.org/abs/2011.13456) 证明：每个 diffusion SDE 都对应一个 probability flow ODE，marginal distribution 完全一样。这件事 opens the door——ODE 没有随机性，理论上可以迈大步。但是用 black-box ODE solver（比如 RK45）去解这个 ODE，在 ~10 步以内还是崩的，FID 动不动几十上百。60 步勉强能用，100 步以内都很难达到 1000 步 SDE 的质量。

所以问题就变成：**能不能设计一个专门的 ODE solver，让 diffusion ODE 在 10 步以内就 sample 出像样的图？**

---

## 最核心的发现：diffusion ODE 是 semi-linear 的

你把 diffusion ODE 写出来：

$$\frac{d\mathbf{x}_t}{dt} = f(t)\mathbf{x}_t + \frac{g^2(t)}{2\sigma_t}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$$

变量含义：
- $\mathbf{x}_t$ 是 time $t$ 的 latent state（从噪声往 clean 走的中间状态）
- $f(t) = \frac{d\log\alpha_t}{dt}$ 是 drift，关于 $t$ 的标量函数
- $g(t)$ 是 diffusion coefficient
- $\alpha_t, \sigma_t$ 是 noise schedule（决定每个时刻信噪比的函数）
- $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 是 U-Net，预测加进去的 noise

仔细一看右边有两项：
- $f(t)\mathbf{x}_t$：关于 $\mathbf{x}_t$ 是**线性**的，系数只跟 $t$ 有关
- $\frac{g^2(t)}{2\sigma_t}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$：非线性的，因为有 neural network

这就叫 **semi-linear ODE**。

为什么这件事重要？因为线性 ODE $\frac{d\mathbf{x}}{dt} = f(t)\mathbf{x}$ 是可以**解析求解**的，答案就是 $e^{\int f d\tau}\mathbf{x}_s$。你完全不需要 discretize 这部分。black-box RK solver 把线性项也一起 discretize，等于在白白积累误差。线性项的 exact solution 是指数函数，对步长敏感得要命，误差会被指数放大，所以 RK 在大步长下不稳定。

DPM-Solver 的第一个 insight 就是：**把线性部分解析地剥出来，只让 nonlinear 的 neural network 部分参与数值近似**。

这一步在 numerical ODE 文献里叫 variation of constants（https://en.wikipedia.org/wiki/Variation_of_parameters），是非常 classical 的技巧。解的形式是：

$$\mathbf{x}_t = e^{\int_s^t f(\tau)d\tau}\mathbf{x}_s + \int_s^t \left(e^{\int_\tau^t f(r)dr}\frac{g^2(\tau)}{2\sigma_\tau}\boldsymbol{\epsilon}_\theta(\mathbf{x}_\tau, \tau)\right)d\tau$$

第一项 $e^{\int f d\tau}\mathbf{x}_s$ 完全 known，没误差。但后面那个 integral 仍然 ugly，$f(\tau), g(\tau), \sigma_\tau$ 这些 noise schedule 系数和 neural network 耦合在一起，看起来还是难搞。

---

## 真正的 magic：换到 $\lambda$ 坐标系

paper 里第二个 insight 是真正的"aha moment"。定义：

$$\lambda_t := \log\frac{\alpha_t}{\sigma_t}$$

这就是 half-logSNR（信噪比的对数的一半）。因为 SNR $\alpha_t^2/\sigma_t^2$ 在 diffusion 中严格递减，$\lambda_t$ 也严格递减，所以可以反过来用 $\lambda$ 参数化时间。

现在做一个 change of variable。先算一下 $g^2(t)$：

$$g^2(t) = 2\sigma_t^2\left(\frac{d\log\sigma_t}{dt} - \frac{d\log\alpha_t}{dt}\right) = -2\sigma_t^2\frac{d\lambda_t}{dt}$$

代入那个 variation of constants 公式，并用 $\lambda$ 替换 $t$ 作为积分变量，所有 noise schedule 系数塌缩成 kernel $e^{-\lambda}$：

$$\mathbf{x}_t = \frac{\alpha_t}{\alpha_s}\mathbf{x}_s - \alpha_t\int_{\lambda_s}^{\lambda_t} e^{-\lambda}\hat{\boldsymbol{\epsilon}}_\theta(\hat{\mathbf{x}}_\lambda, \lambda)d\lambda$$

变量含义：
- $\hat{\mathbf{x}}_\lambda := \mathbf{x}_{t_\lambda(\lambda)}$，用 $\lambda$ 当时间参数化的 latent
- $\hat{\boldsymbol{\epsilon}}_\theta(\hat{\mathbf{x}}_\lambda, \lambda)$ 是以 $\lambda$ 为输入的 noise prediction model

这个公式美在哪里？

**第一**，$e^{-\lambda}$ 这个 kernel 与具体 noise schedule **完全无关**。无论你用 linear schedule、cosine schedule 还是别的什么，kernel 形式一模一样。这意味着 diffusion model 实质上是在 $\lambda$ 域上定义的，时间 $t$ 只是个 reparameterization 罢了。

**第二**，$e^{-\lambda}$ 有清晰的物理直觉。$\lambda$ 大（接近 $\lambda_0$，clean data 端），$e^{-\lambda}$ 小，意味着接近 clean 时 noise prediction 贡献被压很小；$\lambda$ 小（接近 $\lambda_T$，pure noise 端），$e^{-\lambda}$ 大，意味着早期 noise 的贡献被放大。正好对应去噪的物理直觉：噪声大的时候需要大刀阔斧地修，接近干净的时候只需要微调。

**第三**，现在所有 numerical approximation 工作只剩一件事：怎么用 few function evaluations 高阶近似 $\int e^{-\lambda}\hat{\boldsymbol{\epsilon}}_\theta d\lambda$。这正好是 exponential integrator 文献 (Hochbruck & Ostermann, 2010, https://doi.org/10.1017/S0962492910000048) 研究了几十年的问题。

---

## 怎么近似这个 integral——Taylor 展开登场

给定时间网格 $\{t_i\}_{i=0}^M$，定义 $h_i := \lambda_{t_i} - \lambda_{t_{i-1}}$（每一步在 $\lambda$ 域的跨度）。

把 $\hat{\boldsymbol{\epsilon}}_\theta(\hat{\mathbf{x}}_\lambda, \lambda)$ 在 $\lambda_{t_{i-1}}$ 处做 Taylor 展开：

$$\hat{\boldsymbol{\epsilon}}_\theta(\hat{\mathbf{x}}_\lambda, \lambda) = \sum_{n=0}^{k-1}\frac{(\lambda - \lambda_{t_{i-1}})^n}{n!}\hat{\boldsymbol{\epsilon}}_\theta^{(n)}(\hat{\mathbf{x}}_{\lambda_{t_{i-1}}}, \lambda_{t_{i-1}}) + \mathcal{O}((\lambda - \lambda_{t_{i-1}})^k)$$

变量说明：
- $\hat{\boldsymbol{\epsilon}}_\theta^{(n)}$ 是 $\hat{\boldsymbol{\epsilon}}_\theta$ 关于 $\lambda$ 的 $n$ 阶 **total derivative**（包含 $\hat{\mathbf{x}}_\lambda$ 对 $\lambda$ 的依赖）

代入 integral 后，需要计算形如 $\int e^{-\lambda}\frac{(\lambda - \lambda_{t_{i-1}})^n}{n!}d\lambda$ 的项。这个 integral 通过反复 integration by parts 可以解析算出来，paper 引入 exponential integrator 文献里的 $\varphi_k$ functions：

$$\varphi_k(z) := \int_0^1 e^{(1-\delta)z}\frac{\delta^{k-1}}{(k-1)!}d\delta, \quad \varphi_0(z) = e^z$$

closed form 长这样：
- $\varphi_1(h) = \frac{e^h - 1}{h}$
- $\varphi_2(h) = \frac{e^h - h - 1}{h^2}$
- $\varphi_3(h) = \frac{e^h - h^2/2 - h - 1}{h^3}$

general expansion (Eq. B.4)：

$$\mathbf{x}_t = \frac{\alpha_t}{\alpha_s}\mathbf{x}_s - \sigma_t\sum_{k=0}^{n}h^{k+1}\varphi_{k+1}(h)\hat{\boldsymbol{\epsilon}}_\theta^{(k)}(\hat{\mathbf{x}}_{\lambda_s}, \lambda_s) + \mathcal{O}(h^{n+2})$$

**所有 $\varphi$ 系数解析 known**，只剩 $\hat{\boldsymbol{\epsilon}}_\theta^{(k)}$ 需要数值近似。

---

## DPM-Solver-1：一阶版本，惊喜是它就是 DDIM

取 $n=0$（零阶 Taylor），直接得到 update rule：

$$\tilde{\mathbf{x}}_{t_i} = \frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\tilde{\mathbf{x}}_{t_{i-1}} - \sigma_{t_i}(e^{h_i} - 1)\boldsymbol{\epsilon}_\theta(\tilde{\mathbf{x}}_{t_{i-1}}, t_{i-1})$$

这是 DPM-Solver-1。

现在来看 DDIM (Song et al., 2021, https://arxiv.org/abs/2010.02502) 的 update：

$$\tilde{\mathbf{x}}_{t_i} = \frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\tilde{\mathbf{x}}_{t_{i-1}} - \alpha_{t_i}\left(\frac{\sigma_{t_{i-1}}}{\alpha_{t_{i-1}}} - \frac{\sigma_{t_i}}{\alpha_{t_i}}\right)\boldsymbol{\epsilon}_\theta(\tilde{\mathbf{x}}_{t_{i-1}}, t_{i-1})$$

用 $\frac{\sigma_t}{\alpha_t} = e^{-\lambda_t}$ 代进去：

$$\alpha_{t_i}\left(\frac{\sigma_{t_{i-1}}}{\alpha_{t_{i-1}}} - \frac{\sigma_{t_i}}{\alpha_{t_i}}\right) = \alpha_{t_i}(e^{-\lambda_{t_{i-1}}} - e^{-\lambda_{t_i}}) = \sigma_{t_i}(e^{h_i} - 1)$$

**DDIM 完全等于 DPM-Solver-1**。

这件事解释了一个 long-standing puzzle。之前 Salimans & Ho 在 progressive distillation paper 里也证明 DDIM 是 first-order discretization of diffusion ODE，但他们没法解释为什么 DDIM 比朴素的 Euler discretization 好那么多。现在清楚了——DDIM 隐式地利用了 semi-linear structure，线性部分被 $\frac{\alpha_t}{\alpha_s}$ 解析处理了，所以它的"实际有效阶数"比 naive Euler 高得多。

---

## DPM-Solver-2 和 -3：往上叠加高阶修正

二阶版本需要近似一阶 total derivative $\hat{\boldsymbol{\epsilon}}_\theta^{(1)}$。trick 是用一个中间点估计 derivative：

```
1. h_i = λ_{t_i} - λ_{t_{i-1}}
2. s_i = t_λ((λ_{t_{i-1}} + λ_{t_i})/2)   # 中点对应的时间
3. u_i = (α_{s_i}/α_{t_{i-1}})x̃_{t_{i-1}} - σ_{s_i}(e^{h_i/2} - 1)·ε_θ(x̃_{t_{i-1}}, t_{i-1})
4. x̃_{t_i} = (α_{t_i}/α_{t_{i-1}})x̃_{t_{i-1}} - σ_{t_i}(e^{h_i} - 1)·ε_θ(u_i, s_i)
```

intuition：
- Step 3 用 DPM-Solver-1 走半步到中点 $s_i$，拿到中间 latent $\mathbf{u}_i$
- Step 4 用中点处的 $\boldsymbol{\epsilon}_\theta(\mathbf{u}_i, s_i)$ 作为整步代表值，做一次大 update

这跟 explicit midpoint method (RK2) 长得很像，但 critical difference：linear part 用 $\frac{\alpha_t}{\alpha_s}$ 精确求解，只有 nonlinear $\boldsymbol{\epsilon}_\theta$ 部分用 midpoint 近似。

三阶版本需要近似到二阶 derivative，用两个中间点 $r_1 = 1/3$, $r_2 = 2/3$。Algorithm 2 看起来复杂，本质就是构造 $\mathbf{D}_{2i-1}$ 估计一阶 derivative、$\mathbf{D}_{2i}$ 估计二阶 derivative，最后在 update 里加 correction term。

Theorem 3.2 保证：$k$ 阶 DPM-Solver 的 global error 是 $\mathcal{O}(h_{\max}^k)$。

---

## 为什么 DPM-Solver 比 RK 好那么多

Table 1 的 ablation 数据非常 striking：

| Method | NFE=12 | NFE=18 | NFE=24 |
|--------|--------|--------|--------|
| RK2 (t) | 16.40 | 7.25 | 3.90 |
| RK2 (λ) | 107.81 | 42.04 | 17.71 |
| **DPM-Solver-2** | **5.28** | **3.43** | **3.02** |
| RK3 (t) | 48.75 | 21.86 | 10.90 |
| RK3 (λ) | 34.29 | 4.90 | 3.50 |
| **DPM-Solver-3** | **6.03** | **2.90** | **2.75** |

注意 RK2 (λ) 在 NFE=12 时 FID 是 107.81，惨不忍睹。DPM-Solver-2 也在 $\lambda$ 域操作，FID 只有 5.28。这说明**仅仅 change-of-variable 到 $\lambda$ 是不够的，关键是把 linear 部分解析化**。

intuition 是这样的：在 few-step regime，linear term 的 discretization error dominates。RK 对 $e^{\int f d\tau}\mathbf{x}$ 做有限差分，指数函数的差分误差随步长指数放大；DPM-Solver 直接写 $\frac{\alpha_t}{\alpha_s}$ exact，这部分误差为零。剩下的 nonlinear $\boldsymbol{\epsilon}_\theta$ 部分对步长不那么敏感，可以放心用 Taylor 展开做高阶近似。

这件事在 exponential integrator 文献 (Hochbruck & Ostermann, 2005, https://epubs.siam.org/doi/10.1137/S0036142903405897) 里是 well-known：对 semi-linear ODE，explicit RK 在大步长不稳定，exponential integrators 显著更 robust。DPM-Solver 本质上是把这个 classical numerical method 引入 diffusion model。

---

## 实际怎么用：时间步 schedule 与离散模型适配

### 时间步选 uniform in $\lambda$ 而不是 $t$

因为 solution 在 $\lambda$ 域里 kernel 是 $e^{-\lambda}$，$\boldsymbol{\epsilon}_\theta$ 关于 $\lambda$ 的变化比关于 $t$ 更平滑，所以 $\lambda$-uniform 比 $t$-uniform 更高效。

$$\lambda_{t_i} = \lambda_T + \frac{i}{M}(\lambda_0 - \lambda_T), \quad i = 0, \ldots, M$$

### 适配 discrete-time DPMs

DDPM (Ho et al., 2020, https://arxiv.org/abs/2006.11239) 在离散时间 $t_n = nT/N$ 训练，$N=1000$ 或 4000。model 输入是整数 index。DPM-Solver 通过 reparameterization 把它转成 continuous：

- **Type-1**: $\boldsymbol{\epsilon}_\theta(\mathbf{x}, t) = \tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}, 1000\cdot\max(t - T/N, 0))$
- **Type-2**: $\boldsymbol{\epsilon}_\theta(\mathbf{x}, t) = \tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}, 1000\cdot(N-1)t/(NT))$

实验发现 NFE 小时 Type-1 + $\epsilon = 10^{-3}$ 好；NFE 大时 Type-2 + $\epsilon = 10^{-4}$ 好。

### NFE budget 用完的小 trick

为了让 NFE 完全用完（不被 $\lfloor K/3 \rfloor$ 浪费），paper 提出：先尽量用 DPM-Solver-3，剩余 steps 用 DPM-Solver-1 或 -2 补齐。具体见 Appendix D.3。

---

## 实验结果速览

### CIFAR-10 (continuous-time, VP deep)

- **NFE=10**: FID **4.70** vs 19.55 (RK45, NFE=26) vs 82.42 (Improved Euler, NFE=48)
- **NFE=20**: FID **2.87**
- 1000 步 Euler-Maruyama SDE: FID 2.44

10 步达到接近 1000 步 SDE 的质量，~100× 加速。

### 各 dataset 对比

- CIFAR-10: DPM-Solver 12 步 FID 4.65 vs Analytic-DDIM 12 步 11.68
- CelebA 64×64: DPM-Solver 12 步 FID 4.20 vs DDIM 12 步 9.99
- ImageNet 64×64 (cosine schedule): DPM-Solver 12 步 FID 20.03 vs DDIM 12 步 52.69
- ImageNet 128×128 (classifier guidance): DPM-Solver 12 步 FID 5.84 vs DDIM 12 步 9.38
- LSUN bedroom 256×256: DPM-Solver 12 步 FID 4.21 vs DDIM 12 步 7.51

Runtime 方面，Table 7 显示同 NFE 下 DPM-Solver 和 DDIM 几乎一样快（implementation 优化后甚至略快），NFE 减少直接 translate 到 wall-clock 加速。

---

## Appendix A 的 unifying viewpoint

paper Appendix A.1 给了一个特别 elegant 的结论。对 VP-type DPMs ($\alpha_t^2 + \sigma_t^2 = 1$)，可以解析求出：

$$\alpha_t = \sqrt{\frac{1}{1 + e^{-2\lambda_t}}}, \quad \sigma_t = \sqrt{\frac{1}{1 + e^{2\lambda_t}}}$$

solution 在 $\lambda$ 域写成：

$$\hat{\mathbf{x}}_{\lambda_t} = \frac{\hat{\alpha}_{\lambda_t}}{\hat{\alpha}_{\lambda_s}}\hat{\mathbf{x}}_{\lambda_s} - \hat{\alpha}_{\lambda_t}\int_{\lambda_s}^{\lambda_t} e^{-\lambda}\hat{\boldsymbol{\epsilon}}_\theta(\hat{\mathbf{x}}_\lambda, \lambda)d\lambda$$

其中 $\hat{\alpha}_\lambda := \sqrt{\frac{1}{1+e^{-2\lambda}}}$。

**key point**：integrator $e^{-\lambda}\hat{\boldsymbol{\epsilon}}_\theta$ 只依赖 $\lambda$，不依赖具体 noise schedule。从 $\lambda_s$ 到 $\lambda_t$ 的 solution 完全由 $\lambda_s, \lambda_t, \hat{\boldsymbol{\epsilon}}_\theta$ 决定，与中间怎么 schedule 无关。

Appendix A.3 进一步指出 maximum likelihood training loss 在 $\lambda$ 域里也是 invariant：

$$D_{KL}(q_0 \| p_0) \leq D_{KL}(q_T\|p_T) + \int_{\lambda_T}^{\lambda_0}\mathbb{E}\left[\|\boldsymbol{\epsilon}_\theta(\hat{\mathbf{x}}_\lambda, \lambda) - \boldsymbol{\epsilon}\|^2\right]d\lambda + C$$

这等价于 Kingma et al. (Variational Diffusion Models, https://arxiv.org/abs/2107.00630) 和 Song et al. (https://arxiv.org/abs/2101.09258) 的 importance weighting trick。

**Big picture**: 训练和采样都 invariant to noise schedule。意味着 diffusion model 实质上是在 $\lambda$ 域上定义的，时间 $t$ 只是个 reparameterization 罢了。这暗示我们可以直接在 $\lambda$ 域定义 model，省去 ad-hoc noise schedule 设计。

---

## 整个故事最深刻的几点

1. **Semi-linear structure 是 diffusion ODE 的内在结构**。它来自 $f(t)\mathbf{x}_t$ 这个 pure scaling term，对应 $\alpha_t$ 的衰减。任何"信号 + 噪声"的 model 都会有这个 structure，diffusion 不是特例。

2. **$\lambda$ 域是 diffusion 的"自然坐标系"**。在 $\lambda$ 域里，noise schedule 被吸收成 universal kernel $e^{-\lambda}$，model 只学一个 mapping $\hat{\boldsymbol{\epsilon}}_\theta(\hat{\mathbf{x}}, \lambda)$，sampling 和 training 都在这里最自然。这有点像物理学里换坐标系——选对坐标系（symmetry-aligned）让方程最简洁。

3. **Exponential integrator 是把"已知的线性动力学"和"未知的非线性扰动"分离的标准工具**。DPM-Solver 的贡献是识别出 diffusion ODE 恰好是 semi-linear，把这个工具带过来。这件事在 numerical ODE 文献里 well-established，但 ML 社区之前完全没用上。

4. **DDIM 的"隐式高阶性"**：DDIM 看起来像个 heuristic deterministic sampler，实际上它是利用了 semi-linear structure 的 first-order exponential integrator。这解释了它比朴素 Euler discretization 显著好的原因。这个 equivalence 也是为什么 DPM-Solver 可以"无缝替换 DDIM"——它们用同一个 update structure，只是 DPM-Solver 又往上叠了 higher-order correction。

5. **Few-step regime 的工作原理**：在 NFE=10 时，linear term 用 $\frac{\alpha_t}{\alpha_s}$ exact 计算意味着信号 scaling 完全无误差，剩下的误差只来自 $\boldsymbol{\epsilon}_\theta$ 在 $\lambda$ 上的有限采样。Taylor 展开到 2-3 阶让 nonlinear 部分也有 $\mathcal{O}(h^3)$ 甚至 $\mathcal{O}(h^4)$ 的精度。$h_{\max} \approx (\lambda_0 - \lambda_T)/M$，10 步已经能让 FID 接近 asymptotic limit。

6. **Limitations**：DPM-Solver 解的是 ODE，丢掉了 SDE 的 stochasticity。在 likelihood evaluation 场景下不能用。10 步虽然快，相对 GAN 的单步 forward 仍慢几十倍，real-time 应用仍需 distillation 类方法。后续工作 DPM-Solver++ (https://arxiv.org/abs/2211.01095) 和 UniPC (https://arxiv.org/abs/2302.04867) 进一步改进了 converge order 和 stability。

7. **Connection 到 Flow Matching / Rectified Flow**：后来 Rectified Flow (Liu et al., 2023, https://arxiv.org/abs/2209.03003) 和 Flow Matching (Lipman et al., 2023, https://arxiv.org/abs/2210.02747) 把 $\lambda$ 域的 viewpoint 进一步推到极致——直接在数据到噪声的直线路径上做 simulation，1 步 ODE 就能 sample。DPM-Solver 的 $\lambda$-space formulation 实际上是这条思路的 early version：把"自然的进度变量"作为求解坐标。

---

## References

- DPM-Solver: https://arxiv.org/abs/2206.00927
- Code: https://github.com/LuChengTHU/dpm-solver
- Score-SDE (Song et al., 2021): https://arxiv.org/abs/2011.13456
- DDIM (Song et al., 2021): https://arxiv.org/abs/2010.02502
- DDPM (Ho et al., 2020): https://arxiv.org/abs/2006.11239
- Variational Diffusion Models (Kingma et al., 2021): https://arxiv.org/abs/2107.00630
- Exponential Integrators review (Hochbruck & Ostermann, 2010): https://doi.org/10.1017/S0962492910000048
- Explicit Exponential RK (Hochbruck & Ostermann, 2005): https://epubs.siam.org/doi/10.1137/S0036142903405897
- DPM-Solver++: https://arxiv.org/abs/2211.01095
- UniPC: https://arxiv.org/abs/2302.04867
- Rectified Flow: https://arxiv.org/abs/2209.03003
- Flow Matching: https://arxiv.org/abs/2210.02747
- Diffusion Beats GANs (Dhariwal & Nichol, 2021): https://arxiv.org/abs/2105.05233
- Improved DDPM (Nichol & Dhariwal, 2021): https://arxiv.org/abs/2102.09672

---

# DPM-Solver 详解：把 Diffusion ODE 当作 Semi-linear ODE 来求解

## 1. 背景与动机

Diffusion probabilistic models (DPMs) 生成质量极高，但 sample 速度极慢 —— 通常需要几百到上千步 neural network evaluations (NFE)。这件事的瓶颈在于：DPM 的 reverse process 一般被当作 SDE 或对应的 probability flow ODE 来解，而 black-box ODE solver（比如 RK45）在 few-step regime（~10 步）下误差极大，根本不收敛。

DPM-Solver (Lu et al., 2022, NeurIPS) 的核心 insight 极其简单但 powerful：**diffusion ODE 是 semi-linear 的**，线性部分可以解析求解，不要把它丢给 black-box solver 一起 discretize。再加上 change-of-variable 到 half-logSNR $\lambda$ 空间，整个 problem 简化为对 neural network 的 exponentially weighted integral 做近似。这个 formulation 让 10 步 sample 成为可能。

paper link: https://arxiv.org/abs/2206.00927
code: https://github.com/LuChengTHU/dpm-solver

---

## 2. Diffusion ODE 的 Semi-Linear Structure

从 Score-SDE (Song et al., 2021, https://arxiv.org/abs/2011.13456) 出发，diffusion ODE 写成：

$$\frac{d\mathbf{x}_t}{dt} = \mathbf{h}_\theta(\mathbf{x}_t, t) := f(t)\mathbf{x}_t + \frac{g^2(t)}{2\sigma_t}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$$

变量含义：
- $\mathbf{x}_t \in \mathbb{R}^D$：time $t$ 时的 latent state，$t \in [0, T]$，$T$ 是 forward process 终止时间
- $f(t) = \frac{d\log\alpha_t}{dt}$：drift coefficient，是标量函数
- $g(t)$：diffusion coefficient，由 noise schedule 决定
- $\alpha_t, \sigma_t \in \mathbb{R}^+$：noise schedule 函数，满足 $\alpha_t^2/\sigma_t^2$ 严格递减（SNR 单调下降）
- $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$：neural network，预测加在 $\mathbf{x}_0$ 上的 Gaussian noise

**关键观察**：右边由两项组成：
1. $f(t)\mathbf{x}_t$：关于 $\mathbf{x}_t$ 的**线性项**（系数只依赖 $t$）
2. $\frac{g^2(t)}{2\sigma_t}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$：非线性项，因为 $\boldsymbol{\epsilon}_\theta$ 是 neural network

这种形式在 ODE 文献中叫 **semi-linear ODE**。black-box RK solver 把整个 $\mathbf{h}_\theta$ 当作一个黑盒，对线性项也做 discretization，造成不必要的 error。线性项 $f(t)\mathbf{x}_t$ 的 exact solution 是指数形式 $e^{\int f d\tau}\mathbf{x}_s$，对步长特别敏感（误差随指数放大），所以 RK 在大步长时不稳定。

---

## 3. Exact Solution via Variation of Constants

对 semi-linear ODE $\frac{d\mathbf{x}}{dt} = A\mathbf{x} + N(\mathbf{x}, t)$，标准技巧是 variation of constants（https://en.wikipedia.org/wiki/Variation_of_parameters）：

$$\mathbf{x}_t = e^{\int_s^t f(\tau)d\tau}\mathbf{x}_s + \int_s^t \left(e^{\int_\tau^t f(r)dr}\frac{g^2(\tau)}{2\sigma_\tau}\boldsymbol{\epsilon}_\theta(\mathbf{x}_\tau, \tau)\right)d\tau$$

这个公式 (Eq. 3.1 in paper) 做的事是：**线性部分被解析地"剥离"了**。第一项 $e^{\int_s^t f d\tau}\mathbf{x}_s$ 完全 known，没有任何 discretization error。剩下只需要近似 nonlinear 部分的 integral。

但这个 integral 看起来仍然 ugly，因为 $f(\tau), g(\tau), \sigma_\tau$ 这些 noise schedule 系数和 $\boldsymbol{\epsilon}_\theta$ 耦合在一起。

---

## 4. Change-of-Variable 到 $\lambda$ 空间——这是整个 paper 的 magic

定义 **half-logSNR**：

$$\lambda_t := \log\frac{\alpha_t}{\sigma_t}$$

intuition：$\alpha_t^2/\sigma_t^2$ 是 SNR，$\lambda_t$ 是它的一半对数。由于 SNR 在 diffusion 中严格单调递减，$\lambda_t$ 也严格单调递减，所以存在反函数 $t_\lambda(\cdot)$，使得 $t = t_\lambda(\lambda(t))$。

**为什么这个变量特别自然？** 重新计算 $g^2(t)$：

$$g^2(t) = \frac{d\sigma_t^2}{dt} - 2\frac{d\log\alpha_t}{dt}\sigma_t^2 = 2\sigma_t^2\left(\frac{d\log\sigma_t}{dt} - \frac{d\log\alpha_t}{dt}\right) = -2\sigma_t^2\frac{d\lambda_t}{dt}$$

把 $f(t) = d\log\alpha_t/dt$ 和上式带入 Eq. (3.1)，并做变量替换 $t \to \lambda$，paper 得到 Proposition 3.1：

$$\boxed{\mathbf{x}_t = \frac{\alpha_t}{\alpha_s}\mathbf{x}_s - \alpha_t\int_{\lambda_s}^{\lambda_t} e^{-\lambda}\hat{\boldsymbol{\epsilon}}_\theta(\hat{\mathbf{x}}_\lambda, \lambda)d\lambda}$$

变量说明：
- $\hat{\mathbf{x}}_\lambda := \mathbf{x}_{t_\lambda(\lambda)}$：用 $\lambda$ 作为时间参数化的 latent state
- $\hat{\boldsymbol{\epsilon}}_\theta(\hat{\mathbf{x}}_\lambda, \lambda) := \boldsymbol{\epsilon}_\theta(\mathbf{x}_{t_\lambda(\lambda)}, t_\lambda(\lambda))$：以 $\lambda$ 为输入的 noise prediction

**intuition 详解**：

1. **线性项解析化**：$\frac{\alpha_t}{\alpha_s}\mathbf{x}_s$ 完全 known，对应"信号衰减/放大"的纯线性变换。
2. **noise schedule 被吸收进 $e^{-\lambda}$**：原本 integral 里复杂的 $f(\tau), g(\tau), \sigma_\tau$ 系数，在 $\lambda$ 域里全部塌缩成一个 universal kernel $e^{-\lambda}$。这个 kernel **与具体 noise schedule 无关**！无论 linear schedule 还是 cosine schedule，kernel 形式完全一致。这就是 paper Appendix A 强调的"invariance to noise schedule"。
3. **只需要近似 $\boldsymbol{\epsilon}_\theta$ 的 exponentially weighted integral**：剩下的工作就是如何用 few function evaluations 高阶近似 $\int e^{-\lambda}\hat{\boldsymbol{\epsilon}}_\theta\,d\lambda$。

这件事的几何 picture：在 $\lambda$ 域里，diffusion ODE 的 solution 是一个"指数衰减权重的 score 平均"。$\lambda$ 越大（接近 $\lambda_0$，即 clean data 端），$e^{-\lambda}$ 越小，意味着接近 clean 时 noise prediction 的贡献被压得很小；$\lambda$ 越小（接近 $\lambda_T$，即 pure noise 端），$e^{-\lambda}$ 越大，意味着早期 noise 的贡献被放大。这正好对应 diffusion 的物理直觉：在 noise 大的时候需要大刀阔斧去噪，在接近 clean 的时候只需要小修正。

---

## 5. Taylor 展开与 Exponential Integrators

给定时间网格 $\{t_i\}_{i=0}^M$（从 $t_0 = T$ 到 $t_M = 0$），定义 $h_i := \lambda_{t_i} - \lambda_{t_{i-1}}$（注意：因为 $\lambda$ 递减，所以 $h_i < 0$；paper 里取绝对值方向，$h_i$ 实际是负的，但公式中符号是兼容的）。

对 $\hat{\boldsymbol{\epsilon}}_\theta(\hat{\mathbf{x}}_\lambda, \lambda)$ 在 $\lambda_{t_{i-1}}$ 处做 Taylor 展开：

$$\hat{\boldsymbol{\epsilon}}_\theta(\hat{\mathbf{x}}_\lambda, \lambda) = \sum_{n=0}^{k-1}\frac{(\lambda - \lambda_{t_{i-1}})^n}{n!}\hat{\boldsymbol{\epsilon}}_\theta^{(n)}(\hat{\mathbf{x}}_{\lambda_{t_{i-1}}}, \lambda_{t_{i-1}}) + \mathcal{O}((\lambda - \lambda_{t_{i-1}})^k)$$

其中 $\hat{\boldsymbol{\epsilon}}_\theta^{(n)} := \frac{d^n\hat{\boldsymbol{\epsilon}}_\theta}{d\lambda^n}$ 是 $n$ 阶 total derivative（注意是 total derivative，包含 $\hat{\mathbf{x}}_\lambda$ 对 $\lambda$ 的依赖）。

代入 integral 后，需要计算

$$\int_{\lambda_{t_{i-1}}}^{\lambda_{t_i}} e^{-\lambda}\frac{(\lambda - \lambda_{t_{i-1}})^n}{n!}d\lambda$$

这个 integral 可以通过重复 integration by parts 解析计算。paper 引入 exponential integrator 文献里的 $\varphi_k$ functions (Hochbruck & Ostermann, 2010, https://doi.org/10.1017/S0962492910000048):

$$\varphi_k(z) := \int_0^1 e^{(1-\delta)z}\frac{\delta^{k-1}}{(k-1)!}d\delta, \quad \varphi_0(z) = e^z$$

满足 recurrence: $\varphi_{k+1}(z) = \frac{\varphi_k(z) - \varphi_k(0)}{z}$, $\varphi_k(0) = \frac{1}{k!}$

closed form:
- $\varphi_1(h) = \frac{e^h - 1}{h}$
- $\varphi_2(h) = \frac{e^h - h - 1}{h^2}$
- $\varphi_3(h) = \frac{e^h - h^2/2 - h - 1}{h^3}$

最终 general expansion (Eq. B.4)：

$$\mathbf{x}_t = \frac{\alpha_t}{\alpha_s}\mathbf{x}_s - \sigma_t\sum_{k=0}^{n}h^{k+1}\varphi_{k+1}(h)\hat{\boldsymbol{\epsilon}}_\theta^{(k)}(\hat{\mathbf{x}}_{\lambda_s}, \lambda_s) + \mathcal{O}(h^{n+2})$$

这里所有 $\varphi$ 系数都是解析的，**只剩 $\hat{\boldsymbol{\epsilon}}_\theta^{(k)}$ 需要数值近似**。

---

## 6. DPM-Solver-1：一阶版本（与 DDIM 等价！）

取 $n=0$：

$$\mathbf{x}_{t_i} = \frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\tilde{\mathbf{x}}_{t_{i-1}} - \sigma_{t_i}(e^{h_i} - 1)\boldsymbol{\epsilon}_\theta(\tilde{\mathbf{x}}_{t_{i-1}}, t_{i-1}) + \mathcal{O}(h_i^2)$$

drop 掉 $\mathcal{O}(h_i^2)$，就是 DPM-Solver-1 update rule (Eq. 3.7):

$$\tilde{\mathbf{x}}_{t_i} = \frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\tilde{\mathbf{x}}_{t_{i-1}} - \sigma_{t_i}(e^{h_i} - 1)\boldsymbol{\epsilon}_\theta(\tilde{\mathbf{x}}_{t_{i-1}}, t_{i-1})$$

**一个惊人的等价**：DDIM (Song et al., 2021, https://arxiv.org/abs/2010.02502) 的 update rule 写成：

$$\tilde{\mathbf{x}}_{t_i} = \frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\tilde{\mathbf{x}}_{t_{i-1}} - \alpha_{t_i}\left(\frac{\sigma_{t_{i-1}}}{\alpha_{t_{i-1}}} - \frac{\sigma_{t_i}}{\alpha_{t_i}}\right)\boldsymbol{\epsilon}_\theta(\tilde{\mathbf{x}}_{t_{i-1}}, t_{i-1})$$

由 $\frac{\sigma_t}{\alpha_t} = e^{-\lambda_t}$，立刻得到：

$$\alpha_{t_i}\left(\frac{\sigma_{t_{i-1}}}{\alpha_{t_{i-1}}} - \frac{\sigma_{t_i}}{\alpha_{t_i}}\right) = \alpha_{t_i}(e^{-\lambda_{t_{i-1}}} - e^{-\lambda_{t_i}}) = \sigma_{t_i}(e^{h_i} - 1)$$

**所以 DDIM = DPM-Solver-1**。这解释了一个 long-standing puzzle：为什么 DDIM 比 Euler discretization of diffusion ODE 好？因为 DDIM 隐式地利用了 semi-linear 结构，线性项被解析处理了。Salimans & Ho (progressive distillation) 之前也证明 DDIM 是 first-order discretization，但他们没法解释 DDIM vs Euler 的差异，因为他们没有 semi-linear 这个视角。DPM-Solver 的 formulation 给了一个 clean 的解释。

---

## 7. DPM-Solver-2：二阶版本

二阶需要近似 $\hat{\boldsymbol{\epsilon}}_\theta^{(1)}$（一阶 total derivative）。Exponential integrator 文献里的 standard trick：用一个中间点 $(s_i, \mathbf{u}_i)$ 来估计 derivative。

Algorithm 1 给的 update:

```
1. h_i = λ_{t_i} - λ_{t_{i-1}}
2. s_i = t_λ((λ_{t_{i-1}} + λ_{t_i})/2)   # 中点对应的实际时间
3. u_i = (α_{s_i}/α_{t_{i-1}})x̃_{t_{i-1}} - σ_{s_i}(e^{h_i/2} - 1)·ε_θ(x̃_{t_{i-1}}, t_{i-1})
4. x̃_{t_i} = (α_{t_i}/α_{t_{i-1}})x̃_{t_{i-1}} - σ_{t_i}(e^{h_i} - 1)·ε_θ(u_i, s_i)
```

intuition：
- Step 3 用 DPM-Solver-1 走半步到中点 $s_i$，得到中间 latent $\mathbf{u}_i$
- Step 4 用中点处的 $\boldsymbol{\epsilon}_\theta(\mathbf{u}_i, s_i)$ 作为整步的代表值，做一次大的 update

这非常像 explicit midpoint method（RK2），但关键区别：**linear 部分用 $\frac{\alpha_t}{\alpha_s}$ 系数精确求解，nonlinear 部分用 midpoint 近似**。

证明 $\bar{\mathbf{x}}_t = \mathbf{x}_t + \mathcal{O}(h^3)$ (Appendix B.4) 的核心是验证：

$$h^2\varphi_2(h) - (e^h - 1)\frac{\lambda_{s_1} - \lambda_s}{2r_1} = \mathcal{O}(h^3)$$

代入 $r_1 = 0.5$，$\lambda_{s_1} - \lambda_s = r_1 h = h/2$，可以验证等式成立到 $\mathcal{O}(h^3)$。

---

## 8. DPM-Solver-3：三阶版本

三阶需要近似 $\hat{\boldsymbol{\epsilon}}_\theta^{(1)}$ 和 $\hat{\boldsymbol{\epsilon}}_\theta^{(2)}$。Algorithm 2 用两个中间点 $r_1 = 1/3$, $r_2 = 2/3$：

```
1. s_{2i-1} = t_λ(λ_{t_{i-1}} + (1/3)h_i)
   s_{2i}   = t_λ(λ_{t_{i-1}} + (2/3)h_i)
2. u_{2i-1} = (α_{s_{2i-1}}/α_{t_{i-1}})x̃_{t_{i-1}} - σ_{s_{2i-1}}(e^{h_i/3} - 1)·ε_θ(x̃_{t_{i-1}}, t_{i-1})
3. D_{2i-1} = ε_θ(u_{2i-1}, s_{2i-1}) - ε_θ(x̃_{t_{i-1}}, t_{i-1})
4. u_{2i} = (α_{s_{2i}}/α_{t_{i-1}})x̃_{t_{i-1}} - σ_{s_{2i}}(e^{2h_i/3} - 1)·ε_θ(x̃_{t_{i-1}}, t_{i-1})
            - σ_{s_{2i}}·(2)/(1)·((e^{2h_i/3}-1)/(2h_i/3) - 1)·D_{2i-1}
5. D_{2i} = ε_θ(u_{2i}, s_{2i}) - ε_θ(x̃_{t_{i-1}}, t_{i-1})
6. x̃_{t_i} = (α_{t_i}/α_{t_{i-1}})x̃_{t_{i-1}} - σ_{t_i}(e^{h_i}-1)·ε_θ(x̃_{t_{i-1}}, t_{i-1})
              - (σ_{t_i}/(2/3))·((e^{h_i}-1)/h_i - 1)·D_{2i}
```

intuition：
- $\mathbf{D}_{2i-1}$ 是 $\hat{\boldsymbol{\epsilon}}_\theta^{(1)}$ 的一阶估计（用 first intermediate point）
- $\mathbf{D}_{2i}$ 是更高阶的修正
- 最终 update 包含三个 terms：零阶项 + 一阶 derivative 修正 + 二阶 derivative 修正

证明 (Appendix B.5) 要求验证三个 conditions：
1. $h\varphi_1(h) = e^h - 1$
2. $h^2\varphi_2(h) = (\frac{e^h-1}{h} - 1)h$
3. $h^3\varphi_3(h) = (\frac{e^h-1}{h} - 1)\frac{r_2 h^2}{2} + \mathcal{O}(h^4)$

这些都通过 Taylor expansion 验证成立。

Theorem 3.2 给出 convergence order: $\tilde{\mathbf{x}}_{t_M} - \mathbf{x}_0 = \mathcal{O}(h_{\max}^k)$ for $k$-th order solver.

---

## 9. 为什么 DPM-Solver 比 RK 好？—— Semi-Linear 结构的力量

Table 1 的 ablation 实验数据非常 convincing：

| Method | NFE=12 | NFE=18 | NFE=24 |
|--------|--------|--------|--------|
| RK2 (t) | 16.40 | 7.25 | 3.90 |
| RK2 (λ) | 107.81 | 42.04 | 17.71 |
| **DPM-Solver-2** | **5.28** | **3.43** | **3.02** |
| RK3 (t) | 48.75 | 21.86 | 10.90 |
| RK3 (λ) | 34.29 | 4.90 | 3.50 |
| **DPM-Solver-3** | **6.03** | **2.90** | **2.75** |

注意 RK2 (λ) 极差（FID 107.81 at NFE=12），而 DPM-Solver-2 同样在 $\lambda$ 域操作，FID 只有 5.28。这说明**仅仅 change-of-variable 不够，关键是利用 semi-linear 结构把 linear 部分解析化**。

intuition：
1. RK 直接 discretize $\mathbf{h}_\theta$，linear term $f(t)\mathbf{x}_t$ 也被 discretize。但 linear term 的 exact solution 是 $e^{\int f}\mathbf{x}$，对步长敏感（指数放大误差）。
2. DPM-Solver 把 linear term 用 $\frac{\alpha_t}{\alpha_s}$ exact 写出来，discretization error 只来自 nonlinear $\boldsymbol{\epsilon}_\theta$ 部分。
3. 在 few-step regime，linear term 误差 dominates，所以 RK 翻车；DPM-Solver 把这块误差完全消除，所以能 10 步 sample。

这在 exponential integrator 文献里是 well-known phenomenon (Hochbruck & Ostermann, 2005, https://epubs.siam.org/doi/10.1137/S0036142903405897)：对 stiff/semi-linear ODE，explicit RK 在大步长不稳定，exponential integrators 显著更 robust。

---

## 10. Time Step Schedule 与 Discrete-Time DPMs 的适配

### Uniform in $\lambda$ 而不是 $t$

paper 提议在 $\lambda$ 域 uniform 划分：

$$\lambda_{t_i} = \lambda_T + \frac{i}{M}(\lambda_0 - \lambda_T), \quad i = 0, \ldots, M$$

intuition：因为 solution 在 $\lambda$ 域里 kernel 是 $e^{-\lambda}$，$\boldsymbol{\epsilon}_\theta$ 关于 $\lambda$ 的变化更平滑，所以 $\lambda$-uniform 比时间 uniform 更"高效"。

### Discrete-Time DPMs

DDPM (Ho et al., 2020, https://arxiv.org/abs/2006.11239) 在离散时间 $t_n = nT/N$ 训练，$N=1000$ 或 $4000$。model 输入是整数 index。DPM-Solver 通过 reparameterization 把它转成 continuous：

- **Type-1**: $\boldsymbol{\epsilon}_\theta(\mathbf{x}, t) = \tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}, 1000\cdot\max(t - T/N, 0))$
- **Type-2**: $\boldsymbol{\epsilon}_\theta(\mathbf{x}, t) = \tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}, 1000\cdot(N-1)t/(NT))$

实验发现 NFE 小时 Type-1 + $\epsilon = 10^{-3}$ 好；NFE 大时 Type-2 + $\epsilon = 10^{-4}$ 好。

### "Fast" Combination for NFE ≤ 20

为了让 NFE budget 完全用完（不被 $\lfloor K/k\rfloor$ 浪费），paper 提出：先尽量用 DPM-Solver-3，剩余 steps 用 DPM-Solver-1 或 -2 补齐。具体见 Appendix D.3。这是个小工程细节，但能带来明显 FID 提升。

---

## 11. 实验结果一览

### CIFAR-10 (continuous-time, VP deep)

- **NFE=10**: FID **4.70**（DPM-Solver）vs 19.55 (RK45, NFE=26) vs 82.42 (Improved Euler, NFE=48)
- **NFE=20**: FID **2.87**
- 1000 步 Euler-Maruyama SDE: FID 2.44

也就是 10 步达到接近 1000 步 SDE 的质量，~100× 加速。

### 各 dataset 速度对比

- CIFAR-10: DPM-Solver 12 步 FID 4.65 vs Analytic-DDIM 12 步 11.68
- CelebA 64×64: DPM-Solver 12 步 FID 4.20 vs DDIM 12 步 9.99
- ImageNet 64×64 (cosine schedule): DPM-Solver 12 步 FID 20.03 vs DDIM 12 步 52.69
- ImageNet 128×128 (classifier guidance): DPM-Solver 12 步 FID 5.84 vs DDIM 12 步 9.38
- LSUN bedroom 256×256: DPM-Solver 12 步 FID 4.21 vs DDIM 12 步 7.51

### Runtime

Table 7 显示同 NFE 下 DPM-Solver 和 DDIM 几乎一样快（甚至略快，因为 implementation 优化），所以 NFE 的减少直接 translate 到 wall-clock time 加速。

---

## 12. Appendix A 的深度洞察：Invariance to Noise Schedule

paper Appendix A.1 给了一个非常 elegant 的结论。对 VP-type DPMs ($\alpha_t^2 + \sigma_t^2 = 1$)，可以解析求出：

$$\alpha_t = \sqrt{\frac{1}{1 + e^{-2\lambda_t}}}, \quad \sigma_t = \sqrt{\frac{1}{1 + e^{2\lambda_t}}}$$

定义 $\hat{\alpha}_\lambda := \sqrt{\frac{1}{1+e^{-2\lambda}}}$，那么 solution 写成：

$$\hat{\mathbf{x}}_{\lambda_t} = \frac{\hat{\alpha}_{\lambda_t}}{\hat{\alpha}_{\lambda_s}}\hat{\mathbf{x}}_{\lambda_s} - \hat{\alpha}_{\lambda_t}\int_{\lambda_s}^{\lambda_t} e^{-\lambda}\hat{\boldsymbol{\epsilon}}_\theta(\hat{\mathbf{x}}_\lambda, \lambda)d\lambda$$

**关键**：integrator $e^{-\lambda}\hat{\boldsymbol{\epsilon}}_\theta$ 只依赖 $\lambda$，不依赖具体 noise schedule。所以从 $\lambda_s$ 到 $\lambda_t$ 的 solution 完全由 $\lambda_s, \lambda_t, \hat{\boldsymbol{\epsilon}}_\theta$ 决定，与中间怎么 schedule 无关。

更进一步，Appendix A.3 指出 maximum likelihood training loss 在 $\lambda$ 域里也是 invariant：

$$D_{KL}(q_0 \| p_0) \leq D_{KL}(q_T\|p_T) + \int_{\lambda_T}^{\lambda_0}\mathbb{E}\left[\|\boldsymbol{\epsilon}_\theta(\hat{\mathbf{x}}_\lambda, \lambda) - \boldsymbol{\epsilon}\|^2\right]d\lambda + C$$

这等价于 Kingma et al. (Variational Diffusion Models, https://arxiv.org/abs/2107.00630) 和 Song et al. (Maximum Likelihood Training of Score-Based Diffusion, https://arxiv.org/abs/2101.09258) 的 importance weighting trick。

**Big picture insight**: 训练和采样都 invariant to noise schedule。这意味着 diffusion model 实质上是在 $\lambda$ 域上定义的，时间 $t$ 只是个 reparameterization 罢了。这暗示我们可以直接在 $\lambda$ 域定义 model，省去 ad-hoc noise schedule 设计。这是一个 unifying viewpoint。

---

## 13. 我的 Intuition 总结

把整个 DPM-Solver 拆开看，最深刻的几点：

1. **Semi-linear structure 是 diffusion ODE 的内在结构**，不是巧合。它来自 $f(t)\mathbf{x}_t$ 这个 pure scaling term，对应 $\alpha_t$ 的衰减。任何"信号 + 噪声"的 model 都会有这个 structure。

2. **$\lambda$ 域是 diffusion 的"自然坐标系"**。在 $\lambda$ 域里，noise schedule 被吸收成 universal kernel $e^{-\lambda}$，model 只学一个 mapping $\hat{\boldsymbol{\epsilon}}_\theta(\hat{\mathbf{x}}, \lambda)$，sampling 和 training 都在这里进行最自然。这有点像物理学里换坐标系——选对坐标系（symmetry-aligned）让方程最简洁。

3. **Exponential integrator 是把"已知的线性动力学"和"未知的非线性扰动"分离的标准工具**。DPM-Solver 的贡献是识别出 diffusion ODE 恰好是 semi-linear，把这个工具带过来。这件事在 numerical ODE 文献里是 well-established，但 ML 社区之前完全没用上。

4. **DDIM 的"隐式高阶性"**：DDIM 看起来是个 heuristic deterministic sampler，实际上它是利用了 semi-linear structure 的 first-order exponential integrator。这解释了它比朴素 Euler discretization 显著好。这个 equivalence 也是为什么 DPM-Solver 可以"无缝替换 DDIM"——它们用同一个 update structure，只是 DPM-Solver 又往上叠了 higher-order correction。

5. **Few-step regime 的工作原理**：在 NFE=10 时，linear term 用 $\frac{\alpha_t}{\alpha_s}$ exact 计算意味着信号 scaling 完全无误差，剩下的误差只来自 $\boldsymbol{\epsilon}_\theta$ 在 $\lambda$ 上的有限采样。Taylor 展开到 2-3 阶让 nonlinear 部分也有 $\mathcal{O}(h^3)$ 甚至 $\mathcal{O}(h^4)$ 的精度。所以 NFE=10 的 $\mathcal{O}(h_{\max}^k)$ 在 $h_{\max} \approx (\lambda_0 - \lambda_T)/M$ 上能小到让 FID 接近 asymptotic limit。

6. **为什么 4 阶以上不做**：Exponential integrator 高阶需要更多 intermediate points，而且对 neural network Lipschitz 假设更严格。paper 实测发现 3 阶已经足够 few-step，继续往上 gain 不大。

7. **Limitations**：DPM-Solver 解的是 ODE，丢掉了 SDE 的 stochasticity。在 likelihood evaluation 场景下不能用（likelihood 需要 SDE 的 normalizing constant）。此外 10 步虽然快，但相对 GAN 的单步 forward 仍慢几十倍，real-time 应用仍需 distillation 类方法。后续工作 DPM-Solver++ (https://arxiv.org/abs/2211.01095) 和 UniPC (https://arxiv.org/abs/2302.04867) 进一步改进了 converge order 和 stability。

8. **Connection 到 Flow Matching / Rectified Flow**：后来 Rectified Flow (Liu et al., 2023, https://arxiv.org/abs/2209.03003) 和 Flow Matching (Lipman et al., 2023, https://arxiv.org/abs/2210.02747) 把 $\lambda$ 域的 viewpoint 进一步推到极致——直接在数据到噪声的直线路径上做 simulation，从而用 1 步 ODE 就能 sample。DPM-Solver 的 $\lambda$-space formulation 实际上是这条思路的 early version：把"自然的进度变量"作为求解坐标。

---

## References

- DPM-Solver: https://arxiv.org/abs/2206.00927
- Code: https://github.com/LuChengTHU/dpm-solver
- Score-SDE (Song et al., 2021): https://arxiv.org/abs/2011.13456
- DDIM (Song et al., 2021): https://arxiv.org/abs/2010.02502
- DDPM (Ho et al., 2020): https://arxiv.org/abs/2006.11239
- Variational Diffusion Models (Kingma et al., 2021): https://arxiv.org/abs/2107.00630
- Exponential Integrators review (Hochbruck & Ostermann, 2010): https://doi.org/10.1017/S0962492910000048
- Explicit Exponential RK (Hochbruck & Ostermann, 2005): https://epubs.siam.org/doi/10.1137/S0036142903405897
- DPM-Solver++: https://arxiv.org/abs/2211.01095
- UniPC: https://arxiv.org/abs/2302.04867
- Rectified Flow: https://arxiv.org/abs/2209.03003
- Flow Matching: https://arxiv.org/abs/2210.02747
- Diffusion Beats GANs (Dhariwal & Nichol, 2021): https://arxiv.org/abs/2105.05233
- Improved DDPM (Nichol & Dhariwal, 2021): https://arxiv.org/abs/2102.09672
