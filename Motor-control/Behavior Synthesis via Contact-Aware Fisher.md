---
source_pdf: Behavior Synthesis via Contact-Aware Fisher.pdf
paper_sha256: 86eb3c27ae7007b35fbd25967b3b174f8e6f69763ef80b4ed2fb62cbbb4b30f0
processed_at: '2026-07-18T14:37:55-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Behavior Synthesis via Contact-Aware Fisher Information Maximization — 深度讲解

这篇论文来自 Yale University 的 Hrishikesh Sathyanarayan 和 Ian Abraham，发表于 2024 年前后。核心思想一句话概括：**让机器人通过"主动找接触"来学习物理参数，背后数学是 contact-aware Fisher information maximization**。最终涌现出 (emergent) 类似人类触觉探索的行为——hefting（掂重量）、rubbing（搓摩擦）、squeezing（捏刚度）、contouring（描形状）。

GitHub repo: https://github.com/ialab-yale/contact_aware_active_learning

---

## 1. 为什么这篇 paper 有意思：触觉学习的"信息论"视角

人类要估一个东西的重量，会**颠一颠** (hefting)；要感受摩擦，会**搓一搓** (rubbing)；要感觉软硬，会**捏一捏** (squeezing)；要描一个形状，会**沿着边摸过去** (contouring)。这些不是巧合，是信息效率最高的探索动作。Lederman & Klatzky 的经典 haptic exploration 工作 [https://www.sciencedirect.com/science/article/pii/0001691893900708] 就指出过：不同物理参数需要不同性质的"探索程序 (exploratory procedure)"。

这篇 paper 把这个现象从 information geometry 的角度正式数学化：**哪些接触模式对参数 θ 的 Fisher information 贡献最大，机器人就 synthesize 哪种动作**。结果是 emergent behavior 自然落到了人类的探索程序上。

关键 link：
- 人类触觉探索文献：https://www.sciencedirect.com/science/article/pii/0001691893900708
- Haptic intelligence 综述可参考 https://arxiv.org/abs/2103.15586

---

## 2. 数学骨架：从 MAP 到 Contact-Aware Fisher

### 2.1 参数学习 = Maximum A-Posteriori

给定参数 $\theta \in \Theta \subseteq \mathbb{R}^d$（比如 mass, friction coefficient, stiffness 等），数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$（input $x_i$、传感器读数 $y_i$），先验 $p(\theta)$：

$$p(\theta | \mathcal{D}) = \frac{p(\mathcal{D}|\theta) p(\theta)}{\int_\Theta p(\mathcal{D}|v) p(v) dv}$$

变量含义：
- $\theta$：我们要估的物理参数（比如质量 $m$）
- $p(\mathcal{D}|\theta)$：likelihood，给定参数预测传感器读数的概率
- 分母 $\int_\Theta \cdots$：marginal evidence，通常 intractable

由于分母难算，直接取 mode：

$$\hat\theta = \arg\max_\theta \log\big(p(\mathcal{D}|\theta)p(\theta)\big)$$

$\hat\theta$ 是点估计，不含不确定性信息。

### 2.2 Fisher Information — 信息的几何度量

Fisher information matrix (FIM)：

$$\mathcal{F}(\mathcal{D}|\theta) = \mathbb{E}\left[\nabla_\theta \mathcal{L}(\mathcal{D}|\theta) \nabla_\theta \mathcal{L}(\mathcal{D}|\theta)^\top\right] = -\mathbb{E}\left[\nabla_\theta^2 \mathcal{L}(\mathcal{D}|\theta)\right]$$

变量解析：
- $\mathcal{L}(\mathcal{D}|\theta) = \log p(\mathcal{D}|\theta)$：log-likelihood
- $\nabla_\theta$：对参数求梯度（列向量，长度 $d$）
- $\nabla_\theta^2$：Hessian（$d \times d$ 矩阵）
- 期望 $\mathbb{E}$ 对随机变量 $\mathcal{D}$ 取

直觉：FIM 是 statistical manifold 上的 Riemannian metric，度量"参数小扰动 $\delta\theta$ 引起 likelihood 多大变化"。FIM 大 = 这个测量点对参数特别敏感 = 信息丰富。

参考：https://en.wikipedia.org/wiki/Fisher_information

### 2.3 Cramér-Rao Lower Bound — FIM 越大方差越小

$$\mathcal{F}(\mathcal{D}|\theta) \succeq \text{cov}(\hat\theta)^{-1}$$

含义：估计协方差矩阵 $\text{cov}(\hat\theta)$ 被 FIM 的逆"下界"。FIM 大 → 估计方差小 → 参数更精确。这是这篇 paper 整个框架的 "为什么"。

参考：https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound

### 2.4 Optimal Experimental Design (OED)

经典 OED：选择输入 $X = \{x_i\}_{i=1}^N$（实验设计变量）最大化 FIM 的某个标量函数：

$$\max_X \psi\big(\mathcal{F}(X|\theta)\big)$$

- $\psi$：scalarization。常见选择：
  - **D-optimality**：$\psi = \det(\mathcal{F})$，最大化信息体积
  - **T-optimality**：$\psi = \text{tr}(\mathcal{F})$，最大化信息总和（这篇 paper 用的就是这个）
  - A-optimality：$\psi = -\text{tr}(\mathcal{F}^{-1})$，最小化平均方差

经典参考：
- Fedorov, *Optimal Experimental Design* https://onlinelibrary.wiley.com/doi/abs/10.1002/wics.100
- Atanasov et al. 信息采集 https://arxiv.org/abs/1405.2446

---

## 3. 核心创新：把 Contact 显式塞进 Fisher

### 3.1 Contact-Aware MAP (CA-MAP)

机器人状态 $x_t \in \mathcal{X}$，控制 $u_t \in \mathcal{U}$，接触力 $\lambda_t$。轨迹定义为：

$$\tau = \{(x_t, \lambda_t)\}_{t=0}^{T-1}$$

约束系统：
- 动力学：$x_{t+1} = f_\theta(x_t, u_t, \lambda_t)$
- 接触约束：$\lambda_t \in \mathcal{C}_\theta(x_t)$（非穿透、互补条件、摩擦锥等）
- 传感器：$y_t = g_\theta(x_t)$

CA-MAP 优化问题：

$$\hat\theta = \arg\max_{\theta, \tau} \log p(\mathcal{D}|\tau, \theta) p(\theta)$$

$$\text{s.t.}\quad x_0 \text{ given},\ x_{t+1}=f_\theta(x_t,u_t,\lambda_t),\ \lambda_t \in \mathcal{C}_\theta(x_t),\ \theta \in \Theta$$

关键点：$\theta$ 同时出现在 $f_\theta$（动力学）、$\mathcal{C}_\theta$（接触约束）、$g_\theta$（传感器）里。**接触力 $\lambda_t$ 是显式优化变量，不是测量副产品**。

### 3.2 Lagrangian 与 Contact-Aware FIM

构造 Lagrangian：

$$\mathcal{L}(\mathcal{D}|\tau,\theta,\alpha) = \ell(\mathcal{D}|\tau,\theta) + \alpha\, d(\tau,\theta)$$

变量：
- $\ell(\mathcal{D}|\tau,\theta) = \log p(\mathcal{D}|\tau,\theta) p(\theta)$：增广 log-posterior
- $\alpha$：Lagrange 乘子向量（对应每个约束）
- $d(\tau,\theta) = 0$：等式约束（接触互补、动力学等）

**Theorem 1**（这篇 paper 的核心定理）：contact-aware FIM 定义为

$$\boxed{\mathcal{F}(\mathcal{D}|\tau,\theta,\alpha) = -\nabla_\theta^2 \mathcal{L}(\mathcal{D}|\tau,\theta,\alpha)}$$

它来自 steepest ascent 问题：

$$\delta\theta^\star = \arg\max_{\delta\theta} \nabla_\theta \mathcal{L}^\top \delta\theta + \tfrac{1}{2}\delta\theta^\top \nabla_\theta^2 \mathcal{L}\, \delta\theta$$

对 $\delta\theta$ 求导令其为零，闭式解：

$$\delta\theta^\star = \mathcal{F}(\mathcal{D}|\tau,\theta)^{-1} \nabla_\theta \mathcal{L}(\mathcal{D}|\tau,\theta)$$

**这就是 Newton 法的 step！** 所以前一篇论文证明的本质是：**最大化 FIM 等价于改善 Newton step 的条件数（condition number）**——FIM 越好 conditioned，CA-MAP 的 Newton 迭代越快收敛到 MAP 估计。

直觉总结：把 contact 当成"调节器"，通过它直接塑造 likelihood 的曲率 landscape。

### 3.3 为什么 contact 在 Fisher 里非平凡

注意 $\tau = \{(x_t, \lambda_t)\}$ 是 FIM 的显式参数。普通 OED 的 FIM 是 $\mathcal{F}(\mathcal{D}|\theta)$，contact-aware FIM 是 $\mathcal{F}(\mathcal{D}|\tau, \theta)$——**多了 contact trajectory 这个旋钮**。

物理直觉：要让 $\mathcal{F}$ 对 $\theta$ 敏感，需要让传感器读数 $y_t = g_\theta(x_t)$ 通过接触被 $\theta$ 强烈"调制"。摩擦系数 $\mu$ 只在切向力里出现，所以 $\nabla_\theta g_\theta$ 在 $\theta = \mu$ 上非零，必须让机器人有切向速度 + 法向压力，FIM 才非零。这一条直接预测出 rubbing 行为。

---

## 4. Contact-Aware Optimal Experimental Design (CA-OED)

### 4.1 优化公式

$$\hat{\mathcal{D}}, \hat\tau = \arg\max_{\mathcal{D},\tau} \psi\big(\mathcal{F}(\mathcal{D}|\tau,\theta)\big) - \mathcal{I}(\mathcal{D},\tau)$$

约束同 CA-MAP（动力学、接触、传感器）。

- $\psi = \text{tr}(\cdot)$：trace，T-optimality
- $\mathcal{I}(\mathcal{D},\tau)$：轨迹代价（避免极端行为）

为什么选 trace 而不是 determinant？作者在 Appendix B 给出依据：

$$\kappa(\mathcal{F}) \leq \text{bound}(\|\mathcal{F}\|^2 \|\mathcal{F}^{-1}\|^2) \quad\text{with}\quad \|\mathcal{F}\|^2 = \text{tr}(\mathcal{F}^\top\mathcal{F})$$

trace 大 → FIM 条件数有界 → Newton 步骤数值稳定。这是工程上非常实用的考量。

### 4.2 求解：Predictive Sampling

作者用 Howell et al. 的 predictive sampling [https://arxiv.org/abs/2212.00541]，这是一种 sampling-based MPC 方法：
- 短时间 horizon $t_H = 10$
- $N = 10$ 个 sample，Cubic Hermite spline 构造 control
- 采样方差 $\Sigma = 1.0$
- 每步通过 forward rollout 在 MuJoCo [https://mujoco.org/] 评估 FIM，选最好的样本

Contact 通过 dynamics forward rollout 隐式求解，不用显式 complementarity programming（虽然理论上 Posa 等人的 contact-implicit TO [https://doi.org/10.1177/0278364913506757] 也用 LCP）。

### 4.3 闭环：Belief 更新

执行轨迹后收集真实数据 $\mathcal{D}$，用近似公式更新参数协方差：

$$\Sigma_\theta^+ = \big(\mathcal{F}(\mathcal{D}|\tau, \hat\theta) + \Sigma_\theta^{-1}\big)^{-1}$$

变量：
- $\Sigma_\theta$：先验协方差（执行前）
- $\Sigma_\theta^+$：后验协方差（执行后）
- $\hat\theta$：通过 CA-MAP 算出的点估计

这相当于 Gaussian 近似下的 Bayesian update，把 FIM 当作观测信息矩阵。作者提到这个近似"over-approximates uncertainty"，但实践中反而帮助收敛。

参考 DART（类似思想）：https://arxiv.org/abs/1904.02451

### 4.4 Algorithm 1 流程

```
Input: x_0, prior p(θ), models f_θ, g_θ, C_θ, k_max, T
for k = 0 to k_max:
    1. CA-OED (Eq.12) → 期望数据 D̂ + 参考轨迹 τ̂
    2. 在 robot 上执行 → 收集真实 D
    3. CA-MAP (Eq.8) → 更新 θ̂
    4. 用 Eq.13 → 更新 Σ_θ^+
```

闭环结构：
```
prior → [CA-OED planner] → trajectory → [robot + world] → sensor data
                                ↑                                              ↓
                                ←——— [CA-MAP estimator + Σ update] ←———
```

---

## 5. Contact Sensor Model（关键细节）

论文用的 soft contact model：

$$\lambda_n = \max\big(0,\ -K\phi_n - C|v_n|\big)$$
$$\lambda_t = \hat v_t \max\big(-\mu\lambda_n,\ -R|v_t|\big)$$

变量解析：
- $K$：法向刚度 [N/m]
- $C$：法向阻尼 [Ns/m]
- $\mu$：摩擦系数（无量纲）
- $R$：切向摩擦阻力 [Ns/m]
- $\phi_n$：signed distance function（物体表面到手之间的穿透深度，负值表示穿透）
- $v_c = [v_n, v_t]^\top = J_c(q)\dot q$：contact frame 下的速度
- $J_c(q)$：contact Jacobian，把 joint velocity $\dot q$ 映射到接触点速度
- $\hat v_n, \hat v_t$：单位方向向量

这是解析可微的（除了 max 的非光滑点），让 FIM 的 Hessian 可以通过自动微分计算。

参数表（Table I）：

| Experiment | K [N/m] | C [Ns/m] | μ | R [Ns/m] |
|---|---|---|---|---|
| Hefting | 500 | 0 | N/A | N/A |
| Rubbing | 100 | 1 | **0.4** | 2.0 |
| Pinching | **800** | **10** | N/A | N/A |
| Contouring | 500 | 0 | 0 | 0 |

粗体是待估参数。

---

## 6. 实验设计与 Emergent Behaviors

四个场景（Fig. 3）：

### 6.1 Mass Estimation (Hefting)
- 系统：Allegro hand + Franka
- 物体：网球 $m = 0.05$ kg（实验中乘 100）
- 空间：1D vertical domain $\mathcal{W} = [0, 0.6]$ m
- 速度约束：$v_{ee} \in [-1, 1]$ m/s
- 初值：$p_{ee,0} = 0.2$ m
- 先验方差：$\Sigma_\theta = 10$ kg²

Emergent behavior：**hefting（上下掂）**。机器人在持物状态下做垂直振荡，扰动法向接触力 $\lambda_n$，让 $\partial \lambda_n / \partial m \neq 0$（mass 通过 signed distance $\phi_n$ 和 $v_n$ 进入接触模型）。

为什么不能让球离开手？离开的瞬间接触力=0，FIM 在那段时刻为零贡献，信息损失。所以持续接触 + 法向扰动最优。

### 6.2 Friction Estimation (Rubbing)
- 系统：Franka + 墙
- 真值：$\mu = 0.4$
- 空间：2D wall frame，$\mathcal{W} = [0,1] \times [0.5, 1]$ m
- 初值：$p_{ee, 0} = [0.5, 0.8]$ m
- 先验方差：$\Sigma_\theta = 1$

Emergent behavior：**rubbing（搓）**。同时维持 $v_t \neq 0$（切向速度）和 $\lambda_n > 0$（法向压力），因为 $\mu$ 只在 $\lambda_t$ 项里出现。

### 6.3 Stiffness Estimation (Pinching/Squeezing)
- 系统：gripper + 球
- 真值：$K = 800$ N/m, $C = 10$ Ns/m
- 预夹爪状态：4 mm 距离
- 先验方差：$\Sigma_\theta = [100, 10]$

Emergent behavior：**repeatedly squeezing + releasing（捏-松-捏）**。要 excite $\phi_n$（穿透深度）和 $\partial\phi_n/\partial t$（穿透速度）， stiffness 通过 $\lambda_n = -K\phi_n$ 进入，damping 通过 $-C|v_n|$ 进入。

### 6.4 Shape Estimation (Contouring)
- 系统：Franka + 盒子
- 真值：$l = 0.126$ m, $w = 0.05$ m
- 平面运动约束
- 在线估计（in-the-loop）

Emergent behavior：**沿物体边界 contouring**，通过 make-break contact 在不同表面区域激活 signed distance $\phi_n$ 对 $l, w$ 的偏导。

---

## 7. 实验数据对比

作者对比 baseline ASID [Memmel et al. 2024, https://arxiv.org/abs/2404.12308]，ASID 是 belief space planning 但 contact naive（用 finite differencing 估梯度）。

Fig. 6 显示 %err = $100(\hat\theta - \theta^*)/\theta^*$ 随实验次数的变化：
- Mass：CA-OED 明显更快收敛到 ~0%，ASID 残留约 20%
- Friction：CA-OED ~5%，ASID ~20%
- Material (stiffness + damping)：CA-OED 更稳，但因 1D 约束信息有限，gap 较小
- Shape：CA-OED 在 in-the-loop 下能持续逼近

Fig. 7 robustness：7 个不同 prior，distance $\delta = |\hat\theta - \theta^*|$，CA-OED 在 ~30% 误差范围内都 robust 收敛。

Fig. 8 & 9：**Information landscape 可视化**——
- Hefting：FIM landscape 对 $(\phi_n, v_n)$ 近似线性，最大值在速度边界 → 自然产生 hefting
- Pinching：对 $(\phi_n, \partial\phi_n/\partial t)$ 存在局部最优 trade-off → squeezing 行为
- Rubbing：对 $(v_n, v_t)$ landscape 在两轴同时非零时取峰 → rubbing 行为

这是 paper 的高光时刻：**FIM landscape 的几何结构直接决定了 emergent behavior 的类别**。

---

## 8. 关键 Insight 与 Karpathy 视角的思考

### 8.1 这篇 paper 在概念上的位置

它把三件事缝起来：
1. **Optimal Experimental Design**（统计学传统）
2. **Contact-Implicit Trajectory Optimization**（Posa, Todorov 学派）
3. **Active Perception / Belief Space Planning**（Platt, Kaelbling 学派）

缝合的胶水是 Fisher information。这种做法和现代 RL-based active learning（比如 ASID）的最大区别：**不学 policy，而是直接在 trajectory 层面优化 information-theoretic objective**。优势是 sample efficient，不需要大量 RL 数据；劣势是依赖可微的接触模型，不容易 scale 到高维复杂系统。

相关 link：
- ContactNets (Posa 组)：https://proceedings.mlr.press/v155/pfrommer20a.html
- Impact-invariant control：https://arxiv.org/abs/2203.05666
- ADMM MPC for multi-contact：https://arxiv.org/abs/2203.02535

### 8.2 与 LLM/Scaling 范式的对比

作为 Karpathy 你会想：这种 model-based 信息论方法 vs. 当下 scaling 的 model-free RL+大模型路线。

- **This paper 的赌注**：结构化的 information metric 让 sample efficiency 极高，每一步都有明确目的。
- **Scaling 派的赌注**：让网络自己 figure out 探索策略，可处理不可微接触、未知动力学。

两者未必矛盾。Fisher information 可以作为 RL 的 intrinsic reward，把 information landscape 当 shaping reward。最近工作如 https://arxiv.org/abs/2403.16644（Bayesian sim-to-real）就在尝试类似融合。

### 8.3 和你家 ToyGrad / NanoGPT 那种"microscope into training"哲学的呼应

这篇 paper 有点像 "microscope into robot exploration"：通过 FIM landscape 把"为什么这个动作对学习有帮助"变得可视化、可解释。这和 Karpathy 在 makemore / micrograd 里强调的"看每一步发生了什么"是同一种精神——**拒绝 black box，要看见信息怎么流**。

### 8.4 Limitations

作者诚实承认：
1. **Deterministic dynamics 假设**：没建模 process noise，sim-to-real gap 风险
2. **Non-convex optimization**：predictive sampling 可能陷局部最优
3. **Contact model 调参**：soft contact 参数 $K, C$ 需要先验已知（chicken-and-egg）
4. **Fisher 只保证 precision 不保证 accuracy**：可能有 systematic bias

第 4 点特别重要。Cramér-Rao 是方差下界，但若模型 mispecified，再大 FIM 也只是收敛到错的值。这是所有 Fisher-based 方法的通病。Bayesian model averaging 才能解决，但代价高。

### 8.5 和 Active Perception 历史脉络

- Bajcsy 1988 "Active Perception"（最早提出）：https://www2.cs.duke.edu/courses/spring03/cps296.4/papers/bajcsy_88.pdf
- Aloimonos 1988 "Active Vision"：https://www.cs.cmu.edu/~alex/ITva.pdf
- Platt 2010 belief space：https://arxiv.org/abs/1209.1991
- Atanasov 2014 信息采集：https://arxiv.org/abs/1405.2446

这篇 paper 把 active perception 的精神带到 contact-rich manipulation，并提供了第一个"接触力显式进入 FIM"的形式化。

### 8.6 未来方向思考

如果让我（Karpathy 视角）继续推：
1. **Neural contact model + Fisher**：用可微 sim（Brax, Genesis, MJX）把 contact 整个做成可微网络，FIM 通过 JAX autograd 出。
2. **Diffusion-based OED**：用 diffusion policy 生成候选 trajectory，按 FIM 重新加权 sample。
3. **Foundation model 作为 prior**：用 VLM 提供 $\theta$ 的初始 prior（"看起来像个网球，mass 大概 50g"），再用 contact 闭环精化。
4. **Multi-finger dexterous**：高维接触下，FIM 计算成本爆炸，需要 low-rank approximation 或信息几何上的子流形假设。
5. **Variance Upper Bound**：作者提到 FIM 只控制 lower bound。可以用 Bayesian Cramér-Rao (van Trees) 同时控制先验+后验。

### 8.7 与你 (Karpathy) 最近 eureka Moments 系列的潜在联结

你曾提到 "durative contact" 对机器人学习的重要性——机器人要长时间 maintain contact 才能学到物理。这篇 paper 用 FIM 给出"为什么 durative contact 重要"的数学解释：**contact 断裂时 FIM = 0，参数那一刻不更新**。所以最优策略必然 bias 向 durative contact。

反过来，"make and break" 在 stiffness / shape 任务里更优，是因为那些参数对 $\phi_n$ 的 sign change 敏感（穿透→离开切换），FIM landscape 在切换点附近最陡。这是非常优美的物理-信息 dual。

---

## 9. 实操层面：复现路径

1. **环境**：MuJoCo（https://mujoco.org/）+ Python bindings
2. **优化器**：predictive sampling（参考 https://arxiv.org/abs/2212.00541 的实现）
3. **微分**：JAX 自动微分计算 Hessian $\nabla_\theta^2 \mathcal{L}$
4. **机器人**：Franka + Allegro hand 的 URDF（开源可得）
5. **Contact model**：直接按 Eq. (18) 实现，参数从 Table I 取

GitHub repo: https://github.com/ialab-yale/contact_aware_active_learning

复现难点：
- Contact Jacobian $J_c(q)$ 在多接触点同时激活时要正确选择 active set
- FIM Hessian 在接触 complementarity 边界处不光滑，需要 smoothing 或 subgradient
- 实机 transfer 受 contact sensor 噪声影响大（作者用高 noise 验证 robustness）

---

## 10. 总结：这篇 paper 的"营养密度"

| 维度 | 内容 |
|---|---|
| **Theory** | Contact-aware FIM，Theorem 1 把 Fisher-Newton 联系扩展到带约束 contact 系统 |
| **Method** | CA-OED + predictive sampling + 闭环 Bayesian update |
| **Empirical** | 4 个场景，emergent behaviors 与人类 haptic procedures 一致 |
| **Limitations** | deterministic 假设、non-convex、FIM 只控 precision |

这篇 paper 的最大学术贡献是**把 "找好的接触" 这个直觉变成 information geometry 上的最优化问题**，且不需要 RL。对 manipulation、sim-to-real、haptic exploration 都有启发。

后续若要扩展到大模型时代，路径可能是把这种 information-theoretic objective 作为可微 RL 的 auxiliary loss，或者用作 LLM/VLM 提示工程里"action proposal" 的 ranking signal。

---

## 参考链接汇总

- 论文 GitHub: https://github.com/ialab-yale/contact_aware_active_learning
- Predictive Sampling: https://arxiv.org/abs/2212.00541
- ASID Baseline: https://arxiv.org/abs/2404.12308
- MuJoCo: https://mujoco.org/
- Posa Contact-Implicit TO: https://doi.org/10.1177/0278364913506757
- ContactNets: https://proceedings.mlr.press/v155/pfrommer20a.html
- Lederman & Klatzky haptic: https://www.sciencedirect.com/science/article/pii/0001691893900708
- Cramér-Rao Bound: https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound
- Fisher Information: https://en.wikipedia.org/wiki/Fisher_information
- Optimal Design: https://onlinelibrary.wiley.com/doi/abs/10.1002/wics.100
- Atanasov active sensing: https://arxiv.org/abs/1405.2446
- Platt belief space: https://arxiv.org/abs/1209.1991
- Saltation matrices: https://arxiv.org/abs/2406.12923
- Bayesian sim-to-real: https://arxiv.org/abs/2403.16644
- Impact-invariant control: https://arxiv.org/abs/2203.05666
- MuJoCo MPC: https://github.com/google-deepmind/mujoco_mpc

如果要把这套思想直接带到你的 micrograd/makemore 风格的 "tiny robot active learning" 教学 demo 里，我可以帮你画一个 minimal implementation 草图——一个 1D hefting + 一个 2D rubbing，每个 < 200 行 JAX，能 visualize FIM landscape 和 emergent behavior。要不要？
