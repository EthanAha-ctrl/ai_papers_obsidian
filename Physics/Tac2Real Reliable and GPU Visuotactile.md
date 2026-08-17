---
source_pdf: Tac2Real Reliable and GPU Visuotactile.pdf
paper_sha256: aa548d541da526e02f94a7bb195f32e8f7a5daca714de6689c56fcb19d1fbd7a
processed_at: '2026-08-12T12:04:57-07:00'
target_folder: Physics
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Tac2Real

## 一句话版本

让 robot 在 simulation 里学会用"触觉"做精细的装配任务，然后把 skill 直接搬到真实 robot 上就能用，不用重新学。

---

## 为什么这件事难

想象你闭着眼睛把一根 8mm 的 peg 插进 8mm 的孔里。眼睛看不见，全靠手指的感觉——碰到边了、卡住了、滑了一下——这些信息你都要靠触觉来判断。

Robot 想做这件事，得装一个 GelSight Mini 这样的 visuotactile sensor。它长这样：一块软软的 gel pad，上面印着一堆小 marker 点，碰东西的时候 gel 会变形，marker 会跟着移动，robot 就通过看 marker 移动了多少来判断"我碰到了什么、怎么碰的"。

问题来了：你想让 robot 通过 reinforcement learning 学这个任务，RL 需要大量 trial and error，在真实 robot 上跑几百万次显然不现实，所以得在 simulation 里训练。但 simulation 里要模拟那块软 gel 的变形，这件事非常非常麻烦。

---

## Simulation 的三大门派，各有各的毛病

**门派一：TacSL——"装个弹簧凑合用"**

把 gel pad 和物体都当成 rigid body，碰到了就当弹簧压缩一下，用 SDF（signed distance field）算穿透深度。好处是快，GPU 上可以跑几千个 environments。坏处是——它根本没在模拟软体变形，只是算了个"穿了多深"，marker 的位移场完全是假的。你在 sim 里学到的"触觉经验"到了 real world 上完全对不上。

**门派二：Tacchi——"用 MPM 认真模拟但会炸"**

Material Point Method 是搞 soft body simulation 的经典方法，用一堆粒子 + background grid 来算变形。物理上比 TacSL 靠谱多了。但 MPM 有个老大难问题：遇到大的旋转、剪切、或者反复接触的时候，粒子会"飞出去"（particle splashing），数值会爆。你想，RL 训练时候 robot 动作很 random，contact pattern 各种各样，MPM 动不动就给你一个 NaN 的 tactile feedback，policy 学到的全是垃圾信号。

**门派三：TacIPC——"物理最准但太慢"**

Incremental Potential Contact 是目前 soft body contact simulation 的 gold standard，物理上无懈可击——保证无穿透、无翻转、数值稳定。但它用 Newton's method 求解，每一步都要 factorize 一个巨大的 Hessian matrix，还要做 Continuous Collision Detection（CCD）来确保不穿。这些操作本质是 sequential 的，GPU 根本并行不起来。你跑一个 environment 都慢得要死，更别说 4096 个 environments 并行做 RL 了。

**所以现状就是**：快的（TacSL）不真实，真实的（IPC）太慢，中间的（MPM）会炸。没有一个能同时满足"物理够准 + 跑得够快 + 适合 RL"这三个条件。

---

## Tac2Real 的核心 trick：PNCG-IPC

作者的想法是：我不需要每次迭代都算到 machine precision，我只需要"物理上看起来对、数值上稳定"就够了。

具体怎么做的？把 IPC 里的 Newton's method 换成 Nonlinear Conjugate Gradient。

Newton's method 每一步要干什么？
1. 算一个巨大的 Hessian matrix（$n \times n$，n 是 mesh 节点数 × 3）
2. 把它 factorize（Cholesky 分解，$O(n^{1.5})$）
3. 做 CCD line search 确保不穿透

这三步全是 GPU 的噩梦。Sparse matrix factorization 本质是 sequential 的，你没法简单并行。

Conjugate Gradient 每一步要干什么？
1. 算 gradient（element-wise，GPU 完美并行）
2. 取 Hessian 的 diagonal 做 preconditioner（就是 diagonal matrix 的逆，element-wise）
3. 算几个 vector dot product（GPU 最擅长的事）
4. 用一个解析公式算 step size 上界，**直接跳过 CCD**

第四点特别关键。标准 IPC 每一步 line search 都要做 collision detection 来确保"走这一步不会穿"，这个操作非常贵。PNCG-IPC 发现了一个解析的上界：只要步长不超过 $\hat{d} / (2\|\mathbf{p}\|_\infty)$，就保证不会产生新的穿透。这一下就把 CCD 的 $O(\text{所有 contact pair})$ 检测变成了一个 $O(n)$ 的 inf-norm 计算。

**Trade-off 的本质**：每个 CG step 不如 Newton step 精确（线性收敛 vs 二次收敛），但 CG step 在 GPU 上便宜到可以忽略。于是你跑 50 个 CG iterations 的总时间，可能比 1 个 Newton iteration 还短，而 50 步 CG 已经收敛到"足够好"了。对 tactile simulation 来说，"足够好"就是 marker field 看起来对、物理上没穿透、数值稳定，你不需要算到小数点后 10 位。

这个设计哲学其实挺通用的：**在 RL simulation 里，吞吐量比精度重要得多**，因为 policy 对绝对值的精度不敏感，对 signal 的 pattern 和 ordering 敏感。

---

## 系统设计：做成 plugin，不造轮子

Tac2Real 没有自己写一个完整的 physics engine，而是做成一个 plugin，插到 Isaac Lab 里。

流程是这样的：
1. RL agent 发一个 action
2. Isaac Lab 跑一步 rigid body dynamics（robot 怎么动）
3. 提取 sensor 和物体之间的相对位姿和速度
4. 把这些信息扔给 Tac2Real（在单独的 GPU 上跑 PNCG-IPC）
5. Tac2Real 返回 marker displacement field
6. 和 Isaac Lab 的其他 observation 拼在一起，喂回给 RL agent

用 Ray 做 multi-node multi-GPU 分布式。4 个 node，每个 16 块 4090，一共 64 块 GPU，跑 512 个 environments。每个 GPU 负责大约 8 个 environment 的 tactile simulation。

结果：4096 个 environments 能跑到 4465 FPS。这个数字意味着 RL 训练的数据生成速度足够快，不会成为 bottleneck。

---

## Tactile Representation：为什么用 marker 不用 RGB

GelSight Mini 可以输出两种信号：
- RGB tactile image（320×240 = 230400 维）
- Marker displacement field（9×7 个点 × 2D 位移 = 126 维）

作者做了个简单实验：在 4 种 contact mode（静止、下压、前移、后移）下分别记录两种信号。

RGB image：4 种状态下看起来几乎一样，差别微乎其微。
Marker field：4 种状态下明显不同，一眼就能区分。

所以在 RL 场景下：
- Marker field 信噪比高 → policy 学得快
- 维度低 126 vs 230400 → 网络小，训练快
- 不需要 optical rendering pipeline → simulation 省了一大块计算

这个选择很务实。RGB tactile image 对于 texture identification、surface geometry reconstruction 这些 vision task 有用，但对于"判断 contact mode 和 relative pose"这种 RL 最需要的信息，marker field 其实更直接、更 compact。

---

## TacAlign：Sim-to-Real 的四步校准

就算你的 simulation 物理再准，sim 和 real 之间还是会有 gap。作者把这个 gap 分成两类：

**Structured gap**：系统性的差异，比如 robot 动力学不对、材料参数不对、contact model 不对。
**Stochastic gap**：随机噪声，传感器噪声、环境不确定性。

TacAlign 四步走来缩小这些 gap。

### 第一步：调 controller

Franka robot 用 impedance control，简单说就是弹簧阻尼控制：你想让 end-effector 到某个位置，controller 就施加一个力，力的大小跟"离目标多远"成正比，跟"速度"成反比。比例增益 $\mathbf{k}_p$ 是关键参数。

直觉上你会觉得：sim 里的 $\mathbf{k}_p$ 和 real 里的 $\mathbf{k}_p$ 越接近，轨迹差异越小。

**但实验发现根本不是这样**。作者随机采了 20 组 $(\mathbf{k}_p^{sim}, \mathbf{k}_p^{real})$，发现轨迹差异和参数差异之间没有线性关系，甚至不单调。原因很简单：real robot 有 actuator delay、joint friction、unmodeled dynamics，这些非线性效应让"参数匹配"和"轨迹匹配"完全脱钩了。

所以正确的做法是：直接优化轨迹差异。让 sim 和 real 各做 6 种标准动作（3 轴平移 + 3 轴旋转），比较轨迹差异，然后**两边一起调**——先固定 real 调 sim，再固定 sim 调 real，交替迭代。

结果很惊人：初始轨迹差异 11.11mm，比 socket 孔径 8mm 还大，根本不可能完成 insertion。调完之后 2.521mm，而且 sim 的 $\mathbf{k}_p$ 最终是 (600, 50)，real 的是 (400, 20)——差距很大，但轨迹对上了。这就是"参数匹配 ≠ 行为匹配"的最好例证。

### 第二步：调材料参数

PNCG-IPC 有 4 个材料参数：Young's modulus $E$、Poisson's ratio $\nu$、density $\rho$、friction $\mu$。

用 4 种形状的 indenter（cube、cylinder、moon、triangle）去压 real sensor，记录 marker field。然后在 sim 里复现这个过程，调参数让 sim 的 marker field 和 real 的对上。

因为 IPC 不可微（有 barrier function 和 non-smooth friction），用不了 gradient descent，所以用 CMA-ES（一种进化策略优化算法，black-box，不需要 gradient）。80 次迭代收敛。

### 第三步：针对任务调参

前两步是"通用校准"，但真实任务的 contact pattern 更复杂。比如 peg insertion 里有"静止抓握、下压、前撞、后撞"四种典型 contact state，这些在前面的校准里没覆盖到。

所以在真实 peg insertion 任务里记录 marker field，然后调 Isaac Lab 里的 friction coefficient 和 contact stiffness，让 sim 的 marker field 和 real 对上。

**这一步被 ablation 证明是最关键的**。去掉这一步，real-world success rate 从 91.7% 暴跌到 25%。

### 第四步：Domain Randomization

最后加随机化：controller gain 随机、friction 随机、初始位姿随机、observation 加噪声。让 policy 对这些不确定性 robust。

这四步是互补的：前三步消除"系统性偏差"，第四步覆盖"随机性残余"。

---

## 实验结果：数字说话

### Simulation 里的表现

| 方法 | Peg Insertion (sim) | Nut Threading (sim) |
|------|--------------------|--------------------| 
| Tac2Real | 77.6% | 70.2% |
| TacSL | 78.9% | 70.8% |
| Tacchi | 17.3% | 15.2% |
| No Tactile | 16.8% | 31.3% |

在 sim 里 Tac2Real 和 TacSL 差不多。Tacchi 很差，因为 MPM 数值不稳定给 policy 提供了大量 garbage tactile feedback。No Tactile 也很差，说明 tactile 确实是关键信息。

### Real-world 的表现（关键！）

| 方法 | Real Peg Insertion |
|------|-------------------|
| **Tac2Real (full TacAlign)** | **91.7%** |
| TacSL | 15.0% |
| Tacchi | 8.3% |
| No Tactile | 6.7% |

这就是 paper 的 punchline：sim 里 TacSL 和 Tac2Real 差不多，但搬到 real 上，TacSL 从 78.9% 掉到 15%，Tac2Real 从 77.6% 变成 91.7%。

为什么会这样？因为 TacSL 的 tactile simulation 是假的（penalty-based + SDF），policy 在 sim 里学的是"如何利用假触觉信号"。到了 real world，触觉信号长得完全不一样，policy 就废了。Tac2Real 的 PNCG-IPC 物理上足够真实，sim 里学到的"触觉→动作"mapping 在 real 上依然成立。

### TacAlign Ablation

| 去掉哪步 | Success Rate |
|---------|-------------|
| 全部保留 | 91.7% |
| 去掉 task calibration | 25.0% |
| 去掉 baseline calibration | 53.3% |
| 去掉 randomization | 76.7% |

Task calibration 最关键，baseline calibration 其次，randomization 也有明显贡献。Controller alignment 在这个 ablation 表里没单独列出来，但前面 trajectory alignment 的数据（11.11mm → 2.52mm）已经很说明问题了。

---

## 我的几点直觉性理解

### 1. "Sim 里表现好" 和 "能 sim-to-real" 是两码事

TacSL 在 sim 里 78.9% 成功率，看起来跟 Tac2Real 差不多。但这完全不能预测 real-world 表现。真正决定 sim-to-real 能不能成的是 **simulation 的物理保真度**，不是 sim 里的 success rate。

这个 insight 其实挺深刻的。很多人做 sim-to-real 的时候只盯着 sim success rate 看，觉得"sim 里能 work 就行了"，结果一到 real 就崩。Tac2Real 的对比实验完美演示了为什么 simulation fidelity 才是根本。

### 2. "够用就好"的物理精度

PNCG-IPC 不追求 machine precision，只追求"物理一致 + 数值稳定"。这在 RL 场景下完全够用，因为 policy 学的是 mapping 的 pattern，不是绝对值。

这个思路其实可以推广到很多地方。比如 fluid simulation for RL、deformable object manipulation for RL，都可以考虑"牺牲 per-iteration 精度换 throughput"的策略。Newton 很优雅，但 GPU 不喜欢；CG 不够优雅，但 GPU 很喜欢。

### 3. Sim-to-real 不是一步到位的

TacAlign 的四步校准说明了一个现实：sim-to-real 没有 silver bullet。你不能指望"把 physics engine 写对就行了"或者"加点 domain randomization 就行了"。它是一个系统工程，需要从 controller、material、task-specific contact、randomization 多个层面系统性地消除 gap。

而且每一步的"校准"都不是简单的参数匹配，而是**行为匹配**。Controller 参数匹配那段特别有意思——$\mathbf{k}_p^{sim}$ 和 $\mathbf{k}_p^{real}$ 最终差了 200，但轨迹对上了。这说明在非线性系统里，"参数空间里的近"和"行为空间里的近"完全不同。

### 4. 触觉是被低估的 modality

No tactile 的 real-world success rate 只有 6.7%，有触觉的 91.7%。差距是 14 倍。对于 peg-in-hole 这种 contact-rich 任务，视觉在 peg 进入 socket 之后基本就废了（看不见），纯靠盲操作几乎不可能完成。触觉是唯一的 signal source。

这让我想到，未来的 robot learning 可能需要重新审视 sensor modalities 的优先级。Vision 当然重要，但在 contact-rich manipulation 里，tactile 可能是 make-or-break 的 factor。

### 5. 这个框架的 generalizability

Paper 最后提到可以扩展到 dexterous hand、deformable object、甚至 tactile-based VLA。我觉得这是合理的：

- PNCG-IPC 本身是一个 general soft body solver，不限于 GelSight Mini 这种 sensor
- Plugin 架构意味着可以插到任何 physics engine 里
- TacAlign 的方法论（controller alignment → material calibration → task calibration → DR）是一个通用 recipe

唯一的不确定性是：对于更复杂的 task（比如 dexterous hand in-hand manipulation），contact pattern 远比 peg-in-hole 复杂，task calibration 那一步可能需要更精细的设计。

---

## Links

- [Tac2Real Project Page](https://ningyurichard.github.io/tac2real-project-page/)
- [PNCG-IPC Paper (SIGGRAPH 2024)](https://dl.acm.org/doi/10.1145/3641519.3641529)
- [Original IPC Paper (ACM TOG 2020)](https://graphics.stanford.edu/papers/ipc/)
- [Isaac Lab](https://arxiv.org/abs/2511.04831)
- [Taichi Lang](https://github.com/taichi-dev/taichi)
- [TacSL (IEEE T-RO 2025)](https://arxiv.org/abs/2501.10421)
- [Tacchi (RA-L 2023)](https://arxiv.org/abs/2209.09268)
- [Taccel (concurrent IPC scaling work)](https://arxiv.org/abs/2504.12908)
- [Difftactile (differentiable tactile sim)](https://arxiv.org/abs/2403.08716)
- [rl-games](https://github.com/Denys88/rl_games)
- [Ray Framework](https://www.ray.io/)
- [GelSight Sensors](https://gelsightinc.com/)
- [CMA-ES Tutorial](https://link.springer.com/chapter/10.1007/3-540-32494-1_4)

---

# Tac2Real: 为 Online RL 和 Zero-Shot Sim-to-Real 设计的 Visuotactile Simulation 框架

这篇 paper 来自 Shanghai AI Lab 和 HKUST，核心解决的是 visuotactile sensor 在 online reinforcement learning 训练场景下的 sim-to-real gap 问题。我把它拆成几个层次来讲，从底层 physics solver 一直到系统层面的 sim-to-real pipeline。

---

## 1. 问题背景：为什么 tactile simulation 这么难做

Visuotactile sensor（比如 GelSight Mini）本质上是一块覆盖着 markers 的 elastomer，接触物体时会发生大形变，需要高保真地模拟这种 soft body 的 deformation + frictional contact。现有方法有三类：

| 方法 | Physics | 问题 |
|------|---------|------|
| TacSL / penalty-based | Non-physics, SDF + spring | 失真严重，只建模 interpenetration region |
| Tacchi / MPM | Material Point Method | 大旋转/剪切下 numerical instability, particle splashing |
| TacIPC / IPC | Incremental Potential Contact | Newton + CCD line search, GPU 不友好 |

**Online RL 的矛盾点**：RL 需要 4096+ environments 并行，每秒数千 FPS。但 high-fidelity physics solver（比如 IPC）本身是 Newton 迭代 + Continuous Collision Detection（CCD），CCD 每个 iteration 都要做，几乎不可能在 GPU 上高效并行。这就是 Tac2Real 要破解的核心瓶颈。

---

## 2. PNCG-IPC：用 Conjugate Gradient 替代 Newton 的关键设计

这是 paper 最关键的技术创新。先看原始 IPC 的 formulation（Eq 1, Eq 7-8）：

$$E(\mathbf{x}) = \underbrace{\frac{1}{2}(\mathbf{x} - \hat{\mathbf{x}})^\top \mathbf{M}(\mathbf{x} - \hat{\mathbf{x}})}_{\text{inertia potential}} + \underbrace{h^2 \Psi(\mathbf{x})}_{\text{hyperelastic energy}} + \underbrace{\kappa \sum_{k \in C} b(d_k(\mathbf{x}))}_{\text{log-barrier contact } B(\mathbf{x})} + \underbrace{D(\mathbf{x})}_{\text{friction dissipation}}$$

变量含义：
- $\mathbf{x} \in \mathbb{R}^{3n}$: 当前 timestep 所有 mesh 节点的位置（n 个 tetrahedron 节点）
- $\hat{\mathbf{x}} = \mathbf{x}^t + h \mathbf{v}^t$: inertial prediction，上一时刻位置 + 速度×步长
- $\mathbf{M}$: diagonal mass matrix（lumped mass）
- $h$: time step size
- $\Psi(\mathbf{x})$: hyperelastic potential（Neo-Hookean 或 SNH）
- $d_k(\mathbf{x})$: 第 k 对 contact primitive 的 distance
- $\kappa$: barrier stiffness（自适应增大）
- $b(\cdot)$: $C^2$ log-barrier function，当 $d_k < \hat{d}$ 时激活
- $\hat{d}$: contact activation threshold

Friction dissipation potential（Eq 9）采用 smoothed Coulomb：

$$D(\mathbf{x}) = \mu_f \sum_{k \in C} \lambda_k^n f(\|\mathbf{T}_k \Delta \mathbf{x}_k\|)$$

其中 $\lambda_k^n = -\kappa b'(d_k)$ 是从 barrier gradient 求出的 normal contact force magnitude，$\mathbf{T}_k$ 是切平面投影算子，$\Delta \mathbf{x}_k$ 是当前步内的切向位移，$f(\cdot)$ 是 $C^1$ smooth mollifier（Eq 10）来近似 non-smooth Coulomb cone。

**标准 IPC 的瓶颈**：用 Newton 法求解 $\min_x E(x)$，每步需要：
1. Assemble Hessian $\mathbf{H} = \mathbf{M} + h^2 \nabla^2 \Psi + \kappa \sum \nabla^2 b + \nabla^2 D$
2. Factorize $\mathbf{H}$（$O(n^3)$ 或者稀疏 Cholesky $O(n^{1.5})$）
3. CCD-based line search：每次 line search 都做 collision detection

这三步在 GPU 上都不友好——sparse matrix factorization 本质是 sequential 的，CCD 也很难并行。

**PNCG-IPC 的设计**（Eq 11-12）：用 Dai-Kou nonlinear conjugate gradient 替代 Newton：

$$\mathbf{p}_{k+1} = -\mathbf{P}_{k+1} \mathbf{g}_{k+1} + \beta_k^{DK} \mathbf{p}_k$$

$$\beta_k^{DK} = \frac{\mathbf{g}_{k+1}^\top \mathbf{P}_{k+1} \mathbf{y}_k}{\mathbf{y}_k^\top \mathbf{p}_k} - \frac{(\mathbf{y}_k^\top \mathbf{P}_{k+1} \mathbf{y}_k)(\mathbf{p}_k^\top \mathbf{g}_{k+1})}{(\mathbf{y}_k^\top \mathbf{p}_k)^2}$$

其中：
- $\mathbf{g}_k = \nabla E(\mathbf{x}_k)$: gradient
- $\mathbf{y}_k = \mathbf{g}_{k+1} - \mathbf{g}_k$: gradient difference
- $\mathbf{P} = \text{diag}(\mathbf{H})^{-1}$: **Jacobi preconditioner**，只取 Hessian 的对角线
- $\mathbf{p}_k$: conjugate search direction

关键 trick 在 step size（Eq 12）：

$$\alpha_{\text{upper}} = \frac{\hat{d}}{2 \|\mathbf{p}_{k+1}\|_\infty}, \quad \bar{\alpha} = -\frac{\mathbf{g}_{k+1}^\top \mathbf{p}_{k+1}}{\mathbf{p}_{k+1}^\top \mathbf{H}_{k+1} \mathbf{p}_{k+1}}, \quad \alpha = \min(\alpha_{\text{upper}}, \bar{\alpha})$$

**$\alpha_{\text{upper}}$ 的解析上界**直接替代 CCD！只要步长不超过 $\hat{d} / (2\|\mathbf{p}\|_\infty)$，就保证不会产生新的 penetration。这个 trick 把 CCD 的 $O(\text{primitive pairs})$ 检测变成了 $O(n)$ 的 inf-norm 计算，完全 GPU 友好。

**Trade-off 的核心 insight**：
- Newton：每 iteration 精度高，但单步成本 $O(n^{1.5})$，难并行
- PNCG：每 iteration 精度低（线性收敛 vs 二次收敛），但只需要 gradient + diagonal Hessian + vector dot products，全部都是 GPU 上的 element-wise 操作
- 对 tactile simulation 来说，物理 consistency 比 machine precision 重要，几十次 CG iteration 就够

---

## 3. 系统架构：Multi-Node Multi-GPU 的 Plugin 设计

Tac2Real 不重写 physics engine，而是作为 Isaac Lab / MuJoCo 的 plugin（Fig 3）：

```
RL Agent → Isaac Lab physics step (rigid body dynamics)
       ↓ 
   Extract relative pose between sensor & object
       ↓
   Compute linear/angular velocity
       ↓
   Tac2Real (PNCG-IPC, Taichi kernel) on dedicated GPU cluster
       ↓
   Output marker displacement field (9×7×2)
       ↓
   Combine with EE pose → observation → RL Agent
```

并行架构用 Ray [29] cluster：
- 每个节点 16× RTX 4090
- 每个 GPU 跑一个 Ray-wrapped tactile simulation class，负责一组 environments
- Roll-out 时 iteratively 调用，Ray 跨节点 gather 结果

性能数据（Fig 5c）：
- 4096 environments，**4465 FPS**（pseudo-structured mesh）
- 1665 FPS（unstructured mesh）
- Tacchi 在同样 setup 下效率低，因为 MPM 需要为多 env 共享大 background grid，GPU 利用率低
- TacSL FPS 更高但 physics fidelity 不行（penalty-based + SDF query）

---

## 4. Tactile Representation 选择：Marker Field vs RGB

这是一个值得注意的设计决策。GelSight Mini 可以输出 RGB image (320×240) 或 marker displacement field (9×7)。

Paper Fig 2 做了对比实验，4 种 contact mode（stationary, press-down, move-forward, move-backward）下：
- RGB image：差异很小，对 texture identification 有用，但对 contact state 区分度低
- Marker displacement field：4 种状态下有显著差异，且维度低（9×7×2 = 126 维 vs RGB 230400 维）

在 RL 中：
1. 低维度 → 更强的 regularity
2. 高信噪比对 contact mode → 更强的 signal
3. 模拟成本低（不需要 optical rendering pipeline）

他们用 k-nearest neighbor 把 IPC mesh node positions 加权插值到 marker 位置。

---

## 5. TacAlign：Sim-to-Real 的系统化校准框架

这是 paper 的另一大贡献。Sim-to-real gap 分两类：
- **Structured**：robot dynamics mismatch, material params, contact model mismatch
- **Stochastic**：unmodeled noise, environment uncertainty

TacAlign 四阶段（Fig 4）：

### Level 1: Robot Control Alignment

Franka 用 Cartesian impedance control：

$$\mathbf{F}^{targ} = \mathbf{k}_p * (\mathbf{p}^{targ}(\mathbf{a}) - \mathbf{p}^{ee}) - \mathbf{k}_d * \mathbf{v}^{ee}$$

其中 $\mathbf{k}_d = 2\sqrt{\mathbf{k}_p}$ 保证 critical damping，$\mathbf{k}_p \in \mathbb{R}^6$（3 translation + 3 rotation gains）。

**关键 empirical observation**（Fig 4 upper left）：随机采 20 组 $(\mathbf{k}_p^{sim}, \mathbf{k}_p^{real})$，发现 trajectory discrepancy 和 $|\mathbf{k}_p^{sim} - \mathbf{k}_p^{real}|$ **没有线性关系**，甚至非单调。这说明由于 unmodeled dynamics, actuator delay, friction, contact nonlinearity，单纯匹配 controller 参数不行，必须做 trajectory-level alignment。

定义 trajectory discrepancy：

$$\mathcal{D}(\mathbf{k}_p^{sim}, \mathbf{k}_p^{real}) = \frac{1}{T} \sum_{t=1}^{T} \|\mathbf{x}_t^{sim} - \mathbf{x}_t^{real}\|^2$$

在 6 种 canonical motion（3 axis translation + 3 axis rotation）上计算，然后用 **alternating minimization**（Eq 3-4）：

$$\mathbf{k}_p^{sim,(k+1)} = \arg\min_{\mathbf{k}_p^{sim}} \mathcal{D}(\mathbf{k}_p^{sim}, \mathbf{k}_p^{real,(k)})$$

$$\mathbf{k}_p^{real,(k+1)} = \arg\min_{\mathbf{k}_p^{real}} \mathcal{D}(\mathbf{k}_p^{sim,(k+1)}, \mathbf{k}_p^{real})$$

两边同时调，而不是把 sim 当 ground truth 或 real 当 ground truth。最终结果（Tab 2）：
- 初始：$\bar{\mathcal{D}}_{trans} = 11.11$ mm（比 socket 孔径 8mm 还大，根本做不了 insertion）
- 优化后：$\bar{\mathcal{D}}_{trans} = 2.521$ mm, $\bar{\mathcal{D}}_{rot} = 0.454°$
- 对应 gains：$\mathbf{k}_p^{sim} = (600, 50)$, $\mathbf{k}_p^{real} = (400, 20)$

### Level 2: Baseline IPC Calibration

PNCG-IPC 的 4 个 material 参数需要标定：Young's modulus $E$, Poisson's ratio $\nu$, density $\rho$, friction coefficient $\mu$。

参数范围（Tab 4）：
- $E \in [10^4, 2 \times 10^5]$
- $\nu \in [0.4, 0.497]$（接近 incompressible，橡胶类材料）
- $\rho \in [10^{-3}, 5 \times 10^{-3}]$
- $\mu \in [0.25, 2.5]$

用 4 个 indenter（cube, cylinder, moon, triangle），每个做 3 种 deformation（press 1mm, slide 1mm, rotate 2°）。

Loss function（Eq 5）：

$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{K \cdot N} \sum_{k=1}^{K} \sum_{i=1}^{N} \|\mathbf{u}_{k,i}^{sim}(\boldsymbol{\theta}) - \mathbf{u}_{k,i}^{real}\|_2^2$$

其中 $K$ 是总帧数，$N$ 是 indenter 数量，$\mathbf{u}_{k,i}$ 是第 $i$ 个 indenter 第 $k$ 帧的 marker displacement field。

用 **CMA-ES**（gradient-free black-box optimization）求解（Eq 6），因为 IPC 的 gradient 不可微（CCD + non-smooth friction）。popsize=12，80 iterations 收敛（Fig 11）。

### Level 3: Task-based Calibration

Baseline calibration 不够，因为真实 task 的 contact mode 比简单 indentation 复杂。针对 4 种 task-specific contact state（stationary grasping, press-down, forward/backward collision）fine-tune Isaac Lab 的 $\mu_{isaac}$ 和 contact stiffness $s_{isaac}$，直到 MSE 低于 threshold。

**Ablation 显示这是最关键的一步**（Tab 3）：去掉 Level 3（保留 1,2,4）success rate 从 91.7% 掉到 25%。

### Level 4: Domain Randomization

随机化参数（Tab 5）：
- Controller gains $\mathbf{k}_p \in [400, 800]$
- Peg friction $\in [0.5, 1.0]$
- Socket 初始位置 X 方向 ±5cm
- Holding 初始 orientation $Y$-rot $\in [-35°, 35°]$
- EE pose noise: translation 5mm, rotation 0.2 rad
- IPC perturbation: translation 1mm, rotation 0.05 rad

Episode-level 随机化初始状态，timestep-level 随机化 observation。

---

## 6. 实验：Peg-in-Hole 的 Zero-Shot Transfer

### 任务设定

- Peg 和 socket 直径都是 **8mm**（比 TacSL 的 real-world 设置还小，更难）
- 初始 orientation 随机采样 $[-35°, 35°]$
- **纯盲**：observation 只有 EE pose (7D) + marker field (9×7×2=126D) + previous action，没有 object pose，没有 vision
- Reward: $r_{total} = r_{keypoints} + r_{engage/success} - r_{contact}$

### RL 设置

- PPO from rl-games，shared actor-critic
- LSTM（2 层，1024 units）+ MLP（512, 128, 64, ELU）
- 512 environments across 4 nodes × 16 GPUs
- Horizon 256, minibatch 512, mini-epochs 4, sequence length 128

### 结果（Tab 3）

| Setup | Tac2Real | TacSL | Tacchi | No Tactile |
|-------|----------|-------|--------|-----------|
| Sim Peg Insertion | 0.776 | 0.789 | 0.173 | 0.168 |
| Sim Nut Threading | 0.702 | 0.708 | 0.152 | 0.313 |
| **Real Peg Insertion (full TacAlign)** | **0.917** | 0.150 | 0.083 | 0.067 |
| Real (no task cal) | 0.250 | - | - | - |
| Real (no baseline cal) | 0.533 | - | - | - |
| Real (no randomization) | 0.767 | - | - | - |

**关键观察**：
1. Sim 阶段 Tac2Real 和 TacSL 表现相近（都 ~78%），但 sim-to-real 后 Tac2Real 91.7% vs TacSL 15%——这说明 **sim 阶段高 success rate 是必要但不充分条件**，physics fidelity 决定了 policy 能否 transfer
2. TacSL 在 sim 能 train 出好 policy，但 sim 中的 marker field 失真（penalty-based 只看 interpenetration），policy 学到的是 "假" tactile signal 的利用方式，real 上完全 invalid
3. Tacchi 在 sim 性能就差，因为 MPM 在 contact-rich 场景下 numerical instability，提供大量 noisy tactile feedback
4. No tactile 6.7%，说明 tactile 是这个 task 的核心 signal

### Real-world Deployment 细节

- 60 trials（$0°$, $+15°$, $-15°$ 各 20 trials）
- **55 次成功**，91.7% zero-shot success rate
- 保护机制：连续帧 marker field MSE 超阈值 → halt inference, 回退一步, 重新 inference（防止 gel pad 损坏）

---

## 7. 几个值得深入思考的点

### 7.1 为什么 PNCG 的"低精度高吞吐" trade-off 在 RL 中成立

RL policy 实际上对 tactile signal 的 **rank/ordering** 敏感，对绝对值的 machine precision 不敏感。PNCG 几十次 iteration 的物理 consistency 已经足够让 policy 学到 "marker field 这个 pattern → 这个 action" 的 mapping。这个 insight 可能可以推广到其他 physics-based RL simulation。

### 7.2 Marker Field vs RGB Tactile Image 的选择

Paper 选择了低维 marker field，理由是信噪比和维度。但这也放弃了 texture 信息。对于需要识别 surface texture 的 task（比如判断材质），marker field 可能不够。未来的工作可能需要 hierarchical representation，或者用 small CNN encoder 处理 RGB + marker 一起。

### 7.3 Alternating Minimization 的控制校准

这个设计很有意思——传统 sim-to-real 默认 real 是 ground truth，sim 去拟合 real。但这里两边都调，因为 real robot 本身也有 unmodeled dynamics（actuator delay, joint friction），real 的 $\mathbf{k}_p$ 也不是"真"值。这种 bidirectional calibration 可能在其他 sim-to-real 场景也有价值。

### 7.4 PNCG-IPC 在 large-scale RL 之外的应用

Paper 提到未来可以扩展到 dexterous hand manipulation, deformable object, 甚至 tactile-based VLA training。PNCG-IPC 的 GPU-friendly 特性也适合做 differentiable simulation 的 backend（虽然 paper 没做 differentiation，但 CG 本身可以做 adjoint method）。

---

## 8. 与相关工作的对比

| Method | Physics | Multi-GPU | Online RL | Sim2Real |
|--------|---------|-----------|----------|----------|
| Tacchi [6] | MPM | ×（共享 grid） | 部分 | 部分 |
| TacIPC [10] | IPC + Newton | × | × | × |
| Difftactile [32] | MPM + FEM | × | ×（differentiable） | 部分 |
| TacSL [2] | Penalty-based | ✓ | ✓ | 部分（fidelity 低） |
| TacFlex [47] | FEM | ✓ | BC only | ✓ |
| Taccel [24] | IPC | ✓ | × | ✓ |
| **Tac2Real** | **PNCG-IPC** | **✓** | **✓** | **✓** |

Tac2Real 是目前唯一一个在 high-fidelity physics + multi-GPU + online RL + zero-shot sim2real 四个维度都达标的 framework。

---

## 9. 个人联想 & 可能的扩展方向

1. **GPU cluster 利用率**：Paper 用 4×16=64 个 GPU 跑 512 envs，每个 GPU ~8 envs。如果用更大的 cluster（比如 256 GPU）能否 scale 到 4096 envs full fidelity？Ray 的 communication overhead 在大规模下是否会成为瓶颈？

2. **Learned physics surrogate**：PNCG-IPC 几十次 iteration 还是比纯 penalty-based 慢。能否用 neural surrogate（比如 GNN-based simulator）学习 PNCG-IPC 的输出，做 amortized simulation？这就是 paper conclusion 里提到的 "AI models to accelerate tactile simulation"。

3. **Tactile representation learning**：目前直接用 raw marker field (126D)。能否 pretrain 一个 tactile encoder（contrastive learning on marker sequences）提取更 compact 的 latent？这可能让 RL 更 sample efficient。

4. **Cross-embodiment transfer**：Franka + GelSight Mini 的 calibration 能否 transfer 到其他 gripper + 其他 sensor？TacAlign 的方法论（trajectory alignment + material calibration + task calibration + DR）应该可以泛化，但需要重新跑 Level 1-3。

5. **VLA integration**：Paper 提到可以为 tactile-based VLA 生成数据。想象一下：大规模 PNCG-IPC 生成 (vision, tactile, action) tuples，pretrain 一个 multimodal VLA，再加少量 real data finetune——这可能比纯 vision VLA 在 contact-rich task 上强很多。

---

## References & 进一步阅读

- **PNCG-IPC 原文**: Shen et al., "Preconditioned nonlinear conjugate gradient method for real-time interior-point hyperelasticity", ACM SIGGRAPH 2024. https://dl.acm.org/doi/10.1145/3641519.3641529
- **Original IPC**: Lan et al., "Incremental Potential Contact", ACM TOG 2020. https://graphics.stanford.edu/papers/ipc/
- **Tac2Real Project Page**: https://ningyurichard.github.io/tac2real-project-page/
- **Isaac Lab**: https://arxiv.org/abs/2511.04831
- **Taichi programming language**: https://github.com/taichi-dev/taichi
- **TacSL**: https://arxiv.org/abs/2501.10421 (IEEE T-RO 2025)
- **Tacchi**: https://arxiv.org/abs/2209.09268 (RA-L 2023)
- **Taccel** (concurrent work, scaling IPC): https://arxiv.org/abs/2504.12908
- **Difftactile** (differentiable tactile sim): https://arxiv.org/abs/2403.08716
- **CMA-ES**: Hansen, "The CMA Evolution Strategy: A Comparing Review" https://link.springer.com/chapter/10.1007/3-540-32494-1_4
- **rl-games**: https://github.com/Denys88/rl_games
- **Ray**: https://www.ray.io/
- **GelSight sensors**: https://gelsightinc.com/
- **FrankaPy**: https://github.com/iamhankai/franka-interface

---

总结一下，这篇 paper 的核心 insight 是：**在 tactile simulation for online RL 这个 niche 里，per-iteration 的高精度（Newton + CCD）远不如 high-throughput + 物理一致（PNCG + CCD-free）重要**。配合一个系统化的 4-stage sim-to-real calibration pipeline，第一次把 high-fidelity IPC-based tactile simulation 推到了可以 zero-shot sim2real 的程度。这对于 contact-rich manipulation 这个长期难题是一个很 solid 的 milestone。
