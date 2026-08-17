---
source_pdf: stable-worldmodel-V1 REPRODUCIBLE WORLD MODELING.pdf
paper_sha256: 52bd6b3194034879efad3e2a8e3154aaf7abb07640c5b78e9fe743fed7cebeb2
processed_at: '2026-08-12T10:36:40-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

人话讲，这篇 paper 就是为 World Model 领域造了一个 OpenAI Gym 或者 ImageNet。之前大家做 World Model 研究，每个 lab 都自己从头写环境、写数据收集、写评估代码。这导致一个尴尬的局面：你发了一篇 paper 说你的 model 成功率 90%，我发了一篇说我的 85%，根本没法横向对比，因为连跑的 environment code 都不一样。比如 PLDM (https://openreview.net/forum?id=jON7H6A9UU) 和 DINO-WM (https://arxiv.org/abs/2501.17172) 这两篇 top paper 跑的都是同一个叫 Two-Room 的简单 2D 导航任务，代码 diff 出来 81 处删除，86 处增加。同样的东西，写法完全不一样。SWM (stable-worldmodel) 的出现就是把这些底层轮子全包了，让大家专注于写 model。

下面我尽量用人话结合底层的硬核技术细节，帮你 build 起对这个东西的 intuition。

### 1. API 设计的“人话”拆解

SWM 的核心抽象叫 `World`。它和 Gymnasium (https://arxiv.org/abs/2407.17032) 的 API 有个巨大的区别。Gymnasium 里你调用 `env.step(action)`，把 action 喂进去。在 SWM 里，`world.step()` 括号里是空的。

它怎么拿 action？你提前挂载一个 policy（`world.set_policy(your_policy)`），step 的时候 world 会自动去问 policy 要 action。

这么做有极强工程直觉。你做 robot learning，同一个环境你要跑三遍：第一遍用 expert policy 收集数据，第二遍用 random policy 收集 OOD 数据，第三遍挂上你的 MPC policy 跑评估。如果用老 API，换 policy 就得改 step 循环里的代码。SWM 里你只换 `set_policy` 里的对象，控制逻辑和环境执行彻底解耦。数据全存在 `world.infos` 这个 dict 里原地更新，省了海量 GC 开销。

### 2. FoV (Factors of Variation)：SWM 的灵魂

SWM 真正的灵魂是 FoV。用人话理解，就是环境暴露给你一堆旋钮。推 T 块的任务 PushT (https://arxiv.org/abs/2303.04137) 里有 16 个旋钮，比如 agent 的颜色、block 的形状、背景颜色。

技术上讲，标准 MDP 定义为 $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$。
- $\mathcal{S}$ 代表 state space
- $\mathcal{A}$ 代表 action space
- $P: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$ 代表 transition kernel
- $R: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ 代表 reward function
- $\gamma \in [0, 1)$ 代表 discount factor

SWM 引入了一个环境参数向量 $\xi \in \Xi \subset \mathbb{R}^d$。MDP 变成了一个参数化的族：
$$\mathcal{M}(\xi) = (\mathcal{S}(\xi), \mathcal{A}, P(\xi), R(\xi), \gamma)$$

这里 $P(\xi)$ 代表 dynamics 随 friction、mass 变化；$\mathcal{S}(\xi)$ 代表 state space 随 wall thickness 变化。这把 Domain Randomization (https://arxiv.org/abs/1703.06907) 和 Continual Learning 的研究统一到了一个极其干净的 API 上。你只要写 `options={"variation": ["agent.color"]}`，环境每次 reset 就自动随机化 agent 颜色。

### 3. 规划算法：MPC 与 CEM 的数学直觉

SWM 的评估怎么跑？它用 Model Predictive Control (MPC)。里面有个 CEM (Cross-Entropy Method) solver。

人话讲，就是每次我要决定怎么走，我先随机试 300 条路径，用我的 World Model 在脑子里模拟走一遍，看看哪条路离 goal 最近。把表现最好的前 K 条挑出来，算个平均值，下一步就往那个方向走。

公式上，假设 planning horizon 是 $H$，action dimension 是 $d_a$。
我们维护一个高斯分布 $\mathcal{N}(\mu, \Sigma)$，其中 $\mu \in \mathbb{R}^{H \cdot d_a}$ 代表 action 序列的均值向量，$\Sigma \in \mathbb{R}^{H \cdot d_a \times H \cdot d_a}$ 代表协方差矩阵。
迭代过程：
1. 采样 $N$ 个 action sequences: $\mathbf{a}^{(i)} \sim \mathcal{N}(\mu, \Sigma)$，其中 $i = 1, ..., N$。
2. 计算每条 sequence 的 cost：$J(\mathbf{a}^{(i)}) = \sum_{t=1}^{H} c(\hat{s}_t^{(i)}, a_t^{(i)}, g)$，这里 $\hat{s}_t^{(i)}$ 是 World Model 预测的未来 state，$c$ 是计算到 goal 距离的 cost function。
3. 选出 cost 最小的 $K$ 个样本组成 elite set $E$。
4. 更新分布参数：$\mu \leftarrow \frac{1}{K} \sum_{i \in E} \mathbf{a}^{(i)}$

这个过程在 SWM 里被封装成了 `CEMSolver(model=world_model, num_samples=300)`。你只要写一行代码，底层这套基于数学分布的规划就跑起来了。

### 4. 核心实验：打脸 DINO-WM 的 robustness

paper 里做了一个极其打脸的实验。他们用 SWM 复现了 DINO-WM。DINO-WM 的核心是用 DINOv2 (https://arxiv.org/abs/2304.07193) 提取 visual features，在 latent space 里直接做 planning，号称 zero-shot 效果极好。

SWM 团队复现后，在正常 expert 数据下测，success rate 确实有 94.0%。接着他们拨动 SWM 的 FoV 旋钮，做 zero-shot robustness 测试。

实验数据表如下：

| FoV | Property | SR % (↑) |
|---|---|---|
| None | (Baseline) | 94.0 |
| Color | Agent | 18.0 |
| Color | Background | 10.0 |
| Size | Agent | 4.0 |
| Angle | Agent | 12.0 |
| Position | Anchor | 4.0 |

数据极其惨烈。仅仅是把 agent 的颜色或者大小改了一下，success rate 从 94% 直接跌到 4% 到 18%。

这里面的直觉在于，DINOv2 在 ImageNet 上确实对 color jitter 鲁棒，但在这种纯色背景的 manipulation 任务里，visual distribution 完全 OOD (Out-of-Distribution)。DINO-WM 的所谓 zero-shot 能力被死死锁在了它训练数据的极窄分布里。换一点点视觉外观，latent space 的距离度量 $\|z - z_g\|^2$ （其中 $z$ 代表当前 latent state， $z_g$ 代表 goal latent state）就完全失效了。

### 5. 写给 Andrej 的联想

Andrej，如果你要用这套东西，结合你之前 V-JEPA 2 (https://arxiv.org/abs/2506.04985) 的工作，路径非常清晰。V-JEPA 2 是直接在 video 上做 latent predictive model，本身就是一种 World Model。你可以拿 V-JEPA 2 的 encoder 换掉 DINO-WM 里的 DINOv2，挂到 SWM 的 CEM solver 上跑一遍这个 FoV 测试。因为 V-JEPA 2 原生处理 video dynamics，极有可能在这个 robustness 测试上把 DINO-WM 按在地上摩擦，这是一个直接能发 paper 的实验。

整体来看，SWM 就是一个工业级、高 test coverage、带文档的 research ecosystem。它把 World Model 研究里最脏最乱的 environment 和 evaluation 层标准化了，让真正搞算法的人可以专注在 latent representation 和 dynamics learning 上。

---

# stable-worldmodel (SWM): 一个面向 World Model 研究的可复现基础设施

你好 Andrej。这篇 paper 我读下来感觉它的核心贡献其实不在于算法创新，而在于把 World Model 这个领域从"每个 lab 各自为政"的状态往标准化基础设施方向推进。它和你当年推动 ImageNet、CS231n、micrograd 的精神是一致的——降低 idea-to-experiment 的摩擦。下面我尽量展开讲，包括技术细节、设计权衡，以及与已有工作的连接。

---

## 1. 核心动机：为什么 World Model 需要"基础设施层"

### 1.1 领域 fragmentation 的量化证据

paper 用一个非常具体的例子论证了问题严重性：PLDM (Sobal et al., 2025, https://openreview.net/forum?id=jON7H6A9UU) 和 DINO-WM (Zhou et al., 2025, ICML 2025, https://arxiv.org/abs/2501.17172) 是近期两篇 latent world model 工作，它们都重新实现了同一个 Two-Room 环境，但 diff 结果显示 **81 deletions, 86 additions, 18 updates**——同一个简单 2D 导航任务，两套独立实现几乎完全不同。这种碎片化意味着：

- 跨 paper 的数字不可比（你不知道差异来自方法还是实现）
- bug 难以被 community 发现和修复
- 新人入门成本极高
- generalization 的研究根本无从谈起，因为没有 controlled factor

类比到你熟悉的领域：vision 有 ImageNet (Russakovsky et al., 2015, https://arxiv.org/abs/1409.0575)、COCO (Lin et al., 2014)、RL 有 Atari ALE (Bellemare et al., 2013, https://arxiv.org/abs/1207.4708)、OpenAI Gym (Brockman et al., 2016, https://arxiv.org/abs/1606.01540)、DMC (Tassa et al., 2018, https://arxiv.org/abs/1801.00690)，language modeling 有 MMLU (Hendrycks et al.)、MMLU-Pro (Wang et al., 2024)、Humanity's Last Exam (Phan et al., 2025, https://arxiv.org/abs/2501.14249)。World model 目前缺少这种 canonical shared substrate。Ha & Schmidhuber 2018 (https://arxiv.org/abs/1803.10122) 提出概念之后七年，社区依然没有收敛到统一平台。

### 1.2 SWM 的定位哲学

paper 明确写："people already have their codebase or tool for training their model"——SWM 不去抢 training framework 的位置，而是聚焦在 **environment + data collection + evaluation** 这三层。这个选择很关键，因为：

- training loop 各家有各家的偏好（PyTorch Lightning、JAX、纯 PyTorch、HuggingFace Accelerate）
- 但 environment interaction、dataset format、success rate 评估是相对稳定的接口
- 这种"窄而深"的边界设计，比"广而浅"的全栈框架更容易被采用

这点和 stable-pretraining (Balestriero et al., 2025, https://arxiv.org/abs/2511.19484) 是同一团队、同一哲学——你会发现 SWM 的 DINO-WM reproduction 就是直接用 stable-pretraining 训练的，两者互补。

---

## 2. World 抽象：API 设计的深度解析

### 2.1 与 Gymnasium API 的关键差异

Gymnasium (Towers et al., 2025, https://arxiv.org/abs/2407.17032) 的标准 API 是：

```python
obs, reward, terminated, truncated, info = env.reset(seed)
obs, reward, terminated, truncated, info = env.step(action)
```

SWM 的 World 抽象把这条 API 重构为：

```python
world.reset()
world.step()
world.infos  # dict, in-place updated
```

**关键设计决策**：

1. **step 不接收 action**。Action 通过 `world.set_policy(policy)` 注入，每次 step 时 world 内部 query policy 的 `get_action(info)` 拿 action。这把 control logic 和 environment execution 解耦，policy 可以 hot-swap。

2. **没有 return value**。所有数据（observation、reward、done、内部物理量）都写到 `world.infos` 这个共享 dict 里，in-place 更新。这避免了每个 step 创建大量临时对象（Gym 的 5-tuple 在大规模 vectorized 环境里是 GC 压力来源）。

3. **同步多环境**。`num_envs=8` 一次性 wrap 多个 environment，所有 reset/step 都同步执行。这和 Gymnasium 的 `AsyncVectorEnv`/`SyncVectorEnv` 思路类似，但 World 把 vectorization 当成一等公民。

### 2.2 为什么这个设计能 build 你的 intuition

这里有个深层考量。World model 的研究里，你经常需要：

- 用 expert policy 收集训练数据
- 用 random policy 收集 OOD 评估数据
- 用 world model + MPC policy 做 evaluation
- 在不同 FoV 下重新 reset 环境

Gym 风格的 API 每次换 policy 都要改 step 调用代码。SWM 把 policy 抽象出来，整个实验脚本可以保持不动，只换一个 `set_policy` 调用。这种 design pattern 在 robot learning 里很重要——你经常要在同一个 environment 上跑 expert demonstration、random exploration、learned policy 三种东西，统一接口节省大量样板代码。

### 2.3 Policy 接口的形式化

policy 是任意实现 `get_action(info: dict) -> np.ndarray` 的 Python 对象。返回 shape 是 `(num_envs, action_dim)`。注意这里 info 是 dict 而非 obs——这意味着 policy 可以访问环境的完整内部状态，包括 privileged information。这对：

- **oracle policy**（用 ground-truth state 规划）
- **学习型 policy**（用 pixels + 学习的 representation）
- **MPC policy**（用 world model rollout）

三种场景统一在同一个接口下，研究 fair comparison 时很重要。

---

## 3. Factors of Variation (FoV)：系统化的可控扰动

这是 SWM 最有研究价值的设计。让我把它形式化。

### 3.1 FoV 作为 MDP 的扩展参数

标准 MDP 定义为 $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$，其中：

- $\mathcal{S}$: state space
- $\mathcal{A}$: action space  
- $P: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$: transition kernel
- $R: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$: reward function
- $\gamma \in [0, 1)$: discount factor

SWM 引入一个 **环境参数向量** $\xi \in \Xi \subset \mathbb{R}^d$，使 MDP 变成参数化族：

$$\mathcal{M}(\xi) = (\mathcal{S}(\xi), \mathcal{A}, P(\xi), R(\xi), \gamma)$$

每个 component 都可能依赖 $\xi$：

- $\mathcal{S}(\xi)$: state space 本身可能变（比如 wall thickness 改变会让可达 state 集变化）
- $P(\xi)$: dynamics 变（friction、mass、gravity 改变物理 transition）
- $R(\xi)$: reward 变（goal position 移动）

这连接到几个经典框架：

- **Domain Randomization** (Tobin et al., 2017, https://arxiv.org/abs/1703.06907)：训练时 $\xi \sim p(\xi)$，目标是 policy 在 $\xi$ 上 marginalize 后 robust
- **Robust MDP** (Iyengar, 2005; Wiesemann et al., 2013)：考虑 ambiguity set $\Xi$，求 $\max_\pi \min_{\xi \in \Xi} J(\pi, \xi)$
- **Continual/Lifelong Learning**：$\xi$ 随时间变化，policy 需要 adapt 不 catastrophic forget
- **System Identification**：从 trajectory 反推 $\xi$

SWM 把这些研究方向的实验设置统一在一个 API 下。这是真正的贡献。

### 3.2 FoV 的 hierarchical naming 设计

每个 environment 暴露一组 FoV，命名形如 `key1.key2`。例如 PushT 的 16 个 FoV：

- `agent.angle, agent.color, agent.scale, agent.shape, agent.start_position, agent.velocity`
- `background.color`
- `block.angle, block.color, block.scale, block.shape, block.start_position`
- `goal.angle, goal.color, goal.position, goal.scale`

这里 `agent` 是 wildcard，覆盖所有 `agent.*` 子项。这种 hierarchical structure 让用户能精确控制扰动粒度：

```python
options={"variation": ["agent.color"]}        # 只改颜色
options={"variation": ["agent"]}               # 改所有 agent 属性
options={"variation": ["all"]}                 # 改所有 FoV
options={"variation": ["agent.color"], "variation_values": {"agent.color": [255,0,0]}}  # 固定红色
```

### 3.3 FoV 实现为 Gymnasium Space

paper 提到 FoV 被实现为一种新的 Gymnasium dictionary Space，"stores an internal value that can be initialized, sampled with or without constraint"。这个实现细节很关键——它意味着 FoV 可以：

1. 被 `space.sample()` 随机采样（用于 domain randomization 训练）
2. 被 `space.contains(value)` 验证（用于 reproducibility check）
3. 和 action/observation space 并列存在（vectorized environment 里 FoV 也是 batched）

这给 future 工作留了空间：比如 learnable FoV inference（从观测反推环境参数）、FoV-aware world model（model 显式以 $\xi$ 为 input）等。

### 3.4 环境套件的多样性

Table 3 列了 16 个 environment，涵盖：

| 类别 | Environments | 典型 FoV 数 |
|------|--------------|-------------|
| 2D manipulation | PushT | 16 |
| 2D navigation | TwoRoom | 17 |
| 3D manipulation (MuJoCo) | OGBench Cube/Scene | 11-12 |
| Classic control (DMC) | Pendulum, Cartpole | 6 |
| DMC locomotion | Humanoid, Cheetah, Hopper, Walker, Quadruped | 7-8 |
| DMC manipulation | Reacher, Finger, Manipulator, Ball-in-Cup, Acrobot | 8-10 |

FoV 类型涵盖：

- **Visual**: `agent.color, background.color, floor.color, light.intensity`
- **Geometric**: `agent.scale, block.shape, cube.size, wall.thickness, camera.angle_delta`
- **Physical**: `floor.friction, agent.torso_density, agent.left_knee_locked, agent.mass_density`
- **Task**: `goal.position, door.position, cube.start_position`

注意 DMC 环境里有 `agent.left_knee_locked`、`agent.foot_locked` 这种 joint-level 的 FoV——这相当于部分关节失效的 perturbation，对研究 robust locomotion policy 非常有用。

---

## 4. Evaluation 协议：online vs offline

### 4.1 两种协议的数学表述

**Online evaluation** (PLDM 风格)：

每个 episode $i$：
1. 采样初始状态 $s_0^{(i)} \sim \rho_0$ 和目标 $g^{(i)} \sim \rho_g$
2. 用 policy $\pi$ rollout：$a_t = \pi(\cdot | s_t, g^{(i)})$, $s_{t+1} \sim P(\cdot | s_t, a_t)$
3. 检查是否在 $T_{max}$ 步内 reach goal: $\|s_t - g^{(i)}\| < \epsilon$

Success rate: $\text{SR} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}[\exists t \le T_{max}: \|s_t^{(i)} - g^{(i)}\| < \epsilon]$

**Offline evaluation** (DINO-WM 风格)：

1. 从 dataset 采样完整轨迹 $\tau = (s_0, a_0, s_1, ..., s_T)$
2. 选 $s_0$ 作为初始状态，$s_k$ 作为 goal，约束 $k \le K_{max}$
3. 这个约束保证 task 在 budget 内 feasible
4. 用 policy rollout 检查能否 reach $s_k$

为什么 offline 协议重要？因为它给了你一个 **guaranteed-feasible task**。Online 协议下，随机选的 $(s_0, g)$ 可能根本无法在 budget 内 reach（比如初始距离太远），policy 失败可能是 task infeasible 而非 policy 不好。Offline 协议把 task difficulty 标准化，让你能更干净地比较 model quality。

### 4.2 MPC 与 solver 实现

SWM 提供的 MPC 接口是 `WorldModelPolicy(solver, PlanConfig)`。PlanConfig 包含：

- `horizon`: planning horizon $H$
- `receding_horizon`: receding horizon $h$（执行 $h$ 步后重新规划）

Solver 选项：

- **CEM (Cross-Entropy Method)** (De Boer et al., 2005, https://www.cs.princeton.edu/courses/archive/fall06/cos521/papers/CEM.pdf)
- **MPPI (Model Predictive Path Integral)** (Williams et al., 2017, https://arxiv.org/abs/1703.02945)
- **Gradient-based**: SGD, Adam (要求 dynamics model 可微)

### 4.3 CEM 公式与变量说明

CEM 维护一个 action sequence 分布。设 horizon $H$，action 维度 $d_a$：

- 分布参数：$\mu \in \mathbb{R}^{H \cdot d_a}$, $\Sigma \in \mathbb{R}^{H \cdot d_a \times H \cdot d_a}$（通常对角）
- 每次迭代：
  1. 采样 $N$ 个 action sequences: $\mathbf{a}^{(i)} \sim \mathcal{N}(\mu, \Sigma)$ for $i = 1, ..., N$
  2. 对每个 sequence 用 world model rollout，计算 cost：
     $$J(\mathbf{a}^{(i)}) = \sum_{t=1}^{H} c(\hat{s}_t^{(i)}, a_t^{(i)}, g)$$
     其中 $\hat{s}_t^{(i)}$ 是 model 预测的 state，$c$ 是 cost function（通常是到 goal 的距离）
  3. 选 elite set：$E = \text{top-}K(\{\mathbf{a}^{(i)}\}, J)$，即 cost 最小的 $K$ 个
  4. 用 elite sample 重新拟合分布：
     $$\mu \leftarrow \frac{1}{K} \sum_{i \in E} \mathbf{a}^{(i)}$$
     $$\Sigma \leftarrow \frac{1}{K} \sum_{i \in E} (\mathbf{a}^{(i)} - \mu)(\mathbf{a}^{(i)} - \mu)^T$$
  5. 重复 $M$ 次迭代

最终输出 $\mu$ 的第一个 action $\mu_{1:d_a}$（receding horizon 控制下执行 $h$ 步）。

paper Appendix C 写到复现 DINO-WM 时 CEM 用了和原文一致的参数，但把 planning budget 从 infinite 改成 50 步（2× minimum required 25 步）。这是个非常重要的 normalization——原 DINO-WM paper 可能给了太多步 budget，掩盖了 planning quality 的差异。

---

## 5. 核心实验：DINO-WM zero-shot robustness

这是 paper 里唯一一个真实验证 SWM utility 的实验。让我深入解读。

### 5.1 DINO-WM 的架构回顾

DINO-WM (Zhou et al., 2025, https://arxiv.org/abs/2501.17172) 的核心思想：

- **Encoder**: 用预训练 DINOv2 (Oquab et al., 2023, https://arxiv.org/abs/2304.07193) 提取 visual features 作为 latent state $z = f_{\text{DINO}}(x) \in \mathbb{R}^D$
- **Dynamics model**: $z_{t+1} = g_\phi(z_t, a_t)$，学习 latent space 的一步 transition
- **Planning**: 在 latent space 用 CEM 规划，cost 用 $\|z - z_g\|^2$
- **No decoder required**：因为 DINO features 已经语义对齐，latent distance 直接对应 semantic distance

paper 复现的结果：在 expert demonstration 上 **94.0% success rate**——这复制了 DINO-WM 原文的核心 claim，证明 DINOv2 features 确实可以支持 zero-shot planning（不需要 task-specific training）。

### 5.2 OOD eval：12.0% 的暴跌

paper 把 evaluation data 从 expert trajectory 换成 random policy trajectory，success rate 从 94% 跌到 12%。这是个**非常 revealing 的数字**。

可能的解释：

1. **Cost function 的 distribution dependency**。CEM 的 cost 用 latent distance $\|z - z_g\|^2$。如果 DINO features 在 expert trajectory manifold 上 well-conditioned，但在 random policy 看到的 state 上分布偏移，distance 可能变得无意义。

2. **DINO features 的 manifold collapse**。DINOv2 训练目标是 self-supervised discriminative，它的 feature space 在 ImageNet-like 分布上 well-structured，但 robot manipulation 的 visual observation 分布完全不同。expert trajectory 可能恰好落在 DINO 见过的"hand-like object + structured background"附近，random trajectory 引入的混乱姿态可能 fall off manifold。

3. **Goal representation 的 sampling bias**。expert trajectory 的 goal state 是 task-completion state（block 在 anchor 上），random trajectory 采样到的 "goal" 可能是任意中间态，对 CEM 来说是个 ill-posed target。

这里 SWM 的价值在于：它让你**用一个 API 切换 evaluation data source**，立刻暴露这个 problem。原 DINO-WM paper 没有报告这个数字。

### 5.3 FoV perturbation 的全面失败

Table 2 是 paper 最有信息量的表。让我重新整理一下：

| FoV Category | Property | SR (%) |
|--------------|----------|--------|
| (none, expert eval) | — | 94.0 |
| Color | Anchor | 20.0 |
| Color | Agent | 18.0 |
| Color | Block | 18.0 |
| Color | Background | 10.0 |
| Size | Anchor | 14.0 |
| Size | Agent | **4.0** |
| Size | Block | 16.0 |
| Angle | Anchor | 12.0 |
| Angle | Agent | 12.0 |
| Position | Anchor | 4.0 |
| Shape | Agent | 18.0 |
| Shape | Block | 8.0 |
| Velocity | Agent | 14.0 |

几个观察：

1. **所有 FoV 都让 SR 暴跌到 4-20%**。DINOv2 号称的 robustness（在 ImageNet 上对 color jitter、scale 变化鲁棒）**没有 transfer 到这个 task**。

2. **Agent size 最致命 (4%)**。这符合直觉——agent size 改变意味着 agent 的 visual appearance 偏离 training distribution 最大。DINO features 对 object scale 有一定 invariance，但这里 invariance 不够覆盖这个变化范围。

3. **Position 也致命 (4%)**。Anchor 位置变化让 task 本身变了——cost function $\|z - z_g\|^2$ 中 $z_g$ 对应的 visual configuration 完全不同。DINO-WM 的"zero-shot planning"隐含假设：goal state 的 latent representation 在 task 不变时稳定。但 anchor 位置一变，"goal"在 latent space 的位置就漂移了。

4. **Background color 居然只有 10%**。这很意外，因为 DINOv2 在 pretraining 时见过海量 background variations。可能解释：PushT 的 background 是单一纯色填充，和 ImageNet 的 textured background 分布完全不同，DINOv2 的 invariance 不覆盖这个 OOD type。

### 5.4 这个实验告诉了我们什么

这是 SWM 作为 research tool 的核心 demo。它揭示了一个被原 paper 掩盖的 limitation：**DINO-WM 的 "zero-shot" 是非常脆弱的 zero-shot**——只在 expert demonstration 的 narrow distribution 内 zero-shot。一旦环境有任何 controllable factor 偏移，planning 就崩溃。

这指向几个 future direction：

1. **FoV-aware world model**：dynamics model 显式以 $\xi$ 为 input: $z_{t+1} = g_\phi(z_t, a_t, \xi)$。这让 model 知道环境参数，可以做 conditional prediction。

2. **FoV-augmented training**：训练时 domain randomization over $\xi$，让 model 在 $\xi$ 上 marginalize 出 robust representation。

3. **Goal-conditioned on $\xi$**：让 goal representation 也 conditioned on $\xi$，避免 anchor position 改变时 goal latent 漂移。

4. **Adaptive planning**：online system identification，从近期 trajectory 推断 $\xi$，然后 conditional planning。

SWM 让这些想法可以**用统一接口实验**——这就是基础设施的价值。

---

## 6. 与相关 codebase 的对比

Table 1 给了一个量化对比：

| | SWM | PLDM | DINO-WM |
|---|---|---|---|
| Backend | PyTorch | PyTorch | PyTorch |
| Documentation | ✓ | ✗ | ✗ |
| # Baselines | 4 | 1 | 1 |
| # Environments | 16 | 2 | 4 |
| # FoV (per env) | 6-17 | 0 | 0 |
| Type Checking | ✓ | ✓ | ✗ |
| Test Coverage | 73% | 0% | 0% |
| Last Commit | <1 week | >3 months | >10 months |
| PRs (6 mo.) | 99 | 1 | 0 |
| # LoC | 3562 | 6796 | 4349 |

注意 SWM 用更少 LoC (3562) 提供更多功能——这是好的 abstraction 的标志。99 PRs vs 1/0 显示这是一个**活的项目**，不是 publish-and-abandon 的研究代码。73% test coverage 在研究代码里非常罕见（DINO-WM 和 PLDM 都是 0%）。

PLDM 论文 (Sobal et al., 2025, https://openreview.net/forum?id=jON7H6A9UU) 提出的核心论点是 "stress-testing offline reward-free RL with latent dynamics models"——它本身偏 algorithm paper，codebase 是论文副产品。DINO-WM 同理。SWM 是反过来：**先做 infrastructure，再 demo 一个实验**。这种 positioning 在 ML 圈比较少见但很重要——类比 OpenAI Gym 之于 RL。

---

## 7. 局限性与未来方向

paper Section 4 提到了几个未来方向，我补充一些批判性思考：

1. **没有真实 robot 环境**。当前 16 个环境全是 simulation。真实世界的 FoV（光照、相机视角、物理接触）更难 control。这点 OGBench (https://openreview.net/forum?id=M992mjgKzI) 也是 simulation-only，整个 field 都缺 real-world standardized benchmark。

2. **没有 reward signal**。SWM 聚焦 goal-conditioned evaluation，但很多 WM 应用涉及 reward maximization。未来可能要加 reward prediction head。

3. **没有 video prediction benchmark**。早期 WM 工作 (Ha & Schmidhuber 2018, DreamFlowNet, GAIA-1) 强调 video generation quality。SWM 只测 planning success，这忽略了 representation quality 的其他维度。

4. **Hugging Face Benchmark 的愿景**。paper 末尾提到要做 standardized benchmark——这其实是 SWM 真正能 impact 的地方。如果 community 接受它作为 submission interface，paper 的影响力会指数级放大，类似 ImageNet 之于 image classification。

5. **缺少 multimodal observations**。当前所有环境是 image + state，没有 tactile、audio、language instruction。真实 robot 是 multimodal 的。

6. **Action space 的多样性**。目前都是 continuous 或 discrete action，没有 natural language action (SayCan 风格)、diffusion policy 的 action chunk。

---

## 8. 与你过去工作的潜在连接

Andrej，如果你考虑用 SWM 做后续研究，几个可能的切入点：

1. **Building micrograd-style educational WM**: SWM 的 modular API 适合做教学，写一个几百行代码实现 Dreamer-style world model，用 SWM 的 environment + evaluation 测它。

2. **Eureka-style automatic curriculum via FoV**: 用 LLM 自动设计 FoV curriculum，让 agent 在 progressively harder FoV perturbation 上训练，测试 robust generalization。SWM 的 FoV 接口正好提供这个 lever。

3. **V-JEPA 2 + SWM**: 你和 Yann LeCun 团队的 V-JEPA 2 (https://arxiv.org/abs/2506.04985, Bardes et al.) 是 video版本的 JEPA。V-JEPA 2 的 joint embedding 在 DINO-WM 的位置上应该更 robust——因为它是 video-pretrained, 直接预测 latent dynamics。SWM 的 FoV 正好提供 systematic robustness evaluation。这个实验值得做。

4. **World model interpretability**: SWM 的 controlled FoV 是天然的 interpretability probe——你可以问"world model 对哪个 FoV 敏感"，把它当成 model 诊断工具。

---

## 9. 总结

SWM 这篇 paper 的核心 contribution 是：

- **统一了 World Model 研究的 environment + data + evaluation 层**
- **引入 FoV 作为 first-class concept**，连接到 robustness、domain randomization、continual learning
- **用一个简单的 DINO-WM robustness 实验展示了 utility**，并揭示了 DINO-WM 的 hidden limitation
- **73% test coverage + documentation + 99 PRs**——这是工业级的研究代码，不是 paper 附件

它**不是**算法 paper，**而是**infrastructure paper。它的价值要在 community 采纳之后才会完全显现。如果未来 ICLR/NeurIPS 的 world model 投稿大量用 SWM 作为实验平台，这篇 paper 就成功了。

**关键 web links**:

- SWM paper (preprint, 暂未见正式 arxiv ID，可能跟随 publication)
- stable-pretraining (Balestriero et al., 2025): https://arxiv.org/abs/2511.19484
- DINO-WM: https://arxiv.org/abs/2501.17172
- PLDM: https://openreview.net/forum?id=jON7H6A9UU
- OGBench: https://openreview.net/forum?id=M992mjgKzI
- DINOv2: https://arxiv.org/abs/2304.07193
- Ha & Schmidhuber World Models: https://arxiv.org/abs/1803.10122
- DMC: https://arxiv.org/abs/1801.00690
- Diffusion Policy (PushT): https://arxiv.org/abs/2303.04137
- Gymnasium: https://arxiv.org/abs/2407.17032
- CEM (De Boer et al.): https://www.cs.princeton.edu/courses/archive/fall06/cos521/papers/CEM.pdf
- MPPI (Williams et al.): https://arxiv.org/abs/1703.02945
- V-JEPA 2: https://arxiv.org/abs/2506.04985

希望这些细节能帮你 build 起对 SWM 的 intuition。最值得思考的是 FoV 作为可控 perturbation 这个 abstraction——它把 world model 研究从"会做这个 task 吗"升级到"在 task 的哪些变体上还能做"，这是一个更严格也更 meaningful 的评估维度。
