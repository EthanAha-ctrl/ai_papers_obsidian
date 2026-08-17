---
source_pdf: Squint Fast Visual Reinforcement Learning for Sim-to-Real Robotics.pdf
paper_sha256: 1b3b57211fd74b874f5ca6ef0cf42852b7c1cbd58e2f22ee6073f6bb3f8a03df
processed_at: '2026-08-12T10:18:11-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Squint 用人话讲

---

## 一句话版本

哥几个发现: 机械臂拿摄像头干活, RL 训练太慢了, 几天几夜跑不完。Squint 这套 trick 组合拳打下来, **单张 3090, 15分钟, sim 到 real zero-shot**, 八个 task 平均 91.3% 成功率。核心就是"把 image 弄小、把并行开大、把 critic 弄聪明"。

---

## 为啥 visual RL 一直慢

先 build 你旳 intuition about the landscape:

Robot RL field 一直有个三角困境。**Sample efficiency** 派 (DrQ-v2, TD-MPC2) 交互少但 wall time 长, 因为它们 sequential 跑, GPU 利用率低。**Wall-time** 派 (PPO) 能在 GPU 上开几千个 env 一起跑, 但 sample 浪费严重, 因为 on-policy 不 reuse data。最近两年冒出来第三派——**FastTD3** [https://arxiv.org/abs/2505.22642], **FastSAC** [https://arxiv.org/abs/2512.01996], **PQL** [https://arxiv.org/abs/2305.18534] —— off-policy + 大量并行, 想同时吃两头好处。

但这一派之前只搞了 state-based (humanoid locomotion)。Squint 把这个思路 extend 到 visual RL, 难点在于: image 高维、replay buffer 吃 memory、CNN forward 慢。所以 Squint 本质上是 engineering paper, 没新算法, 全是 careful design choices。

---

## 六个 trick 逐个人话拆解

### Trick 1: Update-to-Data Ratio (UTD)

UTD = gradient steps / env steps

Humanoid locomotion 那帮人 (FastTD3) 用 UTD < 0.06, 1024 envs 跑 1 step 就 update 1 次。因为 locomotion 是周期运动, state space 维度低, Q-function 容易估准。

Squint 发现 manipulation 不行, 要 UTD = 0.25, 即 1024 envs 跑 1 step 后 update 256 次。因为 manipulation 是 contact-rich, Q-function 需要更多 gradient steps 来 refine, 否则 estimate 不准, policy 学不出来。

公式:
$$\text{UTD} = \frac{N_{\text{updates}}}{N_{\text{envs}} \cdot N_{\text{steps}}} = \frac{256}{1024 \times 1} = 0.25$$

变量解释:
- $N_{\text{updates}}$: 每轮 env interaction 后做多少次 gradient update
- $N_{\text{envs}}$: 并行环境数, 这里 1024
- $N_{\text{steps}}$: 每个 env 每轮走几步, 这里 1

**Intuition**: Locomotion 像"原地跑步", 探索空间小, 偶尔 update 就够。Manipulation 像"在暗房间找钥匙", 每一步都得仔细想, update 要多。

---

### Trick 2: Distributional Critic (C51)

Standard SAC critic 输出一个 scalar $Q(s, a)$, 用 MSE loss。C51 [https://arxiv.org/abs/1707.06887] 输出 51 个 bin 上的 categorical distribution, 用 cross-entropy loss。

数学:
$$Q_\theta(s, a) = \sum_{i=1}^{51} v_i \cdot p_i(s, a; \theta)$$

变量:
- $v_i$: 第 $i$ 个 atom 的 value, 范围 $[V_{\min}, V_{\max}]$, 比如 $[-10, 10]$ 均匀切 51 段
- $p_i$: softmax 输出的 probability, $\sum_i p_i = 1$
- $\theta$: critic network 参数

Loss:
$$\mathcal{L}_{Q_\theta} = -\sum_{i=1}^{51} \hat{p}_i \log p_i$$

$\hat{p}_i$ 是 TD target 投影回来的 target distribution (Bellman projection)。

**Intuition**: Standard critic 把 return 压成一个数字, 信号粗糙。Distributional critic 保留 return 的整个分布, gradient 信号丰富 51 倍 (在 distribution 层面)。特别是在 outcome bimodal 的 task 上 (Stack Can 要么成功要么失败, 没有 middle ground), 分布式 critic 能 capture 这种不确定性, scalar critic 只能取平均, 学不清楚。

Table I 里 Stack Can 任务: Squint 81.2%, SAC baseline 只有 18.7% —— 差距 62.5%, 主要归功于 distributional critic。

---

### Trick 3: Resolution Squinting ⭐ 这篇最骚的

直接 16×16 渲染 vs 128×128 渲染再 area downsample 到 16×16, 后者明显更好。

| 方案 | 渲染 | 输入 | 性能 |
|------|------|------|------|
| Direct render | 16×16 | 16×16 | baseline |
| Squinting | 128×128 | 16×16 (downsampled) | 显著更好 |

为啥? 三个原因:

**1. Anti-aliasing**: 16×16 直接渲染相当于 point sampling, 高频信号 alias 成噪声, edges 全糊掉。128×128 渲染后 area downsample (8×8 average pooling) 相当于先 low-pass filter 再 decimate, 符合 Nyquist, 保留 low-frequency scene structure。

**2. Scene structure preservation**: 高分辨率渲染后 downsample, cube 的 edge 方向、gripper 的相对位置这些 coarse geometry 信息能保留。直接 16×16 渲染这些信息被 aliase 掉了。

**3. Sim-to-real 一致性**: 真实相机本来高分辨率, ISP downsample。Squinting 让 sim 的 image formation pipeline 跟 real 一致, 减小 domain gap。

**Intuition**: 想象你眯眼 (squint) 看东西——细节糊了, 但 layout 清楚。这个 layout 对 manipulation 够了。但如果你直接用一个 16×16 sensor 看, 因为 sampling 方式不对, 反而把重要的 edges 丢了。Squinting 就是"用正确的方式眯眼"。

这跟你讲的 "network capacity 要匹配 data complexity" 一脉相承。16×16 = 768 pixels, 比典型 visual RL 的 84×84 (21168 pixels) 少 27 倍。Manipulation 不需要 pixel-level detail, 需要的是 scene layout, 小 resolution 配合正确 downsampling 反而更好。

---

### Trick 4: LayerNorm everywhere

每个 linear layer 后面接 LayerNorm [https://arxiv.org/abs/1607.06450]。

为啥? 并行 RL 里, gradient 来自 1024 个 envs 的 mini-batch, 信号方差大。LayerNorm normalize 每个 sample 的 hidden activation, 让 gradient 更 stable, 允许更大 learning rate, 加速 convergence。

注意是 LayerNorm 不是 BatchNorm。BatchNorm 在 RL 里会崩, 因为 replay buffer 里的 sample 是 correlated 的, batch statistics 不稳定。LayerNorm 是 per-sample normalize, 跟 batch 无关, 安全。

---

### Trick 5: Shared encoder + stop gradient

架构:

```
Image (16x16x3)
  → Conv2d(32, 3x3, stride=2) → ReLU
  → Conv2d(32, 3x3, stride=2) → ReLU
  → Flatten → Linear(proj_dim)
  → z (latent)
  → Actor projection (1 layer) → action
  → Critic projection (1 layer) → Q
```

关键:
- Actor 和 critic 共享 encoder $f_\psi$, 省一半 compute
- Encoder 只通过 critic 的 TD-loss 更新
- Actor update 时 stop gradient on $z_t$, 防止 policy gradient 把 representation 带偏

伪代码对应:
```
z_t ← f_ψ(ō_t)           # encoder forward
z_t^sg ← stopGRAD(z_t)   # for actor, block gradient
ã_t ~ π_φ(·|z_t^sg, s_t^proprio)  # actor sample
```

**Intuition**: Critic 学到 representation 是对 value estimation 最优的。Actor 直接用这个 representation, 别自己瞎搞, 否则 policy gradient 会把 representation 往 policy 方向拉, 跟 critic 打架。Shared encoder 让两个网络看同一个 "世界", 但只让 critic 当 representation 的 teacher。

这个 pattern 来自 DrQ [https://arxiv.org/abs/2004.13649] 和 DrQ-v2 [https://arxiv.org/abs/2107.09645]。

---

### Trick 6: PyTorch Engineering

- **torch.compile**: kernel fusion, 减少 Python overhead
- **cudagraphs**: capture CUDA graph replay, 避免 CPU launch overhead
- **AMP bfloat16**: CNN forward 用 bf16, 减半 memory bandwidth
- **CleanRL base** [https://github.com/vwxyzjn/cleanrl]: single-file implementation, 好调好 debug

组合起来 5x 加速。这就是为啥 15 分钟能跑完——没这些 optimization, 75 分钟也跑不完。

---

## SO-101 Task Set: 八个 manipulation task

| Task | 描述 | 难度 |
|------|------|------|
| Reach Cube | 到 cube 上方 2cm | Easy |
| Reach Can | 到 can 上方 2cm | Easy |
| Lift Cube | 抓 cube 到 rest position | Medium |
| Lift Can | 抓 can 到 rest position | Medium |
| Place Cube | 放 cube 到 container | Medium |
| Place Can | 放 can 到 container | Medium |
| Stack Cube | 红 cube 叠蓝 cube | Hard |
| Stack Can | cube 叠 can (can 会倒) | Very Hard |

Task design:
- 5-DoF SO-101 arm [https://github.com/huggingface/lerobot], 便宜的开源机械臂
- Wrist camera only, 不用 third-person (参考 [https://arxiv.org/abs/2203.12677] "Vision-based manipulators need to also see from their hands")
- Object 位置在 gripper 起点 0.2m × 0.2m xy 范围随机

**为啥 wrist camera?** Wrist camera 给的是 object-centric view, object 永远在视野里。Third-person camera 的 view 跟 gripper 位置耦合, representation 更难学。而且 wrist camera 需要 active vision, agent 要主动调视角, 这跟人类 manipulation 更像。

---

## Domain Randomization 细节

Squint 上了 aggressive DR, 但仍然 15 分钟收敛:

**Visual DR**:
- Camera position / rotation / FOV perturbation
- Lighting changes
- **Color jitter** (关键!)

**Physical DR**:
- Object sizes
- Object frictions
- Gripper closing speed

**Proprioceptive noise**:
- Joint positions 加 Gaussian noise $\mathcal{N}(0, 5^2 I)$

Color jitter ablation (Table IV) 非常 striking:

| Variant | Real success |
|---------|-------------|
| With color jitter | 73/80 (91.3%) |
| Without color jitter | 58/80 (72.5%) |

差 18.8 个点! 说明 policy 对 lighting 极度敏感, color jitter 是 sim-to-real 的关键 robustness 来源。

---

## 实验结果人话版

### Sim (Table I)

| Method | All Tasks Avg |
|--------|---------------|
| **Squint** | **96.1% ± 1.6** |
| SAC (Squint hyperparams, no arch) | 88.3% ± 3.4 |
| PPO | 60.2% ± 4.0 |
| DrQ-v2 (sequential) | 4.5% ± 2.9 |
| BC | 41.9% ± 4.2 |

观察:
- **Squint vs SAC**: 7.8% gap, 全靠架构差异 (distributional critic + squinting + LayerNorm)
- **SAC vs PPO**: 28% gap, off-policy 在 hard task (Place, Stack) 完胜, 因为 data reuse
- **DrQ-v2 崩了**: 4.5%! 因为它是 sequential, 1 个 env, heavy DR 下 sample efficiency 不够。证明 parallelization 对 visual RL wall time 至关重要
- **Stack Can**: Squint 81.2%, SAC 18.7% —— distributional critic 在 bimodal outcome task 上碾压

### Real (Table II)

| Method | Total | Avg |
|--------|-------|-----|
| **Squint** | **73/80** | **91.3%** |
| SAC | 65/80 | 81.3% |
| PPO | 50/80 | 62.5% |
| DrQ-v2 | 8/80 | 10.0% |
| BC | 38/80 | 47.5% |

Sim-to-real gap:
- Squint: 96.1% → 91.3% (-4.8%)
- SAC: 88.3% → 81.3% (-7.0%)

Squint gap 最小, 说明 squinting + heavy DR 让 policy robust。

### vs Visual DAgger (Table III)

| Method | Total | Avg |
|--------|-------|-----|
| Squint | 73/80 | 91.3% |
| Visual DAgger (distill from state SAC) | 53/80 | 66.3% |

差 25 个点! 非常重要——**直接 visual RL > state-to-visual distillation** 在 active vision 设置下。

**Intuition**: State-based SAC expert 探索方式跟 visual agent 不同。State agent 全局视野, 探索是 direct reach。Visual agent wrist camera 局部视野, 探索需要 "peek-and-search" 主动调视角。Distillation 无法 transfer 这种 behavioral prior, student 模仿不到 teacher 的 exploration pattern。

这跟 [https://arxiv.org/abs/2502.08844] "When should we prefer state-to-visual DAgger over visual RL?" 的话题相关, Squint 给了答案: 当你有 fast visual RL 的时候, 直接 visual RL 赢。

---

## Sim-to-Real Transfer 的两个 trick

### Action scaling
- Sim: 10Hz, large position offsets
- Real: action × 0.15, 30Hz (3x faster)

为啥不 match frequency? Real robot 需要更快 recovery control 应对 disturbance, 30Hz 给 smoother trajectories。虽然有 mismatch, 但 policy 学的是 action shape, 不是 absolute timing, 所以 transfer OK。

**Intuition**: Policy 学到的是 "动作的形状", 不是 "每秒多少步"。Real 更高频率 = finer-grained execution, 类似 video game high FPS 让控制更顺滑。

---

## 整个 pipeline 人话总结

1. **开 1024 个并行 env** 在 ManiSkill3 [https://arxiv.org/abs/2410.00425] 里跑
2. 每个 env step: wrist camera render 128×128, area downsample 到 16×16 (squinting)
3. 16×16 image + proprioceptive state 存 replay buffer (1M capacity, 全在 GPU)
4. 每轮 env step 后, 做 256 次 gradient update:
   - Critic: distributional C51 loss, double Q average, encoder joint update
   - Actor: policy gradient with stop gradient on encoder
   - Entropy temperature 自适应
5. LayerNorm everywhere, torch.compile + cudagraph + bf16
6. 15 分钟后拿 final checkpoint, action × 0.15, 30Hz, zero-shot deploy 到 real SO-101

---

## 你 (Andrej) 可能关心的点

### Software 2.0 角度
15 分钟训练改变 research workflow。从 "训一周看一次结果" 变成 "coffee break 回来 policy 就训好了"。这跟你讲的 "Software 2.0 要 fast iteration" 完全一致。Robot learning 终于能像写代码一样 iterate。

### Resolution vs Compute
16×16 = 768 pixels, 比典型 visual RL 的 84×84 少 27 倍。Manipulation 不需要 pixel detail, 需要的是 scene layout。Squinting 证明了 "正确获取低分辨率信号" 比 "直接用低分辨率 sensor" 重要得多。这跟你 "capacity 匹配 data complexity" 的哲学一致。

### Distributional = Richer gradient
C51 用 51 个 atom 把 scalar Q 变 distribution, gradient 信号丰富 51 倍 (在 distribution 层面)。这跟 classification 比 regression stable 的 intuition 一样—— categorical 信号比 continuous 信号好学。

### Active vision 的 imitation limitation
State-to-Visual DAgger 失败说明 **observation space 决定 exploration behavior**。State agent 全局视野 exploration 是 direct, visual agent 局部视野 exploration 是 peek-and-search。Distillation transfer 不了这种 behavioral prior。这跟你在 CS231n 讲 "representation determines what's learnable" 一脉相承。

---

## Limitations 和 open questions

1. **Stack Can 只 81.2%**: 最难 task, distributional critic 也救不了, 可能要更长 training 或更好 exploration
2. **Single robot, single camera**: SO-101 是 toy platform, industrial robot (Franka, UR5) 复杂度高得多
3. **Color jitter 18% gap 大**: 说明 policy 还是 overfit sim rendering distribution, 真正 robust visual RL 需要更多
4. **1M buffer @ 16×16**: 768MB fit GPU, 但小 image 可能丢太多 info, 换更复杂 task 可能不够
5. **没比 TD-MPC2** [https://arxiv.org/abs/2310.16828]: SOTA model-based visual RL 缺席比较可惜
6. **Single task only**: 八个 task 分别训, 没 multi-task generalist
7. **Symmetric training**: Critic 也用 visual input, 没用 privileged information。Asymmetric actor-critic [https://arxiv.org/abs/1710.06542] 可能更快
8. **Simple FIFO buffer**: 没用 prioritized experience replay [https://arxiv.org/abs/1511.05952]

---

## Final takeaway

Squint recipe 人话总结: **"正确地获取低分辨率信号 + 大量并行 + 适度 reuse + 聪明的 critic"**

- Squinting = 用高分辨率 render 再 downsample 获取 clean 低频信号, 不是直接降分辨率
- 1024 envs + 256 updates (UTD=0.25) = 比 humanoid UTD 高 4x, 因为 manipulation 需要更准的 Q
- Distributional critic = 更 rich gradient 信号, 在 bimodal outcomes 上明显赢
- LayerNorm = 让大 batch 并行训练 stable
- torch.compile + cudagraph + bf16 = 5x 加速

整个 pipeline 没 new algorithm, 全是 engineering + careful ablations。15 分钟 single GPU zero-shot sim-to-real, 是 paradigm shift。下一个 milestone 应该是 **1 分钟训练 + multi-task generalist**, 期待后续。

---

## References 速查

- Squint project: https://aalmuzairee.github.io/squint
- FastTD3: https://arxiv.org/abs/2505.22642
- FastSAC: https://arxiv.org/abs/2512.01996
- SAC: https://arxiv.org/abs/1801.01290
- C51: https://arxiv.org/abs/1707.06887
- DrQ-v2: https://arxiv.org/abs/2107.09645
- ManiSkill3: https://arxiv.org/abs/2410.00425
- LeRobot: https://github.com/huggingface/lerobot
- CleanRL: https://github.com/vwxyzjn/cleanrl
- LayerNorm: https://arxiv.org/abs/1607.06450
- Active vision: https://arxiv.org/abs/2203.12677
- Domain randomization: https://arxiv.org/abs/1703.06907

如果想 dive deeper 某个具体点 (C51 projection 细节, ManiSkill3 batched rendering internals, 或 sim-to-real frequency mismatch 的理论分析), 继续问!

---

# Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics 详细解析

Hey Andrej! 这篇paper来自UC San Diego的Henrik Christensen lab, 第一作者Abdulaziz Almuzairee, 核心思想非常工程化——把visual SAC调到能在**单张RTX 3090上15分钟训完**并zero-shot部署到真实SO-101机械臂。Project page: https://aalmuzairee.github.io/squint

---

## I. 核心insight与定位

这篇paper的核心问题非常清晰: **visual RL在wall-clock time上太慢**。整个field面临一个trilemma:

| Axis | 方法代表 | 优势 | 代价 |
|------|---------|------|------|
| Sample efficiency | DrQ-v2, CURL, TD-MPC2 | 交互少 | 慢, 难以parallelize |
| Wall-time (on-policy) | PPO | 并行快 | 浪费samples |
| **Wall-time (off-policy)** | **FastTD3, FastSAC, Squint** | **并行+reuse** | **需调UTD** |

Squint的positioning是第三个axis的**visual版本**。之前的FastTD3 [https://arxiv.org/abs/2505.22642] 和FastSAC只做了state-based, Squint把它extend到visual RL, 关键挑战是:
1. Image是high-dimensional input, 训练dynamics复杂
2. Replay buffer存image吃memory
3. CNN encoder forward pass慢

---

## II. 方法论: 6个设计选择深度拆解

### 2.1 Update-to-Data (UTD) Ratio

UTD = num_updates / num_env_steps_per_iteration

**关键发现**: manipulation domain需要**比humanoid更高的UTD ratio**。

- Humanoid locomotion (FastTD3/FastSAC): UTD < 0.06 (1024 envs, 1 update)
- Manipulation (Squint): UTD = 0.25 (1024 envs, 256 updates)

**Intuition**: Locomotion是周期性运动, exploration相对简单, state space维度低, 所以低UTD就够。Manipulation需要精确的contact-rich control, Q-function需要更准确地估计, 所以需要更多gradient steps来refine value function。

公式:
$$\text{UTD} = \frac{N_{\text{updates}}}{N_{\text{envs}} \times N_{\text{steps}}} = \frac{256}{1024 \times 1} = 0.25$$

这个发现非常重要, 说明**UTD不是越小越好**, 要根据task complexity调。这跟D'Oro et al.的"breaking the replay ratio barrier" [https://openreview.net/forum?id=OpC-9aBBVJe] 形成对比。

### 2.2 Distributional C51 Critic

不用standard MSE critic, 改用C51 [https://arxiv.org/abs/1707.06887] 的**categorical distributional critic**。

**Standard SAC critic loss**:
$$\mathcal{L}_{Q_\theta}(\mathcal{D}) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \sim \mathcal{D}} \left[ (Q_\theta(s_t, a_t) - y)^2 \right]$$

其中target:
$$y = r_t + \gamma \left( Q_{\bar{\theta}}(s_{t+1}, \tilde{a}_{t+1}) - \alpha \log \pi_\phi(\tilde{a}_{t+1}|s_{t+1}) \right)$$

变量解释:
- $y$: TD target, one-step bootstrapped value estimate
- $r_t$: reward at timestep $t$
- $\gamma$: discount factor (通常0.99)
- $Q_{\bar{\theta}}$: target Q-network, $\bar{\theta}$是$\theta$的exponential moving average (EMA), 用于stabilize training
- $s_{t+1}$: next state
- $\tilde{a}_{t+1} \sim \pi_\phi(\cdot|s_{t+1})$: 从policy采样的next action
- $\alpha$: entropy temperature, 控制exploration强度
- $\pi_\phi$: stochastic policy with weights $\phi$
- $\log \pi_\phi(\tilde{a}_{t+1}|s_{t+1})$: log probability, entropy项

**C51 Distributional critic**:
把return建模成categorical distribution over $N_{\text{atoms}}=51$个bins, 每个bin对应一个value $v_i \in [V_{\min}, V_{\max}]$。

$$Q_\theta(s, a) = \sum_{i=1}^{51} v_i \cdot p_i(s, a; \theta)$$

其中$p_i$是softmax over atoms。Loss变成**cross-entropy**:

$$\mathcal{L}_{Q_\theta} = -\sum_{i=1}^{51} \hat{p}_i(s_t, a_t; \theta) \log p_i(s_t, a_t; \theta)$$

$\hat{p}_i$是从TD target投影回来的target distribution。

**Intuition**: Distributional critic给value function更多expressivity, gradient信号更丰富, 在并行训练时更稳定。虽然compute多一点点, 但convergence快得多, net wall time更短。这个在PQL [https://arxiv.org/abs/2305.18534] 和FastTD3中已经验证, Squint继承到visual domain。

### 2.3 Resolution Squinting ⭐ 这篇paper最有趣的设计

**关键insight**: 直接在16×16渲染 vs 在128×128渲染再area-downsample到16×16, 后者**显著更好**。

| 方案 | 渲染分辨率 | 输入分辨率 | 性能 |
|------|-----------|-----------|------|
| Direct render | 16×16 | 16×16 | baseline |
| **Squinting** | **128×128** | **16×16 (downsampled)** | **更高success rate** |

**为什么squinting有效?**

1. **Anti-aliasing**: 16×16直接渲染会aliase掉高频信息, edges丢失。128×128渲染后area downsampling (average pooling 8×8)相当于一个low-pass filter, 保留low-frequency scene structure。

2. **Scene structure preservation**: 高分辨率渲染后downsample能保留物体的relative geometry, 比如cube的edge方向、gripper的相对位置。

3. **Sim-to-real一致性**: 真实世界相机本来就是高分辨率, 然后ISP downsample。Squinting让sim和real的image formation pipeline更一致。

**Intuition (build yours)**: 想象你眯眼看东西——你看不清细节但能看到大致layout。这个layout信息对manipulation已经够了。直接用低分辨率sensor看, 因为physical sampling方式不同, 反而丢失了重要的edges。

这跟人类peripheral vision的coarse-to-fine hierarchy也类似: fovea高分辨率处理detail, periphery低分辨率处理layout。Squint只用了"periphery"信号, 但通过正确的方式获取它。

**Implementation**: 
```python
# Squinting pipeline
o_high = render(camera, resolution=(128, 128))
o_low = F.avg_pool2d(o_high, kernel_size=8)  # 128/16 = 8
# store o_low in replay buffer (16x16, 极小memory)
```

### 2.4 Layer Normalization everywhere

所有linear layer后面接LayerNorm [https://arxiv.org/abs/1607.06450]。

**为什么LayerNorm帮wall time?**

在parallel RL里, gradient来自大量parallel envs, 信号方差很大。LayerNorm normalize hidden activations, 让gradient更stable, 允许用更大learning rate, 加速convergence。

注意是**LayerNorm不是BatchNorm**——BatchNorm在RL里因为有correlated samples (replay buffer) 会出问题, LayerNorm per-sample normalize更安全。

参考Nauman et al.的"Bigger, regularized, optimistic" [NeurIPS 2024] 也是类似发现。

### 2.5 Encoder设计

```
Image (16x16x3) 
  → Conv2d(32, 3x3, stride=2) → ReLU  
  → Conv2d(32, 3x3, stride=2) → ReLU
  → Flatten → Linear(proj_dim)
  → Shared between actor and critic
```

**关键设计**:
- **Shared encoder** between actor and critic (省一半compute)
- **只通过critic的TD-loss更新encoder**, actor gradient通过stop gradient (sg) 阻断
- Actor和critic有separate 1-layer projection heads

这个设计来自DrQ [https://arxiv.org/abs/2004.13649] 和DrQ-v2 [https://arxiv.org/abs/2107.09645] 的shared encoder pattern。

伪代码中对应:
```
z_t ← f_ψ(ō_t)  # encoder forward
z_t^sg ← stopGRAD(z_t)  # for actor
ã_t ~ π_φ(·|z_t^sg, s_t^proprio)  # actor samples
```

**Intuition**: Critic学到representation对value estimation最优, 通过stop gradient防止actor的policy gradient把representation带偏。Shared encoder让两个网络看同一个"世界"。

### 2.6 PyTorch Optimizations

- **torch.compile**: kernel fusion, 减少Python overhead
- **cudagraphs**: capture CUDA graph, replay避免CPU launch overhead
- **AMP bfloat16**: CNN forward用bf16, 减半memory bandwidth
- **LeanRL/CleanRL base** [https://github.com/vwxyzjn/cleanrl]: single-file implementation

组合起来**5x加速**。

---

## III. SO-101 Task Set: 8个manipulation tasks

| Task | 描述 | 难度 |
|------|------|------|
| Reach Cube | 到cube上方2cm | Easy |
| Reach Can | 到can上方2cm | Easy |
| Lift Cube | 抓起cube到rest position | Medium |
| Lift Can | 抓起can到rest position | Medium |
| Place Cube | 放cube到container | Medium |
| Place Can | 放can到container | Medium |
| Stack Cube | 红cube叠蓝cube | Hard |
| Stack Can | cube叠can (难, can会倒) | Very Hard |

**Task design原则**:
- 0.2m × 0.2m xy随机化范围 (gripper起点附近)
- Wrist camera only (不用third-person)
- 5-DoF SO-101 arm [https://github.com/huggingface/lerobot]

**为什么wrist camera?** Hsu et al. [https://arxiv.org/abs/2203.12677] "Vision-based manipulators need to also see from their hands"证明wrist camera比third-person更适合manipulation, 因为:
1. Object总是in view
2. Representation更object-centric
3. Active vision允许agent主动调整视角

---

## IV. Domain Randomization深度

Squint用了**aggressive randomization**, 但依然15分钟收敛:

### Visual DR
- Camera position perturbation
- Camera rotation perturbation  
- FOV perturbation
- Lighting changes
- **Color jitter** ⭐ (key for real transfer)

### Physical DR
- Object sizes
- Object frictions
- Gripper closing speed

### Proprioceptive noise
- Additive isotropic Gaussian noise $\mathcal{N}(0, 5^2 I)$ 加到joint positions

**Color jitter ablation** (Table IV):

| Variant | Real-world success |
|---------|-------------------|
| Squint (with color jitter) | 73/80 (91.3%) |
| Squint (no color jitter) | 58/80 (72.5%) |

**18% absolute drop!** 说明color jitter对real-world robustness至关重要。

---

## V. 实验结果深度分析

### 5.1 Simulation results (Table I)

| Method | All Tasks Avg |
|--------|---------------|
| **Squint** | **96.1% ± 1.6** |
| SAC (Squint hyperparams, no arch changes) | 88.3% ± 3.4 |
| PPO | 60.2% ± 4.0 |
| DrQ-v2 (sequential) | 4.5% ± 2.9 |
| BC | 41.9% ± 4.2 |

**关键观察**:
1. **Squint vs SAC**: 7.8% gap, 来自架构差异 (distributional critic, squinting, LayerNorm)
2. **SAC vs PPO**: 28.1% gap, off-policy在hard tasks (Place, Stack)完胜
3. **DrQ-v2失败**: 4.5%! 因为DrQ-v2是sequential, 1个env, 在heavy DR下sample efficiency不够。这证明了**parallelization对visual RL的wall time至关重要**
4. **Stack Can**: Squint 81.2%, SAC只有18.7% — distributional critic在bimodal outcomes (成功/失败) 的任务上明显更好

### 5.2 Real-world results (Table II)

| Method | Total | Avg |
|--------|-------|-----|
| **Squint** | **73/80** | **91.3%** |
| SAC | 65/80 | 81.3% |
| PPO | 50/80 | 62.5% |
| DrQ-v2 | 8/80 | 10.0% |
| BC | 38/80 | 47.5% |

**Sim-to-real gap**:
- Squint: 96.1% → 91.3% (-4.8%)
- SAC: 88.3% → 81.3% (-7.0%)
- PPO: 60.2% → 62.5% (+2.3%, 因为sim上已经低了)

Squint的sim-to-real gap最小, 说明squinting + heavy DR让policy更robust。

### 5.3 vs Visual DAgger (Table III)

| Method | Total | Avg |
|--------|-------|-----|
| Squint | 73/80 | 91.3% |
| Visual DAgger (distill from state SAC) | 53/80 | 66.3% |

**25% absolute gap!** 这非常重要——说明**直接visual RL > state-to-visual distillation**在active vision设置下。

**Intuition**: State-based SAC expert探索方式跟visual agent不同。State agent能"看到"all objects, exploration trajectory是direct reach。Visual agent用wrist camera需要active search motion。Distill这种mismatch导致visual student学不好。

---

## VI. Pseudocode逐行解析

```
Algorithm 1: Squint
```

**Line 1-2**: 初始化
- $f_\psi$: encoder (shared)
- $\pi_\phi$: actor
- $Q_{\theta_1}, Q_{\theta_2}$: double critics (clipped double Q)
- $\alpha$: entropy temperature
- $\mathcal{B}$: replay buffer
- $Q_{\bar{\theta}_1}, Q_{\bar{\theta}_2}$: target critics, 初始化copy from main

**Line 4**: 观察并squint
```python
ō_t = avg_pool(render(res=128), k=8)  # 16x16
```

**Line 5**: 采样action
$$a_t \sim \pi_\phi(\cdot | f_\psi(\tilde{o}_t), s_t^{\text{proprio}})$$

**Line 7**: 存transition (note: 16×16 image, 极小memory)

**Line 11-13**: Target Q-value计算
$$y = r_t + \frac{\gamma}{2} \sum_{i=1}^{2} \left( Q_{\bar{\theta}_i}(z_{t+1}, s_{t+1}^{\text{proprio}}, \tilde{a}_{t+1}) - \alpha \log \pi_\phi(\tilde{a}_{t+1}|z_{t+1}, s_{t+1}^{\text{proprio}}) \right)$$

注意**average over double Q**而不是clipped double Q (min)。这个ablation显示average slightly更好。

**Line 15**: Critic + encoder joint update
$$(\psi, \theta_i) \leftarrow (\psi, \theta_i) - \nabla_{(\psi, \theta_i)} \frac{1}{|\mathcal{B}|} \sum_{\tau_k \in \mathcal{B}} (Q_{\theta_i}(z_t, s_t^{\text{proprio}}, a_t) - y)^2$$

Encoder gradient从critic loss来, actor update时stop gradient。

**Line 17**: 自适应entropy temperature
$$\alpha \leftarrow \alpha - \nabla_\alpha \frac{1}{|\mathcal{B}|} \sum (\mathcal{H}^{\text{target}} - \mathcal{H}(z_t, s_t^{\text{proprio}})) \cdot \alpha$$

Target entropy $\mathcal{H}^{\text{target}} = -\dim(\mathcal{A})$ (common SAC choice)。

**Line 18-22**: Actor update (less frequent, policy_freq)
$$\phi \leftarrow \phi + \nabla_\phi \frac{1}{2|\mathcal{B}|} \sum_{\tau_k \in \mathcal{B}} \sum_{i=1}^{2} \left( Q_{\theta_i}(z_t^{\text{sg}}, s_t^{\text{proprio}}, \tilde{a}_t) - \alpha \log \pi_\phi(\tilde{a}_t|z_t^{\text{sg}}, s_t^{\text{proprio}}) \right)$$

注意是**gradient ascent** (maximize Q - α log π), stop gradient on $z_t$。

**Line 24**: Polyak update
$$\bar{\theta}_i \leftarrow \rho \bar{\theta}_i + (1-\rho)\theta_i$$

$\rho$通常0.99或0.995。

---

## VII. Sim-to-Real Transfer细节

### Action scaling
- Sim: 10Hz control, large position offsets (agent可以快速移动)
- Real: **action × 0.15** (safety scaling), **30Hz control** (3x faster)

**为什么不match frequency?** 

Real robot需要**faster recovery control**应对disturbance, 而且30Hz能给smoother trajectories。虽然有frequency mismatch, 但policy学的是relative action, 不是absolute timing, 所以transfer OK。

**Intuition**: Policy学到的是"动作的shape", 不是"每秒多少步"。Higher real frequency = finer-grained execution, 类似video game的high FPS让控制更顺滑。

### Control frequency mismatch的影响
- 10Hz sim → 30Hz real意味着sim中1秒的trajectory, real中1秒执行3倍的micro-actions
- 因为action是delta, 每个real step执行0.15倍的小delta, 累积效果类似sim
- 这是一种隐式的temporal smoothing

---

## VIII. 对你(Andrej)的几条可能interesting的点

### 8.1 "Software 2.0"角度
Squint的15分钟训练让你能像写代码一样iterate on robot policies。从"训一周看一次结果"变成"coffee break回来policy就训好了"。这改变research workflow fundamentally。

### 8.2 Resolution vs Compute trade-off
16×16 = 768 pixels, 比典型visual RL的84×84 (21168 pixels)少**27x**。这跟你讲过的"网络capacity要匹配data complexity"的思想一致。Manipulation不需要pixel-level detail, 需要的是scene layout。

### 8.3 Distributional = Better gradient signal
C51用51个atoms把scalar Q变成distribution, gradient信号丰富51倍(在distribution层面)。这跟classification比regression更stable的intuition一样。

### 8.4 Active vision的imitation limitation
State-to-Visual DAgger失败(66.3% vs 91.3%)说明**observation space决定exploration behavior**。State agent全局视野下exploration是direct, visual agent局部视野下exploration是"peek-and-search"。Distillation无法transfer这种behavioral prior。这跟你在CS231n讲"representation determines what's learnable"一致。

---

## IX. 局限性与未来方向

### 9.1 Visual robustness
- Color jitter带来18%提升, 说明还是很fragile
- 未来: pretrained vision encoders (R3M, VIP, Voltron) [https://arxiv.org/abs/2303.12601]
- 或者SAM-based representations [https://arxiv.org/abs/2312.17116]

### 9.2 Single-task only
- 8个task分别训练, 没有多任务
- 未来: multi-task generalist (跟RLDG, RT-2方向)

### 9.3 Symmetric training
- Critic也用visual input, 没用privileged information
- Asymmetric actor-critic (critic用state, actor用image)可能更快 [https://arxiv.org/abs/1710.06542]

### 9.4 No real data co-training
- Sim-only, 没用real demonstrations
- Sim-and-real co-training [https://arxiv.org/abs/2503.24361]可能进一步加速

### 9.5 Simple FIFO buffer
- 没用prioritized experience replay [https://arxiv.org/abs/1511.05952]
- 1M buffer全装GPU, 但sampling uniform

---

## X. 我的critique与open questions

1. **Stack Can只有81.2%**: 最难任务distributional critic也救不了, 可能需要更长training或更好的exploration

2. **Single robot, single camera setup**: SO-101是toy platform, 真正industrial robot (Franka, UR5)复杂度高得多

3. **15 minutes是cherry-picked?**: "most tasks converging in under 6 minutes"但Stack Can可能需要更久

4. **Replay buffer 1M @ 16×16**: 16×16×3×1M = 768MB, 完全fit GPU, 但这种小image可能丢太多信息, 换更复杂task可能不够

5. **Color jitter 18% gap很大**: 说明policy还是overfit到sim的rendering distribution, 真正robust的visual RL需要更多

6. **Comparison with TD-MPC2 missing**: TD-MPC2 [https://arxiv.org/abs/2310.16828]是SOTA model-based visual RL, 没比较可惜

---

## XI. References汇总

**核心方法**:
- Squint project: https://aalmuzairee.github.io/squint
- SAC: https://arxiv.org/abs/1801.01290
- C51 distributional: https://arxiv.org/abs/1707.06887
- DrQ-v2: https://arxiv.org/abs/2107.09645
- LayerNorm: https://arxiv.org/abs/1607.06450

**Fast RL lineage**:
- FastTD3: https://arxiv.org/abs/2505.22642
- FastSAC: https://arxiv.org/abs/2512.01996
- PQL: https://arxiv.org/abs/2305.18534 (大概)
- PQN: https://arxiv.org/abs/2407.04811

**Simulator & Robot**:
- ManiSkill3: https://arxiv.org/abs/2410.00425
- LeRobot (SO-101): https://github.com/huggingface/lerobot
- CleanRL: https://github.com/vwxyzjn/cleanrl

**Sim-to-real & robustness**:
- Domain randomization: https://arxiv.org/abs/1703.06907
- Color jitter ablation: 在文中
- Active vision: https://arxiv.org/abs/2203.12677

**相关future work**:
- Privileged sensing: https://arxiv.org/abs/2405.14853
- Sim-real co-training: https://arxiv.org/abs/2503.24361
- Pretrained encoders: https://arxiv.org/abs/2212.10351

---

## XII. 最终takeaway

Squint的recipe可以总结为**"正确地获取低分辨率信号 + 大量并行 + 适度reuse"**:

1. **Squinting**: 不是降分辨率, 是用更高分辨率render再downsample获取更clean的低频信号
2. **1024 envs + 256 updates (UTD=0.25)**: 比humanoid的UTD高4x, 因为manipulation需要更准的Q
3. **Distributional critic**: 更rich的gradient信号, 在bimodal outcomes (Stack Can)上明显赢
4. **LayerNorm**: 让大batch并行训练stable
5. **Engineering**: torch.compile + cudagraph + bf16 = 5x

整个pipeline的beauty在于**没有新算法, 全是engineering + careful ablations**。这跟你"build intuition通过具体实现"的哲学完全契合。

15分钟从sim到real, single GPU, zero-shot transfer——这个timescale让robot learning research从"看天吃饭"变成"快速iterate", 是field的paradigm shift。下一个milestone应该是**1分钟训练 + multi-task generalist**, 期待后续工作!

---

如果你对某个具体设计选择想dive deeper (比如C51的projection细节, 或ManiSkill3的batched rendering internals), 我可以继续展开。
