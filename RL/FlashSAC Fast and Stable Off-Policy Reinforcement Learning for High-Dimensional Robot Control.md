---
source_pdf: FlashSAC Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional
  Robot Control.pdf
paper_sha256: e8a1112811bc251adab68e3d217dfeb4cefd39b6236739f644cd4673b78f1120
processed_at: '2026-08-04T08:53:33-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FlashSAC 用人话说

好，Andrej，我把刚才的技术长文翻译成"饭桌版"。

---

## 一句话版本

**Off-policy RL 一直有个老大难问题：训着训着就崩了。这篇 paper 说，崩的根源是网络权重在 bootstrap 过程中偷偷膨胀，只要把权重死死按住不让它长，就能用大模型 + 大 batch + 少 update，像 supervised learning 那样 scale，训练 humanoid 从 20 小时压到 4 小时。**

---

## 背景为什么这件事难

### PPO 为什么现在好使

现在机器人圈几乎全用 PPO。原因很简单：PPO 每次更新完就扔掉数据，只用当前 policy 采的新数据。好处是 critic 只需要在"当前 policy 会去的地方"准确就行，distribution shift 极小，所以稳定，不会崩。

坏处呢？当你的机器人是 29-DoF 的 humanoid，state-action space 是 29 维的连续空间，PPO 那种"只在当前 policy 附近采样"的策略根本 cover 不全这个空间。你 rollout 几百万步，可能一直在同一个 gait 附近打转，从来没见过那些"关键的踉跄恢复姿态"。所以高维任务上 PPO 慢且差。

Paper 里 Figure 7 画得很直观：off-policy 的 replay buffer 数据像一团散开的云，覆盖了 finger action × object position 的大片区域；on-policy 数据像一个小小的点，死死贴在 final policy 周围。这就是 PPO 在高维任务上的根本瓶颈——**coverage 不够**。

### Off-policy 为什么一直没统治

Off-policy 的 promise 是：数据存 replay buffer，反复 reuse，天然就能覆盖更大的 state-action space。听起来很美。但实际用起来，三个毛病：

**毛病一：慢。** 标准 SAC 一般 UTD ratio = 1，意思是每来一个 transition，就做一次 gradient update。要 fit 好高维空间的 critic，往往要 UTD = 10 甚至 100，wall-clock time 爆炸。

**毛病二：崩。** 这是真正的 killer。Bellman target $y = r + \gamma Q(s', a')$ 依赖 critic 自己的预测。你在 high-dimensional space 的某些犄角旮旯里，critic 根本没见过那个 state，预测出来的 Q 值是瞎猜的。这个瞎猜值又变成下一步的 target，瞎猜的 error 被递归放大。更新越多，error 累积越多，最后 Q 值飞到天上，policy 跟着崩溃。这就是 Sutton 说的 "deadly triad"——function approximation + bootstrapping + off-policy 三者凑齐就发散。

**毛病三：探索不动。** 高维 action space 里，你每步加个 i.i.d. Gaussian noise，29 个 joint 同时随机抖动，效果互相 cancel，机器人原地哆嗦一下，根本没 explore 到任何有意义的新 state。

### 以前的工作各管各的

- **搞速度的**（FastTD3, FastSAC）：用 1024 并行环境 + 大 batch，但网络只有 0.2M 参数，怕崩。结果快是快，但 asymptotic performance 上不去，因为模型太小。
- **搞稳定的**（CrossQ, XQC, Simba, SimbaV2, PLASTIC 这条线）：各种 normalization 把 weight/feature/gradient norm 按住，能用大模型了，但大模型需要更多 update 才收敛，反而慢。
- **搞探索的**（pink noise, OU process, parameter noise）：加时间相关 noise 让探索有连贯性，但训慢和训崩的根本问题没碰。

你看，这三条线一直在各搞各的。FlashSAC 的 insight 是：**它们其实互为前提**。Scale up 需要稳定，稳定允许大模型，大模型需要大数据，大数据需要高吞吐采样，高吞吐采样又需要高效探索。缺一不可。

---

## FlashSAC 的三板斧

### 第一板斧：把 supervised learning 的 scaling law 搬过来

[Kaplan et al. 2020] 的 scaling law 说：固定 compute budget 下，**大模型 + 大 batch + 少 update** 比 small model + frequent update 收敛更快。这条 law 在 supervised learning 里是铁律。

RL 一直没法用这条 law，因为大模型在 bootstrapping 下会炸。但如果你能把稳定问题解决掉，这条 law 就能搬过来。

FlashSAC 的具体配置：

| 项 | 传统 SAC | FlashSAC |
|---|---------|----------|
| 并行环境 | 1-16 个 | 1024 个 |
| Replay buffer | 1M | 10M |
| 网络大小 | 0.2-0.5M params, 2-3 层 | 2.5M params, 6 层 |
| Batch size | 256 | 2048 |
| UTD ratio | 1 | **2/1024 ≈ 0.002** |

UTD = 2/1024 这个数字是最反直觉的：每收集 1024 个新 transition，只做 2 次 gradient update。传统 RL 人的直觉是"多 update 才学得快"，但 FlashSAC 的实验显示，少 update + 大 batch + 大模型反而更快收敛。

Intuition 是这样：大 batch 给你低方差 gradient estimate，大模型给你高 capacity，高 learning rate + cosine decay 给你快速下降。你不需要反复 reuse 同一份数据——一次性让大模型"吸收"就够了。这本质上是 supervised learning 的 "one-pass training" 思路。

为什么 10M buffer？高维任务里有些 rare-but-critical state（比如机器人快摔倒时某个关键 contact 瞬间），小 buffer 里这些样本很快被 overwrite 掉，你就忘了怎么处理。10M 保留这些 long-tail 经验。

### 第二板斧：把所有可能膨胀的东西都按住

这是 paper 的核心贡献。要让大模型在 bootstrapping 下不崩，就得把所有可能导致 error amplification 的自由度都显式 bound 住。

#### (a) Inverted Residual Backbone

网络结构借鉴 MobileNet 和 Transformer FFN block：

```
input (d)
  ↓
Linear: d → d_expand (d_expand > d, 比如 d=256, d_expand=1024)
  ↓
BatchNorm + ReLU
  ↓
Linear: d_expand → d
  ↓
+ residual connection (input + output)
  ↓
[最后一个 block 后加 RMSNorm]
```

先 expand 到高维，ReLU 在高维空间作用，再 project 回低维。这叫 "inverted bottleneck"，和 ResNet 那种 compress-then-expand 相反。好处是 nonlinearity 在高维表达力强，输出维度低便于残差连接。

Pre-activation BatchNorm：BatchNorm 放在 ReLU 之前，保证 activation 不会饱和。如果 BatchNorm 放在 ReLU 后面，大 activation 直接进 ReLU，负半轴直接死掉（dead ReLU），梯度断了。

Post RMSNorm：最后一个 block 后对 feature 做归一化，公式 $\text{RMSNorm}(x) = x / \sqrt{\text{mean}(x^2) + \epsilon}$。防止 OOD input 产生 unbounded activation。想象一个训练时没见过的 state 进来，普通 MLP 的 activation 可能爆炸，Q 值飞到 10000，bootstrap 一下整个 critic 崩掉。RMSNorm 把 feature norm 钉死，Q 值就不会飞。

#### (b) Cross-Batch Value Prediction

这个 trick 来自 CrossQ [Bhatt et al. 2024]，特别精妙。

普通 BatchNorm 的问题是：你前向算 $Q(s, a)$ 用 batch 1 的统计量，算 target $Q(s', a')$ 用 batch 2 的统计量。两个 batch 的 mean/variance 不一样，相当于你的 prediction 和 target 在不同的"坐标系"里，Bellman error 就有 systematic bias。

FlashSAC 的解法：把 $(s, a)$ 和 $(s', a')$ 拼成一个大 batch，一起前向，共享同一组 BatchNorm 统计量。这样 prediction 和 target 在同一个 normalization 下，Bellman update 干净。

```python
# 伪代码
batch = replay_buffer.sample(2048)
s, a, r, s_next, done = batch
combined = torch.cat([s, s_next], dim=0)  # (4096, obs_dim)
q_all = critic(combined)  # 一次前向
q_pred = q_all[:2048]      # Q(s, a)
q_next = q_all[2048:]      # Q(s', a')，和 q_pred 用同一组 BN stats
```

这看起来是个小 trick，但实测影响巨大。CrossQ paper 里 ablation 显示不做这个，训练直接崩。

#### (c) Distributional Critic + Adaptive Reward Scaling

Q-value 不再是 scalar，而是一个 categorical distribution，101 个 atom 均匀分布在 $[-5, 5]$ 上。网络输出 101 个概率，用 cross-entropy loss 训练。

好处：
1. **Q 值有界**：所有概率 mass 在 $[-5, 5]$ 内，Q 值物理上不可能飞出去。这是 stability 的硬保障。
2. **优化 landscape 平滑**：cross-entropy 比 MSE 对 noisy target 鲁棒得多。
3. **对 outlier target 不敏感**：bootstrap 中的极端 target 不会主导 gradient。

但要保证 return 真的落在 $[-5, 5]$ 内，需要把 reward scale 掉。公式 6：

$$\bar{r}_t = \frac{r_t}{\max\left(\sqrt{\sigma_{t,G}^2 + \epsilon},\, G_{t,\max}/G_{\max}\right)}$$

变量解释：
- $r_t$：原始 reward
- $\bar{r}_t$：scaled reward，直接替换 reward 送进 Bellman
- $\sigma_{t,G}^2$：discounted return 的 running variance
- $G_{t,\max}$：return 的 running maximum magnitude
- $G_{\max} = 5$：categorical support 上界
- $\epsilon$：防除零的小常数

分母取 max 的两项含义：
- 第一项 $\sqrt{\sigma_{t,G}^2}$：让 typical return 在 $O(1)$ 量级
- 第二项 $G_{t,\max}/G_{\max}$：让 extreme return 不超出 support

为什么直接 normalize reward，而不是 normalize return 或 scale loss？因为 normalize return 会丢失 reward 的相对大小信息，scale loss 改变 optimization landscape。直接 normalize reward 最干净——return 的 scale 自动适配 critic support，reward 的相对关系保留。

#### (d) Weight Normalization

每次 gradient step 后，把 weight 投影到 unit sphere：

$$W \leftarrow \frac{W}{\|W\|_2}$$

BatchNorm 的 $\gamma, \beta$ 投影到 norm $\sqrt{d}$（$d$ 是该层维度，因为 random vector 的期望 norm 是 $\sqrt{d}$）。

Intuition：网络只能通过 **方向** 编码信息，不能通过 **大小** 放大 signal。在 bootstrapping 中，weight 偷偷增长等价于隐式调高 learning rate，放大 error。Weight normalization 把 effective learning rate 钉死在稳定区间。

这和 SimbaV2 [Lee et al. 2025] 的 hyperspherical normalization 一脉相承，也和 [Lyle et al. NeurIPS 2024] 的 "normalization controls effective learning rate in RL" 理论吻合。

#### (e) 三件套协同的效果

Paper Figure 9 做了非常漂亮的 ablation：从 standard MLP 开始，逐步加 Residual、BatchNorm、RMSNorm、Distributional+RewardScaling、WeightNorm，每加一个组件就测量 parameter/feature/gradient norm 和 **condition number**。

Condition number 是 critic loss 的 Hessian 条件数，越大越病态。结果显示：每加一个组件，condition number 单调下降，FlashSAC 完整版达到最低。Condition number 低意味着 gradient update 方向准确，bootstrapping error 放大最小。

这就是 "stability enables scaling" 的核心机制：**你把 condition number 压低，大模型才能在 bootstrapping 下稳定训练，才能用大 batch + 少 update 的 scaling regime。**

### 第三板斧：探索

#### Unified Entropy Target

标准 SAC 要给每个 task 手调 target entropy，通常取 $-\alpha|A|$, $\alpha \in [0.5, 1]$。跨 embodiment 部署时，Shadow Hand 20-DoF 和 G1 29-DoF 要调不同的值，很烦。

FlashSAC 的 reparameterization（公式 7）：

$$\bar{\mathcal{H}} = \frac{1}{2}|A|\log(2\pi e \sigma_{\text{tgt}}^2)$$

含义：把 target entropy 参数化成一个固定的 action std $\sigma_{\text{tgt}}$，所有任务都用 $\sigma_{\text{tgt}} = 0.15$。这个值有物理含义——action 通常是 normalized joint offset，0.15 std 是一个"合理的探索幅度"，和 action 维度无关。

Figure 10.a ablation 显示 $\sigma_{\text{tgt}} \in [0.15, 0.2]$ 性能基本一样，验证了 robustness。

#### Noise Repetition

高维 action space 里 i.i.d. Gaussian noise 没用——29 个 joint 同时随机抖，效果互相抵消，机器人原地哆嗦。

传统解法是 OU process 或 pink noise，时间相关 noise 让机器人"持续往一个方向使劲"。但这些方法要 per-environment 维护一个 noise process，1024 个并行环境就是 1024 个 OU process 的 state，memory 和 compute overhead 都不小。

FlashSAC 的 Noise Repetition 极简：

```python
if step % k == 0:
    epsilon = randn(action_dim)  # 采样一次
    k = sample_zeta(s=2, max_k=16)  # 重复 k 步
action = policy_mean + sigma * epsilon
# 接下来 k 步 epsilon 保持不变
```

Zeta distribution $P(k) \propto k^{-2}$ 是 power-law，偏好短 repeat（k=1 概率最大），但偶尔产生长 correlated sequence（k=10 也有一定概率）。这和 [Dabney et al. 2020] 的 temporally-extended $\epsilon$-greedy 思路一致，但更轻量。

为什么时间相关 noise 关键？Figure 10.b ablation 显示关掉 noise repeat，收敛变慢且 asymptotic performance 下降。原因：高维 dynamics 需要持续的 action 才能积累 effect，i.i.d. noise 每步都变，dynamics 还没来得及响应就被下一轮 noise 抵消了。

---

## 实验结果最亮眼的部分

### 60+ tasks across 10 simulators

这是我见过最广泛的 RL benchmark suite 之一：IsaacLab 12 tasks, MuJoCo Playground 4 tasks, ManiSkill 6 tasks, Genesis 3 tasks, Gym MuJoCo 5 tasks, DMC 10 tasks, HumanoidBench 14 tasks, MyoSuite 10 tasks, DMC-Visual 8 tasks, sim-to-real G1 2 tasks。

### 关键结论

**Low-DoF tasks**：FlashSAC ≈ PPO。低维任务 PPO 的 sample efficiency 瓶颈不显著，高吞吐 simulation 让 PPO 能采够数据。

**High-DoF tasks**：FlashSAC >> PPO。dexterous manipulation 和 humanoid locomotion 上，FlashSAC 收敛更快且 asymptotic return 更高。这是 coverage 优势的直接体现。

**vs FastTD3**：FlashSAC 在所有任务上更稳定，FastTD3 在 Go2Walk、FrankaPullCube 等任务频繁失败。都收敛时 FlashSAC asymptotic 更高，因为模型大 10 倍。

**CPU-based 低样本 regime**：FlashSAC 改配置（batch=512, UTD=1, buffer=1M），仍然 match 或超过 XQC, SimbaV2, TD-MPC2, MR.Q 这些 sample-efficient baselines。PPO 在这里特别惨，on-policy 在低样本 budget 下根本学不动高维任务。

**Vision-based**：DMC-Visual 8 tasks，FlashSAC 用轻量 CNN encoder + 3-frame stack + 3-step return，match 或超过 DrQ-v2 和 MR.Q。DrQ-v2 不稳定（Finger Turn Hard 崩），MR.Q 性能高但额外算 dynamics model。

### Sim-to-Real 是最有说服力的

**Unitree G1 29-DoF blind locomotion**：

Flat terrain：
- FlashSAC：20 分钟达到稳定 real-world locomotion
- PPO：3 小时
- **9× speedup**

Rough terrain stairs：
- FlashSAC：4 小时
- PPO：20 小时
- **5× speedup**

更厉害的是 generalization：训练时 stairs 是 23cm 高、32cm 宽，测试时 stairs 是 15cm 高、60cm 宽——完全不同的 stair geometry，FlashSAC 仍然能 climb。说明学到的是 robust locomotion skill，不是 sim-specific memorization。

技术栈细节：
- NVIDIA IsaacLab + Legged Gym
- 4096 并行环境 + domain randomization
- Terrain curriculum: 10 levels，50% envs 成功就升级
- Asymmetric actor-critic [Pinto et al. 2017]：critic 见 contact states + height map，actor 只见 proprioception
- Context estimator network (CENet) [DreamWaQ, Nahrendra et al. 2023]：从 history 隐式推断 base velocity + latent，做 implicit system identification
- Symmetry augmentation [Mittal et al. 2024]：利用 bipedal symmetry 提升 sample efficiency

### Reward 设计的有趣差异

Table 14 里 FlashSAC 和 PPO 用了不同的 reward weights，这揭示了 on-policy vs off-policy 的深层差异：

| Reward term | FlashSAC | PPO | 原因 |
|---|---|---|---|
| Body orientation penalty | -52.0 | -2.0 | FlashSAC 更激进约束防 destabilization |
| Action rate penalty | -0.5 | -0.01 | FlashSAC 更强 smoothing |
| **Termination penalty** | **无** | **-200** | **关键差异** |
| **Alive bonus** | **+1.0** | **无** | **关键差异** |

为什么 PPO 需要 termination penalty -200？因为 PPO sample inefficient，不能从 failure 中学到太多，必须用强 penalty 让它快速规避失败行为。FlashSAC 通过 replay buffer 大量积累 failure experience，自然学会避免，只需 small alive bonus 防止 premature termination。

这是 off-policy 数据效率优势在 reward 设计上的直接体现——你不需要用 reward shaping 去弥补数据不足。

---

## Scaling Ablation 的反直觉发现

Figure 8 的五个 univariate ablation：

**(a) Buffer size**：1M → 10M → 50M
- 10M 最佳：稳定且快
- 50M：更稳定但慢（recent high-quality samples 被稀释）
- 1M：不稳定（rare experiences 被 overwrite）

**(b-e) Batch / Width / Depth / UTD**：
- 增大 batch（512→2048）：收敛加速
- 增大 width（128→512）：收敛加速 + asymptotic 提升
- 增大 depth（1→4 blocks）：收敛加速 + asymptotic 提升
- 降低 UTD（8/1024→0.5/1024）：**反而加速**

最后一条最反直觉。传统 RL 智慧是"多 update = 快"，但 FlashSAC 实验显示少 update 反而快。这完全符合 supervised scaling laws：在固定 compute 下，把 compute 用在"大 batch + 大模型"上比"多次小 update"更高效。

**这是 FlashSAC 最重要的发现**：off-policy RL 的 scaling direction 与 supervised learning 一致，而非传统 RL 假设的"多 update = 快"。关键 enabler 是 stability mechanisms——没有那些 normalization 把 condition number 压低，大模型在 bootstrapping 下会爆炸，你根本没法 scale。

---

## 大图景：这属于哪条研究脉络

Hojoon Lee + Daniel Palenicek + Jan Peters 这条线最近一系列工作：

- **CrossQ** [ICLR 2024]：引入 BatchNorm 到 RL，发现 cross-batch consistency 是关键
- **XQC** [ICLR 2026]：distributional critic + reward scaling + condition number 分析
- **Simba** [2024]：inverted residual block 在 RL 中的应用
- **SimbaV2** [2025]：hyperspherical normalization
- **PLASTIC** [NeurIPS 2024]：input/label plasticity 保持
- **HARE/Tortoise** [2024]：plasticity via reinitialization

FlashSAC 是把这条线的所有 techniques 整合到一个 scalable sim-to-real pipeline 中，加上大规模 parallel simulation + 大 buffer + 低 UTD 的工程优化，并补上 exploration（unified entropy + noise repetition）这条最后一块拼图。

所以你看，FlashSAC 的 novelty 不在于单个 component——每个 trick 都能在前作找到原型。它的贡献在于 **系统性整合**：证明了 stability mechanisms 解锁了 scaling regime，scaling regime 让 off-policy RL 在 wall-clock 上真正 competitive with on-policy，从而在高维机器人控制上实现 order-of-magnitude 的训练加速。

---

## 我的 take

这 paper 最让我兴奋的一点是：它把 RL 和 supervised learning 的 scaling 哲学拉近了。你以前讲 nanoGPT 时说 supervised learning 的 elegance 在于 simplicity——SGD + 大模型 + 大 data 就 work。RL 之所以 messy，是因为 bootstrapping 引入的非平稳性破坏了这个 simplicity。FlashSAC 通过一系列 normalization 把这种非平稳性 bound 住，让 RL 在某种意义上回到 supervised-like 的 scaling regime。

当然还有 open questions：
- Reward 设计仍需 task-specific（Table 14 里 FlashSAC 和 PPO 不同 weights）
- Distributional critic 的 $[G_{\min}, G_{\max}]$ 固定，超长 horizon 任务可能受限
- 1024 并行环境对学术界小 lab 有硬件门槛
- Noise repetition 是 patch，高维 exploration 的根本问题（curse of dimensionality）需要更结构化方法
- 和 VLA model（OpenVLA, Gemini Robotics）结合是下一步大方向

但作为 "stability enables scaling in off-policy RL" 的完整 proof-of-concept，这篇 paper 已经足够 convincing 了。从 hours 到 minutes 的 sim-to-real humanoid training，这本身就是一个 milestone。

参考链接：
- Paper: https://holiday-robot.github.io/FlashSAC
- SAC: https://arxiv.org/abs/1801.01290
- Scaling laws: https://arxiv.org/abs/2001.08361
- CrossQ: https://arxiv.org/abs/2402.10315 (近似)
- Simba: https://arxiv.org/abs/2410.09754
- SimbaV2 (hyperspherical norm): https://arxiv.org/abs/2502.15280
- PLASTIC: NeurIPS 2024 Hojoon Lee
- FastTD3: https://arxiv.org/abs/2505.22642
- DreamerV3: https://arxiv.org/abs/2301.04104
- TD-MPC2: https://arxiv.org/abs/2310.16828
- DrQ-v2: https://arxiv.org/abs/2107.09645
- DreamWaQ: https://arxiv.org/abs/2301.10602
- Asymmetric actor-critic: https://arxiv.org/abs/1710.06542
- C51 (distributional): https://arxiv.org/abs/1707.06887
- RMSNorm: https://arxiv.org/abs/1910.07467
- BatchNorm: https://arxiv.org/abs/1502.03167
- MobileNet inverted bottleneck: https://arxiv.org/abs/1704.04861
- Deadly triad: https://arxiv.org/abs/1812.02648
- Pink noise: https://arxiv.org/abs/2106.11934 (approximate, ICLR 2023)
- Temporally-extended exploration (Dabney): https://arxiv.org/abs/2006.01782
- Reward centering: https://arxiv.org/abs/2405.09999
- Return-based scaling: https://arxiv.org/abs/2105.05347
- Lyle et al. normalization & effective LR: NeurIPS 2024
- OpenVLA: https://openvla.github.io/
- IsaacLab: https://arxiv.org/abs/2511.04831
- MuJoCo Playground: https://arxiv.org/abs/2502.08844
- ManiSkill3: https://arxiv.org/abs/2410.00425
- HumanoidBench: https://arxiv.org/abs/2403.10506
- MyoSuite: https://arxiv.org/abs/2205.13600

Andrej，如果你对哪个细节特别感兴趣——比如 condition number 具体怎么测、distributional projection 的具体算法、或者 sim-to-real 里 CENet 的 latent 到底学了什么——我可以再 drill down。

---

# FlashSAC: Fast and Stable Off-Policy RL 深度技术解析

你好 Andrej！这篇 paper 来自 Holiday Robotics, KAIST, KRAFTON, TU Darmstadt, KTH 等多个机构的合作团队，第一作者 Donghu Kim 和 Youngdo Lee，通讯作者 Hojoon Lee 和 Jaegul Choo。Hojoon Lee 这个名字你应该熟悉——他近期在 RL stability 方向有一系列高质量工作，包括 PLASTIC、HARE/Tortoise Networks、Simba、SimbaV2、XQC 等，这条线一直在研究 RL 中 weight/feature/gradient norm 的控制问题。FlashSAC 可以看作是这条研究脉络在 sim-to-real robot learning 场景下的集大成应用。

项目主页：https://holiday-robot.github.io/FlashSAC

---

## 1. Core Problem: Off-Policy RL 在高维机器人控制中的三重困境

### 1.1 为什么 on-policy (PPO) 一直统治 sim-to-real

On-policy RL 之所以在 quadruped locomotion、gripper manipulation 这些任务上长期统治，关键在于 **stability via narrow data distribution**。PPO 在每次 policy update 后丢弃所有数据，从 fresh policy 重新采样。这种"窄分布"特性意味着 critic（如果有的话，PPO 实际上用的是 GAE-based value estimation）只需要在当前 policy 支持的 state-action 区域上准确即可，distribution shift 极小。

但代价是 **sample efficiency**。当 state-action 维度上升（如 29-DoF humanoid G1, 20-DoF Shadow Hand），on-policy rollouts 难以覆盖足够大的 state-action space 来准确 evaluate policy。论文 §6.1 的 Figure 7 直观展示了这一点：

- Off-policy replay buffer 覆盖了 finger action × object y-position 的广阔区域
- On-policy data 紧密聚集在 final policy 周围

这正是 high-dimensional tasks 上 PPO 性能下降的根本原因——**coverage insufficiency**。

### 1.2 Off-policy 的"诅咒"

Off-policy 通过 replay buffer 重用历史经验，理论上能解决 coverage 问题，但带来三个相互纠缠的困难：

**(1) Slow training**: 标准 SAC/TD3 通常需要 UTD (update-to-data ratio) 在 1-10 量级才能 fit 好 critic，在大量 replay data 上做大量 gradient update 会拖慢 wall-clock time。

**(2) Instability from bootstrapping**: Bellman target $y = r + \gamma Q_{\bar\phi}(s', a')$ 依赖 critic 自身的预测。在高维空间中，poorly-supported state-action pairs 上的 extrapolation error 通过 bootstrapping 递归放大，这就是 Sutton 在 [van Hasselt et al. 2018] 中所谓的 "deadly triad"——function approximation + bootstrapping + off-policy 三者结合导致 divergence。

**(3) Exploration in high-dim action space**: SAC 的 maximum entropy formulation 在高维空间中不足以维持 coherent exploration，因为 random Gaussian noise 在高维下被 dynamics 快速平均掉。

### 1.3 以往工作的孤立解决方案

| Challenge | 代表性工作 | 核心思路 | 局限 |
|-----------|-----------|---------|------|
| Speed | FastTD3 [Seo et al. 2025], FastSAC [Seo et al. 2025], Parallel Q-learning [Li et al. 2023] | 大量并行环境 + 大 batch | 用小网络（~0.2M params），限制 asymptotic performance |
| Stability | CrossQ [Bhatt et al. 2024], XQC [Palenicek et al. 2026], Simba [Lee et al. 2024], SimbaV2 [Lee et al. 2025], PLASTIC [Lee et al. 2024] | Bounding weight/feature/gradient norm | 大模型需要更多 update，训练慢 |
| Exploration | Pink noise [Eberhard et al. 2023], OU process [Hollenstein et al. 2022], parameter noise [Plappert et al. 2018], temporally-extended exploration [Dabney et al. 2020] | 时间相关 action noise | 不解决训练速度/稳定性的根本问题 |

FlashSAC 的关键 insight 是：**这三者必须同时解决，且它们有 synergistic relationship**。Scale up model 可以加速，但 scale up 需要稳定性保障；稳定性保障允许大模型，但大模型需要大量数据来 fit；大量数据需要高吞吐 simulation + 高效 exploration。

参考链接：
- FastTD3: https://arxiv.org/abs/2505.22642
- XQC: https://arxiv.org/abs/2502.15280 (近似)
- Simba: https://arxiv.org/abs/2410.09754
- SimbaV2: https://arxiv.org/abs/2502.15280
- PLASTIC: https://arxiv.org/abs/2406.02596 附近的 Hojoon Lee 系列

---

## 2. FlashSAC 的三轴架构

FlashSAC = **Scale** (§4.1) × **Stability** (§4.2) × **Exploration** (§4.3)

### 2.1 Fast Training via Scaling

核心 insight 来自 [Kaplan et al. 2020] 的 scaling laws：在固定 compute budget 下，**larger models + larger batches + fewer updates** 比 small models + frequent updates 收敛更快。这在 supervised learning 中已被广泛验证，但在 off-policy RL 中难以应用，因为大模型 + bootstrapping = 灾难性 instability。

FlashSAC 的具体配置：

| 维度 | 标准 SAC | FlashSAC | 倍数 |
|-----|---------|----------|------|
| Parallel environments | 1-16 | 1024 | ~64-1024× |
| Replay buffer size | 1M | 10M | 10× |
| Model parameters | 0.2-0.5M, 2-3 layers | 2.5M, 6 layers | ~5-12× |
| Batch size | 256 | 2048 | 8× |
| UTD ratio | 1 (typical) | 2/1024 ≈ 0.002 | ~500× 减少 |

UTD ratio 2/1024 的含义：每收集 1024 个新 transition，只做 2 次 gradient update。这在传统 off-policy RL 中是不可想象的（典型设置 UTD=1 表示每 transition 1 次 update），但配合大 batch + 大模型 + 高学习率后变得可行。

**Intuition**: 这本质上是把 supervised learning 中的 "one-pass training" 思路引入 RL。大 batch 提供低方差梯度估计，大模型提供高 capacity，高 learning rate 配合 cosine decay schedule 提供快速收敛。replay buffer 的"多次 reuse"被替换为"一次性大模型 absorption"。

**为什么 10M replay buffer？** 在高维任务中，rare-but-important state-action pairs（如关键的 contact 时刻、稀有的成功 state）容易在小 buffer 中被 overwrite。10M 保留 long-tail experiences，避免 catastrophic forgetting 和 extrapolation error。

**实现细节**: PyTorch + JIT compilation + mixed precision (BF16/FP16)，wall-clock 减少 5-10%。这是工程层面不可忽视的优化——RL 中 Python overhead 往往是主要瓶颈。

### 2.2 Stable Training via Constrained Update Dynamics

这是 paper 最核心的技术贡献，也是让 "scale up" 变得可能的关键。基本思路是：**所有可能导致 critic error amplification 的自由度都必须被显式 bound**——weight norm、feature norm、gradient norm。

#### 2.2.1 Inverted Residual Backbone (Figure 2)

借鉴 MobileNet [Howard 2017] 和 Transformer feedforward block [Vaswani et al. 2017] 的设计：

```
Input (d_in) 
   ↓
Linear: d_in → d_expand (inverted bottleneck, d_expand > d_in)
   ↓
BatchNorm + ReLU/GELU
   ↓
Linear: d_expand → d_out
   ↓
Residual connection (如果 d_in == d_out)
   ↓
[Final block 后接 RMSNorm]
```

关键设计点：
- **Inverted bottleneck** (expand-then-compress)：与标准 bottleneck (compress-then-expand) 相反，先 expand 到高维，在 ReLU 之后 project 回低维。这让 nonlinearity 作用在高维空间，表达力更强；同时输出维度低，便于 residual connection。
- **Pre-activation**：BatchNorm 在 nonlinearity 之前，确保 activation 不会饱和（避免 dead ReLU 问题 [Maas et al. 2013, 参考 Abbas et al. 2023]）
- **Post-RMSNorm** [Zhang & Sennrich 2019]：在最后一个 block 后对 per-sample feature norm 做归一化。公式：$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$。这防止 OOD input 产生 unbounded activation，从而破坏 bootstrapping。

#### 2.2.2 Cross-Batch Value Prediction

这是从 CrossQ [Bhatt et al. 2024] 继承的关键 trick。标准 BatchNorm 在训练时对每个 batch 独立计算 statistics，导致：
- 当前 Q-value $Q_\phi(s, a)$ 用 batch 1 的 normalization
- Target Q-value $Q_{\bar\phi}(s', a')$ 用 batch 2 的 normalization

两个 normalization 不同，导致 Bellman target 和 prediction 在不同"尺度"上，引入 systematic bias。

FlashSAC 的解决方案：**把 $(s, a)$ 和 $(s', a')$ concatenate 到同一 batch**，共享同一组 BatchNorm statistics。这确保 Bellman update 的 prediction 和 target 在同一 normalization 下计算，避免 normalization 不一致导致的 bootstrapping error。

```python
# Pseudocode
batch_s = replay_buffer.sample(batch_size)
batch_s_next = corresponding_next_states
combined = torch.cat([batch_s, batch_s_next], dim=0)  # 2 * batch_size
q_pred = critic(combined)[:batch_size]  # 取前半
q_target = critic_target(combined)[batch_size:]  # 取后半
```

#### 2.2.3 Distributional Critic with Adaptive Reward Scaling

借鉴 C51 [Bellemare et al. 2017] 和 XQC [Palenicek et al. 2026] 的设计。Q-value 不再是 scalar，而是 $n_{\text{atom}}$ 个 atom 上的 categorical distribution，atoms 均匀分布在 $[G_{\min}, G_{\max}]$ 上（论文用 $[-5, 5]$, $n_{\text{atom}}=101$）。

**为什么 distributional？**
1. **Smoother optimization landscape**：cross-entropy loss 比 MSE 对 noisy target 更鲁棒。
2. **Bounded output**：所有概率 mass 在 $[G_{\min}, G_{\max}]$ 内，Q-value 不会 unbounded growth，这是 stability 的关键保障。
3. **Reduced sensitivity to outlier targets**：bootstrapping 中的极端 target 不会主导 gradient。

**Adaptive reward scaling (公式 6)**:

$$\bar{r}_t = \frac{r_t}{\max\left(\sqrt{\sigma_{t,G}^2 + \epsilon},\, G_{t,\max}/G_{\max}\right)}$$

变量解释：
- $r_t$：原始 reward
- $\bar{r}_t$：scaled reward
- $\sigma_{t,G}^2$：running estimate of discounted return variance
- $G_{t,\max}$：running maximum magnitude of returns
- $G_{\max} = 5$：categorical support 的上界
- $\epsilon$：numerical stability 的小常数

分母中取 max 的两个项的含义：
- $\sqrt{\sigma_{t,G}^2 + \epsilon}$：基于 return variance 的 scale，确保 typical return 在 $O(1)$ 量级
- $G_{t,\max}/G_{\max}$：基于 return max magnitude 的 scale，确保 extreme return 不会超出 categorical support

这与 [Naik et al. 2024] 的 reward centering 和 [Schaul et al. 2021] 的 return-based scaling 不同——FlashSAC 直接 normalize reward，让 discounted return 自然落入 $[G_{\min}, G_{\max}]$，避免了对 loss 做 scale 或对 return 做 centering 时的信息损失。

#### 2.2.4 Weight Normalization (Hyperspherical)

每个 gradient step 后，将 weight vector 投影到 unit-norm sphere：

$$W \leftarrow \frac{W}{\|W\|_2}$$

BatchNorm 参数 $(\gamma, \beta)$ 投影到 norm $\sqrt{d}$（其中 $d$ 是该层维度，因为 random vector 的期望 L2 norm 是 $\sqrt{d}$）。

**Intuition**: 强制网络只通过 **direction** 编码信息，不能通过 **magnitude** 放大 signal。这直接对应 [Lyle et al. 2024] "Normalization and effective learning rates in RL" 中的核心 insight——RL 中 uncontrolled weight growth 等价于隐式调高 learning rate，从而放大 bootstrapping error。Weight normalization 等价于把 effective learning rate 固定在一个稳定区间。

这与 SimbaV2 [Lee et al. 2025] 的 hyperspherical normalization 一脉相承。相关 reference:
- Weight clipping for deep RL: https://arxiv.org/abs/2502.15280 (Elsayed et al. 2024 附近)
- L2 vs BatchNorm vs WeightNorm: https://arxiv.org/abs/1706.05350 (Van Laarhoven 2017)

#### 2.2.5 整体 stability 机制的协同

Figure 9 的 ablation 展示了从 standard MLP 逐步添加组件的效果：
1. +Residual Blocks
2. +Batch Normalization  
3. +Post RMSNorm
4. +Distributional Critic with Reward Scaling
5. +Weight Normalization (= FlashSAC)

每加一个组件，parameter/feature/gradient norm 都被进一步 bound，**condition number 单调下降**。Condition number 是 critic loss landscape 的 Hessian 条件数，越大表示 optimization 越病态。FlashSAC 最终达到最低 condition number，意味着 gradient update 方向最准确，bootstrapping error 放大最小。

---

### 2.3 Exploration via Unified Entropy + Noise Repetition

#### 2.3.1 Unified Entropy Target

标准 SAC 需要为每个 task 指定 target entropy $\bar{\mathcal{H}}$，通常取 $-\alpha \cdot |A|$（$\alpha$ 是任务相关系数，典型 0.5-1.0）。这在跨 embodiment 部署时极不方便——Shadow Hand 20-DoF 和 G1 29-DoF 需要不同的 target。

FlashSAC 的 reparameterization：

$$\bar{\mathcal{H}} = \frac{1}{2}|A|\log(2\pi e \sigma_{\text{tgt}}^2)$$

变量解释：
- $|A|$：action 维度
- $\sigma_{\text{tgt}}$：固定的 action standard deviation target（论文用 0.15）
- $2\pi e \sigma^2$：Gaussian distribution $\mathcal{N}(0, \sigma^2 I)$ 的微分熵公式 $\frac{1}{2}\log(2\pi e \sigma^2)^{|A|}$ 的展开

**Intuition**: 把 "target entropy per dim" 统一固定为对应 std=0.15 的 entropy。这个值与 action 物理含义直接关联——大多数 robot control 中 action 是 normalized joint position/torque offset，0.15 是一个合理的"探索幅度"。

Figure 10.a 的 ablation 显示 $\sigma_{\text{tgt}} \in [0.15, 0.2]$ 区间内性能基本一致，验证了这个参数的 robustness。

#### 2.3.2 Noise Repetition

时间相关 noise 的传统方案有 OU process [Mnih et al. 2015] 和 pink noise [Eberhard et al. 2023]，但它们都需要 per-environment 状态来维持相关性——在 1024 个并行环境上，这是显著的 memory 和 compute overhead。

FlashSAC 的 Noise Repetition 极其轻量：

```python
# Pseudocode
if step % k == 0:
    epsilon = np.random.randn(action_dim)  # 单次采样
    k = sample_from_zeta(s=2)  # 重复长度
action = policy_mean + sigma * epsilon
# epsilon 在接下来 k 步内保持不变
```

**Zeta distribution** $P(k) \propto k^{-s}$，$s=2$：
- 这是 power-law distribution，$P(k=1) \propto 1$, $P(k=2) \propto 1/4$, $P(k=10) \propto 1/100$
- 偏好短 repeat interval，但偶尔产生长 correlated sequence
- 与 [Dabney et al. 2020] 的 temporally-extended $\epsilon$-greedy 思路一致，但用 zeta 替代几何分布，更长尾
- Maximum repeat limit = 16，避免过长 lock-in

**为什么时间相关 noise 重要？** 在高维 action space 中，i.i.d. Gaussian noise 在单步内被 dynamics 平均掉——例如 29-DoF humanoid，一步随机 noise 让 joint torques 互相 cancel，机器人几乎不动。时间相关 noise 让 robot "持续尝试某个动作"，dynamics 才能积累 effect，产生 meaningful exploration trajectory。

---

## 3. Algorithm Pseudocode (我的整理)

```python
# === Init ===
actor = FlashSACNetwork(input_dim=obs_dim, output_dim=2*act_dim, 
                        n_blocks=2, hidden=128)  # 2.5M total params
critics = [FlashSACNetwork(...), FlashSACNetwork(...)]  # double Q
target_critics = copy(critics)
replay_buffer = ReplayBuffer(capacity=10_000_000)
optimizer = Adam(lr=3e-4, betas=(0.9, 0.999))
scheduler = CosineDecay(optimizer, start=3e-4, end=1.5e-4)
temperature = 0.01  # learnable alpha

# === Training loop ===
for step in range(total_steps):
    # 1. Collect data from 1024 parallel envs
    for env in parallel_envs:
        if step % k == 0:
            noise = randn(act_dim)
            k = sample_zeta(s=2, max_k=16)
        action = actor.mean(obs) + temperature * noise  # NOISE REPETITION
        next_obs, reward, done = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, done)
    
    # 2. Update every 1024 steps (UTD = 2/1024)
    if step % 1024 < 2:  # 2 updates per 1024 transitions
        # 2a. Sample batch
        s, a, r, s_next, done = replay_buffer.sample(2048)
        
        # 2b. CROSS-BATCH: concatenate for shared BN statistics
        s_combined = cat([s, s_next], dim=0)  # (4096, obs_dim)
        
        # 2c. Critic update
        with no_grad():
            a_next, logp_next = actor.sample(s_next)
            q_next = [target_critic(s_combined)[batch_size:] 
                      for target_critic in target_critics]
            q_next = min(q_next)  # clipped double Q
            target = r + gamma * (1-done) * (q_next - alpha * logp_next)
            target = categorical_project(target, atoms)  # distributional projection
        
        for critic in critics:
            q_pred = critic(s_combined)[:batch_size]  # same BN stats
            loss = cross_entropy(q_pred, target)
            optimizer.step()
            normalize_weights(critic)  # WEIGHT NORM
        
        # 2d. Actor update (delayed by 2)
        if step % 2 == 0:
            a_sample, logp = actor.sample(s)
            q = min([critic(s, a_sample) for critic in critics])
            actor_loss = (alpha * logp - q).mean()
            optimizer.step()
            normalize_weights(actor)
        
        # 2e. Target network soft update
        for target, source in zip(target_critics, critics):
            target.params = 0.01 * source.params + 0.99 * target.params
        
        # 2f. Temperature update
        with no_grad():
            _, logp = actor.sample(s)
        alpha_loss = -alpha * (logp + target_entropy).mean()
        alpha.optimizer.step()
        
        # 2g. Reward scaling update
        update_running_return_stats(r, gamma)
        scheduler.step()
```

---

## 4. Experiments 详解

### 4.1 Benchmark Suite 覆盖

总计 **60+ tasks across 10 simulators**，这是我见过最广泛的 RL benchmark suite 之一：

| 类别 | Simulator | Tasks | 典型任务 |
|-----|-----------|-------|---------|
| GPU-based, state | IsaacLab | 12 | G1 locomotion, Shadow Hand reorientation |
| GPU-based, state | MuJoCo Playground | 4 | G1/T1 joystick locomotion |
| GPU-based, state | ManiSkill3 | 6 | Franka manipulation |
| GPU-based, state | Genesis | 3 | Go2 walk, panda grasp |
| CPU-based, state | Gym MuJoCo | 5 | HalfCheetah, Humanoid-v4 |
| CPU-based, state | DMC Suite | 10 | Humanoid, Dog (high-dim) |
| CPU-based, state | HumanoidBench | 14 | H1 14 tasks |
| CPU-based, state | MyoSuite | 10 | Musculoskeletal manipulation |
| Vision-based | DMC-Visual | 8 | 状态从 pixels 推断 |
| Sim-to-real | Unitree G1 | 2 | Flat + rough terrain |

### 4.2 主要 baselines

- **PPO**: RSL-RL 实现，sim-to-real 社区的 de facto standard，训练 200M steps（FlashSAC 只训 50M，**3× compute advantage 给 PPO**）
- **FastTD3**: 同样 wall-clock optimized 的 off-policy 方法，但用 0.2M 小网络
- **XQC**: CrossQ 的扩展，sample-efficient off-policy with BN
- **SimbaV2**: hyperspherical normalization 系列最新
- **TD-MPC2**: model-based, world model + planning
- **MR.Q**: model-free with model-based representation learning
- **DrQ-v2**: vision-based 标杆

### 4.3 关键实验结果

#### 4.3.1 GPU-based state-based (Figure 3)

**Low-DoF tasks (15 tasks)**: FlashSAC 与 PPO 性能相当，因为低维任务中 PPO 的 sample efficiency 限制不显著，高吞吐 simulation 让 PPO 能收集足够数据。

**High-DoF tasks (10 tasks)**: FlashSAC 显著优于 PPO：
- Dexterous manipulation (Allegro, Shadow Hand): 更高 asymptotic return + 更快收敛
- Humanoid locomotion (G1, H1, T1): 性能差距最大

vs FastTD3：FlashSAC 在所有任务上更稳定，FastTD3 在 Go2Walk, FrankaPullCube 等任务上频繁失败或欠佳。两者都收敛时，FlashSAC asymptotic performance 更高，**humanoid locomotion 上的优势最大**——这正是 large model capacity 受益最大的场景。

#### 4.3.2 CPU-based state-based (Figure 5.2)

CPU 模拟器用单环境，sample efficiency 优先。这里 FlashSAC 改配置：batch=512, UTD=1, buffer=1M。

**PPO 在这里表现尤其差**——on-policy 无法重用经验，在有限 sample budget 下根本无法学习高维任务。

vs XQC/SimbaV2/TD-MPC2/MR.Q：FlashSAC 在大部分任务上 match 或 exceed 这些 sample-efficient baselines，**且不需要 per-task tuning**——只用统一配置。

#### 4.3.3 Vision-based (Figure 5)

DMC-Visual 8 tasks。Vision 设置的调整：
- Lightweight CNN encoder: 3 conv layers + linear bottleneck
- Frame stacking: 3 frames (84×84×9)
- 3-step returns
- Action repeat 2
- UTD 0.5, batch 256, buffer 1M

FlashSAC 在 asymptotic performance 和 wall-clock 上都 match 或超过 DrQ-v2 和 MR.Q。DrQ-v2 sample-efficient 但不稳定，Finger Turn Hard 失败。MR.Q 性能高但额外计算 dynamics model。

#### 4.3.4 Sim-to-Real (Figure 1.c, Figure 6)

这是最有说服力的实验：**Unitree G1 29-DoF, blind locomotion**（无 exteroceptive sensing）。

**Flat terrain**:
- FlashSAC: ~20 分钟达到稳定 real-world locomotion
- PPO: ~3 小时
- **9× speedup**

**Rough terrain (stairs)**:
- FlashSAC: ~4 小时
- PPO: ~20 小时
- **5× speedup**

更惊人的是 **generalization**：训练时 stair dimensions 是 23cm 高、32cm 宽、3m platform，真实测试 stair 是 15cm 高、60cm 宽、1.5m platform——**完全不同的 stair geometry**。FlashSAC 成功 climb，说明学到的 policy 是 robust locomotion skill，不是 sim-specific memorization。

**关键技术栈**:
- NVIDIA IsaacLab + Legged Gym framework
- 4096 parallel environments (实际训练) + 1024 (paper 算法配置)
- Domain randomization: joint friction, mass, motor strength, terrain params
- Terrain curriculum: 10 levels, 自动升级当 50% envs 成功
- **Asymmetric actor-critic** [Pinto et al. 2017]: critic 接触特权信息 (contact states, height map), actor 只见 proprioception
- **Context estimator network (CENet)** [DreamWaQ, Nahrendra et al. 2023]: 隐式 system identification，从 history 推断 base linear velocity + latent
- **Symmetry augmentation** [Mittal et al. 2024]: 利用 bipedal symmetry 提升 sample efficiency

Reference:
- DreamWaQ: https://arxiv.org/abs/2301.10602
- Asymmetric AC: https://arxiv.org/abs/1710.06542
- Symmetry policy: https://arxiv.org/abs/2407.10470 附近

### 4.4 Reward 设计 (Table 14)

Sim-to-real 部分的 reward 设计值得仔细看，因为它揭示了 on-policy vs off-policy 在 reward engineering 上的不同需求：

| Reward term | FlashSAC weight | PPO weight | 原因 |
|------------|-----------------|-----------|------|
| Track linear vel | 2.0 (σ=0.25) | 1.5 (σ=0.5) | FlashSAC 更紧凑 |
| Feet air time | 1.0 | 0.25 | FlashSAC 学 gait 更高效 |
| Body orientation | -52.0 | -2.0 | FlashSAC 需要更强约束防 destabilization |
| Action rate | -0.5 | -0.01 | FlashSAC 更激进 smoothing |
| Joint deviation (arm) | -1.0 | -0.2 | FlashSAC 更强约束 |
| Termination penalty | 无 | -200 | **关键差异** |
| Alive bonus | +1.0 | 无 | FlashSAC 不需要 termination 塑形 |

**关键 insight**: PPO 需要 termination penalty -200 来快速规避失败行为，因为它 sample inefficient 不能从 failure 中学习太多。FlashSAC 通过 replay buffer 大量积累 failure experience，自然学会避免，只需 small alive bonus 防止 premature termination。这是 off-policy 数据效率优势在 reward 设计上的直接体现。

---

## 5. Analysis 深度解析

### 5.1 Scaling Ablation (Figure 8, §6.2)

五个 univariate ablations：

**(a) Buffer size**: 1M → 10M → 50M
- 10M 最佳：稳定且快
- 50M：更稳定但慢（recent high-quality samples 被稀释）
- 1M：训练不稳定（rare experiences 被覆盖）

**(b-e) Batch / Width / Depth / UTD**:
- 增大 batch size (512→2048): 收敛加速，符合 scaling law
- 增大 width (128→512): 收敛加速 + asymptotic 提升
- 增大 depth (1→4 blocks): 收敛加速 + asymptotic 提升
- 降低 UTD (8/1024→0.5/1024): **加速**（少做无用 update）

这与标准 RL wisdom（多 update 更快收敛）矛盾，但符合 supervised scaling laws。**这是 FlashSAC 最反直觉但最重要的发现**——off-policy RL 的 scaling direction 与 supervised learning 一致，而非传统 RL 假设的"多 update = 快"。

### 5.2 Architectural Ablation (Figure 9, §6.3)

逐步累加组件对 condition number 的影响（measure 在 critic loss Hessian 上）：

| Architecture | Condition number 趋势 | 性能 |
|-------------|---------------------|------|
| MLP | 高且增长 | 差 |
| + Residual | 略降 | 改善 |
| + BatchNorm | 显著降 | 大改善 |
| + Post RMSNorm | 进一步降 | 进一步改善 |
| + Distributional + Reward Scaling | 进一步降 | 进一步改善 |
| + Weight Norm (= FlashSAC) | 最低 | 最佳 |

**Condition number 与 stability 的因果关系**: 高 condition number 意味着 Hessian 在某些方向曲率极大，gradient update 在这些方向步长过大（即使 learning rate 小），导致 weight 在这些方向剧烈震荡，bootstrapping target 在这些方向上预测不稳定，error 递归放大。Bound condition number 等于直接 bound 这种 amplification。

---

## 6. 与相关工作的大图景

### 6.1 与 Scaling Laws 的关系

[Kaplan et al. 2020] 的 scaling laws 在 supervised learning 中描述：$L(N, D) \propto (N^{-\alpha} + D^{-\beta})$，其中 $N$ 是 model size, $D$ 是 data size。FlashSAC 实际上验证了一个 RL 版本的 scaling law：

- 在固定 wall-clock budget 下，**大模型 + 大 batch + 少 update** 收敛更快
- 这与传统 RL "small model + many updates" 的 wisdom 相反
- 关键 enabler 是 **stability mechanisms**，否则大模型 under bootstrapping 会爆炸

### 6.2 与 CrossQ/XQC/Simba 系列的关系

Hojoon Lee + Daniel Palenicek + Jan Peters 这条线近期工作：
- **CrossQ** (ICLR 2024): 引入 BatchNorm 到 RL，发现 BN 的 cross-batch consistency 是关键
- **XQC** (ICLR 2026): 进一步用 distributional critic + reward scaling，condition number 分析
- **Simba** (2024): Inverted residual block 在 RL 中的应用
- **SimbaV2** (2025): Hyperspherical normalization
- **PLASTIC** (NeurIPS 2024): Input/label plasticity 保持
- **HARE/Tortoise** (2024): Plasticity via reinitialization

FlashSAC 是把这条线的所有 techniques 整合到一个 scalable sim-to-real pipeline 中，并加上大规模 parallel simulation + 大 buffer + 低 UTD 的工程优化。

References:
- CrossQ: https://arxiv.org/abs/2402.10315 (approximate)
- XQC: https://arxiv.org/abs/2502.15280
- Simba: https://arxiv.org/abs/2410.09754
- PLASTIC: NeurIPS 2024, Hojoon Lee

### 6.3 与 FastTD3/FastSAC 的关系

[Seo et al. 2025] 的 FastTD3 和 FastSAC 同样追求 wall-clock efficiency，通过 massively parallel simulation。但它们用 0.2M 小网络，因为大网络在 TD3/SAC 上不稳定。FlashSAC 的关键差异：**通过 stability mechanisms 解锁大模型**，从而同时获得 wall-clock speed 和 asymptotic performance。

### 6.4 与 Model-Based RL 的关系

TD-MPC2 [Hansen et al. 2024] 和 DreamerV3 [Hafner et al. 2023] 通过 world model + planning 提升 sample efficiency。FlashSAC 在 wall-clock 上胜过它们，因为 model-based 需要学 dynamics model + 反复 planning，per-step cost 高。但在 sample efficiency 上 model-based 仍有优势——FlashSAC vs TD-MPC2 在 CPU-based tasks 上 FlashSAC 更好，说明 stability 让 model-free 也能达到 sample efficiency。

References:
- TD-MPC2: https://arxiv.org/abs/2310.16828
- DreamerV3: https://arxiv.org/abs/2301.04104

---

## 7. 局限与未来方向

Paper §7 自己提到的：
- 当前 focus 在 state-based control，扩展到 tactile-based learning 是 future work
- Vision-based 还需要 lightweight CNN encoder + frame stacking，未尝试更重的 perception backbone
- 未探索 demonstration + RL 的混合训练（off-policy 天然支持）

我的额外观察：

1. **Reward 设计仍需 task-specific 调整**: Table 14 中 FlashSAC 和 PPO 用不同 reward weights，说明 reward engineering 仍是 manual labor。Auto-reward 或 RLHF-style reward learning 可能是下一步。

2. **Distributional critic 的 categorical support $[G_{\min}, G_{\max}]$ 固定**: 对超长 horizon 或 reward scale 变化大的任务可能受限。Adaptive support 或 quantile-based (QR-DQN) 可能更灵活。

3. **1024 parallel envs 的硬件门槛**: 复现需要 RTX 5090 + 1024 个 IsaacLab envs，对学术界小 lab 不友好。在更少并行度下如何 scale 是 open question。

4. **Asymmetric actor-critic 在 real deployment 的 gap**: critic 见特权信息训练，actor 不见，部署时只有 actor。这个 train-test gap 在更复杂任务上可能放大。

5. **Noise repetition 是 patch，不是根本解决方案**: 高维 exploration 的根本问题（curse of dimensionality）需要更结构化方法，如 goal-conditioned RL、curiosity-driven exploration、或 hierarchical RL。

6. **与 LLM/VLA 的结合**: 当前 FlashSAC 是 model-free RL。但 sim-to-real 机器人未来很可能与 VLA (Vision-Language-Action) model 结合，例如 OpenVLA [Kim et al. 2025]。FlashSAC 的 stability mechanisms 能否 transfer 到 VLA fine-tuning 中是关键 open question。Reference: https://openvla.github.io/

---

## 8. Intuition 总结

如果让我给 Andrej 你一个 30 秒的 intuition：

**FlashSAC 的核心 insight 是把 supervised learning 的 scaling laws 移植到 off-policy RL 中**。但这需要解决 bootstrapping instability——大模型 + bootstrapping = 灾难。通过显式 bound weight/feature/gradient norm（特别是 BatchNorm cross-batch consistency + Distributional critic + Weight normalization 三件套），把 critic loss landscape 的 condition number 压到极低，使得大模型能在 bootstrapping 下稳定训练。这解锁了"大模型 + 大 batch + 少 update"的 scaling regime，把训练时间从 hours 压到 minutes。最后用统一的 entropy target（基于 action std 而非 entropy 值）+ noise repetition 解决高维 exploration。整体上，这是 RL 中"stability enables scaling"哲学的最新最完整的实践。

**与你的 micrograd/nanoGPT 思路呼应的地方**: 你常说 supervised learning 的 elegance 在于 simplicity——SGD + 大模型 + 大 data 就 work。RL 之所以 messy，是因为 bootstrapping 引入的非平稳性破坏了这个 simplicity。FlashSAC 通过一系列 normalization 把这种非平稳性 bound 住，让 RL 在某种意义上回到 supervised-like 的 scaling regime——这是非常有吸引力的方向。

References:
- Paper: https://holiday-robot.github.io/FlashSAC
- SAC 原始 paper: https://arxiv.org/abs/1801.01290
- Scaling laws: https://arxiv.org/abs/2001.08361
- RSL-RL (PPO baseline): https://arxiv.org/abs/2509.10771
- IsaacLab: https://arxiv.org/abs/2511.04831
- MuJoCo Playground: https://arxiv.org/abs/2502.08844
- ManiSkill3: https://arxiv.org/abs/2410.00425
- Genesis: https://github.com/Genesis-Embodied-AI/Genesis
- HumanoidBench: https://arxiv.org/abs/2403.10506
- MyoSuite: https://arxiv.org/abs/2205.13600
- DrQ-v2: https://arxiv.org/abs/2107.09645
- DreamerV3: https://arxiv.org/abs/2301.04104
- TD-MPC2: https://arxiv.org/abs/2310.16828
- CrossQ (Bhatt et al.): ICLR 2024
- Simba: https://arxiv.org/abs/2410.09754
- PLASTIC: NeurIPS 2024
- Pink noise: https://arxiv.org/abs/2106.11934 附近
- Temporally-extended exploration (Dabney): https://arxiv.org/abs/2006.01782
- Reward centering: https://arxiv.org/abs/2405.09999
- Return-based scaling: https://arxiv.org/abs/2105.05347
- RMSNorm: https://arxiv.org/abs/1910.07467
- BatchNorm original: https://arxiv.org/abs/1502.03167
- MobileNet inverted bottleneck: https://arxiv.org/abs/1704.04861
- Deadly triad: https://arxiv.org/abs/1812.02648
- Hyperspherical normalization: https://arxiv.org/abs/2502.15280
- Lyle et al. normalization & effective LR: NeurIPS 2024
- DreamWaQ (sim-to-real): https://arxiv.org/abs/2301.10602
- Asymmetric actor-critic: https://arxiv.org/abs/1710.06542
- Symmetry in robot policy: Mittal et al. ICRA 2024
- OpenVLA: https://openvla.github.io/

希望这个深度解析对你 build intuition 有帮助！如果你想 drill-down 到某个具体方面（比如 condition number 的具体测量方法、distributional critic 的 projection 细节、或 sim-to-real 中 reward weights 的 ablation 逻辑），随时告诉我。
