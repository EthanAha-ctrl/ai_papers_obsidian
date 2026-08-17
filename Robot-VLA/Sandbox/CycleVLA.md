---
source_pdf: CycleVLA.pdf
paper_sha256: 1998a0ac1d5f1942eef844e29a933fa2885ceaa48f5e253701fbbe1c0315e390
processed_at: '2026-08-03T18:13:21-07:00'
target_folder: Robot-VLA/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# CycleVLA 用人话说一遍

## 一句话版

你训练一个 VLA 让机器人做事，它一旦走错就回不来了——杯子打翻了你让它怎么修？CycleVLA 让机器人在错误**还没完全发生之前**就察觉到（"这个 grasp 看着要滑"），自动回退到上一个关键步骤，重新试一遍，重试的时候还用 test-time scaling 从 8 个候选动作里挑最靠谱的那个。就这么个 cycle，subtask 边界预测失败 → backtrack → MBR retry → 成功或者超时。

---

## 为什么这事儿重要

你抓 OpenVLA、π0、GR00T N1 这类 VLA，本质上是 imitation learning——你给它看一堆 human demo，它学会 $\pi_\theta(a_t | o_t, g)$，给定观测和语言指令，输出 7 维 end-effector delta action。

问题在哪？**这些模型没有"我现在做到哪一步了"的概念**。它每一步都 condition 在当前 observation 上，pure reactive，没有 subtask-level 的 phase awareness。一旦走错一步，比如 grasp 的时候 gripper 偏了 1cm，后面 move 到 plate 时杯子就掉了，model 不知道自己已经搞砸了，继续执行下去一路崩盘。

这个现象在 long-horizon task 上特别明显。LIBERO-Long 这类 task 有 4-6 个 subtask 串起来，错误会跨 subtask **compound**（累积放大），paper Table I 里你看 OpenVLA 在 Long suite 只有 53.7% success rate，相比 Spatial 的 84.7% 差了一大截。long-horizon 是 VLA 的死穴。

human 是怎么做的？人**不会等错误完全发生再修**。你抓杯子感觉到 grip 要滑，立刻收紧手指——这是 **proactive** correction，在 error manifest 之前就 intervene。等杯子已经掉了，修正的机会就过了，物理上 irreversible。

CycleVLA 想给 VLA 装上这种 proactive self-correction 能力。核心 insight 来自 paper 引的 [16]-[20] 的观察：**robot task failures 大量集中在 subtask transitions**。比如 "grasp mug → move to plate" 这个 transition，如果 grasp 稍微 misaligned，move 一启动 mug 就掉了。而 transition 临近时的 visual cue（gripper 和 mug handle 的 alignment 偏差）已经能 strongly anticipate 这个 failure。所以如果你在 subtask 边界处 *check* 一下，就能在错误还没发生前 intervene。

---

## 三段式 cycle，逐个拆

### Part 1: 给 VLA 装上 "进度感"

原始 action 是 7 维：
$$a_t = [\Delta x_t, \Delta y_t, \Delta z_t, \Delta u_t, \Delta v_t, \Delta w_t, \gamma_t]^\top \in \mathbb{R}^7$$

- $(\Delta x_t, \Delta y_t, \Delta z_t) \in \mathbb{R}^3$: translation displacement
- $(\Delta u_t, \Delta v_t, \Delta w_t) \in \mathbb{R}^3$: rotation displacement（axis-angle）
- $\gamma_t \in \{0, 1\}$: gripper open/close

CycleVLA 把它扩成 9 维：
$$a_t = [\Delta x_t, \Delta y_t, \Delta z_t, \Delta u_t, \Delta v_t, \Delta w_t, \gamma_t, s_t, p_t]^\top \in \mathbb{R}^9$$

新增两个 scalar：
- $s_t \in \{0, 1\}$: **stop signal**，表示当前 subtask 是否终止
- $p_t \in [0, 1]$: **progress**，按 normalized timestep 离散化成 0.1 bins

paper 强调 $s_t$ 和 $p_t$ 必须 **分开**，不要合成一个。理由很 subtle：stop signal 必须 **precise**——它直接 trigger subtask 切换，错一个 timestep 可能 grab 错位置；progress signal 只需要 **coarse indication**，用来 trigger VLM check（"快到 subtask 末尾了，要不要 check 一下"）。精度要求不同，建模上也分开。

实现上没有任何 architectural change，就是把 action dimension 从 7 widen 到 9，跟其他维度一起 predict。训练时还 oversample 每个 subtask 的最后一个 action step 8 倍（last-action oversampling），强化 stop signal 的 detection。这个 trick 来自 NaVILA。

intuition：传统 VLA 训 $P(a_t | o_t, g)$ 是 *step-level* 的 condition，没有 *phase* awareness。$p_t$ 注入的 supervision 是：
$$p_t = \frac{t - t_{\text{subtask start}}}{t_{\text{subtask end}} - t_{\text{subtask start}}}$$

给 policy 一个 "你在 subtask 的哪个 phase" 的 sense。这跟 LLM 里 position encoding 给 transformer "你在 sequence 的哪个位置" 的 sense 有点像——本质都是 inject temporal structure。

但这里有个工程难题：现有 demo data 没有 subtask label，连 subtask 边界 timestamp 都没有。所以要自动构造。

### Subtask Decomposition Pipeline（这个细节 paper 里被低估）

整个过程用 GPT-4.1 (temp=0.2) 做：

**Step 1**: LLM 把 high-level task instruction 分解成 minimal atomic subtask sequence $(g_1, \dots, g_K)$，constrained action vocabulary 只允许 4 个 verb：`move`、`rotate`、`open`、`close`。例如 "pick up the red mug and place it on the plate" 会被拆成 4 个 subtask。

**Step 2**: 从 robot proprio 提取 per-step 的 movement primitive 和 gripper state。
- **Movement primitive** 用 sliding window of 4 timesteps 算 state difference，thresholding 每个 dimension。LIBERO 初始 threshold $[\tau_{trans}, \tau_{rot}, \tau_{grip}] = [0.02, 0.0075, 0.03]$。格式借鉴 ECoT：
  ```
  move [forward/backward] [left/right] [up/down]
  tilt [up/down]
  rotate [clockwise/counterclockwise]
  [close/open] gripper
  ```
- **Gripper state** 用 multi-threshold voting（三个 threshold [0.028, 0.03, 0.032] 投票）得到 final label $\in \{-1, 0, +1\}$（close/idle/open）

translation threshold 还做 per-trajectory optimization，grid search 最小化：
$$\text{score} = 1.0 \times N_{\text{overlaps}} + 2.5 \times N_{\text{stops}}$$
$N_{\text{overlaps}}$ 是 translation 和 gripper 动作重叠的次数（threshold 太低导致 noise），$N_{\text{stops}}$ 是 spurious stop label 数量（threshold 太高漏掉真实运动）。

**Step 3**: Alignment——这里最聪明。manipulation task 里 gripper state transition 是 reliable subtask boundary：close 段对应 grasp，open 段对应 release，idle 段对应 translation/rotation。
- 如果 LLM 提出的 subtask 数 = gripper segment 数，直接 pairwise 配对 timestamp
- 不匹配，把 movement primitive sequence downsample（trajectory > 100 steps 时 stride $\lceil T/100 \rceil$ uniform sample）后喂给 LLM，让它强制连续分配 timestamp，no gaps

Human evaluation（Table VII）显示平均 absolute error 5.7 steps，relative error 3.8%。LIBERO-Goal 比较难（8.0 steps error）因为 rotation 多、gripper boundary 不清晰。

---

### Part 2: VLM 当"事前纠错裁判"

inference 时，VLA 执行 action 的同时监控 $p_t$。当 $p_t \geq \tau_p = 0.9$（快到 subtask 末尾了），trigger VLM check。

VLM 输入：
- third-person camera view（global context：object identity、gripper pose、是不是在做对的 subtask）
- wrist camera view（fine-grained cue：gripper alignment、contact quality）
- 当前 subtask 描述
- subtask list

VLM 输出 two-way decision：
- **transit**: 继续到下一个 subtask
- **backtrack**: 回到最早能恢复 missing precondition 的 subtask

为啥要两个 view？third-person 看全局（"对不对的 object 在做"），wrist 看细节（"gripper 抓得稳不稳"）。VLM 用 Chain-of-Thought reasoning 融合两个 view 的 evidence，输出结构化的 reason + decision。paper 用 GPT-5.2 (temp=1.0) 做这个。

backtrack 的目标 subtask 选择有讲究——**回到最早能恢复 missing precondition 的 subtask**。比如 grasp 的 mug 半路掉了，不能只 backtrack 到 "move to plate"，要回到 "grasp mug" 才行。这本质上是 classical planning 里 STRIPS precondition checking 的 perception-grounded 版本。

physical backtrack 怎么做？通过 reverse-execute recorded delta actions 把机器人 state restore 到 target subtask 起始位置。这个 trick 来自 [Bellman-Guided Retrials (Du et al. 2024)](https://arxiv.org/abs/2406.15917)。

#### CONFIRM mechanism（Appendix D 的细节）

$s_t, p_t$ 是 noisy 的预测，直接 trigger 会有很多 false positive。paper 用 CONFIRM(·) 过滤，跟踪三个量：
- `first_seen`: 是否见过 high signal
- $c_{\text{consec}}$: 连续 high signal 计数
- $c_{\text{gap}}$: 自上次 high signal 以来 low signal 计数

trigger 条件（Eq.(8)）：
$$c_{\text{consec}} \geq 2 \quad \text{or} \quad (\text{first\_seen} \wedge c_{\text{gap}} \geq 2)$$

要么连续两步 high，要么 high 之后过两步又 high。这 filter 掉 isolated spurious prediction 但仍 responsive。

---

### Part 3: MBR decoding 让 retry 更靠谱

backtrack 后机器人回到起点，如果直接再执行一次 policy $\pi_\theta(\cdot | o_t, g_k)$，可能又走同样的错路（policy 是 deterministic 的，给定相同 input 输出相同 action）。怎么办？**用 stochastic sampling + consensus selection**。

CycleVLA 用的是 diffusion-based action expert（参考 $\pi_0$ 和 RDT-1B），stochasticity 来自 diffusion noise sampling。换 random seed 就能 sample 出不同的 action chunk。paper sample N=8 个 hypotheses $\mathcal{A} = \{a^{(1)}, \dots, a^{(N)}\}$。

然后 **Minimum Bayes Risk decoding** 选 consensus：

#### 原始 MBR 数学

MBR 来自统计机器翻译 ([Kumar & Byrne 2004](https://aclanthology.org/N04-1020/))。核心 idea：给定 distribution $P(a | o, g)$，选一个 action chunk $a^*$ 最小化 expected loss：

$$a^* = \arg\min_a \mathbb{E}_{a' \sim \pi_\theta}[d(a, a')]$$

变量解释：
- $a$: 候选 action chunk，长度 H（paper 里 H=8）
- $a' \sim \pi_\theta$: 从 policy distribution 采样的 "真值" reference action chunk
- $d(\cdot, \cdot)$: distance metric，paper 用 $L_2$
- $\mathbb{E}$: expectation over policy distribution

intuition：选一个 chunk，它到 policy 自己采样的所有 chunk 的 *平均距离最小*——也就是说它处于 distribution 的 **密度最高区域**。

#### Monte Carlo 近似

实际无法积分 policy distribution，用 sample 近似（Eq.(2)）：
$$\mathcal{L}(a_{t:t+H-1}) = \frac{1}{N}\sum_{n=1}^N d(a_{t:t+H-1}, a_{t:t+H-1}^{(n)})$$

$\mathcal{L}$ 是 estimated Bayesian risk——候选 chunk 到所有 sampled chunk 的平均距离。

最终选择（Eq.(3)）：
$$a^{\text{MBR}} = \arg\min_{a \in \mathcal{A}} \mathcal{L}(a)$$

候选集 $\mathcal{A}$ 本身就是 sampled set，所以是 N×N 的 pairwise distance matrix，选 row-mean 最小的那个。

#### Feature space 距离

直接在 9 维 raw action space 算距离有问题：translation 和 rotation scale 不同，gripper/stop/progress 是 categorical。所以 paper 把 action chunk 投影到 trajectory feature space：

$$\phi(a_{t:t+H-1}^{(n)}) \in \mathbb{R}^{6H}$$

这个 feature vector 包含 H 个 timestep 的 position $(x, y, z)$ 和 orientation $(u, v, w)$——即 6 维 × H 步 = $6H$ 维。具体是 cumulative sum translational and rotational deltas 得到 absolute trajectory。

Eq.(4)：
$$a^{\text{MBR}} = \arg\min_{a^{(i)} \in \mathcal{A}} \frac{1}{N}\sum_{j=1}^N d(\phi(a^{(i)}), \phi(a^{(j)}))$$

#### Density-based variant（Appendix C）

实际用的 **不是 vanilla MBR**，是 density-based 变体，更 robust：
1. 对每个 hypothesis $a^{(i)}$，计算 r-NN radius（到第 r 近邻的距离）
2. 自适应选 r：$r = \max(2, \min(4, \lfloor\sqrt{N}\rfloor))$，对 N=8 得 $r=2$
3. 找 r-NN radius 最小的 hypothesis 作为 "pocket center"（最高密度区）
4. 在 pocket 内找 medoid——到 pocket 内其他成员平均距离最小的那个

vanilla MBR 容易被 outlier 拉偏，density-based 先定位稠密区再选代表。这个 trick 来自 [Heineman et al. 2024](https://aclanthology.org/2024.emnlp-main.560/) 的 multi-prompt MBR。

#### Intuition：为什么 MBR 对 VLA 有效

这是 paper 最漂亮的一个 insight。imitation learning 训出来的 policy $\pi_\theta$ 本质上在拟合 demo data 的 conditional distribution $P(a | o, g)$。如果 demos 都成功了，那么 **成功的 action 在 policy output space 里稠密 cluster**，**失败的 action（policy 自己采样的 noise）是稀疏 outlier**。

这意味着：
- Sample 多个 hypothesis → 成功的会互相靠近，失败的散开
- MBR 选 cluster center → 倾向成功 action
- 失败 action 距离 cluster 远 → Bayesian risk 高 → 不会被选

这跟 LLM 里 MBR 有效机制一样：高质量 generation 互相 consistent（coherent、grammatical、on-topic），低质量 generation 互相 inconsistent。consensus selection 自动 filter 掉 outlier。

这也是为什么 paper 发现 **under-trained VLA 从 MBR 获益更大**——200K checkpoint 的 output distribution 更分散，consensus selection 能更有效地 pull back 到 high-density region。Table III 里 200K 的 MBR gain 是 +6.8，500K 是 +5.3。

---

## 实验数据的核心 reading

### Table I: LIBERO 主结果

最 striking 的是 **LIBERO-Long**：

| Method | Long suite SR |
|---|---|
| OpenVLA | 53.7% |
| TraceVLA | 54.1% |
| SpatialVLA | 55.5% |
| FPC-VLA | 82.2% |
| GR00T N1 | 90.6% |
| **CycleVLA** | **93.6%** |

Long suite 是 long-horizon task，错误跨 subtask 累积，传统 VLA 一旦走错就一路错。CycleVLA 在每个 subtask 边界都有机会 reset，所以 long-horizon benefit 最显著。这跟 paper core motivation 完美吻合。

平均 SR 95.3% 是 SOTA。

### Table II: Under-trained VLAs 的 recovery

| Checkpoint | w/o FC | w/ FC | Gain |
|---|---|---|---|
| 200K | 73.2 | 80.0 | +6.8 |
| 350K | 83.2 | 89.2 | +6.0 |
| 500K | 89.3 | 95.3 | +6.0 |

**关键观察**：200K+FC ≈ 350K w/o FC（80.0 vs 83.2），350K+FC ≈ 500K w/o FC（89.2 vs 89.3）。CycleVLA 的 correction 机制相当于 *免费* 给你 boost 了 ~150K training steps 的 capacity。对于计算受限的小 lab，这个实用价值很大。

### Table III: N 的影响

N 从 4 → 64：
- N=4 → 8: 显著提升
- N=8 → 16: 仍有提升
- N > 16: plateau

random selection 几乎等于 base VLA SR，证明 MBR 的 gain 来自 *selection* 而非 *sampling* 本身。N=8 是 sweet spot。

### Table IV: Distance metric

| Metric | CycleVLA-350K |
|---|---|
| Random | 80.3 |
| MBR-$L_1$ | 89.7 |
| **MBR-$L_2$** | **90.2** |
| MBR-$L_\infty$ | 88.2 |
| MBR-cos | 87.5 |
| MBR-r | 86.9 |

$L_2$ 最好，cosine 和 correlation 最差。paper 的 hypothesis：translation 分量 dense（每步都有），rotation 分量 sparse。magnitude-based metric（$L_1, L_2$）比 direction-based metric（cos, r）更能 capture 真实 trajectory 差异。

### Table V: Runtime

A10 GPU 上总 215.3s：
- Action Rollout: 147.6s (68.6%) — 最大头
- Action Sampling: 47.9s (22.2%)
- VLM API: 12.9s (6.0%)
- Backtrack: 6.9s (3.2%)
- **MBR computation: 0.003s (<0.1%)** — 几乎免费！

MBR 的 pairwise distance 在 $\mathbb{R}^{6H}$ 上算 N×N=64 个 pair 是 trivial 的。**全部 cost 来自 forward pass 采样**。Test-time scaling 整体增加约 30% runtime。

### Table VI: Ablation

- **w/o MBR** (random selection 代替): SR 92.5 (-2.8)，time 302.4s（更长！因为 random selection 失败率高，trigger 更多 backtrack retry）
- **alt. VLM** (LLaMA-3.2-11B 替代 GPT-5.2): SR 92.8 (-2.5)，time 172.6s（本地 VLM 快但更倾向 transit，少 backtrack）
- **always-on MBR** (upper bound): SR 96.9 (+1.6)，time 464.3s（VLM 选择性 trigger MBR 避免不必要 cost）
- **pred. failure cutoff** (lower bound): SR 79.7（VLM sycophancy——让它判断失败它就倾向说失败，提前 terminate）

---

## 我的整体 reading

### 漂亮的地方

1. **Progress signal $p_t$ 是整个 pipeline 的 trigger**——它把 VLA 从被动 actor 变成 active monitor。一旦 VLA 知道 "我在哪、走多远"，外部 VLM 就能在合适的时机介入。这个 trigger 机制比单纯的 "每隔 N 步 check 一次" 更 smart，因为它 *只在 risk 最高的时刻*（subtask 边界）trigger。

2. **MBR decoding 的 elegance 在于 zero-shot**——不需要训 verifier，不需要 reward model，只要 policy 能 sample 就行。这跟 RoboMonkey（需要 trained VLM verifier）、Rover（需要 learned reward model）形成对比。

3. **整体 paradigm shift**：robot policy 不应该一次性 feedforward，而应该有 internal cycle of prediction-monitoring-correction。这跟 LLM 里 chain-of-thought 让 model 多步 reasoning 类似——VLA 也需要 inference-time 的 multi-step processing，只是形式上是 *subtask-level cycle* 而非 token-level reasoning。

### 几个我想深挖的 gap

1. **Reversible state assumption**：backtracking 通过 reverse-execute delta actions 恢复 state，但 irreversible 环境不适用。你打翻水 reverse 不回去。这是 fundamental limitation——但也是设计选择，因为 proactive correction 的本质就是 *在错误 manifest 前 intervene*，所以还没有 irreversible damage。

2. **VLM sycophancy**：Table VI 的 lower bound 实验显示，VLM 被 prompt "判断是否失败" 时会倾向 confirm 失败。可能需要 calibration 或 contrastive prompting（同时问 "is this succeeding" 和 "is this failing"）。

3. **MBR 的 failure mode**：如果所有 N 个 hypotheses 都失败（policy 整体走偏），MBR 还是会选一个失败的。需要 detect "all hypotheses are bad" 的情况，trigger 更 high-level replanning。

4. **Subtask granularity sensitivity**：太细 subtask 多 overhead 高，太粗 transition 少 loss correction 机会。这个 trade-off paper 没 study。

5. **Real robot 缺失**：只在 LIBERO simulation 上做，real hardware 实验待补。这是 major gap。

6. **External VLM dependency**：现在靠 GPT-5.2 做 failure prediction。未来可以 end-to-end 训 VLA 自己做 failure reasoning，类似 ECoT 把 reasoning 塞进 action prediction 里。

### 跟 LLM test-time scaling 的对应

最近 LLM 圈 test-time scaling 主要靠 reasoning chain（[Snell et al. 2025](https://arxiv.org/abs/2408.03314), [DeepSeek-R1](https://arxiv.org/abs/2501.12948), [o3/o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)）。VLA 不一样：output 是 continuous action，没有 "多想几步" 的明确对应物。

CycleVLA 的 MBR decoding 是 VLA test-time scaling 的一种 form：sample N 个 hypotheses 然后 select。这跟 LLM 里 **best-of-N** 或 **self-consistency** 是同构的，只是 distance metric 换成 trajectory feature space 上的 $L_2$。

未来方向可能是 *learned* test-time scaling for VLA——train 一个 critic 或者 process reward model 专门 score action chunk quality。但 paper 这里 zero-shot MBR 已经是个 strong baseline。

---

## Reference Links

主参考：
- [CycleVLA paper](https://arxiv.org/abs/2511.00062)
- [OpenVLA project](https://openvla.github.io/) / [paper](https://arxiv.org/abs/2406.09246)
- [OpenVLA finetuning (Kim et al. 2025)](https://arxiv.org/abs/2505.04678)
- [LIBERO benchmark](https://lifelong-robot-learning.github.io/LIBERO/) / [paper](https://arxiv.org/abs/2306.03310)
- [π0 (Black et al. 2024)](https://arxiv.org/abs/2410.24164)
- [π0.5](https://arxiv.org/abs/2504.16054)
- [GR00T N1](https://arxiv.org/abs/2503.14734)
- [RDT-1B](https://arxiv.org/abs/2410.07872)

方法相关：
- [Bellman-Guided Retrials (Du et al. 2024) - "To err is robotic"](https://arxiv.org/abs/2406.15917)
- [PAINT (Xie et al. 2022)](https://arxiv.org/abs/2210.11215)
- [ECoT (Zawalski et al. 2024)](https://arxiv.org/abs/2407.08693)
- [RoboMonkey](https://arxiv.org/abs/2510.10975)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [AHA (Duan et al. 2025)](https://arxiv.org/abs/2410.07384)
- [REFLECT (Liu et al. 2023)](https://arxiv.org/abs/2310.14525)
- [SeqVLA - progress-aware VLA](https://arxiv.org/abs/2509.14138)

MBR decoding 理论：
- [Kumar & Byrne 2004 - 原始 MBR 论文](https://aclanthology.org/N04-1020/)
- [Eikema & Aziz 2022 - sampling-based MBR](https://aclanthology.org/2022.emnlp-main.522/)
- [Heineman et al. 2024 - multi-prompt MBR](https://aclanthology.org/2024.emnlp-main.560/)
- [Snell et al. 2025 - test-time scaling](https://arxiv.org/abs/2408.03314)

LLM/VLM 基础：
- [Chain-of-Thought (Wei et al. 2022)](https://arxiv.org/abs/2201.11903)
- [GPT-4.1](https://openai.com/index/gpt-4-1/)
- [LLaMA 3 herd](https://arxiv.org/abs/2407.21783)

---

## TL;DR

CycleVLA 把 VLA 的 inference 从 "一次性 feedforward" 变成 "subtask-level cycle"：predict → monitor progress → 在 subtask 边界用 VLM check → 要不行就 backtrack 到能恢复 precondition 的起点 → 用 MBR 从 8 个 stochastic hypotheses 里选 consensus action chunk 重试。LIBERO-Long 从 53.7% 拉到 93.6%，under-trained VLA 也能免费 boost ~150K training steps 的 capacity。MBR 几乎零 computational overhead（0.003s），全部 cost 来自 forward pass sampling。整个 system 是 training-free 的 test-time scaling，不需要 reward model 也不需要 verifier，只需要 policy 能 sample。

---

# CycleVLA 深度解读：给 VLA 装上"事前纠错"的神经系统

## 1. 这篇 paper 真正想解决什么问题

让我从最底层 motivation 说起。当你训练一个 VLA（Vision-Language-Action model）做 manipulation，比如 OpenVLA、π0、GR00T N1 这类 generalist robot foundation models，它们本质上是一个 imitation learning policy $\pi_\theta(a_t | o_t, g)$——给定观测和语言目标，输出 7 维 end-effector delta action。问题在于：**这些模型一旦走错路就回不来了**。现有的 failure detection 工作（PAINT、AHA、REFLECT、SAFE）几乎全部是 *post hoc* 的——错误已经发生了，杯子已经打碎了，再去做 residual correction 或者 ask for help。这跟人类完全不一样：人类是 *proactive* 的，手感觉到杯子要滑了就提前收紧 grip，车要偏了就提前转方向盘。等错误完全 manifest 了再修正，往往物理上已经 irreversible。

CycleVLA 的核心 insight 非常 elegant：**robot task failures 大量集中在 subtask transitions**（参考文献 [16]-[20] 的观察），而 subtask 接近完成时的 visual cue（比如 peg 要插进去之前的对齐偏差）已经能 strongly anticipate failure。所以如果让 VLA 学会：(a) 感知自己当前在哪个 subtask、完成到什么程度；(b) 在 subtask 边界处用 VLM 判断要不要 back off；(c) back off 后用 test-time scaling 重试——就能在 *同一个 episode 内* 完成自纠正。

这跟最近的几条 research line 形成对比：
- **Bellman-Guided Retrials** ([Du et al. 2024](https://arxiv.org/abs/2406.15917))：也做 backtracking，但不针对 generalist policy，靠 value function 评估
- **PAINT** ([Xie et al. 2022](https://arxiv.org/abs/2210.11215))：proactive intervention，但需要 human in the loop
- **RoboMonkey** ([Kwok et al. 2025](https://arxiv.org/abs/2510.10975))：VLA 的 test-time scaling，但用 trained VLM verifier 选 action，不是 training-free consensus
- **V-JEPA 2** ([Assran et al. 2025](https://arxiv.org/abs/2506.09985))：用 world model 预测未来状态做 planning，架构 overhead 很大

CycleVLA 想用最轻量的方式——不换 VLA backbone、不引入 world model、不训练 reward model——把 proactive correction 塞进去。

---

## 2. 整体架构：三段式 cycle

整个系统分成三个 component，对应 Fig. 3 的 (a)(b)(c)：

### Component 1: Progress-aware VLA via extended action expert

原始 OpenVLA 的 action 是 7 维：
$$a_t = [\Delta x_t, \Delta y_t, \Delta z_t, \Delta u_t, \Delta v_t, \Delta w_t, \gamma_t]^\top \in \mathbb{R}^7$$

其中 $(\Delta x, \Delta y, \Delta z) \in \mathbb{R}^3$ 是 translation displacement，$(\Delta u, \Delta v, \Delta w) \in \mathbb{R}^3$ 是 rotation displacement（axis-angle 形式），$\gamma_t \in \{0, 1\}$ 是 gripper open/close binary signal。

CycleVLA 把它扩展到 9 维：
$$a_t = [\Delta x_t, \Delta y_t, \Delta z_t, \Delta u_t, \Delta v_t, \Delta w_t, \gamma_t, s_t, p_t]^\top \in \mathbb{R}^9$$

这里：
- $s_t \in \{0, 1\}$: **stop signal**，表示当前 subtask 是否终止
- $p_t \in [0, 1]$: **subtask progress**，按 normalized timestep 离散化成 0.1 bins

为什么把 $s_t$ 和 $p_t$ 分开？这是个 subtle 但重要的设计——stop signal 必须 *precise*，因为它直接 trigger subtask transition，错一个 timestep 可能就 grab 错位置；progress signal 只需要 coarse indication of "快到 subtask 末尾了"，用来 trigger VLM check。两者精度要求不同，所以单独建模。

设计上没有任何 architectural change——就是 widen 了 action dimension，把 $s_t, p_t$ 当成 scalar 跟其他 7 维一起 predict。这跟 NaVILA ([Zhang et al. 2024](https://arxiv.org/abs/2402.15852)) 一样的思路：把 stop 信号当成 action 的一部分预测。

### Component 2: VLM as failure predictor + planner

在 inference 时，当 VLA 预测的 progress $p_t \geq \tau_p = 0.9$，触发 VLM check。VLM 输入是 third-person view + wrist view + 当前 subtask + subtask list，输出 two-way decision：
- **transit**: 继续到下一个 subtask
- **backtrack**: 回到最早能恢复 missing precondition 的 subtask

为啥要两个 view？Third-person view 提供 global context（object identity、gripper pose、是不是在做对的 subtask），wrist view 提供 fine-grained cue（gripper alignment、contact quality）。VLM 用 Chain-of-Thought ([Wei et al. 2022](https://arxiv.org/abs/2201.11903)) 融合两个 view 的 evidence，输出结构化的 reasoning + decision。

paper 用 GPT-5.2 temperature=1.0 做这个 role。Ablation 里换 LLaMA-3.2-11B 发现它会过度倾向 transit（不太 backtrack），导致 success rate 略降但 runtime 也降。

### Component 3: MBR decoding for retry

backtrack 后，机器人 state 被 restore 到 target subtask 的起始位置（通过 reverse-execute recorded delta actions，这是 [Bellman-Guided Retrials](https://arxiv.org/abs/2406.15917) 的思路）。然后 VLA 不是简单 retry 一次，而是 sample N=8 个 stochastic hypotheses，用 Minimum Bayes Risk decoding 选 consensus action chunk。

这里有个关键 insight：VLA 用 diffusion-based action expert（参考 $\pi_0$ [Black et al. 2024](https://arxiv.org/abs/2410.24164) 和 RDT-1B [Liu et al. 2025](https://arxiv.org/abs/2410.07872) 的设计），stochasticity 来自 diffusion noise sampling。imitation learning 训出来的 policy 的成功行为会在 action space 的高密度区域 cluster，所以 consensus selection（找密度最高的那个 hypothesis）倾向于挑到 successful trajectory。

---

## 3. Subtask Decomposition Pipeline 的细节

这部分是 paper 里被低估的一块工程，我觉得很巧妙。它要解决的问题是：现有 demonstration 没有 subtask label，需要自动构造。

### Step 1: LLM-based subtask proposal

用 GPT-4.1 (temperature=0.2) 把 high-level task instruction $g$ 分解成 minimal atomic subtask sequence $(g_1, \dots, g_K)$，constrained action vocabulary 只允许 4 个 verb：`move`、`rotate`、`open`、`close`。比如 "pick up the red mug and place it on the plate" 会被分解成：
1. "Move the gripper to the red mug"
2. "Close the gripper to grasp the red mug"
3. "Move the gripper above the plate"
4. "Open the gripper to release the red mug"

### Step 2: Movement primitive + gripper state extraction

从 robot proprio（关节状态）提取 per-step 的：
- **Gripper state**: open / close / idle，通过 multi-threshold voting 用三个 threshold [0.028, 0.03, 0.032] 投票得到 final label $\in \{-1, 0, +1\}$
- **Movement primitive**: 借鉴 ECoT ([Zawalski et al. 2024](https://arxiv.org/abs/2407.08693)) 的格式：
  ```
  move [forward/backward] [left/right] [up/down]
  tilt [up/down]
  rotate [clockwise/counterclockwise]
  [close/open] gripper
  ```
  通过 sliding window of 4 timesteps 计算 state difference，对每个 dimension 做 thresholding。LIBERO 的初始 threshold 是 $[\tau_{trans}, \tau_{rot}, \tau_{grip}] = [0.02, 0.0075, 0.03]$。

对 translation threshold 还做了 per-trajectory optimization：
$$\text{score} = 1.0 \times N_{\text{overlaps}} + 2.5 \times N_{\text{stops}}$$
通过 grid search over $[\tau_{trans}^{init} - 0.01, \tau_{trans}^{init} + 0.01]$ with 50 steps 最小化。 intuition 是 translation 和 gripper 动作重叠说明 threshold 太低导致 noise，spurious stop 说明 threshold 太高漏掉了真实运动。

### Step 3: Alignment

关键 idea：manipulation task 里 gripper state transition 是 reliable subtask boundary——close 段对应 grasp，open 段对应 release，idle 段对应 translation/rotation。

- 如果 LLM 提出的 subtask 数 = gripper segment 数，直接 pairwise 配对 timestamp
- 否则，把 movement primitive sequence downsample（trajectory > 100 steps 时用 stride $\lceil T/100 \rceil$ uniform sample）后喂给 LLM，让它强制连续分配 timestamp，no gaps

Human evaluation（Table VII）显示平均 absolute error 5.7 steps，relative error 3.8%。LIBERO-Goal 比较难（8.0 steps error）因为 rotation 多、gripper boundary 不清晰。

---

## 4. MBR Decoding 的数学，仔细讲一下

这块是 paper 的理论核心，我从最基础的 Bayes decision theory 讲起。

### 4.1 原始 MBR formulation

Minimum Bayes Risk decoding 来自统计机器翻译 ([Kumar & Byrne 2004](https://aclanthology.org/N04-1020/))。核心 idea：给定一个 distribution $P(a | o, g)$（这里是 VLA policy $\pi_\theta$），我们想选一个 action chunk $a^*$ 最小化 expected loss under some loss function $\ell(\cdot, \cdot)$：

$$a^* = \arg\min_a \mathbb{E}_{a' \sim \pi_\theta}[\ell(a, a')]$$

这里 $a' \sim \pi_\theta$ 是从 reference distribution（policy 自己）采样的"真值"，$\ell$ 衡量 $a$ 偏离真值的风险。

如果 $\ell(a, a') = d(a, a')$ 是某个 distance metric，那就是 paper 的 Eq.(1)：
$$a_{t:t+H-1}^{\text{MBR}} = \arg\min_{a_{t:t+H-1} \in \mathcal{A}} \mathbb{E}_{a' \sim \pi_\theta}[d(a, a')]$$

变量解释：
- $t$: 当前 timestep
- $H$: action chunk size（paper 里 H=8）
- $a_{t:t+H-1}$: 一个候选 action chunk，长度 H
- $\mathcal{A} = \{a^{(1)}, \dots, a^{(N)}\}$: 采样得到的 N 个 hypotheses
- $a' \sim \pi_\theta$: 从 policy distribution 采样的"真值" action chunk

### 4.2 Monte Carlo approximation

实际计算时无法积分 policy distribution，所以用 sample 平均近似 expectation，得到 Eq.(2)：

$$\mathcal{L}(a_{t:t+H-1}) = \frac{1}{N}\sum_{n=1}^N d(a_{t:t+H-1}, a_{t:t+H-1}^{(n)})$$

这里 $\mathcal{L}(\cdot)$ 是 estimated Bayesian risk——某个候选 chunk 到所有 sampled chunk 的平均距离。距离越小，这个 chunk 越"central"，越能代表 cluster center。

最终 MBR 选择（Eq.(3)）：
$$a^{\text{MBR}} = \arg\min_{a \in \mathcal{A}} \mathcal{L}(a)$$

注意：候选集 $\mathcal{A}$ 本身就是 sampled set，所以是 N-by-N 的 pairwise distance matrix，选 row-mean 最小的那个。这是 sampling-based MBR 的标准做法 ([Eikema & Aziz 2022](https://aclanthology.org/2022.emnlp-main.522/))。

### 4.3 Feature representation 和 distance

直接在 9 维 raw action space 算距离有个问题：translation 和 rotation 的 scale 不同，gripper $\gamma$, stop $s$, progress $p$ 是 categorical 性质的。所以 paper 把 action chunk 投影到 trajectory feature space：

$$\phi(a_{t:t+H-1}^{(n)}) \in \mathbb{R}^{6H}$$

这个 feature vector 包含 H 个 timestep 的 position $(x, y, z)$ 和 orientation $(u, v, w)$——即 6 维 × H 步 = $6H$ 维。具体是 cumulative sum translational and rotational deltas 得到 absolute trajectory（而非 delta），然后展开成 vector。

Eq.(4) 的完整形式：
$$a^{\text{MBR}} = \arg\min_{a^{(i)} \in \mathcal{A}} \frac{1}{N}\sum_{j=1}^N d(\phi(a^{(i)}), \phi(a^{(j)}))$$

这就是在 feature space 上算 N×N pairwise distance，对每个 $i$ 求 row mean，选最小的 $i$ 对应的 hypothesis。

### 4.4 Density-based variant（Appendix C 的细节）

实际上 paper 用的是 *density-based* MBR 变体，不是 vanilla MBR。具体步骤：

**Step 1**: 对每个 hypothesis $a^{(i)}$，计算 r-NN radius = 到第 r 近邻的距离（越小代表周围越稠密）

**Step 2**: 自适应选 r：
$$r = \max(2, \min(4, \lfloor\sqrt{N}\rfloor))$$
对 N=8，$\lfloor\sqrt{8}\rfloor = 2$，所以 $r = \max(2, \min(4, 2)) = 2$。

**Step 3**: 找 r-NN radius 最小的 hypothesis 作为 "pocket center"（最高密度区）

**Step 4**: 在这个 pocket 内找 medoid——到 pocket 内其他成员平均距离最小的那个

这比 vanilla MBR 更 robust，因为 vanilla MBR 容易被 outlier 拉偏，而 density-based 先定位到稠密区再选代表。这个 trick 来自 [Heineman et al. 2024](https://aclanthology.org/2024.emnlp-main.560/) 的 multi-prompt MBR 工作。

### 4.5 Intuition：为什么 MBR 对 VLA 有效

这是我觉得 paper 里最漂亮的一个 insight。让我从 imitation learning 的 geometry 角度讲：

 imitation learning 训出来的 policy $\pi_\theta$  essentially 是在 demonstration data 上拟合 conditional distribution $P(a | o, g)$。如果 demonstrations 都成功了，那么 *成功的 action* 在 policy output space 里是稠密的 cluster，而 *失败的 action*（policy 自己采样的 noise）是稀疏的 outlier。

这意味着：
- Sample 多个 hypothesis → 成功的会互相靠近（在 cluster 内），失败的会散开
- MBR 选 cluster center → 倾向于选成功 action
- 失败 action 距离 cluster 远 → Bayesian risk 高 → 不会被选

这跟 LLM 里 MBR 有效的机制一样：高质量 generation 之间互相 consistent（coherent, grammatical, on-topic），低质量 generation 之间 inconsistent。consensus selection 自动 filter 掉 outlier。这也是为什么 paper 发现 *under-trained* VLA（200K, 350K checkpoints）从 MBR 获益更大——它们的 output distribution 更分散，consensus selection 能更有效地 pull back 到 high-density region。

### 4.6 Distance metric 的选择

Table IV 测了 $L_1, L_2, L_\infty$, cosine, correlation，结论是 $L_2$ 最好，$L_1$ 次之，cosine 和 correlation 最差。paper 的 hypothesis：translation 分量沿 trajectory dense（每步都有），rotation 分量 sparse（很多 timestep 没旋转）。所以 magnitude-based metric ($L_1, L_2$) 比 direction-based metric (cos, r) 更能 capture 真实 trajectory 差异。这个观察对设计 future VLA 的 MBR variant 很有用。

---

## 5. Algorithm 1 的完整推理流程

我把 Alg. 1 的两个 phase 讲清楚，因为它体现了很多 engineering 决策：

**MONITOR phase**: 持续执行 policy，monitor $p_t$，当 $p_t \geq \tau_p = 0.9$ 触发 VLM check

**COMPLETE phase**: VLM 说 transit 后，继续执行直到 stop signal $s_t = 1$ 触发，然后切到下一个 subtask

每次都 maintain 一个 queue $\mathcal{Q}$：sample 一个 chunk 后 push 进去，每步 pop 一个 action 执行。这样能 chunk-level predict（H=8），step-level execute，兼顾 throughput 和 responsiveness。

CONFIRM mechanism（Appendix D）很重要：因为 $s_t, p_t$ 是 noisy 的，直接 trigger 会很多 false positive。CONFIRM(·) 跟踪三个量：
- `first_seen`: 是否见过 high signal
- $c_{\text{consec}}$: 连续 high signal 计数
- $c_{\text{gap}}$: 自上次 high signal 以来 low signal 计数

trigger 条件是 Eq.(8)：
$$c_{\text{consec}} \geq 2 \quad \text{or} \quad (\text{first\_seen} \wedge c_{\text{gap}} \geq 2)$$

即要么连续两步 high，要么 high 之后过两步又 high。这 filter 掉 isolated spurious prediction 但仍 responsive。

每个 subtask 最多 retry R=3 次，避免无限循环。

---

## 6. 实验数据的关键 reading

### 6.1 Table I: LIBERO 主结果

最 striking 的是 **LIBERO-Long**：CycleVLA 93.6% vs 其他 strong baselines：
- OpenVLA: 53.7%
- TraceVLA: 54.1%
- SpatialVLA: 55.5%
- FPC-VLA: 82.2%
- GR00T N1: 90.6%

Long suite 是 long-horizon task，错误会跨 subtask 累积，传统 VLA 一旦走错就一路错下去。CycleVLA 的 self-correction 在每个 subtask 边界都有机会 reset，所以 long-horizon 的 benefit 最显著。这跟 paper 的 core motivation 完美吻合。

Spatial/Object/Goal suite 上 CycleVLA 也都 90+，平均 95.3% 是 SOTA。

### 6.2 Table II: Under-trained VLAs 的 recovery

这个表非常 informative。看 200K checkpoint：
- w/o FC（无 failure correction）: avg 73.2%
- w/ FC: avg 80.0% (+6.8)

500K checkpoint：
- w/o FC: 89.3%
- w/ FC: 95.3% (+6.0)

**关键观察**：200K+FC ≈ 350K w/o FC（80.0 vs 83.2），350K+FC ≈ 500K w/o FC（89.2 vs 89.3）。也就是说，CycleVLA 的 correction 机制相当于 *免费* 给你 boost 了 ~150K training steps 的 capacity。这对于计算受限、无法 train 大 model 的小 lab 是巨大的实用价值。

### 6.3 Table III: N 的影响

MBR 的 sample 数 N 从 4 增到 64：
- N=4 → 8: 显著提升（CycleVLA-200K: 78.2 → 78.5，350K: 88.0 → 90.2）
- N=8 → 16: 仍有提升
- N > 16: plateau

random selection 几乎等于 base VLA 的 SR，因为它直接从 policy distribution 采样，marginal behavior 不变。这证明 MBR 的 gain 来自 *selection* 而非 *sampling* 本身。

N=8 是 sweet spot——accuracy 和 compute 的 trade-off。

### 6.4 Table V: Runtime analysis

总 inference time A10 上 215.3s, A100 上 76.9s。Test-time scaling 大约增加 ~30% runtime。

A10 breakdown:
- Action Rollout: 147.6s (68.6%) — 最大头
- Action Sampling: 47.9s (22.2%)
- VLM API: 12.9s (6.0%)
- Backtrack execution: 6.9s (3.2%)
- MBR computation: 0.003s (<0.1%) — 几乎免费！

MBR 的 pairwise distance 在 $\mathbb{R}^{6H}$ 上算 N×N=64 个 pair 是 trivial 的。这说明 MBR decoding 的 marginal cost 几乎为零，**全部 cost 来自 forward pass 采样**。如果未来能用 speculative decoding 或 batch parallel 优化 sampling，整个 pipeline 会更便宜。

### 6.5 Table VI: Ablation

几个有意思的点：
- **w/o MBR** (用 random selection 代替): SR 92.5（-2.8），但 time 反而 302.4s（更长！）。因为 random selection 失败率高，trigger 更多 backtrack retry，runtime 累加。
- **alt. VLM** (LLaMA-3.2-11B 替代 GPT-5.2): SR 92.8 (-2.5), time 172.6s。本地 VLM 快但更倾向 transit，少 backtrack。
- **always-on MBR** (upper bound): SR 96.9 (+1.6)，time 464.3s。说明 VLM 是 *选择性 trigger* MBR，避免了不必要的 cost。
- **pred. failure cutoff** (lower bound): SR 79.7。这是 VLM sycophancy 测试——让它判断失败它就倾向说失败，导致提前 terminate。说明 VLM 的 failure prediction 有 bias。

---

## 7. 几个我想深挖的 intuition

### 7.1 为什么 subtask transition 是 failure hotspot

这是 paper 的 founding observation，但 paper 没给很深的解释。我自己理解：subtask transition 涉及 *state precondition* 切换。比如 "grasp the mug" 完成时，gripper 必须稳定 close 在 mug handle 上。如果 grasp 稍微 misaligned，下一个 subtask "move to plate" 一动 gripper，mug 就掉了。precondition 是 *跨 subtask 传递* 的，前一个 subtask 的微小瑕疵会在下一个 subtask 放大。这就是 long-horizon task 的 fundamental difficulty——错误 compound。

CycleVLA 在 subtask 边界让 VLM 评估 precondition，本质上是在做 *precondition checking*，类似 classical planning 里的 STRIPS precondition 但用 VLM 做 perception-grounded checking。

### 7.2 VLA 缺乏 progress estimation 是什么意思

参考文献 [19] (SeqVLA) 指出现有 VLA "lack mechanisms for stopping or progress estimation"。这是 deep issue：传统 imitation learning 训练的是 $P(a_t | o_t, g)$，每步 condition on 当前 observation，但 *没有显式的 episode-level 或 subtask-level 时间结构*。policy 不知道自己处于 subtask 的哪个 phase，所以也无法判断 "快结束了要小心"。

CycleVLA 通过 supervision signal $p_t = (\text{current step} - \text{subtask start}) / (\text{subtask end} - \text{subtask start})$ 给 VLA 注入 phase awareness。这跟 LLM 里的 position encoding 有点类似——给 policy 一个 "你在哪里" 的 sense。

### 7.3 Reverse execution 的局限

paper 在 Limitations 里承认：backtracking 假设 reversible state transitions，动态环境或 irreversible 环境不适用。比如你打翻了水，reverse 不回去。这是 fundamental limitation——proactive correction 假设你在错误 *manifest* 前就 detect 到，所以还没有 irreversible damage。这跟 human 的 proactive correction 是同构的：人类也是 *在事态恶化前* 收紧 grip。所以这个 limitation 实际上是 *设计选择* 而非缺陷。

### 7.4 跟 test-time compute scaling 的关系

最近 LLM 圈的 test-time scaling ([Snell et al. 2025](https://arxiv.org/abs/2408.03314), [DeepSeek-R1](https://arxiv.org/abs/2501.12948), [o3/o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)) 主要靠 *reasoning chain*——让 model 在 inference 时多想几步。VLA 不一样：它的 output 是 continuous action，不是 discrete token，没有 "多想几步" 的明确对应物。

CycleVLA 的 MBR decoding 是 VLA test-time scaling 的一种 form：sample N 个 hypotheses 然后 select。这跟 LLM 里 *best-of-N* 或 *self-consistency* 是同构的，只是 distance metric 换成 trajectory feature space 上的 $L_2$。Paper 在 Section II-C 也讨论了 RoboMonkey（Gaussian perturbation + trained VLM verifier）和 Rover（learned reward model）作为 alternative test-time scaling for VLA。CycleVLA 的优势是 *training-free*，不需要额外训 verifier。

未来方向可能是 *learned* test-time scaling for VLA——train 一个 critic 或者 process reward model 专门 score action chunk quality。但 paper 这里 zero-shot MBR 已经是个 strong baseline。

---

## 8. Limitations 和 future work 思考

paper 自己列了几个：
1. **Reversible state assumption**：irreversible 环境不适用。其实可以扩展成 "checkpoint-based backtracking"——不 reverse 物理 state，而是 reverse 到 simulator snapshot 或 reset 到上次 safe state。
2. **High-frequency contact-rich task**：MBR 需要 N 次 forward pass，对 100Hz control 不友好。需要 spec decoding 或 parallel sampling 优化。
3. **External VLM dependency**：现在靠 GPT-5.2 做 failure prediction。未来可以 end-to-end 训练 VLA 自己做 failure reasoning，类似 ECoT 把 reasoning 塞进 action prediction 里。
4. **Real robot 实验**：paper 只在 LIBERO simulation 上做，real hardware 实验待补。这是 major gap。

我自己想加几个：
- **Subtask decomposition 的 generalization**：现在 LLM 用 constrained vocabulary (move/rotate/open/close)。对于 *contact-rich* task（insertion, threading, assembly），这些 verb 不够。可能需要更丰富的 primitive library。
- **VLM sycophancy**：Table VI 显示 VLM 倾向 confirm "failure" 假设。可能需要 *calibration* 或者用 *contrastive prompting*（同时问 "is this succeeding" 和 "is this failing"）。
- **MBR 的 failure mode**：如果所有 N 个 hypotheses 都失败了（policy 整体走偏），MBR 还是会选一个失败的。这种情况下 MBR 无效。需要 detect "all hypotheses are bad" 的情况，trigger 更 high-level replanning。
- **Subtask granularity 的 sensitivity**：太细 subtask 多 overhead 高，太粗 transition 少 loss correction 机会。这个 trade-off 没在 paper 里 study。

---

## 9. Reference links

主参考：
- [CycleVLA 完整 paper](https://arxiv.org/abs/2511.00062) (本篇)
- [OpenVLA project](https://openvla.github.io/) / [paper](https://arxiv.org/abs/2406.09246)
- [OpenVLA finetuning (Kim et al. 2025)](https://arxiv.org/abs/2505.04678)
- [LIBERO benchmark](https://lifelong-robot-learning.github.io/LIBERO/) / [paper](https://arxiv.org/abs/2306.03310)
- [π0 (Black et al. 2024)](https://arxiv.org/abs/2410.24164)
- [π0.5](https://arxiv.org/abs/2504.16054)
- [GR00T N1](https://arxiv.org/abs/2503.14734)
- [RDT-1B](https://arxiv.org/abs/2410.07872)

方法相关：
- [Bellman-Guided Retrials (Du et al. 2024) - "To err is robotic"](https://arxiv.org/abs/2406.15917)
- [PAINT (Xie et al. 2022)](https://arxiv.org/abs/2210.11215)
- [ECoT (Zawalski et al. 2024)](https://arxiv.org/abs/2407.08693)
- [RoboMonkey](https://arxiv.org/abs/2510.10975)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [AHA (Duan et al. 2025)](https://arxiv.org/abs/2410.07384)
- [REFLECT (Liu et al. 2023)](https://arxiv.org/abs/2310.14525)

MBR decoding 理论：
- [Kumar & Byrne 2004 - 原始 MBR 论文](https://aclanthology.org/N04-1020/)
- [Eikema & Aziz 2022 - sampling-based MBR](https://aclanthology.org/2022.emnlp-main.522/)
- [Heineman et al. 2024 - multi-prompt MBR](https://aclanthology.org/2024.emnlp-main.560/)
- [Snell et al. 2025 - test-time scaling](https://arxiv.org/abs/2408.03314)

LLM/VLM 基础：
- [Chain-of-Thought (Wei et al. 2022)](https://arxiv.org/abs/2201.11903)
- [GPT-4.1](https://openai.com/index/gpt-4-1/)
- [LLaMA 3 herd](https://arxiv.org/abs/2407.21783)

---

## 10. 总结：CycleVLA 给我的启发

这篇 paper 我觉得最重要的 takeaway 不是某个具体 trick，而是一个 paradigm shift：**robot policy 不应该是一次性 feedforward，而应该有 internal cycle of prediction-monitoring-correction**。这跟 LLM 里 chain-of-thought 让 model 多步 reasoning 类似——VLA 也需要 inference-time 的 multi-step processing，但形式上不是 token-level reasoning 而是 *subtask-level cycle*。

Progress signal $p_t$ 是这整个 pipeline 的 *trigger*——它把 VLA 从被动 actor 变成 active monitor。一旦 VLA 知道 "我在哪、走多远"，外部 VLM 就能在合适的时机介入。这个 trigger 机制比单纯的 "每隔 N 步 check 一次" 更 smart，因为它 *只在 risk 最高的时刻*（subtask 边界）trigger。

MBR decoding 的 elegance 在于它 *zero-shot*：不需要训 verifier，不需要 reward model，只要 policy 能 sample 就行。但它的有效性依赖 policy 自己有 *multi-modal 但 clustered* 的 output distribution——成功 action 在 cluster center，失败 action 散在边缘。这个 assumption 对 imitation-learned policy 成立，但对 RL-learned policy 不一定（RL 的 output distribution 可能更 unimodal）。

总体上，CycleVLA 是 VLA inference-time scaling 这个方向的一个 very clean baseline——未来肯定会有更多 variant：learned critic、process reward model、speculative sampling + MBR、end-to-end failure reasoning。但这篇 paper 把 "proactive correction = progress awareness + VLM planner + MBR retry" 这三件事拼起来，已经在 LIBERO-Long 上把 long-horizon success rate 从 ~55% 推到 93.6%，证明这个 decomposition 是 *对的*。
