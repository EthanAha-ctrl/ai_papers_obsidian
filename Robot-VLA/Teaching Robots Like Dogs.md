---
source_pdf: Teaching Robots Like Dogs.pdf
paper_sha256: 24d871f0d6aba5cc959c9834748399acb8b261f5a44e0c79b15b12f3924a80c0
processed_at: '2026-08-12T12:54:48-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Paper

## 一句话说清楚他们在干嘛

想象你训狗：你说 "过来"，同时招招手，狗就跑过来了。他们说，能不能让 quadruped robot 也这么听话？你不用 joystick，不用 keyboard，就用嘴说 + 手比划，robot 就能懂你要它干嘛，还能跳过障碍、绕过箱子。

这就是整个 paper 的 goal。

## 那这事儿难在哪？

**第一个难处：单个信号不够用。**

你说 "go there"，robot 根本不知道 "there" 是哪。你得同时用手指一下，它才知道是哪。Verbal command 给的是 "干啥"，gesture command 给的是 "往哪"，两个缺一个都不行。这跟人类沟通一模一样——你跟朋友说 "把那个给我"，但没指，朋友也会问 "哪个？"。

**第二个难处：数据太难收集。**

你要让 robot 学会听懂人话+手势，得收集大量 human-robot interaction 数据。但每次实验要两个人——一个发命令，一个用 teaching rod 引导 robot——这事儿慢得要命。他们整个 experiment 也就收集了不到 1 小时的数据。1 小时对 deep learning 来说几乎没东西。所以必须有办法让这 1 小时的数据撑起整个系统。

## 他们怎么搞的：三阶段 pipeline

### Stage 0：先训一个能跑能跳的 "底盘"

先把 low-level 的 locomotion controller 训好。这个 controller 的任务是：给它一个 navigation goal $\mathbf{g}$，它能操控 motor 让 robot 跑过去、跳过去。这部分用 RL 在 Isaac Gym 里训，借鉴了 Extreme Parkour 那篇的工作，能让 robot 做动态跳跃。

公式上，velocity planner 是：
$$v_{com} = K \| \mathbf{g}^{xy} - \mathbf{p}^{xy} \|_2$$

就是看 robot 当前位置 $\mathbf{p}^{xy}$ 和目标 $\mathbf{g}^{xy}$ 的距离，乘个 gain $K$ 得到前进速度。离得远跑快点，离得近慢下来。

$$\phi_{com} = \text{yaw}(R^\top (\mathbf{g} - \mathbf{p}))$$

这个是算旋转——把目标方向转到 robot frame 下，提取 yaw 角。就这么简单。

这部分训好后就冻结了，后面不动它。

### Stage 1：两个人去 "遛 robot"

这一步特别像训狗中的 luring。两个人配合：

- 一个人拿 teaching rod 在前面引导 robot（相当于狗 trainer 手里的 lure）
- 另一个人在旁边发 verbal command + 做 gesture
- Robot 跟着 rod 走，同时记录下所有数据：robot state $\mathbf{x}$、target position $\mathbf{g}^*$、gesture $\mathbf{m}$、verbal $\mathbf{v}$、obstacle $\rho$

这些数据就是 "interaction data" $\mathcal{D}$。

他们设计了 6 个 scenario + 1 个 stop：
- **Go there**：指一下说 "go there"，robot 往指的方向走
- **Come here**：招手说 "come here"，robot 过来
- **Follow me**：抬一只手说 "follow me"，robot 跟着那一侧走
- **Come around**：站箱子前说 "come around"，robot 绕过箱子过来
- **Jump over**：挥手说 "jump over"，robot 跳过箱子
- **Zigzag**：手势比划说 "zigzag"，robot 在轮胎间穿插走
- **Stop**：双手抬起说 "stop"，robot 停

每个 scenario 收集几分钟数据，加起来不到 1 小时。

### Stage 2：在 simulation 里 reconstruct 场景，然后用 DAgger 训练

这是整个 paper 最关键的一步。

**为啥不能直接做 Behavior Cloning（BC）？**

BC 就是拿 demo data 直接监督学习，学一个 $\pi_g(\mathbf{c}, \mathbf{x}, \rho) \to \mathbf{g}$ 的 mapping。但 BC 有个致命问题：**distributional shift**。

具体啥意思？你 demo 里 robot 走的 trajectory 都是 "好的" trajectory——robot 在正确位置、正确速度、正确时刻。但训练完部署时，policy 输出的 goal 会让 robot 跑偏一点点，下一帧 robot 就不在 demo 见过的 state 上了，policy 就更慌了，输出更乱，robot 更偏——snowball effect，很快崩掉。

DAgger 的 solution：让 policy 自己去 explore，跑到 demo 没见过的 state，然后问 expert "这个 state 你会输出啥 goal"，把 expert 的 answer 加进 dataset 重新训。

这里的 "expert" 是一个 **local expert** $\tilde{\pi}_g$——它知道 global target $\mathbf{g}^*$ 在哪，给定 robot 当前 state，它输出一个 local frame 下的 goal 指向 $\mathbf{g}^*$。相当于一个 "永远指向终点" 的导航函数。

然后 domain randomization 强制让 robot 偏离 expert trajectory：
- External push $f \in [0, 0.5]$ m/s，每 3 秒推一下
- Terrain scaling $s \in [0.75, 1.25]$，把 obstacle 间距随机缩放
- Binary height map threshold 0.05m
- 加 square/circle distractor objects

Robot 被推飞了，expert 告诉它 "goal 在那边"，它学会 recover。这就是 DAgger + randomization 的核心：**让 policy 见过 failure mode 并学会 fix**。

参考 DAgger: https://arxiv.org/abs/1011.0686

## Progressive Goal Cueing：一个很聪明的小 trick

这个点很小但很关键。

**问题**：demo 是按时间戳录的。Human 在 robot 完成动作 A 后立刻发 command B。但训练早期 policy 很烂，robot 还在走 A 呢，simulation 时间已经推进到 B 该发出的时刻了。结果 robot 接收到的 command 跟它当前 state 不匹配——它还在 A 的中途，但收到的是 B 的指令。Supervision 信号直接变 noise。

**Fix**：不发新 command，等 robot 到达 current goal 再发下一个。50% 概率用这个 progressive cueing，50% 按正常时间推进。

**直觉**：这就像训狗时狗还没坐下你就喊 "趴下"，狗会懵。好的 trainer 会等狗坐下了再发下一个指令。Progressive cueing 就是让 simulation 里的 "virtual human" 也有这个耐心。

后期 robot 学会了，走得快了，waiting behavior 自然 fade out，因为 50% 还是按时间推进的。

## Gesture 和 Verbal 怎么表示

**Gesture**：用 motion capture 抓 6 个 upper-body keypoints——左右肩、肘、腕。然后表示成：
$$\mathbf{m} = [\mathbf{d}_{sh.r \to el.r}, \mathbf{d}_{el.r \to wr.r}, \mathbf{d}_{sh.l \to el.l}, \mathbf{d}_{el.l \to wr.l}, \mathbf{d}_{sh.l \to sh.r}, (\mathbf{p}_{sh.l} + \mathbf{p}_{sh.r})^{xy}/2, \phi_h]$$

就是各种 **相对 unit vector**——上臂方向、前臂方向、双肩连线方向、双肩中点、human 朝向。全部在 robot frame 下表达。这样跨场景泛化好——不管 human 站哪，相对关系是一致的。

**Verbal**：Silero VAD 切分语音 → Whisper 转录 → text encoder 编码成 vector。然后一个很聪明的 trick：用 GPT-4 把每个 verbal command 改写成 $N_v = 20$ 个 paraphrase，训练时随机采样一个。比如 "come here" 可能变成 "get over here"、"come to me"、"过来" 等等。这样 verbal 的 generalization 大大提升。

参考 Whisper: https://arxiv.org/abs/2212.04356

## 实验结果

### Baseline 对比

| Method | Success Rate | Navigation Error |
|--------|-------------|-----------------|
| BC | ~75% | baseline |
| GAIL | ~46% | worst |
| DAgger (no cueing) | ~89% | -24.6% vs BC |
| **LURE (full)** | **97.15%** | **-15.2% vs DAgger** |

几个观察：
- BC 不够好，因为 distributional shift
- DAgger 比 BC 好，证明 data aggregation 有效
- Progressive cueing 在 DAgger 基础上又提升一截
- GAIL 反而最差，因为 adversarial training 不稳定 + mode collapse

### Modality Ablation

| Scenario | Both | Verbal only | Gesture only |
|----------|------|-------------|--------------|
| Go there | 0.85 | 0.35 | 0.15 |
| Come here | 0.91 | 0.01 | 0.15 |
| Jump over | 0.83 | 0.67 | 0.23 |
| Zigzag | 0.96 | 0.00 | 0.16 |

Gesture only 平均 19.83%，基本废了。Verbal only 在 "Come here" 这种没 spatial grounding 的 task 上直接 0.01——robot 不知道往哪走。Jump over 是个 outlier，因为 box 在视野里，verbal 就够触发了。Both 一起用平均 0.91，error 0.43m。

**结论：verbal 给 semantic，gesture 给 spatial grounding，两个缺一不可。**

### Novel User Adaptation

3 个新 user 各做 ~4.5 min fine-tune：

| Subject | Pre | Post | Δ |
|---------|-----|------|---|
| #1 | 74% | 95.3% | +21% |
| #2 | 80.8% | 91.5% | +11% |
| #3 | 64.3% | 98.7% | +34% |

4.5 分钟就能 adapt 到新人，这个 data efficiency 已经很 consumer-friendly 了。

## 这篇 Paper 的几个核心 Insight

**1. 分层架构是关键。** High-level policy 只学 "intent → goal"，low-level policy 管 motor control。这样 high-level 不用管怎么跳，只管 "用户想让我跳"。解耦让两边各自专心。

**2. Luring 是个 brilliant 的 data collection paradigm。** 一次 luring 同时收集了三样东西：kinesthetic demonstration（rod 引导）、verbal label、gesture label。效率极高。

**3. DAgger 在 limited data 下是 distributional shift 的正解。** 1 小时数据直接 BC 必崩。DAgger + domain randomization 让 policy 见过失败并学会恢复。

**4. Progressive cueing 揭示了一个 subtle 的人机交互本质。** Human 不是固定时间序列，而是 state-conditional feedback controller。人会等 robot 完成动作再发下一指令。Demo replay 必须模拟这个 "等待" 行为。

**5. Multimodal 互补是刚需。** 不是 nice-to-have，是 must-have。单独 verbal 缺空间 grounding，单独 gesture 缺 semantic。这跟人类沟通的本质完全一致。

## 有啥 Limitation

1. **Gesture 没有 semantic augmentation**，只有 noise injection。不像 verbal 有 LLM 改写。所以换 user 时 gesture style 变了就完蛋，得 fine-tune。
2. **Progressive cueing 只是 temporal realignment**，不是真的 conditional human model。理想情况应该是 human 根据 robot 当前 state 动态生成新行为。
3. **Vision 假设太强**：human 穿 motion capture suit，obstacle 是预定义的 square/circle primitive。要 in-the-wild 得用 vision-based pose estimation + scene reconstruction。

参考 WHAM (vision-based human motion): https://wham.is.tue.mpg.de/

---

**一句话总结**：这篇 paper 把 dog training 的 luring 技术搬到 robot 上，用 teaching rod 引导 + verbal + gesture 三模态数据，在 simulation 里用 DAgger + domain randomization + progressive goal cueing 训练，1 小时数据达到 97% success rate。核心 insight 是分层架构 + multimodal 互补 + DAgger 解 distributional shift + progressive cueing 解 temporal alignment。

---

# Teaching Robots Like Dogs: 多模态 Human-in-the-Loop Luring 框架深度解析

## 1. 核心思想与类比直觉

这篇 paper 的核心 insight 在于把 **dog training 中的 luring 技术**迁移到 quadruped robot 上。Dog trainer 用 lure（食物或玩具引导）+ gesture + verbal cue 三层信号教狗技能；作者把 teaching rod 当作 lure，把 verbal command ("jump over") + gesture command（手臂指向）作为 social cue，让 robot 学习解读 human intent 并输出 navigation goal。

这种类比在直觉上有几个关键点：
- **Luring 解决 cold-start 问题**：物理引导让 robot 直接经历成功轨迹，绕开了从零开始的 exploration
- **Multimodal 解决 ambiguity 问题**："go there" 单独是 ambiguous 的，但配上 pointing gesture 就 grounded 到具体空间目标
- **Data aggregation 解决 distributional shift**：BC 只学会 demo trajectory 上的 state distribution，但 deployment 时 robot 会 drift 到 unseen states

参考 DAgger 原始 paper: https://arxiv.org/abs/1011.0686

## 2. 系统架构分层解析 (Fig 3)

整个 framework 是一个 **hierarchical 双层结构**，作者借用 Kahneman 的 System 1/System 2 隐喻：

### 2.1 Low-level Locomotion Controller (π_u, System 1)

这是 fast, reactive 层，负责具体的 motor control。结构分解：

**Velocity Planner**:
$$v_{com} = K \| \mathbf{g}^{xy} - \mathbf{p}^{xy} \|_2$$

- $v_{com}$: heading velocity command (m/s)
- $K$: gain coefficient
- $\mathbf{g}^{xy}$: navigation goal ground-projected 到 xy 平面
- $\mathbf{p}^{xy}$: robot base position ground-projected
- $\| \cdot \|_2$: Euclidean norm

$$\phi_{com} = \text{yaw}(R^\top (\mathbf{g} - \mathbf{p}))$$

- $\phi_{com}$: rotation command (rad)
- $R \in \text{SO}(3)$: robot rotation matrix
- $\mathbf{g}$: full 3D navigation goal
- $\mathbf{p}$: full 3D base position
- $\text{yaw}(\cdot)$: extract yaw angle from vector

**Velocity Tracker**: 一个 50 Hz 的 RL policy，在 Isaac Gym 里训练，tracking $v_{com} \in [0, 1.0]$ m/s, $\omega_{com} \in [-\pi/3, \pi/3]$ rad/s。借鉴 Extreme Parkour [Cheng et al.] 的工作。

### 2.2 High-level Navigation Module (π_g, System 2)

这是慢的、reasoning 的层，把 multimodal interaction signal 解析成 navigation goal。公式 (1) 是整个 training 的目标：

$$\arg\min_{\pi_g \in \Pi} \mathbb{E}_{(\mathbf{c}, \mathbf{x}, \rho, \mathbf{g}^*) \sim \mathcal{D}^{\pi_g}} \left[ \| \mathbf{g}^* - \pi_g(\mathbf{c}, \mathbf{x}, \rho) \|^2 \right]$$

- $\pi_g$: 待优化的 navigation policy
- $\Pi$: policy 函数空间
- $\mathbf{c}$: interaction command = $[\mathbf{m}, \mathbf{v}]$（gesture + verbal）
- $\mathbf{x}$: robot state（base pos/orientation, joint angles, velocities）
- $\rho$: heightmap（terrain/obstacle representation）
- $\mathbf{g}^*$: ground-truth intended navigation goal
- $\mathcal{D}^{\pi_g}$: **关键**——这是 current policy 诱导的 state distribution，对应 DAgger 的 on-policy data aggregation
- $\| \cdot \|^2$: MSE loss

注意这里 **action 是 navigation goal 而不是 motor torque**——这是个 design choice，把 high-level intent 与 low-level execution 解耦。

参考 Isaac Gym: https://arxiv.org/abs/2108.10470  
参考 Extreme Parkour: https://extreme-parkour.github.io/

## 3. MDP Formulation 详解 (Fig 2)

State transition 由三个 function 组成：

$$\mathbf{x}' = f(\mathbf{x}, \mathbf{u})$$ — robot dynamics

$$\mathbf{u} = \pi_u(\mathbf{g}, \mathbf{x}, \rho)$$ — locomotion controller

$$(\mathbf{c}', \mathbf{g}^{*'}) = h(\mathbf{x}', \rho, T)$$ — **human model**

Human model $h$ 是个难点：real-world 中你没法手动编码 "human 在每个 state 应该说什么做什么"。作者用 demonstration data 来近似成 $\tilde{h}$，通过 progressive goal cueing 来动态 replay。

## 4. Data Aggregation 的核心机制

DAgger 的核心 idea：agent 用 current policy $\pi_g$ 去 explore，然后用 expert $\tilde{\pi}_g$ 给出当前 state 的 correct action，加到 dataset 重新训练。这里 **local expert** 是怎么定义的？

$\tilde{\pi}_g$ 在 robot frame 下输出 expert navigation goal $\tilde{\mathbf{g}}$，directing robot 朝向 global target $\mathbf{g}^*$。这就要求 demo 收集时 velocity planner 是在 robot frame 下 driving 的——这是个一致性假设。

Domain randomization 强制 agent 偏离 expert trajectory：
- **External perturbation**: $f \in [0, 0.5]$ m/s，每 3 秒平均一次
- **Terrain scaling**: $s \in [0.75, 1.25]$ 在 $3 \times 3$ tiles 上
- **Binary height map threshold**: $h_{thres} = 0.05$ m
- **Distractor objects**: square + circular 形状

直觉：这种 randomization 让 robot 经历 "差点摔倒但被 expert 纠正" 的 trajectory，学会 recovery behavior。这跟仅做 BC 的根本区别就在这里——BC 看不到 failure mode，DAgger 看到了还能 label。

## 5. Progressive Goal Cueing 的细节

这是 paper 的一个亮点。Naive replay 的问题：
- Demo 是按 timestamp 录的，human 在 robot 完成前一个动作后立刻发出 next command
- 但训练早期 policy 表现差，robot 走得慢，新 command 已经发出但 robot 还没到达前一目标
- 结果：state-command misalignment，supervision 信号变 noise

Progressive goal cueing 的 fix：
- 每个 interaction command 保持 constant 直到 robot 到达 current goal
- 到达后才 update 到 next command
- 训练时以 50% 概率 apply progressive cueing，否则按 simulation clock 推进
- 这种 stochastic mix 防止 robot 一直 waiting，且 waiting behavior 在训练后期自然 fade out

直觉：这其实是给 human model $\tilde{h}$ 加了一个 "state-conditional 延迟"——human 在等 robot。这模拟了 real interaction 中人会等 dog/robot 完成动作再发下一指令的常识。

## 6. Gesture 与 Verbal Representation

**Gesture command**（6 个 upper-body keypoints: shoulders, elbows, wrists）：
$$\mathbf{m} = [\mathbf{d}_{sh.r \to el.r}, \mathbf{d}_{el.r \to wr.r}, \mathbf{d}_{sh.l \to el.l}, \mathbf{d}_{el.l \to wr.l}, \mathbf{d}_{sh.l \to sh.r}, (\mathbf{p}_{sh.l} + \mathbf{p}_{sh.r})^{xy}/2, \phi_h]$$

- $\mathbf{d}_{a \to b}$: 从 point $a$ 到 point $b$ 的 unit vector
- $sh/el/wr$: shoulder/elbow/wrist
- $r/l$: right/left
- $(\mathbf{p}_{sh.l} + \mathbf{p}_{sh.r})^{xy}/2$: 双肩中点的 ground projection
- $\phi_h$: human yaw

这是一个 **body-frame 相对表示**——所有特征都 relative to robot，让 policy 跨场景泛化。

**Verbal command**: Silero VAD 分割 → Whisper 转录 → pretrained text encoder [E5/BGE 系] 编码。然后 LLM (GPT-4) augmentation 生成 $N_v = 20$ 个 paraphrase，训练时随机采样。

参考 Whisper: https://arxiv.org/abs/2212.04356  
参考 Silero VAD: https://github.com/snakers4/silero-vad  
参考 text encoder (E5): https://arxiv.org/abs/2212.03533

## 7. 实验数据深度解读

### 7.1 Baselines 对比 (Fig 6 + Table I)

| Method | 平均 Success Rate | 相对 BC 改进 |
|--------|------------------|-------------|
| BC | ~75% | baseline |
| GAIL | ~46% | -29.32% |
| DAgger (no cueing) | ~89% | +18.6% |
| LURE (full) | 97.15% | +13.7% (vs DAgger) |

**DAgger vs BC**: data aggregation 减少 24.6% navigation error → 验证 distributional shift 是核心问题。

**LURE vs DAgger**: progressive goal cueing 额外减少 15.2% error + 13.7% success rate gain → state-command alignment 极其重要。

**GAIL 反常表现**: 在 Zigzag（最难 task）上比 BC 强，但整体差。作者归因于 adversarial training 的 instability + mode collapse。这跟 GAN-based imitation learning 文献中常见 failure mode 一致。

参考 GAIL: https://arxiv.org/abs/1606.03476

### 7.2 Novel User Adaptation (Table I)

3 个 subject 各做 ~4.5 min fine-tune：

| Subject | Pre-adapt | Post-adapt | Δ |
|---------|----------|-----------|---|
| #1 | 74.00% | 95.33% | +21.33% |
| #2 | 80.83% | 91.50% | +10.67% |
| #3 | 64.33% | 98.67% | +34.34% |

Subject #3 提升最大（34%），说明初始 model 对这个 user 的 gesture style mismatch 最严重——inter-subject variability 在 gesture modality 上确实很大，这呼应了 Limitations 里的讨论。

### 7.3 Modality Ablation (Table II)

| Scenario | Both | Verbal | Gesture |
|----------|------|--------|---------|
| Go there | 0.85 | 0.35 | 0.15 |
| Come here | 0.91 | 0.01 | 0.15 |
| Jump over | 0.83 | 0.67 | 0.23 |
| Zigzag | 0.96 | 0.00 | 0.16 |

关键观察：
1. **Gesture only 平均 19.83%** — gesture alone 几乎不可用，且 robot 倾向于只在 gesture 持续很久时才动
2. **Verbal only 在 "Come here" 上 0.01** — 因为没有 spatial grounding，robot 不知道走到哪
3. **Jump over 是个 outlier**：verbal only 也能到 0.67，因为 box 在视野里，verbal 提示足以触发 jump behavior
4. **Both modality 平均 0.91，error 0.43m** — multimodal 是必须的

直觉：这跟人类沟通的 complementary 性质完全对应——speech 提供 semantic class（"jump" vs "come"），gesture 提供 spatial grounding（"where"）。这跟 embodied cognition 文献中 language + pointing 的 dual coding theory 高度一致。

## 8. Multi-obstacle Course (Fig 7)

这是 paper 的 "money shot"：单一 policy 完成 4 个连续 sub-task：
1. Zigzag through tires → 2. Stop → 3. Jump over box → 4. Go there (return)

连续跑 5 次成功。这说明 **single unified policy** 真的能 sequential reasoning，不是 per-task 单独训的拼凑。这在 system design 层面是个强 evidence：分层架构 + multimodal grounding 让 long-horizon task 变成 tractable。

## 9. Limitations 与 Future Direction

作者点出三个核心 limitation：

1. **Zero-shot user generalization 缺失**：gesture 没有 semantic augmentation（只有 noise $\epsilon_m \sim \mathcal{N}(0, 0.01)$），不像 verbal 有 LLM augment。需要 pretrained motion representation [MotionDiffuse]。

2. **Progressive goal cueing 是 temporal alignment**，不是真正的 conditional generation。理想是 human model 能根据 robot 当前 state 动态生成新行为。

3. **Vision system 受限**：用 motion capture suit + 已知 obstacle 形状（square/circle primitives）。WHAM [Shin et al.] 和 VolumetricSMPL 这些 vision-based pose estimation 方法可以去除这些假设。

参考 WHAM: https://wham.is.tue.mpg.de/  
参考 MotionDiffuse: https://arxiv.org/abs/2208.15901

## 10. Intuition Building 总结

这篇 paper 的几个关键 takeaway：

**A. Decomposition 是关键**: System 1 (locomotion) + System 2 (navigation) 的分层让 high-level policy 只学 intent → goal 映射，不用管 motor control。这跟 RT-2 / NaVILA 等 VLA 模型的思路异曲同工，但更 modular。

参考 NaVILA: https://navila-robot.github.io/

**B. Luring ≈ Kinesthetic Teaching + Multimodal Labeling**: 把 teaching rod 当 lure 等价于同时提供（1）kinesthetic demonstration（2）paired gesture/verbal label，一次采集三样东西。

**C. DAgger + Domain Randomization 是 distributional shift 的 combo**：仅 BC 在 sim 训练时也会 overfit demo distribution；DAgger 让 policy 自己 explore + expert 提供 label；randomization 强制 explore 到 out-of-distribution state。

**D. Multimodal 互补是刚需**：单独 verbal 缺 grounding，单独 gesture 缺 semantic。这跟 embodiment cognition 中 "language is for abstraction, gesture is for deixis" 的双轨理论对应。

**E. Progressive cueing 揭示 human model 的本质**：human 不是一个 oracle 时间序列，而是 state-conditional feedback controller。把 demo 当作 "human policy rollout" 而不是 "static trajectory" 来 use，是整个 framework 的 subtle 但核心 insight。

**F. 数据效率极高**: < 1 小时 demo + 6 个 scenarios + 97.15% success——这逼近了 practical deployment 的阈值。如果再降到 5 分钟 fine-tune，就真的可以是 consumer-grade interface 了。

参考 DAgger 原作者 Ross 的 lecture: https://www.cs.cmu.edu/~ggordon/  
参考 LfD survey: https://www.sciencedirect.com/science/article/pii/S0921889008001915

---

整体来看，这篇 paper 是一个相当 well-engineered 的 system paper：每个 design choice 都对应一个具体 failure mode（distributional shift → DAgger；temporal misalignment → progressive cueing；modality overfit → augmentation；sim-to-real → domain randomization），且 ablation 都能 isolate 每个组件的贡献。它的 contribution 主要不在新算法，而在于把一组 existing techniques (DAgger, RL locomotion, LLM augmentation) 在一个 multimodal human-in-the-loop setting 下做对了。这种 "把已知技术做对" 的工作往往比新算法对实际部署更 valuable。
