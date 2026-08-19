---
source_pdf: HumanCLAW Can Vision-Language Models Act Through a Body.pdf
paper_sha256: e124b1d65ec88c1758b253faae1784e29e3c9256eff31e0512cc5950063bd13f
processed_at: '2026-08-19T11:50:21-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 HumanCLAW

## 这篇 paper 在搞什么

一句话：**把一个 frozen VLM 塞进虚拟 humanoid 身体里，让它自己决定下一步怎么动，结果发现——它看得见世界，但感觉不到自己。**

作者管这个能力叫 **action intelligence**：不是 perception，不是 motor control，而是"我现在身体在哪、我刚才动了什么、下一步该干啥"这种 moment-to-moment 的决策。把 VLM 关在 closed loop 里，每 0.5 秒要做一个决策，决策被身体执行，物理后果回到下一帧 ego view，循环。

**为什么这事难测？** humanoid 身体一旦摔倒，你不知道是 VLM 选错 skill 还是 locomotion controller 没控住平衡。两种 cause 混一起，failure 没法归因。所以大部分 embodied benchmark 要么 abstract 掉 motion（像 ALFRED，action 就是 simulator script），要么 end-to-end 训 VLA（像 RT-2，reasoning 和 motor control 绑一起）。**两条路都没法 isolate "VLM 的决策本身行不行"。**

HumanCLAW 的解法是在中间切一刀：**decision 留给 VLM，motor execution 用一个 zero-shot reliable 的 motion generator + half-physics simulator 执行。** body 会撞墙、会推倒花瓶、会受 gravity 约束，但不会因 lose balance 摔倒。这样每个 failure 都能 attribution 到 "VLM 选错了什么"，而不是 "身体摔了"。

---

## 三层架构，一层一层说

### 第一层：harness（怎么 prompt VLM）

不 fine-tune，纯 prompt engineering。每步给 VLM：
- ego view RGB
- compact text history（最近几步的 visual state + reasoning + action）
- high-level goal（如 "find a sofa and sit on it"）

VLM 要按四层 scaffold 推理：

1. **contextual prompting**：固定信息——你是蓝色 humanoid，goal 是啥，skill pool 长啥样
2. **ego-perception**：先 output 一个 formatted visual state——从左到右看到什么、大致距离、target 是否 visible、附近有没有 obstacle / affordance
3. **high-to-mid-to-low reasoning**：high（该 search / approach / align / interact？）→ mid（几秒级别的 objective，如 "move toward visible object"）→ low（选具体 skill + parameter，输出 JSON）
4. **verifier**：short-context 检查器，看 VLM 提的 skill 在当前 spatial state 下会不会蠢（walk into wall, stop before target reached, climb without visible stair, sit from invalid pose）

**verifier 是关键。** 因为 VLM 在 long rollout context 下 spatial reasoning 会 degrade，会 hallucinate "我已经到了"。verifier 拿当前 ego-view 重新评估，相当于每步 reset 一下，防止 long-context drift。

**mid-level reasoning 也关键。** 去掉它，NavSR 几乎不变但 InteractSR 从 18.9% 跌到 0%。原因：sit 是个 short compositional routine——reach seat → turn align → sit。这几秒的 sequence 靠 pure reactive step-by-step policy 撑不起来，需要 mid-level objective 当 glue。这其实就是 Sutton 的 options framework 思想，但 implementation 是 VLM 显式 reasoning 而非 learned option policy。

### 第二层：motion generator（怎么把 skill 翻译成 100+ DoF 身体运动）

**痛点：** 你跟 VLM 说 "walk forward 0.3m"，body 不能瞬间启动——必须 decelerate, shift weight, recover balance, 产生 transient motion。这些 "imperfections" 是 embodied action 的一部分。但 decouple 要求 zero-shot reliable：说 walk 就得 commonsense walk，不能 moonwalk 或 stagger。

**skill set：** 8 个 atomic primitives，每个带 continuous parameter：

```
walk(x, z, ψ)          # 前进位移 + heading change
side_step(x)           # 侧步
step_back(z)           # 后退
turn_in_place(θ)       # 原地转
climb_upstairs(h, d)   # 爬楼梯（height, depth）
walk_downstairs(h, d)  # 下楼梯
sit_in_place(h)        # 坐下（target pelvis height）
stop
```

**关键设计哲学：故意不暴露 `sit(sofa at (x,z))` 这种 composite command。** find→approach→orient→commit 的 composition 留给 VLM 自己 reasoning，每个 primitive 单独可靠。这给你一个 interpretable action space，fail 归因清楚。

**motion generator 结构：**

base DiT（38M param, 10-layer, 8-head self-attention）学 motion prior，skill-agnostic：
- 输入：last 5 clean body states @ 30fps = 0.167s history
- 输出：next 15 frames @ 30fps = 0.5s future chunk
- 每帧 219 维（SMPL-X pose 表示）

公式 (3) 把 history 直接 concat 到 noisy future 作为 20 个 token：

$$\boldsymbol{u}_\tau = [x_{t-4:t},\ \tilde{x}_{t+1:t+15}(\tau)] \in \mathbb{R}^{20 \times 219}$$

- $x_{t-4:t}$：过去 5 帧 clean pose
- $\tilde{x}_{t+1:t+15}(\tau)$：未来 15 帧 noised state，依赖 flow time $\tau \in [0,1]$（$\tau=0$ 纯噪声，$\tau=1$ clean future）
- 219：每帧 pose 维度

**history-as-token** 这个设计很关键——不把 history 压成单个 condition vector，让 future token 直接 attend 到 exact recent pose 和 velocity。kinematic continuity 显式化。

公式 (4)：网络预测 velocity field $\hat{v}_\theta$，只保留 future token 输出。

公式 (5) flow matching 训练：

$$\tilde{x}(\tau) = (1 - (1-\sigma_{min})\tau)\epsilon + \tau x^{fut},\quad v^\star = x^{fut} - (1-\sigma_{min})\epsilon$$

- $\epsilon \sim \mathcal{N}(0, I)$：Gaussian noise
- $\sigma_{min}$：最小 noise level（防 $\tau=1$ 数值 issue，典型 1e-4）
- $v^\star$：target velocity field
- loss：$\|\hat{v}_\theta - v^\star\|_2^2$

推理从纯噪声开始，30-step midpoint solver 解 ODE。

**per-skill ControlNet adapter：** base DiT frozen，每个 skill 加一个 trainable adapter。公式 (6)：

$$z_{\ell+1} = B_\ell(z_\ell, \tau) + W_\ell\ C_\ell(z_\ell + e_s(c_t), \tau)$$

- $B_\ell$：frozen base block（提供 motion prior）
- $C_\ell$：trainable control block（copy 自 base DiT）
- $e_s(c_t)$：skill-condition encoder。scalar 用 MLP，spatial 用 Fourier features
- $c_t$：skill continuous parameter
- $W_\ell$：**zero-initialized** residual projection（ControlNet 核心技巧，训练初始不影响 base branch）

**最聪明的 trick：training data 的 condition 直接从 motion geometry 读出来。** walking 的 condition = chunk-local final pelvis displacement + yaw change。sitting 的 condition = target pelvis height。stairs = step height/depth。**不需要人工标注 action parameters**，只需 curate AMASS clips 让每个 adapter 单 purpose + commonsense-aligned。

**fidelity 验证（Table 1）：** 对比 MoMask（strong text-to-motion model），achievement ratio = achieved / commanded：

| | walk | side_step | step_back | turn | climb_up | climb_down | sit |
|---|---|---|---|---|---|---|---|
| **Ours** | 0.966 | 1.002 | 0.986 | 0.994 | 0.794 | 0.738 | 0.977 |
| MoMask | 1.921 | 1.717 | 2.072 | 0.608 | 0.820 | 1.540 | 1.817 |

Ours 几乎全 ≈ 1，std ≤ 0.12。MoMask 严重偏离，std 大一个数量级——即使 prompt 里写 "walks at 0.8 m/s"，它还是生成 habitual motion，完全 ignore requested magnitude。**这个 reliability 才让 benchmark failure 可以 attribution 到 decision。**

### 第三层：half-physics simulation

**痛点：** pure kinematic playback 是 ghost——穿墙、空中爬楼梯；fully dynamic torque controller 把 action reasoning 和 locomotor control 纠缠，humanoid 在 cluttered 环境或爬楼梯时容易 lose balance fall，prematurely 终止 episode。

**half-physics**（Siyao et al. 2025, https://arxiv.org/abs/2507.23778）：world 是 rigid-body physics（collision, friction, gravity, articulated objects, movable objects）；但 humanoid 不用 simulated joint torques 驱动，用 motion sequence 推导的 **equivalent kinematic velocities** 驱动：

$$\dot{q}_t = \frac{q_{t+1} - q_t}{\Delta t}$$

- $q_t, q_{t+1}$：相邻两帧 joint configuration
- $\Delta t$：simulation step size
- $\dot{q}_t$：linear joint velocity；rotational component 用 tangent space relative rotation 计算

**核心：** humanoid 不会 lose balance fall，但仍 physically interact——wall block body, contact push movable objects, unsupported vertical motion 受 gravity 约束。AI Habitat + Bullet engine, 120 Hz, passive joint stiffness $\lambda = 1.0$。

**intuition：** 给 humanoid 一个 "kinematic ghost body"——有 collision shape，有 mass（用于 contact response），遵循 gravity；但不会因 motor control 失误摔跤。撞墙停下，撞花瓶推倒，但 fail attribution 始终在 decision level。

---

## Benchmark 设计

### Task

**find-navigate-interact**：指令 "find a `<obj>`, navigate to it with zero distance, and finally sit on it"。progressive——每个 stage 必须完成前置才能进入下个。

数据：HSSD 41 个 validation houses，1218 episodes，6 target categories：
- **Sit subset（597 ep）**：bed, couch, toilet — full find-nav-interact
- **Navigation subset（621 ep）**：chair, potted_plant, tv — 终止于 navigation

movable objects（tabletop items, small furniture）设为 dynamic，contact 可 displace；large fixtures 保持 static。

### Difficulty 分层

每 episode 沿三个 geometric 维度评分：

1. **Distance**：到 nearest same-category goal 的 geodesic distance（A* 在 5cm 2D occupancy grid 跑 shortest path，median ratio 0.99 vs ground truth）
2. **Choice**：navigational branchiness = $>30°$ turns on Douglas-Peucker-simplified path + rooms traversed
3. **Obstacle**：route 1m 内 obstacle 数

Thresholds：distance ≤3.5/≤8.1m, choice ≤2/≤5, obstacle ≤2.5/≤5
分布：278 easy / 566 medium / 374 hard（约 23/46/31%，不强制 equal thirds）

### Metrics（四个维度）

**Progressive Success Rate：**
- **FindSR**：target 在 ego-view ≥100 px in 512×512 + model 在 visible-state text 里 acknowledge
- **NavSR**：pelvis 到 target AABB min distance ≤20cm + agent 主动 stop
- **InteractSR**：sit 后 pelvis link contact target mesh + agent 主动 stop
- **Geo- prefixed variants**：只看 objective criterion，不看 model acknowledgment——隔离 perception 和 decision

**Action Quality：** Motion Jerk = root-rigid jerk（三阶导数），at decision timescale。lower = smoother。
- single skill：4–6（forward walk 5.2, turn 4.3）
- purposeful walk-turn-walk：5.8
- aimlessly sequenced policy：9.7
- 5 forward-walk chunks 然后 5 lateral steps：4.7；interleaving 同样 10 chunks：7.4（purely from ordering）

**Body Awareness & Scene Disturb：**
- Coll.%：collision step fraction（floor-relative non-ground collision）
- #Dtb/ep：disturbed movable objects per episode
- dDtb：average displacement distance

**Cost：** avg tokens per episode（input + output）

---

## 实验结果

### 主表（9 个 VLM，paper date Aug 2026 假设的 future model 版本号）

**Finding 1：没有 VLM 解决 benchmark**

最强 Gemini-3.1：FindSR 64.9%, NavSR 42.4%, InteractSR **16.8%**。四个 model InteractSR ≤ 0.2%。stage-by-stage drop 锐利：

| VLM | FindSR | NavSR | InteractSR | Coll. | Jerk | in tok/step |
|---|---|---|---|---|---|---|
| GPT-5.5 | 55.1% | 13.9% | 3.4% | 43.4% | **4.2** | 4360 |
| **Gemini-3.1** | **64.9%** | **42.4%** | **16.8%** | 39.5% | 5.7 | 5890 |
| Gemini-2.5 | 58.5% | 21.6% | 3.5% | 37.7% | 8.7 | 6412 |
| Claude-4.8 | 32.6% | 8.6% | 1.5% | 44.2% | 5.3 | **7047** |
| Gemma-4-31B | 58.1% | 28.7% | 11.1% | 40.1% | 4.8 | 4632 |
| Qwen3.6-27B | 51.0% | 20.9% | 0.2% | 43.0% | 6.5 | 4862 |
| Qwen3.6-35B-A3B | 44.6% | 5.8% | 0.0% | 34.6% | 7.4 | 4766 |
| Qwen3.5-27B | 37.8% | 13.5% | 0.0% | 34.5% | 6.8 | 4641 |
| InternVL3.5-38B | 46.8% | 0.8% | 0.0% | 51.2% | 7.2 | 4459 |

几个有意思的反直觉点：

**landing rate**（NavSR / FindSR，"找到的"有多少"走到"）：Gemini-3.1 0.65, Gemma-4-31B 0.49, InternVL3.5-38B 0.02。识别目标普遍容易，把识别转化成 body stop next to target 是分水岭。

**cost ≠ performance：** Claude-4.8 用最多 input/output tokens per step（7047/625），trailing on 所有 success metric；Gemini-3.1 用更少 token 领先。

**low disturbance ≠ control：** Qwen3.5-27B 最少 disturb（0.93 obj/ep），最短 episode（37.6 steps），但 0% interact——inaction 不是 control；InternVL3.5-38B 和 Claude-4.8 collision 最高（51.2%, 44.2%）+ NavSR 崩——moves without control。

**action quality decoupled from success：** Gemini-2.5 success 不错但 Motion Jerk 最差（8.7），每 5 步一次 turn-reversal 摇摆；GPT-5.5 最 coherent（4.2）但 success 不强。

**open model 接近 frontier：** Gemma-4-31B 在三个 success rate 上都打平或超过 GPT-5.5 和 Claude-4.8，仅次于 Gemini-3.1。

### Ablation（100-episode mini-val, Gemma-4-31B backbone）

**Finding 2：reasoning scaffolds（memory, mid-level, verifier）lift action intelligence；longer text/visual history 不一定 helpful**

| Setting | FindSR | NavSR | InteractSR | in tok/step |
|---|---|---|---|---|
| Baseline (hist 10, img 1) | 58.0% | 27.0% | 18.9% | 4676 |
| hist 0 | 65.0% | 11.0% | 0.0% | 2582 |
| hist 20 | 59.0% | 27.0% | 7.5% | 6692 |
| hist 50 | 60.0% | 28.0% | 7.5% | 10538 |
| hist 100 | 56.0% | 26.0% | 11.3% | 12904 |
| img 2 | 57.0% | 31.0% | 9.4% | 5234 |
| img 5 | 55.0% | 28.0% | 7.5% | 6335 |
| img 10 | 53.0% | 13.0% | 3.8% | 7685 |
| **w/o verifier** | 51.0% | **2.0%** | **0.0%** | 4095 |
| **w/o mid-level** | 62.0% | 30.0% | **0.0%** | 3769 |

**Verifier 是 decisive component：** 去掉 → NavSR 27→2%, InteractSR 18.9→0%，FindSR 几乎不变。verifier 影响 acting 不影响 seeing。episode 也明显更短（47.5 vs 78.5 steps），agent 在 loop 关闭前 stop 或 drift。verifier 是把 skill proposal stream 转成 closed loop reaches & commits 的 anchor。

**Text history 必要但 saturate 快：** hist 0 → NavSR 11%, InteractSR 0%；hist 10/20/50/100 都在 26-28% NavSR；input token 从 4.7K 涨到 12.9K。compact history captures nearly all benefit at fraction of cost。

**More image frames can hurt：** img 2 neutral, img 5 大致 neutral, img 10 → NavSR 27→13%, InteractSR 18.9→3.8%。decision model 被 current view 的 reasoning bottleneck，flood 它 past frames dilute signal。

**Mid-level reasoning carries long-horizon interaction：** 去掉 mid-level，FindSR/NavSR 几乎不变甚至微升，但 InteractSR **18.9→0%**。原因：sit 是 short compositional routine——reach seat → turn align → sit。这几秒 sequence 靠 pure reactive step-by-step policy 撑不起来，需要 mid-level objective 当 glue。

---

## 四大 Error Analysis Findings

### Finding 3：Perception matters but not the bottleneck

GeoFindSR vs FindSR（acknowledged）gap 很小：Gemini-3.1 69.9 vs 64.9（5pt），GPT-5.5 63.2 vs 55.1（8pt），Gemma-4-31B 67.7 vs 58.1（10pt）。**一旦 target 真的 rendered 进 ego view，几乎一定被 recognize——recognition 是 largely solved。**

Find 失败 root cause 分布：
- 38% ineffective exploration（incoherent search, never render target）
- 23% genuine perception lapse（target rendered but unacknowledged）
- 12% ineffective actions
- 10% approaching without ever turning to view
- 9% unknowingly jammed against geometry
- 6% abandoning too-far goal
- 3% midpoint give up

**bottleneck 在 where & how agent 选择 explore。**

### Finding 4：Egocentric self-localization 是 navigation bottleneck

5473 episodes 中 agent 主动找到 target，3706（68%）仍 fail navigation。

失败 root cause 分布：
- **34% body awareness**：20% "arrived <0.2m 但没 signal arrival, 继续 navigate"；14% "jammed against obstacle 连续多步但继续 forward commands"
- **30% ego-spatial distance hallucination**：agent stop + declare arrival 但 still >0.2m away，相信 distant target 在 reach
- 17% mid-level reasoning（turned away → lost target → cannot re-acquire, 16%；found too late, <1%）
- 10% pure action/locomotion（free move 但 never converge）
- 9% visual perception（target left ego view 但 model 仍 claim see）
- <1% discovery timing

**~2/3 failures 是 egocentric self-spatial awareness 错误**，perception 和 low-level control 都不是主因。

### Finding 5：Reaching is not interacting

726 episodes 成功 navigate（active stop <0.2m），513（71%）仍 fail sit。

失败 root cause：
- **58% sit into thin air**：issue sit 但 pelvis 从未 contact 任何 surface——body 在站立位置直接 lower，下面没有 seat
- **14% sit on wrong thing**：pelvis 落在不同 object / floor / wall（goal 是 couch/bed/toilet 时最容易，邻居 invite 错位 landing）
- **9% stand after sit**：pelvis 短暂 contact seat 但继续 act（mostly turn，偶尔 stand + re-sit），最终 shift off seat
- **19% decision failure**：reach + stop 但 never issue sit

**81% 是 body awareness 错误**（put body in wrong place），19% 是 decision 错误。

### 隐含 Finding 6：collisions 集中在看不见的部位

Table 5：

| VLM | Coll.% | Arm/Hand | Torso | Leg/Foot | Head |
|---|---|---|---|---|---|
| GPT-5.5 | 43.37 | 23.24 | 13.78 | 38.83 | 2.99 |
| Gemini-3.1 | 39.48 | 20.97 | 11.72 | 33.29 | 2.62 |
| Claude-4.8 | 44.18 | 28.97 | 16.90 | 37.94 | 6.93 |
| InternVL3.5-38B | 51.17 | 34.68 | 23.12 | 44.75 | 6.10 |

Leg/Foot collide 最多（28-45% steps），Arm/Hand 第二（20-35%），Head 最少（<7%）。**body 的盲区 = VLM 不看的部位。**

Figure 9 两个 concrete 失败：(a) arm 几帧前明明在 view 里能看见椅子，撞倒了也没 register contact；(b) obstacle 直接在前方可见，agent 仍持续 walk legs into it。

---

## Discussion：核心隐喻

paper 的核心隐喻：**当前 VLM 像 ghost——fluent about the world, oblivious to itself。** 它把 limbs 当作 just more pixels，没有 proprioceptive, consequence-predicting model。原因：VLM 训练自 visual question answering，学 describe what it sees，没学 feel what it does。

作者在 Discussion 里 explicit state：generalizable action intelligence arises from reasoning, not from fitting action data。policy fit on trajectories 只能在 data support 内 generalize；reasoner 能 extrapolate as far as knowledge allows。HumanCLAW 建立在这个 bet 上——decision maker frozen + general, motor layer fixed + reusable, 所有 task-level composition zero-shot 在 reasoning 里完成。

paper 承认：当前 interface 没有 tactile channel，collision displace world 但不被 felt。所以 agent 的 proprioception 只能从 egocentric view reconstruct——可能 intrinsically difficult。body-state 或 contact signal 可能是 missing input，而不仅仅是 missing faculty。**这是一个 honest open question：到底是 VLM 需要 learn feel body，还是 interface 需要补 tactile channel。**

---

## 我的延伸联想

### 1. VLA 路线 vs HumanCLAW 的 philosophical 对比

π0（https://arxiv.org/abs/2410.24164）、OpenVLA（https://arxiv.org/abs/2406.09246）、RT-2（https://arxiv.org/abs/2307.15818）、GR00T N1（https://arxiv.org/abs/2503.14734）这些 end-to-end VLA 把 reasoning 和 motor grounding 绑一起 train，entangle。HumanCLAW 选 inverse 路线——reasoning frozen, motor layer reusable。两条路线在哲学上是相反 bet：

- **VLA bet：** data scaling 让 reasoning 从 action data 里 emerge
- **HumanCLAW bet：** reasoning 在 frozen generalist 里已有，只是没被 body-aware test 到

这其实是 pre-transformer 时代 specialized policy vs after GPT-3 generalist reasoner 之争的 embodied 版本。你 2017 年在 Tesla 提的 "software 2.0" 思路其实就是这个——let the network learn the mapping，but at decision level vs trajectory level 有 debate。

### 2. verifier 是 closed-loop anchor

ablation 显示去掉 verifier → NavSR 27→2%, InteractSR 18.9→0%。这防止 long-context degradation 下 hallucinated progress。这和你 nanoGPT 时代关注的 "model doesn't track state over long context" 是同一 issue。verifier 是 short-context reset，强迫每步重新 ground 到当前 ego-view。**这是 RL 里 value-based reset 的 prompt-engineering 对应物。**

### 3. mid-level reasoning 是 medium-horizon glue

去掉 mid-level → FindSR/NavSR 不变，InteractSR 18.9→0%。reactive step-by-step 不够，需要 medium-horizon commitment。这和 hierarchical RL 的 option framework（Sutton, Precup 1999, https://scholarpedia.org/article/Options）思想相通——但 implementation 是 VLM 显式 reasoning 而非 learned option policy。这其实是你 micrograd 风格 "minimal implementation" 的 prompt-engineering 对应——把 option 的 temporal abstraction 用 natural language 承载。

### 4. embodied self-awareness 是 next frontier

spatial intelligence literature（SpatialVLM https://arxiv.org/abs/2401.12168, Thinking in Space https://arxiv.org/abs/2412.14171, ESI-Bench https://arxiv.org/abs/2605.18746）已经发现 VLM spatial reasoning 弱。但 HumanCLAW 进一步——在 closed loop + physical body 下，spatial reasoning 的 bottleneck 在 **self-localization + body-state tracking**，perception largely solved。

这其实是 cognitive science 里 "body schema" / proprioception 的对应——VLM 缺一种 implicit body model。这让人想起前几年 robot learning 里流行的 "implicit forward model" 概念（learning to predict next state from action），但在 VLM scale 上还没人 systematic 做。**也许应该给 VLM 加一个 self-segmentation 的 auxiliary task，让它学会 "这些 pixel 是我"。**

### 5. atomic skill + continuous parameter 接口设计

比 discrete action space 更 expressive，比 raw 100+ DoF trajectory 更 interpretable。每个 skill 是 minimal unambiguous unit，composition 留给 VLM。这让人想起 hierarchical control 的经典思路（Brooks subsumption architecture, behavior-based robotics, http://people.csail.mit.edu/brooks/papers/AIM-864.pdf），但 high-level 换成 foundation model。

**这是个 generalizable pattern：给 generalist reasoner 一个 minimal reliable motor interface，让 composition emerge from reasoning。** 你同样可以套到 tool use, code execution, GUI 操作。Anthropic 的 computer use agent、OpenAI 的 Operator 都是同一哲学的变体。

### 6. half-physics 是 evaluation 哲学

fully physics 让 evaluation dominated by locomotion；pure kinematics 让 agent 是 ghost。half-physics 是 controlled 中间状态——保留 physical consequence，剥离 motor failure。这是 robotics eval 的 generalizable 思路：**你想 measure A 但不想被 B confound，那就 simulated 掉 B 但保留 A 的真实 consequence。**

这个 idea 可以推广到 manipulation——比如 grasp 不成功可能是 perception issue 或 force control issue，decouple 之后才能 attribute。Stanford 的 BEHAVIOR-1K（https://arxiv.org/abs/2303.07873）可以借鉴这个分层。

### 7. ghost metaphor 让我想起 mirror self-recognition test

classic 的 mirror test（Gallup 1970, https://doi.org/10.1126/science.167.3914.86）测 animal 能不能 recognize 自己。VLM 现在 fail 的方式很有意思——它能看到 scene，但 scene 里如果有自己身体的部位（比如 arm 入镜），它不会把那些 pixel 当作 "我的"。这其实是视觉 self-recognition 的失败。

更广义：**self-model 是 intelligence 的核心 component。** humanoid body 是 self-model 最 rich 的载体——100+ DoF, 物理后果, 时间 continuity。如果 VLM 在这上面 fail，它在其他 self-referential task 上可能也有类似 limit。这和 Douglas Hofstadter 的 strange loop（https://en.wikipedia.org/wiki/I_Am_a_Strange_Loop）思想有关。

### 8. 数据效率 vs generalization 的 tradeoff

HumanCLAW 强调 zero-shot transfer，但承认 16.8% success。VLA 路线在特定 embodiment 上可能更高 success 但 zero-shot 差。这个 tradeoff 在 RL 里经典——model-based vs model-free, generalist vs specialist。HumanCLAW 把这个 tradeoff 拉到一个新维度：**reasoning-based action vs trajectory-fit action**。

类比：LLM 早期 task-specific fine-tune（BERT era）vs prompt-based zero-shot（GPT-3 era）。现在 embodied AI 在 similar transition——从 task-specific VLA 到 reasoning-based zero-shot。HumanCLAW 像是 embodied 版的 GPT-3 moment，但还差一个 "scaling law for action reasoning"。

### 9. motion generator 的 modular 设计

Motion Base DiT + per-skill ControlNet 是个 plug-and-play 设计，新 skill 加一个 adapter 不动其他。这和 LoRA（https://arxiv.org/abs/2106.09685）、ControlNet（https://arxiv.org/abs/2302.05543）的 modular 思路一脉相承。**zero-initialized $W_\ell$ 让 adapter 渐进学习是 ControlNet 的核心技巧。** 用 motion geometry 自动 derive condition 是很聪明的 unsupervised parameterization。

这个设计可以直接套到其他 generative model 的可控 generation 上——比如 music generation 的 per-style adapter, video generation 的 per-camera-movement adapter。

### 10. failure attribution 是 benchmark 设计的核心

整个 paper 的方法论核心是 "decouple to attribute"。这其实是 causal inference 在 embodied eval 里的应用——你想知道 cause 是 A 还是 B，就 controlled 掉 B 看 A 单独的表现。**这个思路应该在更多 embodied benchmark 里推广。**

比如 manipulation benchmark 里，grasp 失败可能是 perception issue（看不见 object）或 force control issue（gripper 没控好力）。decouple 之后才能知道。现在大部分 benchmark 把两层绑一起，没法 diagnose root cause。HumanCLAW 的方法论是个示范。

### 11. embodied AI 的 "GPT-3 moment"

HumanCLAW 让我想到 GPT-3 2020 年那个时刻——"look, we just prompt a frozen general model and see how far it goes"。那时候 zero-shot success rate 也很低（many tasks < 50%），但 philosophical statement 很强：**reasoning is there, we just need better elicitation。**

HumanCLAW 是 embodied 版的 GPT-3 moment：frozen VLM, zero-shot, 16.8% success。但 16.8% 不是终点，是起跑线。下一步可能来自：
- 更好的 reasoning scaffold（coconut style latent reasoning, https://arxiv.org/abs/2412.06769）
- 显式 spatial memory module（不同于 text history）
- proprioceptive / tactile input channel
- implicit forward model（predict next ego-view from action）

这和 LLM 的 evolution 路径平行——GPT-3 → instruction tuning → RLHF → tool use → reasoning models。embodied VLM 现在在 "GPT-3 moment"，下个阶段应该是 "instruction tuning for body awareness"。

### 12. 一个 actionable direction

如果让我做 follow-up，我会试这个：**给 VLM 一个 implicit forward model。** 具体——在 harness 里，每 step 让 VLM 先 predict "如果我执行 skill $s_t$ with param $c_t$，下一帧 ego-view 大致长什么样"。然后用真实下一帧监督这个 prediction。这样 VLM 被迫 learn "action → observation" 的 mapping，也就是 implicit body model。

这个 idea 来自 robot learning 里 的 forward model literature（Finn & Levine 2017, https://arxiv.org/abs/1709.02514），但在 VLM scale 上还没 systematic 应用。HumanCLAW 的 framework 提供了 clean testbed——half-physics 给你 ground truth next observation，motion generator 给你 reliable action execution，failure attribution 已经 isolate 到 decision level。加 forward model auxiliary task 可以直接 measure 它 lift 哪些 failure mode。

---

## 相关 web links

**paper & project：**
- HumanCLAW project page: https://human-claw.github.io
- Half-physics (Siyao et al. 2025): https://arxiv.org/abs/2507.23778
- PRIMAL (Zhang et al. 2025c, ICCV 2025): https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_PRIMAL_Physically_Reactive_and_Interactive_Motor_Model_for_Avatar_Learning_ICCV_2025_paper.html

**仿真环境 & 数据集：**
- AI Habitat: https://arxiv.org/abs/1904.01201
- Habitat 2.0: https://proceedings.neurips.cc/paper/2021/hash/021bbc7ee20b71134d53e20206bd6feb-Abstract.html
- Habitat 3.0: https://openreview.net/forum?id=4znwzG92CE
- HSSD: https://arxiv.org/abs/2307.03584
- AMASS: https://amass.is.tue.mpg.de/
- BABEL: https://babel.is.tue.mpg.de/
- Bullet physics: https://pybullet.org/

**对比 benchmark & 任务：**
- ALFRED: https://arxiv.org/abs/1912.01734
- VirtualHome: https://arxiv.org/abs/1803.05637
- BEHAVIOR-1K: https://arxiv.org/abs/2303.07873
- PARTNR: https://arxiv.org/abs/2412.16115
- LoTa-Bench: https://openreview.net/forum?id=ADSxCpCu9s
- EmbodiedBench: https://arxiv.org/abs/2401.13953
- ESI-Bench: https://arxiv.org/abs/2605.18746
- VisualAgentBench: https://arxiv.org/abs/2408.06327
- VideoGameBench: https://arxiv.org/abs/2505.18134
- Voyager: https://arxiv.org/abs/2305.16291
- Cradle: https://arxiv.org/abs/2403.03186

**VLA & end-to-end control 对比：**
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- π0: https://arxiv.org/abs/2410.24164
- GR00T N1: https://arxiv.org/abs/2503.14734
- PaLM-E: https://arxiv.org/abs/2303.04271
- SayCan / Inner Monologue: https://arxiv.org/abs/2204.01691, https://proceedings.mlr.press/v205/huang23c.html
- Code as Policies: https://arxiv.org/abs/2209.07753
- VoxPoser: https://arxiv.org/abs/2305.18609
- ManipulateAnything: https://arxiv.org/abs/2403.02839

**spatial intelligence & VLM reasoning：**
- SpatialVLM: https://arxiv.org/abs/2401.12168
- Thinking in Space: https://arxiv.org/abs/2412.14171
- Unfolding spatial cognition: https://arxiv.org/abs/2506.04633

**controllable human motion generation：**
- MoMask: https://arxiv.org/abs/2401.11474
- MDM: https://arxiv.org/abs/2209.14991
- OmniControl: https://arxiv.org/abs/2310.08580
- PhysDif: https://arxiv.org/abs/2310.02955
- Bailando: https://arxiv.org/abs/2208.10635
- SAMI / scene-aware motion: https://arxiv.org/abs/2104.03580
- ControlNet: https://arxiv.org/abs/2302.05543
- LoRA: https://arxiv.org/abs/2106.09685

**additional theory & relevant papers：**
- Options framework (Sutton, Precup 1999): https://scholarpedia.org/article/Options
- Subsumption architecture (Brooks 1986): http://people.csail.mit.edu/brooks/papers/AIM-864.pdf
- Gallup mirror test (1970): https://doi.org/10.1126/science.167.3914.86
- Douglas Hofstadter, I Am a Strange Loop: https://en.wikipedia.org/wiki/I_Am_a_Strange_Loop
- Coconut latent reasoning (Hugging Face 2024): https://arxiv.org/abs/2412.06769
- Deep Visual Foresight (Finn & Levine 2017): https://arxiv.org/abs/1709.02514

---

**一句话总结：** HumanCLAW 把 VLM 关在 closed loop + physical body 里测 action intelligence，motor execution factor out 让 failure 全归因到 decision level。结果——VLM 看得见世界但感觉不到自己，瓶颈在 embodied self-awareness，不在 perception。16.8% 是起跑线，下一步可能来自 implicit body model / proprioceptive input / 显式 spatial memory module。

---

# HumanCLAW: 这篇paper想做什么

## 1. 核心 motivation 与 problem setting

paper 抛的问题非常直接：**把一个 frozen off-the-shelf VLM 装进 humanoid body，让它在 closed loop 里 step-by-step 决定身体下一步该做什么——这件事现在到底行不行？** 作者把这个能力命名为 **action intelligence**：在 execution loop 内 moment-by-moment 选择、parameterize、sequence atomic actions，每个决策被 physical body 执行，后果又回到下一个 egocentric observation。他们把它定位为 spatial intelligence 的 operational component。

为什么这件事难 measure？因为 humanoid 上 action outcome 把 **VLM decision** 和 **motor control** 紧紧耦合：爬楼梯失败可能是 VLM 选错 skill，也可能是 locomotion controller lose balance fall。两个 cause 混在一起，failure 就没法 attribution。

HumanCLAW 的 bet 是把这两层 decouple：VLM frozen + general, 输出 atomic skill command；一个 zero-shot reliable 的 motion generator 把 command 翻译成 sub-second 的 100+ DoF 连续 full-body motion；half-physics simulator 执行这个 motion chunk（保留 collision, gravity, contact, object displacement），同时 factor out balance & motor-tracking failures。这样每个失败 episode 都可以 attribution 到 decision level——VLM 选错了什么，而不是身体摔了。

**intuition：** 你可以把这套想成 VLM 在玩第一人称游戏，但 action space 是 parameterized whole-body skills 而不是 key presses；outcomes 由物理决定而不是 script。它填了 symbolic agent（无物理后果）和 end-to-end VLA（action data 绑定 reasoning）之间的空白。

Project page: https://human-claw.github.io

---

## 2. 三层架构解析

### 2.1 Egocentric Action Decision（harness）

harness 不 fine-tune VLM，是 prompt engineering + verifier。每 step VLM 收到：
- egocentric RGB observation $o_t$
- compact textual history $h_t$（recent steps 的 visual state, reasoning trace, action）
- high-level instruction $p$（如 "find a sofa and sit on it"）

VLM 输出一个 skill call：

$$\langle s_t, c_t \rangle = H(p, o_t, h_t) \tag{1}$$

- $H$：整个 harness（prompt + reasoning scaffold + verifier）
- $s_t$：atomic skill 名（来自固定 8-element skill set）
- $c_t$：该 skill 的 continuous parameter
- 下标 $t$：当前 timestep

harness 内部 reasoning scaffold 分四层：

**Layer 1 — Contextual Prompting：** 告诉 VLM 它是蓝色 humanoid robot + egocentric camera，goal 是什么，skill pool 长什么样，rules per skill。这一层 fixed，不随 episode 变。

**Layer 2 — Ego-Perception & Spatial Understanding：** 在选 motion 之前，VLM 必须先 output 一个 formatted visual state——从左到右 visible objects、approximate distance to body、target 是否 visible、immediate area 有没有 obstacle 或 affordance（stairs / open space / sitting surface）。这步把 implicit image-to-action guess externalize 成显式 spatial interpretation。后续决策 depend on 这个 explicit ego-view understanding，而不是 implicit guess。

**Layer 3 — High-to-Mid-to-Low Reasoning：**
- *High-level*：当前应该 search / approach / align / interact？
- *Mid-level*：维持几秒的 objective，如 "move toward visible object" 或 "turn before sitting"。**关键设计：每一步 VLM 都要显式决定 inherit 上一步的 mid-level objective 还是 revise 它**。这是 closed-loop reactive，而非 open-loop plan。同时保留 hierarchical planning 的 stability。
- *Low-level*：选具体 skill + parameter，输出 structured JSON

**Layer 4 — Skill-Specific Spatial Response Verifier：** short-context verifier 检查 VLM 提的 skill 在当前 spatial state 下会不会产生 unintended consequence（walk into obstacle, stop before target reached, climb without visible stair, sit from invalid pose）。**动机关键：VLM 在 long rollout context 下 spatial reasoning 会 degrade，会 hallucinate progress**（说"已经到了"但实际离很远）。verifier 用 compact skill-specific prompt，从当前 ego-view 重新评估。checklist 在 Appendix Table 6：

| Trigger skill | Checks |
|---|---|
| walk forward | path ahead 是否 clear；motion 是否 collide wall/furniture；前进是否 advance 当前 goal |
| stop | high-level goal 是否 complete；body 是否 arrived / contact target |
| climb up | 楼梯是否 directly underfoot；body 是否 facing staircase |
| climb down | 同 climb up |
| turn (before sitting) | body 是否 flush against seat（zero distance）；turn 后 seat 是否 directly behind body |

---

### 2.2 Skill-Conditioned Motion（paper 最 technical 的部分）

#### 2.2.1 为什么需要这一层

code agent 可以瞬间从 "turn left" 切到 "turn right"，可以从 forward walk 直接切到 backward walk。**human motion 不能**——必须 decelerate, shift weight, recover balance, 产生短暂 transient motion。这些 "imperfections" 是 embodied action intelligence 的一部分。所以 evaluation 必须在 natural human motion 的物理后果下进行，而不是 idealized instantaneous controller。

但同时 decouple 要求 zero-shot reliable interface：VLM 说 "walk forward"，motion generator 必须生成 commonsense human walk——不能 moonwalk, stagger, 或伴随无关 upper-body motion。否则 failure 没法 attribution 到 decision。

#### 2.2.2 Atomic Skill Set（8 个 primitives）

$$\text{walk}(x, z, \psi),\ \text{side\_step}(x),\ \text{step\_back}(z),\ \text{turn\_in\_place}(\theta)$$
$$\text{climb\_upstairs}(h, d),\ \text{walk\_downstairs}(h, d),\ \text{sit\_in\_place}(h),\ \text{stop}$$

变量含义：
- $(x, z, \psi)$：local walking displacement + heading change（pelvis 在 chunk-local frame 内的位移和 yaw）
- $\theta$：signed turning angle
- $(h, d)$：stair height 和 depth
- $h$（在 sit 里）：target sitting height（pelvis 高度）

**关键设计哲学：**作者**故意不**暴露 `sit(sofa at (x,z))` 这种 high-level composite command。composition（找沙发→接近→转身→坐下）留给 VLM reasoning，每个 primitive 单独可靠。

#### 2.2.3 Motion Base DiT

backbone 用 receding-horizon continuation（参考 Zhang et al. 2025c, PRIMAL, ICCV 2025: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_PRIMAL_Physically_Reactive_and_Interactive_Motor_Model_for_Avatar_Learning_ICCV_2025_paper.html）：

- 输入：last 5 clean body states $x_{t-4:t}$（0.167s @ 30 fps）
- 输出：next 15 frames @ 30 fps = 0.5s motion chunk
- skill-agnostic：只学 local natural human dynamics

公式 (3)：history 直接 concat 到 noisy future 作为 motion tokens

$$\boldsymbol{u}_\tau = [x_{t-4:t},\ \tilde{x}_{t+1:t+15}(\tau)] \in \mathbb{R}^{20 \times d},\quad d = 219 \tag{3}$$

- $x_{t-4:t}$：过去 5 帧 clean body state（每帧 219 维，应该是 SMPL-X pose 表示，包含 joint rotations + root + hand + face 等）
- $\tilde{x}_{t+1:t+15}(\tau)$：未来 15 帧的 noised state，依赖 flow time $\tau \in [0,1]$（$\tau=0$ 纯噪声，$\tau=1$ clean future）
- $d=219$：每帧 pose dimension
- 总 20 个 token 喂给 38M-param, 10-layer DiT, 8-head self-attention
- flow time $\tau$ 通过 sinusoidal MLP + adaLN-Zero 注入到每个 DiT block

**关键设计：history-as-token conditioning**——不把 history 压成单个 condition vector，而是直接 concat，让 future token 直接 attend 到 exact recent pose 和 velocity。这保留 boundary information，kinematic continuity 显式化。

公式 (4)：网络预测所有 token 的 velocity field，但只保留 future token 输出

$$\hat{v}_\theta = G_\theta([x_{t-4:t},\ \tilde{x}_{t+1:t+15}(\tau)],\ \tau)_{t+1:t+15} \tag{4}$$

- $\hat{v}_\theta$：predicted velocity field（flow matching 里 $dx/d\tau$）
- $G_\theta$：DiT network
- 下标 $t+1:t+15$：只保留 future token 的输出，丢掉 history token 的预测

公式 (5)：flow matching 训练目标

$$\tilde{x}_{t+1:t+15}(\tau) = (1 - (1-\sigma_{min})\tau)\epsilon + \tau x^{fut},\quad v^\star = x^{fut} - (1-\sigma_{min})\epsilon \tag{5}$$

- $\epsilon \sim \mathcal{N}(0, I)$：Gaussian noise
- $\sigma_{min}$：最小 noise level（防止 $\tau=1$ 时 zero noise 导致 numerical issue，典型 1e-4）
- $\tilde{x}$：noise $\epsilon$ 和 clean future $x^{fut}$ 之间线性插值
- $v^\star$：target velocity field（监督信号）
- loss：$\|\hat{v}_\theta - v^\star\|_2^2$ on future frames

推理：从纯噪声开始，30-step midpoint solver 解 ODE。

训练数据：AMASS（https://amass.is.tue.mpg.de/）+ BABEL（https://babel.is.tue.mpg.de/）预处理成 20-frame chunks @ 30Hz。每 segment 先 calibrate 到 chunk-local frame（origin 和 heading 由 initial body state 定义），移除 global translation 和 yaw——所以 base DiT 学 local dynamics 而非 absolute world motion。

#### 2.2.4 Plug-and-Play Skills as ControlNet

base DiT 提供了 kinematic continuity prior。下一步是 align 到 zero-shot skill interface。用 **per-skill ControlNet adapter**，base DiT frozen。

公式 (6)：layer $\ell$ 处的组合

$$z_{\ell+1} = B_\ell(z_\ell, \tau) + W_\ell\ C_\ell(z_\ell + e_s(c_t), \tau) \tag{6}$$

- $B_\ell$：frozen base block（提供 motion prior）
- $C_\ell$：trainable control block（copy 自 base DiT 对应 block）
- $e_s(c_t)$：skill-condition encoder。scalar condition（如 sitting height）用 MLP；spatial condition（如 walking displacement, yaw）用 Fourier features 后再 projection 到 DiT hidden dim
- $c_t$：skill 的 continuous parameter
- $W_\ell$：**zero-initialized** residual projection（ControlNet 核心技巧——训练初始时不影响 base branch，渐进学习）
- $\tau$：flow time，通过 adaLN-Zero 注入

**training data trick（很关键）：** skill 的 parameter 直接从 motion geometry 读出来——walking 的 condition 是 chunk-local final pelvis displacement 和 yaw change，sitting 是 target pelvis height，stairs 是 step height/depth。**所以训练不需要人工标注 action parameters**，只需 curate/filter AMASS clips 让每个 ControlNet 单 purpose + commonsense-aligned。clips with abnormal style 或 mixed intent（unstable walking, walking 同时做无关 upper-body action, 爬楼梯时 excessive body twisting）被 manual review remove。这种 unsupervised parameterization 让 VLM 调用 `walk` 时 adapter 真的生成 ordinary walking，而非 idiosyncratic AMASS motion。

每个 adapter 用同一个 flow matching objective 训练：AdamW, lr $3 \times 10^{-4}$, batch 2048, $5 \times 10^5$ 到 $1.5 \times 10^6$ optimization steps（依 skill 而定）。**不用 classifier guidance 也不用 CFG**——skill control 直接由 selected adapter + deterministic parameter interface 提供。

#### 2.2.5 Skill Fidelity（Table 1，zero-shot reliability 验证）

对比 baseline MoMask（Guo et al. 2024, https://arxiv.org/abs/2401.11474）：

| Method | walk | side_step | step_back | turn | climb_up | climb_down | sit |
|---|---|---|---|---|---|---|---|
| **Ours** | 0.966±0.022 | 1.002±0.123 | 0.986±0.056 | 0.994±0.038 | 0.794±0.019 | 0.738±0.052 | 0.977±0.078 |
| MoMask | 1.921±1.031 | 1.717±1.234 | 2.072±1.067 | 0.608±0.284 | 0.820±0.826 | 1.540±0.421 | 1.817±1.279 |

Achievement ratio = achieved / commanded。Ours 几乎全部 ≈ 1，std ≤ 0.12。MoMask 严重偏离 1，std 大一个数量级——即使 prompt 里写明 "walks forward at 0.8 m/s"，MoMask 还是生成 habitual motion，完全 ignore requested magnitude。**这个 fidelity 才让 benchmark failure 可以 attribution 到 decision。**

---

### 2.3 Locomotion-Decoupled Physical Simulation

**痛点：** pure kinematic playback 是 "ghost"——穿墙、空中爬楼梯；fully dynamic torque controller 把 action reasoning 和 locomotor control 纠缠，humanoid 在 cluttered 环境或爬楼梯时容易 lose balance fall，prematurely 终止 episode。

**Half-physics**（Siyao et al. 2025, https://arxiv.org/abs/2507.23778）：world 是 rigid-body physics（collision, friction, gravity, articulated objects, movable objects）；但 humanoid 不用 simulated joint torques 驱动，用 motion sequence 推导的 **equivalent kinematic velocities** 驱动。

公式：

$$\dot{q}_t = \frac{q_{t+1} - q_t}{\Delta t}$$

- $q_t, q_{t+1}$：相邻两帧的 joint configuration（SMPL-X pose 或类似）
- $\Delta t$：simulation step size
- $\dot{q}_t$：linear joint velocity；rotational component 用 tangent space relative rotation 计算

**核心：** humanoid 不会因 lose balance fall，但仍 physically interact——wall 会 block body, contact 会 push movable objects, unsupported vertical motion 受 gravity 约束。

实现：AI Habitat（Savva et al. 2019, https://arxiv.org/abs/1904.01201）+ Bullet（Coumans 2015, https://pytorch.org/），120 Hz, passive joint stiffness $\lambda = 1.0$（保持 compliant contact response）。

**intuition：** 这相当于给 humanoid 一个 "kinematic ghost body"——body 有 collision shape，有 mass（用于 contact response），遵循 gravity；但不会因 motor control 失误摔跤。它在墙上停下，撞到花瓶推倒，但 fail attribution 始终在 decision level。

---

## 3. HumanCLAW-Bench 设计

### 3.1 Task

**find-navigate-interact**：任务指令 "find a `<obj>`, navigate to it with zero distance, and finally sit on it"。progressive：每个 stage 必须完成前置才能进入下个。

数据集：HSSD（Khanna et al. 2024, https://arxiv.org/abs/2307.03584）的 41 validation houses，1218 episodes，6 target categories：
- **Sit subset（597 episodes）**：bed, couch, toilet — full find-nav-interact progression
- **Navigation subset（621 episodes）**：chair, potted_plant, tv — 终止于 navigation

movable objects（tabletop items, small furniture）设为 dynamic，contact 可 displace；large fixtures（beds, sofas, toilets, walls）保持 static。

### 3.2 Difficulty 分层

每 episode 沿三个 geometric 维度评分：

1. **Distance**：到 nearest same-category goal 的 geodesic distance。用 A* 在 per-floor 5cm 2D occupancy grid 上跑 shortest path，median ratio 0.99 vs ground truth，路径 faithful。
2. **Choice**：navigational branchiness = $>30°$ turns on Douglas-Peucker-simplified path + rooms traversed
3. **Obstacle**：route 1m 内 obstacle 数

Thresholds：distance ≤3.5/≤8.1m, choice ≤2/≤5, obstacle ≤2.5/≤5
分布：278 easy / 566 medium / 374 hard（约 23/46/31%，不强制 equal thirds）

### 3.3 Metrics（四个维度）

**Progressive Success Rate**（objective + subjective 同时满足）：
- **FindSR**：target semantic ID 在 ego-view 出现 ≥100 px in 512×512 + model 在 visible-state text 里 acknowledge
- **NavSR**：agent pelvis 到 target AABB 的 min distance ≤20cm + agent 主动 stop
- **InteractSR**：sit action 后 pelvis link（hip region）contact target mesh + agent 主动 stop
- **Geo- prefixed variants**：只看 objective criterion，不看 model acknowledgment——隔离 perception 和 decision

**Action Quality**：Motion Jerk = root-rigid jerk（kinematic motion 三阶导数），at decision timescale（stride 8 ≈ 0.27s）。lower = smoother。
- single skill：4–6（forward walk 5.2, turn 4.3）
- purposeful walk-turn-walk：5.8
- aimlessly sequenced policy：9.7
- 5 forward-walk chunks 然后 5 lateral steps：4.7；interleaving 同样 10 chunks：7.4（purely from ordering）

**Body Awareness & Scene Disturb**：
- Coll. = collision step fraction per episode（floor-relative non-ground collision）
- #Dtb/ep = disturbed movable objects per episode
- dDtb = average displacement distance

**Cost**：avg tokens per episode（input + output）

---

## 4. 实验结果

### 4.1 主表（Table 2）

9 个 VLM（注：paper 标注 date Aug 5, 2026，模型版本号是 future 假设）：GPT-5.5, Gemini-3.1, Gemini-2.5, Claude-4.8, Gemma-4-31B, Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.5-27B, InternVL3.5-38B

**Finding 1：没有 VLM 解决 benchmark**

最强 Gemini-3.1：FindSR 64.9%, NavSR 42.4%, InteractSR **16.8%**。四个 model InteractSR ≤ 0.2%。stage-by-stage drop 锐利：

| VLM | FindSR | NavSR | InteractSR | Coll. | #Dtb | dDtb | Motion Jerk | avg steps | in tok/step | out tok/step |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-5.5 | 55.1% | 13.9% | 3.4% | 43.4% | 1.53 | 1.56 | **4.2** | 82.4 | 4360 | 354 |
| **Gemini-3.1** | **64.9%** | **42.4%** | **16.8%** | 39.5% | 1.50 | 1.22 | 5.7 | 59.1 | 5890 | 311 |
| Gemini-2.5 | 58.5% | 21.6% | 3.5% | 37.7% | 1.35 | 3.03 | 8.7 | 71.3 | 6412 | 401 |
| Claude-4.8 | 32.6% | 8.6% | 1.5% | 44.2% | 1.73 | 2.76 | 5.3 | 81.0 | **7047** | **625** |
| Gemma-4-31B | 58.1% | 28.7% | 11.1% | 40.1% | 1.66 | 1.86 | 4.8 | 78.5 | 4632 | 322 |
| Qwen3.6-27B | 51.0% | 20.9% | 0.2% | 43.0% | 1.95 | 2.16 | 6.5 | 81.6 | 4862 | 482 |
| Qwen3.6-35B-A3B | 44.6% | 5.8% | 0.0% | 34.6% | 1.45 | 2.38 | 7.4 | 74.0 | 4766 | 477 |
| Qwen3.5-27B | 37.8% | 13.5% | 0.0% | 34.5% | **0.93** | 2.38 | 6.8 | **37.6** | 4641 | 473 |
| InternVL3.5-38B | 46.8% | 0.8% | 0.0% | 51.2% | 1.90 | 2.56 | 7.2 | 93.0 | 4459 | 309 |

**landing rate**（NavSR/FindSR）告诉你"找到的"有多少"走到"：Gemini-3.1 0.65, Gemma-4-31B 0.49, InternVL3.5-38B 0.02。识别目标普遍容易，把识别转化成 body stop next to target 是分水岭。

几个有意思的反直觉点：
- **cost 不等于 performance**：Claude-4.8 用最多 input/output tokens per step（7047/625），trailing on 所有 success metric；Gemini-3.1 用更少 token 领先。
- **low disturbance 不等于 control**：Qwen3.5-27B 最少 disturb（0.93 obj/ep），最短 episode（37.6 steps），但 0% interact——inaction 不是 control；InternVL3.5-38B 和 Claude-4.8 collision 最高（51.2%, 44.2%）+ NavSR 崩——moves without control。
- **action quality decoupled from success**：Gemini-2.5 success 不错但 Motion Jerk 最差（8.7），每 5 步一次 turn-reversal 摇摆；GPT-5.5 最 coherent（4.2）但 success 不强。
- **open model 接近 frontier**：Gemma-4-31B（58.1/28.7/11.1）在三个 success rate 上都打平或超过 GPT-5.5 和 Claude-4.8，仅次于 Gemini-3.1。Progress open to community。

### 4.2 Ablation（Table 3，100-episode mini-val, Gemma-4-31B backbone）

**Finding 2：reasoning scaffolds（memory, mid-level, verifier）lift action intelligence；longer text/visual history 不一定 helpful**

| Setting | FindSR | NavSR | InteractSR | Coll. | #Dtb | dDtb | Jerk | steps | in tok | out tok |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (hist 10, img 1) | 58.0% | 27.0% | 18.9% | 41.4% | 1.35 | 9.38 | 4.9 | 78.5 | 4676 | 322 |
| hist 0 | 65.0% | 11.0% | 0.0% | 38.6% | 1.27 | 6.46 | 5.1 | 88.9 | 2582 | 292 |
| hist 20 | 59.0% | 27.0% | 7.5% | 42.9% | 1.18 | 1.56 | 4.8 | 80.3 | 6692 | 326 |
| hist 50 | 60.0% | 28.0% | 7.5% | 40.4% | 1.24 | 6.16 | 4.8 | 77.6 | 10538 | 320 |
| hist 100 | 56.0% | 26.0% | 11.3% | 37.6% | 1.08 | 7.23 | 4.7 | 79.7 | 12904 | 318 |
| img 2 | 57.0% | 31.0% | 9.4% | 43.0% | 1.19 | 2.55 | 4.7 | 78.8 | 5234 | 321 |
| img 5 | 55.0% | 28.0% | 7.5% | 41.1% | 1.30 | 8.86 | 5.1 | 82.7 | 6335 | 324 |
| img 10 | 53.0% | 13.0% | 3.8% | 34.9% | 0.95 | 3.19 | 5.2 | 92.3 | 7685 | 303 |
| **w/o verifier** | 51.0% | **2.0%** | **0.0%** | 33.2% | 0.71 | 10.40 | 4.8 | 47.5 | 4095 | 247 |
| **w/o mid-level** | 62.0% | 30.0% | **0.0%** | 35.4% | 1.22 | 2.08 | 5.0 | 70.3 | 3769 | 264 |

**Verifier 是 decisive component：** 去掉后 NavSR 27→2%, InteractSR 18.9→0%，FindSR 几乎不变（58→51）。verifier 影响 acting 不影响 seeing。episode 也明显更短（47.5 vs 78.5 steps），agent 在 loop 关闭前 stop 或 drift。verifier 是把 skill proposal stream 转成 closed loop reaches & commits 的 anchor。

**Text history 必要但 saturate 快：** hist 0 → NavSR 11%, InteractSR 0%；hist 10/20/50/100 都在 26-28% NavSR；input token 从 4.7K 涨到 12.9K。compact history captures nearly all benefit at fraction of cost。这跟你之前关于 long-context degradation 的直觉一致——长 history 在 VLA-style closed loop 里容易 dilute signal。

**More image frames can hurt：** img 2 neutral, img 5 大致 neutral, img 10 → NavSR 27→13%, InteractSR 18.9→3.8%。decision model 被 current view 的 reasoning bottleneck，flood 它 past frames dilute signal。

**Mid-level reasoning carries long-horizon interaction：** 去掉 mid-level objective，FindSR/NavSR 几乎不变（甚至微升），但 InteractSR **18.9→0%**。原因：sit 是 short compositional routine——reach seat → turn align → sit。several-second sequence 没法靠 step-by-step reactive policy hold together。这和 hierarchical RL option framework（Sutton & Precup 1999）思想相通，但 implementation 是 VLM 显式 reasoning。

### 4.3 Geo- vs Acknowledged Variants（Table 4）

| VLM | FindSR | GeoFindSR | NavSR@20cm | GeoNavSR@20cm | NavSR@1m | InteractSR | GeoInteractSR |
|---|---|---|---|---|---|---|---|
| GPT-5.5 | 55.1% | 63.2% | 13.9% | 27.7% | 15.5% | 3.4% | 9.7% |
| Gemini-3.1 | 64.9% | 69.9% | 42.4% | 44.7% | 52.5% | 16.8% | 21.4% |
| Gemini-2.5 | 58.5% | 66.4% | 21.6% | 23.6% | 31.0% | 3.5% | 4.5% |
| Claude-4.8 | 32.6% | 55.7% | 8.6% | 13.5% | 11.4% | 1.5% | 2.2% |
| Gemma-4-31B | 58.1% | 67.7% | 28.7% | 35.6% | 33.4% | 11.1% | 17.9% |
| Qwen3.6-27B | 51.0% | 62.8% | 20.9% | 34.8% | 24.2% | 0.2% | 0.3% |
| Qwen3.6-35B-A3B | 44.6% | 59.0% | 5.8% | 14.0% | 11.4% | 0.0% | 0.0% |
| Qwen3.5-27B | 37.8% | 48.9% | 13.5% | 17.7% | 19.2% | 0.0% | 0.0% |
| InternVL3.5-38B | 46.8% | 59.1% | 0.8% | 12.6% | 1.6% | 0.0% | 0.0% |

---

## 5. Error Analysis 四大 Findings

### Finding 3：Perception matters but not the bottleneck

GeoFindSR vs FindSR（acknowledged）gap 很小：Gemini-3.1 69.9 vs 64.9（5pt），GPT-5.5 63.2 vs 55.1（8pt），Gemma-4-31B 67.7 vs 58.1（10pt）。**一旦 target 真的 rendered 进 ego view，几乎一定被 recognize——recognition 是 largely solved。**

Find 失败 root cause 分布（图 8b）：
- 38% ineffective exploration（incoherent search, never render target）
- 23% genuine perception lapse（target rendered but unacknowledged）
- 12% ineffective actions
- 10% approaching without ever turning to view
- 9% unknowingly jammed against geometry
- 6% abandoning too-far goal
- 3% midpoint give up

**bottleneck 在 where & how agent 选择 explore。**

### Finding 4：Egocentric self-localization 是 navigation bottleneck

5473 episodes 中 agent 主动找到 target（≥100 semantic px + acknowledged），3706（68%）仍 fail navigation。

失败 root cause 分布（图 8c）：
- **34% body awareness**：20% "arrived <0.2m 但没 signal arrival, 继续 navigate"；14% "jammed against obstacle 连续多步但继续 forward commands"
- **30% ego-spatial distance hallucination**：agent stop + declare arrival 但 still >0.2m away，相信 distant target 在 reach
- 17% mid-level reasoning（turned away → lost target → cannot re-acquire, 16%；found too late, <1%）
- 10% pure action/locomotion（free move 但 never converge）
- 9% visual perception（target left ego view 但 model 仍 claim see）
- <1% discovery timing

**~2/3 failures 是 egocentric self-spatial awareness 错误**，perception 和 low-level control 都不是主因。

### Finding 5：Reaching is not interacting

726 episodes 成功 navigate（active stop <0.2m），513（71%）仍 fail sit。

失败 root cause（图 8d）：
- **58% sit into thin air**：issue sit 但 pelvis 从未 contact 任何 surface——body 在站立位置直接 lower，下面没有 seat
- **14% sit on wrong thing**：pelvis 落在不同 object / floor / wall（goal 是 couch/bed/toilet 时最容易，邻居 invite 错位 landing）
- **9% stand after sit**：pelvis 短暂 contact seat 但继续 act（mostly turn，偶尔 stand + re-sit），最终 shift off seat
- **19% decision failure**：reach + stop 但 never issue sit

**81% 是 body awareness 错误**（put body in wrong place），19% 是 decision 错误。

### Finding 6（隐含）：collisions 集中在看不见的部位

Table 5：

| VLM | Coll.% | Arm/Hand | Torso | Leg/Foot | Head |
|---|---|---|---|---|---|
| GPT-5.5 | 43.37 | 23.24 | 13.78 | 38.83 | 2.99 |
| Gemini-3.1 | 39.48 | 20.97 | 11.72 | 33.29 | 2.62 |
| Gemini-2.5 | 37.69 | 22.70 | 13.63 | 30.87 | 2.94 |
| Claude-4.8 | 44.18 | 28.97 | 16.90 | 37.94 | 6.93 |
| Gemma-4-31B | 40.09 | 21.21 | 11.38 | 34.72 | 2.66 |
| Qwen3.6-27B | 42.97 | 25.48 | 12.70 | 36.86 | 3.54 |
| Qwen3.6-35B-A3B | 34.58 | 21.20 | 11.40 | 27.99 | 3.71 |
| Qwen3.5-27B | 34.53 | 20.59 | 10.93 | 29.58 | 2.87 |
| InternVL3.5-38B | 51.17 | 34.68 | 23.12 | 44.75 | 6.10 |

Leg/Foot collide 最多（28-45% steps），Arm/Hand 第二（20-35%），Head 最少（<7%）。**body 的盲区 = VLM 不看的部位。** Figure 9 展示两个 concrete 失败：(a) arm 几帧前明明在 view 里能看见椅子，撞倒了也没 register contact；(b) obstacle 直接在前方可见，agent 仍持续 walk legs into it。

---

## 6. Discussion + 我的 interpretation

### 6.1 paper 的 philosophical commitment

作者在 Discussion 里 explicit state："generalizable action intelligence arises from reasoning, not from fitting action data"。policy fit on trajectories 只能在 data support 内 generalize；reasoner 能 extrapolate as far as knowledge allows。HumanCLAW 建立在这个 bet 上——decision maker frozen + general, motor layer fixed + reusable, 所有 task-level composition zero-shot 在 reasoning 里完成。在这个 view 下，agent 改进靠 become better reasoner，不靠 collect more trajectories。新 skill register 不需 retrain，新 task 只需 new rules，foundation model 进步 transfer to body for free。

### 6.2 "VLM = ghost" 的核心隐喻

paper 的核心隐喻：**当前 VLM 像 ghost——fluent about the world, oblivious to itself。** 它把 limbs 当作 just more pixels，没有 proprioceptive, consequence-predicting model。原因：VLM 训练自 visual question answering，学 describe what it sees，没学 feel what it does。

closing gap 需要：
- persistent spatial memory
- calibrated termination（知道何时 arrived）
- internal model of body（with spatial relations each action produces）

paper 承认：当前 interface 没有 tactile channel，collision displace world 但不被 felt。所以 agent 的 proprioception 只能从 egocentric view reconstruct——可能 intrinsically difficult。body-state 或 contact signal 可能是 missing input，而不仅仅是 missing faculty。**这是一个 honest open question：到底是 VLM 需要 learn feel body，还是 interface 需要补 tactile channel。**

### 6.3 我觉得 Karpathy 可能会关心的联想

**1. 这和 VLA 路线的对比：** π0（https://arxiv.org/abs/2410.24164）, OpenVLA（https://arxiv.org/abs/2406.09246）, RT-2（https://arxiv.org/abs/2307.15818）, GR00T N1（https://arxiv.org/abs/2503.14734）这些 end-to-end VLA 把 reasoning 和 motor grounding 绑一起 train，entangle。HumanCLAW 选 inverse 路线——reasoning frozen, motor layer reusable。两条路线在哲学上是相反 bet：VLA bet on data scaling 让 reasoning 从 action data 里 emerge；HumanCLAW bet on reasoning 在 frozen generalist 里已有，只是没被 body-aware test 到。这其实是 pre-transformer 时代 specialized policy vs after GPT-3 generalist reasoner 之争的 embodied 版本。

**2. verifier 是 closed-loop anchor：** ablation 显示去掉 verifier → NavSR 27→2%, InteractSR 18.9→0%。这防止 long-context degradation 下 hallucinated progress。这和你 nanoGPT 时代关注的 "model doesn't track state over long context" 是同一 issue。verifier 是 short-context reset，强迫每步重新 ground 到当前 ego-view。**这是 RL 里 value-based reset 的 prompt-engineering 对应物**。

**3. mid-level reasoning 是 medium-horizon glue：** 去掉 mid-level → FindSR/NavSR 不变，InteractSR 18.9→0%。reactive step-by-step 不够，需要 medium-horizon commitment。这和 hierarchical RL 的 option framework（Sutton, Precup 1999, https://scholarpedia.org/article/Options）思想相通——但 implementation 是 VLM 显式 reasoning 而非 learned option policy。

**4. embodied self-awareness 是 next frontier：** spatial intelligence literature（SpatialVLM https://arxiv.org/abs/2401.12168, Thinking in Space https://arxiv.org/abs/2412.14171, ESI-Bench https://arxiv.org/abs/2605.18746）已经发现 VLM spatial reasoning 弱。但 HumanCLAW 进一步——在 closed loop + physical body 下，spatial reasoning 的 bottleneck 在 **self-localization + body-state tracking**，perception largely solved。这其实是 cognitive science 里 "body schema" / proprioception 的对应——VLM 缺一种 implicit body model。这让人想到前几年 robot learning 里流行的 "implicit forward model" 概念（learning to predict next state from action），但在 VLM scale 上还没人 systematic 做。

**5. 接口设计哲学——atomic skill + continuous parameter：** 比 discrete action space 更 expressive，比 raw 100+ DoF trajectory 更 interpretable。每个 skill 是 minimal unambiguous unit，composition 留给 VLM。这让人想起 hierarchical control 的经典思路（Brooks subsumption architecture, behavior-based robotics, https://en.wikipedia.org/wiki/Subsumption_architecture），但 high-level 换成 foundation model。这其实是个 generalizable pattern：**给 generalist reasoner 一个 minimal reliable motor interface，让 composition emerge from reasoning**——你同样可以套到 tool use, code execution, GUI 操作。

**6. half-physics 是 evaluation 哲学：** fully physics 让 evaluation dominated by locomotion；pure kinematics 让 agent 是 ghost。half-physics 是 controlled 中间状态——保留 physical consequence，剥离 motor failure。这是 robotics eval 的 generalizable 思路：**你想 measure A 但不想被 B confound，那就 simulated 掉 B 但保留 A 的真实 consequence。** 这个 idea 可以推广到 manipulation——比如 grasp 不成功可能是 perception issue 或 force control issue，decouple 之后才能 attribute。

**7. ghost metaphor 让我想起 mirror self-recognition test：** classic 的 mirror test（Gallup 1970）测 animal 能不能 recognize 自己。VLM 现在 fail 的方式很有意思——它能看到 scene，但 scene 里如果有自己身体的部位（比如 arm 入镜），它不会把那些 pixel 当作 "我的"。这其实是视觉 self-recognition 的失败。**也许应该给 VLM 加一个 self-segmentation 的 auxiliary task，让它学会 "这些 pixel 是我"。**

**8. 数据效率 vs generalization 的 tradeoff：** HumanCLAW 强调 zero-shot transfer，但承认 16.8% success。VLA 路线在特定 embodiment 上可能更高 success 但 zero-shot 差。这个 tradeoff 在 RL 里经典——model-based vs model-free, generalist vs specialist。HumanCLAW 把这个 tradeoff 拉到一个新维度：**reasoning-based action vs trajectory-fit action**。

**9. motion generator 本身很有意思：** Motion Base DiT + per-skill ControlNet 是个 plug-and-play 设计，新 skill 加一个 adapter 不动其他。这和 LoRA（https://arxiv.org/abs/2106.09685）、ControlNet（https://arxiv.org/abs/2302.05543）的 modular 思路一脉相承。**zero-initialized $W_\ell$ 让 adapter 渐进学习是 ControlNet 的核心技巧。** 用 motion geometry 自动 derive condition 是很聪明的 unsupervised parameterization。

**10. failure attribution 是 benchmark 设计的核心：** 整个 paper 的方法论核心是 "decouple to attribute"。这其实是 causal inference 在 embodied eval 里的应用——你想知道 cause 是 A 还是 B，就 controlled 掉 B 看 A 单独的表现。**这个思路应该在更多 embodied benchmark 里推广**，比如 manipulation——decouple perception issue 和 force control issue。

---

## 7. 相关 web links（参考与延伸阅读）

**paper & project：**
- HumanCLAW project page: https://human-claw.github.io
- Half-physics (Siyao et al. 2025): https://arxiv.org/abs/2507.23778
- PRIMAL (Zhang et al. 2025c, ICCV 2025): https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_PRIMAL_Physically_Reactive_and_Interactive_Motor_Model_for_Avatar_Learning_ICCV_2025_paper.html

**仿真环境 & 数据集：**
- AI Habitat (Savva et al. 2019): https://arxiv.org/abs/1904.01201
- Habitat 2.0 (Szot et al. 2021): https://proceedings.neurips.cc/paper/2021/hash/021bbc7ee20b71134d53e20206bd6feb-Abstract.html
- Habitat 3.0 (Puig et al. 2024): https://openreview.net/forum?id=4znwzG92CE
- HSSD (Khanna et al. 2024): https://arxiv.org/abs/2307.03584
- AMASS: https://amass.is.tue.mpg.de/
- BABEL: https://babel.is.tue.mpg.de/
- Bullet physics: https://pybullet.org/

**对比 benchmark & 任务：**
- ALFRED (Shridhar et al. 2020): https://arxiv.org/abs/1912.01734
- VirtualHome (Puig et al. 2018): https://arxiv.org/abs/1803.05637
- BEHAVIOR-1K (Li et al. 2023): https://proceedings.mlr.press/v205/li23a.html
- PARTNR (Chang et al. 2025): https://arxiv.org/abs/2412.16115
- LoTa-Bench (Choi et al. 2024): https://openreview.net/forum?id=ADSxCpCu9s
- EmbodiedBench (Yang et al. 2025): https://arxiv.org/abs/2401.13953
- ESI-Bench (Hong et al. 2026): https://arxiv.org/abs/2605.18746
- VisualAgentBench (Liu et al. 2024): https://arxiv.org/abs/2408.06327
- VideoGameBench (Zhang et al. 2025a): https://arxiv.org/abs/2505.18134
- Voyager (Wang et al. 2024): https://arxiv.org/abs/2305.16291
- Cradle (Tan et al. 2024): https://arxiv.org/abs/2403.03186

**VLA & end-to-end control 对比：**
- RT-2 (Zitkovich et al. 2023): https://arxiv.org/abs/2307.15818
- OpenVLA (Kim et al. 2025): https://arxiv.org/abs/2406.09246
- π0 (Black et al. 2024): https://arxiv.org/abs/2410.24164
- GR00T N1 (NVIDIA et al. 2025): https://arxiv.org/abs/2503.14734
- PaLM-E (Driess et al. 2023): https://arxiv.org/abs/2303.04271
- SayCan / Inner Monologue (Ichter et al. 2023, Huang et al. 2023b): https://arxiv.org/abs/2204.01691, https://proceedings.mlr.press/v205/huang23c.html
- Code as Policies (Liang et al. 2023): https://arxiv.org/abs/2209.07753
- VoxPoser (Huang et al. 2023a): https://arxiv.org/abs/2305.18609
- ManipulateAnything (Duan et al. 2024): https://arxiv.org/abs/2403.02839

**spatial intelligence & VLM reasoning：**
- SpatialVLM (Chen et al. 2024): https://arxiv.org/abs/2401.12168
- Thinking in Space (Yang et al. 2024): https://arxiv.org/abs/2412.14171
- Unfolding spatial cognition (Li et al. 2025): https://arxiv.org/abs/2506.04633

**controllable human motion generation：**
- MoMask (Guo et al. 2024): https://arxiv.org/abs/2401.11474
- MDM (Tevet et al. 2023, ICLR 2023): https://arxiv.org/abs/2209.14991
- OmniControl (Xie et al. 2024, ICLR 2024): https://arxiv.org/abs/2310.08580
- PhysDif (Yuan et al. 2023): https://arxiv.org/abs/2310.02955
- Bailando (Siyao et al. 2022): https://arxiv.org/abs/2208.10635
- SAMI / scene-aware motion (Hassan et al. 2021): https://arxiv.org/abs/2104.03580 (ICCV 2021)
- ControlNet (Zhang et al. 2023, ICCV 2023): https://arxiv.org/abs/2302.05543
- LoRA (Hu et al. 2021): https://arxiv.org/abs/2106.09685

**additional theory：**
- Options framework (Sutton, Precup 1999): https://scholarpedia.org/article/Options
- Subsumption architecture (Brooks 1986): https://people.csail.mit.edu/brooks/papers/AIM-864.pdf
- Gallup mirror test (Gallup 1970): https://doi.org/10.1126/science.167.3914.86

---

paper 的核心 message 我重新凝练一下：**它给 "VLM 能不能 act through a body" 这个问题做了一个干净的实验——把 motor execution factor out，让所有 failure attribution 到 decision level。结果发现 reasoning 本身的瓶颈是 embodied self-awareness，perception 已基本 solved。VLM 在 closed loop + physical body 下表现为 ghost——描述世界流畅，对自身身体无感。** 这给你两个 actionable direction：(1) 让 VLM 学 implicit body model / proprioceptive prediction；(2) 在 interface 层补 tactile channel，看 missing input 还是 missing faculty。这两个 hypothesis paper 没 disambiguate，是个干净的下一步。
