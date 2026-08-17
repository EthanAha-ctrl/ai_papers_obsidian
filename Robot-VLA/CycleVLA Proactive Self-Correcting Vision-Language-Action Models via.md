---
source_pdf: CycleVLA Proactive Self-Correcting Vision-Language-Action Models via.pdf
paper_sha256: 1998a0ac1d5f1942eef844e29a933fa2885ceaa48f5e253701fbbe1c0315e390
processed_at: '2026-08-03T18:11:51-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# CycleVLA 人话版

## 一句话总结

机器人执行任务时,错误几乎都发生在"步骤切换"的瞬间——把杯子拿起来准备放下的那一刻,把螺丝对准孔准备拧入的那一刻。CycleVLA 的做法就是:在快切换的时候让 VLM 看一眼,觉得要出事就退回去重做,重做的时候多采样几次选最"主流"的轨迹。

## 核心直觉

你握一个滑溜的杯子,手指会在杯子掉之前就感觉到 slip 然后 tighten。你不会等杯子碎了才说"哦,刚才握得不够紧"。这就是 **proactive correction**——失败还没完全发生就介入。

之前 robot failure correction 的工作基本是 **post hoc**:任务执行完或者执行到炸了,再看哪里出错,用 residual policy 补救。这就好比杯子已经碎了,你再反思"当时应该握紧点",对当下这个 episode 没用。

CycleVLA 想把这种"提前感知"的能力塞进 VLA。它的核心 observation 很朴素:manipulation 任务的失败**高度集中在 subtask 切换点**。peg-in-hole 在 peg 撞到 hole 边缘之前的零点几秒,从 gripper alignment 你已经能看出要失败。这就给了一个窗口——在 transition 之前 intervene。

## 怎么做的

### 1. 让 VLA 知道自己走到哪了

现成的 VLA 有个大问题:它只会一直输出 action,没有"我现在做到第几步了""这一步结束没"的概念。CycleVLA 给 action 加了两个维度:
- `s` = stop signal,这一步 subtask 该结束了吗
- `p` = progress,0 到 1,我在这一步走了多远

这两个跟 action chunk 一起 predict,不加新 head,就 widen 一下 output dimension。训练数据怎么来?用 GPT-4.1 把 task instruction 分解成 atomic subtask,再用 gripper open/close 的状态变化自动对齐 subtask 边界——抓取就是 close,放下就是 open,中间的 idle 就是搬运。这套 pipeline 听起来很 hacky,但 human eval 显示边界误差平均 5.7 步,相对误差 3.8%,够用。

训练时有个小 trick:每个 subtask 的最后一步 oversample 8 倍,因为 stop signal 必须在 termination 准确 fire,但 last step 在 trajectory 里只出现一次,信号太弱。

### 2. 在 progress 快到 1 的时候让 VLM 看一眼

当 $p \geq 0.9$,query GPT-5.2。喂给它两个视角——third-person 看全局(object identity、gripper 跟 target 对齐没),wrist 看细节(contact quality、alignment)。VLM 输出:transit(继续下一步)还是 backtrack(退回去)。

Backtrack 退到哪个 subtask? 退到**最早能恢复 precondition 的那一步**。比如你抓着杯子走到一半杯子掉了,应该退到"抓取杯子"那一步重新抓,不是退到"搬运杯子"那一步——因为 precondition 已经丢了。

退回去靠 reverse-executing recorded delta actions,把机器人状态倒回 target subtask 的起点。

### 3. 重做的时候多采样几次,选最主流的

退回去之后,VLA 重新执行这一步。这里用了一个从机器翻译借来的老 trick: **Minimum Bayes Risk decoding**。

简单说:VLA 采样 N=8 个 action chunk hypothesis(因为 action expert 是 diffusion-based,换 random seed 就能采不同 noise 得不同 sample),然后两两算距离,选那个"到其他所有 hypothesis 平均距离最小"的——也就是最接近 cluster 中心的那个。

为什么 work?因为 imitation learning 训的 VLA,成功行为在 policy 输出空间里 cluster 在高密度区——demo 都是 unimodal 的成功轨迹。所以 consensus selection 命中成功 mode 的概率比 random sample 高。

**MBR 的成本几乎为零**:Table V 显示 MBR 计算 0.003 秒,占总 inference time <0.1%。因为就是在 $\mathbb{R}^{48}$ 上算 8×8 的 pairwise distance,基本免费。这是它相对 learned verifier(Rover、V-GPS 那种)的大优势——不用训 verifier,纯几何 consensus。

## 实验里最有意思的几个点

### Long-horizon 提升最大
LIBERO-Long suite:OpenVLA 53.7 → CycleVLA 93.6,涨 40 个点。这完全符合预期——long task 错误会跨 subtask 累积,在 transition 点干预能截断累积。Short task 错误本来就不多,提升空间小。

### Under-trained VLA 的实验最 compelling
200K steps 的 checkpoint 加 CycleVLA ≈ 350K without it;350K + CycleVLA ≈ 500K without it。这说明 **test-time compute 能 substitute training compute**。跟你 Snell 那篇 "scaling test-time compute can be more effective than scaling parameters" 在 LLM 上得到的结论是呼应的,robotics 这边可能也有类似的 inference-time scaling law。对 compute 受限的 lab 意义很大——训练省下的卡时可以靠 inference 时多采样补回来。

### MBR 对 weak model 帮助更大
200K model 用 MBR 涨 5-9 个点,500K model 只涨 3-5 个点。跟 LLM 那边 MBR 的观察一致:weak model 产生更多 inconsistent candidate,consensus 的 marginal benefit 更大。Strong model 自己就 self-consistent,采样好几个都差不多。

### Sycophancy 的发现很有意思
Ablation 里有个 "predicted failure cutoff" 实验:VLM 一旦预测要失败就立刻 terminate episode 计为失败。结果 SR 掉 10 个点。理论上应该跟 no-correction baseline 持平——因为 VLM 预测失败本来就应该对应真的会失败的情况。这 10% 的 gap 是 VLM sycophancy 造成的:你问它"会不会失败",它倾向于顺着问题回答"会",即使其实不会。这是 LLM-as-judge 在 robotics 场景的一个 specific failure mode,future work 可能需要 contrastive prompting 或 calibration 来 debias。

### Distance metric 的选择
$L_2$ 最好,cosine/correlation 最差。原因:trajectory 里 translation components dense(每步都在动),rotation sparse(很多步几乎不转)。Distance-based metric 捕捉 magnitude 差异,cosine/r 强调方向一致性——对 sparse rotation 不友好。这个 insight 在设计 VLA 的 MBR 时值得记住。

## 跟其他工作的关系

- **vs RoboMonkey**:RoboMonkey 也采样多个 action 但要训一个 VLM selector 来挑。MBR 不用训 verifier,纯 geometric consensus,zero-shot。
- **vs PAINT**:PAINT 做 proactive intervention 但需要人在 loop 里。CycleVLA 用 VLM 替代人。
- **vs Bellman-Guided Retrials**:也做 backtrack + retry,但用 value-based selection 且针对 specific policy。CycleVLA 用 training-free MBR + 针对 generalist VLA foundation model。
- **vs FPC-VLA**:也做 failure prediction + correction,但需要专门训 supervisor module。CycleVLA 用 off-the-shelf GPT-5.2,zero-shot。

## 局限

1. **Backtrack 假设 state 可逆**:reverse-executing delta actions 在 irreversible 环境(易碎物体、不可逆 grasp)会 fail。Real robot 上更严重。
2. **Inference 慢**:test-time scaling 增加 ~30% runtime,contact-rich 任务需要高 control frequency 可能扛不住。
3. **VLM 依赖外部 API**:用 GPT-5.2。Ablation 显示换 LLaMA-3.2-11B 略降但可用,可以本地化。
4. **还没上 real robot**:论文自己承认这是 near future work。

## 我的 take

CycleVLA 的 elegance 在于它没有重新发明轮子——把 MBR decoding(机器翻译 2004 年的老 trick)、backtracking(VLN 领域 2019 年的 trick)、subtask decomposition(manipulation 领域的常见做法),组装成一个 VLA inference pipeline。每个 piece 单独都不 novel,但组合产生 synergistic effect:progress prediction 提供 **when**,VLM 提供 **what + where**,MBR 提供 **how to retry better**。

最关键的 idea shift 是把 failure detection 从 post hoc 推到 proactive——这个 timing shift 是整个 system 价值的 foundation。Under-trained VLA 实验暗示 robotics 也有 inference-time scaling 的空间,跟 LLM 的 test-time compute scaling 呼应。Sycophancy 的发现提醒我们:VLM-as-judge 在 robotics 里有 systematic bias,需要专门设计 debiasing。

---

# CycleVLA: Proactive Self-Correcting VLA 深度解析

## I. 核心Intuition

CycleVLA 的核心 insight 非常 elegant: robot task failures **不是均匀分布** 在 timeline 上, 而是**高度集中在 subtask transition points**。比如 peg-in-hole 任务, peg 撞到 hole 边缘之前的 0.5 秒, 你其实已经能从 gripper alignment 看出要失败。这给了一个 precious window: 在 failure fully manifest 之前 intervene。

这跟人类的 reactive control 是一回事——你握杯子感觉到 slip, 手指会**在杯子掉之前** tighten grip, 而不是等杯子碎了再反思。论文把这种能力叫 **proactive self-correction**, 跟 prior work 的 post hoc correction (REFLECT, AHA, STAR 等) 形成对比。

整个 system 是一个 cycle: **monitor progress → trigger VLM check at τ_p=0.9 → VLM 决定 transit 还是 backtrack → backtrack 到 precondition subtask → 用 MBR 重新 sample → retry**。这个 cycle 重复直到 success 或 max retries (R=3)。

---

## II. Architecture 拆解

### A. Action Space 扩展 (9-dim)

原始 VLA action 是 7-dim delta action:
$$a_t = [\Delta x_t, \Delta y_t, \Delta z_t, \Delta u_t, \Delta v_t, \Delta w_t, \gamma_t]^\top \in \mathbb{R}^7$$

CycleVLA 把它扩展到 9-dim:
$$a_t = [\Delta x_t, \Delta y_t, \Delta z_t, \Delta u_t, \Delta v_t, \Delta w_t, \gamma_t, s_t, p_t]^\top \in \mathbb{R}^9$$

变量含义:
- 下标 $t$ = timestep
- $(\Delta x_t, \Delta y_t, \Delta z_t) \in \mathbb{R}^3$ = end-effector 平移 delta
- $(\Delta u_t, \Delta v_t, \Delta w_t) \in \mathbb{R}^3$ = rotation delta (axis-angle 或 euler)
- $\gamma_t \in \{0, 1\}$ = gripper open/close command
- $s_t \in \{0, 1\}$ = **stop signal**, 表示 subtask termination
- $p_t \in [0, 1]$ = **progress signal**, discretize 成 0.1 bins, 按 normalized timestep within subtask 计算

**关键设计 choice**: $s_t$ 和 $p_t$ 作为 scalar 输出跟 action chunk 一起 predict, **不引入新的 classification head**。作者强调这跟 continuous action space 的 nature 一致——$\gamma_t$ 本身就是 bounded scalar, $s_t$ 和 $p_t$ 也是 bounded, 所以 widen action dim 就够了, no architectural changes。这跟 NaVILA 的设计哲学类似。

**为什么 separate stop 和 progress**: stop 必须 **precise** (要支持正确的 subtask transition), progress 只需要 **approximate** (proximity to completion)。两者语义不同, 合并会损失信息。

### B. Subtask-Decomposed Dataset 构造 Pipeline

这是论文最 engineering-heavy 的部分, 见 Fig. 2。Pipeline 三步:

**Step 1: LLM Subtask Decomposition**
用 GPT-4.1 (temperature=0.2) 把 task instruction $g$ 分解成 atomic subtask sequence $(g_1, \ldots, g_K)$, 限定 verb vocabulary: `move`, `rotate`, `open`, `close`。Prompt 强调 minimal decomposition + object-centric reasoning。

**Step 2: Movement Primitive Extraction**
从 robot proprio 用 sliding window (size=4) 提取 primitive labels:
```
move [forward/backward] [left/right] [up/down]
tilt [up/down]
rotate [clockwise/counterclockwise]
[close/open] gripper
```
对 LIBERO 用的 thresholds:
$$[\tau_{trans}, \tau_{rot}, \tau_{grip}] = [0.02, 0.0075, 0.03]$$

**Per-trajectory translation threshold optimization**: grid search $\tau_{trans} \in [\tau_{trans}^{init} - 0.01, \tau_{trans}^{init} + 0.01]$ (50 steps), minimize:
$$\text{score} = 1.0 \times N_{overlaps} + 2.5 \times N_{stops}$$
overlap 指同时 translation 和 gripper 动, stops 指 spurious "stop" label。权重 2.5 反映 stops 更 undesirable。

**Step 3: Gripper State Segment Alignment**
Gripper state 是最 reliable 的 subtask boundary signal——continuous open/close 对应 grasp/release, idle 对应 pure translation/rotation。

Multi-threshold voting: 用 [0.028, 0.03, 0.032] 三个 grip threshold 各跑一次, average + round 得到 final label $\in \{-1, 0, +1\}$ (close/idle/open)。

Post-filter: 对 idle segment of length $L$, 如果 $L_{left} + L_{right} > L$ (两边 consistent gripper actions 比 idle 长), 替换 idle 为 surrounding value。这清理 spurious idle。

**Step 4: Subtask-Timestamp Alignment**
- 如果 LLM proposed subtasks 数量 == gripper segments 数量: 直接 pair
- 否则: 把 movement primitive sequence downsample (max 100 steps, stride $\lceil T/100 \rceil$) 喂给 LLM, 让它 infer boundaries, 强制 continuous assignment without gaps

**Human evaluation** (Table VII): average absolute error 5.7 steps, relative error 3.8%。LIBERO-Goal 最难 (8.0 steps error), 因为 gripper boundary 不明显 + rotation 多。

### C. Training: Last-Action Oversampling

Following NaVILA, 每个 subtask 的 **last action step** oversample 8×。原因: stop signal 必须在 termination step 准确 fire, 但 last step 在 trajectory 里只有 1 个 sample, 信号弱。Oversample 让 VLA "记住" 什么时候该 stop。

Training hyperparameters (Table VIII):
- 4× A100 40GB, LR=5e-4 decay to 5e-5 at 335K steps
- Effective batch 64 (2/GPU × 8 grad accum)
- 500K steps, diffusion 50 steps
- 224×224 input, 1 third-person + 1 wrist image
- LoRA rank 32, 313M trainable (111M LoRA + 185M action head + 17M proprio projector)
- Chunk size H=8, predict 8 execute all 8 open-loop
- Image aug: 90% random crop, color jitter

---

## III. Inference: VLM as Failure Predictor

当 VLA-predicted progress $p_t \geq \tau_p = 0.9$, query VLM (GPT-5.2, temp=1.0)。VLM 输入:
1. **Third-person view** (global context: object identity, gripper pose, 当前 subtask 是否对)
2. **Wrist view** (fine-grained: gripper alignment, contact quality)
3. Current subtask $g_k$
4. Subtask list $G = (g_1, \ldots, g_K)$

VLM 输出 CoT reasoning + decision: `transit` 或 `backtrack` + `next_subtask`。

如果 backtrack, restore robot state 到 target subtask 起点 via **reverse-executing recorded delta actions** (Bellman-Guided Retrials 的做法)。这假设 state transitions reversible——是论文的 limitation。

### Confirmation Mechanism (Appendix D)

预测的 $s_t$ 和 $p_t$ 有 noise, 用 CONFIRM(·) 机制过滤:
- Track: `first_seen` (是否见过 high signal), `c_consec` (连续 high signal 计数), `c_gap` (自上次 high signal 后的 low-signal 步数)
- Confirm 条件:
$$c_{consec} \geq 2 \quad \text{OR} \quad (\text{first\_seen} \wedge c_{gap} \geq 2)$$

即: 两个连续 high signal, 或者 high signal 出现后隔了 ≥2 步又出现。这过滤 isolated spurious 预测, 同时保持 responsive。

### VLM Prompt 设计要点 (Appendix F)

Prompt 强调:
- Backtrack 到 **earliest subtask that restores missing precondition** (通常是 reach/align, 不是 open/close gripper)
- 区分 visual evidence from front/wrist view, 显式输出 `view_agreement` (agree/partial/disagree + 哪个 view dominates)
- Output `success_likelihood: high/medium/low`, `key_risks`
- 约束: bullets ≤12 words, 用 exact subtask strings

---

## IV. MBR Decoding: Test-Time Scaling

### A. 核心公式

MBR 的目标是从 N 个 sampled hypotheses $\mathcal{A} = \{a^{(1)}, \ldots, a^{(N)}\}$ 选 expected risk 最小的:

$$a_{t:t+H-1}^{\text{MBR}} = \arg\min_{a \in \mathcal{A}} \mathbb{E}_{a' \sim \pi_\theta}[d(a, a')]$$

变量:
- $t:t+H-1$ = chunk time range, $H$ = chunk size (=8)
- $a'$ = 从 policy $\pi_\theta$ sample 的 reference hypothesis
- $d(\cdot, \cdot)$ = distance metric

Monte Carlo 估计 (用 sampled set $\mathcal{A}$ 代替 expectation):
$$\mathcal{L}(a) = \frac{1}{N}\sum_{n=1}^N d(a, a^{(n)})$$

上标 $(n)$ = 第 n 个 sample, $N$ = hypothesis 总数。

采样版 MBR:
$$a^{\text{MBR}} = \arg\min_{a \in \mathcal{A}} \mathcal{L}(a)$$

### B. Feature Space 距离

不是在 raw action space 算距离, 而是累积 delta 得到 predicted trajectory, 用 feature vector:
$$\phi(a^{(n)}_{t:t+H-1}) \in \mathbb{R}^{6H}$$

6H 是因为每步 6 维 (x, y, z, u, v, w), H 步累积。$\gamma_t, s_t, p_t$ **不包含**在 feature 里——只有 end-effector motion。

最终:
$$a^{\text{MBR}} = \arg\min_{a^{(i)} \in \mathcal{A}} \frac{1}{N}\sum_{j=1}^N d(\phi(a^{(i)}), \phi(a^{(j)}))$$

即 N×N pairwise distance matrix, 选 row-mean 最小的 hypothesis。这 favor **high-density region** 的 trajectory。

### C. Density-Based MBR Variant (Appendix C)

论文实际用的不是 standard MBR (选最小 average distance), 而是 **density-based variant**:

**r-NN radius density estimation**: 对每个 hypothesis, 算它到第 r 近邻居的距离, 距离越小 density 越高。r 自适应:
$$r = \max\left(2, \min\left(4, \lfloor\sqrt{N}\rfloor\right)\right)$$

对 N=8: $\lfloor\sqrt{8}\rfloor = 2$, 所以 r=2。

**Medoid selection**: 
1. 找 densest point (smallest r-NN radius) 作为 pocket center
2. 在这个 pocket 内找 medoid (到其他 pocket members 平均距离最小的 hypothesis)
3. 这个 medoid 作为 final action chunk

这比 vanilla MBR 更 robust, 因为它先 localize densest region 再选 representative, 避免被 outlier 拉偏。

### D. Stochastic Sampling 实现

Backbone 是 OpenVLA + diffusion action expert。Stochasticity 来自 diffusion noise sampling——不同 random seed → 不同 noise → 不同 action chunk。N=8 就是 8 个不同 seed 的 forward pass。

### E. 为什么 MBR 对 VLA 有效

Imitation learning trained VLA 的成功 behavior **cluster 在 high-density region** of policy output space (因为 demo 是 unimodal 的 successful trajectory)。所以 consensus selection (选 densest) 更可能命中 successful mode。这跟 RoboMonkey 的 Gaussian perturbation + VLM selection 思路类似, 但 MBR **不需要外部 verifier**——纯 geometric consensus。

---

## V. 实验: Build Intuition

### A. LIBERO Main Results (Table I)

| Method | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| ThinkAct | 88.3 | 91.4 | 87.1 | 70.9 | 84.4 |
| FPC-VLA | 87.0 | 92.0 | 86.2 | 82.2 | 86.9 |
| GR00T N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| **CycleVLA** | **97.6** | **98.1** | 91.7 | **93.6** | **95.3** |

注意 **Long suite**: OpenVLA 53.7 → CycleVLA 93.6, +39.9 points。这验证了核心 hypothesis——long-horizon task 的 error accumulate across subtasks, proactive correction 在 transition points 干预能 dramatic 改善。

### B. Under-Trained VLAs (Table II) — 这是最 insightful 的实验

| Checkpoint | Avg w/o FC | Avg w/ FC | Δ |
|---|---|---|---|
| 200K | 73.2 | 80.0 | +6.8 |
| 350K | 83.2 | 89.2 | +6.0 |
| 500K | 89.3 | 95.3 | +6.0 |

观察: **CycleVLA bridge the gap between model sizes**。200K + FC ≈ 350K without FC; 350K + FC ≈ 500K without FC。这意味着 compute 可以 shift from training-time to test-time scaling——对 compute-constrained 场景很有价值。

### C. MBR Hypothesis Count (Table III)

| N | 200K MBR | 350K MBR | 500K MBR |
|---|---|---|---|
| 4 | 78.2 | 88.0 | 94.1 |
| 8 | 78.5 | 90.2 | 95.5 |
| 16 | 81.2 | 91.3 | 95.7 |
| 32 | 79.7 | 92.3 | 95.7 |
| 64 | 79.7 | 92.2 | 95.6 |

Intuition: 
- N=4→8 gains 最大 (更好近似 expected risk)
- N=16+ plateau (N=32 比 16 还略低 for 200K, 可能 sampling noise)
- Weaker model (200K) gain 更多 (+5.3 to +9.3) vs 500K (+3.3 to +5.3)

这跟 LLM MBR 的观察一致: weaker models 产生更多 inconsistent candidates, consensus selection 的 marginal benefit 更大。

### D. Distance Metric (Table IV)

| Metric | 200K | 350K | 500K |
|---|---|---|---|
| Random | 71.7 | 80.3 | 90.2 |
| MBR-$L_1$ | 78.7 | 89.7 | 94.7 |
| **MBR-$L_2$** | **78.5** | **90.2** | **95.5** |
| MBR-$L_\infty$ | 77.8 | 88.2 | 94.3 |
| MBR-cos | 74.1 | 87.5 | 94.4 |
| MBR-r | 73.5 | 86.9 | 94.8 |

$L_2$ 最好, cos/correlation 最差。Hypothesis: translation components dense along trajectory, rotation sparse (很多 timestep 几乎不旋转)。Distance-based metrics ($L_1, L_2$) 捕捉 magnitude 差异, cos/r 强调 directional agreement——对 sparse rotation 不友好。

### E. Ablation (Table VI)

| Variant | SR | Time (s) |
|---|---|---|
| w/o MBR | 92.5 | 302.4 |
| alt VLM (LLaMA-3.2-11B) | 92.8 | 172.6 |
| w/o stop + LAO | 91.1 | 186.8 |
| always-on MBR (UB) | 96.9 | 464.3 |
| pred. failure cutoff (LB) | 79.7 | 110.2 |
| **main** | **95.3** | **215.3** |

关键 insight:
- **w/o MBR**: SR 92.5 (vs 95.3), 但 time 302s (vs 215s)。Wait, 没 MBR 反而更慢? 因为 retry up to R=3 次都 random select, 失败概率高, 触发更多 backtrack 循环。MBR 虽然每次多 sample, 但减少 retry 次数, 净更快。
- **alt VLM (LLaMA-3.2-11B)**: SR 92.8 (略降), time 172s (大幅降)。LLaMA 更 conservative——更频繁选 transit 而非 backtrack, 所以更少 action sampling。这暗示 VLM 的 "aggressiveness" 在 backtrack frequency 上是个 dial。
- **always-on MBR (UB)**: SR 96.9 (最高), 但 time 464s (2× main)。每次 transition 都 MBR, 不用 VLM decide。这是 upper bound——证明 VLM gating 是 efficiency/effectiveness trade-off 的 sweet spot。
- **pred. failure cutoff (LB)**: SR 79.7, 跟 no-FC baseline (89.3 for 500K) 差 10%。Wait, 应该 match 才对? 解释: VLM sycophancy——被问 "会不会失败" 时倾向 confirm。所以 cutoff 后很多 episode 误判为 failure。这 10% gap 量化了 sycophancy cost。

### F. Runtime (Table V)

A10 GPU breakdown:
- Action Rollout: 147.6s (68.6%) — 主要 bottleneck
- Action Sampling: 47.9s (22.2%)
- VLM API: 12.9s (6.0%)
- Backtrack: 6.9s (3.2%)
- **MBR: 0.003s (<0.1%)** — 几乎免费!

MBR 的 N×N pairwise distance 在 $\mathbb{R}^{6H}$ = $\mathbb{R}^{48}$ 上算 8×8=64 个距离, 完全 negligible。这是 MBR 相对 learned verifier (Rover, V-GPS) 的大优势——verification compute 几乎为零。

A100 上 total 76.9s vs A10 215.3s, action inference 加速但 VLM API latency 占比升到 19.9% (网络 bound)。

---

## VI. Algorithm 1 完整流程

```
Decompose g → G = (g_1, ..., g_K) via VLM
k=1, t=0, phase=MONITOR, Q=∅, r_{1:K}=0

while k ≤ K and t < T_max:
    observe o_t
    if Q empty: sample chunk a_{t:t+H-1} ~ π_θ(·|o_t, g_k), push to Q
    pop a_t, parse (ã_t, s_t, p_t), execute ã_t, observe o_{t+1}
    if episode succeeds: return success
    
    if phase=MONITOR and CONFIRM(p_t ≥ τ_p):
        (j, dec) = V(o_t, g_k, G)  # VLM decision
        if dec=backtrack and r_j < R:
            r_j += 1
            restore robot to start of g_j via reverse execution
            Q = ∅, observe o_t
            sample A = {a^{(n)}}_{n=1}^N ~ π_θ(·|o_t, g_j)
            select a^MBR via Eq(4), Q = a^MBR
            k = j, phase = MONITOR
        else:
            phase = COMPLETE
    elif phase=COMPLETE and CONFIRM(s_t = 1):
        k += 1, phase = MONITOR, Q = ∅

return failure
```

Two-phase per subtask: MONITOR (盯 progress 到 τ_p) → COMPLETE (盯 stop signal)。Backtrack reset phase 到 MONITOR。

---

## VII. 跟相关工作的 Positioning

### vs PAINT [1]
PAINT 做 proactive intervention 但需要 human-in-the-loop。CycleVLA 用 VLM 替代 human decision, fully autonomous。

### vs Bellman-Guided Retrials [23]
Bellman-Guided Retrials 也做 backtracking + retry, 但用 value-based selection 且**不 target generalist policies**。CycleVLA 用 MBR (training-free) + 专门为 VLA foundation model 设计。

### vs RoboMonkey [32]
RoboMonkey: Gaussian perturbation sampling + trained VLM selector。CycleVLA: diffusion noise sampling + MBR consensus (no external verifier)。MBR 的优势是 zero-shot, 不需要训练 selector。

### vs Rover [33], V-GPS [62]
这些用 learned reward/value model score sampled actions。需要训练 verifier。MBR 用 intrinsic geometric consensus, training-free。

### vs FPC-VLA [71]
FPC-VLA 也做 failure prediction + correction, 但用专门的 supervisor module。CycleVLA 用 off-the-shelf VLM (GPT-5.2), zero-shot。

### vs ECoT [51], CoT-VLA [46]
这些在 training 时 inject textual reasoning。CycleVLA 不改 VLA 内部 reasoning, 用 external VLM 做 reasoning at subtask boundaries。

### vs SeqVLA [19], Long-VLA [20]
这些也 tackle long-horizon, 但通过 sequence modeling。CycleVLA 通过 explicit subtask decomposition + progress tracking + backtracking。Orthogonal approaches。

---

## VIII. Limitations & Future Directions

1. **Reversible state assumption**: backtrack via reverse-executing delta actions 假设 state reversible。在 irreversible env ( irreversible grasp, breakable objects) 会 fail。Real robot 上这个问题更严重。
2. **VLM sycophancy**: 被问 "会不会失败" 时 VLM 倾向 confirm, 导致 false positive backtrack。Ablation 显示 10% SR 损失。Future: debiased prompting 或 calibration。
3. **Inference latency**: test-time scaling 增加 ~30% runtime (Table V)。Contact-rich task 需要高 control frequency, 可能不适用。
4. **External VLM dependency**: 需要 GPT-5.2 API。Ablation 显示 LLaMA-3.2-11B 略降但可用, 暗示可以本地化。
5. **No real robot experiments**: 论文承认这是 limitation, "to be added in the near future"。

Future: 把 failure prediction + recovery 内化进 VLA, end-to-end learned self-correction, 而非 external VLM + heuristic backtrack。Tailored test-time scaling for VLA (不只是借用 LLM 的 MBR)。

---

## IX. 我的 Intuition 总结

CycleVLA 的 elegance 在于它**没有重新发明轮子**——把 LLM 领域的 MBR decoding [29], VLN 领域的 backtracking [21], manipulation 领域的 subtask decomposition [16], 全部组装到一个 VLA inference pipeline 里。每个 component 单独都不是 novel, 但组合产生 synergistic effect:

- Subtask progress prediction 提供 **when to check**
- VLM 提供 **what to check** (failure prediction) 和 **where to recover** (backtrack target)
- MBR 提供 **how to recover better** (consensus retry)

而最关键的 insight 是把 failure detection 从 **post hoc** 推到 **proactive**——在 transition 之前用 progress signal trigger check, 而不是 fail 之后才发现。这个 timing shift 是整个 system 价值的 foundation。

Under-trained VLA 的实验特别 compelling: 它说明 test-time compute 可以 substitute training compute, 跟 LLM 那边 "scaling test-time compute can be more effective than scaling parameters" (Snell et al. [59]) 的结论呼应。Robotics 的 "inference-time scaling law" 可能跟 LLM 不一样——MBR 这种 consensus-based method 在 unimodal demo data 上特别 effective, 而 LLM 那边 best-of-N + verifier 更常见。

最后 sycophancy 的发现也很有启发性: VLM 作为 failure predictor 有 systematic bias, 这是 LLM-as-judge 在 robotics 场景的 specific failure mode。Future work 可能需要 uncertainty calibration 或 contrastive prompting ("will this succeed OR fail?") 来 debias。

---

## References

- [OpenVLA](https://openvla.github.io/) — backbone VLA
- [LIBERO benchmark](https://libero-project.github.io/) — evaluation suite
- [MBR decoding original (Kumar & Byrne 2004)](https://aclanthology.org/N04-1022/) — theoretical foundation
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) — action expert
- [ECoT / Robotic Control via Embodied Chain-of-Thought](https://ecot.github.io/) — movement primitive extraction
- [π0 (Physical Intelligence)](https://www.physicalintelligence.company/blog/pi0) — VLA foundation model reference
- [GR00T N1](https://developer.nvidia.com/groot) — humanoid VLA baseline
- [GPT-4.1](https://openai.com/index/gpt-4-1/) — subtask decomposition LLM
- [Bellman-Guided Retrials (Du et al. 2024)](https://arxiv.org/abs/2406.15917) — backtracking mechanism
- [Snell et al. "Scaling LLM test-time compute"](https://arxiv.org/abs/2408.03314) — test-time scaling theory
- [Tactical Rewind (Ke et al. 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ke_Tactical_Rewind_Self-Correction_via_Backtracking_in_Vision-and-Language_Navigation_CVPR_2019_paper) — backtracking in VLN
- [RoboMonkey](https://arxiv.org/abs/2506.05020) — test-time scaling for VLA
- [FPC-VLA](https://arxiv.org/abs/2509.04018) — failure prediction & correction VLA
- [Sycophancy in LMs (Sharma et al.)](https://arxiv.org/abs/2310.13548) — VLM bias analysis
- [TraceVLA](https://arxiv.org/abs/2412.17440) — visual trace prompting baseline
