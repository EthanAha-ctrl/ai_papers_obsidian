---
source_pdf: VLA-Corrector Lightweight.pdf
paper_sha256: 5deb665359c7e05478b89cb6a0d3974d96ad669d2b04ef5e00d53a2639d0033a
processed_at: '2026-08-13T02:42:42-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 VLA-Corrector

## 一句话说清楚

机器人做任务时，每次"想"很慢，所以习惯一次想好十步然后闭眼执行。问题是中间如果偏了它不知道。这篇paper给机器人装了个"监控摄像头"，发现偏了就立刻停下来重新想，而且重新想的时候还会主动往回纠。

---

## 生活类比

想象你开车去一个地方：

**原来的做法**：出发前用导航规划好路线，然后闭着眼睛开10公里，到了再重新看导航。如果中间路被封了、或者你拐错弯了，你根本不知道，继续按原路开，越开越远。

**H=1的做法**：每开1米就停下来重新看导航。准是准，但你永远到不了，光停车了。

**VLA-Corrector的做法**：你还是一次规划10公里，但开车的时候眼睛睁着。一旦发现"诶，路边怎么跟我预想的不一样了"，而且连续好几秒都不对，你就停下来重新导航。重新导航的时候还会特意往"纠偏"的方向偏一点。

---

## 问题到底是什么

现在的VLA模型（π0.5、SmolVLA这些）每次生成action很慢（要跑diffusion/flow matching），所以大家都是一次生成一串action（比如50个），然后只执行前H个。

这叫action chunk，好处是省算力，坏处是执行这H步时是**开环盲跑**的。比如开抽屉，手滑了一下，你接下来的9步还是按"没滑"的计划走，越走越歪。

H大了效率高但容易翻车，H小了准但太慢。Figure 2的实验很扎心：H从10变50，π0.5的成功率从64%掉到49%，但policy调用次数减少4倍。所有backbone都这样，不是某个model的问题。

---

## 三步走的解决方案

### 第一步：装个"直觉监控器"（LVM）

训练一个很小的模型（40M的MLP），它学会了一件事：**在当前画面下做这个动作，画面应该怎么变**。

它不学完整的"世界会变成什么样"（那是world model的活，太难），只学"画面变化的**方向**对不对"。

而且关键设计：它预测的是**残差**（ΔZ = 下一帧latent - 这一帧latent），不是绝对画面。这样背景、没动的物体自动被减掉了，模型只关注任务相关的变化。

线上跑的时候，每步拿实际画面变化和预期变化算个cosine distance，得到一个"偏没偏"的分数E_t。E_t低=正常，E_t高=偏了。

### 第二步：发现真偏了就喊停（Event-Triggered Truncation）

不能E_t一高就停，因为可能是瞬态噪声。用的是一套robust规则：

- 维护最近15步的E_t滑动窗口
- 算median和MAD（比mean/std抗噪声）
- 设两个阈值：T_on（确认偏了）和T_off（确认回来了），T_on > T_off形成hysteresis，防止来回抖
- 连续5步E_t > T_on才真正触发interrupt

触发后：丢掉当前chunk剩下的action，立刻重新调policy。

实际效果（Figure 6）：83.7%的中断发生在critical phases（精确抓取、对齐），只有16.3%发生在non-critical phases（拿着东西移动）。说明这个监控器确实在"该停的时候停，不该停的时候不浪费"。

### 第三步：重新规划时往回纠（OGG）

光停了重新跑一次policy还不够，因为robot已经偏到了training时没见过的状态，naive re-inference可能还是生成失败的action。

OGG的做法：在flow matching的denoising过程中，算一个"纠偏方向"：

- "本来应该往这走" = 从偏之前那步预测的expected dynamics
- "实际偏到这了" = 偏差累积
- 纠偏方向 = expected - actual_deviation

然后在这个方向的gradient上拉一下velocity field，让生成的action偏向"纠偏后的正确方向"。

关键：这是soft的gradient guidance，不是hard constraint。VLA prior仍然是主driver，OGG只是轻轻bias一下方向。所以η=1是sweet spot，η太大（100）反而把prior搞坏了，hard task从47.5%掉到36.7%。

而且OGG只在interrupt后的**一次**policy call里激活，不是每次都激活，所以overhead可控。

---

## 为什么这个思路work

### 1. 把"选H"的问题变成"判断什么时候不信H"

之前大家的思路是"怎么选一个更好的固定H"，但paper的核心insight是：**问题不在H的大小，在于你不知道什么时候这个chunk已经不可信了。**

这就把一个静态参数选择问题变成了一个动态检测问题，本质上是event-triggered control的思路。

### 2. 检测和修正解耦

- 检测器是external的40M小模型，在frozen VLA features上跑，不碰backbone
- 修正器是gradient guidance，在flow matching velocity上注入，不直接改action坐标

为什么不用backbone内部加个head做检测？Table 7的ablation很直接：internal head + OGG = 49.55%，external LVM + OGG = 64.35%。因为internal auxiliary objective会修改backbone representation，污染原来action generation的行为。

### 3. 残差预测让问题变简单

预测"下一帧长什么样"很难，但预测"做了这个动作后变化方向对不对"就容易多了。背景被减掉，只关注task-relevant dynamics，所以40M MLP就够，而且r=0.6的数据量就饱和了（Table 3）。

### 4. 效率不降反升

最反直觉的结果：VLA-Corrector在**提高成功率的同时还减少了policy调用次数**。

比如SmolVLA H=10：成功率61.9%→73.0%（+11.1），调用次数19.27→15.64（-3.63），success-per-call效率+45.3%。

为什么会这样？因为原来的policy call有很多浪费在"执行已经stale的action"上了。VLA-Corrector提前截断这些浪费，每次policy call都更有价值。

---

## 实验里最亮的数据

### Few-shot超越Full Fine-tuned

LIBERO上，few-shot微调的π0.5只有94.0%成功率，加了Corrector之后97.8%，**超过了full fine-tuned的96.95%**。

这说明：few-shot已经学到了大部分正常轨迹，缺的是recovery行为。传统思路是"多收集failure data训练recovery"，VLA-Corrector走另一条路："inference时主动enable recovery"，不依赖额外训练数据。

### 真实机器人disturbance recovery

AgileX PiPER机械臂，人在执行过程中手动挪动目标物体：

- baseline：40.0%成功率
- + VLA-Corrector：68.3%（+28.3）

这是最大的提升，正好对应设计目标：有外部干扰时stale chunk最致命，Corrector收益最大。

### Cross-architecture都涨

不是只对一个model有用：π0.5 +15.65，SmolVLA +4.75，X-VLA +4.05。说明这是paradigm-level的改进，不是model-specific的trick。

---

## 没解决的问题

Paper自己承认的：
1. 物体被挪到够不着的地方，一次OGG recovery救不回来
2. 没有力反馈的6-DoF机械臂在接触密集任务中还是会因为摩擦/几何原因失败
3. OGG依赖frozen backbone prior，backbone本身表达不了的recovery behavior，OGG也造不出来

我能想到的额外问题：
1. M_φ在drifted states上可能OOD，prediction可靠性存疑
2. k=10的prediction horizon是固定的，fast/slow dynamics任务可能需要adaptive k
3. 如果M_φ预测和实际都错在同一个方向，E_t会假低，漏检

---

## 一句话总结

**不要纠结选多大的H，要学会判断什么时候H已经不可信了，然后在那个时刻果断停下来、聪明地重新规划。**

---

# VLA-Corrector 深度技术讲解

## 1. 高层直觉：这个 paper 在解决什么问题

VLA-Corrector 处理的是 action-chunked VLA policies 的一个本质矛盾。现代 VLA foundation models (π0.5, SmolVLA, X-VLA) 都使用 generative action modeling (flow matching 或 diffusion) 来表达 multi-modal action distributions，每个 policy call 的 latency 很高。为了 amortize 这个 cost，几乎所有现代 policy 都采用 action chunk：一次 forward pass 预测 C 个 future actions，但只执行前 H 个作为 action horizon。

这个设计创造了一个 **open-loop blind spot**：在 horizon H 内，controller 拿到了 fresh observations 但 policy 不再被 query。在 contact-rich manipulation 中，即使小的 perturbation 也会在这个 blind spot 内快速放大，导致 compounding errors。等到 horizon 结束再 replan 时，robot 可能已经 drift 到 OOD state，即使新 inference 也无法 recover。

Fig 1 中的对比直观：H=10 时 robot 在 drawer-opening 任务中卡死，H=1 时保持 closed-loop 反应性成功完成任务。但 H=1 否定了 action chunking 的效率初衷。

Paper 的核心 insight 在 Section 1 末尾：
> "the key is not to choose a better fixed horizon, but to decide when the current chunk should stop being trusted."

这是一个 event-triggered 控制的思路，把固定的 H 变成 adaptive H_adaptive = h < H，只在检测到持续 drift 时才缩短。

Reference: Action Chunking Transformer (ACT) 原始论文 https://arxiv.org/abs/2304.13705, Diffusion Policy https://arxiv.org/abs/2303.04137, π0 https://arxiv.org/abs/2410.24164, π0.5 https://arxiv.org/abs/2504.16054

---

## 2. Trade-off 的量化：为什么固定 horizon 不够用

Fig 2 的实验是 motivation 的关键。在 π0.5、SmolVLA、X-VLA 三个不同 backbone 上做 horizon sweep：

- **π0.5**: 从 H 小到 H 大，policy calls 减少约 4×，但 success rate 从 64% 降到 <49%
- **SmolVLA**: 类似 trend
- **X-VLA**: 类似 trend

这个 trend 跨 backbone 一致，说明这是 action chunking paradigm 的内在问题，不是某个 model 的缺陷。

Table 4 给出完整的 horizon sweep 数据，这里值得仔细看：

| Model | Horizon | Baseline Succ | Baseline Calls | Ours Succ | Ours Calls | Eff Gain |
|-------|---------|---------------|----------------|-----------|------------|----------|
| π0.5  | 10 | 64.50 | 20.41 | 72.40 | 17.64 | +29.9% |
| π0.5  | 50 | 48.72 | 5.15 | 58.70 | 4.98 | +24.6% |
| SmolVLA | 10 | 61.90 | 19.27 | 73.00 | 15.64 | +45.3% |
| SmolVLA | 50 | 54.20 | 4.86 | 62.90 | 4.68 | +20.6% |
| X-VLA | 32 | 44.00 | 8.61 | 54.40 | 8.34 | +27.6% |

注意一个微妙现象：VLA-Corrector 在大多数情况下 **既提高 success 又降低 policy calls**。这是一个很强的信号，说明它本质上让每次 policy query 更有价值，truncation 把浪费在 stale actions 上的 query 节省了下来。Efficiency gain 在 long horizon 时特别大（X-VLA H=4 时 +39.1%, SmolVLA H=10 时 +45.3%）。

---

## 3. 方法详解 1：External Latent Dynamics Corrector 训练

### 3.1 设计思路

VLA-Corrector 完全不动 VLA backbone 的 weights，只在它上面训练一个 external lightweight module M_φ。这是一个关键的 engineering 决策：modular design，可以针对每个 benchmark 重新训练 corrector 而不 re-optimize VLA backbone。

为什么不用 VLA backbone 内部的 head 来做 detection？Table 7 的 ablation 给出答案：
- Internal head + OGG: 49.55% avg
- Decoupled LVM + OGG: 64.35% avg (+14.80)

Internal auxiliary objective 会更新 backbone representations，这些 representations 同时也用于 VLM-to-action planning，会污染原 policy 的 action generation 行为。External LVM 在 frozen VLA features 上学 monitoring signal，不直接修改 policy representation。

### 3.2 Residual Prediction 而非绝对 State Prediction

这是 paper 的一个重要设计。给定 transition (o_t, a_t, o_{t+k})，先 compute：

$$Z_t^{\text{real}} = \mathcal{E}(o_t), \quad Z_{t+k}^{\text{real}} = \mathcal{E}(o_{t+k}), \quad \Delta Z_{t+k}^* = Z_{t+k}^{\text{real}} - Z_t^{\text{real}}$$

变量解释：
- $Z_t^{\text{real}}$: 时间 t 的真实视觉 latent，由 VLA visual encoder $\mathcal{E}$ 提取
- $Z_{t+k}^{\text{real}}$: 时间 t+k 的真实视觉 latent
- $\Delta Z_{t+k}^*$: target short-horizon visual latent evolution，即执行 action $a_t$ 后预期的 latent residual

然后 M_φ 预测这个 residual：

$$\Delta \hat{Z}_{t+k} = M_\phi(Z_t^{\text{real}}, a_t)$$

注意：M_φ 不预测绝对的 future latent state $\hat{Z}_{t+k}$，只预测 residual $\Delta \hat{Z}_{t+k}$。这个设计有几个好处：
1. 静态场景内容（背景、未动对象）在 $Z_{t+k} - Z_t$ 中被 naturally subtract 掉
2. 模型 focus 在 task-relevant dynamics 上
3. 任务变得更低维、更局部，40M MLP 就够

### 3.3 Training Objective

$$\mathcal{L}_{\text{corr}} = \|\Delta \hat{Z}_{t+k} - \Delta Z_{t+k}^*\|_2^2 + \beta [1 - \text{CosSim}(\Delta \hat{Z}_{t+k}, \Delta Z_{t+k}^*)]$$

变量解释：
- $\|\cdot\|_2^2$: L2 squared norm，enforce magnitude matching
- $\beta$: 平衡 residual accuracy 和 directional alignment 的超参数
- $\text{CosSim}$: cosine similarity，enforce direction matching
- $1 - \text{CosSim}$: 当两个向量方向完全一致时为 0，完全相反时为 2

为什么需要 cosine 项？L2 alone 会让模型在 magnitude 接近 0 的静态区域数值上 trivially fit 但方向 noise 很大。Cosine term 强制方向一致，对 detection 更 robust，因为后面 E_t 用的就是 cosine-based inconsistency。

### 3.4 Architecture 和 Training Details

从 Appendix C.2: M_φ 是 4-layer residual MLP，hidden width [2048, 2048, 2048, 2048]，约 38-42M params。Action 经过 linear embedding 后和 $Z_t^{\text{real}}$ concatenate 进入 MLP stack。

训练用 AdamW, lr=3e-4, weight decay=1e-4, cosine annealing (η_min = 0.01 × lr), 30 epochs, early stopping patience=5。

关键 insight：corrector 在 demonstration trajectories 上训练。Demo 虽然有 teleoperation jitter，但仍然 reflect on-track execution。M_φ 的目标 **不是建模所有可能的 future 作为 full world model**，而是学 local on-track consistency signal。这是一个非常重要的定位：不是 world model，是 on-track consistency classifier 的 continuous 版本。

Table 3 显示 data efficiency 验证这个定位：r=0.6 时 avg 已经 52.20 (+3.48)，r=1.0 时 54.32 (+5.60)，diminishing returns 明显。一个 40M 的 local dynamics module 不需要 exhaustive task coverage。

---

## 4. 方法详解 2：Online Anomaly Detection (LVM)

### 4.1 Expected vs Actual Residual 对比

在 deployment 时的每个 control step t：

**Expected residual** (在 t 时刻预测，由 M_φ 生成)：
$$\Delta Z_{t+k}^{\text{exp}} = M_\phi(Z_t^{\text{real}}, a_t)$$

这里 $a_t$ 是当前正在执行的 action（来自 stale chunk 但仍然知道）。

**Actual residual** (在 t+k 时刻从 fresh observation 计算)：
$$\Delta Z_{t+k}^{\text{real}} = Z_{t+k}^{\text{real}} - Z_t^{\text{real}}$$

注意一个精妙之处：尽管 policy 不再被 query，但 visual encoder $\mathcal{E}$ 仍然在 fresh observation 上跑（这相对 cheap），所以 $Z_{t+k}^{\text{real}}$ 始终 available。

### 4.2 Inconsistency Score

$$E_t = 1 - \text{cos sim}(\Delta Z_{t+k}^{\text{exp}}, \Delta Z_{t+k}^{\text{real}}), \quad \text{CosSim}(\mathbf{u}, \mathbf{v}) = \mathbf{u}^\top \mathbf{v} / (\|\mathbf{u}\| \|\mathbf{v}\|)$$

变量解释：
- $E_t$: 时间 t 的 inconsistency score，范围 [0, 2]
- $\Delta Z_{t+k}^{\text{exp}}$: M_φ 预测的 expected latent evolution
- $\Delta Z_{t+k}^{\text{real}}$: 实际观察到的 latent evolution
- $E_t = 0$: 完全匹配（on-track）
- $E_t = 1$: 正交（不相关）
- $E_t = 2$: 完全相反

Fig 5 验证这个 score 有意义：successful episodes 集中在低 $E_t$，failed episodes 有 heavier high-score tail 且 trigger 更多 interrupt events。

### 4.3 为什么用 cosine 而不是 L2

L2 distance 在高维 latent space 中数值 scale 不稳定，且对 magnitude 敏感。Cosine 对 magnitude 不敏感，只关注 direction mismatch，更符合 "dynamics 是否还在 expected direction" 的语义。这和 training objective 中的 CosSim 项是 consistent 的。

---

## 5. 方法详解 3：Event-Triggered Truncation

### 5.1 为什么不直接 threshold $E_t$

直接 thresholding $E_t > T$ 不稳定，因为 transient visual outliers（光照变化、瞬态遮挡）会 false trigger。Paper 用 robust event-triggered rule：dynamic thresholds + persistence checking。

### 5.2 Sliding Window Robust Statistics

维护 sliding window $\mathbf{E}_W = \{E_{t-w+1}, \ldots, E_t\}$，window size w=15。

$$M_e = \text{median}(\mathbf{E}_W), \quad \text{MAD} = \text{median}(|E_i - M_e|), \quad E_i \in \mathbf{E}_W$$

变量解释：
- $M_e$: window 内 inconsistency scores 的 median，反映 normal level
- MAD: Median Absolute Deviation，对 outlier robust 的 spread 度量（比 std 对 spikes 不敏感）

### 5.3 Asymmetric Hysteresis Thresholds

$$T_{\text{on}} = M_e + \lambda_{\text{on}} \cdot \text{MAD}, \quad T_{\text{off}} = M_e + \lambda_{\text{off}} \cdot \text{MAD}, \quad \lambda_{\text{on}} > \lambda_{\text{off}}$$

参数 (Appendix B.2): $\lambda_{\text{on}} = 3.0$, $\lambda_{\text{off}} = 2.0$。

- $T_{\text{on}}$: 较高阈值，confirm 持续 deviation
- $T_{\text{off}}$: 较低阈值，提供 hysteresis，防止在 normal/abnormal 边界快速 oscillation

Hysteresis 是 control theory 经典技巧，类似 Schmitt trigger。要进入 abnormal state 需要超过 $T_{\text{on}}$，但要回到 normal state 需要降到 $T_{\text{off}}$ 以下。这避免边界抖动。

### 5.4 Persistence Counter

$$c_t = \begin{cases} c_{t-1} + 1, & \text{if } E_t > T_{\text{on}} \\ 0, & \text{if } E_t < T_{\text{off}} \\ c_{t-1}, & \text{otherwise} \end{cases}$$

Interrupt event triggered 当 $c_t \geq p$，其中 p=5 (patience)。

变量解释：
- $c_t$: persistence counter，记录连续 abnormal 步数
- $p$: patience parameter，要求连续 5 步 abnormal 才触发
- Reset condition: $E_t < T_{\text{off}}$（hysteresis 下界）
- Hold: $T_{\text{off}} \leq E_t \leq T_{\text{on}}$ 时 counter 不变

Cooldown: 触发后 10 步不再 trigger，避免 immediate re-trigger。

### 5.5 Adaptive Horizon

如果在 queue 中已经执行了 h 个 actions，realized horizon 变成：
$$H_{\text{adaptive}} = h < H$$

剩余 stale actions 被丢弃，立即 query policy 做 corrective replan。

这个机制的 elegance：long-horizon 在 stable phase 保留效率，short-horizon 在 drift phase 恢复 precision。Fig 6 显示 83.7% 的 truncation 发生在 critical phases（precise grasping, alignment），只有 16.3% 在 non-critical phases（tolerant transport）。这验证了 adaptive-horizon intuition：不是 uniform 缩短 horizon，而是在 error-sensitive phase 才 shorten。

---

## 6. 方法详解 4：Online Gradient Guidance (OGG)

### 6.1 为什么 naive replan 不够

Truncation 停止 stale actions，但 recovery 取决于下一次 replan。问题是：VLA 在 deviated state 上 naive re-inference 可能再次生成失败的 actions，因为 robot 已经 drift 到 OOD region，policy 没见过这种 state 的 recovery。

OGG 的目标：在 interrupt 后的 single policy call 中，注入 corrective gradient，主动 steer 生成方向回到 intended trajectory。

### 6.2 Flow Matching 背景

VLA action generation 用 flow matching (类似 π0/π0.5)。给定 noisy action $A^\tau$ 在 denoising step $\tau$：
$$v_\tau = \pi_\theta(A^\tau, Z_t^{\text{real}}, \tau)$$

变量解释：
- $A^\tau$: denoising step $\tau$ 时的 noisy action chunk
- $v_\tau$: VLA 预测的 velocity field
- $\tau$: denoising time, 从 $\tau_{\text{max}}$ (纯噪声) 到 0 (clean action)
- $\pi_\theta$: parameterized by $\theta$ 的 VLA policy

标准 update rule:
$$A^{\tau - \Delta \tau} = A^\tau - \Delta \tau \cdot v_\tau$$

### 6.3 Predicted Action Effect

在第 τ 步估计 clean chunk:
$$\hat{A}_0 = A^\tau - \tau \cdot v_\tau$$

取第一个 action: $\hat{a}_t = \hat{A}_0[0]$

M_φ 预测这个 candidate action 的 latent effect:
$$\Delta \hat{Z}_{\text{act}} = M_\phi(Z_t^{\text{real}}, \hat{a}_t)$$

### 6.4 Corrective Target 构造

这是 OGG 的核心创新。设 $t-k$ 为 interrupt event 前最后一个 stable step。

$$\Delta Z_{\text{exp}} = M_\phi(Z_{t-k}^{\text{real}}, a_{t-k})$$

这是从 last stable step 预测的 expected residual，代表 "本来应该发生的 local dynamics"。

$$\Delta Z_{\text{dev}} = Z_t^{\text{real}} - Z_{t-k}^{\text{real}}$$

这是 t-k 到 t 实际累积的 deviation。

Corrective direction:
$$\Delta Z_{\text{corr}} = \Delta Z_{\text{exp}} - \Delta Z_{\text{dev}}$$

变量语义：
- $\Delta Z_{\text{exp}}$: intended local dynamics（"应该往哪走"）
- $\Delta Z_{\text{dev}}$: accumulated drift（"实际偏到哪了"）
- $\Delta Z_{\text{corr}}$: 补偿 drift 后的目标 dynamics 方向

直觉：保留 intended local dynamics 同时补偿 open-loop execution 期间累积的 drift。

### 6.5 Guided Velocity Update

$$\mathcal{L}_{\text{OGG}} = 1 - \text{CosSim}(\Delta \hat{Z}_{\text{act}}, \Delta Z_{\text{corr}})$$

Gradient injection:
$$v_\tau^{\text{guide}} = v_\tau - \eta \nabla_{v_\tau} \mathcal{L}_{\text{OGG}}, \quad A^{\tau - \Delta \tau} = A^\tau - \Delta \tau \cdot v_\tau^{\text{guide}}$$

变量解释：
- $\mathcal{L}_{\text{OGG}}$: OGG loss，衡量 predicted action effect 与 corrective direction 的 misalignment
- $\nabla_{v_\tau} \mathcal{L}_{\text{OGG}}$: loss 对 velocity field 的 gradient
- $\eta$: guidance strength (default 1.0)
- $v_\tau^{\text{guide}}$: 修正后的 velocity field

为什么 modify velocity 而非直接 perturb action coordinates？Paper 解释：modify velocity field 与 flow matching process 兼容，yields smoother corrective replanning。直接 perturb action 会破坏 flow trajectory 的连续性。

### 6.6 OGG 是 event-triggered

OGG **只在 interrupt event 后的 single policy call** 中激活。后续 policy calls 回到 standard inference，除非新的 interrupt 检测到。这避免了 OGG 的 gradient computation overhead 在所有 inference 上累积。

### 6.7 OGG 的思想源头

OGG 概念来自 Park et al. 2025 (ACG: Action Coherence Guidance, https://arxiv.org/abs/2510.22201)。VLA-Corrector 把它特化为 deviation-recovery 场景：用 expected-minus-actual 的方向作为 corrective target，而不是单纯的 coherence guidance。

类似 idea 还出现在 classifier-free guidance (CFG)、Bidirectional Decoding (Liu et al. 2024, https://arxiv.org/abs/2408.17355)。Bidirectional Decoding 也用未来和过去 observation 的约束来 guide action sampling，但 VLA-Corrector 用 latent residual M_φ 而非 raw observation。

---

## 7. 实验结果深度分析

### 7.1 Cross-Architecture Generalization (Table 1)

MetaWorld 上四个难度 split:

| Backbone | Baseline Avg | + Corrector Avg | Gain |
|----------|--------------|------------------|------|
| π0.5 | 48.70 | 64.35 | +15.65 |
| SmolVLA | 61.90 | 66.65 | +4.75 |
| X-VLA | 55.55 | 59.60 | +4.05 |

关键观察：π0.5 获益最大 (+15.65)，特别是 Very Hard split (+24.0)。这暗示 π0.5 在 hard tasks 上原本的 open-loop failure 模式最严重，VLA-Corrector 补救的空间最大。

Very Hard 数据：
- π0.5: 41.0 → 65.0 (+24.0)
- SmolVLA: 61.0 → 63.0 (+2.0)
- X-VLA: 55.0 → 64.0 (+9.0)

为什么 SmolVLA 在 Very Hard 上 gain 小？可能是 SmolVLA 本身 architecture 在 hard tasks 上就有不同 failure 模式（capacity 限制 vs open-loop drift），corrector 救不了 fundamental capacity 不足。

### 7.2 Sample Efficiency on LIBERO (Table 2)

| Model | Object | Spatial | Goal | Long | Avg |
|-------|--------|---------|------|------|-----|
| Full FT π0.5 | 99.4 | 98.2 | 97.8 | 92.4 | 96.95 |
| Few-shot FT π0.5 | 97.8 | 95.4 | 96.2 | 86.6 | 94.00 |
| Few-shot + Corrector | 99.8 | 100.0 | 98.0 | 93.4 | 97.80 |

**Few-shot + Corrector 超过 Full FT baseline (97.80 vs 96.95)**。这是一个很强的结果，说明：

1. Few-shot fine-tuning 已经学到了大部分 normal task trajectories，但缺乏 drifted states 和 recovery behaviors 的覆盖
2. VLA-Corrector 通过 inference-time 早期 interrupt drift + guided recovery，减轻了对 recovery training data 的依赖

这个 insight 很重要：传统思路是 collect 更多 recovery data 来 robustify policy，VLA-Corrector 走另一条路 — 在 inference 时主动 enable recovery，而不是 training 时 exposure recovery cases。

### 7.3 Data Efficiency of Corrector Training (Table 3)

| Ratio | Easy | Medium | Hard | Very Hard | Avg | Δ |
|-------|------|--------|------|----------|-----|---|
| Baseline | 70.54 | 45.00 | 38.33 | 41.00 | 48.72 | - |
| r=0.2 | 71.07 | 49.55 | 36.67 | 36.00 | 48.32 | -0.40 |
| r=0.4 | 70.36 | 45.91 | 38.33 | 42.00 | 49.15 | +0.43 |
| r=0.6 | 70.89 | 52.73 | 39.17 | 46.00 | 52.20 | +3.48 |
| r=0.8 | 71.07 | 53.64 | 38.33 | 46.00 | 52.26 | +5.54 |
| r=1.0 | 73.21 | 55.91 | 44.17 | 44.00 | 54.32 | +5.60 |

Diminishing returns 明显，r=0.6-0.8 已经饱和。这印证了 "M_φ 学的是 local on-track consistency 而非 full world model" 的定位 — 不需要 exhaustive coverage。

注意 r=0.2 时 avg 反而稍降 (-0.40)，说明 too little data 会让 M_φ 学到 noisy dynamics pattern，产生 false positive interrupts 反而伤害 performance。这是一个 failure mode 信号。

### 7.4 Performance-Efficiency Trade-off (Table 4)

最 striking 的数据：

**SmolVLA H=10**: 61.90% → 73.00% (+11.10), calls 19.27 → 15.64 (-3.63), eff gain +45.3%

这意味着 VLA-Corrector 让 SmolVLA 在更少的 policy calls 下达到更高的 success rate。理论上这只有在 "interrupt 后的 recovery 比继续 stale execution 更有效" 时才可能。

**X-VLA H=4**: 68.50 → 72.00 (+3.50), calls 46.58 → 35.20 (-11.38), eff gain +39.1%

X-VLA 在 H=4 时 calls 减少非常显著，但 success 增长相对温和。可能 X-VLA architecture 对 OGG 的响应不同，或者它的 default horizon 选择本身就更短。

**Long horizon 处收益最大**：π0.5 H=50 的 +24.6% 是各 horizon 中 efficiency gain 最大的，因为 long horizon 的 blind spot 最宽，corrector 救回的 stale execution 最多。

### 7.5 Mechanism Analysis

**LVM Detection (Fig 5, 6)**:
- Successful episodes: 低 $E_t$ 分布
- Failed episodes: heavier high-score tail，更多 interrupt events
- 83.7% truncation 发生在 critical phases，5.1× more truncations in critical than non-critical

这验证了 paper 的核心 claim：corrector 在 error-sensitive phases (precise grasping, alignment) 触发，在 tolerant phases (transport) 保留 long-horizon 效率。这是一个非常 clean 的验证。

**OGG Correction (Fig 7)**:
- Standard re-inference vs OGG-guided re-inference 在同一 truncation 后比较
- Recovery 定义: $E_t < T_{\text{off}}$ within next 10 steps
- OGG 在所有 difficulty 上都更好，average gain +0.23

**Controlled Recovery (Fig 8)**:
- 同一初始 state，同一 grasping error
- Baseline (LVM 监控但不 truncate): 继续 stale chunk → 不稳定 grasp → cup drop → failure
- VLA-Corrector: 检测 deviation → truncate → OGG replan → 恢复 stable grasp → success

这是一个 clean controlled comparison，isolate 了 truncate+OGG 的 causal effect。

### 7.6 Real-World Evaluation (Table 5, 15)

| Method | Pick-place | Alignment | Disturbance | Avg |
|--------|------------|-----------|-------------|-----|
| π0.5 Baseline | 70.0 ± 11.6 | 56.7 ± 12.5 | 40.0 ± 12.4 | 55.6 ± 7.3 |
| + VLA-Corrector | 78.3 ± 10.4 | 73.3 ± 11.2 | 68.3 ± 11.8 | 73.3 ± 6.5 |

Gain 模式：
- Pick-place: +8.3 (modest, 容忍度高)
- Alignment: +16.6 (precision-sensitive, accumulated error 危害大)
- Disturbance: +28.3 (online perturbation 让 stale chunk 完全 outdated)

这个 gain 顺序完美匹配 VLA-Corrector 的设计目标：preserve standard execution，improve precision tasks，最大 benefit 在 online disturbances。

Disturbance recovery 从 40% → 68.3% 是惊人的提升。在 moving-object grasp 任务：45.0 → 75.0 (+30.0)，moving-placement target: 40.0 → 70.0 (+30.0)，moving-insertion target: 35.0 → 60.0 (+25.0)。

Per-task 详情 (Table 15):
- 最难任务 moving-insertion target: 35% → 60%，绝对值仍然不高
- 这暗示 disturbance 越极端，单次 OGG recovery 越难

### 7.7 Ablation Studies

**Component Ablation (Table 6)**:
| Variant | Avg | Δ |
|---------|-----|---|
| Baseline | 48.70 | - |
| + Truncation Only | 60.35 | +11.65 |
| + Truncation + OGG | 64.35 | +15.65 |

Truncation 贡献 +11.65，OGG 在此基础上再加 +4.0。说明 stopping stale actions 是主要 effect，但 OGG 的 guided recovery 仍有 measurable value。

**Decoupled vs Coupled (Table 7)**:
| Detector | Avg |
|----------|-----|
| Internal Head + OGG | 49.55 |
| Decoupled LVM + OGG | 64.35 (+14.80) |

Decoupled 大幅胜出。Internal auxiliary head 会修改 backbone representations，污染原 action generation 行为。这是一个 anti-pattern 警示：在 generative VLA 上加 auxiliary head 要非常小心。

**OGG Guidance Strength (Table 8)**:
| η | Easy | Medium | Hard | Very Hard | Avg |
|---|------|--------|------|----------|-----|
| 0.1 | 82.7 | 56.8 | 45.2 | 66.0 | 62.68 |
| 1 (default) | 83.2 | 61.7 | 47.5 | 65.0 | 64.35 |
| 10 | 82.5 | 56.8 | 38.3 | 65.0 | 60.65 |
| 100 | 80.9 | 55.0 | 36.7 | 63.0 | 58.90 |

Sweet spot 在 η=1。η=100 时 Hard split 从 47.5 降到 36.7，说明过强 guidance 让 action 过度偏离 VLA prior，伤害 hard tasks 的 multi-modal action 表达。这印证了 paper 说的 "OGG 依赖 frozen π0.5 action prior"。

**LVM Capacity (Table 9)**:
| Monitor | Avg | Δ vs 10M |
|---------|-----|----------|
| LVM-10M | 56.58 | - |
| LVM-40M | 64.35 | +7.77 |
| LVM-160M | 64.28 | -0.07 |

40M 是 sweet spot，160M 几乎无额外 gain。Local dynamics prediction 不需要大 model。

**Cross-Domain Transfer (Table 10)**:
| Corrector Domain | MetaWorld Avg | Δ |
|------------------|---------------|---|
| Baseline | 48.7 | - |
| LIBERO-trained | 51.8 | +3.1 |
| MetaWorld-trained | 58.7 | +10.0 |

LIBERO-trained corrector 在 MetaWorld 上仍然 +3.1，说明 latent dynamics consistency signal 有一定 cross-domain transferability。但 domain-matched demonstrations 仍然重要 (+10.0 vs +3.1)。

### 7.8 Inference-Time Overhead (Table 11-13)

**Per-step overhead**:
- π0.5: 15.23 → 24.62 ms/step (+9.39, 1.62×)
- SmolVLA: 13.44 → 22.14 ms/step (+8.70, 1.65×)
- X-VLA: 8.66 → 14.55 ms/step (+5.89, 1.68×)

OGG 是 main source of overhead。Standard inference: 278.01 ms avg, OGG-guided recovery: 588.52 ms (2.12×)。

但这是 event-triggered: 只在 interrupt 后 single call 激活。Per-task OGG events: π0.5 71.28/task, SmolVLA 62.24/task, X-VLA 119.76/task。X-VLA 触发更频繁可能是因为它默认 horizon 更短，相对而言 corrector 占比更高。

---

## 8. Intuition Building: 为什么这个方法 Work

### 8.1 解耦的 Detect 和 Correct

VLA-Corrector 的核心架构智慧在于把 detection 和 correction 解耦：

- **Detection**：用 lightweight external module (40M MLP) 学 local dynamics consistency，在 frozen VLA features 上跑。这避免了修改 backbone representation 的污染风险。
- **Correction**：用 gradient guidance 在 flow matching velocity field 上注入 corrective signal，保持与原 VLA prior 的 compatibility。

这种解耦让两个组件各自优化而不互相干扰。如果用 internal head detection，会污染 action generation；如果用 strong action perturbation correction，会破坏 flow trajectory 连续性。

### 8.2 Event-Triggered vs Periodic Replanning

VLA-Corrector 本质上是一个 event-triggered 控制系统，类比 robotics 中的 event-based vision 和 event-triggered MPC。传统 fixed-horizon chunk 是 time-triggered 的，每隔 H 步必然 replan。Event-triggered 只在 dynamics deviation 持续时 replan。

这种思路在 control theory 有深厚传统，参见 Tabuada's event-triggered control work。VLA-Corrector 把它带到 VLA inference 层面，且用 learned latent dynamics model 作为 deviation detector。

### 8.3 Residual Dynamics 而非 Absolute World Model

M_φ 不预测 absolute future state，只预测 residual。这有几个深层好处：

1. **Static content suppression**：背景、未动对象在 residual 中被 subtract，模型 focus 在 task-relevant dynamics
2. **Lower-dimensional learning problem**：残差通常比绝对状态更低维
3. **Robust to encoder scale drift**：cosine-based E_t 对 magnitude 不敏感
4. **Data efficiency**：40M 足够，因为只学 local on-track pattern

这和 world model 路线 (DreamerV3, RNN-VLA) 的根本区别：world model 学完整 dynamics 用于 planning，M_φ 只学 consistency signal 用于 detection 和 lightweight guidance。

### 8.4 OGG 的 Constraint 是 Soft 的

OGG 通过 gradient injection 而非 hard constraint 来 guide action。这意味着：
- VLA prior 仍然是主 driver
- OGG 只是 bias velocity field 朝向 corrective direction
- 当 VLA prior 和 corrective direction conflict 时，η 控制 trade-off
- η=1 是 sweet spot，η 过大破坏 prior (Table 8)

这是 classifier-free guidance 思想的特化：CFG 用 condition 和 unconditional 的差来 scale，OGG 用 expected vs actual dynamics 的差来 inject corrective gradient。

### 8.5 Adaptive Horizon 的统计分析

$H_{\text{adaptive}}$ 实际上是一个 random variable，由 environment dynamics 决定。在 stable phases，$H_{\text{adaptive}} = H$ (full chunk 执行完)；在 drift phases，$H_{\text{adaptive}} < H$ (提前 truncate)。

期望意义上：
$$\mathbb{E}[H_{\text{adaptive}}] = H \cdot P(\text{stable}) + \mathbb{E}[h | \text{drift}] \cdot P(\text{drift})$$

Fig 6 显示 critical phases (drift-prone) 占比相对少，所以 $\mathbb{E}[H_{\text{adaptive}}]$ 仍然接近 H，保留了大部分效率。这正是 Table 4 中 calls 增加不显著的原因。

---

## 9. 局限性和 Future Directions

### 9.1 Paper 自己提到的 limitations (Appendix E.4)

1. **Reachability**: 如果 object 移出 reachable region，单次 OGG recovery 不够
2. **Force feedback 缺失**: 6-DoF arm without force feedback 在 contact-rich 任务中可能因为 friction/geometry 失败
3. **Visual ambiguity**: partial occlusion 或 poor contrast 会让 $E_t$ 信号 noisy
4. **OGG 依赖 prior**: 不能 generate backbone 不能 represent 的 recovery behavior

### 9.2 我观察到的潜在 issues

1. **M_φ 的 OOD 问题**: M_φ 在 demonstration trajectories 上训练，但 deployment 时遇到的 drifted states 可能 OOD。M_φ 在这些 state 上的 prediction 是否仍然可靠？Paper 没有显式 test 这个。

2. **Multi-step OGG**: 现在 OGG 只在 single call 后激活。如果一次 recovery 不够，是否需要 multi-step guided rollout？这类似 MPC 的 receding horizon。

3. **k 的选择**: paper 用 h1-k10 (history 1, predict 10 steps ahead)，但 k=10 是 fixed。在 fast dynamics 任务 (high velocity) 中 k=10 可能太长，slow dynamics 任务中太短。Adaptive k 是 future work。

4. **LVM 的 false negative**: 如果 M_φ 预测 error 本身就和 actual 一致 (both wrong in same way)，$E_t$ 会低，但实际已经 drift。这是 residual prediction 的潜在 failure mode。

5. **OGG gradient 的 instability**: $\nabla_{v_\tau} \mathcal{L}_{\text{OGG}}$ 在高维 velocity space 中可能 noisy，需要 gradient clipping 或 accumulation strategies。Paper 没讨论。

6. **Cross-embodiment transfer**: corrector 在一个 embodiment 的 demos 上训练，能 transfer 到另一个 embodiment 吗？X-VLA 强调 cross-embodiment，但 VLA-Corrector 没有测试这个。

### 9.3 延伸联想

**与 System 1 / System 2 思维的联系**: VLA-Corrector 类似 fast-in-slow (Chen et al. 2025, https://arxiv.org/abs/2506.01953) 的 dual-system 思路。Fast system (chunked VLA) 处理 routine execution，slow system (corrector + OGG) 在 anomaly 时介入。但 VLA-Corrector 的 slow system 是 lightweight 40M module，不是完整 reasoning system。

**与 Speculative Decoding 的联系**: speculative decoding (Leviathan et al. 2023) 在 LLM 中用小 model draft 大 model verify。VLA-Corrector 的 LVM 监控类似 verify 角色，但 verification 是基于 dynamics consistency 而非 token match。"Open-loop planning, closed-loop verification" (Wang et al. 2026b, https://arxiv.org/abs/2604.02965) 是类似思路。

**与 Diffusion Policy Recovery 的联系**: Bidirectional Decoding (https://arxiv.org/abs/2408.17355) 用 future goal 和 past observation 双向约束来 guide action sampling。VLA-Corrector 的 OGG 是单向的 (only past expected vs actual)，可以扩展到 bidirectional。

**与 Hierarchical RL 的联系**: VLA-Corrector 的 adaptive horizon 类似 hierarchical RL 中的 option termination。LVM 是 learned termination function，OGG 是 intra-option correction。

**与 Process Reward Models 的联系**: M_φ 在某种意义上是 process reward model 的 continuous 版本，评估 "是否 on track" 而非 final outcome。OpenAI 的 process reward model work (Lightman et al. 2023) 是类似思想在 LLM 上的应用。

**与 Control Barrier Functions 的联系**: 在 control theory 中，CBF 保证系统 stay in safe set。VLA-Corrector 的 $E_t < T_{\text{off}}$ 类似 CBF 的 invariant condition，但 learned 而非 hand-designed。

---

## 10. 实施细节和 Reproducibility

### 10.1 Hyperparameters Summary

| Component | Param | Value |
|-----------|-------|-------|
| LVM | Window size w | 15 |
| LVM | λ_on | 3.0 |
| LVM | λ_off | 2.0 |
| LVM | Patience p | 5 |
| LVM | Reset steps | 5 |
| LVM | Cooldown | 10 |
| LVM | Prediction horizon k | 10 |
| OGG | Guidance strength η | 1.0 |
| OGG | Activation | Single call post-interrupt |
| M_φ | Architecture | 4-layer MLP [2048]×4 |
| M_φ | Params | ~40M |
| Training | Optimizer | AdamW |
| Training | LR | 3e-4 |
| Training | Weight decay | 1e-4 |
| Training | Epochs | 30 (early stop p=5) |

### 10.2 Compute

- 8× NVIDIA A100-SXM4-40GB
- M_φ training lightweight (40M on frozen features)
- Online LVM: 1 forward pass per step
- OGG: gradient computation only on event-triggered recovery calls

### 10.3 Code

https://github.com/ZJU-OmniAI/vla-corrector

---

## 11. 相关工作的更广联系

### 11.1 Action Chunking 系列

- **ACT (Zhao et al. 2023)**: CVAE + transformer 预测 chunk，开创 action chunking paradigm
- **Diffusion Policy (Chi et al. 2023)**: diffusion 替代 CVAE，更强 multi-modal 表达
- **Mixture of Horizons (Jing et al. 2025)**: 多个 horizon mixture，仍然 static
- **Adaptive Action Chunking (Liang et al. 2026)**: inference-time adaptive，但用不同信号
- **Bidirectional Decoding (Liu et al. 2024)**: test-time sampling guidance
- **ACG (Park et al. 2025)**: action coherence guidance for flow-based VLA，OGG 的源头

### 11.2 VLA Backbone 系列

- **π0 (Black et al. 2024)**: flow matching VLA，3B params
- **π0.5 (Intelligence et al. 2025)**: open-world generalization 的 VLA
- **OpenVLA (Kim et al. 2024)**: open-source VLA based on Llama
- **SmolVLA (Shukor et al. 2025)**: affordable efficient VLA
- **X-VLA (Zheng et al. 2025)**: cross-embodiment VLA with soft prompts
- **GR00T N1 (Bjorck et al. 2025)**: humanoid foundation model
- **CogVLA (Li et al. 2025)**: cognition-aligned VLA
- **CoT-VLA (Zhao et al. 2025)**: chain-of-thought reasoning for VLA
- **FAST (Pertsch et al. 2025)**: efficient action tokenization

### 11.3 Recovery 和 Verification 系列

- **Self-improving Robots (Sharma et al. 2023)**: end-to-end RL with recovery
- **OOD Recovery (Gao et al. 2025)**: object-centric keypoint inverse policy
- **Latent Policy Barrier (Sun and Song 2025)**: stay in-distribution
- **RAC (Hu et al. 2025)**: scaling recovery and correction
- **Robust Imitation via Learning to Search (Jain et al. 2025)**: search-based recovery
- **Action Draft and Verify (Zhao et al. 2026)**: self-verifying framework
- **VLA-in-the-loop (Xu et al.)**: online policy correction with world models
- **Closed-loop Action Chunks (Wu et al. 2026a)**: training-free diffusion policy correction

### 11.4 World Model 系列

- **RNN-VLA (Cen et al. 2025)**: unified VLA and world model
- **DreamerV3**: latent world model for planning
- **VLA-Corrector 的定位**: 不是 full world model，是 local dynamics consistency module

---

## 12. 总结：VLA-Corrector 的核心贡献

1. **量化 action horizon trade-off**: Fig 2 和 Table 4 系统展示了 open-loop blind spot 跨 backbone 一致存在
2. **Modular detect-and-correct framework**: LVM + Event-Triggered Truncation + OGG 三层结构，不动 backbone
3. **Lightweight external M_φ**: 40M MLP 学 residual dynamics consistency，data-efficient
4. **Adaptive horizon**: 通过 event-triggered truncation 把 fixed H 变成 H_adaptive
5. **OGG corrective replanning**: gradient guidance 在 flow matching velocity field 上 inject corrective signal
6. **Cross-architecture generalization**: π0.5, SmolVLA, X-VLA 都获益
7. **Sample efficiency**: few-shot + corrector 超过 full FT baseline

最深的 intuition 是：**问题不在于选择更好的固定 horizon，而在于决定什么时候当前的 chunk 不再可信**。这个 reframing 把 research focus 从 "find optimal H" 转向 "learn when to distrust current H"，这是 paper 的核心 intellectual contribution。

更多 reference:
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054  
- SmolVLA: https://arxiv.org/abs/2506.01844
- X-VLA: https://arxiv.org/abs/2510.10274
- ACT: https://arxiv.org/abs/2304.13705
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Flow Matching: https://arxiv.org/abs/2210.02747
- MetaWorld: https://arxiv.org/abs/1910.10897
- LIBERO: https://arxiv.org/abs/2306.03310
- ACG (OGG source): https://arxiv.org/abs/2510.22201
- Bidirectional Decoding: https://arxiv.org/abs/2408.17355
- LeRobot: https://arxiv.org/abs/2602.22818
- OpenVLA: https://arxiv.org/abs/2406.09246
- GR00T N1: https://arxiv.org/abs/2503.14734
- Fast-in-Slow: https://arxiv.org/abs/2506.01953
- Code: https://github.com/ZJU-OmniAI/vla-corrector
