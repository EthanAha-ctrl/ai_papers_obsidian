---
source_pdf: Real-Time Execution of Action Chunking Flow Policies.pdf
paper_sha256: dd6d602be1f50b894d88517bf68916c4272885b2e29d96cffc426dacff41898f
processed_at: '2026-08-11T21:15:44-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版:RTC到底在干嘛

## 现实世界的麻烦

想象你在开车。你看一眼路况,决定接下来5秒怎么打方向盘,然后闭眼5秒执行。执行完了再睁眼看路,再规划下一个5秒。这听起来很蠢对吧?但这就是现在大部分robot VLA的工作方式。

为什么?因为big VLA model太慢了。$\pi_0$有3B参数,生成一个action chunk要将近100ms。但robot control要50Hz,每20ms就得有一个action。所以model想一次,够robot用好几个step。

这就造成两个问题:

**问题1 - 等待的尴尬:** 你想完了,robot也执行完了,robot就停在那里等你下一个chunk。这叫synchronous inference。机器人会一顿一顿的,动作不流畅。

**问题2 - 接缝的尴尬:** 你为了不让robot等,提前开始想下一个chunk。但你想的时候,robot还在执行上一个chunk的尾巴。等你想完,robot执行到哪你管不了,结果你规划的新动作和robot当前状态可能完全不搭 - 比如上一个chunk要往左走绕开障碍,新chunk决定往右走绕开,中间就是一个巨大的突变。这叫mode jumping,robot会抽搐一下。

## 以前的烂解法

**Temporal Ensembling (TE):** 把好几个chunk的predictions平均一下。听起来合理,但实际上action distribution是multi-modal的 - 往左绕和往右绕都是valid,平均出来是直直撞上去。所以TE在multi-modal任务上直接崩掉。

**Bidirectional Decoding (BID):** 生成64个candidates,挑一个和上一个chunk最consistent的。能work,但计算量巨大 - 你得跑64次inference。

## RTC的聪明做法

RTC的核心insight特别elegant:**把这个问题看成image inpainting**。

想象你有一张照片,中间挖掉一块,你让AI把挖掉的部分fill back in,AI会根据周围的context生成一个consistent的filling。这正好是diffusion/flow model擅长的事情。

映射到robot control:
- "照片" = action chunk (一串future actions)
- "挖掉的部分" = 需要新预测的部分
- "没挖掉的部分" = 我们已经知道必须执行的actions (因为它们在新chunk算完之前就会被执行掉)

具体来说: 假设你提前$d$步开始算下一个chunk。那么新chunk的前$d$个actions,等到新chunk算完时早就执行过了 - 这$d$个actions是**确定**的,必须和上一个chunk的对应部分一模一样。所以你"freeze"它们,让model去inpaint剩下的部分,并且保证和frozen部分consistent。

**这是为什么比BID聪明:** BID是sample一堆然后选,是discrete的,expensive的。RTC是在generation过程中直接用gradient去steer model,让生成的action chunk天然就是consistent的,是continuous的,cheaper的。

**这是为什么比TE聪明:** TE是平均,在multi-modal下average是invalid的。RTC是用generative model的prior去generate一个consistent的sample,不是average,所以multi-modal没问题。

## Soft Masking的妙处

光freeze前$d$个actions还不够。如果$d$很小(比如就1),guidance signal太弱,model还是可能跳到另一个mode。

RTC的进一步招数: 不光freeze前$d$个actions,还把中间**所有overlapping的actions**都用作guidance,但用**decay的weight**。

为什么decay?因为这些中间actions虽然上一个chunk有预测,但等到新chunk算完时,robot可能执行到了,也可能没执行到 - 越靠后的越uncertain,所以weight越小,给model越多freedom。

这本质上是**uncertainty-aware guidance**,类似Kalman filter里给noisy measurement更少weight的思路。

## 真实世界效果

在$\pi_{0.5}$上测试,6个bimanual manipulation tasks,包括点蜡烛(拿火柴,划火柴,点蜡烛)这种很precision的任务。

关键发现:
- **不加延迟时**: RTC和synchronous差不多,但动作更smooth,完成更快
- **加100ms延迟时**: Synchronous开始degrade,TE直接崩(机器人都protective stop了),RTC纹丝不动
- **加200ms延迟时**: 差距更明显,RTC依然robust

最有意思的是,RTC不只是快 - 即使把inference等待时间扣掉,RTC也比synchronous完成得快。说明smooth的continuity本身让policy犯更少错,少retry。

## 为什么这个方法重要

现在VLA越做越大,从$\pi_0$的3B到$\pi_{0.5}$更大,未来还会更大。越大越慢,但real-time control的$\Delta t$不会变。所以**asynchronous inference是唯一出路**,而如何处理chunk之间的continuity是核心难题。

RTC给出一个clean solution:
- Training-free (不改model,直接inference time用)
- 对latency robust (模型再慢一点也能用)
- 利用flow/diffusion model的inherent inpainting能力

这暗示了flow-based VLA相对于autoregressive VLA的一个深层advantage: iterative denoising过程允许你在中间任何一步inject guidance,这是autoregressive model做不到的。Autoregressive生成是left-to-right,不能回头改;flow是iterative refinement,可以随时inject condition。这个property可能在未来更复杂的应用中越来越重要。

## 一句话总结

**RTC = 把asynchronous action chunking看成inpainting problem,freeze掉确定要执行的前缀,用flow model的generative prior + soft masking的uncertainty weighting,生成和previous chunk consistent的new chunk,training-free而且对latency robust。**

---

# Real-Time Execution of Action Chunking Flow Policies - 技术讲解

## 1. Core Problem: 为什么需要RTC

这篇paper来自Physical Intelligence (Kevin Black, Manuel Galliker, Sergey Levine), 解决的是VLA (Vision-Language-Action) model在real-time control中的**latency**问题。核心insight是: 把异步action chunking重新formulate成一个**inpainting problem**。

**问题的本质:**
- Modern VLA (如$\pi_0$, $\pi_{0.5}$, OpenVLA) 参数量达到billions级别 ([pi0 paper](https://arxiv.org/abs/2410.24164), [OpenVLA](https://arxiv.org/abs/2406.09246))
- 例如3B参数的$\pi_0$, 在RTX 4090上KV cache prefill就要46ms, 而50Hz control要求$\Delta t = 20$ms
- Network latency (LAN) 大概13-20ms, 实际deployment中很容易超过20ms
- 所以$\delta > \Delta t$是常态, 必须用asynchronous inference

## 2. Action Chunking 基础

Action chunking policy定义为$\pi(\mathbf{A}_t | \mathbf{o}_t)$, 其中:
- $\mathbf{A}_t = [\mathbf{a}_t, \mathbf{a}_{t+1}, ..., \mathbf{a}_{t+H-1}]$ 是一个chunk of future actions
- $\mathbf{o}_t$ 是observation
- $H$ 是**prediction horizon** (预测时域)
- $s \leq H$ 是**execution horizon** (执行时域), 只有前$s$个action被实际执行

典型设置$s \approx H/2$, 例如ACT ([Zhao et al.](https://arxiv.org/abs/2304.13705)), $\pi_0$, $\pi_{0.5}$都是这样。

**Trade-off:**
- Long $s$ → temporal consistency, 但失去对new observation的reactivity
- Short $s$ → 容易在chunk边界发生mode jumping, 导致jerky behavior

## 3. Flow Matching 基础

这篇paper考虑用conditional flow matching ([Lipman et al.](https://arxiv.org/abs/2210.02747))训练的policy。生成action chunk的过程是从Gaussian noise $\mathbf{A}_t^0$开始, 积分velocity field $\mathbf{v}_\pi$:

$$\mathbf{A}_t^{\tau + \frac{1}{n}} = \mathbf{A}_t^\tau + \frac{1}{n} \mathbf{v}_\pi(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau)$$

变量解释:
- $\tau \in [0, 1)$ 是flow matching timestep (从noise到data的进程)
- $n$ 是denoising steps数量
- $\mathbf{v}_\pi$ 是学到的velocity field (神经网络)
- $\mathbf{A}_t^0$ 是noise, $\mathbf{A}_t^1$ 是最终生成的action chunk

注意diffusion policy也可以在inference time转换为flow policy ([Gao et al.](https://diffusionflow.github.io/)), 所以RTC适用于任何diffusion或flow-based VLA。

## 4. Real-Time Constraint的数学描述

定义:
- $\Delta t$ = controller sampling period (例如20ms对应50Hz)
- $\delta$ = 生成一个action chunk的时间
- $d := \lceil \delta / \Delta t \rceil$ = **inference delay** (controller timesteps)

Real-time条件: 必须在固定时间$\Delta t$内产生response。

如果$\delta \leq \Delta t$, trivial。但modern VLA不可能。所以必须asynchronous: 在执行上一个chunk的同时, 生成下一个chunk。

Asynchronous算法必须在$t = s - d$开始inference (即执行$s-d$步后就开始算下一个chunk)。条件是$d \leq H - s$。但问题是: 在$t = s-d$时无法知道$t = s-d$到$t = s$之间会发生什么, 所以$\mathbf{a}_{s-1|0}$ (上一个chunk的最后一个action) 和$\mathbf{a}_{s|s-d}$ (新chunk的第一个action)之间可能arbitrarily discontinuous, 这是out-of-distribution的根源 (见Figure 2)。

## 5. RTC的核心: Inference-Time Inpainting with Flow Matching

**核心insight:** 把real-time chunking看成一个inpainting problem。

想象一下: 当我们在生成新chunk $\mathbf{A}_{s-d}$时, 上一个chunk $\mathbf{A}_0$的某些actions还remaining (没执行)。这些remaining actions我们**知道会被执行**(因为它们是新chunk生成完之前唯一的action来源), 所以可以"freeze"它们, 让新chunk去"fill in"其余部分, 类似于image inpainting。

具体算法基于Pokle et al. ([Training-free linear image inverses via flows](https://arxiv.org/abs/2310.04432))和ΠGDM ([Song et al.](https://arxiv.org/abs/2305.04391))。

在每一个denoising step, 添加一个gradient-based guidance term到velocity field:

$$\mathbf{v}_{\Pi\text{GDM}}(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau) = \mathbf{v}(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau) + \min\left(\beta, \frac{1-\tau}{\tau \cdot r_\tau^2}\right) (\mathbf{Y} - \hat{\mathbf{A}}_t^1)^\top \text{diag}(\mathbf{W}) \frac{\partial \hat{\mathbf{A}_t^\tau}}{\partial \mathbf{A}_t^\tau}$$

其中:
- $\hat{\mathbf{A}_t^1} = \mathbf{A}_t^\tau + (1-\tau)\mathbf{v}(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau)$ 是对最终denoised chunk的估计 (one-step估计)
- $r_\tau^2 = \frac{(1-\tau)^2}{\tau^2 + (1-\tau)^2}$ 是一个noise-dependent scaling
- $\mathbf{W}$ 是mask (后面soft masking会用到)
- $\mathbf{Y}$ 是target (前一个chunk的对应位置)
- $\beta$ 是guidance weight clipping (paper的addition)

**关键变量解析:**
- $\hat{\mathbf{A}_t^1}$的intuition: 给定当前state $\mathbf{A}_t^\tau$, 如果我们linear extrapolate velocity field, 估计的final action chunk。这是Taylor展开的一阶近似。
- $\frac{\partial \hat{\mathbf{A}_t^\tau}}{\partial \mathbf{A}_t^\tau}$ 是Jacobian, 衡量当前state对final estimate的影响
- $(\mathbf{Y} - \hat{\mathbf{A}}_t^1)$ 是误差: 估计的final chunk与target之间的差
- $\text{diag}(\mathbf{W})$ 对error加权: 哪些位置需要match, 哪些不需要

**Guidance weight clipping $\beta$:**
Paper发现, 原始ΠGDM的weight $\frac{1-\tau}{\tau \cdot r_\tau^2}$在$\tau \to 0$时diverge, 而control问题通常用很少的denoising steps (e.g. $n=5$), 这会导致instability。所以加$\min(\beta, \cdot)$。Ablation显示$\beta = 5$是optimal (Figure 7)。太大会导致jerkiness, 太小guidance不够强。

## 6. Soft Masking: 关键创新

仅仅用hard masking (前$d$个action weight=1, 其余=0) 在$d$小的时候不够强, 仍然会产生mode switching (Figure 4)。

**Soft masking的idea:** 考虑**所有**$H - s$个overlapping actions, 而不仅是前$d$个。

Mask weight $\mathbf{W}_i$的定义:

$$\mathbf{W}_i = \begin{cases} 
1 & \text{if } i < d \\
c_i \frac{e^{c_i} - 1}{e - 1} & \text{if } d \leq i < H - s \\
0 & \text{if } i \geq H - s
\end{cases}$$

其中 $c_i = \frac{H - s - i}{H - s - d + 1}$, $i \in \{0, ..., H-1\}$

**Intuition:**
- 前$d$个actions: weight=1, 因为这些是**frozen**的, 必须match (它们一定会在新chunk生成完之前执行)
- 中间$[d, H-s)$个actions: weight按exponential decay从1到0。因为这些actions虽然overlapping, 但新chunk生成完之前不一定执行到, 越往后越uncertain, 所以weight越小
- 最后$s$个actions: weight=0, 因为这些是新chunk独有的部分, 没有overlap

这个exponential decay的形式$c_i \frac{e^{c_i}-1}{e-1}$是一个normalized exponential, 在$c_i=1$时为1, $c_i=0$时为0, 中间平滑过渡。Ablation (Figure 8)显示exponential和linear decay接近, 但exponential略好。

**为什么soft masking有效?**
对于很小的$d$ (例如$d=1$), hard masking几乎没guidance signal。Soft masking让所有overlapping actions都贡献guidance, 帮助新chunk"看到"前chunk的整个trajectory, 从而保持consistent strategy。

## 7. 完整Algorithm (Algorithm 1)

算法是multi-threaded的, 包含两个部分:

**GetAction(o_next):** Controller每隔$\Delta t$调用一次, 返回当前要执行的action。Acquires mutex, 更新observation, notifies condition variable。

**InferenceLoop:** 后台thread运行。流程:
1. Wait直到至少$s_{\min}$个actions被执行
2. 设$s = t$ (已执行数), 取previous chunk的剩余部分$\mathbf{A}_{\text{prev}} = \mathbf{A}_{\text{cur}}[s:s+1:...:H-1]$
3. 读取当前observation $\mathbf{o}$
4. 预测下一个delay: $d = \max(\mathcal{Q})$ (conservative estimate, 取past delays buffer的最大值)
5. 调用GuidedInference生成新chunk
6. Swap to new chunk, reset $t$

**GuidedInference:**
1. Compute mask $\mathbf{W}$ via Eq. 5, right-pad $\mathbf{A}_{\text{prev}}$到长度$H$, 初始化$\mathbf{A}^0 \sim \mathcal{N}(0, \mathbf{I})$
2. For each denoising step:
   - 定义denoising function $f_{\hat{\mathbf{A}}^1}(\mathbf{A}') = \mathbf{A}' + (1-\tau)\mathbf{v}_\pi(\mathbf{A}', \mathbf{o}, \tau)$
   - Compute weighted error: $\mathbf{e} = (\mathbf{A}_{\text{prev}} - f_{\hat{\mathbf{A}}^1}(\mathbf{A}^\tau))^\top \text{diag}(\mathbf{W})$
   - Compute vector-Jacobian product via autodiff: $\mathbf{g} = \mathbf{e} \cdot \frac{\partial f_{\hat{\mathbf{A}}^1}}{\partial \mathbf{A}'}\bigg|_{\mathbf{A}' = \mathbf{A}^\tau}$
   - Update: $\mathbf{A}^{\tau + 1/n} = \mathbf{A}^\tau + \frac{1}{n}\left(\mathbf{v}_\pi(\mathbf{A}^\tau, \mathbf{o}, \tau) + \min\left(\beta, \frac{1-\tau}{\tau \cdot r_\tau^2}\right)\mathbf{g}\right)$

**关键实现细节:**
- Vector-Jacobian product (VJP) 用reverse-mode autodiff ([Baydin et al.](https://jmlr.org/papers/v18/17-468.html)), 不是full Jacobian, 效率高很多
- 每个denoising step需要backprop一次, 所以RTC的latency是vanilla的~2.5x (Table 3: 14ms → 35ms per denoising step)

## 8. Simulated Benchmark: Kinetix

现有的simulated imitation learning benchmark大多是quasi-static, 用long execution horizon的pseudo open-loop就能接近完美 ([Diffusion Policy](https://arxiv.org/abs/2304.13705))。所以作者创建了新的benchmark: 12个dynamic tasks in [Kinetix](https://arxiv.org/abs/2410.23208)。

- Force-based control, 所以inference delay必然需要asynchronous (没有"holding position"概念)
- 包含throwing, catching, balancing等dynamic motions
- 加Gaussian noise到actions, 模拟imperfect actuation
- 数据: 用RPO ([Rahman & Xue](https://arxiv.org/abs/2212.07536))训练expert policy, 6 seeds × 12 environments, 1M transitions
- Flow policy: $H=8$, 4-layer MLP-Mixer ([Tolstikhin et al.](https://proceedings.neurips.cc/paper/2021/hash/c164de6c1e94740e0c9995f0975999a5-Abstract.html))
- 评估: 2048 rollouts per data point, delay从0到4

**Baselines:**
- **Naive async:** 不考虑previous chunk, 直接switch
- **BID (Bidirectional Decoding, [Liu et al.](https://arxiv.org/abs/2408.17355)):** 用rejection sampling保持continuity, batch size $N=32$
- **TE (Temporal Ensembling, [Zhao et al.](https://arxiv.org/abs/2304.13705)):** 维持action chunk buffer, 执行平均

**Results (Figure 5):**
- TE在所有delay下都差, 因为benchmark是multi-modal的, average of valid actions不一定是valid action
- RTC对delay最robust, 优于BID (BID用64个samples, 32 strong + 32 weak, 计算量大很多)
- Soft masking > hard masking, 特别是$d$小的时候
- RTC能更好地利用closed-loop corrections, $s$减小performance增加

## 9. Real-World Experiments

Base policy: $\pi_{0.5}$ VLA ([pi0.5 paper](https://arxiv.org/abs/2504.16054)), $H=50$, $\Delta t = 20$ms, $n=5$ denoising steps
- Vanilla latency: 76ms
- RTC latency: 97ms (由于backprop, 见Table 3)
- LAN inference加10-20ms, 所以baseline $d \approx 6$, RTC $d \approx 6$
- 还测试了+100ms和+200ms injected latency ($d \approx 11, 16$)

**Tasks:**
1. **Light candle** (5 steps, 40s): 拿match, 划火柴, 点蜡烛, 丢碗里
2. **Plug ethernet** (6 steps, 120s): 拿ethernet cable, 重新orient, 插server rack, 另一头重复
3. **Make bed, mobile** (3 steps, 200s): 移blanket和2 pillows
4. **Shirt folding** (1 step, 300s): 折衣服
5. **Batch folding** (4 steps, 300s): 从bin里拿crumpled衣服, 展平, 折, 叠pile
6. **Dishes in sink, mobile** (8 steps, 300s): 4个物品从counter移到sink

总共480 episodes, 28小时pure robot execution。

**Baselines:**
- **Synchronous:** 默认策略, 执行$s=25$然后暂停等新chunk
- **TE, sparse:** $s=25$, parallel inference, 加TE
- **TE, dense:** 尽量频繁inference, $s=d$

**Results (Figure 6):**
- **Average throughput** (proportion of task completed / duration): RTC在所有delay下最好, +100ms和+200ms有statistically significant advantage
- RTC对injected delay完全robust, 不degrade
- Synchronous linearly degrades with delay
- 两个TE variants在+100/+200ms时根本跑不起来, 因为oscillation太大触发robot protective stop
- 即使去除inference pauses, RTC也比synchronous快 - 反映fewer mistakes和less retrying
- **Light candle** (最precision-sensitive, 唯一无retry): RTC在final score上有大优势
- **Bed making** (最难): RTC同样有强效

## 10. Latency Breakdown (Tables 1-3)

| Method | Latency |
|--------|---------|
| RTC | 97ms |
| BID (N=16, no forward model) | 115ms |
| BID (N=16, shared backbone) | 169ms |
| BID (N=16, full) | 223ms |
| Vanilla $\pi_{0.5}$ | 76ms |

RTC比vanilla慢约28%, 但比full BID快2.3x, 性价比高。

Per-component breakdown (mobile case):
- Model: 96.89ms
- Network: 21.20ms  
- Image resize: 11.22ms
- Other: 9.67ms
- Total: 138.98ms

RTC的overhead主要来自denoising step的backprop: 14ms → 35ms (2.5x)

## 11. 我的Intuition和理解

### 11.1 这个方法的本质

这个方法的本质是: **用generative model的conditional sampling能力来enforce continuity**。Diffusion/flow model本来就有很强的inpaiting能力 (可以从partial information生成consistent完整结果), 这里把frozen actions作为condition, 让model "complete"剩余部分。

这比TE (averaging) 高明得多, 因为:
- TE假设action distribution是unimodal, average有意义
- 但实际manipulation tasks是multi-modal的, 平均两个valid actions可能是invalid
- RTC通过generative process, 生成一个consistent的、符合prior的action chunk

### 11.2 为什么比BID好

BID ([Bidirectional Decoding](https://arxiv.org/abs/2408.17355))用rejection sampling: 生成多个samples, 选一个和previous chunk最consistent的。这种方法:
- 计算量大 (batch size 64)
- Discrete selection, 不smooth
- 弱model sampling增加了noise

RTC用gradient guidance在generation过程中**直接**steer toward consistency, 是continuous optimization, 更高效也更smooth。

### 11.3 Soft Masking的深层含义

Soft masking实际上是一种**uncertainty-aware guidance**。越远的future action越uncertain, 所以weight越小, 给model更多freedom。这类似于particle filter中的confidence weighting, 也类似于Kalman filter中的measurement noise covariance - 越noisy的measurement给越少weight。

这种设计让RTC在$d$很小 (low latency) 和$d$很大 (high latency) 时都能work, 因为:
- $d$小: 大部分guidance来自soft region, 但仍是strong signal
- $d$大: 大部分guidance来自hard region, strongly anchored

### 11.4 和MPC的对比

Paper提到MPC ([Rawlings et al.](https://books.google.ch/books?id=MrJctAEACAAE)) 也用receding horizon + warm-starting, 但MPC需要explicit dynamics model和cost function。RTC可以看作是**learned MPC的implicit版本**:
- Dynamics model: VLA内部learned
- Cost function: 由imitation learning implicitly定义
- Warm-start: soft masking用previous chunk

这种connection暗示了可能可以从MPC理论borrow更多技术, 例如constraint handling, stability analysis。

### 11.5 关于Hierarchical VLA (System 1/2)

Paper提到Gemini Robotics ([Gemini Robotics Team](https://arxiv.org/abs/2503.20020))和$\pi_0$的hierarchical设计: System 2 (high-level planning) 慢但capable, System 1 (low-level action) 快但简单。这种方法和RTC是orthogonal的, 可以combine: System 2用RTC保持real-time, System 1更reactive。

### 11.6 Limitations

Paper自己提到:
- Computational overhead (2.5x per denoising step)
- 只适用于diffusion/flow-based policy
- 没有测试legged locomotion等更dynamic setting

我想到的额外limitations:
- **Delay estimate的robustness**: 用$\max(\mathcal{Q})$保守估计, 但如果delay突然增加 (例如network hiccup), 可能fail
- **Soft masking的schedule**: exponential decay是heuristic, 是否有更principled的选择? 例如基于uncertainty propagation
- **第一个chunk**: 初始化$\mathbf{A}_{\text{init}}$如何处理? 如果是从rest state开始, 没问题, 但如果中途takeover, 可能需要warmup
- **Exploration**: 这个方法完全inference-time, 但训练时如果知道会用RTC, 是否可以optimize训练过程?

### 11.7 对整个VLA领域的启示

这篇paper给VLA领域几个重要启示:
1. **Inference-time algorithm和training同样重要** - RTC不需要retraining就能显著improve performance
2. **Real-time constraint是deployment的关键** - 不能只看success rate, 还要看latency tolerance
3. **Benchmark design很重要** - quasi-static benchmark已经saturation, dynamic benchmark才能区分methods
4. **Asynchronous execution是未来** - 随着model越来越大, synchronous inference不可能keep up

### 11.8 可能的extension

基于这篇paper, 我能想到的extension:
1. **Learned soft masking schedule**: 用learned uncertainty (e.g. ensemble) 来determine weights
2. **Adaptive execution horizon**: 根据task difficulty动态调整$s$
3. **Multi-step lookahead guidance**: 不仅用previous chunk, 还用predicted future observations
4. **Combine with MPC**: 用VLA作为warm-start, 用MPC做local refinement with explicit dynamics
5. **Closed-loop with proprioception**: 当前只用vision observation, 可以加入force/torque feedback
6. **Application to locomotion**: paper提到这是future work, legged locomotion应该会benefit更多
7. **Real-time VLA training**: 在训练时就simulates inference delay, 让model更robust

## 12. 参考链接

- [Paper PDF (arxiv)](https://arxiv.org/abs/2504.16054)
- [Project page](https://pi.website/research/real_time_chunking)
- [GitHub (Kinetix experiments)](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
- [$\pi_0$ paper](https://arxiv.org/abs/2410.24164)
- [$\pi_{0.5}$ paper](https://arxiv.org/abs/2504.16054)
- [ACT (Action Chunking Transformer)](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy](https://arxiv.org/abs/2304.13705)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [ΠGDM](https://arxiv.org/abs/2305.04391)
- [Training-free flow inverses (Pokle et al.)](https://arxiv.org/abs/2310.04432)
- [Bidirectional Decoding](https://arxiv.org/abs/2408.17355)
- [Kinetix](https://arxiv.org/abs/2410.23208)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [Diffusion meets Flow Matching](https://diffusionflow.github.io/)
- [Autodiff survey](https://jmlr.org/papers/v18/17-468.html)
- [MLP-Mixer](https://proceedings.neurips.cc/paper/2021/hash/c164de6c1e94740e0c9995f0975999a5-Abstract.html)
- [Gemini Robotics](https://arxiv.org/abs/2503.20020)
- [Diffuser (Janner et al.)](https://arxiv.org/abs/2205.09991)
- [MPC textbook](https://books.google.ch/books?id=MrJctAEACAAE)

## 13. 总结

RTC是一个非常elegant的inference-time algorithm, 把asynchronous action chunking重新formulate为inpainting problem, 充分利用了flow/diffusion model的generative capability。核心贡献:
1. **Algorithm**: 基于ΠGDM的training-free inpainting, 加soft masking和guidance clipping
2. **Benchmark**: Kinetix-based dynamic tasks
3. **Real-world validation**: 6个bimanual manipulation tasks, 包括highly dexterous的light candle

这个方法对整个VLA deployment有重要意义, 因为它让large VLA能在real-time控制中使用, 而且对inference delay有uniquely robust的tolerance - 这对于cloud-based inference, model scaling, edge deployment都至关重要。从更深层次看, 它揭示了**inference-time algorithm**和**generative model structure**的synergy: flow/diffusion model的iterative denoising过程本质上就是一个"可控"的generation过程, 可以在中间任何步骤inject guidance, 这是autoregressive model做不到的。这可能是flow-based VLA相对于autoregressive VLA的一个重要advantage, 值得进一步探索。
