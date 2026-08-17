---
source_pdf: ROSA A Robotics Foundation Model Serving System for Robot Factories.pdf
paper_sha256: e38ee431e62d498c9c539fa9a3561f7d9e26d8dfe66d64a57ea0ce7eeb7effa5
processed_at: '2026-08-12T02:20:29-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ROSA 人话版

## 一句话概括

**别给每个 robot 配一张 GPU，搞个 GPU 池子大家一起用，然后用一个聪明 scheduler 决定谁什么时候用多少。**

---

## 之前大家怎么想的

两个 default 假设，ROSA 说都错了：

**假设一：robot 推理是 edge 问题，每台 robot 自己背个 GPU 干活。**

问题：GPU 在 robot 执行 action 的那 200ms 里是 idle 的，一个 robot 也凑不出 batch，GPU 利用率极低。而且 onboard GPU 跟 datacenter GPU 性能差几十倍，大模型根本跑不动。还有个现实问题——humanoid 电池一半功率都被 GPU 吃了，offload 掉直接多跑几个小时。

**假设二：serving 的目标是最小化单次推理延迟。**

问题：factory 不在乎单个 action 多快，在乎整条线产出多少合格 action。延迟压到 50ms 和 100ms 对 task success 没区别，但省下的 GPU 能多服务一倍 robot。而且 robot 真正干活不是一个 model，是一堆 model 配合——action model + planner + safety checker + task monitor，只优化 action model latency 是 local optimum。

---

## ROSA 的三个核心想法

### 想法一：GPU 池子，不要一人一张

Robot 上只留一个小 SoC 跑 100Hz 低层控制和 safety fallback，推理全 offload 到 factory 里一堆 H200 上。Robot 发 observation 过网络，server 算完发 action 回来。WiFi 7 延迟 sub-5ms，跟推理延迟几十到几百 ms 比可忽略。

好处：GPU 不闲着了（A robot 执行 action 时 B robot 在推理，还能把 C/D/E 的请求拼成一个 batch），battery 更持久，total GPU 数也更少（不用每个 robot 都配）。

### 想法二：一个 task 是一串 model，不是一个 model

ROSA 让你用 YAML 声明一个 task 需要哪些 model、各自频率多少、SLO 多严、出错了怎么办。典型四个组件：

- **System 1**（action model）：高频，比如每 200ms 一次，生成 action chunk
- **System 2**（planner）：低频，比如每 10 次 System 1 调一次，分解长任务
- **Safety**：周期性检查，比如 2Hz，问 VLM "这个 workcell 安全吗"
- **Monitor**：周期性检查，比如 0.5Hz，问 VLM "task 是 ongoing/done/failed"

System 1 + System 2 是一条 goal-coupled 链——加速它们直接提升 action 产出率。Safety 和 Monitor 是 obligation——跑够要求就行，加再多 GPU 也不多产 action。

这个区分很关键，直接决定 resource 怎么分。

### 想法三：优化 factory 产出，不优化单次延迟

Objective 不是 "让某个 robot 的 action model latency 最小"，是 "整条线最多多少合格 action"。一个 action 算不算合格要看是否满足 SLO（比如 P99 < 200ms），超 SLO 的 action 算 0 价值，因为 stale observation 不安全。

---

## Scheduler 怎么干活的

这是论文最硬核的部分，用大白话讲就是四步：

### 第一步：给 obligation model 分最少 GPU

Safety 和 monitor 有固定频率要求（比如 safety 2Hz、monitor 0.5Hz），把所有 robot 的需求加起来，用 profiling 数据算出最少要几张 GPU 才能让 P99 满足 SLO。剩下的 GPU 全留给 System 1 和 System 2。

### 第二步：搜最大 action rate f

Binary search 一个共享 action rate f。对每个候选 f：

1. **枚举 feasible server configuration**：一个 server 跑哪个 model、batch size 多少、服务几个 robot stream。用 profiling 数据筛掉超 SLO 的组合。
2. **ILP packing**：用整数线性规划选一组 configuration 覆盖所有 robot，不超过 GPU 预算。
3. **Closed-loop check**：robot 是 closed-loop（发请求→等推理→执行 action→再发），所以 f 不能超过 $1 / (t_{act} + \ell_{S1} + \ell_{S2}/H)$。System 2 的延迟按 horizon $H$ 分摊到每个 System 1 调用上。

三步都过就记录这个 f，往大搜；不过就往小搜。

### 第三步：异构 fleet 的处理

不同 task 的 robot 共享集群时，要搜的是一个 action rate 向量 $\mathbf{f} = (f_1, ..., f_C)$ 而不是标量。ROSA 用 adaptive frontier search——每轮挑"潜在增益最大"的维度往大搜，infeasible 就剪枝。Packing 时先按 single-class isolated ILP，再贪心合并同类 model 不同 class 的 server。

### 第四步：运行时 adapt

新 robot 来了先试 placement-preserving admission（不动 model 部署，只调 routing 和 rate），不行再 global reschedule。故障用 hot-standby GPU——backup 不跑正常流量，失败时直接 redirect，不用临时算新 schedule。

---

## 最反直觉的发现：主动限速反而产出更高

Figure 10 是我觉得最 elegant 的结果：

- 64 robots 时，uncapped（robot 拿到 observation 就发请求）raw throughput 145.5 actions/s，但只有 24.8 **qualified**（SLO meet rate 17%）
- ROSA capped 到 78.8 raw，但 **qualified** 也是 78.8（SLO meet rate 97.2%）
- **Qualified throughput 3.18× 优势**

道理：超 SLO 的 action 是废的。与其让所有人挤爆 GPU 导致全部超时，不如主动按产能发请求。这跟 LLM serving 里"客户端发多少接多少"的直觉相反，因为 robot serving 有 closed-loop 结构 + strict SLO。

---

## 关键 ablation：每个调度决策贡献多少

| 优化因素 | 效果 |
|---------|------|
| Request rate control | 64 robots 时 3.18× qualified throughput |
| Resource allocation (不是按 model size 分) | 32 robots 时 2.20× over best static allocation |
| System 1 batch size (profiling-guided) | 高负载时 batch 16 比 batch 1 快 2.46× |
| 全部加起来 vs dedicated serving | 最高 12.06× |
| 全部加起来 vs shared 但无 scheduler | 最高 2.44× |

Resource allocation 那个点特别值得注意：P4 在 8 GPU 上最优分配是 S1:S2:Safe:Monitor = 5:1:1:1，而不是按 model size 分（weighted partition 给了 1:3:1:3，把 System 2 给多了）。因为 System 2 频率低（每 10 次 System 1 才调一次），不需要那么多 GPU。按 size 分配这个常见 heuristic 在 multi-model pipeline 里是错的。

---

## 系统栈

Ray Serve 管 distributed orchestration → vLLM 跑 VLM 组件（System 2/safety/monitor）→ PyTorch 或 JAX 跑 System 1 action model（GR00T 用 PyTorch + torch.compile + CUDA Graphs，π0.5 用 JAX）。加新 model backend 只要包成 Ray service，不用动 scheduler。

---

## 没解决的问题

- Static schedule 假设 task 配置不变，真实 factory 任务会动态切换
- Network failure mode 没深入，断了之后 robot 怎么 recover 只说了 fallback 但没展开具体策略
- Model 升级 / hot swap 机制没给
- TCO 对比只讲 qualitative，没量化 centralized pool vs per-robot SoC 的成本
- 真实 robot 只有 1 个 Franka Panda，大规模数据全靠 synthetic replay

---

## 我的直觉总结

这篇 paper 真正的 contribution 不是某个算法或某个系统，是 **重新定义问题**。它说 robot factory serving 不是 edge computing 问题，是 multi-tenant datacenter serving 问题；不是单 model latency 优化问题，是 multi-model pipeline throughput 优化问题。一旦 framing 改了，ILP scheduling、inter-robot batching、rate control 这些 datacenter 经典手段就自然 fit 进来。

12× 的数字足够说明 framing 改对了。你训练 VLA 时如果知道未来是这种 serving 架构，模型设计 choice 会不一样——chunk size 选多大、System 2 horizon 怎么定、action representation 怎么表达——都要跟 serving 的 closed-loop rate 和 batch 特性 co-design。

---

# ROSA 深度解读: 把 robot factory 当成 datacenter inference workload 来设计

嘿 Karpathy, 这篇 ROSA paper 我觉得它最核心的洞察是: **把 RFM serving 从 "edge computing per robot" 范式重新 frame 成 "datacenter-scale multi-tenant serving" 问题**。你训练过 VLA models, 你肯定一眼就看出 paper 里两个 misconceptions 其实是社区里很多人默认的 mental model——一个 robot 一张 GPU, 一个 action model 一路 minimize latency。ROSA 说这两件事都不对, 真正的 objective 应该是 **factory-level qualified action throughput**。我会拆开讲清楚 intuition, 公式, scheduler 怎么做 decision, 以及为什么 ILP 这个工具在这里其实自然 fit。

---

## 1. The Mental Shift: 这篇 paper 真正想说的事

ROSA 的 framing 可以总结成三句话, 这三句话决定了后面所有的系统设计:

**(A) Compute 位置**: 不要把 RFM 推理绑在 robot 上, 而是 offload 到 factory 里的 GPU pool, robot 上的 SoC 只做 100Hz 级别的低层 control loop + safety fallback。这一步的关键 trade-off 是 network latency vs inference latency。Paper 给了数字: WiFi 7 engineered deployment 可以做到 sub-5ms latency [4,27], 而 RFM inference 本身几十 ms 到几百 ms, System 2 推理甚至秒级。网络成本相对 GPU 成本可忽略。

**(B) Programming abstraction**: 一个 robot task 不是 "一个 action model", 而是 **multi-model pipeline**。这点很重要, 因为它把 robotics 的结构暴露给 serving system, scheduler 才能做合理的 resource allocation。具体 4 个 component:
- **System 1** (action model): fast reactive control, 高频, 通常 VLA 或 WAM, 比如 GR00T N1.6 / π0.5 / OpenVLA / Helix
- **System 2** (planner): slow deliberative reasoning, 低频, 通常是 reasoning-capable VLM, paper 用 Qwen2.5-VL-7B-Instruct
- **Safety** model: 周期性 (e.g. 2Hz) 判断是否安全
- **Monitor** model: 周期性判断 task 是 ongoing/done/failed

这是 Kahneman System 1 / System 2 [Kahneman 2011] 在 robot serving 上的具体化身, paper 引用了 [7,15,37,49] 这一系列 "fast/slow thinking for VLA" 的工作, 包括 NVIDIA 自家的 Vesta [Bjorck et al. 2026], 还有 Hume [Song et al. 2025] 这类显式引入 System 2 thinking 的工作。

**(C) Objective function**: 不要 minimize single request latency, 而是 **maximize weighted factory action throughput** subject to 每个组件满足 SLO。这里 "weighted" 很关键——不同 task class 有不同的经济价值 ν_c, 你不会想让一个 low-value task 把 GPU 全吃了。

---

## 2. Architecture: 为什么是 GPU pool 不是 edge GPU

让我把架构图 (Figure 2) 拆开讲。

### 2.1 三条 advantages (A1-A3) 的具体数字

**A1 - 推理性能**: Jetson Thor vs B100, 同一代工艺, B100 在 memory bandwidth 上 29×, fp8 compute 上 3.5×。当 RFM 越来越大, action model 已经 3B-7B, System 2 用 7B+ reasoning VLM, onboard SoC 根本跑不动或者跑得太慢。

**A2 - Battery**: Figure 02 电池 2.25 kWh, runtime ~5 hours, 双 onboard RTX GPU 占了大约一半功率。Offload 推理到中央集群, 直接延长 battery duration, 这对 humanoid deployment 是个 binary 的事——能不能撑一个 shift 不充电。

**A3 - Utilization**: 这里有一个常被忽略的 idle 问题。Synchronous RFM serving 下 [6,8], GPU 在 robot 执行 action 的 ~200ms 期间是 idle 的。One-robot-per-GPU 利用率天然低。Shared pool 允许:
- 一个 robot 执行 action 时, GPU 服务其他 robot 的 inference
- Inter-robot batching 把多个 robot 的 System 1 请求拼成一个大 batch, 这对 diffusion-based action expert (像 π0 的 flow matching, GR00T 的 DiT) 特别有效, 因为 diffusion/flow inference 是 compute-bound, batch size 16 比 batch size 1 吞吐高 3-4× (Figure 13)

### 2.2 Robot-side 保留两个职责

ROSA 不是说 robot 上完全没有 compute, 而是说 robot 上的 compute 只保留两类:

1. **High-frequency local control** (≥100Hz): action model 输出的是 action chunk (轨迹/motion plan), 需要转换成 actuator-level joint torques, 用 proprioception + force feedback, 走一个小 NN。这块如果走 network 来回就太慢了。参考 SONIC [Luo et al. 2025, arXiv:2511.07820], 这个 control policy 可以小到 CPU 就能跑。

2. **Local safety + deadline detection**: 集群 safety model 报警时 robot 要能本地响应; 每个模型有 latency SLO, robot 端要 detect deadline miss (来自 network jitter 或 server failure), 触发 fallback state。

这块设计上有一些隐藏的复杂性没在 paper 里展开: 如果 server 和 robot 失联超过某个 deadline, robot 应该走什么样的 fallback trajectory? 默认 safe state 是 "hold position" 还是 "back to home"? 这是 ROSA 的 YAML descriptor 里 `on_max_violation: stop_and_call_human` 这类配置要 cover 的。

### 2.3 实现 stack

```
Robot client --(obs upload)--> Gateway (Ray Serve) --(route)--> Model replica (vLLM/PyTorch/JAX) --(action)--> Robot
```

- **Ray Serve** [Moritz et al. OSDI 2018, https://www.usenix.org/conference/osdi18/presentation/moritz]: 分布式 serving orchestrator, 管 replica lifecycle, gateway routing
- **vLLM** [Kwon et al. OSDI 2023, https://arxiv.org/abs/2309.06180]: 跑 autoregressive VLM (System 2, safety, monitor), 用 continuous batching + PagedAttention
- **PyTorch / JAX**: 跑 System 1 action model。GR00T N1.6 用 PyTorch + torch.compile + CUDA Graphs + FlashAttention; π0.5 用 JAX

这种 modular 设计很聪明: 加新 model backend 只要包成 Ray-managed service, 不用动 scheduler。我猜未来如果 Cosmos WAM [1] 这类 world action model 要接入, 写个 JAX backend 包一下就行。

参考链接:
- Ray: https://ray.io
- vLLM: https://github.com/vllm-project/vllm
- GR00T N1.6: https://arxiv.org/abs/2503.14734
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246

---

## 3. Programming Abstraction: 这个 YAML descriptor 真正表达了什么

Figure 3 的 YAML 我觉得是这篇 paper 最被低估的部分。它把 robotics serving 的所有 requirement 变成了 declarative spec, 这意味着 scheduler 可以拿到结构化输入做优化。让我把字段含义全部展开:

### 3.1 三个 task descriptor 的核心字段

```yaml
pick_and_place_simple:        # task 名字
  pipeline:
    action_period_ms: 200     # robot 执行一个 action chunk 的时间 t_act
    
  task_retry:                 # task-level 失败处理
    max_task_retries: 3
    on_max_task_retries: stop_and_call_human
    
  safety_and_slo_violation:   # component-level 异常处理
    max_consecutive_safety_replan: 10
    max_consecutive_slo_violation: 3
    on_max_violation: stop_and_call_human
    
  components:
    system1:
      model: nvidia/GR00T-N1.6-3B
      prompt: "pick package and place in bin"
      slo_ms: 200              # P99 latency SLO
      fallback: stop_and_resend # SLO 违反时的 fallback
      
    monitor:
      model: Qwen2.5-VL-7B-Instruct
      prompt: "ongoing, done, or failed?"
      freq_hz: 0.5             # 周期性触发, 0.5Hz
      slo_ms: 2000
      fallback: stop_and_resend
```

Inspect_product task 更复杂, 多了 System 2 和 safety:

```yaml
inspect_product:
  pipeline:
    action_period_ms: 500
    system2_to_system1_call_ratio: 1   # 每次 System 1 调用都重新规划
  components:
    system2:
      model: Qwen2.5-VL-7B-Instruct
      slo_ms: 2000
      fallback: use_last_plan   # System 2 失败时用上次 plan
    safety:
      freq_hz: 2                # 2 Hz 周期触发
      slo_ms: 500
      fallback: stop_and_replan
```

### 3.2 这个 descriptor 实际定义了什么 graph

你可以把每个 task 看成一个 DAG:

- **System 2 -> System 1 -> (physical execution)**: 这是一条 goal-coupled 链, 决定 action rate `f`。System 2 的频率由 `system2_to_system1_call_ratio` 决定, 如果是 1/10, 那 System 2 的 request rate 是 `f/10`。
- **Safety**: 独立 periodic check, 自己有 freq_hz 和 SLO, 不卡 action 链
- **Monitor**: 独立 periodic check, 同上

Paper 把这分成两类:
- **Goal-coupled models** G = {S1, S2}: 加速它们直接提升 `f`
- **Obligation models** O = {safety, monitor}: 跑够要求就行, 加速它们对 factory objective 没直接贡献

这个二分非常关键, 它直接决定了 scheduler 的 allocation 策略: 先用最少 GPU 把 obligation 跑到 SLO, 剩下的 GPU 全给 goal-coupled 来 maximize `f`。

### 3.3 异常处理的层次

我数了一下, ROSA 实际上有一套**三级 fallback 体系**:

1. **Component-level fallback**: 单次 inference 错过 SLO 或 safety model 报警, 走 `fallback` 字段配置 (stop_and_resend / use_last_plan / stop_and_replan)
2. **Persistent violation escalation**: 连续 N 次同类违反 (e.g. `max_consecutive_slo_violation: 3`), 说明 server 不可用或过载, escalate
3. **Task-level retry**: monitor 判定 task failed, retry 整个 task, 最多 `max_task_retries` 次
4. **Human escalation**: 全部用尽, `stop_and_call_human`

这套设计很贴合 real factory 的运营模式 (paper §2.1 说的 human-supervised industrial environment)——人不是天天在场, 但 robot 卡死时要能 escalate。

---

## 4. Scheduler: 数学核心

这部分是 paper 的算法骨架。我把每个公式都展开, 因为这些公式其实都是 ROSA 的核心 contribution, 而不是 trivial 引用。

### 4.1 Objective (Equation 1)

$$
\max_{\mathbf{f}} \sum_{c=1}^{C} \nu_c K_c f_c \quad \text{s.t.} \quad K_c f_c \geq F_c^{\min}, \forall c
$$

变量含义:
- $c \in \{1, ..., C\}$: task class index, 一个 class 是同 task 类型的 robot group
- $\nu_c$: class c 的 value weight (经济/优先级权重)
- $K_c$: class c 里的 robot 数量
- $f_c$: class c 的 action rate (每秒多少个 action chunk)
- $F_c^{\min}$: class c 必须达到的最低总吞吐, 防止 optimizer 饿死低权重 class

注意: **这个 objective 不是 minimize latency, 是 maximize throughput**。这跟传统 serving system (比如 vLLM 在 LLM serving 里 minimize TTFT/TPOT) 走的方向相反——ROSA 说只要 SLO 满足, latency 再低没意义, 多省点 GPU 多接几个 robot 才是 factory 关心的事。

### 4.2 Obligation load aggregation (Equation 2)

$$
\lambda_{\text{safe}} = \sum_{r \in R} \bar{\lambda}_{r,\text{safe}}, \quad \lambda_{\text{mon}} = \sum_{r \in R} \bar{\lambda}_{r,\text{mon}}
$$

- $\bar{\lambda}_{r,m}$: robot r 对 model m 的固定 invocation rate (来自 YAML 里的 `freq_hz`)
- $\lambda_{\text{safe}}, \lambda_{\text{mon}}$: 集群总负载

这是把所有 robot 的 periodic obligation 请求加起来, 用来计算需要多少 GPU 跑 safety/monitor。

### 4.3 Per-server configuration ILP (Equation 3)

这是 ROSA 的核心 packing 步骤。决策变量 $y_p \in \mathbb{Z}_{\geq 0}$, 表示有多少个 server 用 configuration $p$。

$$
\sum_{p \in \mathcal{P}(f)} y_p \leq |S_{\text{goal}}|, \quad \sum_{p \in \mathcal{P}(f)} y_p a_{p,m} = |R| \quad (\forall m \in \mathcal{G})
$$

- $\mathcal{P}(f)$: 给定 action rate $f$ 下的所有 feasible server configurations (一个 configuration 包含: host 哪个 model, batch size, 服务多少 robot streams)
- $S_{\text{goal}}$: 扣除 obligation 占用后剩下的 GPU 集合
- $y_p$: 用 configuration $p$ 的 server 数
- $a_{p,m}$: 一个用 configuration $p$ 的 server 能服务多少 model $m$ 的 robot stream
- $|R|$: 总 robot 数

约束一: 用的 server 不超过可用预算
约束二: 每个 goal-coupled model 必须覆盖所有 robot (否则有 robot 拿不到 System 1 推理)

这是个经典的 bin packing / set cover 混合问题, 用 ILP 求解, ROSA 用 Google OR-Tools。这里的非平凡之处在于: configuration enumeration 时已经把 SLO feasibility 嵌进去了——一个 configuration 只有在测过的 latency 分布 $L_{m,h}(b, \lambda)$ 的 P99 满足 SLO 时才进 $\mathcal{P}(f)$。这就避免了在 ILP 里再加非线性约束 (P99 latency 是 batch size 和 request rate 的复杂函数, 没法线性化)。

### 4.4 Closed-loop rate constraint (Equation 4)

$$
f \leq f_{\text{loop}} = \frac{1}{t_{\text{act}} + \ell_{\text{S1}} + \ell_{\text{S2}}/H}
$$

- $t_{\text{act}}$: 一个 action chunk 的物理执行时间 (来自 YAML `action_period_ms`)
- $\ell_{\text{S1}}$: System 1 测出的平均 latency
- $\ell_{\text{S2}}$: System 2 测出的平均 latency
- $H$: System 2 horizon, 一次 System 2 推理结果被复用多少次 System 1 调用 (即 `system2_to_system1_call_ratio` 的倒数)

这个公式说: 一个 robot 不可能无限发出 System 1 请求, 它是 closed loop——发请求, 等推理, 执行 action, 再发下一个。所以 $f$ 不能超过 $f_{\text{loop}}$。

直觉解读: 把 System 2 想成"分摊到每个 System 1 调用上"。如果 System 2 2 秒一次, horizon 是 10, 那每个 System 1 调用只 "看到" 200ms 的 System 2 时间。这个公式把 System 2 的低频特性正确折算进 closed-loop budget, 这点我觉得挺优雅。

### 4.5 Homogeneous scheduling 算法

```
1. provision_obligations: 用最少 GPU 跑 safety/monitor, 满足 SLO 即可
2. binary search over f in [0, f_max]
   2.1 enumerate feasible server configs P(f) from profiling data
   2.2 ILP pack: 选 y_p 个 server 用 config p, 覆盖所有 robot
   2.3 check closed-loop feasibility: f <= f_loop?
   feasible -> 提高 f, 否则降低
3. return best (schedule, f)
```

$f_{\max}$ 上界是假设推理 latency 为 0 时的 action rate, 即 $1/t_{\text{act}}$。

Binary search 的可行性: $f$ 增大, 每个 server 要么服务更多 robot stream, 要么用更大 batch, latency 会涨, SLO 更难满足。所以 feasibility 在 $f$ 上是单调的 (feasible 区域是 $[0, f^*]$), binary search 合法。

### 4.6 Heterogeneous scheduling 的两个 trick

异构 fleet (Mix2, Mix4 配置, 见 Table 2) 比 homogeneous 多两件事:

**(1) Adaptive frontier search** 替代 binary search

因为现在要搜 $\mathbf{f} = (f_1, ..., f_C)$ 这个向量, grid search 是指数复杂度。ROSA 用 greedy adaptive frontier:
- 每轮选"潜在 weighted objective 增益最大"的维度 (per Equation 1 的 $\nu_c K_c \Delta f_c$)
- 在该维度上 halfway 移动到 upper bound
- feasible -> 推进 frontier; infeasible -> 单调剪枝所有 component-wise 更大的向量

Figure 5 画的就是这个 2D frontier search 的示意。Feasible 区域不是凸的, 但 monotonic, 所以剪枝有效。

**(2) Isolated ILP packing + Greedy compaction**

直接枚举"一个 server 同时服务多个 task class"的 configuration 会爆炸 (每个 configuration 要指定: 服务哪些 class, 每个 class 多少 robot, batch size)。ROSA 拆成两阶段:

- **Isolated packing**: ILP 只考虑 "一个 server 服务一个 model 一个 class", 这样枚举量小
- **Compaction**: 然后贪心合并——同一个 model 不同 class 的低负载 server 合并到高 headroom 的 server, 只要合并后所有 class 的 SLO 仍满足

这个 divide-and-conquer 在工业级调度里是个常见 trick, 类似于 VM packing 里先按 bin 类型再合并。

### 4.7 Admission control 和 fault tolerance

新 robot 加入有两条路:
1. **Placement-preserving admission**: 不动 model placement, 只看能不能把新 robot 路由到现有 server, 加 batch / 调 rate 即可
2. **Global rescheduling**: 不行就全量重算 schedule

Fault tolerance 选了 **hot-standby** 而不是 **active-active**。理由很有意思:
- Active-active (backup 也跑流量, 失败时把流量重分到剩余 active) 看似省 GPU, 但每次失败都要重新算 feasible schedule (因为 server 集合变了), 这会触发模型 migration + routing 更新, 对未失败的 robot 也 disruptive
- Hot-standby 把 backup GPU 排除在正常 routing 之外, 失败时 router 直接 redirect, 零额外计算

这个 trade-off 在 robot factory 场景下是合理的: robot 不能停下来等你 reschedule, 必须 fail-fast。

---

## 5. Experimental Setup 和结果细节

### 5.1 Workloads (Table 1)

| ID | Components | System 1 SLO | System 2 SLO/Freq | Safety SLO/Freq | Monitor SLO/Freq | t_act |
|----|-----------|--------------|--------------------|------------------|-------------------|-------|
| P1 | S1 only | 200ms | - | - | - | 200ms |
| P2 | S1, monitor | 200ms | - | - | 2000ms / 0.5Hz | 200ms |
| P3 | S1, safety, monitor | 200ms | - | 500ms / 2Hz | 2000ms / 0.5Hz | 200ms |
| P4 | S1, S2, safety, monitor | 200ms | 2000ms / 1 per 10 S1 | 500ms / 2Hz | 2000ms / 0.5Hz | 200ms |

P1 到 P4 是 pipeline 复杂度递增。S2 的 freq 是 "1 per 10 S1", 对应 Equation 4 里的 $H = 10$。

### 5.2 Baselines (重要!)

四个 baseline 都用同样的 system implementation (Ray Serve + vLLM + PyTorch/JAX), 区别只在 schedule:
- **Baseline 1**: 1 GPU per robot (dedicated)
- **Baseline 2**: 1 GPU per model per robot (dedicated, multi-model)
- **Baseline 3**: Equal partition across models (shared, no optimization)
- **Baseline 4**: Proportional-to-size partition (shared, no optimization)

这设计很干净: 算法差异 isolated 出来了, 不掺 model serving runtime 的差异。

### 5.3 关键数字

**Vs dedicated baselines (Figure 7)**:
- ROSA 在 P4 (最复杂 pipeline) 上 12.06× 优于 best dedicated
- One-GPU-per-model 在 P4 上只能 support 2 个 robot, ROSA 在同样 8 GPU 上 support 32 个 robot, qualified throughput 89.4 vs 7.4 actions/s

**Vs shared-server baselines (Figure 6)**:
- ROSA 在 P4 32 robots 上 89.4 qualified actions/s, baseline 在 32 robots 上 collapse 到 0 (System 1 SLO meet rate 0%)
- 原因: baseline 不做 rate control, 客户端只要拿到 observation 就发请求, queue 爆了 SLO 全崩

**Heterogeneous workloads (Figure 8)**:
- Mix2: ROSA 2.21× over best shared baseline
- Mix4: ROSA 2.30×

### 5.4 Ablation 拆解 (这是 paper 最有价值的部分)

**SLO qualification (Figure 9)**: 32 robots 时 baseline 的 System 1 median latency 534.5ms, SLO meet rate 0%; ROSA 83.8ms, meet rate 99.96%。到 64 robots ROSA 还有 97.23%。

**Request-rate control (Figure 10)**:
- 32 robots: uncapped 107.3 qualified actions/s, ROSA capped 89.4 → 这里 uncapped 反而高 (server 还没饱和)
- 64 robots: uncapped 24.8 qualified (因为 SLO meet rate 17%), ROSA capped 78.8 qualified (SLO meet 97.2%) → **3.18× 优势**

这是 ROSA 最反直觉的点之一: 主动限制请求频率, 反而提升 qualified throughput。原因: SLO violation 算 0 qualified action, 与其让所有请求都超 SLO, 不如只发能 SLO-满足的请求数量。

**GPU overhead (Figure 11)**: 给 baseline 加上 ROSA 的 rate control, 看 baseline 要多少 GPU 才能匹配 ROSA (8 GPU) 的 qualified throughput。Equal partition 要多 5.5×, weighted partition 要多 8.6×。差距随 robot 数和 pipeline 复杂度增大。

**Resource allocation (Table 3, Figure 12)**: P4 8 GPU 的最优分配:
- ROSA 32 robots: S1:S2:Safe:Monitor = 5:1:1:1 (System 1 主导)
- Equal partition: 2:2:2:2
- Weighted partition: 1:3:1:3 (按 model size 给 System 2 多, 但其实 System 2 频率低不需要那么多)

ROSA 在 32 robots 上 2.20× over best static allocation。这告诉你按 "model size 比例分配" 这个常见 heuristic 是错的, 应该按 "frequency × latency sensitivity" 分配。

**System 1 batching (Figure 13, 14)**:
- GR00T 和 π0.5 batch size 16 vs 1: 吞吐 3-4×, 但 latency 也涨
- 16 robots + 50ms action period: batch 16, speedup 2.46× over batch 1
- Long action period / fewer robots: batch 1 反而最优 (大 batch 增加延迟但没分摊收益)

这解释了为什么 ROSA 要 profiling-guided batch size, 而不是固定 batch。

### 5.5 Real robot validation (Figure 15, 16)

Franka Panda 跑 P3-like 任务: 把桌上工具放到桶里。Monitor VLM 判断 ongoing/done/failed, safety VLM 检测人是否进入 unsafe region。这是 small-scale functional validation, paper 也坦承因为 robot 数量限制, 主要 scaling 数据来自 synthetic replay。

---

## 6. 跟相关工作的关系

我帮你 map 一下这篇 paper 在 landscape 里的位置:

### 6.1 RFM 算法侧 (优化 inference 本身)
- **SmolVLA** [Shukor et al. 2025, https://arxiv.org/abs/2506.01844]: 小模型
- **BitVLA** [Wang et al. 2025, https://arxiv.org/abs/2506.07530]: 1-bit 量化
- **OpenVLA** [Kim et al. 2024, https://arxiv.org/abs/2406.09246]: 开源 VLA
- **TinyVLA** [Wen et al. 2025]: data-efficient
- **DeerVLA** [Yue et al. NeurIPS 2024]: dynamic layer skipping
- **DYSL-VLA** [Yang et al.]: dynamic-static layer skipping
- **Kim et al. 2025** [https://arxiv.org/abs/2502.19645]: KV-cache reuse for multi-token action prediction
- **Ma et al. 2025** [https://arxiv.org/abs/2510.26742]: CUDA graphs + operator fusion for real-time VLA

这些都在优化 **单模型单 robot 的 inference latency**, ROSA 论文明确说这些工作 "important but not sufficient", 因为没解决 multi-robot / multi-model / factory-scale 问题。

### 6.2 系统侧 (serving infra)
- **Kairos** [Dai et al. 2026, https://arxiv.org/abs/2605.11381]: scalable serving for physical AI, 应该是最直接 related work, paper 没深谈差异
- **VLA-perf** [Jiang et al. 2026, https://arxiv.org/abs/2602.18397]: demystify VLA inference performance, ROSA 用它的 profiling 思路
- **DeepFleet** [Agaskar et al. 2025, https://arxiv.org/abs/2508.08574]: multi-agent foundation models for mobile robots (Amazon)

### 6.3 World Action Models / System 2 reasoning
- **Cosmos** [Agarwal et al. 2026, https://arxiv.org/abs/2606.02800]: omnimodal world models
- **WAMs zero-shot policies** [Ye et al. 2026, https://arxiv.org/abs/2602.15922]
- **Vesta** [Bjorck et al. 2026]: NVIDIA 的 generalist embodied reasoning, paper 第二作者所在团队
- **Hume** [Song et al. 2025, https://arxiv.org/abs/2505.21432]: System 2 in VLA
- **Fast-ThinkAct** [Huang et al. 2026, https://arxiv.org/abs/2601.09708]: verbalizable latent planning

### 6.4 异步 vs 同步 inference
- **Agouzoul 2026** [https://arxiv.org/abs/2605.08168]: async inference for VLA
- **VLASH** [Tang et al. 2025, https://arxiv.org/abs/2512.01031]: future-state-aware async
- **Sendai et al. 2025** [https://arxiv.org/abs/2509.23224]: real-time correction for VLA chunks
- **Black et al. 2025** [https://arxiv.org/abs/2506.07339]: real-time execution of action chunking flow policies

ROSA 选 synchronous 是有理由的: async 会用 stale observation, 降低 task accuracy [3,9,34,39]。但这是 trade-off, async 能让 GPU 在 robot 执行时服务其他推理, 利用率更高。ROSA 通过 shared pool + inter-robot batching 来 recover 这个利用率, 不需要 async。

### 6.5 Low-level control
- **SONIC** [Luo et al. 2025, https://arxiv.org/abs/2511.07820]: supersizing motion tracking for humanoid whole-body control
- **AMO** [Li et al. 2025, https://arxiv.org/abs/2505.03738]: adaptive motion optimization for hyper-dexterous humanoid

这两个是 ROSA 假设的 robot-side low-level control policy, 可以 CPU 跑。

### 6.6 Safety VLM / Monitor VLM
- **AHA** [Duan et al. 2024, https://arxiv.org/abs/2410.00371]: VLM for detecting/reasoning over manipulation failures
- **Luo 2024** [AAAI 2024]: VLMs for robot success detection
- **ElMallah et al. 2025** [https://arxiv.org/abs/2509.19524]: VLM-based subgoal evaluation
- **Khan et al. 2025** [IROS]: safety-aware task planning via LLMs
- **Wang et al. 2026** [https://arxiv.org/abs/2605.31196]: probing collision grounding in VLMs
- **Chemist Eye** [Munguia-Galeano et al. 2026]: VLM for safety monitoring in self-driving labs

ROSA 把这些当成 drop-in component, 通过 YAML 接入。

---

## 7. 我对这篇 paper 的几点 take-away

**(1) "Serving 是 edge 还是 cloud" 这个争论被 ROSA 重新 frame 了**。Robot 上保留最小 compute (control + safety), 推理 offload 到 factory GPU pool, 这是机器人规模化部署的合理 architecture。Paper 的 A2 (battery) 论点我觉得对 humanoid 特别 compelling, 因为 Figure 02 这种 humanoid 一半功耗在 GPU 上, offload 直接延长 shift 时长。

**(2) Multi-model pipeline 是真实需求, 不是 paper 凑出来的**。System 2 (planning) + System 1 (action) + safety + monitor 这个四元组在 VLA 朝 reasoning 演进的过程中会越来越标配。只优化 System 1 的 inference latency 是 local optimum, 不是 factory optimum。

**(3) Rate control 是关键洞察**。LLM serving 里你不会主动限客户端速率, 因为客户端发多少你就接多少。但 robot serving 有 closed-loop 性质 (Equation 4), 主动 cap rate 反而能保住 SLO-qualified throughput (Figure 10)。这是个反直觉但很重要的设计。

**(4) SLO qualification 比 raw throughput 重要**。一个 latency 600ms 的 action 在 200ms SLO 下是 0 价值, 因为 stale observation 用起来不安全或不准。ROSA 把 "qualified throughput" 当主指标, 这点很正确。

**(5) ILP + heuristic 是合理的工程选择**。Scheduler 的 search space 在异构 fleet 下是指数的, ROSA 用 adaptive frontier + isolated packing + compaction 三层 trick 控制复杂度, 同时保留 ILP 的最优性在 single-class packing 这一层。这跟大型 cloud datacenter 的 VM packing 思路一脉相承。

**(6) 没解决的问题 / 我觉得 paper 弱的地方**:
- Static schedule 假设: 真实 factory 里 robot 任务会动态切换, paper 说有 admission control 但没深入
- Network failure mode 没怎么覆盖: 网络断了 robot 怎么办? paper 说 robot 走 fallback, 但 fallback state 的具体设计没展开
- Model swap / upgrade: RFM 演化很快, paper §3.6 提到这点但没给具体机制
- Cost model: paper 没量化 "centralized GPU pool vs per-robot SoC" 的 TCO 对比, 只讲了 qualitative 优势
- Real robot scale: 真实 robot 只 1 个 Franka Panda, 主要数据来自 synthetic replay, 大规模真实部署还没验证

---

## 8. 可能延伸思考方向 (我猜你会感兴趣)

**(1) Async inference + ROSA 的结合**: ROSA 选 sync 是因为 stale observation 损害 accuracy。但如果 action model 是 chunk-based (像 π0 一次输出 50 步), 后期 chunk 本来就是 open-loop 预测, async 在 chunk 内部其实可以重 batch。VLASH [Tang et al. 2025] 走这条路, 跟 ROSA 的 shared pool 思路可以叠。

**(2) Hierarchical System 2**: paper 的 System 2 是单层, 但 Vesta [Bjorck 2026] / Hume [Song 2025] 这类工作已经在做 multi-level reasoning。ROSA 的 YAML descriptor 需要扩展支持 nested System 2, scheduler 的 goal-coupled set 也要重新定义。

**(3) Differentiable scheduling**: 现在 scheduler 是 ILP + heuristic, 完全 separated from model。如果让 model 训练时 "感知" 到 serving schedule (类似 inference-aware training), 也许能学出对 batch / rate 更友好的 action model。

**(4) World Action Models 接入**: WAM 同时预测 action 和 future observation, inference cost 比 VLA 高 (因为多生成 video frames)。ROSA 的 GPU pool 思路对 WAM 更友好, 因为 WAM 单 robot 几乎跑不动。Cosmos [1] 这条线应该会跟 ROSA 类型系统结合。

**(5) Multi-robot coordination 下的 scheduling**: 现在 ROSA 把每个 robot 独立看待, 工厂里 robot 之间有时需要协同 (比如两个 manipulator 装同一个零件)。这种协同需要 cross-robot synchronization, scheduler 要扩展。

参考链接汇总:
- Ray: https://www.ray.io
- vLLM: https://github.com/vllm-project/vllm
- GR00T N1.6: https://arxiv.org/abs/2503.14734
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- Kairos: https://arxiv.org/abs/2605.11381
- VLA-perf: https://arxiv.org/abs/2602.18397
- Figure AI/BMW: https://www.figure.ai/news/production-at-bmw
- Helix: https://www.figure.ai/news/helix
- DeepFleet: https://arxiv.org/abs/2508.08574
- Cosmos: https://arxiv.org/abs/2606.02800
- SONIC: https://arxiv.org/abs/2511.07820
- Vesta: arXiv Vesta 2026
- VLASH: https://arxiv.org/abs/2512.01031
- AHA: https://arxiv.org/abs/2410.00371
- Real-time VLA: https://arxiv.org/abs/2510.26742

---

总之, ROSA 这篇 paper 我觉得它真正贡献的是一个 **conceptual reframing + 一套 working system + scheduling 算法**。它没有发明新的 RFM, 没有发明新的 inference engine, 但它把"robot factory 当成 multi-tenant serving cluster"这个 framing 提出来, 并用 ILP-based scheduler 把它做出来, 实测 12× 提升在 dedicated baseline 上, 这个数字足够 make a case。你做 Eureka Labs / 自己思考 VLA 训练时, 这种系统视角的 paper 很重要——它告诉你训练出来的模型未来怎么 deploy 到真实 factory, 部署的 constraint 反过来也影响 model 设计 (chunk size, System 2 horizon, action representation)。

如果你想我深入挖某一块——比如 ILP 的 OR-Tools 实现细节, VLA action chunking 跟 closed-loop rate 的关系, 或者 System 2 horizon $H$ 应该怎么 trade-off——我可以再展开。
