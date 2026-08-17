---
source_pdf: RoboClaw An Agentic Framework for Scalable.pdf
paper_sha256: 3072ffdd4ac0f8f34e37e60ca5dee588364edb870193c8a10216112451c3d609
processed_at: '2026-08-12T00:35:33-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 RoboClaw

## 1. 这篇 paper 到底在解决什么问题

想象你训练了一个 robot policy，让机器人"把化妆品放进抽屉"。这个 policy 单独跑能成功 80%。

现在你让它做一整套 vanity table 整理任务：放身体乳 + 放 primer + 插口红 + 擦桌上的水。四个 subtask 串起来，理论上成功概率 $0.8^4 = 0.41$，实际更糟，因为前一个 subtask 失败会把后面的 state 全搞乱——口红插歪了，下一个 wipe 任务的起始条件就不对了，error 级联放大。

更要命的是训练数据的问题。传统 pipeline 里：

- 人 teleop 采集数据：每次跑完一个 trajectory，object 被拿走了，人得手动放回去（**reset**），然后跑下一个。这个 reset 占了 human labor 大头。
- 人采的数据是 "happy path"：人会顺顺当当把东西放进去，不会展示 "抓空了怎么办""瓶子倒了怎么办"。
- Robot 部署时遇到的都是 failure state（robot 自己会犯错），但 policy 在这些 state 上没训过 → OOD → 崩。

这三个问题叠加：**data collection 贵 + training data 跟 deployment 不匹配 + long-horizon error cascade**。

RoboClaw 一篇文章想同时解决这三件事。

## 2. 核心招式：Entangled Action Pairs (EAP)

最关键的 insight 一句话：**对每个"把东西放进去"的 forward policy，再训一个"把东西拿出来"的 inverse policy，两个组成一对**。

Robot 工作流变成：
1. 执行 forward："把 primer 放进 drawer"
2. 执行 inverse："把 primer 从 drawer 拿出来放回原位"
3. 环境回到初始状态 → 跑下一个 trajectory
4. 无限循环，不需要人 reset

为什么这能 work？因为 **inverse 任务故意被设计得比 forward 简单**。比如：

- Forward："把口红插进 narrow slot" → 需要 precise alignment + 角度对，tolerance 极紧
- Inverse："把口红从 slot 拔出来放桌上" → 只要 grasp + lift，tolerance 松

Table 2 验证了这个 asymmetry：inverse policy success rate 在 72-86%，远高于 forward policy 第一轮的 4-46%。Reset 比 forward 容易成功，loop 就能稳定持续。

这跟化学平衡常数反过来用：化学反应里我们常常希望"容易前进、难回去"，EAP 反过来——"难前进、容易回去"，这样系统永远能回到原点。

## 3. 系统长什么样

整个系统就一个 agent loop，VLM (类似 GPT-4V 这种多模态大模型) 当大脑：

```
观察 → 记忆 → 推理 (CoT) → 调用工具 → 改变环境 → 写回记忆 → 回到观察
```

记忆结构 $m_t = (r_t, g_t, w_t)$ 三层：

- **$r_t$ role identity**: 告诉 agent "你现在是 data collector 还是 task executor"。同一个 agent core，切 role 就切行为模式，省了维护两套系统。
- **$g_t$ task memory**: 全局任务 + 拆解出的 subtask 列表 + 每个 subtask 的状态。long-horizon 任务能 track 进度就靠这个。
- **$w_t$ working memory**: 当前在跑哪个 skill + tool 调用历史。short-term context。

工具通过 MCP (Model Context Protocol) 接口暴露，包括：Start Policy / Terminate Policy / Change Policy / Env Summary / Fetch Robot Stats / Call Human。MCP 是 Anthropic 提的标准化 tool calling 协议，参考 https://modelcontextprotocol.io。

VLM 的 reasoning 四步走：
1. 看场景，识别相关物体
2. 确定当前 subtask
3. 评估成功条件，判断当前 state 是否满足
4. 决定下一步：retry / 换 policy / 调 recovery / 找人 / 进下一个 subtask

## 4. 底层 policy 是 π_0.5

低层 motor control 用 Physical Intelligence 的 π_0.5，一个 VLA flow matching model。参考 https://arxiv.org/abs/2504.16054。

Forward policy 给的 language instruction $l_t$ **不是人写的**，是 agent 在调 MCP tool 时动态生成的结构化指令。比如 agent 推理出"现在该放 primer 了"，就生成 `{"action": "place", "object": "primer", "target": "drawer"}` 喂给 VLA。

训练用 flow matching：

$$\mathcal{L}^{\tau}(\theta) = \mathbb{E}\left[\| v_{\theta}(A_t^{\tau}, o_t, l_t, q_t) - u(A_t^{\tau} | A_t) \|^2\right]$$

变量解释：
- $o_t$: visual observation
- $l_t$: language instruction (agent 动态生成)
- $q_t$: robot proprioceptive joint state
- $A_t = [a_t, ..., a_{t+H-1}]$: action chunk，长度 H
- $\tau \in [0,1]$: flow matching 时间步，0 是纯噪声，1 是真实 action
- $A_t^{\tau} = (1-\tau)\epsilon + \tau A_t$: 噪声 $\epsilon$ 和真实 action 的线性插值
- $v_\theta$: 神经网络学的 velocity field，把 Gaussian distribution "transport" 到真实 action distribution
- $u$: target conditional velocity field

Flow matching 比 diffusion 好在 inference 快，Table 1 显示只要 **3 inference steps** 就够，比 DDPM 几十步省太多。

LoRA fine-tune 参数：rank 16, alpha 16, dropout 0.1, target all-linear, lr 2.5e-5, 10k steps, bfloat16。每个 task 训一个 LoRA adapter，可以 hot-swap，policy pool 容易扩展。LoRA 原理参考 https://arxiv.org/abs/2106.09685。

## 5. Forward-Inverse 数学表达

公式 (6)(7)(8) 把 EAP 形式化：

Forward trajectory（执行任务）：
$$\tau_k^{\text{fwd}} = \{(o_t, q_t, a_t)\}_{t=0}^{T}, \quad a_t = \pi_{\theta_k}^{\text{fwd}}(o_t, l_t, q_t)$$

Inverse trajectory（reset）：
$$\tau_k^{\text{rst}} = \{(o_t, q_t, a_t')\}_{t=T+1}^{T+T_{\text{reset}}}, \quad a_t' = \pi_{\phi_k}^{\text{rst}}(o_t, l_t, q_t)$$

Entangled pair：
$$\boldsymbol{\tau}_k = (\tau_k^{\text{fwd}}, \tau_k^{\text{rst}})$$

注意 forward 用 $\theta_k$ 参数，inverse 用 $\phi_k$ 参数，是两个不同 checkpoint，但 share 同样的 input interface。$l_t$ 在 forward 时是 "place primer into drawer"，inverse 时是 "take primer out of drawer"。

## 6. 部署时怎么干 long-horizon

部署时 agent 做 **runtime supervision**，循环：

1. 跑当前 subtask 的 forward policy
2. 周期性 Fetch Robot Stats + Env Summary 检查状态
3. 评估 subtask 是否成功
   - 成功 → 更新 $g_t$，进下一个 subtask
   - 失败 → 三选一：
     - **Retry same policy**（non-degrading failure：env 没变，比如 gripper 抓空了但 bottle 还直立原位）
     - **Change Policy**（degrading failure：env 变了，但有合适 alternative policy）
     - **Call Human**（safety critical 或 autonomous recovery 失败）

这个 monitor + recover 机制让 success rate 不再是简单乘法。传统 4 个 0.7 串起来是 0.24，RoboClaw 能 retry + recover，把 success rate 拉高 25%。

### Failure 分类是关键直觉

Section 4.4 把失败分两类：

**Non-degrading failure**：环境没变，直接 retry 就行。例：抓 bottle 时 offset 一点，没抓到，但 bottle 仍 upright 在原位。

**Degrading failure**：环境被搞乱了，retry 没用。例：抓 bottle 时把 bottle 撞倒了，现在 bottle lying on side，policy 的 precondition 是 "bottle upright"，必须先 recovery（把 bottle 扶起来）才能 retry。

随着 rollout 累积，recovery behavior 也被加进 policy library 当 dedicated recovery policy。**Policy pool 是自扩展的**，越跑越 robust。这跟 Hindsight Experience Replay 思路相通，从失败中提取学习信号，参考 https://arxiv.org/abs/1707.01495。

## 7. Closed-loop lifecycle

部署时生成的 trajectory 也被 record 进 dataset $\mathcal{D}$，回流到训练 pipeline。这些 trajectory 的 state distribution 更接近真实 deployment 场景，比 human demo 更 valuable。

```
human demo (seed, 少量)
      ↓
train policy
      ↓
deploy → collect new data (真实分布) → retrain → deploy → ...
```

无限循环改进。这就是 abstract 说的"lifecycle learning"。

## 8. 实验数据怎么看

### 8.1 四个 task 的难度谱

| Task | 难点类型 | 为什么难 |
|------|---------|---------|
| Body Lotion | Long-range pick-and-place | workspace 大，camera view 在 approach/grasp/lift/place 各阶段视角变化大 |
| Primer | Constrained follow-up | 不仅放准还要能关 drawer，occlusion + clearance 受限 |
| Lipstick | Tight insertion | positional + rotational tolerance 紧 |
| Tissue wipe | Sustained surface contact | 不是 single final pose，而是 continuous motion quality |

### 8.2 Iterative rollout 效果（Table 3）

| Iter | Body Lotion | Primer | Lipstick | Tissue |
|------|-------------|--------|----------|--------|
| 1 | 21/50 | 23/50 | 2/50 | 11/50 |
| 5 | 43/50 | 40/50 | 23/50 | 26/50 |

观察：
- 简单任务翻倍：Body Lotion 42% → 86%
- 难任务天花板明显：Lipstick 4% → 46%，纯靠 data scaling 还不够
- On-policy data 比 human demo 更 informative，因为 robot 自己 rollout 会 explore 到 failure state，policy 学会 "how to recover"，不只是 "how to succeed"

### 8.3 Data collection 效率（Figure 4a/b）

- 同样 data 量，manual baseline 要 2.16× 更多 human time
- Rollout 时 manual baseline 要 8.04× 更多 human intervention

8.04× 说明传统 pipeline 几乎每个 episode 都要人介入 reset/recovery，RoboClaw 大部分 rollout 不需要人。总体 human burden 减少 53.7%。

### 8.4 Long-horizon 对比（Figure 4c）

三个 method 在 vanity table 整理任务上：
1. π_0.5 + no framework (open-loop 顺序执行)
2. 4 个 subtask success rate 的 product (理论 open-loop 上限)
3. RoboClaw (agent orchestration + recovery)

RoboClaw 比 baseline 2 高 25%。这 25% 完全来自 **runtime supervision + recovery**。

## 9. 跟其他工作的关系

### Closed-loop data collection 赛道

| Method | 机制 | RoboClaw 区别 |
|--------|------|--------------|
| AnyTeleop / GELLO / Mobile ALOHA | 纯 teleop | RoboClaw 有 autonomy |
| RoboCopilot | human-in-the-loop residual | RoboClaw 不依赖 human 实时介入 |
| Genie Centurion | rewind-and-refine + Task Sentinel | 检测失败后请人，不能 self-recover |
| FieldGen | attraction field 半自动 | 只自动 pre-manipulation 阶段 |
| MimicGen / RoboCasa | sim 里 synthesize | sim-to-real gap |
| RoboTwin 2.0 | MLLM + sim-in-the-loop | 依赖 simulation |
| CyberDemo | Auto Curriculum Learning | simulation-based |

RoboClaw 独特点：fully learning-driven + autonomous recovery + real-world deployment 三者合一。

参考：
- Mobile ALOHA: https://arxiv.org/abs/2401.02117
- Genie Centurion: https://arxiv.org/abs/2505.18793
- FieldGen: https://arxiv.org/abs/2510.20774
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- MimicGen: https://arxiv.org/abs/2310.10696
- RoboCasa: https://arxiv.org/abs/2406.02523

### Foundation model for embodied

VLA 演进链：PaLM-E → RT-1 → RT-2 → OpenVLA → π_0 → π_0.5

Long-horizon 处理：
- Code as Policies / VoxPoser：LLM planning，无 runtime supervision
- Say-Can：hierarchical + affordance
- Inner Monologue：replanning
- HiRobot：hierarchical VLA
- π_0.5：unified VLA + multi-stage reasoning

RoboClaw 跟它们区别：decoupled from task structure，agent 做 **process-level supervision**，不只是 plan-then-execute。

参考：
- Say-Can: https://arxiv.org/abs/2204.01691
- Code as Policies: https://arxiv.org/abs/2209.07753
- VoxPoser: https://arxiv.org/abs/2307.05973
- Inner Monologue: https://arxiv.org/abs/2207.05608
- HiRobot: https://arxiv.org/abs/2502.19417
- HAMSTER: https://arxiv.org/abs/2502.05485
- OpenVLA: https://arxiv.org/abs/2406.09246

## 10. 几个我个人觉得值得吐槽的点

**Limitation 1：Cloud VLM latency**。每个 timestep 都走 CoT，cloud round-trip 延迟累积起来，fast manipulation task 可能跟不上。Paper 自己承认这点。

**Limitation 2：Inverse 假设**。EAP 假设 inverse task 存在且比 forward 简单。但有些 task 根本没有 sensible inverse——"把鸡蛋打碎"的 inverse 是什么？"把水倒进杯子"的 inverse 是"把水从杯子分离出来"？物理不可逆过程 EAP 直接失效。

**Limitation 3：Reset policy 自己 fail 怎么办**。Paper 没讨论 reset policy 在 OOD state 上 fail 的 recovery 机制。理论上应该有"reset reset policy"，但这会无穷递归。实际中他们 fallback 到 Call Human。

**Limitation 4：Policy pool scale**。现在 4 个 task，scale 到几百个 task 时 orchestration 复杂度爆炸，agent context window 装不下所有 policy metadata。

**Limitation 5：Safety judgment 依赖 agent**。Agent 自己 decide 什么时候 Call Human，如果 agent misjudge（比如把危险情况当成 normal failure 想自己 recover），后果可能严重。工业部署需要独立 safety layer。

## 11. 一句话 intuition

RoboClaw 的核心招式：**用同一个 agent loop 同时跑 data collection 和 deployment**，再用 **forward-inverse policy pair 的不对称性** 实现 autonomous self-reset。Data 天然 on-policy，policy 天然能 recover，整个系统形成 self-improving closed loop。

更深层 insight：**robotic system 的 bottleneck 已经从 policy capacity 转移到 data lifecycle management**。π_0.5 在单 task 上能到 80%+ success rate，model 够强了。真正卡 scaling 的是 system-level design——怎么让 training data 跟 deployment 一致、怎么在 execution 时 monitor + recover、怎么把 deployment 经验回流到 training。

这跟 LLM agent framework (Claude Code, Cursor, Devin) 的设计哲学同源：**用统一 agent loop 替代 hand-crafted pipeline，用 tool calling 替代 rigid interface，用 structured memory 替代 hardcoded state machine**。

参考：
- Claude Code: https://www.anthropic.com/news/claude-code
- Devin: https://www.cognition.ai/blog/introducing-devin

## 12. 可以深挖的联想方向

**跟 reversible computing 的类比**：Landauer's principle 说信息擦除需要耗散能量（$E = kT \ln 2$）。EAP 里"env reset"就对应"信息擦除"——把环境从 high-entropy 状态恢复到 low-entropy 初始态。Paper 用 inverse policy 把这个"擦除成本"从 human labor 转移到 robot 自己的 compute + actuation。参考 https://en.wikipedia.org/wiki/Landauer%27s_principle。

**跟 AlphaGo self-play 的结构相似性**：AlphaGo 用 self-play 生成 experience → policy improvement → 生成更强 experience。RoboClaw 用 EAP self-reset 生成 experience → policy improvement → 生成更 robust experience。区别是 AlphaGo 的 env (Go board) 天然可逆（下完一局重开棋盘即可），RoboClaw 要人工构造可逆性。参考 https://www.nature.com/articles/nature16961。

**跟 DAgger 的关系**：DAgger (Dataset Aggregation) 的核心思想是 on-policy 数据收集——让 expert 在 learner 当前 policy 访问的 state distribution 上提供 label，消除 covariate shift。RoboClaw 的 EAP 闭环本质上就是 robotic 版 DAgger：robot 自己 rollout 产生的 state，agent 通过 tool call 提供"label"（选哪个 policy、什么时候 retry），这些 trajectory 又回流到 training。参考 https://arxiv.org/abs/1011.0686。

**跟 React / ReAct agent 的关系**：RoboClaw 的 observe-reason-act 循环跟 ReAct pattern 几乎一致，区别是 RoboClaw 的 "act" 是调用 VLA policy 产生 motor action，而不是 LLM 产生 text。这预示着 robotic agent 跟 software agent 的设计会越来越融合。参考 https://arxiv.org/abs/2210.03629。

**跟 RL 中的 HRL (Hierarchical RL) 的关系**：RoboClaw 的 Skills/Tools/Policies 三层抽象对应 HRL 的 option/framework。但 RoboClaw 用 VLM 的 in-context learning 替代 learned hierarchical policy，好处是 zero-shot composition，坏处是 latency 高且不能 end-to-end optimize。参考 https://arxiv.org/abs/1604.06057。

如果你看完还想知道哪块更细，比如 flow matching 的数学推导、或 LoRA 在 VLA 上的具体效果、或 EAP 在更多 task category 上的可推广性，我可以再展开。

---

# RoboClaw 深度解读

这篇 paper 解决的是 VLA (Vision-Language-Action) scaling 到 long-horizon 任务时的核心痛点。我先 build 一个 mental model，然后逐层拆解。

## 1. 核心问题：为什么 long-horizon robotic task 这么难 scale

传统 robotic pipeline 把整个 lifecycle 切成三段独立 process：
- **Data collection**: human teleoperation + manual env reset
- **Policy learning**: offline train VLA model
- **Deployment**: 把训练好的 policy 拿去执行

这三段切裂带来两个致命 mismatch：

**(a) Semantic mismatch**: data collector 跟 deployer 是不同的人/系统，对 task state、subtask boundary、success criteria 的理解不一致。比如 collector 觉得"放进 drawer"算成功，deployer 期望"放进去 + 关上 drawer"，policy 学到的 termination 信号就跟 deploy 期望的不一样。

**(b) State distribution mismatch**: training data 是从 human demo 的 state distribution 采样的，但 deploy 时 robot 自己 rollout 出来的 state distribution 完全不同（robot 会犯错，会把 object 撞歪、grasp 滑掉），policy 在 OOD state 上就崩。Long-horizon 下 error cascade，一个 subtask 失败会污染下一个 subtask 的 precondition。

RoboClaw 的核心 insight：**用同一个 VLM-driven agent loop 同时承担 data collection 和 task execution**，这样收集到的数据天然就是 on-policy 的，semantic 也天然一致。这跟 RL 里的 DAgger 思想有共鸣，但 RoboClaw 用 agentic framework 而非 RL algorithm 实现。

参考 DAgger 的原始 paper：https://arxiv.org/abs/1011.0686

## 2. System Architecture 解析

Figure 2 展示的三层 abstraction 是关键设计：

```
Skills (high-level reusable procedures)
   ↓ invokes
Tools (MCP-callable system interfaces: Start Policy, Terminate Policy, Env Summary, Fetch Robot Stats, Call Human, Change Policy)
   ↓ invokes
Policies (low-level VLA models, e.g. π_0.5 fine-tuned)
```

**VLM 作为 meta-controller** 通过 in-context learning 工作。在每个 timestep t，它接收：
- 当前 visual observation $o_t$
- structured memory state $m_t = (r_t, g_t, w_t)$

其中：
- $r_t$ = role identity，告诉 agent 当前是 "data collector" 还是 "task executor" 模式，并定义可用的 tool set。这个 design 很聪明：同一个 agent core 通过切换 role 就能切换行为模式，不需要两个不同的 system。
- $g_t$ = task-level memory，存 global task + decomposed subtasks + 每个 subtask 的 execution status (pending/running/success/failed)。这是 long-horizon 任务能 track progress 的关键。
- $w_t$ = working memory，存当前 active skill + tool invocation history，类似 short-term context。

**CoT reasoning procedure** 分四步：
1. Interpret current scene，识别 environment 中 relevant elements
2. Determine current objective/subtask
3. Evaluate success criteria，assess 当前 state 是否满足
4. Decide next action (retry / switch policy / call recovery / call human / proceed to next subtask)

这个 reasoning 走的是 Anthropic-style 的 MCP (Model Context Protocol) 接口。MCP 的好处是把 tool calling 标准化，agent 可以 hot-swap 工具，参考 https://modelcontextprotocol.io。

## 3. Entangled Action Pairs (EAP) —— 这篇 paper 最核心的 contribution

### 3.1 Intuition

传统 data collection 最贵的是 **environment reset**：每跑一个 trajectory，环境状态就变了（object 被拿走了/放到了别处），需要 human 把东西放回原位才能跑下一个。这个 reset 占了 human labor 的大头。

EAP 的 idea：**对每个 forward manipulation policy $\pi_{\theta_k}$，额外学一个 inverse reset policy $\pi_{\phi_k}$**，两者组成 entangled pair。Robot 先执行 forward action（比如"把 primer 放进 drawer"），然后执行 inverse action（"把 primer 从 drawer 拿出来放回原位"），这样环境自动 reset 到 initial state，可以无限循环。

关键 insight 是 **forward 和 inverse 的 asymmetry**：inverse 任务被故意设计得比 forward 简单。比如：
- Forward "把 lipstick 插进 narrow slot"：需要 precise alignment + rotational tolerance 很紧
- Inverse "把 lipstick 从 slot 拔出来放回桌上"：只要 grasp + lift，tolerance 松很多

Table 2 验证了这个 asymmetry：inverse reset policy 的 success rate 在 36/50 到 43/50 之间，远高于 forward policy 早期 iteration 的 21-23/50。这种 asymmetry 让 self-resetting loop 能稳定持续，因为 reset 比 forward 容易成功，loop 不会卡死。

这跟化学动力学里 irreversible reaction 的 idea 反过来：在这里我们要的是 "容易回去" 而不是 "容易前进"。

### 3.2 公式拆解

公式 (1)-(8) 串起来描述整个 EAP 机制：

**Memory state** (公式 1)：
$$m_t = (r_t, g_t, w_t)$$
- $r_t$: role identity (data collector / task executor)
- $g_t$: task-level memory (global task + subtask status)
- $w_t$: working memory (active skill + tool history)

**Subtask selection** (公式 2)：
$$z_t = \text{RoboClaw}(m_t, o_t), \quad z_t \in \mathcal{Z}$$
- $\mathcal{Z}$: candidate subtask set
- $\text{RoboClaw}(\cdot)$: VLM 通过 CoT 推理选 subtask

**VLA policy** (公式 3, 4)：用 π_0.5 作为 base model
$$A_t = \pi_{0.5}(o_t, l_t, q_t)$$
- $o_t$: visual observation
- $l_t$: language instruction，**关键**：这里 $l_t$ 不是 human 给的，是 RoboClaw agent 在 MCP tool call 时 dynamically 生成的 structured instruction
- $q_t$: robot proprioceptive joint state
- $A_t = [a_t, \dots, a_{t+H-1}]$: action chunk，长度 H，predict short-horizon action sequence

**Flow matching training objective** (公式 5)：
$$\mathcal{L}^{\tau}(\theta) = \mathbb{E}_{p(A_t | o_t, l_t, q_t), q(A_t^{\tau} | A_t)} \left[ \| v_{\theta}(A_t^{\tau}, o_t, l_t, q_t) - u(A_t^{\tau} | A_t) \| ^2 \right]$$

这里细节：
- $\tau \in [0,1]$: flow matching time step，从 noise (τ=0) 平滑过渡到 real action (τ=1)
- $A_t^{\tau} = (1-\tau)\epsilon + \tau A_t$: 线性插值，$\epsilon$ 是 Gaussian noise，$A_t$ 是 ground-truth action chunk
- $v_{\theta}$: 神经网络要学的 velocity field，把 standard Gaussian noise distribution transport 到 true action distribution
- $u(A_t^{\tau} | A_t)$: conditional velocity field target，对于 linear interpolation 来说就是 $A_t - \epsilon$ (constant)
- 训练目标就是让 $v_\theta$ 逼近 $u$

这跟 π_0 / π_0.5 原始 paper 一致，参考 https://arxiv.org/abs/2410.24164 (π_0) 和 https://arxiv.org/abs/2504.16054 (π_0.5)。Flow matching 比 diffusion 的好处是 inference 时 deterministic ODE 而非 SDE，Table 1 显示只用 **3 inference steps** 就够，比 DDPM 的几十步快很多。

**Entangled pair** (公式 6, 7, 8)：

Forward trajectory:
$$\tau_k^{\text{fwd}} = \{(o_t, q_t, a_t)\}_{t=0}^{T}, \quad a_t = \pi_{\theta_k}^{\text{fwd}}(o_t, l_t, q_t)$$

Reset trajectory (从 T+1 开始，到 T+T_reset 结束):
$$\tau_k^{\text{rst}} = \{(o_t, q_t, a_t')\}_{t=T+1}^{T+T_{\text{reset}}}, \quad a_t' = \pi_{\phi_k}^{\text{rst}}(o_t, l_t, q_t)$$

Entangled pair:
$$\boldsymbol{\tau}_k = (\tau_k^{\text{fwd}}, \tau_k^{\text{rst}})$$

注意 forward 和 reset 用的是 **不同的 policy checkpoint** ($\theta_k$ vs $\phi_k$)，但 share 同样的 input interface $(o_t, l_t, q_t)$。$l_t$ 在两种情况下内容不同：forward 时是 "place primer into drawer"，reset 时是 "take primer out of drawer"。

### 3.3 LoRA fine-tuning 细节

Table 1 列的 hyperparameter：
- **Precision**: bfloat16 (节省显存，比 float16 数值稳定)
- **Batch size**: 16
- **Training steps**: 10k
- **Warmup**: 100 steps
- **Learning rate**: $2.5 \times 10^{-5}$
- **Gradient checkpointing**: √ (用 compute 换 memory)
- **LoRA rank r**: 16
- **LoRA alpha α**: 16 (通常 α/r = 1 时 scaling factor = 1，这里 α=r 表示 base model 输出和 LoRA 输出 weight 相当)
- **Dropout**: 0.1 (regularization)
- **Target modules**: all-linear (所有 linear layer 都加 LoRA adapter，aggressive 但参数量仍小)
- **Inference steps**: 3 (flow matching 的 ODE solver step)

LoRA 的好处是每个 task 训一个 adapter，可以 hot-swap，policy pool 容易扩展。参考 LoRA 原始 paper https://arxiv.org/abs/2106.09685。

## 4. Deployment-time Process Supervision

这部分是 long-horizon task 提升 25% 的核心机制。

部署时 agent 做 **runtime supervision**，循环：
1. Fetch Robot Stats / Env Summary → 拿到 environment feedback
2. 写入 $w_t$ (working memory)
3. 评估当前 subtask 是否成功
   - 成功 → 更新 $g_t$，进入下一个 subtask
   - 失败 → 三种应对：
     - Retry same policy (non-degrading failure)
     - Change Policy 切到另一个 forward policy (degrading failure, 但有合适 alternative)
     - Call Human (safety-critical 或 autonomous recovery 失败)

这个 monitor + recover 机制避免了 open-loop pipeline 的 "乘法衰减"。传统 pipeline 4 个 subtask 各 0.7 success rate，串起来期望 $0.7^4 \approx 0.24$。RoboClaw 因为能 retry + recover，success rate 接近 min + recovery bonus，Figure 4(c) 显示明显高于 "product of subtask rates" baseline。

### 4.1 Failure taxonomy

Section 4.4 把 failure 分两类，这是 build intuition 的关键：

**Non-degrading failures**: env state 不变，retry 同一个 policy 就能解决。例：gripper close 时 offset 一点，没 grasp 到 bottle，但 bottle 仍 upright 在原位 → 直接 retry。

**Degrading failures**: env state 改变，retry 会一直失败。例：grasp 时把 bottle 撞倒了 → bottle 现在 lying on side，policy 的 precondition 是 "bottle upright"，必须先 recovery (把 bottle 扶起来) 才能 retry。

随着 rollout 积累，agent 把这些 recovery behavior 也加进 policy library 当成 dedicated recovery policy。这意味着 **policy pool 是 self-expanding 的**，越跑越 robust。

这个 idea 跟 RL 里的 Hindsight Experience Replay (HER) 思想类似，都是从失败 trajectory 中提取学习信号，参考 https://arxiv.org/abs/1707.01495。

### 4.2 Closed-loop lifecycle

部署时生成的 trajectory 也被 record 进 dataset $\mathcal{D}$，作为额外训练数据。这些 trajectory 的 state distribution 更接近真实 deployment，比 human demo 更 valuable。这就形成了 **infinite loop of improvement**：

```
human demo (seed) → train policy → deploy → collect new data (real dist) 
                                              ↓
                                         retrain policy ←─────┘
```

## 5. Experiments 深度分析

### 5.1 Platform

Agibot G01, dual-arm mobile manipulation robot：
- 20 DoF (excluding end-effectors)
- AGIBOT OmniPicker gripper: adaptive, single active DoF (被动 conformant，抓不同 size object 不需要换 gripper)

参考 https://www.agibot.com/

### 5.2 Four subtasks 的 difficulty spectrum

Section 4.2 选的四个 task 故意覆盖不同 difficulty dimension：

| Task | Challenge type | 为什么难 |
|------|---------------|---------|
| Body Lotion | Long-range pick-and-place | workspace 大，camera view 变化大 (approach/grasp/lift/place 阶段视角全变) |
| Primer | Constrained follow-up interaction | 不仅放准还要能 close drawer，occlusion + limited clearance |
| Lipstick | Tight insertion | positional + rotational tolerance 紧，small deviation 就 fail |
| Tissue wipe | Sustained surface contact | 不是 single final pose，而是 continuous motion quality，loss of contact = fail |

这个 spread 很有代表性，覆盖了 manipulation 的主要 difficulty category。

### 5.3 Iterative rollout 的效果

Table 3 显示 5 个 iteration 的 forward policy success rate 变化（每 iteration 加 50 个 sample）：

| Iter | Body Lotion | Primer | Lipstick | Tissue |
|------|-------------|--------|----------|--------|
| 1 | 21/50 (42%) | 23/50 (46%) | 2/50 (4%) | 11/50 (22%) |
| 5 | 43/50 (86%) | 40/50 (80%) | 23/50 (46%) | 26/50 (52%) |

关键观察：
- Body Lotion / Primer 这种 "easier" task，5 iter 后 success rate 接近翻倍
- Lipstick 这种 tight insertion task，5 iter 才到 46%，说明 task 本身有 ceiling，纯靠 data scaling 不够
- Tissue wipe 22% → 52%，continuous motion task 也受益于 on-policy data

为什么 on-policy rollout 比 human demo 更 effective？因为 human demo 倾向于 "happy path"，robot 自己 rollout 会 explore 到 failure state，policy 学到 "how to recover" 而不仅仅是 "how to succeed"。这跟 DAgger 的 on-policy 修正思想一致。

### 5.4 Data collection efficiency

Figure 4(a): 同样 data 量，manual baseline 需要 2.16× 更多 human time
Figure 4(b): rollout 时 manual baseline 需要 8.04× 更多 human intervention

8.04× 这个数字很惊人，说明传统 pipeline 的 rollout 几乎每个 episode 都要 human 介入 reset/recovery，而 RoboClaw 因为 EAP 自重置 + agent autonomous recovery，大部分 rollout 不需要 human。

53.7% human burden reduction 是 abstract 里提到的总体数字。

### 5.5 Long-horizon 对比

Figure 4(c) vanity table organization task 对比三个 method：
1. **Baseline 1**: π_0.5 trained on same dataset, no RoboClaw framework (open-loop, 一串 subtask 顺序执行)
2. **Baseline 2**: 4 个 subtask success rate 的 product (theoretical open-loop expectation)
3. **RoboClaw**: agent orchestration + recovery

RoboClaw 比 baseline 2 高 25% 左右。这 25% 完全来自 **runtime supervision + recovery**，因为 baseline 2 是 "理论上的 open-loop 上限"。

### 5.6 Recovery policy 的学习

Section 4.4 提到 degrading failure 的 recovery behavior 会逐渐被加进 policy library。这意味着 RoboClaw 不只是 "fixed policy pool + orchestration"，而是 **policy pool 随 deployment 自增长**。这是跟 SayCan / HiRobot / HAMSTER 这种 static skill library 方法的关键区别。

参考 SayCan https://arxiv.org/abs/2204.01691, HiRobot https://arxiv.org/abs/2502.19417, HAMSTER https://arxiv.org/abs/2502.05485

## 6. 跟 Related Work 的对比

### 6.1 Closed-loop Data Collection 方向

| Method | 机制 | 跟 RoboClaw 区别 |
|--------|------|-----------------|
| AnyTeleop / GELLO / Mobile ALOHA | 纯 teleop | 没 autonomy |
| RoboCopilot | human-in-the-loop residual correction | 仍依赖 human 实时介入 |
| Genie Centurion | rewind-and-refine + Task Sentinel | detect failure 后请求 human，不能 self-recover |
| VLAC | VLA + critic | 类似 Genie Centurion |
| FieldGen | 半自动，attraction field 生成 pre-manipulation trajectory | 只自动 pre-manipulation 阶段 |
| MimicGen / GenH2R-Sim / RoboCasa | simulation 里 synthesize demo | sim-to-real gap |
| RoboTwin 2.0 | MLLM + sim-in-the-loop 验证 execution code | 仍依赖 simulation |
| HumanoidGen | LLM 生成 spatial constraint + STCR tree search | planning 导向，没 process supervision |
| CyberDemo | Auto Curriculum Learning 动态调 difficulty | simulation-based |

RoboClaw 独特点：**fully learning-driven + autonomous recovery + real-world deployment**。其他 method 要么依赖 human 介入，要么依赖 simulation，要么不处理 runtime failure。

参考：
- Genie Centurion: https://arxiv.org/abs/2505.18793
- FieldGen: https://arxiv.org/abs/2510.20774
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- HumanoidGen: https://arxiv.org/abs/2507.00833
- MimicPlay: https://arxiv.org/abs/2302.12422

### 6.2 Foundation Models for Embodied Tasks

VLA model 演进：PaLM-E → RT-1 → RT-2 → OpenVLA → π_0 → π_0.5

Long-horizon 处理：
- Code as Policies / VoxPoser: LLM 做 planning，无 runtime supervision
- Say-Can: hierarchical + affordance grounding
- Inner Monologue: replanning
- HiRobot: hierarchical VLA
- π_0.5: multi-stage reasoning in unified VLA

RoboClaw 跟这些的区别：**decoupled from specific task structure**，agent 做 process-level supervision 而不仅仅是 plan-then-execute。

参考：
- PaLM-E: https://arxiv.org/abs/2303.03378
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- π_0: https://arxiv.org/abs/2410.24164
- π_0.5: https://arxiv.org/abs/2504.16054
- Code as Policies: https://arxiv.org/abs/2209.07753
- VoxPoser: https://arxiv.org/abs/2307.05973
- Inner Monologue: https://arxiv.org/abs/2207.05608

## 7. Limitations 和未来方向

Paper 自己承认的：
- **Cloud-based VLM latency**: VLM reasoning 在 cloud 上跑，每个决策有 round-trip 延迟，fast manipulation task 可能跟不上
- **Assumption of practical inverse reset**: 有些 task 的 inverse 不存在或比 forward 更难（比如 "把鸡蛋打碎" 的 inverse 是 "把鸡蛋复原"，impossible）。这种 task EAP 不 work

我自己补几个：
- **VLM reasoning cost**: 每 timestep 都做 CoT，long-horizon task 累计 cost 高
- **Reset policy 的 sim-to-real**: paper 没讨论 reset policy 在未见过的 state 上 fail 怎么办（reset policy 自己 fail 怎么 reset reset？）
- **Policy pool scale**: 当前 4 个 task，scale 到几百个 task 时 orchestration 复杂度爆炸
- **Safety**: agent 自己 decide call human，但 if agent misjudge safety situation 怎么办

未来可以扩展到：
- Navigation + mobile manipulation
- Multi-modal interaction (语音、gesture)
- Hierarchical agent (多层 VLM, 类似 HAMSTER)

## 8. Building Intuition: 一句话总结

RoboClaw 的核心 trick 是 **用同一个 agentic loop 把 data collection 和 deployment 统一起来**，并通过 EAP 的 forward-inverse asymmetry 实现 autonomous self-reset。这让收集到的 data 天然 on-policy，policy 天然能 recover，整个 system 形成 self-improving closed loop。

更深层的 insight：**robotic system 的 bottleneck 不在 policy capacity，而在 data lifecycle management**。VLA model 已经足够 strong (π_0.5 在 4 个 task 上能到 40-86% success rate)，但 scaling 到 long-horizon 需要 system-level design 解决 data 分布一致性 + runtime supervision。这跟 LLM 时代的 system 优化思路一致：model 越强，infra/orchestration 越关键。

这跟 Anthropic 的 MCP + Agent 趋势、以及最近 LLM agent framework (Claude Code, Cursor, Devin 等) 的 design philosophy 同源：**用统一 agent loop 替代 pipeline，用 tool calling 替代 hand-crafted interface，用 structured memory 替代 hardcoded state machine**。

参考：
- Anthropic MCP: https://modelcontextprotocol.io
- Claude Code: https://www.anthropic.com/news/claude-code
- Devin: https://www.cognition.ai/blog/introducing-devin

---

如果想再深入，可以考虑：
- 把 EAP 跟 reversible computing 类比 (Landauer's principle: 信息擦除需要能量，类比 env reset 需要 human labor)
- RoboClaw 的 closed-loop lifecycle 跟 AlphaGo 的 self-play 有结构相似性 (self-generated experience → policy improvement → more experience)
- VLM 作为 meta-controller 跟 AlphaZero 的 MCTS 作为 policy improvement operator 类比
