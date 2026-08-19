---
source_pdf: HarnessWAM Bridging Prediction and Deliberation in World Action Models.pdf
paper_sha256: e123b7056eb7011730c4ce5fc289f3df7a74b187228cdb5ba029daf46dddcad9
processed_at: '2026-08-19T10:36:13-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话重讲 HarnessWAM

## 一句话总结

WAM 就像猴子的肌肉记忆——会抓、会放、会拧，但不知道整个任务走到哪了；HarnessWAM 就是给这只猴子配了一个带着笔记本、菜谱和 stopwatch 的前额叶皮层，告诉它"现在做第三步，第二步那个抽屉里确实有红杯子我记得，第四步还没轮到别急"。

## 1. 为什么光有 WAM 不够：肌肉记忆 vs 任务大脑

想象你让一个会做家务但记忆力只有 5 秒的人去完成："打开三个抽屉，找到那个装了东西的，回去把杯子放进去。"

WAM 就是这个人的肌肉能力——它能在给定"抓杯子"这个 local instruction 时，可靠地生成接下来 $H$ 步的 action：

$$A_t = W_\theta(o_{\leq t}, q_t, g_k) = (a_t, \dots, a_{t+H-1})$$

- $o_{\leq t}$: 截止时刻 $t$ 的多视角 RGB 历史（agent view + wrist view）
- $q_t$: 机器人本体状态（关节角、gripper 开合）
- $g_k$: 当前激活的局部 skill 文本（比如 "pick up the red cup"）
- $H$: action chunk 时域长度
- $A_t$: 输出的 $H$ 步动作序列

问题出在哪？任务要求打开抽屉A→看内容→关闭→打开B→看内容→关闭→打开C→...→最后回到那个装东西的抽屉拿杯子放进去。等走到最后一步时，当前 RGB 帧里什么线索都没有——三个抽屉都关着。WAM 看到这一帧，无法知道"目标抽屉是A"这个事实。

这就是论文叫 **prediction–deliberation gap** 的东西。WAM 能预测"接下来 0.5 秒物理会怎么演化"，但无法 deliberation"整个任务的状态是什么、下一步该做哪个 skill、上一步到底成功没、失败了怎么回滚"。

这跟你反复强调的 LLM agent harness 完全是同一个道理。ReAct（https://arxiv.org/abs/2210.03629）、Reflexion（https://arxiv.org/abs/2303.11366）、SWE-agent（https://arxiv.org/abs/2405.15793）都在说一件事：一个 foundation model 的实际能力，不取决于它本身的参数规模，而取决于你给它套的那层 external loop——怎么维持状态、怎么调用工具、怎么吸收反馈。

HarnessWAM 干的事，就是把这个 harness 思想从软件 agent（discrete interface、explicit return value、可重复执行）搬到物理 agent（continuous control、partial observability、不可逆状态）。但物理世界逼着你重新设计 harness——这就是这篇 paper 的全部内容。

## 2. Harness 的核心状态：四个东西

整个 harness 在 WAM 外部维护一个 runtime state：

$$z_k = (B_k, G_k, M_k, r_k)$$

- $B_k$: scene belief，对世界的当前信念
- $G_k$: task graph，结构化任务图
- $M_k$: task memory，跨阶段记忆
- $r_k$: 当前激活 skill 的 execution state

每个 event $\tau_k$（比如一个 subtask 完成、一个失败发生、一个新观测到达），harness 递归更新：

$$(z_{k+1}, g_{k+1}) = \mathcal{H}(x, z_k, o_{\tau_k}, e_k)$$

- $x$: 全局指令
- $o_{\tau_k}$: event 时刻的观测
- $e_k$: 触发 deliberation 的事件
- $g_{k+1}$: harness 选出的下一个 local goal，喂给 WAM

**直觉**：WAM 是肌肉，harness 是大脑。肌肉问"接下来这一小段怎么动"，大脑问"现在做哪个动作、什么时候算完、做错了怎么办"。两者解耦。

下面分别看这四个状态是什么。

## 3. Scene Belief $B_k$：侦探的笔记本

每个 scene fact 是一个元组：

$$f = (s, p, o, v, \eta, c, \mathcal{E})$$

- $s$: subject（主语，比如 "drawer_A"）
- $p$: predicate（谓词，比如 "contains"）
- $o$: object（宾语，比如 "red_cup"）
- $v$: value（属性值，比如 "open" 或 "closed"）
- $\eta \in \{\text{observed, inferred, unknown}\}$: epistemic status
- $c \in [0,1]$: confidence
- $\mathcal{E}$: 支持这个 fact 的 RGB 帧引用

**最关键的设计**：$\eta$ 这个 epistemic status 字段。它区分三种情况：
- **observed**: 我直接看到了
- **inferred**: 我从其他证据推断出来
- **unknown**: 我还没搞清楚

为什么这么重要？因为物理世界有个根本不对称：**"没看到"和"看到为假"是两回事**。抽屉关上之后，当前帧里看不到内容——但这并不意味着"抽屉是空的"，只是"现在观测不到了"。如果没这个区分，harness 一关抽屉就把之前的 fact 丢掉，那就跟没有 memory 一样。

这跟 persistent spatial semantic memory（Blukis et al., https://arxiv.org/abs/2110.05414）的思路一脉相承，也跟你在 podcast 里聊的 "object permanence" 的认知科学概念完全契合——婴儿几个月大就知道"被遮住的东西还存在"，robot agent 也得有这个能力。

## 4. Task Memory $M_k$：双重账本

$$M_k = (M_k^{\mathrm{task}}, M_k^{\mathrm{evidence}})$$

- $M_k^{\mathrm{task}}$: 任务账本——哪些 node 完成了、哪些失败了、retry 了几次、variable 绑定到哪、plan 改过几次
- $M_k^{\mathrm{evidence}}$: 证据账本——跟关键事件绑定的视觉帧

只在 event 时刻调 VLM 做联合更新：

$$(B_{k+1}, M_{k+1}) = \mathcal{U}_{\mathrm{VLM}}(x, o_{\tau_k}, B_k, M_k, e_k)$$

**直觉**：这是显式压缩。把长视频历史压缩成可查询、可追溯的结构化 state，避免让 VLM 每次都从 1000 帧视频里隐式重建"我刚才干了啥"。就像你做数学题不会每次都从头读一遍草稿纸，而是记下关键中间结果。

## 5. Task Graph $G_k$：可修订的菜谱

$$G_k = (V_k, E_k, \mathcal{X}_k, \beta_k)$$

- $V_k$: 节点集，有两类——motor node（产生物理动作）和 cognitive node（采集观测、验证状态、绑定变量、更新记忆）
- $E_k$: 依赖边
- $\mathcal{X}_k$: 未解决的符号变量（比如 "target_drawer" 还不知道是 A/B/C 哪个）
- $\beta_k: \mathcal{X}_k \to \mathcal{O}$: 从变量到具体 scene entity 的 partial binding

每个节点长这样：

$$v_i = (\mathrm{op}_i, \mathrm{arg}_i, \mathrm{pre}_i, \mathrm{eff}_i, \mathrm{term}_i, \mathrm{rec}_i)$$

- $\mathrm{op}_i$: 要做什么操作
- $\mathrm{arg}_i$: 带类型的参数
- $\mathrm{pre}_i$: 前置条件（比如"必须先抓到东西才能放"）
- $\mathrm{eff}_i$: 期望效果
- $\mathrm{term}_i$: 终止条件
- $\mathrm{rec}_i$: 失败时的恢复策略

**最骚的设计**：延迟绑定（delayed binding）。当一开始还不知道"目标抽屉"是哪个时，Task Manager 不会硬猜，而是保留一个 symbolic variable $\mathcal{X} \ni \text{target\_drawer}$，等探索完三个抽屉、攒够证据后再更新 $\beta_k(\text{target\_drawer}) = \text{drawer\_A}$。

这个设计完美对应 Figure 1 那个场景：先打开A→记录内容→关闭→打开B→记录内容→关闭→打开C→记录内容→关闭→**此时当前帧毫无线索，但 task graph 里 target_drawer 已经被 bind 到 A**→重新打开A→拿杯子放进去。

这跟传统 symbolic planning 的最大区别：传统 planner 要求所有变量一开始就 ground，HarnessWAM 允许"先挂起、后解决"，这恰恰是 partial observability 下做 long-horizon 任务的必需能力。

## 6. Executable-Space Projection：编译器的类型检查

这是整篇 paper 最重要的工程贡献。Ablation 显示去掉它，task-level success rate 从 59.6% 暴跌到 18.5%——41 个百分点的差距。

### 问题：VLM 和 WAM 说不同的语言

VLM（这里用的是 Qwen3-VL-32B-Instruct）能生成语义上合理的 plan，比如 "pick up the red cup and place it on the table"。但 WAM 听不懂这种自由语言——它只能执行经过验证的、参数化的 primitive。

### Primitive Ontology

定义一组可扩展的 primitive family：

$$\mathcal{P}^\star = \mathcal{P}_{\mathrm{motion}} \cup \mathcal{P}_{\mathrm{grasp}} \cup \mathcal{P}_{\mathrm{contact}} \cup \mathcal{P}_{\mathrm{articulation}} \cup \mathcal{P}_{\mathrm{assembly}} \cup \mathcal{P}_{\mathrm{tool}}$$

六个 family：自由空间运动、抓取释放、接触富交互、articulated object 操作、装配、工具使用。

每个 primitive 的统一表示：

$$p = (\tau_p, \Theta_p, \mathrm{Pre}_p, \mathrm{Eff}_p, \mathrm{Term}_p, \mathrm{Rec}_p)$$

- $\tau_p$: interaction type
- $\Theta_p$: typed parameter space
- $\mathrm{Pre}_p, \mathrm{Eff}_p, \mathrm{Term}_p, \mathrm{Rec}_p$: 前置、效果、终止、恢复

### 为什么是有限的 ontology

这有强经验证据支撑：
- Bullock et al. 2013（https://ieeexplore.ieee.org/document/6778251）：日常活动中 10 种 grasp 类型占 81% 时长、72% 实例
- Moro et al. 2012（https://doi.org/10.3389/fnbot.2012.00010）：两个 kinematic primitives 解释 95% 的 reaching motion variance
- Santello et al. 1998（https://www.jneurosci.org/content/18/23/10105）：两个 hand synergies 解释 80% 的 15-DoF grasp variance

**直觉**：高维 motor control 表面上复杂，实际上有低维、重复、可组合的结构。所以用有限参数化的行为基元去组织是合理的先验。

### Capability Set

针对特定 WAM，能执行的 primitive 子集：

$$\mathcal{P}_W = \{p \in \mathcal{P}^\star \mid p \text{ has a validated realization under } W_\theta\}$$

意思是：这个 primitive 在这个 WAM 上经验证过能跑通。

### Projection 公式

设 $\mathcal{L}(\mathcal{P}_W)$ 是 supported primitives 生成的 plan language，$\mathcal{F}(z_k)$ 是当前 scene/bindings/dependencies/embodiment state 诱导的可行集。投影：

$$G_k^{\mathrm{exec}} = \Pi_{\mathcal{L}(\mathcal{P}_W) \cap \mathcal{F}(z_k)}(G_k^{\mathrm{vlm}})$$

**人话**：把 VLM 生成的开放语义图，折叠到"WAM 能力 ∩ 当前可行"的交集里。

### Compiler 检查什么

一个 deterministic plan compiler 检查：
- argument types 对不对
- node dependencies 满足不满足（图无环性）
- precondition–effect 一致性（比如 PLACE 前必须 GRASP 同一个物体）
- single-arm holding state（一个手只能拿一个东西）
- capability constraints（WAM 训练过这个 skill 没）

不满足就返回 $\bot$，报告违反的约束，让 Task Manager replan。

### History Invariance

新观测可能改变未来分支，但不能改变过去：

$$G_{k+1}[V_k^{\mathrm{executed}}] = G_k[V_k^{\mathrm{executed}}]$$

- $V_k^{\mathrm{executed}}$: 已执行节点子集
- $G_k[V_k^{\mathrm{executed}}]$: 图在已执行节点上的限制

**直觉**：已经发生的事不能反悔。这跟物理因果性一致——打翻的水收不回来，重规划不能假装它没发生。这条约束让 plan revision 只动未执行后缀，保留已执行前缀的因果完整性。

## 7. Dual-Timescale Control：心跳和深思

### 为什么不能每步都调 VLM

两个理由：
1. **太贵**：Qwen3-VL-32B 每步都调，real-time 性能崩溃
2. **太躁**：瞬时光照变化、手抖一下，VLM 就可能误判

所以搞双时间尺度。

### Fast Loop: Progress Estimator

轻量模型 $F_\phi$，每步都跑：

$$(p_t, c_t, \pi_t^{\mathrm{bin}}) = F_\phi(o_{t-L+1:t}, g_k)$$

- $o_{t-L+1:t}$: 最近 $L=5$ 帧 dual-view RGB
- $g_k$: 当前 skill 文本
- $p_t \in [0,1]$: 连续 progress（"做到 0.7 了"）
- $c_t \in [0,1]$: stage completion 概率
- $\pi_t^{\mathrm{bin}}$: progress interval 上的离散分布

架构：frozen SigLIP2-base-patch16-256 编码器（提取多视角空间特征）+ 4 层 causal Transformer（建模局部时序）。

训练目标六项联合：

$$\mathcal{L}_{\mathrm{prog}} = \lambda_r \mathcal{L}_{\mathrm{reg}} + \lambda_b \mathcal{L}_{\mathrm{bin}} + \lambda_r \mathcal{L}_{\mathrm{rank}} + \lambda_e \mathcal{L}_{\mathrm{endpoint}} + \lambda_s \mathcal{L}_{\mathrm{success}} + \lambda_m \mathcal{L}_{\mathrm{mono}}$$

- $\mathcal{L}_{\mathrm{reg}}$: 连续 progress 回归（主任务）
- $\mathcal{L}_{\mathrm{bin}}$: progress interval 分类（粗粒度监督）
- $\mathcal{L}_{\mathrm{rank}}$: 时序 pairwise ranking（保证 $t_1 < t_2 \Rightarrow p_{t_1} < p_{t_2}$）
- $\mathcal{L}_{\mathrm{endpoint}}$: 轨迹端点锚定（开始 progress=0，结束 progress=1）
- $\mathcal{L}_{\mathrm{success}}$: stage completion 二分类
- $\mathcal{L}_{\mathrm{mono}}$: 局部单调性（progress 不能倒退）

**关键约束**：progress estimator **永远不直接推进 task graph**，只生成 candidate milestone event 去触发 slow loop 的 VLM。

### Slow Loop: Task Manager Deliberation

在 candidate event $e_k$ 处调 VLM：

$$y_k = (\nu_k, d_k, \Delta B_k, \Delta \beta_k, \rho_k) = \mathcal{T}_{\mathrm{VLM}}(x, o_{\tau_k}, B_k, G_k, M_k, r_k, \hat{s}_k^{\mathrm{prog}}, e_k)$$

- $\nu_k \in \{\text{success, failure, uncertain}\}$: 物理效果标签
- $d_k \in \{\text{continue, advance, observe, replan, recover, terminate}\}$: 执行决策
- $\Delta B_k, \Delta \beta_k$: scene belief 和 bindings 的增量更新
- $\rho_k$: 是否需要修订未执行后缀

**六个决策分支的直觉**：
- **continue**: 还没做完，继续干
- **advance**: 做完了，激活下一个 node
- **observe**: 不确定，先看一眼再说
- **replan**: 当前策略无效，改 plan
- **recover**: 局部搞砸了，回滚本体再来
- **terminate**: 没救了，认输

只有 $d_k = \mathrm{advance}$ 时才推进任务图。这个保守策略保证了不会因为 progress estimator 的假阳性而错误跳步。

### 为什么 progress-only 不够

Ablation 里有个 progress-only switching 变体——让 progress estimator 直接决定 subtask 完成。结果很微妙：

| 方法 | Avg SR_task | Avg SR_sub |
|---|---|---|
| HarnessWAM | 59.6 | 69.9 |
| progress-only switching | 55.4 | 68.4 |

subtask SR 几乎不变（-1.5），但 full-task SR 掉 4.2。在 sequential family 上更夸张：subtask 97.9%（甚至比 HarnessWAM 的 93.0% 高！），但 full-task 只有 72.5%（vs HarnessWAM 的 86.3%）。

**直觉解释**：progress 是 appearance-level signal（"看起来抓得更紧了"），但 task 完成是 semantic-level predicate（"目标物体已经在目标容器里"）。这俩经常 dissociate：
- 手在动但物体没被抓起来
- 抓到了但放错位置
- 看起来到位了但实际还差一点

纯 progress 会把"看起来做完了"判为完成，跳到下一步，但目标状态没达成，整条 chain 最终失败。subtask SR 高是因为每个 subtask "看起来"都做完了；full-task SR 低是因为最终目标根本没实现。

这就是为什么必须用 VLM 做 semantic-level outcome verification——它问的不是"看起来怎么样"，而是"目标状态达成了没"。这个区别对应 runtime consistency monitoring（Agia et al., https://arxiv.org/abs/2410.04640）里 "progress ≠ success" 的核心观察。

## 8. Embodiment Recovery：游戏的 checkpoint

每个 motor node 开始时记录：
- $q_k^0$: arm joint state
- $u_k^0$: gripper state
- $T_k$: 执行 budget

失败时（$d_k = \mathrm{recover}$），多步控制把本体拉回去：

$$(q_t, u_t) \xrightarrow[\text{multi-step control}]{\text{recover}} (q_k^0, u_k^0)$$

**反直觉的核心**：recovery **只重置机器人本体**，**保留环境、scene belief、所有累积信息**。然后清空 local WAM state，可以选择重试当前 node 或换一个 plan 后缀。

### 为什么这么设计

物理世界有两个不对称性：
1. 机器人本体状态完全已知、完全可控（关节角、gripper 开合）
2. 环境状态部分可观、可能不可逆（打翻的水、撕碎的纸）

所以 recovery 策略只动可控的、保留不可逆的。这跟传统 RL 的 episode restart 完全不同——传统做法是整个 episode 重来，前面探索到的信息全丢。

这也呼应了前面 history-invariance 约束的物理动因：环境已经发生了不可逆变化，harness 必须维护这个"已经发生了什么"的不可篡改历史。

### Recovery 之后干嘛

清空 local WAM state 后，Task Manager 有两个选择：
1. **Retry current node**：用修订过的 local goal 再试一次（比如换一种 grasp type）
2. **Replace suffix**：当前策略根本无效，换一个 plan 后缀

每次尝试的结果都记到 $M_k^{\mathrm{task}}$，让后续 recovery 决策能依赖累积证据和历史失败，而不是固定的 per-node retry count。

这跟 FLARE（https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_FLARE）等 failure-aware correction 框架的区别：HarnessWAM 的 recovery 是 model-external 的、state-preserving 的、graph-suffix-revising 的，是结构化决策而非简单 retry。

## 9. Algorithm 1 的人话解读

```
输入: 全局指令 x, 初始观测 o_0, WAM, task budget Ω
输出: success 或 failure

1. 初始化 scene belief 和 memory
2. VLM 生成初始 task graph G^vlm
3. 把 G^vlm project 到 WAM 可执行的 G
4. while budget 没耗尽 且 G 还有未完成 required node:
   5. 选一个 ready node v
   6. if v 是 cognitive node:
      7. 采集 evidence, 更新 B/M/bindings/图后缀
      8. 标记 v 完成, 重新 project 未执行部分
      9. continue
   10. else (v 是 motor node):
      11. 记录当前 (q^0, u^0), 激活 v
      12. while v 还在激活 且 budget 没耗尽:
         13. WAM 生成 action chunk A_t
         14. 执行 A_t, 观测 o_{t+1}, 扣 budget
         15. progress estimator 算 s^prog
         16. if 触发 event:
            17. VLM 决策 (ν, d, ΔB, Δβ, ρ)
            18. 更新 B/M/bindings, 可能修订后缀
            19. switch d:
               - advance: 标记完成, break
               - observe: 补观测
               - replan: 修订+reproject 后缀, 暂停 v
               - recover: 还原本体, clear WAM state, reproject recovery plan
               - terminate: return failure
34. if 所有 required node 完成 且 VERIFYGOAL(x, o, B):
35.    return success
36. else:
37.    return failure
```

**直觉**：这是个 event-driven 的递归状态估计系统。Cognitive node 负责"看和想"，motor node 负责"动手"。每个 motor node 内部有个 fast loop（WAM + progress estimator）跑高频控制，外部有个 slow loop（VLM Task Manager）在 milestone 处做 deliberation。失败时回滚本体但保留记忆，重新规划只动后缀。

## 10. 实验说话

### RoboMemArena 主结果

RoboMemArena（https://arxiv.org/abs/2605.10921）：26 个长时程任务，平均轨迹 1076 步，68.9% subtask 依赖历史。四个 family：transfer (4)、occlusion (11)、counting (7)、sequence (4)。

| Method | Avg SR_task | Avg SR_sub |
|---|---|---|
| π0.5（https://arxiv.org/abs/2504.16054） | 21.5 | 38.7 |
| HiF-VLA | 16.9 | 39.8 |
| MemoryVLA（https://arxiv.org/abs/2506.15738） | 15.0 | 35.3 |
| MemER | 27.3 | 49.1 |
| PrediMem | 38.5 | 55.2 |
| WAM + Whole Task | 44.4 | 52.3 |
| WAM + Static Plan | 47.9 | 62.0 |
| **HarnessWAM** | **59.6** | **69.9** |

几个有意思的观察：

**第一**，WAM + Whole Task（直接把全局指令喂给 LingBot-VA）就比 PrediMem 高 6 个点。这本身说明 WAM 这种 predictive representation 有不错的 inductive bias，但远没饱和。

**第二**，Static Plan（只在初始化时分解一次，之后不更新）比 Whole Task 略高，证明 task decomposition 这个 prior 本身有用——但提升有限（+3.5/+9.7）。

**第三**，HarnessWAM 比 Static Plan 高 +11.7/+7.9，证明纯分解解释不了增益，必须靠 persistent state + closed-loop management。

**第四**，Transfer family 上 HarnessWAM 的 subtask SR 只有 31.5%，远低于其他 family。但 full-task SR 仍是最高（21.3%）。这暗示 transfer 任务单步难度高，HarnessWAM 在 subtask 切换时更保守导致部分 subtask 卡住，但整体能更完整地组合到终点。

**第五**，Counting family 上 subtask SR 88.2% 略低于 Static Plan 的 90.1%，但 full-task +6.4。HarnessWAM 让"必须数够次数"这类任务更可靠地走完全程，而不是某个子段单独拿高分但整体崩盘。

### RoboCerebra Ideal

RoboCerebra（https://arxiv.org/abs/2412.03514）Ideal subset：长时程组合操作 + 高层推理，static 且 fully observable。

| Method | Ideal SR |
|---|---|
| π0.5 | 1.88 |
| OpenVLA（https://arxiv.org/abs/2406.09246） | 7.84 |
| GPT-4o Planner + OpenVLA | 21.92 |
| HPE Framework | 21.10 |
| **HarnessWAM** | **23.70** |

关键：Ideal 是 static + fully observable，所以 HarnessWAM 的 +1.78/+2.60 提升不能归因于 memory recovery。增益来自 dependency-aware planning、outcome-conditioned transitions、failure-aware adaptation。这证明 harness 的收益不局限于 memory-dependent 任务——即便没 memory 问题，结构化规划 + 执行验证 + 失败恢复依然有用。

### Ablation：谁是关键

按 task-level 影响排序：

| Ablation | Avg SR_task | Avg SR_sub | Δ task |
|---|---|---|---|
| HarnessWAM | 59.6 | 69.9 | — |
| w/o executable projection | 18.5 | 38.3 | **-41.1** |
| w/o progress events | 38.3 | 50.7 | -21.3 |
| w/o task state | 47.7 | 61.1 | -11.9 |
| w/o recovery | 54.2 | 67.7 | -5.4 |
| progress-only switching | 55.4 | 68.4 | -4.2 |

**Executable projection 是绝对核心**：去掉它掉 41 个点。这印证了核心论点——VLM 能产生语义合理的 plan，但如果 operators/arguments/dependencies/embodiment-state transitions 不符合 WAM 执行接口，根本无法可靠 instantiate。

**Progress events 第二重要**：去掉它掉 21 个点。固定频率调 VLM 是 execution-aware event selection 的劣质替代——你需要在"事情正在发生"的时刻 deliberation，而不是每隔 N 步机械地 deliberation。

**Task state 第三**：去掉它掉 12 个点。最大退化在 occlusion family，正好是需要保留被遮挡证据的场景。

**Recovery 第四**：去掉它掉 5 个点。最大退化在 sequential family（-11.3 full-task），因为长 sequence 中一个 unrecovered 局部失败会让整个剩余后缀作废。

### Plan-Level Diagnosis：lex vs semantic

直接看中间 plan 质量：

| Plan representation | Syntax | Dependencies | Binding | Executability |
|---|---|---|---|---|
| Raw VLM plan | 60.8 | 58.1 | 21.3 | 13.8 |
| + normalization & aliasing | 84.6 | 67.5 | 63.8 | 42.3 |
| + executable-space projection | 95.2 | 92.9 | 88.3 | 72.9 |

**两阶段贡献**：
- Lexical normalization（词汇归一化 + entity alias resolution）：syntax +23.8, binding +42.5, executability +28.5。解决表面语义接口不匹配。
- Capability/dependency/binding/embodiment-state constraints：executability 再 +30.6。解决真正的约束满足。

**直觉**：lex normalization 解决"名字对不上"——"pickup" vs "grasp"、"那个红色的" vs "red_cup_3"。但解决不了"前置条件不满足"——WAM 要求抓取前手是空的、place 前必须抓到、open 前抽屉必须是关的。后者必须靠真正的约束检查。

这跟 SWE-agent 的 agent-computer interface 设计哲学一致：interface 上的约束让 raw LLM output 变成可执行程序，光做 string matching 远远不够。

## 11. 把它映射回你熟悉的 LLM agent

这套设计跟软件 agent 框架是一一对应的：

| LLM agent | HarnessWAM |
|---|---|
| LLM | WAM（局部执行能力） |
| ReAct reasoner | Task Manager VLM |
| Tool call | WAM invocation |
| Scratchpad / working memory | Scene belief $B_k$ |
| Plan + execution trace | Task graph $G_k$ |
| Tool schema validation | Executable-space projection |
| Cheap runtime critic | Progress estimator |
| Retry with rollback | Embodiment recovery |

HarnessWAM 的本质贡献是把这些软件 agent 的设计原则系统化地迁移到机器人控制，并针对物理世界的特殊性做了三处关键改造：

1. **Scene belief 用 epistemic status 区分 observed/inferred/unknown**——物理世界的 partial observability 要求"没看到"和"看到为假"必须区分
2. **Executable projection 强制 capability 和 embodiment 约束**——WAM 的能力边界是硬约束，不像软件 tool 那么灵活
3. **Recovery 只重置本体不重置环境**——物理世界有不可逆状态，不能像软件 retry 那样直接 episode restart

## 12. 直觉构建：为什么这套设计 work

### 12.1 为什么 model 外部要维护结构化状态

VLM 的 context window 是有限的，长视频塞进去会被稀释。更糟的是，VLM 从长视频里重建历史是隐式的、不可靠的——它可能"忘记"第三个抽屉里有什么。

显式维护 $B_k$（结构化 fact）+ $M_k$（任务账本 + 证据）相当于给 VLM 一个外部记忆 prosthetic。每次 deliberation 时，VLM 只需要看当前 RGB + 这个结构化 state，不用从头重建。这跟 ReAct 让 LLM 把中间推理写到 scratchpad 是同一个道理——把 working memory 外化到可查询的结构。

### 12.2 为什么需要 dual timescale

物理控制有个根本矛盾：
- 高频控制需要每步反馈（否则 WAM 跑飞了不知道）
- 高频调 VLM 不现实（太贵 + 太躁）

Dual timescale 的精髓：让 cheap model（progress estimator）跑高频做"看起来怎么样"的监测，让 expensive model（VLM）跑低频做"该怎么办"的决策。这跟人类认知的 System 1 / System 2 划分完全对应——System 1（progress estimator）快速、自动、廉价，System 2（VLM Task Manager）慢速、深思熟虑、昂贵。

### 12.3 为什么 projection 这么重要

VLM 是开放语义空间里的规划者，WAM 是受限能力空间里的执行者。两者之间有巨大 gap：
- VLM 说"pick up the red cup"，WAM 的 grasp primitive 要求具体的 grasp type 参数
- VLM 假设"手是空的可以抓"，但当前 gripper 可能已经持有别的物体
- VLM 假设"red cup 在桌子上"，但它可能还没被 bind 到具体 scene entity
- VLM 假设"可以直接 place"，但前置节点（如打开柜子）可能还没完成

Projection 就是这个 gap 的桥。它不是简单的 string matching，而是真正的约束满足——检查类型、依赖、前置条件、能力边界。这跟编译器的类型检查完全一样：raw source code 看起来可能合理，但只有通过类型检查才能保证可执行。

### 12.4 为什么 recovery 只重置本体

物理环境的根本不对称：
- 机器人本体：完全已知、完全可控
- 环境：部分可观、可能不可逆

所以 recovery 策略是"动可控的，保不可逆的"。这跟软件 retry 不一样——软件 retry 可以直接整个 episode restart，因为软件状态都是可重建的。物理世界不行，打翻的水收不回来。

这也意味着 harness 必须维护一个"已经发生了什么"的不可篡改历史——这就是 history-invariance 约束 $G_{k+1}[V_k^{\mathrm{executed}}] = G_k[V_k^{\mathrm{executed}}]$ 的物理动因。已经发生的不能反悔，重规划只能动未来。

## 13. 局限和我想得到的延伸

论文承认的：
- 只在仿真 benchmark 上验证，sim-to-real 未触及
- WAM skill repertoire 有限，ontology $\mathcal{P}^\star$ 需要随能力扩展
- Deliberation 没有显式 calibrated uncertainty，success/failure/uncertain 是 VLM 主观判断

我会额外想到的：

**Latency 问题**。虽然 event-driven，但每个 milestone 都要调一次 Qwen3-VL-32B，real-time 性能论文没报。在 real robot 上，这个延迟可能让 fast loop 的控制频率掉下来。可能的解法：speculative deliberation（让 progress estimator 预测下一个 event，提前调 VLM）或者 VLM 蒸馏到更小模型。

**Ontology 的封闭性**。Plan compiler 是 deterministic 的，遇到 ontology 之外的 skill 直接返回 ⊥。没有 graceful degradation——如果 VLM 提出一个新 primitive（比如"翻转物体"），compiler 直接拒绝。可能的解法：meta-learn the compiler，或者让 compiler 学会 primitive composition（用已有 primitive 组合新行为）。

**Error propagation**。Variable binding 依赖 VLM 视觉识别能力。长时程任务里，如果某个 binding 错了（比如把 red_cup 错认为 blue_cup），这个错误会沿着 task graph 传播，导致后续所有依赖这个 binding 的 node 都失败。可能的解法：binding uncertainty estimation + active verification。

**Meta-recovery**。Recovery 本身可能失败——multi-step joint control 把本体拉回 $(q_k^0, u_k^0)$ 也可能卡住。论文没讨论 recovery 失败怎么办。可能的解法：hierarchical recovery，每层有自己的 fallback。

**学习闭环**。Harness 产生的成功轨迹可以作为 finetuning 数据，反过来提升 WAM 内部能力。这形成"model 外部结构化经验 → model 内部能力提升"的闭环。这跟 you've talked about 的 "system 2 → system 1 distillation" 思路一致——deliberation 产生的 good trajectories 蒸馏回 fast policy。

## 14. 总结性直觉

HarnessWAM 的核心论点可以浓缩成一句话：**Model 的局部能力（WAM 的 finite-horizon prediction + action generation）和 task-level 可靠性之间存在结构性 gap，这个 gap 不能靠把 model 做得更大来解决，而要靠在 model 外部构建一个结构化的、event-driven 的、state-preserving 的 agent runtime**。

这跟你反复强调的 "system 2 thinking"、"agentic loop"、"test-time compute" 完全契合——只是这里 test-time compute 落在了一个有结构化状态空间 ($B_k, G_k, M_k, r_k$)、有约束满足层、有 dual timescale 的具体架构上。

值得追踪的后续方向：
1. Real robot 验证，看 sim-to-real gap
2. Calibrated uncertainty for deliberation
3. Learn the compiler 而非手工设计 ontology
4. Hierarchical harness：多层 task graph，每层有自己的 timescale
5. Harness 产生数据 → WAM finetuning 的闭环

## Reference 汇总

- ReAct: https://arxiv.org/abs/2210.03629
- Reflexion: https://arxiv.org/abs/2303.11366
- SWE-agent: https://arxiv.org/abs/2405.15793
- SayCan: https://arxiv.org/abs/2204.01691
- Inner Monologue: https://arxiv.org/abs/2207.05608
- VoxPoser: https://arxiv.org/abs/2307.05973
- SayPlan: https://arxiv.org/abs/2307.06135
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- RoboCerebra: https://arxiv.org/abs/2412.03514
- Runtime monitoring (Agia et al.): https://arxiv.org/abs/2410.04640
- MemoryVLA: https://arxiv.org/abs/2506.15738
- RoboMemArena: https://arxiv.org/abs/2605.10921
- Bullock grasp taxonomy: https://ieeexplore.ieee.org/document/6778251
- Santello synergies: https://www.jneurosci.org/content/18/23/10105
- Blukis persistent spatial memory: https://arxiv.org/abs/2110.05414
- FLARE: https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_FLARE
- Ebert retrying: https://arxiv.org/abs/1810.05043

---

# HarnessWAM：在 World Action Models 上构建 Agent Harness

## 1. 核心问题：Prediction–Deliberation Gap

World Action Models (WAMs) 把 future-observation prediction 和 action generation 耦合在一起，本质上是学习一个局部、有限时域 (finite-horizon) 的条件分布。形式上，给定观测历史 $o_{\leq t}$、本体状态 $q_t$ 和一个局部 skill 指令 $g_k$，WAM 输出一段 action chunk：

$$A_t = W_\theta(o_{\leq t}, q_t, g_k) = (a_t, \dots, a_{t+H-1}) \quad (1)$$

变量含义：
- $W_\theta$: 参数为 $\theta$ 的 WAM（这里用的是 LingBot-VA）
- $o_{\leq t}$: 截止到时刻 $t$ 的多视角 RGB 历史，$o_t = (I_t^{\mathrm{agent}}, I_t^{\mathrm{wrist}})$，包含 agent view 和 wrist view
- $q_t$: 机器人本体感知状态（关节角等）
- $g_k$: 当前激活的局部 skill 文本指令
- $H$: action chunk 的预测时域长度
- $A_t$: 从 $t$ 开始的 $H$ 步动作序列

**Gap 的本质**：WAM 只解决"在当前 skill 下接下来 $H$ 步做什么"，而 embodied task 需要的是跨阶段的 persistent state、outcome verification、failure recovery、以及 partial observability 下的 evidence accumulation。例如 RoboMemArena 里有"先打开抽屉看内容，关闭后再回去拿目标物体"这类 memory-dependent 任务——当前 RGB 帧已经不包含目标信息，必须靠外部 state 维护。

这个 gap 类似于 LLM agent 的"harness 问题"：ReAct (https://arxiv.org/abs/2210.03629)、Reflexion (https://arxiv.org/abs/2303.11366)、SWE-agent (https://arxiv.org/abs/2405.15793) 都表明，foundation model 的有效能力强烈依赖于它通过哪个外部循环来维持状态、调用工具、吸收反馈。HarnessWAM 把这套思路迁移到 physical interaction——但物理环境多了 continuous control、partial observability、irreversible state change，所以 harness 本身要重新设计。

## 2. 整体架构：Model-External 的递归状态估计

HarnessWAM 把 WAM 视作一个局部 executor，自己在 WAM 外部维护一个 runtime state：

$$z_k = (B_k, G_k, M_k, r_k) \quad (2)$$

- $z_k$: 第 $k$ 个 task event 时刻的 runtime state
- $B_k$: scene belief（场景信念）
- $G_k$: 结构化 task graph（任务图）
- $M_k$: task memory（任务记忆）
- $r_k$: 当前激活 skill 的 execution state

每个 event $\tau_k$，harness 递归更新状态并选择下一个 local goal：

$$(z_{k+1}, g_{k+1}) = \mathcal{H}(x, z_k, o_{\tau_k}, e_k) \quad (3)$$

- $x$: 全局自然语言指令
- $\mathcal{H}$: harness 决策过程
- $e_k$: 触发当前 deliberation 的事件

这个 formulation 关键点：**continuous WAM control 和 discrete task-level deliberation 解耦**。WAM 负责"怎么把这一段 skill 做出来"，harness 负责"做哪个 skill、什么时候终止、结果如何改变任务状态、失败后怎么办"。

## 3. Evidence-Grounded Task State

### 3.1 Scene Belief

每个 scene fact 是一个结构化元组：

$$f = (s, p, o, v, \eta, c, \mathcal{E}) \quad (4)$$

- $s$: subject（主语实体）
- $p$: predicate（谓词，如 "inside", "open"）
- $o$: object（宾语实体）
- $v$: value（属性值）
- $\eta \in \{\text{observed, inferred, unknown}\}$: epistemic status，区分"直接看到"、"推断得到"、"未解决"
- $c \in [0,1]$: confidence score
- $\mathcal{E}$: supporting RGB evidence 引用

核心设计意图：**区分"未观察到"和"观察到为假"**。一旦抽屉被关闭，当前帧不再显示内容，但之前已建立的 fact 仍然有效。这是 partial observability 下做 long-horizon 推理的基本要求，类似 persistent spatial semantic memory (Blukis et al., https://arxiv.org/abs/2110.05414)。

### 3.2 Task Memory

$$M_k = (M_k^{\mathrm{task}}, M_k^{\mathrm{evidence}}) \quad (5)$$

- $M_k^{\mathrm{task}}$: completed/failed nodes、retry counts、variable bindings、plan-revision history
- $M_k^{\mathrm{evidence}}$: 与 salient events 关联的视觉证据

VLM 只在 event 时刻被调用做联合更新：

$$(B_{k+1}, M_{k+1}) = \mathcal{U}_{\mathrm{VLM}}(x, o_{\tau_k}, B_k, M_k, e_k) \quad (6)$$

这是一个显式压缩：把长视频历史压缩成可查询、可追溯的 task state，减少 VLM 隐式从长上下文重建历史的负担。

### 3.3 Task Graph

$$G_k = (V_k, E_k, \mathcal{X}_k, \beta_k) \quad (7)$$

- $V_k$: 节点集（包含 motor nodes 和 cognitive nodes）
- $E_k$: 依赖边
- $\mathcal{X}_k$: 未解决的符号变量
- $\beta_k: \mathcal{X}_k \to \mathcal{O}$: 从变量到 scene entities 的 partial binding

每个节点：

$$v_i = (\mathrm{op}_i, \mathrm{arg}_i, \mathrm{pre}_i, \mathrm{eff}_i, \mathrm{term}_i, \mathrm{rec}_i) \quad (8)$$

- $\mathrm{op}_i$: operation 类型
- $\mathrm{arg}_i$: typed arguments
- $\mathrm{pre}_i$: preconditions（前置条件）
- $\mathrm{eff}_i$: expected effects（期望效果）
- $\mathrm{term}_i$: termination conditions（终止条件）
- $\mathrm{rec}_i$: recovery strategy（恢复策略）

**关键 trick**：当实体无法从初始观测识别时，Task Manager 保留一个 symbolic variable 而不强行 commit 一个无依据的猜测，等后续证据足够时再更新 $\beta_k$。这就是 Figure 1 那个"目标抽屉一开始不知道是哪个"的任务能够成功的机制——延迟绑定 (delayed binding)。

## 4. Capability-Conditioned Executable-Space Projection

这是这篇论文最重要的工程贡献，ablation 显示去掉它 task-level SR 从 59.6% 暴跌到 18.5%。

### 4.1 Primitive Ontology

$$\mathcal{P}^\star = \mathcal{P}_{\mathrm{motion}} \cup \mathcal{P}_{\mathrm{grasp}} \cup \mathcal{P}_{\mathrm{contact}} \cup \mathcal{P}_{\mathrm{articulation}} \cup \mathcal{P}_{\mathrm{assembly}} \cup \mathcal{P}_{\mathrm{tool}} \quad (9)$$

六个 primitive family：自由空间运动、抓取释放、接触富交互、articulated object 操作、装配、工具使用。每个 primitive 形式化：

$$p = (\tau_p, \Theta_p, \mathrm{Pre}_p, \mathrm{Eff}_p, \mathrm{Term}_p, \mathrm{Rec}_p) \quad (10)$$

- $\tau_p$: interaction type
- $\Theta_p$: typed parameter space
- 其余同节点定义

这套 ontology 的 motivation 来自 motor behavior 的低维结构证据：
- Bullock et al. 2013 (https://ieeexplore.ieee.org/document/6778251)：日常活动中 10 种最常见 grasp 占 81% 时长、72% 实例
- Moro et al. 2012 (https://doi.org/10.3389/fnbot.2012.00010)：两个 kinematic primitives 解释 95% 的 discrete reaching motion variance
- Santello et al. 1998 (https://www.jneurosci.org/content/18/23/10105)：两个 hand synergies 解释 80% 的 15-DoF grasp variance
- Morrow & Khosla 1997：刚体之间的相对运动类是有限的

这是把高维控制组织成有限参数化基的强先验。

### 4.2 Capability Set

针对特定 WAM，可执行 primitive 集合由经验验证的 skills 决定：

$$\mathcal{P}_W = \{p \in \mathcal{P}^\star \mid p \text{ has a validated realization under } W_\theta\} \quad (11)$$

### 4.3 Projection

设 $\mathcal{L}(\mathcal{P}_W)$ 是 supported primitives 生成的 plan language，$\mathcal{F}(z_k)$ 是当前 scene / bindings / dependencies / embodiment state 诱导的 feasible set。HarnessWAM 把 VLM 生成的图投影到这两个集合的交集：

$$G_k^{\mathrm{exec}} = \Pi_{\mathcal{L}(\mathcal{P}_W) \cap \mathcal{F}(z_k)}(G_k^{\mathrm{vlm}}) \quad (12)$$

投影由 deterministic plan compiler 实现，检查：
- argument types
- node dependencies（图无环性）
- precondition–effect consistency
- single-arm holding state（例如 PLACE 要求先抓到对象）
- capability constraints

不满足则返回 $\bot$ 并报告违反的约束，让 Task Manager replan。

### 4.4 History Invariance

新观测可能改变变量绑定或未来分支，但为了保持与物理历史的一致性，只允许修订未执行的后缀：

$$G_{k+1}[V_k^{\mathrm{executed}}] = G_k[V_k^{\mathrm{executed}}] \quad (13)$$

- $V_k^{\mathrm{executed}}$: 已执行节点子集
- $G_k[V_k^{\mathrm{executed}}]$: 图在已执行节点上的限制

直觉：已经发生的事不能因为重新规划而"反悔"，避免因果不一致。

## 5. Progress-Conditioned Dual-Timescale Control

### 5.1 Fast Loop: Progress Estimator

每步都调 VLM 太贵且会被瞬时视觉波动干扰。HarnessWAM 在 fast loop 用一个轻量 progress estimator $F_\phi$：

$$(p_t, c_t, \pi_t^{\mathrm{bin}}) = F_\phi(o_{t-L+1:t}, g_k) \quad (14)$$

- $o_{t-L+1:t}$: 最近 $L=5$ 帧 dual-view RGB
- $g_k$: 当前激活 skill 文本
- $p_t \in [0,1]$: 连续 progress
- $c_t \in [0,1]$: stage completion 概率
- $\pi_t^{\mathrm{bin}}$: progress interval 上的离散分布

架构：frozen SigLIP2-base-patch16-256 编码器提取多视角空间特征 + 4 层 causal Transformer 建模局部时序。

训练目标六项联合：

$$\mathcal{L}_{\mathrm{prog}} = \lambda_r \mathcal{L}_{\mathrm{reg}} + \lambda_b \mathcal{L}_{\mathrm{bin}} + \lambda_r \mathcal{L}_{\mathrm{rank}} + \lambda_e \mathcal{L}_{\mathrm{endpoint}} + \lambda_s \mathcal{L}_{\mathrm{success}} + \lambda_m \mathcal{L}_{\mathrm{mono}} \quad (15)$$

- $\mathcal{L}_{\mathrm{reg}}$: 连续 progress 回归
- $\mathcal{L}_{\mathrm{bin}}$: progress interval 分类
- $\mathcal{L}_{\mathrm{rank}}$: 时序 pairwise ranking（保证时间顺序一致性）
- $\mathcal{L}_{\mathrm{endpoint}}$: 轨迹端点锚定
- $\mathcal{L}_{\mathrm{success}}$: stage completion 预测
- $\mathcal{L}_{\mathrm{mono}}$: 局部单调性（progress 不应该倒退）

这套设计借鉴了 runtime monitoring of consistency and progress 的思路 (Agia et al., https://arxiv.org/abs/2410.04640)，以及 Maeda et al. 2020 (https://ieeexplore.ieee.org/document/9340999) 的 appearance-invariant progress embedding。

**关键约束**：progress estimator 永远不直接推进 task graph，只生成 candidate milestone events 触发 slow loop 的 VLM deliberation。

### 5.2 Slow Loop: Task Manager Deliberation

在 candidate event $e_k$ 处：

$$y_k = (\nu_k, d_k, \Delta B_k, \Delta \beta_k, \rho_k) = \mathcal{T}_{\mathrm{VLM}}(x, o_{\tau_k}, B_k, G_k, M_k, r_k, \hat{s}_k^{\mathrm{prog}}, e_k) \quad (16)$$

- $\nu_k \in \{\text{success, failure, uncertain}\}$: 物理效果标签
- $d_k \in \{\text{continue, advance, observe, replan, recover, terminate}\}$: 执行决策
- $\Delta B_k, \Delta \beta_k$: scene belief 和 bindings 的增量更新
- $\rho_k$: 是否需要修订未执行的后缀

只有 $d_k = \mathrm{advance}$ 时当前节点才被标记完成、后继节点才被激活。

## 6. Embodiment-State Recovery

这是另一个反直觉但很重要的设计。每个 motor node 开始时记录：

- $q_k^0$: arm joint state 初始值
- $u_k^0$: gripper state 初始值
- $T_k$: 节点执行 budget

当 $d_k = \mathrm{recover}$，多步控制把本体拉回初始状态：

$$(q_t, u_t) \xrightarrow[\text{multi-step control}]{\text{recover}} (q_k^0, u_k^0) \quad (17)$$

注意：recovery **只重置机器人本体**，保留环境、scene belief、之前累积的信息。然后清空 local WAM state，可以选择重试当前节点（带修订的 local goal）或替换未执行后缀为替代策略。

这与传统 RL/控制里的 retry 机制（如 Ebert et al. 2018a, https://arxiv.org/abs/1810.05043）以及 FLARE (https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_FLARE) 等 failure-aware correction 框架形成对比：HarnessWAM 的 recovery 是 model-external 的、state-preserving 的、graph-suffix-revising 的，而不是简单重试或 episode 重启。

## 7. 整体推理流程（Algorithm 1 解读）

伪代码核心循环：

1. 初始化 $(B, M)$，VLM 生成 $G^{\mathrm{vlm}}$，project 到 $G$
2. while budget 未耗尽且 $G$ 含未完成 required node:
   - select ready node $v$
   - 若是 cognitive node：acquire evidence → update state → reproject 后缀
   - 若是 motor node：
     - 记录 $(q^0, u^0)$，激活 $v$
     - while $v$ active 且 budget 充足:
       - WAM 生成 $A_t$，执行，观测 $o_{t+1}$
       - progress estimator 算 $\hat{s}^{\mathrm{prog}}$
       - 若触发 event：Task Manager 决策
         - advance: 标记完成，break
         - observe: 补充观测
         - replan: 修订并 reproject 后缀
         - recover: 还原本体，clear local state，reproject recovery plan
         - terminate: 返回 failure
3. 最终 verify goal：所有 required node 完成且视觉验证通过 → success

## 8. 实验数据深度解读

### 8.1 RoboMemArena 主结果（Table 1）

RoboMemArena (https://arxiv.org/abs/2605.10921)：26 个长时程任务，平均轨迹 1076 步，68.9% 子任务依赖历史信息。四个 task family：multi-object transfer (4)、occlusion (11)、counting (7)、sequential execution (4)。

| Method | Avg SR_task | Avg SR_sub |
|---|---|---|
| π0.5 (https://arxiv.org/abs/2504.16054) | 21.5 | 38.7 |
| HiF-VLA | 16.9 | 39.8 |
| MemoryVLA (https://arxiv.org/abs/2506.15738) | 15.0 | 35.3 |
| MemER | 27.3 | 49.1 |
| PrediMem | 38.5 | 55.2 |
| WAM + Whole Task | 44.4 | 52.3 |
| WAM + Static Plan | 47.9 | 62.0 |
| **HarnessWAM** | **59.6** | **69.9** |

几个值得注意的点：

1. **WAM + Whole Task 比 PrediMem 还高**：单纯把 LingBot-VA 直接条件化在全局指令上就已经超过一票专门设计的 memory baseline，说明 WAM 这种 predictive representation 本身有不错的 inductive bias，但还远未饱和。

2. **Static Plan 比 Whole Task 略高**：仅在初始化时做一次性分解就有 +3.5/+9.7 的提升，证明 task decomposition 这个 prior 本身有用。

3. **HarnessWAM vs Static Plan (+11.7/+7.9)**：纯分解无法解释增益，必须靠 persistent state 和 closed-loop management。

4. **Transfer family 上 HarnessWAM subtask 只有 31.5%，低于 WAM + Whole Task 的 12.5%**——这个反常值得思考。可能 transfer 任务单步难度高、HarnessWAM 在 subtask 切换时更保守导致部分 subtask 卡住。但 full-task SR 仍是最高（21.3%），说明它能更完整地组合。

5. **Counting family**：HarnessWAM subtask 88.2% 略低于 Static Plan 的 90.1%，但 full-task +6.4 points。这意味着 progress-conditioned switching 让 counting 这种"必须数够次数"的任务更可靠地完成全程，而不是在某个子段单独拿高分但整体崩盘。

### 8.2 RoboCerebra Ideal（Table 2）

RoboCerebra (https://arxiv.org/abs/2412.03514) Ideal subset：长时程组合操作 + 高层推理。

| Method | Ideal SR |
|---|---|
| π0.5 | 1.88 |
| OpenVLA (https://arxiv.org/abs/2406.09246) | 7.84 |
| GPT-4o Planner + OpenVLA | 21.92 |
| HPE Framework | 21.10 |
| **HarnessWAM** | **23.70** |

因为 Ideal 是 static、fully observable 的，所以 HarnessWAM 的 +1.78/+2.60 提升不能归因于 memory recovery，而是来自 dependency-aware planning、outcome-conditioned transitions、failure-aware adaptation——这证明 harness 的收益不局限于 memory-dependent 任务。

### 8.3 Ablation（Table 3）

按 task-level 影响大小排序：

| Ablation | Avg SR_task | Avg SR_sub | Δ task | Δ sub |
|---|---|---|---|---|
| HarnessWAM | 59.6 | 69.9 | — | — |
| w/o executable projection | 18.5 | 38.3 | **-41.1** | **-31.6** |
| w/o progress events | 38.3 | 50.7 | -21.3 | -19.2 |
| w/o task state | 47.7 | 61.1 | -11.9 | -8.8 |
| w/o recovery | 54.2 | 67.7 | -5.4 | -2.2 |
| progress-only switching | 55.4 | 68.4 | -4.2 | -1.5 |

**最重要的 ablation 是 executable projection**：-41.1 points。这印证了核心论点——VLM 能产生语义合理的 plan，但如果 operators / arguments / dependencies / embodiment-state transitions 不符合 WAM 执行接口，就根本无法可靠 instantiate。

**progress-only switching 很有意思**：subtask SR 几乎不变（68.4 vs 69.9），但 full-task SR 掉了 4.2 points。在 sequential family 上甚至出现 subtask 97.9% > full-task 72.5% 的诡异现象。直觉解释：纯 progress 估计会把局部"看起来做完了"判为完成，跳到下一步，但实际目标状态没达成，导致整条链最终失败。这强烈支持"必须用 VLM 做语义级 outcome verification"。

**w/o recovery** 在 sequential 上跌得最狠（-11.3 full-task），因为长 sequence 中一个 unrecovered 局部失败会让整个剩余后缀作废。

### 8.4 Plan-Level Diagnosis（Table 4）

直接对比中间 plan 的质量：

| Plan representation | Syntax | Dependencies | Binding | Executability |
|---|---|---|---|---|
| Raw VLM plan | 60.8 | 58.1 | 21.3 | 13.8 |
| + normalization & aliasing | 84.6 | 67.5 | 63.8 | 42.3 |
| + executable-space projection | 95.2 | 92.9 | 88.3 | 72.9 |

两个阶段：
1. **Lexical normalization**（词汇归一化 + entity alias resolution）解决表面语义接口不匹配：syntax +23.8, binding +42.5, executability +28.5。但 dependency 和 executability 仍停留在 67.5%/42.3%。
2. **Capability/dependency/binding/embodiment-state constraints** 进一步把 executability 推到 72.9%（+30.6）。

这区分了"lexical alignment"和"executable plan construction"——前者是 surface form 修正，后者是真正约束满足。这与 SWE-agent 的 agent-computer interface 设计哲学一致：interface 上的约束让 raw LLM output 变成可执行程序。

## 9. 直觉构建：为什么这套设计 work

### 9.1 类比 LLM agent harness

把这套设计映射回你熟悉的 LLM agent 框架：

- **WAM ≈ tool call**：局部、有限时域、可观测反馈
- **Task Manager ≈ ReAct reasoner**：决定下一步调哪个 tool
- **Scene belief $B_k$ ≈ scratchpad / working memory**：跨 step 的持久状态
- **Task graph $G_k$ ≈ plan + execution trace**：结构化、可修订
- **Executable projection ≈ tool schema validation**：约束 LLM 输出符合 tool signature
- **Progress estimator ≈ cheap runtime critic**：高频信号，触发 expensive reasoning
- **Embodiment recovery ≈ retry-with-rollback**：失败时回滚到 checkpoint 而非重启 episode

HarnessWAM 的本质贡献是把这些软件 agent 的设计原则系统化地迁移到机器人控制，并针对物理世界的特殊性（连续控制、partial observability、不可逆状态）做了三处关键改造：scene belief 用 epistemic status 区分 observed/inferred/unknown；executable projection 强制 capability 和 embodiment 约束；recovery 只重置本体不重置环境。

### 9.2 为什么 progress estimator 不能单独推进任务

这是 progress-only switching ablation 的核心教训。Progress 是 appearance-level signal——"看起来抓得更紧了"——但 task 完成是 semantic-level predicate——"目标物体已经在目标容器里"。两者经常 dissociate：手可能在动但物体没被有效抓取，或者抓到了但放错了位置。所以 progress estimator 提供触发信号，VLM 提供 semantic verification，缺一不可。这呼应了 runtime consistency monitoring (https://arxiv.org/abs/2410.04640) 中"progress ≠ success"的观察。

### 9.3 为什么必须区分 lexical 和 executable

VLM 输出 "pick up the red cup" 看起来合理，但：
- WAM 的抓取 primitive 可能要求具体的 grasp type 参数
- 当前 gripper state 可能已经持有别的物体（holding state constraint）
- "red cup" 可能还没被 bind 到具体 scene entity（symbolic variable 状态）
- 该操作的前置节点（如打开柜子）可能还没完成（dependency）

Lexical normalization 解决"名字对不上"，但解决不了"前置条件不满足"。Executable projection 强制做这些约束检查，把 VLM 的开放语义空间折叠到 WAM 的可行能力空间。

### 9.4 为什么 recovery 只重置本体

物理环境有两个不对称性：
1. 机器人本体状态完全已知、可控（关节角、gripper 开合度）
2. 环境状态部分可观、可能不可逆（打翻的水收不回来）

所以 recovery 策略只重置可控的、保留不可逆的，这与传统 RL 的 episode restart 完全不同。这也意味着 harness 必须维护一个"已经发生了什么"的不可篡改历史——这就是 $G_{k+1}[V_k^{\mathrm{executed}}] = G_k[V_k^{\mathrm{executed}}]$ 这个 history-invariance 约束的物理动因。

## 10. 局限与开放方向

论文承认的局限：
- 仅在仿真 benchmark（RoboMemArena、RoboCerebra Ideal）上验证，sim-to-real gap 未触及
- WAM skill repertoire 有限，Ontology $\mathcal{P}^\star$ 需要随能力增长扩展
- Deliberation 没有显式 calibrated uncertainty，failure/uncertain 的区分是 VLM 主观判断

我会额外想到的潜在问题：
- Task Manager VLM 调用频率：虽然 event-driven，但每个 milestone 都要调一次 Qwen3-VL-32B，real-time 性能堪忧（论文没报 latency）
- Plan compiler 的 deterministic 性质是双刃剑：遇到 ontology 之外的 skill 就直接 ⊥，没有 graceful degradation
- Variable binding 依赖 VLM 视觉识别能力，长时程任务里 evidence 累积错误会传播
- Recovery 的 multi-step joint control 本身可能失败，论文没讨论 meta-recovery

## 11. 相关工作映射

- **SayCan** (https://arxiv.org/abs/2204.01691)：用 affordance grounding language plan，是 executable projection 的早期形式，但没有 structured state 和 recovery
- **Inner Monologue** (https://arxiv.org/abs/2207.05608)：引入环境反馈到 language planning，对应 event-driven feedback，但没解决 partial observability
- **VoxPoser** (https://arxiv.org/abs/2307.05973)：在 3D scene geometry 上 ground reasoning，是 scene belief 的几何版本
- **SayPlan** (https://arxiv.org/abs/2307.06135)：3D scene graph grounded planning，最接近 HarnessWAM 的 $G_k$ 表示
- **MemoryVLA** (https://arxiv.org/abs/2506.15738)：perceptual-cognitive memory in VLA，把 memory 内化到 model；HarnessWAM 反向选择外化到 harness
- **PrediMem**：RoboMemArena 上的 SOTA baseline，把 prediction 用于 memory；HarnessWAM 的 progress estimator 借鉴了这个思路但用途更受限
- **Fast-WAM** (https://arxiv.org/abs/2603.16666)、**DreamZero** (https://arxiv.org/abs/2602.15922)、**Himem-WAM** (https://arxiv.org/abs/2606.10363)：都是 WAM 内部改进，与 HarnessWAM 正交——前者提升 model 内部能力，后者提升 model 外部 orchestration

## 12. 总结性直觉

HarnessWAM 的核心论点可以浓缩成一句话：**Model 的局部能力（WAM 的 finite-horizon prediction + action generation）和 task-level 可靠性之间存在结构性 gap，这个 gap 不能靠把 model 做得更大来解决，而要靠在 model 外部构建一个结构化的、event-driven 的、state-preserving 的 agent runtime**。

这与你在 tweet 和 podcast 里反复强调的"system 2 thinking"、"agentic loop"、"test-time compute"的思想完全契合——只是这里 test-time compute 落在了一个有结构化状态空间 ($B_k, G_k, M_k, r_k$)、有约束满足层 (executable projection)、有层级化时间尺度 (dual-timescale) 的具体架构上。

值得追踪的后续方向：
1. 把 harness 扩展到 real robot，验证 sim-to-real
2. Calibrated uncertainty for deliberation（把 VLM 的"uncertain"标签变成概率分布）
3. 把 executable projection 学出来而非手工设计 ontology（meta-learning the compiler）
4. Hierarchical harness：多层 task graph，每层有自己的 timescale
5. 与 RL finetuning 结合：用 harness 产生的成功轨迹作为 finetuning 数据，反过来提升 WAM 内部能力——形成"model 外部结构化经验 → model 内部能力提升"的闭环

参考链接汇总：
- Paper 本身：通过你提供的附件
- ReAct: https://arxiv.org/abs/2210.03629
- Reflexion: https://arxiv.org/abs/2303.11366
- SWE-agent: https://arxiv.org/abs/2405.15793
- SayCan: https://arxiv.org/abs/2204.01691
- Inner Monologue: https://arxiv.org/abs/2207.05608
- VoxPoser: https://arxiv.org/abs/2307.05973
- SayPlan: https://arxiv.org/abs/2307.06135
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- RoboCerebra: https://arxiv.org/abs/2412.03514
- Runtime monitoring (Agia et al.): https://arxiv.org/abs/2410.04640
- MemoryVLA: https://arxiv.org/abs/2506.15738
- Bullock grasp taxonomy: https://ieeexplore.ieee.org/document/6778251
- Santello synergies: https://www.jneurosci.org/content/18/23/10105
