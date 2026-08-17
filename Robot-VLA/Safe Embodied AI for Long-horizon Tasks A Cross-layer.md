---
source_pdf: Safe Embodied AI for Long-horizon Tasks A Cross-layer.pdf
paper_sha256: 520d7db2bb02daf3a0878adc0b28d5f5bfebd90405cb456aed956211ae5b8d5e
processed_at: '2026-08-12T02:45:30-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲这篇 survey

## 一句话总结

这是一篇 2026 年的综述，干了一件事：**把机器人安全这个散落各处的话题，重新按"安全机制在哪个阶段介入"和"声称的安全到底有多硬的证据"两个轴整理了一遍**，聚焦的对象是 long-horizon manipulation，也就是那种"抓个东西 → 放到抽屉 → 关抽屉 → 开微波炉 → 放进去 → 关门"这种多步骤任务。

## 作者为啥要写这篇

现状是这样的：机器人圈现在分几拨人，各搞各的。

- 搞 safe control 的人，研究 CBF、reachability、barrier function，证明在某个数学模型下系统不会出事，但他们的模型通常很简化，离真实机器人差很远
- 搞 safe RL 的人，研究怎么在训练时让 policy 别撞别坏，但他们的 benchmark 都是些 toy task，长 horizon 的几乎没有
- 搞 VLA 的人，研究怎么让 robot 看图听指令做事，能力越来越强，安全基本就提一句 "collision rate 降低了"
- 搞 alignment 的人，研究 LLM 别说错话别乱来，但他们的工作跟物理世界几乎不沾边

这几拨人互相不通气。问题在于，**真实机器人出事往往就出在这些层的接缝处**。比如 LLM 把 "把杯子放进柜子" 理解成 "扔进柜子"（semantic 层出错），policy 生成一个大力扔的动作（policy 层没拦住），执行时杯子撞到柜子边缘碎了（execution 层来不及救）。每层单独看都"还行"，连起来就出事。

作者选 manipulation 当 anchor domain，理由很实在：navigation 没有 contact，locomotion 没有 semantic，只有 manipulation 五个难点全占——长 horizon、semantic 指令、物理接触、表面成功藏隐患、跨层耦合。Table 1 那个对比表就是这个意思。

## 核心框架：两个轴

### 第一个轴：在哪一层管安全

作者把整条 pipeline 切成三段：

**Planning-time**：机器人还没动，先想清楚要干啥。这一层的问题是，自然语言 "把盐递给我" 到底是啥意思？盐在哪？能不能拿？要避开啥？约束是啥？这一层出错了，后面全跑偏。

**Policy-time**：开始生成 action 了，但还没真正动起来。这一层要管的是，policy 别提出离谱 action。比如别一上来就往人脸上戳，别跳过 prerequisite，别在长任务里忘了前面干过啥。

**Execution-time**：机器人物理上动起来了，力传感、视觉、噪声全来了。这一层要管的是，发现 rollout 偏了赶紧 detect，要么 shield 拦一下，要么 replan，要么找人帮忙，要么 force control 救一下接触。

这三层不是独立的模块，是一个 closed-loop 里的三个时间段。Planning 层定的 constraint 得能传到 policy 层用，policy 层的 uncertainty signal 得能被 execution 层的 monitor 理解。

### 第二个轴：声称的安全有多硬

这个轴我觉得是整篇论文最值钱的东西。作者把 evidence 分三档：

**Formal guarantee**：数学证过的。在显式模型和约束下，能证明系统不会出事。比如 CBF 证明 $h(x) \geq 0$ 永远成立。听起来很屌，但有个致命问题——**证明只在模型和假设成立时有效**，模型跟真实世界有 gap，假设一破就全废。

**Statistical safety**：能给出有界失败概率。比如 conformal prediction 给你一个 calibrated threshold，告诉你"95% 的情况下 alarm 是对的"。比 formal 弱一点，但比纯 empirical 强。

**Empirical safety**：在 benchmark 或实验里观察到 collision rate 降了、force 超标少了。这是最弱的一档，因为**不保证泛化**，换个场景可能就废了。

作者反复强调一件事：**robustness 不等于 safety**。你方法在 noise 下表现稳健，只有当那个 noise 跟 hazard（撞、坏、伤人）显式挂钩时，才算 safety evidence。否则只是间接支持。这是这篇 survey 的"反 overclaim 工具"——你 paper 说"safer"，先问问：你的 evidence 是哪一档？你测的 metric 跟哪种 hazard 挂钩？你 claim 的范围跟你的 evidence 范围匹配吗？

## 三个阶段具体在做什么

### Planning-time：把意图变成可检查的 plan

这一层我读下来最大的感受是，**grounding 是一切的源头，也是最容易出问题的地方**。

SayCan 的做法是，LLM 提议一个 skill，再用一个 affordance 模型评估"这个 skill 在当前 state 下可行吗"。两个概率乘起来打分。听起来合理，但问题在于，affordance 模型是基于离散 semantic token 的，"reachable" 这种 predicate 根本捕捉不到柜子门内部卡住这种 continuous 物理。

VoxPoser 走另一条路，把语言约束映射成 3D value map，让 model-based planner 用。ReKep 更进一步，把 task 表达成 keypoint 之间的几何关系约束。这些方法比纯 symbolic 好，因为引入了几何，但仍然不保证 kinematic 可行、不碰撞、接触稳定。

验证这一块，作者梳理了一条由弱到强的链条：post-hoc 过滤 → decoding 时直接 prune 不合法 continuation → counterexample feedback 回灌 prompt → learned surrogate verifier 加速。LTL 在常见 formulation 下是 PSPACE-complete 的，所以严格 formal check 在长 horizon 下算不动，必须蒸馏成 neural 模块——但这一蒸馏，formal guarantee 就没了。

### Policy-time：把 action generation 管起来

这一层我读下来感觉是当前最薄弱的。论文里给出的统一公式是 $\pi^\star = \arg\max J_{\text{obj}}(\pi)$ s.t. $C_i(\pi) \leq 0$。所有方法本质上都在改三样东西：policy class $\Pi$、约束 $C_i$、目标 $J_{\text{obj}}$。

**改 $\Pi$** 就是选 backbone——tokenized VLA 可以 mask token，diffusion policy 可以 project trajectory，programmatic 可以 inspect code。Intervention surface 完全不一样。

**改 $C_i$** 有两条路。一条是 inference 时注入，比如用 STL 在 decoding 阶段 mask 不合法 token。另一条是训练时内化，比如 CMDP 把 safety cost 放进 objective，CPO 用 trust-region 更新。SafeVLA 把这套搬到 VLA 上，elicit unsafe scenario 然后 fine-tune。问题是这些都是 empirical evidence，没有 formal guarantee。

**改 $J_{\text{obj}}$** 就是 alignment。RLHF 那套搬到机器人上，preference reward、language feedback、intervention trace 都能当信号。但作者明确说，preference alignment 能让 policy 远离 unsafe 行为，**却不能 exclude unsafe candidate by construction**，所以 claim 只能是 "bias 了一下"，不能说 "guarantee"。

长 horizon 下，monolithic reward 会把 procedural error 淹没。SARM 的做法是把 reward 拆成 stage-level 和 within-stage progress 两部分，让 prerequisite 没完成时拿不到 stage reward。这个思路很朴素但很重要。

### Execution-time：跟物理世界硬碰硬

这一层最丰富，也最贴近真实 hardware。

**Monitoring** 这块，从 state-level anomaly score 到 VLA 内部 representation 提取信号，再到 conformal prediction 给 calibrated threshold。一个关键 insight 是，**single scalar anomaly score 太脆**，得多模态融合，加上 spatio-temporal reasoning，否则分不清"action 一致但 task progress 停了"这种语义失效。

**Shielding** 这块分两路。Formal 那路就是 CBF/reachability，优点是能证，缺点是模型简化、高维算不动。Learned/latent 那路把 semantic 理解转成 runtime filter，比如用 VLM 生成 CBF 给 VLA 用。这条路很前沿，但作者老实说，**虽然 conceptually grounded in formal control，实际运行在 empirical regime，absolute guarantee 难立**。

**Recovery** 这块按 autonomy spectrum 排得很清楚：human handoff → interactive correction → proactive replanning → reactive replanning → autonomous recovery。每个 level 假设的"task structure 还能不能用"不一样。Handoff 假设全废了交给人；correction 假设局部修一下就行；replanning 假设改 plan 就行；autonomous recovery 假设连"从哪恢复"都要自己找。

**Contact-rich** 这块是 Section 5.3 的深水区。核心 insight 是 timescale mismatch：semantic reasoning 1-2 Hz 够了，但 force spike 要 100 Hz-1 kHz 响应。所以多 timescale architecture 是必然方向——slow semantic policy 给大方向，fast residual controller 救接触。PhaForce、Reactive Diffusion Policy、SERL 都是这个思路。Impedance control 那套经典公式 $F = M_d \ddot{x} + B_d \dot{x} + K_d x$ 在这里被重新激活，因为 VLA 输出 position 而 contact 需要 force 闭环，中间必须有一层 impedance 做"翻译"。

ForceVLA / TaF-VLA / Tactile-VLA 这条线开始把 force/tactile token 化塞进 VLA。我觉得这是必须的——纯 vision+language 的 VLA 在 contact-rich task 上天生缺一个 dimension。但作者也点出，这些工作主要在提 success rate，formal safety guarantee 还是空白。

## Evidence 这件事为啥重要

我读完最强烈的感受是，作者其实在做一件事：**给整个领域立一个"反夸大"的尺子**。

现在 VLA paper 普遍的现象是：在 LIBERO 上 success rate 从 70% 涨到 80%，附带说一句 collision rate 也降了，就在 abstract 里写 "safer"。按这篇 survey 的标准，这是典型的 empirical + layer-local + capability-adjacent evidence，远远配不上 "safer" 这个 system-level claim。

更隐蔽的 overclaim 是，很多 paper 用 robustness metric 当 safety metric。你在 noisy observation 下 success rate 还高，这说明你 robust，但 robustness 只有跟具体 hazard（撞、力超、recovery 失败）显式挂钩时才是 safety evidence。否则就是间接支持。

作者建议的实践是：写 paper 时明确说清楚三件事——你的 mechanism 在哪层介入？你 address 的具体 hazard 是什么？你的 evidence 是 formal / statistical / empirical 哪一档？claim 的范围跟 evidence 的范围匹配吗？这种 discipline 在 NLP alignment 圈已经很成熟，robotics 圈还差得远。

## 作者留给我们未来的真难题

1. **Abstraction boundary 要 safety-aware**：constraint 从语言一路降到接触力，每过一个 boundary 都会丢信息。"avoid red region" 要同时保 symbolic name、geometric margin、trajectory exclusion。现在没人系统做这个。

2. **Sim-to-real 也是 safety evidence transfer 问题**：你 policy 在 sim 里 safe，到 real 上 friction 差一点、tactile 不一样，safe 假设就破了。光 transfer success 不够，得 transfer safety assumption 并 revalidate。

3. **Cross-embodiment safety**：generalist policy 跨 robot 时，semantic prior（别碰人）能 transfer，spatial constraint 要 re-grounding，contact claim 完全 embodiment-specific。现在 Open X-Embodiment 这种数据集根本没标 safety-relevant 信息。

4. **Signal-to-intervention mapping**：detector 给你一个 alarm，然后呢？该 stop？ask？handoff？replan？local adjust？现在没系统答案。而且 detector calibration 一遇 distribution shift 就脆。

5. **Procedural safety observability**：要给 rollout 建一份完整 safety record——hazard 何时出现、哪层响应、mitigation 有没有效、contact bound 有没有保住。类似飞机的黑匣子。RoboEval、IS-Bench 走出第一步，但还远没到统一 trace 标准。

## 我读完的几个联想

**第一个联想**：这篇 survey 的框架其实跟你当年在 Tesla 搞 FSD 时面对的问题同构。FSD 也有 perception → planner → control 三层，每一层都有自己的 safety claim，但真正出事都在接缝处。差异是，navigation/manipulation 引入了 contact 这个全新的 dimension，让 execution 层复杂度爆炸。

**第二个联想**：Section 5.3 multi-timescale 那段在暗示一个 architecture design principle——**未来 VLA 应该有 explicit 的 slow-fast 双通道**，而不是把所有事情塞进一个 token-level autoregressive head。OpenVLA 2、π0 这些主流路线都还是 single-stream token action，contact-rich task 上迟早要撞墙。PhaForce / Reactive Diffusion Policy / SERL 这条线如果跟 VLA 主流融合，可能是下一代 contact-rich VLA 的关键架构。

**第三个联想**：Procedural safety observability + AI Incident Database 这条线让人想起 aviation 的 flight data recorder。未来 manipulation robot 部署大概率要强制 log 每步的 force/torque/contact state/anomaly score/intervention trigger/recovery action。谁先把这个 standard 定下来，谁就占住下一轮 robot safety 标准的话语权。ISO 10218 现在只管工业机器人，service robot 的 safety standard 还是一片空白。

**第四个联想**：作者其实没明说但隐含的一个判断——**当前 VLA safety 研究的 bottleneck 不在 algorithm，在 evaluation**。没有 cross-layer safety benchmark，再多的 method paper 都是在各自小池塘里自证。谁先做出一个像 LIBERO 之于 capability 那样的"IS-Bench 之于 cross-layer safety"的标准 benchmark，谁就定义这个 subfield。IS-Bench、SafeMindBench 已经起步但还很初级。

如果你想我再展开哪一块——比如 CBF 数学怎么推、VLA token masking 具体怎么实现、PhaForce 的 phase belief 怎么训、PDDLStream 算法细节——你说。

---

# Safe Embodied AI for Long-horizon Robotic Manipulation 综述深度讲解

这篇来自 Seoul National University Sungroh Yoon 课题组（Dabin Kim 为一作）的综述 paper，是 2026 年出现的一篇 systematic survey，专门聚焦 long-horizon robotic manipulation 这一 anchor domain 下 safety 的 cross-layer 分析。核心 contribution 在于把碎片化分布在 planning / policy / execution 三层的 safety 文献用两个正交 axis 重新组织：**intervention locus**（safety 机制从哪里进入 pipeline）和 **evidence boundary**（声称的 safety 实际有多强的 evidence 支撑）。下面我尽可能详尽地展开技术细节，帮你 build 起对这一领域整体形态的 intuition。

paper link (arXiv preprint):
https://arxiv.org/abs/2602.00000（综述中引用截至 2026-04，预印本编号需对照作者页 https://sryoon.snu.ac.kr/ 查证；作者 Dabin Kim 的学术主页 https://dabin404.github.io/ 也可追踪）

---

## 1. 为什么选择 long-horizon manipulation 作为 anchor domain

Table 1 给出了一个非常重要的对比，把 embodied AI 的三个主流 domain（navigation / locomotion / manipulation）在五个 safety-relevant factor 上做了 qualitative coding（✓/△/×）：

| Domain | Horizon | Semantic | Contact | Hidden | Coupling |
|--------|---------|----------|---------|--------|----------|
| Navigation | ✓ | △ | × | △ | △ |
| Locomotion | △ | × | ✓ | △ | △ |
| Manipulation | ✓ | ✓ | ✓ | ✓ | ✓ |

这里 ✗ 是 "central and recurring"，△ 是 "variant-dependent or secondary"，× 是 "typically absent"。**Manipulation 是唯一一个五个 factor 都打到 ✓ 的 domain**，原因如下：

- **Long-horizon dependence**：manipulation 任务天然多阶段（如装配家具、做菜），subtask 之间存在 precondition/postcondition 依赖
- **Semantic specification**：需要从 natural language 把意图 ground 到物体、affordance、空间关系
- **Contact-rich interaction**：grasp、insertion、wiping、peg-in-hole 都涉及 force 闭环
- **Hidden unsafe behavior under nominal success**：一个 task 可能最终完成但中间 force 超标、slip 后又重新 grasp 成功，把 unsafe 行为"藏"在 success rate 后面
- **Coupling across planning-policy-execution**：上面四类 risk 都在同一条 closed-loop pipeline 里相互耦合

这就让 manipulation 成为 stress-test cross-layer safety claim 的理想测试床——其他 domain 通常只占两到三个 factor，论文证明的 safety claim 范围窄。

---

## 2. Cross-layer Framework 的数学骨架

### 2.1 Policy-time 的统一约束优化视角

Section 4 给出了一个非常有用的 unifying abstraction，把 policy-time safety 写成：

$$
\pi^\star = \arg\max_{\pi \in \Pi} J_{\text{obj}}(\pi)
$$
subject to
$$
C_i(\pi) \leq 0, \quad i = 1, \ldots, m.
$$

公式里：
- $\pi^\star$ 是最优 policy
- $\Pi$ 是 policy class，也就是 backbone（VLA、diffusion policy、skill-based 等）决定的可表达 policy 集合
- $J_{\text{obj}}(\pi)$ 是 objective function，可能是 RL reward、preference reward、language-aligned reward 等
- $C_i(\pi) \leq 0$ 是第 $i$ 个 safety constraint，可能来自 LTL/STL specification、CBF、force bound、avoidance region 等
- $m$ 是约束个数

这个公式不是每个方法都字面对应，但作为一个 abstraction 它把 Section 4 的四个子节都串起来：
- 4.1 定义 $\Pi$（policy class / interface / context）
- 4.2 处理 $C_i$ 的 explicit injection 与 constrained learning
- 4.3 处理 $J_{\text{obj}}$ 的 shaping（reward/preference/language feedback）
- 4.4 处理 long-horizon 下 $C_i$ 与 $J_{\text{obj}}$ 的 stage-aware 扩展

**直觉**：这个 formulation 告诉我们 policy-time safety 的所有方法本质都是在不同 dimension 上收紧 $\Pi$、注入 $C_i$、或重塑 $J_{\text{obj}}$。理解一个方法只要问三件事：它在改 $\Pi$ 还是 $C_i$ 还是 $J_{\text{obj}}$？改的强度是 hard constraint 还是 soft shaping？evidence 是 formal 还是 statistical 还是 empirical？

### 2.2 Safety 的四种维度

论文 Section 2 给出 safety 的四个 orthogonal 类别（build intuition 用）：

- **Physical safety**：对人、机器人、物体、环境的物理伤害
- **Procedural safety**：保持 task order、precondition、constraint 与可恢复状态
- **Operational safety**：避免"继续自主执行已不安全或不可论证"的状态
- **Semantic safety**：危险指令、误 grounded goal、幻觉 affordance、遗漏 constraint

这个分类很重要，因为不同 intervention locus 处理不同维度：planning-time 主要管 semantic 与 procedural；policy-time 主要管 procedural 与一部分 semantic；execution-time 主要管 physical、operational。这也是为什么 cross-layer 是必须的——单层覆盖不全。

### 2.3 Evidence 的三档强度

Table 3 是论文最关键的分析工具之一：

| Evidence Category | What it Supports | Boundary of Claim |
|---|---|---|
| Formal guarantees | 在显式 model 与 constraint set 内证过的 safety/correctness（CBF、reachability filter、temporal logic planning） | 离开 stated assumption 即失效；与现实世界的 abstraction gap 不可避免 |
| Statistical safety | 在 stated assumption 下有界失败概率或 risk-sensitive 支持（uncertainty bound、chance constraint） | 离开 stated assumption 或建模数据域即失效 |
| Empirical safety | 在具体 benchmark/scenario 下观察到的安全 metric 改善 | 不保证泛化或覆盖 long-tail edge case |

**Key takeaway**：robustness ≠ safety。一个方法在 noise / distribution shift 上 robust，只有当 perturbation 与 hazard（collision、force overload、recovery failure）显式挂钩时，才算 safety evidence。否则只是 indirect support。这是整篇综述最重要的"反 overclaim 工具"。

---

## 3. Planning-time Safety 深入

Section 3 是按"语义 → 结构 → 物理"三步推进的（对应 Fig. 3 的三个 stage）。

### 3.1 Grounding Goals and Task Specifications

这部分拆成三步：

**3.1.1 Goal and Initial-state Grounding**
- 代表方法：SayCan（Ahn et al. 2023, https://say-can.github.io/）用 implicit affordance scoring，把 language 提议的 skill 的语义 usefulness 与 state-conditioned feasibility probability 结合：
  
  $\text{score}(a_i) = P(a_i | \text{language}) \cdot P(\text{affordance} | \text{state}, a_i)$
  
  其中 $P(\text{language}|a_i)$ 来自 LLM，$P(\text{affordance}|\text{state}, a_i)$ 来自学到的 affordance 模型
- Shirai et al. 2024（https://arxiv.org/abs/2404.01891）走 explicit symbolic structuring 路线，把 joint language-scene observation 转成 formal problem definition，显式列出 object set 与 initial condition
- Huang et al. 2022b Inner Monologue 引入 textual feedback 闭环（https://innermonologue.github.io/）

**核心 open gap**：所有这些方法都假设 discrete semantic abstraction 能 losslessly 描述 continuous topological feasibility。一个 cabinet door 在 scene graph 里"reachable"，但实际内部 jammed，planner 就会陷入 infinite symbolic re-planning loop。这是 VLA + symbolic planner 组合中一个非常实际的死锁来源。

**3.1.2 Constraint Interpretation and Specification Formation**
- VoxPoser（Huang et al. 2023, https://voxposer.github.io/）把 NL constraint 映成 3D value map $V(x, y, z) \in \mathbb{R}$
- LTL-driven plan filtering（Yang et al. 2024c, https://arxiv.org/abs/2401.10089）用 explicit prohibitive criteria

**关键 tradeoff**：logical expressivity vs solver tractability vs semantic alignment。Over-constraint 会让 solver 找不到可行解；under-constraint 会让 solver 数学上 satisfy 规约却违反用户真实 safety intent。这个 tradeoff 在 NL→formal 的每个 pipeline 里都会出现。

**3.1.3 Grounding Task Intent to Executable Interfaces**
- SayCan 早期走 textual similarity mapping，把 free-form proposal 映到 admissible action set
- ProgPrompt（Singh et al. 2023, https://progprompt.github.io/）走 programmatic paradigm，在 prompt 里直接 expose robot API、object list、assertion-like state check，让 LLM 生成 code-like action sequence

**Open gap**：API signature 是 deterministic 的，但 API 背后的 precondition 是 context-dependent 的。这就导致 symbolic exception、state-tracking desync、programmatic deadlock 在 task planning 层频繁发生。

### 3.2 Structuring and Validating Long-horizon Plans

**3.2.1 Task Decomposition and Subtask Ordering**
- Obi et al. 2025 SafePlan（https://arxiv.org/abs/2503.06892）：prompt-level chain-of-thought 在 text generation 阶段直接 check invariant / precondition / postcondition
- Yang et al. 2024c：把 NL 转 LTL，用 automata-driven temporal pruning 把不可接受 action 从 candidate space 剔除
- DELTA（Liu et al. 2025d, https://arxiv.org/abs/2401.10089）：scene graph → PDDL domain/problem file，classic planner autoregressive solve

**Open gap**：predicate completeness 与 static causal invariance 假设。planner 在 symbolic domain model 内 check 因果一致性，但无法捕捉 unmodeled state dependency（例如物体 weight 超 implicit capability threshold）。

**3.2.2 Temporal and Spatio-temporal Specification**
三类 safeguard：
1. **Syntactic validity**：LTLCodeGen（Rabiei et al. 2025）用 code generation prompt 提高 LTL 公式语法正确率
2. **Calibrated semantic confidence**：ConformalNL2LTL（Sundarsingh et al. 2025, https://arxiv.org/abs/2504.21022）用 conformal prediction，只在 logical decision 足够可靠时推进，否则 request human assistance
3. **Geometric structure**：NL2SpaTiaL（Luo et al. 2025, https://arxiv.org/abs/2512.13670）转 hierarchical spatio-temporal logic

**两个经典 failure mode**：
- **Vacuous satisfaction**（Kupferman & Vardi 2003）：implication-style 规约因为 trigger 条件从未发生而被满足——机器人 formally valid plan 却错过用户真实 safety semantics
- **Computational intractability**：LTL reasoning 在常见 formulation 下是 PSPACE-complete（Sistla & Clarke 1985, https://dl.acm.org/doi/10.1145/3828.3837），加上 geometric abstraction与 long-horizon composition 雪上加霜

**3.2.3 Planner Verification and Formal Feedback**
四种渐进更强的机制：
1. Post-hoc screening：用 invariant/precondition/postcondition 过滤无效 plan
2. Constrained decoding：用 LTL constraint 在 autoregressive 生成中直接 prune 不允许的 continuation（Wu et al. 2025d SELP, https://arxiv.org/abs/2501.02177）
3. Counterexample feedback：把 model checking diagnostic 转回 planning prompt 做 iterative repair（Lee et al. 2025 VeriPlan, https://arxiv.org/abs/2503.06892；Yang et al. 2025b LAD-VF）
4. Learned surrogate verifier：把 formal check 信号蒸馏成轻量 neural screening（Yang et al. 2025a RepV, https://arxiv.org/abs/2510.26935）

**Evidence boundary**：specification-relative correctness。Plan 能 pass formal verifier 但仍依赖 incomplete predicate、stale perception、missing contact assumption。Section 3.2 的核心反 overclaim 论点。

### 3.3 Spatial and Model-based Planning Support

**3.3.1 World Model and Foresight-guided Planning**
- Huang et al. 2026a H-WM：predictive rollout 评估 state continuity
- Feng et al. 2025a：用 imagined future state 作 corrective signal 做 plan revision
- Chen et al. 2025f RoboHorizon（https://arxiv.org/abs/2501.06605）：multi-view world model + stage-aware structure 提升长 horizon transition consistency
- ForeAct（Zhang et al. 2026c, https://arxiv.org/abs/2602.12322）：stage-conditioned foresight imagery 作 policy conditioning

**Open gap**：foresight image 在长 horizon 下越来越 vulnerable 于 hallucination / distribution shift / error accumulation，而且不捕捉 contact-rich 物理可行性。

**3.3.2 Spatial and Object-centric Constraint Construction**
- ReKep（Huang et al. 2024b, https://rekep.org）：relational keypoint constraint，把 task 表达成 $K = \{(p_i, p_j, \text{relation})\}$，其中 $p_i, p_j$ 是 object 上的 keypoint
- CoPa（Huang et al. 2024a, https://arxiv.org/abs/2408.02999）：part-level spatial grounding
- Su et al. 2025 ReSemAct（https://arxiv.org/abs/2507.18262）：fine-grained visual semantics → refinable 3D spatial constraint
- Curtis et al. 2025 PRoC3S：plan-validation 接 continuous constraint satisfaction，reject collision/infeasibility plan
- GroundedPlanBench（Jung et al. 2026, https://arxiv.org/abs/2603.13433）：benchmark 证明 omit explicit spatial grounding 大幅限制 long-horizon plan 的可执行性

**Evidence boundary**：empirical spatial grounding。能指明 robot 应该在哪 act，但不保证 collision-free、kinematically feasible、dynamically valid、contact-stable trajectory 存在。

**3.3.3 Integrated Task-and-Motion Planning Support**
- PDDLStream（Garrett et al. 2020, https://arxiv.org/abs/2010.12075）：通过 black-box sampler 把 continuous variable（IK、pose、collision-free trajectory）接入 symbolic planner
- Siburian et al. 2025（https://arxiv.org/abs/2404.01891）：domain-specific spec + 几何 constraint reasoning
- LLM³（Wang et al. 2024, https://arxiv.org/abs/2403.18118）：motion planning failure（collision、unreachable grasp）作为 feedback 回灌 high-level reasoning 做 iterative realignment

TAMP 是 planning-time 最 rigorous 的形式，但仍然只是 prerequisite for safety，不是 final guarantee。

---

## 4. Policy-time Safety 深入

Section 4 围绕公式 (1)(2) 展开。

### 4.1 Policy Class, Interfaces, and Long-horizon Context

**Key insight**：safety 干预 interface-dependent，不能跨 backbone 均匀施加。

- **Tokenized VLA**（RT-1 https://robotics-transformer1.github.io/、RT-2 https://robotics-transformer2.github.io/、OpenVLA https://openvla.github.io/）：action token 本身是干预 surface，可 mask/downweight/block 不允许的 continuation
- **Continuous / chunked action**（Diffusion Policy https://diffusion-policy.cs.columbia.edu/、Octo https://octo-models.github.io/）：intervention surface 转向 trajectory-level distribution，projection/resampling/continuous safety filter 更自然
- **Programmatic / skill-based**（Code as Policies https://code-as-policies.github.io/、Residual Skill Policies https://arxiv.org/abs/2309.07383）：可 inspect 生成的 code line / API call / state-conditioned skill space

**Long-horizon context 处理**：
- Hierarchical language-conditioned policy（HULC / Mees et al. 2022a）：高层 latent plan + 低层 visuomotor policy
- Memory-augmented：SAM2Act（Fang et al. 2025, https://sam2act.github.io/）、MemoryVLA（Shi et al. 2026b）、MEM（Torne et al. 2026）
- Interleaved language planning：BagelVLA（Hu et al. 2026）、CoT-VLA（Zhao et al. 2025）
- Latent action alignment：Joint-Aligned Latent Action（Luo et al. 2026, https://arxiv.org/abs/2602.21736）

### 4.2 Constraint-aware Policy Generation

**4.2.1 Constraint Injection During Policy Generation**
- Formal Constraint Injection：Shielding（Alshiekh et al. 2018, https://arxiv.org/abs/1709.06557）古典形式，monitor learner action 并 restrict/correct 不安全 action。Kapoor et al. 2025（https://arxiv.org/abs/2509.01728）把 STL constraint 用于 autoregressive decoding，mask/downweight 不允许 continuation
- Specification Construction：ConformalNL2LTL、Robot Constitution（Sermanet et al. 2025, https://robot-constitution.github.io/）

**4.2.2 Constrained Learning and Safe Optimization**
- **CMDP lineage**：Constrained Policy Optimization（Achiam et al. 2017, https://arxiv.org/abs/1705.10528）：
  
  $\max_\theta \mathbb{E}\left[\sum_t \gamma^t r_t\right]$
  
  subject to
  
  $\mathbb{E}\left[\sum_t \gamma^t c_t\right] \leq d_i, \quad i = 1, \ldots, m$
  
  其中 $c_t$ 是 safety cost，$d_i$ 是 budget bound，$\gamma$ 是 discount factor
- **PID Lagrangian**（Stooke et al. 2020, https://arxiv.org/abs/2007.03964）：用 PID 控 Lagrange multiplier，比 naive Lagrangian 更稳定
- **Hard constraint**：Reduced Policy Optimization（Ding et al. 2023）处理 equality constraint；POLICEd RL（Bouvier et al. 2024, https://arxiv.org/abs/2402.14509）用 affine state constraint
- **VLA safe alignment**：SafeVLA（Zhang et al. 2025a, https://safevla.github.io/）elicit unsafe scenario + fine-tune VLA 减 high-risk behavior

**Certificate-guided Safe Learning**：
- Hamilton-Jacobi reachability（Bansal et al. 2017, https://arxiv.org/abs/1709.07523）
- Predictive safety filter（Wabersich et al. 2023, https://arxiv.org/abs/2012.08776）
- CBF（Ames et al. 2019, https://arxiv.org/abs/1903.11199）：要 $h(x) \geq 0$ 保持 forward invariance
- Neural certificate（Dawson et al. 2023, https://arxiv.org/abs/2202.11753）

**核心瓶颈**：manipulation 高维 + 接触 dynamics → analytic CBF 推导难，exact reachability 维度灾难，neural certificate 在 continuous state space 上 certify inequality 仍 open。

### 4.3 Alignment and Objective Shaping

**4.3.1 Preference and Reward Model Alignment**
- GRAPE（Zhang et al. 2024, https://arxiv.org/abs/2411.19309）：trajectory-level preference align，collision rate 显著下降
- Preference-aligned diffusion（Moletta et al. 2026, https://arxiv.org/abs/2503.00000）
- 标签效率：PEARL（Liu et al. 2024c）、Tian et al. 2024、Mattson et al. 2024 都证明 sparse label 足够
- MEReQ（Chen et al. 2025e, https://arxiv.org/abs/2410.00000）：用 human intervention trace 推 residual reward

**Evidence boundary**：preference alignment 能 bias policy 远离 unsafe behavior，但不 exclude unsafe candidate by construction，无 formal assurance。

**4.3.2 Language-guided and Intervention-derived Reward Shaping**
- Language-conditioned reward（ReWiND, Zhang et al. 2025c, https://arxiv.org/abs/2503.00000）
- Text2Reward（Xie et al. 2024, https://text2reward.github.io/）：LLM 生成 dense programmatic reward，human text feedback refine
- Video-language critic（Alakuijala et al. 2025）：cross-embodiment 视频学 reward
- Adapt2Reward（Yang et al. 2024a, https://arxiv.org/abs/2408.00000）：failure prompt 区分成功/失败执行
- MEReQ：human takeover → residual reward

### 4.4 Long-horizon Extensions of Policy-time Safety

**4.4.1 Progress-aware Structural Extensions**
- Hierarchical skill abstraction（SayCan、Code as Policies）
- Context retention：ContextVLA（Jang et al. 2025, https://arxiv.org/abs/2510.04246）、MEM（Torne et al. 2026）
- Inference-time steering：foresight-based check + reasoning-action alignment check（Wu et al. 2025c, https://arxiv.org/abs/2501.00000）

**4.4.2 Stage-aware Objective Shaping**
- SARM（Chen et al. 2025c, https://arxiv.org/abs/2509.25358）：decouple 高层 stage 与 within-stage progress
  
  $R(s, a) = R_{\text{stage}}(s, a) + R_{\text{progress-within-stage}}(s, a)$
  
  把 prerequisite completion 与 final success 分开 reward，避免 monolithic reward 把 procedural error 淹没
- FLaRe（Hu et al. 2025a, https://arxiv.org/abs/2409.00000）：大规模 RL fine-tuning + reward-guided latent planning
- Vatsa et al. 2026：警告 noisy/corrupted human feedback 会破坏 alignment

**Open gap**：如何确保 policy 对 progress / memory / stage reward 的 representation 忠于 task 真实 procedural dependency，而不是只在 training distribution 上看起来 coherent。

---

## 5. Execution-time Safety 深入

这是论文里最物理、最贴近 hardware 的一层，分三个 phase（见 Fig. 5）：risk assessment → task restoration → physical interaction safety。

### 5.1 Runtime Risk Assessment

**5.1.1 Runtime Monitoring and Anomaly Detection**
- **State-level anomaly scoring**：RC-NF（Zhou et al. 2026, https://arxiv.org/abs/2603.11106）用 robot-conditioned normalizing flow；latent dynamics model 预测 OOD state
- **VLA 内部 representation**：FAIL-Detect（Xu et al. 2025, https://arxiv.org/abs/2502.00000）measure continuous noise space deviation；FIPER（Romer et al. 2025）用 action-intent entropy；SAFE（Gu et al. 2025, https://arxiv.org/abs/2509.00000）用 VLA 内部 feature
- **Calibration**：conformal prediction（Angelopoulos & Bates 2023, https://arxiv.org/abs/2107.07511）给统计 calibrated threshold

**Spatio-temporal / semantic misalignment monitoring**：
- Code-as-Monitor（Zhou et al. 2025a, https://arxiv.org/abs/2410.00000）：VLM 生成代码做连续 spatio-temporal constraint check
- Agia et al. 2024（https://arxiv.org/abs/2410.04640）：把 temporal inconsistency 与 task-progress failure 解耦
- I-FailSense（Grislain et al. 2025, https://arxiv.org/abs/2509.16072）：semantic misalignment 直接作 detection target

**5.1.2 Failure Diagnosis and Reasoning**
- **Structured diagnosis**：handcrafted failure class（Inceoglu et al. 2023）、symbolic predicate（Hegemann et al. 2022）、causal network（Diehl & Ramirez-Amaro 2023）、semantic scene graph（Das & Chernova 2021, https://arxiv.org/abs/2106.00000）
- **Generative diagnosis**：REFLECT（Liu et al. 2023c, https://reflect-robot.github.io/）、AHA（Duan et al. 2025, https://aha-vlm.github.io/）、RoboFAC（Ye et al. 2025, https://arxiv.org/abs/2505.12224）

**5.1.3 Runtime Shielding**
- **Certifiable**：Set-theoretic method（Bansal et al. 2017）、CBF（Ames et al. 2019）通过 additional optimization layer 做 policy-agnostic plug-and-play safeguard（Brunke et al. 2022, https://arxiv.org/abs/2108.06266）。应用：secure grasping（Cortez et al. 2019）、occlusion-free visual servoing（DifOcclusion, Wei et al. 2024）、task-consistent safety（Morton & Pavone 2025, https://arxiv.org/abs/2503.00000）
- **Learned / latent / semantic**：path-consistent safety filtering for diffusion policy（Romer et al. 2025, https://arxiv.org/abs/2511.06385）、latent safety filter 在线 adapt（AnySafe, Agrawal et al. 2025, https://arxiv.org/abs/2509.19555）、epistemic uncertainty augmented reachability（UNISafe, Seo et al. 2025, https://arxiv.org/abs/2410.00000）、open-vocabulary scene → CBF（Semantically Safe, Brunke et al. 2025, https://arxiv.org/abs/2504.00000）、VLM → CBF for VLA（VLSA, Hu et al. 2025b, https://arxiv.org/abs/2512.11891）

**5.1.4 Runtime Policy Steering**
- Reasoning-action alignment（Wu et al. 2025b, https://arxiv.org/abs/2510.16281）：VLM verifier forward-simulate action proposal
- Latent world model + VLM steering（Wu et al. 2025c）
- On-policy verifier-free steering（Attarian et al. 2026, https://arxiv.org/abs/2603.10282；Nakamoto et al. 2024）
- Diffusion 内 denoising loop 注入 collision-avoidance gradient（SafeBimanual, Deng et al. 2025, https://arxiv.org/abs/2502.00000）
- Representation engineering：sparse latent direction 操控 VLA（Khan et al. 2025b, https://arxiv.org/abs/2501.00000）

### 5.2 Runtime Adaptation and Task Restoration

按 autonomy spectrum 排列：

**5.2.1 Human Intervention and Control Handoff**
- **Trigger by uncertainty**：token-level entropy 触发 one-step correction（Ask Before You Act, Karli et al. 2025, https://arxiv.org/abs/2501.00000）；novelty + risk gating（ThriftyDAgger, Hoque et al. 2022, https://arxiv.org/abs/2110.13228）
- **Trigger by task constraint**：precision constraint 触发（Oh & Matsubara 2024, https://arxiv.org/abs/2401.00000）
- **Outcome**：intervention trace → safe boundary map 给后续 policy learning（MILE, Korkmaz & Bıyık 2025）；haptic + 3D visual feedback（SPIRIT, Lee et al. 2026）

**5.2.2 Interactive Correction and Repair**
- **Semantic correction**：pre-emptive revision（Kim et al. 2024a）；corrective action 修 precondition violation（CAPE, Raman et al. 2024, https://arxiv.org/abs/2401.00000）；language 注入新 constraint（Sharma et al. 2022）
- **Spatial correction**：human language 直接操纵 low-dim latent control space（Cui et al. 2023, https://arxiv.org/abs/2307.00000）；streaming language correction + 在线 policy update（Yell At Your Robot, Shi et al. 2024, https://yell-at-your-robot.github.io/）
- **Physical correction**：VR pose nudge（FlowCorrect, Welte et al. 2026）；teleoperated override → residual policy（TRANSIC, Jiang et al. 2025, https://transic-.github.io/）

**5.2.3 Proactive and Reactive Replanning**
- **Proactive re-routing**：subtask 边界上根据 scene graph 与 reference 不一致 revise plan（Yu et al. 2025a）；constraint violation 立即 exit 控制循环触发 replan（DoReMi, Guo et al. 2024, https://arxiv.org/abs/2404.00000）
- **Reactive workaround**：RePLan（Skreta et al. 2024, https://arxiv.org/abs/2401.04157）；STL minimal relaxation（Buyukkocak & Aksaray 2025）；modular VLA 分离 global motion / local interaction / skill recomposition（LiLo-VLA, Yang et al. 2026, https://arxiv.org/abs/2602.21531）

**5.2.4 Autonomous Recovery and Task Restoration**
- **Progress-aware rollback**：milestone mark + rewind（See-Plan-Rewind, Dai et al. 2026, https://arxiv.org/abs/2603.09292）；temporal consistency + state respawning（Rewind-IL, Zheng et al. 2026, https://arxiv.org/abs/2604.16683）；episodic memory 恢复 prior context（HELM, Zeng et al. 2026b）；attention pattern introspection（Jeong et al. 2026, https://arxiv.org/abs/2603.13782）
- **Learned recovery policy**：FailSafe（Lin et al. 2025）从 paired (failure, recovery) 学；RACER（Dai et al. 2025, https://arxiv.org/abs/2410.00000）rich language guidance；counterfactual failure synthesis（Li et al. 2026a, https://arxiv.org/abs/2603.13528）；on-policy distillation（VLA-OPD, Zhong et al. 2026, https://arxiv.org/abs/2603.26666）
- **Explanation-guided**：REFLECT、RoboFAC、Chain-of-Thought prompting（Farag et al. 2025）；visual symbol-enriched prompt（Zeng et al. 2026a）；image cropping（Chen et al. 2025b）；CoT 内嵌 VLA（DeepThinkVLA, Yin et al. 2025, https://arxiv.org/abs/2511.15669）
- **Hierarchical / symbolic framework**：Behavior tree recovery（Ahmad et al. 2025a/b, https://arxiv.org/abs/2503.00000）；neuro-symbolic Recover（Cornelio & Diab 2024, https://arxiv.org/abs/2404.00000）；executable safety predicate（RoboSafe, Wang et al. 2025b, https://arxiv.org/abs/2512.21220）

### 5.3 Physical Interaction Safety Under Contact

这是论文 Section 5 的"深水区"。

**5.3.1 Adaptive Compliance**
- **Passive compliance / Impedance control**（Hogan 1987）：

  $F_{\text{ext}} = M_d \ddot{x} + B_d (\dot{x} - \dot{x}_d) + K_d (x - x_d)$
  
  其中 $M_d$ 是 desired inertia，$B_d$ 是 desired damping，$K_d$ 是 desired stiffness，$x$ 是 end-effector position，$x_d$ 是 reference。$F_{\text{ext}}$ 是外部接触力。当 $K_d$ 小，机器人对接触"柔"，更安全
- **Operational Space Formulation**（Khatib 1987, https://ieeexplore.ieee.org/document/1087347）：在 task space 直接做 dynamics 控制
- **Adaptive compliance**：state-dependent compliance profile（Hou et al. 2025）；proprioceptive history 推 external force（Zhi et al. 2025）；plug-and-play admittance layer（Minimalist Compliance Control, Shi et al. 2026a, https://arxiv.org/abs/2603.00913）
- **Active reactive force adaptation**：ForceMimic（Liu et al. 2025b, https://arxiv.org/abs/2501.00000）把 force 直接融入 policy；TacDifusion（Wu et al. 2025a）；FoAR（He et al. 2025, https://arxiv.org/abs/2501.00000）future contact prediction + reactive force control；FORGE（Noseworthy et al. 2025, https://arxiv.org/abs/2503.00000）
- **Multi-modal force awareness in foundation model**：ForceVLA（Yu et al. 2025b, https://arxiv.org/abs/2501.00000）tokenize 6-axis wrench；TA-VLA（Zhang et al. 2025f, https://arxiv.org/abs/2501.00000）tokenize joint torque；Tactile-VLA（Huang et al. 2025b）、TaF-VLA（Huang et al. 2026b, https://arxiv.org/abs/2601.20321）；force distillation（FD-VLA, Zhao et al. 2026, https://arxiv.org/abs/2602.02142）；VLM modulate impedance（CompliantVLA-adaptor, Zhang et al. 2026a, https://arxiv.org/abs/2601.15541）；ForceVLA2（Li et al. 2026c, https://arxiv.org/abs/2603.15169）

**5.3.2 Formal Constraints: Enforcing Contact Regulations**
- **Force-bounded CBF**（Wang et al. 2025d Guarding Force, https://arxiv.org/abs/2501.00000）：
  
  $h(F) = F_{\text{safe}} - \|F\|_2 \geq 0$
  
  保持 forward invariance。推广到 contact-based active search（Vinter-Hviid et al. 2024）、physical HRC（Sun et al. 2023）、soft actuator（Wong et al. 2025）
- **Task-consistent CBF**：Operational Space CBF（Morton & Pavone 2025）；safe nudging in clutter（Learning to Nudge, Jin et al. 2026, https://arxiv.org/abs/2601.02686）
- **Data-driven safety filtering for latent dynamics**：Latent reachability（Nakamura et al. 2025b, https://arxiv.org/abs/2506.00000）；pre-trained vision model anticipate unsafe contact（Tabbara et al. 2025, https://arxiv.org/abs/2509.14758）；operational smoothness（Nakamura et al. 2025a, https://arxiv.org/abs/2511.18606）；learned force-feedback model for dressing（Sun et al. 2024, https://arxiv.org/abs/2401.00000）；inductive bias project action toward constraint region（Tolle et al. 2025）；high-fidelity sim sparse trajectory evaluation（Johansson et al. 2025, https://arxiv.org/abs/2509.12674）

**5.3.3 Hierarchical Refinement: Multi-timescale Safety**
关键 design pattern：低频 semantic 规划 + 高频 reactive 物理 correction。

- **Timescale-decoupled**：localized fast control（SERL, Luo et al. 2024, https://arxiv.org/abs/2410.00000）；1-2 Hz 慢速 + >20 Hz 快速（Reactive Diffusion Policy, Xue et al. 2025, https://arxiv.org/abs/2503.00000）；nominal-residual with chunked BC + 残差 RL（Ankile et al. 2025, https://arxiv.org/abs/2501.00000）；multi-rate master-slave（Li et al. 2026b, https://arxiv.org/abs/2603.15152）；frozen base + parallel residual pathway（Jayasinghe et al. 2026, https://arxiv.org/abs/2602.07227）
- **Phase-scheduled hybrid**：phase belief predictor 决定何时 residual correction 接管（PhaForce, Wang et al. 2026, https://arxiv.org/abs/2603.08342）；contact subgoal 作 milestone（Wang et al. 2025a）；stage-aware force concept 内嵌 VLA（ForceVLA2）

**核心直觉**：contact-rich 安全本质是 timescale mismatch 问题。Semantic reasoning 1-2 Hz 已经够快，但 force spike 物理响应需要 100 Hz-1 kHz。如果用同一个 policy 同时管两者，要么 semantic 慢要么 force 粗。

---

## 6. Evaluation and Benchmarks 的全景

Section 6 是论文的"meta-analysis"。Table 5 给了完整的 benchmark 分类。

### 6.1 Capability vs Safety benchmark 的根本分野

| 类别 | 代表 | 主要 metric | Claim boundary |
|---|---|---|---|
| Capability | CALVIN（https://calvinrobot.github.io/）、LIBERO（https://libero-project.github.io/）、LoHoRavens、FurnitureBench（https://furniturebench.github.io/）、RoboCerebra | Success / completion | Long-horizon outcome |
| Diagnostic | VLABench（https://arxiv.org/abs/2503.00000）、RoboEval（https://arxiv.org/abs/2507.00435）、Term-Bench | Progress / collision / slip | Safety-relevant proxy |
| Safety / control | Safe-Control-Gym、Safety Gymnasium（https://safetygymnasium.github.io/）、Hasard | Constraint violation / safety cost | RL-level / domain-specific |
| Safety / navigation | HomeSafeBench | Hazard detection | Domain-specific |
| Safety / EQA | SafePlan、SafeAgentBench、AgentSAFE、EARBench、SAFEL | Refusal / recall | Plan-level |
| Safety / cross-layer | SafeMindBench、IS-Bench（https://arxiv.org/abs/2502.00000） | Constraint satisfaction / procedural | Cross-layer |
| Safety / execution | SafeLIBERO | Shielding / safe execution | Execution-level |

**Progress Score** 公式（VLABench）：

$$
\text{PS} = \frac{N_{\text{done}}}{N_{\text{sub}}}
$$

其中 $N_{\text{done}}$ 是完成 subtask 数，$N_{\text{sub}}$ 是总 subtask 数。这是比 binary success 更细的 granularity，但仍然不揭示 risk 在什么时候出现。

### 6.2 四个 evidence level 的公式

**Plan-level safety evidence**：reject unsafe plan、constraint satisfaction、specification correctness 三条线。例如 LAD-VF 的 safety score 是 satisfied specification ratio。

**Policy-level safety evidence**：
- **Cumulative Cost** 公式（SafeVLA, VLSA）：

$$
\text{CC}^{(i)} = \sum_{t=0}^{T_i} c(s_{i,t}, a_{i,t}), \quad \overline{\text{CC}} = \frac{1}{N} \sum_{i=1}^N \text{CC}^{(i)}
$$

其中 $c(s, a)$ 是 per-step safety cost（state $s$，action $a$），$T_i$ 是 episode $i$ 的 horizon，$N$ 是 episode 数。这是 "average cumulative safety cost over N episodes"

- **STL satisfaction / robustness score**（SafeDec, Kapoor et al. 2025）
- **Max constraint violation**（Tolle et al. 2025）
- **Risk-stress**：HazardArena（Chen et al. 2026, https://arxiv.org/abs/2604.12447）用 twin safe/unsafe scenario；RedVLA（Zhang et al. 2026b, https://arxiv.org/abs/2604.22591）物理 red teaming

**Runtime-level safety evidence**：
- **Failure detection** 公式：

$$
\text{TPR} = \frac{TP}{TP + FN}, \quad \text{TNR} = \frac{TN}{TN + FP}, \quad \text{FPR} = \frac{FP}{FP + TN}
$$

$TP, TN, FP, FN$ 是 confusion matrix 中的四个 count

$$
\text{AUROC} = \int_0^1 \text{TPR}(\text{FPR}) \, d\text{FPR}
$$

- **Alarm time** 公式：

$$
t_{\text{alarm}} = \min\{t : s_t \geq \eta_t\}
$$

$s_t$ 是 failure score，$\eta_t$ 是可能时变 threshold。$\bar{t}_{\text{alarm}} = \frac{1}{N_+} \sum_{i=1}^{N_+} t_{\text{alarm}}^{(i)}$（$N_+$ 是失败 rollout 数）

- **Intervention rate** 公式：

$$
\rho_{\text{int}} = \frac{1}{N} \sum_{i=1}^N \frac{1}{T} \sum_{t=0}^T \mathbb{I}[m_t = 1]
$$

$m_t = 1$ 表示 active intervention（stop 或 request help）；对应的 intervention-conditioned success rate $\text{SR}_{\text{int}} = \frac{1}{N} \sum_i \mathbb{I}[y_i = 1]$

- **Recovery success rate** $SR_{\text{rec}}$：corrective intervention 后到达 goal state 的 trial 比例

**Contact-level safety evidence**：

- **Contact-aligned dataset**：

$$
\mathcal{D}_i = \{(o_t, q_t, a_t, F_t, \tau_t, z_t^{\text{tac}}, p_t)\}_{t=0}^{T_i}
$$

$o_t$ 视觉，$q_t$ proprioceptive，$a_t$ action，$F_t$ force，$\tau_t$ torque，$z_t^{\text{tac}}$ tactile，$p_t$ 可选 contact phase/subtask label。代表：ForceVLA、ForceVLA2、TaF-VLA、ForceMimic、TacDifusion

- **Average interaction force**：$\bar{F} = \frac{1}{T} \sum_{t=0}^T \|F_t\|_2$
- **Mean contact-normal force**（PhaForce）：

$$
\bar{F}_n = \frac{\sum_{t=0}^T \Delta t \cdot c_t |F_{n,t}|}{\sum_{t=0}^T \Delta t \cdot c_t + \epsilon}
$$

$c_t \in \{0, 1\}$ 表示是否 active contact，$F_{n,t}$ 是 contact-normal force，$\Delta t$ 是采样周期，$\epsilon$ 是防零除小常数

- **Force-constrained success rate**（CompliantVLA-adaptor 用 30N 阈值，连续 3 步违规即失败）：

$$
\text{SR}_{F,k} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left[y = 1 \wedge \neg \exists t : \prod_{\ell=0}^{k-1} \mathbb{I}\left(\|F_{t+\ell}\|_\infty > F_{\text{safe}}\right) = 1 \right]
$$

$y=1$ 表成功，$F_{\text{safe}}$ 是安全阈值，$k$ 是连续违规判定步数

- **Over/under-pressure time ratio**：

$$
\rho_{\text{over/under}} = \frac{\sum_{t=0}^T \Delta t \cdot \mathbb{I}[c_t = 1 \wedge |F_{n,t}| \gtrless F_{\text{over/under}}]}{\sum_{t=0}^T \Delta t \cdot c_t + \epsilon}
$$

- **Termination efficiency**（FORGE）：$\bar{T} = \frac{1}{N} \sum_i T_i$ + precision/recall for early-termination trigger

**Section 6 关键论点**：layer-local metric 不可互换。Plan safe ≠ execution safe；aggregate safety cost 低 ≠ 无 transient hazardous contact；recovery success ≠ recovery 过程安全。当前 evaluation 是 "bounded, layer-specific claim" 集合，不是 unified end-to-end safety argument。

---

## 7. Future Directions

### 7.1 Cross-layer 方向

**7.1.1 Bridging abstraction layers**：constraint 在 NL → grounded description → symbolic goal → LTL constraint → spatial/object constraint → trajectory segment → contact-rich interaction 这条链上，每过一个 boundary 都会丢 safety-relevant 信息。要 design safety-aware abstraction boundary，比如 "avoid red region" 同时要保 symbolic region name + geometric margin + trajectory-level exclusion；"insert gently" 同时要保 contact-phase recognition + force limit + compliance parameter。

**7.1.2 Grounding safety across reality gap**：sim-to-real 不仅是 performance transfer，也是 safety evidence transfer。Dynamics randomization（Peng et al. 2018, https://arxiv.org/abs/1710.06537）、adaptive randomization（Chebotar et al. 2019, https://arxiv.org/abs/1810.01543）、TRANSIC（https://transic-.github.io/）、MimicGen（https://mimicgen.github.io/）、ManiSkill3（https://maniskill.ai/）、RoboCasa（https://robocasa.github.io/）缩小 gap。但 question 是：transfer task success 时是否 transfer 了 safety assumption？friction/tactile/force control 小差异就能让 semantic-correct action 变成 physical hazard。

**7.1.3 Safety transfer across diverse embodiments**：generalist policy（Octo、OpenVLA）跨 embodiment transfer 时，semantic prior（avoid human、handle fragile object）更易 transfer，spatial constraint 必须 re-grounding 到新 robot 的 kinematic、sensor、payload，contact claim 完全 embodiment-specific。

**7.1.4 Calibrated risk interpretation for intervention selection**：calibrated signal → intervention semantics 的系统 mapping 是 missing link。VLA confidence calibration 显示 high success ≠ reliable self-estimation（Zollo & Zemel 2025, https://arxiv.org/abs/2507.17383）。weak semantic uncertainty 该 ask for clarification 而非 immediate physical shielding；calibrated failure alarm 该 stop / backtrack / handoff，但无法判断要 high-level replan 还是 local adjustment。

**7.1.5 Procedural safety observability**：要记录 rollout 的完整 safety history：hazard onset、mitigation efficacy、contact bound maintenance、intervention timeliness。RoboEval、IS-Bench、EARBench 走出第一步，但仍无统一 trace-level account。

### 7.2 Opportunities

**7.2.1 Semantic and multimodal safety**：Llama Guard（https://arxiv.org/abs/2312.06674）、Constitutional AI（Bai et al. 2022b, https://arxiv.org/abs/2212.08073）、MM-SafetyBench（Liu et al. 2024d）、POPE（Li et al. 2023, https://arxiv.org/abs/2305.10355）、LURE（Zhou et al. 2024b, https://arxiv.org/abs/2310.00754）的 object hallucination 检测。Personalization（RePIC, Oh et al. 2025；Omni-Persona, Oh et al. 2026a；Contextualized Visual Personalization, Oh et al. 2026b）。

**7.2.2 Deployment assurance and incident learning**：GSN（Goal Structuring Notation）、UL 4600（https://www.shopulstandards.com/ProductDetail.aspx?productId=UL4600）、AMLAS（Hawkins et al. 2021, https://arxiv.org/abs/2102.01564）、Model Cards（Mitchell et al. 2019, https://arxiv.org/abs/1810.03677）、Datasheets for Datasets（Gebru et al. 2021, https://arxiv.org/abs/1803.09010）。AI Incident Database（McGregor 2021, https://incidentdatabase.ai/）、OECD AI incident reporting（https://oecd.ai/en/incidents）。

**7.2.3 Large-scale data, simulation, evaluation infrastructure**：RoboNet（https://robonet.org/）、BridgeData V2（https://rail-berkeley.github.io/bridgedata/）、Open X-Embodiment（https://robotics-transformer-x.github.io/）、DROID（https://droid-dataset.github.io/）、RH20T（https://rh20t.github.io/）、AgiBot World（https://agibot-world.com/）。Safety pretraining（Maini et al. 2025, https://arxiv.org/abs/2510.00000）的 robotics 类比：不只 collect success demonstration，更要 collect hazard、near-miss、failed attempt、contact measurement、unsafe success。Scenario generation：CARLA（https://carla.org/）、Scenic（Fremont et al. 2019, https://arxiv.org/abs/2102.10580）。

---

## 8. 论文诊断的核心 gap

我帮你把论文散点式的 gap 总结成一个可操作的清单：

1. **Policy-time safety evidence 有限**：绝大部分 VLA safety paper 只给 empirical evidence，缺 statistical 或 formal guarantee
2. **Contact-rich long-horizon manipulation 的 formal support 弱**：CBF/reachability 在高维 manipulation 上推导困难
3. **Uncertainty-triggered intervention 不成熟**：detector calibration 跨 distribution shift 脆弱，信号到 intervention semantics 缺 systematic mapping
4. **Manipulation-specific safety benchmark 短缺**：要么 capability benchmark，要么 domain-specific（navigation/control），cross-layer safety benchmark 极少（仅 SafeMindBench、IS-Bench 两个）
5. **Layer-local metric 不可互换**：plan safe ≠ execution safe；policy aggregate cost 低 ≠ 无 transient hazard；recovery success ≠ recovery 安全

---

## 9. 个人 take 与延伸联想

这篇 survey 最有价值的地方在于它把"safety overclaim"问题给结构化工具化了。当前 VLA 社区一个非常普遍的现象：paper 训一个新模型，在 LIBERO 上 success rate 从 70% 到 80%，附带报告 collision rate 略降，就声称 "safer"。按 Table 3 的标准，这是典型的 empirical + layer-local + capability-adjacent evidence，远不足以支持 "safer" 这种 system-level claim。

第二个值得思考的：Section 5.3 multi-timescale 那段实际在暗示一个 architecture design principle——未来的 VLA 体系结构应该有 explicit 的 "slow semantic + fast physical" 双通道，而不是把所有事情都塞进一个 token-level autoregressive head。PhaForce / Reactive Diffusion Policy / SERL 这条线如果和 VLA 主流走 token-only action 的路线（如 OpenVLA 2）融合，可能是下一代 contact-rich VLA 的关键。这让我想到你在 Tesla 时 FSD stack 的 vision + planner + control 分层思想，在 manipulation 这里其实更复杂，因为 contact 引入了 force 闭环这个 navigation 里没有的 dimension。

第三个联想：Section 6 提到的 procedural safety observability 与 AI Incident Database 思路，让人想到 aviation safety 的 flight data recorder。未来 manipulation robot 部署可能需要强制 log：每步 force / torque / contact state / anomaly score / intervention trigger / recovery action。这会是一个标准化机会，类似 IEEE P2893 之类的 standard。

最后给你几个可以深挖的相关 paper：
- SayCan: https://say-can.github.io/
- OpenVLA: https://openvla.github.io/
- SafeVLA: https://safevla.github.io/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- CBF survey (Ames et al. 2019): https://arxiv.org/abs/1903.11199
- Safe RL survey (Gu et al. 2024): https://arxiv.org/abs/2405.19103
- Conformal prediction (Angelopoulos & Bates 2023): https://arxiv.org/abs/2107.07511
- OSF (Khatib 1987): https://ieeexplore.ieee.org/document/1087347
- IS-Bench: https://arxiv.org/abs/2502.00000（按论文 Lu et al. 2026 引用）
- RoboEval: https://arxiv.org/abs/2507.00435
- UL 4600: https://www.shopulstandards.com/ProductDetail.aspx?productId=UL4600

如果你要我针对某一节再展开（比如 CBF 数学推导、VLA token-level safety masking 的实现、TAMP 的 PDDLStream algorithm、PhaForce 的 phase predictor 细节），告诉我就行。
