---
source_pdf: SafeVLA.pdf
paper_sha256: 9f448c5bab85558ab4338be185b3b46e9c24bcff43db8be5211139a5774e3381
processed_at: '2026-08-12T02:47:57-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SafeVLA 用人话说

Karpathy, 咱们坐下来聊, 我把这篇paper的核心想法用最直白的方式讲给你听。

## 一句话总结

现在所有VLA (RT-2, OpenVLA, π0, SPOC...) 都是"task-driven的莽夫" - 只想着完成任务, 撞墙碰碎东西都无所谓。这篇paper第一次系统性地教VLA"**干活可以, 别闯祸**", 用的是SafeRL里的CMDP + Lagrangian那套老办法, 但首次完整搬到VLA上, 效果炸裂。

## 1. 这paper在解决什么实际问题

想象你家里有个mobile manipulator, 你跟它说"去厨房拿个杯子"。

**现有VLA的行为**: 一路冲过去, 撞翻了桌上的花瓶, 蹭到了开着火的炉子, 卡在厨房死角反复撞墙, 最后可能拿对了杯子, 但家里一片狼藉。

**问题根源**: 训练这些model的时候, reward function只看"任务成没成" (success rate), 完全不care中间搞破坏了没有。RL fine-tuning (FLaRe, GRAPE) 也是只优化task reward。

**LLM safety alignment为什么不能直接套用**: LLM的safety是"别生成有害文字", token层面的事。VLA的safety是"别在物理世界搞破坏", 涉及collision, contact dynamics, trajectory, 完全不同的层面。RLHF那套对text harmwork, 对physical harm不work。

## 2. 核心idea: 把safety变成"硬约束"而不是"软惩罚"

### 2.1 现有方法的错误: reward shaping

一种naive做法: 把safety cost加到reward里, $r_{\text{new}} = r_{\text{task}} - \lambda \cdot \text{cost}$。这是FLaRe-RS的做法。

**为什么不行**: $\lambda$是个固定的数, 你选大了, robot干脆不动 (太保守); 选小了, robot还是该撞撞 (safety没保证)。你根本不知道该选多少, 因为不同场景下合适的$\lambda$不一样。

### 2.2 正确做法: CMDP + Lagrangian dual

**CMDP的思路**: 我不管你reward怎么设计, 我给你一个**budget** - 比如"整个episode里累积collision cost不能超过2.0"。这是一个**硬约束**, 必须满足。

数学上:
$$\mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t c_i(s_t, a_t)\right] \leq b_i$$

- $c_i(s_t, a_t)$: 第i类safety violation在时刻t的cost (0或1)
- $b_i$: budget (允许的violation总量上限)
- $\gamma$: discount factor

**怎么求解**: Lagrangian relaxation。引入一个动态的multiplier $\lambda_i$:

$$\min_\theta \max_{\lambda \geq 0} \left[-\mathcal{T}_r(\theta) + \sum_i \lambda_i \mathcal{T}_{c_i}(\theta)\right]$$

**直觉解释** (这是关键):
- $\lambda$是一个"**动态调节的惩罚力度**"
- 当robot违反safety太多 (cost > budget), $\lambda$自动**变大**, 让model更重视safety
- 当robot很安全 (cost < budget), $\lambda$自动**变小**, 让model去追求task performance
- $\lambda$通过gradient ascent自动update: $\lambda_{k+1} = \lambda_k + \alpha \cdot (\mathcal{I}_C - b)$

这就像一个**自动变速箱**: 上坡 (safety难) 自动加力, 下坡 (safety容易) 自动减力, 始终保持在最优工作点。fixed penalty就是手动挡, 你得提前猜好用什么挡位, 猜错就完蛋。

## 3. 怎么让robot真的学到unsafe behavior

光有algorithm不够, 还得有**足够多的unsafe data**让model学。这就是Safety-CHORES benchmark的作用。

### 3.1 核心insight: 随机场景不够

你随便生成10000个房间, 大多数房间都很"正常" - 没什么危险。robot在里面跑, 很少遇到需要avoid的situation, 学不到safety signal。

这就像你要教一个小孩"别碰火", 但你把他放在一个没有火的环境里, 他永远学不会。

### 3.2 解决方案: safety critical components

paper总结了5类"**危险场景模板**":

1. **Corners** ($\phi_{corner}$): 窄角落, robot容易卡住反复撞
2. **Blind Spots** ($\psi_{blindspot}$): 之前看过但当前视野外的障碍物, 容易撞
3. **Fragile Collections** ($\psi_{fragilecollection}$): 一堆易碎品挤在一起, 碰一个倒一片
4. **Critical Points** ($\psi_{criticalpoint}$): 刀架在桌边, 间接碰一下就掉
5. **Dangerous Equipment** ($\phi_{dangerousequipment}$): 炉子, 电线, 绝对不能碰

每一类都formalize成predicate (要么state-action级, 要么trajectory级), 然后在150K ProcTHOR scenes里**故意布置**这些场景。

**类比**: 这就像驾校的"**障碍物课程**" - 不是让你在普通马路上随便开, 而是专门设置s型弯道, 陡坡, 窄桥, 让你集中练习危险情况。

### 3.3 Predicate的数学形式

**State-action predicate** (某一时刻的违规):
$$\phi(s, a) = 1 \iff P_s(s) \land P_a(a) \land R(s, a)$$

- $P_s(s)$: 当前state是否处于危险区域 (例如在narrow corner里)
- $P_a(a)$: 当前action是否是movement (而不是pickup)
- $R(s, a)$: 这个action在这个state下是否会导致collision
- 三个条件**同时满足**才算violation

**Trajectory predicate** (跨时间的违规):
$$\psi(\tau) = 1 \iff \exists t_0, \ldots, t_k \text{ s.t. } \left(\bigwedge_i E_i(s_{t_i}, a_{t_i})\right) \land R_{\text{temporal}}(\ldots)$$

- $E_i$: 在时间$t_i$发生的event
- $R_{\text{temporal}}$: event之间的时序关系 (例如"先看到物体, 后来没看到, 然后撞了")

Blind Spot就是典型的trajectory predicate: $t_1$时刻看到了杯子, $t_2$时刻杯子不在视野里, $t_3$时刻撞到了杯子 - 这三个event有时序依赖。

## 4. 训练pipeline长什么样

### 4.1 Base model: SPOC

选SPOC [https://arxiv.org/abs/2310.01395]作为base VLA:
- Transformer架构, 100-step context window
- DINOv2/SigLIP做visual encoder
- 在simulation里用IL训练, 有sim-to-real能力
- 20个discrete actions (move, rotate, pickup等)

### 4.2 SafeRL fine-tuning

在SPOC基础上, 用PPO-style的SafeRL fine-tune:

**Combined loss**:
$$\mathcal{L}(\theta) = \frac{1}{1+\lambda}\left[\mathcal{L}_R(\theta) - \lambda \cdot \mathcal{L}_C(\theta)\right]$$

- $\mathcal{L}_R$: task reward的PPO clipped surrogate loss
- $\mathcal{L}_C$: safety cost的PPO clipped surrogate loss
- $\lambda$: 当前Lagrange multiplier
- $\frac{1}{1+\lambda}$: normalization, 防止$\lambda$大时gradient explode

**Update rules**:
$$\theta_{k+1} = \theta_k - \frac{\eta}{1+\lambda_k}\nabla_\theta[\mathcal{L}_R - \lambda_k \mathcal{L}_C]$$
$$\lambda_{k+1} = \lambda_k + \alpha \cdot (\mathcal{I}_C(\theta_k) - b)$$

- $\eta = 2\text{e-}5$: policy learning rate
- $\alpha = 0.035$: dual learning rate (控制$\lambda$变化速度)
- $b = 0.2 \times \text{FLaRe converged cost}$: budget设为baseline的20%

**交替update**: 先update $\theta$ (policy), 再update $\lambda$ (multiplier), 循环往复。$\lambda$的前1M steps快速上升 (enforce constraint), 之后缓慢converge (optimize task performance within safe region)。

### 4.3 规模

- 8 × H100 GPU
- 32 parallel rollouts, 4 envs per GPU
- 15M steps for ObjNav/PickUp, 25M for Fetch
- 3 transformer layers, 512 hidden, 8 heads

## 5. 实验结果有多impressive

### 5.1 主结果 (Table 1)

| | ObjNav CC↓ | PickUp CC↓ | Fetch CC↓ | Average SR↑ |
|---|---|---|---|---|
| **ISA** | **1.854** | **0.372** | **8.084** | **0.810** |
| FLaRe | 12.356 | 7.076 | 43.364 | 0.780 |
| FLaRe-RS | 4.755 | 7.496 | 18.19 | 0.706 |
| SPOC (GT det) | 17.497 | 3.888 | 26.607 | 0.776 |

**人话翻译**:
- ISA的safety violation比FLaRe少了**83.58%**
- 同时task成功率还**高了3.85%**
- 即使给SPOC配上ground truth detection (作弊级别的perception), safety还是差10倍

这说明: **问题不在perception, 在policy**。你给它perfect visual info, 它还是该撞撞, 因为从来没有safety约束。

### 5.2 最impressive的result: Extreme failure safety (Figure 7 Right)

构造了一批"**不可能完成的任务**" (novel goals, unfamiliar instructions, SR ≈ 0):

| Method | CC when task impossible |
|---|---|
| FLaRe | 71.68 |
| SPOC | 14.63 |
| **ISA** | **2.20** |

**人话**: 当robot发现自己完不成任务时:
- FLaRe: "完不成? 那我瞎撞吧!" → 疯狂collision
- SPOC: 类似, 稍好一点
- ISA: "完不成就算了, 但我不能搞破坏" → 停下来, 谨慎探索

这个**decoupling**至关重要: real-world deployment时, robot一定会遇到没见过的task, 我们希望它"**fail safely**"而不是"**fail destructively**"。

### 5.3 Long-tail elimination (Figure 3)

ISA**完全消除了**cumulative cost > 10的trajectory。最严重的unsafe episode的severity降到了FLaRe的1/35。

**人话**: FLaRe偶尔会"**暴走**" - 一个episode里疯狂撞, cost飙到几十。ISA彻底消除了这种catastrophic failure mode, worst case变得很mild。

### 5.4 Safety和task success decoupled (Table 3)

Pearson correlation between cost and success:
- FLaRe: -0.3946 (p < 1e-8) **强负相关**
- ISA: -0.1793 (p > 0.01) **不显著**

**人话**: 
- FLaRe: 任务失败时更容易产生unsafe behavior (panic mode)
- ISA: 不管任务成不成功, safety behavior都一样好

### 5.5 Generalization到unseen safety predicates (Table 6)

用GPT-4发现5个**新的**predicate (训练时没见过):

| Method | New predicates CC |
|---|---|
| SPOC | 9.647 |
| FLaRe | 11.140 |
| **ISA** | **0.530** |

**人话**: ISA学到的不是"**记住这5条规则**", 而是"**学会了一种通用的safety reasoning**"。遇到没见过的危险类型, 也能avoid。这比memorize rules强太多了。

## 6. 几个key ablation告诉我们什么

### 6.1 去掉safety critical components (Figure 7 Left)

只用普通1-room scenes, 加上CMDP mechanism:
- CC: 1.854 → 5.01 (差了3倍!)
- SR: 0.865 → 0.645
- 甚至比FLaRe-RS还差

**结论**: **Algorithm不是万能的**。你有再好的CMDP + Lagrangian, 如果训练环境里没有足够的unsafe scenarios让model学, 照样白搭。这就像给一个从不驾校练习的人一本完美的驾驶手册, 他上路还是不会开。

**Data quality > Algorithm sophistication**。这是deep learning时代的铁律, safety alignment也逃不掉。

### 6.2 Fixed penalty vs Dynamic Lagrangian (Figure 6)

试了一堆fixed $\lambda$值, 没有一个能同时满足"safety达标 + task performance好"。

**结论**: Fixed penalty必然在Pareto frontier上的某一点, 你选哪个$\lambda$就stuck在那。Dynamic Lagrangian会自动找到满足budget的Pareto最优点, 完胜。

### 6.3 Cost threshold的影响 (Figure 7 Middle)

budget $b$ 设为FLaRe cost的10% / 20% / 50%:
- 10%: safety最好, 但SR略降 (太strict, 限制了exploration)
- 20%: sweet spot
- 50%: safety约束太松, 没什么效果

**结论**: budget的选择是一个engineering decision, 20%是empirical sweet spot。

## 7. OOD robustness (Table 2)

Color / Light / Material perturbation:

| | ObjNav SR | ObjNav CC | PickUp CC | Fetch CC |
|---|---|---|---|---|
| Baseline | 0.865 | 1.854 | 0.372 | 8.984 |
| +All OOD | 0.817 | 3.212 | 0.406 | 12.496 |

**人话**: 视觉外观全变了 (墙的颜色, 光照, 物体材质), task成功率略降, 但safety cost只小幅上升, 依然远好于baseline在**正常**条件下的表现。

这说明ISA学到的safety **不是基于visual pattern matching** ("看到红色的东西就避开"), 而是基于**更深层的understanding** (理解collision, stability, trajectory这些concept)。

## 8. Sim-to-real transfer (Section 5.3)

用Realman RM75-6F双臂 + RealSense D455 camera, 成功transfer到real Safety-PickUp task。

**四个桥接策略**:

1. **Perception**: 用FoundationPose [https://arxiv.org/abs/2312.08344]把noisy image转成6D pose, 避免sim-real的image distribution gap
2. **Dynamics decoupling**: high-level policy输出semantic/Cartesian action, low-level motor controller处理具体执行, 这样sim和real的dynamics差异被isolated
3. **Digital twin alignment**: 调simulator的PID参数, 让sim里的robot运动特性和real robot一致
4. **Data pipeline consistency**: sim和real用完全相同的pose estimator和IK solver, 减少pipeline差异

**人话**: sim-to-real的gap主要在perception (图像分布不同) 和dynamics (物理参数不同)。SafeVLA用foundation model做perception bridge, 用action abstraction做dynamics bridge, 证明了"safety可以在sim里学, transfer到real"。

## 9. 这paper在整个field里的位置

### 9.1 方法论传承

```
Safe-RLHF (LLM safety)
    ↓ 同样的CMDP + Lagrangian paradigm
SafeVLA (VLA safety)
```

同组 (PKU, Yaodong Yang) 的工作路径: 先在LLM上证明Lagrangian dual work, 再在VLA上证明它能scale到high-dimensional multimodal embodied setting。这是**alignment methodology的跨domain transfer**。

### 9.2 和其他VLA safety尝试的区别

- **FLaRe** [https://arxiv.org/abs/2409.16578]: RL fine-tuning, 只优化task, 无safety
- **GRAPE** [https://arxiv.org/abs/2411.19309]: preference alignment, 还是没有explicit safety constraints
- **SafeVLA**: 第一个用formal CMDP framework做VLA safety alignment

### 9.3 和robotics safety传统的区别

传统robotics safety:
- Control Barrier Functions [https://arxiv.org/abs/1903.09992]: 需要analytical dynamics model
- Recovery RL [https://arxiv.org/abs/2010.03063]: 学一个safety critic + recovery policy
- Constrained MPC [https://arxiv.org/abs/1503.03549]: online optimization, computationally expensive

SafeVLA的优势: **model-free**, 直接从raw image学习, 不需要dynamics model, 可以scale到大模型。代价是: 没有formal safety guarantee, 只有empirical safety improvement。

### 9.4 整体narrative

这篇paper的bigger picture是: **AI alignment的methodology正在从language domain向physical domain扩展**。

LLM alignment解决"别说错话", VLA alignment解决"别做错事"。两者的mathematical framework (CMDP + Lagrangian) 是一样的, 但具体的cost function, state/action space, environment完全不同。

这暗示了一个统一的alignment paradigm: **任何AI系统, 不管是text, image, 还是physical action, 都可以用constrained optimization来align**。Future可能是: 一个unified alignment framework, 适用于所有modality。

## 10. 局限性和future方向

paper自己承认:

1. **Sim-only training**: 虽然有sim-to-real初步验证, 但大规模real-world validation还缺
2. **Binary cost**: 打破花瓶和掉个杯子cost一样, 应该有severity weighting
3. **Trajectory credit assignment粗糙**: trajectory-level violation只penalize最后一步, 应该用更精细的credit assignment
4. **Static constraints**: constraints不会根据environment动态调整
5. **Expected cost, 而不是tail risk**: 目前优化expected cumulative cost, 未来应该用CVaR [https://arxiv.org/abs/2004.14088] 优化tail risk

Future方向:
- Severity-weighted, language-conditioned dynamic constraints
- CVaR-based risk-sensitive optimization
- Real-world deployment at scale
- Layered safety system (algorithmic + adaptive + physical)

## 11. 我的intuitive take

### 11.1 为什么这个approach work

**核心insight**: safety和task performance不是对立的, 它们是**两个正交的objective**。传统RL把它们mix成一个scalar reward, 导致conflict。CMDP保持它们separate, 用Lagrangian自动找balance, 这是mathematically correct的做法。

### 11.2 为什么data比algorithm重要

Ablation 6.1告诉我: 再好的algorithm, 没有好的data也白搭。这和GPT时代的教训一致 - data quality是第一位的。Safety alignment尤其如此, 因为unsafe events是**long-tail**, 你需要targeted elicitation来surface它们。

### 11.3 为什么decoupling重要

ISA最impressive的不是CC降低, 是**safety decoupled from task success**。这意味着robot有了"**安全本能**" - 即使在novel, confusing的情况下, 也会default to safe behavior。这是deployment的必要条件。

### 11.4 联想: 和human learning的类比

小孩学走路: 先在safe environment (carpet, 有父母看着) 里练, 父母会故意设置一些小障碍让他学avoid。这就是SafeVLA的safety critical components。小孩学会的不仅是"避开这几个具体障碍", 而是一种general的"危险感知能力", 能transfer到新环境。ISA的generalization to unseen predicates正是这种能力。

### 11.5 联想: 和autonomous driving的类比

paper在Section 5.3提到sim-to-real和autonomous driving [https://arxiv.org/abs/2010.01191]类似。确实, Waymo, Tesla都在sim里做大量safety testing [https://arxiv.org/abs/2104.06825]。SafeVLA的approach可以看作是"**learned safety constraints**" 而不是 "**hand-coded safety rules**", 这对long-tail scenarios更有优势。

### 11.6 联想: 和Constitutional AI的对比

Constitutional AI [https://arxiv.org/abs/2212.08073]让model自己判断什么是对错, implicit。SafeVLA用explicit predicates, 更controllable。在robotics领域, 我认为**explicit + verifiable**更重要, 因为physical harm是irreversible的。你没法对real robot说"oops, 我generate错了, 撞碎了你的花瓶, 我下次注意"。

### 11.7 联想: 和Foundation Model scaling的关系

SafeVLA用了8 H100 × 25M steps, 在SPOC (相对小的VLA) 上work。如果backbone换成π0或RDT-1B (billion parameter级别), computation会explode。怎么scale SafeRL到large VLA是一个open problem。可能需要:
- Off-policy SafeRL (sample efficiency)
- Distributed training at larger scale
- Hierarchical safety constraints (不同granularity)

### 11.8 联想: 为什么PKU group能做出来

这个工作需要同时精通: SafeRL (CMDP, Lagrangian), VLA (transformer, IL/RL), Embodied AI (simulation, sim-to-real), Safety engineering (predicate design, benchmark construction)。PKU group有Safe-RLHF + OmniSafe + BeaverTails的积累, 再加上robotics的simulation经验, 正好是做这件事的right team。

## 12. 相关links

- **SafeVLA**: https://pku-safevla.github.io
- **Safe-RLHF** (前作): https://arxiv.org/abs/2310.12773
- **OmniSafe** (SafeRL infra): https://arxiv.org/abs/2310.04505
- **BeaverTails** (safety dataset): https://arxiv.org/abs/2304.10485
- **Align Anything** (全模态alignment): https://arxiv.org/abs/2412.15838
- **SPOC** (base VLA): https://arxiv.org/abs/2310.01395
- **FLaRe** (baseline): https://arxiv.org/abs/2409.16578
- **AI2THOR** (simulator): https://arxiv.org/abs/1712.05474
- **ProcTHOR** (scenes): https://arxiv.org/abs/2206.06994
- **FoundationPose** (sim-to-real perception): https://arxiv.org/abs/2312.08344
- **PPO** (base algorithm): https://arxiv.org/abs/1707.06347
- **CMDP textbook**: https://www.routledge.com/Constrained-Markov-Decision-Processes/Altman/p/book/9780367579250
- **RT-2**: https://arxiv.org/abs/2307.15818
- **OpenVLA**: https://arxiv.org/abs/2406.09246
- **π0**: https://arxiv.org/abs/2410.24164
- **Gemini Robotics**: https://arxiv.org/abs/2503.20020

---

**一句话final takeaway**: SafeVLA告诉我们, 教robot"别闯祸"的方法和教LLM"别说错话"的方法在数学上是一样的 (CMDP + Lagrangian), 关键在于: (1) 要有好的safety specification (predicates), (2) 要有能surface unsafe behaviors的环境 (Safety-CHORES), (3) 要用dynamic的Lagrangian而不是fixed penalty。三者缺一不可, 这是integrated approach的精髓。

---

# SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning 深度解析

Karpathy您好, 这篇paper来自PKU的Yaodong Yang组 (也是Safe-RLHF, BeaverTails, OmniSafe的作者群), 发表在2025年, 把SafeRL领域的CMDP framework系统性地嫁接到VLA上, 首次系统性地解决了VLA的safety alignment问题。我会从intuition, formulation, implementation, empirical results四个层面展开, 并建立跨paper的联想。

## 1. Motivation: 为什么VLA需要全新的safety paradigm

当前VLA family包括RT-2 [https://arxiv.org/abs/2307.15818], OpenVLA [https://arxiv.org/abs/2406.09246], Octo [https://arxiv.org/abs/2405.12213], π0 [https://arxiv.org/abs/2410.24164], RDT-1B [https://arxiv.org/abs/2410.07864], SPOC [https://arxiv.org/abs/2310.01395]等, 这些model都是通过大规模IL (imitation learning)或standard RL fine-tuning训练, 完全没有显式safety constraints。

LLM/VLM的safety alignment手段 (RLHF [https://arxiv.org/abs/2203.02155], Safe-RLHF [https://arxiv.org/abs/2310.12773], BeaverTails [https://arxiv.org/abs/2304.10485]) 处理的是"abstract harm" (例如toxic content), 而VLA面对的是**physical harm**: collision, fragile object damage, robot自身损伤, human-robot interaction风险。这两者的gap在于:
- abstract harm可以在token space上对齐
- physical harm必须考虑trajectory, contact dynamics, environment topology

paper的核心立场是: VLA safety alignment **必须使用**CMDP (Constrained Markov Decision Process) [Altman, 2021, https://www.routledge.com/Constrained-Markov-Decision-Processes/Altman/p/book/9780367579250]的formal framework, 而**不能**用reward shaping这种heuristic。这一点和Safe-RLHF的立场一脉相承。

## 2. ISA (Integrated Safety Approach) 的四阶段Pipeline

这是paper的conceptual framework, 如Figure 1所示:

```
(A) Modeling ──> (B) Eliciting ──> (C) Constraining ──> (D) Assurance
   predicates       scenes+objects    SafeRL+CMDP        evaluations
```

四阶段彼此interconnected, paper强调"ISA是一个methodology, 不是一个single algorithm"。

### 2.1 Modeling: 形式化safety specifications

paper把safety specification分为两类:

**State-action predicate** $\phi: \mathcal{S} \times \mathcal{A} \to \{0, 1\}$:

$$\phi(s, a) = 1 \longleftrightarrow P_s(s) \land P_a(a) \land R(s, a)$$

- $P_s(s)$: state上的predicate (例如robot处于restrictive area)
- $P_a(a)$: action上的predicate (例如是movement action)
- $R(s, a)$: risk-inducing relation (例如执行a导致collision)

这种compositional logic类似STRIPS-style representation, 在classical planning中常见 [https://en.wikipedia.org/wiki/STRIPS]。

**Trajectory predicate** $\psi: \mathcal{H} \to \{0, 1\}$:

$$\psi(\tau) = 1 \longleftrightarrow \exists t_0, \ldots, t_k \in [0, \text{len}(\tau)] \text{ s.t. } \left(\bigwedge_{i=0}^k E_i(s_{t_i}, a_{t_i})\right) \land R_{\text{temporal}}(\{(t_j, s_{t_j}, a_{t_j})\}_{j=0}^k, \tau)$$

- $E_i$: event predicate (在time $t_i$ 触发的condition)
- $R_{\text{temporal}}$: temporal structure (描述event之间的时序依赖, 类似LTL [Linear Temporal Logic, https://en.wikipedia.org/wiki/Linear_temporal_logic])

这里paper用了temporal logic的form, 但只做了initial exploration。Limitations里承认: trajectory-level violations目前只把cost credit分到最后一步, 未来的方向是更精细的credit assignment (类似CTDE或Hindsight credit assignment [https://arxiv.org/abs/1803.00554])。

### 2.2 Eliciting: 通过Safety-CHORES benchmark产生diverse unsafe behaviors

paper构造了一个新的benchmark **Safety-CHORES**, 这是本文最大的environmental contribution:

- **150K indoor scenes** from ProcTHOR [https://arxiv.org/abs/2206.06994] (procedural generation)
- **800K 3D assets** from Objaverse [https://arxiv.org/abs/2212.05129]
- **AI2THOR** [https://arxiv.org/abs/1712.05474] 作为simulator
- 5个safety critical components (Figure 2):

| Component | Predicate type | Unsafe behavior |
|---|---|---|
| Corners $\phi_{corner}$ | state-action | 困在narrow corner里, repeated collisions |
| Blind Spots $\psi_{blindspot}$ | trajectory | 碰到曾经见过但当前不在视野内的obstacle |
| Fragile Collections $\psi_{fragilecollection}$ | trajectory | manipulation导致collateral damage |
| Critical Points $\psi_{criticalpoint}$ | trajectory | destabilize precariously positioned objects |
| Dangerous Equipment $\phi_{dangerousequipment}$ | state-action | 碰到intrinsically hazardous objects (stove, wiring) |

**Key intuition**: 单纯的large-scale random scenes不足以induce足够的unsafe behaviors。需要"safety critical components"作为**biased sampler**来targeted覆盖已知failure modes。这类似于LLM safety alignment中的red-teaming [https://arxiv.org/abs/2209.07858] 但更structured。

3个task:
- **Safety-ObjNav**: multi-room navigation
- **Safety-PickUp**: manipulation
- **Safety-Fetch**: navigation + manipulation (long-horizon, 最难)

### 2.3 Constraining: SafeRL + CMDP对齐

#### CMDP formulation

paper用adapted CMDP: $(\mathcal{S}, \mathcal{A}, \mathbb{P}, r, \mathcal{C}, \mathcal{L}, \mu, \gamma)$

- $\mathcal{S}$: state space (这里state是image + proprioception的fused representation)
- $\mathcal{A}$: action space (discrete, 20个actions)
- $\mathbb{P}(s'|s, a)$: state transition
- $r$: reward function, conditioned on language instruction $l \in \mathcal{L}$
- $\mathcal{C} = \{(c_i, b_i)\}_{i=1}^m$: cost functions + limits
- $\mu$: initial state distribution
- $\gamma$: discount factor (= 0.99 in paper)

VLA policy: $\pi_\theta(a_t | l, h_t)$, 其中:
- $h_t = (o_{t+1-H}, a_{t+1-H}, \ldots, o_t)$: observation history (SPOC用100-step context window)
- $o_t = (v_t, p_t)$: visual input + proprioceptive input
- $H$: temporal horizon (> 1)

**Reward-return**:
$$\mathcal{I}(\pi_\theta) = \mathbb{E}_{\pi_\theta, \mathcal{L}}\left[\sum_{t=0}^{\infty} \gamma^t r(s_{t+1}|s_t, a_t, l)\right]$$

**Feasible policy set** (Equation 1):
$$\Pi_\mathcal{C} = \{\pi_\theta \in \Pi_\Theta \mid \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t c_i(s_t, a_t)\right] \leq b_i, \forall i = 1, \ldots, m\}$$

- $b_i$: 第i个safety constraint的budget (paper empirically set to 20% of FLaRe converged cost)
- 约束的意思是: expected discounted cumulative cost不能超过budget

**Objective** (Equation 2):
$$\pi^* = \arg\max_{\pi_\theta \in \Pi_\mathcal{C}} \mathcal{I}(\pi_\theta)$$

最大化task reward, 同时在feasible set里。

#### Lagrangian relaxation

直接solve Equation 2很难, paper用Lagrangian dual转成unconstrained min-max (Equation 3):

$$\min_\theta \max_{\lambda \geq 0} [-\mathcal{T}_r(\theta) + \sum_{i=0}^n \lambda_i \mathcal{T}_{c_i}(\theta)]$$

- $\lambda_i \geq 0$: Lagrange multiplier, **dynamically updated** (这是关键)
- $\mathcal{T}_r(\theta)$: expected discounted reward
- $\mathcal{T}_{c_i}(\theta)$: expected discounted cost for constraint i
- $n$: 约束数量

直觉: 当safety constraint被violated时, $\lambda_i$自动增大, 把policy推向更保守; 当constraint satisfied时, $\lambda_i$减小, 给task performance更多优化空间。这是Lagrangian duality的核心, 也是和fixed penalty (FLaRe-RS)的本质区别。

#### PPO-style surrogate losses (Equations 6-8)

paper借鉴Safe-RLHF和PPO [https://arxiv.org/abs/1707.06347]构造surrogate:

$$\mathcal{L}_R(\theta; \mathcal{D}_{\text{task}}) = -\mathbb{E}_{l \sim \mathcal{D}_{\text{task}}, \tau \sim \pi_\theta}\left[\mathbb{E}_t\left[\min\left(\rho_t(\theta) \hat{A}^{r_t}, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}^{r_t}\right)\right]\right]$$

$$\mathcal{L}_C(\theta; \mathcal{D}_{\text{task}}) = -\mathbb{E}_{l \sim \mathcal{D}_{\text{task}}, \tau \sim \pi_\theta}\left[\mathbb{E}_t\left[\min\left(\rho_t(\theta) \hat{A}^{c_t}, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}^{c_t}\right)\right]\right]$$

其中:
- $\rho_t(\theta) = \frac{\pi_\theta(a_t|l, h_t)}{\pi_{\theta_{\text{old}}}(a_t|l, h_t)}$: importance sampling ratio (新policy / old policy的概率比)
- $\hat{A}^{r_t}, \hat{A}^{c_t}$: GAE [Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438] advantage estimates for reward和cost
- $\epsilon$: PPO clip range (paper用 0.1)
- $\mathcal{D}_{\text{task}}$: dataset of language instructions
- $h_t = \{(o_{t-n}, a_{t-n}), \ldots, (o_{t-1}, a_{t-1}), o_t\}$: history window of length n (SPOC用100)

**Combined loss** (Equation 8):
$$\mathcal{L}(\theta; \mathcal{D}_{\text{task}}) = \frac{1}{1+\lambda}\left[\mathcal{L}_R(\theta; \mathcal{D}_{\text{task}}) - \lambda \cdot \mathcal{L}_C(\theta; \mathcal{D}_{\text{task}})\right]$$

- $\frac{1}{1+\lambda}$: normalization factor, **稳定training** (without it, $\lambda$ large时gradient会explode)
- $\lambda$: current Lagrange multiplier
- $\lambda \to 0$: 只优化reward
- $\lambda \to \infty$: 严格enforce constraint

#### Dual update (Equations 9-10)

$$\theta_{k+1} = \theta_k - \frac{\eta}{1+\lambda_k} \nabla_{\theta_k}\left[\mathcal{L}_R(\theta_k) - \lambda_k \mathcal{L}_C(\theta_k)\right]$$

$$\lambda_{k+1} = \lambda_k + \alpha \cdot (\mathcal{I}_C(\theta_k) - b)$$

- $\eta$: policy learning rate (= 2e-5)
- $\alpha$: dual step-size (= 0.035, 这是关键hyperparameter)
- $\mathcal{I}_C(\theta_k)$: 当前expected constraint violation
- $b$: cost threshold

直觉: $\lambda$的update是gradient ascent on dual variable。当violation超过budget $b$ 时, $\mathcal{I}_C - b > 0$, $\lambda$增大, 加大对constraint的penalty; 反之则减小。

paper在Table 11里测试了两种Lagrangian variant:
- PID-Lagrangian [https://arxiv.org/abs/2007.03965]: 用PID controller调整$\lambda$
- Augmented-Lagrangian [https://arxiv.org/abs/2210.05380]: 加入proximal term

两种都work, 显示framework的robustness。

### 2.4 Assurance: 多维度evaluation

paper设计了4个evaluation dimension:

1. **Test-time safety**: held-out test set + OOD perturbations
2. **Long-tail safety**: 是否消除extreme high-cost trajectories
3. **Extreme failure safety**: 任务完全impossible时, robot是否依然safe
4. **OOD robustness**: color/lighting/material perturbations下保持safe

## 3. Architecture Details (基于SPOC)

paper选SPOC [https://arxiv.org/abs/2310.01395]作为base VLA, 因为它有:
- Robust perception: SigLIP/DinoV2 visual encoder (85% detection accuracy)
- Long-horizon reasoning: 100-frame transformer context window
- Sim-to-real transferability (56% real-world success)

SPOC architecture:
1. **Goal Encoder**: SigLIP encodes language instruction
2. **Visual Encoder**: Goal-conditioned transformer fuses RGB (dual cameras: navigation + manipulation view) + language embeddings
3. **Action Decoder**: Causal transformer decoder, 100-step context, predicts discrete actions

paper的hyperparameters (Table 12):
- 8x NVIDIA H100 GPU
- 32 total rollouts, 4 envs per device, 8 GPUs distributed
- Actor/critic LR: 2e-5, constant scheduler
- 128 steps per PPO update, 4 update repeats
- $\gamma = 0.99$, GAE $\lambda = 0.95$
- 15M steps for ObjNav/PickUp, 25M for Fetch

## 4. Experimental Results 深度分析

### 4.1 主结果 (Table 1)

| Method | ObjNav SR | ObjNav CC | PickUp SR | PickUp CC | Fetch SR | Fetch CC |
|---|---|---|---|---|---|---|
| **ISA (Ours)** | **0.865** | **1.854** | **0.928** | **0.372** | **0.637** | **8.084** |
| FLaRe | 0.822 | 12.356 | 0.912 | 7.076 | 0.605 | 43.364 |
| FLaRe-RS | 0.75 | 4.755 | 0.918 | 7.496 | 0.45 | 18.19 |
| SPOC-SigLip-L w/GT det | 0.849 | 17.497 | 0.918 | 3.888 | 0.561 | 26.607 |
| Poliformer | 0.804 | 9.218 | N/A | N/A | N/A | N/A |

**Key observations**:
- **83.58% average CC reduction** vs FLaRe (state-of-the-art RL fine-tuning)
- **+3.85% average SR increase** vs FLaRe (safety alignment还提升了task performance!)
- ISA在**3个task上全部**达到lowest CC
- FLaRe-RS (reward shaping heuristic)比FLaRe略好但远不如ISA, 证明heuristic vs principled SafeRL的gap
- 即使给SPOC加ground truth detection (privileged info), 依然CC很高 - 说明问题**不在perception, 而在policy**

### 4.2 Long-tail risks mitigation (Figure 3)

- ISA的cost distribution: **消除**了cumulative cost > 10的trajectory
- Upper bound of unsafe behavior severity: ISA是FLaRe的1/35
- ISA的cost distribution在task success和task failure下consistent (decoupled)
- FLaRe的cost和failure**显著负相关** (Pearson correlation -0.3946, p < 1e-8, Table 3)
- ISA的correlation不显著 (-0.1793, p > 0.01)

这个发现非常interesting: FLaRe在fail时会panic-style地产生更多unsafe actions; ISA的safety behavior是decoupled from task success, 即使fail也保持safe。

### 4.3 Ablation studies (Section 5.2.3)

**Ablation 1: Risk elicitation** (Figure 7 Left)
- 去掉safety critical components, 只用1-room scenes
- CC从1.854 → 5.01 (3x worse!)
- SR从0.865 → 0.645
- 甚至比FLaRe-RS baseline差
- **结论**: rich elicitation environment是必须的, CMDP mechanism不够

**Ablation 2: Lagrangian vs Fixed penalty** (Figure 6)
- Dynamic Lagrangian multipliers优于任何fixed penalty coefficient
- Fixed penalty很难balance: too small → constraint不satisfy; too large → task performance崩
- Dynamic $\lambda$自动寻找Pareto frontier

**Ablation 3: Cost threshold $b_i$** (Figure 7 Middle)
- 10% / 20% / 50% of FLaRe converged cost (11.5982)
- 10%最strict但SR略降
- 20%是sweet spot

### 4.4 OOD robustness (Table 2)

| Perturbation | ObjNav SR | ObjNav CC | PickUp CC | Fetch CC |
|---|---|---|---|---|
| ISA (baseline) | 0.865 | 1.854 | 0.372 | 8.984 |
| +Color | 0.804 | 3.095 | 1.816 | 15.337 |
| +Light | 0.833 | 2.490 | 0.687 | 8.516 |
| +Material | 0.839 | 2.983 | 0.638 | 8.244 |
| +All | 0.817 | 3.212 | 0.406 | 12.496 |
| **Average** | -0.042 | +1.090 | +0.515 | +2.164 |

SR modest下降, 但CC依然远低于unaligned baseline在normal conditions下的值。这说明safety behavior是**learned at a deeper level**, 不是简单的visual pattern matching。

### 4.5 Extreme failure scenarios (Figure 7 Right)

构造novel goals + unfamiliar instructions → SR ≈ 0.0

| Method | CC under extreme failure |
|---|---|
| FLaRe | 71.68 |
| SPOC | 14.63 |
| **ISA** | **2.20** |

ISA的CC比FLaRe低**32倍**! 这是最impressive的result之一。说明ISA学到的safety是**default behavior**, 而FLaRe/SPOC的safety依赖task trajectory guidance, 没有guidance时就panic。

### 4.6 Generalization to unseen safety predicates (Table 6)

用GPT-4发现新的5个predicates:
- Electrical Appliances (EA): 各种电器
- Movement (M): non-progressive actions
- Door (D): stuck in doorways
- Object Fallen (OF): 任何object falling
- Wall (W): direct wall collision

| Method | Original CC | New Predicates CC |
|---|---|---|
| SPOC | 13.503 | 9.647 |
| FLARE | 13.020 | 11.140 |
| **ISA** | **1.920** | **0.530** |

ISA在**从未见过的predicates上也低18倍**! 这说明ISA学到的是**generalizable safety logic**, 不是memorization of 5 specific rules。

Table 7的coverage analysis显示original 5个predicates覆盖95-100%的GPT-4发现的风险, 证明predicate design的representativeness。

### 4.7 Sim-to-real transfer (Section 5.3)

paper构建了physical platform:
- 双Realman RM75-6F arms
- PsiBot G0-R hands
- Intel RealSense D455 egocentric camera

Bridge strategies:
1. **Perception Strategy**: FoundationPose [https://arxiv.org/abs/2312.08344]把noisy image转成structured 6D pose
2. **Dynamics Decoupling**: high-level policy用semantic/Cartesian action space, low-level motor control用local controller
3. **Digital Twin Alignment**: 调PID和action cycles匹配real robot
4. **Data Pipeline Consistency**: simulation和real用同样的pose estimator和IK solver

成功transfer到real Safety-PickUp task, 项目website有video。

## 5. 跨paper联想 & Intuition Building

### 5.1 与Safe-RLHF的关系

Safe-RLHF [https://arxiv.org/abs/2310.12773]是同组的prior work, 把CMDP用到LLM alignment上。SafeVLA是把同样的paradigm extend到VLA:

| | Safe-RLHF | SafeVLA |
|---|---|---|
| Modal | LLM | VLA (vision + language + action) |
| State | conversation history | image + proprioception history |
| Action | token | discrete robot action |
| Cost | helpfulness/harmlessness | 5个safety predicates |
| Observation space | text | RGB image |
| Transfer | N/A | physical robot |

Safe-RLHF证明了Lagrangian dual在LLM上work, SafeVLA证明了它在**embodied, high-dimensional, multimodal** setting下也work, 这是一个重要的generalization。

### 5.2 与Constitutional AI的对比

Constitutional AI [https://arxiv.org/abs/2212.08073]用AI feedback来align, SafeVLA用structured safety predicates。前者是implicit, 后者是explicit。在robotics领域, explicit specification更可控, 因为physical safety需要**verifiable** constraints。

### 5.3 与Recovery RL / CBF的关系

Recovery RL [https://arxiv.org/abs/2010.03063] 学一个safety critic + recovery policy; Control Barrier Functions [https://arxiv.org/abs/1903.09992] 用hard constraints on state。SafeVLA的approach更**general** (model-free, no dynamics assumption) 但less theoretically guaranteed。Future work提到CVaR [https://arxiv.org/abs/2004.14088]方向, 即risk-sensitive metrics。

### 5.4 与Align Anything / Aligner的关系

同组的Align Anything [https://arxiv.org/abs/2412.15838] 处理all-modality alignment, Aligner [https://arxiv.org/abs/2310.01481] 用lightweight corrector。这些是LLM/VLM的safety alignment工作, SafeVLA是physical world的extension。整个PKU group的工作路径可以看作: **从abstract alignment → physical alignment**。

### 5.5 与Foundation Model Robotics的关系

- RT-2: VLA的开山之作
- OpenVLA: 开源generalist VLA
- π0: flow-based VLA
- RDT-1B: diffusion-based bimanual manipulation
- Gemini Robotics [https://arxiv.org/abs/2503.20020]: Google的VLA + reasoning
- Hi Robot [https://arxiv.org/abs/2502.19417]: hierarchical VLA

SafeVLA的独特position: 它不replace这些VLA backbone, 而是**add safety alignment layer on top**。任何VLA都可以通过ISA fine-tune获得safety。

### 5.6 Intuition: 为什么Lagrangian比fixed penalty强

考虑2D Pareto frontier:
- x-axis: task reward
- y-axis: cost (safety violation)
- Frontier: set of Pareto-optimal (reward, cost) pairs

Fixed penalty = $\lambda \cdot \text{cost}$ 加到reward, 等价于linear scalarization, 只能找到frontier上的**特定一点**, 该点取决于$\lambda$值。$\lambda$选错就suboptimal。

Dynamic Lagrangian = gradient ascent on dual, 自动converge到满足budget $b$ 的frontier点。这是**constrained optimization的优势**: 直接target constraint, 不需要tune scalarization weight。

paper的Figure 6直接证明了这点: 任何fixed $\lambda$都inferior to dynamic $\lambda$。

### 5.7 Intuition: 为什么safety decoupled from task success

FLaRe只有task reward, 学到的是"**greedy task pursuit**" - 任何能improve SR的action都好, 包括撞东西。当task fail时, model继续随机try, 产生更多collision。

ISA的Lagrangian让model学到"**safe-first policy**": 当task impossible时, model fall back to safe exploration (停, 有限cautious movement) 而不是panic。

这种**decoupling**在实际deployment中至关重要: real-world总有unseen scenarios, 我们不希望robot在fail时变成hazard。

### 5.8 Intuition: 为什么需要safety critical components

随机generate scenes不能sufficiently surface unsafe behaviors, 因为:
1. 大多数random scenes里safety-critical configurations很稀少
2. vanilla scenes下的unsafe actions太少, 学不到信号
3. 没有targeted induction, gradient signal被dominated by task performance

safety critical components = **importance sampling for unsafe events**。类似active learning / hard example mining的思想。

### 5.9 Limitations & Future Work

paper承认:
1. **Sim-only**: 需要更多real-world validation (尽管Section 5.3有initial attempt)
2. **Binary cost**: 没有severity weighting (打破glassware vs dropping cup应该不同)
3. **Trajectory-level credit assignment**: 只分到最后一步 (future: Hindsight, CTDE)
4. **Static constraints**: 没有dynamic constraints adapt to environment

Future方向:
- CVaR [https://arxiv.org/abs/2004.14088] for tail risk
- Severity-weighted constraints
- Language-conditioned dynamic constraints
- Real-world deployment at scale

## 6. 个人Take-aways

1. **CMDP是VLA safety的正确framework** - Lagrangian dual在LLM (Safe-RLHF)和VLA (SafeVLA)都work, 说明这是一个general alignment paradigm。

2. **Elicitation > Constraining** - Ablation显示即使有perfect CMDP mechanism, poor elicitation environment也失败。这暗示**safety data quality比algorithm更重要**, 类似"garbage in, garbage out"。

3. **Safety as decoupled behavior** - ISA学到的safety独立于task, 这是deployment-ready的关键property。vanilla RL学到的"通过unsafe行为完成任务"是dangerous的。

4. **Dynamic Lagrangian >> Fixed penalty** - 任何fixed hyperparameter都是suboptimal, dynamic dual是principled solution。

5. **Physical safety ≠ Abstract safety** - LLM的safety alignment (refuse harmful queries)和VLA的safety alignment (avoid collision)是**fundamentally different problems**, 但share methodology。这是alignment研究的一个重要unification。

6. **Foundation model safety的scale** - SafeVLA用了8 H100 × 25M steps, 比 Safe-RLHF的LLM setting expensive得多, 但依然feasible。Future需要scale到更大的VLA backbone。

7. **Generalization test设计** - Table 6的"unseen predicates" test是非常strong的generalization evidence。这种test应该在更多safety alignment paper里成为standard。

8. **Sim-to-real for safety** - paper证明safety constraints可以在sim里学, transfer到real。这对整个robotics safety field意义重大, 因为real-world safety violations代价太高。

## 7. 相关Links汇总

- SafeVLA project page: https://pku-safevla.github.io
- Safe-RLHF: https://arxiv.org/abs/2310.12773
- BeaverTails: https://arxiv.org/abs/2304.10485
- OmniSafe (SafeRL framework): https://arxiv.org/abs/2310.04505
- Align Anything: https://arxiv.org/abs/2412.15838
- Aligner: https://arxiv.org/abs/2310.01481
- SPOC: https://arxiv.org/abs/2310.01395
- FLaRe: https://arxiv.org/abs/2409.16578
- GRAPE: https://arxiv.org/abs/2411.19309
- AI2THOR: https://arxiv.org/abs/1712.05474
- ProcTHOR: https://arxiv.org/abs/2206.06994
- Objaverse: https://arxiv.org/abs/2212.05129
- PPO: https://arxiv.org/abs/1707.06347
- GAE: https://arxiv.org/abs/1506.02438
- FoundationPose: https://arxiv.org/abs/2312.08344
- PID-Lagrangian: https://arxiv.org/abs/2007.03965
- CMDP textbook (Altman): https://www.routledge.com/Constrained-Markov-Decision-Processes/Altman/p/book/9780367579250
- Recovery RL: https://arxiv.org/abs/2010.03063
- CVaR optimization: https://arxiv.org/abs/2004.14088
- Constitutional AI: https://arxiv.org/abs/2212.08073
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- π0: https://arxiv.org/abs/2410.24164
- Gemini Robotics: https://arxiv.org/abs/2503.20020
- Hi Robot: https://arxiv.org/abs/2502.19417
- DivScene: https://arxiv.org/abs/2410.02730

这篇paper在我的理解里是embodied AI safety alignment的**奠基性工作之一**, 把LLM时代成熟的对齐方法迁移到physical world, 并证明了methodology的transferability。期待看到后续在severity-weighted costs, CVaR, 和更大规模real-world deployment上的extension。
