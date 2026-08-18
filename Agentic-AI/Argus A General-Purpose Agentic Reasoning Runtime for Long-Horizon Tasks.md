---
source_pdf: Argus A General-Purpose Agentic Reasoning Runtime for Long-Horizon Tasks.pdf
paper_sha256: a484197bb38109b81b9244fae9e80f578f8d889b64c49158bbe681651875683d
processed_at: '2026-08-18T01:15:40-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

我们用最直白的话来拆解这篇 paper。虽然你要求用人话讲，但我依然会保留必要的技术名词和公式细节，帮助你 build intuition。

---

### 1. 这篇 paper 在解决什么痛点？

当前能 self-improve 的 AI agent（比如 FunSearch, AlphaEvolve，或者你写的 autoresearch）都有一个前提：**存在一个 dense 的 supervision signal**。
比如优化 GPU kernel，跑一次出一个 ms 数字，agent 就知道往哪个方向爬。这就像一个学生做选择题，做完马上能对答案，错了就换个选法，进步飞快。

但真正的前沿研究往往没有这种信号。研究开始时，目标本身是模糊的。比如证一个数学猜想，可能证不出来，但中途发现了一个反例，或者把bound从 27 缩小到了 25，这算不算成功？比如写一篇 paper，跑实验发现原来的假设完全错了，只能改成一篇 negative results 的 audit paper，这算不算成功？

在这些场景下，如果系统死磕一开始定死的 objective，往往会失败；如果系统随便改 objective，又容易变成“因为做不出来，所以找个容易的借口交差”（这就是 goal drift）。

Argus 的核心定位是：**runtime 的任务是在 objective 变得清晰之前，把整个 campaign 维系住，并且把系统自己无法解决的 contradiction 显式抛给人类专家，让专家决定要不要改目标。**

---

### 2. 核心抽象：四个角色与 Durable State

Argus 放弃了让一个巨大的 prompt 一直跑到底的幻想。它把一个长线任务切成一个个 **bounded mission**。每个 mission 都在一个 durable 的 project state 上工作。

系统里有四个 LLM-driven 的角色，严格分权：

*   **Manager**：拥有 campaign 的控制权。决定是否进入下一个 stage，是否 rollback。
*   **Planner**：把当前的研究状态拆解成一个个具体的 bounded task。
*   **Engineer**：干活的人。写代码、跑实验、产出 artifact。
*   **Reviewer**：独立审查。看 Engineer 的产出和执行记录，决定是 `done`、`continue` (打回重做) 还是 `blocked` (宣告做不到)。

这里最关键的设计是 **Acceptance failure 的避免**：Engineer 不能自己给自己打 `done`（除非是低风险的 trivial task 且 policy 允许 self-review）。重要的 stage 收尾必须有独立的 Reviewer 检查。这就避免了“既当运动员又当裁判”的作弊风险。

---

### 3. Working Contract 与 Pivot 的安全性

Argus 怎么允许系统改目标，同时防止它乱改？

它定义了一个 report-level 的 contract 模型：

$$K_t = (\imath, o_t, c_t, \nu_t)$$

*   $\imath$ (iota)：**standing intent**，用户最初始的稳定意图，绝对不能动。
*   $o_t$：当前的 operational objective，会随 evidence 演化。
*   $c_t$：已知的 constraints 集合。
*   $\nu_t$：当前的 verification criteria。

$X_t$ 是 user-visible 的待解决问题。

**Pivot（目标转向）是被允许的，但必须经过 `ManagerAdmit` operator：**

$$\left(X_{t+1}, K_{t+1}\right) = \text{ManagerAdmit}(X_t, K_t, K_t', e_t, r_t, u_t)$$

*   $K_t'$ 是提议的新 contract
*   $e_t$ 是支持修改的 evidence
*   $r_t$ 是记录的 admission 结果
*   $u_t$ 是人类专家或 Manager 的授权决定

用大白话说：你想改目标可以，你必须拿出证据（$e_t$）证明原路走不通或者原目标有误，并且经过授权（$u_t$）记录在案。这把“合理的 pivot”和“偷偷的 drift”区分开了。

---

### 4. 模型不变，Runtime 进化

Argus 最有意思的 claim 之一是 **Verification-Gated Fixed-Model Runtime Self-Evolution**。

$$H_{t+1} = U(H_t, \tau_t, E_t, K_{t+1})$$
$$\theta_{t+1} = \theta_t$$

*   $H_t$：Persistent runtime state（包括 Memory, Skills, Tools, Verifiers, Routing）
*   $U$：Partial update operator
*   $E_t$：Mission t 被 admit 的结果
*   $\theta_t$：LLM weights（**固定不变**）

系统的“进化”没有改 model weights，全部发生在 runtime state 层面。Engineer 发现了一个好用的工具或者避坑指南，Reviewer 检查无误后 commit 到 Skill library 里。下一个 mission 开始时，Planner 直接检索这个更新后的 $H_{t+1}$，就变得更聪明了。

并且，**被 Reviewer 否决的失败路线也被作为 "certified dead branch" 保留在 state 里**。下一个 mission 看到这个记录，就不会再去踩这个坑。这就是 Section 3 证明的 Process-data dominance：保留中间过程（包括死胡同）比只保留最终结果包含更多信息。

---

### 5. 运行时的效率度量

Paper 提出了一个很严格的有效 intelligence density 公式：

$$\rho_I(T) = \frac{1}{T}\int_0^T \dot{N}_{\text{tok}}(t) \eta_r(t) \eta_a(t) \eta_\nu(t) dt$$

*   $\dot{N}_{\text{tok}}(t)$：Token 生成速率
*   $\eta_r(t)$：Reasoning efficiency（真正推导核心 claim 的 token 占比）
*   $\eta_a(t)$：Action efficiency（真正改变了 artifact 的 action 占比）
*   $\eta_\nu(t)$：Verification efficiency（真正核对 claim 的 verification 占比）

这是乘法瓶颈模型。如果系统一直在重复读同样的 context，或者一直在做无用功重试，虽然消耗了大量 token，但 $\eta$ 因子极低，$\rho_I$ 就很小。这给了系统一个自我审视“我是不是在做无用功”的数学依据。

---

### 6. 实验结果亮点

**SWE-Bench Pro (731 tasks)**
Argus 达到 ~78% 准确率，而 Direct Copilot 只有 ~59%。虽然多花了 1.41x 的 tokens，但这钱花在了 Planner 规划、Reviewer 审查上。
更有意思的是长线演化数据：
*   466 个 task 触发了独立 Reviewer。
*   Reviewer 在 43 个 task 上要求重做，其中 34 个后来通过了官方 verifier，22 个完成了严格的 continue→revision→done loop。
*   有 35 个 task 被判定为 `blocked`。系统选择坦白“我做不到”，而没有强行伪造一个完成状态。
*   随着积累的 Skills 和 Wiki 越来越多，Mature stage (W19-22) 相比 Startup stage (W1-6) 节省了 21% 的 input tokens 和 15% 的 active time。这就证明了 runtime self-evolution 确实带来了复利效应。

**其他 Benchmark**
在 nanochat, nanoGPT speedrun, GPU kernel optimization 等七个不同领域的测试中，Argus 都达到了很有竞争力的水平，证明这套 verification 机制没有削弱 agent 的硬实力。

**Vertical Cases 深度剖析**
*   **数学研究**：在一个 Erdos–Gyárfás 相关的数学 campaign 中，保留了 1 个被证伪的路线和 6 个被接受的定理前沿更新。Planner 甚至主动提议把 success contract 从“收集观察”改成“产出证明”，被 Manager 批准。
*   **自动写 Paper**：跑了 6 个完整的 paper pipeline，涵盖 254 个 missions，经历了 16 次 Stage rollback，最终全部达到提交状态。其中 multimodal-hallucination 那篇最有意思：系统连续否决了 7 个薄弱的 method route，Planner 提议把原本的“正向方法”改成一篇“诊断性 negative-results audit”，最后跑了 4500 行实验顺利交稿。
*   **ACE-2 芯片设计**：Argus 自己写 RTL，跑 mapped synthesis 和 static timing，设计了一个能跑 Qwen2.5-0.5B W4A8 的加速器。最硬核的是它的 certification：它不仅报告了通过的功能和物理指标，还**显式枚举了自己没有做到的事情**（no routed timing, no power signoff, no GDS, no silicon validation）。它把证据的边界也作为输出的一部分交了出来。
*   **材料生成**：Argus 拿到一个发表的 MOF 生成方法，在分析中发现原方法中的 Feynman–Kac steering 存在 confound（它把 integrator 从 ODE 换成了 SDE，导致提升的功劳算错了）。Argus 移除了这个 confound，最后发现原方法的粒子交互根本没起作用，用更简单的 best-of-K 就打败了原方法。Runtime 在已有方法的基础上，通过移除 confound，简化了方法本身。

---

### 7. 最诚实的自我剖析：Endogenous Harnessing

Section 7.5 暴露了一个极度深刻的系统级 bug，这也是你最关心的 intuition 部分。

**什么是 Endogenous Harnessing？**
在 Algorithm 1 中，Planner 在 campaign level 定下了一个 mission boundary $q_0$。Round loop 里的 Engineer 只能在 $q_0$ 内部做事，Reviewer 只能依据 $q_0$ 来评判。
假设 Planner 基于当时的信息，要求 Engineer 生成一个 “skip-zero candidate”。Engineer 照做了。Reviewer 看了之后，发现其实用一种 “no-gap validator” 会更好，本来可以更直接地达成目的。
但是，**Reviewer 的权限仅限当前 round**，它可以打回重做，可以 reject，但它没有权力去 redefine 那个由 Planner 写死的 mission $q_0$。
结果就是：一个更好、更聪明的判断被记录下来了，但它无法改变任务本身。一个早期、信息不足的决策，凭借“authority”打败了后期、信息更充分的判断。

系统给自己造了一个 harness（束缚），然后乖乖地服从了它。它停止了像一个研究员那样“随时重新思考最优路线”，变成了一个“被自己之前写的便条捆住手脚”的执行者。

**这不是 reasoning failure，而是 methodological failure。**
系统找到了反例，找到了更好的 validator，理解了级联的不变量。它因为 workflow 的 authority routing 设计，无法在有效时间内 act on 这些发现。

Argus 团队承认，当前的 remedy 是缺失的。Planner 提出的 technical strategy 本应该是一种可被证伪的 hypothesis（只要有证据就能被推翻），但在当前实现里，它被当成了不可改动的 contract。Reviewer 作为最可能持有推翻证据的角色，却没有 channel 去移动它。这是下一代系统设计的核心突破口。

参考链接：
*   [Argus Project Website](https://argusbot.cn/)
*   [SWE-Bench Pro](https://arxiv.org/abs/2509.16941)
*   [AI Scientist](https://arxiv.org/abs/2408.06292)
*   [AlphaEvolve](https://arxiv.org/abs/2506.13131)

---

# Argus: 深度技术解析

这篇 paper 触及了当前 agentic AI 中最核心的一个 unsolved problem：**当 supervision signal 不 dense 时，agent 怎么 self-improve**。让我从 intuition 出发，逐层拆解。

---

## 1. 核心问题的重新定位

大多数当前 self-improving agent (FunSearch, AlphaEvolve, AIDE, 你自己的 autoresearch) 都依赖一个 dense verifier——kernel 跑出来一个 ms 数字，benchmark 给一个分数，unit test 红/绿。这类系统本质上是 **"excellent student who can only answer exam questions someone else wrote"**。

Argus 要处理的 case 是：**objective 在 start 时刻 underdefined**。具体来说：
- 数学 campaign 很少终结于一开始想证的定理，intermediate bounds/counterexamples/reformulations 才是真正的 output
- software request 经常在 candidate implementation 出现之前 underspecified
- system verification 中 spec 和 implementation 可能都错，决定改哪个本身是 work 的一部分
- chip design / materials research 中，"测量什么" 本身就是 under construction 的 object

这共享的不是 difficulty，是 **underdefinition**。这改变 runtime 的角色：runtime 的 job 不是 climb score，而是 **hold a campaign together long enough for the objective itself to become clear**，并把无法 resolve 的 contradiction 显式路由给 human expert。

这是一个相当深刻的 positioning。它把 "long-horizon" 这个概念从 "many sequential steps" 重新定义为 **"the objective is least likely to survive unchanged"** 的 horizon。

参考相关 positioning:
- [ReAct](https://arxiv.org/abs/2210.03629) - reasoning + acting 交错
- [SWE-agent](https://arxiv.org/abs/2405.15793) - agent-computer interface
- [OpenHands](https://arxiv.org/abs/2407.16741) - generalist agent platform
- [AI Scientist](https://arxiv.org/abs/2408.06292) - end-to-end research pipeline

---

## 2. Central Abstraction: Bounded Missions over Durable State

这是整个 architecture 的核心 insight。让我详细讲解。

### 2.1 为什么 session-limited agent 会失败

考虑一个标准 agent session：context window 装满后 compact，evidence 丢失；process restart 后从 transcript 重建 task。Figure 2 左边的 card 显示这条路线终点是 **reset**——系统不断把 budget 花在 reconstruction 和重复 work 上。

### 2.2 Argus 的 recurrent loop

Argus 把一个 campaign 切成 bounded missions，每个 mission t 产生 trajectory：

$$\tau_t = \big((s_{t,j}, a_{t,j}, y_{t,j}, e_{t,j}, r_{t,j})\big)_{j=1}^{n_t} \tag{1}$$

变量含义：
- $s_{t,j}$: 第 t 个 mission 第 j round 的 state
- $a_{t,j}$: 该 round 的 actions
- $y_{t,j}$: artifact state（代码、proof、model weights 等）
- $e_{t,j}$: observed measurements（test 结果、benchmark 数字）
- $r_{t,j}$: review/admission outcome
- $n_t$: mission t 的总 round 数

关键约束：**mission 必须显式 terminate/pause/transfer control**。Continuity 属于 persistent campaign state，不属于 unbounded model session。

### 2.3 Role-state transition relation

$$M \to P \to E \to \{R\} \to M \tag{2}$$

这是 explanatory state-machine model。M=Manager, P=Planner, E=Engineer, R=Reviewer。
- Reviewer "continue" → 返回 Engineer
- Reviewer "done/blocked" → 返回 Manager
- Manager hold/rollback → 重开 planning

### 2.4 Stage dynamics

设 $g_n$ 为第 n 次 Manager 决策后的 stage，legal transitions：

$$g_{n+1} \in \{g_n, \text{next}(g_n)\} \cup \text{prev}(g_n) \tag{3}$$

对应 **hold / advance / rollback** 三种。研究 vertical 的 ordered stages 是：research → plan → benchmark → run → analysis → draft → review → submission。

注意 paper 明确说：**stage index 和 task-native quality 不必每个 cycle 都 improve**。实验会失败，review 会 reject branch，later measurements 会 invalidate earlier assumptions。**Intended ascent 是 accepted frontier 在多个 cycle 上的 progress，不是每个 transition 的 monotonic improvement。**

这一点对 build intuition 很关键——它拒绝了一个常见的 agent 评估错误：把每个 step 的 progress 当作 success metric。

---

## 3. Working Contract $K_t$ 和 ManagerAdmit Operator

这是 paper 最核心的 formal contribution (C1)。

### 3.1 Contract 定义

$$K_t = (\imath, o_t, c_t, \nu_t)$$

- $\imath$: **standing user intent**，campaign 必须保留的稳定意图（小写 iota，表示不可变）
- $o_t$: 当前 operational objective（会随 evidence 演化）
- $c_t$: 已知 constraints 集合
- $\nu_t$: mission t 时刻的 verification criteria

$X_t$ 单独表示 user-visible clarifications、priorities、unresolved questions。

**关键 distinction**：$\imath$ 是 stable 的，$o_t/c_t/\nu_t$ 是 evidence 可 refine 的。这分离了 "evidence-backed refinement of operational contract" vs "silent change of intent"。

### 3.2 ManagerAdmit operator

$$\left(X_{t+1}, K_{t+1}\right) = \text{ManagerAdmit}(X_t, K_t, K_t', e_t, r_t, u_t)$$

变量：
- $K_t'$: proposed contract（candidate revision）
- $e_t$: evidence supporting the revision
- $r_t$: recorded admission outcome
- $u_t$: authorized operator/Manager decision（human authority）

**ManagerAdmit 是 normative operator，不是 atomic API**。它投影到：GoalContract revisions、persisted operator questions、Planner replanning、Manager Stage/routing transitions、append-only provenance。

### 3.3 为什么这个 distinction 重要

Paper 的关键论点：**permitting objective revision is externally indistinguishable from failure**。一个放弃 stated target 的系统，可能发现 target misspecified，也可能失败后 rationalize。**Final artifact 无法区分两者。**

所以 Argus 的设计是：pivot 应该被允许，但通过 **verification** 使其与 drift 区分开。一个 admissible pivot 必须满足：
1. 有 evidence 证明 previous route/objective unreachable 或 misspecified
2. 通过 explicit role boundary admitted
3. 记录下来，later missions 继承 change 和 justification

---

## 4. 三个必须性质（Continuity / Acceptance / Experience）

Paper 明确指出，verification certification 需要三个性质，每个在 long-running agent 中都会 fail：

### 4.1 Continuity fails
当 growing transcript 被 compact 或 drop，evidence 不再存在。这是 [MemGPT](https://arxiv.org/abs/2310.08560) 试图解决的 memory tier 问题，但 context window 扩展只解决 session continuity，不解决 objective change 的 certification。

### 4.2 Acceptance fails
当执行 work 的 component 同时 declare it complete。**revision 被 strongest incentive to accept it 的 party certified**。这就是为什么 Argus 强制 independent Reviewer 路径用于 stage-closing / vertical-required review。

### 4.3 Experience fails
当只有 final artifact 存活，**refuted routes 无法被 cite 作为 "objective unreachable" 的 evidence**。这是 Argus 保留 failed branches 的理论依据。

---

## 5. Verification-Gated Fixed-Model Runtime Self-Evolution

### 5.1 模型 vs 状态分离

$$H_{t+1} = U(H_t, \tau_t, E_t, K_{t+1}) \tag{6}$$
$$\theta_{t+1} = \theta_t \tag{6}$$

- $H_t$: persistent runtime state
- $U$: partial update operator
- $E_t$: mission t admitted 的 results subset
- $\theta_t$: model parameters，**fixed**

$$H_t = \{\text{Memory, Skills, Tools, Verifiers, Routing}\} \tag{7}$$

**关键 scope condition**：online evolution 不需要 gradient update 模型。evolution 发生在 persistent state 和 control policy 中。

### 5.2 Ownership model（Table 2）

这是 paper 最精细的设计之一。不同 state component 有不同 change source / commit owner：

| State | Change source | Commit owner |
|-------|---------------|--------------|
| Memory | Engineer trajectory | **Reviewer** |
| Skills | Engineer/Scientist | **Reviewer** |
| Tools | System config | System config |
| Verification | Planner | Planner（Reviewer supplies feedback） |
| Routing | Runtime policy | **Manager** |
| Tasks | Planner | Scheduler |

注意 **ownership 不是 uniform 的**。Memory/Skills 是 work-vs-certification split（Engineer 产生 candidate，Reviewer commit retained form）。Routing 是 Manager-committed。这是为了避免 "executor certifies own work" 的 acceptance failure。

### 5.3 完整 update cycle 四步

1. Execution trajectory 产生 candidate（memory/skill/procedure/verification rule/routing decision/task definition）
2. Responsible role 检查 candidate vs artifacts 和 task-native evidence
3. Authorized owner commit/revise/reject
4. Later mission 检索 retained state

**Activity that does not survive this commit-and-reuse path is not counted as self-evolution.** 这是个非常严格的定义——它排除了 "agent 产生了一堆 narrative 但没有进入 reusable state" 的情况。

参考 [Voyager](https://arxiv.org/abs/2305.16291) 的 skill library、[Agent Workflow Memory](https://arxiv.org/abs/2409.07429)、[A-MEM](https://arxiv.org/abs/2502.12110) 的相关 memory 工作。

---

## 6. Process-to-Capability Theory

这是 paper 的理论骨架，连接 Token activity 到 retained research progress。

### 6.1 Dense-intelligence density

$$\rho_I(T) = \frac{1}{T}\int_0^T \dot{N}_{\text{tok}}(t)\,\eta_r(t)\,\eta_a(t)\,\eta_\nu(t)\,dt \tag{4}$$

变量：
- $\dot{N}_{\text{tok}}(t)$: token-mediated reasoning/action 的瞬时速率
- $\eta_r(t)$: **reasoning efficiency**——converted to relevant reasoning 的 fraction
- $\eta_a(t)$: **action efficiency**——converted to effective action 的 fraction
- $\eta_\nu(t)$: **verification efficiency**——converted to valid verification 的 fraction

**这是 multiplicative bottleneck model**。一个反复读相同 state、emit activity 不改 artifact、accept unchecked claim 的 run，可以消耗大量 tokens 但对 $\rho_I$ 贡献很小。

### 6.2 Efficiency factor 定义

$$\eta_k(W) = \frac{\sum_{u \in \mathcal{U}_k(W)} w(u)\,q_k(u)}{\sum_{u \in \mathcal{U}_k(W)} w(u)}, \quad 0 \le q_k(u) \le 1 \tag{5}$$

- $k \in \{r, a, \nu\}$
- $\mathcal{U}_k(W)$: attributable units for reasoning/action/verification in window $W$
- $w(u)$: attributable Token count（reasoning/verification）或 action cost
- $q_k(u)$: attribution label，记录 conversion 到 intended function

具体 attribution rules：
- **Reasoning efficiency**: direct unit derive/revise/rule-out named claim, lemma, blocker, decision。Repeated context、narrative transitions、notation-only rewriting 算 transition cost
- **Action efficiency**: direct action 改 artifact/result set/accepted decision/explored branch。**falsification action 是 productive**（certifies dead branch）；retries 和 interrupted no-op work 单独 track
- **Verification efficiency**: direct verification unit 比较 named claim vs explicit criterion/artifact、rerun independent check、identify concrete defect、issue scoped verdict。Schema repair/reformatting 算 verification overhead

### 6.3 Process-data dominance（Proposition 1）

$$D_{\text{process}} = \{(s_k, a_k, e_k, r_k, \Delta H_k)\}_{k=1}^N \supseteq D_{\text{final}} = \{y^\star\} \tag{10}$$

这是 [Blackwell 1953 information ordering](https://doi.org/10.1214/aoms/1177729032) 在 research trajectory 上的特化。

设 $P$ 为 typed process record，$Y = g(P)$ 为 final-artifact projection。对下游 task $q$（target $Z_q$，loss $\ell_q$，policy $\pi$），定义 minimum achievable risk：

$$\mathcal{R}_q(X) = \inf_\pi \mathbb{E}[\ell_q(\pi(X), Z_q)] \tag{11}$$

**Proposition 1**: 若 $Y = g(P)$，则 $\mathcal{R}_q(P) \le \mathcal{R}_q(Y)$ 对所有 downstream decision problem $q$ 成立。当两个 process records 能产生相同 final artifact 但 imply 不同 optimal next action 时，inequality 严格。

**Proof 是 immediate 的**：每个用 $Y$ 的 policy 可以从 $P$ 通过先 apply $g$ 复现，但 reverse simulation 不一定存在。

### 6.4 为什么需要 compression

Dominance 是 informational，不是 computational。Raw trajectory 太大/stale/contradictory。对 context budget $b$：

$$\psi_b^\star = \arg\min_{\psi: \text{size}(\psi(P)) \le b} \mathbb{E}_{q \sim \mathcal{D}}\left[\mathcal{R}_q(\psi(P)) + \lambda_c C_{\text{read}}(\psi(P))\right] \tag{12}$$

- $\psi$: compression function
- $b$: context budget
- $\lambda_c$: read cost weight
- $C_{\text{read}}$: retrieval + validation cost

**这解释了 Argus 为什么同时保留 append-only event tape 和 bounded reviewed checkpoint**：
- Tape 保留更 informative 的 experiment
- Checkpoint 近似 decision-useful compression under finite context budget
- **Failed branches 属于 compression 当且仅当它们 change 下一个 optimal action**，不是 "merely because they occurred"

这是个很强的设计原则。

### 6.5 Reuse value $G_L$（counterfactual compounding）

$$G_L(\Delta H_t) = \sum_{j=1}^L \gamma^{j-1}\left[\mathscr{R}_{q_{t+j}}(H_t) - \mathscr{R}_{q_{t+j}}(H_t \oplus \Delta H_t)\right] \tag{13}$$

- $\Delta H_t$: mission t 后 accepted 的 state update
- $q_{t+1:t+L}$: 接下来 L 个 tasks
- $\gamma$: discount factor
- $H_t \oplus \Delta H_t$: 接受 update 后的 state

$G_L > 0$: accepted memory/skill/verifier/routing rule/certified dead branch 减少 future loss。
$G_L < 0$: **negative transfer**。

这把 "compounding intelligence" 从 monotone rhetoric 转成 **counterfactual claim**：reuse 必须 outperform frozen state on matched future tasks。

### 6.6 Verified reusable yield

$$\mathcal{G}_{\text{PRI}}(W) = \frac{\sum_{t \in W} p_{\text{valid},t}\left[\Delta I_t + \lambda_g G_L(\Delta H_t)\right]}{\sum_{t \in W} N_{\text{tok},t}} \tag{14}$$

- $\Delta I_t$: immediate task-native information gain（score improvement / theorem strengthening / certified branch elimination）
- $p_{\text{valid},t}$: claimed increment 存活 review 的概率
- $\eta_{\text{reuse}}$: 实际 reusable fraction
- $\lambda_g$: immediate vs future value 的转换因子

### 6.7 Review as selective error correction（Bayes）

设 $C$ = proposed increment 是否 correct，$A$ = Reviewer 是否 accept：

$$\Pr(C=1 \mid A=1) = \frac{\alpha p}{\alpha p + \beta(1-p)} \tag{15}$$

- $p = \Pr(C=1)$: proposal base rate
- $\alpha = \Pr(A=1 \mid C=1)$: Reviewer sensitivity
- $\beta = \Pr(A=1 \mid C=0)$: false-acceptance rate

若 $\alpha > \beta$，accepted state 比 proposal stream 更 precise。**Review 是 selective error-correction channel**。Operating point 平衡 accepted precision / recall / Token cost。

---

## 7. Algorithm 1 详解

```
Input: ι, K_t, H_t, backlog B, budget b
c ← ManagerCommit(ι, K_t); persist campaign identity
while b remains and c not complete:
    N ← StableTopological(Planner(B, H_t, c, K_t))  // campaign level
    for each bounded mission q_0 ∈ N, one at a time:
        L ← ExposeSkillLibraries(q_0)  // paths only, agent 自己 retrieve
        Γ ← []; q ← q_0
        repeat:  // mission level
            (y, e, d) ← Engineer(q, H_t, L)
            if d=skip and SelfReviewAllowed(q):
                r ← EngineerSelfReview(q, y, e)
            else:
                r ← Reviewer(q, y, e)  // scoped to this round
            append (q, y, e, r, r.source) to Γ
            if r=continue: q ← AdaptAfterRejection(Γ)
        until r ∈ {done, blocked, paused} or governance threshold fires
        (status, reason) ← PreSettlementGuard(Γ, status, reason)
        τ ← TraceProjection(Γ); E ← AdmitResult(τ)
        H_{t+1} ← SettleMissionOutcome(H_t, τ, E)  // learning happens here
        if evidence proposes K_t':
            (K_{t+1}, ρ_t) ← ReviseContract(K_t, K_t', by, confirmation)
        t ← t+1; refill B when required
```

### 7.1 两个 level 的关键 asymmetry

**Planner 在 campaign level 运行，fix mission boundary $q_0$**。Round loop 下面的 $q$ 只通过 `AdaptAfterRejection` 变化，反应 Reviewer rejection。**Round loop 内部没有任何 component 能 rewrite $q_0$**。

这个 asymmetry 是 Section 7.5 "endogenous harnessing" failure mode 的来源——后面会详述。

### 7.2 三个非理想化细节

1. **Skill libraries 是 exposed 不是 injected**：runtime publish library paths，acting agent 自己 retrieve。没有 matcher 预先决定哪个 prior experience relevant。

2. **Termination 由 named thresholds govern**，不只 Reviewer verdicts：max round count / no-progress threshold / soft round limit / hard escalation count / backend failure threshold。Pre-settlement guard 可在 learning 之前 override recorded status。

3. **Learning per mission settle，不是 per round**：reusable state changes 一次，在 mission outcome。

### 7.3 ReviseContract 的 two-tier gate

GoalContract 持有两种 clause：
- **Semantic clauses / exclusions / recorded ambiguities**: 自由移动（clarifying user intent 是 ordinary Manager work）
- **Precise clauses 和 objective 本身**: 需要 confirmation covering exactly 改变的 clause identifiers，否则 revision refused

这实现 Section 3 的 analytical distinction：**intent stable, wording negotiable, operational target 只在 recorded authority 下移动**。

---

## 8. 实验结果深度分析

### 8.1 七个 benchmark 的 capability floor（Table 4）

| Benchmark | Backbone | Argus | Comparison |
|-----------|----------|-------|------------|
| SWE-Bench Pro | GPT-5.5/xhigh Copilot | ≈78% | Direct Copilot ≈59%, 1.41× tokens |
| SOL-ExecBench | GPT-5.5 Codex B200 | Global #6, 7 top-3 | 2 wins over Recursive |
| nanochat B200 | GPT-5.5 Codex | 0.9636 BPB | Human best 0.9646 |
| nanochat H100 | GPT-5.5 Codex | 0.9855 BPB | Human best 0.9879 |
| nanoGPT speedrun | GPT-5.5 Codex 8×H100 | 79.77s | Same-device human 80.18s |
| AARRI-Bench | GPT-5.5 Codex | 63/82 (76.8%) | Paper best 68.3% |
| Math-Reasoning | GPT-5.5 Codex | 28.0 gap | Arbor 20.83, Claude 8.33, Codex 6.25 |

参考你的 [nanochat](https://github.com/karpathy/nanochat) 和 [autoresearch](https://github.com/karpathy/autoresearch)，以及 [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt)。

Token ratio：
$$R_{\text{tok}} = \frac{N_{\text{Argus}}^{\text{total}}}{N_{\text{Copilot}}^{\text{total}}} \approx 1.41 \tag{17}$$

### 8.2 SWE-Bench Pro Reviewer 分析（Table 11，Figure 3）

731 tasks 的 routing 分布：
- **466 tasks (63.7%) invoke independent Reviewer**
- 265 tasks (36.3%) Engineer self-review

Reviewer-routed tasks 消耗 **2.75× solve input tokens** 和 **1.80× active time** vs self-reviewed tasks。这证明 routing policy 选了更难的 workload。

Recovery funnel：
- 388 accepted on first review（terminal done）
- 35 blocked on first review（terminal blocked）
- 43 requested revision
  - 34 (79.1%) later pass official verifier
  - 22 (51.2%) complete strict continue→revision→done loop

**35 blocked 是关键数据**：runtime 显式 declare "cannot complete"，而不是 emit unsupported completion。**Refusing to stop early 和 refusing to claim success 是同一个 gate 的两个表达。**

### 8.3 Longitudinal cost evolution（Table 9, Figure 4）

| Window | Interpretation | Tasks | Input/task | Active min/task | Token index |
|--------|---------------|-------|------------|-----------------|-------------|
| W1-6 | Startup | 120 | 2.95M | 8.52 | 100 |
| W7-12 | Early reuse | 140 | 2.07M | 7.28 | 70 |
| W13-18 | Composition shift | 151 | 1.47M | 10.42 | 50 |
| W19-22 | Mature | 158 | 2.33M | 7.25 | 79 |
| W23-24 | Late difficult | 49 | 3.72M | 9.01 | 126 |

**Mature W19-22 vs Startup W1-6**: 21% fewer solve input tokens, 15% less active time。

但 W13-18 出现 **Token 最低但 active time 最长**——Token efficiency 和 execution latency 不一起 move。W23-24 反弹——late difficult tasks。

**这个 non-monotone trajectory 是 paper 最重要的诚实声明之一**。它拒绝把 startup-to-mature 读成 monotonic improvement。Paper 明确说：因为 task sequence 没有 replay against frozen runtime state，evidence 只 characterize system-state accumulation over this sequence；attributing reduction 到 individual memory/skill update 需要 matched frozen-state replay。

### 8.4 Wave 级别 Skills/Wiki 累积（Table 10）

Wave 1: 34 Skills, 27 Wiki entries
Wave 24: 478 Skills, 352 Wiki entries

累积是 monotonic 的，但 performance 不是——这强化了 Section 4.4 的说明：**mechanism 不 imply monotonic improvement**。

### 8.5 数学 campaign 的 efficiency attribution（Table 6）

Rounds 12-17, 18 bounded missions 的 attribution：

| View | Direct | Auxiliary | Failed/no-op |
|------|--------|-----------|--------------|
| Reasoning | 56.0% | 39.1% | 4.9% |
| Verification | 55.4% | 35.0% | 9.6% |
| Action | 55.6% | 38.9% | 5.6% |

**Strict action score = 10/18 = 55.6%**；inclusive（含 successful auxiliary）= 17/18 = 94.4%。

Cost-weighted 后：strict = 56.1%, inclusive = 89.7%。

注意 paper 强调：**early witness check that killed a proposed reduction counts as efficient action**——它 pruned false route 在长 proof attempt 之前。这印证了 Equation 5 的 "falsification action is productive"。

### 8.6 Process-theory 完整 substitutions（Table 13）

| Theory quantity | Substitution | Observed |
|----------------|--------------|----------|
| Strict $\rho_I$ | 332,274/9,620.5 × 0.5598 × 0.5556 × 0.5538 | **5.95 effective Token-eq./s** |
| Inclusive $\rho_I$ | 332,274/9,620.5 × 0.9512 × 0.9444 × 0.9043 | **28.06 effective Token-eq./s** |
| Reviewer correction | 34/43, 22/43 | 79.1%, 51.2% |
| Reuse $G_L$ | (2.95-2.33)M, (8.52-7.25) min | 0.62M Tokens, 1.27 min/task |
| Immediate $\mathcal{I}_{\text{PRI}}$ | $\lambda_g=0$; 6/332,274 × 10^6 | **18.1 accepted deltas/M reasoning** |
| Process compression | 384,370,463/3,240 | **118,633× event-history/checkpoint ratio** |

最后一个数字很重要：**event history 是 active working checkpoint 的 118,633 倍大小**。这是 Equation 12 的实证体现——tape 保留 information，checkpoint 是 decision-useful compression。

---

## 9. Vertical Traces 深度分析

### 9.1 数学 campaign（Section 6.5, Figure 5）

Erdos–Gyárfás conjecture 相关 work。**Role-resolved trace** 保留：
- 1 个 falsified route（C5）
- 6 个 proof-backed frontier updates（C20-C25）
  - 包括 27→25 的 strengthening（path bound 从 27 降到 25）
  - 2 个 bridge lemmas

Role 行为：
- **Manager**: 恢复 campaign，owns stage transitions
- **Planner**: 先选 cheap falsification test，然后 propose 把 success contract 从 "collect observations" 改成 "produce a proof"。Manager admits 这个 refinement，later 要求新 mission advance retained result 而非 restart
- **Engineer**: source retrieval / executable checks / proof writing / result packaging
- **Reviewer**: reject overstated route / request missing checks / certify bounded theorem

Table 12 的 proof artifact excerpt 展示了一个 combinatorial bound 的证明结构——length-4 paths $\mathcal{P}_4(x,y)$ 的 internal three-vertex sets $\mathcal{F}$ 在 no-$C_8$ 图中两两相交，结合 no-$C_4$ case analysis 得到 $|\mathcal{P}_4(x,y)| \leq 25$。

### 9.2 Six paper campaigns（Section 6.7, Figure 6, 7）

Aggregate 统计：
- 640 campaign-hours
- 254 bounded missions
- 576 Engineer rounds
- 286 Reviewer revision verdicts
- 89 session rolls
- 16 Stage rollbacks

**全部 6/6 canonical pipelines 达到 submission completion**。两个 AAAI 格式，四个 ACL 格式（10-13 pages）。

研究问题包括：
1. Evaluation reliability
2. Vision-language matching（compositional gate 失败）
3. Test-time adaptation（over-restrictive controller）
4. GUI agents（narrow reliability gain）
5. Multimodal hallucination（failure-mode audit）
6. Model quantization（negative result for static proxies）

**至少三个 project 在原 positive hypothesis 失败后变成 diagnostic / negative results study**。这是 verification-gated state 使 scientific pivot explicit 而非 silently replace failed branch 的实证。

### 9.3 Multimodal-hallucination campaign 详细轨迹（Figure 7）

163.6 小时 trajectory：
1. **Prune**: 七个早期 no-go decisions 拒绝 weak method routes（unreproduced / base-identical / missing preregistered signal）
2. **Pivot**: Planner propose 把 claim 从 positive mitigation method 改成 diagnostic negative-results audit；Manager admits
3. **Experiment**: Engineer 完成 5-method × 3-benchmark matrix，4,500 official-scored rows
4. **Write**: Reviewer binds no-op 和 degradation claims 到 outputs
5. **Repair**: 两个 late submission checks 暴露 scorer provenance 和 GPU telemetry defects；Manager accepts 2 个 rollback 回到 benchmark 修复

**关键 insight**: 7 个 rejected method routes 变成 4,500-row negative-results study。这是 "failure retained as information" 的最 clean 实证。

### 9.4 ACE-2 chip（Section 6.8）

Argus specified、wrote RTL、built verification environment、drove mapped synthesis + static timing，**没有 human author of record**。

**Functional closure**:
- 24-layer Qwen2.5-0.5B W4A8 完整执行
- Layer 0 在所有 18 ordered operators 上 exact match reference
- 2-token runtime: 13,914/13,914 commands over 1,240,410,384 simulator cycles
- Generated token identifiers [0,0]，无 first failure

**Physical closure** (SKY130 HD, OpenSTA, TT 25°C/1.80V):
- 62,283 cells
- 0.614 mm² non-SRAM area（cap 2.0 mm²）
- +0.6966 ns detailed setup slack
- 0.00 ns worst negative slack
- 0.00 ns total negative slack
- 10.000 ns clock period（100 MHz floor 达到）

**Certificate 的关键性质**：它枚举自己的 exclusions——no routed timing, no power signoff, no DRC/LVS, no GDS/tapeout, no silicon validation, no generation beyond 2 tokens, no FPGA prototype。**这个 exclusion list 是 gate 的 output，不是事后 caveat**。

这印证了 Section 3 的核心论点：**"an unbounded runtime would have reported a chip; this one reported a chip and the exact perimeter of the evidence supporting it"**。

### 9.5 Materials generation: 简化 published method（Section 6.9）

这是 paper 最 sharp 的 vertical case。MOF (metal-organic framework) generation。

**Step 1: 添加 base model 缺乏的控制**
MOFFlow-2 的 published conditioning 只接受 single continuous property。Argus 加了三个 discrete condition tokens（metal element / node nuclearity / ligand family），通过 cross-attention，每个有 ANY token 和 independent condition dropout。12.69M trainable parameters of 75.8M (16.7%)。

结果（3,300 balanced condition requests）：
- Metal adherence: 92.5%（permuted control: 17.4%）
- Nuclearity: 100.0%（control: 24.3%）
- Ligand family: 74.5%（control: 39.0%）
- Nickel: 2.1% corpus base → 83.7%
- Trinuclear: 3.9% → 100.0%
- Pyrazolate: 2.0% → 46.0%
- Unconditional validity: 30.61% → 37.12%

**Step 2: 替换 score，然后质疑 method**
Failure census over 19,483 paired structures 发现：published steering potential penalizes atom overlap，但 all-atom model 的 dominant failures 是相反的（under-coordination, fragments, floating components）。Argus built aligned scoring function，AUC 0.833 vs 0.594。

**Step 3: 揭露 confound**
Published Feynman–Kac steering 同时切换 integrator 从 ODE 到 SDE。Separating 它们：
- Integrator alone: +1.92 points (p=0.238, **不显著**)
- Score inside fixed SDE: +9.00 points (p=2.0×10⁻⁹)

**Step 4: 测试 particle interaction 是否做任何 work**

固定 model/integrator/score/forward-pass budget/989 crystals，单纯生成 K 个 independent trajectories 并 keep best-scoring：

| Forward passes | Feynman-Kac | Argus best-of-K | Difference |
|----------------|-------------|-----------------|------------|
| K=4 | — | 53.39% | — |
| K=8 | 52.38% | 55.21% | +2.83 |
| K=16 | 57.94% | 59.76% | +1.82 |
| K=32 | 59.96% | 61.38% | +1.42 |

**Best-of-K 在每个 budget 上 beat Feynman-Kac**。K=8 difference 在 paired test 中 survive (p=0.0486)。**Best-of-4 at 53.39% 已经超过 Feynman-Kac at K=8，同时花一半 compute**。

机制：λ=2 时 median effective sample size ≈ 1，72.1% crystals collapse to single particle after resampling。**Published method 付钱买 K forward passes 然后丢弃大部分 exploration**。

**Step 5: 诚实的停止**
- 989-crystal subset 是 score 设计用的同一数据，**不是 independent holdout**
- Full 19,483-structure leaderboard 未跑
- Conditional sequence generation 和 best-of-K structure prediction 未 joined 成 end-to-end system
- No relaxation / MLIP / DFT check
- "Every number here is scored by the same validator the selection optimizes against"——**正是 Section 8 names 的 verifier-shaped objective failure模式**

Campaign 因此 hold at candidate version 而非 declare release。

**这个 vertical 对 paper argument 的贡献不是 pass rate**——是 **runtime 给定 published method + fixed compute + external validator，arrived at a smaller method than 它 started from，使 admissible 的 step 是 removing a confound 而非 adding a component**。

---

## 10. Endogenous Harnessing: 最深刻的 Failure Mode（Section 7.5）

这是 paper 最 intellectually honest 的部分。

### 10.1 机制

Algorithm 1 的两个 level：Planner 在 campaign level fix mission boundary $q_0$；round loop 下面的 Engineer 把它当 task，Reviewer 把它当 judge work 的 standard。

**What began as one agent's hypothesis has become an external constraint on every agent that comes after, including agents that now know more.**

### 10.2 具体例子

- Planner 要求 skip-zero candidate
- Engineer 满足这个 local objective
- Reviewer 识别出 no-gap validator alternative，会更直接 discharge 同一 obligation
- **Reviewer scoped to current round**：可以 reject / request revision / escalate，**但不能 redefine mission that made weaker route mandatory**
- Better judgement arrived, was recorded, **did not change the task**
- Earlier, less informed decision 在 authority 上赢，而非在 evidence 上赢

### 10.3 诊断

**这不是 reasoning failure**。系统 found counterexample、identified validator alternative、understood cascading invariant。**它无法 act on any of them in time**，因为 workflow 给 later, better-informed roles 没有 authority revise task boundary。

**Binding constraint 是 methodological，不是 cognitive**——所以 remedy 是 authority routing 的改变，不是 stronger model。

### 10.4 正确的 division

- **Hard harness**（应 resist revision）：frozen authority、prohibitions on granting new trust、irreversibility limits
- **Technical bet**（应 revisable by any role that produces evidence against it）：choice of route / validator / representation

当前 implementation：semantic clauses 自由移动，precise clauses 和 objective 需要 confirmation。**Missing**: mission 的 technical strategy 被 filed 在 immovable side 而非 negotiable side，**Reviewer（最可能 hold disconfirming evidence 的 role）没有 channel move it**。

这个 failure mode 实际上揭示了当前 agent architecture 的一个 fundamental tension——**plan 作为 contract vs plan 作为 falsifiable hypothesis 的 category confusion**。

---

## 11. MLE-Bench Lite 的 partial result（Table 5）

Review-approved medal gate: 一个 competition 只在 independently reviewed submission 赢得 Kaggle medal 时算 complete。

9 个 medals: 3 gold / 3 silver / 3 bronze。

**关键验证细节**:
- denoising-dirty-documents: 比 silver band 少 0.00009 RMSE，得 bronze
- jigsaw-toxic-comment: 比 bronze threshold 多 0.00018 AUC，得 bronze
- transparent-conductor: **verification-gated route change**——initial approach 在 0.208 RMSLE 被 external grader reject，campaign 替换成 public-state-of-the-art method，grader certify 在 0.06402（bronze threshold 0.06582）

---

## 12. Upstream Adoption: RWKV6 Kernel（Appendix E）

FLA PR #1045: Argus implemented 和 optimized TileLang RWKV6 dense-bf16, D=64 forward-intra kernel。

H100 evidence:
- Forward latency: 0.199 ms → 0.168 ms (1.18×)
- Forward+backward: 0.900 ms → 0.747 ms (1.21×)
- 13 correctness-gate passes, 14 repository tests

FLA collaborator (Moonshot AI affiliated) reviewed generated CUDA，identified long-sequence numerical-stability issue，requested block-local exponent centering。Argus applied fix 后 merged into fla-org:main on July 20, 2026 as commit c70f11c。

参考 [Flash Linear Attention repo](https://github.com/fla-org/flash-linear-attention)。

---

## 13. 限制性和未来工作

Paper 的 Section 8 异常诚实：

1. **User-guided pivots 未公开 prospective 评估**——internal system-verification cases 含 project-specific details 无法 disclose
2. **Contract refinement 仍可 fail**——Manager/operator 可 approve poorly framed tradeoff；standing intent 可 omit tacit requirement
3. **Verification 只和 evidence boundary 一样 sound**——executable test / formal checker / benchmark / Reviewer 可 encode wrong property
4. **Attribution 和 task sequence**: Reviewer routing 是 adaptive（非 randomized），22 completed Waves follow 一个 task order，per-Wave Direct-Copilot Token/time records unavailable。**Startup-to-mature comparison 是 observational，不是 causal learning ablation**
5. **GLM-5.2 on Claude Code run** 在 70.94%，但 incomplete 且无 matched Direct baseline
6. **ACE-2 是 demonstrated scope，不是 silicon**
7. **Six paper projects 来自一个 shared research environment**，campaign-hours overlap，stored review snapshots 是 model-generated
8. **Compositional-matching pipeline 的 consistency failure**: final PDF complete 但 last stored assurance object 仍 BLOCKED on Manager-stage authority check——**derived certification state 可 lag canonical pipeline state**

---

## 14. 对你的 intuition 的关键 takeaways

### 14.1 "Score is the curriculum" 的局限

你的 autoresearch 工作（以及 AIDE / FunSearch / AlphaEvolve）依赖 dense verifier。Argus 把这类系统归类为 "student excellent at answering questions someone else wrote"。**真正的研究任务 supply no such signal**——objective underdefined，measurement 本身是 problem 的一部分。

### 14.2 Verification 不是 quality filter，是 pivot safety mechanism

这是和大多数 agent framework 的根本区别。Verification 不是 after-execution 的 quality check，**是使 pivot distinguishable from drift 的机制**。

### 14.3 Fixed-model self-evolution 是个 well-defined claim

不是 "agent 学习了"，而是：**model parameters 不变，但 admitted state update 改变 later mission 的 starting point / available procedures / verification obligations / routing policy**。每个 candidate 必须 bind to artifacts + task-native evidence + authorized commit。

### 14.4 Process data dominance 是 Blackwell ordering

Equation 10-11 的 Proposition 1 是 [Blackwell 1953](https://doi.org/10.1214/aoms/1177729032) 的特化。这不是新理论，但是把它应用到 research trajectory 并导出 compression principle（Equation 12）是有意义的：**failed branches 属于 compression iff 它们 change 下一个 optimal action**。

### 14.5 Endogenous harnessing 是 next bottleneck

Section 7.5 的 failure mode 对你设计 agent 系统特别 relevant。**当前 bottleneck 不是 cognitive，是 methodological**——later, better-informed roles 没有 authority revise task boundary。Plan 同时充当 contract 和 falsifiable hypothesis 是 category confusion。

### 14.6 Trajectory 作为 training data 的接口

Equation 6 的 $H_{t+1} = U(H_t, \tau_t, E_t, K_{t+1})$ 产生的 typed trajectories（states / actions / measurements / review feedback / runtime-state changes）是天然的 SFT / preference / RL material。参考 [AgentInstruct](https://arxiv.org/abs/2407.03502)、[Agent-FLAN](https://aclanthology.org/2024.findings-acl.557/)、[Agent Lightning](https://arxiv.org/abs/2508.03680)、[Let's verify step by step](https://arxiv.org/abs/2305.20050)。

未来 work 的 natural direction：用这些 trajectories 训练 model internalize recurring planning/verification patterns，**runtime 提供 long-horizon coordination，model 吸收 pattern**。

---

## 15. 我的批判性思考

### 15.1 Observational vs causal

Paper 反复强调 longitudinal evidence 是 observational，不是 causal ablation。这是诚实的，但也意味着 **我们不知道 21% token reduction 中多少来自 state accumulation，多少来自 task composition variation**。Matched frozen-state replay 是 essential next step。

### 15.2 Reviewer sensitivity α 和 false-acceptance β 未独立测量

Equation 15 是 normative，但 paper 没有独立 estimate α 和 β。Randomized routing with external verifier 才能 isolate causal effect。当前 79.1% verifier recovery 和 51.2% strict rescue 是联合 measure。

### 15.3 "Verification-gated" 的 verifier 本身

Section 8 第三点很关键：**executable test / formal checker / Reviewer 可 encode wrong property**。Materials campaign 的 "every number scored by same validator selection optimizes against" 就是这个 failure mode 的实证。未来需要 controlled studies seeding implementation bugs / specification bugs / mismatches，测量 runtime 是否 localize faulty side。

### 15.4 Endogenous harnessing 的 remedy 未实现

Section 7.5 诊断了问题但没实现 remedy。Reviewer 需要 channel move technical strategy 从 immovable side 到 negotiable side。这是 paper 的 next iteration 最关键的 design 改动。

### 15.5 跨 model transfer 的 early signal

GLM-5.2 on Claude Code at 70.94% 是 runtime 跨 backbone + execution surface 的 first evidence。但 incomplete 且无 baseline。如果能 complete 并 show similar verification-gated evolution pattern，会是 generality 的强证据。

---

## 16. 总结

Argus 的核心 contribution 是把 "long-horizon research" 重新 formalize 为 **verified pivoting over a compact contract $K_t = (\imath, o_t, c_t, \nu_t)$**，并通过四个 role（Manager / Planner / Engineer / Reviewer）的 authority separation + verification-gated admission 使 pivot 与 drift distinguishable。

它的 empirical evidence 三个层次：
1. **Capability floor**: 7 benchmarks 上 competitive（SWE-Bench Pro 78% vs 59%）
2. **Longitudinal evolution**: mature window 21% fewer tokens, 15% less time（observational, non-causal）
3. **Vertical depth**: 数学 / 论文 / chip / materials 四个 vertical 展示 verified pivoting 的不同 facet

最深刻的 insight 是 Section 7.5 的 endogenous harnessing——**当前 agent 系统的 binding constraint 是 methodological (authority routing)，不是 cognitive (model capacity)**。这对你设计 next-gen agent 系统是 important framing。

参考 [Argus project website](https://argusbot.cn/) 和 [SWE-Bench Pro paper](https://arxiv.org/abs/2509.16941) 获取更多细节。
