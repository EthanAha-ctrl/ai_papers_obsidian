---
source_pdf: EmbodiSkill Skill-Aware Reflection for Self-Evolving Embodied Agents.pdf
paper_sha256: 0e115846512fe0242569c86bef41cb1d7c28064d769f0194cafe7b524bbf4fbe
processed_at: '2026-08-18T10:53:11-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 EmbodiSkill

Andrej，行，那咱们抛开那些公式，用大白话把这个paper的核心思路捋一捋。

---

## 先说这paper想解决啥问题

想象你在教一个机器人做家务。你给它写了一本"家务手册"，里面写着：
- 找东西前先去对应的房间
- 放东西进冰箱前要先开门
- 加热食物要先把食物放进 microwave 再启动

这就是"skill"——一本持续可修改的操作指南。

现在问题来了：机器人执行任务后失败了，你怎么知道是**手册写错了**，还是**机器人没按手册来**？

举个例子：
- 手册说"放冰箱前先开门"
- 机器人失败了，没开门就硬塞
- 如果你直接把手册改成"一定要先开门再放东西"——但手册本来就写了这条啊！改了等于没改，甚至可能改坏

这就是现有方法的问题。以前的 Voyager、Reflexion 这类做法，看到失败就总结成一句 feedback，然后把整个手册重写一遍。这在 Minecraft 那种确定性的环境里还好使，因为失败基本就是策略有问题。

但 embodied 环境不一样。机器人失败可能是因为：
- 手册真的写错了（skill defect）
- 手册没问题，机器人自己没执行好（execution lapse）
- 还有感知错了、物体状态变了、action precondition 没满足等一堆因素 entangle 在一起

如果不区分这些情况，一股脑改手册，改着改着手册就废了——好的内容被覆盖，冗余的内容堆积，错误的内容被保留。手册越改越烂。

---

## EmbodiSkill 的核心思路

一句话：**别急着改，先搞清楚这次 trajectory 告诉你手册哪里出了什么问题，再决定改哪里、怎么改。**

具体怎么做？把 trajectory 和 skill 对照着看，把 trajectory 提供的 evidence 分成 4 类：

### 4 种 reflection type

**成功的 trajectory 也能产生 reflection**（这点很关键，很多方法只在失败时反思）：

1. **DISCOVERY**（发现新大陆）
   - 手册里没写，但这次 trajectory 发现了一个有用的 pattern
   - 比如：原来找 remote control 应该先去 sofa 旁边找
   - 动作：往手册里**加**新内容

2. **OPTIMIZATION**（原来能做得更好）
   - 手册里的某条是对的，但这次 trajectory 发现了更好的做法
   - 比如：手册说"找 apple 可以一个一个房间搜"，但这次发现直接去 kitchen 概率最高
   - 动作：**优化**现有的那条

**失败的 trajectory**：

3. **SKILL DEFECT**（手册写错了/不全/含糊）
   - 失败确实是因为手册某条有问题
   - 比如：手册说"把东西放进 drawer"，但没说要先确认 drawer 是空的
   - 动作：**修正**那条有问题的内容

4. **EXECUTION LAPSE**（机器人自己搞砸了）
   - 手册是对的，是机器人执行时没按手册来
   - 比如：手册明确写了"先开门再放冰箱"，但机器人就是没开门
   - 动作：**不改手册内容**，只在附录里加一条提醒"执行时特别注意这条规则"

---

## 为什么要把 skill 拆成 body 和 appendix

这是这 paper 最聪明的设计。

想想看，如果是 execution lapse 的情况——手册是对的，机器人没执行好——你怎么办？

如果像以前那样重写整个手册，你可能会：
- 把"先开门再放冰箱"这条再写一遍（冗余）
- 重写时措辞微妙变化，反而不如原来准确（corruption）
- 加了一堆强调性的废话（dilution）

EmbodiSkill 的做法：**手册主体（body）不动**，在附录（appendix）里加一条提醒："注意，执行 §3.2 这条时要特别小心，上次就是因为没按这条来才失败的。"

附录里**不引入新规则**，只是把主体里已经存在的、但机器人容易忽略的 valid content 拎出来 highlight 一下。

这就像你给学生发复习资料，教材正文（body）保持稳定，但在旁边贴一张便利贴（appendix）："这个考点容易忘，考试时注意。"

这样既保护了已经验证过的 valid content 不被乱改，又让机器人下次执行时对容易出错的点更敏感。

---

## 整个流程怎么转

把它想象成一个循环：

1. 机器人拿当前手册去做任务，产生一条 trajectory
2. Reflection：拿 trajectory 和手册对比，判断属于哪种情况（4 种 type）
3. 攒够一批 reflection 后，先 consolidation（整合去重、解决冲突）
4. 改手册 body（只改 DISCOVERY / OPTIMIZATION / SKILL DEFECT 指向的部分）
5. 改手册 appendix（只处理 EXECUTION LAPSE 的提醒）
6. 用新手册继续做下一个任务
7. 循环往复

这个循环不是原地打转，是螺旋上升。手册越来越完整、越来越准确，附录让容易犯错的地方越来越显眼。每次做任务都让手册更好一点，更好的手册又让下次任务做得更好，做得更好的任务又能提供更高质量的 reflection。

---

## 为什么这个方法 work

核心就一点：**它把 trajectory signal 做了 attribution。**

以前的失败改写方法是：
- trajectory → "这次失败了，整体 feedback：要加强物体搜索和容器状态检查" → 重写整个 skill

这就像你考试考砸了，老师不告诉你哪道题错哪了，只说"你下次要更认真"——你根本不知道是该复习哪个知识点。

EmbodiSkill 的做法：
- trajectory → "这次失败是因为你没按 §3.2 来（execution lapse）" 或 "这次失败是因为 §5.1 写得不够清楚（skill defect）" → 针对 specific 部分做 specific 动作

这就像老师拿红笔把你错的每道题圈出来，告诉你"这题是公式记错了，那题是计算粗心了"——你知道该改公式还是改做题习惯。

---

## 实验结果说明啥

数字层面最有意思的几点：

**1. 27B 的 Qwen3.5 + evolved skill 打败 GPT-5.2 直接做 agent 31.58%**

一个本地开源的 27B 小模型，配上自己 evolve 出来的 skill，比 OpenAI 最大的 frontier 模型直接做 agent 都强。这说明 skill（外部化的程序知识）能 compensate executor 的能力差距。

**2. Ablation 里的 skill-aware vs skill-unaware 差 14.92%**

同样都 evolve skill，同样都从 trajectory 学，区别只是"有没有区分 4 种 type + 分离 body/appendix"，就差 15%。这是这 paper 最重要的 single number，证明 skill-awareness 本身就是核心贡献，不是 skill evolution 带来的。

**3. Puttwo 从 52.94% 飙到 100%**

Puttwo 是要放两个物体到两个地方，multi-step 任务。这种需要 object search + state tracking + action ordering 的复杂任务，正是 evolved procedural knowledge 最能发挥作用的地方。

**4. Evolution 曲线快速上升然后稳定**

EmbodiSkill 从 static skill 一路爬升到 93.28%，然后稳定在高区间。skill-unaware 也 improve，但收敛在更低区间且波动大。波动大说明 coarse rewrite 在反复 corrupt valid content 再 recover，不稳定。

---

## 几个我自己的质疑

1. **$K=1$（每条 trajectory 最多 1 条 reflection）会不会太保守**？一整条 trajectory 可能包含多个有用的 evidence point，只抓 1 个是不是浪费了大量 signal？

2. **Skill evolution model 用 GPT-5.2，executor 用 Qwen3.5-27B**——这是"强模型当老师，弱模型当学生"。如果反过来呢？弱模型自己 reflect 能不能 evolve 出好的 skill？这关系到能不能完全 self-evolve，不依赖外部 strong model。

3. **Static skill 已经提升 19.51% relative**——说明初始 skill 本身就很有价值。那 evolution 的边际收益（再涨 27.5% relative）虽然大，但有多少是 evolution 带来的，有多少是 consolidation + structured representation 带来的，还能再拆。

4. **没有 cross-environment transfer 实验**。在见过的 layout 上 skill 越来越好，换到没见过的环境呢？这是从 benchmark SOTA 到 real-world embodied agent 的关键 gap。

5. **Appendix 会不会越堆越长**？虽然 paper 说会 merge duplicate 和 remove obsolete，但没给 appendix size 的 growth curve。如果 append 到几百条，executor 反而会被噪声淹没。

6. **Heat subtask 只有 73.91%**，是所有 subtask 里最差的。可能 microwave 的操作细节（开关门、加热时长、食物状态）很难从 trajectory cleanly distill 出来——某些高度 specific 的 procedural detail 有 ceiling。

---

## 一句话总结

传统方法是"失败了就反思，反思完重写整个手册"——问题是 embodied 环境里失败原因很杂，一股脑重写会把好的内容改坏。

EmbodiSkill 的做法是"先对照手册看 trajectory 到底说明了啥问题，分类处理"：
- 手册没写的 → 加
- 手册能更好的 → 优化
- 手册写错的 → 修正
- 手册对但没执行的 → 不动手册，加附录提醒

然后只改该改的地方，不该动的绝对不动。这样手册才能越用越准，越用越稳。

本质上这就是给 reflection 加了个 **type system**——强制 reflection signal 明确声明"我要对手册做什么操作"，避免 model 自由发挥把好东西改坏了。

---

# EmbodiSkill: Skill-Aware Reflection for Self-Evolving Embodied Agents 深度解析

Andrej，这是一篇挺有意思的工作， tackles的核心问题是：embodied agent的skill self-evolution在面对failure trajectory时如何区分"skill本身有缺陷"和"agent执行不到位"这两种本质不同的失败原因。下面尽量详细地讲。

参考链接：
- Paper GitHub: https://github.com/air-embodied-brain/EmbodiSkill
- ALFWorld benchmark: https://alfworld.github.io/
- EmbodiedBench: https://embodiedbench.github.io/
- Voyager (前序skill library工作): https://arxiv.org/abs/2305.16291
- Reflexion: https://arxiv.org/abs/2303.11366

---

## 1. 问题动机：为什么coarse skill update在embodied环境会崩

### 1.1 现有skill self-evolution范式的痛点

Voyager, Trace2Skill, EvoSkills, ExpeL这一类工作的paradigm是：trajectory → summary feedback → 整个skill重写。这个loop在digital environment（如Minecraft）里work得不错，因为execution是deterministic的——失败基本意味着skill有问题。

但embodied environment的关键差异在于：trajectory的outcome被perception、spatial grounding、object state、action precondition、execution reliability五个因素entangled。一个failed trajectory可能源于：
- skill内容确实错了（skill defect）
- skill没问题但agent没follow（execution lapse）

如果对这两种情况都做coarse whole-skill rewrite，后果就是paper Figure 1里展示的：valid guidance被overwritten，redundant content堆积，incorrect prescription被保留。skill会越evolve越烂。

### 1.2 关键insight

paper的核心claim是：**trajectory signal和skill content之间需要explicit的attribution**——trajectory evidence要被分类到不同类型，针对skill的不同部分做targeted modification，而不是free-form rewrite整个skill。

这其实是把"reflection"从一个unstructured verbal feedback重新建模成一个structured signal assignment problem。

---

## 2. 方法：Skill-Aware Reflection + Evolution Spiral

### 2.1 Problem formulation细节

**Embodied trajectory**定义：

$$\tau = (I, o_1, a_1, \ldots, o_T, a_T, r) \tag{1}$$

变量解释：
- $I$: 自然语言task instruction
- $o_t$: 第$t$步的observation（embodied setting下可以是visual frame或symbolic observation）
- $a_t$: 第$t$步的action
- $T$: trajectory长度（变量名）
- $r \in \{0, 1\}$: 最终task success binary signal

**Skill的二元结构**——这是整个framework的设计核心：

$$S^{(n)} = (S_{\mathrm{body}}^{(n)}, S_{\mathrm{app}}^{(n)}) \tag{2}$$

- $S_{\mathrm{body}}^{(n)}$: 第$n$步evolution后的skill **body**——主要的prescriptive procedural content，包含prerequisites, subgoal ordering, object affordance, visual-search strategy, action precondition, recovery strategy等
- $S_{\mathrm{app}}^{(n)}$: skill **appendix**——只highlight body里已经存在的valid content，不引入新规则。这部分是设计上专门用来处理execution lapse的

**Skill-guided execution**：

$$a_t \sim \pi_\theta(\cdot \mid I, S^{(n)}, h_t) \tag{3}$$

- $\pi_\theta$: executor policy，参数$\theta$ frozen
- $h_t = (o_1, a_1, \ldots, o_t)$: within-trajectory history
- 所有improvement必须externalize到evolving skill里，而不是更新$\theta$

**Skill evolution objective**：

$$J(S; \pi_\theta) = \mathbb{E}_{I, \mathcal{E}}[r(\tau)], \quad \tau = \mathrm{Execute}(\pi_\theta, I, \mathcal{E}, S) \tag{5}$$

- $\mathcal{E}$: embodied environment distribution（包含layout, object state, visibility等variation）
- 优化对象是skill $S$，executor $\pi_\theta$固定——所以这是一个**black-box optimization over skill text**的问题

这里我想强调一下：把skill evolution形式化为$J(S;\pi_\theta)$的期望，本质上是在说"我在一个frozen executor + 一个externalized text-skill上做优化"，这个formulation其实把问题简化成了prompt optimization in the wild——只是优化对象是结构化skill而不是单一prompt。

### 2.2 Skill-Aware Reflection：四种reflection type

这是paper最关键的设计。给定trajectory $\tau$和当前skill $S^{(n)}$，skill evolution model $F$输出最多$K$条reflection records：

$$\mathcal{R}_\tau = F(\tau, S^{(n)}, K) = \{\rho_i\}_{i=1}^{m_\tau}, \quad 0 \leq m_\tau \leq K \tag{6}$$

- $F$: skill evolution model（论文里用GPT-5.2或Gemini-3-flash）
- $K$: 单条trajectory最多reflection数，实验中$K=1$
- $m_\tau = 0$ 表示trajectory没提供可靠evidence——这是设计上"宁缺毋滥"的机制

每条reflection record $\rho_i$包含：
- $c_i$: reflection type
- $e_i$: trajectory evidence
- $d_i$: update directive
- $b_i$: target skill content（指向$S_{\mathrm{body}}^{(n)}$的具体部分，DISCOVERY除外）

四种type按trajectory outcome分流：

**成功trajectory ($r=1$)**：
$$c_i \in \{\mathrm{DISCOVERY}, \mathrm{OPTIMIZATION}\} \tag{7a}$$

- **DISCOVERY**: trajectory揭示了当前skill body没覆盖的有用新内容。不需要target $b_i$，因为是要add而非modify。directive $d_i$指定要考虑的新skill content
- **OPTIMIZATION**: 现有target skill content $b_i$ valid，但trajectory提示了更好的执行方式。必须提供revised version

**失败trajectory ($r=0$)**：
$$c_i \in \{\mathrm{SKILL\,DEFECT}, \mathrm{EXECUTION\,LAPSE}\} \tag{7b}$$

- **SKILL DEFECT**: 现有target skill content $b_i$ incorrect/incomplete/underspecified。必须给出corrected skill content
- **EXECUTION LAPSE**: skill valid但agent没follow。directive $d_i$ **不是**body-level revision，而是产生appendix content提醒executor

这里的key insight是：**成功trajectory也能触发reflection**（DISCOVERY/OPTIMIZATION）。这意味着reflection signal不局限于failure——成功执行揭示新的procedural pattern同样重要。这点跟Reflexion、ExpeL这种"只在failure时反思"的paradigm有本质区别。

### 2.3 Skill Revision：分离body和appendix

Reflection buffer达到revision interval $B$后，先按type partition：

$$(\mathcal{R}_{\mathrm{disc}}, \mathcal{R}_{\mathrm{opt}}, \mathcal{R}_{\mathrm{def}}, \mathcal{R}_{\mathrm{lap}}) = \mathrm{PARTITION\,BY\,TYPE}(\mathcal{R}) \tag{8}$$

**Consolidation**（关键步骤）：

$$\widetilde{\mathcal{R}}_{\mathrm{rev}} = F(S_{\mathrm{body}}^{(n)}, \mathcal{R}_{\mathrm{disc}}, \mathcal{R}_{\mathrm{opt}}, \mathcal{R}_{\mathrm{def}}) \tag{9}$$

注意：只有DISCOVERY/OPTIMIZATION/SKILL DEFECT这三种body-level reflection进入consolidation。EXECUTION LAPSE被排除。

Consolidation做的事：
- 去除冗余reflection
- 合并overlapping suggestion
- 按$b_i$（target skill content）分组
- 解决conflict——如果conflict不可靠地resolve，要么reassign type，要么discard，**绝不强制不确定的改动**

这一步很关键——它把多条local reflection聚合成consistent的revision signal set，避免了"逐条改skill"导致的 inconsistency累积。

**Skill body revision**：

$$S_{\mathrm{body}}^{(n+1)} = F(S_{\mathrm{body}}^{(n)}, \widetilde{\mathcal{R}}_{\mathrm{rev}}) \tag{10}$$

这里$F$被用作**constrained editor**：
- DISCOVERY → add新content
- OPTIMIZATION/SKILL DEFECT → modify对应$b_i$
- 未被$\widetilde{\mathcal{R}}_{\mathrm{rev}}$implicate的content **保持不变**（substantively）
- 允许有限的consistency edits（去redundancy、normalize format、解决local conflict）

**Skill appendix update**（在body revision之后）：

$$S_{\mathrm{app}}^{(n+1)} = F(S_{\mathrm{body}}^{(n+1)}, S_{\mathrm{app}}^{(n)}, \mathcal{R}_{\mathrm{lap}}) \tag{11}$$

- 不引入/删除/重写body rules
- 把execution-lapse evidence组织成anchored到updated body的appendix items
- 可以merge duplicate appendix items、remove obsolete items（对应body已经变化的）、incorporate新的execution-lapse reflection

最终skill：

$$S^{(n+1)} = (S_{\mathrm{body}}^{(n+1)}, S_{\mathrm{app}}^{(n+1)}) \tag{12}$$

### 2.4 Skill-Aware Evolution Spiral

这就是一个closed loop：
1. $S^{(n)}$ → executor执行task → 产生trajectory $\tau$
2. $\tau$ → skill-aware reflection → 生成revision signal
3. Revision signal → 更新$S_{\mathrm{body}}$和$S_{\mathrm{app}}$
4. $S^{(n+1)}$ → 指导下一轮task execution → 产生新trajectory

paper管这个叫"spiral"而不是"loop"，因为skill是被progressively accumulated的——body变complete + accurate，appendix让valid content变salient。

### 2.5 Algorithm 1伪代码解读

```
Input: S^(0), π_θ, F, task stream I, revision interval B, max reflections K
Output: evolved skill S

(S_body, S_app) ← S^(0)
S ← (S_body, S_app)
R ← ∅  // reflection buffer

foreach task instruction I ∈ I do
    τ ← Execute(π_θ, I, S)
    R_τ ← SkillAwareReflect(τ, S, F, K)
    if R_τ ≠ ∅ then
        R ← R ∪ R_τ
    if |R| ≥ B then
        (R_disc, R_opt, R_def, R_lap) ← PartitionByType(R)
        R̃_rev ← ConsolidateRevisions(F, S_body, R_disc, R_opt, R_def)
        S_body ← ReviseSkillBody(F, S_body, R̃_rev)  // targeted body edits
        S_app ← UpdateSkillAppendix(F, S_body, S_app, R_lap)  // update appendix only
        S ← (S_body, S_app)
        R ← ∅

return S
```

关键设计点：
- 每条trajectory最多$K=1$条reflection——限制signal noise
- Buffer达到$B$才触发revision——batch consolidation避免单条reflection引发冲动修改
- Partition后body和appendix走不同路径——避免execution lapse污染body

---

## 3. 实验：数字细节

### 3.1 ALFWorld主结果（Table 1）

ALFWorld有6类subtask：Put, Clean, Heat, Cool, Examine, Puttwo。训练3553 tasks，测试134 tasks。

| 配置 | Overall | Put | Clean | Heat | Cool | Examine | Puttwo |
|---|---|---|---|---|---|---|---|
| GPT-5.2 direct | 70.89 | 87.50 | 67.74 | 56.52 | 76.19 | 83.33 | 52.94 |
| Gemini-3-flash direct | 82.09 | 91.67 | 83.87 | 65.22 | 85.71 | 83.33 | 82.35 |
| Qwen3.5-27B + G-Memory | 74.62 | 62.50 | 77.42 | 56.52 | 85.71 | 72.22 | 47.06 |
| **EmbodiSkill (Qwen3.5-27B + GPT-5.2)** | **93.28** | 95.83 | 96.77 | 73.91 | 95.24 | **100.00** | **100.00** |
| EmbodiSkill (Qwen3.5-27B + Gemini-3-flash) | 87.31 | 95.83 | 93.55 | 69.57 | 95.24 | 83.33 | 82.35 |

几个有意思的点：
1. **27B Qwen3.5 + evolved skill > GPT-5.2直接agent 31.58%**——smaller open-weight model + externalized skill超过了frontier closed model直接使用
2. **Puttwo从52.94%（G-Memory）→ 100%（EmbodiSkill）**——这个task需要object search + state tracking + action ordering，evolved skill在这种multi-step household task上优势明显
3. **Heat subtask仍然最低（73.91%）**——可能是microwave的precondition（要先open/close门）+ heat时长这类细节skill evolve还没充分覆盖

### 3.2 EmbodiedBench结果（Table 2）

EB-Habitat（visual object interaction in 3D）和EB-Navigation（visual navigation）。

EB-Habitat上：Qwen3-VL-32B + Gemini-3-flash → 52.33% avg（最强memory baseline G-Memory 45.00%，最强closed-source direct Gemini 46.00%）

EB-Navigation上：Qwen3-VL-32B + GPT/Gemini → 61.33% avg（比最强memory baseline高17.94%，比最强direct agent高6.98%）

特别注意EB-Navigation的Long子集——这是long-horizon navigation，所有memory baseline基本在Long上都很差（5-23%），但EmbodiSkill能到32-33%。这说明evolved skill对long-horizon procedural knowledge的累积尤其有效。

### 3.3 Ablation（Table 3）——核心证据

四种配置：
- **No skill**: 直接执行
- **Static skill**: 初始skill，不evolve
- **Skill-unaware**: 从trajectory更新skill，但coarsely rewrite，不区分revision type和body/appendix
- **EmbodiSkill**: 完整skill-aware reflection + revision

以Qwen3.5-27B + GPT-5.2为例：

| 配置 | 成功率 | 相对前一阶段 |
|---|---|---|
| No skill | 61.19% | baseline |
| Static skill | 73.13% | +19.51% relative |
| Skill-unaware evolution | 78.36% | +7.16% relative |
| **EmbodiSkill** | **93.28%** | **+19.04% relative over skill-unaware** |

$\Delta_{\mathrm{aware}}$ = +14.92%，意味着**skill-awareness（区分四种reflection type + 分离body/appendix）单独贡献了14.92%的绝对提升**——这是从skill-unaware到skill-aware的纯增量。

这个数字是paper最有说服力的single number：**同样有skill evolution，做不做skill-aware attribution，效果差15%**。

### 3.4 Evolution curve（Figure 3）

paper还展示了10个revision stage的test success curve：
- EmbodiSkill从static skill 73.13%快速爬升到93.28%，之后稳定在高区间
- Skill-unaware evolution也improve但converge到lower range，且**fluctuation更大**

这条曲线的形状是典型的good optimization——快速上升然后plateau。fluctuation小说明revision是reliable的，不是noisy的。skill-unaware的不稳定正好印证了前面说的"coarse rewrite会corrupt valid content"。

---

## 4. 关键设计直觉

### 4.1 为什么需要appendix这个独立component

这是paper最subtle的设计。考虑这个场景：skill body正确写了"检查容器是否open后再place object"，但executor在某次trajectory里没检查直接place——这是execution lapse。

如果按skill-unaware的做法，model可能从这次failure总结出"skill应该强调检查container state"，然后改写skill body——但body本来就写了这条rule，rewrite可能：
- 加入redundant表述
- 改写时subtly改变原意
- overwrite原本的精确表述

EmbodiSkill的做法：识别为EXECUTION LAPSE，**body不动**，只在appendix里加一条"Reminder: must verify container open state before placing (see skill body §X)"。这样valid content被保留并salient highlight，没有corruption风险。

这本质上是把"hard rule"（body）和"soft attention"（appendix）做了architectural separation，类似training里hard parameter vs soft attention的区别。

### 4.2 为什么DISCOVERY/OPTIMIZATION也要做reflection

传统reflection只在failure时触发。但embodied task里，成功的trajectory同样包含procedural pattern——比如agent找到了某个object在特定layout下的efficient search顺序，或者用了一个novel action sequence。

DISCOVERY和OPTIMIZATION让success trajectory也能贡献到skill evolution。这把reflection从"failure-driven"扩展到"evidence-driven"——任何reliable evidence都该被capture。

### 4.3 为什么需要consolidation这一步

如果每条reflection都直接改skill body，会有几个问题：
- 多条reflection可能针对同一$b_i$，直接逐条改会冲突
- 冗余的similar reflection堆积
- 早期reflection可能和后期reflection矛盾

Consolidation（公式9）在batch层面解决这些conflict，让进入skill body的revision signal是consistent的。同时不可靠的conflict会被discard而不是forced apply——这是一个"保守优先"的设计，宁可skip也不corrupt。

### 4.4 整体类比：这是什么类型的优化

把整个framework抽象看：
- $S$是optimization variable（text-valued）
- $\pi_\theta$是fixed evaluator
- $J(S;\pi_\theta)$是objective
- $F$既做reflection（梯度估计）又做revision（parameter update）

本质上是一个**LLM-as-optimizer的text-space black-box optimization**，类似Prompt Optimization with LLM-as-judge那类工作，但加了：
1. 结构化variable（body + appendix）
2. 结构化update signal（4种type + target reference）
3. Batch consolidation

这跟STaR、Self-Refine、PromptBreeder这类方法在哲学上同源——把LLM当作optimizer在prompt/skill text上做search。但EmbodiSkill的特殊之处是**对embodied setting下的signal-attribution problem做了explicit modeling**。

---

## 5. 一些可以质疑的点

为了build your intuition，我也提几个值得思考的地方：

1. **$K=1$的reflection budget是不是太保守**？每条trajectory最多1条reflection意味着大量trajectory的secondary evidence被丢弃。这可能是为了noise control，但也可能限制了skill evolution的data efficiency。

2. **Skill evolution model $F$用GPT-5.2，executor用Qwen3.5-27B**——这其实是"strong model as teacher, weak model as student"的asymmetric setup。如果swap（weak model做reflection，strong model做executor），效果会怎样？这关系到reflection quality vs execution quality的相对重要性。

3. **Static skill已经有19.51% relative improvement**——这个数字其实非常大。意味着initial skill本身设计就很有价值，evolution的边际收益（73.13% → 93.28%，约27.5% relative）虽然大，但不如static-vs-no-skill那么戏剧化。Skill的prior value vs evolution value的相对贡献值得更细的decomposition。

4. **Heat subtask的73.91%**——这是EmbodiSkill表现最差的subtask。可能microwave的specific procedural knowledge（如多次开关门、特定时长）很难从trajectory中cleanly distill。这暗示skill evolution对某些highly specific procedural detail有ceiling。

5. **没有cross-environment generalization实验**——所有结果都是same-distribution的train-test split。Evolved skill在new layout / new object set上transfer如何？这是从"benchmark performance"到"real-world embodied agent"的关键gap。

6. **Appendix的boundedness**——随着execution lapse累积，appendix会不会无限膨胀？paper提到可以merge duplicate和remove obsolete，但没有quantify appendix size随evolution的growth curve。如果appendix膨胀到几百条，executor的attention会被dilute。

---

## 6. 与相关工作的定位

paper在Section 2里把这工作定位在两个轴线交叉处：
- **Memory-based methods**（Reflexion, ExpeL, Mem0, G-Memory, A-MEM, LangMem）：trajectory-level reuse
- **Skill self-evolution**（Voyager, Trace2Skill, EvoSkills, SkillRL, MemSkill, XSkill）：procedural knowledge evolution

EmbodiSkill的position是：**skill-level procedural guidance + embodied-aware signal attribution**。它从skill evolution那边继承了persistent + revisable procedural representation，从memory那边借鉴了trajectory-level evidence utilization，但加入了"skill-aware reflection"这个新维度——把trajectory evidence按对skill的影响类型分类，做targeted modification。

参考链接：
- Voyager: https://arxiv.org/abs/2305.16291
- Reflexion: https://arxiv.org/abs/2303.11366
- ExpeL: https://arxiv.org/abs/2308.10144
- Mem0: https://arxiv.org/abs/2504.19413
- G-Memory (NeurIPS 2025): https://arxiv.org/abs/2506.13637
- SayCan: https://arxiv.org/abs/2204.01691
- Code as Policies: https://arxiv.org/abs/2209.07753
- ProgPrompt: https://arxiv.org/abs/2209.11353

---

## 7. 总结性直觉

如果让我用一句话提炼这paper的insight：

> **Embodied skill evolution的失败模式不是"信息不够"，而是"信号attribution错误"——把execution noise误判为skill defect会corrupt valid knowledge，把skill defect误判为execution noise会保留错误rule。**

EmbodiSkill的解法本质上是给reflection加了一个**structured type system**：四种reflection type对应skill的不同动作（add/optimize/correct/preserve-emphasize），强制reflection signal declare自己要做什么，而不是让model自由rewrite。

这跟程序语言设计里的type system philosophy很像——不限制表达能力，但强制commit到某种semantic commitment，从而避免ambiguous operation。

Andrej，希望这个build了你的intuition。如果你对appendix vs body的分离设计、或者skill evolution model和executor的asymmetric setup有更深的疑问，可以继续聊。
