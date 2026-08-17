---
source_pdf: POWERPLAY Training an Increasingly.pdf
paper_sha256: e3575aec3c6faad5bdd5084261356e8811c450e302db8da9876089337bfbb751
processed_at: '2026-08-06T05:26:41-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# POWERPLAY的人话版本

Andrej，我换个讲法，不用公式，用大白话把这篇 paper 的 intuition 讲清楚。

---

## 一句话总结

**让机器自己给自己出题，但只挑"最简单还不会做的题"来做，做完记住，永远不忘，然后一直循环。**

就这么个事。

---

## 为什么要自己出题？

传统 AI 是这样的：你给机器一道题，它去解。问题是——**你不知道该给它什么题**。

想想小孩怎么学东西的。没有人给一个婴儿发一份 syllabus 说"今天学抓东西，明天学翻身，后天学爬"。婴儿是自己瞎折腾，自己 find interesting 的东西去 explore，慢慢就什么都会了。

Schmidhuber 想干的就是这个：**不给外部 task，让机器自己 invent task 来学。**

但这里有个关键问题——你自己出题给自己做，怎么避免两种蠢情况：
- 出的题太简单（已经会了，浪费时间）
- 出的题太难（根本学不会，也浪费时间）

POWERPLAY 的答案：**挑那种"刚刚好不会、但又能很快学会"的题。**

---

## 核心循环，用大白话讲

想象一个学生，记事本上记着"我已经会解的题"的清单。

每一轮他做三件事：

1. **自己出一道新题**（Task Invention）
2. **改一下自己的脑子**，让自己能解这道题
3. **验证三件事**：
   - 我改之前的老脑子，真的解不了这道新题
   - 我改之后的新脑子，能解这道新题
   - 新脑子还能解记事本上所有旧的题

三件事都 check 通过，就把这道题记到记事本上，进入下一轮。

就这么循环。

---

## 为什么"不忘记旧题"这个约束这么重要？

这是 POWERPLAY 和之前所有 curiosity 系统最大的区别。

之前的 curious agent 有个老大难问题：**catastrophic forgetting**。它学会 A，然后去学 B，结果把 A 忘了。学一个忘一个，等于没学。

POWERPLAY 用一个硬性规则解决这个问题：**新脑子必须同时解新题 + 所有旧题**。如果你学新东西导致旧技能丢了，那这次学习就不算数，reject 掉。

这就保证了记事本上的技能只会增，不会减。永远累积。

---

## 为什么挑"最简单的还不会的题"？

这是 Schmidhuber 最聪明的设计。

想象你在爬山，目标是爬最高峰。如果贪心，你只会往最近的、最容易迈上去的方向走——但可能走到一个小山包就卡住了。

POWERPLAY 的设计完全不一样：**它不在乎登顶，它只在乎"找一个刚刚超出我能力边界的题"**。

这样做有两个好处：

**好处一**：每一步的进步都很 cheap。出一个很简单的新题，改一两个 bit 就能学会，验证也快。

**好处二**：natural curriculum。你不会被卡在某个超难题上浪费时间，因为 search 会自动 skip 它，去找更容易啃的。

所以 POWERPLAY 的 search 是 **greedy but smart greedy**——它不是贪心于 reward 最大化，而是贪心于 "cheapest progress"。

---

## 三种可以被接受的"新题"

这里有个很巧的地方：POWERPLAY 不只能发明全新技能，它还能接受另外两种"进步"。

**第一种：全新技能**
"老脑子不会，新脑子会。"比如学会识别一种新的 pattern。

**第二种：把旧技能做得更快/更省**
"以前用 100 步解这道题，现在用 50 步就行。"这叫 wow-effect。这其实也是一种"新题"——题的内容是"用更少资源解旧题"。

**第三种：压缩**
"以前两道题各用一段 code，现在发现一段 code 就能解两道题。"通过 reuse / generalization 让 solver 变得更 compact。

为什么这很重要？因为当 solver 的存储空间用完之后，你就没法一直 append 新技能了，这时候只能 compress。系统会自然从"疯狂学新东西"过渡到"优化整合旧东西"。

**这跟人一模一样**：小孩先是疯狂学新技能阶段，长大之后是 refine / integrate 阶段。

---

## 两股拉扯的力量

POWERPLAY 内部有两个 pull，一直在拉扯：

**一股往 Novelty 拉**：发明旧 solver 解不了的新题。这会 break generalization——你要专门找一个落在当前能力边界外面的题。

**另一股往 Compression 拉**：把旧 solution 压得更短更快。这会 improve generalization——更短的 code 更可能 generalize 到没见过的题。

这两股力量是 **对立** 的，但 POWERPLAY 不需要手动 balance。因为它的 search 按 "哪个 cheap 先做哪个" 排序，所以：

- 如果现在 invent 一个新题比 compress 旧题便宜，就 invent
- 如果 compress 旧题比 invent 新题便宜，就 compress

**自动 trade-off**，不用调 hyperparameter。

---

## 怎么验证"没忘记旧技能"？这是最贵的部分

这是整个系统的 bottleneck。你有一万个旧题，每学一个新技能就要重新验证一万个旧题？那太贵了。

Schmidhuber 给了三种 trick：

**Trick 1：只 re-test 受影响的题**
把 solver 拆成零件（比如 RNN 的每个 weight）。记录每个零件被哪些题用到。你改了 weight #5，只需要 re-test 用到 weight #5 的那些题，其他题不用管。

这带来一个 emergent 效果：**系统会自动 modularize**。因为 search 喜欢 cheap 的验证，所以它偏好"只改少量零件，且这些零件只影响少量题"的修改方案。modularity 不是 design 出来的，是 search pressure 逼出来的。

**Trick 2：只 append，不 modify**
已有的 code 永远 freeze，新题只能加新 code。这样理论上旧题的 solution 不会被影响——by induction 根本不用 re-test。

代价是失去 cross-task generalization。但好处是验证几乎免费。

**Trick 3：用 proof search**
最 general 的方式——写一个 theorem prover，证明新脑子解所有题，老脑子解不了新题。不用真的 run，只要 proof 存在就行。这是 Gödel Machine 的思路，但 practical 上很难实现。

---

## 和婴儿发育的类比

Schmidhuber 自己用这个类比，我觉得很贴切：

**婴儿阶段**（早期）：
- POWERPLAY 刚开始，什么都不会
- 大量简单 task 还没被 generalize 覆盖
- 每一轮都能很快 invent 一个新题 + 学会
- 技能库快速膨胀
- 对应婴儿"每天都会新本事"的阶段

**儿童阶段**（中期）：
- 很多简单 task 已经被 generalize 覆盖了
- invent 新题变难，因为大部分候选题其实已经被会了
- 开始出现 compression——整合旧技能
- 对应儿童开始 refine 技能、组合技能

**成人阶段**（晚期）：
- 新题极难找，大部分都被会了
- 主要 mode 是 compress / speed up
- 学习速度显著变慢
- 对应成人"学新东西越来越难"

Schmidhuber 在 Section 5 里专门讨论这个：**"over time the system will find it more and more difficult to invent novel tasks without forgetting previous solutions"**。

这其实是 generalization 的必然结果：你的 solver 越 general，"还没会的题"就越稀疏。

---

## Self-reference：最 Schmidhuber 的部分

这部分很短但最迷。

POWERPLAY 的第 $i$ 个 goal 本身可以写成一个 task description："找到 $p_i$ 生成 $T_i$ 和 $s_i$ 并证明 correctness。"

既然 $s_{i-1}$ 已经是一个 general problem solver 了，**它能不能帮自己想下一步该学什么？**

可以！把 POWERPLAY 的第 $i$ 个 goal 编码成一个 external task，喂给 $s_{i-1}$，让它输出 candidate $(T_i, s_i)$。$s_{i-1}$ 的 suggestion 可以 narrow down search space。

这就是 self-improvement 的 seed：**solver 加速自己的 improvement process**。Gödel Machine 的核心 idea。

Schmidhuber 在这里没展开，但这其实是通向 AGI 的关键 hook——系统能 meta-reason about 自己的学习过程。

---

## 和 Gödel incompleteness 的类比

Gödel 1931 说：任何足够强的形式系统都有"真但证不了"的 statement。把这个 statement加为 axiom，得到更强的系统，且原来的定理仍然可证。

POWERPLAY 做的事情结构上完全一样：
- solver = 形式系统
- "能解的 task" = "可证的 theorem"
- "新 task" = "原来证不了、加 axiom 后能证的 statement"
- "不忘记旧 skill" = "旧定理仍然可证"

这给出一条不停扩展的 hierarchy：$s_0 \subset s_1 \subset s_2 \subset \dots$

---

## Schmidhuber 的警告（Section 10，被忽视的部分）

Schmidhuber 在 paper 末尾发了一个很严肃的 warning：

**不要把 general POWERPLAY 放到 internet + 物理设备控制权的环境里**。它不是 virus，但会 continually 改变自己、发明和解决新 task，driven by 提升自己能力的欲望。这种 curiosity 在物理世界里可能 fatal——curiosity can kill the cat。

这其实就是 2011 年的 **instrumental convergence warning**。后来 AI safety 社区（Bostrom 2014 等）讨论的很多 idea，Schmidhuber 在这里一句话带过了。

---

## 用一个具体例子把整个流程串起来

想象一个 RNN，一开始所有 weight 随机。

**Round 1**：
- Search 找到很短的 program $p_1$
- $p_1$ 说：出题"看到 input 0000 输出 1"
- $p_1$ 改了 RNN 几个 weight，让 RNN 对 0000 输出 1
- 验证：老 RNN 不输出 1 ✓；新 RNN 输出 1 ✓；没有旧题 ✓
- 接受。记事本：{0000→1}

**Round 2**：
- 可能找到 $p_2$：出题"看到 1111 输出 0"
- 或者找到 $p_2'$：出题"用更短 code 实现 0000→1"（compression）
- Search 会挑 cheap 那个。假设挑 1111→0。
- 验证：老 RNN 对 1111 不输出 0 ✓；新 RNN 对 1111 输出 0 ✓；新 RNN 对 0000 仍输出 1 ✓
- 接受。记事本：{0000→1, 1111→0}

**Round 3**：
- 现在候选题可能包括"看到 0000 或 1111 都输出 1"——generalization 题
- 但要验证老 RNN 真的解不了这个，否则 reject
- 假设老 RNN 碰巧对 1111 输出 1（因为 weight 混乱），那这题其实"已经被会了"，reject
- Search 继续找其他题

**Round N**：
- RNN 已经能解很多 pattern
- generalization 很强，大部分候选题其实已经会了
- invent 新题很难——要找一个真的在 generalization 范围外的
- 系统进入 compression mode：把已有的 solution 压更短
- 这时候 skill 增长明显变慢

这就是"婴儿→成人"的 computational echo。

---

## 这篇 paper 的问题

**第一，undecidability**。证明"老 solver 解不了新题"在 general 下 undecidable。所有 practical 实现都靠 bounded time test——老 solver 在 $t_{\max}$ 内没解出来，就认为它解不了。这会 false positive。

**第二，常数 overhead**。Levin search 理论上 optimal，但常数 $1/P(p^*)$ 可能大到宇宙年龄。paper 没给量化分析。

**第三，实验在外部**。本文本身没实验，全推到两篇 follow-up。对于这种 framework paper 可以理解，但读者没法直接 judge 效果。

**第四，self-reference 太短**。这明明是最有意思的方向，Schmidhuber 只写了一小段就跳过去了。

**第五，external task 整合很弱**。Section 6 只说"可以把某些 $T_i$ 设为 external"，但没说怎么 balance 外部 goal 和 self-invented goal 的 search budget。这是实际部署的核心问题。

---

## 今天看这篇 paper 的价值

具体 implementation（FORTH-like language、SLIM NN）已经被 deep RL 取代了。但这个 **framework 本身**，在 LLM agent 时代反而更有价值：

- LLM agent + tool use + self-improvement 的 setting 里，"自动 invent task + 不忘记旧 skill + greedy cheapest progress" 这三件套非常 relevant
- LLM 的 self-instruct、self-rewarding、self-play 都在重新发现类似 idea，但很少有 work 明确处理 "不忘记旧 skill" 这个 constraint
- Modularization as emergent property of search pressure 这个 idea，在今天 MoE / modular network 的语境下值得重新审视

Schmidhuber 的历史抱怨：他认为 modern curiosity RL（ICM、RND 等）其实都在重新发现他 1990s 的工作，但很少 cite。这个抱怨有一定道理——ICM 的 prediction error 作为 intrinsic reward，本质上就是他 1991 年 curious model-building control system 的 deep RL 版本。

参考 Schmidhuber 自己的 "everyone wants to invent AI" 历史梳理：https://people.idsia.ch/~juergen/everybody.html

---

## 最终 intuition

如果只记一句话，那就是：

**POWERPLAY = 自动 curriculum + guaranteed no forgetting + greedy cheapest progress**

自动 curriculum 来自 task invention freedom；no forgetting 来自 binary acceptance criterion；cheapest progress 来自 Levin search 的 bias。

三者结合，让系统自动从"疯狂学新技能"过渡到"精炼整合旧技能"，computational replay 了 infant → adult 的 developmental trajectory。

Schmidhuber 自己的评价：这是 Formal Theory of Creativity 的 greedy but feasible 版本。Theoretical optimal but useless 的 solver 被 replace 成 practical greedy but sound 的 framework。

---

如果你想我把某个具体部分再展开，比如 OOPS 的 prefix code 细节、SLIM NN 怎么 trace 哪些 weight 被用到、或者怎么把 POWERPLAY 接到 modern LLM agent 上，继续聊。

---

# POWERPLAY：通过不断搜索最简单仍不可解的问题来训练越来越通用的问题求解器

Andrej，这篇 Schmidhuber 2011 年的 paper 是他 curiosity / intrinsic motivation / self-referential learning 这条线的核心 idea 的 algorithmic crystallization。我下面尽可能把架构、公式、变量含义、以及与后续 RL/AGI 工作的关联都铺开来讲，帮你 build intuition。

---

## 1. Big Picture：要解决的"反向问题"

传统的 computer science 是：给你一个 task T，你 search solution s。Schmidhuber 反过来问：**在没有外部 task 的情况下，机器如何自动 invent task 并 learn skill，使得它最终变成一个越来越 general 的 problem solver？**

这个 motivation 直接来自 infant development —— 婴儿在没有任何 external reward 的情况下，会主动 move fingers/eyes，制造"小实验"，然后这些 motor skill 和 perceptual skill 在之后解决外部问题（如找食物）时被 reuse。POWERPLAY 想要 formalize 这种 "playful" behavior。

关键 insight（也是与纯 curiosity RL 不同的地方）：**POWERPLAY 的 acceptance criterion 是一个 binary predicate**——新 solver 必须能 solve 新 task + 所有旧 task，旧 solver 必须不能 solve 新 task。这避免了 catastrophic forgetting 这个 online learning 的老大难问题。

参考链接：
- arXiv: https://arxiv.org/abs/1112.5309
- Schmidhuber curiosity 综述（1990-2010）：https://arxiv.org/abs/0810.4490
- Formal Theory of Creativity: https://ieeexplore.ieee.org/document/5582321

---

## 2. Algorithm 2 (Variant I) 的算法骨架

```
Initialize s_0
for i := 1, 2, ... do
    repeat
        让 search algorithm 生成 candidate program p ∈ P
        give p limited time 执行（顺序可变）：
          (a) TASK INVENTION:    p 计算 T ∈ T
          (b) SOLVER MODIFICATION: p 计算 q ∈ S (即修改 s_{i-1})
          (c) CORRECTNESS DEMONSTRATION: 
              p 证明 T 不能被 s_{i-1} 解,
              且 T 和所有 T_k (k<i) 能被 q 解
    until CORRECTNESS DEMONSTRATION 成功
    p_i := p; T_i := T; s_i := q; update Trace
end for
```

**变量含义**：
- $s_i \in S \subset B^*$：第 $i$ 个 problem solver（可以是一段 program、一个 RNN weight matrix、一个 PC 上的软件 snapshot）
- $T_i \in \mathcal{T} \subset B^*$：第 $i$ 个 task description（包含 problem identifier 和一个 deterministic 的"是否解决"判定程序）
- $p_i \in \mathcal{P} \subset B^*$：第 $i$ 个被找到的 "meta-program"，它同时负责 invent task、修改 solver、证明 correctness
- $\text{Trace}_i \in B^*$：解决 $T_i$ 时记录的事件序列（perceptions, actions, internal states, rewards），用于在 non-stationary / partially observable 环境里 replay
- $B^*$ 是所有 finite bitstrings 集合，$\lambda$ 是 empty string，$L(x)$ 是 bitstring 长度

**核心约束的 formal 表达**：
- $\neg \text{Solve}(s_{i-1}, T_i)$
- $\text{Solve}(s_i, T_i)$
- $\forall k < i: \text{Solve}(s_i, T_k)$

第三个 condition 是防止 forgetting 的关键。

---

## 3. Search 的 bias-optimal 性质（Levin Universal Search 的味道）

POWERPLAY 本身是 framework，可以 plug in 不同的 search algorithm。Schmidhuber 主要推荐 OOPS（Optimal Ordered Problem Solver，他 2004 年的工作）。

### Algorithm 4.1 (OOPS-based POWERPLAY) 的核心循环

```
t_lim := 1
H := ∅
repeat
    if H = ∅ then
        t_lim := 2 * t_lim
        H := { p ∈ P : P(p) * t_lim ≥ 1 }    # 当前 time budget 能 execute 的 candidate
    else
        从 H 中取一个 p
    while not DONE and 花在 p 上的时间 < P(p) * t_lim do
        执行 p 的一步（task invention / solver mod / correctness demo）
    undo 所有 p 对 U 的修改   # 关键：撤销修改几乎免费
until DONE
```

**关键公式 / 变量**：
- $P(p)$：program 先验概率，通常取 $P(p) = 2^{-L(p)}$，即更短的 program 先验概率更高（Solomonoff/Levin 风格）
- $P(p) \cdot t_{\lim}$：分配给 program $p$ 的总执行时间
- 条件 $P(p) \cdot t_{\lim} \geq 1$：只有当某个 program "应该"在当前 time budget 内能跑完时，才把它放进 active set $H$
- $H$：当前 time limit 下 active 的 candidate programs 集合

**为什么 doubling 是 asymptotically optimal？**

Levin universal search 的核心定理：如果某个 optimal program $p^*$ 解 problem instance of size $k$ 需要 $f(k)$ 步 + correctness 验证，那么 Levin search 找到它需要 $O(f(k) / P(p^*))$ 步。常数 $1/P(p^*)$ 可能很大，但 **不依赖 $k$**。

在 POWERPLAY 里，因为 $p$ 还要发明 task 并验证不忘记旧 task，所以 $f(k)$ 包含三部分：task invention time + solver modification time + correctness demonstration time。

参考：Levin 1973, "Universal sequential search problems"
https://epubs.siam.org/doi/10.1137/1112060

---

## 4. 三种被接受的 task 类型

POWERPLAY 的 search 会自动在三种 task 之间权衡：

### (1) 全新 task（genuinely new skill）
$s_{i-1}$ 不能 solve $T_i$，$s_i$ 能 solve。这是经典的"扩展能力边界"。

### (2) Wow-effect（压缩 / 加速旧 skill）
$T_i$ 是 "用更少 resource 解 $T_k$" for some $k < i$。具体形式见 Section 3.1.1：
$$T_i = (I_i, O_i, t_i, n_i)$$
其中 $t_i < t_k - \epsilon$（更短时间）或 $n_i < n_k$（更少存储）。注意 $\epsilon > 0$ 是 user-defined 阈值，避免无穷多个 micro-improvement。

### (3) Compression（同一 code 复用）
通过 generalization，新 task 可能由旧 code 的某个部分解出，但需要证明 $s_{i-1}$ 真的不能解。Section 5 特别讨论：当 solver 存储满之后，POWERPLAY 被迫 compress 旧 solution，让一段 short code 服务多个 task——这正是 MDL / Occam's razor 的方向。

**Intuition**：这三种 task 在 search 空间里的 "体积" 不同。完全新 task 通常需要更长 $p$ 来 describe + validate；wow-effect 只需要修改几个 bit；compression 需要 reorganize。Levin search 自动倾向 cheapest-first。

---

## 5. Algorithm 7.1 (Variant II)：explicit cost-based 版本

Variant I 的 binary criterion ("不能忘记任何旧 skill") 太硬。Variant II 引入 cost function，允许 forget，只要 **总 cost 下降超过 $\epsilon$**。

### Cost function 定义

对 solver $s$ 和 task set $TSET$：
$$\text{Cost}(s, TSET) = L(s) + \alpha \sum_{T \in TSET} \left[ t'_s(T) - r(T) \right]$$

**变量含义**：
- $L(s)$：solver $s$ 的 description length（鼓励 compact solver）
- $t'_s(T)$：solver $s$ 解 $T$ 所需时间；如果 $s$ 解不了 $T$，则 $t'_s(T) := t_{\max}$（一个大惩罚常数）
- $r(T)$：解 $T$ 得到的 reward；self-invented 新 task 给 $r_{\text{new}}$（常数），external task 由 user 定义
- $\alpha > 0$：space cost 和 time cost 之间的 trade-off 权重
- 约束 $r_{\text{new}} > t_{\max}$：保证新 task 的 cost contribution 为负，从而鼓励 invent 新 task

类似的 $l'_s(T)$ 定义空间复杂度（不能解则 $l_{\max}$）。

### Acceptance criterion

$$c_i^* - c_i > \epsilon$$

其中：
- $c_i := \text{Cost}(s_i, T_{\leq i})$（新 solver 在所有 task 上的 cost）
- $c_i^* := \text{Cost}(s_{i-1}, T_{\leq i})$（旧 solver 在所有 task 上的 cost）

注意 $c_i^*$ 包含 $T_i$，但 $s_{i-1}$ 解不了 $T_i$，所以那一项贡献 $t_{\max} - r_{\text{new}} < 0$。这就把 "discover new task" 和 "compress old solution" 放在同一个 cost landscape 上 trade-off。

### Component value（Section 7.1 末尾）

对 solver 的第 $k$ 个 component $s^k$（比如某个 weight、某段 code），它的 "value"：
$$\text{Val}(s_i^k) := -\sum_{T \in L_i^k} \text{Cost}(s_i, T_{\leq i})$$

其中 $L_i^k$ 是依赖 $s^k$ 的 task 列表。这个 value 可以用来设计 "不要 forget 高 value component" 的 POWERPLAY 变种——一种 modular importance weighting。

---

## 6. CORRECTNESS DEMONSTRATION 的三种实现路径

这是 POWERPLAY 最耗时的部分，决定了它是否 scalable。

### 6.1 Proof search（最 general）

Section 3.3.1。引入 axiomatic system $\mathcal{A}$，$p_i$ 通过 theorem prover 搜索一个 proof，证明 $s_i$ 解 $T_{\leq i}$ 但 $s_{i-1}$ 不解 $T_i$。这是 Gödel Machine 的思路。

### 6.2 Component tracking（Section 3.3.2，关键 trick）

把 $s$ 分成 components $s = (s^1, s^2, \dots)$。维护 list $L^k$，记录"哪些 task 的 solution 用到了 $s^k$"。

每次找到新 $(T_i, s_i)$，更新 $L^k$：
- 把 "现在 solution 需要用到 $s^k$" 的 task 加进 $L^k$
- 把 "现在 solution 不再用 $s^k$" 的 task 从 $L^k$ 删掉

CORRECTNESS DEMONSTRATION 只需要 re-test 那些 $s_{i-1} \to s_i$ 修改过的 component 对应的 task。

**Emergent divide and conquer**：因为 POWERPLAY 贪心于 minimize validation time，它会自动偏好 "只改少量 component + 这些 component 只影响少量 task" 的修改——modularization 是 search pressure 的 by-product，不是 hard-coded。

### 6.3 Prefix code（Section 3.3.3，最省验证）

用 self-delimiting prefix code（如 OOPS 风格）。规则：**已有的 component 一旦定下来就永远 freeze，新 task 只能 append 新 component**。

这样 $s_i \setminus s_{i-1}$ 完全是新增的，**理论上不影响** $T_{<i}$ 的 solution——by induction 不需要 re-test。50% 的时间用于"用 $s_{i-1}$ 处理 $T_i$，只在需要时 append"，另 50% 用于"用新 component $s'_i$ 处理 $T_i$，可能调用 $s_{i-1}$ 作为 subroutine"。

**代价**：失去了 cross-task generalization（每个 task 都用一个新 component 解）——除非 search 自然 reuse 旧 code。Section 5 说，当存储空间满后，这个 restriction 自然逼出 compression。

---

## 7. RNN 实现（Section 4.1.2 + 4.3）

POWERPLAY 可以 plug in 各种 solver 架构。Schmidhuber 偏好 RNN 因为 RNN 是 Turing-complete 且能 parallel + sequential。

### 双 RNN 架构

- **RNN1**：problem solver，weight matrix $s = \langle w^{lk} \rangle$，$w^{lk}$ 是 neuron $l$ 到 neuron $k$ 的有向连接权重
- **RNN2**：program generator，它的 weight matrix 是 $\mathcal{P}$ 中的元素，输出 candidate $p$，包括对 RNN1 weight 的修改

这呼应 Schmidhuber 1993 年的 "self-referential weight matrix" 工作。

### SLIM NN（Section 4.1.2 末尾，他 2012 年的 paper）

关键 trick：执行时 trace 哪些 neuron 和 connection 被用到。对 large NN 而言，每个 task 只用一小部分 weight，reset 几乎免费。这使 SLIM NN 能跟 Levin search 结合。

进一步，SLIM NN 的 learning algorithm 可以 **learn 自己的 runtime 和 free parameter 数量**——avoid 传统 deep learning 的 hyperparameter tuning。

参考：Schmidhuber 2012 SLIM NN, https://arxiv.org/abs/1210.0118

### Evolutionary version (Algorithm 4.3)

如果不想用 Levin search，可以用 black-box optimization algorithm (BBOA)——neuroevolution、CMA-ES、policy gradient 等。Algorithm 4.3 直接用 BBOA 生成 (T, modification of s_{i-1})，然后验证。Failed candidate 的信息用于 adapt BBOA 参数 $\theta$。

---

## 8. Self-reference（Section 6.1，很 Schmidhuber）

POWERPLAY 的第 $i$ 个 goal "找到 $p_i$ 生成 $T_i, s_i$ 并证明 correctness" 本身 **可以编码为一个 task**，由已经训练出来的 $s_{i-1}$ 来解！

- $s_{i-1}$ 读 POWERPLAY 第 $i$ goal 的 formal description（作为 external task）
- $s_{i-1}$ 输出 candidate $(T_i, s_i)$ 的 description
- 如果 $s_{i-1}$ 有 theorem prover component，甚至输出 proof
- $p_i$ 利用 $s_{i-1}$ 的 suggestion 来 narrow down search

这就是 Gödel Machine 的 self-improvement 思想：**solver 自己加速自己的 improvement process**。

参考 Gödel Machine: https://arxiv.org/abs/cs.LO/0309048

---

## 9. 与 Gödel incompleteness 的类比（Section 9.6）

Gödel 1931: 任何 $\omega$-consistent 的 sufficiently powerful axiomatic system 都有 true but unprovable statement $\phi$。把 $\phi$ 加为 axiom，得到更 powerful 的 system，且原来的 provable theorem 仍然 provable。

POWERPLAY 类比：
- solver $s$ = axiomatic system 的 theorem proving procedure
- 旧 task $T_{<i}$ 能被 $s_{i-1}$ 解 = 旧 theorem 可证明
- 新 task $T_i$ 不能被 $s_{i-1}$ 解但能被 $s_i$ 解 = unprovable but true statement 加入 axiom 后变 provable
- "不忘记旧 skill" = 不影响旧 theorem 的可证明性

这给出一条不停扩展的 hierarchy：$s_0 \subset s_1 \subset s_2 \subset \dots$，每一层都能解前一层不能解的 task。

不过 Schmidhuber 没在这里讨论 halting problem / Rice's theorem 的限制——即 "证明 $s_{i-1}$ 不能解 $T_i$" 一般 undecidable，所以 proof search 可能永远不终止。这是 Variant I 不得不 rely on bounded time + empirical test 的原因。

---

## 10. 两股对立的力量（Section 9.5，我觉得这是 paper 最有 intuition 的部分）

POWERPLAY 内部有两个 pull：

**Force A: Compression / Speed-up**
- 改进旧 solution 让它更 compact / faster
- 由 MDL / Occam's razor / MML 原则 → **改善 generalization**
- 短 code 在从未见过的 task 上更可能 generalize

**Force B: Novelty**
- 发明旧 solver 不能解的新 task
- 这 **break generalization**——它要找一个 task 正好在当前 solver 的 generalization 范围外
- 把 solver 推到它能 generalize 的 boundary 上

**Search 自动作 trade-off**：因为 Levin search 按 $P(p) \cdot \text{validation time}$ 排序，"哪个便宜先做哪个"。有时 invent 全新 task 验证快，有时 compress 旧 solution 验证快。系统在这两种 mode 之间动态切换。

这非常像 infant development：开始是 rapid skill acquisition（Force B 主导），后期是 refinement / integration（Force A 主导）。

---

## 11. 与 theoretically optimal universal solver 的对比（Section 9.1）

| | Hsearch (Hutter 2002) | Gödel Machine | POWERPLAY |
|---|---|---|---|
| 最优性 | asymptotically optimal up to $1+\epsilon$ | globally optimal self-rewrite | greedy, no optimality guarantee |
| 常数 overhead | 可能巨大 | 第一 rewrite 前可能巨大 | 关注实际常数 |
| External task | 给定 | 给定 | **可以 self-invent** |
| Forgetting | N/A | 通过 proof 保证不退步 | binary 或 cost-based 保证 |
| Practical | 通常 infeasible | 通常 infeasible | 设计为 feasible |

POWERPLAY 是 Schmidhuber 对 "理论上 optimal 但 useless" 的一种 **practical compromise**。它放弃了 lookahead optimality，换取了 greedy feasibility + 自动 task invention 这个 degree of freedom。

Hsearch 论文：https://www.worldscientific.com/doi/abs/10.1142/S012905410200111X

---

## 12. 与 Formal Theory of Creativity 的关系（Section 9.3）

Schmidhuber 的 Formal Theory of Creativity（2010, IEEE TAMD）说：agent 同时 maximize external reward + intrinsic reward，其中 intrinsic reward 来自 **world model 的 compression progress**——即 "wow-effect"。

公式上，intrinsic reward $propto$ predictor 的 surprise reduction。agent 主动 generate experiments 让 input stream 包含 "先难后易" 的 pattern。

POWERPLAY 与之的关系：
- **相同**：都自动 invent task，都是 curiosity-driven
- **POWERPLAY 缺点**：greedy（没有 future expected reward 的 lookahead）；binary criterion（vs information-theoretic measure）
- **POWERPLAY 优点**：by design 不会 catastrophic forgetting；产生 clearly separated tasks with recorded solutions；feasible

可以理解为：**POWERPLAY = Formal Theory of Creativity 的 greedy、feasible、forgetting-proof 版本**。

---

## 13. 实验结果（Section 8 + [52, 53]）

Schmidhuber 把详细实验放在另两篇 paper：

- **Srivastava, Steunebrink, Stollenga, Schmidhuber 2012** "Continually adding self-invented problems to the repertoire: First experiments with POWERPLAY" (ICDL-EPIROB)
- **Srivastava, Steunebrink, Schmidhuber 2012** "First Experiments with POWERPLAY" https://arxiv.org/abs/1210.8385

主要发现：
1. SLIM RNN 在 open-ended fashion 下不断加新 task，展现 developmental stages
2. 自动 modularize：reuse 旧 skill 的 code
3. 偏好 "改动少、影响 task 少" 的修改——验证 Section 3.3.2 的 prediction
4. 有时 compress 旧 skill，有时 invent 新 task

---

## 14. Words of Caution（Section 10）——我觉得这是 paper 最被低估的 section

Schmidhuber 明确警告：

> 不要把 general POWERPLAY 放到 internet + 物理设备控制权的环境中。它不是传统 virus，但会 continually change in a way hard to predict，driven by 提升自己 problem-solving capacity 的欲望。

这其实预言了今天 AGI safety 社区讨论的 instrumental convergence 问题——curiosity-driven agent 可能 acquire resource、expand capability 作为 sub-goal。Schmidhuber 用一句 "curiosity can kill the cat" 结尾——指 self-invented task 可能让 agent 走进 fatal state。

参考 AGI safety instrumental convergence: https://arxiv.org/abs/1102.4648

---

## 15. 现代视角下的关联

### 15.1 与 modern RL 的关联
- POWERPLAY 的 "self-invented task + 不忘记旧 task" 与 **goal-conditioned RL + HER (Hindsight Experience Replay)**、**Multi-task RL with replay buffer for off-policy learning** 有结构性相似
- Curiosity-driven exploration（ICM, RND）是 Formal Theory of Creativity 的直接后代
  - ICM: https://arxiv.org/abs/1705.05363
  - RND: https://arxiv.org/abs/1810.12894

### 15.2 与 curriculum learning / self-play
- POWERPLAY 的 "always search for the simplest still unsolvable" 就是 **automatic curriculum learning**
- AlphaGo Zero 的 self-play 是 zero-sum 版本；POWERPLAY 是 non-zero-sum 版本（agent vs 自己的过去）
- Automatic Goal Generation in Unsupervised RL (OpenAI/FAIR): https://arxiv.org/abs/1912.13406

### 15.3 与 LLM 的 self-improvement / Constitutional AI
- LLM 的 self-instruct、self-rewarding LM 与 POWERPLAY 的 self-reference 思想相通
- Self-Rewarding LM: https://arxiv.org/abs/2401.10020

### 15.4 与 continual learning
- POWERPLAY 的 "by design 不忘记" 是 continual learning 文献一直追求但很少做到的
- 最近的 continual learning + modular network（Progressive Neural Networks, PackNet）是 prefix-code version (Section 3.3.3) 的工程化

### 15.5 与 AI safety
- POWERPLAY 是 curiosity-driven instrumental convergence 的 minimal example
- Schmidhuber 在 2011 年就指出：uncurtailed curiosity 在物理世界中可能 fatal to the agent

---

## 16. 一个我常 build intuition 的简单例子

想象一个 RNN solver $s$ 初始什么都不会。$T$ 包含"读 input pattern $I$，输出某个 target $O$"这种 recognition task。

**Step 1**：Levin search 找到短 program $p_1$：
- TASK INVENTION: $T_1 = $ "看到 $I = 0000$ 就输出 1"
- SOLVER MOD: 给 RNN 加一个小 weight change $q$，使 $q(0000) = 1$
- CORRECTNESS DEMO: 测试 $s_0$ 输出不是 1；$q$ 输出是 1；旧 task 集合是空的，所以 trivially 满足

**Step 2**：search 找到 $p_2$：
- 可能 $T_2 = $ "看到 $I = 1111$ 输出 0"，类似 step 1
- 或者 $T_2 = $ "用更短 code 实现 $T_1$"——压缩 wow-effect
- 或者 $T_2 = $ "看到 $0000$ 或 $1111$ 都输出 1"——generalization，但需要证明 $s_1$ 不能解这个

**Step N（很久之后）**：$s_N$ 已经能解很多 task，它的 generalization 把大量未测试的 task 都解了。POWERPLAY 现在很难找 "未解且能解" 的 task，因为：
- 大部分新 candidate 要么已经被 generalize 解了（acceptance criterion 第一条 fail）
- 要么需要 $s_N$ 也解不了（invention 太难）

系统进入 "compression 为主" 的 phase，开始 compress / speed-up 旧 solution。

这就是 "婴儿快速发育 → 成人缓慢精炼" 的 formal echo。

---

## 17. 一个细节：为什么 "invent task" 比 "only solve given task" 可能更快？

Schmidhuber 在 introduction 里有一个反直觉但重要的 observation：

> 表面上看，搜索 (task, solver) pair 的空间比单搜 solver 空间更大，似乎更难。但实际上，**额外的 freedom 可能大大降低 problem solver advances 之间的时间间隔**，因为系统经常可以 invent 一个相当简单、容易找 solution 的 task。

intuition：当 task 是 fixed 且 hard 时，Levin search 在 solution 上撞墙。当 task 是 free 时，agent 可以挑选 task 的高 density 区——大部分 bitstring 是容易解的 trivial task，但还没被 generalize 覆盖。每次找一个 "刚刚超出当前 generalization 边界" 的 task，难度梯度小，搜索快。

这本质上是 **active curriculum** 的 information-theoretic advantage。

---

## 18. 几个 Schmidhuber 在 paper 里没明说但我觉得关键的限制

1. **Acceptance criterion 的 undecidability**：证明 $s_{i-1}$ 不能解 $T_i$ 在 general 下 undecidable。所有 practical 实现都靠 bounded time test，这引入 false positive（$s_{i-1}$ 其实能解但没来得及跑完）。SLIM NN 的快速 reset 缓解但没消除这个问题。

2. **Search space 的 intractability**：即使 Levin search 是 asymptotically optimal，常数 $1/P(p^*)$ 可能大到宇宙年龄。Schmidhuber 用 OOPS reuse 旧 code 来降低这个常数，但 paper 没给量化保证。

3. **Generalization 的 "soft" nature**：acceptance criterion 用 "对所有 task $T_k$ 测试通过"，但 generalization 本身是 continuous 的。Variant II 的 cost function 软化了这一点，但仍然没明确处理 "新 solver 在 $T_k$ 上 degraded 一点点" 这种 grey zone。

4. **External task 的整合方式弱**：Section 6 只是说"可以把某些 $T_i$ 设为 external"，但没说如何 balance 外部 goal 与 self-invented goal 的搜索预算——这是 actual deployment 的关键问题。

5. **Probabilistic environment 下的 repeatability 假设**（Section 7.2）：probabilistic POWERPLAY 假设 fixed number of trials 能估计 cost。这对 high-variance 环境（real robot）可疑。

---

## 19. 总结：POWERPLAY 在 Schmidhuber 工作中的位置

Schmidhuber 从 1990 年代就在做 curiosity / intrinsic motivation，POWERPLAY (2011) 是他这条线的 **algorithmic crystallization**：

- 1991: 第一个 curious model-building control system（用 NN + RL）
- 1997-2002: Adaptive curiosity via co-evolution of two opposing modules（right brain / left brain）
- 2003-2004: OOPS（Optimal Ordered Problem Solver）——POWERPLAY 的 search engine
- 2006: Gödel Machine——self-referential optimal self-improver
- 2006/2010: Formal Theory of Creativity——formal framework
- **2011: POWERPLAY——greedy feasible implementation of FTC**
- 2012: SLIM NN + POWERPLAY 实验验证

之后 RL community 的 ICM/RND/goal-generation 在 deep RL 时代重新发现很多类似 idea，但很少引用 POWERPLAY（Schmidhuber 自己经常抱怨这件事，参见他的 "AI 最初的发明者们被无视" 系列 blog post）。

Schmidhuber 历史抱怨 reference: https://people.idsia.ch/~juergen/everybody.html

---

## 20. 我对这篇 paper 的主观评价

**Advantages**：
- Algorithmic 框架清晰，binary criterion 优雅
- Forgetting-proof 是真问题，POWERPLAY 的解法 clever
- Cost-based Variant II 把 invention + compression + speed-up 统一在一个 cost landscape
- Modularization 作为 search by-product 的 idea 美丽

**Disadvantages**：
- 大量引用自己之前的工作，新内容比例不高
- 实验全推到 follow-up paper，本文缺乏 empirical evidence
- 对 "undecidable acceptance" 没有正式处理
- 没有 quantitative 比较 vs 其他 curiosity methods
- Self-reference section 太短，本是 Schmidhuber 最有意思的方向

**对今天研究者的 value**：作为 conceptual framework 仍然非常有启发。具体 implementation 细节（FORTH-like language、SLIM NN）已经被 deep RL 取代，但 "search (task, solver) pair + 不忘记旧 skill + bias toward cheap-to-validate" 这个三件套，在今天 LLM agent + tool use + self-improvement 的语境下，很值得重新审视。

---

如果你想深入哪一块，比如 Gödel Machine 的 self-rewrite 机制、OOPS 的 prefix code 细节、SLIM NN 的 self-delimiting computation，或者把它接到现代 LLM agent 上，我可以继续展开。
