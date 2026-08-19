---
source_pdf: Improving Pre-Trained Vision-Language-Action.pdf
paper_sha256: e6f7f03cc0b706d369de4e0ba360d5e34bb16ebc80ca6fe400e114bf87d49672
processed_at: '2026-08-19T12:24:39-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版本

好,我把刚才那堆公式和术语倒出来,只留 intuition。

---

## 一句话

VLA 模型有个毛病:它只会"下意识"做动作,不会"想一下后果"。这篇 paper 给它装了个"脑补模拟器" — 先在脑子里预演一下,看哪条路走得通,再真去做。

---

## 问题到底难在哪

VLA 模型本质上是模仿学习。它看了几百万条人类演示,学会了"看到这个画面 + 听到这句话 → 应该这么动"。很像一个熟练工,手熟,但完全凭直觉。

问题出在 OOD:稍微换个场景,或者手抖一下把物体掉地上,VLA 就懵了。因为它训练数据里没有"怎么从掉东西救回来"的演示 — 演示里一旦失败,episode 就停了,数据里根本没记录"如何 recover"。

所以 VLA 会犯一些很蠢的小错:抓物没抓牢、太早关抽屉、把碗放歪。一个 0.5 秒的小失误,整段任务就废了。

另一边,model-based planning (像 MCTS、MPC) 天生会"想后果" — 它有 world model,可以 rollout 看未来。但 robotics 的 action space 太大,reward 又稀疏,直接搜就是 $2000^{100}$ 这种天文数字,根本搜不动。

所以两边的弱点刚好互补:
- VLA: 知道大概怎么动,但不会想后果
- MCTS: 会想后果,但不知道从哪开始搜

VLAPS 把它们拼起来。

---

## VLAPS 干了什么 (三步人话版)

### 第一步:让 VLA 先指个方向

每个决策点,VLA 先跑一次,输出一段它"觉得"该做的 4 步小动作 (action chunk)。这个输出当 anchor — 告诉 search "大概往这个方向搜"。

### 第二步:在 anchor 附近捞候选

paper 预先建了个 action chunk 库 $\Phi$ (2000 个,从历史成功 trajectory 聚类出来的)。从库里挑 10 个最接近 VLA 输出的 chunk,当这个节点的候选动作。

直觉: VLA 说"我大概要往左伸手去抓",那 search 就只考虑"往左伸手"附近的几种 chunk,别浪费时间想"开抽屉"或"往后退"这种完全不相关的事。

这一步把 branching factor 从 2000 砍到 10 — 200 倍压缩,这是 tractable 的关键。

### 第三步:用 world model 看后果,选最靠谱的

对这 10 个候选 chunk,每个都用 world model rollout 一下,看"假如我真这么做,环境会演化成什么样,最后任务能完成吗"。

- 如果某个 chunk 的 rollout 撞到 success → 直接 return 这条路
- 如果都没成功 → 用 PUCT-style 的 explore/exploit 在这 10 个里反复挑,边搜边扩树
- 预算用完还没成功 → return 最常被访问的那条

然后执行选中的 chunk,拿到新的真实 observation,循环再来一遍。这是 receding-horizon MPC 风格,边走边重新规划。

---

## 一个具体 example 帮 intuition

paper Figure 3 里有三个 example,我挑最直观的:

**任务**: 把碗放进抽屉然后关上。

**VLA-only 怎么失败**: VLA 看到 bowl,伸手抓,放进抽屉,但还没放稳就 close gripper + 推抽屉 — 碗卡在抽屉边缘,抽屉关不上,任务 fail。VLA 这么做是因为它训练数据里"关抽屉"这个 action chunk 出现得早,它就 mimic 了,完全没考虑"碗还没进去就关抽屉会怎样"。

**VLAPS 怎么救**: 在 close 抽屉这个决策点,VLAPS 在 world model 里 rollout 一下 — 发现"假如我现在关抽屉,碗会卡住,任务 fail"。于是 back off,选另一个 chunk (多伸一下把碗推到底再关),rollout 显示成功,执行。任务 pass。

关键: VLA 自己完全没这个"假如...会怎样"的能力。它只是 pattern match 训练数据。VLAPS 给它装了 *imagination*。

---

## 为什么 work (核心 intuition)

三个原因:

**(1) VLA prior 让 search tractable**
没有 prior 的 uniform search 在 $2000^{100}$ 空间里死定。VLA 输出相当于说"答案大概在这附近",search 只在 *behaviorally-relevant manifold* 上走,200 倍压缩 + depth 压缩,直接从 intractable 变成 600 秒内能搜完。

**(2) Search 修补 VLA 的 brittle error**
VLA 失败大多是 *小错* (抓物没抓牢、太早 close、放偏一点)。这些错在 world model rollout 里立刻暴露 — 比如掉物、卡住 — search 看到 fail 就 back off 换路。这等价于让 VLA 在 inference 时拿到"假如你这么做会摔"的 hypothetical demonstration,而它训练数据里根本没有这种 demonstration。

**(3) Receding horizon 让 sim-to-real error 不累积**
即使 world model 不完美,每 4 步重新拿真实 observation 重新规划,error 不会无限放大。这跟 MPC 的鲁棒性同源。

---

## 实验结果一句话

93M 参数的 Octo (相对小的 VLA) + VLAPS,在 LIBERO 上打平甚至超过 3.3B 参数的 π0-FAST (SOTA 大模型)。也就是说,**用 inference-time compute 换 model size**,在 robotics 上首次被 clean 地展示出来。

而且 search time 自适应:VLA 弱的时候多搜 (50k checkpoint 平均 144 秒),VLA 强的时候少搜 (200k checkpoint 平均 13 秒)。VLA 越好,VLAPS 越快 — 因为 VLA 直接给对答案,search 早 terminate。这是 test-time compute scaling 在 robotics 的第一个 concrete instance。

---

## 局限 (paper 自己认 + 我加的)

**Paper 自认**:
- 依赖 accurate world model。真机部署时 sim-to-real gap 是问题。paper 留给 future work。
- VLA inference 慢导致 search latency 高。可以靠 batch query、parallel rollout、quantization 缓解。

**我加的**:
- action chunk 距离用 Euclidean 简单粗暴,不反映 behavioral similarity。两个 chunk 可能 semantically 等价但 Euclidean 差很远。可以学一个 chunk embedding 来做 metric。
- $Q \equiv 0$ 是当前限制。等 multi-task value function 成熟,plug 回 PUCT 会更准。
- Object suite 上 VLA 200k 反而比 100k 差 (过拟合),VLAPS 跟着跌 — 说明 prior 太尖时 search 也救不回来。可以 adaptive 调 $\alpha$ 跟 VLA uncertainty 挂钩。

---

## 串到更大的图景

这个 paper 站在三条线交汇处:

**Line 1: LLM 上的 test-time compute scaling**
Chain-of-Thought、Tree of Thoughts、DeepSeek-R1 都在说:让模型多"想一会儿"能显著提升表现。VLAPS 把这套搬到 robotics — VLA 是 robot 的 "LLM",world model 是 "reasoning sandbox",MCTS 是 "System 2 deliberation"。这跟你 [Liang et al. 2025 survey](https://arxiv.org/abs/2501.02497) 里讲的 System 1 → System 2 过渡完全对应。

**Line 2: AlphaZero 谱系**
AlphaGo / AlphaZero / MuZero 都是 "policy + value + MCTS + self-play" 闭环。VLAPS 是这个范式在 robotics 上的 *inference-time-only 版本* — 没有训练,只用 pre-trained VLA 当 policy prior + simulator 当 world model。差一个 *self-distillation loop* (用 VLAPS 生成成功轨迹反过来 finetune VLA) 闭合。如果有人补上这一步,就是完整的 "AlphaZero for VLA"。

**Line 3: World model 谱系**
现在 world model 靠 simulator,但 [Genie 2](https://deepmind.google/technologies/generative-interactive-environment/genie-2/) 这种 generative interactive environment 一成熟,VLAPS 可以直接套 — robot 不再需要精确物理引擎,而是"想象"环境。这把 search 从 state-space 推到 frame-space,跟你谈过的 *video as world model surrogate* 方向接上。

---

## 我觉得最值得 follow-up 的两条路

1. **VLAPS + generative world model (Genie 2)**: 把 search 从 simulator 推到生成式 environment。一旦 work,real-world deployment 难题解决一半。

2. **VLAPS 的成功轨迹 → self-distillation → 闭合 AlphaZero loop**: 现在 VLAPS 是 inference-time-only。如果把它搜出来的成功轨迹攒起来当 finetune data,反喂 VLA,就是 AlphaGo 的 self-play 训练循环在 robotics 上的版本。这条路 robotics 上还没人完整跑通,挑战主要在 sample efficiency 和 reward 信号。但一旦通了,就是 robotics 的 "R1 moment"。

---

## 参考链接

- LIBERO: https://libero-project.github.io/
- Octo: https://octo-models.github.io/
- π0-FAST: https://arxiv.org/abs/2501.09747
- OpenVLA: https://openvla.github.io/
- AlphaZero: https://www.nature.com/articles/nature24270
- MuZero: https://www.nature.com/articles/s41586-020-03051-4
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Test-time compute survey: https://arxiv.org/abs/2501.02497
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Genie 2: https://deepmind.google/technologies/generative-interactive-environment/genie-2/
- ACT (Action Chunking, ALOHA): https://tonyzhaozh.github.io/aloha/
- ECoT (Embodied Chain-of-Thought): https://embodied-cot.github.io/
- SayCan: https://say-can.github.io/
- TD-MPC2: https://tdmpc2.github.io/

---

要继续聊的话,我最好奇你怎么看那条 *self-distillation 闭合 AlphaZero loop* 的路 — 你在 LLM 上看着 R1 走通了一遍,robotics 上这条路差什么。

---

# VLAPS: 把 VLA 当 prior, 把 MCTS 当 inference-time compute 旋钮

你好 Andrej。这篇 paper 我读得很兴奋,因为它正好落在你最近关心的几条线交汇处 — *test-time compute scaling*、*System 1 vs System 2*、以及把 AlphaZero-style 的 search 范式从棋盘搬到 robot manipulation。我先把方法、公式、实验细节逐层拆给你, build intuition, 再把它的相邻工作谱系都串起来,这样你能很快判断它的位置和延伸空间。

---

## 1. 这篇 paper 想解决的问题

pre-trained VLA model (像 [Octo](https://octo-models.github.io/)、[RT-2](https://robotics-transformer2.github.io/)、[π0](https://www.physicalintelligence.company/blog/pi0)、[OpenVLA](https://openvla.github.io/)) 在 in-distribution 下表现亮眼,但在 OOD 时会出现 brittle behaviors 和 unsafe failures。根因: VLA 本质上是 behavior cloning,forward pass 时没有任何 mechanism 去 *imagine* 自己 action 的后果 — 它只能 mimic 训练数据里见过的轨迹分布。

而 model-based planning (MCTS、MPC) 能显式 reason 未来,但 robotics state-action space 维度高、reward sparse,搜索空间 exponential blow-up,通常需要 handcrafted heuristics。

VLAPS 的核心主张: 用 VLA policy 当 *prior* 来 bias 一个 modified MCTS,既能让 search 在天文数字级别的 action space 里变得 tractable,又能让 VLA 拿到 *lookahead capability* 来纠正自己的 brittle 决策。这是一条 **inference-time compute** 路径,training-free,VLA agnostic。

直觉上, VLA 提供 "我大概知道接下来这一小段该怎么动"的局部 prior, MCTS 在这个 prior 上做 lookahead, world model 给"假如我这么做,环境会演化成什么样"的 simulator。三者拼起来就完成了 System-1 (VLA 的 fast reactive policy) 到 System-2 (search with consequences) 的过渡。

参考: 你自己 [Liang et al. 2025](https://arxiv.org/abs/2501.02497) 对 test-time compute 的 survey, [Hao et al. 2023](https://arxiv.org/abs/2305.14992) 把 LLM 当 world model + planner, [Yao et al. 2023 Tree of Thoughts](https://arxiv.org/abs/2305.10601) 把 thought 当 search node。

---

## 2. Formulation: MDP、task、VLA、MCTS 四件套

### 2.1 环境与任务

环境形式化为 MDP $\mathcal{M} = (S, A, T, R)$:

- $S$: state space。包含 object locations、proprioceptive measurements、image $I$
- $A \subset \mathbb{R}^n$: 原始 robot action, $n$ 是 action 维度 (e.g. end-effector pose + gripper)
- $T: S \times A \to S$: deterministic transition (paper 假设, 留 stochastic 给 future)
- $R$: reward function

任务 $\mathcal{T} = (L_\mathcal{T}, S_\mathcal{T})$:
- $L_\mathcal{T}$: natural language instruction (例如 "Place the orange juice in the basket")
- $S_\mathcal{T} \subseteq S$: 任务完成状态集合

sparse reward: $R_\mathcal{T}(s) = 1$ if $s \in S_\mathcal{T}$ else $0$。这种 sparse reward + 长horizon 是 paper 想 tackle 的硬骨头。

### 2.2 VLA 模型

VLA 是 large transformer,输入 (image $I_t$, language $L_\mathcal{T}$, history), 输出 action 序列。训练通常 finetune 一个 pre-trained VLM,要么 autoregressive predict action tokens (像 [OpenVLA](https://openvla.github.io/), [π0-FAST](https://arxiv.org/abs/2501.09747)),要么 guide diffusion (像 [π0](https://www.physicalintelligence.company/blog/pi0))。Action 通常 tokenized 成 discrete tokens (FAST tokenizer,见 [Pertsch et al. 2025](https://arxiv.org/abs/2501.09747))。

VLAPS paper 用 [Octo](https://octo-models.github.io/) (93M params, transformer with diffusion head) 当 backbone,finetune 在 [LIBERO](https://libero-project.github.io/) 上。

### 2.3 MCTS 四阶段 (背景)

经典 MCTS ([Browne et al. survey](https://arxiv.org/abs/1407.7186) 是好 review):
1. **Selection**: 从 root $v_0$ 沿 PUCT criterion 递归到 leaf
2. **Expansion**: 给 leaf 加 child
3. **Simulation**: rollout policy 估 leaf value
4. **Backpropagation**: 沿路径更新 statistics

VLAPS 在每一步都"魔改"了这套: expansion 用 VLA 来 propose action chunks (而不是 enumerate all actions); selection 用 VLA-derived prior 代替 learned value (Q ≡ 0); simulation 直接用 VLA 自己当 rollout policy; backprop 因为 Q≡0 简化成 visit count update,terminal 时直接 return success。

---

## 3. VLAPS 算法详解

### 3.1 整体流程

每个决策点 $t$:
1. Root node $v_0$ 包含当前 state $s_t \in S$
2. **Selection**: 用 VLA-biased PUCT 从 $v_0$ 走到 leaf $v$
3. **Expansion**: 用 VLA 从库 $\Phi$ 中采样 $k$ 个 contextually-relevant action chunks,组成 $\Phi_v$ (节点 $v$ 的"局部候选集"),用 world model $\widehat{\mathcal{M}}$ simulate 每个 chunk 得到 child nodes
4. **Simulation**: 从每个 child 用 VLA policy 做 rollout,直到 task complete 或 max horizon (300 步)
5. **Backprop**: 更新 visit count $N(v, u^i)$;若 rollout 撞到 success,直接 return 该路径的 action sequence
6. 若 budget 用完没找到 success: return root node 处最常被访问的 action chunk
7. 在真环境执行 action sequence,拿到新 observation,循环

注意这是 *receding-horizon* 的 MPC 风格,而非 open-loop 执行整棵树。

### 3.2 Action chunk 抽象 (§IV-B 的关键设计)

直接搜原始 action 不可行: $A \subset \mathbb{R}^n$ 连续、动作短 timescale、horizon 长 → branching factor + depth 双重爆炸。

VLAPS 借鉴 [ACT (Action Chunking with Transformers, Zhao et al. 2023)](https://tonyzhaozh.github.io/aloha/) 的 action chunk 思想:

$$u_t = (a_t, a_{t+1}, \ldots, a_{t+H}) \in U \subset \mathbb{R}^{H \times n}$$

变量:
- $u_t$: 一个 *temporally-abstract action chunk*,长度 $H$ 的 action 序列
- $H$: chunk horizon (paper 里 $H = 4$,与 Octo 训练对齐)
- $n$: 单步 action 维度

这样每个 tree node 的"一次 decision"是 $H$ 步原始动作,tree depth 在 chunk space 里被压缩 $H$ 倍。$H$ 同时是"VLA 被查询的频率" — paper 把它做成可配 knob,允许用户在 planning latency 与 granularity 间权衡。

### 3.3 候选库 $\Phi$ 的构造

action chunk 空间 $\mathbb{R}^{H \times n}$ 太大,paper 不直接搜它,而是建一个 finite library $\Phi \subseteq U$:

- 实验里: aggregate VLA-only policy 下成功 trajectory 里的 action sequences
- per-dimension normalize
- 用 K-Medoids clustering 萃取 2000 个 prototype

paper 顺便提了 alternative: 直接用 VLA 训练数据里所有长度 $H$ 的 action sequences 当 $\Phi$ — 这等于把 demonstration 数据当成 "behaviorally-relevant manifold" 搜,而非搜 $\mathbb{R}^{H \times n}$ 这个 whole space。

intuition: 这相当于先验说"robot 实际会做的动作,是低维流形"; 搜这个流形上的离散点比搜连续 $\mathbb{R}^{H \times n}$ 容易得多。这个 idea 跟 [Hubert et al. 2021](https://arxiv.org/abs/2104.06378) 在 MuZero 上做 complex action space 的工作,以及 RRT-style motion planning 中"从 demonstration 学 primitive"思路是亲戚。

### 3.4 Context-aware subset sampling: $\beta_\Phi$ 分布 (公式 1)

库 $\Phi$ 仍有 2000 个,从每个 node 都全搜太多。于是用 VLA 当 prior 来选 *这个 node 现在该关心哪 k 个 chunk*。

公式 1:
$$\beta_\Phi(u^i | I_t, L_\mathcal{T}) = (1 - \epsilon) \cdot \frac{\exp\bigl(-\alpha \cdot \rho(u^i, u^{vla})\bigr)}{\sum_{j=1}^{|\Phi|} \exp\bigl(-\alpha \cdot \rho(u^j, u^{vla})\bigr)} + \frac{\epsilon}{|\Phi|}$$

变量逐个拆:

- $u^i \in \Phi$: 候选库中第 $i$ 个 chunk
- $I_t$: 当前 image observation
- $L_\mathcal{T}$: language task instruction
- $u^{vla}(I_t, L_\mathcal{T}) \sim P_{vla}(I_t, L_\mathcal{T})$: VLA 在当前 context 下采样的一个 reference chunk,作为分布的"锚点"
- $\rho(u^i, u^{vla})$: 两 chunk 间距离 metric。paper 用 Euclidean distance over flattened chunks: $\rho(u^1, u^2) \triangleq \|u^1_{flat} - u^2_{flat}\|_2$,即把 $H \times n$ chunk 沿时间维 flatten 成 $Hn$ 维向量后取 L2
- $\alpha \in \mathbb{R}_+$: inverse temperature。$\alpha$ 越大,分布越尖锐地集中在 VLA 输出附近。实验里 $\alpha_{\beta_\Phi} = 10.0$
- $\epsilon \in [0,1]$: epsilon-uniform exploration 项,确保 $\Phi$ 里每个 chunk 都有非零概率被采到。实验里 $\epsilon_{\Phi_\beta} = 0.1$
- $|\Phi|$: 库大小 (2000)

形式上是 *softmax-distance prior* + *uniform tail* — 等价于把 epsilon-greedy 的 greedy 部分换成"以 VLA 输出为中心的 softmax"。这给 VLA 输出附近 chunks 高权重,但留 $\epsilon$ 概率给完全 OOD 的 chunk (万一 VLA 错了还能 explore 别处)。

每个 leaf expansion 时,从 $\beta_\Phi$ 采 $k = 10$ 个 chunks 组成 $\Phi_v \triangleq \{u^1, \ldots, u^k\}$。$\Phi_v$ 一旦采样就在该 node 整个 search 期间固定,reuse 到 episode 结束。这是 paper 把 branching factor 从 $|\Phi| = 2000$ 降到 $k = 10$ 的关键 — 直接压缩 200 倍。

**Intuition**: VLA 在这里给的是"局部 contextually-relevant manifold 上的 prior"。当 robot 已经抓到物体、离 basket 很近时,VLA 输出大概会是"move toward basket + release";公式 1 就把 search 限制在"接近这个意图"的 chunks 集合上。Paper 没必要让 search 重新发现"现在该 close gripper 还是 open gripper"。

### 3.5 VLA-biased selection: $\psi_{\Phi_v}$ prior (公式 2)

有了 $\Phi_v$ 后,在这 $k$ 个 chunk 中怎么选? 经典 PUCT 是:
$$\text{PUCT}(v, a) = Q(v, a) + c \cdot \pi_{prior}(a|s) \cdot \frac{\sqrt{N(v)}}{1 + N(v, a)}$$

VLAPS 砍掉 $Q$ 项 (设 $Q \equiv 0$),只保留 prior + exploration bonus:

$$\text{SCORE}(v, u^i) = \psi_{\Phi_v}(u^i | I_t, L_\mathcal{T}) \cdot \frac{\sqrt{N(v, u^i)}}{1 + N(v, u^i)}$$

变量:

- $v$: 当前 node
- $u^i \in \Phi_v$: 该 node 候选 chunk
- $\psi_{\Phi_v}(u^i | I_t, L_\mathcal{T})$: VLA-derived prior,跟 $\beta_\Phi$ 同形 softmax centered on $u^{vla}$,实验里 $\alpha_{\psi} = 5.0$ (比 $\beta_\Phi$ 的 10.0 更软,留更多 explore 余地)
- $N(v, u^i)$: (node $v$, chunk $u^i$) 的访问计数
- 第二项 $\frac{\sqrt{N(v)}}{1+N(v,u^i)}$ 是 PUCT 经典 exploration bonus: 鼓励访问次数少的 child (UCB1 风格)

**注意几个 design choice**:

1. **$Q \equiv 0$**: paper 明确说省略 value estimate,因为 generalist multi-task robot policy 的 value function 是 open problem — 任何 learned critic 都可能错。这种"先验依赖"模式适合 sparse-reward + strong VLA prior 的 setting。一旦 rollout 撞 success 就 terminate,根本不需要 Q 来区分 child 优劣。

2. **selection prior 与 expansion sampling prior 用同一个 VLA 输出 $u^{vla}$ 当中心**: 这是重要细节 — VLA 在该 state 会被查询一次,得到 $u^{vla}$;这个 $u^{vla}$ 同时被 $\beta_\Phi$ (用来选 $\Phi_v$) 和 $\psi_{\Phi_v}$ (用来在 $\Phi_v$ 内选 child) 复用。所以 VLA 的 inference cost 是可控的,每个 node 一次。

3. **$\alpha_\beta > \alpha_\psi$**: expansion 时更尖 (10.0 vs 5.0),即"采样的 $\Phi_v$ 高度集中在 VLA 输出附近";selection 时更平 (5.0),即"在 $\Phi_v$ 里还是给其它 chunk 一些机会"。这种"先 tight narrowing 再 loose traverse"的设计很 eloquent。

### 3.6 为什么这套能 tractable?

paper 算了一笔账 (§V-B5): 假设无 VLA refinement,branching factor $b = |\Phi| = 2000$,典型 search depth $d = 100$。一棵 full $b$-ary tree 高 $d$ 有 $\approx b^d = 2000^{100}$ 节点 — 直接搜 intractable。uniform random 期望要 $b^d$ 次才采到 success trajectory。VLAPS 把 branching factor 降到 $k = 10$,实际有效 depth (chunk space) 降到 $d / H = 25$,加上 prior-directed,自然能在 600s wall-clock 内找到 solution。

---

## 4. 实验细节与表解读

### 4.1 Setup

- **Benchmark**: [LIBERO](https://libero-project.github.io/) 全 5 个 suite — Spatial、Goal、Object、90、10。每 suite 10 task,每 task 10 initial conditions
- **VLA backbone**: [Octo-base-1.5](https://octo-models.github.io/),93M params,finetune 在 LIBERO demonstrations,256×256 fixed cam + 128×128 wrist cam,input 是 image+language,输出 end-effector pose + gripper command
- **Checkpoints**: 10k, 50k, 100k, 150k, 200k gradient steps — 把 finetune 程度当 VLA quality 的 controllable proxy
- **VLAPS hyperparams**: 300 MCTS samples/iter, $k=10$ children/node, rollout max 300 steps, $H=4$, max tree depth 100, $|\Phi|=2000$, $\alpha_\beta=10.0$, $\alpha_\psi=5.0$, $\epsilon=0.1$ 两者, A100 GPU, 600s/task wall-clock cap
- **Comparison**: VLA-only baseline (Octo 直接 deploy), $\pi_0$-FAST (3.3B params, SOTA)

### 4.2 Table I 深入读

我重新整理成 mental model — 每个 suite 在每个 checkpoint 下,(VLA 成功率, VLAPS 成功率, VLA runtime, VLAPS runtime):

| Suite | ckpt | VLA% | VLAPS% | Δ | VLAPS runtime (s) |
|---|---|---|---|---|---|
| Spatial  | 50k  | 34  | 97  | +63 | 71.9 |
| Spatial  | 100k | 83  | 99  | +16 | 17.9 |
| Spatial  | 200k | 86  | 98  | +12 | 13.5 |
| Goal     | 50k  | 50  | 86  | +36 | 54.3 |
| Goal     | 100k | 87  | 94  | +7  | 19.8 |
| Goal     | 200k | 91  | 91  | 0   | 15.3 |
| Object   | 50k  | 6   | 73  | **+67** | 144.8 |
| Object   | 100k | 32  | 82  | +50 | 41.8 |
| Object   | 200k | 19  | 54  | +35 | 39.8 |
| 90       | 50k  | 12  | 51  | +39 | 147.3 |
| 90       | 100k | 65  | 94  | +29 | 37.8 |
| 90       | 200k | 70  | 91  | +21 | 32.6 |
| 10       | 100k | 37  | 74  | +37 | 87.5 |
| 10       | 200k | 63  | 84  | +21 | 58.3 |

几个直觉:

**(a) 当 VLA quality 极差 (10k)**,VLAPS 几乎全 fail (Spatial 0→0, Object 0→2, 10 0→0)。这说明 VLA prior 至少要有最低 signal 来 anchor $\beta_\Phi$ 和 $\psi_{\Phi_v}$。这跟 LLM tree-search 的直觉一致: 如果 base policy 完全 random,search 也救不回来 — prior 是 prerequisite。但 Goal 10k 是 3→24、90 10k 是 3→9,说明即便成功率低,VLA 只要"在某种程度上"输出 reasonable chunk,search 就能找到 solution。

**(b) 最大相对 gain 出现在 VLA 中等弱 (50k Object)**: 6→73 (+67pp)。这是 paper 标题级别的 result。直觉是: 当 VLA "知道大概怎么做但执行 brittle" 时, search 来纠正小错误 (抓物失败、premature gripper close) 收益最大 — Figure 3 的 qualitative examples 正好印证这点。

**(c) Goal 200k 是 saturation edge**: VLA 91% vs VLAPS 91%,无提升。说明 VLA 在该 suite 已经接近 ceiling,search 找不到多少可纠正的 error。这定义了 VLAPS 的上限。

**(d) Runtime 随 VLA quality 提升而急剧下降**: Spatial 50k → 100k 从 71.9s 降到 17.9s。因为 quality 高时, VLA 自己的 rollout 容易撞 success,search 早早 terminate。这是 paper 强调的 *adaptive test-time compute*: 困难任务多搜,容易任务少搜。

**(e) Object suite 怪异现象**: 100k (32% VLA → 82% VLAPS) 比 200k (19% → 54%) 表现更好。这其实揭示 finetune 过拟合, VLA 在 200k 时 distribution shift 加剧,prior quality 反而下降。VLAPS 完全 follow VLA quality 的 trend — paper 借此 argue "VLAPS 性能 monotonically 随 base VLA 改进而改进,所以未来 VLA 进步会自动 lift VLAPS"。

### 4.3 Table II: 用 93M Octo 打 3.3B π0-FAST

| Suite | Octo (93M) | Octo + VLAPS | π0-FAST (3.3B) |
|---|---|---|---|
| Spatial | 83 | 99 | 96 |
| Goal    | 87 | 94 | 96 |
| Object  | 32 | 82 | 99 |
| Libero-10 | 37 | 74 | 71 |

直觉:

- **Spatial、Goal、Libero-10** 上 Octo+VLAPS ≈ 或超过 π0-FAST。用 93M 参数 + test-time search 匹配 3.3B 参数的 SOTA。这跟你谈过多次的 *smaller model + more test-time compute* 的 trade-off 完全一致,跟 DeepSeek-R1 ([Guo et al. 2025](https://arxiv.org/abs/2501.12948)) 在 LLM 上展示的 RL-then-search 思路异曲同工
- **Object suite 是 outlier**: VLAPS 82 vs π0-FAST 99。Object suite 涉及多种物体 manipulation,VLA prior 可能在这里更 noisy,paper 没深挖。直觉是: 当任务需要 *finer-grained motor control* 而非 *high-level planning* 时,search 收益小,因为 success 更多由低层 motor quality 决定,而非 lookahead

### 4.4 为什么 VLA + MCTS 比 VLA-only 强这么多? (Intuition)

paper §V-B1 的 qualitative 观察: VLAPS 更频繁地避免 small errors — 掉物、移到 uncommon state。这些 small errors 之所以致命,是因为 VLA 训练数据里很少见 *error recovery* demonstration (一旦失败,通常 episode 就终止了,数据里没有"如何从失败中救回来"的部分)。

VLAPS 的 lookahead 直接 *在 world model 里* 看到这种 small error 的后果,然后 back off 选别的 chunk。这等价于让 VLA 在 inference 时获得一个 *hypothetical demonstration* —— "假如我这么做会摔,那我换一种做法"。这跟 [ECoT (Robotic Control via Embodied Chain-of-Thought)](https://embodied-cot.github.io/) 的 motivation 同源,只是 ECoT 用 language 当 thought,VLAPS 用 world-model rollout 当 thought。

---

## 5. 相邻工作谱系 (mapping the territory)

### 5.1 VLA model 谱系

- [RT-1 (Brohan et al. 2023)](https://robotics-transformer1.github.io/): early transformer-based VLA, real-world Google robot
- [RT-2 (Zitkovich et al. 2023)](https://robotics-transformer2.github.io/): Co-fine-tune VLM + robot data,transfer web knowledge
- [PaLM-E (Driess et al. 2023)](https://palm-e.github.io/): embodied multimodal LLM
- [Octo (2024)](https://octo-models.github.io/): open-source,93M,diffusion head
- [OpenVLA (Kim et al. 2024)](https://openvla.github.io/): 7B,open-source,SOTA-ish
- [π0 (Black et al. 2024)](https://www.physicalintelligence.company/blog/pi0): flow matching,3.3B
- [π0-FAST (Pertsch et al. 2025)](https://arxiv.org/abs/2501.09747): FAST tokenizer 替代 flow,加速 inference
- [Hi Robot (Shi et al. 2025)](https://hi-robot.github.io/): hierarchical VLA,open-ended instruction following

VLAPS 跟这些是 orthogonal — 它是个 *inference-time wrapper*,理论上可以套任何 VLA。Paper 用 Octo 是为了 simplicity,但你可以想象 OpenVLA+VLAPS 或 π0+VLAPS 也成立。

### 5.2 LLM 上的 test-time compute 与 search

- [Chain-of-Thought (Wei et al. 2022)](https://arxiv.org/abs/2201.11903): linear thought
- [Self-Consistency (Wang et al. 2023)](https://arxiv.org/abs/2203.11171): sample 多条 CoT 投票
- [Tree of Thoughts (Yao et al. 2023)](https://arxiv.org/abs/2305.10601): branching thought tree + BFS/DFS
- [Reasoning with Language Model is Planning with World Model (Hao et al. 2023)](https://arxiv.org/abs/2305.14992): LLM 自当 world model + MCTS
- [AlphaZero-like tree-search for LLM decoding (Feng et al. 2023)](https://arxiv.org/abs/2309.15879): LLM 上做 AlphaZero-style search
- [DeepSeek-R1 (Guo et al. 2025)](https://arxiv.org/abs/2501.12948): RL on reasoning
- [Test-time Computing survey (Ji et al. 2025)](https://arxiv.org/abs/2501.02497)
- [Survey of reasoning LLMs (Li et al. 2025)](https://arxiv.org/abs/2502.17419)

VLAPS 直接对应这个 line 在 robotics 的落地 — VLA 是 robot 的"LLM",world model 是"reasoning sandbox",MCTS 是"System 2 deliberation"。

### 5.3 Model-based RL 与 planning 谱系

- [AlphaGo (Silver et al. 2016)](https://www.nature.com/articles/nature24270): supervised + RL + MCTS,Go
- [AlphaGo Zero / AlphaZero (Silver et al. 2017)](https://www.nature.com/articles/nature24270): self-play, no human data
- [MuZero (Schrittwieser et al. 2020)](https://www.nature.com/articles/s41586-020-03051-4): learned latent world model + planning
- [DreamerV2 (Hafner et al. 2021)](https://arxiv.org/abs/2010.02193): discrete latent world model, Atari
- [TD-MPC (Hansen et al. 2022)](https://tdmpc.github.io/): temporal difference + MPC
- [TD-MPC2 (Hansen et al. 2024)](https://tdmpc2.github.io/): scalable world models for continuous control
- [Bertsekas 2024](https://doi.org/10.1016/j.ifacol.2024.09.056): MPC 跟 RL 的 unified framework,展示单步 lookahead 能纠正大量 value function error

VLAPS 跟 MuZero 在精神上最近,但有重要差异: MuZero 联合训练 world model + policy + value;VLAPS 是 *training-free*,直接复用 VLA 的 prior 当 policy,world model 假设是 black-box simulator。这跟 Bertsekas 的"single-step lookahead 就能纠正 substantial error"观察完全契合 — VLAPS 100 步 lookahead 远超单步,自然收益更大。

### 5.4 World model 谱系

paper §VI 说 future work 要去 jointly learn world model。可选项:

- 高保真 simulator: [Isaac Gym (Makoviychuk et al. 2021)](https://arxiv.org/abs/2108.10470), [Robosuite (Zhu et al. 2020)](https://arxiv.org/abs/2009.12293), [LIBERO](https://libero-project.github.io/)
- Learned dynamics: Dreamer 系列, [TD-MPC2](https://tdmpc2.github.io/)
- Video-prediction model: [IRIS (Yang et al. 2024)](https://arxiv.org/abs/2312.05415), [Genie (Bruce et al. 2024)](https://arxiv.org/abs/2402.15595), [Genie 2 (DeepMind 2024)](https://deepmind.google/technologies/generative-interactive-environment/genie-2/), [Video Language Planning (Du et al. 2024)](https://video-language-planning.github.io/)

我猜 Genie 2 这种 generative interactive environment 一成熟,VLAPS 直接套它当 world model 是条很有想象力的路 — 相当于 robot 不再需要精确物理 simulator,而是"想象"环境。这跟你谈过的 *video as world model surrogate* 方向直接对接。

### 5.5 Language-conditioned planning 谱系

- [SayCan (Ahn et al. 2022)](https://say-can.github.io/): LLM high-level plan + affordance,low-level motion primitives
- [Plan-Seq-Learn (Dalal et al. 2023)](https://arxiv.org/abs/2312.11575): LLM-guided RL for long-horizon
- [LGMCTS (Chang et al. 2024)](https://arxiv.org/abs/2407.07018): language-guided MCTS over object rearrangement primitives
- [Video Language Planning (Du et al. 2024)](https://video-language-planning.github.io/): VLM 提 plan + video model predict 后果
- [Reflective Planning (Feng et al. 2025)](https://arxiv.org/abs/2502.16707): VLM + env model, single-path rollout revision
- [ECoT (Zawalski et al. 2024)](https://embodied-cot.github.io/): embodied chain-of-thought on robot
- [Hi Robot (Shi et al. 2025)](https://hi-robot.github.io/): hierarchical VLA

VLAPS 的独特点: 它搜 *VLA-derived action chunks* (low-level motor commands),而非搜 high-level text plans 或 symbolic primitives。这意味着 VLAPS 输出可以直接 robot-execute,不需 motion planner 或 goal-conditioned low-level policy 中介。这是 vs SayCan、LGMCTS 等 method 的关键差异。

---

## 6. 一些直觉性的 critique 与延伸思考

**(a) action chunk 距离 metric 选 Euclidean 是否合理?**

paper 用 $\rho(u^1, u^2) = \|u^1_{flat} - u^2_{flat}\|_2$。这是把 chunk 当 vector 的最简选择。但 end-effector pose + gripper 这种 action 里,Euclidean distance 未必反映 *behavioral similarity* — 例如两个 chunk 都"approach basket from left",但中间 trajectory 不同,Euclidean 可能差很远但 semantically 等价。

更聪明的选择可能是:
- 用 VLA 的 latent representation 当 metric (像 [SURF](https://arxiv.org/abs/2203.13860) style embedding-based retrieval)
- 用 DTW (Dynamic Time Warping) 处理 trajectory
- 用 contrastive learning 训一个 chunk similarity

这是个明显可改进 knob,审稿可能会问。

**(b) $Q \equiv 0$ 的放弃是否可惜?**

paper 主动放弃 value function 是 *当前* 限制,不是 design virtue。一旦 community 有 reliable multi-task value function (像 [VLM as in-context value learner, Ma et al. 2024](https://arxiv.org/abs/2310.09856)),可以 plug 回 PUCT 公式:

$$\text{SCORE} = Q(v, u^i) + c \cdot \psi_{\Phi_v}(u^i) \cdot \frac{\sqrt{N(v)}}{1+N(v,u^i)}$$

这会让 search 更聚焦。但 paper 的实验说明: 仅靠 prior + visit count 已经够强。

**(c) World model 的 fidelity 是 core bottleneck**

paper §VI 承认依赖 accurate simulator。Real-world deploy 时 sim-to-real gap 会怎样? 这里 receding-horizon 结构帮了大忙 — 每 $H$ 步重新拿 observation 重启 search,等于持续纠偏。但要换成 learned world model 时,paper 没做实验。

直觉上,如果用 [Genie 2](https://deepmind.google/technologies/generative-interactive-environment/genie-2/) 这种 generative world model,误差是 *frame-level* 而非 *state-level*;VLA 在 frame 上做 inference 反而 robust,所以这套 VLAPS + generative world model 是个未实验但很有想象力的方向。

**(d) 与 RL post-training 的关系**

LLM 上现在 RL post-training 主流 ([DeepSeek-R1](https://arxiv.org/abs/2501.12948), [o1-style reasoning](https://arxiv.org/abs/2502.17419))。Robot 上 RL post-training VLA 难得多 (sample efficiency、sim-to-real、reward design)。VLAPS 给了条 alternative: 不训练,只 inference-time search。可以理解为 "test-time RL surrogate"。

但更激进的版本: 用 VLAPS 在 inference 时 generate 的成功 trajectory 当作 *self-generated demonstration*,再 finetune VLA。这等价于 AlphaGo 的 *self-play training loop* 移植到 robot — 你可能对这条最感兴趣。从 VLAPS 到 "AlphaZero for VLA" 的距离,主要是 *self-generated data → policy distillation* loop 没闭合,paper 没做。

**(e) Adaptive compute 的 scaling law**

paper Table I 显示: VLA 10k → 200k 期间,search time 从近 600s timeout 降到十几秒。这给出一个 scaling law intuition: search cost $\propto 1 / \text{VLA quality}$,准确形式可能更像 $\propto (1 - \text{success rate}) \cdot \text{avg depth to recovery}$。如果画出 search-time vs VLA-success-rate 曲线,极可能是 sigmoid-ish 衰减 — 这跟 LLM 上 test-time compute vs base model capability 的 scaling law 会有惊人平行。值得做一个 clean 版本。

**(f) Failure mode: Object suite 上 VLA 200k 比 100k 差**

Object 100k: VLA 32% → VLAPS 82%;Object 200k: VLA 19% → VLAPS 54%。200k 反而更差,说明 finetune over-fit 到训练 demonstration,VLA 在 200k 输出分布更窄,prior 集中度反而过高 ($\beta_\Phi$ 太尖锐) — 这能解释 VLAPS 也跟着跌。Fix 思路: 让 $\alpha$ 跟 VLA uncertainty 联动 (高 uncertainty 时降低 $\alpha$),或加 entropy regularization 在 VLA 输出上。

---

## 7. 局限与 future work (paper §VI + 我的延伸)

paper 自陈:
- 依赖 accurate world model (simulator 或 learned)
- VLA inference 慢导致 planning latency 高
- 可优化: batch VLA query、parallel expansion/rollout、调 branching factor 与 max depth、VLA quantization/distillation

我加几条延伸:
- **Value function integration**: plug in [VLM as value learner](https://arxiv.org/abs/2310.09856) 类工作,补回 $Q$ 项
- **World model = Genie 2 / video model**: 把 search 从 state-space 推到 latent 或 frame space
- **Self-play distillation**: 用 VLAPS 生成成功轨迹当 finetune data,闭合 AlphaZero-style loop
- **Adaptive $\alpha$**: 让 VLA uncertainty 调节 prior 锐度,缓解 over-fit 时 prior 太尖的问题
- **Distance metric learning**: 用 contrastive 学 chunk similarity,替代 Euclidean
- **Multi-task Φ library**: 现在 $\Phi$ 是 LIBERO-specific,跨 task suite 的 universal $\Phi$ 需要更大库与更好 retrieval

---

## 8. 一句话总结直觉

VLAPS 把 VLA 当成 *fast System-1 prior*,把 MCTS 当成 *System-2 lookahead*,用 world model 当 *imagination sandbox*。VLA 输出 anchor 在哪儿搜、搜多深、prior 在 $\Phi$ 里指哪儿;MCTS 用 visit count + VLA prior 做 explore/exploit 平衡;world model 提供 "假如我这么做,环境会怎样"。三者拼起来,93M Octo 在 LIBERO 上能 match 3.3B π0-FAST,且 search time 自适应 — VLA 弱时多搜、强时少搜。这给 robotics 带来了 *test-time compute scaling law* 的第一个 concrete instantiation。

---

## References (web links)

- LIBERO benchmark: https://libero-project.github.io/
- Octo model: https://octo-models.github.io/
- OpenVLA: https://openvla.github.io/
- π0: https://www.physicalintelligence.company/blog/pi0
- π0-FAST: https://arxiv.org/abs/2501.09747
- RT-1: https://robotics-transformer1.github.io/
- RT-2: https://robotics-transformer2.github.io/
- PaLM-E: https://palm-e.github.io/
- Hi Robot: https://hi-robot.github.io/
- ECoT (Robotic Control via Embodied Chain-of-Thought): https://embodied-cot.github.io/
- AlphaGo Zero / AlphaZero: https://www.nature.com/articles/nature24270
- MuZero: https://www.nature.com/articles/s41586-020-03051-4
- DreamerV2: https://arxiv.org/abs/2010.02193
- TD-MPC2: https://tdmpc2.github.io/
- Bertsekas MPC-RL unified: https://doi.org/10.1016/j.ifacol.2024.09.056
- Chain-of-Thought: https://arxiv.org/abs/2201.11903
- Self-Consistency: https://arxiv.org/abs/2203.11171
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Hao et al. "Reasoning with Language Model is Planning with World Model": https://arxiv.org/abs/2305.14992
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Test-time Computing survey: https://arxiv.org/abs/2501.02497
- Survey of reasoning LLMs: https://arxiv.org/abs/2502.17419
- ACT (Action Chunking with Transformers, ALOHA): https://tonyzhaozh.github.io/aloha/
- SayCan: https://say-can.github.io/
- LGMCTS: https://arxiv.org/abs/2407.07018
- Video Language Planning: https://video-language-planning.github.io/
- Reflective Planning (Feng et al. 2025): https://arxiv.org/abs/2502.16707
- Genie (DeepMind): https://arxiv.org/abs/2402.15595
- Genie 2: https://deepmind.google/technologies/generative-interactive-environment/genie-2/
- Isaac Gym: https://arxiv.org/abs/2108.10470
- Robosuite: https://arxiv.org/abs/2009.12293
- Hubert et al. "Learning and planning in complex action spaces" (MuZero extension): https://arxiv.org/abs/2104.06378
- VLM as in-context value learners (Ma et al. 2024): https://arxiv.org/abs/2310.09856
- MCTS survey (Browne et al.): https://arxiv.org/abs/1407.7186
- PaliGemma: https://arxiv.org/abs/2407.07726

---

如果你想接着聊 — 我特别好奇你怎么看 (1) VLAPS + Genie 2 这种 *generative world model 当 imagination sandbox* 路线,和 (2) 把 VLAPS 的成功轨迹做 *self-distillation* 闭合 AlphaZero-style loop 这条路 — 后者是你熟悉的 territory,在 robotics 上还没人完整跑通过。
