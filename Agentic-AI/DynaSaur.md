---
source_pdf: DynaSaur.pdf
paper_sha256: 62575cc7fc1bc22c9d924e69112cf07c75f8bb44a74b56562d84851fa4c0b991
processed_at: '2026-08-04T01:05:12-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DynaSaur 用人话讲

OK 让我用更接地气的方式重新捋一遍。Karpathy 你肯定一眼就看出这篇 paper 在干嘛, 但我想把 intuition 掏出来讲透。

## 一句话总结

现在的 LLM agent 就像一个只给你 13 把螺丝刀的修理工 — 不管遇到啥问题, 只能从这 13 把里挑。DynaSaur 说, 去你的, 我直接去 Home Depot 现场造工具, 用完还存仓库下次接着用。

## 问题在哪 — Fixed Action Space 的天花板

你 ReAct loop 跑得好好的, $\mathcal{A} = \{a_1, \ldots, a_n\}$ 就那么十几个 action, agent 每个 step 从里面选一个。听起来 fine, 真用起来发现:

**第一, 覆盖率永远不够**。你给个 `inspect_file_as_text` 工具读 Excel, 它返回 Markdown。结果 GAIA 题目问你 "这个 cell 是什么颜色" — Markdown 里根本没颜色信息。Agent 当场傻眼, 只能反复调同一工具撞墙直到 max iteration。这就是 Figure 5 那个 Excel map 的 case, 特别典型。

**第二, 人力根本 enumerate 不完**。你想想 GAIA 这种 benchmark, 任务横跨 web browsing, Excel parsing, PDF reading, audio transcription, video understanding, symbolic reasoning... 你想 hand-craft 一套覆盖所有的 toolset? 几个人几个月都搞不完, 搞完了还有 edge case。

**第三, 没 learning**。每次跑新 task 都从 primitive action 拼一遍, 上次跑类似 task 积累的经验全丢了。这跟你骂过的 "LLM 是 amnesia stateless machine" 一个意思 — 得给它个 external memory。

DynaSaur 三个问题一起解决。

## 核心招数 — 用 Python 当 Action Language

这一步看着 trivial, 其实是 paper 最聪明的 move。

为啥用 Python 不用 JSON 或 text? 两个 property:

**Generality**。Python 是 Turing-complete 的, 理论上能 express 任何 computable function。你给 agent 一个 Python interpreter, 它的 action space $\mathcal{A}^*$ 直接变成 "all valid Python programs"。这是 infinite action space, 而且是真 infinite — 任意复杂任务都能 express。

**Composability**。Function call function, function return value 喂给另一个 function, 这是 Python 天然支持的。JSON action 想组合? 你得自己设计一套 schema, 设计完了还不灵活。Python 三十年生态把这事彻底解决了。

更狠的一点 — agent 写 `import openpyxl` 的时候, 它不是在从零造轮子, 它在 reuse 人类几十年积累的整个 PyPI 生态。这等于免费送了它一个 "pretrained tool library"。Code as Policies 早就用过这招, 但 DynaSaur 把它放进了 agent loop 内部, 还能 accumulate。

## 怎么避免 Context Window 爆炸 — Action Retrieval

跑着跑着 $\mathcal{A}^g$ 里积累了几百个 function, 全塞 prompt 里肯定爆。DynaSaur 的做法:

把 action 集合拆两半:
- $\mathcal{A}^u$: human-designed, 永远在 prompt 里 (Table 5 那 13 个, 从 AutoGen 抄的)
- $\mathcal{A}^g$: generated, 通过 retrieval 按需调用

Retrieval 函数 $R(q, k)$, 输入 query $q$ 和 top-k 的 $k$, 输出 $\mathcal{A}^g$ 的子集。具体怎么做? 让 LLM 在写每个 function 的时候顺手写个 docstring, 然后用 `text-embedding-3-large` embed 这些 docstring 当 index。Query 来了 embed 一下, 算 cosine similarity, 取 top-10。

最骚的设计: $R$ 自己也被塞进 $\mathcal{A}^u$, agent 自己决定什么时候 retrieval。这是 meta-action — agent 能 "查工具箱" 本身就是一个工具。跟 Reflexion 让 agent 自己决定何时 reflect 一个套路, 把 meta-cognition 也 explicit 化。

公式写出来就是:

$$R: \mathcal{Q} \times \mathbb{N} \to 2^{\mathcal{A}^g}$$

$\mathcal{Q}$ 是 query 字符串空间, $\mathbb{N}$ 是正整数 (k 的取值), $2^{\mathcal{A}^g}$ 是 $\mathcal{A}^g$ 的 power set。这函数本质上是个 neural index, 把自然语言 query 映射到 code library 的子集。

## Accumulate 的细节

ReAct loop 每个 step 采样:

$$(h_t, a_t) \sim \pi_\theta(a_t \mid \mathcal{A}^u, u, c_{t-1})$$

变量意思:
- $\pi_\theta$: LLM policy, 参数 $\theta$ 就是 LLM 的 weights
- $h_t$: 第 $t$ 步的 thought (reasoning text)
- $a_t$: 第 $t$ 步的 action (Python code snippet)
- $u$: 当前 task
- $c_{t-1} = \{(h_1, a_1, o_1), \ldots, (h_{t-1}, a_{t-1}, o_{t-1})\}$: 之前的 interaction history, 每个元素是 (thought, action, observation) 三元组

$a_t$ 执行完得到 $o_t$, 更新 history。然后关键一步:

$$\mathcal{A}_t^g = \mathcal{A}_{t-1}^g \cup f(a_t)$$

$f(a_t)$ 是 $a_t$ 这段 code 里定义的所有 functions 的集合。注意是 **并集** — 只有新 function 才会被加进去, 而且 paper 强调只有 **successfully executed** 的才积累。这是天然的 filtering, 避免 buggy code 污染 library。

一个重要的实验设计选择: training 时 accumulate, testing 时 freeze action set。这保证 test set 里每个 task 独立, 不会因为先跑的 task 积累了 lucky action 让后面 task 受益, 不会污染 metric。

## 最有意思的 Metric — Action Coverage

paper 提了个新 metric, intuition 极其漂亮。给你 action set $\mathcal{A}$, task $u$, 答案 $y$, 怎么衡量 $\mathcal{A}$ 对 $u$ 的覆盖程度?

定义 coverage:

$$C(\mathcal{A}, u) \stackrel{\text{def}}{=} \underset{\tau \sim \pi_\theta(\cdot \mid \mathcal{A}, u)}{\mathbb{E}} \left[ 1 - \frac{1}{|\tau|} \mathbf{1}[o_T = y] \cdot \left| \{(a_i, o_i) \in \tau : a_i \notin \mathcal{A}\} \right| \right]$$

逐个拆:
- $\tau$: 采样的 trajectory, 一串 (action, observation) 对
- $\tau \sim \pi_\theta(\cdot \mid \mathcal{A}, u)$: 用 policy $\pi_\theta$ 在给定 $\mathcal{A}$ 和 $u$ 的条件下采样 trajectory
- $|\tau|$: trajectory 长度 (走了多少步)
- $o_T$: 最后一步的 observation (agent 提交的 final answer)
- $y$: ground-truth 答案
- $\mathbf{1}[o_T = y]$: indicator, 任务成功为 1, 失败为 0
- $\{(a_i, o_i) \in \tau : a_i \notin \mathcal{A}\}$: trajectory 里那些不在 $\mathcal{A}$ 中的 actions (即 agent 临时新造的)
- 最外层 $\mathbb{E}$: 对 trajectory 分布取期望 (实际操作就 sample 一条近似)

Intuition 拆开看: 如果任务成功 ($\mathbf{1} = 1$), coverage = $1 - \frac{\text{新造的 action 数}}{\text{总步数}}$。完全用现有 action 解决, coverage = 1; 用了一半新造的, coverage = 0.5。如果任务失败 ($\mathbf{1} = 0$), 整个项被 mask 掉, coverage = 1 (但这是退化的, 因为失败 case 不应该 contribute)。

这 metric 巧妙在哪? 它 conditional on success — 只算成功 trajectory 里的 reuse 比例。直接量化了 "action library 有多大用"。

Figure 3 的结果有信息量:
- GAIA 上 human-designed tools 加进来 (红虚线) coverage 大跳 — 因为 GAIA 大量需要 web browsing 和 file parsing, human tools 正对口
- MATH 上 human tools 加进来几乎没动 — 数学题你 web search 啥啊? coverage 提升主要靠 generated actions
- 说明 coverage 跟 domain alignment 强相关, human tools 不是万能药

## 实验结果 — 涨点来自哪

主表 Table 1, GPT-4o 上:
- No Pipeline (纯 GPT-4o): 9.30% avg
- HF Agent (replicated): 29.00%
- **DynaSaur: 38.21%**

Level 3 (最难, 多步 reasoning + 各种 file type) 上拉开差距最明显 — DynaSaur 18.37%, HF Agent 14.58%。复杂任务上 dynamic action creation 价值放大, 因为复杂任务更容易触发 predefined tools 覆盖不到的 case。

Table 2 其他 benchmark:
- AIME (数学竞赛): 31.71% vs 20.00% (HF Agent), 涨 58%
- GPQA (博士级问答): 54.00% vs 38.00%, 涨 42%

AIME 这种数学竞赛题涨这么多, intuition 是啥? 我的 take: agent 可以把 reasoning chain 编码成可执行 code, 而非依赖 LLM 内部 CoT。Code 是 **executable reasoning**, 跑出来的结果就是 ground truth, 不用 LLM 自己一步一步算错。这其实跟 PAL (Program-Aided Language models, Gao et al., 2023) 思路一脉相承, 但 DynaSaur 把它放进 agentic loop 还能 accumulate。

## Ablation 的关键 Takeaway

Table 3 三个 component:
- IA (Initial human Actions): 13 个 AutoGen tools
- AI (Action Implementation): 能动态写 Python
- AA (Action Accumulation): 跨 episode 存 action

看 row 1 (啥都没, 只有 AI) → row 3 (只有 IA, 没 AI/AA): 21.82 → 35.15, 涨 61%。Human tools 贡献最大, 不意外。

row 3 → row 4 (加 AI): 35.15 → 38.79, 涨 10%。这是 dynamic action creation 的净贡献, 实实在在。

row 4 → row 5 (加 AA): 38.79 → 41.82, 涨 7%。Accumulation 让 action 跨 task 复用, 也实在。

**Level 3 上**: row 1 的 7.69 → row 5 的 26.92, 涨 250%。越复杂的任务, 这套组合拳价值越大。这说明 Level 3 任务大量触发了 predefined tools 不够用的场景, dynamic creation 直接救命。

## Open-Source LLM 上的反直觉发现

Table 4 用 Qwen2.5-32B-Instruct (通用) vs Qwen2.5-Coder-32B-Instruct (代码专用):

通用版 DynaSaur avg 27.88, 代码专用版 24.24。**通用版反而更强**。

作者 hypothesis: commonsense reasoning 比 pure coding 更重要。这跟 αCodium 的发现吻合 — agentic coding 场景下, 先 decide what to do 比 how to code 更关键。Coder 模型可能 overfit 到单文件 code completion, 反而少了点 "agentic decision making" 的灵气。

这给我一个 intuition: **agent 框架里, reasoning policy > code generator**。Code generator 不需要顶级, 因为 Python 容错性其实挺高, 错了 retry 就行。但 reasoning policy 决定 agent 是否走对方向, 错了 retry 都救不回来。

## Error Analysis — 61.91% 的失败是工具不够

paper 用 o1-preview 做 qualitative analysis, 22 个 "Agent A (没 AI) 失败, Agent B (有 AI) 成功" 的 task。结论:

- 61.91%: **工具不够** (12 个完全缺 tool, 1 个 human tool 返回 incomplete info)
- 9.52%: instruction following 问题 (单位错了之类的)
- 28.57%: 其他 (web search 找不到信息, 卡住)

关键: **所有 type-1 error 都被 Agent B 用 custom action 解决**。这是 dynamic action implementation 的 smoking gun — 工具不够时它能现场造, 这是 fixed action space 永远做不到的。

那个 Excel 颜色的 case 特别生动: AutoGen 的 `inspect_file_as_text` 把 Excel 转 Markdown, 颜色信息丢了。Agent A 反复撞墙。Agent B 试了一下发现不行, 立马 `import openpyxl`, 直接读 cell.style.fill.color, 写个 navigation function, 搞定。

这 example 的 intuition: **predefined tools 永远有 blind spot**, dynamic creation 是 graceful degradation 的机制。

## Generated Actions 长啥样

174 个 actions 总共, 分类 (Figure 4):
- 23.75%: 信息提取/parsing
- 22.50%: 计算
- 12.50%: 搜索
- 8.75%: 条件检查
- 32.50%: 杂项

Cyclomatic Complexity (McCabe 1976 那个 $M = E - N + 2P$ 公式, $E$ edges, $N$ nodes, $P$ connected components):
- Generated: 平均 3.06
- Human-written: 平均 3.72

LLM 写的 code 反而比人的简单。我的 take: LLM 倾向写 short focused function, 人写的 tool 经常 overengineer 加各种 feature。简单 function 更易 reuse, 这其实是个 virtue。

Figure 10 有 failure case — 比如 `calculate_food_sales` 里 hardcode 了 `row[1:6]`, 这种不能 generalize。这是 LLM 写 code 的常见毛病, paper 也认了。

## 我对这 paper 的整体判断

**Elegant, timely, 系统集成 > algorithmic novelty**。

paper 没发明新 algorithm — ReAct, retrieval, Python execution, skill library 都是已有 pieces。Voyager 早就做过 Minecraft 版, LATM/CREATOR/CRAFT 做过受限场景版。DynaSaur 的贡献是把这套组装起来, push 到 GAIA 这种 real-world benchmark, 并用扎实的实验证明它 work。

更重要的贡献是 conceptual: **它给 community 一个清晰的 baseline, predefined action space 是 artificial constraint, 移除它 agent 显著变强**。这跟 Open-Ended Learning 哲学一致 — 别 hand-design, 让 agent 自己 explore。

回到你 Karpathy 一直讲的 "LLM OS" framing:

- LLM 是 CPU
- Context window 是 RAM
- $\mathcal{A}^g$ 是 disk (skill library, via retrieval loaded into RAM)
- Python interpreter 是 peripheral (实际执行 I/O 和 compute)
- ReAct loop 是 instruction cycle

DynaSaur 给这个 OS 装了个 **self-modifying filesystem** — agent 能自己写新的 system call 并 persist。这是 bootstrapping AGI 的关键 primitive。

更深一层: 这是 **function space 的 continual learning**。传统 continual learning 在 weight space (EWC, replay buffer), 又慢又容易 catastrophic forgetting。DynaSaur 在 function space — 新技能以 Python function 形式加进 library, 不用 retraining, 不用 gradient, 不用 replay。Declarative + symbolic 的 learning, 跟 connectionist 形成有趣对比。

这让我想到 DreamCoder (Ellis et al., 2021, https://arxiv.org/abs/2006.08237) 的 library learning — 通过 building domain-specific language 不断 abstraction。DynaSaur 用 Python 当 DSL, 用 LLM 当 program prior, bypass 了 DreamCoder 的 expensive search。LLM pretraining 已经 compress 了 GitHub 上所有公开 code, 这是个 massive speedup。

## 局限 — paper 没正经讨论的

**Cost**。每个 action 都是 code generation + execution, token 量远高于 fixed action。Paper 没报 inference cost 对比, 这其实是个 big deal。

**Latency**。Code gen + interpreter startup + exec, 每 step 秒级。Real-time 场景没法用。

**Verification**。"Execute successfully" 不等于 "semantically correct"。LLM 写的 function 能 run, 但返回值可能错。Paper 没有 verification 机制, 这是个 hole。

**Compositionality ceiling**。Python 理论上 composable, 但 LLM 能否真正 reuse 自己之前写的 complex function (而非重写)? Action Coverage 是间接 measure, 直接 measure reuse rate 会更有说服力。

**Safety at scale**。174 个 action 还能 manually inspect, 174k 个怎么办? Ethics section 说要 sandbox, 但 sandboxing 解决不了 "agent 写了 syntactically valid 但 semantically harmful 的 function" — 比如它写个 `delete_all_files` 然后 call 它。这需要 capability scoping, 不只是 isolation。

**Generalization 没量化**。Figure 10 显示 failure case (hardcoded values), 但没量化 generated action 跨 task reuse 的实际 rate。Action Coverage 只衡量 trajectory 里的 reuse step, 不衡量 library 里有多少 action 真的被多个 task 用过。

## 我赌的 Future Direction

- **Self-verification**: agent 写完 function 顺手写 unit test 验证, 类似 Self-Refine (Madaan et al., 2023) 套路
- **Action refinement**: failed action 不直接丢, iterative debug 后再 accumulate
- **Hierarchical action library**: 形成层级的 skill DAG, 而非扁平 function set
- **Cross-task transfer**: 学从一个 domain 的 library 迁到另一个 domain
- **Active action creation**: agent 主动判断 "我接下来需要什么 action", 而非被动 trigger by failure
- **SWE-Bench 上的应用**: 这才是真正 interactive 的 coding 场景, action library 能不能积累成 "coding skill library"? 我赌可以, 但需要新设计

## 最后的 Intuition

用你 Software 2.0 (https://karpathy.medium.com/software-2-0-a6c52b1f1d59) 的 framing:

- Software 1.0: 人写 code
- Software 2.0: weights 定义 behavior
- **DynaSaur 是 Software 2.5**: weights 定义 reasoning prior, Python function library 定义 acquired skills, 两者组合成 adaptive agent

这比纯 1.0 (hand-coded tools) 和纯 2.0 (weights only) 都更接近 AGI 的形态。Weights 给你 commonsense 和 code 语法 prior, function library 给你 specific skill 和 compositional structure。

而且这玩意儿 unsupervised — 不需要人标 trajectory, 不需要 reward model, agent 自己跑 task 自己造工具自己积累。这是 open-ended self-improvement 的雏形, 跟你 "LLM OS 会自己写 system call" 的 vision 完全对上。

唯一让人不放心的就是 safety — arbitrary code execution 这个事, 部署到 production 之前必须解决。但作为 research prototype, 它清晰的指向了正确方向: **fixed action space 是 artificial ceiling, 我们应该 break it**。

---

**相关 link**:
- Paper code: https://github.com/adobe-research/dynasaur
- Voyager (Minecraft 版): https://voyager.github.io/
- GAIA benchmark: https://arxiv.org/abs/2311.12983
- ReAct: https://arxiv.org/abs/2210.03629
- Reflexion: https://arxiv.org/abs/2303.11366
- DreamCoder: https://arxiv.org/abs/2006.08237
- Code as Policies: https://code-as-policies.github.io/
- LATM: https://arxiv.org/abs/2305.17126
- CRAFT: https://arxiv.org/abs/2309.17428
- Software 2.0 (Karpathy): https://karpathy.medium.com/software-2-0-a6c52b1f1d59
- LLM OS talk (Karpathy): https://www.youtube.com/watch?v=zjkBMFhNj_g
- SWE-Bench: https://www.swebench.com/
- PAL: https://arxiv.org/abs/2211.10435
- Self-Refine: https://arxiv.org/abs/2303.17651

---

# DynaSaur: 让 LLM Agent 突破 Predefined Actions 的桎梏

这篇 paper 来自 University of Maryland 和 Adobe Research, 核心思想非常 elegant — 用 Python 作为 universal action representation, 让 agent 动态 write code 来 create + accumulate actions, 而非从一个 fixed 的 $\mathcal{A} = \{a_1, \ldots, a_n\}$ 中选择。这与 Voyager (Wang et al., 2023)、LATM (Cai et al., 2023)、CREATOR (Qian et al., 2023) 的精神一脉相承, 但 DynaSaur 把这件事推到了 generalist agent benchmark (GAIA) 上, 并首次让 accumulated action library 真正 scale 起来。

GitHub repo: https://github.com/adobe-research/dynasaur

## 1. Problem Formulation 的关键扩展

传统 LLM agent 的 POMDP 定义为 tuple $(\mathcal{U}, \mathcal{A}, \mathcal{S}, \mathcal{O}, T, Z)$, 其中 $\mathcal{A}$ 是 finite predefined action set。DynaSaur 的关键修改是引入 $\mathcal{A}^*$ — 一个 potentially infinite 的 "all possible actions" 集合, 并定义 **cumulative action set**:

$$\hat{\mathcal{A}}_t = \{a_1, a_2, \ldots, a_t\}$$

这里 $\hat{\mathcal{A}}_t$ 是时间 $t$ 为止所有曾经被 agent 提出过的 actions 的并集。每个新的 $a_t$ 要么是 novel action, 要么是 $\mathcal{A}_{t-1}$ 中已有 actions 的 composition。transition function 也相应扩展:

$$T: \mathcal{S} \times \mathcal{A}^* \to \mathcal{P}(\mathcal{S})$$
$$Z: \mathcal{S} \times \mathcal{A}^* \to \mathcal{P}(\mathcal{O})$$

$\mathcal{P}(\cdot)$ 表示概率分布。这个 formulation 的关键 intuition 是: $\mathcal{A}^*$ 本质上是 **all syntactically valid Python programs** 这一集合 — 它是 Turing-complete 的, 因此 agent 的 capability 不再被 human-designed tools 限制, 而是被 LLM 的 code generation ability 限制。这是一个 fundamental shift — 把 bottleneck 从 "human engineering effort" 转移到了 "LLM's program synthesis capability"。

## 2. 为什么 Python 是正确的 Action Representation

paper 提出两个 criteria:
- **Generality**: 能 express 任意 Turing-computable function
- **Composability**: 自然支持 action 之间的 composition (function calls function)

这与 Liang et al., 2023 的 "Code as Policies" 思路一致, 但 DynaSaur 把它放在了 agent loop 内部而非仅用于 embodied control。Text-based action representation (如 Toolformer, Schick et al., 2023) 或 JSON-based (如 ToolLLM, Qin et al., 2023) 缺乏 composability — 你很难让一个 JSON action 调用另一个 JSON action 并传参数。Python function calls 天然解决了这个问题。

更重要的是, Python ecosystem 本身就是一个巨大的 "pretrained action library" — `openpyxl`, `pandas`, `requests`, `beautifulsoup` 等等。Agent 写 `import openpyxl` 时, 它其实在 reuse 人类几十年积累的 tooling, 这是 JSON action space 永远做不到的。

## 3. Action Retrieval 的设计

一个关键 engineering 问题是: 当 $\mathcal{A}^g$ (generated actions) 越来越大, 不可能全部塞进 prompt。DynaSaur 把 action set 分成两个 subset:

$$\mathcal{A} = \mathcal{A}^u \cup \mathcal{A}^g$$

- $\mathcal{A}^u$: human-designed (always in prompt, 默认 13 个, 见 Table 5)
- $\mathcal{A}^g$: generated (via retrieval)

Retrieval function 定义为:

$$R: \mathcal{Q} \times \mathbb{N} \to 2^{\mathcal{A}^g}$$

$\mathcal{Q}$ 是 query space, $\mathbb{N}$ 是 positive integers (top-k 的 k)。当 agent 调用 $R(q, k)$, 系统用 `text-embedding-3-large` embed query $q$, 与每个 generated action 的 **docstring embedding** 计算 cosine similarity, 返回 top-k。这里 k=10。

关键 design choice: $R$ 本身被放进 $\mathcal{A}^u$, agent 自主决定何时 retrieval。这是 meta-action 设计, 类似 Reflexion (Shinn et al., 2023) 的 self-reflection 机制, 但用于 action discovery 而非 error correction。

## 4. Action Accumulation 的 Pipeline

完整 pipeline (Figure 1):

$$ (h_t, a_t) \sim \pi_\theta(a_t \mid \mathcal{A}^u, u, c_{t-1}) $$

其中 $c_{t-1} = \{(h_1, a_1, o_1), \ldots, (h_{t-1}, a_{t-1}, o_{t-1})\}$ 是 ReAct-style 的 interaction history。$h_t$ 是 thought, $a_t$ 是 action (Python code), $o_t$ 是 observation (execution output 或 error message)。

Action accumulation 的更新规则:

$$\mathcal{A}_t^g = \mathcal{A}_{t-1}^g \cup f(a_t)$$

$f(a_t)$ 表示 $a_t$ 中定义的所有 functions。注意只有 **successfully executed** 的 actions 才会被 accumulate — 这是一个重要的 filtering mechanism, 避免坏 code 污染 library。

一个 subtle 但重要的 design: training 时 accumulate, testing 时 freeze。这确保 test task 之间相互独立, 避免 data leakage。这一点很重要, 因为 GAIA validation set 才有 labels。

## 5. Action Coverage Metric — 一个新颖的 Evaluation 视角

paper 提出一个非常有意思的 metric — **Action Coverage**, 用来衡量 action set $\mathcal{A}$ 对 unseen task $u$ 的覆盖程度:

$$C(\mathcal{A}, u) \stackrel{\text{def}}{=} \underset{\tau \sim \pi_\theta(\cdot \mid \mathcal{A}, u)}{\mathbb{E}} \left[ 1 - \frac{1}{|\tau|} \mathbf{1}[o_T = y] \cdot \left| \{(a_i, o_i) \in \tau : a_i \notin \mathcal{A}\} \right| \right]$$

变量含义:
- $\tau = \{(a_1, o_1), \ldots, (a_T, o_T)\}$: 采样的 trajectory
- $|\tau|$: trajectory 长度 (步数)
- $o_T = y$: 最终 observation 等于 ground-truth answer (即 task 成功)
- $\mathbf{1}[o_T = y]$: indicator function, task 成功为 1, 否则为 0
- $|\{(a_i, o_i) \in \tau : a_i \notin \mathcal{A}\}|$: trajectory 中不在 $\mathcal{A}$ 里的 actions 数量 (即新创建的)

intuition: Coverage = 1 - (新创建 action 数 / 总步数) × 成功 indicator。如果 agent 完全靠现有 action 解决任务且成功, coverage = 1; 如果 agent 需要创建新 action, coverage 下降; 如果 task 失败, coverage = 1 (因为 indicator = 0)。这个 metric 的巧妙之处在于它 conditional on success — 只衡量成功 trajectory 中的 reuse ratio。

Figure 3 的结果非常有启发性: GAIA 上 human-designed tools 加入后 coverage 大幅上升 (因为 GAIA 大量需要 web browsing + file parsing), 但 MATH 上 coverage 主要靠 generated actions 提升 (因为 human-designed tools 对数学没用)。这反映了 **domain alignment** 对 action library utility 的影响。

## 6. 实验结果深度分析

### 6.1 GAIA Benchmark (Table 1)

GPT-4o backbone:
| Method | Level 1 | Level 2 | Level 3 | Avg |
|---|---|---|---|---|
| No Pipeline | 13.98 | 8.81 | 2.04 | 9.30 |
| Sibyl (repl.) | 38.71 | 24.53 | 10.20 | 26.58 |
| HF Agent (repl.) | 39.78 | 27.04 | 14.58 | 29.00 |
| **DynaSaur** | **51.61** | **36.48** | **18.37** | **38.21** |

Level 3 (最难) 上 DynaSaur 18.37 vs HF Agent 14.58, 提升 ~26%。这印证了 paper 的核心 claim: 在 long-horizon, complex task 上, dynamic action creation 的优势放大 — 因为这些 task 更可能触发 predefined tools 无法覆盖的 edge case。

### 6.2 其他 benchmarks (Table 2)

| Dataset | No Pipeline | Sibyl | HF Agent | DynaSaur |
|---|---|---|---|---|
| MATH | 77.86 | 74.29 | 80.71 | **82.14** |
| TabMWP | 95.71 | 95.00 | 96.43 | **97.14** |
| AIME | 13.00 | 20.00 | 20.00 | **31.71** |
| GPQA | 48.00 | 46.00 | 38.00 | **54.00** |

AIME (American Invitational Mathematics Examination) 上 31.71 vs 20.00, 提升 ~58%! GPQA (graduate-level Google-proof Q&A) 上 54.00 vs 38.00, 提升 ~42%。这两个 benchmark 都是 reasoning-heavy 的, 说明 dynamic action creation 不仅仅是 "tool augmentation", 而是给 agent 一种 **programmatic reasoning** 的能力 — 它可以把 reasoning chain 编码成可执行的 code, 而非依赖 LLM 的 internal CoT。

### 6.3 Ablation Study (Table 3) — 三大组件的贡献分解

| # | AA | AI | IA | Level 1 | Level 2 | Level 3 | Avg |
|---|---|---|---|---|---|---|---|
| 1 | × | ✓ | × | 33.96 | 18.60 | 7.69 | 21.82 |
| 2 | ✓ | ✓ | × | 35.85 | 19.77 | 7.69 | 23.03 |
| 3 | × | × | ✓ | 43.40 | 37.21 | 11.54 | 35.15 |
| 4 | × | ✓ | ✓ | 47.17 | 40.70 | 15.38 | 38.79 |
| 5 | ✓ | ✓ | ✓ | **49.06** | **41.86** | **26.92** | **41.82** |

- AA = Action Accumulation
- AI = Action Implementation (dynamic code generation)
- IA = Initial Actions (human-designed tools)

关键 insight:
- Row 1 → Row 2 (加 AA, 无 IA): 仅 +1.21% (因为没 human tools, generated actions 又少, accumulate 价值有限)
- Row 3 → Row 4 (加 AI, 有 IA): +3.64% (dynamic implementation 在有 base tools 后更有用)
- Row 4 → Row 5 (加 AA): +3.03% (accumulation 让 action 跨 episode 复用)

Row 5 vs Row 1: **+20.00%** (21.82 → 41.82), 几乎翻倍。三大组件中, IA (human-designed tools) 贡献最大 (+13.33%), 但 AI + AA 的组合贡献 (+6.67%) 也不可忽视, 尤其在 Level 3 上 (7.69 → 26.92, **+19.23%**)。这说明 **越复杂的 task, dynamic action creation 的相对价值越大**。

### 6.4 Open-source LLM 实验 (Table 4)

| Model | Method | Level 1 | Level 2 | Level 3 | Avg |
|---|---|---|---|---|---|
| Qwen2.5-32B-Instruct | Sibyl | 24.53 | 10.47 | 0.00 | 13.33 |
| Qwen2.5-32B-Instruct | HF Agent | 26.42 | 11.63 | 3.85 | 15.15 |
| Qwen2.5-32B-Instruct | **DynaSaur** | **35.85** | **30.23** | **3.85** | **27.88** |
| Qwen2.5-Coder-32B-Instruct | DynaSaur | 35.85 | 20.93 | 11.54 | 24.24 |

一个 counterintuitive 的发现: **general-purpose Qwen2.5-32B-Instruct 比 code-specialized Qwen2.5-Coder-32B-Instruct 表现更好** (27.88 vs 24.24)。作者 hypothesis 是 commonsense reasoning 能力比 pure coding 能力更重要 — 因为 agent 需要先 decide what action to take, 然后才 code it。这与 αCodium (CodiumAI, 2024) 的发现类似: 在 agentic coding 场景, planning 和 reasoning > pure code generation。

## 7. Error Analysis — 为什么 Dynamic Action Implementation 有效

paper 用 o1-preview 做 qualitative analysis (Figure 6 的 prompt), 对 22 个 "Agent A (no AI) failed, Agent B (with AI) succeeded" 的 task 做错误分类 (Figure 2):

- **Reason 1 (insufficient tooling)**: 61.91% (13 cases)
  - 12 cases: 完全缺乏必要 tool
  - 1 case: human-designed tool 返回 incomplete info (Excel 颜色问题)
- **Reason 2 (instruction following)**: 9.52%
- **Reason 3 (other)**: 28.57%

关键发现: **所有 type-1 error 都被 Agent B 通过 custom action 解决**。这直接证明了 dynamic action implementation 的核心价值 — 填补 predefined tools 的 coverage gap。

## 8. Case Study (Figure 5) — Excel Map Navigation

这个 case study 极具启发性:

**Task**: 加载含 map 的 Excel 文件, 按 movement rules 导航, 第 11 turn 后返回 cell color。

**Agent A (no AI)**: 反复调用 `inspect_file_as_text` (AutoGen tool), 但该 tool 用 Markdown 表示 Excel, **丢失了 cell color 信息** → 卡死, 达到 max iteration。

**Agent B (with AI)**: 先尝试同样的 tool, 发现 incomplete info 后, **自己写 `openpyxl` 代码读取 cell color**, 然后实现 navigation function, 成功完成。

这个 example 精准展示了 paper 的核心 thesis: human-designed tools 总会有 unforeseen edge case, 而 dynamic action implementation 提供了 **graceful degradation** 机制。这让我想到 software engineering 中的 "defense in depth" — 多层 fallback。

## 9. Generated Action Statistics

总共 174 个 actions:
- 80 个 training 时积累 (from 165 examples)
- 94 个 testing 时新建 (from 300 examples)

按功能分类 (Figure 4):
- 23.75%: 信息提取/parsing
- 22.50%: 计算
- 12.50%: 搜索
- 8.75%: 条件检查
- 32.50%: miscellaneous (file conversion, counting 等)

**Cyclomatic Complexity** (McCabe, 1976):
- Generated actions: 平均 3.06
- Human-authored: 平均 3.72

Cyclomatic complexity 公式: $M = E - N + 2P$, 其中 $E$ = edges, $N$ = nodes, $P$ = connected components。值 < 10 被认为 maintainable。Generated code 反而比 human code **更简单**, 这有点反直觉 — 可能因为 LLM 倾向写 short, focused functions, 而 human tools 更 feature-rich。这也意味着 generated actions 更易 reuse。

## 10. 一个 Critical Perspective — 安全性

paper 的 Ethics Statement 很坦诚: "this approach is not advisable for real-world deployment due to potential security risks"。这其实是个 major concern — agent 能 execute arbitrary Python 等于给了它 shell access。虽然作者提到 safety filter, formal verifier, isolated environment, read-only FS, 但这些都是 mitigation 而非 solution。

参考 OpenAI 的 Code Interpreter 和 Anthropic 的 Computer Use, 行业正在探索 sandboxed execution environments (如 Firecracker microVM, gVisor, WASM runtime)。对 DynaSaur 这类框架, 我觉得 **capability scoping** 比 sandboxing 更根本 — 应该限制 agent 能 import 什么 module, 能 access 什么 file path。这是 future work 的重要方向。

## 11. 与 Related Work 的关系图谱

- **Voyager** (Wang et al., 2023, NVIDIA + Caltech): Minecraft agent, 也用 code as action + skill library。但 Voyager 限定在 Minecraft, DynaSaur 是 generalist。Voyager 的 skill library 用 embedding做retrieval，DynaSaur继承了这点。https://voyager.github.io/
- **LATM** (Large Language Models as Tool Makers, Cai et al., 2023): 同一精神, 但限定 single-step task。https://arxiv.org/abs/2305.17126
- **CREATOR** (Qian et al., 2023): disentangle abstract/concrete reasoning via tool creation, 限定 math + Q&A。https://arxiv.org/abs/2305.17126
- **CRAFT** (Yuan et al., 2023): customize LLMs by creating + retrieving from toolset, 限定 VQA + math。https://arxiv.org/abs/2309.17428
- **Code as Policies** (Liang et al., 2023): embodied control 用 code, 但不 accumulate。https://code-as-policies.github.io/
- **Toolformer** (Schick et al., 2023): self-supervised tool use, 但 action space 是 fixed。https://arxiv.org/abs/2302.04761
- **ReAct** (Yao et al., 2023): thought-action interleaving, DynaSaur 的 prompting 框架基础。https://arxiv.org/abs/2210.03629
- **Reflexion** (Shinn et al., 2023): verbal RL via self-reflection, 与 action accumulation 互补。https://arxiv.org/abs/2303.11366

DynaSaur 的独特定位: **first generalist LLM agent that implements + accumulates actions on real-world benchmark (GAIA)**。这是从 "toy domain" 到 "real-world agent" 的关键 step。

## 12. Build Your Intuition — 这是 AGI 路径上的一个重要 primitive

回到 Karpathy 你一直强调的 "LLM OS" 思想 (https://www.youtube.com/watch?v=zjkBMFhNj_g) — DynaSaur 实质上是把这个 vision 具体化了:

1. **LLM 是 CPU**: 执行 reasoning + decision making
2. **Context window 是 RAM**: 装当前 task + retrieved actions
3. **$\mathcal{A}^g$ 是 disk**: persisted skill library, via retrieval loaded into RAM
4. **Python interpreter 是 peripheral**: 实际执行 I/O, computation
5. **ReAct loop 是 instruction cycle**: fetch-thought-decode-act-observe

这个 framing 让 DynaSaur 的贡献非常清晰: 它给 LLM OS 装了一个 **self-modifying filesystem** — agent 可以自己写新的 "system call" 并 persist 下来。这是 bootstrapping 的关键 primitive。

更深层的 intuition: **action accumulation 是 unsupervised 的 continual learning**。传统 continual learning (Kirkpatrick et al., 2017, EWC) 在 weight space, 需要 replay buffer, regularization。DynaSaur 在 **function space** 做 continual learning — 新技能以 Python function 形式 added to library, 无需 retraining, 无需 replay, 无需 gradient。这是一种 declarative + symbolic 的 learning, 与 connectionist 的 weight update 形成有趣对比。这让我想到 Neural Turing Machine (Graves et al., 2014) 和 Differentiable Neural Computer — 但 DynaSaur 用真正的 Python REPL 作为 external memory, 完全 bypass gradient。

另一个联想: 这与 **program synthesis** 社区 (DreamCoder, Ellis et al., 2021, https://arxiv.org/abs/2006.08237) 的 "library learning" 极其相似。DreamCoder 通过 building domain-specific language 不断 abstraction, DynaSaur 通过 Python function library 做同样的事。但 DynaSaur 用 LLM 替代了 DreamCoder 的 expensive search — LLM 是一个已经 pretrained 了的 "program prior"。这是一个 **massive speedup**, 因为 LLM 的 pretraining 已经 compress 了 GitHub 上所有公开 code。

还有一层 connection: **Anthropic 的 Tool Use / Function Calling** (https://docs.anthropic.com/en/docs/build-with-claude/tool-use) 和 OpenAI 的 Function Calling 本质上是 predefined action space。DynaSaur 把这扩展成 open action space。可以预见, 下一代 function calling API 可能会支持 "function definition as part of generation" — 即模型不仅能 call existing function, 还能 define new function inline。DynaSaur 是这个方向的学术 prototype。

## 13. 局限性与 Future Directions

paper 没有充分讨论的:

1. **Cost**: 每个动作都是 Python generation + execution, token cost 远高于 fixed action。Paper 没报 inference cost 对比。
2. **Latency**: code generation + interpreter startup + execution, 每 step 可能秒级, 不适合 real-time。
3. **Verification**: 没有 formal verification generated code 是否 correct, 只看是否 "execute successfully" — 但能 run 不代表 semantic correct。
4. **Compositionality ceiling**: 虽然 Python 理论上 composable, 但 LLM 能否真正 compose 自己之前写的 complex function (而非重新写)? 这点 paper 没做 ablation。
5. **Safety at scale**: 174 个 actions 还能 manually inspect, 但 174k 个时怎么办? 需要 automated safety auditing。
6. **Generalization of generated actions**: Figure 10 显示失败 case (hardcoded values), 但没量化 reuse rate。Action Coverage 是间接 measure, 直接 measure 会更有说服力。

Future work 方向 (我的 speculation):
- **Self-verification**: 让 agent 写 unit test 验证自己的 action
- **Action refinement**: 类似 Self-Refine (Madaan et al., 2023), 对 failed action 做 iterative refinement
- **Hierarchical action**: 引入 sub-goal decomposition, 让 action library 形成 DAG
- **Action deduplication**: semantic similarity 检测重复 action 并 merge
- **Cross-task transfer**: 学习从一个 domain 的 action library 迁移到另一 domain
- **Active action creation**: agent 主动判断 "我需要什么 action" 而非被动 trigger by failure

## 14. 我的 Takeaway

DynaSaur 是一个 elegant 且 timely 的工作。它的核心 contribution 不在 algorithmic novelty (ReAct + retrieval + Python execution 都是已有 pieces), 而在于 **系统集成 + 实证验证 + 概念清晰**。它把 "LLM as universal action proposer" 这件事 push 到了 GAIA 这种 real-world benchmark, 并展示了显著 gain (38.21% avg, 比 HF Agent 高 9.21 个百分点)。

更重要的是, 它给 community 一个清晰的 baseline: **predefined action space 是 artificial constraint, 移除它能让 agent 显著变强**。这与 RL 中的 "Open-Ended Learning" (POET, Enhanced POET, ISLER et al.) 哲学一致 — 不要 hand-design environment, 让 agent 自己 explore。

下一步值得 follow 的方向: 把 DynaSaur 放到真正 interactive environment (WebArena, SWE-Bench, https://www.swebench.com/), 而非 GAIA 这种 static dataset。SWE-Bench 上一个 action library 能否积累成真正的 "coding skill library"? 这是个 open question, 我赌是可以。

最后, 如果用 Karpathy 你喜欢的 "Software 2.0" framing (https://karpathy.medium.com/software-2-0-a6c52b1f1d59), DynaSaur 可以看作 **Software 2.5**: weights 定义 reasoning prior, Python function library 定义 acquired skills, 两者组合形成 adaptive agent。这比纯 Software 1.0 (hand-coded) 和纯 Software 2.0 (weights only) 都更接近 AGI 的形态。

---

**Key References**:
- Paper: https://arxiv.org/abs/2509.xxxxx (DynaSaur)
- Code: https://github.com/adobe-research/dynasaur
- Voyager: https://voyager.github.io/
- GAIA: https://arxiv.org/abs/2311.12983
- ReAct: https://arxiv.org/abs/2210.03629
- Reflexion: https://arxiv.org/abs/2303.11366
- DreamCoder: https://arxiv.org/abs/2006.08237
- Karpathy "Software 2.0": https://karpathy.medium.com/software-2-0-a6c52b1f1d59
- Karpathy "State of GPT / LLM OS": https://www.youtube.com/watch?v=zjkBMFhNj_g
- SWE-Bench: https://www.swebench.com/
- Code as Policies: https://code-as-policies.github.io/
- Toolformer: https://arxiv.org/abs/2302.04761
