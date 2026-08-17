---
source_pdf: VitaBench Benchmarking LLM Agents with Versatile Interactive Tasks in
  Real-world Applications.pdf
paper_sha256: 3b899382a095f5dc3aa2e6e49daed912ecdb0a9d40d640afadf815908815618e
processed_at: '2026-08-13T02:32:15-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 VitaBench

简单讲，这篇 paper 是一群美团的人想给 LLM agent 出一套"真正像样的考题"。

他们看了一圈现有 benchmark，觉得都不够狠。于是自己造了一个叫 VitaBench 的，结果把市面上最强的模型（GPT-5、o3、Claude-4.1-Opus 这种）全塞进去考了一遍，发现最好的也就考 30 分。满分 100。

下面用人话讲讲为啥这套题这么难，以及它到底揭示了个啥。

---

## 一、这篇 paper 在抱怨什么

现有的 agent benchmark 有个共同的毛病：**考得太假**。

举个例子，你让 agent 帮你订个餐厅。以前的考题是这么出的：
- 给你 8 个 tool
- 用户一句话说清楚全部需求："订个 4 人桌，7 点，素食"
- Agent 调一下 API，完事

但真实世界哪有这么干净。真实情况是：
- 用户上来只说"我周末想带家人吃个饭"
- Agent 得追问"几个人？哪天？有没有老人小孩？预算多少？"
- 用户答到一半突然说"对了还要给老人买拐杖送过来"
- Agent 得在脑子里同时挂着订餐厅、买拐杖、协调送达时间这几件事
- 中途 tool 调用失败还得自己想办法绕过去
- 用户被问烦了还会发脾气

τ-bench 那一类 benchmark 想模拟真实场景，但它们走了一条路：**给 agent 塞一本 100 页的 policy document**，告诉它"你必须先做 A 再做 B，C 之后才能做 D"。Agent 就变成了一个 policy follower —— 在考试里分数不错，但根本没锻炼到"自己摸索"的能力。

VitaBench 想换一条路：**不给 policy，让 agent 自己从 tool 的描述里推断**。比如 `modify_order` 这个 tool 的描述里写了 "pre-condition: 必须先调用 `get_order_detail`"。Agent 看到这个 schema 自己就懂了，不用读手册。

---

## 二、VitaBench 这套题长什么样

三个 domain：
- **Delivery**：外卖（20 个 tool）
- **In-store**：到店消费，订餐厅订位（24 个 tool）
- **OTA**：在线旅游，订酒店、景点、机票、火车票（38 个 tool）

一共 66 个 tool。听起来不算多，但 tool 之间有依赖关系。比如你要改订单，得先查订单；你要付钱，得先建订单。这种依赖关系构成一张有向图，OTA 这张图有 309 条边、密度 22%。

考题分两种：
- **Single-scenario**：只在单个 domain 里做事，300 道
- **Cross-scenario**：跨三个 domain 同时做事，100 道（这是 paper 的 main result）

Cross-scenario 是什么意思呢？看 Appendix C 那个例子你就懂了：

> 用户说：我们一家三代周末去大连坐邮轮，下午 3 点登船。想先在港口附近找个适合三代同堂、有 accessibility 设施的餐厅，12 点订 6 人桌。还得给老人买拐杖和成人纸尿裤，直接外卖送到餐厅，中午 12 点左右到。对了，我阿姨从北京过来，你帮她买张 7/27 早班高铁 first class，最好 11 点前到大连。然后帮我规划下阿姨从大连北站到餐厅的路线。最后再查个天气。

一句话里塞了 6 件事，跨三个 domain，有时间约束（高铁 10:47 到、外卖 12:02 到、用餐 12:00、登船 15:00），有空间约束（餐厅要在港口附近 2km 内、药店要离餐厅近），有隐含约束（"三代同堂" 意味着要有老人菜和儿童菜、"有 accessibility" 意味着餐厅要有无障碍设施）。

Agent 要在 60 多轮对话里把这堆事都办了。最强模型跑通的概率 30%。

---

## 三、他们怎么造的"真实用户"

这是我觉得最巧的部分。他们用一个 GPT-4.1 来扮演用户。Prompt 里写了这么几条规矩：

1. **信息要分轮透露**。不能第一句话就把所有需求都说出来。"Break down information from instructions into multiple independent points, mentioning them separately in different rounds"
2. **被问什么答什么，没问的不主动说**。Agent 不问"老人有啥饮食限制"，用户就不说。
3. **如果 agent 重复问同一个问题 3 次，就表现出不耐烦并拒绝回答**。这条特别狠，逼着 agent 学会"记住自己问过什么"。
4. **性格要一致**。配一个 persona —— "冷淡寡言、缺乏耐心、blue-collar worker、避开高嘌呤食物"，所有回答都得符合这个性格。

人话总结：他们把用户做成一个"会变脸、会藏信息、会不耐烦"的真实人类模拟器，而不是一个"问什么答什么"的查表机器。

---

## 四、评判打分的 tricky part

60 多轮的对话，怎么判分？早期 benchmark 的做法是：**跑完了看数据库最终状态**。比如最终是不是真订了一个 6 人桌、是不是真买了一张高铁票。

但这种做法有两个问题：
1. 有些动作不改变数据库状态。比如 agent 推荐了几家餐厅让用户挑，这种 "recommendation" 类动作在 final state 里看不出来。
2. 中间过程对不对没法监督。Agent 可能订对了餐厅但忘了给老人备注 accessibility，这种细节只有看中间过程才能发现。

VitaBench 的做法是：**给每个 task 写一组 rubric**。比如：

- Rubric 1: 选的餐厅在大连港 2km 内
- Rubric 2: 餐厅有 accessibility tag
- Rubric 3: 订了 7/27 12:00 6 人桌
- Rubric 4: 备注 accessibility pathway
- Rubric 5: 买了 walking cane（黑色可调节款）
- Rubric 6: 买了 L size adult diapers
- Rubric 7: dispatch_time 设为 10:45 左右（保证 12:02 送达）
- Rubric 8: 高铁是 G901 first class
- Rubric 9: 高铁到达时间早于 11:00
- ...一共十几条

打分时 strict all-or-nothing：全满足才算 1 分，差一条都是 0 分。

但 60 轮对话塞给 evaluator model 评估，context 不够用。他们就用 **sliding window**：每 10 轮一个窗口，相邻窗口重叠 2 轮，每个窗口只判这个窗口里的 rubric 是否满足。然后维护一个 state vector，一条 rubric 一旦在某窗口被标记为满足就永久保持（除非后续窗口明确推翻）。

最后所有窗口跑完，看 state vector 是不是全 1。

Ablation 显示这套设计（sliding window + rubric）跟人工标注的 Cohen's κ 有 0.828，相当高。去掉 rubric 只看 trajectory 直接 κ 掉到 0.018，基本等于瞎判。

---

## 五、实验里最有意思的几个发现

### 发现 1：数据库越大不一定越难

直觉上产品越多越难找。但实验结果是反的：

| Domain | 产品数 | 难度分数 |
|---|---|---|
| In-store | 3,277 | 42.1 (最容易) |
| Delivery | 788 | 38.0 |
| OTA | 9,693 | 20.7 |
| Cross | 6,946 | 16.2 (最难) |

In-store 产品最多但最容易，因为推理点少（5.6 个/task）。OTA 产品也多但推理点多（9.7 个/task），而且 tool 之间的依赖图密度最高（22%），所以最难。

**人话：难度取决于"要拐几个弯的推理链"，不取决于"候选有多少个"。**

### 发现 2：稳定性比能力更稀缺

他们跑了 Pass@4（4 次里至少跑通 1 次的概率）和 Pass^4（4 次全部跑通的概率）：

- Claude-4-Sonnet：Pass@4 = 51%，Pass^4 = 6%
- 顶级模型 Pass^4 普遍跌到个位数

意思是：**这些模型偶尔能跑出一条完美 trajectory，但让它再来一遍同样的任务，大概率跑不出来。**

这是个非常 production-relevant 的发现。你 deploy 一个 agent，今天它能办成 5 件事，明天同样的事可能就办砸了。这个 variance 不是来自模型本身的随机性（temperature 都设 0 了），而是来自 multi-turn 交互里 stochastic user simulator 引入的累积扰动 —— 第 5 轮 user 多说一句话，agent 第 6 轮的 plan 就分叉了，后面越走越远。

这个发现对 RL 训练有直接含义：如果用 single rollout 估 reward，policy gradient 会被这个 variance 淹没。需要多条 rollout average、或者 importance sampling、或者 variance reduction 技巧。

### 发现 3：Reasoning error 一家独大

把 76 个失败 rubric 分类：

- 61.8% reasoning error（推理错）
- 21.1% tool-use error（选错 tool、传错参数）
- 7.9% interaction error（对话管理失败）
- 9.2% user simulator 自己出 bug

更狠的是，作者分析失败 trajectory 时发现三个 recurring pattern：

1. **跨时空推理崩**。比如外卖送达时间和用餐时间没对齐、高铁到站时间和吃饭时间冲突、餐厅到港口距离算错导致赶不上船。
2. **Agent 对自己的能力边界没数**。明明有合适的 tool，却直接告诉用户"这个我做不到，你自己处理吧"。Poor self-awareness。
3. **遇到错误不会恢复**。Tool 调用失败了，agent 就一直 retry 同一个调用，不会换个角度（比如换一家店、换一个产品）。

第 2、3 两个发现我觉得是这篇 paper 最 value-add 的洞察。它说明当前 LLM agent 的瓶颈已经不是"会用 tool"了 —— 工具调用本身做得挺好的 —— 而是 **元认知**：知道自己什么时候该坚持、什么时候该换路、什么时候该承认失败。

### 发现 4：Thinking 模式既提分又省 turn

thinking 模型平均 23.8% / 61.1 turns，non-thinking 17.9% / 69.9 turns。

Claude-4.1-Opus 从 21.8% → 29.0%（thinking on），提升 33%。同时 turn 数还更少。

为啥？因为 thinking 让 agent 在脑子里先把 multi-step plan 拆好，问 user 的问题也更精准（不会瞎问），所以总交互次数反而降下来。

**人话：thinking 模式不只是"想得更深"，是"想得更深导致问得更准、走得更快"。**

### 发现 5：Cross-scenario 是个悬崖

| 任务类型 | 顶级模型 Avg@4 |
|---|---|
| 单 domain | 50%+ |
| Cross-scenario | 30% |

从 50% 掉到 30%，跌了 40%。这说明 LLM 在多个独立 context 之间切换的能力非常弱。一个 domain 的 tool schema 把 context window 占了之后，另一个 domain 的推理就被稀释。

跟 in-context learning 里的 "context pollution" 现象是一回事 —— 你往 prompt 里塞的东西越多，每件事的 attention 权重越分散。

---

## 六、这套 benchmark 的局限

说点 paper 自己不会承认的：

1. **User simulator 还是 LLM**。GPT-4.1 模拟 user，fidelity 9.48/10 看起来不错，但真实人类有更多 irrationality：突然改主意、报错自己的 user ID、记错时间。下一代 benchmark 应该是 human-in-the-loop。

2. **3 个 domain 偏生活服务**。金融、医疗、法律这些高 stakes domain 没碰。在这些领域里 agent 错一步代价巨大，rubric 设计要复杂得多。HealthBench（https://arxiv.org/abs/2505.08775）已经在医疗方向探索了，思路可以借鉴。

3. **All-or-nothing 打分太狠**。15 条 rubric 满足 14 条也是 0 分。做 leaderboard 比较的时候会低估 partial success 的模型。不过 paper 里也说了 rubric 可以提供 dense signal 给 RL 训练用，只是 leaderboard 比较时用 strict scoring。

4. **66 个 tool 还是不够多**。真实美团/携程 backend 有几千个 API。但 66 已经是当前 benchmark 里最多的，再扩可能就跑不动 evaluator 了。

5. **User simulator 9.2% 自己出错**这个数字。意味着 benchmark 本身有 ~10% 的 noise。虽然他们靠跑 4 次平均来 mitigate，但这是个 fundamental limit。

---

## 七、对你 build agent 的直觉启示

如果让我从这篇 paper 抽三句话给你：

**第一句**：**当前 agent 的瓶颈在元认知，不在工具使用。**

Reasoning error 61.8% + poor self-awareness + poor error recovery 这三连击说明，下一代 agent 的突破口是 self-reflection、self-monitoring、self-correction 这类 meta-cognitive 能力，不是再加 100 个 tool。

**第二句**：**多 domain 切换是 LLM 的真正软肋。**

Cross-scenario 的 50% → 30% 悬崖告诉我们，in-context learning 在多 context 切换时会互相干扰。如果做 production agent，要么把不同 domain 拆成 sub-agent 各跑各的，要么在 context management 上下功夫（比如 hierarchical memory、domain-specific context window）。

**第三句**：**Stability 比 capability 难得多。**

Pass@4 = 51% 但 Pass^4 = 6% 意味着 LLM agent 在 stochastic 多轮交互里本质上是个高方差随机系统。要 deploy 到生产，需要：
- 多次 rollout 投票（self-consistency）
- 显式 state tracking（不要全靠 context memory）
- 失败检测 + retry 策略
- 最好有个 evaluator model 在线监控 trajectory 健康

---

## 八、最后讲讲 Appendix C 那个例子

那个 trajectory 我建议你直接去看 paper Appendix C，很长但很值得看。Agent 干了大概这些事：

1. 调 `address_to_longitude_latitude` 把"大连港"转成经纬度
2. 调 `get_nearby` 找附近的餐厅和药店
3. 调 `longitude_latitude_to_distance` 算距离
4. 调 `instore_book` 订 6 人桌
5. 调 `pay_instore_book` 付订金
6. 调 `delivery_product_search_recommand` 找 walking cane 和 adult diapers
7. 调 `create_delivery_order` 下外卖单
8. 调 `pay_delivery_order` 付款
9. 调 `train_ticket_search` 找北京到大连的高铁
10. 调 `get_ota_train_info` 查 first class 票价
11. 调 `create_train_order` + `pay_train_order` 买票
12. 调 `get_train_order_detail` 拿电子票信息
13. 调 `longitude_latitude_to_distance` 算大连北站到餐厅的距离
14. 调 `weather` 查大连 7/27 天气
15. 给用户整理出租车短信模板
16. 设置一堆 reminder（05:15 出发、10:35 到站、11:52 外卖、14:10 出租车）

整条 trajectory 跑下来 60+ 轮对话，agent 干得相当漂亮。但 paper 说同样这种任务，最强模型平均跑通率 30%。

为啥同样的题能差这么多？因为 stochastic user simulator 在不同 run 里说的话不完全一样。第 8 轮 user 多问一句"颜色能不能换"，agent 第 9 轮就得分叉去查颜色，整个 plan 后面就全偏了。Multi-turn agent 对早期 disturbance 极度敏感，这是 chaos theory 在 LLM 上的体现。

---

## 九、一些后续可能值得挖的方向

如果你想顺着这篇 paper 往下挖，几个方向：

1. **Meta-cognition 训练**：怎么让 LLM agent 学会"我现在该 stop and reflect 了"？VitaBench 的 rubric 提供 dense signal，可以直接拿来做 RL，奖励 agent 在 trajectory 中段就 self-correct。参考 Self-Refine（https://arxiv.org/abs/2303.17651）、Reflexion（https://arxiv.org/abs/2303.11366）这条线。

2. **Cross-scenario 的 context management**：怎么让一个 agent 在 66 个 tool 的 prompt 里还能 focus 当下任务？Hierarchical memory、tool retrieval、task-specific context isolation 都是可能的方向。看看 Anthropic 的 Constitutional AI 和 OpenAI 的 function calling with retrieval。

3. **Variance reduction for multi-turn agents**：Pass^4 = 6% 这个数字说明 multi-turn agent 的 reward estimation variance 极大。可以借鉴 RL 里的 GAE、PPO clip、importance sampling，或者干脆用 self-consistency voting。

4. **Sliding window evaluator 的泛化**：这套技术可以搬到 SWE-bench、WebArena、长对话评估上。是个 methodological small gem。

5. **真实用户混合模拟**：把 GPT-4.1 user simulator 跟真实 human user 数据混合，搞一个 hybrid benchmark。UserBench（https://arxiv.org/abs/2507.22034）在这方向有一些探索。

6. **Tool graph 作为 inductive bias**：把 tool 之间的 pre/post-condition 编码进 agent 的 planning 模块，而不是靠 LLM 从 schema 自己推断。可以是 tool retrieval、可以是 graph neural network、可以是显式的 state machine。

---

## 一句话总结

**VitaBench 是第一个真正把 reasoning、tool、interaction 三个维度同时拉满的 agent benchmark。它揭示的最重要的事实是：当前最强的 LLM agent 在真实世界复杂任务上只能考 30 分，而且 4 次跑里只有 1 次能稳定跑通。瓶颈在元认知和稳定性，不在工具调用本身。**

Project page: https://vitabench.github.io

---

# VitaBench: 一个面向 real-world LLM Agent 的全方位交互任务 benchmark

Andrej，这篇 paper 来自 Meituan LongCat Team，发布在 2025 年末，定位是"目前最复杂的生活服务类 agent 模拟环境"，目的就是把 LLM agents 从 toy benchmark 推向真实世界的 messy complexity。下面我从 motivation、formalization、construction、evaluator、experiments 五个层面给你做一个 deep dive，尽量把 intuition 给出来。

参考链接：
- VitaBench 主页：https://vitabench.github.io
- τ-bench (前作 baseline)：https://arxiv.org/abs/2406.12045
- τ2-bench：https://arxiv.org/abs/2506.07982
- ToolSandbox：https://aclanthology.org/2025.findings-naacl.75
- UserBench：https://arxiv.org/abs/2507.22034
- Kimi K2 (报告里 top 模型之一)：https://arxiv.org/abs/2507.20534
- GLM-4.5：https://arxiv.org/abs/2508.06471
- LongCat-Flash：https://arxiv.org/abs/2509.01322

---

## 1. Motivation: 现有 benchmark 漏掉了什么

作者把现有 user-interaction agent benchmark 全部摆上桌面对比，得出几个核心 gap：

- **ToolTalk / MINT / IN3** 这些早期工作要么 turn 数太短（2-10），要么 tool 数太少（0-8），inter-tool dependency 基本是空的。
- **ToolSandbox / τ-bench / τ2-bench** 这一类 stateful execution benchmark 有 dependency，但是用一坨 verbose policy document 把 agent 框死，agent 实际上是 "policy follower"，不是 autonomous explorer。
- **UserBench** 捕捉了 user preference 但 tool 复杂度太低（5 个 tool）。
- 所有 benchmark 都没有同时挑战 reasoning + tool + interaction 三个维度。

VitaBench 想做的是"同时把三轴都拉满"，并且通过 graph-based tool design 把 domain policy 内嵌到工具的前置/后置条件里，让 agent 自己去探索，而不是读 manual。

Table 1 里 VitaBench 满格状态：66 tools、[50,100] turns、cross-scenario、user profile + behavior attributes 全开。

---

## 2. Formalization: POMDP + Three-axis Complexity Framework

### 2.1 POMDP 形式化

作者把整个 environment 集合记作 $\mathcal{E}$，对单一环境 $e \in \mathcal{E}$，agent 任务建模为：

$$(\mathcal{U}, \mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, r)_e$$

各符号含义：
- $\mathcal{U}$: instruction space，user 真实需求的集合
- $\mathcal{S}$: state space，分解为 $\mathcal{S} = \mathcal{S}_{\mathrm{db}} \otimes \mathcal{S}_{\mathrm{user}}$，即数据库状态与 user 心理/记忆状态的笛卡尔积
- $\mathcal{A}$: action space，包含两种类型 action —— tool invocation（对 db）和 natural dialogue（对 user）
- $\mathcal{O}$: observation space，$\mathcal{O} = \mathcal{O}_{\mathrm{db}} \otimes \mathcal{O}_{\mathrm{user}}$，agent 只能看到 db 的 tool 返回值和对话历史，看不到 db 内部隐藏状态
- $\mathcal{T}: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$: 状态转移函数，分裂成 deterministic 的 $\mathcal{T}_{\mathrm{db}}$（Python function 实现）和 stochastic 的 $\mathcal{T}_{\mathrm{user}}$（LLM 实现）
- $r: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$: reward function，最终给出 $[0,1]$ 区间分数

整条 trajectory 写成：

$$\tau = (s_0, a_1, s_1, a_2, s_2, \ldots, a_T, s_T) \sim \pi_\theta(\tau | e, u) \tag{1}$$

- 下标 $t \in \{0, 1, \ldots, T\}$ 表示时间步
- $T$ 是 trajectory 总长度（VitaBench 里 50-100 turns）
- $\pi_\theta$ 是参数 $\theta$ 的 LLM agent policy
- 关键点：agent 只能观察到 $\{o_0, a_1, o_1, \ldots, a_{t-1}, o_{t-1}\}$ 这个 partial observable 的子集，而 $s_t$ 整体对 agent 不可见 —— 这就是 partial observability 的来源

### 2.2 Three-axis Complexity Framework

这是 paper 的核心 contribution，公式 (2)：

$$\mathcal{C}_{\mathrm{task}} = \langle \mathcal{C}_{\mathrm{reason}}, \mathcal{C}_{\mathrm{tool}}, \mathcal{C}_{\mathrm{interact}} \rangle \tag{2}$$

**Reasoning complexity $\mathcal{C}_{\mathrm{reason}}$** 用两个量来度量：
- Observation entropy $H(\mathcal{O})$ —— observation space 越大，agent 看到的东西越乱越杂
- Partial observability degree $\eta = 1 - \frac{|\mathcal{O}|}{|\mathcal{S}|}$
  - 分子 $|\mathcal{O}|$：agent 可观测到的状态数量
  - 分母 $|\mathcal{S}|$：完整状态空间大小
  - $\eta \to 1$ 说明 agent 几乎看不到真实状态，需要在 darkness 里推

**Tool complexity $\mathcal{C}_{\mathrm{tool}}$** 把所有 tool 建成 directed graph $G = (V, E)$：
- $V$: tool 节点集，$|V|$ 是 tool 数量（VitaBench 共 66）
- $E$: directed edges，编码 inter-tool dependency（pre-condition/post-condition）
- Graph density $\rho = \frac{|E|}{|V|(|V|-1)|}$：完全有向图归一化边密度
- Task-relevant subgraph coverage ratio $\frac{|V_{\mathrm{task}}|}{|V|}$：当前任务实际涉及多少 tool

**Interaction complexity $\mathcal{C}_{\mathrm{interact}}$** 三个子分量：
- User profile：性别、年龄、饮食限制等持久属性
- Behavior attributes：emotional expression (impatient/anxious/indifferent)、interaction pattern (detail-oriented/dependent/logical)
- Dynamic state $\mathcal{S}_{\mathrm{user}}$：user 状态在对话过程中演化（比如 agent 反复问同一个问题，user 会变得不耐烦并拒绝回答）

这套 framework 给后续 benchmark 设计提供了一张 checklist，任何人想造 agent benchmark 都可以先填这三维表格。

---

## 3. Benchmark Construction: 两阶段流水线

### Stage I: Framework Design

三个 domain：
- **Delivery**: food + product delivery，20 tools
- **In-store Consumption**: dining + services，24 tools
- **OTA**: hotel + attraction + flight + train，38 tools

工具设计的关键 trick：用 **pre-condition + post-condition** 的形式 augment tool description，把 domain policy 编码进 tool graph 本身。例如 `modify_order` 这个 tool 的 pre-condition 是必须先调用 `get_order_detail`。这样 agent 不需要读 100 页 policy doc，从 tool schema 就能推断合法 workflow。

User simulator 用 GPT-4.1-2025-04-14 实现，输入包含 complete instruction（多需求），但通过 prompt 控制：信息逐轮 reveal，implicit constraint 只在被问及时才说出。Prompt 里明确写：
- "Break down information from instructions into multiple independent points, mentioning them separately in different rounds"
- "If the agent repeats the same question you have already answered in the past 3 times, show impatience and refuse to answer"

### Stage II: Task Creation

四个 component：
1. **User profiles**：来自真实平台 anonymized data，加上 emotional expression (impatient, anxious, indifferent) 和 interaction pattern (detail-oriented, dependent, logical) 标签
2. **Task instructions**：多个真实 user request 合成 composite objective，比如 Appendix C 那个例子 —— 同时要订餐厅、给老人送 walking cane 和 adult diapers 到餐厅、帮 aunt 买 G901 高铁 first class 票
3. **Environmental data**：service provider + product 数据混入 distractor，每个 task 平均 5-20 个 provider，最多 100+ 个 product
4. **Rubrics**：人工拆 atomic criteria

数据规模 (Table 2)：

| Domain | Service Providers | Products | Transactions | Tools (Write/Read/General) | Tasks |
|---|---|---|---|---|---|
| Cross-Scen. | 1,324 | 6,946 | 447 | 66 (27/33/6) | 100 |
| Delivery | 410 | 788 | 48 | 20 (4/10/6) | 100 |
| In-store | 611 | 3,277 | 28 | 24 (9/10/5) | 100 |
| OTA | 1,437 | 9,693 | 154 | 38 (14/19/5) | 100 |

注意 OTA 一个 domain 就有 9,693 个 product，但 search space 大不等于难 —— in-store 虽然产品多但 reasoning point 少，反而最容易。

---

## 4. Rubric-based Sliding Window Evaluator

这是 paper 的第二个 methodological contribution。

### 4.1 为什么需要 sliding window

长 trajectory (50-100 turns) 会超出 evaluator model 的 effective context length，即便号称 200k context 的模型在 100 turn trajectory 上 long-range retrieval 也会崩。作者用 Claude-3.7-Sonnet 做 evaluator，ablation 实验 (Table 4) 显示 full trajectory + rubric 比 sliding window + rubric 的 Cohen's κ 从 0.828 掉到 0.604。

### 4.2 算法细节

对每个 task，人工写一组 rubric：

$$\mathcal{R} = \{r_1, r_2, \ldots, r_k\}$$

- $r_j$: 第 $j$ 条 atomic criterion，比如 "restaurant within 500m"、"user only eats vegetarian food"、"book 6-person table at 12:00 on July 27"

Trajectory 切分成 overlapping windows：

$$W_i = (\text{turn}_{i \cdot (w - \delta) + 1}, \ldots, \text{turn}_{i \cdot (w - \delta) + w})$$

- $w$: window size（每个 window 含 $w$ 个 turn）
- $\delta$: 相邻 window 的 overlap turn 数，确保信息连续
- $i$: window index

Evaluator 维护一个 state vector：

$$\mathbf{s} \in \{0, 1\}^k$$

- $s_j = 1$ 表示 rubric $r_j$ 已经被某 window 满足
- 设计上 $s_j$ 是 monotone —— 一旦置 1 就永久保持，除非后续 window 明确显示该 rubric 被推翻（论文里允许 true $\to$ false 的反向翻转）

最终评分是 strict all-or-nothing：

$$\text{score} = \mathbb{1}\left[\sum_{j=1}^{k} s_j = k\right]$$

- $\mathbb{1}[\cdot]$: indicator function
- 全部 $k$ 条 rubric 都满足才得 1 分，否则 0

这套设计有两个 downstream 用途：
- benchmark 评估：strict scoring
- RL training：rubric 提供 dense reward signal，每条 $s_j$ 都是一个 binary reward

### 4.3 Ablation 验证 (Table 4)

| Method | Score | Task Acc. | Rubric Acc. | Cohen's κ |
|---|---|---|---|---|
| Baseline (sliding + rubric) | 20.0 | 95.0 | 88.5 | **0.828** |
| w/o Sliding Window | 19.0 | 90.0 | 87.6 | 0.604 |
| w/o Rubric Checklist | 91.0 | 22.0 | — | 0.018 |
| w/o Both | 82.0 | 32.0 | — | 0.067 |

去掉 rubric checklist 之后 Cohen's κ 直接掉到 0.018，说明 rubric 是 evaluator 的灵魂。Sliding window 也贡献了从 0.604 到 0.828 的提升。

---

## 5. Experiments

### 5.1 Setup

- **Agent models**: GPT-4.1, GPT-5, o3, o4-mini, Claude-4-Sonnet, Claude-4.1-Opus, Gemini-2.5-Flash/Pro, DeepSeek-V3-0324, DeepSeek-R1-0528, DeepSeek-V3.1, DeepSeek-V3.2, Qwen3-32B, Qwen3-235B-A22B, Qwen3-Max, Kimi-K2, Seed-1.6, GLM-4.5, LongCat-Flash-Chat, LongCat-Flash-Thinking
- **User simulator**: gpt-4.1-2025-04-14
- **Evaluator**: claude-3.7-sonnet（特意避开 agent model）
- **Temperature**: 0.0
- **Runs per task**: 4
- **Metrics**:
  - Avg@4: 4 次平均分
  - Pass@4: 至少一次成功的概率
  - Pass^4 (Pass-hat-4): 4 次全部成功的概率

### 5.2 Main Results (Table 3)

Top-tier 表现：

| Model | Cross-Scen Avg@4 | Delivery | In-store | OTA |
|---|---|---|---|---|
| o3 (high) | **30.0** | 53.5 | 53.5 | 37.8 |
| Claude-4.1-Opus (w/ thinking) | 29.0 | 47.5 | 52.5 | 32.3 |
| LongCat-Flash-Thinking | 24.3 | 42.3 | 56.8 | 28.3 |
| Gemini-2.5-Pro | 23.5 | 49.0 | 43.8 | 26.5 |
| Claude-4-Sonnet (w/ thinking) | 23.0 | 46.0 | 51.5 | 29.0 |
| GPT-5 (high) | 22.8 | 54.0 | 52.5 | 37.5 |
| GLM-4.5 (w/ thinking) | 22.8 | 44.5 | 52.8 | 28.8 |

Cross-scenario 最强也就 30%。这数字对比 single domain 上 o3 能跑到 53.5%（Delivery / In-store），说明 cross-scenario 是个真正的 cliff —— 不是简单线性叠加，是组合爆炸。

### 5.3 几个有意思的发现

**A. 难度跟 database 规模反相关**

Table 5 显示 in-store 的 search space 是 3,916，比 OTA 的 11,284 小，但 reasoning points 也少（5.6 vs 9.7），结果 in-store Avg@4 是 42.1 而 OTA 只有 20.7。这告诉我们 reasoning complexity 主导，不是 search space size 主导。

| Domain | Perf | Reas. Pts | Search Space | Tools | Edges | Density |
|---|---|---|---|---|---|---|
| In-store | 42.1 | 5.6 | 3,916 | 24 | 68 | 12.3% |
| Delivery | 38.0 | 7.4 | 1,246 | 20 | 50 | 13.2% |
| OTA | 20.7 | 9.7 | 11,284 | 38 | 309 | **22.0%** |
| Cross-Scen. | 16.2 | 10.3 | 8,717 | 66 | 512 | 11.2% |

OTA 的 graph density 22% 是最高的，512 条 dependency edge 是 in-store 的 7.5 倍 —— 这是 OTA 难的真正原因。

**B. Pass@4 vs Pass^4 的剪刀差**

Figure 4 显示 Pass@4 (explore good case) 和 Pass^4 (always stable) 之间的 gap 巨大。以 Claude-4-Sonnet 为例：
- Pass@4 ≈ 51%
- Pass^4 ≈ 6%

意思是同一个 task，4 次独立运行里至少有 1 次能跑通的概率是 51%；但 4 次全部跑通的概率只有 6%。Agent 在 stochastic 多轮交互下 fundamental instability 非常严重。这一点对 RL 的 implication 很大 —— 如果用 single rollout 估 reward，policy gradient 噪声会爆炸。

**C. Thinking mechanism 同时提升 effect 和 efficiency**

Figure 5 显示 thinking models 平均 23.8% perf / 61.1 turns，non-thinking 17.9% / 69.9 turns。Thinking 既能让 perf 上去还能把 turn 数降下来，主要因为：
1. 更好的 multi-step plan decomposition
2. 更精准的 clarifying question（不会瞎问）

Claude-4.1-Opus 从 21.8%（non-thinking）→ 29.0%（thinking），提升 33%。GLM-4.5 从 20.0% → 22.8%，提升 14%。

**D. 错误分布 (Figure 9)**

| Error Category | % |
|---|---|
| Reasoning errors | **61.8%** |
| Tool-use errors | 21.1% |
| Interaction errors | 7.9% |
| User simulator errors | 9.2% |

Reasoning error 一家独大。作者把失败进一步拆解成几个 recurring pattern：
1. Spatiotemporal + commonsense reasoning 失败，跨多个信息源集成能力弱
2. **Poor self-awareness** —— agent 明明有合适 tool 却放弃任务
3. **Limited error recovery** —— tool 调用失败后只会 retry，不会换策略

第 2、3 点特别有意思，因为它揭示当前 LLM agent 的核心瓶颈不是"会不会用 tool"，而是 "对自身能力边界和 error 的元认知"。

### 5.4 Reliability Analysis

User simulator (Figure 6)：
- Information fidelity 9.48/10
- Persona consistency 9.34/10（cooperative persona 最高，scattered persona 最低）

Evaluator 上面已经讨论过 Cohen's κ = 0.828。

Evaluation 统计稳定性 (Figure 7)：用 32 次 trial 做 reference，对每个 $k \in [1, 20]$ 计算 $\text{MSE}(\bar{X}_k, \bar{X}_{32})$，发现 $k=4$ 已经把 MSE 降了 77.5%，再往上 $k=8$ 只 marginal 改善。这是为什么 main result 都跑 4 次。

### 5.5 Interaction Complexity Ablation (Figure 8)

三个 condition：
1. Default user simulator (with persona + behavior)
2. Neutral user (without persona)
3. Solo agent（complete instruction 一次性给全，no user interaction）

Claude-4-Sonnet 在 solo mode 下提升明显，说明它在一次性处理复杂 instruction 上很强；但 default user mode 下被对话管理拖累。GPT-4.1-Mini 的 default vs neutral gap 更大，说明 conversational style 对弱模型杀伤力更大。

---

## 6. 一个 Concrete Example (Appendix C)

Appendix C 给了一个完整的 60+ turn cross-scenario trajectory，是理解 VitaBench 难度的最好入口。

**User Profile**：
- 30-35 岁，blue-collar worker，Harbin 人
- 饮食限制：避免高嘌呤（动物内脏/海鲜汤）、避免油炸
- 性格：Cold, concise, lacks emotional communication

**Task**（一个人带着全家 + 阿姨去大连坐邮轮）：
1. 在大连港附近找一个适合三代同堂、有 accessibility 的餐厅，7/27 12:00 订 6 人位
2. 给老人买 walking cane 和 adult diapers，配送时间 12 点附近送到餐厅
3. 帮北京来的阿姨买 7/27 早班高铁 first class，最好 11 AM 前到大连
4. 给阿姨安排从大连北站到餐厅的路线，组织好短信模板
5. 安排 14:10 餐厅到港口的 taxi 提醒
6. 查天气

Agent 需要协调 66 个 tool，同时管餐厅 / 外卖 / 高铁三套独立 db，跨 domain 推理（外卖送达时间要跟用餐时间对齐，高铁到站时间要早于用餐时间），还要处理 spatiotemporal 计算（多次调用 `longitude_latitude_to_distance` 算餐厅-港口-车站-药店距离）。

最后 Agent 给了一个近乎完美的 multi-turn execution —— 但 paper 里说 top model 平均成功率才 30%，意味着 70% 的 case 跑不出这个 trajectory，会在某条 rubric 上 fail（比如忘了加 `dispatch_time` 的精确分钟、或者选错 walking cane 颜色、或者没把 accessibility 注释进 order note）。

---

## 7. 跟相关工作的对比

| Benchmark | 差异点 |
|---|---|
| ToolTalk (2023) | 多步 tool exec 但 trajectory 预定义，agent 没自主权 |
| MINT (2024) | 强调 NL feedback，但 tool 数只有 8 |
| IN3 (2024) | 只看 implicit intention detection，没 tool complexity |
| ToolSandbox (2025) | stateful 但 tool 数 34，没 cross-scenario |
| τ-bench (2024) | 28 tool，靠 verbose policy doc 约束，agent 是 policy follower |
| τ2-bench (2025) | dual-control，38 tool，引入 behavior attributes 但仍受 policy 约束 |
| UserBench (2025) | 5 tool，preference-driven，task 复杂度太低 |
| **VitaBench** | **66 tool, cross-scenario, graph-embedded policy, persona + behavior + dynamic state** |

VitaBench 跟 τ-bench 系列最核心的差异在于 **policy 在哪里**：
- τ-bench：policy 是给 agent 读的 markdown document
- VitaBench：policy 编码在 tool 的 pre/post-condition 里，agent 必须 explore 才能发现

这跟现实世界更像 —— 你不需要先读美团 200 页用户协议，而是从工具的 schema 推断合法 workflow。

---

## 8. 局限与思考

1. **User simulator 仍是 LLM-based**。GPT-4.1 模拟 user，9.5/10 fidelity 已经很好，但仍然是 LLM 的 distribution，真实人类 user 会有更多 irrationality（突然改主意、记错信息、报错 user ID）。下一代的 benchmark 应该 hybrid human-in-the-loop。

2. **3 个 domain 有限**。生活服务能 cover，但金融、医疗、法律这些高 stakes domain 还没有。Healthbench ([Arora et al. 2025](https://arxiv.org/abs/2505.08775)) 和 ExpertLongBench ([Ruan et al. 2025](https://arxiv.org/abs/2506.01241)) 的 rubric 思路可以借鉴到这些领域。

3. **Strict all-or-nothing scoring 过于 harsh**。Score = $\mathbb{1}[\sum_j s_j = k]$ 意味着 15 条 rubric 满足 14 条也是 0 分。对 RL 训练可以提供 dense reward，但 leaderboard 比较下会低估 partial-success 的 agent。

4. **Pass^4 = 6% 的稳定性 cliff** 是最重要的 finding。如果 agent deployment 不能保证 reproducibility，production-grade 部署就有问题。这可能跟 RL training 的 variance reduction、temperature schedule、self-consistency 的多 vote 机制都有关。

5. **Reasoning error 61.8% + poor self-awareness** 这个组合暗示，下一代 agent 的核心改进方向不是加更多 tool，而是 **meta-cognition** —— agent 需要能 reflect on 自己的能力边界和已执行 trajectory。

6. **Cross-scenario cliff（30% vs 50%+）** 说明 LLM 在多个独立 policy context 之间切换的能力还很弱，跟 in-context learning 的 "context window 污染" 现象可能相关 —— 一个 domain 的 tool schema 占了 context 之后，另一个 domain 的 reasoning 就被稀释。

7. **Sliding window evaluator** 这套技术可以推广到任何 long-horizon agent trajectory 评估，比如 SWE-bench、WebArena 这类，是一个 methodological 上的 small gem。

---

## 9. 总结

VitaBench 的核心 contribution：
- **Three-axis complexity framework**：$\mathcal{C}_{\mathrm{task}} = \langle \mathcal{C}_{\mathrm{reason}}, \mathcal{C}_{\mathrm{tool}}, \mathcal{C}_{\mathrm{interact}} \rangle$，把"什么是 real-world agent task 的复杂度"这个问题量化
- **Graph-embedded policy**：用 tool pre/post-condition 替代 verbose policy doc，让 agent 必须 explore
- **Sliding window rubric evaluator**：解决 long trajectory 评估的 context length + monotone state tracking 问题
- **400 task / 66 tool / 50-100 turn** 的最大规模 life-serving simulation
- **揭示了 thinking vs non-thinking、Pass@4 vs Pass^4、cross-scenario cliff** 几个 actionable insight

对你 build intuition 来说，最关键的两个 takeaway：
1. **Agent 评估必须三维同时拉满**，单一维度上刷分对 deployment 没意义
2. **Reasoning error 61.8% + poor self-awareness + poor error recovery** 三连击是当前 agent 的真实瓶颈，下一个突破口大概率来自 meta-cognition 和 self-reflection，而不是再加 tool 数量

VitaBench project page: https://vitabench.github.io

如果你想往下挖，建议看：
- τ2-bench 的 dual-control 设计：https://arxiv.org/abs/2506.07982
- UserBench 的 preference-driven user modeling：https://arxiv.org/abs/2507.22034
- Kimi K2 技术报告（VitaBench top-tier 之一）：https://arxiv.org/abs/2507.20534
- LongCat-Flash-Thinking 技术报告（作者自家 model）：https://arxiv.org/abs/2509.18883
