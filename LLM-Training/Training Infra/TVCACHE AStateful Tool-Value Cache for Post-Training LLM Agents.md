---
source_pdf: TVCACHE AStateful Tool-Value Cache for Post-Training LLM Agents.pdf
paper_sha256: 8a0e9b5c02217fa47d967893440060e4a157f9db61f16f83b4876ce2f78a0223
processed_at: '2026-08-12T18:26:04-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# TVCACHE 用人话讲

## 一句话说清楚

训练 AI agent 的时候，AI 会调用很多外部工具（比如跑 shell command、查数据库、处理视频），这些工具跑得很慢，GPU 在干等。TVCACHE 发现同一个问题的多个 rollout 经常做一样的事，于是把工具调用结果缓存起来复用，但关键是要保证 sandbox state 完全一致才复用，避免出错。

## 一、问题长啥样

想象你在训练一个 AI 学修 bug。每个 training step 会生成 8 个 rollout 并行跑，每个 rollout 都是 AI 边想边敲命令：

```
Rollout 1: git clone repo → make build → run tests → cat error.log → ...
Rollout 2: git clone repo → make build → run tests → grep "fail" → ...
Rollout 3: git clone repo → cmake .. → make → ...
Rollout 4: git clone repo → make build → run tests → vim foo.py → ...
```

你看出来没——前 3 步经常一模一样。但 GPU 端在等 `make build` 跑完（可能几十秒），这期间 GPU 空转。Paper 实测，terminal-bench 平均 43% 的时间 GPU 在等 tool 跑完，99th percentile 超过 92%。这等于花大钱租 GPU 然后让它干等。

## 二、为什么不直接缓存

你说这不就是 cache 嘛，key 是 tool 调用，value 是结果，有啥难的？

难在 **tool 是 stateful 的**。看这个例子：

```
Step 1: cat foo.py       → 看到原始代码
Step 2: patch foo.py     → 改了文件
Step 3: cat foo.py       → 看到改后的代码
```

Step 1 和 Step 3 命令完全一样，但输出不同，因为中间 sandbox 状态变了。如果用 naive cache，Step 3 会返回 Step 1 的 stale 结果，AI 就学错了。

## 三、TVCACHE 的核心招数

### 招数 1：用图组织工具调用历史

每个 task 维护一张图 $\mathcal{G}(p) = (\mathcal{V}(p), \mathcal{E}(p))$，节点 $v = (t, r, s)$：
- $t$：tool 调用（名字+参数）
- $r$：执行结果
- $s$：sandbox 快照（可选）

边表示"先执行 $u$ 再执行 $w$"。

直觉上，这就像 git 的 commit graph——共享前缀的 rollout 在图上共享路径，分叉的地方开新分支。比哈希表好，因为哈希表看不到前缀共享；比树好，因为树会重复存共享前缀。

### 招数 2：最长前缀匹配

当前 rollout 要调 tool $t_j$，前面已经调过 $t_1, \dots, t_{j-1}$。把整个序列 $q = \langle t_1, \dots, t_j \rangle$ 拿去图里找最长前缀 $\tilde{q}$：

- $\tilde{q} = q$（完全匹配）→ **cache hit**，直接返回结果
- $\tilde{q} \neq q$（部分匹配）→ **cache miss**，fork 掉 $\tilde{q}$ 末端节点的 sandbox，只跑剩下没匹配的部分

正确性来自一个不变量：**图上从 root 到某 node 的路径唯一确定该 node 的 sandbox state**。所以前缀完全匹配 = state 完全相同 = 下一个 tool 输出必然相同。

### 招数 3：选择性快照

每个节点都存 sandbox 快照太费内存（一个 Docker container 快照可能几百 MB）；一个都不存又导致 cache miss 时要从 root 重放整个 trajectory。

权衡策略：

$$\text{存快照} \iff \text{tool 执行成本} > \text{快照开销}$$

实际效果：
- `make build` 跑 30 秒 → 存快照
- `cat foo.py` 跑 1ms → 不存

这就像数据库 checkpoint——慢操作要 checkpoint，快操作频繁 checkpoint 反而浪费。

### 招数 4：Stateful Prefix Matching（最优雅的优化）

很多 tool 其实不改变 sandbox state。比如 SQL 的 `SELECT` 只读不写，EgoSchema 里 6 个 tool 只有 2 个真正改 state。

如果开发者能 annotate 哪些 tool 不改 state，TVCACHE 做最长前缀匹配时就跳过它们，只在 state-modifying tool 上匹配。

**形式化保证**（Appendix B 的定理）：

假设：
1. Tool 输出只依赖当前 state 和参数
2. 被 annotate 为 state-preserving 的 tool 确实不改 state

那从同一初始 state 出发，执行 $\mathcal{P} = \langle F_1, S_1, F_2, S_2, \dots \rangle$ 和 $\mathcal{P}' = \langle F_1, F_2, \dots \rangle$（去掉所有 $S$）会到达同一 state。

证明用反证法：假设 state 不同，那一定有某个 $S_j$ 改了 state，但 $S_j$ 被 annotate 为不改 state，矛盾。

**实际例子**（EgoSchema）：

```
Rollout 1: load_video → preprocess → caption(0,10) → qna(5)
Rollout 2: load_video → preprocess → qna(5) → caption(0,10)
```

不看 stateful 优化：两条不同路径，Rollout 2 全 miss
看 stateful 优化：`caption` 和 `qna` 都不改 state，所以两者在 `preprocess` 后共享 state，Rollout 2 的 `qna` 和 `caption` 都是 hit

这让 EgoSchema 的 cache hit rate 从很低飙升到 34-73.9%。

### 招数 5：Proactive Forking

Cache miss 时要 fork sandbox，Docker container 启动有开销。TVCACHE 在训练 step 开始前就预先 fork 好：
- 多个 root sandbox 暖机待命
- 对每个有快照的 TCG node 预先 fork 一份副本

这样 cache miss 时几乎零延迟拿到 ready-to-use sandbox。

## 四、实际效果

### Cache hit rate

| Workload | Hit Rate |
|---|---|
| terminal-bench | 15-32% |
| SkyRL-SQL | 27-57%，平均 33% |
| EgoSchema | 34-74%，平均 64% |

### Tool call 加速

| 配置 | No Cache | Cache | Speedup |
|---|---|---|---|
| Qwen3-4B Easy | 8.67s | 1.40s | 6.18× |
| Qwen3-4B Med | 18.68s | 2.70s | **6.92×** |
| Qwen3-14B Easy | 8.07s | 2.35s | 3.44× |
| Qwen3-14B Med | 36.23s | 6.53s | 5.55× |

### Reward 不退化

Figure 6 显示有 cache 和没 cache 的 reward 曲线几乎重合——这是 correctness 的实证验证。因为 TVCACHE 是 exact cache，保证返回结果和真实执行一致。

### SQL 场景的细节

SkyRL-SQL 的 SQL `SELECT` 是 stateless，cache hit 把单次调用从 56.6ms 降到 6.5ms（8.7×），结合 33% hit rate，整体 2.9× 加速。

### EgoSchema 的 API 费用节省

`caption_retrieval` 内部调 OpenAI API，cache 复用让 token usage 降低 3×。Paper 还特别论证了为什么不能用传统 API cache：不同 task 的 `caption(segment 1, segment 2)` 签名一样但底层视频不同，输出不同。正确 cache 必须把 sandbox state 绑进 key——这恰好就是 TVCACHE 的设计。

## 五、工程上踩的坑

### Docker 扩展性问题

Proactive forking 一次要起几百个 container，terminal-bench 默认 harness 扛不住。Paper 发现三个瓶颈：

1. Docker Compose 默认每个 sandbox 创建独立 bridge network——网络创建慢
2. 很多 task 根本不需要网络
3. 并发 fork 太多导致 cgroup kernel 资源争用

解决方案：
- 预创建 network pool 复用
- 解析 docker-compose 判断是否真需要 network
- Rate-limited fork pipeline，把并发 cap 在饱和点前

### 并发控制

多个 rollout 同时访问同一个 cached sandbox，用 reference counting：
- LPM 命中后 server 端 increment ref count
- Client fork 后 decrement
- Eviction 只能驱逐 ref count = 0 的 sandbox

避免 race condition：A 正在 fork，B 触发 eviction 把它删了。

## 六、和已有工作的关系

- **Semantic caching**（GPTCache 等）假设 prompt→response 是 stateless mapping，不适用 stateful tool call
- **Kimi K2** 用另一个 LLM 模拟 tool response——approximate，可能损失精度；TVCACHE 是 exact
- **VERLTOOL** 异步并行 tool execution——调度优化但不减少总计算量；TVCACHE 是 complementary
- **SGLang RadixAttention / vLLM PagedAttention** 用树/图组织 KV cache 共享前缀——TVCACHE 把这个思想从"KV tensor"迁移到"sandbox state"

## 七、最核心的 intuition

TVCACHE 的精髓就一句话：**当资源的"等价性"不能从单个 key 推断时，必须把 key 扩展为 trajectory，并在 trajectory 级别做 caching**。

这个 pattern 在很多地方都能看到：
- Build system（Bazel）按 dependency graph 缓存
- OS fork 用 copy-on-write 共享内存页
- Git commit graph 共享历史前缀
- Database MVCC 用 version chain

TVCACHE 把这个思想应用到 RL post-training 的 sandbox state 上，结合 cost-aware snapshotting 和 stateful prefix compression，把 tool call 的冗余度高效压缩。对 build intuition 而言，记住这个 pattern 比记住具体实现更重要——下次遇到类似的"看起来像 cache 但 stateful"的问题，你会知道该往哪个方向想。

## 参考链接

- 论文 arXiv：[https://arxiv.org/abs/2506.07251](https://arxiv.org/abs/2506.07251)
- veRL：[https://github.com/volcengine/verl](https://github.com/volcengine/verl)
- Tinker：[https://thinkingmachines.ai/tinker/](https://thinkingmachines.ai/tinker/)
- Terminal-Bench：[https://github.com/laude-institute/terminal-bench](https://github.com/laude-institute/terminal-bench)
- SGLang RadixAttention：[https://arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104)
- vLLM PagedAttention：[https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- Kimi K2：[https://arxiv.org/abs/2507.20534](https://arxiv.org/abs/2507.20534)
- GRPO：[https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)

---

# TVCACHE 深度解析

## 一、Motivation 与核心 Insight

在 RL post-training 的 LLM agent 训练场景下，rollout 的总时间通常分解为 reasoning token generation 和 tool call execution 两部分。paper 的 Figure 2 给出了一个关键数据：terminal-bench 平均 43% 时间花在 tool execution，99th percentile 甚至超过 92%；SkyRL-SQL 平均 7%，95th percentile 43%；EgoSchema 平均 12%。这段时间内，GPU 是 idle 的，因为 rollout 的 LLM forward pass 必须等 tool 执行完才能继续生成下一个 token。

paper 的关键观察在于：同一个 prompt 触发的多个并行 rollouts 经常执行**相似甚至完全相同的 tool call 序列**。例如同一个 codebase 的 build、同一套 test suite、同一张表的 SQL query。这构成了 caching 的机会。

但 naive caching 不成立的核心原因在于 **tool call 的 statefulness**：tool 的 output 不仅由其参数决定，还由整个历史 tool call 序列所诱导出的 sandbox state 决定。paper 给出的反例是一个 code-debugging agent 先 `cat foo.py`，然后 apply patch，再 `cat foo.py`——第二次 `cat foo.py` 的参数完全一样，但 sandbox state 已经变了，stale cache 会静默污染训练信号。

## 二、Tool Call Graph (TCG) 的设计直觉

### 2.1 形式化定义

对每个 task (prompt) $p$，TVCACHE 维护一张有向图：
$$\mathcal{G}(p) = (\mathcal{V}(p), \mathcal{E}(p))$$

其中每个节点 $v \in \mathcal{V}(p)$ 表示三元组：
$$v = (t, r, s)$$

- $t$：tool descriptor，即 tool name + arguments 的序列化字符串
- $r$：tool execution result（tool 执行后返回的内容）
- $s$：serialized sandbox snapshot，immediately after tool execution；$s$ 可以为 null（选择性 snapshot）

每条有向边 $(u, w) \in \mathcal{E}(p)$ 表示在一次 rollout 中 $u$ 紧接在 $w$ 之前执行。

### 2.2 为什么是图而不是树或哈希表

这里有一个设计选择上的直觉：rollouts 之间会**共享前缀**。比如 4 个 rollouts 都从 `git clone ...` 开始，然后部分走 `make build`、部分走 `cmake ..`。如果用哈希表按 tool descriptor 做 key，前缀共享信息丢失；如果用树，相同前缀会被重复存储；用 DAG/图则能共享前缀同时保留分叉。

paper 在 Figure 3 给出的示意：rollout 1 调用 $t_1, t_2, t_3, t_4, t_5, t_6$；rollout 2 在 $t_3$ 处分叉到 $t_6$；rollout 3 从 root 走到 $t_1, t_2, t_3$ 后分叉；rollout 4 又是新分支。TCG 自然地把"哪些前缀共享了"压缩进了图结构。

### 2.3 与 radix tree / PagedAttention 的类比

这个图结构让我联想到 SGLang 的 [RadixAttention](https://arxiv.org/abs/2312.07104)，以及 vLLM 的 [PagedAttention](https://arxiv.org/abs/2309.06180)——它们都是用树/图来组织 KV cache 的共享前缀，最大化 reuse。TVCACHE 本质上是把这个思想从"prompt token prefix 的 KV cache"迁移到"tool call prefix 的 sandbox state cache"。区别在于：

- RadixAttention 的"state"是 attention 的 KV tensor，只读、可线性拼接
- TVCACHE 的"state"是整个 sandbox（Docker container、SQLite DB、文件夹等），需要通过 fork 来复制，开销大得多

这引出了后面 selective snapshotting 的必要性。

## 三、Cache Lookup：Longest Prefix Matching

### 3.1 算法描述

设当前 rollout 已执行 $t_1, t_2, \dots, t_{j-1}$，现在要执行 $t_j$。构造查询序列：
$$q = \langle t_1, t_2, \dots, t_j \rangle$$

在 $\mathcal{G}(p)$ 上做 longest prefix match（LPM）：
- 设匹配到的最长前缀为 $\tilde{q}$
- **Cache hit**：$\tilde{q} = q$，直接返回 $t_j$ 节点存的 $r$
- **Cache miss**：$\tilde{q} \neq q$，记 $q_{\text{unmatched}}$ 为 $q \setminus \tilde{q}$ 的后缀
  - 若 $\tilde{q}$ 末端节点 $t'$ 有 snapshot $s$：fork $s$，在 forked sandbox 中顺序执行 $q_{\text{unmatched}}$
  - 若没有 snapshot：开新 sandbox，执行完整 $q$
  - 执行结果 append 到 TCG 作为新路径

时间复杂度 $O(\log |\mathcal{V}(p)|)$，paper 在 microbenchmark 中报告 P95 lookup latency 3.3 ms at 256 RPS（单 server）。

### 3.2 正确性论证

正确性来自一个关键不变量：**TCG 中任意从 root 到 node $v$ 的路径，唯一确定了 $v$ 处的 sandbox state**。

证明 sketch：
- Root 节点对应初始 sandbox state（base image 或 reset 状态）
- 每条边 $(u, w)$ 表示在 $u$ 的 state 上执行 $w.t$ 得到 $w$ 的 state
- 因此路径 → state 是函数（假设 tool 是 deterministic，或 paper 假设的 sandbox 行为可复现）

LPM 命中意味着当前 rollout 的 trajectory 和 TCG 中某条 root-to-node 路径完全相同，因此**当前 sandbox state 和该 node 处的 state 完全相同**，于是下一个 tool 的 output 必然相同，cache 命中是正确的。

这是 paper 的核心 correctness argument，Figure 9 的真实 TCG 可视化展示了在 terminal-bench 训练中实际生成的图结构。

## 四、Selective Sandbox Snapshotting

### 4.1 Trade-off 分析

如果每个节点都存 snapshot，cache 内存爆炸；如果一个都不存，每次 cache miss 都要从 root 重放整个 trajectory，省下来的 tool execution 时间被重放成本吃掉。

paper 的策略是**基于成本的决策**：
$$\text{snapshot}(v) \iff \text{exec\_cost}(v.t) > \text{snapshot\_overhead}(v)$$

- $\text{exec\_cost}(v.t)$：执行 tool $v.t$ 的耗时
- $\text{snapshot\_overhead}(v)$：serialize + later restore sandbox snapshot 的开销

实际效果：
- 长 latency tool（如 full test suite、large codebase compilation）会被 snapshot
- 便宜 tool（如 read small file）不 snapshot

这个 policy 让我想到 OS 里的 copy-on-write fork 和 database 里的 checkpoint 频率权衡——快操作频繁 checkpoint 浪费，慢操作不 checkpoint 重放昂贵。

### 4.2 Sandbox 生命周期优化

paper 提了三种 forking 策略：

**Proactive forking**：训练 step 开始前预创建
- 多个 root sandbox（warm standby）
- 对每个已有 snapshot 的 TCG node 预先 fork 一份 copy

**Reactive forking**：cache miss 时的 critical path 检查
- 先看 background thread 是否已 fork 好
- 没有再同步 fork

**Background instantiation**：snapshot 同步存，fork 异步做
- snapshot 写入 critical path 必须同步（保证 cache hit 的正确性）
- 但从 snapshot 重建可用的 forked sandbox 可异步

**Bounding cache size**：每个 task 有 sandbox budget，超限时按"预期 reuse 价值"驱逐。Eviction 考虑 node depth 和 children 数量——浅节点+多孩子代表常用公共前缀，应保留。

### 4.3 并发控制

并行 rollouts 会同时访问同一个 cached sandbox。paper 用 **reference counting**：
- LPM 命中后 server 端 increment ref count of prefix-end node 的 sandbox
- client fork 后 decrement
- eviction 只能驱逐 ref count = 0 的 sandbox

这避免了 race condition：rollout A 正在 fork 一个 sandbox，rollout B 触发 eviction 把它删了。

## 五、Stateful Prefix Matching 的形式化证明

这是 paper Appendix B 中我最喜欢的部分，因为它把一个工程优化变成了可证明的定理。

### 5.1 两个 Assumption

**Assumption 1 (State determines tool output)**：tool $T$ 的 output 仅由当前 sandbox state 和 $T$ 的 arguments 决定。

**Assumption 2 (Correct state annotation)**：若 `will_mutate_state()` 对 tool $S$ 返回 false，则执行 $S$ 不改变 sandbox state。

### 5.2 定理

设一个 rollout 的 prefix $\mathcal{P}$ 交替包含 state-modifying tools $F_1, F_2, \dots, F_N$ 和 state-preserving tools $S_1, S_2, \dots, S_N$。定义：
$$\mathcal{P}' = \langle F_1, F_2, \dots, F_N \rangle$$

即从 $\mathcal{P}$ 中移除所有 $S$ 工具但保留 $F$ 顺序。则 $\mathcal{P}$ 和 $\mathcal{P}'$ 在 TCG 中对应**同一 sandbox state**。因此只在 $F$ 子序列上做 LPM 是正确的。

### 5.3 证明（反证法）

假设从同一初始 state 执行 $\mathcal{P}$ 和 $\mathcal{P}'$ 得到不同 sandbox state。两者唯一差别是 $S$ 工具的执行。设 $S_j$ 是第一个使 state 分歧的 $S$ 工具。则 $S_j$ 修改了 sandbox state。但由 Assumption 2，标为 state-preserving 的 tool 不修改 state。矛盾。

故 $\mathcal{P}$ 和 $\mathcal{P}'$ 必导致相同 state。由 Assumption 1，后续任何 tool 的 output 仅依赖当前 state，因此 cache 命中正确。

### 5.4 实际意义

在 EgoSchema 工作负载上，6 个工具中只有 `load_video` 和 `preprocess` 会 mutate state，其余 4 个（`object_memory_querying`、`segment_localization`、`caption_retrieval`、`visual_qna`）都是 stateless。这个优化让 cache hit rate 从原本的较低水平跃升到 34-73.9%。

paper Appendix D 给了两个具体例子：

**Example 1**：两个 rollout 分别是
$$\langle \text{load\_video}, \text{preprocess}, \text{caption\_retrieval}(0,10) \rangle$$
$$\langle \text{load\_video}, \text{preprocess}, \text{segment\_localization}(\dots) \rangle$$

不做 stateful prefix matching 时第三个 tool 不同就导致前缀分叉；做了之后，因为第三步都是 stateless，LPM 在 `preprocess` 处命中，第二个 rollout 可重用第一个的 sandbox snapshot 只跑 `segment_localization`。

**Example 2**：stateless tool 顺序不同的两个 rollout
$$\langle \text{load}, \text{preprocess}, \text{caption}(0,10), \text{qna}(\dots,5) \rangle$$
$$\langle \text{load}, \text{preprocess}, \text{qna}(\dots,5), \text{caption}(0,10) \rangle$$

传统 cache 视为不同路径；stateful prefix matching 下两者在 `preprocess` 后共享 state，`qna` 和 `caption` 在 TCG 中都是 `preprocess` 节点的孩子（Figure 10），第二个 rollout 全部命中。

## 六、实验数据深度解读

### 6.1 Workload 配置（Table 1）

| Dataset | Agent | Hardware | # Rollouts | Max length |
|---|---|---|---|---|
| terminal-bench (easy) | Qwen3-4B-Instruct-2507 | 2×A100 80G | 8 | 2048 |
| terminal-bench (med) | Qwen3-4B-Instruct-2507 | 2×A100 80G | 8 | 2048 |
| terminal-bench (easy) | Qwen3-14B-Instruct | 8×A100 80G (cloud) | 4 | 2048 |
| terminal-bench (med) | Qwen3-14B-Instruct | 8×A100 80G (cloud) | 4 | 2048 |
| SkyRL-SQL | Qwen2.5-Coder-7B-Instruct | 8×A100 80G (cloud) | 5 | 3000 |
| EgoSchema | Qwen3-30B-A3B-Instruct-2507 | Tinker API (cloud) | 8 | 32768 |

注意 EgoSchema 用的是 Tinker（Thinking Machines 的 RLaaS，参考 [Tinker](https://thinkingmachines.ai/tinker/)），但 tool 在 self-hosted L40S 上执行。

### 6.2 Cache Hit Rate（Figure 5）

- **terminal-bench**：15%–32%
  - 4B easy：20.2%，4B med：14.2%
  - 14B easy：25.3%，14B med：16.7%
  - Larger model hit rate 更高，因为更倾向重复 tool call
- **SkyRL-SQL**：27.0%–57.2%，平均 33.11%
- **EgoSchema**：34%–73.9%，平均 64.3%

Hit rate 随 epoch 增长，因为 TCG 不断扩展分支，更多机会命中。

### 6.3 Tool Call Latency 加速（Table 2）

| Model | Difficulty | No Cache (s/call) | Cache (s/call) | Speedup |
|---|---|---|---|---|
| Qwen3-4B | Easy | 8.67 | 1.40 | 6.18× |
| Qwen3-4B | Med | 18.68 | 2.70 | 6.92× |
| Qwen3-14B | Easy | 8.07 | 2.35 | 3.44× |
| Qwen3-14B | Med | 36.23 | 6.53 | 5.55× |

最高 6.92× 的 median speedup，主要来自 proactive forking 节省的 container startup 开销。

### 6.4 SQL 场景的特殊性

SkyRL-SQL 是 stateless workload，每个 cache hit 把 tool execution 从 56.6ms 降到 6.5ms（8.7× 单次加速），平均 hit rate 33.11% → 整体 2.9× expected speedup。

### 6.5 EgoSchema 的 API token 节省

`caption_retrieval` 内部调用 OpenAI API。TVCACHE 让 token usage 降低 3×。这里 paper 还论证了为什么不能用传统 API cache：不同 task 的 `caption(segment 1, segment 2)` 签名相同但底层视频不同，输出不同。正确 cache 必须把 sandbox state（视频文件）绑进 key，这恰好就是 TVCACHE 的设计。

### 6.6 微基准（Figure 8）

- 单 server P95 latency 3.3ms @ 256 RPS，512 RPS 时 P95 超 1s（饱和）
- 16 shard 时 4096 RPS 下 P95 仅 6.1ms——近线性扩展
- Memory overhead 1-2GB（5 个 step 训练 Qwen3-4B 时）

### 6.7 Reward 不退化（Figure 6）

这是 correctness 的实证验证：TVCACHE 的 reward 曲线和 no-cache baseline 几乎完全重合。这符合预期，因为 TVCACHE 是 exact cache，保证返回的结果和真正执行的结果一致。

## 七、工程实现细节

### 7.1 Server-Client 架构（Figure 4）

- **Server**：HTTP service，管理 TCG 和 snapshots
  - 端点：`PUT /put`、`GET /get`、`POST /prefix_match`
  - Thread-safe API
  - 周期持久化 TCG 到磁盘（防 GPU crash）
- **Client**：`tvclient` pip 包，提供 `ToolCallExecutor` 和 `ToolCallEnvironment` 类
- **Sandbox lifecycle**：`ToolExecutionEnvironment` 抽象类，每个 dataset 实现 `start/stop/fork/execute` 四个方法

### 7.2 Docker Sandbox 工程优化（Appendix E）

这是 paper 的隐藏工程亮点。terminal-bench 默认 harness 用 Docker Compose，但 proactive forking 一次要起几百个 container。paper 测出三个瓶颈：

1. **Docker Compose 默认为每个 sandbox 创建独立 bridge network**——网络创建是慢操作
2. **很多 task 不需要 networking**——只有暴露端口或多 service 才需要
3. **并发 fork 过多导致 cgroup kernel-level 资源争用**——系统调用 timeout

解决：
- Pre-create pool of bridge networks + reuse
- 解析 docker-compose 判断是否真的需要 network，按需分配
- Rate-limited fork pipeline，把并发 cap 在 saturation point 之前

Figure 13 显示这些优化后 container creation rate 大幅提升。

### 7.3 Fork 实现

用 Docker 的 commit API（带 `--no-pause` 避免阻塞运行中工作负载）+ 保存环境变量和 working directory + 从 committed image 启动新 container 并恢复状态。

## 八、与 Related Work 的关系

### 8.1 Semantic Caching 系列

[Semantic caching](https://arxiv.org/abs/2502.03771)（如 GPTCache、vCache）假设 prompt → response 是 stateless mapping。TVCACHE 把这个 assumption 打破并重建：stateful tool call 的 cache key 是 **trajectory** 而非单次调用。

### 8.2 Tool-use RL 系列

- [Kimi K2](https://arxiv.org/abs/2507.20534)：用另一个 LLM 模拟 tool response 来 amortize 成本——这是 approximate，可能损失精度
- [OTC-PO](https://arxiv.org/abs/2504.00024)：把 tool cost 加进 reward function——改变了训练目标
- [VERLTOOL](https://arxiv.org/abs/2509.01055)：异步并行 tool execution——调度优化但不减少总计算量
- TVCACHE 是 **complementary** 的，可以叠加这些方法

### 8.3 veRL 和 Tinker 集成

[veRL](https://github.com/volcengine/verl) 是字节跳动的 hybrid RLHF framework，TVCACHE 用它做 self-hosted 和 cloud-hosted 实验。[Tinker](https://thinkingmachines.ai/tinker/) 是 Thinking Machines 的 RLaaS 平台，TVCACHE 用它做 EgoSchema 的 30B 模型实验。

## 九、Intuition 总结与延伸思考

### 9.1 核心 mental model

TVCACHE = **Trajectory-indexed cache** + **Copy-on-write sandbox state** + **Cost-aware snapshotting** + **Stateful prefix compression**

类比来说：
- Trie / Radix tree for prefix sharing（如网络路由、SGLang RadixAttention）
- Copy-on-write for state replication（如 OS fork、ZFS snapshot）
- LRU with cost weighting for eviction（如 database buffer pool）

这三个思想组合起来，把 RL post-training 的 tool call 冗余度高效压缩。

### 9.2 为什么这个 paper 重要

1. **指出了一个被忽视的瓶颈**：7-43% rollout time 在 GPU idle，这在 RL scaling 报告里很少被讨论
2. **正确性可证明**：stateful prefix matching 有形式化证明，这在 systems paper 里少见
3. **工程上可落地**：开源、集成主流框架、scaling 测试齐全

### 9.3 可能的延伸方向

- **Fuzzy LPM**：tool arguments 的微小差异（SQL 中别名、whitespace）导致 miss，可结合 embedding 做近似匹配——paper 在 Section 5.2 提到这个 future work
- **Cross-task sharing**：当前每个 task $p$ 独立维护 $\mathcal{G}(p)$。不同 task 间可能有共享前缀（如同一 base image 的 Docker setup），可考虑 hierarchical TCG
- **Adaptive snapshot policy**：当前是 binary 决策，可考虑 learning-based policy，预测哪些 state 更可能被 reuse
- **GPU-aware scheduling**：tool execution 期间 GPU idle，可在这段时间插入其他 rollout 的 forward pass（这接近 VERLTOOL 的异步思想，但需要更精细的调度）

### 9.4 与 PagedAttention 的深层类比

vLLM 的 PagedAttention 把 KV cache 按 page 管理，实现非连续物理内存上的逻辑连续 KV，从而支持 sharing 和 dynamic allocation。TVCACHE 把 sandbox state 按 TCG node 管理，实现非连续执行路径上的逻辑等价 state，从而支持跨 rollout sharing。两者的本质都是**把"看起来连续的资源"解构为"可组合的单元"，然后在单元级别做复用**。

### 9.5 Cache 正确性的更深层直觉

Paper 的 correctness 论证依赖 Assumption 1（tool 输出只依赖当前 state 和参数）和 sandbox 的 deterministic 假设。这在大多数 sandbox 中成立，但有几个边界情况值得思考：
- **Wall-clock dependent tool**：如 `date`、random number generator——这类 tool 本身就违反 Assumption 1。实践中需要 annotate 为 non-cacheable。
- **External API call**：tool 调用 OpenAI API 等，返回结果有 stochasticity。Paper 在 EgoSchema 中通过把整个 response 缓存（包括 LLM 生成）来保证 determinism，但需要确认训练对这种 cache 的容忍度——reward curve 不退化（Figure 6）是实证支持。
- **Network race condition**：sandbox 内如有网络依赖，不同 rollout 可能拿到不同 external state。需要 sandbox 提供 network isolation，paper 在 Appendix E 的 selective network allocation 部分隐含处理了这点。

## 十、参考链接

- 论文 arXiv（推测）：[https://arxiv.org/abs/2506.07251](https://arxiv.org/abs/2506.07251)（需自行核实）
- veRL framework：[https://github.com/volcengine/verl](https://github.com/volcengine/verl)
- Tinker RLaaS：[https://thinkingmachines.ai/tinker/](https://thinkingmachines.ai/tinker/)
- Terminal-Bench：[https://github.com/laude-institute/terminal-bench](https://github.com/laude-institute/terminal-bench)
- SkyRL：[https://github.com/NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)
- EgoSchema：[https://egoschema.github.io/](https://egoschema.github.io/)
- VideoAgent：[https://arxiv.org/abs/2503.07757](https://arxiv.org/abs/2503.07757)
- GRPO（DeepSeekMath）：[https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
- PPO：[https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
- SGLang RadixAttention：[https://arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104)
- vLLM PagedAttention：[https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- Kimi K2：[https://arxiv.org/abs/2507.20534](https://arxiv.org/abs/2507.20534)
- VERLTOOL：[https://arxiv.org/abs/2509.01055](https://arxiv.org/abs/2509.01055)
- Toolformer：[https://openreview.net/forum?id=Yacmpz84TH](https://openreview.net/forum?id=Yacmpz84TH)

这篇 paper 的精彩之处在于它把一个看似简单的"缓存"问题，通过图结构 + 最长前缀匹配 + 选择性快照 + 状态感知过滤四个机制组合，把 RL post-training 的工程效率推到了一个新的台阶。对 build intuition 而言，核心 takeaway 是：**当资源的"等价性"不能从单个 key 推断时，必须把 key 扩展为 trajectory（或更广义的 dependency graph），并在 trajectory 级别做 caching**。这个 pattern 在很多其他领域（如 build system 的 Bazel、ML pipeline 的 do-it、OS 的 process snapshot）都能找到回响。
