---
source_pdf: From Perception to Action.pdf
paper_sha256: 93eb86634d0ce33d7d7f89e10b60c35614468013198eeae10de2a29711fd9abd
processed_at: '2026-08-04T10:45:17-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Paper

## 一句话总结

这篇 paper 说了一件特别简单的事：**现在的 AI 能看懂图片能聊天，但真让它去物理世界里干活，它蠢得令人发指。** 然后这篇 paper 花了 200 页解释为啥蠢，怎么才能不蠢。

---

## 到底蠢在哪？

举个特别具体的例子。你给 GPT-4V 看一张厨房照片，问它 "杯子在哪"，它能回答 "红色杯子在桌子上"。看起来挺聪明对吧？

但如果你让一个机器人去拿那个杯子，它会：
- 撞桌角
- 抓空气
- 估算距离差好几个数量级

GPT-4V 在 SpatialBench 上 **40% 的空间关系题答错**。这不是随机噪声级别的错误，是系统性的 spatial blindness。

作者的核心论点是：**"能把图片和文字关联起来" 和 "理解物理空间" 是两个完全不同的能力。**

前者叫 symbolic grounding——VLM 干这个很溜。后者叫 spatial grounding——需要对几何、物理、动作后果有 metric 级别的理解。LLM 的架构里压根没有 3D structure、physical dynamics、geometric constraints 的 internal model。

你可以把 LLM 想成一个**从来没碰过东西的盲人学者**。他读了所有关于物理的书，能跟你聊量子力学，但你让他倒杯水，他把水壶打翻了。

---

## 三个维度看这个问题

作者搞了个三轴 taxonomy，听着学术，其实特别直觉。

### 第一轴：你要干啥

四种活：
- **Navigation**：从 A 走到 B，室内导航、自动驾驶都属于这类
- **Scene Understanding**：看懂 3D 场景，知道哪是墙哪是桌子
- **Manipulation**：抓东西、放东西、用工具
- **Geospatial**：卫星图、城市交通、平方公里级别的分析

### 第二轴：你怎么思考和行动

- **Memory**：记不住东西就干不了长任务。短期记忆靠 context window，长期靠 RAG，但 spatial memory 最关键——你得记住厨房的门在哪、柜子在哪，这些是 persistent geometric structure
- **Planning**：从 reactive（看到啥直接反应）到 world model-based（先在脑子里 simulate 一遍再动）是一个谱系
- **Tool Use**：调 API、写代码、执行物理动作

### 第三轴：多大尺度

这个最关键，作者反复强调 **scale mismatch 是迁移失败的主要原因**：

- **Micro（<1m）**：抓个杯子，厘米级精度，靠触觉传感器和机械臂
- **Meso（1m-100m）**：在房间里导航，靠相机和激光雷达，SLAM 能搞定
- **Macro（>100m）**：城市规划、卫星图分析，靠外部传感器，没法直接 actuate

为啥 scale 重要？因为一个优化到厘米精度的 grasping policy，完全没法规划城市路线。一个交通预测器里根本没有 contact force 的概念。不同 scale 的 features、physics、action spaces 是 qualitatively different 的。

作者发现 **68% 的方法都在搞 meso-scale**（房间级导航），micro 和 macro 严重没人做，尽管这俩商业价值巨大。

---

## LLM 做 planning 到底有多烂

作者给了一组数据，我觉得特别有价值：

| 失败模式 | Benchmark | 数据 |
|---------|-----------|------|
| 忽略几何约束，plan 出物理上不可能的动作序列 | BEHAVIOR-1K | 只有 12% 成功率 |
| 不满足 action precondition | VirtualHome | 35% 的 plan 失败 |
| 长任务 credit assignment 崩溃 | ALFRED | 5步任务 65% 成功 → 20步任务 18% |
| 遇到意外不会 replan | RoboTHOR | 40% 失败率 |

翻译成人话：LLM 生成的 plan 看着很流畅，"先开冰箱，拿牛奶，关冰箱，倒杯子"——但物理上可能根本走不通。它不知道冰箱门被椅子挡住了，不知道牛奶盒太大拿不出来，不知道杯子在柜子最上层够不着。

经典 planner（PDDL、HTN）能保证 precondition satisfaction，能 verify plan correctness。LLM 把这些都丢了，换来的是 semantic flexibility。问题是 safety-critical 场景里，你不能 "看起来对但实际会撞死人"。

---

## Memory 为啥是核心问题

这篇 paper 有一句话我觉得特别精准：

> 一个 30 observations/second 的 agent，128K token context window，**90 分钟就耗尽了**。

你想想一个真实的 home robot，它得从早上 8 点干到晚上 8 点。12 小时。context window 根本撑不住。

所以 memory 不是 nice-to-have，是 **existential requirement**。

更关键的是，spatial memory 和普通 memory 本质不同：

- **Episodic memory** 记录 "下午 3 点机器人撞了椅子"——这是一个 event
- **Spatial memory** 记录 "椅子在桌子左边 0.5 米处"——这是一个 persistent geometric structure，跟时间无关

VLMaps（一个 semantic map 方法）在 100 步以上不更新地图后出现 **semantic drift**——它开始 "忘记" 东西在哪，或者把位置记错。这就是 temporal drift failure mode。

---

## GNN + LLM 为啥是个好方向

这篇 paper 的第二个 key finding。

LLM 的 context window 是 transient 的，记不住 structured spatial state。Transformer 把 input 当 unordered set 处理，对 "谁挨着谁"、"哪个路口连哪个路口" 这种 relational structure 天然 agnostic。

GNN 的 message passing 特别直觉：每个 node 听邻居说话，聚合信息，更新自己。一个机器人 joint 通过 sensing 相邻 joint 来理解整个 arm 的 configuration。

**Graph 本质上是 LLM 的 spatial memory prosthetic。**

把 scene graph、road network、object configuration 编码成 graph，用 GNN 处理，agent 就有了 persistent、structured 的 spatial state，能跨 reasoning step 存活，能 survive context window 限制。

具体的例子：Graph WaveNet 学了一个 adaptive adjacency matrix。预定义的 "物理上相连的路口" 图谱会遗漏重要 dependency——两个地理上很远但通勤模式相同的区域可能强相关。GNN 从 traffic pattern 中自动 discover 这些隐藏连接。

---

## World Model 为啥是 safety 的前提

第三个 key finding。

**没有预测能力，agent 就无法避免不可逆的物理伤害。** 打碎的鸡蛋不能复原，撞坏的车不能撤销。

World model 的核心 idea：不直接预测未来图像（太贵、compounding error 严重），而是压缩到 latent space，在 latent space 里预测 dynamics。

Dreamer 系列的演进路径特别清晰：
- Dreamer（2020）：continuous latent，sample-efficient
- DreamerV2（2021）：discrete latent，Atari 人类水平
- DreamerV3（2023）：symlog prediction，**一个算法搞定 150+ 任务不用调参**

DayDreamer 更猛——在 simulation 里学的 world model 直接 transfer 到物理机器人，minimal real-world fine-tuning。这是 sim-to-real gap 的一条 promising path。

但作者做了一个关键区分，我觉得整个 paper 最 sharp 的 insight 之一：

**Visual fidelity ≠ Actionable prediction。**

Sora 能生成极其逼真的视频。Genie 2 能生成交互式 3D 环境。但 Sora **没有 action conditioning**——你没法问它 "如果我采取 action A，会发生什么？"

一个 world model 如果要用于 planning，必须有 action-conditioned structure。纯粹的 video generation 再逼真，对 agent decision-making 没用。这就像一段高清录像和一台驾驶模拟器的区别——前者好看，后者能学东西。

---

## Industry Patterns 其实就是四种姿势

作者把工业界的部署抽象成 4 个 design pattern，比直接列公司能力清单有 insight 得多。

### Pattern 1: Human-in-the-Loop

AI 提方案 → 人验证 → 反馈更新系统。高风险场景必备。Palantir 的 geospatial intelligence、ESRI 的 GIS workflow 都是这种。每一步都要人签字，throughput 低但安全。

### Pattern 2: Weakly Supervised Planetary-Scale

NASA-IBM 的 Prithvi 是典型：用 petabyte 级无标注卫星图做 self-supervised pretraining，然后少量标注 fine-tune。行星尺度上 dense labeling 经济上根本不可能。

### Pattern 3: Agent-Assisted

跟 HITL 的区别很关键：这里 **AI 是 primary analyst，人只管异常**。AutonomousGIS、GeoGPT 是代表。Throughput 高很多，但要求错误是 recoverable 的。

### Pattern 4: Embodied AI at Scale

Safety 是 **primary design constraint**，不是 feature。Waymo 用 lidar + camera 做冗余 depth estimation，高成本但 safety margin 大。Tesla 纯视觉，成本低但对 perception model 的 reliability 要求极端。两种 paradigm 都遵循：massive simulation → 谨慎上线 → fleet data 持续学习。

---

## Benchmark 有多烂

作者的分析特别犀利。现有 benchmark 有 5 个 structural deficit：

1. **Sim-to-Real 没人测**：RT-1 在 simulation 97% 成功，真实机器人 68%。但没有任何 benchmark 专门测这个 degradation
2. **Metric 太粗糙**：binary success 忽略 partial progress。离 goal 1cm 的 agent 和压根没动的 agent 得分一样
3. **Safety 没人测**：near-miss 不扣分，risky behavior 不扣分
4. **Long-horizon 没人测**：没有 multi-day benchmark。大部分 episode 就几十到几百步
5. **Cross-scale 没人测**：这是最 critical 的。一个 home robot 得同时做 macro（规划房间路径）+ meso（避障）+ micro（抓杯把）。没有任何 benchmark 测这种 integration

这些不是 "missing rows in a table"，是 **field 根本没在 measure 这些维度**。说明 community 还没真正把 spatial agent 当 integrated system 来 evaluate。

---

## 六个 Grand Challenges 翻译成人话

1. **Unified Representation**：现在抓东西用 point cloud，导航用 topological map，地理分析用 raster image。能不能搞一个 representation 从 object part 一直 span 到 city infrastructure？

2. **Grounded Long-Horizon Planning**：LLM 能想但不懂几何，TAMP 懂几何但不够灵活。能不能搞个 hybrid，LLM 负责 semantic reasoning，geometric verifier 负责 physical feasibility check？

3. **Safe Deployment**：自动驾驶、手术机器人、基础设施——这些场景需要 formal safety guarantee，当前系统完全没有

4. **Sim-to-Real**：simulation 里学的 policy 到真实世界就崩。需要 photorealistic simulation + accurate physics + domain randomization + real-world fine-tuning

5. **Multi-Agent Coordination**：仓库机器人、自动驾驶车队——大量 agent 协调，communication 有限，partial observability

6. **Edge Deployment**：foundation model 太大，edge device 跑不动。需要 compression without capability loss

---

## 给研究者的 take-away

如果我要从这篇 paper 里提取对做研究最有用的 insight，大概是这几个：

**Spatial grounding 和 symbolic grounding 是 orthogonal capabilities。** 你把 LLM 做再大，把 VLM 做再强，spatial grounding 不会 emergent。它需要 fundamentally different 的 architectural components：metric representations、geometric constraint verifiers、physical dynamics models。

**Scale 是 causal 维度，不只是分类标签。** 选错 scale 你的 representation 就不对，physics 就不对，action space 就不对。做 micro 的方法不能直接套到 macro，反过来也不行。

**Memory 是 existential requirement，不是 optimization。** 真实世界的 task 跨小时跨天，context window 必然耗尽。不解决 persistent spatial memory，就没法做 long-horizon agent。

**World model 的价值在 action conditioning，不在 visual fidelity。** 别被 Sora 的视频质量迷惑了。对 agent planning 有用的是 "如果我做 A，世界会变成什么样"，不是 "生成一段好看的视频"。

**Benchmark 的缺失反映了 field 的 fragmentation。** navigation 社区、manipulation 社区、geospatial 社区各自为政，没有 cross-scale、cross-domain 的 unified evaluation。这是 structural barrier，不是加几行数据能解决的。

---

# From Perception to Action: Spatial AI Agents and World Models 深度解析

## 1. Paper 核心动机与定位

这篇 survey 的核心论点：**LLM 在 symbolic domain 的成功不能直接迁移到 physical world**。作者提出了一个 critical distinction：

- **Symbolic grounding**：将 image 和 text 关联（VLM 擅长）
- **Spatial grounding**：对 geometry、physics、action consequences 的 metric 级理解（LLM 完全缺乏）

关键 insight 是 perception ≠ agency。一个 model 能描述 "there is a red cup on the table"，但完全无法 grasp 那个 cup。这个 gap 是 embodied AI 的核心 bottleneck。

Paper 引用超过 2000 篇，citing 742 works，时间范围 2018-2026，下界选 2018 是因为 BERT 开启 foundation model era，Habitat 和 AI2-THOR 建立了 standardized embodied AI benchmarks。

参考链接：
- Habitat: https://habitat.ai/
- AI2-THOR: https://ai2thor.allenai.org/

---

## 2. 三轴 Taxonomy 深度解析

这是 paper 的核心贡献。三个 axis 构成一个 3D design space：

### Axis 1: Spatial Task（做什么）

四个 category：
- **Navigation**：point-goal, object-goal, VLN, autonomous driving, off-road
- **Scene Understanding**：3D geometry perception, object recognition, spatial relationship reasoning
- **Manipulation**：grasping, placement, tool use
- **Geospatial Analysis**：satellite imagery, urban computing, GIS

### Axis 2: Agentic Capability（如何推理和行动）

- **Memory**：short-term (in-context), long-term (RAG), episodic, spatial memory
- **Planning**：reactive → hierarchical → search-based → world model-based（一个 spectrum）
- **Tool Use & Action**：API integration, code generation, physical action primitives, skill libraries

### Axis 3: Spatial Scale（空间粒度）

这个 axis 最关键，paper 强调 scale mismatch 是 transfer failure 的 primary source：

| Scale | Range | Sensing/Actuation Boundary |
|-------|-------|---------------------------|
| Micro | <1m | Tactile sensors, robot arm workspace |
| Meso | 1m-100m | Onboard cameras/lidar, metric SLAM tractable |
| Macro | >100m | Satellites, city-wide sensor networks, no direct actuation |

**关键发现**：68% 的方法 target meso-spatial tasks，micro 和 macro 严重 underexplored。

---

## 3. Memory Systems 技术深度

### 3.1 四种 Memory 类型对比

Paper 做了一个非常清晰的区分，我展开讲：

**Short-Term Memory (In-context)**：
- 机制：attention mechanism 条件化 on prompt demonstrations
- 局限：受 context window 限制。Paper 给出一个惊人数字：30 observations/second 的 agent，128K token context 在 90 分钟内耗尽

**Long-Term Memory (RAG)**：
- MemGPT: hierarchical memory management，类似 OS 的 memory hierarchy
- AMEM: agentic memory for LLMs
- MemEvolve: meta-evolution of agent memory

**Episodic Memory**：
- 记录 "what happened where"："robot collided with chair at 3pm"
- 代表方法：Neural Episodic Control (Pritzel et al., 2017)
- 公式核心：$Q(s,a) \leftarrow Q(s,a) + \alpha(r + \gamma \max_{a'} Q(s',a') - Q(s,a))$，但 episodic 版本用 k-NN lookup 替代 gradient update

**Spatial Memory**（paper 强调这是最 critical 的）：
- 记录 "the structure of where itself"：chair 相对于 table 的位置，independent of event
- 实现分类：
  - Cognitive maps (Tolman 1948, O'Keefe & Nadel 1978, Moser grid cells 2008)
  - Topological representations (Kuipers SSH, Choset & Nagatani)
  - Metric maps (SLAM: ORB-SLAM3, DROID-SLAM)
  - Neural SLAM (Chaplot et al.)
  - Semantic maps (VLMaps)
  - Scene graphs (3D Dynamic Scene Graph, Hydra)

### 3.2 Spatial Failure Modes（关键诊断）

Paper 识别了 4 类 failure，每类都 traceable 到具体的 representational gap：

1. **Spatial hallucination**：GPT-4V 在 SpatialBench 上 40% 的 spatial relationship questions 失败
2. **Reference frame confusion**：VLN agents 因 egocentric/allocentric 混淆导致 15-20% error
3. **Scale insensitivity**：SayCan 的 affordance model 在 object scale 和 training 不同时失败
4. **Temporal drift**：VLMaps 在 100+ steps 无 map update 后出现 semantic drift

这些数字非常有价值，说明当前系统的 failure 不是随机噪声，而是 systematic representational deficiency。

参考：
- SpatialVLM: https://arxiv.org/abs/2401.12168
- VLMaps: https://arxiv.org/abs/2210.05714

---

## 4. Planning Systems 架构解析

### 4.1 Planning 谱系

Paper 把 planning 描述为一个 spectrum，从 reactive 到 model-based：

```
Reactive Policy ──── Hierarchical ──── Search-based ──── World Model-based
(fast, brittle)     (temporal abs)    (MCTS, ToT)        (slow, robust)
```

### 4.2 Chain-of-Thought 和 Tree of Thoughts

CoT 的本质是 step-by-step decomposition。Self-consistency 通过 multiple reasoning paths 投票提升 reliability。

Tree of Thoughts (ToT) 的搜索结构：

```
        Root
       / | \
      A  B  C     ← Thought level 1
     /|  |  |\
    ... ... ...   ← Thought level 2
```

每个 node 是一个 reasoning state，通过 LLM 生成 children，用 evaluation function 剪枝。

### 4.3 TAMP (Task and Motion Planning)

这是 paper 认为连接 symbolic 和 geometric 的关键方法。TAMP 的架构：

```
Symbolic Planner (PDDL/HTN)
        ↓ subgoals
Motion Planner (RRT, trajectory optimization)
        ↓ trajectories
Geometric Verifier (collision, kinematics)
        ↓ feedback
Symbolic Planner (replan if infeasible)
```

Paper 指出 LLM-based planner 的 4 类 failure，这些数字非常重要：

| Failure Mode | Benchmark | Metric |
|-------------|-----------|--------|
| Geometric constraint violation | BEHAVIOR-1K | 12% success only |
| Precondition unmet | VirtualHome | 35% plans fail |
| Long-horizon credit assignment | ALFRED | 65% → 18% (5→20 steps) |
| No dynamic replanning | RoboTHOR | 40% failure on obstacles |

这些数据说明：LLM 能生成 fluent 但 physically impossible 的 plans。

参考：
- BEHAVIOR-1K: https://arxiv.org/abs/2210.07528
- ALFRED: https://arxiv.org/abs/1912.01734

---

## 5. GNN-LLM Integration 深度技术讲解

这是 paper 的第二个 key finding。Graph 作为 LLM 的 externalized spatial memory。

### 5.1 为什么 GNN 适合 spatial reasoning

Transformer 把 input 当作 unordered set 或 sequence with learned positional encoding，对 spatial data 的 relational structure 是 agnostic 的。GNN 显式 encode relationships as edges。

### 5.2 Message Passing Framework 公式解析

核心公式（公式 4 和 5）：

$$\mathbf{m}_v^{(l)} = \text{AGGREGATE}^{(l)}\left(\{\mathbf{h}_u^{(l-1)} : u \in \mathcal{N}(v)\}\right)$$

$$\mathbf{h}_v^{(l)} = \text{UPDATE}^{(l)}\left(\mathbf{h}_v^{(l-1)}, \mathbf{m}_v^{(l)}\right)$$

变量解释：
- $\mathbf{h}_v^{(l)}$：node $v$ 在第 $l$ 层的 hidden representation
- $\mathbf{m}_v^{(l)}$：node $v$ 在第 $l$ 层从邻居聚合得到的 message
- $\mathcal{N}(v)$：node $v$ 的邻居集合
- 上标 $(l)$：layer index，从 0 到 $L$
- AGGREGATE：permutation invariant 函数（sum, mean, max）
- UPDATE：通常是一个 MLP

**几何直觉**：每个 node "听" 邻居说话，收集信息，然后基于听到的更新自己的 representation。类比：robot joint 通过 sensing 相邻 joint 的位置来理解整个 arm 的 configuration。

### 5.3 Spatio-Temporal GNN（宏观尺度核心）

公式 1（general STGNN）：

$$\mathbf{H}^{(l+1)} = \sigma\left(\mathbf{A}\mathbf{H}^{(l)}\mathbf{W}^{(l)} + \text{Temporal Conv}(\mathbf{H}^{(l)})\right)$$

变量解释：
- $\mathbf{H}^{(l)} \in \mathbb{R}^{N \times d}$：所有 $N$ 个 nodes 在第 $l$ 层的 feature matrix
- $\mathbf{A} \in \mathbb{R}^{N \times N}$：adjacency matrix，encode 哪些 locations 互相 influence
- $\mathbf{W}^{(l)} \in \mathbb{R}^{d \times d'}$：第 $l$ 层的可训练 weight matrix
- $\sigma$：nonlinear activation (ReLU, GELU)
- Temporal Conv：捕获 time 维度的 pattern evolution

第一项 $\mathbf{A}\mathbf{H}^{(l)}\mathbf{W}^{(l)}$ 是 spatial message passing，第二项是 temporal evolution。

### 5.4 DCRNN（Diffusion Convolutional Recurrent NN）

公式 2：

$$\mathbf{H}^{(l)} = \sum_{k=0}^{K}\left(\mathbf{P}_f^k \mathbf{X}\mathbf{W}_{k,1} + \mathbf{P}_b^k \mathbf{X}\mathbf{W}_{k,2}\right)$$

变量解释：
- $\mathbf{P}_f$：forward transition matrix，建模 downstream propagation（congestion 从 intersection 传播到出去的路）
- $\mathbf{P}_b$：backward transition matrix，建模 upstream effects（downstream congestion 倒灌回 feeding roads）
- $k$：diffusion step，从 0 到 $K$，表示传播的 hop 数
- $\mathbf{X}$：input feature matrix
- $\mathbf{W}_{k,1}, \mathbf{W}_{k,2}$：第 $k$ hop 的 forward/backward weights

**物理直觉**：交通拥堵像热扩散一样在 road network 上传播。这个公式是 random walk on graph 的 K-step diffusion 的离散化。

参考：
- DCRNN: https://arxiv.org/abs/1707.01926
- Graph WaveNet: https://arxiv.org/abs/1906.00127

### 5.5 Graph WaveNet 的自适应邻接矩阵

公式 3：

$$\tilde{\mathbf{A}} = \text{SoftMax}\left(\text{ReLU}\left(\mathbf{E}_1\mathbf{E}_2^T\right)\right)$$

变量解释：
- $\mathbf{E}_1 \in \mathbb{R}^{N \times d}$, $\mathbf{E}_2 \in \mathbb{R}^{N \times d}$：learnable node embeddings
- $\mathbf{E}_1\mathbf{E}_2^T \in \mathbb{R}^{N \times N}$：source-destination compatibility matrix
- $\text{ReLU}$：滤除 negative correlations
- $\text{SoftMax}$：normalize 成 probability distribution

**关键 insight**：predefined adjacency（基于物理 road connection）会遗漏重要 dependency。两个 distant locations 可能因 shared commuter patterns 强相关，即使没有直接 road 连接。这个 learnable adjacency 从 traffic pattern 中 discover 哪些 locations 应该 connected。

### 5.6 GNN-LLM Integration 的概念

Graphs serve as externalized spatial memory for LLM agents。LLM 擅长 semantic reasoning 但 lack persistent structured spatial representations。通过把 scene graphs, road networks, object configurations 编码为 graphs，用 GNN 处理，agents 可以 maintain spatial state 跨 reasoning steps，survive context window limitations。

代表方法：
- LLaGA: language-graph alignment
- GraphGPT: graph reasoning through language models
- Graph instruction tuning

### 5.7 Equivariant Networks

公式 6：

$$f(T_g \cdot x) = T_g \cdot f(x)$$

变量解释：
- $T_g$：group transformation（rotation, translation, reflection）
- $f$：neural network function
- $x$：input（point cloud, molecular structure）

**含义**：如果你 transform input by $T_g$，output 也自动 transform by $T_g$。这对 spatial reasoning 至关重要：一个 cup 的 graspability 不应该因为它被旋转了 90 度就改变。

参考：
- E3NN: https://e3nn.org/
- SE(3)-Transformers: https://arxiv.org/abs/2006.10503

---

## 6. World Models 深度解析

这是 paper 的第三个 key finding。

### 6.1 Latent Dynamics Models 架构

三个核心组件（公式 7-9）：

**Encoder**：
$$\mathbf{z}_t = q_\phi(\mathbf{z}_t | \mathbf{o}_{\leq t}, \mathbf{a}_{<t})$$

变量解释：
- $\mathbf{z}_t$：时刻 $t$ 的 latent state
- $\mathbf{o}_{\leq t}$：observation history（images, depth, etc.）
- $\mathbf{a}_{<t}$：action history
- $q_\phi$：encoder network，参数 $\phi$
- 这个是 posterior，给定 observation 推断 latent

**Dynamics Model**：
$$\hat{\mathbf{z}}_{t+1} = p_\theta(\hat{\mathbf{z}}_{t+1} | \mathbf{z}_t, \mathbf{a}_t)$$

变量解释：
- $\hat{\mathbf{z}}_{t+1}$：predicted next latent state
- $\mathbf{z}_t$：current latent state
- $\mathbf{a}_t$：current action
- $p_\theta$：transition model，参数 $\theta$
- 这个是 prior，不依赖 observation，pure prediction

**Decoder**：
$$\hat{\mathbf{o}}_t = p_\psi(\hat{\mathbf{o}}_t | \mathbf{z}_t)$$

变量解释：
- $\hat{\mathbf{o}}_t$：reconstructed observation
- $p_\psi$：decoder network，参数 $\psi$

### 6.2 Dreamer 系列演进

Paper 强调 Dreamer → DreamerV2 → DreamerV3 的演进是一个关键 research trajectory：

| Model | Latent Space | Key Innovation | Achievement |
|-------|-------------|----------------|-------------|
| Dreamer (2020) | Continuous (RSSM) | Latent imagination for sample efficiency | - |
| DreamerV2 (2021) | Discrete | Human-level Atari | 55 Atari games |
| DreamerV3 (2023) | Mixed (symlog) | Single algorithm, cross-domain | 150+ tasks, no tuning |

**RSSM (Recurrent State-Space Model)** 是 Dreamer 的核心架构：

```
           Deterministic Path
z_{t-1} ──────GRU──────→ h_t
  |                         |
  |    Stochastic Path     |
  └─────── q(z_t|h_t,o_t) ←─┘
            ↓
           z_t
```

$h_t$ 是 deterministic recurrent state，$z_t$ 是 stochastic latent。两者结合 captures both predictable dynamics 和 stochastic aspects of environment。

### 6.3 DayDreamer 的 Sim-to-Real 突破

Paper 特别强调 DayDreamer (Wu et al., 2023) 展示了 world model 从 simulation transfer 到 physical robot 的潜力，with minimal real-world fine-tuning。这是克服 sim-to-real gap 的 promising path。

### 6.4 Video World Models 分类

Paper 做了一个 critical distinction：

**Controllable World Models**（适合 agent planning）：
- Genie: 控制 character in generated world
- Genie 2: 扩展到 3D environments
- Action-conditioned prediction

**Generative World Models**（视觉惊艳但不适合 planning）：
- GAIA-1: realistic driving videos
- Sora: large-scale video generation
- 缺少 action-conditioned structure，无法用于 decision-making

这个 distinction 非常重要：**visual fidelity ≠ actionable prediction**。一个 model 能生成逼真视频，但如果没有 action conditioning，就无法回答 "如果我采取 action A，会发生什么？"

参考：
- DreamerV3: https://arxiv.org/abs/2301.04104
- DayDreamer: https://arxiv.org/abs/2206.14176
- Genie 2: https://arxiv.org/abs/2412.13212
- Sora: https://openai.com/sora

---

## 7. VLA Models 架构解析

### 7.1 RT-1 到 RT-X 演进

| Model | Data Scale | Key Innovation |
|-------|-----------|----------------|
| RT-1 (2022) | 130K demos | Large-scale robot learning |
| RT-2 (2023) | Web-scale | VLM pretraining transfer |
| RT-X (2023) | Multi-institution | Cross-embodiment learning |
| OpenVLA (2024) | Open-source | 7B params, open weights |
| π₀ (2024) | Flow matching | General robot control |

### 7.2 π₀ 的 Flow Matching 创新

π₀ 使用 flow matching 替代 standard action prediction。Flow matching 的核心：

给定 action distribution $p(a)$，学习一个 vector field $v_\theta(a_t, t)$ 使得：

$$\frac{da_t}{dt} = v_\theta(a_t, t)$$

从 noise $a_0 \sim \mathcal{N}(0, I)$ flow 到 target action $a_1$。这比 discrete action tokens 更适合 continuous control。

参考：
- OpenVLA: https://arxiv.org/abs/2406.09246
- π₀: https://arxiv.org/abs/2410.24164
- RT-2: https://arxiv.org/abs/2307.15818

---

## 8. Industry Design Patterns 分析

Paper 把 industry deployments 抽象为 4 个 design patterns，这是一个非常聪明的做法。

### 8.1 Pattern 1: Human-in-the-Loop Spatial Reasoning

```
AI proposes → Human validates → Feedback updates memory/policy
```

适用场景：high-stakes, accountability requirements
代表：Palantir geospatial intelligence, ESRI ArcGIS

### 8.2 Pattern 2: Weakly Supervised Planetary-Scale Learning

```
Self-supervised pretraining (petabytes unlabeled) 
    → Task-specific fine-tuning (minimal labels)
```

代表：NASA-IBM Prithvi (Harmonized Landsat Sentinel-2)
关键 insight：planetary scale dense labeling 经济上不可行

### 8.3 Pattern 3: Agent-Assisted Workflows（与 HITL 对比）

Paper 给出了一个清晰的对比表：

| Dimension | HITL (Pattern 1) | Agent-Assisted (Pattern 3) |
|-----------|-----------------|---------------------------|
| Primary operator | Human | AI Agent |
| AI role | Proposal generator | Primary analyst |
| Human role | Mandatory validator | Exception handler |
| Validation | Every decision | Anomalies only |
| Throughput | Lower (human-gated) | Higher (agent-driven) |
| Risk tolerance | Low | Moderate |

代表：AutonomousGIS, GeoGPT, Foursquare, Carto

### 8.4 Pattern 4: Embodied AI at Scale（Safety-first）

**Safety 是 primary design constraint，不是 feature**。

代表：Autonomous vehicles
- Waymo/Cruise: lidar-centric, redundant depth, high safety margin
- Tesla: vision-only, lower cost, extreme demand on perception reliability

两种 paradigm 都遵循：massive simulation → cautious real-world deployment → continuous fleet learning

---

## 9. Benchmark Critical Analysis

### 9.1 Benchmark 覆盖度分析

Table 3 揭示的关键 pattern：

- Navigation benchmarks dominate meso-scale
- Micro-scale manipulation 和 macro-scale geospatial 严重 underrepresented
- 没有 benchmark 评估 sim-to-real transfer degradation
- 没有 benchmark 评估 cross-scale reasoning

### 9.2 SPL Metric 公式解析

$$\text{SPL} = \frac{1}{N} \sum_{i=1}^{N} S_i \cdot \frac{\ell_i}{\max(\ell_i, p_i)}$$

变量解释：
- $N$：episode 数量
- $S_i \in \{0, 1\}$：episode $i$ 是否成功到达 goal
- $\ell_i$：reference shortest path length（ground truth optimal）
- $p_i$：agent 实际走过的 path length
- $\max(\ell_i, p_i)$：防止 agent 走 shortcut 时分母小于 $\ell_i$

**直觉**：同时 reward task completion（$S_i$）和 path efficiency（$\ell_i/p_i$ ratio）。如果 agent 成功但绕远路，efficiency term < 1 会惩罚。

### 9.3 评估的 5 个 Critical Gaps

1. **Sim-to-Real Gap**：RT-1 在 simulation 97% success，real robot 68%
2. **Metric Limitations**：binary success 忽略 partial progress
3. **Safety Metrics 缺失**：no penalty for near-misses
4. **Long-Horizon**：没有 multi-day benchmark
5. **Cross-Scale**：最 critical gap，没有 single task 评估 micro+meso+macro integration

参考：
- RT-1: https://arxiv.org/abs/2212.06817
- R2R: https://arxiv.org/abs/1806.02724

---

## 10. Grand Challenges 详细解读

### 10.1 Challenge 1: Unified Spatial Representation

当前问题：不同 scale 用不同 representation
- Point clouds for grasping
- Topological maps for navigation
- Raster imagery for geospatial

**目标**：hierarchical scene graphs spanning object parts → city infrastructure

### 10.2 Challenge 2: Grounded Long-Horizon Planning

LLM 生成 high-level plans 但 struggle with geometric constraints。TAMP 处理 geometry 但 lack semantic flexibility。

**解决方案方向**：
- Hybrid neuro-symbolic planners（LLM reasoning + geometric verification）
- Hierarchical planning with learned abstractions
- World models predicting 语义和几何 consequences

### 10.3 Challenge 3: Safe Deployment Under Uncertainty

需要：
- Uncertainty quantification for spatial predictions
- OOD detection for novel environments
- Formal verification of spatial reasoning
- Graceful degradation under adversarial conditions

### 10.4 Challenge 4: Sim-to-Real Transfer

方法：
- Photorealistic simulation with accurate physics
- Domain randomization and adaptation
- Real-world fine-tuning with minimal data
- Hybrid simulation-real training pipelines

### 10.5 Challenge 5: Scalable Multi-Agent Coordination

- Emergent communication protocols
- Decentralized planning with global consistency
- Heterogeneous agent coordination
- Reliable coordination under partial observability

### 10.6 Challenge 6: Efficient Edge Deployment

- Model compression without capability loss
- Efficient architectures for spatial reasoning
- Hardware-software co-design
- Adaptive compute allocation

---

## 11. 核心直觉总结

从 Karpathy 的视角，这篇 paper 给我几个关键直觉：

**直觉 1：Scale 是 causal 维度，不只是 taxonomic**
一个 grasping policy 优化到 centimeter precision 无法 plan city-scale routes。这不是工程问题，是 representation 的 fundamental mismatch。Scale 决定了 relevant features、physics、action spaces 都 qualitatively different。

**直觉 2：Perception 和 Agency 之间的鸿沟**
VLM 能描述 scene 但无法 act。这个 gap 需要 memory（persistent spatial state）、planning（geometric constraint verification）、tool use（physical action primitives）三个 capability 同时具备才能 bridge。

**直觉 3：Graph 作为 LLM 的 spatial memory prosthetic**
LLM 的 context window 是 transient 的，无法 maintain structured spatial state。GNN + graph representation 提供了一种 externalized, persistent 的 spatial memory，让 LLM 能 "记住" scene graph、road network、object configuration。

**直觉 4：World Model 是 spatial safety 的 prerequisite**
没有 predictive capability，agent 无法避免 irreversible physical damage。Latent dynamics models (Dreamer) 提供了 sample-efficient 的 imagination，但 visual fidelity (Sora) 和 action-conditioned prediction (Genie) 是 fundamentally different 的 capability。

**直觉 5：Benchmark 的 structural deficit 反映 field 的 fragmentation**
没有 cross-scale benchmark、没有 long-horizon benchmark、没有 safety metric——这些不是 missing rows，是 field 根本没有在 measure 这些维度。这说明 community 还没有真正把 spatial agent 当作 integrated system 来 evaluate。

---

## 12. 对未来的启示

Paper 提出的 SpatialAgentBench 框架的 8 个 research direction 值得关注，特别是：
- **ManipSeq**：long-horizon manipulation with state tracking
- **SafeNav**：safety-constrained navigation
- **MultiAgent**：coordinated multi-agent spatial tasks

这三个方向如果实现，将填补当前最大的 evaluation gaps。

对于研究者来说，最有价值的 insight 是：**spatial grounding 和 symbolic grounding 是 orthogonal capabilities**，需要 fundamentally different 的 architectures。当前 LLM + VLM 的 paradigm 只解决了 symbolic grounding，spatial grounding 需要 metric representations、geometric constraints、physical dynamics——这些是 LLM 架构本身无法 subsume 的。

参考汇总：
- Paper GitHub (如有): 建议关注 AtlasPro AI 后续工作
- DreamerV3: https://danijar.com/project/dreamerv3/
- OpenVLA: https://openvla.github.io/
- Habitat: https://aihabitat.org/
- NASA-IBM Prithvi: https://huggingface.co/IBM/nasa-geospatial
- Waymo EMMA: https://waymo.com/blog/2024/10/introducing-emma

这篇 survey 的最大价值在于建立了一个 unified framework，让 fragmented subcommunities (robotics, navigation, geospatial, GNN) 有了 common vocabulary，并系统性地暴露了 cross-scale integration、long-horizon planning、safety verification 这些 structural barriers。
