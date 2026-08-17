---
source_pdf: ROBOBENCH.pdf
paper_sha256: 4325a9ca270901caec35ef63d487e72fe31ec9b5361d0d9ea1f3d91bad93c7a4
processed_at: '2026-08-12T00:14:07-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这篇 paper 其实就干了一件事: 给现在那些号称能给 robot 当大脑的 MLLM 做了一次 "全身体检",然后发现大家都不及格,甚至连最强的 Gemini-2.5-Pro 也被人类吊打。

下面我用最直白的话帮你 build 一下 intuition,顺便把里面的技术细节扒开看看。

---

## 1. 为什么我们需要这个体检

现在搞 embodied AI 的人特别喜欢拿 MLLM 当 System 2(负责思考和规划的那个慢系统),把 VLA policy 当 System 1(负责执行的那个快系统)。Karpathy 你自己也在 x.com 上聊过很多次这个 dual-system paradigm。

但问题来了: 我们怎么知道这个 MLLM 大脑到底行不行?以前的 eval 方法基本属于 "瞎子摸象":
- 看 robot 有没有把任务做完: 失败了你根本不知道是没听懂话、眼瞎了、还是手残了。
- 用 BLEU 算生成 plan 的文本相似度: "开抽屉拿杯子" 和 "打开抽屉取出杯子" 都会被判错。
- 让 LLM 当裁判打分: 主观且不稳定,经常自己人包庇自己人。
- 在 simulator 里跑 task success rate: 有 sim-to-real gap,仿真里满分到了真实世界可能直接摔盘子。

RoboBench 的核心思路就是: 把 robot 执行任务的整个认知 pipeline 切成 5 个独立维度,每个维度单独考试,而且全用真实世界的数据。

---

## 2. 五个体检项目: 沿着 pipeline 切片

### (1) Instruction Comprehension: 听懂人话吗

这个维度测的是 MLLM 能不能把人的意图翻译成 actionable plan。核心设计是 paired explicit-implicit。

Explicit instruction 就是直说: "把红杯子放到盘子上"。
Implicit instruction 就是暗示: "我渴了"。

实验数据非常 striking。Table 4 显示,explicit instruction 模型平均能拿 40-50 分,一旦换成 implicit,直接掉到 10-20 分,大概 30% 的 degradation。

这说明什么? 说明现在的 MLLM 在做 string-level pattern matching,根本没有真正把语言 grounding 到 scene context 和 human intent 上。你直说 object name 它能查表查到,你给它个 demand 它就懵了。

### (2) Perception Reasoning: 看清环境吗

这个维度拆成四个 sub-task:
- **Robotic-centric**: 识别 robot type 和 viewpoint。GPT-4o 的 robot-view recognition 只有 39.38,说明 MLLM 对 embodiment-aware perception 基本没有 built-in prior。
- **Object-centric**: 物体的 static attribute 和 functional attribute。
- **Scene-centric**: spatial relation、temporal grounding、causality。Temporal grounding 是最难的,Gemini-2.5-Pro 也只有 49.68。
- **Task-centric**: 在 long-horizon instruction 里找到 relevant object。

### (3) Generalized Planning: 会做计划吗

这个是最复杂的维度,涵盖 cross-embodiment、cross-object、cross-view、cross-task。

几个直觉:
- **Dual-arm coordination**: 很多 plan 里默认 single-arm,根本没考虑 left-right arm assignment。
- **Rare object**: 遇到不常见的物体或者需要 world knowledge 的场景,plan 质量直接崩盘。这是 LLM knowledge 和 visual grounding 的 binding 问题。
- **Multi-view**: 加个 wrist camera,GPT-4o 从 33.66 涨到 38.51。有收益但远没饱和。

### (4) Affordance Prediction: 知道怎么动手吗

High-level plan 有了,具体怎么执行?这个维度测的是把 subgoal 转成 spatial cue 的能力:
- **Static affordance**: 单点 contact point,比如抓苹果抓哪儿。
- **Dynamic affordance**: 轨迹,比如拉抽屉怎么拉。
- **Navigation affordance**: base position,比如走近微波炉停在哪儿。

Metric 用的是 Euclidean distance 转成 score:
$$\text{score} = 100 \times (1 - d)^{\alpha}, \quad \alpha = 2.5$$

这里 $d$ 是 normalized distance(预测点到 ground truth 的欧氏距离除以 image 对角线长度),$\alpha = 2.5$ 是个 power-law exponent。

为什么用 $\alpha = 2.5$? 因为这是 super-linear 衰减,模拟了 robot control 里 "差一点点就抓不住" 的物理 reality。$d=0.5$ 时 score 只有 $100 \times 0.5^{2.5} \approx 17.7$,而 $d=0.2$ 时 score 是 $100 \times 0.8^{2.5} \approx 57.2$。差一点距离,分数差好几倍。

Gemini-2.5-Pro 拿了 65.21,人类是 82.63,还差 17 分。

### (5) Failure Analysis: 知道自己错了吗

这个维度最难。分成 execution-level 和 planning-level:
- **Execution failure**: 位置偏了、轨迹歪了、gripper 没夹紧。
- **Planning failure**: 抓错东西了、漏步骤了、顺序反了。

Table 6 的数据非常 telling:
- Execution failure diagnosis: 所有模型 10-20 分,人类只有 47.30 分。
- Planning failure diagnosis: 模型 40-60 分,人类 80.67 分。

这个 asymmetry 特别重要。Planning error 是 symbolic level 的,人类一眼就能看出来。Execution error 需要你理解 gripper 在 3D 空间里差了多少毫米,这是 expert-level embodied understanding,连人都觉得难。这暗示 execution failure diagnosis 可能根本不是一个 well-posed task for 当前 MLLM,我们需要专门的 fine-grained perception module。

---

## 3. World Simulator: 这篇 paper 最牛的创新

传统的 plan evaluation 有三种失效模式: BLEU 测 surface form,exact match 太严格,LLM-as-judge 太主观。

RoboBench 提出了一个全新的思路: **MLLM-as-World-Simulator**。

核心 idea 是: plan 的正确性不是文本相似度,而是 "能不能在物理世界里达成关键状态转移"。

### 公式拆解

任务被建模成 partially ordered set of atomic actions:
$$a = \langle \text{skill}, \text{object}, \text{args} \rangle$$

每个 action 由 skill(动作类型)、object(作用对象)、args(参数)组成。

然后用 DAG(Directed Acyclic Graph)来编码 action 之间的依赖关系:
$$G = (V, E)$$

- $V$: action node 集合,每个 $v \in V$ 是一个 atomic action。
- $E$: 有向边集合,每条 $(u \to v) \in E$ 表示 $u$ 必须在 $v$ 之前完成。DAG 允许 valid permutation,比如 "开抽屉" 和 "拿杯子" 可以并行,但 "开抽屉" 必须在 "从抽屉拿东西" 之前。

两个 metric:

**NodeCorrectness**:
$$\text{NodeCorrectness} = \left\lfloor \frac{|V^{\star} \cap \hat{V}|}{|V^{\star}|} \times 10 \right\rfloor$$

- $V^{\star}$: ground-truth action nodes
- $\hat{V}$: predicted action nodes
- $|V^{\star} \cap \hat{V}|$: MLLM 做 one-to-one matching 下的 exact alignment on {skill, object, parameter}
- $\lfloor \cdot \rfloor$: floor operation,把 score 离散化到 0-10 的整数
- 除以 $|V^{\star}|$ 而不是 $\max(|V^{\star}|, |\hat{V}|)$: 这是 recall-oriented,不惩罚冗余 action,冗余由 rollout 那边 catch

**TaskCompletion**: 这是核心。流程是:
1. 从初始帧 $I_0$ 提取 world state $W_0$(比如 `drawer=closed`, `apple on table`)。
2. Parse $A^{\star}$ 和 $G$,提取 state-transition predicates,aggregate 成 $S^{\star}$。$S^{\star}$ 是 critical milestones 集合,比如 `{drawer=open, apple=on_plate, apple=in_hand}`。
3. 用 $G$ enforce precedence 和 allowable parallelism,建立 causal links。
4. Step-by-step 执行 predicted actions,每步 check precondition against 当前 $W_t$,update 到 $W_{t+1}$。当某个 $s \in S^{\star}$ 变成 true 且保持 valid to its last consumer,标记为 achieved & protected,累积 $\hat{S} \subseteq S^{\star}$。

$$\text{TaskCompletion} = \left\lfloor \frac{|\hat{S}|}{|S^{\star}|} \times 10 \right\rfloor$$

最终:
$$\text{LongHorizon} = \frac{\text{NodeCorrectness} + \text{TaskCompletion}}{20} \in [0, 1]$$

除以 20 是因为两个 component 都是 0-10 的整数,加起来 0-20,normalize 到 [0,1]。

### Intuition

举个具体例子。任务是 "把苹果放到盘子里,然后关上抽屉"。

- Plan A: `pickup(apple); place(apple, plate); close(drawer)` ✅
- Plan B: `open(drawer); pickup(apple); place(apple, plate); close(drawer)` ✅(多了 open drawer 但无害,apple 本来就在桌上,不影响 critical states)
- Plan C: `pickup(apple); close(drawer); place(apple, plate)` ❌(close drawer 在 place 之前,可能 arm 被 drawer 挡住)

传统 BLEU 会把 B 判差(多了 step),exact match 把 A/B 都判差。RoboBench 的 rollout 会: 执行 B 时,`open(drawer)` 不影响 `apple=on_plate` 这个 milestone,所以 TaskCompletion 满分;NodeCorrectness 略低因为有冗余 node。综合判 B 大致正确。

这跟你想熟悉的 model-based RL 和 world model 的思路完全同源。如果 MLLM 能 rollout 出 critical state transitions,那这个 rollout 本身就能作为 RL 的 dense reward signal。

---

## 4. 实验结果: 现在的 MLLM 还差得远

### 4.1 Leaderboard 速读

Gemini-2.5-Pro 几乎在所有维度领先:
- Perception Reasoning: 62.96(human 74.30)
- Generalized Planning: 41.81(human 54.50)
- Affordance Prediction: 65.21(human 82.63)
- Failure Analysis: 45.14(human 63.99)

所有 model 在所有 dimension 上都比人类差,而且 gap 巨大,是 10-20 分量级,不是 1-2 分。

### 4.2 几个 surprising 的点

**(a) Text-only GPT-4o 居然有信号**:
- Perception: 20.81
- Planning: 33.95
- Failure Analysis: 31.55

这暗示 LLM 的 commonsense prior 在 embodied reasoning 里确实有贡献,纯 visual 信息不完全是决定性的。

**(b) Closed vs Open gap 巨大**:
- Gemini-2.5-Pro 41.81 vs Qwen2.5-VL-72B 37.73(planning)
- 但 LLaVA-OneVision-7B 只有 12.15,open-source 的 tail 非常长

**(c) Embodied-specific training 有用但不大**:
RoboBrain-2.0-7B 在 planning 上 25.35,超过 LLaVA-OneVision-7B 的 12.15,但不如 Qwen2.5-VL-72B 的 37.73。说明 embodied fine-tuning 在 small scale 下收益有限,model size 的 scaling law 仍然主导。

### 4.3 Error Analysis

Paper 把 planning error 分成四类(Figure 6a):
1. **Execution Errors**(majority): missing steps、impossible actions、redundant steps、wrong function。这是 procedural reasoning 弱。
2. **Identification Errors**: visual aliasing(把 crumpled paper 当 popcorn)、parameter mismatch、wrong object。这是 object-action binding 弱。
3. **Common Sense Errors**: physics violations(同时折多件衣服)、spatial reasoning errors(拧水龙头方向错)。这是 physical prior 弱。
4. **Mode-Specific Errors**: 不按 symbolic reference 输出(说 "the red cup" 而不是说 "Object 3")。

这个 taxonomy 可以直接拿来作为 VLA 训练数据 curation 的指南。每一类 error 都对应一种 data augmentation strategy。

---

## 5. 给 VLA 训练和架构设计的直觉启发

### 5.1 Data curation

你的 training data 不仅要覆盖 action distribution,还要覆盖 (implicit instruction × embodiment × viewpoint × failure mode) 的 joint distribution。RoboBench 给你提供了这个 joint distribution 的 measurement grid。

一个直接的 data augmentation strategy:
- 从 existing robot datasets 里 sample explicit instruction
- 用 LLM rewrite 成 K 种 implicit variants(demand-based、context-based、emotional)
- 在 VLA training 里混合 explicit + implicit

### 5.2 Architecture design

三件事分别对应 perception / reasoning / execution:
- **Visual encoder 需要 embodiment-conditioned**: robot 的 wrist camera view 跟 third-person view 的 spatial semantics完全不同。当前 MLLM 把所有 image 当成 "generic image",没有 whose eyes 的概念。
- **Language model 需要物理常识 prior**: 现在的 MLLM 在 abstract symbol level 很强,但 symbols 和 physical world 的 binding 很弱。
- **Decoder 需要能输出 structured action with parameter binding**: 现在 MLLM 输出的 plan 经常 missing step 或者 wrong function,需要更强的 structured output supervision。

### 5.3 与 VLA 生态的 connection

RoboBench 评的是 System 2,但 paper 里多次 reference 了 VLA 生态:
- **π0 / π0.5**: flow matching VLA,System 1 + System 2 unified
- **OpenVLA**: open-source autoregressive VLA
- **CogAct**: cognition + action synergistic
- **HybridVLA**: diffusion + autoregressive hybrid
- **Gemini Robotics**: Google 的 embodied foundation model
- **GR00T N1**: NVIDIA 的 humanoid foundation model
- **RoboBrain 2.0**: BAAI 的 embodied MLLM

RoboBench 实际上给 VLA 训练提供了一个 pre-training target specification。如果你想训一个 VLA,RoboBench 的 5 个 dimensions 就是你的 capability checklist,每个 dimension 的 error pattern 就是你的 failure mode coverage。

更深一层: RoboBench 的 MLLM-as-world-simulator 给 VLA 评估提供了一个 meta-level 的思路。能不能用 world model 来做 RL reward? 这跟 model-based planning 的思路完全同源。

### 5.4 几个 critical thoughts

**(a) Self-reference 风险**:
MLLM-as-world-simulator 用 MLLM 评 MLLM,本质是 self-referential evaluation。如果 evaluator MLLM 和被评 MLLM 有 similar blind spots,错误会被 mutually mask。Paper Section 5.4 用 human Pearson correlation 做了 validation,但 438 sample 可能不够。更严谨的做法是 ablate 不同 evaluator MLLM,看 score variance。

**(b) DAG annotation 的 scalability**:
每个 task 都要 human-annotated DAG,相当 expensive。如果想 scale 到 100k+,需要 automated DAG construction。可能用 LLM 从 video transcript + frame diff 自动 infer state transitions。

**(c) 3D affordance 的 extension**:
RoboBench 现在只测 2D point prediction。未来一个自然的 extension 是加入 3D affordance prediction dimension,用 point cloud 或 NeRF 表达,metric 用 Chamfer distance 或 Earth Mover's Distance。

**(d) Multi-agent coordination**:
RoboBench 现在主要评 single-agent,但 multi-agent coordination 是未来 humanoid robot 的关键能力。建议 RoboBench v2 加入 multi-agent dimension。

---

## 6. 总结

RoboBench 告诉我们三件事:

1. **MLLM 当 embodied brain 还远远不够**。最强的 Gemini-2.5-Pro 在 planning 上 41.81,人类 54.50;在 failure analysis 上 45.14,人类 63.99。Gap 是 10-20 分量级。

2. **当前 MLLM 的 weakness pattern 非常 specific**: implicit instruction grounding、embodiment-aware perception、long-horizon plan 的 physical feasibility、fine-grained affordance、execution failure diagnosis。这五个 weakness 都是 grounding 问题,MLLM 在 abstract symbol level 很强,但 symbols 和 physical world 的 binding 很弱。

3. **评估方法本身需要 evolve**。BLEU / exact match / LLM-as-judge 都不够。MLLM-as-world-simulator 是一个 promising direction,它把 evaluation 从 "string matching" 升级到 "state transition matching"。

对 VLA 训练的指导: 你的 training data 不仅要覆盖 action distribution,还要覆盖 (implicit instruction × embodiment × viewpoint × failure mode) 的 joint distribution。

对 model architecture 的指导: visual encoder 需要 embodiment-conditioned,language model 需要物理常识 prior,decoder 需要能输出 structured action with parameter binding。

---

## 参考链接

- RoboBench project page: https://robo-bench.github.io/
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- RoboMIND: https://arxiv.org/abs/2412.13877
- AgiBot World: https://arxiv.org/abs/2503.06669
- RH20T: https://arxiv.org/abs/2307.00595
- Thinking in Space: https://arxiv.org/abs/2412.14171
- RoboVQA: https://arxiv.org/abs/2311.00899
- EgoPlanBench: https://arxiv.org/abs/2312.06722
- OpenEQA: https://arxiv.org/abs/2312.06701
- EmbodiedBench: https://arxiv.org/abs/2502.09560
- VLABench: https://arxiv.org/abs/2412.18194
- RoboBrain 2.0: https://arxiv.org/abs/2507.02029
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246
- Octo: https://arxiv.org/abs/2405.12213
- CogAct: https://arxiv.org/abs/2411.19650
- HybridVLA: https://arxiv.org/abs/2503.10631
- RDT-1B: https://arxiv.org/abs/2410.07864
- Gemini Robotics: https://arxiv.org/abs/2503.20020
- GR00T N1: https://arxiv.org/abs/2503.14734
- Gemini 2.5: https://arxiv.org/abs/2507.06261
- PaLM-E: https://arxiv.org/abs/2303.04211
- VoxPoser: https://arxiv.org/abs/2307.05973
- ReKep: https://arxiv.org/abs/2409.01652
- RoboPoint: https://arxiv.org/abs/2406.10721
- MMbench: https://arxiv.org/abs/2307.06281
- LIBERO: https://arxiv.org/abs/2306.03310
- RoboOS: https://arxiv.org/abs/2505.03673
- Code-as-Monitor: https://arxiv.org/abs/2502.01407
- Chain-of-Affordance: https://arxiv.org/abs/2412.20451
- Data Scaling Laws in IL: https://arxiv.org/abs/2410.18647

如果你想往任何一个 sub-direction 深挖,比如 MLLM-as-world-simulator 的 self-reference 风险、affordance prediction 的 3D extension、或者 implicit instruction 的 data augmentation strategy,我可以再展开。

---

# RoboBench: 一份给 MLLM 作为 Embodied Brain 的全面体检报告

Karpathy, 这篇 paper 我觉得正好戳中了你这几年反复强调的一个 tension: MLLM 看似啥都能做, 但一旦把它放到 physical world 里当 "brain" 用, 我们对它真实能力的测量方式其实非常粗糙。RoboBench (Peking University + BAAI, https://robo-bench.github.io/) 试图把这件事 systematize。下面我 build up 一下 intuition。

---

## 1. 为什么需要这个 benchmark:现有 eval 的三重失真

Paper 在 Table 1 把现有 embodied AI benchmark 摆在一起对比,我觉得这张表本身就是一个 mini-survey,值得逐行读:

| Benchmark | 关键缺陷 |
|---|---|
| RoboVQA (Sermanet et al., 2024) | 用 BLEU 评 plan,完全忽略 plan 的结构性和 physical feasibility |
| EgoPlanBench (Chen et al., 2024a) | Multi-choice 形式,偏离真实开放任务 |
| MMRo (Li et al., 2024c) | LLM Score 主观且不稳定 |
| EAI / EmbodiedBench (Yang et al., 2025) | 完全依赖 simulator task success rate,有 sim-to-real gap |
| OpenEQA (Majumdar et al., 2024) | 只测 perception 单维度 |
| VLABench (Zhang et al., 2024a) | 只用 sim,真实 robot 缺失 |

RoboBench 的卖点用三个关键词概括:
1. **Comprehensiveness**: 5 个 dimensions 沿着 manipulation pipeline 走,而不是孤立的 perception / planning。
2. **Realism**: 数据混合 Open X-Embodiment (https://arxiv.org/abs/2310.08864)、RoboMIND (Wu et al., 2024, https://arxiv.org/abs/2412.13877)、AgiBot World (https://arxiv.org/abs/2503.06669)、RH20T (Fang et al., 2023)、in-house collection——全部来自真实 robot。
3. **Planning 评估的范式转换**: 提出 MLLM-as-world-simulator,跳过 symbolic matching。

这一点我觉得和你之前在 "State of GPT" 演讲里讲的 "eval 太像 train distribution 就废了" 完全一致。BLEU / multi-choice / LLM-as-judge 都是在测量 "语言表面上的相似度",但 embodied plan 的本质是 *状态转移*。

---

## 2. 五个 dimensions:沿着 execution pipeline 切片

RoboBench 把 embodied brain 的认知流程切成五个 stage,这是 paper 的核心组织逻辑:

### (1) Instruction Comprehension
关键设计是 paired explicit-implicit:
- Explicit: "Pick up the red cup and place it on the plate."
- Implicit: "I'm thirsty." → 需要推断出 "retrieve a drink"。

实验结果(Table 4)非常 striking:explicit 平均 40–50 分,implicit 直接掉到 10–20 分,**30% 的 degradation**。这跟你常说的 "LLM 在 surface form 上 overfit" 高度一致——模型在做 string-level pattern matching,没有真正 grounding 到 scene context 和 human intent。

### (2) Perception Reasoning
四个 sub-dimensions:
- **Robotic-centric**: 识别 robot type (single-arm / dual-arm / mobile / humanoid) 和 viewpoint。这里 GPT-4o 的 robot-view recognition 只有 39.38,说明 MLLM 对 *embodiment-aware perception* 基本没有 built-in prior。
- **Object-centric**: static attribute(颜色、形状)+ functional attribute(可抓 / 可推 / 可开)。
- **Scene-centric**: spatial relation + temporal grounding + causality。Temporal grounding 是最难的——Gemini-2.5-Pro 也只有 49.68。
- **Task-centric**: 在 long-horizon instruction 中 identify 出 relevant object。

### (3) Generalized Planning
最 complex 的 dimension,涵盖 cross-embodiment / cross-object / cross-view / cross-task。这里有几个直觉值得 build:
- **Cross-embodiment**: dual-arm 的 left-right arm assignment 是个 hidden bottleneck。Plan 里 "pick up cup" 没说哪只手,模型默认 single-arm。
- **Cross-object**: rare object + world knowledge 触发的 plan 质量崩盘。这是 LLM knowledge 与 visual grounding 的 *binding* 问题。
- **Cross-view**: 加 wrist camera 让 GPT-4o 从 33.66 → 38.51,Claude-3.7 从 44.51 → 48.19。多视角收益真实存在但远未 saturate。

### (4) Affordance Prediction
把 high-level subgoal 转成 spatial cue 喂给 System 1:
- **Static affordance**: 单点 contact point(grasp an apple 的 grasp point)
- **Dynamic affordance**: 轨迹(open a drawer 的拉出轨迹)
- **Navigation affordance**: base position(approach microwave 的停靠点)

Metric 用 Euclidean distance,然后 transform 成 score:
$$\text{score} = 100 \times (1 - d)^{\alpha}, \quad \alpha = 2.5$$
其中 $d$ 是 normalized distance(到 ground truth 的欧氏距离除以 image 对角线),$α=2.5$ 是一个 power-law exponent,作用是 *punish large errors heavily*——$d=0.5$ 时 score 只有 $100 \times 0.5^{2.5} ≈ 17.7$,而 $d=0.2$ 时 score 是 $100 \times 0.8^{2.5} ≈ 57.2$。这种 super-linear 衰减模拟了 robot control 里"差一点点就抓不住"的物理 reality。

Gemini-2.5-Pro 拿到 65.21,比第二名 Qwen2.5-VL-72B 的 56.67 高出 8.5 分,但人类是 82.63——还差 17 分。这跟 RoboPoint (Yuan et al., 2024, https://arxiv.org/abs/2406.10721) 的发现一致:**VLM 对 pixel-level spatial grounding 的精度是当前最大的 affordance bottleneck**。

### (5) Failure Analysis
最难的一个 dimension。分 execution-level(position misalignment、trajectory deviation、gripper error)和 planning-level(wrong object、missing step、wrong order)。

Table 6 显示:
- Execution failure diagnosis: 所有模型 10–20 分,**人类只有 47.30 分**!
- Planning failure diagnosis: 模型 40–60 分,人类 80.67 分

这个 asymmetry 我觉得特别重要。Planning error 是 symbolic level 的,人类一眼就能看出"步骤错了";execution error 需要你 *理解 gripper 在 3D 空间里差了多少毫米*,这是 expert-level embodied understanding,连人都觉得难。这是 paper 里最 actionable 的 insight 之一:**execution failure diagnosis 可能根本不是一个 well-posed task for 当前 MLLM**,我们需要专门的 fine-grained perception module。

---

## 3. MLLM-as-World-Simulator:这篇文章的核心方法创新

这是我觉得最值得深挖的部分。传统 plan evaluation 有三种失效模式:
1. BLEU / ROUGE:测 surface form,plan "先开抽屉再拿杯子" vs "先打开抽屉然后取出杯子" 会被判为不同。
2. Exact match:太严格,任何 valid permutation 都判错。
3. LLM-as-judge:subjective,可能 GPT-4 自己生成就给自己高分。

RoboBench 提出用 **DAG + critical state milestones + MLLM rollout**:

### 公式拆解

任务建模为 partially ordered set of atomic actions:
$$a = \langle \text{skill}, \text{object}, \text{args} \rangle$$

Reference DAG:
$$G = (V, E)$$
- $V$:action node 集合,每个 $v \in V$ 是一个 atomic action。
- $E$:有向边集合,每条 $(u \to v) \in E$ 表示 $u$ 必须在 $v$ 之前完成。DAG 允许 valid permutation(比如"开抽屉"和"拿杯子"可以并行,但"开抽屉"必须在"从抽屉拿东西"之前)。

两个 metric:

**NodeCorrectness**:
$$\text{NodeCorrectness} = \left\lfloor \frac{|V^{\star} \cap \hat{V}|}{|V^{\star}|} \times 10 \right\rfloor$$
- $V^{\star}$:ground-truth action nodes
- $\hat{V}$:predicted action nodes
- $|V^{\star} \cap \hat{V}|$:MLLM 做 one-to-one matching 下的 exact alignment on {skill, object, parameter}
- $\lfloor \cdot \rfloor$:floor operation,把 score 离散化到 0–10 的整数
- 除以 $|V^{\star}|$ 而不是 $\max(|V^{\star}|, |\hat{V}|)$:这是 recall-oriented,不惩罚冗余 action(冗余由 rollout 那边 catch)

**TaskCompletion**:这是真正的核心。流程是:
1. **Visual constraint analysis**:从初始帧 $I_0$ 提取 world state $W_0$(e.g., `drawer=closed`, `apple on table`)。
2. **Critical object-state detection**:parse $A^{\star}$ 和 $G$,提取 state-transition predicates,aggregate 成 $S^{\star}$。$S^{\star}$ 是 "critical milestones" 集合,比如 `{drawer=open, apple=on_plate, apple=in_hand}`。
3. **State order & concurrency validation**:用 $G$ enforce precedence 和 allowable parallelism,建立 causal links。
4. **Rollout simulation**:step-by-step 执行 predicted actions,每步:
   - Check precondition against 当前 $W_t$
   - Update 到 $W_{t+1}$
   - 当某个 $s \in S^{\star}$ 变成 true 且保持 valid to its last consumer,标记为 achieved & protected
   - 累积 $\hat{S} \subseteq S^{\star}$

$$\text{TaskCompletion} = \left\lfloor \frac{|\hat{S}|}{|S^{\star}|} \times 10 \right\rfloor$$

最终:
$$\text{LongHorizon} = \frac{\text{NodeCorrectness} + \text{TaskCompletion}}{20} \in [0, 1]$$
除以 20 是因为两个 component 都是 0–10 的整数,加起来 0–20,normalize 到 [0,1]。

### Intuition
这个设计的精髓在于:**plan 的 "正确性" 不是文本相似度,而是 "能否在物理世界中达成 critical state transitions"**。

举个例子:任务是"把苹果放到盘子里,然后关上抽屉"。
- Plan A: `pickup(apple); place(apple, plate); close(drawer)` ✅
- Plan B: `open(drawer); pickup(apple); place(apple, plate); close(drawer)` ✅(多了 open drawer 但无害,apple 本来就在桌上,open drawer 不影响 critical states)
- Plan C: `pickup(apple); close(drawer); place(apple, plate)` ❌(close drawer 在 place 之前,可能 apple 已经被 arm 挡住)

传统 BLEU 会把 B 判差(多了 step),exact match 把 A/B 都判差。RoboBench 的 rollout 会:执行 B 时,`open(drawer)` 不影响 `apple=on_plate` 这个 milestone,所以 TaskCompletion 满分;NodeCorrectness 略低因为有冗余 node。综合判 B 大致正确。

### Q2 和 Q3

**Q2 (Next-step planning)**:
$$\text{NextStep} = \frac{s_{\text{skill}} + s_{\text{obj}} + s_{\text{param}}}{3} \in [0, 1]$$
- $s_{\text{skill}} \in \{0, 1\}$:skill 名字必须 exact match
- $s_{\text{obj}}, s_{\text{param}} \in \{0, 0.5, 1\}$:object 和 parameter 用 MLLM judge reasonableness under visual constraints
- 除以 3 normalize

**Q3 (Task state estimation)**:
$$\text{StateAcc} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}\{\hat{y}_i = y_i\}$$
- $n$:样本数
- $\hat{y}_i \in \{0, 1\}$:模型预测
- $y_i \in \{0, 1\}$:ground truth
- $\mathbb{1}\{\cdot\}$:indicator function

### Human validation
Paper Section 5.4 做了 human study:从 Q1 inference set 均匀采样 438 个 task,expert 打 0–20 分,然后比较:
- MLLM-as-world-simulator 的 Pearson $r$
- LLM pairwise trajectory scoring baseline 的 Pearson $r$

Figure 6b/c 显示 MLLM-as-world-simulator 与 human judgment 的 alignment 显著更高。这是 metric validity 的关键证据。

---

## 4. 实验结果的关键 takeaways

### 4.1 Leaderboard 速读

Gemini-2.5-Pro (https://arxiv.org/abs/2507.06261) 几乎在所有 dimensions 领先:
- Perception Reasoning: 62.96(human 74.30)
- Generalized Planning: 41.81(human 54.50)
- Affordance Prediction: 65.21(human 82.63)
- Failure Analysis: 45.14(human 63.99)

**所有 model 在所有 dimension 上都比人类差,而且 gap 巨大**。这跟你常说的 "LLM 看起来很强,但真实任务上还是远不及人类" 完全吻合。

### 4.2 几个 surprising 的点

**(a) Text-only GPT-4o 居然有信号**:
- Perception: 20.81(已经很低但有)
- Planning: 33.95
- Failure Analysis: 31.55

这暗示 LLM 的 commonsense prior 在 embodied reasoning 里 *确实有贡献*,纯 visual 信息不完全是决定性的。这跟 PaLM-E (Driess et al., 2023, https://arxiv.org/abs/2303.04211) 的发现一致。

**(b) Closed vs Open gap 巨大**:
- Gemini-2.5-Pro 41.81 vs Qwen2.5-VL-72B 37.73(planning)
- 但 LLaVA-OneVision-7B 只有 12.15——open-source 的 tail 非常长

**(c) Embodied-specific training 有用但不大**:
- RoboBrain-2.0-7B (https://arxiv.org/abs/2507.02029) 在 planning 上 25.35,超过 LLaVA-OneVision-7B 的 12.15,但不如 Qwen2.5-VL-72B 的 37.73。
- 说明 embodied fine-tuning 在 small scale 下收益有限,模型 size 的 scaling law 仍然主导。

**(d) Multi-view 的边际收益**:
- GPT-4o: 33.66 → 38.51(+4.85)
- Claude-3.7: 44.51 → 48.19(+3.68)
- 远没有 close the gap to human

### 4.3 Error Analysis(Figure 6a)

Paper 把 planning error 分成四类:
1. **Execution Errors**(majority): missing steps、impossible actions、redundant steps、wrong function。这是 procedural reasoning 弱。
2. **Identification Errors**: visual aliasing(把 crumpled paper 当 popcorn)、parameter mismatch、wrong object。这是 object-action binding 弱。
3. **Common Sense Errors**: physics violations(同时折多件衣服)、spatial reasoning errors(拧水龙头方向错)。这是 physical prior 弱。
4. **Mode-Specific Errors**: 不按 symbolic reference 输出(说 "the red cup" 而不是 "Object 3")。

这个 taxonomy 我觉得可以直接拿来作为 *VLA 训练数据 curation* 的指南——每一类 error 都对应一种 data augmentation strategy。

---

## 5. 与 VLA 生态的 connection

RoboBench 评的是 System 2,但 paper 里多次 reference 了 VLA 生态:

- **π0 / π0.5** (Black et al., 2024/2025, https://arxiv.org/abs/2410.24164): flow matching VLA,System 1 + System 2 unified。
- **OpenVLA** (Kim et al., 2024, https://arxiv.org/abs/2406.09246): open-source autoregressive VLA。
- **Octo** (https://arxiv.org/abs/2405.12213): generalist robot policy。
- **CogAct** (Li et al., 2024e, https://arxiv.org/abs/2411.19650): cognition + action synergistic。
- **HybridVLA** (Liu et al., 2025, https://arxiv.org/abs/2503.10631): diffusion + autoregressive hybrid。
- **RDT-1B** (Liu et al., 2024b, https://arxiv.org/abs/2410.07864): bimanual diffusion foundation model。
- **Gemini Robotics** (Team et al., 2025, https://arxiv.org/abs/2503.20020): Google 的 embodied foundation model。
- **GR00T N1** (Bjorck et al., 2025, https://arxiv.org/abs/2503.14734): NVIDIA 的 humanoid foundation model。
- **RoboBrain 2.0** (https://arxiv.org/abs/2507.02029): BAAI 的 embodied MLLM,paper 里被作为 embodied baseline。

我读 paper 时的联想:**RoboBench 实际上给 VLA 训练提供了一个 *pre-training target specification***。如果你想训一个 VLA,RoboBench 的 5 个 dimensions 就是你的 capability checklist,每个 dimension 的 error pattern 就是你的 failure mode coverage。

更深一层:RoboBench 的 MLLM-as-world-simulator 给 VLA 评估提供了一个 *meta-level* 的思路——**能不能用 world model 来做 RL reward?** 这跟你想必熟悉的 "world model as imagination-based planner" 思路(Dreamer, Ha & Schmidhuber)是同源的。如果 MLLM 能 rollout 出 critical state transitions,那么这个 rollout 本身可以作为 RL 的 dense reward signal。

---

## 6. 数据构建 pipeline 的细节

Figure 3 把每个 dimension 的 data construction workflow 画出来了:

### Instruction Comprehension
- Explicit instruction 来自 daily-life scenarios
- 用 LLM rewrite 成 implicit(demand-based)form
- Prompt 见 Figure 20

### Perception Reasoning
- **Robotic-centric**: real robot data + type/view metadata → template-based QA
- **Object-centric**: static attributes from Gao et al. (2024, https://arxiv.org/abs/2404.07914) + GPT-generated functional properties + distractors
- **Scene-centric**: Gemini 做 video step segmentation → temporal grounding;manual annotation → spatial relations + keypoints
- **Task-centric**: human-labeled bounding boxes 关联 long-horizon instructions 到 target objects

### Generalized Planning
- Planning pool 来自 Open X-Embodiment、RoboMIND、AgiBot World、RH20T、Thinking in Space (Yang et al., 2024, https://arxiv.org/abs/2412.14171)
- Gemini 生成 structured annotations:task summary、step-wise instructions with timestamps、metadata(objects/actions/scenes/embodiments)
- Human annotator refine
- 每步映射到 function template:`pickup(object)`、`moveto(object, target)` 等,group 成 manipulation 或 navigation skill list
- 三种 question types:Q1(long-horizon)、Q2(next-step)、Q3(state estimation)

### Affordance Prediction
- 从 planning pool 采样 representative frames
- 三类标注:static contact points、dynamic trajectories、mobile base positions

### Failure Analysis
- **Execution-level**: 从 RoboMIND 收集真实 failure case,expert 标注
- **Planning-level**: 由于真实 planning failure 数据稀缺,*synthesize* by perturbing correct instructions(wrong object / missing step / wrong order)

### Quality Control
两阶段:
1. **Construction-time filtering**:general rules(image quality、task validity)+ sub-benchmark-specific rules
2. **Post-construction validation**: 20 个 trained annotator 做 majority vote。所有 model 都答对的题删掉(太简单),所有 model 都答错的题进入 manual review(可能题本身有问题)

这个 majority vote 的设计很巧妙——它把 *model performance* 当成 quality signal,跟 MMbench (Liu et al., 2024d, https://arxiv.org/abs/2307.06281) 的思路一脉相承。

### Dataset 统计(Table 2)
- Total items: 4038
- Total questions: 6092
- Multiple-choice: 1875(perception + affordance + failure)
- Open-ended(planning + instruction): 1973 + 842 + 1192 = 4007
- Affordance point prediction: 252 + 150 = 402
- Avg planning steps: 6.74
- Unique task instructions: 1403
- Unique answers: 1462

---

## 7. 我的几个 critical thoughts

### 7.1 MLLM-as-world-simulator 的 self-reference 风险
这个方法用 MLLM 评 MLLM,本质是 *self-referential evaluation*。风险在于:
- 如果 evaluator MLLM(他们用 Gemini)和被评 MLLM 有 *similar blind spots*,那么错误会被 mutually mask。
- Paper Section 5.4 用 human Pearson correlation 做了 validation,但 438 sample 是否足够?
- 更严谨的做法应该是 ablate 不同 evaluator MLLM,看 score variance。

### 7.2 DAG annotation 的 scalability
每个 task 都要 human-annotated DAG,这相当 expensive。Paper 说 6092 QA,但 DAG 的 annotation 成本没有量化。如果想 scale 到 100k+,需要 *automated DAG construction*——可能用 LLM 从 video transcript + frame diff 自动 infer state transitions。

### 7.3 Implicit instruction 的 30% gap
这是 paper 最 actionable 的 finding。当前 MLLM 训练数据里 implicit instruction 太少。一个直接的 data augmentation strategy:
- 从 existing robot datasets 里 sample explicit instruction
- 用 LLM rewrite 成 K 种 implicit variants(demand-based、context-based、emotional)
- 在 VLA training 里混合 explicit + implicit

这跟你常提的 "data is all you need" 完全 aligned。

### 7.4 Embodiment-aware perception 的缺失
Robotic-view recognition 只有 39.38(GPT-4o),这个数字让我想到一个问题:**MLLM 的 visual encoder 是否需要 embodiment-conditioned positional encoding?** 一个 robot 的 wrist camera view 跟 third-person view 的 spatial semantics完全不同。当前 MLLM 把所有 image 当成 "generic image",没有 *whose eyes* 的概念。

### 7.5 与 Eureka、VoxPoser、ReKep 的关系
Paper reference 了 VoxPoser (Huang et al., 2023, https://arxiv.org/abs/2307.05973) 和 ReKep (Huang et al., 2024, https://arxiv.org/abs/2409.01652)。这两个工作的核心是 *LLM generates spatial constraints as 3D value maps / keypoints*。RoboBench 的 affordance prediction dimension 实际上是在 eval 这种能力的 *primitive version*——如果模型连 2D point 都预测不准,3D value map 更无从谈起。

未来一个自然的 extension:在 RoboBench 里加入 **3D affordance prediction dimension**,用 point cloud 或 NeRF 表达,metric 用 Chamfer distance 或 Earth Mover's Distance。

### 7.6 System 1 / System 2 的 dual-system paradigm
Paper 反复提到 dual-system paradigm,引用了 Black et al. (2024)、Bjorck et al. (2025)、Driess et al. (2023)。这跟 Kahneman 的 "Thinking, Fast and Slow" 框架对应:
- **System 1**: low-level control,VLA policy,fast、reactive
- **System 2**: MLLM embodied brain,slow、deliberative

RoboBench 只评 System 2,这是个合理的 scope decision,但也意味着它 *不评端到端 task success*。完整的 embodied eval 应该是 RoboBench(System 2)+ LIBERO (Liu et al., 2023a, https://arxiv.org/abs/2306.03310) 或 RLBench (James et al., 2020) 的 System 1 benchmark 联合使用。

### 7.7 RoboOS 与 multi-agent
Paper reference 了 RoboOS (Tan et al., 2025, https://arxiv.org/abs/2505.03673),这是 hierarchical framework for cross-embodiment 和 multi-agent collaboration。RoboBench 现在主要评 single-agent,但 *multi-agent coordination* 是未来 humanoid robot 的关键能力。建议 RoboBench v2 加入 multi-agent dimension。

### 7.8 Code-as-Monitor 的 connection
Paper 还 reference 了 Code-as-Monitor (Zhou et al., 2025b, https://arxiv.org/abs/2502.01407),这是个 *constraint-aware visual programming* 框架,用于 reactive failure detection。RoboBench 的 failure analysis dimension 可以看作 Code-as-Monitor 能力的 *retrospective* eval——Code-as-Monitor 是 reactive(实时检测),RoboBench 是 retrospective(事后诊断)。两者的能力 spectrum 应该联合考虑。

### 7.9 Affordance 与 Chain-of-Affordance
Paper reference 了 Chain-of-Affordance (Li et al., 2024b, https://arxiv.org/abs/2412.20451),这是 VLA 训练里把 affordance 作为中间 representation 的工作。RoboBench 的 affordance dimension 实际上可以用来 *measure Chain-of-Affordance 风格 VLA 的 intermediate representation quality*。

### 7.10 数据 scaling law
Paper reference 了 Data Scaling Laws in Imitation Learning (Lin et al., 2024a, https://arxiv.org/abs/2410.18647)。一个 open question:**RoboBench 的 score 与 VLA 训练数据量是什么 scaling 关系?** 如果能 plot 出 "RoboBench planning score vs training demos count" 的 power law,这会是一个非常有指导意义的 figure。

---

## 8. 一个总体的 intuition 总结

RoboBench 告诉我们三件事:

1. **MLLM 当 embodied brain 还远远不够**。最强的 Gemini-2.5-Pro 在 planning 上 41.81,人类 54.50;在 failure analysis 上 45.14,人类 63.99。Gap 是 10–20 分量级,不是 1–2 分。

2. **当前 MLLM 的 weakness pattern 非常 specific**:implicit instruction grounding、embodiment-aware perception、long-horizon plan 的 physical feasibility、fine-grained affordance、execution failure diagnosis。这五个 weakness 都是 *grounding* 问题——MLLM 在 abstract symbol level 很强,但 symbols 和 physical world 的 binding 很弱。

3. **评估方法本身需要 evolve**。BLEU / exact match / LLM-as-judge 都不够。MLLM-as-world-simulator 是一个 promising direction,它把 evaluation 从 "string matching" 升级到 "state transition matching",这跟 RL 里 model-based planning 的思路一致。

对 VLA 训练的指导:**你的 training data 不仅要覆盖 action distribution,还要覆盖 (implicit instruction × embodiment × viewpoint × failure mode) 的 joint distribution**。RoboBench 给你提供了这个 joint distribution 的 *measurement grid*。

对 model architecture 的指导:**visual encoder 需要 embodiment-conditioned,language model 需要物理常识 prior,decoder 需要能输出 structured action with parameter binding**。这三件事分别对应 perception / reasoning / execution,RoboBench 的五个 dimensions 就是这三件事的 *fine-grained decomposition*。

---

## 参考链接

- RoboBench project page: https://robo-bench.github.io/
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- RoboMIND: https://arxiv.org/abs/2412.13877
- AgiBot World: https://arxiv.org/abs/2503.06669
- RH20T: https://arxiv.org/abs/2307.00595
- Thinking in Space: https://arxiv.org/abs/2412.14171
- RoboVQA: https://arxiv.org/abs/2311.00899
- EgoPlanBench: https://arxiv.org/abs/2312.06722
- OpenEQA: https://arxiv.org/abs/2312.06701
- EmbodiedBench: https://arxiv.org/abs/2502.09560
- VLABench: https://arxiv.org/abs/2412.18194
- RoboBrain 2.0: https://arxiv.org/abs/2507.02029
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246
- Octo: https://arxiv.org/abs/2405.12213
- CogAct: https://arxiv.org/abs/2411.19650
- HybridVLA: https://arxiv.org/abs/2503.10631
- RDT-1B: https://arxiv.org/abs/2410.07864
- Gemini Robotics: https://arxiv.org/abs/2503.20020
- GR00T N1: https://arxiv.org/abs/2503.14734
- Gemini 2.5: https://arxiv.org/abs/2507.06261
- PaLM-E: https://arxiv.org/abs/2303.04211
- VoxPoser: https://arxiv.org/abs/2307.05973
- ReKep: https://arxiv.org/abs/2409.01652
- RoboPoint: https://arxiv.org/abs/2406.10721
- MMbench: https://arxiv.org/abs/2307.06281
- LIBERO: https://arxiv.org/abs/2306.03310
- RoboOS: https://arxiv.org/abs/2505.03673
- Code-as-Monitor: https://arxiv.org/abs/2502.01407
- Chain-of-Affordance: https://arxiv.org/abs/2412.20451
- Data Scaling Laws in IL: https://arxiv.org/abs/2410.18647

如果你想往任何一个 sub-direction 深挖(比如 MLLM-as-world-simulator 的 self-reference 风险、affordance prediction 的 3D extension、或者 implicit instruction 的 data augmentation strategy),我可以再展开。
