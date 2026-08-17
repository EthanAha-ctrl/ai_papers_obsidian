---
source_pdf: Long-Horizon Manipulation via.pdf
paper_sha256: 59e860a23e538f964be256e31debe84d45cc129345693825625161a00922c618
processed_at: '2026-08-05T15:51:59-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 LoHo-Manip

Andrej, 我尽量把jargon都剥掉, 用最直白的方式说清楚这篇paper在干嘛、为什么这么干、work不work。

---

## 这篇paper到底在解决什么问题

一句话: **机器人做长任务时, 一步错就全崩了。**

现在VLA模型(π0, π0.5, OpenVLA这些)做单步任务很猛——抓个杯子、推个盒子, 成功率很高。但你让它"烧一壶水", 这中间要: 找杯子→抓杯子→走到水龙头→接水→关水→走到 kettle→开盖→倒水→关盖。十几个步骤, 每步哪怕只有5%的失误率, 累积下来成功率就掉到一半以下。

更要命的是, 训练时模型见的是expert demo的smooth trajectory, 一旦真机执行偏离一点, 后续observation就进入training distribution外, 模型直接懵掉。

这就是long-horizon manipulation的核心痛点。paper开头一句话点得很准: **real tasks rarely end after a single grasp**。

参考: π0.5 paper https://arxiv.org/abs/2504.16054

---

## 核心idea: 把"想"和"做"拆开

paper的思路特别clean, 就两句话:

1. **让一个VLM负责"想"**: 看当前画面, 决定接下来该干啥、去哪儿干
2. **让一个VLA负责"做"**: 只管眼前这一小段动作, 听VLM的指挥

这跟人干活一个道理。你做饭的时候不会在脑子里把每一秒的动作都规划好, 你是: 看一眼灶台→决定"该切洋葱了"→切→再看→决定"该开火了"→开火。每一步都是基于当前状态重新decision, 不是一开始就open-loop走到底。

VLM用的是Qwen3-VL-4B, VLA用的是π0.5。两个模块各干各的, 互不干扰。

参考: Qwen3-VL https://arxiv.org/abs/2502.13923

---

## 关键设计1: Receding Horizon(每步重新想)

这是整个paper最精彩的地方, 名字借自control theory里的Model Predictive Control。

**做法**: 每隔100个executor step, 把manager重新调一次, 让它看当前画面, 重新输出"还剩哪些事没干"。

**为什么这么牛**: 假设第3步"抓杯子"失败了, 杯子掉地上了。下一帧manager重新看, 发现杯子还在地上没被抓起来, 它输出的remaining plan里依然包含"grasp the cup", trace重新指向地上的杯子。Executor再试一次。

**整个recovery过程没有任何hand-crafted规则, 没有failure detector, 没有state machine**。全是receding horizon + observation-conditioned re-prediction自然涌现出来的。

对比一下SayCan或Inner Monologue那种显式语言feedback loop, 它们需要模型verbalize"我失败了", LoHo-Manip让recovery完全implicit在re-prediction里, 更elegant。

参考:
- SayCan https://say-can.github.io/
- Inner Monologue https://arxiv.org/abs/2207.05608

---

## 关键设计2: Progress-aware Plan(同时输出"干完了啥"和"还剩啥")

manager每次输出两部分:

$$
C_t^\star = [\bar{s}^{(1)}, \ldots, \bar{s}^{(k(t)-1)}], \quad R_t^\star = [\bar{s}^{(k(t))}, \ldots, \bar{s}^{(K)}]
$$

人话翻译:
- $C_t^\star$: 已完成的subtask列表, 比如"done: 找杯子, 抓杯子"
- $R_t^\star$: 还没干的subtask列表, 比如"remaining: 走到水龙头, 接水, 关水"
- $k(t)$: 当前在第几个primitive
- $K$: 总共多少个primitive

**为什么不只输出remaining, 还要output completed?**

三个原因, 都是工程上的实战考虑:

1. **Self-check**: 模型自己predict已完成序列, 等于强制它做progress verification, 减少幻觉
2. **Stable interface**: 如果只输出"next subtask", 失败时指令会抖动(这次说grasp cup, 下次又grasp cup, 没结构); 输出list保持指令流稳定
3. **Compact memory**: $C_t^\star$ 作为language token传给下一帧, 替代visual history, 既compact又不怕drift

---

## 关键设计3: Visual Trace(画条路线给VLA看)

manager不光输出language subtask, 还输出一条2D trajectory:

$$
\tau_t^\star = \{p_t, p_{t+1}, \ldots, p_{t_K^e}\}
$$

人话: 从当前时刻到任务结束, end-effector在image plane上要走的pixel坐标序列。$p_t \in \mathbb{R}^2$ 是某时刻gripper在画面上的2D位置, $t_K^e$ 是最后一个subtask结束的frame。

**这条trace直接render到observation image上**, 作为visual overlay给VLA看。VLA学会"follow这条线"。

**为什么trace是个好interface?**

- 对VLM来说, 它是output modality——VLM擅长把language ground到image region, 画条线很容易
- 对VLA来说, 它是input conditioning——VLA学习follow pixel-space trajectory, 比理解language直接
- 比language更spatial(精确到pixel), 比raw action更abstract(不用predict motor command)
- 直接render到画面上, 无需extra encoding

这就像你给工人一张地图上画了个箭头说"去这儿", 比口头说"往左走三米再往右"清楚得多。

trace的灵感直接来自TraceVLA, 区别是TraceVLA的trace是single-step的, LoHo-Manip是remaining trace + 闭环re-planning。

参考: TraceVLA https://trace-vla.github.io/

---

## 关键设计4: Current Frame Only(只看当前帧)

这是paper里最争议的设计, 但我觉得是合理的trade-off。

**做法**: manager训练和推理时都只condition on当前一帧observation, 历史progress通过textual summary $C_{t-1}$ 传入。

**为什么不feed visual history?**

paper给的reason很实在: 真机执行会imperfect, 一旦偏离expert demo, 历史frames就进入training distribution外, 反而hurt manager的决策质量。Visual history在长任务下也贵(latency高)。

**代价**: 丢失fine-grained temporal info, 比如object刚刚被推到的精确位置。完全依赖VLM每帧re-perceive。

paper在Limitations里诚实承认这点, 说future work可以incorporate richer temporal guidance。我觉得这是对的选择, 因为VLM的perception能力足够强, 每帧re-perceive比维护drift的history更robust。

concurrent work MEM走的是另一条路, 在VLA内部维护multi-scale memory。两种approach的长期优劣还要看后续work。

参考: MEM https://www.pi.website/download/Mem.pdf

---

## 数据怎么搞的: 全自动pipeline

这是工程上很值得学的一块。paper不靠人工标注, 全用VLM自动从demo视频里提取supervision。

### Subtask decomposition

用VLM对视频做temporal segmentation, prompt大意是:
- 把机器人行为拆成atomic subtask(grasp, place, insert, push, cut)
- 排除motion-only(move, reach)和preparatory(prepare to...)
- 每个subtask对应一个object state change
- 输出JSON: `{"10": "Grasp the knife", "25": "Place on board"}`

### Trace extraction

对每一帧用VLM detect end-effector bounding box:
- 严格定义target为"wrist + grippers", 排除上游arm link
- 取box center作为 $p_t$
- Normalize到[0, 1000]
- Resample为compact waypoints

### Failure recovery合成数据

这个特别聪明。从Bridge数据里filter grasp-and-place episodes, 找到grasp和place的transition frame, 然后把grasped object替换成scene里其他graspable items。形成"假失败"数据: 机器人抓错东西了。Manager需要predict "Drop the wrong object"作为recovery subtask。

这种augmentation让模型见过error → recovery pattern, 不用在真机上collect failure demo(成本极高)。

参考: BridgeData https://rail-berkeley.github.io/bridge_data/

---

## 实验结果: work不work

### Long-horizon reasoning

RoboVQA上BLEU-4 = 53.5, 击败Qwen3-VL-8B(51.4)和ThinkAct-7B(52.4)。4B打8B。

EgoPlan-Bench2上Avg = 56.7, ThinkAct-7B是48.2, gap很大。说明progress-aware training让模型在human-level planning上generalize更好。

### Trajectory prediction

ShareRobot-T上DFD = 0.2309, Qwen3-VL-4B是0.3808。三个距离metric(DFD, HD, RMSE)全部显著低, 说明trace prediction精度高, 直接影响下游VLA执行。

### LIBERO

Avg = 97.5, Long track = 95.2。Long是long-horizon专测, 比StarVLA的93.8还高, 说明hierarchy design在long任务上有正贡献。

### VLABench(真正的硬骨头)

Semantic Instruction track: π0.5只有0.17, LoHo-Manip到0.42, **2.5x提升**。这条track是"prepare vegetable skewers"这种隐式instruction, 完全靠VLM reasoning decouple后才解开。这是monolithic VLA根本做不了的任务。

### 真机OOD

Real robot上, 单步OOD和multi-stepOOD都显著超过π0.5 baseline(同样fine-tune在100个demo上)。Decoupling让VLM的zero-shot grounding能力直接transfer给VLA, 这是monolithic model做不到的。

### Modular ablation

把executor从π0.5换成StarVLA, 加上LoHo-Manip manager还能涨33%。证明manager是executor-agnostic的, 可以plug-and-play不同VLA backbone。

参考:
- LIBERO https://libero-project.github.io/
- VLABench https://github.com/VLABench/VLABench
- EmbodiedBench https://embodiedbench.csail.mit.edu/

---

## 为什么这个设计work: 三条intuition

### 1. Division of labor

VLA的policy $\pi(a | o, x)$ 要同时encode三层info:
- Semantic grounding("cup" → image region)
- Spatial planning(先去哪后去哪)
- Motor control(precise joint trajectory)

monolithic VLA把三层压在一个network里, training signal互相interfere。LoHo-Manip把前两层offload到VLM, VLA只剩motor control + trace-following。各司其职, 各自scale。

### 2. Trace是正确的abstraction boundary

Visual trace对VLM是output modality(画线容易), 对VLA是input signal(follow线容易)。它比language更spatial, 比3D waypoint更visual, 比action sequence更abstract。这个abstraction层选得刚刚好。

### 3. Receding horizon把long变short

10-step task, open-loop累积error让成功率指数衰减。Receding horizon每步re-plan, 只要有一步意识到"前一步没成功"就能补救, 实质把10-step task变成10个1-step task的序列。这跟DAgger的intuition一样: closed-loop比open-loop sample-efficient得多。

参考: DAgger https://www.cs.cmu.edu/~bmdf/

---

## 几点实话实说的局限

1. **2D trace表达不了contact-rich任务**: 拧螺丝、组装这种需要force/pose/contact mode的, 2D keypoint太impoverished。扩展到3D + contact mode是natural next step

2. **Semantic perception failure recover不了**: 如果VLM把corn看成sushi, 它会输出错误recovery"drop the sushi"。Receding horizon能recover execution failure, recover不了semantic perception failure。这是VLM grounding本身的能力上限

3. **Manager频率是固定的100步一次**: 如果subtask特别短(grasp只要50步), 可能错过及时replan。Paper没给频率sensitivity analysis

4. **真测的long-horizon深度有限**: VLABench的subtask大概3-7步, 真机也是类似scale。20+ steps的household任务没测过

5. **与end-to-end scaling的tension**: 未来π1可能直接scale到long-horizon, modular approach的相对优势可能缩小。但modular在data efficiency和compositional generalization上应该一直有优势

---

## 跟你teaching的关联

Andrej, 如果你要做nanoVLA教学, LoHo-Manip是个perfect case study:

- Hierarchy: manager + executor, 经典two-level
- Receding horizon: MPC思想在symbolic层面的应用
- Visual prompt: trace作为intermediate representation
- Modular design: 各component可以分别debug、分别upgrade
- 全用off-the-shelf model: Qwen3-VL + π0.5, 学生可以reproduce

比直接教π0.5这种monolithic model更pedagogical, 因为每个component的role都清楚, 学生能build intuition about why hierarchy matters。

参考: Software 2.0 https://karpathy.medium.com/software-2-0-a64152b37c35

---

## 一句话总结

**把VLM和VLA拆开, VLM每一步重新想"还剩啥没干"并画条路线图, VLA只管follow路线图做眼前这一步。Recovery、replanning、progress tracking全是receding horizon的自然涌现, 不用写任何规则。**

就这么简单。Paper的beauty在于: idea clean, execution solid, 实验覆盖面广, 4B打8B, modular还work。如果你要写"hierarchy is all you need for embodied AI"的essay, 这篇是strong evidence。

Project page: https://www.liuisabella.com/LoHoManip

想drill into哪个细节(trace的render具体怎么做的、failure case具体长啥样、跟π0.5 internal architecture怎么对比), 随时说。

---

# LoHo-Manip: Trace-Conditioned VLA Planning for Long-Horizon Manipulation

Andrej, 这篇paper我读下来感觉非常对你teaching时反复强调的"hierarchy + locality + closed loop"哲学。下面我把技术细节、设计动机、与经典方法的联系都拆开讲一遍, 帮你build intuition。

---

## 1. 问题动机:为什么long-horizon manipulation是VLA的真正瓶颈

当前主流VLA(π0, π0.5, OpenVLA, GR00T N1, RT-2)在short-horizon atomic skill上表现很强, 但long-horizon任务的成功率会随horizon呈指数级衰减。原因paper里点出三个核心:

1. **Compounding execution error**: 模仿学习经典问题, 参考Ross et al. DAgger [1] 的理论分析, error随步数O(T)或更差累积
2. **Distribution shift under imperfect rollout**: 训练时见的是expert trajectory的smooth分布, 部署时一旦偏离, 后续observation就进入training manifold外
3. **Modularity缺失**: 把planner和executor融合在一个monolithic model里, 升级embodiment / action space / training domain时要重训整个stack

paper的论点很clean: **high-level reasoning应该用general-purpose VLM, low-level control应该用specialized VLA, 二者通过visual trace这种structured intermediate representation解耦**。

参考链接:
- LoHo-Manip project page: https://www.liuisabella.com/LoHoManip
- DAgger原文(Ross, Gordon, Bagnell): https://www.cs.cmu.edu/~bmdf/
- π0.5: https://arxiv.org/abs/2504.16054

---

## 2. 系统架构总览

整个系统是一个**two-level hierarchical controller**, 类似经典Options framework [2]或FeUdal Networks [3]的现代VLA版本:

```
        ┌─────────────────────────────────────┐
        │  Task Manager (VLM, ~4B params)     │
        │  Input:  (x, o_t, C_{t-1})          │
        │  Output: (C_t, R_t, τ_t)            │
        │  Freq:   ~2 Hz (receding horizon)   │
        └──────────────┬──────────────────────┘
                       │  subtask text + visual trace τ_t
                       ▼
        ┌─────────────────────────────────────┐
        │  Executor (VLA, π0.5 backbone)      │
        │  Input:  (o_t, rendered trace, s_t) │
        │  Output: robot action a_t           │
        │  Freq:   ~10 Hz                     │
        └──────────────┬──────────────────────┘
                       │
                       ▼ action chunk → environment
                  new observation o_{t+1}
                       │
                       ▼ feedback to manager (every 100 steps)
```

关键design choice: **manager只看current frame**, 不依赖visual history buffer。这个决定非常关键, 下面会详细分析为什么。

---

## 3. Progress-aware Plan Representation: 公式深度解析

### 3.1 式(1): completed prefix + remaining suffix

$$
C_t^\star = [\bar{s}^{(1)}, \ldots, \bar{s}^{(k(t)-1)}], \quad R_t^\star = [\bar{s}^{(k(t))}, \ldots, \bar{s}^{(K)}]
$$

变量含义:
- $\bar{s}^{(k)} \in \mathcal{S}$: 第 $k$ 个atomic interaction primitive(如"grasp the cup", "pour into kettle")
- $k(t) \in \{1, \ldots, K\}$: 当前时间 $t$ 所处或下一个待执行的primitive的索引
- $K$: 整个episode的primitive总数
- $C_t^\star \in \mathcal{S}^{k(t)-1}$: 已完成subtask序列(语言memory)
- $R_t^\star \in \mathcal{S}^{K-k(t)+1}$: 剩余subtask序列

**为什么同时输出completed和remaining, 而不只输出remaining?**

直觉上有三点:
1. **Self-consistency check**: 模型自己predict已完成序列, 等于强制它显式进行progress verification, 减少hallucination
2. **Stable interface under failure**: 如果只输出"next subtask", 当某步失败时next指令会反复抖动(这次说grasp cup, 下次又grasp cup, 无结构); 输出remaining list则保持指令流的稳定结构
3. **Compact textual memory替代visual history**: 把 $C_t^\star$ 作为language token序列传给下一帧的manager, 携带了discrete progress信息, 而visual history会drift

### 3.2 式(2): visual trace label

$$
\tau_t^\star = \{p_t, p_{t+1}, \ldots, p_{t_K^e}\}
$$

变量含义:
- $p_t \in \mathbb{R}^2$: 时间 $t$ 时robot end-effector在image plane上的2D pixel coordinate
- $t_K^e$: 最后一个primitive $\bar{s}^{(K)}$ 的end frame index
- $\tau_t^\star$: 从当前时刻到episode结束的所有未来end-effector位置(在2D pixel space)

注意几个细节:
- 这是**2D pixel space**, 不是3D workspace, 也不是joint space。这种choice让trace可以直接render到observation image上, 作为visual prompt给VLA
- 实际存储时是**resampled waypoints**, 不是逐帧位置(否则序列太长)
- 真机上用VLM逐帧detect end-effector bounding box取中心, 再normalize到[0, 1000]范围

---

## 4. Receding-Horizon闭环: 最精彩的设计

这部分是paper的核心contribution, 我觉得非常elegantly工程化。

### 4.1 与classical MPC的对应

"Receding horizon"这个词直接来自Model Predictive Control [4]。MPC的核心思想是:

> 每个时刻求解一个有限horizon最优控制问题, 只执行第一个控制input, 然后下一时刻重新求解。

LoHo-Manip把这套思想搬到symbolic + spatial层面:

| MPC | LoHo-Manip |
|-----|-----------|
| 系统动态模型 $f(x, u)$ | VLM implicit的task model |
| 优化horizon N | 剩余subtask数 $K - k(t) + 1$ |
| Cost function | 任务完成度(implicit) |
| Control input $u_t$ | VLA action $a_t$ |
| First input applied | 第一个subtask执行 |
| Re-solve next step | Re-invoke manager |

### 4.2 Implicit closed loop的机制

假设step k失败(比如grasp cup没抓起来)。下一帧manager重新观测:

- World state没变(cup还在桌上, 没被抓起来)
- Manager看到current observation, 输出的remaining plan里依然包含"grasp cup"
- Trace会重新指向cup的位置
- Executor再次尝试grasp

**没有hand-crafted failure detector, 没有recovery logic, 没有state machine**。所有recovery都是receding horizon + observation-conditioned planning的emergent property。

这种设计哲学和SayCan [5]、Inner Monologue [6]那种explicit language feedback loop不一样, 那些需要显式verbalize失败。LoHo-Manip让recovery完全implicit在re-prediction里。

### 4.3 与MEM [7]的对比

paper里提到concurrent work MEM(Multi-scale Embodied Memory), 它用short-term visual context + long-term language memory。区别在于MEM还在VLA内部维护memory, 而LoHo-Manip把memory外化到task manager的textual progress summary里, manager自身每帧re-predict。后者更modular但少了fine-grained temporal context。

参考:
- SayCan: https://say-can.github.io/
- Inner Monologue: https://arxiv.org/abs/2207.05608
- MEM: https://www.pi.website/download/Mem.pdf

---

## 5. Data Pipeline: 自动化subtask + trace生成

这是工程上很值得学习的一块。paper用vision-language foundation models自动从demo视频里提取supervision, 流程是:

### 5.1 Subtask decomposition

用VLM(Sec D, Table 9的prompt)对视频做temporal segmentation:
- 识别atomic interaction events(grasp, place, insert, push, cut)
- **排除motion-only actions**(move, reach, approach)和preparatory phrases(prepare to...)
- 每个subtask对应一个object state change
- 输出JSON: `{"subtasks": {"10": "Grasp the knife", "25": "Place the knife on cutting board"}}`

### 5.2 Trace extraction

对每一帧用VLM做end-effector detection:
- 严格定义target为"wrist + grippers/fingers", **排除Link 7及上游arm**
- 输出tight bounding box `[xmin, ymin, xmax, ymax]`
- 取box center作为 $p_t$
- Normalize到[0, 1000]
- Resample为compact waypoints

### 5.3 Failure recovery数据合成

这是非常聪明的augmentation:
1. 从Bridge数据里filter grasp-and-place episodes
2. 找到grasp发生和place发生的transition frames
3. 把原视频里grasped object替换成scene里其他detected graspable items
4. 形成"假失败"数据: 机器人grasp了错误物体
5. Manager需要predict "Drop the wrong object"作为recovery subtask

这种synthetic failure data让模型见过error → recovery pattern, 而不需要在真机上collect failure demonstrations(成本极高)。

参考:
- BridgeData: https://rail-berkeley.github.io/bridge_data/
- Open X-Embodiment: https://robotics-transformer-x.github.io/

---

## 6. Training Paradigm细节

### 6.1 Task Manager训练

- Init: Qwen3-VL-4B [8]
- Frozen: vision encoder
- Fine-tuned: language model
- Supervision:
  - (a) progress-aware plan text(同时predict $C_t$ 和 $R_t$)
  - (b) 2D trace的waypoint sequence
- Conditioning: current frame only + textual progress summary $C_{t-1}$
- Training data混合:
  - Bridge subset(Open X-Embodiment format)
  - RoboVQA(long-horizon reasoning)
  - EgoPlan-BenchIT(human-level planning)
  - Synthetic failure-recovery samples

### 6.2 Executor adaptation

- Init: π0.5 base checkpoint [9]
- Fine-tune: condition on rendered trace prompt(可选: current subtask text)
- 目标: 让VLA学会"follow the trace"这个generic skill

### 6.3 Inference loop

```
At time t (every 100 executor steps):
  1. Manager observes (o_t, C_{t-1})
  2. Predicts (C_t, R_t, τ_t)
  3. Render τ_t onto o_t as visual overlay
  4. Executor generates action chunk {a_t, ..., a_{t+h}}
  5. Execute actions
  6. Update C_t as new memory (no history frames fed back)
  7. Go to step 1
```

executor频率~10Hz, manager频率~2Hz, overhead约14s/episode(86s vs 72s)。

参考:
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- π0: https://www.physicalintelligence.company/blog/pi0

---

## 7. 实验数据深度分析

### 7.1 Table 1: RoboVQA + EgoPlan-Bench2

| Method | RoboVQA B-4 | RoboVQA Avg | EgoPlan2 Avg |
|--------|-------------|-------------|--------------|
| Gemini-3.0-Flash | 32.2 | 37.3 | 48.8 |
| Qwen3-VL-8B | 51.4 | 60.8 | 36.6 |
| ThinkAct-7B | 52.4 | 59.8 | 48.2 |
| RynnBrain-8B | 52.6 | 62.1 | 34.8 |
| **LoHo-Manip-4B** | **53.5** | **63.1** | **56.7** |

注意LoHo-Manip只有4B参数, 击败8B的Qwen3-VL和ThinkAct-7B。EgoPlan2上优势尤其大(56.7 vs 48.2), 说明progress-aware training让模型在human-level planning任务上generalize得更好。

### 7.2 Table 2: Trajectory Prediction

三个metrics都是距离度量, 越低越好:
- **DFD (Discrete Fréchet Distance)**: 衡量两条曲线的"形状相似度", 类比人走两条路径的最坏配对距离
- **HD (Hausdorff Distance)**: 一条曲线上每个点到另一条曲线最近距离的最大值
- **RMSE**: 点对点误差

LoHo-Manip在ShareRobot-T上DFD=0.2309, 显著低于Qwen3-VL-4B的0.3808和Embodied-R1-3B的0.3426。说明trace prediction精度很高, 这直接影响下游VLA执行质量。

### 7.3 Table 5: LIBERO

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| π0-fast | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| StarVLA | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 |
| **LoHo-Manip** | **98.0** | **98.6** | **98.0** | **95.2** | **97.5** |

Long track上从StarVLA的93.8提到95.2, gap虽小但Long是long-horizon专测, 说明hierarchy design在long任务上确实有正贡献。

### 7.4 Table 4: VLABench

VLABench是long-horizon reasoning + generalization的硬骨头:

| Method | In Dist | Cross Cat | Common Sense | Semantic Instr | Unseen Tex | Avg |
|--------|---------|-----------|--------------|----------------|------------|-----|
| π0-fast | 0.29 | 0.18 | 0.21 | 0.20 | 0.24 | 0.22 |
| π0.5 | 0.37 | 0.22 | 0.21 | 0.17 | 0.25 | 0.24 |
| **LoHo-Manip** | **0.54** | 0.23 | **0.36** | **0.42** | **0.39** | **0.39** |

注意Semantic Instruction track: π0.5只有0.17, LoHo-Manip到0.42, **2.5x提升**。这条track是"prepare vegetable skewers"这种隐式instruction, 完全靠VLM的reasoning能力decouple后才解开。

### 7.5 Table 7: Ablation - Executor无关性

| Config | VLABench Avg |
|--------|--------------|
| StarVLA alone | 0.18 |
| StarVLA + LoHo-Manip manager | 0.24 |

把π0.5换成StarVLA, 加上manager还能涨33%。证明modular design的executor-agnostic性, manager可以plug-and-play不同VLA backbone。

参考:
- LIBERO: https://libero-project.github.io/
- VLABench: https://github.com/VLABench/VLABench
- StarVLA: https://github.com/STAR-VLA/STAR-VLA
- RoboVQA: https://robovqa.github.io/
- EmbodiedBench: https://embodiedbench.csail.mit.edu/

---

## 8. 核心Intuition:为什么这个设计work

### 8.1 Division of labor的information-theoretic视角

VLA的policy $\pi(a | o, x)$ 需要同时encode三层信息:
- **Semantic grounding**: 把"x里的cup"映射到image中的pixel region
- **Spatial planning**: 决定先去哪、后去哪
- **Motor control**: 生成precise joint trajectory

monolithic VLA把这三层压在一个network里, training signal互相interfere。LoHo-Manip把前两层offload到VLM, VLA只剩motor control + trace-following。

### 8.2 Trace作为abstraction boundary

Visual trace $\tau_t$ 是一个非常精妙的interface:
- 对VLM而言, 它是output modality(VLM擅长grounding language到image region)
- 对VLA而言, 它是input conditioning signal(VLA可以学习follow pixel-space trajectory)
- 它**比language更spatial**, 比3D waypoint更visual(可以直接render)
- 它**比raw action sequence更abstract**, 避免VLM需要predict precise motor commands

这种abstraction让我想到你Software 2.0 essay里关于"适当的inductive bias"的讨论——trace是人为设计的intermediate representation, 把VLM和VLA各自的strength leveraged起来。

### 8.3 Receding horizon > open-loop planning

想象一个10-step task, open-loop plan在第1步生成后, 每步error rate假设5%, 累积到第10步成功率 $0.95^{10} \approx 0.6$。Receding horizon每步re-plan, 只要有一步意识到"前一步没成功"就能补救, 实质上把10-step task变成10个1-step task的序列。

这和DAgger的intuition一样: closed-loop control比open-loop imitation sample-efficient得多。

### 8.4 Current-frame-only conditioning的trade-off

paper明确说不用visual history, 只用textual progress summary $C_{t-1}$。这个决定有得有失:

**得**:
- 避免imperfect rollout下的visual distribution shift
- Inference快
- Memory compact

**失**:
- 丢失fine-grained temporal information(比如object刚刚被推到的精确位置)
- 完全依赖VLM每帧重新perceive所有信息

paper在Limitations里承认这点。我认为这是个合理的trade-off, 因为VLM的perception能力足够强, 每帧re-perceive比维护drift的history更robust。

参考:
- Software 2.0 (Karpathy): https://karpathy.medium.com/software-2-0-a64152b37c35
- Options framework: https://www.cs.ubc.ca/labs/lci/mlss06/papers/sutton-precup-singh-1999.pdf

---

## 9. 与相关工作的脉络梳理

### 9.1 Hierarchical RL的经典传承

- **Options framework (Sutton, Precup, Singh, 1999)** [10]: option = (initiation set, policy, termination), LoHo-Manip的subtask就是option
- **FeUdal Networks (Vezhnevets et al., 2017)** [11]: manager output subgoal, worker follow, LoHo-Manip的trace就是subgoal
- **HIRO (Nachum et al., 2018)** [12]: off-policy hierarchical RL with goal-conditioned worker

### 9.2 LLM-as-planner的evolution

- **SayCan (Ahn et al., 2022)** [5]: LLM选skill, affordance model过滤
- **PaLM-E (Driess et al., 2023)** [13]: embodied multimodal LLM
- **Code as Policies (Liang et al., 2023)** [14]: LLM生成可执行code
- **VoxPoser (Huang et al., 2023)** [15]: LLM生成3D value map
- **SayPlan (Rana et al., 2023)** [16]: 3D scene-graph grounded planning
- **ThinkAct (Huang et al., 2025)** [17]: reinforced visual latent planning

LoHo-Manip的position: 用visual trace替代code/value map/latent plan作为interface, 同时把planner彻底decouple成独立VLM。

### 9.3 Trace-conditioned VLA

直接predecessor是**TraceVLA (Zheng et al., 2025)** [18], 它把trajectory作为visual prompt给VLA。LoHo-Manip继承这个idea, 但把trace的generation交给独立的task manager, 并加入receding-horizon re-planning。TraceVLA是single-step trace, LoHo-Manip是remaining trace + 闭环。

### 9.4 VLA memory方向

- **RoboVQA (Sermanet et al., 2024)** [19]: multimodal long-horizon reasoning
- **MEM (Torne et al., 2026)** [7]: multi-scale embodied memory for VLA
- **Long-VLA (Fan et al., 2025)** [20]: 直接scale VLA到long-horizon

LoHo-Manip选择外化memory到textual progress summary, 而不是放在VLA内。这是modularity vs end-to-end的trade-off。

参考:
- PaLM-E: https://palm-e.github.io/
- Code as Policies: https://code-as-policies.github.io/
- VoxPoser: https://voxposer.github.io/
- TraceVLA: https://trace-vla.github.io/
- OpenVLA: https://openvla.github.io/
- GR00T N1: https://developer.nvidia.com/groot

---

## 10. 几点critical thinking

### 10.1 Visual trace的expressiveness限制

paper Limitations里诚实承认: **2D trajectory无法表达contact-rich interaction**(比如拧螺丝、组装)。这种任务需要force / pose / contact mode信息, 2D keypoint太impoverished。扩展到3D trajectory + contact mode可能是一个natural next step。

### 10.2 Task manager的semantic correctness bottleneck

如果VLM mis-perceive object(比如把corn看成sushi), 它会输出错误的recovery subtask"drop the sushi"(paper Fig. 4)。Receding horizon能recover from execution failure, 但recover不了semantic perception failure。这是VLM grounding本身的能力上限。

### 10.3 Manager频率的trade-off

每100 executor steps调一次manager。如果subtask非常短(grasp只要50 steps), manager可能错过及时replan。Paper没给出频率sensitivity analysis, 我觉得这是future work需要explore的axis。

### 10.4 评测的long-horizon深度

VLABench的subtask数大概3-7步, 真机实验也是类似scale。真正的"household-level long-horizon"(20+ steps)还没测过。Paper作者自己承认future work要测dynamic environment。

### 10.5 与end-to-end VLA scaling的tension

Physical Intelligence的π0.5和未来π1可能直接scale到long-horizon(更多data, 更大model)。LoHo-Manip的modular approach在scaling law下是否还有优势是个open question。我的intuition是: modular在data efficiency和compositional generalization上一直会赢, 但absolute ceiling可能被sufficiently-scaled monolithic model超越。这点在language modeling领域已经发生(RNN+attention vs pure transformer)。

---

## 11. 给你的几个可能的延伸思考

### 11.1 Trace representation的learned alternative

当前trace是手工设计的2D keypoint。能不能让manager和executor**jointly learn**一个latent trace representation? 类似VQ-VAE [21]的codebook, 让trace成为discrete latent code, 既保持modularity又allow richer semantics。

### 11.2 把receding horizon推广到probabilistic planning

现在manager每帧输出deterministic remaining plan。如果改成输出**distribution over remaining plans**(类似diffusion policy [22]在action层面做的事), 可以表达task ambiguity。比如"kettle may need refill OR may need clean", manager输出两个candidate plan, executor根据current state去disambiguate。

### 11.3 Hierarchy的recursion

当前是2-level(manager + executor)。能不能做n-level? 比如meta-manager分配phase, phase-manager分配subtask, subtask-manager分配trace, executor执行。这对应经典MAXQ [23]或HIRO的multi-level hierarchy。VLM的能力可能allow更深的stack。

### 11.4 与你的nano系列教学的关系

如果你做nanoVLA教学, LoHo-Manip是个非常好的case study: 它把hierarchy、receding horizon、visual prompt三个概念都用到了, 而且每个component都是off-the-shelf(VLM + VLA), 学生可以分别debug和理解。比直接教π0.5这种monolithic model更pedagogical。

参考:
- VQ-VAE: https://arxiv.org/abs/1711.00937
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- MAXQ: https://www.cs.umass.edu/publication/docs/1995/UM-CS-1995-038.pdf

---

## 12. 总结

LoHo-Manip给我的核心take-away:

1. **Decoupling planner from executor**让两个module各自scale, 各自upgrade, 这在product层面是巨大优势
2. **Visual trace**作为interface非常巧妙, 既spatial又visual, 既abstract又actionable
3. **Receding-horizon re-prediction**让recovery / replanning / progress tracking全部emergent, 不需要hand-crafted logic
4. **Current-frame + textual memory**是个pragmatic的trade-off, 避开visual history drift问题
5. **Synthetic failure data**是个低成本高效果的augmentation, 值得借鉴

paper的实验覆盖面很广(reasoning QA + planning benchmark + sim manipulation + real robot + OOD), 4B manager击败8B baselines, modular design让manager能plug到StarVLA, 这些都让结论可信度比较高。

如果你后续要写一篇关于"hierarchy is all you need for embodied AI"的essay, 这篇paper提供了非常strong的实证支持。

---

**主要参考链接汇总:**

- Project: https://www.liuisabella.com/LoHoManip
- π0.5: https://arxiv.org/abs/2504.16054
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- RoboVQA: https://robovqa.github.io/
- LIBERO: https://libero-project.github.io/
- VLABench: https://github.com/VLABench/VLABench
- EmbodiedBench: https://embodiedbench.csail.mit.edu/
- TraceVLA: https://trace-vla.github.io/
- SayCan: https://say-can.github.io/
- PaLM-E: https://palm-e.github.io/
- VoxPoser: https://voxposer.github.io/
- Code as Policies: https://code-as-policies.github.io/
- OpenVLA: https://openvla.github.io/
- GR00T N1: https://developer.nvidia.com/groot
- BridgeData: https://rail-berkeley.github.io/bridge_data/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- StarVLA: https://github.com/STAR-VLA/STAR-VLA
- NVILA: https://github.com/NVlabs/NVILA
- DAgger (Ross et al.): https://www.cs.cmu.edu/~bmdf/
- Options framework: https://www.cs.ubc.ca/labs/lci/mlss06/papers/sutton-precup-singh-1999.pdf
- Software 2.0 (Karpathy): https://karpathy.medium.com/software-2-0-a64152b37c35

如果你想drill into某个具体aspect(比如visual trace的render细节、failure recovery的具体case、或与π0.5 internal architecture的对比), 告诉我, 我可以继续展开。
