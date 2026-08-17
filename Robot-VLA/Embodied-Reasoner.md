---
source_pdf: Embodied-Reasoner.pdf
paper_sha256: d62967fd240ac0789706246db47a063941e71e49fad17f60e977300e66b8facd
processed_at: '2026-08-04T03:41:02-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Embodied-Reasoner

## 一句话版本

这帮人想搞个事儿：o1 在数学题上 think 很厉害，那能不能让机器人在房间里找东西的时候也 think 一番再行动？结果发现还真可以，而且比 o1 自己干这事儿还强。

## Problem 到底是什么

想象你让一个机器人在陌生厨房里找鸡蛋。你作为人怎么干这事儿？你会 think："鸡蛋一般在冰箱里，先去冰箱看看"。打开冰箱没有，你会 reflect："奇怪，那可能在柜子里"。然后你记得自己已经看过冰箱了，不会重复去开冰箱。

但是 o3-mini 干这事儿就很有意思了——它会反复去开同一个冰箱，或者导航到 sofa 之后忘了自己要干嘛了。Figure 10 里那个 GPT-o1 的 case 特别搞笑，step 13 到 16 全是 `move forward`，step 18 到 21 又全是 `move forward`，卡在那里疯狂往前蹭。

核心 problem 有两个：

**第一，extended multimodal interaction**。o1 之前的 reasoning benchmark 基本都是 single-turn：给个图给个问题，think 完输出答案。但 embodied 是 multi-turn 的，每一步都有新 image 进来，context 越来越长，模型还得保持 coherent。

**第二，diverse reasoning modalities**。数学题主要是 logical deduction，但 embodied 场景需要 commonsense（鸡蛋放冰箱）、spatial reasoning（厨房布局）、temporal reasoning（我刚才看过哪儿）、self-reflection（找错了反思一下）。完全不同的 reasoning 类型。

## 他们怎么干的

核心 idea 其实挺直觉的：**既然人找东西的时候脑子里会想各种东西，那我就用数据把这个 thinking 过程 explicit 地造出来，然后让模型学着这么想。**

### Data Engine 是关键

这里有个很聪明的 trick。你不能直接让 GPT-4o 去生成 embodied trajectory，因为它不知道 AI2-THOR 里某个 scene 到底有没有 sofa、sofa 能不能 move。所以他们搞了个 Affiliation Graph：

```
Kitchen → Fridge → Apple
                 (include)
```

从 metadata 构建这个 graph，然后要生成 "找 Apple" 这个 task 的时候，graph 告诉你 Apple 在 Fridge 里。Key action sequence 就是：navigate to Fridge → open Fridge → pickup Apple。这个 sequence 是确定性的，保证合成数据正确。

然后他们再插入 exploratory actions：先去 sidetable 没找到、去 desk 没找到、最后去 fridge 找到。这样 trajectory 就 realistic 了，模拟了真实的探索过程。

**这个 graph-based approach 解决了一个 fundamental problem：reward signal 从哪来。** 你怎么知道模型生成的一条 trajectory 是 "好" 的？有了 graph 推导的 key action sequence，就能拿这个当 ground truth，当 PRM 用。这一点很关键，比直接用 LLM judge 靠谱多了。

### Thought 怎么造

他们定义了 5 种 thinking pattern：
- **Situation Analysis**: "我看到这是个厨房，有 fridge、countertop"
- **Task Planning**: "我先看 fridge，因为鸡蛋大概率在那"
- **Spatial Reasoning**: "fridge 在房间另一边，需要 navigate 过去"
- **Self-Reflection**: "fridge 里没有，我判断错了，得换个地方"
- **Double Verification**: "我已经拿到鸡蛋了，确认一下 task 完成"

用 GPT-4o 在每个 observation 和 action 之间插入这些 thought。Figure 4 里那个 transition graph 挺有意思：Plan→Plan 55%，Plan→Spatial 45%，失败后 Action→Reflection 33%。这说明 thought pattern 之间不是随机的，有结构。

### 三阶段训练

这个 pipeline 的 intuition 很清楚：

**Stage 1 Imitation**: 先让 Qwen2-VL-7B 学会基本交互。1,128 条短轨迹，纯模仿。学完之后模型能动了，但是很傻——打开冰箱发现没鸡蛋就说 "egg does not exist"。因为它只见过 "直接找到" 的 trajectory，没见过 "找错了继续找" 的。

**Stage 2 Rejection Sampling**: 这是关键的一步。用 Stage 1 模型在高温下采样大量 trajectory，然后用 PRM（就是前面说的 data engine）筛选出成功的 6,246 条。这些 trajectory 里包含了模型自己摸索出来的探索路径。用这些做 SFT，模型就学会了 "找不到就换地方" 的能力。Success rate 从 25% 跳到 65%。

这步其实就是在做一种弱化的 RL——不用 PPO，不用 reward shaping，就是 rejection sampling + SFT。跟 DeepSeek-R1 的思路一脉相承，但更简单粗暴。

**Stage 3 Reflection Tuning**: 模型现在会找了，但 long-horizon 任务里会 hallucinate，而且真实机器人有 hardware fault。所以他们造了两类 correction data：
1. 成功 trajectory 中间插入异常状态（navigate 到错误地点），然后让模型 reflect 并 retry
2. 失败 trajectory 定位第一个错误 action，插入 reflection，后面接正确 trajectory

Loss 只在 reflection 和 correct part 上算，error part mask 掉。这样模型学会了 "我刚才搞错了，现在纠正" 的能力。

## 为什么 work

我 think 这篇 paper work 的核心原因有三个：

### 1. Thought 作为 Episodic Memory

VLM 在长 context 里会 forget。o3-mini 在 composite task 里 RER (Repetitive Exploration Rate) 高达 54%，就是因为它忘了自己看过哪儿。

Embodied-Reasoner 的 thought 里会显式 recall："我刚才看过 fridge 和 countertop，都没找到，接下来去看 cabinet"。这个 thought 本身就是 episodic memory 的载体。模型不用去 100 步之前的 image 里找信息，thought 已经把关键信息 summarize 了。

这其实暗示了一个更深层的问题：**当前 VLM 的 long-context attention 机制对 embodied 场景不够好。** Image tokens 太多了，几十步交互下来 context 里几千个 image tokens，attention 很难 focus。Thought 把关键信息压缩成几百个 text tokens，相当于做了 memory consolidation。

### 2. Test-Time Scaling 在 Embodied 里也成立

Figure 5 下半张图特别说明问题。baseline 模型的 output tokens 基本不随 task complexity 变化，卡在 1000 左右，success rate 暴跌。Embodied-Reasoner 的 output tokens 从 1000 涨到 3500，success rate 保持 60%+。

这说明 deep thinking 的 inference-time compute scaling 在 embodied 领域同样适用。但前提是模型得先学会 "什么时候该多想"——这恰恰是 Stage 2 rejection sampling 学到的：复杂任务需要更多探索和 planning，这些 trajectory 里的 thought 自然更长。

### 3. Symbolic-Neural Hybrid

Affiliation Graph 是 symbolic 的，GPT-4o 合成的 thought 是 neural 的，Qwen2-VL 学的也是 neural 的。这个 hybrid 架构很聪明：

纯 neural approach（让 LLM 自己规划）在 embodied 场景 success rate 很低，因为没有结构化先验。纯 symbolic approach（hardcode 规则）不 flexible，没法泛化到新场景的自然语言指令。

Data Engine 用 symbolic graph 保证了数据质量和 reward signal，然后把这些知识 "蒸馏" 成 neural 的 thought，最终内化到 VLM 权重里。模型 inference 时不需要 graph，自己就能 think 出类似的 planning。

## 我的几个联想

**关于 hierarchical structure**。这篇 paper 的 9 个 high-level action 其实已经是 abstracted 过的了。真实机器人控制需要 low-level motor command。我觉得最终形态应该是：Embodied-Reasoner 做 high-level planning 和 reasoning，生成 sub-goal，然后一个 VLA model（像 OpenVLA 或 RT-2）把 sub-goal 转成 low-level action。这种 hierarchical 设计能解决 latency 问题——high-level think 慢一点没关系，low-level control 得快。

**关于 sim-to-real**。Real-world 实验里人手持摄像头当 robot，这其实是 cheat 了。真实机器人的 locomotion 有 noise，机械臂有 compliance，物体可能被遮挡。56.7% 的 success rate 说明 generalization 有一定基础，但 gap 肯定比 paper 展示的大。AI2-THOR 的 visual 太 clean 了，domain gap 明显。如果他们在 Habitat 或 robosuite 上测过会更有说服力。

**关于 thought 的 representation**。现在 thought 是 natural language，冗长且慢。有没有可能把 thought 压缩成 latent representation？类似 PaLI 或 Flamingo 的 perceiver resampler，但用于 thought。这样 inference 速度能快很多，同时保留 reasoning 能力。或者搞个 dual-system：System 1 快速直觉反应，System 2 深度思考，类似 AlphaGo 的 policy network + value network + MCTS。

**关于 exploration vs exploitation**。Paper 里提到 Embodied-Reasoner 在简单 search task 上会 over-explore 导致漏检近处物体。这其实是个 exploration-exploitation tradeoff 问题。模型没有 calibrated 的 uncertainty estimation——它不知道 "这个任务很简单，不需要多想"。未来可能需要引入 confidence calibration，或者让模型自己判断 task complexity 来动态调整 reasoning depth。

**关于 data efficiency**。9,390 条 trajectory 训练出 80%+ success rate，这个 data efficiency 其实挺不错的。但其中 6,246 条是 Stage 2 rejection sampling 来的，需要先有 Stage 1 model 在 AI2-THOR 里 rollout，这个 sampling cost 不低。如果能在 real-world 直接用这套 pipeline 会怎样？Affiliation Graph 没法构建，因为真实世界没有 metadata。可能需要用 VLM 来 online 构建 scene graph，然后实时规划。这就回到 scene understanding 的经典问题了。

## 一些具体技术细节

**Loss function 细节**：多轮 dialogue 格式下，observation 和 simulator feedback 是 User Input，thought 和 action 是 Assistant Output。Loss 只在 Assistant tokens 上算。这个设计很标准，但有个 subtlety：image tokens 在 User Input 里，模型需要 cross-attend 到 image。Qwen2-VL 的 architecture 支持这个，但 image resolution 和 context length 需要仔细 control。64K image × 如果每个 image 几百 tokens，context 很快就爆了。

**Constraint check 公式细节**：
$$ \text{Valid}(A, B) = \text{Pickupable}(A) \land \neg\text{Openable}(\text{Parent}(A)) \land \text{Openable}(B) $$

这个 formula 是 "Exposed-to-Enclosed Object Transfer" 的约束。$A$ 是被 pickup 的物体，要求可抓取且其父容器不可打开（说明 $A$ 暴露在外）。$B$ 是目标位置，要求可打开（是封闭容器）。每种 task type 对应一个不同的 constraint formula，都在 Appendix D 的 table 里。

**RER 计算细节**：
$$ RER = \frac{N_{revisit}}{N_{total}} \times 100\% $$

举例：轨迹是 Place_a → Place_b → Place_b → Place_c → Place_c。Place_b 和 Place_c 各被 revisit 一次，总共 5 步里 2 步是重复，RER = 40%。这个 metric 简单但有效，直接量化了 "模型有没有在原地打转"。

## References

- Paper project page: https://embodied-reasoner.github.io/
- AI2-THOR: https://ai2thor.allenai.org/
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- DeepSeek-R1 (rejection sampling inspiration): https://arxiv.org/abs/2501.12948
- OpenVLA (hierarchical VLA 联想): https://openvla.github.io/
- RT-2 (VLA baseline 对比): https://robotics-transformer2.github.io/
- ReAct (thought-action 交替的 prompt 版本): https://arxiv.org/abs/2210.03629
- OpenAI o1: https://openai.com/o1/
- Self-Consistency / Self-Refine 相关思路: https://arxiv.org/abs/2203.07805

---

这篇 paper 提出了 **Embodied-Reasoner**，核心目标是把 o1-style reasoning 范式扩展到 embodied interactive tasks 中。以往的 deep thinking models 擅长 math 和 coding，虽然这些任务需要长链条逻辑演绎，但是 embodied scenarios 需要的是 spatial understanding, temporal reasoning 以及基于 interaction history 的 ongoing self-reflection。作者通过构建一个 data engine 合成 Observation-Thought-Action 交错轨迹，并设计了三阶段训练管线，成功让 Qwen2-VL-7B 在 AI2-THOR 环境中的 success rate 达到了 80.96%，超越了 OpenAI o1, o3-mini 和 Claude-3.7。

### 1. Core Architecture & Task Formulation

**Task Environment**: 基于 AI2-THOR simulator，包含 107 个 indoor scenes 和 2100 个 objects。任务分为 Search, Manipulate, Transport, Composite 四类。
**Action Space**: 封装了 9 个 high-level actions：`Observe`, `Move Forward`, `Navigate to {}`, `Put in {}`, `Pickup {}`, `Toggle {}`, `Close {}`, `Open {}`, `Termination`。
**Trajectory Format**: 模型需要处理 image-action interleaved context，轨迹定义为 $o_1, a_1, o_2, a_2, \dots, o_n, a_n$。$o_i$ 表示第 $i$ 步的第一人称视角图像，$a_i$ 表示执行的动作。在插入 thinking thoughts 后，格式变为 $(o_n, t_n^1, t_n^2, \dots t_n^k, a_n)$，其中 $t_n^k$ 表示在第 $n$ 步产生的第 $k$ 种思考模式。

### 2. Data Engine: Observation-Thought-Action Corpora

为了训练出具备自发推理能力的模型，作者构建了一个全自动 data engine。这是本文最核心的创新点。

**2.1 Instruction Synthesis**
因为 LLM 直接生成的指令可能包含场景中不存在的物体或非法动作，所以作者设计了 Constraint Check。
约束逻辑公式化表示为：
$$ \text{Valid}(A, B) = \text{Pickupable}(A) \land \neg \text{Openable}(\text{Parent}(A)) \land \text{Openable}(B) $$
其中，$A$ 是目标物体，$B$ 是容器，$\text{Parent}(A)$ 表示 $A$ 的父节点。只有满足此约束，指令 "pickup A put in B" 才合法。利用 GPT-4o 生成 code 基于 scene metadata 进行过滤，确保了数据的有效性。

**2.2 Affiliation Graph & Key Action Sequence**
为了自动推导出完成任务的最小动作序列，作者从 simulator metadata 构建了 Affiliation Graph。
图节点表示物体，边表示从属关系。比如 keychain 在 drawer 里，drawer 在 mudroom 里，表示为 $\text{Leaf}(\text{keychain}) \to \text{Node}(\text{drawer}) \to \text{Node}(\text{mudroom})$。
通过从 Leaf 节点向上追溯，生成 Key Action Sequence: $A_1$: Nav to Mudroom $\to$ $A_2$: Nav to Drawer $\to$ $A_3$: Open Drawer $\to$ $A_4$: Pickup。

**2.3 Interleaving Thought with Observation-Action**
定义了 5 种 thinking patterns 模拟人类认知：
1. **Situation Analysis**: 分析环境状态
2. **Task Planning**: 制定搜索计划
3. **Spatial Reasoning**: 空间布局推理
4. **Self-Reflection**: 失败后反思
5. **Double Verification**: 完成后验证

GPT-4o 根据 historical trajectory $(o_1, t_1, a_1, \dots, o_n)$ 和 upcoming action $(a_n)$，生成 thought $(t_n)$。要求 $t_n$ 不仅要提供 action 的 rationale，还要与 historical thoughts $t_{1:n-1}$ 保持逻辑一致。

### 3. Three-Stage Training Pipeline

为了 build up intuition，可以把这个管线看作是让模型从 "模仿操作" 到 "自主探索" 再到 "自我纠错" 的渐进过程。

**Stage 1: Imitation Learning (Learn to Interact)**
在 1,128 条短轨迹上微调 Qwen2-VL-7B-Instruct。Loss 计算仅针对 thought 和 action tokens：
$$ L_{SFT} = - \sum_{t} \log P(x_t | x_{<t}) $$
其中 $x_t$ 是 thought 或 action token，$x_{<t}$ 是 context (包含之前的 images, thoughts, actions)。
得到 **Embodied-Interactor**。虽然学会了交互，但是缺乏探索能力，遇到空冰箱会直接回答 "egg does not exist" 而不去别处找。

**Stage 2: Rejection Sampling Tuning (Learn to Search)**
借鉴 DeepSeek-R1 的思想。用 Stage 1 模型在高温下对新指令采样大量轨迹。
Data Engine 充当 Process Supervision Reward Model (PRM)。PRM 的评估逻辑可以形式化为：
$$ PRM(Traj) = \mathbb{1} \left[ \bigwedge_{i=1}^{n} (a_i \in \text{KeyActionSeq}) \lor (a_i \in \text{ValidExploration}) \right] $$
保留 6,246 条成功轨迹做 SFT。得到 **Embodied-Explorer**。Success rate 从 25.4% 跃升至 65.4%。

**Stage 3: Reflection Tuning (Learn to Self-reflect)**
为了解决 long-horizon hallucination 和 hardware anomaly。构造两类自修正数据：
1. **成功轨迹插入异常**：$\{ \dots, a, o_-, t_r, a, o_+ \dots \}$。$a$ 是动作，$o_-$ 是异常状态（如导航到错误地点），$t_r$ 是 reflective thought，重试 $a$ 得到正常状态 $o_+$。
2. **失败轨迹修正**：$\{ Traj_-^{1:t}, t_r^t, Traj_+^{t:n} \}$。$Traj_-^{1:t}$ 是错误前缀，$t_r^t$ 是反思，$Traj_+^{t:n}$ 是修正后的正确轨迹。
Loss 只在 $t_r$ 和 $Traj_+^{t:n}$ 上计算，mask 掉 $Traj_-^{1:t}$。得到最终模型 **Embodied-Reasoner**，Success rate 达到 80.96%。

### 4. Experiments & Data Analysis

**Dataset Statistics (Table 1)**
*   Train 1st: 1,128 traj, 4,636 img, avg 4.11 actions
*   Train 2nd: 6,246 traj, 45.8k img, avg 7.33 actions (轨迹变长，因为包含探索)
*   Train 3rd: 2,016 traj, 13.8k img, avg 8.63 actions
*   Test: 809 cases, 4.9k img, avg 6.06 actions
*   总计 8M thought tokens。

**Performance Comparison (Table 2)**
*   Qwen2-VL-7B 基线只有 14.79%。
*   GPT-4o 达到 66.67%。
*   OpenAI o1 达到 71.73%。
*   Embodied-Reasoner 达到 80.96%。
在 Composite tasks (最难) 上，GPT-4o 只有 14.42%，o1 只有 13.16%，而 Embodied-Reasoner 达到 54.29%。在简单 Search 任务上，o3-mini 反超 Embodied-Reasoner，因为 Embodied-Reasoner 偶尔会 over-explore 导致漏检近处物体。

**Thoughts Transition Analysis (Figure 4)**
思考模式之间的转移概率显示了动态认知图：
*   Task Planning $\to$ Task Planning (55%) 或 Spatial Reasoning (45%)。
*   Action $\to$ Spatial Reasoning (42%) 或 Self-Reflection (33%)。
这说明模型在未知区域倾向空间推理，搜索失败后倾向自我反思。

### 5. Intuition Building: 为什么 Deep Thinking 在 Embodied 中有效？

**5.1 Test-Time Scaling in Embodied AI**
Figure 5 展示了 task length 与 output tokens 的关系。随着任务复杂度增加，baseline (如 Gemini-2.0-flash-thinking) 的 output tokens 保持在 1000 左右，且 success rate 暴跌。但是 Embodied-Reasoner 的 output tokens 会随着复杂度从 1000 增长到 3500，并保持高 success rate。这证明了在 embodied 场景下，inference time scaling 同样适用。这种能力来源于 Stage 2 的 Rejection Sampling，挖掘出了长尾的合理探索轨迹。

**5.2 Temporal Reasoning 克服 Repetitive Exploration**
定义了 Repetitive Exploration Rate (RER)：
$$ RER = \frac{N_{revisit}}{N_{total}} \times 100\% $$
其中 $N_{revisit}$ 是重复访问已探索地点的次数，$N_{total}$ 是总探索动作数。
o3-mini 在 Composite tasks 中 RER 高达 54%，因为它在长上下文中 "忘记" 了探索历史。Embodied-Reasoner 的 RER 降到 26%。因为 Self-Reflection 显式地 recall 了 past observations，thought 在这里充当了 episodic memory 的载体，缓解了 VLM 在长上下文中的遗忘问题。

**5.3 符号-神经融合**
Affiliation Graph 实际上充当了 Symbolic World Model。Data Engine 在后台用图算法推导 Key Action Sequence，再用 LLM 合成自然语言 thought，最后用这些 thought 训练 VLM。这是一种巧妙的 symbolic-neural hybrid 方法，纯靠 LLM 自发产生合理动作序列概率极低，通过 graph 注入结构化先验，使得 PRM 能够提供可靠的 reward signal。

### 6. Associations & Hallucinations (深度联想)

*   **ReAct vs. Embodied-Reasoner**: ReAct 是在 prompt 中交替生成 Thought, Action, Observation。本文相当于把 ReAct loop 内化到了模型的权重中。模型无需显式 prompt 指导，自发产生 ReAct 结构，这降低了推理时的 prompt engineering 成本，但也降低了 zero-shot 泛化到其他 prompt 格式的能力。
*   **VLA Models (RT-2, OpenVLA)**: 它们通常直接预测 action chunk，没有显式的高层 reasoning tokens。对于 low-level 控制（如机械臂位姿），VLA 更合适；对于 high-level planning（如找东西），Embodied-Reasoner 这种显式 thought 范式更具可解释性和可调试性。未来可能的发展方向是 hierarchical architecture: high-level 用 Embodied-Reasoner 生成 thoughts 和 goals，low-level 用 VLA 执行 motor control。
*   **Sim-to-Real Gap**: 论文在 real-world 实验中用人类手持摄像头作为代理。真实世界的光照、视角畸变、动态障碍物远比 AI2-THOR 复杂。虽然达到了 56.7% 的 success rate，但这距离 robust deployment 还有距离。未来需要引入 domain randomization 或 large-scale real-world video pre-training。
*   **Latency Problem**: 生成 3500 个 thought tokens 会带来显著延迟。在真实机器人交互中，机器人不能停下来思考 10 秒。解决方案可能是 speculative thinking：在一个 thread 里慢思考，在另一个 thread 里执行 default fallback actions；或者将 thought distillation 为更紧凑的 latent representations。

### References & Web Links

*   **AI2-THOR Simulator**: https://ai2thor.allenai.org/
*   **DeepSeek-R1 (Rejection Sampling)**: https://arxiv.org/abs/2501.12948
*   **Qwen2-VL**: https://arxiv.org/abs/2409.12191
*   **OpenAI o1**: https://openai.com/o1/
*   **ReAct Prompting**: https://arxiv.org/abs/2210.03629
*   **RT-2 (Vision-Language-Action Models)**: https://robotics-transformer2.github.io/
*   **OpenVLA**: https://openvla.github.io/

总而言之，Embodied-Reasoner 证明了 o1-style reasoning 在具身智能中的巨大潜力。其核心贡献在于 Data Engine 的设计，巧妙利用 Affiliation Graph 和 PRM 解决了合成数据的逻辑一致性和 reward signal 问题。三阶段训练使得模型从模仿走向自主探索，最终具备了长周期规划与自我纠错能力。
