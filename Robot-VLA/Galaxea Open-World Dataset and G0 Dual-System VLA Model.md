---
source_pdf: Galaxea Open-World Dataset and G0 Dual-System VLA Model.pdf
paper_sha256: 151aa3493bcf6c52a62a30e0f84a1ba686dd22a84fc7de602a92abd5e49639c1
processed_at: '2026-08-04T11:49:33-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Galaxea G0

## 一、这 paper 在讲什么故事

最近 robotics 社区有个 default 假设：**多机器人、多场景、大数据 = generalist robot policy**。Open-X-Embodiment [1] 推动了这个叙事，大家觉得只要 data 够多够杂，VLA model 就能 generalize 到任何 robot 上。

Galaxea team 说：**等一下，我们做了个实验发现这假设有问题**。

他们搞了个 500 小时的 dataset，全部用一个 robot（Galaxea R1 Lite，bimanual mobile manipulator）在真实人类环境（厨房、办公室、餐厅、住宅）里采集。然后他们训了个 dual-system VLA，跑了严格 controlled experiment。

**核心发现**：当你用一堆乱七八糟的 robot 数据（OXE）pre-train VLA，然后 fine-tune 到一个 morphologically 差别很大的 target robot 上，效果可能**比 from scratch 还差**。尤其在 embodiment-specific 的 skill 上（比如 mobile base 移动、torso 俯仰协调），cross-embodiment pre-training 会产生 **negative transfer**。

这跟 Stanford 最近的 "Is Diversity All You Need" [27] 形成呼应——diversity 有 diminishing returns，甚至会反噬。

Paper link: https://opengalaxea.github.io/G0/

---

## 二、为什么这个发现反直觉但 make sense

### 2.1 LLM scaling law 为什么 work

LLM 之所以 scale law 成立，是因为所有 data 共享同一个 **token space**。GPT 训英文、中文、code、数学，都是 token sequence，transfer 是天然的。

### 2.2 Robotics 的 action space 是 embodiment-specific 的

Robotics 不一样。一个 Franka Panda 的 7-DoF arm action 跟一个 Galaxea R1 Lite 的 23-DoF bimanual mobile manipulator action，**dimensionality 和 semantics 完全不同**。你以为它们都是 "robot action"，但从 model 的角度，它们是不同的 distribution。

所以 cross-embodiment pre-training 学到的 "action prior"，在 fine-tune 到一个 morphologically distant 的 robot 上时，可能跟 target embodiment 的 action manifold 不兼容，需要 **unlearn**。

### 2.3 这个问题之前为什么没被充分讨论

因为大部分 VLA paper 的 benchmark 不够严格。Open-X-Embodiment [1] 的 evaluation 主要是 simulation 或 lab scene，embodiment gap 不明显。π0 [12] 的 benchmark 虽然在 real world，但没有系统性地 disentangle cross-embodiment 和 single-embodiment pre-training 的贡献。

Galaxea 的 contribution 是**设计了一个能 isolate 这个变量的实验**——同一个 target robot，同一个 downstream task，唯一变量是 pre-training data 的来源。

π0 paper: https://arxiv.org/abs/2410.24164

---

## 三、Dataset 为什么 quality 高

### 3.1 Hardware 选择

Galaxea R1 Lite：23-DoF bimanual + mobile base + pitch torso。这个配置很 important——它有 **whole-body coordination** 的能力（chassis + torso + arms 一起动），这是大部分 OXE robot 没有的。

### 3.2 Isomorphic teleoperation 的关键

他们不用 VR teleop，用 **isomorphic teleoperation**——operator 的 arm motion 直接 mapping 到 robot kinematics。这意味着 operator 永远在 kinematically feasible posture 里操作，不会出现 IK failure 或 retargeting artifact。

这是为什么 dataset 的 action quality 高。VR teleop（比如 Apple Vision Pro）听起来 fancy，但 human arm 跟 robot arm morphology 不一样，retargeting 会引入 noise。Isomorphic teleop 虽然不够 intuitive，但 data 更干净。

参考 DROID [22] 也讨论过 teleop interface 对 data quality 的影响：https://droid-dataset.github.io/

### 3.3 Subtask-level annotation 的价值

每个 episode 切成 atomic subtasks，annotation 走 fixed schema（从标准化描述池里选，不是 free-form text）。

这很关键，因为 Open-X-Embodiment 的 annotation 是出了名的 noisy——不同 source 的 granularity 差异巨大，有的是 task-level（"make a sandwich"），有的是 step-level（"pick up the bread"）。Galaxea 的 schema-based annotation 让 language 和 action 精确对齐，这是 Stage-2 pre-training 能 work 的前提。

---

## 四、Dual System 架构 — 在干什么

### 4.1 灵感来源

Kahneman《Thinking, Fast and Slow》[4] 的 System 1 / System 2 框架。最近 robotics 社区流行这个 pattern（SayCan [3]、Hi Robot [19]、OpenHelix [18]）。

- **System 2 (G0-VLM)**: 慢思考，planner。人话说"我想吃饭" → VLM 拆成 "走到冰箱前 → 打开冰箱 → 拿食物 → 关冰箱 → 放进微波炉 → ..."
- **System 1 (G0-VLA)**: 快执行，reactive policy。接收 subtask instruction + 视觉 → 输出 action chunk

### 4.2 为什么要 dual system

单个 VLA model 同时做 planning 和 execution 有问题：
- Planning 需要长 context reasoning（几秒到几十秒）
- Execution 需要高频 control（10-50 Hz）
- 如果每帧都跑 VLM reasoning，latency 太高，real-time deployment 不可能

Dual system 的 trick 是 **异步频率**：System 2 只在 subtask 切换时 trigger，System 1 高频执行。这跟 RT-2 [25] 的 single-stream 设计形成对比。

RT-2: https://arxiv.org/abs/2307.15818

### 4.3 G0-VLA 内部结构

```
[3 cameras] ──► SigLIP encoder ──► MLP projector ──┐
                                                      ├─► PaLiGemma Transformer
[language] ──────────────────────────────────────────┤      │
[proprioception] ────────────────────────────────────┘      │ KV cache
                                                              ▼
                                                    [Action Expert]
                                                              │
                                                    flow matching loss
                                                              │
                                                              ▼
                                                    action chunk (H steps)
```

VLM (PaLiGemma [29]) 负责 semantic grounding，Action Expert（新初始化的 Transformer）负责 continuous control，condition 在 VLM 的 KV cache 上。这个设计跟 π0 [12] 类似。

PaLiGemma: https://arxiv.org/abs/2407.07726

---

## 五、3-Stage Training Curriculum — 最有教学价值

### Stage-1: Cross-embodiment pre-training（只训 VLM）

**数据**: 1000h OXE + 500h Galaxea（只用 high-level task description）+ 200h in-house

**关键设计**: Stage-1 **只训 VLM，不训 action expert**。

为什么？两个理由：
1. Cross-embodiment data 的 action quality 不一致，action expert 学不到 informative 东西
2. Flow matching loss 对 representation quality 敏感，VLM 还没 stable 时 attach flow head 会破坏 VLM 的 semantic representation

**Loss** (autoregressive next-token prediction):

$$p(\mathbf{A}_t^d) = \prod_{i=1}^{N} p(a_i^d \mid a_{<i}^d, o_t, l_t, s_t)$$

变量解释：
- $\mathbf{A}_t^d$: 时间 $t$ 的离散化 action token 序列，上标 $d$ 表示 discrete
- $N$: token 序列长度
- $a_i^d$: 第 $i$ 个 action token
- $a_{<i}^d$: 第 $i$ 个之前的所有 tokens（causal mask）
- $o_t$: 三路视觉 observation（head cam + 2 wrist cams）
- $l_t$: language instruction
- $s_t$: proprioceptive state（joint positions 等）

用 **FAST tokenizer** [11] 把 continuous action chunk 转成 discrete tokens。FAST 基于 DCT (Discrete Cosine Transform) 压缩 action chunk 的时间冗余。

FAST: https://arxiv.org/abs/2501.09747

### Stage-2: Single-embodiment pre-training（VLM + Action Expert）

**数据**: Galaxea Open-World Dataset（带 subtask annotation）

这时候 action expert 才被 attach 进来。VLM 的 weights 从 Stage-1 继承。

**Loss** (flow matching):

Maximum likelihood objective:

$$\max_\theta \mathbb{E}_{p(A_t, o_t, l_t, s_t)} \left[\log \pi_\theta(A_t \mid o_t, l_t, s_t)\right]$$

Flow matching loss:

$$\mathcal{L}_{\text{flow}}(\theta) = \mathbb{E}_{p(A_t^\tau \mid o_t, l_t, s_t)} \left[\left\|\nu_\theta(A_t^\tau, \tau, o_t, l_t, s_t) - u(A_t^\tau \mid A_t)\right\|^2\right]$$

变量详解：
- $A_t$: ground-truth action chunk，horizon $H$
- $A_t^\tau = \tau A_t + (1-\tau)\varepsilon$: noisy interpolated action
  - $\tau \in [0,1]$: flow time parameter，$\tau=0$ 纯噪声，$\tau=1$ ground truth
  - $\varepsilon \sim \mathcal{N}(0, I)$: Gaussian noise
- $\nu_\theta(\cdot)$: VLA 预测的 **velocity field**，告诉你当前 $A_t^\tau$ 应该往哪个方向走
- $u(\cdot)$: target flow，从 action trajectory 推导出来的 ground-truth velocity
- $\theta$: 整个 VLA 的参数

**Intuition**: Flow matching [Lipman et al. 2023] 是 diffusion 的 cousin。它学一个 vector field 把简单分布（Gaussian）transport 到复杂分布（action distribution）。跟 DDPM 比，flow matching 的 ODE trajectory 更直、sampling 更快。

Flow Matching 原论文: https://arxiv.org/abs/2208.14518

**Stage-2 的两个关键 enabler**:
1. **Single embodiment**: action space 一致，action expert 不需要跨 embodiment adapt
2. **Language-action alignment**: subtask-level segmentation 让 instruction 和 trajectory 精确对齐

### Stage-3: Post-training（task-specific fine-tuning）

每个 downstream task 最多用 100 trajectories，4 epochs。Loss 跟 Stage-2 一样。这是用来测试 pre-trained weights 的 generalization 的——如果 pre-training 有效，post-training 应该 sample efficient。

---

## 六、实验结果 — 最有说服力的部分

### 6.1 Pre-trained weights comparison (Figure 9)

配置：
- G0 (Stage-1): 仅 Stage-1
- G0 (Stage-2 200h): 仅 Stage-2，200h 数据
- G0 (Stage-2 400h): 仅 Stage-2，400h 数据
- G0 (Full): Stage-1 → Stage-2 (400h)
- G0 (Scratch): 无 action pre-training
- π0: Physical Intelligence 官方 pre-trained weights

**Key findings**:

1. **G0 (Full) 在 average progress score 上最高**，尤其在 object-picking 类 task

2. **G0 (Stage-2) 在 language following + action consistency + whole-body control 上最强**

3. **G0 (Stage-1) 是所有 pre-trained models 里最差的**——cross-embodiment pre-training 单独用，效果比 from scratch 还差

4. **π0 baseline 也不如 G0 (Stage-2)**——π0 在大量 cross-embodiment data 上训的，morphology 跟 Galaxea R1 Lite 差距大

### 6.2 Few-shot transfer (Figure 10)

用 20 trajectories fine-tune，10 epochs。

- **Stage-2 pre-training 显著提升 few-shot 性能**——success rate 和 action smoothness 都好
- **Stage-1 pre-training 几乎没有 advantage over scratch**——cross-embodiment action pre-training 对 few-shot adaptation 帮助很小

Intuition：few-shot 场景下，model 需要快速 adapt 到新 embodiment 的 dynamics。Stage-2 已经让 model 学会了 target embodiment 的 action manifold，few-shot 只是 refinement。Stage-1 学的是 "general robot action priors"，但这些 priors 跟 target embodiment 的 action space 不兼容，成了 noise。

### 6.3 Embodiment-specific actions (Figure 11) — 最关键

Bed Making 任务按 skill 拆解：
- Moving toward bed (chassis)
- Lifting torso + grasping quilt (torso + arms)
- Leaning torso back (torso)
- Flattening quilt (chassis + arms)

**Stage-2 pre-training 在 chassis 和 torso control 上大幅领先**

**Stage-1 pre-training 和 π0 在这些 embodiment-specific skills 上比 from scratch 还差**——negative transfer！

Paper 的 hypothesis：OXE 里的 embodiment 跟 Galaxea R1 Lite（omnidirectional base + pitch torso）morphologically 差距太大，cross-embodiment pre-training 学到的 action priors 是 misleading 的。

---

## 七、G0-VLM Training — 一个被低估的 trick

### 7.1 Base model

Qwen2.5-VL [30]，instruction tuning。

Qwen2.5-VL: https://arxiv.org/abs/2502.13923

### 7.2 数据构造 pipeline

这是 paper 里最 clever 的工程：

1. **Sample episodes** from Galaxea dataset，key frames（subtask 即将结束、gripper state change）sampling weight 更高
2. **Extract** head camera images + subtask annotations
3. **Feed k-frame history**: 当前帧 + 前 k 秒的 frames 和 actions（1-second interval），让 VLM 能 handle long temporal context
4. **LLM reasoning augmentation**: 用 **DeepSeek-R1** 在 $D_{\text{labeled}}$ 上生成 human-style high-level instruction 和 robot verbal response

第 4 步特别有意思。他们**不给 reasoning LLM 看图像**，只给它 task name + historic subtasks + next subtask，让 LLM 推理出 human-style verbal instruction。

例子：
- Input: task="pull and push chairs", historic subtasks=[...], next subtask="pull chair out"
- DeepSeek-R1 推理: "I am going to be seated, could you help pull the chair out?"
- Robot response: "I am working on it!"

这是一个 **LLM-as-data-augmenter** 的 pattern——用 reasoning LLM 把 mechanical subtask annotation 扩写成 natural human-robot dialogue。比雇佣 human writer 写 natural language instruction 便宜得多。

DeepSeek-R1: https://arxiv.org/abs/2501.12948

### 7.3 Evaluation 结果

Table 1 很 striking：

| Model | Table bussing | Microwave | Make bed | Build blocks |
|---|---|---|---|---|
| Gemini-2.5-pro | 32.0 | 15.8 | 54.2 | 55.0 |
| Qwen2.5-VL-72B | 26.3 | 16.8 | 48.1 | 21.7 |
| Qwen2.5-VL-32B | 21.3 | 14.8 | 54.2 | 21.0 |
| Qwen2.5-VL-7B | 26.3 | 17.2 | 46.9 | 24.7 |
| **G0-VLM** | **83.3** | **74.2** | **78.2** | **75.6** |

G0-VLM 在 4 个 task 上平均超 baseline 50%+。这说明 general-purpose VLM（哪怕 72B）在 robotic action grounding 上根本不够用——必须 domain-specific fine-tune。

---

## 八、用人话总结 core insights

### 8.1 Robotics 不是 NLP

LLM 的 scaling law 成立是因为 token space 统一。Robotics 的 action space 是 embodiment-specific 的，**cross-embodiment data 的 transferable knowledge 主要在 vision-language 层面，不在 action 层面**。

所以 **VLM pre-train on cross-embodiment 是 OK 的，action expert pre-train on cross-embodiment 是 dangerous 的**。这正是 Galaxea Stage-1 只训 VLM 的原因。

### 8.2 Quality > Quantity 在 robotics 里更成立

Galaxea 的 500h single-embodiment high-fidelity data > OXE 的 1000h heterogeneous data（对于 target embodiment 的 fine-tuning 而言）。

但这不意味着 quantity 不重要。Stage-2 从 200h 到 400h 有明显提升。只是说，**当 embodiment gap 大时，quality 的边际收益远大于 cross-embodiment quantity**。

### 8.3 Flow matching loss 对 representation quality 敏感

Stage-1 只训 VLM 不训 action expert 的设计选择，背后的 insight 是：**flow matching 的 gradient 会 backprop 回 VLM，如果 VLM 还没 stable，会破坏 semantic representation**。

这是 VLA 训练的一个 subtle 的 engineering detail。OpenVLA [10] 的全 autoregressive 设计没这个问题，但 throughput 低。π0 [12] 的 joint training 设计可能面临这个问题，但 Physical Intelligence 没明确讨论。

### 8.4 Dual system 是 pragmatic engineering，不是 scientific breakthrough

System 2 的 plan 能力 + System 1 的 reactive execution，本质上是在 approximate 一个 ideal VLA 的 hierarchical computation。异步频率带来的 efficiency gain 让 real-world deployment 成为可能。

但从 research 角度，理想终点应该是 **single VLA 能同时做 planning 和 execution**，dual system 是过渡方案。参考 Hi Robot [19] 和 OpenHelix [18] 的讨论。

### 8.5 Reasoning LLM as data augmenter 是被低估的 pattern

如果你有 high-quality atomic annotations，可以用 reasoning LLM 扩写成各种 surface forms（human dialogue, plan explanation, etc.），比直接训 VLM 生成这些更 sample efficient。

这个 pattern 在 NLP 里已经 common（GPT-4 生成 SFT data），但在 robotics 里还比较新。Galaxea 把它用在了 VLM planner 的 instruction tuning 上。

---

## 九、Open questions 和 limitations

1. **Single embodiment = limited generalization to other robots**。Galaxea dataset 只有一个 embodiment，如果你想在 Franka 或 UR5 上 deploy，Stage-2 pre-training 完全用不上。

2. **Cross-embodiment pre-training 的 negative transfer 是否可逆？** Paper 没探索：如果 Stage-1 之后加一个 "embodiment adaptation" 阶段，能否 recover？

3. **500h 数据够吗？** 相比 AgiBot World [24] 的 100K+ hours，Galaxea 的 500h 算 small。Scaling laws 在 single-embodiment 设定下还成立吗？

4. **G0-VLM 的 plan 错误如何 propagate 到 VLA？** Dual system 的 classic problem：如果 VLM 输出了错误的 subtask instruction，VLA 会忠实执行错误。Paper 没讨论 error recovery。

5. **Flow matching vs autoregressive 的 trade-off**。Paper 用了两种 generation paradigm（Stage-1 autoregressive, Stage-2 flow matching），但没 ablation 比较 pure autoregressive vs pure flow matching。参考 HybridVLA [17] 的 hybrid 设计。

AgiBot World: https://arxiv.org/abs/2503.06669
HybridVLA: https://arxiv.org/abs/2503.10631

---

## 十、给 Andrej 的 personal intuition

这篇 paper 对你（Karpathy）应该有几个特别 resonate 的点：

### 10.1 "Software 2.0" 在 robotics 里的 boundary

你之前提过 Software 2.0——神经网络替代 handwritten rules。在 NLP/CV 里这个 transition 很 clean，因为 data space 统一。但在 robotics 里，**action space 是 embodiment-specific 的，所以 Software 2.0 的 scaling law 有 boundary**。

Galaxea 的实验显示，cross-embodiment action pre-training 的 scaling 会在 embodiment gap 大时 break down。这跟你早年在 Tesla 自动驾驶里遇到的问题有点像——不同 sensor suite 和 vehicle dynamics 的 data 不能直接混。

### 10.2 "Data quality > data quantity" 的 robotics 版本

你经常强调 data quality。Galaxea 给了一个 robotics 版本的 evidence：**single-embodiment high-fidelity data > cross-embodiment noisy data**。这跟你在 Tesla 强调 "data engine" 的 quality control 思路一致。

### 10.3 VLM 作为 "world model" 的 role

Stage-1 只训 VLM 的设计，暗示了一个 deeper 的 claim：**VLM 在 VLA 里扮演的是 world model 的 role，提供 semantic grounding；action expert 只是 motor control**。如果这个 claim 成立，那么 VLA 的 scaling 主要应该 scale VLM（用更多 vision-language data），而 action expert 可以保持 small（只需要 single-embodiment data）。

这跟 LeCun 的 JEPA 思路有呼应——世界模型和 action policy 可以 decouple。

### 10.4 Dual system 跟人类认知的类比

Kahneman 的 System 1/System 2 框架在 AI 里被用得很多，但大部分是 metaphor。Galaxea 的 dual system 是为数不多的 **operationalization**——System 2 低频 reasoning，System 1 高频 execution，异步运行。

这跟你讨论过的 "System 1/2 in LLM" 的 distinction 有 connection。LLM 的 chain-of-thought 是 System 2 的 internal monologue，但 robotics 里 System 2 需要 output actionable plan 给 System 1。

---

## Reference links

- [Galaxea G0 Project]: https://opengalaxea.github.io/G0/
- [PaLiGemma]: https://arxiv.org/abs/2407.07726
- [π0]: https://arxiv.org/abs/2410.24164
- [π0.5]: https://arxiv.org/abs/2504.16054
- [FAST tokenizer]: https://arxiv.org/abs/2501.09747
- [Open-X-Embodiment]: https://robotics-transformer-x.github.io/
- [Qwen2.5-VL]: https://arxiv.org/abs/2502.13923
- [DeepSeek-R1]: https://arxiv.org/abs/2501.12948
- [Flow Matching]: https://arxiv.org/abs/2208.14518
- [SayCan]: https://arxiv.org/abs/2204.01691
- [Hi Robot]: https://arxiv.org/abs/2502.19417
- [OpenHelix]: https://arxiv.org/abs/2505.03912
- [OpenVLA]: https://arxiv.org/abs/2406.09246
- [RT-2]: https://arxiv.org/abs/2307.15818
- [BridgeData V2]: https://arxiv.org/abs/2308.12952
- [DROID]: https://droid-dataset.github.io/
- [AgiBot World]: https://arxiv.org/abs/2503.06669
- [RoboMIND]: https://arxiv.org/abs/2412.13877
- [HybridVLA]: https://arxiv.org/abs/2503.10631
- [Is Diversity All You Need]: https://arxiv.org/abs/2507.06219
- [RDT-1B]: https://arxiv.org/abs/2410.07864
- [CogACT]: https://arxiv.org/abs/2411.19650
- [Kahneman, Thinking Fast and Slow]: https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow
- [UniPi]: https://uni-pi.github.io/
- [RT-X]: https://robotics-transformer-x.github.io/

---

**一句话 takeaway**：在 VLA 训练里，**VLM 的 semantic grounding 可以 cross-embodiment transfer，但 action policy 的 motor control 不能**。Galaxea 的 3-stage curriculum 就是这个 insight 的 operationalization——Stage-1 只训 VLM（cross-embodiment OK），Stage-2 训 action expert（必须 single-embodiment），Stage-3 fine-tune（task-specific）。Cross-embodiment pre-training 的 default 假设需要社区重新审视。

Figure 11 是最有说服力的 evidence——Stage-2 在 chassis control 上几乎是 Scratch 的 2-3 倍，而 Stage-1 和 π0 反而低于 Scratch。这个 result 应该会改变 community 对 cross-embodiment pre-training 的默认假设。

---

# Galaxea Open-World Dataset & G0 Dual-System VLA — 深度讲解

## 一、Paper 的核心 thesis

这篇 paper 的核心 thesis 可以浓缩成一句话：**在 VLA 训练里，cross-embodiment pre-training 的价值被社区高估了，single-embodiment high-fidelity data 才是 VLA generalization 的真正 bottleneck**。这个发现非常反直觉，因为过去两年社区的主流叙事是 "data scale + embodiment diversity = generalist robot policy"（Open-X-Embodiment 推动的方向 [1]）。Galaxea team 通过一个 500 小时 single-embodiment dataset + 严格 controlled benchmark，给出了相反的 evidence：当你把 cross-embodiment 数据（OXE）pre-train 完的 weights 拿到一个 morphologically distant 的 target robot 上 fine-tune，效果可能比 from scratch 还差，尤其在 embodiment-specific skills（chassis、torso 协调）上。

这跟 Stanford 最近的 "Is Diversity All You Need" [27] 的结论形成呼应——diversity 在 scaling 时是有 diminishing returns 的，甚至会出现 negative transfer。

Project page: https://opengalaxea.github.io/G0/

---

## 二、Dataset — Galaxea Open-World Dataset

### 2.1 规模与构成

| 维度 | 数值 |
|---|---|
| Total hours | ~500h |
| Trajectories | 100K |
| Task categories | 150 |
| Scenes | 50 (across 11 physical sites) |
| Objects | 1,600+ |
| Skills | 58 |
| Embodiment | single (Galaxea R1 Lite) |

### 2.2 Hardware — Galaxea R1 Lite

23-DoF bimanual mobile manipulator:
- 2× 6-DoF arms（spherical wrists + parallel grippers）
- 3-DoF torso（vertical + pitch）
- 6-DoF vector-drive omnidirectional base（≤1.5 m/s）
- Payload: 5 kg; reach: 60 cm
- Perception: 1× stereo RGB head cam + 2× Intel RealSense D405 wrist RGB-D

关键设计选择：**isomorphic teleoperation**（同构遥操作），把 human operator 的 arm motion 直接 mapping 到 robot kinematics，这避免了 VR teleop 中常见的 retargeting failure 和 IK failure。这是为什么 dataset 的 action quality 高——operator 永远在 kinematically feasible posture 里操作。

参考 DROID [22] 也讨论过 teleop interface 对 data quality 的影响：https://droid-dataset.github.io/

### 2.3 Annotation schema

每个 episode 被 segmented 成 atomic subtasks，annotation 走 fixed schema（从标准化描述池里选，而非 free-form text）。这点很关键——free-form annotation 在 Open-X-Embodiment [1] 里是出了名的 noisy，不同 source 的 annotation granularity 差异巨大。Galaxea 用 schema-based annotation 提高了 labeling speed 和 consistency，这是为什么 subtask-level language-action alignment 能 work 的前提。

### 2.4 跟现有 dataset 的对比

| Dataset | Embodiment | Scene realism | Annotation |
|---|---|---|---|
| BridgeData V2 [21] | single | lab/staged | task-level |
| DROID [22] | single-ish | in-the-wild | task-level |
| Open-X-Embodiment [1] | multi (60+) | heterogeneous | inconsistent |
| AgiBot World [24] | single | staged lab | subtask |
| RoboMIND [23] | multi | staged | subtask |
| **Galaxea** | **single** | **real open-world** | **subtask** |

Galaxea 的差异化卖点：**single embodiment + real open-world + subtask annotation** 三者同时满足。AgiBot World 也有 subtask annotation 但 scene 是 staged 的；OXE scene diverse 但 embodiment 杂、annotation noisy。

---

## 三、G0 Dual-System 架构

### 3.1 Kahneman System 1 / System 2 框架

灵感来自 Kahneman《Thinking, Fast and Slow》[4]，这是 robotics 社区最近流行的设计 pattern（参考 SayCan [3]、Hi Robot [19]、OpenHelix [18]）。

- **System 2 (G0-VLM)**: slow, deliberative planner。接收 human high-level instruction → 理解 scene → 输出 atomic subtask instruction 给 System 1。基于 Qwen2.5-VL [30]。
- **System 1 (G0-VLA)**: fast, reactive executor。接收 subtask instruction + 三路视觉 + proprioception → 输出 action chunk。基于 PaLiGemma [29] + flow-matching action expert。

两个 model **异步运行在不同频率**——这是 dual system 的核心 efficiency trick。System 2 不需要每帧都跑，它只在 subtask 切换时被 trigger；System 1 高频执行。这跟 π0 [12] 的 single-stream 设计形成对比。

### 3.2 G0-VLA 内部架构

```
[img1, img2, img3] ──► SigLIP encoder ──► MLP projector ──┐
                                                            ├─► Transformer (PaLiGemma LM)
[language tokens] ──────────────────────────────────────────┤      │
[proprioception tokens] ────────────────────────────────────┘      │ KV cache
                                                                    ▼
                                                          [Action Expert (Transformer)]
                                                                    │
                                                            flow matching
                                                                    │
                                                                    ▼
                                                          action chunk A_t (H steps)
```

注意：action expert 是**新初始化**的（在 Stage-2 才加进来），它 condition 在 VLM 的 KV cache 上，这是一个类似 π0 [12] 的设计——VLM 提供 semantic grounding，action expert 专注 continuous control。

π0 paper: https://arxiv.org/abs/2410.24164

### 3.3 Action representation

每个 timestep 输出 **action chunk** $\mathbf{A}_t = a_{t:t+k}$，horizon k。Chunk-based prediction 是从 ACT (Action Chunking Transformer) 来的传统，能减少 compounding error 并提高 throughput。π0 用 horizon=50，Galaxea 没明确说 k 值。

---

## 四、3-Stage Training Curriculum — 技术核心

这是 paper 最有教学价值的部分。

### Stage-1: Cross-embodiment pre-training（仅训 VLM）

**数据**: ~1000h OXE + 500h Galaxea（只用 high-level task description，不用 subtask annotation）+ 200h in-house（high-level only）

**为什么 Stage-1 只训 VLM 不训 action expert？** Paper 给了两个理由：
1. Cross-embodiment data 的 annotation quality 和 action accuracy 不一致，action expert 学不到 informative 东西
2. Diffusion/flow-matching loss 在 VLM representation 还没稳定之前会 harm learning

这是一个很重要的 engineering insight——**flow matching loss 对 representation quality 敏感**。如果你在 VLM 还没 converge 的时候就开始训 flow head，noise 会 backprop 回 VLM 破坏它的 semantic representation。所以先纯 autoregressive 把 VLM 训到 stable，再 attach action expert。

**Loss function (autoregressive next-token prediction)**:

$$p(\mathbf{A}_t^d) = \prod_{i=1}^{N} p(a_i^d \mid a_{<i}^d, o_t, l_t, s_t)$$

变量解释：
- $\mathbf{A}_t^d$: 时间 t 的离散化 action token 序列，上标 $d$ 表示 discrete
- $N$: token 序列长度
- $a_i^d$: 第 $i$ 个 action token
- $a_{<i}^d$: 第 $i$ 个之前的所有 tokens（causal mask）
- $o_t$: 三路视觉 observation
- $l_t$: language instruction
- $s_t$: proprioceptive state

用 **FAST tokenizer** [11] 把 continuous action chunk 转成 discrete tokens。FAST 基于 DCT (Discrete Cosine Transform) 压缩 action chunk 的时间冗余，比 naive binning 效率高很多。

FAST paper: https://arxiv.org/abs/2501.09747

### Stage-2: Single-embodiment pre-training（VLM + Action Expert）

**数据**: Galaxea Open-World Dataset（带 subtask annotation）

**Loss function (flow matching)**:

先看 maximum likelihood objective:

$$\max_\theta \mathbb{E}_{p(A_t, o_t, l_t, s_t)} \left[\log \pi_\theta(A_t \mid o_t, l_t, s_t)\right]$$

- $\theta$: 整个 VLA 的参数
- $\pi_\theta$: policy（VLA model）
- $A_t$: action chunk，horizon $H$

然后 flow matching loss:

$$\mathcal{L}_{\text{flow}}(\theta) = \mathbb{E}_{p(A_t^\tau \mid o_t, l_t, s_t)} \left[\left\|\nu_\theta(A_t^\tau, \tau, o_t, l_t, s_t) - u(A_t^\tau \mid A_t)\right\|^2\right]$$

变量详解：
- $A_t$: ground-truth action chunk（horizon $H$ 的连续动作序列）
- $A_t^\tau$: noisy interpolated action，$A_t^\tau = \tau A_t + (1-\tau)\varepsilon$
- $\tau \in [0,1]$: flow time parameter，$\tau=0$ 是纯噪声，$\tau=1$ 是 ground truth
- $\varepsilon$: 通常采样自 Gaussian $\mathcal{N}(0, I)$
- $\nu_\theta(\cdot)$: VLA 预测的 **velocity field**（flow），告诉你在当前 $A_t^\tau$ 位置应该往哪个方向走
- $u(\cdot)$: target flow，从 action trajectory 推导出来的 ground-truth velocity
- $o_t, l_t, s_t$: 同上

**Intuition**: Flow matching [Lipman et al. 2023] 是 diffusion 的 cousin，但更简单——它直接学一个 vector field 把简单分布（Gaussian）transport 到复杂分布（action distribution）。跟 DDPM 比，flow matching 的 ODE trajectory 更直、sampling 更快。π0 [12] 用这个取得了 SOTA throughput。

Flow Matching 原论文: https://arxiv.org/abs/2208.14518

**Stage-2 的两个关键 enabler**:
1. **Single embodiment**: action space 一致，action expert 不需要跨 embodiment adapt
2. **Language-action alignment**: subtask-level segmentation 让 instruction 和 trajectory 精确对齐

### Stage-3: Post-training（task-specific fine-tuning）

每个下游 task 最多用 100 trajectories，4 epochs。Loss 跟 Stage-2 一样（flow matching）。这是用来测试 pre-trained weights 的 generalization 的——如果 pre-training 真的有效，post-training 应该 sample efficient。

---

## 五、G0-VLM Training — 细节值得单独讲

### 5.1 Base model

Qwen2.5-VL [30]，instruction tuning。Qwen2.5-VL paper: https://arxiv.org/abs/2502.13923

### 5.2 数据构造 pipeline

这是 paper 里我觉得最 clever 的工程：

1. **Sample episodes** from Galaxea dataset，key frames（subtask 即将结束、gripper state change）sampling weight 更高
2. **Extract** head camera images + subtask annotations
3. **Feed k-frame history**: 当前帧 + 前 k 秒的 frames 和 actions（1-second interval），让 VLM 能 handle long temporal context
4. **LLM reasoning augmentation**: 用 **DeepSeek-R1** [reasoning LLM] 在 $D_{\text{labeled}}$ 上生成 human-style high-level instruction 和 robot verbal response

第 4 步特别有意思。他们**不给 reasoning LLM 看图像**，只给它 task name + historic subtasks + next subtask，让 LLM 推理出 human-style verbal instruction。

例子：
- Input: task="pull and push chairs", historic subtasks=[...], next subtask="pull chair out"
- DeepSeek-R1 推理: "I am going to be seated, could you help pull the chair out?"
- Robot response: "I am working on it!"

这是一个 **LLM-as-data-augmenter** 的 pattern——用 reasoning LLM 把 mechanical subtask annotation 扩写成 natural human-robot dialogue。这跟 Hi Robot [19] 的思路类似。

DeepSeek-R1: https://arxiv.org/abs/2501.12948

### 5.3 Evaluation

Table 1 很 striking：

| Model | Table bussing | Microwave | Make bed | Build blocks |
|---|---|---|---|---|
| Gemini-2.5-pro | 32.0 | 15.8 | 54.2 | 55.0 |
| Qwen2.5-VL-72B | 26.3 | 16.8 | 48.1 | 21.7 |
| Qwen2.5-VL-32B | 21.3 | 14.8 | 54.2 | 21.0 |
| Qwen2.5-VL-7B | 26.3 | 17.2 | 46.9 | 24.7 |
| **G0-VLM** | **83.3** | **74.2** | **78.2** | **75.6** |

G0-VLM 在 4 个 task 上平均超 baseline 50%+。这说明 general-purpose VLM（哪怕 72B）在 robotic action grounding 上根本不够用——必须 domain-specific fine-tune。这也是为什么 dual system 需要 System 2 自己训，而不是直接调 Gemini API。

---

## 六、实验结果 — 最有信息量的部分

### 6.1 Pre-trained weights comparison (Figure 9)

配置：
- G0 (Stage-1): 仅 Stage-1 pre-train
- G0 (Stage-2 200h): 仅 Stage-2，200h 数据
- G0 (Stage-2 400h): 仅 Stage-2，400h 数据
- G0 (Full): Stage-1 → Stage-2 (400h)
- G0 (Scratch): 无 action pre-training
- π0: Physical Intelligence 的官方 pre-trained weights [12]

**Key findings**:

1. **G0 (Full) 在 average progress score 上最高**，尤其在 object-picking 类 task（Table Bussing, Microwave, Bed Making）

2. **G0 (Stage-2) 在 language following + action consistency + whole-body control 上最强**——这很重要，说明 Stage-2 学的是 embodiment-specific 的东西

3. **G0 (Stage-1) 是所有 pre-trained models 里最差的**——这是 paper 最反直觉的发现。Cross-embodiment pre-training 单独拿出来用，效果比 from scratch 还差

4. **π0 baseline 表现也不如 G0 (Stage-2)**——π0 是在大量 cross-embodiment data 上训的，morphology 跟 Galaxea R1 Lite 差距大

### 6.2 Few-shot transfer (Figure 10)

用 20 trajectories fine-tune，10 epochs。

- **Stage-2 pre-training 显著提升 few-shot 性能**——success rate 和 action smoothness 都好
- **Stage-1 pre-training 几乎没有 advantage over scratch**——cross-embodiment action pre-training 对 few-shot adaptation 帮助很小

Intuition：few-shot 场景下，model 需要快速 adapt 到新 embodiment 的 dynamics。Stage-2 已经让 model 学会了 target embodiment 的 action manifold，few-shot 只是 refinement。Stage-1 学的是 "general robot action priors"，但这些 priors 跟 target embodiment 的 action space 不兼容，反而成了 noise。

### 6.3 Embodiment-specific actions (Figure 11) — 最有说服力

Bed Making 任务按 skill 拆解：
- Moving toward bed (chassis)
- Lifting torso + grasping quilt (torso + arms)
- Leaning torso back (torso)
- Flattening quilt (chassis + arms)

**Stage-2 pre-training 在 chassis 和 torso control 上大幅领先**

**Stage-1 pre-training 和 π0 在这些 embodiment-specific skills 上比 from scratch 还差**——negative transfer！

Paper 的 hypothesis：OXE dataset 里的 embodiment 跟 Galaxea R1 Lite（有 omnidirectional base + pitch torso）morphologically 差距太大，cross-embodiment pre-training 学到的 action priors 是 misleading 的。

---

## 七、Intuition building — 这篇 paper 教会我们什么

### 7.1 Cross-embodiment pre-training 的 hidden cost

社区默认假设：更多 data = 更好 generalization。但这篇 paper 显示，当 **embodiment gap 大到一定程度**，cross-embodiment pre-training 会引入 **negative transfer**。原因：

1. **Action space mismatch**: OXE 里的 robot 大部分是 single-arm fixed-base，action dimensionality 和 semantics 跟 bimanual mobile manipulator 完全不同
2. **Dynamics mismatch**: mobile base 的 whole-body coordination 在 OXE 里几乎不存在
3. **Annotation mismatch**: OXE 的 task-level annotation vs Galaxea 的 subtask-level

所以 pre-training 学到的 "pick and place" prior，在 fine-tune 时需要 unlearn 才能适应新 embodiment 的 kinematics。

### 7.2 VLA 训练的 "VLM-first" principle

Stage-1 只训 VLM 不训 action expert，这个设计选择背后的 insight 是：**VLM 的 semantic representation 是 VLA 的 foundation，flow matching loss 对 representation quality 敏感**。如果你在 VLM 还没 stable 时就 attach flow head，gradient 会破坏 VLM。

这跟 OpenVLA [10] 的全 autoregressive 设计、π0 [12] 的 joint training 设计都不同。Galaxea 的 curriculum 是更保守、更 stable 的训练策略。

OpenVLA: https://arxiv.org/abs/2406.09246

### 7.3 Dual system 的异步频率是关键

System 2 (VLM) 低频跑，System 1 (VLA) 高频跑。这避免了每帧都跑 VLM 的 latency cost。Long-horizon task（Bed Making）需要 VLM 重新 plan subtask，但 subtask 一旦确定，VLA 可以连续执行几秒不需要 VLM 介入。

这种架构跟端到端 single-stream VLA（如 RT-2 [25]）相比，在 long-horizon task 上有巨大 efficiency 优势。

RT-2: https://arxiv.org/abs/2307.15818

### 7.4 Reasoning LLM as data augmenter

用 DeepSeek-R1 不看图、只看 subtask sequence 就生成 human-style instruction——这说明 **high-quality atomic action annotation 已经包含了足够 semantic information**，reasoning LLM 只是在做 surface form 变换。这是一个很 scalable 的 data augmentation pattern，比雇佣 human writer 写 natural language instruction 便宜得多。

---

## 八、Limitations 和 open questions

Paper 没明确讨论的：

1. **Single embodiment = limited generalization to other robots**。Galaxea dataset 只有一个 embodiment，如果你想在 Franka 或 UR5 上 deploy，Stage-2 pre-training 完全用不上。这是 single-embodiment 策略的 trade-off。

2. **Cross-embodiment pre-training 的 negative transfer 是否可逆？** Paper 没探索：如果 Stage-1 之后加一个 "embodiment adaptation" 阶段，能否 recover？参考 Embodied Chain-of-Thought [Google] 的思路。

3. **500h 数据够吗？** 相比 AgiBot World 的 100K+ hours，Galaxea 的 500h 算 small。Scaling laws 在 single-embodiment 设定下还成立吗？这需要后续实验。

4. **G0-VLM 的 plan 错误如何 propagate 到 VLA？** Dual system 的 classic problem：如果 VLM 输出了错误的 subtask instruction，VLA 会忠实执行错误。Paper 没讨论 error recovery。

5. **Flow matching vs autoregressive 的 trade-off**。Paper 用了两种 generation paradigm（Stage-1 autoregressive, Stage-2 flow matching），但没有 ablation 比较 pure autoregressive vs pure flow matching 在 single-embodiment 设定下的差异。参考 HybridVLA [17] 的 hybrid 设计。

HybridVLA: https://arxiv.org/abs/2503.10631

---

## 九、跟相关工作的 positioning

| 维度 | Galaxea G0 | π0 [12] | OpenVLA [10] | RT-2 [25] | Hi Robot [19] |
|---|---|---|---|---|---|
| Architecture | Dual (VLM + VLA) | Single VLA | Single VLA | Single VLA | Dual (LLM + VLA) |
| Action generation | Flow matching | Flow matching | Autoregressive | Autoregressive | Diffusion |
| Pre-training data | Cross + Single | Cross (large) | Cross (OXE) | Web data | - |
| Key claim | Single-embodiment > cross | Scale + flow | Open-source repro | Web knowledge transfer | Open-ended instruction |
| Embodiment gap analysis | ✅ (核心贡献) | ❌ | ❌ | ❌ | ❌ |

Galaxea G0 的独特贡献是**系统性地 disentangle 了 cross-embodiment 和 single-embodiment pre-training 的贡献**，并给出了 negative transfer 的 evidence。这是 community 之前回避的问题。

---

## 十、个人 takeaways（给 Andrej 的 intuition）

1. **"Data scale is all you need" 这个叙事在 robotics 里需要 refine**。LLM 的 scaling law 之所以 work，是因为所有 data 共享同一个 token space。Robotics 的 action space 是 embodiment-specific 的，cross-embodiment data 的 "transferable knowledge" 主要在 vision-language 层面，不在 action 层面。所以 **VLM pre-train on cross-embodiment 是 OK 的，action expert pre-train on cross-embodiment 是 dangerous 的**。

2. **Dataset 的 quality > quantity** 在 robotics 里比在 NLP 里更成立。Galaxea 的 500h single-embodiment high-fidelity data > OXE 的 1000h heterogeneous data（对于 target embodiment 的 fine-tuning 而言）。

3. **Dual system 是 pragmatic 的工程选择，不是 scientific breakthrough**。System 2 的 plan 能力 + System 1 的 reactive execution，本质上是在 approximate 一个 ideal VLA 的 hierarchical computation。但 pragmatically，异步频率带来的 efficiency gain 是巨大的，让 real-world deployment 成为可能。

4. **Reasoning LLM as data augmenter** 是一个被低估的 pattern。如果你有 high-quality atomic annotations，可以用 reasoning LLM 扩写成各种 surface forms（human dialogue, plan explanation, etc.），这比直接训 VLM 生成这些更 sample efficient。

5. **Open question**: 是否存在一个 "universal action representation" 能让 cross-embodiment pre-training 真 work？比如学习一个 embodiment-agnostic 的 action embedding（类似 LLM 的 token embedding）。目前 community 还没找到。参考 UniPi [Du et al.] 和 RT-X [1] 的尝试。

UniPi: https://uni-pi.github.io/
RT-X: https://robotics-transformer-x.github.io/

---

## Reference links

- [Galaxea G0 Project]: https://opengalaxea.github.io/G0/
- [PaLiGemma]: https://arxiv.org/abs/2407.07726
- [π0]: https://arxiv.org/abs/2410.24164
- [π0.5]: https://arxiv.org/abs/2504.16054
- [FAST tokenizer]: https://arxiv.org/abs/2501.09747
- [Open-X-Embodiment]: https://robotics-transformer-x.github.io/
- [Qwen2.5-VL]: https://arxiv.org/abs/2502.13923
- [DeepSeek-R1]: https://arxiv.org/abs/2501.12948
- [Flow Matching]: https://arxiv.org/abs/2208.14518
- [SayCan]: https://arxiv.org/abs/2204.01691
- [Hi Robot]: https://arxiv.org/abs/2502.19417
- [OpenHelix]: https://arxiv.org/abs/2505.03912
- [OpenVLA]: https://arxiv.org/abs/2406.09246
- [RT-2]: https://arxiv.org/abs/2307.15818
- [BridgeData V2]: https://arxiv.org/abs/2308.12952
- [DROID]: https://droid-dataset.github.io/
- [AgiBot World]: https://arxiv.org/abs/2503.06669
- [RoboMIND]: https://arxiv.org/abs/2412.13877
- [HybridVLA]: https://arxiv.org/abs/2503.10631
- [Is Diversity All You Need]: https://arxiv.org/abs/2507.06219
- [RDT-1B]: https://arxiv.org/abs/2410.07864
- [CogACT]: https://arxiv.org/abs/2411.19650
- [Kahneman, Thinking Fast and Slow]: https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow

---

如果你想 dive deeper，我建议重点看 Figure 11 的 per-skill breakdown——那是整篇 paper 最有说服力的 evidence，直接展示了 negative transfer 在 embodiment-specific skills 上的表现。Stage-2 在 chassis control 上几乎是 Scratch 的 2-3 倍，而 Stage-1 和 π0 反而低于 Scratch。这个 result 应该会改变 community 对 cross-embodiment pre-training 的默认假设。
