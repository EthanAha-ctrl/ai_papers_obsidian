---
source_pdf: Embodied AI From LLMs to World Models.pdf
paper_sha256: ec1904b14429e297a2014fa538c2c4d347546fe7ac50953d02d89e497fc564aa
processed_at: '2026-08-04T03:29:59-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话总结

作者的核心 thesis 很简单：**MLLMs 懂语义但不懂物理，WMs 懂物理但不懂语义，两者必须联合才能造出真正能在真实世界里跑的 embodied agent**。

---

## 论文真正想说的事

如果让我把这篇 50 页的综述压缩成一个 insight，那就是：

> **Embodied AI 的本质挑战是 Harnad 1990 symbol grounding problem 的物理化版本。LLM 解决了 symbol → symbol 的 grounding，但没解决 symbol → physics 的 grounding。**

LLM 训练数据来自互联网文本，它学到的是 "把杯子放在桌子上" 这句话与其他话的统计关系，但它不知道这个动作涉及重力、摩擦、杯子是否会碎。它对物理世界的理解是 **间接的、通过人类语言折射的**。

WM 反过来，它直接从 sensorimotor 数据学习物理 dynamics，但缺乏 high-level 抽象。一个 Dreamer agent 能预测球会滚到哪里，但它不能回答 "把客厅收拾干净" 这个指令该分解成哪些步骤。

这就是为什么作者说 joint architecture 是必经之路——它要 bridge 两个 representation space：一个符号的、一个物理的。

参考 Harnad 原文：http://www.cs.uregina.ca/Research/CIC/ExposingCS/Course_Reading_Exercises/B_Z_Harnad.pdf

---

## 论文抓对了什么

我觉得作者抓对了三件大事：

### 1. Embodied AI 的范式演变是对的

从 unimodal (vision-only / language-only / action-only) → multimodal → LLM-driven → WM-driven → joint MLLM-WM，这个演化路径在历史上是清晰的。

每一代解决了上一代的瓶颈：
- Unimodal 解决不了 modality gap
- Multimodal (CLIP-style) 解决了 alignment 但没有 action
- LLM-driven (SayCan) 加了 planning 但 action space 是固定 library
- VLA (RT-2) 加了 end-to-end action generation 但没有 physical imagination
- WM-driven (Dreamer) 加了 imagination 但没有 semantic understanding
- Joint MLLM-WM 把两端连起来

这个脉络是真实的、有说服力的。

### 2. MLLM 和 WM 的互补性诊断是对的

Table IV 那张表虽然用了 LOW/MEDIUM/HIGH 这种粗略评分，但诊断是对的。我特别认同这两行：

- **Physics Compliance**: MLLM "ignores physical constraints" vs WM "physics-aware simulation"
- **Future Prediction**: MLLM "lacks imagination-based reasoning" vs WM "long-horizon multi-step prediction"

这两点其实是同一件事的两个面：MLLM 没有 forward model，所以它无法想象 "如果我做 X，世界会变成什么样"。这种 imagination 缺失是 MLLM 在 robotics 上的根本瓶颈。

参考 LeCun 的 position paper：https://openreview.net/pdf?id=BZ5a1r-kVsf

LeCun 在 2022 年就讲过 LLM 不能 plan 因为它没有 world model。这篇综述把这个论点用 systematic 的方式重新包装了一遍。

### 3. Memory 三层结构抓住了 embodied cognition 的本质

作者提出的 memory 三层结构——forgetting past、renewing current、predicting future——非常对应认知科学里的：
- **Episodic memory** (过去)
- **Working memory** (现在)  
- **Prospective memory** (未来)

这不是随便分的，这是 hippocampus 真实的工作方式。Reference: [Predictive Coding in the Hippocampus](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(20)30192-5)

---

## 论文没说透的地方

### 1. Joint architecture 部分非常 hand-wavy

Section V-B 给出了 Fig. 7 的三段式 workflow，用箭头描述信息流，但**没有任何具体的实现细节**。

真正难的问题被回避了：
- MLLM 的 token representation 和 WM 的 latent state 如何 align？
- 谁的 representation 应该 dominate？是 MLLM 蒸馏到 WM，还是 WM 蒸馏到 MLLM？
- MLLM 的 reasoning 是秒级延迟，WM 的 simulation 是毫秒级，两者时钟不一致怎么办？
- LLM 是 open-ended generation（概率空间无限），WM 是 bounded simulation（受物理约束），两个概率空间怎么 reconcile？

作者在 Section V-C 提到了这些 challenge，但没给方向。这其实是论文最大的 weakness——thesis 很对，但怎么实现完全没讲。

### 2. EvoAgent 作为 evidence 太弱

作者引自己的 EvoAgent (https://arxiv.org/abs/2502.05907) 作为 joint architecture 的 early implementation。但 EvoAgent 主要是在 Minecraft 这种 constrained environment 里做 long-horizon tasks，离真正 physical world 的 joint architecture 还很远。

把它作为 "joint MLLM-WM 已经 work" 的 evidence 是过度乐观了。

### 3. 对 Diffusion-based WMs 讨论不够

2024 年最大的 trend 是 diffusion 作为 WM 的 backbone——[Diffusion Forcing](https://arxiv.org/abs/2407.01392)、[Sora](https://arxiv.org/abs/2405.03520)、[Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)。

Diffusion-based WMs 的优势是能生成 high-fidelity visual prediction，劣势是 sampling 慢、不利于 real-time control。这个 tension 是当前 WM 研究的核心 tension，但论文只在 Fig. 6 里一笔带过。

Sora 到底是不是 "world simulator" 这个争论（[Is Sora a World Simulator?](https://arxiv.org/abs/2405.03520)）应该深入讨论——Sora 没有 action conditioning，所以它是 **passive world model**，对 embodied AI 价值有限。Genie 2 才是真正意义上的 **active world model**，因为它能响应 latent action。

这个区分论文没讲清楚。

### 4. Sim-to-real 的核心难题被简化

Hardware embodiment 部分讲了一堆 quantization/pruning/TPU/FPGA，但**没讲 sim-to-real 的真正难题**：

- Domain randomization 如何平衡多样性 vs 真实性？
- System identification 如何从少量真实数据校准 simulator？
- Real2Sim2Real pipeline 中的 representation gap 怎么处理？

这些都是 embodied AI 真正 hard 的 engineering 问题，论文把它们藏在 "hardware embodiment" 的一句话里。

---

## 我对未来方向的直觉判断

让我跳出这篇 paper 讲讲我对这个领域的直觉。

### 直觉 1: World Models 会不会 scale？

这是 Karpathy 你会关心的核心问题。LLM 之所以能 scale 是因为互联网上有无限文本。WM 需要 physical interaction 数据，天然受限。

**但有几个可能的 scale 路径**：

1. **Video as implicit WM training data**：YouTube 上有海量视频，video 可以作为 action-conditioned WM 的训练数据。Genie 2 走的就是这条路——从 video 学 latent action space，无需 labeled action。参考：https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/

2. **Self-play in simulation**：MuZero 在 Atari/Go/Chess 上证明了 self-play 可以 scale。如果 simulator 足够好（比如 Genesis, Isaac Gym），robot 可以在 sim 里 self-play 无限次，然后把学到的 policy distill 到 real world。

3. **Real-world fleet learning**：Tesla 的 approach。几千辆车在跑，数据自然 scale。但这个 scale 受限于部署规模，不是 internet scale。

我的直觉是 **video pretraining + self-play finetuning** 是 WM 真正能 scale 的路径，类似于 LLM 的 web pretraining + RLHF finetuning。

### 直觉 2: MLLM-WM 联合的真正形式可能不是这篇 paper 讲的那样

作者讲的是 MLLM 和 WM 作为两个模块，通过箭头连接。但我觉得真正的联合可能是：

**Language 作为 WM 的 conditioning interface，WM 作为 language model 的 imagination engine。**

具体来说：
- LLM 输出 language plan + latent goal
- WM 接收 language conditioning，imagine 出 future trajectory
- LLM 看 imagination 结果，refine plan
- 循环

这其实就是 LeCun 在 H-JEPA position paper 里讲的架构（https://openreview.net/pdf?id=BZ5a1r-kVsf），只是他没强调 language conditioning。V-JEPA + language conditioning 可能就是 joint MLLM-WM 的最终形态。

### 直觉 3: Bitter Lesson 在 embodied AI 里如何体现？

Sutton 的 bitter lesson 说：general methods that leverage computation 终将胜过 specialized methods。

在 embodied AI 里，这意味着什么？我猜是：
- **End-to-end VLA + WM pretraining on massive video** 会胜过 modular architecture (perception module + planning module + control module)
- **Action prediction as next-token prediction** 会胜过 explicit policy optimization
- **Latent world model pretraining** 会胜过 task-specific representation learning

但 bitter lesson 的反例也存在：pure RL 在 Go/Atari 上 work，但在 robotics 上没 scale，因为 sample efficiency 太低。WM-driven RL 是不是 "general method that leverages computation"？我倾向于 yes，但需要 video pretraining 这个 unlock。

### 直觉 4: Embodied AI 和 Agentic AI 的真正区别

Agentic AI（AutoGPT, BabyAGI, SWE-agent）是 digital embodiment，在 cyberspace 里 act。

Embodied AI 是 physical embodiment，在 physical world 里 act。

两者在 LLM-driven planning 层共享，但分化在 grounding：
- Agentic AI 的 grounding 是 API spec + code execution environment
- Embodied AI 的 grounding 是 physics + sensorimotor feedback

Agentic AI 的 grounding 相对容易（API 是确定性的），embodied AI 的 grounding 非常难（physics 是连续的、stochastic 的、partial observable 的）。

这就是为什么 embodied AI 比 agentic AI 慢一个 generation——不是算法慢，而是 grounding 难。

### 直觉 5: 真正的 open problem 是 representation alignment

如果让我指出这篇 paper 没讲清楚的最关键的 open problem，那就是：

**MLLM 的 representation 和 WM 的 representation 是同一个 space 还是两个 space？**

如果是同一个 space：那 MLLM 内部其实已经隐含了一个 WM，只是没显式 train。Llama-3 内部有没有 emergent world model？这个 [Othello-GPT](https://arxiv.org/abs/2210.13346) 类工作提示有，但很弱。

如果是两个 space：那需要 learn 一个 alignment。这个 alignment 怎么 train？contrastive learning？distillation？co-training？

我倾向于认为：**最终是同一个 space，但需要显式 WM objective 来 sharpen LLM 内部 emergent world model**。这类似于 LLaVA 用 visual instruction tuning sharpen 了 LLM 的 visual understanding——WM training 可以 sharpen LLM 的 physical understanding。

这个 idea 其实就是 [Xiang et al. 2023 "Language Models Meet World Models"](https://arxiv.org/abs/2305.18626) 的思路，但还没 scale 起来。

---

## 给 Karpathy 的几个具体 takeaway

1. **如果你想做 embodied AI 研究，joint MLLM-WM 是正确的 framing**，但具体的 alignment mechanism 是真正的 research opportunity。这篇 paper 给了 framework，没给 solution。

2. **Video pretraining for action-conditioned WMs** 是下一个 internet-scale opportunity。Genie 2 是 early signal，但 action space learning 还很初步。谁能把 latent action space 学得更结构化（hierarchical、disentangled），谁就 unlock 了 embodied AI 的 GPT-3 时刻。

3. **Diffusion + flow matching + WM** 这个组合是 real-time embodied control 的 promising direction。π-0 已经展示了 flow matching for action generation，下一个 step 是 flow matching for world simulation。

4. **Self-play in learned WMs** 可能是 sample efficiency 的 unlock。与其在 real world trial-and-error，不如在 learned WM 里 self-play 无限次，distill 出 policy 再 transfer。Dreamer 在 Atari 上证明了这点，但还没 scale 到 general robotics。

5. **The bitter lesson for embodied AI** 我预测是：**massive video pretraining of action-conditioned WMs + simple policy extraction** 会胜过精心设计的 modular architecture。这是 5-10 年的 bet。

---

## 最后的 meta 评论

这篇 paper 写得是 qualified 的 survey，但**作为 position paper 它的核心 proposal (joint MLLM-WM) 没有 validate**。这是中国学术圈 survey 的常见模式——用全面性代替深度，用框架代替实验。

真正想理解 joint MLLM-WM 的人应该读：
- [LeCun H-JEPA position paper](https://openreview.net/pdf?id=BZ5a1r-kVsf) — 架构哲学
- [DreamerV3](https://arxiv.org/abs/2301.04104) — WM + RL 的工程实现
- [Genie 2 blog](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) — video-pretrained WM
- [π-0](https://arxiv.org/abs/2410.24164) — VLA + flow matching 的最新尝试
- [EvoAgent](https://arxiv.org/abs/2502.05907) — 作者团队的 concurrent work

真正想做 joint MLLM-WM 的人应该思考：**你的 language conditioning 是怎么 inject 到 WM 里的？token concatenation？cross-attention？FiLM layer？adapter？这个选择会决定整个架构的 inductive bias**。

这是真正的 research frontier，paper 给了我们 map，但 territory 还要自己走。

---

# Embodied AI: From LLMs to World Models 深度讲解

这是 Tongtong Feng, Xin Wang, Yu-Gang Jiang, Wenwu Zhu（清华大学计算机系）于 2025 年发表的综述论文。论文核心论点是 Embodied AI 正经历从 LLMs/MLLMs 驱动到 World Models (WMs) 驱动的范式转变，并提出 MLLM-WM 联合驱动架构作为下一代方向。

参考链接：
- arXiv: https://arxiv.org/abs/2502.06860
- Tsinghua 实验室: http://netmedia.cirowanglab.org/
- SayCan paper: https://say-can.github.io/
- DreamerV3: https://danijar.com/project/dreamerv3/
- Genie (Google): https://sites.google.com/view/genie-interactive/home
- JEPA (LeCun): https://openreview.net/forum?id=BZ5a1r-kVsf

---

## I. 核心论点与论文整体结构

论文提出一个清晰的 thesis：**MLLMs 擅长 semantic reasoning 与 task decomposition，但缺乏 physical grounding；WMs 擅长 physics-aware simulation，但缺乏 high-level semantics。两者联合才能 bridge 符号智能与物理交互**。

论文整体组织如下：

| Section | 内容 | 核心贡献 |
|---------|------|----------|
| II | Embodied AI 基础 | History, Key Technologies, 3 Key Components, Hardware |
| III | LLMs/MLLMs 驱动 | SayCan, PaLM-E, RT-2, VLA 分类 |
| IV | WMs 驱动 | RSSM, JEPA, Transformer-based 三大类 |
| V | MLLM-WM 联合架构 | 本论文的核心 insight |
| VI | Applications | Service Robotics, Rescue UAVs, Industrial Robots |
| VII | Future Directions | Autonomous, Hardware, Swarm, Explainability |

---

## II. Embodied AI 基础

### A. 历史脉络

论文追溯了 Embodied AI 的思想根源：

1. **1950 Turing Embodied Turing Test**：Turing 在 "Computing Machinery and Intelligence" 中提出 intelligence 与 physical experience 内在关联。
2. **1980s Cognitive Science**：Lakoff & Johnson 的 "Metaphors We Live By" 强调 human cognition 源自 bodily experience（具身认知的哲学根源）。Harnad 提出 **symbol grounding problem**——符号表征必须连接到 sensorimotor reality。
3. **Late 1980s Robotics**：Rodney Brooks 提出 **subsumption architecture**（[Brooks 1986](https://ieeexplore.ieee.org/document/1088935)），通过 layered reactive modules 实现 behavior-based control，反对传统的 sense-plan-act 三段式。MIT **Cog project** 构建能 developmentally learn、imitate、social interaction 的 humanoid robot。
4. **2010s Deep Learning era**：从 motion control 转向 adaptive interaction。Levine 等 [End-to-end training of deep visuomotor policies](http://arxiv.org/abs/1604.07816) 让 robot 从 raw sensor 直接学习 action policy。
5. **2020s LLM/WM era**：当前阶段，LLMs/MLLMs 和 WMs 推动新范式。

### B. Key Technologies（5 大基础领域）

论文 Fig. 3 展示了五大技术栈驱动 Embodied AI：

| 技术领域 | 代表方法 | 在 Embodied AI 中的作用 |
|---------|---------|----------------------|
| **CV** | AlexNet, GAN, ResNet, ViT, DDPM, MAE, SAM | 视觉感知基础 |
| **NLP** | Transformer, BERT, T5, ChatGPT, Vicuna, LLaMA | 语言理解、任务规划、instruction following |
| **RL** | DQN, AlphaGo, PPO, SAC, RLHF, GRPO | 核心算法框架 |
| **LLMs/MLLMs** | Flamingo, Qwen-VL, Gemini-1.5, GPT-4o, Deepseek-R1 | cross-modal 理解、跨任务泛化 |
| **WMs** | Mental Model, RSSM, JEPA, DreamerV3, Sora, Genie | 环境动态建模、imagination-based planning |

### C. 三个核心组件

#### 1) Active Perception（主动感知）

论文将 active perception 分为三类（Table I）：

**a) Visual SLAM**：Simultaneous Localization and Mapping。论文区分 geometric-based（dense scene flow, triangulation consistency, graph structure）vs semantic-based（SLAM++ object-level semantics, DS-SLAM deep learning）。
- ORB-SLAM (2015): RGB-D + Stereo, geometric features
- TwistSLAM (2022): geometric + semantic，constrained SLAM in dynamic environment
- GS-SLAM (2024): 用 3D Gaussian Splatting，volumetric object-level mapping

**b) 3D Scene Understanding**：用 vision-language 模型增强 3D 理解
- Clip2Scene, OpenScene: vision-language embeddings 实现 open-vocabulary 3D understanding
- GraphDreamer: 用 scene graph 建模 object-level relations
- HUGS, RegionPLC: region-level multimodal grounding

**c) Active Environment Exploration**：
- Model-based: MAX, Active Neural SLAM（预测式建模）
- Model-free: APT, DBMF-BPI（直接交互）
- Multimodal: ActiveRIR (RGB + Audio)

#### 2) Embodied Cognition（具身认知）

Embodied cognition 是核心，论文 Table II 分三类：

**a) Task-driven Self-Planning**：基于 task goal、environment context、internal knowledge 自主生成 plan。
- L3P, Egoplaner, ETPNav: structured learning latent planning space
- LLM-Planner, AutoAct: LLMs 集成 planning
- RPG: generative + multimodal reasoning

**b) Memory-driven Self-Reflection**：用过去 experience 进行长程推理、错误纠正、自我改进。
- Reflexion [Shinn et al. 2023](https://arxiv.org/abs/2303.11366): self-generated linguistic feedback 存入 episodic memory buffer
- EvoAgent (本文作者团队 2025): continual world model + memory-driven planner，全自主 evolution

**c) Embodied Multimodal Foundation Models**：
- SayCan, GATO: affordance-grounded agents
- EmbodiedGPT, Kosmos-2: vision-language pretraining
- MultiPLY, ManipLLM: object-centric manipulation

#### 3) Dynamic Interaction（动态交互）

Table III 分三类：

**a) Action Control**：生成 motor commands
- Control theory / RL 早期方法
- **VLA models**: PaLM-E, RT-2, OpenVLA, CogAgent, Octo, CrossFormer, HPT
- PaLM-E: visual + language encoder end-to-end + LLM，continuous sensor modalities
- RT-2: encode image+language+action 为 text tokens，LLM 推理后 de-tokenize 为 action

**b) Behavioral Interaction**：high-level behavior patterns
- GAIL, MGAIL, TrafficSim, TrajGen: imitation learning
- BEHAVIOR-1K: 1000 everyday tasks 大规模 benchmark

**c) Collaborative Decision**：多智能体协作
- QMIX, QTRAN, QPLEX, MAT: 多智能体 RL
- MetaGPT, CoELA, AgentVerse: LLM-driven coordination
- COMBO: modular WMs 支持 scalable cooperation

### D. Hardware（4 大方向）

Embodied AI 硬件优化包括：

1. **Hardware-aware Model Compression**：quantization（weights/activations → lower bit-width）+ pruning（移除 redundant parameters）。PPA (power, performance, area) 指标指导 bit-width allocation。
   - SmoothQuant [Xiao et al. 2023](https://arxiv.org/abs/2211.10438)
   - HAQ: hardware-aware automated quantization

2. **Compiler-level Optimization**：TVM（基于 LLVM + CUDA）通过 operator fusion、redundant computation elimination、loop reordering、tiling 优化。

3. **Domain-specific Accelerators (DSAs)**：
   - TPU: matrix multiplication 加速
   - FPGA: reconfigurable
   - CGRA: structured dataflow
   - ASIC: high throughput + energy efficiency

4. **Hardware-Software Co-design**：algorithm-system 与 algorithm-hardware 联合优化。

### E. Unimodal → Multimodal 范式演变

论文 Fig. 4 阐述了从 unimodal 到 multimodal 的演变。Unimodal 方法的局限：
- 单模态信息有限（visual-only 在 dynamic/ambiguous 场景失效，auditory 在 real-world noise 失效）
- 异构 modality 之间难以 transfer（perception → cognition → interaction 之间存在 gap）

Multimodal 范式通过 MLLMs + WMs 集成多感官，打破上述限制。

---

## III. Embodied AI with LLMs/MLLMs

### A. LLMs 驱动

LLMs 通过两条路径驱动 Embodied AI：

#### 1) Semantic Reasoning

LLMs 通过 transformer 架构将 input tokens 映射到 latent representations，跨 syntactic 与 pragmatic levels 形成 hierarchical abstraction。Attention 机制加权 relevant semantic cues 同时抑制 noise。

LLM 内部计算的简化形式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$: query matrix
- $K \in \mathbb{R}^{m \times d_k}$: key matrix  
- $V \in \mathbb{R}^{m \times d_v}$: value matrix
- $d_k$: key/query 维度，$\sqrt{d_k}$ 用于 scaled dot-product 防止梯度消失
- $n, m$: sequence length

#### 2) Task Decomposition

通过 chain-of-thought (CoT) prompting，LLMs 层级化解析 instruction 为 actionable steps，检查 contextual dependencies。

#### SayCan 详解

SayCan [Ahn et al. 2022](https://say-can.github.io/) 是 LLM-driven Embodied AI 的代表作，三步骤：

1. **Natural language action library**：预训练真实世界动作库，约束 LLMs 提出可行 action
2. **Language → action sequence**：LLM 将 instruction 转为 action sequence
3. **Value function verification**：用 value function 验证 action sequence 在物理环境中的可行性

形式化：

$$p(a_i | \text{instruction}, \text{context}) = \underbrace{p_{\text{LLM}}(a_i | \text{instruction}, a_{<i})}_{\text{language prior}} \cdot \underbrace{V(a_i | \text{state})}_{\text{affordance}}$$

其中 $V(\cdot)$ 是 affordance value function 评估在当前 state 下执行 $a_i$ 的可行性概率。

**SayCan 的局限**：固定 action library + 特定物理环境，难以扩展到新 robot 和新环境。

### B. MLLMs 驱动

MLLMs 解决 LLMs 的问题，bridge high-level multimodal input 与 low-level motor action sequence 为 end-to-end。MLLMs 包括 VLMs 和 VLAs。

#### 1) VLMs for Embodied AI

PaLM-E [Driess et al. 2023](https://palm-e.github.io/)：
- visual + language encoder end-to-end 训练
- 与 pretrained LLM 联合
- continuous sensor modalities encoding 集成到 VLM
- fixed action space mapping 多任务完成

PaLM-E 的多模态融合公式（简化）：

$$h = \text{LLM}([W_o \cdot O, W_t \cdot T, W_p \cdot P])$$

其中：
- $O$: observation embedding (visual)
- $T$: text instruction embedding
- $P$: proprioception embedding (robot state)
- $W_o, W_t, W_p$: 各自的 projection matrices
- $[\cdot]$: token concatenation
- $h$: 输出 hidden state，再映射到 action

#### 2) VLAs for Embodied AI

**RT-2** [Brohan et al. 2023](https://robotics-transformer2.github.io/)：
1. encode robot image + language instruction + robot actions → text tokens
2. LLM 进行 semantic reasoning 与 task decomposition
3. de-tokenize 输出 tokens 为 final action

RT-2 用 action tokenization 将连续 7-DoF action 离散化为 256 bins，每个 dimension 一个 token，从而让 LLM 的 next-token prediction 直接输出 action：

$$a_t = \text{Detokenize}(\text{LLM}(I_t, L_t, a_{<t}))$$

其中 $I_t$ 是 image tokens，$L_t$ 是 language tokens，$a_{<t}$ 是历史 action tokens。

**Octo** [OpenVLA team 2024](https://octo-models.github.io/)：在 100K robot demonstrations + language annotations 上预训练，实现 cross-embodiment tool use。

**PerAct**：用 3D voxel representations 实现 millimeter-level grasp accuracy，适合 dexterous manipulation。

### C. MLLM 分类

MLLMs 增强 Embodied AI 的三个组件：

#### 1) MLLMs for Active Perception

- **3D SLAM 增强**：MLLMs 提供 object categories、spatial relations、scene semantics
  - SEO-SLAM: MLLM 生成 specific 描述性 labels，动态更新 multiclass confusion matrix 抑制 object detection bias
- **3D Scene Understanding**：camera-based (RGB) 主导
  - EmbodiedGPT: 2D visual inputs → semantically rich features aligned with language goals
- **Active Environment Exploration**：
  - LLM³: structured motion-level feedback (collision detection) 进入 planning loop
  - MART: interaction feedback 改进 retrieval quality

#### 2) MLLMs for Embodied Cognition

- **Task-driven Self-Planning**：
  - CoT-VLA: 预测 intermediate subgoal images，可视化 subtask outcomes
- **Memory-driven Self-Reflection**：
  - Reflexion: self-generated linguistic feedback 存入 episodic memory buffer
- **Embodied Multimodal Foundation Models**：
  - Qwen-VL, InternVL, Qwen2.5-Omni

#### 3) MLLMs for Dynamic Interaction

- **Action Control**：
  - autoregressive action generation（OpenVLA, RT-2）
  - auxiliary policy heads（Octo）
  - **executable code generation**（Code as Policies [Liang et al. 2023](https://code-as-policies.github.io/)）：LLM 生成可执行 Python 代码控制 robot
- **Behavioral Interaction**：
  - **π-0** [Physical Intelligence 2024](https://arxiv.org/abs/2410.24164): VLM backbone + flow-matching decoder 产生 smooth trajectories
- **Collaborative Decision**：
  - Combo: decentralized agents 仅用 egocentric visual observations
  - VLAS: 语音 + LLaVA-style MLLM 实现 conversational human-agent interaction

---

## IV. Embodied AI with World Models

WMs 通过 internal representations 与 future predictions 赋能 Embodied AI，实现 physics-law-compliant 交互。

### A. WMs 的两个核心能力

#### 1) Internal Representations

Internal representations 压缩 sensory inputs 到 structured latent spaces，捕获：
- Object dynamics
- Physics laws
- Spatial structures

关键属性：
- **Hierarchical relationships**：实体与环境之间的层级关系
- **Counterfactual reasoning**：通过 disentangled variables 维持物体 intrinsic properties 与 extrinsic relations
- **Causal understanding**：通过压缩 sensory experience 形成直觉理论

形式化表示一个 WM 的 latent state：

$$h_t = f_\theta(h_{t-1}, a_{t-1}, o_t)$$

其中：
- $h_t \in \mathbb{R}^d$: latent state at time $t$
- $f_\theta$: transition function with parameters $\theta$
- $a_{t-1}$: previous action
- $o_t$: current observation

#### 2) Future Predictions

Future predictions 模拟 sequence actions 的 potential rewards，跨多个 time horizons，aligned with physical laws。

关键能力：
- **Long-horizon prediction**：平衡 exploration-exploitation
- **Uncertainty quantification**：区分 predictable regularities 与 stochastic events
- **Sample efficiency**：用 mental rehearsal 替代 costly trial-and-error
- **Self-correction**：通过 prediction-error minimization 迭代 refine model

预测公式（rollout）：

$$\hat{o}_{t+k}, \hat{r}_{t+k} = g_\phi(\hat{h}_{t+k}), \quad \hat{h}_{t+j} = f_\theta(\hat{h}_{t+j-1}, \hat{a}_{t+j-1})$$

其中：
- $\hat{h}_{t+k}$: 预测的 latent state at future time $t+k$
- $\hat{o}_{t+k}$: 预测的 observation
- $\hat{r}_{t+k}$: 预测的 reward
- $g_\phi$: decoder with parameters $\phi$
- $k$: prediction horizon

### B. WMs 三大架构

#### 1) RSSM-based WMs

RSSM (Recurrent State Space Model) 是 **Dreamer algorithm family** 的基础架构。

DreamerV3 [Hafner et al. 2023](https://arxiv.org/abs/2301.04104) 的核心结构：

**RSSM 的关键创新**：将 hidden state 正交分解为 probabilistic 与 deterministic components：

$$h_t = f_\theta(h_{t-1}, a_{t-1}, s_{t-1})$$
$$s_t \sim q_\theta(s_t | h_t, o_t) \quad \text{(posterior, training)}$$
$$\tilde{s}_t \sim p_\theta(\tilde{s}_t | h_t) \quad \text{(prior, rollout)}$$

其中：
- $h_t \in \mathbb{R}^{d_h}$: deterministic recurrent state (GRU/LSTM output)
- $s_t \in \mathbb{R}^{d_s}$: stochastic state (通常是 Gaussian with mean $\mu$ 与 std $\sigma$)
- $o_t$: observation
- $a_{t-1}$: previous action
- $q_\theta$: posterior (encoder during training, 用了 $o_t$)
- $p_\theta$: prior (during imagination rollout，无 $o_t$)

**Loss function**：

$$\mathcal{L} = \underbrace{\mathbb{E}_q[\log p(o_t | s_t, h_t)]}_{\text{reconstruction}} + \underbrace{\mathbb{E}_q[\log p(r_t | s_t, h_t)]}_{\text{reward}} + \underbrace{\mathbb{E}_q[\log p(c_t | s_t, h_t)]}_{\text{continue}} - \underbrace{\beta \cdot \text{KL}[q_\theta(s_t|h_t,o_t) \| p_\theta(s_t|h_t)]}_{\text{KL regularization}}$$

actor-critic 在 imagination rollout 中训练：

$$\mathcal{L}_{\text{actor}} = -\mathbb{E}\left[\sum_{\tau=t}^{t+H} \gamma^{\tau-t} V_\psi(\hat{s}_\tau, \hat{h}_\tau)\right]$$
$$\mathcal{L}_{\text{critic}} = \mathbb{E}\left[(V_\psi(\hat{s}_\tau, \hat{h}_\tau) - \lambda_\tau^\text{SG})^2\right]$$

其中 $\lambda_\tau$ 是 $\lambda$-return，$\text{SG}$ 表示 stop-gradient。

DreamerV3 在 150+ tasks 上达到 SOTA，从 visual inputs 学习，无需 task-specific 调整。

#### 2) JEPA-based WMs

JEPA (Joint-Embedding Predictive Architecture) 由 Yann LeCun [2022](https://openreview.net/pdf?id=BZ5a1r-kVsf) 提出，路径通往 autonomous machine intelligence。

**JEPA 的核心思想**：在 abstract latent space 中进行 prediction，而非 pixel-wise reconstruction。这避免了 generative models 处理 irrelevant details 的浪费。

JEPA 的核心损失（以 I-JEPA 为例）：

$$\mathcal{L}_{\text{JEPA}} = -\frac{1}{|\mathcal{P}|} \sum_{i \in \mathcal{P}} \cos\left(\phi_y(y_i), \overline{\phi_x(x)}\right)$$

其中：
- $x$: context (visible part of input)
- $y_i$: target block $i$（masked regions）
- $\mathcal{P}$: set of target blocks
- $\phi_x$: context encoder
- $\phi_y$: target encoder
- $\overline{\phi_x(x)}$: predicted representations from context via predictor network
- $\cos(\cdot, \cdot)$: cosine similarity

**关键设计**：
- Target encoder $\phi_y$ 用 EMA (Exponential Moving Average) 更新，避免 representational collapse
- Predictor 用 mask tokens 知道要预测的位置

JEPA 衍生：
- **I-JEPA** [Assran et al. 2023](https://arxiv.org/abs/2301.08243): 图像自监督
- **V-JEPA** [Bardes et al. 2024](https://arxiv.org/abs/2310.12991): 视频
- **MC-JEPA**: motion + content
- **A-JEPA**: audio

#### 3) Transformer-based WMs

Transformer-based WMs 用 attention 处理序列，克服 RNN 的 sequential processing 约束。

**TransDreamer** [Chen et al. 2022](https://arxiv.org/abs/2207.05936) 与 **IRIS** [Micheli et al. 2023](https://arxiv.org/abs/2206.08139)：

IRIS 的核心：用 GPT-like Transformer 在 discrete token space 进行 rollout。

Tokenization: VQ-VAE 将 observation 编码为 discrete tokens $z_t \in \{1, ..., |V|\}^N$，其中 $|V|$ 是 codebook size，$N$ 是 token 数量。

自回归预测：

$$p(z_{t+1} | z_{\leq t}, a_t) = \prod_{i=1}^{N} p(z_{t+1}^{(i)} | z_{\leq t}, a_t, z_{t+1}^{(<i)})$$

**Genie** [Bruce et al. 2024](https://arxiv.org/abs/2402.15391) 是 Google 的代表作，用 **ST-Transformer** (Spatial-Temporal Transformer) 通过大规模 self-supervised video pretraining 创建 synthetic interactive environments。

ST-Transformer 用 factorized spatial + temporal attention：

$$\text{Attn}_{\text{ST}} = \text{Attn}_{\text{temporal}}(\text{Attn}_{\text{spatial}}(X))$$

先在 spatial dimension 做 attention（同一时间步内的空间 token），再在 temporal dimension 做 attention（同一空间位置的时间序列）。

### C. WMs 与 RL 的关系

Dreamer 系列将 WMs 与 actor-critic RL 结合。在 imagination 中 rollout 多步，用 $\lambda$-return 训练 critic，用 critic 的 value 反向传播训练 actor。这与 model-free RL (PPO, SAC) 形成对比，sample efficiency 显著更高。

---

## V. Joint MLLM-WM-driven Embodied AI Architecture（本文核心贡献）

### A. MLLM 与 WM 各自的局限

| 维度 | MLLM-only | WM-only |
|------|-----------|---------|
| Semantic Understanding | 强：contextual reasoning, NL understanding | 弱：open-ended semantic |
| Task Decomposition | 强：language prompts sub-task planning | 弱：缺乏 generalizable decomposition |
| Physics Compliance | 弱：忽略 physical constraints | 强：physics-aware simulation |
| Future Prediction | 弱：无 imagination-based reasoning | 强：long-horizon multi-step prediction with uncertainty |
| Real-time Interaction | 弱：reasoning latency, poor feedback response | 强：real-time predictive control |
| Memory Structure | 弱：sparse, unstructured | 强：structured latent space with causal relationships |
| Scalability | 弱：limited to pre-trained task space | 弱：poor transfer to unseen tasks without retraining |

### B. 互补关系

**MLLMs 增强 WMs Reasoning**（论文指出这是 potential direction，尚未在现有工作中完全实现）：

1. MLLM 融合 visual/auditory/textual 数据为 unified semantic representations
   - CLIP-based architectures align visual scenes 与 linguistic cues
2. MLLM 增强 WM 的 task decomposition，将 high-level goals 分解为 executable sub-tasks
   - GPT-4V 生成 step-by-step plans
   - Code-as-Policies 翻译 NL 指令为 code snippets
3. MLLM 通过 human feedback refine WM priors
   - RLHF 更新 WM priors

**WMs 增强 MLLMs Interaction**（同样是 potential direction）：

1. WMs 提供 physical laws (gravity, friction) 显式表示，constrain MLLM action proposals
   - Physion++ 集成 biomechanical models 过滤违反 torque limits 的 motion
   - RoboGuide 注入 spatial occupancy maps 防止 collision
2. WMs 维持 spatiotemporal context 稳定 MLLM reasoning
   - MemPrompt 用 WM buffers 对齐 visual trajectories 与 linguistic descriptions
3. WMs 通过 closed-loop interaction iteratively refine MLLM outputs
   - Reflexion 存 task-execution history 在 WM 中，让 MLLM 识别 failure patterns

### C. Joint MLLM-WM Architecture 详解（论文 Fig. 7）

论文提出一个三段式 closed-loop architecture：

**Workflow 1: Robots → Self-State Inputing → MLLMs/WMs → Hardware Embodiment → Robots**

- Self-state inputing：tracking proprioceptive metrics (degrees of freedom, number of sensors)
- WMs 用这些 metrics 构建 agent physical state 的 internal representation
- MLLMs 将这些 states contextualize 为 task alignment
- Hardware embodiment 实现 sim-to-real

**Workflow 2: MLLMs → Task Planning → WMs → Memory Updating → MLLMs**

- MLLM 分解 abstract instruction 为 subtasks
- Forward arrow: plan → WMs，WMs 基于现有 environment modeling 预测 outcomes
- Execution 中 WMs log outcomes 到 memory
- Vertical arrow: logs → memory updating module
- Memory 三层结构：
  - Forgetting of past task memories
  - Renewal of current task memories
  - Prediction of future task memories
- Backward arrow: enriched memory → MLLMs，实现 lifelong learning

**Workflow 3: Environments → Active Perception → MLLMs/WMs → Dynamic Interaction → Environments**

- WMs 驱动 active perception 预测 key environmental changes
- Multimodal inputs 构建 internal representation (WMs) + semantic reasoning (MLLMs)
- MLLM 的 task decomposition + WM 的 future prediction → action selection
- Adaptive perception 与 interaction 通过 continuous iteration 实现

### D. Table IV: 三种架构对比

| Performance | LLM/MLLM-only | WM-only | Joint MLLM-WM |
|-------------|--------------|---------|---------------|
| Semantic Understanding | contextual task reasoning, NL understanding | limited open-ended semantic | combines high-level semantic abstraction + grounded contextual alignment |
| Task Decomposition | sequential logic via language prompts | lacks generalizable decomposition | semantic plans refined through physical feasibility via joint planning-execution loop |
| Physics Compliance | ignores physical constraints | physics-aware simulation with temporal consistency | enforces semantic-physical alignment for safe executable plans |
| Future Prediction | lacks imagination-based reasoning | long-horizon multi-step with uncertainty | combines symbolic foresight + physically grounded imagination |
| Real-time Interaction | poor responsiveness, significant reasoning latency | supports real-time predictive control | enables online adaptation through iterative plan refinement + memory updating |
| Memory Structure | sparse, unstructured | structured latent space with causal relationships | integrates semantic memory + world modeling for lifelong learning |
| Scalability | limited to pre-trained task space | poor transfer to unseen tasks | cross-task, cross-domain generalization through symbolic + sensorimotor synergy |

### E. EvoAgent（联合架构的早期实现）

EvoAgent [Feng et al. 2025](https://arxiv.org/abs/2502.05907) 是本论文作者的同期工作，实现了 joint MLLM-WM-driven Embodied AI：

- **Continual World Model**：持续学习的环境模型
- **Memory-driven Planner**：基于记忆的规划器
- **Self-planning + Self-reflection + Self-control**：全自主 evolution
- 无需 human intervention 完成 long-horizon tasks

---

## VI. Applications

### A. Service Robotics

- RT-2, SayCan: stacking dishes, cooking
- AED: few-shot learning 新 skill
- Healthcare: 提醒、康复、陪伴
- Habitat, RT-X: navigation, item delivery
- Joint MLLM-WM: 长程任务如 "clean up the living room"

### B. Rescue UAVs

- 传统 UAV: 人工控制 or 预建地图，无法自主适应
- Embodied drones: 实时感知，快速响应
- LLMs 理解 voice instructions: "search near the collapsed bridge"
- WMs 模拟危险环境，规划 safe path
- Multi-drone collaboration: 寻找幸存者，绘制受损区域

### C. Industrial Robots

- Tesla 工厂: robots 自主找并修复未对齐零件
- JD: 多传感器按尺寸和地址分拣
- Tmall 仓库: thermal cameras + LiDAR + RGB 检测库存问题

### D. Other Applications

- Smart manufacturing: human-robot collaboration
- Education: social robots 调整 speech/gaze/gestures
- Virtual environments: embodied agents 学习多步任务 + memory
- Space exploration: 通信延迟下自主决策

---

## VII. Future Directions

### A. Autonomous Embodied AI

- **Adaptive Perception**：自主选择 input data，动态融合多感官
- **Environmental Awareness**：快速适应变化、预测 action consequences、transfer 到新环境
- **MLLM + Real-time Physical Interaction**：bridge 高层 language 与 low-level control

### B. Embodied AI Hardware

1. Hardware-aware model compression（quantization + pruning + PPA metrics）
2. Graph-level compilation optimization（operator fusion, scheduling, memory access）
3. Domain-specific accelerators（FPGA, CGRA, ASIC）
4. Hardware-software co-design（消除 algorithm-hardware mismatch）

### C. Swarm Embodied AI

- **Collaborative WMs**：基于多 agent 观测建立 shared dynamic environmental representation
- **Multi-agent Representation Learning**：理解自身状态 + 其他 agent 状态
- **Social Behavior Modeling**：role allocation + group decision-making
- **Human-Swarm Interaction Interfaces**：multimodal language + gesture-based control

### D. Explainability and Trustworthiness

- Real-time human-understandable justifications for agent actions
- Ethical principles adherence in morally ambiguous scenarios
- Verifiable safety guarantees + certification standards
- Robustness against adversarial attacks, sensor noise, distribution shifts

### E. Other Directions

- **Lifelong Learning**：持续学习新 skill 不遗忘旧 skill
- **Human-in-the-loop Learning**：human feedback 显著提升 agent performance
- **Moral Decision-making**：识别 moral hazard，遵循 human values

---

## VIII. 关键 Insight 总结

### Insight 1: Embodied AI 的范式三角

论文揭示了从 perception → cognition → interaction 三个 module 的协同：

```
                ┌──────────────────┐
                │  Embodied Cognition │
                │  (MLLMs: reasoning │
                │   + decomposition) │
                └────────┬──────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
┌──────────────────┐         ┌──────────────────┐
│ Active Perception │         │ Dynamic Interaction│
│ (CV + VLMs +      │◄────────►│ (VLA + WMs +      │
│  3D understanding)│         │  policy learning) │
└──────────────────┘         └──────────────────┘
```

### Insight 2: MLLM 与 WM 的互补本质

- **MLLM 处理 "What" 与 "Why"**：semantic interpretation, goal decomposition, contextual understanding
- **WM 处理 "How" 与 "What if"**：physical dynamics, future prediction, counterfactual simulation

二者结合形成 **semantic-physical alignment**，这正是 human intelligence 的核心特征——我们既理解语言的抽象含义，又能将之 grounded 到物理世界。

### Insight 3: Memory 的三层结构

论文提出的 memory 三层结构很有意思：
- **Forgetting past task memories**：防止 catastrophic interference
- **Renewing current task memories**：实时更新
- **Predicting future task memories**：anticipatory planning

这对应了人类认知中的 episodic memory (过去) + working memory (现在) + prospective memory (未来)。

### Insight 4: Sim-to-Real 的关键角色

Hardware embodiment 是 joint architecture 的关键环节。论文强调 sim-to-real 问题，即 WMs 在 simulation 中训练的策略如何迁移到真实世界。这涉及：
- Domain randomization
- System identification
- Real2Sim2Real pipelines
- Hardware-aware quantization 保留 critical precision

### Insight 5: 与 LeCun 的 JEPA 路线对比

LeCun 在 "A Path Towards Autonomous Machine Intelligence" 中提出 JEPA 作为通向 AGI 的路径。本文的 joint MLLM-WM architecture 与 LeCun 的 vision 高度一致：
- LeCun: WP (World Model) + Actor + Critic + Perception + Short-term Memory + Configuration
- 本文: MLLM (configurator + perception) + WM (world model + memory) + Actor (VLA)

但本文更强调 **MLLM 与 WM 的双向信息流**，而非 LeCun 的层级 control flow。

### Insight 6: VLA 的演化路径

VLA 模型从 RT-1 (Transformer only) → RT-2 (co-finetune on web data) → OpenVLA (open-source) → π-0 (flow matching) 的演化，反映了从纯 imitation 到 generative policy 的转变。未来的 joint MLLM-WM VLA 可能将 flow matching 与 WM 的 latent imagination 结合。

### Insight 7: Embodied AI 的本质挑战

论文隐含一个深刻洞察：**Embodied AI 的本质挑战是 grounding problem 的物理化**。Harnad 1990 的 symbol grounding problem 是抽象符号如何连接 sensory-motor reality。Embodied AI 把这个问题扩展为：**LLM 的语义表征如何 grounded 到物理动力学**。这是为什么 joint MLLM-WM 是必经之路。

---

## IX. 我对这篇论文的批判性思考

### Strengths

1. **架构洞察深刻**：MLLM-WM 联合架构的提出抓住了 Embodied AI 的核心矛盾
2. **覆盖全面**：从 history 到 hardware，从 algorithms 到 applications
3. **Tables 组织清晰**：Table I-IV 提供了 systematic 的方法对比
4. **EvoAgent 案例**：作者自己的工作佐证了 joint architecture 的可行性

### Weaknesses

1. **缺乏实验验证**：joint MLLM-WM 架构更多是 conceptual proposal，缺乏 end-to-end 实验对比
2. **量化不足**：Table IV 用 LOW/MEDIUM/HIGH 定性评估，缺定量指标
3. **WMs 部分略简**：对 diffusion-based 和 hierarchical WMs 的讨论不够深入
4. **硬件部分偏 textbook**：缺乏 embodied AI 特定的 hardware challenges 讨论
5. **MLLM-WM 交互机制不清晰**：具体如何实现 "WMs refine MLLM outputs" 的工程细节缺失
6. **Open problems 不够具体**：Section V-C 提的挑战（real-time sync, semantic-physical misalignment, scalable memory）没有给潜在 solutions

### Connections to Other Lines of Research

1. **World Models vs Foundation Models**：本文观点与 Sutton 在 "The Bitter Lesson" 中的论点形成有趣对比。Sutton 认为 general methods + computation 终将胜过 specialized methods。WMs 是 specialized for physics 还是 general method？这是开放问题。

2. **Embodied AI vs Agentic AI**：本文专注 physical embodiment，而 agentic AI (AutoGPT, BabyAGI) 是 digital embodiment。两者在 LLM-driven planning 层面共享，但在 physical grounding 层面分化。

3. **Cognitive Science Connection**：论文引用 Lakoff, Harnad, Brooks，但未深入 Clark 的 "predictive brain" 理论。Predictive processing 与 WMs 的 future prediction 高度契合，是潜在理论桥梁。

4. **Neuroscience Inspiration**：hippocampus 的 place cells 与 WMs 的 spatial representations，cerebellum 的 forward models 与 WMs 的 prediction，这些都可作为 biological inspiration。

5. **与 Diffusion World Models 的关系**：Diffusion 系 WMs（如 Diffusion Forcing, Sora-like video generation models）未充分讨论。Sora 作为 "world simulator" 的争议（[相关综述](https://arxiv.org/abs/2405.03520)）值得更多笔墨。

---

## X. 总结

这篇综述的核心贡献在于系统阐述了 Embodied AI 从单模态 → 多模态 → LLM-driven → WM-driven → 联合 MLLM-WM 的演化路径，并提出了联合架构作为下一代方向。

论文的最大价值是**抓住了 Embodied AI 的核心矛盾**：semantic intelligence 与 physical grounding 之间的鸿沟。MLLMs 与 WMs 的联合是 bridge 这一鸿沟的必经之路。

未来研究的几个关键问题：
1. MLLM 与 WM 之间如何实现高效的 information flow？token-level alignment 还是 latent-space alignment？
2. 如何在 sim-to-real 过程中保持 joint architecture 的稳定性？
3. Swarm Embodied AI 中的 collaborative WMs 如何设计？
4. 如何评估 joint architecture 的 emergent capabilities？

这些问题的答案将决定 Embodied AI 能否真正迈向 AGI。

参考阅读：
- [World Models survey](https://arxiv.org/abs/2411.14499)
- [LeCun JEPA position paper](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [EvoAgent](https://arxiv.org/abs/2502.05907)
- [SayCan](https://say-can.github.io/)
- [RT-2](https://robotics-transformer2.github.io/)
- [Genie](https://sites.google.com/view/genie-interactive/home)
- [EmbodiedBench](https://arxiv.org/abs/2502.09560)
- [Sora as World Simulator?](https://arxiv.org/abs/2405.03520)
- [VLA Survey](https://arxiv.org/abs/2405.14093)
- [Foundation Models in Robotics](https://arxiv.org/abs/2311.04292)
