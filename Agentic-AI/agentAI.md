---
source_pdf: agentAI.pdf
paper_sha256: 62fa6655a8eecfef68bd5ae7965a84726457488b37b29a169397e85c9c0eea6e
processed_at: '2026-07-18T04:43:07-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Agent AI: Surveying the Horizons of Multimodal Interaction - 深度解析

Andrej，这是一篇来自Stanford (Li Fei-Fei lab)、Microsoft Research (Jianfeng Gao)、UCLA、UW (Yejin Choi)等机构的重磅survey paper。让我从多个维度深入剖析，帮你build intuition。

## 1. Core Thesis & Philosophical Foundation

这篇paper的核心论点建立在 **Aristotelian Holism** 之上。1956年Dartmouth Conference定义AI为"artificial life forms that could collect information from the environment and interact with it in useful ways"。Minsky的"Copy Demo" (1970)试图实现这个愿景，但AI领域随后fragmented成各个subfield。

**Agent AI的定义**：
$$\text{Agent AI} = \{S, A, \mathcal{O}, \pi\}$$

其中：
- $S$ : state space (visual stimuli, language inputs, environmental data)
- $A$ : action space (embodied actions)
- $\mathcal{O}$ : observation function
- $\pi$ : policy $\pi: S \rightarrow A$

关键insight：通过grounding agents在embodied environments中，可以mitigate LLMs的hallucinations问题，因为environment提供了天然的feedback signal。

参考链接：
- Paper: https://arxiv.org/abs/2401.03568
- Project page: https://microsoft.github.io/agentai/

## 2. Agent Transformer - 核心架构创新

### 2.1 架构对比

**传统范式** (Fig. 6 - Flamingo/BLIP-2/LLaVA范式)：
```
frozen Visual Encoder (CLIP) → Adapter → frozen LLM
```

**Agent AI范式** (Fig. 7)：
引入第三类token - **agent tokens** $\mathbf{a} \in \mathbb{R}^{N_a \times d}$

Unified input sequence：
$$\mathbf{X} = [\mathbf{x}_v^1, ..., \mathbf{x}_v^{N_v}, \mathbf{x}_l^1, ..., \mathbf{x}_l^{N_l}, \mathbf{x}_a^1, ..., \mathbf{x}_a^{N_a}]$$

其中：
- $\mathbf{x}_v^i \in \mathbb{R}^d$ : visual token (来自image patch embedding)
- $\mathbf{x}_l^j \in \mathbb{R}^d$ : language token (来自text tokenization)
- $\mathbf{x}_a^k \in \mathbb{R}^d$ : agent token (specialized for agentic behaviors)

### 2.2 Self-Attention with Agent Tokens

Standard multi-head self-attention：
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

其中：
- $\mathbf{Q} = \mathbf{X}\mathbf{W}^Q \in \mathbb{R}^{N \times d_k}$
- $\mathbf{K} = \mathbf{X}\mathbf{W}^K \in \mathbb{R}^{N \times d_k}$  
- $\mathbf{V} = \mathbf{X}\mathbf{W}^V \in \mathbb{R}^{N \times d_v}$
- $N = N_v + N_l + N_a$ (total sequence length)
- $d_k$ : key/query dimension

**关键创新**：agent tokens保留了input/output space中的dedicated subspace for agentic behaviors。对于robotics，这可以是controller的input action space；对于tool use，可以是API calls的encoding。

### 2.3 为什么这个设计重要？

**Intuition building**：

想象你在训练一个Minecraft agent。传统方法用自然语言描述动作："move forward", "turn left"。但low-level controller inputs（mouse movements, key presses）难以用自然语言表达。

Agent tokens提供了一个**learned latent space**，让model直接学习从perception到action的mapping，without forcing everything through natural language bottleneck。

这与Gato (Reed et al., 2022)的思路类似：https://arxiv.org/abs/2205.06175

## 3. Five-Module Agent Paradigm (Fig. 5)

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent AI System                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Environment │  │   Memory    │  │  Cognition  │        │
│  │ & Perception│←→│             │←→│             │        │
│  └──────┬──────┘  └─────────────┘  └──────┬──────┘        │
│         ↓                                  ↑                │
│  ┌─────────────┐                    ┌─────────────┐        │
│  │    Agent    │←──────────────────→│   Agent     │        │
│  │   Learning  │                    │   Action    │        │
│  └─────────────┘                    └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 Environment & Perception Module
- Task planning: LLM decomposes goals → subgoals
- Skill observation: 从environment中提取affordance信息

### 3.2 Memory Module
Critical for long-horizon planning。存储：
- Episodic memory: past interactions
- Semantic memory: learned knowledge
- Working memory: current task context

### 3.3 Cognition Module
Reasoning, inference, theory of mind

### 3.4 Agent Learning
Combines RL, IL, in-context learning

### 3.5 Agent Action
Output: agent tokens → executable actions

## 4. Learning Strategies - 技术深度

### 4.1 Reinforcement Learning Challenges

**Reward Design Problem**:
$$R(s, a) = \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t r_{t+1} \Big| s_0 = s, a_0 = a\right]$$

其中：
- $\gamma \in [0, 1)$ : discount factor
- $r_t$ : reward at time $t$
- $T$ : horizon

LLMs可以辅助design reward functions (Eureka: https://arxiv.org/abs/2310.12931)

**Credit Assignment Problem**:
Long-horizon tasks中，action和reward之间的temporal distance很大：
$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

其中 $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k+1}$ 是return。

**TAMP (Task and Motion Planning) Solution**:
```
High-level: LLM → subgoal sequence [g_1, g_2, ..., g_n]
Low-level:  RL policy π_i executes each g_i
```

### 4.2 Imitation Learning - Behavioral Cloning

**Standard BC objective**:
$$\mathcal{L}_{BC} = -\mathbb{E}_{(s, a) \sim \mathcal{D}_{expert}} \left[\log \pi_\theta(a|s)\right]$$

其中 $\mathcal{D}_{expert}$ 是expert demonstrations的dataset。

**RT-1/RT-2 (Google DeepMind)**:
- RT-1: https://arxiv.org/abs/2212.06817
- RT-2: https://arxiv.org/abs/2307.15818

Input: image sequence + language
Output: action sequence for base + arm

**Decoupling → Generalization**:

Paper提出了一个重要的conceptual framework：

1. **Imitation Learning → Decoupling**: 不直接学习expert policy，而是学习implicit reward function
2. **Decoupling → Generalization**: 与task-specific reward function解耦，policy可以跨任务transfer
3. **Generalization → Emergent Behavior**: 简单rules的组合产生复杂behaviors

### 4.3 In-Context Learning for Agents

Few-shot prompt with environmental feedback:
$$P(a_t | s_t, \mathcal{E}_1, ..., \mathcal{E}_k)$$

其中 $\mathcal{E}_i = (s_i, a_i, r_i)$ 是in-context examples。

## 5. Gaming Applications - 详细实验

### 5.1 GPT-4V for Gaming Action Prediction

**实验设置**：
- Games: Minecraft, Bleeding Edge, Microsoft Flight Simulator, Assassin's Creed Odyssey, Gears of War 4, Starfield
- Input: video frames as grid (up to 48 frames)
- Output: high-level action description

**Key finding**: GPT-4V可以predict high-level actions，但对low-level actions（mouse movements）需要supplemental modules。

**Minecraft Example (Fig. 9)**:
```
Input: action history + gaming target + current frame
Output: "The player is holding wooden logs. 
         Next action: craft planks, then build shelter."
```

### 5.2 Small Agent Pretraining Model

**Architecture**: 250M parameters
- Training data: 78K Minecraft videos (5K for first round, 6% of pretraining data)
- Hardware: 16 NVIDIA V100 GPUs, 1 day training
- Task: masked video prediction + action prediction

**Masked Video Prediction** (Fig. 10):
$$\hat{\mathbf{X}}_{masked} = f_\theta(\mathbf{X}_{visible})$$

Loss:
$$\mathcal{L} = \|\mathbf{X}_{masked} - \hat{\mathbf{X}}_{masked}\|_2^2$$

### 5.3 CuisineWorld - Multi-Agent Benchmark

**Collaboration Score (CoS)**:
$$\text{CoS} = \frac{\text{Tasks completed}}{\text{Theoretical optimal time}} \times \text{Collaboration efficiency}$$

这个metric量化multi-agent collaboration的效率。

参考: https://arxiv.org/abs/2309.09971

## 6. Robotics - 深入技术细节

### 6.1 Vision-Language Navigation (VLN)

**Reinforced Cross-Modal Matching (RCM)** (Wang et al., 2019):
https://arxiv.org/abs/1910.03763

**Architecture**:
```
Instruction → Reasoning Navigator → Action
                ↑                    ↓
                ←── Matching Critic ←─
```

**Reasoning Navigator**:
$$\alpha_t = \text{softmax}(\mathbf{W}_a [\mathbf{h}_t; \mathbf{v}_t])$$
$$\mathbf{c}_t = \sum_i \alpha_{t,i} \mathbf{v}_{t,i}$$
$$a_t = \pi_\theta(\mathbf{h}_t, \mathbf{c}_t)$$

其中：
- $\mathbf{h}_t$ : hidden state at step $t$
- $\mathbf{v}_t$ : visual features
- $\alpha_t$ : cross-modal attention weights
- $\mathbf{c}_t$ : context vector

**Matching Critic** provides intrinsic reward:
$$r_{intrinsic} = \text{sim}(\text{instruction}, \text{trajectory})$$

**Results**: 10% improvement on SPL (Success weighted by Path Length)
$$\text{SPL} = \frac{1}{N} \sum_{i=1}^{N} S_i \cdot \frac{\ell_i^*}{\max(\ell_i, \ell_i^*)}$$

其中：
- $S_i \in \{0, 1\}$ : success indicator
- $\ell_i$ : actual path length
- $\ell_i^*$ : shortest path length

### 6.2 Self-Supervised Imitation Learning (SIL)

**Generalization gap reduction**:
- Before SIL: 30.7% gap between seen/unseen environments
- After SIL: 11.7% gap

**Algorithm**:
```
1. Agent explores unseen environment
2. Identifies "good" decisions (high confidence)
3. Adds these to training set
4. Retrains policy
5. Repeat
```

### 6.3 GPT-4V for Robot Manipulation

**Pipeline** (Fig. 15):
```
Video + Text → Vision Analyzer (GPT-4V) → Text instructions
            → Scene Analyzer (GPT-4V) → Object list + spatial relations
            → Task Planner (GPT-4) → Action sequence
```

**Scene Analyzer Output Example**:
```json
{
  "objects": ["table", "can", "tray"],
  "graspable": ["can"],
  "spatial_relations": {
    "can": "on table",
    "tray": "nearby table"
  }
}
```

**Affordance Extraction**:
- Waypoints for collision avoidance
- Grasp types (pinch, power, precision)
- Upper-limb postures

参考: https://microsoft.github.io/GPT4Vision-Robot-Manipulation-Prompts

## 7. Healthcare Applications

### 7.1 Current Capabilities (Fig. 19-20)

**GPT-4V Healthcare Experiments**:
1. CT scan image understanding: ✓ (describes equipment, procedures)
2. EKG scan analysis: limited (safety training restricts)
3. Skin lesion (ISIC dataset): limited diagnostic
4. Clinical bedside activity detection: ✓ (from video)
5. Echocardiogram assessment: limited (safety)

**Key limitation**: Safety training prevents diagnostic outputs，但model possess significant medical knowledge。

### 7.2 Knowledge Retrieval for Hallucination Reduction

**Retrieve-then-Generate**:
$$P(y|x) = \sum_{k \in \mathcal{R}(x)} P(y|x, k) \cdot P(k|x)$$

其中：
- $\mathcal{R}(x)$ : retrieval function
- $k$ : retrieved knowledge
- $y$ : generated response

参考: https://arxiv.org/abs/2302.12813

## 8. Multimodal Video Understanding

### 8.1 Video-Language Architecture

**InstructBLIP extension for video**:
```
Video frames → EVA-CLIP-G (frozen spatial layers)
            → Temporal attention (trainable)
            → Q-former (frozen)
            → Flan-T5-XL (frozen)
            → Caption/Answer
```

**Divided Space-Time Attention** (from Frozen in Time):
$$\text{Attention}_{temporal}(\mathbf{Q}_t, \mathbf{K}_t, \mathbf{V}_t) \rightarrow \text{Attention}_{spatial}(\mathbf{Q}_s, \mathbf{K}_s, \mathbf{V}_s)$$

训练数据：5M video-caption pairs from WebVid10M
参考: https://arxiv.org/abs/2104.00650

### 8.2 Audio-Video-Language Integration (Fig. 26)

**Pipeline**:
```
Video frames → GPT-4V → Frame captions
Audio → Whisper → Transcriptions
(Frame captions + Audio transcriptions) → GPT-4V → Video summary
```

**Key finding**: Audio modality significantly reduces visual hallucinations。

Example:
- Without audio: "in a playful twist, he bites down on it while holding it horizontally" (HALLUCINATION)
- With audio: "holding the broomstick perpendicular to the body and rotating it downwards" (CORRECT)

### 8.3 VideoAnalytica Benchmark

**Tasks**:
1. Video Text Retrieval (with hard negatives from LLMs)
2. Video Assisted Informative Question Answering

**Hard Negative Generation**:
```
Primary query: "Measuring tree height with broomstick"
Negative 1: "Measuring tree width with broomstick" (semantic modification)
Negative 2: "Measuring tree height with ruler" (object swap)
Negative 3: "Cutting tree with broomstick" (action change)
```

## 9. KAT - Knowledge Augmented Transformer

### 9.1 Architecture (Fig. 23)

```
Image → Visual Encoder → Visual features ─┐
                                          ↓
Question → Text Encoder → Text features → Fusion → Decoder → Answer
                                          ↑
Knowledge → ┌── GPT-3 (implicit) ────────┤
             └── Web retrieval (explicit) ┘
```

### 9.2 Knowledge Integration

**Implicit knowledge** (from GPT-3):
$$\mathbf{k}_{implicit} = \text{GPT-3}(\text{question}, \text{context})$$

**Explicit knowledge** (from web):
$$\mathbf{k}_{explicit} = \text{retrieve}(\text{question}, \mathcal{K}_{web})$$

**Contrastive retrieval loss**:
$$\mathcal{L}_{contrastive} = -\log \frac{\exp(\text{sim}(\mathbf{q}, \mathbf{k}^+))}{\sum_{\mathbf{k}^-} \exp(\text{sim}(\mathbf{q}, \mathbf{k}^-))}$$

其中：
- $\mathbf{q}$ : question embedding
- $\mathbf{k}^+$ : positive knowledge
- $\mathbf{k}^-$ : negative knowledge

**Results**: 6% improvement on OK-VQA (https://arxiv.org/abs/1906.00067)

参考: https://github.com/guilk/KAT

## 10. VLC - Vision-Language Transformer from Captions

### 10.1 Architecture (Fig. 24)

```
Image → Patch Embedding (linear) → ┐
Text → Token Embedding →           ┤→ ViT (12 layers, MAE init) → MLM + ITM
                                    └→ MLP (fine-tuning) → Downstream tasks
```

**Key insight**: 使用simple linear projection代替object detector，用image-caption pairs预训练。

### 10.2 Pretraining Objectives

**Masked Image Modeling (MIM)**:
$$\mathcal{L}_{MIM} = \mathbb{E}_{\mathbf{x} \sim D} \left[\|\mathbf{x}_{masked} - \hat{\mathbf{x}}_{masked}\|^2\right]$$

**Image-Text Matching (ITM)**:
$$\mathcal{L}_{ITM} = -\mathbb{E}\left[y \log p(\text{match}) + (1-y) \log(1 - p(\text{match}))\right]$$

其中 $y \in \{0, 1\}$ 表示image-text pair是否匹配。

参考: https://arxiv.org/abs/2206.04115

## 11. Logic Transformer for NLP

### 11.1 Logic-Aware Embeddings (Fig. 29)

**Process**:
1. **Logic Detection**: 识别text中的logical structures
2. **Logic Mapping**: 映射到logical roles和types
3. **Hierarchical Logical Projections**: 多层projection注入embeddings

**Logic Embedding Constraint**:
$$\mathbf{e}_{i,j}^{logic} = \begin{cases} 
\mathbf{v}_{role_i, type_j} & \text{if same color (same role+type)} \\
\text{different vector} & \text{if different color}
\end{cases}$$

其中：
- $role_i$ : logical role (premise, conclusion, etc.)
- $type_j$ : logical type (conditional, causal, etc.)
- $\mathbf{v}_{role_i, type_j}$ : shared embedding for same role+type

参考: https://arxiv.org/abs/2305.07144

## 12. Cross-Reality & Sim-to-Real Transfer

### 12.1 Domain Randomization

**Training distribution**:
$$p_{train}(\xi) = \prod_{i} p(\xi_i)$$

其中 $\xi_i$ 是simulation parameter (lighting, texture, physics)。

目标：policy在real world的分布 $p_{real}$ 下表现良好。

### 12.2 Domain Adaptation with CycleGAN

**Cycle consistency loss**:
$$\mathcal{L}_{cycle} = \mathbb{E}_x[\|G_F(G_B(x)) - x\|_1] + \mathbb{E}_y[\|G_B(G_F(y)) - y\|_1]$$

其中：
- $G_F$ : forward generator (sim → real)
- $G_B$ : backward generator (real → sim)
- $x$ : sim image
- $y$ : real image

参考: https://arxiv.org/abs/1703.10593

### 12.3 RL-CycleGAN for Reinforcement Learning

**Modified objective**:
$$\mathcal{L}_{RL-CycleGAN} = \mathcal{L}_{GAN} + \lambda_{cycle} \mathcal{L}_{cycle} + \lambda_{task} \mathcal{L}_{task}$$

其中 $\mathcal{L}_{task}$ 是RL task performance loss。

参考: https://arxiv.org/abs/2006.09001

## 13. Continuous Self-Improvement

### 13.1 Human-Based Interaction Data

**Three strategies**:
1. **Additional training data**: Filter successful interactions
2. **Human preference learning**: RLHF
3. **Safety training (red-teaming)**: Adversarial testing

### 13.2 Foundation Model Generated Data

**Knowledge Distillation Pipeline**:
```
Large LLM (teacher) → Generate instructions → Small LLM (student) fine-tune
```

**Alpaca example** (https://arxiv.org/abs/2303.09500):
- 175 seed tasks → 52K instruction-following examples
- Fine-tune LLaMA 7B → Alpaca 7B

**Limitation**: Student models have significant capability gaps vs teacher (https://arxiv.org/abs/2305.15717)

## 14. Experimental Results Summary

### 14.1 Gaming Results

| Game | GPT-4V Capability | Low-level Action | High-level Action |
|------|-------------------|------------------|-------------------|
| Minecraft | Excellent | Limited | Good |
| Bleeding Edge | Moderate | Limited | Moderate |
| Flight Simulator | Good | Limited | Good |
| AC Odyssey | Good | Limited | Good |
| Gears of War 4 | Good | Limited | Good |
| Starfield | Good | Limited | Good |

### 14.2 Robotics Results

| Task | Method | SPL | Success Rate (Seen) | Success Rate (Unseen) |
|------|--------|-----|---------------------|----------------------|
| VLN | Baseline | Low | High | Low |
| VLN | RCM | +10% | High | Higher |
| VLN | RCM + SIL | +10% | High | High (gap 11.7%) |

### 14.3 Video Understanding

| Method | Modality | Hallucination Rate | Detail Level |
|--------|----------|-------------------|--------------|
| Video-instruction tuned | Video only | High | Low |
| GPT-4V + Frame captions | Video + Text | Medium | Medium |
| GPT-4V + Frame + Audio | Video + Text + Audio | Low | High |

## 15. Key Insights & Intuition Building

### 15.1 Why Agent Tokens Matter

**Traditional multimodal models** force everything through natural language：
$$\text{Perception} \rightarrow \text{Language} \rightarrow \text{Action}$$

**Agent Transformer** allows direct perception-action mapping：
$$\text{Perception} \rightarrow \text{Agent Tokens} \rightarrow \text{Action}$$

这解除了language bottleneck，特别适合难以verbalize的动作（mouse movements, continuous control）。

### 15.2 Emergent Mechanism: Mixed Reality with Knowledge Inference

**Micro-reactions** (cross-modality):
- Collect relevant individual knowledge from web
- Implicit inference from pretrained models

**Macro-behavior** (reality-agnostic):
- Improve interactive dimensions in language + multimodality
- Adapt based on characterized roles

### 15.3 The Hallucination Solution

Paper的核心claim：**embodied grounding reduces hallucinations**。

**Mechanism**:
1. Environment provides immediate feedback
2. Actions have consequences (success/failure)
3. Policy must be physically consistent
4. Hallucinated actions fail → negative reward → policy correction

这与pure language models不同，后者没有environmental consistency check。

### 15.4 Why Cross-Reality Training?

**Data scarcity problem**:
- Real world data: expensive, slow, limited
- Simulated data: cheap, fast, unlimited
- Virtual worlds (games): structured, diverse

**Solution**: Train on cross-reality data to learn reality-agnostic representations。

$$\mathcal{D}_{train} = \mathcal{D}_{real} \cup \mathcal{D}_{sim} \cup \mathcal{D}_{virtual}$$

## 16. Future Directions & Open Problems

### 16.1 Continuous Learning
Current agents are largely tied to pretrained foundation models。Future: agents that continuously learn from environment interactions。

参考: RoboCat (https://arxiv.org/abs/2306.11706)

### 16.2 Neuro-Symbolic Integration
**TP-N2F approach**:
- TPR binding: encode symbolic structure in vector space
- TPR unbinding: generate sequential programs

$$\mathbf{v}_{bound} = \sum_i \mathbf{r}_i \otimes \mathbf{f}_i$$

其中 $\otimes$ 是tensor product binding, $\mathbf{r}_i$ 是role vector, $\mathbf{f}_i$ 是filler vector。

### 16.3 Emotional Reasoning
**NICE Dataset** (https://arxiv.org/abs/2104.09527):
- 2M images + comments + emotion annotations
- MAGIC model for empathetic image commenting

### 16.4 Multi-Agent Conventions
**Diversity in conventions** (https://arxiv.org/abs/2310.14295):
- Self-play → brittle agents
- Solution: discover diverse conventions
- Train agents aware of wide range of conventions

## 17. Critical Analysis

### 17.1 Strengths
1. **Comprehensive taxonomy**: Covers gaming, robotics, healthcare, multimodal
2. **Practical frameworks**: Agent Transformer, MindAgent, RCM
3. **Real implementations**: GPT-4V experiments across multiple games
4. **New benchmarks**: CuisineWorld, VideoAnalytica

### 17.2 Limitations
1. **Hallucination claim**: 更多是hypothesis而非proven fact
2. **Scalability**: 250M parameter model vs GPT-4 scale
3. **Evaluation**: 多为qualitative results
4. **Sim-to-real**: 仍是open problem

### 17.3 Missing Pieces
1. **Memory architecture**: 没有详细specify memory module
2. **Cognition module**: 理论性描述多于implementation
3. **Agent token design**: 如何assign unique tokens没有详述

## 18. Connection to Broader AGI Discussion

Paper将Agent AI定位为towards AGI的route：
- **Holistic approach**: 整合multiple modalities + reasoning + action
- **Reality-agnostic training**: Cross-domain generalization
- **Emergent abilities**: 从simple rules到complex behaviors

这与你（Karpathy）在Tesla的工作和"Software 2.0"理念有resonance：learned policies取代hand-coded rules，multimodal perception直接mapping到actions。

参考: 
- Software 2.0: https://karpathy.medium.com/software-2-0-a64152b37c35
- AGI discussion: https://arxiv.org/abs/2303.12712

---

这篇paper最重要的contribution在于：**提供了agent AI的systematic framework**，将分散的研究方向（gaming AI, robotics, multimodal understanding）统一在agent paradigm下。Agent tokens的概念尤其有insight，为future multimodal agent设计提供了new direction。

希望这个deep dive帮你build了intuition，Andrej！如果你想dive deeper into某个specific section（比如RCM的数学细节，或者GPT-4V prompting strategies），我可以继续展开。
