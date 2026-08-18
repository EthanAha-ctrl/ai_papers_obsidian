---
source_pdf: DREAMGEN Unlocking Generalization in.pdf
paper_sha256: e4e2c5c9e664d9331adb49c230053a76cf5bb4c3501341f4d9b372ecc585684c
processed_at: '2026-08-18T06:48:12-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 DREAMGEN

## One-liner

Robot learning 的 data problem，被 video generation models 给绕过去了。

## The Problem

Robot learning 的瓶颈就一个字：**data**。

你想让 robot 干新活，就得找人 teleop 采集。一个 task 采集几百条 trajectory，换到新 environment 又得重来。这东西又贵又慢，还 scale 不起来。

Simulation 看似能解决，但是 sim2real gap 太烦人。液体怎么 sim？毛巾怎么 sim？articulated objects 怎么 sim？搞这些的工程量比采 data 还大。

## The Core Idea

这帮人的 insight 很 simple：**video generation models 在 internet 上看了海量视频，它们已经 "懂" 物理了，懂物体怎么动，懂 "pour" 长什么样，懂 kitchen 长什么样**。

我们只需要：
1. 在 robot 自己的 teleop data 上 fine-tune 一下，让 video model 学会这个 robot 的 kinematics
2. 给它一张 initial frame + 一句 instruction，让它 "dream" 出一段 robot 干活的 video
3. 从 video 里反推出 actions（因为没有 action labels）
4. 拿这些去训 policy

就这样。Four steps，没有任何 fancy 的 architecture innovation，pure pipeline engineering。

## Why It Works

关键在于 **composition**。

Video model 有两个 knowledge source：
- **Internet priors**：见过无数人倒水、用锤子、叠衣服的视频，知道这些 action 长什么样
- **Fine-tuned kinematics**：在 GR1 的 teleop data 上 adapt 过，知道这个 robot 的 arm 怎么动

当 you prompt 它 "pour water from pitcher to cup"，它会 combine 两者：用 internet knowledge 知道 pouring 的 visual pattern，用 fine-tuned knowledge 让 robot arm 做出 plausible 的 motion。

这就是为什么 **behavior generalization** 能 work：training data 里只有 pick&place，但 video model 见过 pouring，所以它能 dream 出 GR1 做 pouring 的视频。

也是为什么 **environment generalization** 能 work：video model 在 internet 上见过各种 environment，给它一个新 kitchen 的 initial frame，它能 imagine 出 GR1 在那儿干活的样子。

## The Results That Matter

**Simulation (RoboCasa)**：
- 光用 neural trajectories（零 real data）就达到 20.55% success rate
- 加上 real data co-training，从 49.6% → 57.6%
- **Log-linear scaling**：neural trajectories 越多，效果越好，没有饱和迹象

**Real-world GR1 (humanoid)**：
- 4 个 dexterous tasks（hammering, wiping, folding, stacking）
- 只用 10 条 real trajectories + neural trajectories
- GR00T N1：37% → 46% average success rate

**Behavior Generalization（最 wild 的结果）**：
- 只在 pick&place 上训 video model
- 生成 14 个完全 novel behaviors 的 neural trajectories
- 只用 neural trajectories 训 policy
- **11.8% → 43.2%**

**Environment Generalization**：
- Video model 只见过 1 个 lab environment
- Prompt 10 个新环境的 initial frames
- Baseline 0% → DREAMGEN 28.5%

这俩 generalization 结果是 zero-to-one 的 improvement。Baseline 根本干不了，DREAMGEN 能干。

## The DreamGen Bench

他们还搞了个 benchmark，测 video models 能不能 good robot videos。

两个 metrics：
- **Instruction Following**：video 有没有 follow instruction
- **Physics Alignment**：物理合不合理

关键发现：**benchmark score 和 downstream policy performance 正相关**。

这意味着 video model researchers 可以不碰 robot 就能 contribute to robotics。你把 video model 在 DreamGen Bench 上刷高了，robot policy 就会变强。这是一个很 nice 的 bridge。

## My Take

这个 paper 的核心 message 就一句：**robot data 的问题，可以靠 video generation 来 solve**。

之前 robot learning 的 paradigm 是：每个新 task + 新 environment → 人去 teleop 采集。这是 O(n) 的人力 cost。

DREAMGEN 说：你只需要一个 task 的 teleop data 来 bootstrap，剩下的让 video model 去 dream。Data bottleneck 从 "physical collection" 变成了 "GPU hours"。

几个让我兴奋的点：

**1. Scaling law 出现了**。Log-linear relationship 意味着只要 video models 继续变强（这是必然的），robot learning 会跟着变强。这跟 LLM 的 scaling law 是一个 flavor，但这里是 data scaling 而非 model scaling。

**2. Zero-to-one generalization**。从 0% 到 28.5% 的 environment generalization，从 11.8% 到 43.2% 的 behavior generalization。这些不是 incremental improvement，是 enabling previously impossible capabilities。

**3. Video model = robot data generator**。这个 framing很重要。之前 video models 在 robotics 里被当作 real-time planner（test-time 生成 video 然后 execute），那个 setup 太 constrained。DREAMGEN 把 video model 当作 offline data generator，解耦了 generation 和 execution，让两边各自 scale。

**4. The bottleneck shifts**。Paper 在 Appendix 里提到，bottleneck 在 neural trajectory 质量而非 IDM。这意味着 video model 的进步会直接 translate 到 robot learning 的进步。两个 field 的进步耦合在一起了。

**Concerns**：

Real-world success rates 还不高。最好的 GR1 high-data 也才 69%。还有很大空间。

Compute cost 很贵。240k samples 要 54 hours on 1500 L40 GPUs。这不是普通 lab 能 afford 的。

Initial frames 还得手动拍。这是 pipeline 里 last remaining 的 human bottleneck。如果能把这步也 automate（比如用 image generation models 生成 initial frames），就完全 closed loop 了。

## The Bigger Picture

DREAMGEN 代表的是 robot learning 的一个新 paradigm：**learning from imagined experience**。

Human child 学会新东西，不完全靠亲身体验。他们看 video、看书、听故事，也能学到很多。DREAMGEN 给 robot 提供了类似的 capability：通过 video model 的 "imagination" 来学习，而不完全靠 physical practice。

如果 video models 按照 current trajectory 继续进步（而且很可能比 robot hardware 进步更快），robot learning 的 main bottleneck 会从 data collection 彻底转移到 policy architecture 和 evaluation methodology。

那个时候，robot learning 就真的变成 "Software 2.0" 了。

---

# DREAMGEN: 用 Video World Models 解锁 Robot Learning 的 Generalization

## 核心直觉：为什么这个工作重要

Karpathy 你会喜欢这篇 paper 的地方在于，它直接攻击了 robot learning 的核心瓶颈：**data scaling**。传统 paradigm 是 human teleoperation 逐任务逐环境采集，成本高且不可扩展。Simulation 可以 scale 但有 sim2real gap，尤其对 deformable objects（毛巾、液体）和 articulated objects 几乎无解。

DREAMGEN 的 key insight 很 clean：**video world models 在 internet-scale 视频上预训练时，已经学到了 physical reasoning、naturalistic motion 和 language grounding 的强大 priors**。我们只需要让它们适应特定 robot embodiment 的 kinematics，就可以让它们"dream"出大量 photorealistic 的 robot videos，然后从中 recover 出 pseudo-actions 来训练 policy。

这是一个从 "data collection" 到 "data generation" 的 paradigm shift。

---

## 4-Stage Pipeline 详解

### Stage 1: Video World Model Fine-tuning

**Base model**: 主要用 WAN2.1，也测了 Hunyuan、CogVideoX、Cosmos

**Fine-tuning 策略**：
- **LoRA** (Low-Rank Adaptation) rank=4, alpha=4, lr=1e-4
  - LoRA 的核心公式：$W = W_0 + \Delta W = W_0 + BA$，其中 $W_0 \in \mathbb{R}^{d \times k}$ 是 frozen 的 original weight，$B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ 是 learnable low-rank matrices，$r \ll \min(d, k)$
  - 用 LoRA 而非 full fine-tuning 的原因：mitigate catastrophic forgetting of internet video knowledge
- **Multi-view 处理**：对于 RoboCasa 和 DROID 这种多视角数据集，将 left/right/wrist camera views 拼成 2×2 grid（bottom-right 放 black image）。这是一个简单但有效的 trick，让 single-view video model 也能 handle multi-view data
- **Training config**：
  - RoboCasa: 100 epochs, batch size 32
  - GR1: 75 epochs, batch size 64
  - DROID: 5 epochs, batch size 64
  - SO-100: 200 epochs, batch size 8

**Key observation**: 每个 video world model + fine-tuning data pair 的 optimal fine-tuning amount 不同。这暗示了 embodiment complexity 和 data distribution 的匹配问题。

### Stage 2: Video World Model Rollout

给定 (initial frame, language instruction)，模型生成 video rollouts。

**关键设计 choices**：
1. **Initial frames 来源**：
   - Simulation: 从 simulator 采集，randomize object locations
   - Real-world: 手动拍摄，randomize target object location
   - Environment generalization: 拍摄 new environments 的 initial frames
   
2. **Behavior generalization**: 手动构造 novel behavior prompts（pouring, opening/closing articulated objects, tool manipulation 等）

3. **关键约束**: 对于 environment generalization，video world model 只在 single environment 上 fine-tune，但用 new environments 的 initial frames 来 prompt。这是真正的 zero-shot transfer。

### Stage 3: Pseudo Action Labeling（最技术性的部分）

这是 paper 的核心技术贡献。Video 只有 pixels，没有 actions，需要 recover 出 executable action sequences。

#### Option A: Inverse Dynamics Model (IDM)

**架构**: Diffusion Transformer + SigLIP-2 vision encoder + flow matching objective

**输入**: 两个 image frames $(o_t, o_{t+H})$
**输出**: action chunk $\hat{a}_{t:t+H}$

**为什么不加 language 和 proprioception**？因为想让 IDM 只 capture robot dynamics，不依赖 task context。这让 IDM 更通用。

**Flow Matching Objective**（核心公式）：
$$\mathcal{L}_{FM} = \mathbb{E}_{t, \epsilon, x_0, x_1} \| v_\theta(x_t, t) - (x_1 - x_0) \|^2$$

其中：
- $x_0 \sim \mathcal{N}(0, I)$ 是 noise sample
- $x_1$ 是 ground-truth action chunk
- $t \sim \mathcal{U}(0, 1)$ 是 time step
- $x_t = (1-t)x_0 + t x_1$ 是 interpolation
- $v_\theta$ 是 learnable velocity field（diffusion transformer）
- $\epsilon$ 是 conditioning（两个 frames的 SigLIP-2 embeddings）

**Sliding Window Inference**（重要的工程细节）：
1. IDM 预测 $\hat{a}_t$ 到 $\hat{a}_{t+H}$（H 个 actions）
2. Slide 一个 window，预测 $\hat{a}_{t+1}$ 到 $\hat{a}_{t+1+H}$
3. 重复直到覆盖整个 video

这会产生 overlapping predictions，但 paper 没有明确说如何 aggregate（可能取 average 或只取 non-overlapping 部分）。

#### Option B: Latent Action Model (LAPA)

**架构**: Transformer encoder-decoder + VQ-VAE objective

**训练数据**: Table 3 显示了大规模混合数据：
- Real robot: GR-1 (88.4h), DROID (428.3h), RT-1 (338.4h), Bridge-v2 (111.1h), Agibot-Alpha (1979.4h)
- Simulation: DexMG (61.64h), RoboCasa (268h)
- Human: Ego4D (2144.7h), Sth-v2 (105.7h)
- **Total**: 438.1M frames, 5721.3 hours

**VQ-VAE Objective**：
$$\mathcal{L}_{VQ} = \| x - D(z_q) \|^2 + \| \text{sg}[E(x)] - z_q \|^2 + \beta \| E(x) - \text{sg}[z_q] \|^2$$

其中：
- $x$ 是 visual delta（current frame 和 future frame 之间的信息）
- $E$ 是 encoder
- $D$ 是 decoder  
- $z_q$ 是 quantized latent（从 codebook 中 lookup）
- $\text{sg}$ 是 stop-gradient operator
- Codebook size = 8, sequence length = 16

**关键 insight**: LAPA 使用 pre-quantized continuous embedding（不是离散的 codebook indices），这遵循 GR00T N1 的设计。

**LAPA 的核心优势**: 不需要 target robot 的 ground-truth actions！只需要 visual frames 就能 extract latent actions。这意味着可以用于完全没有 action labels 的 embodiment。

#### IDM vs LAPA 对比

| 特性 | IDM | LAPA |
|------|-----|------|
| 需要 GT actions | 是（target robot） | 否 |
| Action space | Robot-specific | Embodiment-agnostic |
| Training data | Same as video world model | Massive cross-embodiment mixture |
| Architecture | Diffusion Transformer | Transformer encoder-decoder + VQ |
| Vision encoder | SigLIP-2 | - |
| Training objective | Flow matching | VQ-VAE |
| Inference | Sliding window | Single forward pass |
| Default in paper | 是 | 否 |

**Paper 的结论**: IDM 是 default，因为：(1) 可以 solely 训练在 neural trajectories 上，(2) 在所有实验中都有 teleoperation data 训练 strong IDMs。

### Stage 4: Policy Training on Neural Trajectories

**输入**: $o_t$ (image observation), $i_t$ (task instruction)
**输出**: $\hat{a}_{t:t+H}$ (latent actions 或 IDM-labeled actions)

**关键设计**: State information 用 zeros conditioning，因为 neural trajectories 没有 state info。

**三种 policy architectures 测试**:
1. **Diffusion Policy** (Chi et al., 2023)
2. **π₀** (Physical Intelligence, 2024)
3. **GR00T N1** (NVIDIA, 2025)

**Co-training 策略**:
- Neural trajectories 和 real trajectories 以 1:1 ratio sampling
- 对 GR00T N1，两种 trajectories 用 separate action encoder 和 decoder（treat as different embodiments）
- 这解释了为什么 GR00T N1 + Neural Traj 提升更大：separate parameters 缓解了 neural trajectories 用 0 state 的问题

---

## 关键实验结果

### Simulation: RoboCasa Scaling Law

**Figure 4 的核心发现**：

| Ground-truth data | Baseline | + Neural Traj (333x) | Gain |
|-------------------|----------|----------------------|------|
| Low (720) | ~17.4% | ~39.9% | +22.5% |
| Mid (2.4k) | ~32.1% | ~57.6% | +25.5% |
| High (7.2k) | ~49.6% | ~57.6% | +8.0% |

**Log-linear scaling**: Policy performance 与 neural trajectories 数量呈 log-linear 关系。这是一个 exciting 的发现，类似于 LLM 的 scaling law，但这里是 data scaling 而非 model scaling。

**ONLY Neural Trajectories**: 仅用 240k neural trajectories（无任何 real data）达到 **20.55%** average success rate across 24 tasks！这说明 neural trajectories 质量已经相当接近 ground truth。

### Real-World: Three Embodiments

**GR1 Humanoid** (4 tasks: Hammering, Wiping, Folding, Stacking):
| Model | Baseline | + Neural Traj | High Data |
|-------|----------|---------------|-----------|
| DP | 22.0% | 27.0% | 54.0% |
| GR00T N1 | 37.0% | **46.0%** | 69.0% |

**Franka** (3 tasks: Pick&Place, Cube Stacking, Tool Use):
| Model | Baseline | + Neural Traj |
|-------|----------|---------------|
| DP | 10.0% avg | 20.0% avg |
| π₀ | 20.0% avg | 26.7% avg |
| GR00T N1 | 23.3% avg | **36.7%% avg** |

**SO-100** (2 tasks: Strawberry picking, Tic-Tac-Toe):
| Model | Baseline | + Neural Traj |
|-------|----------|---------------|
| GR00T N1 | 21.0% avg | **45.5% avg** |

**关键 insight**: GR00T N1 的提升最显著，paper 假设是 separate action/decoder parameters for IDM actions 缓解了 0-state conditioning 问题。

### Behavior Generalization（最 impressive 的结果）

**Setup**: 只用 2,884 条 GR1 pick-and-place trajectories 训练 video world model，然后生成 14 个 novel behaviors 的 neural trajectories，只用这些训练 policy。

**Table 1 结果**:
| Setting | Baseline (GR00T N1) | + Neural Traj Only |
|---------|---------------------|---------------------|
| 14 New Behaviors, Seen Env | 11.8% | **43.2%** |

14 个 novel behaviors 包括：Pour Water, Water Flowers, Light Candle, Use Vacuum, Iron Shirt, Open Microwave, Hit Tambourine, Move Mouse 等等。这些 verbs 完全不在 training data 里！

### Environment Generalization

**Setup**: Video world model 只在 1 个 lab environment 训练，用 10 个 new environments 的 initial frames 来 prompt。

**Table 1 结果**:
| Setting | Baseline | + Neural Traj Only |
|---------|----------|---------------------|
| 6 Seen Behaviors, New Env | 0% | 28.5% |
| 7 New Behaviors, New Env | 0% | 28.5% |

Baseline 是 0% 因为 GR00T N1 只见过一个环境，完全无法 generalize。DREAMGEN 通过 video world model 的 internet priors 实现了 zero-shot environment transfer。

---

## DreamGen Bench: Video World Model Benchmark

### 设计动机
想要一个 low-cost diagnostic tool，不需要 physical robot 就能评估 video world models 对 robotics 的适用性。

### 两个核心 Metrics

**1. Instruction Following (IF)**:
- 用 Qwen2.5-VL-7B-Instruct 判断 video 是否遵循 instruction
- Binary score (0 or 1)
- 也用 GPT-4o 和 human evaluation 对比
- Pearson correlation > 90% with human judgment

**2. Physics Alignment (PA)**:
- 用 VideoCon-Physics（专门训练评估物理一致性的 VLM）
- 也用 Qwen2.5-VL 作为补充
- Average of two scores

### Benchmark Results (Table 2)

8 个 models 测试（4 zero-shot + 4 fine-tuned）：

| Model | RoboCasa IF | GR1-Object IF | GR1-Behavior IF | GR1-Env IF |
|-------|-------------|----------------|------------------|-------------|
| Hunyuan-zero | 1.0 | 0.0 | 2.1 | 0.0 |
| CogVideoX-zero | 0.0 | 0.0 | 0.0 | 0.0 |
| WAN2.1-zero | 0.0 | 2.0 | 2.1 | 6.7 |
| Cosmos-zero | 22.9 | 32.0 | 31.9 | 24.1 |
| Hunyuan-sft | 81.3 | 52.0 | 14.9 | 35.4 |
| CogVideoX-sft | 79.2 | 72.0 | 21.3 | 51.3 |
| WAN2.1-sft | **91.7** | 80.0 | **74.5** | 66.5 |
| Cosmos-sft | **93.8** | **84.0** | 68.1 | **59.4** |

**Key findings**:
1. Zero-shot models 基本完全 fail（除了 Cosmos 稍好）
2. Fine-tuning 带来巨大提升
3. **WAN2.1-sft 和 Cosmos-sft 表现最好**
4. Behavior generalization 最难（即使是 best model 也只有 74.5%）
5. Environment generalization 相对容易（Cosmos-sft: 59.4%）

### Correlation with Downstream Policy Performance

**Figure 6 的核心发现**: DreamGen Bench score 与 RoboCasa policy success rate 呈正相关。

这意味着：**更强的 video world model → 更好的 neural trajectories → 更强的 downstream robot policy**。

这为 video model researchers 提供了一个 accessible pathway：不需要 physical robot，只需优化 DreamGen Bench score，就能间接贡献 robot learning。

---

## 为什么这个方法 Work？构建直觉

### 1. Video World Models 的 Priors

Internet-scale video pretraining 让模型学到了：
- **Physical reasoning**: 物体如何运动、重力、碰撞
- **Object affordances**: 物体可以被如何 manipulate
- **Naturalistic motion**: 人类和机器人的自然运动模式
- **Language grounding**: "pour" 对应什么视觉变化

### 2. Fine-tuning 的作用

Fine-tuning 在 robot trajectories 上让模型：
- 学习 specific embodiment 的 kinematics（关节角度范围、运动模式）
- 适应 camera viewpoint 和 visual appearance
- 但通过 LoRA 保留了 internet priors

### 3. Generalization 的来源

**Behavior generalization** 之所以 work：
- Video world model 见过 internet 上无数 pouring, hammering, folding 的视频
- Fine-tuning 教会它 GR1 的 kinematics
- 结合两者，模型可以生成 GR1 执行 novel behaviors 的视频

**Environment generalization** 之所以 work：
- Initial frame 提供了 new environment 的 visual context
- Video world model 的 internet priors 知道 kitchens, labs, outdoor 长什么样
- Fine-tuning 的 kinematics 约束让 robot motion 保持一致

### 4. Neural Trajectories 为什么有效

Paper 在 Appendix A 提到一个重要观察：**bottleneck 在 neural trajectories 质量，而非 IDM**。他们在 simulation 中 replay IDM actions 发现，如果 neural trajectory 质量好，IDM replay 也很 accurate。

这暗示：未来 video model 的进步会直接 translate 到 robot learning 的进步。

---

## Limitations 和 Open Questions

### Compute Cost
- 240k RoboCasa samples: 54 hours on 1500 L40 GPUs
- 这是一个 massive compute investment
- 未来需要更高效的 generation 方法

### Manual Initial Frames
- 需要人手动拍摄 initial frames
- 这是 pipeline 中 remaining 的 human bottleneck
- 未来方向：automated initial frame generation/selection

### Task Complexity
- Current tasks 相对简单
- Dexterous, rich control behaviors 仍是挑战
- 需要更 diverse 的 training behaviors

### Evaluator Limitations
- Automatic evaluator 基于 lightweight open-source models
- 偶尔 hallucinate，特别是物理合理性判断
- 这是一个 chicken-and-egg 问题：更好的 evaluator 需要更好的 video understanding

---

## 我的个人 Take

这个工作让我想起你之前关于 "Software 2.0" 和 data-driven AI 的论述。DREAMGEN 本质上是把 robot learning 从 "hardware problem"（需要 physical robots 采集 data）转化为 "software problem"（用 video models 生成 data）。

几个让我 particularly excited 的点：

1. **Scaling law 的出现**: Log-linear relationship between neural trajectories 和 policy performance 暗示了一个新的 scaling axis。如果 video models 继续进步（很可能），robot learning 的 bottleneck 会进一步降低。

2. **Zero-to-one generalization**: 从 0% 到 28.5% 的 environment generalization，从 11.8% 到 43.2% 的 behavior generalization，这些是 paradigm-shifting 的结果。

3. **Embodiment-agnostic 的可能性**: LAPA 路径展示了不需要 target robot actions 的可能性。如果这条路走通，未来可以 zero-shot transfer 到全新 robot platforms。

4. **Benchmark 作为 bridge**: DreamGen Bench 连接了 video model 和 robotics 两个社区，让 video researchers 可以无门槛贡献。

5. **Compositional generalization 的 hope**: Video world models 可以组合 internet priors 和 fine-tuned kinematics，这暗示了 compositionality 的 emergence。

Potential concerns:
- Real-world success rates 仍然不高（最高的 GR1 High Data 也只有 69%）
- Long-horizon tasks 可能需要更好的 temporal reasoning
- Safety 和 robustness 在 real deployment 中仍需验证

---

## References

- [Paper PDF](https://research.nvidia.com/labs/gear/dreamgen)
- [WAN2.1](https://arxiv.org/abs/2503.20314)
- [LAPA](https://openreview.net/forum?id=VYOe2eBQeh)
- [GR00T N1](https://arxiv.org/abs/2503.14734)
- [RoboCasa](https://robocasa.ai/)
- [DROID](https://droid-dataset.github.io/)
- [π₀](https://arxiv.org/abs/2410.24164)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [VideoCon-Physics](https://arxiv.org/abs/2406.03520)
- [Cosmos](https://arxiv.org/abs/2501.03575)
- [CogVideoX](https://arxiv.org/abs/2408.06072)
- [HunyuanVideo](https://arxiv.org/abs/2412.03603)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [VPT (IDM origin)](https://arxiv.org/abs/2206.11795)
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)

这个工作让我看到了 robot learning 从 "data-scarce" 到 "data-abundant" 的转折点。如果 video models 继续按照 current trajectory 进步，未来 robot learning 的 bottleneck 可能不再是 data collection，而是 policy architecture 和 evaluation methodology。
