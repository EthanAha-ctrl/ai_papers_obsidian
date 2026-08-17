---
source_pdf: OPENHELIX.pdf
paper_sha256: 0052c13a0a75078446a6b74d9328ae40ee93c001fc45254bd81f0f0398b8f8b5
processed_at: '2026-08-06T00:39:38-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 OpenHelix

---

## 这篇 paper 到底在干嘛

现在 robot manipulation 领域有个很火的架构叫 **dual-system VLA**——就是搞两个 model，一个大一个小，大的负责"想"，小的负责"动"。

这背后的 intuition 其实特别简单：你让人去抓个杯子，你不会每一毫秒都在"思考"杯子在哪、手该怎么动。你是先看一眼环境、理解任务（慢思考），然后手就自动去抓了（快反应）。Kahneman 的 *Thinking, Fast and Slow* 说的就是这个。

所以大家就想：MLLM（像 LLaVA 这种）当"大脑"负责理解任务和环境，一个 lightweight policy 当"小脑"负责高频输出 action。两全其美——既有 generalization 又有 speed。

但问题是：**这个领域一堆 paper，每个搞法都不一样，没人系统比较过到底什么 work 什么不 work**。而且大部分不开源。

OpenHelix 这篇 paper 就干三件事：

1. 把现有方法梳理一遍，总结出 7 个关键设计选择
2. 做一堆 ablation，搞清楚哪些选择真的重要
3. 基于这些发现，搞一个 open-source baseline 给社区用

---

## 7 个设计选择是什么

你做 dual-system VLA，必须回答这些问题：

1. **MLLM 选什么**：LLaVA？Qwen-VL？大的小的？
2. **policy 选什么**：Diffusion？DiT？Autoregressive？
3. **MLLM 传什么给 policy**：最后一层 hidden？中间层？特殊 token？这个最复杂，各家用法五花八门
4. **MLLM 怎么训**：frozen？LoRA fine-tune？full fine-tune？
5. **policy 怎么训**：从 scratch？还是拿预训练的 fine-tune？
6. **两个 system 怎么连**：projector 怎么设计？训练顺序怎么排？
7. **异步怎么做**：MLLM 慢、policy 快，两者频率不一样怎么处理

paper 固定 1、2、3 的部分设置，重点 ablate 4、5、6——因为这些才是普适的。

---

## 几个 counterintuitive 的发现

### 发现 1：异步频率根本无所谓

你以为是 dual-system 的核心卖点——MLLM 慢慢更新 high-level guidance，policy 快速执行。所以 MLLM 更新频率应该影响 performance。

**结果是：MLLM 1 步推理和 60 步推理，性能几乎一样。**

这就很奇怪了。paper 深挖了一下，把 MLLM 输出的 latent mapping 到 vocabulary 看它"在说什么"——发现 **MLLM 基本只 encode 了 instruction 的文字语义，根本没怎么看图**。

换句话说，你给 MLLM 一张图和一句话 "take the blue block and rotate it to the right"，它输出的 latent 基本就是 "blue block"、"right"、"rotate" 这些词的语义表示。图里 block 到底在哪、环境怎么变，它不管。

**那 MLLM 就退化成了一个 text encoder。** 既然只 encode 文字，那频率当然无所谓——instruction 不变，latent 就不变。

### 发现 2：Projector 必须先单独训

MLLM 输出 4096 维，policy 输入 512 维，中间需要一个 projector（就是个 linear layer）来对齐维度。

如果你一开始就把 MLLM、projector、policy 一起训——**直接崩溃，全是 0**。

必须分两步：
1. 先 freeze MLLM 和 policy，**只训 projector**，让它学会"翻译" MLLM 输出到 policy 能懂的格式
2. 然后再 unfreeze policy，一起 fine-tune

**Intuition**：projector 一开始是 random 的，输出 random noise 给 policy，policy 预训练的 representation 直接被搞坏。先让 projector 学好翻译，再让 policy 适应。

### 发现 3：Prompt Tuning 完胜 Fine-tuning

怎么用 MLLM 有三种选择：
- **Frozen**：MLLM 完全不动
- **Fine-tuning**：LoRA 或者 full fine-tune MLLM
- **Prompt Tuning**：MLLM 参数全冻，只加一个新 token `<ACT>`，只训这一个 token 的 embedding

在 standard benchmark 上三者差不多。但在 **language generalization**（CALVIN-E）上，prompt tuning 大幅领先。

**Intuition**：Fine-tuning MLLM 会破坏它预训练得到的 generalization 能力。Prompt Tuning 只加一个 token，等于在 MLLM 的 vocabulary 里"长"出一个专门给 downstream 用的"触角"，backbone 完好无损。

更 surprising 的是：prompt tuning 时 **连 CLIP loss 都不需要**——性能反而更好。CLIP loss 本来是用来约束 latent 接近 text embedding 的，但这个约束反而限制了 MLLM。

### 发现 4：Auxiliary Task 是关键中的关键

既然 MLLM 退化成了 text encoder，怎么让它真正用 vision？

paper 的方案简单粗暴：**给 MLLM 加一个 auxiliary head，直接让 latent 预测 action**。

就是把 MLLM 输出的 latent 接几个 MLP，直接预测 end-effector 的 location、rotation、gripper state。这样 MLLM 被迫在 latent 里 encode visual info——不 encode 就预测不准 action。

效果：
- Standard task：3.45 → 4.01（+16%）
- Language generalization：2.26 vs 1.72 vs 1.42（巨大提升）

**这是整篇 paper 的核心 contribution**。

---

## OpenHelix 最终长什么样

很简单的架构：

**System 2（MLLM）**：
- LLaVA-7B，参数全冻
- 输入：第三视角 RGB + instruction + `<ACT>` token
- 输出：`<ACT>` token 的 hidden embedding
- Auxiliary head：3 个 MLP 预测 action 的 location/rotation/gripper

**System 1（Policy）**：
- 3D Diffuser Actor（pretrained），fine-tune
- 输入：两个视角 RGB-D + proprioception + MLLM 的 latent（经过 projector 降维）
- 输出：diffusion denoise 出来的 action trajectory

**训练**：
- Stage 1：freeze MLLM 和 policy，只训 `<ACT>` token + projector，2000 iterations
- Stage 2：freeze MLLM，unfreeze policy，一起训，到 100k iterations
- Loss = auxiliary loss + diffusion policy loss

就这么多。没有花哨的 trick。

---

## 为什么这个工作有价值

1. **它问对了问题**：latent representation 到底传什么信息？这个问题之前没人认真验证过，大家都默认"MLLM 在做 visual reasoning"。

2. **它给出了可复现的 baseline**：dual-system VLA 这个方向之前都是 closed paper，社区没法 iterate。OpenHelix 开源了，大家可以在上面改。

3. **auxiliary task 这个 idea 很 general**：不限于 dual-system VLA。任何 MLLM-as-controller 的场景，如果 gradient signal 不够 force MLLM 用 vision，都可以加 auxiliary supervision。这个 insight 可能对 VLA 整个领域都有启发。

---

## 一句话总结

**现有 dual-system VLA 的 MLLM 其实没真正"看"图，它只把 instruction 文字 encode 了传下去。加个 auxiliary task 逼它预测 action，它就被迫真的用 vision 了。**

就这么个事。简单，但之前没人发现和解决。

---

参考链接：
- 项目主页：https://openhelix-robot.github.io/
- GitHub：https://github.com/OpenHelix-robot/OpenHelix/
- LCB：https://arxiv.org/abs/2405.04798
- 3D Diffuser Actor：https://arxiv.org/abs/2402.10885
- RoboDual：https://arxiv.org/abs/2410.08001
- DP-VLA：https://arxiv.org/abs/2410.09932
- HiRT：https://arxiv.org/abs/2410.05273
- CALVIN benchmark：https://calvinrobot.github.io/
- Prompt Tuning 原论文：https://arxiv.org/abs/2104.08691
- Kahneman *Thinking, Fast and Slow*：https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow

---

# OpenHelix：Dual-System VLA 的实证分析与开源实现

Andrej 你好，这篇 paper 挺有意思的——它做的不单是一个 model，而是先 survey、再 ablation、最后给出一个 open-source baseline。这种"先理解再构造"的研究范式在 VLA 这个方向上其实挺稀缺的。下面我把这篇 paper 拆开来详细讲，顺便把一些直觉（intuition）和相关的联想都串起来。

---

## 1. 背景：从 VLA 到 Dual-System VLA

### 1.1 VLA 的诞生与困境

RT-2（https://arxiv.org/abs/2307.15818）开创了 Vision-Language-Action Model 这个范式——把 robotic trajectory data 和 internet-scale vision-language data 一起 co-fine-tune，让 MLLM 直接输出 action token。emergent capability 很强，但有两个硬伤：

- **Inference latency**：RT-2 55B 跑 1-3 Hz，5B 跑 ~5 Hz；而传统 lightweight policy（如 BC-Transformer）可以到 50 Hz。robotic control 的 real-time 要求是 ~20-50 Hz，这个 gap 太大了。
- **Catastrophic forgetting & domain shift**：end-to-end fine-tune 容易把 MLLM 预训练得到的多模态理解能力给抹掉。

### 1.2 Dual-Process Theory 的引入

这个思路来自 Kahneman 的 *Thinking, Fast and Slow*（https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow）：

- **System 1**：fast, automatic, intuitive, unconscious — 对应 lightweight policy network，high frequency（~50 Hz），但 task-specific
- **System 2**：slow, deliberate, effortful, conscious — 对应 MLLM/VLA，low frequency（~1-5 Hz），但 generalizable

关键 insight：两个系统 **并行运行**，但 update frequency 不同。System 2 以低频生成 high-level representation，System 1 以高频消费这个 representation 生成 low-level action。这正好解决了 latency vs. generalization 的 trade-off。

DP-VLA（https://arxiv.org/abs/2410.09932）是第一个把 dual-process theory 拿来给 robotic dual-system architecture 做"personified explanation"的工作。

---

## 2. 现有 Dual-System VLA 方法对比

Table 1 把 6 个方法做了横向对比。我重新整理一下关键差异：

| Method | System 2 Model | System 2 Input | Latent Representation | System 1 Policy | System 1 Sensory | Training |
|--------|---------------|----------------|----------------------|-----------------|------------------|----------|
| **LCB** | LLaVA-7B | L+R | Lang(`<ACT>`) | 3D Diffusion Actor | R+P+PC | Pretrain |
| **DP-VLA** | OpenVLA-7B | L+R | Vis+Lang | Transformer | R+P | Scratch |
| **HiRT** | InstructBLIP-7B | L+R | MaxPool(Vis+Lang) | RT-1 | R | Scratch |
| **RoboDual** | OpenVLA-7B | L+R | Action+Lang | DiT | R+D+T+P | Scratch |
| **DexVLA** | Qwen2-VL-2B | L+R | Lang | ScaledDP | R+P | Scratch |
| **Helix** | N/A | L+R+P | N/A | Transformer | R+P | N/A |

几个值得注意的点：

1. **π0（https://arxiv.org/abs/2410.24164）和 GR00T-N1（https://arxiv.org/abs/2503.14734）被排除在 dual-system 之外**——理由是它们的 System 1 不接收 real-time perception input（如 RGB）。这是个比较严格的定义，意味着 dual-system 必须满足"System 1 有自己的 real-time sensory pathway"。

2. **Latent representation 的选择五花八门**：LCB 用 `<ACT>` token 的 final layer embedding；DP-VLA 用 encoder 的 last layer hidden；HiRT 用 MaxPool；RoboDual 用 Action+Lang 多个 latent；DexVLA 只用 Lang。这是 paper 后面 empirical analysis 的重点。

3. **System 1 sensory input 的丰富度**：RoboDual 用了 RGB+Depth+Tactile+Proprioception，是最 rich 的。但 rich 不一定好——需要 ablation 来验证。

---

## 3. 七个关键设计维度（Figure 1）

paper 总结出 7 个 core design choices，这是这篇 survey 最有价值的部分：

### 3.1 MLLM Selection
- **Size vs. capability trade-off**：Flower（https://flower-robot.github.io/）用 strong spatial awareness 的 foundation model 拿到 SOTA；MiniVLA（https://arxiv.org/abs/2412.09212）用 Qwen-VL 0.25B 降低 inference cost。
- **Open question**：是否需要 robot-data pre-trained MLLM？RoboDual 实验显示 robot pretraining 让 language instruction following 更 robust，但代价是 pretraining 资源巨大。

### 3.2 Policy Selection
当前 consensus：**DiT-based** 和 **Flow Matching-based** 都能 work。但新架构如 CARP（https://arxiv.org/abs/2412.06782）、Dense Policy（https://arxiv.org/abs/2503.13217）可能带来新设计。我个人觉得 autoregressive action prediction（如 CARP 的 coarse-to-fine）是个被低估的方向。

### 3.3 Latent Feature Representation（最复杂、最值得研究的维度）
不同方法的选择差异巨大：
- **Layer selection**：DP-VLA 用 last layer；GR00T-N1 用 middle layer（理由：middle layer 含更多 visual info，且能 reduce inference time）。
- **Pooling strategy**：HiRT 和 Roboflamingo（https://arxiv.org/abs/2311.01378）用 MaxPool over last layer。
- **Special token**：LCB 引入 `<ACT>` token，RoboDual 用 multiple `<ACT>` tokens。
- **Beyond robotics**：MetaQuery（https://arxiv.org/abs/2504.06256）和 LEGO（https://arxiv.org/abs/2410.07040）用了更 sophisticated 的 latent selection 方法，值得借鉴。

### 3.4 MLLM Training Strategy
- Frozen vs. Fine-tuning vs. Prompt Tuning
- paper 自己的实验表明 **Prompt Tuning** 是 sweet spot——见后面 Section 2.3.2。

### 3.5 Policy Training Strategy
- From scratch vs. Fine-tuning from pre-trained
- paper 实验表明 **Fine-tuning pre-trained policy** 既快又好。

### 3.6 Dual-System Integration Strategy
这是最 tricky 的部分。LCB 用 CLIP loss 约束 upstream latent 接近 text CLIP embedding——但 paper 指出这会 **限制 model 只能处理训练过的 case**，negate 了 MLLM 的 generalization。

更关键的是 **Projector pre-alignment**：如果同时 unfreeze 上游和下游 + projector 一起训，**训练直接 collapse**（Table 6 显示全 0）。必须先 freeze MLLM，单独训 projector + policy，然后再 unfreeze policy。这是一个非常重要的工程 insight。

### 3.7 Dual-System Asynchronous Strategy
- LCB：synchronous training + asynchronous testing（最 naive）
- HiRT：用 buffer 在训练时引入 asynchronous
- RoboDual：real-time replacement of coarse action

paper 的实验（Figure 4）显示一个 **counterintuitive** 的结果：asynchronous step 从 1 到 60，性能变化不大。这意味着 MLLM 其实对当前 environment 不敏感——这是 paper 后面 analysis 的起点。

---

## 4. 实验设置

### 4.1 控制变量
paper 固定了 condition 1（MLLM = LLaVA-1.0）、2（Policy = 3DDA）、3（Integration = LCB-style）和 7（synchronous training + asynchronous testing），只 ablate condition 4、5、6。这是非常 fair 的 comparison。

### 4.2 三类 evaluation 环境（Figure 2）
1. **CALVIN ABC-D**（standard）：standard objects, standard instructions
2. **CALVIN-E**（Enriched language）：测试 language generalization
3. **CALVIN-D**（Dynamic）：物体以 Left / Forward / Diagonal / Circle 四种方式运动，测试 dynamic robustness

CALVIN-D 是 paper 的一个 contribution——之前的工作很少在 dynamic scenario 下测试，但 dual-system 的核心卖点恰恰是 high-frequency adaptation to dynamic environment。

### 4.3 Single System 在 CALVIN-D 上的失败（Table 2）

| Model | Static | Left | Forward | Diagonal | Circle |
|-------|--------|------|---------|----------|--------|
| **RoboFlamingo (RF)** | 100 | 0 | 0 | 0 | 0 |
| **3DDA** | 82 | 84 | 46 | 67 | 80 |

RF 在 dynamic scenario 下完全崩溃（0%），原因很有 intuition 价值：RF 用 previous 6 frames 通过 LSTM 推断 action，training 时 latent 稳定，但 testing 时 object 移动导致 latent 变得 unstable，train-test gap 巨大。

但 RF 在 Static 上拿到 100%，远超 3DDA 的 82%——说明 **MLLM 作为 brain 的上限很高，但 generalization 的 robustness 不够**。这是 dual-system 想解决的问题。

---

## 5. Training Strategy 的 Ablation

### 5.1 Policy Training（Table 3）

| MLLM | 1 task | 2 | 3 | 4 | 5 | Avg. Len |
|------|--------|---|---|---|---|----------|
| Fine-tuning | 96 | 83 | 68 | 58 | 48 | 3.53 |
| From-scratch | 89 | 71 | 49 | 42 | 34 | 2.85 |

**结论**：pre-trained policy fine-tuning 全面碾压 from scratch。这符合 transfer learning 的 intuition——3DDA 已经学到了 3D scene representation 和 action manifold，从头训既慢又差。

### 5.2 MLLM Training + CLIP Loss（Table 4）

| MLLM | Integration | 1 | 2 | 3 | 4 | 5 | Avg.Len |
|------|-------------|---|---|---|---|---|---------|
| Frozen | w/ CLIP | 94 | 80 | 64 | 51 | 41 | 3.30 |
| Frozen | w/o CLIP | 90 | 74 | 61 | 54 | 40 | 3.33 |
| Fine-tuning | w/ CLIP | 96 | 83 | 68 | 58 | 48 | 3.53 |
| Fine-tuning | w/o CLIP | 88 | 72 | 56 | 46 | 30 | 3.13 |

**关键 insight**：
- MLLM frozen 时，CLIP loss 影响不大（因为 CLIP loss 就是用来 compensate frozen MLLM 的）
- MLLM fine-tune 时，**CLIP loss 必须有**——否则会破坏 small model 的 attention mechanism，性能从 3.53 掉到 3.13

### 5.3 Prompt Tuning 的引入（Table 5）

paper 提出一个 hypothesis：能不能 freeze MLLM 参数，只训一个新加的 `<ACT>` token 的 embedding？这就是 **Prompt Tuning**——只更新 lm-head 层对应新 token 的 embedding，其他参数 frozen。

CALVIN（standard）上和 fine-tuning 持平，但 **CALVIN-E（language generalization）上 prompt tuning 大幅领先**：

| MLLM | Integration | CALVIN-E Avg.Len |
|------|-------------|------------------|
| Prompt-tuning | w/ CLIP | 2.09 |
| Prompt-tuning | w/o CLIP | 2.13 |
| Fine-tuning | w/ CLIP | 1.74 |
| Frozen | w/ CLIP | 1.46 |

**Intuition**：Prompt Tuning 只学一个新 token，不破坏 MLLM 的 generalization capability。这和 LoRA 的哲学类似，但更激进——LoRA 至少改了 attention 的 low-rank 部分，prompt tuning 完全不动 backbone。

更 surprising 的是：**w/o CLIP loss 反而更好**（2.13 > 2.09）。这意味着 CLIP loss 这个约束本身可能是 suboptimal 的，prompt tuning 已经足够 align latent 和 downstream。

### 5.4 Projector Pre-alignment（Table 6）

| Pre-alignment | MLLM | 1 | 2 | 3 | 4 | 5 |
|---------------|------|---|---|---|---|---|
| √ | Frozen w/ CLIP | 94 | 80 | 64 | 51 | 41 |
| √ | Fine-tuning w/ CLIP | 96 | 83 | 68 | 58 | 48 |
| √ | Prompt-tuning w/o CLIP | 94 | 77 | 67 | 60 | 47 |
| **×** | Frozen w/ CLIP | **0** | 0 | 0 | 0 | 0 |
| **×** | Fine-tuning w/ CLIP | **0** | 0 | 0 | 0 | 0 |
| **×** | Prompt-tuning w/o CLIP | **0** | 0 | 0 | 0 | 0 |

**没有 pre-alignment，全部崩溃到 0**。这是非常重要的工程结论。

**Intuition**：MLLM 的 output space（4096 dim, LLaVA-7B 的 hidden size）和 3DDA 的 input space（512 dim）分布完全不同。如果 projector 随机初始化，MLLM 输出的 random signal 会直接 destroy 3DDA 已经学好的 representation。必须先 freeze MLLM 和 policy，只训 projector 让它学会"翻译"，然后再 unfreeze policy 让它适应新 signal。

---

## 6. 测试策略的 Counterintuitive 发现

### 6.1 Asynchronous Inference 的失效（Figure 4）

paper 在 CALVIN-D 上测试 asynchronous step 从 1 到 60，发现 **性能几乎不变**。

这非常 counterintuitive——如果 System 2 真的在提供 "real-time high-level guidance"，那它的 update frequency 应该影响 performance。结果不变意味着 System 2 传给 System 1 的信息 **不依赖于当前 environment state**。

### 6.2 Action Token 在说什么？（Figure 5）

paper 做了一个很有创意的实验：把 `<ACT>` token 的 latent embedding 通过 cosine similarity 映射到 vocabulary space，看它最接近哪些 word。

实验设置：blue block 持续向左移动。

**结果**：
1. "right" 的 probability 始终高于 "left"，无论 robot arm 实际向左还是向右移动
2. Top 10 similar words 主要是 **instruction 中的 object、spatial relation、action verb**，加上一些 noise
3. Latent embedding 主要 **summarize textual instruction**，对 visual 变化不敏感

**这揭示了一个 fundamental problem**：现有 dual-system VLA 的 MLLM 实际上 **退化成了 text encoder**——它没有真正用 visual reasoning capability。LLaVA 把 "take the blue block and rotate it to the right" 这个 instruction encode 成 latent，传给 3DDA。视觉输入虽然在 model 里走了，但对 latent 输出几乎没贡献。

这也解释了为什么 asynchronous step 不影响 performance——既然 latent 只 encode instruction，那 instruction 不变，latent 就不变，frequency 自然无所谓。

---

## 7. OpenHelix 的核心贡献：让 MLLM 真正"看"到东西

### 7.1 三个 variant 对比（Table 7）

| Type of MLLM | Auxiliary | Avg.Len (CALVIN) |
|--------------|-----------|-------------------|
| MLLM (Prompt Tuning) | × | 3.45 |
| **LLM (Prompt Tuning)** | × | **1.77** |
| MLLM (Prompt Tuning) | √ | **4.01** |

**三个对比非常 revealing**：
1. MLLM → LLM（去掉 visual input）：性能从 3.45 暴跌到 1.77 — 说明 vision encoder 确实在贡献，但贡献不够
2. MLLM + Auxiliary task：性能从 3.45 跳到 4.01 — **这是 paper 的核心 finding**

**Intuition**：auxiliary task 强制 MLLM 在 latent embedding 里 encode action-relevant visual info（position、rotation、gripper state）。光靠 downstream policy 的 gradient 没法 force MLLM 用 vision——因为 gradient 经过 projector 后被 dilute 了。直接在 latent 上加 supervision，逼 MLLM 把视觉信息"压"进 latent。

### 7.2 架构详解（Figure 6, 7）

整个系统分两部分：

**System 2（High-level MLLM）**：
- Input：third-view RGB $o'_t$ + language instruction $l$ + learnable token `<ACT>`
- Processing：LLaVA-7B 全部 frozen，只更新 `<ACT>` token 的 embedding
- Output：`<ACT>` token 在 LLM final layer 的 hidden embedding $z_t^{<ACT>} \in \mathbb{R}^{4096}$
- Auxiliary head：3 个 MLP 分别预测 action 的 location、rotation、gripper state

**System 1（Low-level Policy）**：
- Input：3D scene tokens $o_t$（来自 RGB-D 双视角）+ proprioception $c_t$ + latent $z_t^{<ACT>}$（经过 linear projector 降到 512）+ noisy trajectory $\tau_t^i$ + diffusion step $i$
- Architecture：3D Diffuser Actor（Transformer-based diffusion model with cross-attention）
- Output：denoised action trajectory $\tau_t = (a_{t:t+T}^l, a_{t:t+T}^r)$ + gripper state $a_{t:t+T}^g$

**Action 分解**：
$$a_t = \{a_t^l \in \mathbb{R}^3, a_t^r \in \mathbb{R}^6, a_t^g \in \{0,1\}\}$$
- $a_t^l$：3D location（end-effector position）
- $a_t^r$：6D rotation representation（来自 Zhou et al. 的 *On the Continuity of Rotation Representations*，https://arxiv.org/abs/1812.07035，比 quaternion 更适合 learning）
- $a_t^g$：gripper open/close binary state

### 7.3 Loss Function 深度解析

#### 7.3.1 Auxiliary Loss $\mathcal{L}_{lm}$（公式 1）

$$
\begin{aligned}
\mathcal{L}_{lm}(<ACT>) = & \text{BCE}(\text{MLP}(f_\phi^g(o'_t, l')), a_{t:t+T}^g) \\
& + \omega_1 \cdot ||\text{MLP}(f_\phi^l(o'_t, l')) - a_{t:t+T}^l|| \\
& + \omega_2 \cdot ||\text{MLP}(f_\phi^r(o'_t, l')) - a_{t:t+T}^r||
\end{aligned}
$$

变量解释：
- $f_\phi^g, f_\phi^l, f_\phi^r$：分别对应 gripper / location / rotation 三个 head 的 MLLM forward。实际实现中，应该是同一个 forward $f_\phi(o'_t, l')$ 输出 $z_t^{<ACT>}$，然后三个不同的 MLP 分别处理。
- $o'_t$：third-view RGB image（System 2 只用 single view RGB，System 1 用两个 RGB-D view）
- $l' = \{l, <ACT>\}$：在 instruction 末尾拼接 `<ACT>` token
- $a_{t:t+T}^l \in \mathbb{R}^{T \times 3}$：未来 $T$ 步的 location trajectory
- $a_{t:t+T}^r \in \mathbb{R}^{T \times 6}$：未来 $T$ 步的 6D rotation
- $a_{t:t+T}^g \in \{0,1\}^{T}$：未来 $T$ 步的 gripper binary state
- $\omega_1, \omega_2$：balance hyperparameters（paper 没给具体值，需要查 code）

**Loss 设计的 intuition**：
- Location 和 rotation 用 $L_1$ loss（比 $L_2$ 对 outlier 更 robust，trajectory prediction 里常用）
- Gripper 用 BCE（binary classification）
- 这个 loss 直接 supervision 在 MLLM 的输出 latent 上，强制它 encode action-relevant info

#### 7.3.2 Diffusion Policy Loss $\mathcal{L}_{policy}$（公式 2）

$$
\begin{aligned}
\mathcal{L}_{policy}(\theta, <ACT>) = & \text{BCE}(\pi_\theta^g(o_t, z_t^{<ACT>}, c_t, \tau_t^i, i), a_{t:t+T}^g) \\
& + \omega_3 \cdot ||\epsilon_\theta^l(o_t, z_t^{<ACT>}, c_t, \tau_t^i, i) - \epsilon_{t:t+T}^l|| \\
& + \omega_4 \cdot ||\epsilon_\theta^r(o_t, z_t^{<ACT>}, c_t, \tau_t^i, i) - \epsilon_{t:t+T}^r||
\end{aligned}
$$

变量解释：
- $\pi_\theta^g$：policy network 的 gripper head
- $\epsilon_\theta^l, \epsilon_\theta^r$：diffusion model 的 noise prediction head（分别预测 location 和 rotation 的 noise）
- $o_t$：System 1 的 3D scene representation（来自 RGB-D 双视角）
- $z_t^{<ACT>}$：System 2 输出的 latent（经过 projector）
- $c_t$：proprioception（robot joint state）
- $\tau_t^i$：noisy trajectory at diffusion step $i$
- $i$：diffusion step embedding
- $\epsilon_{t:t+T}^l, \epsilon_{t:t+T}^r$：ground truth noise（随机采样的 Gaussian noise）

这是标准 DDPM（https://arxiv.org/abs/2006.11239）的 denoising objective，只不过 condition 包括了 $z_t^{<ACT>}$、$c_t$、$o_t$、$i$ 四个部分。

#### 7.3.3 Total Loss（公式 3）

$$
\mathcal{L}_{total} = \mathcal{L}_{lm} + \mathcal{L}_{policy}
$$

简单相加，没有 dynamic weighting。这点我觉得可以改进——Uncertainty Weighting（https://arxiv.org/abs/1705.07115）或 GradNorm 可能更好。

### 7.4 Two-Stage Training

- **Stage 1（Pre-alignment，2000 iterations）**：freeze MLLM 和 policy，只训 `<ACT>` token embedding + projector MLP
- **Stage 2（Fine-tuning，到 100k iterations）**：freeze MLLM，unfreeze policy，一起训

Projector 是一个 linear layer：4096 → 512。

**Intuition**：Stage 1 让 projector 学会"翻译" MLLM output 到 policy input space。如果一开始就 unfreeze policy，random projector 输出的 noise 会 destroy 3DDA 的 pretrained representation。

---

## 8. 最终结果（Table 8）

| Type | Method | CALVIN Avg.Len | CALVIN-E Avg.Len |
|------|--------|----------------|-------------------|
| Single | Only Policy | 3.27 | 1.42 |
| Dual | MLLM(PT) + Policy(P) | 3.30 | 1.72 |
| **Dual + AUX** | + Asy(10) | **3.45** | **2.26** |
| Dual + AUX | + Asy(60) | 3.44 | 2.20 |

**几个结论**：
1. CALVIN 上，Dual vs. Single 几乎没差距（3.30 vs 3.27）——standard task 上 MLLM 加成有限
2. CALVIN-E 上，Dual 比 Single 有明显提升（1.72 vs 1.42）——**MLLM 的价值在 language generalization**
3. Auxiliary task 在 CALVIN 和 CALVIN-E 上都有大幅提升——这是 paper 的核心 contribution
4. Asy(10) vs Asy(60) 差距很小——再次验证 asynchronous frequency 不重要

---

## 9. 相关联想与 Open Questions

### 9.1 和 Helix 的关系
paper 标题提到 "Open-Source Dual-System VLA Model for Robotic Manipulation"，但 Helix（Figure 8 里的 Helix entry）是另一个 closed-source 的 dual-system model。OpenHelix 想做的是 **Helix 的 open-source reproduction**。Section 4 列出了 5 个未完成目标：
1. Real robot deployment
2. Fast downstream policy execution
3. Physical robot running
4. Humanoid robot deployment
5. Humanoid collaboration

这些都需要后续工作。我建议关注他们的 GitHub（https://github.com/OpenHelix-robot/OpenHelix/）追踪更新。

### 9.2 和 π0 的对比
π0（Physical Intelligence，https://arxiv.org/abs/2410.24164）用 Flow Matching 代替 Diffusion，inference 更快。但 paper 严格把它排除在 dual-system 之外，因为 π0 的 System 1 不接收 real-time perception。这是一个值得 debate 的定义——π0 实际上通过 action chunking + adaptive execution 实现了类似 dual-system 的效果，只是 architectural form 不同。

### 9.3 Latent Representation 的更深层思考
paper 的 Figure 5 实验揭示的问题其实在 LLM-as-controller 的工作里很常见。比如 Code as Policies（https://code-as-policies.github.io/）也是 LLM 把 instruction 转成 code，但 LLM 自己不"看"环境。Visual programming（ViperGPT，https://viper.cs.cornell.edu/）尝试让 LLM 调用 vision module，但仍然是 LLM 主导。

OpenHelix 的 auxiliary task 本质上是在 latent space 加 supervision，让 MLLM 的 vision encoder 真正参与。但还有其他思路：
- **Visual Prompting**：让 MLLM 输出 bounding box / segmentation mask 作为 latent（如 SAM-2，https://arxiv.org/abs/2408.00714）
- **Cross-attention bridge**：让 System 1 直接 attend 到 MLLM 的中间 layer（而不是 final layer）
- **Iterative refinement**：System 2 多次推理，每次基于 System 1 的 action result 修正

### 9.4 Prompt Tuning vs. LoRA
paper 用 Prompt Tuning 而非 LoRA（https://arxiv.org/abs/2106.09685），原因是 Prompt Tuning 完全不动 backbone，更好保留 generalization。但 Prompt Tuning 的 capacity 有限——只学一个 token。可能 **Prompt Tuning + tiny LoRA on vision encoder only** 是更好的折中。

### 9.5 Action Token 的信息瓶颈
`<ACT>` token 的 embedding 是 4096 dim，但 paper 没分析 information bottleneck。如果用 information theory 视角（像 InfoNCE，https://arxiv.org/abs/1808.06670）分析 latent 的 mutual information with visual vs. textual input，可能能更精确量化 Figure 5 揭示的问题。

### 9.6 Diffusion vs. Flow Matching vs. Autoregressive
paper 用 3D Diffuser Actor（diffusion），但提到了 CARP（autoregressive）和 Dense Policy（bidirectional autoregressive）。我个人觉得 action prediction 用 autoregressive（如 ARIA，https://arxiv.org/abs/2410.05251）有天然优势——action 本身是 sequential，autoregressive 能 capture temporal dependency。Flow Matching（如 π0）则在 inference speed 上有优势。

### 9.7 6D Rotation Representation
paper 用 6D rotation representation（来自 Zhou et al.）。如果想深入，可以对比：
- Euler angles：discontinuity at gimbal lock
- Quaternion：连续 but antipodal symmetry
- 6D representation：first two columns of rotation matrix，continuous and unique
- 9D representation：full rotation matrix，redundant but stable

6D 是目前 robotics learning 的事实标准。

### 9.8 Multi-view Fusion
System 1 用两个 RGB-D view，System 2 用 single RGB view。这个 asymmetry 值得思考——为什么 System 2 不用 depth？我的猜想：
1. LLaVA 预训练在 RGB 上，加 depth 需要 fine-tune vision encoder
2. System 2 关注 high-level semantics，depth 信息不那么关键
3. 减少 System 2 的 inference cost

但 depth 对 spatial reasoning 很重要。未来可以用 DepthFM（https://depthfm.github.io/）或 Metric3D（https://arxiv.org/abs/2307.10984）把 RGB 转 depth，作为 auxiliary input。

### 9.9 CALVIN Benchmark 的局限
CALVIN（https://calvinrobot.github.io/）是当前 VLA 最常用的 benchmark，但有几个问题：
1. 只有 4 个 task，太简单
2. Object set 固定，generalization 测试有限
3. 没有真正的 dynamic obstacle

OpenHelix 加的 CALVIN-D 是个改进，但还不够。LIBERO（https://libero-project.github.io/）和 RoboCasa（https://robocasa.github.io/）可能是更好的选择。

### 9.10 和 World Model 的关系
world model（如 DreamerV3，https://arxiv.org/abs/2301.04104；Genie，https://arxiv.org/abs/2402.19463）学的是 environment dynamics，而 MLLM 在 dual-system 里其实可以看作 "task-level world model"——它理解 "blue block 旋转后是什么样"。如果能显式让 System 2 做 future prediction（如 video prediction），可能比单纯 auxiliary action prediction 更强。

---

## 10. 总结

这篇 paper 做了三件事：

1. **Survey**：把 dual-system VLA 的 7 个 design choice 系统化，并指出 latent representation selection 是最 underexplored 的。

2. **Empirical Analysis**：通过 ablation 揭示几个 counterintuitive finding：
   - Asynchronous frequency 不重要（因为 MLLM 没真正用 visual info）
   - Projector pre-alignment 必须（否则训练崩溃）
   - Prompt Tuning > Fine-tuning > Frozen（在 generalization 上）
   - Auxiliary task 是激活 MLLM visual reasoning 的关键

3. **OpenHelix**：基于以上 insight 构建的 open-source model——Prompt Tuning + Auxiliary Action Prediction + Two-stage Training。

**核心 intuition**：现有 dual-system VLA 的 MLLM 退化成了 text encoder，paper 用 auxiliary task 强制 MLLM 真正用 visual info。这个 finding 可能对整个 VLA 领域都有启示——end-to-end training 时，gradient signal 不一定能 force MLLM 用所有 modality，显式 auxiliary supervision 可能是必要的。

参考链接：
- 项目主页：https://openhelix-robot.github.io/
- GitHub：https://github.com/OpenHelix-robot/OpenHelix/
- LCB：https://arxiv.org/abs/2405.04798
- 3D Diffuser Actor：https://arxiv.org/abs/2402.10885
- RoboDual：https://arxiv.org/abs/2410.08001
- DP-VLA：https://arxiv.org/abs/2410.09932
- HiRT：https://arxiv.org/abs/2410.05273
- DexVLA：https://arxiv.org/abs/2502.05855
- CALVIN：https://calvinrobot.github.io/
- 6D Rotation：https://arxiv.org/abs/1812.07035
- DDPM：https://arxiv.org/abs/2006.11239
- Prompt Tuning：https://arxiv.org/abs/2104.08691
- LoRA：https://arxiv.org/abs/2106.09685
- Flower：https://flower-robot.github.io/
- π0：https://arxiv.org/abs/2410.24164
- GR00T N1：https://arxiv.org/abs/2503.14734
- CARP：https://arxiv.org/abs/2412.06782
- Dense Policy：https://arxiv.org/abs/2503.13217
- RoboFlamingo：https://arxiv.org/abs/2311.01378
- MetaQuery：https://arxiv.org/abs/2504.06256
- LEGO：https://arxiv.org/abs/2410.07040

希望这个详尽的讲解能帮你 build 起对 dual-system VLA 的 intuition。如果对某个具体 ablation 或 architecture 细节有进一步兴趣，可以追 code 或者我们继续讨论。
