---
source_pdf: RynnBrain Open Embodied Foundation Models.pdf
paper_sha256: f3718860b959adbd5ff1e6d6146cb8e6c9dad327bc5f1864833a2a3b3cebeca8
processed_at: '2026-08-12T02:31:23-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RynnBrain 用人话说

## 一句话总结

**让一个 VLM 学会"用坐标说话"，这样它 reasoning 的时候每一步都能指着画面里的东西，而不是凭空想象，从而变成一个靠谱的 robot brain。**

---

## 这个 paper 在解决什么真实问题

假设你有一个很聪明的 VLM（比如 Qwen3-VL），你让它当 robot 的"大脑"。你跟它说"去把桌子上的红色杯子拿过来"，它会面临三个尴尬：

1. **它知道什么是红色杯子，但不知道"桌子上的"具体在画面的哪个位置**——它从来没有真正"指过"东西
2. **它 reasoning 的时候全程在文字空间里转**——"我需要先找到杯子，然后规划路径..."这些文字 reasoning 跟实际画面里的像素位置完全脱节，说着说着就开始 hallucinate
3. **它没法输出"动作"**——它会写小作文，但不会输出"grab at (350, 420)"这种 robot 能执行的东西

RynnBrain 的核心 thesis 就一句话：**把坐标变成跟文字平起平坐的 token，让模型在 reasoning 的过程中随时能"指着画面说事"。**

---

## 他们怎么做到的

### 1. 输出空间的重设计

传统 VLM 的 output 只有 text tokens。RynnBrain 说：我们让 output 可以是坐标。

具体怎么做？把所有 spatial 东西都 normalize 到 [0, 1000]，然后当成整数 token 来 generate。比如一个 bounding box 就是吐出四个整数 token：(x0, y0, x1, y1)。一条 trajectory 就是吐出一串点。

**人话翻译**：模型本来就会预测下一个 token 是 "the" 还是 "cup"，现在它也能预测下一个 token 是 "350" 还是 "420"。用的是同一套 autoregressive 机制，同一套 training objective，同一套 inference pipeline。这个 unified 的设计很 clean，你不需要给 spatial output 单独搞一个 head 或者 decoder。

### 2. Chain-of-Point (CoP)—— 这篇 paper 最 elegant 的 idea

普通 Chain-of-Thought 你见过：模型先 reasoning 一段文字，然后给出答案。问题是对于 embodied task，纯文字 reasoning 会飘。

RynnBrain 的 CoP 是这样玩的：模型 reasoning 的时候，**文字和坐标交替出现**。

举个例子，任务是"把杯子放到水槽旁边"：
```
我需要先找到杯子。[找到杯子] → <object> frame 5: (340, 200), (420, 280) </object>
然后找到水槽旁边的空位。[找空位] → <area> frame 5: (700, 400), (750, 450) </area>
规划从杯子的把手到目标位置...[抓取点] → <affordance> frame 5: (380, 240) </affordance>
```

**人话翻译**：模型每 reasoning 一步，就"指"一下画面里的具体位置。这样 reasoning 的每一步都有 visual evidence backing it，从根上抑制 hallucination。你没法瞎编，因为坐标要么对要么不对，画面会立刻给你反馈。

这个 idea 本质上是给 reasoning 加了一个 grounding constraint，让 language reasoning 不能脱离 visual reality 乱跑。

### 3. Data flywheel—— 怎么搞到 20M 训练数据

这是最工程化的部分，也是最真实的 bottleneck。Embodied data 极度稀缺，纯靠人标注 20M 样本不现实。

他们的策略是 **human-model collaborative**：便宜的部分让模型干，贵的部分让人干。

具体 pipeline（以 Object Understanding 为例）：
1. Qwen2.5-VL 识别视频里有什么东西（便宜）
2. Grounding DINO 1.5 做 detection（便宜）
3. SAM2 做 segmentation 和 tracking（便宜）
4. Qwen2.5-VL 生成 QA pair（便宜）
5. **人 review 和 filter 质量**（贵，但只在关键点）

Spatial Understanding 数据更有意思。他们用 **MASt3R-SLAM** 从 RGB video 重建 3D point cloud，然后基于真实 3D geometry 生成 spatial QA。这意味着模型训练时看到的不只是 2D pixel patterns，而是通过 3D reconstruction 获得的 real metric distances、relative positions。

**人话翻译**：很多 VLM 在 spatial reasoning 上翻车，是因为它们只学到 2D appearance，没学到真正的空间关系。RynnBrain 通过 SLAM 重建把 3D ground truth 喂给模型，让模型能 learn 到"这个东西离那个东西 1.5 米"这种 metric reasoning。

### 4. Load balancing 的工程优化

这个细节看着不起眼，但解决了一个真实的 training 痛点。

混合训练 short tasks（object localization，几十 tokens）和 long tasks（long video reasoning，上万 tokens）时，naive data parallel 会让某些 worker 拿到一堆 long sequences，变成 straggler。

他们的解法：**greedy 贪心算法做 load balancing**。按 sequence length 降序排序，每个 sequence 分给当前最闲的 worker。这是经典 multiprocessor scheduling 的 LPT 算法。

还有个更 subtle 的优化：传统 per-token loss 需要全局 all-gather 来算 denominator（总 token 数），有同步 overhead。他们改成 per-sample loss，每个 sample 先除以自己的 length 再 average，这样 denominator 是常数，不用通信。**paper 声称这 double 了 training efficiency，而且 convergence properties 没坏。**

**人话翻译**：训练大模型的时候，GPU 之间要同步统计信息。如果每个 GPU 处理的 sequence 长度不一样，同步就变慢。他们的 loss function 改写让每个 GPU 不需要互相通信就能算自己的 loss，省去了这个 bottleneck。

---

## 四个 post-training 方向

Pretraining 出来的 RynnBrain 是个 generalist embodied brain。然后他们 show 了这个 brain 能 efficiently adapt 到四个下游 task：

### RynnBrain-CoP（Chain-of-Point Reasoning）
用 GRPO 做 RL training。Reward 设计得很 task-specific：
- Trajectory 用 Discrete Fréchet Distance（衡量两条曲线的 shape + sequential alignment）
- Affordance 用 bidirectional Chamfer distance（同时管 precision 和 recall）
- Area 用 point-in-polygon accuracy

**直觉**：RL 让模型从 SFT 的"模仿标注"进化到"自己探索 grounding"。只保留 intermediate difficulty 的 samples 做 RL（40-80 分的），太简单学不到东西，太难又 noisy。

### RynnBrain-Nav（Navigation）
把 VLN task 套成 multi-turn 对话格式：
```
[observation_0, action_0, observation_1, action_1, ...]
```
History 作为 explicit memory buffer。用 DAgger 做多轮迭代提升。

**实验结果亮点**：8B model 在 R2R 上 SR 58.6%，surpass 很多用 panoramic view + depth + odometry 的方法，而他们只用 single RGB。这很 impressive。

**一个有意思的 ablation**：30B MoE 在 navigation 上没有 beat 8B dense。这说明 sparse activation 对需要 dense spatial-temporal processing 的 task 可能不友好。MoE 的 routing 机制可能没法充分 capture 这种 fine-grained pattern。

### RynnBrain-Plan（Manipulation Planning）
这里有个让人惊讶的 empirical observation：**只 fine-tune 几百个 samples 就能让 model 学会 long-horizon planning**。

秘诀是 multi-turn dialogue format。Single-turn 训练的 model 在 Hard difficulty 上几乎全军覆没（0%），multi-turn 训练的 model 在 Hard 上还能保持 30-100%。这说明 **temporal context 是 planning 的关键**，model 需要看到自己之前做了什么决策，才能 coherent 地 plan 下一步。

**OOD 泛化**：Table Bussing 这个 task 在 fine-tuning 时完全没见过，但 30B model 在 Hard difficulty 上能达到 100% task progress。而 Qwen3-VL 在同样 task 上 < 10%。这个 gap 说明 RynnBrain 学到的是 generalizable planning logic，不是 task-specific pattern matching。

### RynnBrain-VLA（Vision-Language-Action）
基于 RynnBrain-2B，用 flow matching framework 预测 action chunk。VLM backbone 当 Diffusion Transformer 用。

Input 格式很 clever：pointing 信息（affordance point, object bounding box）以 text 形式塞进 conversation format。Actions 放在 sequence 末尾，这样 inference 时能用 KV cache。

**实验结果**：RynnBrain-VLA 在 SR 上 0.77，beat π0.5-finetuned 的 0.47。关键是 RSR（recognition success rate）上 RynnBrain 0.97 vs π0.5 的 0.57。这说明 **embodied pointing pretraining 给了 VLA 强大的 localization 能力**，而 π0.5 这种纯 action model 在 fine-grained image-text alignment 上有 structural weakness。

---

## 结果好到什么程度

几个 highlight：

1. **RynnBrain-CoP-8B average 73.8**，beat Gemini-3-Pro 的 65.1 和 RoboBrain2.0-32B 的 57.7。8B 干翻 32B，说明 CoP 的 reasoning paradigm 比 raw parameter scaling 更 effective。

2. **RynnBrain-Nav-8B** 在 R2R 上 SR 58.6%，只用 single RGB 就 beat 用 panoramic + depth + odometry 的方法。OS 71.6% 说明 model 很擅长 coarse navigation，但 terminal stopping precision 还有 gap（SR 58.6%）。

3. **RynnBrain-Plan 30B** 在 OOD task 上接近 100% task progress，而 baseline 几乎全 fail。这个 generalization 程度让人印象深刻。

4. **RynnBrain-VLA** 在 multi-object grasping 上 SR 0.77，significantly beat π0.5 的 0.47。

---

## 我从这篇 paper 学到的几个 intuition

### 1. Unified output space 的威力
把坐标当 token generate 这个 idea 看着简单，但 implication 很深。它意味着你不需要 separate vision head、detection head、planning head——一个 autoregressive LLM 能 unified 处理所有这些。这是 scaling friendly 的设计。

### 2. Grounding as reasoning constraint
CoP 的核心 insight 不是"让 reasoning 更准确"，而是"用 spatial grounding 来 constrain reasoning 不能乱飘"。每一步 reasoning 都 anchor 在画面里的具体位置，这 create 了一个 verification loop。Hallucination 的 root cause 是 reasoning 脱离 evidence，CoP 从 architecture level 解决这个问题。

### 3. Pretraining 的 data quality > parameter scale
8B model 干翻 32B 的 case 在这篇 paper 里反复出现。这说明对于 embodied task，pretraining data 的 task-specific quality 比 raw model size 重要。19.89M 的 carefully curated data 比 generic web data 有效得多。

### 4. Multi-turn memory 是 long-horizon planning 的关键
Single-turn 训练的 model 在 hard task 上 0%，multi-turn 训练的能到 100%。这个 ablation 非常 clean 地 show 了：**long-horizon 不是 model capacity 问题，是 memory format 问题**。你给 model 一个 explicit memory buffer（dialogue history），它就能 plan；你不给，它就只能 react 单步。

### 5. MoE 在 dense spatial-temporal task 上可能不 work
30B-A3B 在 navigation 上没 beat 8B dense。这是个 cautionary tale：MoE 的 sparse routing 对需要 fine-grained spatial-temporal processing 的 task 可能是个 liability，不是 asset。dense model 在这类 task 上可能更 sample efficient。

### 6. Embodied pointing 是 VLA 的 missing piece
π0.5 这种纯 action model 在 RSR 上只有 0.57，因为它没有学过"指着画面里的东西"这个 capability。RynnBrain-VLA 继承了 pretraining 的 pointing 能力，在 multi-object scene 里能准确 identify target。这 suggest **VLA 不只是 action prediction，更是 grounded perception 的延伸**。

---

## 这篇 paper 在 research landscape 里的位置

我觉得 RynnBrain 代表了 embodied AI 的一个 phase transition：

**Phase 1（2020-2023）**: Task-specific models。每个 task 训一个 model（navigation model、grasping model、planning model）。Generalization 差，但每个 task 上 SOTA。

**Phase 2（2024-2025）**: VLA models。RT-2、Octo、π0 这些 work 把 actions tokenize，用 VLM 做 end-to-end action prediction。Generalization 好了，但 high-level reasoning 还是弱。

**Phase 3（2025-2026）**: Embodied foundation models。RynnBrain、RoboBrain 2.0、Pelican-VL 这些 work。Core idea 是 VLM 本身可以 serve as embodied brain，只要你在 pretraining 时 inject 足够的 embodied-specific data（grounding、planning、spatial reasoning）。

RynnBrain 在这个 landscape 里的 unique contribution 是 **CoP reasoning paradigm + unified coordinate output space**。别的 embodied brain model 多半还是 textual reasoning + separate grounding module，RynnBrain 把这两者 interleave 在一个 autoregressive process 里。

---

## 跟相关工作的直觉联系

- **RT-2** [https://arxiv.org/abs/2307.15818]: 把 actions tokenize 成 text，RynnBrain 把 coordinates tokenize，思路类似但 RynnBrain 更 systematic
- **π0** [https://arxiv.org/abs/2410.24164]: Flow matching VLA，RynnBrain-VLA 借了这个 framework 但加了 pointing condition
- **Hi Robot** [https://arxiv.org/abs/2502.19417]: Hierarchical VLA，RynnBrain 的 planning data schema 跟它一致
- **RoboBrain 2.0** [https://arxiv.org/abs/2507.02029]: BAAI 的 embodied brain，RynnBrain 在多数 benchmark 上 outperform 它
- **PaLM-E** [https://arxiv.org/abs/2303.03371]: Google 的 embodied multimodal LM，但没 explicit spatial output space
- **VLM-3R** [https://arxiv.org/abs/2505.20279]: 3D-augmented VLM，跟 RynnBrain 的 MASt3R-SLAM data pipeline 类似
- **NaVILA** [https://arxiv.org/abs/2412.04453]: Legged robot VLA，RynnBrain-Nav 在 R2R/RxR 上 surpass 它
- **StreamVLN** [https://arxiv.org/abs/2507.05240]: Streaming VLN，RynnBrain-Nav 借了它的 multi-turn format

---

## 几个我觉得值得深挖的点

1. **CoP 的 RL reward 是 rule-based**，没用 learned reward model。这在 spatial task 上可行是因为坐标可以 exact match，但对于更 abstract 的 embodied reasoning（比如"efficient path planning"）可能需要 learned reward。未来 work 可能 explore hybrid reward。

2. **30K RL samples** 就够了，这个 data efficiency 很高。Suggest SFT 已经把 model 带到 "reasonable regime"，RL 只需要 refine。这跟 R1-style RL 的 observation 一致。

3. **VLA 的 flow matching + VLM backbone** 这个组合很有意思。VLM 既能做 language understanding，又能当 DiT 用，这是个 unified architecture。但 paper 没 ablate 这个设计 vs 纯 diffusion policy，值得进一步研究。

4. **MoE 在 VLN 上失效** 这个 observation 我觉得很重要。MoE 的 scaling 在 embodied task 上可能不是 free lunch，需要 task-specific 的 routing strategy 或者 dense activation。

5. **RynnBrain-Bench** 强调 spatio-temporal video understanding，这填补了现有 benchmark 的 gap。现有 benchmark 要么是 static image grounding，要么是 single-frame pointing，RynnBrain-Bench 要求 model 在 long video 里做 temporal localization + spatial grounding，这更接近 real embodied scenario。

---

## 最后的 take

RynnBrain 这个工作给我几个 deep impression：

**第一，output space design 很重要。** 把坐标当 first-class token 这个 idea 看着简单，但让整个 training pipeline 统一了。你不需要 separate detection head、planning head、action head。一个 autoregressive LLM 吃一切、吐一切。

**第二，grounding 是 reasoning 的 anchor，不是 add-on。** CoP 的核心 insight 是 reasoning 需要 continually anchored to evidence。纯 textual reasoning 在 embodied 场景会飘，因为 language 不足以 constrain spatial facts。每一步指一下画面，这个 action 看着 redundant，但 create 了一个 verification loop。

**第三，data flywheel 是 embodied AI 的 enabler。** 20M samples 不可能纯人标注。Human-model collaborative 的关键是：让 model 干 cheap part，人只 adjudicate expensive part。这个 paradigm 以后会越来越重要。

**第四，pretraining 的 task-specific data 比 model scale 重要。** 8B 干翻 32B 在这篇 paper 里反复出现。对于 embodied task，你需要在 pretraining 时 inject grounding、planning、spatial reasoning 的 data，光 scaling 一个 generic VLM 不够。

**第五，embodied foundation model 是个 real direction。** 之前大家觉得 VLM 和 robot policy 是两个 world，RynnBrain 这种 work show 了：一个 well-pretrained VLM 可以 serve as embodied brain，然后 efficiently adapt 到 navigation、planning、VLA 等 downstream task。这个 hierarchical architecture（brain + policy）可能比 end-to-end VLA 更 practical。

总之，RynnBrain 是个 systematic 的工作，从 data、architecture、training、evaluation 全栈都做了 careful design。它不是 single brilliant idea，而是 multiple good ideas 的组合，每个 idea 都 well-executed。这种 work style 我觉得是 embodied AI 走向 practical 的必要路径。

---

**相关 Links**:
- Project: [https://alibaba-damo-academy.github.io/RynnBrain.github.io](https://alibaba-damo-academy.github.io/RynnBrain.github.io)
- Code: [https://github.com/alibaba-damo-academy/RynnBrain](https://github.com/alibaba-damo-academy/RynnBrain)
- Models: [https://huggingface.co/collections/Alibaba-DAMO-Academy/rynnbrain](https://huggingface.co/collections/Alibaba-DAMO-Academy/rynnbrain)
- Qwen3-VL: [https://arxiv.org/abs/2511.21631](https://arxiv.org/abs/2511.21631)
- GRPO: [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
- MASt3R-SLAM: [https://arxiv.org/abs/2504.12348](https://arxiv.org/abs/2504.12348)
- π0: [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)
- Hi Robot: [https://arxiv.org/abs/2502.19417](https://arxiv.org/abs/2502.19417)
- StreamVLN: [https://arxiv.org/abs/2507.05240](https://arxiv.org/abs/2507.05240)
- DeepEP: [https://github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
- SAM2: [https://arxiv.org/abs/2408.00714](https://arxiv.org/abs/2408.00714)
- RT-2: [https://arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818)
- RoboBrain 2.0: [https://arxiv.org/abs/2507.02029](https://arxiv.org/abs/2507.02029)

---

# RynnBrain: Open Embodied Foundation Models 深度技术解析

## 1. Paper Overview 与 Motivation

RynnBrain 是 Alibaba DAMO Academy 在 2026 年 2 月发布的 open-source embodied foundation model family。这个工作非常 solid，它试图解决一个核心矛盾：现有的 VLMs 具备语义泛化能力 but 缺乏 physical grounding；而 embodied models 训练在 action-centric data 上 but 丢失了 high-level semantic abstraction。RynnBrain 的核心 thesis 是构建一个 unified foundation model，既保留 VLM 的 semantic breadth，又 explicitly structured around physical space、temporal dynamics 和 embodiment constraints。

**Model Family**:
- 三个 scale：RynnBrain-2B (dense), RynnBrain-8B (dense), RynnBrain-30B-A3B (MoE)
- 四个 post-trained variants：RynnBrain-CoP (chain-of-point reasoning), RynnBrain-Nav (navigation), RynnBrain-Plan (manipulation planning), RynnBrain-VLA (vision-language-action)
- 基于 Qwen3-VL [https://arxiv.org/abs/2511.21631] 构建

**四个核心能力**:
1. Comprehensive egocentric understanding
2. Diverse spatio-temporal localization
3. Physically grounded reasoning
4. Physics-aware planning

Reference: [https://alibaba-damo-academy.github.io/RynnBrain.github.io](https://alibaba-damo-academy.github.io/RynnBrain.github.io)

---

## 2. Architecture 详解

### 2.1 基础架构

RynnBrain 采用 decoder-only vision-language architecture，基于 Qwen3-VL 的设计原则。三个核心组件：

1. **Vision Encoder**: 从 Qwen3-VL 继承的 vision encoder
2. **Vision-Language Projector**: 将 visual tokens 投影到 language space
3. **LLM Backbone**: Qwen3-VL-2B/8B/30B-A3B-Instruct

两个关键技术增强：

**DeepStack** [https://arxiv.org/abs/2410.02278]: Deeply stacking visual tokens，将深层 transformer layer 的 visual tokens 也输送到 LLM，这样可以提供 multi-level visual representation。这种做法与传统的只用最后一层 visual feature 不同，让 LLM 能 access 到更细粒度的 visual information。

**Interleaved MRoPE** [https://arxiv.org/abs/2510.23095]: Revisiting multimodal positional encoding。MRoPE (Multimodal Rotary Position Embedding) 是 Qwen-VL 系列采用的 positional encoding 方法，将 position embedding 分解为 temporal、height、width 三个维度。Interleaved 版本可能是改进了 multi-image/video 场景下 position encoding 的方式，避免 frame 之间 position 信息混淆。

### 2.2 Input/Output 空间设计

**Input**: multimodal signals，包括 single-view images, multi-view images, videos, spatio-temporal coordinates

**Output**: 这是一个关键创新点，输出空间 explicit 地包含 spatial grounding primitives：
- Natural language text
- Points (x, y)
- Bounding boxes (x0, y0, x1, y1)
- Trajectories (ordered point sequences)

所有 spatial entities 都 normalized 到 [0, 1000] 范围并 encoded 为 integer tokens。这个设计很聪明，它把 continuous spatial prediction 问题转化为 classification 问题，让 autoregressive generation 机制可以直接处理 spatial output。

Formally，visual input V 表示为 frame sequence {I_t}_{t=1}^{T}：
- T = 1: static image
- T > 1: video (uniformly sampled)

每个 frame encoded 为 visual tokens，并 augmented with temporal positional embeddings。

---

## 3. Infrastructure: Load Balancing 创新

这一节的技术细节值得仔细讲，因为这是一个真实的工程瓶颈解决方案。

### 3.1 问题背景

Embodied data 有一个非常显著的 long-tail sequence length distribution。Video tasks、long-horizon planning tasks 与 short-response tasks (e.g., object localization) 混合训练时，naive data parallel (DP) 分配会导致 straggler effect：拿到 heavy workload 的 worker 成为 throughput bottleneck。

### 3.2 Online Load-Balancing Pipeline

RynnBrain 的解决方案：

1. **Sequence length estimation**: 根据预计算的 image sizes 和 text token 数量 estimate 所有 samples 的 sequence length
2. **Batch sampling redistribution**: 在 DP group 内 aggregate 所有 samples，然后基于"minimize max cumulative sequence length within each DP worker"目标 redistribute
3. **Greedy approximation algorithm**:
   - Initialize n 个 buffers (n = DP world size)
   - 按 sequence length 降序排序
   - 迭代地将每个 sequence 分配给当前 total length 最小的 buffer

这个 greedy algorithm 类似于 multiprocessor scheduling problem 的 LPT (Longest Processing Time first) algorithm，理论上是 4/3-approximation。

在 SPMD (Single Program, Multiple Data) framework 下，使用 stable sorting 确保 global data distribution 在所有 worker 间一致。这个动态 allocation 避免了 hyperparameters 或 datasets 变化时的 costly data pre-processing。

### 3.3 Per-Sample Loss Reduction

这是一个很 elegant 的工程优化。传统 per-token loss formulation (Equation 1)：

$$
\mathcal{L} = \frac{1}{\sum_{i=1}^{n} \sum_{j=1}^{b_i} s_{ij}} \sum_{i=1}^{n} \sum_{j=1}^{b_i} \sum_{k=1}^{s_{ij}} l_{ijk}
$$

变量含义：
- n: DP world size (worker 数量)
- b_i: i-th worker 上的 local batch size
- s_{ij}: i-th worker 上 j-th sequence 的 sequence length
- l_{ijk}: i-th worker 上 j-th sequence 的 k-th token 的 per-token loss

这个公式的 denominator (global token count) 需要 all-gather 操作 across DP group，引入同步 overhead，降低 training efficiency。

RynnBrain 改为 per-sample loss reduction (Equation 2)：

$$
\mathcal{L} = \frac{1}{b} \sum_{i=1}^{n} \sum_{j=1}^{b_i} \frac{1}{s_{ij}} \sum_{k=1}^{s_{ij}} l_{ijk}
$$

变量含义：
- b: global batch size (constant known to each worker)
- 其他变量同 Equation 1

这个改动的核心 insight：每个 sample 先 normalize 自己的 loss (除以自己的 sequence length)，然后再 average。这样 denominator 变成 constant b，每个 worker 独立计算，无需额外通信。Paper 声称这 holistic approach **doubles training efficiency while preserving model stability and convergence properties**。

这个 trade-off 的 intuition：per-token loss 让每个 token 贡献相等；per-sample loss 让每个 sample 贡献相等。后者对 long sequence sample 有 smoothing effect，但在 batch 足够大、sample 分布相对均匀的情况下，convergence properties 仍然可以保持。

### 3.4 Memory 优化

- **2B 和 8B**: ZeRO-1 optimizer + per-block gradient checkpointing + selective logits filtering (filter out multimodal tokens 不需要 loss 的)
- **30B-A3B**: ZeRO-2 + Expert Parallel (EP) world size 2 + DeepEP [https://github.com/deepseek-ai/DeepEP] for cross-GPU token dispatching + CUTLASS-based grouped linear operation for MoE

---

## 4. Physics-Aware Spatio-temporal Pretraining

### 4.1 Training Recipe

Pretraining 采用 standard next-token prediction objective (Equation 3)：

$$
\mathcal{L} = -\sum_{i=1}^{L} \log P(y_i | \mathbf{y}_{<i}, \mathbf{V}, \mathbf{\Theta})
$$

变量含义：
- L: sequence length (text + coordinate tokens 的混合序列长度)
- y_i: i-th token (可能是 textual token 或 coordinate token)
- y_{<i}: 前面所有 tokens
- V: visual input
- Θ: model parameters

这是一个统一的自回归目标，text 和 spatial coordinates 共享同一个 generation mechanism。

**Training Hyperparameters** (Table 1):

| Parameter | RynnBrain-2B | RynnBrain-8B | RynnBrain-30B-A3B |
|-----------|--------------|--------------|-------------------|
| Base Model | Qwen3-VL-2B-Instruct | Qwen3-VL-8B-Instruct | Qwen3-VL-30B-A3B-Instruct |
| Optimizer | AdamW | AdamW | AdamW |
| Learning Rate | 5e-6 | 2e-6 | 2e-6 |
| Learning Rate Vision | 1e-6 | 2e-6 | 2e-6 |
| Global Batch Size | 512 | 1024 | 1024 |
| Warmup Ratio | 0.03 | 0.03 | 0.03 |

注意：vision encoder 的 learning rate 通常比 LLM 更小（除了 2B 是 1e-6 vs 5e-6），这是因为 vision encoder 已经在 Qwen3-VL pretraining 中 well-trained，需要更小的 perturbation。

### 4.2 Pretraining Data 详解 (19.89M samples)

数据规模和构成是这篇 paper 的核心 contribution 之一。

**General MLLM Data (4.80M)**: 
- LLaVA-OV-SI, LLaVA-Video, ShareGPT-4o-video, VideoGPT-plus, FineVideo, CinePile, ActivityNet, YouCook2, LLaVA-SFT
- 目的：保留 broad multimodal understanding capability

**Multi-Dimensional Cognition Data**:

**Object Understanding (1.10M)**:
- Data format: `<object> <frame n>: (coordinates) </object>`
- Pipeline: Qwen2.5-VL 识别 → Grounding DINO 1.5 detection → SAM2 segmentation & tracking
- 每个视频限制每类最多两个 instance 减少 redundancy
- 712K high-quality egocentric QA samples

**Spatial Understanding (2.50M)**:
- 这个数据特别 interesting，因为 spatial reasoning 是很多 VLM 的 weakness
- Pipeline: MASt3R-SLAM [https://arxiv.org/abs/2504.12348] 重建 3D point clouds + camera extrinsics
- Instance segmentation 投影到 3D space
- RANSAC 检测 ground plane 并 enforce gravity-aligned world coordinate system
- 基于 calibrated 3D scenes 生成 metric distances、relative positions、heights 等 spatial QA
- 855K video-based + 272K image-based spatial QA

这里有个 intuition 可以 build：很多 VLM 在 spatial reasoning 上失败是因为它们只看到了 2D pixel，没有 access 到 3D metric information。MASt3R-SLAM 提供了从 RGB video 重建 3D 的能力，让模型在 training 时可以学习到真正的 spatial relationship，而不仅仅是 2D appearance patterns。

**Counting (0.30M)**: 222K Molmo2 counting subset + 42K egocentric

**OCR (1.00M)**:
- 来源: Ego4D, Charades-Ego, EPIC-KITCHENS
- GoMatching 检测 scene text
- 视频按 text appearance pattern 分成 3-15 second clips
- 85,324 text-containing segments
- Human annotators 标注 first appearance frame、clearest frame、text transcription、bounding polygons
- 两种 QA 生成策略：
  - GPT-5.2 [https://openai.com/index/introducing-gpt-5-2/] 生成 goal-oriented first-person questions (256K)
  - Template-based 生成 structured questions (722K)

**Egocentric Task Understanding (2.77M)**:
- Env-QA, EgoTaskQA, RoboVQA, EgoRe-5M, QAEgo4D, Robo2VLM, ShareRobot
- 排除 < 3s 的视频

### 4.3 Spatio-Temporal Location Data

这个 dataset design 非常有 systematic，覆盖了 embodied agent 需要的所有 location types。

**Object Location (1.20M)**:
- Representation: (V, Q, B, t)，其中 B = {(x0, y0, x1, y1)} 是 normalized bounding box
- 900K public (ADE20K, COCO, Mapillary, PACO-LVIS, PASCAL-Part, VG, RoboAford++)
- 300K egocentric，使用相同的 segmentation pipeline
- Referring expressions 两种类型：simple (category/position) 和 situational (task-level inference)

**Area Location (3.37M)**:
- Representation: (V, Q, P, t)，其中 P = {(x_i, y_i)}_{i=1}^n 是 normalized point set
- 这个 task 很重要因为 robot 需要识别 non-object regions (surfaces, empty space, functional areas)
- 6K egocentric house-touring video + 222K Molmo2-VideoPoint + 448K image-area + 2.2M RoboAford++/RefSpatial

**Affordance Location (1.13M)**:
- Representation: (V, Q, p, t)，其中 p = (x, y) 是 normalized affordance point
- Affordance 是 actionable points (handles, buttons, interaction hotspots)
- 6K video + 476K image (from 500K indoor images) + 260K RoboAford++

**Trajectory Location (0.56M)**:
- Representation: (V, Q, T, t_s)，其中 T = {(x_i, y_i)}_{i=1}^m 是 up to 10 个 normalized trajectory points
- 6K video + 507K image + 13K FSD

**Grasp Pose Location (1.00M)**:
- Representation: (I, Q, G)，其中 G = {(x_i, y_i)}_{i=1}^4 是 ordered corner points
- 来源: Grasp-Anything [https://arxiv.org/abs/2405.13043]
- 参数化: center (c_x, c_y), dimensions (w, h), rotation angle θ
- 通过 rotation 转换为 4 个 corner points
- 995K images at 416×416 resolution
- 1.3M training samples (avg 1.44 samples/image)
- Weighted prompt strategy: 40% object-centric + 30% scene-aware + 30% task-oriented

### 4.4 Physics-Aware Planning Data (0.16M)

- Following Hi Robot [https://arxiv.org/abs/2502.19417] 的 atomic actions 概念
- Long-horizon tasks 分解为 temporally ordered sub-tasks
- 每个 sub-task annotated with: target object bounding box, placement area points, affordance points
- Representation: (V, Q, M)，M 是 textual tokens + grounding annotations 的 mixed sequence
- 数据来源: AgibotWorld Alpha [https://github.com/OpenDriveLab/AgiBot-World], Open X-Embodiment [https://arxiv.org/abs/2310.08864]

---

## 5. Physically Grounded Chain-of-Point (CoP) Reasoning

这是 paper 的一个 key innovation，我觉得是 most elegant 的部分。

### 5.1 问题动机

Most multimodal reasoning models [https://arxiv.org/abs/2503.21776] [https://arxiv.org/abs/2501.05452] 依赖 purely textual reasoning paradigm。即使有一些方法 [https://arxiv.org/abs/2505.14362] incorporate auxiliary tools (e.g., region zooming)，reasoning process 仍然 detached from physical spatial structure。

Visual imagination 方法 [https://arxiv.org/abs/2505.17022] [https://arxiv.org/abs/2505.11409] 有 hallucinated visual content 问题。

对 embodied agent 来说，reasoning 必须 grounded in observable physical evidence。CoP 的核心 idea：**interleave textual reasoning with explicit spatial grounding**。

### 5.2 Cold-Start SFT

**Training Recipe**:
- Full-parameter SFT with AdamW + cosine learning rate
- Peak LR: 1e-5 for LM + projector, 2e-6 for vision encoder
- 3% warmup
- 1 epoch, global batch size 128
- 2 FPS frame sampling (up to 2048 frames)
- Max context: 16,384 tokens
- DeepSpeed ZeRO-1

**Data Construction Pipeline**:

这是一个 human-model collaborative flywheel 的典型例子：

1. Given task instruction + video frames，用 Qwen3-VL-235B pre-generate step-by-step textual reasoning chain
2. Reasoning chain 中用 square brackets 标记 potential entities，e.g., `[white flower-patterned wallpaper]`
3. In-house model 分类每个 entity 为 "area" 或 "object"
4. Human annotators review + annotate:
   - "area" → 一组 representative points
   - "object" → 2D bounding box
   - 选择 most relevant and clear frame
5. Grounding results 插入 reasoning text，格式：`<object/area> <frame n>: (coordinates) </object/area>`

最终 sample extends to (V, Q, P_final, t_s, R)，R 是 interleaved reasoning chain。

这个方法的 intuition：reasoning 不是 abstract 的 chain of thought，而是 continually anchored to specific visual evidence in physical space。这从根本上抑制了 hallucination，因为每个 reasoning step 都有 spatial evidence backing it up。

### 5.3 Reinforcement Learning with GRPO

**GRPO Objective** (Equation 4):

$$
\mathcal{I}_{\mathrm{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G} \sum_{i=1}^{G} \left(\min\left(\rho_i A_i, \mathrm{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i\right) - \beta \mathbb{D}_{KL}(\pi_\theta(o_i|q) || \pi_{\mathrm{ref}}(o_i|q))\right)\right]
$$

变量含义：
- G: group size (sampled outputs 数量)
- o_i: i-th sampled output
- ρ_i: importance sampling ratio = π_θ(o_i|q) / π_{θ_old}(o_i|q)
- A_i: advantage of i-th output
- ε: clipping range
- β: KL divergence penalty coefficient
- π_ref: reference model
- q: query

GRPO [https://arxiv.org/abs/2402.03300] 与 PPO [https://arxiv.org/abs/1707.06347] 的关键区别：不需要 value function (critic)，而是从 group 内 multiple sampled outputs 的 scores estimate baseline，显著减少 memory usage 和 training complexity。

**Advantage** (Equation 5):

$$
A_i = \frac{r_i - \mathrm{mean}(\{r_1, ..., r_G\})}{\mathrm{std}(\{r_1, ..., r_G\}) + \epsilon}
$$

变量含义：
- r_i: i-th output 的 reward
- mean, std: group 内 rewards 的均值和标准差
- ε: small constant for numerical stability

这是 group-relative normalization，让 advantage 反映 sample 在 group 内的相对优劣。

**Training Config**:
- SGLang [https://arxiv.org/abs/2411.02735] inference engine for efficient rollout
- G = 5 (group size)
- 10 epochs, batch size 128
- Cosine LR starting at 2e-6, 3% warmup
- Clipping range: [0.2, 0.28] (注意是 range 不是单一值)
- KL coefficient β = 0.02
- Max sequence length: 16,384 tokens

### 5.4 Reward Design (Task-Specific Rule-Based)

所有 spatial coordinates 在 reward computation 前 normalized 到 [0, 1]。

**Trajectory Reward** (Equation 6, 7):

Discrete Fréchet Distance (DFD) 是衡量两条 polygonal curves 相似度的经典 metric，比 Hausdorff distance 更能 capture sequential alignment。

$$
c(i, j) = \max\left(\|p_i - g_j\|_2, \min\left(c(i-1, j), c(i, j-1), c(i-1, j-1)\right)\right)
$$

变量含义：
- p_i: predicted path 的 i-th point
- g_j: ground truth 的 j-th point
- c(i, j): coupling distance between prefixes p_{1:i} 和 g_{1:j}
- 初始条件: c(0, 0) = ||p_1 - g_1||_2

最终 distance: D_F = c(M, N)

Reward:
$$
r_{\mathrm{traj}} = \exp(-\lambda_{\mathrm{traj}} \cdot D_F)
$$

λ_traj 控制衰减速度。Exponential decay 让 reward 平滑地从 1 (perfect match) 衰减到 0。

**Affordance Reward** (Equation 8):

Bidirectional Mean Euclidean Distance (Chamfer distance 变体):

$$
D_{\mathrm{bidir}}(\mathcal{P}, \mathcal{G}) = \frac{1}{2}\left(\frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \min_{g \in \mathcal{G}} \|p - g\|_2 + \frac{1}{|\mathcal{G}|} \sum_{g \in \mathcal{G}} \min_{p \in \mathcal{P}} \|g - p\|_2\right)
$$

变量含义：
- P: predicted interaction points set
- G: ground truth points set
- 第一项: 每个 predicted point 到最近 ground truth point 的平均距离 (precision)
- 第二项: 每个 ground truth point 到最近 predicted point 的平均距离 (recall)

这个 metric jointly captures precision 和 recall，避免了只看单向距离的 bias。Reward: r_aff = exp(-λ_aff · D_bidir)

**Area Reward** (Equation 9):

Point-retrieval within valid polygon:

$$
r_{\mathrm{area}} = \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \mathbb{I}(p \in S_\mathcal{G})
$$

变量含义：
- P: predicted points set
- S_G: ground truth polygon 定义的 geometric region
- I(·): indicator function

这是 strict accuracy，predicted points 必须在 ground truth polygon 内才算正确。

### 5.5 RL Data Curation

- 基于 pretraining 的 spatiotemporal localization data
- **Difficulty-aware filtering**: pretrained SFT model 评分，只保留 intermediate difficulty (40-80 分)
- 包含 SFT model 选错 key frame 的 failure cases
- 最终 30K training samples

这个 curation 策略很聪明：trivial samples 无法让 RL 学到东西，excessively noisy/ambiguous cases 又会引入 bad gradients。Intermediate difficulty 是"最有学习价值"的 zone。

---

## 6. Post-training for Embodied Tasks

### 6.1 Vision-Language Navigation (RynnBrain-Nav)

**Problem Formulation**:
- Input: natural language instruction Q, visual observations O = {o_0, o_1, ..., o_t}
- Output: action a_t
- Action space: A = {↑, ←, →, STOP} (forward 30cm, turn left/right 15°, halt)
- 每个观察 o_i ∈ R^(3×H×W) 是 RGB image

**Data Format** (Equation 10):

Following StreamVLN [https://arxiv.org/abs/2507.05240]，采用 multi-turn conversational format：

$$
\{o_0, a_0, o_1, a_1, ..., o_n, a_n\}
$$

这是 interleaved image-text sequence，训练目标是 predict next action based on current observation + conversational history。这个 format 的核心 benefit：history 作为 explicit memory buffer，让模型 bridge individual steps into coherent trajectory。

**Data Collection**:
- 450K video clips from R2R [https://arxiv.org/abs/1711.11465], R2R-EnvDrop, RxR across 60 Matterport3D environments
- 300K samples from ScaleVLN subset 增强 scene diversity
- Multi-turn DAgger [https://arxiv.org/abs/1011.0686] for iterative improvement

**Fine-tuning Settings**:
- Full-parameter SFT, AdamW + cosine LR
- Peak LR: 2e-5 for LM + projector, 2e-6 for vision encoder
- 3% warmup, 1 epoch, batch size 256
- 2 FPS, up to 2048 frames
- Max context: 16,384 tokens
- DeepSpeed ZeRO-1

### 6.2 Manipulation Planning (RynnBrain-Plan)

这里有个非常 interesting 的 empirical observation：**只需几百个 samples 就能 endow model with robust long-horizon planning capability**。

关键设计：
- Multi-turn dialogue format，interaction history 作为 explicit memory buffer
- Grounding annotations 只 apply 到每个 dialogue turn 的 final frame
- 这确保 current decisions conditioned on immediate observation + accumulated memory

这种 data efficiency 的 intuition：RynnBrain 的 pretraining 已经让 model 具备了 spatial grounding 和 basic planning capability。Post-training 主要是在 teach model 如何使用 multi-turn memory 来 bridge individual planning steps。

### 6.3 VLA (RynnBrain-VLA)

**Architecture**:

基于 RynnBrain-2B，采用 **Flow Matching** framework [https://arxiv.org/abs/2410.24164] 预测 action chunk。

VLM backbone 作为 single-stream **Diffusion Transformer (DiT)**，输入是 packed sequence (condition + noisy actions)。

三个 linear projections 用于 align dimensions：
1. Input noises → VLM hidden size
2. Input timestamp embeddings → VLM hidden size
3. Output actions → VLM hidden size

**关键设计**: 使用 VLM 的 native conversation format 组织 input sequence，pointing information 以 text-based format 传递，task 的 initial frame prepended to input sequence。

Input example:
```
<|im_start|>user
INSTRUCTION:
<start_frame>
Pick the <affordance> (x,y) </affordance> of the <object> (x0,y0),(x1,y1) </object>
OBSERVATION:
<camera_1><camera_2><camera_3>
STATE:
<state>
What action should the robot take?<|im_end|>
<|im_start|>assistant
<action>
```

Following π_0 [https://arxiv.org/abs/2410.24164]，actions 放在 sequence 末尾以 enable KV cache during inference。

**Fine-tuning**:
- 6 个 pick-and-place tasks，3 个 distinct objects
- Manual teleoperation on Franka Emika arm
- 60K steps, LR 2e-5, batch size 32
- Images resized to short-side 384 pixels

---

## 7. RynnBrain-Bench: 新的 Evaluation Suite

### 7.1 Overview

- 3,616 video clips
- 577,998 frames
- 12,000 open-ended questions
- 21 specialized sub-capabilities
- 4 dimensions: Object Cognition, Spatial Cognition, Grounding, Pointing

### 7.2 Evaluation Metrics

**Acc@0.5** (Equation 11):

$$
\mathrm{Acc@0.5} = \mathbb{I}\left(\mathcal{G}_t \neq \emptyset \wedge \mathrm{IoU}(B, \mathcal{G}_t) > 0.5\right)
$$

变量含义：
- G_t: frame t 的 ground truth
- B: predicted bounding box
- IoU: Intersection over Union
- I(·): indicator function

只有当 model 选择了 valid ground truth 的 frame t 且 IoU > 0.5 时才算正确。这 jointly evaluates temporal localization (选对 frame) 和 spatial localization (bbox 准确)。

**Pointing Metrics**:

Trajectory: 用 DFD (Equation 7)，resample 到 15 个 points uniformly distributed along arc length

Area: 用 Equation 9 (point-in-polygon ratio)

Affordance (Equation 12):

$$
D(\mathcal{P}, \mathcal{G}) = \exp\left(-\frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \min_{g \in \mathcal{G}} \|p - g\|_2\right)
$$

这是单向 Chamfer distance 的 exponential decay 形式。

---

## 8. Experimental Results 分析

### 8.1 Embodied Cognition (Tables 3, 4)

RynnBrain-8B vs Qwen3-VL-8B 的关键提升：
- VSI-Bench: 71.0 vs 60.3 (+10.7)
- RoboSpatial: 73.1 vs 58.2 (+14.9)
- RynnBrain-Object: 71.2 vs 41.8 (+29.4)
- RynnBrain-Spatial: 59.9 vs 35.0 (+24.9)

RynnBrain-30B-A3B vs Qwen3-VL-30B-A3B：
- VSI-Bench: 74.5 vs 65.8 (+8.7)
- Open-X VQA: 83.4 vs 76.8 (+6.6)
- RynnBrain-Object: 73.3 vs 42.6 (+30.7)
- RynnBrain-Spatial: 59.3 vs 30.7 (+28.6)

这些巨大的提升表明 pretraining data 的质量直接 translate 到 downstream performance。

### 8.2 Physically Grounded Reasoning (Table 5)

RynnBrain-CoP-8B vs 其他 thinking models:
- Affordance: 90.3 (唯一突破 90 的)
- Area: 59.6 (vs Gemini-3-Pro 50.7)
- Trajectory: 71.2 (vs GPT-5.2 70.5)
- Average: 73.8 (vs Gemini-3-Pro 65.1, vs RoboBrain2.0-32B 57.7)

这个结果 very impressive：8B model 超越了 32B model 16.1%，证明了 CoP 的 interleaved reasoning 比单纯 parameter scaling 更 effective。

### 8.3 Vision-Language Navigation (Table 6)

RynnBrain-Nav-8B on R2R-CE Val-Unseen:
- SR: 58.6% (top-ranked)
- SPL: 49.6% (second-best)
- NE: 4.92 (lowest)
- OS: 71.6% (exceeds all competitors)

值得注意：OS (71.6%) 远高于 SR (58.6%)，说明 model 擅长 coarse-level navigation 但缺乏 terminal stopping 的 precision。

**Ablation on Pre-training**:
- 2B RynnBrain-Nav vs 2B Qwen3-VL: +7.2% SR, +7.6% SPL
- 这证实了 RynnBrain pretraining 的 clear efficacy

**MoE Scaling 异常**: 30B MoE (3B active) 在 VLN 上没有 outperform 8B dense。Paper 推测 sparse activation 机制可能没被 VLN task 充分 leverage，或需要 alternative training strategies。

这个 observation 很 interesting，让我联想到 MoE 在 dense prediction task 上的 scaling 确实有时不如 dense model，因为 sparse routing 可能无法 capture 细粒度的 spatial-temporal patterns。

**Multi-turn DAgger**:
- Baseline: 50.6% SR
- 1st iteration: 56.4%
- 2nd iteration: 58.5%
- 3rd iteration: marginal improvement

### 8.4 Planning and Manipulation (Tables 7, 8, Figure 6)

**Multi-turn Dialogue Ablation** (Table 7):

| Method | Object Cls (E/M/H) | Desk Org (E/M/H) | Distribute (E/M/H) | Table Bussing (E/M/H) |
|--------|---------------------|-------------------|---------------------|----------------------|
| RynnBrain-Plan-ST 8B | 72/20/0 | 60/0/0 | 34/0/0 | 90/0/0 |
| RynnBrain-Plan-MT 8B | 100/100/75 | 100/41/55 | 92/61/78 | 100/71/30 |
| RynnBrain-Plan-MT 30B | 85/91/62 | 84/92/75 | 95/62/75 | 100/90/100 |

Single-turn 训练的 model 在 Hard difficulty 上几乎完全失败 (0%)，但 multi-turn 训练的 model 在 Hard 上仍能保持 30-100% 的 performance。这 strongly validates temporal context 的 necessity。

**OOD Generalization (Table Bussing)**:
- RynnBrain-Plan 30B-A3B: Easy 100%, Medium 90%, Hard 100%
- Qwen3-VL: Hard < 10%
- Gemini-3-Pro: Hard ~60%

这个 OOD 表现非常 impressive，说明 model 学到的是 generalizable planning capability 而非 task-specific patterns。

**VLA Evaluation** (Table 8):

| Method | Overall PSR | Overall RSR | Overall SR |
|--------|-------------|-------------|------------|
| π0.5-Finetuned | 0.67 | 0.57 | 0.47 |
| Qwen3-VL-Finetuned | 0.60 | 1.00 | 0.60 |
| RynnBrain-VLA | 0.80 | 0.97 | 0.77 |

π0.5 的 bottleneck 是 RSR (0.57)，因为它 limited capacity for fine-grained image-text alignment。RynnBrain-VLA 凭借 embodied pointing pretraining 在 localization accuracy 上显著领先。

---

## 9. Intuition Building & 个人观察

### 9.1 Core Insight: Spatial Coordinates as First-Class Tokens

RynnBrain 的核心 contribution 在我看来是把 spatial coordinates 提升为 first-class tokens，与 text tokens 平等地参与 autoregressive generation。这个 design choice 有几个重要 implications：

1. **Unified generation mechanism**: 同一个 next-token prediction objective 同时处理 text 和 spatial output，避免了 separate head 或 multi-stage pipeline 的 complexity
2. **Physical grounding as natural language**: Model 在 reasoning 时自然地 "speak" in coordinates，这 creates a grounding constraint——每个 reasoning step 都可以 verify against visual evidence
3. **Scalability**: 坐标 [0, 1000] 的 discretization 让 model 可以利用现有 LLM 的所有 scaling laws

### 9.2 Chain-of-Point 与 Chain-of-Thought 的对比

CoT 在 language reasoning 上有效是因为 language 本身就是 symbolic 的。但 embodied reasoning 涉及 spatial relationships，purely textual CoT 容易 hallucinate spatial facts。

CoP 的 elegance 在于：它不是在 textual reasoning 中"想象"spatial 位置，而是在每一步 reasoning 中 explicitly generate spatial coordinates，这些坐标可以直接 verify against visual input。这 create 了一个 grounding loop：
- Textual reasoning → 提出候选 entity
- Spatial grounding → verify entity location
- Verified location → inform下一步 reasoning

### 9.3 Human-Model Collaborative Data Flywheel

Paper 提到 data construction framework strategically leverages pretrained foundation models 的 priors，introducing human supervision only at critical decision points。这是大规模 data generation 的 key insight：

- **Cheap parts (automation)**: Entity detection, frame selection, template generation
- **Expensive parts (human)**: Quality verification, ambiguous case adjudication, final annotation

这种分工让 19.89M samples 的规模成为可能，同时保持 high quality。

### 9.4 与相关工作的联系

- **RT-2** [https://arxiv.org/abs/2307.15818]: 把 actions tokenize 为 text tokens，RynnBrain 把 coordinates tokenize，思路相似但 application 不同
- **PaLM-E** [https://arxiv.org/abs/2303.03371]: Embodied multimodal language model，但没 explicit spatial output space
- **Octo** [https://arxiv.org/abs/2405.12213]: 通用 robot policy，但 focus 在 low-level control
- **NaVILA** [https://arxiv.org/abs/2412.04453]: Legged robot VLA，RynnBrain-Nav 在 R2R/RxR 上 surpass 它
- **VLM-3R** [https://arxiv.org/abs/2505.20279]: Vision-language models + 3D reconstruction，与 RynnBrain 的 spatial understanding 数据 construction 类似
- **RoboBrain 2.0** [https://arxiv.org/abs/2507.02029]: BAAI 的 embodied brain model，RynnBrain 在多数 benchmark 上 outperform 它
- **π_0** [https://arxiv.org/abs/2410.24164]: Flow matching VLA，RynnBrain-VLA 借鉴了这个 framework
- **DeepEP** [https://github.com/deepseek-ai/DeepEP]: DeepSeek 的 EP 通信库，RynnBrain 30B 训练用到

### 9.5 Limitations 推测

虽然 paper 没有显式讨论 limitations，我从 results 中可以推断：

1. **MoE scaling 在 VLN 上失效**: 30B-A3B 没有超越 8B dense，说明 sparse activation 可能不适合需要 dense spatial-temporal processing 的 task
2. **Terminal stopping precision**: OS (71.6%) vs SR (58.6%) 的 gap 说明 model 在精确 stopping 上还有提升空间
3. **Data scale for planning**: 只有 0.16M planning samples，虽然 post-training 只需几百 samples，但 pretraining data 规模可能限制 long-horizon 复杂度
4. **Real-world transfer**: 大部分 evaluation 在 simulation (Habitat) 上，real-world deployment 的 robustness 还需更多验证

### 9.6 Future Directions

Paper 提到未来 embodied intelligence systems 可能包含：
- Brain (类似 RynnBrain)
- Cerebellum (low-level control)
- Memory modules
- Sensorimotor interface

这让我联想到 human brain 的 functional architecture：cortex 做 high-level reasoning，cerebellum 做 motor coordination，hippocampus 做 memory。RynnBrain 当前主要扮演 "cortex" 角色，未来可能需要与 "cerebellum-like" policy models 组成 hierarchical system。

---

## 10. 总结

RynnBrain 是一个 systematic 的 embodied foundation model 工作，它的 key contributions 在我看来是：

1. **Unified output space** integrating text + coordinates + trajectories
2. **Chain-of-Point reasoning** interleaving textual + spatial grounding
3. **Data flywheel** leveraging human-model collaboration for 19.89M samples
4. **Comprehensive post-training** covering Nav, Plan, VLA, CoP
5. **Open release** of models, code, and benchmarks

这个工作 build on Qwen3-VL 的 strong foundation，但通过 task-specific data curation 和 architecture innovations 显著提升了 embodied capabilities。8B model 在多个 benchmark 上超越 32B 和 proprietary models 的事实，suggests data quality 和 task formulation 比 raw parameter scale 更重要。

从 research direction 看，RynnBrain 代表了 embodied AI 从 task-specific models 向 general-purpose foundation models 的 transition。未来 work 可能会 explore:
- 更 long-horizon 的 planning (e.g., hours-long tasks)
- Multi-agent embodied scenarios
- Sim-to-real transfer 的 robustness
- Integration with proprioceptive 和 tactile sensing
- Continual learning in deployed environments

**Resources**:
- Project page: [https://alibaba-damo-academy.github.io/RynnBrain.github.io](https://alibaba-damo-academy.github.io/RynnBrain.github.io)
- GitHub: [https://github.com/alibaba-damo-academy/RynnBrain](https://github.com/alibaba-damo-academy/RynnBrain)
- HuggingFace: [https://huggingface.co/collections/Alibaba-DAMO-Academy/rynnbrain](https://huggingface.co/collections/Alibaba-DAMO-Academy/rynnbrain)
- ModelScope: [https://www.modelscope.cn/collections/DAMO_Academy/RynnBrain](https://www.modelscope.cn/collections/DAMO_Academy/RynnBrain)

**相关 References**:
- Qwen3-VL: [https://arxiv.org/abs/2511.21631](https://arxiv.org/abs/2511.21631)
- GRPO/DeepSeekMath: [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
- DeepStack: [https://arxiv.org/abs/2410.02278](https://arxiv.org/abs/2410.02278)
- Interleaved MRoPE: [https://arxiv.org/abs/2510.23095](https://arxiv.org/abs/2510.23095)
- MASt3R-SLAM: [https://arxiv.org/abs/2504.12348](https://arxiv.org/abs/2504.12348)
- Hi Robot: [https://arxiv.org/abs/2502.19417](https://arxiv.org/abs/2502.19417)
- π_0: [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)
- π_0.5: [arXiv:2502.19417](https://arxiv.org/abs/2502.19417)
- StreamVLN: [https://arxiv.org/abs/2507.05240](https://arxiv.org/abs/2507.05240)
- NaVILA: [https://arxiv.org/abs/2412.04453](https://arxiv.org/abs/2412.04453)
- RoboBrain 2.0: [https://arxiv.org/abs/2507.02029](https://arxiv.org/abs/2507.02029)
- DeepEP: [https://github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
- SGLang: [https://arxiv.org/abs/2411.02735](https://arxiv.org/abs/2411.02735)
- SAM2: [https://arxiv.org/abs/2408.00714](https://arxiv.org/abs/2408.00714)
- Grounding DINO 1.5: [https://arxiv.org/abs/2405.10300](https://arxiv.org/abs/2405.10300)
- Grasp-Anything: [https://arxiv.org/abs/2405.13043](https://arxiv.org/abs/2405.13043)
- Open X-Embodiment: [https://arxiv.org/abs/2310.08864](https://arxiv.org/abs/2310.08864)
- R2R: [https://arxiv.org/abs/1711.11465](https://arxiv.org/abs/1711.11465)
- RT-2: [https://arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818)
- PaLM-E: [https://arxiv.org/abs/2303.03371](https://arxiv.org/abs/2303.03371)

希望这个深度技术解析能 help 你 build intuition about RynnBrain 的设计 philosophy 和 technical contributions。如果需要 dive deeper 到某个 specific component (e.g., CoP 的 RL training dynamics, 或 VLA 的 flow matching details)，可以继续探讨。
