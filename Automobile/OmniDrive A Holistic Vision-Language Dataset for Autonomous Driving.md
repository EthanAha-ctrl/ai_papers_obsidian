---
source_pdf: OmniDrive A Holistic Vision-Language Dataset for Autonomous Driving.pdf
paper_sha256: 6e6c55f384fda86334006869bd8721fb57037ff7daf933e34e40d9b643134fd6
processed_at: '2026-08-05T23:09:59-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 OmniDrive

好，咱们换个路子，不搞学术八股，就当在 NVIDIA cafeteria 聊天那种感觉。

---

## 一句话概括

**让 VLM 看一堆"假如我这么开会怎样"的 simulated trajectory，然后逼它用语言解释每条轨迹安不安全，从而学到 dense supervision**。然后再对比两种架构（从 LLaVA 扩 vs 从 StreamPETR 扩），发现前者更靠谱。

GitHub: https://github.com/NVlabs/OmniDrive

---

## 这篇 paper 到底想解决什么问题

### 问题 1：Expert Trajectory 是个超稀疏的监督信号

传统 end-to-end driving（UniAD、VAD 这种）训练时，每帧就一个 expert trajectory——"人类司机这么开的，你学着点"。这相当于给模型一个点，让它学一个 manifold。问题是 trajectory space 是高维的，一个点根本约束不了什么。模型学到的是"模仿表面"，不是"理解为什么"。

类比一下：教小孩骑自行车，你只给他看一个老司机的完美骑行视频，不告诉他"如果你身体太前倾会摔"、"如果转弯太急会滑"。他学到的就是 mimicry，遇到新情况就崩。

### 问题 2：Open-loop Benchmark 被ego status 搞坏了

BEV-Planner [22] 揭露一个尴尬事实：你只要把 ego vehicle 的速度、加速度喂给一个 MLP，就能在 nuScenes open-loop planning 上打败一堆 SOTA。因为 nuScenes 的 expert trajectory 大部分是"匀速直行"，你预测"继续匀速"就对了。

这就像考试题太简单，背答案就能过，根本测不出真本事。

paper Table 2 里的 Ego-MLP：L2 0.35m，collision 0.37%。看着很美，其实啥也没学。

参考：https://arxiv.org/abs/2312.03031

### 问题 3：VLM 从 2D 到 3D 的扩展没有标准答案

LLaVA 这种 2D VLM 在 image reasoning 上很强，但 driving 是 3D 问题——需要理解 multi-view、depth、spatial relation。怎么把 2D 能力扩展到 3D？两条路：
- 路 A：LLaVA 加几个 view，加点 3D position encoding，看能不能 work
- 路 B：StreamPETR 这种 3D perception 模型加个 language head，看能不能 work

paper 的 contribution 就是把两条路都试一遍，告诉你哪条更对。

---

## Counterfactual Reasoning 的核心 insight

人类司机开车时脑子里不停地跑 "what-if simulation"：
- "如果我 now 变道，那辆车会不会撞我？"
- "如果我 brake，后面那辆 truck 能不能停下来？"
- "如果我走这条 trajectory，会不会闯红灯？"

这就是 counterfactual reasoning——评估"如果做了 X，会怎样"。

OmniDrive 的 idea：**把这种 what-if thinking 显式地做成训练数据**。具体做法：

1. 对每个 scene，simulate 多种 trajectory（stop、forward、left turn、right turn、U-turn、accelerate、decelerate、constant speed）
2. 用 rule-based checklist 检查每条 simulated trajectory 是否违反 traffic rules（collision、red light、road boundary）
3. 把这些 trajectory + 后果信息喂给 GPT-4，让它生成 Q&A 解释
4. 训练 VLM 时，这些 Q&A 就是 dense supervision

**直觉**：一条 expert trajectory = 1 个 supervision 点。N 条 counterfactual trajectory = N 个 supervision 点，每个点还带语言解释。信息密度爆炸式增长。

更深的 intuition：counterfactual 相当于在 trajectory space 上做 **contrastive sampling**——告诉模型"这是对的，那些是错的，错在哪"。这比单纯模仿 expert 学到的 representation 要 robust 得多。

相关 idea 早期工作：
- LingoQA: https://arxiv.org/abs/2312.14115
- Hydra-MDP (NVIDIA 后续): https://arxiv.org/abs/2406.06978

---

## Data Pipeline 细节

这部分是 paper 最有工程价值的。我们一步步拆。

### Step 1: Key-frame Selection（避免在冗余数据上烧钱）

nuScenes 1000 个 scene，每个 scene 40 帧，大部分是 boring 的直行。直接全用太贵。作者用两阶段 K-means：

**Stage A - Semantic 聚类**：
- 提 front view 的 CLIP embedding
- K-means，取 20% cluster centers
- 保证覆盖 landmarks、traffic lights、lane markings 各种 visual diversity

**Stage B - Trajectory 聚类**：
- 对 future trajectory 做 K-means
- 取 200 cluster centers
- 覆盖 stop、forward、left、right、U-turn、accel、decel、constant speed

**Intuition**：driving scene 的"代表性"由两个正交维度决定——"看起来像什么"（semantic）和"ego 在做什么"（dynamics）。两个维度都做聚类才不会漏 corner case。

### Step 2: Counterfactual Simulation + Checklist

对每个 selected key-frame，simulate 那 8 类 trajectory，然后用 checklist 检查：

- **Object collision**：trajectory 与 nuScenes 3D detection 的最小距离
- **Road boundary collision**：用 OpenLane-v2 的 centerline 和 topology
- **Red light**：traffic light annotation

但 annotation 不全，覆盖不了所有 traffic rules。于是作者又把 trajectory 转成 high-level decision（"在 lane X 变道到 lane Y"），让 GPT-4 基于 image + high-level decision 判断是否安全。

这里有个 subtle 的 design choice：**GPT-4 直接看 raw 3D scene（image + objects + lanes）效果很差**。因为 traffic elements 多了之后，GPT-4 搞不清 spatial 关系。所以必须先抽象成 high-level decision，把 3D 几何问题转化成 language reasoning 问题。

OpenLane-v2: https://opendrivelab.com/Challenge%202024/OpenLaneV2.html

### Step 3: Caption 生成（避免 GPT-4 忽略图像细节）

作者发现一个反直觉现象：**同时给 GPT-4 喂 image 和大量 scene annotation，它会 ignore image**。solution 是先让 GPT-4 只看 image 生成 caption。

具体 trick：
- 三个前视拼成一张，三个后视拼成一张
- Prompt 要求 caption "describe position relative to ego vehicle"，不要独立描述每个 view
- 这迫使 GPT-4 从 egocentric perspective 组织语言，和下游 driving task 对齐

**Intuition**：VLM 在多模态输入时有"模态偏置"——当 text context 太丰富时它会 lazy 地只用 text，忽略 image。先做 image-only caption 强制它真的"看"。

这个 trick 在 general VLM 数据生成里也通用。LLaVA-NeXT、ShareGPT4V 都有类似设计。

### Step 4: Q&A 生成（五类 response）

最终 Q&A 分五类，每类有不同作用：

| 类型 | 内容 | 作用 |
|------|------|------|
| Scene Description | Caption | 基础 perception |
| Attention | Simulation 找 close object + GPT-4 common sense | 学"什么重要" |
| Counterfactual | 检查 simulated trajectory 是否违规 | 核心 dense supervision |
| Decision & Planning | GPT-4 解释 expert trajectory 为什么安全 | 学"为什么对" |
| General Conversation | Object counting、color、OCR | 增强 long-tail recognition |

**关键 insight**：单一 task 训练会让 model collapse 到 shortcut。纯 planning 训练会让 language distribution 退化（Table 2 里 Omni-L‡ 那行验证了）。混合 task 迫使 model 同时学 perception、reasoning、planning，互为 regularizer。

---

## 模型架构对比

这部分我详细讲，因为这是 paper 的另一个核心 contribution。

### 共享 backbone

两个 model 都用 EVA-02-L 作为 visual encoder，提取 multi-view feature：

$$F_m \in \mathbb{R}^{N \times C \times H \times W}$$

变量说明：
- $N$：view 数量（nuScenes 6 个 camera）
- $C$：feature channel 数
- $H, W$：feature map 空间维度
- $m$ 下标：multi-view

加 3D positional encoding $P_m$ 后送入 projector。Projector 输出对齐到 LLM token 维度（LLaMA2-7B 是 4096 维）。

### Omni-L：从 LLaVA 出发

简单粗暴：把 LLaVA 的单图扩展到多图。

- $F_m$ flatten 成 token sequence
- 加 3D PE $P_m$ 区分 view（但 weight 初始化为 0）
- MLP projector 对齐到 LLM token
- 喂 LLM

**zero-init PE 的 intuition**：2D pretraining 阶段 LLM 适应的是没有 3D PE 的输入分布。如果 finetune 开始就给 random PE，input distribution 突变，pretraining knowledge 被破坏。zero-init 让 PE 渐进生效，模型 smooth transition 到 multi-view 3D 场景。

这个 trick 在 LoRA、controlnet、3D-LLM 里都有类似应用。本质是 continual learning 的 "elastic weight consolidation" 思想——别一次性扰动太大。

LoRA paper: https://arxiv.org/abs/2106.09685

### Omni-Q：从 StreamPETR 出发

灵感来自一个观察：**BLIP-2 的 Q-Former 和 StreamPETR 的 query-based detection 架构高度相似**。两者都是 sparse query + cross-attention + self-attention。能不能 unify？

公式 (1) - Self-attention 在 detection query 和 carrier query 之间：

$$
\begin{aligned}
(Q, K, V) &= ([Q_c, Q_d], [Q_c, Q_d], [Q_c, Q_d]) \\
\tilde{Q} &= \text{Multi-head Attention}(Q, K, V)
\end{aligned}
$$

变量说明：
- $Q_c$：carrier queries（继承 BLIP-2 的 learned query 思路，数量固定，负责 vision-language alignment）
- $Q_d$：detection queries（继承 StreamPETR 的 object query，负责 3D perception）
- $[\cdot]$：concatenation
- 省略了 position encoding

**物理意义**：detection query 和 carrier query 通过 self-attention 互相 exchange 信息。detection query 把 "scene 里有哪些 object、在哪" 的信息传给 carrier query，carrier query 把 language-related 语义传给 detection query。这是 information bottleneck 设计。

公式 (2) - Cross-attention 从 image feature 收集信息：

$$
\begin{aligned}
(Q, K, V) &= ([Q_c, Q_d], P_m + F_m, F_m) \\
\tilde{Q} &= \text{Multi-head Attention}(Q, K, V)
\end{aligned}
$$

变量说明：
- $Q$ 仍然是 $[Q_c, Q_d]$ 联合 query
- $K = P_m + F_m$：image feature 加 3D PE（DETR-style position injection）
- $V = F_m$：纯 image feature 做 value

之后：
- $Q_d$ → 预测 object 类别和 3D 坐标（有 perception supervision）
- $Q_c$ → MLP → LLM token 维度 → LLM 做 text generation

**Omni-Q 的核心设计**：carrier query 不直接看 image，而是通过 self-attention 从 detection query 那里"借" 3D geometric prior。这是一种 indirect 的 vision-language alignment，希望利用 3D perception 的结构化表示。

StreamPETR: https://arxiv.org/abs/2303.11926
BLIP-2: https://arxiv.org/abs/2301.12597

### 第三种范式：BEV-MLP（baseline）

为了 completeness，paper 还试了 BEV-MLP：用 LSS [36] 把 perspective feature 投到 BEV，temporal modeling 用 SOLOFusion [35]，BEV feature flatten 后喂 LLM。

结果：所有 task 上都最差。原因后面讲。

LSS: https://arxiv.org/abs/2008.05711

### 训练流程

两阶段：
1. **2D Pretraining**：完全 follow LLaVA v1.5（batch size、LR、optimizer 都一样），初始化 projector
2. **3D Finetuning**：在 OmniDrive + DriveLM 上 finetune

Hyperparameter：
- Optimizer: AdamW
- Batch size: 16
- Projector LR: 4e-4
- Visual encoder + LLM LR: 2e-5
- LR schedule: cosine annealing

**注意**：没用 BLIP-2 的 contrastive loss 和 matching loss。纯 text generation loss。简化训练目标，让模型 focus 在 language generation。

LLaVA v1.5: https://arxiv.org/abs/2310.03744

---

## 实验结果重点解读

### Table 2: nuScenes Open-loop Planning

最关键的几个数字（带 ego status 的版本）：

| Method | L2 (m) ↓ | Collision (%) ↓ | Intersection (%) ↓ |
|--------|----------|-----------------|---------------------|
| UniAD | 0.46 | 0.77 | 1.93 |
| VAD-Base | 0.37 | 0.33 | 2.47 |
| Ego-MLP | 0.35 | 0.37 | 2.93 |
| BEV-Planner++ | 0.35 | 0.34 | 3.16 |
| Omni-Q++ | 0.33 | 0.30 | 3.00 |
| Omni-L++ | 0.40 | 0.35 | 2.45 |

VLM-based 方法（Omni-L/Q）能 match 甚至超过专门设计的 planning model。这是 VLM 在 driving 上的 milestone。

但**更有意思的是不带 ego status 的对比**：

| Method | L2 | Collision | Intersection |
|--------|-----|-----------|---------------|
| Omni-L | 2.34 | 1.90 | 3.29 |
| Omni-Q | 1.98 | 3.79 | 4.59 |
| Omni-L‡ (无 Q&A) | 2.43 | 3.22 | 3.90 |

几个 observation：

**1. Omni-L 的 L2 略差但 collision 更好**：说明 Omni-L 学到的 trajectory 更"安全"，Omni-Q 的 trajectory 在 L2 metric 上贴近 ground truth 但实际会撞车。这暴露了 L2 metric 的局限——L2 低不一定 safe。

**2. Omni-Q 更容易 overfit ego status**：Omni-Q++ 加 ego status 后 L2 从 1.98 暴降到 0.33，但 Omni-L 从 2.34 降到 0.40。Omni-Q 下降更剧烈，说明它走 shortcut 更狠。原因：Omni-Q 的 language 能力弱，更倾向于利用简单 feature（ego status）做 shortcut。

**3. Q&A 数据是 regularizer**：Omni-L‡（只用 trajectory prediction 训练）collision 3.22%，Omni-L（加 Q&A 数据）collision 1.90%。纯 trajectory training 让 LLM 的 distribution modeling 退化，Q&A 数据 keep model 的 language understanding alive。

### Table 3: DriveLM Benchmark

| Training Data | Accuracy | Score |
|---------------|----------|-------|
| DriveLM only | 0.60 | 0.53 |
| +OmniDrive | 0.70 | 0.56 |
| +LLaVA665k | 0.76 | 0.57 |
| +Both | 0.78 | 0.58 |

OmniDrive pretraining 贡献 +3%，主要在 Accuracy 和 Closure（counterfactual safety 判断）上。LLaVA665k 贡献 +3%，主要在 general language 能力。两者叠加 +5%（有 overlap，不完全 additive）。

### Table 5: 综合对比（最关键）

| Architecture | Counterfactual AP | AR | Language CIDEr | Col(%) | Inter(%) |
|--------------|------------------|----|----------------|--------|----------|
| Omni-L | 53.7 | 63.0 | 73.2 | 1.90 | 3.29 |
| Omni-Q | 52.3 | 59.6 | 68.6 | 3.79 | 4.59 |
| BEV-MLP | 45.6 | 49.5 | 59.5 | 4.43 | 8.56 |

**核心发现**：Language CIDEr 和 counterfactual AP/AR、planning collision rate **strong positive correlation**。

**Intuition**：LLM 的 language fluency 是 reasoning 能力的 proxy indicator。如果模型能用流畅、准确的语言描述 scene 和 reasoning，说明它真"理解"了。Language 不流畅的模型即使 metric 上"蒙对"，也是 shortcut learning，遇到 OOD 就崩。

这个 finding 对未来 VLM driving model 设计有重要启示：**不要为 perception accuracy 牺牲 language capability**。Omni-Q 就是反面教材——加了 3D perception supervision，但 language foundation 被削弱，结果综合表现更差。

### BEV-MLP 为什么最差

BEV-MLP 的 Safe Recall 只有 17.3%（Table 4），基本无法识别"safe scene"。原因：

1. **Pretraining gap**：BEV feature 没经过 language-aligned pretraining（CLIP、LLaVA pretrain），LLM 看不懂
2. **Spatial structure 丢失**：BEV feature 是 dense spatial grid，flatten 成 sequence 后 LLM 没有 inductive bias 去 reconstruct spatial relation
3. **Sequence length 爆炸**：BEV feature 通常 100x100 以上，flatten 后 token 数太多，LLM 处理不了

正确做法可能是：
- Sparse query（像 Omni-Q 那样用 detection query）
- BEV patch tokenize（像 ViT 那样切 patch）
- Cross-attention with BEV（不 flatten，用 cross-attention 访问）

相关：
- 3D-LLM: https://3d-llm.github.io/
- VILA: https://arxiv.org/abs/2312.09185

---

## 个人 Insight 和延伸联想

### 1. Counterfactual Reasoning 本质是 Trajectory Space 的 Contrastive Learning

传统 imitation learning 是 positive-only learning——只有 expert demonstration。Counterfactual reasoning 引入 negative samples + explanation，相当于在 trajectory space 上做 contrastive learning，而且每个 negative sample 都有 language caption 说明"为什么 negative"。

这和 RLHF 有点像：RLHF 用 human preference 做 contrastive，OmniDrive 用 simulator outcome 做 contrastive。两者都是把 sparse reward 转成 dense supervision。

参考：
- RLHF: https://arxiv.org/abs/2203.02155
- DPO: https://arxiv.org/abs/2305.18290

### 2. 为什么 Omni-L > Omni-Q：Foundation 的力量

Paper 的实验结论"2D VLM → 3D 扩展比 3D Perception → Language 扩展更容易"。这个结论可能有 deeper reason：

**Language capability 是 VLM 的 foundation，perception 是 add-on**。原因：
- LLM 通过 pre-training 学到了大量 world knowledge（"car 会动"、"red light 要停"、"行人脆弱"），这些是 driving reasoning 的 prior
- 3D perception supervision 只能教 model "object 在哪"，不能教 "object 会做什么"
- 没有 language foundation 的 model 即使知道 object 在哪，也不知道这个 object 对 driving 意味着什么

类比：教一个 5 岁孩子认路上的车 vs 教一个会说话的成人认路上的车。前者要先建立"什么是危险"的概念体系，后者只需要补充"这个 specific 场景里什么是危险"。

这个 insight 对 embodied AI general 都成立。RT-2、PaLM-E、3D-LLM 都验证了：保留 LLM foundation + 加 modality alignment 比 从 perception 模型加 language 更 work。

RT-2: https://arxiv.org/abs/2307.15818
PaLM-E: https://arxiv.org/abs/2303.03378

### 3. Omni-Q 的失败是不是 architecture 问题

我觉得 Omni-Q 的失败可能不是 architecture 本质问题，而是 implementation 选择问题：

- 用 detection query 做 3D perception supervision，本身没有错
- 问题是 carrier query 和 detection query 之间的 information flow 太弱（只通过 self-attention 一次交互）
- 如果让 carrier query 也直接 cross-attention image feature，可能能保留更多 language capability

后续工作 LMDrive、EMMA (Waymo) 可能在这个方向上改进。

EMMA: https://arxiv.org/abs/2410.23262
LMDrive: https://arxiv.org/abs/2407.12935

### 4. Closed-loop 是终极方向

Paper 自己承认 limitation：counterfactual simulation 不考虑其他 agent 反应。这意味着：

- 如果 simulated trajectory 会撞到一辆车，但那辆车实际会刹车避让，simulation 错标 collision
- 如果 simulated trajectory 看起来 safe，但其他车会突然 cut-in，simulation 错标 safe

根本解决需要 closed-loop simulator。NVIDIA 自己的 NVSim、CARLA-based Bench2Drive 是方向。Hydra-MDP（NVIDIA 后续工作）已经在 closed-loop 上探索。

Bench2Drive: https://arxiv.org/abs/2406.08845
Hydra-MDP: https://arxiv.org/abs/2406.06978

### 5. 这个 Pipeline 能 Generalize 到其他 Domain

OmniDrive 的 data generation pipeline（key-frame selection → counterfactual simulation → checklist → GPT-4 prompt → human-in-loop）是一个 general paradigm，可以迁移到：

- **Robotics manipulation**：counterfactual trajectory in robot arm motion（"如果手抓这么动会撞到杯子吗"）
- **Medical AI**：counterfactual treatment（"如果病人做了 X 治疗会怎样"）
- **Game AI**：counterfactual action（"如果 agent 选了 A 会怎样"）
- **Code generation**：counterfactual execution（"如果代码这么写会 crash 吗"）

核心 pattern：**simulator 生成 counterfactual outcome，LLM 把 outcome 翻译成 language reasoning，得到 dense supervision**。这是 LLM 时代数据生成的一个 universal recipe。

### 6. Position Encoding Zero-init 的深层含义

Omni-L 把 3D PE 初始化为 0。这不只是 training trick，背后有 optimization theory：

- 2D pretraining 后 LLM 适应 input distribution $D_1$（无 3D PE）
- 3D finetuning 引入 3D PE，目标 input distribution $D_2$（有 3D PE）
- 如果 PE 初始化 random，$D_1$ 和 $D_2$ 差异大，catastrophic forgetting
- Zero-init PE 让 $D_2$ 在初始时刻等于 $D_1$，然后渐进 drift

这和 LoRA 的 zero-init α、ControlNet 的 zero-init convolution 是同一套思想。本质是 **Newton's method 的 trust region**——小步迭代，别一次跨太大。

这个 trick 在 multi-modal continual learning 里应该会越来越重要。

ControlNet: https://arxiv.org/abs/2302.05543

### 7. Open-loop Evaluation 的根本局限

虽然 OmniDrive 缓解了 ego-status overfit，但 open-loop 本质上还是 broken：

- 没有 other agent reaction
- 没有 distribution shift（model 输出 ≠ expert 时场景会不同）
- L2 metric 不反映 safety（Omni-Q 的 L2 低但 collision 高就是证据）

未来 driving model 的 evaluation 一定是 closed-loop dominant。Tesla FSD v12、Waymo 的 closed-loop sim 都在这个方向。学术圈需要更多 Bench2Drive 这种 closed-loop benchmark。

Bench2Drive 论文：https://arxiv.org/abs/2406.08845

### 8. Counterfactual Reasoning 和 Chain-of-Thought 的关系

Counterfactual reasoning 可以看作 **CoT 的"对称版"**：

- CoT：给 input，让 model generate intermediate reasoning，然后 output
- Counterfactual：给 input 和 alternative output，让 model generate "为什么这个 output 不对"

两者都是把 reasoning 显式化，得到 dense supervision。但 counterfactual 的 supervision 更密集——每个 alternative output 都是一个 supervision point。

未来可能 unify：对每个 input，simulate 多个 output，对每个 output 都做 CoT-style reasoning，得到全方位的 reasoning supervision。

CoT paper: https://arxiv.org/abs/2201.11903

### 9. 和 LLM Reasoning 的 Scaling Law 关系

OmniDrive 的 counterfactual reasoning 数据量越大，model 的 driving reasoning 越强。这暗示一个 scaling law：

$$\text{Driving Reasoning} \sim f(\text{Counterfactual Data Size})$$

如果能 scale 到 billions of counterfactual trajectory，VLM driving 可能突破当前 bottleneck。这和 LLM reasoning 的 scaling、AlphaGo 的 self-play scaling 是同一类 phenomenon。

未来方向：用 closed-loop simulator 自动生成 billions of counterfactual trajectory，让 VLM 在 driving 上做 self-play。

AlphaGo: https://www.nature.com/articles/nature24270

---

## 总结

OmniDrive 的三个 take-away：

**1. Counterfactual reasoning 把 sparse trajectory supervision 变 dense**——simulator 生成 alternative outcome，GPT-4 翻译成 language reasoning，每个 alternative 都是 supervision point。

**2. 保留 LLM language foundation 比加 perception supervision 重要**——Omni-L > Omni-Q 说明从 2D VLM 扩展到 3D 比从 3D perception 加 language 更容易。Language CIDEr 和 reasoning/planning performance 正相关。

**3. Open-loop benchmark 仍是局限**——counterfactual 缓解 ego-status overfit，但根本解决需要 closed-loop simulator 做 reactive counterfactual。

Paper 工程实现扎实，ablation 充分，对两个 architecture paradigm 的对比非常有启发。代码开源加分。后续 Hydra-MDP、EMMA、LMDrive 都在这个方向继续推进。

最值得 follow 的方向：
- Closed-loop counterfactual reasoning（Hydra-MDP、Bench2Drive）
- VLM driving 的 scaling law（more counterfactual data → better reasoning）
- Language capability 和 perception capability 的 unified architecture（保留 Omni-L 的 language foundation + Omni-Q 的 3D supervision）

希望这个 version 更"人话"一点。核心是 build intuition：**用 what-if simulation 生成 dense language supervision，保住 LLM foundation 是关键，open-loop 还不够 closed-loop 才是未来**。

---

# OmniDrive: 用 Counterfactual Reasoning 构建 3D Driving VLM Dataset

这篇 paper 是 NVIDIA 团队的工作，第一作者 Shihao Wang，发表在 2024 年。核心 idea 是用 **counterfactual reasoning** 生成大规模高质量 driving Q&A 数据，然后对比两种 VLM 架构（Omni-L 和 Omni-Q）来探索如何将 2D VLM 能力扩展到 3D driving 场景。

GitHub: https://github.com/NVlabs/OmniDrive

---

## 1. Motivation: 为什么需要 Counterfactual Reasoning

### 1.1 既有方法的局限

传统 end-to-end autonomous driving 方法（UniAD [14]、VAD [17]）依赖 expert trajectory 作为监督信号。问题是 expert trajectory 只反映"安全驾驶的最终结果"，是 **sparse supervision**——模型只看到"对的轨迹长什么样"，看不到"为什么对"、"为什么其他轨迹不对"。

BEV-Planner [22] 和 Ego-MLP [54] 进一步揭示了一个尴尬的现象：open-loop benchmark 上 SOTA 方法的高分很多时候来自对 **ego status**（ego vehicle 的速度、加速度等）的 overfit，而不是真的理解场景。Table 2 里可以看到 Ego-MLP 这种纯 ego-status baseline 拿到 L2 0.35m / Collision 0.37% 的成绩，竟然超过 UniAD 不带 ego status 的版本。

这意味着传统 open-loop planning 评估方式存在根本问题：**模型可以不"看"就答对**。

### 1.2 Counterfactual Reasoning 的直觉

人类司机在开车时不是只考虑"我现在要做什么"，而是不断考虑"如果我现在变道会怎样"、"如果我加速会怎样"——这种"what if"思维就是 counterfactual reasoning。

OmniDrive 的核心 insight：**通过模拟多种轨迹（包括"错误"轨迹），让 VLM 理解每条轨迹的后果，从而学到密集的 supervision signal**。这比单纯让模型模仿 expert trajectory 信息量大得多——一条 expert trajectory 只有一个 supervision 点，但 N 条 counterfactual trajectory 就有 N 个 supervision 点，而且每个点都带有"为什么这样做会/不会出事故"的语言解释。

参考链接：
- UniAD: https://arxiv.org/abs/2212.10156
- BEV-Planner: https://arxiv.org/abs/2312.03031
- LingoQA (counterfactual QA 的早期工作): https://arxiv.org/abs/2312.14115
- DriveLM: https://arxiv.org/abs/2312.14150

---

## 2. Dataset Generation Pipeline 详解

### 2.1 Planning-oriented Key-frame Selection

nuScenes [3] 数据量大但有冗余——大量帧是直行、低速场景。直接全部用来生成 QA 既贵又没意义。作者用两阶段 K-means 聚类压缩数据：

**Stage 1 - Semantic 聚类**：
- 提取 front view 图像的 CLIP [39] embedding
- K-means 聚类，取 20% 的 cluster centers
- 目的：覆盖 landmarks、traffic lights、lane markings 等多样化 perceptual elements

**Stage 2 - Trajectory 聚类**：
- 对 vehicle 未来 trajectory 做 K-means
- 取 200 个 cluster centers
- 这些 centers 对应不同的 driving dynamics：stopping, moving forward, turning left/right, U-turn, accelerating, decelerating, constant speed

这里的 intuition 是：driving 数据的"代表性"由两个维度定义——**场景看起来像什么**（semantic）和**ego 要做什么**（trajectory dynamics）。两个维度都做聚类能保证下游 QA 生成覆盖到关键 corner cases。

### 2.2 Counterfactual Checklist + Prompt Design

这是 paper 最有意思的部分。作者面临一个核心 challenge：**GPT-4 直接看 3D scene（images + 3D objects + lane markings）效果不好**，因为 traffic elements 多了之后 GPT-4 无法理解它们之间的 spatial 关系。

作者的解决方案是"以 trajectory 为中心的场景表示"：

#### 2.2.1 Simulated Trajectories

从 nuScenes 全集 trajectory 聚类得到几类 canonical trajectory（停止、前进、左转、右转、U-turn、加速、减速、匀速）。每个 scene 都 simulate 这些 driving behavior，然后用 rule-based checklist 检查是否违反 traffic rules。

#### 2.2.2 Rule-based Checklist

固定类别违规用规则检查：
- Object collision：用 nuScenes 3D detection 检查 trajectory 与 object 的最小距离
- Road boundary collision：用 OpenLane-v2 [45] 的 centerline 和 road element topology
- Running red light：用 traffic light annotation

但这些 annotation 覆盖不了所有 traffic rules。所以作者还把 trajectory 转成 **high-level decision making**（比如"在 lane X 上变道到 lane Y"），然后让 GPT-4 基于 image + high-level decision 判断是否安全。

#### 2.2.3 Expert Trajectory 处理

对 nuScenes log replay 的 expert trajectory，分类成不同 high-level decision types。对于每个 object，如果在未来 3 秒内与 trajectory 的最小距离 < 10 米，标记为 "close object"，列在 expert trajectory 下面作为 context。这个设计让 GPT-4 不用自己推断"哪些 object 重要"。

#### 2.2.4 Caption 生成

作者发现一个反直觉的现象：**同时给 GPT-4 喂 image 和大量 scene info，它会忽略 image 里的细节**。所以先单独让 GPT-4 基于 multi-view image 生成 caption。

具体做法：把三个前视图像拼成一张，三个后视图像拼成一张，分别喂给 GPT-4。Prompt 要求：
1. 提到 weather、time of day、scene type 等
2. 理解每个 view 的大致方向（比如 first frontal view 是 front-left）
3. 不要独立描述每个 view 的内容，而是描述相对于 ego vehicle 的位置

这个 "describe position relative to ego vehicle" 的设计很关键——它迫使 GPT-4 从 egocentric 视角组织语言，这和下游 driving task 的 perspective 一致。

### 2.3 Q&A Generation 的五类响应

Table 1 展示了生成的 Q&A 分四类（实际是五类）：

1. **Scene Description**：直接用 caption
2. **Attention**：simulation 找 close objects + GPT-4 common sense 找 threatening elements
3. **Counterfactual Reasoning**：检查 simulated trajectory 是否违反 traffic rules（red light、collision、road boundary）
4. **Decision Making & Planning**：GPT-4 reasoning expert trajectory 为什么安全
5. **General Conversation**：multi-turn dialogue，object counting、color、relative position、OCR 等——主要是为了增强 long-tail object recognition

这种混合 task 设计的目的：**单一 task 训练会让模型 collapse 到某种 shortcut**。比如纯 planning 训练会让 LLM 的语言分布能力退化（Table 2 的 Omni-L‡ 行就证明了这一点——只训 trajectory prediction 的 Omni-L 在 planning 上反而差）。混合多种 task 迫使模型同时学 perception、reasoning、planning，避免 shortcut learning。

---

## 3. 模型架构：Omni-L vs Omni-Q

这是 paper 的另一个核心 contribution：**对比两种 VLM 架构范式**。

两者共享 visual encoder（EVA-02-L [12]，用 MIM 蒸馏 CLIP），提取 multi-view image features：

$$F_m \in \mathbb{R}^{N \times C \times H \times W}$$

变量解释：
- $N$：view 数量（nuScenes 是 6 个相机）
- $C$：feature channel
- $H, W$：feature map 的空间维度
- 下标 $m$：表示 multi-view

加上 3D positional encoding $P_m$，送入 projector，对齐到 text embedding 后喂给 LLM（LLaMA2-7B [43]）。

### 3.1 Omni-Q: 从 3D Perception 出发

灵感来自 StreamPETR [46] 的 BEV 架构 + BLIP-2 [20] 的 Q-Former 设计。核心 idea：**Q-Former 的 Transformer decoder 和 sparse query-based 3D perception 模型架构高度相似**，所以可以统一。

公式 (1) - 检测 query 和 carrier query 之间的 self-attention：

$$
\begin{aligned}
(Q, K, V) &= ([Q_c, Q_d], [Q_c, Q_d], [Q_c, Q_d]) \\
\tilde{Q} &= \text{Multi-head Attention}(Q, K, V)
\end{aligned}
$$

变量解释：
- $Q_c$：carrier queries（用于 vision-language alignment，数量固定，类似 BLIP-2 的 learned queries）
- $Q_d$：detection queries（用于 3D object detection，类似 StreamPETR 的 object queries）
- $[\cdot]$：concatenation
- $Q, K, V$ 都是由 $Q_c$ 和 $Q_d$ concat 而来，所以这是一次 self-attention
- 这里省略了 position encoding

物理意义：carrier query 和 detection query 在 self-attention 中互相 exchange 信息。detection query 告诉 carrier query "场景里有哪些 object、在哪里"，carrier query 帮 detection query 注入语言相关的语义。

公式 (2) - 从 multi-view image feature 收集信息：

$$
\begin{aligned}
(Q, K, V) &= ([Q_c, Q_d], P_m + F_m, F_m) \\
\tilde{Q} &= \text{Multi-head Attention}(Q, K, V)
\end{aligned}
$$

变量解释：
- $Q$ 仍然是 $[Q_c, Q_d]$（carrier + detection queries）
- $K = P_m + F_m$：image feature 加上 3D position encoding
- $V = F_m$：只用 image feature
- $P_m$ 加到 $K$ 上是 DETR-style 的 position injection

之后：
- $Q_d$（detection queries）→ 预测 object 类别和 3D 坐标（有 3D perception supervision）
- $Q_c$（carrier queries）→ MLP 对齐到 LLM token 维度（4096 维，LLaMA2-7B）→ 喂给 LLM 做 text generation

**Omni-Q 的关键设计**：carrier query 通过 self-attention 与 detection query 交互，间接获得了 3D position encoding 提供的 geometric priors 和 query-based representation 的结构化信息。这是一种**让 language 端"借用" 3D perception 结构化表示**的方式。

### 3.2 Omni-L: 从 2D VLM 出发

Omni-L 就是 LLaVA [25] 的多视图扩展。简单粗暴：

- 把 multi-view image feature $F_m$ flatten
- 加上 3D positional encoding $P_m$ 区分不同 view（但为了 training stability，$P_m$ 的 weight 初始化为 0）
- 通过 MLP projector 对齐到 LLM token 维度
- 喂给 LLM

这里有个小细节：3D position encoding 初始化为 0 是为了 training 稳定。如果一开始就给 random init 的 PE，模型可能适应不过来多视图 + 3D PE 的组合。从 0 开始让模型渐进学习。

### 3.3 两种范式的本质对比

| 维度 | Omni-L | Omni-Q |
|------|--------|--------|
| 出发点 | 2D VLM (LLaVA) | 3D Perception (StreamPETR) |
| Projector | MLP | Q-Former |
| 3D 信息注入 | 通过 3D PE + flatten multi-view | 通过 detection query + 3D PE + perception supervision |
| 训练目标 | 纯 text generation loss | text generation + 3D perception loss |
| 优势 | 保留 LLaVA 的 2D reasoning 能力 | 有显式 3D geometric supervision |

直觉上：Omni-L 像"会看图的 LLM 多了几个眼睛"，Omni-Q 像"会做 3D detection 的模型多了语言头"。**前者保 language 能力，后者保 perception 能力**。实验会告诉我们哪个更对。

### 3.4 BEV-MLP Baseline

paper 还提到一个 ablation baseline：BEV-MLP，用 LSS [36] 把 perspective feature 转 BEV，temporal modeling 用 SOLOFusion [35]，BEV feature 经过 MLP 后喂 LLM。这是第三种范式——**BEV feature + MLP**，结果上比 Omni-L/Q 都差。

### 3.5 训练策略

两阶段：
1. **2D Pretraining**：和 LLaVA v1.5 一样的 data + strategy，初始化 projector
2. **3D Finetuning**：在 OmniDrive + DriveLM 上 finetune

Hyperparameters：
- Optimizer: AdamW [29]
- Batch size: 16
- Projector LR: 4e-4
- Visual encoder + LLM LR: 2e-5
- LR schedule: cosine annealing

注意：**没用 BLIP-2 的 contrastive learning 和 matching loss**，只算 text generation loss。这是有意为之——简化训练，让模型聚焦在 language generation 上。

---

## 4. 实验结果深度解析

### 4.1 nuScenes Open-loop Planning (Table 2)

关键 observations：

**(1) Ego Status 的 overfit 问题**：
- Omni-Q 不带 ego status: L2 1.98m, Collision 3.79%
- Omni-Q++ 带 ego status: L2 0.33m, Collision 0.30%

加 ego status 后 L2 暴降 6 倍。但 Omni-Q 比 Omni-L 更容易 overfit ego status（collision rate 反弹）——因为 Omni-Q 的 language 能力弱，更容易走 shortcut。

**(2) 不带 ego status 时的真实能力**：
- Omni-L: L2 2.34m, Collision 1.90%, Intersection 3.29%
- Omni-Q: L2 1.98m, Collision 3.79%, Intersection 4.59%

Omni-L 的 L2 略差但 collision 和 intersection 都好很多。说明 **Omni-L 学到的 trajectory 更"安全"**，Omni-Q 的 trajectory 在 metric 上贴近 ground truth 但实际上会撞。

**(3) 纯 trajectory prediction 训练的问题**：
- Omni-L‡ (不用 Q&A 数据): Collision 3.22%, Intersection 3.90%
- Omni-L (用 Q&A 数据): Collision 1.90%, Intersection 3.29%

只用 trajectory prediction 训练，language model 的 distribution modeling 能力会退化。**OmniDrive Q&A 数据起到了 regularizer 作用**，让模型保持 language understanding 的同时学 planning。

**(4) 与 SOTA 对比**：
- UniAD (带 ego status): L2 0.46m, Collision 0.77%
- VAD-Base (带 ego status): L2 0.37m, Collision 0.33%
- Omni-L++ (带 ego status): L2 0.40m, Collision 0.35%

VLM-based 方法能 match 甚至超过专门设计的 planning 模型。这是 VLM 在 driving 上的一个 milestone。

### 4.2 DriveLM Benchmark (Table 3)

DriveLM [41] 包含 696 scenes、4072 samples、~0.3M image-question pairs，覆盖 perception、prediction、planning、behavior。Score = 0.4 × GPT + 0.2 × Language + 0.2 × Match + 0.2 × Accuracy。

| Training Data | Acc. | Score |
|---------------|------|-------|
| DriveLM only | 0.60 | 0.53 |
| +OmniDrive | 0.70 | 0.56 |
| +LLaVA665k | 0.76 | 0.57 |
| +Both | 0.78 | 0.58 |

观察：
- OmniDrive pretraining 单独贡献 +3% score
- LLaVA665k +3%
- 两者叠加 +5%（不是完全叠加，有 overlap）

OmniDrive 在 DriveLM 上的提升主要来自 Accuracy（0.60 → 0.78）和 Closure（0.07 → 0.15）。说明 counterfactual reasoning 数据帮助模型理解 driving-specific reasoning，特别是 close-loop style 的 safety 判断。

### 4.3 Counterfactual Reasoning Ablation (Table 4)

四个 task 的 Precision / Recall：Safe、Red Light、Collision、Drivable Area。

**Architecture 对比**（不带 ego status）：
- Omni-L: Safe P=72.1 R=58.0, Collision P=34.3 R=71.3
- Omni-Q: Safe P=70.7 R=49.0, Collision P=32.3 R=72.6
- BEV-MLP: Safe P=70.2 R=17.3 ← recall 极差

BEV-MLP 的 Safe recall 只有 17.3%，几乎不能识别"safe"场景。这说明 **BEV feature 直接喂 LLM 存在 pretraining gap**——BEV feature 没有经过 language-aligned 预训练，LLM 看不懂。

**Perception Supervision 对比**（Omni-Q 上的 ablation）：
- Full: Collision P=32.3 R=72.6
- No Lane: Collision P=31.0 R=56.7
- No Object & Lane: Collision P=30.0 R=53.2

去掉 lane supervision 后 collision recall 从 72.6 降到 56.7。**lane supervision 对 collision 预测很重要**——这符合直觉，因为 collision 本质上是 trajectory 与 lane boundary 的关系。

### 4.4 综合对比 (Table 5)

| Architecture | Counterfactual AP | AR | Language CIDEr | Col(%) | Inter(%) |
|--------------|------------------|----|----------------|--------|----------|
| Omni-L | 53.7 | 63.0 | 73.2 | 1.90 | 3.29 |
| Omni-Q | 52.3 | 59.6 | 68.6 | 3.79 | 4.59 |
| BEV-MLP | 45.6 | 49.5 | 59.5 | 4.43 | 8.56 |

**核心发现**：Language CIDEr 和 counterfactual AP/AR、open-loop Col/Inter **正相关**。

直觉解释：
- LLM 的语言能力是 foundation——它决定了模型能否正确 interpret scene 和 reasoning
- Omni-L 保留了 LLaVA 的语言能力，所以 reasoning 和 planning 都好
- Omni-Q 牺牲了语言能力换 3D perception supervision，但 perception 帮助有限，反而 language foundation 被削弱
- BEV-MLP 完全没有 language-aligned 预训练，foundation 最差

**结论**：**从 2D VLM 扩展到 3D 比从 3D perception 加 language 头更容易**。这是 paper 的核心 take-away。

---

## 5. Limitations 和 Future Work

作者承认：**counterfactual simulation 不考虑其他 agent 的反应**。这意味着如果 simulated trajectory 会撞到一辆车，但那辆车实际上会刹车避让，simulation 会错误标记为"collision"。

未来的方向是用 closed-loop simulator（比如 CARLA [9] 或 NVSim）来做 counterfactual simulation，这样可以模拟其他 agent 的 reactive behavior。这也是 VADv2 [5]、MILE [13] 等 closed-loop 工作的方向。

参考：
- VADv2: https://arxiv.org/abs/2402.13243
- MILE: https://arxiv.org/abs/2210.07129
- Hydra-MDP (NVIDIA 后续工作): https://arxiv.org/abs/2406.06978

---

## 6. 个人 Insight 和延伸联想

### 6.1 Counterfactual Reasoning 的本质

这个 idea 的本质是 **data augmentation for reasoning**。传统 driving data 只有 expert trajectory，相当于只有一个 positive example。Counterfactual reasoning 通过 simulation 创造了大量 "near-miss" 和 "would-be-accident" examples，让模型学到**轨迹空间的几何结构**而不只是单个 trajectory point。

类比：这就像 teaching a child "不要碰热炉子"——你不需要真的让 ta 烫到，但需要让 ta 理解"如果碰了会怎样"的 counterfactual。VLM 通过语言推理这些 counterfactual，相当于在 mental simulation 中学习。

### 6.2 Omni-L vs Omni-Q 的更深思考

Paper 的实验结论是 Omni-L > Omni-Q。但这个结论可能 **不能 generalization 到所有 3D task**。原因：

- OmniDrive 的 task 主要是 reasoning + planning，这些 task 偏 language-heavy
- 如果是纯 3D detection、tracking task，Omni-Q 的 perception supervision 应该更有优势
- 真正理想的架构可能是 **Omni-L + 部分 Omni-Q**——保留 LLaVA 的 language foundation，同时引入 detection query 做 explicit 3D grounding

后续的 LMDrive (https://arxiv.org/abs/2407.12935)、DriveMLM [48]、EMMA (Waymo 的多模态 driving model) 都在这个方向上探索。

### 6.3 与 Reason2Drive、DriveVLM 的对比

- **Reason2Drive** [34]：chain-based reasoning，但没有 counterfactual
- **DriveVLM** [42]：CoT design，关注 thinking process
- **LingoQA** [33]：最早引入 counterfactual questions，但只在 2D image 上
- **OmniDrive**：第一个把 counterfactual reasoning 系统化用到 3D trajectory 上

OmniDrive 的独特价值在于**3D trajectory-level counterfactual**——不只是问"如果这辆车加速会怎样"，而是问"如果 ego vehicle 走这条 simulated trajectory 会怎样"，这需要 3D geometric understanding。

### 6.4 数据生成 Pipeline 的 Generalization

OmniDrive 的 pipeline (key-frame selection → counterfactual simulation → checklist → GPT-4 prompt → human-in-loop) 是一个 **scalable 自动化数据生成 paradigm**。这个 paradigm 可以 generalize 到：

- **Robotics manipulation**：counterfactual trajectory in robot arm motion
- **Medical imaging**：counterfactual "如果病人做了 X 治疗"
- **Game AI**：counterfactual "如果 agent 选择了 action A"

核心思想是：**用 simulator 生成 counterfactual outcomes，用 LLM 把 outcomes 翻译成 language reasoning，得到密集 supervision**。这是 LLM 时代数据生成的一个 general pattern。

### 6.5 与 Hydra-MDP 的联系

paper reference [23] 提到的 Hydra-MDP 是 NVIDIA 后续的工作。它把 multi-target hydra-distillation 引入 end-to-end planning，每个 trajectory 都有多个 metric（collision、comfort、efficiency 等）的 evaluation。OmniDrive 的 counterfactual reasoning 可以看作是 Hydra-MDP 的 "language version"——Hydra-MDP 用 numeric score，OmniDrive 用 language reasoning。

结合起来：未来可能有 **Hydra-MDP 风格的多目标 counterfactual reasoning**——对每条 candidate trajectory 都做 language reasoning，得到多维度评估。

### 6.6 BEV-MLP 失败的深层原因

BEV-MLP 在所有 task 上都最差（Table 5）。这不只是 "pretraining gap"——更深层的原因是 **BEV feature 是 dense spatial representation**，而 LLM 是 sequence model。把 BEV feature flatten 喂给 LLM 会丢失 spatial structure，而且 LLM 没有 spatial inductive bias 去理解 BEV coordinate。

正确的做法可能是：
1. **Sparse query**: 像 Omni-Q 那样用 detection query（已经在 StreamPETR/DETR3D 验证有效）
2. **Tokenize BEV**: 像 VILA / 3D-VLA 那样把 BEV 切 patch tokenize
3. **Cross-attention with BEV**: LLM 通过 cross-attention 访问 BEV feature，而不是 flatten

### 6.7 Open-loop Benchmark 的根本局限

paper 在 limitation 里诚实地承认了 open-loop 的局限。但更激进地说，**nuScenes open-loop planning 评估本身可能是有害的**——因为它鼓励模型 overfit ego status。OmniDrive 的 counterfactual reasoning 一定程度上缓解了这个问题（因为 model 必须真的"看"场景才能回答 counterfactual QA），但根本解决需要 closed-loop benchmark。

NVIDIA 自己的 NVSim、Waymo 的 Motion Dataset、CARLA-based benchmark（如 Bench2Drive https://arxiv.org/abs/2406.08845）是未来方向。

### 6.8 Language CIDEr 和 Planning Performance 的相关性

Table 5 的发现"language CIDEr 与 counterfactual AP/AR 正相关"非常重要。这意味着：

**VLM 的 language generation quality 不是表面功夫，而是 reasoning 能力的 indicator**。

直觉：如果一个模型能用流畅、准确的语言描述场景和推理，说明它真的"理解"了。反之，language 不流畅的模型即使能在 metric 上"蒙对"，也是 shortcut learning。

这对未来 VLM driving model 的设计有重要启示：**不要为了 perception accuracy 牺牲 language capability**。Omni-Q 的失败 case 就证明了这一点。

### 6.9 Position Encoding 初始化为 0 的玄机

Omni-L 把 3D positional encoding 权重初始化为 0。这是个有意思的细节。从 optimization 角度：

- 如果 PE 初始化为 random，模型在 2D pretraining 后突然引入 3D PE，会让 LLM 的 input distribution 突变，破坏 pretraining knowledge
- 初始化为 0 意味着 3D finetuning 开始时 PE 完全不起作用，模型 behave like vanilla LLaVA
- 随着训练 PE 渐进生效，LLM 渐进学习利用 multi-view 3D 信息

这是 **continual learning 领域的"zero-init" 技巧**，类似 LoRA 中的 zero-init α。参考：
- LoRA: https://arxiv.org/abs/2106.09685
- 类似的 PE handling 在 LLaVA-NeXT、3D-LLM 也有

---

## 7. 总结：OmniDrive 的 Three Take-aways

1. **Counterfactual Reasoning 是 driving VLM 的 dense supervision 方案**——比 expert trajectory-only 的 sparse supervision 信息量大得多，能学到 "why" 而不只是 "what"。

2. **2D VLM → 3D 扩展比 3D Perception → Language 扩展更容易**——保留 LLM 的 language foundation 是关键，perception supervision 的边际收益不如 language foundation 重要。

3. **Open-loop benchmark 仍是局限**——counterfactual reasoning 缓解了 ego-status overfit，但根本解决需要 closed-loop simulator 做 reactive counterfactual。

paper 的实验设计严谨，ablation 充分，对两个 architecture 范式的对比非常有启发。代码开源在 https://github.com/NVlabs/OmniDrive 是个加分项。

后续值得 follow 的工作：
- Hydra-MDP: https://arxiv.org/abs/2406.06978
- DriveMLM: https://arxiv.org/abs/2312.09245
- EMMA (Waymo): https://arxiv.org/abs/2410.23262
- Bench2Drive: https://arxiv.org/abs/2406.08845
- 3D-LLM: https://3d-llm.github.io/
- LMDrive: https://arxiv.org/abs/2407.12935
