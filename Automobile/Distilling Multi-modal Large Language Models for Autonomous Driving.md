---
source_pdf: Distilling Multi-modal Large Language Models for Autonomous Driving.pdf
paper_sha256: 0236b6722cc82429af08f28116d254e1620db0b2dd48f6c92e4b58714485f407
processed_at: '2026-08-18T06:01:11-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我用 coffee chat 的方式再讲一遍。

## 一句话 version

你想用 LLM 的 world knowledge 来搞自动驾驶，but LLM inference 太慢车上跑不动。所以 training 时让 LLM 当 teacher，vision model 当 student，training 完了你把 LLM 踢掉，只部署 vision model。结果 student 居然能学到 long-tail 场景的 robustness，甚至比 LLM teacher 还好。

## 为什么这个 problem 值得搞

现在 end-to-end AV 基本两条路：

- **Vision planner**：VAD、UniAD 这种，直接 camera 进、trajectory 出，inference 60ms 很快，but long-tail 烂得一塌糊涂。3-point turn 这种 maneuver 训练集里压根没有，model 直接懵掉。
- **LLM planner**：TOKEN、DriveVLM 这种，把 MLLM 拼进来做 reasoning。long-tail 上明显好，but LLaVA-7B 一次 forward 200ms+，real-time 车上根本用不了。

所以你陷入两难：要么快但脆，要么 robust 但慢。

DiMA 的 idea 是：**train 时候又快又 robust 的 I 都要，inference 时只留快的那个**。这个 asymmetric setup 其实是 knowledge distillation 的经典思路，[Hinton 2015](https://arxiv.org/abs/1503.02531) 那套搬到 multimodal AV 上。

## 核心 trick：structured token，而不是 patch token

这其实是 paper 里最关键的 insight，也是最容易被忽略的。

之前 LMDrive、DriveVLM 把 image encoder 的 dense patch embedding 喂给 LLM。问题在于 driving 任务关心的根本不是 pixel-level 细节，而是 object-level relation——前车在 brake 吗？左车要 merge 吗？

DiMA 的做法：先让 scene encoder 把 image 处理成结构化 token，叫做 **BEAM**：

- **B**EV token：鸟瞰图特征
- **E**go token：自车的一个 learnable embedding，专门学自车跟其他东西的 interaction
- **A**gent token：周围车辆（cross-attention 出来）
- **M**ap token：地图元素（lane graph 等）

然后把这四个 token 序列喂给 LLM。等于告诉 LLM："这是 ego，这是 agent1、agent2，这是 map，你帮我 reason 它们之间关系"。

这个设计跟 BLIP-2 的 Q-former 思路一脉相承——都是把 dense visual 压成 compact structural token 给 LLM。只不过 DiMA 压得更狠更结构化，因为 driving 任务天然就有 object decomposition。

参考 [BLIP-2 paper](https://arxiv.org/abs/2301.12597) 和 [Q-former 实现](https://github.com/salesforce/LAVIS)。

## Teacher 给 student 布置的作业

MLLM teacher 在 training 时被要求做四件事：

### 1. VQA（常规操作）

拿 [DriveLM](https://github.com/OpenDriveLab/DriveLM) 的 QA pair 训练，问题类似 "前车在干嘛？ego 该怎么走？" 标准 next-token cross-entropy loss。

### 2. Masked reconstruction

随机 mask 掉 BEV token 的一部分（masking ratio 0.2-0.4），让 LLM 根据剩下 token 重建。

$$\mathcal{L}_{recon} = \|\hat{B} - B\|^2$$

- $B$：原始完整 BEV token
- $\hat{B}$：LLM penultimate layer 出来过个 head 重建的 BEV

intuition：如果 LLM 能重建被 mask 的空间区域，说明 BEAM token 在 LLM 内部确实编码了空间信息，而不只是 superficial pattern。

借鉴的是 [MAE](https://github.com/facebookresearch/mae) 的思路。

### 3. Future BEV prediction

让 LLM 预测 $t+1$、$t+2$ 时刻的 future BEV token：

$$\mathcal{L}_{future} = \|\hat{F}_t - B_{t+1}\|^2 + \|\tilde{F}_t - B_{t+2}\|^2$$

- $\hat{F}_t, \tilde{F}_t$：当前时刻预测的两个 future BEV embedding
- $B_{t+1}, B_{t+2}$：ground-truth 的未来 BEV

这个 task 强迫 LLM 学 spatio-temporal dynamics。Planning 本质就是预测未来 + 做决策，所以让 LLM 显式学 dynamics 是 indirect 但有效的 supervision。

### 4. Scene editing（最有意思的）

这个 task 是 paper 最有创意的地方。流程是：

1. 给一个 scene，**人为地在里面加一辆车或删一辆车**（通过 linear layer 创建 new agent token）
2. 构造一个对应 QA："if a car is added to the left, what should ego do?"
3. 把 modified BEAM + QA 喂给 LLM
4. 用一个专门的 head 预测 ego 在 modified scene 下的 trajectory
5. 用 collision loss 监督

intuition：这本质上是 **counterfactual reasoning**。Standard supervised learning 学的是 correlation——见过的配置对应的 trajectory。Long-tail 是没见过的配置。Scene editing 强迫 model 学 "if agent 突然出现在 X，ego path 该怎么变" 这种 causal 关系。

这跟 [CRIS](https://arxiv.org/abs/2310.07284)、[WorldDreamer](https://arxiv.org/abs/2404.09511) 这类 world model 思路有点像——学 dynamics 而非 memorize pattern。

## Feature distillation：把 teacher 的脑子复制给 student

最后一个 trick，让 vision planner 的 penultimate layer feature 跟 LLM penultimate layer 的 ego token feature 对齐：

$$\mathcal{L}_{distill} = D_{KL}(P_{llm} \| P_{vis})$$

- $P_{vis}$：vision planning transformer 倒数第二层 ego token feature 分布
- $P_{llm}$：MLLM 倒数第二层对应 ego token 的 hidden feature 分布

注意只 distill ego token，不 distill 整个 sequence。这也合理——最终 planning 关心的就是 ego motion，distill ego-specific representation 足矣。

## 训练流程

两阶段：

1. **Stage 1**：单训 vision planner（VAD 或 UniAD）60 epochs，让 scene encoder 先学好基础 representation
2. **Stage 2**：joint training 30 epochs，MLLM 部分用 [LoRA](https://github.com/microsoft/LoRA) fine-tune LLaVA-7B

为啥要 LoRA？full fine-tune 7B 在合理 GPU 上不可行。AdamW，lr = 2e-4，cosine annealing，weight decay 0.01。

## 实验结果，看几个关键数字

### 整体 nuScenes validation

| Method | Traj L2 (m) ↓ | Collision (%) ↓ |
|--------|--------------:|----------------:|
| VAD-Base baseline | 0.78 | 0.30 |
| PARA-Drive (SOTA vision) | 0.56 | 0.17 |
| TOKEN (LLM-based, inference 要 LLM) | 0.68 | 0.15 |
| **DiMA (VAD-Base)** | **0.47** | **0.06** |

VAD-Base 经过 DiMA 训练后，trajectory error 降 40%，collision 降 80%。而且 inference 不需要 LLM，比 TOKEN 又快又准。

### Long-tail 上才是真正差异

**3-point turn（zero-shot，training 里压根没见过）：**

| Method | L2 (m) ↓ | Collision (%) ↓ |
|--------|---------:|----------------:|
| VAD-Base | 1.57 | 0.00 |
| PARA-Drive | 1.29 | 5.33 |
| TOKEN | 1.18 | 4.00 |
| **DiMA (VAD-Base)** | **1.05** | **0.00** |

VAD 在这里崩了（1.57m 误差），TOKEN 借 LLM 好一点但 collision 4%。DiMA 不用 LLM inference 就拿到 1.05m / 0 collision。

这数字说服力很强：training data 里没见过的 maneuver，LLM 通过 general reasoning 理解了 "3-point turn 是什么"，distill 给了 vision planner。

参考 [nuScenes benchmark](https://www.nuscenes.org/) 和 [TOKEN paper](https://arxiv.org/abs/2407.00959)。

### Ablation，看每个 trick 贡献多少

VAD-Tiny baseline 0.60m → full DiMA 0.38m，约 37% 提升。拆开看：

- 加 VQA + LLM planning，只用 BEV token：0.62（**反而略升**！说明 LLM 没足够信息）
- 加上 Map token：0.56
- 加上 Agent + Ego token（all BEAM）：0.52
- 加 distillation：0.48
- 加 masked recon：0.42
- 加 future pred：0.39
- 加 scene editing：0.38

最 interesting 的是 ID-1 → ID-2 这步——单纯把 LLM 拼上来不 work，性能还略降。这反直觉但合理：LLM 拿到一堆 BEV token 不知道是啥，反而干扰了 vision planner。直到你给它 structured all-BEAM token，LLM 才真正发挥 reasoning 价值。

## 跟其他方法的关系

- **vs TOKEN**：TOKEN 用 frozen scene encoder 做 tokenizer，DiMA 让 scene encoder 和 MLLM joint train。结果 DiMA 不用 LLM inference 还更好。
- **vs DriveVLM**：DriveVLM inference 要 LLM，DiMA 不用，且性能更好。
- **vs EMA (Waymo)**：思路类似都强调 structured representation 给 LLM，但 EMA 没 distillation。

参考 [DriveVLM](https://arxiv.org/abs/2402.12289) 和 [EMA](https://arxiv.org/abs/2410.23262)。

## 我觉得哪些地方不够好

1. **只 open-loop eval**：nuScenes 是 open-loop，predicted trajectory 不会影响后续 scene。Closed-loop（CARLA / [NuPlan](https://www.nuscenes.org/nuplan)）才是真实 driving。Open-loop 上 trajectory L2 跟 closed-loop 表现经常脱节，[AD-MLP paper](https://arxiv.org/abs/2305.10430) 已经指出过这点。
2. **Scene editing 增强方式简单**：只加 car/truck，没考虑 pedestrian 突发鬼探头、cyclist 突然变道这种真正危险的 long-tail。
3. **Long-tail 上没单独 ablation**：只 general validation 上做了 ablation。Scene editing 在 long-tail 上贡献应该更大，但 paper 没量化。
4. **nuScenes 6Hz 太慢**：高速场景 dynamic modeling 不充分。

## 我的 takeaway

DiMA 让我觉得 "obvious in hindsight" 的工作。核心 idea——asymmetric training, teacher student distillation——很经典。但 execution 上的几个 design choice 才让它真正 work：

- Structured BEAM token（不 dense patch）
- Scene editing counterfactual training
- Penultimate layer ego-only distillation
- Two-stage training

这种 "training 用 expensive model 提升 representation，inference 用 cheap model" 的范式我觉得会在很多 edge AI 场景被借鉴——robotics manipulation、medical imaging、长尾 detection 都可以套。

参考 [VAD repo](https://github.com/hustvl/VAD)、[UniAD repo](https://github.com/OpenDriveLab/UniAD)、[PARA-Drive paper](https://arxiv.org/abs/2406.06377)、[LMDrive paper](https://arxiv.org/abs/2312.07388)、[OmniDrive paper](https://arxiv.org/abs/2405.01533)。

下一个值得 follow 的方向：把 DiMA 套到 closed-loop（NuPlan / CARLA）上验证 robustness 是否真的 transfer。Open-loop 上的 trajectory L2 好看，但 closed-loop 上能不能不掉链子才是 AV 真正关心的。

---

我来详细讲讲 DiMA 这篇 paper。这是 Qualcomm AI Research 和 JHU 合作的工作，解决的是 autonomous driving 里一个非常实际的问题：**如何在不付 inference cost 的前提下，把 MLLM 的 world knowledge 灌进 vision-based planner**。

## 1. 问题 motivation：为什么需要 DiMA

先建立整个领域的 landscape。当前 end-to-end autonomous driving 大致有两条路线：

**Vision-based planner**（如 UniAD、VAD、PARA-Drive）：
- 直接从 multi-view image 学 latent scene representation，输出 trajectory waypoint
- Inference 快（VAD-Tiny 大约 60ms latency，16.8 FPS）
- 但在 long-tail scenario（3-point turn、overtake、resume from stop）上 brittle，因为这些 case 在 training data 里稀少

**LLM-based planner**（如 TOKEN、DriveVLM、OmniDrive、LMDrive）：
- 用 MLLM 做 reasoning 和 planning
- 借助 LLM 在 internet-scale 数据上学到的 world knowledge，泛化到 rare event 更好
- 但 inference cost 极高 - 7B LLM 一次 forward 至少几百 ms，无法满足 real-time AV 的要求

DiMA 的核心 question 是：**能不能训练时让 MLLM 当 teacher，inference 时只用 vision-based planner**？这本质上是 knowledge distillation 思路在 multimodal AV 上的应用。

我个人觉得这个 motivation 非常 sound。AV 不能接受 200ms+ 的 planning latency，但又需要 long-tail robustness。Distillation 是合理解法。

参考：
- nuScenes benchmark: https://www.nuscenes.org/
- VAD repo: https://github.com/hustvl/VAD
- UniAD repo: https://github.com/OpenDriveLab/UniAD

## 2. DiMA 框架整体架构

整个框架有两个主分支，共享一个 scene encoder：

```
Multi-view image sequence + question text
                │
        ┌───────┴───────┐
        │  Scene Encoder │  (shared, trainable)
        │  → BEAM tokens  │
        └───┬─────────┬───┘
            │         │
   ┌────────┘         └────────┐
   │                           │
   ▼                           ▼
┌──────────────┐         ┌──────────────────┐
│ Planning     │         │ Q-former adapters│
│ Transformer  │         │ (Map/BEV/Ego/Agent)│
│ → waypoints  │         └────────┬─────────┘
└──────────────┘                  │
                                  ▼
                          ┌───────────────┐
                          │   MLLM (LLaVA) │
                          └───┬───┬───┬───┘
                              │   │   │
                          ┌───┘   │   └───┐
                          ▼       ▼       ▼
                       VQA    Planning  Surrogate tasks
                       head   head      (recon/future/edit)
```

关键设计点（这点很重要）：
- **Scene encoder 既服务 vision planner 也服务 MLLM**，充当 MLLM 的 trainable tokenizer
- 这跟 TOKEN 不同。TOKEN 用 frozen PARA-Drive scene encoder 做 tokenizer，而 DiMA 让 scene encoder 和 MLLM 联合训练
- 这意味着 BEAM representation 会同时被 planning loss 和 language grounding loss 优化，更 grounded

参考：
- LLaVA: https://github.com/haotian-liu/LLaVA
- BLIP-2 (Q-former 来源): https://github.com/salesforce/LAVIS

## 3. BEAM Token Embedding：structured 输入的核心

这是 paper 的一个重要 insight。作者 argue 之前 MLLM-based AV 方法（如 LMDrive、DriveVLM）把 dense unstructured image patch token 喂给 LLM，这种方式在 driving 场景下 suboptimal。Driving 需要的是 explicit 的 scene decomposition。

BEAM = **B**EV + **E**go + **A**gent + **M**ap：

| Token | 来源 | 作用 |
|-------|------|------|
| B (BEV) | Visual backbone feature → BEV projection (类似 BEVFormer) | 鸟瞰图空间特征 |
| E (Ego) | 随机初始化的 learnable embedding | 学习 ego 与其他 component 的 interaction |
| A (Agent) | BEV feature × agent query (cross-attention) | 周围车辆的结构化表示 |
| M (Map) | BEV feature × map query (cross-attention) | 地图元素（lane、road graph 等） |

这个 design 让 LLM 看到的是 explicit 分解过的 scene，类似于告诉 LLM "这是 ego、这是 agent1、agent2、这是 map"，而不是甩一堆 patch embedding 让 LLM 自己 figure out。

Intuition 上来想，这跟 Flamingo、BLIP-2 之于 general VQA 不同 - general VQA 视觉细节是关键，driving 更关心的是 object-level relation reasoning。

## 4. MLLM 监督任务：四个 task head

MLLM 被训练做四类任务，每个都有具体目的：

### 4.1 Visual Question Answering (VQA)

用 DriveLM 的 QA pair（perception / prediction / planning / behavior 四类），监督信号是 standard next-token cross-entropy：

$$\mathcal{L}_{LLM} = -\sum_t \log p(y_t | y_{<t}, \text{BEAM}, q)$$

其中 $y_t$ 是 answer 的第 $t$ 个 token，$q$ 是 question embedding，BEAM 是 projected scene token。这部分让 LLM 学会基于 structured scene input 做语言推理。

### 4.2 Masked Token Reconstruction

借鉴 MAE (Masked Autoencoder) 思路。对 BEV token 做随机 masking（masking ratio 0.2-0.4），让 LLM 根据剩下 context 重建 masked BEV：

$$\mathcal{L}_{recon} = \|\hat{B} - B\|^2$$

变量含义：
- $B \in \mathbb{R}^{N \times d}$：原始的 BEV token embedding（完整未 mask 的）
- $\hat{B}$：reconstruction head（用 LLM penultimate layer 输出）预测的 masked 位置 BEV
- $\|\cdot\|^2$：L2 norm 平方

这个 loss 的 intuition：让 LLM 不仅理解语言，还要把视觉 grounding 学好 - 如果 LLM 能重建被 mask 的 BEV region，说明 BEAM token 在 LLM 内部确实编码了 spatial 信息。

参考 MAE: https://github.com/facebookresearch/mae

### 4.3 Future BEV Prediction

让 LLM 预测 $t+1$ 和 $t+2$ 时刻的 future BEV token embedding：

$$\mathcal{L}_{future} = \|\hat{F}_t - B_{t+1}\|^2 + \|\tilde{F}_t - B_{t+2}\|^2$$

变量含义：
- $\hat{F}_t, \tilde{F}_t \in \mathbb{R}^{N \times d}$：基于当前时刻 $t$ 的 BEAM token，由 future prediction head 预测的两个未来时刻 BEV embedding
- $B_{t+1}, B_{t+2}$：未来时刻 ground-truth BEV token（由 scene encoder 对未来 frame 产生）

这个 task 强迫 LLM 学 spatio-temporal 动态 - 这正是 planning 需要的能力。我觉得这个设计很聪明：直接让 LLM 在 token 空间预测未来 BEV，相当于让 LLM 学 implicit dynamics model。

### 4.4 Scene Editing（novelty 最大的一个）

这个 task 我觉得是 paper 最有创意的部分。流程是：
1. 给定当前 scene，**人为地 add 或 remove 一个 agent**（通过 linear layer 创建 new agent token）
2. 配合一个对应 QA pair（"if there is a car to the left, what should ego do?"）
3. 把 modified BEAM + text prompt 喂给 LLM
4. 用 scene editing decoder head 预测 modified ego trajectory
5. 用 $\mathcal{L}_{scene}$ = ego-agent collision constraint loss（在 modified scene 上）监督

这本质上是一种 **counterfactual reasoning** 训练 - "如果 scene 里多/少一个 agent，ego 应该怎么走？" 强迫 model 学到 agent 对 ego path 的因果影响，而不只是 correlation。

这点我觉得特别 valuable，因为 long-tail scenario 本质上就是 training data 里没见过的 agent 配置。通过 scene editing 数据增强 + counterfactual loss，model 学到的是 agent-ego interaction 的 generic 规律。

## 5. Feature Distillation Loss

为了让 vision planner 的 penultimate layer feature 对齐 MLLM 的对应 feature：

$$\mathcal{L}_{distill} = D_{KL}(P_{llm} \| P_{vis})$$

变量含义：
- $P_{vis}$：vision planning transformer penultimate layer 的 ego token feature 分布
- $P_{llm}$：MLLM penultimate layer 对应 ego token 的 hidden feature 分布
- $D_{KL}$：KL divergence

注意这里只 distill ego token 对应的 feature，不 distill 整个 sequence。我觉得这是合理的 - 最终 planning 关心的是 ego 未来的 motion，distill ego-specific representation 就够了。

总 loss：

$$\mathcal{L} = \mathcal{L}_{planning} + \mathcal{L}_{LLM} + \mathcal{L}_{recon} + \mathcal{L}_{future} + \mathcal{L}_{distill}$$

权重设成让各 loss scale 一致。

## 6. 训练 pipeline

两阶段：
1. **Stage 1**：只训练 vision-based planner（VAD 或 UniAD）60 epochs，学 informative scene representation
2. **Stage 2**：joint training vision planner + MLLM 30 epochs，MLLM 部分用 LoRA fine-tune LLaVA-v1.5-7B

LoRA 选择是因为 full fine-tune 7B LLM memory 不可行。AdamW optimizer + cosine annealing，lr = $2 \times 10^{-4}$, weight decay = 0.01。

参考 LoRA: https://github.com/microsoft/LoRA

## 7. 实验结果分析

### 7.1 Standardized Evaluation（PARA-Drive 提出的 fair eval）

在 nuScenes validation set 上：

| Method | Traj L2 avg (m) ↓ | Collision avg (%) ↓ |
|--------|------------------:|--------------------:|
| UniAD | 0.83 | 0.40 |
| VAD-Base | 0.78 | 0.30 |
| PARA-Drive | 0.56 | 0.17 |
| TOKEN (LLM-based) | 0.68 | 0.15 |
| **DiMA (VAD-Base)** | **0.47** | **0.06** |
| DiMA+ (VAD-Base) | 0.46 | 0.06 |

DiMA 相对 VAD-Base：
- Trajectory L2 error: 0.78 → 0.47，约 **40% reduction**
- Collision rate: 0.30 → 0.06，**80% reduction**

更惊人的是 DiMA 不需要 LLM inference，但比需要 LLM 的 TOKEN 还要好。这意味着 distillation 把 LLM 的有用 knowledge 真的转移过去了。

### 7.2 Long-tail Scenarios（最具说服力的实验）

Table 3 的 long-tail 结果是 paper 最 strong 的 evidence。三个 long-tail scenario：

**3-point turn（zero-shot，training data 中没有）：**

| Method | L2 avg (m) ↓ | Collision (%) ↓ |
|--------|-------------:|----------------:|
| VAD-Base | 1.57 | 0.00 |
| PARA-Drive | 1.29 | 5.33 |
| TOKEN | 1.18 | 4.00 |
| **DiMA (VAD-Base)** | **1.05** | **0.00** |
| DiMA-Dual (VAD-Tiny) | 1.04 | 0.00 |

这是 zero-shot scenario！VAD 在这里表现很差（1.57m），TOKEN 借助 LLM 提升到 1.18m，但 collision 还有 4%。DiMA 不用 LLM inference 就达到 1.05m 且 0 collision。这个 case 完美展示了 MLLM world knowledge distillation 的价值 - training data 里没见过的 maneuver，LLM 通过 general 推理理解了 "3-point turn 是什么"，distill 给了 vision planner。

**Overtake 场景：**
- VAD-Base: 1.06m / 2.49% collision
- TOKEN: 0.74m / 0.00% collision
- DiMA (VAD-Base): 0.66m / 1.29% collision
- DiMA-Dual: 0.67m / 1.30% collision

注意 DiMA 这里 collision 1.30%，比 TOKEN 的 0% 略差，但 trajectory error 显著低。这个 trade-off 在 AV 里其实合理 - 路径更准确但偶尔贴边 vs 路径保守但偏大。

### 7.3 DiMA-Dual：hybrid inference 选项

paper 还提出 DiMA-Dual，inference 时同时跑 vision planner 和 MLLM，max-pool 两边 penultimate layer 的 ego feature，再 feed back 到两边 head：

$$F_{pooled} = \max(P_{vis}, P_{llm})$$

然后 $\hat{F}_{pooled}$ 同时输入到 vision planning head 和 MLLM planning head。

这种 ensemble 思路让 DiMA-Dual (VAD-Tiny) 的性能达到 DiMA (VAD-Base) 水平（Table 2: 0.29 vs 0.29 L2 avg），latency 286ms / 3.5 FPS。这个 latency 对 real-time 不够，但作为可选 mode 在 safety-critical 场景可以接受。

### 7.4 VQA Performance

paper 给的 qualitative example 显示 DiMA MLLM 能正确回答 DriveLM-style question，例如：

Q: "What are the important objects in the scene?"
DiMA: "There is a traffic cone, a car, and two barriers in the scene. The traffic cone is to the left of the car..."

与 GPT-4 的对比显示，DiMA 对 driving-specific 关系理解更准确，GPT-4 有时给泛泛描述或错误推断（Figure 13 row 4，GPT-4 建议 ego 直行，但实际应该右转）。

参考 DriveLM: https://github.com/OpenDriveLab/DriveLM

## 8. Ablation Study 深度分析（Table 4）

这个 ablation 是 build intuition 的关键，我逐 row 解读：

| ID | VQA | Scene tokens | Distill | LLM Plan | Surrogate tasks | Traj L2 avg ↓ |
|----|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | ✗ | ✗ | ✗ | ✗ | ✗ | 0.60 (VAD-Tiny baseline) |
| 2 | ✓ | BEV only | ✗ | ✓ | ✗ | 0.62 (略升！) |
| 3 | ✓ | BEV+Map | ✗ | ✓ | ✗ | 0.56 |
| 4 | ✓ | All BEAM | ✗ | ✓ | ✗ | 0.52 |
| 5 | ✓ | All BEAM | ✓ | ✓ | ✗ | 0.48 |
| 6 | All + recon | | | | | 0.42 |
| 7 | + future pred | | | | | 0.39 |
| 8 | + scene edit | | | | | 0.38 (full DiMA) |

关键 takeaways：

1. **ID-1 → ID-2**：单纯加 VQA + LLM planning，用 BEV only，性能反而略升（0.60→0.62）。这说明单纯把 LLM 拼上来不 work - LLM 没足够信息。
2. **ID-2 → ID-3 → ID-4**：逐步加 Map、Agent、Ego token，性能持续提升（0.62 → 0.56 → 0.52）。这是 paper 核心 insight 的强证据 - structured multi-component scene token 是关键。
3. **ID-4 → ID-5**：加 distillation loss，0.52 → 0.48。Distillation 单独贡献约 8%。
4. **ID-5 → ID-6 → ID-7 → ID-8**：逐步加 surrogate task，每个都贡献 1-3 cm。Scene editing 贡献相对小（0.39 → 0.38），但这是 VAD evaluation 下；long-tail 上 scene editing 应该更重要（paper 没单独 ablate long-tail 上的 scene editing 贡献，有点遗憾）。

整体从 baseline 0.60 降到 full DiMA 0.38，约 37% improvement，跟 paper claim 一致。

## 9. 跟其他方法的关系对比

### vs. TOKEN

TOKEN（同一时期工作，Stanford/USC）：
- 用 PARA-Drive 的 frozen scene encoder 做 tokenizer
- 训练时 LLM 看到 structured token
- inference 时需要 LLM

DiMA 优势：
- Scene encoder 与 MLLM joint training，representation 更 grounded
- inference 不需要 LLM
- 实测性能更好（long-tail 上 DiMA 1.05 vs TOKEN 1.18 on 3-point turn）

### vs. DriveVLM

DriveVLM 用 MLLM + VAD dual-branch，inference 也需要 LLM。DiMA 在 VAD eval 上 outperform DriveVLM-Dual (VAD-Base) (0.27 vs 0.31 L2 avg)，且 latency 更优。

### vs. EMA (Waymo)

EMA 是 waymo 最近的工作，用 end-to-end multimodal model。DiMA 没直接对比 EMA（数据集不同），但思路有相似 - 都强调 structured representation 给 LLM。

参考 EMA: https://arxiv.org/abs/2410.23262

## 10. 我的思考与 intuition

### 10.1 为什么 distillation 在这里 work？

Knowledge distillation 经典做法是软标签 distill（Hinton 2015）。这里 distillation 之所以 work，我直觉是：
- MLLM 通过 VQA + surrogate task，在 BEAM representation 上学到了 language-grounded 的语义结构
- 这种语义结构通过 distillation loss + shared scene encoder 渗透到 vision planner 路径
- 在 long-tail 场景，training data 里 vision planner 没直接见过，但 MLLM 通过 LLM pre-training 见过类似语义概念（"3-point turn" 文本描述过），所以能引导出合理 BEAM feature

### 10.2 为什么 structured token 比 dense patch token 更好？

这点直觉上：driving 任务的关键 reasoning 是 object-level relation（"前车要 brake"、"左车要 merge"），而不是 pixel-level 细节。Dense patch token 让 LLM 浪费 capacity 去 reconstruct 视觉细节；structured BEAM 把 visual preprocessing 已经做完，LLM 直接拿到 object-centric representation，能专注于 reasoning。

这跟 Flamingo 在 general VQA 上的成功不矛盾 - general VQA 的 reasoning 经常依赖 fine-grained visual（"图里有几只猫"），但 driving 不需要。

### 10.3 Scene editing 的 counterfactual intuition

这是 paper 最有思想深度的设计。Standard supervised learning 学到的是 correlation - training data 里见过的 agent-ego 配置。Long-tail 是没见过的配置。

Scene editing 强迫 model 学："如果 agent 突然出现在位置 X，ego 路径该如何变化？" 这是 causal query。通过大量这种 counterfactual training，model 学到的不是 "this configuration → this trajectory" 的 lookup，而是 "agent influences ego via this mechanism" 的因果模型。

这跟最近 CRIS/CycleWorld 等 world model 思路呼应 - model driving dynamics 而非 memorize pattern。

### 10.4 Limitations 我看到的

- Open-loop evaluation only：nuScenes 是 open-loop，predicted trajectory 不会影响后续 scene。闭环 evaluation（CARLA, Waymo Sim）更真实
- 静态场景：nuScenes 6Hz 频率低，高速场景 dynamic modeling 不充分
- Scene editing 增强方式相对简单（只 add/remove car/truck），没考虑 pedestrian/cyclist 突发行为
- Surrogate task 间相对独立，没探索它们之间是否冗余
- Long-tail 上 ablation 缺失（只在 general validation 上做 ablation）

### 10.5 可扩展性直觉

这种 distillation paradigm 本质上适用于任何"训练时用昂贵 model 提升 representation，推理时用便宜 model"的场景。可能的延伸：

- Robotics manipulation：训练时用 VLM 做 reasoning，部署时用轻量 policy network
- Medical imaging：训练时用 large radiology LLM，部署时用 small CNN
- 长尾 detection：把 GPT-4V 的 reasoning 能力 distill 给 small detector

这种 "asymmetric train/inference" 范式在 edge deployment 场景特别有价值。

参考：
- Knowledge Distillation Hinton 2015: https://arxiv.org/abs/1503.02531
- PARA-Drive paper: https://arxiv.org/abs/2406.06377
- TOKEN paper: https://arxiv.org/abs/2407.00959
- DriveVLM paper: https://arxiv.org/abs/2402.12289
- OmniDrive paper: https://arxiv.org/abs/2405.01533
- LMDrive paper: https://arxiv.org/abs/2312.07388
- VAD paper: https://arxiv.org/abs/2303.12077
- UniAD paper: https://arxiv.org/abs/2212.10156
- nuScenes paper: https://arxiv.org/abs/1903.11027
- BEVFormer: https://arxiv.org/abs/2203.17270
- Chain-of-Thought: https://arxiv.org/abs/2201.11903
- LLaMA 3: https://arxiv.org/abs/2407.21783
- LoRA: https://arxiv.org/abs/2106.09685
- MAE: https://arxiv.org/abs/2111.06377
- BLIP-2: https://arxiv.org/abs/2301.12597

## 11. 总结

DiMA 是个让我觉得 "obvious in hindsight" 的工作。它的核心 idea 简单清晰：把 MLLM 当 teacher，vision planner 当 student，用 shared structured tokenizer + carefully designed surrogate tasks + feature distillation 实现 knowledge transfer，inference 时甩掉 teacher。

但 execution 上有很多 careful design：
- BEAM structured token（不是 dense patch）
- Q-former adapter 分 component
- Scene editing counterfactual training
- Penultimate layer feature distillation (ego only)
- Two-stage training（先训 vision planner，再 joint）

结果在 long-tail 上确实令人信服，特别是 3-point turn 这个 zero-shot case。这种 asymmetric train/inference 范式我觉得会在很多 edge AI 场景被借鉴。

下一篇值得 follow 的方向：把这套思路用在 closed-loop evaluation（CARLA/NuPlan）上，看 distill 出来的 robustness 在 closed-loop 里是否还能维持。Open-loop 上的 trajectory L2 有时跟 closed-loop 表现脱节（这点 AD-MLP paper 已经指出过），所以 closed-loop 验证很重要。

希望这个解读对你 build intuition 有帮助！如果你想 deep dive 某个具体部分（比如 Q-former 细节、scene editing 算法、或者与某个 baseline 的更细对比），告诉我，我可以再展开。
