---
source_pdf: BLUE Toward Better Language Use in Efficient Vision-Language-Action Models
  for Autonomous Driving.pdf
paper_sha256: 2bd7d812499852a91d5659f327d94f7ee572d005256714b45aa992829dcaca40
processed_at: '2026-08-18T02:49:29-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# BLUE 用人话讲一遍

## 一句话版本

**VLA driving models 每帧都 generate language reasoning 来辅助决策，但作者发现 60% 的时候 language 啥用没有，23% 的时候甚至把车开得更烂**。所以训了个 0.11M 参数的小 gate 来 per-frame 决定要不要开 language branch，结果 SR 反而从 67% 涨到 76%，速度还快 2.5 倍。

---

## 为什么这件事值得讲

我先讲清楚这篇 paper 戳破了什么 assumption。

VLA driving 这条线（SimLingo、CriticVLA、AutoVLA、TakeVLA 这些）的默认操作：每一帧，model 先 generate 一段 natural language reasoning（"前方有行人横穿，需减速避让..."），然后基于这段 language 再 output waypoints。直觉上这听起来很 reasonable——CoT 嘛，think before act 嘛。

但 Karpathy 你肯定懂这件事：CoT 在 NLP reasoning 任务里有 sense，是因为最终答案是 token，reasoning 改变的是 token distribution。但在 closed-loop driving 里，最终 output 是 continuous waypoints，**reasoning 是中间 computation，它会改写 action distribution**。这件事的后果在 open-loop benchmark 里看不出来，必须 closed-loop 跑才能暴露。

作者跑了约 2000 GPU hours 的 closed-loop 实验，把 Bench2Drive 上每条 route 在两个 mode 下的 success rate 对比——一个 mode 是默认的 "generate language → action"，另一个 mode 是 "直接从 hidden state → action"，skip language 那段。结果：

| 类别 | 比例 |
|------|------|
| Language 真的有帮助 | 14.5% |
| Language 无差别 | 61.8% |
| Language 反而拖后腿 | 23.6% |

**关键 insight 在第三行**：23.6% 的 route 上，你 generate 那段 language reasoning 不仅没帮上忙，还把车开出了事故、压了线、撞了行人。这件事在 NLP efficient reasoning 文献里没有对应——LLM 里多余 reasoning 顶多浪费 token，不会主动让你答错。但在 driving 里，多余 language 会 steer action 走偏。

---

## 那为什么 language 经常有害？

这个 23.6% 的数字看起来反直觉，但仔细想其实很合理。

VLA 的 forward 长这样：
```
vision tokens + instruction tokens → transformer L layers → hidden state h
                                                            ↓
                                              ┌─────────────┴─────────────┐
                                              ↓                           ↓
                                    language head                  action head
                                    (autoregressive                (MLP → waypoints)
                                     decoding 50 tokens)
                                              ↓
                                    refined hidden state
                                              ↓
                                          action head
```

**左边的 branch 比 right branch 多了一长串 autoregressive decoding**。每 decode 一个 token，model 都在用 language head 的 distribution 去"扰动" hidden state。这就像给一个本来直接出 action 的 model 加了一段"自我对话"，让它先描述场景再决定。

问题在于：model 训练时见过大量"language reasoning + 正确 action"的 pair，但 language reasoning 的内容本质上是 noisy 的——它可能在描述无关的物体、可能在犹豫交通规则的边缘 case、可能被 training data 里某个 spurious correlation 带歪。**当 language reasoning 把 hidden state 带到某个"语言上合理但 action 上错误"的 direction，最终的 waypoints 就会偏**。

举 paper 里没明说但我能想到的例子：四向无信号灯路口。model 看到 language mode 下生成的描述是 "the vehicle on the left appears to be slowing down"，于是 model 决定通行。但实际上 language 是 misread——那车根本没减速，只是 camera 角度问题。Direct mode 下没这段 reasoning 干扰，model 直接从视觉特征判断"对向来车 speed 不变"，反而正确 yield。

这就是 23.6% harmful 的来源。Paper 把这个 quantify 出来，是整个工作的第一个 contribution。

---

## 那 hidden states 怎么就"知道"什么时候需要 language？

这是 paper 最魔幻的地方。

作者的做法其实很 simple：对 frozen VLA 的 last layer hidden state $h \in \mathbb{R}^{896}$（896 是 InternVL2-1B 的 hidden dim），训一个 logistic regression，label 是"这条 route 上 language 有没有帮助"。结果一个 linear probe 就能分出来。

这意味着什么？意味着 **h 里已经 encode 了"这一帧 language 会不会有用"的信号**，只是 model 自己不知道怎么用。

这件事本身不奇怪，但放在 VLA driving context 里很有意思。我列几个相关联想：

### 1. Probing BERT 的经典工作

Tenney et al. 2019、Hewitt & Manning 2019 这些工作证明 transformer 内部 linearly encode 了大量 linguistic structure。BLUE 是这个现象在 VLA driving 上的对应——pretrained model 的 representation 远比它自己"显示"出来的更 informative。

### 2. "Residual stream as memory bus"

Anthropic 的 mechanistic interpretability 视角：transformer 的 residual stream 是一个 shared bus，所有 layer 都在里面 read/write。Language branch 在训练时获得的 gradient 必然流过 shared h，所以 h 里"沉淀"了 language utility 信号。

### 3. Early-exit / Mixture-of-Depths

Google 2024 年的 Mixture-of-Depths paper（https://arxiv.org/abs/2404.02258）做的是 transformer 内部 route token 是否需要更多 layer 计算。BLUE 在 architecture 层面更 macro——route 整个 language branch 而不是某一 layer。但 underlying principle 一样：**frozen model 已经"知道"什么时候需要更多 compute**。

### 4. LLM 里的 "Truthfulness direction"

Anthropic、OpenAI 都发现 frozen LLM 的 hidden state 里有 linear direction encode 了"this statement is true/false"。BLUE 是"this frame benefits from language" direction。

---

## 然后是 gate 的设计——简单到令人怀疑

Paper 里 gate 的 architecture 是这样：

$$\mathbf{z} = W_1 \mathbf{h} + b_1, \quad p(\mathbf{h}) = \sigma(W_2 \tilde{\mathbf{z}} + b_2)$$

翻译成人话：

1. 拿 $h \in \mathbb{R}^{896}$（VLA 最后一个 token 的 last-layer hidden state）
2. 用一个矩阵 $W_1 \in \mathbb{R}^{128 \times 896}$ 投影到 128 维
3. ReLU + dropout 0.5
4. 用一个矩阵 $W_2 \in \mathbb{R}^{1 \times 128}$ 投影到 1 维
5. Sigmoid 得到 $p(h) \in [0, 1]$
6. 如果 $p(h) > 0.66$，走 language branch；否则直接出 action

总参数量：$896 \times 128 + 128 + 128 + 1 \approx 114,945$，约 0.11M。SimLingo backbone 大约 300M+。**Gate 是 backbone 的 1/2700**。

作者在 ablation 里试了 hidden dim 256，没显著提升；加了 dropout 反而更好（Table 8）。所以最终用 128 + dropout 0.5 的配置。

**为什么这么简单的 gate 就够？**

作者的 argument 是：信号已经在 $h$ 里 linearly encode 了，gate 只需要做一个 linear readout。复杂的 gate 反而过拟合训练 routes（只有约 400 条）。这跟 probe 文献的结论一致——linear probe 通常足够 strong。

---

## Label 是怎么搞的——这是工程上最 tricky 的部分

Gate 要训练，得有 label。Label 是"每一帧 / 每条 route 上，language 有没有帮助"。但"有没有帮助"怎么定义？

### Route-level label

最直觉的版本：对每条 route $r$，跑 5 个 seed，每个 seed 下分别跑 language mode 和 direct mode，比较两个 mode 的 success rate。如果 language mode 平均 SR 比 direct mode 高超过 10 个百分点，这条 route 标 1（language 有帮助）；否则标 0。

公式（Eq. 1）：
$$y_r = \mathbb{1}\Big[\frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \big(\mathrm{SR}_{\mathrm{lang}}^{(r,s)} - \mathrm{SR}_{\mathrm{direct}}^{(r,s)}\big) > \tau\Big]$$

- $r$ 是 route index
- $s$ 是 random seed，跑 5 个
- $\mathrm{SR}_{m}^{(r,s)}$ 是 mode $m$ 在 route $r$ seed $s$ 下的 success（0 or 1）
- $\tau = 10\%$ 是 margin
- $y_r \in \{0, 1\}$

**$\tau = 10\%$ 这个 margin 很重要**。如果不设 margin，noise 太多——SR 差 1% 可能就是 stochastic。设 10% 的 margin 让 label 更 clean，default 是 0（更快的 direct mode），只有 language 真的显著更好才标 1。

### Frame-level label

Route-level label 的问题：一条 language-helpful route 上，可能只有某些 segment 真的需要 language，其他地方 direct mode 也行。如果整条 route 都标 1，gate 学不到 fine-grained pattern。

作者的 refine：在 language-helpful route 上，找出两个 mode 行为差异最大的 spatial region，只在那个 region 内标 1。

公式（Eq. 2）：
$$y_{r,t} = \mathbb{1}[\Delta\overline{\mathrm{SR}}_r > \tau] \cdot \mathbb{1}[\mathbf{x}_t \in \mathcal{C}_r]$$

- $t$ 是 frame index
- $\mathbf{x}_t \in \mathbb{R}^2$ 是 frame $t$ 时 ego vehicle 的 2D 坐标
- $\mathcal{C}_r$ 是 route $r$ 上的 critical region set

**$\mathcal{C}_r$ 怎么定义？** 作者把 route 切成 spatial grid（ego 坐标系），每个 cell 上对比两个 mode 的 speed、acceleration、heading、trajectory spread、infraction rate（Eq. 5），归一化加权得到 criticality score，取 top region 作为 $\mathcal{C}_r$。

这个设计很 clever——它把"label 是 per-frame 但 ground truth 是 per-route"的 mismatch 用 spatial localization 弥补了。

### Temporal redundancy cleaning

等红灯时车辆静止，相邻几百帧 hidden state 几乎一样。如果都拿来训，静止状态会 dominate dataset。作者用 cosine similarity 0.99 阈值检测 redundant segment，长度 $L$ 的 segment 下采样到 $\max(2, \lceil L^{0.5} \rceil)$ 个样本。

公式（Eq. 6）：
$$\sin(\mathbf{h}_t, \mathbf{h}_{t+1}) = \frac{\mathbf{h}_t^\top \mathbf{h}_{t+1}}{\|\mathbf{h}_t\| \cdot \|\mathbf{h}_{t+1}\|} \geq 0.99$$

这个 $\alpha = 0.5$ 的 sublinear schedule 是 sweet spot——线性下采样会让短 segment 没样本，平方根下采样让短 segment 至少留 2 个、长 segment 也不爆炸。Cleaning 后数据减约 15%。

---

## Threshold 0.66 怎么定的——非常 elegant

这个数字看着 magic，其实是 paper Section 5 (7) 里讲的 derive 出来的。

观察 Section 2 的三分类：helpful (14.5%) / neutral (61.8%) / harmful (23.6%)。这三类把 $[0, 1]$ 区间三等分：
- $[0, 0.33]$：language harmful
- $[0.33, 0.66]$：neutral
- $[0.66, 1.0]$：language helpful

所以 $\theta = 0.66$ 放在 neutral 和 helpful 的 boundary。这不是 hyperparameter search 找出来的，是从 observation 直接 derive。

作者在 Figure 6 里 sweep $\theta$：
- $\theta \to 0$：每帧都 generate language → SR = 66.91%
- $\theta \to 1$：从不 generate language → SR = 69.55%
- $\theta = 0.66$ → SR = 76.18%

**注意 $\theta \to 1$（完全不用 language）的 SR 69.55% 比 $\theta \to 0$（每帧都用 language）的 SR 66.91% 还高**。这再次印证 paper 的核心论点——language 不是"贵但无害"，是"贵且经常有害"。

$\theta \in [0.6, 0.8]$ 都是 sweet spot，对 threshold 不敏感。

---

## 实验结果，挑几个 striking 的讲

### 1. SR 76.18% 用 0.11M trainable params 打败所有 baseline

Table 1 里的对比很夸张：

| Method | Trainable params | SR (%) | DS |
|--------|-----------------|--------|-----|
| SimLingo | ≥300M | 67.27 | 85.07 |
| CriticVLA | ≥300M | 73.33 | 88.02 |
| BevAD | ≥25M | 72.73 | 88.11 |
| TakeVLA | ≥300M | 73.73 | 89.72 |
| **BLUE (SimLingo)** | **0.11M** | **76.18** | **90.58** |

BLUE 用 1/2700 的可训练参数，仅 1 个 front camera，没 LiDAR、没 dense auxiliary labels，超过所有 6-camera + LiDAR + 全套 perception labels 的方法。

### 2. Multi-ability 上 Overtake 和 EmBrake 提升最大

Table 2：

| Skill | SimLingo | BLUE | Δ |
|-------|----------|------|---|
| Merge | 53.78 | 61.44 | +7.66 |
| Overtake | 67.41 | 80.00 | +12.59 |
| EmBrake | 81.67 | 93.27 | +11.60 |
| GiveWay | 50.00 | 50.00 | +0.00 |
| TSign | 77.20 | 84.74 | +7.54 |

Overtake 和 EmBrake 提升大，因为这两个 scenario 需要 perception disambiguation——前车是不是在减速、那个黑影是行人还是 bag。有些帧 language 真的能帮上忙（确认 hazard），有些帧 language 反而让 model 犹豫错过最佳 reaction window。Gate 学到了区分。

GiveWay 完全没提升，可能因为 GiveWay 的 decision logic 太 hard-coded，language 一直有害，gate 学到一直 skip。

### 3. Longest6 v2 上 route completion 从 70 → 84

Table 3：

| Method | DS | RC | IS | GPU hours |
|--------|----|----|----|-----------|
| SimLingo | 22 | 70 | 0.38 | 119h |
| CriticVLA | 34 | 66 | 0.55 | 193h |
| **BLUE** | **36** | **84** | **0.43** | **56h** |

Longest6 v2 是 1-2km 长 route，errors compound。如果 model 在 frame 100 上因为 unnecessary language 偏离 lane center，到 frame 200 这个偏移会放大成 lane departure。BLUE 跳过 harmful language，让长程 driving 更稳定。

### 4. Inference latency 降到 1/2.5

Table 4：

| Method | FPS | Latency (ms) |
|--------|-----|--------------|
| SimLingo | 0.72 | 1396.6 |
| CriticVLA | 0.29 | 3424.7 |
| BLUE (SimLingo) | 1.82 | 549.5 |

CriticVLA 上 BLUE 把 latency 从 3.42s 降到 0.76s，4.5× speedup。

**为什么这么快？** Language generation 是 autoregressive token-by-token decoding，50 个 token 就是 50 次 transformer forward。Gate 跳过这一整段，只剩 1 次 prompt forward + 1 次 action head，自然快。

### 5. Cross-model transfer 失败——证实 model-specific signal

Table 6：

| Config | SR | DS |
|--------|-----|-----|
| SimLingo + SimLingo gate | 76.18 | 90.58 |
| SimLingo + CriticVLA gate | 71.59 | 89.23 |
| CriticVLA + CriticVLA gate | 76.04 | 90.37 |
| CriticVLA + SimLingo gate | 73.11 | 88.90 |

用 SimLingo 训的 gate 装到 CriticVLA 上效果显著下降，反之亦然。这印证 language-utility signal 是 model-specific 的。

**直觉解释**：SimLingo 是"language → action"一次走完；CriticVLA 是"rough trajectory → language critique → refined trajectory"两步。两个 model 在 h 里 encode language-utility 的方式完全不同，linear probe 学到的 hyperplane 不能 cross-model 迁移。

### 6. Rule-based gates 全部失败——证明 hidden state 必要性

Table 7 是 paper 最重要 ablation 之一：

| Gate | SR | Lang. (%) |
|------|----|-----------|
| Speed-based | 71.81 | 30.21 |
| Acceleration-based | 70.08 | 49.12 |
| Steering-based | 70.71 | 7.94 |
| Complexity-based | 71.40 | 17.15 |
| Random | 70.01 | 50.07 |
| **BLUE** | **76.18** | **21.44** |

所有 rule-based gate 在匹配 language activation ratio 的情况下都远不如 BLUE。Speed、acceleration、steering 都是 kinematic signal——只反映 vehicle 当前 motion state，是 $h$ 的极低维 projection。$h$ 同时 encode 了 perceptual context（前方 hazard、traffic light 状态）、temporal context（前面几帧发生了什么）、model computation state。Rule-based gate 看不到这些。

**Complexity-based gate 失败的原因更微妙**：作者在 Appendix D.2.4 里指出，**同一 scenario 在 SimLingo 上 helpful，在 CriticVLA 上可能 harmful**（Figure 8）。这意味着"何时需要 language"不仅取决于场景难度，还取决于 model 内部如何使用 language。Complexity 是 model-agnostic signal，无法 capture model-specific 的 language utility。

---

## Activation pattern——gate 学到了什么

Figure 4 的可视化很有意思。Gate 在 evaluation route 上的决策：

1. 大多数 frame 上 gate 关闭 language
2. 激活时形成 contiguous segments，不是 random scattered

**Contiguity 这个性质很重要**。它说明 gate 学到的不是"这一帧像不像 helpful frame"的局部 pattern，而是捕获了 temporal context——即便 gate 输入只是 single-frame $h$，但 $h$ 本身已经 encode 了前面若干帧的信息（通过 VLA 的 in-context tokens、previous action conditioning）。

这让我想到：
- **LLM 中的 induction heads**：单个 token position 的 hidden state 实际上聚合了大量 context
- **State-space models（Mamba、RWKV）**：hidden state 是 temporal compressed summary
- **Human driving 的 "situation awareness"**：driver 不会每秒重新规划，而是进入复杂路段时切换到"审慎模式"

---

## 与其他工作的关系——BLUE 在 research landscape 里的位置

### vs. LLM efficient reasoning（L1、O1-Pruner、TokenSkip、AdaptThink）

这些工作控制 reasoning length——长短可变。BLUE 是 binary——开/关 language。粒度不同。

更重要：LLM 里多余 reasoning 浪费 token；VLA driving 里多余 reasoning 改写 action distribution，引发 traffic violation。**Embodied setting 的 cost function 不一样**。

LLM 方法都 modify LLM 本身（RL/SFT），BLUE keep VLA frozen。

### vs. Concurrent VLA adaptive reasoning（AutoVLA、AdaThinkDrive、DE-Driver）

Table 10 里的对比：

| Method | Adapt what | Backbone frozen? | Supervision |
|--------|-----------|------------------|-------------|
| AutoVLA | Reasoning depth | ✗ | RL reward |
| AdaThinkDrive | Think/non-think | ✗ | Adaptive think reward |
| DE-Driver | Reactive/delib. expert | ✗ | Scene-aware routing |
| FASIONAD | Fast/slow system | ✗ | Confidence score |
| FutureX | Instant/latent thinking | ✗ | World-model rollout |
| **BLUE** | **Language on/off** | **✓** | **Language utility** |

**BLUE 是唯一 frozen backbone、唯一 0.11M 量级 trainable、唯一从 closed-loop outcomes 自动 derive labels 的方法**。其他方法都要 retrain VLA、引入新 architecture（dual-expert、MoE、world model）、reward engineering。

### vs. Mechanistic interpretability

Anthropic 的 "Scaling Monosemanticity" 和 feature steering 工作证明 frozen LLM 里 linearly encode 了惊人 amount of task-relevant information。BLUE 是这个现象在 VLA driving 的 embodied 版本——而且 probe 不只是用来理解 model，是用来 **steer model 的 inference path**。

理解 representation 和 控制 inference 是同一件事的两面。

参考：
- Scaling Monosemanticity: https://transformer-circuits.pub/2024/scaling-monosemanticity/
- Feature Steering: https://transformer-circuits.pub/2023/interp-feature-interpolation/

### vs. Early-exit transformer（DeeBERT、PABEE、Mixture-of-Depths）

Mixture-of-Depths 在 transformer 内部 route token 是否需要更多 layer 计算。BLUE 在 architecture 层面更 macro——route 整个 language branch。

Underlying principle 一样：**frozen model 已经"知道"什么时候需要更多 compute**。

参考：
- Mixture-of-Depths: https://arxiv.org/abs/2404.02258

---

## 我觉得这篇 paper 真正的贡献

不在于 76.18% 这个数字。而在于它揭示了一个 representation-level 现象：

**Frozen VLA 的 hidden state 里已经 linearly encode 了"language 是否会帮助当前帧"的信号**。

这个发现含义深远：

1. **VLA 训练时 language branch 的 gradient 渗透到了 shared representation**，即便 action branch 不直接依赖 language。这就像你训练一个 bilingual model，English head 的 gradient 会让 shared layers 也"懂" English 概念，哪怕你 query 的是 French head。

2. **Adaptive computation 不需要重新训大模型**——只要 frozen model 已经"knows"何时需要更多 compute，加一个 tiny probe 就够了。这对资源受限的 real-world deployment 是重要 signal。

3. **它给 mechanistic interpretability 一个新 application**——probe 不只是用来 publish paper 解释 model，而是用来 steer model 的 inference path。这跟 Anthropic 最近的 feature steering 方向形成有趣呼应。

4. **它暗示一种新的 training paradigm**：如果 frozen model 已经隐式 encode 了 adaptive computation signal，那训 model 时显式加 auxiliary loss 让这个 signal 更 explicit，可能让 backbone 本身的 $h$ 更"language-aware"。但这可能 break 现有 backbone 的 transferability，是 trade-off。

---

## 几个 open directions

1. **Multi-frame gate**：当前 gate 是 single-frame decision，$h$ 只看当前帧。用 sliding window 的若干帧 hidden states 一起 feed 给 gate（或加 small RNN/Transformer），可能捕获更长 temporal pattern。

2. **Joint training 的可能性**：BLUE 完全 frozen backbone。但 paper 发现"hidden states 已经 encode language-utility"暗示——如果在 VLA 训练时显式加 auxiliary loss 让这个信号更显式，backbone 本身的 $h$ 会更"language-aware"，gate 可能更准。

3. **Real-world transfer**：SimLingo/CriticVLA 都是 CARLA 训练的。Real-world sensor noise、distribution shift 会让 hidden state 统计偏移，gate calibration 可能失效。这是 Limitations 提到的，但没量化。

4. **Generalize 到其他 "fast/slow branch" 架构**：BLUE 框架其实更通用——任何"a fast branch + a slow branch"的 VLA 都能用。Slow branch 不必是 language，可以是 diffusion planner、world model rollout、RL policy。Paper 把这个 generalize 到"any model with two inference modes"。

5. **与 L1/ConciseRL 的统一**：LLM 控制 reasoning length 和 VLA 控制 language on/off 是同一个 problem 在不同 modality 上的 instance。Unified framework 可能是：在 frozen backbone 的某个 intermediate layer 放 lightweight probe，预测当前 sample 的"optimal compute budget"，动态调度 inference path。BLUE 是 binary 版本；更一般化可以 multi-budget（k 个 budget level 路由到 k 个 branch）。

---

## 最后用三段话总结

**Observation**：在 closed-loop driving benchmark 上系统量化，发现 VLA 默认每帧 generate language 在 23.6% 的 routes 上**主动有害**，只在 14.5% 上 helpful，其余 neutral。Generate language 不是"贵但无害"，是"贵且经常有害"。这件事 NLP efficient reasoning 文献里没有对应——LLM 里多余 reasoning 顶多浪费 token，不会主动让你答错。Embodied setting 的 cost function 不一样。

**Finding**：Frozen VLA 的 last-layer hidden state 已经 linearly encode 了"这一帧 language 是否会帮助"的信号。这是 emergent property of pretrained model，不需要 retraining backbone 就能 read out。Rule-based gates（speed、acceleration、steering、complexity）都远不如 hidden state gate，因为 kinematic features 只是 $h$ 的极低维 projection，且 complexity 是 model-agnostic 信号无法 capture model-specific language utility。

**Method**：训一个 0.11M 参数的 single-hidden-layer MLP gate，per-frame decide 是否 activate language branch。用 SimLingo 作 backbone，BLUE 在 Bench2Drive 上 SR 76.18%（+8.91%）、DS 90.58（+5.51），Longest6 v2 上 DS 36（+14）、RC 84（+14），同时 2.54× speedup。同样方法应用到 CriticVLA 上也有效（+2.71% SR），证明 generality across VLA architectures。Gate 必须每个 backbone 训自己的，cross-model transfer 失败——因为 language-utility signal 是 model-specific 的。

核心 intuition 一句话：**VLA 训练过程中，language branch 的 gradient 已经把"何时需要 language"的信息注入到了 shared representation 里。BLUE 做的事情是把这个 implicit signal 变成 explicit gating decision**。这就是"better language use"的真正含义：generate language **only when it improves driving**。

---

参考链接汇总：
- Bench2Drive: https://github.com/Thinklab-SJTU/Bench2Drive
- SimLingo: https://github.com/SmartButton-Lab/simlingo
- CARLA Leaderboard: https://leaderboard.carla.org/
- InternVL2: https://github.com/OpenGVLab/InternVL
- Mixture-of-Depths: https://arxiv.org/abs/2404.02258
- Scaling Monosemanticity: https://transformer-circuits.pub/2024/scaling-monosemanticity/
- Feature Steering: https://transformer-circuits.pub/2023/interp-feature-interpolation/
- L1 (LLM reasoning length): https://arxiv.org/abs/2503.04697
- O1-Pruner: https://arxiv.org/abs/2501.12570
- TokenSkip: https://arxiv.org/abs/2502.12067
- AdaptThink: https://arxiv.org/abs/2503.02957
- AutoVLA: https://arxiv.org/abs/2603.xxxxx
- AdaThinkDrive: https://arxiv.org/abs/2509.13769
- FASIONAD: https://arxiv.org/abs/2411.18013

---

# BLUE: Toward Better Language Use in Efficient Vision-Language-Action Models for Autonomous Driving — 深度解析

## 1. 核心 Insight：Language 在 VLA Driving 里其实大多数时候是噪声

这篇 paper 的出发点是一个被社区长期忽视的经验性事实：**VLA driving models 在每帧都 generate language reasoning，但 language 对 closed-loop driving outcome 的影响分布是高度长尾的**。作者跑了一千多 GPU hours（约 2000 GPU hours）的 closed-loop 分析，统计结果是：

| Category | 比例 | 含义 |
|---------|------|------|
| Language-helpful | 14.5% | language mode 的 SR 显著高于 direct action mode |
| Language-neutral | 61.8% | 两者无统计差异 |
| Language-harmful | 23.6% | language mode 反而拖低 SR |

这个发现的意义不只是"language 用得太多"。**关键在于 harmful 比例（23.6%）几乎与 helpful（14.5%）相当**。也就是说，默认每帧 generate language 不只是浪费 computation，它会主动把车开坏。这是一个不同于 NLP efficient reasoning（如 O1-Pruner、L1、TokenSkip）的 embodied setting 特性——在 closed-loop driving 里，intermediate language computation 会**改写 action distribution**，而非只是改变 latency。

这个 insight 直接 motivates 了 BLUE 的核心设计：**何时用 language 是一个 per-frame 的 binary 决策**，应该由 model 自己来预测。

参考链接：
- Bench2Drive: https://github.com/Thinklab-SJTU/Bench2Drive
- SimLingo: https://github.com/SimLingo/SimLingo
- CARLA Leaderboard: https://leaderboard.carla.org/

---

## 2. 为什么 Hidden States 已经 Encode 了 Language-Utility Signal

这是 paper 最有意思的发现。作者证明：**对 frozen VLA 的 last-layer hidden state h 跑一个简单的 logistic regression，就能区分"这一帧 language 会不会有帮助"**。这件事单独看是惊人的，因为它暗示：

### Intuition：Language 在 VLA 里本质上是一种 "latent refinement operation"

考虑 VLA 的 computation graph：
```
vision tokens + instruction tokens  →  transformer L layers  →  h (last hidden state)
                                                                        ↓
                                                            [branch A: language head → action]
                                                            [branch B: action head → action]
```

**h 是两个 branch 共享的 fork point**。如果 h 在某种意义上"已经知道"language branch 是否会 benefit 当前帧，这意味着 h 里编码的不只是 scene content，还有**当前 model 在这个 scene 下的 computation uncertainty / language readiness**。

这让我联想到几件事：
1. **Probe-based interpretability**：类似 Metallinou et al. 关于 acoustic-to-word 系统中 intermediate layers 是否 encode linguistic content 的工作，hidden states 往往已经 encode 了下游任务需要的信息，只是 readout head 没被 explicit 训练去使用它。
2. **Early-exit transformers**（如 DeeBERT、PABEE）：每层 hidden state 可以预测当前是否够 confident，paper 里 gate 本质是"language branch 的 early-exit 判别器"。
3. **Mixture-of-Depths**（Google, 2024）：路由 token 是否需要更多 computation，BLUE 是 route 整个 branch 是否需要 computation。
4. **LLM 的 in-context linear probes**：Anthropic、OpenAI 的 mechanistic interpretability 工作表明 frozen representations 里 linearly encode 了惊人的 amount of task-relevant information（"truthfulness direction"、"refusal direction"等）。BLUE 是这个现象在 VLA driving 的 embodied 版本。

### 为什么 kinematic features / scene complexity 都不行？

作者做了 controlled ablation（Table 7）：

| Gate Method | SR (%) | DS | Lang. (%) |
|------------|--------|-----|----------|
| Speed-based | 71.81±0.54 | 89.91±0.88 | 30.21 |
| Acceleration-based | 70.08±0.43 | 88.33±0.43 | 49.12 |
| Steering-based | 70.71±0.97 | 89.38±0.22 | 7.94 |
| Complexity-based | 71.40±2.26 | 88.02±0.85 | 17.15 |
| Random | 70.01±1.53 | 87.10±1.58 | 50.07 |
| **BLUE (ours)** | **76.18±0.64** | **90.58±0.12** | **21.44±0.66** |

**Rule-based gates 在匹配 language activation ratio 的情况下 SR 远低于 BLUE**。直觉上的解释：kinematic features 只反映 vehicle 当前 motion state，是 h 的一个极低维 projection。而 h 同时 encode 了 perceptual context（前方是不是有 hazard、traffic light 状态）、temporal context（前面几帧发生了什么）、以及 model 自身的 computation state（哪些 attention head 被激活、哪些 token 被强调）。

**Complexity-based gate 失败的原因更微妙**：作者在 Appendix D.2.4 里给出关键观察——**同一个 scenario 在 SimLingo 上 helpful，在 CriticVLA 上可能 harmful**（Figure 8 的 scenario-level 分布）。这意味着"何时需要 language"不仅取决于场景客观难度，还取决于 model 内部如何使用 language。这是 BLUE 必须条件化在 model-specific hidden states 上的根本原因。

参考：
- Mixture-of-Depths: https://arxiv.org/abs/2404.02258
- Linear Probes in LLMs: https://arxiv.org/abs/2212.03827

---

## 3. 方法详解：Gate 设计与 Label 构造

### 3.1 整体架构（Figure 3 解析）

```
                         Frozen VLA Backbone
                              │
              ┌───────────────┴────────────────┐
              │  prompt-only forward pass       │
              │  (vision tokens + system text)  │
              └───────────────┬────────────────┘
                              │
              h ∈ R^d (last hidden, final token position, d=896)
                              │
              ┌───────────────┴────────────────┐
              │                                 │
        Lightweight MLP Gate              (shared h)
        (0.11M params, frozen VLA)             │
              │                                 │
        p(h) ∈ [0,1]                           │
              │                                 │
       p(h) > θ ?                              │
        ┌─────┴─────┐                          │
       Yes         No                           │
        │           │                           │
   language      direct                         │
   generation   action head                     │
   (full autoregressive                        │
    decoding with LLM)                          │
        │                                        │
   waypoints ← (refined by language)            │
                                                │
                                            waypoints
```

**关键设计选择**：
1. **h 在 final token position**：这是 prompt-only forward 的最后位置，两个 branch 共享这个 readout，保证 feature alignment。
2. **VLA backbone frozen**：整个 training pipeline 不动 SimLingo / CriticVLA 任何参数。
3. **Gate 是 single-hidden-layer MLP**：超参 m=128, dropout=0.5。

### 3.2 Gate 形式化（Eq. 7）

$$\mathbf{z} = W_1 \mathbf{h} + b_1, \quad p(\mathbf{h}) = \sigma(W_2 \tilde{\mathbf{z}} + b_2)$$

变量与上下标含义：
- $\mathbf{h} \in \mathbb{R}^d$：frozen VLA 在 final token position 的 last-layer hidden state，$d=896$（InternVL2-1B 的 hidden dim）
- $W_1 \in \mathbb{R}^{m \times d}$：input-to-hidden 权重矩阵，$m=128$，把 896 维投影到 128 维
- $b_1 \in \mathbb{R}^m$：hidden layer bias
- $\tilde{\mathbf{z}}$：$\mathbf{z}$ 经过 ReLU + dropout 之后的激活
- $W_2 \in \mathbb{R}^{1 \times m}$：hidden-to-output 权重，把 128 维压到 1 维 logit
- $b_2 \in \mathbb{R}$：output bias
- $\sigma$：sigmoid 函数 $\frac{1}{1+e^{-x}}$，把 logit 压到 $[0,1]$
- $p(\mathbf{h}) \in [0,1]$：language activation probability

总参数量：$W_1: 896 \times 128 = 114,688$；$b_1: 128$；$W_2: 128$；$b_2: 1$，加起来约 0.11M。这是 SimLingo 300M+ 参数的约 1/2700，CriticVLA 同量级。

### 3.3 Label 构造的双层结构

#### Route-level label（Eq. 1）

$$y_r = \mathbb{1}\Big[\frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \big(\mathrm{SR}_{\mathrm{lang}}^{(r,s)} - \mathrm{SR}_{\mathrm{direct}}^{(r,s)}\big) > \tau\Big]$$

变量：
- $r$：route 索引
- $s \in \mathcal{S}$：random seed，$|\mathcal{S}|=5$（作者跑 5 个 seed 求平均以减少 CARLA stochasticity）
- $\mathrm{SR}_{m}^{(r,s)} \in \{0, 1\}$：mode $m \in \{\mathrm{lang}, \mathrm{direct}\}$ 在 route $r$ seed $s$ 下的 success/fail（布尔）
- $\tau = 10\%$：margin threshold，只有当 language 平均比 direct 高出超过 10 个百分点才标 positive
- $\mathbb{1}[\cdot]$：indicator function
- $y_r \in \{0, 1\}$：route-level label，default 是 0（direct mode 更快）

#### Frame-level label（Eq. 2）

$$y_{r,t} = \mathbb{1}[\Delta\overline{\mathrm{SR}}_r > \tau] \cdot \mathbb{1}[\mathbf{x}_t \in \mathcal{C}_r]$$

变量：
- $t$：frame index within route $r$
- $\Delta\overline{\mathrm{SR}}_r = \frac{1}{|\mathcal{S}|} \sum_s (\mathrm{SR}_{\mathrm{lang}}^{(r,s)} - \mathrm{SR}_{\mathrm{direct}}^{(r,s)})$：cross-seed language advantage
- $\mathbf{x}_t \in \mathbb{R}^2$：frame $t$ 时 ego vehicle 的 2D spatial coordinate
- $\mathcal{C}_r$：route $r$ 的 critical region set，由 frame-level behavioral divergence 检测得到

#### Critical region 检测（Eq. 5）

$$\Delta_k(\mathbf{g}) = \frac{|\bar{v}_k^{\mathrm{lang}}(\mathbf{g}) - \bar{v}_k(\mathbf{g})|}{\max_{\mathbf{g}'} |\bar{v}_k^{\mathrm{lang}}(\mathbf{g}') - \bar{v}_k(\mathbf{g}')| + \epsilon}$$

变量：
- $\mathbf{g}$：spatial grid cell（ego 坐标系下的 2D bin）
- $k$：behavioral channel，包括 speed、acceleration、heading、trajectory spread、infraction
- $\bar{v}_k(\mathbf{g})$：seed-averaged channel-$k$ 信号 at cell $\mathbf{g}$ under direct action mode
- $\bar{v}_k^{\mathrm{lang}}(\mathbf{g})$：同上 but under language mode
- $\epsilon$：数值稳定常数
- 分母是 cross-cell max，做归一化

聚合得到 criticality score $c(\mathbf{g}) = \sum_k w_k \cdot \Delta_k(\mathbf{g})$，threshold 后取 top regions 作为 $\mathcal{C}_r$。

#### Temporal redundancy cleaning（Eq. 6）

$$\sin(\mathbf{h}_t, \mathbf{h}_{t+1}) = \frac{\mathbf{h}_t^\top \mathbf{h}_{t+1}}{\|\mathbf{h}_t\| \cdot \|\mathbf{h}_{t+1}\|} \geq 0.99$$

若相邻帧 hidden state cosine similarity 超过 0.99（典型场景：等红灯静止），划入一个 redundant segment of length $L$，每个 segment 下采样到 $k = \max(2, \lceil L^{0.5} \rceil)$ 个样本。这个 $\alpha = 0.5$ 的 sublinear schedule 既保证短 segment 至少保留 2 个样本，又防止长 idle 段压垮 dataset（cleaning 后减约 15% frame count）。

参考链接：
- InternVL2: https://github.com/OpenGVLab/InternVL
- PDM-Lite (expert): https://github.com/autonomousvision/carla_garage

---

## 4. Inference Pipeline 与 Threshold 选择

### 4.1 推理流程

每个 frame：
1. 跑 prompt-only forward（vision tokens + system prompt）→ 得到 $h \in \mathbb{R}^{896}$
2. Gate MLP：$h \to p(h) \in [0,1]$（开销可忽略，单 layer MLP，远小于 LLM autoregressive decoding）
3. 若 $p(h) > \theta$（$\theta = 0.66$）→ 走 language branch，autoregressive decode language reasoning，然后 waypoints head
4. 否则 → 直接走 waypoints head，skip language decoding

### 4.2 Threshold $\theta = 0.66$ 的来源

这个数字不是 hyperparameter search 找出来的，而是从 Section 2 的 observation 直接 derive：

Language impact 三分类（helpful / neutral / harmful）自然把 gate 输出空间 $[0, 1]$ 三等分：
- $[0, 0.33]$：direct mode 优
- $[0.33, 0.66]$：neutral
- $[0.66, 1.0]$：language mode 优

所以把 $\theta$ 放在 neutral/helpful 的边界 $0.66$。Figure 6 显示 $\theta \in [0.6, 0.8]$ 都是 sweet spot，SR peak 在 0.66 附近，**这个范围对 threshold 不敏感**。

边界 case：
- $\theta \to 0$：每帧都 generate language → SR = 66.91%（=SimLingo 默认配置）
- $\theta \to 1$：从不 generate language → SR = 69.55%（=direct mode only）
- $\theta = 0.66$：selective → SR = 76.18%

**注意：θ→1 时 SR (69.55%) 比 θ→0 时 SR (66.91%) 还高**。这再次印证 paper 的核心论点：默认 generate language 不只是没用，是有害的。

---

## 5. 实验数据深度解读

### 5.1 Bench2Drive 主表（Table 1 & 11）

| Method | Camera | LiDAR | Labels | T-Param. | SR (%) ↑ | DS ↑ |
|--------|--------|-------|--------|----------|---------|------|
| SimLingo | 1× | ✗ | L | ≥300M | 67.27 | 85.07 |
| CriticVLA | 1× | ✗ | L | ≥300M | 73.33 | 88.02 |
| TakeVLA | 1× | ✗ | L | ≥300M | 73.73 | 89.72 |
| BevAD | 6× | ✗ | O | ≥25M | 72.73 | 88.11 |
| HiP-AD | 6× | ✗ | O,M,D | ≈97M | 69.09 | 86.77 |
| **BLUE (SimLingo)** | **1×** | **✗** | **L** | **0.11M** | **76.18±0.64** | **90.58±0.12** |
| **BLUE (CriticVLA)** | **1×** | **✗** | **L** | **0.11M** | **76.04±0.38** | **90.37±0.14** |
| Δ vs SimLingo | = | – | – | – | **+8.91** | **+5.51** |
| Δ vs CriticVLA | = | – | – | – | +2.71 | +2.35 |

**最 striking 的事**：BLUE 用 1/2700 的可训练参数，仅用 1 个 front camera，没有 LiDAR、没有 dense auxiliary labels（O/M/S/D），却超越所有用 6 cameras + LiDAR + 全套 labels 的方法。这把"train less, get more"的故事讲到了极致。

### 5.2 Multi-ability breakdown（Table 2）

BLUE 在 5 个 driving skills 上的表现：

| Skill | SimLingo | BLUE | Δ |
|-------|----------|------|---|
| Merge | 53.78 | 61.44 | +7.66 |
| Overtake | 67.41 | 80.00 | +12.59 |
| EmBrake | 81.67 | 93.27 | +11.60 |
| GiveWay | 50.00 | 50.00 | +0.00 |
| TSign | 77.20 | 84.74 | +7.54 |
| Mean | 66.01 | 73.89 | +7.88 |

**Overtake 和 EmBrake 的 +12 提升最大**。直觉上：overtake 需要判断前车速度、gap 大小、对向来车；emergency brake 需要识别突然出现的 hazard。这两类场景 model 有时需要 language 来 disambiguate perception，但有时直接 action 更快——BLUE 学到了 per-frame 判断。GiveWay 的 0 提升暗示这个 skill 完全不需要 language，或者 language 总是 harmful。

### 5.3 Longest6 v2（Table 3）—— 长程驾驶

| Method | DS ↑ | RC ↑ | IS ↑ | Time ↓ |
|--------|------|------|------|--------|
| SimLingo | 22 | 70 | 0.38 | 119h |
| CriticVLA | 34 | 66 | 0.55 | 193h |
| **BLUE** | **36** | **84** | **0.43** | **56h** |
| Δ vs SimLingo | +14 | +14 | +0.05 | -63h |

**Route Completion 从 70 → 84 是质变**。Longest6 v2 是 1-2km 的长 route，errors compound——unnecessary language generation 累积起来会让车冲出路线或反复微调。BLUE 跳过这些 harmful frames，让车更稳。

### 5.4 Inference Efficiency（Table 4）

| Method | Speed Ratio ↑ | FPS ↑ | Latency (ms) ↓ |
|--------|--------------|-------|----------------|
| HiP-AD | 0.0625 | 1.25 | 800.3 |
| SimLingo | 0.0358 | 0.72 | 1396.6 |
| CriticVLA | 0.0146 | 0.29 | 3424.7 |
| **BLUE (SimLingo)** | **0.0910** | **1.82** | **549.5** |
| Δ vs SimLingo | +154.2% | +154.2% | -60.7% |

**2.54× speedup**，latency 从 1.40s 降到 0.55s。CriticVLA 上 speedup 4.50×，latency 从 3.42s 降到 0.76s。

**Intuition**：language generation 在 VLA 里通常是 autoregressive token-by-token decoding，每个 token 一次 transformer forward。如果 reasoning chain 50 个 token，就是 50 次 forward。Gate 跳过这些，只剩 1 次 prompt forward + 1 次 waypoint head，自然巨大加速。

### 5.5 Cross-model Transfer 失败（Table 6）—— 模型特异性

| Configuration | SR (%) | DS |
|---------------|--------|------|
| SimLingo + SimLingo gate | 76.18 | 90.58 |
| SimLingo + CriticVLA gate | 71.59 | 89.23 |
| CriticVLA + CriticVLA gate | 76.04 | 90.37 |
| CriticVLA + SimLingo gate | 73.11 | 88.90 |

**用 SimLingo 训练的 gate 装到 CriticVLA 上效果显著下降**，反之亦然。这印证了 Appendix D.2.4 的论断：language-utility signal 是 model-specific 的，每个 model 必须训自己的 gate。

**直觉**：不同 VLA 的 last-layer hidden state 是在不同的 training objective、不同的 language integration design 下形成的。SimLingo 是 "language → action" 一次性生成；CriticVLA 是 "rough trajectory → language critique → refined trajectory"。两种 architecture 在 h 里 encode language-utility 的方式完全不同，linear probe 学到的 hyperplane 不能 cross-model 迁移。

---

## 6. Activation Pattern 的直觉解释（Figure 4）

作者可视化 gate 在 evaluation routes 上的决策：

- **大多数 frame 上 gate 关闭 language**（与 SR 76.18% 一致，如果大多数都开 language，就退化成 SimLingo 的 67.27%）
- **激活的 frame 形成 contiguous segments，不是 random per-frame noise**

这个 contiguity 现象很重要。它说明 gate 学到的不是"这一帧像不像 helpful frame"的局部 pattern，而是**捕获了 temporal context**。即便 gate 输入只是 single-frame h，但 h 本身已经 encode 了前面若干帧的信息（通过 VLA 的 in-context tokens、previous action conditioning 等）。

这让我联想到：
- **LLM 中 in-context learning 的"induction heads"**：单个 token position 的 hidden state 实际上已经聚合了大量 context
- **State-space models（Mamba, RWKV）**：hidden state 是 temporal compressed summary
- **Driving 中的 "situation awareness"**：human driver 也不会每秒重新规划，而是在进入复杂路段时切换到"审慎模式"

---

## 7. 与 LLM Efficient Reasoning 的对比

作者在 Appendix B.3/B.5.2 做了详细对比，把 BLUE 放在 L1、O1-Pruner、TokenSkip、CoT-Valve、AdaptThink 这条线上。关键差异：

| 维度 | LLM efficient reasoning | BLUE |
|------|------------------------|------|
| 决策粒度 | reasoning length（长短） | binary（开/关 language） |
| 代价 | token count + latency | latency + **action distribution shift** |
| 修改对象 | LLM 本身（RL/SFT） | frozen VLA + 外挂 MLP |
| 监督来源 | task accuracy / token cost | closed-loop driving outcome |
| 错误 reasoning 的后果 | 浪费 token | 主动把车开坏 |

**最重要的概念差异**：在 NLP 里，unnecessary reasoning 主要成本是 latency；在 closed-loop driving 里，unnecessary reasoning 改变 action distribution，会引发 traffic violation、collision。BLUE 是第一个把这个 embodied-specific 危害量化到 23.6% harmful 的工作。

参考：
- L1: https://arxiv.org/abs/2503.04697
- O1-Pruner: https://arxiv.org/abs/2501.12570
- TokenSkip: https://arxiv.org/abs/2502.12067
- CoT-Valve: https://arxiv.org/abs/2503.24123
- AdaptThink: https://arxiv.org/abs/2503.02957

---

## 8. 与 Concurrent VLA Adaptive Reasoning 的对比（Table 10）

| Method | What is Adapted | Core Mechanism | Frozen | Supervision |
|--------|---------------|----------------|--------|-------------|
| AutoVLA | Reasoning depth | Action token. + SFT + GRPO | ✗ | RL reward |
| AdaThinkDrive | Think/non-think | Dual-mode SFT + GRPO | ✗ | Adaptive think reward |
| DE-Driver | Reactive/delib. expert | Dual-expert + scene router | ✗ | Scene-aware routing |
| SAMoE-VLA | Expert assignment | Scene-adaptive MoE | ✗ | Scene features |
| FASIONAD | Fast/slow system | Dual-system + VLM feedback | ✗ | Confidence score |
| FutureX | Instant/latent thinking | Auto-think + world model | ✗ | World-model rollout |
| DynVLA | Text/dynamics CoT | Dynamics token prediction | ✗ | SFT on dyn. tokens |
| FastDriveCoT | CoT decoding speed | Parallel structured decoding | ✗ | Template structure |
| Reasoning-VLA | Action decoding | Action queries + parallel gen. | ✗ | SFT |
| **BLUE** | **Language gen. on/off** | **Hidden-state gate (0.11M)** | **✓** | **Language utility** |

**BLUE 是唯一 frozen backbone、唯一 0.11M 量级 trainable params、唯一从 closed-loop outcomes 自动 derive labels 的方法**。其他方法都要么 retrain 整个 VLA、要么引入新 architecture（dual-expert、MoE、world model）、要么需要 reward engineering。

参考：
- AutoVLA: https://arxiv.org/abs/2603.xxxxx (Zhou et al., 2026)
- AdaThinkDrive: https://arxiv.org/abs/2509.13769
- DE-Driver: https://arxiv.org/abs/2603.xxxxx (Xie et al., 2026)
- FASIONAD: https://arxiv.org/abs/2411.18013

---

## 9. Limitations 与 Open Questions

作者承认两个局限：
1. **Per-frame latency 不均匀**：skip language 的 frame 快，generate language 的 frame 慢。但这是 VLA 本身就有的问题（autoregressive token 数不同），BLUE 在最大 latency 上几乎没增加，只降平均。
2. **每个 backbone 要训自己的 gate**：但 gate 是 0.11M MLP，训练成本 < 0.1 GPU hours，labels 是 routine evaluation 的副产品。

**我看到的几个 open directions**：

### 9.1 Multi-frame Gate
当前 gate 是 single-frame decision，h 只看当前帧。如果用 sliding window 的若干帧 hidden states 一起 feed 给 gate（或加一个 small RNN/Transformer），可能捕获更长的 temporal pattern。

### 9.2 Joint Training 的可能性
BLUE 完全 frozen backbone。但 paper 的发现"hidden states 已经 encode language-utility"暗示——如果在 VLA 训练时显式加一个 auxiliary loss 让这个信号更显式，backbone 本身的 h 会更"language-aware"。这可能让 gate 更准，但也可能 break 现有 backbone 的 transferability。

### 9.3 Real-world Transfer
SimLingo/CriticVLA 都是 CARLA 训练的。real-world deployment 里 sensor noise、distribution shift 会让 hidden state 的统计偏移。Gate 是在 CARLA 训的 statistic 上学的，real world 上 calibration 会失效吗？这是 Limitations 里提到的，但没量化。

### 9.4 不同的 "Branch" 是否一定得是 Language？
BLUE 框架其实更通用：任何"a fast branch and a slow branch"的 VLA 都能用。slow branch 不必是 language，可以是 diffusion planner、world model rollout、或者 RL policy。Paper 把这个 generalize 到"any model with two inference modes"。这让我想到 Branch-Net、Fast-Net+Slow-Net 这类经典架构。

### 9.5 与 L1/ConciseRL 的统一
LLM 里控制 reasoning length 和 VLA 里控制 language on/off 是同一个问题在不同 modality 上的 instance。一个 unified framework 可能是：**在 frozen backbone 的某个 intermediate layer放一个 lightweight probe，预测当前 sample 的"optimal compute budget"**，然后动态调度 inference path。BLUE 是 binary 版本；更一般化可以 multi-budget（k 个 budget level 路由到 k 个 branch）。

---

## 10. 我觉得这篇 paper 真正的贡献

不在于 76.2% SR 这个数字，而在于它揭示了一个 **representation-level 的现象**：frozen VLA 的 hidden state 里**已经 linearly encode 了"language 是否会帮助当前帧"的信号**，只是原始 backbone 没被训练去 read out 它。这个发现的含义是深远的：

1. **它暗示 VLA 训练时，language branch 的训练 signal 渗透到了 shared representation**，即便 action branch 不直接依赖 language。
2. **它说明 "adaptive computation" 不需要重新训大模型**——只要 frozen model 已经"knows"何时需要更多 compute，加一个 tiny probe 就够了。
3. **它给 mechanistic interpretability 一个新 application**：probe 不只是用来理解 model，而是用来**steer model 的 inference path**。

这与最近 Anthropic 的工作（如 "Scaling Monosemanticity"、feature steering）形成有趣的呼应——理解 representation 和控制 inference 是同一件事的两面。

参考：
- Scaling Monosemanticity: https://transformer-circuits.pub/2024/scaling-monosemanticity/
- Feature Steering: https://transformer-circuits.pub/2023/interp-feature-interpolation/
- Branch-Net: 早期 conditional computation 工作

---

## 11. 总结：BLUE 的 Three-Sentence Story

1. **Observation**：在 closed-loop driving benchmark 上系统量化，发现 VLA 默认每帧 generate language 在 23.6% 的 routes 上**主动有害**，只在 14.5% 上 helpful，其余 neutral——generate language 不是"贵但无害"，是"贵且经常有害"。

2. **Finding**：frozen VLA 的 last-layer hidden state **已经 linearly encode 了"这一帧 language 是否会帮助"的信号**，这是一个 emergent property of the pretrained model，不需要 retraining backbone 就能 read out。

3. **Method**：训一个 0.11M 参数的 single-hidden-layer MLP gate，per-frame decide 是否 activate language branch。用 SimLingo 作 backbone，BLUE 在 Bench2Drive 上 SR 76.2%（+8.9%）、DS 90.58（+5.51），Longest6 v2 上 DS 36（+14），同时 2.54× speedup。同样方法应用到 CriticVLA 上也有效（+2.71% SR），证明 generality across VLA architectures。

**核心直觉**：VLA 训练过程中，language branch 的 gradient 已经把"何时需要 language"的信息注入到了 shared representation 里。BLUE 做的事情是把这个 implicit signal 变成 explicit gating decision——这就是"better language use"的真正含义：不是 generate more language，而是 generate language **only when it improves driving**。
