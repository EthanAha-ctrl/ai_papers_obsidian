---
source_pdf: Empowering Vision-Language-Action Model with.pdf
paper_sha256: 6fc134eb086b08a2ced09f69c1cf5e09324e29fcfd46dfe7df55e8fe1b84d80c
processed_at: '2026-08-04T04:10:21-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 ReMem-VLA

## 这篇 paper 到底在解决什么问题

想象你让一个 robot 把杯子放回原位。Robot 拿起杯子、按了个按钮、现在要把杯子放回去——但"原位"在哪？它得回忆起几秒前看到的位置。这是个 trivial 的人类记忆任务，但当前 VLA model 几乎全做不到。

为什么？因为几乎所有 VLA (RT-2, OpenVLA, π0, π0.5, OpenVLA-OFT) 都遵循一个 Markov 假设：

$$\mathcal{A}_{t:t+k} = \pi_\theta(o_t, \ell)$$

只看当前帧 $o_t$ 加 instruction $\ell$，预测未来 action chunk $\mathcal{A}_{t:t+k} = (a_t, \ldots, a_{t+k})$。Robot 是个 amnesiac——它不知道自己刚才做了什么，不知道环境怎么变化。

作者把这个 limitation 拆成 5 种 memory dimension (Fig. 1):
- **Spatial memory**: 记住物体位置
- **Sequential memory**: 多步任务顺序
- **Episodic memory**: 记住自己执行过什么动作 (如"已经舀了两勺米")
- **Temporal memory**: 持续时间感 (如保持浇水姿态 6 秒)
- **Visual memory**: 过去的视觉细节

现有方法各有短板，作者用一张表罗列得非常清楚：

| 范式 | 代表方法 | 问题 |
|---|---|---|
| Retrieve from memory bank | MemoryVLA [45], MemER [47] | 当前帧没 cue 就 retrieve 失败；distractor 一多就乱 |
| Extend frame window | CronusVLA [32], HAMLET [30] | 固定窗口，超过就忘 |
| Sparse keyframes | TraceVLA [62], HistRISE [12] | 依赖外部 foundation model (SAM2, CoTracker) 稳定性 |
| Naive recurrence | RoboFlamingo [33], AVA-VLA [11] | 没人在 memory-dependent task 上验证过，实际不行 |

最后这条特别有意思——之前的工作加了 LSTM/recurrent state 但根本没测过记忆任务，作者直接 ablation 揭穿：naive recurrence + truncated BPTT 在 long horizon 上彻底失败。

## 核心招式：Dual-Level Recurrent Queries

### 直觉

人脑有两种记忆系统协同工作：**working memory** (前额叶, sub-second, 容量小) 和 **episodic memory** (hippocampus, minutes-to-hours, 容量大)。前者管"我现在在干什么"，后者管"我之前到过哪、做过什么"。

ReMem-VLA 模拟这个结构，用两组 learnable query token 充当 memory slot：

- **Frame-level queries** $Q^f \in \mathbb{R}^{N_f \times D}$: 每个 timestep 都更新，吃当前帧信息——这是 working memory
- **Chunk-level queries** $Q^c \in \mathbb{R}^{N_c \times D}$: 每 $k$ 步 (论文里 $k=30$) 才更新一次——这是 episodic memory

为什么这个 dual structure 必要？因为单一 update frequency 永远在 trade-off：更新快了能跟踪短期动作细节但长期信息被快速覆盖；更新慢了能保留长期 context 但短期细节丢失。Human hippocampus 与 neocortex 的 interplay 早就 solved 这个 problem，论文本质上是把这个 multi-timescale trick 翻译到 transformer query 维度。

### EMA Update: 公式 5-6

Frame-level 用指数移动平均：

$$Q_t^f = \beta_f \cdot \tilde{Q}_{t-1}^f + (1-\beta_f) \cdot Q_{t-1}^f \tag{5}$$

变量解读：
- $Q_t^f$: timestep $t$ 的 frame-level memory state
- $\tilde{Q}_{t-1}^f$: 当前帧通过 attention 从 VLM hidden states 提取的 "新写入值" (相当于 memory query 看一眼当前 frame 后输出)
- $\beta_f \in [0,1]$: EMA 系数，ablation 显示 $\beta=0.5$ 最佳
- $1-\beta_f$: retention factor

Chunk-level 在 chunk boundary 才更新：

$$Q_t^c = \begin{cases} \beta_c \cdot \tilde{Q}_{t-k}^c + (1-\beta_c) \cdot Q_{t-k}^c, & \text{if } t \bmod k = 0 \\ Q_{t-1}^c, & \text{otherwise} \end{cases} \tag{6}$$

关键 insight: chunk-level 等效时间常数是 $(1-\beta_c)^{n/k}$ vs frame-level 的 $(1-\beta_f)^n$。同样 $\beta=0.5$, $k=30$，300 步后 frame-level 衰减到 $2^{-300} \approx 0$，chunk-level 衰减到 $2^{-10} \approx 0.001$——还能保留信号。这就是多尺度结构的力量。

EMA 本质是 low-pass filter。$\beta$ 大 = filter cutoff 高 = 更新激进 = 快速忘旧；$\beta$ 小 = filter cutoff 低 = 惰性 = 难吸收新。这与 signal processing 中 IIR filter 的 pole 位置直接对应。

### Gradient-Free Recurrent Path：最 critical 的设计

这是全文最 striking 的 design choice，也是 ablation 最戏剧化的发现。

通常训 recurrent model 想学到 long-term dependency，理论上要 backprop through time 走完整个 trajectory。但 2B 参数的 VLM 这样做根本不可行。实践中只能用 Truncated BPTT (TBPTT [34])——比如 AVA-VLA 用 $T=4$，意味着 gradient 只往回传 4 步。这样 action loss 根本无法教 model 在 300 步前该记住什么。

作者的解法非常 elegant：**把 recurrent path $\mathcal{F}$ 完全冻结**。VLM 是 frozen 的，EMA 系数也是 fixed 的。Memory 怎么传播 (how) 由 deterministic forward update 保证；learning 只决定 memory query 写什么 (what)，通过 attention front-end 学。

Ablation (Fig. 6a) 在 "Put Block Back" 任务上做实验：
- Frozen VLM + fixed EMA: 完整 memory 能力
- Trainable VLM: memory 能力几乎归零
- Replace EMA with learnable GRU: memory 能力几乎归零
- Replace EMA with MLP: memory 能力几乎归零

任何引入 trainable parameter 到 recurrent path 的尝试都 destroy memory。这强烈暗示：在 large VLA + TBPTT 框架下，**learnable recurrent dynamics 与 long-term memory 不兼容**。Gradient 信号被 truncation 切断后，任何 learnable transition 都会 drift 到方便短期 loss reduction 的状态，长期 retention 信息被 silently overwritten。

这与 Hinton 的 **Forward-Forward Algorithm** [https://arxiv.org/abs/2212.13345] 和 Scellier-Bengio 的 **Equilibrium Propagation** [https://arxiv.org/abs/1602.04705] 共享一个 deep insight：把 credit assignment 从 BPTT 中解脱，让 inference-time dynamics 保证 representation 的 propagation。ReMem-VLA 可以视为这个哲学在 large VLA 上的实证。

## Architecture 全景 (Fig. 2)

5 个组件串起来：

1. **Frozen VLM backbone**: Qwen3-VL-2B [https://arxiv.org/abs/2502 (Qwen3-VL tech report)]，输出 hidden states $\mathbf{H}_t \in \mathbb{R}^{L \times D}$
2. **Learnable queries** appended 到输入序列末尾：
   - Action queries $Q^{action} \in \mathbb{R}^{N_a \times D}$：管 action 生成
   - Hindsight queries $Q^{img} \in \mathbb{R}^{N_{img} \times D}$：管 past image reconstruction (借鉴 GR-1 [https://arxiv.org/abs/2312.13139])
3. **Dual-level recurrent memory queries**: $Q^f, Q^c$——核心创新
4. **12-layer bidirectional transformer connector**: 因为 VLM 内部 causal attention 不让 $Q^{action}, Q^{img}$ 看到放在末尾的 memory queries，connector 让 4 类 query 在 latent space 充分交互
5. **Prediction heads**:
   - Action diffusion head (DDPM [https://arxiv.org/abs/2006.11239], DDIM 推理 [https://arxiv.org/abs/2010.02502])
   - ViT-style patch decoder for past image

数据流：当前帧进 VLM → 4 类 query 通过 attention 提取 features → connector bidirectional attention 让 memory 与 action/img query 融合 → action head 出未来动作 chunk，image head 出过去帧。

## 训练目标 (公式 7-9)

**Action diffusion loss**:

$$\mathcal{L}_{action} = \mathbb{E}_{\tau, \epsilon \sim \mathcal{N}(0, I)} \left[ \| \epsilon - \epsilon_\theta(\mathcal{A}_\tau, \tau, Q^{action}) \|^2 \right] \tag{7}$$

- $\tau$: diffusion timestep (跟 control timestep 是不同概念，别混)
- $\epsilon$: 添加的 Gaussian noise
- $\mathcal{A}_\tau$: noised action chunk
- $\epsilon_\theta$: denoising network 预测 noise
- $Q^{action}$: 通过 cross-attention 注入条件

**Past Observation Prediction (POP)**:

$$\mathcal{L}_{image} = \| o_{t-m} - \hat{o}_{t-m} \|_2^2 \tag{8}$$

- $o_{t-m}$: $m$ 步之前的真实 RGB frame
- $\hat{o}_{t-m}$: 重建的 past frame
- 实践中 $m$ 取 episode 第一帧最有效

**Total**:
$$\mathcal{L}_{total} = \mathcal{L}_{action} + \lambda_{img} \mathcal{L}_{image}, \quad \lambda_{img} = 0.5 \tag{9}$$

POP 的 intuition 很漂亮：让 $Q^{img}$ 通过 connector 与 memory queries 交互，强制 memory queries 携带足够 visual detail 才能重建过去帧。这是个 information bottleneck 形式的 self-supervised auxiliary loss，把视觉细节"压"进 recurrent state。

注意这里 predict past 而非 future——因为 past 是 ground truth 已知 (从 replay buffer 取)，future 是 multi-modal 难监督。这与 **Predictive Coding** 理论在大脑中的机制呼应，只不过方向反了 (大脑通常预测 future，这里 predict past 利用确定性)。

## Streaming Slot-Based Batching: 工程关键

Recurrent training over variable-length episodes 的 batching 是个工程 nightmare。传统方法把 episode 切成 fixed window，但这就破坏 temporal continuity，long-horizon context 全丢。

作者提出 slot-based streaming：维护 $B$ 个 concurrent slots，每个 slot 跟踪不同 episode 在当前 timestep。每个 training step 从每个 slot 取 1 帧，组成 batch size $B$。Slot 内 recurrent state 跨整个 episode 持续累积。Episode 边界 hard-reset state 防止 cross-episode leakage。

这个设计完美 mirror 了 **State Space Models** (Mamba [https://arxiv.org/abs/2312.00752], S4) 的 training paradigm：gradient 局部流动 (BPTT truncation = 1)，forward state 全局演化。Ablation 里 BPTT truncation 设为 1 仍然能学到 long-term memory，因为 state 通过 deterministic EMA forward 累积，不依赖 gradient flow。

## 实验数字硬核解读

### MemoryBench (Table 1)

| Method | Put Block Back | Rearrange Block | Reopen Drawer | Long Horizon (>600 frames) | Avg |
|---|---|---|---|---|---|
| OpenVLA-OFT [27] | 0 | 3 | 0 | 0 | 0.75 |
| π0.5 [22] | 6 | 20 | 3 | 4 | 8.25 |
| MemoryVLA [45] | 0 | 5 | 0 | 1 | 1.5 |
| **ReMem-VLA** | **93** | **99** | **100** | **86** | **94.5** |

最 striking 是 Long Horizon Task (>600 frames)：ReMem-VLA 86% vs MemoryVLA 1%。MemoryVLA 这种 retrieval-based 方法在 long horizon 完全失效，因为检索 query 无法 disambiguate 几百步历史中的相关 frame。ReMem-VLA 通过 chunk-level EMA 把早期信息持续 compress 到固定大小 query set，绕开了 sequence length bottleneck。

注意作者有意修改了 MemoryBench 协议：button randomization 降到 70% 避免 joint limit 干扰；Rearrange Block 强制统一 trajectory 杜绝 cue-based 求解。这是 rigorous evaluation 的体现。

### 真实世界实验 (Fig. 4)

4 个任务各 50 trials，平均 82.5% vs π0.5 11%, MemoryVLA 8%:
- Water Flower (~6s, temporal memory)
- Scoop Two Spoons Rice (episodic memory)
- Press Buttons Sequence (sequential + temporal, 还引入 disturbance 验证 closed-loop)
- Put Fruit Back (visual memory)

特别值得一提的是 Press Button 任务作者会突然挪动按钮位置，验证 model 真在做 closed-loop control 而非学了个超长 open-loop trajectory。这种 sanity check 在 robot learning paper 里太少见了，应该成为 standard practice。

### Ablation: 双层 query 必要性 (Table 2)

| Config | Avg |
|---|---|
| No Recurrent Query | 17.75 |
| Frame Level Only | 87.75 |
| Chunk Level Only | 84.5 |
| Dual Level | **94.5** |

Fig. 5a failure analysis 显示纯 frame-level 在 long horizon 上 memory-related failure 多；纯 chunk-level 在需要短期记忆的 button-press 任务失败多。两者互补性清晰可见。

### Ablation: EMA $\beta$ 与 query 数量 (Fig. 6b, 6c)

$\beta \in \{0, 0.3, 0.5, 0.7, 0.9, 1\}$，$\beta=0.5$ 最佳——retention-adaptation 平衡点。

Query 数量 $N \in \{4, 16, 32, 64, 128, 256, 512\}$，$N=128$ 最佳。太少 capacity 不足，太多 redundancy 增加 attention noise 和 optimization 难度。

### Ablation: POP 的作用 (Fig. 5b)

对 visual memory-intensive 的 Return Fruit 任务：34% → 82%，巨大提升。对其他非 visual memory 任务贡献有限。证明 POP 专门补强 visual dimension，与 recurrent query 互补。

## 我的直觉与延伸联想

### Memory Query 是 Differentiable Working Memory Slot

ReMem-VLA 的 memory query 可视为 **Differentiable Neural Computer** (DNC [https://www.nature.com/articles/nature20101]) 的轻量版。EMA update 等价于 soft write，attention read 等价于 content-based addressing。但比 DNC 简单得多——没有 explicit location-based addressing, 没有 allocation mechanism。这种简化换来 training stability 和 large-scale 可行性。

### Multi-timescale 与 Hierarchical RL

Frame-level vs chunk-level 的结构让我想到 **Options Framework** (Sutton, Precup) 和 **FeUdal Networks** (Vezhnevets et al. [https://arxiv.org/abs/1703.01161])。Chunk-level 像 manager 提供的 subgoal，frame-level 像 worker 的执行。ReMem-VLA 没有 explicit hierarchy，通过 update frequency 隐式实现 temporal abstraction。

### 与 Mamba/SSM 的潜在融合

Slot-based streaming training 与 State Space Models training paradigm 几乎一致。一个 natural extension 是用 Mamba-style selective state update 替代 EMA。但 ablation 显示 learnable recurrent dynamics 反而 degrade performance，暗示 **固定低秩结构 + learnable attention write** 在 VLA memory 上更稳健。这与 Mamba 在 language modeling 上 success 但在 RL 上 uncertain 的现状呼应——RL 的 credit assignment 需要的稳定性可能比选择性更重要。

### 与 Future Prediction 流派的互补

GR-1 [https://arxiv.org/abs/2312.13139], Seer [https://arxiv.org/abs/2412.15109], DreamVLA [https://arxiv.org/abs/2501.18862], InternVLA-A1 [https://arxiv.org/abs/2601.02456] 都 predict future state，ReMem-VLA predict past。两者可形成 **hindsight (past) + insight (current) + foresight (future)** 三位一体的 temporal representation。一个 model 同时 reconstruct past、attend present、predict future 是非常有 principled 的 objective 组合，类似 **JEPA** (LeCun) 的 multi-temporal-scale 扩展。

### Hippocampal Replay 与 POP

POP 强制 reconstruct 过去 observation，让人想到 hippocampal replay during sleep——大脑重新激活过去 experience 来 consolidate memory。或许 future work 可以探索：在 episode 结束后做几轮 POP-only "replay" fine-tuning，模拟离线 memory consolidation。

### Memory Capacity 与 Information Bottleneck

Query 数量 $N=128$ 是 sweet spot。这让我想到 **Information Bottleneck** theory——memory capacity 有限时，model 被迫压缩最 task-relevant 的信息。$N=128$ 可能是当前 task 复杂度的 Kolmogorov complexity 的某种 proxy。如果 task 复杂度增加，最优 $N$ 也会增加。一个 dynamic query allocation (类似 Mixture of Experts) 可能更 scalable。

### 局限与开放问题

作者承认：没在 large-scale robot datasets (Open-X-Embodiment [https://arxiv.org/abs/2310.08864], DROID [https://arxiv.org/abs/2403.12945], AgiBot World [https://arxiv.org/abs/2503.06669]) 上 pretrain，generalization 可能受限。Future work 是把 memory mechanism 整合到现有 pretrained VLA (π0.5, GR00T-N1 [https://arxiv.org/abs/2503.14734])。

我认为还有几个有趣方向：
- **Lifelong memory**: 跨 episode、跨 session 的 persistent memory (类似 episodic memory in hippocampus + semantic memory in neocortex)
- **Hierarchical chunk level**: multi-level chunk size 形成 tree-structured memory timescale
- **Read gate**: 让 model 学当前哪些 memory query relevant，减少 attention noise
- **Attention sparsification**: connector 12-layer transformer 计算量大，可探索 sparse attention 或 linear attention
- **Compression via quantization**: memory query 可以 int8 或更低精度存储，减少 inference memory footprint
- **Cross-embodiment memory transfer**: 同一 task 不同 robot 之间的 memory 共享

## 总结一句话

ReMem-VLA 告诉我们：在 large VLA 上做 memory，关键是 **冻结 propagation、只学 content**。固定 EMA + learnable attention write + dual timescale = 一个简单但有效的 recipe，绕开了 TBPTT 的 fundamental limitation。这个 insight 不仅适用于 VLA，可能对所有需要 long-term memory 的大型 sequence model 都有启发价值。

## 关键 References

- MemoryVLA (ICLR 2026): https://arxiv.org/abs/2508.19236
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- MemoryBench (SAM2Act, ICML 2025): reference [17], https://arxiv.org/abs/2506
- VLA-Adapter: https://arxiv.org/abs/2509.09372
- AVA-VLA: https://arxiv.org/abs/2511.18960
- GR-1 (hindsight queries 来源): https://arxiv.org/abs/2312.13139
- DDPM: https://arxiv.org/abs/2006.11239
- DDIM: https://arxiv.org/abs/2010.02502
- Truncated BPTT: https://arxiv.org/abs/1803.06396
- Dilated RNN: https://arxiv.org/abs/1710.02211
- Skip RNN: https://arxiv.org/abs/1708.06834
- Forward-Forward Algorithm: https://arxiv.org/abs/2212.13345
- Equilibrium Propagation: https://arxiv.org/abs/1602.04705
- Differentiable Neural Computer: https://www.nature.com/articles/nature20101
- Mamba: https://arxiv.org/abs/2312.00752
- FeUdal Networks: https://arxiv.org/abs/1703.01161
- Open-X-Embodiment: https://arxiv.org/abs/2310.08864
- DROID: https://arxiv.org/abs/2403.12945
- GR00T-N1: https://arxiv.org/abs/2503.14734
- TraceVLA: https://arxiv.org/abs/2412.10345
- Qwen3-VL: https://arxiv.org/abs/2502 (Qwen3-VL tech report)
- DreamVLA: https://arxiv.org/abs/2501.18862
- Seer: https://arxiv.org/abs/2412.15109
- InternVLA-A1: https://arxiv.org/abs/2601.02456
- ViT: https://arxiv.org/abs/2010.11929
- AgiBot World: https://arxiv.org/abs/2503.06669

---

# ReMem-VLA: Dual-Level Recurrent Queries 为 VLA 注入记忆能力

## 一、Motivation: 为什么 VLA 需要记忆？

当前 VLA 主流范式遵循 Markov 假设，即 $\mathcal{A}_{t:t+k} = \pi_\theta(o_t, \ell)$，policy 只基于当前 observation $o_t$ 和 language instruction $\ell$ 预测未来 action chunk $\mathcal{A}_{t:t+k} = (a_t, \ldots, a_{t+k})$。这种范式在需要历史依赖的任务上系统性地失败。

作者将 VLA 需要的记忆能力分为五种 (Fig. 1)：

1. **Sequential memory**: 多步任务按正确顺序执行
2. **Visual memory**: 用过去视觉线索指导当前决策 (如把物体放回原位)
3. **Spatial memory**: 记住物体位置
4. **Episodic memory**: 跟踪已执行动作 (如舀两勺米就停止)
5. **Temporal memory**: 表示持续时间 (如保持浇水姿态 6 秒)

论文核心 contribution 在于提出一种统一机制覆盖这五种记忆，且不增加 inference cost。

**现有方法局限对比**：
- Retrieval-based (MemoryVLA [45], MemER [47]): 依赖当前帧是否有强 cue，易被 distractor 干扰
- Horizon extension (CronusVLA [32], HAMLET [30]): 固定窗口，超过窗口即失效
- Sparse history (TraceVLA [62], HistRISE [12]): 依赖外部 foundation model (CoTracker, SAM2 等) 稳定性
- Naive recurrence (RoboFlamingo [33], AVA-VLA [11]): 用 truncated BPTT 训练，长期记忆无法通过 gradient 学到

## 二、核心思想：Dual-Level Recurrent Queries

### 2.1 Problem Formulation (公式 1-4)

标准的 Markovian VLA：
$$\mathcal{A}_{t:t+k} = \pi_\theta(o_t, \ell) \tag{1}$$

ReMem-VLA 扩展为 history-aware formulation with auxiliary past observation prediction：
$$\mathcal{A}_{t:t+k}, \hat{o}_{t-m} = \pi_\theta(o_t, \ell, h_t) \tag{2}$$

其中 $h_t = (h_t^c, h_t^f)$ 由两个不同粒度的 component 构成：

**Frame-level memory**（每一 timestep 更新）：
$$h_t^f = \mathcal{F}_f(h_{t-1}^f, o_{t-1}, \ell) \tag{3}$$

**Chunk-level memory**（chunk boundary 才更新）：
$$h_t^c = \mathscr{F}_c(h_{t-k}^c, o_{t-k}, \ell, h_{t-k}^f) \tag{4}$$

变量含义：
- $h_t^f$: timestep $t$ 的 frame-level memory state，承载短期记忆
- $h_t^c$: timestep $t$ 的 chunk-level memory state，承载长期记忆
- $\mathcal{F}_f, \mathscr{F}_c$: memory update path (核心创新点 - **gradient-free**)
- $k$: chunk size (论文中 = action chunk size = 30)
- $m$: past observation prediction 的回溯步数

### 2.2 双层频率的设计 intuition

这个设计让我联想到 **dilated RNN** [10] 和 **Skip RNN** [9] 的思想，但应用于 query token 维度。Frame-level recurrence 每 step 更新，提供细粒度短期信息 (如当前正在执行的动作细节)；chunk-level recurrence 每 $k$ steps 才更新一次，更新频率低导致 EMA 衰减系数 $(1-\beta_c)^{n/k}$ 远慢于 frame-level 的 $(1-\beta_f)^n$，从而保持长期信息 (如任务起始时刻的物体配置)。

这种 **multi-timescale** 结构与人脑类似——working memory (前额叶皮层, sub-second scale) 与 episodic memory (hippocampus, minutes-to-hours scale) 共同作用。

## 三、Architecture 细节 (Fig. 2)

### 3.1 五大组件

1. **Frozen VLM Backbone**: Qwen3-VL-2B [2]，2B 参数，处理 $o_t \in \mathbb{R}^{V \times H \times W \times 3}$ 和 $\ell$，输出 hidden states $\mathbf{H}_t \in \mathbb{R}^{L \times D}$，内部 maintain causal attention
2. **Learnable queries**:
   - Action queries: $Q^{action} \in \mathbb{R}^{N_a \times D}$
   - Hindsight queries: $Q^{img} \in \mathbb{R}^{N_{img} \times D}$ [54, GR-1 启发]
3. **Dual-level recurrent queries**:
   - Frame-level: $Q^f \in \mathbb{R}^{N_f \times D}$
   - Chunk-level: $Q^c \in \mathbb{R}^{N_c \times D}$
4. **Transformer connector**: 12-layer bidirectional self-attention
5. **Prediction heads**:
   - Action diffusion head (conditioned on $Q^{action}$ via cross-attention)
   - ViT-style patch decoder (conditioned on $Q^{img}$)

### 3.2 EMA Update 机制 (公式 5-6)

**Frame-level update**:
$$Q_t^f = \beta_f \cdot \tilde{Q}_{t-1}^f + (1-\beta_f) \cdot Q_{t-1}^f \tag{5}$$

**Chunk-level update** (条件更新):
$$Q_t^c = \begin{cases} \beta_c \cdot \tilde{Q}_{t-k}^c + (1-\beta_c) \cdot Q_{t-k}^c, & \text{if } t \bmod k = 0 \\ Q_{t-1}^c, & \text{otherwise} \end{cases} \tag{6}$$

变量含义：
- $Q_t^f, Q_t^c$: timestep $t$ 的 memory query 状态
- $\tilde{Q}_{t-1}^f, \tilde{Q}_{t-k}^c$: 当前帧通过 attention 从 VLM hidden states 提取的 "新写入" 信息
- $\beta_f, \beta_c \in [0, 1]$: EMA 系数，ablation 显示 $\beta = 0.5$ 最佳 (retention-adaptation trade-off)
- $\tilde{Q}$ 的计算：memory queries 在 VLM 内部 attend to current frame features 得到 updated representation

**直觉解读**：EMA 是 low-pass filter。$\beta$ 大则 memory 更新激进，快速覆盖历史；$\beta$ 小则 memory 惰性，难以吸收新信息。Chunk-level 由于更新稀疏，等效时间常数更长。

### 3.3 Gradient-Free Recurrent Path $\mathcal{F}$ 的关键设计

这是论文最 critical 的设计 choice。论文论证：在大 VLA 上做完整 BPTT 不可行，实践中必须用 truncated BPTT (TBPTT [34])，例如 AVA-VLA 用 $T=4$。这意味着 action loss 无法通过 gradient 教模型在数百 timestep 上保留什么信息。

ReMem-VLA 的解决方案：把 $\mathcal{F}$ (VLM backbone + EMA) **完全冻结**，让 gradient 不通过 recurrent path 流动。Learning 只通过 **front-end attention** 决定 memory query 写什么 (what to write)，而 memory 如何传播 (how to propagate) 由 deterministic forward update 保证。

**Ablation (Fig. 6a) 验证**：在 "Put Block Back" 任务上，引入任何 trainable component 进 recurrent path (trainable VLM / GRU / MLP) 几乎完全消除记忆能力。这与 **Equilibrium Propagation** 和 **Forward-Forward Algorithm** 的哲学相通——把 credit assignment 从 long-range BPTT 中解脱出来。

### 3.4 Connector 设计

VLM 内部 causal attention 阻止 $Q^{action}, Q^{img}$ attend to memory queries (memory queries 放在序列末尾)。Connector 是 12-layer bidirectional transformer，让所有 4 类 query 在 latent space 通过 self-attention 充分交互：
- $Q^{action}$ 整合当前帧 features 与 $Q^f, Q^c$ 中累积的时序上下文，产生 memory-enriched 表示供 action head
- $Q^{img}$ 同样整合上下文供 past image reconstruction

## 四、训练目标 (公式 7-9)

### Action Diffusion Loss

$$\mathcal{L}_{action} = \mathbb{E}_{\tau, \epsilon \sim \mathcal{N}(0, I)} \left[ \| \epsilon - \epsilon_\theta(\mathcal{A}_\tau, \tau, Q^{action}) \|^2 \right] \tag{7}$$

变量：
- $\tau$: diffusion timestep (与 control timestep 不同)
- $\epsilon \sim \mathcal{N}(0, I)$: 添加的 Gaussian noise
- $\mathcal{A}_\tau$: noised action chunk
- $\epsilon_\theta$: denoising network 预测的 noise
- $Q^{action}$: 条件 cross-attention 输入

推理用 DDIM [46]，20 步 denoising。

### Past Observation Prediction (POP)

$$\mathcal{L}_{image} = \| o_{t-m} - \hat{o}_{t-m} \|_2^2 \tag{8}$$

- $o_{t-m}$: $m$ steps 之前的真实 RGB observation
- $\hat{o}_{t-m}$: 通过 ViT-style patch decoder 基于 $Q^{img}$ 重建的 past image
- $m$ 在 ablation 中显示预测 episode 第一帧最有效 (Return Fruit 任务 34% → 82%)

### Total Loss

$$\mathcal{L}_{total} = \mathcal{L}_{action} + \lambda_{img} \mathcal{L}_{image} \tag{9}$$

$\lambda_{img} = 0.5$。

**POP 的 intuition**：让 $Q^{img}$ 通过 connector 与 memory queries 交互，强制 memory queries 携带足够 visual details 以重建过去帧。这是一种 **information bottleneck** 形式的 self-supervised auxiliary loss，把视觉细节"压"进 recurrent state。

## 五、Streaming Slot-Based Batching

这是工程上的关键创新。Recurrent training over variable-length episodes 的 batching 挑战：

- 传统 chunk-based batching: 把 episode 切成 fixed window，破坏 temporal continuity，丢弃 long-horizon context
- Slot-based streaming: 维护 $B$ 个 concurrent slots (batch size)，每个 slot 跟踪不同 episode 在其当前 timestep。每个 training step 从每个 slot 采样 1 帧，产生 batch size $B$。Slot 内 recurrent state 在整个 episode 期间持续累积。Episode 边界 hard-reset recurrent state，确保严格 isolation。

这相当于一种 **continual evolution of recurrent state** 的训练范式，gradient 只在每个 step 内流动 (BPTT truncation = 1)，但 forward state 通过整个 episode 持续演化。这非常像 **State Space Models** (Mamba, S4) 的 training paradigm：gradient 局部，state 全局。

## 六、实验结果

### 6.1 MemoryBench 仿真实验 (Table 1)

| Method | Put Block Back | Rearrange Block | Reopen Drawer | Long Horizon | Avg |
|---|---|---|---|---|---|
| OpenVLA-OFT [27] | 0 | 3 | 0 | 0 | 0.75 |
| π0.5 [22] | 6 | 20 | 3 | 4 | 8.25 |
| MemoryVLA [45] | 0 | 5 | 0 | 1 | 1.5 |
| **ReMem-VLA** | **93** | **99** | **100** | **86** | **94.5** |

注意 Long Horizon Task (>600 frames) 上 ReMem-VLA 86% vs MemoryVLA 1%，差距巨大。MemoryVLA retrieval-based 方法在此失效说明 long-horizon retrieval 难度大。

**实验协议调整**：作者把 MemoryBench 的 button-position randomization 降到 70%，避免 joint limit failures 干扰 memory 评估；Rearrange Block 强制统一 trajectory across configurations，杜绝 cue-based 求解。

### 6.2 真实世界实验 (Fig. 4)

4 个任务各 50 trials：
- Water Flower (~6s, temporal memory): ReMem-VLA 远超 baseline
- Scoop Two Spoons Rice (episodic memory)
- Press Buttons Sequence (sequential + temporal, 还引入 disturbance 验证 closed-loop)
- Put Fruit Back (visual memory)

平均 82.5% vs π0.5 11%, MemoryVLA 8%。

### 6.3 Ablation Studies

**Q1: 双层 recurrent query 的贡献 (Table 2)**

| Config | Avg |
|---|---|
| No Recurrent Query | 17.75 |
| Frame Level Only | 87.75 |
| Chunk Level Only | 84.5 |
| Dual Level | **94.5** |

Fig. 5a 的 failure analysis 显示：纯 frame-level 在 long-horizon 上有更高 memory-related failure；纯 chunk-level 因 button-press 任务需要 short-term memory 而 overall success rate 低；dual level 互补。

**Q2: Gradient-free recurrent path (Fig. 6a)**

Trainable VLM / GRU / MLP 替代 EMA 全部使 memory capability 几乎归零。这是论文最 striking 的发现——**learning the recurrent dynamics 与 long-term memory 在 TBPTT 框架下不兼容**。

**Q3: EMA 系数 (Fig. 6b)**

$\beta_f = \beta_c \in \{0, 0.3, 0.5, 0.7, 0.9, 1\}$ 扫描，$\beta = 0.5$ 最佳。

- $\beta$ 太大: 更新激进，快速覆盖有用历史
- $\beta$ 太小: 惰性 memory，无法吸收新信息

**Q4: Number of recurrent queries (Fig. 6c)**

$N \in \{4, 16, 32, 64, 128, 256, 512\}$，$N=128$ 最佳。太少则 capacity 不足，太多则 redundancy 增加 attention noise 和 optimization 难度。

**Q5: Chunk-level update interval**

0.5×, 1×, 2×, 3× action chunk size，1× (即 30 frames) 最佳，平衡 stability 与 freshness。

**Q6: Past Observation Prediction (Fig. 5b)**

对 visual memory-intensive 的 Return Fruit 任务贡献巨大 (34% → 82%)，对其他非 visual memory 任务贡献有限。证明 POP 专门增强 visual dimension。

## 七、与相关工作对比的深度分析

### 7.1 VLA 范式对比

| 类别 | 代表方法 | 局限 |
|---|---|---|
| Discrete action | RT-2 [6], OpenVLA [29], FAST [42] | 离散化损失精度 |
| Continuous action (regression) | OpenVLA-OFT [27] | 不够平滑 |
| Continuous action (diffusion/flow) | π0 [5], π0.5 [22], GR00T-N1 [4], InternVLA-M1 [13] | 主流趋势 |
| Action + future state prediction | GR-1 [54], Seer [50], DreamVLA [60], InternVLA-A1 [8] | 仅 future，无 past memory |

ReMem-VLA 是首个用 **past observation prediction** 增强 visual memory 的，与 future prediction 流派互补。

### 7.2 History awareness 范式对比

| 范式 | 方法 | 局限 |
|---|---|---|
| Extending window | CronusVLA [32], HAMLET [30], PTP [51], PAM [21] | 固定窗口上限 |
| Sparse history | TraceVLA [62], HistRISE [12], Bpp [39] | 依赖外部 model |
| Memory bank retrieval | MemoryVLA [45], MemER [47] | query-dependent, distractor 干扰 |
| Recurrence | RoboFlamingo [33], AVA-VLA [11] | 未验证长期记忆 |

### 7.3 与 VLA-Adapter [53] 的关系

VLA-Adapter 证明无需大规模 robot pretraining 也能达到 competitive performance，通过 learnable action queries 作为 VLM 与 policy 输出 interface。ReMem-VLA 沿用此设计 (frozen Qwen3-VL-2B + learnable queries)，进一步把 query 思想扩展到 memory dimension。

## 八、我的 intuition 与延伸思考

### 8.1 Memory Query as Working Memory Slot

ReMem-VLA 的 memory query 可视为 **differentiable working memory slots**，类似 Differentiable Neural Computer (DNC) [Graves 2016] 但更轻量。EMA update 等价于 **soft write**，attention read 等价于 **content-based addressing**。这比 LSTM hidden state 更显式，比 retrieval-based memory bank 更 differentiable。

### 8.2 Multi-timescale 与 Hierarchical RL

Frame-level vs chunk-level 让我联想到 **Options framework** (Sutton, Precup) 和 **FeUdal Networks** (Vezhnevets et al.)。Chunk-level memory 类似 manager 提供的 goal，frame-level 类似 worker 的执行。但 ReMem-VLA 没有 explicit hierarchical structure，而是通过不同 update frequency 隐式实现。

### 8.3 Gradient-Free Path 与 Forward-Forward Algorithm

冻结 VLM 和 EMA，让 gradient 不通过 recurrent path，让我想到 Hinton 的 **Forward-Forward Algorithm** 和 **Equilibrium Propagation**。这类方法把 long-range credit assignment 从 BPTT 解放出来，避免 gradient vanishing/exploding。ReMem-VLA 实证了这种哲学在 large VLA 上的有效性。

### 8.4 与 Mamba/SSM 的潜在结合

Slot-based streaming training 与 State Space Models (S4, Mamba) 的 training paradigm 极其相似。一个自然延伸：用 Mamba-style selective state update 替代 EMA。但 ablation 显示 learnable recurrent dynamics 反而 degrade performance，这暗示 **固定低秩结构 + learnable attention write** 在 VLA memory 上可能更稳健。

### 8.5 POP 与 Predictive Coding

POP 强制 reconstruction past observation，类似 **Predictive Coding** 理论在大脑中的机制。大脑持续预测下一时刻 sensory input，预测误差驱动 learning。这里 predict past 而非 future 的 choice 有趣——可能是 past 是确定性的 (从 replay buffer 取)，而 future 是 multi-modal 难以监督。

### 8.6 局限与未来方向

论文自承认：未在 large-scale robot datasets 上 pretrain，可能影响 generalization。未来可整合 memory mechanism 进现有 pretrained VLA (π0.5, GR00T-N1)。

我认为还可以探索：
- Lifelong memory (跨 episode, 跨 session)
- Hierarchical chunk-level (multi-level chunk size)
- Read gate (哪些 memory query 当前 relevant)
- Attention sparsification 减小 connector 计算量

## 九、关键 Web Links Reference

- 论文 (推测 ECCV 2026 投稿): arXiv 链接暂未公开，可关注 https://arxiv.org/abs/ 搜索 "ReMem-VLA"
- Qwen3-VL-2B backbone: https://arxiv.org/abs/2502 (technical report in reference [2])
- MemoryVLA baseline: https://arxiv.org/abs/2508.19236
- π0.5 baseline: https://arxiv.org/abs/2504.16054
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- MemoryBench (SAM2Act, ICML 2025): https://arxiv.org/abs/2506 (reference [17])
- VLA-Adapter: https://arxiv.org/abs/2509.09372
- AVA-VLA: https://arxiv.org/abs/2511.18960
- DDPM (Ho et al.): https://arxiv.org/abs/2006.11239
- DDIM (Song et al.): https://arxiv.org/abs/2010.02502
- Truncated BPTT (Liao et al.): https://arxiv.org/abs/1803.06396
- Dilated RNN: https://arxiv.org/abs/1710.02211
- Skip RNN: https://arxiv.org/abs/1708.06834
- GR-1 (hindsight queries 来源): https://arxiv.org/abs/2312.13139
- ViT (Dosovitskiy et al.): https://arxiv.org/abs/2010.11929
- Open-X-Embodiment: https://arxiv.org/abs/2310.08864
- DROID dataset: https://arxiv.org/abs/2403.12945
- TraceVLA: https://arxiv.org/abs/2412.10345
- CoTracker: https://arxiv.org/abs/2407.21 (reference [24])
- SAM2: https://arxiv.org/abs/2408.00714
- Equilibrium Propagation (Scellier & Bengio): https://arxiv.org/abs/1602.04705
- Forward-Forward Algorithm (Hinton): https://arxiv.org/abs/2212.13345
- Differentiable Neural Computer: https://www.nature.com/articles/nature20101
- Mamba: https://arxiv.org/abs/2312.00752
- Options Framework (Sutton): https://papers.nips.cc/paper/1999

## 十、总结

ReMem-VLA 的核心 insight 可以浓缩为三句话：

1. **Dual-timescale recurrence** 通过 frame-level (short-term) 与 chunk-level (long-term) query 覆盖全谱 memory demand，避免单一 update frequency 的 retention-adaptation 困境。

2. **Gradient-free recurrent path** 是在 large VLA + TBPTT 框架下学到 long-term memory 的关键。Learning 只决定 memory query 写什么 (via attention front-end)，固定 EMA 决定 memory 如何 propagate。这与 forward-forward / equilibrium propagation 哲学共鸣。

3. **Past Observation Prediction** 作为 auxiliary loss 专门补强 visual memory dimension，弥补 recurrent query 在视觉细节保存上的不足。这与未来 prediction 流派 (GR-1, DreamVLA) 互补，可形成 hindsight + insight + foresight 三位一体。

整体而言，这是一篇 architecture + training paradigm 双层创新的工作，把 RNN-style memory 显式注入 transformer-based VLA，并 careful 处理 large-scale training 的工程挑战。Memory capability evaluation 框架覆盖 5 个 dimension 也是社区亟需的 benchmark。在 VLA 走向 long-horizon, real-world deployment 的趋势下，memory mechanism 会成为关键 piece，ReMem-VLA 提供了一个 strong baseline 与可扩展 framework。
