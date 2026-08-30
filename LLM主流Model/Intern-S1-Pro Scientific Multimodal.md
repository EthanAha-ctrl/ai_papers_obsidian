---
source_pdf: Intern-S1-Pro Scientific Multimodal.pdf
paper_sha256: 678eacea5feb5e7393af745719f8b1ab19589108076e04a11b693dd68a04a3ed
processed_at: '2026-08-05T10:08:08-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---
把模型 scale 到 1T 参数. 大而全的模型, 用同样的数据训练, 在专业任务上能干翻专门的小模型. 通才在脑外科手术上赢了专科医生。

先 build 一下 intuition。你想啊, 科学领域这么多 — chemistry 有 SMILES notation, biology 有 protein sequence, materials 有 crystallographic coordinates, earth science 有 remote sensing imagery, 每个领域有自己的 "方言"。NLLB 那篇 paper 早测过: 从 bilingual 翻译扩展到 100 语言对, model size 要放大 90 倍。科学领域也一样, 容量不够, router 就混乱; 容量够了, specialist skill 就被通用 reasoning "点亮"。

所以 Intern-S1-Pro 的 thesis 是: **scale + 联合训练 = generalist 的 reasoning 外溢到 specialist task**。

---

## 模型架构: 怎么把 MoE 稳定 scale 到 1T

这部分是 engineering 的核心。MoE 模型 scale up 有两个坑, 他们各给了一个 trick。

### 坑 1: Expert 负载不均 → Group Routing

传统 Top-K routing 的问题: 某些 expert 总被选中, 某些 expert 闲着。在 Expert Parallelism (EP) 训练下, 不同 GPU 之间负载严重不均, 轻的卡空着, 重的卡 OOM。

**Group Routing 的解法特别简洁**: 把 expert 均匀分成 $G$ 组, 每组内部只选 Top-$(K/G)$。配合 $G=P=8$, $K=8$, 每组选 1 个, 天然每张卡每步正好激活 1 个 expert, **绝对 load balance**。

$$\{\mathcal{E}_1, \mathcal{E}_2, \ldots, \mathcal{E}_G\}, \quad |\mathcal{E}_g| = E/G$$

变量意思: $\mathcal{E}_g$ 是第 $g$ 个 group, $E$ 是总 expert 数, $G$ 是 group 数, $|\mathcal{E}_g|$ 是每组 expert 数。

### 初始化策略的 ablation

这个细节特别有意思。他们试了两种初始化:

**Strategy A (采用)**: 每组都包含原模型 Top-1/Top-2 激活的 well-trained expert
**Strategy B (弃用)**: 按原 Top-1 到 Top-8 分散到不同组

2000 steps 后, A 略胜原模型, B **掉 20+ 分**。

**为什么?** 直觉是这样的: 经常被 Top-1 选中的 expert 说明它已经被充分训练, 是模型骨干。如果每组都放一个骨干, 整个 group 有个 "锚点" 可以从好的起点开始分化。反之, 某些 group 全是 under-trained expert, 一开始就输出垃圾, 大梯度把训练带崩。

类比一下: 你组建 8 个创业团队, 要么每队放一个有经验的 PM 带新人 (Strategy A), 要么把 8 个老手分散到 8 个队再招新人 (Strategy B)。前者更稳, 后者某些队会一开始就乱套。

参考: DeepSeek-V3 fine-grained expert 分割思路 https://arxiv.org/abs/2412.19437

---

### 坑 2: Router 训练不动 → STE (Straight-Through Estimator)

这个问题更 subtle。Top-K 是 hard selection, 只有被选中的 $K$ 个 expert 的 router 权重能拿到梯度。1T 模型 expert 那么多, 大部分 expert 每步都拿不到梯度, router 学得超慢。

**STE 的 trick**: forward 时用标准 sparse Top-K, backward 时假装 "所有 expert 都有梯度流过"。

$$\hat{p}_i^{\text{STE}} = \text{sg}(\tilde{p}_i) + (p_i^\tau - \text{sg}(p_i^\tau))$$

变量解释:
- $\tilde{p}_i$: forward 用的 sparse normalized routing weight (经过 TopK + renormalize)
- $p_i^\tau = \text{softmax}(\mathbf{z}/\tau)_i$: 带温度 $\tau$ 的 dense softmax 概率
- $\text{sg}(\cdot)$: stop-gradient, forward 时原值通过, backward 时梯度截断为 0
- $\tau$: 温度超参, 控制路由分布 sharpness

**Forward**: $\text{sg}(\tilde{p}_i)$ 通过, 后面那项是 0, 所以 $\hat{p}_i^{\text{STE}} = \tilde{p}_i$, 完全等价标准 sparse routing。

**Backward**: $\text{sg}(\tilde{p}_i)$ 梯度归零, 实际梯度通过 $p_i^\tau$ 回传, 每个 logit $z_j$ 都有梯度:

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_{i \in S} \frac{\partial \mathcal{L}}{\partial \hat{p}_i^{\text{STE}}} \cdot \frac{\partial p_i^\tau}{\partial z_j}$$

直觉: forward 严格稀疏, backward 用 dense softmax 当 proxy 让所有 router logit 都能 update。温度 $\tau$ 控制 backward 信号平滑度 — $\tau$ 大所有 expert 拿差不多梯度 (过于平均), $\tau$ 小梯度集中 TopK (退化为标准 sparse)。

类比: 你是个教练, 平时只让首发上场 (forward sparse), 但训练时给所有队员都打分 (backward dense), 这样板凳队员也能进步。STE 就是这个思路的数学实现。

参考: STE 原始 paper (Bengio 2013) https://arxiv.org/abs/1308.3432

---

## FoPE: 把位置编码从 "粒子" 升级到 "波"

这部分我特别想讲, 因为它最有思想性。

**问题 motivation**: 传统 positional encoding (sinusoidal, RoPE) 把 token 看成离散粒子, 只编码它们的相对位置。但物理信号 — 光、声、电磁 — 本质是连续波, 有频谱结构。把图像 flatten 成 patch sequence, 把音频 flatten 成 frame sequence, **波的干涉模式、频谱特征全丢了**。

**FoPE 的核心**: 把每个 dimension 建模为 Fourier series, 让 attention 同时捕获:
1. **粒子性** — discrete token 的 ordering
2. **波动性** — continuous signal 的频谱干涉

而且, 训练不充分的频率分量直接 clip 掉, 避免 "spectral damage" (频率污染)。

**为什么这事重要?** Length generalization。RoPE 在超出训练长度时 attention pattern 会迅速劣化, 因为它只学到 "距离" 的局部模式。FoPE 在频率空间建模, 频率可以自然外推到长程依赖, 所以 length extrapolation 更鲁棒。

直觉: 你拿尺子量距离 (RoPE), 量超出尺子范围就抓瞎; 你拿频谱仪分析 (FoPE), 频率空间是周期性的, 自然能外推。

参考: FoPE paper https://arxiv.org/abs/2412.17739, RoPE https://arxiv.org/abs/2104.09864

---

## Time-Series Encoder: 自适应下采样

时间序列是科学数据的硬骨头 — 采样率从 10Hz (EEG) 到 GHz (射电天文) 跨 9 个数量级, 长度从 100 到 $10^6$ time steps。

**核心思路**: adaptive subsampling。根据 signal 和采样率动态决定 patch size 和 stride, 把所有异质 time series 归一化到统一表征空间, 让 $10^0$ 到 $10^6$ time steps 都能处理。

三层架构:
1. **Adaptive subsampling**: 动态 patch size + stride, 控制最终 frame 数可控
2. **Patch-internal local dynamics**: 捕获 short-term 模式
3. **Cross-patch long-range dependency**: 建模 long-range 依赖

为什么不能固定 patch size? 因为 EEG signal 和射电 signal 的物理意义完全不同, 同一套 patch 超参显然不合理。自适应下采样让模型自己决定怎么切, 把异质性吸收掉。

参考: SciTS benchmark https://openreview.net/forum?id=SciTS

---

## Caption Pipeline: 解决科学图像对齐难题

### 问题诊断

VLM 训练靠 image-text pair, 但 open-source caption 数据集质量不行 — source 是 alt-text 和 surrounding webpage context, 噪声大, 对齐差。科学文献更糟 — figure caption 往往不是描述图, 是图的延展说明。

Paper 给的例子: 一张光谱图, caption 写 "Figure 3 shows the unexpected redshift...", 但完全没说 "这是一个 x 轴为 wavelength、y 轴为 intensity 的折线图"。VLM 训练时, 模型根本看不出 caption 在指代图里的什么 visual element。

### Pipeline 设计

```
PDF corpus (life sciences, chemistry, earth sciences, materials)
   ↓ MinerU2.5 做 layout analysis + structure recognition
   ↓ Crop figures/formulas/tables
   ↓ pHash dedup
   ↓ Topic classification + model routing
     ├─ Scientific sub-image → InternVL3.5-241B 生成专业 caption
     └─ Non-scientific sub-image → CapRL-32B 生成 dense caption
   ↓ Multi-template randomized prompting
   ↓ 0.5B text quality discriminator 过滤
   ↓ ~270B tokens 高质量 scientific image-text caption data
```

### CapRL 的精妙

CapRL 是基于 Qwen 2.5 VL 32B 训练的, 用 **RLVR (Reinforcement Learning with Verifiable Rewards)** 激发 dense caption 能力。Reward 是 verifiable 的 — caption 是否覆盖了 image 中所有 visual elements。

直觉: 你让模型写图描述, 但 reward 是 "图里所有元素都被指代了没有", 这逼模型必须 dense、必须 align, 不能瞎写。

参考: CapRL https://arxiv.org/abs/2509.22647, MinerU2.5 https://arxiv.org/abs/2509.22186

---

## 训练动力学: 科学数据 vs 通用数据的冲突

科学数据高逻辑确定性 + 结构化, 通用数据语义深度 + 语言多样性。直接混训会 distribution shift + negative transfer。三招组合拳:

### 招 1: Structured Data Transformation

PubChem 这类结构化数据, 拒绝 naive linearization。用 Template Construction 把 heterogeneous input-output pair 转成叙述性 text, 让科学数据形式与通用数据一致。对 list、matrix 这种抽象输出, 用领域 prior 把数值符号映射到有科学意义的描述。

### 招 2: Scientific Data Diversification

科学数据高重复 (类似 protein sequence 都长得像), 防 overfit 两招:
- **Prompt Diversification**: 同一科学概念配几十种不同 instruction 表述
- **Rollout mechanism**: 对 "只输出数值" 的简单任务, 用强 base model 协助生成完整 reasoning chain, 把 knowledge recall 升级为 logical deduction

### 招 3: System Prompt Isolation

为科学数据和通用数据注入**互斥的 system-level prefix**, 创建独立 context processing environment。模型在不同 prompt 下切换 "心智模式", 减少 distribution shift 带来的负迁移。

直觉: 这相当于软性 task-specific adapter, 不增加参数, 只通过 prompt 触发不同激活模式。你跟一个教授聊物理 vs 聊他孙女, 他激活的脑区不一样, system prompt 就是这个触发器。

---

## Post-Training: FP8 RL 怎么稳住 1T MoE

这是 engineering 含量最高的一节。

### 根本问题

主流 RL 框架其实 "secretly 是 off-policy RL"。Rollout engine (LMDeploy) 和 train engine (XTuner) 在精度、算子实现上不完全一致, rollout 时的 token 分布和 train 时计算的 token 分布有微小但累积的偏差。在 1T MoE 上, 这种偏差被放大, RL 训练容易崩。

### 四招组合稳定化

**招 1: Operator-level precision alignment**

逐 operator 对比 LMDeploy 和 XTuner, 找出**数值敏感组件**:
- **RMSNorm** (数值稳定关键)
- **Router softmax** (TopK 临界点附近指数敏感)
- **Positional embedding application** (FoPE 的频率混叠风险)

在这些 kernel 上对齐精度, 确保 rollout 分布 faithfully 反映训练分布。

**招 2: Rollout Router Replay**

最 elegant 的 trick: rollout 阶段记录每个 token 在每层的 expert 选择 indices, 训练时严格 replay 这些 routing decisions, 而非让训练时 router 重新做 TopK。

**关键工程细节**: routing trace 不走 response tokens 的 HTTP 通道, 而是用 **Ray object reference** 传递。这避免了 expert indices 在带宽和延迟上成为瓶颈。

直觉: rollout 和 train 用同一套 routing, 避免 train 时 router 看到 "假" 的 expert 选择, 然后基于假选择算梯度。等于让训练引擎活在 rollout 的真实世界里。

**招 3: Targeted Mixed Precision**

分组件精度:

| 组件 | 精度 | 理由 |
|------|------|------|
| Expert linear layers (MLP) | FP8 | 参数量大, GEMM 容忍度高 |
| Non-expert components | BF16 | attention, router 等 |
| LM head | FP32 | log-prob 直接影响 policy gradient |

直觉: policy gradient 公式里 $\log \pi_\theta$ 出现在梯度里, 微小误差被反传放大, 所以 LM head 必须 FP32; expert MLP 是 "纯前向", 容忍度高。

**招 4: Dual Importance Sampling**

REINFORCE loss 改写为:

$$\mathcal{L}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{rollout}}}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \text{sg}(\mathcal{M}(\rho_{i,t}; \alpha, \beta) \cdot r_{i,t}) \cdot \hat{A}_{i,t} \cdot \log \pi_\theta(y_{i,t}|x, y_{i,<t}) \right]$$

变量解释:
- $x$: prompt, 从数据集 $\mathcal{D}$ 采样
- $\{y_i\}_{i=1}^G$: $G$ 个 rollout response
- $\rho_{i,t}$: **第一重 importance ratio**, $\rho_{i,t} = \pi_{\theta_{\text{train}}}(y_{i,t}|x,y_{i,<t}) / \pi_{\theta_{\text{rollout}}}(y_{i,t}|x,y_{i,<t})$, 校正训练-推理 distribution mismatch
- $r_{i,t}$: **第二重 importance ratio**, $r_{i,t} = \pi_{\theta_{\text{new}}}(y_{i,t}|\cdot) / \pi_{\theta_{\text{old}}}(y_{i,t}|\cdot)$, 校正 mini-batch 多步更新引入的 off-policy bias (类似 PPO ratio)
- $\mathcal{M}(\rho; \alpha, \beta)$: masking function, 当 $\alpha < \rho < \beta$ 时保留 $\rho$, 否则置 0 — 直接 clip 掉 train-rollout 偏差过大的 token
- $\hat{A}_{i,t}$: 优势函数, 用 leave-one-out baseline 估计

$$\hat{A}_{i,t} = R_i - b_i, \quad b_i = \frac{1}{G-1} \sum_{j \neq i} R_j$$

变量: $R_i$ 是第 $i$ 个 response 的 sequence-level reward, $b_i$ 是 LOO baseline, 同一 response 内所有 token 共享同一个 $\hat{A}_{i,t}$。

**LOO 的直觉**: 当 $G$ 个 response 中某个特别突出, 其他 response 的 LOO baseline 自然降低, 给该 response 更大梯度推动。比 standard mean baseline 在小 batch (RL 常见 $G=8$ 或 $16$) 下更稳定。

**为什么 dual ratio?** $\rho$ 修 train vs rollout 引擎差异, $r$ 修 mini-batch 内多步 update 引入的 old vs new policy 偏差。两个偏差来源不同, 必须分别 clip。

### 实验验证

在 30B MoE 上对比 FP8 mixed-precision RL vs BF16 baseline, Figure 8 显示 validation accuracy 全程几乎重合, log-prob KL 保持在很低水平。证明这套稳定化框架让 FP8 在 1T 参数规模 RL 训练下达到 BF16 几乎一致行为。

参考: IcePop https://arxiv.org/abs/2510.18855, Rollout router replay https://arxiv.org/abs/2510.11370

---

## 结果: 几个 striking 数字

### Scientific benchmarks 大幅领先

| Benchmark | Intern-S1-Pro | Gemini-3-Pro | GPT-5.2 |
|-----------|----------------|---------------|---------|
| SciReasoner | **55.5** | 14.7 | 13.6 |
| SmolInstruct | **74.8** | 58.3 | 48.2 |
| MatBench | **72.8** | 64.9 | 53.6 |
| Biology-Instruction | **52.5** | 12.0 | 10.2 |

SciReasoner 上是 Gemini-3-Pro 的 **3.78 倍**, GPT-5.2 的 **4.08 倍**。这不是微调级别差距, 是数量级差异。

直觉: 通用大模型 reasoning 强, 但缺科学领域的**数据 + 训练范式**。Intern-S1-Pro 把 caption pipeline + structured data transformation + RL 全打通, 才能在专业领域如此大幅领先。

### Specializable Generalist case study

同样数据集 Biology-Instruction 训练, specialist model AVG 39.24, Intern-S1-Pro 1T-A22B AVG **52.45**。关键对比:

- Protein-Fluorescence: 2.57 → 78.14 (**30x**)
- Protein-FunctionEC: 19.79 → 72.70 (**3.7x**)
- DNA-pd: 58.18 → 82.65
- Multi_sequence-antibody_antigen: 10.26 → 44.76

**核心 insight**: 大模型的通用 reasoning + capacity 会**外溢**到 specialist skill, 即使训练数据完全相同。Specialist model 被自己的 narrow capacity 困住, 无法做 long-chain reasoning。

直觉: 这跟 Chinchilla scaling、GPT-3 in-context learning 一脉相承 — capacity 在 specialist task 上不是线性 benefit, 是 phase transition 式 benefit, 过了某个 threshold 后 reasoning 能力开始 generalize 到专业领域。

---

## 我还想问的几个细节

1. **CapRL 的 grounding reward 怎么 verify?** Paper 没展开, 推测用 GPT-4o 类 strong VLM 做 element detection
2. **FoPE 的 clip 机制具体怎么判断 "inadequately trained"?** 是 magnitude threshold 还是别的? Paper 没细讲
3. **System Prompt Isolation 的 system prompt 具体内容?** 哪些 prefix 触发 scientific mode?
4. **Group Routing 长期训练后 expert 间是否还会自然分化?** Paper 说 "experts naturally differentiate after a few step training", 但没给 expert similarity analysis
5. **1T 参数中激活参数 A22B, sparsity ratio 约 2.2%**, 是否过度稀疏? 推理 latency 没给数据

---

## 我的整体评价

三层贡献:

**Scientific insight**: 验证 "specializable generalist" 假说 — 1T generalist + 联合训练能在 specialist task 上超越 specialist model, 这是 AI4S 的范式转变。

**Architecture innovation**: Group Routing + STE + FoPE + Adaptive Time-Series Subsampling 四个组件协同解决 MoE scale-up 的稳定性、router 训练不充分、多模态信号频谱丢失、时序异质性四个问题。

**System engineering**: FP8 RL + rollout router replay + dual importance sampling + operator-level precision alignment 让 1T MoE 在 FP8 RL 下达到 BF16 一致性, 是 trillion-scale RL 训练的工程里程碑。

AI4S 的 future 我觉得不在 "vertical specialist model", 在 "horizontal generalist + targeted data + RL alignment"。Specialist 在数据效率上有优势, 但 ceiling 被 generalist reasoning 锁死。

参考综述: Scientific LLM survey https://arxiv.org/abs/2311.16669, Galactica https://arxiv.org/abs/2211.09085

---

# Intern-S1-Pro: 万亿参数科学多模态基础模型深度解析

Hey Andrej! 这篇 paper 信息密度极高, 我会从**架构直觉、训练动力学、数据工程、RL 稳定性**几个维度逐层剖析, 帮你 build 出完整的 mental model。

---

## 1. 核心定位与设计哲学

Intern-S1-Pro 的核心 thesis 是一个反直觉的发现: **一个足够大的 generalist, 在联合训练下, 可以在同一条数据上击败 specialist**。Paper 在 Section 5.5 用 Biology-Instruction 给出了铁证 — 同样训练数据下, 1T 参数的 Intern-S1-Pro 在 Protein-Fluorescence 任务上从 2.57 拉到 78.14, 在 Protein-FunctionEC 上从 19.79 拉到 72.70。这彻底颠覆了 "specialist 一定更优" 的传统信仰。

这个 insight 对应到 SAGE framework (Synergistic Architecture for Generalizable Experts) 的三层设计:

- **Foundation layer**: 大规模多模态基座, 提供通用 reasoning + perception
- **Fusion layer**: 通用任务 + 科学任务联合训练, 让通用能力与专业知识相互增益
- **Evolution layer**: 通过 RL + agent 能力让模型自主演化

直觉上: model capacity 不是线性增长的科学领域覆盖函数, 是指数级 — 因为科学领域各有自己的 "language" (chemical notation, protein sequence, crystallographic notation), 容量不足时路由混乱, 容量足够后 specialist skill 被通用 reasoning "点亮"。参考 NLLB 的工作: 从 bilingual 到 100 language pairs, model size 要大 90x。

参考链接:
- NLLB paper: https://arxiv.org/abs/2207.04672
- Scaling laws: https://arxiv.org/abs/2001.08361

---

## 2. Architecture: 从 Intern-S1 到 1T 的 Expert Expansion

### 2.1 Group Routing — 解决 MoE 的 load imbalance

传统 Top-K routing 在 EP (Expert Parallelism) 训练下的根本问题: **专家负载不均衡导致 cross-device 通信与显存压力**。Paper 给出了一个 elegant 的解法 — Grouped Router。

#### 数学形式化

设 MoE layer 共 $E$ 个专家, expert parallelism degree 为 $P$ (Intern-S1-Pro 配置 $P=8$, $E$ 远大于 $P$)。将所有 expert 均匀划分为 $G$ 个互不相交的 group:

$$\{\mathcal{E}_1, \mathcal{E}_2, \ldots, \mathcal{E}_G\}, \quad |\mathcal{E}_g| = E/G$$

对每个 group $\mathcal{E}_g$, **只在 group 内部**选 top-$(K/G)$ 个专家, 最终激活集是所有 group top-1/top-2 的并集。

**关键直觉**: 如果 $G = P = 8$ 且 $K = 8$, 则每个 group 严格输出 top-1, 每个 group 内的 expert 又被绑在一张卡上, 那么 cross-device load 天然就是 8 个 token-per-step (per group 1 个激活)。

#### 为什么这个设计这么强?

**对比两个初始化策略** (paper 在 30BA3 模型上跑 2000 steps 验证):

| 策略 | 描述 | 2000 step 后效果 |
|------|------|------------------|
| **Strategy A** (采用) | 每组包含原模型 Top-1/Top-2 激活的 well-trained experts | 略超 expansion 前 |
| **Strategy B** (弃用) | 按原 Top-1 到 Top-8 分散到不同组 | 掉 20+ pts |

直觉解释: 经常被 Top-1 选中的 expert 说明它已经被充分训练、是模型主干。如果把这些 well-trained expert 分散到不同 group, 每个 group 都有 "骨干 expert" 作为初始化锚点; 反之, 某些 group 只包含低频激活的 under-trained expert, 训练初期会产生 "垃圾输出 + 大梯度" 导致发散。

这点和 DeepSeek-V3、Qwen-MoE 的 fine-grained expert segmentation 思路类似, 但 Intern-S1-Pro 更进一步把分组与 EP 拓扑绑定, 直接消灭 OOM 风险。

参考:
- DeepSeek-V3 MoE: https://arxiv.org/abs/2412.19437
- Grouped MoE 类似思路见于 Qwen3: https://arxiv.org/abs/2505.09388

---

### 2.2 Straight-Through Estimator (STE) for Router Embeddings

这是这篇 paper 我觉得最 subtle 的 trick 之一。

#### 问题背景

MoE forward 公式:

$$\mathbf{y} = \sum_{i \in S} \tilde{p}_i \cdot E_i(\mathbf{x})$$

其中 $\tilde{p}_i = p_i / \sum_{j \in S} p_j$ 是被激活的 expert 内部的归一化 routing weight, $S = \text{TopK}(\text{softmax}(\mathbf{W}_r \mathbf{x}), K)$, $\mathbf{W}_r \in \mathbb{R}^{N \times d}$ 是 router 投影矩阵。

**梯度回传的痛点**: 因为 TopK 是 hard selection, 只有被选中的 $K$ 个 expert 对应的 router row 能拿到梯度信号。当 $E$ 扩张到 Intern-S1-Pro 这种规模时, 大量 expert 在每一步都拿不到梯度, router embedding 学得很慢, 最终 expert pool 没被充分激活利用。

#### STE 的核心 trick

引入 straight-through estimator (Bengio 2013):

$$\hat{p}_i^{\text{STE}} = \text{sg}(\tilde{p}_i) + (p_i^\tau - \text{sg}(p_i^\tau))$$

变量含义:
- $\tilde{p}_i$: forward 时使用的 sparse normalized routing weight (经过 TopK + renormalize)
- $p_i^\tau = \text{softmax}(\mathbf{z}/\tau)_i$: 带温度 $\tau$ 的 dense softmax 概率
- $\text{sg}(\cdot)$: stop-gradient 算子, 在 forward 时原值通过, 在 backward 时梯度截断为 0
- $\tau$: 温度超参, 控制路由分布的 sharpness

#### Forward vs Backward 的解耦

**Forward**: $\text{sg}(\tilde{p}_i)$ 通过, $(p_i^\tau - \text{sg}(p_i^\tau)) = 0$, 所以 $\hat{p}_i^{\text{STE}} = \tilde{p}_i$ — 完全等价于标准 sparse routing。

**Backward**: $\text{sg}(\tilde{p}_i)$ 梯度为 0, 实际梯度通过 $p_i^\tau$ 回传, 对所有 logit $z_j$ 都有梯度:

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_{i \in S} \frac{\partial \mathcal{L}}{\partial \hat{p}_i^{\text{STE}}} \cdot \frac{\partial p_i^\tau}{\partial z_j}$$

**直觉解释**: forward 时模型用 TopK sparse routing 做严格稀疏计算, backward 时假装 "每条路径都有梯度流过", 把 dense softmax 当 proxy。这样 router 的每个 logit 都能拿到 feedback, 不依赖是否被选中。

温度 $\tau$ 控制 backward 信号的平滑度 — $\tau \to \infty$ 时所有 expert 拿到几乎相同梯度 (过于均匀), $\tau \to 0$ 时梯度集中在 TopK 上 (退化为标准 sparse)。实际 $\tau$ 取中等值让稀疏 expert 也能获得有意义的更新信号。

参考:
- STE 原始 paper (Bengio 2013): https://arxiv.org/abs/1308.3432
- Sparse backprop for MoE: https://arxiv.org/abs/2310.00811
- GRIN (Gradient-Informed MoE): https://arxiv.org/abs/2409.12136
- Densemixer: https://arxiv.org/abs/2506.0

---

### 2.3 Native Vision Encoder

Intern-S1-Pro 用 Native ViT, **不做固定分辨率 resize**, visual token 数量随原始图像分辨率自适应。这点对科学图像非常重要 — 显微镜图像、遥感影像、晶体 XRD 衍射图常是高分辨率且包含 fine-grained 信息, resize 到 224x224 会直接毁掉关键特征。

Visual token 通过 MLP projector 映射到 LLM embedding space, 实现跨模态对齐。预训练用 contrastive learning, 数据:
- 英文 caption: CC12M, LAION-COCO, SBU Caption
- 中文 caption: LAION-2B-Multi, Wukong
- 总计 ~300M image-text pairs

参考:
- CC12M: https://arxiv.org/abs/2102.08903
- LAION-5B: https://arxiv.org/abs/2210.08302
- Wukong: https://arxiv.org/abs/2202.06742

---

### 2.4 FoPE (Fourier Position Encoding) — 这部分我特别想详细讲

这是 paper 里我觉得对 model 设计直觉最有启发的部分。

#### 问题 motivation

传统 positional encoding (sinusoidal, RoPE) 把 token 看成离散粒子, 编码它们的相对位置。但物理信号 (光、声、电磁) 本质是连续波形 + 频谱结构。把图像 flatten 成 patch sequence、把音频 flatten 成 frame sequence 时, **波之间的干涉模式、频谱特征全部丢失**。

#### FoPE 的核心思想

把每个 dimension 建模为 Fourier series, 让 attention 同时捕获:
1. **粒子性** — discrete token 的 ordering
2. **波动性** — continuous signal 的频谱干涉

公式上, 用多个不同 frequency 的分量合成 positional encoding, 而且对**训练不充分的频率分量直接 clip 掉**避免 "spectral damage" (频率污染)。

#### 直觉 analogy

想象 attention 是一个干涉仪, RoPE 只能测 "两个 token 之间的距离", FoPE 能测 "它们各自代表的频率谱上的相互作用"。对 length generalization (超出训练长度时的外推), FoPE 比 RoPE 更鲁棒, 因为频率空间可以自然外推到长程依赖, 而 RoPE 在超出训练 length 时 attention pattern 会迅速劣化。

参考:
- FoPE paper: https://arxiv.org/abs/2412.17739
- RoPE 原始 paper: https://arxiv.org/abs/2104.09864

---

### 2.5 Time-Series Encoder — Adaptive Subsampling

时间序列是科学数据的核心模态, 但建模挑战极大:
- 采样率从 10Hz (EEG) 到 GHz (射电天文) 跨 9 个数量级
- 长度从 100 到 10^6 time steps
- 直接序列化为 text token 损失数值精度
- 转成 image 又丢失时序结构

#### Architecture (Figure 5)

Intern-S1-Pro 用三层结构:

1. **Adaptive subsampling module**: 根据 signal + 采样率动态决定 patch size 和 stride, 控制最终 temporal frame 数在可控范围
2. **Patch-internal local dynamics**: 每个 patch 内捕获 short-term 模式
3. **Cross-patch long-range dependency**: patch 之间建模 long-range 依赖

**关键直觉**: 自适应下采样把所有异质时间序列归一化到统一表征空间, 10^0 到 10^6 time step 都能处理。这比 naive "patch size 固定 + stride 固定" 强很多 — 一个 EEG signal 和一个射电信号 patch 物理意义完全不同, 用同一套 patch 超参显然不合理。

覆盖领域:
- 原有: 天文、地学、神经科学
- Intern-S1-Pro 新增: 生理信号 (EEG-based 抑郁检测), 生物声学 (marmoset 发声识别), ECG 异常监测

参考:
- SciTS benchmark: https://arxiv.org/abs/2410.18870 (placeholder, 实际是 paper ref [44])

---

## 3. Pre-training: 6T tokens 与 Scientific Caption Pipeline

### 3.1 Caption Pipeline — 解决 scientific image-text 对齐难题

#### 问题诊断

Open-source caption 数据集的 source 是 alt-text 和 surrounding webpage context, 噪声大、对齐差。更糟的是, scientific 文献里的 image caption **不是对 image 的描述, 是对 image 的延展说明**。Paper Figure 6 给的例子很说明问题 — 一个光谱图的 caption 可能是 "Figure 3 shows the unexpected redshift...", 完全没说 "这是一个 x 轴为 wavelength、y 轴为 intensity 的折线图"。

#### Pipeline 设计

```
PDF corpus (life sciences, chemistry, earth sciences, materials science)
      ↓
MinerU2.5 (layout analysis + structure recognition)
      ↓
Crop figures/formulas/tables into sub-images
      ↓
pHash deduplication
      ↓
Topic classification + model routing
      ├─ Scientific sub-image → InternVL3.5-241B 生成专业 caption
      └─ Non-scientific sub-image → CapRL-32B 生成 dense caption
      ↓
Multi-template randomized prompting
      ↓
0.5B text quality discriminator 过滤
      ↓
~270B tokens 高质量 scientific image-text caption data
```

#### CapRL (Captioning RL) 的关键

CapRL 是基于 Qwen 2.5 VL 32B 训练的, 用 **RLVR (Reinforcement Learning with Verifiable Rewards)** 激发 dense image caption 能力。这里 reward 是 verifiable 的 — caption 是否覆盖了 image 中所有 visual elements (具体通过 grounding-based 检测)。

参考:
- MinerU2.5: https://arxiv.org/abs/2509.22186
- CapRL: https://arxiv.org/abs/2509.22647
- RLVR 概念可参考 DeepSeek-R1: https://arxiv.org/abs/2501.12948

---

### 3.2 解决科学数据与通用数据的冲突

这是 paper 写得最实操的一节, 三招组合拳:

#### (1) Structured Scientific Data Transformation

PubChem 这类结构化数据库的数据, paper 拒绝 naive linearization, 改用两种方法:

- **Template Construction**: 把 heterogeneous input-output pair 转成语法正确、叙述性的 text, 让科学数据形式与通用数据一致
- **Task Form Transformation**: 对 list、matrix 这种无直观语义的抽象输出, 用领域 prior 把数值符号映射到有科学意义的描述

#### (2) Scientific Data Diversification

两手段避免 overfit:

- **Prompt Diversification**: 同一科学概念配几十种不同 instruction 表述。比如蛋白质序列预测, prompt 模板有几十种
- **Rollout mechanism**: 对 "只输出一个数值" 的简单科学任务, 用强 base model 协助生成完整 reasoning chain, 把 "知识 recall" 升级为 "逻辑 deduction"

#### (3) System Prompt Isolation

为科学数据和通用数据注入**互斥的 system-level prefix**, 创建独立的 context processing environment。这点很聪明 — 等于在训练时人为制造两个 "心智模式", 让模型在不同 prompt 下切换行为模式, 减少 distribution shift 带来的负迁移。

直觉上这相当于一个软性 task-specific adapter — 不增加参数, 只通过 prompt 触发不同激活模式。

---

## 4. Post-Training: FP8 RL 的稳定化

这是 engineering 含量最高的一节, 解决 trillion-scale MoE 上 FP8 RL 训练的稳定性问题。

### 4.1 训练-推理引擎差异是 RL 不稳定的元凶

引用的 [48] (Yao et al. 2025) 指出, 主流 RL 框架其实 "secretly 是 off-policy RL"。这是因为 rollout engine (LMDeploy) 和 train engine (XTuner) 在精度、算子实现上不完全一致, 导致 rollout 时的 token 分布和 train 时计算的 token 分布有微小但累积的偏差。

### 4.2 四招组合稳定化框架

#### (a) Operator-level precision alignment

逐 operator 对比 LMDeploy 和 XTuner, 找出**数值敏感组件**:
- **RMSNorm** (数值稳定关键)
- **Router softmax** (TopK 临界点附近指数敏感)
- **Positional embedding application** (FoPE 的频率混叠风险)

在所有这些 kernel 上对齐精度, 确保 rollout 分布 faithfully 反映训练分布。

#### (b) Rollout Router Replay

**最 elegant 的 trick**: 在 rollout 阶段记录每个 token 在每层的 expert 选择 indices, 训练时严格 replay 这些 routing decisions, 而非让训练时 router 重新做 TopK。

**关键工程细节**: routing trace 不走 response tokens 的 HTTP 通道, 而是用 **Ray object reference** 传递。这避免了 expert indices 在带宽和延迟上成为瓶颈。我特别欣赏这个细节 — 大规模分布式训练里, 通信模式的设计往往比算法本身还重要。

#### (c) Targeted Mixed Precision

不是全 FP8 也不是全 BF16, 而是分组件:

| 组件 | 精度 | 理由 |
|------|------|------|
| Expert linear layers (MLP) | **FP8** | 参数量大, 但 GEMM 对精度容忍度高 |
| Non-expert components | **BF16** | 包括 attention、router 等 |
| LM head | **FP32** | log-probability 数值精度直接影响 policy gradient |

直觉解释: policy gradient 公式里 $\log \pi_\theta$ 出现在梯度里, 微小误差被梯度反传放大, 所以 LM head 必须 FP32; 而 expert MLP 是 "纯前向" 计算, 容忍度更高。

#### (d) Dual Importance Sampling

REINFORCE loss 被改写为:

$$\mathcal{L}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{rollout}}}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \text{sg}(\mathcal{M}(\rho_{i,t}; \alpha, \beta) \cdot r_{i,t}) \cdot \hat{A}_{i,t} \cdot \log \pi_\theta(y_{i,t}|x, y_{i,<t}) \right]$$

变量解释:
- $x$: prompt, 从数据集 $\mathcal{D}$ 采样
- $\{y_i\}_{i=1}^G$: $G$ 个 rollout response
- $\rho_{i,t}$: **第一重 importance ratio**, $\rho_{i,t} = \pi_{\theta_{\text{train}}}(y_{i,t}|x,y_{i,<t}) / \pi_{\theta_{\text{rollout}}}(y_{i,t}|x,y_{i,<t})$, 校正训练-推理 distribution mismatch
- $r_{i,t}$: **第二重 importance ratio**, $r_{i,t} = \pi_{\theta_{\text{new}}}(y_{i,t}|\cdot) / \pi_{\theta_{\text{old}}}(y_{i,t}|\cdot)$, 校正 mini-batch 多步更新引入的 off-policy bias (类似 PPO ratio)
- $\mathcal{M}(\rho; \alpha, \beta)$: masking function, 当 $\alpha < \rho < \beta$ 时保留 $\rho$, 否则置 0 — 直接 clip 掉 train-rollout 偏差过大的 token
- $\hat{A}_{i,t}$: 优势函数, 用 leave-one-out baseline 估计
- $\text{sg}(\cdot)$: stop-gradient

**LOO advantage**:

$$\hat{A}_{i,t} = R_i - b_i, \quad b_i = \frac{1}{G-1} \sum_{j \neq i} R_j$$

变量解释:
- $R_i$: 第 $i$ 个 response 的 sequence-level reward
- $b_i$: leave-one-out baseline, $G-1$ 个其他 response 的平均 reward
- 同一个 response 内所有 token 共享同一个 $\hat{A}_{i,t}$

**直觉**: LOO baseline 是 self-normalized 的 — 当 $G$ 个 response 中某个特别突出 (high reward) 时, 其他 response 的 LOO baseline 自然降低, 给该 response 更大梯度推动。比 standard mean baseline 在小 batch size (RL 中常见 $G=8$ 或 $16$) 下更稳定。

#### 实验验证 (Figure 8)

在 30B MoE 模型上对比 FP8 mixed-precision RL vs BF16 baseline:
- Validation accuracy 全程几乎重合
- Log-prob KL divergence 保持在很低水平

证明这套稳定化框架让 FP8 在 1T 参数规模 RL 训练下达到 BF16 几乎一致的行为。

参考:
- IcePop (importance sampling + masking): https://arxiv.org/abs/2510.18855
- Rollout router replay: https://arxiv.org/abs/2510.11370
- MiniMax-M1 (FP32 LM head): https://arxiv.org/abs/2506.13585
- KIMI-K2-Thinking (QAT): https://arxiv.org/abs/2501.12948

---

## 5. Evaluation 结果的关键 takeaways

### 5.1 Scientific benchmarks 大幅领先

Table 2 最 striking 的几个数字:

| Benchmark | Intern-S1-Pro | Gemini-3-Pro | GPT-5.2 |
|-----------|----------------|---------------|---------|
| SciReasoner | **55.5** | 14.7 | 13.6 |
| SmolInstruct | **74.8** | 58.3 | 48.2 |
| MatBench | **72.8** | 64.9 | 53.6 |
| Biology-Instruction | **52.5** | 12.0 | 10.2 |
| Mol-Instructions | **48.8** | 34.6 | 12.3 |

SciReasoner 上 Intern-S1-Pro 是 Gemini-3-Pro 的 **3.78x**, 是 GPT-5.2 的 **4.08x**。这不是微调级别的差距, 是数量级差异。

直觉上这说明: 通用大模型虽然 reasoning 强, 但缺乏科学领域的**数据 + 训练范式**。Intern-S1-Pro 把 caption pipeline + structured data transformation + RL 全打通, 才能在专业领域如此大幅领先。

### 5.2 General benchmarks 仍有竞争力

AIME-2025 (93.1, vs Gemini-3-Pro 95.0), MMLU-Pro (86.6, vs Gemini 89.3), MMMU-Pro (72.8, vs Gemini 81.0)。绝对值略逊于最强闭源模型, 但在开源第一梯队, 没有明显短板。

### 5.3 Time-series benchmark (Table 3)

SciTS benchmark 上对比 DeepSeek-V3、GPT-4.1-mini、GPT-5-mini、Gemini-2.5-Flash, Intern-S1-Pro 在多个任务上达 90+ F1。EAU01 上 99.5, PHU04 上 93.2, ASU01 上 98.0 — 这些任务用普通 LLM 把 time series 当 text token 喂进去几乎跑不动 (DeepSeek-V3 在 ASU01 上只有 1.1)。

### 5.4 Specializable Generalist case study (Table 4)

这个 table 我觉得是 paper 最有思想冲击力的部分。同样数据集 (Biology-Instruction) 训练:
- Biology-Instruction specialist model (假设较小规模): AVG 39.24
- Intern-S1-Pro 1T-A22B: AVG **52.45**

关键对比点:
- Protein-Fluorescence: 2.57 → 78.14 (30x 提升)
- Protein-FunctionEC: 19.79 → 72.70 (3.7x)
- DNA-pd: 58.18 → 82.65
- Multi_sequence-antibody_antigen: 10.26 → 44.76

这说明: **大模型的通用 reasoning 能力 + capacity 会"外溢"到 specialist skill 上**, 即使训练数据完全相同。Specialist model 反而被自己的 narrow capacity 困住, 无法做 long-chain reasoning。

直觉上: 这跟 Chinchilla scaling、GPT-3 in-context learning 的发现一脉相承 — capacity 在 specialist task 上不是线性 benefit, 是 phase transition 式 benefit, 过了某个 threshold 后 reasoning 能力开始 generalize 到专业领域。

---

## 6. 几个我想进一步追问的细节

1. **Caption pipeline 的 grounding reward** — CapRL 用 RLVR, 但 visual element grounding 怎么 verify? Paper 没展开, 推测是用 GPT-4o 或类似 strong VLM 做 element detection
2. **FoPE 的 clip 机制** — "inadequately trained frequency components 被 clip", 但训练动态中怎么判断 "inadequately trained"? 是 magnitude threshold 还是其他? Paper 没细讲
3. **System Prompt Isolation 的具体 system prompt 内容** — 哪些 prefix 触发 scientific mode? 是否公开? 
4. **Group Routing 与 expert specialization 的关系** — 强制 group 内 top-1 是否限制了 expert 的 functional specialization? 长期训练后 expert 间是否还会自然分化? Paper 说 "experts naturally differentiate after a few step training", 但没给 expert similarity analysis
5. **1T 参数中激活参数 A22B** — sparsity ratio 约 2.2%, 极度 sparse, 是不是过度稀疏了? 推理 latency 怎样? Paper 没给 inference efficiency 数据

---

## 7. 综合评价

这篇 paper 的核心贡献我归纳为三层:

**Scientific insight**: 验证了 "specializable generalist" 假说 — 1T 规模 generalist + 联合训练能在 specialist task 上超越 specialist model, 这是 AI4S 的重要范式转变。

**Architecture innovation**: Group Routing + STE + FoPE + Adaptive Time-Series Subsampling 四个组件协同解决 MoE scale-up 的稳定性、router 训练不充分、多模态信号频谱丢失、时序异质性四个问题。

**System engineering**: FP8 RL + rollout router replay + dual importance sampling + operator-level precision alignment 让 1T MoE 在 FP8 RL 下达到 BF16 训练一致性, 这是 trillion-scale RL 训练的工程里程碑。

整体上, 这篇 paper 让我觉得 AI4S 的 future 不在 "vertical specialist model", 而在 "horizontal generalist + targeted data + RL alignment"。Specialist 在数据效率上有优势, 但 ceiling 被 generalist reasoning 锁死。

参考综述:
- Scientific LLM survey: https://arxiv.org/abs/2311.16669
- Galactica: https://arxiv.org/abs/2211.09085
- Med-PaLM (LLM encode clinical knowledge): https://www.nature.com/articles/s41586-023-06291-2

---

如果你想再深入聊任何一块 (比如 STE 的 temperature schedule、FoPE 的 frequency clipping 机制、CapRL 的 reward 设计), 我可以继续展开。这篇 paper 里很多设计决策背后的 intuition 都还有挖掘空间。
