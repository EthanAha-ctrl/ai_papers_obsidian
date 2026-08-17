---
source_pdf: DefeatingtheTraining-InferenceMismatchviaFP16.pdf
paper_sha256: edeb1ff8a3250f55dc0273213477add64590c0a5e05468023917d8bbf9356d7d
processed_at: '2026-08-03T19:07:03-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说：这篇paper到底在讲啥

---

## 一、先说背景：大家在搞啥

现在训LLM（比如DeepSeek-R1、Qwen这种reasoning模型），最后一关是**用强化学习让模型学会"思考"**——给它一道数学题，它生成一堆答案，对的奖励、错的惩罚，慢慢学会解题。这叫RL fine-tuning。

但这件事**特别难训稳**，经常训着训着模型就崩了——accuracy突然掉、reward乱跳、policy collapse。整个2025年industry花了巨大力气研究"怎么让RL不崩"。

---

## 二、发现了什么"罪魁祸首"

大家之前以为是**算法问题**，于是发明一堆patch：
- GRPO加token-level importance sampling（Yao et al.）
- 加sequence-level masked IS（Liu et al.）
- GSPO专门处理MoE的mismatch
- 等等

但这篇paper说：**别搞了，根子根本不在算法，在数值精度**。

具体来说，现代训练默认用**BF16**（Google搞出来的16-bit浮点数），但这玩意儿精度不够——只有7个mantissa bit。

### 这事儿为什么会出问题

RL训练时，同一个模型其实跑在**两个不同的engine**上：
- **Inference engine**（vLLM/SGLang）：专门优化来快速generate response，快但实现跟training engine不一样
- **Training engine**（DeepSpeed/FSDP）：算gradient用的

理论上这俩应该数学等价，因为用的是同一份weights $\theta$。但实际上：
- BF16只有7个有效bit
- 两个engine的CUDA kernel实现细节不同（reduction顺序、tensor parallel切法、fusion策略）
- 每个token的logit都有~0.78%的相对误差
- Auto-regressive生成几百几千个token，这些误差**累乘**起来，probability能差几个数量级

结果就是：你以为在按policy $\pi$ 训练，实际采样来自policy $\mu$，这俩差远了。

---

## 三、之前大家怎么"打补丁"

### 补丁1：Importance Sampling
数学上的标准fix——给每个样本重新加权：
$$\text{权重} = \frac{\pi(y)}{\mu(y)}$$
理论上能把biased gradient变unbiased。但问题：sequence很长时这个权重**方差爆炸**——有的样本权重1e-5，有的1e10，gradient噪声大得没法训。

### 补丁2：Truncate / Mask
把过大过小的权重clip掉（设上限 $C=3$）。代价是又引入了bias。而且Yao et al.和Liu et al.的实现都需要**额外一次forward pass**来算 $\pi(\cdot|\theta')$，训练慢25%。

### 补丁3：GSPO
干脆不用 $\mu$ 了，全用 $\pi$。但这样rollout效率低，等于绕开了inference engine的优化。

### 补丁的共同问题
1. **慢**：多一次forward pass
2. **Deployment gap还在**：你训练是针对 $\pi$ 优化的，但部署用的是 $\mu$，所以训出来的模型部署时性能掉
3. **治标不治本**：相当于"用算法弥补数值bug"

---

## 四、Paper的核心发现：换回FP16就完事了

### FP16 vs BF16 到底差在哪

两个都是16 bit，但分配方式不同：

| | FP16 | BF16 |
|---|---|---|
| Exponent bits | 5 | 8 |
| Mantissa bits | **10** | 7 |
| 动态范围 | 小（$6 \times 10^{-5}$ 到 $6 \times 10^4$） | 大（接近FP32） |
| 精度 | **高8倍** | 低 |

**直觉**：BF16是"覆盖范围大但每个数表示得粗糙"；FP16是"覆盖范围小但每个数表示得精细"。

Pre-training阶段：
- Weights乱初始化、gradient在不同layer magnitude差几个数量级
- 需要**大动态范围**，所以BF16合适
- Pre-training用的是forward+backward，gradient noise会平均掉

RL fine-tuning阶段：
- Weights已经稳定了，不需要那么大range
- 但sampling时是**autoregressive的乘法链**——一个token的概率误差会传到后面所有token
- Importance sampling ratio是**两个相近数相除**——BF16下7 bit精度根本不够区分
- 这时候**精度**比**range**重要得多

### FP16的"配套机制"早就成熟了

FP16有个老问题：gradient太小会underflow到0。但2017年NVIDIA的Micikevicius paper就给出了**Loss Scaling**这个标准解法：

1. Loss乘一个大数 $S$（比如 $2^{16}$）
2. Backward时gradient自动被放大，从underflow区域抬出来
3. Optimizer更新前再除以 $S$ 还原
4. 如果某个step检测到gradient overflow（出现inf），就把 $S$ 减半；连续N步没overflow就把 $S$ 翻倍

这套机制在PyTorch AMP、Megatron、DeepSpeed里**都是标配**，开一行config就行。所以切回FP16几乎**零工程成本**。

---

## 五、实验结果：简单到令人发指

### Sanity Test设计

作者搞了个很巧的benchmark：从MATH数据集里筛出"模型初始accuracy在20%-80%之间"的问题。这种问题是"模型能学会但还不会"的——理论上RL应该能把它推到接近100%。

然后测各种算法在这套perfectible dataset上能不能收敛到95%以上。结果：

### BF16下：
- **Vanilla GRPO**：73%-84%就崩了
- **GRPO + Token-TIS**：82%-88%就崩了
- **GSPO**：训得更久但最终也不行
- **GRPO + Seq-MIS**：稳但慢，最高95%
- 所有fancy算法**没有一个能到99%**

### FP16下：
- **最naive的policy gradient（PG-Seq-IS）直接到99%**
- 所有算法（GRPO、TIS、MIS、GSPO）**几乎重合**——因为mismatch没了，所有correction都成了冗余

也就是说：**之前所有fancy algorithm的工作，本质上都是在"BF16的数值noise"上打补丁**。把noise source去掉，naive方法就秒杀一切。

### Ablation的key insight

测了不同的training/inference precision组合：
- BF16 training + FP32 inference：能稳定，但**3倍慢**
- BF16 training + FP16 inference：稍微好点，但终崩
- FP16 training + FP16 inference：**最佳**，又快又稳

**关键点**：必须两边都FP16，只改一边没用——因为mismatch的本质是两边的"数值口音不同"，只改一边相当于让一个人改口音。

### 跨场景验证

不只small model上work：
- **MoE模型**（Qwen3-30B-A3B）：top-k expert selection对精度超敏感，FP16照样大幅改善
- **LoRA**（Qwen2.5-Math-1.5B + LoRA rank 32）：BF16下600步崩，FP16全程稳
- **14B dense model** + DAPO算法：FP16训练reward上升更快
- **OctoThinker-3B**（Llama3.2-3B mid-trained）：BF16 150步崩，FP16稳

跨**model family、algorithm、framework、size**都work。

---

## 六、几个深层intuition

### 1. Bias-Variance Tradeoff被打破了

BF16下存在这么个规律：
- **高bias低variance**方法（GRPO、TIS、GSPO）：前期收敛快但最终崩
- **低bias高variance**方法（PG-IS、Seq-MIS）：稳但慢

大家一直在这两个之间tradeoff。FP16直接**同时降低bias和variance**——mismatch小了所以bias小，importance ratio tight了所以variance小。naive方法反而最优。

### 2. 实际上把off-policy变回了on-policy

RL文献里讨论on-policy vs off-policy的tradeoff，但都默认 $\mu = \pi$。实际上现代RL framework从来不是真正on-policy的——因为两个engine的数值差异。FP16让我们第一次接近理论假设的世界。

### 3. Collapse前都有mismatch增长的"预警信号"

观察到一个现象：所有最终崩掉的训练，**崩之前mismatch都先快速增长**。具体表现是 $\pi(\cdot|\theta')$ 和 $\mu(\cdot|\theta')$ 走向极端——一个policy给某token 0.99概率，另一个给0.01。

这是一个**positive feedback loop**：
- mismatch → biased gradient
- biased gradient → policy push到某些tokens
- 这些tokens在两个engine下分歧更大
- mismatch进一步放大
- 最终爆炸

FP16在源头就把mismatch压住，根本进入不了这个blow-up regime。

### 4. 为什么GSPO不用 $\mu$ 反而更稳

paper里一个未完全解释的发现：GSPO完全不用inference policy $\mu$，比TIS还稳。我猜是因为GSPO的clip设计更保守（高bias低variance），BF16下mismatch放大了bias但没让policy进入collapse regime。本质上是用更多bias换稳定性。

### 5. 为什么这种简单fix被忽略了

几个原因：
- **Industry惯性**：A100/H100默认BF16，"BF16 = better"深入人心
- **发paper激励**：算法改进更"publishable"，precision问题显得trivial
- **Pre-training经验误导**：pre-training BF16确实更好，大家惯性延用到RL

但RL的numerics demand完全不同——sampling对精度敏感，pre-training对range敏感。这是这篇paper最深刻的insight之一。

---

## 七、工程实现有多简单

```python
# 之前
model = model.to(torch.bfloat16)
autocast(dtype=torch.bfloat16)

# 现在
model = model.to(torch.float16)
autocast(dtype=torch.float16, init_scale=2**16, growth_interval=2000)
```

VeRL/Oat框架里就是改个config flag。Loss scaling是成熟infrastructure，开箱即用。

Caveats：
- **大模型（70B+）要小心overflow**：FP16最大65504，某些activation可能突破。建议保留dynamic loss scaling
- **Adam optimizer state保持FP32**：这是mixed precision的标准做法
- **vLLM inference本来默认FP16**：所以对齐很容易

---

## 八、几个open question

1. **GSPO为什么不用 $\mu$ 反而更稳？** Paper没完全解释，需要更深入分析
2. **100B+模型上FP16的overflow风险？** 14B验证过OK，但更大scale没测
3. **PPO、ReMax、RLOO等算法呢？** 理论上一致但没empirical验证
4. **Long context（32K+）RL？** Mismatch随sequence length增加，FP16下也增加但慢24×。超长reasoning是否还work？
5. **FP8呢？** FP8的mantissa更少（2-3 bit），我猜RL fine-tuning会更糟。但paper提到FP8 pre-training的成功暗示range限制可以工程解决

---

## 九、一句话总结

**整个2025年大家花大力气设计fancy RL algorithm来"修"训练不稳定的问题，结果发现只要把数值精度从BF16切回FP16，所有问题就没了——什么importance sampling、token-level correction、sequence-level masking，全都成了冗余。最naive的policy gradient在FP16下秒杀所有BF16下的复杂算法。**

这跟Karpathy你常说的"first principles thinking"完全一致——很多看似复杂的algorithmic问题，本质是被numerical issue mask的simpler problem。先把root cause fix掉，再去搞algorithm。

Paper链接：https://github.com/sail-sg/Precision-RL

希望这次"人话版"讲清楚了。如果你想深入某个具体方面（比如loss scaling的实现细节、importance sampling的数学推导、或MoE上mismatch的特殊性），我可以再展开。

---

# Defeating the Training-Inference Mismatch via FP16 — 深度解读

这篇paper来自Sea AI Lab的Penghui Qi、Min Lin等人，发现了一个反直觉但极其elegant的结论：**RL fine-tuning LLMs时的training-inference mismatch，根源在BF16本身，切回FP16即可解决**。下面我尽量把intuition、公式细节和实验数据都讲透。

---

## 1. 问题的根源：Training-Inference Mismatch

现代RL框架（VeRL、OpenRLHF、Oat等）为了效率，rollout用vLLM/SGLang这种高度优化的inference engine，training用DeepSpeed/FSDP这种training engine。理论上两边用同一份weights $\theta$，policy应该完全一致 $\mu = \pi$。但实际上CUDA kernel实现、parallel策略、autoregressive采样顺序的微小数值差异，会让 $\mu(\cdot|x,\theta)$ 和 $\pi(\cdot|x,\theta)$ 产生divergence。

这带来两个深层问题：

### 1.1 Biased Gradient

理想的policy gradient（REINFORCE estimator）：

$$\nabla_\theta \mathcal{J}(x,\theta) = \mathbb{E}_{y \sim \pi(\cdot|x,\theta)}\left[\nabla_\theta \log \pi(y|x,\theta) \cdot R(x,y)\right]$$

变量含义：
- $x$：从prompt分布 $p_{\mathcal{X}}$ 采样的输入
- $y$：response序列
- $R(x,y)$：reward函数
- $\pi(\cdot|x,\theta)$：训练engine算出的policy

实际操作中我们从 $\mu$ 采样response，但用 $\pi$ 算gradient：

$$\nabla_\theta \mathcal{J}_{\text{biased}}(x,\theta) = \mathbb{E}_{y \sim \mu(\cdot|x,\theta)}\left[\nabla_\theta \log \pi(y|x,\theta) \cdot R(x,y)\right] \neq \nabla_\theta \mathcal{J}(x,\theta)$$

这个estimator是有偏的，因为采样分布和评估分布不一致。

### 1.2 Deployment Gap

更隐蔽但更致命的问题——训练针对 $\pi$ 优化，但部署用 $\mu$：

$$\arg\max_\theta \mathbb{E}_{x \sim p_\mathcal{X}, y \sim \mu(\cdot|x,\theta)}[R(x,y)] \neq \arg\max_\theta \mathbb{E}_{x \sim p_\mathcal{X}, y \sim \pi(\cdot|x,\theta)}[R(x,y)]$$

即使算法层面用importance sampling修复了biased gradient，deployment gap依然存在，因为最终 $\theta$ 是在 $\pi$ 的分布下优化出来的，对 $\mu$ 不是optimal的。这点让我想起off-policy RL里的classic问题，但在这里更棘手，因为 $\pi$ 和 $\mu$ 看似是同一个模型。

---

## 2. 现有Algorithmic Patches及其局限

### 2.1 Importance Sampling Correction

理论上正确的unbiased estimator：

$$\nabla_\theta \mathcal{J}_{\text{pg-is}}(x) = \mathbb{E}_{y \sim \mu(\cdot|x,\theta')}\left[\frac{\pi(y|x,\theta)}{\mu(y|x,\theta')} \nabla_\theta \log \pi(y|x,\theta) \cdot A(x,y)\right]$$

变量含义：
- $\theta'$：用于sampling的参数（off-policy下可能与 $\theta$ 不同）
- $A(x,y) = \bar{R}(x,y) - B(x)$：advantage，$B(x)$ 是baseline用于variance reduction
- $\frac{\pi(y|x,\theta)}{\mu(y|x,\theta')}$：importance sampling ratio

但这个ratio在LLM场景下variance爆炸，因为sequence很长，token-level概率连乘导致ratio极不稳定。所以引入两类truncation：

**Truncated IS (TIS)** — Yao et al. 2025:
$$\nabla_\theta \mathcal{J}_{\text{pg-tis}}(x) = \mathbb{E}_{y \sim \mu}\left[\min\left(\frac{\pi(y|x,\theta)}{\mu(y|x,\theta')}, C\right) \cdot \nabla_\theta \log \pi(y|x,\theta) \cdot A(x,y)\right]$$

其中 $C$ 是clipping threshold，tradeoff bias和variance。

**Masked IS (MIS)** — Liu et al. 2025:
$$\nabla_\theta \mathcal{J}_{\text{pg-mis}}(x) = \mathbb{E}_{y \sim \mu}\left[\frac{\pi(y|x,\theta)}{\mu(y|x,\theta')} \cdot \mathbb{I}\left\{\frac{\pi(y|x,\theta)}{\mu(y|x,\theta')} \leq C\right\} \cdot \nabla_\theta \log \pi(y|x,\theta) \cdot A(x,y)\right]$$

$\mathbb{I}\{\cdot\}$ 是indicator function，超过 $C$ 的整个sequence直接mask掉。

### 2.2 GRPO-based Implementations

工业界主流是基于GRPO的，原始GRPO gradient：

$$\nabla_\theta \mathcal{J}_{\text{grpo}}(x) = \mathbb{E}_{y \sim \mu(\cdot|x,\theta')}\left[\sum_{t=1}^{|y|} \nabla_\theta \min\left(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

其中：
- $r_t = \frac{\pi(y_t | x, y_{<t}, \theta)}{\pi(y_t | x, y_{<t}, \theta')}$：training policy在新旧参数下的token-level ratio
- $A_t = R(x,y) - \frac{1}{G-1}\sum_{i=1}^{G-1} R(x, y_i)$：group-relative advantage
- $G$：每个prompt的rollout数量
- $\epsilon$：clip ratio (默认0.2，paper里设clip_high=0.28)

Yao et al. 加了token-level TIS patch：

$$\nabla_\theta \mathcal{J}_{\text{grpo-tok-is}}(x) = \mathbb{E}_{y \sim \mu}\left[\sum_{t=1}^{|y|} \min(\rho_t, C) \cdot \nabla_\theta \min\left(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

其中 $\rho_t = \frac{\pi(y_t|x,y_{<t},\theta')}{\mu(y_t|x,y_{<t},\theta')}$ 是training engine和inference engine的token-level概率比。

Liu et al. 改成sequence-level MIS：

$$\nabla_\theta \mathcal{J}_{\text{grpo-seq-mis}}(x) = \mathbb{E}_{y \sim \mu}\left[\rho \cdot \mathbb{I}\{\rho \leq C\} \cdot \sum_{t=1}^{|y|} \nabla_\theta \min\left(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

其中 $\rho = \frac{\pi(y|x,\theta')}{\mu(y|x,\theta')}$ 是sequence-level ratio。

**核心问题**：这些patch都需要额外forward pass算 $\pi(\cdot|\theta')$（同一个engine下），按backward = 2×forward估算，增加约25%训练开销。更根本的问题是deployment gap依然存在——optimizer依然在 $\pi$ 的流形上收敛。

---

## 3. FP16 vs BF16：为何Precision是Key

这是paper最核心的insight。我先把floating point格式讲透。

### 3.1 Bit Allocation

| Property | FP16 | BF16 |
|----------|------|------|
| Total bits | 16 | 16 |
| Sign bits | 1 | 1 |
| Exponent bits | 5 | 8 |
| Mantissa bits | 10 | 7 |
| Bias | 15 | 127 |
| Smallest normal | ~$6.1 \times 10^{-5}$ | ~$1.2 \times 10^{-38}$ |
| Largest value | ~$6.6 \times 10^{4}$ | ~$3.4 \times 10^{38}$ |
| Next representable > 1 | $1 + 2^{-10} \approx 1.000977$ | $1 + 2^{-7} \approx 1.007812$ |

公式化表示：
$$x = (-1)^s \times (1 + m/2^{M}) \times 2^{e - bias}$$

- $s$：sign bit
- $m$：mantissa，$M$ bits
- $e$：exponent，$E$ bits
- $bias = 2^{E-1} - 1$

FP16的precision比BF16高8倍（$2^{10}$ vs $2^7$ 表示精度），但dynamic range小得多。这意味着BF16能表示的数值跨度大（接近FP32的$10^{-38}$到$10^{38}$），但每个数值的相对精度差。

**Intuition**：当两个engine实现略有不同（kernel fusion顺序、reduction tree、tensor parallel的all-reduce），BF16只保留7个有效bit，很容易在第7 bit之后产生不同rounding，这些误差在autoregressive generation中累乘/累加放大。

### 3.2 为什么BF16在Pre-training OK但RL不行

Pre-training是forward + backward，gradient有stochastic noise自然average out，dynamic range更重要（gradient在不同layer、不同training stage的magnitude跨度大）。

RL fine-tuning阶段：
1. 模型weights的数值范围已经在pre-training稳定了，不需要BF16的extra range
2. Sampling依赖precise的logit → softmax → categorical sampling，precision差一个bit可能选不同的token
3. Importance sampling ratio $\frac{\pi}{\mu}$ 是两个相近数的除法，BF16下7 bit mantissa意味着相对误差$2^{-7} \approx 0.78\%$，对每个token的logprob都加这个noise，长sequence下指数放大

具体看Figure 2的数字：BF16的sequence-level log-probability ratio distribution随sequence length变宽，FP16保持tight约24×。这是cumulative autoregressive error的直接证据。

### 3.3 Loss Scaling：FP16的"配套机制"

FP16的range问题（gradient underflow）早就有成熟解法——Loss Scaling (Micikevicius et al. 2017)：

1. Forward pass正常算
2. Loss乘以scaling factor $S$（典型 $S = 2^{16}$ 到 $2^{24}$）
3. Backward时所有gradient自动放大 $S$ 倍，从underflow区域"抬"出来
4. Optimizer step前把gradient除以 $S$ 还原

**Dynamic Loss Scaling**：
- 若某step检测到gradient含inf/nan → $S$ 减半，跳过该step
- 连续 $N$ steps（如1000）无overflow → $S$ 翻倍

这套机制在PyTorch AMP、Megatron、DeepSpeed里都已是标准组件，开一行config即可，对用户几乎透明。这点很关键——切回FP16没有engineering cost，因为infrastructure早就ready了。

### 3.4 为什么BF16能"反超"FP16成为主流

历史角度：BF16起源于Google TPU（2017年左右），NVIDIA Ampere架构（A100, 2020）原生支持。BF16的最大吸引力是**drop-in FP32 replacement**——dynamic range和FP32一样，意味着不需要loss scaling这种全局同步机制，分布式训练的communication和optimizer step都更简单。

Pre-training大规模上BF16的turning point大概是GPT-3时代（2020）到Megatron-LM/DeepSpeed广泛采用之后。整个industry的肌肉记忆变成"BF16是默认"，这种惯性延续到RL fine-tuning，但RL的numerics demand完全不同。

---

## 4. Sanity Test：一个干净的诊断benchmark

Paper设计了一个很巧妙的test，我特别喜欢这个design：

### 4.1 构造方法

从MATH数据集（Hendrycks et al. 2021）对每个问题rollout 40次，保留initial accuracy在[20%, 80%]的问题，对DeepSeek-R1-Distill-Qwen-1.5B得到1460个questions。

**Intuition**：
- 太简单的问题（initial acc > 80%）：浪费compute，模型已经会了
- 太难的问题（initial acc < 20%）：分不清是algorithm不行还是model本身能力天花板
- 中间区间：每个问题都在模型"边界"上，RL应该能push到接近100%

### 4.2 Pass Criterion

RL算法在perfectible dataset上训练accuracy > 95% 算pass。这是个很强的诊断——任何fail的算法都是fundamentally broken的，因为它连"理论上能做到"的事都没做到。

这个思路很像debugging——给一个minimal reproducible case，看algorithm是否能收敛到ground truth已知的解。我觉得这个test应该成为RLHF/RL研究的standard benchmark。

---

## 5. 实验结果：BF16下算法对比

### 5.1 BF16下的表现（Section 4.2）

| Algorithm | VeRL Peak Acc | Oat Peak Acc | 是否Collapse |
|-----------|---------------|--------------|--------------|
| Vanilla GRPO (Dr.GRPO) | 73% | 84% | Yes, early |
| GRPO + Token-TIS (Yao) | 82% | 88% | Yes, delayed |
| GSPO (Zheng et al.) | >TIS | >TIS | Stable longer, no μ used |
| GRPO + Seq-MIS (Liu) | 95% peak | 95% peak | Stable but slow |
| PG-Seq-IS (vanilla) | — | — | High variance |
| **FP16 PG-Seq-IS** | **99%** | **99%** | **Stable + fast** |

关键观察：
1. **Vanilla GRPO最早collapse** — 因为没有任何mismatch correction
2. **Token-TIS延长但终collapse** — 与Liu et al.的发现一致，token-level correction有biased gradient
3. **GSPO比TIS更稳** — 有趣的发现，GSPO根本不用inference policy $\mu$，反而避开了mismatch！这暗示mismatch的来源比想象中更复杂
4. **Seq-MIS最稳但慢** — unbiased但sequence-level ratio方差大，convergence慢，且peak只有95%（vs FP16的99%），deployment gap依然可见

### 5.2 FP16下所有算法表现（Section 4.3）

切到FP16后，所有算法表现几乎重合（Figure 4）！这点极其深刻：

- **Vanilla PG-Seq-IS**（最naive的estimator）在FP16下大幅超越所有BF16下的复杂算法
- GRPO、TIS、MIS、GSPO在FP16下差异几乎消失
- FP16把问题从"off-policy with mismatch"变成"approximately on-policy"，所有correction都变得冗余

这个结果让我想到一个类比：以前大家花大力气设计各种bias-variance tradeoff的算法，结果发现只要把底层noise source去掉，naive方法就够了。这跟"don't fix symptoms, fix root cause"的工程哲学完全一致。

### 5.3 Mismatch的量化（Figure 2）

Token-level scatter plot（左两图）：
- BF16：data points散落对角线两侧，明显deviation
- FP16：data points紧贴对角线

Sequence-level log probability ratio分布（右两图）：
- BF16：随sequence length增加，ratio distribution指数级展宽
- FP16：保持tight，约24× smaller mismatch

这个24×是怎么来的呢？大概可以这样估算：FP16 vs BF16的mantissa精度比 $2^{10}/2^7 = 8$，每个token的relative error小8倍，sequence length $L$ 下cumulative error ratio约 $8^L$？但实际是24×是因为长sequence下还有其他因素（softmax normalization、temperature）的混合效应。Paper没给精确推导，但empirical数字clear。

### 5.4 Ablation：Training vs Inference Precision（Section 4.4）

这个ablation非常重要，单独列出：

| Training | Inference | 结果 |
|----------|-----------|------|
| BF16 | BF16 | Collapse (baseline) |
| BF16 | FP16 | 延长但终collapse |
| BF16 | FP32 | 完全stable，但3× slower |
| FP16 | BF16 | 中等改善 |
| FP16 | FP16 | 最佳：fast + stable |
| FP16 | FP32 | 应该也稳但未测 |

**关键结论**：
- 单边提升inference precision能缓解，但不能根治（因为training engine的 $\pi$ 还是BF16的noisy版）
- BF16 training + FP32 inference完全stable，但3× slowdown使该方法impractical
- **必须双端FP16**，正好对应"消除mismatch"的对称性要求

这个ablation强有力地支持了"precision本身是root cause"的论断——只要任何一边还是BF16，mismatch就还存在。

---

## 6. 泛化验证

### 6.1 MoE RL (Section 5.1)

用Qwen3-30B-A3B-Base（A3B = 3B activated params, 30B total）。MoE特别关键因为：
- top-k expert selection对precision极其敏感（一个logit差一点就选不同expert）
- training和inference的parallel策略差异更大
- 现有工作（如GSPO）就是专门为MoE mismatch设计的

结果显示FP16在GRPO-Seq-MIS、GRPO-Token-TIS、PG-Seq-TIS三种算法上都比BF16稳定得多。这表明FP16的fix跨architecture generalize。

### 6.2 LoRA RL (Section 5.2)

用Qwen2.5-Math-1.5B + LoRA (rank=32, $\alpha=64$)，全层LoRA。BF16下600步collapse，FP16全程stable。

LoRA的interesting之处在于：updates在低秩子空间，按理说noise应该被compress了，但BF16下还是collapse。这暗示mismatch的来源不是gradient noise本身，而是sampling时的policy divergence——LoRA改变了部分weights但base model还是BF16，所以 $\mu$ 和 $\pi$ 的divergence依然大。

参考 Schulman & Thinking Machines Lab 最近的blog "LoRA without regret"（https://thinkingmachines.ai/blog/lora/）也讨论了类似问题。

### 6.3 Large Dense Model (Section 5.3)

Qwen3-14B-Base + DAPO algorithm（Yu et al. 2025）。FP16训练reward上升更快，AIME 2024 validation acc更高。

这个scale上的验证对industry adoption很重要——14B已经接近production规模。FP16的overflow风险在14B上没出现，说明loss scaling够用。

### 6.4 OctoThinker-3B (Section 5.4)

基于Llama3.2-3B mid-trained on reasoning data。BF16下150步destabilize，FP16 smooth。这证明FP16 fix不仅Qwen-family适用，跨model family也work。

---

## 7. 一些深层思考与Intuition

### 7.1 Bias-Variance Tradeoff的重新理解

Section 6的discussion讲到一个很深刻的观察：BF16下存在bias-variance tradeoff
- 高bias低variance方法（GRPO、Token-TIS、GSPO）：快但不稳，终collapse
- 低bias高variance方法（PG-Seq-IS、GRPO-Seq-MIS）：稳但慢

FP16打破了这个tradeoff——同时降低bias（mismatch本身小）和variance（importance ratio tight）。这是为什么naive PG在FP16下反而能超越所有fancy algorithm。

这让我想到numerical analysis里的经典教训：很多"算法问题"本质是"数值问题"。当底层精度足够，简单算法往往胜出；精度不够时，需要各种regularization/clipping/importance sampling来"修补"。Karpathy你之前在numerical stability of RNN上讲过类似的intuition——gradient clipping是一种band-aid，根本问题是exploding gradient的math structure。

### 7.2 On-Policy近似的恢复

FP16实际上把off-policy RL（with mismatch）变回了approximately on-policy。这是一个deep insight：很多RLHF文献讨论on-policy vs off-policy的tradeoff，但都默认 $\mu = \pi$。实际上现代RL framework由于engine不同，从来不是真正on-policy的。FP16让我们更接近理论假设的世界。

### 7.3 为什么这种"简单fix"被忽略了

我觉得有几点原因：
1. BF16的industry惯性极强，A100/H100都默认BF16 mixed precision
2. "BF16 = better"的mental model深入人心，因为pre-training确实如此
3. Algorithmic correction（TIS/MIS）更"publishable"，每个team都想发自己的fix
4. Precision问题看似trivial，不会被paper重点讨论

这种"obvious in hindsight"的发现往往是最valuable的——类似BatchNorm、Adam、dropout这种简单但fundamental的改进。

### 7.4 与FP8训练的关系

最近FP8 training（NVIDIA H100、Blackwell）兴起，paper在Discussion里提到FP8的成功暗示FP16的range限制可以engineering解决。但我个人觉得FP8在RL fine-tuning可能更糟——FP8的mantissa更少（E4M3或E5M2，3或2 mantissa bits），mismatch会更严重。FP8更适合pre-training的forward/backward，对sampling-based RL可能不友好。这是个值得探索的方向。

### 7.5 训练动态观察

Section 4.2最后提到一个有意思的现象：collapse前的算法都会先经历mismatch growing phase。$\pi(\cdot|\theta') - \mu(\cdot|\theta')$ 趋向极端值（一个→1另一个→0），即使同一份weights。这暗示存在一个positive feedback loop：mismatch导致biased gradient，biased gradient让policy push到某些tokens，这些tokens在两个engine下分歧更大，mismatch进一步放大。

FP16打破这个loop：mismatch初始就小，不会进入这个blow-up regime。这是个很干净的critical phenomenon解释。

---

## 8. 工程实现建议

### 8.1 代码改动量

Paper说只需few lines of code change。具体在PyTorch ecosystem：

```python
# Before (BF16)
model = model.to(dtype=torch.bfloat16)
amp_config = {"dtype": torch.bfloat16}

# After (FP16)
model = model.to(dtype=torch.float16)
amp_config = {"dtype": torch.float16, "init_scale": 2**16, "growth_interval": 2000}
```

VeRL/Oat框架里也是config flag切换。

### 8.2 Caveats

- **Large model上的overflow风险**：14B验证过OK，但100B+需要谨慎，建议保留dynamic loss scaling
- **Activation checkpointing**：FP16下保存的activation精度也变高，memory可能略增
- **Communication**：FP16的gradient all-reduce与BF16带宽相同，无额外开销
- **Optimizer state**：建议Adam的exp_avg/sq保持FP32（这是mixed precision的标准做法）

### 8.3 与vLLM/SGLang的兼容

vLLM默认支持FP16 inference，所以切到FP16 training后两边对齐很容易。这里有个subtle点：vLLM的PagedAttention在FP16下的KV cache与BF16可能有slightly不同的memory footprint，但实测无显著影响。

---

## 9. 我的批判性思考

虽然paper的发现很strong，但有几个open question值得讨论：

1. **为什么GSPO不用 $\mu$ 反而更稳？** Paper没完全解释。我猜测是因为GSPO的clip设计更conservative，本质是高bias低variance，BF16下mismatch放大了它的bias但没让它进入collapse regime。需要更深入分析。

2. **FP16在更大model上的极限？** 14B work，但70B、405B呢？FP16的overflow风险随scale增加（某些activation可能突破65504上限）。需要看Llama-3-70B或DeepSeek-V3级别的实验。

3. **Other RL algorithms？** 只测了GRPO family和PG。PPO（with value function）、ReMax、RLOO呢？理论上结论应该一致，但empirical验证更好。

4. **Long context下的mismatch**：Figure 2显示mismatch随sequence length变宽，FP16下也变宽但慢。32K+ context下的RL（如LongRL）是否还能work？这关系到o1-like long reasoning训练。

5. **Numerical determinism的更深层问题**：即使FP16，不同GPU间也可能有non-determinism（atomicAdd顺序）。是否需要更激进的deterministic mode？Horace He的blog（https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/）讨论了inference侧的determinism，但训练侧的determinism vs efficiency tradeoff还未解决。

---

## 10. Related Work链接

为了方便深挖，把关键related work列一下：

- **Paper GitHub**: https://github.com/sail-sg/Precision-RL
- **VeRL framework**: https://github.com/volcengine/verl
- **Oat framework**: https://github.com/sail-sg/oat
- **Yao et al. off-policy RL blog**: https://fengyao.notion.site/off-policy-rl
- **Liu et al. RL collapse analysis**: https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Inference-Training-Mismatch-271211a558b7808d8b12d403fd15edda
- **Horace He on inference determinism**: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- **Schulman on LoRA**: https://thinkingmachines.ai/blog/lora/
- **Mixed precision training (Micikevicius et al.)**: https://arxiv.org/abs/1710.03740
- **BF16 study (Kalamkar et al.)**: https://arxiv.org/abs/1905.12322
- **DeepSeek-R1**: https://arxiv.org/abs/2501.12948
- **GRPO (DeepSeekMath)**: https://arxiv.org/abs/2402.03300
- **DAPO**: https://arxiv.org/abs/2503.14476
- **GSPO**: https://arxiv.org/abs/2507.18071
- **OctoThinker**: https://arxiv.org/abs/2506.20512
- **vLLM**: https://arxiv.org/abs/2309.06180
- **DeepSpeed**: https://arxiv.org/abs/2208.06555 (或者原始technical report)
- **PyTorch FSDP**: https://arxiv.org/abs/2304.11277

---

## 11. 最终Intuition总结

用一个mental model总结：把training-inference mismatch想象成两个engine在数值上"口音不同"。BF16像把音素quantize到26个phoneme，每个token的发音都有~1%的口音偏差，autoregressive generation说1000个token的句子，cumulative口音偏差让两个engine的"sentence probability"差出几个数量级。FP16像quantize到208个phoneme，口音偏差~0.1%，cumulative effect小24×，足够让两个engine的输出"听起来一样"。

这个fix的优雅之处在于：
1. **Root cause fix**：直接消除noise source，不是补偿
2. **Zero algorithmic complexity**：删掉所有TIS/MIS machinery
3. **Deployment gap closed**：因为 $\mu \approx \pi$，对 $\pi$ 优化就是对 $\mu$ 优化
4. **Universal**：跨model family、algorithm、framework、size都work
5. **Essentially free**：FP16 inference速度与BF16相同，loss scaling是成熟infrastructure

Karpathy我觉得这是2025年至今最重要的RL fine-tuning paper之一。它不是发明新算法，而是揭示了一个被overlooked的numerical root cause，让整个field重新审视precision tradeoff。这种"用engineering/numerics视角解决algorithmic problem"的taste，让我想起你之前在Tesla AI Day讲的"first principles thinking"——很多看起来需要fancy algorithm的问题，本质上是被numerical/scale issues mask的simpler problem。

希望这个解读对你有intuition帮助，如果想深挖某个section我可以继续展开。
