---
source_pdf: SAGEATTENTION ACCURATE 8-BIT ATTENTION FOR PLUG-AND-PLAY INFERENCE ACCELERATION.pdf
paper_sha256: 3b80a15770f24e1622b0548b58b5d925960b82b3b34627551fb29e28b970d451
processed_at: '2026-08-12T02:49:19-07:00'
target_folder: LLM-engine/Torch
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SageAttention 用人话讲一遍

## 这篇paper到底在搞啥？

一句话总结：**把attention计算从FP16压缩到INT8，速度翻倍，精度几乎不掉**。

---

## 为啥要搞这个？

### 场景痛点

现在所有大模型都用Transformer架构，Transformer的核心是attention。attention有个致命问题：计算量跟sequence length的平方成正比。

举个例子：
- sequence length = 1024，attention算 $1024 \times 1024 = 1M$ 次
- sequence length = 8192，attention算 $8192 \times 8192 = 67M$ 次
- sequence length = 128K，attention算 $128K \times 128K \approx 16B$ 次

涨了16000倍！而linear layer（比如MLP）只跟sequence length成正比，涨128倍。

所以当sequence很长时（长文本、视频生成、图像生成），**attention成了最大瓶颈**。

### 现有方案的尴尬

大家都知道quantization能加速，比如把FP16变成INT8，速度能快2-4倍。但现有quantization方法都在搞linear layer（比如AWQ、GPTQ、SmoothQuant），没人敢动attention。

FlashAttention3倒是搞了个FP8 attention，但：
1. 只能在H100（Hopper架构）上跑，普通RTX4090/3090用不了
2. 在很多模型上精度直接崩了——Unidiffuser生成的图全是糊的，Llama2在MMLU上掉到25.5%（随机猜水平）

所以attention quantization一直是个hard problem。

---

## 难在哪？两个核心挑战

### 挑战C1：K矩阵有outlier

文章做了个观察（Figure 4）：attention里的K矩阵有channel-wise outlier。啥意思呢？

K矩阵shape是 $N \times d$，N是token数，d是head dimension（64或128）。某些channel（列）的值远大于其他channel。

这就麻烦了——quantization的本质是把值映射到[-127, +127]范围。如果有几个特别大的值，scale就会被拉大，其他正常值就被压缩到很小的范围，精度全没了。

**为什么不能用SmoothQuant的标准trick？**

SmoothQuant的做法是：把K的outlier"搬"到Q上去
$$K' = K \cdot \gamma, \quad Q' = Q / \gamma$$

但这里Q自己也有outlier，你把K的outlier搬到Q，Q就更糟了。这条路走不通。

### 挑战C2：P和V的量化精度不稳

就算Q、K搞定了，P（softmax后的权重）和V量化到INT8在某些层会有很大误差。Table 3显示worst-case cosine similarity只有76%，这在生成任务里是致命的。

---

## SageAttention的三个关键创新

### 创新1：Smooth K（最精妙的trick）

**观察**：K的channel outlier其实是个**shared bias**——所有token共享一个大bias，再加上每个token的小波动。

数学上：$K[t, :] = \text{mean}(K) + \epsilon_t$

**做法**：直接减掉mean
$$\gamma(K) = K - \text{mean}(K)$$

**为什么这招有效且无损？**

这里用到了softmax的**translation invariance**：softmax对所有logit加同一个常数，结果不变。

对任意query $q$：
$$\sigma(q(K - \text{mean}(K))^\top) = \sigma(qK^\top - q \cdot \text{mean}(K)) = \sigma(qK^\top)$$

因为 $q \cdot \text{mean}(K)$ 对所有token是同一个值（不依赖token index），softmax会把它消掉。

**人话解释**：K的所有token都有个共同的大bias，这个bias对每个query贡献的是同一个常数，softmax一算就没了。所以提前减掉，attention结果完全不变，但K的值分布变得很nice，quantization精度大幅提升。

**效果**（Table 18）：
- 不smooth：cosine sim 62.24%（崩了）
- smooth后：cosine sim 99.47%（完美）

**开销**：小于0.2%，基本免费。

### 创新2：选INT8不选FP8

这个决策很反直觉——大家都觉得FP8比INT8先进。

**原因1：硬件速度**

在RTX4090/3090（Ampere/Ada Lovelace架构）上：
- INT8 Matmul：4x faster than FP16
- FP8 Matmul：只跟FP16持平或稍快

因为FP8在这些GPU上用的是FP16 accumulator模式，没真正享受低精度的红利。而INT8有专门的Tensor Core指令。

**原因2：精度更高**

Table 2的实验：
- Q,K用INT8：cosine sim 99.54%
- Q,K用E4M3（FP8的一种）：92.83%
- Q,K用E5M2（FP8的另一种）：77.95%

为啥？因为Q、K的值分布不是那种"少数大值+很多小值"的long-tail（FP8擅长处理这种），而是相对均匀的分布，INT8的均匀采样更合适。

### 创新3：P和V保持FP16，用FP16 accumulator

这是最"离经叛"的设计。

**传统教条**：Matmul的accumulator必须用高精度（FP32），否则大数相加会丢精度。

**SageAttention的反套路**：
- $\widetilde{P}$（softmax中间结果）保持FP16
- $V$ 保持FP16
- accumulator也用FP16

**为什么敢这么干？**

1. **数值范围friendly**：$\widetilde{P} = \exp(S - m)$，每个block里max已经减掉了，值域在[0, 1]。FP16表示这个范围绰绰有余。

2. **V也friendly**：V通常是LayerNorm后的值，分布比较规整。

3. **没有catastrophic cancellation**：accumulator精度损失主要来自"大正数加大负数"相消，但这里都是正值小数相加，不会出问题。

**实验验证**（Table 4 & 5）：
- FP32 accumulator：avg 99.98%, worst 99.84%
- FP16 accumulator：avg 99.98%, worst 99.84%

完全一样！没有任何精度损失。

**速度红利**：在RTX4090上，FP16+FP16 accumulator比FP16+FP32 accumulator快2倍。同时节省register资源。

---

## 完整流程串起来

### 数学表达

**量化阶段**：
$$\hat{Q} = \text{INT8}(Q/\sqrt{d}), \quad \hat{K} = \text{INT8}(K - \text{mean}(K))$$
$$\widetilde{P}, V \text{ 保持 FP16}$$

**计算阶段**（FlashAttention的tiling框架内）：
$$S = \hat{Q}\hat{K}^\top \times \delta_Q \times \delta_K \quad \text{（INT8 mma + dequant）}$$
$$\widetilde{P} = \exp(S - m) \quad \text{（FP32域内做softmax）}$$
$$O = \widetilde{P} \cdot V \quad \text{（FP16 mma + FP16 accumulator）}$$

### 实现细节

1. **Fusion trick**：把quantization融合进前一个op。比如Q在ROPE计算后，结果还在shared memory时直接量化成INT8再写回HBM，避免写高精度数据。

2. **$1/\sqrt{d}$前移**：本来在attention内部做的scaling，提前到quantize Q时做掉。

3. **Block size**：$b_q = 128, b_{kv} = 64$。这个选择平衡了parallelism和memory reuse。

4. **Adaptive selection**：离线测试每层用更激进的SAGEAttn-vB（全INT8）的cosine similarity，如果>99.8%就用vB（更快），否则用B（FP16 PV，更稳）。这样在CogvideoX上能多拿11.7%速度。

---

## 速度数字解读

### Kernel级速度

**RTX4090, headdim=64**：
- FlashAttention2：约165 TOPS
- xformers：约120 TOPS
- SageAttention：**341 TOPS**

提速幅度：
- vs FlashAttention2：**2.1x**
- vs xformers：**2.7x**

RTX4090的INT8理论peak是656 TOPS，SageAttention达到52%的utilization，这在kernel优化里算很高的了。

对比H100上的FlashAttention3：490 TOPS。RTX4090的SageAttention 341 TOPS已经很接近，但RTX4090便宜多了。

### End-to-end速度

**RTX4090** (Table 7)：
- CogvideoX：163.37 → 327.57 TOPS，**2.01x**
- Llama2：130.99 → 231.74 TOPS，1.77x
- UltraPixel：152.03 → 325.18 TOPS，2.14x
- Unidiffuser：105.68 → 246.93 TOPS，2.34x
- TIMM：18.91 → 111.41 TOPS，5.89x（这个夸张，因为baseline是naive torch attention）

平均2.83x speedup。

---

## 精度损失验证

这是最关键的部分——速度快没用，精度崩了就白搭。

### Llama2（语言模型）
- WikiText PPL：5.823 → 5.824（几乎无差）
- LAMBADA Acc：0.886 → 0.887（甚至更好）
- MMLU Acc：0.46 → 0.46（完全一样）

### CogvideoX（文本生成视频）
- CLIPSIM（文本对齐）：0.1837 → 0.1836
- CLIP-T（时间一致性）：0.9976 → 0.9976
- VQA-a（美学质量）：68.962 → 68.839
- VQA-t（技术质量）：75.925 → 75.037
- FScore（时序一致性）：3.7684 → 3.8339（甚至更好）

### Unidiffuser（文本生成图像）
- FID：163.33 → 166.49（轻微退化，但比FP8的395.99好太多）
- sFID：145.08 → 143.18（甚至更好）
- CLIP：0.3152 → 0.3154（几乎一样）

### TIMM（图像分类）
- ImageNet：84.79% → 84.74%（-0.05%）
- Sketch：45.32% → 45.78%（+0.46%，更好）
- ImageNet-r：59.55% → 60.32%（+0.77%，更好）

### Llava1.6（视觉问答）
- TextVQA：60.25% → 60.09%
- POPE：86.45% → 86.44%
- VQAv2：77.55% → 77.47%

**总结**：所有任务精度损失都在0.2%以内，有些甚至更好。这种波动基本在噪声范围内。

---

## 为什么能work的深层原因

### Insight 1：attention里的K有"双重身份"

K其实承担两个角色：
1. **Bias角色**：所有token共享的static信息
2. **Signal角色**：token-specific的dynamic信息

Smooth K就是把bias部分剥离掉，让quantizer只处理signal部分。

这跟Attention Sinks（Xiao et al., 2023）的发现呼应：transformer会学到"占位"token，承担system-level bias。

### Insight 2：outlier控制是quantization成败关键

对比Llama2和Unidiffuser：
- Llama2的Q、K、V分布uniform，没outlier，直接INT8量化几乎无损
- Unidiffuser的K有严重outlier，直接量化直接崩

这说明：**模型设计时如果能控制outlier，quantization就更容易**。

### Insight 3：FP16 accumulator被严重低估

传统教条说accumulator必须高精度，但SageAttention证明：如果数值分布friendly（范围小、正值、无对消），FP16 accumulator完全够用。

这个insight可能启发更多场景的低精度优化。

### Insight 4：硬件native格式 > 理论最优格式

FP8理论上比INT8更先进（浮点数表示），但在RTX4090上反而更慢。选quantization format不能光看paper spec，要看实际硬件的microarchitecture。

---

## 跟其他方法的关系

### 正交关系

SageAttention跟以下方法**正交**，可以叠加：
- AWQ（weight quantization for LLM）
- GPTQ（weight quantization）
- Q-Diffusion（diffusion model quantization）
- ViDiT-Q（video diffusion quantization）

这些方法主要搞linear layer，SageAttention搞attention，互不冲突。

Table 13的实验：AWQ + SageAttention on Llama2，PPL从5.5988（AWQ only）到5.5998（AWQ + Sage），几乎无损叠加。

### 跟FlashAttention3的对比

| 维度 | SageAttention | FlashAttention3 |
|------|---------------|-----------------|
| 硬件要求 | RTX4090/3090 | H100（Hopper独占）|
| 数据格式 | INT8 QK + FP16 PV | FP8 |
| Unidiffuser FID | 166.49 | 394.13（崩了）|
| UltraPixel FID | 179.79 | 383.61（崩了）|
| CogvideoX FScore | 3.8339 | 3.394 |

SageAttention在消费级GPU上精度全面碾压H100上的FP8 FlashAttention3。

---

## 实际使用建议

### 直接plug-and-play

```python
# 原来
attention = FlashAttention()

# 替换成
attention = SageAttention()  # 默认用SAGEAttn-B
```

### 选版本策略

1. **求稳**：用SAGEAttn-B（FP16 PV + FP16 accumulator）
2. **求极致速度**：用adaptive selection，对每层测cosine sim，>99.8%用SAGEAttn-vB
3. **LLM场景**：如果Q、K没outlier（如Llama2），smooth K可以关掉省那0.2%

### 注意事项

1. **硬件限制**：主要优化RTX4090/3090。在H100上FlashAttention3 FP8可能更快，需要benchmark。
2. **Headdim限制**：测试了64和128，256+的headdim需要自己验证。
3. **只支持inference**：training backward需要更高精度，这个方法不适用。
4. **Self-attention为主**：cross-attention（encoder-decoder）没明确测试。

---

## 我的核心takeaway

1. **Smooth K的softmax translation invariance trick**是这篇paper最漂亮的地方。免费的精度提升，因为数学上就保证无损。

2. **FP16 accumulator**打破了"accumulator必须高精度"的教条。关键在于理解数值分布——如果都是[0,1]的正值小数，FP16完全够用。

3. **硬件native > 理论先进**：INT8在RTX4090上比FP8快，是因为microarchitecture设计。选quantization format必须结合实际硬件。

4. **Attention quantization是可行的**，但需要task-specific design。直接套linear layer的quantization方法会崩，因为attention有softmax这个非线性操作 + K的outlier pattern。

5. **Worst-case比average重要**：Table 2的average精度看起来都还好，但Table 3的worst-case暴露了INT8 PV的问题。quantization评估必须看worst-case layer。

---

## 参考链接

- SageAttention paper: https://arxiv.org/abs/2410.02367
- SageAttention code: https://github.com/thu-ml/SageAttention
- FlashAttention2: https://arxiv.org/abs/2307.08691
- FlashAttention3: https://arxiv.org/abs/2407.08608
- SmoothQuant: https://arxiv.org/abs/2211.10438
- AWQ: https://arxiv.org/abs/2306.00978
- Attention Sinks: https://arxiv.org/abs/2309.17453
- Online Softmax: https://arxiv.org/abs/1805.02867
- Triton: https://github.com/openai/triton
- xformers: https://github.com/facebookresearch/xformers

---

# SageAttention: Accurate 8-bit Attention for Plug-and-Play Inference Acceleration 深度解析

## 1. 核心动机与问题定位

Attention computation 在 long sequence scenario 中已经成为 primary bottleneck。文章开篇用 Figure 2 给出了一个关键观察：当 sequence length 增长到 8K-128K 时，attention 的 latency 远超 linear transformation，原因是 attention 的复杂度是 $O(N^2)$，而 linear layer 是 $O(N)$。

这里 $N$ 表示 sequence length（token 数量），$d$ 表示 head dimension（通常是 64 或 128）。

现有 quantization 工作（如 AWQ、GPTQ、SmoothQuant）几乎都聚焦在 **linear layer** 上，把 attention 留在 FP16 高精度计算。FlashAttention3 虽然提供 FP8 版本，但是：
- 独占 Nvidia Hopper 架构（如 H100）
- 在很多模型上直接 FP8 attention 会带来严重性能退化

文章 Figure 3 给出一个 striking example：在 Unidiffuser 这个 text-to-image 模型上，直接用 INT8 或 FlashAttention3 的 FP8 实现会生成完全模糊的图像。Llama2 在 MMLU 上甚至退化到 25.5% 的 random-guessing level。

**核心挑战 C1**: matrix K 存在显著的 channel-wise outlier
**核心挑战 C2**: 简单把 (P, V) 量化到 INT8 不能保证 PV 的精度

参考链接：
- FlashAttention2: https://arxiv.org/abs/2307.08691
- FlashAttention3: https://arxiv.org/abs/2407.08608
- SmoothQuant: https://arxiv.org/abs/2211.10438

---

## 2. 数学公式与 FlashAttention 基础回顾

### 2.1 标准 Self-Attention 公式

$$S = QK^\top / \sqrt{d}, \quad P = \sigma(S), \quad O = PV$$

变量含义：
- $Q, K, V \in \mathbb{R}^{N \times d}$：query, key, value matrices
- $S \in \mathbb{R}^{N \times N}$：attention logits（未归一化的 score）
- $P \in \mathbb{R}^{N \times N}$：attention probability（softmax 后的权重）
- $O \in \mathbb{R}^{N \times d}$：output
- $\sigma(S)_{ij} = \exp(S_{ij}) / \sum_k \exp(S_{ik})$：softmax 函数
- $\sqrt{d}$：scaling factor，防止 dot product 过大导致 softmax 饱和

### 2.2 FlashAttention 的 Online Softmax

FlashAttention 的核心 idea 是 **tiling** + **online softmax**，避免把 $N \times N$ 的 $S, P$ 写回 global memory。

把 $Q, K, V$ 沿 token 维度切成 blocks $\{Q_i\}, \{K_j\}, \{V_j\}$，block size 为 $b_q, b_{kv}$。

迭代公式：

$$S_i^j = Q_i K_j^\top / \sqrt{d}$$
$$(m_i^j, \widetilde{P}_i^j) = \tilde{\sigma}(m_i^{j-1}, S_i^j)$$
$$l_i^j = \exp(m_i^{j-1} - m_i^j) l_i^{j-1} + \text{rowsum}(\widetilde{P}_i^j)$$
$$O_i^j = \text{diag}(\exp(m_i^{j-1} - m_i^j)) O_i^{j-1} + \widetilde{P}_i^j V_j$$

变量含义：
- 上标 $j$ 表示第 $j$ 个 K/V block 的迭代
- 下标 $i$ 表示第 $i$ 个 Q block
- $m_i^j \in \mathbb{R}^{b_q \times 1}$：running max，初始化为 $-\infty$
- $l_i^j \in \mathbb{R}^{b_q \times 1}$：running rowsum，初始化为 0
- $\widetilde{P}_i^j = \exp(S_i^j - m_i^j)$：unnormalized probability（还没除以 rowsum）
- $\tilde{\sigma}$：online softmax operator，$m_i^j = \max\{m_i^{j-1}, \text{rowmax}(S_i^j)\}$

最终输出：$O_i = \text{diag}(l_i^{T_n})^{-1} O_i^{T_n}$，即除以累积的 rowsum。

**Intuition**: online softmax 的关键是 max 的 rescaling trick。当新 block 的 max 比之前的大时，旧的累积值需要乘以 $\exp(\text{old\_max} - \text{new\_max})$ 来 "discount"，这个 trick 让 streaming 计算 softmax 成为可能。

参考：https://arxiv.org/abs/1805.02867 (Online normalizer calculation for softmax)

### 2.3 Dynamic Quantization 基础

矩阵乘法 $C = AB$ 的量化加速：

$$(\delta_A, \hat{A}) = \psi(A), \quad (\delta_B, \hat{B}) = \psi(B), \quad \hat{C} = \hat{A}\hat{B}, \quad C = \psi^{-1}_{\delta_A \delta_B}(\hat{C})$$

变量含义：
- $\psi$：quantizer，把 FP32 矩阵转成 INT8/FP8 + scale
- $\hat{A}, \hat{B}$：量化后的低精度矩阵
- $\delta_A, \delta_B$：scale factor
- $\psi^{-1}$：dequantizer，$C \approx \delta_A \delta_B \hat{A}\hat{B}$

**Granularity 选项**:
- **Per-tensor**: 整个 tensor 一个 scale，$\delta_A = \max(|A|)/127$
- **Per-token**: 每个 token 一个 scale，$\delta_A[i,:] = \max(|A[i,:]|)/127$
- **Per-channel**: 每个 channel 一个 scale，$\delta_A[:,i] = \max(|A[:,i]|)/127$
- **Per-block**: 每 $b=m-n$ 个 token 共享一个 scale，$\delta_A = \max(|A[m:n,:]|)/127$

---

## 3. SageAttention 的三大核心技术

### 3.1 Smooth Matrix K：解决 channel-wise outlier

**问题诊断**: Figure 4 展示了 CogvideoX 和 Unidiffuser 的 Q, K, V 分布。关键观察：K 存在显著的 **channel-wise outlier**，即某些 channel 的值远大于其他 channel。

为什么不能用 SmoothQuant 的标准方案？SmoothQuant 把 K 的 outlier "smooth" 到 Q 上：
$$K' = K \cdot \text{diag}(\gamma), \quad Q' = Q / \text{diag}(\gamma)$$

但这里 Q 本身也受 outlier 影响，所以这种迁移策略会损害 Q 的精度。

**SageAttention 的洞察**: K 的 channel outlier 其实是 **shared bias**，而非 token-wise variation。也就是说：

$$K[t, :] = \text{mean}(K) + \epsilon_t$$

其中 $\text{mean}(K) = \frac{1}{N}\sum_{t=1}^N K[t,:]$ 是 $1 \times d$ 的平均 key，$\epsilon_t$ 是 token-wise 的小波动。

**Smooth 操作**:

$$\gamma(K) = K - \text{mean}(K)$$

**为什么这个操作不改变 attention score P**？因为对任意 query $q$：

$$\sigma(q(K - \text{mean}(K))^\top) = \sigma(qK^\top - q \cdot \text{mean}(K)) = \sigma(qK^\top)$$

这里用到了 softmax 的 **translation invariance** 性质：softmax 对所有 logit 加同一个常数不变。

**Intuition**: 当 K 的所有 token 共享同一个大 bias 时，这个 bias 对每个 query 都贡献同一个常数，softmax 后被消除。所以减去 mean(K) 等于 "免费" 地把 outlier 消除掉。

**实验验证** (Table 18):
- Without smooth K, per-token quant: Cos Sim 62.24%, Relative L1 1.187
- With smooth K, per-token quant: Cos Sim 99.47%, Relative L1 0.045
- Without smooth K, per-block quant: Cos Sim 30.60%（几乎崩溃）
- With smooth K, per-block quant: Cos Sim 99.31%

Speed overhead < 0.2%（Table 10），CogvideoX 从 327.57 TOPS 降到 327.52 TOPS，几乎无损。

### 3.2 INT8 vs FP8 的选择

**为什么选 INT8 而非 FP8**？两个原因：

1. **硬件速度**: 在 RTX4090/3090 上，INT8 Matmul 比 FP16 快 4 倍，比 FP8 快 2 倍。FP8 在这些消费级 GPU 上反而更慢。

2. **精度**: Table 2 显示 Q, K 量化到 INT8 比 E4M3（4位指数3位尾数+1位符号）和 E5M2（5位指数2位尾数）精度更高。

**FP8 格式详解**:
- E4M3: 1 sign + 4 exponent + 3 mantissa = 8 bits。exponent bias = 7，范围约 ±448，精度高，适合 forward pass
- E5M2: 1 sign + 5 exponent + 2 mantissa = 8 bits。exponent bias = 15，范围约 ±57344，精度低但动态范围大，适合 gradient

Table 2 平均精度（Cos Sim）:
- Q, K = INT8, P, V = E4M3: 99.94%
- Q, K = INT8, P, V = E5M2: 99.81%
- Q, K = INT8, P, V = INT8: 99.70%
- Q, K = E5M2, P, V = INT8: 99.13%（最差）

Table 3 worst-case 精度：
- INT8 + INT8: Cos Sim 76.36%
- INT8 + FP16: **Cos Sim 99.99%**（最关键的数据点）

### 3.3 FP16 Accumulator for PV：精度与速度的双赢

这是文章最巧妙的设计。问题：INT8 quantize P, V 会带来 worst-case 大误差（Table 3）。

**SageAttention 的方案**: 把 $\widetilde{P}, V$ 保持 FP16，并用 **FP16 accumulator** 而非 FP32 accumulator 计算 $\widetilde{P}V$。

**为什么 FP16 accumulator 可行**？
1. Table 4 & 5 显示 FP16 vs FP32 accumulator 的精度完全一致（平均 99.98% vs 99.98%，worst 99.84% vs 99.84%）
2. 在 RTX4090 上 FP16+FP16 accumulator 的 Matmul 比 FP16+FP32 accumulator 快 2 倍
3. FP16 accumulator 节省寄存器资源，间接加速

**Intuition**: $\widetilde{P}$ 是 $\exp(S - m)$，每个 block 内 max 已经被减掉，值域在 [0,1]，FP16 表示这个范围完全够用。V 通常是经过 LayerNorm 后的值，分布也比较 friendly。

### 3.4 完整 Formulation

SageAttention 的完整数学表达：

**Quantization**:
$$(\delta_Q, \hat{Q}) = \psi_Q(Q/\sqrt{d}), \quad (\delta_K, \hat{K}) = \phi_K(K), \quad (\delta_P, \hat{P}) = \psi_P(\widetilde{P}), \quad (\delta_V, \hat{V}) = \psi_V(V)$$

**Attention**:
$$S = \psi^{-1}_{\delta_Q \delta_K}(\hat{Q}\hat{K}^\top), \quad (m', P) = \tilde{\sigma}(m, S)$$
$$O = \text{diag}(\exp(m' - m)) O + \psi^{-1}_{\delta_P \delta_V}(\hat{P}\hat{V})$$

其中 $\phi_K = \psi_K \circ \gamma$，即先 smooth 再 quantize。

**四个 kernel 变体** (Table 6):

| Kernel | Q, K 量化 | P 量化 | V 量化 |
|--------|----------|--------|--------|
| SAGEAttn-T | per-token, INT8 | FP16, FP16 acc | FP16, FP16 acc |
| SAGEAttn-B | per-block, INT8 | FP16, FP16 acc | FP16, FP16 acc |
| SAGEAttn-vT | per-token, INT8 | per-block, INT8 | per-channel, INT8 |
| SAGEAttn-vB | per-block, INT8 | per-block, INT8 | per-channel, INT8 |

**Adaptive selection**: 离线测试每层 SAGEAttn-vB 的 cosine similarity，如果 > 99.8% 则用 vB（更快），否则用 B（更准确）。Table 11 显示在 CogvideoX 上提升 11.7% 速度，Llama2 上从 208.59 到 231.74 TOPS。

---

## 4. CUDA/Triton 实现细节

### 4.1 Fusion Tricks

1. **ROPE + Quantization fusion**: 在 ROPE 计算 $A$ 后、写回 global memory 之前，直接在 shared memory 里完成 $\delta_A, \hat{A} = \psi(A)$，然后写量化后的数据。这避免了高精度 Q 的 HBM 写入。

2. **$1/\sqrt{d}$ fusion**: 把 attention 内部的 scaling factor 移到 quantization 阶段，在 quantize Q 时直接乘 $1/\sqrt{d}$。

### 4.2 硬件指令

- INT8 mma: `u8.u8.s32`（输入 u8，累加 s32）
- FP16 mma with FP16 accumulator: `f16.f16.f16`

### 4.3 Hyper-parameters (Table 12)

| HeadDim | Causal Mask | Num_Warps | Num_Stages |
|---------|-------------|-----------|------------|
| 64 | False | 4 | 3 |
| 64 | True | 4 | 4 |
| 128 | False | 8 | 3 |
| 128 | True | 8 | 5 |

Block size: $b_q = 128$, $b_{kv} = 64$。

Num_Warps 是 warp scheduler 数量，Num_Stages 是 pipeline 阶段数（软件 prefetching depth）。

### 4.4 理论 throughput

RTX4090 INT8 理论 peak 656 TOPS。SageAttention 达到 340 TOPS，达到 52% 理论值。FlashAttention2 在 FP16 上只有 165 TOPS。

对比 H100 上的 FlashAttention3 490 TOPS，SageAttention 在 RTX4090 上接近这个水平，但 H100 的 INT8 peak 远高于 RTX4090。

参考：https://github.com/thu-ml/SageAttention

---

## 5. 实验结果深度分析

### 5.1 速度 (Figure 6-9)

**RTX4090, headdim=64**:
- SageAttention peak 341 TOPS
- vs FlashAttention2: 2x faster
- vs xformers: 2.9x faster

**RTX4090, headdim=128**: 类似 speedup

**RTX3090** (Table 19):
- CogvideoX: 71.57 → 129.87 TOPS, 1.81x
- Llama2: 56.54 → 108.91 TOPS, 1.93x
- UltraPixel: 65.86 → 131.74 TOPS, 2.00x
- Unidiffuser: 47.64 → 108.91 TOPS, 2.29x
- TIMM: 12.33 → 66.34 TOPS, 5.38x

### 5.2 End-to-end Metrics (Table 8)

**Llama2** (Language):
- WikiText PPL: 5.823 → 5.824（几乎无差）
- LAMBADA Acc: 0.886 → 0.887
- MMLU Acc: 0.46 → 0.46

**CogvideoX** (Text-to-Video):
- CLIPSIM: 0.1837 → 0.1836
- CLIP-T: 0.9976 → 0.9976
- VQA-a: 68.962 → 68.839
- VQA-t: 75.925 → 75.037
- FScore: 3.7684 → 3.8339（甚至略好）

**Unidiffuser** (Text-to-Image):
- FID: 163.33 → 166.49（轻微退化）
- sFID: 145.08 → 143.18
- CLIP: 0.3152 → 0.3154
- ImageReward: 0.1609 → 0.1521

**UltraPixel** (High-res Image):
- FID: 179.78 → 179.79（几乎无差）
- sFID: 141.35 → 141.63
- CLIP: 0.3132 → 0.3131
- ImageReward: 0.6169 → 0.6110

**TIMM** (Image Classification):
- ImageNet: 84.79% → 84.74%
- Sketch: 45.32% → 45.78%（甚至更好）
- ImageNet-r: 59.55% → 60.32%（甚至更好）

**Llava1.6** (VQA):
- TextVQA: 60.25% → 60.09%
- POPE: 86.45% → 86.44%
- VQAv2: 77.55% → 77.47%

### 5.3 与其他 quantization 方法的正交性

Table 13: SageAttention + AWQ (W4A16) on Llama2
- Full: PPL 5.4721
- SageAttention only: 5.4729
- AWQ only: 5.5988
- AWQ + SageAttention: 5.5998

这说明 SageAttention 与 weight quantization（AWQ）**正交**，可以叠加使用。

Table 14: SageAttention vs Q-diffusion on Unidiffuser
- Q-diffusion W8A8: FID 395.99（崩坏）
- SageAttention: FID 166.49

Table 15: SageAttention vs ViDiT-Q on CogvideoX
- SageAttention: end-to-end 34.3% speedup
- ViDiT-Q: 理论最大 22%（无开源加速代码）

---

## 6. 关键 Insights 与直觉构建

### 6.1 为什么 LLM 的 attention 量化"容易"？

Table 1 + Appendix A.6 的 insight：Llama2 的 Q, K, V 分布相对 **uniform**，没有显著 outlier，所以 INT8 / FP8 quantization 几乎无损。这说明：**outlier 控制是 quantization 成功的关键**。

### 6.2 为什么 attention quantization 比 linear layer 难？

1. **Softmax 是非线性**: 小误差在 softmax 中可能被放大（特别是当 max 不稳定时）
2. **两个 matmul 串联**: $QK^\top$ 的误差会传入 $P$，再传入 $PV$，error compounding
3. **outlier 位置不同**: K 的 outlier 在 inner axis（channel），不能用 standard per-channel quantization

### 6.3 Smooth K 的深层含义

这个 trick 揭示了一个重要现象：**attention 的 key 实际上承担着双重角色**：
- Bias 项：所有 token 共享的 "static" 信息
- Signal 项：token-specific 的 "dynamic" 信息

减去 mean(K) 就是把 bias 部分剥离掉，让 quantizer 只处理 signal 部分。这跟 Attention Sinks（Xiao et al., 2023）的观察呼应：transformer 学到了"占位" token，承担 system-level bias。

参考：https://arxiv.org/abs/2309.17453 (Efficient Streaming Language Models with Attention Sinks)

### 6.4 FP16 accumulator 为什么无损？

标准 wisdom 是 Matmul accumulator 必须用高精度。但 SageAttention 反其道而行：
- $\widetilde{P} \in [0, 1]$，FP16 表示精度足够
- V 经过 LayerNorm，分布 friendly
- Accumulator 的精度损失来自大数相加，但 $\widetilde{P}$ 都 < 1，不会出现 catastrophic cancellation

这挑战了传统 quantization 的"accumulator 必须高精度"教条。

### 6.5 Per-block quantization 的 motivation

为什么选 per-block 而非 per-token？per-token 的 scale factor 数量 = N（很大），per-block = N/b（小很多）。scale factor 是 FP32，存储和计算开销不小。在 FlashAttention tiling 框架下，per-block 自然匹配 block-level 计算，scale 读取一次可复用整个 block 的计算。

---

## 7. Algorithm 1 详解 (SAGEAttn-B)

```
Input: Q, K, V ∈ R^{N×d} (FP16), block sizes b_q, b_{kv}

Preprocessing:
    K = K - mean(K)          // smooth K，消除 channel bias
    
Quantization:
    (δ_Q, Q̂) = ψ_Q(Q/√d)    // per-block INT8
    (δ_K, K̂) = ψ_K(K)       // per-block INT8

Tiling:
    Divide Q̂ into T_m = N/b_q blocks {Q̂_i}
    Divide K̂, V into T_n = N/b_kv blocks {K̂_j}, {V_j}

for i in [1, T_m]:           // parallel over SMs
    Load Q̂_i, δ_Q[i] into SM
    for j in [1, T_n]:
        Load K̂_j, V_j, δ_K[j] into SM
        S_i^j = Matmul(Q̂_i, K̂_j^T) × δ_Q[i] × δ_K[j]   // INT8 mma + FP32 dequant
        m_i^j = max(m_i^{j-1}, rowmax(S_i^j))
        P̃_i^j = exp(S_i^j - m_i^j)
        l_i^j = exp(m_i^{j-1} - m_i^j) l_i^{j-1} + rowsum(P̃_i^j)
        O_i^j = diag(exp(m_i^{j-1} - m_i^j)) O_i^{j-1} + 
                Matmul(P̃_i^j.to(FP16), V_j, accum=FP16)  // FP16 mma with FP16 accum
    O_i = diag(l_i^{T_n})^{-1} O_i^{T_n}
    Write O_i
```

关键点：
1. K 的 smooth 在所有 tile 处理之前完成（一次性）
2. QK^T 用 INT8，但 dequant 后在 FP32 域做 softmax
3. PV 用 FP16 + FP16 accumulator
4. Online softmax 仍保持 full precision（FP32）

---

## 8. 性能数字背后的硬件逻辑

### 8.1 RTX4090 关键 spec

- FP16 Tensor Core: 165 TFLOPS
- INT8 Tensor Core: 330 TFLOPS（理论 peak）
- FP8 Tensor Core: 165 TFLOPS（注意：FP8 在 Ada Lovelace 上没有 INT8 的 4x 优势，只跟 FP16 持平或更快，因为 FP8 用的是 FP16 accumulator 模式）

但文章报告 SageAttention 在 INT8 上达到 340 TOPS，超出 165 TFLOPS 的 FP16 peak，这正是因为 INT8 mma 的 2x 加速 + FP16 accumulator 的 2x 加速叠加。

### 8.2 Memory bandwidth

- 量化后 Q, K 从 FP16 (2 bytes) 变成 INT8 (1 byte)，HBM 读取减半
- $\widetilde{P}, V$ 保持 FP16，所以这块 bandwidth 没省
- 但 P 是 on-chip 计算（FlashAttention style），不写回 HBM

### 8.3 Register pressure

FP16 accumulator 比 FP32 accumulator 节省一半 register。在线 softmax 本身就占大量 register（m, l, O 累积），节省的 register 可以增大 block size 或增加 pipeline stage。

---

## 9. 局限性与未来方向

### 9.1 当前局限

1. **架构绑定**: 实现针对 RTX4090/3090（Ampere/Ada Lovelace）。Hopper 的 FP8 优势没利用。
2. **Headdim 限制**: 主要测试 64 和 128，更大 headdim（如 256）的 KV cache 场景未充分验证
3. **Training 未支持**: 仅 inference，因为 backward 需要更高精度
4. **Only self-attention**: cross-attention（encoder-decoder）未明确测试

### 9.2 与后续工作的联系

作者后续工作 SageAttention2 (ICML 2025)、SageAttention3 引入：
- Q 也 smooth（不只是 K）
- Per-thread INT4 quantization
- Microscaling FP4 attention
- 8-bit training 探索

其他相关工作：
- SparseGen: https://arxiv.org/abs/2502.01776
- SpargeAttn: https://arxiv.org/abs/2502.18137
- JetFire: INT8 training
- MoA: Mixture of sparse attention

---

## 10. 实践建议与 takeaways

如果你要在自己模型上应用 SageAttention：

1. **直接 plug-and-play**: 替换 `attn = FlashAttention()` 为 `attn = SageAttention()`
2. **先跑 SAGEAttn-B**: 最稳的版本，精度几乎无损
3. **若需要更极致速度**: 跑 adaptive quantization，对每层做 cosine sim 测试，选择用 vB 的层
4. **监控 K 的 outlier**: 如果你的模型 K 没有 outlier（如纯 LLM），smooth K 开销可关掉（虽然 < 0.2%）
5. **注意 H100/Blackwell**: 在 Hopper 上 FlashAttention3 FP8 可能更快，需要 benchmark 对比

**核心 takeaway**:
- Attention quantization 是可行的，但需要 task-specific design（smooth K, FP16 accum）
- "硬件 native 数据类型" 优于 "理论更优" 的格式（INT8 > FP8 on consumer GPUs）
- Quantization 的精度瓶颈往往在 worst-case 层，而非 average（Table 3 vs Table 2）
- FP16 accumulator 在数值分布 friendly 时无损，是个被低估的优化

参考链接汇总：
- Paper: https://arxiv.org/abs/2410.02367
- Code: https://github.com/thu-ml/SageAttention
- FlashAttention2: https://arxiv.org/abs/2307.08691
- FlashAttention3: https://arxiv.org/abs/2407.08608
- SmoothQuant: https://arxiv.org/abs/2211.10438
- Triton: https://github.com/openai/triton
- xformers: https://github.com/facebookresearch/xformers
- AWQ: https://arxiv.org/abs/2306.00978
- Q-Diffusion: https://arxiv.org/abs/2302.04304
- ViDiT-Q: https://arxiv.org/abs/2406.02540
- Attention Sinks: https://arxiv.org/abs/2309.17453
- Online Softmax: https://arxiv.org/abs/1805.02867
