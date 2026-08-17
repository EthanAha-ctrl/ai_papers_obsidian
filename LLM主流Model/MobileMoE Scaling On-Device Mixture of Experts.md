---
source_pdf: MobileMoE Scaling On-Device Mixture of Experts.pdf
paper_sha256: c305708eb67b5f26ee13f512fce633ad1658b0dea8c2fff4ae9415dd27e904ad
processed_at: '2026-08-05T19:36:10-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MobileMoE 用人话说

Andrej，咱换个聊法，抛开公式，就讲核心故事。

---

## 一句话版本

手机上跑LLM一直是dense model的天下，这篇paper说：**别用dense了，MoE在手机上其实更香**——同样精度，推理快2-3倍，内存还能装下。

---

## 为什么手机上一直没用MoE

MoE在server上早就赢了——DeepSeek V3、Mixtral、Qwen3全是MoE。但手机上没人认真做过，原因有三个：

1. **没人知道sub-billion scale的MoE该怎么design**。Server上256个expert很正常，手机上memory就5GB，你塞256个expert直接爆了
2. **手机推理框架不支持MoE**。XNNPACK只有dense的INT4 kernel，MoE的sparse routing没现成的op
3. **大家觉得MoE在small scale没意义**。传统观点是MoE要scale大才有benefit

这篇paper把这三个问题全解了。

---

## 核心insight：MoE解耦了memory和compute

Dense model有个死结：**你想推理快，就得参数少；参数少，model就笨**。因为active params = total params，compute和memory绑死。

MoE打破了这个死结：

- **Total params**决定能装多少知识（memory）
- **Active params**决定每个token算多少（compute）

这两个一旦分开，你就可以：**塞一个大model到手机memory里，但每次只激活一小部分来算**。

iPhone 17有12GB DRAM，装个5B参数的MoE完全没问题，但每次只激活0.9B来算——这就是MobileMoE-L。

---

## 怎么找到最优架构的：三个ablation

作者没瞎试，而是formulate了一个scaling law，然后做了三个独立实验：

### Experiment 1: 多少个expert最好？

试了1, 2, 4, 8, 16, 32个expert。

结果：**8个最好**。再多（16, 32）收益递减，反而因为每个expert太小、routing overhead变大而regression。再少（2, 4）sparsity不够，没发挥MoE的优势。

Intuition：手机memory budget 5GB，sub-billion active scale下，8是个sweet spot——够sparse让compute省，又不会太sparse导致expert碎片化。

### Experiment 2: expert要不要切细？

把每个expert切成g份，变成g×E个fine-grained expert。

结果：**g=8最好**。

Intuition：粗粒度expert（比如8个大家伙）就像8个全栈工程师，每个啥都能干但overlap大。细粒度（64个小expert）就像64个专精工程师，router可以组合出更precise的team。组合数从C(8,2)=28变成C(64,8)=44亿，路由空间爆炸。

但g=16就过度了——每个sub-expert太小（768×384的FFN），GEMM效率塌掉，wall-clock多50%但loss只少0.01。

### Experiment 3: 要不要加一个shared expert？

Shared expert就是每个token都必经的"通用expert"，其他routed expert是"专门expert"。

结果：**加了更好**。

Intuition：有些知识是所有token都需要的（比如基本语法、common sense），如果让routed expert都学一遍这些common knowledge，就浪费了。Shared expert兜底common knowledge，routed expert专注specialization。

DeepSeek V3用了shared expert，Qwen3没用。这篇paper在sub-billion scale上验证了shared expert确实有用。

### 最终架构

**60个fine-grained routed expert + 1个shared expert + top-4 routing**

每层都是这个配置。三个scale：
- Small: 0.3B active, 1.3B total（INT4才0.68GB）
- Medium: 0.5B active, 2.8B total（INT4才1.48GB）
- Large: 0.9B active, 5.3B total（INT4才2.75GB）

全都能装进手机。

---

## Training: 四阶段流水线

### Stage 1: Pre-training（6T tokens）

只用了6T tokens，但效果比Llama 3.2 1B（9T tokens + distillation）和SmolLM2 1.7B（11T tokens）都好。

**MoE在sub-billion scale是token-efficient的**。原因是每个token被routed到specialized expert，gradient signal更efficient，不浪费在irrelevant parameters上。

MoE训练有两个坑，作者都填了：

**坑1: Expert load不均衡**。有些expert被疯狂调用，有些几乎没被调。解法：auxiliary-loss-free balancing（DeepSeek V3的trick），根据load动态调expert bias，不用额外loss。还有router z-loss防数值溢出。

**坑2: 小expert的GEMM效率极差**。60个768×384的FFN，如果sequential算，GPU利用率低到离谱。解法：grouped MLP，把所有expert batch成一个fused grouped matmul。为了buffer对齐，用drop-and-pad给每个expert固定size的token buffer。

### Stage 2: Mid-training（500B tokens）

从2048 context扩展到8192，同时把data从web-heavy（62%）切到domain-specific（knowledge 32%, code 22%, math 21%）。

这步是**knowledge injection**的主要阶段。MMLU涨5-10点，DROP涨10-11点。

### Stage 3: SFT（80M samples）

Instruction tuning。这步主要**unlock reasoning**。GSM8K从36涨到52（+16），从55涨到77（+22）。

关键细节：SFT换成dropless dispatch。因为instruction-response pair的结构不能丢token，丢了learning signal就歪了。

### Stage 4: INT4 QAT

把所有linear layer权重量化到4-bit，activation量化到8-bit。

**关键trick：router保持FP32**。Router的输出决定了整个routing decision，量化误差会在这里被放大。0.5%的memory overhead换routing stability，非常值。

量化后memory：
- S: 0.68GB（iPhone 13的4GB都装得下）
- M: 1.48GB
- L: 2.75GB

精度只掉2-3点，但memory压了4倍。

---

## 性能：新Pareto Frontier

### vs Dense baseline

MobileMoE-S（272M active, 0.68GB INT4）vs MobileLLM-Pro（1.1B, 0.55GB INT4）：
- 精度几乎一样（44.0 vs 45.5）
- 但MobileMoE-S用了**3倍少的active params**

MobileMoE-L（922M active）直接干翻所有sub-2B dense model，包括Qwen3.5 2B（1.9B active）。

### vs SOTA MoE

MobileMoE-L vs OLMoE-1B-7B（1.3B active, 6.9B total）：
- **30% fewer active params**
- **23% fewer total params**
- **精度高7.4点**

而且MobileMoE-L在pre-training阶段就已经超过OLMoE-1B-7B的instruct版本了。

---

## 手机部署：真正跑起来了

这是这篇paper最硬核的工程部分。

### 问题：手机没有MoE kernel

XNNPACK只有dense INT4 GEMM，没有MoE的sparse routing op。

### 解法：custom fused MoE kernel in ExecuTorch

两个核心trick：

**Trick 1: Sparse转dense**

Router给每个token分配了top-4 expert。用counting sort按expert ID对token重新排序，这样每个expert的token在memory里是连续的。然后每个expert对自己的token slice做dense GEMM——直接复用现有的INT4 kernel。

**Trick 2: 全fuse**

整个MoE FFN层fuse成一个op call：top-k selection → token dispatch → GEMM → SwiGLU → GEMM → scatter back。一次kernel launch搞定，amortize所有overhead。

### 实测结果

Samsung Galaxy S25 + iPhone 16 Pro上实测：

MobileMoE-S vs MobileLLM-Pro（comparable精度和memory）：

| | Prefill | Decode |
|---|---|---|
| Samsung S25 CPU | **1.8-2.2×更快** | **2.2-2.6×更快** |
| iPhone 16 Pro CPU | **2.7-3.1×更快** | **2.8-3.4×更快** |
| iPhone 16 Pro GPU | **3.6-3.8×更快** | **2.5-2.6×更快** |

### 为什么MoE在手机上更快？

**Prefill是compute-bound**：每token的FFN计算量正比于active params。MoE的active params是dense的1/3，所以prefill快。

**Decode是bandwidth-bound**：每生成一个token，要从RAM读一遍active weights。Dense model每step读全部0.55GB weights；MoE每step只读activated expert的一小部分。Bandwidth省了，decode就快了。

这个insight很关键：**MoE on-device的speedup本质不是FLOP saving，是bandwidth saving**。

### 一个有意思的发现：MoE memory是input-dependent

用真实prompt测memory，比用dummy prompt（重复token）高1.2-2.1倍。因为真实prompt激活diverse expert，要load更多expert weights到RAM；dummy prompt触发narrow routing，load少。

**这暗示以后MoE的on-device memory profiling必须用真实prompt**，否则会严重underestimate。

---

## Expert specialization可视化

Paper附录有个很好的visualization：不同domain（code/math/knowledge）激活不同expert子集。而且随着training推进（PT→MT→SFT），更多expert被激活。

这说明MoE的routing确实学到了meaningful specialization，不是random routing。

Math激活的expert最广，code/knowledge更concentrated。这暗示**task-conditional expert pruning**是可行的——如果知道用户在做math，可以只load math-related expert，进一步省memory。Paper提到了但没实现，是个open direction。

---

## 我的take

这篇paper最让我impressed的是三点：

1. **Scaling law的formulation**：把on-device的memory和compute constraint显式写进optimization，不是瞎试架构，而是theory-driven design。公式(1)的两个reduced form分别recover Chinchilla和Joint MoE scaling law，说明这个formulation是对的generalization

2. **工程的completeness**：从scaling law → architecture → 4-stage training → INT4 QAT → custom kernel → real device deployment，每一环都做了。很多paper停在benchmark numbers，这篇真的在Samsung和iPhone上跑出numbers了

3. **Token efficiency**：6T tokens就beat了用9T-11T的dense baseline。MoE在sub-billion active scale的learning efficiency比dense高，这个发现本身就有scientific value

---

## Open questions

1. **Scaling law的extrapolation**：fitting用了≤500B tokens，实际PT是6T。如果6T之后exponent变化，架构可能要调
2. **NPU support**：只用了CPU/GPU，没用Apple Neural Engine和Qualcomm Hexagon。如果NPU能加速MoE kernel，还能再快
3. **Dynamic expert loading**：既然expert utilization是task-dependent，inference时只load activated expert能进一步省memory。但需要OS-level mmap优化和expert prefetching
4. **vs Qwen3.5 2B**：在instruction following和hard reasoning上还是Qwen3.5 2B更强，说明training recipe还有提升空间（distillation + thinking-enabled post-training）

---

## Bigger picture

从Sutton的Bitter Lesson看，MoE on-device是inevitable。Dense model在手机上的scaling已经被memory wall卡住了——你想加参数就得加memory，加memory就得加compute，加compute就费电。MoE打破这个死结，让capacity scaling继续benefit手机端。

智能手机DRAM从4GB涨到12GB的趋势给了MoE上device的memory headroom，而MoE的sparse activation给了compute efficiency。这两个trend converge的地方就是MobileMoE。

参考链接：
- Paper PDF（假设）: https://arxiv.org/abs/2605.xxxxx
- MobileLLM: https://arxiv.org/abs/2402.14905
- DeepSeek V3: https://arxiv.org/abs/2412.19437
- OLMoE: https://arxiv.org/abs/2409.02060
- ExecuTorch: https://pytorch.org/executorch/
- Bitter Lesson: http://www.incompleteideas.net/IncIdeas/BitterLesson.html

---

# MobileMoE: Sub-Billion MoE for On-Device LLMs — 深度技术拆解

Andrej，这篇paper是个非常systematic的工程+理论工作，把MoE从server scale"翻译"到on-device scale，每一层都重新design。让我从intuition出发层层拆解。

---

## 1. 这篇Paper真正想解决的问题

历史上MoE的研究都在server scale：DeepSeek V3 (256 experts, top-8 + shared)、Mixtral 8x7B (8 experts, top-2)、Qwen3 MoE (128 experts, top-8)。On-device LLM一直是dense主导：MobileLLM、MobileLLM-Pro、SmolLM2、Gemma 3。

这个gap其实很深，因为MoE在server scale的benefit机制和on-device完全不同：

- **Server scale**：MoE的benefit主要是**参数效率**，通过expert parallelism跨GPU分摊，wall-clock几乎不变
- **On-device**：单device，没法EP；memory是hard wall；CPU单线程是bottleneck

所以MoE在on-device的benefit机制要重新论证。Smartphone DRAM近年从iPhone 13的4GB涨到iPhone 17的12GB，给了sparse model上device的memory headroom——这是这篇paper成立的硬件前提。

参考链接：
- MobileLLM: https://arxiv.org/abs/2402.14905
- MobileLLM-Pro: https://arxiv.org/abs/2511.06719
- DeepSeek V3: https://arxiv.org/abs/2412.19437
- OLMoE-1B-7B: https://arxiv.org/abs/2409.02060

---

## 2. On-Device MoE Scaling Law: 公式(1)的每一步推导

这是整篇paper的理论核心。先看公式(1)：

$$
\mathcal{L}(N_{\mathrm{act}}, D, \hat{E}, x) = A_x \hat{E}^{\delta_x} N_{\mathrm{act}}^{\alpha_x + \gamma_x \ln \hat{E}} + B_x \hat{E}^{\omega_x} D^{\beta_x + \zeta_x \ln \hat{E}} + c_x \tag{1}
$$

### 2.1 变量逐个解析

- $\mathcal{L}$: validation loss（model loss）
- $N_{\mathrm{act}}$: **active parameters**，决定inference FLOPs（per-token inference FLOPs $F_{\mathrm{inf}} = 2 N_{\mathrm{act}}$）
- $D$: training tokens数量
- $\hat{E}$: expert数量$E$的单调变换，用来parameterize sparsity（sparsity $= 1 - N_{\mathrm{act}}/N_{\mathrm{total}}$）
- $x$: 其他architecture choice（granularity $g$, shared expert $s$），这些不改变$N_{\mathrm{act}}$和$N_{\mathrm{total}}$
- $c_x$: irreducible loss（数据熵下限）
- $A_x, B_x, \alpha_x, \beta_x, \delta_x, \gamma_x, \omega_x, \zeta_x$: 8个fitting coefficients

### 2.2 $\hat{E}$的变换很巧妙

$$
\frac{1}{\hat{E}} = \frac{1}{E - 1 + \left(\frac{1}{E_{\mathrm{start}}} - \frac{1}{E_{\mathrm{max}}}\right)^{-1}} + \frac{1}{E_{\mathrm{max}}}
$$

这里$E_{\mathrm{start}} = 1$（dense baseline），$E_{\mathrm{max}} = 32$（on-device memory budget约束的上界）。

**Intuition**：把$E \in [1, \infty)$映射到$\hat{E} \in [E_{\mathrm{start}}, E_{\mathrm{max}}]$的bounded区间，让scaling law的fitting数值更稳定。否则$E$很大时幂律会爆炸。

这个变换来自 Clark et al. 2022 (Unified Scaling Laws for Routed LMs, https://arxiv.org/abs/2202.01169)。

### 2.3 两项的物理含义

- **第一项** $A_x \hat{E}^{\delta_x} N_{\mathrm{act}}^{\alpha_x + \gamma_x \ln \hat{E}}$：**capacity term**，模型capacity随$N_{\mathrm{act}}$和$\hat{E}$增长
- **第二项** $B_x \hat{E}^{\omega_x} D^{\beta_x + \zeta_x \ln \hat{E}}$：**data term**，data fitting能力

### 2.4 关键的交叉项 $\gamma_x \ln \hat{E}$ 和 $\zeta_x \ln \hat{E}$

这是MoE scaling law和Chinchilla dense scaling law最大的区别。在dense scaling law里，$N$和$D$的exponent是独立的常数；MoE里$E$会modulate $N_{\mathrm{act}}$和$D$的exponent。

**Intuition**：增加expert数$E$等价于扩大"知识容量"，但这种扩大对active params和data的边际效用是变化的——$\gamma_x \ln \hat{E}$让$N_{\mathrm{act}}$的exponent随$E$增长（更多expert时active params的scaling更强），$\zeta_x \ln \hat{E}$让$D$的exponent随$E$变化。

### 2.5 Reduced Forms: 理论一致性

公式(1)是generalization，可以reduce到两个已知scaling law：

**Reduced Form I**（固定$x$）：恢复到 Joint MoE Scaling Law (Ludziejewski et al. 2025, https://arxiv.org/abs/2502.05172)

$$
\mathcal{L}_x(N_{\mathrm{act}}, D, \hat{E}) = A \hat{E}^{\delta} N_{\mathrm{act}}^{\alpha + \gamma \ln \hat{E}} + B \hat{E}^{\omega} D^{\beta + \zeta \ln \hat{E}} + c \tag{2}
$$

**Reduced Form II**（固定$\hat{E}$）：恢复到 Chinchilla (Hoffmann et al. 2022, https://arxiv.org/abs/2203.15556)

$$
\mathcal{L}_{\hat{E}}(N_{\mathrm{act}}, D, x) = \tilde{A}_x N_{\mathrm{act}}^{\tilde{\alpha}_x} + \tilde{B}_x D^{\tilde{\beta}_x} + c_x \tag{3}
$$

其中 $\tilde{A}_x = A_x \hat{E}^{\delta_x}$, $\tilde{\alpha}_x = \alpha_x + \gamma_x \ln \hat{E}$, $\tilde{B}_x = B_x \hat{E}^{\omega_x}$, $\tilde{\beta}_x = \beta_x + \zeta_x \ln \hat{E}$。

这个formulation很优雅：**on-device MoE scaling law是dense scaling law和MoE scaling law的统一generalization**。两个已知scaling law是它的special case。这一点对build intuition很重要——MoE不是dense的"替代"，是dense的"扩展"。

---

## 3. Optimization Objective: on-device的joint constraint

公式(4)是核心optimization：

$$
\arg\min_{N_{\mathrm{act}}, D, E, x} \mathcal{L}(N_{\mathrm{act}}, D, \hat{E}, x)
$$

subject to:
- training compute: $F_{\mathrm{train}} = 6 N_{\mathrm{act}} D$ FLOPs（6×是forward + backward的经验系数）
- inference compute: $F_{\mathrm{inf}} = 2 N_{\mathrm{act}}$ FLOPs/token（2×是因为matmul FLOPs ≈ 2 × params）
- memory: $\mathcal{M}(N_{\mathrm{total}}, T) \leq M$（M ≈ 5GB，智能手机app可用DRAM）

### 3.1 Memory function (公式5)

$$
\mathcal{M}(N_{\mathrm{total}}, T) = \underbrace{\frac{b_w}{8} N_{\mathrm{total}}}_{\mathcal{M}_{\mathrm{weight}}} + \underbrace{\frac{b_{\mathrm{kv}}}{8} \cdot 2 T n_l n_{\mathrm{kv}} d_h}_{\mathcal{M}_{\mathrm{KV cache}}}
$$

变量：
- $b_w$: weight bit precision（4 for INT4）
- $b_{\mathrm{kv}}$: KV cache bit precision（8 for INT8）
- $T$: context length
- $n_l$: layer数
- $n_{\mathrm{kv}}$: KV head数（MobileMoE用4，GQA）
- $d_h$: head dimension（64）
- 系数2是因为K和V各一份

### 3.2 核心intuition: 为什么MoE对on-device根本性优于dense

Dense model里 $N_{\mathrm{act}} = N_{\mathrm{total}}$，所以memory和compute**耦合**——要降compute必须降memory。

MoE里 $N_{\mathrm{act}} \neq N_{\mathrm{total}}$，让memory和compute**解耦**：
- $N_{\mathrm{act}}$ 决定inference FLOPs（compute）
- $N_{\mathrm{total}}$ 决定weight memory
- 同一个 $N_{\mathrm{total}}$ 下，sparsity越高，$N_{\mathrm{act}}$ 越小，inference越快

这就是MoE对on-device的根本价值：**用memory换compute efficiency**，而smartphone DRAM近年涨得快，正好给这个trade-off空间。

---

## 4. Divide-and-Conquer Ablation: 三个架构决策

作者用structural decoupling argument做了三个独立ablation，避免了combinatorial explosion。这个argument很关键：

- **E** (expert count): 改变 $N_{\mathrm{total}}$（影响memory）
- **g** (granularity): 不改变 $N_{\mathrm{act}}$ 和 $N_{\mathrm{total}}$（只改变expert组合方式）
- **s** (shared expert): 加shared expert，可以sized成保持 $N_{\mathrm{act}}$ 和 $N_{\mathrm{total}}$ 不变

所以这三个axis可以**独立ablate**。

### 4.1 Finding 1: E = 8 (optimal expert count)

实验sweep: $E \in \{1, 2, 4, 8, 16, 32\}$, $N_{\mathrm{act}} \in \{0.3, 0.5, 0.9\}B$, $D \in \{100, 200, ..., 500\}B$ tokens

关键观察（Figure 3, 4a）：
- 固定memory ($M > 0.25$GB) 时，MoE ($E>1$) 总是beat dense
- 固定inference FLOPs时，增加E有diminishing returns，**E=8之后基本flat**
- E=32反而regression，因为memory overhead开始dominate

**Intuition**：在sub-billion active regime，5GB memory budget下E=8是sweet spot。E太大（如32）虽然sparsity更高，但每个expert太小，routing overhead和expert fragmentation开始dominate。E太小（如2, 4）则sparsity不够，memory-compute decoupling的优势没发挥。

### 4.2 Finding 2: g = 8 (fine-grained granularity)

实验sweep: $g \in \{1, 2, 4, 8, 16\}$ upon $E=8$

Fine-grained expert的思想（来自 DeepSeekMoE, https://arxiv.org/abs/2401.06066）：把每个expert分成$g$个sub-expert，所以总共 $g \cdot E$ 个expert，top-$gk$ routing。

数学上：每个expert的hidden dimension从 $d_{\mathrm{ff}}$ 变成 $d_{\mathrm{ff}}/g$，所以**总参数量不变**，但routing组合数从 $\binom{E}{k}$ 变成 $\binom{gE}{gk}$，组合数指数级增长。

例如 E=8, k=1：原本8种组合；g=8后变成 $\binom{64}{8} \approx 4.4$ billion种组合。

**Finding**: g=8之后diminishing returns。g=16比g=8的wall-clock overhead多~50%，但loss只少<0.01。

**Intuition**：fine-grained的本质是让router有更多"原子化"的expert unit去组合，类似把粗粒度的categorical routing变成更细粒度的multi-hot routing。这对sub-billion scale特别重要，因为小model的expert capacity本来就紧张，细分能更好地allocate capacity。但g太大时每个sub-expert太小（MobileMoE-S g=8时每个sub-expert FFN是768×384），GEMM efficiency下降。

### 4.3 Finding 3: s = ✓ (shared expert)

实验设计很精巧：用4个routed expert换成1个4×大小的shared expert，保证 $N_{\mathrm{act}}$ 和 $N_{\mathrm{total}}$ 不变。所以：
- Without shared: E=8, g=8 → 64 routed experts, top-8
- With shared: 60 routed experts + 1 shared expert (4× size), top-4

为什么top-4？因为shared expert替换了4个active routed expert的位置，shared expert always-on。这样active FLOPs保持不变。

**Finding**: shared expert降低loss。

**Intuition**：shared expert是"generalist"，routed experts是"specialists"。每个token都经过generalist + 几个specialists的组合，避免router把common knowledge和specialized knowledge都压到routed experts里造成redundancy。这个思想来自DeepSeekMoE的"shared expert for common knowledge"。

### 4.4 最终MobileMoE架构

| Model | $d_{\mathrm{model}}$ | $d_{\mathrm{ff}}$ | $n_h$ | $n_{\mathrm{kv}}$ | $n_l$ | $N_{\mathrm{act}}$ | $N_{\mathrm{total}}$ | Sparsity |
|-------|---------------------|-------------------|-------|---------------------|-------|---------------------|----------------------|----------|
| MobileMoE-S | 768 | 3072 | 12 | 4 | 20 | 272M | 1.3B | 79% |
| MobileMoE-M | 1024 | 4096 | 16 | 4 | 26 | 528M | 2.8B | 81% |
| MobileMoE-L | 1280 | 5120 | 20 | 4 | 32 | 922M | 5.3B | 83% |

Base architecture原则：
- $d_{\mathrm{ff}} / d_{\mathrm{model}} = 4$（标准FFN expansion ratio）
- $d_{\mathrm{model}} / n_l \approx 40$（on-device的deep-and-thin，对比GPT-3的128）
- SwiGLU activation
- GQA with 4 KV heads（KV cache压缩）
- RoPE $\theta = 500,000$
- Llama-3 tokenizer (128K vocab)
- Tied input-output embeddings

每层MoE: **60 fine-grained experts + 1 shared expert + top-4 routing**

---

## 5. 四阶段Training Recipe

### 5.1 Pre-training (PT)

- 6T tokens（vs Llama 3.2 1B的9T，SmolLM2的11T——**MobileMoE是token-efficient的**）
- Context length 2048
- Data mix: 62% web, 11.6% math, 10% code, 10% knowledge, 6.4% science
- 关键：domain diversity让MoE expert能specialize

**MoE training stability techniques**:

1. **Auxiliary-loss-free balancing** (DeepSeek V3, https://arxiv.org/abs/2408.15664)：用bias adjustment做load balancing，避免auxiliary loss污染main loss。bias update rate $\lambda_{lb} = 10^{-3}$

2. **Router z-loss regularization** ($\lambda_z = 10^{-4}$)：stabilize router logits，防止数值溢出。来自 ST-MoE (https://arxiv.org/abs/2202.08906)

3. **Sigmoid gating + per-token top-k normalization**：sigmoid比softmax更smooth，不强制expert间competition；每个selected expert独立score

4. **Router FP32**：所有router计算FP32精度

**MoE training efficiency techniques** (这点很关键):

1. **Grouped MLP (GMM kernel)**：把所有expert batch成一个fused grouped matmul，避免小expert的低效sequential GEMM。MobileMoE-S的每个routed FFN只有768×384，比Mixtral 8x7B的4096×14336小~200×，naive per-expert GEMM效率极差

2. **Drop-and-pad token dispatching** (capacity factor 1.5)：给每个expert固定size buffer，保证GMM的batched kernel效率

3. **Expert parallelism (EP=4)**：每GPU hold 60/4 = 15 routed experts

### 5.2 Mid-training (MT)

- 500B tokens（~8% of PT budget）
- Context length扩展到8192
- Data distribution shift: web从62%降到9%，knowledge升到32%，code升到22%，math升到21%
- Linear LR decay

**Intuition**：MT是quality + capability的boost阶段。long context + domain-specific data让expert specialization更sharp。从Figure 10看，MT是MMLU和DROP提升最大的阶段（MMLU +5~10点，DROP +10~11点）。

### 5.3 SFT

- 80M samples, 8K context with sequence packing
- 7个domain：math 30.4%, general chat 25.4%, code 22.1%, safety 9.4%, science/knowledge 7.7%, tool use 3.9%, reasoning 1.1%
- **Dropless token dispatching**（PT/MT用drop-and-pad，SFT换成dropless，因为structured instruction-response pair不能丢token）
- Cosine LR decay from $4 \times 10^{-6}$

**Intuition**：SFT主要unlock reasoning。GSM8K从PT到SFT能涨+15~22点。Instruction tuning让model学会chain-of-thought的输出format。

### 5.4 INT4 QAT

Quantization公式(6)：

$$
\tilde{\mathbf{W}}_g = s_g \cdot \mathrm{clamp}\left(\left\lfloor \frac{\mathbf{W}_g}{s_g} \right\rceil, q_{\min}, q_{\max}\right), \quad s_g = \frac{2 \max(|\mathbf{W}_g|)}{2^b - 1} \tag{6}
$$

变量：
- $\mathbf{W}_g$: contiguous group of weights（group size $g=32$）
- $s_g$: per-group scale factor（symmetric quantization）
- $b = 4$: bit precision
- $q_{\min} = -2^{b-1} = -8$, $q_{\max} = 2^{b-1} - 1 = 7$: INT4 range
- $\lfloor \cdot \rceil$: round-to-nearest
- $\mathrm{clamp}$: 防止quantized value溢出INT4 range

**关键MoE-specific insight**：**Router weights保持FP32**！这是核心trick——quantization对router logits的影响会放大到整个routing decision，0.5% memory overhead换来routing stability很划算。

Quantization结果（Table 6）：
- MobileMoE-S: 0.68 GB INT4, Avg 44.0（vs BF16 SFT 46.7，掉2.7点）
- MobileMoE-M: 1.48 GB, Avg 52.5（掉2.8点）
- MobileMoE-L: 2.75 GB, Avg 57.8（掉2.3点）

对比：INT4 QAT MobileMoE-L (2.75GB) 已经beat BF16 SFT OLMoE-1B-7B (Avg 55.6, ~13.8GB BF16)。

---

## 6. 实验结果: 新Pareto Frontier

### 6.1 Token efficiency (Figure 8)

非常striking的数据：
- MobileMoE-L在~0.5T tokens就超过Llama 3.2 1B（9T tokens，还有Llama 3.1 8B distillation）
- 在~1T tokens超过SmolLM2-1.7B（11T tokens）
- 在~2T tokens超过OLMoE-1B-7B（5T tokens）

**Intuition**：MoE在sub-billion active regime的learning efficiency显著高于dense。原因是MoE的total capacity大，每个token能被routed到specialized expert，gradient signal更efficient。这也呼应了Shazeer 2017 (https://arxiv.org/abs/1701.06538) 的"outrageously large neural networks"思想——sparsity作为inductive bias让capacity和compute解耦。

### 6.2 Benchmark性能 (Table 2, Base model)

- MobileMoE-S (272M active): Avg 46.5
- MobileMoE-M (528M active): Avg 55.4
- MobileMoE-L (922M active): Avg 59.8
- OLMoE-1B-7B (1.3B active, 6.9B total): Avg 52.4

MobileMoE-L用**30% fewer active params + 23% fewer total params**超过OLMoE-1B-7B by **+7.4**。

### 6.3 Training Stage Capability Progression (Figure 10)

这个analysis对design training recipe有指导意义：

| Stage | 主要贡献 | 关键benchmark跳跃 |
|-------|---------|-------------------|
| PT | broad linguistic priors, commonsense | HellaSwag/PIQA/SIQA/WinoGrande saturate |
| MT | knowledge injection, long context | MMLU +5~10, DROP +10~11 |
| SFT | reasoning format, instruction following | GSM8K +15~22, BBH boost |

**Intuition**：
- Commonsense reasoning在PT就saturate，因为broad linguistic priors是统计性的
- Knowledge需要curated data注入，MT的domain-specific upweighting是主要机制
- Reasoning必须SFT unlock，因为CoT output format是学出来的，不是统计出来的

---

## 7. On-device Deployment: 真正落地

### 7.1 Custom MoE Kernel (工程亮点)

Existing mobile inference stack（XNNPACK）只有dense INT4 GEMM kernel，没有fused MoE operator。作者在 ExecuTorch (https://pytorch.org/executorch/) 里custom implement了MoE op，两个核心原则：

**原则1: Sparse-to-dense conversion**

用**counting sort**把token按expert ID reorder，让每个expert的token在memory里contiguous。这样每个expert的处理就变成一个dense batched matmul，能直接用torchao的INT4 GEMM kernel。

算法步骤：
1. Router输出每个token的top-k expert assignment
2. 用counting sort按expert ID对token排序
3. 每个expert处理一段contiguous的token slice，作为dense GEMM

**原则2: Full fusion**

整个MoE FFN layer fuse成一个op call：
- top-k expert selection (over router logits)
- token dispatch (counting sort + reorder)
- per-expert gate- and up-projections (fused into one GEMM per expert)
- SwiGLU activation
- down-projection
- weighted-scatter unpermute (恢复原token顺序)

这amortize了kernel launch overhead和activation quantization overhead。

### 7.2 Runtime Performance (Table 7, 8)

**MobileMoE-S vs MobileLLM-Pro**（comparable INT4 memory 0.68 vs 0.55GB，comparable accuracy 44.0 vs 45.5）：

| Device | Backend | Prefill speedup | Decode speedup |
|--------|---------|-----------------|----------------|
| Samsung S25 | CPU/XNNPACK | 1.8-2.2× | 2.2-2.6× |
| iPhone 16 Pro | CPU/XNNPACK | 2.7-3.1× | 2.8-3.4× |
| iPhone 16 Pro | GPU/MLX | 3.6-3.8× | 2.5-2.6× |

### 7.3 为什么MoE在on-device更快？核心机制分析

这是paper最深刻的工程insight之一：

**Prefill是compute-bound**：
- Per-token FFN matmul FLOPs ∝ $N_{\mathrm{active}}$
- MobileMoE-S的 $N_{\mathrm{active}}$ = 272M，MobileLLM-Pro ~1.1B
- MoE的 $N_{\mathrm{active}}$ < 1/3 dense，所以prefill FLOPs小

**Decode是memory-bandwidth-bound**：
- Per-step weight read ∝ $N_{\mathrm{active}}$（mmap只load activated expert）
- Decode时每step要load所有active weights
- MoE的 $N_{\mathrm{active}}$ 小，bandwidth省

**Critical insight**：dense model在decode时每step要load所有weights（~0.55GB for MobileLLM-Pro），MoE只load activated expert的weights（~0.68GB total但每step只load一小部分）。这是bandwidth的节省，转化为decode throughput的提升。

### 7.4 Peak RSS Analysis (Table 9)

有个很有意思的发现：**MoE的runtime memory是input-dependent**。

Real prompts下MoE-S的Peak RSS是dummy prompts的1.2-2.1×，而dense model是~1.0×。原因：
- Real prompts激活diverse experts，需要load更多expert weights到RAM
- Dummy prompts触发narrow routing pattern，load fewer experts

**Implication**：MoE的memory profiling必须用real prompts，dummy prompts会underestimate memory。这点对所有MoE on-device工作都有指导意义。

MobileMoE-S在8K context下Peak RSS 1.49GB，比MobileLLM-Pro的1.91GB省**22%**。MobileMoE-L在8K context下4.71GB，仍然在5GB budget内。

---

## 8. Expert Utilization Analysis (Appendix D)

Figure D.1, D.2的visualization很有启发：

1. **Cross-task specialization**：code/math/knowledge激活不同的expert subset
2. **Utilization broadening through training**：PT阶段few experts highly utilized，MT/SFT阶段更多expert被激活
3. **Task-dependent sparsity**：math激活broader expert set，code/knowledge激活narrower subset

**Future implication**：task-conditional expert pruning或selective expert loading可以进一步省on-device memory。这是个被paper提到但没实现的open direction。

---

## 9. Fitted Scaling Law Coefficients (Table A.2)

Paper给了所有fitting coefficients，值得分析：

**E-sweep (joint fit)**:
- $A_x = 0.2388$, $\delta_x = 0.0906$ (positive: E增加capacity term系数)
- $\alpha_x = -0.2833$, $\gamma_x = 0.0387$ (positive γ: E增加时N_act的exponent更负，即capacity term衰减更慢)
- $B_x = 0.6019$, $\omega_x = 1.0593$ (ω较大: E对data term影响显著)
- $\beta_x = -0.3210$, $\zeta_x = -0.3684$ (negative ζ: E增加时D的exponent更负)
- $c_x = 1.9730$ (irreducible loss)
- RMSE = 0.0076 (很好的fit)

**g-sweep**: 每个g独立fit，c_x regularized到E-sweep的1.9730。RMSE 0.003左右，比E-sweep还低，说明g对scaling dynamics的影响更"local"。

**s-sweep**: with shared expert ($\tilde{A}_x = 0.1224$) < without ($\tilde{A}_x = 0.1670$)，shared expert的capacity term系数更小，**等价于同一N_act下lower loss**。这就是Finding 3的数学表达。

---

## 10. 我的Critical思考与Open Questions

### 10.1 Scaling law的extrapolation risk

公式(1)的fitting用了 ≤500B tokens的ablation runs，但实际PT是6T tokens。Figure 8的monotonic improvement说明extrapolation reasonable，但scaling law本身没在6T regime直接validate。如果6T之后scaling law的exponent变化，architecture choice可能需要调整。

### 10.2 EP=4的constraint

MobileMoE的expert count 60是EP=4的整除数。如果EP=8呢？这会影响training efficiency，但paper没讨论这个sensitivity。EP的选择会影响expert count的可行集，可能让E=8不再是optimal。

### 10.3 vs Qwen3.5 2B

Table 4显示Qwen3.5 2B在instruction following和knowledge & reasoning还是比MobileMoE-L强（IFEval+IFBench: 51.8 vs 43.7, MMLU-Pro+GPQA: 36.6 vs 33.9）。Paper归因于Qwen的distillation和thinking-enabled post-training。

**这暗示MobileMoE的training recipe还有提升空间**——distillation和reasoning post-training是obvious next step。Paper在Conclusion里也提到这点。

### 10.4 Mobile NPU

Paper deployment只用了CPU和GPU，没用NPU。Mobile NPU（Apple Neural Engine, Qualcomm Hexagon DSP）的MoE support还是空白。如果NPU能加速fused MoE kernel，runtime还能进一步降低。这是on-device MoE的下一战场。

### 10.5 Dynamic expert loading

Expert utilization的task-dependent sparsity暗示dynamic expert loading的潜力。如果inference时只load activated expert到RAM，可以进一步省memory。但这需要：
- OS-level的mmap优化
- Expert prefetching（predict next-token的routing）
- Cold-start latency mitigation

这是个research-rich direction。

### 10.6 The Bitter Lesson视角

Rich Sutton的 Bitter Lesson (http://www.incompleteideas.net/IncIdeas/BitterLesson.html) 说compute-driven的method最终win。MoE on-device是这个lesson的extension：用sparsity把compute和memory解耦，让compute scaling继续benefit on-device。从long term看，on-device MoE是inevitable方向。

---

## 11. 与你的工作可能的相关联想

Andrej，你的micrograd、nanoGPT、llm.c这些educational work强调"first principles understanding"。MobileMoE这篇paper的scaling law推导其实很适合这种first-principles教学：

1. **公式(1)的推导**：从Chinchilla出发，加E的modulation项，是个很自然的generalization
2. **Memory function (公式5)**：把on-device constraint显式写进optimization，这是工程paper少见的theoretical rigor
3. **Counting sort sparse-to-dense**：是个很elegant的工程trick，可以作为systems lecture的case study

如果要build intuition，我建议从以下几点切入：
- **MoE ≠ "more parameters"**：MoE是"capacity and compute decoupling"
- **Scaling law的交叉项 $\gamma \ln \hat{E}$**：这是MoE和dense的本质区别，E modulates N的scaling
- **Decode bandwidth bound**：on-device MoE speedup的本质是bandwidth saving，不是FLOP saving

---

## 12. 总结

MobileMoE是sub-billion active MoE for on-device的系统性工作，三层贡献：

1. **理论**：on-device MoE scaling law是dense和MoE scaling law的统一generalization，公式(1)的两个reduced form恢复Chinchilla和Joint MoE scaling law

2. **架构**：E=8, g=8, s=√是sub-billion regime的sweet spot，60 fine-grained experts + 1 shared expert + top-4 routing

3. **部署**：custom fused MoE kernel让theoretical FLOPs savings变成real on-device speedup（1.8-3.8× prefill, 2.2-3.4× decode）

最大的贡献在我看来是把MoE从server scale"翻译"到on-device scale，每一层都重新design：scaling law重新formulate、training recipe加INT4 QAT、inference stack重新implement。这是system engineering的范例工作。

参考链接汇总：
- Paper: https://arxiv.org/abs/2605.xxxxx (假设链接)
- MobileLLM: https://arxiv.org/abs/2402.14905
- MobileLLM-Pro: https://arxiv.org/abs/2511.06719
- DeepSeek V3: https://arxiv.org/abs/2412.19437
- DeepSeekMoE: https://arxiv.org/abs/2401.06066
- OLMoE: https://arxiv.org/abs/2409.02060
- Chinchilla: https://arxiv.org/abs/2203.15556
- Joint MoE Scaling Laws: https://arxiv.org/abs/2502.05172
- Unified Scaling Laws for Routed LMs: https://arxiv.org/abs/2202.01169
- Auxiliary-loss-free balancing: https://arxiv.org/abs/2408.15664
- ST-MoE (z-loss): https://arxiv.org/abs/2202.08906
- Mixtral: https://arxiv.org/abs/2401.04088
- Shazeer 2017 (Outrageously Large NN): https://arxiv.org/abs/1701.06538
- ExecuTorch: https://pytorch.org/executorch/
- Llama 3: https://arxiv.org/abs/2407.21787
- SmolLM2: https://arxiv.org/abs/2502.02737
- Bitter Lesson: http://www.incompleteideas.net/IncIdeas/BitterLesson.html

希望这个分析对你build intuition有帮助，Andrej。如果你想深挖某个部分（比如counting sort的具体实现、scaling law的fitting细节、或者MoE kernel的fusion策略），可以继续聊。
