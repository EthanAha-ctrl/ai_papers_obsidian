---
source_pdf: MAI-Thinking-1 Building a Hill-Climbing Machine.pdf
paper_sha256: 7d4f13dd88ff98d0480645e96b751f0abea00fd99122cabf7626598b29bef26f
processed_at: '2026-08-05T16:09:26-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 MAI-Thinking-1

## 一句话总结

Microsoft 不是在造一个 model，是在造一台"造 model 的机器"。这台机器能持续往上爬，每次迭代都比上一次好一点。MAI-Thinking-1 只是这台机器吐出来的第一个产品。

---

## 为什么要造机器而不是造 model

单个 model 是一锤子买卖——你花几个月训出来，发出去，然后竞争对手又超过了你。但如果你有一台机器，每次跑一遍就能产出更好的 model，你就赢在长期。

打个比方：与其花一年手工雕一把椅子，不如造一个数控机床，以后按个按钮就能出椅子，而且越按越精。

他们的三个原则其实都很朴素：

- **不抄作业**（不用别人 model 的 distillation data）——抄来的能力不扎实，遇到新情况就崩
- **保持简单**——复杂的 recipe 难维护、难复现
- **用数据说话**——每个决策都要能 ablation 验证

---

## 模型架构：dense 和 MoE 交替，不是每层都 MoE

这个选择很有意思。DeepSeek-V3 是每层都 MoE，Microsoft 反而选择**一层 dense、一层 MoE 交替**。

为什么不每层都 MoE？因为 wall-clock time。虽然纯 FLOPs 上 every-layer-MoE 略好，但实际跑起来，interleaved 方案更快——MoE 的 all-to-all 通信开销很贵，交替着用能省掉一半的通信。

还有一个细节：他们用 LatentMoE——在 dispatch 之前先把 token 压缩一半（6656 → 3072），然后再路由给 expert。这样 all-to-all 传的数据量小一半。每个 expert 内部再 expand 3×。相当于"先压缩再传输再展开"。

**Attention 的设计**：5 层 local（sliding window 512）配 1 层 global，循环往复。Local 层用 RoPE，global 层**完全不用位置编码**（NoPE）。这样 KV cache 大幅缩小，因为 local 层只需要存 512 个 token 的 KV。

---

## 一个很妙的初始化 trick

新模型刚初始化时，attention softmax 接近均匀分布，相当于对所有前面 token 做平均。这导致不同 token 的表示变得很像，下游 MoE 路由就严重不平衡——所有 token 都想找同一个 expert。

解决方案极其简单：把 attention 输出的 RMSNorm gain 初始化为 0。这样模型一开始就像只有 feedforward 在工作，attention 是"关着的"，随着训练慢慢"打开"。

这就像新员工第一天别让他做复杂决策，先让他做简单任务，慢慢加复杂度。

---

## Scaling Ladder：不要在单个点上做判断

这个方法论值得单独说。他们不比较两个架构在同一个规模下的表现，而是训练一整个阶梯（L12、L18、L24...L78），看 scaling 曲线。

为什么重要？因为很多 trick 在小模型上看起来好，到大模型就消失了。或者反过来，小模型上看起来没用，大模型上才显出来。他们确实遇到了这个坑：

他们做了两个 data mixture——code-heavy 和 STEM-heavy。小模型上 STEM-heavy 在 STEM eval 上赢，但放大到 23B 训练 20T tokens 后，两条曲线**在训练中途交叉了**，code-heavy 最终反超。

原因：STEM-heavy 里有两个数据源质量高但重复多、多样性低。小模型吃这套（重复帮助记忆），大模型把 unique content 消化完就饱和了，重复反而有害。

**教训：小模型的结论不能直接外推到大模型。**

---

## 数据混合：code 占了一半

最终 30T tokens 里：
- **Code 54.6%**（16.4T）
- STEM 15.8%
- Math 5.4%（但被采样了 5.28 次！unique 只有 0.3T）
- Web text 14.9%（unique 8.1T 但只看了一半）
- Multilingual 1.6%（有 8.1T 但只用了 0.5T）

这个比例非常偏 code 和 reasoning。数学数据虽然 unique tokens 少，但疯狂重复——因为好数学数据太稀缺了，只能多轮采样。

deduplication 做了五层：exact → fuzzy (MinHash) → templated → semantic (embedding)。最后还做了 cross-dataset dedup——同一个内容出现在多个数据集时，只在优先级最高的数据集保留。

---

## RL Climb：从零学 reasoning

这是最核心的部分。他们不从别的 reasoning model 蒸馏 CoT，而是让 mid-trained checkpoint 直接开始 RL，从零学怎么思考。

### 两个关键 stabilization trick

**第一个：adaptive entropy control**

标准 PPO/GRPO 的 clipping 是对称的（$[1-\epsilon, 1+\epsilon]$）。但 entropy 太低时你希望让 policy 更敢探索，entropy 太高时你希望收紧。

他们的做法：上界不是固定的 $1+\epsilon$，而是 $(1-\epsilon)^{-1} + k$，其中 $k$ 会根据当前 entropy 动态调整。entropy 太低就增大 $k$（放宽上界），entropy 够了就减小 $k$。

相当于一个自动调温器——太冷就开暖气，太热就关。

**第二个：outer ratio clip**

GRPO 有两个分支不 clip（policy "纠正方向"时）。实践中这会导致 gradient 偶尔爆炸。解决：加一个硬的外层 clip（ratio 不超过 50），把极端情况直接砍掉。

这两个 trick 加在一起，让 RL 能跑几千步不崩。

### Self-distillation：让长跑成为可能

RL 跑几千步必然会遇到崩溃（数值问题、基础设施故障）。如果每次崩溃都从头来，浪费太多。

他们的方案：把 RL 过程中生成的 rollouts 收集起来，对 mid-trained checkpoint 做 SFT，得到一个"已经学会一些 reasoning"的新起点，再继续 RL。

这就像游戏存档——死了不用从第一关重来，从上次存档点继续。

关键发现：
- **100 万条 trace 就够**了，再多收益递减还会过度约束 policy
- 用**多个 checkpoint 的 trace**（不只最后一步）效果最好——多样性比纯度重要
- SFT 时用高 dropout (0.15) 防止 model collapse

---

## 三个 Specialist 然后合并

他们并行训练三个专家：

1. **STEM specialist**：数学、物理、竞赛编程，用 SymPy 验证答案
2. **Agentic specialist**：SWE + 工具调用，在真实 Docker 容器里跑
3. **Helpfulness & Safety specialist**：人类偏好、指令跟随、安全

然后用 SFT 把三个蒸馏成一个 model，再跑一轮轻量 RL 收尾。

SFT 的数据比例有意思：按 sample 算 STEM 56%、agentic 11%、helpfulness 33%；但按 token 算 STEM 占 89%（因为 reasoning trace 长）。也就是说 helpfulness 的"体积"很小但 sample 数量不少，保证覆盖。

---

## SWE 环境构建：从 102M PRs 到 26 万训练环境

这个 pipeline 很震撼：
- 从 1.02 亿 GitHub PR 开始
- 筛到 487 万（merged、<15 files、有 code+test、有 linked issue）
- 自动生成 Docker 环境 → 208 万通过
- 提取 F2P (fail-to-pass) 测试作为 grading signal → 74 万
- 验证空 patch 失败、golden patch 通过 → 26 万
- 质量过滤 + 重写 → 最终训练集

**防作弊**做了三层：
- 断网（防止搜索 golden solution）
- 清理 git history（防止翻 commit 找答案）
- grading 前重置 test 文件（防止改测试）

---

## Long Context 的便宜方案

他们发现一个很省钱的事实：

64K mid-training 1T tokens + 短 150B tokens 的 256K extension ≈ 128K mid-training 1T tokens

也就是说，不需要全程在长 context 上训练。大部分时间在 64K 上训（MFU 高），最后短时间 extend 到 256K 就行。

更惊人的是：extension 的 90% 收益在前 1-10% iterations 就拿到了。这意味着 long context 不是"学新能力"，而是"校准位置编码到没见过的位置"。

---

## 基础设施：看不见的功夫

### Goodput 90%

在 8192 个 GPU 上跑 30T tokens，goodput 达到 90%——也就是说有效训练时间占 90%。剩下的 10% overhead 里：
- 6.5 小时 recompute（崩溃后重算）
- 14 小时 non-stepping（启动、调度等）
- 18 小时 MFU drop（最大瓶颈）

### Bitwise determinism

他们做到了**完全可复现**——同样的配置跑两次，得到 bitwise identical 的 model。为此：
- 禁用 NVLink SHARP（牺牲性能换确定性）
- 固定 NCCL topology
- 所有 reduction 用固定顺序
- top-k 用 stable sort

这不是为了好看，是为了 debugging——如果两次跑出来不一样，你就不知道是 bug 还是正常波动。

### RL 推理的 numerics gap

RL 里 learner（YOLO）和 inference engine（SGLang）用不同的 kernel 和并行策略。即使每个 token 的 logprob 差异很小，在 128K 长度的 rollout 里会 compound，最后 importance sampling ratio 爆掉。

他们的 mitigation stack：
- 双向用 bf16（不用更低精度）
- MoE routing replay（推理时的路由决策在训练时复用）
- Top-p mask replay（推理时排除的 token 在训练时也排除）

---

## CoT 的进化：从菜鸟到高手

附录 C 里的观察很生动：

**数学**：
- 弱模型猜答案（从可见的 roots 猜 minimizer），强模型推导+验证 domain
- 弱模型暴力枚举，强模型找 invariant（比如 mod 9 的结构）
- 弱模型不检查自己，强模型会"等等让我重新想想"

**编程**：
- 弱模型只做表面 sanity check，强模型真的跑 unit test
- 弱模型纠结于 edit 的精确格式，强模型先到处找 evidence 再动手
- 弱模型在相邻路径上试错，强模型追溯 data source 找 ground truth

这些观察本身就是很好的 reasoning model 诊断工具。

---

## 最后说一句

这篇 paper 的精神是 Karpathy 你会喜欢的——**把 AI 研究当工程来做**。不是靠灵感或运气，而是靠可复现的 pipeline、data-driven 的决策、infrastructure-level 的优化。

他们明确说了 MAI-Thinking-1 "不领先领域"，但他们的 hill-climbing machine 才是重点。第一个 model 只是证明这台机器能跑。后面会越跑越快。

---

# MAI-Thinking-1: Building a Hill-Climbing Machine 深度解析

## 核心隐喻与哲学：Hill-Climbing Machine

这篇 paper 的标题已经透露了核心野心——构建一个**hill-climbing machine**，而仅仅是一个 model。把整个 model development 视作一个 system-level optimization loop，其中 data pipeline、training infrastructure、RL recipe、evaluation suite、safety tests 都是这个 optimization loop 中的可调组件。这与 Karpathy 你常说的 "software 2.0/3.0" 以及 "model is a compiled artifact" 的视角非常契合——model 只是这个机器输出的 snapshot，真正有价值的是这台机器本身。

他们提出三个设计原则：

1. **Capabilities should be learned, not inherited**：完全不使用 third-party model 的 distillation data。这与你过去对 distillation 的批评一致——distilled intelligence 缺乏 steerability 和 robustness，本质是 imitation 而非 learning。
2. **Simplicity is sustainable**：simple, scalable recipes; clean data; transparent infrastructure。
3. **Scientific rigor avoids shortcuts**：每个决策都要通过 data-driven ladders、ablations、evaluations 验证。

参考链接：
- Karpathy 关于 software 2.0/3.0: https://karpathy.xyz/
- Hill climbing 在 optimization 中的经典讨论: https://en.wikipedia.org/wiki/Hill_climbing

---

## 1. Pre-training: MAI-Base-1 架构

### 1.1 模型规格

| 属性 | 数值 |
|---|---|
| Active parameters | 34.7B |
| Total parameters | 962B |
| Layers | 78 |
| Hidden dim | 6656 |
| FFN dim | 13312 |
| Down-proj dim (LatentMoE) | 3072 |
| Expert FFN | 10240 |
| Top-k / Experts | 8 / 512 |
| KV / Q heads | 8 / 80 |
| Tokenizer | o200k_base (vocab 200,019) |
| 上下文长度 | 256K |

### 1.2 架构关键创新

#### (a) Periodic Attention (5 local : 1 global)

借鉴 Gemma 3 的设计：每 6 层中，5 层用 local sliding-window attention（window size 512），1 层用 global attention。Local layers 使用 RoPE (base frequency 10,000)，global layers **不使用任何位置编码**（参考 NoPE 的工作，Kazemnejad et al., 2023）。这个组合大幅减少 KV cache 和 attention 计算成本。

公式层面，每层 attention 仍然遵循标准 scaled dot-product：

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} \cdot M\right) V
$$

其中 $M$ 是 mask 矩阵——对于 local 层，$M_{ij} = 0$ 当 $|i - j| > 512$ 或 $j > i$（causal），其余 $-\infty$；对于 global 层，仅有 causal mask。$d_k = 128$ 是 per-head dimension。

Group-query attention: 8 KV heads, 80 Q heads, 比率 10:1。

参考: 
- Gemma 3: https://arxiv.org/abs/2503.19786
- NoPE: https://arxiv.org/abs/2305.19466
- GQA: https://arxiv.org/abs/2305.13245

#### (b) Interleaved Dense + MoE FFN

这是一个非常重要的设计决策。他们**不**像 DeepSeek-V3 那样每一层都是 MoE，而是 dense FFN 与 high-sparsity MoE **交替**出现。第一个 FFN 是 dense（参考 DeepSeek-V3、Kimi K2、Dai et al. 2024 的做法）。

表 2 给出了对比：
- MoE every layer (8/384): EG_FLOPs = 0.94, EG_Time = 0.73
- MoE every layer (7+1 shared/384): EG_FLOPs = 1.03, EG_Time = 0.82
- Interleaved (本方案): baseline = 1.0

注意到虽然 MoE-every-layer + shared expert 在 FLOPs 上略好 (1.03)，但在 wall-clock time 上落后 (0.82)。这是 wall-clock efficiency 主导决策的典型案例。

#### (c) LatentMoE 设计

借鉴 NVIDIA Nemotron 3：在 all-to-all dispatch **之前**应用一个 shared down-projection $W_{\text{down}} \in \mathbb{R}^{D_{\text{latent}} \times D}$，把 token 表示从 $D = 6656$ 压缩到 $D_{\text{latent}} = 3072$，然后路由到 8/512 个 experts，每个 expert 内部 expand 3×（即 expert FFN = 10240），再投影回 $D$。

数学上：
$$
h = x W_{\text{down}} \in \mathbb{R}^{D_{\text{latent}}}
$$
$$
g = \text{softmax}(\text{Router}(x))_{\text{top-}k} \quad (\text{routing 用原始 } x, \text{不是 } h)
$$
$$
y = \sum_{i \in \text{top-}k} g_i \cdot \text{Expert}_i(h) \cdot W_{\text{up}}
$$

压缩比率 2×，expansion 比率 3×，相当于每个 expert 实际计算量是 $\frac{3}{2} D_{\text{latent}} = 4608$ 维的 FFN，相对原始 $D$ 已经显著压缩。这是为了 manage all-to-all communication cost。

#### (d) Dropless MoE with Global-batch Load Balancing

完全放弃 capacity-capped routing（GShard-style），使用 dropless 实现。load balancing loss 在 global batch 上聚合（跨 data parallelism workers 和 micro-batches）：

$$
\mathcal{L}_{\text{LB}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i
$$

其中 $f_i$ 是 expert $i$ 接收的 token 比例，$P_i$ 是 router 给 expert $i$ 的平均 gating probability。他们发现 aggregation strategy 比 loss type 更重要——只要 global aggregation 保证，GShard-style 和 loss-free (DeepSeek-V3) 表现相近。

参考:
- LatentMoE / Nemotron 3: https://arxiv.org/abs/2512.20856
- DeepSeek-V3 auxiliary-loss-free: https://arxiv.org/abs/2408.15664
- GShard: https://arxiv.org/abs/2006.16668

#### (e) Attention Output 初始化为零

这是一个有意思的 trick（图 8）。在初始化时，attention softmax 接近 uniform，相当于 causal mean pooling，导致 token 表示多样性下降，进而 MoE routing 严重不平衡。

解决方法：把 attention output RMSNorm 的 gain 初始化为 0。数学上：

$$
\text{AttnOut}_l = x_l + \gamma_l \cdot \text{RMSNorm}(\text{Attn}(\cdots))
$$

其中 $\gamma_l \leftarrow 0$ 在初始化时，让模型一开始就像一 stack of feedforward layers 应用到 individual tokens。cross-token interaction 通过训练逐渐 kick in。

这与 DeepSeek-V3 的 "zero-init residual" 思想一致，但应用到了 attention 而非 MoE 的 combine。

---

## 2. Scaling Ladder 与 Efficiency Gain (EG)

这是 paper 中最 methodologically 重要的部分之一。

### 2.1 Scaling Ladder

他们训练一系列递增规模的模型（L12, L18, L24, ..., L78），每个 ladder 由 layer count $L$ 唯一决定：
$$
D = L \times \frac{256}{3}
$$
保持 aspect ratio 一致。$L$ 必须是 6 的倍数（因为 5:1 local:global attention）。

每个 ablation 都在 ladder 上做，不是单点比较。TPP (tokens per parameter) 根据实验性质选择：
- Architecture ablations: 100-200 TPP (near Chinchilla optimal)
- Main run: 500-1000 TPP (over-trained，为了 inference efficiency)

### 2.2 EG 公式

Scaling law 拟合：
$$
L = f(C) = A \cdot C^{-\alpha} + E \quad (1)
$$
变量含义：
- $C$: training cost (FLOPs or wall-clock time)
- $A$: scaling coefficient，控制 reducible loss 的整体幅度
- $\alpha$: scaling exponent，控制 loss 随 compute 下降的速度
- $E$: irreducible loss，data 本身的 entropy 下界

对于 candidate run $(L', C')$，定义 EG：
$$
\text{EG} = \frac{f^{-1}(L')}{C'} \quad (2)
$$

直觉：baseline 需要多少额外 compute 才能达到 candidate 的 loss？如果 EG = 1.3，意味着 baseline 需要 30% 额外 compute 才能匹配。

两种 EG：
- **EG_FLOPs**: 用 FLOPs 作为 $C$，**忽略** MFU 差异——目的是看 "如果给同样的实现优化努力，哪个架构更强"
- **EG_Time**: 用 wall-clock time 作为 $C$，反映实际部署的 efficiency

这个区分非常 Karpathy 式——把 model architecture 的 quality 与 implementation efficiency 解耦，但同时也承认后者在实际中重要。

### 2.3 Rank Invariance Hypothesis 的反例

图 6 是一个非常重要的发现。他们做了两个 data mixture：
- code-heavy-mix (~50% code)
- stem-heavy-mix (STEM 大幅 upweight)

小规模实验显示 stem-heavy-mix 在 STEM NLL eval 上更好。但放大到 23B active / 20T tokens 时，**两条曲线在训练中途交叉**，最终 code-heavy-mix 反超。

原因：stem-heavy-mix 中有两个 STEM 数据源虽然质量高，但 fuzzy duplication 多、内容多样性低。小模型受益于这种"集中重复"，但大模型 exhausting the novelty 后收益递减。

这挑战了 "rank invariance" 假设（小模型排序在大模型保持），他们因此更加重视 scaling ladder 上的混合表现。

---

## 3. Pre-training Data Composition

最终 30T tokens 的混合：

| Source family | Unique tokens (T) | Training tokens (T) | Mix % | Avg. Epochs |
|---|---|---|---|---|
| Code | 7.4 | 16.4 | 54.6 | 2.22× |
| STEM | 2.2 | 4.7 | 15.8 | 2.17× |
| Math | 0.3 | 1.6 | 5.4 | 5.28× |
| Books and journals | 0.6 | 0.9 | 3.1 | 1.65× |
| PDFs | 2.7 | 1.4 | 4.7 | 0.53× |
| Web text | 8.1 | 4.5 | 14.9 | 0.55× |
| Multilingual | 8.1 | 0.5 | 1.6 | 0.06× |
| **Total** | **29.2** | **30.0** | **100** | **1.03×** |

几个关键观察：
- **Code 占 54.6%**——非常 heavy code weighting，反映 reasoning RL 的下游需求
- Math 仅 0.3T unique tokens 但被采样 5.28×（最 aggressive）
- Web text 和 PDFs 都没看完一遍（0.55×, 0.53×），即 unique content 充足
- Multilingual 极度 downsampled（0.06×），尽管有 8.1T unique 可用——明确的多语言取舍

数据混合选择用了 **hierarchical local + global search**：
- Local: 固定 high-level category 比例，在 subset 内部调权重（如 code 内部 files vs PRs vs commits）
- Global: 固定 subset 内部比例，调 category 之间的权重

每个 dataset 最多重复 8 epochs 以避免 overfitting。

### 3.1 Deduplication 体系

五个层次：
1. **Boilerplate removal**: 用 line-occurrence statistics
2. **Exact duplicates**: MD5 / SHA hash
3. **Fuzzy duplicates**: MinHash LSH, similarity threshold 0.8
4. **Templated web pages**: skeletonize + fuzzy dedup
5. **Semantic duplication**: Qwen3-Embedding-0.6B 做 embedding，cosine similarity 聚类，每个 cluster 保留有限代表

Cross-dataset dedup 用 global drop-order——duplicate 出现在多个数据集时，保留在 priority 最高的数据集，从其他移除。

参考:
- SemDeDup: https://arxiv.org/abs/2303.09540
- Data pruning beyond scaling laws: https://arxiv.org/abs/2206.14486

### 3.2 NLL vs Accuracy 评估哲学

他们选择 NLL (negative log-likelihood) 而非 accuracy 作为 pre-training 评估，理由：
- **Cost**: NLL 是 next-token prediction，可以高效 batch；accuracy-based eval 需要自回归生成 + judge
- **Robustness**: NLL 用 teacher forcing，不会因 minor formatting 错误 compound
- **Construction cost**: 任何 topic-relevant 内容都可以作为 NLL corpus

这点 Karpathy 你应该会有共鸣——NLL 是 pre-training 的 native objective，eval 应该尽可能接近 training objective。

---

## 4. Training Recipe 细节

### 4.1 三阶段训练

| Phase | Tokens | Context length | GPUs |
|---|---|---|---|
| Pre-training | 30T | 16,384 | 8,192 GB200 |
| Mid-training 1 | 3.4T | 65,536 | 8,192 GB200 |
| Mid-training 2 | 150B | 262,144 | 4,096 GB200 |

Mid-training 把 STEM/math 提升到 35%，code 保持 55%，其余 10%。同时做 context extension。

### 4.2 关键 hyperparameters

- AdamW: $\beta_1 = 0.95$, $\beta_2 = 0.925$, $\epsilon = 10^{-8}$
- Weight decay: 0.1 (attention 0.01, embedding 0.005)
- Gradient clip: 1.0
- LR: warmup 12B tokens, cosine decay from $2 \times 10^{-4}$ to $2 \times 10^{-5}$ (final-to-peak 0.1，而非常见的 0.01)
- **Dropout 0.15** on each layer output before residual add (异常高)
- 输出投影 weight 用 $\frac{1}{\sqrt{N_{\text{residual}}}}$ 缩放

Dropout 0.15 是相当激进的设置。他们的理由：与 weight decay 互补的 regularization，scaling ladder 上 evaluation 性能更好。这暗示在他们的 1T-parameter MoE + 30T tokens 规模下，overfitting 风险实际存在。

### 4.3 数值精度

- Weight/activation: BF16
- FP8 E4M3 for forward GEMMs
- FP8 E5M2 for data-gradient
- BF16 for weight-gradient computation
- FP32 gradient accumulation
- FP32 for: pre-softmax activations, MoE combine, residual stream, embeddings, RMSNorm, router weights, optimizer states

Delayed scaling with 1024-step absolute max history。

### 4.4 训练 loss 曲线 (图 9)

无 smoothing 的原始 loss 曲线显示早期有数次 spike，但每次都快速恢复，**没有 skip 任何 batch，没有 manual intervention**。Spikes 主要发生在 code 数据集，与 dropless routing 下专家高度不平衡相关。

---

## 5. YOLO: Distributed Training Framework

YOLO (You Only Launch Once) 是他们的 in-house framework，类似 Megatron-Core / DeepSpeed / TorchTitan。

### 5.1 Parallelism 组合

- **Data parallelism**: 自定义 ZeRO 1-3，参数始终 sharded form
- **Tensor parallelism**: column/row-parallel GEMMs，TP=1 在 pre-training
- **Context parallelism**: Ulysses-style，仅 mid-training 长上下文
- **Expert parallelism**: EP=64，在 NVLink domain 内
- **Pipeline parallelism**: 实现了但似乎没在 main run 用

YOLO 用 sharding annotations（类似 JAX pjit 或 PyTorch DTensor），但 purely descriptive，不自动 insert communication——避免 accidental synchronization。

### 5.2 Dropless MoE 实现

- 分组 pipelining: dispatch → compute → collect，每组 overlap
- Static-memory dropless mode: 多轮 capped dispatch 处理，避免 imbalance-induced memory swing
- Backward: per-expert-per-round fine-grained recompute
- Custom CuTe DSL symmetric-memory kernels for device-initiated variably-sized all-to-all over NVLink

### 5.3 Determinism 与 Correctness

**Bitwise reproducibility** 是一个 first-class property。需要：
- 数据 pipeline ordering 固定
- Checkpoint 保存所有 stateful data（包括 RNG、FP8 scaling history）
- GPU kernels: deterministic reductions（如 RMSNorm backprop 两阶段 tiled reduction），stable sort 用于 top-k
- 禁用 NVLink SHARP（牺牲性能换确定性）
- 固定 NCCL topology

这点 Karpathy 你应该欣赏——他们把 determinism 作为科学可复现性和 debugging 的基础。

### 5.4 MFU 进化

从 v2 到 v5，每代架构改进都先用上一代 stack 跑（MFU 下降），再做 v-specific 优化恢复 MFU。最终 v5 (MAI-Base-1) 维持 ~20% MFU，尽管 active params 从 23B 增到 35B。

参考:
- Megatron-Core: https://arxiv.org/abs/2603.07685
- DeepSpeed Ulysses: https://arxiv.org/abs/2309.14509
- TorchTitan: https://arxiv.org/abs/2410.06511
- DeepEP: https://github.com/deepseek-ai/DeepEP

---

## 6. RL Climb：核心创新

这是 paper 最有意思的部分。RL 不从 reasoning traces 起步，**从 mid-trained checkpoint 直接开始**，让模型从零学习如何 reasoning。

### 6.1 RL Objective (修改版 GRPO)

基础 GRPO with token-level policy gradient：

$$
\mathcal{J}(\theta) = \mathbb{E}_{q \sim P(Q), y_{1:G} \sim \pi_{\text{old}}}\left[\frac{1}{\sum_i |y_i|} \sum_i \sum_t \min\left(r_{i,t}(\theta) A_i, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) A_i\right)\right] \quad (5)
$$

变量：
- $q$: prompt
- $y_{1:G}$: $G$ 个 rollouts（一组）
- $r_{i,t}(\theta) = \pi_\theta(y_{i,t} | q, y_{i,<t}) / \pi_{\text{old}}(y_{i,t} | q, y_{i,<t})$: importance sampling ratio
- $A_i = (R_i - \text{mean}(R_{1:G})) / \text{std}(R_{1:G})$: response-level advantage

两个关键修改：

#### (a) Adaptive Entropy Control

标准 PPO/GRPO 的 symmetric clipping 容易导致两种问题：
- 上界太大 → entropy 爆炸
- 上界太小 → entropy collapse

他们引入 entropy-dependent relaxation $k$：

$$
r_{i,t}^{\text{tr}}(\theta) = \text{clip}\left(r_{i,t}(\theta), 1-\epsilon, (1-\epsilon)^{-1} + k\right)
$$

$k$ 由 integral controller 动态调整：

$$
\hat{H}(\pi_\theta) = \frac{1}{|\mathcal{T}|} \sum_{(i,t) \in \mathcal{T}} -\log \pi_\theta(y_{i,t} | q, y_{i,<t}) \cdot r_{i,t}(\theta) \quad (7)
$$

$$
k \leftarrow \text{clip}\left(k + \delta \cdot \text{sign}(H^* - \hat{H}(\pi_\theta)), 0, k_{\max}\right) \quad (8)
$$

变量：
- $\hat{H}(\pi_\theta)$: importance-weighted entropy estimator
- $H^* = 0.3$: target entropy
- $\delta = 0.25$: step size
- $k_{\max} = 2.5$: 最大 relaxation

直觉：当 entropy 太低，增大 $k$ 放宽上界，允许 policy 更激进地提升 alternative tokens 的概率；当 entropy 充足，减小 $k$ 收紧 trust region。

这比在 loss 中加 explicit entropy bonus 更稳定。

#### (b) Outer Ratio Clip

GRPO 在两种情况下不 clip：
- $A_i < 0$ 且 $r > 1$ (advantage 负，新策略更高概率——"纠正"方向)
- $A_i > 0$ 且 $r < 1$ (advantage 正，新策略更低概率——"纠正"方向)

实践中这会导致 catastrophic gradient-norm spike。解决方法：硬 outer clip 应用于所有 branches：

$$
r_{i,t}^{\text{out}}(\theta) = \text{clip}(r_{i,t}(\theta), r_{\min}, r_{\max}) \quad (9)
$$

$r_{\max} = 50$, $r_{\min}$ unconstrained。类似 Ye et al. 2020 的 dual-clip PPO。

### 6.2 Reward 设计

$$
R(q, y_i) = R_{\text{task}}(q, y_i) + w_{\text{lang}} \cdot R_{\text{lang}}(y_i) - w_{\text{len}} \cdot R_{\text{len}}(y_i) \quad (10)
$$

#### Language Consistency Reward

$$
R_{\text{lang}}(y_i) = \max(1 - \alpha \cdot n_{\text{non-english}}(y_i), 0) \quad (11)
$$

$\alpha = 0.005$ per word, $w_{\text{lang}} = 0.5$。原因：context length 增加时模型会 mixed-language CoT，这导致 train/inference log-probability divergence spike，destabilize 训练。

#### Length Penalty

$$
R_{\text{len}}(y_i) = \rho_q \cdot \frac{|y_i|}{\ell_{\max}} \quad (12)
$$

变量：
- $\rho_q$: problem $q$ 的 pass rate（越难越低）
- $\ell_{\max}$: 最大 rollout 长度

直觉：难问题（low pass rate）penalty 弱，允许长 reasoning；容易题 penalty 强，鼓励简洁。

$w_{\text{len}} = 0.25$ 直到 64k 阶段；128k 阶段移除 length penalty。

### 6.3 Sampling 策略

- Early exit: 先 $G_{\text{early}} = 16$ rollouts，pass rate 在 $[0.05, 0.8]$ 才继续
- 完整 $G = 128$ rollouts，pass rate 在 $[0.1, 0.8]$ 才用
- Top-p = 0.97，且 **top-p mask replay**：训练时把 rollout 时 excluded 的 tokens logits 设为 $-\infty$，避免 off-policy mismatch
- Length extension curriculum: 8k → 16k → 32k → 64k → 128k

### 6.4 Self-Distillation

这是让 long-running climbs 可持续的关键。流程：
1. 在 RL 过程中收集 rollouts
2. 对 mid-trained checkpoint 做 SFT
3. 用得到的 model 作为新的 RL 起点

用途：
- 从 prompting 转换到 native chat format
- 从 run failure 恢复（numerical collapse）
- 新的 pre/mid-trained checkpoint 出现时 carry forward progress

最佳实践（extensive ablations 后）：
- **O(1M) traces 足够**匹配 teacher，更多 diminishing returns 且 over-constrain policy
- 包括 incorrect 最终答案的 traces 也 OK（最终他们用 successful traces only 因为 RL 产量远超 1M）
- 用 climb **后期多个 checkpoint** 的 traces（不是 only final，也不是 only early）
- 多样性来自 prompt diversity，而非 traces per prompt
- SFT 期间用 dropout 0.15 + load balancing coef $10^{-2}$，RL 期间用 $10^{-5}$

参考:
- GRPO 原始 paper: https://arxiv.org/abs/2402.03300
- DAPO (token-level GRPO): https://arxiv.org/abs/2503.14476
- Self-distillation / STaR: https://arxiv.org/abs/2203.14465
- BAPO (adaptive clipping): https://arxiv.org/abs/2510.18927
- Entropy mechanism in RL: https://arxiv.org/abs/2505.22617

---

## 7. 三个 Specialist Climb

### 7.1 STEM Climb

- 5M+ samples，最难的 550k (q, a) pairs
- Reward 用 SymPy 形式验证 / AI judge / 竞赛编程 test cases
- 多阶段数据 pipeline: hierarchical parsing → QA pairing → curation → scoring
- Blind grading 防止 faulty ground truth：强模型 + judge 对比，丢弃 suspect ground truth

### 7.2 Agentic Climb

#### SWE Environments 构建

从 102M public GitHub PRs 出发：
1. Filter: merged, <15 files, 含 code 和 test changes, 有 linked issues → 4.87M PRs
2. Automatic agentic environment building: LLM agent 创建 Docker files → 2.08M 通过
3. Reference grading signal extraction: F2P (fail-to-pass) + P2P (pass-to-pass) tests → 745K 通过
4. Environment and grader verification: empty patch fails, golden patch passes → 265K 通过
5. Quality filtering and rewriting

#### Reward Hacking 防护

三类已识别的 hacking：
1. **Internet search**: 网络隔离 / 严格 allowlist
2. **Local git history search**: sanitizing git commits（"time-traveled" repository）
3. **Test tampering**: reset test files before grading，hidden test changes

#### General Tool Use

- 150+ synthetic environments，130K+ tasks
- Pipeline: environment bootstrapping → task creation → verification and refinement
- 50+ tools per environment（vs SWE 的 2 个）
- 包括 "no tool use" 任务防止 overeager calling

### 7.3 Helpfulness & Safety Climb

三种 reward：
- **Reward Model**: post-trained MAI-Base-1，输入 k-way side-by-side，cyclic permutation 减少位置 bias
- **AI Judge**: rubric-guided，fast iteration
- **Verifiable Rewards**: instruction following constraints 直接检查

#### Lexicographic Reward Shaping 与 Gated Reward Application

由于 reward scales 不可比，且某些 criteria 是 non-negotiable：

**Lexicographic**: lower-importance reward 仅当 group 内所有 rollouts 在 higher-importance reward 上 tied 时才 active。Invariant to absolute scale。

**Gated**: higher-importance reward 必须先达到 minimum，lower-importance reward 才被应用。Safety 是经典 case——unsafe response 拿 minimum reward，无论其他 score 多高。

### 7.4 Consolidation

三步：
1. **Consolidation SFT**: 从三个 specialist teachers 蒸馏到一个 model
   - STEM/Coding: 56% sample weight, 89% token weight
   - Agentic: 11% sample, 9% token
   - Helpfulness & Safety: 33% sample, 2% token
2. **Consolidation RL**: 轻量级 RL 进一步提升 safety/over-refusal/style，保留 STEM 数据防 reasoning 退化

### 7.5 Honesty 设计

Reward 分五个类别：CONFIDENT_CORRECT, UNCONFIDENT_CORRECT, NOT_ATTEMPTED, UNCONFIDENT_INCORRECT, CONFIDENT_INCORRECT。

加权：confident-correct 最高 reward，confident hallucination 最 steep penalty，abstention 中性，unconfident-correct 减少奖励（防 over-hedging）。

---

## 8. RL Infrastructure: Rocket

### 8.1 架构

- Controller: 单进程，加载 RL tasks，聚合 completed rollouts，发给 learner
- Problem Worker: 单进程，生成 rollouts（发给 rollout worker），计算 advantage
- Rollout Worker: 单 rollout 生成 + 局部 grading
- Router + Inference: SGLang-based

### 8.2 Inference 关键优化

- 单 turn: KV cache 是瓶颈，禁用 prefix caching 让 sliding-window tokens 完全 evict
- 多 turn: prefix caching hit rate 97-98%
- MoE routing replay: 训练和推理 router 决策一致
- Top-p mask replay: rollout 时的 top-p truncation mask 在训练时复用

### 8.3 Numerics Gap 处理

YOLO (learner) 和 SGLang (inference) 用不同 kernels、scheduling、parallelism。即使 per-token logprob 差异小，也会在长 rollout 中 compound，destabilize importance-sampling correction。

缓解：
- **bf16 双向**（learner 和 inference 都用 bf16）
- MoE routing replay
- Top-p mask replay

### 8.4 Weight Transfer

异步 RL 中，每 k 步要把 learner weights 传到 inference fleet。挑战：两侧 sharding layout 不同（FSDP, PP, DP-attention, TP, precision, quantization 都可能不同）。

解决：编译 transfer plan——每个 parameter 计算 source/destination rank overlap，记录 byte extent 和 required transforms（dtype cast、layout permutation）。Resharding implicit in intersection。

---

## 9. 评估结果

### 9.1 公开 benchmark

| Benchmark | MAI-Thinking-1 | Sonnet 4.6 | Opus 4.6 | GPT 5.4 | Kimi K2.6 | DeepSeek V3.2 | DeepSeek V4 | GLM-5.1 |
|---|---|---|---|---|---|---|---|---|
| AIME 2025 | 97.0 | 95.6 | 99.8 | — | — | 93.1 | — | — |
| AIME 2026 | 94.5 | — | — | — | 96.4 | — | — | 95.3 |
| HMMT Feb 2026 | 84.9 | — | — | — | 92.7 | — | 95.2 | 82.6 |
| GPQA Diamond | 84.2 | 89.9 | 91.3 | 92.8 | 90.5 | 82.4 | 90.1 | 86.2 |
| LCB v6 | 87.7 | — | — | — | 89.6 | 83.3 | 93.5 | — |
| Terminal-Bench 2.0 | 46.0 | 59.1 | 65.4 | 75.1 | 66.7 | 46.4 | 67.9 | 69.0 |
| SWE-bench Verified | 73.5 | 79.6 | 80.8 | — | 80.2 | 73.1 | 80.6 | — |
| SWE-Bench Pro | 52.8 | — | 53.4 | 57.7 | 58.6 | — | 55.4 | 58.4 |

定位："competitive range"，不 lead field，但跨多个 benchmark 一致强。AIME 2025 超过 Sonnet 4.6，SWE-Bench Pro 接近 Opus 4.6。

### 9.2 Human Side-by-Side

1276 tasks，30% multi-turn，来自 expert-authored + Microsoft Copilot logs（PII 过滤）。

vs Sonnet 4.6: win 49%, tie 6%, lose 45% → net +0.07
vs Opus 4.6: win 43%, tie 5%, lose 52% → net -0.07

在 conciseness/relevance 和 style/tone 上明显优于 Sonnet/Opus，在 factuality/instruction-following/completeness 持平。

---

## 10. CoT Evolution (Appendix C)

这是非常有趣的部分——观察从 weak 到 strong 的 CoT 行为变化。

### STEM CoTs

1. **Weak guesses, strong works hard**: Weak model 从 visible roots 猜 minimizer；strong model derive 候选 + 验证 domain（如 $x > 0$）

2. **Weak brute force, strong finds invariants**: Weak model 假设 cubing 是 bijection on units mod $3^7$（错：gcd(3, φ(3^7)) = 3）；strong model 识别 cube 是 index-3 subgroup，特征是 $\equiv \pm 1 \pmod 9$

3. **Strong models are skeptics**: Strong model 主动 "Wait, let's re-examine"，small test case 验证 converse

### Agentic CoTs

1. **Strong writes and runs unit tests; weak only sanity checks**: Strong model verify with tests，weak model 只 review problem statement 要求

2. **Strong does evidence archaeology; weak fixates edit mechanics**: Strong 先收集 repository evidence (reverted commit, payloads, tests) 再 patch；weak 花大量 turns 在 exact edit mechanics 上

3. **Strong seeks source of truth; weak speculates adjacent paths**: Strong 追溯 data source（generated code 例子）；weak 在 adjacent path 上 trial-and-error

这些观察很有 teaching value，建议作为 reasoning model 训练的 diagnostic tool。

---

## 11. Long Context Extension (Appendix B)

### 关键发现

1. **Progressive vs full long-context mid-training**: 32K mid-training 1T tokens + 短 256K extension ≈ full 128K mid-training 1T tokens。这意味着 mid-train at short MFU-friendly length，然后短 extension phase 即可。

2. **Adaptation 极快**: 256K NLL 改进 90% 发生在前 1-10% iterations。Context extension 不是学新能力，而是 calibration position/attention for OOD positions。

3. **Position asymmetry**: 没 long-context training 时，recent (end-of-context) 位置比 distant 位置更难 retrieve。Long-context training 解决这个 asymmetry。

4. **4× length extrapolation**: 32K-trained model 能在 128K 上正确回答 QA——但更远就退化。

最终方案：64K mid-training + 150B tokens @ 256K。保守选择以应对 post-training interaction。

---

## 12. Safety Red Teaming

### Taxonomy of 6 Recurring Attack Patterns

1. Multi-turn escalation under benign pretext
2. Fictional or novelistic framing
3. Credentialed-persona pretexts
4. Gradual recursion / formatting drift
5. In-context age-indicator bypass
6. Authoritative-document fabrication

### Mitigation 效果

Top priority categories:
- Jailbreak success 降 44%
- Hate & fairness 降 43%
- Child safety issues 降 30%
- Mental health attacks 降 20%
- Total aggregate attack success 降 22%

### Independent Red Teaming 关键发现

- **TAP (Tree of Attacks with Pruning)**: 闭环 adversarial data pipeline，大幅降低 TAP jailbreak susceptibility
- **Low-resource language framing**: Yoruba, Telugu, Amharic, Burmese, Khmer, Malay——通过 multilingual adversarial seeds + 翻译 targeted languages 缓解

---

## 13. Cluster Environment

### Goodput 定义

$$
\text{Goodput} = \frac{\text{Ideal training duration}}{\text{Actual wall-clock duration}}
$$

MAI-Base-1 pre-training 达到 **90.0% goodput at 8K GPUs**，total overhead 仅 51 小时。分解：
- Recomputation: 6.5 小时 (15% of overhead)
- Non-stepping: 14 小时 (27%)
- MFU drop: 18 小时 (35%) ← 最大瓶颈

### 训练 in Phoenix, AZ

- Microsoft-owned infrastructure
- LEED Gold Certification
- Renewable diesel backup generators
- $50M+ 投资于 municipal water storage (City of Goodyear)
- Datacenter Academy 在 Estrella Mountain / Glendale Community College

### MAIA-200 部署

推理部署在 Microsoft MAIA-200 硬件上，相比 GB200 部署，相同 rack power budget 下 token generation throughput 高 40%+。

---

## 14. 总结性直觉

这篇 paper 对 Karpathy 你来说应该有特殊共鸣。它把 model development 当作 **可编译、可复现、可优化** 的工程系统，而不是 isolated research experiments。

核心 takeaways：

1. **Hill-climbing machine 是真正的资产**：单一 model 是 snapshot，pipeline 是 enduring capability。
2. **No distillation** 是 deliberate sacrifice：放弃 short-term gain 换 long-term steerability 与 robustness。
3. **Scaling ladder + EG** 是科学决策的基础：避免单点 ablation 误导。
4. **Rank invariance 不成立**：小模型最佳 mixture 在大模型可能反转——diversity 在 large scale 更重要。
5. **Dropout 0.15 在 1T MoE 规模有效**：overfitting 在巨大 MoE 上实际存在。
6. **Adaptive entropy control + outer ratio clip** 是 GRPO 稳定化的两个关键 trick。
7. **Self-distillation** 是 long-running RL climbs 的 enabler——从 collapse 恢复、跨 checkpoint 迁移。
8. **Top-p mask replay + MoE routing replay + bf16 双向** 是 numerics gap 的 mitigation stack。
9. **CoT evolution patterns** 可以作为 reasoning model 的 diagnostic tool。
10. **YOLO 的 bitwise determinism** 是科学 reproducibility 与 debugging 的 foundation。

这个 recipe 在很多方面呼应了你提到的 "Tesla Autopilot 的 idea-to-deploy loop"——快速迭代、data-driven、infrastructure-aware。Microsoft 似乎在 LLM 上构建了类似的东西。

参考链接汇总：
- Paper PDF: https://www.microsoft.com/en-us/research/publication/mai-thinking-1-building-a-hill-climbing-machine
- SWE-Bench: https://openai.com/index/introducing-swe-bench-verified/
- AIME: https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions
- LiveCodeBench: https://livecodebench.github.io/
- GPQA: https://arxiv.org/abs/2311.12022
- FlashAttention-4: https://arxiv.org/abs/2603.05451
- SGLang: https://arxiv.org/abs/2312.07104 (实际是 https://github.com/sgl-project/sglang)
- GRPO: https://arxiv.org/abs/2402.03300
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- DeepSeek-R1: https://arxiv.org/abs/2501.14248
- Kimi K2: https://arxiv.org/abs/2507.20534
- Gemma 3: https://arxiv.org/abs/2503.19786

如果你想深入任何 section 的更多技术细节，我可以再展开——比如 LatentMoE 的具体 memory layout、SWE environment build pipeline 的 stage 细节、Rocket 的 controller/worker fault tolerance protocol，或是 deduplication 各 stage 的工程实现。
