---
source_pdf: falcon-h1-tiny-a-series-of-extremely-small-yet-powerful-language-models-redefining-capabilities-at-small-scale.pdf
paper_sha256: 7e04455d6bb0bc21342679b4c73fddc1264933f760e5e36b07c067fc2fd32eea
processed_at: '2026-08-04T06:31:21-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用户要求用"人话"——也就是更直白、更通俗的语言来重新讲一遍。虽然要求用中文，但专有名词和名词用英文。用户的指令里特别提到"I hate 不是 ... 而是 句式"，所以要避免这种句式。另外，用户要求尽可能多地回答，宁愿出现一些不准确的内容也不要错过任何可能的联想，同时需要提供详细的技术讲解（比如方法公式、架构图解析、实验数据表），还要带上网络链接作为参考，回答的目的是帮助用户建立直觉。

现在要用"人话"——也就是更口语化、更易懂的方式——来解释这篇关于极小规模语言模型（small scale LLM）的论文。核心是探讨"为什么能做出这么小但这么强的模型"。

整篇论文其实在问一个根本问题：100M参数的模型，到底能干多少事？答案是，如果能巧妙地组合数据策略（data strategy）、优化器（optimization）和架构（architecture），小模型的表现会远超人们的想象。

先从核心直觉说起。传统观点认为模型规模决定一切，大模型吃大数据，小模型吃小数据。但这篇论文指出，问题不在于大小，而在于你如何管理模型的记忆和遗忘。模型其实就像人一样，有个"记忆窗口"（memorization window）。如果你让模型反复看同样的数据，只要间隔足够长，它早忘了上次看的内容了，所以多看几遍并不会overfitting（过拟合）。这个发现打开了一扇门：你可以重复用高质量数据，而不用担心模型死记硬背。这对小模型尤其重要，因为小模型本身记性差，反而更不怕重复。

论文里做了实验测量这个记忆窗口。他们在一个7B的模型上观察到，看过一段时间后，模型对曾经看过的数据，loss会上升，说明它忘了。根据线性缩放估算，100M的小模型记忆窗口大约是5GT（5 billion tokens）。这意味着如果你有5GT的高质量数据，你可以让100M模型看100遍都不会过拟合，因为它根本记不住。这引出了论文最大的反直觉发现：对小模型而言，直接用SFT（supervised fine-tuning，监督微调）数据预训练，比传统的先预训练再微调效果更好。这叫"anti-curriculum"（反课程学习）。传统做法是先喂通用数据，再针对微调；但论文发现，如果你在第一天就把SFT数据混进预训练里，模型有整整800GT的训练时间慢慢消化这些指令模式（instruction pattern），效果远超只在最后用4GT来微调。数据表显示，在IFEval（一个指令遵循评测）上，anti-curriculum比传统方法高出10分以上，DPO之后更是达到66分，而传统方法只有53分。背后的逻辑是：小模型学得慢，需要更多时间来吸收复杂模式，所以越早暴露SFT数据，模型就有越长的优化窗口（optimization horizon）去学习。在推理（reasoning）任务上，结果更夸张。直接在推理数据上预训练的小模型，AIME24从3/30直接翻倍到6/30，MATH500也从0.2涨到0.4。这背后的道理是：推理需要长序列的逻辑链，小模型如果只在最后短期接触这些数据，根本学不会，必须在预训练阶段就让它泡在推理数据里。

当然，数据策略只是一方面，论文还做了两个关键的技术创新。首先是优化器（optimizer）。他们没有用主流的AdamW，而是用了Muon优化器。Muon的核心想法是：当你更新权重矩阵时，不要直接用梯度，而是先对梯度做一种谱归一化（spectral normalization），让不同方向上的更新强度更均衡。这样训练更稳定，效果也更好。论文还配合了一种叫LRM（Learnable Row/column Multipliers，可学习的行/列乘子）的技术。传统的权重矩阵有个问题：它的范数（norm，可以理解为整体大小）被学习率和权重衰减这两个超参数锁死了，模型无法根据数据自己调整每行每列的大小。LRM的做法是给每一行每一列都加上一个可学习的乘子，相当于让模型自己决定每行每列的规模。这听起来是个小改动，但实测能带来20%的相对性能提升。

架构方面，他们基于Falcon-H1，这是一种把Mamba（SSM，State Space Model，状态空间模型）和Attention（注意力机制）并行组合的混合架构。他们做了大量消融实验（ablation study），发现两个关键结论：第一，对小模型而言，SSM的容量比MLP（Multi-Layer Perceptron，多层感知机）更重要，因为小模型的瓶颈主要在于无法记住长程上下文，而SSM正是处理长程依赖的。第二，模型深度和宽度需要权衡，太深会损害吞吐量（throughput），太浅又会损害STEM任务，他们最终选了一个中庸的配置。

但小模型也有它的极限。论文发现，在需要生成复杂长文本的任务上，小模型会崩溃。最典型的是Function Calling（函数调用）任务里，如果把chain-of-thought（CoT，思维链）数据混进训练，模型会陷入死循环，不断重复同一个token。原因很简单：CoT对90M的模型太复杂了，它压缩不了，只能退化成最简单的行为——重复。解决办法是直接把所有CoT数据过滤掉，只保留干净的函数调用示例。同样的现象在多语言（multilingual）上也出现了。他们在17种语言上做了实验，发现anti-curriculum的优势消失了，和传统做法效果几乎一样。论文的解释是，100M的容量太小了，无法同时消化17种语言的指令模式，再多的暴露（exposure）也撞上了容量天花板（capacity wall）。这说明了一个很直觉的道理：小模型的极限不在于数据多少，而在于它本身的复杂度处理能力。

论文最惊人的结果在推理模型上。他们训了一个0.6B的推理模型，直接在推理数据上预训练，然后做了60步GRPO（一种强化学习算法）。这个0.6B的模型在AIME24上达到了75分（pass@1），超过了1.7B的Qwen3（47分）和1.5B的DeepSeek-R1-Distill（29分）。从测试时扩展（test-time scaling）的角度看，0.6B模型采样16次的成本，可能比7B模型采样1次还低，但效果可能更好。这重新定义了小模型在推理任务上的实用性。

代码生成方面，他们专门做了个Coder模型，支持FIM（Fill-in-the-Middle，填空补全）任务。有个很有意思的发现：传统上预训练时会把非FIM部分的loss mask掉（不计算损失），但他们发现不mask掉反而更好。因为这样每个token都在学习，信号更密集。但另一个反直觉的发现是，Dropout（一种正则化技术，随机丢弃神经元）在极端数据重复的场景下重新变得有用。因为数据重复太多遍后，模型会开始死记硬背，Dropout能有效防止这种记忆化（memorization），虽然会牺牲一点代码生成的绝对性能，但FIM补全会稳定很多。

这告诉我们，大模型时代的"Dropout无用论"在小模型多轮训练（multi-epoch）的场景下不成立，深度学习的基本功在小模型领域重新变得重要。总的来说，这篇论文给了一套新的小模型训练范式：数据上利用记忆窗口来安全地重复高质量数据，训练上越早暴露目标任务的数据越好，优化上用Muon加LRM释放权重矩阵的潜力，架构上优先保障SSM容量，同时要小心复杂任务导致的退化现象。

这篇论文的真正价值在于把小模型从一个"大模型缩水版"变成了一个有独立设计原则的研究对象。它告诉我们，小模型不是在和大模型比谁更大，而是在回答一个更根本的问题：给定有限的参数预算，如何榨干每一份算力和数据的潜力？

---

# Falcon-H1-Tiny 论文深度解析

这是 Technology Innovation Institute 的 Falcon-LLM Team 在 2026年1月15日发布的工作,目标是在极小参数规模(90M-100M)下重新定义 small language model 的能力边界。这篇 paper 同时是 model release + research blog,涵盖了 architecture、optimization、data strategy 三个独立但耦合的 axis。

**主仓库/集合**: https://huggingface.co/tiiuae
**Falcon-H1 主 paper**: https://arxiv.org/abs/2507.22448
**LRM paper**: https://arxiv.org/abs/2601.04890
**Muon paper**: https://arxiv.org/abs/2502.16982

---

## 一、整体动机与核心 hypothesis

这篇 paper 想挑战两个 common belief:
1. **小模型只能做小任务** —— 他们用 90M 模型做到 tool calling、code FIM、instruction following、reasoning 全部在 comparable size 中 SoTA
2. **必须先 pretrain 再 SFT** —— 他们提出 anti-curriculum,直接把 SFT 数据混入 pretraining

背后关键的 insight 是:**model 的 memorization window 是数据重复的真正瓶颈,而不是单纯的 epoch 数**。这给了多 epoch 训练一个理论上的"安全边界",使得 SFT 数据可以从训练第一天就暴露给模型。

---

## 二、Memorization-Aware Repetition —— 全文最关键的 insight

### 2.1 公式与变量

设:
- $D$ = high quality (HQ) data 源的 token 总量
- $T$ = 整个 pretraining 的总 token budget
- $p$ = HQ 数据在 mixture 中的 fraction
- $D_{ep}$ = HQ 数据的 epoch size(每看一遍 HQ 数据需要的 token 数)
- $M$ = 模型的 memorization window(模型"忘记"过去看过 token 所需的 token 数)

关键关系:

$$
D_{ep} = \frac{D}{p}
$$

含义:当 HQ 占 mixture 的 fraction $p$ 越大,看一遍 HQ 需要的总 token 越少,即 epoch 越短。

经典 single-epoch pretraining 给出上界:

$$
p \le p_{ep} \equiv \frac{D}{T}
$$

含义:如果不允许 HQ 数据重复,HQ 的 fraction 上界是 $D/T$。这导致一个真实问题 —— 若 $T \gg D$,HQ fraction 就被锁死在很小比例,严重限制下游能力。

memorization-aware repetition 给出新上界:

$$
p \lesssim p_{mem} \equiv \frac{D}{M}
$$

含义:只要 epoch size $D_{ep} \gtrsim M$,模型在看到第二遍 HQ 数据时已经"忘记"第一遍细节,所以重复无害。这把 HQ fraction 的上界从 $D/T$ 解耦成 $D/M$,可以远超 single-epoch 限制。

### 2.2 memorization window 的测量

Figure 9 of Falcon-H1 paper 的实验:FalconMamba-7B 在训练后期,把训练样本按"上一次被模型看到的时间"排序,测当前 checkpoint 在这些样本上的 loss。

观测:训练时正常 new token 的 loss(蓝线) 与"模型已经看过的 token 的 rollback loss"(橙线) 之间存在 gap,这个 gap 反映 memorization 程度。随着 delay 增大,gap 衰减 → 说明存在 forgetting curve。

**实证估计**:
- FalconMamba-7B 的 $M$ ~ 100 GT(乐观估计)~ 500 GT(保守估计)

线性 scaling 到 100M 参数模型:
$$
M_{100M} \approx 500 \text{ GT} \times \frac{100\text{M}}{7\text{B}} \approx 7 \text{ GT}
$$

paper 这里实际取 5 GT 作为 working estimate,所以 SFT 数据 ~5 GT 时:

$$
p_{mem} = \frac{D_{SFT}}{M} = \frac{5 \text{ GT}}{5 \text{ GT}} = 100\%
$$

—— 这正是 paper 给的数字。意思是 **100M 模型直接用 100% SFT 数据 pretrain 也是安全的**,因为模型本身记不住所有细节。

参考链接: https://arxiv.org/abs/2507.22448

---

## 三、Anti-Curriculum 训练范式

### 3.1 两种策略对比

**经典 curriculum**(standard pretrain → SFT):
- Stage 1: 在 general mixture 上长程 pretraining,例如 800 GT
- Stage 2: 短程 SFT 在 chat/instruction 数据上,通常 ≤4 个 epoch

**Anti-curriculum**(SFT-pretraining,即把 SFT 直接 mix 进 pretraining):
- 单一 stage:SFT 数据以 memorization-aware 上界 $p = D_{SFT}/M$ 混入 pretraining mixture
- 训练超参与 stage 1 一致

### 3.2 关键结果(IFEval,90M 模型)

| 训练 recipe | pre-DPO | post-DPO |
|---|---|---|
| SFT (Curriculum) | 40.77 | 53.47 |
| SFT-pretrain (Anti-curriculum) | 50.11 | **66.08** |

Anti-curriculum 在 IFEval 上比 curriculum 高 ~10 points,且只用单一 stage。背后的 intuition:**对小模型而言,模型 capacity 本身就限制 memorization,所以早期暴露 SFT 数据让模型有更长 optimization horizon 去消化 instruction-following pattern**。

### 3.3 Reasoning 数据上同样的现象

直接对比 reasoning-pretraining vs pretrain-then-SFT(90M 模型,无 GRPO,best checkpoint):

| Benchmark | Reasoning SFT | Reasoning pretraining |
|---|---|---|
| AIME24 pass@16 | 3/30 | **6/30** |
| AIME25 pass@16 | 2/30 | **9/30** |
| MATH500 | 0.2 | **0.4** |

reasoning pretraining 在所有 metric 上接近翻倍。这是个非常强的信号 —— 小模型直接在 reasoning traces 上 pretrain 远比先学 general 再 finetune 更有效率。

直觉:reasoning traces 是长 sequence、结构化 pattern,小模型需要更多 gradient steps 才能 fit。把 reasoning 数据放在 pretraining 阶段(800 GT 而非 4 GT)等于给模型 ~200x 更长的 optimization 时间去吸收这些 pattern。

---

## 四、Learnable Multipliers (LRM) + Muon optimizer

### 4.1 LRM 的动机 —— noise-WD equilibrium trap

Falcon-LLM team 在 Velikanov et al. (2026) 中提出:standard weight matrix 的 norm 在 training 中被锁定在一个由 hyperparameter 决定的 equilibrium,而**不**是由 data 决定。具体来说,AdamW 更新规则下,weight 的 norm $||W||_F$ 满足:

$$
\frac{d ||W||^2}{dt} \approx 2 \text{LR} \cdot \mathbb{E}[g^T W] - 2 \text{WD} \cdot ||W||^2
$$

平衡时 $||W||^2 \to \text{LR} \cdot \mathbb{E}[g^T W] / \text{WD}$,即 norm 由 LR/WD 决定。这叫 noise-WD equilibrium,因为 weight 只能在这个特定 norm 附近做小波动,无法学到 data-driven 的 row/column scale。

参考: https://arxiv.org/abs/2601.04890

### 4.2 LRM 公式

对 weight matrix $W \in \mathbb{R}^{m \times n}$,LRM 把它替换为:

$$
W_{\text{LRM}} = \text{diag}(u) \cdot W \cdot \text{diag}(v)
$$

其中:
- $u \in \mathbb{R}^m$ 是 row-wise learnable multiplier(对应输出维度)
- $v \in \mathbb{R}^n$ 是 column-wise learnable multiplier(对应输入维度)
- $W$ 本身仍由 Muon 更新,$u$、$v$ 由 Adam 更新

行/列的 norm 现在变成 $|u_i| \cdot ||W_{i,:}|| \cdot |v_j| \cdot ...$,允许模型自学每行每列的 scale,跳出了 noise-WD equilibrium。

### 4.3 实验验证(200 GT, Falcon-H1-Tiny 架构)

| Setting | MMLU | BBH | GSM8K |
|---|---|---|---|
| Muon baseline | random-ish | random-ish | random-ish |
| Muon + LRM | +20% rel gain | +20% rel gain | +20% rel gain |

(论文 Table 1 from LRM paper,这里仅给大致趋势,原文数字可在 LRM paper 查询)

参考: https://arxiv.org/abs/2601.04890

### 4.4 Muon optimizer

Muon(Keller Jordan 等 → 后被 Liu et al. 2025 系统化)对 2D matrix 参数用 Newton-Schulz iteration 计算 update 的 orthogonal polar factor,等价于对 gradient 做一个 spectral normalization,使得每个 singular direction 的 update 强度被均衡化。

更新规则(简化版):

$$
W_{t+1} = W_t - \text{LR} \cdot \eta \cdot \text{NewtonSchulz}(G_t / ||G_t||_F)
$$

其中 $G_t$ 是 gradient,$\eta$ 是 scaling factor 用来 match AdamW 的 RMS norm。paper 中采用 Liu et al. (2025) 的 modification:加 weight decay + scale update 的 RMS norm。

实测:stable training 几乎同样的最优 LR,evaluation 比 AdamW 好。所以 Falcon-H1-Tiny 全系列用 Muon + LRM。

参考: https://arxiv.org/abs/2502.16982

---

## 五、架构 Ablation —— 在 90M 参数预算内寻找最优

### 5.1 Base 架构

Falcon-H1 架构 = parallel Mamba SSM + Attention 在每个 mixer block 内并行组合。Vocab=32768(最小),embedding 与 output head tied,tokenizer 含 LaTeX token、digit/punctuation split。

### 5.2 Exploration 1: Depth vs Width

固定 90M,比较三个配置:
- shallow + wide(13 layers)
- mid(27 layers)
- deep + narrow(50 layers)

结果:
- deep 模型在 MMLU/MMLU-Pro 上明显更好
- mid 模型在 hellaswag/commonsense 上更好
- deep 模型 throughput 比 mid 慢 2x

**最终选择 mid 27 layers**(再减到 24 layers 让总参数落在 90M 附近)—— 这是一个 throughput vs STEM 的 trade-off。

直觉:小模型容量有限,更多 layers 给模型更多 nonlinear transformation 的深度,但每层 SSM 状态机的 capacity 反而被压缩。对 commonsense 任务,SSM 的 long context 需要 width;对 STEM 任务,深层 composition 需要 depth。

### 5.3 Exploration 2: MLP factor vs SSM dimension

固定参数预算,比较四个配置(增大 d_mlp 同时减小 d_ssm):

| Config | d_ssm | d_mlp | hidden | Result |
|---|---|---|---|---|
| cfg2 (baseline) | 768 | 768 | 512 | **best** |
| cfg5 | 256 | 2360 | 368 | poor |
| cfg6 | 256 | 1700 | 448 | poor |
| cfg7 | 256 | 1280 | 512 | slightly worse |
| cfg8 | 256 | 700 | 640 | worse |

**结论:对 tiny model,SSM capacity 比 MLP width 更有价值**。直觉:MLP 提供 per-token 的 nonlinear transformation,而 SSM 提供 long-range context integration;在 90M scale 下,模型的 bottleneck 主要在"无法记住 long-range context",所以加 SSM 比加 MLP 更划算。

### 5.4 Exploration 3: KV heads

固定参数,增加 KV heads 同时减少 MLP:

| Config | total_heads | kv_heads | MLP dim | Result |
|---|---|---|---|---|
| baseline | 8 | 2 | 768 | best |
| cfg10 | more | more | less | second best |
| cfg11 | even more | even more | large | worse than cfg10 |

存在一个 sweet spot,太多 KV heads 反而伤害。这与 GQA/MQA literature 一致 —— 太多 KV heads 会稀释 per-head capacity,太少又退化成 MQA。

---

## 六、Falcon-H1-Tiny-English 训练 recipe

### 6.1 Base model

- 800 GT 总训练
- WSD schedule:100 MT warmup, exponential decay ×64 over 100 GT
- 100 GT 后转 Power scheduler(sqrt LR decay)
- Batch size rampup 40 GT,LR 随 sqrt(batch) scaling
- μP(Maximal Update Parameterization)配合 35 个 multipliers 从 Falcon-H1 转移过来

### 6.2 Web data ratio ablation

| Web ratio | STEM | Commonsense |
|---|---|---|
| 10% | preserved | lower |
| 20% | preserved | **better** |

最终选 20% web(FineWeb + FineWeb-EDU) + 80% STEM/reasoning/code。

### 6.3 Anti-curriculum 最终 recipe

- 25% SFT data + 75% base mixture
- 训练 800 GT,decay stage 100 GT
- **不需要额外 SFT 阶段**(实测加 SFT 阶段在 checkpoint 上无明显 gain)
- DPO 阶段加在 anti-curriculum model 之上

### 6.4 DPO 调参

LR sweep:1e-5、3e-7、3e-6、1e-6

发现:
- 多 epoch DPO 在第一个 epoch 后开始退化(尽管 DPO reward 仍在上升 —— 经典 reward overfitting)
- 最终只用 1 epoch DPO + cosine decay
- 最优 LR 在 1e-6 与 3e-6 之间

IFEVAL 从 ~50 飙到 65+,这个 boost 远大于 SFT-curriculum 路线。

参考: https://arxiv.org/abs/2311.07911 (IFEval)

### 6.5 与 SmolLM2-135M、Mobile-LLM-140m 对比

Base model 在 STEM 上(MMLU 32.3 vs SmolLM2 24.2, MMLU-pro 7.18 vs 1.0)显著领先,在 commonsense 上略弱于 SmolLM2(因为 SmolLM2 用了更多 web 数据)。

SFT model:
- IFEval: 66.08(Falcon-H1-Tiny) vs 30.69(SmolLM2-135M)
- MT-Bench: 4.33 vs 2.68
- LiveBench: 15.69 vs 8.25

参考: https://arxiv.org/abs/2502.02737 (SmolLM2), https://arxiv.org/abs/2509.24945 (Mobile-LLM-R1)

### 6.6 量化部署

Q8_0 量化后 footprint ~90 MB。可以本地跑在 llama.cpp 上。

---

## 七、Falcon-H1-Tiny-Multilingual

### 7.1 配置

- Vocab 增到 65k,参数 ~100M
- 17 种语言:Czech、German、Spanish、French、Hindi、Italian、Japanese、Korean、Dutch、Polish、Portuguese、Romanian、Russian、Swedish、Urdu、Chinese
- 数据分布:50% Common Crawl multilingual + 33% Wikipedia + 17% textbooks
- 800 GT training, 100 GT decay

### 7.2 数据混合(pretrain-SFT 变体)

- 40% English pretrain
- 20% English SFT
- 20% multilingual pretrain(10% CC + 6.67% wiki + 3.33% textbooks)
- 20% multilingual SFT(19.5% post-training dataset + 0.5% conversational)

### 7.3 关键发现 —— 与 English 不同的结论

**Anti-curriculum 在 multilingual 上没有明显优势**!Curriculum-SFT 和 SFT-pretrain 在 multilingual_benchmarks 上几乎打平。

paper 给出的 hypothesis:
- 100M 容量太小,无法吸收多语言 SFT 的 pattern,所以再多 exposure 也无济于事
- multilingual SFT 数据质量本身有限,并非 exposure timing 问题

**DPO 仍然有效**:IFEval 从 46.50 → 52.00(Curriculum-SFT-DPO)。

**M-MMLU 大幅领先 SmolLM2**:55.00 vs 25.63(虽然绝对值还是 modest)。

直觉:multilingual 是 cross-lingual generalization 问题,需要 embedding space 上不同语言共享 representation;100M 模型容量不足以同时学 17 种语言 + 多任务,所以再多 SFT 数据也撞到 capacity wall。

---

## 八、Falcon-H1-Tiny-R:Reasoning model

### 8.1 训练 strategy

WSD schedule:
- Warmup: 100 MT
- Constant LR: 500 GT
- Decay stage 1: exponential ×4 over 50 GT
- Decay stage 2: exponential ×256 over 350 GT
- 总计 900 GT

paper 强调:**大部分 gain 来自 decay 阶段**。这印证了 WSD schedule 的特性 —— decay 阶段做"知识压缩",模型在 constant 阶段积累 raw signal,decay 阶段固化。

### 8.2 0.6B vs 0.09B 的对比

Falcon-H1-Tiny-R-0.6B(post-GRPO):

| Benchmark | 0.6B post-GRPO | 0.6B pre-GRPO | 0.09B | Qwen3-1.7B | DeepSeek-R1-Distill-1.5B |
|---|---|---|---|---|---|
| AIME24 pass@1 | **75.0** | 67.5 | 5.0 | 47.0 | 29.1 |
| AIME25 pass@1 | **67.3** | 60.0 | 7.9 | 37.0 | 23.4 |
| LCBv6 acc | **39.0** | 35.0 | 4.5 | 29.8 | 19.9 |
| MATH500 acc | **94.0** | 92.5 | 39.7 | 89.4 | 83.2 |

0.6B post-GRPO 全面超过 1.5B-1.7B reasoning model。**Test-time scaling 视角下更有意义**:pass@16 / maj@16 上,0.6B 接近 7B 模型水平,而 0.6B inference 成本远低 —— 这是 compute-equivalent 比较下小模型的真实优势。

### 8.3 0.09B 的失败模式 —— Repetition trap

paper 报告 0.09B 模型容易陷入 repetition loop,与 Pipis et al. (2025) "Wait, Wait, Wait... Why Do Reasoning Models Loop?" 的发现一致。

机制:小模型 capacity 不足以压缩 reasoning pattern,转而学到最简单的行为 —— 重复 token。repetition penalty 能 partially 缓解,但根本原因还是 capacity。

参考: https://arxiv.org/abs/2512.12895

### 8.4 GRPO 阶段

Group Relative Policy Optimization(Shao et al., 2024):

$$
\mathcal{L}_{GRPO} = -\mathbb{E}\left[\frac{1}{|G|}\sum_{i=1}^{G} \frac{A_i - \bar{A}}{\sigma_A} \log \pi_\theta(y_i | x) - \beta \text{KL}(\pi_\theta || \pi_{ref})\right]
$$

变量:
- $G$ = group size(同一 prompt 采样的 response 数)
- $A_i$ = 第 i 个 response 的 reward(通常 rule-based,如 math 正确性)
- $\bar{A}, \sigma_A$ = group 内的 mean / std,用来做 advantage normalization
- $\beta$ = KL penalty 系数
- $\pi_{ref}$ = reference policy(SFT 模型)

GRPO 相对 PPO 的核心简化:**用 group statistics 代替 value network**,不需要 critic。paper 中只跑 60 steps GRPO,response context 32k tokens。

关键超参:**LR 极度敏感**。太低收敛慢,太高 entropy 爆炸。最终选 LR=3e-6。

副作用观察:GRPO 让 mean response length 从 ~16k tokens 降到 ~8k tokens —— 模型学会"短而准"地表达 reasoning。

参考: https://arxiv.org/abs/2402.03300 (DeepSeekMath/GRPO)

---

## 九、Falcon-H1-Tiny-Function-Calling

### 9.1 数据覆盖

8 种 pattern:single-turn、multi-turn、sequential、parallel、relevance、irrelevance、schema diversity、edge cases。这覆盖了实际 tool calling 场景的大部分长尾。

### 9.2 关键失败模式 —— CoT 数据 toxic

把 chain-of-thought traces 混入 tool calling training data → **模型进入 infinite generation loop**,重复 token。

解决:**过滤掉所有 reasoning/thinking content,只保留 direct tool calling examples**。模型立刻能产出干净的 function calls。

直觉(与 0.09B repetition trap 一致):CoT 是复杂 long-form pattern,90M 容量不足以压缩。模型 fall back 到最简单的 local pattern = repetition。所以对小模型而言,**结构化输出 > 复杂中间推理**。

### 9.3 Curriculum vs Anti-curriculum 在 tool calling 上打平

与 English chat 不同,tool calling 上两种策略结果几乎相同。paper 给的 interpretation:**tool calling 能力上限由 model capacity 决定,而非 exposure timing**。JSON 语法、schema matching、parameter extraction 是 hard skill,更多训练时间不会改变模型能学到的最大特征 complexity。

### 9.4 Tool calling % scaling

| Tool calling % | BFCL v3 global |
|---|---|
| 20% | 32.1% |
| 50% | 36.8% |
| 75% | 39.4% |
| **85%** | **41.2%** |

monotonic 上升,有 diminishing returns。

### 9.5 与 FunctionGemma 270M 对比

| Model | Size | Non-LiveAST | LiveAST | Relevance | Irrelevance | Multi-turn | Global |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 600M | 71.79 | 56.62 | 75.00 | 80.84 | 3.62 | 57.57 |
| FunctionGemma | 270M | 48.40 | 26.40 | 61.10 | 70.60 | 0 | 41.30 |
| **Falcon-H1-Tiny-Tool-Calling-90M** | **90M** | 36.06 | 14.27 | **94.44** | 61.37 | 0 | **41.23** |

90M 在 global score 上与 270M FunctionGemma 打平(3x 参数效率)。**Relevance detection 大幅领先**(94.44 vs 61.10)—— 模型更知道"何时调用工具"。但 AST accuracy 落后 —— 模型在精确 schema 生成上受限于 capacity。

直觉:90M 模型 capacity 应花在"决策"而非"细节生成"。这是一个有趣的 inductive bias 实证 —— 小模型在判断类任务上比生成类任务表现好。

---

## 十、Falcon-H1-Tiny-Coder

### 10.1 FIM format

采用 PSM(Prefix-Suffix-Middle)format:

$$
\text{prompt} = \texttt{<|prefix|>} + \text{prefix} + \texttt{<|suffix|>} + \text{suffix} + \texttt{<|middle|>}
$$

模型在 `<|middle|>` 后生成中间内容。三个 special token 替换 tokenizer 中 reserved token。

### 10.2 Structure-Aware FIM data generation

参考 Gong et al. (2025),FIM 样本不是从代码随机 span 切割,而是按 AST 结构切分,使得 prefix/suffix/middle 是语法完整的单元(如完整 statement、function body)。同时混入 random-split 样本以增强 robustness。

参考: https://arxiv.org/abs/2506.00204

### 10.3 关键 debugging 经历 —— Indentation 问题

最初 HumanEval-FIM signal 全无。vibe check 发现:
- 模型在新行 prompt 后会预测一个全新 function,而不是续写
- 在 indented prompt 下能正确续写

问题:HumanEval-FIM 的 prefix 不包含 indentation,模型默认 reset。解决:在 FIM 数据中构造 50% 样本让模型必须预测 indentation,另 50% 把 indentation 放在 prefix 之后。这反映了一个工程教训 —— **tokenizer 不感知语法结构,模型只能从训练数据分布中学习格式 prior**。

### 10.4 Masking ablation

| Setting | FIM | Code gen |
|---|---|---|
| Mask non-FIM tokens(prompt 部分不算 loss) | lower | lower |
| **No masking**(全 token 都算 loss) | **better** | **better** |

直觉:un-masked 数据相当于让模型同时在学 FIM 和 next-token prediction,double effective signal per token。**给定 token + compute budget,un-masked 数据更高效**。

### 10.5 Dropout 在极端 data repetition 下的作用

数据 mixture:80% FIM + 20%(FineWeb-EDU + math)。FIM 数据相对少,需要重复 ~40x。比较 dropout=0 vs dropout=0.1(所有 linear projection 后):

| Setting | HumanEval-FIM | HumanEval / MBPP |
|---|---|---|
| No dropout | degraded(aggressive memorization) | higher |
| Dropout=0.1 | **stable / better** | slightly lower |

Dropout 在极端数据重复场景下作为 regularization 防止 memorization,代价是 code generation 略降。最终选 dropout 版本作为发布 checkpoint。

直觉:经典 deep learning regularization(dropout)在 LLM 极端 multi-epoch 训练中重新变得重要。这是个很有 deep learning fundamentals 味道的发现 —— 当 scaling laws 让 single-epoch 成为 default 时,dropout 被遗忘;但回到 multi-epoch tiny model regime,dropout 重新工作。

### 10.6 与 Qwen2.5-Coder-0.5B 对比

| Benchmark | Falcon-H1-Tiny-Coder-90M | Qwen2.5-Coder-0.5B |
|---|---|---|
| HumanEval+ @1 | 14.63 | 23.17 |
| HumanEval @1 | 16.46 | 27.44 |
| MBPP @1 | 41.26 | 54.76 |
| MBPP+ @1 | 34.92 | 48.67 |
| HumanEval-FIM @1 | 22.66 | 72.95 |
| HumanEval-FIM @10 | 40.75 | 91.38 |
| **HumanEval-FIM-RS @1** | **30.96** | 31.76 |
| **HumanEval-FIM-RS @10** | **56.7** | 56.7 |

90M 在 standard code generation 上明显落后 0.5B,但在 **random-span FIM** 上居然打平 —— 这暗示 Structure-Aware FIM + random-split mixing 让小模型在 random completion 场景下学到了更鲁棒的 inductive bias,弥补了 capacity gap。

参考: https://arxiv.org/abs/2409.12186 (Qwen2.5-Coder), https://arxiv.org/abs/2207.14255 (FIM origin)

---

## 十一、可 inference 的工具链

llama.cpp、ollama、MLX、transformers、vLLM、sglang 全部支持。Q8_0 量化后 ~90MB footprint,可在 MacBook 上流畅跑。

```bash
ollama run hf.co/tiiuae/Falcon-H1-Tiny-90M-Instruct-GGUF:Q8_0
```

代码补全用 Continue + llama-server 配置 FIM template:

```
autocomplete: <|prefix|>{{{prefix}}}<|suffix|>{{{suffix}}}<|middle|>
```

---

## 十二、核心 takeaway 与直觉

1. **Memorization window 是 multi-epoch 训练的真正瓶颈**。$p \le D/M$ 给出 HQ fraction 上界,与 $T$ 解耦。这把"小模型 + 大量 HQ 数据"从理论可行变成工程可行。
2. **小模型 anti-curriculum 优于 curriculum,但仅当 model 有足够 capacity 吸收 SFT pattern**。multilingual、tool calling 等 capacity-bound 任务上 anti-curriculum 失去优势。
3. **LRM + Muon 是 tiny model 上的强 leverage**。LRM 解放 row/column norm,跳出 noise-WD equilibrium,带来 20% 相对 gain。
4. **Tiny model 在 capacity-bound 的复杂任务(如 CoT、精确 schema)上会 fall back 到 repetition loop**。这是 capacity 的 hard limit,需要从数据/架构层面绕开,而非数据更多/更长训练解决。
5. **Reasoning pretraining 直接对 reasoning traces pretrain 远胜 pretrain-then-SFT**。这是 small model regime 的特性 —— 更长 optimization horizon 对低 capacity 模型更重要。
6. **Test-time scaling 让 0.6B reasoning 模型在 pass@16 等价于 7B**。这是 paper 的 strong claim —— small reasoning model 在 compute-equivalent 比较下是 viable deployment option。
7. **Dropout 在 extreme data repetition 下重新有用**。这是对 scaling law 时代"dropout 无用论"的回拨,提示小模型 regime 与大模型 regime 在 regularization 需求上不同。

---

## 十三、可能的研究方向联想

基于这篇 paper 的发现,几个直觉上 promising 的方向:

1. **Memorization window 的精确测量**。当前 estimate 来自 FalconMambi-7B 的 linear scaling,但 memorization window 与模型架构(SSM vs Attention)、训练数据 distribution 应该有 strong interaction。需要 per-architecture 的 systematic 测量。

2. **LRM 与 LoRA 的关系**。LRM 给每行每列一个 scalar multiplier,本质上是低秩 adapter 的 rank-1 特例。值得探索 LRM 与 full LoRA / IA3 / BitFit 等 parameter-efficient fine-tuning 方法的理论联系。LRM paper: https://arxiv.org/abs/2601.04890

3. **Tiny model 的 repetition trap 的 architecture-level solution**。可能与 attention head 的 entropy regularization、SSM 状态的 reset 机制相关。Pipis et al. 的论文给了现象,机制仍 open: https://arxiv.org/abs/2512.12895

4. **Anti-curriculum 在中等 scale(500M-1B)的最优 SFT fraction**。在 100M 时 25% 是 sweet spot,在 1B 时应该下降 —— 与 memorization window 的 scaling 一致。

5. **Model merging for tiny models**。paper future work 中提到,如果 anti-curriculum 让 tiny model 学到 specialization,那么 merge 多个 specialized tiny model 是否能逼近 general-purpose model?这与 task arithmetic、TIES merging 等技术可以桥接。

6. **GRPO 的 LR sensitivity 与 Muon 的相互作用**。Muon 对 GRPO policy gradient 的 spectral normalization 可能解释稳定性差异 —— paper 没在这条线做 ablation。

7. **Multilingual 的 capacity wall 的实证测量**。paper 用 17 种语言打平 anti-curriculum 与 curriculum,但没有 ablation 减少 language 数(比如 3 种)看 anti-curriculum 是否重新变得 better。这能精确定位是 language 数还是绝对 capacity 是瓶颈。

---

## 十四、相关 references 汇总

| Topic | Paper | Link |
|---|---|---|
| Falcon-H1 主架构 | Zuo et al., 2025 | https://arxiv.org/abs/2507.22448 |
| Learnable Multipliers | Velikanov et al., 2026 | https://arxiv.org/abs/2601.04890 |
| Muon optimizer | Liu et al., 2025 | https://arxiv.org/abs/2502.16982 |
| Falcon-H1R reasoning | Falcon-LLM Team, 2026 | https://arxiv.org/abs/2601.02346 |
| SmolLM2 baseline | Allal et al., 2025 | https://arxiv.org/abs/2502.02737 |
| Mobile-LLM-R1 | Zhao et al., 2025 | https://arxiv.org/abs/2509.24945 |
| Qwen3 baseline | Yang et al., 2025 | https://arxiv.org/abs/2505.09388 |
| Reasoning loop | Pipis et al., 2025 | https://arxiv.org/abs/2512.12895 |
| GRPO/DeepSeekMath | Shao et al., 2024 | https://arxiv.org/abs/2402.03300 |
| IFEval | Zhou et al., 2023 | https://arxiv.org/abs/2311.07911 |
| FIM origin | Bavarian et al., 2022 | https://arxiv.org/abs/2207.14255 |
| Qwen2.5-Coder | Hui et al., 2024 | https://arxiv.org/abs/2409.12186 |
| Structure-Aware FIM | Gong et al., 2025 | https://arxiv.org/abs/2506.00204 |
| LiveBench | White et al., 2025 | https://arxiv.org/abs/2410.12370 (近似) |
| MMLU-Pro | Wang et al., 2024 | https://arxiv.org/abs/2406.01574 |
| OLMo 3 Zero-RL | Olmo Team, 2025 | https://arxiv.org/abs/2512.13961 |
| AlpacaEval | Li et al., 2023 | https://github.com/tatsu-lab/alpaca_eval |
| MT-Bench | Bai et al., 2024 | https://arxiv.org/abs/2402.14681 (近似) |

---

这篇 paper 最大的价值在于:**它把 small-scale LLM 训练从"压缩大模型 recipe"转向"针对 tiny scale 重新设计 data strategy、optimization、architecture"**。memorization-aware repetition 给了一个 quantitative framework 来推理 multi-epoch 训练的安全性,LRM 给了跳出 weight norm equilibrium 的方法,anti-curriculum 给了打破 pretrain-SFT 两阶段 paradigm 的实证。这些 insights 在 100M 上验证后,理论上可以反向 push 到中等规模(1B-7B),可能在 medium scale 上重新产生 significant gains。

对教学/直觉而言,最值得 internalize 的两点:
- **数据 fraction 上界不是 D/T 而是 D/M** —— 把 memorization window 当成一阶变量
- **CoT 对 tiny model 是 toxic** —— capacity 不足以压缩时,模型 fall back 到 repetition,这告诉你 CoT 的 benefit 在某个 capacity threshold 以下消失
