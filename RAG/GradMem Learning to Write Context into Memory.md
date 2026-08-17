---
source_pdf: GradMem Learning to Write Context into Memory.pdf
paper_sha256: 41efbfa70764a11e050511be6b96d2dd50e99f6be0cf05ec0b60d59580f9f47b
processed_at: '2026-08-04T22:19:37-07:00'
target_folder: RAG
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GradMem 人话版

## 这 paper 在解决什么问题

想象你在用 LLM 处理一份 100 页的合同。每个问题(query)你都得让模型从头读一遍这 100 页,因为 attention 需要看见全部 token。KV-cache 帮你省了重计算,但每个 token 每个 layer 的 activation 都得挂在显存里——100 页就是天文数字。

另一条路:让模型**读一遍 context,把它压缩成一小撮 memory token,以后所有 query 都只看这撮 memory**。这就是 context removal setting,很纯粹——压缩好不好一目了然。

GradMem 就是干这个的。它的核心 thesis 其实特别简单:**与其让模型 forward 一遍"猜"该往 memory 里写什么,不如直接对 memory 跑几步 gradient descent,让 loss 告诉你哪里还没写好。**

## 为什么这件事有意思

你想往一个 8 个 vector 的 memory 里塞 96 对 key-value。

**Forward-only 写法** (RMT 那种):让模型跑一遍 context,它在前向过程中输出 8 个 vector 就完事了。问题是模型一旦 emit 这 8 个 vector,就没法回头——它不知道自己写错没写错、漏没漏东西。实测 8 对 KV 能存,16 对就崩到 45% accuracy。8 个 vector 硬容量就是这么大。

**GradMem 写法**:同样是这 8 个 vector,但当作"可训练参数",对 reconstruction loss 跑 5 步 GD。结果 96 对 KV 还能存 88%。同样大小的 memory,容量涨了十几倍。

这个 gap 很反直觉——你怎么可能往 8 个 vector 里塞 96 对 KV?答案在于:**gradient signal 告诉 memory "你哪里还没编码好"**,每一步 GD 都在补窟窿。而 forward pass 没有这个 feedback 通道。

## 核心机制三句话

1. **Memory 是 prefix embeddings**:8 个 (或 32 个) soft token prepended 在 input 前面,跟 prefix-tuning 一样的 parameterization。

2. **WRITE loss 是 context reconstruction**:拿 memory 当 prefix,让模型预测 context 里的每个 token。如果 memory 里有信息,预测就准;没信息就只能靠 context 前缀的局部统计。所以最小化这个 loss 等于逼 memory 把 "context 里从前缀猜不出来的部分" 存进去——天然优先存 high-entropy / novel / 出乎意料的内容。

3. **Meta-learning 让 few-step 成立**:已知几千步 GD 能 lossless 把 1568 tokens 塞进一个 vector (Kuratov et al. 2025),但几千步太贵。GradMem 的 trick 是 meta-learn 一个好的 initialization $\mathcal{M}_0$,让 GD 从这点出发跑 1-5 步就到有用的 memory。代价是训练时要 backprop 透过 unrolled GD——二阶梯度,工程上很痛。

## 为什么 work 的直觉

这个 paper 最值得 internalize 的 insight:**SGD 本质上就是一个 "把数据写进参数" 的机制**。

平时训练:你有一个 model,有一堆数据,SGD 跑几万步,数据被 "写" 进 model weights 里。这是 global memorization——所有样本共享一份参数。

GradMem:你有一个 memory state,有一个 context,SGD 跑 5 步,这个 context 被 "写" 进 memory 里。这是 local memorization——每个 context 有自己的 memory state,model weights 保持 frozen。

关键差别:**loss 是 per-example 的**。Forward-only write 是 "盲写"——模型 emit memory 时不知道对不对。GD write 是 "看着 loss 写"——每一步 gradient 都指向 "哪里还缺信息"。Figure 4(b) 有个很漂亮的证据:在 KV-retrieval 上,他们把 reconstruction loss 拆成 key tokens vs value tokens,发现 **value loss 显著下降而 key loss 几乎不动**。说明 memory 学会了选择性——把容量优先给"必须输出的 value",key 只作为 retrieval cue 让 query 触发就行。Reconstruction objective 表面对所有 token 一视同仁,但 meta-learning 之后 model 自己学到了该往哪分配容量。

## 实验说了啥

KV-retrieval (合成 benchmark) 上的核心数:

- Forward-only write (RMT, 8 mem vectors): 8 pairs 100%, 16 pairs 45%, 96 pairs 13%
- GradMem K=1 (同样 8 vectors): 16 pairs 96%, 96 pairs 33%
- GradMem K=5: 96 pairs 88%
- 重复 forward write 2-5 次几乎没用,而且 variance 巨大

这证明:在同样 memory size 下,gradient write 比 forward write 存得多;而且 gradient steps 能 scale capacity,forward passes 重复跑没用。

NLP transfer 上 (GPT-2 124M 做 base):
- Short SQuAD: GradMem K=1 54.9 vs RMT 42.6 vs ARMT 39.0,距离 full-context GPT-2 的 64.2 不远
- bAbI QA2/QA3 信息密集的任务上,GradMem 跟 RMT 持平或略好;Mamba 因为预训练时已经学过 recurrent state 操作所以略强
- LM on WikiText: 跟 RMT/Mamba 同档

Transfer 成立说明 reconstruction objective 是 task-agnostic 的——你不用为每个下游 task 重新设计 WRITE loss。

## 什么时候划算

工程上 GradMem 有个前置 cost: WRITE 阶段要跑 K 步 GD,比单次 forward 贵 K 倍 (大概 K=1 时 R ≈ 3-5 倍)。但 WRITE 只跑一次。之后每次 READ 只 attend 几个 memory token,不用看整个 context。

Break-even 条件大致是:$N \gtrsim c(RK - 1) / q$,其中 N 是同一个 context 上要问多少 query、c 是 context 长度、q 是 query 长度、K 是 GD 步数。

Figure 7 实测:A100 上,context=1024 时大约跑 64 个 query GradMem 总延迟反超 GPT-2 with KV-cache。对 agent 反复 query 同一 codebase / long doc 这种场景,正好对口。

## 局限

1. **二阶梯度训练贵得离谱**。你得 backprop 透过 K 步 GD,等于把 K 步 forward 的 computation graph 全留住。FlashAttention / SDPA 都不支持高阶微分,作者自己写了个 manual double-backward kernel 才勉强 scale 到 L=1024。这是 GradMem 上不了 long context / 大 model 的主因。

2. **Reconstruction loss 可能不是最优 WRITE objective**。它对所有 context token 一视同仁,但下游 task 可能只关心一小部分信息。Figure 4(b) 显示 meta-learning 已经 implicit 学到 selectivity,但显式 design task-aware self-supervised objective 是明显的改进方向。

3. **没 scale 到大 model / 真 long context**。实验最大 context ~1024 tokens,最大 base model 是 GPT-2 124M / Pythia 160M。7B+ 模型 + 100k context 能不能 work 是 open question——主要受 double-backward 工程瓶颈限制。Paper 提了 implicit differentiation (iMAML) 和 Reptile 作为 future work。

4. **$K_{\text{train}}$ vs $K_{\text{eval}}$ 的 extrapolation 现象**说明 inner loss 没真正收敛。Figure 4(a) 显示在 inference 时把 K 调大还能涨点,暗示 meta-learned init 还没充分利用。但 $K_{\text{train}}$ 大又跟 (1) 冲突——unroll 长 backprop 爆炸。这是根本的 tension。

## 一句话总结

GradMem 的贡献是把 test-time training 从 per-layer / per-token 的 online 更新拉回到 model-level / per-context 的 prefix memory,用 meta-learning 把 GD 步数从几千压到 5 以内,并在严格 context removal 设定下证明 **gradient write 在同 memory size 下显著优于 forward write,且能 scale with test-time compute**。核心 insight 是把 SGD 当 "data writing mechanism" 用在 per-example memory 上——loss 天然提供 forward pass 给不了的 per-example feedback 信号。代价是二阶梯度训练的工程痛。

跟 Titans、Atlas、TTT layers 是同一波趋势的不同 parameterization:都在探 "如何用 test-time gradient 把 context 压成可复用 state"。GradMem 选了最纯粹的设定 (model-level, reconstruction, few-step meta-learned),换来了干净的 ablation 和反直觉的容量提升证据。

---

# GradMem: Test-Time Gradient Descent 写入 Memory 的深入解析

## 1. 核心定位与动机

GradMem 处于一个很特定的 corner case：**context removal setting**。即把 inference 拆成两阶段——WRITE 阶段把 context C 压缩到固定大小 memory $\mathcal{M}$，READ 阶段仅用 $\mathcal{M}$ 加上 query Q 预测 Y，**完全屏蔽 C**。这把"压缩好不好"这件事彻底隔离出来测，避免了 KV-cache 之类方法的"作弊"——只是省重计算但仍然挂着全部 activations。

核心 thesis 用一句话讲：**forward-only write 是一次性的、无反馈的"硬编码"，gradient-based write 是有 per-example feedback 的"软优化"**。论文用实证证明：在同样 memory size 下，哪怕只跑 1 步 gradient descent，也比一次 forward pass 写入得更多；多跑几步 (K≤5) 能让 8 个 memory vector 存 96 对 KV。同样重复 forward-only write (x2–x5) 收益很弱且不稳定——这是关键反直觉发现。

参考链接：
- TTT layers: https://openreview.net/forum?id=wXfuOj9C7L  
- Titans: https://arxiv.org/abs/2501.00663  
- Cartridges: https://arxiv.org/abs/2506.06266  
- Kuratov et al. cramming 1568 tokens: https://aclanthology.org/2025.acl-long.948/

## 2. 问题形式化

每个样本拆成 $(C, Q, Y)$ 三段。标准 causal LM 是 $f_\theta(Y \mid [C;Q])$，每次 query 都要重新 attend 全 context。GradMem 引入 memory $\mathcal{M} \in \mathbb{R}^{m \times d}$，定义两个 phase：

**WRITE**：$\mathcal{M} = \mathcal{E}_\theta(C)$，把 context 编码进 memory。  
**READ**：$f_\theta(Y \mid [\mathcal{M};Q])$，用 memory 和 query 预测。

Memory 用 $m$ 个 $d$ 维向量做 **prefix embeddings**，即 prepended 到 input 前面的 learned soft tokens。这点跟 prefix-tuning / RMT 是一致的，只是 update rule 不同。

## 3. WRITE objective：context reconstruction

公式 (4) 是核心：

$$\mathcal{L}_{\text{write}}(\mathcal{M}; C) = -\sum_{i=1}^{N} \log f_\theta(t_i \mid [\mathcal{M}; t_{<i}])$$

变量解释：
- $C = (t_1, \dots, t_N)$ 是 context 的 N 个 token
- $\mathcal{M}$ 是当前 memory state（prefix）
- $t_{<i} = (t_1, \dots, t_{i-1})$ 是第 i 个 token 前的 context prefix
- $f_\theta(t_i \mid \cdot)$ 是 LM 给出的 next-token 概率
- 求和 $\sum_{i=1}^{N}$ 跑遍整个 context

**直觉**：如果 $\mathcal{M}$ 里有关于 C 的信息，那么模型以 $\mathcal{M}$ 为 prefix、再加上 C 的前缀 $t_{<i}$，就能更好地预测 $t_i$。如果 $\mathcal{M}$ 是空/随机，预测只能依赖 $t_{<i}$ 的局部统计。所以最小化 reconstruction loss 等价于**逼着 memory 把 context 中"不可由前缀推断"的信息存进去**——天然 high-entropy、novel、surprising 的部分。这一点跟 information bottleneck 的思想吻合。

注意这个 loss 是 **task-agnostic** 的，不依赖下游 supervision，所以可以 transfer 到不同 task（bAbI, SQuAD, LM）。这是关键设计——他们没让 WRITE loss 知道 query 是什么，否则就退化成 RAG 之类的 supervised compression。

## 4. UPDATE rule：test-time gradient descent on memory

公式 (5)：

$$\mathcal{M}_{k+1} = \mathcal{M}_k - \alpha \nabla_{\mathcal{M}_k} \mathcal{L}_{\text{write}}(\mathcal{M}_k; C)$$

变量：
- $k$ 是 WRITE step 索引，$k = 0, 1, \dots, K-1$
- $\alpha$ 是 inner-loop learning rate（论文调到 $\alpha = 0.4$ 是 NLP task 上的 default）
- $\nabla_{\mathcal{M}_k}$ 只对 memory 求梯度，**模型参数 θ frozen**
- 起点 $\mathcal{M}_0$ 是 meta-learned initialization（所有样本共享）

公式 (6) 把 K 步 GD 写成 encoder 算子：

$$\hat{\mathcal{M}} = \mathcal{E}_\theta(C) \triangleq \text{GD}_K(\mathcal{M}_0, \mathcal{L}_{\text{write}}(\cdot; C))$$

直觉上：把 SGD 训练当成"把数据写进参数"的机制，这里 memory 就是 parameter-like state。区别于 standard training 把整个 train set 写进 θ（global memorization），这里把**单个 context 写进 per-example memory state**（local memorization）。

## 5. META-LEARNING：MAML-style 双层优化

这是论文最 nontrivial 的部分。Outer loss 是下游 task loss：

公式 (8)：$\mathcal{L}_{\text{task}}(\hat{\mathcal{M}}, Q, Y) = -\log f_\theta(Y \mid [\hat{\mathcal{M}}; Q])$

训练时同时对 $\theta$ 和 $\mathcal{M}_0$ 求导，**必须 backprop 透过 unrolled K 步 WRITE**——也就是二阶梯度（MAML 风格，Finn et al. 2017, https://proceedings.mlr.press/v70/finn17a.html）。

- **Inner loop**：对每个样本跑 K 步 GD on $\mathcal{M}$，得到 $\hat{\mathcal{M}}$
- **Outer loop**：用 $\hat{\mathcal{M}}$ 在 READ phase 计算 task loss，梯度反传到 $\theta$ 和 $\mathcal{M}_0$，让"K 步 GD 起点好、模型会用 GD 写信息"

直觉：meta-learning 让 $\mathcal{M}_0$ 处于一个"loss landscape 上跑几步就能到好位置"的 starting point。没有 meta-learning 的实验（Table 1 "w/o meta-learning"）在 8 pairs 上只有 12.9% 准确率，说明 first-order approximation 远远不够，**second-order signal 必须存在**。

## 6. 跟 TTT layers 的精确对比

论文 Table 3 给了一个很清晰的对比：

| 维度 | TTT layers (Sun et al. 2025) | GradMem |
|---|---|---|
| Usage pattern | sequence-modeling layer，online per-token update | 显式 WRITE/READ 两 phase |
| Inner-loop input | 单个 token $x_t$ 或 mini-batch | 整个 context segment $C$，once per context |
| Test-time parameters | per-layer $W_t$，从 layer input/activation 学 | 单个 prefix memory $\mathcal{M}_k \in \mathbb{R}^{m\times d}$，model-level |
| Self-supervised loss | activation/input reconstruction，e.g. $\|f(\tilde{x}_t;W) - x_t\|_2^2$ | **context token reconstruction** $\mathcal{L}_{\text{write}}$ |
| Outer-loop objective | next-token prediction (LM training) | downstream task loss with C removed at READ |
| Outer-loop parameters | $\theta$ + reconstruction task/view params | $\theta$ + memory init $\mathcal{M}_0$ |

直觉总结：TTT 是"在每个 layer 里塞一个 online 学习的小模型"，GradMem 是"在 model input 处塞一个一次性写完的 memory state"。前者更像"per-layer 的 fast weight"，后者更像"per-context 的 prefix-tuning + test-time SGD"。

## 7. KV-retrieval 实验解读（Table 1）

Associative KV-retrieval 是干净的合成 benchmark。Context 形如：
$$C = :k_1:v_1:!k_2:v_2!\cdots:!k_N:v_N!$$
Query $Q = ?!k_j!$，target $Y = v_j$。模型必须把 key→value mapping 写进 memory 才能答对。

主要数字（accuracy, mean±std）：

| Model | 8 pairs | 16 | 32 | 64 | 96 |
|---|---|---|---|---|---|
| Transformer KV-cache (upper bound) | 100.0 | 100.0 | 99.8 | 96.5 | 98.8 |
| Mamba (per-layer state) | 98.9 | 98.7 | 90.2 | 95.2 | 92.2 |
| ARMT (per-layer associative) | 98.5 | 97.4 | 54.9 | 22.6 | 15.2 |
| RMT forward-only, m=8 | 100.0 | 45.5 | 44.3 | 19.3 | 12.9 |
| RMT x2 forward writes | | 69.6±28.1 | 18.7±3.4 | | |
| RMT x3 forward writes | | 60.0±42.0 | 38.1 | | |
| GradMem K=1, m=8 | 99.7 | 96.3 | 86.9 | 58.6 | 32.6 |
| GradMem K=2 | 100.0 | 99.6 | 98.3 | 72.8 | 34.2 |
| GradMem K=5 | 100.0 | 100.0 | 99.9 | 99.1 | 88.4 |
| GradMem K=1 w/o meta | | 12.9±8.1 | 3.0 | | |

**几个直觉**：

1. **Forward-only write 在 16 pairs 就崩了**（45.5%），而 KV-cache 100%。说明 8 个 memory vector 真的塞不下 16 对 KV——这是 forward-only 的硬容量上限。

2. **重复 forward-only write 收益弱且不稳定**：x2 在 16 pairs 上跳到 69.6 但 std 是 28.1，x3 反而掉到 60.0 std 42.0。这说明"再看一遍同样的 context"对 memory 提升非常有限——forward pass 没有"哪里写错了"的信号。

3. **GradMem K=1 就吊打 forward-only**：96.3% @ 16 vs 45.5%，gap 巨大。这表明**单步 GD 提供的 per-example feedback 信号本身**就比一次 forward 强很多。

4. **K 越大容量越大**：K=5 能在 96 pairs 上 88.4% accuracy。同样的 8 个 memory vector，多跑 4 步 GD 容量涨了十几倍。这是 test-time compute 换 capacity 的直接证据。

5. **w/o meta-learning 几乎不工作**：8 pairs 只有 12.9%。证明"few-step regime"靠的不是 GD 本身的魔力，是 meta-learned initialization 让 GD 在 1-5 步内能收敛到有用的 memory。

## 8. Figure 4：value reconstruction loss 比 key 降得更猛

这个细节很有意思。在 96-pair KV-retrieval 上，把 inner loss 拆成 key tokens vs value tokens 分别画曲线：
- **Key reconstruction loss 几乎不降**——memory 不太能预测下一个 key 是什么
- **Value reconstruction loss 显著下降**——memory 越来越能预测 value

直觉：**memory 是选择性的**，它把容量优先分配给"必须输出"的 value，而 key 只需要作为"retrieval cue"被外部 query 触发即可。这说明 reconstruction objective 表面看着 symmetric（对 context 所有 token 一视同仁），但经过 meta-learning 之后模型学会了**把容量放在 task-relevant 的部分**。这跟 information bottleneck / 计算图上的 implicit gradient bias 很相关。

## 9. Table 2：NLP tasks 上的 transfer

用 GPT-2 (124M) 作 base model：

| Model | QA1 | QA2 | QA3 (~300 tok) | QA4 | QA5 | Short SQuAD | LM CE↓ |
|---|---|---|---|---|---|---|---|
| GPT-2 full ctx (upper) | 100 | 100 | 99.8 | 100 | 99.4 | 64.2 | 2.72 |
| GPT-2 ctx=128 (limit) | 100 | 99.7 | 95.5 | 100 | 99.0 | 48.9 | 3.20 |
| Mamba-130m | 100 | 100 | 96.7 | 100 | 99.7 | 63.3 | 2.84 |
| RMT (GPT-2) | 100 | 93.9 | 87.9 | 100 | 93.9 | 42.6 | 2.69 |
| ARMT (GPT-2) | 100 | 93.8 | 92.3 | 100 | 98.9 | 39.0 | 2.85 |
| GradMem K=1 | 100 | 93.9 | 79.3 | 100 | 99.2 | 54.9 | 2.92 |
| GradMem increased K | 100 | 94.2 | 80.0 | 100 | 99.2 | (↑) | (↓) |

**直觉**：
- QA1/QA4/QA5 context 很短（~20-40 tokens），大家都接近满分，不区分方法
- QA2/QA3 信息密度更高。Mamba 强是因为它在 pretraining 时就学过 recurrent state 操作，而 GradMem 和 RMT 是从 GPT-2 重新 fine-tune 学 memory operation 的
- QA3 是 GradMem 唯一明显输给 RMT 的：~300 tokens 信息密集，K=1 还不够；可以推测加 K 会改善但论文没具体报
- **Short SQuAD 上 GradMem K=1 (54.9) 显著超过 RMT (42.6) 和 ARMT (39.0)**，距离 GPT-2 full ctx (64.2) 不远。这是 transfer 的关键证据：reconstruction WRITE objective 在自然语言 QA 上 work
- LM 任务：GradMem K=1 CE 2.92，跟 RMT 2.69、Mamba 2.84 同档。RMT 略好可能因为 LM 是 RMT 训练目标本身（next-token prediction on segment）

## 10. Compute efficiency: 什么时候 GradMem 划算

公式 (9)–(11) 是工程上的核心问题。记 context 长度 $c$，query 长度 $q$，memory token 数 $m$，每 context 的 query 数 $N$，gradient step 数 $K$，$R$ 是单步 memory update 相对单次 forward 的 cost ratio。

- Full-context transformer (with KV-cache reuse): $T_{\text{full}} \approx c^2 + cqN$（一次性处理 $c$ + 每 query $cq$ 的 cross-attention）
- GradMem: $T_{\text{GradMem}} \approx R(c+m)^2 K + m^2 + mqN$（WRITE 一次性 + 每 query 只 attend $m$ 个 memory）

**Break-even**（公式 11）：
$$N > \frac{c^2(RK - 1) + (1+RK)m^2 + 2cmRK}{q(c - m)}$$

简化启发式：$N \gtrsim c(RK-1)/q$，即"每 context 至少被 query 这么多次才划算"。

Figure 6 画了 break-even 曲线，$q=128, m=32$。Figure 7 是实测 latency：
- context=64：GradMem 几乎总是赢
- context=256 / 1024：在 N≈64 次 READ 后 GradMem 反超 GPT-2 with KV-cache
- Mamba 在所有 context length 都被 GradMem 击败

直觉：GradMem 是**前置 WRITE cost + 后续便宜 READ cost** 的 amortization 模型，适合 $c \gg m$ 且 $N$ 大的场景——比如 agent 对同一 codebase / long doc 反复 query，正好是 RAG / long-context 应用的高频场景。

## 11. 工程实现：double-backward 是真正的痛点

GradMem 训练需要在 inner K 步 GD 上做 backward，**即 backward of backward**——二阶导数。这把现有 attention kernel 全打废了：
- FlashAttention (https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conf.pdf) 不支持高阶微分
- PyTorch SDPA 也是 first-order only

论文 Appendix C 实现了几个版本：
- Eager (naive autograd)：基线
- Fast forward → manual backward：SDPA forward + 解析一阶 backward + autograd 推二阶
- Fast forward → autograd：backward 里 recompute forward，让 autograd 推二阶
- Manual HVP：纯解析 forward + backward + double backward
- **Flash HVP**：fused forward+backward kernel + 解析 double backward

在 L=1024、GPT-2、8 mem tokens、batch 16、K=1 上：
- Eager: ~1000ms backward, ~60GB peak memory
- 最佳的 Fast forward → autograd: ~600ms
- Manual HVP: 最省 memory ~30GB
- Flash HVP: 速度第二、memory 第二，**最 balanced**

这是 scaling GradMem 到 long context 的实际工程瓶颈——long K + long context 会让二阶图爆炸。

## 12. 局限与开放问题

1. **二阶梯度训练贵**：K 大时 backprop through unrolled GD cost 爆炸。论文提了 implicit differentiation (iMAML, https://proceedings.neurips.cc/paper_files/paper/2019/file/072b030ba126b2f4b2374f342be9ed44-Paper.pdf) 和 Reptile (https://arxiv.org/abs/1803.02999) 作为未来改进方向。

2. **Reconstruction objective 可能不最优**：对所有 context token 一视同仁，但下游 task 可能只关心一部分信息。Figure 4(b) 显示 meta-learning 已经 implicit 学到了 selectivity，但**显式 design task-aware self-supervised WRITE objective** 是开放方向。

3. **$K_{\text{train}}$ vs $K_{\text{eval}}$ 的 extrapolation**：Figure 4(a) 显示在 inference 时把 K 调大能涨点，说明 inner loss 没真正收敛到 basin。这暗示 meta-learned $\mathcal{M}_0$ 还没充分利用——理论上 $K_{\text{train}}$ 应该够大让 GD 真正收敛，但这跟 (1) 冲突。

4. **没 scale 到 long context / 大 model**：实验最大 context ~1024，base model 是 GPT-2 124M、Pythia 160M。能不能上到 100k context、7B+ model 未知——主要受 double-backward 工程瓶颈限制。

5. **跟 Titans/Atlas 的关系**：Titans (https://arxiv.org/abs/2501.00663) 是 per-layer 的 test-time memorize，Atlas (https://arxiv.org/abs/2505.23735) 是 test-time optimization 写入 associative memory。GradMem 是 model-level、reconstruction-based、显式 prefix memory。这三条线本质都在探"如何用 test-time gradient 把 context 压缩成 persistent state"，只是 parameterization 不同。

## 13. 给直觉的总结

GradMem 的核心 insight 可以浓缩成三层：

**(a) SGD = data writing mechanism**。训练时 SGD 把 train set 写进 θ；这里把单 context 写进 per-example memory state。memory 是 "local parameter"，θ 是 "global parameter"。

**(b) Loss 提供 per-example feedback**。Forward-only write 一旦 emit memory 就没法回头；gradient write 看 reconstruction loss 高在哪些 token 上，gradient 自然把"还没编码好"的信息往 memory 里塞。Figure 4(b) 的 value vs key loss 分化是这点的直接证据——loss signal 让 memory 学会选择性存 value。

**(c) Meta-learning 让 few-step regime 成立**。Kuratov et al. 2025 证明数千步 GD 能 losslessly 把 1568 tokens 塞进一个 vector。GradMem 用 meta-learned $\mathcal{M}_0$ 把这个数字压到 K≤5 步。代价是 outer training 要二阶梯度，工程上痛苦。

跟 attention-as-associative-memory（Hopfield, Ramsauer et al. 2020 https://openreview.net/forum?id=tL89R7xxr1）、跟 fast weights (Schmidhuber 1992)、跟 RMT/ARMT、跟 TTT layers、跟 Cartridges (https://arxiv.org/abs/2506.06266) 都在解决同一个根本问题——**如何让 model 在 inference 时把 context 压成可复用的 state**——GradMem 的贡献是证明了"显式 test-time optimization + few-step meta-learning" 在严格 context removal 设定下比 forward-only write 在同等 memory size 下能塞更多信息，并且能 scale with test-time compute。

最值得 follow 的方向我猜是：(i) 用 implicit differentiation 替代二阶 unroll 解掉训练 cost；(ii) 设计 task-aware 的 self-supervised WRITE objective（比如 masked/span reconstruction 而非 strict next-token reconstruction）；(iii) 在 7B+ model 和 long context 上验证能否 scale。
