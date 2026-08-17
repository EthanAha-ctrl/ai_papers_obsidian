---
source_pdf: Cramming 1568 Tokens into a Single Vector and Back Again.pdf
paper_sha256: 7ac99dbc6143c68930895ca63affaa78bc8a7bd6020d9b36a82547fc42b24c57
processed_at: '2026-08-03T17:43:07-07:00'
target_folder: LLM-engine/Torch
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 一句话版本

**把一个frozen LLM当成超大图书馆，[mem] vector当成索书号，用SGD去搜出能解码目标text的索书号——结果发现一个索书号能找回1568个token的原文，比传统encoder-based方法高两个数量级。**

paper: https://arxiv.org/abs/2505.20800

---

## 背景是什么困惑

LLM的input embedding维度一直在涨：Llama-1B是2048维，405B是16384维。一个2048维bfloat16 vector理论上能装32768 bits，按一个token 17 bits算，能装1931个token。**但实际我们一个vector就代表一个token，浪费得离谱**。

之前的context compression工作（AutoCompressors、ICAE、500xCompressor、Gisting、LLMLingua之类）试图把长text压成几个vector，lossless顶多x10。两个数量级的gap去哪了？这就是这篇paper想搞清楚的。

参考：
- AutoCompressors: https://arxiv.org/abs/2305.14788  
- ICAE: https://arxiv.org/abs/2305.14788  
- 500xCompressor: https://arxiv.org/abs/2408.03094  
- Gisting: https://arxiv.org/abs/2310.18558  
- LLMLingua: https://arxiv.org/abs/2310.05736  

---

## 方法的核心trick

**别训encoder了，用SGD代替encoder。**

具体怎么干：
1. LLM权重全冻结
2. 在text前面prepend一个或几个trainable vector $[\text{mem}]$
3. 让LM做next-token prediction，输入是 $[\text{mem}] + \text{前文}$，目标是预测下一个token
4. 用AdamW优化$[\text{mem}]$，直到teacher-forcing accuracy = 1.0
5. 训完，这个$[\text{mem}]$就是这段text的"压缩码"

跟Prompt Tuning（https://arxiv.org/abs/2104.08691）形式上一样，但用法相反：Prompt Tuning是固定task训练prompt适配下游任务，这里是固定text训练vector让LM能重建这个text。

**为什么这能测capacity上限？** 因为传统方法训一个encoder网络，encoder要泛化到没见过的text，generalization gap吃掉大量capacity。这里per-sample SGD没有泛化问题——每段text单独搜，搜到的就是"最优解附近"。所以测出来的是upper bound。

类比：传统方法是训练图书管理员对任何书都能找索书号；这paper直接绕过管理员，用SGD搜坐标，所以能测出图书馆真实容量。

---

## 理论上限公式

$$L \leq \frac{d_{model} \times b}{\log_2 |\mathcal{V}|}$$

人话翻译：
- $L$ = 最多能装几个token
- $d_{model}$ = vector维度（Llama-3.1-8B是4096）
- $b$ = 每个float多少bit（bfloat16就是16）
- $|\mathcal{V}|$ = vocab大小（Llama-3是128256）
- $\log_2|\mathcal{V}|$ ≈ 17，是每个token的worst case信息量

代入Llama-3.1-8B：$4096 \times 16 / 17 \approx 3862$ token理论上限。实际1568，utilization 40.7%。

为什么实际 < 理论？因为vector必须落在LLM input embedding manifold的"合法区域"，不是任意bit pattern都行。manifold的有效维度比ambient维度小很多。

---

## 三个metric，人话版

### Eq. (2) Decoding Capacity $L_{max}$

能perfect重建的最长text length。threshold用0.99。

直白说：**"这个vector能存多少个token？"**

### Eq. (3) Token Gain $C_{tokens}$

有$[\text{mem}]$时LM猜对的token数 减去 没$[\text{mem}]$时LM猜对的token数。

直白说：**"vector净贡献了多少token，扣除LM自己靠prior猜对的部分。"**

这很关键——LM对natural text能猜对很多token（比如"the cat sat on the ___"LM自己就能猜出"mat"），这部分不算vector的功劳。

### Eq. (4) Information Gain $C_H$

$$C_H = H_{LM} - H_{LM+[\text{mem}]}$$

- $H_{LM}$ = 没memory时text的cross-entropy
- $H_{LM+[\text{mem}]}$ = 有memory时text的cross-entropy
- $C_H$ = vector减少了多少不确定性（bits）

直白说：**"vector存了多少bit的信息。"**

这是最干净的metric，因为它完全用information theory语言，与text长度、text类型、vocab都无关，**几乎只跟model本身相关**。

---

## 实验数字解读

### Table 1 核心

| Model | PG-19 Max | PG-19 Gain | PG-19 $C_H$ | Random Max | Random Gain | Random $C_H$ |
|---|---|---|---|---|---|---|
| Pythia-160M | 80 | 71 | 396 | 65 | 61 | 501 |
| Pythia-1.4B | 160 | 158 | 793 | 139 | 144 | 1108 |
| Llama-3.2-1B | 512 | 426 | 2120 | 316 | 295 | 2265 |
| Llama-3.2-3B | 1024 | 720 | 3292 | 460 | 457 | 3383 |
| Llama-3.1-8B | 1568 | 1094 | 4866 | 792 | 623 | 4541 |

四个反直觉发现：

### 发现1：PG-19和Fanfics几乎一样

Fanfics是2024年10月之后的AO3同人小说（https://archiveofourown.org/），模型pre-training肯定没见过。但数字和PG-19基本一致（Llama-8B: 1568 vs 1568 max）。

**这证明压缩能力不靠memorization。** 模型不需要"见过"这段text，照样能从vector重建。

### 发现2：Random text上$C_H$和natural text几乎一样

Llama-8B: random $C_H$=4541, PG-19 $C_H$=4866，只差7%。但Max差很多（792 vs 1568）。

人话解释：**vector存的information几乎与text是否natural无关，只与"LM对这段text的surprise"有关。** Random text的CE高，LM觉得不可预测，需要更多bits来描述每一个token；但vector能存的bits固定（~4500 bits），所以能存的token数少。Natural text CE低，LM能猜一部分，vector只填"猜不到的部分"，能存更多token。

这就是**rate-distortion theory的empirical体现**：vector提供固定rate（bits），distortion取决于source entropy。

### 发现3：Max和Gain在random text上几乎相等

Llama-8B random: Max=792, Gain=623，差166。这166是LM对random text也有一丁点prior（比如常见bigram）。

而PG-19上Max=1568, Gain=1094, 差474。这474是LM"白送"的——natural language的predictive prior。

**人话：vector本身存多少information（Gain），与LM自己能猜多少（Max - Gain），是两件事。** Random text上这两件事几乎合并，因为LM对random几乎无prior可猜。

### 发现4：Llama家族utilization随size上升

1B → 3B → 8B 的 $C_H$：2120 → 3292 → 4866 bits。Sub-linear但接近linear。

同样2048维hidden dimension：
- Pythia-1.4B: 793 bits
- Llama-3.2-1B: 2120 bits
- **2.7x差距，纯训练质量差异**

人话：**embedding space的effective capacity不只看维度，更看模型训练得好不好。** 训练越好，每个dimension承载的信息越密，capacity越高。Pythia是早期under-trained模型，所以同样维度容量小很多。

---

## 最美的发现：CE floor现象

Fig. 3 是paper里最elegant的图。把每段text：
- x轴：压缩前CE $H_{LM}$
- y轴：压缩后CE $H_{LM+[\text{mem}]}$

观察：
- 完美压缩的text落在y=x对角线（压缩后CE=压缩前CE，意味着无损）
- 没完美压缩的text全被拉到**同一个CE floor**，无论输入CE多高

公式化：

$$H_{LM+[\text{mem}]} \approx \max(0, H_{LM} - C_H)$$

人话：**vector像个固定容量的桶，装满$C_H$ bits信息就把CE拉低$C_H$。如果text原本CE就低于$C_H$，桶装不满，text完美重建；如果原本CE高于$C_H$，桶装满后溢出部分就是residual CE，text有损。**

**这个$C_H$是model-specific常数，与text length, text type, text domain都无关**。Larger model对应larger $C_H$，所以能perfect reconstruct的text更长。

这是非常clean的empirical law——一个model对应一个固定的representation capacity bits，跟输入是什么完全无关。这强烈暗示$C_H$是某种intrinsic property，可能与parameter count、hidden dim、训练quality的某种组合有关。

---

## Linear scaling with number of vectors

Fig. 4：1个vector存512 tokens（Llama-1B），16个vector存7168 tokens。几乎完美线性。

外推："The Hobbit"约120K tokens → Llama-8B只需128个vector，Llama-1B只需256个。

人话：**每个vector独立贡献固定capacity，几乎不互相干扰。** 这意味着压缩预算就是vector数量，简单线性可预测。

不过Llama-1B在16个vector时略微偏离linear，作者猜测是model本身在利用多vector时有inefficiency，但没深挖。这个偏离值得follow-up研究——可能是attention的capacity bottleneck？

---

## Mamba也能work，说明architecture无关

Mamba-1.4B：512 tokens进单vector，和Llama-3.2-1B、Sheared-LLaMA-1.3B持平。

人话：**这现象是architecture-agnostic的**——SSM的hidden state和Transformer的KV-cache在充当"decoding channel"上能力相当。$[\text{mem}]$ capacity来自representation space的几何性质，与attention/recurrence机制无关。

这是个强证据：capacity是关于"hidden state manifold能存多少information"的几何问题，不是关于具体序列建模机制的问题。

Mamba paper: https://arxiv.org/abs/2312.00752

---

## 与已有方法的差距

| 方法 | 压缩率 | Lossless? |
|---|---|---|
| RMT | ~x10 | No |
| AutoCompressors | ~x10 | No |
| ICAE | x4 | Yes |
| SelfCP | ~x12 | No |
| 500xCompressor | x480 | Lossy |
| Gisting | x26 | Lossy |
| LLMLingua | x20 | Lossy |
| KV-Distill | x100 | Lossy |
| LLMZip (arithmetic coding) | ~x7 | Yes |
| **本paper** | **x1568** | **Yes** |

差距来源：所有其他方法都train一个general encoder，encoder的generalization gap吃掉capacity。本paper用per-sample SGD绕过encoder。

代价是计算开销巨大——每段text要SGD几千步，A100上几分钟到20分钟。所以这是capacity probe，不是实用compression。

---

## 几个intuition

### LLM as codebook

Frozen LLM = 超大图书馆。Token sequences = 馆藏内容。$[\text{mem}]$ vector = 索书号。SGD = 找书过程。Decoder = 走到书架取书。

传统compression：训练图书管理员（encoder网络）对任何书都能找坐标。Per-sample optimization：绕过管理员，直接搜坐标，所以能测图书馆真实容量。

### 为什么capacity与text无关，只与CE有关

LM是个"猜测引擎"。
- CE低的text（cliché natural language）→ LM能猜大部分 → vector只填猜不到的少数 → 同样bits能存更多token
- CE高的text（random words）→ LM几乎猜不到 → vector要填全部 → 同样bits只能存少数token

所以$C_H$（bits）固定，$L_{max}$（tokens）随text complexity变化。

### 为什么不unique

附录E做了实验：同一段text用不同random init，训出多个$[\text{mem}]$ vector。发现：
- intra-sample cosine similarity几乎都低于0.8
- intra与inter-sample分布高度overlap
- linear interpolation between intra-sample vectors在中间失真

人话：**图书馆很大，同一本书可以放多个不同书架。但书架之间不是连续的，中间是过道（invalid region），所以线性插值会走到过道失真。**

这对实用compression是阻碍——解不unique意味着没法直接做hash/arithmetic。但对capacity probe反而说明：embedding manifold有足够volume容纳许多独立解，所以才能存这么多information。

### Capacity utilization作为pre-training quality指标

Fig. 5：同1B参数，Llama/OLMo/Sheared-LLaMA utilization 25-40%，Pythia只有10-20%。Pythia越大utilization越低（under-trained）。

人话：**模型训练得好，每个dimension承载更多information，utilization高。** 这可能比perplexity更直接反映representation quality——perplexity衡量predictive power，utilization衡量storage capacity，两者相关但不完全等价。

可能成为pre-training scaling law的新维度：不只看loss曲线，还看capacity utilization曲线。

---

## 与latent reasoning的联系

paper引用了Coconut（https://arxiv.org/abs/2412.06769 ），让LLM在latent space做multi-step reasoning。 Coconut本质是把intermediate reasoning state塞进continuous vector。

这篇paper的发现——单个vector能存上千token——为latent reasoning提供了**capacity justification**：一个latent step的vector理论上能承载很多"latent tokens"的information。所以latent reasoning的瓶颈不是vector维度，是训练目标和优化方法。

更深一层：如果vector能存1568个token的精确信息，那"在latent space思考"理论上能保持非常长的中间推理链。Coconut类工作还有很大空间可挖。

---

## 与test-time training的联系

这paper本质是**test-time training的一种特殊形式**——只optimize input vector，不optimize weights。Sun et al. 2024（https://arxiv.org/abs/2407.04620 ）是optimize weights。两者都是用SGD at inference time。

可以看作：这篇paper是test-time training的capacity probe——固定weights，只看input vector能承载多少。Sun等人的TTT是看weights能适应多少。两者结合可能是未来方向。

---

## 我想到的open questions

1. **$C_H$的scaling law是什么？** Fig. 5暗示随parameter count sub-linear增长，但paper没fit公式。如果能predict $C_H(N_{params}, d_{model}, training\_tokens, data\_quality)$，就能指导模型设计。这是最有价值的follow-up。

2. **训练data的什么property决定$C_H$？** 是diversity？scale？某种information density？如果搞清楚，可以从data角度提升capacity而不只堆参数。

3. **$[\text{mem}]$ vector能做arithmetic吗？** 如果vector A编码text A，vector B编码text B，A+B能否decode出某种"组合"？如果可以，这是latent reasoning的强大tool。附录E的interpolation实验暗示**线性组合不行**，但非线性组合未测。

4. **能否设计真正encoder逼近upper bound？** 如果能训练一个网络per-sample SGD的x1500效果，就实现实用lossless context compression。这是million dollar question。可能需要某种iterative refinement架构，或者test-time optimization baked-in。

5. **$C_H$与memorization的关系？** 模型pre-training时memorize的内容，是否贡献$C_H$？如果用unlearning去掉某些memorization，$C_H$会降吗？这能区分"模型capacity"和"模型knowledge"。

6. **跟Titans（https://arxiv.org/abs/2501.00663 ）等test-time memory架构的关系？** Titans让model在inference时学习memorize。这paper的发现能指导Titans类架构的memory module设计——memory vector应该多长，能存多少。

7. **跟mechanistic interpretability的交叉？** $[\text{mem}]$ vector如何在LLM内部trigger出1568个token？是激活某些circuit，还是直接写到KV-cache的某些position？如果能逆向工程这个process，能理解LLM如何"lookup"information。

8. **跟grokking的关系？** per-sample optimization的loss曲线长什么样？是sudden grokking还是gradual下降？如果sudden，说明vector找到某个"attractor basin"瞬间解锁重建。这能连接representation learning theory。

9. **跟adversarial example的关系？** $[\text{mem}]$ vector某种程度上是adversarial input——它让model生成特定长sequence。这跟adversarial attack的geometry有什么共同点？两者都是在input space搜索特定behavior的trigger。

10. **text的什么structure最难压缩？** paper只看了PG-19/fanfics/random。如果看code、math、structured data，$C_H$会变吗？如果不变，证实$C_H$是纯model property；如果变，说明$C_H$与text distribution有关。

---

## 与传统compression的bit-wise对比

Table 4 数据：

| Method | PG-19 | Random |
|---|---|---|
| zlib | 2.28x | 1.80x |
| bz2 | 2.46x | 1.94x |
| lzma | 2.28x | 1.86x |
| Huffman | 1.81x | 1.77x |
| AC + pythia-160m | 6.77x | 2.83x |
| **本paper (Llama-8B)** | ~1568x (token-wise) | ~792x (token-wise) |

**但bit-wise上本paper其实不如传统压缩**：4096维bfloat16 vector = 8KB，存1568个token，原文约6KB（按每token 4 bytes算），所以bit-wise是0.8x，**比原文还大**。

为什么token-wise x1568这么惊人？因为比较的是"embedding数量"，不是"bit数量"。原方法用1568个embedding表示1568个token，本方法用1个embedding表示1568个token，所以token-wise是x1568。

**这paper的目标不是做实用compression algorithm，是测LLM representation capacity**。作者明确说："Our goal is not to outperform standard compressors in bits-per-byte efficiency, but rather to show that LLMs can store significant amounts of text with only a single embedding."

这是关键定位——这paper是scientific probe，不是engineering product。

---

## 一图胜千言的mental model

把整个story压缩成一个画面：

> 想象一个巨大的图书馆，每本书的内容都能从一个索书号找到。传统图书管理员（encoder）对每本书只能给一个approximate索书号，因为要泛化到所有书。这paper直接用SGD搜坐标，发现图书馆真实容量比管理员用出来的高100倍。更神奇的是，每个图书馆（model）有一个固定容量$C_H$ bits，与放什么书无关，只与图书馆本身有关。容量随图书馆规模sub-linear增长，随索书号数量linear增长，与图书馆结构（Transformer vs Mamba）无关。

这个画面捕捉了所有关键发现。

---

## 为什么这paper重要

1. **建立capacity baseline**：给context compression、memory-augmented architecture、latent reasoning提供了硬数字——x1500是可达的，现有x10的方法有100x改进空间。

2. **揭示clean empirical law**：CE floor现象、linear scaling、architecture-agnostic，这三个law非常clean，能指导future work。

3. **提出capacity utilization指标**：可能成为pre-training quality的新维度，比perplexity更直接反映representation quality。

4. **连接多个领域**：context compression、latent reasoning、test-time training、memory architectures、mechanistic interpretability——这paper是这些领域的交汇点。

5. **简单方法，深刻insight**：per-sample optimization这trick极简，但揭示的现象极深。这是好science的标志——用最简单的方法问最基本的问题。

---

## 总结

一句话：**LLM input embedding space的真实capacity比我们用出来的高100倍，这capacity由model的representation quality决定，与text无关，随vector数量linear scale，与architecture无关。**

方法简单，数字惊人，law clean，意义深远。这是2026年我读过的最让我rethink LLM capacity的paper。

paper: https://arxiv.org/abs/2505.20800  
code (推测): https://github.com/airi-compute/cramming-tokens  
相关: Coconut https://arxiv.org/abs/2412.06769 , Test-time Training https://arxiv.org/abs/2407.04620 , Titans https://arxiv.org/abs/2501.00663

---

# Cramming 1568 Tokens into a Single Vector: 深度解读

这篇paper来自AIRI/MIPT的Kuratov等人(Recurrent Memory Transformer原作者),2026年的工作。核心question极其简洁:**一个LLM input embedding vector的信息容量到底有多大?**他们用一个简单的per-sample optimization trick,把frozen LLM当decoder,把trainable [mem] vector当code,发现Llama-3.1-8B能从一个4096维bfloat16 vector重建出1568个token的原文,而现有的encoder-based compression方法lossless只能x10。**两个数量级的gap**。

paper链接:https://arxiv.org/abs/2505.20800 (GitHub代码作者说available,从paper描述看应该是 https://github.com/airi-compute/cramming-tokens )

## 1. 核心方法:per-sample optimization把encoder换成SGD

### 1.1 Setup

给定token序列 $[t_1, t_2, \dots, t_N]$,引入K个trainable vectors $[\text{mem}] = [m_1, \dots, m_K]$ prepend到text前面:

$$\text{input} = [m_1, \dots, m_K, t_1, t_2, \dots, t_i]$$

frozen LM预测下一个token $t_{i+1}$,loss就是标准next-token prediction cross-entropy。**只有$m_k$被更新,LLM权重完全冻结**。这与Lester et al. 2021的Prompt Tuning形式上一样(https://arxiv.org/abs/2104.08691),但用法相反——prompt tuning是固定task优化prompt以适配下游任务,这里是为每段text单独optimize一个"address",让LLM能从地址反查回原文。

训练细节:AdamW,lr=0.01,β1=β2=0.9,weight decay=0.01,最多5000 steps,达到token-level accuracy=1.0就early stop。A100 80GB上一段text从十几秒到10-20分钟。

### 1.2 为什么这是个upper bound experiment

关键insight:他们不训练一个general-purpose encoder,而是**对每段text单独做optimization**。这意味着:
- encoder的能力被替换为SGD在vector space中的搜索
- 没有generalization gap,因为不要求encoder对没见过的text泛化
- 得到的是该LLM的**representation capacity上限**,不是实际系统能达到的压缩率

这就是为什么能比AutoCompressors/ICAE/500xCompressor多两个数量级——那些方法有encoder generalization的负担,per-sample optimization没有。

## 2. 理论上限公式详解

### Eq. (1): information-theoretic upper bound

$$L \leq \frac{d_{model} \times b}{\log_2 |\mathcal{V}|}$$

变量含义:
- $L$:理论上单个vector能编码的最大token数
- $d_{model}$:LLM的hidden dimension(如Llama-3.1-8B是4096)
- $b$:每个float element的bit数(bfloat16就是16)
- $|\mathcal{V}|$:vocabulary size(Llama-3是128256)
- $\log_2|\mathcal{V}|$:每个token在uniform分布下的information上限,约17 bits

代入Llama-3.1-8B:$(4096 \times 16) / \log_2(128256) \approx 65536 / 17 \approx 3862$ tokens理论上限。实际达到1568,utilization约40.7%。

Llama-3.2-1B:$(2048 \times 16) / 17 \approx 1931$ theory vs 512 empirical,utilization 26.5%。

注意这里有个隐含的assume:**uniform token distribution**。真实text的empirical entropy远低于$\log_2|\mathcal{V}|$,所以这个bound是悲观估计;但反过来说,如果想让LM完美重建,$\log_2|\mathcal{V}|$是worst case,因为vector中存的信息必须能区分任意可能的下一token。

## 3. 三个核心metric及公式

### Eq. (2): Decoding Capacity $L_{max}$

$$L_{max} = \max\{L \mid \text{Acc}(\text{LM}(t_{[1:L]} \mid [\text{mem}])) > \text{thr}\}$$

- $L_{max}$:能perfect重建的最长text length
- $\text{Acc}$:teacher-forcing下的token-level accuracy
- $\text{thr}$:threshold,paper里用0.99
- 这衡量的是"能存多少token"

### Eq. (3): Token Gain $C_{tokens}$

$$C_{tokens} = \sum_{i=1}^{N} \mathbb{1}(t_i = \text{LM}(t_{[1:i-1]} \mid [\text{mem}])) - \sum_{i=1}^{N} \mathbb{1}(t_i = \text{LM}(t_{[1:i-1]}))$$

- 第一项:有$[\text{mem}]$时LM正确预测的token数
- 第二项:无$[\text{mem}]$时LM正确预测的token数(baseline)
- $\mathbb{1}(\cdot)$:indicator function
- 差值就是$[\text{mem}]$ vector**净贡献**的token数

这是关键decoupling:**LM自己基于language prior能猜对很多token**(尤其PG-19这种natural text),我们要扣除这部分,才能知道vector本身存了多少信息。

### Eq. (4): Information Gain $C_H$

$$C_H = H_{LM} - H_{LM+[\text{mem}]}$$

- $H_{LM} = H(P_\theta(t_{[1:N]}))$:无memory时的cross-entropy(单位bits)
- $H_{LM+[\text{mem}]} = H(P_\theta(t_{[1:N]} \mid [\text{mem}]))$:有memory时的cross-entropy
- $C_H$就是$[\text{mem}]$减少的不确定性,单位bits

这是最干净的metric——直接用information theory语言衡量vector存了多少bit信息。**它独立于text长度,独立于text domain**,几乎只与model相关。

## 4. 实验数据深度解读

### Table 1 关键数字

| Model | PG-19 Max | PG-19 Gain | PG-19 Info Gain | Random Max | Random Gain | Random Info Gain |
|---|---|---|---|---|---|---|
| Pythia-160M | 80 | 70.9±11.0 | 396.4±46.0 | 65 | 61.3±6.6 | 500.8±38.9 |
| Pythia-1.4B | 160 | 158.0±29.1 | 792.8±143.4 | 139 | 144.4±17.5 | 1108.2±136.2 |
| Llama-3.2-1B | 512 | 426.2±79.2 | 2119.9±364.8 | 316 | 294.9±64.8 | 2265.2±498.7 |
| Llama-3.2-3B | 1024 | 720.3±80.2 | 3292.2±320.0 | 460 | 456.9±72.1 | 3382.6±585.2 |
| Llama-3.1-8B | 1568 | 1094.1±127.6 | 4865.7±546.6 | 792 | 623.2±97.3 | 4541.2±758.6 |

四个关键观察:

**(a) PG-19 vs Fanfics几乎相同**:fanfics是2024年10月之后发表的AO3同人小说(https://archiveofourown.org/),模型pre-training肯定没见过,但数字与PG-19基本一致(Llama-3.1-8B: 1568 vs 1568,Gain 1094 vs 1072)。这说明**压缩能力不依赖memorization**。

**(b) Random text上Info Gain居然和natural text差不多**:Llama-3.1-8B上random是4541 bits,PG-19是4866 bits,只差7%。但Max tokens差很多(792 vs 1568)。这意味着:
- $[\text{mem}]$ vector存的information几乎与text是否natural无关
- natural text的额外bonus来自LM自己的language prior(可以猜token)
- **vector本身是一个language-agnostic的episodic memory**

**(c) Max与Gain在random text上几乎相等**:Llama-3.1-8B random: Max=792, Gain=623.2。差166。这是因为random text几乎没有LM能猜的prior,所以vector存多少就能decode多少。而在PG-19上,Max=1568但Gain=1094,差474,这474是LM"白送"的——LM对natural language的predictive prior。

**(d) Llama家族capacity utilization随size单调上升**:1B→3B→8B的Info Gain:2119→3292→4866。这是sub-linear但接近linear的scaling。这告诉我们**embedding space的effective capacity依赖于模型size,不仅仅是hidden dim**。同样2048维的hidden dimension,Pythia-1.4B只能存792 bits的信息,Llama-3.2-1B能存2120 bits,**2.7x差距**——pure training quality差异。

### 4.3 Cross-entropy threshold现象(Fig. 3)

这是paper最有意思的发现。把每段text的"压缩前CE" $H_{LM}$ 放x轴,"压缩后CE" $H_{LM+[\text{mem}]}$ 放y轴:

- 完美压缩:text落在y=x对角线上($H_{LM+[\text{mem}]} = H_{LM}$,意味着text的CE本来就低于model capacity,加memory后无信息增益但也无损)
- 实际上,所有没被完美压缩的text都被压到同一个CE值——也就是模型有一个**固定的residual entropy floor**,无论输入CE多高都被拉到这个floor
- 红色虚线就是threshold:$H_{LM} < C_H \Rightarrow$ perfect reconstruction

公式化表达:
$$H_{LM+[\text{mem}]} \approx \max(0, H_{LM} - C_H)$$

这是一个非常干净的**linear compression law**:vector提供固定的$C_H$ bits信息减除,余下的CE由LM自己承担。如果余下<0,说明vector不仅能填满信息,还有冗余,自然perfect reconstruct。

**这个$C_H$是model-specific常数,与text length, text type, text domain都无关**(Table 1跨PG-19/fanfics/random Info Gain稳定)。这强烈暗示**$C_H$是某种representation capacity的intrinsic property**——可能类似model的expressive power dimension,与parameter count, hidden dim, training qualityjointly决定。

Larger model对应larger $C_H$,所以red threshold更靠右,能perfect reconstruct的text更长。

### 4.4 Linear scaling with number of vectors (Fig. 4)

Pythia-160M:32个vectors→2016 tokens(达到context上限)
Llama-3.2-1B:16个vectors→7168 tokens

虚线是ideal linear scaling,实测点几乎贴着虚线。这说明每个$[\text{mem}]$ vector**独立贡献固定的capacity**,几乎不互相干扰。

外推:"The Hobbit"约120K tokens,Llama-3.1-8B用128个vectors可压缩,Llama-3.2-1B用256个。一个vector是4096维bfloat16 = 8KB,128个=1MB存120K tokens原文(原文gzip约300KB,所以这个方法bit-wise其实不如传统压缩,见Table 4)。

注意Llama-3.2-1B在16个vector时略微偏离linear,作者指出可能是model本身在利用多vector时有inefficiency,但没深入。

### 4.6 Mamba也能work

Mamba-1.4B:512 tokens进单vector,与Llama-3.2-1B(512)、Sheared-LLaMA-1.3B(512)持平。这说明**现象是architecture-agnostic的**——SSM的hidden state和Transformer的KV-cache在充当"decoding channel"上能力相当。这是一个强证据:$[\text{mem}]$ capacity来自representation space的几何性质,与具体的recurrence/attention机制无关。

paper提到Mamba-1.4B的Info Gain:1599.5 bits(PG-19),2062.3 bits(random)。Random更高,印证natural text的LM prior贡献。

## 5. 与已有context compression方法对比

| 方法 | 类型 | 压缩率 | Lossless? |
|---|---|---|---|
| RMT (Bulatov 2022) | Train encoder via recurrence | ~x10 | No |
| AutoCompressors (Chevalier 2023) | Train encoder | ~x10 | No |
| ICAE (Ge 2024) | Autoencoder + LoRA | x4 | Yes |
| SelfCP (Gao 2024) | LLM as compressor + adapter | ~x12 | No |
| 500xCompressor (Li 2024b) | Autoencoder + layer connections | x480 (with quality loss) | No |
| Gisting (Mu 2023) | LM自压缩 | x26 | Lossy |
| LLMLingua (Jiang 2023) | Token pruning | x20 | Lossy |
| KV-Distill (Chari 2025) | KV cache distillation | x100 | Lossy |
| LLMZip (Valmeekam 2023) | Arithmetic coding | ~x7 | Yes |
| **本paper** | Per-sample optimization | **x1568** | **Yes** |

关键差距来源:以上所有方法都train一个general encoder。这个paper用per-sample SGD替代encoder,所以测的是upper bound。

## 6. 我的intuitive理解

### 6.1 这本质是"LLM as codebook"

把frozen LLM看成一个巨大的codebook,token sequences是其索引的内容,$[\text{mem}]$ vector是这个codebook的address。SGD在address space搜索,找到对应目标text的address。这跟传统compression的根本区别:传统compression是encoder-decoder两个explicit function;这里encoder是隐式的(由SGD近似),decoder是LLM的forward pass。

### 6.2 为什么capacity与text complexity无关,只与CE有关

CE $H_{LM}$ 衡量LM对text的surprise。CE高的text(如random words)LM觉得不可预测,要存的information多;CE低的text(如cliché natural language),LM本身就能predict,要存的information少。$[\text{mem}]$ vector只需要填补"LM不知道的部分",所以vector capacity用bits衡量($C_H$)是固定的,但用tokens衡量($L_{max}$)会随text complexity变化。

这其实是**rate-distortion theory**的一种empirical体现:vector提供固定rate(bits),distortion取决于source的entropy。

### 6.3 为什么2048维vector能存几千bits?

2048维bfloat16 vector理论有32768 bits。但是vector必须落在LLM input embedding manifold的"valid region"内——不是任意bit pattern都是合法embedding。empirical utilization只有26-40%,说明manifold的"effective dimension"远小于ambient dimension。

paper附录E做了一个有意思的实验:对同一段text用不同random init训练出多个$[\text{mem}]$ vector,发现:
- intra-sample cosine similarity几乎都低于0.8
- intra与inter-sample分布高度overlap
- linear interpolation between intra-sample vectors在中间会失真

这说明:**压缩解不unique,在embedding space中是scattered的,不形成convex basin**。这是这个方法用作实用compression的一个阻碍——但作为capacity probe反而说明:embedding manifold有足够大的volume容纳许多独立的解,所以才能存这么多information。

### 6.4 与最近Coconut/latent reasoning工作的联系

paper引用了Hao et al. 2024 (Coconut, https://arxiv.org/abs/2412.06769 )。Coconut让LLM在latent space做multi-step reasoning,本质上是把intermediate reasoning state塞进continuous vector。这篇paper的发现——单个vector能存上千token——为latent reasoning提供了capacity justification:一个latent step的vector理论上能承载很多"latent tokens"的information,所以latent reasoning不是bottlenecked by vector dimension,而是bottlenecked by训练目标。

### 6.5 capacity utilization作为pre-training质量指标

Fig. 5显示同样1B参数,Llama/OLMo/Sheared-LLaMA的utilization约25-40%,Pythia只有~10-20%,Pythia越大utilization越低(under-trained)。这很intuitive:如果模型representation space被well-trained,每个dimension承载更多information,自然utilization高。这可能比perplexity更直接反映representation quality——perplexity衡量predictive power,utilization衡量**storage capacity**,这两者相关但不同。

## 7. Limitations与open questions

paper自己承认:
- 没研究$[\text{mem}]$ vector的semantic structure(附录E只看了cosine similarity,没看下游task utility)
- 只测到8B,大模型行为未知
- 计算开销巨大(每段text要SGD数千步),实用价值待商榷
- Random text用dictionary word采样,实际vocab tokenize可能更难

我额外想到的几个open question:
1. $C_H$的scaling law是什么?Fig.5暗示随parameter count sub-linear增长,但paper没fit具体公式。如果能predict $C_H(N_{params}, d_{model}, training\_tokens)$,就能指导模型设计。
2. training data的什么property决定了$C_H$?是diversity?scale?某种information density?
3. $[\text{mem}]$ vector能做arithmetic吗?如果vector A编码text A,vector B编码text B,A+B能否decode出某种"组合"?如果可以,这是latent reasoning的强大tool。附录E的interpolation实验暗示**不行**(线性插值会失真),但非linearity组合未测。
4. 把这个方法倒过来:能否设计一个**真正的encoder**网络,逼近per-sample optimization的upper bound?如果能,就实现了x1000级别的实用lossless context compression,对long-context LLM是巨大savings。
5. 与test-time training (Sun et al. 2024, https://arxiv.org/abs/2407.04620 )的联系:两者都是用SGD at inference time。这个paper可以看作test-time training的capacity probe——只是optimize的是input而不是weights。

## 8. 总结

这篇paper的关键贡献是一个**simple but powerful的probe**:用per-sample optimization测量LLM input embedding的information capacity。结果惊人(单vector存1568 tokens,x1500 vs 现有x10),且揭示了三个clean empirical law:

1. **Capacity与text complexity无关,只与cross-entropy有关**:$H_{LM+[\text{mem}]} \approx \max(0, H_{LM} - C_H)$
2. **Capacity随vector数量linear scaling**:实用上vector数就是"压缩预算"
3. **Capacity与architecture无关**:Transformer与Mamba表现一致

这给long-context compression、memory-augmented architectures、latent reasoning都提供了基础数字——representation space远比我们利用得充分。两个数量级的gap就是当前encoder设计的失败,也是future work的金矿。

参考链接:
- Prompt Tuning: https://arxiv.org/abs/2104.08691
- Memory Transformer: https://arxiv.org/abs/2006.11527
- Recurrent Memory Transformer: https://arxiv.org/abs/2204.05892
- ICAE: https://arxiv.org/abs/2305.14788
- 500xCompressor: https://arxiv.org/abs/2408.03094
- Gisting: https://arxiv.org/abs/2310.18558
- LLMLingua: https://arxiv.org/abs/2310.05736
- Coconut (latent reasoning): https://arxiv.org/abs/2412.06769
- Language Modeling is Compression: https://arxiv.org/abs/2302.04404
- DeepMind Compression paper: https://arxiv.org/abs/2302.04404
- Test-time Training: https://arxiv.org/abs/2407.04620
- Pythia: https://arxiv.org/abs/2304.01383
- Llama 3: https://arxiv.org/abs/2407.21783
- Mamba: https://arxiv.org/abs/2312.00752
- Titans (test-time memory): https://arxiv.org/abs/2501.00663
- PG-19: https://arxiv.org/abs/1911.05516
