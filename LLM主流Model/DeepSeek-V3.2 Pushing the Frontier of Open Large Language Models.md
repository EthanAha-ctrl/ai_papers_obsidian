---
source_pdf: DeepSeek-V3.2 Pushing the Frontier of Open Large Language Models.pdf
paper_sha256: 2bec0671778769c159ec389412727d1f3d4889fe1c71564b61edaa24705bd17b
processed_at: '2026-08-03T18:45:28-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DeepSeek-V3.2 用人话讲讲

Andrej，这篇paper本质上就是在说一件事：**怎么让open-source model追上闭源model**。他们通过三个技术axis来破局，我会把每个axis背后的intuition讲清楚。

---

## 1. DSA：让attention从 $O(L^2)$ 变成 $O(Lk)$

### 1.1 问题在哪

vanilla attention在long-context下compute是 $O(L^2)$，$L$ 是sequence length。当 $L = 128K$ 的时候，$L^2 = 1.6 \times 10^{10}$，这个计算量在deployment时就是灾难。尤其是RL post-training阶段需要反复跑long context，这个bottleneck会kill整个pipeline。

### 1.2 DSA的核心idea

think一下人类怎么读long document：你不会逐字逐句去attend每个词，你会先scan一遍找relevant paragraph，再仔细读那几个paragraph。DSA就是这个思路的两阶段实现：

**Stage 1: Lightning Indexer做粗筛**

$$I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \text{ReLU}(\mathbf{q}_{t,j}^I \cdot \mathbf{k}_s^I)$$

- $t$：当前token的position
- $s$：前面某个token的position
- $H^I$：indexer的head数量，很小（为了speed）
- $\mathbf{q}_{t,j}^I$：token $t$ 在indexer第 $j$ 个head的query vector，维度 $d^I$
- $\mathbf{k}_s^I$：token $s$ 的key vector
- $w_{t,j}^I$：第 $j$ 个head的scalar weight，learned
- ReLU：保留positive correlation

intuition上，每个head $j$ 独立算一个similarity score，ReLU过滤掉negative correlation（语义无关的token对），然后 $w_{t,j}^I$ 做weighted sum得到最终的index score。这个score告诉query token $t$："前面哪个token $s$ 对你最relevant"。

paper特别提到用ReLU而非softmax是为了throughput，并且可以FP8实现。这说明indexer的compute bottleneck是memory bandwidth bound，ReLU比softmax省了normalization的开销。

参考: [Native Sparse Attention](https://aclanthology.org/2025.acl-long.1126/)

**Stage 2: Top-k selection + sparse attention**

$$\mathbf{u}_t = \text{Attn}\big(\mathbf{h}_t, \{\mathbf{c}_s \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}\big)$$

- $\mathbf{h}_t$：token $t$ 的hidden state
- $\mathbf{c}_s$：MLA的latent key-value entry（压缩后的KV cache单元）
- Top-k：选 $k=2048$ 个index score最高的token
- 只对这 $k$ 个token做standard attention

intuition：与其对全部128K个token做attention，不如先让indexer告诉你"哪2048个token最relevant"，然后只对这2048个做full attention。compute从 $O(L^2)$ 降到 $O(L \cdot k)$，$L=128K$, $k=2048$，speedup大约是62x。

### 1.3 为什么基于MLA的MQA mode实现

这点很technical但很关键。MLA（Multi-head Latent Attention）是DeepSeek-V3引入的，核心是KV cache compression——用low-rank latent vector代替原始的KV。MLA有两种mode：

- **MHA mode**：每个query head有独立的latent KV access pattern（用于training和prefilling）
- **MQA mode**：所有query heads共享同一个latent KV entry（用于decoding）

DSA选择在MQA mode上实现，原因是kernel-level efficiency。如果每个query head选不同的top-k tokens，GPU需要做scatter/gather，memory access pattern非常unfriendly。MQA mode下，所有query heads共享同一个selection，只需要gather一次KV entries，所有heads复用。

这是一个hardware-aware design choice。paper引用了Yuan et al., 2025的hardware alignment principle。

参考: [Shazeer MQA](https://arxiv.org/abs/1911.02150)

### 1.4 Continued Pre-Training的两阶段策略

这里是最体现工程功力的地方。你已经有一个训练好的DeepSeek-V3.1-Terminus（128K context），现在要加入DSA，但不能破坏已有capability。怎么做？

**Stage 1: Dense Warm-up（初始化indexer）**

冻结所有model parameters，只训练lightning indexer。target distribution的构造：

$$p_{t,:} \in \mathbb{R}^t, \quad p_{t,s} = \frac{\sum_h \text{AttnScore}_{t,s,h}}{\sum_{s'} \sum_h \text{AttnScore}_{t,s',h}}$$

- $p_{t,s}$：token $t$ 对token $s$ 的target attention weight
- $h$：attention head index
- 分子：token $t$ 对token $s$ 在所有head上的attention score之和
- 分母：L1 normalization，使得 $\sum_s p_{t,s} = 1$

Loss：

$$\mathcal{L}^I = \sum_t \mathbb{D}_{\text{KL}}\big(p_{t,:} \big\| \text{Softmax}(I_{t,:})\big)$$

- $p_{t,:}$：target distribution（detached，no gradient flow back to main model）
- $\text{Softmax}(I_{t,:})$：indexer输出的probability distribution
- $\mathbb{D}_{\text{KL}}$：KL divergence，$D_{KL}(p \| q) = \sum_s p_s \log(p_s / q_s)$

intuition：让indexer学习去**mimic** main attention的pattern。main attention说"token $t$ 应该attend token $s_1, s_2, s_3$"，indexer就要学会输出高的 $I_{t,s_1}, I_{t,s_2}, I_{t,s_3}$。这是一个distillation objective。

训练1000步，每步16个sequence × 128K tokens，总共2.1B tokens。learning rate $10^{-3}$，相对较大，因为indexer从零开始。

**Stage 2: Sparse Training（全模型适应）**

引入top-k selection，训练所有参数。Loss变成只在selected token set上做KL：

$$\mathcal{L}^I = \sum_t \mathbb{D}_{\text{KL}}\big(p_{t,S_t} \big\| \text{Softmax}(I_{t,S_t})\big)$$

- $S_t = \{s \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}$：被选中的token set
- $p_{t,S_t}$：target distribution只在selected set上重新normalize
- $\text{Softmax}(I_{t,S_t})$：indexer输出只在selected set上做softmax

关键engineering detail："we detach the indexer input from the computational graph"。意思是indexer的gradient只来自 $\mathcal{L}^I$，main model的gradient只来自language modeling loss。两个optimization目标decoupled，避免gradient打架。

training 15000步，每步480个sequence × 128K tokens，总共943.7B tokens。learning rate $7.3 \times 10^{-6}$，很小，避免破坏pre-trained knowledge。

### 1.5 实际效果

paper的Figure 3展示了在H800集群上（2 USD/GPU hour）的inference cost对比。在long sequence decoding时，DSA的token cost远低于V3.1-Terminus的dense MLA。AA-LCR benchmark上V3.2-Exp比V3.1-Terminus高4分，Fiction.liveBench上多个metrics也consistent outperform。

---

## 2. Scaling GRPO：让RL训练稳定scale up

### 2.1 为什么RL训练容易炸

RL training LLM的核心困难是off-policy问题。你sample一大batch rollout data，然后split成多个mini-batch做多次gradient update。每次update后policy就变了，但你的data还是old policy采的，这个gap会destabilize training。

paper列举了四个stabilizer，我逐个讲intuition。

参考: [GRPO原paper](https://arxiv.org/abs/2402.03300)

### 2.2 Unbiased KL Estimate

原版K3 estimator (Schulman, 2020)的KL近似有个bug。当 $\pi_\theta(o_{i,t}) \ll \pi_{\text{ref}}(o_{i,t})$，即当前policy对某token的probability远低于reference policy时，gradient会assign unbounded weight去maximize这个token的likelihood。这会导致noisy updates，累积后degrade sample quality。

DeepSeek-V3.2的修正（公式7）：

$$\mathbb{D}_{\text{KL}}(\pi_\theta(o_{i,t}) \| \pi_{\text{ref}}(o_{i,t})) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t}|q, o_{i,<t})} \left(\frac{\pi_{\text{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - \log\frac{\pi_{\text{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - 1\right)$$

- $\pi_\theta$：current policy
- $\pi_{\text{old}}$：sampling policy（rollout时的policy）
- $\pi_{\text{ref}}$：reference policy（frozen，通常是SFT checkpoint）
- $o_{i,t}$：第 $i$ 个response的第 $t$ 个token
- $q$：prompt
- $o_{i,<t}$：前 $t$ 个token

intuition：因为 $o_{i,t}$ 是从 $\pi_{\text{old}}$ 采样的，你要在 $\pi_\theta$ 的expectation下算KL，就需要做importance weighting correction。乘以 $\pi_\theta / \pi_{\text{old}}$ 使得estimator unbiased。

paper提到，不同domain benefit from不同强度的KL penalty。math domain甚至可以完全omit KL penalty，因为math reasoning需要exploration，太强的KL constraint会limit exploration。

参考: [Schulman KL approximation blog](http://joschu.net/blog/kl-approx.html)

### 2.3 Off-Policy Sequence Masking

公式(8-9)引入一个binary mask $M_{i,t}$：

$$M_{i,t} = \begin{cases} 0 & \hat{A}_{i,t} < 0, \quad \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\log\frac{\pi_{\text{old}}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} > \delta \\ 1 & \text{otherwise} \end{cases}$$

- $\hat{A}_{i,t}$：advantage，$R_i - \text{mean}(R)$
- $R_i$：第 $i$ 个response的reward
- $\delta$：policy divergence threshold
- 第二个条件是sequence-level的平均KL between $\pi_{\text{old}}$ and $\pi_\theta$

关键：**只mask negative advantage的sequences**。

intuition：
- Positive advantage + off-policy：虽然data是old policy采的，但"what to do"的signal依然valuable，model可以learn from it
- Negative advantage + off-policy：policy已经移动了，这些negatives可能不再代表当前policy的真实failure mode，继续learn from them会misleading

这个asymmetry设计很精妙。模型最benefit from learning自己的mistakes，但高度off-policy的negative samples反而会destabilize optimization。

### 2.4 Keep Routing（MoE-specific）

MoE model的issue：inference framework和training framework对同一个input可能route到不同的expert。这导致active parameter subspace突变，gradient update的方向和实际inference时active的参数mismatch。

Solution：在sampling阶段记录每个token的expert routing path，training阶段强制使用相同的routing path。这保证了identical expert parameters被优化。

这个technique从DeepSeek-V3-0324起就采用了，说明是工程实践中踩坑后发现的critical issue。

### 2.5 Keep Sampling Mask

Top-p/top-k sampling会truncate action space，砍掉low-probability tokens。但training时如果 $\pi_\theta$ 的action space和 $\pi_{\text{old}}$ 不同，violates importance sampling principle。

Solution：保存sampling时的truncation mask，training时apply到 $\pi_\theta$，保证两个policy share相同的action subspace。

Empirically，这对维持language consistency很重要。如果不keep mask，RL training后model可能会generate一些采样时被truncated的low-probability tokens，导致language degeneration。

### 2.6 Post-Training Compute Budget

paper提到post-training compute budget超过pre-training cost的10%。这是一个很关键的data point。

传统上大家认为pre-training是compute大头，post-training只是fine-tuning。但DeepSeek-V3.2的经验表明，reasoning capability需要significant RL compute才能unlock。paper hypothesize这还能继续scale。

这和OpenAI的o1/o3系列、DeepMind的Gemini-3.0-Pro的方向一致——test-time compute和post-training compute是新的scaling axis。

参考: [DeepSeek-R1 Nature paper](https://www.nature.com/articles/s41586-025-08865-9)

---

## 3. Thinking in Tool-Use：让reasoning和agent能力融合

### 3.1 Context Management问题

DeepSeek-R1的原始策略：每轮new message到来时discard reasoning content。这在pure reasoning场景OK，但在tool-use场景下非常wasteful。

想象一个multi-step tool calling场景：
1. Model think一段，call tool A
2. Tool A返回结果
3. Model think一段，call tool B
4. Tool B返回结果
5. Model think一段，给final answer

如果每次tool返回后都discard之前的reasoning，model在step 3和step 5都需要重新reason整个problem，token效率极差。

DeepSeek-V3.2的规则：
- **Only new user message triggers reasoning discard**
- Tool-related messages（tool outputs）append时，保留之前的reasoning content
- Reasoning被discard时，tool call history和results依然保留在context中

intuition：tool output是对reasoning的补充信息，不需要重新reason。只有当用户提出new question时，之前的reasoning context才失效。

**Important caveat**：某些agent framework（Roo Code, Terminus）通过user message模拟tool interaction。这种架构下，tool output会被当成new user message，导致reasoning被错误discard。Paper建议这些场景用non-thinking model。

### 3.2 Cold-Start策略

这是把reasoning和tool-use两个capability unify到一个trajectory里的关键。paper设计了三个system prompt template：

**Template 1: Pure reasoning data**

```
You are an expert Python programmer. Please first reason 
before giving the final answer. The reasoning process enclosed 
within . The final answer is output after 
the  tag.
```

Output format: `[FINAL ANSWER]`

**Template 2: Non-reasoning agent data**

```
Use Python interpreter tool to execute Python code. 
## Tools: {TOOL-DESCRIPTIONS}
Important: ALWAYS adhere to this exact format for tool use: 
{TOOLCALL-FORMAT}
```

Output format: `[MULTI-TURN TOOLCALL] [FINAL ANSWER]`

**Template 3: Reasoning + Agent data（这是关键创新）**

```
You are a helpful assistant with access to a Python interpreter.
- You may use the Python tool **multiple times** during your 
  reasoning, a.k.a in , with a maximum of 20 
  code executions.
- Call the Python tool early in your reasoning to aid in solving 
  the task.
- Do NOT invoke any tools in your presented final solution steps.
- Prefer code execution over language-based reasoning whenever 
  possible.
```

Output format: `

---

# DeepSeek-V3.2 Paper深度技术讲解

Andrej, 这篇paper的核心贡献可以分解为三个相互正交的技术axis，我会逐个深入剖析其数学formulation和工程实现细节。

---

## 1. DSA (DeepSeek Sparse Attention) - 架构核心

### 1.1 核心动机与设计哲学

DeepSeek-V3.2的key architectural innovation是DSA。传统vanilla attention的complexity是 $O(L^2)$，在long-context场景下成为deployment瓶颈。DSA的核心idea是：**用一个轻量级的"lightning indexer"先做粗筛，再用fine-grained top-k selection做稀疏attention**。

这与Native Sparse Attention (Yuan et al., 2025)的hardware-aligned思路一脉相承，但DSA的特殊之处在于它instantiated under MLA (Multi-head Latent Attention)，并且通过continued training从DeepSeek-V3.1-Terminus平滑迁移。

参考: [Native Sparse Attention paper](https://aclanthology.org/2025.acl-long.1126/)

### 1.2 Lightning Indexer的数学formulation

Index score的计算如公式(1)：

$$I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \text{ReLU}(\mathbf{q}_{t,j}^I \cdot \mathbf{k}_s^I)$$

变量含义：
- $t$: 当前query token的position index
- $s$: preceding token的position index ($s < t$)
- $H^I$: indexer heads的数量（small，for efficiency）
- $\mathbf{q}_{t,j}^I \in \mathbb{R}^{d^I}$: 第$j$个indexer head对应position $t$的query vector，由$\mathbf{h}_t$通过linear projection得到
- $\mathbf{k}_s^I \in \mathbb{R}^{d^I}$: position $s$的key vector，由$\mathbf{h}_s$派生
- $w_{t,j}^I \in \mathbb{R}$: 第$j$个head的scalar weight，可学习的head importance
- $\text{ReLU}$: 选择ReLU而非softmax是为throughput，且可FP8实现

**Intuition building**: 这个indexer本质是一个**learned sparse pattern predictor**。每个head $j$独立计算一个$q \cdot k$的similarity score，ReLU保留positive correlation（语义相关），然后通过$w_{t,j}^I$做weighted aggregation。这个score不直接进入attention computation，只用于top-k的routing决策。

### 1.3 Fine-grained Token Selection + Attention

公式(2)定义了实际的attention计算：

$$\mathbf{u}_t = \text{Attn}\big(\mathbf{h}_t, \{\mathbf{c}_s \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}\big)$$

- $\mathbf{c}_s$: MLA中的latent key-value entry（压缩后的KV cache单元）
- Top-k选择$k=2048$个tokens（在sparse training stage）
- 只对选中的KV entries做standard attention

**Key insight**: 这里的$\mathbf{c}_s$是MLA的latent vector，所以在kernel level可以shared across all query heads（即MQA mode of MLA），这就避免了per-head KV selection的scatter/gather开销。这是DSA能与MLA co-design的关键。

### 1.4 DSA under MLA的instantiation

Figure 2展示的架构值得仔细解析。MLA有两种mode：
- **MHA mode**: 用于training和prefilling（每个query head有独立的latent KV access pattern）
- **MQA mode**: 用于decoding（所有query heads共享同一个latent KV entry）

DSA选择在MQA mode上实现，因为：
1. Kernel-level efficiency要求每个KV entry被多个query复用（Yuan et al., 2025的hardware alignment principle）
2. Decoding阶段的memory bandwidth bound特性使得KV cache size成为瓶颈，MQA的shared latent vector天然适配

参考: [Shazeer MQA paper](https://arxiv.org/abs/1911.02150)

### 1.5 Continued Pre-Training的两阶段策略

这是工程上最精妙的部分。从已经训练到128K context的DeepSeek-V3.1-Terminus出发，DSA的引入不能破坏已有的capability。

#### Stage 1: Dense Warm-up (Indexer initialization)

冻结所有model parameters，只训练lightning indexer。Target distribution的构造：

$$p_{t,:} \in \mathbb{R}^t, \quad p_{t,s} = \frac{\sum_h \text{AttnScore}_{t,s,h}}{\sum_{s'} \sum_h \text{AttnScore}_{t,s',h}}$$

即main attention scores跨head sum后做L1-normalization。Loss为：

$$\mathcal{L}^I = \sum_t \mathbb{D}_{\text{KL}}\big(p_{t,:} \big\| \text{Softmax}(I_{t,:})\big)$$

- $p_{t,:}$: target distribution（detached，no gradient）
- $\text{Softmax}(I_{t,:})$: indexer输出的probability distribution
- KL: $D_{KL}(p \| q) = \sum_s p_s \log(p_s / q_s)$

**Intuition**: Indexer要学习去**mimic** main attention的attention pattern分布。这是一个distillation objective：把dense attention的"哪里重要"知识distill到sparse indexer中。

训练配置：
- Learning rate: $10^{-3}$（相对较大，因为indexer从零初始化）
- Steps: 1000
- Batch: 16 sequences × 128K tokens
- Total: 2.1B tokens

#### Stage 2: Sparse Training (Full adaptation)

引入top-k selection，训练所有参数。关键变化：

$$\mathcal{L}^I = \sum_t \mathbb{D}_{\text{KL}}\big(p_{t,S_t} \big\| \text{Softmax}(I_{t,S_t})\big)$$

其中 $S_t = \{s \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}$，即只在selected token set上做KL alignment。

**Critical engineering detail**: "we detach the indexer input from the computational graph for separate optimization"。这意味着indexer的gradient只来自$\mathcal{L}^I$，main model的gradient只来自language modeling loss。两个optimization目标decoupled，避免了gradient interference。

训练配置：
- Learning rate: $7.3 \times 10^{-6}$（较小，避免破坏pre-trained knowledge）
- k=2048 selected KV tokens per query
- Steps: 15000
- Batch: 480 sequences × 128K tokens
- Total: 943.7B tokens

### 1.6 Complexity Analysis

- Main attention: $O(L \cdot k)$ where $k \ll L$ (e.g., $k=2048$, $L=128K$)
- Lightning indexer: $O(L^2)$ but with very small $H^I$ and FP8，constant factor极小
- End-to-end speedup在long-context场景显著

Figure 3的cost analysis显示，在H800集群上（2 USD/GPU hour），DSA在long sequence decoding时的token cost远低于V3.1-Terminus的dense MLA。

---

## 2. Scaling GRPO - RL训练的稳定性recipes

### 2.1 GRPO基础objective回顾

公式(5)是GRPO的核心：

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\text{old}}(\cdot|q)} \left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\big(r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_{i,t}\big) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]$$

变量：
- $G$: group size（同一prompt采样的response数量）
- $o_i$: 第$i$个response，$|o_i|$是其token长度
- $r_{i,t}(\theta) = \pi_\theta(o_{i,t}|q, o_{i,<t}) / \pi_{\text{old}}(o_{i,t}|q, o_{i,<t})$: importance sampling ratio
- $\hat{A}_{i,t} = R_i - \text{mean}(R)$: group-normalized advantage
- $\varepsilon$: PPO-style clipping range
- $\beta$: KL penalty strength
- $\pi_{\text{ref}}$: reference policy（frozen）

参考: [GRPO原paper](https://arxiv.org/abs/2402.03300), [DeepSeek-R1](https://www.nature.com/articles/s41586-025-08865-9)

### 2.2 Unbiased KL Estimate (公式7)

原版K3 estimator (Schulman, 2020)的KL近似：

$$\hat{D}_{\text{KL}}^{K3}(\pi_\theta \| \pi_{\text{ref}}) \approx \frac{\pi_{\text{ref}}(o_{i,t})}{\pi_\theta(o_{i,t})} - \log\frac{\pi_{\text{ref}}(o_{i,t})}{\pi_\theta(o_{i,t})} - 1$$

问题：当$\pi_\theta(o_{i,t}) \ll \pi_{\text{ref}}(o_{i,t})$时，即当前policy对某token的probability远低于reference，K3的gradient会assign unbounded weight去maximize这个token的likelihood，导致noisy updates。

DeepSeek-V3.2的修正：引入importance sampling ratio $\pi_\theta / \pi_{\text{old}}$：

$$\mathbb{D}_{\text{KL}}(\pi_\theta(o_{i,t}) \| \pi_{\text{ref}}(o_{i,t})) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t}|q, o_{i,<t})} \left(\frac{\pi_{\text{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - \log\frac{\pi_{\text{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - 1\right)$$

**Intuition**: 因为$o_{i,t}$是从$\pi_{\text{old}}$采样的，我们需要在expectation上做importance weighting correction。乘以$\pi_\theta/\pi_{\text{old}}$使得estimator在$\pi_\theta$的expectation下unbiased。这消除了systematic estimation error。

参考: [Schulman KL approximation blog](http://joschu.net/blog/kl-approx.html)

### 2.3 Off-Policy Sequence Masking

工程现实中，rollout data会被split成多个mini-batch做多次gradient update，引入off-policyness。同时inference framework和training framework的implementation差异进一步加剧这个问题。

公式(8-9)引入binary mask $M_{i,t}$：

$$M_{i,t} = \begin{cases} 0 & \hat{A}_{i,t} < 0, \quad \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\log\frac{\pi_{\text{old}}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} > \delta \\ 1 & \text{otherwise} \end{cases}$$

- $\delta$: policy divergence threshold (hyperparameter)
- 第二个条件是sequence-level的average KL between $\pi_{\text{old}}$ and $\pi_\theta$

**Critical asymmetry**: 只mask negative advantage的sequences。Intuition是：
- Positive advantage的sequences即使off-policy，依然提供"what to do"的signal
- Negative advantage的highly off-policy sequences提供"what not to do"的误导signal，因为policy已经移动了，这些negatives可能不再represent当前policy的真实failure mode

### 2.4 Keep Routing (MoE-specific)

MoE models在inference和training framework间的routing inconsistency会导致active parameter subspace突变，破坏optimization stability。

Solution: 在sampling阶段记录每个token的expert routing path，training阶段强制使用相同的routing path。这保证了identical expert parameters被优化。

这个technique从DeepSeek-V3-0324起就采用了，说明是工程实践中发现的关键issue。

### 2.5 Keep Sampling Mask

Top-p/top-k sampling truncates action space，但training时如果$\pi_\theta$的action space与$\pi_{\text{old}}$不同，violates importance sampling principle。

Solution: 保存sampling时的truncation mask，training时apply到$\pi_\theta$，保证两个policy share相同的action subspace。

Empirically，这个策略对维持language consistency很重要。

### 2.6 Post-Training Compute Budget

值得注意的data point: DeepSeek-V3.2的post-training compute budget超过pre-training cost的10%。这是open-source models中罕见的resource allocation。Paper hypothesize reasoning capability还能继续scale。

---

## 3. Thinking in Tool-Use - Agentic能力集成

### 3.1 Thinking Context Management (Figure 4)

DeepSeek-R1的原始策略：每轮new message到来时丢弃reasoning content。这在tool-use场景下token-inefficient，因为每次tool call后都要重新reason整个problem。

DeepSeek-V3.2的规则：
- **Only new user message triggers reasoning discard**
- Tool-related messages (tool outputs) appended时，保留reasoning content
- Reasoning被discard时，tool call history和results依然保留

**Important caveat**: 某些agent framework（Roo Code, Terminus）通过user message模拟tool interaction，这种架构无法benefit from这个机制。Paper建议这些场景用non-thinking model。

### 3.2 Cold-Start策略

三个system prompt template（Appendix Tables 6-8）：

1. **Reasoning data** (Table 6): `
