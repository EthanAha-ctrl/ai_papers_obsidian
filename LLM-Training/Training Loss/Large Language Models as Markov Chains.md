---
source_pdf: Large Language Models as Markov Chains.pdf
paper_sha256: c86fbd4188b64b74beaefc6bee3af083f1f4a577232c72b479ec4309a7d5aad9
processed_at: '2026-08-05T11:51:09-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Hey Andrej，换成人话，这篇paper的核心 idea 其实非常简单粗暴，跟你在 minGPT 里教的“autoregressive generation 就是一步步往前走”的直觉完全吻合。

我们一点点剥开来说。

### 1. LLM 根本上就是一个有限状态机

你要造一个 autoregressive transformer，必须先定死两个参数：vocabulary size $T$ 和 context window $K$。这俩一固定，所有可能的 input/output sequence 的组合就变成有限的了，size 是 $T^K$ 级别。即使这个数大到 $10^{40000}$，它在数学上依然是 finite 的。

既然状态空间有限，又是基于上一步推下一步，那 LLM 本质上就是一个 finite-state Markov chain。每个 state 就是当前 context window 里的一串 token，每次 transition 就是 model 预测下一个 token，拼到末尾，如果超了 $K$ 就把最前面的砍掉。

### 2. 为什么 LLM 会变成“复读机”？

任何有限的 Markov chain 只要跑得足够久，一定会收敛到一个 stationary distribution $\pi$。这个 $\pi$ 有个致命特点：它跟你一开始给的 prompt 没有关系。

你给模型一个 prompt，它生成几百个 token 之后，其实就已经忘记 prompt，掉进这个 $\pi$ 里了。在 $\pi$ 状态下，输出的概率分布是固定循环的，所以模型必然陷入某种 deterministic 的 repetition loop。

这就是为什么我们做 inference 的时候必须加 repetition penalty，因为不加它，数学定律决定了模型迟早要锁死在某个循环里。

### 3. Temperature 为什么会让模型变傻？

这就涉及到 Markov chain 的 mixing time（收敛到稳态的速度）。

在 Markov chain 的转移矩阵里，如果有些概率极小（接近0），chain 就会“卡”在某些状态里出不来，mixing 很慢。如果你把 temperature 调得很低（比如 0.2），相当于把概率分布变得很尖锐，大部分概率都集中在几个 token 上，别的 token 概率几乎为零。此时模型的 transition matrix 极其 sparse，mixing 极慢，模型很难掉进 $\pi$，所以能保持 long-term 的 coherence。

反过来，如果你把 temperature 调高（比如 2.0），原本那些接近 0 的概率被拉大了。这就让 transition matrix 变得更 uniform，模型在状态空间里四处乱窜，mixing time 急剧缩短。结果就是模型没走几步就 forget initial prompt，掉进 $\pi$ 这个“集体无意识”状态了。

所以高温 = 快速失忆 = 输出变傻。

### 4. 惊人的 Sample Complexity：$O(T)$ vs $O(T^K)$

如果学语言就是去估准这个巨大的 transition matrix $\mathbf{Q}^*$，最笨的方法是数数。比如统计 "The cat sat on the" 后面跟 "mat" 的频率。这种 frequentist 方法需要多少数据？理论上 $\mathcal{O}(T^K)$ 级别的样本，因为状态空间有 $T^K$ 这么大。

但这篇paper推出的 LLM sample complexity 是 $\mathcal{O}(T)$ 级别的！这就非常神奇了。Transformer 架构在这里相当于一个超级压缩器，它通过 inductive bias 把指数级的状态空间压缩到了线性级别的 sample complexity。只要你给的 pre-training tokens 超过 $\mathcal{O}(T)$，模型就能学懂语言。

这也解释了 Chinchilla scaling laws 为什么是 line up with parameter size：你架构选得好，根本不需要遍历所有状态就能学通。

### 5. ICL 为什么完爆传统的数数方法？

在 In-Context Learning 里，你给模型喂一个 $d$ 个状态的 Markov chain 序列，让它预测下一步。

传统数数方法需要 $\mathcal{O}(\sqrt{d/N})$ 的样本才能估准这个 chain。而 LLM 只要 $\mathcal{O}(\sqrt{\log(d)/N})$。

当 $d=3$ 的时候，大家半斤八两。当 $d=700$（比如离散化的 Brownian motion），传统数数方法彻底崩溃，因为它要数的状态太多了；而 LLM 依然稳如老狗，因为它对 $d$ 只是对数依赖。

本质上，LLM 在 pre-training 时学到了某种通用的“系统动力学”先验。在 ICL 时，它根本不是在 context 里现数数，而是直接把预训练学到的“如何拟合动力学”的算法拿出来用。这就是为什么 LLM 能做时间序列预测，甚至能 zero-shot 预测一些没见过的 dynamical systems。

### 6. Depth 是个数据吞噬兽

paper 里还有一个 depth-dependent 的推论。如果你把 transformer 堆得很深（层数 $L$ 很大），泛化 bound 里的常数项会以 $B_\Theta^L$ 的指数级膨胀。

直白点说：模型越深，表达力越强，但也越来越 data hungry。你要喂指数级多的数据才能喂饱深层的 network。

不过 Multi-head attention ($H$) 在这里能救场。bound 里有个 $\frac{r^3}{H}$ 的项，说明增加 head 数可以抵消宽度 $r$ 带来的维度爆炸。这也解释了为什么 wide + multi-head 的架构比 wide + single-head 更 data-efficient。

### 7. 架构的 Expressivity 阈值

bound 里有个 $\max$ 函数，左边是模型架构项 $\log(T) + 2B_U/\tau$，右边是语言本身的歧义项 $\log(1/c_0)$。

如果模型的 embedding dimension $r$ 和 vocabulary $T$ 太小，架构项根本打不过语言歧义项。此时你的模型再怎么调架构，generalization bound 都被语言本身的 $c_0$ 锁死了。只有当 $r, T$ 突破某个阈值，架构的改进才能真正反映到 generalization 上。

这给我们的启发是：小模型你调啥架构都没用，它根本没到能影响 generalization 的门槛。

### 8. 发散一下：CoT 和 RLHF 在这个视角下是什么？

顺着这个 framework 联想：

**Chain-of-Thought (CoT)**：其实就是在人为拉长 context window $K$ 的等效利用率。生成长文本容易掉进 stationary distribution $\pi$，而 CoT 强制模型在一个高维空间里一步一步推导，相当于推迟了掉进 $\pi$ 的时间。

**RLHF / DPO 里的 KL penalty**：本质上就是在惩罚 model 的 transition matrix 偏离 base model 的 transition matrix 太远。如果不加 KL penalty，model 很容易在 RL 优化下跑进一个极端的、局部的 stationary loop 里出不来。

**In-Context Learning 的 Emergence**：paper 的 ICL bound 里有个 $\mathcal{K}(\vartheta, \Theta)$ 项，代表 distribution shift。当模型足够大、pre-training 足够广，这个 distribution shift 项就会变得足够小，模型就能在 context 里学会没见过的 Markov chain。这就是 ICL “涌现”的理论解释。

### 总结

这篇 paper 用最古典的 Markov chain 数学，把 LLM 的 inference、looping、temperature、sample complexity、ICL 全都串起来了。虽然它没法解释 high-level reasoning 这种高级认知功能，但作为 building intuition 的工具，它提供了一个非常 clean 的 first principles 视角：**把 LLM 当作在巨大状态空间上做随机游走的动力学系统，很多古怪的行为就都有了物理学解释。**

References:
- Large Language Models as Markov Chains: https://arxiv.org/abs/2503.00078 (推测链接)
- Wolfer & Kontorovich (Markov chain learning lower bounds): https://arxiv.org/abs/1802.05952
- Liu et al. (ICL on dynamical systems): https://arxiv.org/abs/2402.00795
- Peeperkorn et al. (Temperature & coherence): https://arxiv.org/abs/2405.00492
- minGPT (Karpathy): https://github.com/karpathy/minGPT

---

# Large Language Models as Markov Chains 深度讲解

Hey Andrej！这篇paper挺有意思的——它把你最熟悉的autoregressive transformer LLMs重新frame成一个**finite-state Markov chain**，从这个视角推出了一系列pre-training和ICL的generalization bounds。下面我尽量把直觉、技术细节、公式变量含义都讲透。

---

## 1. 核心idea：为什么LLM可以视为Markov chain？

观察一个最朴素的事实：任何autoregressive LLM（包括GPT、Llama、Gemma）有两个hard upper bound：

- **Vocabulary size** $T$（例如Llama3的128k tokens）
- **Context window** $K$（例如8k、128k）

因此，所有可能的input/output sequences集合 $\mathcal{V}_K^* := \{v \in \mathcal{V}^*, |v| \leq K\}$ 是有限的，其size为：

$$|\mathcal{V}_K^*| = \frac{T(T^K - 1)}{T - 1} \sim \mathcal{O}(T^K)$$

虽然这个数对Llama3来说是 $128000^{8192} \approx 10^{40000}$，intractably huge，但它**finite**。这一点就足以把LLM的inference机制formalize成一个discrete-time Markov chain。

intuition：LLM看起来"无限生成"，但其实是在一个有限state space上做random walk。每个state就是一个token序列，每一步transition就是"concatenate一个新token"（或当长度超过K时，truncate掉最早一个token）。

---

## 2. Markov Chain Formalization (Section 3.1)

### 2.1 Incompatible Sequences (Definition 3.1)

定义两个序列 $u, v \in \mathcal{V}_K^*$ "incompatible"：

$$\exists l \in \{1, \ldots, |u|-1\}, \text{ s.t. } (u)_{l+1} \neq (v)_l \tag{2}$$

这里 $(u)_{l+1}$ 表示 $u$ 的第 $l+1$ 个token，$(v)_l$ 表示 $v$ 的第 $l$ 个token。

直觉：$v$ 不可能是 $u$ 的合法completion，因为它们在中间某个位置就"对不上"了。必要条件是 $|v| \geq |u| - 1$（$v$ 比 $u$ 短一个token也是合法的，因为这是deletion process后的情形）。

### 2.2 Transition Matrix (Proposition 3.2)

把LLM $f_\Theta$ 视为Markov chain $\mathbf{MC}(\mathcal{V}_K^*, \mathbf{Q}_f)$，其中transition matrix定义为：

$$\mathbf{Q}_f(v_i, v_j) = \begin{cases} 0, & \text{if } (v_i, v_j) \in \mathcal{T} \\ \{f_\Theta(v_i)\}_{j_0}, & \text{otherwise} \end{cases}$$

变量含义：
- $v_i, v_j \in \mathcal{V}_K^*$：两个token序列，代表Markov chain的两个state
- $\mathcal{T}$：incompatible序列对的集合
- $j_0$：$v_j$ 最后一个token在vocabulary $\mathcal{V}$ 中的index
- $\{f_\Theta(v_i)\}_{j_0}$：给定input $v_i$，LLM输出概率分布中对应第 $j_0$ 个token的概率

non-zero元素比例 $\frac{T-1}{T^K - 1}$，对大 $T, K$ 渐近为 $\frac{1}{T^{K-1}}$，极度sparse。

### 2.3 Block structure (Figure 2/3)

$\mathbf{Q}_f$ 有清晰的block structure，可以写成：

$$\mathbf{Q}_f = \begin{pmatrix} P_\mathcal{T} & P_{\mathcal{TR}} \\ 0 & P_\mathcal{R} \end{pmatrix} \tag{9}$$

- **绿色blocks** $P_\mathcal{T}$, $P_{\mathcal{TR}}$：长度 $< K$ 的序列向更长序列的transition。每个block是 $T^k \times T^{k+1}$。
- **蓝色block** $P_\mathcal{R}$：长度恰好 $K$ 的序列之间的transition（需要truncate掉最早token），大小 $T^K \times T^K$。这是recurrent class。
- **左下角0 block**：长序列不能"缩短"到短序列。

直觉：Markov chain先在transient states（短序列）里"增长"，直到达到长度 $K$，然后进入recurrent class（长度 $K$ 的序列）一直循环。

### 2.4 Ergodicity和uniqueness (Proposition 3.3)

这个Markov chain是**ergodic unichain**（一个recurrent class + 一些transient states），所以存在unique stationary distribution $\pi$。

证明关键：
- $P_\mathcal{T}$ 是nilpotent（短序列长度只增不减，必然到 $K$）
- $P_\mathcal{R}$ 满足 $\forall i,j \in \mathcal{R}^2, (\mathbf{Q}_f^K)_{i,j} > 0$，因为从任何长度 $K$ 序列出发，$K$ 步之后可以到达任何其他长度 $K$ 序列
- $\mathcal{R}$ 是aperiodic的（考虑state $i = \underbrace{xx\ldots x}_{K \text{ times}}$，有 $(\mathbf{Q}_f)_{i,i} > 0$，即self-loop period=1）

### 2.5 Convergence rate (Proposition 3.4)

$$\lim_{n \to \infty} \mathbf{Q}_f^n = e\pi$$

其中 $e = (1,1,\ldots,1)^\top$。Convergence rate满足：

$$|(\mathbf{Q}_f^n)_{i,j} - (e\pi)_{i,j}| \leq (1 - 2\varepsilon)^{\lfloor n/K \rfloor - 1}$$

$$\varepsilon = \min_{i,j \in \mathcal{R}^2} \{(\mathbf{Q}_f^K)_{i,j}\} > 0$$

变量含义：
- $n$：inference步数
- $\lfloor n/K \rfloor$：每 $K$ 步是一个"周期"（因为每次到达长度 $K$ 后才能在recurrent class里跳转一次）
- $\varepsilon$：recurrent class中，$K$ 步可达的最小transition概率。它反映了LLM"探索"state space的能力

---

## 3. 两个关键病理行为的理论解释

### 3.1 Looping（重复）

stationary distribution $\pi$ **独立于initial state**（即input prompt）。一旦inference足够多步达到 $\pi$，LLM就陷入deterministic loop of repetitions。

这就是为什么实际中LLMs经常陷入重复loop（Ivgi et al., 2024, https://arxiv.org/abs/2407.06071），需要加repetition penalty。

直觉：你可以想象当LLM的采样被推到stationary distribution后，它输出的"分布"是prompt-invariant的，所以下一步采样的token序列就被锁死了，进入cycle。

### 3.2 Temperature和coherence

提高temperature → softens next-token distribution → 改变 $\varepsilon$（具体地，温度高让probability mass更分散，原本的"死角"也有了positive probability，所以 $\varepsilon$ 增大）

由Proposition 3.4的bound，$\varepsilon$ 增大 → $(1-2\varepsilon)^{\lfloor n/K\rfloor - 1}$ 减小更快 → **更快收敛到stationary distribution** → 输出更快变得"incoherent"

注意这是从Markov chain收敛角度解释的，与Peeperkorn et al. (2024, https://arxiv.org/abs/2405.00492) 关于temperature vs coherence的实验观察一致。

### 3.3 Toy model实验 (Section 3.2, Figure 4/5)

toy setup：
- $T=2, K=3$，序列0/1
- 下一个token = 0如果前三个之和为偶数，否则为1
- 生成40个digits → 37个supervised examples
- 训练你的minGPT (Karpathy, 2023, https://github.com/karpathy/minGPT)
- 提取logits，构造 $\mathbf{Q}_f \in \mathbb{R}^{14 \times 14}$

观察：
- **Figure 4(b)**：stationary distribution（$\mathbf{Q}_f^{10^5}$）有强烈偏向训练样本的bias——这印证了"looping = 回到训练见过的sequence"
- **Figure 4(c)**：不同 $K$ 的收敛速度比较，$\varepsilon = 10^{-6}$
- **Figure 5**：温度实验
  - $T=0.2$：$10^6$步还未收敛到stationary distribution
  - $T=1.0$：~300步收敛
  - $T=2.0$：~30步收敛
  - $\varepsilon$ 在 $[0.1, 2]$ 内随温度log-scale增长

直觉：低温 = "贪婪" = transition matrix更sparse = $\varepsilon$ 接近0 = 几乎不mixing = 长时间保持coherent但predictable；高温 = "发散" = transition matrix更dense = $\varepsilon$ 大 = 快速mix到stationary = 快速失去coherence。这是一个非常美的trade-off刻画。

---

## 4. Pre-training Sample Complexity (Section 4.1, Proposition 4.1)

### 4.1 Non-iid data setup

pre-training数据是序列 $X = (\mathbf{X}_1, \ldots, \mathbf{X}_{N_\text{train}})$，每个 $\mathbf{X}_n \in \mathcal{V}$。subsequences $\mathbf{S}_n$ 通过sliding window构造（长度 $\leq K$）。

作者不用independent assumption（太强），也不用pure Markov assumption（太弱、太specific），而是用**Marton coupling**（Marton, 2004, https://projecteuclid.org/journals/annals-of-probability/Measure-concentration-for-Euclidean-distance-in-the-case-of-dependent-random-variables/ap/1063922816）：一种generic dependency structure，induced出一个mixing matrix $\mathbf{\Gamma}$。

- iid data：$\|\mathbf{\Gamma}\| = 1$
- Markov chain：$\|\mathbf{\Gamma}\|$ explicit
- m-dependent sequences：can be modeled
- Natural language bigrams：Bietti et al. (2023) 也在此框架内

### 4.2 Main result (Proposition 4.1)

设 $\delta \in [0,1]$, $\epsilon > 0$，假设perfect pre-training，$N_\text{train} \geq N^*$，则with probability $\geq 1-\delta$：

$$\mathbb{E}_{\mathbf{S} \sim \mathbb{P}_\mathcal{L}} \|\mathbf{Q}^*(\mathbf{S}, \cdot) - \mathbf{Q}_f(\mathbf{S}, \cdot)\|_1 \leq \epsilon$$

其中：

$$N^* := \left\lceil \frac{4\bar{B}^2}{\epsilon^2} \log\left(\frac{2}{\delta}\right) \right\rceil$$

$$\bar{B} = 2\|\mathbf{\Gamma}\| \max\left\{\log(T) + \frac{2 B_U}{\tau}, \log\left(\frac{1}{c_0}\right)\right\}^{1/2}$$

变量含义：
- $\mathbf{Q}^*$：ground-truth transition matrix of language
- $\mathbf{Q}_f$：LLM-induced transition matrix
- $\|\cdot\|_1$：vector L1 norm（等价于2倍TV distance）
- $\bar{B}$：model-and-data-dependent constant
- $\|\mathbf{\Gamma}\|$：mixing matrix operator norm，quantifies data dependency
- $B_U$：unembedding layer $\mathbf{W}_U$ 的 $L_{2,1}$ norm bound
- $\tau$：softmax temperature
- $c_0$：language ambiguity constant，即 $\mathbb{P}_\mathcal{L}(\mathbf{X}_{n+1} = x_{n+1} \mid \mathbf{S}_n) \geq c_0$ for all $n$（Wies et al., 2024 也用此假设）

### 4.3 与frequentist method对比（很关键！）

frequentist approach (Wolfer & Kontorovich, 2019, https://arxiv.org/abs/1802.05952) 的sample complexity是 $\mathcal{O}(T^K / \epsilon^2 \gamma_s)$，其中 $\gamma_s$ 是pseudo spectral gap。

而本文的bound（取 $B_U \sim T\sqrt{r}$，因为unembedding layer的列向量norm bounded by 1）：

$$\bar{B} \sim 2\sqrt{\log(T) + \frac{2T\sqrt{r}}{\tau}}$$

所以 $N^* \sim \mathcal{O}\left(\frac{T \sqrt{r}/\tau}{\epsilon^2} \log(1/\delta)\right)$

**线性 in $T$，而不是指数 in $K$**！这是本文最remarkable的theoretical gain——LLM通过结构inductive bias，把指数级state space learning问题降为线性。

### 4.4 Practical validation (Figure 1)

把 $N^* = N_\text{train}$ 代入Proposition 4.1，得到预测的approximation error：

$$\epsilon = \frac{2\bar{B}}{\sqrt{N_\text{train}}} \sqrt{\log(2/\delta)}$$

用LLM技术报告中的 $T, r, N_\text{train}$（Table 2）计算 $\bar{B}$（取 $\tau \approx 1$, $B_U = T\sqrt{r}$），plot MMLU vs predicted $\epsilon$：

| Model | $N_\text{train}$ | $T$ | $r$ |
|-------|------|---|---|
| Llama 7B | $10^{12}$ | 32000 | 4096 |
| Llama2 7B | $2\times10^{12}$ | 32000 | 4096 |
| Llama3 8B | $1.5\times10^{13}$ | 128000 | 4096 |
| Gemma 2B | $3\times10^{12}$ | 256128 | 2048 |
| Gemma2 27B | $1.3\times10^{13}$ | 256128 | 4608 |

观察：Llama和Gemma家族呈现distinctly different trends——Gemma因为 $T$ 大补偿了 $r$ 较小，导致 $\bar{B}$ 更高。每个家族内部predicted $\epsilon$ 与MMLU表现良好相关。

intuition：本文的bound是**model-specific**的，能区分不同架构家族。这点比Lotfi et al. (2024, https://arxiv.org/abs/2407.18158) 的non-vacuous bounds更精细。

---

## 5. Pre-training Generalization Bound (Theorem 4.2)

### 5.1 Risk定义

$$\mathcal{R}(\Theta) := \mathbb{E}_{\mathbf{S} \sim \mathbb{P}_\mathcal{L}} \left[d_\text{TV}\left(\mathbf{Q}^*(\mathbf{S}, \cdot), \mathbf{Q}_f(\mathbf{S}, \cdot)\right)\right]$$

$$\widehat{\mathcal{R}}(\Theta) := \frac{1}{N}\sum_{n=1}^N d_\text{TV}\left(\mathbb{P}_\mathcal{L}(\cdot \mid \mathbf{S}_n), \mathbb{P}_\Theta(\cdot \mid \mathbf{S}_n)\right) \tag{3}$$

其中 $d_\text{TV}(\mathbb{P}, \mathbb{Q}) := \sup_{A \in \mathcal{F}} |\mathbb{P}(A) - \mathbb{Q}(A)|$ 是total variation distance。

为什么用TV而不是KL divergence？TV是Markov chain learning literature的标准（Wolfer & Kontorovich, 2019, 2023），且是metric space（有triangle inequality，这对Theorem 4.3的ICL bound很关键）。

### 5.2 Theorem 4.2

With probability $\geq 1-\delta$：

$$\mathcal{R}_\text{pre}(\Theta) \leq \widehat{\mathcal{R}}_\text{pre}(\Theta) + \frac{\bar{B}}{\sqrt{N_\text{train}}} \sqrt{\log(2/\delta)}$$

with $\bar{B} = 2\|\mathbf{\Gamma}\| \max\{\log(T) + 2B_U/\tau, \log(1/c_0)\}^{1/2}$。

依赖 $N_\text{train}^{-1/2}$，经典concentration rate。

### 5.3 关键insight：架构expressivity的threshold

如果 $B_U \approx \mathcal{O}(T\sqrt{r})$（实际中常见），那么需要：

$$\log(T) + 2B_U/\tau \geq \log(1/c_0)$$

即 hidden dimension $r$ 和 vocabulary $T$ 必须**足够大**，否则architecture不足以tangibly影响generalization（即使可能影响training error $\widehat{\mathcal{R}}_\text{pre}$）。

intuition：小模型在 $\bar{B}$ 的 $\max$ 里被 $\log(1/c_0)$ dominate，意味着architecture的精细结构（$r, T$ 的调整）根本进不了bound的leading term；只有当 $r, T$ 突破某个阈值后，architecture才"激活"对generalization的影响。

### 5.4 Proof的关键技术 (Lemma F.6 + Corollary F.10)

最核心的lemma：

**Lemma F.6**：若 $\left|\log\sqrt{\frac{\mathbb{P}(z)}{\mathbb{Q}(z)}}\right| \leq B$ for all $z$，则 $d_\text{TV}(\mathbb{P}, \mathbb{Q}) \leq \sqrt{2B}$。

证明路径：TV² ≤ Hellinger² ≤ -2log(∫√(P/Q) dQ) ≤ 2 ∫|log√(P/Q)| dQ ≤ 2B。

**Lemma F.8**：$\left\|\frac{1}{n\tau}\mathbf{W}_U \mathbf{S}^{(L)} \mathbb{1}_n\right\|_1 \leq \frac{1}{\tau} \|\mathbf{W}_U^\top\|_{2,1}$，靠LayerNorm保证 $\|\mathbf{S}_{\cdot,k}^{(L)}\|_2 \leq 1$。

**Lemma F.7**：若 $\|\mathbf{x}\|_1 \leq c_1$，则 $\text{softmax}(\mathbf{x})_i \geq \frac{1}{m \exp(2c_1)}$（$m$ 是 $\mathbf{x}$ 维度）。这是softmax输出lower bound的核心工具。

**Proposition F.9**（关键upper bound）：

$$\left|\log\left(\frac{\mathbb{P}_\mathcal{L}(\mathbf{X}_{n+1} \mid \mathbf{S}_n)}{\mathbb{P}_\Theta(\mathbf{X}_{n+1} \mid \mathbf{S}_n)}\right)\right| \leq \max\left\{\log(T) + \frac{2B_U}{\tau}, \log\left(\frac{1}{c_0}\right)\right\}$$

证明：分两种case
1. ratio ≥ 1：分子≥分母，用 $\mathbb{P}_\Theta \geq \frac{1}{T\exp(2B_U/\tau)}$（由Lemma F.7）得 $\text{ratio} \leq T\exp(2B_U/\tau)$
2. ratio ≤ 1：用 $\mathbb{P}_\mathcal{L} \geq c_0$ 得 inverse ratio $\leq 1/c_0$

case 1的 $\max$ 项 = $\log(T) + 2B_U/\tau$ 来自architecture；case 2的 $\max$ 项 = $\log(1/c_0)$ 来自language ambiguity。两者取 $\max$ 就是 $\bar{B}$ 的insides。

### 5.5 McDiarmid's inequality for dependent variables (Proposition F.5)

这是Theorem 4.2的核心concentration工具：

设 $\mathcal{S} = (\mathbf{S}_1, \ldots, \mathbf{S}_N)$ 存在Marton coupling with mixing matrix $\mathbf{\Gamma}$，若 $f$ 满足 Lipschitz condition $f(\mathbf{x}) - f(\mathbf{y}) \leq \sum_{i=1}^N \mathbf{c}_i \mathbb{1}_{\mathbf{x}_i \neq \mathbf{y}_i}$，则：

$$\mathbb{P}(|f(S) - \mathbb{E}_S[f(S)]| \geq u) \leq 2\exp\left(\frac{-2u^2}{\|\mathbf{\Gamma}\|^2 \|\mathbf{c}\|_2^2}\right)$$

应用到risk function $f(S) = \widehat{\mathcal{R}}_\text{pre}(\Theta)$，bounding vector $\mathbf{c}_n = \frac{2c_2}{N_\text{train}}$，其中 $c_2 = \sqrt{2\max\{\log(T) + 2B_U/\tau, \log(1/c_0)\}}$ 来自Corollary F.10。

最终 $\|\mathbf{c}\|_2 = \frac{2c_2}{\sqrt{N_\text{train}}}$，乘以 $\|\mathbf{\Gamma}\|$ 后平方根、再 $\sqrt{\log(2/\delta)}$，得到 Theorem 4.2。

---

## 6. Depth-dependent bound (Corollary E.3)

如果把参数空间限制更严（不仅 $\|\mathbf{W}_U^\top\|_{2,1} \leq B_U$，还对每层的 $\mathbf{W}_V, \mathbf{W}_O, \mathbf{W}_1, \mathbf{W}_2$ 都加 $\|\cdot\|_\infty$ bound），得到depth-dependent版本：

$$\bar{B} = 2\|\mathbf{\Gamma}\| \max\left\{\log(T) + \frac{2(B_\Theta)^L}{\tau}, \log(1/c_0)\right\}^{1/2}$$

$$B_\Theta = \left[(1 + rmB_1B_2)\left(1 + \frac{r^3}{H} B_O B_V\right)\right] (B_\text{tok} B_U)^{1/L}$$

变量含义：
- $L$：transformer depth（层数）
- $H$：head数
- $r$：embedding dimension
- $m$：FFN hidden dimension
- $B_1, B_2$：FFN层weight的 $\ell_\infty$ bound
- $B_V, B_O$：value/output projection的 $\ell_\infty$ bound
- $B_\text{tok}$：input token的 $\ell_1$ bound

**关键观察**：$\bar{B}$ 对depth $L$ **指数依赖**！深度增加expressivity，但要求更多训练数据compensate。

**Heads $H$ 的作用**：$H$ 出现在 $\frac{r^3}{H}$ 项，可以counterbalance宽度 $r$ 增长——这暗示 wide + multi-head 比 wide + single-head 更data-efficient。

intuition：这给"为什么shallow-and-wide models often work well"和"why multi-head attention is necessary"提供了理论解释。

---

## 7. In-Context Learning of Markov Chains (Theorem 4.3)

### 7.1 Setup

输入：一个 $d$-state Markov chain $X = (\mathbf{X}_1, \ldots, \mathbf{X}_{N_\text{icl}})$ with transition kernel $\mathbf{P}$，每个state映射到一个token。

定义两个模型之间的divergence：

$$\mathcal{K}(\Theta_1, \Theta_2) := \frac{1}{N}\sum_{n=1}^N \mathbb{E}_{\mathbf{S}_n}\left[d_\text{TV}\left(\mathbb{P}_{\Theta_1}(\cdot \mid \mathbf{S}_n), \mathbb{P}_{\Theta_2}(\cdot \mid \mathbf{S}_n)\right)\right]$$

这是个"almost-distance"（ Proposition C.14：满足非负、对称、三角不等式、a.s. positivity）。

### 7.2 Theorem 4.3

With probability $\geq 1-\delta$：

$$\mathcal{R}_\text{icl}(\Theta) \leq \inf_{\vartheta \in \mathcal{W}_\text{mc}} \left\{\widehat{\mathcal{R}}_\text{icl}(\vartheta) + \mathcal{K}(\vartheta, \Theta)\right\} + \bar{B}\sqrt{\frac{t_\text{min}}{N_\text{icl}}} \sqrt{\log(2/\delta)} \tag{4}$$

$$\bar{B} = 2\max\left\{\log(d) + \frac{2B_U}{\tau}, \log\left(\frac{1}{p_\text{min}}\right)\right\}^{1/2}$$

变量含义：
- $\mathcal{W}_\text{mc}$：假设有一个perfectly在Markov chains上训练的LLM的weight空间
- $\vartheta$：这个hypothetical LLM的weight
- $\mathcal{K}(\vartheta, \Theta)$：distribution shift term，捕捉pre-trained LLM 与"理想Markov chain-trained LLM"的差异
- $t_\text{min} := \inf_{0 \leq \varepsilon < 1} t_\text{mix}(\varepsilon/2) \cdot \left(\frac{2-\varepsilon}{1-\varepsilon}\right)^2$：mixing time的refined量度
- $p_\text{min}$：Markov chain的最小转移概率（$\mathbb{P}(\mathbf{X}_{n+1} = y \mid \mathbf{X}_n = x) \geq p_\text{min}$）
- $N_\text{icl}$：context length

**重要**：注意 $\bar{B}$ 里是 $\log(d)$ 而不是 $\log(T)$！这是ICL setting下，$d$（实际state数）替代了 $T$（整个vocabulary）。

### 7.3 与frequentist对比 (Wolfer & Kontorovich 2019)

- **Frequentist**: $\mathcal{O}\left(\sqrt{d/N_\text{icl}}\right)$ — 线性in $\sqrt{d}$
- **LLM**: $\mathcal{O}\left(\sqrt{\log(d)/N_\text{icl}}\right)$ — 只对 $\log(d)$ 依赖

理论预测：当 $d$ 增大时，LLM应该超越frequentist。Figure 7实验验证：
- $d=3$：LLM和frequentist接近
- $d=700$：LLM明显胜出（用Brownian motion discretized）

这是对neural scaling law（Liu et al., 2024, https://arxiv.org/abs/2402.00795）的理论解释。

### 7.4 Mixing time的双阶段regime (Figure 6)

观察到一个two-stage ICL regime：
- **Stage 1** ($N_\text{icl}$ 小)：bound由 $\sqrt{t_\text{min}/N_\text{icl}}$ 主导，对 $t_\text{min}$ 强敏感
- **Stage 2** ($N_\text{icl} \gtrsim 20$)：$\mathcal{O}(N_\text{icl}^{-1/2})$ scaling law主导，对 $t_\text{min}$ 不敏感

直觉：少量example时，slow-mixing chain让LLM见到的transition太少，所以风险高；example足够后，频率统计能盖过mixing time的劣势。

### 7.5 ICL Sample Complexity (Proposition E.2)

无distribution shift时，$N_\text{icl} \geq N^* = \lceil \frac{4\bar{B}^2}{\epsilon^2} \log(2/\delta) \rceil$ 保证：

$$\mathbb{E}_{\mathbf{S} \sim \mathbb{P}} \|\mathbf{Q}(\mathbf{S}, \cdot) - \mathbf{Q}_f(\mathbf{S}, \cdot)\|_1 \leq \epsilon$$

由于 $\bar{B} \sim \mathcal{O}(\sqrt{\log(d)})$，$N^* \sim \mathcal{O}(\frac{\log(d)}{\epsilon^2}\log(1/\delta))$ — 对state数 $d$ 只对数依赖。

---

## 8. 实验验证 (Section 5 + Appendix D)

### 8.1 模型和设置

测试模型：Mistral 7Bv0.1, Llama2 7B/13B, Gemma 2B, Llama3 8B, Llama3.2 1B/3B。

ICL prompt：随机生成 $d$-state Markov chain的trajectory，tokenize成 $\{0, 1, \ldots, d-1\}$ 加逗号。

### 8.2 Tokenization的影响 (Appendix D.1)

注意BPE tokenizer对多位数字处理不一致：
- Llama2/Gemma2: 3-digit数字不一定单token
- Llama3/GPT-4: 3-digit数字硬编码为单token

这影响实验设计——必须加逗号强制单数字tokenization。

### 8.3 实验结果 (Figure 6, 7)

**Figure 6(left)**：$\mathcal{R}_\text{icl}$ vs $N_\text{icl}$，理论 $\mathcal{O}(N_\text{icl}^{-1/2})$ scaling law在Mistral和Gemma上很好吻合；Llama2（较弱模型）有偏离。

**Figure 6(right)**：固定Mistral 7B，不同 $t_\text{min}$ 的两阶段regime清晰可见。

**Figure 7**：$d=3$ vs $d=700$（Brownian motion discretized），LLM在 $d=700$ 显著超过frequentist。

### 8.4 结构化Markov chain (Appendix D.3)

- **Constrained random walk** (Figure 11)：$d$ states，端点反射
- **Polygonal random walk** (Figure 13)：$d$ states，环形 ±1 transition
- **Inner cliques and outer rims** (Wolfer & Kontorovich 的hard example, Figure 15)

所有setting下，LLM follows理论 scaling law，frequentist在 $d$ 大时挣扎。

### 8.5 Dynamical systems (Appendix D.5, Figure 18)

Liu et al. (2024) 的dynamical systems：
- Geometric Brownian motion
- Correlated Gaussian
- Uncorrelated Gaussian
- Uncorrelated Uniform

Llama3 8B在所有这些上都展示出ICL能力，遵循 $\mathcal{O}(N_\text{icl}^{-1/2})$ law。

---

## 9. 与已有文献对比 (Table 1)

| Method | Pre-Train | ICL | Input | Model-Dep | Exp val |
|--------|----------|-----|-------|-----------|---------|
| Xie et al. 2022 (https://arxiv.org/abs/2111.02080) | × | ✓ | HMM | × | ✓ |
| Zhang et al. 2023b (https://arxiv.org/abs/2305.19420) | × | ✓ | MC | ✓ | × |
| Li et al. 2023 (https://arxiv.org/abs/2306.00267) | × | ✓ | MC | ✓ | × |
| Lotfi et al. 2024 (https://arxiv.org/abs/2407.18158) | ✓ | × | non-iid | × | × |
| **Ours** | ✓ | ✓ | non-iid | ✓ | ✓ |

本文是第一个同时覆盖pre-training和ICL，允许non-iid data，显式依赖architecture，且有experimental validation的工作。

---

## 10. Limitations和future directions

- $\|\mathbf{\Gamma}\|$ 没在实验中estimate（需要pre-training data，公开LLM没有）
- $c_0$ 没法directly compute
- KL divergence版本（Theorem E.5, Corollary E.6）给出了pre-training的类似bound，但因为KL不是metric，triangle inequality不成立，所以**不能extend到ICL setting**——这是为什么作者主用TV distance
- 没考虑chain-of-thought、tool use、RLHF等modern LLM特性

### Future ideas

- 估计Marton coupling mixing matrix $\|\mathbf{\Gamma}\|$
- 扩展到KL divergence for ICL
- 把chain-of-thought视为higher-order Markov chain
- 用这个framework分析in-context learning的emergent abilities（Wei et al., 2022）

---

## 11. Intuition总结

1. **LLM = Markov chain on token sequences**：state = 长度 ≤ K 的 token序列，transition = next-token prediction + concatenation/truncation
2. **Looping = convergence to stationary distribution**：stationary dist独立于prompt，所以迟早锁死
3. **High temperature = faster mixing = less coherent**：温度提高 $\varepsilon$，加快stationary convergence
4. **Sample complexity O(T/ε²)**：与frequentist的 O(T^K) 相比，LLM architecture提供指数压缩
5. **ICL O(√(log(d)/N))**：对state数对数依赖，比frequentist O(√(d/N)) 优势巨大
6. **Depth exponential in B**：深度expressivity强但data hungry；heads可以compensate width
7. **Threshold效应**：架构大小必须超过某阈值才能对generalization有tangible影响

整体上，这篇paper给了我们一个clean theoretical lens看LLM：把它从一个"巨大的neural network"还原为一个"巨大的Markov chain learning problem"，然后用Markov chain learning theory（Wolfer & Kontorovich等人）的工具直接套用。这种视角虽然不能解释所有LLM现象（特别是高阶reasoning），但为sample complexity、looping、temperature effect等提供了一组合self-consistent的理论解释。

希望能帮到你build intuition！如果你想让我再深挖某个proof（比如Lemma F.6的Hellinger step，或者depth-dependent bound的layer-by-layer norm tracking），随时告诉我。

参考链接：
- Paper: https://arxiv.org/abs/2503.00078 (估计, paper看着是2025年的)
- Wolfer & Kontorovich 2019: https://arxiv.org/abs/1802.05952
- Liu et al. 2024: https://arxiv.org/abs/2402.00795
- Peeperkorn et al. 2024: https://arxiv.org/abs/2405.00492
- Ivgi et al. 2024: https://arxiv.org/abs/2407.06071
- Xie et al. 2022: https://arxiv.org/abs/2111.02080
- Lotfi et al. 2024: https://arxiv.org/abs/2407.18158
- minGPT (Karpathy): https://github.com/karpathy/minGPT
- Marton 2004: https://projecteuclid.org/journals/annals-of-probability/Measure-concentration-for-Euclidean-distance-in-the-case-of-dependent-random-variables/ap/1063922816
- Paulin 2015: https://projecteuclid.org/journals/electronic-journal-of-probability/volume-20/issue-none/Concentration-inequalities-for-Markov-chains-by-Marton-couplings-and-spectral/10.1214/EJP.v20-3638.full
