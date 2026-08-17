---
source_pdf: MemGen WeavingGenerativeLatentMemoryfor Self-Evolving Agents.pdf
paper_sha256: f4565d36adbf927d4aff8bd025606af23c94c16cbc61c1fa4730650ddc1d977e
processed_at: '2026-08-05T17:23:47-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MemGen 用人话说

## 先讲个故事

你有一个特别聪明的大学生，叫Reasoner。他什么都会一点，但有个问题：他从不记住自己的经验。

他第一次做某道数学题，错了。第二次遇到类似的题，还是错。你问他为什么？他说："我这学期刚开学，脑子里一片空白。"

这就是vanilla LLM agent的困境。

## 现有的两种解法

**方法A：直接给他动手术改脑子**（parametric memory）

你把他按在手术台上，打开头盖骨，用microcurrent刺激某些neuron。他学会了这道题，但代价是——他突然忘了怎么做菜、怎么骑自行车、怎么说法语。

这叫**catastrophic forgetting**。GRPO、SFT、REINFORCE++都是这个套路，越学越像偏科生。

**方法B：给他一个笔记本**（retrieval-based memory）

你说："别动脑子了，我给你个笔记本，把经验写下来。"每次做新题前，你翻翻笔记本，把相关页撕下来贴在他脑门上。

问题来了：
1. 他做题做到一半，遇到新的卡壳点，笔记本帮不上忙（你只在开始时贴一次）
2. 撕下来的页和他脑子里的思考方式对不上号，他得在两个representation之间来回翻译，很别扭
3. 小脑门的容量有限，贴太多页他反而看不过来，被噪音干扰

ExpeL、AWM、MemoryBank都是这个路子。

## MemGen的思路

回到大学生的故事。你不改他的脑子，也不给他笔记本。你给他雇两个助手。

**助手A（Memory Trigger）**：坐在他旁边，盯着他的眼神。当他开始皱眉、停顿、犹豫——总之就是"思考卡壳了"的信号——助手A就按一下铃。

**助手B（Memory Weaver）**：听到铃响，立刻冲上前。他不是给他念笔记本（太慢、太脱节），而是**当场编一段"咒语"**——这段咒语只有Reasoner能听懂，是Reasoner的native dialect。助手B把这段咒语低语给他听，Reasoner瞬间开窍，继续解题。

注意几个关键点：
- **什么时候按铃**由助手A决定，他学会了"看脸色"
- **念什么咒语**由助手B现编，不是从本子上抄
- **Reasoner的脑子从未被改过**——他不偏科了
- 咒语是**机器native的**，人类听不懂，但对Reasoner来说是precise的instruction

## 拆开来看

### Reasoner = frozen LLM
就是一个普通的Qwen或者SmolLM，参数$\theta$冻住，永远不动。它autoregressively吐token，吐到第$j$个位置时，已经吐了$\mathbf{z}_{t,<j}$这串token，对应hidden states $\mathbf{H}_{t,<j}$。

### Trigger = 一个小LoRA + Bernoulli采样
Trigger也是挂在同一个LLM上的LoRA。它看$\mathbf{H}_{t,<j}$，吐一个probability $p_j$：

$$p_j = \sigma(\mathcal{T}_{trigger}(\mathbf{h}_{t,1}, \dots, \mathbf{h}_{t,j-1}))$$

- $\sigma$就是sigmoid，把任意实数压到$(0,1)$区间
- $\mathbf{h}_{t,k}$是第$k$个token位置的hidden state，$d_{model}$维向量
- 输出$p_j$意思是"现在这个时刻，该不该叫助手B的概率"

然后从$\text{Bernoulli}(p_j)$里采样一个0或1。1就是INVOKE，0就是SKIP。

**省钱技巧**：不是每个token都判断，只在逗号、句号这种语义边界判断。理由是人类思考也是一句一句思考的，在句子中间插入memory会很怪。

### Trigger怎么训练的

这是paper最巧妙的地方。目标函数（公式8）：

$$\max_\phi \mathbb{E}\left[R(\tau_i) - \lambda \sum_{i,j} \max(0, \tilde{d}_{i,j} - \bar{p})\right]$$

- $\phi$：Trigger的LoRA参数
- $R(\tau_i)$：trajectory的reward，做对了得分
- $\tilde{d}_{i,j}$：第$i$条trajectory的第$j$个激活点上，trigger实际做出的0/1决策
- $\lambda$：惩罚系数
- $\bar{p}$：**关键变量**——"好学生平均按铃频率"

$\bar{p}$怎么算的（公式9）？把batch里reward超过median的trajectory挑出来，这些是"好学生"，看他们平均按了多少次铃，这个平均值就是$\bar{p}$。

惩罚项$\max(0, \tilde{d}_{i,j} - \bar{p})$的意思：如果你按铃频率超过了好学生的平均水平，就罚你。

**直觉**：让Trigger学会"像好学生一样节制地按铃"。按太多反而干扰思考，按太少又错过关键时刻。这个$\bar{p}$是adaptive的——简单任务好学生按得少，$\bar{p}$低；难任务好学生按得多，$\bar{p}$高。

### Weaver = 另一个LoRA
Weaver也是挂在同一LLM上的LoRA。它接收同样的$\mathbf{H}_{t,<j}$，吐出一个矩阵$\mathbf{M}_t$：

$$\mathbf{M}_t = \mathcal{W}_{weaver}(\mathbf{H}_{t,<j}) \in \mathbb{R}^{K \times d_{model}}$$

- $K$：latent memory的长度，paper里试了2、4、8，到32还在提升
- $d_{model}$：每个latent token的维度，和LLM的hidden state同维
- $\mathbf{M}_t = [\mathbf{m}_{t,1}, \dots, \mathbf{m}_{t,K}]$是$K$个latent token组成的sequence

然后这$K$个latent token被prepend到$\mathbf{H}_{t,<j}$前面，Reasoner接着生成下一个token（公式6）：

$$\mathbf{z}_{t,j} \sim \pi_\theta(\cdot \mid s_t, \mathbf{z}_{t,<j}, \mathbf{M}_t)$$

### Weaver怎么训练的

两种方式：SFT和GRPO。

**SFT版本**：你有expert trajectories $\tau_i^* = (s, a^*, s, a^*, \dots)$，让Reasoner在Weaver的辅助下，尽量生成和expert一样的action。loss就是negative log-likelihood：

$$\mathcal{L}_{SFT} = -\mathbb{E}\left[\sum \log \pi_\theta(\mathbf{z}^* \mid s, \mathbf{z}^*_{<j}, \mathbf{M}_{i,t,j})\right]$$

梯度只更新Weaver的$\theta'$，不动Reasoner的$\theta$。

**GRPO版本**：sample $K$条trajectory，算group-relative advantage $A(\tau) = R(\tau) - \bar{R}$，然后policy gradient：

$$\mathcal{I}_{GRPO} \propto A(\tau) \log \Pi(\tau \mid x) - \beta \text{KL}$$

也是只更新Weaver。GRPO的intuition：不需要expert，让agent自己rollout，做得比平均好的trajectory，强化；差的，弱化。

## 最惊艳的实验结果

### 结果1：性能暴涨
ALFWorld（一个embodied agent任务）+ SmolLM3-3B：
- Vanilla：18.96%
- 最强的retrieval方法AWM：40.50%
- 最强的parametric方法GRPO：55.35%
- **MemGen_GRPO：63.60%**

提升44个点，碾压所有baseline。

### 结果2：Cross-domain自动适应
在GSM8K（数学）上训练，然后在3个domain上看Trigger的按铃频率（Figure 4）：
- GSM8K自己：按铃最多，性能+19.64%
- GPQA（科学）：按铃中等，+6.06%
- KodCode（代码）：按铃最少，+3.1%

**Trigger自己学会了"我什么时候需要回忆"**。在熟悉的domain（数学）多回忆，在不熟悉的domain（代码）少回忆避免干扰。这是meta-cognition的emergence。

### 结果3：Catastrophic forgetting大幅缓解
Sequential training：AQuA → GPQA → GSM8K → KodCode

训练完KodCode后，回头看AQuA的成绩：
- SFT：28.61%（基本忘了AQuA）
- ExpeL：27.14%
- **MemGen：40.34%**（保留了大部分）

为什么？因为Reasoner frozen，所有"记忆"都编进了Weaver的LoRA里。LoRA的low-rank constraint本身就有regularization效果，不会把old knowledge overwrite掉。

### 结果4：推理反而变快了
KodCode + Qwen2.5-1.5B：
- Vanilla：11.96秒，24.55%准确率
- SFT：2.01秒，55.83%
- MemGen：2.94秒，58.16%

MemGen比vanilla快4倍，准确率还高33个点。原因是MemGen让agent少走弯路，更快reach correct answer，生成更少token。

## 最迷的发现：Memory Hierarchy自发涌现

这是paper最神的part。作者想知道：Weaver吐出来的那些latent token，到底在干什么？

他们用K-means把所有latent memory聚类成4个cluster，然后做**ablation experiment**——推理时故意把某个cluster的memory去掉，看agent在哪些failure mode上变差。

Failure mode有8种：planning failure、tool parsing error、answer formatting failure等等。

结果（Figure 6 right）：

- **去掉Cluster 2** → planning failure和compositional reasoning大幅增加 → 这是**Planning Memory**
- **去掉Cluster 3** → tool response error、parsing failure、formatting mistake增加 → 这是**Procedural Memory**（管工具调用、格式这些"操作技能"）
- **去掉Cluster 1, 4** → task misunderstanding、think-act inconsistency增加 → 这是**Working Memory**（管上下文一致性）

**完全没有任何supervision**，Weaver自己学会了把memory组织成人类大脑那样的hierarchy。

人类的memory分类：
- Episodic memory（hippocampus）：具体事件
- Procedural memory（basal ganglia）：操作技能，如骑自行车
- Working memory（prefrontal cortex）：当前任务的上下文

MemGen的4个cluster对应到了这个taxonomy上。这是emergent behavior的strong evidence。

## Latent Memory长什么样

作者强行decode了一些latent token，得到的是人类看不懂的"机器方言"：

```
[keyword-kindërgetAs-slide]def even_sorted(lst):
[.keyword_pick]"""
[LTRetical] Returns a new list containing only the even integers...
```

这些token不是英语、不是Python、是Weaver和Reasoner之间的private language。Weaver学会了针对Reasoner的内部representation生成最优的"指令"。

## 我的看法

### 好的地方
1. **Conceptual elegance**：frozen reasoner + 2个LoRA，simple到elegant。整个system就是"主从架构"，主不变，从可换。
2. **Engineering elegance**：与optimization algorithm无关，SFT、GRPO都能用。换backbone也容易。
3. **Emergent hierarchy**：这是deep finding。无supervision下emerge出planning/procedural/working memory，意味着这种hierarchy可能是latent memory的某种attractor state。
4. **Efficiency win**：不仅不慢，反而快，因为它让agent少生成token。

### 让我担心的地方
1. **LoRA capacity**：Weaver是r=16的LoRA，能encode多少experience？paper没test大规模experience积累下weaver何时saturate。如果experience呈scale增长，weaver会不会需要变成full SFT甚至更大？
2. **Trigger的robustness**：trigger学会的meta-cognition能否scale到更复杂的multi-hop reasoning？paper的实验相对short-horizon。
3. **K的选择**：sensitivity analysis只到K=32。如果K=128或者K=1024会怎样？会不会有phase transition？
4. **Causality问题**：post-hoc intervention study是correlational的。要真正prove causality，应该用activation patching或causal tracing，像Anthropic的[Attribution Graphs](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)那样。

### 更深的speculation

如果继续这个方向，会出现什么？

1. **Hierarchical MemGen**：trigger可以分级，低层trigger管"现在要不要回忆"，高层trigger管"现在要不要反思整个strategy"。
2. **Multi-Weaver**：不同Weaver负责不同domain，trigger学会路由。类似MoE but for memory。
3. **Sleep phase**：类似人类的sleep consolidation，offline阶段把白天的experience重新consolidate进Weaver，可能更efficient。
4. **Memory forgetting mechanism**：人类会忘，这是feature不是bug。MemGen目前没有forgetting mechanism，所有memory永远保留，长期可能overfit。
5. **Self-model**：trigger实际上是一种emerging self-model——agent在monitor自己的cognitive state。如果能把这个self-model explicit化，可能通往更强的meta-cognition。

### 跟Karpathy自己工作的一些联想

这个工作让我想到几个Karpathy经常提的theme：

1. **Software 2.0**：Trigger用RL学会"何时invoke memory"，这是把meta-cognitive decision交给learned policy，典型的Software 2.0 pattern。

2. **Build from scratch的intuition**：MemGen的设计非常hackable，一个frozen LLM + 两个LoRA + 一个Bernoulli采样，就实现了generative latent memory。这种简洁性是good system design的标志。

3. **Micrograd的philosophy**：Karpathy的[micrograd](https://github.com/karpathy/micrograd)强调"understand by implementing from scratch"。MemGen的trigger、weaver、training loop都不复杂，应该可以从scratch实现一个简化版本来build intuition。

4. **[Eureka Labs](https://eurekalabs.ai)的direction**：如果这种modular memory系统能deploy到education场景，agent可以一边teach一边learn from student interaction，把每次teaching的经验compile进Weaver，下次更adaptive。

## 参考链

主要paper：
- [MemGen GitHub](https://github.com/KANABOON1/MemGen)

技术背景：
- [DeepSeek-R1 / GRPO](https://arxiv.org/abs/2501.12948)
- [Coconut - Continuous Latent Reasoning](https://arxiv.org/abs/2412.06769)
- [Anthropic Attribution Graphs](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- [Spens & Burgess - Generative Memory Model](https://www.nature.com/articles/s41562-023-01785-z)

Memory相关baseline：
- [ExpeL](https://arxiv.org/abs/2308.10144)
- [Agent Workflow Memory](https://arxiv.org/abs/2409.07429)
- [MemoryBank](https://arxiv.org/abs/2305.10250)
- [MemGPT](https://arxiv.org/abs/2310.08560)
- [LatentSeek](https://arxiv.org/abs/2505.13308)
- [SoftCoT](https://arxiv.org/abs/2502.12134)
- [Memory³](https://dx.doi.org/10.4208/jml.240708)

Reasoning failure分析：
- [Song et al. - LLM Reasoning Failures](https://openreview.net/forum?id=hsgMn4KBFG)

Karpathy相关：
- [Neural Nets: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [micrograd](https://github.com/karpathy/micrograd)

---

**一句话总结**：MemGen把memory从"外挂硬盘"变成"hippocampus-like的实时reconstruction faculty"，用两个LoRA实现了reasoning和memory的fluid interweaving，还spontaneously emerged出人类式的memory hierarchy。这是通往self-evolving agent的一个elegant step。

---

# MemGen: Weaving Generative Latent Memory for Self-Evolving Agents — 深度技术解析

## 1. 高层直觉：这篇paper在解决什么问题？

现有的LLM agent memory机制存在两个极端：

**Parametric memory**（如FireAct、Agent-FLAN、GRPO等）直接把experience编进model参数里。问题是catastrophic forgetting——学新的就忘旧的，破坏backbone的general capability。

**Retrieval-based memory**（如ExpeL、AWM、MemoryBank）把experience外化到database里。问题是这是rigid pipeline——只在task开始时检索一次，把retrieved context粗暴地append到query里。这与人类认知差异巨大：人类大脑中frontoparietal control network的reasoning和hippocampus的memory retrieval是**interweaving**的，形成continuous flow of thoughts（参考[Spens & Burgess, 2024](https://www.nature.com/articles/s41562-023-01785-z)；[Su et al., 2025](https://www.nature.com/articles/s41467-025-6433-w)）。

**MemGen的核心idea**：把memory做成一个**dynamic, generative, latent**的cognitive faculty，在reasoning过程中**按需**触发，在**token-level**上insert latent memory tokens。核心类比是hippocampus consolidating fragments of recollection into human memory的过程——memory不是verbatim的复述，而是selective reconstruction。

---

## 2. 问题形式化：数学表达

### 2.1 Agent与环境交互

Agent $\pi_\theta$（LLM参数为$\theta$）在环境$\mathcal{E}$中交互，trajectory：

$$\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$$

其中：
- $s_t$：环境状态
- $a_t$：agent在step $t$采取的高层action，本身是一个token序列 $a_t = (\mathbf{z}_{t,1}, \mathbf{z}_{t,2}, \ldots, \mathbf{z}_{t,L_t})$
- $\mathbf{z}_{t,j}$：第$t$个action的第$j$个token

Token生成是autoregressive的（Eq. 1）：

$$\mathbf{z}_{t,j} \sim \pi_\theta(\cdot \mid s_t, \mathbf{z}_{t,<j})$$

这里$\mathbf{z}_{t,<j} = (\mathbf{z}_{t,1}, \dots, \mathbf{z}_{t,j-1})$是之前已经生成的tokens。

### 2.2 Memory引入后的形式

引入memory system $\mathcal{M}$后，目标变成（Eq. 2）：

$$\max_{\theta, \mathcal{M}} \mathbb{E}_{x \sim \mathcal{D}, \tau \sim \pi_{\theta, \mathcal{M}}} [R(\tau)]$$

其中：
- $\mathcal{D}$：task distribution
- $R(\tau)$：trajectory的reward
- 关键：$\mathcal{M}$产生memory representation $m$，conditioning agent的policy

Action采样变成（Eq. 3）：

$$a_t \sim \pi_\theta(\cdot \mid s_t, m_t)$$

其中$m_t$是在step $t$插入的memory，由memory生成函数$f_\mathcal{M}$生成：

$$m_t = f_\mathcal{M}(s_t, \mathcal{H}, m_{<t})$$

变量解释：
- $s_t$：当前状态
- $\mathcal{H} = \{(x_i, \tau_i)\}_{i=1}^N$：历史experience
- $m_{<t}$：之前生成的memories

**这里有一个关键的granularity区分**：
- Task-level memory（如ExpeL, G-Memory）：$f_\mathcal{M}$只在$t=0$调用一次，$m_t = m_0$ for all $t$
- Step-level memory（如AgentKB）：每个step调用一次
- MemGen：**token-level**，agent自己决定何时regenerate $m_t$

---

## 3. MemGen架构：Trigger + Weaver

### 3.1 整体流程

Reasoner $\pi_\theta$是frozen的，autoregressively生成action $a_t = (\mathbf{z}_{t,1}, \dots, \mathbf{z}_{t,L_t})$。在每个token生成步骤$j$：

1. **Memory trigger** $\mathcal{T}_{trigger}$监控reasoner的internal cognitive state（hidden states $\mathbf{H}_{t,<j}$）
2. 决定是否invoke memory：$d_j \in \{\text{INVOKE}, \text{SKIP}\}$
3. 如果SKIP：$\pi_\theta$正常生成下一个token
4. 如果INVOKE：暂停reasoning，调用**memory weaver** $\mathcal{W}_{weaver}$
5. Weaver接收相同的$\mathbf{H}_{t,<j}$作为stimulus，合成latent memory $\mathbf{M}_t \in \mathbb{R}^{K \times d_{model}}$
6. $\mathbf{M}_t$被prepend到$\mathbf{H}_{t,<j}$上，reasoner resume生成

### 3.2 Memory Trigger的细节

**Instantiation**：$\mathcal{T}_{trigger}$是attached to $\pi_\theta$的LoRA adapter。

在decoding step $j$，它接收hidden states $\mathbf{H}_{t,<j} \in \mathbb{R}^{(j-1) \times d_{model}}$，输出invocation probability（Eq. 4）：

$$p_j = \sigma\left(\mathcal{T}_{trigger}(\mathbf{h}_{t,1}, \dots, \mathbf{h}_{t,j-1})\right)$$

变量解释：
- $\sigma$：sigmoid函数
- $\mathbf{h}_{t,k} \in \mathbb{R}^{d_{model}}$：reasoner在第$k$个token位置的hidden state vector
- $d_{model}$：LLM的hidden dimension

然后采样binary decision：$d_j \sim \text{Bernoulli}(p_j) \in \{\text{INVOKE}, \text{SKIP}\}$

**Sentence-granularity激活策略**（Eq. 7）：

为了计算效率，trigger只在delimiter tokens处激活。定义delimiter token set $\mathcal{D}$（commas, periods等），则：

$$d_j = \text{Bernoulli}(p_j), \quad p_j = \begin{cases} 0 & \text{if } z_j \notin \mathcal{D} \\ \mathcal{T}_{trigger}(\mathbf{H}_{t,<j}) & \text{if } z_j \in \mathcal{D} \end{cases}$$

这个设计的motivation来自[Anthropic 2025关于LLM interpretability的研究](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)和[SepLLM (Chen et al., 2024)](https://arxiv.org/abs/2412.12094)——interventions在sentence boundary上更有效。

### 3.3 Memory Trigger的RL训练

这是这个paper最elegant的部分之一。Trigger要balance两个competing desiderata：
- 必要时invoke memory来提升task performance
- 避免spurious invocation浪费computation和disrupt reasoning

**Reward-adaptive penalty objective**（Eq. 8）：

$$\max_\phi \mathbb{E}_{\tau_i \sim \pi_\theta, \tilde{\mathbf{d}} \sim \mathcal{T}_{trigger}^\phi} \left[ R(\tau_i) - \lambda \sum_{i,j} \max(0, \tilde{d}_{i,j} - \bar{p}) \right]$$

变量解释：
- $\phi$：trigger的LoRA参数
- $\tilde{d}_{i,j}$：在第$i$个trajectory的第$j$个激活点的binary decision（0或1）
- $\lambda$：penalty系数
- $\bar{p}$：自适应penalty threshold

$\bar{p}$的构造（Eq. 9）非常巧妙——它是**high-reward trajectories上的平均激活概率**：

$$\bar{p} = \frac{1}{|\mathcal{H}_{high}|} \sum_{i \in \mathcal{H}_{high}} \frac{1}{|\tau_i|} \sum_j \tilde{d}_{i,j}$$

其中$\mathcal{H}_{high} = \{i : R(\tau_i) \geq \text{median}_k(R(\tau_k))\}$，即reward超过batch median的trajectory集合。

**Intuition**：高reward的trajectory是"好学生"，trigger要学会模仿它们的invocation pattern。如果当前invocation rate超过这个baseline，就penalize。这是一个自适应的sparsity regularization——不会一刀切，而是根据任务难度调整invocation频率。

这个idea与GRPO的group-relative baseline（[DeepSeek-R1, 2025](https://arxiv.org/abs/2501.12948)）有精神上的相似——都是用relative reference来构造adaptive objective。

### 3.4 Memory Weaver的细节

**Instantiation**：$\mathcal{W}_{weaver}$也是attached to $\pi_\theta$的LoRA adapter。

给定hook $\mathbf{H}_{t,<j} \in \mathbb{R}^{(j-1) \times d_{model}}$，weaver输出latent memory matrix（Eq. 5）：

$$\mathbf{M}_t := [\mathbf{m}_{t,1}, \mathbf{m}_{t,2}, \cdots, \mathbf{m}_{t,K}] = \mathcal{W}_{weaver}(\mathbf{H}_{t,<j})$$

变量解释：
- $\mathbf{M}_t \in \mathbb{R}^{K \times d_{model}}$：latent memory matrix
- $K$：固定的latent memory sequence长度（实验中设为$\{2, 4, 8\}$，sensitivity analysis显示到32还在提升）
- $\mathbf{m}_{t,k} \in \mathbb{R}^{d_{model}}$：第$k$个latent memory token

**关键：这不是verbatim的restatement，而是selective reconstruction**。Weaver可能只用自己的parametric knowledge，也可能incorporate外部retrieved information（见Section 4.3最后一段和Appendix E）。

生成后，$\mathbf{M}_t$的hidden states被prepend到$\mathbf{H}_{t,<j}$，reasoner在这个enriched context上resume生成（Eq. 6）：

$$\mathbf{z}_{t,j} \sim \pi_\theta(\cdot \mid s_t, \mathbf{z}_{t,<j}, \mathbf{M}_t)$$

这个设计在spirit上非常像[Coconut (Hao et al., 2024)](https://arxiv.org/abs/2412.06769)的continuous thought，但有关键区别——Coconut是在reasoning过程中插入latent thought来enable longer deliberation，而MemGen是把memory作为first-class citizen，显式interweave在reasoning中。

### 3.5 Weaver的训练

Weaver的训练目标是（Eq. 10）：

$$\max_{\theta_{lora}} \mathbb{E}_{(x_i, \tau_i) \sim \mathcal{H}} \mathbb{E}_{\tau \sim \Pi_\theta^{\mathcal{W}_{\theta'}, \tau}(\cdot \mid x_i)} [R(x_i, \tau)]$$

变量解释：
- $\theta_{lora}$（或$\theta'$）：weaver的LoRA参数
- $\Pi_\theta^{\mathcal{W}_{\theta'}, \tau}(\cdot \mid x_i)$：由frozen reasoner $\pi_\theta$ + weaver $\mathcal{W}_{\theta'}$ + trigger $\tau$ 组成的rollout policy
- $R(x_i, \tau)$：task $x_i$上trajectory $\tau$的reward

**关键：gradients只propagate to $\theta'$，不更新$\theta$**。这保证了reasoner的general capability完全保留，从根本上避免catastrophic forgetting。

这个modularity使得MemGen与具体optimization algorithm无关。Paper里给了两个variant：

#### 3.5.1 MemGen_SFT（Eq. 11-13）

给定expert demonstration trajectories $\mathcal{H} = \{(x_i, \tau_i^*)\}_{i=1}^N$，最小化negative log-likelihood：

$$\mathcal{L}_{SFT}(\theta') = -\mathbb{E}_{(x_i, \tau_i^*) \sim \mathcal{H}} \left[ \sum_{\ell=0}^{T_i-1} \sum_{j=1}^{L_t} \log \pi_\theta(\mathbf{z}_{i,t,j}^* \mid s_{i,t}, \mathbf{z}_{i,t,<j}^*, \mathbf{M}_{i,t,j}) \right]$$

其中 $\mathbf{M}_{i,t,j} = \mathcal{W}_{\theta'}(\mathbf{H}_{i,t,<j})$（Eq. 12）

梯度更新（Eq. 13）：$\theta' \leftarrow \theta' - \eta \nabla_{\theta'} \mathcal{L}_{SFT}(\theta')$

这里有个细节：$\mathbf{M}_{i,t,j}$只在trigger决定invoke时才生成，否则conditioning term被omit。

#### 3.5.2 MemGen_GRPO（Eq. 14-16）

GRPO的核心是group-relative baseline。对每个task $x_i$，sample $K$条trajectory $\mathcal{G}_i = \{\tau_{i,1}, \dots, \tau_{i,K}\}$，baseline是group average（Eq. 14）：

$$\bar{R}(\mathcal{G}_i) = \frac{1}{K} \sum_{k=1}^K R(\tau_{i,k})$$

Advantage（Eq. 15）：

$$A(\tau_{i,k}) = R(\tau_{i,k}) - \bar{R}(\mathcal{G}_i)$$

GRPO objective（Eq. 16）：

$$\mathcal{I}_{GRPO}(\theta') = \mathbb{E}_{x_i \sim \mathcal{H}, \mathcal{G}_i \sim \Pi_\theta^{\mathcal{W}_{\theta'}, \tau}} \left[ \frac{1}{K} \sum_{k=1}^K A(\tau_{i,k}) \log \Pi_\theta^{\mathcal{W}_{\theta'}, \tau}(\tau_{i,k} \mid x_i) - \beta \text{KL}\left(\Pi_\theta^{\mathcal{W}_{\theta'}, \tau}(\cdot \mid x_i) \parallel \Pi_{ref}(\cdot \mid x_i)\right) \right]$$

变量解释：
- $\Pi_{ref}$：fixed reference policy（KL正则化用）
- $\beta$：KL系数（paper里设为0.0）
- $K$：每个task的rollout数

---

## 4. Integration with Retrieval-based Memory（Appendix E）

这个extension很有意思，展示了MemGen的extensibility。当trigger决定invoke时：

1. 把当前生成的tokens decode成自然语言query（Eq. 17）：
   $$q_{t,j} = \text{Decode}(\mathbf{z}_{t,<j})$$

2. 用外部retriever $\mathcal{R}$检索（Eq. 18）：
   $$\mathcal{C}_t = \mathcal{R}(q_{t,j}; \mathcal{M}_{ext})$$
   其中$\mathcal{C}_t = \{c_1, \dots, c_P\}$是$P$条retrieved snippets

3. 把snippet encode成embeddings $\mathbf{E}_t \in \mathbb{R}^{L_c \times d_{model}}$

4. Weaver接收concatenated context（Eq. 19）：
   $$\mathbf{M}_t = \mathcal{W}_{weaver}\left(\left[\mathbf{H}_{t,<j}; \mathbf{E}_t\right]\right)$$

Table 8的结果非常striking：即使disable weaver的parametric memory（只用retrieved snippets），MemGen还是把ExpeL的ALFWorld从36.18%提到45.60%。当weaver的parametric memory启用时，ALFWorld达到75.90%，TriviaQA达到76.40%。这说明weaver不是简单append retrieval，而是**actively reconstruct**成更potent的latent memory。

---

## 5. 实验结果深度分析

### 5.1 Main Results（Table 1, Table 3）

最striking的几个数据点：

**ALFWorld + SmolLM3-3B**：
- Vanilla: 18.96%
- SFT: 32.36%
- GRPO: 55.35%
- ExpeL: 36.18%
- AWM: 40.50%
- **MemGen_SFT: 50.60%** (+31.64% over vanilla)
- **MemGen_GRPO: 63.60%** (+44.64% over vanilla)

**Qwen3-8B上的KodCode**：
- Vanilla: 49.10%
- GRPO: 73.35%
- **MemGen_GRPO: 76.16%** (+27.06% over vanilla)

值得注意的细节：在knowledge-intensive任务上（如GPQA），retrieval-based方法表现差（ExpeL在GPQA+Qwen2.5-1.5B只有8.12%，甚至低于vanilla的11.62%）。这说明retrieval-based memory严重依赖backbone capacity——小model context不够，反而被retrieved context干扰。

### 5.2 Cross-Domain Generalization（Figure 3, 9, 10）

这是最有意思的finding之一。在GSM8K上训练MemGen_SFT，在不同domain上的invocation frequency（Figure 4）：

- GSM8K：最高invocation frequency，+19.64% improvement
- GPQA：中等invocation frequency，+6.06% improvement
- KodCode：最低invocation frequency，+3.1% improvement

**Intuition**：trigger自动学会了"什么时候需要memory"。在熟悉的domain上频繁invoke；在unfamiliar domain上少invoke以avoid disruption。这是一个**meta-cognitive**的能力——agent知道自己的知识边界。

### 5.3 Continual Learning（Table 4）

Sequential training: AQuA → GPQA → GSM8K → KodCode，每个stage后在所有4个benchmark上评估。

最关键的对比：在KodCode stage之后，AQuA上的performance：
- SFT: 28.61%（catastrophic forgetting严重）
- ExpeL: 27.14%
- **MemGen_SFT: 40.34%**（保留了大量AQuA能力）

GPQA上的performance：
- SFT: 2.53%（几乎完全遗忘）
- ExpeL: 6.23%
- **MemGen_SFT: 20.09%**

**为什么MemGen能缓解catastrophic forgetting？** 因为reasoner是frozen的，所有experience都被compress到weaver的LoRA参数里。LoRA的低rank结构天然提供了implicit regularization，而且weaver只输出latent memory作为contextual conditioning，不直接修改reasoner的knowledge representation。

### 5.4 Ablation Study（Table 5）

Memory invocation策略对比（Qwen2.5-1.5B）：

| Strategy | GPQA | KodCode | TriviaQA |
|----------|------|---------|----------|
| Random (p=0.2) | 15.66 | 54.55 | 63.55 |
| Random (p=0.5) | 16.66 | 52.95 | 57.28 |
| Random (p=0.8) | 12.63 | 53.60 | 62.22 |
| All delimiters | 17.34 | 56.20 | 64.15 |
| **MemGen Trigger** | **18.28** | **58.16** | **65.02** |

两个关键观察：
1. Sentence-level intervention > random invocation——confirm了在semantic boundary上intervene更有效
2. Trained trigger > all delimiters——selective activation提供了最佳balance

### 5.5 Weaver Parameterization（Table 6）

- LoRA (r=16, α=32): GPQA 18.28, KodCode 58.16, TriviaQA 65.02
- Full SFT: GPQA 21.21, KodCode 60.00, TriviaQA 67.10

LoRA已经相当competitive，full SFT有提升但不dramatic。这说明even lightweight adaptations就足够endow weaver with sufficient capacity。

### 5.6 Efficiency Analysis（Table 7）

最令人惊讶的发现——MemGen不仅不慢，反而更快！以KodCode + Qwen2.5-1.5B为例：
- Vanilla: 11.96s, 24.55% accuracy
- SFT: 2.01s, 55.83% accuracy
- MemGen_SFT: 2.94s, 58.16% accuracy

MemGen比vanilla快75.4%，同时accuracy提升33.61%。原因是MemGen让model更快reach correct answer，生成更少的tokens。这与[MEM1 (Zhou et al., 2025)](https://arxiv.org/abs/2506.15841)和[MemAgent (Yu et al., 2025)](https://arxiv.org/abs/2507.02259)的发现一致——好的memory机制实际上可以减少reasoning长度。

---

## 6. Memory Hierarchy的Emergent Behavior（Section 5.3, Appendix G）

这是整个paper最fascinating的部分。作者通过post-hoc intervention study，发现MemGen spontaneously evolved human-like memory hierarchy。

### 6.1 可视化方法

对latent memory sequence $\mathbf{M}_i = (\mathbf{m}_{i,1}, \ldots, \mathbf{m}_{i,K})$，计算mean embedding（Eq. 20）：

$$\bar{\mathbf{m}}_i = \frac{1}{K} \sum_{l=1}^K \mathbf{m}_{i,l}$$

然后用t-SNE降维到2D，再用K-means聚类（$N=4$）。

Figure 5显示：
- 不同domain的latent memory形成separate distributions
- Related domains cluster closely：KodCode & BigCodeBench, GSM8K & MATH

### 6.2 Failure Mode Taxonomy

参考[Song et al., 2025](https://openreview.net/forum?id=hsgMn4KBFG)，定义8种agent failure：
1. **Planning Failure**：高层task decomposition错误
2. **Compositional Reasoning**：无法integrate多个信息piece
3. **Tool Parsing Error**：tool call格式错误
4. **Tool Response Error**：误用tool返回值
5. **Answer Formatting Failure**：最终output格式错误（如忘记`\boxed{}`）
6. **Demand Misunderstanding**：误解task意图
7. **Think-Act Inconsistency**：reasoning和action不一致
8. **False Belief**：基于错误假设reasoning

### 6.3 Intervention Study

对每个cluster $C_j$，计算centroid $\mu_j \in \mathbb{R}^{d_{model}}$。Inference时，新memory sequence $\mathbf{M}_t$的mean embedding（Eq. 21）：

$$\bar{\mathbf{m}}_{new} = \frac{1}{K} \sum_{l=1}^K \mathbf{m}_{t,l}$$

把$\bar{\mathbf{m}}_{new}$和reference set $\mathcal{E}_{comp} = \mathbf{E}_{vocab} \cup \{\mu_1, \dots, \mu_N\}$比较，找top-$k$ nearest neighbors（$k=10$）。如果target cluster的centroid $\mu_j$在top-$k$中，就filter掉整个$\mathbf{M}_t$（Eq. 22）：

$$\mu_j \in S_k(\bar{\mathbf{m}}_{new})$$

### 6.4 发现的Memory Hierarchy

Figure 6 (Right)的结果揭示了functional specialization：

- **Cluster 2 = Planning Memory**：移除后planning和compositional reasoning failures大幅增加。支持高层task planning和strategic reasoning。

- **Cluster 3 = Procedural Memory**：移除后tool response errors、parsing failures、answer formatting mistakes显著增加。捕获task-specific operational knowledge（tool usage、formatting）。

- **Clusters 1, 4 = Working Memory**：移除Cluster 1导致task misunderstanding和think-act inconsistency增加。管理context retention和reasoning consistency。

**这些cluster并非完全独立**——移除Cluster 1也negatively影响planning，说明memory faculties是interacting的。

这个发现让人联想到neuroscience中的memory taxonomy：
- Episodic memory (hippocampus)
- Procedural memory (basal ganglia)
- Working memory (prefrontal cortex)

MemGen在没有explicit supervision的情况下emerged出这种hierarchy，这suggests了一种朝向更naturalistic机器认知的emergent trajectory。

---

## 7. Latent Memory是Machine-Native的（Appendix F）

Appendix F的case study非常有意思。当作者强行decode latent tokens时，得到的是人类不可读的序列，但exhibit regularities：

**TriviaQA的Cluster 0**：经常以`[...]SOC`结尾
**GSM8K的Cluster 3**：经常采用`[...]\_pick`格式

例如KodCode的输出：
```
[keyword-kindërgetAs-slide]def even_sorted(lst):
[.keyword_pick]"""
[LTRetical] Returns a new list containing only the even integers...
[.keyword_pick]"""
[LTRetical] even_numbers = [num for num in lst if num % 2 == 0]
```

这些latent tokens是"machine-native"——它们是weaver针对frozen reasoner的内部representation optimized的，类似一种emergent的"machine pidgin language"。

---

## 8. 与相关工作的深度对比

### 8.1 与Latent Computation工作的对比

- **Coconut (Hao et al., 2024)**：让LLM在continuous latent space中reason，是一种architectural change。MemGen不改变架构，只在生成过程中插入latent memory tokens。

- **CODI (Shen et al., 2025)**：通过self-distillation把CoT compress到continuous space。

- **LatentSeek (Li et al., 2025a)**、**SoftCoT (Xu et al., 2025c)**：用latent embedding steer LLM generation，但仍然是retrieval-based（fetch memories by embedding similarity）。

- **Co-processor (Liu et al., 2024)**：在latent space中做deliberation/cache augmentation。

MemGen与这些工作的关键区别是**generative**而非retrieval-based，且与reasoning interweaving而非单次插入。

### 8.2 与RLVR的对比

[DeepSeek-R1](https://arxiv.org/abs/2501.12948)的GRPO是全参数RL，会修改reasoner的knowledge。MemGen的trigger用RL训练但只更新LoRA，weaver也只更新LoRA。这是一种"modular RL"——把RL的影响isolate到auxiliary module，保护core LLM。

### 8.3 与Speculative Decoding的对比

[Medusa (Cai et al., 2024)](https://arxiv.org/abs/2401.10774)、[Eagle (Li et al., 2025b)](https://arxiv.org/abs/2401.15077)等speculative decoding方法在形式上类似——drafter model接收current context，生成drafted tokens。但目的是加速inference，MemGen的目的是把latent state作为memory载体。

### 8.4 与Memory³、MemoryLLM的对比

[Memory³ (Yang et al., 2024)](https://dx.doi.org/10.4208/jml.240708)用explicit memory在attention中reuse。[MemoryLLM (Wang et al., 2024b)](https://arxiv.org/abs/2402.04624)和[M+ (Wang et al., 2025a)](https://arxiv.org/abs/2502.00592)用latent token embedding store experience，但仍依赖invasive LLM parameter updates。MemGen完全frozen reasoner，从根本上不同。

---

## 9. 个人Critical Reflections

### 9.1 Strengths

1. **Conceptual elegance**：把memory做成cognitive faculty而非external database，与neuroscience的memory theory对齐。
2. **Modularity**：reasoner frozen，所有更新都在LoRA上，避免catastrophic forgetting。
3. **Emergent hierarchy**：无supervision下emerge出planning/procedural/working memory的specialization，这是一个deep finding。
4. **Efficiency**：不仅不慢，反而更快，因为减少了生成长度。

### 9.2 潜在的Concerns

1. **LoRA capacity上限**：weaver用LoRA（r=16）能encode多少experience？随着experience增长，weaver会saturate吗？paper没讨论这个scalability问题。

2. **Trigger的meta-cognitive能力的robustness**：trigger学会判断"什么时候需要memory"，但这种能力能否在更复杂的multi-hop reasoning上generalize？

3. **K的选择**：sensitivity analysis显示K=32还在提升，但没explore更大的K。是否有scaling law？

4. **Memory hierarchy的可解释性**：post-hoc intervention study是correlational的，不能证明causal relationship。可以做一些activation patching实验来更rigorously establish causality，类似[Anthropic的attribution graphs工作](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)。

5. **与人类memory的类比深度**：paper经常类比hippocampus、prefrontal cortex，但这些analogy更多是metaphorical的。要真正validate这种analogy，需要更精细的mapping，比如working memory的capacity limit（人类是7±2）是否在MemGen的K上有类似体现？

### 9.3 与Karpathy自己的intuition的连接

如果 recall Karpathy在[Neural Nets: Zero to Hero](https://karpathy.ai/zero-to-hero.html)和[Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)中强调的"build intuition by building from scratch"——MemGen的设计实际上非常hackable。一个frozen LLM + 两个LoRA，加一个trigger和一个weaver，就可以实现generative latent memory。这种简洁性是engineering elegance的体现。

另外，Karpathy经常强调的"software 2.0"——用gradient descent代替explicit programming。MemGen的trigger通过RL学会"何时invoke memory"，这正是一个典型的software 2.0 pattern——把meta-cognitive decision交给learned policy。

---

## 10. Reference Links

主要paper：
- [MemGen GitHub](https://github.com/KANABOON1/MemGen)
- [DeepSeek-R1 (GRPO)](https://arxiv.org/abs/2501.12948)
- [Coconut - Continuous Thought](https://arxiv.org/abs/2412.06769)
- [ExpeL](https://arxiv.org/abs/2308.10144)
- [AWM - Agent Workflow Memory](https://arxiv.org/abs/2409.07429)
- [MemoryBank](https://arxiv.org/abs/2305.10250)
- [MemGPT](https://arxiv.org/abs/2310.08560)
- [LatentSeek](https://arxiv.org/abs/2505.13308)
- [SoftCoT](https://arxiv.org/abs/2502.12134)
- [Co-processor](https://arxiv.org/abs/2412.17747)
- [Anthropic Biology of LLM](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- [Spens & Burgess - Generative model of memory](https://www.nature.com/articles/s41562-023-01785-z)
- [Su et al. - Continuous flow of thoughts](https://www.nature.com/articles/s41467-025-6433-w)
- [Song et al. - LLM Reasoning Failures](https://openreview.net/forum?id=hsgMn4KBFG)
- [Mem1](https://arxiv.org/abs/2506.15841)
- [Memory³](https://dx.doi.org/10.4208/jml.240708)
- [MemoryLLM](https://arxiv.org/abs/2402.04624)
- [M+](https://arxiv.org/abs/2502.00592)
- [A-Mem](https://arxiv.org/abs/2502.12110)
- [Medusa](https://arxiv.org/abs/2401.10774)
- [Eagle](https://arxiv.org/abs/2401.15077)
- [G-Memory](https://arxiv.org/abs/2506.07398)
- [Survey of Self-Evolving Agents](https://arxiv.org/abs/2507.21046)
- [Survey of Latent Reasoning](https://arxiv.org/abs/2507.06203)

---

**总结**：MemGen代表了agent memory研究的一个新paradigm——把memory从rigid的外部数据库，变成interleaved的、generative的、latent的认知faculty。其modular design（frozen reasoner + LoRA-based trigger & weaver）既避免了catastrophic forgetting，又提供了strong cross-domain generalization和continual learning能力。最令人excited的是emergent memory hierarchy的发现——这suggests了一个path toward更naturalistic的machine cognition。未来的direction可能包括：scaling weaver capacity、在更complex multi-agent settings中测试、用mechanistic interpretability工具更精确地understand emergent memory clusters的functional roles。
