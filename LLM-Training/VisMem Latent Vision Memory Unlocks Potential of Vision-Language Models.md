---
source_pdf: VisMem Latent Vision Memory Unlocks Potential of Vision-Language Models.pdf
paper_sha256: 51a3e3e2d4dfb5c5664decec46acd14847339e21c562be24b951b1620f42495c
processed_at: '2026-08-13T01:52:08-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VisMem 用人话讲

## 这个paper到底想干啥

你得先理解VLM有个很烦的毛病。

你给模型一张图，问它"左边那个红色的杯子里有几只笔"。模型一开始看图，visual tokens排在序列最前面。然后它开始生成文字，一个token一个token往外蹦。生成到第200个token的时候，注意力机制要做softmax over所有之前的tokens——包括最初的几百个visual tokens和后来生成的200个text tokens。

问题就出在这。Softmax是个零和游戏，attention权重加起来等于1。你后面生成的text tokens越多，分给前面visual tokens的attention就越少。visual information被textual context给**稀释**了。

这就好比你读一本书，读到第10页的时候让你回忆第1页的一个细节，你能记住大概情节，但具体的画面已经模糊了。VLM也是一样，生成越长，对原始画面的记忆越淡。

这个paper管这叫"visual processing bottleneck"。

## 现有方法都有啥毛病

之前大家想了四种招：

**第一招：直接改参数**。拿RL或SFT硬train，让模型变强。VLM-R1、Vision-R1就是这路子。问题是你会**catastrophic forgetting**——学了新的把旧的忘了。Table 8的数据很扎眼：VLM-R1在stage 0的时候MMVet能到77.8，经过4个stage的continual learning后掉到66.4，几乎把initial gain全吐回去了。你改的是core parameters，新知识必然覆盖旧知识。

**第二招：在pixel层面操作**。让模型在推理过程中画bounding box、crop图片、zoom in/out。DeepEyes、Sketchpad、OpenThinkImg干的就是这个。听起来很美，但计算开销爆炸。Table 12里DeepEyes在MMVet上要3.21秒，vanilla只要0.76秒，慢了4倍多。每次操作你都要重新跑一遍vision encoder，成本太高。

**第三招：在token层面操作**。把已经encode好的visual tokens重新select或re-weight一下。MINT-CoT、VPT、ICoT这类。但有个根本限制——你只能re-surface模型已经看到的信息，没法生成新的visual representation。就像你只能翻回去看同一张照片，没法让模型"想象"出一个新的画面。

**第四招：在latent space操作**。这是最近兴起的。Coconut在language-only的模型上用continuous latent tokens替代textual CoT，证明latent space reasoning是可行的。Mirage尝试在VLM上构建latent visual space，但需要大量manually labeled images来训练。

VisMem属于第四招，但做得更彻底——不需要额外标注数据，而且区分了short-term和long-term两种memory。

## VisMem的思路：抄人脑

Paper找了个很优雅的理论依据——Dennis Norris的memory theory。

人脑有两种视觉memory：

**Short-term memory**是visually dominant的。你看到一张货架的图，短时间内能记住每个商品的摆放位置、颜色、形状。这玩意儿靠visual cortex和posterior superior temporal lobe，处理的是raw visual information。

**Long-term memory**是semantically dominant的。你知道"促销标签"长什么样，知道"classic Lay's"的包装是什么风格，这些是从过去经验中抽象出来的语义知识。靠medial temporal lobe，处理的是abstract semantic representations。

举个例子：你问模型"找到货架上的classic Lay's并判断是否在促销"。

- 要识别"那个蓝黄色包装的薯片在哪" → 需要short-term memory，记住当前画面的fine-grained details
- 要判断"这是否是促销商品" → 需要"促销标签长什么样"这个general knowledge → 靠long-term memory

这两种memory功能不同、神经基础不同、信息表征也不同。VisMem就把这两种memory用两个不同的module来实现。

## 架构怎么搭的

整个系统有三个核心组件。

### 组件一：Special Tokens

VisMem给VLM的vocabulary加了4个特殊token：

- `<m_I^s>`：触发short-term memory的invocation
- `<m_E^s>`：short-term memory结束，恢复正常decoding
- `<m_I^l>`：触发long-term memory的invocation
- `<m_E^l>`：long-term memory结束

工作流程是这样的：模型在autoregressive生成过程中，某个时刻"决定"需要调memory了，就生成一个`<m_I^s>`token。这时memory former立刻启动，生成一串latent memory tokens，插到序列里。然后自动append一个`<m_E^s>`，模型继续正常生成文字。

**公式(4)** $x_{t,i} \sim \mathcal{P}(\cdot | s_t, x_{t,<i}, \{m_I, m_1, ..., m_N, m_E\})$ 

这个公式说的就是：生成第i个token的时候，condition不只是之前的tokens $x_{t,<i}$，还包括刚才插入的memory tokens $\{m_1, ..., m_N\}$。这些memory tokens参与了attention计算，给后续generation提供visual或semantic information。

初始化的时候，invocation tokens用delimiter token的embedding加上小扰动来初始化。为什么用delimiter？因为delimiter语义上是"中性的"，不会给模型太强的先验。end tokens用类似初始化但learning rate更低，因为它们只是结构标记，不需要学太多。

### 组件二：Query Builder B

这是个lightweight transformer encoder，负责把模型当前的cognitive state转换成一个query，用来"问"memory要什么信息。

**公式(5)** $\mathbf{Q} = \mathcal{B}([\mathbf{H}, \mathbf{Q}_{init}])[-K:]$

拆解一下：
- $\mathbf{H} = \{v_1, ..., v_y, h_1, ..., h_z\}$：当前所有hidden states。$v$是vision encoder输出的visual hidden states，长度$y$；$h$是language model当前生成的textual hidden states，长度$z$。这两个拼在一起就是模型此刻的"完整认知状态"
- $\mathbf{Q}_{init} = \{q_1, ..., q_K\}$：$K$个learnable vectors，是query的"种子"。实验里$K=8$
- $\mathcal{B}([\mathbf{H}, \mathbf{Q}_{init}])$：把hidden states和query种子拼一起，过transformer encoder
- $[-K:]$：取最后$K$个输出作为最终的memory query $\mathbf{Q}$

这里有个关键设计——**asymmetric attention mask**。

**公式(11)** $M_{ij} = \begin{cases} -C, & i < K \text{ and } j \geq K \\ 0, & \text{otherwise} \end{cases}$

$i < K$是query token的位置，$j \geq K$是hidden states的位置。$-C$是个很大的负数，让attention weight接近0。

这个mask的意思是：query可以attend to hidden states（查询方主动看信息源），但hidden states不能attend to query（信息源不被查询方污染）。如果你让双向attention，hidden states会被query的信息"污染"，影响后续的generation质量。

这个设计其实就是cross-attention的变体，但巧妙地放在了self-attention的framework里实现。

### 组件三：Dual Memory Formers

两个LoRA adapter，分别负责short和long memory。

**公式(6)** $\mathbf{M}_{s/l} = \mathcal{F}_{s/l}([\mathbf{X}, \mathbf{Q}, \mathbf{M}_{init}])[-N_{s/l}:]$

- $\mathbf{X}$：当前的target token sequence
- $\mathbf{Q}$：query builder刚生成的query
- $\mathbf{M}_{init}$：learnable memory tokens，是memory的"模板"
- $N_s = 4$：short-term memory输出4个tokens
- $N_l = 8$：long-term memory输出8个tokens

**两个former的attach位置不同，这是最核心的设计**：

Short-term former $\mathcal{F}_s$挂在vision encoder上。它生成的memory tokens和visual token stream拼接，然后过original projector对齐到language model的表示空间。这样生成的memory保留了visual features的fine-grained details——像素级别的颜色、位置、形状信息。

Long-term former $\mathcal{F}_l$挂在language model的输出端。它直接在semantic space生成representations，编码的是abstract knowledge——"促销标签通常是什么颜色"、"数学公式中变量之间的关系"这类跨场景的语义知识。

**为什么这么设计？** 回到Dennis Norris theory：short-term memory靠visual cortex，所以attach到vision encoder；long-term memory靠medial temporal lobe（语义处理），所以attach到language model。这是architecture和cognitive theory的alignment。

用LoRA（rank=16, α=32, target=q-proj和v-proj）是为了non-intrusive——不动base model的核心参数，只train adapter。这直接导致了后面catastrophic forgetting的mitigation。

## 训练怎么搞

这里有个**chicken-and-egg problem**：你得同时训"什么时候调memory"（invocation policy）和"调出什么memory"（formation quality）。两个一起训会互相干扰——bad memory让invocation学到错pattern，bad invocation让formation收到noisy gradient。

Paper用两阶段RL解决这个：

### Stage I：先训Memory Formation

Freeze policy model，只train query builder B和两个memory formers $\mathcal{F}_{s/l}$。

**公式(7)** $\max_{\mathcal{F}_{s/l}, B} \mathbb{E}[\Delta S(\tau)]$

$\Delta S(\tau) = S(\tau) - S(\tau_{base})$——有memory的trajectory比没memory的baseline好多少。这个relative reward很聪明，直接优化memory带来的增量价值。

训练时有个curriculum策略：一开始只在delimiter处random invoke memory（让former先学简单的），然后逐渐扩展到可以在sequence任意位置invoke。

用的GRPO，**公式(14)**的核心是group-relative advantage：

$$\hat{A} = \frac{S(\tau) - \overline{S}}{\hat{S} + \epsilon}$$

$\overline{S}$是group内所有trajectory的平均分，$\hat{S}$是标准差。每个trajectory的advantage是相对于group平均的normalized score。这比absolute reward更稳定，因为different tasks的reward scale不同。

Group size $G=16$，clip ratio $\epsilon=0.2$，KL penalty $\beta=0.015$。这些是标准GRPO超参。

### Stage II：再训Memory Invocation

Freeze所有memory formation组件，只train policy model的部分参数$\theta$。

**公式(8)** $\max_\theta \mathbb{E}[\Delta S(\tau) - \alpha(p_{type} + p_{neg})]$

两个penalty很关键：

**Type penalty** $p_{type} = \max(0, S(\tau_{rev}) - S(\tau))$：如果模型选错了memory type（该用short的用了long），$\tau_{rev}$是用alternative type的结果。如果alternative反而更好，就penalize当前选择。

**Negative penalty** $p_{neg} = \max(0, \overline{S} - S(\tau))$：如果invocation带来的效果低于group平均，penalize。防止模型过度invoke。

$\alpha = 0.3$控制penalty强度。

这两个penalty让模型学会两件事：选对memory type，不要瞎invoke。从ablation的random invocation实验看，100% invoke反而比75% invoke差（MMVet 73.4 vs 73.6），说明过多invocation是有害的，模型需要学会"克制"。

## 实验结果说了什么

### 主实验：全面提升

Table 1里12个benchmarks，VisMem平均65.5%，vanilla 54.5%，**绝对提升11.0%**。

分能力看：
- Understanding：+8.9%
- Reasoning：+14.4%
- Generation：+10.6%

Reasoning提升最大，符合预期——reasoning需要多步inference，visual forgetting最严重，memory带来的gain也最大。MV-Math从18.9%到41.4%，提升22.5%，这是all benchmarks里最大的单项提升。

### Cross-Model：9个base model都work

Table 2测了Qwen2.5-VL-3B/7B/32B、LLaVA-OV-1.5-4B/8B、InternVL-3.5-4B/8B/14B/38B。

有个interesting pattern：smaller models在perception tasks上gain更大（3B的BLINK +18.3%），larger models在reasoning tasks上gain更大（38B的MV-Math +21.2%）。

我的解读：smaller models本身perception能力弱，memory给的是"雪中送炭"；larger models本身perception够用，但在复杂reasoning时仍需要memory提供semantic knowledge来"锦上添花"。

### Catastrophic Forgetting：最大优势

Table 8的four-stage continual learning：

```
          Stage 0  Stage 1  Stage 2  Stage 3  Retention
SFT        71.4    70.6    62.3    60.1    -15.8%
VLM-R1     77.8    74.0    66.1    66.4    -14.6%
Vision-R1  76.9    74.5    63.4    62.9    -18.2%
VisMem     78.6    78.9    71.3    72.1    -8.3%
```

VisMem的retention rate 91.7%，远高于其他方法。而且Stage 1比Stage 0还高（78.9 vs 78.6），说明有positive forward transfer——学新任务的memory反而帮助了原始任务。

根本原因是VisMem不碰base model参数，所有memory都存在LoRA adapter里。新任务训练时只更新adapter，原有knowledge完整保留。

### Memory Invocation是自适应的

Figure 5和9分析了invocation pattern：

- **MuirBench**（multi-image understanding）：short-term invocation >> long-term。需要fine-grained perception来区分多张图的细节
- **MV-Math**（math reasoning）：long-term invocation > short-term。需要abstract mathematical knowledge
- **MMVet**（general understanding）：两者均衡
- **Relative position**：两种memory都在output sequence早期invoke更多，后期减少。早期establish grounding，后期主要是reasoning

这些pattern完全是通过Stage II的RL学出来的，没有hardcoded rules。模型自己discovered了"什么时候需要什么memory"的策略。

### Efficiency：overhead很小

Table 12的inference time：
- Vanilla: 0.76s (MMVet)
- VisMem: 0.84s (+10.5%)
- DeepEyes: 3.21s (+322%)
- OpenThinkImg: 3.68s (+384%)

VisMem的overhead只有8-44%，和direct training methods差不多，远低于image-level methods。因为memory generation只是几个LoRA adapter的前向传播，不需要重新encode图像。

## 这篇paper的真正贡献

让我总结一下visMem的几个关键insight：

**Insight 1：Latent space是visual memory的right abstraction level**

Pixel-level太贵，token-level太弱，latent space刚好。你可以在continuous space生成新的visual representations，成本只是几个LoRA adapter的前向传播。

**Insight 2：Short和Long memory的功能分离很重要**

Table 3的ablation说明，single memory type在特定task上可能更好（short在MuirBench，long在MV-Math），但dual memory在所有task上都最优。不同task需要不同类型的memory support，一个module搞不定所有事。

**Insight 3：Memory invocation必须learned，不能hardcoded**

Random invocation实验说明，固定比例invoke效果不稳定。75%比100%好，50%比25%好，但75%比50%提升有限。最优invocation策略是task-dependent、position-dependent的，必须通过RL让模型自己学。

**Insight 4：Non-intrusive design是continual learning的关键**

用LoRA + special tokens，不动core parameters。这让VisMem天然具备catastrophic forgetting mitigation能力。Direct training方法（VLM-R1、Vision-R1）虽然initial gain大，但在continual learning中forgetting严重。

## 我的一些思考

几个可能继续探索的方向：

**Memory consolidation**：目前long-term memory是task-specific的。如果能让memory across sessions积累和consolidate，像人脑在睡眠中consolidate memory一样，可能实现更general的knowledge transfer。

**Dynamic memory length**：固定$N_s=4, N_l=8$可能不是最优。简单task可能2个token够，复杂task可能需要32个。可以加个"memory budget"的RL signal让模型自己决定memory length。

**Memory interpretability**：latent memory tokens目前是黑箱。如果能visualize出memory tokens对应什么visual或semantic concept，对debugging和trust很有帮助。可以用sparse autoencoder或attention rollout来probe。

**Multi-hop memory**：目前每次invocation是独立的。如果允许memory tokens之间也能attend（memory about memory），可能支持更复杂的reasoning chain。

**Cross-modal memory**：video、audio的latent memory。Video understanding有更严重的temporal forgetting问题，latent memory可能有更大gain。

**Memory retrieval vs generation**：目前memory是generated的。如果有一个external memory bank，让模型retrieve relevant memories而不是generate from scratch，可能更适合knowledge-heavy tasks。

**Connection to test-time compute scaling**：VisMem的memory invocation本质上是一种test-time compute scaling——通过在推理时增加compute来提升performance。这和o1、R1的reasoning scaling是同一大方向，但visMem是在latent space而非token space scale。

**Connection to working memory in neuroscience**：VisMem的short/long memory distinction很接近Baddeley的working memory model（phonological loop vs episodic buffer）。如果再加个"central executive"来控制memory allocation，可能更接近人脑的完整working memory architecture。

References:
- [VisMem GitHub](https://github.com/YU-deep/VisMem.git)
- [Dennis Norris Memory Theory](https://psycnet.apa.org/record/2017-26421-001)
- [Coconut - Latent Reasoning](https://arxiv.org/abs/2412.06769)
- [GRPO - DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [VLM-R1](https://arxiv.org/abs/2504.07615)
- [Mirage - Latent Visual Space](https://arxiv.org/abs/2506.17218)
- [DeepEyes](https://arxiv.org/abs/2505.14362)
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)
- [Baddeley Working Memory Model](https://www.annualreviews.org/doi/10.1146/annurev-psych-120710-100422)

---

# VisMem: Latent Vision Memory Unlocks Potential of Vision-Language Models 深度讲解

## 1. 核心问题与动机

这篇paper要解决的核心问题是VLMs在**autoregressive generation**过程中的"visual processing bottleneck"。具体来说，随着token-by-token生成的进行，模型会优先积累textual context，逐渐丢失对初始visual evidence的grounding，同时缺乏contextualized visual experience。

让我从更底层的角度来理解这个问题：在标准transformer的self-attention机制中，每一个new token的生成都需要attend to所有previous tokens。如果initial visual tokens占据了序列前端，而生成过程中不断累积text tokens，那么后期生成的token在计算attention weights时，由于softmax normalization的存在，对visual tokens的attention权重会被大量text tokens稀释。这导致了**visual information dilution**现象。

参考相关工作：
- [Mitigating visual forgetting via take-along visual conditioning](https://arxiv.org/abs/2410.19732) - 类似的visual forgetting问题
- [Rethinking visual dependency in long-context reasoning](https://arxiv.org/abs/2410.19732) - visual dependency的系统性分析

## 2. 四种Paradigms的系统对比

Paper将现有方法归纳为四种paradigms，这种分类非常有启发性：

### (a) Direct Training Paradigm
代表方法：SFT, Visual-RFT, VLM-R1, Vision-R1, PAPO

**核心机制**：直接通过fine-tuning或reinforcement learning更新model parameters。比如VLM-R1基于GRPO策略，Vision-R1使用类似DeepSeek-R1的reasoning enhancement方法。

**根本缺陷**：catastrophic forgetting。从Table 8的continual learning实验可以看出，SFT在4个stage后性能下降超过10%，VLM-R1和Vision-R1的initial gains（+11.8%和+10.9%）在stage 4时几乎完全消失（retention < 0.5%）。这是因为直接修改core parameters会overwrite之前encoded的general knowledge。

### (b) Image-Level Paradigm
代表方法：Visual CoT, DeepEyes, Sketchpad, PixelReasoner, MVoT, OpenThinkImg, GRIT

**核心机制**：在pixel space操作，通过bounding boxes或外部工具生成new visual inputs。例如Sketchpad让模型在推理过程中sketch intermediate visual representations，DeepEyes通过zoom-in/zoom-out机制提供细粒度visual evidence。

**计算开销**：从Table 12可以看出，这类方法的inference time几乎是vanilla model的2倍。例如DeepEyes在MMVet上3.21秒（vanilla 0.76秒），OpenThinkImg 3.68秒，VPT甚至达到2.98秒。这是因为需要调用external vision tools或重新encode synthesized images。

### (c) Token-Level Paradigm
代表方法：ICoT, MINT-CoT, Scaffold, VPT, Chameleon

**核心机制**：在visual token层面操作，select或re-weight已有的visual representations。比如MINT-CoT在mathematical CoT中interleave visual tokens，VPT引入visual perception tokens。

**根本局限**：这类方法本质上non-generative，只能re-surface模型已经encoded的信息。这限制了它处理需要novel visual synthesis或abstract reasoning的任务。

### (d) Latent Space Paradigm
代表方法：Coconut, MemGen, LatentSeek, SoftCoT, CODI, Mirage, **VisMem**

**核心机制**：在continuous latent space引入implicit memory representations。Coconut是language-only的先驱工作，MemGen探索generative latent memory，Mirage首次尝试构建latent visual space但需要大量manually labeled images。

**VisMem的独特之处**：这是第一个将short-term和long-term latent vision memory系统整合到VLM generation过程的方法，无需额外labeled visual data。

## 3. VisMem架构深度解析

### 3.1 问题描述的形式化

Paper用reinforcement learning的framework来formulate问题。给定policy model P（base VLM），instruction-vision pair (I,V)从task distribution D采样，模型展开trajectory τ。在timestep t，state s_t包括textual contexts和visual observations，action a_t通过token-by-token autoregressive decoding生成。

**公式(1)**：$x_{t,i} \sim \mathcal{P}(\cdot | s_t, x_{<i})$

这里：
- $x_{t,i}$：在timestep t生成的第i个output token
- $s_t$：当前environment state（包括所有之前的textual和visual context）
- $x_{<i}$：之前生成的所有tokens
- $\mathcal{P}(\cdot|\cdot)$：policy model的条件分布

**公式(2)**：$\max_{\mathcal{P}, \mathcal{M}} \mathbb{E}_{(I,V) \sim \mathcal{D}, \tau \sim (\mathcal{P}, \mathcal{M})} [S(\tau)]$

这里：
- $\mathcal{M}$：vision memory system
- $S(\cdot)$：quantifiable performance（accuracy或reward model signal）
- 优化目标是joint optimization of policy model和memory system

### 3.2 Memory Invocation机制

这是VisMem最核心的设计之一。扩展vocabulary $\mathcal{V}$ 到 $\mathcal{V}' = \mathcal{V} \cup \{<m_I^s>, <m_E^s>, <m_I^l>, <m_E^l>\}$

**Token设计细节**：
- `<m_I>`：invocation token，trigger memory formation
- `<m_E>`：end token，structural marker恢复token decoding
- 上标s：short-term memory
- 上标l：long-term memory

**Embedding初始化策略**：invocation tokens用delimiter token embedding + small perturbations初始化，end tokens用类似方式但lower learning rate。这种设计很巧妙——让invocation tokens从一个语义上"中性"的起点开始学习，加速convergence。

**公式(3)**的分支逻辑：
$$x_{t,i} \to \begin{cases} \text{invocation}, & x_{t,i} \in \{<m_I^s>, <m_I^l>\} \\ \text{continue}, & \text{otherwise} \end{cases}$$

**公式(4)**：$x_{t,i} \sim \mathcal{P}(\cdot | s_t, x_{t,<i}, \{m_I, m_1, ..., m_N, m_E\})$

这个公式展示了memory insertion的完整过程：当模型生成invocation token $m_I$后，memory former立即生成latent memory tokens $\{m_1, ..., m_N\}$并插入stream，然后自动append end token $m_E$恢复decoding。

### 3.3 Query Builder B的细节

Query Builder是连接hidden states和memory formers的桥梁，paper在Appendix 7.1给出了详细实现。

**公式(5)**：$\mathbf{Q} = \mathcal{B}([\mathbf{H}, \mathbf{Q}_{init}])[-K:]$

其中：
- $\mathbf{H} = \{v_1, ..., v_y, h_1, ..., h_z\} \in \mathbb{R}^{(y+z) \times d}$：multi-modal cognitive state
  - $v_1, ..., v_y$：visual encoder产生的visual hidden states
  - $h_1, ..., h_z$：language model产生的textual hidden states
  - $y, z$：对应序列长度
  - $d$：model dimension
- $\mathbf{Q}_{init} = \{q_1, ..., q_K\}$：learnable memory query，$q \in \mathbb{R}^d$
- $K$：query sequence length（实验中设为8）
- $[-K:]$：取encoder输出的最后K个vectors

**Transformer Encoder的层结构**（Appendix公式9-10）：

$$\text{SA}(x) = \text{SM}\left(\frac{(xW_q)(xW_k)^T}{\sqrt{d_k}} + M\right)(xW_v)$$

$$x^\ell = \text{FF}(\text{LN}(x^{\ell-1} + \text{SA}(\text{LN}(x^{\ell-1})))) + x^{\ell-1}$$

这里使用了Pre-LN架构（先LayerNorm再attention），这是现代transformer的标准实践，训练更稳定。

**关键设计 - Masked Attention（公式11）**：
$$M_{ij} = \begin{cases} -C, & i < K \text{ and } j \geq K \\ 0, & \text{otherwise} \end{cases}$$

这个mask非常关键！它只允许query tokens attend to hidden states，但禁止hidden states attend to query tokens。这里：
- $i < K$：query tokens的位置
- $j \geq K$：hidden states的位置
- $C \gg 0$：大常数，使attention接近$-\infty$

**Intuition**：这个asymmetric attention设计保证了query是"主动询问者"，而hidden states是"被动信息源"。如果允许双向attention，hidden states的信息会被query污染，影响后续generation。

### 3.4 Latent Memory Former

这是VisMem的另一核心创新。paper设计了两个lightweight LoRA adapters作为memory formers。

**公式(6)**：$\mathbf{M}_{s/l} = \mathcal{F}_{s/l}([\mathbf{X}, \mathbf{Q}, \mathbf{M}_{init}])[-N_{s/l}:]$

其中：
- $\mathbf{X}$：target token sequence
- $\mathbf{Q}$：query builder生成的memory query
- $\mathbf{M}_{init}$：learnable memory tokens
- $\mathbf{M}_{s/l} \in \mathbb{R}^{N_{s/l} \times d}$：生成的latent memory
- $N_s = 4$（short-term），$N_l = 8$（long-term）

**架构区别**：
- **Short-term former** $\mathcal{F}_s$：attached to vision encoder，生成tokens与visual token stream拼接，经过original projector对齐到language model空间
- **Long-term former** $\mathcal{F}_l$：attached to final language model，直接生成semantic-level representations

**为什么这样设计？** 基于Dennis Norris Theory：
- Short-term memory neurologically relies on visual cortex → 所以attach到vision encoder，保留fine-grained perceptual information
- Long-term memory relies on medial temporal lobe（semantic processing）→ 所以attach到language model输出端，捕获abstract semantic knowledge

**LoRA配置**（Table 4）：
- rank: 16
- α: 32
- dropout: 0.1
- target_modules: [q-proj, v-proj]

这里只对attention的query和value projection做LoRA，保持key projection和FFN不变，这是一个computationally efficient的选择。

## 4. Two-Stage Training Recipe

Paper设计了基于GRPO的两阶段训练，这个设计解决了一个重要的**chicken-and-egg problem**：如果同时训练memory formation和invocation，两者会相互干扰——bad memory会让invocation学到错误pattern，bad invocation会让memory formation收到noisy gradient。

### Stage I: Memory Formation Optimization

**公式(7)**：$\max_{\mathcal{F}_{s/l}, B} \mathbb{E}_{\tau \sim \mathcal{P}(\cdot|x, \mathbf{M}_{s/l}), \mathbf{M}_{s/l} \sim \mathcal{F}_{s/l}(\mathbf{Q}), \mathbf{Q} \sim \mathcal{B}(\mathbf{H})} [\Delta S(\tau)]$

这里 $\Delta S(\tau) = S(\tau) - S(\tau_{base})$，即相对于无memory的baseline的性能提升。

**训练策略**：
1. Freeze policy model $\mathcal{P}$
2. 初始时在delimiter处random invoke short或long memory
3. 逐渐扩展invocation scope到delimiter之间的任意位置
4. 这个curriculum-like策略让memory former先学习简单的context，再适应复杂场景

**完整GRPO目标（公式14）**：
$$\mathcal{T}_{GRPO}^{stage1}(\phi) = \mathbb{E}_{\tau, \mathbf{M}_{s/l}, \mathbf{Q}} \left[\frac{1}{G} \sum_{i=1}^G \min(\rho_i(\phi)\hat{A}_i, \text{clip}(\rho_i(\phi), 1-\epsilon, 1+\epsilon)\hat{A}_i)\right] - \beta D_{KL}[\pi_\tau^\phi \| \pi_{ref}^\phi]$$

其中：
- $G$：group size = 16
- $\rho_i(\phi) = \pi^\phi(\tau_i) / \pi_{old}(\tau_i)$：importance ratio
- $\hat{A}_i$：group-relative advantage（公式13）
- $\epsilon$：clip ratio = 0.2
- $\beta$：KL penalty coefficient = 0.015
- $\pi^\phi = \pi^\phi(\mathbf{Q}|\mathbf{H}) \cdot \pi^\phi(\mathbf{M}_{s/l}|\mathbf{Q})$：联合policy

### Stage II: Memory Invocation Optimization

**公式(8)**：$\max_\theta \mathbb{E}_{\tau \sim \mathcal{P}(\cdot|x, \mathbf{M}_{s/l})} [\Delta S(\tau) - \alpha(p_{type} + p_{neg})]$

这里引入两个penalty：

1. **Type penalty**：$p_{type} = \max(0, S(\tau_{rev}) - S(\tau))$
   - $\tau_{rev}$：使用alternative memory type的trajectory
   - 惩罚错误选择memory type的情况
   
2. **Negative penalty**：$p_{neg} = \max(0, \overline{S} - S(\tau))$
   - $\overline{S}$：候选trajectories的平均score
   - 惩罚带来negative return的invocation

**Intuition**：这两个penalty确保模型学会"何时invoke"和"invoke哪种memory"。Type penalty防止short/long混用，negative penalty防止无效invocation（这很重要，因为random invocation实验显示过多invocation反而有害）。

**Training hyperparameters**（Table 4）：
- Stage I: lr = 5e-5, warmup = 0.2, KL target = 0.03
- Stage II: lr = 1e-5, warmup = 0.1, KL target = 0.05, penalty intensity α = 0.3

Stage II用lower learning rate因为只更新部分policy parameters，避免破坏base model能力。

## 5. 实验结果深度分析

### 5.1 主实验结果（Table 1）

12个benchmarks覆盖三大能力：

**Understanding**：MMStar, MMVet, MMT, BLINK, MuirBench
- VisMem平均68.2% vs vanilla 59.3%（+8.9%）
- 相比最强baseline Vision-R1（65.0%）仍有+3.2%提升

**Reasoning**：MMMU, LogicVista, MathVista, MV-Math
- VisMem平均60.2% vs vanilla 46.6%（+14.4%）
- 在MV-Math上提升最显著：41.4% vs 18.9%（+22.5%）

**Generation**：HallBench, MultiTrust, MMVU
- VisMem平均68.3% vs vanilla 57.7%（+10.6%）
- MultiTrust上提升12.2%，说明memory有效减少hallucination

**关键观察**：reasoning的提升（+14.4%）远大于understanding（+8.9%），这符合预期——reasoning需要多步inference，visual forgetting问题更严重，因此memory带来的gain更大。

### 5.2 Cross-Model Compatibility（Table 2）

这是非常impressive的结果——VisMem在9个不同base models上都有效：

| Base Model | Size | Avg Improvement |
|------------|------|-----------------|
| Qwen2.5-VL-3B | 3B | +11.7% |
| Qwen2.5-VL-7B | 7B | +9.7% |
| Qwen2.5-VL-32B | 32B | +10.2% |
| LLaVA-OV-1.5-4B | 4B | +13.0% |
| LLaVA-OV-1.5-8B | 8B | +10.4% |
| InternVL-3.5-4B | 4B | +10.2% |
| InternVL-3.5-8B | 8B | +8.6% |
| InternVL-3.5-14B | 14B | +8.0% |
| InternVL-3.5-38B | 38B | +6.9% |

**Interesting pattern**：smaller models往往获得更大improvement（3B的BLINK +18.3% vs 38B的+7.2%）。这暗示larger models本身已经internalize了一些memory功能，而smaller models更依赖external memory augmentation。

但larger models在dense reasoning benchmarks（如MV-Math）上gain更显著：38B的MV-Math从35.7%到56.9%（+21.2%）。这说明memory帮助larger models突破reasoning瓶颈。

### 5.3 Ablation Studies（Table 3, 9）

**Random Invocation实验**：
- 25% invocation: MMVet 69.2%（vs vanilla 66.0%）
- 50%: 71.9%
- 75%: 73.6%（peak）
- 100%: 73.4%（开始下降）

这个non-monotonic trend非常有意思——说明memory invocation有一个"sweet spot"。过多invocation会导致information redundancy或distribution shift。

**Single Memory Type**：
- Only Short-term: MMVet 71.5%, MuirBench 65.6%, MV-Math 29.6%, MultiTrust 73.6%
- Only Long-term: MMVet 69.4%, MuirBench 60.2%, MV-Math 36.1%, MultiTrust 69.8%
- Complete: 75.1%, 69.8%, 41.4%, 77.0%

**Task-specific advantages**：
- Short-term memory在MuirBench（multi-image understanding）和MultiTrust（generation robustness）上更强
- Long-term memory在MV-Math（mathematical reasoning）上更强（36.1% vs 29.6%）
- 两者结合在所有任务上都最优

这验证了dual memory system的设计——不同任务需要不同类型的memory support。

### 5.4 Memory Invocation分析（Figure 5, 9）

Paper分析了memory invocation的ratio和relative position：

**Invocation ratio的task-specificity**：
- MuirBench：short-term invocation显著高于long-term（multi-image需要fine-grained perception）
- MV-Math：long-term invocation更高（mathematical reasoning需要semantic knowledge）
- MultiTrust和MMVet：两者相对均衡

**Relative position trend**：两种memory的invocation都呈现**downward trend**——在output sequence早期invocation更频繁，后期减少。这符合intuition：早期需要establish visual grounding，后期主要是textual reasoning。

这个自适应行为完全是通过Stage II的reinforcement learning学到的，而非hardcoded rules。

### 5.5 Sensitivity Analysis（Table 10, 11）

**Memory query length K**：
- K=2: MMVet 69.6%
- K=8: 73.1%
- K=32: 74.5%

**Short memory length $N_s$**：
- $N_s=2$: 67.2%
- $N_s=8$: 71.8%
- $N_s=32$: 73.0%

**Long memory length $N_l$**：
- $N_l=2$: 66.4%
- $N_l=8$: 69.7%
- $N_l=32$: 70.8%

Performance随length增加而提升，但存在diminishing returns。Paper选择 $K=8, N_s=4, N_l=8$ 作为effectiveness-efficiency的平衡点。

### 5.6 Catastrophic Forgetting Mitigation（Table 8, Figure 4）

这是VisMem相对于direct training methods的最大优势之一。

**Four-stage continual learning**：
- Stage 0: MMVet training only
- Stage 1: + BLINK, MuirBench
- Stage 2: + LogicVista, MathVista
- Stage 3: + MultiTrust, MMVU

**Results on MMVet**：
- Vanilla: 66.0 → 66.0（stable但low performance）
- SFT: 71.4 → 60.1（catastrophic forgetting！下降11.3%）
- VLM-R1: 77.8 → 66.4（initial gain完全消失）
- Vision-R1: 74.5 → 62.9（forgetting）
- VisMem: 78.6 → 72.1（retention rate 91.6%）

**Why VisMem mitigates forgetting?** 因为memory存储在lightweight LoRA adapters中，不修改core VLM parameters。新任务训练时，只更新memory formers和invocation policy，原有knowledge被保留。

更impressive的是，VisMem在Stage 1和Stage 3的性能甚至**高于**Stage 0，说明存在positive forward transfer——新任务的memory training反而帮助了原始任务。

### 5.7 Inference Efficiency（Table 12, Figure 6）

**Time overhead comparison**：
- Vanilla: 0.76s (MMVet)
- VisMem: 0.84s (+10.5%)
- DeepEyes: 3.21s (+322%)
- OpenThinkImg: 3.68s (+384%)
- VPT: 2.98s (+292%)

VisMem的overhead仅8.2%-43.8%，与direct training methods（如VLM-R1的+1.3%）和token-level methods（如MINT-CoT的+6.6%）comparable，但远低于image-level methods。

**Speed-performance trade-off**：从Figure 6的bubble plot可以看出，VisMem位于左上角（high performance, low latency），而image-level methods位于右下方（high latency, moderate performance）。

## 6. Sub-task Analysis（Table 5, 6）

### MuirBench Subsets

VisMem在9个subsets中的7个达到最优：

| Subset | Vanilla | VLM-R1 | Vision-R1 | VisMem | Improvement |
|--------|---------|--------|-----------|--------|-------------|
| Counting | 44.1 | 52.5 | 53.8 | 60.8 | +16.7% |
| Grounding | 34.2 | 38.1 | 39.2 | 52.3 | +18.2% |
| Geographic | 53.7 | 56.7 | 57.9 | 65.5 | +11.8% |
| Retrieval | 76.1 | 79.4 | 78.9 | 89.8 | +13.7% |

这些fine-grained perception tasks的提升主要来自short-term memory。

### LogicVista Subsets

在reasoning和capability subsets上：

| Subset | Vanilla | VisMem | Improvement |
|--------|---------|--------|-------------|
| Inductive | 44.6 | 59.4 | +14.8% |
| Deductive | 45.0 | 59.8 | +14.8% |
| Graphs | 34.4 | 52.8 | +18.4% |
| Tables | 36.8 | 57.9 | +21.1% |

这些abstract reasoning tasks的提升主要来自long-term memory提供的semantic knowledge。

## 7. 与相关工作的深度对比

### vs. Coconut ([arXiv:2412.06769](https://arxiv.org/abs/2412.06769))
Coconut是latent CoT的先驱，在language space用continuous latent tokens替代textual CoT。VisMem借鉴了latent space的idea但扩展到visual domain，并引入dual memory system。

### vs. Mirage ([arXiv:2506.17218](https://arxiv.org/abs/2506.17218))
Mirage首次尝试latent visual space，但需要substantial manually labeled images来训练。VisMem通过RL-based training避免了这一限制，并且区分了short/long-term memory的功能。

### vs. MVoT ([arXiv:2503.01815](https://arxiv.org/abs/2503.01815))
MVoT（Multimodal Visualization-of-Thought）让模型在pixel space imagine visual reasoning过程。这是image-level paradigm的extreme case，computational cost极高。VisMem用latent space实现类似功能但efficient得多。

### vs. VLM-R1 ([arXiv:2504.07615](https://arxiv.org/abs/2504.07615))
VLM-R1使用GRPO直接优化VLM的reasoning能力。虽然initial performance gain显著，但suffer from catastrophic forgetting。VisMem的memory-based approach避免了这个问题，并且在continual learning中表现更稳定。

### vs. Visual CoT ([arXiv:2402.09211](https://arxiv.org/abs/2402.09211))
Visual CoT通过bounding boxes提供visual evidence during CoT。这是image-level paradigm的代表，需要explicit visual annotation。VisMem用latent representations替代了explicit visual annotation。

## 8. Theoretical Foundations（Appendix 6）

Paper基于Dennis Norris Theory ([Psychological Bulletin, 2017](https://psycnet.apa.org/record/2017-26421-001))：

**Short-term memory**：
- Functionally：processing new visual information, temporarily storing multiple tokens, enabling variable signals
- Neurologically：relies on visual cortex和posterior superior temporal lobe
- Character：visually dominant

**Long-term memory**：
- Functionally：abstract semantic representations, generalized knowledge
- Neurologically：relies on medial temporal lobe和mid-temporal lobe
- Character：semantically dominant

VisMem通过以下方式align这一理论：
- Short-term former attached to vision encoder → 模拟visual cortex功能
- Long-term former attached to language model → 模拟semantic processing regions
- Dynamic invocation mechanism → 模拟working memory的灵活调用

## 9. Critical Analysis与Future Directions

### 9.1 Strengths
1. **Theoretical grounding**：基于cognitive psychology，不是ad-hoc设计
2. **Non-intrusive**：通过LoRA和special tokens，不破坏base model
3. **Dynamic adaptation**：RL学到的invocation策略task-adaptive
4. **Strong empirical results**：在12个benchmarks上一致提升
5. **Cross-model compatibility**：9个base models都有效

### 9.2 Potential Limitations

1. **Memory capacity**：固定长度 $N_s=4, N_l=8$ 可能不足以应对非常complex tasks。未来可以探索dynamic memory length。

2. **Memory interpretation**：latent memory tokens的semantics不transparent。可视化分析（如attention rollout, causal tracing）可以提供更多insight。

3. **Training cost**：两阶段RL training需要大量GPU hours。Paper使用8×H200，对academic labs可能prohibitive。

4. **Memory consolidation**：目前long-term memory是task-specific的。如果能让memory across sessions积累和consolidate，可能实现更general的knowledge transfer。

### 9.3 Future Directions

1. **Hierarchical Memory**：引入working memory（ultra short-term）和episodic memory（session-level）的更精细分级。

2. **Memory Compression**：用vector quantization或autoencoder压缩memory representations，支持更大memory capacity。

3. **Cross-modal Memory**：探索audio、video等modalities的latent memory。

4. **Meta-learning Memory**：让模型学会how to learn memory，实现few-shot adaptation到新domains。

5. **Memory Attention Visualization**：开发工具可视化memory如何influence generation，提升interpretability。

## 10. 实现细节补充

### Memory Token Insertion的工程实现

从代码实现角度，memory insertion需要在autoregressive decoding的KV-cache中插入对应tokens的keys和values。这要求：

1. **Constrained decoding**：当生成invocation token时，强制下一个token是memory token，然后是end token
2. **KV-cache management**：memory tokens的KV需要正确插入到cache的相应位置
3. **Position encoding**：memory tokens的position IDs需要正确处理，避免位置混乱

### 训练数据构建

Paper使用多个benchmarks的训练split，加上Visual CoT和Mulberry数据集增强reasoning能力。关键的是，**VisMem不需要额外标注的memory targets**——memory formation是通过RL的reward signal间接学习的。

这种implicit supervision是VisMem的elegant之处：模型自己探索什么样的memory representations有助于task performance，而不是human-specify memory content。

## 11. 总结

VisMem提出了一个cognitively-inspired framework，通过dual latent vision memory系统解决VLMs的visual processing bottleneck。核心创新包括：

1. **Dual memory architecture**：short-term（visually dominant）+ long-term（semantically dominant），align human cognitive theory
2. **Dynamic invocation**：通过special tokens和RL-learned policy实现自适应memory调用
3. **Non-intrusive design**：LoRA-based formers保护base model能力
4. **Two-stage training**：解耦memory formation和invocation learning，避免chicken-and-egg problem

实验结果demonstrate了11.0%的平均提升，在9个base models上consistent improvement，并且有效mitigate catastrophic forgetting。这项工作为latent space memory augmentation开辟了new paradigm，potential applications extends到更广泛的multi-modal reasoning scenarios。

**Reference Links**：
- [VisMem GitHub (即将发布)](https://github.com/YU-deep/VisMem.git)
- [Dennis Norris Theory - Psychological Bulletin](https://psycnet.apa.org/record/2017-26421-001)
- [Coconut - Training LLMs to Reason in Continuous Latent Space](https://arxiv.org/abs/2412.06769)
- [Mirage - Machine Mental Imagery](https://arxiv.org/abs/2506.17218)
- [VLM-R1](https://arxiv.org/abs/2504.07615)
- [Visual CoT](https://arxiv.org/abs/2402.09211)
- [DeepEyes](https://arxiv.org/abs/2505.14362)
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- [GRPO - DeepSeekMath](https://arxiv.org/abs/2402.03300)
