---
source_pdf: Whenalanguage model is optimized for reasoning.pdf
paper_sha256: ac1f9f51d589d2ff0b9d09b033c0bedb2c4b799a4fbd88a50304eee4fe5477f3
processed_at: '2026-08-13T04:19:46-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：o1还带着autoregression的"基因"吗？

## 这paper到底在问什么

一句话：**o1被RL训练成reasoning machine之后，它骨子里那个"next-word predictor"的本性还在不在？**

答案是：在。表面上o1能搞定很多previous LLMs搞不定的事情，但你仔细看它的behavior pattern，那个"sensitive to probability"的影子还在。

---

## 为什么这个question有意思

你想啊，传统LLM的训练objective特别简单：

$$\mathcal{L} = -\sum_t \log P(w_t \mid w_{1:t-1})$$

就是predict下一个token。这个objective在model脑子里刻下了一个非常深的"习惯"——它天生倾向于产生高概率的text。

McCoy他们2023年那篇paper就发现这个习惯会留下两个behavioral signature：

1. **Output probability sensitivity**：如果correct answer本身在LM看来是"奇怪"的、低概率的string，model就做得差
2. **Task frequency sensitivity**：如果这个task在training data里很少见（比如"取每个word的第二个字母组成acronym"），model就做得差

这就叫"embers of autoregression"——autoregression的余烬。即使你让model做reasoning，它的reasoning还是被probability shape着。

那o1来了。o1不一样：它是explicitly用RL trained to reason的，不是单纯的next-word predictor了。所以一个natural question：**RL这把"火"能不能把autoregression的余烬彻底烧掉？**

---

## 怎么测的

作者用了两类manipulation：

### 第一类：让answer的probability变

同一个task，比如"反转一个word list"：
- 如果反转后的结果刚好是一个high-probability的English sentence，model做得好
- 如果反转后的结果是一串"不像话"的tokens，model做得差

四个tasks：shift cipher decoding, Pig Latin, article swapping, reversal。

### 第二类：让task的frequency变

同一个task family，分common和rare variant：
- Common：取每个word的**第一个**字母组成acronym（training data里到处都是）
- Rare：取每个word的**第二个**字母组成acronym（training data里几乎没有）

五个task pairs。

---

## 看到了什么

### Finding 1：Output probability sensitivity——还在

Figure 1的数据非常清楚。o1在四个tasks上全部表现出：high-probability outputs的accuracy显著高于low-probability outputs。

比如shift cipher：
- low-probability answer：47% accuracy
- high-probability answer：92% accuracy

差了45个点。这跟previous LLMs的pattern qualitatively完全一样，只是整体level更高了。

### Finding 2：Thinking tokens是个新lens

这是这篇paper最聪明的设计。OpenAI不reveal chain of thought的内容，只reveal token数量。作者就把这个数量拿来当difficulty proxy。

Figure 2显示：**low-probability examples需要显著更多thinking tokens**。

这是一个非常nice的finding，因为accuracy可能在ceiling处看不出差别，但token count能。Model在"挣扎"的时候会"叹气"更多次。

### Finding 3：Task frequency sensitivity——大部分被磨掉了，但没完全磨掉

Figure 3左图：o1在common和rare variants上accuracy几乎一样，而GPT-4、Llama-2等都在rare variant上collapse。

乍看o1好像不sensitive了。但作者很谨慎，说这可能是ceiling effect——dataset太简单，model都接近100%，看不出差别。

所以他们做了两个harder versions：

**Sorting hard mode**：让所有words都以"i"开头，迫使model看前两个字母。
- Common（alphabetical）：明显好
- Rare（reverse alphabetical）：明显差

**Shift cipher with low-probability outputs**：
- High-probability下：common ≈ rare
- Medium-probability下：common > rare  
- Low-probability下：common >> rare

这是非常漂亮的**interaction effect**。Output probability和task frequency两个factor会**叠加**。当其中一个有利时，可能mask另一个的效应；当两个都不利时，effect才fully显现。

### Finding 4：Token count能reveal accuracy看不出的sensitivity

Figure 4最striking：

Acronyms task：
- Common variant：accuracy 100%, tokens ~50
- Rare variant：accuracy 99.9%, tokens ~150

**Accuracy几乎一样，但rare variant花了3倍的thinking tokens**。

这意味着task frequency sensitivity在accuracy层面被ceiling mask了，但在compute层面完全visible。o1在rare variant上"做对了"，但"费力得多"。

---

## 为什么会这样

作者给两个hypothesis：

### Hypothesis A：Generation-level bias

CoT的生成本身是autoregressive的。每一步：

$$P(c_t \mid x, c_{1:t-1})$$

这个conditional probability还是被pretraining的LM manifold shape着。RL改了"drift direction"，但没改"landscape geometry"。

打个比方：pretraining给model一个"地形"，高概率区域是平原，低概率区域是山地。RL像给model装了个引擎，让它能爬山。但爬山的时候，model还是prefer走在平原上。

### Hypothesis B：Selection-level bias

如果o1内部有"生成多条CoT然后选一条"的机制，selection可能implicit地favor高概率的chains/answers。

这跟AlphaGo对比很有意思。AlphaGo是pure RL，policy network也是neural network，但training signal是win率，不是text probability。所以AlphaGo不会"prefer高概率move"——它prefer高win率move。

o1本质上还是**LM + RL wrapper**。Pretraining那阶段的probability imprint渗进了RL后的generation process。

---

## Intuition层面的类比

我自己想到几个analogy：

### Analogy 1：基因 vs 训练

Pretraining像model的"基因"，RL like"后天训练"。你训练一个人成为 Olympic weightlifter，他的performance确实大幅提升。但他的skeletal structure、muscle fiber distribution这些"先天bias"还在。训练不能改变骨架，只是在骨架上optimize。

### Analogy 2：Optimization landscape

想象loss landscape是一个3D surface。Pretraining把surface塑造成有deep valleys in high-probability regions。RL在surface上加了一个"reward wind"，推model往reasoning-correct方向走。但wind只能改变trajectory，不能reshape the valleys。

Model在低概率区域要"逆风"做reasoning，所以更费力（更多tokens），且更容易被wind吹回高概率区域（accuracy下降）。

### Analogy 3：Karpathy你自己的"software 2.0"

你之前讲过software 2.0——用data替代code。LLM是software 2.0的极致。但reasoning某种程度上需要software 1.0的determinism——精确的symbolic manipulation、verifiable的step。

o1是hybrid。RL让model学会"做software 1.0 style的事情"，但model本身还是software 2.0的artifact。所以它的reasoning带有software 2.0的特征：probabilistic、sensitive to distribution、approximate而非exact。

Embers of autoregression本质上就是**software 2.0 artifact做software 1.0 task时的"指纹"**。

---

## 这意味着什么

### Implication 1：Test-time compute不是万能的

Low-probability cases需要更多tokens，但即便用了更多tokens，accuracy还是更低。Scaling test-time compute能帮你爬更高的山，但山的高度本身受probability landscape限制。

参考[Snell et al. 2024](https://arxiv.org/abs/2408.03314)关于test-time compute scaling的工作。

### Implication 2：要真正消除embers，需要架构创新

作者结尾说：可能需要"non-probabilistic modules"，比如Python code execution。

这指向**neuro-symbolic**：用LM生成candidates，用symbolic verifier过滤、verify、execute。AlphaProof、AlphaGeometry就是这条路。

### Implication 3：Probability sensitivity作为"alignment tax"

从safety角度，probability bias其实是feature：model倾向于"正常"的输出，不会突然产生weird stuff。但这同时是bug：model在counterfactual、counterintuitive reasoning上struggle。

那些"trick questions"——correct answer是low-probability的——会一直是个soft spot。

---

## 我的take

这篇paper最valuable的contribution其实不是"o1还有embers"这个conclusion——这有点expected。最valuable的是：

1. **Thinking tokens作为新的analysis lens**：accuracy saturates之后，token count还能reveal underlying difficulty。这是一个很好的methodology contribution。

2. **Ceiling effect的处理**：通过making tasks harder来unmask被隐藏的effects。这是很好的experimental design。

3. **Interaction effects**：output probability × task frequency两个factor会叠加。这个finding让"embers"这个concept更precise了。

4. **Teleological perspective的validation**：通过分析training pressures来predict model behavior——这个framework被这篇paper进一步validated。

对你Karpathy来说，我觉得最interesting的angle是：**这paper给了一个quantitative way to measure "how much RL can override pretraining"**。答案是：能override很多，但override不彻底。Pretraining的inductive bias是一个"conservation law"——你可以把它从一个地方push到另一个地方，但很难eliminate。

这跟你自己之前说过的"RLHF只是微调surface behavior，不会改变model的core capabilities"是同一个insight的延伸。现在这个insight延伸到了reasoning上：**RL for reasoning也只是微调surface reasoning pattern，不会改变model的core probability manifold**。

---

## 一个speculation

如果这个hypothesis是对的——pretraining的probability manifold是"conserved"的——那未来真正突破reasoning的系统，可能不是"更强的RL on top of stronger pretraining"，而是**架构上就不同于autoregressive LM的东西**。

比如：
- Non-autoregressive generation（像diffusion models for text，虽然现在还不成熟）
- Explicit symbolic reasoning modules wired into the architecture
- Verifier-driven search where the verifier is not a probability model
- Hybrid systems where LM只做"proposal"，deterministic engine做"verification"

o3、o4如果继续走"更强的RL on autoregressive LM"这条路，我predict embers会dim一些，但不会灭。要灭掉embers，需要architecture层面的paradigm shift。

这是我个人speculation，paper本身没说这么远。

---

## References

1. [McCoy et al. 2023 - Embers of Autoregression (original)](https://arxiv.org/abs/2309.13638)
2. [Prabhakar et al. 2024 - CoT factors](https://arxiv.org/abs/2407.01687)
3. [Snell et al. 2024 - Scaling test-time compute](https://arxiv.org/abs/2408.03314)
4. [OpenAI o1 blog](https://openai.com/index/learning-to-reason-with-llms/)
5. [Bubeck et al. 2023 - Sparks of AGI](https://arxiv.org/abs/2303.12712)
6. [Wei et al. 2022 - Chain-of-thought prompting](https://arxiv.org/abs/2201.11903)
7. [AlphaProof / AlphaGeometry - DeepMind](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)

---

如果你想go deeper，我会推荐接下来看：
- [Noam Brown的talk on reasoning vs search](https://www.youtube.com/watch?v=YM5a7VHGqMA) — 关于什么时候search beats RL
- [Anthropic的reasoning models interpretability工作](https://transformer-circuits.pub/) — 看CoT里面到底在干嘛
- [Yi Tay等人的non-autoregressive generation工作](https://arxiv.org/abs/2202.04826) — 架构alternative

这三个方向分别tackle"search能否替代probability"、"CoT内部mechanism"、"架构能不能escape autoregression"——都是这篇paper留下的open questions的自然延伸。

---

# 《When a language model is optimized for reasoning, does it still show embers of autoregression?》深度解析

这篇paper是McCoy et al. (2023)那篇《Embers of Autoregression》的延续工作，专门针对OpenAI o1做的分析。核心question非常深刻：**当一个system被explicitly optimize for reasoning之后，它作为language model的autoregressive nature还能不能被磨掉？**

答案很扎心：quantitative improvements很大，但qualitative patterns依然存在。

---

## 1. 理论框架：Teleological Perspective

作者采用的视角来自Marr (1982)、Shepard (1987)、Anderson (1990)、Griffiths (2020)的"目的论"分析——通过分析塑造一个system的**pressures**来预测它的behavior。

这个framework本质上是一个Marr's levels的特例：

| Level | 对应内容 |
|-------|---------|
| Computational | Next-word prediction / Reasoning |
| Algorithmic | Autoregressive generation / Chain of thought |
| Implementational | Transformer architecture |

**核心hypothesis**：一个system的training objective会留下behavioral signatures，即使后来被fine-tune去做别的任务。

---

## 2. Autoregression的数学formulation

标准language model的training objective是minimize negative log-likelihood：

$$\mathcal{L}_{\text{AR}}(\theta) = -\sum_{t=1}^{T} \log P_\theta(w_t \mid w_{1:t-1})$$

其中：
- $w_t$：第 $t$ 个token
- $w_{1:t-1}$：前 $t-1$ 个tokens组成的context
- $\theta$：model parameters
- $P_\theta$：由neural network参数化的conditional distribution

这个objective隐式地impose了一个强烈的**inductive bias**：model倾向于产生在training distribution下高概率的sequences。

**Embers of Autoregression**的核心insight：这种bias会manifest为两种sensitivity：

$$\text{Performance}(x, y, T) = f\big(P_{\text{LM}}(y), P_{\text{LM}}(T)\big)$$

其中：
- $P_{\text{LM}}(y)$：output string $y$ 在language model下的probability
- $P_{\text{LM}}(T)$：task $T$ 在training distribution中的frequency

---

## 3. o1的特殊之处

o1和previous LLMs的关键difference：

| 维度 | Traditional LLMs | o1 |
|------|-----------------|-----|
| Primary objective | Next-word prediction | Reasoning (via RL) |
| Generation | Single-pass autoregression | Hidden chain-of-thought + final answer |
| Observable | Full output | Final answer + **thinking tokens count** |
| Training signal | Text likelihood | Reasoning correctness (reward) |

**关键设计决策**：OpenAI只reveal thinking tokens的**数量**，不reveal内容。这给作者创造了一个新颖的analysis维度——用token count作为**difficulty proxy**。

公式上，可以把o1看作：

$$P_{\text{o1}}(y \mid x) = \sum_{c \in \mathcal{C}} P(c \mid x) \cdot P(y \mid x, c)$$

其中 $c$ 是一条chain of thought，$\mathcal{C}$ 是所有可能的CoT集合。RL优化的目标是：

$$\max_\theta \mathbb{E}_{x, c \sim P_\theta}\big[R(y, y^*)\big]$$

$R$ 是reward function，$y^*$ 是ground truth。

---

## 4. Experiment 1: Output Probability Sensitivity

### 4.1 Setup

四个tasks：
1. **Shift ciphers**：Caesar cipher decoding
2. **Pig Latin**：解码猪拉丁语
3. **Article swapping**：冠词与前后词交换位置
4. **Reversal**：反转word list

关键manipulation：同一个task，output string的language model probability有高低之分。

### 4.2 Results

Figure 1的核心数据（accuracy）：

| Task | Low prob | High prob | 差距 |
|------|----------|-----------|------|
| Shift cipher | 47% | 92% | 45% |
| Pig Latin | ~75% | ~95% | 20% |
| Article swap | ~85% | ~98% | 13% |
| Reversal | ~70% | ~95% | 25% |

**o1的performance**：
- 定性上：所有tasks都显示probability sensitivity（slope显著非零）
- 定量上：比GPT-3.5、Llama等有显著提升，尤其在article swapping

### 4.3 Thinking tokens分析

Figure 2显示：low-probability examples需要**显著更多**的thinking tokens。

这是一个很有意思的finding。可能的interpretation：

$$\text{Difficulty}(x) \propto \text{Tokens}(x) \propto -\log P_{\text{LM}}(y^*)$$

即thinking tokens数量近似与output的negative log-probability成正比。这暗示o1的internal search过程可能在某种意义上**与LM probability耦合**。

---

## 5. Experiment 2: Task Frequency Sensitivity

### 5.1 Setup

五个task pairs（common vs rare variant）：

| Task | Common variant | Rare variant |
|------|---------------|--------------|
| Shift ciphers | +13 offset | +12 offset |
| Pig Latin | Encoding | Decoding |
| Acronyms | First letter | Second letter |
| Linear functions | slope=2 | slope=3 |
| Sorting | Alphabetical | Reverse alphabetical |

### 5.2 初步结果：Ceiling effects

Figure 3左图：o1在common和rare variants上表现**几乎相同**。这与其他LLMs（GPT-4、Llama-2等）形成鲜明对比——后者在rare variants上collapse。

但作者很谨慎：这可能是ceiling effect。当accuracy接近100%时，无法检测frequency sensitivity。

### 5.3 Harder versions：打破ceiling

**Sorting challenge**：让所有words都以字母"i"开头，迫使model至少看前两个字母。

结果（Figure 3 top right）：
- Common（alphabetical）：accuracy显著高
- Rare（reverse alphabetical）：accuracy显著低

**Shift cipher challenge**：用medium/low probability的examples。

结果（Figure 3 bottom right）：
- High prob：common ≈ rare
- Medium prob：common > rare
- Low prob：common >> rare

这是一个非常漂亮的**interaction effect**：

$$\text{Accuracy} = g\big(P_{\text{LM}}(y) \times P_{\text{LM}}(T)\big)$$

当两个factor都不利时，effect叠加显现；当其中一个有利时，可能mask另一个。

### 5.4 Tokens分析揭示hidden sensitivity

Figure 4最striking的finding：

**Acronyms task**：
- Common variant accuracy：100%
- Rare variant accuracy：99.9%
- **Common tokens：~50**
- **Rare tokens：~150**

这意味着：即使accuracy相同，o1在rare variant上**费力3倍**。Token count是一个比accuracy更sensitive的difficulty metric。

这让我联想到compute-optimal scaling的思路——**有效推理**应该用与任务难度匹配的compute。

---

## 6. 深层interpretation

### 6.1 为什么reasoning optimization不能消除probability sensitivity？

作者提出两个hypothesis：

**Hypothesis A: Generation-level bias**

CoT生成过程本身是autoregressive。即使RL优化了最终reward，每一步token sampling仍然受 $P_\theta(w_t \mid w_{1:t-1})$ 影响。

$$P_{\text{CoT}}(c \mid x) = \prod_{t} P_\theta(c_t \mid x, c_{1:t-1})$$

high-probability的CoT路径会被preferentially sampled。

**Hypothesis B: Selection-level bias**

如果o1内部有"多条CoT候选+选择"机制：

$$P(y \mid x) = \sum_c P(c \mid x) \cdot \text{select}(c)$$

selection可能基于某种implicit probability scoring，偏向高概率answers。

### 6.2 与AlphaGo的对比

这非常有启发性。AlphaGo/AlphaZero是pure RL system，没有任何language modeling component。它们不会显示probability sensitivity，因为：

$$P_{\text{AlphaGo}}(\text{move}) = f(\text{policy network} + \text{MCTS})$$

policy network虽然也是neural network，但training signal来自**win率**而非**文本概率**。

而o1本质上是**language model + RL wrapper**，pretraining阶段的probability imprint无法被RL完全override。

### 6.3 Pretraining的不可逆性

这呼应了一个深刻的问题：**pretraining的inductive bias是否可以逆转？**

形式化地，假设model parameters $\theta$ 经过pretraining得到 $\theta_0$，然后RL fine-tuning得到 $\theta_{\text{RL}}$。在KL-regularized RL中：

$$\mathcal{L}_{\text{RL}}(\theta) = \mathbb{E}\big[R(y)\big] - \beta \cdot \text{KL}\big(P_\theta \parallel P_{\theta_0}\big)$$

$\beta$ 控制偏离pretraining distribution的程度。即使 $\beta \to 0$，optimization landscape的geometry仍然由 $\theta_0$ 附近的basin决定，pretraining的bias会以"soft"形式persist。

参考：[KL-regularized RLHF](https://arxiv.org/abs/2203.02155)

---

## 7. 与相关工作的连接

### 7.1 Prabhakar et al. (2024)

作者引用这篇来说明CoT系统对probability的sensitivity：

[Prabhakar et al. 2024 - Deciphering the factors influencing the efficacy of chain-of-thought](https://arxiv.org/abs/2407.01687)

核心finding：CoT reasoning在high-memorization、low-noise、high-probability设置下最有效。这支持Hypothesis A。

### 7.2 Bubeck et al. (2023) "Sparks of AGI"

[Sparks of AGI paper](https://arxiv.org/abs/2303.12712)

作者在结尾引用"sparks of AGI continue to be accompanied by embers of autoregression"——非常诗意，也很accurate。意思是：GPT-4/o1展示出AGI的**sparks**，但同时带着autoregression的**embers**（余烬，会复燃）。

### 7.3 Chain-of-thought lineage

- [Nye et al. 2021 - Scratchpads](https://arxiv.org/abs/2112.00114)
- [Wei et al. 2022 - CoT prompting](https://arxiv.org/abs/2201.11903)
- [Kojima et al. 2022 - Zero-shot CoT](https://arxiv.org/abs/2205.11916)

o1是这条lineage的延伸，但把CoT从**prompting trick**变成**explicit training target**。

### 7.4 McCoy et al. (2023) 原始paper

[Embers of Autoregression](https://arxiv.org/abs/2309.13638)

这是base paper，包含完整的task suite设计。本paper只evaluated a subset，因为o1成本高。

---

## 8. 实验细节与潜在critique

### 8.1 Sample size问题

paper提到"o1 has a fairly high cost per example"，所以只用了原dataset的subset。这可能导致：
- 标准误较大（看Figure 1的error bars确实不算小）
- 某些effects可能不显著

### 8.2 Thinking tokens的解读问题

thinking tokens count是一个**coarse** metric。我们不知道：
- 是model在挣扎（low confidence路径长）
- 还是model在verify（deliberate reasoning）
- 还是model在explore multiple paths

同样accuracy下，更多tokens可能意味着：
- 更仔细的reasoning（positive interpretation）
- 更多wasted computation on dead ends（negative interpretation）

### 8.3 o1版本问题

实验用的是o1-preview-2024-09-12。后续o1、o1-mini、o3可能有不同behavior。特别是如果o3用了更激进的RL + 更强的search，probability sensitivity可能进一步降低。

参考：[OpenAI o1 blog](https://openai.com/index/learning-to-reason-with-llms/)

### 8.4 Task selection bias

选择的tasks都是**symbolic manipulation**类（ciphers、acronyms、sorting）。这些tasks：
- 有明确correct answer
- 容易构造probability manipulation
- 但可能与o1最strength的math/code reasoning距离较远

o1在math/code上的probability sensitivity可能更弱，因为这些domain的RL training更密集。

---

## 9. 对未来工作的implication

### 9.1 Test-time compute scaling的limits

本paper暗示：test-time compute（thinking tokens）虽然是有效的difficulty knob，但**不能完全compensate** probability bias。Low-probability cases需要更多tokens，但仍achieving lower accuracy。

这关联到[Snell et al. 2024 - Scaling test-time compute](https://arxiv.org/abs/2408.03314)。

### 9.2 真正消除embers的可能路径

作者在conclusion提到：incorporate non-probabilistic modules（如Python code execution）。这其实指向：
- **Tool use**作为reasoning的"scaffolding"
- **Symbolic verification**覆盖probabilistic generation
- **Hybrid neuro-symbolic systems**

这与AlphaProof、AlphaGeometry的approach类似：用language model生成candidates，用symbolic verifier过滤。

### 9.3 Probability sensitivity作为alignment tax

从safety角度，probability sensitivity是双刃剑：
- Positive：model倾向于"common sense" outputs，避免weird/unhinged responses
- Negative：model难以做真正counterfactual、counterintuitive reasoning

这可能解释为什么o1在"trick questions"上仍然struggle——这些questions的correct answer往往是low-probability的。

---

## 10. 我的intuition summary

这篇paper给了一个非常clean的empirical finding：

$$\boxed{\text{RL for reasoning} \neq \text{erasure of autoregressive inductive bias}}$$

o1用RL把performance surface整体提升了，但surface的**shape**——probability sensitivity的slope——qualitatively unchanged。

类比一下：pretraining像给model一个"惯性参考系"，RL是外力。外力可以改变速度，但惯性参考系本身（probability manifold的geometry）不变。要真正消除这种bias，需要的不是更强的外力，而是**架构层面的non-autoregressive reasoning**——比如explicit symbolic search、verifier-guided decoding、或neuro-symbolic hybrid。

这也呼应了Karpathy你之前讲过的"software 2.0"vs"software 1.0"的tension：autoregressive LM是software 2.0的极致体现，但reasoning某种程度上需要software 1.0的determinism和verifiability。o1是两者的hybrid，但hybrid的天平仍偏向2.0一边。

---

## References

1. [McCoy et al. 2023 - Embers of Autoregression](https://arxiv.org/abs/2309.13638)
2. [Prabhakar et al. 2024 - CoT factors](https://arxiv.org/abs/2407.01687)
3. [Bubeck et al. 2023 - Sparks of AGI](https://arxiv.org/abs/2303.12712)
4. [Wei et al. 2022 - CoT prompting](https://arxiv.org/abs/2201.11903)
5. [Kojima et al. 2022 - Zero-shot CoT](https://arxiv.org/abs/2205.11916)
6. [Nye et al. 2021 - Scratchpads](https://arxiv.org/abs/2112.00114)
7. [OpenAI o1 blog](https://openai.com/index/learning-to-reason-with-llms/)
8. [Snell et al. 2024 - Scaling test-time compute](https://arxiv.org/abs/2408.03314)
9. [Ouyang et al. 2022 - InstructGPT/RLHF](https://arxiv.org/abs/2203.02155)
10. [Marr 1982 - Vision](https://mitpress.mit.edu/9780262512820/vision/)

---

如果你想深入，我建议下一个值得读的是Anthropic最近关于**interpretability of reasoning models**的工作，以及Noam Brown关于**search vs RL**的talks——这两个方向直接tackle"如何让reasoning真正脱离probability bias"的问题。
