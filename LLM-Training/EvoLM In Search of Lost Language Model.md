---
source_pdf: EvoLM In Search of Lost Language Model.pdf
paper_sha256: 15e5ebea59606b5e2a23358a9f17eac0d0c92bd4052db269ac0a99c05e6bea61
processed_at: '2026-08-04T06:03:38-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# EvoLM 用人话说

Andrej，我换个style，就当咱们在咖啡店聊这个paper，不用academic腔调。

## 这paper到底在干嘛

现在train一个LLM，你得走四步：先pre-train一大堆web文本，然后CPT灌点domain data，再SFT教它答题格式，最后RL让它更聪明。问题是——这四步每步到底贡献了多少？没人说得清。因为大家都是拿别人的base model直接做post-training，变量完全不受控；要么就是拿intermediate checkpoint做分析，但那个checkpoint的LR还没decay完，根本不是final model的真实状态。

EvoLM的做法很暴力但很诚实：train了100多个1B/4B模型，from scratch，每一步都严格控制变量，全部开源。相当于在实验室里把LLM training pipeline做成了一个controlled experiment。

## 几个反直觉的发现

### 1. Pre-train太多反而有害

常识是"pre-train越多越好"。但实际上，1B模型pre-train到80B tokens就基本saturate了：

| Tokens | Upstream Acc |
|--------|-------------|
| 20B | 46.44% |
| 80B | 51.88% |
| 160B | 52.30% |
| 320B | 52.49% |

从80B到320B，upstream涨了0.6个百分点，几乎可以忽略。更扎心的是downstream：OOD的Maj@16和Pass@16在160B之后开始**下降**。也就是说，你花4倍compute去pre-train，下游OOD性能反而变差了。

直觉上理解：model在general web text上drill太久，representation变得很rigid，后面再去做domain-specific fine-tune时，model"不愿意"离开之前那个comfort zone。就像一个人刷了太多高考题，突然让他做奥数，反而转不过弯。

### 2. 同样算力下，小模型可能赢大模型

这个很反scaling law。同样FLOPs预算，1B×320B tokens 比 4B×80B tokens 效果更好：

| Config | ID Maj@16 (SFT/RL) |
|--------|-------------------|
| 1B-320BT | 16.1 / 25.0 |
| 4B-80BT (same compute) | 13.2 / 20.0 |

Chinchilla告诉你compute-optimal是20 tokens/param，4B模型应该配80B tokens——这恰好是compute-optimal点。但下游reasoning任务上，"过度训练的小模型"反而赢。

只有当4B也pre-train到160B tokens（进入saturation regime），它才突然"解锁"规模优势，ID Maj@16跳到26.4%。

直觉：model size的红利需要pre-training充分saturation后才能释放。在under-trained regime，大模型的capacity是浪费的——那些额外的参数没有被fully utilize，post-training时也借不上力。

### 3. 不做CPT直接SFT+RL，RL可能帮倒忙

这个发现很practical。看Figure 5里的no-CPT model（就是`1B-160BT-100Kep1`，直接从pre-train跳到SFT）：

- Pure SFT的Maj@16还行
- 加了RL之后，Maj@16/RM@16/Pass@16反而**变差**

只有CPT做到10-20B tokens之后，RL才开始有正收益。

直觉：RL是policy optimization，它需要一个decent的initial policy来explore。如果model只经过pre-train + SFT，它的math reasoning是"表面"的——只在QA format上memorize了一些pattern，底层representation并不solid。RL的policy gradient一推，model就drift到乱七八糟的地方。CPT先把deep mathematical knowledge建起来，SFT再wrap成conversational format，RL才能在这个solid foundation上refine。

这就像是：你要教一个人做菜（RL），他得先有食材知识（CPT）和基本刀工（SFT）。直接让一个只会背菜谱名的人去创新，他只会搞出黑暗料理。

### 4. CPT的replay有个sweet spot

CPT时加FineWeb replay能防forgetting，但比例很关键：

| Replay Config | GSM8K Pass@1 |
|---------------|-------------|
| 0% replay (50B FineMath) | 19.27% |
| 3.2% replay (1.6B + 48.4B) | 16.21% |
| 16% replay (8B + 42B) | **21.01%** |
| 32% replay (16B + 34B) | 15.22% |

太少不管用，太多稀释domain signal。16%左右是sweet spot。

直觉：replay是给model一个"锚"，提醒它别忘了general knowledge。但锚太重，model就漂不到domain那边去。5-16%这个比例刚好够维持general knowledge的weight不collapse，又不会拖累domain adaptation。这跟人学新东西一样——完全不复习旧知识会忘，但天天复习旧知识就没时间学新的。

### 5. SFT的epochs：3这个magic number有道理

Varying SFT epochs（100K examples fixed）：

- ID accuracy：持续涨，8 epochs后saturate
- OOD accuracy：2-4 epochs达峰，之后下降

这解释了为什么社区普遍用3 epochs——正好在OOD peak附近。

直觉：少epochs时，model学的是"reasoning pattern"，能transfer。多epochs后，model开始memorize具体QA pair，这种memorization对in-domain有效（test分布类似），但破坏OOD。就像学生：做几套题学会解题方法能举一反三，做太多套就开始背题，遇到新题型就懵了。

### 6. RL其实不教新东西，只是让model更自信

这个是paper最provocative的发现。看RL scaling：

- Greedy/Maj@16/RM@16：涨
- Pass@16（16个sample里至少1个对）：早期saturate，然后**下降**
- Correct Ratio（对了的里面correct response占比）：持续涨

Pass@16下降说明model能解决的问题集合没有扩大，甚至在缩小。但Maj@16上升说明model对"能解决"的那些问题，output更concentrated、更confident了。

PPO的objective是这个：

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

变量解释：
- $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$：新policy相对旧policy的概率比
- $\hat{A}_t$：advantage，这里reward是binary的，对了就是正advantage，错了就是负
- $\epsilon$：clip参数，限制update幅度

binary reward下，PPO就是：把correct response的log-prob往上推，错误的往下压。本质是**mode-finding**——强化model已有的"侥幸答对"的trajectory，让它在概率分布上更prominent。

直觉：RL不教model新的reasoning capability，它只是把model"偶尔能答对"变成"大概率答对"。Pass@16下降是因为probability mass集中到少数trajectory上，sample diversity降低了。这跟"刷题能提高准确率但不提高智商"是一个道理。

### 7. Intermediate checkpoint会骗你

这个对做scaling law研究的人特别重要。从160B run里切出来的20B/40B checkpoint，upstream accuracy和单独run 20B/40B的model几乎一样（差距<0.5%），但downstream Pass@16差一大截：

| Model | Upstream | Math L1 Pass@16 |
|-------|----------|-----------------|
| 20BT full run | 46.43 | 17.85 |
| 20BT checkpoint (from 160BT run) | 46.07 | 11.44 |

直觉：cosine LR schedule的最后20% tokens是"settling phase"——LR降到很低，model在loss landscape里做fine-grained adjustment，找到sharper minima。Intermediate checkpoint在LR还很高时取出，model处于"过渡态"，representation没consolidate。upstream accuracy测的是next-token prediction，对这种状态不敏感；downstream generative task需要完整的reasoning chain，对representation quality敏感得多。

所以用intermediate checkpoint做scaling law分析，pre-training loss可能对，但downstream prediction会严重mislead。Pythia那种checkpoint suite用来做downstream分析是有系统性bias的。

### 8. PPL没用，ORM score才有用

Post-training后validation PPL和accuracy的correlation接近0（Figure 14）。但ORM score和Maj@16的Pearson correlation有0.62-0.84（Figure 10）。

直觉：post-training后model不再是pure next-token predictor了。SFT/RL让output distribution大幅shift，PPL衡量的是 $P_{model}(x_{1:T})$ 跟某个reference的KL，但这个reference对post-trained model本来就不calibrated。你拿一个"背菜谱的loss"去衡量"做菜能力"，当然不相关。

ORM直接评估output quality，是task-aligned的。用8B的reward model去score 1B model的output，correlation能到0.8+，说明ORM捕获了reasoning quality的signal，而不是surface format。

Practical takeaway：post-training阶段别看PPL，用ORM score做proxy metric，尤其data-constrained场景下没法collect足够test set时。

## 整成一句话的recipe

如果你要train一个small reasoning model：

1. **Pre-train**到80-160x model size就够，别over-train
2. **CPT**必须做，30B+ domain tokens，加5-16% replay
3. **SFT**用3 epochs，100K high-quality examples够
4. **RL**做4-8 epochs on 100K disjoint examples
5. **Validation**用ORM score，别用PPL
6. **Data allocation**：重ID给SFT多点，重OOD给RL多点

## 我觉得paper没做但很重要的

1. **Multi-stage CPT**：真实场景是code → math → biology sequential CPT，interaction effect没人研究
2. **DPO/GRPO对比PPO**：paper只用PPO，不同RL algorithm的scaling dynamic可能完全不同
3. **Larger scale**：只到4B，70B/100B是否hold unknown
4. **Test-time compute tradeoff**：与其多train，不如多inference（R1路线），这个tradeoff在哪
5. **Why 5% replay**：有没有theoretical explanation，还是纯empirical

Paper的core message我觉得是：**scale matters, but how you allocate scale across stages matters more**。这对resource-constrained的lab是好消息——你不需要train 100B model也能做有意义的reasoning research，关键是每一步的配比要对。

---

# EvoLM: In Search of Lost Language Model Training Dynamics - 深度解析

Andrej, 这篇paper是Harvard/Stanford/EPFL/CMU团队合作的工作，核心目标是systematically unpack modern LM training的四个stage之间的complex interactions。下面我尽量从intuition出发，把技术细节讲透。

## 1. 核心动机: 为什么需要EvoLM?

当前LLM开发已经分裂成 Pre-training → CPT → SFT → RL 这条pipeline, 但downstream developer几乎无法追溯每个stage的design choice对最终性能的contribution。之前的scaling law研究(如Chinchilla)主要focus在pre-training loss与compute的关系:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

其中 $N$ 是model parameter数量, $D$ 是training token数量, $E$ 是irreducible loss(entropy of natural language), $A, B$ 是常数, $\alpha \approx 0.34$, $\beta \approx 0.28$ (Chinchilla拟合值)。这个law只描述pre-training loss, 无法预测post-training后的downstream problem-solving能力。

EvoLM的solution: 训练100+个1B/4B模型from scratch, 完整LR decay, 控制所有variables, 开源全部data/code/models。

参考链接:
- Chinchilla paper: https://arxiv.org/abs/2203.15556
- EvoLM项目: https://github.com/evo-lm/EvoLM (推测, paper提到release)

## 2. 训练Pipeline详解

### 2.1 Architecture (LLaMA-2 style)

| Model Size | Hidden Size | Intermediate Size | Vocab | Context | Heads | Layers | KV Groups |
|------------|-------------|-------------------|-------|---------|-------|--------|-----------|
| 0.5B | 1536 | 3216 | 32000 | 2048 | 32 | 20 | 4 |
| 1B | 2048 | 4896 | 32000 | 2048 | 32 | 22 | 4 |
| 4B | 4096 | 7792 | 32000 | 2048 | 32 | 28 | 4 |

注意 #Query Groups = 4 表示用了GQA (Grouped Query Attention), 32个query heads共享4组KV heads, 这样能减少KV cache memory, 对inference friendly。1B模型只有22层但hidden 2048, 是一个相对"wide and shallow"的设计, 与LLaMA-2-1B的28层不太一样。

### 2.2 四阶段配置

**Pre-training**: FineWeb-Edu (~1.3T tokens pool), token budget从10B到320B。Chinchilla optimal ratio是20 tokens/param, 所以1B模型20B tokens是compute-optimal, 320B tokens是16x over-training。

**CPT (Continued Pre-training)**: FineMath (~50B tokens pool), 2B到42B domain-specific tokens, 加上FineWeb-Edu replay(0-16B)来对抗catastrophic forgetting。

**SFT**: 100K-400K examples from MetaMathQA + OpenMathInstruct2 + NuminaMath mixture, 用model correctness consistency过滤低质量样本。

**RL**: PPO with binary verifiable reward (math problem有ground truth answer可以verify)。与SFT data disjoint。

模型签名例如 `1B-160BT-8+42BT-100Kep1-100Kep16`:
- 1B params
- 160B pre-training tokens
- CPT: 8B FineWeb replay + 42B FineMath
- SFT: 100K examples × 1 epoch
- RL: 100K examples × 16 epochs

## 3. 关键发现逐一拆解

### 3.1 Pre-training Scaling的Saturation (Takeaway 1-2)

实验结果(Table 4 + Figure 2/3):
- 1B模型: 20BT→46.44%, 80BT→51.88%, 160BT→52.30%, 320BT→52.49% (upstream avg)
- 4B模型: 80BT→50.22%, 160BT→55.30%, 320BT→56.94%

Diminishing returns在80x-160x model size处出现。更surprising的是downstream: 1B SFT model的ID Maj@16从20BT的8%涨到80BT的15%, 但到320BT只升到17%。OOD的Maj@16/RM@16/Pass@16在160BT之后开始下降!

**Intuition**: 过度pre-training让model的representation过度specialize到general web text的distribution, 对后续domain-specific的fine-tuning反而产生rigidity。这与Springer et al. 2025的"overtrained models are harder to fine-tune"一致(https://arxiv.org/abs/2503.19206), 他们发现over-training增加parameter update的sensitivity。

更有趣的对比(Table 1):

| Base | ID Greedy (SFT/RL) | ID Maj@16 (SFT/RL) | OOD Pass@16 (SFT/RL) |
|------|--------------------|--------------------|----------------------|
| 1B-320BT | 14.1/20.1 | 16.1/25.0 | 54.4/62.6 |
| 4B-80BT (same compute) | 11.3/15.7 | 13.2/20.0 | 52.2/60.2 |
| 4B-160BT | 22.0/27.8 | 26.4/34.8 | 57.3/66.2 |

相同FLOPs下(1B×320B tokens ≈ 4B×80B tokens in compute), 1B-320BT反而比4B-80BT更好! 这违反naive scaling law直觉。但当budget足够让4B也达到saturation(160BT), 4B突然"解锁"规模优势, ID Maj@16从14.2%跳到26.4%。

**Intuition**: model size的gain需要pre-training达到saturation regime才能manifest。在under-trained regime, 更大的model只是浪费capacity, 因为representation还没develop出rich structure供post-training利用。

### 3.2 CPT的Catastrophic Forgetting与Replay (Takeaway 3-6)

Figure 4显示CPT tokens增加时, upstream accuracy持续下降(no replay baseline)。这是经典catastrophic forgetting: model在FineMath上specialize, FineWeb-Edu的知识被overwrite。

Replay策略: 在CPT batch中随机interleave少量FineWeb-Edu数据。Table 2的关键数据:

| CPT Config | GSM8K-Platinum Pass@1 |
|------------|----------------------|
| No CPT | 6.04 |
| FineMath 50BT (no replay) | 19.27 |
| FineWeb 1.6BT + FineMath 48.4BT (3.2% replay) | 16.21 |
| FineWeb 8BT + FineMath 42BT (16% replay) | 21.01 |
| FineWeb 16BT + FineMath 34BT (32% replay) | 15.22 |

最优replay比例约5-16%, 过少不够mitigate forgetting, 过多稀释domain-specific signal。

**Intuition**: 这是一个regularization vs. specialization的trade-off。Replay相当于在loss landscape中拉一个anchor, 防止model drift太远, 但anchor太强就限制了domain adaptation。5%这个magic number可能与 FineMath/FineWeb的relative entropy差异有关 - 大约5%的"reminder"足以维持general knowledge的synaptic weight, 剩余95%的capacity用于domain-specific learning。

Figure 5还揭示一个critical finding: **没有CPT, RL可能degrade SFT性能**! 看 `1B-160BT-100Kep1` (no CPT) 的Maj@16/RM@16/Pass@16, SFT+RL version比pure SFT还差。只有CPT tokens达到一定量(约10-20B), RL才开始产生positive gain。

**Intuition**: RL需要policy有一个合理的initial distribution来explore。No CPT时, SFT学到的math solving pattern是"shallow"的(只在QA format上memorize), RL的policy gradient会push model进入low-reward region (因为base policy的reasoning chain是fragile的)。CPT先建立了deep mathematical knowledge representation, SFT再shape成conversational format, RL才能在这个solid foundation上refine。

### 3.3 SFT的Over-specialization (Takeaway 7-8)

Figure 6 - varying SFT epochs (100K examples fixed):

- ID metrics: 持续上升, 8 epochs后saturate
- OOD metrics: 2-4 epochs达到peak, 然后下降
- RL marginal gain: 随SFT epochs增加而缩小

Figure 7 - varying SFT dataset size (1 epoch fixed):

- ID: monotonically improves
- OOD: fluctuates, can decline

Power law relationship (之前Raghavendra et al. 2024在≤10K examples上验证):

$$\text{Acc}(D_{sft}) \propto D_{sft}^{-\alpha_{sft}}$$

但EvoLM在50K-400K scale上发现这个power law主要hold for ID, OOD会break down。

**Intuition**: SFT本质是behavior cloning。Few epochs时, model学到的format和reasoning pattern是"generalizable"的(因为没memorize具体样本)。多epochs后, model开始memorize specific QA pairs, 这种memorization对in-domain有效(因为test distribution类似), 但破坏OOD generalization (因为memorized pattern不transfer)。3 epochs这个common practice正好在OOD peak附近, 是经验性optimal。

RL的marginal gain缩小也合理: SFT over-train后, model已经"locked in"到specific solution pattern, RL的policy gradient没空间explore alternative reasoning paths。

### 3.4 RL的Confidence Sharpening (Takeaway 9-11)

Figure 8a/b - varying RL epochs/examples:

- Greedy/Maj@16/RM@16: 在4-8 epochs或150-200K examples处saturate
- Pass@16: 早期就saturate, 然后decline
- Correct Ratio: 持续上升

这是关键insight: RL主要在sharpen已经correct的output的probability, 而不是扩展可解决问题的集合。

PPO objective (简化版):

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

其中:
- $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t)$ 是importance sampling ratio
- $\hat{A}_t$ 是advantage estimate (这里用binary reward - 0或1)
- $\epsilon$ 是clip parameter (通常0.2)

当reward是binary的, advantage对correct response是正的, 对错误的是负的。Policy gradient会push correct response的log-prob上升, 错误的下降。这本质上是一种**mode-finding**操作 - 强化model已有的"侥幸correct"的output, 而不是teach新的reasoning capability。

Pass@16 (16个sample中至少1个correct)的decline尤其enlightening: RL让model更"deterministic" - 把probability mass集中在少数high-reward trajectory上, 牺牲了diversity。所以Pass@16下降(因为sample diversity下降), 但Maj@16上升(因为sampled output更集中在correct answer周围)。

350K-400K examples处的性能collapse很有意思 - 因为response length爆炸, 超过context window。这是RL known failure mode: model发现"longer response = higher chance of hitting reward", 学会了padding with redundant tokens。这种reward hacking在epoch-based scaling中没观察到, 说明是data coverage导致的某种distribution shift。

Figure 9的SFT/RL data allocation实验: 100K total budget, 5个split (10/90, 30/70, 50/50, 70/30, 90/10):
- ID accuracy随SFT比例单调上升, 70K后plateau
- OOD accuracy随RL比例单调上升, peak在10K SFT/90K RL

**Intuition**: SFT提供"format prior" - 让model学会怎么output valid math solution。这个prior很快就能建立(10K examples足够)。RL的marginal value在于explore reasoning strategy的diversity, 这种exploration对OOD transfer更有价值。ID task的distribution已经被CPT+SFT覆盖, RL的additional exploration价值有限。

### 3.5 Intermediate Checkpoints不是Reliable Surrogate (Section 4.1)

Table 3:

| Model | Upstream | Math L1 (Greedy/Pass@16) | Math L2 |
|-------|----------|--------------------------|---------|
| 20BT full | 46.43 | 2.75/17.85 | 3.36/15.10 |
| 20BT int. (from 160BT run) | 46.07 | 2.52/11.44 | 1.90/12.64 |
| 40BT full | 49.38 | 2.97/17.96 | 3.36/14.88 |
| 40BT int. | 49.06 | 1.37/9.38 | 2.68/8.72 |

Upstream accuracy几乎一样(<0.5% gap), 但downstream Pass@16差距巨大(17.85 vs 11.44)。

**Intuition**: 这与LR schedule有关。Cosine decay的tail阶段(最后10-20% tokens)对final model quality至关重要 - 它让model"settle"到sharper minima。Intermediate checkpoint在LR还没decay时取出, model处于"transitioning"状态, representation还没有fully consolidated。所以Pythia(https://arxiv.org/abs/2304.01373)那种用intermediate checkpoint做scaling law分析的方法, 对downstream prediction可能严重mislead。

### 3.6 ORM Score作为Unsupervised Validation Metric (Section 4.2)

Figure 10显示ORM score (avg@16)与Maj@16 accuracy的Pearson correlation:
- 大多数task: 0.62-0.84
- StrategyQA: 较低(可能因为commonsense task, ORM不specialize)

而validation PPL与accuracy的correlation接近0 (Figure 14)!

**Intuition**: Post-training后, model的output distribution shift很大 - SFT/RL让model变成"conversational assistant", 不再是pure next-token predictor。PPL衡量的是 $P_{model}(x_{1:T})$ 与reference distribution的KL, 但这个reference distribution本身对post-trained model就miscalibrated。

ORM score则直接评估output quality, 是task-aligned的metric。用Skywork-Reward-Llama-3.1-8B-v0.2 (https://arxiv.org/abs/2410.18451)这种large ORM能给出smooth, calibrated的quality score, 比binary accuracy更informative。

实操意义: post-training阶段没法直接用test set (因为overfit风险), validation PPL又uninformative, ORM score提供了一个**task-relevant的proxy metric**。

## 4. 与Concurrent Work的Positioning

- **SFT memorizes, RL generalizes** (Chu et al. 2025, https://arxiv.org/abs/2501.17161): EvoLM部分agree, 但发现RL的generalization主要在OOD, ID上SFT的memorization其实更有用。
- **RL amplifies pretrained patterns** (Zhao et al. 2025, https://arxiv.org/abs/2504.07912): EvoLM的Takeaway 10直接support这个 - RL是"echo chamber", 强化已有pattern。
- **Does RL really incentivize reasoning?** (Yue et al. 2025, https://arxiv.org/abs/2504.13837): EvoLM的Pass@16 decline证据strongly support "no" - RL不fundamentally improve reasoning, 只sharpen confidence。
- **Overtrained models harder to fine-tune** (Springer et al. 2025, https://arxiv.org/abs/2503.19206): EvoLM的Takeaway 1确认这一点, 并extend到downstream generative task。
- **D-CPT law** (Que et al. 2024, https://arxiv.org/abs/2410.17835): EvoLM的replay实验提供empirical evidence for这个theoretical framework。

## 5. Limitations与Future Direction

作者自己承认:
1. 只到4B参数, 大model是否hold unknown
2. 只focus reasoning, 没cover safety/instruction-following/tool-use
3. 只用PPO with verifiable reward, GRPO/DPO等未探索

我会补充几个方向:
- **Multi-stage CPT**: 真实production会sequential CPT多个domain (code → math → biology), interaction effect复杂
- **Synthetic data scaling**: FineMath本身可能是synthetic augmented, RL on synthetic的scaling dynamic可能不同
- **Model merging**: post-training后model merge的effect在这个framework下怎么分析
- **Test-time compute**: RL scaling vs. test-time compute scaling的trade-off (类似DeepSeek R1的思路)

## 6. Practical Recipe (我的总结)

基于EvoLM的evidence, 对想train small reasoning model的practitioner:

1. **Pre-training**: 不要盲目over-train。1B模型80-160B tokens, 4B模型160-320B tokens就够。超过这个范围, downstream收益微乎其微, 甚至hurt OOD。

2. **CPT**: Domain adaptation必须做CPT, 直接SFT/RL效果差。CPT数据量至少30B+ tokens, 配合5-10% general data replay。Replay比例不是越高越好, 16%可能已经too much。

3. **SFT**: 3 epochs是经验最优(对OOD)。ID task可以多epochs但收益diminishing。Data quality > quantity, 100K high-quality examples基本够。

4. **RL**: 在CPT+SFT基础上, RL能提供consistent uplift。但不需要太多 - 4-8 epochs on 100K examples足够。RL data与SFT disjoint很关键, 避免overfitting。Monitoring response length防reward hacking。

5. **Validation**: 别信PPL, 用ORM score做post-training的proxy metric。

6. **Data allocation**: ID task多给SFT, OOD task多给RL。100K total budget下, 70K SFT/30K RL是balanced choice。

## 7. Open Questions for Future Research

- Why does 1B-320BT outperform 4B-80BT under same compute? 是否说明over-training在small model上更efficient? 这与Chinchilla的compute-optimal建议矛盾, 可能是因为downstream task的representation需求不同。

- 5% replay的theoretical justification? 是否与Eigenvalue spectrum of gradient updates有关? 或与task-relatedness的mutual information bound有关?

- Pass@16 decline的具体mechanism? 是entropy collapse, 还是mode dropping? 用entropy regularizer (如GRPO的KL term)能否mitigate?

- ORM score correlation 0.62-0.84, 在什么task上会break? StrategyQA的低correlation是anomaly还是pattern?

EvoLM是rare的systematic empirical study, 在当前"scale is all you need"的narrative下提供了重要counter-evidence: scale matters, but how you scale across stages matters more。这对resource-constrained lab尤其valuable - 不需要train 100B+ parameter model也能做meaningful reasoning research。

相关paper链接汇总:
- EvoLM: https://arxiv.org/abs/2506.19128 (推测, based on author list)
- Chinchilla: https://arxiv.org/abs/2203.15556
- FineWeb-Edu: https://arxiv.org/abs/2412.15109
- SmolLM2 (FineMath source): https://arxiv.org/abs/2502.02737
- MetaMathQA: https://arxiv.org/abs/2309.12284
- OpenMathInstruct2: https://arxiv.org/abs/2410.01560
- NuminaMath: https://huggingface.co/AI-MO/NuminaMath-CoT
- GSM8K: https://arxiv.org/abs/2110.14168
- MATH: https://arxiv.org/abs/2103.03874
- Overtrained harder to fine-tune: https://arxiv.org/abs/2503.19206
- SFT memorizes RL generalizes: https://arxiv.org/abs/2501.17161
- RL echo chamber: https://arxiv.org/abs/2504.07912
- Does RL incentivize reasoning: https://arxiv.org/abs/2504.13837
- D-CPT law: https://arxiv.org/abs/2410.17835
- Skywork-Reward: https://arxiv.org/abs/2410.18451
- PPO: https://arxiv.org/abs/1707.06347
- CRUXEval: https://arxiv.org/abs/2401.03065
- BoardgameQA: https://arxiv.org/abs/2310.14552
- GSM8K-Platinum: https://arxiv.org/abs/2502.03461
