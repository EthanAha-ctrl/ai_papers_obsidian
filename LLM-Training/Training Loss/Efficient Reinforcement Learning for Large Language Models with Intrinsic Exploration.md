---
source_pdf: Efficient Reinforcement Learning for Large Language Models with Intrinsic
  Exploration.pdf
paper_sha256: 79cb15fb4b8d73e17bcc99b532ae5f70aab9d14712163da3fabac5f1f68584dc
processed_at: '2026-08-04T01:57:08-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 PREPO

Andrej，我换一种讲法，把它当成在白板前跟你聊。

---

## 这篇paper想解决什么问题？

想象你在训练一个AI做数学题。标准的RLVR流程是这样的：

1. 从题库里抽一批题
2. 对每道题，让AI生成8个答案
3. 用verifiable reward打分（对=1，错=0）
4. 算advantage，做PPO update

问题在哪？**rollout generation贵得要命**。生成8个答案的forward pass开销远超backward pass。而且你仔细看，其实大部分rollout是浪费的：

- 太简单的题，AI每次都对，8个答案reward全是1，advantage全是0，gradient为零
- 太难的题，AI每次都错，reward全是0，advantage还是0，gradient还是零
- 答案很confident的rollout（entropy低），policy分布很sharp，gradient小
- 只有那些"AI有点犹豫但能做对/做错"的rollout，才真正提供学习信号

**核心痛点**：你花了80%的compute在生成那些对gradient没贡献的rollout上。

---

## 两个关键观察

### 观察1：Prompt PPL能反映难度

PPL（perplexity）就是模型对这道题的"困惑度"。

$$P_i = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log\pi_\theta(x_{i,t}|x_{i,<t})\right)$$

- $x_{i,t}$：prompt的第$t$个token
- $\pi_\theta(x_{i,t}|x_{i,<t})$：模型给这个token的概率
- $T$：prompt总长度
- 整个公式就是"模型对这道题的likelihood的几何平均取倒数"

PPL低 = 模型对这道题的token很熟悉 = 概率高 = 容易题
PPL高 = 模型觉得陌生 = 难题或OOD

作者在DAPO-Math-17K上验证了这点，Spearman correlation大概-0.17到-0.23，**统计显著**。PPL越低的题，pass rate@16越高（16次里至少做对一次的比例）。

这其实跟language modeling里PPL作为OOD detector的经典用法一脉相承。但这里更妙的是：PPL不仅告诉你这道题"离训练分布多远"，还告诉你"模型当前觉得它多难"。

参考：OOD detection with PPL https://arxiv.org/abs/2106.05858

### 观察2：Low-PPL和High-PPL的训练动力学完全不同

作者做了个实验：把题库按PPL分成LOW-PPL（最低20%）和HIGH-PPL（最高20%）两组，分别训练，看动态：

| 指标 | Low-PPL组 | High-PPL组 |
|------|-----------|------------|
| Entropy | 低（答案很confident） | 高（答案很uncertain） |
| 早期reward上升 | 快 | 慢 |
| 后期reward | plateau | 还在涨 |
| Entropy collapse | 快（exploration死掉） | 慢（exploration活着） |
| 最终AIME24 | 被反超 | 更好 |

**直觉**：
- Low-PPL的题，模型已经"半会"了，RL是在sharpen已有能力，所以早期reward涨得快。但很快entropy collapse——模型对这些题太confident了，8个答案几乎一样，没有exploration，gradient变小，训练stall。
- High-PPL的题，模型很uncertain，8个答案五花八门，entropy高，exploration活得好好的，但早期reward低（因为模型还不会），后期才慢慢追上来。

这就像健身：你不能一直只做轻重量（low-PPL），进步快但很快plateau；也不能一直做极限重量（high-PPL），受伤风险高还学不到东西。**你需要一个从轻到重的渐进schedule**。

---

## PREPO的两个组件

### 组件1：PPL-Schedule（给prompt排课表）

这个idea特别简单。每个训练step：

1. 计算candidate batch $B$里每道题的**当前PPL**（注意是当前模型算的，不是固定的）
2. 按PPL从低到高排序
3. 选一个长度为$K$的滑动窗口

$$l(\rho) = \lfloor \rho \cdot (|B| - K) \rfloor$$

- $\rho \in [0,1]$：训练进度（当前step / 总steps）
- $|B|$：candidate batch大小
- $K$：要选的子batch大小（实验里是20%）
- $l(\rho)$：窗口起始位置

$\rho=0$时窗口在最左端（最easy的题），$\rho=1$时窗口滑到最右端（最hard的题）。**线性curriculum**，从易到难。

**为什么用dynamic PPL而不是static PPL？** 因为训练中模型在变化。一道题刚开始是high-PPL（模型不会），训练一阵后变成low-PPL（模型会了）。用当前模型算PPL，自动适应这个变化。Figure 8显示整个训练过程中prompt PPL的range和mean都很稳定，说明这个adaptive机制work。

**开销**：算PPL就是一次forward pass，不用backward。相比生成rollout的forward pass（要生成几千个token），算prompt PPL（prompt就几百token）的开销几乎可以忽略。Figure 9证实了这点。

### 组件2：Relative-Entropy Weighting（给rollout加权）

PPL-Schedule解决了"选哪些题"，但还有个问题：**早期选的low-PPL题，生成的rollout往往entropy很低**（模型很confident，8个答案几乎一样）。这导致gradient小，exploration不足。

作者的做法：对每个rollout算sequence-level entropy，然后**相对于batch mean做归一化**，作为这个rollout的权重。

**Token-level entropy**：

$$H_t = -\sum_{v \in \mathcal{V}}\pi_\theta(v|o_{<t}, x)\log\pi_\theta(v|o_{<t}, x)$$

- $v$：词表里的每个token
- $\pi_\theta(v|o_{<t}, x)$：给定前文，下一个token是$v$的概率
- $H_t$高：多个token都有 plausible 概率，模型uncertain
- $H_t$低：概率集中在一个token上，模型confident

**Sequence-level entropy**（对一条rollout的所有token取平均）：

$$\bar{H}_i = \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}H_t$$

- $|o_i|$：rollout $i$ 的长度
- $\bar{H}_i$：这条rollout整体的uncertainty

**Batch-average entropy**：

$$\bar{H} = \frac{1}{B}\sum_{k=1}^{B}\bar{H}_k$$

**Relative weight**：

$$w_i = \frac{\bar{H}_i}{\bar{H}}$$

- $w_i > 1$：这条rollout比batch平均更uncertain → 权重大，被放大
- $w_i < 1$：这条rollout比batch平均更confident → 权重小，被缩小

**这个归一化是关键**。如果直接用绝对entropy做权重，一个极端high-entropy的rollout会主导整个batch的gradient。用relative的，outlier不会无限放大自己（因为分母也被拉高了），而是**让其他rollout被相对缩小**。这是一种robust normalization。

Appendix E给了偏导数分析：

$$\frac{\partial w_i}{\partial \bar{H}_j} = \begin{cases} \frac{1}{\bar{H}} - \frac{\bar{H}_j}{\bar{H}^2}\frac{|o_j|}{\sum_k|o_k|}, & i = j \\ -\frac{\bar{H}_i}{\bar{H}^2}\frac{|o_j|}{\sum_k|o_k|}, & i \neq j \end{cases}$$

- $i=j$时（自己对自己）：两项近似抵消，极大entropy的rollout对自己的weight影响反而小
- $i \neq j$时（对别人）：严格负，极大entropy的rollout会拉低所有其他rollout的weight

Figure 21验证：weight分布集中在0.7-1.5，长尾超过2的很少，超过4的罕见。没有outlier主导gradient。

**直觉**：这就像在一个班里给学生打分，你不是看绝对分数，而是看相对排名。一个特别聪明的学生不会把平均分拉太高让自己看起来没那么突出（self-sensitivity小），但会让其他学生看起来分数更低（cross-sensitivity负）。

### 两个组件怎么配合？

**PPL-Schedule**是"seek certainty"——选模型相对熟悉的题
**Relative-entropy weighting**是"seek uncertainty within certainty"——在这些题里，选模型相对uncertain的答案

这听起来矛盾，其实很聪明：
- 选familiar的题（low-PPL）→ reward信号强，模型能学到东西
- 但在familiar的题里选uncertain的rollout（high relative entropy）→ 保留exploration，防止entropy collapse

这是multi-scale的exploration策略：**macro层面走curriculum（从易到难），micro层面做importance weighting（放大uncertain的）**。

---

## 完整objective

$$\mathcal{J}_{\text{PREPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{T}_\rho, \{o_i\}_{i=1}^G \sim \pi_{\text{old}}}\left[\frac{1}{G}\sum_{i=1}^G w_i \cdot \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\Big(s_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(s_{i,t}(\theta), 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\hat{A}_{i,t}\Big)\right]$$

拆开看：
- $\mathcal{T}_\rho$：PPL-Schedule选出的题
- $G$：每题生成几个rollout（实验里是8）
- $w_i = \bar{H}_i/\bar{H}$：relative-entropy weight
- $s_{i,t}(\theta) = \pi_\theta(o_{i,t}|x,o_{i,<t})/\pi_{\text{old}}(o_{i,t}|x,o_{i,<t})$：PPO的importance ratio
- $\hat{A}_{i,t} = (r_i - \text{mean}(r_j))/\text{std}(r_j)$：GRPO风格的group advantage
- $\epsilon_{\text{low}}=0.2, \epsilon_{\text{high}}=0.28$：asymmetric clipping（DAPO的trick）

跟标准GRPO/DAPO的区别就两点：
1. Prompt从随机抽样变成PPL-Schedule选
2. 每个rollout乘了个$w_i$

就这么简单。没有额外训练predictor，没有replay buffer，没有复杂的数据结构。PPL和entropy都是forward pass的副产品，几乎free。

---

## 实验结果讲人话

**主要结论**：在Qwen2.5-Math-7B、Qwen2.5-Math-1.5B、Qwen3-4B、Qwen2.5-7B、Llama3.1-8B上，PREPO跟random selection比，**rollout用量减少2-3倍，性能持平甚至更好**。

最夸张的是Qwen2.5-Math-1.5B：random要用3.0M rollouts，PREPO只要1.1M，性能还略高（36.23 vs 35.86）。

Llama3.1-8B上更明显：PREPO用115K rollouts，random用266K，性能36.55 vs 30.61。这说明**对base能力较弱的模型，data selection的收益更大**——因为weak model对无用rollout的"消化能力"更差，更需要把compute集中在informative的数据上。

**Ablation**：PPL-Schedule alone已经比random好很多，加relative-entropy再提升一点。两个组件是互补的：PPL-Schedule管"选哪些题"，relative-entropy管"同一道题里选哪些rollout重点学"。

**Training dynamics**：
- PREPO的entropy下降更慢（exploration保持得好）
- PREPO的zero-advantage ratio更低（更多rollout产生non-trivial gradient）
- PREPO的gradient norm保持稳定（没有不稳定）

**Diversity**：PREPO选的题覆盖更多MSC（Mathematics Subject Classification）类别。这有点反直觉——你本来以为PPL-Schedule会bias到某类题，但dynamic PPL让不同阶段选不同类型的题，自然扩大了coverage。

**Memorization check**：把prompt截断40%后测试，pass rate接近0，说明模型靠reasoning不是靠memorization。

---

## 跟其他工作的区别

**跟curriculum learning的区别**：
- 传统curriculum需要估计难度（pass rate、额外LLM打分），贵
- PREPO用PPL，几乎free
- 参考：Self-evolving curriculum https://arxiv.org/abs/2505.14970，Curriculum RL https://arxiv.org/abs/2506.06632

**跟data selection方法的区别**：
- LearnAlign需要训gradient alignment predictor
- DUMP需要distribution-level建模
- PREPO直接用intrinsic signals，不训额外模型
- 参考：LearnAlign https://arxiv.org/abs/2506.11480，DUMP https://arxiv.org/abs/2504.09710

**跟entropy相关工作的区别**：
- Cui et al.用covariance-based update缓解entropy collapse
- Wang et al.发现high-entropy "forking tokens"驱动reasoning improvement
- PREPO在sequence level做reweighting，跟token-level的forking tokens可以互补
- 参考：Entropy mechanism https://arxiv.org/abs/2505.22617，Forking tokens https://arxiv.org/abs/2506.01939

---

## 更深的intuition

### 1. PPL-Schedule本质上是"confidence-gated curriculum"

Low-PPL prompt对模型来说是"我知道一点"的区域。在这个区域做RL，模型在**refining已有的reasoning paths**，reward信号明确，gradient方向清晰。但一旦refine到极致，entropy collapse，模型只会那几条path，遇到新题就傻了。

High-PPL prompt是"我完全不会"的区域。在这个区域做RL，模型在**探索新reasoning paths**，但reward稀疏（大部分都错），gradient信号弱。需要模型先有一定基础才能在这个区域有效学习。

PPL-Schedule就是：**先用low-PPL建立基础，再慢慢引入high-PPL保持exploration**。这跟human learning的intuition一致——先练基本功再挑战难题。

### 2. Relative-entropy weighting本质上是"attention within rollouts"

Standard GRPO里，同一道题的8个rollout权重一样。但直觉上，8个答案的"信息量"不一样：
- 8个答案都一样（low entropy rollouts）→ 没什么信息，模型已经很confident了
- 8个答案五花八门（high entropy rollouts）→ 很多信息，模型在探索不同path

Relative-entropy weighting就是**让informative的rollout权重更大**。这跟attention mechanism的本质很像：不是所有信息都同等重要，需要adaptive weighting。

### 3. 为什么PPL能work而pass rate更直接但不work？

Pass rate@k确实更直接反映难度，但：
1. 要算pass rate@k，你需要生成k个rollout，**这本身就是你想省的compute**
2. Pass rate是binary的（对/错），丢失了"模型多uncertain"的信息
3. Pass rate对k敏感，k小了noise大，k大了又贵

PPL只需要一次forward pass on the prompt（几百token），就能给你一个连续的difficulty score。虽然相关性没有pass rate那么强（Spearman -0.2左右），但**性价比极高**。

### 4. 跟self-distillation的隐约联系

在low-PPL prompt上做RL，某种程度上像self-distillation：模型在"自己已经会"的题上强化自己。跟knowledge distillation里teacher=student的退化情况类似，但reward signal让它不退化成trivial copying——模型是在**reinforce正确的reasoning path**而不是copy自己的output。

这个视角挺有意思：**RLVR在low-PPL区域的早期训练，可能本质上是self-refinement**，等refinement到瓶颈了，PPL-Schedule自然把模型推向high-PPL区域，开始真正的exploration。

参考：Self-distillation https://arxiv.org/abs/2006.07733

### 5. 为什么不用entropy regularization？

经典RL（SAC等）会加$-\beta H(\pi)$到objective里鼓励exploration。但在LLM RLVR里这容易出问题：
- Token-level entropy regularization会让模型生成gibberish（高entropy但无意义）
- 需要tune $\beta$，而RLVR已经很sensitive to hyperparameters

PREPO的sequence-level relative weight更selective：**只放大batch内相对uncertain的rollout**，不直接push policy distribution变flat。reward signal对policy的影响还在，只是informative的rollout主导gradient。

### 6. 跟"forking tokens"的关系

Wang et al. 2025发现少数high-entropy的"forking tokens"驱动了大部分reasoning improvement。PREPO在sequence level做weighting，一个rollout如果有更多forking tokens，sequence-level entropy就更高，weight更大。所以PREPO和forking tokens的发现是**一致的**，只是粒度不同。

理论上，更细粒度的做法是在token level做weighting（给forking tokens更大的gradient weight），但这跟PPO的clip机制交互复杂。PREPO选择在sequence level做，更简单更稳定。**未来工作可以探索token-level的relative-entropy weighting**。

参考：Forking tokens https://arxiv.org/abs/2506.01939

---

## 我的caveats

1. **只测了math**：PPL作为difficulty signal在math上work，但在code、writing、dialogue上未验证。Code的PPL可能主要反映syntax familiarity而非logical difficulty。

2. **只测了20% selection ratio**：如果ratio太小，PPL-Schedule的窗口可能miss中间难度的题；太大则接近random。最优ratio可能跟model size和dataset相关，需要系统ablation。

3. **Long response没测**：实验限制在32K tokens。但DeepSeek-R1经常生成100K+ tokens。在超长response上，sequence-level entropy的average可能被多数boring token稀释，少数high-entropy tokens的影响被wash out。可能需要weighted average（给后半部分token更大权重？）。

4. **Dynamic PPL的overhead**：每个step要对整个candidate batch $B$算PPL。虽然单次forward比rollout generation便宜，但$|B|$大的时候累积开销也不小。论文说"negligible"但没给绝对数字。可以用partial PPL（只算前N tokens）进一步省compute。

5. **没跟DeepSeek-R1/Kimi K1.5比**：这些更recent的方法用更大rollout budget和更复杂reward shaping，PREPO在那些setting下能否保持advantage未知。

6. **Asymmetric clipping跟relative-entropy的交互没分析**：$\epsilon_{\text{high}} > \epsilon_{\text{low}}$鼓励positive advantage的rollout有更大update，而high-entropy rollout往往advantage接近0（uncertain → 可对可错），这个clipping可能跟weighting有implicit synergy或者conflict，值得理论分析。

---

## 一句话总结

**RLVR里大部分rollout是浪费的。PREPO用模型自己的PPL做prompt curriculum（从易到难），用entropy给rollout加权（放大uncertain的），两个free signal把rollout用量砍到1/3，性能还更好。**

核心insight：**不是所有data都同等informative，利用模型自己的forward pass信号做selection和weighting，是cheap yet powerful的inductive bias**。这跟SFT里"data quality > data quantity"的发现呼应，但在RL中更subtle——需要同时考虑prompt difficulty distribution和rollout exploration distribution。

参考verl框架：https://github.com/volcengine/verl
参考DAPO：https://arxiv.org/abs/2503.14476
参考GRPO：https://arxiv.org/abs/2402.03300
参考DeepSeek-R1：https://arxiv.org/abs/2501.12948

---

# PREPO: Efficient RLVR via Intrinsic Exploration 深度解析

Andrej, 这篇 paper 处理的是 RLVR (Reinforcement Learning with Verifiable Rewards) 训练中的核心痛点: **rollout generation 是计算瓶颈**, 而大量 rollout 对 gradient 贡献微乎其微。作者提出 PREPO (Perplexity-Schedule with Relative-Entropy Policy Optimization), 通过利用 data 的 **intrinsic properties** (几乎是 free 的信号) 来 prune prompts 和 reweight rollouts, 在 Qwen/Llama 上实现 up to 3× rollout reduction 而不损失性能。

---

## 1. Motivation: 为什么 RLVR 这么贵?

RLVR 的训练 loop 大致是: sample prompt $x$ → generate $G$ rollouts $\{o_i\}$ → compute verifiable reward $r_i$ → compute advantage → PPO-style update。**bottleneck 在 rollout generation**, 因为 LLM inference 的 FLOPs 远超 backward pass 在 actor 上的开销。

关键观察:
- **Prompt side**: 一些 prompt 太 trivial (模型已经会了, zero advantage ratio 高) 或太 difficult (模型完全不会, 也是 zero advantage), 这些对 gradient 贡献小
- **Rollout side**: confident (low entropy) 的 response 产生小 gradient, uncertain (high entropy) 的 response 产生大 gradient, 后者才真正驱动 exploration

作者想用这两个 **intrinsic signals** (PPL 和 entropy) 来做 data selection, 而 **不用** 额外训练参数化模型 (像 Qu et al. 2025 那样) 或者维护 replay buffer (像 Liu et al. 2025)。

**Intuition**: PPL 和 entropy 都是 forward pass 的副产品, 几乎 free, 如果能利用它们做 inductive bias, 就能省下大量无用的 rollout。

参考:
- DAPO: https://arxiv.org/abs/2503.14476
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- DeepSeekMath (GRPO): https://arxiv.org/abs/2402.03300

---

## 2. Preliminary Analysis: PPL 与 Pass Rate 的负相关

### 2.1 经验观察

作者在 DAPO-Math-17K 上发现 prompt PPL 与 passrate@16 之间存在 **statistically significant 负相关** (Spearman correlation around -0.17 ~ -0.23 across Qwen 和 Llama 模型, $p < 0.001$)。

$$\text{PPL}(x) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log\pi_\theta(x_t|x_{<t})\right)$$

这里 PPL 是 **prompt 自回归 likelihood 的几何平均**。低 PPL 意味着模型对 prompt 的 token 分布感到"熟悉" (high probability), 通常对应于容易题; 高 PPL 意味着模型对 prompt 感到"陌生", 通常是难题或 OOD prompt。

**Intuition**: 这跟 language modeling 中 PPL 作为 OOD detector 的经典用法一脉相承。但这里有意思的是, PPL 不仅是 OOD signal, 还能反映 **task difficulty for the current model**。

### 2.2 Low-PPL vs High-PPL 的 Training Dynamics

作者把 DAPO-Math-17K 分成 LOW-PPL (bottom 20%) 和 HIGH-PPL (top 20%) 两组, 分别训练, 观察到:

| Metric | LOW-PPL | HIGH-PPL |
|--------|---------|----------|
| Entropy | 低 (confident responses) | 高 (uncertain responses) |
| Reward gain | 早期更快提升 | 早期慢, 后期持续 |
| All-correct ratio | 快速饱和 | 缓慢上升 |
| Zero-advantage ratio | 后期高 (saturation) | 后期低 (still informative) |
| AIME24 final | 早期领先, 后期被反超 | 最终更好 |

**关键 insight**: LOW-PPL prompt 在 **早期** 提供强信号 (rapid reward gain), 但 **后期 entropy collapse 导致 exploration 死亡**; HIGH-PPL prompt 早期 signal 弱, 但 **保留 exploration 能力**, 最终 generalization 更好。

这就指向一个 **curriculum strategy**: 早期用 LOW-PPL 建立基础, 后期切换到 HIGH-PPL 保持 exploration。

参考:
- Curriculum learning for LLM: https://arxiv.org/abs/2505.14970
- Online difficulty filtering: https://arxiv.org/abs/2504.03380
- Forking tokens: https://arxiv.org/abs/2506.01939

---

## 3. PREPO 方法: 两个互补组件

### 3.1 PPL-Schedule Online Batch Selection

设当前 candidate batch $B = \{x_i\}_{i=1}^{|B|}$, 训练 progress $\rho \in [0, 1]$ (即 current step / total steps)。

**Step 1: 计算 dynamic PPL**

$$P_i(\rho) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log\pi_\rho(x_{i,t}|x_i, x_{i,<t})\right)$$

注意: $\pi_\rho$ 是 **当前模型** 的分布, 所以 $P_i(\rho)$ 是 **动态难度**——随着训练进行, 同一个 prompt 的 PPL 会变化。这跟静态 PPL 不同, 它反映了模型对 prompt 的 "current familiarity"。

**Step 2: 排序 + 滑动窗口选择**

$\sigma$ 是把 $B$ 按 $P_i(\rho)$ 升序排列的 permutation (低 PPL 在前)。然后选一个长度 $K$ 的 contiguous window:

$$\mathcal{T}_\rho = \{\sigma(j) : l(\rho) \leq j \leq l(\rho) + K - 1\}$$

其中起始 index:

$$l(\rho) = \lfloor \rho \cdot (|B| - K) \rfloor$$

**变量解释**:
- $\rho \to 0$: $l(\rho) \to 0$, window 从最 LOW-PPL 开始
- $\rho \to 1$: $l(\rho) \to |B| - K$, window 覆盖最 HIGH-PPL
- $K$ 是固定的 sub-batch size (实验中 $K/B = 20\%$)

**Intuition**: 这是一个 **linear pacing** curriculum, 从 easy 移动到 hard。比起 random selection, 它强制模型先巩固 familiar 区域, 再逐步推进到 unfamiliar 区域。作者提到也可以用 quadratic / exponential pacing, 但 linear 已经足够。

**为什么 dynamic 而非 static PPL?** 因为训练中模型对 prompt 的熟悉度在变化, static PPL 会很快过时 (例如一些 prompt 训练初期是 HIGH-PPL, 后期变成 LOW-PPL)。Dynamic PPL 自动 adapt 这个 shift, Figure 8 也验证了 prompt PPL range 在训练中保持稳定。

### 3.2 Relative-Entropy Weighting

PPL-Schedule 解决了 prompt 选择, 但 **rollout side** 还有问题: 早期 LOW-PPL prompt 产生的 rollout 往往 low entropy (confident), 训练梯度小, 而且容易导致 entropy collapse (即 entropy 在训练中快速下降, exploration 死亡)。

**Token-level entropy**:

$$H_t = -\sum_{v \in \mathcal{V}}\pi_\theta(v|o_{<t}, x)\log\pi_\theta(v|o_{<t}, x)$$

- $v$: vocabulary 中的 token
- $\pi_\theta(v|o_{<t}, x)$: 给定 prefix 的 next-token distribution
- $H_t$ 高 = 多个 token 都有 plausible probability = uncertain
- $H_t$ 低 = 集中在某个 token = confident

**Sequence-level entropy** (公式 5):

$$\bar{H}_i = \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}H_t$$

对 rollout $i$ 的所有 token 求 average, 反映整条 response 的 overall uncertainty。

**Batch-average entropy** (公式 6):

$$\bar{H} = \frac{1}{B}\sum_{k=1}^{B}\bar{H}_k$$

**Relative weight** (公式 7):

$$w_i = \frac{\bar{H}_i}{\bar{H}}$$

**这个 normalization 是关键**: 
- $w_i > 1$: rollout $i$ 比 batch average 更 uncertain, **被放大**
- $w_i < 1$: rollout $i$ 比 batch average 更 confident, **被缩小**
- $w_i$ 是 **scale-invariant** 的, 只取决于 entropy 与 batch mean 的 ratio

**为什么叫 "seek uncertainty within certainty"?** 因为 PPL-Schedule 早期选 low-PPL prompt (general 上更确定), 但 relative-entropy weighting 在这些 prompt 的 $G$ 个 rollout 中 **进一步放大相对不确定的**, 从而在确定的 prompt 中保留 exploration 信号。这是一个 **multi-scale** 的 exploration mechanism。

### 3.3 完整 Objective (公式 8)

$$\mathcal{J}_{\text{PREPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{T}_\rho, \{o_i\}_{i=1}^G \sim \pi_{\text{old}}(\cdot|x)}\left[\frac{1}{G}\sum_{i=1}^G w_i \cdot \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\Big(s_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(s_{i,t}(\theta), 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\hat{A}_{i,t}\Big)\right]$$

**变量逐一解释**:
- $\mathcal{T}_\rho$: PPL-Schedule 选出的 sub-batch prompts
- $G$: 每个 prompt 的 rollout 数 (实验中 $G=8$)
- $w_i = \bar{H}_i/\bar{H}$: relative-entropy weight
- $|o_i|$: rollout $i$ 的 token 长度
- $s_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|x, o_{i,<t})}{\pi_{\text{old}}(o_{i,t}|x, o_{i,<t})}$: token-level importance ratio (PPO 标配)
- $\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$: group-based advantage (GRPO 风格, 用同 prompt 内 $G$ 个 rollout 的 reward 归一化)
- $\epsilon_{\text{low}} = 0.2$, $\epsilon_{\text{high}} = 0.28$: asymmetric clipping (DAPO 风格, 上界更宽松以鼓励 exploration)

跟标准 GRPO/DAPO 的区别:
1. Prompt 从 $B$ 变成 $\mathcal{T}_\rho$ (PPL-Schedule)
2. Each rollout 乘以 $w_i$ (relative-entropy weighting)
3. 没有 KL regularization (follow DAPO)

---

## 4. 理论性质 (Appendix E)

### 4.1 Sum of Weights vs Batch Size (公式 11)

更精确的归一化 (考虑 sequence length):

$$\bar{H} = \frac{1}{\sum_{k=1}^B|o_k|}\sum_{k=1}^B\sum_{t=1}^{|o_k|}H_{k,t}$$

(注意: Appendix E 的 $\bar{H}$ 用的是 token-weighted 版本, 跟正文公式 6 略有不同, 更精细)

那么:

$$\frac{1}{B}\sum_{i=1}^B w_i \cdot |o_i| = \frac{1}{B\bar{H}}\sum_{i=1}^B|o_i|\bar{H}_i = \frac{1}{B\bar{H}}\sum_{i=1}^B\sum_{t=1}^{|o_i|}H_{i,t} = \frac{\sum_{i=1}^B|o_i|}{B}$$

**Intuition**: token-weighted average weight = average sequence length, 这保证了 effective batch size 不会漂移。如果所有 sequence 等长, $\frac{1}{B}\sum_i w_i = 1$。

Figure 20 显示 effective batch size 从 1.04 (训练初期, few high-entropy rollouts 被放大) 慢慢下降到 0.98 (训练后期, 更多 high-PPL prompts 进入, 整体 entropy 上升, normalization 让 average weight 略低于 1)。

### 4.2 对 Extreme Entropies 的 Sensitivity

偏导数:

$$\frac{\partial w_i}{\partial \bar{H}_j} = \begin{cases} \frac{1}{\bar{H}} - \frac{\bar{H}_j}{\bar{H}^2}\frac{|o_j|}{\sum_k|o_k|}, & i = j \\ -\frac{\bar{H}_i}{\bar{H}^2}\frac{|o_j|}{\sum_k|o_k|}, & i \neq j \end{cases}$$

**关键 insight**:
- **Self-sensitivity** ($i=j$): 两项近似抵消, 极大 $\bar{H}_j$ 对自己的 weight $w_j$ 影响小
- **Cross-sensitivity** ($i \neq j$): 严格负, 极大 $\bar{H}_j$ 会 **拉低所有其他 rollouts 的 weight**

这跟 naive "high entropy → high weight" 不同: outlier 不会无限放大自己, 而是 **拉高 batch mean, 让其他 rollout 被相对 shrink**。这是一种 **robust normalization**, 防止一个 outlier rollout 主导 gradient。

Figure 21 验证: weight distribution 集中在 0.7-1.5, long tail 超过 2 的极少, 超过 4 的罕见。

---

## 5. 实验结果

### 5.1 Main Results (Table 2 & 3)

| Model | Method | AIME25 | AIME24 | MATH | Olympiad | Avg ↑ | # Rollouts ↓ |
|-------|--------|--------|--------|------|----------|-------|--------------|
| Qwen2.5-Math-7B | Base | 9.17 | 20.80 | 72.26 | 39.56 | 35.45 | - |
| | + Random | 10.00 | 26.67 | 77.80 | 43.26 | 39.45 | 905K |
| | + GRESO | 18.33 | 25.83 | 77.80 | 26.83 | 37.46 | 654K |
| | **+ PREPO** | 12.81 | 26.15 | 77.85 | 41.58 | **39.59** | **540K** |
| Qwen2.5-Math-1.5B | + Random | 20.00 | 16.67 | 76.25 | 30.50 | 35.86 | 3.0M |
| | + GRESO | 15.38 | 20.00 | 76.65 | 24.17 | 34.16 | 2.5M |
| | **+ PREPO** | 20.00 | 16.67 | 76.25 | 32.00 | **36.23** | **1.1M** |
| Qwen3-4B | + Random | 60.00 | 70.00 | 96.00 | 59.33 | 71.33 | 553K |
| | + GRESO | 56.67 | 69.17 | 96.40 | 57.33 | 69.89 | 472K |
| | **+ PREPO** | 66.67 | 80.00 | 96.60 | 60.67 | **75.99** | **348K** |
| Llama3.1-8B | + Random | - | - | 14.60 | 46.63 (GSM8K) | 30.61 | 266K |
| | + GRESO | - | - | 16.80 | 41.77 (GSM8K) | 29.29 | 273K |
| | **+ PREPO** | - | - | 21.81 | 51.10 (GSM8K) | **36.55** | **115K** |

**Rollout Reduction 汇总**:
- Qwen2.5-Math-1.5B: **3× reduction** (3.0M → 1.1M, 63.3% 节省)
- Qwen2.5-Math-7B: **1.7× reduction** (905K → 540K, 40.3%)
- Qwen3-4B: **1.6× reduction** (553K → 348K, 37.1%)
- Qwen2.5-7B: **2.4× reduction** (716K → 304K, 57.5%)
- Llama3.1-8B: **2× reduction** (266K → 115K, 48.9%)

值得注意: GRESO 在某些 benchmark 上反而比 PREPO 高 (如 Qwen2.5-Math-7B 的 AIME25, GRESO 18.33 vs PREPO 12.81), 但 **OlympiadBench 上 GRESO 26.83 显著低于 PREPO 41.58**, average 还是 PREPO 更好。这暗示 PREPO 在 **out-of-distribution generalization** 上更稳健。

### 5.2 Ablation (Table 4)

| Model | Method | AIME25 | AIME24 | MATH | Olympiad | Avg |
|-------|--------|--------|--------|------|----------|-----|
| Qwen2.5-Math-7B | PPL-Schedule only | 10.00 | 23.33 | 74.60 | 39.21 | 36.79 |
| | + Relative-entropy (PREPO) | 12.81 | 26.15 | 77.80 | 41.58 | 39.59 |
| Qwen2.5-7B | PPL-Schedule only | 6.98 | 16.41 | 75.70 | 38.47 | 34.39 |
| | + Relative-entropy (PREPO) | 10.20 | 16.09 | 76.30 | 39.85 | 35.61 |

**两个组件互补**: PPL-Schedule 处理 prompt side 的 difficulty scheduling, relative-entropy 处理 rollout side 的 exploration preservation。两者加起来在 most benchmark 上都有 consistent gain。

### 5.3 Training Dynamics (Figure 5, 6, 17, 18)

对比三种配置 (Qwen2.5-Math-7B):
1. **High-PPL only**: entropy 高, 但 reward 上升慢
2. **Low-PPL only**: reward 上升快, 但 entropy 快速 collapse
3. **PPL-Schedule**: 平衡, entropy 缓慢下降, zero-advantage ratio 一直保持低

加入 relative-entropy 后, **zero-advantage ratio 进一步降低**, 说明更多 rollouts 贡献了 non-trivial gradient。

### 5.4 多样性分析 (Figure 10)

PREPO 在 MSC (Mathematics Subject Classification) categories 上的 coverage 比 random selection 更广, 说明 PPL-Schedule 不仅选 easy/hard, 还自动 cover 了 **更多 reasoning types**。这有点反直觉——本来以为 PPL-Schedule 会 bias 到某类问题, 但实际上 **dynamic PPL 让不同训练阶段选不同类型问题**, 自然扩大了 coverage。

### 5.5 Memorization Check (Figure 11)

为排除 PREPO 只是 memorize training data 的可能, 作者 truncate 40% 的 prompt 然后测试。如果模型靠 memorization, partial prompt 应该也能解决; 结果大部分 partial problems pass rate 接近 0, 说明模型 **依赖完整 context**, 是 reasoning 而非 memorization。

参考:
- GRESO: https://arxiv.org/abs/2506.02177
- Entropy mechanism (Cui et al.): https://arxiv.org/abs/2505.22617
- DAPO: https://arxiv.org/abs/2503.14476

---

## 6. 跟相关工作的关系

### 6.1 Curriculum Learning for LLM

- **Chen et al. 2025a (Self-evolving curriculum)**: 用模型自己定义难度, 需要额外 prompting
- **Parashar et al. 2025**: easy-to-hard curriculum, 用 pass rate 估计难度 (昂贵)
- **Zhang et al. 2025**: adaptive difficulty curriculum + expert-guided self-reformulation

PREPO 的优势: PPL 是 **forward pass 副产品**, 几乎 free, 不需要额外 LLM call 或者 pass rate estimation。Figure 9 显示 PPL computation 相对 rollout generation 的开销可以忽略。

### 6.2 Data Selection for RLVR

- **Qu et al. 2025 (LearnAlign)**: gradient alignment, 需要 training predictor
- **Wang & Guofeng 2025 (DUMP)**: distribution-level data selection
- **Liu et al. 2025**: rollout replay buffer

PREPO 跟这些方法的本质区别: **不需要训练参数化模型**, 直接用 intrinsic signals。

### 6.3 Entropy in RLVR

- **Cui et al. 2025**: entropy collapse 是 RLVR 的 key failure mode, 用 covariance-based update 减缓
- **Wang et al. 2025 (Forking tokens)**: high-entropy "forking tokens" 占少数但驱动大多数 reasoning improvement

PREPO 的 relative-entropy weighting 跟 "forking tokens" 思路呼应: 高 entropy 的 rollout 才是 informative 的。但 PREPO 在 **sequence level** 做 weighting, forking tokens 在 **token level**, 两者可以互补 (未来工作)。

### 6.4 跟 DAPO 的关系

PREPO 直接借用了 DAPO 的 asymmetric clipping ($\epsilon_{\text{low}} = 0.2 < \epsilon_{\text{high}} = 0.28$), 这本身是为了缓解 "advantage collapse" 问题 (一些 rollout 的 advantage 总是 0, 被 clip 掉的 ratio 没法贡献)。PREPO 在此基础上加了 **rollout-level reweighting**, 进一步保证 informative rollout 不被淹没。

参考:
- Self-evolving curriculum: https://arxiv.org/abs/2505.14970
- LearnAlign: https://arxiv.org/abs/2506.11480
- Down-sampling rollouts: https://arxiv.org/abs/2504.13818
- Replay buffer for RLVR: https://arxiv.org/abs/2506.05316

---

## 7. 一些 Deep Intuition & 个人 Commentary

### 7.1 PPL-Schedule 跟 Self-Distillation 的关系

LOW-PPL prompt 在某种意义上类似于 self-distillation: 模型在熟悉的 prompt 上做 RL, 早期就像在 **refining 自己已经会的 reasoning**, 起到 sharpening 的作用。这跟 knowledge distillation 中 teacher=student 的 degenerate case 有点像, 但 RLVR 的 reward signal 让它不只是 trivial copying, 而是 **reinforcing 正确的 reasoning paths**。

但这种 sharpening 有上限: 一旦 entropy collapse, 模型不再 explore 新 paths, 性能 plateau。这时切换到 HIGH-PPL prompt 强迫模型面对新场景, 重新激活 exploration。

### 7.2 Relative-Entropy Weighting 跟 Importance Sampling 的关系

从 RL 角度看, $w_i = \bar{H}_i / \bar{H}$ 可以理解为 **一个 implicit importance weight**, 但 weight 的 target 是 **entropy distribution** 而非 reward distribution。这跟 standard importance sampling (where weight = target/source) 不完全一样, 它更像一个 **heuristic reweighting**。

理论上, 这个 $w_i$ 不能保证 unbiased gradient estimator (除非加上 mild assumptions, 见 Appendix E), 但经验上 work。这可能因为:
1. PPO clipping 已经截断了 extreme gradients
2. $\sum_i w_i \cdot |o_i|$ 仍 close to batch size (Section 4.1), 总体 gradient scale 没漂移
3. RLVR 的 reward variance 本来就大, slightly biased reweighting 不显著影响收敛方向

### 7.3 为什么不直接做 entropy regularization?

经典 entropy regularization (如 Soft Actor-Critic) 在 objective 中加 $-\beta H(\pi)$, 鼓励高 entropy policy。但 RLVR 中:
- token-level entropy regularization 容易让模型生成 gibberish (高 entropy 但无意义)
- 需要小心 tune $\beta$

PREPO 的 sequence-level relative weight 更 **selective**: 它只 **放大 batch 内相对 uncertain 的 rollout**, 而 **不直接 push policy distribution 变 flat**。这保留了 reward signal 对 policy 的影响, 同时让 informative rollouts 主导 gradient。

### 7.4 PPL-Schedule 是否是 Universal?

PPL 作为 difficulty signal 在 math reasoning 上 work, 但 **其他 domain 可能不 work**。例如:
- **Code generation**: prompt PPL 可能主要反映 code syntax 而非 logical difficulty
- **Creative writing**: "difficulty" 概念本身 ill-defined
- **Multi-turn dialogue**: prompt 是 short user query, PPL 信号弱

作者 limitation section 也承认只测了 math。但 **核心 insight** ("use model's intrinsic signals to select data") 应该 generalize, 只是具体 signal 可能换成别的 (e.g., response length, self-consistency score, etc.)。

### 7.5 跟最近 Reasoning Model 工作的 Connection

最近 Reasoning LLM (DeepSeek-R1, OpenAI o1-style) 的 RL training 都面临一个 trade-off:
- **More rollouts per prompt**: 更稳定的 advantage estimate, 但贵
- **Less rollouts**: 便宜, 但 advantage noise 大, 训练不稳

PREPO 通过 **selective rollout** + **rollout reweighting** 在两者之间找平衡: 不是均匀分配 rollout budget, 而是 **集中到 informative prompts 和 informative rollouts**。这个思路跟 "data quality > data quantity" 在 SFT 中的发现高度一致。

参考:
- Reasoning or memorization (data contamination): https://arxiv.org/abs/2507.10532
- HybridFlow (verl): https://arxiv.org/abs/2409.13221
- vLLM: https://arxiv.org/abs/2309.06180

---

## 8. 个人 Caveats / Open Questions

1. **Online selection ratio 固定 20%**: 作者承认没系统测试其他 ratio。如果 ratio 太小, PPL-Schedule 的 window 可能 miss 中间难度的 prompt; 太大, 则接近 random。最优 ratio 可能跟 model size 和 dataset 相关。

2. **Response length 限制 32K**: 现 reasoning model (如 DeepSeek-R1) 经常生成 100K+ tokens。PREPO 的 sequence-level entropy 在长 response 上的行为没验证——可能少数 high-entropy tokens 主导整个 $\bar{H}_i$。

3. **PPL computation 在 long prompt 上的开销**: 论文说 PPL 是 forward pass 副产品, 但如果 prompt 长 (e.g., 4K-32K tokens), 计算 PPL 仍需要 full forward。是否可以用 **partial PPL** (前 N tokens 的 PPL) 作为近似? 这能进一步省 compute。

4. **Dynamic PPL 的 cost**: 每个 step 都要 re-compute PPL on the candidate batch $B$, 这意味着每 step 都 forward 整个 $B$ (尽管只是 forward, 没 backward)。如果 $|B|$ 很大, 这个 cost 也累积。Figure 9 说 "negligible" 但那是相对 rollout generation; 绝对开销有多大没量化。

5. **跟 recent RLVR methods 的 head-to-head**: 缺少跟 DeepSeek-R1, Kimi K1.5 等更 recent 方法的对比。这些方法用的是更大的 rollout budget 和更复杂的 reward shaping, PREPO 在这些 setting 下能否保持 advantage?

6. **Why asymmetric clipping works with PREPO?** $\epsilon_{\text{high}} > \epsilon_{\text{low}}$ 是 DAPO 的 trick, 鼓励 advantage > 0 的 rollout 有更大 update。PREPO 的 relative-entropy weight 跟这个 clipping 的相互作用没分析。理论上, 高 entropy rollout 往往 advantage 接近 0 (uncertain → 可对可错), clipping 上界更宽松可能让它更频繁被 "boost"。这是个 implicit synergy 值得探索。

---

## 9. 总结

PREPO 是一个 **simple yet effective** 的 RLVR 加速方法, 核心思想是 **利用 intrinsic data properties**:
- **Prompt side**: PPL 作为 dynamic difficulty signal → curriculum scheduling
- **Rollout side**: sequence-level entropy 作为 confidence signal → relative reweighting

两个组件互补: PPL-Schedule 解决 prompt selection, relative-entropy 解决 rollout 内的 exploration preservation。理论上通过 normalization 保证 effective batch size 稳定, 经验上在 Qwen 和 Llama 上达到 2-3× rollout reduction。

**核心 takeaway**: 在 RLVR 中, 不是所有 rollout 都同等 informative。利用模型自己的 forward pass 信号 (PPL, entropy) 做 data selection 和 reweighting, 是一个 cheap yet powerful 的 inductive bias。这跟 SFT 中 data curation 的成功故事呼应, 但在 RL 中更 subtle——需要同时考虑 **prompt side 的 difficulty distribution** 和 **rollout side 的 exploration distribution**。

参考论文主页 (推测, 基于作者):
- Yan Sun (NUS): https://www.comp.nus.edu.sg/~yansun/
- Stanley Kok (NUS): https://www.comp.nus.edu.sg/~kocks/
- Ant Group RL team: https://github.com/antgroup

代码 (作者说会 release): 应该会在 GitHub 上, 可以关注 verl 的 ecosystem: https://github.com/volcengine/verl

---

如果你想进一步深挖某个方向 (例如 entropy collapse 的 mathematical analysis, 或 PPL curriculum 的最优 pacing 函数), 我可以展开。
