---
source_pdf: Rethinking On-Policy Distillation of Large Language.pdf
paper_sha256: 7b43c679b5179aade605a2d9636c6a03e49234e0561bb5245e4de4ca6fcee53a
processed_at: '2026-08-11T23:29:08-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

好,刚才那版太学术了,我用更直觉的方式重新讲一遍,带上更多类比和直觉解释。

---

## 1. 这篇paper在解决什么问题?

现在的大模型post-training pipeline里,有一个特别流行的技术叫**On-Policy Distillation (OPD)**。简单说就是:

- 让student model自己generate答案
- 然后让teacher model对student generate的每一个token打分(给log-probability)
- student根据teacher的打分来调整自己

这听起来很美好——每个token都有supervision,比RL那种只在最后给一个outcome reward要dense得多。Qwen3、MiMo、GLM-5这些model都在用。

**但问题是**:实践中OPD经常莫名其妙失败。一个更强的teacher反而可能完全无法提升student,而一个更弱的teacher却能成功distill。这个反直觉的现象没人解释过。

这篇paper就是来搞清楚:**OPD到底在什么情况下work,什么情况下fail,以及为什么。**

参考: Qwen3技术报告 https://arxiv.org/abs/2505.09388, Thinking Machines Lab的OPD复现 https://thinkingmachines.ai/blog/on-policy-distillation

---

## 2. 核心发现:OPD成功需要两个条件

### 条件一:Thinking Pattern要match

这个可以用一个类比来理解:

想象你学打网球。你原本是个"底线型"选手(习惯在底线打持久战),现在请了个教练。如果教练是"上网型"选手(习惯冲到网前截击),即使他技术比你强很多,你也很难学——因为你们的"打法"完全不一样。你得先完全改变自己的打法,才能开始学他的技术。

但如果你请的教练也是底线型选手,只是技术比你更精湛,那你就能直接学他的步伐、击球点这些细节。

paper里的实验就是这样:

- Student: Qwen3-1.7B-Base (一个base model)
- Teacher A: Qwen3-4B (Non-thinking) — 这个model经过了完整post-training,thinking pattern和base model差异大
- Teacher B: Qwen3-4B-Base-GRPO — 直接在base model上做GRPO,thinking pattern和base model接近

结果Teacher B虽然benchmark分数和Teacher A差不多,但因为thinking pattern更match,OPD效果显著更好。

**直觉**: OPD本质上是让student模仿teacher的"思考方式"。如果两个人的思考方式完全不同,student根本无法理解teacher在每一步为什么这么选token,自然学不到东西。

### 条件二:Teacher要有"新知识"

这个类比:

想象你在学数学。你已经学完了高中课本,现在找两个老师:
- 老师甲:也只学过高中课本,但他比你学得好,做题更熟练
- 老师乙:学过大学数学,知道很多你没见过的方法和定理

虽然老师甲可能做题比你快,但他能教你的东西有限——因为你和他用的是同一套知识体系,只是熟练度差异。老师乙虽然可能某些高中题做得还不如老师甲快,但他能教你全新的解题思路。

paper里的实验:

DeepSeek family:
- Student: R1-Distill-1.5B (1.5B参数)
- Teacher A: R1-Distill-7B (7B,同一条distillation pipeline产出的,只是更大)
- Teacher B: Skywork-OR1-Math-7B (在R1-Distill-7B基础上又做了RL post-training)

Teacher A虽然更大,但和student是从同一批data里"学"出来的,只是scale的fitting不同。Teacher B通过额外RL获得了new capability,这些capability是student没见过的。

结果:Teacher B带来的gap recovery rate远高于Teacher A。

**关键metric**: Gap Recovery Rate

$$\text{Gap Recovery Rate} = \frac{\text{Acc}_{\text{after OPD}} - \text{Acc}_{\text{before OPD}}}{\text{Acc}_{\text{teacher}} - \text{Acc}_{\text{before OPD}}}$$

这个公式的意思:student原本和teacher之间有个性能gap,经过OPD后student能恢复这个gap的百分之多少。

---

## 3. 最震撼的实验:Reverse Distillation

这是整篇paper最striking的实验,也是最能揭示OPD本质的实验。

**实验设计**:

- 有个model叫JustRL-1.5B,它是通过RL从R1-Distill-1.5B训练出来的,性能比R1-Distill-1.5B更强
- 现在**反过来**:把JustRL-1.5B当student,把R1-Distill-1.5B(它的pre-RL版本)当teacher来做distillation
- 同时也试试用R1-Distill-7B(更大、benchmark略强)当teacher

**结果令人震惊**:无论用哪个teacher,JustRL-1.5B都退回到了它RL之前的水平!而且两个teacher的distillation轨迹几乎一模一样。

这个实验说明几个深刻的事实:

### Insight 1: OPD本质上是在transfer thinking pattern

distill到pre-RL checkpoint会把RL学到的gains全部抹掉。这说明OPD不是在"叠加性能",而是在"覆盖thinking pattern"。

用类比:你通过努力训练把网球从业余3.0水平提升到4.0,现在让你跟一个3.0水平的教练学(他教的是3.0水平的打法),你会慢慢回到3.0水平——因为你在imitate他的打法,而不是保留你已有的4.0能力。

### Insight 2: Benchmark性能完全不predict OPD效果

R1-Distill-7B比JustRL-1.5B略强,但distillation效果和更弱的R1-Distill-1.5B一样。两个teacher在student-visited states上的local target distribution几乎相同。

**直觉**: OPD是local optimization,只在student访问的states上计算divergence。teacher的benchmark性能反映的是teacher自己generate时的表现,但student-visited states可能和teacher-natural states完全不同。在这种OOD states上,7B和1.5B teacher可能给出相似的distribution。

### Insight 3: Scale差异不等于新知识

R1-Distill-7B和R1-Distill-1.5B是同一条pipeline的不同scale版本,本质上是对同一批data的不同拟合。它们没有"genuinely different"的knowledge可以transfer。

---

## 4. Token-level机制:Progressive Alignment

### 4.1 成功的OPD长什么样?

paper定义了三个关键metric来monitor训练动态:

**Overlap Ratio**: student和teacher的top-k token set的交集比例

$$\mathcal{M}_{\text{overlap}} = \mathbb{E}_t\left[\frac{|S_t^{(p)} \cap S_t^{(q)}|}{k}\right]$$

其中:
- $S_t^{(p)}$: student在step $t$的top-k token set
- $S_t^{(q)}$: teacher在step $t$的top-k token set  
- $k$: 默认16

**直觉解释**: 这个metric问"student和teacher在每一步都考虑的候选词,有多少是重叠的?" 如果重叠很低,说明student和teacher"想到一块儿去"的程度很差。

**Overlap-Token Advantage**: 衡量overlap tokens上的分布匹配度

$$A_t(\nu) = \bar{p}_t(\nu)(\log \bar{q}_t(\nu) - \log \bar{p}_t(\nu))$$

$$\mathcal{M}_{\text{adv}} = \mathbb{E}_t\left[\frac{1}{|S_t^{(p)} \cap S_t^{(q)}|}\sum_{\nu} A_t(\nu)\right]$$

其中 $\bar{p}_t, \bar{q}_t$ 是在交集上renormalize后的分布。

**直觉**: 这个metric问"在两人都考虑的那些词里,student的confidence和teacher匹配吗?" 接近0说明匹配好,大负值说明student overconfident。

**Entropy Gap**: 

$$\Delta H_t = |H(q_t) - H(p_t)|$$

**直觉**: student和teacher的"不确定性"差距。差距小说明student的犹豫程度和teacher一致。

### 4.2 成功vs失败的signature

实验:R1-Distill-1.5B作student,对比两个teacher:
- JustRL-1.5B → 成功,gap recovery 80%+
- R1-Distill-7B → 失败,几乎无提升

成功的run里,三个metric都呈现明确的趋势:
- Overlap ratio: 72% → 91% 稳步上升
- Overlap-token advantage: 向0收敛
- Entropy gap: 持续缩小

失败的run里,三个metric全部停滞。

**更惊人的数据**: overlap tokens承载了**97%-99%**的total probability mass!这说明alignment不是表面上的set coincidence,而是在概率真正集中的tokens上对齐。

### 4.3 Self-Reinforcing的Virtuous Cycle

paper发现一个正反馈循环:

1. 一旦某些token进入student和teacher的共同high-probability region
2. Teacher favor这些token,reverse KL update在这些token上concentrate更多mass
3. 这些token在student的top-k里越来越稳固
4. Competing的non-overlap tokens被挤出top-k
5. Overlap region扩大,alignment加强
6. 回到第1步,循环加速

这解释了为什么早期overlap低就很难recover——self-reinforcing cycle启动不了。

### 4.4 Ablation实验:只优化overlap就够了

为了验证causality(不只是correlation),paper做了ablation:

- **Student Top-k**: 优化student的完整top-k set
- **Overlap Top-k**: 只优化student和teacher top-k的交集
- **Non-Overlap Top-k**: 只优化对称差(只在一边的tokens)

结果:Overlap Top-k几乎完全recover Student Top-k的性能,Non-Overlap Top-k显著更差。

**结论**: OPD的optimization signal主要来自overlap region的gradient,non-overlap tokens贡献很少。这不仅是correlation,是causation。

---

## 5. 怎么fix失败的OPD?

paper提了两个practical strategy:

### 5.1 Off-Policy Cold Start

**问题**: student和teacher thinking pattern差距太大,OPD从一开始就无法exploit teacher signal。

**解决方案**: 先用off-policy SFT把student拉到teacher附近,再开始OPD。

具体做法:
1. 让teacher (Qwen3-4B Non-thinking) generate 200K条response
2. 用这些response对student (Qwen3-1.7B-Base)做SFT,得到Qwen3-1.7B-SFT
3. 从SFT checkpoint开始做OPD

**效果**: 比直接从base model做OPD显著更好。初始overlap ratio更高,训练更stable,最终性能ceiling也更高。

**直觉**: 这就像学新语言之前先背单词。直接让你跟native speaker对话(OPD),你啥都听不懂;先背单词(SFT),有了基础后再对话,才能学到语法和表达方式。

### 5.2 Teacher-Aligned Prompts

**Idea**: teacher的policy是由它post-training时看到的prompts塑造的。如果OPD时用类似的prompts,student生成的states会更compatible with teacher。

**Template实验**:
- 同样的math problems
- 原template: "Solve the following math problem step by step. The last line..."
- Teacher-aligned template: "{Question} Please reason step by step, and put your final answer within \boxed{}."

仅仅切换template,三个benchmark都有提升!

**Content实验**:
- DAPO-Math-17K (teacher的RL training data)
- DeepMath deduplicated子集 (in-domain但不overlap)

teacher-aligned content性能更好,但有一个subtlety: overlap ratio反而更低,但cumulative probability mass on overlap tokens更高——student concentrate mass on fewer but more strongly shared tokens。

**副作用**: student entropy显著降低,可能over-suppress exploration。Practical建议是混合teacher-aligned和out-of-distribution prompts。

---

## 6. Dense Supervision的代价

这是paper最philosophical的部分——揭示OPD"free lunch"背后的tension。

### 6.1 Trajectory Depth的Sweet Spot

实验:不同max response length (0.5K到15K)做OPD。

结果:
- 0.5K, 1K: 太短,supervised tokens不够,学不到东西
- 3K, 7K: 最佳sweet spot
- 10K, 15K: 性能plateau甚至decline,训练后期collapse

**为什么?** Student生成的prefix越来越长,这些prefix drift远离teacher熟悉的states。Teacher在OOD prefix上给出的reward越来越noisy。

**Instability的传播方向**: 从response末尾开始,entropy先在suffix升高,然后逐步向前传播。

**Teacher continuation实验**: 
- Student generate完整rollout,truncate到不同position,让teacher continue
- 1K prefix: teacher能提升+0.37 accuracy
- 16K prefix: 只能提升+0.02

Teacher的advantage随prefix depth单调下降。

**直觉**: 想象你跟着老师学写作。老师能轻松纠正你第一段的用词,但如果你写了一万字,老师在最后一段给出的修改建议可能已经不是基于"好文章应该怎么写",而是基于"这一万字已经这样了,接下来怎么续"——这种指导质量远不如从头开始写。

### 6.2 全局Informative不等于Local Exploitable

这个发现很反直觉:

失败的7B teacher给的sequence mean reward,和成功的JustRL-1.5B teacher一样informative(AUROC分别是0.75和0.73)。也就是说,两个teacher都能正确区分correct和incorrect rollouts。

**那为什么7B teacher失败了?**

paper提出一个hypothesis(未直接验证): **anisotropy**。

7B teacher的per-token advantage虽然individually large,但在sequence内的不同positions上方向不一致。当这些heterogeneous signals聚合为gradient时,它们partially cancel,导致small effective gradient despite large per-token rewards。

JustRL-1.5B因为thinking pattern compatible,把advantage集中在更coherent的token subset上,gradient方向一致,虽然per-token signal小但合力大。

**类比**: 拔河比赛。10个人每人用力100斤但方向各异,合力可能还不如5个人每人50斤但方向一致。Global reward signal强不代表local gradient可exploitable。

### 6.3 Sampled-Token OPD足够了

这个发现很实用:

对比Top-k OPD (k=1, 4, 16, 64) vs sampled-token OPD(只用student实际sample的那个token):

- Sampled-token ≈ Top-4/16/64 性能
- Top-1明显更差且unstable

**为什么sampled-token够用**: 每一步sample不同token,proportionally to student自己的distribution,提供unbiased coverage of high-probability region across training。

**为什么Top-1不行**: 始终选argmax,小policy change会flip哪个token排第一,reward signal不稳定且biased。

**直觉**: Top-1就像只听老师讲一个标准答案,sampled-token就像听老师对不同可能性的评价,后者更全面。

---

## 7. 整篇paper的Intuition Stack

### 7.1 层次化理解

**Layer 1 - 现象**: OPD有时成功有时失败,更强teacher不保证更好distillation。

**Layer 2 - 条件**: 成功需要两个条件同时满足
- Thinking pattern compatibility (决定能否开始学)
- New knowledge beyond student (决定学到什么)

**Layer 3 - 机制**: 成功的OPD表现为progressive alignment on high-probability tokens,这是一个self-reinforcing的dynamic。Overlap region是optimization的locus,承载97-99%概率质量。

**Layer 4 - 局限**: Dense supervision随trajectory depth degrade,global reward informative不保证local gradient exploitable。

### 7.2 一句话总结

**OPD本质上是在student-visited states上做thinking pattern transfer。它work的条件是:student能理解teacher在想什么(初始overlap高),且teacher有student没见过的新知识。它的"dense reward"优势在long-horizon上会degrade,因为student prefix drift让teacher的supervision变得noisy。**

### 7.3 实用take-away

如果你要implement OPD:

1. **选teacher**: 同family + post-trained with new data > 更大但同pipeline的teacher
2. **准备student**: 如果thinking pattern差异大,先做off-policy SFT cold start
3. **选prompts**: 优先用teacher post-training data中的prompts,但要mix一些OOD prompts保持entropy
4. **选实现**: Sampled-token OPD足够,避免Top-1
5. **注意长度**: 3K-7K是sweet spot,超过10K要小心collapse

---

## 8. 相关联想与open questions

### 8.1 与Exposure Bias的关系

Off-policy distillation有exposure bias (Bengio et al., 2015, https://arxiv.org/abs/1506.03099): student在teacher-generated sequence上训练,但inference时从自己distribution generate,error累积。

OPD通过在student自己的rollout上计算supervision来mitigate这个问题。但paper揭示OPD有自己的failure mode——thinking pattern mismatch和new knowledge缺失。

这就像:off-policy的问题是"在别人走过的路上学,on-policy让你自己走,但你可能走到完全陌生的地方,老师也帮不了你"。

### 8.2 与Curriculum Learning

Cold start策略本质上是curriculum:先简单(SFT在teacher outputs上)再复杂(OPD)。这与Bengio et al. 2009 (https://arxiv.org/abs/2207.03232)的curriculum learning精神一致。

Trajectory length的sweet spot也暗示需要progressive length extension curriculum。

### 8.3 与RLHF的对比

RLHF用sparse outcome reward + learned reward model。OPD用dense per-token reward from teacher。

paper的发现表明: dense不等于better。Reward reliability比reward density更重要。这呼应了RLHF的一些实践发现——过于dense的reward modeling容易reward hacking。

参考: InstructGPT https://arxiv.org/abs/2203.02155

### 8.4 对Self-Distillation的启示

Self-distillation (https://arxiv.org/abs/2601.20802, https://arxiv.org/abs/2601.19897)用单一model作为自己的teacher with privileged information。

"New knowledge" condition在self-distillation中变成:privileged information是否提供genuinely new knowledge beyond student training data?

如果privileged information只是rephrase student已知的东西,OPD会overwrite existing patterns without adding capability——这解释了Kim et al. 2026 (https://arxiv.org/abs/2603.24472)观察到的self-distillation有时degrade reasoning。

### 8.5 对Long-Horizon的Implication

paper明确指出OPD在long-horizon (extended CoT或agentic multi-turn)上可能不extend cleanly。Trajectory length ceiling需要hybrid方法:

- 短segment上用dense token-level supervision
- 长horizon上用sparse outcome-level reward
- 可能需要curriculum策略progressively extend supervised horizon

这与AlphaGo的思路类似: policy network + value network + MCTS,在不同time scale上用不同supervision。

### 8.6 "New Knowledge"的formalization是open problem

paper用empirical way定义"new knowledge" (post-trained vs same pipeline)。但formal definition缺失。

可能的information-theoretic formalization:
- Teacher conditional $\pi_T(\cdot | x, y_{<t})$ 相对于student training data的conditional mutual information
- 或者teacher能解决而student training data无法cover的问题空间比例

这个方向值得future work。

### 8.7 与Mode Collapse的关系

Section 5.2发现teacher-aligned prompts会降低student entropy。这是mode collapse的前兆。

OPD的reverse KL本身有mode-seeking特性,加上teacher-aligned prompts会加剧这个问题。Mixing OOD prompts是mitigation策略。

这与GAN训练中的mode collapse问题有类似structure: 优化目标可能induce distribution sharpening,需要explicit entropy regularization或diversity promotion。

### 8.8 Anisotropy Hypothesis的open question

paper未直接验证anisotropy hypothesis。如果验证,需要分析per-token gradient的方向结构。

这个hypothesis如果成立,暗示需要新的objective:
- 能exploit anisotropic reward structures
- 对gradient direction consistency加权
- 可能类似natural gradient或Fisher information加权

这是paper最exciting的open direction之一。

---

## 9. 最后的meta-reflection

这篇paper的methodology值得学习:

1. **从现象到机制**: 先empirically identify governing conditions,再probe token-level mechanism,最后propose recipe。这种decomposition方式为理解其他training dynamics提供了template。

2. **Reverse distillation**: 这个实验设计很巧妙——通过"反向"distill来暴露OPD的本质。这种"negative experiment"比positive实验更有informative价值。

3. **Controlled ablation**: Overlap vs Non-Overlap Top-k的ablation建立causality,不只是correlation。

4. **Dynamic metrics**: Overlap ratio, overlap-token advantage, entropy gap三个metric构成alignment的signature。这种metric design可以apply到其他training dynamics分析。

5. **承认limitation**: paper明确指出anisotropy hypothesis未验证,long-horizon是open problem,"new knowledge"缺乏formalization。这种intellectual honesty很重要。

paper的deeper message可能是: **"Free lunch" narratives in ML往往掩盖了fundamental tensions。Dense supervision的代价是reliability degradation;thinking pattern transfer的代价是existing capability overwrite。理解这些tensions比盲目apply技术更重要。**

这让我想到Rich Sutton的"The Bitter Lesson" (http://www.incompleteideas.net/IncIdeas/BitterLesson.html) ——我们对learning dynamics的intuition经常是错的,只有systematic empirical investigation才能揭示真相。

---

## 参考链接汇总

- Paper code: https://github.com/thunlp/OPD
- MiniLLM (OPD首次formalization): https://arxiv.org/abs/2306.08543
- GKD (on/off-policy unified framework): https://arxiv.org/abs/2306.13649
- Qwen3 technical report: https://arxiv.org/abs/2505.09388
- Thinking Machines Lab OPD: https://thinkingmachines.ai/blog/on-policy-distillation
- Exposure bias (Bengio et al. 2015): https://arxiv.org/abs/1506.03099
- Teacher assistant (Mirzadeh et al. 2020): https://arxiv.org/abs/1902.03393
- Distillation scaling laws (Busbridge et al. 2025): https://arxiv.org/abs/2502.08606
- Self-distillation (Hübotter et al. 2026): https://arxiv.org/abs/2601.20802
- InstructGPT (RLHF): https://arxiv.org/abs/2203.02155
- The Bitter Lesson (Rich Sutton): http://www.incompleteideas.net/IncIdeas/BitterLesson.html

---

# Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe

这篇paper来自Tsinghua University的thunlp团队, 系统性地解构了On-Policy Distillation (OPD) 的训练动态。paper的核心贡献在于从三个层次剖析OPD: Phenomenology (现象学), Mechanism (机制), 以及Practical Recipe (实用配方)。

参考链接:
- Paper: https://arxiv.org/abs/2503.17147 (推测, 基于thunlp/OPD repo)
- Code: https://github.com/thunlp/OPD
- 相关工作: MiniLLM (https://arxiv.org/abs/2306.08543), GKD (https://arxiv.org/abs/2306.13649)

---

## 1. Background: On-Policy Distillation的数学基础

### 1.1 基本设定

给定一个student model $\pi_{\theta}$ 和一个teacher model $\pi_{T}$, 两者都在vocabulary $\mathcal{V}$ 上定义next-token distribution。输入prompt $x = (x_1, \ldots, x_n)$, response $y = (y_1, \ldots, y_m)$, 前缀记为 $y_{<t} \triangleq (y_1, \ldots, y_{t-1})$。

OPD的核心特征是student自己采样rollout $\hat{y} \sim \pi_{\theta}(\cdot | x)$, 然后在student-visited states上获取teacher的per-token log-probabilities作为dense reward signal。

### 1.2 序列级到token级的精确分解

OPD的标准目标函数是sequence-level reverse KL (公式1):

$$\mathcal{L}_{\mathrm{OPD}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}_x}\Big[D_{\mathrm{KL}}\big(\pi_{\theta}(\cdot | x) \| \pi_T(\cdot | x)\big)\Big]$$

**变量解释**:
- $\mathcal{D}_x$: prompt dataset, $\mathcal{D}_x = \{x^{(i)}\}_{i=1}^N$
- $D_{\mathrm{KL}}(P \| Q)$: reverse KL divergence, 注意这里是 $\pi_{\theta}$ 在前 (即reverse KL), 而非forward KL。reverse KL具有mode-seeking特性, 防止student把概率质量分散到teacher认为不可能的区域。
- $\mathbb{E}_{x \sim \mathcal{D}_x}$: 对prompt dataset的期望

利用autoregressive factorization, 可以精确分解为token-level (公式2):

$$\mathcal{L}_{\mathrm{OPD}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}_x, \hat{y} \sim \pi_{\theta}(\cdot|x)}\left[\sum_{t=1}^{T} D_{\mathrm{KL}}(p_t \| q_t)\right]$$

**变量解释**:
- $T \triangleq |\hat{y}|$: rollout的长度
- $p_t(\nu) \triangleq \pi_{\theta}(\nu | x, \hat{y}_{<t})$: student在prefix $\hat{y}_{<t}$处的next-token分布
- $q_t(\nu) \triangleq \pi_T(\nu | x, \hat{y}_{<t})$: teacher在同一prefix处的next-token分布
- $\hat{y}_{<t}$: student自己生成的前缀, 这是"on-policy"的本质——在student访问的states上计算divergence

### 1.3 三种实现粒度

paper对比了三种实现方式:

**Sampled-Token OPD** (公式3): 最轻量, 只评估student实际采样的那个token

$$\mathcal{L}_{\mathrm{OPD}}^{\mathrm{sample}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}_x, \hat{y} \sim \pi_{\theta}(\cdot|x)}\left[\sum_{t=1}^{T} \ell_t^{\mathrm{sample}}\right]$$

其中 $\ell_t^{\mathrm{sample}} \triangleq \log p_t(\hat{y}_t) - \log q_t(\hat{y}_t)$。

这里的关键insight: $\mathbb{E}_{\hat{y}_t \sim p_t}[\ell_t^{\mathrm{sample}}] = D_{\mathrm{KL}}(p_t \| q_t)$, 所以每个 $\ell_t^{\mathrm{sample}}$ 是token-level reverse KL的**unbiased single-sample estimator**。这就是为什么sampled-token OPD在实验中表现接近top-k, 因为它对high-probability region提供了unbiased coverage (Section 6.3验证)。

**Full-Vocabulary OPD** (公式4): 在整个vocabulary上计算divergence, 梯度最dense但memory cost是 $O(B \cdot T \cdot M)$, 其中 $B$ 是batch size, $M = |\mathcal{V}|$ 是词表大小。

**Top-$k$ OPD** (公式5): 中间方案, 选择student的top-$k$个token作为subset $S_t = \mathrm{TopK}(p_t, k)$, 然后在renormalized分布上计算KL:

$$\bar{p}_t^{(S_t)}(\nu) = \frac{p_t(\nu)\mathbf{1}[\nu \in S_t]}{\sum_{u \in S_t} p_t(u)}, \quad \bar{q}_t^{(S_t)}(\nu) = \frac{q_t(\nu)\mathbf{1}[\nu \in S_t]}{\sum_{u \in S_t} q_t(u)}$$

$$\mathcal{L}_{\mathrm{OPD}}^{\mathrm{top-k}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}_x, \hat{y} \sim \pi_{\theta}(\cdot|x)}\left[\sum_{t=1}^{T} D_{\mathrm{KL}}\big(\bar{p}_t^{(S_t)} \big\| \bar{q}_t^{(S_t)}\big)\right]$$

**变量解释**:
- $\bar{p}_t^{(S_t)}, \bar{q}_t^{(S_t)}$: 在subset $S_t$上renormalized的student和teacher分布
- $\mathbf{1}[\nu \in S_t]$: 指示函数, token $\nu$ 在top-k set中为1, 否则为0
- 这个formulation丢弃了 $S_t$ 之外的mass, 是对full-vocabulary reverse KL的近似

---

## 2. Phenomenology: OPD成功的两个必要条件

### 2.1 Dynamic Metrics定义

paper定义了三个关键metrics来monitor OPD训练动态:

**Overlap Ratio** (公式6): student和teacher的top-k set的交集大小归一化

$$\mathcal{M}_{\mathrm{overlap}} \triangleq \mathbb{E}_t\left[\frac{|S_t^{(p)} \cap S_t^{(q)}|}{k}\right]$$

**变量解释**:
- $S_t^{(p)} = \mathrm{TopK}(p_t, k)$: student的top-k token set
- $S_t^{(q)} = \mathrm{TopK}(q_t, k)$: teacher的top-k token set
- $|S_t^{(p)} \cap S_t^{(q)}|$: 两个set的交集大小
- $k$: top-k的k值 (默认16)
- 这个metric衡量student的probability mass是否集中在与teacher相同的一组token上

**Overlap-Token Advantage** (公式7): 衡量overlap tokens上的分布一致性

$$A_t(\nu) \triangleq \bar{p}_t(\nu)(\log \bar{q}_t(\nu) - \log \bar{p}_t(\nu))$$

$$\mathcal{M}_{\mathrm{adv}} \triangleq \mathbb{E}_t\left[\frac{1}{|S_t^{(p)} \cap S_t^{(q)}|}\sum_{\nu \in S_t^{(p)} \cap S_t^{(q)}} A_t(\nu)\right]$$

**变量解释**:
- $\bar{p}_t, \bar{q}_t$: 在交集 $S_t^{(p)} \cap S_t^{(q)}$ 上renormalized的分布
- $A_t(\nu)$: 这是一个token-level的"advantage"信号, 衡量teacher相对于student在该token上的log-ratio, 加权student的概率
- 当 $\mathcal{M}_{\mathrm{adv}} \to 0$: student在overlap tokens上的confidence与teacher匹配
- 当 $\mathcal{M}_{\mathrm{adv}}$ 为较大负值: student overconfident (高 $p_t$ 但低 $q_t$)

**Entropy Gap** (公式8):

$$\Delta H_t = |H(q_t) - H(p_t)|$$

**变量解释**:
- $H(p_t), H(q_t)$: student和teacher在student-generated prefix处的entropy
- $\Delta H_t \to 0$: student的uncertainty profile匹配teacher的

### 2.2 Condition 1: Thinking-Pattern Consistency

**实验设置** (Section 3.1):
- Student: Qwen3-1.7B-Base
- Teacher 1: Qwen3-4B (Non-thinking) — thinking pattern与base model不匹配
- Teacher 2: Qwen3-4B-Base-GRPO — 通过GRPO对Qwen3-4B-Base做zero-RL得到, thinking pattern与base model匹配
- Dataset: DAPO-Math-17K
- 评估: AIME 2024, AIME 2025, AMC 2023, avg@16

**结果** (Figure 2-3): GRPO teacher虽然benchmark分数与Non-thinking teacher相当, 但因为thinking pattern与student (base model) 更兼容, 初始overlap ratio更高, 导致OPD效果显著更好。关键观察: **即使后期overlap ratio趋同, 性能差距仍然存在**, 表明早期thinking-pattern mismatch造成的distillation损失无法后期recover。

这个发现让我联想到Bengio et al. 2015的exposure bias, 但这里的问题更微妙——是thinking pattern层面的"semantic exposure bias"。

### 2.3 Condition 2: Higher Scores ≠ New Knowledge

**实验设置** (Section 3.2):

两个model family的对比:

**DeepSeek family**:
- Student: R1-Distill-1.5B
- Teacher A (same pipeline): R1-Distill-7B — 同一个distillation pipeline, 只是scale不同
- Teacher B (new knowledge): Skywork-OR1-Math-7B (SW-7B) — 在R1-Distill-7B基础上进一步RL post-training

**Qwen family**:
- Student: Qwen3-1.7B (Non-thinking)
- Teacher A: Qwen3-4B (Non-thinking) — 同pipeline
- Teacher B: Qwen3-4B-Non-Thinking-RL-Math — 在Qwen3-4B上用DeepMath子集做RL

**结果** (Figure 4): 同pipeline teacher带来的提升有限, post-trained teacher不仅绝对性能更高, 而且gap recovery rate显著更大。

**Gap Recovery Rate**定义:
$$\text{Gap Recovery Rate} = \frac{\text{Acc}_{\mathrm{after OPD}} - \text{Acc}_{\mathrm{before OPD}}}{\text{Acc}_{\mathrm{teacher}} - \text{Acc}_{\mathrm{before OPD}}}$$

这个metric衡量student能恢复多少teacher的性能优势。

**Intuition**: 同一个distillation pipeline产出的1.5B和7B模型, 实际上是从同一个data distribution中"学到"的能力的不同scale拟合, 没有genuinely new knowledge可以transfer。post-trained teacher通过RL获得了新的能力, 这些能力是可transfer的。

### 2.4 Reverse Distillation: 两个条件的联合验证

**实验设置** (Section 3.3): 这是最striking的实验。

- Student: JustRL-1.5B (通过RL从R1-Distill-1.5B训练得到)
- Teacher A: R1-Distill-1.5B (student的pre-RL checkpoint, 更弱)
- Teacher B: R1-Distill-7B (同family更大model, benchmark略强于student)

**结果** (Figure 5): 两个teacher都导致student退回到pre-RL水平, 且两个teacher的distillation轨迹几乎indistinguishable!

**关键insights**:

1. **OPD fundamentally learns thinking patterns**: distill到R1-Distill-1.5B (pre-RL)会overwrite RL获得的gains, 回到pre-RL性能。这表明OPD actively acquire teacher's thinking patterns, 而不是叠加性能。

2. **Benchmark performance does not predict OPD outcome**: R1-Distill-7B虽然benchmark略强, 但distillation效果与R1-Distill-1.5B相同 (都导致regression)。因为OPD minimize reverse KL on student-visited states, 而两个teacher在student-visited states上的local target distribution几乎相同。

3. **Higher scores ≠ new knowledge**: R1-Distill-7B和R1-Distill-1.5B只是同data不同scale的fit, 不提供新知识。scale差异不能转化为transferable signal。

这个实验让我想到meta-learning里的"catastrophic forgetting"——OPD overwrite了RL学到的东西, 因为它本质上是在做distribution matching而非capability accumulation。

---

## 3. Mechanism: Token-level的Progressive Alignment

### 3.1 成功vs失败OPD的对比

**实验设置** (Section 4.1):
- Student: R1-Distill-1.5B
- Teacher A (success): JustRL-1.5B — 80%+ gap recovery
- Teacher B (fail): R1-Distill-7B — 几乎无提升

**Dynamic Metrics观察** (Figure 6):

成功的OPD (JustRL-1.5B):
- Overlap ratio: 从~72%稳步上升到~91%
- Overlap-token advantage: 向0收敛
- Entropy gap: 持续缩小

失败的OPD (R1-Distill-7B):
- 三个metrics都停滞不前

**关键数据**: overlap tokens承载了**97%-99%的total probability mass** (Appendix B.1, Figure 18)。这意味着alignment不是set-level的coincidence, 而是在概率dominant tokens上的实质alignment。

### 3.2 Overlap Mass的精确定义

paper在Appendix B.1定义了overlap mass:

$$\mathcal{M}_{\mathrm{overlap-mass}}^{(p)} = \mathbb{E}_t\left[\sum_{\nu \in S_t^{(p)} \cap S_t^{(q)}} p_t(\nu)\right]$$

$$\mathcal{M}_{\mathrm{overlap-mass}}^{(q)} = \mathbb{E}_t\left[\sum_{\nu \in S_t^{(p)} \cap S_t^{(q)}} q_t(\nu)\right]$$

这两个metric分别衡量student和teacher在overlap tokens上的概率质量。实验中两者都稳定在97-99%, 证实overlap tokens是probability的真正locus。

### 3.3 Ablation: Overlap Tokens Sufficient的因果验证

**实验设置** (Section 4.2, Figure 7):

基于成功的JustRL-1.5B → R1-Distill-1.5B设置, 对比三种top-k变体:
- (i) Student Top-$k$: 优化完整student top-$k$ support $S_t^{(p)}$
- (ii) Overlap Top-$k$: 仅优化交集 $S_t^{(p)} \cap S_t^{(q)}$
- (iii) Non-Overlap Top-$k$: 仅优化对称差 $S_t^{(p)} \triangle S_t^{(q)}$ (symmetric difference)

**结果**:
- Overlap Top-$k$ 几乎完全recover Student Top-$k$的性能
- Non-Overlap Top-$k$ 显著更弱
- 训练动态: Overlap Top-$k$ 和 Student Top-$k$ 的overlap ratio曲线几乎indistinguishable (都从72%升到91%+), Non-Overlap先下降后部分恢复

**Causal insight**: OPD的primary optimization signal来自overlap region的gradients, non-overlap tokens贡献很少。这解释了为什么Student Top-$k$ 和 Overlap Top-$k$ 行为如此相似——student-only的额外tokens携带很少概率质量。

### 3.4 Self-Reinforcing Dynamic

paper提出一个重要的动态机制: **overlap optimization是self-reinforcing的**。

一旦某个token进入shared high-probability region并被teacher favor, reverse-KL updates会在其上concentrate更多mass, 逐步把competing non-overlap tokens挤出student的top-$k$ set。这创造了一个virtuous cycle, 维持alignment throughout training。

这种self-reinforcing机制让我联想到rich-get-richer现象, 或者说是一个正反馈循环——一旦alignment启动, 它会自我加速。

### 3.5 Auxiliary Optimization Diagnostics

Appendix B.2提供了额外的optimization-level诊断 (Figure 19):

1. **PG Loss**: 成功run从large initial mismatch稳步下降; 失败run初始loss就很小且变化不大 (小loss不是好事, 反映weak teacher signal)

2. **Gradient Norm**: 成功run的gradient norm初始大且持续; 失败run的gradient norm持续很小

3. **Extreme-advantage token的probability difference** $p_t(\nu) - q_t(\nu)$: 成功run稳步减小最大advantage token上的probability discrepancy; 失败run保持larger gap

---

## 4. Recipe: 两个修复失败OPD的策略

### 4.1 Off-Policy Cold Start (Section 5.1)

**动机**: 当student和teacher的thinking pattern差异太大, pure OPD失败, 因为student的initial policy无法exploit teacher的token-level supervision。

**方法**: 两阶段框架:
1. **Stage 1 (Off-policy SFT)**: student SFT在teacher-generated rollouts上, 拉近thinking pattern
2. **Stage 2 (Standard OPD)**: 从SFT-initialized checkpoint继续OPD

**具体配置**:
- Student: Qwen3-1.7B-Base
- Teacher: Qwen3-4B (Non-thinking)
- SFT Data: 200K teacher rollouts on OpenThoughts3-1.2M math subset
- SFT hyperparameters (Table 3): learning rate $1 \times 10^{-5}$, sequence length 14,336, cosine scheduler, warmup ratio 0.05, BF16
- OPD: 从Qwen3-1.7B-SFT继续, 用dedup后的30K prompts

**结果** (Figure 8): 
- SFT-initialized student大幅优于base-initialized student
- 性能gap在整个训练过程中持续, 说明cold start不仅改善早期优化, 还提高最终性能ceiling
- Overlap dynamics: SFT-initialized student初始overlap ratio高且trajectory smooth; base-initialized student初始低且unstable
- Entropy gap: SFT-initialized显著更小

**Intuition**: off-policy distillation通过SFT将student的初始distribution拉向teacher, 使teacher的token-level supervision在OPD开始时立即可exploit。这类似于RL里的warm-up策略。

参考: 类似思路在Sequence-Level Knowledge Distillation (Kim & Rush, 2016) 中也有体现, 但这里作为OPD的前置阶段。

### 4.2 Teacher-Aligned Prompts (Section 5.2)

**动机**: teacher的policy由post-training时看到的prompts塑造, 使用teacher-aligned prompts应该能提供更effective的supervision。

**两个粒度的实验**:

**(a) Prompt Template Alignment** (Figure 9):
- Student: R1-Distill-1.5B
- Teacher: JustRL-1.5B
- Prompt set: DAPO-Math-17K, 仅template不同

Original DAPO Template:
```
Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem. {Question} Remember to put your answer on its own line after "Answer:".
```

Teacher-Aligned Template:
```
{Question} Please reason step by step, and put your final answer within \boxed{}.
```

**结果**: 仅切换template就在三个benchmark上都提升性能。teacher-aligned template的初始overlap ratio更高, 收敛到更高水平。即使是minor的template变化也能materially影响OPD, 通过使student的generated states与teacher更compatible。

**(b) Prompt Content Alignment** (Figure 10):
- Teacher: Qwen3-4B-Base-GRPO (Section 3.1)
- Student: Qwen3-1.7B-Base
- 对比: DAPO-Math-17K (aligned with teacher RL training) vs DeepMath子集 (deduplicated)

Deduplication细节 (Appendix C.3): 两阶段——exact-match + semantic deduplication (使用all-mpnet-base-v2 embeddings, FAISS index, cosine similarity阈值0.6)。

**结果**: teacher-aligned prompts产生更强性能, 但有subtlety:
- Overlap ratio更低, 但cumulative student probability mass on overlap tokens更高
- 这表明student concentrates mass on fewer but more strongly shared tokens
- **副作用**: student entropy显著降低, 可能overly suppress探索

**Practical recommendation**: 混合teacher-aligned prompts和out-of-distribution prompts, 以preserve policy entropy和exploration能力。

---

## 5. Discussion: OPD Dense Supervision的Cost

### 5.1 Reward Quality随Trajectory Depth退化

**Response Length的sweet spot** (Figure 11a):

实验: R1-Distill-1.5B distill到JustRL-1.5B, 六种max response length (0.5K, 1K, 3K, 7K, 10K, 15K), 200 steps。

结果:
- 0.5K, 1K: 太少supervised tokens, sample-inefficient
- 3K, 7K: 最佳
- 10K, 15K: plateau或decline, 训练后期出现collapse

**Instability的back-to-front传播** (Figure 12, 13):

15K setting的训练动态显示: overlap ratio在late stage急剧下降, 伴随student entropy和gradient norm的spikes。

通过分析student entropy随output position的变化 (Step 180-250), 发现high entropy首先出现在response末尾, 然后逐步向前传播。

**Teacher entropy的类似pattern** (Appendix D.1, Figure 23): teacher也呈现suffix-to-prefix的entropy增长趋势, 与teacher在later positions遇到unfamiliar prefixes导致noisier reward一致。

**Teacher Continuation的degradation** (Figure 11b):

实验: 从student-generated rollouts中truncate不同position, 让teacher continue。
- 1K prefix: +0.37 accuracy gain
- 16K prefix: 仅 +0.02

teacher的accuracy advantage随prefix depth单调下降。

**Intuition**: Dense reward在moderate-length reasoning上effective, 但reliability随depth degrade, 因为student prefix drift远离teacher熟悉的states。这暗示OPD可能无法干净地扩展到long-horizon (extended CoT或agentic multi-turn)。

### 5.2 全局Informative不保证Local Exploitable

**实验** (Section 6.2, Figure 14): 

对比Section 4.1的success (JustRL-1.5B) 和fail (R1-Distill-7B) 设置, 计算sequence mean reward:

$$\bar{r}(y) = \frac{1}{T}\sum_{t=1}^{T}\left[\log \pi_T(y_t | x, y_{<t}) - \log_{\theta}(y_t | x, y_{<t})\right]$$

**变量解释**: 这是sampled-token OPD下的per-token advantage的时间平均, 衡量teacher相对于student的整体"偏好"。

**结果**: 两个teacher都给correct rollouts更高的reward (AUROC: JustRL-1.5B = 0.73, R1-Distill-7B = 0.75)。失败的7B teacher的global signal同样informative, 与rollout correctness相关性相当。

**Anisotropy Hypothesis**: paper提出一个未直接验证的假设: 7B teacher的per-token advantages虽然individually large, 但在sequence内的不同positions上anisotropic (方向不一致)。当这些heterogeneous signals聚合为gradient update时, 它们partially cancel, 导致small effective gradients despite large per-token rewards。

对比之下, JustRL-1.5B因为与student thinking pattern compatible, 将advantage集中在更coherent的token subset上, gradient虽然per-token signals更小, 但方向一致, reverse KL的mode-seeking behavior可以amplify。

这个hypothesis与Section 4.1观察的"high per-token advantage + low gradient norm"的co-occurrence一致。理解reward landscape的geometry和开发能exploit anisotropic reward structures的objectives是open question。

这个insight让我想到RL里的credit assignment problem, 但在OPD中, 问题变成了"dense per-token reward的方向一致性"。

### 5.3 Sampled-Token OPD已经足够

**实验** (Section 6.3, Figure 15, 16):

设置: R1-Distill-1.5B student, JustRL-1.5B teacher, 对比Top-$k$ OPD ($k \in \{1, 4, 16, 64\}$) vs sampled-token OPD。

**结果**:
- Sampled-token OPD性能与Top-$k$ settings相当
- Top-1明显更差, 训练unstable (overlap增长不稳定, entropy和gradient norm有sharp spikes)
- Top-4显著更稳定但仍有late-stage dip
- Top-16, Top-64全程smooth
- Enlarging $k$ beyond 4收益negligible, 但computational overhead更大

**为什么sampled-token OPD有效**: 每步draw不同token, proportionally to student's own distribution, 提供high-probability region的unbiased coverage across training。

**Top-1失败的真正原因**: 始终select argmax token, concentrate reward在单一mode上。small policy changes可以flip哪个token排第一, 造成unstable reward signal that does not average out。本质是biased, mode-concentrated selection rule, 不是"tokens太少"。

---

## 6. 核心Intuition总结

### 6.1 OPD的本质: Thinking Pattern Transfer

从reverse distillation实验可以提炼出最深的insight: **OPD本质上是在做thinking pattern transfer, 而非capability stacking**。当distill到一个thinking pattern不同 (即使是更弱的) teacher时, student会overwrite自己已有的gains。

这提醒我们, OPD的"free lunch"叙事需要谨慎——dense supervision的代价是student的thinking pattern被重塑。

### 6.2 成功OPD的三个Signature

成功的OPD表现为三个co-occurring的dynamic signatures:
1. **Overlap ratio稳步上升** (72% → 91%+)
2. **Overlap-token advantage向0收敛**
3. **Entropy gap持续缩小**

这三个signature都指向同一机制: student progressively locate teacher的high-probability region, 在其上calibrate mass, 并match teacher的local confidence。

### 6.3 失败OPD的三个Condition

OPD失败当且仅当满足任一:
1. **Thinking pattern mismatch过大** (初始overlap低, 无法recover)
2. **Teacher无new knowledge** (同pipeline, 仅scale差异)
3. **Reward landscape locally flat** (虽然globally informative, 但anisotropic gradients cancel out)

### 6.4 长horizon的fundamental tension

paper揭示了一个fundamental tension: **supervision density vs supervision reliability**。Dense per-token reward在moderate length (3K-7K) 上effective, 但随depth degrade, 因为student prefix drift导致teacher的reward变得noisier。

这指向long-horizon reasoning和agentic settings的limitation, 可能需要hybrid方法: 短segment上的dense token-level supervision + 长horizon上的sparse outcome-level reward。

参考: 类似hybrid思路在InstructGPT (https://arxiv.org/abs/2203.02155) 的RLHF + SFT混合中也有体现, 但paper这里提出的是segment-level的hybrid。

---

## 7. Future Directions

paper在Section 8提出的几个方向:

1. **Beyond Math Reasoning**: 所有实验都在math benchmarks上, 代码和open-ended settings的OPD是否被同样的conditions和mechanism govern是open question。

2. **Pre-Training Impact**: "new knowledge" condition隐式依赖pre-training corpora差异, 但cross-family distillation (Qwen → LLaMA) confounds data divergence和tokenizer/architecture差异。Controlled pre-training ablation成本prohibitively expensive。

3. **Self-Distillation Dynamics**: 近期工作采用self-distillation (单一model作为自己的teacher with privileged information)。将insights扩展到self-distillation regime, where thinking-pattern consistency guaranteed但knowledge novelty来自privileged access。

4. **Long-Horizon and Agentic Settings**: trajectory-length ceiling motivates hybrid approaches和curriculum strategies。

---

## 8. 个人思考与相关联想

### 8.1 与Reverse KL的mode-seeking特性

paper强调使用reverse KL ($D_{\mathrm{KL}}(\pi_{\theta} \| \pi_T)$) 而非forward KL。Reverse KL的mode-seeking特性意味着student会concentrate mass在teacher的high-probability region, 而不会spread到teacher认为unlikely的区域。这解释了为什么overlap region是optimization的locus——reverse KL天然地drive student向teacher的modes收敛。

参考MiniLLM (Gu et al., 2023): https://arxiv.org/abs/2306.08543 首次formalize OPD for LLMs under reverse KL。

### 8.2 与Exposure Bias的联系

Off-policy distillation (SFT on teacher outputs) suffer from exposure bias (Bengio et al., 2015, https://arxiv.org/abs/1506.03099): student在teacher-generated sequences上训练, 但inference时从自己的distribution生成, errors accumulate。OPD通过在student自己的rollouts上计算supervision来mitigate这个问题。

但paper揭示OPD有自己的failure modes——thinking pattern mismatch和new knowledge缺失——这些是off-policy distillation没有的问题。

### 8.3 与Capacity Gap文献的对话

Mirzadeh et al., 2020 (https://arxiv.org/abs/1902.03393) 提出teacher assistant来解决large teacher-student capacity gap。Busbridge et al., 2025 (https://arxiv.org/abs/2502.08606) 给出distillation scaling laws, 发现U-shaped capacity regime。

paper指出这些分析主要集中在off-policy KD, OPD的capacity gap和distillability问题underexplored。Reverse distillation实验实际上show: 在OPD中, 即使是更大 (7B) 的同family teacher, 也不比更小的 (1.5B) teacher提供更多transferable signal, 只要两者thinking pattern相同。

### 8.4 与Thinking Machines Lab的工作

Thinking Machines Lab (https://thinkingmachines.ai/blog/on-policy-distillation) replicate了Qwen3的OPD recipe, 在fraction of RL compute cost下取得comparable gains, 独立confirm了on-policy dense supervision的practical efficiency。

paper的发现可以解释为什么Thinking Machines Lab的replication有效——他们的teacher (Qwen3) 和student shares thinking pattern, 且teacher有post-training new knowledge。

### 8.5 对Self-Distillation的启示

近期self-distillation工作 (Hübotter et al., 2026; Shenfeld et al., 2026; Zhao et al., 2026b) 用单一model作为自己的teacher with privileged information。paper的"new knowledge" condition在self-distillation中变成: privileged information (ground-truth solutions或execution feedback) 提供的knowledge是否genuinely new beyond student's training data?

这可能解释了为什么某些self-distillation设置degrade reasoning capability (Kim et al., 2026, https://arxiv.org/abs/2603.24472): 如果privileged information只是rephrase了student已经知道的东西, OPD会overwrite existing patterns without adding new capability。

### 8.6 与Curriculum Learning的潜在联系

Section 5的off-policy cold start本质上是一种curriculum: 先用简单 (off-policy, SFT) 阶段拉近thinking pattern, 再用复杂 (on-policy, OPD) 阶段refine。这与curriculum learning (Bengio et al., 2009, https://arxiv.org/abs/2207.03232) 的精神一致。

Section 6.1的trajectory length ceiling也暗示需要curriculum策略: 逐步extend supervised horizon。

### 8.7 Reward Hacking的视角

Anisotropy hypothesis (Section 6.2) 与reward hacking有微妙的联系: 当teacher的reward landscape locally flat或anisotropic, student可能无法有效exploit signal, 甚至被misleading的local optima吸引。这是dense supervision的一个hidden risk。

### 8.8 关于"New Knowledge"的formalization

"New knowledge" condition目前是empirical observation, 但缺乏formal definition。可能的formalization方向:
- Information-theoretic: teacher的conditional distribution $\pi_T(\cdot | x, y_{<t})$ 相对于student training data的mutual information
- Capability-theoretic: teacher能解决而student training data无法解决的问题的比例

这可能是future work的重要方向。

---

## 9. 实验数据汇总表

### Table: 主要实验设置和结果

| Section | Student | Teacher | Dataset | Key Result |
|---------|---------|---------|---------|------------|
| 3.1 | Qwen3-1.7B-Base | Qwen3-4B (Non-thinking) vs Qwen3-4B-Base-GRPO | DAPO-Math-17K | GRPO teacher更优 (thinking pattern匹配) |
| 3.2 (DS) | R1-Distill-1.5B | R1-Distill-7B vs SW-7B | DAPO-Math-17K | Post-trained SW-7B gap recovery更大 |
| 3.2 (Qwen) | Qwen3-1.7B (Non-thinking) | Qwen3-4B vs Qwen3-4B-RL-Math | DAPO-Math-17K | Post-trained teacher更优 |
| 3.3 | JustRL-1.5B | R1-Distill-1.5B vs R1-Distill-7B | DAPO-Math-17K | 两个teacher都导致regression到pre-RL |
| 4.1 | R1-Distill-1.5B | JustRL-1.5B (success) vs R1-Distill-7B (fail) | DAPO-Math-17K | 80%+ vs ~0% gap recovery |
| 4.2 | R1-Distill-1.5B | JustRL-1.5B | DAPO-Math-17K | Overlap Top-$k$ ≈ Student Top-$k$ > Non-Overlap Top-$k$ |
| 5.1 | Qwen3-1.7B-Base/SFT | Qwen3-4B (Non-thinking) | OpenThoughts3 + DAPO | SFT cold start显著更优 |
| 5.2 (a) | R1-Distill-1.5B | JustRL-1.5B | DAPO-Math-17K (不同template) | Teacher-aligned template更优 |
| 5.2 (b) | Qwen3-1.7B-Base | Qwen3-4B-Base-GRPO | DAPO vs DeepMath (dedup) | Teacher-aligned content更优但entropy更低 |
| 6.1 | R1-Distill-1.5B | JustRL-1.5B | DAPO-Math-17K (6 lengths) | Sweet spot 3K-7K; 10K+ collapse |
| 6.3 | R1-Distill-1.5B | JustRL-1.5B | DAPO-Math-17K | Sampled ≈ Top-4/16/64 > Top-1 |

---

## 10. 架构图解析 (Figure 1 Overview)

Figure 1提供paper的visual overview:

- **Top panel**: 展示"high score teacher may not provide additional information gain"——JustRL-1.5B和SW-7B的对比, 强调score ≠ transferable knowledge。

- **Middle panel**: 展示"OPD primarily learns thinking patterns"——Reverse distillation示意, 强调OPD本质是pattern transfer而非capability stacking。

- **Bottom panel**: 展示"OPD leads to progressive alignment on high-probability overlap tokens"——Overlap ratio随训练的上升曲线, 体现self-reinforcing dynamic。

这三层正好对应paper的三个核心section: Phenomenology (§3), Mechanism (§4), 和Recipe (§5)。

---

## 11. 总结性Intuition

这篇paper最深的贡献在于揭示了OPD的"free lunch"叙事背后的tensions:

1. **Dense supervision的代价**: OPD提供per-token reward, 但这种dense signal的reliability随trajectory depth degrade, 形成density vs reliability的tension。

2. **Thinking pattern overwrite**: OPD本质上transfer thinking patterns, 这意味着它会overwrite student已有的gains, 除非teacher提供genuinely new knowledge。这是一个zero-sum aspect of OPD。

3. **Local vs Global reward**: 即使teacher的global reward signal informative (AUROC ~0.75), local optimization geometry可能flat或anisotropic, 导致gradient无法exploit signal。

4. **Self-reinforcing alignment**: 成功的OPD有self-reinforcing的overlap dynamic, 但这也意味着一旦初始overlap低, 就很难recover——early stage的thinking pattern compatibility至关重要。

这些insights对于理解modern LLM post-training pipeline (Qwen3, MiMo, GLM-5等都采用OPD) 的成功和failure mode提供了重要基础, 也为未来的long-horizon reasoning和agentic settings的hybrid方法提供了方向。

paper的方法论也值得学习: 通过reverse distillation实验同时验证两个conditions, 通过controlled ablation (Overlap vs Non-Overlap Top-$k$) 建立causality, 通过dynamic metrics monitoring揭示机制。这种phenomenology → mechanism → recipe的decomposition方式为理解其他training dynamics提供了template。
