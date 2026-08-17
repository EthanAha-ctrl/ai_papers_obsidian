---
source_pdf: To MixorToMerge.pdf
paper_sha256: 8acb0bf8a83a123452c9d45d4fcd634e3778550a59437820226caaf06e9ea7ec
processed_at: '2026-08-12T16:29:31-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇paper

## 这篇paper在干嘛

现在大家都在搞LLM的post-training，目标是让一个model在math、coding、science、instruction following这几件事上都强。问题是有两条路可以走，到底走哪条？

**第一条路**: 把math、code、science、IF的数据混在一起，直接一口气RL训练。叫 **mixed multi-task RLVR**。DeepSeek-R1和Qwen3就是这么干的。

**第二条路**: 先各管各的，分别训练出math expert、code expert、science expert、IF expert这四个模型，然后再把这四个模型的权重揉到一起，搞出一个unified model。叫 **separate RL + model merging**。GLM-4.5和MiMo-V2走这条。

这两派都有自己的technical report，但谁都只说自己好，没仔细对比。这篇paper就是来当裁判的，认认真真做controlled experiment对比一下。

## 关键发现一: 混合训练竟然没有interference

按理说multi-task learning最怕的就是 **gradient interference** —— 学math的梯度把code的方向带偏了，互相打架。这是几十年来multi-task learning的老问题。

但这篇paper发现: 在RLVR场景下，这个问题基本不存在。

看Table 3的数据，你单独训math得到的expert，拿到AIME'24是71.51分。但你单独训code、science、IF得到的model，拿去做math题也能拿60-64分。也就是说，**训一个domain顺带把另外几个domain也带起来了**。

而且mixed multi-task RL直接在AIME'24拿73.85分，比任何单独merge的方法都好。这真的很反直觉。

更夸张的是cost: 四个domain单独训加起来6524 GPU hours，multi-task混着训只要2166小时，**只占33.2%**。性能还更好或者持平。

**直觉解释**: math、code、science这三个reasoning domain，底下共用一套"逻辑推理能力"。你训math的时候其实是在打磨这套共享能力，所以code和science也跟着涨。IF虽然没有reasoning那么强，但也能帮一把reasoning model的formatting和planning。

## 关键发现二: 为什么没有interference? weight更新的几何证据

光看分数不够，得看model内部到底发生了什么。

**实验1**: 看哪些参数被RL更新了

定义: 一个weight $w$ 算"被更新"当:
$$|w_{RL} - w_{SFT}| > \eta \cdot \max(|w_{RL}|, |w_{SFT}|), \quad \eta = 10^{-3}$$

- $w_{RL}$: RL之后的权重值
- $w_{SFT}$: SFT之后的权重值（作为anchor）
- $\eta = 10^{-3}$: bfloat16精度噪声之上的阈值
- max归一化: 避免大参数和小参数用同一个绝对阈值

得到每个model的更新位置mask $M_{RL} \in \{0,1\}^d$。然后算两个model的Jaccard overlap:

$$J(RL_1, RL_2) = \frac{|M_{RL_1} \wedge M_{RL_2}|}{|M_{RL_1} \vee M_{RL_2}|}$$

分子是两个model都更新的位置数，分母是至少一个model更新的位置数。

**结果**: math-coding的overlap是0.45，math-science是0.48，而random baseline只有0.18。也就是说**四个domain的RL更新了高度重叠的参数子集**。

**实验2**: 看更新方向是否align

光看"更新了哪些位置"不够，还要看"更新方向是否一致"。直接算cosine similarity在高维会有curse of dimensionality问题，所以先用orthogonal random projection把weight shift vector降到256维，再算cosine。

**结果**: 跨domain的cosine similarity都是positive的，而且math/code/science之间相似度比它们和IF的相似度更高。

**直觉**: 这就像是四个domain在参数空间里都在"同一个方向上使劲"，虽然各自推的力度不一样。这就解释了为什么没有interference——大家的目标方向不冲突，反而互相帮忙。

## 关键发现三: Policy neighborhood transfer —— 最有意思的机制

这部分是paper的精华。

之前Shenfeld et al.那篇"RL's Razor"说: KL divergence越大，性能退化越多。这篇paper在multi-domain场景下发现: **这个规律不成立**。

看Figure 4那个矩阵。以math测试为例: math expert和multi-task model的KL最小（符合直觉），但**coding expert和multi-task model的KL也很小**，同时multi-task model在math上还涨了0.8分。

意思是: coding expert在math问题上输出的分布，和multi-task model很接近。这就是**policy neighborhood**——一个domain的expert，在另一个domain的问题上，恰好跟最终model的想法很合得来。

形式化定义: 给domain A和它的expert $E_A$，domain B是A的neighborhood当:

$$\mathbb{E}_{x \sim A, \hat{y} \sim \pi_{E_B}(\cdot|x)} \left[ \log \frac{\pi_{E_B}(\hat{y}|x)}{\pi_{multi}(\hat{y}|x)} \right] < \varepsilon$$

变量意思:
- $x \sim A$: 从domain A的prompt分布采样input
- $\hat{y} \sim \pi_{E_B}(\cdot|x)$: 用B的expert生成response
- $\pi_{E_B}(\hat{y}|x)$: B expert给这个response的概率
- $\pi_{multi}(\hat{y}|x)$: 多domain合并model给这个response的概率
- 整个式子: 在A的input上、用B生成response、测B expert和multi model的KL
- $\varepsilon$: threshold，要跟 $\text{KL}(\pi_{E_A} \| \pi_{multi})$ 比较来定

**关键**: neighborhood不是input相似度，而是output policy在另一个domain上的相似度。

更妙的是这种关系**不对称**。看Table 6:

- Math+Coding merge后，math性能涨 (71.51→72.14，+0.63；AIME25甚至+2.66) → coding是math的neighbor
- Coding+Math merge后，coding性能跌 (LCB v5: 59.40→57.97，-1.43) → math不是coding的neighbor

为什么不对称? coding expert能合理处理math问题（因为code也包含逻辑推理），但math expert缺乏code所需的执行和语法knowledge。所以merge coding进math是"帮一把"，merge math进code是"拖后腿"。

**这给了merge的recipe**: 别盲目merge所有expert，先看看谁是谁的neighbor，只merge那些互为neighbor的。

## 关键发现四: Multi-task训练会产生emergent capability

这部分也挺妙的。问题: multi-task model和merged model，学到的"技能"和single-task model一样吗?

定义: 对每个test sample $i$，算model $m$ 相对SFT baseline的gain:
$$g_m^t(i) = \max(a_m^t(i) - a_{sft}^t(i), 0)$$

- $a_m^t(i)$: model $m$ 在sample $i$ 上的accuracy
- $a_{sft}^t(i)$: SFT baseline在sample $i$ 上的accuracy
- max(..., 0): 只保留涨的，丢掉退化的

然后union gain vector是所有single-task model在每个sample上取最大:
$$g_{union}^t(i) = \max(g_{math}^t(i), g_{science}^t(i), g_{coding}^t(i), g_{IF}^t(i))$$

最后算 $g_{union}^t$ 和multi-task model / Ties-merge / MT-OPD的gain vector的cosine similarity，看它们"学到的新技能"和single-task collective的"技能集合"有多overlap。

**结果** (Figure 5):
- Math的consistency最高——数学技能比较homogeneous，不容易被interference搞乱
- **RL-Multi的consistency最低**——它学到的技能和single-task最不像
- Merging方法的consistency高——它们主要是继承single-task的能力

**意思**: mixed multi-task训练时，不同domain之间互相"激发"，产生了一些single-task根本学不出来的能力。这就是**emergent capability**。而model merging只是在weight space上做几何平均，没法产生新能力，只能保留原来每个expert会的。

这也解释了为什么multi-task有时反而比single-task差——它有emergent能力，但也会有interference，两者并存。

## 关键发现五: RLVR免费送你一个Reward Model

这是最"哲学"的部分。把训练好的model拿来当judge，给同domain的response打分。两种方式:

- **Outcome verification**: 只看最终答案（System 1 直觉）
- **Process verification**: 看完整CoT（System 2 逻辑）

**Finding 1: 不同domain的error藏在不同的地方**

- Math/Code: 答案是推导过程的"lossy compression"，只看答案就是黑箱猜。process verification可以"white-box debug"，在推导链出错前就抓住。所以process verification更好。
- IF: error在执行表面（JSON语法、关键词限制），CoT反映的是intention，execution才反映failure。process verification会被"halo effect"骗——看着plan挺对，实际execution没跟上。所以outcome verification更好。

**Finding 2: RL训练会按domain特性特化verification能力**

- IF RL: outcome判断能力飙升（System 1 sharp），process仍然低
- Math/Code RL: process判断能力保持或加强（System 2 anchor强）

**Finding 3: Multi-task RL同时继承两套能力**

这是synergy的真正体现。Multi-task model同时拥有:
- IF训练带来的outcome sensitivity（System 1直觉）
- Math/Code训练带来的process verification（System 2逻辑）

看Table 7: Multi-task RL的Avg Judge (Outcome/Process) = 77.5/64.7，outcome judge全场最高，process也保持高水位。这就像一个人既会"凭直觉判断答案对不对"，又会"逐步检查推理过程"。

**Finding 4: MT-OPD (multi-teacher on-policy distillation)在process verification上更强**

可能因为student暴露在多个teacher的diverse reasoning trace和error pattern下，supervision更broad，sharpen了step-level consistency check，不依赖surface plausibility。

## 实操结论: 你应该怎么做

给Karpathy这种practitioner的take-away:

**1. Default选择mixed multi-task RLVR**
- 性能持平甚至更好
- GPU hours只要1/3
- 还可能有emergent能力
- 简单，不用先训四个model再merge

**2. 什么情况下用separate + merge?**
- 团队distributed，各domain并行开发，最后要集成
- 数据隐私/权限分开，没法混数据
- 想快速迭代单个domain expert，再merge

**3. 如果用merge，选Ties-Merging**
- 看Table 4平均分，Ties 59.41最高
- SCE和MT-OPD紧随其后
- 但每种方法有seesaw effect，看你要哪个domain强

**4. 别盲目全merge**
- Policy neighborhood关系不对称
- 想让math强，merge coding expert进来（coding是math的neighbor）
- 想让coding强，merge IF expert进来（IF是coding的neighbor）
- 但反过来不一定成立

**5. Verification horizon要match task**
- Logic task（math, code）: 用process verification当judge
- Constraint task（IF）: 用outcome verification当judge
- 通用judge: multi-task训练出来的model自带两套能力

**6. 你不需要单独训reward model**
- RLVR训练完，model本身就有self-discrimination能力
- 用它当Gen-RM直接judge自己trajectories

## 与其他工作的关系

- **vs Shenfeld "RL's Razor"**: 那篇说KL divergence和性能退化正相关，这篇在multi-domain场景下show这个correlation不成立。原因是multi-domain时KL relationship复杂，需要看policy neighborhood而非global KL。
- **vs Wu "Imbalanced gradients"**: 那篇担心multi-task RL post-training有gradient imbalance，这篇发现这种imbalance在RLVR+reasoning domain场景下问题不大。可能是verifiable reward的稀疏性+reasoning capability的sharing导致的。
- **vs Wen "RLVR implicitly incentivizes correct reasoning"**: 那篇说RLVR让base model自发学会正确推理，这篇进一步发现RLVR还自发学会verification（self-discrimination）。两者都说明RLVR的"免费bonus"很多。
- **vs Model Soups / Task Arithmetic**: 那些weight merging方法建立在linear mode connectivity假设上，这篇发现merging确实能保留能力，但产生不了emergent能力。Multi-task训练走的是non-linear trajectory，所以能emerge。

## 我的几个critique

虽然这篇paper很有价值，但有几个地方我会保持怀疑:

**1. 只在4B做了实验**
在大model上，reasoning substrate的sharing可能更强，synergy更明显。也可能反过来，大model本身能力已经饱和，multi-task的边际增益变小。不知道。

**2. SFT→RL简化pipeline**
真实post-training是SFT和RL交替多轮的，每轮SFT用上一轮RL的data。这篇paper的结论在多轮场景下是否成立? emergent capability是否能在多轮迭代中累积? 没测。

**3. Nemotron数据可能特殊**
Nemotron的SFT数据本身就multi-domain混好（Table 1: math 13% + math tools 8% + code 28% + science 16% + chat 30%），所以SFT出来的model已经有multi-domain底子。在这么mix的SFT上做RLVR没interference，可能是因为SFT已经"预热"了共享representation。换纯math SFT后再做multi-task RL，interference可能就显现了。

**4. Domain选偏**
math、code、science都是verifiable reward domain，reward是deterministic binary的。如果加入creative writing、safety alignment这种subjective reward domain，gradient interference可能就来了。verifiable reward domain之间的synergy可能是个特例，不能推广到所有multi-domain场景。

**5. Threshold $\eta = 10^{-3}$ 的选择**
这是为bfloat16精度选的，但不同model architecture、不同precision（fp16, fp32, int8）下应该有不同合理阈值。得到的"30% weight被更新"这个数字是否稳定? 如果换成 $\eta = 10^{-4}$，mask率会大幅变化，Jaccard overlap的interpretation也会变。

## 一句话总结

**Mixed multi-task RLVR在reasoning domain间没有传统multi-task learning的interference，反而有synergy，因为reasoning capability是共享substrate、且不同domain在policy space存在不对称的neighborhood关系；只用1/3 GPU hours能达到merging的效果，还能产生single-task训练出不来的emergent能力，顺便免费送你一个双模态（System 1+2）的self-discriminator。**

References:
- Paper repo: https://github.com/mosAI25/M2RL
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Qwen3: https://arxiv.org/abs/2505.09388
- GLM-4.5: https://arxiv.org/abs/2508.06471
- GRPO: https://arxiv.org/abs/2402.03300
- Nemotron 3 Nano: https://arxiv.org/abs/2512.20848
- Ties-Merging: https://arxiv.org/abs/2306.01708
- Task Arithmetic: https://arxiv.org/abs/2212.04089
- MT-OPD: https://arxiv.org/abs/2306.08543
- RL's Razor: https://arxiv.org/abs/2509.04259
- Imbalanced gradients: https://arxiv.org/abs/2510.19178
- RLVR implicitly incentivizes correct reasoning: https://arxiv.org/abs/2506.14245
- DAPO: https://arxiv.org/abs/2503.14476
- DARE: https://arxiv.org/abs/2402.02713
- Model Soups: https://arxiv.org/abs/2203.05482

---

# To Mix or To Merge: 多domain RLVR的深度技术解析

## 1. Paper核心问题与动机

这篇paper直击当前LLM post-training领域的一个核心实践问题: 当我们需要训练一个在math、coding、science、instruction following多个domain都达到expert-level的general模型时, 应该选哪种training paradigm?

两种主流paradigm:
- **Mixed multi-task RLVR**: 直接在混合的heterogeneous reward数据上同时训练
- **Separate RLVR + Model Merging**: 各domain单独训练expert, 再用weight merging或distillation合并

工业界两派分裂: DeepSeek-R1 (https://arxiv.org/abs/2501.12948) 和 Qwen3 (https://arxiv.org/abs/2505.09388) 采用mixed multi-task; GLM-4.5 (https://arxiv.org/abs/2508.06471) 和 MiMo-V2-Flash (https://arxiv.org/abs/2601.02780) 采用separate + merging。两边都没给详细对比分析, 这篇paper就是来填这个gap。

---

## 2. 实验设置的技术细节

### 2.1 模型与数据

Base model: **Qwen3-4B-Base** (https://arxiv.org/abs/2505.09388), 选4B是credibility + operability的折中。

SFT数据来自 **Nemotron 3 Nano** (https://arxiv.org/abs/2512.20848) 开源数据, 总共14M samples, 配比如Table 1:

| Domain | Samples | Proportion |
|---|---|---|
| Formal Proofs | 335,122 | 2.37% |
| Math | 1,878,601 | 13.30% |
| Math w/ Tools | 1,071,924 | 7.59% |
| Science | 2,263,340 | 16.04% |
| Code | 3,927,984 | 27.81% |
| Chat | 4,309,780 | 30.52% |
| Conversational Agent | 335,122 | 2.37% |

RLVR数据从4个domain提取, 总共约77k samples (Math 22k + Coding 19k + Science 20k + IF 17k)。

### 2.2 RL算法: GRPO

使用 **GRPO (Group Relative Policy Optimization)** (https://arxiv.org/abs/2402.03300), 核心思想是在一个group内用relative advantage替代critic。Reward是binary的verifiable reward $r \in \{0, 1\}$ 加上formatting reward, 用于强制CoT和最终answer的结构分离。

关键training settings (Table 2):
- batch size: 128 (single-domain) / 256 (MT-OPD)
- rollout: 16 per prompt (single) / 4 (MT-OPD)
- 单domain: 200 steps; multi-task: 400 steps
- 学习率: $2 \times 10^{-6}$ constant
- Max generation length: 32k tokens
- Sampling temperature: 1.0 (高温度促进exploration)

**GPU hours对比** (这是paper最striking的数字):
- Math: 2172.8h, Coding: 3187.2h, Science: 787.2h, IF: 377.6h → separate总和: **6524.8h**
- Multi-Task: **2166.4h** → 只占 **33.2%**
- MT-OPD额外: 816h

---

## 3. 核心实验结果

### 3.1 主表解读 (Table 3)

9个benchmark: AIME'24/AIME'25 (math), LiveCodeBench v5/v6 (code), HLE/GPQA-Diamond (science), IFEval/IFBench (IF), MMLU-Redux (general)。

关键观察:

**(1) Single-domain expert的cross-domain transfer现象非常明显**
- RL-Math在AIME'24得71.51, 但RL-Coding/Science/IF也分别得60.78/63.65/64.06 → Math RL训练竟提升了coding/science/IF能力
- RL-Science在GPQA-Diamond 56.19, 但RL-Math在GPQA-Diamond上得56.82甚至更高 → science benchmark实际需要更多logical reasoning而非science knowledge

**(2) Multi-task性能可比merging, 但GPU消耗仅33.2%**
- RL-Multi平均score与Model Merging非常接近
- 多个benchmark (AIME'24 73.85, LCB v5 59.77, IFEval 90.34) 上Multi-task甚至更好

**(3) Verification against official Qwen3-4B (Thinking mode)** (Table 8)
本文开源复现版本在AIME'24 (73.85 vs 73.80) 持平, 在LCB v5/v6 (59.77/56.57 vs 54.20/46.86) 大幅超越, 在IFEval (90.34 vs 81.90) 大幅超越, 但在MMLU-Redux (80.00 vs 83.70) 略低。

### 3.2 Model merging方法对比 (Table 4)

7种方法比较: Average, SCE (https://arxiv.org/abs/2508.05971), Ties (https://arxiv.org/abs/2306.01708), Ties+DARE (https://arxiv.org/abs/2402.02713), TA (https://arxiv.org/abs/2212.04089), TA+DARE, MT-OPD (https://arxiv.org/abs/2306.08543)。

平均分: Ties 59.41 ≈ MT-OPD 59.26 > SCE 59.00 > TA+DARE 58.91 > Ties+DARE 58.87 > TA 57.85 > Average 55.86

关键insight: 不同merging方法存在 **seesaw effect** —— 在某些benchmark好的方法在其他变差, 需要按平均分选best method。这里Ties最佳。

---

## 4. Weight Shift Footprint分析 (Section 3.3)

这是paper最有意思的部分之一, 给出了cross-domain synergy的几何解释。

### 4.1 Weight changed mask

定义一个weight $w \in \mathbb{R}$ 是"changed"的条件:
$$|w_{RL} - w_{SFT}| > \eta \cdot \max(|w_{RL}|, |w_{SFT}|), \quad \eta = 10^{-3}$$

变量含义:
- $w_{RL}$: RL后该参数值
- $w_{SFT}$: SFT后该参数值 (anchor)
- $\eta$: 相对阈值, 用max归一化避免对不同magnitude参数的不公平
- $10^{-3}$ 阈值是为bfloat16精度噪声选的, 太小会被浮点误差污染

得到mask $M_{RL} \in \{0,1\}^d$, $d$是参数维度。

### 4.2 Jaccard overlap

$$J(RL_1, RL_2) = \frac{|M_{RL_1} \wedge M_{RL_2}|}{|M_{RL_1} \vee M_{RL_2}|}$$

- 分子: 两个mask都是1的位置数 (intersection)
- 分母: 两个mask至少一个是1的位置数 (union)
- 范围 [0,1], 越大表示更新位置overlap越多

Table 5的关键数据:
- math-coding: ~0.45
- math-science: ~0.48
- random reference: **0.18**
- 所有domain对的overlap远超random, 说明RL更新的weight位置在不同domain间有强烈overlap

但需要更仔细思考: 30%的weights都被更新, 所以random应该期望 $0.3 \times 0.3 / (0.3 + 0.3 - 0.3 \times 0.3) = 0.13$。Table中random baseline是0.18, 说明他们用了一个稍微高一些的mask率。无论如何, 实测0.45+远高于random, 这是domain间共享更新subspace的强证据。

### 4.3 Cosine similarity on overlapping regions

只看overlap位置还不够, 要看更新方向的alignment。直接在高维算cosine similarity有 **curse of dimensionality** 问题 (高维下随机向量也接近正交), 所以用 **orthogonal random projection (LSH)** 降到256维再算。

Figure 3显示: 跨domain的cosine similarity保持positive (modest level), reasoning domains (math/coding/science)之间相似度更高, 与IF的相似度稍低。

**Intuition**: 不同domain的RLVR虽然更新不同的"知识", 但都倾向于在共享的reasoning substrate上做相似的directional调整。这与"reasoning是transferable capability"的intuition一致。

---

## 5. Policy Neighborhood Transfer (Section 3.4) - 最核心机制

### 5.1 KL divergence的反直觉发现

之前Shenfeld et al. (https://arxiv.org/abs/2509.04259) 的工作认为SFT/RL造成的性能退化与base model的KL divergence相关。但本文在multi-domain场景下发现: **KL divergence和性能变化没有显著correlation**。

Figure 4的cross-comparison矩阵:
- 行(y): expert model domain
- 列(x): sampled trajectories的domain
- Cell value: KL divergence
- ΔPerf: multi-domain模型相对domain expert的性能变化

举例: 在math domain上评估时, math expert与multi-task model的KL最小 (符合直觉), 但**coding expert也显示较低KL**, 同时multi-domain模型在math上还获得了 +0.8 的性能提升。

### 5.2 Policy neighborhood的形式化定义

给定domain A及其expert $E_A$, domain B是A的policy neighborhood当:
$$\mathbb{E}_{x \sim A, \hat{y} \sim \pi_{E_B}(\cdot|x)} \left[ \log \frac{\pi_{E_B}(\hat{y}|x)}{\pi_{multi}(\hat{y}|x)} \right] < \varepsilon$$

变量含义:
- $x \sim A$: 从domain A的prompt分布采样
- $\hat{y} \sim \pi_{E_B}(\cdot|x)$: 用B的expert生成response
- $\pi_{E_B}(\hat{y}|x)$: B expert给response的概率
- $\pi_{multi}(\hat{y}|x)$: 多domain合并模型的概率
- 整体是: 在A的input分布上, 用B expert生成, 测B expert与multi模型的KL
- $\varepsilon$是threshold, 通过与$\text{KL}(\pi_{E_A} \| \pi_{multi})$比较确定

这个公式精妙之处: neighborhood不是input相似度定义的, 而是**output policy相似度**在另一个domain的input分布上测度。它捕捉了"B expert在A domain上给出的判断, 与multi模型有多接近"。

### 5.3 Neighborhood关系的不对称性

Table 6的ablation:
- Math + Coding (math domain): AIME24 71.51 → 72.14 (+0.63), AIME25 63.54 → 66.20 (+2.66)
- Coding + Math (coding domain): LCB v5 59.40 → 57.97 (-1.43), LCB v6 55.43 → 52.00 (-3.43)

**Coding是math的neighbor, 但math不是coding的neighbor** → 这种asymmetry来自domain的"覆盖关系": coding expert能合理处理math problem, 但math expert缺乏coding所需的执行/语法knowledge。

**Intuition**: 这种asymmetric neighborhood解释了为什么merged multi-domain模型在某些domain能增益而另一些domain变差。这也给了model merging recipe的指导: 选择与目标domain构成neighbor关系的expert进行merge, 可能比"全merge"更优。

---

## 6. Gain Consistency Analysis (Section 3.5) - emergent capability的证据

### 6.1 Gain vector定义

对每个task $t$ 和model $m$:
$$g_m^t = \left( \max(a_m^t(1) - a_{sft}^t(1), 0), \ldots, \max(a_m^t(n_t) - a_{sft}^t(n_t), 0) \right)$$

变量含义:
- $a_m^t(i)$: model $m$ 在task $t$ 第 $i$ 个样本的accuracy (这里应该是Avg@K多次采样的成功率)
- $a_{sft}^t(i)$: SFT baseline在该样本的accuracy
- $n_t$: task $t$ 的test set大小
- max(·, 0): 只保留positive gain, 排除capability退化

Union gain vector (单任务模型集体技能代理):
$$g_{union}^t = \max(g_{math}^t, g_{science}^t, g_{coding}^t, g_{IF}^t)$$

即对每个样本取所有single-task模型中的最大gain。

### 6.2 Consistency score

通过计算$g_{union}^t$与RL-Multi / Ties-Merging / MT-OPD各自gain vector的cosine similarity得到。

Figure 5的关键观察:
- Math的consistency最高 → 数学技能homogeneous, 抵抗inter-task interference
- RL-Multi的consistency **最低** (尤其IFBench、AIME'24、LCB-v5)
- Ties-Merging和MT-OPD的consistency高于RL-Multi

**重要implication**: 
- Model merging主要是**继承**single-task模型的original capabilities
- Multi-task training学到了**与single-task divergent的能力** → emergent capabilities存在, 来自任务间的mutually promotion
- 既然multi-task性能不一定优于single-task, 也暗示inter-task interference同时存在

这与DeepSeek-R1报告中"multi-task RLVR激活了single-task中没有的emergent reasoning"的claim形成共鸣 (https://arxiv.org/abs/2501.12948)。

---

## 7. Self-Discrimination与Verification (Section 3.6) - 最有思想深度的部分

### 7.1 实验setup

把RL-trained模型当作 **Generative Reward Model (Gen-RM)**, 在自己生成的trajectories上做verification:
- **Outcome-based verification**: 只看最终答案 (System 1 intuition)
- **Process-based verification**: 看完整CoT (System 2 reasoning)

Table 7数据格式: Gen = generation accuracy, Judge (Out/Proc) = outcome/process judge accuracy。

### 7.2 四个核心Finding

**Finding 1: Verification horizon取决于error locus**

- **Logic-intensive tasks (Math/Code)**: SFT baseline上process verification显著优于outcome verification → final answer是derivation process的"lossy compression", 仅看outcome相当于"black-box guess", process verification提供"white-box debugging"。
- **Constraint-intensive tasks (IF)**: outcome verification反而更好 → IF error在execution surface (JSON syntax, keyword), reasoning trace反映intention ("I plan to output JSON..."), 实际execution才反映failure。Process verification会产生"halo effect", 被正确plan误导而忽视execution failure。

**Finding 2: RL induces modality specialization**

- **IF RL**: sharpen outcome-based judgment (System 1), process保持低 → 因为IF数据强调surface constraint
- **Code/Math RL**: 强化process-based verification → 因为logic domain需要step-by-step作为deterministic anchor

这是**RLVR自然induces self-discrimination能力的emergent现象**, 即使没有显式reward modeling supervision。

**Finding 3: Multi-task RL fuses cross-domain advantages**

Figure 7b显示Multi-task RL同时:
- 继承IF RL的outcome sensitivity (System 1 直觉)
- 保持math/code RL的process-based verification (System 2 逻辑)
- 实现metacognitive alignment: 既critique outcome (intuition) 又critique trajectory (logic), 允许inference时self-correction

Table 7数据: Multi-task RL的Avg Judge (Out/Proc) = 77.5/64.7, 是所有方法中最高的outcome judge (77.5), 且process保持64.7 → 完美的cross-domain synergy。

**Finding 4: MT-OPD improves process verification**

假设: multi-teacher OPD让student模型暴露于diverse reasoning traces和error patterns, 这broader supervision能sharpen step-level consistency checks, 减少对surface plausibility的over-reliance。

Table 7: MT-OPD在AIME24 process judge = 75.0 (高于multi-task RL 76.9?), LCB v5 process judge = 81.0 (最高)

---

## 8. 综合Intuition与Implications

### 8.1 为什么mixed multi-task没出现显著interference?

传统multi-task learning (如Yu et al. https://arxiv.org/abs/2510.19178) 担心gradient interference, 但RLVR场景特殊:

1. **Verifiable reward的稀疏性**: 每个sample的gradient信号是group-relative的, 不同domain的gradient在参数空间不一定冲突
2. **Reasoning capability的共享substrate**: math/coding/science共享underlying的logical reasoning + planning能力, 所以更新方向有alignment (cosine similarity positive的evidence)
3. **Policy neighborhood transfer**: 一个domain的RL更新会把policy推向"邻域"domain的optimal policy方向

### 8.2 Multi-task与merging的本质差异

- **Model merging = 加权几何组合**: 在weight space上组合, 保留各expert的capability分布, 但无法发现expert都没学过的capability
- **Multi-task training = 动态梯度耦合**: gradient信号在训练时interact, 可以emerge出single-task training无法到达的regions

这与最近model soups (https://arxiv.org/abs/2203.05482) 和task arithmetic (https://arxiv.org/abs/2212.04089) 的linear mode connectivity理论一致: 权重空间存在linear path时merging有效, 但有些emergent capability需要走non-linear trajectory。

### 8.3 实用recipe建议

基于这篇paper, 实践建议:
1. **首选mixed multi-task RLVR**: 性能可比, GPU消耗1/3, 还可能有emergent synergy
2. **若必须用separate training (e.g. 不同团队并行)**: 用Ties-merging > MT-OPD > 其他
3. **Model merging时要选neighbor**: 不一定要merge所有expert, 只merge有policy neighborhood关系的
4. **Verification horizon要match task**: logic task用process verification, constraint task用outcome verification
5. **Self-RM是免费bonus**: RLVR训练后模型自带verification能力, 无需额外reward modeling

### 8.4 与其他recent work的联系

- **Shenfeld et al. RL's Razor** (https://arxiv.org/abs/2509.04259): 提出online RL forgets less due to KL constraint, 但他们的KL-performance关系在multi-domain场景被这篇paper部分推翻/细化
- **DAPO** (https://arxiv.org/abs/2503.14476): 这篇paper的math RL数据来源, GRPO改进
- **Wen et al.** (https://arxiv.org/abs/2506.14245): 提出RLVR implicitly incentivizes correct reasoning, 与本篇的self-discrimination发现互相印证
- **Wu et al. Imbalanced gradients** (https://arxiv.org/abs/2510.19178): 提出multi-task RL post-training的gradient imbalance问题, 本篇发现这种imbalance在RLVR场景下not critical

---

## 9. 我的批判性思考

### 9.1 论文的strengths
1. **Systematic comparison**: 在控制良好的实验设置下做head-to-head对比, 这在工业界report中很缺乏
2. **Multi-perspective analysis**: weight geometry + policy KL + capability overlap + verification horizon, 四个独立angle互相印证
3. **Practical impact**: 33.2% GPU hours的节省对大规模训练极其重要

### 9.2 潜在limitations
1. **只测4B scale**: 在70B+/MoE规模上是否还成立? reasoning substrate的sharing可能在更大model上更显著
2. **SFT→RL简化pipeline**: 真实post-training是多轮alternating SFT-RL, 这里的结论可能不全transfer
3. **Nemotron数据可能特殊**: 它的SFT数据blend本身已经multi-domain, 也许在这种混合SFT上的RLVR才没interference
4. **Domain选取偏向reasoning**: 都是verifiable reward domain, 如果加入creative writing、safety等subjective reward domain, interference可能显现
5. **η=1e-3的threshold**: 在bfloat16下这个阈值合理, 但mask比例30%是否对其他model architecture一致?

### 9.3 Future directions
1. **Adaptive routing RL**: 基于policy neighborhood关系动态分配gradient权重, 可能更高效
2. **Asymmetric merge recipes**: 既然neighborhood不对称, 可以设计非均匀merge系数
3. **Cross-domain distillation from neighborhood**: 用neighbor expert distill而非all expert
4. **Self-distillation loop**: 用Multi-task model的self-discrimination能力当reward model, 进一步RL self-improve

---

## 10. 关键take-aways

如果让我用几句话总结这篇paper给Karpathy:

1. **Mixed multi-task RLVR在RLVR场景下没出现传统multi-task learning的interference**, 反而有reasoning domain间的synergy, 只需1/3 GPU hours
2. **Mechanism是policy neighborhood transfer**: 一个domain的RL更新会把policy推得靠近neighbor domain的optimal, 这种transfer是asymmetric的
3. **Multi-task training产生emergent capability**, 而model merging只是inherit single-task capability
4. **RLVR自然induce self-discrimination能力**, 表现为System 1 (outcome verification, 来自IF RL) 和System 2 (process verification, 来自math/code RL) 在multi-task training下synergy
5. **Verification horizon要match task error locus**: logic task用process, constraint task用outcome

这对designing multi-domain post-training pipeline有直接指导意义: **default to mixed multi-task RLVR, reserve model merging for parallel-team development scenarios**。

---

**相关references:**
- Paper GitHub: https://github.com/mosAI25/M2RL
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Qwen3: https://arxiv.org/abs/2505.09388
- GLM-4.5: https://arxiv.org/abs/2508.06471
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- DAPO: https://arxiv.org/abs/2503.14476
- Nemotron 3 Nano: https://arxiv.org/abs/2512.20848
- Ties-Merging: https://arxiv.org/abs/2306.01708
- Task Arithmetic: https://arxiv.org/abs/2212.04089
- Model Soups: https://arxiv.org/abs/2203.05482
- DARE: https://arxiv.org/abs/2402.02713
- Multi-teacher OPD: https://arxiv.org/abs/2306.08543
- RL's Razor: https://arxiv.org/abs/2509.04259
- Imbalanced gradients in RL post-training: https://arxiv.org/abs/2510.19178
- RLVR implicitly incentivizes correct reasoning: https://arxiv.org/abs/2506.14245
