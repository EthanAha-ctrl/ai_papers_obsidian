---
source_pdf: Self-Distilled Reasoner.pdf
paper_sha256: 9883fbd6daf7ee219ca7e36807a55ad17111dd71352de7adc3778fe7b654290f
processed_at: '2026-08-12T04:39:21-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 OPSD

## 1. 一句话概括

**让同一个模型,在"看过答案"和"没看过答案"两种状态下,对自己做题时的每一步进行"事后复盘",然后把复盘得到的那一点点"啊原来应该这样"的信号,变成训练梯度,推着模型往"看过答案的自己"靠拢。**

这就是整个 paper 的核心,剩下的都是工程细节。

---

## 2. 为什么要这么做 — 从人的学习说起

想象你在学数学。一道积分题做错了,你会怎么做?

**做法 A(STaR)**:重新做 100 遍,做对的那几次留下来背下来。问题是做错的 99 次**完全浪费**,而且背下来的只是"这道题这么答",没学到"为什么这么想"。

**做法 B(GRPO)**:做 8 遍,跟同学比对一下,全对就给 +1,全错就给 -1,部分对就按比例。但具体哪一步错了、应该怎么改,你不知道,只知道"这题做对了/做错了"。而且如果 8 遍全对或全错,就完全没有学习信号了。

**做法 C(SFT)**:直接背标准答案。问题是标准答案经常写得很简洁,你背下来之后只会写"因为 A 所以 B",但实际考试时一紧张就忘了中间的 reasoning。

**做法 D(OPSD)**:你做一遍题(可能做错),然后**翻开答案,边看答案边对照自己刚才写的每一步**,心想"如果我当时就知道答案是 $\ln 2$,我在写第三步的时候应该会想到换元"。然后把这种"事后诸葛亮"的领悟内化,下次做题时第三步就会自然想到换元。

这第 D 种就是 OPSD。它背后有个心理学事实:**理解一个解法比凭空生成解法容易**(evaluation is easier than generation,Naor 1996)。你看答案能看懂,不代表你能自己想出来——但你能看懂,就说明你的模型有"判断对错"的能力,只是缺一个"引导"。

---

## 3. 算法最朴素的描述

训练时,同一个 LLM 玩两个角色:

- **Student 角色**:只看到题目,正常做题,写下 $\hat{y}$ 这条 trajectory。
- **Teacher 角色**:看到题目 + 标准答案 $y^\star$,然后**看着 student 写的每一个 token**,在每个位置上重新算一下"如果是我、而且我知道答案,我在这个位置上会给每个 candidate token 多少概率"。

然后让 student 在每个位置上,把自己的 next-token 分布往 teacher 的分布上靠(forward KL)。

梯度只回传到 student,teacher 那边 freeze。

就这么简单。

---

## 4. 三个关键设计决策,每个都有 intuition

### 4.1 为什么 teacher 不真的生成 token?

因为 teacher 要在每个 position 上给一个**完整的 next-token 分布**,而不是只给一个 token。如果让 teacher autoregressive 地生成,它只会生成一条 trajectory,student 只能学这一条。但 forward KL 要求 student 覆盖 teacher 所有 high-mass 的 next token——也就是要学 teacher 觉得"可能也对"的所有分支。所以 teacher 必须用 **forward pass 隐式 rationalize**,在每个位置直接输出 logits,而不是 decode。

这也是为什么叫 "self-distillation"——teacher 不输出任何东西,它的"知识"完全通过分布形状传递。

### 4.2 为什么用 forward KL 而不是 reverse KL?

Forward KL 是 "我要覆盖 teacher 所有可能的 mode",reverse KL 是 "我只要抓住 teacher 最强的那个 mode"。

在 reasoning 任务里,teacher 看过答案之后,它觉得"接下来这几个 token 都有可能对"——比如 "$x = $" 后面可能是 "2"、"3"、"4" 中的某一个,取决于具体 reasoning 路径。Forward KL 强迫 student 把这几个分支都学到,这样 student 在 test-time 才能灵活选择。Reverse KL 会让 student 把概率全堆在 teacher 最高那个 mode 上,失去多样性。

Table 3 数据:forward KL +7.2,reverse KL 几乎不动,JSD 略降。完全符合直觉。

### 4.3 为什么 teacher 要用 initial policy $\theta_0$ 而不是 current $\theta_t$?

如果 teacher 也跟着 student 一起更新,会出现一个诡异的现象:student 越学越像 teacher,但 teacher 也在变,两者的差距始终维持在某个水平,失去 anchor。用 $\theta_0$ 当 teacher 等于给训练一个 **fixed reference point**,类似 GRPO 里那个 KL-to-reference 的 regularization,防止 student 漂得太远。

实验上 100 步就收敛了,所以 teacher 也不会过时太久。

---

## 5. Per-Token Clipping — 一个非常实用的发现

作者做了一个 ablation:把 vocabulary 里的 token 分成 style 类("maybe", "therefore", "okay", "first", "next"...) 和 math 类("logarithm", "inequality", "exponent"...)。然后量每一类 token 上的 KL 散度。

**惊人发现**:style token 的 KL 比 math token 高 5-8 倍!

为什么?因为 style token 的分布对 context 高度敏感。Teacher 多看了一个答案,它对 "好,接下来用 maybe" 还是 "好,接下来用 actually" 的偏好可能完全变了。但 math token 的分布主要取决于数学内容本身,teacher 看了答案反而不会大改——因为数学是 math,答案告诉你的是结论,不是推导风格。

如果不 clip,student 学到的全是"如何说话"而不是"如何推理"。所以作者对每个 vocab entry 的 KL 贡献设一个上限 $\tau$,超过的部分 clip 掉,让训练信号不被 style token 主导。

Figure 4 显示:不 clip 的话 60 步就崩,clip 之后稳定上升。这是个很值得复用的 insight。

---

## 6. TM-off Student × TM-on Teacher — 一个反直觉但很妙的选择

Qwen3 有 thinking mode。直觉上你可能觉得 student 和 teacher 都该用同一个 mode,保持公平。

但作者发现 **student 关 thinking mode + teacher 开 thinking mode** 效果最好。

为什么?Teacher 在 thinking mode 下,看到答案后会做 explicit rationalization:"啊原来这题要用对数展开,那么我应该..." 这些 thinking token 里的 math content 浓度很高。Student 在 TM-off 下直接写答案,没有 thinking,所以两者的分布在 math token 上 gap 最大——这个 gap 恰恰就是 "需要 think 才能得到、student 现在还不会" 的部分,正是我们想 distill 的。

如果 student 也开 thinking mode,student 自己也在 think,跟 teacher 的 gap 反而小了,学习信号变弱。

---

## 7. 为什么 OPSD 比 GRPO 高效 128 倍 token,还能赢?

GRPO 训练 1 个 prompt 要:8 个 rollout × 16k token = 128k token,得到 8 个 0/1 reward,然后 group-normalize 算 advantage。

OPSD 训练 1 个 prompt 只要:1 个 rollout × 1024 token = 1024 token,在每个 token 上都有一个 KL loss 信号。

token 比 = 128:1。信号密度比 = 1024 个 KL vs 8 个 binary reward,差距更大。

更致命的是 GRPO 在 OpenThoughts 这种数据上有个 failure mode:这个数据集题目难度集中在某个范围,model 一开始做 8 遍要么全对要么全错,group std=0,advantage=0,梯度消失。Paper 里 Figure 3 右图显示 100 步之后超过一半 batch 都是这样,等于烧算力但什么都没学。

OPSD 永远不会陷入这个 trap——teacher 看了答案之后总能给出有意义的 next-token distribution,即使 student 整条 trajectory 全错,每个 position 上还是能学到"你在这个 token 上应该往这个方向偏"。

---

## 8. 为什么 SFT 反而变差了?

Table 2 里 SFT 在 Qwen3-1.7B 上把 AIME24 从 51.5 拉到 48.4,**变差了**。

原因是 OpenThoughts 的 reference solution 写得特别简洁,像标准答案一样直接给结论。SFT 学到的是"写简洁答案",inference 时模型就开始偷懒,reasoning chain 变短,test-time 的自我反思消失。

OPSD 不学 token 本身,学的是"看过答案的自己对没看过答案的自己"的分布差。这个分布差里**包含**了简洁答案背后的 reasoning process(因为 teacher 要在 thinking mode 里 rationalize 出来),所以 student 内化的是 reasoning 而不是 surface form。

这是个挺深刻的对比:**同样是学同一份数据,你学"答案长什么样"vs 学"知道答案的人会怎么想",效果天差地别**。

---

## 9. Generation Length 为什么不影响结果?

Figure 5:1024 token vs 4096 token,几乎一样。

直觉:开头几个 token 是"关键岔路口"——student 在这里走对了,后面一路顺畅;走错了,后面越写越偏。Teacher 在这些早期 token 上跟 student 的 gap 最大,因为这是"决定 reasoning 方向"的时刻。

后期 token 给定前缀之后,teacher 和 student 的预测趋同,因为前缀已经决定了路径,剩下就是顺着算。所以多生成 3 倍 token,大部分是无效信号。

这跟 LIMO(Less is More)的哲学一致——reasoning 的精华在早期 branching point,不在 length。

---

## 10. 跟其他 self-training 方法的本质区别

| 方法 | 信号粒度 | 是否 on-policy | 是否需要 external teacher | 是否用答案 |
|------|---------|--------------|------------------------|----------|
| STaR | sequence-level(0/1) | 否,filter 后 SFT | 否,但需要 reward | 是,filter 用 |
| ReST | sequence-level(reward) | 半 on-policy | 否,但需要 reward | 是 |
| GRPO | sequence-level(0/1) | 是 | 否,但需要 reward | 是 |
| GKD | token-level(distribution) | 是 | **是**,要 external teacher | 否 |
| Context Distill | token-level(hard) | 否 | 否,自己当 teacher | 是 |
| **OPSD** | **token-level(distribution)** | **是** | **否,自己当 teacher** | **是** |

OPSD 是唯一一个**同时拿到所有好处**的:细粒度信号 + on-policy + 不需要外部 teacher + 利用 ground-truth。

它的核心 trick 是用 **context asymmetry**(给不给看答案)来制造 teacher-student gap,而不是用 model capacity gap(大模型教小模型)。这是个**很省资源**的思路——你不需要 GPT-4 来教你的 7B 模型,你的 7B 模型看了答案就能当自己的 GPT-4。

---

## 11. 我能想到的几个延伸联想

**1. 这本质上是一种 "hindsight distillation"**

跟 robotics 里的 Hindsight Experience Replay(HER)神似。HER 的思路是:你打篮球没进,但你可以假装"我本来就想扔到这个位置",于是失败动作变成成功数据。OPSD 是:你做题错了,但 teacher 看了答案之后,在你错的每一步上都能告诉你"如果当时知道答案,这里应该想什么",于是失败 trajectory 变成 dense supervision。

两者都是用 **hindsight 信息** 把 sparse reward 变成 dense reward。HER 改 goal,OPSD 加 context。

**2. 这是 implicit Process Reward Model**

Process Reward Model(PRM)需要人工标每一步对错,极贵。OPSD 里 teacher 看了答案后的 logit 分布,本身就是一个免费的 per-step reward——它告诉你"在这个位置,知道答案的我觉得什么 token 更对"。可以用来替代 PRM 做 tree search、best-of-n 之类的推理时算法。

**3. 可以扩展到其他 privileged context**

不止 ground-truth answer。比如:
- 代码题:teacher 的 privileged context 是 code execution trace
- 多步推理:teacher 的 privileged context 是中间结论的 verification
- Agent 任务:teacher 的 privileged context 是 environment 的 final state

这就是 concurrent work SDPO (Hubotter 2026) 在探索的方向。

**4. 模型规模够大之后,OPSD 可能成为 RL 的替代品**

DeepSeek-R1 用海量 GRPO 把 reasoning 训出来,token 消耗惊人。如果 OPSD 的 scaling law 跟 RL 类似但 token 效率高 100 倍,那可能未来 reasoning post-training 的主流就是 self-distillation 而不是 RL。

**5. Curriculum 的重要性被低估了**

Paper 的 Appendix A 自己说:题太难时 teacher 看了答案也 rationalize 不出来(就像你看一道你没学过的高深数学题的答案,看完还是不懂)。所以 OPSD 的 sweet spot 是"略高于模型能力"的题目,这暗示需要一个 adaptive curriculum,把题目维持在模型的"frontier difficulty"。

我猜这也是为什么 paper 只跑 100 步就停——可能再跑下去就遇到"题目都被学完了、剩下的太难 teacher 也救不了"的瓶颈。

**6. Self-referential training 的哲学意味**

这其实是我在读 paper 时最着迷的一点。模型在做的是:**"让自己变得更像那个信息更完备的自己"**。

这个结构在人类认知里很常见——你看完答案之后,重新审视自己刚才的思考,意识到"我当时要是注意到这个条件,就会用对数展开"。这种"事后诸葛亮"不是没用的,它正是 metacognition 的核心,也是 learning 的本质。

OPSD 把这个内化了。它训练的不是"知道答案",而是**"在没有答案时,也能像有答案时那样思考"**。这是一种把 hindsight foresight 化的训练,挺哲学的。

---

## 12. 一句话再总结一次

**OPSD 让模型在每个训练 step 上做一次"开卷对答案、然后闭卷重做"的练习,把"开卷时的自己"作为"闭卷时的自己"的 soft target,用 forward KL 在自己生成的每个 token 上对齐——这等价于一个 dense per-token reward 的 policy gradient,但不需要 external teacher、不需要 reward model、不需要 rejection sampling,token 效率比 GRPO 高 128 倍,还避开了 SFT 的 exposure bias 和 length collapse。**

核心 insight 就一句:**理解比生成容易,所以"看过答案的自己"可以教"没看过答案的自己"**。

---

## References

- OPSD repo: https://github.com/siyan-zhao/OPSD
- GKD: https://arxiv.org/abs/2306.13649
- Thinking Machines on-policy distillation: https://thinkingmachines.ai/blog/on-policy-distillation
- STaR: https://arxiv.org/abs/2203.14465
- GRPO / DeepSeek-Math: https://arxiv.org/abs/2402.03300
- Context Distillation (Snell 2022): https://arxiv.org/abs/2209.15189
- HER (Hindsight Experience Replay, Andrychowicz 2017): https://arxiv.org/abs/1707.01495
- Naor 1996 (evaluation easier than generation): https://dl.acm.org/doi/10.1145/237814.237821
- LIMO (less is more for reasoning): https://arxiv.org/abs/2502.03387
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Qwen3 technical report: https://arxiv.org/abs/2505.09388
- OpenThoughts dataset: https://arxiv.org/abs/2506.04178
- Let's Verify Step by Step (PRM): https://arxiv.org/abs/2305.20050
- SDPO (concurrent self-distillation RL): https://arxiv.org/abs/2601.20802

---

# On-Policy Self-Distillation (OPSD) — 深度解析

## 1. Core Intuition: "Rationalization 比 Generation 容易"

这篇 paper 的核心 insight 来自一个朴素的认知科学类比：当一个学生做错题后，他不会无限试错，而是去看标准答案 $y^\star$、理解每一步、然后反推为什么这个解法 work。也就是说 **rationalization（解释/合理化）比 generation（凭空生成）更容易**。这在 LLM 上的对应事实是 "evaluation is easier than generation"（Naor, 1996）。

OPSD 的精妙之处在于：**它把同一个 LLM 拆成两个 persona**——一个是看过答案 $y^\star$ 的 "teacher"，一个是只看到题目的 "student"，两个 persona 共享同一组参数 $\theta$，区别只在 conditioning context。这避免了传统 on-policy distillation 需要一个更大 external teacher 的负担。

---

## 2. Method 详解

### 2.1 Teacher / Student 两套 conditionals

设原始 LLM 的概率分布为 $p_\theta$。定义两个 conditional：

**Teacher policy**（看到 privileged info）：
$$p_T(\cdot \mid x, y^\star) \triangleq p_\theta(\cdot \mid x, y^\star)$$

**Student policy**（只看 question，与 inference-time condition 一致）：
$$p_S(\cdot \mid x) \triangleq p_\theta(\cdot \mid x)$$

变量说明：
- $x$ — 问题（problem statement）
- $y^\star$ — ground-truth reference solution（包含 chain-of-thought）
- $p_\theta$ — shared LLM
- $p_T, p_S$ — 通过不同 prompt template（Figure 2）实例化出来的两个角色

Teacher 的 prompt 是：
> "Problem: ... Here is a reference solution: ... After understanding the reference solution, please try to solve this problem using your own approach below:"

注意 **teacher 不真的 decode tokens**，它只是通过 prefill / 一次 forward pass 在 internal state 中"rationalize"，然后给 student 的 rollout 上每一步打一个 logit distribution。

### 2.2 On-Policy Rollout

Student 在自己当前 policy 下采样一条 trajectory：
$$\hat{y} = (\hat{y}_1, \dots, \hat{y}_{|\hat{y}|}) \sim p_S(\cdot \mid x)$$

其中 $\hat{y}_{<n} \triangleq (\hat{y}_1, \dots, \hat{y}_{n-1})$ 是 student 已经生成的前缀。

然后在每个 position $n$ 上，两个 policy 都对 next token $y_n \in \mathcal{V}$ 给一个 distribution：
$$p_S(y_n \mid x, \hat{y}_{<n}), \quad p_T(y_n \mid x, y^\star, \hat{y}_{<n})$$

关键：两个 distribution 共享同一个 prefix $\hat{y}_{<n}$（student 自己生成的），区别只在前缀的初始 conditioning（teacher 多了 $y^\star$）。

### 2.3 训练目标

核心 loss（Equation 1 / 6 / 8）：

$$\mathcal{L}_{\text{OPSD}}(\theta) = \mathbb{E}_{(x, y^\star) \sim S}\, \mathbb{E}_{\hat{y} \sim p_S(\cdot \mid x)} \sum_{n=1}^{|\hat{y}|} D\Big(p_T(\cdot \mid x, y^\star, \hat{y}_{<n}) \,\Big\|\, p_S(\cdot \mid x, \hat{y}_{<n})\Big)$$

变量含义：
- $\mathcal{S}$ — 训练数据集 $\{(x_i, y_i^\star)\}_{i=1}^N$
- 外层 $\mathbb{E}$ — 在数据集上取平均
- 内层 $\mathbb{E}$ — 在 student 自己的 rollout 上取期望（on-policy）
- $\sum_{n=1}^{|\hat{y}|}$ — 在 trajectory 上每个 token 位置求和
- $D(\cdot \| \cdot)$ — 任意 divergence（paper 实验发现 **forward KL** 最好）

梯度只通过 student logits 回传，teacher 是 fixed target（基于 initial policy $\theta_0$，而不是不断更新的 $\theta_t$，这一招起到 implicit regularization 防止 over-deviation 的作用，类似 GRPO 里的 reference policy KL penalty）。

### 2.4 为什么用 Forward KL？

Forward KL: $D_{KL}(p_T \| p_S) = \sum_v p_T(v) \log \frac{p_T(v)}{p_S(v)}$

Forward KL 是 **mean-seeking**（student 要 cover teacher 所有 high-mass 区域），它对 $p_T(v) > 0$ 而 $p_S(v) \to 0$ 的位置给无穷大惩罚，这强迫 student 至少要 teacher 的所有 mode 上都有非零概率。

Reverse KL 是 mode-seeking，JSD 是某种折中。在 reasoning 任务里，teacher 看了答案之后分布很集中（解法路径相对确定），mean-seeking 行为反而让 student 把所有合理的 next-token branch 都学一遍，generalization 更好。Table 3 数据印证：Forward KL 把 AIME25 从 36.7 拉到 43.9，reverse KL 几乎没变化（37.5），JSD 略降（36.9）。

---

## 3. Per-Token Pointwise KL Clipping

这是 paper 里很务实的一个 trick。作者发现（Table 5）token-level KL 在 vocabulary 上是 **高度 skewed** 的：stylistic tokens（"maybe", "okay", "therefore", "however", "first", "second" 这类连接词）的 KL 比 math tokens（"exponential", "logarithm", "inequality" 这类）高一个数量级。

如果不 clip，loss 会被 stylistic tokens 完全主导，student 学到的只是模仿 teacher 的"说话风格"而不是"推理内容"。

定义 per-position per-vocab entry 的 f-divergence 分量：
$$\ell_{n,v}^{(f)} = p_T(v \mid \cdot) \, f\!\left(\frac{p_S(v \mid \cdot)}{p_T(v \mid \cdot)}\right)$$

其中 $f$ 是 f-divergence 的生成函数（KL 时 $f(t) = t \log t$；正向 KL 这里其实写的是 $p_T f(p_S/p_T)$ 形式，等价于 $\sum_v p_T(v) f(p_S/p_T)$，对 forward KL 就是 $\sum_v p_T \log(p_S/p_T)$，符号 conventions 注意）。

Clipping 版本：
$$D_{\text{clip}}^{(f)}(p_T \| p_S) = \frac{1}{|\hat{y}|} \sum_{n=1}^{|\hat{y}|} \sum_{v \in \mathcal{V}} \min\big(\ell_{n,v}^{(f)}, \tau\big)$$

- $\tau$ — clip 上限，控制单 token 的最大贡献
- 效果（Figure 4）：不 clip 时训练 60 步后崩溃，clip 之后稳定上升

这个 idea 跟 Mixtral、Minillm (Gu et al. 2024) 里的 length-normalized loss 思路有异曲同工之妙——都是意识到 uniform weighting 在 autoregressive distillation 里是 suboptimal 的。

---

## 4. TM-off Student × TM-on Teacher 的最优组合

Qwen3 有 thinking mode on/off 两种 prompt format。Table 5 显示：

| Student | Teacher | Style KL | Math KL | Other KL |
|---------|---------|----------|---------|----------|
| TM-off | TM-off | 0.68 | 0.12 | 0.11 |
| TM-on | TM-off | 0.51 | 0.10 | 0.17 |
| TM-on | TM-on | 0.51 | 0.09 | 0.08 |
| **TM-off** | **TM-on** | **0.85** | **0.14** | **0.25** |

TM-off student（直接回答，不写 thinking）配 TM-on teacher（看了答案之后开启 thinking 模式做 rationalization），math tokens 上的 KL 信号最大。Intuition：teacher 在 thinking mode 下会做 explicit rationalization（"啊原来这个 hint 意味着应该用对数展开..."），而 student 在 TM-off 下不能 think、只能 surface 出结论，所以两者的 distribution 差异恰恰反映了 "需要 think 才能得到但 student 当前还不会的数学推理步骤"——这正是我们想 distill 的部分。

---

## 5. Algorithm 1 全流程

```
Input: dataset S = {(x_i, y_i*)}; LLM p_θ; divergence D (e.g. JSD_β)
1. Instantiate p_S(.|x) and p_T(.|x, y*) as same p_θ under different conditioning
2. while not converged:
3.   Sample minibatch B ⊂ S
4.   for each (x, y*) in B:
5.     Sample on-policy response ŷ ~ p_S(.|x)         # student rollout
6.     Compute per-token divergence:
         ℓ(x, y*) ← D(p_T‖p_S)(ŷ|x) = (1/|ŷ|) Σ_n D(p_T(.|x,y*,ŷ_<n) ‖ p_S(.|x,ŷ_<n))
7.     L_OPSD(θ) = (1/|B|) Σ ℓ(x, y*); update θ (gradient only through student)
```

实操要点：
- LoRA (rank 64, alpha 128) on q/k/v/o + gate/up/down_proj
- Learning rate $5\times10^{-6}$
- Effective batch size 32
- Student generation length 1024 token（4x 增加几乎无效，见 Figure 5，因为 early tokens 是 critical branching points）
- Sampling temperature 1.1
- 训练只跑 100 步就收敛

---

## 6. 实验数据深度分析

### 6.1 Main results (Table 2)

**Qwen3-1.7B**：
- Base: AIME24 51.5 / AIME25 36.7 / HMMT25 23.1 → avg 37.1
- +SFT: 48.4 / 36.3 / 22.7 → 35.8 (退化！)
- +GRPO: 51.1 / 38.3 / 23.7 → 37.7
- +OPSD: **57.2 / 43.9 / 29.2 → 43.4** (6.3 点提升，远超 GRPO)

**Qwen3-8B**：OPSD avg 64.8 vs GRPO 64.0 vs Base 61.8，差距缩小但仍领先。

为什么 SFT 反而变差？因为 OpenThoughts 数据集的 reference solution 风格很 concise，SFT 学到的是"简短回答"，inference 时减少了 reasoning length，但 concise solutions 缺乏 test-time 的 self-reflection 内容。OPSD 把 concise solutions 转化为 dense per-token supervision（通过 teacher rationalization），所以避开了 SFT 的这个 trap。

### 6.2 Token Efficiency (Figure 3)

OPSD 每 prompt 只采 1 个 rollout、每个 rollout 1024 token；
GRPO 每 prompt 8 个 rollout、每个 rollout 16k token。
计算量差 ~128x，但 OPSD 在 100 步内就超过 GRPO 500 步的训练效果。

更深的原因（右图）：GRPO 在 OpenThoughts 这种"answer distribution 偏单峰"的数据上，大约 100 步之后超过一半 batch 的 reward std 为 0（要么全对要么全错），$A_i = 0$ → 梯度消失。OPSD 用 dense per-token KL，即使 student rollout 全错，teacher 看了 $y^\star$ 后仍然能给出有意义的 next-token distribution，loss 信号永不消失。

### 6.3 Generation Length (Figure 5)

1024 vs 4096 token，AIME24/AIME25 上几乎无差异。作者的假设：early tokens 是 critical branching points，late tokens 给定长 prefix 后 teacher 和 student 的 distribution 趋于一致，KL penalty 趋零。这与 Thinking Machines Lab (Lu & Lab, 2025) 的 on-policy distillation 观察一致。

### 6.4 Full-vocab vs Sampled-token (Table 4)

- Full-vocab logit distillation (GKD-style, Agarwal 2024): AIME25 84.1 / HMMT25 60.0
- Sampled-token advantage PG (Lu & Lab 2025-style): 82.1 / 57.3

Full-vocab 多算了 ~$|\mathcal{V}|$ 倍，但每个 position 都监督整个 next-token 分布，让 student 学到"teacher 在所有 plausible next token 上的偏好"，而不只是 sampled token 的 log-prob 差。代价是 peak memory 高（要在所有 position 存 vocab-size logits），有 performance-efficiency tradeoff。

---

## 7. 与 STaR / GRPO / GKD 的关系

### 7.1 vs STaR (Appendix D)

STaR 可以写成 policy gradient with sequence-level reward $\mathbf{1}(y = y^\star)$：
$$\nabla_\theta J_{\text{STaR}}(\theta) = \sum_i \mathbb{E}_{(r,y) \sim p_\theta(\cdot|x_i)}\big[\mathbf{1}(y = y_i^\star) \nabla_\theta \log p_\theta(r,y|x_i)\big]$$

- Sequence-level reward：所有 token 同等 credit
- Filter by correctness：错的 trajectory 完全扔掉，没梯度
- 等于 rejection sampling + SFT on correct ones

OPSD 的 sampled-token variant（Equation 9）：
$$\mathcal{L}(\theta) = -\mathbb{E}\Big[\frac{1}{|\hat{y}|}\sum_n A_n(x,\hat{y}) \log p_S(\hat{y}_n|x,\hat{y}_{<n})\Big]$$

with $A_n = \log p_T(\hat{y}_n|x,y^\star,\hat{y}_{<n}) - \log p_S(\hat{y}_n|x,\hat{y}_{<n})$.

这是 **dense-reward policy gradient**——每个 position 都有 reward $r_n$，即使整条 trajectory 错了，teacher 仍能给每个 position 提供 per-token shaping signal。$A_n$ stop-gradient，所以形式上就是 advantage-weighted policy gradient。

### 7.2 vs GRPO

GRPO 的 advantage：
$$A_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$$

GRPO 的几个问题 OPSD 都解决了：
1. Sparse sequence-level reward vs dense per-token KL
2. 全对全错时 std=0 → 梯度消失 vs OPSD 永远有信号
3. 8 个 rollout × 16k token vs 1 个 rollout × 1024 token，128x 计算节省

### 7.3 vs GKD / On-Policy Distillation

GKD (Agarwal et al., 2024) 和 Thinking Machines on-policy distillation 都用 on-policy student rollouts + teacher per-token distribution matching，但都需要 external teacher。

OPSD = GKD 的 on-policy 版本 + teacher = student 自己（通过 privileged context 实现）。

关键 trick 是 **context-distillation 的 on-policy 扩展**——context distillation (Snell et al., 2022) 用同一模型 + privileged context 生成 trajectory 然后 SFT（off-policy hard target），OPSD 改成 on-policy soft target (per-token distribution matching)。

### 7.4 vs ReST / Self-rewarding LM

ReST (Gulcehre 2023): iterative self-training，generate + filter by reward + SFT on filtered，off-policy。
Self-Rewarding LM (Yuan 2024): model 作为自己的 judge 给 reward。
OPSD 与它们的根本区别是 **soft (distribution-level) + on-policy + privileged context**, 而不是 hard (token-level) + off-policy + reward-based filtering。

---

## 8. 直觉图景

把 OPSD 想成是在 student rollout 的每个 token 位置上做一次 "考试后的批改"：
- Student 把题目看完，自己写一个答案 $\hat{y}$
- Teacher 拿到 $\hat{y}$ + 标准答案 $y^\star$
- Teacher 在每个位置 $n$ 重新计算 "如果当时我在这个位置上、又看了答案，我会怎么分配下一个 token 的概率"
- Student 用 forward KL 把自己向 teacher 靠拢

由于 student 和 teacher 同参数，每次 gradient update 等于在说："在 student 自己走过的状态下，让自己变得更像那个看过答案的自己的状态分布"。

这里有一个非常迷人的 self-referential 结构：模型在 self-improve 时，supervision signal 来自 model 在"信息更完备版本下"对自己的预测。这跟 human metacognition（"我现在再回头看这道题，确实应该这么做"）非常像。

---

## 9. 我会想到的联想与延伸

1. **Privileged information 不止 $y^\star$**：可以扩展到 verifier feedback, code execution trace, proof checker 的 intermediate state, partial credit。Concurrent work SDPO (Hubotter et al. 2026) 已经开始探索用 environment feedback 作为 privileged info。

2. **Curriculum 的重要性**：Appendix A 自己提到，如果题目超过 model 的 comprehension threshold，teacher 即使看了答案也无法 rationalize（"无法理解的答案看完也还是不懂"）。这暗示 OPSD 的 sweet spot 是 "frontier difficulty" 的题目，可以用 adaptive curriculum 持续保持难度边界。

3. **跟 hindsight experience replay (HER) 的相似**：robotics 里 HER 通过把目标改成已实现的状态来提供 dense reward，OPSD 通过把 "已知答案" 注入 context 来提供 dense per-token reward。本质都是 "用 hindsight 信息生成 dense supervision"。

4. **跟 DAgger 的关系**：DAgger (Ross et al., 2011) 是 on-policy imitation learning，teacher 在 student 访问的 state 上给 action。OPSD 就是 DAgger 的 soft 版本——teacher 在 student visited state 上给 action distribution，然后 student match distribution。这把 imitation learning 的 no-regret 理论（DAgger 的 $\tilde{O}(T^{1/2})$ bound）有可能迁移过来。

5. **跟 Decision Transformer / Trajectory matching 的 connection**：用 hindsight 信息（return-to-go）做 conditioning 是 DT 的核心。OPSD 用 $y^\star$ 做 conditioning 也是一种 "hindsight conditioning"，只不过目标不是生成而是 distillation。

6. **Process Reward Model 替代品**：PRM (Lightman 2023) 需要 expensive per-step human label，OPSD 可以视为 implicit PRM——teacher 看了 $y^\star$ 之后的 logit distribution 本身就是一个免费的 process reward。这给 PRM-free 的 dense reward RL 提供了路径。

7. **In-context editing (Qi 2025)** 已经证明在 knowledge editing setting 下用 distribution matching 可以把 context-induced knowledge 内化。OPSD 是这套思想在 reasoning setting 下的扩展。

8. **Reasoning length 的压缩效应**：SFT on concise solutions 让 model 学到 "短回答"，OPSD 通过 dense distillation 把 concise solution 的信息"展开"到每个 token 的 distribution 上，避免了 length collapse。这跟 LIMO (Ye 2025)、LIMA (Zhou 2023) "less is more" 的哲学有张力——是否可以用极少量 OPSD 样本达到同样效果？

9. **Self-consistency 的内化**：通常 self-consistency 是 test-time 采样多次取多数。OPSD 训练时强迫 student 在每个 token 位置匹配 "看过答案后的 distribution"，本质上是在训练阶段做 self-consistency 的内化。

10. **与 verifier-based RL 的桥接**：如果 $y^\star$ 替换成可验证 environment（mathematica, code interpreter），OPSD 可以演化成 "privileged context = verifier output" 的 on-policy RL，避免 sparse reward 问题。

---

## 10. 局限性

Paper 自己列了：
- 只验到 8B，更大模型是否如此 unknown
- 没用 correctness verification，只 match distribution
- 题太难时 teacher 无法 rationalize

我额外想到的：
- Teacher 用 initial policy $\theta_0$ 而非 $\theta_t$，随着训练进行 teacher 与 student gap 拉大，可能后期信号变弱（虽然 paper 实测 100 步内收敛）
- LoRA 限制了 capacity，full FT 是否同样 work 未验证
- 只在 math 上验，code reasoning、multi-hop QA、agentic reasoning 都没测
- clipping $\tau$ 没调，可能是 free lunch

---

## 11. References & Web Links

**核心 paper**：
- OPSD 原文 GitHub repo: https://github.com/siyan-zhao/OPSD
- GKD (Agarwal et al. 2024, On-Policy Distillation): https://arxiv.org/abs/2306.13649
- Thinking Machines Lab on-policy distillation: https://thinkingmachines.ai/blog/on-policy-distillation
- STaR (Zelikman 2022): https://arxiv.org/abs/2203.14465
- ReST (Gulcehre 2023): https://arxiv.org/abs/2308.08998
- GRPO / DeepSeek-Math (Shao 2024): https://arxiv.org/abs/2402.03300
- DeepSeek-R1 (Guo 2025): https://arxiv.org/abs/2501.12948
- Context Distillation (Snell 2022): https://arxiv.org/abs/2209.15189
- Minillm (Gu 2024): https://arxiv.org/abs/2306.08543
- Self-Rewarding LM (Yuan 2024): https://arxiv.org/abs/2401.10020
- DAgger (Ross 2011): https://arxiv.org/abs/1011.0686
- Self-Play Fine-Tuning SPIN (Chen 2024): https://arxiv.org/abs/2401.01335

**Reasoning datasets / models**:
- OpenThoughts (Guha 2025): https://arxiv.org/abs/2506.04178
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- LIMO (Ye 2025): https://arxiv.org/abs/2502.03387
- LIMA (Zhou 2023): https://arxiv.org/abs/2305.11206

**Concurrent / related**:
- SDPO (Hubotter 2026): https://arxiv.org/abs/2601.20802
- SDFT (Shenfeld 2026): https://arxiv.org/abs/2601.19897
- In-context Editing (Qi 2025): https://openreview.net/forum?id=...... 
- SFT memorizes, RL generalizes (Chu 2025): https://arxiv.org/abs/2501.17161
- Let's Verify Step by Step (Lightman 2023): https://arxiv.org/abs/2305.20050

---

## 12. 总结一句话

OPSD 把 "看答案学习" 这个日常学习行为变成了一套算法：同一个 model 在 privileged context（看过答案）下的 next-token distribution 作为 soft target，让"没看过答案的同一个 model"在 student 自己的 rollout 上 forward-KL 靠拢，per-token clipping 去掉 stylistic 噪声，从而避开 SFT 的 exposure bias、GRPO 的 sparse reward 和外部 teacher 的依赖。在 AIME/HMMT 上以 ~128x 的 token 效率优势跑赢 GRPO，揭示了一个值得关注的训练范式：**self-conditioned dense distillation 可能比 reward-based RL 更高效**。
