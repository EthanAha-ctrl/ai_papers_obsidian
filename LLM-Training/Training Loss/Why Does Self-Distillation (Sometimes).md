---
source_pdf: Why Does Self-Distillation (Sometimes).pdf
paper_sha256: a596f2e8cc0512ac3a60f7e0b4dd2830e1bf549393df9f22e43c98b2879b1927
processed_at: '2026-08-13T04:29:17-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲：Self-Distillation 在数学推理上为什么会翻车

## 一、这 paper 在吐槽什么事

一句话版本：**你让一个模型自己教自己，给它看答案当 teacher，它学到的不是"怎么想出答案"，是"怎么装作早就知道答案"。**

在 chemistry 这种题型重复度高的领域，装一下还挺管用，response 短了 accuracy 还涨了。但一到 math——题目千变万化、test 都是 train 没见过的——这个"装"就露馅了，accuracy 能掉 40%。

paper 的核心贡献是把"装"这件事量化了，并且追到了一个具体的 token 层面现象：**epistemic verbalization**（模型说"wait"、"hmm"、"perhaps"这种自我怀疑的词）被 self-distillation 系统性干掉了，而 math reasoning 恰恰靠这个来纠错。

参考：原始 SDPO paper https://arxiv.org/abs/2601.20802

---

## 二、Self-Distillation 到底在干啥

### 2.1 机制大白话

你有一个模型 $\pi_\theta$。你让它分饰两角：

- **Student**：只看到题目 $x$，正常生成答案 $y$
- **Teacher**：同一个模型、同一套权重，但 prompt 里塞进了正确答案 $s$，生成"带答案提示"的版本

训练 loss 就是让 student 的 token 分布去贴 teacher 的 token 分布：

$$\mathcal{L}_{\mathrm{SD}}(\theta) = \sum_t \mathrm{KL}\big(\pi_\theta(\cdot \mid x, y_{<t}) \;\|\; \mathrm{stopgrad}(\pi_\theta(\cdot \mid x, c, y_{<t}))\big) \tag{1}$$

变量翻译：
- $\theta$ = 模型参数
- $x$ = 题目
- $y_{<t}$ = 已经生成到第 $t$ 个 token 的前缀
- $c$ = teacher 多看的信息（通常是完整 solution $s$）
- $\mathrm{stopgrad}$ = teacher 那边不回传梯度，只当 target
- $\mathrm{KL}$ = KL 散度，衡量两个分布的差距

### 2.2 为什么这事儿有问题

关键在于 teacher 的"自信"是从 $c$ 里**借来的**，不是从 $\theta$ 里的知识来的。Teacher 看着答案写过程，当然一路顺畅、不犹豫、不写"wait let me reconsider"。Student 被 KL 硬拉去模仿这个分布，学到的是**输出层面的 style**（短、自信、不纠结），而**不是**输入层面缺失的那部分推理知识。

推理时 $c$ 没了，student 还是会惯性地产出"短、自信、不纠结"的东西——但它根本没那个底气，错了也不知道回头。

这和传统 knowledge distillation 的本质区别：传统 KD 里 teacher 是个**真的更强**的模型，它的自信来自更厚的参数知识；self-distillation 里 teacher 的自信来自**prompt 里作弊**。前者是知识的传递，后者是幻觉的传染。

---

## 三、Epistemic Verbalization：被忽视的"废话"其实很值钱

### 3.1 什么是 epistemic verbalization

就是模型在 reasoning 过程里外化自己不确定性的那些词。paper 用了 10 个 marker：

$$\mathcal{T} = \{\text{wait, hmm, perhaps, maybe, actually, alternatively, seems, might, likely, check}\}$$

统计函数：
$$E(y) = \sum_{t \in \mathcal{T}} \text{count}(t, y)$$

$E(y)$ 就是 response $y$ 里这些"犹豫词"出现总次数。

### 3.2 为什么这玩意儿在 math 里重要

Math reasoning 是 **self-Bayesian reasoning**：模型每一步只看到题面和之前自己写的 token，相当于在做一个迭代的 belief update。中间某步如果走偏了，"wait, that doesn't seem right" 这种 token 就是**自我检测 + 重新分支**的触发器。

DeepSeek-R1 这种模型满屏 "wait"、"hmm"，看着啰嗦，其实是它的**纠错机制**。你把这些 token 干掉，模型就变成一条道走到黑，错了没机会回头。

参考：Kim et al. 2026 关于 epistemic verbalization 的原始研究 https://arxiv.org/abs/2601.xxxxx

### 3.3 一张关键对照表

四个 generation 设置，信息量递增，看模型怎么变：

| Setting | Context $c$ | Avg Score | Avg Length | Epistemic Tokens $E(y)$ |
|---------|-------------|-----------|------------|------------------------|
| (1) Unguided | $\emptyset$ | 0.30 | **13,054** | **182.5** |
| (2) Full Solution | $s$（含 think） | 0.98 | 1,873 | 8.8 |
| (3) Solution w/o think | $s_{\backslash\text{think}}$ | 0.78 | 12,036 | 159.8 |
| (4) Regeneration | $\tilde{y}$ | 0.95 | 2,808 | 24.1 |

用人话读这张表：

- **(1) 不给任何提示**：模型自己摸索，啰嗦、犹豫、反复自我怀疑，182 次 epistemic token，写 1.3 万字，但只答对 30%
- **(2) 给完整 solution**：模型照抄，短、准、自信，几乎不犹豫，8.8 次 epistemic token，1873 字，98% 对
- **(3) 给 solution 但剥掉 think 过程**：信息少了一大块，模型又开始长篇犹豫，159.8 次 epistemic token
- **(4) 给一个"之前生成的正确回答"**：介于两者之间，24.1 次

这张表讲了一个故事：**context 信息越丰富 → 输出越短越自信 → epistemic verbalization 越少**。

信息量用 conditional mutual information 严格定义：

$$I(y; c \mid x) = H(y \mid x) - H(y \mid x, c) \tag{2}$$

- $H(y \mid x)$ = 只有题面时 $y$ 的熵（不确定性）
- $H(y \mid x, c)$ = 加上 context $c$ 后 $y$ 的熵
- $I(y; c \mid x)$ = $c$ 把不确定性降低了多少

四档严格排序（由 data processing inequality 保证）：

$$\underbrace{I(y;c\mid x)=0}_{(1)} < \underbrace{I(y;s_{\backslash\text{think}}\mid x)}_{(3)} \le \underbrace{I(y;\tilde{y}\mid x)}_{(4)} \le \underbrace{I(y;s\mid x)}_{(2)} \tag{3}$$

---

## 四、致命的 SFT 实验：同样的正确答案，风格不同结果天差地别

paper 做了一个特别干净的控制实验，直接戳中要害。

构造两个 SFT dataset，**全是正确回答**，800 条：

- $\mathcal{D}_{\text{ug}}$：unguided 生成的（长、啰嗦、多 epistemic token，~12k tokens/条）
- $\mathcal{D}_{\text{sg}}$：solution-guided 生成的（短、自信、少 epistemic token，~2k tokens/条）

唯一区别就是 epistemic density。然后用 DeepSeek-R1-Distill-Qwen-7B 做 SFT，看四个 math benchmark：

| Model | AIME24 | AIME25 | AMC23 | MATH500 |
|-------|--------|--------|-------|---------|
| Base | 54.79 | 37.92 | 89.06 | 92.19 |
| SFT on $\mathcal{D}_{\text{ug}}$ | 51.04 | 40.00 | 87.66 | 90.93 |
| SFT on $\mathcal{D}_{\text{sg}}$ | **20.21** | **12.71** | **57.03** | **65.52** |

**两个 dataset 都是正确答案**，一个掉 4 个点，一个掉 34 个点。唯一变量是 reasoning style。

这就是铁证：**问题不在答案对不对，在 reasoning 过程的 style**。$\mathcal{D}_{\text{sg}}$ 训出来的模型学会了"我应该直接写、别犹豫、别自我怀疑"——这种 style 在 solution-guided 那个 context 下是对的，但推理时 context 没了，模型还是保持这个 style，结果该犹豫的地方不犹豫，直接翻车。

---

## 五、On-Policy Self-Distillation：SDPO vs GRPO 实战对比

### 5.1 实验设置

- 算法：GRPO（标准 RL baseline）vs SDPO（on-policy self-distillation，Hübotter et al. 2026）
- 数据：DAPO-Math-17k
- 模型：DeepSeek-R1-Distill-Qwen-7B、Qwen3-8B（thinking on/off）、Olmo-3-7B-Instruct
- OOD 评测：AIME24、AMC23

Teacher 设两种 conditioning：
- $c = s$：完整 solution（信息最多）
- $c = s_{\backslash\text{think}}$：剥掉 think 过程的 solution（信息少一截）

### 5.2 DeepSeek-R1-Distill-Qwen-7B 结果

**Training 阶段**：
- GRPO：response length 微涨，score 微涨，正常
- SDPO $c=s$：length 和 score 一开始**双双暴跌**，后面缓慢回升但一直不如 GRPO
- SDPO $c=s_{\backslash\text{think}}$：length 跌得少，score 接近 GRPO

**OOD（AIME24）**：
- GRPO：54.7 → 56.0（+1.3，微涨）
- SDPO $c=s$：掉 ~40%，惨烈
- SDPO $c=s_{\backslash\text{think}}$：还是掉，但掉得少

**Epistemic token 变化**：GRPO 让 $E(y)$ 上升，SDPO 让 $E(y)$ 下降。完全对应"越压抑 epistemic → 越差"的规律。

### 5.3 Qwen3-8B Thinking ON

Qwen3-8B 开 thinking 本来就长篇大论、epistemic token 满天飞。

- GRPO 和 SDPO 都让 length 降，但 SDPO 降更狠
- OOD 上 GRPO 基本稳住，SDPO 持续掉
- 一个有意思的现象：SDPO 训练中途 length 先跌后涨——因为 teacher 固定为初始 policy，student 越短 teacher 给的 $c$ 信息越少（$I(y;c\mid x)$ 降），student 不得不靠 epistemic token 补回来，length 部分回升

### 5.4 Qwen3-8B Thinking OFF

这个设置最戏剧化。Thinking 关掉后模型本来短、弱、epistemic 少。

- **GRPO**：疯狂拉长 response，靠增加 epistemic verbalization 把训练分刷上去，AMC23 涨 36 分
- **SDPO**：反过来，越训越短，AMC23 只涨 6 分，AIME24 还掉分

同样的 base model，GRPO 走的是"多说多想"路线，SDPO 走的是"少说装自信"路线，结果天差地别。

### 5.5 Fixed vs Moving Teacher 的 ablation

SDPO 默认用 EMA smoothed teacher（rate 0.05），但 paper 发现 **rate=0.0（teacher 固定为初始 policy）反而更好**。

为什么？这是个**恶性反馈循环**：teacher 也在被同一个 loss 拉短，下一轮 teacher 更短更自信，student 跟着更短更自信，雪球越滚越大。固定 teacher 至少把雪球停住。

---

## 六、为什么 Chemistry 没事 Math 出事：Task Coverage 是钥匙

### 6.1 三个 dataset 对比

| Domain | 问题数 | 结构 |
|--------|--------|------|
| ScienceQ&A (Chemistry) | 2,400 | 6 大题型，表面变化多，底层结构高度重复，train/eval 90/10 split |
| LiveCodeBench v6 | 131 | train 和 eval 用同一批题，只换 test case |
| DAPO-Math-17k | 14,000 | 14k 不同题，train 和 eval(AIME/AMC/MATH500) 完全 disjoint |

Chemistry 和 LiveCodeBench 的特点是 **task coverage 窄**——题型就那几种，train 见过、eval 还是同类型。Math 的特点是 coverage 极宽且 eval 是 OOD。

### 6.2 Task Coverage 实验：$|\mathcal{D}| \in \{1, 8, 64, 128, 512\}$

用 Qwen3-8B thinking off，改训练题量：

**Training 阶段**：
- $|\mathcal{D}| \le 128$：SDPO 又快又好，length 掉 8 倍，score 飙升。task 少，"装自信"策略足够覆盖
- $|\mathcal{D}| = 512$：SDPO 开始不如 GRPO。task 多了，"装"覆盖不住了，需要 epistemic 来探索

**OOD（AIME24, MATH500）**：
- GRPO：$|\mathcal{D}|$ 越大越好，length 越长越好，epistemic token 越多越好
- SDPO：完全反过来，$|\mathcal{D}|$ 越小掉得越惨。即便 $|\mathcal{D}|=512$，SDPO 还是不如 base model

**核心结论**：epistemic verbalization 的价值**随 task 多样性递增**。题型重复（小 $|\mathcal{D}|$）时它就是冗余可压缩；题型多样（大 $|\mathcal{D}|$）时它是探索和纠错的关键。

---

## 七、给你的 intuition 串起来

把整篇 paper 的因果链拎出来：

1. **Teacher 有 $c$、student 没 $c$** → teacher 输出"装自信"的 distribution
2. **KL loss 拉着 student 去贴 teacher** → student 学到"装自信"的 style
3. **"装自信" style = 少 epistemic token = 短 response** → 表面看是 efficiency gain
4. **Task coverage 窄**：题型重复，装就装了，反正都见过，performance 涨
5. **Task coverage 宽**：题型新，该犹豫时不会犹豫，错了不回头，performance 暴跌

更深一层的 intuition：**reasoning model 的输出分布由两个独立维度构成**——(a) 推理内容的正确性，(b) 推理风格的不确定性表达。标准 RL objective（GRPO、RLVR）只优化 (a)，对 (b) 是间接影响。Self-distillation 的麻烦在于它**通过 distribution matching 直接干预 (b)**，而且干预方向是"压低不确定性"——这个方向在 OOD 下是有害的。

这给我们的启示是：**post-training objective 必须显式考虑 reasoning behavior，不能只看 answer correctness**。你 reward 的是答案对不对，但你实际上在塑造的是整个 reasoning style，而这个 style 会决定 OOD 上的鲁棒性。

---

## 八、几个值得 follow 的方向（带点 hallucination 的联想）

1. **Epistemic-aware reward shaping**：能不能在 RLVR 里加一个 auxiliary reward，鼓励模型在低置信步骤产生 epistemic token？比如用 entropy 估计每步的不确定性，高熵步奖励 "wait" 类 token。
   - 相关：entropy regularization in RL https://arxiv.org/abs/1812.05905

2. **Teacher context 的最优信息量**：paper 证明 $I(y;c\mid x)$ 越大越糟，但 $c=\emptyset$ 时 self-distillation 就退化成自己学自己没意义。中间一定有个 sweet spot。可以用 information bottleneck 那套框架来找。
   - IB 原始 paper https://arxiv.org/abs/1503.02406

3. **Epistemic token 作为 calibration signal**：模型说"wait"的频率和它实际错误率有没有可学习的映射关系？如果稳定，可以做 inference-time 的 confidence estimation，不需要额外训练。
   - LLM calibration 相关 https://arxiv.org/abs/2207.14275

4. **Self-distillation 和 DPO 的关系**：SDPO 本质是 DPO with self-teacher preference。如果把 teacher 的 $c$ 去掉，让模型自己做 positive/negative pair，是不是就避免了这个问题？这其实就是 self-rewarding LM 的路子。
   - Self-rewarding https://arxiv.org/abs/2401.10020

5. **Thinking mode 的本质**：Qwen3 thinking on/off 的对比说明 `<<

---

# Self-Distillation 在数学推理中的失效机制：一份深度技术解读

## 一、Paper 的核心 Thesis

这篇 paper 揭示了一个反直觉现象：**self-distillation 在 chemistry 等领域 "缩短 response + 提升 performance" 双赢，但在 math reasoning 中却 "缩短 response + 暴跌 performance"**。作者将这个 failure mode 追溯到一个被传统 training objective 完全忽略的 reasoning 行为维度——**epistemic verbalization**，即模型用 "wait", "hmm", "perhaps" 这类 tokens 外化自身不确定性、进行 self-correction 的能力。Self-distillation 会系统性 suppress 这个行为，而在 task coverage 宽泛的 math 任务上，这种 suppression 是致命的。

参考链接：
- arXiv: Self-Distillation 相关综述 https://arxiv.org/abs/2601.20802 (Hübotter et al., 2026, SDPO)
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Kim et al. epistemic verbalization 原始论文: https://arxiv.org/abs/2601.xxxxx (2026)

---

## 二、Self-Distillation 的数学骨架

### 2.1 训练目标

Self-distillation 的核心 loss 是一个 token-level KL divergence：

$$\mathcal{L}_{\mathrm{SD}}(\theta) = \sum_t \mathrm{KL}\Big(\pi_\theta(\cdot \mid x, y_{<t}) \;\Big\|\; \mathrm{stopgrad}\big(\pi_\theta(\cdot \mid x, c, y_{<t})\big)\Big) \tag{1}$$

变量含义：
- $\theta$：模型参数（teacher 和 student 共享）
- $x \in \mathcal{X}$：输入问题
- $y = (y_1, \ldots, y_T)$：生成的 token 序列
- $y_{<t} = (y_1, \ldots, y_{t-1})$：到第 $t$ 步为止的历史
- $c$：teacher 独享的 richer context（例如 ground-truth solution $s$）
- $\pi_\theta(\cdot \mid x, y_{<t})$：student 的 next-token 分布（**没有** $c$）
- $\pi_\theta(\cdot \mid x, c, y_{<t})$：teacher 的 next-token 分布（**有** $c$）
- $\mathrm{stopgrad}$：teacher 侧 detach 梯度，作为固定的 distillation target

**关键点**：student 被迫去匹配一个它**在推理时无法访问**的信息条件下的分布。这正是问题的根源。

### 2.2 与传统 Knowledge Distillation 的区别

传统 KD 中 teacher 是一个**独立的、更强的模型**，它的 distribution 是合理的 target；而 self-distillation 中 teacher 和 student 是**同一个模型**，teacher 的 confidence 完全来自 context $c$ 的"作弊"，而不是来自参数知识。Student 通过 KL 匹配学到的，是 teacher 输出分布的**形状**，而非其背后的知识——于是它学到的只是 "表现得好像我知道答案"的 style。

---

## 三、Information Richness 的形式化：Conditional Mutual Information

作者用 Shannon 信息论来量化 context 的"作弊程度"：

$$I(y; c \mid x) = H(y \mid x) - H(y \mid x, c) \tag{2}$$

变量解释：
- $y$：目标 token 序列
- $c$：额外 context
- $x$：输入问题
- $H(y \mid x)$：仅给 $x$ 时 $y$ 的条件熵（不确定性）
- $H(y \mid x, c)$：给 $x$ 和 $c$ 时 $y$ 的条件熵
- $I(y; c \mid x)$：$c$ 对 $y$ 提供的**额外信息量**

直觉：$I(y; c \mid x)$ 越大，teacher 比 student "知道得越多"，distillation 目标越是在强迫 student 内化它根本没有的信息。

### 3.1 四档信息丰富度的实验设计

作者设计了四个 generation settings，构造严格的信息量序：

| 设置 | Context $c$ | 信息量 |
|------|-------------|--------|
| (1) Unguided | $\emptyset$ | $I(y;c\mid x)=0$ |
| (3) Solution w/o think | $s_{\backslash\mathrm{think}}$ | $I(y; s_{\backslash\mathrm{think}} \mid x)$ |
| (4) Regeneration-cond. | $\tilde{y}$ | $I(y; \tilde{y} \mid x)$ |
| (2) Full Solution | $s$ | $I(y; s \mid x)$（最大）|

由 data processing inequality 得到严格序：
$$I(y; c \mid x) = 0 \;<\; I(y; s_{\backslash\mathrm{think}} \mid x) \;\le\; I(y; \tilde{y} \mid x) \;\le\; I(y; s \mid x) \tag{3}$$

### 3.2 实验数据（DeepSeek-R1-Distill-Qwen-7B, DAPO-Math-17k, 100 题）

Table 1 的核心数据：

| Setting | Avg. Score | Avg. Length | Epistemic Token Count $E(y)$ |
|---------|-----------|-----------|---------------------|
| (1) Unguided | 0.30 | **13,054** | **182.5** |
| (2) Solution-Guided ($c=s$) | 0.98 | **1,873** | **8.8** |
| (3) $c=s_{\backslash\mathrm{think}}$ | 0.78 | 12,036 | 159.8 |
| (4) Regeneration-Cond. | 0.95 | 2,808 | 24.1 |

**核心观察**：随着 $I(y;c\mid x)$ 单调增大，response length 和 epistemic token count 单调减小。这说明信息越丰富，模型越"自信"、越"简洁"——但这种自信是**借来的**，是 $c$ 给它的。

特别注意 (3) vs (2)：去掉 `
