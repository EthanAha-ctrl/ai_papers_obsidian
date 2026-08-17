---
source_pdf: Direct Preference Optimization.pdf
paper_sha256: 92cb3a2b71362acda98a789b03d88688fd33cf5fcf13f81d2b1de30ee7d3b67a
processed_at: '2026-08-03T22:06:06-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DPO 用人话说

好，我们抛开公式，用大白话把 DPO 的故事讲清楚。

## 故事的起点：RLHF 到底在干嘛

你有一个 language model，它什么都会一点，但你想要它更"听话"——比如让它给的好答案比坏答案更频繁。

最直观的办法是 **SFT**：拿一堆好答案，让它模仿。问题是好答案不好收集，而且人类更擅长做 **比较**（"A 比 B 好"）而不是绝对评分。于是有了 **RLHF**：

1. 拿一堆 prompt，让 model 生成两个答案 $y_1, y_2$
2. 让人类标注员说哪个好，得到 $(y_w, y_l)$ 对
3. 训一个 **reward model** $r_\phi(x, y)$ 去拟合这些偏好
4. 用 **PPO** 这种 RL 算法，让 policy 去最大化 reward，同时别跑得太远（KL constraint）

听起来很合理，但工程上是噩梦：
- PPO 要 on-policy sampling，每个 step 都要从 model 采样
- 显存里同时塞四个 model（policy / value / reward / reference）
- 训练不稳定，reward hacking 是家常便饭
- 调参调到怀疑人生

## DPO 的核心一句话

**"你这个 RL 问题，其实有 closed form 解，那为啥还要做 RL？"**

具体来说，他们发现：

> 在 KL-constrained reward maximization 这个 objective 下，最优 policy 可以写成 closed form；把 reward 用 policy 反过来表示，塞进 Bradley-Terry 偏好模型里，partition function 自动消掉，最后剩下的就是一个简单的 binary cross entropy loss。

翻译成人话：**你本来要解一个复杂的 RL 优化问题，结果发现这个问题有解析解，而解析解刚好让你能直接用 supervised learning 的方式拟合 preference data，把整个 RL loop 给跳过了。**

## 数学直觉：为什么能跳过 RL

### Step 1：KL-constrained RL 的最优解是 closed form

RLHF 优化的东西写成数学就是：

$$
\max_{\pi_\theta} \mathbb{E}_{x, y \sim \pi_\theta}\big[ r_\phi(x, y) \big] - \beta \, \mathbb{D}_{\text{KL}}\big[ \pi_\theta \,\|\, \pi_{\text{ref}} \big]
$$

变量含义：
- $\pi_\theta$：你要训的 policy（就是 LM 本身）
- $\pi_{\text{ref}}$：reference policy（一般是 SFT 模型，frozen 不动）
- $r_\phi$：reward function
- $\beta > 0$：KL penalty 强度，$\beta$ 小 = 允许大幅偏离 reference，$\beta$ 大 = 几乎不动
- $\mathbb{D}_{\text{KL}}$：KL 散度，衡量两个分布差异

这是 entropy-regularized RL 的标准形式，变分推断告诉你它的最优解是：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y|x) \, \exp\Big( \frac{1}{\beta} r(x, y) \Big)
$$

变量含义：
- $\pi^*$：optimal policy
- $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(\frac{1}{\beta} r(x,y))$：partition function，只是个归一化常数确保 $\sum_y \pi^*(y|x) = 1$
- 这就是个 **Boltzmann / softmax distribution**：reference 乘上 reward 的 exponential 再归一化

直觉：最优 policy = reference 偏向高 reward 的方向 reweight 一下。reward 高的 $y$ 概率被放大，reward 低的被压缩，$\beta$ 控制锐度。

### Step 2：把 reward 用 policy 反过来写

把上面 closed form 两边取 log，做代数变换：

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

变量含义：
- 等号左边是 reward function
- 等号右边是 $\beta$ 乘以 log-ratio 加上一个只依赖 $x$ 的常数项 $\beta \log Z(x)$

人话翻译：**reward 就是"policy 相对 reference 的 log-ratio"，外加一个与 $y$ 无关的常数。**

这个 step 看起来很 trivial，但它是 DPO 的命门——它告诉你 **reward 完全可以用 policy 表达出来**。

### Step 3：Bradley-Terry 让常数项消失

人类偏好建模用 Bradley-Terry (BT)：

$$
p^*(y_1 \succ y_2 | x) = \frac{\exp\big(r^*(x, y_1)\big)}{\exp\big(r^*(x, y_1)\big) + \exp\big(r^*(x, y_2)\big)} = \sigma\big( r^*(x, y_1) - r^*(x, y_2) \big)
$$

变量含义：
- $p^*(y_1 \succ y_2 | x)$：人类偏好 $y_1$ 胜过 $y_2$ 的概率
- $\sigma(\cdot)$：sigmoid
- $r^*$：ground-truth reward

把 Step 2 的表达式代入 $r^*$：

$$
r^*(x, y_1) - r^*(x, y_2) = \beta \log \frac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)} - \beta \log \frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)}
$$

注意 $\beta \log Z(x)$ 在做差的时候 **直接消掉了**！因为 $Z(x)$ 只依赖 $x$，对 $y_1$ 和 $y_2$ 是同一个值。

于是得到论文里的 Eq. 6：

$$
p^*(y_1 \succ y_2 | x) = \sigma\left( \beta \log \frac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)} - \beta \log \frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)} \right)
$$

这一步是整个推导的 "啊哈！" 时刻：**你不需要知道 $Z(x)$，不需要做归一化，因为偏好只关心 reward 的差，常数项自动约掉。**

### Step 4：把 $\pi^*$ 换成 $\pi_\theta$，做 MLE

现在等号左边是人类的 preference probability，右边是 optimal policy $\pi^*$ 的函数。但我们没有 $\pi^*$——我们想要训练一个参数化 policy $\pi_\theta$ 去逼近它。

直接把 $\pi^*$ 替换成 $\pi_\theta$，对 preference dataset $\mathcal{D} = \{(x, y_w, y_l)\}$ 做 maximum likelihood，取负 log-likelihood：

$$
\boxed{\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]}
$$

变量含义：
- $y_w$：preferred completion（"winner"）
- $y_l$：dispreferred completion（"loser"）
- $\pi_\theta(y_w|x)$：当前 policy 给 $y_w$ 的概率
- $\pi_{\text{ref}}(y_w|x)$：frozen reference model 给 $y_w$ 的概率
- $\beta$：控制 KL penalty 强度，也充当 temperature

人话：这就是一个 **binary cross entropy loss**，label 永远是"preferred 该赢"，logit 是两个 implicit reward 的差。整个 RLHF pipeline 被压缩成一个 ~10 行 PyTorch 代码的 loss function。

## 梯度在干什么

论文里把 $\mathcal{L}_{\text{DPO}}$ 的梯度写得特别清楚：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \, \mathbb{E}_{(x, y_w, y_l)} \bigg[ \underbrace{\sigma\big( \hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w) \big)}_{\text{weight}} \cdot \bigg( \underbrace{\nabla_\theta \log \pi(y_w|x)}_{\text{拉高 } y_w} - \underbrace{\nabla_\theta \log \pi(y_l|x)}_{\text{压低 } y_l} \bigg) \bigg]
$$

其中 implicit reward：

$$
\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

直觉解读：

- **方向**：拉高 $y_w$ 的 log-prob，压低 $y_l$ 的 log-prob。这和朴素的 "maximize $p(y_w)$, minimize $p(y_l)$" 看起来差不多，但差一个关键 weight。
- **权重**：$\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))$ 是 implicit reward 把 $y_l$ 评得比 $y_w$ 高的程度。如果 model 已经正确排序（$\hat{r}_\theta(y_w) \gg \hat{r}_\theta(y_l)$），weight → 0，几乎不更新；如果 model 现在排序错了（$\hat{r}_\theta(y_l) > \hat{r}_\theta(y_w)$），weight → 1，强力 push。
- **效果**：自动 **focus on hard examples**——model 越错的样本梯度越大。这就是 DPO 不需要复杂 sampling 调度也能稳定收敛的关键。

### 一个反面教材：naïve 版为什么不行

Appendix Table 3 里有个 ablation：把 weight 去掉，直接用 $-\log p(y_w) + \log p(y_l)$。结果 model 训练后 degenerate，输出全是 "when when when when..." 这种垃圾。

为啥？没有 weight，所有样本被一视同仁 push，包括 model 已经学对的样本也继续被 push，最后过拟合到 mode collapse。weight 起到了 **早停 per-example** 的作用——学对了就别再 push。

## 理论部分讲了啥

Section 5 主要是给 DPO 的 reparameterization 提供理论 justification。

### Reward 的等价类

**Definition 1**：两个 reward $r, r'$ 等价当且仅当 $r(x,y) - r'(x,y) = f(x)$（差一个只依赖 prompt 的函数）。

**Lemma 1**：等价 reward 诱导相同的 preference distribution（在 BT / Plackett-Luce 下）。

证明 idea：BT 是 reward 差的 sigmoid，$f(x)$ 在做差时消掉。

**Lemma 2**：等价 reward 在 KL-constrained RL 下诱导相同的 optimal policy。

证明 idea：$f(x)$ 在 $\exp(\frac{1}{\beta}(r + f))$ 中提出 $\exp(\frac{1}{\beta} f(x))$，分子分母同时出现被消掉。

### Theorem 1：reparameterization 没损失表达能力

任何 reward equivalence class 都可以用 $r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$ 这种形式表示。

证明 idea：构造一个 projection operator $f(r; \pi_{\text{ref}}, \beta)$，对任意 reward $r$，减去 $\beta \log Z(x)$，把它投影到 equivalence class 中一个特定代表元上。结果刚好就是 $\beta \log \frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)}$。

直觉：reward 有冗余（可以任意加 $f(x)$），DPO 的 reparameterization **选了一个特殊的代表元——那个让 partition function $Z(x) = 1$ 的**。这就是为什么 DPO 不需要显式估计 $Z(x)$，它在等价类里挑了一个让 $Z$ 消失的 reward。

### Proposition 1：每个等价类里这种特殊 reward 唯一

证明 idea：反证法。如果有两个 reward $r, r'$ 都能写成 $\beta \log \frac{\pi}{\pi_{\text{ref}}}$ 形式且 $r' = r + f(x)$，那对应的 policy $\pi' = \pi \cdot \exp(\frac{1}{\beta} f(x))$。两个都是 valid distribution，sum over $y$ 必须 = 1，推出 $\exp(\frac{1}{\beta} f(x)) = 1$，即 $f(x) = 0$。

## PPO 为什么不稳：DPO 视角的诊断

Section 5.2 给了一个很有意思的分析。把 PPO 优化的目标重写成：

$$
\max \mathbb{E}_{\pi_\theta} \bigg[ \underbrace{r_\phi(x, y) - \beta \log \sum_y \pi_{\text{ref}}(y|x) \exp\Big(\frac{1}{\beta} r_\phi(x, y)\Big)}_{\text{normalized reward}} - \underbrace{\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}}_{\text{KL penalty}} \bigg]
$$

中间那个 $\beta \log \sum_y \pi_{\text{ref}}(y|x) \exp(\frac{1}{\beta} r_\phi)$ 就是 **soft value function**，它确保 reward 归一化。

这个 term 不影响最优解，但影响训练稳定性：
- 不估计它：policy gradient 方差爆炸，PPO 训不动
- 用 value network 估计它：又多一个 network 难调
- 用 human completion 当 baseline：single-sample MC 估计，方差还是大

DPO 的 reparameterization 选了一个让 $Z(x) = 1$ 的 reward，**这个 normalization term 直接是 0**，不需要 baseline，不需要 value network。这就是 DPO 数值稳定的核心原因之一。

## 实验讲了啥

### Sentiment Control (IMDb)

人造 ground truth reward = sentiment classifier，这样可以画 reward vs KL 的 frontier。

| 算法 | 观察 |
|------|------|
| DPO | frontier 上 strictly dominate PPO |
| PPO | 被 DPO 全面压制 |
| PPO-GT (用 ground truth reward) | 也被 DPO 超过 |

关键 takeaway：**DPO 不只是"够用"，而是比 PPO 更高效地优化同一个 objective**。原因猜测：PPO 用 stochastic policy gradient，方差大；DPO 是 supervised gradient，方差小很多。

### Summarization (TL;DR)

用 GPT-4 当 evaluator，对比 human reference summaries：

| Method | Win rate | 备注 |
|--------|----------|------|
| DPO (temp 0) | ~61% | best |
| PPO (best temp) | ~57% | |
| Best of N | < DPO | 推理昂贵 |
| Preferred-FT | ≈ SFT | 几乎无提升 |

关键观察：
- DPO **跨 temperature 鲁棒**——PPO 在高 temp 下退化到 base GPT-J 水平，DPO 几乎不变
- DPO 几乎没调 $\beta$（直接用 0.5），说明 **超参不敏感**

### Single-turn Dialogue (Anthropic HH)

| Method | 表现 |
|--------|------|
| DPO | 唯一能稳定超过 dataset 内 preferred completion 的可行方法 |
| Best of 128 | 能追平 DPO，但推理要 sample 128 次 |
| PPO (公开实现) | 找不到 prompt/temp 能超过 base Pythia-2.8B |

### OOD Generalization (Table 1)

在 TL;DR 上训，在 CNN/DailyMail 上测：

| Method | Temp 0 | Temp 0.25 |
|--------|--------|-----------|
| DPO | 0.36 | 0.31 |
| PPO | 0.26 | 0.23 |

DPO 在 distribution shift 下依然赢。有意思，因为直觉上 explicit reward model 应该 generalize 更好。猜测：DPO 直接对 preference data 做 MLE，结构上更像 supervised learning，inductive bias 不一样。

### Human Study 验证 GPT-4 evaluator

Table 2：GPT-4 与 human 的 agreement rate 大致等于 human 之间的 agreement rate，justify 用 GPT-4 当 evaluator 是合理的。

| Comparison | GPT-4(C)-H agree | H-H agree |
|-----------|------------------|-----------|
| DPO vs PPO-0 | 67% | 65% |
| PPO-1 vs PPO-0 | 85% | 87% |

## 算法 pipeline 人话版

1. **准备 reference model $\pi_{\text{ref}}$**：
   - 有 SFT 模型 → 直接用
   - 没有 SFT 模型 → 在 preferred completion 上做 MLE 训一个 pseudo-SFT
2. **冻结 $\pi_{\text{ref}}$**，整个 DPO 训练中不更新
3. **初始化 $\pi_\theta = \pi_{\text{ref}}$**（或者从它开始）
4. **对每个 batch $(x, y_w, y_l)$**：
   - 计算 $\pi_\theta(y_w|x), \pi_\theta(y_l|x)$（policy forward）
   - 计算 $\pi_{\text{ref}}(y_w|x), \pi_{\text{ref}}(y_l|x)$（reference forward，no grad）
   - 算两个 log-ratio
   - 算 BCE loss
   - 反向传播更新 $\pi_\theta$
5. **没有**：on-policy sampling、reward model、value network、PPO 那套 actor-critic

代码就几行（Appendix B）：

```python
pi_logratios = pi_yw_logps - pi_yl_logps      # policy: winner - loser
ref_logratios = ref_yw_logps - ref_yl_logps  # reference: winner - loser
losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
```

## 直觉串起来

把 DPO 串成一句话 narrative：

> RLHF 的 RL loop 在解一个有 closed form 的问题；DPO 利用 closed form 把 reward 吸收进 policy，靠 Bradley-Terry 做差消掉 partition function，最后把 RLHF 变成一个简单的 binary cross entropy。

更深的直觉：**variational inference 的结构**。KL-constrained RL 本质是 entropy-regularized RL，对应一个 Boltzmann distribution 形式的最优解。DPO 显式利用了这个结构，把 stochastic policy gradient 的方差问题直接绕过去。

## 实践中的坑

1. **$\beta$ 选择**：
   - 太小（0.01）→ 偏离 reference 太狠，reward hacking
   - 太大（5）→ 几乎不更新
   - 常用 0.05–0.5
2. **Reference model 必须 frozen**：只 forward，no grad
3. **Length bias**：log-prob 随长度累积，DPO 倾向更长 completion。SimPO 后来用 length-normalized log-prob 修复
4. **Data quality**：DPO 对 noise label 敏感——梯度聚焦在 model 觉得错的样本，noise label 会放大错误信号
5. **SFT 先做好**：$\pi_{\text{ref}}$ 质量决定 DPO 上限，SFT 不行 DPO 救不回来

## 后续发展

DPO 之后涌现一大批工作：

- **IPO** (Identity Preference Optimization)：用 L2 loss 替代 logistic loss，避免 DPO 在 saturated 区域梯度消失。https://arxiv.org/abs/2310.12036
- **KTO** (Kahneman-Tversky Optimization)：用 prospect theory 的 loss aversion，只需要好坏标签不需要成对。https://arxiv.org/abs/2402.01306
- **SimPO**：去掉 reference model，用 average log-prob 当 reward，简洁高效。https://arxiv.org/abs/2405.14734
- **ORPO**：把 SFT 和 preference learning 合并成一步。https://arxiv.org/abs/2403.07691
- **Iterative DPO / Online DPO**：用 DPO policy 生成新样本，迭代提升
- **Diffusion-DPO**：扩展到 diffusion model，对齐 image generation

工业应用：
- **Zephyr 7B**：用 DPO 把 Mistral-7B 对齐成 chat 模型。https://arxiv.org/abs/2310.16944
- **Llama-3**：用了 DPO 的变种做 alignment
- **StableLM-Zephyr**：DPO 训小模型

## 一句话总结

**DPO 把 RLHF 从一个 RL 问题变成一个 supervised learning 问题，靠的是识别出 KL-constrained RL objective 有 closed form 最优解，再用 BT 偏好模型消掉不可计算的 partition function。**

工程价值：训练简单、稳定、便宜。
理论价值：揭示了 policy 和 reward 之间的对偶关系，把 RLHF 和 variational inference 串起来。

References：
- 原论文：https://arxiv.org/abs/2305.18290
- 官方代码：https://github.com/eric-mitchell/dpo
- HuggingFace TRL DPO trainer：https://huggingface.co/docs/trl/dpo_trainer
- Bradley-Terry 原始 paper：https://doi.org/10.2307/2334029
- Zephyr 7B：https://arxiv.org/abs/2310.16944
- Levine 的 control as inference tutorial：https://arxiv.org/abs/1805.00909

---

# Direct Preference Optimization (DPO) 深度讲解

Andrej, 这篇 paper 是 preference learning 领域的一个标志性工作，核心 insight 非常 elegant。我尽量从 first principles 出发 build your intuition。

## 1. 高层 motivation：RLHF 到底哪里痛

标准 RLHF pipeline 分三步：

1. **SFT (Supervised Fine-Tuning)**：用高质量 demonstrations 微调 base LM，得到 $\pi^{\text{SFT}}$
2. **Reward Modeling**：从人类偏好数据 $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}$ 训练一个 reward model $r_\phi(x, y)$
3. **RL Fine-Tuning**：用 PPO 优化 policy $\pi_\theta$ 来最大化 $r_\phi$ 同时用 KL constraint 约束它别离 reference $\pi_{\text{ref}}$ 太远

痛点集中在 step 3：
- PPO 需要 on-policy sampling（每步从 $\pi_\theta$ 采样 completion），计算昂贵
- 需要 value network、reward model、reference model、policy 四个 model 同时在显存里
- reward hacking、mode collapse、训练不稳定都是常态
- 超参敏感，调参是门黑魔法

DPO 的核心质问是：**step 3 这个 RL loop 真的必要吗？** 答案是不必要，因为有一个 closed-form 的数学捷径。

## 2. 核心 insight 的推导路径

### 2.1 RLHF 的 KL-constrained objective

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} \big[ r_\phi(x, y) \big] - \beta \mathbb{D}_{\text{KL}}\big[ \pi_\theta(y|x) \,\|\, \pi_{\text{ref}}(y|x) \big]
$$

变量含义：
- $x$：prompt
- $y$：completion
- $\pi_\theta$：要训练的 policy（LM 本身）
- $\pi_{\text{ref}}$：reference policy，通常是 $\pi^{\text{SFT}}$
- $r_\phi$：学到的 reward function
- $\beta > 0$：KL penalty 的强度（小 $\beta$ = 激进偏离 reference；大 $\beta$ = 保守）
- $\mathbb{D}_{\text{KL}}$：KL 散度，衡量 $\pi_\theta$ 相对 $\pi_{\text{ref}}$ 的偏移

### 2.2 这个 objective 的最优解有 closed form

把 objective 展开成 pointwise 形式（per $x$ 单独优化），用 Gibbs 变分推断的标准技巧（Appendix A.1 完整推导）：

$$
\pi_r^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x, y) \right)
$$

变量含义：
- $\pi_r^*$：reward $r$ 诱导的 optimal policy
- $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\big(\frac{1}{\beta} r(x, y)\big)$：**partition function**，per-prompt 的归一化常数，确保 $\pi_r^*$ 是合法分布

这个公式本身就是 **softmax energy form**，和 Boltzmann distribution、maximum entropy RL 的 soft policy 完全一致。它告诉你：**最优 policy 就是 reference policy 乘以 reward 的 exponential，然后归一化**。KL penalty 等价于在 reward 上加了一个 entropy regularization，最终产出一个 soft（非 greedy）的 optimal policy。

### 2.3 Key move：用 policy 反过来参数化 reward

对 $\pi_r^*$ 取 log，做代数变换：

$$
r(x, y) = \beta \log \frac{\pi_r^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

这就是 **reward = $\beta \times$ log-ratio + 一个只依赖 $x$ 的常数项**。

这一步是整个 DPO 的命门：reward function 可以用 policy 显式表达出来，而且 partition function $Z(x)$ 只依赖 $x$，**不依赖 $y$**。

### 2.4 Plug 进 Bradley-Terry，partition function 自动消失

Bradley-Terry 偏好模型说人类比较 $y_1$ vs $y_2$ 的偏好概率是：

$$
p^*(y_1 \succ y_2 | x) = \frac{\exp(r^*(x, y_1))}{\exp(r^*(x, y_1)) + \exp(r^*(x, y_2))}
$$

把 2.3 的 $r^* = \beta \log \frac{\pi^*}{\pi_{\text{ref}}} + \beta \log Z$ 代入：

- 分子分母都出现 $\exp(\beta \log Z(x))$ 这个只依赖 $x$ 的因子
- **被约掉了**

于是：

$$
p^*(y_1 \succ y_2 | x) = \sigma\left( \beta \log \frac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)} - \beta \log \frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)} \right)
$$

这就是论文 Eq. 6。符号含义：
- $\sigma(\cdot)$：logistic sigmoid
- $y_1, y_2$：两个候选 completion
- $\pi^*$：optimal policy（对应 ground-truth reward $r^*$）
- 里面是两个 log-ratio 的差，本质上就是 **implicit reward** 的差

直觉：**偏好概率等于 implicit reward 差的 sigmoid**。我们不需要知道绝对 reward，只需要知道 reward 的相对顺序。

### 2.5 Maximum likelihood → DPO loss

把 $\pi^*$ 替换成参数化的 $\pi_\theta$，对 preference dataset 做 MLE，取 negative log-likelihood：

$$
\boxed{\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]}
$$

变量含义：
- $y_w$：preferred ("winning") completion
- $y_l$：dispreferred ("losing") completion
- $\beta$：temperature，控制偏离 reference 的强度（也是 KL strength）
- $\pi_\theta(y|x)$：当前 policy 对 completion 的 likelihood
- $\pi_{\text{ref}}(y|x)$：frozen reference model 的 likelihood

这个 loss 看起来就是 **binary cross entropy**，label 永远是 "preferred"，logit 是两个 implicit reward 之差。极其简单。

## 3. 梯度分析：DPO 在做什么

论文里给的梯度公式非常 illuminating：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \bigg[ \underbrace{\sigma\big( \hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w) \big)}_{\text{implicit reward 越错，权重越大}} \bigg[ \underbrace{\nabla_\theta \log \pi(y_w|x)}_{\text{提高 } y_w \text{ 概率}} - \underbrace{\nabla_\theta \log \pi(y_l|x)}_{\text{降低 } y_l \text{ 概率}} \bigg] \bigg]
$$

其中 implicit reward 定义为：

$$
\hat{r}_\theta(x, y) := \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

直觉拆解：

- 梯度方向：**拉高 $y_w$ 概率，压低 $y_l$ 概率**，和朴素 SFT-on-preferred 完全不一样，因为有负项
- 权重 $\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))$：当 implicit reward 把 $y_l$ 评得比 $y_w$ 高（即 model 当前认为 loser 更好），权重趋近 1，强力 push；当 model 已经正确排序，权重趋近 0，几乎不更新
- 这是一种 **self-paced / hard-example mining** 机制：训练自动聚焦在 model 还没学对的样本上

对比一个 naïve 版本（没有 weighting，就是 $-\log p(y_w) + \log p(y_l)$），论文 Appendix Table 3 显示 model 会 degenerate（输出 "when when when when..." 这种 garbage）。weighting 是关键，它本质上模拟了 PPO 中 reward model 提供的 gradient scale 信号。

## 4. 理论分析：Your LM is Secretly a Reward Model

### 4.1 Reward 的 equivalence class

**Definition 1**：两个 reward $r, r'$ 等价当且仅当 $r(x,y) - r'(x,y) = f(x)$，即差一个只依赖 prompt 的函数。

**Lemma 1**：在 Bradley-Terry / Plackett-Luce 框架下，等价 reward 诱导相同的 preference distribution。

证明核心：BT 是 reward 差的 sigmoid，$f(x)$ 在分子分母同时出现被消掉。这就是为什么 absolute reward 不可识别，只有相对 reward 可识别。

**Lemma 2**：等价 reward 在 KL-constrained RL 下诱导相同的 optimal policy。

证明：$f(x)$ 在 $\exp(\frac{1}{\beta}(r+f))$ 中以 $\exp(\frac{1}{\beta}f(x))$ 形式提出，分子分母同时出现被消掉。

### 4.2 Theorem 1：reparameterization 不损失表达能力

**Theorem 1**：在 $\pi_{\text{ref}}(y|x) > 0$ 且 $\beta > 0$ 的 mild 假设下，**任何**与 Plackett-Luce 一致的 reward equivalence class 都可以用 $r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$ 这种 reparameterization 表示。

证明 idea：构造 projection operator $f(r; \pi_{\text{ref}}, \beta)$，对任意 reward $r$，减去 $\beta \log Z(x)$ 把它投影到等价类中的某个特定代表元，结果就是 $\beta \log \frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)}$。

直觉：reward function 有冗余（任意加 $f(x)$ 都不影响 preference），DPO 的 reparameterization 选了一个 **特定的、可表达的、使得 partition function = 1 的代表元**。这是为什么 DPO 不需要估计 $Z(x)$——它在等价类里挑了一个 $Z(x) = 1$ 的 reward。

### 4.3 Proposition 1：唯一性

每个 equivalence class 中只有一个 reward 能写成 $\beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$ 这种归一化形式。证明用反证法：如果有两个，差 $f(x)$，但两者都是 valid 归一化 distribution，sum over $y$ 必须 = 1，推出 $f(x) = 0$。

## 5. PPO 不稳定性的诊断

Section 5.2 给了一个很有意思的分析。把 PPO 优化的 objective 写成：

$$
\max \mathbb{E}_{\pi_\theta} \bigg[ \underbrace{r_\phi(x, y) - \beta \log \sum_y \pi_{\text{ref}}(y|x) \exp(\frac{1}{\beta} r_\phi(x, y))}_{f(r_\phi, \pi_{\text{ref}}, \beta), \text{normalized reward}} - \underbrace{\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}}_{\text{KL penalty}} \bigg]
$$

中间那个 normalization term $\beta \log Z(x)$ 是 **soft value function**，它确保 reward 归一化。这个 term 不影响最优解，但是：

- 如果不估计它：policy gradient 方差巨大，PPO 不稳定
- 如果用 value network 估计它：又一个要训的 network，难优化
- 业界常用 trick：用 human completion 当 baseline，single-sample MC 估计
- DPO 的 reparameterization 选了一个 $Z(x) = 1$ 的 reward，**不需要任何 baseline**，这就是 DPO 数值稳定的核心原因之一

## 6. 实验结果解读

### 6.1 Sentiment control (IMDb)

构造 ground truth reward 是 sentiment classifier，可以做 reward-KL frontier 分析。

Figure 2 (Left) 的关键发现：
- DPO 在 reward-KL frontier 上 **strictly dominate** PPO（任何 KL 预算下，DPO 拿到更高 reward）
- DPO 甚至超过 PPO-GT（PPO 拿到 ground truth reward 的 oracle 版本）

这个结果说明：DPO 不是仅仅"够用"，而是 **比 PPO 更高效地优化同一个 objective**。原因猜测：PPO 的 stochastic gradient 估计有高方差，而 DPO 是 supervised gradient，方差小得多。

### 6.2 Summarization (TL;DR dataset)

Figure 2 (Right) 显示 GPT-4 eval 的 win rate vs human reference summaries：

| Method | Win rate vs reference |
|--------|----------------------|
| DPO (temp 0) | ~61% |
| PPO (best temp) | ~57% |
| Best of N | < DPO |
| Preferred-FT | ≈ SFT |
| SFT | 低 |

关键观察：
- DPO **跨 temperature 鲁棒**，PPO 在高 temperature 下退化严重（甚至降到 base GPT-J 水平）
- DPO 几乎没怎么调 $\beta$（直接用 0.5），说明 DPO **超参不敏感**

### 6.3 Single-turn dialogue (Anthropic HH)

Figure 3 (Left)：DPO 是 **唯一** 一个能稳定超过 dataset 内 preferred completion 的计算可行方法。Best of 128 能追平 DPO 但推理代价巨大（128x sampling）。

### 6.4 OOD generalization (Table 1)

在 TL;DR 上训练的模型在 CNN/DailyMail 上测试：

| Method | Temp 0 | Temp 0.25 |
|--------|--------|-----------|
| DPO | 0.36 | 0.31 |
| PPO | 0.26 | 0.23 |

DPO 在 distribution shift 下依然优于 PPO。这是个有意思的发现，因为直觉上 explicit reward model 应该 generalize 更好。可能的原因：DPO 的 policy 直接对 preference 数据 MLE，类似 supervised learning，结构 inductive bias 不同。

### 6.5 Human study (Table 2)

GPT-4 vs human agreement：

| Comparison | GPT-4(C)-H agree | H-H agree |
|-----------|------------------|-----------|
| DPO vs PPO-0 | 67% | 65% |
| PPO-1 vs PPO-0 | 85% | 87% |

GPT-4 与人类的 agreement 大致等同于人与人之间的 agreement，justify 了用 GPT-4 当 evaluator。

## 7. 工程实现要点

### 7.1 Reference model 怎么初始化

- 如果有 $\pi^{\text{SFT}}$：直接 $\pi_{\text{ref}} = \pi^{\text{SFT}}$
- 没有 SFT 模型（如 Anthropic HH 场景）：在 preferred completion 上做 MLE 得到一个 pseudo-SFT 当 $\pi_{\text{ref}}$：

$$
\pi_{\text{ref}} = \arg\max_\pi \mathbb{E}_{x, y_w \sim \mathcal{D}} [\log \pi(y_w|x)]
$$

这是为了 mitigating distribution shift——preference dataset 是从 $\pi^{\text{SFT}}$ 采的，$\pi_{\text{ref}}$ 必须尽可能接近那个分布。

### 7.2 实际超参

- $\beta = 0.1$ (默认), $\beta = 0.5$ (TL;DR)
- batch size = 64
- RMSprop, lr = 1e-6, 150 steps linear warmup
- **没有 RL loop，没有 on-policy sampling，没有 reward model，没有 value function**

### 7.3 PyTorch 代码 (论文 Appendix B)

核心几行：

```python
pi_logratios = pi_yw_logps - pi_yl_logps          # preferred - dispreferred (policy)
ref_logratios = ref_yw_logps - ref_yl_logps      # preferred - dispreferred (reference)
losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
rewards = beta * (pi_logps - ref_logps).detach()  # implicit reward (for logging)
```

注意 `detach()`：rewards 只是用来监控，不参与梯度。

## 8. 直觉总结（build your mental model）

把 DPO 串起来的核心 narrative：

1. **RLHF objective 的 optimal policy 有 closed form**：$\pi^* \propto \pi_{\text{ref}} \cdot \exp(r/\beta)$，这是 variational inference 的标准结论，KL regularization 等价于 entropy-regularized RL
2. **Reward 可以反过来用 policy 表达**：$r = \beta \log(\pi/\pi_{\text{ref}}) + \beta \log Z$，把 reward model 直接"吸收"进 policy 里
3. **Preference data 只看 reward 的差**：BT 模型让我们摆脱 $Z(x)$ 这个不可计算项，preference loss 简化成纯 policy 的函数
4. **MLE on preference data 直接给出最优 policy**：等价于一个 reparametrized Bradley-Terry model，是个二分类 cross entropy
5. **梯度自动产生 importance weighting**：implicit reward 错得越离谱的样本，梯度权重越大，类似 self-attention 中的 "focus on hard examples"

更深层的 intuition：**RLHF 的 RL loop 是为了求解一个有 closed form 的问题**。当 closed form 存在且 MLE 可行时，RL 是绕远路。DPO 是把 variational inference 的结构显式利用起来，绕开 stochastic optimization 的方差问题。

## 9. 局限与后续方向

论文 Discussion 中提到的开放问题：
- **OOD generalization**：DPO 不显式建 reward model，out-of-distribution 时表现如何？初步结果显示至少不比 PPO 差
- **Reward over-optimization**：Figure 3 right 略有性能下降，可能是 reward over-optimization 的表现
- **Scale**：论文只到 6B 参数，scaling 到 70B+ 是 open question（后来 Llama-2/3、Zephyr 等都验证了 DPO 在大 scale 上 work）
- **Multi-modal**：能否扩展到 image / video generation？（后续工作如 Diffusion-DPO 已验证）

后续发展：
- **IPO (Identity Preference Optimization)**：用 L2 loss 替代 logistic loss，避免 DPO 在 saturated 区域梯度消失
- **KTO (Kahneman-Tversky Optimization)**：用 prospect theory 的 loss aversion，只需要好坏标签不需要成对
- **SimPO**：去掉 reference model，用 average log-prob 当 reward
- **ORPO (Odds Ratio Preference Optimization)**：把 SFT 和 preference learning 合并成一步
- **Iterative DPO / Online DPO**：用 DPO policy 生成新样本，迭代提升

## 10. 实践中常见的坑

1. **$\beta$ 选择**：太小（如 0.01）→ 严重偏离 reference，可能 reward hacking；太大（如 5）→ 几乎不更新。常用范围 0.05–0.5
2. **Reference model 必须 frozen**：训练时 $\pi_{\text{ref}}$ 不更新，只做 forward pass 计算 log-prob
3. **Length bias**：DPO 容易偏好更长 completion（implicit reward 里 log-prob 随长度累积）。后续工作如 SimPO 用 length-normalized log-prob 解决
4. **Data quality**：DPO 对 noise label 敏感，因为梯度聚焦在"model 觉得错"的样本，noise label 会放大错误信号
5. **SFT 先做好**：$\pi_{\text{ref}}$ 质量决定 DPO 上限。如果 SFT 不行，DPO 修不回来

## References

- 原论文：https://arxiv.org/abs/2305.18290
- 官方代码：https://github.com/eric-mitchell/dpo
- HuggingFace TRL 中的 DPO 实现：https://huggingface.co/docs/trl/dpo_trainer
- Bradley-Terry 原始 paper：Bradley & Terry, *Biometrika* 1952, https://doi.org/10.2307/2334029
- Plackett-Luce model：Plackett, *JRSS Series C* 1975, https://doi.org/10.2307/2346567
- RLHF pipeline 参考：Stiennon et al. 2022, https://arxiv.org/abs/2009.01325
- PPO：Schulman et al. 2017, https://arxiv.org/abs/1707.06347
- Zephyr 7B（用 DPO 训练的 7B 模型）：https://arxiv.org/abs/2310.16944
- Llama-2 chat（RLHF，作 DPO 对比对象）：https://arxiv.org/abs/2307.09288
- IPO：https://arxiv.org/abs/2310.12036
- KTO：https://arxiv.org/abs/2402.01306
- SimPO：https://arxiv.org/abs/2405.14734
- ORPO：https://arxiv.org/abs/2403.07691
- Control as inference (Levine tutorial)：https://arxiv.org/abs/1805.00909

如果你想我们 deep dive 某一块——比如 DPO 梯度推导细节、和 PPO 的等价性、或者后续 IPO / SimPO 的改进——可以告诉我，我们可以继续展开。
