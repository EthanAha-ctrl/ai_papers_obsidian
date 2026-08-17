---
source_pdf: REFERENCES IMPROVE LLM ALIGNMENT IN NON-VERIFIABLE DOMAINS.pdf
paper_sha256: c8e26dea001119b93b1110a033b5b38abbdf872e4bc6c4a3d047a4d16f28c493
processed_at: '2026-08-11T22:01:53-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇 paper

## 一句话概括

**想让 8B 小模型自己教自己变强，但 8B 当 judge 太菜。解决办法：塞一份 DeepSeek-V3 的参考答案给它看，让它对照着打分，这样它判分准了，自学效果就起来了。**

---

## 这 paper 在解决什么实际问题？

你 Karpathy 肯定遇到过这种纠结：

- RLVR 在 math/code 上特别 work，因为有 ground truth 可以 verify
- 但 alignment 这种活儿，"好回答"长啥样？没法用 rule 验证
- 传统做法是训一个 reward model（RLHF/RLAIF），但训 RM 需要 preference data，成本高
- Self-rewarding（让模型给自己打分）听起来很美，但 8B 模型打分能力太弱，noise 太大，训出来没用

这篇 paper 的 insight 特别朴素：**既然 8B 自己判不准，那就给它一份"标准答案"对照着判**。

这跟我们改作业一个道理：让一个本科生批改本科生作业，质量堪忧；但如果给他一份教授写的 reference answer，他对照着看哪个学生答卷更接近 reference，准确率就上来了。

---

## 关键发现：光给 reference 没用，得告诉它怎么用

这是 paper 最有意思的一点。

前人（HREF、LLMBar-Ref）也试过塞 reference 进 prompt，但效果就提升 1 个点，几乎没用。这帮人得出结论：reference 没啥用。

这篇 paper 发现：**问题出在 prompt 上**。如果你只是把 reference 当"参考信息"附在 prompt 里，judge 根本不 care 它。你得**明确告诉 judge**："你这个判断的核心逻辑是看哪个 candidate 更接近 reference 的 quality 和 content"。

他们设计的两个 prompt：
- **RefEval**："评估哪个 output 在 quality 和 content 上更贴近 reference 展现的标准"
- **RefMatch**："你就是个 matcher，判断哪个 output 跟 reference 更像"

RefEval 在 11 个 open-source model 上 average accuracy 79.1%，比 reference-free 的 72.3% 高 6.8 个点。比前人 naive 加 reference 的 HREF-Ref (74.8%) 也高 4.3 个点。

---

## 小模型受益最大——这才是精华

看 Table 2 这组数据特别震撼：

| Judge | Base (无ref) | RefEval (有ref) | 提升 |
|-------|-------------|----------------|------|
| Mistral-7B-v0.3 | 47.0% | 69.6% | **+22.6** |
| Llama-3-8B | 60.1% | 77.5% | **+17.4** |
| Qwen-2.5-72B | 79.4% | 84.6% | +5.2 |

**7B 模型加个 reference，judge 能力直接拉到接近 70B 水平**。这非常震撼。

直觉上理解：reference 给了 judge 一个外部 anchor，让它从"绝对质量评估"这个开放问题，退化成"跟 reference 对照匹配"这个相对问题。小模型做绝对判断没 internal knowledge 支撑，但做相对匹配就容易多了。

类比一下：让一个初中生评两篇作文哪篇好，他可能懵逼；但给他一篇范文明确说"这是好作文的标准"，他对照着打分就靠谱多了。

---

## 训练 pipeline：两步走

Stage 1: 直接 SFT on DeepSeek-V3 的 reference answer
- UltraFeedback 60K instructions
- DeepSeek-V3 生成 reference
- 普通监督学习，让模型先学会"高质量回答长啥样"

Stage 2: 用自己当 judge 跑 DPO
- 对每个 instruction，从自己采样 5 个 candidate（temperature 0.8）
- 用 RefEval prompt 让自己给自己 pairwise 打分（10 次两两比较选 best/worst）
- 拿 best vs worst 当 DPO 的 preference pair 训练

**关键设计**：judge 就是模型自己（self-improvement），但 prompt 里塞了 DeepSeek-V3 的 reference。这是"半自学"——参考答案外部给，判分自己做。

---

## 为什么先 SFT 再 DPO？

Table 3 这个发现挺反直觉的：

- Base model 直接 DPO（用 ArmoRM 这个 SOTA reward model）：49.2 (AlpacaEval)
- 先 SFT on DeepSeek-V3 reference：53.9

**SFT distillation 竟然比用 SOTA reward model 跑 DPO 还强**。

这其实说明一件事：**alignment 的瓶颈可能不在 preference signal，而在 starting point 的分布质量**。DeepSeek-V3 的回答本身质量足够高，光 SFT 就把分布拉到位了。DPO 在这个好的 starting point 上做精调，才能进一步 gain。

这也解释了为什么 OpenAI / Anthropic 做了这么多 RLHF 之后，最终 publish 出来的模型还是看着像 SFT distillation 的产物——RLHF 可能只是 polish，distillation 才是主力。

---

## 最终结果有多强？

Llama-3-8B-Instruct 经过这套流程：
- AlpacaEval: 73.1 (vs SimPO baseline 51.6，**+21.5**)
- Arena-Hard: 58.7 (vs SimPO 36.2，**+22.5**)

Qwen2.5-7B-SFT 经过这套流程：
- Arena-Hard: 74.1 (vs 官方 Qwen2.5-7B-Instruct 58.0，**+16.1**)

8B 模型通过这套流程，在某些 benchmark 上能超过 Qwen 官方 publish 的 instruct 版本。这对资源有限的团队来说价值很大。

---

## 几个有意思的细节

### 1. Reference 质量影响绝对分数，但不影响 mechanism work
用 GPT-4o-mini 替代 DeepSeek-V3 生成 reference，绝对分数掉很多（AE 从 73.1 掉到 44.4）。但 RefEval vs RefFree 的相对 gain 还在（+1.8 AE, +16.6 AH）。说明 reference-guided 这个机制本身有 structural 优势，不依赖顶级 reference。

### 2. Coding/Math 任务受益最大，Creative 任务看 base model
Llama-3-8B-Instruct 在 Creative Tasks 上也能从 reference 受益，但 Qwen2.5-7B-SFT 几乎不行。说明 **reference utilization 本身是一种能力，需要充分 post-training 才能掌握**。这有点反直觉——你以为是开放性任务用不上 reference，其实是模型得先有足够 post-training 才知道怎么用 reference。

### 3. Reference 降低了 judge 之间的 disagreement
11 个 judge 之间的 pairwise agreement 从 76.6% 涨到 81.4%。Reference 给了共享 anchor，让不同模型的判分趋同。这间接证明 reference 真的在 grounding decision，不是 placebo。

### 4. Multi-reference voting 边际收益递减
用 5 个 frontier model 生成 5 个 reference 投票，accuracy 从 81.4% 涨到 82.3%，平均每多一个 reference 涨 0.2 个点。所以单 reference 已经够用，省钱。

---

## 我的几点直觉判断

### 1. 这套方法最 practical 的价值是省掉了训 reward model
ArmoRM 是 8B finetuned reward model，在 RewardBench 上 SOTA。要训出这样的 RM 需要 preference data + 训练成本。这篇 paper 证明：**用 LLM 自己 + reference 的 prompt 就能 match 上 ArmoRM 的效果**。对没资源训 RM 的团队，这是降维打击。

### 2. 但有个隐忧：分布漂移
Stage 1 SFT 把模型分布拉向 DeepSeek-V3，Stage 2 DPO 的 reference 又是 DeepSeek-V3，judge 也是被 DeepSeek-V3 reference 引导的——整个 pipeline 都在 DeepSeek-V3 的引力下。最后训出来的模型会不会变成"DeepSeek-V3 lite"？paper 没讨论这个。

实测看 AlpacaEval 73.1 vs DeepSeek-V3 自己 84.8，还差 11 个点，没完全 collapse。但 Arena-Hard 上 Qwen2.5-7B 训出来 74.1 超过 DeepSeek-V3 之外的一些 frontier——说明确实是学到 reference 的"质量维度"而不是 surface form mimic。

### 3. 真正的瓶颈可能是 reference 生成成本
60K instructions 用 DeepSeek-V3 跑一遍，假设平均 1K tokens 输出，按 $0.27/M input + $1.1/M output 算大概几百美金。可承受。但要是 scale 到 1M instructions，成本就上来了。

### 4. 这 paper 没做的：online reference update
现在是 offline 一次性生成 reference。如果 Stage 2 DPO 训练过程中，定期用更新后的 model 自己生成 reference（EMA teacher 之类的），可能能突破 DeepSeek-V3 的 ceiling。这是 self-improvement 的真正终极形态。

### 5. 跟 RLVR 的统一视角
RLVR 的 verifier 提供 hard ground truth (0/1)，这 paper 的 reference 提供 soft ground truth (相似度)。Reward model 提供 learned scalar reward。这三者在 preference learning 框架下其实是 spectrum：

- Hard verifier：信号最干净，但只适用于 verifiable domain
- Reference-grounded judge：信号 medium，domain-agnostic
- Learned RM：信号最 flexible，但需要训练成本 + 可能过拟合

paper 把中间这条路走通了，技术意义是把 RLVR 的方法论 leverage 到 alignment domain。

---

## 给你的实操建议

Karpathy 你如果要在 eureka labs 类项目里试这套方法：

1. **基础 recipe**：先 SFT on DeepSeek-V3 answer，再 DPO with self-judge + reference。Table 3 证明这套比单独 DPO 强很多。

2. **小模型尤其推荐**：7B/8B 模型 judge 能力提升 20+ 个点，ROI 巨高。70B+ 模型提升 marginal，看你 cost-benefit。

3. **别用 ROUGE/BERTScore 当 reward**：Table 3 里 ROUGE 才 56.4，BERTScore 58.8，远不如 RefEval 73.1。Lexical/surface metric 在 alignment 上不够用。

4. **β 要 grid search**：DPO 的 KL coefficient 在 0.005-0.1 范围扫一遍，sensitive。

5. **On-policy sampling 是关键**：用 temperature 0.8 采 5 个 candidate 做 pairwise，比用静态 preference data 强。这跟 SimPO 的发现一致。

6. **别用 multi-reference**：边际收益太小，单 reference 已经够。

---

## 最后的吐槽

这篇 paper 最大的问题：实验太多，narrative 不够聚焦。11 个 judge × 5 个 dataset × 13 个 prompting protocol = 数据表一大堆，读起来累。核心 insight 其实就一句话能讲清楚：**reference 改变了 judge 任务的性质，从绝对评分变成相对匹配，让小模型 judge 能力提升到接近 frontier 水平**。

但作为技术 contribution，把这套流程跑通并证明它 work，对资源有限的团队非常有价值。这套方法不用训 RM，不用标 preference data，只用一次 API 调用生成 reference + 几十 GPU-hours DPO，就能让 8B 模型在某些 benchmark 上超过 SimPO / 官方 Instruct 版本。这种 pragmatic 的方法学贡献，比那些花哨的 algorithm 调整实在多了。

论文代码应该没放出来，但复现难度不高——核心就是 prompt + DPO，prompt 都在 Appendix H 里贴出来了，照着抄就行。

---

# 论文详解：References Improve LLM Alignment in Non-Verifiable Domains

## 1. 核心动机与问题定位

这篇 paper 处理的是 post-training 中一个很 fundamental 的 gap：**RLVR 在 reasoning task 上成功，但 alignment 这种 non-verifiable domain 无法直接套用**。Karpathy 你应该很清楚，RLVR 的核心假设是 reward 可被 rule-based verifier 验证（math 有 gold answer, code 可 execution），但 instruction-following 这种 task 没有 ground truth。

作者提出了一个非常 pragmatic 的 insight：**reference outputs（来自 frontier LLM 如 DeepSeek-V3、GPT-4o）可作为 soft verifier 的 grounding signal**。这本质上是把 NLG evaluation 里 traditional 的 reference-based metric 思路搬到 LLM-as-a-Judge 范式，但用 LLM 替代 BLEU/ROUGE 这种 lexical metric。

paper 链接：https://arxiv.org/abs/2504.19087 (假设性链接，作者 Kejian Shi, Yixin Liu 等 Yale/Scale AI/Salesforce)
相关背景 paper：
- RLVR / ProRL: https://arxiv.org/abs/2412.06593
- Self-Rewarding LMs (Yuan et al.): https://arxiv.org/abs/2401.10020
- DPO (Rafailov et al.): https://arxiv.org/abs/2305.18290
- LLMBar (Zeng et al.): https://arxiv.org/abs/2310.05470
- ArmoRM: https://arxiv.org/abs/2406.12845

---

## 2. 方法论拆解

### 2.1 Reference-guided LLM-Judge 的 prompt 设计

paper 的核心 insight 是：**naive 把 reference 塞进 prompt 几乎没用**（HREF-Ref 只比 reference-free 高 1.1 个点），必须 **explicit 指导 judge 如何使用 reference**。

设计了两个核心 prompting protocol：

#### RefEval（最优方法）
核心 instruction：让 LLM judge 判断 **哪个 candidate output 在 quality 和 content 上更贴近 reference 展现出的标准**，同时仍需满足原 instruction。这相当于把 reference 当作 "成功 instruction-following 的 exemplar"。

#### RefMatch
更激进：让 judge 主要扮演 **semantic + stylistic matcher**，明确告诉它 "determine which output demonstrates closer similarity to the reference"。这个方法在 LLMBar-Adversarial 上表现最好（74.1% vs RefEval 74.9% 接近），因为 adversarial example 的 dispreferred output 有 superficial appealing qualities，reference 起到了 anchor 作用。

这种设计 philosophy 与 RLVR 中 gold solution 的作用形成类比：reference 不是 ground truth，但提供了 **"目标分布的 sample"**，让 judge 的 evaluation 从 "absolute quality assessment" 退化为 "relative similarity matching"——后者对小模型更容易。

### 2.2 关键实验数据（Evaluation 阶段）

Table 1 是核心结果，11 个 open-source LLM 作为 judge，5 个 dataset 平均：

| Method | Avg Accuracy |
|--------|--------------|
| LLMBar-Base (vanilla) | 72.3% |
| CoT | 71.2% |
| HREF-Ref (naive ref) | 74.8% |
| LLMBar-Ref | 74.0% |
| RefMatch | 77.7% |
| **RefEval** | **79.1%** |

RefEval 相对 reference-free baseline 提升 **+6.8 absolute points**，且 statistical significant (p<0.05)。

#### 关键发现：小模型受益最大
Table 2 显示：
- **Llama-3-8B**: Base 60.1% → RefEval 77.5% (+17.4 points！)
- **Mistral-7B-v0.3**: Base 47.0% → RefEval 69.6% (+22.6 points！！)
- **Qwen-2.5-72B**: Base 79.4% → RefEval 84.6% (+5.2 points)

这非常有意思——**reference 把小模型的 evaluation 能力"拉"到了接近大模型水平**。这暗示 reference 提供了 "knowledge/procedure grounding"，弥补了小模型 internal knowledge 不足。可以类比 reasoning model 中的 CoT——内部 reasoning vs 外部 grounding 是互补的两条路径。

#### Oracle Human Reference 实验（Appendix A.8）
对 LLMBar-Adversarial 用 human 编辑过的 "oracle reference"，frontier judge 也受益：
- GPT-4o: RefEval 86.8% → RefEval-Oracle 88.4%
- GPT-4.1: 86.7% → 88.6%
- Qwen-2.5-72B: 79.9% → 81.8%
- Llama-3.1-70B: 82.8% → 84.6%

说明 reference quality 是 monotonically 影响因素，frontier model 也能从更高质量 reference 获益。

---

## 3. Training Pipeline：Reference-Guided Self-Improvement

这是 paper 的 second contribution，也是更重要的部分。

### 3.1 两阶段 Training Process

**Stage 1: SFT Distillation on Reference Outputs**
- 用 DeepSeek-V3 在 UltraFeedback 60K instructions 上生成 reference
- Direct SFT on these references
- Hyperparameter: 2 epochs, batch size 128, lr 5e-6, linear scheduler, 3% warmup, max seq len 2048
- Filter 后 883K SFT instances

**Stage 2: DPO with Reference-Guided Self-Judge**
DPO objective (Eq. 1)：

$$\mathcal{L}_{\mathrm{DPO}}(p_\theta; p_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma\left( \beta \log \frac{p_\theta(y_w | x)}{p_{\mathrm{ref}}(y_w | x)} - \beta \log \frac{p_\theta(y_l | x)}{p_{\mathrm{ref}}(y_l | x)} \right) \right]$$

变量含义：
- $x$: input instruction（来自 UltraFeedback）
- $y_w, y_l$: preferred / dispreferred output pair
- $p_\theta$: 当前训练的 policy model
- $p_{\mathrm{ref}}$: reference policy（DPO 的 reference model，从待 fine-tune 的 checkpoint 初始化，**注意这与论文中 "reference output" 的 reference 是不同概念**——前者是 DPO 公式里的 KL anchor，后者是 DeepSeek-V3 生成的高质量回答）
- $\sigma(\cdot)$: sigmoid function
- $\beta$: KL regularization 强度超参（grid search 0.005-0.1）
- $D$: preference dataset

**On-policy data generation** (关键设计)：
- 对每个 instruction，从 $p_{\mathrm{ref}}$ 中 temperature=0.8 采样 5 个候选
- 用 LLM-judge 做 pairwise comparison 排序
- 选 best 和 worst 构成 $(y_w, y_l)$
- 60K instructions × $\binom{5}{2}=10$ pairwise = 600K judgments

这里有一个很重要的 pipeline design choice：**LLM-judge 是模型自己**（self-improvement），但用了 reference-guided prompting——即 judge 模型 = policy model，但 evaluation 时 prompt 中包含 DeepSeek-V3 的 reference。

### 3.2 实验结果（Table 3, 关键数据）

| Method | Llama-AE | Llama-AH | Qwen-AE | Qwen-AH |
|--------|----------|----------|---------|---------|
| Base | 25.0 | 27.1 | 14.4 | 23.4 |
| ArmoRM-Base (DPO from base) | 49.2 | 40.4 | 32.6 | 58.6 |
| DSV3-Distill (Stage 1 SFT) | 53.9 | 42.2 | 48.8 | 56.5 |
| ROUGE (from DSV3-Distill) | 56.4 | 52.1 | 50.9 | 67.4 |
| BERTScore | 58.8 | 53.0 | 55.3 | 64.5 |
| ArmoRM (from DSV3-Distill) | 73.9 | 58.6 | 66.8 | 72.2 |
| RefFree (self-improve) | 67.5 | 53.8 | 65.1 | 71.8 |
| **RefEval (self-improve)** | **73.1** | **58.7** | **70.0** | **74.1** |

**三个 key takeaways**：

1. **SFT distillation on frontier reference > DPO from base with finetuned RM**
   DSV3-Distill (53.9 AE) > ArmoRM-Base (49.2 AE) on Llama
   这说明 high-quality SFT data 的价值被低估了——直接 distill frontier model 的 answer 比用 reward model 做 DPO 更有效作为 starting point。

2. **Self-improvement works**
   RefFree > DSV3-Distill: +13.6 (Llama AE), +16.3 (Qwen AE)
   LLM 可作为自己的 judge 提供有效 preference signal。

3. **Reference 在 self-improvement 中提供额外 boost**
   RefEval > RefFree: +5.6 (Llama AE), +4.9 (Qwen AE)
   而且 **RefEval 与 ArmoRM（一个 finetuned 的 8B reward model，在 RewardBench 上 SOTA）performance 相当**——这说明不需要训练专门的 reward model，reference-guided LLM-judge 就够用。

### 3.3 与 SimPO 和 Qwen2.5-7B-Instruct 比较（Table 5）

| Model | AlpacaEval | Arena-Hard |
|-------|-----------|------------|
| DeepSeek-V3 | 84.8 | 94.9 |
| SimPO-Llama3-8B-Inst | 51.6 | 36.2 |
| RefEval-Llama3-8B-Inst | 73.1 | 58.7 |
| Qwen2.5-7B-Inst | 29.9 | 58.0 |
| RefEval-Qwen2.5-7B | 70.0 | 74.1 |

RefEval pipeline 用 8B 模型达到了 +21.5 AlpacaEval / +22.5 Arena-Hard 超过 SimPO baseline，且 Qwen2.5-7B 经过此 pipeline 后 Arena-Hard 74.1，超过其官方 Instruct 版本 +16.1 points。

---

## 4. 关键 Ablation 与分析

### 4.1 Reference Quality Ablation（Table 6）

用 GPT-4o-mini 替代 DeepSeek-V3 生成 reference：

| Method (GPT-4o-mini ref) | AE | AH |
|--------------------------|-----|-----|
| Distill | 28.7 | 40.7 |
| RefFree | 42.6 | 41.7 |
| RefEval | 44.4 | 58.3 |

RefEval 仍然超过 RefFree (+1.8 AE, +16.6 AH)。说明 **reference-guided 机制本身有结构性优势**，不仅依赖 reference quality。但 high-quality reference 的绝对收益更大。

### 4.2 Task Category 分析（Figure 3）

按 GPT-4o 分类 AlpacaEval 和 Arena-Hard 的 instruction 为四类：
- Coding&Math
- Creative Tasks
- Information Seeking
- Reasoning&Planning

**Coding&Math 类** reference-guided 优势最大——因为这类问题有相对确定的正确答案，reference 能提供 procedure grounding。

**Creative Tasks** 上 Qwen2.5-7B-SFT 几乎没有 reference gain，但 Llama-3-8B-Instruct 有显著 gain——**充分 post-trained 模型才能 leverage reference for open-ended tasks**。这个发现很重要：reference utilization 是一种能力，需要 post-training 来培养。

### 4.3 Inter-Judge Agreement（Appendix C.2, Table 18）

| Dataset | Ref-Free | RefEval | Diff |
|---------|---------|---------|------|
| LLMBar-Nat | 80.98 | 85.31 | +4.33 |
| MTBench | 75.35 | 82.42 | +7.07 |
| HREF | 74.17 | 81.83 | +7.66 |
| **Average** | **76.61** | **81.37** | **+4.76** |

Reference 提供了 **shared grounding**，显著降低了不同 judge 之间的 variance。这是 reference 起作用的另一个 mechanism——把 evaluation 从主观的 "absolute quality judgment" 变成更客观的 "similarity to anchor"。

### 4.4 Multi-Reference Voting（Figure 5, Appendix C.3）

用 Claude-3.5-Sonnet, Claude-3.7-Sonnet, Gemini-2.0-Flash, DeepSeek-V3, GPT-4o 五个 frontier 生成 reference，做 majority vote：
- Single best reference: 81.4%
- 5 references vote: 82.3%
- 边际收益递减（+0.4% per additional ref）

所以 paper 主实验用 single reference 已经足够 efficient。

---

## 5. Intuition Building：为什么这个方法 work？

### 5.1 与 RLVR 的类比

RLVR (e.g., DeepSeek-R1, Tulu 3) 的关键在于 **verifier 提供了 ground-truth distribution 的 sample**（math 的正确答案、code 的 passing test）。Policy gradient 实际上是在最大化 $p(y_{\text{correct}}|x)$ 的 likelihood。

在 alignment domain，没有 verifiable correct answer，但 **DeepSeek-V3 的回答是 "高质量回答分布的 sample"**。RefEval 的 prompting 等价于让 judge 学习一个 implicit function：

$$f(x, y, r) \approx P(y \text{ is at least as good as } r | x)$$

而不是绝对的 quality assessment $P(y \text{ is good} | x)$。前者更容易学，因为：
1. Relative comparison < absolute scoring
2. Reference 提供 anchor，减少 distribution shift
3. 把 open-ended 生成任务变成 reference-grounded matching

### 5.2 为什么 Self-Improvement Work

self-improvement (Wu et al. 2024, Yuan et al. 2024) 的本质是 **iterative self-distillation with self-generated preference labels**。Yuan et al. 的 Self-Rewarding LM 用 LLM-as-a-Judge 给自己生成的 candidate 打分。

paper 的关键 insight 是：**naive self-rewarding 受限于 judge 能力**——8B 模型 judgment 能力弱，导致 preference label noisy。Reference-guided judge 把 judgment 任务简化，让 8B judge 能力被 "amplified" 到接近 70B 水平。这是 reference 的 multiplier 效应。

### 5.3 与 RLAIF / Constitutional AI 的对比

Anthropic 的 RLAIF / Constitutional AI (Bai et al. 2022, https://arxiv.org/abs/2212.08073) 用 AI feedback 替代 human feedback，但仍需 external AI（typically stronger model）作为 feedback provider。

paper 的 setup 是 **semi-self-improvement**：reference 由外部 frontier 生成（DeepSeek-V3），但 preference label 由自己（被训练的模型）生成。这意味着：
- Reference 生成是一次性 cost（API 调用）
- Preference labeling 是 self-renewable，可以 on-policy 迭代

这是介于 pure RLAIF 和 pure self-improvement 之间的 sweet spot，工程上很 pragmatic。

### 5.4 与 Rubric-as-Reward 的对比

近期工作 Gunjal et al. 2025 "Rubrics as Rewards" (https://arxiv.org/abs/2507.17746) 也用 reference 信息，但用于 **construct rubric for RL**。paper 与之不同：直接用 reference 作为 judge 的 grounding，跳过 rubric 设计这一步。Rubric 是 discrete criteria list，而 reference 是 continuous exemplar——后者信息密度更高。

### 5.5 与 BLEU/ROUGE/RevisEval 的对比

RevisEval (Zhang et al. 2025, https://arxiv.org/abs/2410.05495) 让 LLM 生成 response-adapted reference 用于 evaluation。Chang et al. 2025 "BLEUberi" (https://arxiv.org/abs/2505.11080) 用 BLEU 作为 RL reward。

paper 在 Table 3 中直接对比了 ROUGE 和 BERTScore 作为 DPO reward——结果远差于 RefEval（ROUGE 56.4 vs RefEval 73.1 on Llama AE）。原因：
- BLEU/ROUGE 是 surface-level lexical metric
- BERTScore 是 embedding cosine similarity
- 都无法 capture instruction-following 的 semantic quality
- LLM-judge with reference 是 semantic-level matching

---

## 6. Limitations 与 Potential Issues

paper 没有充分讨论的问题：

1. **Distribution shift between reference generator and policy model**：DeepSeek-V3 的回答风格可能 dominant over policy model 自己的 exploration。Stage 1 SFT 已经把 policy 推向 DeepSeek-V3 分布，Stage 2 DPO 中的 self-judge 用同样的 reference 来 anchor——这可能加剧 mode collapse 到 DeepSeek-V3 风格。

2. **On-policy sampling cost**：60K × 10 pairwise = 600K LLM calls for judge。论文用 8B judge 还算可承受，但若 scale up 到 70B judge，cost 显著。

3. **Reference generator 的天花板**：RefEval pipeline 的上限受限于 DeepSeek-V3 能力。如果 reference 本身在某 domain 较弱（如 specialist domain），整个 pipeline 会 fail。Appendix C.1 用 GPT-4o-mini 验证了这点——绝对分数大幅下降。

4. **Evaluation benchmark leakage**：DeepSeek-V3 训练时可能见过 AlpacaEval/Arena-Hard 的 prompts。paper 没讨论这点。

5. **Self-improvement 的 stability**：paper 没做 multiple iterations of self-improvement，只做了单轮。Multi-turn self-improvement 是否会 compound error 或 collapse，未知。

---

## 7. 个人 Intuition 与可能的延伸方向

### 7.1 Reference as "soft verifier" 的 generalization

paper 的 framing：reference-guided LLM-judge 是 **soft verifier**，介于 RLVR 的 hard verifier 和 RLAIF 的 reward model 之间。

更 general 的视角：任何 **anchor signal**（reference, rubric, gold partial answer, expert demonstration）都可作为 soft verifier。可设计 spectrum：
- Hard verifier (RLVR): binary correctness
- Soft verifier (this paper): similarity to reference
- Reward model (RLHF): learned scalar reward
- Constitutional principle (CAI): rule-based scoring

### 7.2 Reference quality 的 information-theoretic view

reference 的 value 可量化为 $I(\text{reference}; \text{true quality})$。Frontier model reference 的 mutual information 高，weak model reference 的低。RefEval 即使在 weak reference 上仍有 gain，说明 prompting 机制本身提取了 residual information。

### 7.3 与 Process Reward Model (PRM) 的可能结合

paper 用 outcome-level reference（完整 answer）。若用 process-level reference（如 DeepSeek-V3 的 CoT trace），可构造 step-level soft verifier，用于 PRM 训练。这对 reasoning-heavy task 可能更有效。

### 7.4 Online vs Offline Reference

paper 是 offline reference（一次性生成）。如果 reference 可在 training loop 中 update（如 self-distillation 中的 EMA teacher），可构成 EM-like 算法。这是 self-improvement 与 reference-guided 的更深 integration。

### 7.5 Mixture of References for Diversity

Multi-ref voting（Appendix C.3）显示边际收益递减，但若 reference 来自不同能力维度的 frontier（e.g., DeepSeek-V3 for coding, Claude for writing, Gemini for math），而非同一类 frontier 的不同版本，diversity gain 可能更高。

---

## 8. 工程实践建议

如果你 Karpathy 想在 eureka labs / 实际项目中 use 这个方法：

1. **Stage 1 SFT 不可省略**：直接 DPO from base model 用 reference-guided judge 不如先 SFT on reference。Table 3 显示 DSV3-Distill 已经超过 ArmoRM-Base。

2. **Reference 数量不需要多**：single frontier reference 已经足够，multi-ref voting 边际收益小。

3. **On-policy sampling 是关键**：用 temperature=0.8 采样 5 个 candidate 做 pairwise，比用 static preference data 好。这与 Meng et al. SimPO 的发现一致。

4. **β (KL coefficient) 需要 grid search**：paper 在 0.005-0.1 范围 grid search。这是个 sensitive 超参。

5. **For small model (<10B)**：RefEval 比 reference-free 提升 15-22 points，ROI 极高。For 70B+ model，提升 marginal（5 points 左右），但 reference 帮助 frontier model 也获益，说明 quality ceiling 还未达到。

6. **Pairwise comparison 的 computational cost**：60K instructions × 10 pairs = 600K judge calls。用 8B judge 在 H100 上约几十 GPU-hours。是可承受 cost。

---

## 9. 与当前 frontier 工作 的 broader context

- **Tulu 3 (Lambert et al. 2024, https://arxiv.org/abs/2411.15124)**: 用 RLVR 但限于 verifiable domain。本方法 extend 到 non-verifiable domain。
- **DeepSeek-R1 (https://arxiv.org/abs/2501.12948)**: RL with verifiable rewards for reasoning。本方法是其 alignment counterpart。
- **Self-Rewarding LM (Yuan et al. 2024)**: pure self-improve without reference。本方法是其 reference-grounded 增强版。
- **SimPO (Meng et al. 2024, https://arxiv.org/abs/2405.14734)**: reference-free DPO variant。本方法 show reference-guided DPO 超过 SimPO。
- **GenRM (Zhang et al. 2024, https://arxiv.org/abs/2408.15240)**: LLM as generative reward model。本方法可视为 reference-conditioned GenRM。

paper 在 alignment post-training 方法论上提供了一个 lightweight 但 effective 的 recipe，特别适合 **resource-constrained 场景**（无法训练 large RM）和 **non-verifiable domain**（无法用 RLVR）。

---

## 10. 公式再深入：DPO 的 gradient 视角

DPO 的 gradient 可写作：

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \mathbb{E}_{(x,y_w,y_l)} \left[ \sigma(\hat{r}_\theta(y_l|x) - \hat{r}_\theta(y_w|x)) \cdot \nabla_\theta (\log p_\theta(y_w|x) - \log p_\theta(y_l|x)) \right]$$

其中 implicit reward $\hat{r}_\theta(y|x) = \beta \log \frac{p_\theta(y|x)}{p_{\text{ref}}(y|x)}$。

Reference-guided judge 提供的 preference label $(y_w, y_l)$ **决定了 gradient 的方向**。如果 judge accuracy 高，gradient 指向"高质量分布"的方向；如果 judge noisy，gradient 也 noisy。

paper 的 RefEval 把 judge accuracy 从 72.3% (Base) 提升到 79.1%，意味着 **error rate 从 27.7% 降到 20.9%**——preference label noise 减少 ~25%。这直接转化为 DPO 训练的 sample efficiency 提升，解释了 Stage 2 DPO 的 +5.6 AE gain over RefFree。

更精确地，DPO 对 preference noise 的 robustness 与 $\beta$ 相关——低 $\beta$ 对 noise 更敏感（KL constraint 弱，容易 overfit noise）。这解释了 paper 中 $\beta$ grid search 的必要性。

---

## 11. 总结

这篇 paper 在 alignment tuning 方法论上做了一个看似简单但 effective 的 contribution：**用 frontier LLM 的 reference output 作为 LLM-judge 的 grounding，使小模型 judge 能力提升到接近 frontier 水平，进而 enable effective self-improvement DPO**。

核心 mechanisms：
1. **Reference as anchor**：把 absolute quality judgment 退化为 relative similarity matching
2. **Shared grounding**：减少 inter-judge variance
3. **Soft verifier**：bridge RLVR 和 RLHF 的方法论 gap
4. **SFT + DPO 两阶段**：distillation 先锚定分布，DPO 再精调 preference

未来方向：
- Multi-iteration self-improvement with reference update
- Process-level reference for reasoning task
- Domain-specific reference selection
- Reference 的 information-theoretic 量化

相关 references：
- 主 paper (假设): https://arxiv.org/abs/2504.19087
- Self-Rewarding LM: https://arxiv.org/abs/2401.10020
- DPO: https://arxiv.org/abs/2305.18290
- LLMBar: https://arxiv.org/abs/2310.05470
- ArmoRM: https://arxiv.org/abs/2406.12845
- Tulu 3: https://arxiv.org/abs/2411.15124
- Constitutional AI: https://arxiv.org/abs/2212.08073
- SimPO: https://arxiv.org/abs/2405.14734
- GenRM: https://arxiv.org/abs/2408.15240
- RevisEval: https://arxiv.org/abs/2410.05495
- Rubrics as Rewards: https://arxiv.org/abs/2507.17746
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
