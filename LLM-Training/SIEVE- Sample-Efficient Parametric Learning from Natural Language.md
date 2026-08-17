---
source_pdf: SIEVE- Sample-Efficient Parametric Learning from Natural Language.pdf
paper_sha256: f2cb8062c0551c5ebf73d7e056a6c87c48c212699c2de493b6a2478fe4333184
processed_at: '2026-08-12T06:01:59-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SIEVE 用人话说一遍

## 一句话说清楚这篇 paper 在干嘛

你有一份规则书（比如 30 条折扣规则，或者一本语法书），想把这些规则"烤"进 LLM 的 weights 里，让模型以后不用每次看 prompt 就能照规则办事。但传统烤法需要几千条带答案的训练数据，没人有。SIEVE 给你个偷懒办法：**只要 3 个 query 例子 + 一段规则书，就能自动造出几万条训练数据，烤完之后推理时连规则都不用给**。在 Retail domain 上还反过来比把规则塞 prompt 里（ICL）还高 37.7%。

---

## 这个问题为什么一直没被解决

先把 ICL 和 parametric learning 两条路的痛点摆出来：

**ICL（in-context learning）** 就是每次推理把规则全塞 prompt。好处：sample efficient，扔 3 个例子进去就能 work。坏处：每次推理都要重新"看"一遍规则，50K token 的 grammar book 每次都要 prefill，又贵又慢，还受 context window 限制，规则越多 attention 越稀释。

**Parametric learning / context distillation** 就是把规则烤进 weights。好处：persistent，跨 session 不丢，推理时 prompt 很短。坏处：data hungry。Snell 2022 那篇 *Learning by Distilling Context* 就要一大堆 query distribution 才能 distill，Bhargava 2024 的 Prompt Baking 也一样。实际场景里你拿不到几千条 query。

中间一直有个 gap：ICL 省样本，parametric 省推理。两者你只能选一个。SIEVE 想同时拿到两边的好处。

---

## 核心 insight，就一句话

**一段 context 里，每个问题其实只用到一小撮规则。**

听起来像废话，但是这条观察直接改变了你怎么造训练数据。举个例子，Retail 有 30 条 discount rule，某个 query 是"senior citizen 买了 $85 鞋子 + $60 外套 + $45 咖啡机"。真正适用的可能就 3 条：

1. senior citizen 且 spend ≥ $50，total 15% off
2. apparel 类 spend ≥ $75，category 10% off
3. member 4 年（≥3 年档），total 10% off

剩下 27 条对这个 query 是 noise。

传统 context distillation 不管这个，把全部 30 条塞 teacher prompt 去 rollout。teacher 模型一边生成答案，一边要在 30 条里"挑"，经常挑错、被无关规则干扰、甚至 hallucinate 用错规则。这种脏 rollout 拿去训练 student，student 学的是被污染的 reasoning。

SIEVE 的操作很直接：**生成训练数据时，每个 query 只配它真正适用的那几条规则**。teacher 看到的 context 是干净的，rollout 就干净，student 学到的就是干净 reasoning。就这么个 insight，把 sample efficiency 从"需要几千 query"拉到"需要 3 个 query"。

---

## SIEVE-GEN 怎么造训练数据（核心 pipeline）

整个 pipeline 在 Figure 1，我把它拆成 4 步用人话讲：

### Step 1：拆 context

把整段规则书丢给 instruct model，让它拆成 atomic unit。每个 unit 是一条自包含、可独立判断适用性的规则。Retail 的 30 条 rule 就拆成 30 个 unit，比如：

> "If customer is a student AND total spend is at least $50, apply 10% discount to total purchase"

拆完你拿到一个 set $\mathcal{C} = \{u_1, u_2, \ldots, u_{30}\}$。Prompt 在 Appendix C.1，要求每个 item 必须 "evaluable independently"，也就是单看这一条就能判断它对某 query 是否 applicable。

长 context 场景（MTOB 50K token）超 context window，就 chunk 一下：8192 token 一块，512 overlap，每块独立拆。

### Step 2：反向造 query（backtranslation）

这一步是最巧妙的。传统 synthetic data 是"给一个 query 生成答案"，SIEVE 是"给一些规则反向生成一个会用得上这些规则的 query"。

具体两小步：

**(a) 选 seed context**：从 $\{u_1, \ldots, u_n\}$ 里 sample 一个 subset $c_\text{seed}$ 作为种子。

**关键 trick**：这里要用 **base model**（纯 pretrained，没做 instruction tuning 的）来 sample，不用 instruct model。为什么？BARE (Zhu 2025) 发现 instruct model 会 mode collapse，每次选几乎一模一样的 subset，生成的 query 覆盖面极窄。Base model 的分布保留更多 entropy，能选到长尾组合，覆盖更广的规则空间。

这个观察其实挺反直觉的，一般人都觉得 instruct model 更好用，结果 synthetic data generation 里 base model 反而更合适，因为你要的是 diversity 不是 helpfulness。

**(b) 生成 query**：用 instruct model 根据 $c_\text{seed}$ 加上 3 个 example query，生成一个新的 query $q$。例子主要起格式参考作用，告诉模型"query 长这样"。

### Step 3：验证哪些规则真的适用

刚才反向生成的 query，它真正需要的规则可能跟 seed 不完全一致 —— 可能需要再加几条，也可能 seed 里有冗余。所以让 instruct model 逐个检查每个 $u_i$："这条规则对回答这个 query 是 necessary 的吗？" 输出 binary decision，得到 verified applicable context $c_a \subseteq \mathcal{C}$。

这一步是 SIEVE 和 "naive synthetic data" 的本质区别。naive 方法直接把 seed 当成 applicable context 用，SIEVE 多了一道 verification，把 noise 过滤掉。长 context 时 verification compute 太大，就改成 batch 模式（一次评估多个 unit）。

MTOB 这个 domain 跳过了 verification，因为 Kalamang 是极低资源语言，模型本身没 background knowledge，query 里能涉及的概念只能是 seed 里有的，不会有"额外需要的规则"。这是 domain-specific 的简化。

### Step 4：Rollout + Distillation

拿到 $(q, c_a)$ pair 之后：

**Rollout**：teacher model 看 query + applicable context 生成答案 $r = M_\text{inst}([q, c_a])$。

**Distillation**：student model 只看 query $q$，训练它去模仿 teacher 的输出分布。具体 loss 用 forward KL，teacher 那边保留 top-K=100 logits 形成 truncated soft target。后面公式部分会展开。

---

## 公式部分：Context Distillation Objective 讲清楚

这部分基于 Snell 2022。核心是让 student 在没 context 的情况下，输出分布去匹配 teacher 在有 context 情况下的输出分布。

### Teacher 端

Teacher 的输入是 query 和 applicable context 拼起来：

$$
c = [q; c_a] \quad \text{(Eq. 1)}
$$

- $q$：query 文本
- $c_a$：applicable context（就是 SIEVE-GEN 验证后筛出的几条规则）
- $[;]$：文本拼接
- $c$：teacher 的完整 input

Teacher 在这个 input 下产生 token 级别的概率分布：

$$
p_T(y \mid q, c_a) = M_\theta(c) \quad \text{(Eq. 2)}
$$

- $p_T$：teacher 的输出分布
- $y$：要生成的 response（一串 token）
- $M_\theta$：原始模型，参数 $\theta$
- 模型是 Qwen3 8B（或者 Llama 3.1 8B / Rnj 1 8B，看实验）

完整 vocab 通常 15 万 token 量级，做 KL 全 vocab 太贵。所以截断到 top-K logits：

$$
\tilde{p}_T(y \mid q, c_a) = \text{TopK}\big(p_T(y \mid q, c_a)\big), \quad K=100
$$

- $\tilde{p}_T$：truncated soft target
- $K=100$：只保留 logits 排前 100 的 token，剩下的概率 mass 抹掉，重新归一化

为什么 top-K=100 够？因为真正承载 preference 信号的 token 集中在 top 几十个，长尾的 noise token 留着只会增加优化方差。Snell 2022 已经验证过这个数。

### Student 端

Student 输入只有 query，没 context：

$$
M_\phi(y \mid q)
$$

- $M_\phi$：student model，参数 $\phi$
- 论文里 student 和 teacher 是同款 model（self-distillation），只是输入不同
- 注意 paper 试过 LoRA，发现效果比 full FT 差，所以用 full FT

### Loss

$$
\mathcal{L}_\text{CD} = \mathrm{KL}\big(\tilde{p}_T(y \mid q, c_a) \,\|\, M_\phi(y \mid q)\big) \quad \text{(Eq. 3)}
$$

- $\mathcal{L}_\text{CD}$：context distillation loss
- $\mathrm{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$：forward KL，$P$ 是 teacher truncated 分布，$Q$ 是 student 分布
- 这个是 forward KL（teacher 在前），不是 reverse KL

**为什么用 forward KL？** Forward KL 是 mean-seeking 的，student 会被推着去覆盖 teacher 所有高概率 mode，避免 mode collapse 到某个单一 token。如果用 hard label（argmax），student 会过拟合到 teacher 偶尔的 bad sample；用 soft label + forward KL，teacher 的 noise 会被平均化掉。这是 SIEVE 的一个 implicit robustness 来源。

**为什么不用 cross-entropy on hard label？** 因为 teacher rollout 不是 ground truth，它有错。Hard label 假设 teacher 永远对，soft label 承认 teacher 有 uncertainty，把这个 uncertainty 传给 student。

### 训练超参（Table 3）

| Hyperparameter | Value |
|---|---|
| Learning Rate | $1 \times 10^{-5}$ |
| Batch Size per device | 1 |
| Gradient Accumulation | 8 |
| Effective Batch Size | 64 |
| Temperature $\tau$ | 1.0 |
| Warmup Steps | 50 |
| Top-K Tokens | 100 |
| Max Sequence Length | 16384 |
| Optimizer | AdamW |
| DeepSpeed | ZeRO-3 |
| GPU | 8× H100 |
| Epochs | Retail 2 / NBA 2 / MTOB 5 |

LR $1e-5$ 是 distillation 通行范围，相对保守，防止 catastrophic forgetting。Max seq 16K 对 Retail / NBA 够，MTOB 那种 50K 用 chunking 处理。

---

## 三个 Domain：测什么、怎么测

### Retail（作者自建）

**任务**：30 条 discount rule，给一个 shopping cart，算最终价格。规则之间组合叠加，按固定顺序 apply：
1. category-specific % discount（每类取最高，apply 到 category subtotal）
2. total purchase % discount（取最高，apply 到 step 1 剩余）
3. fixed amount discount（全加起来，从 step 2 剩余里减）

**测试要素**：6 种 customer type，8 种 product category，7 个 promo code，3 种 membership tier。256 个 programmatic query，ground truth 精确知道。

**Metric**：误差 $\leq 0.01$ 的 binary accuracy。这个 task 真正测的是 compositional reasoning —— 模型要在 30 条里挑对 applicable 的几条，还要按顺序组合 apply。

**为什么这个 domain 重要**：它要求模型在 inference 时 "compose" 内部化的 rule，单纯 memorize 没用。

### RuleArena (NBA)

来自 Zhou 2025，~20K token 的 NBA CBA trade 规则，判断一串 trade 操作是否 illegal 并说明原因。Level 2 是最难档。Metric 是 legality 判断 + violation ID 的 exact match。

跟 Retail 互补：Retail 规则短但组合复杂，NBA 规则长但 query 触发的规则相对少。

### MTOB

Tanzer 2024，翻译极低资源语言 Kalamang → English。50K token grammar book + 375 句 parallel examples，超过 32K context window，ICL baseline 要用 2× RoPE scaling。

Metric 是 chrF（character n-gram F-score）。这个 domain 偏 memorization 不是 compositional reasoning，跟前两个互补。Cartridges 这个 baseline 只在这个 domain 用，因为它本来就是为长文档 fact recall 设计的。

---

## 实验结果，关键数字摆出来

### SIEVE 量越大越好（Figure 2）

固定 3 个 seed query，scale SIEVE-GEN 到 16K synthetic data：

| Domain | 8K data | 16K data | ICL baseline |
|---|---|---|---|
| Retail | ~33% | ~36% | ~26% |
| NBA | 接近 ICL | match ICL | matched |
| MTOB | 低于 ICL | match ICL | baseline |

Retail 上 SIEVE 超 ICL 37.7%，这是 paper 最 strong 的 claim。机制上理解：ICL 时模型要在 30 条规则里 attention search，rule 越多 attention 越稀释；internalize 后 rule 变成 weights 里的"先验"，模型直接调用，组合应用反而比 prompt search 流畅。

### 对比 baseline context distillation（Figure 3）

三个 baseline：

- $V_\text{CD}$（3 seeds）：传统 CD，只给 3 个 seed query + 全 context
- $V_\text{CD-S}$（8K，no filter）：用 SIEVE 的 synthetic query，但 rollout 时塞全 context 不做 filtering —— 这其实是 SIEVE 的 ablation
- Cartridges（只在 MTOB）：Eyuboglu 2025，长文档 KV cache 注入

| Domain | $V_\text{CD}$ 3 seeds | $V_\text{CD-S}$ 8K | SIEVE | ICL |
|---|---|---|---|---|
| Retail | 3% | 30% | **36%** | 26% |
| NBA | low | -10% vs SIEVE | **+10% over $V_\text{CD-S}$** | matched |
| MTOB | infeasible | - | **24.48 chrF** vs Cartridges 19.10 | higher |

最 striking 的 ablation 是 $V_\text{CD-S}$ vs SIEVE：同样的 synthetic query，只是 rollout 时塞全 context 而不是 applicable-only，Retail 从 36% 掉到 30%。这是 "applicable context filtering" 单独贡献 6 个点。

### Oracle Query Experiment（Table 1）— 最重要的一张表

为了 isolate "filtering" 的作用，作者做了一个极端对比：

- Vanilla CD 用 programmatic 生成的 **perfect ground-truth query**（分布和 eval 完全一致），但 rollout 塞全 context
- SIEVE 用 synthetic query，rollout 塞 applicable context
- 都用 4096 examples

| Method | Accuracy |
|---|---|
| Vanilla CD (oracle queries, all context) | 27.11% |
| SIEVE (synthetic queries, applicable context) | **33.98%** |

**这个表的意思**：oracle query + 全 context，输给 synthetic query + applicable context，差 6.87%。换句话说，**training data 的 purity 比 query 的真实性还重要**。这是 SIEVE 最反直觉也最有用的发现。你不用花大钱请专家写 query，只要把 rollout 的 context 弄干净，效果就比完美 query + 脏 context 强。

### Multiple Rollouts vs Distinct Queries（Table 2）

Guha 2025 (OpenThoughts) 说每个 query 生成多个 rollout 能提升 distillation 效果。SIEVE 在 Retail 上做 controlled tradeoff，固定 total generation 量：

| Setting | Distinct Queries | Accuracy |
|---|---|---|
| 512×8 | 512 | 30.23% |
| 4096×1 | 4096 | 33.98% |
| 1024×8 | 1024 | 37.97% |
| 8192×1 | 8192 | 35.78% |

**怎么读这个表**：
- 4096×1 > 512×8：低 data regime 下 distinct query diversity 比 multiple rollouts 重要，多样 query 比重复采样同 query 更有用
- 1024×8 > 8192×1：高 data regime 下 diversity 饱和，multiple rollouts 反超，因为 stochastic sampling 能让 student 看到同一 query 的多种解法

实操 recipe：先把 distinct query scale 到 saturation（~1024 这种量级），再额外 compute 投入到 per-query multiple sampling。

### Model Family Generalization（Figure 4）

在 Retail 上换 base model：

| Model | ICL | SIEVE (8K) |
|---|---|---|
| Qwen3 8B | 26% | 36% |
| Rnj 1 8B | 13.98% | 17.03% |
| Llama 3.1 8B | 4.53% | 3.44% |

**Llama 3.1 8B 失败**。SIEVE 训完反而比 ICL 低。但你看 Llama 3.1 8B 的 ICL 自己就只有 4.53%，说明它 base reasoning 能力不足以处理这个 compositional task。

这是 SIEVE 的 hard floor：**base model 必须对 target domain 有 reasonable 能力**。SIEVE 不是凭空创造能力，是放大已有能力。Base model 自己用 context 都做不好的事，烤进 weights 也做不好。

---

## 跟相关工作的关系，简单理一下

| Work | 关系 |
|---|---|
| Snell 2022 (Learning by Distilling Context) | SIEVE 的 distillation objective 直接来自这，公式 3 就是 Snell 那篇的。但 Snell 假设有 query distribution，SIEVE 只要 3 个 seed |
| Bhargava 2024 (Prompt Baking) | 另一种 context distillation variant，同样 data hungry |
| Eyuboglu 2025 (Cartridges) | 长文档 KV cache 注入，主攻 fact recall；SIEVE 攻 compositional reasoning |
| Lin 2025 (Active Reading), Yang 2024 (Synthetic Continued Pretraining) | 用 synthetic data 注入 facts，目标是 memorization 不是 reasoning |
| Zhu 2025 (BARE) | SIEVE 用 base model 做 seed sampling 的依据，BARE 发现 base model 比 instruct model 多样性好 |
| Scheurer 2022, Chen 2024 (NL Feedback) | 用文本 feedback 做 training，但要 expert traces / verifiers，SIEVE 不要 |
| Shenfeld 2026, Hubotter 2026 (Self-distillation RL) | 同期 self-distillation 工作，paradigm 不同 |

SIEVE 的 niche：**没有 query distribution、没有 expert traces、没有 verifiers，只有 context + 3 个 query 例子**。所有 prior work 至少要其中一项，SIEVE 全部省掉。

---

## 我的几个额外思考

1. **Verification 步骤本身的可靠性没量化**。SIEVE-GEN Step 3 用 LLM 判断 context unit 是否 applicable，这本身是个 LLM judgment。如果 instruct model 判断错（漏判或者多判），rollout 还是会被污染。论文没给 verification accuracy 的 ablation。改进方向：用 self-consistency 多次 verify，或者引入 verifier model 跟 generator 解耦。

2. **Decomposition 的 granularity 没定义清楚**。什么是 "atomic"？一条 rule 可以 atomic，但 "customer is a student AND total spend ≥ $50" 这种 conjunction 是不是该再拆？Appendix C.1 的 prompt 让模型自己决定，没有 quality metric。不同 decomposition 粒度可能效果差很多。

3. **Compositional generalization 测试其实不够**。Retail 用 30 rule，训练和 eval 的 query 都从这些 rule 组合生成，是 *interpolation* 测试。真正的 OOD compositional generalization（用没见过的 rule 组合）没测。如果训练时模型没见过 rule A 和 rule C 同时触发，eval 时让它俩同时触发，能不能 generalize？这个不知道。

4. **Self-distillation 的 ceiling 问题**。Rollout 是同款 model 生成的（"We sample rollouts for distillation from the same model we train in all experiments"）。如果 base model 在某条 rule 上推理就错，SIEVE 会把这个错误蒸馏进 weights，没 verifier 纠正。论文 future work 提到 decouple generator 和 student，但没实验。这是和 RLHF 路线最大的差距。

5. **Compute 成本没量化**。SIEVE-GEN 生成 16K data 要多少 H100 hours？Verification phase 要对每个 query 逐个检查每个 context unit，n × N 次 LLM call，量级可能很大。Long context extension 部分已经承认 compute 大才改成 batch 模式，但主实验没给 wall clock / 美元成本数字。

6. **Inference cost 节省没量化**。Parametric learning 的卖点之一是省 inference cost。MTOB 那种 50K token ICL baseline 推理一个 query 就要烧 50K token prefill，SIEVE 只要 query 本身的 token。这个 delta 对 production 是巨大的，但 paper 没给 latency / token cost 对比表。这是 missing 但很重要的数字。

7. **跟 Anthropic Constitutional AI 的 hidden link**。SIEVE 的 decompose + verify 跟 Constitutional AI 的 critique-revise 范式结构上很像。如果把 SIEVE-GEN 的 verification 改成 "critique rollout with applicable context"，可能进一步提升 quality。这是潜在的改进路径。

8. **Mixture-of-Adapter 方向**。Cartridges 是塞 KV cache，SIEVE 是塞 weights，中间态可能是按 chunk 学多个 LoRA / adapter，inference 时 routing。Paper future work 提到 specialized memory layers，方向正确但没实验。

---

## 把 Intuition 压成几条

1. **Context 是稀疏激活的**，每个 query 只触发一小撮规则。SIEVE 把这个 sparsity 显式化用于 training data 构造。

2. **Training data 的 purity 比 query 的真实性还重要**。Oracle 实验证明，perfect query + 全 context 输给 synthetic query + applicable context 6.87 个点。

3. **Base model 比 instruct model 多样**，做 synthetic data generation 时 sampler 选 base model，因为 instruct model entropy collapse，覆盖面窄。这个反直觉但已被 BARE 验证。

4. **Forward KL + top-K soft label 抗 teacher noise**，比 hard label 鲁棒，不会过拟合到 teacher 偶尔的坏 sample。

5. **Parametric 在 compositional reasoning 上反而能超 ICL**。Retail 上 SIEVE 超 ICL 37.7% 说明 attention 在长 context 上 search rule 是低效的，把 rule internalize 成 weights 让模型"自动"组合应用。这跟 Hopfield network 的 attractor 机制有点像 —— 学完之后 retrieval 比 online search 快且准。

6. **Base model capability 是 hard floor**。Llama 3.1 8B 在 Retail ICL 自己就 4.53%，SIEVE 救不回来。SIEVE 放大已有能力，不凭空创造能力。

7. **Sample efficiency 的代价是 self-distillation ceiling**。没有 verifier 纠正，teacher 的错误会被烤进 student。要打破这个 ceiling 得引入外部 verifier 或 stronger teacher。

---

## 参考链接

- SIEVE 原文：作者 Parth Asawa, Alexandros G. Dimakis, Matei Zaharia，标题 "SIEVE: Sample-Efficient Parametric Learning from Natural Language"，可在 https://arxiv.org/ 搜索
- Context Distillation (Snell 2022): https://arxiv.org/abs/2209.15189
- Cartridges (Eyuboglu 2025): https://arxiv.org/abs/2506.06266
- BARE (Zhu 2025): https://arxiv.org/abs/2502.01697
- RuleArena (Zhou 2025): https://arxiv.org/abs/2412.08972 ，repo: https://github.com/SkyRiver-2000/RuleArena
- MTOB (Tanzer 2024): https://arxiv.org/abs/2309.16575
- Prompt Baking (Bhargava 2024): https://arxiv.org/abs/2409.13697
- Active Reading (Lin 2025): https://arxiv.org/abs/2508.09494
- Synthetic Continued Pretraining (Yang 2024): https://arxiv.org/abs/2409.07431
- Training with NL Feedback (Scheurer 2022): https://arxiv.org/abs/2204.14146
- Qwen3 Tech Report: https://arxiv.org/abs/2505.09388
- Llama 3: https://arxiv.org/abs/2407.21783
- Rnj-1 (Essential AI): https://huggingface.co/EssentialAI/rnj-1
- OpenThoughts (Guha 2025): https://arxiv.org/abs/2506.04178
- Self-Distillation Enables Continual Learning (Shenfeld 2026): https://arxiv.org/abs/2601.19897
- Reinforcement Learning via Self-Distillation (Hubotter 2026): https://arxiv.org/abs/2601.20802

---

## 最后一句

SIEVE 把 "context 是稀疏激活的" 这个 insight 翻译成 "training data 应该只包含 applicable context" 这条 mechanical rule，加 base-model sampling for diversity，让 context distillation 从"需要大量 expert data"退化成"需要 3 个 seed + 一段 context"。在 Retail / NBA / MTOB 三个 verifier-friendly domain 上验证，给出了 parametric learning 第一次在 sample efficiency 上追平甚至超过 ICL 的 strong evidence。Llama 3.1 8B 的失败提示 base model capability 是 floor，self-distillation 是 ceiling —— 这两面墙是未来要拆的。

---

# SIEVE: Sample-Efficient Parametric Learning from Natural Language — 深度讲解

## 1. 核心问题：ICL vs Parametric Learning 的二分

这篇 paper 来自 UC Berkeley / Sky Computing Lab (Parth Asawa, Alexandros G. Dimakis, Matei Zaharia)。它要打破一个非常古老的二分法：

- **In-Context Learning (ICL)**：sample efficient (3 个例子就能 work)，但无法 persist 到 weights，受 context window 限制，每次推理都要重复消耗 KV cache。
- **Parametric Learning / Context Distillation**：能 persist 到 weights，但是 data hungry，需要大量 query examples 或者 expert traces + verifiers。

SIEVE 的目标是：用 **3 个 query 例子**，把一大段 natural language context (规则书、grammar book) 烤进 model weights，推理时完全不需要 context，性能还能 match 甚至超过 ICL baseline。

这个 setting 在实际中非常重要 —— 一个 developer 写了一份 system prompt 想长期复用，或者一个 domain expert 想把 medical/legal knowledge 永久注入给模型，传统方案要么每次 prompt 巨长，要么需要花大价钱请专家写几千条 traces。SIEVE 给出第三条路。

---

## 2. 核心 insight：Context Decomposability

这篇文章最关键的一行话藏在 Section 1 里：

> Natural language context is decomposable. Natural language context often consists of independent context units (u) where only a subset applies to any given query.

也就是说，一段 context C 实际上是一组 atomic units：

$$
\mathcal{C} = \{u_1, u_2, \ldots, u_n\}
$$

每个 query 只需要其中一小部分 $c_a \subseteq \mathcal{C}$。

**为什么这个 insight 重要？** 之前的 context distillation 工作 (Snell 2022, Bhargava 2024) 都是直接把 *整段* context 塞进 prompt 去生成 teacher rollout。如果 context 有 30 条规则，但某个 query 只用得上 3 条，teacher rollout 的时候模型其实在被 27 条无关 context "污染"——它会犹豫、混入不相关 rule、可能直接 hallucinate 用错规则。这种 rollout 拿去当 training data，自然质量很差。

SIEVE 的关键操作：**生成 (query, *applicable-only context*) pairs**。Teacher 只看到它该看到的 context，rollout 质量就高得多，distill 出来的 student 也准。这个 insight 在 Table 1 里得到非常漂亮的验证（后面会讲）。

---

## 3. SIEVE-GEN：三阶段 Synthetic Data Pipeline

整个 pipeline 见 Figure 1，可以拆成 4 个 step：

### Step 1: 收集输入
- 一段 natural language context $\mathcal{C}$（rules / grammar / instructions）
- 三个 seed query examples $[e_1, e_2, e_3]$（主要做格式参考）

### Step 2: SIEVE-GEN 生成 (q, c_a) pairs
分 3 个 sub-phase：

**(a) Decomposition**
用 instruction-tuned model $M_\text{inst}$ 把 $\mathcal{C}$ 拆成 atomic units $\{u_1, \ldots, u_n\}$。每个 unit 必须 self-contained，可独立评估 applicability。Prompt 在 Appendix C.1：

> Break down the following feedback/guidelines/knowledge into atomic, independent items. Each atomic item should: 1. Express a single, self-contained rule, fact, definition, or example 2. Be evaluable independently 3. Preserve the exact meaning and wording from the original

举例 Retail domain 的 30 条 discount rule 被拆成 30 个 unit，每个 unit 长这样：
> "If customer is a student AND total spend is at least $50, apply 10% discount to total purchase"

**(b) Backtranslation**
这是最巧妙的一步，从 context 反向生成 query：

1. 用 **base model** $M_\text{base}$（只做 next-token prediction 的预训练模型）sample 一个 subset 作为 seed：$c_\text{seed} \subseteq \{u_1, \ldots, u_n\}$。
2. 用 $M_\text{inst}$ 根据 $c_\text{seed}$ 和 examples 生成 query q。

**为什么用 base model？** 这是论文引 BARE (Zhu et al. 2025) 的发现 —— instruction-tuned model 会 mode collapse，几乎每次选一模一样的 subset，导致 synthetic query 覆盖不到整个 context 空间。Base model 的 distribution 更 "raw"，能产生多样 seed，进而覆盖更广的规则组合。

这个观察非常重要，它告诉我们：synthetic data generation 里，"diversity" 不是温度调高一点就能搞定的，而是要选对 sampler。Instruction tuning 把 model 的分布往 average mode 拉，base model 保留更多 entropy tail。

**(c) Verification**
生成了 query 之后，用 $M_\text{inst}$ 逐个检查每个 $u_i$ 是否 applicable，得到 verified $c_a \subseteq \mathcal{C}$。Prompt 在 Appendix C，判断的是 "this context unit is necessary to answer the query"。

这一步为什么必要？因为 query 生成时用的 seed context 不一定就是 query 真正需要的 context —— 可能少加了几条规则，也可能 seed 里有冗余。

**Long context 扩展**（针对 MTOB 这种 50K token）：context 按 8192 token 一块 chunking，512 token overlap，每块独立 decomposition。Verification 改成 batch 模式（一次评估多个 unit），减少 compute。

### Step 3: Rollout
$$
r = M_\text{inst}([q, c_a])
$$
用 teacher model (同款 model) 在有 context 的情况下生成 response。

### Step 4: Context Distillation
Student model $M_\phi$ 只看 query $q$，被训练去模仿 teacher 在 $(q, c_a)$ 下的 distribution。

---

## 4. Context Distillation Objective — 公式拆解

这部分基于 Snell et al. 2022 的 *Learning by Distilling Context*。

### Teacher 输入和 distribution

把 query 和 applicable context 拼成 teacher input：

$$
c = [q; c_a] \quad \text{(Eq. 1)}
$$

其中 `;` 表示 concatenation。teacher 分布为：

$$
p_T(y \mid q, c_a) = M_\theta(c) \quad \text{(Eq. 2)}
$$

- $p_T$：teacher 在 token 级别的输出概率分布
- $y$：要生成的 response sequence
- $M_\theta$：参数为 $\theta$ 的原始模型
- $c$：拼好的 teacher input

### Top-K truncation

完整 vocab 通常 15 万 token 量级，做 KL 全 vocab 太贵。所以保留 logits 最大的 top-K（这里 $K=100$）形成 truncated soft target：

$$
\tilde{p}_T(y \mid q, c_a) = \text{TopK}(p_T(y \mid q, c_a))
$$

这个 truncated distribution 既保留了 fine-grained preference（不只是 argmax 的 hard label），又保持计算高效。$K=100$ 跟着 Snell 2022 设定。

### Student 训练 loss

Student $M_\phi$ 输入只有 $q$（没有 context $c_a$），目标是匹配 teacher 的 truncated distribution：

$$
\mathcal{L}_\text{CD} = \mathrm{KL}\big(\tilde{p}_T(y \mid q, c_a) \,\|\, M_\phi(y \mid q)\big) \quad \text{(Eq. 3)}
$$

- $\mathcal{L}_\text{CD}$：context distillation loss
- $\mathrm{KL}(P \| Q)$：forward KL divergence，$Q$ 是 student distribution $M_\phi(y|q)$，$P$ 是 truncated teacher distribution
- $M_\phi(y|q)$：student model 在只看 query 时的输出分布
- $\phi$：student model 参数（paper 中 full FT，不是 LoRA —— 因为他们尝试 LoRA 表现更差）

**Intuition**：forward KL 是 "mean-seeking"，student 会倾向于把概率分到 teacher 给过高概率的多个 mode 上，避免 mode collapse。这就是为什么用 soft label 而不是 hard argmax label —— hard label 模式下 student 会过拟合到 teacher 偶尔的 noise 上。

### 训练超参（Table 3）

| Hyperparameter | Value |
|---|---|
| Learning Rate | $1 \times 10^{-5}$ |
| Effective Batch Size | 64 (1 per device × 8 GPU × 8 grad accum) |
| Temperature $\tau$ | 1.0 |
| Warmup Steps | 50 |
| Top-K Tokens | 100 |
| Max Sequence Length | 16,384 |
| Optimizer | AdamW |
| DeepSpeed | ZeRO-3 |
| GPU | 8× H100 |

Epochs: Retail=2, NBA=2, MTOB=5。

---

## 5. Evaluation Domains — 三个测试设置

作者故意挑了三个非常不同、都能 verify 的 domain：

### 5.1 Retail (作者自建，compositional reasoning)
- 30 条 discount rule，组合叠加（category discount → total % discount → fixed discount，按顺序 apply 到 running total）
- 6 种 customer type，8 种 product category，7 个 promo code，3 种 membership tier
- 256 个 programmatically generated query（ground truth 精确知道）
- Accuracy: 误差 $\leq 0.01$ 的 binary exact accuracy

这个 domain 测试的是：**模型需要内部化 30 条 rule，并在 inference 时组合应用**。这是 compositional reasoning，不是 fact recall。

### 5.2 RuleArena (NBA)
- 来自 Zhou et al. 2025，原 benchmark 是 instruction following
- ~20K token 的 NBA CBA (Collective Bargaining Agreement) trade rules
- 判断一串 trade 操作是否 illegal 并指出原因
- Level 2 setting（最难）
- Exact match on legality + violation ID

### 5.3 MTOB (Machine Translation from One Book)
- Tanzer et al. 2024，translate 极低资源语言 Kalamang → English
- ~50K token grammar book + 375 句 parallel examples
- 超出 32K context window，需要 2× RoPE scaling 做 ICL baseline
- Metric: chrF (character n-gram F-score)
- 这个 domain 偏向 memorization，是前两个的 complementary

---

## 6. 主要实验结果

### 6.1 Scaling data (Figure 2)

用 fixed 3 个 seed query，scale SIEVE-GEN 生成到 16K (query, c_a) tuples。跨 domain 一致观察：

| Domain | 8K data | 16K data | ICL baseline |
|---|---|---|---|
| Retail | ~33% | ~36% | ~26% (SIEVE **超越 37.7%**) |
| NBA | match ICL | match ICL | match ICL |
| MTOB | 低于 ICL | match ICL | baseline |

重点：Retail domain 上 SIEVE 在 16K synthetic data 时 *超越* ICL 37.7%。这是论文一个 strong claim —— internalize 进 weights 反而比塞 prompt 里好用。

**为什么 parametric 能超过 ICL？** 一种解释：in-context 时模型要在 prompt 里 "search" relevant rule，长 context 会稀释 attention；internalize 后，rule 变成"先验"，模型直接调取，组合应用更流畅。这个其实跟 Anthropic 的 Cartridges paper 有类似观察。

### 6.2 与 baseline context distillation 比较 (Figure 3)

三个 baseline：

1. **$V_\text{CD}$ (3 seeds)**：传统 context distillation，只给 3 个 seed query + 全 context，train 到 converge
2. **$V_\text{CD-S}$ (8K)**：给 baseline 用 SIEVE-GEN 生成的 8K synthetic query，但 rollout 时塞 *全 context*（不过滤 applicable）—— 这其实是 SIEVE 的 "no filtering" ablation
3. **Cartridges** (只在 MTOB 用)：Eyuboglu et al. 2025，针对长 context 设计的 KV cache 注入方法

| Domain | $V_\text{CD}$ (3 seeds) | $V_\text{CD-S}$ (8K, no filter) | SIEVE | ICL |
|---|---|---|---|---|
| Retail | 3% | 30% | **36%** | 26% |
| NBA | (low) | -10% vs SIEVE | **+10% over $V_\text{CD-S}$** | matched |
| MTOB | infeasible (50K > 32K ctx) | - | **24.48 chrF** | higher |
| MTOB (Cartridges baseline) | - | - | 24.48 vs 19.10 chrF | higher |

**关键 insight from $V_\text{CD-S}$ ablation**：即使给 baseline 同样 8K synthetic query，只要 rollout 时塞全 context 而不是 applicable-only，Retail 从 36% 掉到 30%。这证明 "selective context filtering" 本身就是 SIEVE 的核心贡献，比单纯多生成 query 重要。

### 6.3 Oracle Query Experiment (Table 1) — 最重要的一张表

为了彻底 isolate "applicable context filtering" 的作用，作者做了一个 oracle 实验：

- 用 programmatic pipeline 生成 perfect ground-truth query（分布和 eval 完全一致）
- 给 vanilla CD 用这些 oracle query 训练，rollout 时塞全 context
- vs. SIEVE 用 synthetic query 训练，rollout 时塞 applicable context
- 都用 4096 examples

| Method | Mean Accuracy (%) |
|---|---|
| Vanilla CD (oracle queries, all context) | 27.11 |
| SIEVE (synthetic queries, applicable context) | **33.98** |

**这个结果极其 striking**：oracle query + 全 context 训练，反而输给 synthetic query + applicable context 训练，差 6.87%。这说明 query 质量 (perfection) 远没有 "context cleanliness" 重要。这也是为什么 SIEVE 比简单做大 synthetic data pipeline 强 —— 它对 training data 的 *purity* 做了根本性改进。

### 6.4 Multiple Rollouts vs Scaling Queries (Table 2)

Guha et al. 2025 (OpenThoughts) 提出每个 query 生成 multiple rollouts 可以不增加 distinct data 就提升 distillation。SIEVE 在 Retail 上做 controlled tradeoff：

| Setting | Distinct Queries | Accuracy (%) |
|---|---|---|
| 512×8 (8 rollouts each) | 512 | 30.23 |
| 4096×1 | 4096 | 33.98 |
| 1024×8 | 1024 | 37.97 |
| 8192×1 | 8192 | 35.78 |

**Interpretation**：
- 低 data regime 下 (512 vs 4096)，distinct query diversity 比 multiple rollouts 重要 (4096×1 > 512×8)
- 高 data regime 下 (8192×1 vs 1024×8)，diversity 饱和后，multiple rollouts 反超 (1024×8 > 8192×1)
- 这给了一个 "compute allocation recipe"：先 scale distinct query 到 saturation point，再把额外 compute 投入到 per-query multiple sampling

### 6.5 Model Family Generalization (Figure 4)

除了 Qwen3-8B，还在 Retail 上试了 Llama 3.1 8B 和 Rnj 1 8B（EssentialAI 的新 model）：

| Model | ICL | SIEVE (8K) |
|---|---|---|
| Qwen3 8B | 26% | 36% |
| Rnj 1 8B | 13.98% | 17.03% |
| Llama 3.1 8B | 4.53% | 3.44% |

**Llama 3.1 8B 失败了**，SIEVE 训练后反而比 ICL 还差。关键发现：Llama 3.1 8B 在 Retail 上 ICL 自己就只有 4.53%，说明它的 base reasoning 能力不足以处理这个 compositional rule task。当 base model 太弱，它既生成不出好 synthetic data，也 internalize 不了 training signal。

这是一个重要 caveat：SIEVE **不是 universal upgrade**，它需要 base model 本身对 target domain 有 reasonable 能力。Llama 3.1 8B 在这个 task 上不够格。

---

## 7. 与相关工作的关系

| Work | 关系 | 区别 |
|---|---|---|
| Snell 2022 (Learning by Distilling Context) | SIEVE 的 distillation objective 直接来自这个 | Snell 假设 access to query distribution；SIEVE 只要 3 seed |
| Bhargava 2024 (Prompt Baking) | Context distillation variant | 同样 data hungry |
| Eyuboglu 2025 (Cartridges) | 长文档 KV cache 注入 | 主攻 fact recall / memorization；SIEVE 攻 compositional reasoning |
| Lin 2025 (Active Reading), Yang 2024 (Synthetic Continued Pretraining) | Synthetic data 注入 facts | 同上，目标是 memorization 不是 reasoning |
| Zhu 2025 (BARE) | SIEVE 用 base model 做 seed sampling 的依据 | BARE 关注 few-shot synthetic data generation 本身 |
| Scheurer 2022, Chen 2024 (NL Feedback) | 用文本 feedback 做 training | 需要专家 traces / verifiers，SIEVE 不需要 |
| Hubotter 2026 (Self-distillation RL) | 同期 RL+distillation 工作 | 不同 paradigm，SIEVE 是 supervised distillation |

SIEVE 的 niche 是：**没有 query distribution, 没有 expert traces, 没有 verifiers，只有 context 和 3 个例子**。所有 prior work 至少要其中一项。

---

## 8. Limitations 和我的思考

### 8.1 论文自己承认的
- 只测了 verifiable domains，但作者承认 personalization 这种 non-verifiable domain 可能是更大用武之地
- Llama 3.1 8B 失败提示 base model capability 是 hard requirement
- Long context 扩展用 chunking，跨 chunk 的 reasoning unit 可能被切碎

### 8.2 我额外想到的

1. **Verification 的可靠性**：Step 2(c) 用 LLM 判断 context unit 是否 applicable，这本身是个 LLM judgment。如果 $M_\text{inst}$ 判断错（漏判/多判），rollout 还是会被污染。论文没给 verification accuracy 的 ablation。一个改进方向是用 self-consistency 多次 verify。

2. **Context decomposition 的 granularity**：什么是 "atomic"？一条 rule 可以 atomic，但 "customer is a student AND total spend ≥ $50" 这种 conjunction 是不是应该再拆？Appendix C.1 的 prompt 让模型自己决定，没有 quality metric。这是个 potential failure mode。

3. **Compositional generalization 测试不够**：Retail 用 30 rule，但训练 query 是从这些 rule 组合生成的；eval query 也是同分布生成（256 个）。真正的 OOD compositional generalization（用没见过的 rule 组合）没测。这其实是 *interpolation* 测试，不是 *extrapolation*。

4. **Self-distillation 的 ceiling**：rollout 是同款 model 生成的（"We sample rollouts for distillation from the same model we train"）。如果 base model 在某个 rule 上推理就错，SIEVE 会把这个错误蒸馏进 weights，且没有 verifier 纠正。这是和 RLHF / RLAIF 路线最大的差距。作者在 future work 提到 decouple generator 和 student，但没实验。

5. **Compute 成本没量化**：SIEVE-GEN 生成 16K data 要多少 H100 hours？verification phase 要对每个 query 逐个检查每个 context unit，n × N 次 LLM call，量级可能很大。Long context extension 部分已经承认 compute 大才改成 batch 模式，但主实验没给 wall clock / 美元成本。

6. **Inference cost 节省没量化**：parametric learning 的卖点之一是省 inference cost（不用每次塞 context）。但 paper 没给出 ICL vs SIEVE 的 inference latency / token cost 对比。MTOB 这种 50K token ICL baseline 推理一个 query 就要烧 50K token 的 prefill，SIEVE 只要 query 本身的 token —— 这个 delta 对 production 部署是巨大的，但没量化。

7. **Llama 失败的更深层解读**：Llama 3.1 8B 在 Retail ICL 上 4.53% 太低了，远低于预期。这其实可能暗示 Llama 3.1 8B 在 *multi-rule composition* 任务上有 systematic weakness，而 Qwen3 8B / Rnj 1 8B 强很多。这可能不是 SIEVE 的 limitation 而是 base model 本身的能力 cliff。值得在更多 base model 上 repeat。

8. **跟 Anthropic Constitutional AI 的 hidden link**：SIEVE 的 decompose + verify 跟 Constitutional AI 的 critique-revise 范式有结构相似性。如果 SIEVE-GEN 把 verification 改成 "critique rollout with applicable context"，可能进一步提升 quality。

9. **Mixture-of-Cartridges 方向**：Cartridges 是把长文档塞 KV cache，SIEVE 是塞 weights。中间态可能是按 chunk 学多个 LoRA / adapter，inference 时 routing。论文 future work 提到 specialized memory layers，方向正确但没实验。

---

## 9. Intuition 总结：为什么 SIEVE work？

把整个 mechanism 压成几条 core intuition：

1. **Context 是稀疏激活的**。一个 query 只触发 context 的一小部分。这是 sparsity 的体现。SIEVE 把这个 sparsity 显式化，用于 training data 构造。

2. **Training data purity > quantity**。Oracle 实验证明，即使 query 完美对齐 eval 分布，只要 rollout 时塞全 context，就打不过 synthetic query + applicable context。这是数据清洁度比数据真实性更重要的证据。

3. **Base model 比 instruct model 多样**。这是 BARE 发现的，SIEVE 复用。Instruction tuning 的 entropy collapse 让 model "average 化"，base model 保留 distributional tail。Synthetic data generation 的 diversity 关键在 sampler，不在 prompt。

4. **Forward KL 是 mean-seeking**。soft label distillation 不会 mode collapse 到 teacher 偶尔的 bad mode，会平均化。这就是为什么 top-K=100 的 soft target 比 hard argmax label 更鲁棒。

5. **Parametric > ICL 在 compositional reasoning 上**。Retail 上 SIEVE 超 ICL 37.7% 说明，attention 在长 context 上找 rule 是低效的，把 rule internalize 成 weight 才能让模型"自动"组合应用。这跟 hopfield network 的 attractor 机制有点像 —— 学习后 retrieval 比 online search 快且准。

6. **Base model capability 是 floor**。Llama 3.1 8B 在 Retail 上 ICL 都只有 4.53%，说明 SIEVE 不是 magic —— 它放大已有 capability，不能凭空创造 capability。

---

## 10. 参考链接

- SIEVE 原文（这篇 paper）: arXiv 链接需要从 paper metadata 推断，作者 Parth Asawa, Alexandros G. Dimakis, Matei Zaharia；主题词 "SIEVE: Sample-Efficient Parametric Learning from Natural Language" — 可以在 https://arxiv.org/ 搜索
- Context Distillation (Snell 2022): https://arxiv.org/abs/2209.15189
- Cartridges (Eyuboglu 2025): https://arxiv.org/abs/2506.06266
- BARE (Zhu 2025): https://arxiv.org/abs/2502.01697
- RuleArena (Zhou 2025): https://arxiv.org/abs/2412.08972 ；repo: https://github.com/SkyRiver-2000/RuleArena
- MTOB (Tanzer 2024): https://arxiv.org/abs/2309.16575
- Prompt Baking (Bhargava 2024): https://arxiv.org/abs/2409.13697
- Active Reading (Lin 2025): https://arxiv.org/abs/2508.09494
- Synthetic Continued Pretraining (Yang 2024): https://arxiv.org/abs/2409.07431
- Training with NL Feedback (Scheurer 2022): https://arxiv.org/abs/2204.14146
- Qwen3 Tech Report: https://arxiv.org/abs/2505.09388
- Llama 3: https://arxiv.org/abs/2407.21783
- Rnj-1 (Essential AI): https://huggingface.co/EssentialAI/rnj-1
- OpenThoughts (Guha 2025): https://arxiv.org/abs/2506.04178
- Self-Distillation Enables Continual Learning (Shenfeld 2026): https://arxiv.org/abs/2601.19897
- Reinforcement Learning via Self-Distillation (Hubotter 2026): https://arxiv.org/abs/2601.20802

---

## 11. 一句话总结

SIEVE 把 "context 是稀疏激活的" 这个 insight 翻译成 "training data 应该只包含 applicable context" 这条 mechanical rule，加上 BARE 启发的 base-model sampling for diversity，让 context distillation 从 "需要大量 expert data" 退化成 "需要 3 个 seed query + 一段 context"。在 Retail/RNBA/MTOB 三个 verifier-friendly domain 上验证，给出了 parametric learning 第一次在 sample efficiency 上追平 ICL 的 strong evidence。Llama 3.1 8B 的失败提示这方法的 floor 是 base model capability，self-distillation 是 ceiling 是 teacher 质量 —— 这是未来要拆掉的两面墙。
