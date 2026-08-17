---
source_pdf: Infinity Instruct.pdf
paper_sha256: 34fab43c80b08c8f2a8cbac250f340f49a8f8ebcd9848d92ebc34b16f9b73f4c
processed_at: '2026-08-05T09:37:14-07:00'
target_folder: LLM-Training/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Infinity Instruct

## 这篇 paper 在干嘛

一句话：**用一堆乱七八糟的开源 instruction data，通过精选 + 合成，搞出一个 mega dataset，让开源 model 的聊天能力追上 GPT-4**。

为什么这事难？你手上有 116 million 条 instruction data（math、code、knowledge、chat 乱七八糟混在一起），你 fine-tune 一个 Mistral-7B，结果发现：在 GSM8K 上可能还行，但 MATH、HumanEval 跟 GPT-3.5 差了一大截。你多塞 data 也没用，因为 garbage in garbage out。

所以核心问题变成：**怎么从 116M 大池子里挑出真正有用的 7.4M，再合成出 1.5M 高质量 chat data**。

---

## 他们干了三件事

### 第一件：从 116M 挑到 7.4M（InfInstruct-F-7.4M）

想象你面前有一堆 sand，里面有 gold。你怎么淘？

**对 knowledge domain（88.5M → 3.3M）**：直接把低质量 source 扔掉。比如 SST-2 这种 sentiment classification 数据——你 fine-tune 一个 LLM 去判断 movie review 是 positive 还是 negative，这对 general intelligence 帮助不大。Flan 2022 collection 里这种"low knowledge density"的东西统统扔掉。然后 dedup，因为同一条 seed 往往被 augmented 出几十条几乎一样的样本。

**对 math domain（11.8M → 1.4M）**：用 DSIR。这个方法的核心 idea 很简单——你有少量 GSM8K + MATH 的 prompt 当 "anchor"，你希望从大池子里选出的 data 分布跟 anchor 接近。

数学上就是 importance sampling：pool 里每条 data $x$ 有一个被选中的概率 $\propto \frac{q_{\text{target}}(x)}{p_{\text{pool}}(x)}$。其中 $q_{\text{target}}$ 是 anchor 分布，$p_{\text{pool}}$ 是 pool 分布。这两个分布通过 n-gram features + log-linear model 估计。

intuition：你想让 model 学好 GSM8K 风格的 math，那就选那些"长得像 GSM8K"的 data。简单粗暴但管用。

**对 code domain（7.1M → 1.5M）**：同样用 DSIR，但 anchor 换成 HumanEval 的 prompt。你想让 model 学会写 HumanEval 风格的 Python function，那就从 pool 里选那些 prompt 分布接近 HumanEval 的。

**还有一个 trick**：他们 fine-tune Mistral-7B 在当前版本 data 上，看跟 GPT-3.5 哪里差距大。比如发现 MATH 还差很远，就放宽 math 的 selection criteria，多塞点 math data。这就是 closed-loop data curation——用 model performance signal 反馈 data selection。

### 第二件：从 9M 合成出 1.5M chat data（InfInstruct-G-1.5M）

chat data 跟 foundational data 不一样。你想要的是真实场景下用户会问的问题，要 diverse、要 challenging。

**先建 labeling system**：用 Qwen1.5-72B 给每条 instruction 打标签。两层结构：26 个 first-level（粗粒度，比如 "Logic & Reasoning"、"Information Processing"），15000 个 second-level（细粒度，比如 "multi-step arithmetic word problem with variables"）。

**然后从 9M 里挑 1.2M seed**，四条准则：

1. **Long-tail priority**：某个 second-level label 在 pool 里出现 20~200 次的全保留，出现 200~500 次的随机抽 1/3。intuition：常见的东西 model 已经见过了，罕见的东西才是 model 的 weakness。

2. **Multi-capability priority**：一条 instruction 涉及多个 label 的优先保留。比如 "请用 Python 实现 quick sort 并解释时间复杂度"——同时涉及 code + algorithm + explanation，这种 data 价值高。

3. **High LM loss**：用 Qwen1.5-7B 算 answer 部分的 perplexity。loss 高说明 model 对这个能力不熟，值得学。公式就是标准 next-token loss：

$$
\mathcal{L} = -\frac{1}{|y|} \sum_{t=1}^{|y|} \log p(y_t | y_{<t}, x)
$$

4. **排除 high convergence loss**：如果 SFT 前后某条 data 的 loss 掉得太快，说明 model 对它过拟合了——这种 data 要么是 noise，要么是 outlier，要么会让 model 产生 harmful bias，扔掉。

这四条加起来就是在找：**model 还不会但又能学得会的 data**。已经会的不用学，学不会的也别硬学。

**接着 evolve**：对每个 seed，用 WizardLM 的 4 种策略 rewrite——加 constraint、deepening、concretizing、增加 reasoning steps。比如原 instruction 是"写一个 sorting function"，evolve 后变成"写一个 sorting function，要求 O(n log n) 时间复杂度，用 iterative 而非 recursive，并解释为什么这个复杂度成立"。

**最后 diagnosis**：拿 evolved data 让 Mistral-7B 和 Llama3-8B 回答，用 GPT-4 打分。两个 model 都答得差的，说明这块是 open-source model 的集体 weakness，push 到下一轮 evolution 继续 augment。

### 第三件：Two-stage training

Phase 1：用 InfInstruct-F-7.4M（7.4M foundational data + 1.2M replay seed）fine-tune base model
Phase 2：用 InfInstruct-G-1.5M 继续 fine-tune

为什么要 two-stage 而不是直接 mix 一起训？

Ablation 给了答案：
- F-3M only：conversational 16.0，foundational 54.4
- G-300K only：conversational 23.8，foundational 43.9
- One-stage mixing：conversational 22.3，foundational 50.9
- **Two-stage：conversational 25.5，foundational 52.0**

One-stage mixing 比 G-only 还差！为什么？因为 foundational data 把 chat signal dilute 了。model 同时见到 math 题 and chat 题，注意力被分散，chat 能力没训透。

Two-stage 的 intuition 就像人类学习：先学知识（math、code、knowledge），再学沟通（chat）。你不会让一个小孩同时学微积分和演讲技巧，会两边都学不好。先 foundation 打牢，再 chat 在这个 substrate 上自然涌现。

---

## 结果有多猛

InfInstruct-Llama3.1-70B vs GPT-4-0314：
- AlpacaEval 2.0：46.1 vs 35.3（+10.8）
- Arena-Hard：66.0 vs 50.0（+16.0）
- MT-Bench：8.9 vs 9.0（持平）
- Foundational overall：69.1 vs 70.3（接近持平）

关键是这是 pure SFT，没有 RLHF，没有 DPO。而 Llama3.1-70B-Instruct 是经过完整 RLHF + DPO pipeline 的——InfInstruct-70B 在 Arena-Hard 上还比它高 10.3。

这说明什么？**SFT data 质量是 alignment 的真正 bottleneck，不是 RLHF**。你如果有 7.4M 精筛 foundational data + 1.5M 合成 chat data，纯 SFT 就能超越经过 RLHF 的同尺寸 model。RLHF 之前被神化了，好的 SFT data 能顶住。

---

## 我觉得最 clever 的几个点

1. **DSIR 用 benchmark train set 当 target**：不是瞎选，是有目标分布对齐。简单但有效。

2. **Closed-loop weak-domain supplement**：fine-tune → evaluate → 找 weakness → 放宽 criteria 补 data。这就是 data-centric 版的 RL。

3. **Labeling system 15000 个 second-level label**：颗粒度极细，能精准识别 model 的 capability gap。比粗粒度 category 有用得多。

4. **High LM loss + 排除 high convergence loss**：这两个 filter 组合很精妙，找的是"可学习的未知"而非"不可学的 outlier"。

5. **Diagnosis loop**：用 GPT-4 当 weakness detector，反过来指导 data synthesis。这就是 self-improving data pipeline 的雏形。

---

## 我觉得有问题的地方

1. **Decontamination threshold 0.3 太松**：cosine similarity 0.3 就过滤，可能漏掉很多 benchmark leakage。Table 2 里 OpenHermes 在 GSM8K 上 73.0 异常高，很可能就是 contamination。

2. **Multi-turn 太少**：F-7.4M 单 turn 占 92.6%，G-1.5M 单 turn 占 97.7%。但 MT-Bench 是 multi-turn benchmark，InfInstruct 在 MT-Bench 上提升小（8.9 vs Llama3.1-70B-Instruct 的 8.6），可能就是 multi-turn data 不足。

3. **GPT-4 judge 偏差**：AlpacaEval、Arena-Hard、MT-Bench 都用 GPT-4 当 judge，Diagnosis 也用 GPT-4。如果 InfInstruct 的 response 风格偏向 GPT-4 风格（因为 evolve 时可能用 GPT-4 或 Qwen 生成），那 GPT-4 judge 会 self-preference。

4. **DSIR target 太小**：GSM8K train 仅 7.4K，HumanEval 仅 164 题。用这么小的 target 估计 importance weight，variance 会很大。

5. **Replay seed 跟 Phase 2 重复**：1.2M seed 既塞进 F-7.4M 当 replay，又在 Phase 2 训 G-1.5M，相当于这 1.2M 被训了两遍。这个 redundancy paper 没解释。

6. **只跑 Mistral-7B 做 ablation**：two-stage 是否对 Llama3.1-70B 也 optimal？没验证。可能 70B 的 ablation 成本太高。

---

## 这篇 paper 对 field 的真正意义

我觉得就一句话：**SFT data curation 是被严重低估的 alignment lever**。

之前大家觉得要追上 GPT-4 必须 RLHF + DPO + Constitutional AI + 各种 fancy alignment 技巧。Infinity Instruct 证明了：你只要把 data 选对、合成对、训两阶段，pure SFT 就能干掉 GPT-4-0314 的 chat 能力。

这对开源社区是巨大赋能——RLHF 需要 reward model、需要人类标注 preference、需要 PPO 训练 infra，门槛极高。而 SFT data curation 门槛低得多，任何一个有 H100 集群的小团队都能复现。

所以这篇 paper 的 contribution 不是某个 single technique，是 demonstration：**正确的 data pipeline 比花哨的 alignment algorithm 更重要**。

参考：
- Infinity Instruct dataset: https://huggingface.co/datasets/BAAI/Infinity-Instruct
- Code: https://github.com/BAAI-DCAI/Infinity-Instruct
- LIMA（同样证明 SFT data 质量重要性）: https://arxiv.org/abs/2305.11206
- DSIR: https://arxiv.org/abs/2302.03169
- WizardLM: https://arxiv.org/abs/2304.12244

---

# Infinity Instruct: 深度技术解析

## 1. 论文动机与核心贡献

### 1.1 问题背景
当前 open-source instruction dataset 普遍存在 domain 偏窄（math 或 code 等）问题，导致 fine-tuned 后的 model 与 GPT-4 这类 proprietary model 存在 capability gap。同时 naive SFT 在 instruction data 上会触发 catastrophic forgetting，让 pretrained model 的 foundational capability（reasoning、knowledge）流失。

### 1.2 核心贡献
- **InfInstruct-F-7.4M**：从 116.4M 原始 instruction pool 中 curate 出 7.4M foundational instructions
- **InfInstruct-G-1.5M**：基于 two-layer labeling system 合成出 1.5M conversational instructions
- **Two-stage training pipeline**：foundational SFT → conversational SFT，验证两阶段优于 one-stage mixing
- **InfInstruct-Llama3.1-70B** 在 conversational benchmark 上超过 GPT-4-0314 8.6%，foundational 接近 GPT-4-0314

参考链接:
- Infinity Instruct dataset: https://huggingface.co/datasets/BAAI/Infinity-Instruct
- Code repo: https://github.com/BAAI-DCAI/Infinity-Instruct
- WizardLM evolve-instruct: https://arxiv.org/abs/2304.12244
- DSIR paper: https://arxiv.org/abs/2302.03169
- MAGPIE: https://arxiv.org/abs/2406.08464
- OpenHermes-2.5: https://huggingface.co/datasets/teknium/OpenHermes-2.5

---

## 2. 整体架构图解析（Figure 1）

整个 pipeline 分为四个阶段，构建出两条 dataset 分支：

```
Open-source Instruction Pool (116.4M)
        │
        ├──────────────────────┬─────────────────────┐
        │                      │                     │
   Knowledge              Math/Code            Conversational
   (88.5M)               (18.9M)                (9.0M)
        │                      │                     │
        ▼                      ▼                     ▼
   ┌────────────────────────────────────────────────────┐
   │       Phase 1: Data Selection (Figure 2)            │
   │  - Rule-based filtering                             │
   │  - DSIR importance resampling                       │
   │  - Coverage-based selection                         │
   │  - Weak-domain supplement                           │
   └────────────────────────────────────────────────────┘
        │
        ▼
   InfInstruct-F-7.4M (foundational)
        │
        │  + 1.2M seed instructions (replay strategy)
        │
        ▼
   ┌────────────────────────────────────────────────────┐
   │      Phase 2: Data Synthesis (Figure 3)             │
   │  - Instruction labeling system (26 L1 + 15K L2)     │
   │  - Seed selection (difficulty + diversity)          │
   │  - Evolve-instruct (4 evolution strategies)         │
   │  - Weakness diagnosis (GPT-4 judge)                │
   └────────────────────────────────────────────────────┘
        │
        ▼
   InfInstruct-G-1.5M (conversational)
```

关键 insight：**两阶段衔接处使用 1.2M seed instruction 作为 replay buffer**，这呼应了 continual learning 中的 experience replay 技术，避免在 conversational SFT 阶段遗忘 foundational 能力。

---

## 3. Data Selection 模块技术细节

### 3.1 三种 selection strategy

#### (1) Source filtering（针对 Knowledge domain）
从 Flan 2022 collection 中过滤掉低 knowledge density 的子集（如 SST-2 sentiment classification、IMDb movie reviews），并实施 deduplication 减少 augmentation 样本占比。这部分数据从 88.5M 压缩到 3.3M。

#### (2) Rule-based filtering
基于任务特征设计启发式规则，例如 code 任务的 length、syntax、test pass rate。

#### (3) DSIR（Data Selection for Language Models via Importance Resampling）

DSIR 是这篇 paper 在 math 和 code domain 选择的核心技术，由 Xie et al. 提出。

**核心数学原理**：

给定一个 unlabeled pool $D_{\text{pool}} \sim P$，希望从中选取一个子集 $D_{\text{sel}}$，使其分布 $Q_{\text{sel}}$ 接近 target distribution $Q_{\text{target}}$（target 由 GSM8K / MATH / HumanEval 等少量样本给出）。

形式化目标函数：

$$
\min_{D_{\text{sel}} \subset D_{\text{pool}}} D_{\text{KL}}\left( Q_{\text{sel}} \,\|\, Q_{\text{target}} \right) - \lambda \cdot |D_{\text{sel}}|
$$

其中：
- $D_{\text{KL}}$ 是 KL divergence
- $\lambda$ 是数据量惩罚项
- $Q_{\text{sel}}$ 是子集上的经验分布

**Importance weight 推导**：

理论上每个样本 $x$ 的 importance weight 为：

$$
w(x) = \frac{q_{\text{target}}(x)}{p_{\text{pool}}(x)}
$$

其中 $p_{\text{pool}}(x)$ 和 $q_{\text{target}}(x)$ 通过 kernelized density estimation 估计：

$$
\hat{p}_{\text{pool}}(x) = \frac{1}{|D_{\text{pool}}|} \sum_{x_i \in D_{\text{pool}}} K_H(x - x_i), \quad K_H(u) = |H|^{-1/2} K(H^{-1/2} u)
$$

$H$ 是 bandwidth matrix，$K$ 是 kernel function（通常 Gaussian）。

实际工程实现：用 n-gram features $\phi(x)$ 做特征，并用 log-linear model 拟合 log importance ratio：

$$
\log \frac{\hat{q}_{\text{target}}(x)}{\hat{p}_{\text{pool}}(x)} \approx \theta^\top \phi(x)
$$

参数 $\theta$ 通过最大化如下 objective 求得：

$$
\theta^* = \arg\max_\theta \, \sum_{x \in D_{\text{target}}} \log \frac{\exp(\theta^\top \phi(x))}{\sum_{x' \in D_{\text{pool}}} \exp(\theta^\top \phi(x'))}
$$

实际抽样阶段：对 pool 中每个样本 $x_i$，以概率 $\propto \exp(\theta^\top \phi(x_i))$ 进行 weighted sampling。

**在 Infinity Instruct 中**：
- Math domain：用 GSM8K + MATH 的 training prompt 作为 $D_{\text{target}}$，从 11.8M pool 中选 1.4M
- Code domain：用 HumanEval prompt 作为 $D_{\text{target}}$，从 7.1M pool 中选 1.5M

进一步 Infinity Instruct 还额外合成 CoT 和 PoT 数据增强 math 对数字变体的敏感度，参考了同作者的 InfinityMath 工作：https://arxiv.org/abs/2407.07410

---

### 3.2 Evaluation and Weak-domain Instruction Supplement

这是 paper 里一个**很关键但容易被忽略**的设计：

```
Loop:
  1. fine-tune Mistral-7B on current version of dataset
  2. evaluate on GSM8K/MATH/HumanEval/MBPP/MMLU/C-EVAL
  3. compare with GPT-3.5 baseline
  4. if gap in domain d is large:
       relax selection criteria for domain d → add more data
  5. repeat until saturation
```

这是一个 closed-loop data curation 思路，本质上是在做 **data-centric RL**——把 model performance signal 反馈到 data selection 决策中。这与 Google 的 "Did you train on my test set?" 系列、以及 self-improving data selection 工作 都有思想上的关联。

---

### 3.3 最终 InfInstruct-F-7.4M 构成

| Domain | Pool (M) | Used (M) | Selection 策略 |
|---|---|---|---|
| Code | 7.1 | 1.5 | DSIR (HumanEval target) |
| Math | 11.8 | 1.4 | DSIR (GSM8K + MATH target) + CoT/PoT synthesis |
| Knowledge | 88.5 | 3.3 | Flan 2022 source filtering + dedup |
| Instruction Follow | 9.0 | 2.8 | (作为 Phase 2 的输入池) |
| **Total foundational** | **116.4** | **6.2** | |
| **+ Replay seeds** | | **+ 1.2** | 从 Phase 2 的 1.2M seeds 中复用 |
| **Final F-7.4M** | | **7.4** | |

---

## 4. Data Synthesis 模块技术细节

### 4.1 Instruction Labeling System

这是 paper 中最有创新的部分之一。系统分两层：

**Layer 1（first-level label）**：26 个粗粒度能力类型
**Layer 2（second-level label）**：~15,000 个细粒度能力类型

构建流程：

```
Step 1: 用 Qwen1.5-72B 对每个 instruction 打 second-level 标签
Step 2: 用 embedding clustering + 人工 tuning 规范化 L2 标签
Step 3: 利用 LLM 的 generalization 能力从 L2 → L1 抽象
```

可视化（Figure 5 的 t-SNE）显示 Logic & Reasoning、Information Processing 等复杂类型分布更分散，符合其覆盖更广 knowledge point 的预期。这与 Instag（Lu et al., 2023, https://arxiv.org/abs/2308.07087）思想相似，但 label 颗粒度更细。

参考其 prior 工作：https://arxiv.org/abs/2409.07045

---

### 4.2 High-quality Seed Instruction Selection（9M → 1.2M）

paper 提出四条过滤准则，混合 difficulty 与 diversity 两个维度：

#### (1) Long-tail data（Diversity）
对每个 L2 label，按 frequency 分桶：
- frequency ∈ [20, 200]：全保留
- frequency ∈ [200, 500]：随机抽 1/3

数学表达：设 $f(c)$ 为 label $c$ 在 pool 中的频次，则

$$
\mathbb{1}[\text{keep} \mid x, c(x)] = \begin{cases} 1 & 20 \le f(c(x)) \le 200 \\ \text{Bernoulli}(1/3) & 200 < f(c(x)) \le 500 \\ 0 & \text{otherwise} \end{cases}
$$

#### (2) Multidimensional capabilities（Diversity）
优先保留涉及多个 L2 label 的 instruction。设 $C(x) \subseteq \mathcal{C}$ 是 instruction $x$ 触发的 label 集合，则 priority score：

$$
S_{\text{multi}}(x) = |C(x)|
$$

#### (3) High language modeling loss（Difficulty）
用 Qwen1.5-7B 对 candidate instruction 的 answer 部分 forward 计算 per-token loss：

$$
\mathcal{L}_{\text{LM}}(x) = -\frac{1}{|y|} \sum_{t=1}^{|y|} \log p_\theta(y_t \mid y_{<t}, x)
$$

其中 $x$ 是 instruction，$y$ 是 gold answer。loss 越高说明 model 对该能力越不熟，越应保留。

#### (4) High convergence loss（Difficulty，防过拟合）
参考 Self-Guided paper（https://arxiv.org/abs/2308.12032）的 insight：若一个样本在 SFT 前后 loss 差异过大，model 容易对其过拟合产生 harmful bias。具体做法：

设 $\mathcal{L}_{\text{before}}(x)$ 为 SFT 前 loss，$\mathcal{L}_{\text{after}}(x)$ 为 SFT 后 loss。若

$$
\Delta \mathcal{L}(x) = \mathcal{L}_{\text{before}}(x) - \mathcal{L}_{\text{after}}(x) > \tau
$$

则剔除该样本。

**这里我想到的一个 insight**：第 (3) 和第 (4) 条看起来有些 tension——既要 high LM loss，又不要 loss 下降太快。本质上是想找那些 model **未掌握但可学** 的 instruction，过滤掉那些 outlier（label noise 或 unlearnable）。这与 "Cherry Picking" 思路、以及 Common Crawl deduplication 中的 "learnability filter" 思想呼应。

---

### 4.3 Instruction Evolution（WizardLM evolve-instruct）

对每个 seed instruction，应用 WizardLM 的 4 种 evolution 策略：

1. **Add Constraints**：增加约束（如 "用 100 字以内"、"不要使用循环"）
2. **Deepening**：增加深度（如增加推理链长度）
3. **Concretizing**：具体化（把抽象问题落到具体场景）
4. **Increase Reasoning**：增加 reasoning 步骤

rewriting model 同时验证 evolution 后语义是否保持一致、是否引入 harmful information。

公式上可以理解为 instruction space 上的 operator $T_k: \mathcal{I} \to \mathcal{I}$（k=1..4），迭代若干轮：

$$
x^{(t+1)} = T_{k^*}(x^{(t)}), \quad k^* = \arg\max_k \text{Quality}(T_k(x^{(t)}))
$$

并接受准则 $\text{Accept}(x^{(t+1)}) = \text{SemEq}(x^{(t)}, x^{(t+1)}) \wedge \neg\text{Toxic}(x^{(t+1)})$

---

### 4.4 Diagnosis（Weakness detection）

```
Loop:
  1. extract evolved instructions grouped by L1 ability type
  2. evaluate Mistral-7B, Llama3-8B responses with GPT-4 judge
  3. collect instructions where both models score poorly
  4. push them to next round of evolution
```

这是一个 **adversarial-style data synthesis**，类似 Adversarial ICL、Self-Instruct Refine、FLARE。paper 没给具体 prompt 模板，但参考 LMSYS Arena Hard（https://lmsys.org/blog/2024-04-19-arena-hard/）的 scoring 方式。

---

## 5. Deduplication & Decontamination

用 BGE embedding（https://arxiv.org/abs/2309.07597）将每条 instruction 转为 vector $e(x) \in \mathbb{R}^d$，计算 cosine similarity：

$$
\text{sim}(x_i, x_j) = \frac{e(x_i)^\top e(x_j)}{\|e(x_i)\| \|e(x_j)\|}
$$

阈值 $\tau = 0.3$ 人工设定——这个阈值其实偏激进（常见做法是 0.7~0.9），说明他们更关注 contamination 防御而非只是 dedup。

潜在改进：可以用 MinHash + LSH 做 jaccard dedup（更快），embedding similarity 留给 semantic dedup。Anthropic 的 DCPR、OpenAI 的 WebGPT 都有类似设计。

---

## 6. 实验结果深度分析

### 6.1 Table 2：open-source dataset baseline 比较

| Dataset | GSM-8K | MATH | HumanEval | MBPP | MMLU | C-EVAL |
|---|---|---|---|---|---|---|
| GPT-3.5 | 57.1 | 28.0 | 48.1 | 68.2 | 70.0 | 54.4 |
| Mistral-7B base | 48.1 | 11.8 | 14.0 | 38.0 | 56.5 | 34.6 |
| + OpenHermes | **73.0** | 18.2 | 41.5 | 41.8 | 61.7 | 43.0 |
| + Mammoth-v2 | 50.0 | 17.6 | 27.4 | 33.0 | 57.9 | 40.4 |
| + Bagel | 58.9 | 18.3 | 35.4 | 39.2 | 54.9 | 37.5 |

观察：
- OpenHermes 在 GSM8K 上以 73.0 异常突出 → 这说明 OpenHermes 内很可能有 GSM8K-like 数据（或 augmentation），存在一定 contamination 风险
- 所有开源 dataset 在 MATH、HumanEval 上远落后于 GPT-3.5，这是 Infinity Instruct 重点攻坚的方向

### 6.2 Table 3 + 4：InfInstruct fine-tuned model 表现

InfInstruct-Llama3.1-70B 关键数字：

| | AlpacaEval 2.0 | Arena-Hard | MT-Bench | Foundational Overall |
|---|---|---|---|---|
| GPT-4-0314 | 35.3 | 50.0 | 9.0 | 70.3 |
| Llama3.1-70B-Instruct | 38.1 | 55.7 | 8.6 | 68.4 |
| **InfInstruct-Llama3.1-70B** | **46.1** | **66.0** | 8.9 | 69.1 |

- AlpacaEval 2.0 提升 +5.8 over Llama3.1-70B-Instruct，+10.8 over GPT-4-0314
- Arena-Hard 提升 +10.3 over Llama3.1-70B-Instruct，+16.0 over GPT-4-0314
- Foundational 几乎持平（-0.1 vs Llama3.1-70B-Instruct）

**intuition**：AlpacaEval 2.0 用 GPT-4 作为裁判，本身存在 length bias（虽然 2.0 版本已经 length-controlled），可能 InfInstruct 倾向于更长 response。Arena-Hard 500 个 query 都是 challenging query，更能反映真实对话能力，提升 +16 是真正的硬核突破。

### 6.3 Table 8：Ablation（核心 insight！）

| Setting | Conversational (AlpacaEval 2.0) | Foundational (avg) |
|---|---|---|
| InfInstruct-F-3M only | 16.0 | 54.4 |
| InfInstruct-G-300K only | 23.8 | 43.9 |
| One-Stage (mixing) | 22.3 | 50.9 |
| **Two-Stage** | **25.5** | **52.0** |

关键观察：
1. **G-300K 单独训练只能拿 23.8**，但 F-3M → G-300K 两阶段能到 25.5，多 +1.7 是 curriculum bonus
2. **One-stage mixing 比 G-only 差**（22.3 vs 23.8），说明把 foundational data 简单混进 conversational data 反而 dilute 了 chat 信号
3. **Two-stage 比 G-only foundational 高**（52.0 vs 43.9），因为 foundational capability 给 chat 提供了 reasoning substrate
4. F-only 比 G-only foundational 高（54.4 vs 43.9）符合预期
5. **F-only conversational 16.0 也非零**，说明 foundational data 本身包含一定 conversational signal

这条 ablation 直接验证了 paper 核心 thesis：**foundational 与 conversational 能力正相关，two-stage curriculum 是当前 SOTA 方案**。

这呼应了 Meta 的 Llama 3.1 paper 中"first learn skills, then learn style"的设计哲学，也呼应 Anthropic Constitutional AI 的 staged training。

---

## 7. 关联联想与延伸思考

### 7.1 与其他 instruction dataset 的横向对比

| Dataset | 规模 | 构造方式 | 优势 |
|---|---|---|---|
| OpenHermes-2.5 | ~1M | synthetic + filtering | diversity 好，GSM8K 偏强 |
| UltraChat | ~1.5M | GPT-4 multi-turn dialog | multi-turn 质量高 |
| Evol-Instruct (WizardLM) | ~250K | evolutionary rewrite | difficulty 推进 |
| MAGPIE-Pro | ~1M | self-prompting aligned LLM | diversity 高 |
| WildChat | 1M | real ChatGPT logs | 真实分布 |
| LMSYS-Chat-1M | 1M | real arena logs | 真实分布 |
| **Infinity Instruct** | **8.9M (F+G)** | selection + evolution + diagnosis | 兼顾 F 和 G |

参考：MAGPIE: https://arxiv.org/abs/2406.08464

### 7.2 关于 scaling curve（Figure 4）
paper 在 Qwen2.5-1.5B 上做了 scaling curve，结果显示 reasoning 任务（MATH、GSM-8K）随 data 量增长显著提升，而 conversational benchmark 提升更快但更早饱和。

这与 "Chinchilla"-style scaling law 不同——Chinchilla 关注 pretraining，Infinity Instruct 关注 instruction tuning。两者 form 类似：

$$
L(N, D) = A \cdot N^{-\alpha} + B \cdot D^{-\beta} + L_\infty
$$

但 instruction tuning 的 $\beta$ 通常更大（数据有效性更高）。Anthropic 的 InstructGPT 系列工作（https://arxiv.org/abs/2203.02155）有类似观察。

### 7.3 Paper 没说但值得深挖的点

**(a) Reward model 标注**：Section 3.3 中提到用 reward model 给每条 instruction-response 打 reward score 用于 scaling sampling，但没说 reward model 是哪个。我猜是 Skywork-Reward 或 InternLM2-Reward 之类的 open-source RM。这是潜在 weakness，因为 RM 偏好会直接 inject bias 到 data selection 中。

**(b) Multi-turn 占比偏低**：F-7.4M 单 turn 占 92.6%，G-1.5M 单 turn 占 97.7%——这跟当前主流 multi-turn benchmark（MT-Bench 等）的分布严重 mismatch。这可能是为什么 Arena-Hard（500 challenging single-turn query）涨幅大，但 MT-Bench（multi-turn）涨幅小的原因。

**(c) Phase 1 与 Phase 2 衔接**：1.2M seed 是 replay buffer，但 paper 没说明这 1.2M 是 G 的子集还是 G 的全部。从规模上看 G 只有 1.5M，1.2M 占了 80%，相当于把 G 的 80% 直接塞进 F 里——这与 Phase 2 又 fine-tune G 形成一定冗余。

**(d) DSIR 的 target distribution 选取**：用 GSM8K + MATH training prompt 作为 math 的 target，用 HumanEval 作为 code 的 target——这些 benchmark 本身的 train set 较小（GSM8K train 仅 7.4K，HumanEval 仅 164 个 problem），$D_{\text{target}}$ 很小，会带来 high variance。Paper 没做 DSIR target size ablation。

**(e) Evolve-instruct 的"语义保持"判定**：paper 说"ask rewriting model to determine whether the rewritten instruction was semantically identical"，但 LLM 自己判断语义相似度有 known failure mode（hallucinated equivalence），建议用 embedding similarity 辅助验证。

**(f) Diagnosis step 的 GPT-4 依赖**：让 GPT-4 评估 open-source model 的 weakness 来指导 data 合成，这本质上是把 GPT-4 当作 weak capability 探测器。这有 bootstrapping 的味道，也带来 GPT-4 bias 的 inheritance。

### 7.4 与 RLHF / DPO 的关系
Infinity Instruct 只做了 SFT，没有 RLHF/DPO 阶段，但 Llama3.1-70B-Instruct 是经过 RLHF + DPO 的。InfInstruct-70B 在 pure SFT 下超过 Llama3.1-70B-Instruct 在 Arena-Hard 上 +10.3，说明 **高质量 SFT data 的天花板比预想高得多**。这呼应了 LIMA paper 的"1K high-quality examples is enough" thesis，只不过 Infinity 把规模推到了 7.4M+1.5M。

参考 LIMA: https://arxiv.org/abs/2305.11206

### 7.5 与 SFT scaling law 的关系
最近 Allen NLP、Princeton 的"SFT scaling law"研究（https://arxiv.org/abs/2405.10934, https://arxiv.org/abs/2405.13481）指出 SFT 也存在 compute-optimal 配比，Infinity Instruct 没做 compute matching，所有 model 都跑 3 epoch。

### 7.6 潜在的下一步工作
1. **multi-turn scaling**：G-1.5M 多 turn 比例太低，需要专门扩 multi-turn evolution
2. **RM-free data selection**：当前 scaling curve 依赖 RM，可以用 self-consistency 或 reference model loss 替代
3. **Domain-conditional DSIR**：让 DSIR target 来自多个 distribution 而非单一 benchmark
4. ** continual learning formalization**：two-stage 训练其实是 continual learning 特例，可以用 EWC、A-GEM 等方法进一步 reduce forgetting
5. **Self-improving pipeline**：把 Diagnosis step 升级为 full self-play（类似 SPIN, https://arxiv.org/abs/2401.01335）

---

## 8. 局限性

paper 自己提到的（Section E）：
1. Labeling/synthesis 依赖 open-source model，引入 bias
2. Pipeline 计算开销大
3. 评估主要看 benchmark，未验证 real-world robustness、safety、long-term retention

我补充：
4. **Benchmark leakage 风险**：从 116M 池子选 data，无法保证 pool 里没有 benchmark 测试集 contamination；decontamination 阈值 0.3 cosine 也不能完全消除
5. **GPT-4 judge 依赖**：AlpacaEval 2.0、Arena-Hard、MT-Bench 都用 GPT-4 当 judge，而 Diagnosis step 也用 GPT-4——存在 GPT-4 self-preference 风险
6. **single model base**：ablation 只在 Mistral-7B 上做，generalization 待验证
7. **没有涉及 reasoning model**：未涵盖 o1-style reasoning、tool use、agentic 场景

---

## 9. 总结性 Intuition

我从 paper 中提炼的 3 个 core insight：

1. **Foundational capability 是 conversational capability 的 substrate**。Two-stage curriculum > one-stage mixing，因为 mixing dilutes signal。这与人类学习"先学知识再学沟通"逻辑一致。

2. **Data selection 是 model capability 的 bottleneck，不是 compute**。从 116M → 7.4M 的精筛让 Llama3.1-70B SFT 超过 GPT-4-0314 chat 能力，说明 GPT-4 的优势很大程度上来自其训练数据 curating 工艺，而非单纯参数规模。

3. **Closed-loop data curation**：用 model performance 反馈 data selection 决策，本质上是把 data pipeline 也看作一个可优化的 model，可以迭代。这与 RLHF 的 reward model、DPO 的 preference model 一样，data-centric AI 的下一个 frontier 是 closed-loop data flywheel。

参考 closed-loop data curation: https://arxiv.org/abs/2310.12888

---

## 10. 关键 web links 汇总

- Paper: (Infinity Instruct, BAAI)
- Dataset: https://huggingface.co/datasets/BAAI/Infinity-Instruct
- Code: https://github.com/BAAI-DCAI/Infinity-Instruct
- DSIR: https://arxiv.org/abs/2302.03169
- WizardLM Evolve-Instruct: https://arxiv.org/abs/2304.12244
- OpenHermes-2.5: https://huggingface.co/datasets/teknium/OpenHermes-2.5
- MAGPIE: https://arxiv.org/abs/2406.08464
- UltraChat: https://arxiv.org/abs/2305.14233
- WildChat: https://arxiv.org/abs/2405.01470
- LMSYS-Chat-1M: https://arxiv.org/abs/2309.11981
- BGE embedding: https://arxiv.org/abs/2309.07597
- Self-Guided data selection: https://arxiv.org/abs/2308.12032
- Instag: https://arxiv.org/abs/2308.07087
- Arena-Hard: https://lmsys.org/blog/2024-04-19-arena-hard/
- AlpacaEval 2.0: https://github.com/tatsu-lab/alpaca_eval
- MT-Bench: https://arxiv.org/abs/2306.05685
- LIMA: https://arxiv.org/abs/2305.11206
- SPIN: https://arxiv.org/abs/2401.01335
- InfinityMath: https://arxiv.org/abs/2407.07410
- Flan 2022: https://arxiv.org/abs/2301.13688
- Beyond IID: https://arxiv.org/abs/2409.07045

这篇 paper 真正的 contribution 不在于某个 single technique（DSIR、evolve-instruct、labeling system 都已有），而在于把它们串成了一个 closed-loop、domain-aware、two-stage 的 holistic pipeline，并通过扎实的 ablation 证明了 foundational 与 conversational 训练的 synergy。Open-source community 长期把 SFT 当作 alignment 的 phase 1，paper 则示范了 SFT 本身也能达到 GPT-4 level，对 future open-source 模型发展提供了 concrete path forward。
