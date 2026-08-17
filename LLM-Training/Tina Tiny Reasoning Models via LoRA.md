---
source_pdf: Tina Tiny Reasoning Models via LoRA.pdf
paper_sha256: bff08df4b6af6cb5684aa175dea428ea3c8b614c70dbd45b8c18470dc55f0ee1
processed_at: '2026-08-12T16:22:33-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Tina: 用人话讲讲这篇paper

## 这篇paper在讲什么故事

一个很朴素的问题：**训练一个会推理的小模型，到底能多便宜？**

USC的团队给出的答案：**$9**。

不是$900，不是$9000，是9美元——一杯Starbucks咖啡的钱。然后他们在AIME24（美国数学邀请赛）上跑出了43.33% Pass@1，平均5个benchmark 50.60%，打平甚至超过了同样base model上full-parameter RL的SOTA（DeepScaleR、STILL-3、Open-RS系列）。

这个gap有多夸张？看Figure 1就知道：横向是cost（对数scale），纵向是performance，Tina的点孤零零地挂在左上角，其他baseline全堆在右下。cost差了约**260倍**，performance反而更高。

项目链接：https://shangshangwang.notion.site/tina  
代码：https://github.com/shangshang-wang/Tina

---

## 怎么做到的：三个"小"叠在一起

### 1. Base model小

用 `DeepSeek-R1-Distill-Qwen-1.5B`，1.5B参数。这个选择有讲究：它是DeepSeek-R1蒸馏到Qwen2.5-1.5B上的产物，所以**已经天生带着一点reasoning的底子**。这样后续RL的增量提升才能干净归因于"LoRA-RL本身"，而base太弱的话增量就没法测量。

参考：DeepSeek-R1 paper https://arxiv.org/abs/2501.12948

### 2. Parameter update小

用LoRA（https://arxiv.org/abs/2106.09685）。原始权重 $W_0 \in \mathbb{R}^{d \times k}$ 全程冻结，只训练两个小矩阵 $A \in \mathbb{R}^{d \times r}$ 和 $B \in \mathbb{R}^{r \times k}$，其中 $r \ll \min(d, k)$。Forward变成：

$$\hat{h}(x) = W_0 x + AB x$$

这里 $r$ 是LoRA的rank，paper默认 $r=32$，配合 $\alpha=128$，scaling factor $\alpha/r = 4$。

直觉：原模型的知识不动，只学一个"很小但很有用"的correction。可训练参数量比full fine-tune少了好几个数量级。

### 3. 硬件小

只用2块 NVIDIA L40S GPU，约$1/GPU-hour（Cudo Compute, https://www.cudocompute.com/products/gpu-cloud/nvidia-l40s）。

正常GRPO配置往往要3+块GPU——一块专门跑vLLM做rollout采样，其余做training。Tina偏要把training和vLLM co-locate在同样的2块GPU上，靠限制vLLM的GPU memory utilization（设成0.4）硬塞。

代价：wall-clock时间变长。收益：硬件门槛砍掉一半。

---

## RL算法：GRPO

GRPO（Group Relative Policy Optimization，来自DeepSeekMath，https://arxiv.org/abs/2402.03300）是PPO的简化版，核心改动：**去掉value network**，改用group内相对比较来估计advantage。

对每个question $q$，从old policy $\pi_{\theta_{old}}$ 采样一组output $G = \{o_1, \ldots, o_G\}$（本文 $G=4$），然后算每个output的advantage：

$$A_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$

- $r_i$：第 $i$ 个output拿到的reward
- 减mean：相对于"这组平均水平的output"，这个output好多少
- 除std：归一化scale，避免reward magnitude影响过大

这比PPO的GAE简单太多——**不需要训练critic，省一半参数，省一半梯度，省一半memory**。对小模型来说太香了。

Policy gradient objective（带clipping防止step太大）：

$$\mathbb{E}\left[\frac{1}{G}\sum_i \min\left(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i\right) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref})\right]$$

其中 $\rho_i = \pi_\theta(o_i|q) / \pi_{\theta_{old}}(o_i|q)$ 是importance sampling ratio，$\epsilon$ 是clip范围，$\beta$ 是KL penalty权重，$\pi_{ref}$ 是冻结的reference policy（防止policy漂得太远）。

---

## Reward怎么设的

看Table 6，发现一个很一致的设计哲学：**Accuracy reward权重=2，format类reward权重=1**。

reward类型包括：
- **Accuracy**：答案对不对（verifiable reward，math problem能验证正误）
- **Format**：有没有用 `<answer>...</answer>` 这种结构
- **Length**：response长度
- **Cosine**：cosine-scaled reward（按难度scaling）
- **Tag Count**：reasoning step tag数量
- **Reasoning Steps**：中间推理步数
- **Repetition Penalty**：惩罚重复

Tina-OpenR1 / Tina-OpenThoughts用了7种reward组合，每个权重都是1；其他配置只挑2种reward（比如Accuracy+Format，权重2:1）。

---

## 实验结果：Tina vs. full-parameter baseline

### Baseline重新评估（Table 2）

paper特别负责的一点：先用统一framework（lighteval + vLLM）重新评估所有baseline，因为原paper用的eval framework五花八门（verl, lighteval, lm-eval-harness），generation参数也不同，直接比对就是苹果比橘子。

重评后baseline平均分：
- DeepSeek-R1-Distill-Qwen-1.5B（base）：41.18
- STILL-3-1.5B-preview：44.86
- DeepScaleR-1.5B-Preview：48.74
- Open-RS3：46.06

### Tina主结果（Table 3）

| Tina Model | Best ckpt位置 | Avg | 对应baseline Avg |
|---|---|---|---|
| Tina-STILL-3 | 53% of 1 epoch | 48.16 | 44.86 |
| Tina-DeepScaleR | 19% of 1 epoch | 48.38 | 48.74 |
| Tina-Open-RS1 | 34% | 48.56 | 44.47 |
| **Tina-Open-RS2** | **51%** | **50.60** | **41.60** |
| Tina-Open-RS3 | 57% | 49.45 | 46.06 |

**Tina-Open-RS2是最亮的**：比对应baseline高了9分，AIME24上43.33%（baseline 26.67%），且只用$9就跑出来。

更关键的一点：**best checkpoint都出现在1个epoch以内**——19%到57%。这意味着你根本不需要把数据跑完，跑个零头就够了。这和full-parameter RL的"training越长越好"完全相反。

---

## Ablation：什么因素重要

### Dataset大小（最surprising的发现）

| Dataset | 大小 | Avg |
|---|---|---|
| OpenR1 | 93.7k | 49.26 |
| OpenThoughts | 66.1k | 49.19 |
| DeepScaleR | 40.3k | 48.38 |
| STILL-3 | 33k | 48.16 |
| Open-S1 | 18.6k | 48.56 |
| **Open-RS** | **7k** | **50.60** |
| LIMR | 1.39k | 48.47 |

**7k样本打败93.7k样本**。这和s1（https://arxiv.org/abs/2501.19393）"1k high-quality CoT就够"的结论呼应——**数据质量 >> 数据量**。给一堆垃圾数据，跑得再久也没用。

### Learning rate

测了5e-6 / 1e-6 / 5e-7，**1e-6最佳**（48.47 vs 47.87 vs 47.91）。差距不大，说明对lr不太敏感，不需要extensive tuning。

注意这是LoRA参数的lr。考虑scaling factor 4，等效full FT lr约4e-6。

### LoRA rank

| Rank | Avg |
|---|---|
| 4 | 47.72 |
| 8 | 47.89 |
| **16** | **48.92** |
| 32 | 48.47 |
| 64 | 46.95 |

中间rank最优，两端退化。rank太小capacity不够，rank太大容易过拟合+噪声多。

### RL算法：GRPO vs Dr.GRPO

Dr.GRPO（https://arxiv.org/abs/2503.20783）修正了GRPO对长response的偏置。结果：
- peak performance基本一样（49.45 vs 49.53）
- **Dr.GRPO在17% epoch就达到best，GRPO要57%**

也就是说Dr.GRPO的**sample efficiency**高得多，实际工程上省时间省电省钱。

---

## 最有意思的发现：Phase Transition

### "Less is More"现象

Figure 3里把performance vs. training FLOPs画出来，结果：
- Full-parameter baselines：FLOPs越多，performance越高（符合scaling law直觉）
- Tina (LoRA)：**FLOPs越多，performance反而下降**

完全反直觉。一般我们相信"more compute = better"，但LoRA-RL里相反——best checkpoint出现得非常早，继续train反而退化。

### 为什么会这样？Format Hypothesis

作者的核心假说：**LoRA-RL主要学的是output format/structure，base model的knowledge基本不动**。

逻辑链：
1. RL的reward很大程度上奖励format（step-by-step CoT, `<answer>` tag等）
2. LoRA只改很小一部分参数，没法深度重构knowledge
3. 所以LoRA快速学会"怎么把已有知识组织成reasoning format"
4. Full-parameter RL则会深度重构knowledge，代价是更多compute且可能遗忘

这个假说和Allen-Zhu & Li的"Physics of LMs Part 3.3"（https://arxiv.org/abs/2503.06504）吻合：大模型存knowledge，小模型可以被有效引导出format。

### Phase Transition现象（Figure 4-12）

看training logs，会发现一个很sharp的现象：
- **Format reward** 会先上升、到一个点后突然崩塌或不稳定
- **Completion length** 会先下降、到一个最低点后反弹
- **Accuracy reward** 变化相对平稳，没有对应的sharp transition

把这个turning point叫phase transition。最关键的observation是：

**Best-performing checkpoint总是出现在phase transition之前或附近**。

直觉解读：LoRA快速学会format → format reward上升 → length压缩到最精炼 → 某个点上format优化饱和/不稳定 → 继续train就开始退化。

也就是说：**学会format就够用了，继续train反而把已有的好状态破坏掉**。这正好解释了"less is more"现象。

不同数据集上的phase transition观察（Appendix E）：
- Tina-DeepScaleR / Tina-STILL-3 / Tina-Open-RS1/2/3：清晰的phase transition，best ckpt在transition之前
- Tina-OpenR1 / Tina-OpenThoughts：有transition，但best ckpt在transition之后（例外）
- Tina-LIMR系列：没观察到transition，可能因为数据太小（1.39k）训练太短没形成稳定pattern

---

## Cost Breakdown

Table 1的细节值得看：

| 任务 | Training cost | Eval cost | Total |
|---|---|---|---|
| 复现所有实验 | $396 | $130 | $526 |
| 复现主实验 | $213 | $62 | $275 |
| 复现每个主实验的best ckpt | $80 | $5 | $85 |
| 复现best performance任务的所有ckpt | $14 | $17 | $31 |
| **复现best performance任务的best ckpt** | **$8** | **$1** | **$9** |

$9就是Abstract里那个$9的来历——只跑Tina-Open-RS2的step 450这个checkpoint。如果你想复现所有实验，总共也才$526，比一张A100一天的cloud价格还便宜。

---

## 局限和我的take

### 局限

1. **Base model ceiling**：1.5B的绝对reasoning能力天花板有限，复杂多步reasoning可能够不着
2. **任务范围**：只测了math/science reasoning，coding reasoning没验证
3. **Hyperparameter**：作者故意不做hyperparameter search（省成本），可能还有未挖掘的性能

### 我的intuition

这篇paper最大的贡献不是"$9训练一个reasoning model"这个噱头，而是**揭示了一个反直觉的现象**：在RL+reasoning场景下，LoRA这种极简parameter-efficient方法不仅够用，甚至**因为它的限制反而成了优势**——逼着model只学format不破坏knowledge，于是best checkpoint出现得早，又快又好。

这背后的深层insight可能是：**reasoning能力的核心不在knowledge容量，而在format/schema的结构化能力**。Full-parameter RL想同时学knowledge和format，代价高且容易互相干扰；LoRA-RL只学format，反而干净。

对社区的意义：
- 普通研究者$10就能复现/扩展reasoning model研究
- RL+reasoning的训练成本被打破垄断，不再是"大厂专属"
- 对"参数高效微调在RL中的角色"这个研究方向开了一个头

后续值得探索的方向：
- 这个phase transition现象在更大model（7B/32B）上还成立吗？
- 如果把LoRA换成其他PEFT方法（DoRA, rsLoRA等），phase transition行为会怎么变？
- 能不能设计一个early stopping rule，自动停在phase transition之前？

---

## 关键参考链接

- 项目主页：https://shangshangwang.notion.site/tina
- 代码：https://github.com/shangshang-wang/Tina
- Training logs：https://wandb.ai/upup-ashton-wang-usc/Tina
- 模型权重：https://huggingface.co/Tina-Yi
- OpenR1 (training framework)：https://github.com/huggingface/open-r1
- DeepSeek-R1 (base distillation source)：https://arxiv.org/abs/2501.12948
- GRPO (DeepSeekMath)：https://arxiv.org/abs/2402.03300
- Dr.GRPO：https://arxiv.org/abs/2503.20783
- LoRA原paper：https://arxiv.org/abs/2106.09685
- Allen-Zhu & Li "Physics of LMs Part 3.3"：https://arxiv.org/abs/2503.06504
- s1 (1k data的灵感来源)：https://arxiv.org/abs/2501.19393
- DeepScaleR (baseline)：https://agentica-project.com/
- Open-RS (baseline)：https://arxiv.org/abs/2503.16219
- KL approximation (Schulman k3 estimator)：http://joschu.net/blog/kl-approx.html
- L40S pricing：https://www.cudocompute.com/products/gpu-cloud/nvidia-l40s

---

# Tina: Tiny Reasoning Models via LoRA 深度讲解

## 1. 论文核心思想与定位

这篇paper来自USC的Shangshang Wang等人，核心问题非常直接：**用最少的钱，能不能在RL里让小模型学会reasoning？**

答案令人意外——能，而且便宜到令人发指。最佳checkpoint（step 450 of Tina-Open-RS2）在AIME24上达到43.33% Pass@1，平均分50.60%，而total cost（training + evaluation）仅 **$9 USD**。对比之下，同等base model上full-parameter RL的SOTA（如DeepScaleR-1.5B-Preview）花费高出约260倍。

项目主页：https://shangshangwang.notion.site/tina  
代码：https://github.com/shangshang-wang/Tina  
Weights & Biases logs：https://wandb.ai/upup-ashton-wang-usc/Tina  
模型权重：https://huggingface.co/Tina-Yi

---

## 2. 技术架构拆解

### 2.1 Base Model选择

base是 `DeepSeek-R1-Distill-Qwen-1.5B`，这是一个经过distillation的1.5B模型。选择它有一个关键动机：**它已经有distillation带来的初始reasoning aptitude**。这一点很重要，因为paper的核心claim是"LoRA-RL带来增量提升"，所以base必须足够强，否则增量无法干净地归因于LoRA-RL。

这一点和DeepSeek-R1-Distill系列的lineage（DeepSeek + Qwen2.5架构）有关。Allen-Zhu and Li的"Physics of Language Models Part 3.3"（https://arxiv.org/abs/2503.06504 提到large LMs store broader world knowledge）的insight被用来支撑hypothsis——大模型存knowledge，小模型则可以被有效引导出reasoning format。

### 2.2 训练框架：OpenR1 + LoRA + GRPO

训练pipeline基于 **OpenR1**（https://github.com/huggingface/open-r1），它本身是DeepSeek-R1的复现。OpenR1整合了：
- **Accelerate**（https://github.com/huggingface/accelerate）
- **TRL**（Transformer Reinforcement Learning，https://github.com/huggingface/trl）
- **DeepSpeed ZeRO**（https://arxiv.org/abs/1910.02054）

RL算法核心是 **GRPO**（Group Relative Policy Optimization），来自DeepSeekMath（https://arxiv.org/abs/2402.03300），以及在ablation中测试的 **Dr.GRPO**（https://arxiv.org/abs/2503.20783）。

### 2.3 硬件最小化策略

这里有一个非常聪明的工程决策：他们只用 **2块 NVIDIA L40S GPU**（约 $1/GPU-hour on Cudo Compute, https://www.cudocompute.com/products/gpu-cloud/nvidia-l40s）。常规GRPO配置往往用3+ GPU（一台专门跑vLLM inference engine），但Tina把training和vLLM co-locate在同样的2块GPU上，通过限制vLLM的GPU memory utilization（设为0.4）来实现。

代价是wall-clock时间更长，但硬件门槛大幅降低——这对democratization很关键。

---

## 3. GRPO公式详解

GRPO是PPO的变体，核心创新是**去掉value network**，用group-based baseline估计advantage。这对1.5B这种小模型特别友好，因为不需要额外维护一个critic model。

完整objective（见paper附录B.1）：

$$
\mathbb{E}_{\{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \min\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clipped}\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon \right) A_i \right) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) \right) \right]
$$

变量含义：
- $q$：输入question/prompt
- $G = \{o_1, o_2, \ldots, o_G\}$：从old policy $\pi_{\theta_{old}}$ 采样的一组output（group size，本文中是4）
- $\pi_\theta$：当前正在优化的policy（带LoRA的model）
- $\pi_{\theta_{old}}$：用于采样的旧policy（参数被周期性同步）
- $\pi_{ref}$：reference policy（通常冻结，用于KL约束防止漂移太远）
- $A_i$：第 $i$ 个output的advantage
- $\epsilon$：clipping range（控制policy update幅度，类似PPO）
- $\beta$：KL penalty权重
- $\mathbb{D}_{KL}$：KL散度

**Advantage计算**（group-relative normalization）：

$$
A_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}
$$

- $r_i$：第 $i$ 个output获得的reward（可能是accuracy + format + length等组合）
- 减去group mean：消除baseline偏差
- 除以group std：归一化scale

这个normalization非常关键——它让reward信号变成相对的（"这个output比group内平均水平好/差多少"），而不是绝对的。对小数据集尤其有用。

**KL divergence近似**：

$$
\mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) = \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - \log \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - 1
$$

这是 **Schulman的k3 estimator**（http://joschu.net/blog/kl-approx.html），unbiased且计算上只需forward pass over $\pi_{ref}$ 和 $\pi_\theta$，无需遍历所有tokens的积分。

---

## 4. LoRA公式与变量

LoRA（https://arxiv.org/abs/2106.09685）的核心：**冻结原权重，只训练低秩增量**。

给定frozen weight $W_0 \in \mathbb{R}^{d \times k}$，forward pass从：

$$
h(x) = W_0 x
$$

变为：

$$
\hat{h}(x) = W_0 x + AB x
$$

变量含义：
- $W_0 \in \mathbb{R}^{d \times k}$：预训练好的frozen weight（$d$ = output dimension, $k$ = input dimension）
- $A \in \mathbb{R}^{d \times r}$：down-projection矩阵（随机初始化，通常Gaussian）
- $B \in \mathbb{R}^{r \times k}$：up-projection矩阵（初始为0，保证训练开始时 $\Delta W = AB = 0$）
- $r \ll \min(d, k)$：rank（Tina默认 $r=32$，ablation测了4/8/16/32/64）
- $x$：input activation

Tina的配置（Table 5）：
- LoRA modules：query, key, value, dense（attention和FFN都加）
- Rank = 32, Alpha = 128（即scaling factor $\alpha/r = 4$）
- Dropout = 0.05
- Precision：BF16-mixed

**为什么Alpha=128 / Rank=32 = 4的scaling？**  
LoRA实际生效的更新是 $\frac{\alpha}{r} AB x$，所以scaling=4意味着把低秩更新的magnitude放大4倍。这对补偿"低秩容量小"的问题有帮助。

---

## 5. Reward设计

Table 6展示了reward组合，非常值得细看：

| Model | Reward Type | Weights |
|---|---|---|
| Tina-STILL-3 | Accuracy, Length | 2, 1 |
| Tina-DeepScaleR | Accuracy, Format | 2, 1 |
| Tina-Open-RS3 | Cosine, Format | 2, 1 |
| Tina-Open-RS2 | Accuracy, Format | 2, 1 |
| Tina-OpenR1 | Accuracy, Cosine, Format, Length, Tag Count, Reasoning Steps, Repetition Penalty | 1,1,1,1,1,1,1 |
| Tina-OpenThoughts | Accuracy, Cosine, Format, Length, Tag Count, Reasoning Steps, Repetition Penalty | 1,1,1,1,1,1,1,1,1,1,1 |

值得注意：**Accuracy reward权重总是2**，format类reward权重是1。这暗示作者认为"对不对"比"格式对不对"重要2倍，但format奖励仍然必要——这恰好对后面要讲的phase transition现象埋下伏笔。

reward类型包括：
- **Accuracy**：最终答案对不对（verifiable reward，如math problem答案）
- **Format**：是否符合 `<answer>...</answer>` 这种结构
- **Length**：response长度相关
- **Cosine**：可能是cosine-scaled reward，按难度/进度scaling
- **Tag Count**：reasoning step tag数量
- **Reasoning Steps**：中间推理步数
- **Repetition Penalty**：惩罚重复

---

## 6. 实验结果深度解读

### 6.1 Baseline re-evaluation（Table 2）

这是这篇paper非常负责任的一点——他们用统一的lighteval + vLLM重新评估所有baseline，因为不同paper用的eval framework不一样（verl, lighteval, lm-eval-harness），generation hyperparameters也不同，直接比分数不靠谱。

重新评估结果（平均分）：
- DeepSeek-R1-Distill-Qwen-1.5B（base）：41.18
- STILL-3-1.5B-preview：44.86
- DeepScaleR-1.5B-Preview：48.74
- Open-RS1：44.47
- Open-RS2：41.60
- Open-RS3：46.06

### 6.2 Tina主结果（Table 3）

| Tina Model | Steps (% of 1 epoch) | Avg | Baseline Avg |
|---|---|---|---|
| Tina-STILL-3 | 53% | 48.16 | 44.86 |
| Tina-DeepScaleR | 19% | 48.38 | 48.74 |
| Tina-Open-RS1 | 34% | 48.56 | 44.47 |
| Tina-Open-RS2 | 51% | **50.60** | 41.60 |
| Tina-Open-RS3 | 57% | 49.45 | 46.06 |

**关键观察**：
1. 几乎所有Tina模型都打平或超过对应full-parameter baseline
2. 最强的是Tina-Open-RS2，平均50.60，比baseline（41.60）高出9分
3. 训练只用19%-57% of 1 epoch就达到best checkpoint——这是"less compute, more performance"的核心证据

AIME24上Tina-DeepScaleR / Tina-Open-RS1 / Tina-Open-RS2都达到43.33%，已经超过了DeepScaleR-1.5B-Preview baseline的36.67%。

### 6.3 Ablation Studies（Table 4）

**Dataset大小**：最surprising的是Tina-Open-RS只用 **7k样本**就达到50.60 avg，而Tina-OpenR1用93.7k样本只有49.26。这强力支持了"数据质量 >> 数据量"的intuition，和s1（https://arxiv.org/abs/2501.19393）的"1k high-quality CoT数据"思想一致。

**Learning rate**：测试5e-6 / 1e-6 / 5e-7，1e-6最佳（48.47 avg）。注意这个lr是相对LoRA parameters的，所以等效于full fine-tune的lr要乘以scaling factor（4），即等效4e-6 full FT lr。

**LoRA rank**：测4/8/16/32/64，结果：
- rank 4：47.72
- rank 8：47.89
- rank 16：48.92（最佳）
- rank 32：48.47
- rank 64：46.95

中间值最优，两端下降。这和LoRA文献中"中等rank最稳定"的常见发现一致——rank太小capacity不够，太大容易过拟合且退化。

**RL算法**：GRPO vs Dr.GRPO。Dr.GRPO（https://arxiv.org/abs/2503.20783）针对GRPO的length bias做了修正，结果peak performance差不多（49.45 vs 49.53），但Dr.GRPO在17% epoch就达到best，GRPO需要57%。这说明Dr.GRPO的sample efficiency显著更高。

---

## 7. 核心Insight：Phase Transition

这是paper最theoretical的部分，也是最有意思的发现。

### 7.1 "Less is More"现象（Figure 3）

把reasoning performance vs. 训练FLOPs画出来，会发现：
- Full-parameter baseline：performance随FLOPs增加而上升（符合scaling law直觉）
- Tina (LoRA)：**performance随FLOPs增加而下降**

这是反直觉的。一般假设more compute = better，但LoRA这里却相反——best checkpoint出现得非常早。

### 7.2 "Learn Format, Maintain Knowledge"假说

作者提出的核心hypothsis：**LoRA-RL的有效性来自它能快速学习RL奖励的output format/structure，同时保留base model的预训练知识**。

这个假说的支撑逻辑：
1. Reasoning tasks的reward很大程度上是format-based（step-by-step chain-of-thought、`
