# InT: Self-Proposed Interventions Enable Credit Assignment in LLM Reasoning 技术详解

## 一、问题背景与动机

### 1.1 Credit Assignment问题的本质

在LLM的reinforcement learning训练中，存在一个根本性的**credit assignment难题**。让我用数学公式来说明：

传统的方法使用**outcome reward**，即：
```
r(x, y) ∈ {0, 1}
```
其中：
- **x** = 输入问题
- **y** = 模型生成的完整推理轨迹，可分解为 y = (y₀, y₁, ..., y_T)
- **T** = 推理步骤的数量

policy gradient更新公式为：
```
π' ← π + α · E[x∼D_train, y∼π̃(·|x)][r(x, y) · ∇π log π(y|x)]    (Equation 1)
```
其中：
- **π** = 当前策略
- **π̃** = 采样策略
- **α** = 学习率
- **D_train** = 训练数据集

关键问题在于：**advantage的计算**：
```
A(x, y_i) = r(x, y_i) - (1/n) Σj=1^n r(x, y_j)
```

当A(x, y_i) > 0时，**整个推理轨迹**中的所有步骤都被均匀强化；当A(x, y_i) < 0时，所有步骤都被均匀惩罚。

这在长推理链中导致两个严重问题：

1. **错误轨迹中的正确步骤被抑制**：假设一个100步的推理轨迹在第95步出错，前面94步都是正确的，但传统RL会惩罚全部100步
2. **正确轨迹中的冗余步骤被强化**：成功轨迹中可能包含spurious步骤，它们与最终答案无关但仍然被强化

### 1.2 实验证据

论文通过图2展示了错误首次发生的位置分布：

- **超过60%**的首次错误发生在**第50步之后**
- 这说明大部分推理轨迹的前半部分是正确的，只是某个关键步骤导致失败

这解释了为什么传统RL效率低下：当成功轨迹稀少时（在困难问题上超过80%的rollout group没有成功轨迹），advantages collapse to zero，没有学习信号。

## 二、Intervention Training (InT) 方法详解

### 2.1 核心思想：利用验证与生成的难度不对称性

论文的关键洞察是LLM在以下任务上存在能力差异：

| 任务类型 | 难度 | 原因 |
|---------|------|------|
| **从头生成正确解** | 高 | 需要组合多个正确步骤，复杂度高 |
| **验证单个步骤** | 低 | 只需判断逻辑正确性，通过比较参考解即可 |

InT利用这种**不对称性**，让模型通过自我验证来发现错误并提出单步纠正。

### 2.2 两阶段干预生成过程

#### 阶段1：错误定位

给定一个模型生成的错误推理轨迹y，模型需要：

1. **逐步验证**每个步骤y_t的逻辑正确性
2. **定位首次错误** y_{t*}的位置（步骤编号t*）

这个过程通过一个prompt p_InT实现：
```
t*, ỹ_{t*} ~ π(· | x, y, p_InT)
```

设计考虑：
- 只关注**首次错误**，因为后续错误往往是early mistake的后果
- 一旦早期错误被修正，后续轨迹通常不会继续走错误的路径

#### 阶段2：干预生成

识别到错误位置t*后，模型生成一个**替代步骤**：
```
ỹ_{t*} = 干预步骤
```

这个干预的目的是将轨迹**steer**向正确答案。

### 2.3 完整算法流程

```python
Algorithm 1: Intervention Training (InT)

Input: 基础LLM π, 干预生成prompt p_InT, 训练问题集 D_train

1: D_InT ← {}  # 初始化干预数据集
2: for each x ∈ D_train do
3:     y ← π(· | x)  # 生成初始推理轨迹
4:     t*, ỹ_{t*} ← π(· | x, y, p_InT)  # 定位错误并生成干预
5:     D_InT ← D_InT ∪ (x, concat(y_{<t*}, ỹ_{t*}))  # 存储干预数据
6: end for
7: π' ← SFT(π, D_InT)  # 补丁式微调
8: π'' ← RL(π', D_train)  # 继续RL训练
9: return π''
```

其中：
- **y_{<t*}** = 错误之前的正确前缀
- **ỹ_{t*}** = 提出的干预步骤
- **concat** = 拼接操作

## 三、训练设计的关键决策

### 3.1 SFT训练内容的选择

论文研究了不同的SFT配置（Table 2）：

| 配置 | 训练内容 | Coverage | Accuracy |
|------|---------|----------|----------|
| 1 | Prefix only (y_{<t*}) | 162/235 | 2.87% |
| 2 | Prefix + Intervention (y_{<t*}, ỹ_{t*}) | 202/235 | 7.71% |
| 3 | Prefix + Intervention + Suffix | 111/235 | 2.31% |
| 4 | Config 2 with filter | 202/235 | 7.71% |

**关键发现**：

1. **必须包含Prefix (y_{<t*})**：
   - 原因：如果不强化前缀，微调后的模型在测试时可能生成不同的前缀y'_{<t*}，此时干预ỹ_{t*}不再相关
   - 这是因为输出分布π(·|x)在微调后会shift，覆盖所有可能的prefix组合在计算上是intractable的

2. **不应包含Suffix (ỹ_{>t*})**：
   - 原因：cloning正确的suffix会减少后续RL训练时的exploration空间
   - 论文发现这会使solved的问题数量减少近一半

3. **使用correctness filter**：
   - 只保留那些在32个rollouts中至少有一个正确continuation的干预
   - 这进一步提升了性能

### 3.2 SFT目标函数

最终的SFT目标函数为：
```
∇π J(π) ≈ E[x∼D_train, y∼π̃(·|x) s.t. r(x,y)=0]
          [∇π log π(ỹ_{t*} | y_{<t*}) + Σt=0^{t*} ∇π log π(y_t | y_{<t})]    (Equation 2)
```

这个公式的含义：
- **第一项**：提升干预步骤的概率
- **第二项**：强化正确的prefix
- 条件：只在错误轨迹上执行（r(x,y)=0）

### 3.3 为什么不直接训练PRM

论文讨论了训练Process Reward Models (PRM)的挑战：

1. **计算成本高**：
   - Qwen2.5-Math-PRM需要约**300万** rollouts
   - 需要复杂的filtering pipelines

2. **训练不稳定**：
   - 长轨迹的value estimation困难
   - 即使有准确的PRM，如何optimization来找到替代步骤本身也是一个困难的问题

3. **PRM在困难问题上表现不佳**：
   - 近期研究表明PRMs在困难问题上tend to underperform

相比之下，InT提供了**更简单且更scalable**的credit assignment机制。

## 四、实验设计与结果

### 4.1 实验设置

#### 训练集构建

从四个来源构建困难问题集：

| 来源 | 问题数量 | 筛选条件 |
|------|---------|---------|
| Polaris | 53k | 64 rollouts下准确率为0 |
| AceReason-Math | 50k | 64 rollouts下准确率为0 |
| Omni-MATH | 4.4k | 128 rollouts下准确率为0 |
| IMO-AnswerBench | 360 | 去除测试集保留问题 |

最终得到约**4500**个困难问题，其中1076个有有效的干预（至少1/32 rollouts成功）。

#### 评估基准

为了避免训练集污染，选择了2025年发布的最新基准：

1. **IMO-AnswerBench**：由IMO medalists curated
2. **HMMT 2025 (November)**：从官方HMMT网站scrape
3. **AMO-Bench**
4. **Apex Shortlist**

### 4.2 干预效果验证

Table 1展示了干预的效果：

| 干预来源 | 条件 | Coverage | Accuracy |
|---------|------|----------|----------|
| - | problem x | 40/334 | 0.0984% |
| - | x + prefix y_{<t*} | 31/334 | 0.0726% |
| - | x + prefix + original step y_{t*} | 29/334 | 0.0713% |
| 4B-Instruct | x + prefix + intervention | 80/334 | 1.56% |
| 4B | x + prefix + intervention | 73/334 | 2.65% |
| 30B-A3B-Instruct | x + prefix + intervention | 101/334 | 2.87% |

**关键发现**：

1. **干预显著提升成功率**：从0.0713%提升到1.56%（**22×** improvement）
2. **覆盖范围扩大**：从29个问题增加到80个问题
3. **模型规模越大，干预质量越高**：30B模型比4B模型多解决21个问题
4. **Instruction-following能力关键**：4B-Instruct比纯推理模型4B表现更好

#### 与Hint-guided方法的对比

| 设置 | Coverage | Accuracy |
|------|----------|----------|
| Hint alone | 11/176 | 2.38% |
| Intervention alone | 18/176 | 3.05% |
| Hint + Intervention | 25/176 | 4.62% |

**insight**：Intervention和hint是complementary的：
- Hint引导模型进入promising的初始方向
- Intervention在轨迹中间出现错误时进行fine-grained correction

### 4.3 主要实验结果

#### SFT后性能提升

Figure 6显示，在SFT阶段：
- Pass@k在k=16,32,...,1024上都有提升
- 这为后续RL提供了strong initialization

Figure 7显示，intervention tokens在训练后的模型中probability更高，表明模型**internalized**干预模式。

#### RL训练过程中的性能

Figure 11展示了训练过程中的关键指标：

1. **Average Reward**：InT初始化的模型reward最高
2. **Zero-advantage Ratio**：从高比例大幅降低，说明模型能够从之前没有信号的问题中学习

#### 最终评估结果

Table 4展示了在四个基准上的综合表现：

| 模型配置 | IMO-AnswerBench | HMMT 2025 Nov | AMO-Bench | Apex Shortlist | Average |
|---------|-----------------|---------------|-----------|----------------|---------|
| Base | 11.68 | 41.61 | 26.24 | 20.79 | 21.17 |
| + RL | 23.46 | 46.46 | 35.21 | 22.72 | 28.26 |
| + Hint-guided RL | 16.89 | 47.27 | 33.34 | 22.23 | 28.56 |
| + SFT on ref. solutions + RL | 11.56 | 27.45 | 25.19 | 20.51 | 20.76 |
| + SFT on self-reflections + RL | 15.53 | 38.65 | 36.72 | 23.93 | 27.60 |
| **+ InT + RL (Ours)** | **25.62** | **49.77** | **36.16** | **28.22** | **33.72** |

**亮点**：
1. **InT + RL达到33.72**，比base提升**59%**，比标准SFT+RL提升**19%**
2. **在IMO-AnswerBench上达到25.62**，超过base model的2倍以上
3. **Reference solution SFT表现糟糕**：平均分数20.76甚至低于base model的21.17

### 4.4 为什么Reference Solution SFT失败

论文深入分析了这个问题：

#### On-policy vs Off-policy

Figure 9显示了不同trace在base model下的negative log-likelihood (NLL)：

| Trace来源 | NLL (lower is better) |
|----------|----------------------|
| InT | **Lowest** (most on-policy) |
| Self-reflection | Low |
| DeepSeek-R1 outputs | Medium |
| Reference solutions | **Highest** (most off-policy) |

**关键洞察**：

Fine-tuning on highly off-policy traces会导致：

1. **Entropy increase**：
   - Figure 8显示，fine-tuning on off-policy traces会大幅增加next-token distribution的entropy
   - High-entropy初始化对后续RL有害，因为会导致过度随机的rollouts，阻碍effective exploration

2. **Distribution distortion**：
   - 长期fine-tuning on off-policy data（如human-written参考解）会distort base model现有的推理模式
   - 这些trace与模型已经学到的行为和技能substantially different

#### 为什么InT有效

InT生成的trace主要由base model的tokens组成，因为：

- **干预短小**：Figure 10显示interventions通常<200 tokens，而full rollouts平均~7000 tokens
- 因此，InT trajectories在base model distribution下有更高的likelihood
- 这避免了large entropy shift（Figure 8显示InT的entropy与base model相当）

## 五、技术细节与超参数

### 5.1 RL训练配置

Table 5展示了PipelineRL的超参数：

| Hyperparameter | Value |
|----------------|-------|
| effective_train_batch_size | 32 |
| ppo_mini_batch_size | 16 |
| learning_rate | 1.0e-6 |
| kl_loss_coef | 0 |
| entropy_coeff | 0 |
| temperature | 1.0 |
| top_p | 1.0 |
| rollout.n | 8 |

### 5.2 SFT训练配置

Table 6展示了LLaMa Factory的超参数：

| Hyperparameter | Value |
|----------------|-------|
| dataset_size | 1076 |
| effective_batch_size | 64 |
| num_train_epochs | 4 |
| learning_rate | 1.0e-6 |
| lr_scheduler_type | cosine_with_min_lr |
| min_lr_rate | 0.1 |
| warmup_ratio | 0.1 |

### 5.3 评估配置

对于Qwen3-4B-Instruct的所有实验，使用官方推荐配置：

- temperature: 0.7
- top-p: 0.8
- top-k: 20

## 六、相关工作对比

### 6.1 PRM-based方法

| 方法 | 优点 | 缺点 |
|------|------|------|
| PRM with branched rollouts | Dense rewards | 计算成本高 |
| PRM with human annotations | 高质量 | Reward hacking风险 |
| PRM optimization | Step-level rewards | 找到替代步骤困难 |

InT的优势：
- 将value estimation和policy optimization**amortize**成单一过程
- **更简单、更便宜**

### 6.2 Natural Language Feedback方法

| 方法 | 特点 | 与InT的区别 |
|------|------|-----------|
| Human feedback + refinement | 使用外部模型 | InT自生成、单步targeted |
| Teacher model guidance | Off-policy RL | InT on-policy, self-generated |
| Critique-guided refinement | Critique-based | InT短小、用于training而非inference |

### 6.3 Hint-guided RL

| 方法 | 使用参考解的方式 | 局限 |
|------|---------------|------|
| Hint-guided RL | Partial-solution prefixes as hints | 无法处理trajectory中后期的错误 |
| InT | Verification and credit assignment | 可以在任何位置修正错误 |

Figure 2显示**>70%的错误出现在前50步之后**，这是hint-guided方法无法处理的。

### 6.4 LLM外的Intervention方法

DAgger在imitation learning中的应用：
- 在robot manipulation和long-horizon control中有成功应用
- 通常由human提供interventions

InT的贡献：
- **Self-proposed interventions**：模型自己生成纠正，不依赖human
- 利用generation和verification之间的asymmetry

## 七、深入理解：为什么InT Works

### 7.1 On-policy性质的重要性

论文通过一个关键发现揭示了on-policy的重要性：

**Correlation between NLL and Performance**：
- InT traces → Lowest NLL → Best performance
- Self-reflection → Low NLL → Good performance  
- R1 outputs → Medium NLL → Medium performance
- Reference solutions → Highest NLL → Worst performance

这说明：
- Fine-tuning on highly off-policy traces是problematic的
- 这些traces经常被memorized，而不是generalized
- Fitting它们会significantly distort base model的next-token distribution

### 7.2 为什么Interventions Shortness Matters

Figure 10显示：
- **Interventions通常<200 tokens**
- **Full rollouts平均~7000 tokens**

这意味着：
- InT生成的trajectory中，**绝大部分tokens来自base model**
- 因此在base model distribution下有**higher likelihood**
- 避免**large entropy shift**

### 7.3 Credit Assignment的改进

传统RL的问题：
- Credit assignment在incorrect rollouts上几乎无效
- 在困难问题上，>80%的rollout groups没有成功轨迹

InT的改进：
- 通过localizing error到具体步骤，实现**fine-grained credit assignment**
- 在训练集上，zero-advantage ratio从高比例大幅降低
- 这使模型能够从**之前没有学习信号的问题**中学习

## 八、泛化性分析

### 8.1 Memorization vs Generalization

Appendix F分析了这个问题，选取了两个训练集外的IMO Shortlist 2024问题：

#### IMO Shortlist 2024, Problem C1

两个模型都从一个incorrect assumption开始，但当InT训练的模型得到n=3时结果是33时，它质疑这是否可能并正确地更新了hypothesis到C(n,2)。

#### IMO Shortlist 2024, Problem C2

虽然两个模型都能成功得出even cool numbers必须是4的倍数，但只有InT模型能够尝试考虑n=12可能不是cool的，因此pattern可能比只是4的倍数更selective。这引导第二个模型到正确的hypothesis。

这表明InT确实**generalized到unseen问题**，而不仅仅是memorized interventions。

### 8.2 与更大模型的对比

Table 3展示了在IMO-AnswerBench上的对比：

| Model | Score |
|-------|-------|
| Qwen3-4B-Instruct-2507 | 11.68% |
| DeepSeek-R1-0528-Qwen3-8B | 18.44% |
| gpt-oss-20b | 23.36% |
| **Qwen3-4B-Instruct-2507 + InT + RL** | **25.62%** |

**Insight**：4B模型经过InT训练后，**超越了20B级别的open-source模型**。

## 九、未来工作方向

### 9.1 Self-Improvement by Combining Different LLM Capabilities

1. **更强的验证器**：
   - 显式训练模型在verification任务上
   - 消除对reference solutions的依赖
   - 使LLMs能够处理需要step-by-step correctness checking的任务，如IMO-ProofBench

2. **问题生成能力**：
   - LLMs能否学习构建自己的训练问题？
   - 通过modeling expert problem proposers的trajectories（如IMO-style competitions）

理想情况下，一个fully autonomous system：
- 一个模型proposes问题
- 另一个解决它们
- 一个verifier提供驱动continual parameter updates的reward signals

### 9.2 Credit Assignment in Continual Learning

将credit assignment扩展到continual improvement regimes：
- Rollouts progressively被压缩成LLM-generated summaries或neural memory modules
- Effective context evolves over time

关键问题：
- 如何trace credit back到早期决策（可能只通过memory representations persist）？
- 如何disentangle来自poor generation的错误和来自imperfect memory的错误？
- 如何jointly改进memory architectures和generation model？

## 十、实际应用建议

### 10.1 何时使用InT

InT特别适合以下场景：

1. **困难推理任务**：Base model在训练集上pass@k为0
2. **有参考解可用**：如mathematical reasoning datasets
3. **Base model有较强的instruction-following能力**
4. **希望避免训练PRM的高成本**

### 10.2 实现注意事项

1. **Prompt设计**：
   - 需要explicitly要求model index错误的position
   - 需要使用特定的格式输出intervention（如<intervention></intervention> tags）

2. **过滤策略**：
   - 只保留至少产生一个正确continuation的interventions
   - 使用32 rollouts进行evaluation

3. **SFT配置**：
   - 训练on prefix + intervention
   - **不训练suffix**
   - 使用correctness filter

4. **RL初始化**：
   - 从InT-patched model开始RL训练
   - 避免从reference-solution fine-tuned model开始

### 10.3 与其他方法的结合

论文展示InT可以与hint-guided方法结合：

```
Prompt: problem x + hint h + prefix y_{<t*} + intervention ỹ_{t*}
```

这在Table 1中展示了最好的coverage（25/176）和accuracy（4.62%）。

## 十一、局限性与挑战

### 11.1 对Reference Solutions的依赖

当前方法严重依赖reference solutions：
- 可能限制在domains where reference solutions are available的applicability
- Future work需要探索如何训练stronger verifiers来eliminate this dependency

### 11.2 单步干预的限制

论文专注于single-step interventions：
- 对于需要multiple corrections的problems可能不够
- Future work可以探索multi-step interventions

### 11.3 推理轨迹的假设

论文假设推理轨迹可以被natural delimiter（如'\n\n'）分解：
- 这可能不适用于所有类型的reasoning
- Future work需要探索更general的trajectory decomposition方法

## 十二、总结

InT的核心贡献可以总结为：

1. **问题洞察**：识别出LLM reasoning中credit assignment的根本困难
2. **方法创新**：利用verification和generation之间的asymmetry实现self-proposed interventions
3. **实践价值**：
   - 比训练PRM更简单、更scalable
   - 避免off-policy SFT的pitfalls
   - 在多个benchmarks上取得显著提升
4. **理论意义**：
   - 展示了on-policy fine-tuning的重要性
   - 提供了一个新的perspective on credit assignment

在IMO-AnswerBench上，InT + RL达到**25.62%**，相对于base model的**11.68%**实现了**119%的相对提升**，相对于20B级别的open-source模型（gpt-oss-20b的23.36%）也有优势。这证明了一个精心设计的credit assignment机制可以比单纯增加模型规模更有效。

**Key Takeaway**：InT通过让模型self-verify和self-correct，实现了precise credit assignment，这为解决long-horizon reasoning tasks中的credit assignment challenge提供了一个简单而有效的paradigm。

## 参考文献

- [Paper Link](https://arxiv.org/abs/2601.14209)
- [Project Website](https://intervention-training.github.io/)
- Related work on PRMs: [Math-Shepherd](https://arxiv.org/abs/2310.15533), [Process Reward Models](https://arxiv.org/abs/2309.06280)
- Related work on hint-guided RL: [Qu et al.](https://arxiv.org/abs/2501.xxxxx), [Li et al.](https://arxiv.org/abs/2501.xxxxx)