# 我来详细讲解这篇关于**GDPO（Group reward-Decoupled Normalization Policy Optimization）**的论文。这是一篇针对多reward强化学习优化的重要工作。

## 📋 论文概览

**论文标题**：GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization

**作者**：Shih-Yang Liu等人（NVIDIA）

**核心问题**：在多reward强化学习设置中，直接应用**GRPO**（Group Relative Policy Optimization）存在严重的**reward collapse问题**，导致训练信号分辨率降低、收敛次优甚至训练失败。

**核心贡献**：提出了**GDPO**方法，通过解耦每个reward的group-wise归一化来保持不同reward组合之间的精细差异，从而实现更准确的多reward优化。

---

## 🔍 问题分析：GRPO的Reward Collapse问题

### 2.1 GRPO在多Reward设置下的问题

论文详细分析了直接应用GRPO到多reward优化时存在的根本缺陷。

#### 核心公式对比：

**GRPO方法**（原有方法）：
```
r_sum(i,j) = r_1(i,j) + ... + r_n(i,j)  // (1) 先求和所有rewards

A_sum(i,j) = [r_sum(i,j) - mean{r_sum(i,1),...,r_sum(i,G)}] / std{r_sum(i,1),...,r_sum(i,G)}  // (2) 然后归一化
```

**问题示例**（二进制reward，r1, r2 ∈ {0,1}）：
- Rollout组合 (0,2) 的 advantage = (−0.7071, 0.7071)
- Rollout组合 (0,1) 的 advantage = (−0.7071, 0.7071)

尽管一个是获得两个reward（0→2），另一个是只获得一个reward（0→1），但归一化后它们得到的**advantage完全相同**！这严重压缩了学习信号。

#### 直观理解：
```
(0,2): 一个rollout得了0分，另一个得了2分（两个reward都满足）
(0,1): 一个rollout得了0分，另一个得了1分（仅满足一个reward）

从学习角度看，(0,2)应该产生比(0,1)更强的学习信号，
因为同时满足两个reward比只满足一个reward更困难。
但GRPO的归一化抹去了这种区分！
```

Figure 2清晰展示了这个问题：两种不同的reward组合被映射到相同的advantage值。

---

## 💡 方法创新：GDPO架构

### 3.1 GDPO核心思想

GDPO的关键创新是**解耦归一化**：在求和之前，先对每个reward单独进行归一化。

#### 关键公式：

**Step 1: 单独归一化每个reward**
```
A_1(i,j) = [r_1(i,j) - mean{r_1(i,1),...,r_1(i,G)}] / std{r_1(i,1),...,r_1(i,G)}

A_n(i,j) = [r_n(i,j) - mean{r_n(i,1),...,r_n(i,G)}] / std{r_n(i,1),...,r_n(i,G)}  // (4)
```

**Step 2: 求和归一化后的advantages**
```
A_sum(i,j) = A_1(i,j) + ... + A_n(i,j)  // (5)

A^_sum(i,j) = [A_sum(i,j) - mean{A_sum(i',j')}] / [std{A_sum(i',j')} + ε]  // (6) batch-wise归一化
```

#### 架构优势：

| 特性 | GRPO | GDPO |
|------|------|------|
| 归一化时机 | 求和后归一化 | 求和前归一化（逐reward） |
| 信息保留 | 丢失reward维度差异 | 保持每个reward的相对差异 |
| Advantage分辨率 | 压缩（多个组合→相同值） | 精细（不同组合→不同值） |
| 数值稳定性 | 随reward数量增大而增大 | Batch-wise归一化保持稳定 |

### 3.2 效果量化分析

论文Figure 3展示了关键实验结果：

**设置1**: 二reward场景（固定2个rewards），变化rollout数量
```
不同rollout数量下的distinct advantage组数对比：

Rollout数=2:  GDPO显著多于GRPO
Rollout数=4:  GDPO的优势进一步扩大
Rollout数=8:  GDPO保持显著优势，而GRPO增幅很小
```

**设置2**: 固定4个rollouts，变化reward数量
```
不同reward数量下的distinct advantage组数对比：

Reward数=2:   GDPO > GRPO
Reward数=3:   GDPO的优势明显扩大
Reward数=5:   GDPO保持大优势，GRPO增长缓慢
```

**关键发现**：
- **GDPO consistently preserve a substantially larger number of distinct advantage groups**
- 随着rollout数量或reward数量的增加，GDPO的优势更加明显
- 这使得advantage估提供更具表达力的训练信号

---

## ⚙️ 实验验证

### 4.1 Experiment 1: Tool Calling

**任务设置**：
- 训练模型学习外部工具调用
- 两个rewards：
  - Format reward ℛ_format ∈ {0,1}: 检查输出格式是否正确
  - Correctness reward ℛ_correct ∈ [−3, 3]: 评估生成的tool call准确性

**训练配置**（Appendix D）：
```table
Parameter | Value
--- | ---
trainer.total_epochs | 15
data.train_batch_size | 512
actor_rollout_ref.actor.ppo_mini_batch_size | 128
data.max_prompt_length | 2048
actor_rollout_ref.actor.optim.lr | 1.00E-06
actor_rollout_ref.rollout.n | 4
algorithm.kl_ctrl.kl_coef | 0.001
```

**评估基准**: BFCL-v3（Berkeley Function Call Leaderboard）

#### 结果分析（Table 1）：

**Qwen2.5-Instruct-1.5B**:
```
指标                     | 无RL    | GRPO    | GDPO
---                      | ---     | ---     | ---
Live Overall Acc         | 37.89%  | 50.63%  | 55.36% (+4.73%)
Multi Turn Overall Acc   | 0.12%   | 2.04%   | 2.50% (+0.46%)
Non-Live Overall Acc     | 15.63%  | 37.87%  | 40.58% (+2.71%)
Avg Acc                  | 17.88%  | 30.18%  | 32.81% (+2.63%)
Correct Format           | 4.74%   | 76.33%  | 80.66% (+4.33%)
```

**Qwen2.5-Instruct-3B**:
```
指标                     | 无RL    | GRPO    | GDPO
---                      | ---     | ---     | ---
Live Overall Acc         | 63.57%  | 69.23%  | 71.22% (+1.99%)
Multi Turn Overall Acc   | 1.38%   | 3.14%   | 4.59% (+1.45%)
Non-Live Overall Acc     | 30.75%  | 45.24%  | 46.79% (+1.55%)
Avg Acc                  | 31.90%  | 39.20%  | 40.87% (+1.67%)
Correct Format           | 58.37%  | 81.64%  | 82.23% (+0.59%)
```

**关键观察**（Figure 4）：
- **GDPO在format和correctness两个rewards上都达到更高收敛值**
- GDPO在格式reward上收敛较慢（更大方差），但最终超越GRPO
- 在early stage，GDPO在correctness上进步更快

#### 4.1.1 消融实验：移除标准差归一化的效果

论文测试了GRPO w/o std（移除公式2的分母项）：

**结果（Table 2）**：
```
Qwen2.5-1.5B-Instruct:
- GRPO: Avg Acc = 30.18%, Correct Format = 76.33%
- GRPO w/o std: Avg Acc = 29.26%, Correct Format = 0% ❌
- GDPO: Avg Acc = 32.81%, Correct Format = 80.66% ✅
```

**关键发现**：
- GRPO w/o std在format reward上完全失败（0%正确格式率）
- 说明单纯增加advantage diversity可能导致训练不稳定
- GDPO通过解耦归一化既保持了优势多样性又确保稳定性

---

### 4.2 Experiment 2: Mathematical Reasoning

**任务设置**：
- 训练模型解决数学竞赛问题
- 两个隐含竞争的rewards：
  - Correctness reward ℛ_correct ∈ {0,1}: 最终答案是否正确
  - Length reward ℛ_length ∈ {0,1}: 响应长度≤4000 tokens

**评估基准**: AIME-24, AMC 2022/23, MATH, Minerva, Olympiad Bench

#### 训练行为分析（Figure 5）：

**DeepSeek-R1-1.5B训练曲线**：
```
阶段1 (0-100 steps):
- 两种方法都快速最大化length reward（容易优化）
- 长度reward达到峰值
- Correctness reward短暂下降（竞争效应）

阶段2 (100-400 steps):
- GDPO比GRPO更快恢复correctness reward
- GDPO在comparable steps达到更高correctness

阶段3 (400+ steps):
- GRPO: correctness开始下降 ❌
- GDPO: correctness继续提升 ✅
- GRPO: maximum response length急剧上升（违反长度约束）
- GDPO: maximum response length持续下降 ✅
```

#### 评估结果（Table 3）：

**DeepSeek-R1-1.5B**:
```
基准       | GRPO Acc | GDPO Acc | GRPO Exceed | GDPO Exceed
---        | ---      | ---      | ---         | ---
MATH       | 83.6%    | 86.2% (+2.6%) | 1.5% | 0.8% (-0.7%)
AIME       | 23.1%    | 29.4% (+6.3%) | 6.5% | 0.2% (-6.3%) ⭐
AMC        | 64.5%    | 69.0% (+4.5%) | 3.2% | 0.3% (-2.9%)
Minerva    | 43.5%    | 44.0% (+0.5%) | 1.7% | 0.3% (-1.4%)
Olympiad   | 44.3%    | 46.6% (+2.3%) | 2.6% | 0.4% (-2.2%)
```

**DeepSeek-R1-7B** (更大规模):
```
基准       | GRPO Acc | GDPO Acc | GRPO Exceed | GDPO Exceed
---        | ---      | ---      | ---         | ---
MATH       | 93.6%    | 94.1% (+0.5%) | 0.5% | 0.1% (-0.4%)
AIME       | 50.2%    | 53.1% (+2.9%) | 2.1% | 0.2% (-1.9%)
AMC        | 82.9%    | 84.0% (+1.1%) | 0.6% | 0.3% (-0.3%)
Minerva    | 53.2%    | 53.8% (+0.6%) | 0.2% | 0.1% (-0.1%)
Olympiad   | 60.2%    | 59.7% (-0.5%) | 1.1% | 0.4% (-0.7%)
```

**Qwen3-4B-Instruct**:
```
基准       | GRPO Acc | GDPO Acc | GRPO Exceed | GDPO Exceed
---        | ---      | ---      | ---         | ---
MATH       | 93.9%    | 94.6% (+0.7%) | 0.8% | 0.1% (-0.7%)
AIME       | 54.6%    | 56.9% (+2.3%) | 2.5% | 0.1% (-2.4%)
AMC        | 84.5%    | 85.2% (+0.7%) | 0.7% | 0.1% (-0.6%)
Minerva    | 50.7%    | 52.4% (+1.7%) | 0.3% | 0.1% (-0.2%)
Olympiad   | 66.8%    | 67.5% (+0.7%) | 1.6% | 1.0% (-0.6%)
```

**关键观察**：
1. **GDPO在AIME等挑战性任务上提升显著**（DeepSeek-R1-1.5B: **+6.3%**）
2. **Length约束控制能力**: GDPO将长度超出率从~2-6%降低到<1%
3. **训练稳定性**: Figure 9-10显示GDPO持续改善correctness并更好遵守长度约束

#### 4.2.1 Reward Priority分析

**问题**: 当不同objectives难度差异显著时，简单的权重调整可能无法实现预期的prioritization。

**实验设计**（Figure 6）:
- 固定 correctness reward weight w_correct = 1
- 变化 length reward weight w_length ∈ {1.0, 0.75, 0.5, 0.25}
- 测试在有/无conditioned length reward的情况下的表现

**条件化Length Reward** (Section 3.2, Equation 8):
```
ℛ̃_length = {1, if response length ≤ l AND ℛ_correct = 1
              {0, otherwise}
```
只有当回答正确时才能获得长度reward！

**结果对比**（Table 4 + Figure 6）:

**无conditioning**（仅调整权重）:
```
w_length | MATH Acc | MATH Exceed | AIME Acc | AIME Exceed
---      | ---      | ---         | ---      | ---
1.0      | 94.1%    | 0.5%        | 53.1%    | 0.2%
0.75     | 94.1%    | 0.5%        | 54.4%    | 0.5%
0.5      | 94.2%    | 0.6%        | 53.8%    | 0.8%
0.25     | 94.2%    | 0.5%        | 54.7%    | 0.9%  ⚠️
```
降低w_length到0.25才能看出长度约束放松，但accuracy变化不明显

**有conditioning**:
```
w_length | MATH Acc | MATH Exceed | AIME Acc | AIME Exceed
---      | ---      | ---         | ---      | ---
1.0      | 93.2%    | 1.4%        | 53.1%    | 2.7%
0.75     | 94.6%    | 2.8%        | 56.0%    | 3.8%
0.5      | 94.5%    | 3.1%        | 57.7%    | 4.8%
0.25     | 94.9%    | 5.1%        | 57.9%    | 6.2%  ✅
```

**关键发现**：
- Conditioned reward使模型无法过度优化容易的length reward
- 降低w_length时，模型行为更可预测（长度超出率稳定增加）
- **GDPO + conditioned reward在AIME上达到57.7% accuracy**（vs GRPO的53.3%）

**训练曲线**（Figure 7）:
- Conditioned reward防止模型在训练early阶段过度优化长度
- Correctness reward下降幅度很小，然后逐渐恢复

---

### 4.3 Experiment 3: Coding Reasoning

**任务设置**：
- 训练模型生成正确、简洁、无bug的代码
- 三个rewards：
  - Passrate reward ℛ_pass ∈ [0,1]: 测试用例通过比例
  - Conditioned Length reward ℛ̃_length ∈ {0,1}: 长度约束AND ℛ_pass=1
  - Bug reward ℛ_bug ∈ {0,1}: 代码是否无运行时/编译错误

**评估基准**: Apps, CodeContests, Codeforces, Taco

#### 结果（Table 5）：

**Two-Objective设置** (ℛ_pass + ℛ̃_length):

**Apps**:
```
指标   | GRPO_2-obj | GDPO_2-obj | 差异
---    | ---        | ---        | ---
Pass   | 28.1%      | 67.2% ✅   | +39.1%
Exceed | 5.2%       | 5.0%       | -0.2%
Bug    | 32.9%      | 25.0% ✅   | -7.9%
```

**Codeforces**:
```
指标   | GRPO_2-obj | GDPO_2-obj | 差异
---    | ---        | ---        | ---
Pass   | 68.1%      | 71.2% ✅   | +3.1%
Exceed | 18.1%      | 18.4%      | +0.3%
Bug    | 7.0%       | 5.6% ✅    | -1.4%
```

**Three-Objective设置** (ℛ_pass + ℛ̃_length + ℛ_bug):

**Codeforces**:
```
指标   | GRPO_3-obj | GDPO_3-obj | 差异
---    | ---        | ---        | ---
Pass   | 69.5%      | 69.4%      | -0.1%
Exceed | 16.9%      | 13.6% ✅   | -3.3%
Bug    | 2.5%       | 1.8% ✅    | -0.7%
```

**Codecontests**:
```
指标   | GRPO_3-obj | GDPO_3-obj | 差异
---    | ---        | ---        | ---
Pass   | 68.3%      | 68.1%      | -0.2%
Exceed | 19.3%      | 15.8% ✅   | -3.5%
Bug    | 3.9%       | 2.5% ✅    | -1.4%
```

**关键发现**：
1. **GDPO在2-objective设置中显著提升pass rate**（Apps上+39.1%巨大提升）
2. **在3-objective设置中，GDPO实现更好的多目标平衡**：
   - 保持相似的pass率
   - 大幅降低length exceed率（-3.3%到-3.5%）
   - 降低bug率（-0.7%到-1.4%）
3. **随reward数量增加，GDPO保持有效性**

---

## 📊 技术细节总结

### 5.1 归一化策略对比

| 方法 | 归一化步骤 | Advantage计算 | 信息保持 |
|------|----------|--------------|---------|
| **GRPO** | Group-wise归一化aggregated reward | 单步，直接归一化和 | ❌ 丢失reward维度信息 |
| **GRPO w/o std** | 直接归一化和（无标准差） | 单步，但幅度不稳定 | ⚠️ 部分恢复但不稳定 |
| **GDPO** | 1) Per-reward归一化<br>2) 求和<br>3) Batch-wise归一化 | 三步，逐层处理 | ✅ 保持每个reward相对差异 |

### 5.2 数学推导：为什么会Collapse？

设两个rollouts，两个binary rewards:

**GRPO计算**:
```
组合(0,2): r_sum = [0, 2], mean=1, std≈1.414
         → A_sum = [(0-1)/1.414, (2-1)/1.414] = [-0.7071, 0.7071]

组合(0,1): r_sum = [0, 1], mean=0.5, std≈0.7071
         → A_sum = [(0-0.5)/0.7071, (1-0.5)/0.7071] = [-0.7071, 0.7071]

结果：不同组合 → 相同advantage！
```

**GDPO计算**:
```
组合(0,2): r1=[0,1], r2=[0,1]
         → A1 = [-0.7071, 0.7071], A2 = [-0.7071, 0.7071]
         → A_sum = [-1.4142, 1.4142] (归一化前)

组合(0,1): r1=[0,0], r2=[0,1] (假设)
         → A1 = [0, 0], A2 = [-0.7071, 0.7071]
         → A_sum = [-0.7071, 0.7071] (归一化前)

结果：不同组合 → 不同advantage！
```

### 5.3 Batch-wise归一化的作用

**目的**: 确保advantage的数值范围不随reward数量增加而增大

**公式**:
```
A^_sum(i,j) = [A_sum(i,j) - mean_batch] / [std_batch + ε]
```

**实验证明**（Appendix A, Figure 8）:
- 不进行batch-wise归一化时，GDPO偶尔收敛失败
- 该步骤提供了额外的训练稳定性

### 5.4 Reward Function设计

#### Tool Calling Rewards（Appendix C）:

**Format reward**:
```
ℛ_format = {1, if all required fields appear and are in correct order
           {0, otherwise
```

**Correctness reward**（三部分）:
```
1) Tool name matching:
   r_name = |N_G ∩ N_P| / |N_G ∪ N_P|

2) Parameter name matching:
   r_param = Σ[G_j∈G] (|keys(G_j) ∩ keys(P_j)| / |keys(G_j) ∪ keys(P_j)|)

3) Parameter content matching:
   r_value = Σ[G_j∈G] Σ[k∈keys(G_j)] 𝟙[P_G[k] = P_P[k]]

Total match score: R_match = r_name + r_param + r_value

Final correctness:
   ℛ_correct = 6 × (R_max / S_max) - 3, 其中 S_max = 1 + |G| + Σ|keys(G_j)|
```

#### Coding Reward公式:

**Passrate reward**:
```
ℛ_pass = (number of passed test cases) / (total test cases)
```

**Conditioned Length reward**:
```
ℛ̃_length = {1, if response length ≤ l AND ℛ_pass = 1
              {0, otherwise
```

---

## 🌟 论文贡献总结

### Theoretical Contributions:

1. **GRPO Reward Collapse分析**
   - 首次系统分析了GRPO在多reward设置下的根本缺陷
   - 证明了不同reward组合会坍缩成相同advantage
   - 这导致学习信号分辨率降低、性能下降、训练不稳定

2. **GDPO方法提出**
   - 解耦归一化：先对每个reward单独归一化
   - 保持reward维度差异，提高advantage表达能力
   - Batch-wise归一化确保数值稳定性

3. **Reward Priority系统**
   - 分析了权重调整的局限性（当reward难度差异大时）
   - 提出了条件化reward函数设计（Section 3.2）
   - 提供了优先级控制的实用指导

### Empirical Contributions:

1. **Three comprehensive tasks**:
   - Tool calling（准确率+格式）
   - Math reasoning（准确率+长度约束）
   - Coding reasoning（通过率+长度+bug率）

2. **Consistent improvements**:
   - 所有任务、所有模型尺寸上GDPO优于GRPO
   - 关键指标：AIME +6.3%（DeepSeek-R1-1.5B），BFCL +4.73%

3. **Stability验证**:
   - Figure 4-5, 9-10展示更稳定的训练曲线
   - Batch-wise归一化的关键作用

---

## 🔗 相关工作对比

### GRPO Variants（Section 5.1）:

| 方法 | 核心创新 | 与GDPO关系 |
|------|---------|-----------|
| **GSPO** | Sequence-level clipping | 侧重不同，GDPO解决多reward问题 |
| **DAPO** | Clip-Higher, Dynamic Sampling | 互补技术，可结合 |
| **GFPO** | Length explosion control | 同关注效率，但方法不同 |
| **DLER** | Batch-wise normalization, higher clipping | 更接近，是GDPO的启发来源之一 |

### Multi-Reward RL（Section 5.2）:

| 方向 | 代表工作 | GDPO定位 |
|------|---------|---------|
| **Human preferences** | Safe RLHF, RLPHF, ALARM | GDPO针对reward collapse而非preference建模 |
| **Reasoning efficiency** | O1-Pruner, L1, DLER | 直接相关，GDPO更好地平衡accuracy和效率 |
| **DeepSeek V3.2** | Multi-reward for reasoning | 工业应用验证，但未解决GRPO的deficiency |

---

## 💡 实践指导

### 当何时使用GDPO:

✅ **推荐场景**：
- 多个rewards（≥2）需要同时优化
- Rewards之间可能存在竞争关系（如accuracy vs length）
- 不同rewards难度或scale差异显著
- 需要精细的preference control

✅ **不推荐场景**：
- 单一reward优化（GRPO足够）
- Rewards的scale和难度高度相似
- 对计算效率要求极端严格（GD多一次归一化）

### 优先级控制最佳实践:

**Step 1: 识别难度差异**
```
- 简单reward：如格式检查、长度约束
- 困难reward：如准确性、安全性
```

**Step 2: 选择控制策略**
```
策略A: 简单权重调整（适用难度相近）
策略B: 条件化reward（适用难度差异大）
策略C: 权重调整 + 条件化（最强大）
```

**Step 3: 实现条件化reward**
```python
def conditioned_reward(primary_reward, secondary_reward, threshold):
    """
    Model only gets secondary_reward when primary_reward ≥ threshold
    """
    if primary_reward >= threshold:
        return secondary_reward
    else:
        return 0.0
```

**示例**（Math reasoning）:
```python
# Without conditioning: model may first optimize length
reward_correctness = is_correct(answer)
reward_length = 1.0 if len(response) <= 4000 else 0.0

# With conditioning: model must first ensure correctness
reward_correctness = is_correct(answer)
reward_length = (1.0 if len(response) <= 4000 else 0.0) if reward_correctness else 0.0
```

---

## 🎯 核心代码对比（概念性）

### GRPO实现（简化）:
```python
def GRPO_advantage(rewards_group):
    """
    rewards_group: [G, n rewards per rollout]
    Returns: [G] advantages
    """
    # Step 1: Sum all rewards
    r_sum = np.sum(rewards_group, axis=1)  # [G]
    
    # Step 2: Normalize
    mean = np.mean(r_sum)
    std = np.std(r_sum) + 1e-8
    advantages = (r_sum - mean) / std  # [G] - COLLAPSE!
    
    return advantages
```

### GDPO实现（简化）:
```python
def GDPO_advantage(rewards_group, batch_advantages):
    """
    rewards_group: [G, n rewards per rollout]
    batch_advantages: List of all advantages in batch
    Returns: [G] normalized advantages
    """
    # Step 1: Normalize each reward separately
    normalized_rewards = np.zeros_like(rewards_group)
    for i in range(rewards_group.shape[1]):  # For each reward
        r_i = rewards_group[:, i]
        mean_i = np.mean(r_i)
        std_i = np.std(r_i) + 1e-8
        normalized_rewards[:, i] = (r_i - mean_i) / std_i
    
    # Step 2: Sum across rewards
    A_sum = np.sum(normalized_rewards, axis=1)  # [G]
    
    # Step 3: Batch-wise normalization
    mean_batch = np.mean(batch_advantages)
    std_batch = np.std(batch_advantages) + 1e-8
    final_advantages = (A_sum - mean_batch) / std_batch  # [G]
    
    return final_advantages
```

---

## 📈 实验数据深度解读

### 为什么GDPO在AIME上提升最大？

**AIME特点**：
- 美国高中数学最竞赛，极难
- 平均正确率仅~15%（human）, ~25-50%（advanced models）

**推测原因**：
```
AIME任务的特点：
1. 需要更复杂的多步推理
2. 长度约束是hard constraint（必须简短但准确）
3. Accuracy和Length存在强竞争关系

GDPO的优势：
1. 更精确的advantage估计：区分"正确但较长"vs"正确且简短"
2. 更好的多reward平衡：避免牺牲accuracy换short length
3. 更稳定的训练：correctness持续提升而非后期下降
```

**数据支持**（Table 3）:
- DeepSeek-R1-1.5B: GRPO 23.1% → **GDPO 29.4%** (+6.3%)
- DeepSeek-R1-7B: GRPO 50.2% → **GDPO 53.1%** (+2.9%)
- 同时length exceed率从6.5%/2.1%降到0.8%/0.2%

### 为什么在Apps上提升特别大？

**Apps特点**：
- 编程竞赛问题，包含多个test cases
- 需要生成完整、可运行的代码

**推测原因**：
```
Apps任务需要：
1. 代码要pass所有test cases（ℛ_pass）
2. 代码要简洁（ℛ_length）
3. 代码要无bug（ℛ_bug）

GRPO的问题：
- 三个reward求和后，可能丢失"高通过率但有bug"vs"低通过率无bug"的区分
- Averaging across dimensions loses critical information

GDPO的优势：
- 每个reward独立归一化，保持各自的信号
- A(0.8 pass, 0 length, 1 bug) ≠ A(0.3 pass, 1 length, 0 bug)
- 允许模型学习复杂的多目标平衡
```

**数据支持**（Table 5）:
- Apps Pass: GRPO 28.1% → **GDPO 67.2%** (+39.1%!) 
- 同时Bug: 32.9% → 25.0% (-7.9%)
- Exceed: 5.2% → 5.0% (稳定)

---

## 🔬 技术细节深度解析

### Advantage表达的数学本质

**Group-relative advantage的核心思想**：
```
传统advantage: A(s,a) = Q(s,a) - V(s)
Group-relative: A_i = r_i - mean(r_group) / std(r_group)

优势：
1. 无需value model（vs PPO需要critic）
2. 减少估计方差（使用同一group的相对比较）
3. 适用于LLM的离散output space
```

**多reward设置下的issue**：
```
Single reward: r ∈ [0, 1] → A ≈ [-1, 1]
Multi-reward: r_sum ∈ [0, n] → 但归一化后仍≈[-1, 1]

问题：r_sum的信息维度远大于A的表达维度！
- n=2: r_sum可能为0,1,2（3种值），A可能为-1,0,1（3种）✓
- 但不同(0,2)和(0,1)都产生相同归一化advantage ✗

信息论视角：Group-wise normalization是一个many-to-one mapping
```

**GDPO的解决方案**：
```
Per-reward归一化：
- r_1 ∈ [0, 1] → A_1 ∈ [-1, 1]
- r_2 ∈ [0, 1] → A_2 ∈ [-1, 1]
- A_sum = A_1 + A_2 ∈ [-2, 2]

表达能力提升：
- Single GRPO: 3 distinct r_sum → 2 distinct A values
- GDPO: 2×2 = 4 distinct (r1, r2) → 4 distinct A values (理论)
```

### 为什么Batch-wise归一化是必要的？

**问题：随着reward数量n增加，A_sum的range会增大**
```
假设每个reward被归一化到[-1, 1]，且独立：
n=2: A_sum ∈ [-2, 2]
n=3: A_sum ∈ [-3, 3]
...
n=10: A_sum ∈ [-10, 10]

这将导致：
1. 训练梯度幅度增大（更新不stable）
2. 不同n设置的结果不可比
3. 可能触发exploding gradient
```

**Batch-wise归一化的作用**：
```
A^_sum = (A_sum - mean_batch) / std_batch

无论n有多大，最终A^_sum的range始终≈Batch的动态范围
→ Stable训练跨不同n设置
```

**实验验证**（Appendix A）：
- 无batch-wise normalization时，GDPO偶尔（~20% runs）收敛失败
- Figure 8: 有normalization的runs始终收敛

### Conditioned Reward的机制分析

**数学表达**（Equation 8）:
```
标准reward: r = f(x)
条件化reward: r_conditional = {r if g(x) ≥ t
                               {0 otherwise

其中g(x)是gating function，t是threshold
```

**为什么有效？**

**信息流视角**：
```
Without conditioning:
r_length → Policy update ← (r_length + r_correctness)
Model may: 1) Maximize r_length first (easy)
            2) Try to fix correctness later (hard)
           Result: Early convergence on bad local optimum

With conditioning:
r_correctness → r_length → Policy update
Model must: 1) Maximize r_correctness first
             2) Then optimize r_length
            Result: Avoid premature convergence on easy objective
```

**优化landscapes视角**：
```
2D reward空间 (correctness, length):

Without conditioning:
- Gradient points towards high length region first
- May get stuck at (low correctness, high length)

With conditioning:
- Length gradient is masked until correctness > threshold
- Force exploration of correctness region first
- Then navigate towards (high correctness, high length)
```

**实验证据**（Figure 6）:
```
w_length降低时（1.0→0.25）:

无conditioning:
- Length exceed: 0.2% → 0.9% (MATH), 0.2% → 0.9% (AIME)
- Accuracy几乎不变

有conditioning:
- Length exceed: 1.4% → 5.1% (MATH), 2.7% → 6.2% (AIME) ✅
- Accuracy: 93.2% → 94.9% (MATH), 53.1% → 57.9% (AIME) ✅

结论: Conditioning使权重调整更predictable和effective
```

---

## 🧬 理论分析

### Information Loss Analysis

**定义Reward Combination Space**:
```
设n个binary rewards, each ∈ {0,1}
Total combinations = 2^n

GRPO: 2^n → aggregate r_sum ∈ [0, n]
归一化后 → A_grpo ∈ R, 但distinct values << 2^n

Empirical result (Figure 3):
n=2, G=4: GRPO产生2 distinct A groups
n=2, G=8: GRPO产生~3 distinct A groups
```

**信息论度量**：
```
Entropy of reward space:
H(R) = -Σ p(r) log p(r)

For uniform binary rewards:
H(R) = n bits  (fully informative)

GRPO归一化后：
H(A) << n bits  (information loss)

GDPO:
H(A_GDPO) ≈ n × H(A_single) ≈ n bits (信息preserved)
```

### Convergence Analysis

**Policy Gradient update**:
```
∇θ J(θ) = E[s,t ∼ πθ][∇θ log πθ(s|a) · A(s,a)]

A的precision直接影响gradient quality:
- Low precision (collapse): Many states share same A → noisy gradient
- High precision (GDPO): Different states have distinct A → precise gradient
```

**Training Dynamics**（Figure 5）:
```
Phase 1 (0-100 steps): 
- Both methods maximize easy reward (length)
- GDPO maintains richer A distribution → more nuanced updates

Phase 2 (100-400 steps):
- GDPO recovers correctness faster due to better A estimation
- GRPO's collapsed A may misguide updates

Phase 3 (400+ steps):
- GRPO's accumulated errors lead to instability (correctness declines)
- GDPO's consistent A estimates maintain stable improvement
```

### Multi-objective Optimization Perspective

**问题表述**:
```
Maximize: f_1(θ), f_2(θ), ..., f_n(θ)
Subject to: θ ∈ Policy space

GRPO: Maximize f_sum = Σ w_i f_i
        - Implicitly assumes linear separability
        - May fail when objectives interdependent

GDPO: Individual normalization → weighted sum
       - Preserves relative importance of each f_i
       - Better handles interdependent objectives
```

**Pareto Front分析**：
```
假设correctness vs length的Pareto front:

GRPO可能的行为:
- May converge to suboptimal point on Pareto front
- Because it doesn't distinguish fine-grained tradeoffs

GDPO的优势:
- Better exploration of Pareto front
- Can find points that balance objectives more optimally
```

---

## 🔗 实现细节

### Pseudocode对比

**GRPO Algorithm**:
```python
def GRPO_update(model, batch):
    # Generate G rollouts per prompt
    rollouts = [generate(model, prompt) for _ in range(G)]
    
    # Compute rewards
    for rollout in rollouts:
        rollout.reward = sum([r_i for r_i in rollout.rewards])
    
    # Compute advantages (COLLAPSE!)
    rewards = [r.reward for r in rollouts]
    mean, std = np.mean(rewards), np.std(rewards)
    advantages = [(r - mean) / std for r in rewards]
    
    # Policy update
    for rollout, advantage in zip(rollouts, advantages):
        loss = -log_prob(rollout) * advantage
        loss.backward()
```

**GDPO Algorithm**:
```python
def GDPO_update(model, batch, batch_advantages_all):
    # Generate G rollouts per prompt
    rollouts = [generate(model, prompt) for _ in range(G)]
    
    # Compute and normalize EACH REWARD SEPARATELY
    for rollout in rollouts:
        # Per-reward normalization
        normalized_rewards = []
        for i, reward_i in enumerate(reward_components):
            r_i_values = [r.rewards[i] for r in rollouts]
            mean_i, std_i = np.mean(r_i_values), np.std(r_i_values)
            normalized_i = (reward_i - mean_i) / (std_i + eps)
            normalized_rewards.append(normalized_i)
        
        # Sum across rewards
        rollout.advantage_unnormalized = sum(normalized_rewards)
    
    # Batch-wise normalization
    all_advantages = [r.advantage_unnormalized for r in rollouts] + batch_advantages_all
    mean_batch, std_batch = np.mean(all_advantages), np.std(all_advantages)
    for rollout in rollouts:
        rollout.advantage = (rollout.advantage_unnormalized - mean_batch) / (std_batch + eps)
    
    # Policy update
    for rollout in rollouts:
        loss = -log_prob(rollout) * rollout.advantage
        loss.backward()
```

### Reward Function Implementation

**Tool calling format reward** (Equation 9):
```python
def format_reward(response, required_fields, required_order):
    """
    Checks if all fields appear in correct order
    """
    # Extract fields in order
    extracted_fields = extract_fields(response, required_fields)
    
    # Check if all required fields present
    if set(extracted_fields) != set(required_fields):
        return 0.0
    
    # Check order
    for i in range(len(required_order) - 1):
        field1 = required_order[i]
        field2 = required_order[i + 1]
        if response.find(field1) > response.find(field2):
            return 0.0
    
    return 1.0
```

**Tool calling correctness reward** (Equation C):
```python
def correctness_reward(predicted_calls, ground_truth_calls):
    """
    Computes similarity between predicted and ground truth tool calls
    """
    # Hungarian algorithm for optimal matching
    match_matrix = compute_match_matrix(predicted_calls, ground_truth_calls)
    optimal_matching = hungarian(match_matrix)
    
    # Compute match score
    total_score = 0.0
    for pred, gt in optimal_matching:
        # 1. Tool name matching
        name_score = 1.0 if pred.name == gt.name else 0.0
        
        # 2. Parameter name matching
        param_names_pred = set(pred.parameters.keys())
        param_names_gt = set(gt.parameters.keys())
        param_score = len(param_names_pred & param_names_gt) / len(param_names_pred | param_names_gt)
        
        # 3. Parameter content matching
        value_score = sum(1.0 for k in param_names_pred if pred.parameters[k] == gt.parameters[k])
        value_score /= len(param_names_pred) if param_names_pred else 1.0
        
        total_score += name_score + param_score + value_score
    
    # Normalize to [-3, 3] range
    max_possible_score = 1 + len(ground_truth_calls) + sum(len(gt.parameters) for gt in ground_truth_calls)
    normalized = 6 * (total_score / max_possible_score) - 3
    
    return normalized
```

---

## 📋 实验装置细节

### 完整Hyperparameters（Appendices D & E）

**Tool Calling Experiment** (Table D):
```table
参数 | Value | 说明
---|---|---
trainer.total_epochs | 15 | 训练总轮数
data.train_batch_size | 512 | 每batch样本数
actor_rollout_ref.actor.ppo_mini_batch_size | 128 | PPO mini-batch大小
data.max_prompt_length | 2048 | 最大prompt长度
actor_rollout_ref.actor.optim.lr | 1.00E-06 | 学习率
actor_rollout_ref.rollout.n | 4 | 每个prompt的rollout数（G=4）
algorithm.kl_ctrl.kl_coef | 0.001 | KL散度惩罚系数
```

**Math/Coding Reasoning** (Table E):
```table
参数 | Value | 说明
---|---|---
data.train_batch_size | 512 | -0.0
actor_rollout_ref.actor.ppo_mini_batch_size | 64 | 较大的rollout数，较小的mini-batch
actor_rollout_ref.actor.ppo_epochs | 1 | 每batch的PPO epochs
data.max_prompt_length | 1024 | Math/coding需要更长的output
actor_rollout_ref.actor.optim.lr | 1.00E-06 | 与tool calling相同
actor_rollout_ref.rollout.temperature | 1 | Sampling temperature
actor_rollout_ref.rollout.n | 16 | 更多rollouts（G=16）
actor_rollout_ref.actor.clip_ratio_low | 0.2 | Lower bound for clipping
actor_rollout_ref.actor.clip_ratio_high | 0.28 | Upper bound for clipping（DAPO技术）
algorithm.filter_groups.enable | TRUE | DLER的group filtering
algorithm.filter_groups.metric | seq_reward | Filter metric
actor_rollout_ref.actor.kl_loss_coef | 0.0005 | KL惩罚系数
actor_rollout_ref.actor.kl_loss_type | mse | KL loss type
```

**关键技术选择解释**：
```
1. Rollout数量从4（tool calling）增加到16（math/coding）:
   - Math/coding更难，需要更多samples估计reliable advantage
   - 与DLER和GFPO的最佳实践一致

2. Mini-batch从128降到64同时rollout增加到16:
   - 总compute保持相似但increase exploration diversity
   - Advantage estimation更reliable

3. Higher clipping ratio（0.2-0.28 vs PPO默认0.1-0.2）:
   - DAPO技巧：允许更大的steps for challenging tasks
   - Prevent excessive conservatism in multi-objective setting

4. KL惩罚降到0.0005（vs tool calling的0.001）:
   - Math/coding需要更多exploration
   - Reward signals more complex,需要允许policy shift
```

---

## 🌐 扩展应用

### Beyond the Paper（扩展思考）

**1. 更多Reward维度**:
```
论文已验证2-3个rewards，可扩展到更多：
- Safety: Harmlessness评分
- Style: Formality, tone, readability
- Efficiency: Token usage, latency, cost
- Domain-specific: Factuality, citations, logical consistency

GDPO的优势随n增加而更加显著（Figure 3）
```

**2. Dynamic Reward Weighting**:
```
当前方法使用固定权重，可扩展为：
- Adaptive weights based on training stage
- Per-prompt importance weighting
- User preference modeling for personalization

Example:
if training_stage == early:
    w_accuracy = 1.0, w_length = 1.0
elif training_stage ==中期:
    w_accuracy = 1.5, w_length = 0.8
```

**3. Hierarchical Reward Structure**:
```
Inspired by ALARM (Section 5.2):

Level 1: High-level objectives (accuracy, safety)
Level 2: Mid-level (format, style)
Level 3: Low-level (length, token usage)

GDPO可应用于multi-level归一化：
- Within-level decoupled normalization
- Cross-level weighted summation
```

**4. Integration with Preference Learning**:
```
Current: Rule-based rewards
Future: Learned reward from human preferences

可结合：
- RLHF with multiple reward models
- Preference datasets aligned to different objectives
- Multi-dimensional preference modeling

GDPO的优势：
- Each reward model can have different uncertainties/scales
- Decoupled normalization handles this naturally
```

**5. Multi-Agent Settings**:
```
Current: Single LLM
Future: Multi-agent systems with diverse objectives

可应用于：
- Debate agents with correctness + civility rewards
- Creative collaboration with novelty + coherence rewards
- Hierarchical agents with different priorities

GDPO适合：
- Different agents may optimize different objectives
- Shared normalization framework ensures coherent system behavior
```

---

## 📚 相关资源链接

**论文链接**:
- ArXiv: https://arxiv.org/abs/2601.05242
- HTML: https://arxiv.org/html/2601.05242v1

**实现链接**（Abstract中提到的）:
- HF-TRL: HuggingFace TRL implementation
- verl: https://github.com/volcengine/verl (Hybridflow framework)
- Nemo-RL: NVIDIA NeMo RL

**相关工作**:
- GRPO: DeepSeek-R1 paper (arXiv:2501.12948)
- DAPO: arXiv:2503.14476
- DLER: arXiv:2510.15110
- ToolRL: arXiv:2504.13958

**数据集链接**:
- BFCL-v3: Berkeley Function Call Leaderboard
- DeepScaleR-Preview: https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview
- AIME: https://maa.org/math-competitions/aime-problems
- Eurus-2-RL: arXiv:2502.01456（Coding data）

**代码框架**:
- verl (Hybridflow): https://arxiv.org/abs/2409.19256

---

## 🎓 学习要点总结

### 核心知识点:

1. **GRPO的Reward Collapse问题**（Section 2）
   - 多reward求和后归一化导致信息丢失
   - 不同reward组合产生相同advantage
   - 降低学习信号分辨率

2. **GDPO的解决方案**（Section 3.1）
   - 解耦归一化：先对每个reward单独归一化
   - 保持reward维度差异
   - Batch-wise归一化确保数值稳定性

3. **Priority Control策略**（Section 3.2, 4.2.1）
   - 简单权重调整：适用于相似难度rewards
   - 条件化reward：强制优先级（难度差异大时必需）
   - 组合使用：最佳实践

4. **实验验证**（Section 4）
   - Tool calling: +4.73% accuracy, +4.33% format
   - Math reasoning: +6.3% on AIME
   - Coding reasoning: +39.1% pass on Apps

### 实践指导:

**何时使用GDPO**:
```
✅ Multiple rewards (≥2)
✅ Rewards may compete or have different scales
✅ Need precise preference alignment
✅ Training stability concerns
```

**如何设计rewards**:
```
1. 识别所有objectives
2. 定性评估每个objective的难度
3. 设计reward函数（考虑conditioning）
4. 设置初始权重（从equal开始）
5. 训练观察，调整权重/conditions
```

**Debug tips**:
```
If model converges on easy objective only:
- Consider conditioned rewards
- Increase weight on hard objective
- Check if rewards are well-defined

If training unstable:
- Ensure batch-wise normalization
- Check reward scales
- Verify clipping parameters
```

---

## 🔮 未来方向

**论文提到的局限性**:
- 主要关注离散reward，可能需要扩展到continuous rewards
- 需要在更多domain验证
- 与其他RL方法（如DAPO, DLER）的系统整合

**潜在研究方向**:
```
1. Theoretical analysis:
   - Formal information-theoretic analysis
   - Convergence proofs
   - Sample complexity bounds

2. Technical improvements:
   - Adaptive normalization strategies
   - Hierarchical multi-reward optimization
   - Dynamic importance weighting

3. Applications:
   - Multi-agent systems
   - Personalized LLMs
   - RLHF with multiple preference models

4. Evaluation:
   - Comprehensive multi-objective benchmarks
   - Human evaluation frameworks
   - Robustness analysis
```

---

这篇论文的核心价值在于**systematically diagnosing**了一个重要的实际问题（GRPO reward collapse），并提出了**elegant yet powerful solution**（GDPO），通过**comprehensive experiments**证明了其有效性。它为multi-reward RL alignment领域提供了一个更stable、accurate和controllable的基础方法。

特别值得关注的是，论文不仅提出了方法，还提供了**practical guidance for reward design和priority control**，这使得研究具有direct real-world applicability。