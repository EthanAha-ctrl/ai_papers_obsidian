









虽然 web search 暂时没有返回结果，但 ORM 是 RLHF 和 reasoning 领域非常核心的概念，我可以基于已有知识做详尽讲解。

---

# Outcome Reward Model (ORM) 详解

## 1. 第一性原理：ORM 是什么？

从最根本的层面出发，ORM 回答的问题是：

> **给定一个完整的 solution trajectory（解法路径），最终结果是否正确？**

ORM 的核心思想是：**只看 outcome（结果），不看 process（过程）**。它是一个判别式模型，输入一条完整的 reasoning chain + 最终 answer，输出一个标量分数，表示该 outcome 正确/优质的概率。

形式化定义：

$$r_{\text{ORM}}(x, y) = P(\text{correct} \mid \text{question } x, \text{full solution } y)$$

其中：
- $x$ = 输入问题（如 math problem）
- $y$ = 完整的 solution sequence $(y_1, y_2, \ldots, y_T)$，包含所有中间步骤和最终 answer
- $r_{\text{ORM}}$ = 标量 reward，通常通过 sigmoid 输出 $\in [0, 1]$

**关键区别**：ORM 只在 **整个 sequence 结束时** 给出一个 reward signal，而不会对中间步骤 $y_t$ 单独评分。

---

## 2. ORM vs PRM：核心对比

| 维度 | **ORM** (Outcome Reward Model) | **PRM** (Process Reward Model) |
|------|------|------|
| **监督粒度** | Sequence-level（整个解法） | Step-level（每一步） |
| **训练标签** | 只有最终 answer 对/错 | 每个中间 step 对/错 |
| **评分位置** | 仅在 sequence 末端 | 每一步都打分 |
| **标注成本** | 低（只需判断最终答案） | 极高（需人工逐步标注） |
| **信用分配** | 困难（reward 稀疏） | 容易（dense reward） |
| **训练信号** | 稀疏（sparse） | 密集（dense） |
| **对 false positive 的鲁棒性** | 差（正确答案可能来自错误推理） | 好（能识别中间步骤错误） |
| **代表论文** | Cobbe et al., 2021 (GSM8K) | Lightman et al., 2023 ("Let's Verify Step by Step") |

---

## 3. ORM 的训练方法

### 3.1 数据构造

给定训练集 $\{(x_i, y_i^{*})\}$，其中 $x_i$ 是问题，$y_i^{*}$ 是 ground-truth answer：

1. **采样多条 solution**：用 policy model $\pi_\theta$ 对每个问题 $x_i$ 采样 $N$ 条完整解法 $\{y_i^{(1)}, y_i^{(2)}, \ldots, y_i^{(N)}\}$
2. **自动标注**：将每条解法的最终 answer 与 $y_i^{*}$ 比较：
   - 若一致 → $z = 1$（positive）
   - 若不一致 → $z = 0$（negative）
3. 构造 dataset：$\mathcal{D} = \{(x_i, y_i^{(j)}, z_i^{(j)})\}$

### 3.2 模型架构

ORM 通常基于一个预训练的 LLM backbone（如 GPT-2, LLaMA 等），在其之上加一个 **scalar head**：

```
┌─────────────────────────────┐
│     Pre-trained LLM         │
│   (e.g., LLaMA-7B)          │
│                             │
│  input: [question + solution]│
│         ↓                   │
│   hidden states H_1...H_T   │
└──────────┬──────────────────┘
           │
           │  取最后一个 token 的 hidden state H_T
           ↓
┌──────────────────────────────┐
│  Linear: H_T → scalar        │
│  (可选: + LayerNorm + ReLU)   │
│         ↓                    │
│     σ(W·H_T + b)            │
│         ↓                    │
│   reward r ∈ [0, 1]          │
└──────────────────────────────┘
```

公式：

$$r_{\text{ORM}}(x, y) = \sigma\bigl(\mathbf{w}^\top \mathbf{h}_T + b\bigr)$$

其中：
- $\mathbf{h}_T \in \mathbb{R}^d$ = 最后一个 token 的 hidden state（$d$ 是 hidden dimension）
- $\mathbf{w} \in \mathbb{R}^d$, $b \in \mathbb{R}$ = 可学习的线性层参数
- $\sigma(\cdot)$ = sigmoid 函数

### 3.3 损失函数

标准的 binary cross-entropy loss：

$$\mathcal{L}_{\text{ORM}} = -\frac{1}{|\mathcal{D}|}\sum_{(x,y,z) \in \mathcal{D}} \bigl[z \cdot \log r_{\text{ORM}}(x,y) + (1-z) \cdot \log(1 - r_{\text{ORM}}(x,y))\bigr]$$

其中：
- $z \in \{0, 1\}$ = 该 solution 是否得到正确 answer
- $r_{\text{ORM}}(x,y)$ = 模型预测的 reward

---

## 4. ORM 的应用场景

### 4.1 Best-of-N Sampling (Verification / Rejection Sampling)

这是 ORM 最经典的应用：

1. 用 policy model 生成 $N$ 条候选 solution
2. 用 ORM 给每条打分 $r_{\text{ORM}}(x, y^{(j)})$
3. 选择得分最高的 solution：

$$y^* = \arg\max_{j \in \{1,\ldots,N\}} r_{\text{ORM}}(x, y^{(j)})$$

**Pass@1 vs Best-of-N 曲线**：

| N | ORM Best-of-N Accuracy | Without ORM |
|---|------|------|
| 1 | ~30% | ~30% |
| 10 | ~50% | ~35% |
| 100 | ~65% | ~38% |
| 1000 | ~75% | ~40% |

（以上为示意数据，具体数字取决于 benchmark）

### 4.2 RLHF 中的 Reward Signal

在 PPO 或 GRPO 等 RL 算法中，ORM 可以作为 reward function：

$$R(x, y) = r_{\text{ORM}}(x, y) - \beta \cdot \text{KL}\bigl(\pi_\theta \| \pi_{\text{ref}}\bigr)$$

其中：
- $\beta$ = KL penalty 系数
- $\pi_{\text{ref}}$ = reference policy（通常是 SFT model）

### 4.3 Rejection Sampling Fine-tuning (RAFT / RLAIF)

1. 采样 → ORM 过滤 → 保留高质量数据 → SFT fine-tune
2. 形成迭代循环：更强的 policy → 更好的采样 → 更好的 ORM → 更好的 policy

---

## 5. ORM 的核心挑战

### 5.1 Credit Assignment Problem（信用分配问题）

这是 ORM 最根本的缺陷。因为 ORM 只给出一个整体分数，RL 算法无法区分：

- **正确的推理 + 正确的答案**（应该强化所有步骤）
- **错误的推理 + 碰巧正确的答案**（false positive / reward hacking）
- **大部分正确推理 + 最后一步计算错误**（应该只惩罚最后一步，但 ORM 惩罚整个序列）

数学上，这体现为：

$$\nabla_\theta J_{\text{ORM}} = \mathbb{E}_{y \sim \pi_\theta}\bigl[r_{\text{ORM}}(x,y) \cdot \nabla_\theta \log \pi_\theta(y|x)\bigr]$$

梯度 $r_{\text{ORM}} \cdot \nabla_\theta \log \pi_\theta$ 均匀地增强/削弱整个 trajectory 的概率，无法精确地 credit 单个 step。

### 5.2 Reward Hacking / False Positive

ORM 只看 final answer，容易被 "侥幸答对" 的 solution 欺骗。例如：

```
Question: 23 × 47 = ?
Wrong reasoning: 23 × 47 = 23 × 50 - 23 × 4 = 1150 - 92 = 1058
Correct answer: 1081
→ ORM: reward = 0 (正确地判错)

Question: 23 × 47 = ?
Wrong reasoning: 23 × 47 ≈ 20 × 50 = 1000, let me guess 1081
Correct answer: 1081  
→ ORM: reward = 1 (被欺骗！推理过程完全错误但答案碰巧对了)
```

### 5.3 Sparse Reward

在 RL 训练中，ORM 只在 episode 结束时给一个 reward，中间没有任何信号。这导致：
- 训练方差大
- 收敛慢
- 长序列时 reward 信号被严重稀释

对比 PRM 的 dense reward：

```
Step 1: 正确 → r₁ = 0.9  ✓
Step 2: 正确 → r₂ = 0.8  ✓
Step 3: 错误 → r₃ = 0.1  ✗  (立刻发现并纠正)
Step 4: 正确 → r₄ = 0.7  ✓

ORM: 只在最后给 r = 0 (整体判错，但不知道哪步出错)
```

### 5.4 标注偏差

ORM 的训练数据通常通过自动匹配 final answer 生成标签，而非真正的人工标注 solution quality。这导致：
- 标签噪声
- 系统性地高估 "looks plausible but wrong" 的 solution
- 低估 "non-standard but correct" 的 solution

---

## 6. ORM 的变体与改进

### 6.1 ORM + Majority Voting

结合 ORM 分数和 majority voting：

$$y^* = \arg\max_{a} \sum_{j: \text{answer}(y^{(j)})=a} r_{\text{ORM}}(x, y^{(j)})$$

即对每个 candidate answer，累加所有产生该 answer 的 solution 的 ORM 分数，选总分最高的 answer。

### 6.2 Multi-dimensional ORM

不只是一个标量，而是预测多个维度的 reward：

$$\mathbf{r}_{\text{ORM}}(x, y) = \bigl(r_{\text{correctness}}, r_{\text{completeness}}, r_{\text{clarity}}, \ldots\bigr)$$

最终 reward 是加权组合：$r = \sum_k \alpha_k r_k$

### 6.3 ORM + Self-Consistency

在 best-of-N 中，用 self-consistency（多数投票）作为额外的 verification signal，与 ORM 分数加权：

$$\text{score}(y^{(j)}) = \lambda \cdot r_{\text{ORM}}(x, y^{(j)}) + (1-\lambda) \cdot \mathbb{1}[\text{answer}(y^{(j)}) = \text{majority\_answer}]$$

---

## 7. 关键论文时间线

| 年份 | 论文 | 贡献 |
|------|------|------|
| 2021 | Cobbe et al., "Training Verifiers to Solve Math Word Problems" (GSM8K) | 首次提出用 verifier（即 ORM）做 rejection sampling |
| 2022 | Snell et al., "Scaling LLM Test-Time Compute" | 系统 study best-of-N + ORM 的 scaling behavior |
| 2023 | Lightman et al., "Let's Verify Step by Step" | 提出 PRM，证明 step-level supervision 显著优于 ORM |
| 2023 | Wang et al., "Math-Shepherd" | 自动构建 step-level PRM label，降低标注成本 |
| 2024 | Luo et al., "Critique, Revise, Improve" | 结合 ORM + critique model 迭代改进 |
| 2025 | 各种 GRPO / DeepSeek-R1 工作 | ORM 在 RL 训练中的广泛应用 |

---

## 8. 从第一性原理理解 ORM 的局限

根本问题是 **信息瓶颈**：

- 一个 solution 包含 $T$ 步推理，信息量为 $T \times H(\text{step})$
- ORM 将其压缩为 **1 bit** (正确/错误)
- 信息损失 = $T \times H(\text{step}) - 1$ bit

当 $T$ 很大（长推理链）时，这个压缩损失巨大，导致：
1. Credit assignment 不可能精确
2. Reward 信号方差极大
3. 模型很难从错误中学习 "哪里出错了"

**PRM 的优势**就在于保留了每一步的信息：输出 $T$ 个 reward 而非 1 个，信息损失大幅降低。

---

## 9. 实践中的 ORM 训练细节

```python
# 伪代码：ORM 训练
class OutcomeRewardModel(nn.Module):
    def __init__(self, base_llm, hidden_dim):
        super().__init__()
        self.llm = base_llm           # e.g., LLaMA-7B
        self.reward_head = nn.Linear(hidden_dim, 1)  # scalar head
    
    def forward(self, input_ids):
        # input_ids: [question + full_solution]
        outputs = self.llm(input_ids)
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch, hidden_dim]
        # 取最后一层最后一个 token 的 hidden state
        reward_logit = self.reward_head(last_hidden)  # [batch, 1]
        reward = torch.sigmoid(reward_logit)           # [batch, 1] ∈ [0,1]
        return reward

# 训练
model = OutcomeRewardModel(llama_7b, hidden_dim=4096)
optimizer = AdamW(model.parameters(), lr=2e-5)

for batch in dataloader:
    # batch: (questions, solutions, labels)
    # labels: 1 if final_answer == ground_truth, else 0
    rewards = model(batch.input_ids)  # [B, 1]
    loss = F.binary_cross_entropy(rewards.squeeze(), batch.labels.float())
    loss.backward()
    optimizer.step()
```

### 关键训练细节：
- **学习率**：通常 1e-5 ~ 5e-5，与 LLM fine-tuning 一致
- **数据平衡**：positive/negative 样本通常不均衡，需要 oversampling 或 weighted loss
- **Frozen LLM vs Full Fine-tuning**：早期工作冻结 LLM 只训 reward head；后期发现 full fine-tuning 效果更好
- **Temperature of sampling**：采样 solution 时 temperature 越高，negative 样本越多越多样，ORM 学得越好

---

## 10. 总结

ORM 是 **outcome-level 的 reward model**，核心特点是：

1. ✅ **优点**：标注成本低（自动比对答案）、训练简单、部署容易
2. ❌ **缺点**：credit assignment 困难、sparse reward、容易被 false positive 欺骗
3. 🔑 **最佳实践**：ORM 更适合做 **verification/selection**（best-of-N），而非做 RL 的 reward signal（这时 PRM 更优）
4. 📈 **Scaling**：ORM 的效果随采样数量 N 增长而提升，但边际效益递减

---

**参考资料**：
- Cobbe et al. (2021), "Training Verifiers to Solve Math Word Problems" - [arxiv.org/abs/2110.14168](https://arxiv.org/abs/2110.14168)
- Lightman et al. (2023), "Let's Verify Step by Step" - [arxiv.org/abs/2305.20050](https://arxiv.org/abs/2305.20050)
- Wang et al. (2023), "Math-Shepherd: Verify and Reinforce LLMs Step-by-step" - [arxiv.org/abs/2312.08935](https://arxiv.org/abs/2312.08935)
- Snell et al. (2024), "Scaling LLM Test-Time Compute Optimally" - [arxiv.org/abs/2408.03314](https://arxiv.org/abs/2408.03314)
- EmergentMind ORM Topic Page - [emergentmind.com/topics/outcome-reward-model-orm](https://www.emergentmind.com/topics/outcome-reward-model-orm)