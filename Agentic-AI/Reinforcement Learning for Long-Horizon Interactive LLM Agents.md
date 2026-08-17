---
source_pdf: Reinforcement Learning for Long-Horizon Interactive LLM Agents.pdf
paper_sha256: 632efe95b1d8c4ea09b66734e3fc5088ed994cbe8da021fc4d17f749f5be59dd
processed_at: '2026-08-11T22:17:00-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 一句话总结

**用RL训练agent在复杂app环境里干活，用最朴素的reward（任务做对了没），32B小模型干翻了OpenAI o1。**

## 故事背景

想象你有个assistant，你要让它帮你做这种事：

> "Kristin帮我付了 groceries，你查查我们短信里说的金额，用Venmo还她钱，附言写'Groceries'，然后给她发条短信说'It is done.'"

对人来说这事儿不难，但对AI来说简直是地狱级难度。你得：
- 先登录phone app
- 查contacts找到Kristin的号码
- 搜text messages找到那笔$54
- 切换到venmo app登录
- 搜user找到Kristin
- 发钱
- 切回phone app发短信

整个过程中有**457个API**，**1470个参数**，环境state有**30M tokens**。一个任务可能要**40轮交互**。最强的模型OpenAI o1也只能勉强过一半。

## 核心问题

以前大家怎么做？要么prompt engineering，要么SFT（拿专家写的solution去模仿）。但问题是：

**SFT会教坏agent**。你给它看专家solution，它就开始背答案，而不是学会"怎么跟环境互动"。论文里有个特别惨的实验：用ground truth solution做SFT，结果test性能从39掉到6。Agent学会了背train set的答案，遇到没见过的task就傻眼了。

## LOOP算法到底干了啥

核心idea特别简单，就三件事：

### 1. 多次尝试，自己跟自己比

对每个task，让agent跑6遍（temperature=1.0采样）。比如3遍成功、3遍失败。然后：

- 成功的那些rollout，advantage = 这次的reward - 其他5次的平均reward > 0 → 鼓励
- 失败的那些rollout，advantage < 0 → 抑制

这就是**Leave-One-Out baseline**：用其他5个sample的平均return当baseline，不学value function。简单、无偏、低variance。

### 2. Per-token而不是per-trajectory做importance weighting

这是论文一个关键insight。假设你跑了rollout，现在要做gradient update。由于policy已经更新了，你需要importance weight来修正：

$$\frac{p_\theta(\text{new})}{p_\psi(\text{old})}$$

有两种做法：

**Per-trajectory**：把所有token的概率乘起来算一个ratio。问题在于：trajectory可能有几百个token，一个token概率稍微变一点，整个ratio就爆了，gradient直接被clip掉，啥也学不到。

**Per-token**：每个token单独算ratio。一个token的ratio爆了，只影响那个token的gradient，其他token照学不误。

实验数据：per-trajectory 53.3 TGC，per-token **71.3 TGC**。差了18个点。就这么一个看似trivial的选择，效果天差地别。

### 3. PPO的trust region + off-policy reuse

纯on-policy（RLOO）的问题是：采完一波rollout只能做一次gradient update，太浪费了。LOOP允许你把rollouts存在buffer里，跑多个epoch，amortize rollout的成本。

但off-policy会引入bias（policy已经变了，老的rollouts不那么relevant了）。PPO的clip机制就是处理这个：如果new policy和old policy的ratio偏离太远（超过1±ε），就clip掉，不trust这个gradient。

## 跟GRPO/RLOO的关系

讲清楚这三个的关系很重要：

- **RLOO**：on-policy，Leave-One-Out baseline，采一波rollout做一次update。最简单。
- **GRPO**（DeepSeek搞的）：在RLOO基础上，advantage除以returns的standard deviation做normalization。想法是让"sometimes成功sometimes失败"的task贡献更stable。
- **LOOP**：在RLOO基础上，加PPO的off-policy机制（多epoch、per-token clip），**不做normalization**。

关键发现：**GRPO的normalization反而有害**。LOOP with normalization掉9个点，接近GRPO水平。LOOP without normalization才是71.3。

为什么？normalization会压制那些"sometimes solvable"的task——恰恰是信息量最大的task。全成功或全失败的task，normalize后advantage趋近0，学不到东西。而"有时成功"的task，恰恰告诉你"成功路径和失败路径的区别在哪"。

## 训练完agent学会了啥

这是论文最有意思的部分。作者不是只报个数字就完了，而是深入分析了agent行为的变化：

### 1. 不再open-loop批量提交

训练前：一口气写3个code cell，假设每个都会成功
训练后：写一个cell，看结果，再写下一个

频率下降6x。这就是从"open-loop control"变成"closed-loop control"。

### 2. 主动查API文档

训练前：凭记忆猜API参数
训练后：调用前先`show_api_doc`

频率上升1.6x。环境有457个API，不查文档就是作死。

### 3. 不乱assume

训练前：
```
"Get the list of roommates (assuming roommates are friends in Venmo)"
```
然后就用venmo friends当roommates，结果错了。

训练后：老老实实用`phone.search_contacts(relationship="roommate")`查。

"assuming"相关词出现频率下降**30x**。

### 4. 不用dummy值

训练前：
```python
apis.venmo.login(password='dummy_venmo_pass')  # 凭空捏造密码
```

训练后：
```python
passwords = apis.supervisor.show_account_passwords()
venmo_password = [p for p in passwords if p['account_name']=='venmo'][0]['password']
```

"dummy"出现频率下降6x。

### 5. 遇挫不放弃

训练前：API报错→"看来这个app有问题，我们换一个吧"
训练后：API报错→debug→重试

放弃率下降3x。

## 最反直觉的发现：小数据为什么不overfit

只用了24个scenario训练，72个task。RL方法却没overfit，反而generalize得很好。为什么？

**因为LLM sampling的天然多样性**。

看Figure 4：同一个task跑100个rollout，98个成功，但94个的API call sequence都不一样。至少有4种不同的解法strategy。

这种多样性有两个好处：
- **训练早期**：相当于free的exploration，不同rollout探索不同路径，总有一些能成功
- **训练后期**：防止policy collapse到单一mode。即使在小数据上，也不会死记硬背一种解法

对比SFT：RFT在successful rollouts上做SFT，loss→0，agent开始collapse到单一solution。RL的policy gradient天然保持分布的spread。

## 一些技术细节值得注意

### 为什么不用learned critic

论文也试了traditional PPO with value network。结果：
- Value MSE一直在0.01以上，学不准
- $\lambda_{GAE} < 1$时直接diverge
- 需要pretrain critic 10 iterations

Long-horizon下value estimation太难了。Monte Carlo baseline（leave-one-out）虽然variance高，但unbiased。在这个setting下，unbiased比low-variance-but-biased更好。

### POMDP形式化的细节

environment会主动往context里加tokens（API返回结果）。所以importance weight只算LLM产生的tokens，environment产生的tokens ratio为1。这个细节很重要，否则你会错误地对environment response也算importance weight。

### LoRA + CCE

用LoRA（r=16）fine-tune，加Cut Cross-Entropy节省memory。否则32B模型materialize所有logits会爆显存。

## 我的理解与critique

### 亮点

1. **POMDP形式化很干净**：正确处理了environment产生tokens的问题
2. **Per-token vs per-trajectory的insight**：简单但重要，解释清楚了很多RLHF实现里的坑
3. **行为分析特别深入**：不只是报数字，而是真正分析了agent学到了什么
4. **小数据泛化的解释**：多样性作为implicit regularization，这个视角很有启发

### 可能的concerns

1. **计算成本**：42小时 × 16 H100。 academia复现困难
2. **单一benchmark**：只在AppWorld上测，是否transfer到其他agent setting未知
3. **Reward hacking没讨论**：unit test通过≠真正完成任务，agent有没有找到trick绕过test
4. **为什么normalization有害**：论文给了intuition但没严格证明。可能与AppWorld的reward分布特性有关，其他benchmark未必如此

### 对未来工作的启示

1. **Agent训练就该用RL**：SFT会教坏，RL让agent自己探索
2. **Long-horizon下别用learned critic**：Monte Carlo baseline够用
3. **保持sampling多样性**：temperature、多个rollout、per-token clip，都是为了维护多样性
4. **简单reward就够了**：不需要复杂的reward shaping，task completion就够

### 与DeepSeek-R1的联系

这篇paper和DeepSeek-R1的训练方法是一脉相承的。R1用的GRPO，LOOP是GRPO的改进版（去掉normalization + 加PPO off-policy）。R1证明了RL能教reasoning，这篇paper证明了RL能教agent interaction。共同insight：**RL的exploration + LLM的intrinsic diversity = 涌现能力**。

## 代码直觉

如果你要实现LOOP，核心loop长这样：

```python
for iteration in range(N_iterations):
    # 1. Rollout collection
    rollouts = []
    for task in sample_tasks(dataset, n=40):
        for k in range(K=6):
            trajectory = rollout(agent, task.env, max_steps=40)
            rollouts.append((task, trajectory))
    
    # 2. Compute leave-one-out advantages
    for task, group in group_by_task(rollouts):
        returns = [r.reward for r in group]
        mean_return = np.mean(returns)
        for r in group:
            # leave-one-out: subtract mean of OTHERS
            r.advantage = (r.reward - mean_return) * K / (K - 1)
    
    # 3. Off-policy PPO updates (multiple epochs)
    for epoch in range(N_epochs):
        for minibatch in shuffle_and_batch(rollouts, batch_size=M):
            for trajectory in minibatch:
                for token in trajectory.llm_tokens:  # only LLM tokens!
                    ratio = exp(log_prob_new(token) - log_prob_old(token))
                    clipped_ratio = clip(ratio, 1-eps, 1+eps)
                    loss = -min(ratio * adv, clipped_ratio * adv)
                    loss.backward()
```

注意几个关键点：
- `group_by_task`：advantage在同一个task的rollouts内计算
- `trajectory.llm_tokens`：只对LLM产生的token算log_prob，跳过environment response
- 多个epoch：这是off-policy的部分，amortize rollout cost

## 总结

这篇paper的核心message其实很simple：

**别把agent训练想得太复杂。让agent在环境里多跑几次，用Leave-One-Out算个简单baseline，用per-token PPO clip做trust region，给个task completion reward，就够了。小模型也能干翻大模型。**

但simple不意味着trivial。里面的每个设计选择（per-token vs per-trajectory、no normalization、Monte Carlo vs learned critic）都有empirical evidence支撑。这才是好的research——simple method + solid analysis + clear insights。

---

# Reinforcement Learning for Long-Horizon Interactive LLM Agents 深度解析

## 1. 论文核心问题与动机

这篇论文解决了一个非常实际的问题：如何让LLM-based agent在复杂的、有状态的数字环境中完成长期任务。AppWorld benchmark是当前最具挑战性的IDA benchmark，包含：

- **9个apps**（email, payments, music, shopping, phone, file system等）
- **457个API endpoints**，总共**1470个function parameters**
- 环境state可达**30M text tokens**
- 单个任务可能需要**40次交互**，消耗**32K tokens**

在这个benchmark上，最强open-weights模型（Llama 3 70B）成功率仅24.4%（Test-N），OpenAI o1也仅61.9%。这揭示了一个根本问题：instruction-tuned LLM虽然能react to environment feedback，但从未在目标环境中被trained过。

## 2. POMDP形式化

### 2.1 状态空间定义

论文将IDA任务formalize为POMDP。这是关键的理论贡献，因为与传统的LM generation不同，这里environment会主动产生tokens。

**State定义**：$[\mathbf{s}_0, \mathbf{c}, x_{1:t}]$

- $\mathbf{s}_0$：environment的初始状态（Python REPL + relational database）
- $\mathbf{c}$：task context（user prompt，m个tokens：$[c_1 \dots c_m]$）
- $x_{1:t}$：至今的generation，混合了LLM产生的tokens和environment产生的tokens

### 2.2 转移动力学

大部分transition只是简单append一个token：
$$[\mathbf{s}_0, \mathbf{c}, x_{1:t}] \to [\mathbf{s}_0, \mathbf{c}, x_{1:t+1}]$$

但当LLM emit stop token时，会触发**code execution**，此时transition会同时append generated token $x_{t+1}$ 和tokenized environment response $x_{t+2:t+1+k}$：
$$[\mathbf{s}_0, \mathbf{c}, x_{1:t}] \to [\mathbf{s}_0, \mathbf{c}, x_{1:t+1+k}]$$

### 2.3 轨迹分布

定义 $a(\mathbf{x}) \subseteq \{1, \dots, T\}$ 为trajectory中由LLM产生的token子集（vs environment response的tokens）。$\mathbb{I}(\mathbf{s}_0, \mathbf{x}) \in \{0, 1\}$ 是indicator，表示trajectory的API responses是否与initial state $\mathbf{s}_0$ 一致。

轨迹分布为：
$$\rho_\theta(\mathbf{x}|\mathbf{s}_0, \mathbf{c}) := \mathbb{I}(\mathbf{s}_0, \mathbf{x}) \prod_{t \in a(\mathbf{x})}^T p_\theta(x_t|\mathbf{c}, x_{1:t-1})$$

**关键insight**：importance weight只需要对LLM产生的tokens计算，environment产生的tokens的ratio为1：
$$\frac{\rho_\theta(\mathbf{x}|\mathbf{s}_0, \mathbf{c})}{\rho_\psi(\mathbf{x}|\mathbf{s}_0, \mathbf{c})} = \prod_{t \in a(\mathbf{x})} \frac{p_\theta(x_t|\mathbf{c}, x_{1:t-1})}{p_\psi(x_t|\mathbf{c}, x_{1:t-1})}$$

## 3. LOOP算法详解

### 3.1 Leave-One-Out Advantage Estimation

LOOP的核心是结合了PPO的trust region机制和RLOO的Monte Carlo baseline。

**RLOO的advantage估计**：对于K个i.i.d. samples $\mathbf{x}_1, \ldots, \mathbf{x}_K \sim p_\theta(\cdot|\mathbf{c})$：

$$A(\mathbf{c}, \mathbf{x}_k) = R(\mathbf{c}, \mathbf{x}_k) - \frac{1}{K-1} \sum_{i=1, i \neq k}^K R(\mathbf{c}, \mathbf{x}_i)$$

等价形式（计算更方便）：
$$A(\mathbf{c}, \mathbf{x}_k) = \frac{K}{K-1} \left(R(\mathbf{c}, \mathbf{x}_k) - \frac{1}{K} \sum_{i=1}^K R(\mathbf{c}, \mathbf{x}_i)\right)$$

**变量解释**：
- $K$：每个task采样的rollout数量（论文中K=6）
- $R(\mathbf{c}, \mathbf{x}_k) \in [0, 1]$：第k个rollout的reward（通过的unit tests比例）
- $\frac{1}{K} \sum R(\mathbf{c}, \mathbf{x}_i)$：所有rollouts的平均return，作为baseline

这个estimator是**无偏的**，且variance比standard REINFORCE低很多，因为每个sample的baseline使用了其他K-1个samples的信息。

### 3.2 Per-Token PPO Objective

LOOP使用per-token importance weights，而非per-trajectory或per-turn：

$$L_\theta^{\text{MDP}}(\mathbf{s}_0, \mathbf{c}) = \mathbb{E}_{\mathbf{x} \sim \rho_\psi(\cdot|\mathbf{s}_0, \mathbf{c})} \left[\frac{1}{|a(\mathbf{x})|} \sum_{t \in a(\mathbf{x})} \min\left(\frac{p_\theta(x_t|\mathbf{c}, x_{1:t-1})}{p_\psi(x_t|\mathbf{c}, x_{1:t-1})} A(\mathbf{s}_0, \mathbf{c}, \mathbf{x}), g_\epsilon(A(\mathbf{s}_0, \mathbf{c}, \mathbf{x}))\right)\right]$$

其中clipping function：
$$g_\epsilon(A) = \text{clip}(A, -\epsilon A, \epsilon A) = \begin{cases} A & \text{if } \frac{p_\theta}{p_\psi} \in [1-\epsilon, 1+\epsilon] \\ (1+\epsilon)A & \text{if } A > 0 \text{ and } \frac{p_\theta}{p_\psi} > 1+\epsilon \\ (1-\epsilon)A & \text{if } A < 0 \text{ and } \frac{p_\theta}{p_\psi} < 1-\epsilon \end{cases}$$

**为什么per-token比per-trajectory好？**

- Per-trajectory：单个token概率变化会stop整个trajectory的gradient update
- Per-token：单个token概率变化只影响自己的gradient

实验数据证实了这一点：

| Action定义 | Test-N TGC | Test-C TGC |
|-----------|-----------|-----------|
| trajectory (bandit) | 53.3 ± 3.4 | 27.7 ± 1.5 |
| turn | 64.1 ± 2.2 | 40.8 ± 1.5 |
| **token** | **71.3 ± 1.3** | **45.7 ± 1.3** |

### 3.3 Algorithm 1伪代码解析

```
Algorithm 1: Leave-One-Out Proximal Policy Optimization
Input: Policy p_θ, dataset of tasks and initial states D
Output: Policy p_θ maximizing E_{s_0, c ~ D}[L_θ(s_0, c)]

1: for iteration Ψ = 1, 2, ... do
2:   B ← {}                                    # Initialize rollout buffer
3:   for (s_0, c) ~ D do                       # Rollout collection
4:     Collect K rollouts x_1, ..., x_K ~ ρ_θ(·|s_0, c)
5:     Estimate advantages A_1, ..., A_K using Eq. 3
6:     B ← B ∪ {(x_1, A_1), ..., (x_K, A_K)}
7:   for epoch = 1, ..., N_epoch do             # Policy update
8:     for mini-batch {(x_i, A_i)}_{i=1}^M ~ B do
9:       Update policy using PPO gradient (Eq. 5)
```

**关键设计点**：
1. **Line 4-5**：每个task采样K=6个rollouts，计算leave-one-out advantage
2. **Line 6**：所有rollouts放入buffer，irrespective of their initial state-context pair
3. **Line 7-9**：可以多epoch训练（off-policy），amortize rollout cost

### 3.4 与RLOO和GRPO的关系

**LOOP作为RLOO**：当 $N_{\text{epoch}} = 1$ 且无mini-batches时（纯on-policy），PPO update reduce到REINFORCE，LOOP = RLOO。

**LOOP作为GRPO**：LOOP和GRPO的主要区别在于advantage estimation。GRPO使用：
$$A_{\text{GRPO}} = \frac{A_{\text{RLOO}}}{\sigma_R + \epsilon}$$

其中 $\sigma_R$ 是returns的standard deviation。这种normalization**disproportionally favors低标准差的trajectories**（即LLM获得consistent return的trajectories）。

实验发现，**forgoing这种normalization是有益的**：

| 方法 | Normalized reward | Test-N TGC | Test-C TGC |
|-----|-----------------|-----------|-----------|
| GRPO | √ | 58.0 ± 1.8 | 39.5 ± 1.9 |
| GRPO no kl | √ | 59.0 ± 1.4 | 42.7 ± 1.3 |
| LOOP (token) | × | **71.3 ± 1.3** | **45.7 ± 1.3** |
| LOOP RwNorm (token) | √ | 61.9 ± 4.0 | 39.8 ± 1.3 |

Reward normalization导致9 pp下降！原因是在AppWorld中，那些"sometimes solvable"的scenarios提供了最informative的training signal，而normalization会压制这些scenarios的贡献。

## 4. 实验设置细节

### 4.1 训练配置

- **Base model**：Qwen2.5-32B-Instruct
- **Fine-tuning**：LoRA (rank r=16, α=32)，applied to self-attention (Q, K, V, O) 和 MLP
- **Memory optimization**：Cut Cross-Entropy (CCE) 避免materializing logits for all tokens
- **Training data**：24 scenarios × 3 tasks = 72 tasks（仅difficulty 1和2）
- **Rollouts per task**：K = 6
- **Tasks per iteration**：40（total 240 rollouts）
- **Max interactions**：40 (training) / 50 (evaluation)
- **Hardware**：2× NVIDIA H100 8-GPU nodes（1 for rollout, 1 for training）
- **Training time**：42 hours

### 4.2 Reward设计

$$R(\mathbf{s}_0, \mathbf{c}, \mathbf{x}) \in [0, 1]$$

简单定义为通过的unit tests比例。AppWorld的unit tests检查：
1. 请求的environment state changes是否成功
2. 是否没有extraneous changes
3. Final answer是否匹配ground truth

### 4.3 Performance optimization

- 移除low advantage的rollouts（$|\hat{A}^{(i,j)}| < 0.01$）再计算gradient
- Early-stop rollout collection：当至少4 rollouts per task和90% total rollouts完成时
- Recompute per-token log-probabilities under generating policy（而非使用vLLM报告的值）

## 5. 主要实验结果

### 5.1 主表对比

| Type | Algorithm | Test-N TGC | Test-C TGC |
|-----|----------|-----------|-----------|
| NFT | GPT-4o | 48.8 | 30.2 |
| NFT | OpenAI o1 | 61.9 | 36.7 |
| NFT | Llama 3 70B | 24.4 | 7.0 |
| NFT | Qwen2.5-32B | 39.2 ± 3.5 | 21.0 ± 1.4 |
| SFT | SFT-GT | 6.2 ± 0.7 | 0.8 ± 0.2 |
| SFT | RFT | 47.9 ± 3.7 | 26.4 ± 1.8 |
| SFT | EI | 58.3 ± 2.8 | 32.8 ± 0.7 |
| DPO | DPO-MCTS | 57.0 ± 1.5 | 31.8 ± 1.3 |
| DPO | DMPO | 59.0 ± 1.2 | 36.3 ± 1.8 |
| RL | PPO (learned critic) | 50.8 ± 3.7 | 26.4 ± 0.5 |
| RL | RLOO | 57.2 ± 2.6 | 36.7 ± 1.6 |
| RL | GRPO | 58.0 ± 1.8 | 39.5 ± 1.9 |
| RL | **LOOP (token)** | **71.3 ± 1.3** | **45.7 ± 1.3** |

**关键发现**：
1. LOOP (token)比OpenAI o1高9 pp（15% relative on Test-N, 24% on Test-C）
2. 比base Qwen2.5-32B高81%（Test-N）和117%（Test-C）
3. 所有fine-tuning方法都显著优于base model
4. Performance在~59 TGC附近saturate，除了LOOP (turn, token)
5. 所有Monte Carlo baseline方法都优于PPO with learned value function

### 5.2 SFT-GT的失败

SFT-GT（在ground truth solutions上supervised fine-tuning）表现极差（6.2 TGC on Test-N）。原因：

1. Ground truth solutions需要a priori knowledge of AppWorld state来构造
2. 有些任务的solution不可能without environment interaction来构造
3. Fine-tuning导致agent从"attempting environment interaction"转向"memorization of solution steps"
4. Train performance先degrade再recover，但dev/test performance不recover

这证明了**单纯模仿专家轨迹是不够的**，agent需要学会如何与环境交互。

### 5.3 PPO with Learned Critic的问题

论文implement了traditional PPO with learned critic，但表现不佳（50.8 TGC on Test-N）：

- Value function用3-layer MLP：[(5120×3072), (3072×2048), (2048×1)]
- 输入是policy network的last hidden state（dim=5120）
- Value predictions的MSE大部分时间>0.01，说明value estimation本质上很难
- $\lambda_{\text{GAE}} < 1.0$ 导致training divergence
- 需要pre-train critic 10 iterations（2400 rollouts）

**根本原因**：在long-horizon tasks中，value estimation的误差会被amplify。Monte Carlo baseline（RLOO）虽然variance高，但unbiased。

## 6. 涌现行为分析

这是论文最fascinating的部分。通过简单的task completion reward，agent自发学会了多种好的behaviors：

### 6.1 避免Open-Loop Control

**问题**：早期agent会一次性submit多个Python code cells，假设每个都会成功。这是decision-theoretically suboptimal的open-loop control。

**学习效果**：多code cell per turn的频率减少**~6x**，但total code submitted没有显著减少。

**Intuition**：Agent学会了"执行-观察-执行"的闭环控制，而非"批量执行-祈祷成功"。

### 6.2 一致性查阅API Documentation

**问题**：AppWorld有457个API endpoints，1470个function parameters，不可能记住所有细节。

**学习效果**：`show_api_doc`调用频率增加**~1.6x**。Agent学会了在调用任何API前先查阅文档。

### 6.3 减少Assumptions

**典型反模式**：
```
"Get the list of roommates (assuming roommates are friends in Venmo)"
```

这种early assumption不会被revisit，可能causes mistakes far downstream。

**学习效果**："assuming"及相关词汇减少**~30x**。Agent学会了explicitly search for the 'roommate' relationship in phone app，而非assume friends = roommates。

### 6.4 减少Placeholder Values

**典型反模式**：
```python
login_result = apis.venmo.login(username='mel.bailey@gmail.com', password='dummy_venmo_pass')
```

**学习效果**："dummy"词汇减少**~6x**。Agent学会了从supervisor app获取真实password。

### 6.5 从Setback中恢复

**典型反模式**：API error后立即give up，开始做其他subtask。

```
"It seems there's an issue with accessing the 'phone' app...
 Since we can't currently use this app to retrieve the roommates..."
```

**学习效果**：failed API call give up rate减少**~3x**。Agent学会了persevere和debug。

### 6.6 行为多样性的重要性

Figure 4展示了100个i.i.d. rollouts on the same task：
- 98/100成功完成任务
- 94/98有unique API call sequences
- 至少4种distinct strategies：
  1. Directly search roommate contacts via `phone.search_contacts`
  2. Browse Venmo social feed first via `venmo.show_social_feed`
  3. Query all contact relationships via `phone.contact_relationships`
  4. `venmo.show_social_feed` followed by `phone.contact_relationships`

这种多样性是RL在小数据上有效的关键原因：
- **Early training**：fosters exploration，发现improve over base model的solutions
- **Late training**：prevents collapse onto single solution，fosters generalization

## 7. 为什么RL在小数据上有效？

这是论文Section 5.6讨论的核心问题。24 scenarios, 72 tasks——这么少的数据，为什么RL没有overfit？

### 7.1 LLM采样的内在多样性

**微观层面**：LLM的token-level sampling rarely产生相同的solution。即使late in training，同一task的6个rollouts通常有不同的API call sequences。

**宏观层面**：LLM维护多个distinct solution "phenotypes"并jointly improve all of them。

### 7.2 与SFT的对比

SFT方法（RFT, SFT-GT, EI）倾向于collapse到单一solution mode：
- RFT在successful rollouts上fine-tune，loss → 0
- EI显示higher degree of overfitting to training data

RL方法（RLOO, GRPO, LOOP）通过policy gradient的stochastic nature保持了多样性。

### 7.3 Leave-One-Out的Exploration效果

RLOO的advantage估计有个interesting property：当一个task有时成功有时失败时（$\sigma_R > 0$），成功的rollouts获得positive advantage，失败的获得negative advantage。这比binary的"成功=1, 失败=忽略"（RFT）提供了更rich的training signal。

## 8. 失败案例分析

### 8.1 Spotify下载失败案例（Appendix G.5）

Task: "Download all the songs from my Spotify song library that I have liked."

**Agent行为**：
1. Login to Spotify ✓
2. Get liked songs ✓
3. Try to download all liked songs → Error: "already downloaded"
4. Get downloaded songs list
5. Download liked songs not in downloaded list ✓

**失败原因**：Agent下载了所有liked songs，包括那些not in user's library的。任务要求是"songs from my Spotify song library that I have liked"，但agent理解为"all liked songs"。

这个case说明：即使经过RL训练，agent仍然会在subtle的semantic distinctions上犯错。

### 8.2 成功案例的Debugging行为（Appendix G.4）

Task: "Kristin paid for my grocery... Send them the owed money..."

**Agent的debugging过程**：
1. Login with phone number '48886643554' → Error: Invalid credentials
2. Realize typo, retry with '4886643554' → Success
3. Search text messages with phone_number="Kristin" → Error: user not exist
4. Realize需要先找Kristin的phone number
5. Search contacts for "Kristin" → Find phone number
6. Search text messages with correct phone number → Find $54 grocery payment
7. Continue with Venmo payment...

这个case展示了RL训练后的agent的resilience：遇到error不give up，而是diagnose并retry。

## 9. 技术细节补充

### 9.1 Training Reward的具体定义

每个AppWorld task有一组unit tests检查：
1. **State changes**：请求的environment state changes是否成功made
2. **No extraneous changes**：是否introduced任何undesired changes
3. **Final answer**：Final answer是否匹配ground truth（如适用）

Reward = 通过的unit tests比例 ∈ [0, 1]

### 9.2 Failed API Call Give Up Rate的计算

这个metric的定义比较复杂（Appendix A.1）：

1. Track set of failed API endpoints not yet retried
2. For each turn:
   - If execution error: add attempted endpoints to failed set
   - If success: remove endpoints from failed set, count as recovered
3. Give up rate = (total failed - total recovered) / total failed

### 9.3 PPO with Learned Critic的实现细节

Value function architecture：
- 3-layer MLP: [(5120×3072), (3072×2048), (2048×1)]
- ReLU activations
- Input: last hidden state h (dim=5120) from policy network
- L2 loss with coefficient linearly decaying from 0.1 to 0.001 over 200 iterations
- Value predictions clipped to [0.0, 1.0]
- Value loss gradients not propagated to policy's LoRA weights（会destabilize training）

**关键发现**：$\lambda_{\text{GAE}} \ll 1$时，advantage estimates bootstrap from value predictions later in the (potentially very long) rollout，amplifying critic's errors。$\lambda_{\text{GAE}} = 1.0$（无discounting）最stable。

## 10. 局限性与未来方向

### 10.1 当前局限

1. **成功率仍不够高**：最好的agent也只~70% TGC on Test-N
2. **AppWorld仍缺少real-world features**：
   - Non-determinism
   - Transient failures
   - Unsolvable and ambiguous tasks
   - Adversarial scenarios (e.g. scams)
   - User clarification steps
   - Interactive counterparties

### 10.2 未来研究方向

1. **更复杂的reward shaping**：当前是binary task completion，可以加入intermediate rewards
2. **Curriculum learning**：从简单task到复杂task
3. **Multi-agent scenarios**：与customer service representatives等interactive counterparties
4. **Robustness to adversarial inputs**：scams, prompt injection等
5. **Larger scale RL**：当前只用了24 scenarios，可以探索更多数据的效果

## 11. 相关工作对比

### 11.1 与ArCHer的区别

ArCHer (Zhou & Zanette, 2024)使用hierarchical approach combining off-policy和on-policy training。LOOP更简单，不需要hierarchical decomposition。

### 11.2 与AgentQ的区别

AgentQ (Putta et al., 2024) combines MCTS with DPO。论文implement了simplified version（DPO-MCTS），不使用LLM critic heuristic。LOOP在AppWorld上显著优于DPO-MCTS。

### 11.3 与WebShop工作的区别

WebShop (Yao et al., 2022)只有8个actions，最多1 parameter per turn。AppWorld有457个API endpoints，最多17 parameters，需要nontrivial logic。LOOP是首次在如此复杂的环境中应用RL的工作。

## 12. 个人insight与批评

### 12.1 论文的亮点

1. **POMDP形式化很优雅**：正确处理了environment产生tokens的情况
2. **Per-token importance weight的insight**：解释了为什么per-token > per-trajectory
3. **涌现行为分析很深入**：不仅仅是performance数字，还分析了agent学到了什么
4. **与RLOO和GRPO的关系澄清**：LOOP是RLOO的off-policy generalization

### 12.2 可能的concerns

1. **计算成本**：42小时 × 16 H100 GPUs，对于academia可能prohibitive
2. **只测了一个benchmark**：AppWorld虽然复杂，但仍是一个specific environment
3. **Reward hacking**：论文没有讨论agent是否找到了ways to pass unit tests without真正完成任务
4. **Generalization**：训练在24 scenarios上，test在195 scenarios上，但都是同一benchmark的variants

### 12.3 与DeepSeek-R1的联系

论文引用了DeepSeek-R1 (DeepSeek-AI, 2025)，而GRPO正是DeepSeek提出的方法。LOOP可以看作是GRPO的改进版本：去掉了reward normalization，改用PPO的off-policy机制。这建立了一个有趣的connection：reasoning LLM的训练方法可以adapted到agent训练。

### 12.4 对LLM agent训练的启示

1. **RL > SFT for agents**：所有RL方法都显著优于SFT方法
2. **Monte Carlo > Learned critic**：在long-horizon tasks中，value estimation太难
3. **多样性是feature不是bug**：LLM的stochastic sampling提供了natural exploration
4. **简单reward可以induce复杂behavior**：task completion reward就足够了

## 参考文献

- 原论文：[Reinforcement Learning for Long-Horizon Interactive LLM Agents](https://arxiv.org/abs/2502.?????)
- AppWorld benchmark: [Trivedi et al., 2024](https://arxiv.org/abs/2407.?????)
- RLOO: [Kool et al., 2019](https://arxiv.org/abs/1905.?????), [Ahmadian et al., 2024](https://arxiv.org/abs/2402.?????)
- GRPO/DeepSeekMath: [Shao et al., 2024](https://arxiv.org/abs/2402.03300)
- PPO: [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- VinePPO: [Kazemnejad et al., 2024](https://arxiv.org/abs/2410.01679)
- AgentQ: [Putta et al., 2024](https://arxiv.org/abs/2408.07199)
- ArCHer: [Zhou & Zanette, 2024](https://arxiv.org/abs/2402.?????)
- Qwen2.5: [Yang et al., 2024](https://arxiv.org/abs/2412.15115)
- LoRA: [Hu et al., 2022](https://arxiv.org/abs/2106.09685)
- Cut Cross-Entropy: [Wijmans et al., 2024](https://arxiv.org/abs/2411.09009)
- ReAct: [Yao et al., 2023](https://arxiv.org/abs/2210.03629)
- DeepSeek-R1: [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948)

---

这篇论文的贡献不仅仅是LOOP算法本身，更重要的是它**系统性地验证了RL在LLM agent训练中的有效性**，并提供了丰富的empirical evidence和behavioral analysis。它established了几个重要的empirical findings：per-token > per-trajectory importance weights, Monte Carlo > learned critic, 多样性是RL成功的关键。这些findings对未来的LLM agent研究有重要指导意义。
