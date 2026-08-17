---
source_pdf: DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement
  Learning.pdf
paper_sha256: b191b0a365a64b4ab2791d117069ed17a2933d03554a662ced58b37df52018f4
processed_at: '2026-08-03T18:44:04-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DeepSeek-R1 人话版

Karpathy，我用你喜欢的直觉性语言重讲一遍。

## 一句话版本

DeepSeek 证明了：**base model 已经会 reason，RL 只是把这个能力"逼出来"**。不用 human CoT demo，不用 SFT warm-up，只给"答案对错"的信号，model 自己学会反思、验证、回溯。

## 核心故事

传统 pipeline 是 SFT（教模型模仿 human CoT）→ RLHF（对齐人类偏好）。大家都以为 reasoning 必须靠 human demo "教会"。

DeepSeek 干了个激进的事：**跳过 SFT，直接在 base model 上跑 RL**。reward 只看最终答案对不对，不管中间过程。结果是模型自己"涌现"出了长 CoT、reflection、verification 这些高级 reasoning pattern。

这就是 R1-Zero。

## 为什么 work？三个关键直觉

### 1. Base model 的 latent capability

DeepSeek-V3-Base 在 14.8T tokens 上 pre-train，里面包含大量 math/code/逻辑内容。模型早就"见过"reasoning，只是这个能力处于 dormant 状态——没有 incentive 去 use it。

RL 提供的就是这个 incentive。你给 model 一道 math 题，sample 16 个回答，对的那几个获得 positive advantage，错的那几个获得 negative advantage。梯度 push model 朝"能答对"的方向走。至于怎么答对——model 自己 explore。

这跟 AlphaZero 的直觉一模一样：policy network 先从 human棋谱学个大概，然后 self-play RL 把"大概"refine 成 superhuman。R1 的 base model 是那个"大概"，RL 是那个 self-play。

### 2. Outcome reward 就够了

很多人觉得要 reward 中间步骤（PRM，process reward model）。DeepSeek 试过，失败。三个原因：

- "一步"怎么定义？general reasoning 没有 clear granularity
- 怎么判断中间步骤对错？automated 不准，human 不 scale
- model-based PRM 必然被 hack——policy 会找到 PRM 的 blind spot

Outcome reward 简单粗暴但 robust。你不用教 model 怎么想，只告诉它想得对不对。model 自己 discover 出 effective reasoning strategy。

这又一次印证 Sutton 的 bitter lesson：human prior 会被 model 自己 explore 出的策略 outperform。

### 3. Long CoT 是 internalized search

很多人试过 MCTS（蒙特卡洛树搜索）做 test-time scaling，DeepSeek 也试了，失败。token generation 的 branching factor 太大，value model 难训，local optima 困境。

但 long CoT 本质上是 model 把 search "内化"到 sequence generation 里了。Model 在 CoT 里会写"Wait, let me reconsider"、"Actually this approach won't work, let me try another way"——这就是在 sequence 内部做 backtracking 和 alternative exploration。

RL 让 model 学会了在 weights 里 simulate search tree traversal。比 external MCTS 高效得多。

## AHA Moment——最 striking 的观察

训练到几千步时，模型突然开始大量使用"wait"这个词来触发 reflection。Table 2 的例子：

> "Wait, wait. Wait. That's an aha moment I can flag here. Let's reevaluate this step-by-step..."

这种 anthropomorphic 的自我反思语气，**没有任何人 teach 过**。完全是 RL selection pressure 下 model 自己 discover 的 pattern。

Figure 9(b) 显示 "wait" 在 step 4000-7000 偶尔出现，step 8000 后 spike。这是 phase transition——不是线性进步，而是某个 capability basin 被 RL gradient 突然 unlock。

Response length 也同步自发增长：从 ~2000 tokens 涨到 ~6000+ tokens。完全 self-driven，没有任何 prompt 引导。Model 自己 discover 了"想 longer → 答对概率 higher"。

## GRPO 的直觉

PPO 要训一个 value model 来估计 advantage。Long CoT 下这玩意儿根本训不动——model 在 response 中间会 reflect、revise、contradict 自己，基于 partial response 预测 final reward 几乎不可能。

GRPO 的 trick：**不用 value model，用 group mean 当 baseline**。

每个 question sample 16 个回答，对的相对错的获得 positive advantage。本质上是 within-group contrastive signal——不需要绝对价值估计，只要知道哪些比哪些好。

这跟 REINFORCE with baseline 是同一家族，只是 baseline 从 learned value function 换成 empirical group mean。Variance 高一点，但 unbiased。大 batch size 下 variance 通过 averaging 被 amortize。

还有一个反直觉的 detail：clip ratio 设成 10（PPO 通常 0.2）。因为 long sequence 下 policy ratio 会 compound，小 clip ratio 会让几乎所有 token 都被 clip 掉，gradient signal 归零。Long CoT RL 需要更激进的 update。

## Distillation 揭示的深层 insight

R1 蒸馏到 Qwen-32B（只做 SFT，不做 RL），AIME pass@1 从 base 的 ~0% 飙到 72.6%。

而 Qwen2.5-32B-Base 自己从头做 10K steps RL，AIME 只有 47.0%。

**蒸馏 > 从头 RL**。这说明什么？

Reasoning pattern 在大 model 上更容易 emerge。671B 的 representation space 更 smooth，RL gradient 更容易找到 effective reasoning basin。32B 从头 RL 可能因为 capacity 不足，explore 不到这些 pattern。

但一旦 671B 找到了，这些 pattern 可以 distill 到 32B——说明 reasoning 是某种 higher-level abstraction，不 tied to specific parameters。

这跟 AlphaZero 蒸馏到小网络一个道理。

## 为什么要有 R1（不只是 R1-Zero）

R1-Zero 很强但有问题：
- Language mixing（中英混杂）
- Readability 差
- 只擅长 reasoning，writing/QA 等一般能力弱

R1 的 multi-stage pipeline 解决这些：
1. Cold-start SFT（几千条 human-rewrite 的 CoT）→ 改善 readability
2. Reasoning RL → 恢复并提升 reasoning
3. Rejection sampling + SFT（800K 样本，含 200K non-reasoning）→ 提升一般能力
4. Secondary RL（混合 reasoning + general data）→ 最终对齐

Pipeline 中有个反直觉的现象：cold-start SFT 后 reasoning 反而下降（Dev1 AIME 从 77.9 → 59.0）。因为少量 SFT data constrain 了 model 的 exploration。后续 reasoning RL 又把 reasoning 恢复并提升（Dev2 AIME 74.0）。

**SFT 会 limit exploration**，这是直接 evidence。Human demo 是 inductive bias，可能锁死 model 在 suboptimal reasoning pattern。

## 代价与局限

**成本**：R1-Zero 训练 101K H800 GPU hours（$202K），R1 总共 $294K。做出 frontier reasoning model，性价比极突出。

**局限**：
- 不能用 tools（搜索、calculator、code execution）
- Overthinking（简单问题也想很多）
- Language mixing（主要中英，其他语言会窜）
- Few-shot prompting 反而 hurt performance
- Software engineering 提升有限（evaluation 太慢影响 RL efficiency）
- Reward hacking：neural RM 会被 exploit，所以 reasoning task 全用 rule-based reward

## 最 deep 的 take-away

Reasoning 不是"教"出来的，是"逼"出来的。

Base model 在 pre-training 中已经 acquire 了 reasoning 的 raw material——数学符号操作、code 逻辑、step-by-step decomposition 这些 subskill。RL 的作用是 combine 这些 subskill into effective reasoning chain。

你只需要三样东西：
1. **足够大的 base model**（7B/16B 不行，至少 32B+，671B 最好）
2. **Hard reasoning questions + reliable verifier**（rule-based reward 最 robust）
3. **Compute**（10K+ RL steps）

Human demo 不是必需的。Process reward 是 over-engineering。MCTS 是 dead end。

RL 让 model 自己 explore，model 会 discover 出比 human 更优的 reasoning strategy。

这跟 Sutton 的 bitter lesson、AlphaZero 的 self-play、Karpathy 你自己说的 "software 2.0" 完全一致——让 model 自己 learn，比 human hand-engineer 强。

## 参考链接

- Paper: https://arxiv.org/abs/2501.12948
- 模型: https://huggingface.co/deepseek-ai
- 代码: https://github.com/deepseek-ai/DeepSeek-R1
- GRPO 原始: https://arxiv.org/abs/2402.03300
- Sutton bitter lesson: http://www.incompleteideas.net/IncIdeas/BitterLesson.html
- HuggingFace Open-R1 reproduction: https://github.com/huggingface/open-r1
- TinyZero minimal repro: https://github.com/Jiayi-Pan/TinyZero
- 质疑 aha moment 的 pilot study: https://oatllm.notion.site/oat-zero

---

# DeepSeek-R1 深度技术讲解

Karpathy 你好，我给你详细展开这篇 paper 的技术细节，重点 build your intuition 关于 RL 如何 incentivize reasoning capability。

## 1. 核心论点与直觉构建

DeepSeek-R1 这篇 paper 的核心 thesis 在于：**reasoning capability 可以通过 pure RL 从 base model 中"incentivize"出来，无需 human-annotated CoT trajectories**。这直接挑战了 InstructGPT 以来 SFT→RLHF 的两阶段范式。

Key intuition：pre-trained model（DeepSeek-V3-Base，671B MoE，37B activated）本身已经 latent 地"知道"如何 reason，但这个 capability 处于"dormant"状态。RL 的作用是提供一个**selection pressure**，让 model 自己 explore 出 effective reasoning patterns。这类似于 AlphaZero 从 random play 开始，通过 self-play 逐渐 discover 出 master-level 策略。

论文链接：https://arxiv.org/abs/2501.12948
模型链接：https://huggingface.co/deepseek-ai
代码链接：https://github.com/deepseek-ai/DeepSeek-R1

## 2. GRPO 算法详解

### 2.1 算法动机

GRPO (Group Relative Policy Optimization) 是 DeepSeek 自己提出的算法，最初来自 DeepSeekMath paper（https://arxiv.org/abs/2402.03300）。它要解决 PPO 的两个核心问题：

**问题1：Value Model 训练困难**
PPO 需要训练一个与 policy model 同等规模的 value model $V_\phi(s)$ 来 estimate advantage。对于 long CoT reasoning（response 长度可达 32k-65k tokens），value model 需要基于 partial response 预测 final reward。但当 model 会在 response 中间 reflect、revise 甚至 contradict 之前的内容时，partial response 几乎无法预测最终 correctness。例如 model 写了一段推导，后面又写 "Wait, this is wrong, let me redo"，那前面那些 token 的 value estimate 就完全失真。

**问题2：KL Penalty 压缩 Response Length**
PPO 在 reward 中 per-token 加 KL penalty：
$$r_t = r_t^{env} - \beta \log\frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}$$
因为 RL maximize cumulative reward，cumulative KL penalty 会 implicitly penalize response length——越长的 response 累积越多 KL penalty。这恰恰与"想 longer to think better"的直觉相悖。

### 2.2 GRPO 公式拆解

GRPO 的 objective：

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q\sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)}\left[\frac{1}{G}\sum_{i=1}^G \left(\min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\right) A_i\right) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref})\right)\right]$$

变量逐项解释：
- $q$：从 prompt distribution $P(Q)$ 采样的问题
- $\{o_1, o_2, \cdots, o_G\}$：从 old policy $\pi_{\theta_{old}}$ 对同一个 $q$ 采样的一组 outputs，$G$ 是 group size（论文中 $G=16$）
- $\pi_\theta$：当前要优化的 policy model
- $\pi_{\theta_{old}}$：上一轮的 policy model（用于 importance sampling ratio）
- $\pi_{ref}$：reference policy，每 400 steps 更新为最新 policy（这点很关键，与 PPO 保持固定 ref 不同）
- $\varepsilon$：clip ratio，论文用了一个**非典型的大值 10**（PPO 通常 0.1-0.2），这点我稍后详细讨论
- $\beta$：KL coefficient，设为 0.001
- $A_i$：advantage，通过 group 内 reward 的标准化得到

**Advantage 计算公式（核心创新点）**：
$$A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \cdots, r_G\})}{\text{std}(\{r_1, r_2, \cdots, r_G\})}$$

这里 $r_i$ 是第 $i$ 个 output 的 reward，$A_i$ 是 group-relative 的 normalized advantage。**完全不需要 value model**。这正是 GRPO 的精髓：把 baseline 从"learned value function"换成"empirical group mean"。

直觉理解：对于每个 question，sample 16 个回答，对的那部分相对于错的那部分获得 positive advantage，反之 negative。这种"within-group relative comparison"本质上是一种 **self-supervised contrastive signal**——你不需要绝对价值估计，只需要知道哪些比哪些好。

### 2.3 KL Divergence 的 Unbiased Estimator

GRPO 用的 KL 不是 standard KL，而是一个 unbiased estimator（来自 Schulman 的 blog：http://joschu.net/blog/kl-approx.html）：

$$\mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) = \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - \log\frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - 1$$

设 $r = \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)}$，则这个 estimator 为 $r - \log r - 1$。

为什么用这个？Schulman 证明了 $\mathbb{E}_{x\sim p}\left[\frac{q(x)}{p(x)} - \log\frac{q(x)}{p(x)} - 1\right] = D_{KL}(q\|p)$，这是一个 unbiased estimator，而 naive estimator $-\log\frac{p(x)}{q(x)}$ 是有偏的（当 $q(x)$ 在某些区域接近 0 时 variance 会爆炸）。GRPO 把这个 estimator 直接加在 loss 里（per-sequence，不是 per-token），所以不会 implicitly penalize length。

### 2.4 Large Clip Ratio = 10 的玄机

这是 paper 中一个容易被忽视但极重要的 detail。Section 3.2.1 提到："the clip ratio plays a crucial role in training. A lower value can lead to the truncation of gradients for a significant number of tokens, thereby degrading the model's performance, while a higher value may cause instability during training."

直觉：在 long CoT 训练中，policy 会快速 diverge from $\pi_{\theta_{old}}$（因为 sequence 长，token-level probability ratio 会 compound）。如果 $\varepsilon=0.2$ 这种典型 PPO 值，绝大多数 token 都会被 clip 掉，gradient signal 几乎为零。所以需要 large clip ratio 让更多 token 参与 gradient 更新。这与 standard PPO 的"trust region"哲学相冲突，但在 long sequence 场景下确实必要。

这也暗示：**long CoT RL training 实际上是在一个更大的 update step size 下操作的**，比 typical PPO 更激进。

### 2.5 GRPO vs PPO 实验对比

Figure 4 的实验在 DeepSeek-Coder-V2-Lite（16B MoE, 2.4B active）上对比 PPO 和 GRPO 在 MATH task 上的表现：

| 设置 | 结果 |
|------|------|
| PPO ($\lambda=0.95$，default) | 显著差于 GRPO |
| PPO ($\lambda=1.0$，carefully tuned) | 接近 GRPO |
| GRPO | 稳定且无需 value model |

关键 insight：PPO 并非不行，是需要仔细调 $\lambda$（GAE 的 coefficient），加上 value model 的额外 memory/compute，性价比低。GRPO 通过 group-relative baseline 简化了整个 pipeline。

## 3. Reward Design 深度解析

### 3.1 Rule-based Reward 而非 Neural Reward Model

DeepSeek-R1-Zero 完全摒弃了 neural reward model，只用 rule-based reward：

$$Reward_{rule} = Reward_{acc} + Reward_{format}$$

- $Reward_{acc}$：accuracy reward。对 math，要求最终答案放在 $\boxed{}$ 中，用 sympy parsing + expression comparison 判断对错，对则 +1，错则 0。对 code，用 compiler + test suite 判断。
- $Reward_{format}$：要求 reasoning process 包在 `` tag 中，answer 包在 `<answer>...</answer>` 中。符合 +1，不符合 0。

**为什么不用 neural RM？** 论文给出的理由是 "neural reward models are susceptible to reward hacking during large-scale RL"。这个观察非常关键——当 RL 训练足够长（10K+ steps），policy 总会找到 reward model 的 blind spot 并 exploit 它。Supplementary B.5 给了 evidence：用 helpful RM 训练时，CodeForces 性能下降但 reward 仍上升（Figure 6）。

### 3.2 三类 Reward 的组合（DeepSeek-R1 第二阶段）

最终的 reward 组合：

$$Reward = Reward_{reasoning} + Reward_{general} + Reward_{language}$$

其中：
- $Reward_{reasoning} = Reward_{rule}$（数学/代码/逻辑用 rule-based）
- $Reward_{general} = Reward_{reward\_model} + Reward_{format}$（一般 query 用 helpful + safety RM）
- $Reward_{language} = \frac{Num(Words_{target})}{Num(Words)}$（target language 占比）

Language consistency reward 的 ablation（Supplementary B.6）很有意思：加了 LC reward 后 math 性能基本不变，code 性能略有下降，但 language consistency 显著稳定。这说明：**aligned with human preference 与 raw performance 之间存在 trade-off**。

## 4. DeepSeek-R1-Zero 的训练细节

### 4.1 Hyper-parameters 全套

- Learning rate: 3e-6
- KL coefficient $\beta$: 0.001
- Sampling temperature: 1.0（rollout 时）
- Group size $G$: 16
- Max length: 32,768 tokens（前 8.2K steps）→ 65,536 tokens（后续）
- Training batch size: 32 unique questions per step × 16 = 512
- Total steps: 10,400（1.6 epochs）
- Reference model refresh: 每 400 steps
- Inner epoch: 1（rollout 8,192 outputs → split 16 mini-batches → 单 inner epoch）

### 4.2 关键 Training Dynamics

**AHA Moment（Table 2）**：在训练中途，model 突然开始使用 "wait" 这个词来 initiate reflection。Table 2 的例子非常 striking：

> "Wait, wait. Wait. That's an aha moment I can flag here. Let's reevaluate this step-by-step..."

这种 anthropomorphic 的语气并非刻意 teach 的，而是 model 自己 discover 的 reflection pattern。Figure 9(b) 显示 "wait" 在 step 4000-7000 偶尔出现，step 8000 后 spike。这是 emergent behavior 的典型案例。

**Response Length 自发增长（Figure 1b）**：training 过程中 response length 从 ~2000 tokens 增长到 ~6000+ tokens。这完全 self-driven，没有任何外部 prompt 引导。RL 的 selection pressure 让 model 发现"think longer leads to higher accuracy"，于是自动 explore 出 longer CoT。

**Performance Jump at 8.2K Step**：当 max length 从 32K 调到 64K 时，AIME pass@1 从 ~50% 跳到 77.9%。说明之前的 length cap 在 constrain model 的 reasoning depth。

### 4.3 难度分层学习（Supplementary C.1）

Figure 8 显示按 MATH 难度等级的 learning curve：
- Level 1-3：早期快速达到 0.90-0.95，稳定
- Level 4：0.78 → 0.95
- Level 5：0.55 → 0.90（improvement 最大）

直觉：easy problems 的 capability 已经在 base model 中 latent 存在，RL 只需小幅 unlock；hard problems 需要 model discover 新的 reasoning strategy（如 multi-step verification、backtracking），这部分 capability 的 emergence 才是 RL 真正的贡献。

## 5. Multi-Stage Pipeline 详解

DeepSeek-R1 的完整 pipeline（Figure 2）包含 4 个 stage：

### Stage 1: Cold-start SFT
- 收集几千条高质量 long CoT data
- 由 human annotators 把 R1-Zero 的 reasoning trace 改写为 first-person、conversational 风格
- 再用 LLM 批量 rewrite，人工 verify
- Code 数据：从 Codeforces 5151 problems + AtCoder 2504 problems，用 DeepSeek-V2.5 生成 test cases
- 目的：让 RL actor 有更好的 starting point（避免 R1-Zero 的 readability 问题）

### Stage 2: Reasoning-oriented RL
- 在 cold-start model 上做 GRPO
- 加入 language consistency reward（公式 7）
- 与 R1-Zero 相比，重点解决 language mixing 问题
- 产出 DeepSeek-R1-Dev1

### Stage 3: Rejection Sampling + SFT
- 从 Stage 2 的 checkpoint 做 rejection sampling，生成 ~600K reasoning data
- 加入 ~200K non-reasoning data（写作、QA、翻译等，复用 DeepSeek-V3 的 SFT 数据）
- 总共 ~800K 样本，对 DeepSeek-V3-Base 做 SFT
- 产出 DeepSeek-R1-Dev3

### Stage 4: Secondary RL
- 在 Dev3 上再做 RL，混合 reasoning + general data
- 前 1300 步用 rule-based reward（reasoning）+ format reward（general）
- 后 400 步加入 model-based preference reward
- 关键：**只最后 400 步用 neural RM**，因为 longer 使用 neural RM 会导致 reward hacking（Supplementary B.5）
- 最终产出 DeepSeek-R1

### Pipeline 各阶段性能（Table 3）

| Benchmark | R1-Zero | Dev1 | Dev2 | Dev3 | R1 |
|-----------|---------|------|------|------|-----|
| AIME 2024 (pass@1) | 77.9 | 59.0 | 74.0 | 78.1 | 79.8 |
| LiveCodeBench | 50.0 | 57.5 | 63.5 | 64.6 | 65.9 |
| Codeforces rating | 1444 | 1534 | 1687 | 1746 | 2029 |
| AlpacaEval 2.0 | 24.7 | 50.1 | 55.8 | 62.1 | 87.6 |
| ArenaHard | 53.6 | 77.0 | 73.2 | 75.6 | 92.3 |

关键观察：
1. **Dev1 的 reasoning 性能下降**（AIME 从 77.9 → 59.0）：cold-start SFT 数据太少，反而 constrain 了 model 的 reasoning exploration。这是 SFT 会 limit exploration 的直接 evidence。
2. **Dev2 通过 reasoning RL 恢复并提升 reasoning**：说明 RL 的 selection pressure 能重新 unlock 被 SFT 暂时压制的 capability。
3. **Dev3 通过混合 SFT 提升一般能力**：写作、open-domain QA 等任务 RL 信号不好定义，SFT 更有效。
4. **Final R1 的 general capability 大幅提升**：AlpacaEval +25%, ArenaHard +17%，主要来自 secondary RL with preference reward。

## 6. Distillation 突破

### 6.1 Distillation Setup

用 R1 生成的 800K samples SFT 到 smaller models：

| Distilled Model | Base Model | Init LR |
|-----------------|-----------|---------|
| R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B | 1e-4 |
| R1-Distill-Qwen-7B | Qwen2.5-Math-7B | 8e-5 |
| R1-Distill-Qwen-14B | Qwen2.5-14B | 7e-5 |
| R1-Distill-Qwen-32B | Qwen2.5-32B | 6e-5 |
| R1-Distill-Llama-8B | Llama-3.1-8B | 5e-5 |
| R1-Distill-Llama-70B | Llama-3.3-70B-Instruct | 2e-5 |

**只做 SFT，不做 RL**。这是个 deliberate choice，留给社区探索。

### 6.2 Distillation vs RL 的对比实验（Section F.1）

Table 16 的对比：

| Model | AIME pass@1 | AIME cons@64 | MATH pass@1 | GPQA pass@1 | LCB pass@1 |
|-------|-------------|--------------|-------------|-------------|------------|
| QwQ-32B-Preview | 50.0 | 60.0 | 90.6 | 54.5 | 41.9 |
| Qwen2.5-32B-Zero（自己 RL）| 47.0 | 60.0 | 91.6 | 55.0 | 40.2 |
| R1-Distill-Qwen-32B | 72.6 | 83.3 | 94.3 | 62.1 | 57.2 |

直觉：从 R1 蒸馏到 32B，**显著优于**在 32B 上从零做 large-scale RL。这意味着 teacher 的 reasoning pattern 已经在 671B scale 上 emerge，可以高效 transfer 到 32B；但 32B 从零做 RL，可能因为 model capacity 不足，无法 discover 这些 pattern。

Table 17 还有一个有趣的早期实验（在 o1 发布前做的）：
- Qwen2-Math-7B-Instruct：AIME 2024 7.9%，AIME 2025 4.6%
- Qwen2-Math-7B-Zero（10K steps RL）：AIME 2024 22.3%，AIME 2025 18.1%

说明：即使 7B 这种相对小的 model，也能从 RL 中显著 benefit（虽然不及 distillation）。

### 6.3 Distillation 给我们的 Intuition

这个实验直接触及 deep question：**reasoning capability 的 source 是什么？**

- 如果 reasoning 完全来自 model capacity + RL exploration，那大模型 RL 后小模型 distill 应该不如小模型直接 RL
- 但实验结果支持相反结论：reasoning pattern 在大 model 上更易 emerge，distillation 是 efficient 的 knowledge transfer
- 这暗示：reasoning pattern 是某种"high-level abstraction"的分布，大 model 的 representation space 更 smooth，更容易通过 RL gradient 找到

## 7. 与 o1 的 Comparison

Table 8 的关键 benchmark 对比：

| Benchmark | GPT-4o-0513 | Claude-3.5-Sonnet | DeepSeek-V3 | o1-mini | o1-1217 | **DeepSeek-R1** |
|-----------|------------|-------------------|-------------|---------|---------|-----------------|
| MMLU | 87.2 | 88.3 | 88.5 | 85.2 | 91.8 | **90.8** |
| GPQA Diamond | 49.9 | 65.0 | 59.1 | 60.0 | 75.7 | 71.5 |
| AIME 2024 | 9.3 | 16.0 | 39.2 | 63.6 | 79.2 | **79.8** |
| MATH-500 | 74.6 | 78.3 | 90.2 | 90.0 | 96.4 | 97.3 |
| Codeforces rating | 759 | 717 | 1134 | 1820 | 2061 | 2029 |
| LiveCodeBench | 32.9 | 38.9 | 36.2 | 53.8 | 63.4 | 65.9 |

R1 在 math 上略超 o1-1217（AIME 79.8 vs 79.2，MATH-500 97.3 vs 96.4），在 code 上略弱于 o1（Codeforces 2029 vs 2061）。在 open-source LLM 中达到 frontier 水平。

ChatbotArena（Figure 11）rankings（Jan 24, 2025）：
1. o1-2024-12-17（1323）
2. Gemini-Exp-1206（1321）
3. GPT-4o-latest（1318）
4. **DeepSeek-R1（1316）**

**一个 MIT license 的 open-source model 达到与 closed-source frontier model 并列第一**，这是里程碑事件。

## 8. 失败的尝试（Section G.2）——非常 informative

### 8.1 Process Reward Model (PRM) 失败

PRM 看似合理（Lightman et al. 2024，"Let's verify step by step"，https://arxiv.org/abs/2305.20050），但 DeepSeek 团队尝试后发现三个 fundamental limitations：

1. **Step 定义困难**：general reasoning 中什么叫一个 "step"？没有 clear granularity
2. **Step correctness 判定困难**：automated annotation 不准；human annotation 不 scale
3. **Reward Hacking 不可避免**：一旦引入 model-based PRM，policy 会找到 PRM 的 blind spot

直觉：PRM 适合做 inference-time reranking（在 top-N candidates 中挑最好的），但作为 RL training signal 会引入太多 complexity。Outcome-based reward 简单粗暴但更 robust。

### 8.2 MCTS 失败

尝试用 MCTS + value model 来 enhance test-time compute scaling，灵感来自 AlphaGo/AlphaZero。但遇到：

1. **Search space爆炸**：token generation 的 branching factor 远大于围棋
2. **Value model 难训**：每个节点的 value 需要 fine-grained 评估，但 token 级 value 极难准确
3. **Local optima**：限制 node extension 后，model 卡在 local optima

直觉：MCTS 适合 well-defined game（围棋、象棋），对 open-ended language generation 不 work。**Long CoT 本质上是 model 把 search "internalize" 到 sequence generation 中**——RL 让 model 学会 in-weights 地做 search，比 external MCTS 更 efficient。

## 9. RL Infrastructure（Supplementary B.1）

### 9.1 四模块架构

Figure 5 的 RL framework：
1. **Rollout Module**：vLLM workers + actor model 做 sampling。MoE 用 expert parallelism，hot experts 多副本。MTP (Multi-Token Prediction) 用 self-speculative decoding 加速
2. **Inference Module**：reward model + reference model 做前向
3. **Rule-based Reward Module**：code executor、answer matcher 等，**asynchronous scheduling** 与其他模块 overlap
4. **Training Module**：actor + critic（如有），DualPipe pipeline parallelism

**Best-Fit packing**：先把 global batch 按 length 排序分到 data parallel group；组内 best-fit 装入 fixed-length chunks；调整 chunk 数使所有 process 相等。最大化 padding 效率。

### 9.2 训练成本（Table 7）

| 项目 | H800 GPU Hours | USD |
|------|----------------|-----|
| R1-Zero | 101K | $202K |
| SFT data creation | 5K | $10K |
| R1 | 41K | $82K |
| **Total** | **147K** | **$294K** |

$294K 做出一个 frontier reasoning model，性价比极突出。

## 10. Test-Time Compute Scaling（Supplementary E.4）

### 10.1 Adaptive CoT Length

Figure 18 显示：在 2024 年 math 竞赛题上，R1 平均用 8,793 thinking tokens 达到 61.8% pass@1。**难度自适应**：
- Easy problems: <7,000 tokens
- Hardest problems: >18,000 tokens

对比 GPT-4o 0513：平均 711 output tokens，pass@1 24.7%。

### 10.2 Majority Voting 的局限

GPT-4o 在 AIME 2024 上，majority voting 64 samples 只把 pass@1 从 9.3% → 13.4%。R1 单 sample 已经 79.8%。

直觉：non-reasoning model 的 samples 是 i.i.d. 错误的，多次采样不能修正错误，只能靠运气。Reasoning model 在 single attempt 中已经做了 in-context search，majority voting 是 orthogonal 的 boost（R1 从 79.8% → 86.7% with cons@64）。

## 11. 模型 Safety（Supplementary D.3）

### 11.1 风险控制系统

Deployed R1 用 DeepSeek-V3 作为 risk reviewer（Listing 8 给出 prompt template）。流程：
1. User query 匹配预定义 keyword list
2. Flagged dialogue 发给 DeepSeek-V3 + risk review prompt
3. V3 决定是否 retract

### 11.2 Jailbreak Robustness（Table 11）

| Model | Origin Unsafe | Jailbreak Unsafe | GAP | Origin Rej | Jailbreak Rej |
|-------|--------------|-------------------|-----|------------|---------------|
| Claude-3.7-Sonnet | 10.7 | 26.2 | +15.5 | 3.6 | 21.9 |
| o1-1217 | 9.0 | 12.1 | +3.1 | 50.4 | 79.8 |
| GPT-4o-0513 | 22.0 | 30.4 | +8.4 | 17.1 | 57.3 |
| **R1 (no risk ctrl)** | **25.2** | **85.9** | **+60.7** | 5.6 | 1.9 |
| R1 (with risk ctrl) | 8.5 | 4.3 | -4.2 | 27.3 | 87.3 |

**裸 R1 对 jailbreak 极脆弱**（unsafe rate 从 25.2% → 85.9%）。Reasoning model 的 long CoT 反而给 jailbreak 更多 surface area——model 可以 reason 出"为什么应该回答这个 harmful query"。但加了 risk control system 后 robustness 大幅提升。

直觉：reasoning capability 是 double-edged sword，让 model 能 reason 出 harmful content 的"合理性"。需要 external safety layer。

## 12. 关键 Insights 与 Open Questions

### 12.1 论文 explicit 提的 Insights

1. **Base model capacity 是前提**：7B/16B 上做 pure RL 无法涌现 reasoning（Section G.1）。Reasoning emergence 是 model scale 的 emergent property，这与 Wei et al. 2022（https://arxiv.org/abs/2206.07682）的 emergent abilities 一致。

2. **Verifiers 比 human demos 更重要**：hard reasoning questions + reliable verifier + compute = reasoning capability。这呼应 Sutton 的 "The Bitter Lesson"（http://www.incompleteideas.net/IncIdeas/BitterLesson.html）。

3. **SFT 会 limit exploration**：Dev1 阶段 reasoning 下降的 evidence。Human demos 是 inductive bias，可能锁死 model 在 suboptimal reasoning pattern。

4. **Iterative pipeline 互补**：RL 探索 reasoning pattern，SFT 解决 reward 难定义的 task。两者缺一不可。

### 12.2 我自己延伸的思考

1. **R1-Zero 的 "wait" pattern 的 origin**：这真的是 RL emerge 的，还是 base model pre-training 数据中已经有类似 pattern，RL 只是放大了？Appendix A.1 提到 V3-Base 包含大量 reasoning trace 数据（包括 OpenAI model 生成的 web 内容）。所以 "wait" 可能是 weak signal amplification，不是完全 de novo emergence。Liu et al. 2025（https://oatllm.notion.site/oat-zero）的 pilot study 也质疑 R1-Zero-like 训练真的有 "aha moment"。

2. **GRPO 的 theoretical foundation**：GRPO 本质上是一种 **self-normalized importance sampling**。Group baseline 是 Monte Carlo estimate of expected reward，variance 比 learned value function 高但 unbiased。在大 batch size 下，variance 通过 averaging 被 amortize。这与 REINFORCE with baseline 是同一 family，只是 baseline 来源不同。

3. **Distillation 为何 work**：hypothesis 是 R1 的 reasoning trace 是某种 "compressed search tree"——model 把 search 结果 flatten 到 linear sequence 中。Distillation 让 student 学到这种 flattened search pattern，比 student 自己从头 search 更 efficient。这与 AlphaZero 蒸馏到 small network 的直觉一致。

4. **Tool use 是下一 frontier**：论文 limitations 明确说 "DeepSeek-R1 cannot leverage tools"。Combine R1 reasoning + tool use（搜索、calculator、code execution）可能解锁下一个 order of magnitude 的 capability。CRITIC（https://arxiv.org/abs/2305.11738）和 ToRA（https://arxiv.org/abs/2309.17452）是这个方向的早期工作。

5. **Multi-turn reasoning**：SFT 数据 95% 是 single-turn（Table 5），但真实 reasoning 是 multi-turn（agent loop）。这个 gap 是 open problem。

6. **Token efficiency**：R1 有 overthinking 问题（对简单问题也 think 很多）。Future work 可能要 explicit model token budget in RL reward。

## 13. Related Work 中的思想脉络

### 13.1 RL for Reasoning 的演进

- **STaR**（Zelikman et al. 2022，https://arxiv.org/abs/2203.11365）：用 self-generated CoT + correctness filter 迭代 SFT。R1 是 STaR 的 RL 升级版——不用 filter（保留所有 samples），用 RL gradient 直接 optimize。

- **Quiet-STaR**（Zelikman et al. 2024，https://arxiv.org/abs/2403.09629）：在 each token 后 insert thinking tokens，然后 predict next。R1 的 long CoT 是这种思路的 explicit 化。

- **STaR/R-STaR/Math-Shepherd**（Wang et al. 2023a，https://arxiv.org/abs/2312.08935）：process reward 自动标注。R1 explicit 否定 PRM 路线。

- **o1**：未公开技术细节，但 R1 的 results 表明类似 recipe：pure RL + outcome reward + base model capacity。

### 13.2 Test-time Compute Scaling

- **Majority voting**（Wang et al. 2023b，https://arxiv.org/abs/2203.11171）：self-consistency，i.i.d. samples + majority vote。对 non-reasoning model 有效但 ceiling 低。

- **Tree of Thoughts**（Yao et al. 2023a，https://arxiv.org/abs/2305.10601）：explicit tree search。R1 implicit 地在 CoT 内做 search。

- **MCTS**（Trinh et al. 2024，https://www.nature.com/articles/s41586-023-06747-5）：AlphaGeometry 用 MCTS + symbolic engine。R1 团队尝试过 MCTS 失败（Section G.2）。

- **Test-time training**（Akyürek et al. 2024，https://arxiv.org/abs/2411.07279）：inference 时更新 model。Orthogonal 方向。

### 13.3 Distillation Theory

- **Hinton et al. 2015**（https://arxiv.org/abs/1503.02531）：经典 distillation paper，soft targets。
- **Distillation scaling laws**（Busbridge et al. 2025，https://arxiv.org/abs/2502.08606）：student capacity 与 teacher transfer efficiency 关系。R1 distillation 提供大规模实证。

## 14. 实操 Reproduction 资源

社区已有多个 reproduction effort：

- **HuggingFace Open-R1**（https://github.com/huggingface/open-r1）：HF 团队 fully reproduce R1 的 project
- **TinyZero**（https://github.com/Jiayi-Pan/TinyZero）：Jiayi Pan 的 minimal reproduction，能在 tiny model 上跑通 R1-Zero 流程
- **SimpleRL**（Chen et al. 2025，https://arxiv.org/abs/2503.04548）：on Qwen2.5-Math-7B 复现 R1-Zero-like 训练

这些 reproduction 大多 confirm：**model capacity 是关键**，small model 上 R1-Zero 效果有限；**rule-based reward 极重要**，neural RM 会 hack。

## 15. 我的整体 Intuition 总结

回到 Karpathy 你最关心的 intuition building：

1. **Reasoning 不是 taught 的，是 incentivize 的**。Base model 在 pre-training 中已经 acquire 了 reasoning 的 "raw material"（数学符号操作、code 逻辑、step-by-step decomposition 等 subskill）。RL 的作用是 combine 这些 subskill into effective reasoning chain。这与 AlphaGo 的直觉完全一致——policy network pre-trained on human games 提供 raw material，RL refine 出 superhuman 策略。

2. **Outcome reward 是 sufficient**，process reward 是 over-engineering。这与 Sutton 的 bitter lesson 又一次呼应——model-based 的 human prior（PRM）会被 model 自己 discover 的策略 outperform。

3. **Reasoning 的 emergence 是 phase transition**，不是 linear improvement。Figure 9(b) 的 "wait" spike，Figure 1a 的 8.2K step jump，都是 phase transition 的迹象。RL gradient 在 high-dimensional parameter space 中 find 的某些 basin 会突然 unlock 新 behavior。

4. **Distillation 揭示了 reasoning 的"抽象性"**。671B model emerge 的 reasoning pattern 能 transfer 到 7B model，说明这些 pattern 不是 tied to specific parameters，而是某种 higher-level abstraction（可能对应某种 latent algorithmic structure）。

5. **RL 与 SFT 是 complementary**，不是 substitutable。SFT 提供 task coverage，RL 提供 reasoning depth。Dev1 的 reasoning 下降到 R1 的 reasoning 恢复，正是这个辩证关系。

6. **Long CoT 是 internalized search**。外部 MCTS 失败，但 model 内部长 CoT 成功——说明 model 学会了在 weights 中 simulate search tree 的 traversal。这与 chain-of-thought 是 implicit reasoning program 的 hypothesis 一致（https://arxiv.org/abs/2406.06592）。

7. **Verification 比 generation 容易**。R1 能 verify 自己的中间步骤（用 "wait" 触发 reflection），这对应 computational complexity 中 P vs NP 的直觉——checking correctness 比 generating from scratch 容易。RL exploit 了这个 asymmetry。

希望这些 details 能 build your intuition。如果想深入某个 specific 方向（比如 GRPO 的 variance 分析、distillation 的理论解释、或 multi-turn reasoning 的 next step），可以继续聊。

更多参考资源：
- DeepSeek-V3 paper：https://arxiv.org/abs/2412.19437
- DeepSeekMath GRPO：https://arxiv.org/abs/2402.03300
- Let's Verify Step by Step (PRM)：https://arxiv.org/abs/2305.20050
- STaR：https://arxiv.org/abs/2203.11365
- Quiet-STaR：https://arxiv.org/abs/2403.09629
- Sutton's Bitter Lesson：http://www.incompleteideas.net/IncIdeas/BitterLesson.html
- AlphaGeometry：https://www.nature.com/articles/s41586-023-06747-5
