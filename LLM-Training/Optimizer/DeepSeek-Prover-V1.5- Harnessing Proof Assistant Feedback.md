---
source_pdf: DeepSeek-Prover-V1.5- Harnessing Proof Assistant Feedback.pdf
paper_sha256: 18b9bf9ce4a553b7c54ed4aab49f1f5a008f9b7e9915f2f164c8ef4da59559cd
processed_at: '2026-08-03T18:42:27-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 DeepSeek-Prover-V1.5

好，用最白话的语言再过一遍这篇 paper。

## 这个事儿的本质

你让一个 LLM 去证明一个数学定理，写成 Lean 代码。Lean 是个严格的 checker，要么你全对，要么你错一个字符它就报错。

这就好比你闭着眼睛走迷宫，只有走到终点裁判才会告诉你"对了"或"错了"。中间你完全不知道自己在哪。

V1 的做法是：模型直接从头到尾把整段 proof 写出来，交给 Lean 验证。错了就重来。问题是你写了一大段，中间早错了，模型自己还不知道，后面越写越偏，这个叫 compounding error。

V1.5 想了一个聪明的办法。

## Truncate-and-resume：核心招数

这个机制用一句话讲就是：**写一段，验一段，错了就切掉重来**。

具体流程：
1. 模型写一段 proof code
2. 交给 Lean 验证
3. Lean 说"第 3 行错了"——那就把第 3 行后面的全删掉
4. 把第 3 行正确的部分留着，把 Lean 当前的 state（"你现在应该要证明的目标是 X")作为注释附在代码末尾
5. 模型看着这个 "已经写到这儿 + 当前状态" 的 prompt 继续往下写
6. 重复

这个就是 paper 的灵魂。它解决了一个核心矛盾：
- 你想一次写完整个 proof（高效，不用频繁跟 Lean 通信）
- 但你又想知道中间状态（避免 compounding error）

trick 就是：还是一次写一大段，但 Lean 反馈告诉你"你这一大段里前 N 步是对的，从这里开始错了"，那前 N 步就用上了，状态也拿到了，模型基于真实状态续写，不是瞎猜。

## 跟 MCTS 怎么结合

光有 truncate-and-resume 还不够。模型可能反复在同一条路上死磕，换不同写法也只是换不同方式走错路。需要 search。

经典 MCTS (Monte-Carlo Tree Search) 是 AlphaGo 用的那套：建一棵搜索树，每个节点是一个"局面"，反复选有潜力的节点展开，模拟到结局，回传结果更新估值。

在定理证明里，"局面"就是 tactic state——Lean 编译器告诉你"现在要证的东西是 X"。每次成功应用一个 tactic，state 就变一次，树就长一层。

怎么把 MCTS 套到 LLM 的 whole-proof generation 上？truncate-and-resume 就是桥梁：
- **Selection**：从树根往下选一个值得展开的节点
- **Expansion**：把那个节点对应的 "未完成 proof + 当前 state 注释" 喂给 LLM，LLM 续写一大段
- **Truncate**：续写的部分交给 Lean 验证，错了就切到错误处，正确部分 parse 成一串新 tactic（每个 tactic 一个新节点），接到选中节点下面
- **Backprop**：沿选择路径回传 reward，更新各节点的 Q 值

一个关键点：**一次 expansion 可能一下子加好几个节点**（因为 LLM 一次写了一串 tactic）。这跟围棋一次下一个子不一样。这叫 "bulk expansion"。

## 节点身份问题

Lean 里有个现象：不同的 tactic 可能导致相同的 tactic state。比如 `simp` 和 `auto` 可能都把目标 `x + 0 = x` 变成 `True`。

如果按 tactic 文本来分节点，你会有一堆冗余节点（不同 tactic，同一结果）。

他们用 **tactic state 作为节点身份**。同一个 state 对应的不同 tactic 写法存在一个 set 里，展开时随机选一个当 prompt。这相当于做了状态归一化。

## RMaxTS：好奇心驱动探索

定理证明有个要命的问题：**reward 极度稀疏**。只有整个证明写完才给 1，否则就是 0。中途你不知道自己离成功还有多远。这在 RL 文献里叫 "hard exploration problem"——narrow set of leaves delivering non-zero rewards。

经典 MCTS 靠 UCB 公式做 exploration-exploitation 权衡。但 UCB 需要 reward 信号——如果所有 reward 都是 0，UCB 的 exploitation 项全一样，只剩纯 count-based exploration，效果跟随机采样差不多。

他们的解法：**给探索本身发奖**。

公式 (3)：
$$R_{\text{intrinsic}}(\tau) = \mathbb{I}[\text{at least one new node is added to the search tree}]$$

翻译成人话：**你这次 expansion 如果产生了一个之前没见过的 tactic state，给你 1 分；如果全是见过的 state，0 分**。

这就是 RMax 算法（2002 年的老算法）的核心思想——把"未知状态"当作最大奖励。逼着 agent 去覆盖整个 state space，而不是死磕几条已知路径。

在定理证明里这特别合理，因为难点根本不是 exploit——你已经证完了就完了。难点是找到那条路。所以纯探索反而对路子。

## 为什么用 Discounted UCB

RMax 的 intrinsic reward 有个问题：**非平稳**。

刚开始建树，到处都是新 state，intrinsic reward 期望很高。随着树长大，发现新 state 越来越难，intrinsic reward 期望下降。

经典 UCB1 假设 reward 是 stationary 的，会平均所有历史 reward。结果就是：早期发现的 node 累积了一堆 1-reward，平均下来显得特别高，UCB 一直认为这些 node "很有价值"，持续往那儿跑——但其实那些是旧 reward，现在的真实期望已经低了。

他们的解法：**用 discounted UCB**，公式 (7)：

$$Q_{DUCB}(s, a) = \frac{W_\gamma(s, a)}{N_\gamma(s, a)} + \text{UCB bonus}$$

其中 $W_\gamma$ 和 $N_\gamma$ 都是 discounted 版本，最近的访问权重高，老的访问权重指数衰减。$\gamma = 0.99$，大约对应"70 次访问前的 reward 衰减到一半"。

直观上：这就是 EWMA（指数加权移动平均）套到 UCB 上。让 value 估计跟踪当前的 reward 期望，而不是被历史污染。

## 整套 pipeline

总结一下整个 pipeline：

1. **Pre-train**：在 DeepSeekMath-Base 7B 上继续预训练，加 Lean / Isabelle / Metamath 数据。

2. **SFT**：做两件事——
   - 用 DeepSeek-Coder V2 236B 给 Lean 代码加自然语言 CoT 注释（"这段在证什么、思路是什么"），让模型学会先想后写
   - 在每个 tactic 后插入 tactic state 注释，训练模型"看见未完成 proof + 当前 state → 续写"

3. **RL（GRPO）**：Lean verifier 给 0/1 reward，用 GRPO 做 RL。关键 trick 是选 SFT model 中等成功率的问题——保证每组采样里既有对的也有错的，这样 group-relative advantage 才有信号。

4. **Inference**：要么单次采样，要么 RMaxTS 搜索。

## 为什么 RL 在 formal proving 里特别有用

有个有趣观察：在自然语言数学（DeepSeekMath）里，RL 主要 lift TopK——多次采样里最好那次更可能对，但单次成功率提升有限。

在 formal theorem proving 里，RL **真的提升了 base capability**——哪怕只采样一次，正确率也涨了。

我猜原因是：Lean verifier 是绝对准确的 ground truth，没有 reward model 的噪声，RL 学到的是"形式化正确性"本身，不是"匹配某个有偏好的 reward model"。这种 setting 下 RL 能真正优化模型能力，而不是只优化 sampling 分布。

## Mixture Strategy 的小 trick

CoT 模式（先想后写）和 non-CoT 模式（直接写 tactic）各有优势：
- CoT 适合需要数学推理规划的问题
- non-CoT 适合可以用 Lean 高级 tactic（如 `nlinarith`, `field_simp`）暴力解决的计算题

简单做法：sample budget 各分一半给两种模式。结果涨了 1.3%。

这说明 prompt template 是个 latent variable，激活不同的证明策略先验。未来可以做 per-problem adaptive routing，但 paper 没做。

## 关键实验数字

miniF2F-test（高中竞赛级）：
- GPT-4 via COPRA：26.6%
- DeepSeek-Prover-V1：50.0%
- DeepSeek-Prover-V1.5-RL（单次采样 128 次）：51.6%
- DeepSeek-Prover-V1.5-RL + RMaxTS（32×6400 mixture）：**63.5%** (SOTA)

ProofNet-test（本科级）：
- ReProver：13.8%
- InternLM2-StepProver：18.1%
- DeepSeek-Prover-V1.5-RL + RMaxTS：**25.3%** (SOTA)

计算效率方面：InternLM2-StepProver 用 64×32×100 = 204800 次搜索达到 54.5%；DeepSeek-Prover-V1.5 + RMaxTS 只用 3200 次就到 55%，效率高 60 倍以上。

## 消融实验讲清楚什么

去掉 intrinsic reward（只剩 UCB 探索）：退化到接近单次采样水平。**说明光有 UCB 在零 reward 环境下没用，必须有 intrinsic reward 提供探索信号**。

把 DUCB 换成 UCB1：退化。**说明处理 non-stationarity 关键，否则旧 reward 污染新估计**。

去掉 tactic state 注释：退化。**说明给模型看中间 state 真的能提升长程规划能力，不是可有可无的装饰**。

## 跟其他方法比

之前主流两条路：
- **Proof-step generation** (GPT-f, ReProver, Thor)：每次只写一个 tactic，靠 Lean 拿 state。慢但稳。
- **Whole-proof generation** (DeepSeek-Prover-V1, Lyra)：一次写完。快但容易 compounding error。

DeepSeek-Prover-V1.5 是 hybrid：还是一次写一大段（保持 whole-proof 的高效），但通过 truncate-and-resume 重新引入了 step-level state（解决 compounding error）。**一个模型同时支持两种范式**。

## 我觉得最重要的几点 intuition

**(1) Truncate-and-resume 是 LLM generation 与 tree search 之间的桥梁**

LLM 的输出是连续 token 序列，而 MCTS 需要离散的 state-action。这个机制把连续生成切片成离散 steps，每片对应一个 tactic 和一个 state。这跟 Tree of Thoughts (Yao et al. 2023) 的思路一致——都是"把 LLM generation 重新映射到树结构"。

这个 abstraction 应该可以推广到任何有 incremental verifier 的领域：SQL 生成 + parser feedback、Rust borrow checker、formal hardware verification。只要 verifier 能告诉你"这一步对了、下一步错了"，你就能 truncate-and-resume。

**(2) RMax 在 LLM reasoning 领域的回归很有意思**

RMax 是 2002 年的老算法，理论漂亮但 sample inefficient。在定理证明这种 state space 巨大但 verifier 可枚举（Lean tactic state 是结构化文本，可 hash 比较）的环境里，"新 state reward 1" 这个简单 indicator 反而 work。因为探索本身就是目标——证明的难点不在 exploit 而在找路。

对比 Pathak 的 ICM (curiosity module) 或 Burda 的 RND——那些需要训练额外预测网络，在 continuous state space（图像）里 work。这里 state 是离散结构化的，RMax 直接够用。

**(3) RLPAF ≈ AlphaGo 的 self-play**

精神上和 AlphaGo 的 self-play 是一致的：
- AlphaGo：policy network 下棋 + terminal reward (胜负) + MCTS planning
- DeepSeek-Prover-V1.5：LLM 生成 proof + terminal reward (Lean 验证) + RMaxTS planning

但 DeepSeek 少了一个关键组件：**value network**。AlphaGo 有 value network 评估 partial position，AlphaZero 靠 value network 引导 search。DeepSeek 完全靠 intrinsic reward + UCB 探索，没有 exploitation 信号评估 partial proof。

这就是 paper 在 Conclusion 里承认的最大局限。如果下一版加上 partial-proof critic（评估 incomplete proof 的完成概率），把这个 value 作 Q 的 prior，应该能再涨一截。这相当于 AlphaGo 的 rollout policy + value network 联合，但 V1.5 只有 rollout 没有 value。

**(4) Verifier 准不准决定 RL 能不能真正提升 capability**

Lean 是绝对 ground truth，没有噪声。RL 优化的就是"形式化正确性"本身，所以能提升 base capability。

对比 NL 数学：RL 的 verifier（rule-based 或 model-based）有噪声，RL 主要学到"匹配 verifier 偏好"，提升的是 TopK 而不是 base capability。这个差异挺重要，暗示着：**verifier 越准，RL 越能真正提升模型能力，而不只是采样效率**。

**(5) CoT 在 formal proving 里是真的有用**

CoT 在 NL 数学里争议很大——有时候模型先输出推理再回答，不一定真用上了推理。但在 formal proving 里，CoT 直接作为 comment 写在 Lean 代码里，模型必须先生成数学推理再生成对应的 tactic。这种"思考-行动"交错，比单纯让模型一口气写 tactic 更稳。实验也证实 CoT 模式稳定优于 non-CoT。

这跟 Lean-STaR (Lin et al. 2024) 思路一样，但 DeepSeek 的实现更巧妙——直接用 236B 大模型 annotate 既有数据，质量比 Lean-STaR 自己生成的高。

## Reference 链接

主 paper 相关：
- DeepSeek-Prover-V1.5 GitHub: https://github.com/deepseek-ai/DeepSeek-Prover-V1.5
- DeepSeek-Prover-V1 (前置): https://arxiv.org/abs/2405.14333
- DeepSeekMath (GRPO 算法来源): https://arxiv.org/abs/2402.03300
- DeepSeek-Coder V2 (用于 annotate): https://arxiv.org/abs/2406.11931

Benchmark:
- miniF2F: https://github.com/openai/miniF2F
- ProofNet: https://arxiv.org/abs/2302.12433
- Lean 4: https://lean-lang.org/

对比方法:
- GPT-f: https://arxiv.org/abs/2009.03393
- LeanDojo (ReProver): https://leandojo.org/
- HTPS (Hypertree Proof Search): https://arxiv.org/abs/2205.11491
- Lean-STaR: https://arxiv.org/abs/2407.10040
- InternLM2-StepProver: https://arxiv.org/abs/2407.17227
- COPRA: https://arxiv.org/abs/2310.04653
- Tree of Thoughts: https://arxiv.org/abs/2305.10601

算法基础:
- MCTS survey (Browne 2012): https://ieeexplore.ieee.org/document/6145622
- UCB1 (Auer 2002): https://link.springer.com/article/10.1023/A:1013689704352
- DUCB (Garivier 2011): https://arxiv.org/abs/0807.0889
- RMax (Brafman 2002): https://www.jmlr.org/papers/volume3/brafman02a/brafman02a.pdf
- Curiosity ICM (Pathak 2017): https://arxiv.org/abs/1705.05363
- RND (Burda 2018): https://arxiv.org/abs/1810.12894
- PPO (Schulman 2017): https://arxiv.org/abs/1707.06347

类似思想的工作:
- AlphaGo / AlphaZero: https://www.nature.com/articles/nature24270
- AlphaTensor: https://www.nature.com/articles/s41586-021-04170-2
- AlphaProof (DeepMind IMO-level): https://deepmind.google/discover/blog/ai-mathematical-olympiad-advances-ai-mathematical-reasoning/

后续发展:
- miniCTX (file-level context): https://arxiv.org/abs/2408.07312
- DeepSeek-Prover V2 (推测后续): https://arxiv.org/abs/2412.09974

## 一句话总结

把 LLM 的连续生成用 Lean verifier 切成离散的 state-action 树，用 RMax 给"见到新 state"发奖驱动探索，用 DUCB 处理非平稳 reward，用 GRPO 让 Lean 的 0/1 binary reward 训出更强的 policy。本质上是把 AlphaZero 那套 pipeline 搬到 formal theorem proving，但还缺 value network 这一块。

---

# DeepSeek-Prover-V1.5: 深度技术讲解

Andrej，这篇 paper 我觉得是定理证明领域一个挺重要的工作，因为它把 AlphaZero 那套 pipeline 思想真正搬到了 formal theorem proving 里来。下面我会非常详细地拆解它的方法、公式、实验和直觉。

## 1. 背景：为什么 theorem proving 难

Formal theorem proving (用 Lean 4 / Isabelle / Metamath) 与自然语言数学推理最大的差别在于 verification oracle 的存在——Lean compiler 是 ground truth，证明要么过要么不过，没有中间分。这意味着 reward 极其 sparse，proof search 是典型的 hard-exploration 问题。

传统两条路线：
- **Proof-step generation** (GPT-f, ReProver, Thor)：每次只生成一个 tactic，调用 verifier 得到新的 tactic state，然后用 tree search 拼接。优点是能利用中间 state，缺点是 communication overhead 大、generation 慢。
- **Whole-proof generation** (DSP, Lyra, DeepSeek-Prover-V1)：直接生成整个 proof code，一次提交。优点是高效，缺点是 long-horizon autoregressive 生成中存在 compounding error——模型对中间 tactic state 的 belief 会逐渐偏离真实状态，导致后续 tactic 基于错误前提继续生成。

DeepSeek-Prover-V1.5 的核心 insight 是：**用 truncate-and-resume 机制把两者统一**——保留 whole-proof 的高效，又重新引入中间 tactic state 反馈。

参考:
- Lean 4: https://lean-lang.org/
- DeepSeek-Prover-V1: https://arxiv.org/abs/2405.14333
- GPT-f: https://arxiv.org/abs/2009.03393
- LeanDojo (ReProver): https://leandojo.org/

## 2. 整体 Pipeline

整体框架是三段式训练 + 一套推理时搜索：

**Pre-training (DeepSeek-Prover-V1.5-Base)**：在 DeepSeekMath-Base (7B) 上继续预训练，加入 code + 数学 + formal language (Lean, Isabelle, Metamath) 数据。

**Supervised Fine-tuning (DeepSeek-Prover-V1.5-SFT)**：两个关键 data augmentation：
1. **Thought-augmented proof generation**：用 DeepSeek-Coder V2 236B 给 Lean 代码加自然语言 CoT 注释，把自然语言推理和 formal tactic 交错放在一起。
2. **Tactic state prompt augmentation**：在 valid proof 的每个 tactic 后面插入 Lean REPL 返回的 tactic state 作为 comment (`/- tactic state: ... -/`)，训练模型预测这些 state（auxiliary objective）和后续 tactic（main objective）。

这第二个 augmentation 是为 MCTS 的 truncate-and-resume 服务的——模型必须学会"看到未完成 proof 的 prefix + 当前 tactic state 注释 → 继续生成"。

**Reinforcement Learning (DeepSeek-Prover-V1.5-RL)**：GRPO 算法，Lean verifier 给 0/1 binary reward。

**Inference**：single-pass sampling 或 RMaxTS (他们提出的 MCTS 变体)。

## 3. GRPO 与 RLPAF 细节

GRPO (Group Relative Policy Optimization) 来自 DeepSeekMath (https://arxiv.org/abs/2402.03300)。核心思想：对每个 prompt $q$，从当前 policy $\pi_\theta$ 采样一组 $G$ 个 outputs $\{o_1, ..., o_G\}$，然后用组内相对 advantage 替代 PPO 中需要训练 critic 估计的 advantage。

GRPO 的 loss 形式大致为：

$$\mathcal{L}_{GRPO} = -\mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left(\rho_{i,t}\hat{A}_{i,t}, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\right) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) \right]$$

其中：
- $\rho_{i,t} = \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}$ 是 importance ratio
- $\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$ 是组内 normalized advantage
- $r_i \in \{0, 1\}$ 是 Lean verifier 的 binary reward
- $\beta$ 是 KL coefficient，paper 里设为 0.02
- $\pi_{ref}$ 是 SFT model（KL anchor）

**关键 prompt 选择策略**：选 SFT model "有中等成功率"的 theorem（≈4.5k 条），保证一组 32 个 sample 中既有正确也有错误证明，这样 group-relative advantage 才有信号。这本质是把 reward sparsity 问题降维——选择"难度刚好"的问题，让 verifier 提供有梯度的 supervision。

**KL penalty 系数 0.02** 比较小，意味着他们让 policy 偏离 reference 较多。learning rate 5e-6，constant。

参考:
- GRPO: https://arxiv.org/abs/2402.03300
- PPO 原始: https://arxiv.org/abs/1707.06347

## 4. Truncate-and-Resume 机制（核心创新）

这个机制是整个 RMaxTS 能在 whole-proof generation 框架下工作的基础。

**Truncate 阶段**：
1. 给定一个 incomplete proof prefix（已经验证成功的部分），LLM 继续生成后续 proof code
2. 把完整 code 提交给 Lean 4 prover 验证
3. 如果完全正确 → 完成
4. 如果有错，找**第一个错误位置**，截断后续所有 code
5. 把成功部分 parse 成 tactic 序列，每个 tactic 对应一个 tree node

**Resume 阶段**：
1. 从 tree 中选一个 node 扩展
2. 取该 node 对应的 incomplete proof prefix
3. **在末尾加一个 tactic state comment** (`/- tactic state: <Lean 返回的当前 state> -/`)
4. 让 LLM 续写——SFT 阶段已经训练过这种格式
5. 续写出来的再 truncate，再 parse 成新 nodes 加进 tree

这个设计有几个微妙之处：
- **同一个 tactic state 可以对应多种 tactic code**（Lean 中不同 tactic 可能达到相同 state），node 里存一组等价 tactic，扩展时随机选一个当 prompt。这相当于做了 state abstraction——以 tactic state 为节点身份而非 tactic text。
- **Virtual node 技术**：每个 node 都有一个 imaginary child $\oslash$ 表示"选择本节点扩展"，这样 tree policy 既可以走到 child 也可以"原地展开"，因为 LLM 的输出空间是开放的，不像围棋是固定 branching factor。
- 一次 expansion 可能插入**一整条 node path**（因为 whole-proof 生成是一串 tactic），而传统 MCTS 一次只扩一层。

## 5. RMaxTS：Intrinsic Reward 驱动的 MCTS

### 5.1 Tree Policy (Selection)

公式 (1)：
$$\text{TreePolicy}(s) = \arg\max_{a \in \text{Children}(s) \cup \{\oslash\}} Q_{UCB}(s, a)$$

变量解释：
- $s$：当前 node（对应一个 tactic state）
- $a$：动作，要么是移动到 child node ($a \in \text{Children}(s)$)，要么是 virtual node $a = \oslash$ 表示原地展开
- $Q_{UCB}(s, a)$：state-action 的乐观价值估计

公式 (2)：
$$Q_{UCB}(s, a) = \underbrace{Q(s, a)}_{\text{Exploitation}} + \underbrace{UCB(s, a)}_{\text{Exploration}}$$

经典 UCB1 (公式 4-6)：
$$Q_{UCB1}(s, a) = \frac{W(s, a)}{N(s, a)} + \sqrt{\frac{2 \ln \sum_{a'} N(s, a')}{N(s, a)}}$$

- $W(s, a) = \sum_{\tau \in \Gamma(s, a)} R(\tau)$：累积 reward，$\Gamma(s, a)$ 是所有包含 $(s,a)$ 的 selection trajectory 集合
- $N(s, a) = |\Gamma(s, a)|$：访问次数
- 第一项是 average reward (exploitation)
- 第二项是 UCB 探索 bonus (随 $N(s,a)$ 增大衰减，随 sibling 总访问增大增大)

### 5.2 Intrinsic Reward (核心创新)

公式 (3)：
$$R_{\text{intrinsic}}(\tau) = \mathbb{I}[\text{at least one new node is added to the search tree}]$$

也就是说，每次 expansion 如果产生**新 tactic state**（之前没见过的），就给 reward 1，否则 0。这就是 RMax (Brafman & Tennenholtz 2002) 的核心思想——把"未知状态"当作最大奖励状态，迫使 agent 探索整个 state space。

在 theorem proving 里这是 ZeroRMax 设定——纯 intrinsic reward，extrinsic reward (proof 完成) 只作为终止信号不进入价值估计。

为什么这个 reward 是 non-stationary？因为随着 tree 扩大，发现新 state 越来越难，所以 $R_{\text{intrinsic}}$ 的期望值随时间衰减。

参考:
- RMax: https://www.jmlr.org/papers/volume3/brafman02a/brafman02a.pdf
- ZeroRMax: https://arxiv.org/abs/2007.03003

### 5.3 Discounted UCB (DUCB) 处理 non-stationarity

公式 (7)-(9)：
$$Q_{DUCB}(s, a) = \frac{W_\gamma(s, a)}{N_\gamma(s, a)} + \sqrt{\frac{2 \ln \sum_{a'} N_\gamma(s, a')}{N_\gamma(s, a)}}$$

$$W_\gamma(s, a) = \sum_{t=1}^{N(s, a)} \gamma^{N(s, a) - t} R(\tau_t)$$

$$N_\gamma(s, a) = \sum_{t=0}^{N(s, a)-1} \gamma^t$$

变量解释：
- $\gamma \in (0, 1)$：discount factor，paper 设 $\gamma = 0.99$
- $\tau_t$：第 $t$ 次访问该 $(s, a)$ 的 trajectory（按时间排序）
- $W_\gamma$：discounted 累积 reward，**越新访问权重越大**（指数为 $N(s,a) - t$，$t = N(s,a)$ 时指数为 0，权重为 1）
- $N_\gamma$：discounted 访问次数，等于 $\frac{1 - \gamma^{N(s,a)}}{1 - \gamma}$

直觉：因为 $R_{\text{intrinsic}}$ 在树长大后变得稀疏（早期容易发现新 state，后期难），UCB1 假设 stationary reward，会把过时的早期 1-reward 平均进来，导致 exploitation 信号过强（误以为这个节点一直高 reward）。DUCB 通过指数 discount 让旧 reward 衰减，使 value estimate 跟踪当前 intrinsic reward 的真实期望。

paper 里 ablation 证实：去掉 intrinsic reward (UCT) 性能退化到接近 single-pass；用 UCB1 替代 DUCB 也退化（因为 UCB1 的渐近保证在有限样本下不适用）。这是 paper 一个相当精彩的设计。

注意：$\gamma$ 这里**不是** MDP 里 horizon discount，而是**树搜索迭代间的 discount**——它影响的是"第几次访问该 (s,a)"，不是 trajectory 内的 step discount。这是 paper 里明确指出的微妙区别。

参考:
- DUCB: https://arxiv.org/abs/0807.0889

### 5.4 Expansion

Expansion = 调 LLM 做 whole-proof generation，提交 Lean 验证：
- 完全通过 → 搜索成功终止
- 有错误 → 截断到第一个错误，parse 成功部分为 tactic 序列，每个 tactic → 新 node 接到 selected node 下面

这与传统 MCTS 一次只扩一层不同——一次可能扩整条 path。这种"bulk expansion"在 LLM-based MCTS 里很常见（Tree of Thoughts, Yao et al. 2023 也是类似思路）。

参考:
- Tree of Thoughts: https://arxiv.org/abs/2305.10601

### 5.5 Backpropagation

沿着 selection trajectory $\tau = \{(root, s^{(1)}), (s^{(1)}, s^{(2)}), ..., (s^{|\tau|-1}=s_t, \oslash)\}$ 把 $R(\tau) = R_{\text{intrinsic}}(\tau)$ 更新到所有 $(s, a) \in \tau$ 的 $W_\gamma$ 和 $N_\gamma$ 上。

注意 virtual loss 实现：并发 worker 选中某个 node 时先 backprop 一个临时 reward 0，避免其他 worker 重复选同一 node。完成后更新为真实 reward。

### 5.6 并行化

- **Root parallelization**: 256 个 MCTS runner 每个 GPU 一个 LLM，batch size 512；Lean REPL 在 CPU cluster 上跑，每个 proof verification 单独 process + sandbox。Generation 和 verification 异步。
- **Tree parallelization**: 每个搜索树 32 个 thread worker 并行 MCTS 循环。
- **Virtual loss**: 临时 R=0 阻止重复选择。

这套并行度（256 × 32 = 8192 并发线程）非常激进，背后是 DeepSeek 的算力支撑。

参考:
- Parallel MCTS (Chaslot 2008): https://link.springer.com/chapter/10.1007/978-3-540-87608-3_7

## 6. 实验结果深度分析

### 6.1 Benchmark

- **miniF2F** (https://github.com/openai/miniF2F): 244 valid + 244 test，高中竞赛级 (AMC/AIME/IMO)，Lean 4.9.0
- **ProofNet** (https://arxiv.org/abs/2302.12433): 185 valid + 186 test，本科级数学 (real/complex analysis, linear algebra, abstract algebra, topology)

### 6.2 阶段对比 (Figure 3)

miniF2F-test Pass@128:
- Base (3-shot): 29.7%
- SFT (non-CoT): 49.8%
- SFT (CoT): 50.4%
- RL (non-CoT): 50.5%
- RL (CoT): 51.6%

每个阶段都在累加。RL 在 formal proving 里的提升比 NL 数学 (DeepSeekMath) 里更明显——NL 数学 RL 主要 boost TopK（多次采样里最好那次的概率），但这里 RL 提升了 base capability（即使小样本 budget 也提升）。

### 6.3 主结果 (Table 1, miniF2F-test)

| Method | Sample budget | Pass rate |
|--------|---------------|-----------|
| GPT-4 (via COPRA) | 1×60 | 26.6% |
| ReProver | 1×32×100 | 26.5% |
| Hypertree Proof Search | 64×5000 | 41.0% |
| Lean-STaR | 64×1×50 | 46.3% |
| InternLM2-StepProver | 64×32×100 | 54.5% |
| DeepSeek-Prover-V1.5-RL (single-pass) | 32 | 50.0% |
| DeepSeek-Prover-V1.5-RL (single-pass) | 128 | 51.6% |
| DeepSeek-Prover-V1.5-RL (single-pass) | 16×6400 | 60.2% |
| DeepSeek-Prover-V1.5-RL + RMaxTS | 1×3200 | 55.0% |
| DeepSeek-Prover-V1.5-RL + RMaxTS | 32×6400 (mixture) | **63.5%** |

关键观察：
- **Single-pass 60.2% (16×6400=102400 attempts)** 已经超过 InternLM2-StepProver 的 54.5%
- **RMaxTS 用 3200 budget 达到 55.0%**，比 InternLM2-StepProver 用的 204800 budget 高效得多
- **Mixture (non-CoT + CoT)** 在 32×6400 达到 SOTA 63.5%

### 6.4 Mixture strategy (Table 3)

non-CoT 擅长用 Lean 高级 tactic (nlinarith, field_simp 等) 解决计算型问题；CoT 擅长需要数学推理规划的问题。Mixture 把 sample budget 各分一半给两种模式，简单组合就涨 ~1%。

### 6.5 RMaxTS Ablation (Figure 5)

| 变体 | 4×6400 | 16×6400 |
|------|--------|---------|
| Single-Pass | 58.4% | 60.2% |
| UCT (no intrinsic) | 58.2% | 61.1% |
| RMaxTS (DUCB→UCB1) | 58.6% | 60.7% |
| RMaxTS (no tactic state) | 58.4% | 61.1% |
| RMaxTS (full) | **59.6%** | **62.7%** |

三个组件都不可或缺：
- 去掉 intrinsic reward → 退化到 single-pass 水平（因为没有任何 reward 信号引导 UCB 选择）
- DUCB 换 UCB1 → 退化（旧 reward 污染 current estimate）
- 去掉 tactic state 注释 → 退化（模型失去中间 state 的 grounding，长程规划能力降级）

## 7. 与其他方法对比

### 7.1 Hypertree Proof Search (HTPS, Lample et al. 2022)

也是 MCTS + formal proving，但：
- HTPS 是 proof-step generation，每个 node 一个 tactic
- HTPS 训练 value/critic 模型评估 partial proof
- DeepSeek-Prover-V1.5 不训练 critic，靠 intrinsic reward + DUCB 解决 exploration

### 7.2 Lean-STaR (Lin et al. 2024)

也是在每个 tactic 前加 CoT，但：
- Lean-STaR 的 CoT 是 isolated reasoning before each step
- DeepSeek-Prover-V1.5 把 CoT 整合进 proof code 作为 comment，并用 RL 进一步优化
- DeepSeek 用 236B 大模型 annotate，质量更高

### 7.3 InternLM2-StepProver (Wu et al. 2024)

64×32×100 = 204800 budget 达到 54.5%；DeepSeek-Prover-V1.5+RMaxTS 用 3200 达到 55%，**计算效率差 60 倍以上**。

参考:
- HTPS: https://arxiv.org/abs/2205.11491 (NeurIPS 2022)
- Lean-STaR: https://arxiv.org/abs/2407.10040
- InternLM2-StepProver: https://arxiv.org/abs/2407.17227

## 8. 几个值得思考的设计 choice

1. **为什么用 RMax 而不是 ICM/RND?** Pathak 的 ICM (Intrinsic Curiosity Module) 和 Burda 的 RND (Random Network Distillation) 都需要训练额外的预测网络。RMax 只需要"是否新 state"的 indicator，在 state space 离散且可枚举（tactic state 是结构化的）的情况下极其简洁高效。Lean 的 tactic state 是结构化文本，可以直接 hash 比较。

2. **为什么 reward 用 proof 完成 (0/1) 而非中间信号?** Paper 提到 Lean 的 binary reward 是准确的 (ground truth from compiler)，但 sparse。他们通过 prompt 选择（中等难度）来缓解 sparsity，而不是设计 shaped reward。这与 AlphaGo 的做法不同——AlphaGo 训练了 value network。Paper 在 Conclusion 里明确说：未训练 critic 是局限，未来方向。

3. **为什么 mixture 比 single-mode 好?** CoT 和 non-CoT 在不同问题类型上互补。CoT 适合需要推理规划的问题（如 numbertheory_x5neqy2p4 用 mod 11 讨论所有余数），non-CoT 适合可被 Lean automation 解决的计算问题（如 induction_pord1p1on2powklt5on2 用 nlinarith + field_simp）。这暗示 LLM 的不同 prompting 模式激活了不同的"证明策略先验"。

4. **Truncate-and-resume 的真正威力**：它把 LLM 的 generation 当作一个 implicit rollout policy——一次 generation 等于传统 MCTS 的 simulation rollout。但和传统 rollout 不同，generation 是 conditioned on 整个 prefix 的，所以可以"看到"已经走过的 path。这种 rollout 又被 parse 成 discrete states 进 tree，相当于把连续 generation 离散化进树结构。

## 9. Limitation 和未来方向

Paper 自己点出了：
1. **没训练 critic model**。AlphaZero 的 self-play 之所以能超越人类，关键是 value network 引导 search。DeepSeek-Prover-V1.5 完全靠 intrinsic reward 探索，没有 exploitation 信号评估 partial proof 价值。这是最大的未来方向——partial-proof critic 可以做 temporal credit assignment（Sutton 1984, RUDDER Arjona-Medina 2019）。
2. **File-level context**。当前只证 single theorem，但 real Lean project 是多 theorem 文件（miniCTX, Hu et al. 2024 已经开始探索）。模型需要理解 file-level context。
3. **Expert iteration 的循环**: 当前 SFT 用 expert iteration 生成数据，但没有 self-play 的闭环。AlphaZero pipeline 是 train → search → self-play generate data → retrain，DeepSeek 这版只做了前半段。

参考:
- RUDDER: https://arxiv.org/abs/1806.07857
- miniCTX: https://arxiv.org/abs/2408.07312
- Sutton credit assignment: https://scholarworks.umass.edu/dissertations/AAI8410337/

## 10. 对你的几个直觉构建

Karpathy，我猜你读这篇 paper 会想几个 angle：

**(a) 关于 tree search + LLM 的统一**: 这工作本质是把 LLM 的 token-level generation "重映射" 到 state-action 层面。Truncate-and-resume 就是这个映射函数。这和 AlphaCode 的 cluster-then-rank、Tree of Thoughts 的 thought-as-node、Swift-Sage 的 planning-acting split 都是一个 family——把 LLM 的连续 generation 切片成离散 state 序列喂给经典 search 算法。

**(b) 关于 RMax 在 LLM 时代的回归**: RMax 是 2002 年的算法，理论漂亮但 sample inefficient。在 Lean theorem proving 这种 state space 巨大但 verifier 可枚举的环境里，"新 state reward 1" 这个简单 indicator 反而 work，因为 (i) Lean tactic state 结构化，可以直接 hash 比较；(ii) 探索本身就是目标——定理证明的难点不是 exploit 而是找到那条 proof path。这个 reward 在 sparse-reward MDP 里就是"好奇心"的离散化版本。

**(c) 关于 DUCB 的物理意义**: 你可以把它理解为 EWMA (exponentially weighted moving average) 应用到 UCB。早期 high-reward 信号不应该永远主导 current value estimate——因为 R_max 的 reward 期望随探索深度衰减。$\gamma = 0.99$ 大约对应"~70 次访问前的 reward 衰减到 1/2"，这对中等频次访问的 node 是合理的 memory horizon。

**(d) 关于 RL 在 formal vs NL 的差异**: DeepSeekMath 发现 RL 主要 lift TopK，但这里 RL lift base capability。我猜原因是 NL 数学 RL 的 verifier (rule-based / model-based) 有噪声，模型学到的是"匹配 verifier 偏好"，而 Lean verifier 是绝对准确的——RL 真正在学"形式化正确性"本身。这点很值得深入挖。

**(e) Critic model 的缺失**: Paper 自己承认没做 critic。这意味着他们的 MCTS 完全是 model-free 的（靠 intrinsic reward + UCB exploration bonus）。AlphaZero 的 power 来自 model-based（value + policy network）。如果 DeepSeek 下一版训一个 partial-proof value model（评估 incomplete proof 完成概率），再把这个 value 作 Q 的 prior，应该能再涨一截。这相当于 AlphaGo 的 rollout policy + value network 联合，但目前 DeepSeek 只有 rollout (LLM generation) 没有 value。

**(f) 关于 mixture strategy 的简单胜出**: CoT 和 non-CoT 各 50% sample budget 比纯 CoT 多 1.3%。这暗示 prompt template 是个 latent variable——不同 template 召回不同证明策略。这其实可以做得更精细，比如按 problem type 学一个 router 决定 sample budget 分配。但 paper 没做这一步，留作 future work。

## 11. 一些更广的关联

- 这套 pipeline 几乎是 AlphaZero (https://www.nature.com/articles/nature24270) 的形式化定理证明版本：pre-training ≈ policy initialization from human data；SFT ≈ supervised policy learning；RL+verifier ≈ self-play with environment reward；RMaxTS ≈ MCTS planning。缺的就是 value network。
- Fawzi et al. 2022 (AlphaTensor, https://www.nature.com/articles/s41586-021-04170-2) 在矩阵乘法上用了类似的 RL+search 框架。Lutz et al. 2023 (protein design, https://www.science.org/doi/10.1126/science.add1964) 用 RFdiffusion + search 做蛋白质设计。
- Curiosity-driven exploration (Pathak ICM, https://arxiv.org/abs/1705.05363; Burda RND, https://arxiv.org/abs/1810.12894) 在 video game 上 work，但 LLM reasoning 上还很少见到 application。RMaxTS 是这个 family 在 reasoning 领域的一个代表。
- RLPAF 和 RLHF (https://arxiv.org/abs/2203.02155) 在精神上很类似，但 verifier 从 human preference model 换成了 Lean compiler——这其实更接近 AlphaGo 的 terminal reward from game outcome。

## 12. 我会问的问题

如果我和作者讨论，我会问：

1. RMax 的 intrinsic reward 在 tree 长大后极度稀疏，DUCB $\gamma=0.99$ 是怎么 tune 的？larger tree (e.g. 32×6400) 时这个 $\gamma$ 还合适吗？是否需要 adaptive $\gamma$？
2. Truncate-and-resume 的 truncate 是"第一个错误处"，但如果中间有个 tactic 暂时通过但其实把证明引向 dead end，怎么办？是否考虑过 backjumping 或者 multi-error truncate？
3. Node identity 是 tactic state，但 Lean 的 tactic state 文本表示有时很长。有没有 state abstraction / canonicalization 来减少冗余 node？
4. RL 的 GRPO 没有 critic，但在 theorem proving 中间 step 是有信息的（即使最终没证完，某些 path 离成功更近）。是否考虑过 process reward model (PRM, https://arxiv.org/abs/2305.20050) 给 step-level reward？
5. Mixture CoT/non-CoT 各 50% 这个分配是怎么决定的？是否有 per-problem adaptive allocation？
6. Base model 用 DeepSeekMath-Base 7B，为什么不用更大模型？是 compute 限制还是 7B 已经够？
7. Lean 4 REPL 调用是 bottleneck 吗？paper 说每 verification 单独 process + sandbox，overhead 应该不小。

## Reference 链接汇总

**主 paper**:
- DeepSeek-Prover-V1.5 GitHub: https://github.com/deepseek-ai/DeepSeek-Prover-V1.5
- arXiv (推测): https://arxiv.org/abs/2408.08152

**前置工作**:
- DeepSeek-Prover-V1: https://arxiv.org/abs/2405.14333
- DeepSeekMath (GRPO): https://arxiv.org/abs/2402.03300
- DeepSeek-Coder V2: https://arxiv.org/abs/2406.11931

**Benchmark**:
- miniF2F: https://github.com/openai/miniF2F
- ProofNet: https://arxiv.org/abs/2302.12433
- Lean 4: https://lean-lang.org/
- Mathlib4: https://github.com/leanprover-community/mathlib4
- Aesop: https://arxiv.org/abs/2311.02040

**对比方法**:
- GPT-f: https://arxiv.org/abs/2009.03393
- ReProver (LeanDojo): https://leandojo.org/
- HTPS: https://arxiv.org/abs/2205.11491
- Lean-STaR: https://arxiv.org/abs/2407.10040
- InternLM2-StepProver: https://arxiv.org/abs/2407.17227
- COPRA: https://arxiv.org/abs/2310.04653
- Llemma: https://arxiv.org/abs/2310.10631
- Tree of Thoughts: https://arxiv.org/abs/2305.10601

**算法基础**:
- MCTS (Browne 2012): https://ieeexplore.ieee.org/document/6145622
- UCB1 (Auer 2002): https://link.springer.com/article/10.1023/A:1013689704352
- DUCB (Garivier 2011): https://arxiv.org/abs/0807.0889
- RMax: https://www.jmlr.org/papers/volume3/brafman02a/brafman02a.pdf
- ZeroRMax: https://arxiv.org/abs/2007.03003
- Curiosity ICM: https://arxiv.org/abs/1705.05363
- RND: https://arxiv.org/abs/1810.12894
- PPO: https://arxiv.org/abs/1707.06347
- AlphaZero: https://www.nature.com/articles/nature24270
- AlphaTensor: https://www.nature.com/articles/s41586-021-04170-2
- RLHF: https://arxiv.org/abs/2203.02155
- PRM (Process Reward): https://arxiv.org/abs/2305.20050
- Sutton credit assignment: https://scholarworks.umass.edu/dissertations/AAI8410337/
- RUDDER: https://arxiv.org/abs/1806.07857

**最新发展 (post-paper)**:
- miniCTX (file-level context): https://arxiv.org/abs/2408.07312
- AlphaProof (DeepMind, IMO-level): https://deepmind.google/discover/blog/ai-mathematical-olympiad-advances-ai-mathematical-reasoning/
- DeepSeek-Prover-V2 (后续): https://arxiv.org/abs/2412.09974

希望这个深度讲解帮你 build 起对这篇 paper 的完整直觉。我个人觉得这篇 paper 最重要的 contribution 不是 GRPO 也不是 CoT，而是 **truncate-and-resume 这个 abstraction**——它把 LLM 的 generation 离散化为 tree search 的 action，并用 Lean 的中间 state 作为 node identity。这个 abstraction 应该可以推广到任何有 incremental verifier 的领域（比如 SQL 生成 + parser feedback、Rust borrow checker、formal verification in hardware design）。RMax + DUCB 那套是 exploration 算法层面的锦上添花，但 abstraction 本身是 structural insight。
