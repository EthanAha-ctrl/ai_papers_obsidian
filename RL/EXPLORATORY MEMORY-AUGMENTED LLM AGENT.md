---
source_pdf: EXPLORATORY MEMORY-AUGMENTED LLM AGENT.pdf
paper_sha256: c8dd70bda97ac1fcf5535ceb7f99a6256bfac26638839aff1492c9a4edd1b36d
processed_at: '2026-08-04T06:19:29-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 EMPO²

## 一句话版本

**给一个失忆的探险家一本笔记本，让他边探索边记笔记，而且定期把笔记里的经验"长进脑子里"，这样他既会用笔记本也会不用笔记本探索。**

---

## 问题长什么样

你训练一个 LLM agent 玩 ScienceWorld，任务是"开红灯泡"。问题是红灯泡不在当前房间，你得先去找。但 GRPO 训练下的 agent 做了一件很蠢的事：

- 它看到任务说"focus on red light bulb"，就真的去 focus
- 环境说"这里没有红灯泡"，agent 失败
- 下一轮训练，它**又做一模一样的事**

为什么？因为 GRPO 的 rollout 之间**完全没联系**。每一轮训练，agent 都是"失忆的探险家"，不知道上一轮为什么失败。它只收到一个 scalar reward 说"你得了 -100 分"，然后继续从零开始摸索。

这就像你让一个人走迷宫，他每次走到死胡同就传送回起点，而且**清空记忆**。他永远不会记得"左边那条路是死的"。

---

## 核心思路：给他一本笔记本

EMPO² 的第一个想法非常朴素：**让 agent 每次失败后写一句话总结**，存到一个 memory buffer 里。下一次探索时，先翻笔记本看看相关的经验。

比如 agent 写的 tip 会有这种递进：

```
第1轮：我在厨房和浴室转了一圈，没找到绿灯泡
第2轮：我连了绿灯泡但没找到电池
第3轮：电路接好了但灯没亮，可能door是关的
第4轮：灯亮了，但门重复关了
```

下一次 agent 进环境前看到这些 tip，就会避免重复犯错，直接去探索新的方向。

这个 idea 本身不新，Reflexion (https://arxiv.org/abs/2303.11366) 早就做过。问题是：**参数固定的 agent 写出来的 tip 很快就饱和了**，因为它的"脑子"没在进步，tip 质量上不去。

---

## 关键创新：把笔记本内容"长进脑子"

EMPO² 做的事是：**一边用笔记本，一边把笔记本的好处内化到模型参数里**。

怎么内化？这里有个很巧妙的 trick。

### 三种训练模式

Rollout 阶段有两种模式：
1. **不带笔记本**：agent 直接看环境做决策 $a_t \sim \pi_\theta(\cdot | s_t, u)$
2. **带笔记本**：agent 先翻笔记本，带着 tip 做决策 $a_t \sim \pi_\theta(\cdot | s_t, u, \text{tips}_t)$

Update 阶段也有两种模式：
- **(a) On-policy with tips**：update 时分子分母都带 tips，这是标准做法
- **(b) Off-policy**：update 时分子**去掉 tips**，分母保留 tips

Mode (b) 是核心魔法。让我详细解释：

### Off-Policy Update 为什么是"知识蒸馏"

想象两个 agent：
- **老师**：带着笔记本的 agent，因为看到 tip 所以表现好
- **学生**：不带笔记本的 agent，是我们要训练的最终目标

Off-policy update 的 importance sampling ratio 是：

$$\rho_\theta(a_t^{(i)}) = \frac{\pi_\theta(a_t^{(i)} | s_t^{(i)}, u)}{\pi_{\theta_{old}}(a_t^{(i)} | s_t^{(i)}, u, \text{tips}_t)}$$

变量解释：
- 分子 $\pi_\theta(a_t | s_t, u)$：学生（当前 policy，不看 tips）给这个 action 的概率
- 分母 $\pi_{\theta_{old}}(a_t | s_t, u, \text{tips}_t)$：老师（旧 policy，看 tips）给这个 action 的概率

这个 ratio 有点 weird，因为分子分母的 conditioning 不一样。但效果很清楚：

- 如果老师带着 tip 采了一个**好 action**（$A > 0$），advantage 是正的，学生会增加这个 action 的概率 → **学生在模仿老师的优秀行为**
- 如果老师带着 tip 采了一个**差 action**（$A < 0$），advantage 是负的，学生会减少这个 action 的概率 → **学生在避开老师的失败行为**

这就是 **reward-guided knowledge distillation**。老师因为看 tip 所以能采到好 action，学生看不到 tip 但通过 advantage 信号学到"这个 action 是好的"，于是学生在没有 tip 的时候也能做出好 action。

### 与 Context Distillation 的区别

Snell 2022 (https://arxiv.org/abs/2209.15189) 的 context distillation 是 offline SFT：先让 teacher 带着长 prompt 生成答案，再让学生用最短 prompt 学会生成同样的答案。

EMPO² 把这个 idea 搬到 **online RL** 里，用 advantage 替代 uniform imitation loss，做到 selective distillation。而且 tip 是 agent **自己生成**的，不需要外部 teacher。

---

## 为什么要 Masking

Off-policy 训练有个坑：student 给某个 token 的概率可能极低，teacher 给的概率很高，importance sampling ratio $\rho = \pi_{student} / \pi_{teacher}$ 会爆炸。

Yang et al. 2025 (https://arxiv.org/abs/2505.12929) 分析过这个现象：low-probability tokens 通过 unbounded ratio 放大 gradient，导致 NaN collapse。

EMPO² 的修复是在 loss 里加一个 mask：

$$\mathbf{1}_{\pi_\theta(a_t | s_t, u) \geq \delta}$$

当 student 给该 token 的概率低于 $\delta$ 时，这个 token 的 advantage 项被 mask 掉，不参与 gradient 计算。

Figure 6 展示了效果：no masking → 训练 collapse 到 NaN；with masking → 稳定收敛。

这本质上是承认 **off-policy 训练有 distribution shift 风险**，通过 truncation 换取稳定性。

---

## Intrinsic Reward：维持好奇心

还有一个补充组件：**intrinsic reward 鼓励 novelty**。

$$r_{intrinsic} = \frac{1}{n}$$

- $n$：跟当前 state 相似的历史 state 数量
- 全新 state（$n$ 小）→ 高 intrinsic reward
- 已见 state（$n$ 大）→ 低 intrinsic reward

这是经典 count-based exploration (Bellemare 2016, https://arxiv.org/abs/1606.01868) 的简化版。

Figure 7 显示：没有 intrinsic reward，policy entropy 快速 collapse，agent 越来越保守；有 intrinsic reward，entropy 维持，agent 持续探索。

---

## 为什么这个组合 work

Figure 9 的 ablation 说明三者缺一不可：

1. **On-policy without memory**（标准 GRPO）：提供 baseline stability，但 exploration 不足
2. **On-policy with memory**：memory-augmented rollout 的直接学习，保持 distribution consistency
3. **Off-policy distillation**：把 memory 的 benefit 内化到 weights，是 generalization 的关键

移除 off-policy → memory benefit 无法内化，test 时（不用 memory）性能差
移除 on-policy with memory → 训练信号不稳定，容易 collapse

三者的关系可以理解为：
- On-policy no-memory 是 **保险**，保证最差也能学到 GRPO 级别
- On-policy with memory 是 **放大器**，利用 memory 提升训练信号质量
- Off-policy 是 **桥梁**，把 memory benefit 蒸馏到 no-memory policy

---

## 实验结果的故事

### ScienceWorld：128.6% 提升

| Method | Avg |
|---|---|
| Naive Qwen2.5-7B | -61.3 |
| Reflexion (memory only) | 17.1 |
| Retrospex (offline RL) | 33.8 |
| GRPO (online RL) | 33.2 |
| EMPO² | **75.9** |

7 个 task 从负分达到满分 100。Power-component task 从 -90 到 94.3。

Figure 1a 的 training curve 最直观：GRPO 很快 plateau 在 suboptimal，EMPO² 持续上升。这就是 exploration 的威力——GRPO agent 卡在"找不到红灯泡"就不再尝试新策略，EMPO² agent 通过 memory 知道"上一轮在厨房没找到，试试别的房间"。

### WebShop：11.3% 提升

提升幅度比 ScienceWorld 小，因为 WebShop 的 horizon 更短，reward 更 dense，exploration bottleneck 没那么严重。但 EMPO² 仍然超过 GiGPO (https://arxiv.org/abs/2505.10978) 这个更强的 baseline。

### OOD 实验：最有意思的结果

Figure 8 展示了最 exciting 的发现：**训练完的 EMPO² model 在完全新的 task 上，只用 memory（no weight update）就能快速适应**。

- Biology 1 → Biology 2：相似 topic 转移，快速适应
- Biology 2 → Electricity：不同 topic，仍能适应
- Electricity → Chemistry：完全不同 domain，仍有效

10 步内平均 136% 改善。GRPO 在 OOD 场景下表现极差，有时甚至低于 base model。

**这说明 EMPO² 学到的 meta-skill 是 "如何用 memory 探索新环境"，而不仅仅是"怎么解某个 task"**。这是 generalizable intelligence 的关键特征。

---

## 与我熟悉的方法的联系

### AlphaGo 的 bootstrap

AlphaGo 先 SL from human，再 RL self-play bootstrap。EMPO² 类似：memory-augmented policy 是 "强 policy"，no-memory policy 是 "弱 policy"，通过 advantage-weighted distillation 让弱 policy 赶上强 policy。

### Voyager 的 skill library

Voyager (https://arxiv.org/abs/2305.16291) 在 Minecraft 中构建 code skill library，是 non-parametric memory。EMPO² 的 tip 是 verbal memory，更轻量但 less structured。

### Decision Transformer

DT (https://arxiv.org/abs/2106.01345) 把 RL 变成 sequence modeling，用 return-to-go 作 conditioning。EMPO² 的 tips 也是一种 context conditioning，只不过 tips 是 cross-trajectory 的，DT 的 RTG 是 within-trajectory 的。

### STaR (Self-Taught Reasoner)

STaR (https://arxiv.org/abs/2203.14465) 用 model 自己生成 rationale 作为训练信号。EMPO² 的 tips 是 cross-trajectory 的 rationale，STaR 的 rationale 是 within-trajectory 的。

### In-context learning vs weight learning

EMPO² 本质上是在回答一个 deep question：**何时用 in-context learning（memory），何时把知识 bake 进 weights**。Off-policy update 是把 in-context knowledge 蒸馏到 weights 的桥梁。这与 Anthropic 的 induction heads 研究 (https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) 主题呼应。

---

## Compute Cost

Figure 12：memory mechanism 增加 19% rollout time，主要来自 tip generation。

Figure 13：按 wall-clock time 比较，EMPO² 仍显著优于 GRPO。Response length 更长说明 agent 在做更多 reasoning，这是好事。

---

## Open Questions

1. **Memory retrieval 太简单**：只用 cosine similarity + threshold 0.5。可以用 learned retriever，或者 hierarchical memory
2. **7B 模型限制**：scaling 到 70B 会怎样？tip 质量会更好吗？
3. **Domain 限制**：math, coding, multi-hop QA 上效果未知
4. **Tip 质量 vs 数量**：没有深入分析什么时候 tip 有用什么时候有害
5. **Memory buffer 固定 1000**：更好的 forgetting 机制可能提升
6. **Off-policy 稳定性依赖 masking**：threshold $\delta$ 需要调，理论上更 elegant 的 importance sampling variant 可能存在

---

## 最终 Intuition

**Memory 是脚手架，weights 是最终归宿。**

EMPO² 的故事：
1. Agent 失败 → 写 tip（self-reflection）
2. 下一轮看 tip → 避免重复错误，探索新区域（exploration boost）
3. 用 tip 采到好 action → advantage 信号更好（training signal quality 提升）
4. Off-policy update 把 "有 tip 才能做到的好行为" 蒸馏到 "无 tip 也能做到"（knowledge internalization）
5. 更强的 policy 生成更好的 tip → 正循环（bootstrap）
6. 最终 policy 学会了 "如何用 memory 探索" 这个 meta-skill → OOD generalization

整个 paper 没有发明新 RL 算法，是三个 idea 的巧妙组合：self-reflection memory + importance sampling distillation + count-based intrinsic reward。LLM 的语言能力作为 glue 把三者粘起来。

对你（Karpathy）的 micrograd 视角，可以理解为：**memory 在 trajectory 之间做 information carry，weights 在 trajectory 之内做 gradient update，两者协同形成 amortized learning**。这跟你最近谈过的 "Software 2.0 / 3.0" 讨论里 "prompt 作为 program" 的思想是相通的——tip 就是 agent 自己写的 prompt program，weights 通过蒸馏把这个 program 编译成 native capability。

**Key references**:
- Paper project: https://github.com/agent-lightning/empo2
- Agent Lightning framework: https://arxiv.org/abs/2508.03680
- GRPO: https://arxiv.org/abs/2402.03300
- Reflexion: https://arxiv.org/abs/2303.11366
- Context Distillation: https://arxiv.org/abs/2209.15189
- Voyager: https://arxiv.org/abs/2305.16291
- Decision Transformer: https://arxiv.org/abs/2106.01345
- Go-Explore: https://arxiv.org/abs/1901.10995
- Count-based exploration: https://arxiv.org/abs/1606.01868
- RND: https://arxiv.org/abs/1810.12894
- Low-prob token issue: https://arxiv.org/abs/2505.12929

---

# EMPO² 深度解析：Memory-Augmented Hybrid RL for LLM Agents

## 1. 核心问题：Exploration Bottleneck

LLM agent 在 RL 训练中存在一个根本性矛盾：**预训练 prior 与环境真实 dynamics 的不匹配**。Paper Section 3 用 ScienceWorld 的 "turn on the red light bulb" 案例展示了这个 pathology：

- Agent 视野里没有 red light bulb
- 但 agent 仍 literal 地执行 "focus on red light bulb"
- 失败后 GRPO 只给 scalar reward，agent 无法诊断失败原因
- Score stagnates，policy collapse 到 suboptimal solution

这个问题的本质是 **GRPO 的 credit assignment 是 trajectory-level 的**，每个 rollout 是独立的，rollout 之间没有 information continuity。Agent 无法 "记住" 上一次为什么失败。

**Intuition building**：把 GRPO 想象成一个失忆的探险家，每次进入迷宫都从零开始，永远不会积累 "这条路是死胡同" 的知识。Memory augmentation 就是给探险家一本 notebook。

## 2. 方法的核心 idea：Dual-Update Paradigm

EMPO² 的关键 insight 在 Figure 2：

```
Non-parametric updates (memory) → bootstraps → Parametric updates (weights)
```

这是一个 **bootstrap 循环**：
1. Memory 积累 trial-and-error 经验
2. 经验引导下一次 rollout 探索 novel region
3. Novel rollout 提供了高质量 training signal 给 policy gradient
4. 更新后的 policy 生成更好的 tips 写入 memory
5. 最终 policy 把 memory 的 benefit 内化到 weights 中

**与 Reflexion 的本质区别**：Reflexion 只做 step 1-2，参数固定，所以会 saturate。EMPO² 完成整个闭环。

参考链接：
- Reflexion: https://arxiv.org/abs/2303.11366
- Context Distillation (Snell et al. 2022): https://arxiv.org/abs/2209.15189

## 3. Self-Generated Memory Mechanism

### 3.1 Tip 生成

Memory buffer $\mathcal{M} = \{\text{tip}_1, \text{tip}_2, ...\}$ 存储 policy 自己生成的反思 tip。

$$\text{tip}_i \sim \pi_\theta(s_t, u, \text{tip-generation prompt})$$

变量含义：
- $\text{tip}_i$：第 i 条 tip（自然语言句子）
- $s_t$：trajectory 终止时的 final state
- $u$：task instruction
- $\pi_\theta$：当前 policy（同一个被训练的 LLM）

Paper Appendix B 的 tip-generation prompt 要求 "summary the trajectory, what information you get, how far from completion, one sentence under 100 words"。

### 3.2 Tip 示例（来自 paper）

ScienceWorld <power-component> task 的 tips 演化：
- 早期："You moved between kitchen and bathroom but did not find a green wire"
- 中期："Connected green wire to green light bulb but did not find the battery"
- 后期："The green light bulb was turned on, but the door to workshop was closed repeatedly"

注意 tip 的 **progressive refinement** —— 从 "找不到对象" 到 "电路没接好" 到 "灯亮了但有其他问题"。这就是 memory 在做 information accumulation。

### 3.3 Retrieval

$$\text{Retr}(o_t; \mathcal{M}) \subseteq \mathcal{M}$$

实现细节（来自 Appendix D.2 的 code）：
- 用 embedding 做 cosine similarity search
- 阈值 > 0.5
- 按 score 排序取 top-10
- Memory buffer 每 environment 维护 1000 条，FIFO eviction

## 4. Hybrid Policy Optimization：三种 Mode

Figure 5 展示的 2×2 矩阵：

| Rollout \ Update | On-Policy | Off-Policy |
|---|---|---|
| No memory | (1)+(a) Regular GRPO | N/A |
| With memory | (2)+(a) On-policy w/ tips | (2)+(b) Off-policy distillation |

### 4.1 Rollout Modes

每步以概率 $p$ 选 memory-augmented，概率 $1-p$ 选 no-memory：

**Mode 1**: $a_{t+1} \sim \pi_\theta(\cdot | s_t, u)$

**Mode 2**: $a_{t+1} \sim \pi_\theta(\cdot | s_t, u, \text{tips}_t)$

其中 $\text{tips}_t = \text{Retr}(s_t; \mathcal{M})$

### 4.2 Update Modes

**Mode (a) On-Policy with Tips**:
$$\rho_\theta(a_t^{(i)}) = \frac{\pi_\theta(a_t^{(i)} | s_t^{(i)}, u, \text{tips}_t)}{\pi_{\theta_{old}}(a_t^{(i)} | s_t^{(i)}, u, \text{tips}_t)}$$

分子分母都用 tips conditioning，distribution 一致，所以是 on-policy。

**Mode (b) Off-Policy**:
$$\rho_\theta(a_t^{(i)}) = \frac{\pi_\theta(a_t^{(i)} | s_t^{(i)}, u)}{\pi_{\theta_{old}}(a_t^{(i)} | s_t^{(i)}, u, \text{tips}_t)}$$

这是 EMPO² 的 **核心创新**：
- Rollout 时 action 是在 tips-conditioned 分布下采样的（teacher）
- Update 时 student policy 不看 tips
- 这本质上是 **reward-guided knowledge distillation**

### 4.3 Knowledge Distillation 视角

- **Teacher**: $\pi(\cdot | s, u, \text{tips})$ - 有 memory scaffold 的强 policy
- **Student**: $\pi(\cdot | s, u)$ - 无 memory 的弱 policy
- **Selective distillation**:
  - $A_t > 0$ 的轨迹被 reinforce
  - $A_t < 0$ 的轨迹被 suppress
- 只有 beneficial behaviors 被内化

这与 Snell 2022 的 context distillation 类似，但有两点关键不同：
1. EMPO² 是 online 而非 offline SFT
2. EMPO² 是 reward-guided 的（advantage 加权），而非 uniform distillation

## 5. 修改后的 Loss Function

GRPO 原始 loss（Eq. 1）：

$$\mathcal{L}_{GRPO} = \mathbb{E}_{\tau \sim \pi_{\theta_{old}}} \left[ \frac{1}{NT} \sum_{i,t} \min(\rho_\theta A, \text{clip}(\rho_\theta, 1-\epsilon, 1+\epsilon) A) \right] - \beta D_{KL}(\pi_\theta \| \pi_{ref})$$

变量：
- $\rho_\theta$：importance sampling ratio
- $A = A(a_t^{(i)})$：group-relative advantage
- $\epsilon$：clip range（实验用 0.2-0.3）
- $\beta$：KL 系数（ScienceWorld 设 0.0，WebShop 设 0.01）

EMPO² 修改后的 loss（Eq. 2）：

$$\mathcal{L}_{EMPO^2} = \mathbb{E}\left[\frac{1}{NT}\sum_{i,t} \min(\rho_\theta^{(i,t)} A, \text{clip}(\rho_\theta^{(i,t)}, 1-\epsilon, 1+\epsilon) A) \cdot \mathbf{1}_{\pi_\theta(a_t^{(i)}|s_t^{(i)},u) \geq \delta}\right] - \beta D_{KL}(\pi_\theta \| \pi_{ref})$$

新增 masking term $\mathbf{1}_{\pi_\theta(a_t|s_t, u) \geq \delta}$：当 student policy 给 action 的概率低于阈值 $\delta$ 时，mask 掉该 token 的 advantage 项。

**为什么需要 masking？**

参考 Yang et al. 2025 (https://arxiv.org/abs/2505.12929)：low-probability tokens 会通过 unbounded likelihood ratios 放大 gradient magnitude，导致训练 NaN collapse。Off-policy 训练特别脆弱，因为 $\rho_\theta$ 的分母 $\pi_{\theta_{old}}(a_t | s_t, u, \text{tips})$ 可能给某些 token 很高的概率，但 student $\pi_\theta(a_t | s_t, u)$ 给很低概率，ratio 爆炸。

Figure 6 展示了这一点：without masking 训练 collapse 到 NaN，with masking 稳定收敛。

## 6. Intrinsic Reward for Exploration

$$r_{intrinsic} = \frac{1}{n}$$

- $n$：与当前 state 相似的历史 state 数量
- Novel state（n 小）→ 高 intrinsic reward
- Visited state（n 大）→ 低 intrinsic reward

这是经典 count-based exploration (Bellemare 2016, https://arxiv.org/abs/1606.01868) 的简化版。Paper Appendix F.2 也测试了 RND (Random Network Distillation, https://arxiv.org/abs/1810.12894) 作为替代，效果类似。

Figure 7 显示：without intrinsic reward，policy entropy 快速 collapse；with intrinsic reward，entropy 维持，exploration 持续。

## 7. 完整 Algorithm 解析

Algorithm 1 的伪代码核心循环：

```
for each iteration:
    # Rollout phase
    sample B tasks, N envs each
    sample m_rollout: memory-augmented with prob p
    
    for t = 0 to T-1:
        if memory-augmented:
            tips_t = Retr(s_t; M)  # retrieval
            a_t ~ π_old(·|s_t, tips_t, u)
        else:
            a_t ~ π_old(·|s_t, u)
        execute a_t, observe r_t, s_{t+1}
    
    # Memory update (non-parametric)
    for each trajectory:
        tips ~ π_old(·|s, u, tip-gen-prompt)
        append tips to M
    
    # Policy update phase
    if rollout was memory-augmented:
        sample m_update: off-policy with prob q
        if off-policy:
            recompute log_prob WITHOUT tips  # 关键
            (replace log π_old(a|s, tips, u) with log π_old(a|s, u))
    
    update θ using Eq. 2
```

## 8. 实验数据深度分析

### 8.1 ScienceWorld 主结果（Table 1）

| Method | Average |
|---|---|
| Naive Qwen2.5-7B-Instruct | -61.3 |
| Reflexion (non-parametric) | 17.1 |
| Retrospex (offline RL) | 33.8 |
| GRPO (online RL) | 33.2 |
| **EMPO²** | **75.9** |

相对 GRPO 提升：$(75.9 - 33.2) / 33.2 = 128.6\%$ ✓

**值得关注的细节**：
- 7 个 task 从 negative reward 达到 100（满分）
- Power-component task：Naive -90.0 → EMPO² 94.3
- Find-living-thing：Naive -65.1 → EMPO² 100.0
- Retrospex 在某些 task 上比 Reflexion 还差（red 高亮），说明 offline RL 的 generalization 弱

### 8.2 WebShop 主结果（Table 2）

| Method | Score | Succ% |
|---|---|---|
| Naive | 26.4 | 7.8 |
| Reflexion | 58.1 | 28.8 |
| Retrospex | 73.1 | 60.4 |
| GRPO | 79.3 | 66.1 |
| GiGPO w/ std | 84.4 | 72.8 |
| GiGPO w/o std | 86.2 | 75.2 |
| **EMPO²** | **88.3** | **76.9** |

相对 GRPO：$(88.3 - 79.3)/79.3 = 11.3\%$ ✓

WebShop 提升比 ScienceWorld 小，因为 WebShop 是 shorter-horizon + clearer reward signal，exploration bottleneck 不那么严重。

### 8.3 OOD Adaptation（Figure 8）

这是 paper 最 exciting 的结果。训练后 model 在新 task 上 **只用 memory（no weight update）** 适应：

- Biology 1 → Biology 2（相似 transition）
- Biology 2 → Electricity（不同 topic）
- Electricity → Chemistry（完全不同）

10 steps 内平均 136% 改善。GRPO 在某些情况下甚至低于 base model，因为没有学会使用 memory。

**Intuition**：EMPO² 不只是学了一个 task 的 solution，而是学会了 "如何用 memory 探索新 task" 这个 meta-skill。这是 generalizable intelligence 的关键。

### 8.4 Ablation Studies

**Mode combination ablation（Figure 9）**：
- 移除 off-policy learning → 性能下降
- 移除 on-policy with memory → 性能下降
- 三者都需要：on-policy 提供 stability，off-policy 提供 distillation benefit

**Hyperparameter p（Figure 10a）**：
- $p=0.1$：collapse 到 GRPO 水平
- $p=0.25$：稳定收敛（默认）
- $p=0.4, 0.7$：早期学习更快，后期波动

**Hyperparameter q（Figure 10b）**：
- $q=0.3$：knowledge internalization 太慢
- $q=2/3$：默认，robust
- $q=0.85$：早期更快
- $q=0.95$：overemphasize distillation，伤害 memory policy 训练

**Intrinsic reward ablation（Figure 11）**：
- 移除 → plateau 在 lower level
- 0.5× / 2× scale → 主要影响 speed 不影响 final
- RND 替代 → 类似效果

## 9. Compute Cost 分析

Figure 12 显示 memory mechanism 增加 ~19% rollout time，主要来自 tip generation。

Figure 13 显示：即使按 wall-clock time 比较，EMPO² 仍显著优于 GRPO。Response length 也更长，说明 agent 在做更多 reasoning。

## 10. 联想与扩展思考

### 10.1 与 AlphaGo 的类比

AlphaGo 的 policy network 也是 self-play bootstrap：先 supervised learning from human expert，再 self-play RL。EMPO² 是类似的 bootstrap，但 "teacher" 是 memory-augmented policy，"student" 是 base policy。

### 10.2 与 STaR (Self-Taught Reasoner) 的关系

STaR (https://arxiv.org/abs/2203.14465) 用 model 自己生成 reasoning chain 作为 training data。EMPO² 的 tips 类似 STaR 的 rationale，但 tips 是 cross-trajectory 的（trajectory 之间共享），rationale 是 within-trajectory 的。

### 10.3 与 Voyager 的对比

Voyager (https://arxiv.org/abs/2305.16291) 在 Minecraft 中构建 skill library，也是 non-parametric memory + parametric learning。但 Voyager 的 skill 是 code，EMPO² 的 tip 是 verbal guidance。

### 10.4 与 Decision Transformer 的联系

Decision Transformer (https://arxiv.org/abs/2106.01345) 把 RL 转化为 sequence modeling。EMPO² 的 memory 也可以看作一种 "context" conditioning，类似 DT 的 return-to-go token。

### 10.5 Meta-learning 视角

EMPO² 实际上是在学一个 "learning to use memory" 的 meta-skill。OOD 实验证明了这个 meta-skill 的 transferability。这与 MAML (https://arxiv.org/abs/1703.03400) 的 fast adaptation 类似，但 EMPO² 是用 memory 而非 gradient steps 做 adaptation。

### 10.6 In-context Learning vs Weight-based Learning

EMPO² 的 hybrid 本质是在回答 "何时用 in-context learning（memory），何时把知识 bake 进 weights"。Off-policy update 是把 in-context knowledge 蒸馏到 weights 的桥梁。这与 Anthropic 的 In-Context Learning and Induction Heads (https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) 主题相关。

### 10.7 与 RLHF / RLAIF 的关系

RLAIF (https://arxiv.org/abs/2309.00267) 用 AI feedback 代替 human feedback。EMPO² 用 self-generated tips 作为 auxiliary signal，类似 self-RLAIF。这也呼应 Reflexion 的 "verbal RL" 概念。

### 10.8 Skill Library / Replay Buffer 的 RL 经典方法

EMPO² 的 memory 与 prioritized experience replay (https://arxiv.org/abs/1511.05952) 在 spirit 上类似，但 EMPO² 存的是 verbal reflection 而非 raw transition。

### 10.9 与 Curriculum Learning 的可能结合

Paper Section 7 提到未来工作包括 math, coding, multi-hop QA。这些 domain 天然有 curriculum structure，EMPO² 的 memory 可能自动 emerge 出 curriculum signal。

### 10.10 与 Constitutional AI 的对比

Constitutional AI (https://arxiv.org/abs/2212.08073) 用 self-critique + self-revision。EMPO² 的 tip generation 是类似的 self-critique，但 tip 是 cross-episode 的，CAI 的 critique 是 within-episode 的。

## 11. Paper 的 Limitations & Open Questions

1. **Memory retrieval 简单**：只用 cosine similarity + threshold。可以用 learned retriever (e.g., DPR-style)
2. **7B 模型限制**：scaling laws 未知
3. **Domain 限制**：只在 ScienceWorld + WebShop 测试
4. **Off-policy 稳定性**：依赖 masking，threshold δ 需要调
5. **Memory buffer 大小固定 1000**：没有探索更 sophisticated 的 forgetting 机制
6. **Tip 质量 vs 数量 trade-off** 没有深入分析

## 12. 我的 Intuition Summary

EMPO² 的核心 insight 可以浓缩为：

> **Memory is the scaffolding, weights are the destination.**

具体机制：
- Memory 提供 **cross-trajectory information continuity**，解决 GRPO 的 "amnesia" 问题
- Off-policy update 是 **reward-weighted distillation**，把 memory benefit bake 进 weights
- On-policy with memory 维持 **distribution consistency**，防止训练 collapse
- Intrinsic reward 维持 **exploration entropy**，防止 premature convergence

这个 paper 最 elegant 的地方在于：它没有发明新的 RL 算法，而是巧妙组合了三个 ideas（self-reflection memory + importance sampling + intrinsic reward），用 LLM 的语言能力作为 glue，解决了一个具体的 bottleneck（exploration in LLM agent RL）。

**Key reference links**:
- Project page: https://github.com/agent-lightning/empo2
- Agent Lightning: https://arxiv.org/abs/2508.03680
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- verl framework: https://arxiv.org/abs/2409.19256
- GiGPO: https://arxiv.org/abs/2505.10978
- ScienceWorld: https://arxiv.org/abs/2203.07540
- WebShop: https://arxiv.org/abs/2207.01206
- Reflexion: https://arxiv.org/abs/2303.11366
- Go-Explore: https://arxiv.org/abs/1901.10995
- Random Network Distillation: https://arxiv.org/abs/1810.12894
- Context Distillation: https://arxiv.org/abs/2209.15189
- Low-prob token issue: https://arxiv.org/abs/2505.12929
- Qwen2.5: https://arxiv.org/abs/2412.15115

这个 paper 对你（Karpathy）的 micrograd / nanoGPT 视角下，可以理解为：**用 non-parametric memory 做 "forward pass 之间的 gradient accumulation"**——memory 在 trajectory 之间 carry information，weights 在 trajectory 之内做 gradient update。两者协同形成 amortized learning。
