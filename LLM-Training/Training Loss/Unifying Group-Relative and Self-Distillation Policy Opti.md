---
source_pdf: Unifying Group-Relative and Self-Distillation Policy Opti.pdf
paper_sha256: ac3fc88db1c7fd25b6a8009dd46c8ca17f2c10479cea73f0a199acd6f46412e7
processed_at: '2026-08-12T19:50:40-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 SRPO

---

## 1. 这篇 paper 在解决什么"人话版"问题

想象你是个老师，手头有一班学生（policy），你要让他们做练习题（rollouts）然后给反馈。

**GRPO 的做法**：每道题让 8 个学生做，做完后你只告诉每个学生"你做对了/做错了"，并且在 8 个人里排名。做对的就表扬（正 advantage），做错的就批评（负 advantage），但是批评的时候你不会指出他到底错在哪一步 —— 整个解题过程都被 uniformly 批评。

这种方式简单粗暴但是 effective，因为大多数情况下学生知道"这条路是对的"，但效率不高 —— 一个学生可能在第三步就走错了，但后面 5 步都被批评，他不知道该 focus 改哪。

**SDPO 的做法**：你让一个做对的学生把自己的解题过程写出来（teacher info），然后让做错的学生对照着这个答案，逐字逐句（token-by-token）地学习。这比 GRPO 精准多了 —— 每个位置都有明确的 "correct distribution" 可以学。

但是 SDPO 有两个问题：

1. **对已经做对的学生也强制这么做**：如果你本来做对了，老师还让你去模仿另一个做对的同学的写法，你会很懵 —— "为什么我非要学他的写法？我自己的写法也对啊！" 这就是 paper 说的 **optimization ambiguity**。

2. **老师越来越糊涂**：这个 "做对的学生" 其实是同一个 policy 的 EMA 版本，随着 training 推进，policy 在变强，但 self-teacher 也在变，两者 gap 越来越小，老师越来越像学生自己 —— 一个糊涂的老师教学生，学生也跟着糊涂。Figure 1(c) 直接显示 teacher 的 entropy 在上升，就是说老师在每个位置越来越 "不确定"。

---

## 2. SRPO 的核心 idea（一句话版）

**把做对的样本送给 GRPO 处理，把做错但有"参考答案"的样本送给 SDPO 处理**。同时，在 SDPO 这边，老师越不确定的位置，越不要听老师的。

这其实就是个 **conditional routing** —— 根据 sample 的 state 决定用哪种 optimization signal。这跟你做 software engineering 时根据 input type 选不同 handler 是一回事。

---

## 3. 为什么这个 routing 是合理的？三个直觉

### Intuition 1：做对的样本不需要 dense supervision

如果一个 rollout 已经 correct 了，说明它的 reasoning path 是 reward-aligned 的。这时候用 sequence-level advantage（GRPO）就够了 —— 整条路径都该被 reinforce，具体哪个 token 该多 reinforce 不是 reward signal 能告诉你的，强行让 model 去学另一个 correct sibling 的 token 分布反而引入 noise。

这跟 classic RL 里的 "don't fix what isn't broken" 一致 —— sparse reward 在 success case 下是 sufficient 的，dense supervision 反而会 overfit 到 spurious details。

### Intuition 2：做错的样本需要 dense supervision

如果一个 rollout 错了，sequence-level 的 "你做错了" 没什么用 —— policy 不知道是哪一步走错的。这时候如果有 sibling solution 做 reference，token-level 的 "在这一步你应该倾向于选这些 token" 能精准定位错误位置。

这跟 code debugging 一样 —— "你这代码跑不通" vs "你这代码第 47 行的 `==` 应该是 `>=`" 差别巨大。

### Intuition 3：老师的"不确定"不应该被学生学

Self-teacher 在某个位置如果 entropy 很高（比如在 5 个 token 间犹豫），它给的 logit distribution 是 noisy 的。学生强行去 match 这个 noisy distribution，相当于在学 noise。Dynamic weighting 就是说 "老师不确定的地方，学生少学点；老师确定的地方，学生多学点"。

这跟 KD (knowledge distillation) 里调 temperature 是类似的思路 —— 但 temperature 是 global 的，dynamic weighting 是 per-token 的，更 fine-grained。

---

## 4. 实现细节用大白话讲

### Routing mask

每个 rollout 算两个 flag：

- $c_i$：做对了吗？（reward ≥ 0.5 算对）
- $m_i$：有老师吗？（group 里至少有一个 sibling 是 correct，并且不是自己）

然后：
- 错了 + 有老师 → SDPO branch
- 其他情况 → GRPO branch

注意一个 corner case：如果你是 group 里唯一做对的，你不能做自己的老师（避免 self-leakage），所以 $m_i = 0$，但你本来就走 GRPO，没问题。

### DW-SDPO 的权重

每个位置 $t$ 算 teacher 的 entropy $H_{i,t}$：

$$\tilde{w}_{i,t} = \exp(-\beta H_{i,t})$$

变量：
- $H_{i,t}$：teacher 在位置 $t$ 上的 entropy（越大越不确定）
- $\beta$：敏感度，论文用 $\beta=1$（default）
- $\tilde{w}_{i,t}$：未归一化权重，teacher 越 confident 权重越大

然后 normalize 一下让总 loss scale 不漂移：

$$w_{i,t} = \frac{\tilde{w}_{i,t}}{\frac{1}{|\Omega_{\text{sdpo}}|}\sum_{(j,s) \in \Omega_{\text{sdpo}}} \tilde{w}_{j,s}}$$

变量：
- $\Omega_{\text{sdpo}}$：所有被 routed 到 SDPO 的 (rollout, position) pair 集合
- $|\Omega_{\text{sdpo}}|$：这个集合的 size
- 分母是 mean，不是 sum —— 这个选择保证了无论 SDPO branch 处理多少 token，整体 loss 的 magnitude 不会被自动放大或缩小

### 最终 loss

$$\mathcal{L}_{\text{final}} = \frac{\sum_{i,t} z_i^{\text{GRPO}} \ell_{i,t}^{\text{GRPO}} + \sum_{i,t} z_i^{\text{SDPO}} \ell_{i,t}^{\text{DW-SDPO}}}{\sum_{i,t} z_i^{\text{GRPO}} + \sum_{i,t} z_i^{\text{SDPO}}}$$

变量：
- $z_i^{\text{GRPO}}, z_i^{\text{SDPO}}$：routing mask，每个 rollout 只有一个是 1
- $\ell_{i,t}^{\text{GRPO}}$：GRPO 的 token-level loss（sequence advantage 均分到 token 上）
- $\ell_{i,t}^{\text{DW-SDPO}} = w_{i,t} \ell_{i,t}^{\text{SDPO}}$：加权后的 SDPO token loss
- 分母：总 routed token 数，做归一化

这个归一化设计有个 emergent property —— **不需要 tune mixing ratio**。早期 failed sample 多，SDPO branch 自然占大头；后期 correct sample 多，GRPO branch 自然占大头。Mixing 是 sample distribution 驱动的，自动 adapt。

---

## 5. 实验结果用大白话讲

### 主结果（Table 1）

Qwen3-8B 上 5 个 benchmark 平均（10h budget）：

- Base：49.5
- GRPO：74.0
- SDPO：71.1（比 GRPO 还低 —— 因为 late-stage collapse 拖了后腿）
- **SRPO：77.4**（比 GRPO +3.4，比 SDPO +6.3）

SRPO 在所有 5 个 benchmark 上都不差于 GRPO 和 SDPO，在 Chemistry、Physics、Biology、Materials 上明显更好，Tool Use 上和 GRPO 持平。

这说明 SRPO 的 gain 不是 task-specific 的 —— 它既能享受 SDPO 在 dense reasoning task 上的优势，又能避免 SDPO 在 Tool Use 上的 collapse（Figure 3(c) 显示 SDPO 在 Tool Use 上后期一路退化，SRPO 跟住 GRPO）。

### Ablation 两个关键发现

**1. Sample routing > advantage mixing**

对比 SRPO w/o dynamic weighting vs Advantage Mix（在 advantage 层做凸组合）：
- 1h：Advantage Mix 略好（+0.7）
- 5h：Sample routing 反超（-2.5）
- 10h：Sample routing 优势扩大（-3.3）

为什么？Advantage mixing 在每个 token 上都同时应用 GRPO 和 SDPO signal，SDPO 的 noise 会渗透到 correct samples 上。Sample routing 是 disjoint partition —— 一个 sample 要么走 GRPO 要么走 SDPO，物理隔离。

**2. Dynamic weighting 提供 late-stage gain**

SRPO vs SRPO w/o dynamic weighting：
- 1h：+0.4
- 5h：+0.7
- 10h：+1.8

Gain 随训练扩大 —— 因为后期 teacher entropy 上升，dynamic weighting 抑制 noise 的作用越来越大。这跟 Diagnosis 2 完全自洽。

### Response length 的故事（Figure 4a）

GRPO 生成越来越长（verbose，inference 贵）；SDPO 生成越来越短（过度压缩，损害 epistemic reasoning，参见 [Kim et al. 2026](https://arxiv.org/abs/2603.24472)）；SRPO 居中 —— 早期接近 SDPO 的短，后期慢慢稳定到 GRPO 和 SDPO 之间。

这其实是 SRPO 的一个 hidden benefit —— response length 适中既省钱又不损害 reasoning quality。

### Compute cost 的故事（Figure 4b）

早期 SRPO 比 GRPO 慢 17.4%（额外 self-teacher forward pass）；10h 时 SRPO 比 GRPO 快 17.2%，比 SDPO 快 9.4%。

为什么后期变快？两个原因：
1. Failed sample 减少 → SDPO branch 激活频率降低 → self-teacher forward 减少
2. SRPO 的 response 比 GRPO 短 → inference 更快

这是一个 **self-stabilizing compute profile** —— 早期多花点钱买 correction，后期自动降本。

---

## 6. 我看完 paper 后的几个直觉

### Intuition A：SRPO 是 "adaptive curriculum" 的一种形式

早期 policy 弱，failed sample 多，dense correction 有用 → SDPO branch 主导；后期 policy 强，correct sample 多，reward signal 足够 → GRPO branch 主导。这是一种 implicit curriculum，由 sample distribution 驱动，不需要手动 schedule。

这跟 [Hübotter et al. 2025 "Learning on the Job"](https://arxiv.org/abs/2510.04786) 的 test-time curriculum 思路类似 —— 都是用 policy 的 current state 决定 optimization signal。SRPO 的 routing 是 sample-level 的 "learning on the job"。

### Intuition B：Dynamic weighting 是 token-level attention over teacher signal

$\exp(-\beta H)$ 这个 form 在 physics 里就是 Boltzmann distribution，在 ML 里就是 attention weight。把 teacher 的 entropy 当作 "energy"，low entropy = low energy = high weight。这其实是一种 self-attention —— model 在决定 "该听老师的哪些话"。

对比一下：
- Soft attention in transformer：$\text{softmax}(QK^T/\sqrt{d})$ 决定该 attend to 哪些 token
- Dynamic weighting in SRPO：$\exp(-\beta H)$ 决定该 attend to teacher 的哪些 positions

这是一个很自然的 design pattern —— 当 supervision signal 质量不均匀时，用 confidence 做 weighting。

### Intuition C：Routing 是 "signal gating" 思想的应用

RL post-training 里的核心问题：**用 sparse reward signal 还是 dense teacher signal？** 这两者是 complement，但 mixing 是 art：

- 全用 reward signal（GRPO）：稳定但稀疏，correct samples 上浪费了 dense supervision 的机会
- 全用 teacher signal（SDPO）：dense 但可能 noisy，correct samples 上引入 ambiguity
- 固定 mixing ratio：无法 adapt 到 sample distribution 变化
- Sample routing：根据 sample state 自适应选 signal —— best of both worlds

这跟 Mixture of Experts (MoE) 的思想是通的 —— 用 gating 决定哪个 expert 处理哪个 input。SRPO 是把 MoE 的思想应用到 optimization signal 选择上。

### Intuition D：EMA teacher 是 "slow student"

Self-teacher 是 student policy 的 EMA（update rate 0.05），意思是 teacher 每 20 步才"追上" student 一次。这有两个 effect：

1. **Quasi-stationary target**：teacher 比 student "慢一拍"，提供相对稳定的 distillation target（类似 DQN 里的 target network）。
2. **Information gap**：teacher 永远比 student 略 outdated，所以 student 能从 teacher 学到东西（如果完全同步，distillation 就 degenerate 了）。

但 EMA 也是 paper Diagnosis 2 的根源 —— gap 单调缩小，teacher entropy 上升，distillation signal 稀薄。Dynamic weighting 是对这个问题的一个 mitigation，但没有根本解决。

如果想根本解决，可以考虑：
- 外部 teacher（GPT-4、Claude）替代 self-teacher，但 cost 高
- Verifier-based teacher（PRM）替代 sibling-based teacher，但需要 train extra model
- Multi-teacher ensemble，降低单 teacher 的 entropy

### Intuition E：Why SRPO works in science but ties in Tool Use？

从 Figure 3 看，SRPO 在 Chemistry、Biology 上明显超 GRPO（SDPO 也强），但在 Tool Use 上 SRPO 跟 GRPO 持平、SDPO 崩了。

我的解读：science reasoning 任务里 reasoning chain 长，token-level 错误的 cost 高（一步错全错），dense correction 价值大；Tool Use 任务里答案空间相对小（通常是特定的 API call format），sequence-level reward 已经足够定位对错，dense correction 帮助不大甚至会引入 noise（teacher 在 format token 上 entropy 低，在 reasoning token 上 entropy 高，但 reasoning 在 tool use 里不重要）。

这暗示 SRPO 的 gain 是 **task-dependent** 的 —— 在 "reasoning-dense" 任务上更有价值。如果要扩展到 math、code 等其他 reasoning task，可以预测 SRPO 在 math 上应该有效（reasoning chain 长），在 simple QA 上可能 gain 有限。

---

## 7. 我看到的几个潜在 issue

### Issue 1：当 group 里只有 1 个 correct rollout 时

7 个 failed rollout 共享 1 个 sibling 作为 teacher。如果这个 sibling 走了 "lucky path"（碰巧对），7 个 failed sample 都被蒸馏向一个 sub-optimal path。Routing 没有机制过滤 "sub-optimal but correct" sibling。

一个可能的改进：对每个 failed sample，不只用一个 sibling，而是用所有 correct siblings 的 ensemble 作为 teacher（降低单 sibling 的 noise）。

### Issue 2：Late training 时 SDPO branch "forget how to correct"

Figure 5 显示 SDPO branch 激活频率后期降到很低（~10%）。如果突然遇到 OOD difficult prompt（group 里全错或大部分错），SDPO branch 能否迅速激活？还是它已经 "forget" 了如何做 dense correction？

这跟 continual learning 里的 catastrophic forgetting 类似 —— 一个 branch 长期不被激活，可能 lose capability。Paper 没有测试这种 robustness scenario。

### Issue 3：$\beta$ 的 sensitivity 没分析

Dynamic weighting 的 $\beta$ 用 default = 1，但没有 sensitivity analysis。如果 $\beta \to \infty$，退化为 hard masking（只用 entropy = 0 的 token，可能太稀）；如果 $\beta \to 0$，退化为 vanilla SDPO。$\beta$ 应该是 task-dependent 的，paper 没探索这个。

### Issue 4：Correct samples 上 GRPO 仍是 coarse

SRPO 把 correct samples 路由到 GRPO，但 GRPO 仍是 sequence-level advantage。如果 correct samples 上也能用 dense signal（比如 process reward model），SRPO 可以进一步细化。

这是 SRPO + PRM 的自然 extension —— 但 PRM 本身有自己的 noise 问题，需要单独研究。

### Issue 5：Multi-turn agentic extension

当前实验是 single-turn（一次 generation 得 reward）。Multi-turn agentic（tool calling、code execution）中，每个 turn 都可以做 routing。SRPO 的 framework 应该可以扩展，但需要解决跨 turn credit assignment。

比如：一个 multi-turn rollout 整体失败了，但前 3 turn 正确、第 4 turn 错。SRPO 应该把前 3 turn 当 "correct" route 到 GRPO？还是整体作为 failed route 到 SDPO？这个设计需要仔细考虑。

---

## 8. 一张图总结 SRPO 的 mental model

```
For each prompt x:
  Sample G rollouts {y_1, ..., y_G} from current policy
  Get rewards {r_1, ..., r_G}
  
  For each rollout y_i:
    ┌─────────────────────────────────────────────┐
    │  Is y_i correct? (c_i = 1)                  │
    │  OR                                          │
    │  No teacher info available? (m_i = 0)       │
    └─────────────────────────────────────────────┘
         │                          │
         │ Yes                      │ No (incorrect + has sibling)
         ▼                          ▼
  ┌─────────────┐           ┌─────────────────┐
  │ GRPO Branch │           │ SDPO Branch     │
  │             │           │                 │
  │ Sequence    │           │ Self-teacher    │
  │ advantage   │           │ + DW weighting  │
  │ (coarse)    │           │ (dense, gated)  │
  └─────────────┘           └─────────────────┘
         │                          │
         └──────────┬───────────────┘
                    ▼
         ┌──────────────────┐
         │ Combined Loss    │
         │ (normalized by   │
         │  routed tokens)  │
         └──────────────────┘
                    │
                    ▼
         Update policy θ
```

核心：根据 sample 的 "health state" 决定用哪种"疗法" —— 健康的用 sparse reward 维持，生病的用 dense correction 治疗但只听靠谱医生的话。

---

## 9. 一些容易忽视的细节

### 9.1 SGLang 替代 vLLM

Appendix B.1 提到用 SGLang 替代 vLLM 做 inference backend。理由是与环境兼容性更好。但 paper 强调这不影响 sampling fairness，因为两个 backend 实现相同的 sampling algorithm。这是负责任的工程实践 —— 避免被 reviewer 质疑 inference engine 影响结果。

### 9.2 Learning rate 选在 GRPO 和 SDPO 中间

SRPO 的 lr = $5 \times 10^{-6}$，刚好是 GRPO ($10^{-6}$) 和 SDPO ($10^{-5}$) 的几何平均附近。这个选择是为了 balance 两个 branch 的 gradient scale —— GRPO advantage 通常较小，SDPO KL gradient 较大，lr 取中间值避免一边 dominate。

但 paper 没有 ablation 这个 lr 选择。考虑到两个 branch 的 gradient magnitude 在不同 stage 会变化（早期 SDPO 主导，后期 GRPO 主导），可能需要 adaptive lr schedule。

### 9.3 Asymmetric clipping 借鉴 DAPO

GRPO branch 用 $\varepsilon_{\text{high}} = 0.28$（比 $\varepsilon_{\text{low}} = 0.2$ 大），这是 [DAPO (Yu et al. 2025)](https://arxiv.org/abs/2503.14476) 的设计，用来缓解 entropy collapse。说明 paper 是站在 best practice 之上的，不是从头造轮子。

### 9.4 Jensen-Shannon divergence 而非 KL

SDPO branch 用 JS divergence 做 distillation，不是 forward 或 reverse KL。JS 是对称的，bounded，避免 KL 的极端行为（forward KL mode-covering，reverse KL mode-seeking）。这个选择对训练稳定性的影响 paper 没单独 ablation，但经验上 JS 更 robust。

### 9.5 Top-K = 100 distillation

只对 top-100 logits 做 distillation，避免长尾 vocab 噪声。这跟 MiniLLM 等工作的做法一致 —— 长 tail 的 logits 通常 noisy 且不重要，focus 在 top-K 能提升 signal-to-noise ratio。

### 9.6 No thinking mode

Qwen3 的 thinking mode 在实验里关闭。所以 SRPO 的 gain 是在 non-CoT setting 下测的。如果在 thinking mode 下，每一步 thinking 都是 explicit decision point，SDPO 的 dense logit signal 可能效果不同 —— 可能更有用（每步都重要），也可能更 noisy（thinking 的 diversity 本身是有价值的）。

---

## 10. 最后用一句话总结

SRPO 说的是：**RLVR post-training 里，不要一刀切地用 sparse reward 或 dense distillation —— 让数据自己说话，做对的用 sparse reward 锁住，做错的用 dense correction 纠正，但只信靠谱的 correction**。

这是一个简洁、robust、可扩展的设计，核心 idea 可以迁移到很多其他 RL+distillation 的场景。

---

## Reference 链接

- [SRPO paper (推测 ID，需核实)](https://arxiv.org/abs/2606.07894)
- [GRPO - DeepSeekMath (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)
- [SDPO - Self-Distillation RL (Hübotter et al., 2026)](https://arxiv.org/abs/2601.20802)
- [Learning on the Job (Hübotter et al., 2025)](https://arxiv.org/abs/2510.04786)
- [Why Self-Distillation Degrades Reasoning (Kim et al., 2026)](https://arxiv.org/abs/2603.24472)
- [DAPO (Yu et al., 2025)](https://arxiv.org/abs/2503.14476)
- [PRIME - Process Reinforcement (Cui et al., 2025)](https://arxiv.org/abs/2502.01456)
- [PRM (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050)
- [On-policy Distillation (Agarwal et al., 2024)](https://arxiv.org/abs/2402.04616)
- [Thinking Machines - On-policy Distillation](https://thinkingmachines.ai/blog/onpolicy-distillation)
- [Learning by Distilling Context (Snell et al., 2022)](https://arxiv.org/abs/2209.15189)
- [MiniLLM (Gu et al., 2023)](https://arxiv.org/abs/2306.08543)
- [Hinton KD (2015)](https://arxiv.org/abs/1503.02531)
- [Causal Credit Assignment (Khandoga et al., 2026)](https://arxiv.org/abs/2602.09331)
- [GRPO-λ (Parthasarathi et al., 2025)](https://arxiv.org/abs/2510.00194)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [OpenAI o1 System Card](https://arxiv.org/abs/2412.16720)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [SciKnowEval (Feng et al., 2024)](https://arxiv.org/abs/2406.09098)
- [ToolAlpaca (Tang et al., 2023)](https://arxiv.org/abs/2306.05301)
- [SGLang](https://arxiv.org/abs/2312.07104)
- [verl framework](https://arxiv.org/abs/2409.19256)
- [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)

如果想继续聊任何一个细节（比如 EMA teacher 的数学推导、JS vs KL 的 trade-off、扩展到 multi-turn agentic 的具体设计、跟 PRM 结合的可能性），随时问。

---

# SRPO 深度解读：用 Sample Routing 统一 GRPO 与 SDPO

Andrej，这篇 paper 处理的是当前 RLVR (Reinforcement Learning with Verifiable Rewards) post-training 阶段最核心的一对 tension —— **reward-driven coarse credit assignment** vs **teacher-driven dense logit-level supervision**。核心 idea 简洁到几乎有 18-20 世纪 control theory 的味道：根据 sample 的 correctness 状态把它 route 到最适合的 optimization signal。下面我尽量把背后的机制、公式细节、诊断、设计选择都展开，并补一些相关的联想链。

---

## 1. 问题背景：为什么需要 unifying GRPO 与 SDPO？

当前 LLM post-training 的主流范式是 RLVR（DeepSeek-R1, OpenAI o1, Kimi K1.5, Qwen3 等）。在这之下，credit assignment 粒度是一个未被解决的问题：

- **GRPO (Group Relative Policy Optimization)** ([Shao et al., 2024](https://arxiv.org/abs/2402.03300))：对每个 prompt $x$ 采样一组 $G$ 个 rollouts $\{y_i\}_{i=1}^G$，把 outcome reward 做组内 normalize 得到一个 sequence-level scalar advantage。优点是简单、stable、critic-free。缺点是 advantage uniformly 分配给每个 token，failed rollout 上无法定位"哪一步走错了"。
  
- **SDPO (Self-Distillation Policy Optimization)** ([Hübotter et al., 2026](https://arxiv.org/abs/2601.20802))：构造一个 feedback-conditioned self-teacher $\pi_\theta(\cdot|x,f)$（$f$ 是 sibling rollout 或 environment feedback），用 logit-level KL 散度做 dense supervision。早期收敛快，但 late stage 会 collapse。

paper 的 Figure 1(a) 在 Chemistry + Qwen3-8B 上给出非常 clean 的诊断曲线：SDPO 在 1h 时领先 GRPO，约 5h 时被反超，10h 时彻底 collapse；GRPO 后期稳定但 plateau 较低。

这种"早期快后期崩"的 pattern 在 Kim et al. (2026) 的 [Self-distillation degrades reasoning](https://arxiv.org/abs/2603.24472) 中也有讨论，他们归因为 epistemic verbalization 被压制。SRPO 提供了一个 **complementary** 的诊断视角（这点很重要，paper 写得很谨慎，没有断言自己的诊断是 exclusive 的）。

---

## 2. SRPO 的两个核心诊断（这是论文的真正 contribution）

### Diagnosis 1：对 correct samples 做 self-distillation 会引入 optimization ambiguity

考虑 SDPO 的 update：student 被强制去 match 一个 sibling rollout（同样 correct）的 logit 分布。但两条 reasoning path 的 outcome reward 是 equivalent 的（都得到 reward = 1），强行让一条 path 的 logits match 另一条，相当于施加了一个 **arbitrary logit-level preference**。这相当于在两个等价 local optima 之间做无意义的 pull。

Figure 1(b) 的 ablation 直接验证：
- SDPO only on incorrect samples：retain 大部分收益
- SDPO only on correct samples：性能 degrade 且训练不稳定

这本质上是一个 **reward-equivalent trajectory collapsing** 的问题。GRPO 在这里反而更合理 —— sequence-level advantage 对 correct rollout 给正 advantage，rollout 内 token 间的具体分布由 policy 自己决定。

### Diagnosis 2：self-teacher 的 distillation signal quality 退化

随着 training 推进，self-teacher 和 student 的 gap 缩小（EMA 参数让 teacher 缓慢跟随 student），distillation signal 信息量递减。更关键的是 Figure 1(c)：**teacher 的 token-level entropy 在训练中上升**，意味着 distillation target 越来越 noisy。

这是 self-distillation 的一个内在问题 —— 没有 external anchor，teacher 的 uncertainty 会渗透到 student。这与经典 knowledge distillation ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)) 中强 teacher 的 "dark knowledge" 形成对比：在那里 teacher 的 softmax temperature 提供了类间相对信息；而 self-teacher 随着自身能力被 student 追上，"dark knowledge" 越来越稀薄。

---

## 3. SRPO 框架细节

### 3.1 Sample-Level Routing

定义两个 binary indicator：
- $c_i = \mathbf{1}[y_i \text{ is correct}]$：correctness flag
- $m_i = \mathbf{1}[\text{teacher info available for } y_i]$：teacher availability flag

Routing mask：

$$z_i^{\text{SDPO}} = (1 - c_i) \cdot m_i, \quad z_i^{\text{GRPO}} = 1 - z_i^{\text{SDPO}}$$

变量含义：
- $c_i \in \{0,1\}$：第 $i$ 个 rollout 是否 correct（reward $\geq 0.5$ 视为 correct）
- $m_i \in \{0,1\}$：第 $i$ 个 rollout 是否有可用的 teacher info（即 group 中至少有一个 sibling 是 correct 的，且不是 $y_i$ 自己）
- $z_i^{\text{SDPO}}$：路由到 SDPO branch 的 mask
- $z_i^{\text{GRPO}}$：路由到 GRPO branch 的 mask

**Decision matrix（Table 7）**：

| $c_i$ | $m_i$ | Teacher prompt | Route |
|-------|-------|----------------|-------|
| 1 | 1 | Question + sibling solution | GRPO |
| 1 | 0 | Question only | GRPO |
| 0 | 1 | Question + sibling solution | SDPO |
| 0 | 0 | Question only | GRPO (fallback) |

这里有一个 subtle 的设计：当 rollout 自己是 group 中唯一 correct 的，它不能做自己的 teacher（避免 leakage），所以 $m_i = 0$，但仍走 GRPO（因为它本就 correct）。

### 3.2 GRPO 与 SDPO 的 advantage 形式统一

paper 一个 elegant 的处理是把两个方法都写成 policy gradient form：

GRPO gradient：
$$\nabla_\theta \mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(y_t | x, y_{<t}) \cdot A_i^{\text{GRPO}}\right]$$

其中 $A_i^{\text{GRPO}} = (r_i - \bar{r}) / (\sigma_r + \epsilon)$：
- $r_i$：rollout $i$ 的 scalar reward
- $\bar{r} = \frac{1}{G}\sum_{i=1}^G r_i$：group mean reward
- $\sigma_r = \sqrt{\frac{1}{G}\sum_i (r_i - \bar{r})^2}$：group std
- $\epsilon$：数值稳定项（避免 $\sigma_r = 0$ 时除零）
- $A_i^{\text{GRPO}}$ 在同一个 rollout 内对所有 token 是常量 → coarse credit assignment

SDPO gradient（[Hübotter et al., 2026](https://arxiv.org/abs/2601.20802) 推导）：
$$-\nabla_\theta \mathcal{L}_{\text{SDPO}} = \mathbb{E}\left[\sum_t \sum_{v \in \mathcal{V}} \nabla_\theta \log \pi_\theta(v | x, y_{<t}) \cdot A_t^{\text{SDPO}}(v)\right]$$

变量：
- $\mathcal{V}$：vocabulary 集合
- $v$：候选 token
- $A_t^{\text{SDPO}}(v)$：token $t$ 位置上 token $v$ 的 logit-level advantage，由 self-teacher 分布与 student 分布的 discrepancy 诱导
- 注意这里 advantage 是 $(t, v)$ 二维的 → dense credit assignment

paper 的关键 framing：**两者都是 advantage estimator，只是粒度和来源不同**。GRPO 是 reward-derived + sequence-level；SDPO 是 teacher-derived + logit-level。Routing 在保持 policy gradient 形式不变的前提下，给每个 sample 选合适的 advantage estimator。

### 3.3 Dynamic-Weighted SDPO (DW-SDPO)：entropy-aware 加权

这是针对 Diagnosis 2 的直接修复。设：

$$q_{i,t}(v) = \pi_\theta(v | x, f_i, y_{i,<t})$$

为 self-teacher 在 rollout $i$ 的位置 $t$ 上的 vocab 分布。其 entropy：

$$H_{i,t} = -\sum_{v \in \mathcal{V}} q_{i,t}(v) \log q_{i,t}(v)$$

unnormalized weight：

$$\tilde{w}_{i,t} = \exp(-\beta H_{i,t})$$

变量：
- $\beta > 0$：temperature，控制对 entropy 的敏感度（论文 $\beta = 1$）
- 当 $H_{i,t} \to 0$（teacher 很 confident）：$\tilde{w}_{i,t} \to 1$（权重满）
- 当 $H_{i,t} \to \log|\mathcal{V}|$（teacher uniform）：$\tilde{w}_{i,t} \to 1/|\mathcal{V}|^\beta$（权重小）

normalized weight：

$$w_{i,t} = \frac{\tilde{w}_{i,t}}{\frac{1}{|\Omega_{\text{sdpo}}|} \sum_{(j,s) \in \Omega_{\text{sdpo}}} \tilde{w}_{j,s}}$$

变量：
- $\Omega_{\text{sdpo}}$：所有路由到 SDPO branch 的有效 $(i, t)$ pair 的集合
- 分母用 mean 而非 sum 来 normalize，保证整体 loss scale 大致守恒
- 这个设计避免了引入额外的 mixing hyperparameter（与 Advantage Mix 对比时这点很重要）

这个权重形式其实非常类似 **softmax with negative-entropy logits**。它的几何意义：在 token space 上做一个 soft gating，把梯度质量从 high-entropy region 重新分配到 low-entropy region。对比经典 KD 中的 temperature $T$ 调节，这是 finer-grained 的 per-position 调节。

### 3.4 最终 loss

$$\mathcal{L}_{\text{final}} = \frac{\sum_{i,t} z_i^{\text{GRPO}} \ell_{i,t}^{\text{GRPO}} + \sum_{i,t} z_i^{\text{SDPO}} \ell_{i,t}^{\text{DW-SDPO}}}{\sum_{i,t} z_i^{\text{GRPO}} + \sum_{i,t} z_i^{\text{SDPO}}}$$

变量：
- $\ell_{i,t}^{\text{GRPO}}$：token-level GRPO loss（sequence-level advantage 均分到 response token 上）
- $\ell_{i,t}^{\text{DW-SDPO}} = w_{i,t} \ell_{i,t}^{\text{SDPO}}$：weighted SDPO token loss
- 分母：所有 routed token 数量，做归一化以避免引入额外 mixing hyperparameter

这个设计有一个 emergent behavior：**early training** failed samples 多，SDPO branch token 占比高（~40% from Figure 5）；**late training** correct samples 多，GRPO branch 占比上升。整个 mixing ratio 是 sample distribution 自然驱动的，不需要 schedule。

---

## 4. 实验数据深度分析

### 4.1 Main results (Table 1)

Qwen3-8B（avg@16，10h budget）：

| Method | Chemistry | Physics | Biology | Materials | Tool Use | Avg |
|--------|-----------|--------|---------|-----------|----------|-----|
| Base | 41.1 | 58.7 | 30.5 | 59.3 | 57.9 | 49.5 |
| GRPO | 78.9 | 73.6 | 68.1 | 77.6 | 71.8 | 74.0 |
| SDPO | 80.6 | 74.0 | 58.5 | 76.6 | 65.7 | 71.1 |
| **SRPO** | **83.0** | **80.6** | **72.8** | **81.5** | **71.2** | **77.4** |

SRPO vs GRPO：+4.1 / +7.0 / +4.7 / +3.9 / -0.6（Tool Use 上 SRPO ≈ GRPO，符合 Pattern 2）

SRPO vs SDPO：+2.4 / +6.6 / +14.3 / +4.9 / +5.5（Biology 上 SRPO 几乎是 SDPO 的 1.24 倍）

Qwen3-4B 趋势一致，平均 +7.5 over SDPO，+4.5 over GRPO，说明 scaling 不改变 behavior pattern。

### 4.2 两个 Pattern（Section 4.2，Figure 3）

**Pattern 1**：当 self-distillation 有效时，SRPO 延续优势。Chemistry & Biology：SDPO 早期快，但 SRPO 5h 后超越并保持上升，10h 达到 peak。

**Pattern 2**：当 self-distillation 无效时，SRPO 保持稳定。Tool Use：SDPO 一路退化，SRPO 跟住 GRPO 走势（甚至略好）。

这两个 pattern 实际上揭示了一个重要性质：**routing 机制让 SRPO 自动"探测" self-distillation 的有效性**。在 SDPO 有效的 domain（需要 dense logit-level correction 的科学推理），SRPO 把它榨干；在 SDPO 无效的 domain（tool use，可能正确答案空间小、不太需要 token-level correction），SRPO 自动 fallback 到 GRPO 主导。

### 4.3 Ablation（Table 2）

两个 ablation，重点不同：

**Ablation 1: Mixing strategy** —— SRPO w/o dynamic weighting vs Advantage Mix
- Advantage Mix：$A_{i,t}^{\text{Mix}}(v) = \lambda A_{i,t}^{\text{GRPO}}(v) + (1-\lambda) A_{i,t}^{\text{SDPO}}(v)$，$\lambda=0.9$
- 1h：Advantage Mix +0.7；5h：-2.5；10h：-3.3
- 解读：早期 dense signal 帮助，后期 noise propagation 阻碍稳定。Sample routing 把 SDPO 限制在 failed samples，避免污染 correct samples，长期更稳。

**Ablation 2: Dynamic weighting** —— SRPO vs SRPO w/o dynamic weighting
- 1h：+0.4；5h：+0.7；10h：+1.8
- 解读：gain 随训练增加而扩大，正是因为后期 teacher entropy 上升，dynamic weighting 抑制 noise 的作用越来越明显。

### 4.4 Response length & Compute（Figure 4）

- Response length: GRPO > SRPO > SDPO（GRPO verbose，SDPO 过短且损害 epistemic reasoning，SRPO 适中）
- Per-step compute：1h 时 SRPO 比 GRPO 慢 17.4%（额外 self-teacher forward），10h 时 SRPO 比 GRPO 快 17.2%，比 SDPO 快 9.4%
- 解释：late training failed sample 减少 → SDPO branch 激活频率降低 → self-teacher forward 减少；同时 SRPO 比 GRPO 短的 response length 也降低 inference cost

### 4.5 Routing statistics（Figure 5）

- 早期 ~40% samples 走 SDPO，~60% 走 GRPO
- 后期 SDPO 占比单调下降，GRPO 占比上升
- teacher info 可用比例始终高（说明 fallback 到 GRPO 主要由 correctness 驱动，不是 teacher 缺失）

这条曲线其实给了一个非常 clean 的 visual signature：**SRPO 是一个 self-stabilizing system**，training progression 自动调整两个 branch 的相对权重。

---

## 5. 关键设计选择的 deeper intuition

### 5.1 为什么 routing > advantage mixing？

Advantage mixing 等价于在 token level 做凸组合，但 GRPO advantage 是 sequence-level scalar（对同 rollout 所有 token 一样），SDPO advantage 是 logit-level vector，两者的 **scale 和 dimensionality 不匹配**。$\lambda$ 需要小心 tune。更重要的是，advantage mixing 在每个 token 上都同时应用两个 signal，无法避免 SDPO noise 污染 correct samples。

Routing 是 **disjoint partition**，每个 sample 只走一个 branch，物理上隔离了两种 signal 的相互污染。从信息论角度看，这是把 $H(\text{signal})$ 拆分成两个 conditional distribution $H(\text{GRPO}|\text{correct})$ 和 $H(\text{SDPO}|\text{incorrect})$，每个 conditional 都更纯净。

### 5.2 为什么 dynamic weighting 选 entropy 而非 probability margin？

考虑两个 alternative：

(a) Probability margin $1 - \max_v q_{i,t}(v)$：只关注 top-1 confidence，但忽略了分布形状（多个高概率 token 也是 uncertain）
(b) Entropy $H_{i,t}$：捕捉整分布的 uncertainty

Entropy 更 robust，因为它捕捉了完整的 distributional uncertainty。这与 Bayesian active learning 中用 entropy 做 acquisition function 是同一思路 —— "不确定的地方不要提供 supervision"。

形式上 $\tilde{w} = \exp(-\beta H)$ 其实是一个 **Boltzmann weighting**，把"高信息量（低 entropy）"作为能量低的状态。$w$ 的归一化形式 $\tilde{w}/\text{mean}(\tilde{w})$ 保证总 loss scale 不漂移，这点在 long-horizon training 中很重要 —— 避免后期 dynamic weighting 让 loss 越来越小。

### 5.3 EMA teacher 的微妙之处

self-teacher 是 student 的 EMA（update rate 0.05 from Table 3）。这有两个作用：
1. 让 teacher 比 student "慢一拍"，提供 quasi-stationary target（类似 target network in DQN）
2. EMA 平滑让 teacher 的 entropy 比 student 略低，提供相对 clean 的 distillation signal

但 EMA 也有问题：teacher 永远在追 student，gap 单调缩小，distillation signal 逐渐稀薄。这是 paper Diagnosis 2 的根源，dynamic weighting 部分缓解但无法根治。

### 5.4 Loss 归一化的 emergent behavior

$$\mathcal{L}_{\text{final}} = \frac{\text{routed losses}}{\text{routed token count}}$$

这个 normalization 让 SRPO 不需要 mixing hyperparameter $\lambda$。这是一个非常重要但容易被忽视的设计 —— 在 Advantage Mix 里，$\lambda = 0.9$ 是 tuned 出来的，依赖 task 和 stage。SRPO 的归一化让 mixing ratio 自动随 sample distribution 变化，**这种 implicit scheduling 比 explicit schedule 更鲁棒**，因为 sample distribution 本身就是 policy quality 的 proxy。

---

## 6. 与相关方法的联系（building your intuition map）

### 6.1 Process Reward Models (PRMs) vs SRPO

[Lightman et al. (2023)](https://arxiv.org/abs/2305.20050) 的 PRM 和 [Cui et al. (2025)](https://arxiv.org/abs/2502.01456) 的 PRIME 试图用 process-level reward 提供 dense supervision，但需要训练一个额外的 reward model，且 PRM 误差会传播。SRPO 用 self-teacher 的 logit 分布做 dense supervision，**不需要额外 reward model**，但代价是 signal 质量受 self-teacher 限制。

可以联想：如果 SRPO 用一个**外部强 teacher**（GPT-4、Claude）替代 self-teacher，Diagnosis 2（teacher entropy 退化）就消失，但 Diagnosis 1（correct samples 上 ambiguity）仍存在。SRPO 的 routing 设计是 task-agnostic 的，可以扩展到 external-teacher SDPO。这是 paper Section 5 提到的 future work。

### 6.2 DAPO 的 asymmetric clipping

[DAPO (Yu et al., 2025)](https://arxiv.org/abs/2503.14476) 引入 asymmetric clipping（$\varepsilon_{\text{low}} = 0.2, \varepsilon_{\text{high}} = 0.28$）来缓解 entropy collapse。SRPO 的 GRPO branch 直接采用这个 trick（Table 3: $\varepsilon$-high = 0.28）。这反映了一个深层问题：**RLVR 训练中 entropy 维护是核心难题**，DAPO 在 sequence level，SRPO 在 logit level（dynamic weighting）。

### 6.3 VinePPO / GRPO-λ / Causal Credit Assignment

[VinePPO (Kazemnejad et al., 2024)](https://arxiv.org/abs/2410.01679)、[GRPO-λ (Parthasarathi et al., 2025)](https://arxiv.org/abs/2510.00194)、[Causal Credit Assignment (Khandoga et al., 2026)](https://arxiv.org/abs/2602.09331) 都试图改善 GRPO 的 coarse credit assignment，但都停留在 reward signal 内部。SRPO 走的是**另一条路** —— 引入 dense supervision 但保留 reward alignment，并 routing 到不同 sample。两条路可以结合，但 SRPO 的 simplicity 优势在工程上明显。

### 6.4 RLOO / ReMax / ReST 等 on-policy distillation

[On-policy distillation (Agarwal et al., 2024)](https://arxiv.org/abs/2402.04616)、[Thinking Machines Lab's on-policy distillation](https://thinkingmachines.ai/blog/onpolicy-distillation)、[ReST^EM] 等都用 external teacher 做 on-policy supervision。SRPO 的 self-distillation 设计避免了 external teacher 的 cost 和 distribution mismatch，但限制在 self-improvement 范式内，天花板受 base model 限制。

### 6.5 Sequence-level KD vs token-level KD

经典 [Sequence-level KD (Kim & Rush, 2016)](https://arxiv.org/abs/1606.07947) 和 [MiniLLM (Gu et al., 2023)](https://arxiv.org/abs/2306.08543) 用 reverse KL 而非 forward KL，因为 forward KL 有 mode-covering 倾向（学生覆盖老师所有 mode，导致 over-generation）。SRPO 用 Jensen-Shannon divergence（Table 3），这是 forward 和 reverse KL 的折中，避免极端行为。这个选择对训练稳定性的影响值得单独 ablation。

### 6.6 Context distillation 的脉络

[Snell et al. (2022)](https://arxiv.org/abs/2209.15189) "Learning by distilling context" 是 self-distillation 的早期形式：让 model 在有 privileged context 下生成，然后蒸馏回 no-context 版本。SDPO 把这个 idea 推到 RL setting，让 self-teacher 在 privileged context（sibling solution）下 re-score student trajectory。SRPO 在这个脉络上加 routing，把 context distillation 局限在"需要 correction"的 sample 上。

### 6.7与 Epistemic Verbalization 的关系

[Kim et al. (2026)](https://arxiv.org/abs/2603.24472) 指出 SDPO 会让模型生成过短的 reasoning chain，损害 epistemic verbalization（模型自言自语、检查自己的能力）。SRPO 的 Figure 4(a) 显示 response length 适中，说明 GRPO branch 的 reward signal 对抗了 SDPO 的"过度压缩"倾向。这一点和 Kim et al. 的发现形成 mutual support —— 不同机制，同一现象。

---

## 7. Limitations & 我看到的一些 open questions

paper 自己提到的 limitation：只考虑 sibling rollout 作为 teacher info，没有用 environment feedback（execution trace、error message 等）。在 agentic / code task 上，environment feedback 是更丰富的 teacher signal。

我看到的几个 deeper open questions：

1. **Routing 的 boundary case**：当一个 group 中只有 1 个 correct rollout，其他 7 个都 fail 时，7 个 fail rollout 共享同一个 sibling 作为 teacher。如果这个 sibling 恰好走了"lucky path"（碰巧对），那么 7 个 fail sample 都被蒸馏向一个 sub-optimal path。Routing 没有机制处理这种"唯一 correct 但 sub-optimal"的情况。

2. **Catastrophic forgetting on SDPO branch**：late training 时 SDPO branch 激活频率降低，policy 在 failed sample 上的 correction 能力是否退化？如果突然遇到 OOD difficult prompt，SRPO 是否还能像早期那样快速纠正？Figure 5 的 SDPO fraction 下降是好事（policy 变好），但也是某种"忘记如何纠正"的信号。

3. **Entropy weighting 的极端 case**：$\beta = 1$ 是 default，但 paper 没有 sensitivity analysis。如果 $\beta \to \infty$，dynamic weighting 退化为 hard masking（只用 entropy = 0 的 token），可能 signal 过稀；如果 $\beta \to 0$，退化回 vanilla SDPO。$\beta$ 应该是 task-dependent，paper 没探索这个。

4. **GRPO branch 是否应该用更细的 advantage**：SRPO 把 correct samples 路由到 GRPO，但 GRPO 仍是 sequence-level。如果在 correct samples 上做 process reward（每个 reasoning step 是 verified 的），SRPO 可以进一步细化。这是 SRPO + PRM 的自然 extension。

5. **Self-teacher 的"knowledge saturation"**：Diagnosis 2 说 teacher entropy 上升，但 paper 没分析这个 entropy 上升是因为 teacher 真的更 uncertain，还是 teacher 在某些 token 上 over-confident 在某些 token 上 over-uncertain。如果是后者，dynamic weighting 可能不是最优解 —— 一个 token 看似 high entropy，可能是它在两个 valid choice 间犹豫（这种"犹豫"反而是有信息的）。

6. **Multi-turn agentic setting 的适配**：当前实验是 single-turn（一次 generation 得 reward）。Multi-turn agentic（tool calling、code execution）中，每个 turn 都可以做 routing。SRPO 的 framework 应该可以扩展，但需要解决跨 turn credit assignment。

---

## 8. 实施细节的几个亮点（容易忽视但重要）

- **SGLang 替代 vLLM**：Appendix B.1 说明换 backend 只影响 throughput 不影响 sampling，保证与 SDPO 公平比较。这是负责任的工程实践。

- **Learning rate $5 \times 10^{-6}$ 选在 GRPO ($10^{-6}$) 和 SDPO ($10^{-5}$) 中间**：这个选择是为了 balance 两个 branch 的 gradient scale。但没有 ablation 说明这个中点是否最优。考虑到两个 branch 的 gradient magnitude 在不同 stage 会变化，可能需要一个自适应 lr schedule。

- **Asymmetric clipping** $\varepsilon_{\text{high}} = 0.28$：来自 DAPO，缓解 entropy collapse。SRPO 在 GRPO branch 用，SDPO branch 不用（KL loss 不需要 clipping）。

- **Jensen-Shannon divergence** 而非 KL：避免 KL 的 unbounded 问题，对称性更好。

- **Top-K = 100 distillation**：只对 top-100 logits 做 distillation，避免长尾 vocab 噪声。

- **EMA update rate = 0.05**：teacher 每 20 步"追上"student 一次。这是个慢速 EMA，确保 teacher 相对 stable。

- **No thinking mode**：Qwen3 的 thinking mode 关闭，所以实验是在 non-thinking setting 下做的。这暗示 SRPO 的 gain 可能与 chain-of-thought reasoning 的 explicit step 结构有关。如果在 thinking mode 下，每一步 thinking 都是一个 explicit decision point，SDPO 的 dense logit signal 可能效果不同。

---

## 9. 总结直觉：SRPO 是什么？

如果你要在脑海里画一张 mental model：

- **GRPO** 像"全班考试后给整个卷子打分"，错了不知道哪一步错。
- **SDPO** 像"全班考试后让优等生把自己的解题过程抄给你"，但优等生自己也越学越糊涂，且对本来答对的同学也强行让他改抄别人的解法。
- **SRPO** 像"答对的同学继续按卷面分激励，答错的同学抄优等生的对应步骤，但只抄优等生确定的步骤，不确定的步骤不抄"。后期大家都答对了，就只按卷面分激励。

paper 的核心 contribution 不是新算法，而是 **正确诊断 + 简单且 robust 的组合方式**。Routing 是一个 sample-level 的 mixture-of-experts 思想，dynamic weighting 是一个 token-level 的 attention 机制，两者结合形成一个 hierarchical 的 signal-quality gating。这种 design pattern 在 ML 里反复出现 —— 用 hierarchical gating 来处理 signal quality 的 heterogeneity。

---

## Reference 链接汇总

- [SRPO (this paper)](https://arxiv.org/abs/2606.07894)（基于内容推测的 ID，可能需要核实）
- [GRPO - DeepSeekMath (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)
- [SDPO - Reinforcement Learning via Self-Distillation (Hübotter et al., 2026)](https://arxiv.org/abs/2601.20802)
- [Learning on the Job - Test-time curricula (Hübotter et al., 2025)](https://arxiv.org/abs/2510.04786)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [OpenAI o1 System Card](https://arxiv.org/abs/2412.16720)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [DAPO (Yu et al., 2025)](https://arxiv.org/abs/2503.14476)
- [PRIME - Process Reinforcement (Cui et al., 2025)](https://arxiv.org/abs/2502.01456)
- [PRM - Let's Verify Step by Step (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050)
- [On-policy Distillation (Agarwal et al., 2024)](https://arxiv.org/abs/2402.04616)
- [Thinking Machines - On-policy Distillation](https://thinkingmachines.ai/blog/onpolicy-distillation)
- [Learning by Distilling Context (Snell et al., 2022)](https://arxiv.org/abs/2209.15189)
- [MiniLLM (Gu et al., 2023)](https://arxiv.org/abs/2306.08543)
- [Sequence-level KD (Kim & Rush, 2016)](https://arxiv.org/abs/1606.07947)
- [Distilling Knowledge (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
- [Why Self-Distillation Degrades Reasoning (Kim et al., 2026)](https://arxiv.org/abs/2603.24472)
- [Causal Credit Assignment (Khandoga et al., 2026)](https://arxiv.org/abs/2602.09331)
- [GRPO-λ (Parthasarathi et al., 2025)](https://arxiv.org/abs/2510.00194)
- [SciKnowEval (Feng et al., 2024)](https://arxiv.org/abs/2406.09098)
- [ToolAlpaca (Tang et al., 2023)](https://arxiv.org/abs/2306.05301)
- [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [verl RLHF framework](https://arxiv.org/abs/2409.19256)
- [SGLang](https://arxiv.org/abs/2312.07104)
- [REINFORCE (Williams, 1992)](https://link.springer.com/article/10.1007/BF00992696)

如果你想继续 deep-dive 任何一个 aspect（比如 SDPO 的 gradient 形式怎么从 KL 推出来、dynamic weighting 与 Bayesian active learning 的关系、扩展到 multi-turn agentic setting 的具体设计），我可以继续展开。
