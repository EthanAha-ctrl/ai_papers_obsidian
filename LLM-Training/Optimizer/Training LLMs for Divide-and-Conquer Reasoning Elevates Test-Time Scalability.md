---
source_pdf: Training LLMs for Divide-and-Conquer Reasoning Elevates Test-Time Scalability.pdf
paper_sha256: c4b9c6027426f005bbea6aa40c3597fdda50aedc3c992c72143730dfef86e030
processed_at: '2026-08-12T17:56:59-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇paper到底在干嘛

## 一句话故事

**现在的LLM都是一根筋解题，遇到难题就卡死或绕圈子；这篇paper说，教model学会"先把难题拆成小问题再逐个击破"这个技能，需要专门用RL训练，训完之后不仅难题解得更好，连原来那根筋的解法也变强了。**

---

## 背景的尴尬

大家都觉得LLM reasoning很强了，O1、R1这些model解AIME题目很猛。但你仔细想，它们其实都是**一条直线走到底**——生成一长串CoT，然后吐答案。

这有个问题：遇到真正难的题（比如HMMT、Beyond-AIME这种竞赛级），一条线走到底经常走不通，或者走到一半发现错了又折返，self-reflection半天，token花了一堆。

**人的直觉很简单**：难题拆成几个小问题，每个小问题单独解，最后拼起来。这是计算机科学最经典的divide-and-conquer思想，小学生都会用。

而且你test-time想scale的时候，CoT只能"多sample几条独立的trajectory然后vote"——这很暴力，没有structure。DAC天然有structure：你可以sample多种不同的decomposition strategy，每种strategy下面再sample多条solution path，形成一棵树。这比"傻傻多sample"efficient多了。

**所以DAC inference这件事，别人早就试过了**（Tree of Thoughts, DeAR, Seed-Prover, DeepSeek-Prover-V2...）。结果呢？**直接拿来用，性能普遍比CoT还差**。

Figure 2的数据很打脸：拿一堆instruction-tuned和reasoning model直接做DAC inference，几乎全都比CoT差。比如Qwen2.5-7B在AIME上，CoT Pass@32有24.1%，DAC只有9.2%，直接腰斩还有余。

---

## 为什么直接用不行

Paper给了一个很清晰的诊断：**misalignment**。

你的model在post-training阶段被训成了"看到问题→一条线推理→给答案"的shape。它的整个policy distribution都是围绕这个behavior shape的。你突然在inference时让它"先decompose再conquer"，这对它来说是个distribution外的behavior，它不会干。

这就好比你把一个短跑运动员拉去跑跨栏，他体能可能很强，但不会跨栏技术，直接上肯定不行——你得专门训。

**这就是这篇paper的核心contribution**：不是又一个inference-time的prompt engineering trick，而是**把DAC纳入training，用RL专门训model学会这个reasoning style**。

---

## 方法：DAC-RL到底怎么训的

### 两个stage，一个loop

每个training step，对每个问题 $x$：

**Stage 1 - Divide（绿）**: 让model把 $x$ 拆成一组subproblems $\mathcal{P} = \{p_1, p_2, ..., p_{n_g}\}$，这步生成的是 $y_d$。

**Stage 2 - Conquer（蓝）**: 把 $\mathcal{P}$ 和原问题 $x$ 拼一起做成conquering prompt，让model先依次解subproblems，再解原问题，生成 $y_c$。

**然后两个都进RL buffer，都用GRPO更新policy。**

### 关键难点：reward怎么给

**Conquering的reward好办**：
$$\mathbf{R}(y_c) = \mathbf{1}\{\text{Extract}(y_c) = a\}$$

最终答案对了给1，错了给0。$a$是原问题的ground truth。这跟普通RLVR一样。

**Division的reward才是真问题**。你拿什么标准判断"这组subproblems拆得好不好"？你没有subproblem的答案，没法直接判对错。

作者试过用"conquering accuracy的平均值"作为division reward，结果model学坏了——它在division阶段直接把原题解了塞进去，根本不decompose（Appendix C, Figure 8）。因为model发现"输出解题过程"比"输出subproblems"能拿到更高reward，就shortcut了。

**最终的design是公式(2)的relaxed reward**：

$$\mathbf{R}(y_d) = \begin{cases} 
0, & \text{if } |\mathcal{P}_g| < N_s \lor \neg\text{Format}(y_d) \\
0, & \text{if } \text{CA}(\mathcal{P}_g) = 0 \land \text{CA}(\{\mathcal{P}_i\}_{i=1}^{G_d}) > 0 \\
1, & \text{otherwise}
\end{cases}$$

人话翻译：

- **第一个条件**：subproblems数量不到 $N_s=3$ 个，或者格式parse不出来 → 给0。这是硬约束，防止model偷懒不分。
- **第二个条件**：如果这组subproblems一个正确答案都没产出（$\text{CA}(\mathcal{P}_g)=0$），但其他组有产出（$\text{CA}(\{\mathcal{P}_i\})>0$）→ 给0。意思是"别人能帮model解出来，你这组不行，你没用"。
- **其他情况**：给1。包括"这组帮model解出来了"或者"所有组都解不出来"（那不能怪你这组）。

**关键insight**: 这个reward是**下界保证**，不是精确评分。它只惩罚"明显没用"的decomposition，对"有用"的decomposition给pass，但不去精细比较哪个更有用。这给model留了exploration空间，防止它greedy地optimization某个proxy metric。

这个design philosophy其实挺有意思——**reward设计上"宽松"反而比"精确"好**，因为精确的proxy容易被hacking。

### Lemma 2.1：为什么final answer能当surrogate

这是个理论justify：你没有subproblem的ground truth，凭什么用final answer来train subproblem solving？

**Assumption**: $\mathbf{s} \to C$ 有causal direction（subproblem解对与否causally影响原问题解对与否），且 $P(C=1|\mathbf{s})$ 对每个 $s_i$ 单调递增（多解对一个subproblem不会让原问题更难解）。

**结论**:
$$\text{Cov}_\theta(\mathbf{1}\{s_i=1\}, \mathbf{1}\{C=1\}) \geq 0$$

意思：**"原问题解对"和"subproblem $i$ 解对"正相关**。

所以你reward $C=1$，统计上会prefer那些解对更多subproblems的trajectory。proof很直接，用Bayes rule展开 $P(S_i=1|C=1)$ 就能看出来。

**这解决了"无subproblem标注"的困境**——你不需要subproblem答案，final answer本身就是一个consistent的surrogate signal。

---

## 结果：到底涨了多少

### 主实验（Table 1）

拿Qwen3-4B-Instruct-2507做base model：

| 指标 | Init CoT | RL-CoT | RL-DAC | DAC提升 |
|------|----------|--------|--------|---------|
| Avg Pass@1 | 42.7 | 37.5↓ | 46.1 | +8.6 |
| Avg Pass@32 | 72.1 | 69.0↓ | 75.3 | +6.3 |

**最striking的点**：

1. **RL-CoT反而降了**。Qwen3-4B这个model已经被post-training训得很充分了，你继续用CoT-RL训它，Pass@1从42.7掉到37.5。AIME 24从62.6掉到45.9，直接崩盘。这说明**CoT的trajectory space已经saturated了**，RL找不到新信号，反而把policy带偏了。

2. **RL-DAC大幅涨**。Pass@1 +8.6, Pass@32 +6.3。DAC打开了CoT够不到的ceiling。

3. **初始DAC其实不弱**。Init-DAC 40.2 vs Init-CoT 42.7，差距很小。说明Qwen3-4B本来就有一定DAC能力，只是没被释放。这跟Qwen2.5-7B形成对比——后者Init-DAC只有0.4%，基本不会，但RL-DAC仍然能超过RL-CoT +3.4% Pass@32。**证明DAC-RL即使从很弱的起点也能work**。

### Deep DAC（难题专项）

在3.7k最难的问题上训10个epoch，token budget扩到16k/24k：

| Setting | Avg Pass@1 | Avg Pass@32 |
|---------|-----------|-------------|
| RL-D-CoT (32 rollouts) | 49.9 | 76.9 |
| RL-D-DAC (32 rollouts) | 51.3 | 81.6 |

**重点**: 控制了rollout budget相同（都是32），DAC还是+4.7% Pass@32。**这个gain不是"我budget更大所以更好"，是paradigm本身的优势**。

---

## 几个"反直觉"但很重要的发现

### 1. DAC训练居然让CoT也变强了

Mix-RL实验：难题用DAC训，简单题用CoT训。结果**CoT inference的性能也涨了10%以上**（Figure 5左）。

这说明DAC training教给model的不只是"怎么分治"，而是**一个meta-skill**——如何把复杂问题structurally拆解。这个skill transfer回了CoT推理，让CoT的轨迹也更structured。

类比一下：你练crossfit，虽然不是直接练跑步，但你的跑步成绩也会涨，因为underlying的体能和协调性提升了。

### 2. DAC反而更省token

直觉上"先拆再解"应该更长，但Figure 7显示**DAC的response更短，clip ratio更低，policy entropy更高**。

Appendix D的case study很清楚：拿同一道方程组题看，DAC和CoT都引入了 $x=ab, y=bc, z=ac$ 的substitution，都解了同一个linear system。但：

- **DAC**：因为subproblems是预定义的（"rewrite equations"→"introduce substitution"→"solve system"→"recover values"→"compute answer"），model直接顺着走，每步干净利落。3328 tokens。
- **CoT**：model反复self-doubt，"Wait no", "Wait solve carefully", "But wait — we previously found..."，一遍遍verify已经推过的东西。5072 tokens。

**DAC压缩的不是数学，是narrative redundancy**。CoT的冗长很多来自self-correction和restatement，DAC的structure把这些砍掉了，同时exploration diversity反而更高（entropy更高）。

### 3. 严格format约束反而有害

Table 3：强制conquering response必须严格按"subproblem 1: ..., subproblem 2: ..."格式 → Pass@1从51.3掉到45.2。

这是个alignment tax现象。Model有时候想跳着解某个subproblem来启发其他subproblem，你硬要它按顺序，就限制了创造力。

**启示**：structure要有，但不能太死。reward给"下界保证"而非"精确评分"的design，跟这个发现是consistent的——都强调给model留flexibility。

### 4. Test-time: diversity > attempts

Figure 6：固定budget $k=1024$，变 $n \times m$（$n$组decomposition，每组$m$条conquering）。

发现：$n$越大（更多decomposition strategies）、$m$越小，性能越好。

直觉：**多几个perspective比在同一个perspective上多试几次更有效**。每个decomposition是一种看待问题的方式，不同decomposition给出不同的solution space，比在同一个space里重复sample efficient。

---

## 我的intuition总结

这篇paper的核心story可以这样理解：

**CoT是一条高速公路，post-training已经把它修到很宽很平了，RL想再拓宽发现边际收益递减，甚至overfit搞坏了。**

**DAC不是另一条更宽的高速公路，它是一个road network——有分叉、有层次、有多条路径同时探索。这个network的exploration space比单条高速公路大得多，RL有大量信号可以学。**

而且学这个network的navigation skill，反过来让你在single highway上开车也更structured了——因为你知道"原来这段路可以这样拆开看"。

**技术上的clever之处**：

1. 用final answer作为subproblem correctness的surrogate（Lemma 2.1），绕开了"无subproblem标注"的困境
2. 用relaxed lower-bound reward而非精确proxy，防止reward hacking
3. 把division和conquering都纳入unified RL，让两者co-evolve
4. 靠GRPO + Clip-Higher保持entropy，防止exploration collapse

**这篇paper对行业的implication**：

我们可能过度依赖"把CoT越拉越长"这一条路了。O1、R1的成功让人以为reasoning就是"longer CoT"。但这篇paper说，**reasoning的structure比length更重要**。改变structure（从linear到hierarchical），比增加length能打开更大的ceiling。

这跟人类的认知也吻合——expert解题不是比novice想得更多，而是想得更structured。Expert会自动decompose，novice只会linear grind。

**所以这可能不只是另一个trick，而是一个paradigm shift的信号**：从"more computation"到"better structured computation"。

---

## 几个我觉得值得继续想的问题

1. **Recursive DAC**：现在只有single-level divide。如果subproblem可以再divide，形成tree，exploration space会指数增长。但reward怎么propagate是个难题。

2. **Adaptive depth**：简单题直接CoT，难题用DAC，超难题用recursive DAC。怎么让model自己学会判断难度并选择depth？

3. **DAC + verification**：如果conquering阶段引入formal verifier（比如Lean）来验证subproblem solution，reward signal会精确得多。DeepSeek-Prover V2已经在做subgoal decomposition + formal verification了。

4. **DAC和agentic的结合**：DAC的decomposition本质是planning，conquering是execution。这跟agentic framework的plan-then-execute很像。DAC-RL训出来的model是不是更好的agentic planner？

5. **为什么CoT会saturate但DAC不会**：我猜是因为CoT的trajectory space被post-training压成了一个低维manifold，RL在这个manifold上找不到gradient。DAC的hierarchical structure是更高维的space，RL有真正的exploration room。但这个intuition需要更formal的characterization。

6. **RL-CoT下降的原因**：Table 1里Qwen3-4B的RL-CoT从42.7掉到37.5，这个现象本身很值得研究。是entropy collapse？是distribution shift？是overfit to training set的某种pattern？paper没深入讨论，但这关系到"什么时候该停RL"这个重要问题。

paper链接：https://arxiv.org/abs/2502.07957 （注：实际arxiv ID以repo为准）
代码：https://github.com/MasterVito/DAC-RL

---

# Training LLMs for Divide-and-Conquer Reasoning Elevates Test-Time Scalability 深度解析

## 1. Core Motivation: The Misalignment Problem

这篇 paper 的核心 insight 在于识别了一个 fundamental misalignment：**general-purpose post-training（主要是 CoT-style RL）和 DAC-style inference 之间存在 distribution gap**。

Figure 2 的数据非常 striking：拿 instruction-tuned 或 reasoning model 直接做 DAC inference，性能普遍比 CoT 差。比如 Qwen2.5-7B-Instruct 在 AIME 上的 CoT Pass@32 是 24.1%，但 DAC 只有 9.2%。这说明 **model 被训成了 step-by-step CoT 的 shape，但 DAC 要求完全不同的 behavior pattern（先 decompose 再 conquer），这个 mismatch 限制了 model 即使在简单问题上发挥 DAC 的潜力**。

Paper 的 GitHub repo: https://github.com/MasterVito/DAC-RL

---

## 2. Method: DAC-RL Framework

### 2.1 Task Formalization

**CoT reasoning**:
- Input: $x$
- Policy $\pi_\theta$ 直接生成 trajectory $y$
- Answer: $a = \text{Extract}(y)$

**DAC reasoning** 分两个阶段：

**Division step**:
$$\mathcal{P} = \{p_i\}_{i=1}^{n_g} \sim \pi_\theta(\mathcal{P} | x)$$

- $\mathcal{P}$: 一组 subproblems
- $n_g$: subproblems 数量（varies per group）
- $y_d$: division response

**Conquering step**:
$$S = \{s_i\}_{i=1}^{n} \sim \pi_\theta(S | x, \mathcal{P})$$

- $S$: subproblem solutions
- $\mathcal{P}$ 和 $x$ concat 成 conquering prompt
- Model 依次解 subproblems，然后解原问题
- $y_c$: 完整 conquering response

### 2.2 Optimization Objective

公式 (1):
$$\mathcal{J}(\theta) = \mathbb{E}_{y_d, y_c \sim \pi_\theta}[\mathbf{R}(y_d) + \mathbf{R}(y_c)]$$

- $\mathcal{J}(\theta)$: 期望回报的目标函数
- $\theta$: policy 参数
- $\mathbf{R}(y_d)$: division reward（详见 2.3）
- $\mathbf{R}(y_c)$: conquering reward（详见 2.4）

### 2.3 Subproblem Division Reward

公式 (2) 是关键的 piecewise reward：

$$\mathbf{R}(y_d) = \begin{cases} 
0, & \text{if } |\mathcal{P}_g| < N_s \lor \neg\text{Format}(y_d) \\
0, & \text{if } \text{CA}(\mathcal{P}_g) = 0 \land \text{CA}(\{\mathcal{P}_i\}_{i=1}^{G_d}) > 0 \\
1, & \text{otherwise}
\end{cases}$$

变量解释：
- $|\mathcal{P}_g|$: group $g$ 中 subproblems 的数量
- $N_s$: 最少 subproblems 数量（实验中 $N_s = 3$）
- $\text{Format}(y_d)$: 格式是否 valid（能否用 regex parse）
- $\text{CA}(\mathcal{P}_g)$: conquering accuracy，即用 $\mathcal{P}_g$ 作为 conquering prompt 时能否解出原问题
- $G_d$: division groups 数量（实验中 $G_d = 4$）

**Key design insight**: 这个 reward 设计了三个 component：

1. **Format validity**: subproblems 必须能被 regex parse
2. **Quantity validity**: 至少 $N_s$ 个 subproblems，否则给 0 reward（防止 collapse 到不分解）
3. **Helpfulness lower bound**: 如果所有 groups 都解不出但某个 group 能解出，其他解不出的 group 给 0 reward；否则给 1

**Why not use average conquering accuracy directly?** Appendix C 揭示了一个重要的 failure mode：如果用 average accuracy 作为 division reward，model 会 **shortcut 到在 division stage 直接解题**，而不是真的 decompose（见 Figure 8 的 case）。这个 relaxed lower bound reward 防止 greedy behavior，强迫 model 真的进行 decomposition。

这个 observation 让我联想到 RL 中常见的 **reward hacking** 现象 - 当 reward 和 desired behavior 不完全 aligned 时，model 会找到 shortcut。Paper 的解决方案是设计一个 **relaxed reward**，只保证 lower bound，给 model 留出 exploration 空间。

### 2.4 Conquering Reward

公式 (3):
$$\mathbf{R}(y_c) = \mathbf{1}\{\text{Extract}(y_c) = a\}$$

- $a$: 原问题的 ground-truth answer
- 只用最终答案正确性作为 reward

**Lemma 2.1** (Final-answer reward positively associates with subproblem correctness):

公式 (4):
$$\text{Cov}_\theta(\mathbf{1}\{s_i = 1\}, \mathbf{1}\{C = 1\}) \geq 0$$

- $s_i \in \{0,1\}$: subproblem $i$ 是否解对
- $C \in \{0,1\}$: 原问题是否解对
- $\text{Cov}_\theta$: 在 policy $\pi_\theta$ 下的 covariance

**Proof intuition** (Appendix B):
1. 用 law of total probability 展开 $P_\theta(C=1)$:
$$P_\theta(C=1) = \sum_{\mathbf{s} \in \{0,1\}^m} P(C=1 | S=\mathbf{s}) P_\theta(S=\mathbf{s}) = \mathbb{E}_{S \sim P_\theta}[g(S)]$$

其中 $g(\mathbf{s}) := P(C=1 | S=\mathbf{s})$。

2. **Assumption 1 (Monotonicity)**: $g$ 是 increasing function - 解对更多 subproblems 不会降低解对原问题的概率。

3. 用 Bayes rule:
$$P_\theta(S_i=1 | C=1) = \frac{\sum_{\mathbf{s}: s_i=1} P(C=1|S=\mathbf{s}) P_\theta(S=\mathbf{s})}{P_\theta(C=1)}$$

4. 因为 $P(C=1|S=\mathbf{s})$ 在 $s_i=1$ 时更大，所以:
$$P_\theta(S_i=1 | C=1) \geq P_\theta(S_i=1)$$

5. 等价于 $\text{Cov}_\theta(\mathbf{1}\{S_i=1\}, \mathbf{1}\{C=1\}) \geq 0$。

**这个 lemma 的意义**: 虽然没有 subproblem 的 ground-truth，但 final answer correctness 是 subproblem correctness 的 **consistent surrogate signal** - reward $C=1$ 会 preferentially upweight 有更多正确 subproblems 的 trajectories。

### 2.5 Training Algorithm

Algorithm 1 的核心循环：

```
for t = 1 to T do
    Sample mini-batch d ~ D
    for each (x, a) in d do
        # DIVIDE stage
        Generate G_d subproblem groups {P_g}_{g=1}^{G_d} ~ π_θ(x)
        
        for each P_g do
            # CONQUER stage  
            Generate G_c solutions {y_{g,v}}_{v=1}^{G_c} ~ π_θ(P_g; x)
            Compute rewards R(y_{g,v}) w.r.t. ground truth a
            Store to buffer: B ← B ∪ {([x; P_g], y_{g,v}, R(y_{g,v}))}
        end
        
        # DIVISION REWARD computation
        Evaluate format {f_g} and quantity {q_g} validity
        Compute division rewards {R(P_g)} via Eq. (2)
        Store to buffer: B ← B ∪ {(x, P_g, R(P_g))}
    end
    
    # POLICY UPDATE
    Update π_θ using buffer B (GRPO)
    Clear buffer
end
```

关键超参数：
- $G_d = 4$: division groups
- $G_c = 8$: conquering solutions per group
- $N_s = 3$: minimum subproblems
- batch size = 256
- max rollout length = 8192
- temperature = 1.0
- Clip-Higher upper bound $\varepsilon_h = 0.28$
- mini-batch size = 64
- 400 training steps (~6 epochs)

---

## 3. GRPO Background (Appendix A)

公式 (5) - Advantage computation:
$$A_{i,t} = \frac{r_i - \text{mean}(\{r_i\}_{i=1}^G)}{\text{std}(\{r_i\}_{i=1}^G)}$$

- $A_{i,t}$: response $i$ 中 token $t$ 的 advantage
- $r_i$: response $i$ 的 reward
- $G$: group size
- 注意 advantage 对所有 token 相同（token-level 区别在 loss 中体现）

公式 (6) - GRPO loss:
$$\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, Y \sim \pi_{\theta_{\text{old}}}}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\left(\min(k_{i,t}(\theta)A_{i,t}, \text{clip}(k_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon)A_{i,t}) - \beta D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})\right)\right]$$

变量解释：
- $k_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x, y_{i,<t})}$: probability ratio
- $\varepsilon$: clip 参数（standard PPO clip range）
- $\varepsilon_h$: Clip-Higher 上界（0.28，比 standard 更宽松）
- $\beta$: KL penalty 系数
- $D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})$: policy 和 reference model 的 KL divergence
- $|y_i|$: response $i$ 的长度（token-level loss 的归一化）

**Clip-Higher** from DAPO (Yu et al., 2025a): 放宽上界允许更大的 policy update，防止 entropy collapse。这个 technique 对 DAC training 特别重要，因为 DAC 需要保持 exploration diversity。

Reference: DAPO paper https://arxiv.org/abs/2503.14476

---

## 4. Experimental Results Analysis

### 4.1 Main Results (Table 1)

| Model | Method | AIME 24 P@1 | AIME 24 P@32 | AIME 25 P@1 | AIME 25 P@32 | Beyond-AIME P@1 | Beyond-AIME P@32 | HMMT P@1 | HMMT P@32 | Avg P@1 | Avg P@32 |
|-------|--------|-------------|--------------|-------------|--------------|-----------------|------------------|----------|-----------|---------|----------|
| Qwen3-4B | Init-CoT | 62.6 | 90.0 | 45.7 | 76.7 | 32.1 | 65.0 | 30.3 | 56.7 | 42.7 | 72.1 |
| Qwen3-4B | Init-DAC | 59.6 | 90.0 | 43.2 | 73.3 | 29.6 | 61.0 | 28.2 | 63.3 | 40.2 | 71.9 |
| Qwen3-4B | RL-CoT | 45.9 | 85.8 | 52.1 | 77.4 | 30.4 | 58.1 | 21.8 | 54.4 | 37.5 | 69.0 |
| Qwen3-4B | RL-DAC | **63.9** | **87.7** | **54.2** | **78.8** | **34.6** | **67.9** | **31.9** | **66.6** | **46.1** | **75.3** |
| **Δ(RL)** | | **+18.0** | **+1.9** | **+2.1** | **+1.4** | **+4.2** | **+9.8** | **+10.1** | **+12.2** | **+8.6** | **+6.3** |

**Critical observations**:

1. **Init-DAC ≈ Init-CoT**: 对于 Qwen3-4B，初始 DAC 和 CoT 性能接近（40.2% vs 42.7%）。这说明这个 model 已经有一定 DAC capability，但还没被充分开发。

2. **RL-CoT 反而下降**: Qwen3-4B 的 CoT RL 训练后 Pass@1 从 42.7% 降到 37.5%（AIME 24 从 62.6% 降到 45.9%）。这是非常 striking 的现象 - **CoT reasoning 已经 saturated，继续 RL 反而有害**（可能 overfit 到 training distribution 或 entropy collapse）。

3. **RL-DAC 大幅提升**: Pass@1 +8.6%, Pass@32 +6.3%。DAC 打开了新的 exploration 空间。

4. **Qwen2.5-7B 的不同 pattern**: Init-DAC 极差（0.4% avg），但 RL-DAC 仍然超过 RL-CoT（+3.4% Pass@32）。这证明 DAC training 即使在初始 capability 很弱的情况下也有效。

### 4.2 Deep DAC Training (Bottom of Table 1)

| Setting | Avg P@1 | Avg P@32 |
|---------|---------|----------|
| RL-D-CoT | 49.9 | 76.9 |
| RL-D-DAC | 51.3 | 81.6 |
| Δ(RL) | +1.4 | +4.7 |

Deep DAC: 在 3.7k 最难的问题上训 10 epochs，budget 扩展到 16,384/24,576 tokens。

**Key insight**: 在 equal rollout budget (32 per problem) 下，DAC 仍然超过 CoT +4.7% Pass@32。这说明 **DAC 的 scalability advantage 不仅是 budget 扩大的结果，而是 reasoning paradigm 本身的优势**。

### 4.3 Intermediate Checkpoints (Figure 4)

Figure 4 显示训练 dynamics：
- 两个 model 都从 DAC < CoT 开始
- 随训练进行，DAC 增长更快，最终超过 CoT
- CoT-RL 很快 plateau，DAC-RL 持续上升

这个 pattern 暗示 **CoT 的 trajectory space 已经被 post-training 填满，RL 难以找到新信号；DAC 的 trajectory space 是新的，有大量信号可学**。

---

## 5. Key Analysis

### 5.1 Mix-RL: DAC Enhances CoT (Section 4.1)

Mix-RL setup: 对难题（accuracy < 25%）用 DAC training，对简单题保留 CoT training。

Figure 5 的 striking 发现：
- **DAC training 甚至能 enhance CoT reasoning**（+10% on all benchmarks under CoT inference）
- Mix-RL 也能 activate DAC capability

**Intuition**: DAC training 让 model 学会了 "如何 decompose" 这个 meta-skill，这个 skill 可以 transfer 回 CoT。Model 内化了 "把复杂问题拆解" 的思维模式，即使在 CoT 推理时也会更 structured。

### 5.2 Test-time Scalability (Section 4.2, Figure 6)

固定 budget $k = 1024$，变化 $n$ (groups) 和 $m$ (conquering per group)，$n \times m = k$。

**发现**: 分配更多 groups（更大 $n$，更小 $m$）性能更好。

**Intuition**: 
- CoT baseline 是 1024 independent generations，没有结构
- DAC 的 $n$ groups 提供 **diverse decomposition strategies**
- 每个 decomposition 是一个 "perspective"，更多 perspectives 比 more attempts on same perspective 更 valuable
- 这说明 **subproblem diversity 比 conquering attempts 更重要**

### 5.3 Concise Reasoning (Section 4.3, Figure 7)

Figure 7 显示：
- DAC response length 更短
- Clip ratio 更低（fewer truncations）
- **Policy entropy 更高**（more exploration）

**Counter-intuitive 发现**: 引导 model 解 subproblems 理应增加 reasoning steps，但实际产生更 compact 的 traces。

**Explanation** (Appendix D case study): DAC 的 sub-solution 遵循 predefined decomposition，每个 subproblem 直接映射到必要的 transformation，**避免了 CoT 中的 redundant restatements 和 self-corrections**。

对比 Figure 10 (DAC) vs Figure 11 (CoT) 的 partial solutions：
- DAC: 直接 introduce $x=ab, y=bc, z=ac$，直接解 linear system
- CoT: 反复 verify、re-derive、self-doubt（"Wait no", "Wait solve carefully", "But wait"）

**DAC compacts reasoning by reducing narrative redundancy, not simplifying math**。

### 5.4 Cold-Start Distillation (Section 4.4, Table 2)

| Setting | Avg P@1 | Avg P@32 |
|---------|---------|----------|
| CD-CoT | 43.4 | 74.8 |
| CD-DAC | 46.4 | 76.2 |
| CD-RL-CoT | 52.2 | 79.7 |
| CD-RL-DAC | 53.5 | 82.1 |
| Δ(RL) | +1.3 | +2.4 |

Cold-start: 用 Qwen3-235B-A22B 生成 3k CoT + 3k DAC 数据，SFT 后再 RL。

**发现**: 
- Cold-start 后 DAC 已经略优于 CoT（46.4 vs 43.4）
- RL 后 DAC 优势放大（+2.4 vs +1.3）
- **DAC 在 RL 阶段 enable richer exploration**，导致更大 gain

### 5.5 Format Constraint Harmful (Section 4.5, Table 3)

强制 conquering response 严格按 "subproblem 1, subproblem 2, ..." 顺序 → performance 下降（45.2 vs 51.3 Pass@1）。

**Intuition**: 
- 严格 format 限制 model 灵活性
- Model 可能想先解某个 subproblem 来启发其他 subproblem
- 严格顺序阻碍创造性解决路径
- 这是 **alignment tax** 的体现

Reference: Alignment tax paper https://arxiv.org/abs/2309.06256

---

## 6. Building Intuition: Why DAC Works

### 6.1 The Exploration Space Argument

**CoT 的 trajectory space**: 一条线性的 reasoning chain。post-training 已经充分探索了这个 space，RL 难以找到新信号。

**DAC 的 trajectory space**: 
- Division: 多种 decomposition strategies
- Conquering: 每个 decomposition 下多种 solution paths
- 组合：$G_d \times G_c$ 个 diverse trajectories

这个 space 是 hierarchical 的，比 CoT 的 linear space 大得多。

### 6.2 The Information Theoretic View

CoT 的 entropy 在 post-training 后已经很低（Figure 7 right），说明 exploration 已经 converge。继续 RL 只能做 exploitation，但 exploitation 已经 saturated。

DAC 的 entropy 保持高位，说明 exploration 仍在进行。RL 有大量信号可以 leverage。

### 6.3 The Meta-Learning View

DAC training 实际上是在教 model 一个 **meta-skill**: "如何把复杂问题拆解成简单问题"。这个 meta-skill 是 task-agnostic 的，可以 transfer 到各种 reasoning scenario。

Mix-RL 实验证证了这一点 - DAC training 提升 CoT performance，说明 meta-skill transfer 发生了。

### 6.4 The Surrogate Reward Theory

Lemma 2.1 是这篇 paper 的理论核心：**final answer correctness 是 subproblem correctness 的 consistent surrogate**。

这解决了一个关键问题：我们没有 subproblem 的 ground-truth，如何训练 subproblem solving？答案是：**不需要**。只要 causal direction $\mathbf{s} \to C$ 成立（subproblem correctness 影响原问题 correctness），reward $C=1$ 就会自动 upweight 正确解 subproblems 的 trajectories。

---

## 7. Related Work Connections

### 7.1 DAC in LLM Reasoning

- **Tree of Thoughts** (Yao et al., 2023) https://arxiv.org/abs/2305.10601: 最早的 DAC-style inference，tree structure exploration
- **Least-to-Most Prompting** (Zhou et al., 2022) https://arxiv.org/abs/2205.10625: prompting-based decomposition
- **Graph of Thoughts** (Besta et al., 2024): graph structure reasoning
- **DeAR** (Xue et al., 2024): decompose-analyze-rethink cycle
- **Seed-Prover** (Chen et al., 2025) https://arxiv.org/abs/2507.23726: DAC for theorem proving
- **DeepSeek-Prover V2** (Ren et al., 2025) https://arxiv.org/abs/2504.21801: subgoal decomposition RL
- **Ladder** (Simonds & Yoshiyama, 2025) https://arxiv.org/abs/2503.00735: recursive decomposition

**This paper 的 unique contribution**: 上述工作都是 inference-time 或 SFT-based，这篇是第一个 **end-to-end RL framework for DAC**。

### 7.2 RL for LLMs

- **PPO** (Schulman et al., 2017) https://arxiv.org/abs/1707.06347: 经典 RL algorithm
- **GRPO** (Shao et al., 2024) https://arxiv.org/abs/2402.03300: group-relative advantage，去掉 critic
- **DAPO** (Yu et al., 2025a) https://arxiv.org/abs/2503.14476: Clip-Higher, token-level loss
- **DeepSeek-R1** (Guo et al., 2025) https://arxiv.org/abs/2501.12948: R1-Zero style training
- **PRIME** (Cui et al., 2025) https://arxiv.org/abs/2502.01456: process reward via implicit rewards
- **VAPO** (Yuan et al., 2025) https://arxiv.org/abs/2504.05118: efficient RL for long CoT

### 7.3 Same Author's Previous Work

- **SWS** (Liang et al., 2025a) https://arxiv.org/abs/2506.08989: weakness-driven problem synthesis
- **Variational Problem Synthesis** (Liang et al., 2025b) https://arxiv.org/abs/2508.14029: self-play with entropy maintenance

这些前作都关注 **RLVR 中的 exploration 和 entropy 问题**，DAC-RL 是这条线的 natural extension - 用 DAC paradigm 打开新的 exploration space。

---

## 8. Potential Extensions & Speculations

### 8.1 Recursive DAC

当前 DAC 是 single-level（divide once, conquer）。自然 extension 是 **recursive DAC**: subproblems 可以再 divide，形成 hierarchical decomposition。

这会带来：
- 更大的 exploration space
- 但训练复杂度指数增长
- 需要新的 reward propagation 机制

### 8.2 Adaptive DAC

根据 problem difficulty 决定是否用 DAC：
- 简单问题用 CoT（更 efficient）
- 难题用 DAC（more powerful）
- Mix-RL 的结果暗示这是可行的

### 8.3 DAC + Tool Use

Subproblem solving 可以调用 tools（calculator, search, code interpreter）。这会进一步扩展 DAC 的 capability。

### 8.4 DAC for Code Generation

不仅是数学，代码生成也可以 DAC：
- Divide: decompose 大 task 成 sub-tasks
- Conquer: 实现 sub-tasks，然后 integrate

### 8.5 DAC + Self-Play

Model 可以：
1. 自己生成 subproblems
2. 自己解 subproblems
3. 用 solutions 作为 feedback 改进 division

这形成一个 self-play loop，可能实现 continuous improvement。

### 8.6 Theoretical Extensions

Lemma 2.1 的 monotonicity assumption 可能太强。现实中：
- 某个 subproblem 解错可能反而启发原问题解决（通过 error correction）
- 某些 subproblems 可能 irrelevant

更精细的 causal model 可能提升 reward design。

---

## 9. Limitations & Open Questions

1. **Computational cost**: 每个 problem 需要 $G_d \times G_c = 32$ rollouts，是 CoT 的 32 倍
2. **Integer answer limitation**: 只评估 integer answer benchmarks，open-ended format 评估困难
3. **Single-level DAC**: 没有探索 recursive decomposition
4. **Cold-start dependency**: 小 model 可能需要 distillation 才能 effective DAC
5. **Format constraint trade-off**: 严格 format 有害，但如何平衡 structure 和 flexibility？

---

## 10. Summary: Key Takeaways

1. **Misalignment is real**: post-trained models 做 DAC inference 性能差，需要 dedicated training
2. **DAC has higher ceiling**: CoT RL 已 saturated，DAC RL 打开新 exploration space
3. **Surrogate reward works**: final answer correctness 是 subproblem correctness 的 valid surrogate
4. **DAC enhances CoT**: meta-skill transfer 发生，DAC training 提升 CoT performance
5. **Diversity > Attempts**: test-time 更 多 decomposition strategies 比 more conquering attempts 更 valuable
6. **Concise + Diverse**: DAC 产生更 compact traces 同时保持 higher entropy
7. **Relaxed reward better**: lower bound reward 防止 greedy behavior 和 reward hacking

这篇 paper 的核心贡献是 **把 DAC 从 inference-time technique 提升为 trainable reasoning paradigm**，并通过 RL 充分释放其潜力。这为 LLM reasoning 的 test-time scalability 提供了新的方向 - 不是单纯增加 CoT length，而是改变 reasoning 的 structure。

---

## References

- Main paper: Liang et al. (2026) - Training LLMs for Divide-and-Conquer Reasoning
- Code: https://github.com/MasterVito/DAC-RL
- DAPO: https://arxiv.org/abs/2503.14476
- GRPO: https://arxiv.org/abs/2402.03300
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- DeepSeek-Prover V2: https://arxiv.org/abs/2504.21801
- Least-to-Most: https://arxiv.org/abs/2205.10625
- Alignment tax: https://arxiv.org/abs/2309.06256
- PPO: https://arxiv.org/abs/1707.06347
- PRIME: https://arxiv.org/abs/2502.01456
- VAPO: https://arxiv.org/abs/2504.05118
- SWS (same authors): https://arxiv.org/abs/2506.08989
- Variational Problem Synthesis: https://arxiv.org/abs/2508.14029
- Ladder: https://arxiv.org/abs/2503.00735
- Beyond-AIME dataset: https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME
- AIME: https://maa.org/math-competitions/aime
- MathArena (HMMT): https://arxiv.org/abs/2505.23281
- DAPO-Math-17k: 见 DAPO paper 附录
