---
source_pdf: Efficient Online Reinforcement Learning with Offline Data.pdf
paper_sha256: a87cb856a5294e71c21474006106fd7cadcff0e986da494535f44b13117c10d0
processed_at: '2026-08-04T01:50:22-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RLPD 人话版

## 一句话说清楚

**你手头有一些别人收集的旧数据（可能是专家演示，也可能是菜鸟随便跑的轨迹），你想用 online RL 训练一个好 policy。传统做法要么先 offline pretrain 再 finetune（麻烦），要么加一堆约束把 policy 锁在旧数据附近（限制探索）。这篇 paper 说：都不用，你就拿 SAC 这种标准 off-policy 算法，加三个小 trick，就能 beat 之前所有复杂方法。**

关键就三个 trick：50/50 混着采样旧数据、critic 加 LayerNorm、用大 ensemble + 多 gradient steps。

参考: https://arxiv.org/abs/2306.07914

---

## 为什么这件事难

先讲清楚问题本质。

Online RL（比如 SAC）训练时候，agent 自己跟环境交互，收集 (s, a, r, s') transition，存到 replay buffer，然后从 buffer 采样做 Bellman backup 更新 Q function。这个流程很干净，因为 replay buffer 里的数据都是当前 policy 自己产生的，distribution 是 matched 的。

现在你有一堆 offline data —— 别人之前收集的，可能是专家演示，也可能是某个菜鸟 policy 随便跑的。你想利用这些数据加速训练。最 naive 的做法（叫 SACfD）就是把 offline data 塞进 replay buffer 初始化，然后正常训练。

问题来了：**这个 naive 做法效果很差**，paper Figure 1 里 "SAC + Offline Data" 曲线明显不如 IQL + Finetuning。

为什么差？三个原因：

1. **Offline data 很快被稀释**：replay buffer 不断被新 online data 填满，offline data 占比越来越小，白塞了
2. **Q function 在 OOD action 上 extrapolate 失控**：offline data 只覆盖 state-action space 的一小部分，但 Bellman backup 要 query $Q(s', a')$ 其中 $a' \sim \pi(\cdot|s')$，这个 $a'$ 很可能是 offline data 没见过的，Q network 在这地方会乱猜，猜出很高很离谱的值
3. **Offline data 利用太慢**：标准 SAC 每个 env step 只做 1 个 gradient step，offline data 里的好东西要很久才能通过 Bellman backup 传播到 Q

传统方法怎么解决这三个问题？

- **IQL + Finetuning**: 先 offline pretrain 一个好 policy，再 online finetune。解决问题 2（pretrain 时用 IQL 的 expectile trick 避免 extrapolation），但多了 pretrain 阶段和它的 hyperparameter
- **AWAC**: 加 KL constraint $\pi \approx \pi_\beta$（behavior policy）。解决问题 2（不让 policy 乱跑到 OOD），但牺牲了 exploration 自由度
- **Off2On**: balanced replay + pessimistic ensemble。复杂

RLPD 的 insight：**这三个问题各有更简单的解法，不需要 pretrain，不需要 constraint**。

---

## 三个 Trick 逐一讲

### Trick 1: Symmetric Sampling（解决问题 1）

最简单的解法：**不要把 offline data 塞进 replay buffer，单独放一个 offline buffer，每次 minibatch 一半从 online buffer 采样，一半从 offline buffer 采样。**

```
每次 gradient step：
  sample 128 个 transition from online replay buffer R
  sample 128 个 transition from offline buffer D
  拼成 256 个，做一次 update
```

就这么简单，没有 hyperparameter（固定 50/50），没有额外计算开销。

**为什么 50/50 有效？** 

直觉上讲，offline data 的作用是持续给 Bellman backup 提供"高质量信号源"。如果你只用 buffer init，offline data 几步之后就被稀释到 1% 以下，基本没用了。50/50 保证 offline data 始终占一半，每个 gradient step 都在用。

理论上 Song et al. 2023 的 "Hybrid RL" paper 证明了 balanced sampling 是 sample-efficient 的必要条件。RLPD 发现 50/50 是个 robust 的 sweet spot，不需要调。

参考: https://openreview.net/forum?id=yyBis80iUuU

**为什么不是 25/75 或其他比例？** Paper Figure 12 做了 ablation：25% offline 在 dense reward locomotion 上 marginal 更好，但在 sparse reward (AntMaze) 上 sample efficiency 明显下降。50% 是 cross-domain 的最佳折中。

**对比 buffer init 的失败模式**：Figure 11 显示，当你有大量 sub-optimal offline data（比如 medium-replay locomotion），buffer init 一开始快（offline data 多），但 asymptotic 卡住——因为 online data 占比太低，policy 被 sub-optimal behavior 拖住出不来。Symmetric sampling 持续注入 50% fresh online data，让 policy 能 escape sub-optimal region。

---

### Trick 2: Critic 加 LayerNorm（解决问题 2，最关键的 insight）

这是 paper 最有 depth 的部分，也是最反直觉的。

#### 问题：Q function 在 OOD action 上会爆炸

Bellman target 是：
$$y = r(s,a) + \gamma Q_{\theta'}(s', a')$$

其中 $a' \sim \pi_\phi(\cdot|s')$，$a'$ 是 policy 在 $s'$ 处采样出来的 action。

关键：$a'$ 可能是 offline data 从来没见过的 action。因为 policy 是 stochastic 的，会 explore 各种 action，包括 offline distribution 之外的。

标准 MLP 的 Q function 在这种 OOD input 上会**unbounded extrapolate**——它会根据训练数据学到的 trend，在 training data 外部继续往上飙。Figure 3 是一个很好的 visualization：用一个 2-layer MLP 拟合圆形区域内的数据 $y = \|x\|$，MLP 在圆外会 extrapolate 出 wild 的高值。

结果：critic 的 target 不断被 OOD action 的高 Q 值拉高，target 越高，critic 跟着越高，正反馈循环，Q value diverge。Figure 2 显示 SAC + symmetric sampling 在 Adroit 任务上 Q 值发散，performance 崩。

#### 传统解法 vs RLPD 的解法

传统 offline RL 方法（CQL, IQL, AWAC）怎么解决？**限制 policy 不让它跑到 OOD action**。

- CQL: 显式 penalty OOD action 的 Q 值
- AWAC: KL constraint 让 $\pi \approx \pi_\beta$
- IQL: 用 expectile 让 Q 学在 behavior data 的范围内

但 RLPD 是 online setting！我们有 exploration 的权利！限制 policy 等于 anti-exploration，这跟 online RL 的精神冲突。

RLPD 的 insight：**我们不需要限制 policy，我们只需要限制 Q function 的 extrapolation**。policy 你随便 explore，但 Q function 不要在没见过的地方乱猜。

怎么实现？**在 Q network 里加 LayerNorm**。

#### LayerNorm 为什么能 bound Q value

数学上，Q network 的结构是：
$$Q_{\theta, w}(s, a) = w^T \text{relu}(\text{LN}_\gamma(\text{LN}_\beta(\psi_\theta(s, a))))$$

简化讲，假设最后一层是：
$$Q(s, a) = w^T \text{relu}(\text{LN}(\psi_\theta(s, a)))$$

其中：
- $\psi_\theta(s, a) \in \mathbb{R}^d$：feature extractor 的输出，$d$ 是 hidden dim
- $\text{LN}$：LayerNorm，对 $\psi$ 做 normalize（减均值除标准差，再 affine）
- $\text{relu}$：激活函数
- $w \in \mathbb{R}^d$：最后一层 weight

LayerNorm 之后，$\text{LN}(\psi_\theta(s, a))$ 的 L2 norm 被 bound 住了（大致是 $\sqrt{d}$ 量级，受 affine parameter 影响）。

那么对任意 input $(s, a)$（包括 OOD action）：

$$\|Q(s, a)\| = \|w^T \text{relu}(\text{LN}(\psi_\theta(s, a)))\|$$

用 Cauchy-Schwarz:
$$\leq \|w\| \cdot \|\text{relu}(\text{LN}(\psi_\theta(s, a)))\|$$

ReLU 不增加 L2 norm（element-wise positive part）：
$$\leq \|w\| \cdot \|\text{LN}(\psi_\theta(s, a))\|$$

LayerNorm 把 norm bound 住：
$$\leq \|w\| \cdot C$$

其中 $C$ 是与 $\psi_\theta$ 输入无关的常数。

**结论：Q value 的绝对值被 $\|w\|$ bound 住，无论 input 是什么。OOD action 不会让 Q 值爆炸。**

这是个 architectural trick，不需要额外 loss term，不需要限制 policy。Policy 仍然可以自由 explore 任何 action，但 Q function 不会在没见过的 action 上给出离谱的高值。

#### 实验证据

Figure 7 最 striking 的实验：在 "Expert Adroit Sparse Tasks" 上只用 22 条专家轨迹（原始 500 条的 4.4%），data 极稀疏极 narrow。

- **With LayerNorm**: 仍能达到 prior SoTA 水平
- **Without LayerNorm**: **完全 collapse，0 progress**

这说明 data 越稀疏越 narrow coverage，Q extrapolation 问题越严重，LayerNorm 越关键。

Figure 3 的 toy example 也很直观：标准 MLP 在 training data 外部 extrapolate 出 wild 的高值，加 LayerNorm 后外部值被 bound 住。

参考: LayerNorm paper https://arxiv.org/abs/1607.06450

#### 更深的 intuition

为什么 LayerNorm 能 bound Q 而 BatchNorm 或 weight decay 不行？

- **BatchNorm**: 依赖 batch 统计量，OOD input 会被 batch 内的其他 data 拉动，bound 不严格
- **Weight decay**: 只在 training time penalty $\|w\|$，但 inference 时 Q 仍可任意大
- **LayerNorm**: inference 时也对 $\psi_\theta(s, a)$ 做 normalize，architectural 保证

这有点像 Wasserstein GAN 用 spectral normalization 保证 discriminator Lipschitz 连续性的思路。LayerNorm 隐式地给 Q function 加了 Lipschitz bound，让 Q function 在 input space 上变化幅度有限。

参考: WGAN https://arxiv.org/abs/1701.07875

更深一层：Bellman operator 的 contraction 性质要求 Q function 是 Lipschitz 的。LayerNorm 通过保持 Lipschitz 性，间接帮助 Bellman iteration 收敛。这个 connection paper 没深入讲，但是个有趣的 theory 方向。

---

### Trick 3: Large Ensemble + High UTD（解决问题 3）

#### 问题：Offline data 利用太慢

标准 SAC: 1 env step → 1 gradient step。这意味 offline data 里的好 transition 要经过很多次 Bellman backup 才能传播到 Q function 的相关区域。

比如 offline data 里有一个 high-reward 的 transition $(s_t, a_t, r, s_{t+1})$，这个 reward 信息要 backup 到 $s_{t-1}, s_{t-2}, ...$ 各个前置 state 的 Q 值。每 gradient step 只 backup 一层，需要很多 step 才能传播开。

解决：**提高 update-to-data (UTD) ratio**，每个 env step 做 G=20 个 gradient step（state-based），让 offline data 快速被 backup。

#### 但 high UTD 有个 paradox

Li et al. 2022 "Efficient deep RL requires regulating statistical overfitting" 指出：提高 UTD 会导致 critic 在有限 data 上 overfit。Q function 记住了 training data 的 noise，generalization 变差，sample efficiency 反而下降。

参考: https://openreview.net/forum?id=Jwfa-oyQduy

#### 解决：Random Ensemble Distillation (REDQ-style)

借鉴 Chen et al. 2021 REDQ 的做法：
- 训练 $E=10$ 个 Q network（standard SAC 只有 2 个）
- 每次 gradient step，从这 10 个里随机 sample 2 个来算 target 的 min

Algorithm 1 line 15:
```
Sample Z ⊂ {1, 2, ..., 10} of size 2
y = r + γ min_{i ∈ Z} Q_{θ'_i}(s', a')
```

为什么这能 prevent overfitting？

**Intuition 1: Random target 添加 noise**
每次 update 的 target 来自随机选的 2 个 critic，target 本身有随机性，critic 学到的是 averaged behavior 而非 memorize 每个 data point。

**Intuition 2: 大 ensemble 的 averaging 效应**
Actor loss 用所有 10 个 critic 的 average Q：
$$L_{actor} = \frac{1}{10} \sum_{i=1}^{10} Q_{\theta_i}(s, \tilde a) - \alpha \log \pi(\tilde a | s)$$

单个 critic 可能 overfit，但 average 后 overfit 的部分被 average out。

**Intuition 3: In-target min 提供 pessimism 但不过分**
随机选 2 个取 min，相比固定 2 个取 min（standard CDQ），pessimism 程度更随机更温和。

参考: REDQ https://openreview.net/forum?id=AY8zfZm0tDd

#### 为什么 Dropout 不行（在 sparse reward 上）

Paper Figure 9 比较 three regularization 方案：
- Weight decay: 全部场景都不如 ensemble
- **Dropout (DropQ)**: dense reward locomotion 上 OK，sparse reward (AntMaze, Adroit) 上 collapse
- **Ensemble**: 全部场景最强

为什么 Dropout 在 sparse reward 上不行？

直觉：Sparse reward 下，Q function 要精确识别少数 high-reward trajectory 的"关键 state"。Dropout 随机关掉 neuron，破坏了 critic 对这些关键 region 的精确识别。Ensemble 通过 averaging 提供稳健性，更 robust。

而 dense reward 下，每个 state 都有 reward 信号，Dropout 的 perturbation 没那么致命。

#### Pixel-based 的额外 trick

对 V-D4RL 这种 vision-based 任务，RLPD 还加：
- **Random shift augmentation** (DrQ)：pixel input 的 regularization，随机平移图像几个 pixel
- UTD 可以提到 10（state-based 是 20）

Figure 6 显示 Cheetah Run Expert 上 UTD=10 效果显著。

这是首次证明 model-free pixel-based continuous control 上 high UTD 有效的 paper。

参考: DrQ https://openreview.net/forum?id=GY6-6sTvGaf

---

## Per-Environment Design Choices（paper Section 4.4）

除了三个 universal trick，paper 还发现有些 design choice 是 environment-specific 的，不能一套打天下。这部分是 paper 另一个实用贡献。

### CDQ (Clipped Double Q-Learning)

Standard SAC 用 $\min(Q_1, Q_2)$ 作为 target，这相当于 underestimate 1 std。

发现：
- **Locomotion (dense reward)**: 用 CDQ (subset 2) 好
- **AntMaze (sparse reward, long horizon)**: **不用 CDQ (subset 1) 好**
- **Adroit**: 用 CDQ 好

**Intuition**: AntMaze 这种 long-horizon navigation，需要找到"罕见但 high reward"的 path。CDQ 的 conservatism 会低估这些 rare path 的 value，导致 policy 不愿意去 explore 它们。Dense reward 下没这个问题，因为 reward 到处都是。

### Entropy Backups

Standard SAC 在 target 里加 entropy: $y = r + \gamma(\min Q + \alpha H(\pi))$。

发现：
- Locomotion: 保留 entropy backup
- **AntMaze / Adroit / DMC Pixels**: **移除 entropy backup**

注意：actor loss 里仍然保留 entropy 项（$Q(s, \tilde a) - \alpha \log \pi(\tilde a|s)$），只是 target 里不加。

**Intuition**: Entropy backup 让 Q function 把"高熵 action"也当作有 reward。Dense reward 下这帮助 explore。Sparse reward 下，Q function 会学到"乱动 = 高 value"的错觉，破坏 value 估计准确性。移除 entropy backup 让 Q 更纯粹反映 reward，actor 端的 entropy 仍保证 exploration。

### Architecture Depth

- Locomotion, DMC Pixels: 2 layer MLP
- **AntMaze, Adroit: 3 layer MLP**

**Intuition**: 复杂 task 需要更深网络做 representation，但深网络在少 data 上更难训。Locomotion data 多，2 layer 更稳；AntMaze/Adroit 需要更深 capture 复杂 dynamics。

### 推荐 Workflow

Paper 推荐按这个顺序 ablate：
1. CDQ on/off
2. Entropy backup on/off  
3. 2 vs 3 layer MLP

按对 performance 影响大小排序。

---

## 完整 Algorithm (Algorithm 1 人话版)

```
准备：
  - Q networks: 10 个，每个带 LayerNorm
  - replay buffer R (空)
  - offline buffer D (预先填好 offline data)
  - hyperparams: γ=0.99, α=1.0, ρ=0.005, UTD=20

循环：
  收到 initial state s_0
  for t = 0, 1, 2, ...:
    # Online 交互
    a_t ~ π(·|s_t)  # actor 采样 action
    执行 a_t，得 r_t, s_{t+1}
    把 (s_t, a_t, r_t, s_{t+1}) 存进 R
    
    # 做 20 个 gradient step
    for g = 1, ..., 20:
      # Symmetric sampling
      b_R = sample 128 from R
      b_D = sample 128 from D
      b = concat(b_R, b_D)
      
      # Random subset for target (REDQ-style)
      Z = random 2 indices from {1, ..., 10}
      
      # Compute target
      for each (s,a,r,s') in b:
        ã' ~ π(·|s')  # actor 采样 next action
        y = r + γ * min_{i ∈ Z} Q_{θ'_i}(s', ã')
        # [Optional, env-specific] y += γα log π(ã'|s')
      
      # Update all 10 critics
      for i = 1, ..., 10:
        θ_i ← minimize (y - Q_{θ_i}(s, a))²
      
      # EMA update targets
      for i = 1, ..., 10:
        θ'_i ← 0.005 * θ_i + 0.995 * θ'_i
    
    # Update actor (用所有 10 个 critic 的 average)
    ã ~ π(·|s)
    ϕ ← maximize (1/10) Σ_i Q_{θ_i}(s, ã) - α log π(ã|s)
```

关键点：
- 10 个 critic 全部更新，但每次 target 只随机用 2 个 → REDQ 思想
- Actor loss 用所有 10 个 critic 的 average，不取 min → actor 不被某个悲观 critic 主导
- Symmetric sampling 在 line 12-14
- LayerNorm 在每个 Q network 的 hidden layer 后

---

## 实验结果总结

### 三个 benchmark domain

| Domain | Tasks | Prior SoTA | RLPD | 改进 |
|--------|-------|-----------|------|------|
| Sparse Adroit | 3 (Pen, Door, Relocate) | IQL + FT | RLPD | Door 上 2.5× |
| D4RL AntMaze | 6 (Umaze, Medium, Large × diverse/play) | IQL + FT | RLPD | 首次全部 solve，1/3 timesteps |
| D4RL Locomotion | 12 (halfcheetah, walker2d, hopper × medium/med-expert/med-replay) | Off2On | RLPD | 匹配或超越 |

### Pixel-based (V-D4RL)

在 Walker Walk, Cheetah Run, Humanoid Walk 上，对比 BC, Online (同 architecture 无 offline data), DrQ-v2。RLPD 一致最强或匹配。

特别值得注意：Humanoid Walk 这种 partial observability 任务（body part 遮挡），RLPD 远好于 BC——online exploration 补全了 BC 无法从 pixel 中提取的信息。

### LayerNorm ablation (Figure 7)

最 striking：**22 条 expert demos 的 Adroit**
- With LN: 达到 SoTA
- Without LN: 完全 collapse

说明 data 越窄越稀，LayerNorm 越关键。

### Symmetric sampling vs buffer init (Figure 10, 11)

Figure 10 的 Pen vs Door 对比有意思：
- Pen: symmetric sampling 通过持续注入 reward-bearing transition 提高 batch 内 reward density → 加速 exploration
- Door: 不通过 reward density，而通过降低 critic target variance → 提高 stability

Figure 11: 大量 sub-optimal data 场景，buffer init asymptotic 卡住（被 sub-optimal behavior 锁死），symmetric sampling 持续注入 50% fresh online data，让 policy escape sub-optimal region。

---

## 跟其他思想的联系

### 跟 Offline RL 的关系

Offline RL 的核心痛点是 distribution shift——不能 explore，所以必须限制 policy 不跑出 data coverage。传统方法（CQL, IQL, AWAC）都用 explicit constraint。

RLPD 的 insight：**只要你能 online explore，就不需要限制 policy，只要限制 Q function 的 extrapolation**（通过 LayerNorm 这种 architectural trick）。

这说明 offline RL 之所以难，核心是"不能 explore"这个约束本身。一旦能 explore，问题简化很多。

### 跟 RLHF 的关系

现代 LLM 的 RLHF (InstructGPT) 流程：
1. SFT
2. Reward model  
3. PPO with KL constraint to SFT model

这本质就是 "online RL with offline demo data + policy constraint"。RLPD 暗示：**如果用 LayerNorm 在 critic + symmetric sampling 混合 demo data，可能不需要 KL constraint**。

这是个潜在应用方向。Anthropic 的 Constitutional AI 思路也类似——用 feedback data 持续训练。

参考: InstructGPT https://arxiv.org/abs/2203.02155

### 跟 Pretrain-Finetune 范式的关系

NLP/CV 里 pretrain + finetune 是 dominant。RL 里对应是 offline pretrain + online finetune。

RLPD 发现：**在 RL 中这个范式不必要**，直接 online + offline data 混合训练 + 好 design，可以 beat pretrain-then-finetune。

Intuition: RL pretrain 难是因为 offline RL 的 Q extrapolation 问题。LayerNorm 解决了这个问题，pretrain 就不必要了。

### 跟 Model-Based RL 的关系

MBRL (MBPO) 用 learned model rollouts 增加 sample efficiency。RLPD 用 offline data 达到类似效果，但不需要 model——避免 model bias。

两者本质都是"用 extra data 加速 Bellman backup"，但 RLPD 更简单。

### 跟 Decision Transformer 的关系

DT 把 RL 当 sequence modeling，不需要 Q function，所以没 extrapolation 问题。但 DT 需要 near-optimal data。RLPD 在 sub-optimal data 上更强（AntMaze），因为 online exploration 可以 "stitch" trajectories。

---

## 对你的可能关注点的延伸

### Implementation Matters 精神

Engstrom et al. 2020 "Implementation Matters in Deep RL" 强调 RL 对 implementation 细节敏感。RLPD 延续这种 spirit——"SAC + offline data 不 work"不是 algorithm 不行，是 implementation 细节没对（CDQ on/off, entropy backup on/off, LayerNorm）。

这跟 software 2.0 思想一致：data-driven programming 中，hyperparameter / architectural choice 就是 program 的一部分。

参考: https://openreview.net/forum?id=r1etN1rtPB

### minGPT / minRLPD 教学项目

RLPD 实现简单（Algorithm 1 只 25 行），适合做 minGPT 之后的教学项目 minSAC-RLPD。LayerNorm bound Q value 的 derivation 是个 elegant 小数学 exercise，适合教学。

JAX 实现，单 GPU 可跑，复现成本低。

### Eureka / LLM-as-RL-designer

Eureka (Nvidia 2023) 用 LLM 设计 reward function。RLPD 的 design choice ablation 启示：LLM 可以进一步做 environment-specific hyperparameter selection（CDQ on/off, entropy on/off 等），自动化 paper Section 4.4 的 workflow。

### Q-Diffusion 类工作

Q-Diffusion 用 Q function guide diffusion sampling。RLPD 的 LayerNorm insight 可能帮助这类方法的 stability。

---

## Limitations

Paper 没讨论的：
- 没有理论分析（只 empirical + 单个 LayerNorm bound argument）
- LayerNorm 的 affine parameter 对 bound 的影响没讨论
- 没跟 model-based 方法对比
- 没在 real robot 上验证
- 只考虑 continuous control，discrete action (Atari) 没试
- Offline data quality 远低于 random 时，symmetric sampling 是否还 robust？

可能的 future work：
- LayerNorm vs spectral norm / weight norm 的系统比较
- 结合 representation learning (contrastive loss)
- Multi-task / Meta-RL 场景
- Diffusion-based planner + RLPD symmetric sampling

---

## 最直觉的 Mental Model

用比喻总结：

**Offline data 是"地图"，online exploration 是"探险"。**

- **Pure online RL**: 没地图瞎探险，效率低
- **Pure offline RL**: 只有地图不探险，被地图限制
- **AWAC / 约束方法**: 有地图但被强制按地图走，探险受限
- **Pretrain + Finetune**: 先背地图再探险，但背地图可能背错
- **RLPD**: 地图始终在手边参考（symmetric sampling），自由探险（不约束 policy），但有"理智判断力"防止被地图外的幻觉误导（LayerNorm bounds Q）

这个 mental model 解释了 RLPD 既 sample efficient（地图加速）又 exploratory（自由行动）又 stable（LayerNorm 防 hallucination）。

---

## One Takeaway

如果只记一件事：**LayerNorm 在 critic 里不只是稳定训练的 trick，它通过 architectural constraint 隐式 bound 了 Q function 的 extrapolation，替代了传统 offline RL 的 explicit policy constraint。这让 online RL 能自由 explore 同时安全利用 offline data。**

这是个"simpler is better"的典范——在 RL 这个 complex field，找到 simple 但 effective 的 design choice 是最有价值的工作。

参考: 
- Paper: https://arxiv.org/abs/2306.07914
- Code: https://github.com/ikostrikov/rlpd
- SAC: https://arxiv.org/abs/1801.01290
- REDQ: https://openreview.net/forum?id=AY8zfZm0tDd
- IQL: https://openreview.net/forum?id=68n2s9ZJWF8
- AWAC: https://arxiv.org/abs/2006.09359
- CQL: https://arxiv.org/abs/2006.04779
- LayerNorm: https://arxiv.org/abs/1607.06450
- DrQ-v2: https://openreview.net/forum?id=_SJ-_yyes8
- D4RL: https://arxiv.org/abs/2004.06139
- Hybrid RL: https://openreview.net/forum?id=yyBis80iUuU
- Implementation Matters: https://openreview.net/forum?id=r1etN1rtPB
- InstructGPT: https://arxiv.org/abs/2203.02155
- DropQ: https://openreview.net/forum?id=xCVJMsPv3RT
- Off2On: https://openreview.net/forum?id=AlJXhEI6J5W
- WGAN: https://arxiv.org/abs/1701.07875

---

# RLPD: Reinforcement Learning with Prior Data 深度解析

## 1. Paper 的核心 motivation

这篇 paper 想回答一个非常实际的问题：**是否可以直接用现有的 off-policy RL 算法（如 SAC）来利用 offline data，而不需要 offline pre-training 或 explicit constraint（如 AWAC 的 KL 约束）？**

答案是可以，但是 naive 地用 SAC + offline data 会导致性能很差（见 paper Figure 1 中的 "SAC + Offline Data" 曲线）。作者通过 systematic 的 ablation 发现了**3 个关键 design choices**，组合起来可以达到 SOTA，并且在一些 task 上超过 prior work 2.5×。

这条路线相对于 prior work 的优势：
- **AWAC (Nair et al., 2020)**: 用 KL constraint 把 policy 约束在 offline data 附近 → 限制了 exploration
- **IQL + Finetuning (Kostrikov et al., 2022)**: 先 offline pre-train，再 online fine-tune → 多了 pre-training 阶段和它的 hyperparameter
- **Off2On (Lee et al., 2021)**: balanced replay + pessimistic Q-ensemble → 复杂

RLPD 的核心思想：**保持 online RL 的"自由度"，但通过 architectural choices 隐式地解决 offline data 带来的 distribution shift 问题**。

参考链接：
- Paper: https://arxiv.org/abs/2306.07914
- Code: https://github.com/ikostrikov/rlpd
- IQL: https://openreview.net/forum?id=68n2s9ZJWF8
- AWAC: https://arxiv.org/abs/2006.09359

---

## 2. 三个核心 Design Choices 深入讲解

### 2.1 Design Choice 1: Symmetric Sampling

**核心机制**：每个 gradient step 的 minibatch 中，50% 来自 online replay buffer $\mathcal{R}$，50% 来自 offline data buffer $\mathcal{D}$。

```
batch size N = 256
b_R ← sample N/2 = 128 from R (online replay)
b_D ← sample N/2 = 128 from D (offline)
b = concat(b_R, b_D)
```

**为什么这个简单方案有效？** 从理论上讲，Song et al. (2023, "Hybrid RL") 证明了 balanced sampling 是 sample-efficient 的必要条件。RLPD 的贡献是发现 50/50 是一个 robust 的"通用 sweet spot"，无需调参。

**与 SACfD (Vecerík et al., 2017) 的对比**：SACfD 只是把 offline data 塞进 replay buffer 初始化，随着 training 进展 offline data 被 dilute。Figure 10-11 显示这种 approach 在 sparse reward 任务上 sample efficiency 差，且在 large offline data 场景下 asymptotic performance 差（online data 比例太低）。

**Intuition**：symmetric sampling 确保了 offline data 的"持续在场"，让 Bellman backup 能反复从 high-quality transitions 传播 value。同时 online data 提供 exploration 的新信息。两者通过 50/50 的 ratio 持续混合，避免了"用过即弃"。

**为什么 50% 而不是其他？** Figure 12 ablation 显示 25% offline 会让 sparse reward 任务 sample efficiency 下降，而 100% offline（纯 offline RL）直接 collapse — 这证明 RLPD 不是 offline method，关键在于 online exploration + offline 数据的有效混合。

参考：Song et al. 2023 "Hybrid RL": https://openreview.net/forum?id=yyBis80iUuU

---

### 2.2 Design Choice 2: LayerNorm 抑制 Value Over-extrapolation（最重要的 insight）

这是这篇 paper 最有 depth 的部分。

#### 2.2.1 问题本质

Standard off-policy RL 在 Bellman backup 时会 query Q-function 在 OOD action 上的值：
$$a' \sim \pi(\cdot|s'), \quad y = r + \gamma Q_{\theta'}(s', a')$$

由于 $\pi$ 是 stochastic 的，它会 sample 到 offline data 不支持的 action $(s', a')$。Function approximation 在 OOD 区域会**unconstrained extrapolate**，导致 Q 值爆炸性 overestimation（Thrun & Schwartz 1993）。

**关键观察**：在 offline RL 中，这个问题的解决是**限制 policy**（如 AWAC 的 KL 约束，IQL 的 expectile，CQL 的 conservative penalty）。但 RLPD 的 insight 是：**在线 setting 下，我们不需要限制 policy**（policy 应该自由 explore！），**我们只需要限制 Q-function 的 extrapolation**。

#### 2.2.2 LayerNorm 为什么能 bound Q value

考虑 Q-network 的参数化：
$$Q_{\theta, w}(s, a) = w^T \text{relu}(\text{LN}(\psi_\theta(s, a)))$$

其中：
- $\psi_\theta: \mathcal{S} \times \mathcal{A} \to \mathbb{R}^d$ 是 feature extractor（参数 $\theta$）
- $\text{LN}$ 是 LayerNorm（在 feature 维度上归一化）
- $\text{relu}$ 是激活函数
- $w \in \mathbb{R}^d$ 是 output layer

**推导 Q value 的上界**：

$$\|Q_{\theta, w}(s, a)\| = \|w^T \text{relu}(\text{LN}(\psi_\theta(s, a)))\|$$

由 Cauchy-Schwarz 不等式：
$$\leq \|w\| \cdot \|\text{relu}(\text{LN}(\psi_\theta(s, a)))\|$$

由 ReLU 的性质（不会增加 L2 norm，因为是 element-wise 的 positive part）：
$$\leq \|w\| \cdot \|\text{LN}(\psi_\theta(s, a))\|$$

由 LayerNorm 的性质（将 vector 归一化，norm 为 $\sqrt{d}$，其中 $d$ 是 feature dim，但 affine 后可能改变；如果只看 scale，有 bound）：
$$\leq \|w\| \cdot C$$

其中 $C$ 是与 $\psi_\theta$ 无关的常数。这意味着 **Q value 对任意 input $(s, a)$，包括 OOD action，都被 $\|w\|$ bound**。

#### 2.2.3 为什么这个 bound 重要

对比 standard MLP（无 LayerNorm）：
$$Q_{\theta, w}(s, a) = w^T \text{relu}(\psi_\theta(s, a))$$

$\psi_\theta(s, a)$ 在 OOD 区域可以任意大，导致 Q 值任意大。Figure 3 是一个漂亮的 visualization：在一个 2D 圆形 data 区域，标准 MLP 在外部 extrapolate 出 wild 的高值，而 LayerNorm 的 MLP 在外部保持 bounded。

#### 2.2.4 与其他方法的对比

- **CQL (Kumar et al., 2020)**: 显式 penalty OOD action 的 Q 值 → 限制了 Q function 的 learning，不利于 exploration
- **AWAC**: 限制 policy $\pi$ 接近 behavior $\pi_\beta$ → 限制了 exploration
- **LayerNorm**: 只限制 Q 的 Lipschitz 性质，**policy 仍可以自由 explore**

这是一种 **implicit regularization** —— 通过 architecture 实现，无需额外 loss term。这非常 elegant。

#### 2.2.5 实验证据

Figure 7 中 "Expert Adroit Sparse Tasks"（22 trajectories）的实验非常 striking：
- With LayerNorm: 持续进步，达到 prior SoTA 水平
- Without LayerNorm: **完全 collapse**，无法学习

这说明在 data 稀疏且 narrow coverage 时，Q 的 extrapolation 是 fatal 的。

参考：LayerNorm paper: https://arxiv.org/abs/1607.06450

---

### 2.3 Design Choice 3: Sample-Efficient Learning (Large Ensemble + High UTD)

#### 2.3.1 为什么需要 high update-to-data (UTD) ratio

Online RL 的 sample efficiency 瓶颈在于：每个 environment step 只做 1 个 gradient step，那 offline data 需要"long time"才能被 Bellman backup 传播到 Q function。

**High UTD 的 paradox**：提高 gradient steps per env step 会让 sample efficiency 变好（更多 update 用上 offline data），但会 overfit（Li et al., 2022 "Efficient deep RL requires regulating statistical overfitting"）。

#### 2.3.2 解决方案：Random Ensemble Distillation (REDQ-style)

RLPD 采用了 Chen et al. (2021) REDQ 的策略：
- **Large ensemble** of Q-functions: $E = 10$（standard SAC 只用 2）
- 每次 update 随机 sample $Z \in \{1, 2\}$ 个 critic 来计算 target 的 min（in-target minimization）

Algorithm 1 line 15: 
$$\text{Sample } \mathcal{Z} \text{ of size } Z \text{ from } \{1, 2, ..., E\}$$
$$y = r + \gamma \min_{i \in \mathcal{Z}} Q_{\theta'_i}(s', \tilde a')$$

**Intuition**：大 ensemble + random subset 在 target 上提供了 implicit regularization，避免了 overfitting 同时允许 high UTD。REDQ 证明这种 ensemble 是 unbiased 的，不会引入额外的 bias。

#### 2.3.3 为什么 Dropout 不行

Figure 9 的 ablation 比较：
- **Weight decay**: 全都不如 ensemble
- **Dropout (Hiraoka et al., 2022, DropQ)**: 在 dense reward locomotion 上 OK，但在 sparse reward (AntMaze, Adroit) 上 collapse
- **Ensemble**: 全部场景最强

**Intuition**: Dropout 在 sparse reward 上可能破坏了 critic 对 "good trajectory" 的精确识别。Ensemble 通过 averaging 提供 robustness，更适合 sparse reward 的 critical 区域识别。

#### 2.3.4 Pixel-based 任务的特殊处理

对于 V-D4RL (vision-based) 任务，RLPD 还加了：
- **Random shift augmentation** (Kostrikov et al., 2021, DrQ): pixel input 的 regularization
- UTD 可以提到 10（Figure 6 显示 Cheetah Run Expert 上效果显著）

这是首次 model-free pixel-based continuous control 上证明 high UTD 有效的 paper。

参考：
- REDQ: https://openreview.net/forum?id=AY8zfZm0tDd
- DrQ: https://openreview.net/forum?id=GY6-6sTvGaf
- DrQ-v2: https://openreview.net/forum?id=_SJ-_yyes8

---

## 3. Environment-Specific Design Choices (Paper Section 4.4)

这部分是 paper 的另一大贡献：**指出某些 design choices 是 environment-specific 的，不应该 inherited from prior implementations**。

### 3.1 Clipped Double Q-Learning (CDQ)

Standard SAC 用 $\min_{i=1,2} Q_{\theta_i}$ 作为 target。这等价于 underestimating 1 std below true target。

Paper 发现：
- **Locomotion**: CDQ 帮助
- **AntMaze**: CDQ **有害**！需要 subset 1 critic（不用 min）
- **Adroit**: CDQ 帮助

**Intuition**: 在 sparse reward + 需要精细 exploration 的任务上，CDQ 的 conservatism 会阻碍发现"罕见但 high reward" 的 path。AntMaze 这种 long-horizon navigation 任务特别敏感。

### 3.2 Maximum Entropy Backups

Standard SAC 在 target 中加 entropy: $y = r + \gamma(\min Q + \alpha H(\pi))$。

Paper 发现：
- **Locomotion**: 保留 entropy backup
- **AntMaze / Adroit / DMC Pixels**: **移除 entropy backup**（但保留 entropy 在 actor loss 中）

**Intuition**: entropy backup 在 target 中相当于 "假想" reward 是 $r + \alpha H$，这在 dense reward 下帮助 exploration。但在 sparse reward 下，entropy backup 会让 Q function 学到"高熵 = 高 reward"的错觉，破坏 value 估计的准确性。

### 3.3 Architecture Depth

- Locomotion, DMC Pixels: 2 layer MLP
- AntMaze, Adroit: 3 layer MLP

**Intuition**: 更深的网络在复杂任务上提供更好的 representation capacity，但需要更多 data 来训练。在数据丰富的 locomotion 上 2 layer 更稳定。

### 3.4 Workflow 推荐

Paper 推荐 ablation 顺序：
1. **Subset 2 critics (CDQ)** vs subset 1
2. **Remove entropy** backups vs keep
3. **3-layer** vs 2-layer MLP

这个顺序很 pragmatic，按对 performance 影响最大的顺序排。

---

## 4. Algorithm 详细分析 (Algorithm 1)

让我逐行解析关键部分：

```
1: Select LayerNorm, Large Ensemble Size E, Gradient Steps G, architecture
2: Init Critics θ_i (i=1,...,E) with targets θ'_i = θ_i, Actor ϕ
   Discount γ, temperature α, critic EMA weight ρ
3: Determine subset size Z ∈ {1, 2}    # CDQ or not
4: Init empty replay buffer R
5: Init buffer D with offline data
6: while True:
7:   Receive s_0
8:   for t = 0, T:
9:     a_t ~ π_ϕ(·|s_t)
10:    Store (s_t, a_t, r_t, s_{t+1}) in R
11:    for g = 1, G:    # G gradient steps per env step (high UTD)
12:      Sample b_R of N/2 from R    # symmetric sampling
13:      Sample b_D of N/2 from D
14:      Combine into batch b of size N
15:      Sample set Z of Z indices from {1,...,E}    # REDQ random subset
16:      y = r + γ min_{i∈Z} Q_{θ'_i}(s', ã')   # ã' ~ π_ϕ
17:      [Optional] y = y + γα log π_ϕ(ã'|s')   # entropy backup (env-specific)
18:      for i = 1, E:
19:        Update θ_i minimizing L = (1/N) Σ (y - Q_{θ_i}(s,a))²
20:      end
21:      Update targets: θ'_i ← ρθ'_i + (1-ρ)θ_i
22:    end
23:    Update ϕ maximizing (1/E) Σ_i Q_{θ_i}(s, ã) - α log π_ϕ(ã|s)
24:  end
25: end
```

**关键观察**：
- Line 11-22 是 high UTD inner loop（G=20 for state-based, up to 10 for pixels）
- Line 18-19 是 large ensemble update（E=10）
- Line 23 中 actor loss 用所有 E 个 critic 的 average，而非 min — 这是 REDQ 的设计，让 actor 不会被 ensemble 中某个悲观的 critic 主导

---

## 5. 实验数据深度解读

### 5.1 主实验结果 (Figure 4)

| Domain | Prior SoTA | RLPD | Improvement |
|--------|-----------|------|-------------|
| Sparse Adroit (3 tasks) | IQL + Finetune | RLPD | Door 上 2.5× |
| D4RL AntMaze (6 tasks) | IQL + Finetune | RLPD | 首次全部 solve，用 1/3 timesteps |
| D4RL Locomotion (12 tasks) | Off2On | RLPD | 匹配或超越 |

### 5.2 Pixel-based (Figure 5)

在 V-D4RL 上，对比：
- **BC**: behavior cloning baseline
- **Online**: 纯 online RL (同 architecture)
- **DrQ-v2**: vision-based SOTA
- **RLPD**: 一致最优

特别值得注意：在 "Humanoid Walk" 这种 partial observability 的任务（agent 身体部分遮挡），RLPD 比 BC 强很多 — 因为它通过 online exploration 补全了 BC 无法捕捉的信息。

### 5.3 LayerNorm Ablation (Figure 7)

最 striking 的实验是 **Expert-only Adroit**（22 trajectories）：
- **RLPD (with LN)**: 仍然达到 prior SoTA
- **RLPD (no LN)**: 完全 collapse，0 progress

这证明了在 narrow data coverage 下，Q extrapolation 是致命的。

### 5.4 Symmetric Sampling vs Buffer Init (Figure 10, 11)

Figure 10 的 Pen vs Door 对比很有启发：
- **Pen**: symmetric sampling 通过持续注入 reward-bearing transitions 提高 reward density → 加速 exploration
- **Door**: symmetric sampling 不通过 reward density，而是通过**降低 critic target variance** 提高 stability

Figure 11: 在大量 sub-optimal data 场景下，buffer init 的 asymptotic performance 比 symmetric sampling 差 — 因为 buffer init 后 offline data 被 dilute，policy 无法 escape sub-optimal behavior 的"引力"。

---

## 6. 与其他思想的联系

### 6.1 与 Offline RL 的关系

RLPD 提供了一个有趣的视角：offline RL 的核心问题是 distribution shift，传统方法通过 **显式约束** 解决（CQL 的 penalty，IQL 的 expectile，AWAC 的 KL）。RLPD 证明：**只要你能 online explore，就只需 architectural constraint on Q function**（LayerNorm），不需要 policy constraint。

这暗示了 offline RL 的难度本质：限制 policy 是不得已的，因为不能 explore。如果有 explore 权利，问题简化很多。

参考：
- CQL: https://arxiv.org/abs/2006.04779
- IQL: https://openreview.net/forum?id=68n2s9ZJWF8

### 6.2 与 Pretrain-Finetune 范式的对比

NLP/CV 中 pretrain + finetune 是 dominant 范式。RL 中类似的是 offline pretrain + online finetune。RLPD 的发现是：**在 RL 中，这种范式不必要**，直接 online + offline data 混合训练，配合好的 design choices，可以更好。

**Intuition**: RL 的 pretrain 之所以困难，是因为 offline RL 的 Q extrapolation 问题。RLPD 用 LayerNorm 解决了这个问题，让 pretrain 变得不必要。这类似于 NLP 中发现 prompt tuning 可以替代 pretrain-then-finetune 范式。

### 6.3 与 LLM RLHF 的联系

Ouyang et al. (2022) 的 RLHF（InstructGPT）流程：
1. SFT (supervised fine-tune)
2. Reward model
3. PPO with KL constraint to SFT model

这其实就是 "offline data (demonstrations) + online RL with policy constraint" 的实例。RLPD 的发现暗示：**如果用 LayerNorm 在 critic 中 + symmetric sampling，可能不需要 KL constraint**。这是一个潜在的应用方向。

参考：InstructGPT: https://arxiv.org/abs/2203.02155

### 6.4 与 Decision Transformer / Trajectory Modeling 的对比

Decision Transformer 把 RL 当作 sequence modeling，不需要 Q function，所以没有 extrapolation 问题。但 DT 需要 near-optimal data。RLPD 在 sub-optimal data 上更强（AntMaze），因为它通过 online exploration "stitch" trajectories。

### 6.5 与 Model-Based RL 的对比

MBRL (如 MBPO) 用 model rollouts 增加 sample efficiency。RLPD 用 offline data 达到类似效果，但不需要 model — 这避免了 model bias 问题。两者本质都是"利用 extra data 加速 Bellman backup"。

### 6.6 关于 LayerNorm 在 RL 中的更深意义

LayerNorm 在 supervised learning 中主要用于训练稳定性。RLPD 发现了一个 novel use case：**作为 Q function Lipschitz constant 的隐式 regularizer**。

这引出一系列问题：
- BatchNorm 是否有类似效果？（答案：可能 less，因为 batch-dependence 引入 noise）
- Weight normalization 呢？（可能 similar，但 paper 没系统研究）
- Spectral normalization（explicit Lipschitz bound）会更好吗？（值得探索）

Lipschitz-constrained networks 在 GAN (Wasserstein GAN) 中是 critical 的。RL 中 Q function 的 Lipschitz 性质理论上与 Bellman contraction 相关 — LayerNorm 可能通过保持 Lipschitz 性间接帮助 convergence。

参考：Wasserstein GAN: https://arxiv.org/abs/1701.07875

---

## 7. 对 Practitioner 的建议总结

如果你要在新 task 上用 RLPD：

1. **必须**：LayerNorm in critic + actor
2. **必须**：Symmetric sampling (50/50 online/offline)
3. **必须**：Large ensemble (E=10) + high UTD (G=20 for state-based)
4. **必须**：Random shift augmentation (if pixel-based)

5. **按顺序 ablate**:
   - CDQ (subset 2 vs 1)
   - Entropy backups (keep vs remove)
   - Architecture (2 vs 3 layer)

对于 sparse reward + long horizon → 倾向 no CDQ, no entropy backup, 3 layer
对于 dense reward → 倾向 CDQ, entropy backup, 2 layer

---

## 8. 限制与未来方向

### 8.1 Paper 的局限
- 没有理论分析（只是 empirical + 单个 LayerNorm bound 的 argument）
- LayerNorm 的 affine parameter 是否影响 bound 没讨论
- 没有与 model-based 方法 (如 MBPO with offline data) 对比
- 没有在 real-world robot 上验证

### 8.2 可能的扩展
- **Discrete action**: paper 只考虑 continuous control。LayerNorm 在 discrete action setting（如 Atari）效果如何？
- **Multi-task / Meta-RL**: offline data from different tasks
- **Offline data quality 的定量分析**: 当 data quality 远低于 random 时，symmetric sampling 还有效吗？
- **LayerNorm vs 其他 norm 的系统比较**: spectral norm, weight norm, group norm
- **结合 representation learning**: contrastive loss on state representation + RLPD

### 8.3 与 Decision Diffusion / Diffuser 的潜在结合
Diffusion-based planners (Janner et al., 2022) 可以用 offline data 生成 diverse plans。RLPD 的 symmetric sampling 思想可以用于 diffusion model 的 training data composition。

### 8.4 Q-Diffusion 类工作的关联
Q-Diffusion 用 Q function guide diffusion sampling。RLPD 的 LayerNorm insight 可能帮助这类方法的 stability。

---

## 9. Intuitive Summary

如果用一个 mental model 理解 RLPD：

**Offline data 是"地图"，online exploration 是"探险"。**

- **Pure online RL**：没有地图，瞎探险，效率低
- **Pure offline RL**：只有地图，不探险，被地图限制
- **AWAC / 约束方法**：有地图，且被强制按地图走 — 探险受限
- **Pretrain + Finetune**：先背地图，再探险 — 但背地图可能背错
- **RLPD**：地图始终在手边参考（symmetric sampling），自由探险（不约束 policy），但有"理智的判断力"防止被地图外的"幻觉"误导（LayerNorm bounds Q）

这个 mental model 解释了为什么 RLPD 既 sample efficient（地图加速）又 exploratory（自由行动）又 stable（LayerNorm 防 hallucination）。

---

## 10. 对 Karpathy 你可能的关注点的延伸

考虑到你的背景，几个特别相关的点：

### 10.1 关于 Implementation Matters
Engstrom et al. 2020 "Implementation Matters in Deep RL" 强调 RL 对 implementation 细节的敏感性。RLPD 是这种 spirit 的延续 — 它发现 "SAC + offline data 不 work" 不是 algorithm 不行，是 implementation 细节没对（CDQ, entropy, LayerNorm）。这与 software 2.0 的思想一致：在 data-driven programming 中，"hyperparameter / architectural choice" 就是 program 的一部分。

### 10.2 关于 MicroGrad / minGPT 类教学项目的意义
RLPD 的实现非常 simple（Algorithm 1 只有 25 行），适合作为 minGPT 之后的教学项目 minSAC-RLPD。LayerNorm bound Q value 的 derivation 是一个 elegant 的小数学 exercise，适合教学。

### 10.3 关于 nn-zero-to-hero 系列
如果你做 RL 版本的 zero-to-hero，RLPD 是一个很好的 "intro to modern RL" 主题，因为它：
- 简单（基于 SAC）
- 但揭示深刻问题（Q extrapolation, distribution shift）
- 有 actionable insights（三个 design choices）
- 复现成本低（JAX 实现，单 GPU 可跑）

### 10.4 关于 Eureka / LLM-as-RL-designer
Eureka (Nvidia, 2023) 用 LLM 设计 reward function。RLPD 的 design choice ablation 启示：LLM 可以进一步用来做 environment-specific hyperparameter selection（CDQ on/off, entropy on/off 等），自动化 paper Section 4.4 的 workflow。

### 10.5 关于 RLHF 的联系
现代 LLM 的 RLHF 实际上是 "online RL with offline preference data"。RLPD 的 symmetric sampling 思想可能直接适用 — 在每个 PPO update batch 中混入 high-quality demonstration 数据。这与 Anthropic 的 Constitutional AI 思路（用 feedback data 训练）有呼应。

---

## 11. 总结

RLPD 是一个**用 minimal 修改让 simple 方法 work** 的典范。它的核心贡献：

1. **Symmetric sampling**：50/50 online/offline，no hyperparameter
2. **LayerNorm in critic**：通过 architectural regularization 隐式 bound Q value，无需 explicit policy constraint
3. **Large ensemble + high UTD**：sample efficiency 的关键

这三个 insights 中，**LayerNorm 的作用是最深刻的**，它揭示了一个 architectural detail 可以替代 explicit regularization，从而保持 algorithm 的 simplicity。

Paper 的精神与 Karpathy 你常说的 "simpler is better" 一致 — 在 RL 这个 complex field，找到 simple 但 effective 的 design choices 是最有价值的工作。

---

## 参考文献

- RLPD paper: https://arxiv.org/abs/2306.07914
- RLPD code: https://github.com/ikostrikov/rlpd
- SAC: https://arxiv.org/abs/1801.01290
- REDQ: https://openreview.net/forum?id=AY8zfZm0tDd
- IQL: https://openreview.net/forum?id=68n2s9ZJWF8
- AWAC: https://arxiv.org/abs/2006.09359
- CQL: https://arxiv.org/abs/2006.04779
- DrQ-v2: https://openreview.net/forum?id=_SJ-_yyes8
- LayerNorm: https://arxiv.org/abs/1607.06450
- D4RL: https://arxiv.org/abs/2004.06139
- Hybrid RL (Song et al.): https://openreview.net/forum?id=yyBis80iUuU
- Implementation Matters: https://openreview.net/forum?id=r1etN1rtPB
- InstructGPT (RLHF): https://arxiv.org/abs/2203.02155
- Decision Transformer: https://arxiv.org/abs/2106.01345
- Dropout Q: https://openreview.net/forum?id=xCVJMsPv3RT
- Off2On: https://openreview.net/forum?id=AlJXhEI6J5W
