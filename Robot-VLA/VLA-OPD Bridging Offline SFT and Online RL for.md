---
source_pdf: VLA-OPD Bridging Offline SFT and Online RL for.pdf
paper_sha256: 14bc9f2b0daa7e5a6bf95734bbe2aef8922c7e840fb6d3db0fc012649aab9056
processed_at: '2026-08-13T02:50:21-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 VLA-OPD

## 先说问题

假设你训练了一个 robot brain（VLA model），它能看图、听指令、输出动作。预训练完之后它啥都会一点，但不够精确，部署到真 robot 上容易翻车。所以需要 post-training 调一调。

现在有两条路，各有毛病：

**SFT 路线**：拿一堆 expert 示范数据，让 model 模仿。好处是每一步都有 label，学得快。坏处是——你在 expert 走过的路上学开车，真自己上路就懵了。Expert 永远走完美路径，student 一旦手抖偏离一点，就进入没见过的状态，越走越歪，最后 crash。这叫 covariate shift / exposure bias。而且因为是在静态数据上 hard fit，很容易把预训练的通用能力给 fit 没了，叫 catastrophic forgetting。

**RL 路线**（GRPO 那套）：让 model 自己在 environment 里 rollout，自己试错。好处是天然 on-policy，见到的就是自己 induce 的状态分布。坏处是 robotics 的 reward 太 sparse 了——任务做完才知道成功没，中间几百步全是黑箱。Credit assignment 极其痛苦，sample efficiency 灾难性低，训练曲线 zig-zag。

所以现在的状况：SFT 快但脆，RL 鲁棒但慢。两个极端。

## VLA-OPD 的 insight

核心 idea 特别简单：**让 student 自己 rollout（on-policy），但每一步都找一个 teacher 问"这步你咋走"，把老师的回答当 dense reward**。

就这么一句话。但里面藏了很多门道。

### 三个 Phase

1. **Student 自己跑**：student policy 在 environment 里 rollout，生成 trajectory。因为它菜，会经常跑偏，进入 OOD state。这些 OOD state 恰恰是 SFT 见不到、但部署时必须面对的。主动把它们暴露出来。

2. **Teacher 标注**：对 student 走过的每一个 state，拿一个 frozen 的 expert teacher（比如别人 RL 训好的 checkpoint）forward 一下，拿到它的 action logits。Teacher 不用真去执行动作，只给 logits 就行。

3. **Student 更新**：用 Reverse-KL 把 student 往 teacher 的分布上 align。

这样你就同时拿到了：on-policy 的 distribution shift 解决 + dense token-level supervision 的 sample efficiency。

## 最关键的技术点：为什么是 Reverse-KL

这是整篇 paper 的灵魂。

想象一个 OOD state，teacher 自己也没见过，输出一个很 flat 的高熵分布（它也不确定）。

现在你要把 student 往 teacher 上 align，有三种选择：

**Forward-KL**（$D_{KL}(\pi_{tea} \| \pi_\theta)$）：这个方向的性质叫 mode-covering——student 必须覆盖 teacher 所有有概率的 action。Teacher 不确定 → student 被迫也不确定 → entropy 爆炸 → policy 犹豫不决 → 执行精度没了。相当于学生把老师的"迷茫"也学来了。

**Hard-CE**（DAgger 的标准做法）：只取 teacher 的 argmax 当 hard label。问题是 teacher 在多模态边界上 argmax 会反复横跳，student rigidly 追这些跳变目标 → entropy 瞬间 collapse → 没了探索的 action diversity → 卡在 local optimum。

**Reverse-KL**（$D_{KL}(\pi_\theta \| \pi_{tea})$）：性质叫 mode-seeking + zero-forcing。只要 student 选的 action 落在 teacher 的有效概率质量里，就不被惩罚。Student 可以自信地 commit 到 teacher 的某个 mode 上。老师长尾的不确定性被自然过滤掉。

直觉说法：Forward-KL 是"老师啥都有一点可能，我全学"；Hard-CE 是"老师你指哪我打哪，不管你犹豫不犹豫"；Reverse-KL 是"老师你给几个选项，我挑一个自己有把握的 mode 钻进去"。

在老师自己都不确定的 OOD state 下，第一种 = 学噪声，第二种 = 丢失多样性，第三种 = 最稳。所以 Reverse-KL 给了一个 bounded entropy——既不爆也不塌，decisive yet exploratory。

这个 insight 其实 LLM distillation 里早就有人发现了（MiniLLM https://arxiv.org/abs/2306.08543，GKD https://aclanthology.org/2023.findings-acl.116/），VLA-OPD 把它搬到 action token 空间。

## 数学形式

Objective：

$$\max_\theta \mathcal{J}(\theta) = \mathbb{E}_{s \sim \pi_\theta}\left[-D_{KL}(\pi_\theta(\cdot|s) \| \pi_{tea}(\cdot|s))\right]$$

Token-level reward：

$$r_t^{OPD}(s_t, a_t) = -\log \frac{\pi_\theta(a_t|s_t)}{\pi_{tea}(a_t|s_t)}$$

- $s_t$：student rollout 到的 state
- $a_t$：student 实际采样的 action
- $\pi_\theta(a_t|s_t)$：student 给这个 action 的概率
- $\pi_{tea}(a_t|s_t)$：teacher 给这个 action 的概率
- $r_t^{OPD}$：student 和 teacher 一致时接近 0，偏离时为负（penalty）

然后就是标准 REINFORCE policy gradient：

$$\nabla_\theta \mathcal{J}(\theta) \approx \frac{1}{G}\sum_{i=1}^G \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_{t,i}|s_{t,i}) \cdot r_t^{OPD}(s_{t,i}, a_{t,i})$$

$G$ 是 group size，论文默认 8，但 ablation 显示 $G=2$ 也能到 80%+，省很多 rollout 算力。

一个细节：reward 里的 $\log \pi_\theta(a_t|s_t)$ 要 stop_gradient，不然会 self-referential 不稳定。梯度只从外层的 policy gradient term 流。

## 实验说了啥

**LIBERO（单臂）**：student 用 1 条 demo 做 SFT 初始化，平均成功率 48.9%。纯 distillation 拉到 87.4%（已经超过 OpenVLA 用 50 条 demo 训出来的 76.5%）。Distill + GRPO 组合到 93.4%，几乎完全 recover teacher 的 93.9%。

**RoboTwin2.0（双臂）**：1000-traj SFT 初始化只有 45.2%。Distillation 拉到 71.1%，接近 teacher 的 74.0%。π₀ 才 50.5%，RDT-1B 才 32.0%。

**效率**：LIBERO-Long 上 50 步到 80%，baseline GRPO 要 150 步，3× speedup。LIBERO-Object 上 10 步就到 90%+。

**Catastrophic forgetting**：seen-unseen trade-off 图上，offline SFT 的 unseen task performance 直接崩到 0，VLA-OPD 和 RL 都保住了。因为 on-policy update 锚定在 student 自己的 behavioral manifold 上，是 gentle alignment 而不是 aggressive overwrite。

**Ablation 验证 Reverse-KL**：Forward-KL 早期 success rate 掉 50%（performance valley，entropy 爆炸），Hard-CE 卡在最低 plateau（entropy collapse），Reverse-KL steady 上升（bounded entropy）。

## 一句话总结

**让 student 自己 rollout 暴露 OOD 状态，用 teacher 给 dense token-level 指导，用 Reverse-KL 避免学到老师的不确定性——同时拿到 SFT 的快和 RL 的稳，还顺手解决了 catastrophic forgetting。**

这给 foundation model 迭代一条 scalable 路径：不用每次从头 RL 训，拿个 teacher checkpoint 蒸馏一下就行。Teacher 可以是开源的、API 调的、或者单任务训的小 policy。

Project page: https://irpn-lab.github.io/VLA-OPD/

## 顺便联想到的

- 这个范式跟 LLM 里的 GKD / MiniLLM 高度同构，都是 on-policy + Reverse-KL distillation。LLM 那边的成熟经验可以继续往 VLA 迁。
- Teacher 的质量是 upper bound。如果 teacher 在 OOD 上也很烂，distillation 天花板就低。未来可能要 ensemble teacher 或 iterative self-distillation。
- 对 continuous action space（diffusion policy / flow matching 那套，比如 π₀、RDT）的迁移是 open question。Reverse-KL 在 continuous space 的 mode-seeking 性质可能要重新推导。
- Figure 3 的 seen-unseen trade-off 暗示这个框架天然适合 continual learning，不用 experience replay buffer，这对 lifelong robot learning 很有吸引力。

参考：
- SimpleVLA-RL (teacher): https://arxiv.org/abs/2509.09674
- OpenVLA-OFT (student init): https://arxiv.org/abs/2502.19645
- RoboTwin2.0: https://arxiv.org/abs/2506.18088
- MiniLLM (Reverse-KL distillation insight): https://arxiv.org/abs/2306.08543
- GKD: https://aclanthology.org/2023.findings-acl.116/
- π₀: https://arxiv.org/abs/2410.24164
- DAgger: https://arxiv.org/abs/1011.0686
- RL mitigates forgetting: https://arxiv.org/abs/2509.04259

---

# VLA-OPD: 通过 On-Policy Distillation 桥接 Offline SFT 与 Online RL

## 一、核心直觉与动机

这篇论文来自 HKUST(GZ)，解决的是一个 VLA (Vision-Language-Action) post-training 中的根本性矛盾。我来帮你 build intuition。

当前 VLA post-training 有两条主流路径：

**Offline SFT** (Behavior Cloning)：在静态数据集 $\mathcal{D}_{demo}$ 上做 maximum likelihood。优点是 dense supervision + 收敛快，缺点是 **off-policy** —— 训练时见到的是 expert states，部署时却是 student 自己 induce 的 states。一旦 student 产生微小执行误差，这些误差会 compound，把 agent 推到 SFT 从未见过的 OOD states，policy 就 crash 了。同时，在静态 disjoint 数据集上做 aggressive 参数更新，会 catastrophic forgetting 掉预训练的 generalist 能力。

**Online RL** (以 GRPO 为代表)：让 policy 自己 rollout，on-policy 暴露 distribution shift 问题。问题在于 robotics 的 reward 是 sparse outcome reward $R(\tau) \in \{0,1\}$，只有任务结束才知道成功与否。这导致 credit assignment 极其困难，sample efficiency 灾难性地低。

VLA-OPD 的 insight：用 **on-policy rollout** 解决 distribution shift，用 **teacher 提供 token-level dense supervision** 解决 reward sparsity，用 **Reverse-KL** 作为 alignment objective 解决 optimization stability。

参考 GRPO 原文: https://arxiv.org/abs/2509.09674 (SimpleVLA-RL)

---

## 二、Framework 三阶段架构

参考 Figure 1 与 Algorithm 1，整个 pipeline 是一个闭环迭代过程：

### Phase 1: On-Policy Sampling (Student Exploration)

Student policy $\pi_\theta$ 在 environment 中 rollout，生成 trajectories：

$$\mathcal{D}_k = \{\tau \mid \tau = (s_0, a_0, s_1, a_1, \ldots, s_T)\}$$

其中 $a_t \sim \pi_{\theta_k}(\cdot|s_t)$，$s_{t+1} \sim \mathcal{P}(\cdot|s_t, a_t)$。

关键点：states $s_t$ 来自 student 的 induced distribution $d^{\pi_{\theta_k}}$，而非 expert distribution。这意味着当 student 因为 1-traj SFT 初始化而能力薄弱时，它会频繁偏离 expert path，进入 OOD "failure states" $s_{err}$。这些 states 恰恰是 SFT 无法覆盖、但部署时必须面对的。

**Intuition**: 这一步把"未知 OOD 区域"主动暴露出来，转变成"已知训练数据"，将 alignment 从 passive imitation 转变为 active correction。

### Phase 2: Dense Teacher Labeling

对 student 走过的每个 state $s_t$，frozen teacher $\pi_{tea}$ 提供 action logits：

$$q_t(a) = \pi_{tea}(a|s_t)$$

这里 teacher 可以是 RL-trained 的 expert (论文用 SimpleVLA-RL)，也可以是开源 checkpoint / API。Teacher 不需要执行 action，只需要 forward pass 给出 logits。

**Intuition**: 这相当于把 delayed RL 问题转化为 immediate supervised signal，每一步都有梯度方向。同时 teacher 在 OOD states 上给出的 recovery behavior，是一种 structural knowledge transfer。

### Phase 3: Reverse-KL Optimization

Student 通过最小化 Reverse-KL divergence 来 align teacher：

$$\max_\theta \mathcal{J}(\theta) = \mathbb{E}_{s \sim \pi_\theta}\left[-D_{KL}(\pi_\theta(\cdot|s) \| \pi_{tea}(\cdot|s))\right]$$

Token-level intrinsic reward：

$$r_t^{OPD}(s_t, a_t) = -\left(\log \pi_\theta(a_t|s_t) - \log \pi_{tea}(a_t|s_t)\right) = -\log \frac{\pi_\theta(a_t|s_t)}{\pi_{tea}(a_t|s_t)}$$

变量含义：
- $\pi_\theta(a_t|s_t)$: student 在 state $s_t$ 下采取 action $a_t$ 的 log-prob
- $\pi_{tea}(a_t|s_t)$: teacher 在同一 state 下给同一 action 的 log-prob
- $r_t^{OPD}$: 当 student 与 teacher 一致时接近 0，偏离时为负值 (penalty)

计算 policy gradient 时，对 student 的 $\log \pi_\theta(a_t|s_t)$ 项做 stop_gradient，只让 teacher 项作为 reward signal 传梯度。

Group-based gradient estimation：

$$\nabla_\theta \mathcal{J}(\theta) \approx \frac{1}{G}\sum_{i=1}^G \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_{t,i}|s_{t,i}) \cdot r_t^{OPD}(s_{t,i}, a_{t,i})$$

- $G$: group size (论文默认 8)
- 外层对 trajectory 求和，内层对 timestep 求和
- 这是标准 REINFORCE 形式，reward 用 Reverse-KL 替代 sparse outcome reward

---

## 三、Reverse-KL vs Forward-KL vs Hard-CE 的深度分析

这是论文最核心的理论贡献。我详细讲讲为什么 divergence 方向如此关键。

考虑一个 OOD state $s_{err}$，teacher 因为也没见过这个 state，给出一个 high-entropy 的 flat distribution (epistemic uncertainty)。

### Forward-KL: $D_{KL}(\pi_{tea} \| \pi_\theta)$

梯度对 teacher samples 求期望：$\mathbb{E}_{a \sim \pi_{tea}}[\log \pi_\theta(a|s)]$。

性质是 **mode-covering** (mass-covering)：student 必须给 teacher 所有有概率的 action 都分配概率，才能 minimize 这个 KL。当 teacher 是 flat distribution 时，student 被迫也变成 flat，**entropy explosion**。结果就是 policy 变得犹豫不决，失去执行精度。

参考 Hinton "Dark Knowledge": https://arxiv.org/abs/2306.08543 (MiniLLM，最早在 LLM distillation 中用 Reverse-KL)

### Hard-CE (Argmax Matching, 标准 DAgger)

只取 teacher 的 argmax action 作为 hard label，丢弃 soft probabilities ("dark knowledge")。

问题：当 teacher 在 multi-modal decision boundary 上 argmax oscillate 时，student 被迫 rigidly track 这些跳变目标，**premature entropy collapse**。Policy 失去探索所需的 action diversity，陷入 local optimum。

参考 DAgger: https://arxiv.org/abs/1011.0686

### Reverse-KL: $D_{KL}(\pi_\theta \| \pi_{tea})$

梯度对 student samples 求期望：$\mathbb{E}_{a \sim \pi_\theta}[\log \pi_\theta(a|s) - \log \pi_{tea}(a|s)]$。

性质是 **mode-seeking** + **zero-forcing**：
- 只要 student 选的 action 落在 teacher 的可接受概率质量内，就不被惩罚
- Student 可以自信地 commit 到 teacher 的 primary mode
- Teacher 长尾的 uncertain 部分被 filtered out

这给了一个 **bounded entropy**：既不 explosion (像 Forward-KL)，也不 collapse (像 Hard-CE)。Student 保持 decisive 又有 sufficient stochasticity 做探索。

**Intuition**: Forward-KL 是"覆盖老师所有可能性"，Hard-CE 是"只学老师的 argmax"，Reverse-KL 是"在老师的分布里找一个 mode 自信地站进去"。在 OOD states 下，老师自己都不确定，覆盖所有不确定性 = 学到噪声；只学 argmax = 丢失多样性；找一个 mode 站进去 = 最稳健的策略。

---

## 四、实验数据深度解读

### 4.1 LIBERO 上的 Efficiency (Figure 2)

- **LIBERO-Object**: VLA-OPD Distill 在 10 steps 内达到 90%+ success rate，baseline GRPO 是 gradual climb
- **LIBERO-Long**: 50 steps 达到 ~80%，baseline 需要 150 steps，**3× speedup**
- Distill + GRPO 组合进一步突破 ceiling：Object 95%+，Long 90%+
- 训练曲线平滑，GRPO baseline 有 zig-zag fluctuation

### 4.2 主表 (Table 2): LIBERO 四个 suite

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| Teacher (SimpleVLA-RL) | 94.2 | 96.1 | 94.6 | 90.7 | 93.9 |
| OpenVLA-OFT (1-traj SFT init) | 63.6 | 54.9 | 59.6 | 17.3 | 48.9 |
| VLA-OPD (Distill) | 84.3 | 93.8 | 92.5 | 78.9 | 87.4 |
| VLA-OPD (Distill + GRPO) | 93.4 | 95.3 | 94.5 | 90.2 | 93.4 |

关键观察：1-traj SFT init 只有 48.9% avg，纯 distillation 拉到 87.4% (已经超过 OpenVLA 50-traj full-dataset 的 76.5%)，加 GRPO 后到 93.4%，几乎完全 recover teacher 的 93.9%。

这意味着：**只需要 1 条 demo + 一个 teacher checkpoint，就能逼近 RL-from-scratch 训练的 expert 性能**。

### 4.3 RoboTwin2.0 双臂 (Table 3)

| Method | Pick dual bottles | Place Empty Cup | Handover Block | Stack Bowls Two | Avg |
|--------|-------------------|-----------------|----------------|-----------------|-----|
| Teacher | 68.3 | 94.2 | 57.8 | 75.8 | 74.0 |
| π₀ | 50.0 | 60.0 | 39.0 | 53.0 | 50.5 |
| RDT-1B | 18.0 | 42.0 | 26.0 | 42.0 | 32.0 |
| OpenVLA-OFT (1000-traj SFT) | 29.7 | 77.3 | 33.1 | 40.6 | 45.2 |
| VLA-OPD (Distill) | 66.4 | 90.6 | 52.3 | 75.0 | 71.1 |

双臂任务 morphological complexity 高，1000-traj SFT 也只有 45.2%。Distillation 拉到 71.1%，接近 teacher 74.0%。说明框架对 morphology 有 generalization。

参考 RoboTwin2.0: https://arxiv.org/abs/2506.18088

### 4.4 Catastrophic Forgetting 分析 (Figure 3)

Seen-Unseen trade-off 图：
- **Offline SFT**: seen task success 上升时，unseen task performance 崩塌 (Object 接近 0，Spatial 大幅下降)
- **Online RL + VLA-OPD**: 都避免了 collapse，VLA-OPD 在多个 axis 上 match 或超过 RL

**Intuition**: On-policy data 让 gradient update 锚定在 student 当前 behavioral manifold 上，这是一种 "gentle alignment"。Offline SFT 强行 fit 一个 disjoint target distribution，需要 aggressive parameter shift，覆盖掉预训练知识。

参考 RL mitigates forgetting: https://arxiv.org/abs/2507.05386 ; https://arxiv.org/abs/2509.04259

### 4.5 Ablation: Alignment Objective (Figure 4)

在 RoboTwin2.0 Beat Block Hammer 任务上：
- **Reverse-KL**: steady improvement，stable bounded entropy
- **Forward-KL**: 早期 success rate 掉 50%+ (performance valley)，entropy explosion
- **Hard-CE**: plateau 在最低 success rate，premature entropy collapse

这直接验证了 Section 3.4 的理论分析。

### 4.6 Ablation: Group Size G (Figure 5)

$G \in \{2, 4, 8\}$:
- $G=8$: 最平滑，~89% final
- $G=2$: 也能到 80%+，competitive

大 G 给更 robust 的 Monte Carlo approximation $\mathbb{E}_{\tau \sim \pi_\theta}$，但小 G 仍有 sufficient signal-to-noise ratio。实际意义：可以用小 G 大幅降低 rollout + teacher inference 的 wall-clock cost。

---

## 五、Methodology 细节补充

### 5.1 为什么用 Reverse-KL 等价于 dense RL reward

公式 (5) 展开后：

$$\mathcal{J}(\theta) = \mathbb{E}_{s \sim \pi_\theta}\left[\sum_a \pi_\theta(a|s) \log \frac{\pi_{tea}(a|s)}{\pi_\theta(a|s)}\right]$$

这可以 reframe 成 standard policy gradient：reward function 就是 $r(s,a) = \log \pi_{tea}(a|s) - \log \pi_\theta(a|s)$。每个 token 都有一个 reward，dense supervision 就这样实现了。

### 5.2 与 GRPO 的关键区别

GRPO 用 sparse outcome reward $R(\tau) \in \{0,1\}$，通过 group-based normalization 算 advantage：

$$\hat{A}_i = \frac{R(\tau_i) - \text{mean}(R(\tau_1...G))}{\text{std}(R(\tau_1...G))}$$

VLA-OPD 直接用 Reverse-KL reward 作为 advantage，不需要 normalization。因为 reward 本身已经 dense 且有方向性 (正负都代表含义)，不需要 group relative 来提供 signal。

### 5.3 Stop-gradient 的必要性

公式 (6) 中 $\log \pi_\theta(a_t|s_t)$ 项如果传梯度，会变成一个 self-referential 的优化目标，导致不稳定。Stop_gradient 让这一项只作为 reward 的组成部分，梯度只通过外层的 $\nabla_\theta \log \pi_\theta(a_t|s_t)$ (policy gradient term) 流动。

---

## 六、整体 Intuition 总结

VLA-OPD 的精妙之处在于它把三个问题用统一框架解决：

1. **Distribution shift** ← On-policy rollout 让 student 见到自己的 induced states
2. **Reward sparsity** ← Teacher 提供 token-level dense reward (Reverse-KL)
3. **Optimization instability** ← Reverse-KL 的 mode-seeking property 给 bounded entropy

而且 on-policy 的特性天然 mitigates catastrophic forgetting，因为 update 锚定在 student 自己的 manifold 上，是 gentle alignment 而非 aggressive overwrite。

这给了 foundation model 持续迭代的一条 scalable pathway：不需要每次从头 RL 训练，只需要一个 teacher (开源 checkpoint / API / 单任务 policy) 就能 distill 出 robust behavior 到新的 student backbone。

---

## 七、可能的延伸思考

- **Teacher 质量依赖**: 论文承认 reliance on teacher 是 limitation。如果 teacher 本身在 OOD states 上很差，distillation 的 upper bound 就受限。未来可能需要 ensemble teacher 或 self-distillation。
- **与 RLHF/DPO 的联系**: Reverse-KL 在 LLM alignment 中也有广泛应用 (参考 MiniLLM https://arxiv.org/abs/2306.08543 和 GKD https://aclanthology.org/2023.findings-acl.116/)。VLA-OPD 本质上是把这些 insight 迁移到 action token 空间。
- **与 Diffusion Policy 的关系**: π₀ 用 flow matching，RDT 用 diffusion。VLA-OPD 目前针对 discretized action token (FAST tokenizer)。对 continuous action space 的 distillation 是 open question。
- **Continual Learning 场景**: Figure 3 的 seen-unseen trade-off 暗示这个框架天然适合 lifelong learning，不需要 experience replay buffer。

Project page: https://irpn-lab.github.io/VLA-OPD/

参考 OpenVLA-OFT: https://arxiv.org/abs/2502.19645
参考 π₀: https://arxiv.org/abs/2410.24164
参考 Interactive Post-training (Tan et al.): https://arxiv.org/abs/2505.17016

希望这个讲解帮你 build 起对 VLA post-training landscape 的 intuition。核心 takeaway：**on-policy + dense teacher supervision + Reverse-KL mode-seeking = 同时拿到 SFT 的效率和 RL 的 robustness**。
