---
source_pdf: SRPO Self-Referential Policy Optimization for Vision-Language-Action.pdf
paper_sha256: 784f0ec1b006283949103e391130b4fe7663cd2c1e1a63352c9766e2fc958945
processed_at: '2026-08-12T10:19:57-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SRPO 用人话讲

Karpathy，让我换一种讲法，用更直白的方式帮你建立 gut feeling。

---

## 这篇 paper 到底在解决什么问题？

想象你在训练一个 robot arm 学抓东西。传统的 RL 方式是：robot 试一把，成功了你给它 1 分，失败了给 0 分。

问题在哪？假设 robot 试了 8 次（一个 batch），只有 1 次成功，7 次失败。传统 GRPO 的做法是：那 1 次成功的给它正梯度，7 次失败的**全部一样对待**——统一给负梯度。

但这 7 次失败里，有的可能是"抓起来了但最后没放对位置"，离成功就差一步；有的可能是"一开始就抓错了东西"。传统方法把这俩**完全同等对待**，这太浪费了。就好像考试批卷子，一道大题没做出来，你不给学生 partial credit，那学生怎么知道自己是哪一步卡住的？

VLA-RL 领域的人都知道这个问题，现有解法是要么 hand-craft 中间步骤的 reward（比如"先抓起来给 0.3 分，放到正确位置再给 0.7 分"），要么用 expert demonstration 来定义什么叫"做对了一半"。但这俩都**不 scale**，每个新 task 都要人工设计。

SRPO 的 insight 特别简单也特别 elegant：**batch 里不是有成功的 trajectory 吗？拿成功的那条当参照物，衡量失败的 trajectory 离它有多近，近的就给高分，远的给低分**。就这样。

关键在于"多近"怎么衡量——你不能直接比较 pixel，因为同样是"抓杯子放到微波炉"，robot 可能从左边抓也可能从右边抓，pixel 完全不一样但语义上是一回事。所以需要一个能理解"物理语义"的 embedding space，这就是 V-JEPA 2 的 latent space 派上用场的地方。

---

## V-JEPA 2 为什么 work？一个 intuition

V-JEPA 2 是 Meta 在大规模 robotics video 上自监督训练的 world model。它的 latent space 有个特性：**在物理上类似的 trajectory，embedding 也接近**。

打个比方，你给 V-JEPA 2 看两段视频：一段是 robot 从左边抓起红杯子放到微波炉，另一段是 robot 从右边抓起蓝杯子放到微波炉。Pixel 层面这俩完全不同，但 V-JEPA 2 的 embedding 会把这俩放到很近的位置，因为它学到了"抓起东西→移动→放进容器"这个物理动作的 abstract structure。

这就是为什么用 V-JEPA 2 的 latent distance 做 reward 比用 pixel distance 或者 ImageBind 都好。Pixel distance 对"相机抖一下"这种无关变化敏感；ImageBind 是通用视觉模型，没在 robotics 数据上训过，它不懂"gripper pause"不代表 progress 倒退；V-JEPA 2 天生理解 physical interaction 的 semantics。

论文里 Table 3 的数据特别 striking：SRPO 的 Standardized Mean Difference（衡量 success 和 failure 在 reward space 分得多开的指标）是 **188.8**，ImageBind 只有 18.1，pixel-level 只有 2.1。差了一个数量级。这说明 V-JEPA 2 的 latent space 确实把"成功的 trajectory"和"失败的 trajectory"在 embedding 空间里推得很远，这正是 reward function 能 work 的基础。

V-JEPA 2 论文：https://arxiv.org/abs/2506.09985

---

## 公式用人话拆解

### Reward 怎么算

假设一个 batch 有 8 条 trajectory，其中 2 条成功，6 条失败。

**Step 1**：用 V-JEPA 2 把每条 trajectory 编码成一个 vector。

$$h_i = \mathcal{W}(o_{0:T}^{(i)})$$

$h_i$ 就是第 $i$ 条 trajectory 的 "指纹"——一个固定维度的 vector。$o_{0:T}^{(i)}$ 是这条 trajectory 的所有 observation frames。

**Step 2**：把成功的 trajectory 做 clustering。

$$C = \text{DBSCAN}(S)$$

$S$ 是所有成功 trajectory 的 embedding 集合。DBSCAN 会把语义相近的成功 trajectory 聚成几个 cluster，每个 cluster 有一个 centroid $h_j \in C$。

为什么要 cluster 而不是直接用单条成功 trajectory？因为同一个 task 可能有多种成功策略——比如"先放 A 再放 B"和"先放 B 再放 A"都是对的。如果你只拿一条成功 trajectory 当参照，那走另一条成功路径的失败 trajectory 就会被错误地判为"离成功很远"。Cluster 让你自动发现"有哪几种成功模式"，失败 trajectory 和它最接近的那种模式比。

**Step 3**：对每条失败 trajectory，算它离最近的 cluster center 有多远。

$$d_i = \min(\{\|h_i - h_j\|^2; h_j \in C\})$$

$d_i$ 就是"这条失败 trajectory 离最近的成功模式有多远"。

**Step 4**：把距离转成 reward。

$$g_i = \begin{cases} 1.0 & \text{成功} \\ \phi\left(\frac{d_i - \bar{d}}{\sigma_d}\right) & \text{失败} \end{cases}$$

$\bar{d}$ 是这个 batch 里所有失败 trajectory 距离的均值，$\sigma_d$ 是标准差。$\frac{d_i - \bar{d}}{\sigma_d}$ 是 z-score——如果你离成功比 batch 平均水平近，z-score 是正的；比平均远，z-score 是负的。$\phi$ 是 sigmoid，把 z-score 压到 $(0, 1)$。

为什么用 z-score 而不是直接用 raw distance？因为不同 batch 里，成功 trajectory 的 distribution 不一样，绝对距离的 scale 也不一样。Z-score 做的是 **batch-relative normalization**——不管这个 batch 的成功 trajectory 分布得散还是聚，你都只看"这条失败 trajectory 相对于这个 batch 的其他失败 trajectory，是不是更接近成功"。

### Advantage 怎么用

拿到 $g_i$ 之后，代入 GRPO 的标准流程：

$$\hat{A}_i = \frac{g_i - \mu_g}{\sigma_g}$$

$\mu_g$ 和 $\sigma_g$ 是整个 batch 的 $g_i$ 的均值和标准差。这步是在做 group normalization——把 reward 变成"相对于这个 batch 平均水平的 advantage"。

如果 $g_i > \mu_g$，说明这条 trajectory 比 batch 平均好，advantage 是正的，policy 会往这个方向靠。反之 advantage 是负的，policy 会远离。

和传统 GRPO 的区别：传统 GRPO 里成功的 trajectory $g_i = 1$，失败的 $g_i = 0$，advantage 只有"好"和"坏"两种。SRPO 里失败的 trajectory 也有 continuous 的 $g_i$，所以 advantage 是 graded 的——"差点成功"的失败拿正 advantage，"差很远"的失败拿负 advantage。

---

## 为什么 Self-Referential 比用 Expert Reference 好

论文 Appendix C.1 做了个 ablation：把 self-referential 换成 50 条固定的 expert trajectory 当参照。

结果：初期训练速度差不多，但后期 plateau，最终性能更低。

Intuition 是这样的：假设你是学打篮球的学生。如果你一直拿 Michael Jordan 的动作当参照，你一开始能学到基本姿势，但当你已经能稳定投篮之后，你和 Jordan 动作的差异就不再是"你哪里做错了"的信号了——因为你的身体条件、你的风格本来就和 Jordan 不一样。你需要的参照是"和你身体条件类似、比你稍好一点的人"，这样你才知道下一步该改进什么。

Self-referential 就是这个逻辑。Batch 内的成功 trajectory 是"当前 policy 能达到的最好水平"，拿这个当参照，advantage signal 始终是"相对于你当前水平的 incremental improvement"。Expert trajectory 是固定的，policy 进化到一定程度后，expert trajectory 的 reference 就太 coarse 了——所有 rollout 都离 expert 很远，你区分不出"哪个 rollout 稍微好一点"。

这就是为什么 self-referential 形成了一个 **dynamic curriculum**——参照物随 policy 成长而成长，始终提供 informative 的 gradient signal。

---

## $\alpha = 0.8$ 这个 magic number

论文 Appendix D 测试了不同的 progress reward weight $\alpha$：

- $\alpha = 0$（纯 binary reward）：最差
- $\alpha = 0.3, 0.5$：逐渐变好
- $\alpha = 1.0$（progress reward 和 success reward 同等权重）：好但不是最优
- $\alpha = 0.8$：最优

Intuition：$\alpha = 0$ 就是纯 GRPO，sparse reward 问题没解决。$\alpha = 1.0$ 的问题在于，progress reward 再好也是 proxy——它衡量的是"看起来像不像成功"，不是"真的成功了"。如果你给 progress reward 太大权重，policy 会过度优化"看起来像成功"的 trajectory，哪怕它最终没成功。比如 robot 可能学会"摆出快要成功的姿势"来骗 high progress reward，而不是真的完成任务。

$\alpha = 0.8$ 是 sweet spot：progress reward 提供 dense gradient signal 帮助学习，但 outcome reward（$g_i = 1.0$ for success）始终保持最高优先级，确保 policy 不会忘掉"最终目标是完成任务"。

这个 trade-off 在 RL reward shaping 里很经典——Sutton 的 "bitter lesson"（http://www.incompleteideas.net/IncIdeas/BitterLesson.html）其实就在讲这个：hand-crafted 的 dense reward 短期 help，长期可能 misleading。SRPO 用 0.8 这个 weight 保留了 outcome reward 的主导地位，progress reward 只是辅助。

---

## 最 Fascinating 的发现：Trajectory Diversity

Section 5.3 的 Figure 6 和 7 让我特别兴奋。

他们比较了 full-shot SFT（用 50 条 expert demonstration 训练）和 one-shot SFT + SRPO（只用 1 条 demonstration 起步，然后 RL）的 action space 分布。

结果：**SRPO 的 action distribution 覆盖了 SFT 覆盖不到的区域**。即使只给 1 条 demonstration，SRPO fine-tune 后的 policy 能发现全新的 grasping position 和 spatial path。

这解释了为什么 SRPO 在 LIBERO-Plus（perturbed environment）上 generalization 那么强。SFT 的 policy 本质上是"复现 demonstration"，demonstration 没覆盖的 action 它不敢做。SRPO 的 policy 通过 RL exploration 发现了 demonstration 之外的 strategy，这些 strategy 在 perturbed environment 下更 robust，因为 policy 学到的是 "affordance understanding" 而不是 "specific trajectory memorization"。

这也是为什么 one-shot SFT + SRPO（59.6%）在 LIBERO-Plus zero-shot 上能接近 full-shot SFT（69.6%）——RL exploration 带来的 trajectory diversity 部分弥补了 demonstration 数量的劣势。

---

## Real-World Transfer 的 intuition

论文在 real robot 上用 offline RL（AWR, Advantage-Weighted Regression）而不是 online RL，因为 real robot 上 online exploration 太危险且 reset 慢。

AWR 的核心 idea：你有一堆 trajectory（有成功有失败），算每条 trajectory 的 advantage，然后用 advantage 加权的 maximum likelihood 来更新 policy。advantage 高的 trajectory，policy 往那个方向靠；advantage 低的，远离。

SRPO 的 progress reward 直接 plug 进 AWR 的 advantage 计算：

$$A_{i,t} = \frac{D_{i,t} - \mu}{\sigma}$$

$D_{i,t} = R_{i,t} - R_{i,t-1}$ 是 step $t$ 的 incremental progress。这里用 step-level 而不是 trajectory-level，因为 offline RL 里你没有 online rollout 的 exploration 信号，需要更细粒度的 credit assignment。

Real-world 实验（Figure 8）显示 $\pi_0$ 和 $\pi_0$-FAST 两个 backbone 都有 +60%~+86% 的提升。更重要的是 Table 4：V-JEPA 2 的 progress reward 在 real-world 数据上的 SC=0.989, Mono=0.978, SMD=169.5，和 simulation 上的指标（SC=0.998, SMD=188.8）几乎一样好。

这说明 V-JEPA 2 的 latent space 从 simulation 到 real-world **没有 domain gap**——它在大规模 video 上 pretrain，自然 capture 了 real-world 的 physical pattern，不需要在 real robot 数据上 fine-tune。这是 latent world representation 相比 pixel-level world model 的根本优势。

---

## 一些我想到的 concern

1. **Cold start 问题**：如果一个 batch 全失败，没有 success reference，progress reward 算不出来。论文的实验是从 48.9% success rate 的 one-shot SFT 起步的，这个 success rate 足够保证每个 batch 大概率有成功 trajectory。如果从 random initialization 起步，这个方法能不能 bootstrap？论文没讨论。

2. **V-JEPA 2 的 coverage**：V-JEPA 2 在什么数据上 pretrain 决定了它的 latent space 能理解什么 physical interaction。论文的 Fold Towel task 结果支持这个担忧——deformable object manipulation 在 robotics video 里相对少见，V-JEPA 2 可能没充分学过。如果 task 涉及 V-JEPA 2 没见过的 physics，progress reward 会失准。

3. **DBSCAN 参数**：$\epsilon$ 和 min_samples 的选择对 cluster 数量和质量影响很大，论文没讨论 sensitivity。如果 batch 里成功 trajectory 很少（比如只有 1-2 条），DBSCAN 可能聚不出有意义的 cluster，这时候退回到 "nearest single trajectory" 模式，论文 Appendix C.2 证明这个模式后期性能会下降。

4. **Long-horizon 的 credit assignment**：SRPO 是 trajectory-level advantage，对 100 步的 long-horizon task，"第 50 步的关键错误"和"第 50 步的随机抖动"在 trajectory-level reward 里区分不开。可能需要 step-level progress reward，但那样会引入更细粒度的 noise。

---

## 跟其他工作的思想连接

1. **HER (Hindsight Experience Replay)**：https://arxiv.org/abs/1707.01495 — HER 的 idea 是"把失败 trajectory 当成追求另一个 goal 的成功 trajectory"。SRPO 的 self-referential 有相似的哲学——**利用 batch 内已有的信息，而不是引入 external supervision**。但 HER 改的是 goal，SRPO 改的是 reward signal。

2. **Contrastive Learning**：SRPO 本质上是一种 implicit contrastive learning——成功 trajectory 是 positive，失败 trajectory 是 negative，latent distance 是 similarity metric。但它把 contrastive signal 直接 inject 到 policy gradient 里，而不是学一个 representation。

3. **LeCun 的 JEPA philosophy**：https://arxiv.org/abs/2301.08243 — LeCun 一直主张 world model 应该在 latent space 做 prediction 而不是 pixel space。这篇 paper 是 JEPA 思想在 robotics RL reward shaping 上的应用案例，验证了 "latent representation 更 transferable" 这个 thesis。

4. **Curriculum Learning**：Self-referential 形成的 dynamic curriculum，和 fixed curriculum 的对比在 Appendix C.1 里很清楚。这和 self-paced learning（https://arxiv.org/abs/1010.0801）有思想上的呼应——都是让模型根据自己的 current state 选择学习信号。

---

## 一句话的 intuition

SRPO 说的是：**别浪费失败，batch 里总有成功的，拿成功的当锚点衡量失败离它多远，远的当反面教材近的当 partial credit，用 V-JEPA 2 这个懂物理的 encoder 来量距离，policy 自然学得快**。整个 framework 没有 external supervision，没有 hand-crafted reward，全靠 batch 内 self-reference 和 pretrain 的 world model，这就是 autonomous learning 的范式。

如果你想 drill into 任何一个点——比如 V-JEPA 2 的 architecture 细节、DBSCAN 在这个 context 里的参数选择、或者 AWR 和 GRPO 在 offline vs online 设定下的数学差异——我可以继续展开。

---

# SRPO: Self-Referential Policy Optimization for VLA Models

Karpathy你好, 让我深入拆解这篇paper, 重点是帮你建立intuition - 为什么这个方法work, 它在解决什么本质问题, 以及它和现有VLA-RL工作的差异在哪里。

---

## 1. Problem Setup: 为什么需要这篇paper

当前VLA-RL (比如用GRPO fine-tune VLA) 的核心痛点是 **reward sparsity**。在LIBERO这种manipulation benchmark里, reward只有0或1 - 任务完成=1, 失败=0。这导致两个问题:

1. **失败trajectory完全浪费**: 一条rollout只要最后没成功, 整条trajectory的gradient信号都是一样的 (uniform negative), 哪怕它实际上离成功只差最后一步
2. **训练效率低**: 每次environment rollout很贵 (相比LLM的一次forward pass), 浪费任何trajectory都是巨大的成本

现有的两条路径各有问题:
- **GRPO (binary reward)**: https://arxiv.org/abs/2402.03300 - 简单但sparse, 失败trajectory信息丢失
- **PRM (process reward model)**: 需要expert demonstration定义intermediate milestone, 或者hand-crafted task decomposition, 这违反了autonomous learning的目标, 无法scale

SRPO的核心insight: **batch内自己生成的成功trajectory, 就可以作为失败trajectory的progress reference**。这把问题从 "how to get expert labels" 转化为 "how to extract progress signal from our own successes"。

---

## 2. Method核心: Self-Referential Progress Reward

### 2.1 整体pipeline

给定一个training batch, 包含M条rollout trajectories $\{o_{0:T}^{(i)}, a_{0:T}^{(i)}\}_{i=1}^M$, 其中一些成功 (R=1), 一些失败 (R=0)。

Pipeline:
1. 用world model encoder $\mathcal{W}$ (V-JEPA 2) 把每条trajectory的observation sequence编码成latent vector
2. 对成功trajectory的latent vectors做DBSCAN clustering, 得到cluster centers $C$
3. 对失败trajectory, 计算它到最近cluster center的L2 distance $d_i$
4. 用 $d_i$ 通过z-score normalization + sigmoid, 映射成 $(0, 1)$ 的progress reward $g_i$
5. 用 $g_i$ 替代binary reward做GRPO-style advantage estimation

### 2.2 数学公式详解

**Trajectory encoding (Eq. 2)**:
$$h_i = \mathcal{W}(o_{0:T}^{(i)})$$

- $h_i \in \mathbb{R}^d$: 第 $i$ 条trajectory的latent representation
- $\mathcal{W}$: world model encoder, 这里用V-JEPA 2 (https://arxiv.org/abs/2506.09985)
- $o_{0:T}^{(i)}$: 第 $i$ 条trajectory的observation sequence (frames 0到T)

Intuition: V-JEPA 2是在大规模robotics video上自监督pretrain的, 它的latent space自然捕获物理progress pattern, 比如物体被抓取、移动、放置这些动作的semantic structure。

**DBSCAN Clustering (Eq. 3)**:
$$C = \text{DBSCAN}(S)$$

- $S = \{o_{0:T}^{(i)}; R(z_{0:T}^{(i)}, l) = 1, \forall i\}$: 所有成功trajectory的observation set
- $C$: cluster centers的集合

为什么用cluster而不是单条reference? 两个原因:
- 同一task可能有多种策略 (先放A再放B, 或先放B再放A), 失败trajectory应该和它最接近的那一类成功策略比
- Cluster centroid比单条trajectory更robust, 能去除单条成功trajectory里的noise (比如抓取瞬间gripper小幅后退)

**Distance calculation (Eq. 4)**:
$$d_i = \min(\{\|h_i - h_j\|^2; h_j \in C\})$$

- $d_i$: 第 $i$ 条失败trajectory到最近cluster center的L2 distance squared
- $\|h_i - h_j\|^2$: L2 distance squared in latent space
- $\min$: 取到最近cluster center的距离

Intuition: 失败trajectory离成功cluster center越近, 说明它在behavior space上越接近某种成功模式, 应该拿更高的reward。

**Reward computation (Eq. 5)**:
$$g_i = \begin{cases} 1.0 & \text{for success trajectory} \\ \phi\left(\frac{d_i - \bar{d}}{\sigma_d}\right) & \text{for failed trajectory} \end{cases}$$

- $g_i \in (0, 1)$: 第 $i$ 条trajectory的final reward
- $\phi(\cdot)$: activation function (sigmoid), 把z-score映射到 $(0, 1)$
- $d_i$: 失败trajectory到最近cluster center的距离
- $\bar{d}$: 所有失败trajectory距离的mean
- $\sigma_d$: 所有失败trajectory距离的standard deviation

这是个z-score normalization + sigmoid的设计。$\frac{d_i - \bar{d}}{\sigma_d}$ 是标准化的距离 - 距离比mean小则positive, 比mean大则negative。Sigmoid把它压到 $(0, 1)$。

为什么这么设计? 因为不同batch内, absolute distance scale可能不同 (取决于batch里成功trajectory的diversity), 用z-score做batch-relative normalization让reward在不同batch间comparable。

### 2.3 为什么不用pixel-level reward?

Appendix E讨论了一个alternative: 用pixel-level world model (比如Cosmos-Predict2, https://arxiv.org/abs/2511.00062) 生成reference trajectory, 然后在pixel level计算similarity。

实验发现:
- Zero-shot generation的scene consistency很差 (Figure 16), 根本不能用
- 需要task-specific SFT才能让generation质量可用, 但这又回到了依赖expert demonstration的老路

Pixel-level reward的另一个问题: 它对perceptual change敏感, 不理解physics。比如gripper pause一下, pixel变化大, 但其实task progress没变。

### 2.4 为什么不用ImageBind?

ImageBind (https://arxiv.org/abs/2303.05107) 是general-purpose vision embedding, 可以encode video, 但它没在robotics data上训过, 缺乏physical intuition。

实验 (Table 3): ImageBind的reward曲线oscillatory, 有时反而reward下降, 因为它把"jerky motion"误判为progress change。

---

## 3. SRPO Optimization Objective

### 3.1 GRPO backbone

SRPO基于GRPO, 但用progress reward $g_i$ 替代binary reward。关键公式:

**Importance ratio, advantage, regularization (Eq. 6)**:
$$r_{i,t}(\theta) = \frac{\pi_\theta(a_t^{(i)} | o_t^{(i)}, l)}{\pi_{\theta_{old}}(a_t^{(i)} | o_t^{(i)}, l)}, \quad \hat{A}_i = \frac{g_i - \mu_g}{\sigma_g}, \quad \omega(\theta) = \beta D_{KL}(\pi_\theta \| \pi_{ref})$$

- $r_{i,t}(\theta)$: importance sampling ratio, 第 $i$ 条trajectory第 $t$ 步, new policy $\pi_\theta$ 和old policy $\pi_{\theta_{old}}$ 的概率比
- $\hat{A}_i$: 第 $i$ 条trajectory的advantage, 在group内做normalization (mean $\mu_g$, std $\sigma_g$)
- $\omega(\theta)$: KL regularization, 防止policy偏离reference policy $\pi_{ref}$ 太远
- $\beta$: KL weight

注意这里advantage是**trajectory-level**的, 不是step-level。整条trajectory共用一个advantage $\hat{A}_i$, 这和trajectory-level credit assignment一致。

**Clipped surrogate objective (Eq. 7)**:
$$\mathcal{L}_{t,i}^{CLIP}(\theta) = \min\left(r_{i,t}(\theta) \hat{A}_i, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i\right)$$

- $\epsilon$: clipping range, 限制importance ratio在 $[1-\epsilon, 1+\epsilon]$ 内, 防止policy update过大
- $\min$: 取clipped和unclipped的较小值, 这是PPO的标准做法

**Full objective (Eq. 8)**:
$$\mathcal{L}_{SRPO}(\theta) = \mathbb{E}_{t,i} \mathcal{L}_{t,i}^{CLIP}(\theta) + \omega(\theta)$$

- $\mathbb{E}_{t,i}$: 对time steps $t$ 和trajectory id $i$ 取期望
- $\omega(\theta)$: KL regularization term

**Group statistics (Eq. 9)**:
$$\mu_{\hat{R}} = \frac{1}{M}\sum_{j=1}^M \hat{R}_j, \quad \sigma_{\hat{R}} = \sqrt{\frac{1}{M}\sum_{j=1}^M (\hat{R}_j - \mu_{\hat{R}})^2 + \epsilon}$$

- $M$: group size (论文用8 samples per group)
- $\hat{R}_j$: 第 $j$ 条trajectory的reward (success=1.0, fail=progress reward)
- $\epsilon$: numerical stability

### 3.2 关键差异: SRPO vs GRPO

| 维度 | GRPO | SRPO |
|------|------|------|
| Reward type | Binary (0/1) | Continuous progress reward $g_i \in (0, 1)$ |
| Failed trajectory | 完全浪费 (reward=0, advantage negative uniform) | 提供dense signal (离成功近的failure拿更高reward) |
| External info | 不需要 | 不需要 (只用batch内自己的success) |
| 信号密度 | Sparse | Dense (trajectory-level) |

Intuition: GRPO里, 一条 "几乎成功, 只差最后放置" 的trajectory, 和一条 "一开始就抓错东西" 的trajectory, 拿到完全一样的信号 (reward=0)。SRPO让前者拿到高progress reward, 后者拿到低progress reward, gradient signal自然更informative。

---

## 4. 实验结果深度分析

### 4.1 Main results (Table 1)

LIBERO benchmark的4个suite (Spatial, Object, Goal, Long), 每个suite 10个task:

| Model | Avg |
|-------|-----|
| OpenVLA*-One (baseline) | 48.9% |
| + Offline SRPO | 92.5% (+43.6) |
| + Online SRPO | **99.2%** (+50.3) |
| RLinf (previous SOTA among RL) | 98.0% |
| RIPT-VLA | 96.9% |

几个关键发现:
1. **One-shot SFT起步极低** (48.9%), 但SRPO能拉到99.2%, 说明RL exploration的力量
2. **200 RL steps就达到SOTA**, 相比SFT需要tens of thousands of steps, 效率提升巨大
3. **只用third-view image + language instruction**, 击败了用wrist image + proprioception + depth的方法 (比如3D-CAVLA 98.1%)

### 4.2 Generalization (Table 2)

LIBERO-Plus有7个perturbation dimension: Camera, Robot-Init, Language, Light, Background, Noise, Layout。

| Model | Zero-shot Avg |
|-------|---------------|
| OpenVLA*-One (baseline) | 19.4% |
| + Online SRPO | 59.6% (+40.2) |
| OpenVLA-OFT (full-shot) | 69.6% |
| OpenVLA*-Full | 51.1% |

令人惊讶的是, SRPO从one-shot SFT起步, 在perturbed环境下的zero-shot performance (59.6%), 接近甚至超过了一些full-shot SFT的baseline。这说明RL的trajectory diversity比static dataset的diversity更重要。

With augmented data setting (在perturbed环境里训练), SRPO达到82.1%, 超过所有baseline包括OpenVLA-OFT+ (79.5%)。

### 4.3 Progress Reward Quality Benchmark (Table 3)

这是一个非常有价值的benchmark, 衡量reward function本身的质量, 不只是downstream task performance:

| Method | SC ↑ | Mono ↑ | MMD ↑ | JS ↑ | SMD ↑ |
|--------|------|--------|-------|------|-------|
| Pixel-level | 0.125 | 0.498 | 0.274 | 0.548 | 2.100 |
| ImageBind | 0.957 | 0.837 | 0.356 | 0.408 | 18.111 |
| **SRPO** | **0.998** | **0.992** | **0.615** | **0.572** | **188.799** |

Metric解释 (Appendix A):

**Spearman Correlation (SC, Eq. 10)**:
$$\rho = \frac{1}{N} \sum_{k=1}^N \frac{\sum_{i=1}^{T_k} (x_i^{(k)} - \bar{x}^{(k)})(y_i^{(k)} - \bar{y}^{(k)})}{\sqrt{\sum_{i=1}^{T_k} (x_i^{(k)} - \bar{x}^{(k)})^2 \sum_{i=1}^{T_k} (y_i^{(k)} - \bar{y}^{(k)})^2}}$$

- $N$: task数量
- $T_k$: 第 $k$ 个task的trajectory长度
- $x_i^{(k)}$: frame number (时间步)
- $y_i^{(k)}$: progress value (reward)
- 衡量reward和frame number的rank correlation, 越接近1说明progress越单调递增

**Monotonicity (Mono, Eq. 11)**:
$$M_{\text{mono}} = \frac{1}{N} \sum_{k=1}^N \frac{1}{T_k - 1} \sum_{t=1}^{T_k - 1} \mathbb{I}(r_{t+1}^{(k)} > r_t^{(k)})$$

- $\mathbb{I}$: indicator function
- $r_t^{(k)}$: 第 $k$ 个task第 $t$ 步的progress reward
- 衡量reward严格递增的step比例, 越接近100%越好

**MMD (Eq. 12)**: Maximum Mean Discrepancy, 衡量success和failure trajectory的分布在RKHS里的距离, 越大说明reward越能区分success/failure。

**JS Divergence (Eq. 13)**: Jensen-Shannon divergence between success和failure的reward分布, 越接近 $\ln 2$ 越好。

**SMD (Eq. 14)**: Standardized Mean Difference, effect size衡量success和failure mean的separation。

关键insight: SRPO的SMD是188.799, 远超ImageBind的18.111。这意味着latent world representation把success和failure在reward space里分得非常开, 这是RL能work的基础。

---

## 5. Ablation Studies - 深入理解设计选择

### 5.1 Self-Referential vs Fixed Expert Reference (Appendix C.1, Figure 10)

把self-referential替换成50条fixed expert trajectories per task:
- 初期训练速度comparable
- 后期plateau, 需要1.4x的steps但performance更低

**Intuition**: 固定expert trajectory限制了exploration。Policy学到一定程度后, 它的rollout已经超出expert trajectory的diversity, fixed reference无法提供nuanced progress assessment, 导致performance ceiling。

Self-referential的美妙之处: reference随policy evolution而evolve, 始终是 "policy当前能做到的最好", 形成dynamic curriculum。

### 5.2 Cluster vs Nearest Single Trajectory (Appendix C.2, Figure 10)

用nearest single success trajectory替代cluster centroid:
- 初期comparable
- 后期性能差距拉大

**Intuition**: 后期成功trajectory数量多、diversity大, 单条trajectory有noise (比如gripper pause)。Cluster centroid是 "prototypical success pattern", 对noise robust。

### 5.3 Progress Reward Weight α (Appendix D, Figure 11)

Reward公式实际是 $g_i = 1.0$ (success) 或 $\alpha \cdot \phi(\cdot)$ (fail), 测试不同 $\alpha$:
- $\alpha = 0$: 0
- $\alpha = 0.3 < 0.5 < 1.0 < 0.8$: 最优

**关键insight**: $\alpha = 0.8$ 优于 $\alpha = 1.0$。这说明即使progress reward再informative, 也不能overweight它, 否则policy会过度追求 "看起来progress" 的trajectory而忽略最终outcome。0.8是progress awareness和outcome correctness的sweet spot。

$\alpha = 0$ (纯binary reward) 最差, 直接验证了sparse reward是核心瓶颈。

---

## 6. Real-World Transfer (Section 5.4, Appendix G)

### 6.1 Setup

Real robot实验用offline RL (AWR - Advantage-Weighted Regression, https://arxiv.org/abs/1910.00177), 因为online exploration在real robot上dangerous且reset慢。

5个task: Put apple/pear into plate, Fold towel, Clean whiteboard, Select poker。

用两个VLA backbone: $\pi_0$ (diffusion-based, https://arxiv.org/abs/2410.24164) 和 $\pi_0$-FAST (autoregressive with frequency tokenization, https://arxiv.org/abs/2501.09747)。

### 6.2 AWR integration

**Advantage formula (Eq. 15)**:
$$A_{i,t} = \frac{D_{i,t} - \mu}{\sigma}$$

- $D_{i,t} = R_{i,t} - R_{i,t-1}$: incremental progress at step $t$
- $R_{i,t}$: expected cumulative reward (progress reward) at step $t$
- $\mu, \sigma$: mean和std across all trajectories

这里用的是step-level incremental progress, 不是trajectory-level。AWR是off-policy的, 用advantage加权maximum likelihood。

### 6.3 Real-world results (Figure 8, Table 4)

| Backbone | SFT baseline | + SRPO reward | Gain |
|----------|--------------|---------------|------|
| $\pi_0$ | varies | varies | +66.8% avg |
| $\pi_0$-FAST | varies | varies | +86.7% avg |

Progress Reward Benchmark在real-world上也work (Table 4): SC=0.989, Mono=0.978, SMD=169.5, 说明V-JEPA 2的latent space从simulation到real-world有良好的transferability, 不需要domain-specific fine-tuning。

这是非常强的结果 - 整个framework从simulation design到real-world deployment, world model的latent representation没有retrain。

---

## 7. Trajectory Diversity Analysis (Section 5.3, Figure 6, 7)

这是我个人觉得最fascinating的部分。比较full-shot SFT和one-shot SFT + SRPO在action space的分布:

1. **SRPO explores unreachable regions**: action space里SFT policy覆盖不到的区域, SRPO policy覆盖到了
2. **SRPO generates more dispersed trajectories**: 不只是模仿demonstration的特定路径, 而是spatial exploration

Figure 7更直接: 即使只给1条success demonstration, SRPO fine-tune后的policy能发现novel grasping positions和spatial paths。

**Intuition**: 这解释了为什么SRPO在LIBERO-Plus (perturbed) 上generalization好 - 它本质上学到的是affordance understanding和motor skill的diversity, 而不是specific demonstration trajectory。这是超越imitation learning ceiling的关键。

---

## 8. Training Efficiency (Section 5.2, Figure 5)

不同suite达到SOTA所需的RL steps:
- Spatial: 79 steps
- Object: 59 steps
- Goal: 103 steps
- Long: 219 steps

对比SFT需要tens of thousands of steps, 这是数量级的提升。

Long suite (219 steps) 比Object (59 steps) 难, 因为long-horizon task的成功率更低, 失败trajectory更多, progress reward的价值更明显。Figure 5显示SRPO在Long suite的efficiency slope比GRPO陡得多, 正是因为它从failure里extract了信号。

---

## 9. 我对这篇paper的critique和open questions

### 9.1 Strengths
1. **Idea elegant**: self-referential消除了external supervision需求, 真正autonomous
2. **Latent space choice合理**: V-JEPA 2的physics-aware representation是key, 不是随便一个video encoder都work
3. **Benchmark design好**: Progress Reward Benchmark (Table 3) 提供了评估reward function quality的framework, 这对未来VLA-RL工作有借鉴意义
4. **Real-world transfer强**: simulation design直接deploy到real robot, 没有domain gap issue

### 9.2 Open questions
1. **Batch内必须有成功trajectory**: 早期训练 (policy很差) 时, 如果一个batch全失败, 没有reference可用。论文没讨论这个cold start问题。可能依赖one-shot SFT提供initial success rate (48.9%) 来bootstrap
2. **V-JEPA 2的limitation**: 如果task涉及V-JEPA 2 pretrain data里没见过的physical interaction (比如deformable object manipulation), latent representation可能不capture progress。论文Fold Towel task结果支持这个担忧
3. **Cluster数量的sensitivity**: DBSCAN的eps和min_samples参数对结果的影响没详细讨论
4. **Long-horizon task的credit assignment**: SRPO是trajectory-level advantage, 对很长的task, "中间一步关键错误" 和 "中间一步noise" 区分不开。可能需要step-level progress reward

### 9.3 更deep的connection

这篇paper让我想到几个相关工作:

1. **Contrastive RL**: 用success trajectory做positive sample, failure做negative sample, 学representation。SRPO本质上是implicit contrastive, 但用reward shaping而不是representation learning
2. **Hindsight Experience Replay (HER)**: https://arxiv.org/abs/1707.01495 - 用achieved goal替代intended goal。SRPO的self-referential和HER有哲学上的相似 - 都是利用batch内已有的信息
3. **Decision Transformer / Trajectory-level RL**: 把RL当sequence modeling, SRPO的trajectory-level reward和这些方法有connection
4. **V-JEPA 2作为world model**: Yann LeCun的JEPA系列 (https://arxiv.org/abs/2301.08243) 主张学习latent space的predictive model而不是pixel-level reconstruction, 这篇paper是JEPA思想在robotics RL reward shaping上的应用, 验证了LeCun的thesis - latent representation更transferable

---

## 10. Key References

- V-JEPA 2 (world model): https://arxiv.org/abs/2506.09985
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- LIBERO: https://arxiv.org/abs/2306.03310
- LIBERO-Plus: https://arxiv.org/abs/2510.13626
- $\pi_0$: https://arxiv.org/abs/2410.24164
- $\pi_0$-FAST: https://arxiv.org/abs/2501.09747
- AWR: https://arxiv.org/abs/1910.00177
- DBSCAN: https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf
- ImageBind: https://arxiv.org/abs/2303.05107
- Cosmos-Predict2: https://arxiv.org/abs/2511.00062
- HER: https://arxiv.org/abs/1707.01495
- JEPA (original): https://arxiv.org/abs/2301.08243
- SiiRL framework: https://arxiv.org/abs/2507.13833
- SimpleVLA-RL: https://arxiv.org/abs/2509.09674
- RLinf: https://arxiv.org/abs/2510.06710
- RIPT-VLA: https://arxiv.org/abs/2505.17016
- VLA-RL: https://arxiv.org/abs/2505.18719
- TGRPO: https://arxiv.org/abs/2506.08440
- World-Env: https://arxiv.org/abs/2509.24948
- PPO: https://arxiv.org/abs/1707.06347
- GR00T N1: https://arxiv.org/abs/2503.14734
- 3D-CAVLA: https://arxiv.org/abs/2505.05800
- UniVLA: https://arxiv.org/abs/2505.06111
- SmolVLA: https://arxiv.org/abs/2506.01844
- WorldVLA: https://arxiv.org/abs/2506.21539
- NORA: https://arxiv.org/abs/2504.19854
- CoT-VLA: https://arxiv.org/abs/2503.22020
- TraceVLA: https://arxiv.org/abs/2412.10345
- RT-2: https://arxiv.org/abs/2307.15818
- Code: https://github.com/sii-research/siiRL
- Models: https://huggingface.co/collections/Sylvest/srpo

---

## 11. Build Intuition: 一句话总结

SRPO把RL训练里的 "failure is wasted" 变成 "failure is reference for future success", 通过V-JEPA 2的physics-aware latent space做behavioral similarity的度量, 让policy从自己的成功里学习, 从自己的失败里也学习, 完全autonomous, 不需要external supervision。这本质上是把contrastive learning的思想融入policy gradient, reference是dynamic的 (batch内自己的success), encoding是physics-aware的 (V-JEPA 2 latent), 信号是dense的 (progress reward而不是binary)。

希望这个讲解能帮你建立intuition, Karpathy。如果你想深入某个具体aspect (比如V-JEPA 2的latent space为什么physics-aware, 或者DBSCAN参数的影响, 或者step-level vs trajectory-level credit assignment的trade-off), 我可以展开讲。
