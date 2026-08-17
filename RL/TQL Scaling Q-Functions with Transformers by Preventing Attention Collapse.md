---
source_pdf: TQL Scaling Q-Functions with Transformers by Preventing Attention Collapse.pdf
paper_sha256: 12a71da1ad5fd21728d8ab8bd9787c7ce123e9240c37d24db8eb348e47db1cda
processed_at: '2026-08-12T17:42:21-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的，我来用更接地气的"人话"帮你 build 一下这篇 paper 的 intuition，把那些公式和架构翻译成画面感更强的故事。

# 用人话讲讲 TQL：Transformer 在 RL 里为什么 Scale 不起来，以及怎么救

## 1. 一句话总结这篇 paper 在干嘛

这篇 paper 就在解决一件事：**在 reinforcement learning 里，你想把 Q-function 的 Transformer 模型做得更大（参数更多），结果发现性能不仅没提升，反而崩了。作者查出罪魁祸首是 Attention Collapse（注意力坍塌），然后发明了 TQL 来强行保住 attention 的多样性。**

---

## 2. 核心矛盾：大模型在 RL 里为什么"变蠢"

如果在做 NLP 或者 vision，你把 Transformer 从 1M 参数 scale 到 26M，loss 肯定是往下掉的，大家都很开心。但在 RL 里，你这么搞，**success rate 直接从 46% 掉到 6%**。

这就很反直觉了。因为 policy 是从 Q-function 派生出来的，Q-function 变强，policy 理应变强。而且 Transformer 在别的地方 scale 得那么好，凭啥到了 value learning 就拉胯？

作者没有停留在"RL 不稳定"这种玄学结论上，他跑去拿显微镜看了一下大模型到底在干嘛。

---

## 3. 诊断结果：Attention 被"垄断"了

作者给不同大小的 Transformer 模型画了几张图，发现了三个并发的症状：

**(a) Attention 变得极度不平衡**
小模型的 attention 还算均匀，各个 token 都能照顾到。大模型的 attention 几乎全部集中在那么一两个 token 上，其他的 token 它看都不看。用术语说就是 **Attention Entropy（注意力的熵）趋近于 0**。熵是衡量分布均匀程度的，全集中在一个点上，熵就是 0。

**(b) Q-value 的地貌变得像锯齿**
如果把 Q(s, a) 画出来，小模型画出来是平滑的曲面，大模型画出来全是高频的毛刺、断崖。对于一个 value function 来说，相邻的 state-action 应该有相近的 value，这种锯齿状的地貌是完全不合理的，说明模型在 overfitting 噪声。

**(c) 性能狂跌**

### 为什么 RL 里特别致命？（Bootstrapping 的诅咒）

其实在 supervised learning 里，也有人发现过大模型会出现 attention collapse（比如 Zhai et al. 2023 在 ViT-22B 上）。但 SL 里这事没那么快搞死模型，因为 **SL 的 target 是固定的 ground truth**。就算 attention 抽风只看一个 token，只要 loss 还在，gradient 就会把模型往正确方向拉。

但在 RL 的 Q-learning 里，这叫 **Bootstrapping（自举）**。看看这个公式：

$$\mathcal{L}_Q(\phi) = \mathbb{E}\left[ \left( Q_\phi(s,a) - r - \gamma Q_{\phi'}(s', a') \right)^2 \right]$$

- $Q_\phi(s,a)$: 当前模型预测的当前状态的价值
- $r$: 当前拿到的 reward
- $\gamma$: discount factor
- $Q_{\phi'}(s', a')$: **target network** 预测的下一个状态的价值。这个 target network 其实就是当前模型的一个旧副本。

也就是说，**RL 的 target 是模型自己生成的**。
一旦 attention collapse 了，模型对输入极度敏感，Q-value 地图全是毛刺。那么它生成的 target $Q_{\phi'}(s', a')$ 也是毛刺。模型拿毛刺当 target 去学习自己，学出来的东西更毛刺。这叫正反馈死循环，直接把模型带沟里去了。

这就是为什么 attention collapse 在 RL 里比在 SL 里致命得多——**Bootstrapping 把 instability 放大成灾难**。

---

## 4. TQL 的解法：强行给 Attention "降温"

既然问题是 attention 太集中，TQL 的思路简单粗暴：**直接控制 attention 的 entropy，让它往一个预设的 target 值 $\bar{H}$ 靠拢，不让它塌缩。**

这个思路是从 SAC (Soft Actor-Critic) 那里借来的。SAC 是控制 policy 的 entropy 防止 policy 太 deterministic；TQL 是控制 attention 的 entropy 防止 attention 太 sparse。

### 4.1 怎么控制？双重 Loss 调节法

TQL 引入了一个可学习的温度参数 $\alpha$，并且构造了两个 loss 打配合：

**Loss 1: Attention Loss（尽量让 attention 分散）**
$$\mathcal{L}_{\text{att1}}(\phi) = -\frac{1}{L} \sum_{\ell=1}^{L} \alpha^\ell H^\ell$$
- $H^\ell$: 第 $\ell$ 层的 attention entropy
- $\alpha^\ell$: 第 $\ell$ 层的温度参数（权重）
- 前面有负号，意味着最小化这个 loss 就是在**最大化 entropy**，逼着 attention 分布得开。
- $\alpha^\ell$ 是权重，如果 $\alpha^\ell$ 很大，逼模型分散 attention 的力度就很大。

**Loss 2: Temperature Loss（动态调节温度 $\alpha$）**
$$\mathcal{L}_{\text{temp}}(\alpha) = \frac{1}{L} \sum_{\ell=1}^{L} \alpha^\ell (H^\ell - \bar{H})$$
- $\bar{H}$: 你设定的目标 entropy。
- 这个 loss 是用来调节 $\alpha$ 的。如果当前 entropy $H^\ell$ 低于 target $\bar{H}$（attention 塌缩了），梯度下降会让 $\alpha^\ell$ 变大；如果 $H^\ell$ 高于 $\bar{H}$，$\alpha^\ell$ 就变小。

**形成了一个自动控制的闭环（Homeostasis）：**
1. Attention 塌了 -> $H^\ell < \bar{H}$ -> $\alpha^\ell$ 自动变大 -> Attention Loss 的惩罚力度变大 -> 强迫模型把 attention 摊开 -> Entropy 回升。
2. Attention 太散了 -> $H^\ell > \bar{H}$ -> $\alpha^\ell$ 自动变小 -> 模型可以自由地集中 attention。

这比直接写死一个惩罚权重聪明得多，它是在动态平衡。

### 4.2 两个关键的工程细节

作者发现，光有上面那个大框架还不够，还有两个细节必须做对：

**(a) 每层要有自己的 $\alpha$**
Transformer 不同层学的东西不一样，底层可能看局部特征，高层看全局。如果强行让所有层的 entropy 一样，会破坏这种层次结构。所以必须 layer-wise。

**(b) [VALUE] token 要有专门的 $\alpha$**
在这个架构里，前面 prepend 了一个特殊的 `[VALUE]` token（类似 BERT 的 `[CLS]`），它的输出就是最终的 Q 值。它的任务是当"汇总者"，把所有 token 的信息收拢起来。它的 attention pattern 跟普通 token 肯定不一样，所以它的 $\alpha$ 也要单独设。

---

## 5. 实验结果一句话：真的 Scale 起来了

作者在 OGBench 的 25 个任务上，把模型从 0.4M 一直 scale 到 26M：
- **以前的 baselines**（不管是 MLP 的 FQL，flow-matching 的 floq，还是 transformer 的 PAC）：全都在大模型上掉点，平均跌 10.6%。
- **TQL**：一路涨，从最小到最大提升 43%。

这说明 TQL 真的把 Transformer 在 RL 里的 scaling 能力解锁了。

---

## 6. 脑洞大开：我的发散与联想

这篇 paper 虽然看着是 RL 的 trick，但背后的洞察其实很深，我联想到了几个方向：

### 6.1 和 Self-Supervised Learning (SSL) 的联系
RL 的 Q-learning 是一种 bootstrapping，target 是模型自己生成的。这和 BYOL、SimSiam 这类不需要负样本的自监督学习方法非常像——它们也是用一个网络的输出去预测另一个网络（target network）的输出。
既然 attention collapse 在 bootstrapping 下这么致命，那在 SSL 里大模型训练会不会也有类似的问题？也许 TQL 这种 attention entropy control 能直接搬到 BYOL 之类的架构里防止 representation collapse。这是一个很值得挖的坑。

### 6.2 Online RL 里的探索
Paper 只测了 offline RL（数据是固定的）。在 online RL 里，情况更复杂。因为 attention 如果太均匀，模型可能无法 focus 到关键 state 上去做 exploration；如果太集中，又会 collapse。TQL 在 online RL 的 exploration-exploitation tradeoff 里会扮演什么角色？如果配合 network reset（Schwarzer et al. 2023 的 BBF）一起用，能不能彻底解决 online RL 的 plasticity loss 问题？

### 6.3 如果 State 是高维图片呢？
OGBench 的 state 都是低维的向量（比如物体坐标、关节角度），所以每个 scalar 当一个 token 没问题。如果 state 是图片，先经过 CNN/ViT patch 化，那 token 数量会爆炸。这时候 attention entropy 的 target $\bar{H}$ 怎么设？是不是需要 hierarchical 的 attention？Paper 没碰这个，但如果是做机器人 VLA (Vision-Language-Action) 模型，这是必须面对的。

### 6.4 Attention Entropy 的 Curriculum
Paper 用的是一个固定的 target $\bar{H}$。直觉上，训练刚开始时，模型啥也不会，应该让它 attention 均匀点去 explore patterns；训练后期，模型学到了，应该允许它 attention 集中点去 commit 到最优 pattern。如果给 $\bar{H}$ 加个 schedule（比如从高到低 decay），会不会比固定值更好？这和 learning rate warmup 是一个道理。

### 6.5 MoE (Mixture of Experts) 的类比
其实 Attention 有点像一种 soft routing。Attention collapse 就是 routing 塌缩到了一个 expert 上。在 MoE 领域，也有类似的 load balancing loss 来防止所有 token 都只去一两个 expert。TQL 的 entropy regularization 和 MoE 的 load balancing 在哲学上是同构的。如果把 Transformer 的每一层看成一组 experts（每个 token 位置是一个 expert），TQL 就是在做 attention 级别的 load balancing。

---

## 7. 总结一下你的 Intuition

如果要带走一句话，那就是：**大模型在 RL 里崩溃，是因为它在 Bootstrapping 的放大效应下，Attention 退化成了只看几个 token 的"瞎子"，导致 Q-value 地图变成锯齿状并自我强化。TQL 借用 SAC 的自动温度调节机制，强行把 Attention 的熵拉回正常值，保住了模型的视野，从而解锁了 Scaling。**

这个工作的 taste 很好，它不是无脑堆架构或者堆 trick，而是先找到一个非常具体、可量化的病灶（attention entropy 趋近于 0），然后对症下药。这种"诊断驱动"的研究方法，其实比方法本身更有价值。

## 参考链接

- **TQL 项目主页**: https://pd-perry.github.io/transformer-q-learning/
- **Zhai et al. 2023 (SL 里的 Attention Collapse 开山之作)**: https://arxiv.org/abs/2303.06296
- **SAC (TQL 的 Entropy 控制思想来源)**: https://arxiv.org/abs/1801.01290
- **Q-Transformer (之前用 Transformer 做 Q-function 的尝试)**: https://arxiv.org/abs/2309.10150
- **OGBench (评测基准)**: https://arxiv.org/abs/2410.20092
- **FQL (TQL 用的 Policy Extraction 方法)**: https://arxiv.org/abs/2502.02538

---

# TQL: 通过防止 Attention Collapse 来 Scaling Q-Functions

这篇 paper 来自 Stanford 的 Perry Dong, Dorsa Sadigh, Chelsea Finn 等人，发表时间应该是 2025 年底左右。核心 idea 非常 elegant：**RL 中用 transformer 做 value function 时，scaling 不起来的根本原因是 attention entropy collapse**，而解决方法是直接把 attention entropy 当成一个被调控的量，往 target 值上拉。让我一层层拆给你看，build 一些 intuition。

---

## 1. The Puzzle: 为什么 Transformer 在 RL Value Learning 中 Scale 不起来？

在 supervised learning (NLP, vision) 里，transformer 的 scaling law 是近乎宗教般的存在——参数翻倍，loss 按幂律下降。但 paper 第一段 Figure 1 / Figure 3 给出一个反直觉的实验事实：

在 OGBench 的 cube-double 任务上，把一个 transformer value network 从 **0.4M → 1M → 7M → 26M** scale 上去，**success rate 从 46% 掉到 6%**。这不是 plateau，是**反向 scaling**——给更多 capacity 反而更差。

这个现象本身就很诡异，因为：
- Policy 最终是从 value function 派生出来的，value function 更强，policy 应该更强；
- transformer 在其它领域 scale 这么好，为什么偏偏在 value function 上崩溃？

paper 没有停留在"RL training unstable"这种模糊结论上，而是去**找了一个 measurable 的失败模式**。

---

## 2. The Diagnosis: Attention Entropy Collapse

作者做了一个非常漂亮的 empirical analysis (Section 5.2, Figure 3, Figure 7, Figure 8)：

### 2.1 三个并发的病态现象

随着 model size 增大：

**(a) Attention entropy 单调下降**
- 0.4M 模型的 attention distribution 还相对均匀
- 26M 模型的 attention 几乎完全 collapse 到少数几个 token 上，entropy 接近 0
- Figure 7 显示这个趋势在 5 个 environment 上一致出现

**(b) Q-value landscape 变得非平滑**
- 小模型的 Q(s,a) contour map 是光滑的曲面
- 大模型出现高频振荡、不连续、像 overfitting 的锯齿状表面
- 这非常致命——value function 本质上应该是 Lipschitz 连续的（相邻 state-action 应该有相近的 value），而 collapse 的 attention 把这个归纳偏置破坏了

**(c) Performance 反向 scale**
- 上述两个现象和性能下降高度相关

### 2.2 为什么是 Attention Collapse？Intuition

这里我帮你 build 一下直觉。在 supervised learning 里，attention collapse 其实也发生过——Zhai et al. 2023 的 *Stabilizing Transformer Training by Preventing Attention Entropy Collapse* 就是在 ViT-22B 之类的规模上观察到这个问题。但 SL 里它出现得晚、没那么致命，原因有两个：

1. **SL 的目标是 fixed target**：label 不会因为模型变强而变。即使 attention 尖锐化，gradient signal 还在把模型往正确方向推。
2. **RL 的 target 是 bootstrapped 的**：target Q 是 $r + \gamma Q_{\phi'}(s', a')$，**target 自己就是模型自己**（delayed copy）。如果 attention collapse 导致 Q-value landscape 出现锯齿，那么 bootstrap target 也会出现锯齿，于是 model 学到的是**自己的噪声**，形成正反馈崩溃。

这就是为什么 RL 对 attention collapse 特别敏感——**bootstrapping 把 instability 放大了**。这一点和 why value-based RL 比 policy gradient 更难 scale 的根本原因是一脉相承的。

### 2.3 另一个 angle：value function 是 regression 问题

Q-learning 是 regression，MSE loss。在 regression 里，大模型很容易 fit 出 high-frequency noise（这是 classical bias-variance tradeoff 在 deep net 里的表现，见 Belkin et al. 的 double descent）。SL 里我们靠 regularization、early stopping、weight decay 等控制；但 RL 里因为 bootstrap 的存在，这些手段不够。

---

## 3. The Method: TQL 的 Attention Entropy Control

TQL 的核心思想一句话：**直接把 attention entropy 当作一个被调节的量，强制它往一个 target 值 H̄ 靠拢**。这非常像 SAC (Soft Actor-Critic) 里的 entropy-regularized RL，只不过 SAC 调的是 policy 的 entropy，TQL 调的是 attention 的 entropy。

### 3.1 基础架构

先把 Q-function 用 transformer 实现（Section 4.1）：
- state $s \in \mathbb{R}^{n_s}$ 和 action $a \in \mathbb{R}^{n_a}$ 的**每个 scalar 维度当作一个 token**
- 每个 token 线性投影到 hidden dimension $H$
- 加上 learnable positional encoding
- 前面 prepend 一个 `[VALUE]` token，类似 BERT 的 `[CLS]`
- 经过 L 层 transformer decoder（full self-attention，no mask）
- `[VALUE]` token 的最终表示过 MLP head → Q 值

标准 Q-learning loss（式 1）：

$$\mathcal{L}_Q(\phi) = \mathbb{E}\left[ \left( Q_\phi(s,a) - r - \gamma Q_{\phi'}(s', a') \right)^2 \right]$$

变量含义：
- $\phi$: Q-network 参数
- $\phi'$: target network 参数（polyak averaged: $\phi' \leftarrow \tau \phi + (1-\tau)\phi'$）
- $\gamma \in [0,1]$: discount factor (paper 用 0.99)
- $r$: immediate reward

### 3.2 Attention Entropy 的形式化

对于第 $\ell$ 层，attention score matrix $\bar{A}^\ell \in \mathbb{R}^{n \times n}$（softmax 之后），其中 $n = 1 + n_s + n_a$（1 是 [VALUE] token）。

第 $i$ 个 query token 的 attention entropy（式 2）：

$$H_i^\ell = -\sum_{j=1}^{n} A_{ij}^\ell \log A_{ij}^\ell$$

- $A_{ij}^\ell$: layer $\ell$ 中 query $i$ 对 key $j$ 的 attention weight（softmax 后，行和为 1）
- $i$: query token index
- $j$: key token index
- 当 attention 完全均匀分布时 $H_i^\ell = \log n$，完全集中时 $H_i^\ell = 0$

整层的 entropy 平均：$H^\ell = \frac{1}{n}\sum_{i=1}^{n} H_i^\ell$

### 3.3 Temperature 调节机制（核心）

借鉴 SAC 的 maximum entropy 框架，引入一个 learnable temperature $\alpha$（实际参数化为 $\exp(\hat{\alpha})$ 保证正性），两个 loss：

**Temperature loss**（式 3）：

$$\mathcal{L}_{\text{temp}}(\alpha) = \frac{1}{L} \sum_{\ell=1}^{L} \alpha^\ell (H^\ell - \bar{H})$$

这个 loss 是 SAC 里那个 dual 的形式：当 $H^\ell > \bar{H}$（entropy 太高），$\alpha^\ell$ 增大；当 $H^\ell < \bar{H}$（entropy 太低，collapse），$\alpha^\ell$ 减小。

**Attention loss**（式 4）：

$$\mathcal{L}_{\text{att1}}(\phi) = -\frac{1}{L} \sum_{\ell=1}^{L} \alpha^\ell H^\ell$$

这个 loss 最大化 entropy（前面有负号），权重是 $\alpha^\ell$。

**联合训练**（式 5）：

$$\mathcal{L}_{\text{critic}}(\phi, \alpha) = \mathcal{L}_Q(\phi) + \mathcal{L}_{\text{att}}(\phi) + \mathcal{L}_{\text{temp}}(\alpha)$$

让我帮你推导一下这个机制为什么 work。对 $\alpha^\ell$ 求导：

$$\frac{\partial \mathcal{L}_{\text{temp}}}{\partial \alpha^\ell} = H^\ell - \bar{H}$$

当 $H^\ell < \bar{H}$（attention collapse 了），梯度是负的，gradient descent 会让 $\alpha^\ell$ **增大**。

对 $\phi$ 求导 $\mathcal{L}_{\text{att}}$：

$$\frac{\partial \mathcal{L}_{\text{att}}}{\partial \phi} = -\alpha^\ell \frac{\partial H^\ell}{\partial \phi}$$

这会**最大化** $H^\ell$（因为前面是负号），强度正比于 $\alpha^\ell$。

所以反馈环是：
- entropy 太低 → $\alpha^\ell$ 增大 → attention loss 的权重增大 → 模型被更强地推往高 entropy 方向 → entropy 回升
- entropy 太高 → $\alpha^\ell$ 减小 → attention loss 权重减小 → 模型可以自由降低 entropy

这是一个 **自动调节的 homeostasis**，比 fixed entropy penalty 灵活得多。Section 5.5 的 ablation 也验证了这点：用 max entropy loss（固定惩罚）比 adaptive target entropy 差很多。

### 3.4 两个关键的设计细节

paper 在这里有两个 design choice 我觉得很关键，分别对应 Section 4.2 末尾：

**(a) Layer-wise temperature $\alpha^\ell$**：每层有自己的 temperature，因为不同层学到不同的 attention pattern。底层可能关注局部 token 关系，高层关注全局。强行让所有层 entropy 一样会破坏这种 hierarchy。

**(b) Token-wise temperature for [VALUE] token $\alpha^\ell_{[\text{VALUE}]}$**：[VALUE] token 的角色和其他 token 不同——它是 aggregator，要从所有 token 收集信息产出最终 Q 值。它的 attention pattern 应该有自己的 entropy profile，不能和普通 token 混在一起调。

ablation (Figure 5) 显示：用单一 temperature 替代 layer-wise + token-wise 会让性能明显下降，attention map 出现 over-specialization。

### 3.5 Modality Embedding（小但有用）

Section 4.3 提到一个小 trick：给 state token 和 action token 分别加 learnable embedding $e_s$ 和 $e_a$。这让模型能区分"这个维度是 state 还是 action"，对 attention 学习有帮助。这其实和 ViT 的 patch embedding / segment embedding 思路类似，paper 里强调这对大模型尤其重要，因为大模型 capacity 足够，需要这些"hint"来知道往哪里分配 attention。

### 3.6 Policy Extraction

TQL 本身只关心 value function，policy extraction 可以 plug 任何方法。paper 用了 FQL (Flow Q-Learning, Park et al. 2025b) 的 scheme：

- 一个 BC flow policy $\pi_\theta^\beta$（imitation，式 6）
- 一个 one-step flow policy $\pi_\omega$（task policy，式 7）
- 通过 distillation 让 $\pi_\omega$ 在 behavior-constrained 的前提下最大化 Q

具体公式（式 7）：

$$\mathcal{L}_{\text{OS}}(\omega) = \underbrace{\mathbb{E}_{s \sim \mathcal{D}, a^\pi \sim \pi_\omega}[-Q_\phi(s, a^\pi)]}_{\text{Q maximization}} + \underbrace{\alpha \mathcal{L}_{\text{Distill}}(\omega)}_{\text{BC constraint}}$$

- $\omega$: task policy 参数
- $\alpha$: BC coefficient（环境相关，见 Table 3，比如 cube-double 是 300）
- $\mathcal{L}_{\text{Distill}}$（式 8）: 让 $\pi_\omega$ 的 mean $\mu_\omega(s,z)$ 逼近 BC policy 的 mean $\mu_\theta(s,z)$，$z \sim \mathcal{N}(0, I_d)$ 是 flow 的 noise variable

---

## 4. 实验结果的核心信号

### 4.1 Scaling（Figure 4，核心结果）

作者对比了 4 个 backbone：
- **FQL** (MLP)
- **floq** (flow-matching based)
- **PAC** (transformer based, Springenberg et al. 2024)
- **Transformer baseline** (TQL 的架构但去掉 entropy control)
- **TQL** (本文)

scale 范围：0.4M → 1M → 7M → 26M

结果总结：
- **所有 baseline 在大模型上都退化**，平均下降 10.6%
- **TQL 单调上升**，从最小到最大提升 43%
- transformer baseline 在 7M 时还能稍微涨一点，到 26M 就崩了（和 attention collapse 程度最严重对应）

### 4.2 和 9 个 baseline 的全面对比（Table 1）

25 个 OGBench 任务，TQL 在 4/5 domain 上最好，总平均 40±7（次好 floq 是 34±7）。比较有意思的是：
- **cube-double**: TQL 67 vs floq 41 vs FQL 29 vs IQL 6
- **cube-triple**: 所有方法都很差（4-7），TQL 7 略胜
- 这个 cube-triple 几乎所有方法都接近 0，说明任务很难，benchmark 还有 headroom

### 4.3 Ablation（Figure 5, 6）

四个变体在 26M 模型上的 cube-double：
1. 去掉 entropy guidance（vanilla transformer）→ 性能最低（attention collapse 重现）
2. Fixed entropy penalty（max entropy loss）→ 比 TQL 差，因为不稳定
3. Single temperature → 比 layer+token-wise 差
4. Full TQL → 最好

Figure 6 的 attention map 可视化非常直观：
- vanilla: 极度 sparse，几个 token 独大
- fixed penalty: 比 vanilla 好但仍 over-specialized
- single temp: 部分层 collapse
- TQL: 最 balanced，所有 token 都有 reasonable attention

### 4.4 和 SL 的 stabilization 技巧对比（Appendix B.2）

这是一个很有意思的实验：作者把 supervised learning 里防 attention collapse 的几种方法（QK Norm, σReparam, RMSNorm+SandwichNorm+QKNorm+SwiGLU）搬到 RL 上，发现它们确实能缓解 collapse（Figure 13），但性能不如 TQL（Figure 12）。这印证了 RL value learning 对 attention stability 的要求比 SL 更严格——SL 的 trick 是"在 boundary 上推一下"，RL 需要的是"持续的、有 target 的 homeostasis"。

---

## 5. Hyperparameter 选择（Section C.3）

target entropy $\bar{H}$ 是新增的唯一关键超参。paper 给了一个很实用的 recipe：

1. 计算上界 $H_{\max} = \ln(1 + n_s + n_a)$（均匀分布的 entropy）
2. 初始 target 设为 $0.8 \times H_{\max}$
3. 在 ±0.5 范围内 local search
4. 给 output layer 一个固定 -0.5 的更低 target（鼓励输出层更 deterministic）

Table 3 里的实际值：
- cube-double: `((3.0, 3.0), (2.5, 2.5))` — 第一层 [VALUE]=3.0, 其他=3.0；第二层 [VALUE]=2.5, 其他=2.5
- cube-triple/puzzle/scene: 类似的递减 pattern

注意**第二层 entropy target 比第一层低**，这和"深层应该更 deterministic"的直觉一致。

---

## 6. 一些 Critical Thoughts 和 Open Questions

### 6.1 为什么 attention entropy collapse 在 RL 里这么严重？

paper 给了 empirical evidence 但**没有给出严格的理论解释**。我自己的理解是：
- RL 的 bootstrap loss 把模型的 high-frequency error 放大成 target 的 high-frequency error
- attention collapse 后，模型对输入扰动极其敏感（因为只看几个 token）
- 在 bootstrapping 下，这种 sensitivity 被 self-amplified

如果是这样，那么任何 self-distillation 类的训练（包括 BYOL, SimSiam 一类的 self-supervised learning）应该也会有类似问题。这值得深挖。

### 6.2 Target entropy 的 schedule

paper 用固定的 $\bar{H}$，但其实可以想象一个 curriculum：训练初期保持高 entropy（让模型 explore attention patterns），后期逐渐降低（让模型 commit 到最优 pattern）。这有点像 learning rate warmup + decay 的思路。paper 没做这个 ablation。

### 6.3 [VALUE] token 的特殊地位

给 [VALUE] token 独立 temperature 是个很聪明的 design，因为它的 attention pattern 直接决定 Q 值的 smoothness。但 paper 没有更细的 ablation 区分 [VALUE] token 的 entropy 和普通 token 的 entropy 各自的相对重要性。

### 6.4 只测了 offline RL

TQL 在 offline RL 上 work，但 online RL 里情况可能更复杂——因为 online RL 还有 exploration-exploitation tradeoff，UTD (update-to-data) ratio 的影响，network reset 等问题。paper Section 2 提到 online RL 里常用 network reset (Schwarzer et al. 2023, Nauman et al. 2024) 来维持 plasticity。TQL 和这些方法是否兼容、是否能替代它们，是个开放问题。作者说 "TQL can in principle be applied to any value-based policy extraction scheme"，但没在 online setting 验证。

### 6.5 Modality embedding 的局限性

state 和 action 的区分通过 learnable embedding 实现很好，但如果 state 本身是 heterogeneous 的（比如 image + proprioception + language instruction），可能需要更细粒度的 modality encoding。这是个自然的 extension。

### 6.6 Attention collapse vs. 其他 collapse

attention collapse 是 value function collapse 的一个 proximate cause，但还有其他可能的 failure mode：
- representation collapse (feature 维度塌缩)
- gradient explosion/vanishing
- overfitting on offline dataset

paper 的 attention entropy regularizer 是一个非常针对性的 fix。如果 RL scaling 还有其他 bottleneck，TQL 可能不够。但作为一个 diagnostic-driven 的、最小侵入的方法，它已经覆盖了相当大的问题空间。

---

## 7. 与相关工作的脉络

帮你把这个工作放在更大的图景里：

### 7.1 Attention entropy collapse 在 SL 中
- **Zhai et al. 2023** *Stabilizing Transformer Training by Preventing Attention Entropy Collapse*: 提出 σReparam (spectral norm based) 防 ViT 大规模训练的 attention collapse
- **Zhuo et al. 2025** *HybridNorm*: hybrid normalization 改善 transformer training stability
- **Wortsman et al. 2023** *Small-scale proxies for large-scale transformer training instabilities*

这些方法在 SL 里 work，但 paper 的 Appendix B.2 显示它们在 RL value learning 里不如 TQL——因为 SL 的方法是在 boundary 上"被动"防 collapse，TQL 是"主动"维持 target entropy，在 bootstrapping 的不稳定环境下更鲁棒。

### 7.2 RL scaling 的其他方向
- **Network reset**: Schwarzer et al. 2023 (BBF), Nauman et al. 2024 — 通过周期性重置维持 plasticity，但 compute heavy 且 risk catastrophic forgetting
- **Normalization tricks**: Lee et al. 2025 (Simba) — 专用 normalization
- **MoE / multi-skip**: Obando-Ceron et al. 2024, Castanyer et al. 2025 — 架构改变
- **Categorical value**: Farebrother et al. 2024 — 把 regression 改 classification
- **深度 scaling**: Wang et al. 2025 — 1000 层 network 在 self-supervised RL 里 work

TQL 的定位是"minimal、general、不需要架构改动"，这点和上面这些 orthogonal，可以叠加。

### 7.3 Transformer 在 RL 里的其他用法
- **Trajectory modeling**: Decision Transformer (Chen et al. 2021), Online Decision Transformer (Zheng et al. 2022), Elastic DT (Wu et al. 2023)
- **World model**: Cheng et al. 2025
- **Joint actor-critic**: PAC (Springenberg et al. 2024), Q-Transformer (Chebotar et al. 2023)

TQL 区别于这些：它专注 value function scaling，不和 trajectory modeling 混。

### 7.4 Physical Intelligence 的工作
- **π* (2025a)**: 用 Monte Carlo return 训 value function，避免 bootstrap 的 instability
- **π0.5 (2025b)**: VLA 的 open-world generalization

paper 里对比提到：π* 用 MC return，而 TQL 用 bootstrapping（"shown to work better but often has more training instabilities"）。这意味着 TQL 和 π* 是两种思路——TQL 是让 bootstrap 稳定，π* 是绕开 bootstrap。两者可能可以结合：用 TQL 稳定 bootstrap，同时用 MC return 做辅助监督。

---

## 8. 公式再梳理（intuition 版）

让我把 TQL 的优化目标用一个更直观的方式重写：

最终 minimize 的总 loss 是：

$$\mathcal{L}_{\text{critic}}(\phi, \alpha) = \underbrace{\mathcal{L}_Q(\phi)}_{\text{TD learning}} + \underbrace{\left( -\sum_\ell \alpha^\ell H^\ell \right)}_{\text{maximize entropy}} + \underbrace{\left( \sum_\ell \alpha^\ell (H^\ell - \bar{H}) \right)}_{\text{tune }\alpha\text{ to hit target}}$$

把它看成关于 $\alpha^\ell$ 的 saddle point 问题：
- $\phi$ 的目标：minimize TD error + maximize (entropy × $\alpha$)
- $\alpha^\ell$ 的目标：让 entropy 收敛到 $\bar{H}$

这个 saddle-point 结构和 SAC 完全同构，只是 SAC 调的是 action distribution 的 entropy，TQL 调的是 attention distribution 的 entropy。SAC 让 policy 不要太 deterministic 以保持 exploration；TQL 让 attention 不要太 peaked 以保持 value landscape smoothness。

如果你把 transformer 想象成"在 token space 上做 soft routing"，那么 attention entropy 就是这个 routing 的"temperature"。collapse 就是 routing 退化为 hard routing（只选一个 token），失去平滑性。在 bootstrapping 下 hard routing 的 Q function 会把 high-frequency error 传给 target，于是训练崩溃。

---

## 9. 我的整体评价

这是一个**diagnostic-driven 的好 paper**：

**优点**：
1. **诊断清晰**：先找出可观测的 failure mode（attention entropy collapse），再给 fix。这种"先 understand 再 fix"的 research taste 很 Karpathy-style。
2. **方法 minimal**：不加架构改动，只加一个 loss term + 一个超参，可以 plug 进任何 transformer value function。
3. **实验严谨**：scale 在 4 个 backbone、5 个 domain、25 个任务上验证；ablation 清晰；和 SL 的 stabilization trick 对比；visualize Q landscape 和 attention map。
4. **和 SAC 的类比优雅**：把 maximum entropy RL 的成熟 framework 迁移到 attention，这种 cross-pollination 很聪明。

**可以追问的方向**：
1. Online RL 上是否 work？
2. 和 network reset / plasticity maintenance 方法如何共存？
3. Attention entropy 的 target 是否应该有 schedule？
4. 对 image-based state（高维 perceptual input）是否还有效？OGBench 都是 state-based 的低维 input。
5. 如果 value function 是 ensembled（paper 用了 K 个 MLP head），不同 head 的 attention 是否应该 share entropy target？

paper 的核心 insight——**bootstrapping 放大 attention collapse**——我觉得是 RL scaling 这个领域一个比较 deep 的发现。它把"RL training unstable"这个模糊的痛点精确化成了"attention entropy collapse under bootstrapping"这个可量化、可处理的 specific problem。这种"把模糊问题变成精确问题"的能力，是好的 empirical research 的标志。

---

## References & 链接

- TQL 项目页: https://pd-perry.github.io/transformer-q-learning/
- OGBench (benchmark): https://arxiv.org/abs/2410.20092
- SAC (maximum entropy RL，TQL 的 entropy 控制灵感来源): https://arxiv.org/abs/1801.01290
- Stabilizing Transformer Training by Preventing Attention Entropy Collapse (Zhai et al.，SL 里类似问题的开创性工作): https://arxiv.org/abs/2303.06296
- Q-Transformer (transformer value function 的早期工作): https://arxiv.org/abs/2309.10150
- Perceiver Actor-Critic (PAC): https://arxiv.org/abs/2402.05546
- FQL (Flow Q-Learning，TQL 的 policy extraction 来源): https://arxiv.org/abs/2502.02538
- floq (flow-matching critic): https://arxiv.org/abs/2509.06863
- IQL: https://arxiv.org/abs/2110.06169
- CQL: https://arxiv.org/abs/2006.04779
- Simba (normalization based RL scaling): https://arxiv.org/abs/2410.09754
- MoE for RL (Obando-Ceron et al.): https://arxiv.org/abs/2402.08609
- Stop Regressing (categorical value): https://arxiv.org/abs/2403.03950
- BBF (network reset for scaling): https://arxiv.org/abs/2305.19452
- π* (Physical Intelligence, MC return for value): https://arxiv.org/abs/2511.14759
- π0.5: https://arxiv.org/abs/2504.16054
- Decision Transformer: https://arxiv.org/abs/2106.01345
- Online Decision Transformer: https://arxiv.org/abs/2202.05607
- 1000-layer RL networks (depth scaling): https://arxiv.org/abs/2503.14858
- Value Flows (Dong et al. 2025): https://arxiv.org/abs/2510.07650
- Expo (stable RL with expressive policies): https://arxiv.org/abs/2507.07986

如果你想 build 更深 intuition，我特别推荐先读 Zhai et al. 2023 那篇——它给的是 SL 视角的 attention collapse 分析，读完再看 TQL，会发现 TQL 是把 SL 的诊断框架搬到 RL，但用了更强的 entropy 控制 mechanism。两者对照阅读非常有启发。
