---
source_pdf: Learning Interactive World Model for Object-Centric Reinforcement Learning.pdf
paper_sha256: 3e5765b3f5265fcfc450bb17c1fce16adfd1628151bee80e3acf4832b9ac5175
processed_at: '2026-08-05T13:22:46-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 FIOC-WM

好，我换个方式来讲，像在白板前聊天那样。

## 一句话说清楚这篇paper在干嘛

**你想让robot学会做家务，比如"把kettle从灶台搬到microwave旁边再打开light"。这种long-horizon任务很难，因为步骤多、object多。FIOC-WM的核心insight是：与其让neural network从pixel直接学"怎么做这个任务"，不如先学"object之间怎么交互"，然后把复杂任务拆成一系列interaction的组合。**

就这么简单。剩下都是execution细节。

## 为什么这个想法reasonable

想象你教小孩做家务。你不会跟他说"把pixel从(x1,y1)移到(x2,y2)，joint torque设为τ"。你会说"先抓住kettle的把手，然后抬起来，移到stove上方，松手"。这里"抓住"、"抬起"、"松手"就是interaction primitives。小孩脑子里有个**object-interaction vocabulary**，复杂任务是这个vocabulary的composition。

现有RL方法的问题是：它们把所有东西塞进一个大latent space，让网络自己figure out哪些信息重要、哪些object在交互。这在简单环境work，但compositional generalization差——训练时见过"push red cube"，没见过"push blue sphere"，就要重学。

FIOC-WM做的就是**explicitly学这个object-interaction vocabulary**。

## 技术上怎么落地

### Step 0: 看图，提feature

raw pixel → DINO-v2 [[1]](https://arxiv.org/abs/2304.07193) → feature map → Slot Attention [[2]](https://arxiv.org/abs/2006.15055) → N个object slots

DINO-v2是Meta的self-supervised ViT，已经能提取很好的semantic feature。Slot Attention把这些feature cluster成N个object的representation。这一步是**object discovery**，告诉你"图里有几个object，每个object大概长啥样"。

参考VideoSAUR [[3]](https://arxiv.org/abs/2310.18643)的trick：下一帧的slot用上一帧的slot + GRU predictor初始化，做temporal continuity。

### Step 1: 学每个object的"身份证"和"状态"

每个object的latent $\mathbf{s}^i$ 被进一步拆成两部分：

$$\mathbf{s}^i = (\mathbf{d}^i, \mathbf{c}^i)$$

- $\mathbf{d}^i$: dynamic stuff，会变的——position、velocity、orientation
- $\mathbf{c}^i$: constant stuff，不变的——color、shape、mass、friction

**为什么这么拆？** 因为你推一个block，改变的是它的position和velocity（dynamic），不改变它的color和mass（constant）。如果你不拆，network可能把color和position混在一起编码，结果推block时color也"变"了，这不合理。拆开后transition model只需要predict dynamic部分，维度小，容易学，而且natural支持compositional generalization——换color不影响"怎么推"的规律。

怎么强制 $\mathbf{c}^i$ 时序不变？用两个loss：

**Temporal consistency loss**:
$$\mathcal{L}_{\text{static}} = \sum_t \sum_i \|f_c(\mathbf{s}_{t+1}^i) - f_c(\mathbf{s}_t^i)\|^2$$

直觉：相邻帧的static feature应该一样，所以算L2 distance，penalty掉差异。

**Contrastive loss**:
$$\mathcal{L}_{\text{con}} = -\sum_t \sum_i \log \frac{\text{sim}(f_c(\mathbf{s}_t^i), f_c(\mathbf{s}_{t'}^i))}{\text{sim}(f_c(\mathbf{s}_t^i), f_c(\mathbf{s}_{t'}^i)) + \sum_{j \neq i} \text{sim}(f_c(\mathbf{s}_t^i), f_c(\mathbf{s}_{t'}^j))}$$

- 分子：同一object不同时间的static feature相似度（要大）
- 分母加项：不同object的static feature相似度（要小）
- 这是InfoNCE [[4]](https://arxiv.org/abs/1807.03748)的形式

直觉：防止所有object编码成同一个static vector（collapse）。

### Step 2: 学object之间怎么交互

这是最关键也最技术的部分。每两个object $(i, j)$ 在每个时刻 $t$ 可能在交互也可能不交互。用binary adjacency matrix $G_t \in \{0, 1\}^{N \times N}$ 表示。

Transition equation长这样：
$$\mathbf{d}_{t+1}^i = f_{\text{self}}(\mathbf{d}_t^i, \mathbf{c}^i, \mathbf{a}_t, \epsilon_t) + \sum_{j \in \mathcal{N}_t(i)} f_{\text{inter}}(\mathbf{d}_t^i, \mathbf{d}_t^j, \mathbf{c}^j, \delta_t)$$

人话翻译：
- $f_{\text{self}}$: "如果没人理我，我自己怎么动"。考虑action、自己的mass、friction
- $f_{\text{inter}}$: "如果object $j$ 撞了我，我的dynamic怎么变"。只看 $j$ 的state
- $\mathcal{N}_t(i)$: 当前时刻跟 $i$ 交互的object集合，由graph $G_t$ 决定
- $\epsilon_t, \delta_t$: 两套独立noise，model不确定性

**关键设计**：interaction只影响 $\mathbf{d}$（dynamic），不影响 $\mathbf{c}$（static）。这符合物理直觉——碰撞改变速度，不改变颜色。

**怎么学 $G_t$？** 三种方法，论文都做了实验：

#### Method 1: Variational masks with Gumbel-Softmax

对每对 $(i, j)$ 用GRU encoder算一个embedding $\mathbf{u}^{ij}$，然后：
$$G^{ij} \sim \text{Softmax}(\mathbf{u}^{ij} + g / \tau)$$
- $g$: Gumbel(0,1) noise，提供stochasticity
- $\tau$: temperature，高时接近uniform，低时接近hard one-hot
- 训练时anneal $\tau$ 从高到低

ELBO:
$$\mathcal{L}_{\text{mask}} = \mathbb{E}_{q(G|\mathbf{s})}[\log p(\mathbf{s}|G)] - \text{KL}[q(G|\mathbf{s}) \| p(G)]$$
- $p(G)$: sparsity prior，鼓励graph稀疏

来自 Amortized Causal Discovery [[5]](https://arxiv.org/abs/2206.08163) 和 NRI [[6]](https://arxiv.org/abs/1802.08652)。

#### Method 2: Codebook quantization

维护一组prototype vectors $\{e_1, ..., e_k\}$，每个prototype对应一种interaction type（碰撞、抓取、滑动...）。对 $\mathbf{u}^{ij}$ 做nearest neighbor lookup:
$$z = \arg\min_{j \in [k]} \|\mathbf{u}^{ij} - e_j\|_2$$
$$G \sim g_{\text{dec}}(e_z)$$

来自 ICML 2024 [[7]](https://arxiv.org/abs/2402.17957)。

**为什么这个有意思？** 它把interaction**离散化**成几种典型mode。物理上collision和grasping本来就是qualitatively不同的process，用discrete code建模很natural。类似VQ-VAE [[8]](https://arxiv.org/abs/1711.00937)的discrete latents。

#### Method 3: Conditional Independence Testing

用 Conditional Mutual Information (CMI) 检测 $i$ 是否对 $j$ 的dynamics有信息增益：
$$\text{CMI}_{i,j} = \mathbb{E}\left[\log \frac{p_s(\mathbf{s}_{t+1}^j | \mathbf{a}_t, \mathbf{s}_t)}{p_s(\mathbf{s}_{t+1}^j | \mathbf{a}_t, \mathbf{s}_t \setminus \mathbf{s}_t^i)}\right]$$

- 分子：用全部state（包括 $i$）predict $j$ 的next state
- 分母：去掉 $i$ 的state，predict $j$ 的next state
- 比值 > threshold说明 $i$ 对 $j$ 有信息贡献，即存在interaction

来自 Causal Dynamics Learning [[9]](https://arxiv.org/abs/2205.14803)。

**三种方法实验对比**（Table A1）：

| Method | 9 objects (SHD↓) | Comp Gen 10 objects |
|---|---|---|
| Variational (Categorical) | **0.27** | **0.32** |
| Variational (Codebook) | 0.31 | 0.37 |
| CIT | 0.35 | 0.39 |
| Attention-based | 0.41 | 0.48 |

观察：
- Variational categorical 最好，object多时优势明显
- Attention-based 最差，说明直接看attention weights推interaction不靠谱
- Codebook第二，CIT在少object时还可以

### Step 3: 学reward

reward decoder $p_r(r_t | \mathbf{s}_t, \mathbf{a}_t)$，用MSE训：
$$\mathcal{L}_{\text{rew}} = \sum_t \|\hat{r}_t - r_t\|^2$$

注意reward是global的（不factorize），因为task通常涉及多object关系（"把block放进basket"涉及block和basket两个object）。

### Step 4: World model整体loss

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{pred}} + \gamma \mathcal{L}_{\text{KL}} + \eta \mathcal{L}_{\text{rew}} + \mathcal{L}_{\text{static}} + \mathcal{L}_{\text{con}} + \mathcal{L}_{\text{mask}}$$

权重 $\{\alpha, \beta, \gamma, \eta\} = \{1, 0.05, 0.1, 0.2\}$，lr $3 \times 10^{-4}$。

这个stage是**offline**的，用pre-trained DreamerV3收集的2000 episodes（Sprites用3000 random action episodes）训。Compute成本：Sprites 3小时1xA100，其他8-9小时6x4090或1xA100。

## Stage 2: 怎么用world model做policy

这是另一个核心创新。**Hierarchical policy**：

### Low-level policy $\pi^l$: 执行单个interaction

输入：当前state $\mathbf{s}_t$ + target interaction graph $G_t^g$
输出：action sequence

两种实现：

#### MPC version (CEM)

用Cross-Entropy Method [[10]](https://arxiv.org/abs/2004.13609)在action space搜索：
1. 从Gaussian $\mathcal{N}(\mu, \Sigma)$ 采样action sequences
2. 用world model forward rollout，得到predicted state $\hat{\mathbf{s}}_{t+k}$
3. 计算 $\mathcal{L} = \|\hat{\mathbf{s}}_{t+k} - \mathbf{s}_g\|^2$（$\mathbf{s}_g$ 是target state）
4. 选top elite samples，更新 $\mu, \Sigma$
5. 迭代

简单说就是"用world model当simulator，在action space里stochastic search"。

#### PPO version

policy $\pi^l(\mathbf{a}_t | \mathbf{s}_t, \mathbf{s}_g, \mathbf{u}_g)$，用PPO [[11]](https://arxiv.org/abs/1707.06347)训。
- lr $3 \times 10^{-4}$
- clip 0.1
- hidden [512, 512]
- GAE 0.95, entropy 0.1

Low-level policy在Stage 1 offline训过一遍（warm start），Stage 2 online fine-tune。

### High-level policy $\pi^h$: 选interaction sequence

policy $\pi^h(G_t^g | \mathbf{s}_t)$——给定state，选下一个target interaction graph。

Action space是所有可能的interaction graph，组合爆炸。所以**实际操作时限制每次只focus 1-2个object作为anchor**，再选其他object之一组成pair。这把action space从 $2^{N^2}$ 降到 $O(N)$。

Reward = task reward $r_{\text{task}}$ + diversity reward $r_{\text{div}}$:
$$r_{\text{div}} = \frac{1}{\sqrt{|G_{\text{visited}}|}}$$
- $|G_{\text{visited}}|$: 已经visit过的interaction graph数

直觉：鼓励探索未访问过的interaction pattern，防止policy collapse到单一mode。

### 具体怎么执行"move kettle to stove"

high-level policy输出subgoal graph序列：
1. $G_1$: arm ↔ kettle edge开启（"抓kettle"）
2. $G_2$: arm ↔ kettle保持，kettle ↔ counter edge关闭（"提起"）
3. $G_3$: 移动到stove上方
4. $G_4$: kettle ↔ stove edge开启（"放到stove上"）
5. $G_5$: arm ↔ kettle edge关闭（"松手"）

每一步low-level policy执行对应interaction。

## 实验结果人话版

### World model重建质量（LPIPS，越低越好）

Table A3:
| Env | Dreamer-V3 | TD-MPC2 | EIT | DINO-WM | FIOC |
|---|---|---|---|---|---|
| Sprites | 0.026 | 0.019 | 0.006 | 0.012 | **0.004** |
| Fetch | 0.042 | 0.039 | 0.026 | 0.009 | **0.007** |
| Kitchen | 0.102 | 0.123 | 0.096 | **0.035** | 0.038 |
| iGibson | 0.135 | 0.092 | 0.085 | **0.063** | 0.068 |
| Libero | 0.089 | 0.061 | 0.040 | 0.035 | **0.027** |

FIOC在3个env最好，2个略输DINO-WM但差距<0.005。整体comparable or better。

### Disentanglement质量

Sprites有ground-truth attributes（color, shape, position, velocity）。用linear probing评估learned representation能不能线性recover这些attributes（Fig 5a）：
- vanilla DINO对dynamic features还行，static features差
- object-centric DINO提升dynamic但损害static
- FIOC加上disentanglement module后两者都好

### Interaction graph learning（nSHD，越低越好）

Fig 6a：FIOC (variational categorical) 全面最佳。Attention-based最差，object多时差距大。Generalization gap（bar延伸部分）FIOC < Attention，说明structure learning更稳定。

### Policy learning（success rate）

Table 2:
| Setting | FIOC | Dreamer-V3 | EIT | TD-MPC2 |
|---|---|---|---|---|
| Single task avg | 0.82 | 0.72 | 0.77 | 0.78 |
| Attribute gen | **0.82** | 0.70 | 0.78 | 0.76 |
| Compositional gen | **0.78** | 0.70 | 0.74 | 0.71 |
| Skill gen | **0.77** | 0.63 | 0.69 | 0.64 |

**Single task差距不大**，但**generalization tasks FIOC显著领先**，尤其skill composition（训练push+switch，测试2-push+3-switch组合）。

### Object count generalization（Table A6）

训3 objects，测6/8 objects:
| Objects | FIOC | Dreamer-V3 | EIT | TD-MPC2 |
|---|---|---|---|---|
| 6 | 0.81 | 0.54 | 0.77 | 0.62 |
| 8 | 0.70 | 0.44 | 0.62 | 0.53 |

Dreamer-V3/TD-MPC2这种monolithic latent在object数量shift时崩盘，FIOC因为weight sharing还好。

### Ablation最关键的发现

Table 3:
| Ablation | Single Task | Comp Gen |
|---|---|---|
| FIOC full | 0.81 | 0.70 |
| **w/o interaction modeling** | 0.63 (↓0.18) | 0.52 (↓0.18) |
| **w/o hierarchical policy** | 0.58 (↓0.23) | 0.42 (↓0.28) |
| w/o factorization | 0.77 (↓0.04) | 0.64 (↓0.06) |
| w/o pre-trained $\pi^l$ | 0.69 (↓0.12) | 0.59 (↓0.11) |
| w/o diversity | 0.62 (↓0.19) | 0.50 (↓0.20) |

**最关键两个**：interaction modeling + hierarchical policy。去掉任一个都掉0.2+。说明光有object-centric representation不够，必须**explicitly model interaction + 用hierarchical结构exploit它**。

### Baselines也用DINO features（Table A5）

让Dreamer-V3和TD-MPC2也用DINO embedding作为input，FIOC仍然赢：
- Kitchen: FIOC 0.82 vs Dreamer-V3(DINO) 0.77
- Libero: FIOC 0.81 vs Dreamer-V3(DINO) 0.69

**说明光有strong visual features不够，object-centric structure + interaction modeling才是关键**。

## 我的几个intuition

### 1. 为什么factorization帮助generalization

假设你训过"push red cube"和"grasp blue sphere"。Test时让你"push blue sphere"。
- 没factorization：latent是red+cube+push和blue+sphere+grasp混在一起，新组合没见过
- FIOC factorization：static（red, cube, blue, sphere）和dynamic（push, grasp）分开，新组合是已知static + 已知dynamic的rearrangement，transition model直接reuse

这就是compositional generalization的本质。

### 2. 为什么interaction graph要explicit

Implicit（让attention自己学）的问题：attention weights在不同setup下不稳定，object多了attention分散，sparsity没法enforce。

Explicit graph $G_t$ 的好处：
- 可以加sparsity prior
- 可以discretize成codebook
- 可以做conditional independence testing验证
- policy可以直接在graph上planning

### 3. 为什么hierarchical policy关键

Long-horizon任务如"开microwave → 移kettle → 开stove → 开light"，直接flat policy要jointly optimize 4个sub-task，exploration困难。

Hierarchical decomposition：
- High-level: 在interaction graph space planning（小空间）
- Low-level: 每个interaction单独optimize（short-horizon，easy）

这是经典HRL的value，但FIOC用interaction作为sub-goal比用latent goal或random skill更有结构。

### 4. 和JEPA / LeCun思路的连接

JEPA [[12]](https://arxiv.org/abs/2301.08243) 系列强调在latent space预测而非pixel space。FIOC其实是个object-centric JEPA：
- 在DINO latent上predict + reconstruct
- 加interaction graph作为latent structure
- 加static/dynamic factorization

V-JEPA2 [[13]](https://arxiv.org/abs/2506.09985) 和 Cosmos [[14]](https://arxiv.org/abs/2501.03575) 都在往physical AI world model方向走，FIOC提供object-centric inductive bias。

### 5. 和VLA foundation model的关系

OpenVLA [[15]](https://arxiv.org/abs/2406.09246), π0 [[16]](https://arxiv.org/abs/2410.24164), Octo [[17]](https://arxiv.org/abs/2405.12213)这些VLA model是end-to-end的，从(text, image)直接到action。FIOC是complementary的——它学structured world model，可以做planning和long-horizon reasoning。

未来可能的combination：VLA做low-level control，FIOC-style world model做high-level planning和compositional generalization。

### 6. Limitations和我会怎么改进

作者承认的：
- 依赖Slot Attention做object discovery，复杂场景不完美
- Interaction只generalize到seen categories
- 没在real robot上验证

我会加：
- Occlusion和slot collapse问题（object被遮住slot可能消失）
- Higher-order interactions（3-body如stacking A on B on C）
- Long-horizon时GRU可能不稳，换Mamba [[18]](https://arxiv.org/abs/2312.00752) 或 Transformer
- Reward decoder也可以factorize成per-object sub-reward
- Scale到更多object（20+）
- 用DINOv2-large或ViT-giant试试

### 7. 这篇paper给RL community的信号

**Inductive bias matters even in the era of large pre-trained models**。DreamerV3 + DINO features不够，加上object-centric structure和interaction graph就显著提升。Foundation model提供的features需要正确的structure来organize for decision-making。Structured world model在VLA时代可能不是过时，而是更有价值。

## 总结一句

FIOC-WM告诉我们：**让agent像人一样理解object和interaction，比让它从pixel硬学everything更sample efficient、更generalizable**。技术上通过两层factorization（object-level + static/dynamic attribute-level）+ explicit interaction graph + hierarchical policy实现。Long-horizon任务变成interaction graph上的planning问题，compositional generalization自然emerge。

参考链接汇总：
- [[FIOC-WM paper]](https://arxiv.org/abs/2506.23181) (NeurIPS 2025 submission)
- [[DINOv2]](https://arxiv.org/abs/2304.07193)
- [[Slot Attention]](https://arxiv.org/abs/2006.15055)
- [[VideoSAUR]](https://arxiv.org/abs/2310.18643)
- [[DreamerV3]](https://arxiv.org/abs/2301.04104)
- [[TD-MPC2]](https://arxiv.org/abs/2310.16828)
- [[DINO-WM]](https://arxiv.org/abs/2411.04983)
- [[EIT]](https://openreview.net/forum?id=uDxeSZ1wdI)
- [[SKILD]](https://arxiv.org/abs/2410.10966)
- [[NRI]](https://arxiv.org/abs/1802.08652)
- [[Amortized Causal Discovery]](https://arxiv.org/abs/2206.08163)
- [[Causal Dynamics Learning]](https://arxiv.org/abs/2205.14803)
- [[V-JEPA 2]](https://arxiv.org/abs/2506.09985)
- [[Cosmos]](https://arxiv.org/abs/2501.03575)
- [[OpenVLA]](https://arxiv.org/abs/2406.09246)
- [[π0]](https://arxiv.org/abs/2410.24164)
- [[Gumbel-Softmax]](https://arxiv.org/abs/1611.01144)
- [[PPO]](https://arxiv.org/abs/1707.06347)
- [[VQ-VAE]](https://arxiv.org/abs/1711.00937)
- [[InfoNCE / CPC]](https://arxiv.org/abs/1807.03748)
- [[LIBERO]](https://arxiv.org/abs/2306.03310)
- [[Franka Kitchen]](https://arxiv.org/abs/1910.11972)
- [[iGibson]](https://arxiv.org/abs/2010.01820)
- [[Sprites/COBRA]](https://arxiv.org/abs/1905.09275)
- [[Mamba]](https://arxiv.org/abs/2312.00752)

---

# FIOC-WM: Factored Interactive Object-Centric World Model 深度讲解

Karpathy 你好，这篇paper让我非常兴奋，因为它把object-centric representation learning, causal structure discovery, hierarchical RL, 还有pre-trained vision encoders四条线缝到一起，提出了FIOC-WM。下面我从motivation、formulation、stage 1 (offline world model)、stage 2 (online policy)、experiments、ablation、以及一些personal联想角度，尽量把intuition讲透。

## 1. Motivation: 为什么 explicit interactions 很关键

现有object-centric RL通常只把state按object factorize，但把object之间的interactions（碰撞、stacking、friction、containment）压在transition network里，让神经网络自己figure out。问题在于：当agent要做 long-horizon planning 时，这种implicit representation不利于**compositional generalization**——比如训练时见过"拿起kettle放到stove"，没见过"拿switch推到盒子"上的组合，就要重学一遍。

FIOC-WM的核心想法：**interactions本身就是sub-skills**。把world model分解成 (a) 每个object的static/dynamic属性，以及 (b) object之间的interaction graph $G_t$，long-horizon任务就变成了interaction graph上的path planning问题。这个想法明显受 SKILD [[1]](https://arxiv.org/abs/2410.10966) 启发，但FIOC-WM把它推到pixel层面并joint learning world model，比SKILD的state-based版本更通用。

参考[[2]](https://arxiv.org/abs/2411.04983) DINO-WM证明了"用pre-trained DINO特征 + world model on latent + MPC"就已经能在像素任务上做zero-shot planning，FIOC-WM在这条思路上加了object-centric结构。

## 2. FIOC-POMDP: 形式化设置

环境被建模为 Partially Observable MDP，state分解为N个object：
$$\mathbf{s}_t^i = \{\mathbf{d}_t^i, \mathbf{c}^i\}$$
- $\mathbf{s}_t^i$: object $i$ 在 timestep $t$ 的完整state
- $\mathbf{d}_t^i$: **dynamic variables**（时变），如position、velocity、orientation
- $\mathbf{c}^i$: **constant properties**（时不变），如color、mass、friction coefficient
- 上标 $i \in \{1, ..., N\}$ 索引object，下标 $t$ 索引时间

核心transition equation：
$$\mathbf{d}_{t+1}^i = f_{\text{self}}(\mathbf{d}_t^i, \mathbf{c}^i, \mathbf{a}_t, \epsilon_t) + \sum_{j \in \mathcal{N}_t(i)} f_{\text{inter}}(\mathbf{d}_t^i, \mathbf{d}_t^j, \mathbf{c}^j, \delta_t)$$

这个公式非常重要，需要逐项拆解：
- $f_{\text{self}}$: object **独自**演化函数（不和其他object交互时）。输入是它自己的dynamic state、它的constant属性、action $\mathbf{a}_t$ 和噪声 $\epsilon_t$
- $f_{\text{inter}}$: **pairwise interaction**函数。注意它只对 $\mathbf{d}$ (dynamic) 写，意味着interaction只改变dynamic attributes，不变static attributes（color不会因为碰撞而改变）
- $\mathcal{N}_t(i)$: time $t$ 与 object $i$ 交互的object集合——这个就是**稀疏interaction graph** $G_t$ 的邻域
- $\epsilon_t, \delta_t$: 两个独立noise variables，分别模型self-transition和interaction的stochasticity

这个formulation的inductive bias是强假设，但很reasonable：物体相撞只影响彼此的速度位置，不会影响mass、color；friction作为一个static property通过 $f_{\text{self}}$ 影响减速过程，但不会被碰撞改变。这本质上对应 Koopman operator / Hamiltonian mechanics 的minimal sufficient factorization思想。

观察模型：
$$\mathbf{o}_t^i = g(\mathbf{s}_t^i, \epsilon_t^i)$$
- $\mathbf{o}_t^i$: object $i$ 的observation（图像patch或feature）
- $g$: observation function
- $\epsilon_t^i$: i.i.d. observation noise

Reward $r_t = h(\mathbf{s}_t, \mathbf{a}_t)$ 是global function（不能factorize，因为task通常涉及多object关系）。

这个graphical model对应论文Fig. 2的plate diagram，可以类比Dynamic Bayesian Network with time-varying structure。

## 3. Stage 1: Offline World Model Learning

整个pipeline的工程化非常值得仔细看。

### 3.1 Vision encoder → slot attention → factored observation

raw pixel $\mathbf{o}_t$ → pre-trained encoder → embedding → slot attention → $\{\hat{\mathbf{o}}^1, ..., \hat{\mathbf{o}}^N\}$

- **DINO-v2 ViT-Base** [[3]](https://arxiv.org/abs/2304.07193) 或 **R3M ResNet-50** [[4]](https://arxiv.org/abs/2203.12601) 作为frozen visual backbone，输出feature dim 768
- **Slot Attention** [[5]](https://arxiv.org/abs/2006.15055)（Locatello et al.）做object discovery，slot数量 = 真实物体数 + 2（slack slots）。初始化时random slots，后续帧用前一帧的slots + GRU predictor更新——这种做法来自 VideoSAUR [[6]](https://arxiv.org/abs/2310.18643) 的temporal warm-start trick
- Slot dim在不同环境不同：SpritesWorld 32, Fetch 64, 其他 128

### 3.2 VAE: 从 factored observation 到 factored state

对每个slot独立训练VAE：
- encoder: $q_\phi(\mathbf{s}^i | \hat{\mathbf{o}}^i)$
- decoder: $p_\psi(\hat{\mathbf{o}}^i | \mathbf{s}^i)$
- 所有slot共享参数（weight sharing 是 object-centric generalization 的关键）

reconstruction loss:
$$\mathcal{L}_{\text{recon}} = \sum_{t=1}^T \|\hat{\mathbf{o}}_t - \hat{\mathbf{o}}_t^{\text{decoded}}\|^2$$

prediction loss（next step prediction）:
$$\mathcal{L}_{\text{pred}} = \sum_{t=1}^T \|\hat{\mathbf{o}}_{t+1} - \hat{\mathbf{o}}_{t+1}^{\text{decoded}}\|^2$$

两者都用MSE，区别在于recon用的是encoder+decoder，pred用transition model + decoder。两个loss共同确保latent既reconstruct当前又能predict下一步。

### 3.3 Static/Dynamic factorization

这是论文最巧妙的部分。latent $\mathbf{s}$ 被两个head进一步分解：
- $f_c(\mathbf{s})$: static feature extractor（MLP）
- $f_d(\mathbf{s})$: dynamic feature extractor（GRU-based，建模时序）

**Static consistency loss**强制 $f_c$ 输出时序不变：
$$\mathcal{L}_{\text{static}} = \sum_{t=1}^{T-1} \sum_{i=1}^N |f_c(\mathbf{s}_{t+1}^i) - f_c(\mathbf{s}_t^i)|^2$$
- $T$: trajectory length
- $N$: object数量
- 直觉：color、shape不该随时间变化，所以让 $f_c$ 输出对相邻帧差异penalty

**Contrastive loss** 防止不同object编码成同一个static：
$$\mathcal{L}_{\text{con}} = -\sum_{t=1}^{T-1} \sum_{i=1}^N \log \frac{g(f_c(\mathbf{s}_t^i), f_c(\mathbf{s}_{t'}^i))}{g(f_c(\mathbf{s}_t^i), f_c(\mathbf{s}_{t'}^i)) + \sum_{j \in \mathcal{N}} g(f_c(\mathbf{s}_t^i), f_c(\mathbf{s}_{t'}^j))}$$
- $t'$: 同一object不同时刻（positive pair）
- $\mathcal{N}$: 同一scene其他object $j \neq i$（negative set）
- $g$: cosine similarity

这个是 InfoNCE [[7]](https://arxiv.org/abs/1807.03748) 的形式，目的让同一object跨时间static feature接近，不同object static feature远离。

### 3.4 KL between posterior 和 prior

posterior（观测信息）：
$$q_\phi(\mathbf{s}_t | \hat{\mathbf{o}}_t, \mathbf{h}_t), \quad \mathbf{h}_t = \text{GRU}(\mathbf{s}_{t-1}, \mathbf{h}_{t-1})$$

prior（基于transition model）：
$$p_s(\mathbf{d}_t | \mathbf{d}_{t-1}, \mathbf{a}_{t-1}, G_t) = \prod_{i=1}^N p_s(\mathbf{d}_t^i | \mathbf{d}_{t-1}, \mathbf{a}_{t-1}, G_t)$$
- $G_t$: 当前时刻interaction graph（binary adjacency matrix $N \times N$）
- prior是factored的，每个object的dynamic独立，但条件在graph上——这其实是一种 amortized variational inference with structured prior

KL loss:
$$\mathcal{L}_{\text{KL}} = \sum_{t=1}^T \text{KL}\left(q_\phi(\mathbf{s}_t | \hat{\mathbf{o}}_t, \mathbf{h}_t) \| p_s(\mathbf{s}_t | \mathbf{s}_{t-1}^s, \mathbf{a}_{t-1}, G_t)\right)$$

这是RSSM (Recurrent State-Space Model) [[8]](https://arxiv.org/abs/1911.05422) 的标准做法，但FIOC-WM在prior上加了interaction graph conditioning。

### 3.5 Reward decoder

$$\mathcal{L}_{\text{rew}} = \sum_{t=1}^T \|\hat{r}_t - r_t\|^2$$

reward head $p_r(r_t | \mathbf{s}_t, \mathbf{a}_t)$，保证latent可以predict reward。

### 3.6 Interaction graph learning

这是论文技术含量最高的部分，给了**三种方法**：

**(i) Variational Masks with Categorical Distribution**（来自 Amortized Causal Discovery [[9]](https://arxiv.org/abs/2206.08163)）

对每对object $(i, j)$ 用GRU encoder：
$$\mathbf{u}_t^{ij} = f_{\text{enc}, \phi_u}(\mathbf{s}_t^i, \mathbf{s}_t^j)$$
- $\mathbf{u}_t^{ij}$: pair embedding

然后Gumbel-Softmax采样：
$$G^{i,j} \sim \text{Softmax}(\mathbf{u}_{ij} + g/\tau)$$
- $g$: Gumbel(0, 1) noise
- $\tau$: temperature，anneal from high to low
- $G^{i,j} \in [0, 1]$: edge $(i, j)$ 的存在概率（approximated categorical）

ELBO:
$$\mathcal{L}_{\text{mask}} = \mathbb{E}_{q_{\phi_u}(G | \mathbf{s})}[\log p_{\theta_u}(\mathbf{s} | G)] - \text{KL}[q_{\phi_u}(G | \mathbf{s}) \| p(G)]$$
- $p(G)$: graph prior（sparsity-inducing，如Bernoulli with low prob）
- $p_{\theta_u}(\mathbf{s} | G)$: graph-conditioned dynamics likelihood

**(ii) Latent Codebook**（来自 ICML 2024 [[10]](https://arxiv.org/abs/2402.17957)）

维护一组codebook prototypes $\mathbf{e} = \{e_1, ..., e_k\}$:
$$e = e_z, \quad z = \arg\min_{j \in [k]} \|\mathbf{u} - \mathbf{e}_j\|_2$$
- $k$: 不同interaction pattern数（Sprites 16, Fetch 8, 其他 10）
- $z$: 选中的codebook index

quantization后decode回adjacency matrix:
$$G \sim g_{\text{dec}}(\mathbf{e}_z)$$

直觉：把"interaction类型"离散成几种典型模式（碰撞、抓取、滑动...）。类似VQ-VAE的离散latents。

**(iii) Conditional Independence Testing**（来自 [[11]](https://arxiv.org/abs/2205.14803)）

用 Conditional Mutual Information (CMI) 检测是否存在interaction：
$$\text{CMI}_{i,j} = \mathbb{E}_{\mathbf{s}_t, \mathbf{a}_t, \mathbf{s}_{t+1}^j}\left[\log \frac{p_s(\mathbf{s}_{t+1}^j | \mathbf{a}_t, \mathbf{s}_t)}{p_s(\mathbf{s}_{t+1}^j | \{\mathbf{a}_t, \mathbf{s}_t \setminus \mathbf{s}_t^i\})}\right]$$
- 分子: 用object $i$ 的state作为条件，predict $j$ 的next state
- 分母: 去掉object $i$ 的state，predict $j$ 的next state
- 比值大意味着 $i$ 对 $j$ 的prediction有信息增益，即存在interaction

阈值threshold: Sprites 0.02, Fetch 0.15, 其他 0.05。

**三种方法对比**（Table A1）：
- Variational (Categorical) 整体最佳，特别是object数量多时（9 objects: 0.27 vs Attention-based 0.41）
- Codebook第二
- CIT适合少量object，多object时退化
- Attention-based最差——说明光看attention weights推断interaction不可靠

## 4. Stage 2: Online Hierarchical Policy Learning

### 4.1 Low-level interaction policy $\pi^l$

给定target interaction graph $G_t^g$，用MPC或PPO生成actions。

**MPC版本**（CEM, Cross-Entropy Method [[12]](https://arxiv.org/abs/2004.13609)）:

目标：找到action sequence $\mathbf{a}_t, ..., \mathbf{a}_{t+k-1}$ 使transition从 $t$ 到 $t+k$ 的predicted state $\hat{\mathbf{s}}_{t+k}$ 接近target $\mathbf{s}_g$：
$$\mathcal{L}_{\text{MPC}} = \|\mathbf{s}_{t+k} - \mathbf{s}_g\|_2^2$$
- $\mathbf{s}_g$: 通过transition model从target interaction graph $G_t^g$ 反推的目标state
- $k$: planning horizon

CEM迭代：从Gaussian采样action sequences → forward rollout → 选top精英 → 更新Gaussian mean/cov。

**PPO版本** [[13]](https://arxiv.org/abs/1707.06347):

policy $\pi^l(\mathbf{a}_t | \mathbf{s}_t, \mathbf{s}_g, \mathbf{u}_g)$，给goal信息，PPO训。
- lr: $3 \times 10^{-4}$
- clip ratio: 0.1
- hidden: [256, 256] (Fetch) 或 [512, 512]
- GAE: 0.95
- entropy: 0.1

Low-level policy 在 Stage 1 offline 训过一遍（用pre-trained Dreamer-V3 收集的2000 episodes），Stage 2 在线fine-tune。

### 4.2 High-level policy $\pi^h$

policy: $\pi^h(G_t^g | \mathbf{s}_t)$
- action space = 全部可能interaction graph
- combinatorial explosion → 用anchor object策略：每次只focus 1-2个object作为candidate，选其他object之一形成pair

Diversity reward防止陷入局部interaction：
$$r_{\text{div}} = \frac{1}{\sqrt{|G_{\text{visited}}|}}$$
- $|G_{\text{visited}}|$: 已经visit过的interaction graph数量
- intuition: 鼓励探索未访问过的interaction模式

PPO训练，task reward $r_{\text{task}}$ + $r_{\text{div}}$。

这个设计明显借鉴 SKILD [[1]](https://arxiv.org/abs/2410.10966)，但加task reward来直接optimize task performance。

### 4.3 实际操作的graph transition

"move kettle from counter to stove"分解为：
1. 先建立 arm ↔ kettle 的interaction edge
2. kettle 离开 counter
3. kettle 接近 stove
4. 建立 kettle ↔ stove 的interaction edge
5. 释放 arm ↔ kettle

high-level policy就是在选这些subgoal graphs。

## 5. Experiments 全面分析

### 5.1 Environments

5个benchmark:
- **SpritesWorld** [[14]](https://arxiv.org/abs/1905.09275): synthetic sprites，有ground-truth attributes（color, shape, position, velocity），用于disentanglement评估
- **OpenAI Gym Fetch** [[15]](https://arxiv.org/abs/1606.01540): Fetch arm 推立方块、按switch
- **Franka Kitchen** [[16]](https://arxiv.org/abs/1910.11972): 7-DoF Franka 操作厨房（microwave、kettle、stove、light）
- **iGibson** [[17]](https://arxiv.org/abs/2010.01820): Fetch robot 真实household，peach任务
- **LIBERO** [[18]](https://arxiv.org/abs/2306.03310): tabletop manipulation

### 5.2 World Model reconstruction (LPIPS)

Table A3结果（越低越好）:

| Environment | Dreamer-V3 | TD-MPC2 | EIT | DINO-WM | **FIOC** |
|---|---|---|---|---|---|
| Sprites | 0.026 | 0.019 | 0.006 | 0.012 | **0.004** |
| Fetch | 0.042 | 0.039 | 0.026 | 0.009 | **0.007** |
| Kitchen | 0.102 | 0.123 | 0.096 | **0.035** | 0.038 |
| iGibson | 0.135 | 0.092 | 0.085 | **0.063** | 0.068 |
| Libero | 0.089 | 0.061 | 0.040 | 0.035 | **0.027** |

观察：
- FIOC在Sprites/Fetch/Libero上最好
- DINO-WM在Kitchen/iGibson上略好
- FIOC从未大幅落后，差距 < 0.005

### 5.3 Disentanglement (Sprites)

Linear probing MSE against ground-truth attributes（Fig 5a）:
- vanilla DINO对dynamic features（position/velocity）还行
- object-centric DINO 提升 dynamic features 但损害 static features（color/shape）
- FIOC加上disentanglement后两者都好

### 5.4 Interaction learning (nSHD)

Fig 6a，对Sprites用normalized Structured Hamming Distance评估recovered interaction graph：
- 单一任务下FIOC (variational categorical)最佳
- generalization gap (bar的延伸部分) FIOC < Attention-based，说明generalize到新组合时structure learning更稳定
- object数量越多FIOC优势越大（9 objects差距 0.20+）

### 5.5 Policy learning (Table 2, A2)

| Setting | Task | FIOC | Dreamer-V3 | EIT | TD-MPC2 |
|---|---|---|---|---|---|
| Single | Fetch Task1 | 0.95 | **0.98** | 0.93 | 0.97 |
| Single | Kitchen Task1 | 0.82 | 0.75 | 0.69 | **0.83** |
| Single | iGibson Task1 | **0.76** | 0.69 | 0.74 | 0.72 |
| Single | Libero Task1 | **0.81** | 0.65 | 0.78 | 0.76 |
| Attri Gen | Push&Switch | 0.91 | 0.90 | 0.92 | **0.95** |
| Attri Gen | iGibson | **0.79** | 0.62 | 0.70 | 0.65 |
| Comp Gen | Libero | **0.70** | 0.58 | 0.65 | 0.63 |
| Skill Gen | Push&Switch | **0.81** | 0.66 | 0.73 | 0.65 |
| Skill Gen | Kitchen | **0.73** | 0.59 | 0.65 | 0.62 |

关键发现：
- Single-task FIOC略胜，但差距不大
- **Generalization tasks FIOC显著领先**，特别是skill composition（把push+switch组合成2-push+3-switch）

### 5.6 Generalization to more objects (Table A6)

训3 objects, 测6/8 objects:
- 6 objects: FIOC 0.81, Dreamer-V3 0.54, EIT 0.77, TD-MPC2 0.62
- 8 objects: FIOC 0.70, Dreamer-V3 0.44, EIT 0.62, TD-MPC2 0.53

这验证了 object-centric weight sharing 的 transferability。Dreamer-V3/TD-MPC2这种monolithic latent在 object数量shift时崩溃。

### 5.7 Ablation studies (Table 3)

| Ablation | Single Task | Comp Gen | 影响 |
|---|---|---|---|
| FIOC full | 0.81 | 0.70 | baseline |
| w/o Factorization | 0.77 (↓0.04) | 0.64 (↓0.06) | 中等影响 |
| **w/o Interaction** | 0.63 (↓0.18) | 0.52 (↓0.18) | **最关键** |
| w/ random actions | 0.64 (↓0.17) | 0.48 (↓0.22) | 重要 |
| **w/o hierarchical policy** | 0.58 (↓0.23) | 0.42 (↓0.28) | **最关键** |
| w/o pre-trained $\pi^l$ | 0.69 (↓0.12) | 0.59 (↓0.11) | 重要 |
| w/o diversity | 0.62 (↓0.19) | 0.50 (↓0.20) | 重要 |

**Key takeaways**:
- Interaction modeling 和 hierarchical policy 是必须的——这两者去掉都会大幅掉点
- Static/dynamic factorization贡献相对小，但comp gen差距说明还是有价值
- Pre-training low-level policy提供warm-start，避免online时low-level需要冷启动
- Diversity reward防止high-level policy collapse到单一interaction

### 5.8 Online tuning (Table A4)

默认不用online world model tuning，已经足够好。Online tuning对单任务有小幅提升但对某些generalization setting有下降，可能因为online data分布shift。

### 5.9 Baselines加DINO特征 (Table A5)

让Dreamer-V3和TD-MPC2也用DINO embeddings，但仍输给FIOC：
- Kitchen: FIOC 0.82 vs Dreamer-V3(DINO) 0.77
- iGibson: FIOC 0.76 vs Dreamer-V3(DINO) 0.71
- Libero: FIOC 0.81 vs Dreamer-V3(DINO) 0.69

说明光有strong visual features不够，object-centric结构 + interaction modeling才是关键。

## 6. Architecture 细节（Appendix E）

### 6.1 Vision encoder
- DINO-v2 ViT-Base，patch 16, feature dim 768
- R3M ResNet-50
- Image resize: Sprites 64, 其他 224
- Gradient clip 0.05

### 6.2 Slot Attention
- 3 iterations of clustering
- Slot dim: Sprites 32, Fetch 64, 其他 128
- 每帧用前帧slots + GRU预测初始化

### 6.3 Latent dimensions ($\mathbf{s}^s, \mathbf{s}^c$)
- Sprites: 8, 6
- Fetch: 10, 8  
- 其他: 16, 12

### 6.4 Transition model MLP
| Environment | MLP Layers | GRU Layers |
|---|---|---|
| Sprites | 2, hidden 32 | 3, hidden 128 |
| Fetch | 2, hidden 64 | 3, hidden 256 |
| iGibson/Libero/Kitchen | 3, hidden 128 | 3, hidden 256 |

### 6.5 Compute
- Sprites: 3h on 1xA100
- Fetch: 8h on 6x4090
- iGibson: 9h on 6x4090
- Libero: 8h on 1xA100
- Kitchen: 6h on 1xA100

不算大，因为用了frozen DINO。

### 6.6 Loss weights
$\{\alpha, \beta, \gamma, \eta\} = \{1, 0.05, 0.1, 0.2\}$，lr $3 \times 10^{-4}$。

### 6.7 数据
- Sprites: 3000 episodes random actions
- 其他: 2000 episodes from Dreamer-V3 pre-trained policy

## 7. 与相关工作的连接

### 7.1 Object-centric RL 谱系

- **Slot Attention** [[5]](https://arxiv.org/abs/2006.15055) → object discovery from pixels
- **Object-centric RL** [[19]](https://arxiv.org/abs/2104.03620) Zadaianchuk et al. → goal-conditioned hierarchical with object goals
- **CSWM** (Contrastive Structured World Models) [[20]](https://arxiv.org/abs/1911.12247) Kipf et al. → 用contrastive学object interactions
- **EIT** (Entity-centric RL) [[21]](https://arxiv.org/abs/2403.01823) Haramati et al. → model-free，直接from pixels学object interactions做compositional generalization，是FIOC的主要baseline之一
- **GATSBI** [[22]](https://arxiv.org/abs/2103.15494) → agent-centric spatio-temporal interaction
- **FOCUS** [[23]](https://arxiv.org/abs/2501.14884) → robotic manipulation with object-centric WM

### 7.2 Hierarchical RL 谱系

- **Options framework** [[24]](https://arxiv.org/abs/9905110) Sutton et al. → 经典option-critic
- **SKILD** [[1]](https://arxiv.org/abs/2410.10966) Wang et al. → 用conditional independence testing发现skills guided by factor interactions，FIOC直接借鉴其high-level policy结构
- **Granger-causal skill discovery** [[25]](https://arxiv.org/abs/2306.13781) Chuck et al. → Granger因果找sub-skills
- **NCF** Null Counterfactual Factor Interactions [[26]](https://arxiv.org/abs/2505.03172) → 最新工作，counterfactual推理factor interactions

### 7.3 Causal discovery 谱系

- **NRI** Neural Relational Inference [[27]](https://arxiv.org/abs/1802.08652) Kipf et al. → VAE推断interaction graph，FIOC的variational mask方法基于此
- **ACD** Amortized Causal Discovery [[9]](https://arxiv.org/abs/2206.08163) Löwe et al. → 从time series学causal graph，FIOC直接用其框架
- **CIT** Conditional Independence Testing [[11]](https://arxiv.org/abs/2205.14803) Wang et al. → ICML 2022，Causal Dynamics Learning
- **NOTEARS** [[28]](https://arxiv.org/abs/1803.01422) → continuous optimization for DAG
- **DiffAN** Lippe et al. [[29]](https://arxiv.org/abs/2107.10483) → efficient causal discovery

### 7.4 Pre-trained visual representation

- **DINO / DINOv2** [[3]](https://arxiv.org/abs/2304.07193) → self-distillation ViT
- **R3M** [[4]](https://arxiv.org/abs/2203.12601) Nair et al. → universal visual representation for manipulation
- **DINO-WM** [[2]](https://arxiv.org/abs/2411.04983) Zhou et al. → 第一个用DINO features做world model的，FIOC基于此加结构
- **SpawnNet** [[30]](https://arxiv.org/abs/2310.06852) → generalizable visuomotor skills
- **V-JEPA 2** [[31]](https://arxiv.org/abs/2506.09985) → self-supervised video models enable planning
- **Dynamo** [[32]](https://arxiv.org/abs/2411.04932) → in-domain dynamics pretraining

### 7.5 World models 谱系

- **World Models** Ha & Schmidhuber [[33]](https://arxiv.org/abs/1807.05061) → 经典RNN+VAE+Controller
- **PlaNet** [[34]](https://arxiv.org/abs/1811.04551) → RSSM with MPC
- **Dreamer** [[35]](https://arxiv.org/abs/1912.01603), **DreamerV2** [[36]](https://arxiv.org/abs/2010.02193), **DreamerV3** [[37]](https://arxiv.org/abs/2301.04104) Hafner et al. → 现代world model标准
- **TD-MPC** [[38]](https://arxiv.org/abs/2203.09116), **TD-MPC2** [[39]](https://arxiv.org/abs/2310.16828) Hansen et al. → 用Q-learning + MPC，连续控制SOTA
- **RoboDreamer** [[40]](https://arxiv.org/abs/2404.12377) → compositional WM for robot imagination
- **Cosmos** [[41]](https://arxiv.org/abs/2501.03575) NVIDIA → physical AI foundation
- **V-JEPA 2** [[31]](https://arxiv.org/abs/2506.09985) Meta → video world models
- **Navigation World Models** [[42]](https://arxiv.org/abs/2412.03572) Bar et al. → LeCun组

### 7.6 VLA & foundation models (附录提到)

- **OpenVLA** [[43]](https://arxiv.org/abs/2406.09246)
- **π0** [[44]](https://arxiv.org/abs/2410.24164), **π0.5** [[45]](https://arxiv.org/abs/2504.16054) Physical Intelligence
- **Octo** [[46]](https://arxiv.org/abs/2405.12213) Open source generalist

作者在Limitations里提到下一步是extend到real robots，potentially with这些VLA models作为backbone。

## 8. 我的intuition / 个人联想

### 8.1 为什么两层factorization工作？

FIOC的双层结构对应**信息论上的minimal sufficient statistic**：dynamic attributes是控制决策真正关心的，static attributes提供context（如friction决定how hard to push）。把它们factorize后：
- transition model只需要学 $\mathbf{d}_{t+1}^i$，维度小
- interaction只modulate dynamic部分，graph稀疏
- 不同场景下static change不影响dynamic的dynamics规律（这就是compositional generalization的本质）

这和 [[47]](https://arxiv.org/abs/2307.07487) Energy-based HMM with structure priors、[[48]](https://arxiv.org/abs/2202.06456) NeSyCoCo neurosymbolic grounding思路类似。

### 8.2 与 Slot Attention + DreamerV3 的差异

如果把Slot Attention和DreamerV3简单组合，问题是 DreamerV3 的RSSM没有 object-centric structure prior，所有slot latent在transition里完全cross-attend。FIOC用interaction graph显式gate信息流，使transition结构化。

这和 **Graph Dreamer** / **GNN-based WM** 的思路类似，但FIOC更轻——dynamic transition function $f_{\text{inter}}$ 是per-pair的MLP，不需要GNN message passing。

### 8.3 与 Neurosymbolic / LeCun JEPA 的关系

JEPA系列（I-JEPA, V-JEPA, V-JEPA2）强调**predict in latent space**而不是pixel space。FIOC其实是object-centric JEPA：在DINO latent上做predict + reconstruct，加interaction graph作为latent structure。

Yann LeCun 在 Cosmos [[41]](https://arxiv.org/abs/2501.03575) 和 V-JEPA2 [[31]](https://arxiv.org/abs/2506.09985) 都强调 world model for physical AI 是未来方向，FIOC正好提供object-centric inductive bias。

### 8.4 用 codebook 表征 interaction types 的哲学

我非常喜欢codebook版本的设计。它暗示interactions可以被**discretize**成几种prototype：
- 碰撞 (collision)
- 抓取 (grasping)
- 滑动 (sliding)
- contain (containment)
- stack (stacking)

这种discrete codebook相当于**low-level physics primitives**。和 VQ-VAE [[49]](https://arxiv.org/abs/1711.00937) 思路一致，也和 SlotDiffusion [[50]](https://arxiv.org/abs/2305.11281) 的离散slot representation类似。

### 8.5 Long-horizon planning via interaction graphs

把long-horizon task变成graph上的search，这和 **Task and Motion Planning (TAMP)** [[51]](https://arxiv.org/abs/2010.01083) 经典AI思路呼应。但FIOC是learned的，不需要predefined PDDL operators。这或许是foundation model for planning的一个可行路径。

### 8.6 Limitations 和 未来方向

作者承认：
1. 依赖pre-trained object-centric model (Slot Attention在复杂场景object discovery不完美)
2. interaction models 主要 generalize to seen categories
3. 没在real robot上验证

我会补充几点：
- 没考虑occlusion导致slot collapse的问题
- Dynamic feature 用GRU建模，对very long horizon可能不稳，可换 Mamba [[52]](https://arxiv.org/abs/2312.00752) 或 Transformer [[53]](https://arxiv.org/abs/1706.03762)
- Reward decoder是global的，是否可分per-object sub-reward？
- 3 objects → 8 objects generalization但更多呢？比如训练3-5个object，迁移到20+个object？
- 用更大DINOv2-large或ViT-giant会更好吗？
- Interactions目前只考虑pairwise，没考虑higher-order interactions (3-body)，复杂stacking场景可能不够

### 8.7 对 RL 社区的启示

我觉得这篇paper的最重要的takeaway不是FIOC本身，而是**inductive bias matters even in the era of large pre-trained models**。DreamerV3 + DINO features不够好，加上object-centric structure和interaction graph就显著提升generalization。这给community一个信号：在VLA时代，structured world model可能不是过时，而是正变得更有价值，因为foundation model提供的features需要正确的structure来organize for decision-making。

## 9. 总结

FIOC-WM在四个维度上做了组合创新：

1. **Representation**: 两层factorization (object + attribute static/dynamic)
2. **Dynamics**: factored transition with explicit interaction graph $G_t$
3. **Discovery**: variational mask / codebook / CIT 三种方式学interaction graph
4. **Policy**: hierarchical (high-level选graph, low-level执行)

实验结果显示，在compositional generalization上FIOC显著优于DreamerV3/TD-MPC2/EIT/DINO-WM，特别是skill composition和object count generalization。Ablation证明interaction modeling和hierarchical policy是最关键的两个component。

未来工作方向：combine with VLA foundation models (OpenVLA, π0)、scale到real robots、用transformer-based transition model、引入higher-order interactions。

主要参考链接：
- Paper: FIOC-WM (NeurIPS 2025 submission, Fan Feng, Phillip Lippe, Sara Magliacane)
- [[DINOv2]](https://arxiv.org/abs/2304.07193)
- [[R3M]](https://arxiv.org/abs/2203.12601)
- [[Slot Attention]](https://arxiv.org/abs/2006.15055)
- [[DreamerV3]](https://arxiv.org/abs/2301.04104)
- [[TD-MPC2]](https://arxiv.org/abs/2310.16828)
- [[DINO-WM]](https://arxiv.org/abs/2411.04983)
- [[SKILD]](https://arxiv.org/abs/2410.10966)
- [[EIT]](https://openreview.net/forum?id=uDxeSZ1wdI)
- [[NRI]](https://arxiv.org/abs/1802.08652)
- [[ACD]](https://arxiv.org/abs/2206.08163)
- [[Causal Dynamics Learning]](https://arxiv.org/abs/2205.14803)
- [[LIBERO]](https://arxiv.org/abs/2306.03310)
- [[Franka Kitchen / Relay Policy Learning]](https://arxiv.org/abs/1910.11972)
- [[iGibson]](https://arxiv.org/abs/2010.01820)
- [[Sprites World / COBRA]](https://arxiv.org/abs/1905.09275)
- [[V-JEPA 2]](https://arxiv.org/abs/2506.09985)
- [[Cosmos]](https://arxiv.org/abs/2501.03575)
- [[Gumbel-Softmax]](https://arxiv.org/abs/1611.01144)
- [[PPO]](https://arxiv.org/abs/1707.06347)
- [[OpenVLA]](https://arxiv.org/abs/2406.09246)
- [[π0]](https://arxiv.org/abs/2410.24164)

希望这个分析能让你建立对FIOC-WM的完整intuition。如果哪部分你想再深入，比如interaction graph learning的具体derivation、或hierarchical policy的graph search算法、或与JEPA/LEOPARD的更深比较，请告诉我，我可以再展开。
