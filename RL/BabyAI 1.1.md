---
source_pdf: BabyAI 1.1.pdf
paper_sha256: 1127bd2c430c6f0839222030ad9e350e08c1e967e40630b1ec25d7f8ddf7938b
processed_at: '2026-07-18T13:29:26-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# BabyAI 1.1 深度技术讲解

下面我会从架构直觉、具体修改、数学公式、实验数据到 broader 联想，把这篇 paper 拆开来给你。

## 1. Big picture: 这篇 paper 在做什么

BabyAI 本身是 Mila（Bengio、Bahdanau、Chevalier-Boisvert 等人）建的一个 gridworld grounded language learning 平台，目的是测一个 agent 能多 sample-efficient 地学会听人话办事（"go to the red ball"、"put the green box next to the blue key"）。BabyAI 1.0 [Chevalier-Boisvert et al., 2019] 给了一个 baseline agent，BabyAI 1.1 这篇是它的**architectural retrofit**：改了三件小事，把 RL sample efficiency 拉到 3×、把最难 BossLevel 的 IL success rate 从 77% 拉到 90.4%。

paper 的精神很 "engineering report"，重要的是这三处改动里每一处背后的 inductive-bias 论证，下面逐一拆。

参考：
- BabyAI repo: https://github.com/mila-iqia/babyai
- BabyAI 1.0 paper (ICLR 2019): https://arxiv.org/abs/1810.08272

---

## 2. BabyAI 平台背景

- 19 个 levels，按难度递增。paper 把它们大致分成 small (single-room) 和 big (3×3 rooms)。
- 每个 level 都是一组 procedurally generated gridworld，给 agent 一条 templated/合成语言指令，要求 agent 在 $n_{max}$ 步内完成。
- 评估指标两个：
  - **success rate**：完成任务的比例
  - **sample efficiency**：达到 ≥99% success rate 所需的 episodes / demonstrations 数

这套设计哲学跟 OpenAI 的 procgen、DeepMind 的 DM-Lab-30 是一个流派——**procedural curriculum + symbolic obs**，但 BabyAI 特别强调 symbolic input 因为它想隔离 language grounding 这一变量。

19 个 levels 的名字（你会在表里看到）大致是：
GoToObj, GoToRedBallGrey, GoToRedBall, GoToLocal, PutNextLocal, PickupLoc, GoToObjMaze, GoTo, Pickup, UnblockPickup, Open, Unlock, PutNext, Synth, SynthLoc, GoToSeq, SynthSeq, GoToImpUnlock, BossLevel。

---

## 3. Agent architecture（baseline）

整体 forward pass:

```
visual input (7×7×3)          instruction (token seq)
       │                              │
   conv encoder                    GRU
       │                              │
       └──────►  FiLM ×2 (batchnorm) ◄┘
                    │
                  LSTM (跨 timestep)
                    │
              policy head + value head
```

- **Visual encoder**：原始是几层 conv + maxpool，把 7×7×3 的小 symbolic map 压成更小的 feature map。
- **Language encoder**：GRU，small config 用单向 128-d，big config 用双向 128-d + Bahdanau attention。
- **FiLM**：把 language embedding 当作 conditioning signal，对 visual feature 做 feature-wise affine modulation。这是 grounded language learning 里非常自然的"用 language gating vision"。
- **LSTM**：跨 timestep 的 memory，dimension 128 (small) / 2048 (big)。
- **Policy/Value heads**：标准 A2C 形式。

训练方式：
- **RL**：A2C + PPO + GAE，batch 64 rollouts × 40 steps，4 PPO epochs，Adam。
- **IL**：BPTT truncated，small 20 steps，big 80 steps；big 还加 entropy regularizer 0.01。

参考：
- FiLM paper: https://arxiv.org/abs/1709.07871
- GRU (Cho et al., EMNLP 2014): https://arxiv.org/abs/1406.1078
- Bahdanau attention: https://arxiv.org/abs/1409.0473
- PPO: https://arxiv.org/abs/1707.06347
- GAE: https://arxiv.org/abs/1506.02438

---

## 4. 三个修改（这是 paper 的核心）

paper 把改动分成 **architectural（2个）** 和 **representational（1个，但有 2 种替代选项，最终采用 BOW）**。

### 4.1 Mod 1: Removing low-level maxpooling → `endpool`

**BabyAI 1.0**：visual encoder 是经典 conv→maxpool→conv→maxpool 的金字塔结构。
**BabyAI 1.1**：只在 visual encoder **最后** 做一次 pooling，中间全部用 stride-1 的 conv，filter size 从 $2\times2$ 改为 $3\times3$ 来 keep feature map shape consistent。

**为什么这个改动有效 — intuition:**

- 在 ImageNet 上，pooling 的归纳偏置是 **translation invariance**，因为同一只猫在图像的左上或右下都该被识别为猫。
- 在 BabyAI 里，输入只有 **7×7**，"位置"本身是 grounded instruction 的一部分。"go to the red ball on the left"——left 这个信息是语义信息，需要保留 spatial layout。
- 7×7 → pooling 后变成 3×3 甚至 1×1，**信息瓶颈把 grounding 的位置结构毁了**。这相当于在 ImageNet 上把 224×224 一开始就 pool 到 28×28，但又没有对应的 receptive field 来恢复细节。
- 移除 pooling 让中间层激活能保留每个 tile 的 identity，CNN 后面的 FiLM 才能做 tile-level language-conditioned reasoning。

这条改动在 Table 1a 里非常显眼：GoToLocal 从 **1311 → 381** episodes（~3.4×），PutNextLocal 从 **2984 → 2169**，PickupLoc 从 **1797 → 743**。

### 4.2 Mod 2: Residual connections around conv + FiLM → `res`

FiLM layer 本身的形式（Perez et al. 2017）：

$$
\text{FiLM}(x \mid c) = \gamma(c) \odot x + \beta(c)
$$

- $x$：visual feature map（某个 channel）
- $c$：conditioning signal，这里是 GRU 输出的 language embedding
- $\gamma(c), \beta(c)$：从 $c$ 经 MLP 学到的 affine 参数，shape 与 $x$ 对应
- $\odot$：element-wise (Hadamard) product

BabyAI 1.1 在 FiLM 外面套了 residual：

$$
y = x + \text{FiLM}(x \mid c)
$$

**Intuition:**

- FiLM 本身是 **conditional affine transform**，本质跟 conditional batch norm (cBN) 是一家，能让 language "rescale" 视觉特征。但 FiLM 是 **完全替换**——identity 通路被 $\gamma, \beta$ 重写。
- 加 residual 后，default 行为变成 "identity + modulation"，对梯度回传更友好（类似 ResNet 的 highway）。
- 当语言指令含糊或 weak 时，network 退化为 identity，保留 visual feature；当指令明确时，modulation kick in。
- 这种 "soft modulation" 在 NLP 的 adapter、vision-language 的 FiLM 后续工作里都看得到影子。

Table 1a 显示这个改动效果是 mixed 的：GoToLocal 从 endpool 的 381 → 437（略差），但 PutNextLocal 从 2169 → **1009**（显著好）。paper 选择 adopt，因为 PutNextLocal 是最难那批，improvement 最 weighted。

### 4.3 Mod 3: Visual representation 改用 BOW embedding

BabyAI 1.0 用 **triple-integer tile encoding**：每个 tile 由 (object_type, color, door_state) 三个整数表示，整个 7×7 grid 拼成 **7×7×3** tensor 喂给 conv。

BabyAI 1.1 主推的 BOW 改法：
- 准备 **3 个 trainable lookup tables**（embeddings），每个对应一个 integer key（object_type、color、door_state）。
- 每个 tile 用这三个 integer 各自 lookup 出 $d=128$ 的 vector，然后取平均：

$$
e_{\text{tile}} = \frac{1}{3} \left( E_{\text{type}}[t_1] + E_{\text{color}}[t_2] + E_{\text{door}}[t_3] \right)
$$

- $E_{\text{type}}, E_{\text{color}}, E_{\text{door}}$：三个 nn.Embedding lookup table，每个 $|\text{vocab}|\times 128$
- $t_1, t_2, t_3$：tile 的三个 integer key
- 整个 7×7 grid 拼成 **7×7×128** tensor

第三种替代是 **pixels**：每个 tile 渲染成 8×8×3 RGB 小图，整个 grid 变 56×56×3 大图，喂标准 image CNN。

**Intuition:**

- Triple-integer 是一种非常 **leaky symbolic** 表示：第 1 channel 是 type ID、第 2 channel 是 color ID、第 3 是 door state ID，conv 的第一层必须自己学会把这些 ordinal integer 解码成 categorical semantics，这一步浪费 capacity。
- BOW 是 NLP 标准做法 (word2vec [Mikolov et al., 2013])，把离散 symbol 直接映射到 dense 128-d 空间，**让网络从 semantic space 起步**。这跟 symbolic RL 里 gym-sokoban、AI safety gridworlds [Leike et al., 2017] 的做法一致。
- Pixels 在 gridworld 里没意义：gridworld 渲染成 RGB 并没有引入 occlusion / illumination / viewpoint 这些真实视觉的 challenge，所以 pixels 输入只是**增加了计算 cost 而没增加 task difficulty**——这跟 paper conclusion 里说 "pixel representation 不 bridge reality gap" 一致。

参考：
- word2vec: https://arxiv.org/abs/1301.3781
- AI safety gridworlds: https://arxiv.org/abs/1711.09883
- gym-sokoban: https://github.com/mpSchrader/gym-sokoban
- Attend-Adapt-Transfer (Rajendran et al.): https://arxiv.org/abs/1606.05376

---

## 5. 完整架构图（结合 Figure 1 的 5 个变体）

paper 命名规则：`<visual_repr>_<arch_mods>`，其中：
- visual_repr ∈ {original, bow, pixels}
- arch_mods ∈ {endpool, res}（可组合，可空）

| 名字 | visual 表示 | pool 位置 | res |
|---|---|---|---|
| original | 7×7×3 | 多处 maxpool | 否 |
| original_endpool | 7×7×3 | 仅末尾 | 否 |
| original_endpool_res | 7×7×3 | 仅末尾 | 是 (FiLM+conv) |
| bow_endpool_res | 7×7×128 | 仅末尾 | 是 |
| pixels_endpool_res | 56×56×3 | 仅末尾 | 是 |

`(d)` 和 `(e)` 在 paper 里画的是：visual representation 改成 BOW / pixels 之后，整个分支只改第一层，**从第一个 conv 起就与 (c) 同构**，这降低了 implementation cost。

---

## 6. 实验设置详解

### 6.1 RL setup

- Algorithm：A2C [Wu et al., 2017] + PPO [Schulman et al., 2017] + GAE [Schulman et al., 2015]
- Batch：64 rollouts × 40 steps = **2560 timesteps per update**
- PPO epochs：4
- GAE：$\lambda = 0.99$
- Discount：$\gamma = 0.99$
- Optimizer：Adam [Kingma & Ba, 2015]
  - $\alpha = 1\times10^{-4}$（默认），pixels 实验下调到 $5\times10^{-5}$
  - $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-5}$

**Reward shaping**（这里很关键）：

$$
r = 1 - 0.9 \cdot \frac{n}{n_{max}}
$$

- $n$：agent 完成任务用的步数
- $n_{max}$：该 level 设定的最大步数
- 系数 0.9 让"满步刚好完成"的 reward = 0.1，"立即完成"接近 1.0
- **未完成时 r = 0**，所以这是个 dense-in-success, sparse-in-failure 的 reward

**Returns**:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}, \quad \gamma = 0.99
$$

**GAE**:

$$
A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

- $\delta_t$：TD error
- $\lambda = 0.99$：bias-variance trade-off，接近 1 = 接近 Monte Carlo
- $V(s_t)$：critic head 输出

**Sample efficiency 定义**：达到 ≥99% success rate 所需的 episodes 数（越小越好）。

### 6.2 IL setup

- Algorithm：behavior cloning，cross-entropy on expert demonstrations
- Small architecture：batch size 256，1 epoch = 25600 demos，BPTT truncated at 20 steps
- Big architecture：batch size 128，1 epoch = 102400 demos，BPTT truncated at 80 steps，entropy regularizer 系数 0.01
- Optimizer：Adam，small $\alpha=1\times10^{-4}$，big $\alpha=5\times10^{-5}$
- Demonstration 数量梯度：5k, 10k, 50k, 100k, 500k（对应 1M demos 的 1/200, 1/100, 1/20, 1/10, 1/2）

参考：
- A2C (Kfac TRPO 论文里也提到 A2C baseline): https://arxiv.org/abs/1708.05144
- Adam: https://arxiv.org/abs/1412.6980
- BPTT (Werbos 1990): classic, no arxiv
- LSTM (Hochreiter & Schmidhuber 1997): https://www.bioinf.jku.at/Members/brain83/lstm.pdf
- Zołna et al. 2020 (adversarial IL false negatives, BabyAI 同作者): https://arxiv.org/abs/2002.00412

---

## 7. 实验数据逐表精读

### Table 1a：architecture ablation（RL sample efficiency, 千 episodes）

| Level | original | original_endpool | original_endpool_res |
|---|---|---|---|
| GoToRedBallGrey | 21±5 | 21±6 | 21±5 |
| GoToRedBall | 273±27 | **200±16** | **179±17** |
| GoToLocal | 1311±251 | **381±30** | 437±45 |
| PickupLoc | 1797±290 | **743±132** | 710±166 |
| PutNextLocal | 2984±172 | 2169±739 | **1009±128** |
| GoTo | 1601±463 | **454±69** | 813±278 |

观察：
- GoToRedBallGrey 太简单（21），所有架构都立刻 solve，没差别——天花板效应。
- endpool 在 GoToLocal 上拿到 **3.4×** 提升，在 PickupLoc 上 2.4×，在 GoTo 上 3.5×。
- res 在 PutNextLocal 上拿到 **2× over endpool**，但在 GoTo / GoToLocal 上反而略差。这正是 paper Section 3.1 说的 "increases and decreases"，但他们选 res 因为 PutNextLocal 是最难的 small level。

### Table 1b：visual representation, lr=1e-4

| Level | original_endpool_res | bow_endpool_res | pixels_endpool_res |
|---|---|---|---|
| GoToRedBallGrey | 21±5 | 24±2 | 34±4 |
| GoToRedBall | 179±17 | 177±2 | 172±2 |
| GoToLocal | 437±45 | 611±760 (huge std!) | **242±15** |
| PickupLoc | 710±166 | 982±266 | 1082±385 |
| PutNextLocal | 1092±143 | **876±104** | Not Trainable |
| GoTo | 813±278 | 817±502 | Not Trainable |

观察：
- pixels 在最难两个 level 上 **Not Trainable**——这告诉我们 pixels 对 small symbolic gridworld 是一个 fragile choice，需要 lr warmup。
- bow 在 PutNextLocal 上比 original 好（876 vs 1092），但 std 都不小。

### Table 1c：visual representation, lr=5e-5（halved for pixels）

| Level | original_endpool_res | bow_endpool_res | pixels_endpool_res |
|---|---|---|---|
| GoToRedBallGrey | 35±5 | 30±4 | 44±11 |
| GoToRedBall | 263±22 | **164±3** | 155±5 |
| GoToLocal | 606±81 | 449±176 | 336±28 |
| PickupLoc | 1732±579 | 1461±422 | 1308±421 |
| PutNextLocal | 1277±252 | **876±104** | 1301±320 |
| GoTo | 984±484 | 803±525 | 845±329 |

观察：
- lr 调小后 original_endpool_res 在多数 level 变差（273→263, 437→606, 813→984）。这暗示 original 是 lr-sensitive 的。
- pixels 在 lr=5e-5 下终于 trainable，且在 GoToRedBall / GoToLocal / PickupLoc 上比 original 好，说明 **pixels 需要更保守的 optimization，但调好后跟 symbolic 持平甚至略优**。

### Table 2：throughput

| Architecture | original | original_endpool | original_endpool_res | bow_endpool_res | pixels_endpool_res |
|---|---|---|---|---|---|
| FPS | 1139±128 | 927±72 | 907±69 | 855±58 | 540±67 |

- pixels 比 original 慢约 **2.1×**，因为 image 大了（56×56×3 vs 7×7×3）+ conv 多了。
- bow_endpool_res 比 original_endpool_res 慢一点点（855 vs 907），来自 embedding lookup overhead，但很小。
- 关键 trade-off：sample efficiency 3× 但 throughput 20% 慢，**net wall-clock gain 大约 2.4×**。

### Table 3：IL success rate（六 easiest levels + GoTo）

我只挑几行精彩数据：

**GoToRedBall**（5k demos）：

| Architecture | Success |
|---|---|
| original | 89.6±0.3 |
| original_endpool_res | 91.3±0.6 |
| **bow_endpool_res** | **99.3±0.3** |

→ BOW 在小 demo regime 下 **碾压**。embedding lookup 给了网络一个 free semantic prior。

**PutNextLocal**（5k demos）：

| Architecture | Success |
|---|---|
| original | 22.3±1.7 |
| original_endpool_res | 12.0±1.8 |
| bow_endpool_res | 12.5±1.2 |

→ 5k demos 太少，endpool + res **反而 hurt**。这很反直觉：架构改进让 model 容量大，在小 demo 下更容易 overfit。100k demos 时反超：original_endpool_res 99.5% vs original 93.9%。

**GoTo**（10k demos, big arch）：

| Architecture | Success |
|---|---|
| original | 70.4±1.1 |
| original_endpool_res | 76.3±5.0 |
| **bow_endpool_res** | **96.1±0.4** |

→ BOW + big arch 在 10k demos 时就 96%，这是个非常强的结果。意味着 BOW prior 在 IL 的 low-data regime 是一个 game-changer。

### Table 4：全 19 levels benchmark（IL, 1M demos）

挑有意思的几行：

| Level | BabyAI 1.0 | BabyAI 1.1 | Demo Length |
|---|---|---|---|
| GoToObj | 100 | 100 | 5.18±2.38 |
| GoToLocal | 99.8 | 100 | 5.04±2.76 |
| PutNextLocal | 99.2 | 100 | 12.4±4.54 |
| GoTo | 99.4 | 100 | 56.8±46.7 |
| Unlock | 98.4 | 100 | 81.6±61.1 |
| PutNext | 98.8 | 99.6 | 89.9±49.6 |
| Synth | 97.3 | 100 | 50.4±49.3 |
| GoToSeq | 95.4 | 96.7 | 72.7±52.2 |
| SynthSeq | 87.7 | 93.9 | 81.8±61.3 |
| GoToImpUnlock | 87.2 | **84.0** | 110±81.9 |
| **BossLevel** | **77** | **90.4** | 84.3±64.5 |

观察：
- 4 个新 "solved" level：Unlock, PutNext, Synth, SynthLoc（>=99% 才算 solved）。
- **GoToImpUnlock 反而退化**：87.2 → 84.0。这是 paper 没大讨论但很有意思的点——这个 level 是 implicit unlock（要先开门但 instruction 没说），需要更长程的 planning；BOW + endpool 可能让 model 短视了。
- BossLevel 77→90.4 是 headline number，paper abstract 主打。
- Demo length 也很有意思：GoToImpUnlock 110±81.9 是最长，std 几乎等于 mean，说明这个 level 任务长度 bimodal（要么快速 unlock 要么卡很久）。

---

## 8. 我自己的直觉与联想

### 8.1 关于 "为什么 endpool 这么有效"

7×7 input 是个非常特殊尺寸。在 224×224 上，3 次 stride-2 pool 让你到 28×28，receptive field 还有空间。在 7×7 上，2 次 2×2 pool 就到 1×1 了。所以 pooling 在 BabyAI 的 visual encoder 里相当于 **collapse 到 global average pooling**，这一步本来应该发生在网络最后，1.0 把它放到了开头，相当于告诉网络"位置不重要"——而 grounded instruction 的核心就是位置。

类似的事情在 AlphaGo / AlphaZero 里也有：board state 19×19，他们也是 stride-1 conv 一直跑到底，没有 aggressive pooling。**Symbolic grid perception 不需要 invariance，需要 equivariance**。

### 8.2 关于 FiLM + Residual

FiLM 跟 Conditional Batch Normalization (cBN, de Vries et al. 2017, https://arxiv.org/abs/1707.00683) 几乎是同一个 idea 的两种实现。cBN 用 BN 的 $\gamma, \beta$，FiLM 用 free-form affine。两者的共同问题是：**conditioning signal 一旦弱，整个 feature map 被 zero out 或 scale 异常**。Residual 让 default 通路是 identity，modulation 是 perturbation——这正是 ControlNet  后来在 diffusion 里用过的 trick。

我直觉是：未来 grounded language agent 的标配会是 "ResNet backbone + cross-attention 或 FiLM adapter"，FiLM 因为 cheap 适合 layer-wise，cross-attention 因为 expressive 适合 final fusion。BabyAI 选 FiLM 是合理 cost-benefit。

### 8.3 关于 BOW > triple-integer

这是 paper 里最 underplayed 但我个人觉得最重要的 finding。triple-integer encoding 的本质问题是：**它把 categorical variable 当作 continuous scalar 输入**。第 1 channel 是 type ID 1-5，conv 的第一层是固定权重的 linear filter $Wx + b$，它必须学会 "type=3 跟 type=4 不一定比 type=2 离 type=3 更近"。这就是让 CNN 学 ordinal-to-categorical 的 decoding，浪费 capacity。

BOW 直接绕开这步：lookup table 是 one-hot → dense 的可微 map，本质上等价于 conv 第一层只学 categorical embedding。

**Intuition**: 当你的 input 已经是 symbolic 且 categorical 时，**第一条可微变换就应该是 embedding lookup，不是 conv**。这跟 NLP 不用 conv 第一层而用 nn.Embedding 是同一个 lesson。

### 8.4 关于 "Pixels 不 work"

paper conclusion 说 pixels 没用因为 reality gap。我同意，但我想加一个直觉：gridworld 的 RGB 渲染是 **deterministic function of symbolic state**，所以 pixels 和 symbolic input 信息论上等价（mutual information 100%）。模型从 pixels 学到的就是 "如何 decode pixels 回到 symbolic state"——这是把 capacity 浪费在 idempotent transform 上。

这跟 MiniGrid、NetHack Learning Env、Procgen 的 symbolic variant 一个套路。当 task 难度真正在 reasoning/planning 而不在 perception 时，**应该用 symbolic input 把 perception bottleneck 去掉**，让 evaluation 聚焦 reasoning。

### 8.5 Sample efficiency vs throughput 的工程权衡

paper Table 2 给了一个非常工程师的视角。最终 bow_endpool_res 比 original 慢 25% (855 vs 1139 FPS)，但 sample efficiency 3×。Wall-clock 训练时间约：

$$
T_{\text{wall}} = \frac{N_{\text{episodes}} \times \bar{L}_{\text{episode}}}{\text{FPS}}
$$

例如 GoToLocal：original 1311k episodes × 5 steps / 1139 FPS ≈ 5757 秒；bow_endpool_res 假设 449k episodes × 5 / 855 ≈ 2625 秒。**~2.2× wall-clock speedup**。这对做大量 ablation 的研究机构（Mila、Element AI）是非常实在的 cost saving。

### 8.6 关于 IL 在 small-demo regime 的反直觉

Table 3 PutNextLocal @ 5k demos：original 22.3%, 改进版 12%。这是一个非常 ML-engineering-style 的现象：**big model + small data = overfit**。endpool + res 增加了 model 容量，5k demos 下 overfit，100k demos 才翻盘。这暗示 paper 的架构改动可能有一个 sweet spot 在 demo 数量的某个阈值，低于这个阈值 baseline 反而更鲁棒。

这让我联想到 GPT-3 的 in-context learning：在小 prompt 下，小模型比大模型稳；在大 prompt 下，大模型碾压。BabyAI IL 在 demo 维度上重现了这一规律。

### 8.7 关于 GoToImpUnlock 退化

paper 没深挖但这是个有意思的 negative result。GoToImpUnlock = "GoTo" + 隐含需要先 unlock 一个门。instruction 里没说 unlock，但 agent 必须 infer。这需要 **non-myopic planning**：先做 instruction 没说的 action，才能完成 instruction 说的 action。

bow_endpool_res 让网络更聚焦 local visual feature，可能 lost 了某些 long-horizon cue。这暗示一个 architectural trade-off：**spatial fidelity 和 long-horizon reasoning 在 small grid 上是 tension**。如果你想解决 GoToImpUnlock 这类 implicit-subgoal 任务，可能需要 explicit hierarchical structure（options / sub-policies / sub-goal prediction），而不是单纯调 visual encoder。

---

## 9. Limitations & 我会想做的 follow-up

1. **Statistical rigor**: Table 4 全 19 levels 只有 1 seed。这在 sample-efficiency benchmark 里其实不够，特别是 GoToImpUnlock 那个 84.0 vs 87.2 的 3 个百分点完全可能在 noise 范围。我会想跑 ≥5 seeds 给 confidence interval。

2. **Architecture search 太窄**: paper 只 ablate 了 pooling 位置、residual、visual representation。没 ablate FiLM layer 数量、GRU vs Transformer、LSTM vs nothing。Transformer encoder 已经在 2020 显示对 grounded instruction 有效（DETR、ViLBERT），BabyAI 没试是个 missed opportunity。

3. **No curriculum**: 19 levels 实际上天然有 curriculum，但 paper 只 train on single level。我会想看 bow_endpool_res 在 multi-level joint training 下如何。

4. **Self-attention on 7×7**: 7×7 = 49 tokens，完全可以做 ViT-style flat attention。直觉上 attention 比 conv 更适合 gridworld 的 object-centric reasoning（每个 tile 是一个 object）。

5. **Reward shaping 1-0.9n/n_max**: 这个 reward 在 n=0 时 r=1, n=n_max 时 r=0.1。这意味着"满步完成"还有 0.1 reward。一个 alternative 是稀疏 reward (只在 done 时给 1.0)，这样更能测真正的 sparse-reward sample efficiency。paper 没对比。

6. **Asymmetric encoder/decoder**: BOW 是双向的——既能从 symbol 到 dense，也能从 dense 想象 symbol。可以加 auxiliary loss "reconstruct tile symbols from feature map"，类似 VAE-style grounding。直觉上这会让 representation 更 compositional。

---

## 10. 关键 references 与链接

| 主题 | 文献 | 链接 |
|---|---|---|
| BabyAI 1.0 | Chevalier-Boisvert et al., ICLR 2019 | https://arxiv.org/abs/1810.08272 |
| BabyAI repo | mila-iqia | https://github.com/mila-iqia/babyai |
| FiLM | Perez et al., AAAI 2017 | https://arxiv.org/abs/1709.07871 |
| cBN (related) | de Vries et al. 2017 | https://arxiv.org/abs/1707.00683 |
| GRU | Cho et al., EMNLP 2014 | https://arxiv.org/abs/1406.1078 |
| LSTM | Hochreiter & Schmidhuber 1997 | https://www.bioinf.jku.at/Members/brain83/lstm.pdf |
| Bahdanau attention | Bahdanau et al. 2015 | https://arxiv.org/abs/1409.0473 |
| PPO | Schulman et al. 2017 | https://arxiv.org/abs/1707.06347 |
| GAE | Schulman et al. 2015 | https://arxiv.org/abs/1506.02438 |
| Adam | Kingma & Ba 2015 | https://arxiv.org/abs/1412.6980 |
| word2vec (BOW) | Mikolov et al. 2013 | https://arxiv.org/abs/1301.3781 |
| AI safety gridworlds | Leike et al. 2017 | https://arxiv.org/abs/1711.09883 |
| gym-sokoban | Schrader 2018 | https://github.com/mpSchrader/gym-sokoban |
| A2C / Kfac TRPO | Wu et al. 2017 | https://arxiv.org/abs/1708.05144 |
| Adversarial IL false negatives (BabyAI 同作者) | Zołna et al. 2020 | https://arxiv.org/abs/2002.00412 |
| Attend-Adapt-Transfer | Rajendran et al. 2017 | https://arxiv.org/abs/1606.05376 |
| Gaussian Processes | Rasmussen & Williams 2006 | http://www.gaussianprocess.org/gpml/ |
| BPTT | Werbos 1990 | classic, IEEE Proc. |
| MiniGrid (related platform) | Chevalier-Boisvert et al. | https://github.com/maximecb/gym-minigrid |

---

## 11. 一句话总结 intuition

BabyAI 1.1 教给我们三件事：(1) **小 input 不该早 pooling**——symbolic grid 上的位置是 grounding 的语义，需要保留；(2) **conditional modulation 需要 identity 通路**——FiLM 加 residual 让语言条件化变成"perturb 而不是 replace"，更稳；(3) **categorical 输入第一步该 embedding 化**——把 integer 当 scalar 喂 conv 是浪费 capacity，BOW embedding 给网络一个 free semantic prior。这三条加起来让 grounded language agent 在 gridworld 上 3× sample-efficient、BossLevel 90.4%，且 wall-clock 更快——纯架构工程的三连胜。
