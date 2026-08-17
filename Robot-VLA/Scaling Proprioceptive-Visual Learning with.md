---
source_pdf: Scaling Proprioceptive-Visual Learning with.pdf
paper_sha256: e4cee3f35f6d2bd03ecc1d114809794042f2aa17a2f48304b0d7a90c094e48ba
processed_at: '2026-08-12T03:39:36-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 HPT

## 一句话概括

你想训练一个"什么机器人都能干"的通用policy，但发现各家机器人的数据完全没法混在一起用——因为机器人长得不一样、传感器不一样、动作空间不一样。HPT的解法很简单：**给每个机器人配个"翻译官"，让大家都说同一种"内部语言"，然后让一个共享的大脑去学怎么处理这种统一语言。**

---

## 为什么这事难

假设你想搞一个"机器人界的GPT"，把全世界所有robot manipulation数据喂进去训练。你打开Open-X Embodiment这个数据集一看，傻眼了：

- Franka Panda是7-DoF arm，end-effector pose是7维（xyz + quaternion）
- Kuka是另外一套joint angles，维度都不一样
- Aloha是bimanual，两个arm加起来14+维
- 有的机器人有wrist camera，有的是third-person view
- 有的action是absolute pose，有的是relative displacement
- 有的数据带language instruction，有的没有
- 仿真数据更乱，MuJoCo、Isaac、PyBullet各玩各的

传统做法要么只挑一种机器人训练（overfit，不generalize），要么花大量人力把所有数据**手动统一格式**（Octo、RT-X就是这么干的，比如统一成language-conditioned format，但这样就把proprioception信息丢了或者硬凑成一样维度）。

核心矛盾：**数据多但杂，硬统一会丢信息，不统一又没法联合训练。**

参考: https://robotics-transformer-x.github.io

---

## HPT的核心idea

想象联合国开会，各国代表说不同语言。解法不是逼所有人都学英语，而是**每人配个同声传译**，传译后输出统一格式的要点，然后一个中央委员会基于这些要点做决策。

HPT把policy网络拆成三段：

```
传感器原始数据 → [翻译官 Stem] → 32个token的统一语言 → [中央大脑 Trunk] → 高层语义 → [执行器 Head] → 具体动作
   (每个机器人一个)            (全世界共享一个)          (每个任务一个)
```

- **Stem**: 每个embodiment配一个，负责把"我的proprioception + 我的vision"翻译成16+16=32个固定维度的token
- **Trunk**: 所有人共享的transformer，吃32个token，吐32个token，参数全世界共用
- **Head**: 每个task配一个，把trunk输出翻译成这个task需要的action

这样设计的好处：
1. Trunk的输入永远是32个固定dim的token，不管原始数据多乱，**对trunk来说都是同构的**
2. 不同embodiment的数据可以**同时、联合训练trunk**，因为它们在token层面被align了
3. Trunk学到的是"机器人无关、任务无关"的高层representation
4. 迁移到新机器人时，**只需要训一个新stem和head**，trunk冻结

这模仿了人类神经系统的层级：spinal cord做local sensorimotor reflex（stem），brain做abstract planning（trunk），motor cortex下发specific command（head）。Paper ref [68] https://www.science.org/doi/10.1126/scirobotics.add5434

---

## Stem具体怎么"翻译"

这是最key的技术细节。问题：proprioception维度从7到30+不等，vision有单/多view，怎么压成固定32个token？

### Proprioception Tokenizer

以Franka 7-DoF为例：

1. 输入向量 $p \in \mathbb{R}^7$（end-effector的xyz + quat）
2. 过一个MLP升维到 $d$（比如256）：$\tilde{p} \in \mathbb{R}^{256}$
3. 准备16个learnable query token $Q_p \in \mathbb{R}^{16 \times 256}$
4. 用cross-attention：query是$Q_p$，key和value都是$\tilde{p}$（加sinusoidal positional encoding）
5. 输出16个256维的proprioception token

为什么用cross-attention？这是Perceiver (https://arxiv.org/abs/2107.14795) 的套路：用少量learnable token去"询问"变长输入，把信息压缩成固定长度。好处是无论输入是7维还是30维，输出永远是16个token，trunk不感知到异构性。

### Vision Tokenizer

1. 图像过frozen ResNet18，拿到7×7=49个spatial feature
2. Flatten成49个token，project到dim 256
3. 准备16个learnable query token
4. Cross-attention压缩成16个vision token

最后concatenation：16 proprio + 16 vision = 32 token，加modality embedding和position embedding，喂给trunk。

**关键insight**：固定32 token是一个strong bottleneck。强制模型学会compress information。如果给256个token（像Octo那样），model会lazy地memorize而不是abstract。16是经验值，paper没仔细ablate这个数量，是个潜在future work。

参考 Octo 对比: https://octo-models.github.io

---

## Trunk和Head

Trunk就是standard decoder-only transformer，和GPT架构一样：

- HPT-Small: 16层, 128 dim, 3.1M params
- HPT-Huge: 80层, 1024 dim, 1.1B params

输入32个token，输出32个token，然后mean pool成一个feature vector。

Head把feature vector映射到action。Paper支持多种head：
- MLP（默认）
- Transformer decoder（用于ACT-style任务）
- Diffusion policy（用于real-world高精度任务）

Action horizon=8，observation horizon=4。训练时随机mask时间维度，这样transfer时能适配不同horizon需求。

---

## 训练目标

公式1其实很朴素，就是behavior cloning的Huber loss：

$$\min_\theta \sum_{k=1}^{K} \mathcal{L}(\theta_k^{\text{stem}}, \theta^{\text{trunk}}, \theta_k^{\text{head}}; \mathcal{D}_k)$$

- $K$ = 52个dataset
- $\theta_k^{\text{stem}}$, $\theta_k^{\text{head}}$ = 第$k$个dataset专属
- $\theta^{\text{trunk}}$ = 全局共享
- $\mathcal{D}_k$ = 第$k$个dataset的轨迹集合
- $\mathcal{L}$ = Huber loss between normalized action prediction and ground truth

Huber loss的$\delta=0.1$是经验值。好处：对"difficult frame"（contact瞬间梯度大）robust，对"easy lengthy frame"保持平滑。

数据采样用温度采样防大dataset主导：$p_k \propto \sqrt{M_k}$，$M_k$是第$k$个dataset的trajectory数。这是multitask learning标准操作。

---

## 最有意思的发现：Scaling Laws在Robotics也成立

这是paper最exciting的部分，直接复现了LLM的scaling laws (https://arxiv.org/abs/2001.08361) 在robotics的版本。

### Data Scaling (Figure 5a)

- 蓝线：固定HPT-Small，只增加数据 → 早期plateau（1000 traj/dataset就到顶）
- 红线：数据+模型+compute同步增加 → 稳定下降

这就是Chinchilla (https://arxiv.org/abs/2203.15556) 说的"data和compute必须tandem scale"在robotics的复现。你光堆数据不堆模型，model会underfit。

### Dataset Diversity Scaling (Figure 5b)

固定评估10个dataset，增加pre-training的dataset数量（10→52）。发现**更多embodiment diversity → 更低validation loss**。这暗示trunk确实学到了embodiment-agnostic的representation，否则加更多不相关dataset只会hurt。

### Model Scaling (Figure 7)

1M到1B参数。需要数据足够时才有用，否则大模型underfit。Depth vs width scaling无显著差异。

### Epoch Scaling (Figure 6)

大batch size不仅加速，还能**降低heterogeneous data sampling的variance**，类似SGD noise reduction。

**Intuition**：这给robotics foundation model指了条路——只要scaling laws成立，数据越多、模型越大、compute越多，性能就越涨。目前1B参数才刚接近收敛，还有很大空间。

---

## 跨域数据也能用

HPT不止用real robot teleop数据，还加了：

- **7个仿真dataset**: Drake, MuJoCo, Isaac, PyBullet, Sapien, Flex
- **人类视频**: EPIC-Kitchens（用2D bbox center当proprio，frame difference当action），PoCo（3D hand position via ICP当proprio）
- **Deployed robot**: FrodoBot-2K（mobile robot户外驾驶，IMU当proprio）

Figure 8显示joint pre-training这些"奇怪"数据带来小幅提升。这说明HPT的framework足够flexible，连"假"proprio/action（从视频提取的）都能吸收信号。

**Insight**: 这为未来"用YouTube视频pretrain机器人policy"开了个口子。虽然信号弱，但数据量巨大。参考R3M https://arxiv.org/abs/2203.12601 之前只pretrain vision，HPT把proprio也纳入了。

---

## Transfer效果

### Simulation

Meta-world, RoboMimic, Fleet-Tools, Simpler benchmark。Pre-trained HPT一致优于from-scratch。HPT-XL > HPT-L > HPT-B。预训练在real robot数据上，transfer到sim也能work，说明representation确实abstract。

### Real World（最impressive）

4个contact-rich任务：Sweep Leftover, Fill Water, Scoop Food, Switch Insertion。两个不同embodiment，不同camera setup。

Table 3的Sweep Leftover结果：

| Method | Success (%) |
|--------|-------------|
| From Scratch | 43.3 |
| R3M (vision-only pretrain) | 50.0 |
| VC-1 (vision-only pretrain) | 53.3 |
| No Prop. Finetuned (vision pretrain + 后加proprio) | 63.3 |
| **HPT-B Finetuned** | **70.0** |
| **HPT-XL Finetuned** | **76.7** |

**关键insight**：
1. HPT比vision-only pretrain高20%+
2. Joint proprio+vision pretrain明显优于"先vision后加proprio"
3. Transfer时只训stem+head（~3MB，占~2%参数），trunk冻结
4. 从HPT-B到HPT-XL还有提升，说明pre-training scale matters

这给robotics community一个strong signal：**proprioception应该和vision一起joint pretrain，不能post-hoc拼接**。

---

## 为什么之前没人这么干

- **Octo**: 用256个token，unified obs/action space。需要手动align数据格式。https://octo-models.github.io
- **RT-X**: 用language conditioning统一，但牺牲proprioception。https://robotics-transformer-x.github.io
- **OpenVLA**: VLA框架，依赖language。https://openvla.github.io
- **R3M/VC-1/Voltron**: 只pretrain vision backbone，proprioception留给下游

HPT的差异化：**让architecture吸收heterogeneity，而不是让data engineering去uniform化**。每个embodiment保留自己的"方言"，靠stem翻译成共享语言。

---

## Limitations（paper自承认 + 我加的）

1. **Data curation粗放**: 简单balanced sampling，没做quality filtering
2. **只用了supervised loss**: 没探索self-supervised或world model objective
3. **Scale还小**: 1B参数 vs LLM的几百B，还没到compute-optimal
4. **评估局限**: 只测short-horizon manipulation，没测long-horizon、bimanual、mobile
5. **可靠性不够**: 76.7%还远低于production-grade
6. **Token数量(16)没仔细ablate**: 为什么不是8或32？
7. **Stem架构可能不是最优**: 为什么不用transformer encoder而用单层cross-attention？
8. **MoE视角没展开**: 每个embodiment一个expert，但没做soft routing让相近embodiment共享
9. **Failure mode**: Figure 19显示spatial overshoot/undershoot（倒水倒到杯子前面），可能vision encoder精度不够或data quality问题

---

## 我觉得最有潜力的方向

1. **Soft MoE**: 让相近embodiment（比如所有7-DoF arm）共享stem参数，而不是每个embodiment完全独立。可能提升data efficiency
2. **Self-supervised pretraining**: 当前只用action label，浪费了大量无label video。可以加world model或contrastive objective
3. **3D/tactile/audio extension**: Stem架构理论上支持，但paper没实验。Tactile对contact-rich task很关键
4. **URDF as inductive bias**: 把robot morphology作为graph prior注入stem，可能比纯MLP更sample efficient
5. **Long-horizon hierarchy**: 加个"meta-trunk"做subgoal planning，trunk做low-level control
6. **Scaling to 10B+**: 当前1B可能不够，需要更大scale验证scaling law是否持续
7. **Closed-loop evaluation metric**: Validation loss和closed-loop success rate有gap，需要更好intrinsic metric

---

## 实用资源

- HPT project: https://liruiw.github.io/hpt
- LeRobot (HuggingFace实现): https://github.com/huggingface/lerobot
- Open-X dataset: https://robotics-transformer-x.github.io
- DROID dataset: https://droid-dataset.github.io
- Octo: https://octo-models.github.io
- Simpler benchmark: https://simpler-env.github.io
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu
- Scaling laws: https://arxiv.org/abs/2001.08361
- Chinchilla: https://arxiv.org/abs/2203.15556
- Perceiver IO: https://arxiv.org/abs/2107.14795
- R3M: https://arxiv.org/abs/2203.12601
- VC-1: https://arxiv.org/abs/2310.16488

---

## 一句话总结Intuition

**HPT把"数据统一"问题转化成"架构对齐"问题**——通过给每个机器人配个翻译stem，让所有异构数据在token层面变成同构，从而让一个shared transformer trunk能从所有数据里学shared representation。这比强迫所有数据塞进同一格式要natural得多，也更保留信息。Scaling laws在robotics也成立，只是目前scale还不够。未来如果scale到10B+参数和10M+ trajectories，加上self-supervised objective和cross-domain data，有可能真的接近robotic foundation model。

---

# HPT: Heterogeneous Pre-trained Transformers 深度讲解

## 1. 核心动机与问题定位

机器人学习的根本困境在于 **heterogeneity**（异构性）。与传统NLP/CV不同，robotics data的异构性体现在多个层面：

- **Proprioception异构**：不同机器人有不同DOF、end-effector类型、motion controller（如position control vs. impedance control）
- **Vision异构**：wrist camera vs. third-person view，不同mounting pose，不同光照
- **Action space异构**：absolute pose vs. relative pose，continuous vs. discrete
- **Embodiment异构**：Franka, Kuka, Sawyer, xArm, Aloha, mobile robots等

Previous approaches如RT-1/RT-2、Octo、OpenVLA大多采用 **homogeneous data format**（如统一language conditioning）或只pre-train vision部分，后接proprioception。HPT的核心思想是 **tokenize each embodiment**，把异构性吸收到modular architecture中，让shared trunk学到task-agnostic、embodiment-agnostic的representation。

参考链接：
- Project page: https://liruiw.github.io/hpt
- Open-X Embodiment: https://robotics-transformer-x.github.io
- Octo: https://octo-models.github.io

---

## 2. Architecture详解

### 2.1 三段式模块化设计

HPT将policy network $f_\theta(o) \to a$ 分解为三个sub-networks：

```
[Observation o] → [Stem θ_stem] → [Latent tokens] → [Trunk θ_trunk] → [Features] → [Head θ_head] → [Action a]
                  embodiment-specific     shared               task-specific
```

### 2.2 Stem细节（Figure 3）

**Proprioception Tokenizer**:

输入：第$k$个embodiment的proprioception vector $p \in \mathbb{R}^{d_p^k}$（如7维end-effector pose：$x,y,z,q_w,q_x,q_y,q_z$）

步骤：
1. MLP：$p \to \tilde{p} \in \mathbb{R}^{d}$（$d$为trunk的latent dimension，128~1024）
2. Sinusoidal positional encoding：$PE(\tilde{p})$
3. Cross-attention with $N_p=16$ learnable query tokens $Q_p \in \mathbb{R}^{N_p \times d}$
   - $\text{Attn}(Q_p, PE(\tilde{p}), PE(\tilde{p})) \in \mathbb{R}^{N_p \times d}$

输出：$N_p = 16$个proprioception tokens，维度为$d$。

**Vision Tokenizer**:

输入：image $I \in \mathbb{R}^{H \times W \times 3}$

步骤：
1. Frozen pretrained ResNet18：$I \to F \in \mathbb{R}^{7 \times 7 \times C}$（49个spatial features）
2. Flatten：$F \to \mathbb{R}^{49 \times C}$
3. Linear projection到$d$
4. Cross-attention with $N_v = 16$ learnable tokens

输出：$N_v = 16$个vision tokens。

**Concatenation**:
最终stem输出为 $2N = 32$个tokens（16 proprio + 16 vision），加上modality embedding和sinusoidal positional embedding。

**为什么这样设计？** 这模仿了人类spinal cord的hierarchical sensorimotor control（paper引用[68]）：低层peripheral处理specific motor responses，高层CNS处理abstract stimuli。Stem承担"translation"角色，trunk承担"reasoning"角色。

### 2.3 Trunk

- Architecture: decoder-only Transformer
- Parameters: $3.1M$ (Small) ~ $1.1B$ (Huge)
- Input/Output sequence length相同（保持$2N=32$个tokens）
- Output: pooled feature $\bar{z} = \text{MeanPool}(\text{Trunk}(\text{tokens}))$

Table 1详细参数：

| Model | Depth | Width | Heads | Params |
|-------|-------|-------|-------|--------|
| HPT-Small | 16 | 128 | 8 | 3.1M |
| HPT-Base | 16 | 256 | 8 | 12.6M |
| HPT-Large | 16 | 512 | 8 | 50.5M |
| HPT-XLarge | 32 | 768 | 16 | 226.8M |
| HPT-Huge | 80 | 1024 | 16 | 1.1B |

### 2.4 Head

支持多种架构：
- **MLP head**: 3-layer MLP，input为pooled feature $\bar{z}$
- **Transformer decoder head**: concat learnable action tokens，1D conv regression
- **Diffusion policy head**（DDPM）：用于real-world high-precision tasks

关键设计：action horizon = 8，observation horizon = 4，并采用**random temporal masking**来兼容不同downstream horizon需求。

---

## 3. Training Objective数学详解

### 3.1 主公式

$$\min_\theta \sum_{k=1}^{K} \mathcal{L}(\theta_k^{\mathrm{stem}}, \theta^{\mathrm{trunk}}, \theta_k^{\mathrm{head}}; \mathcal{D}_k) \tag{1}$$

变量解释：
- $K$: dataset总数（最多52）
- $\theta_k^{\mathrm{stem}}$: 第$k$个embodiment专属stem参数
- $\theta^{\mathrm{trunk}}$: 所有embodiment共享的trunk参数（单一集合）
- $\theta_k^{\mathrm{head}}$: 第$k$个task专属head参数
- $\mathcal{D}_k = \{\tau^{(i)}\}_{1 \leq i \leq M_k}$: 第$k$个dataset，包含$M_k$条轨迹
- $\tau^{(i)} = \{o_t^{(i)}, a_t^{(i)}\}_{1 \leq t \leq T}$: 第$i$条轨迹，长度$T$

**Parameter set**:
$$\theta = \bigcup_{k=1}^K \{\theta_k^{\mathrm{stem}}, \theta_k^{\mathrm{head}}\} \cup \theta^{\mathrm{trunk}}$$

### 3.2 Loss function

$\mathcal{L}$是behavior cloning loss，使用**Huber loss**（而非MSE）：

$$\mathcal{L}_{\text{Huber}}(a, \hat{a}; \delta) = \begin{cases} \frac{1}{2}(a - \hat{a})^2 & \text{if } |a - \hat{a}| \leq \delta \\ \delta(|a - \hat{a}| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$

其中$\delta = 0.1$（empirical best）。Huber loss的好处：对"dificult frames"（如contact瞬间的大梯度）robust，同时对lengthy easy部分保持平滑。

Actions在训练前按dataset statistics做element-wise normalization到$[-1, 1]$，保证不同embodiment的loss scale一致。

### 3.3 数据采样策略

采用 **temperature-based inverse probability sampling**:

对于第$k$个dataset，采样概率：
$$p_k = \frac{\sqrt{M_k}}{\sum_{j=1}^K \sqrt{M_j}}$$

这防止large datasets（如RoboNet的47k trajectories）dominate整个epoch。这是multitask learning的standard practice。

### 3.4 Optimization

- Optimizer: AdamW
- Weight decay: 0.05
- Base learning rate: $2 \times 10^{-4}$，cosine schedule with warmup
- Learning rate按batch size成比例scaling（linear scaling rule）
- Batch size: 256（default）~ 2048（scaled）

---

## 4. Scaling Behaviors实验详解

这是paper最有趣的部分，类似LLM scaling laws [Kaplan et al. 2020, Hoffmann et al. 2022]在robotics的复现。

### 4.1 Data Scaling (Figure 5a)

实验设置：
- X轴：每个dataset最多trajectories数（10, 100, 1000, 10000, 100000）
- Y轴：final validation loss（averaged over 27 held-out datasets）
- 蓝线：HPT-Small fixed budget
- 红线：HPT-L with compute scaling（每10x数据增加，model size 4x，batch size 2x）

关键发现：
- 当data单独增加但model size固定 → 早期plateau（~1000 traj/dataset）
- 当data + model + compute同步scaling → 稳定下降
- 这与 [Hoffmann et al. 2022] 的Chinchilla findings一致：data和compute需要tandem scaling

### 4.2 Dataset Diversity Scaling (Figure 5b)

- 固定10个dataset作为evaluation subset
- 增加pre-training dataset数量：10, 20, 30, 40, 52
- 固定2 epochs
- 在4个model sizes上重复4次

发现：**更多embodiment diversity → 更低validation loss**，暗示trunk确实学到了embodiment-agnostic representation。

### 4.3 Model Scaling (Figure 7)

- 固定27 datasets, 1000 traj/dataset
- Model从1M到1B参数
- Red line：同时增加data（到170k）和batch size（256→2048）
- Blue line：fixed data

发现：data和compute充足时，model size scaling有效；depth vs. width scaling无显著差异。

### 4.4 Epoch Scaling (Figure 6)

- Fixed 27 datasets, 1000 traj/dataset
- 增加batch size → 实际增加tokens seen
- 大batch size还能 **reduce variance** from heterogeneous data sampling

---

## 5. Heterogeneous Pre-training Domain Coverage

HPT不仅用real robot teleop data，还探索了：

### 5.1 Simulation datasets (7个)
- Drake [Fleet-Tools]
- MuJoCo [MetaWorld, RoboMimic]
- Isaac Sim [ARnold]
- PyBullet [TriFinger, Grasping]
- Sapien [ManiSkill]
- Flex [Deformable]

### 5.2 Human video datasets
- EPIC-Kitchens: 用2D bounding box center作为pseudo-proprioception，frame difference作为pseudo-action
- PoCo: 3D hand positions via ICP作为proprio，6-DoF poses作为action

### 5.3 Deployed robot datasets
- FrodoBot-2K: mobile robot driving in the wild，IMU作为proprio，linear/angular velocity作为action

Figure 8显示joint pre-training带来小幅提升，说明HPT能吸收 **cross-domain heterogeneity**。

---

## 6. Transfer Learning实验

### 6.1 Simulation Benchmarks

**Meta-world, RoboMimic, Fleet-Tools**:
- 训练数据：20-100 trajectories/task
- 评估：50 episodes with random init
- 5 independent runs averaged

Baselines对比：
1. No Trunk: 只用stem + head，from scratch
2. From Scratch: 完整network从零训练
3. Pretrained Frozen: 冻结pre-trained trunk
4. Pretrained Finetuned: 端到端finetune
5. Pretrained Finetuned (HPT-XL): 用更大pre-trained trunk

结果（Figure 10a）：pre-trained models一致优于baselines，HPT-XL > HPT-L > HPT-B > No Trunk ≈ From Scratch。

### 6.2 Simpler Benchmark (Figure 10b)

对比RT-1, RT-2, Octo等generalist models：
- Google EDR embodiment
- 3 tasks: Close Drawer, Move Near, Pick Coke Can
- 300 episodes total

### 6.3 Real-world Experiments (核心亮点)

4个tasks（Figure 11, 12）：
1. **Sweep Leftover**: 把granular piles扫到plate里
2. **Fill Water**: 倒水到bowl
3. **Scoop Food**: 舀狗粮
4. **Switch Insertion**: PCB板精确插拔（3 pins）

两个different embodiments，different camera configs，different action spaces。

**Ablation Table 3 (Sweep Leftover)**:

| Method | Success (%) |
|--------|-------------|
| From Scratch No Prop. | 26.7±3.3 |
| From Scratch | 43.3±3.8 |
| R3M | 50.0±3.0 |
| Voltron | 46.7±3.8 |
| VC-1 | 53.3±2.6 |
| No Prop. Finetuned | 63.3±2.6 |
| **HPT-B Finetuned** | **70.0±3.0** |
| **HPT-XL Finetuned** | **76.7±3.3** |

关键发现：
- HPT比vision-only pretraining（R3M, Voltron, VC-1）高约20%
- **Joint proprioception + vision pretraining** > vision-only + post-hoc proprioception
- 只需finetune ~2%参数（stem + head ~3MB），其余trunk frozen

---

## 7. Architecture Intuition

### 7.1 为什么是stem-trunk-head结构？

类比人类sensorimotor system [Seminara et al. 2023, ref 68]:
- **Spinal cord**: 局部reflex circuits，处理specific motor responses → Stem
- **Brain/CNS**: 抽象reasoning，shared planning → Trunk
- **Motor cortex output**: task-specific motor commands → Head

### 7.2 为什么用cross-attention tokenize？

- Perceiver [Jaegle et al. 2021]的思想：用learnable query tokens从variable-length input提取fixed-length latent
- 好处：
  - **Fixed context length** → 享受transformer scaling benefits
  - **Variable input**：不同DOF的proprioception、不同camera数量都能处理
  - **Information bottleneck**：16个tokens强制compress到essential信息

### 7.3 为什么proprioception和vision jointly pretrain？

之前的工作（R3M, VC-1, Voltron, Octo）的局限：
- 只pretrain vision backbone
- Proprioception作为post-hoc feature concatenation
- 问题：vision features对proprioception context不敏感

HPT的优势：
- Vision tokens attend to proprioception tokens in trunk
- 学到的representation是 **cross-modal aligned**

---

## 8. Limitations与Future Directions

Paper自承认的局限：
1. **Data curation**: 简单balanced sampling，没有quality filtering
2. **Supervised objective only**: 没探索self-supervised或RL objectives
3. **Scale**: 1B参数 vs. LLM的几百B，仍未达到compute-optimal
4. **Evaluation**: 只测short-horizon manipulation，没测long-horizon、bimanual、mobile
5. **Reliability**: 最高76.7%，远未达到production-grade >90%

Failure modes (Figure 19):
- Spatial overshoot/undershoot：倒水倒到杯子前面
- 可能因为data quality和vision encoder精度不足

---

## 9. 与相关工作的对比

Table 5对比：

| Method | #Dataset | #Traj | Model Size | Hetero Prop |
|--------|----------|-------|------------|-------------|
| RT-1 | 1 | 0.1M | 16M | ✗ |
| RT-2X | 12 | - | 55B | ✗ |
| Octo | 25 | 0.8M | 93M | ✗ |
| OpenVLA | 25 | 1M | 7B | ✗ |
| **HPT** | **52** | 0.2M | 1.1B | **✓** |

HPT是第一个explicitly handle **proprioception heterogeneity**的generalist policy。

关键区别：
- Octo [Walke et al.]: 用256 tokens，unified observation/action space，需手动align
- RT-X [Open X]: language conditioning统一，但牺牲proprioception
- OpenVLA [Kim et al.]: VLA framework，依赖language
- HPT: modular stems，保留original heterogeneity，靠architecture align

参考链接：
- LeRobot implementation: https://github.com/huggingface/lerobot
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- ACT (Aloha): https://tonyzhaozh.github.io/aloha/

---

## 10. 个人思考与Intuition

### 10.1 HPT的本质

HPT做的事情类似于 **multimodal foundation models（如ImageBind、Flamingo）在robotics的应用**。把不同embodiment视为不同"modality"，通过learnable projection layers对齐到shared space。

### 10.2 Tokenization as Inductive Bias

固定数量的tokens（16+16=32）是一个strong inductive bias。这强制模型学会compress information，类似LLM中fixed vocabulary size。如果设太大 → overfit；太小 → information loss。16是经验值，paper没有详细ablate。

### 10.3 MoE视角

Paper提到与MoE的联系：每个embodiment有自己的stem/head expert，trunk是shared。但这与标准MoE不同：
- Standard MoE: dynamic routing per input
- HPT: static routing per embodiment（dataset identity决定）

未来方向可能是 **soft routing**，让相近embodiment共享更多参数。

### 10.4 为什么validation loss可作为metric？

Paper引用NLP practice（perplexity）。但在robotics中：
- Training loss gap vs. closed-loop success rate gap
- 类似LLM的perplexity vs. downstream task gap

这个gap在robotics中可能更大（因为closed-loop execution的compounding errors）。但作为 **scaling indicator**仍有价值。

### 10.5 未来的关键问题

1. **Scaling ceiling在哪？** 1B参数还不够converge，需要更大scale验证
2. **Self-supervised pretraining?** 当前只用action labels，未来可能结合world model objectives
3. **Cross-embodiment transfer efficiency?** Paper中new embodiment只需2% trainable params，理论上可few-shot
4. **Long-horizon tasks?** 当前只测短horizon，需要hierarchical extension
5. **3D, tactile, audio modalities?** Stem架构理论上可扩展，但实际integration未验证
6. **URDF作为inductive bias?** Paper在future work提到，可能用robot morphology作为graph structure prior

---

## 11. 实用资源

- **代码**: https://liruiw.github.io/hpt (open-source)
- **LeRobot实现**: https://github.com/huggingface/lerobot
- **Open-X dataset**: https://robotics-transformer-x.github.io
- **DROID**: https://droid-dataset.github.io/
- **Octo (baseline)**: https://octo-models.github.io
- **Simpler benchmark**: https://simpler-env.github.io
- **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu/
- **Scaling laws paper**: https://arxiv.org/abs/2001.08361
- **Chinchilla paper**: https://arxiv.org/abs/2203.15556
- **Perceiver IO**: https://arxiv.org/abs/2107.14795

---

## 12. 总结

HPT的核心贡献是把robotics的heterogeneity问题转化为一个 **architecture design problem**，而非data engineering问题。通过modular stem-trunk-head和cross-attention tokenization，让一个shared transformer trunk能从52个heterogeneous datasets中学习shared representation。

关键empirical findings:
1. **Heterogeneous pretraining scales**: data + model + compute tandem scaling有效
2. **Joint proprioception-vision pretraining > vision-only**: ~20% improvement
3. **Modular transfer**: 只需finetune 2%参数即可适配new embodiment
4. **Cross-domain data helps**: simulation + human video提供complimentary signal

这为robotic foundation models指明了一条不同于VLA（language-conditioned）的路径：**architecture-level alignment rather than data-format-level unification**。

未来的关键看这条路线能否scale到10B+参数和10M+ trajectories，以及是否能扩展到locomotion、bimanual、mobile manipulation等更challenging的setting。
