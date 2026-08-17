---
source_pdf: Open X-Embodiment.pdf
paper_sha256: 13f16aff5afdee583dc25b3c570e9bb0cf6baefb59b539928d919588cd8d9246
processed_at: '2026-08-06T00:01:32-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的，Karpathy，让我们抛开学术黑话，用最直白的方式拆解这篇 paper。这篇 paper 的核心目的就是验证一个你肯定天天在念叨的信仰：**The Bitter Lesson 在 robotics 领域到底能不能生效？**

### 1. 终极痛点：Robotics 的 "ImageNet Moment" 缺席

在 NLP 和 CV 里，大家早就习惯了先拿 web 上的海量数据 (WebLI, LAION) 预训练一个 generalist model，然后再在下游任务上微调。但在 robotics 里，过去十年的现状是：每个 lab 买个机械臂，自己在上面采几千条 trajectory，训练一个 model，发一篇 paper。下个 lab 换个机械臂，又从零开始。

这导致 robotics 数据极度碎片化。最大的 robotics dataset 规模只有几百万，跟 NLP 的 billion 级 tokens 相比，连零头都算不上。更致命的是，单 lab 的数据 narrow 到极点——就一个场景、一套光照、一种物体。

Open X-Embodiment (OXE) 这篇 paper 想做的就是把这帮孤岛联合起来：凑齐 21 个机构、22 种机器人、100 万条 trajectory，看能不能像 CV 一样训出一个通用的 foundation model。

### 2. 异构数据怎么混合：粗暴对齐 + 暴力解码

最大的工程挑战是：WidowX 机械臂和 Franka 机械臂的底层控制指令完全不一样，关节角度、运动学结构全都不通。怎么把它们放进一个 batch 里训练？

paper 的做法非常 "lazy" 但极其 scalable：

**Action Space 转换**
抛弃底层 joint control，统一转成 7-DoF end-effector representation：
$$a = (x, y, z, \text{roll}, \text{pitch}, \text{yaw}, \text{gripper})$$
- $x, y, z$: 末端执行器在空间中的位置坐标。
- $\text{roll}, \text{pitch}, \text{yaw}$: 末端执行器的姿态欧拉角。
- $\text{gripper}$: 夹爪的开合程度。

**Per-dataset Normalization 与 Tokenization**
每个 dataset 自己做自己的归一化，然后把连续值砍成 256 个 bins。也就是把每个维度的 action 变成一个 0-255 的整数。最后拼成一串文本，比如 `"1 128 91 241 5 101 127"`。

**关键的 "不作为"**
paper 明确指出，他们**没有**对齐不同 dataset 的 camera 视角，也**没有**对齐坐标系（有的 dataset 是绝对坐标，有的是速度相对坐标）。同一个 action token `(x=100)`，在 WidowX 和 Franka 上代表的物理运动完全不同。

这里的 intuition 是：与其花巨大人工成本去标定和映射所有机械臂的物理参数，不如把对齐的活儿扔给 transformer。只要数据量够大、model 容量够大，网络自己能学会"当前画面如果是 WidowX，token 100 就往左跑；如果是 Franka，就往前跑"。这种软对齐策略极大地降低了数据并入成本。

### 3. 为什么 RT-1 (35M) 扛不住，必须上 RT-2 (55B)

实验里最有意思的发现是 capacity bottleneck。

当他们在 5 个小规模 dataset 上做实验时，35M 参数的 RT-1-X 表现极好，比原作者自己训的 model 平均提升 50%。小数据集本来样本就少，自己训容易 overfit，混入其他机器人的数据相当于注入了强大的先验。

但是，在 Bridge 和 RT-1 这两个本身就已经是大规模的 dataset 上，RT-1-X 居然 underfitting 了，性能反而比只在自己数据上训练的 RT-1 还差！

原因很简单：RT-1 太小了。EfficientNet-B3 提取视觉特征，过一个 35M 的 Transformer，它根本没有足够的参数容量去同时记忆 9 种机器人的控制映射，还要在 Bridge 这种大数据集上做到精细泛化。

所以必须上 RT-2-X。它基于 PaLI-X，参数量直接拉到 55B。

#### RT-2-X 的架构直觉

RT-2-X 根本没改 VLM 的架构，它直接把 VLM 当成了一个多模态大脑。图像过 ViT 提取 patch tokens，文本指令过 UL2 编码。然后那串 action tokens `"1 128 91 241 5 101 127"` 就被当成了普通的文本输出。

在 RT-1 里，视觉和语言的融合用的是 FiLM (Feature-wise Linear Modulation) 机制，公式如下：
$$\text{FiLM}(F | L) = \gamma(L) \odot F + \beta(L)$$
- $F \in \mathbb{R}^{H \times W \times C}$: 视觉 feature map。
- $L$: language embedding。
- $\gamma(L), \beta(L)$: 由语言生成的 scale 和 shift 参数，对视觉特征做 channel-wise 的仿射变换。

这是一种很精巧但能力有限的融合方式。而在 RT-2-X 里，由于 55B 参数的 VLM 骨干本身就在 web-scale 数据上学过极其丰富的 cross-modal grounding，所有模态直接在 attention 层里互相 attend，这种暴力融合的效果远超 FiLM。

### 4. 最震撼的实验：Emergent Skills (跨机身技能涌现)

Table II 是整篇 paper 的灵魂。他们在 Google Robot 上做实验，让它做 RT-1 dataset 里**从来没有过**的技能（这些技能只存在于 WidowX 的 Bridge dataset 里）。

我们来看这几个 key rows：

| Row | Setup | Emergent Skills 成功率 |
|-----|-------|-----------------|
| 1 | RT-2 (55B) 只在 Google Robot 上训 | 27.3% |
| 2 | RT-2-X (55B) 混合 9 种机器人训练 | **75.8%** |
| 3 | RT-2-X 但剔除 Bridge dataset | 42.8% |
| 4 | RT-2-X (5B) 混合训练 | 44.4% |
| 6 | RT-2-X (5B) 从零开始训（无 Web pretrain）| 0% |

**Row 1 vs Row 2**：同样 55B 参数，同样 web pretrained，只要在训练时混入 WidowX 的数据，Google Robot 突然就会做 WidowX 的任务了，成功率从 27.3% 飙到 75.8%。技能直接跨过了 embodiment 的壁垒。

**Row 3**：为了防止是别的数据起作用，他们把 Bridge dataset 删了再训。结果成功率掉到 42.8%。证明那 75.8% 里的高出来的部分，确确实实是从 WidowX 数据里学来的。

**Row 2 vs Row 4**：模型从 55B 缩小到 5B，成功率从 75.8% 掉到 44.4%。Capacity bottleneck 再次显现，小脑子装不下跨机身的复杂映射。

**Row 6**：5B 模型不用 web 数据预训练，直接从零开始在 robotics 数据上训。成功率 0%，泛化能力 1%。这彻底印证了你常说的：robotics 数据再大，也拼不过 web 数据的广度。没有 web pretrain 提供 visual concept grounding，model 根本不知道 "apple" 是什么，就根本学不会 "pick up apple"。

### 5. 总结：这篇 paper 到底赢在哪

从直觉上看，这篇 paper 真正的胜利**不在于**网络架构（FiLM 或者 Transformer），**更不在于**那个把 7 维 action 拆成 256 bins 的简陋 tokenization。

它的胜利在于证明了：**Robotics 可以走 scaling law。**

只要模型够大（55B），数据够杂（9 种机器人），预训练够好（Web VLM），Transformer 就能自己解开盘根错节的 embodiment 差异。这为后续所有的 VLA (Vision-Language-Action) 工作（比如 OpenVLA, Octo, $\pi_0$）铺平了道路。它让整个 robotics 社区意识到：不要再死磕单任务强化学习或小规模模仿学习了，去堆数据，去堆算力。

### 6. Web Links 参考索引

为了方便你直接查阅相关的源码和后续演化工作，我整理了几个关键链接：

- **Open X-Embodiment 官方项目主页**: https://robotics-transformer-x.github.io
- **Dataset 与 RLDS 格式源码**: https://github.com/google-deepmind/open_x_embodiment
- **RT-2 (VLA 基础模型原版)**: https://robotics-transformer2.github.io
- **OpenVLA (基于 OXE 训练的开源 7B VLA)**: https://openvla.github.io
- **Octo (基于 OXE 的通用 Transformer 策略)**: https://octo-models.github.io
- **RLDS 数据格式标准 (TensorFlow Datasets)**: https://github.com/google-research/rlds

如果你对后续 $\pi_0$ 怎么用 flow matching 替代这种 256 bins 的粗糙 tokenization 感兴趣，或者想聊聊 OpenVLA 怎么把 55B 压缩到 7B 还能保持性能，我们可以继续往下挖。

---

# Open X-Embodiment: Robotic Learning Datasets and RT-X Models 深度解析

## 1. 这篇 paper 在解决什么问题

Open X-Embodiment (OXE) paper 想回答一个根本性的问题：robotics 能不能复现 CV 和 NLP 里的 "pretrain on diverse data, finetune on specific task" 范式？

Karpathy 你自己说过 "the bitter lesson"，这是这一 lesson 在 robotics 的具体实践尝试。具体来说：

- CLIP [1] 在 web-scale image-text pairs 上 pretrain → 下游 open-vocab classification 远超固定词表
- GPT 系列 [2,3] 在海量文本上 pretrain → 几乎所有 NLP 任务都能用 finetune 或 in-context learning 解决
- RT-1 [8] / RT-2 [9] 这些 robotics model，但数据局限在单一 embodiment

robotics 的困境很特殊：每个 lab 自己采数据，规模只有几千到几万 trajectories，远小于 CV (5-18M images [4,5]) 和 NLP (1.5B-4.5B tokens [6,7])。而且单 lab 数据 narrow——单一 robot、单一 environment、单一 task suite。

OXE 的核心 insight：**单个 dataset 太窄，但所有 dataset 的 union 可以覆盖更广的 variations**。这就是 X-embodiment 的 idea。

参考链接：
- 项目主页: https://robotics-transformer-x.github.io
- RT-1 paper: https://arxiv.org/abs/2212.06817
- RT-2 paper: https://arxiv.org/abs/2307.15818

## 2. Open X-Embodiment Dataset：规模与多样性

### 2.1 规模统计

| 维度 | 数值 |
|------|------|
| Trajectories 总数 | 1M+ |
| Robot embodiments | 22 种 |
| 参与机构 | 21 所 |
| 原始 datasets | 60 个 |
| 来源 lab | 34 个 |
| Skills | 527 种 |
| Tasks | 160,266 |

数据范围从 single arm 到 bimanual 到 quadruped（带 manipulation附件），形态非常异构。

### 2.2 数据存储格式：RLDS

采用 RLDS (Reinforcement Learning Datasets) 格式 [119]，本质上是序列化的 tfrecord。RLDS 的几个关键设计：

- 支持任意 action space 维度（不同 robot 的 DoF 不同）
- 支持任意 sensor modality（RGB、depth、point cloud，camera 数量可变）
- 主要 DL framework（TF、JAX、PyTorch）都能高效并行加载
- episode-based 结构：`Episode → Steps → (observation, action, reward, ...)`

参考: https://github.com/google-research/rlds

### 2.3 Dataset 分布的 long tail

Fig.2 的几个子图给出关键观察：

- **Datasets per embodiment**：Franka 最多，因为大量 lab 用 Franka Panda
- **Scenes per embodiment**：Franka 视觉 scene 最多样（因为多 lab 多场景）
- **Trajectories per embodiment**：xArm 和 Google Robot 最多（因为 RT-1、QT-Opt 等几个大 dataset）
- **Skills**：pick-and-place 占主导，long tail 包含 wiping、assembling、cable routing 等
- **Objects**：从 appliance 到 food 到 utensils

**Intuition**：datasets 分布严重不均，Franka 的 scene diversity 最丰富（视觉泛化潜力大），xArm/Google Robot 的 trajectory 最多（行为学习潜力大）。

## 3. 数据对齐：让不同 robot 的数据可共训练

### 3.1 Observation space 对齐

每个 dataset 选一个 **canonical camera view**，resize 到 common resolution。这里 paper 没明确说统一到多少分辨率，但 RT-1 用 320×320，RT-2 用 224×224 的 ViT 输入。

关键：**camera pose 不对齐**——同一个 camera 在不同 robot 上安装位置不同，所以同一个 "object 在画面中央" 在不同 robot 上对应不同 world coordinates。

### 3.2 Action space 对齐

每个 dataset 的原始 action 转成 **7 DoF end-effector representation**：

$$a = (x, y, z, \text{roll}, \text{pitch}, \text{yaw}, \text{gripper})$$

其中：
- $(x, y, z)$：end-effector 位置
- $(\text{roll}, \text{pitch}, \text{yaw})$：end-effector 姿态（Euler 角）
- $\text{gripper}$：gripper 开合度

**关键设计**：每个 dataset 在 tokenization 之前 per-dataset normalize。这里 normalize 不是 z-score，而是为了把每个 dataset 的 action range 对齐到统一 discretization bins。

Per-dataset normalization 形式上可以写为：

$$a^{(d)}_{\text{norm}} = \frac{a^{(d)} - \mu^{(d)}}{\sigma^{(d)}}$$

其中 $\mu^{(d)}, \sigma^{(d)}$ 是 dataset $d$ 的 per-dim 统计量。或者用 min-max normalization 到 $[0, 1]$。

然后 tokenization：

$$a_{\text{token}} = \text{clip}\left(\left\lfloor a_{\text{norm}} \times 256 \right\rfloor, 0, 255\right)$$

得到 256 bins 中的某个整数 token。

### 3.3 关键的"不完美对齐"哲学

paper 在 Section IV-A 明确说：

> "the same action vector may induce very different motions for different robots."

这是设计上的让步，而非缺陷。**不对齐 coordinate frame，也不对齐 absolute vs velocity 控制模式**。原因：
- 强行对齐需要大量手工 calibration
- 不同 robot 的 kinematics 本质不同（Franka 7-DoF arm vs WidowX 4-DoF arm vs bimanual）
- 让 model 自己学到"对当前 embodiment 怎么解释 action token"

**Intuition**：这其实是一个软对齐策略——action 是一个语义化的 affordance 指示，但具体执行由 embodiment 决定。类似人类学开不同车，"踩油门"这个指令在不同车上力度不一样。

## 4. RT-1-X 架构详解

RT-1 [8] 是 Google 2022 年底提出的 Robotics Transformer，专门为 robotics control 设计。RT-1-X 在架构上完全没改，只是用了 OXE 多 embodiment 数据训练。

### 4.1 整体 pipeline

```
15 帧 image history (each 320×320×3)
        ↓
EfficientNet-B3 (ImageNet pretrained)
        ↓ 9×9×384 feature map
FiLM conditioning ← USE(text instruction)
        ↓
81 vision-language tokens (9×9 flatten)
        ↓
Token Learner (压缩到 8 tokens)
        ↓
Causal Transformer (8 layers, 256-dim)
        ↓
8 action tokens, each 256-way classification
```

参数量：35M

### 4.2 EfficientNet-B3 作为视觉 backbone

EfficientNet [117] 用 compound scaling：

$$\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi$$

约束 $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$。B3 配置对应 $\phi=3$。

为什么用 EfficientNet？因为 RT-1 要在 robot 上本地跑 3-10 Hz，效率优先。EfficientNet-B3 在 ImageNet top-1 81.6% 但 FLOPs 只有 1.8B。

### 4.3 FiLM 调制

FiLM (Feature-wise Linear Modulation) [116] 是关键的语言-视觉融合机制：

$$\text{FiLM}(F | L) = \gamma(L) \odot F + \beta(L)$$

其中：
- $F \in \mathbb{R}^{H \times W \times C}$：visual feature map
- $L$：language embedding（从 USE 得到）
- $\gamma(L), \beta(L) \in \mathbb{R}^C$：从 $L$ 通过 MLP 投影得到的 scale 和 shift 参数
- $\odot$：channel-wise 乘法

**Intuition**：language 不是 concat 进去，而是作为"调节器"改变视觉特征的 scale 和 bias。这比简单 concat 更 parameter-efficient，也更强。比如 "pick up the apple" 这个 instruction 让 apple 区域的 feature 被 amplify，background 被 suppress。

### 4.4 USE (Universal Sentence Encoder)

USE [120] 把 natural language instruction 编码成 512-d embedding。USE 在 web-scale 文本上预训练（DAN / Transformer 两种变体），是 sentence-level embedding 的工业标准。

注意：RT-1 用 frozen USE，没有 co-train。这和 RT-2 用 LLM backbone 形成对比。

### 4.5 Action tokenization 与 loss

action 8 维（7 EE + 1 terminate），每维 256 bins → 8 个 categorical 输出。Loss 是标准 categorical cross-entropy：

$$\mathcal{L}_{\text{CE}} = -\frac{1}{N \cdot T \cdot D} \sum_{n=1}^{N} \sum_{t=1}^{T} \sum_{d=1}^{D} \sum_{k=0}^{255} y_{n,t,d,k} \log p_{n,t,d,k}$$

其中：
- $N$：batch size
- $T$：序列长度（一般 1，因为是 stateless prediction）
- $D = 8$：action 维度
- $k$：256 个 bin index
- $y_{n,t,d,k}$：one-hot ground truth
- $p_{n,t,d,k}$：softmax 输出概率

注意：每维独立分类，没有建模 action 维度间相关性（这其实是局限，后续 Octo 等用 diffusion action head 改进）。

## 5. RT-2-X 架构详解

RT-2 [9] 是 VLA (Vision-Language-Action) model，核心 idea：把 action 看成 text token，复用 VLM 的能力。

### 5.1 Action as text

一个 7 维 EE action 转成字符串：

$$a = (a_1, ..., a_7) \rightarrow \text{"1 128 91 241 5 101 127"}$$

每个数字是 0-255 的 token id，作为 VLM vocabulary 的一部分。这样不需要改 VLM 架构，只需要 co-fine-tune。

### 5.2 PaLI-X backbone

RT-2-X 用 PaLI-X [121] 作为 backbone：
- Visual encoder: ViT [124]
- Language model: UL2 [125]
- 参数量：55B（最大版本）
- Pretrain data: WebLI (web-scale multilingual image-text)

UL2 是 unifying language learning paradigm 的 LLM，混合了 denoising objective (类似 T5) 和 causal LM。PaLI-X 把 ViT 和 UL2 接成 encoder-decoder VLM。

### 5.3 Co-fine-tuning

这是 RT-2 的核心训练技巧。不是单纯用 robotics data finetune，而是混合 VLM pretrain data 和 robotics data 一起训练：

$$\mathcal{L}_{\text{total}} = \lambda_{\text{VLM}} \mathcal{L}_{\text{VLM}} + \lambda_{\text{robot}} \mathcal{L}_{\text{robot}}$$

paper 中提到 "approximately one to one split"，即 $\lambda_{\text{VLM}} : \lambda_{\text{robot}} \approx 1:1$（按 batch 内 sample 数量）。

**Intuition**：保留 VLM 的 web knowledge（visual concept、language understanding），同时让 model 学到 action mapping。如果只 finetune robotics data，会 catastrophic forgetting。

### 5.4 为什么 RT-2-X 比 RT-1-X 强

不只是参数量。RT-2-X 的关键是 web knowledge 直接注入到 policy network 里。RT-1-X 的 EfficientNet 只是 ImageNet 分类 pretrain，没有 language reasoning 能力。

## 6. 实验设置

总 evaluation：**3600 trials，6 个 robot**。这是 real-world robotics paper 里最大规模的 evaluation 之一。

### 6.1 数据 mixture

训练用 9 个 manipulators，来自 RT-1、QT-Opt、Bridge、Task Agnostic Robot Play、Jaco Play、Cable Routing、RoboTurk、NYU VINN、Austin VIOLA、Berkeley Autolab UR5、TOTO、Language Table。

注意：训练用 9 个，比 dataset 全量 22 个少。原因是数据集随时间在扩，做实验时是当时的全部数据。

### 6.2 Baselines

每个 evaluation setting 两个 baseline：

1. **Original Method**：dataset 原作者训练的 model，只用本 dataset
2. **RT-1**：用同样数据集单独训练 RT-1（control architecture）

这样能分离两个 factor：(a) co-training 的数据多样性的好处；(b) architecture 的好处。

## 7. 实验结果深度分析

### 7.1 In-distribution performance (Fig.4, Table I)

#### Small datasets（数据稀缺，5 个 setting）

RT-1-X 比 Original Method 在 4/5 个 dataset 上更好，平均提升 50%。

| Dataset | Original Method | RT-1 (single) | RT-1-X |
|---------|-----------------|---------------|--------|
| Kitchen Manipulation (Jaco) | ~ baseline | 类似 | 大幅提升 |
| Cable Routing | ~ baseline | 类似 | 大幅提升 |
| NYU Door Opening | ~ baseline | 类似 | 大幅提升 |
| Autolab UR5 | ~ baseline | 类似 | 持平 |
| Robot Play | ~ baseline | 类似 | 大幅提升 |

**Intuition**：小 dataset (几百到几千 traj) 单独训练容易 overfit 或欠拟合。X-embodiment co-training 相当于免费送了大量 prior，让 small dataset 也能学到 reasonable policy。

#### Large datasets (Table I)

| Evaluation | Original | RT-1 | RT-1-X | RT-2-X (55B) |
|------------|----------|------|--------|-------------|
| Bridge (Stanford IRIS, WidowX) | 13% | 40% | 27% | **50%** |
| Bridge (UCB RAIL, WidowX) | 13% | 30% | 27% | 30% |
| RT-1 6 skills (Google Robot) | - | 92% | 73% | 91% |

**关键观察**：RT-1-X 在大数据集上 underfitting！这是反直觉的——你以为加更多数据应该更好。但 RT-1 只有 35M 参数，无法 absorb 这么多异构数据。

而 RT-2-X (55B) 在 Bridge (Stanford) 上 50%，比 RT-1 (single) 的 40% 还高 10 个点。

**Intuition**：model capacity 和 data scale 必须匹配。这其实是 scaling law 的体现——加 data 不免费，需要 model 容量承接。Karpathy 你应该很熟悉这个。

### 7.2 Emergent skills (Table II, Fig.5)

这是最 surprising 的实验。Google Robot 上测试 Bridge dataset 里的 skills（Google Robot 原数据没有的）。

| Row | Model | Size | History | Web Co-train | Initial | Emergent Skills | Generalization |
|-----|-------|------|---------|--------------|---------|-----------------|----------------|
| 1 | RT-2 | 55B | none | Yes | Web-pretrained | 27.3% | 62% |
| 2 | RT-2-X | 55B | none | Yes | Web-pretrained | **75.8%** | 61% |
| 3 | RT-2-X (no Bridge) | 55B | none | Yes | Web-pretrained | 42.8% | 54% |
| 4 | RT-2-X | 5B | 2 | Yes | Web-pretrained | 44.4% | 52% |
| 5 | RT-2-X | 5B | none | Yes | Web-pretrained | 14.5% | 30% |
| 6 | RT-2-X | 5B | 2 | No | From scratch | 0% | 1% |
| 7 | RT-2-X | 5B | 2 | No | Web-pretrained | 48.7% | 47% |

#### 几个关键 ablation 结论

**(a) Cross-embodiment transfer 是真的发生**

Row 1 vs Row 2：同样 55B、同样 web pretrain，区别只是 RT-2-X 用了 OXE 多 robot 数据，RT-2 只用 Google Robot data。Emergent skills 从 27.3% 飙到 75.8%，**3 倍提升**。

**这不是简单加数据**——是 Bridge 数据（WidowX robot）的 skill 知识迁移到了 Google Robot。

Row 3 验证了这一点：把 Bridge 从训练数据中剔除，emergent skills 从 75.8% 降到 42.8%。

**(b) Web pretraining 是基础**

Row 4 vs Row 6：从 scratch 训练的 5B model，emergent skills 是 0%，generalization 1%。基本没学到东西。

Row 4 vs Row 7：都 web-pretrained，区别是 row 7 没有 co-train VLM data，emergent skills 48.7% vs 44.4%（差不多）。说明 web pretrain 作为初始化就够了，但完全不用 web 完全不行。

**Intuition**：robotics data 不够大到从 scratch 学到 generalizable representation。Web data 提供了 visual concept grounding 和 language understanding 的基础。

**(c) Model size 强相关**

Row 2 (55B) vs Row 4 (5B): emergent skills 75.8% vs 44.4%。10x 参数换 1.7x 性能。

**Intuition**：cross-embodiment transfer 需要足够 model capacity 来同时记忆多个 embodiment 的 mapping，并且抽象出 cross-embodiment 的共性。小 model 学不到这种抽象。

**(d) History matters**

Row 4 (history=2) vs Row 5 (history=0): emergent skills 44.4% vs 14.5%。3 倍提升。

**Intuition**：单帧 observation 无法推断 robot dynamics 和物体 state。2 帧 history 提供时序信息，model 能 infer velocity、物体状态变化等。

**(e) Co-fine-tuning vs Fine-tuning**

Row 4 vs Row 7: 都是 5B、history=2、web-pretrained。区别 row 4 是 co-fine-tune (VLM + robot data mix)，row 7 是 fine-tune (从 web-pretrained 出发只训 robot data)。

性能：44.4% vs 48.7%，差不多。这和原 RT-2 paper 结论相反（原 RT-2 说 co-fine-tune 更好）。

paper 解释：因为 RT-2-X 的 robotics data 比 RT-2 用的大很多、多样很多，所以即使 fine-tune 也不会 catastrophic forgetting。**这其实暗示了一个有意思的点**：当 robotics data 多样性达到一定阈值，就不需要 web data 来"防止遗忘"了。

## 8. 重要的 design 决策回顾

让我总结几个 paper 体现的 implicit design philosophy：

### 8.1 软对齐胜过硬对齐

不用统一坐标系、不用统一控制模式（absolute vs velocity）、不强求 sensor 配置一致。只对齐到一个 coarse 的 7-DoF EE representation + 256 bins discretization。

**代价**：model 必须学到 per-embodiment 解释。这要求足够大 model 和足够多 data。

**好处**：dataset onboarding 成本极低，加新 robot 不需要重新设计。

### 8.2 Action as text token 的优雅

RT-2-X 的核心 trick——action 当 text。这避免了：
- 设计专门的 action head
- 处理 multiple action spaces 的复杂分支
- 重新设计 VLM 输出层

代价是 action 维度间相关性被忽视。后续工作 (Octo, π0) 改用 diffusion action head 或 continuous output 解决。

### 8.3 Web pretrain 几乎免费

Web pretrain 提供 visual grounding + language understanding + 一般化抽象能力。从 scratch 训练几乎不可能（row 6 的 0%）。这印证了 NLP/CV 的 lesson：**预训练 representation 是 robotics 必要基础**。

## 9. 与后续工作的关联

虽然 paper 没说，但作为 Karpathy 你应该会联想到后续：

- **Octo** (https://octo-models.github.io)：开源版 RT-X，用 diffusion action head + ViT backbone，更通用
- **π0 / π0-mini** (Physical Intelligence)：VLM backbone + flow matching action generation
- **OpenVLA** (https://openvla.github.io)：7B 开源 VLA，在 OXE 上训练
- **RDT-1B** (https://thu-rdt.github.io)：bimanual 双臂，1B 参数 diffusion transformer
- **CogACT**：用 VLM 编码 action tokens 但不直接输出，而是用 flow head

这些后续工作都把 OXE 作为标准 benchmark。OXE 的 dataset 本身比 paper 里的 model 影响更大——它实际上定义了 robotics 的 "ImageNet moment"。

## 10. 局限与未解决问题

paper Section VI 自陈局限：

1. **没有研究 sensing modality 差异大的 robot**：纯 RGB 到 depth 到 point cloud 跨越没探索
2. **没有研究 generalization 到全新 robot**：只在训练见过的 robot 上做 in-distribution 测试
3. **没有 transfer 发生与否的判别准则**：什么时候 cross-embodiment transfer 会失败？没有理论
4. **Action representation 简化**：7-DoF EE 对 dexterous hand、mobile manipulation 不够

我自己补充几个:

5. **Skill imbalance 严重**：pick-place 主导，long-tail skill 数据极少
6. **Camera view 不对齐**：可能 model 学到的是 per-dataset visual feature 而非真正 invariant representation
7. **没有 task family stratification**：emergent skill 只测了 Bridge → Google Robot 一个方向

## 11. 总结 intuition

如果让我提炼这篇 paper 的核心 intuition：

1. **Data diversity > data scale alone**：OXE 的 1M 不算大，但 22 个 embodiment 的多样性是关键
2. **Model capacity bottleneck**：35M 的 RT-1-X 在大数据上 underfit，55B 的 RT-2-X 才能受益
3. **Web pretrain 是地基**：robotics 数据不足以从 scratch 学到 generalizable representation
4. **软对齐 + 大 model > 硬对齐 + 小 model**：让 model 学 alignment 比人工 alignment 更 scalable
5. **Cross-embodiment transfer 是真实现象**：不是简单数据增加，是 skill 知识在 robot 间迁移

这篇 paper 的历史地位更多在于 **infrastructure**：OXE dataset + RLDS 格式 + 21 个机构合作模式。它开启了 robotics 的 "data scaling" 时代，类似 ImageNet 之于 CV。

参考链接汇总：
- https://robotics-transformer-x.github.io
- https://arxiv.org/abs/2310.08864 (paper)
- https://github.com/google-deepmind/open_x_embodiment
- https://octo-models.github.io
- https://openvla.github.io
- https://robotics-transformer.github.io (RT-1)
- https://robotics-transformer2.github.io (RT-2)
- https://github.com/google-research/rlds
- https://arxiv.org/abs/2104.08745 (RLDS)

如果你想继续深入某个方面——比如 RT-2 的 action tokenization 和后续 diffusion action head 的对比，或者 OXE 数据集的 RLDS 实现细节，或者 OpenVLA 等后续工作怎么 build on 这个 foundation——可以告诉我，我可以再展开讲。
