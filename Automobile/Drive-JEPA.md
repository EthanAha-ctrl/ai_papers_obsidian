---
source_pdf: Drive-JEPA.pdf
paper_sha256: ec48b492e00fa03b012efc1d6ee7670bb8fdaa953eb79f9708a0515e8d0142bd
processed_at: '2026-08-03T23:34:54-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Drive-JEPA 用人话说

## 一、这paper到底在搞什么

Imagine 你要train一个self-driving model。你有tonnes of driving videos和对应的human driver操作。最naive的做法就是behaviour cloning - 拿video当input，human driver的trajectory当target，train一个network去mimic人类。

听起来简单，但有两个很fundamental的问题：

### Problem 1: Vision representation不够好

你用啥encoder去encode那些video frames？以前大家用ResNet在ImageNet上pretrain，但ImageNet是classify cats and dogs的，跟driving没半毛钱关系。后来有人尝试用video generation model去pretrain，但pixel-level reconstruction太expensive了，而且花大量capacity去学怎么generate天空的texture、路边的广告牌，这些东西对planning decision完全没用。

### Problem 2: 单一trajectory supervision太弱

每个driving scene只有一个human driver的trajectory。但imagine一个场景：前面路口，你可以left turn也可以right turn，两条路都valid。但dataset里human driver只走了left turn，所以你的model只能学到left turn这个mode。这就是经典的mode collapse问题 - behaviour cloning的L2 loss会让model collapse到average behaviour，变成一个"indecisive"的driver。

Drive-JEPA用两个trick分别打这两个问题。

## 二、Trick 1: V-JEPA做video pretraining

### 先理解V-JEPA是啥

V-JEPA全称Video Joint-Embedding Predictive Architecture。LeCun搞出来的一套self-supervised video pretraining method。

核心idea其实很intuitive：你给model看一段video，把后面的frames randomly mask掉一些patches，让model去predict那些masked patches的latent representation（不是pixel！是feature）。

用人话说就是：**"看video的前半段，猜后半段的feature长啥样，但不需要还原pixel"**

为什么这么搞？因为：
- Pixel prediction太wasteful - 你花了大量computation去学天空怎么render，但天空对driving decision没用
- Latent prediction让model focus on meaningful dynamics - 车怎么动、lane怎么变化、agent怎么走
- 不需要expensive的pixel-level reconstruction

### V-JEPA怎么防collapse

这里有个tricky的问题：如果你直接predict "未来的feature"，model可以学到trivial solution - 把所有input都map到同一个constant vector，这样predict永远是同一个constant，loss为零。这就是representation collapse。

V-JEPA的防collapse mechanism借鉴了BYOL：
- **Student encoder** $E_\theta$: 处理masked view $x$
- **Teacher encoder** $E_{\bar\theta}$: 处理完整target view $y$，但用stop-gradient $\mathrm{sg}(\cdot)$阻断gradient
- Teacher的参数是student的EMA (exponential moving average)

```
Student sees masked video → predict masked positions' features
Teacher sees full video → provides target features (no gradient)
Loss = L1 distance between student prediction and teacher target (only at masked positions)
```

这个EMA teacher + stop-gradient的combination非常巧妙：
- Student不能直接copy teacher的weights因为没gradient
- Teacher一直在slowly追student，但不会exactly match
- 结果就是representations保持informative而不会collapse

### Drive-JEPA怎么用V-JEPA

他们拿了V-JEPA 2的pretrained weights（Meta在Ego4D等通用video数据集上pretrain的），然后在curated driving video dataset上continue pretrain：

Data curation:
- CoVLA + DrivingDojo + OpenScene 三个dataset混合
- 只用front-view camera（这个choice很有意思，后面讲）
- 8-frame clips, 2Hz, 512×256 resolution
- 总共208 hours

用8张H800 GPU训练50 epochs，约3天。非常efficient compared to Epona那种1.1B参数的video generation model。

**Intuition**: V-JEPA pretraining让encoder学到了driving video的temporal dynamics - 什么东西在动、怎么动、scene怎么evolve。这些features比ImageNet classification features或MAE reconstruction features更适合planning task。

## 三、Trick 2: Multimodal Trajectory Distillation

### 先看naive approach有啥问题

他们用了一个叫"proposal-centric"的planner架构（借鉴自iPad paper）：

1. 初始化32个learnable query proposals $\mathbf{Q}_0 \in \mathbb{R}^{32 \times M \times D}$
2. Iteratively refine这些proposals，用deformable attention从BEV features采样信息
3. 每个proposal最终decode成一条trajectory

训练loss用的是Social GAN那种min-over-N loss:
$$\mathcal{L}_{traj} = \sum_\ell \lambda^{L-\ell-1} \min_n \| W_t - \tilde{W}_\ell^{(n)} \|_2$$

翻译成人话：32个proposals里，找跟human trajectory最近的那个，只惩罚那个。这样鼓励diversity - 不同proposals可以specialize到不同modes。

**但问题来了**：你只有一个human trajectory $W_t$作为target。所以32个proposals最终都会collapse到human trajectory附近的一个小区域。虽然loss说"找最近的"，但因为supervision signal只有一个mode，proposals没有incentive去explore其他valid modes。

### MTD的核心idea：用simulator当多模态teacher

这个idea其实很simple也很clever：

1. 构建一个trajectory vocabulary - 把dataset里100k+ trajectories用k-means cluster成8192个prototypical trajectories
2. 对每个training scene，用NAVSIM v2的rule-based simulator评估这8192条trajectory的quality（EPDMS score）
3. 选score > 0.95的trajectory作为pseudo-teachers
4. 这些pseudo-teachers就是simulator认为"valid"的multiple trajectories

**为什么这样能work？** 因为simulator可以evaluate任意trajectory的quality，不需要真的有human driver走那条trajectory。相当于simulator是一个offline的oracle - 它知道哪些trajectory safe、comfortable、compliant with traffic rules。

### 新的loss

$$\sum_{\ell=1}^{L} \lambda^{L-\ell} \Big( \underbrace{\min\| W_t - \tilde{W}_\ell^{(n)} \|_2}_{\text{fit human}} + \sum_{P \in \mathcal{P}_t} \underbrace{\min\| P - \tilde{W}_\ell^{(n)} \|_2}_{\text{fit pseudo-teacher}} \Big)$$

**关键细节**: 注意每个pseudo-teacher $P$ 都是**独立**做min over $n$的。这意味着proposal 1可以去fit pseudo-teacher A，proposal 2可以同时fit pseudo-teacher B。这样proposals才能真正multimodal。

如果改成 $\min_n \sum_P \|P - \tilde{W}_\ell^{(n)}\|_2$，就变成一个proposal要fit所有teachers，完全失去multimodality。这个algebraic细节非常critical。

### Ablation验证

| Config | Diversity | EC (Comfort) | EPDMS |
|--------|-----------|-------------|-------|
| Baseline | 21% | 74.6 | 85.8 |
| +MTD | **40%** | 47.9 | 84.5 |

看这个table：MTD让Diversity从21%飙到40%（proposal真的分散开了），但Comfort score暴跌到47.9。为什么？因为每帧可能选到不同mode的proposal，导致frame-to-frame jitter。

## 四、Trick 3: Momentum-aware Selection救Comfort

### 问题

MTD让proposals diverse了，但选trajectory的时候有新问题：

Frame t: 可能选了left-turn mode的proposal  
Frame t+1: 可能切到right-turn mode的proposal  
Frame t+2: 又切回left-turn

这样trajectory就jitter了，Comfort score会很差。

### Solution

他们在neural scorer的输出上加一个momentum term:

$$S \gets \frac{7S + S_c}{8}$$

- $S$: neural scorer预测的score (safety, comfort等)
- $S_c$: 当前proposals跟上一帧selected trajectory的distortion-based comfort score
- 7:1的权重来自NAVSIM v2的metric design

**Intuition**: 这就是给trajectory selection加了个motion model prior - 上一帧选了啥，这一帧大概率应该跟上一帧consistent。类似MPC里的warm start或者tracking里的Kalman filter motion model。

### 效果

| Config | Diversity | EC | EPDMS |
|--------|-----------|-----|-------|
| +MTD only | 40% | 47.9 | 84.5 |
| +MTD +Momentum | 40% | **84.8** | **87.8** |

Momentum selection把EC从47.9拉回84.8（+36.9），Diversity保持40%不变。完美的trade-off - 既有多模态diversity又有temporal consistency。

## 五、为什么用单front camera效果更好

这个有点counter-intuitive。看Table 8：

| Method | Input |
|--------|-------|
| Transfuser/HydraMDP++/DriveSuprim/GoalFlow | 1024×256 (front+left+right stitched) |
| iPad | 4×768×432 (front+left+right+back) |
| Drive-JEPA | **2×512×256 (only front, t and t-1)** |

Drive-JEPA只用front camera，resolution还更小，但performance更好。

**可能的reasons**:
1. V-JEPA 2在Ego4D上pretrain时本来就是ego-centric single camera setup，domain gap最小
2. Front camera已经携带了planning需要的大部分信息（lane structure, leading vehicles, traffic lights）
3. Multi-view fusion增加complexity但marginal gain有限，尤其是在open-loop evaluation下
4. 小resolution + fewer cameras = faster training, 可以scale到更多data

## 六、实验结果SOTA分析

### NAVSIM v1

Perception-free setting（不用perception annotations，只supervise by human trajectory）:
- Drive-JEPA ViT/L: **89.0 PDMS**
- Epona ViT/G (1.1B params): 86.2 PDMS
- 我们用307M params + 208h data，beats了1.1B params + 128h data的Epona

这说明V-JEPA latent prediction的scaling efficiency远高于pixel-level generation。

Full setting（有perception supervision）:
- ResNet34: 91.5 PDMS（只用camera，beat了用camera+LiDAR的methods）
- ViT/L: **93.3 PDMS** (new SOTA)

### NAVSIM v2（更严格的metric）

ViT/L results:
| Method | NC | DAC | EP | EC | EPDMS |
|--------|-----|-----|-----|-----|-------|
| iPad | 98.7 | 98.0 | 86.6 | 74.6 | 85.8 |
| DriveSuprim | 98.4 | 98.6 | 90.5 | 78.6 | 87.1 |
| **Drive-JEPA** | 98.4 | 98.6 | 88.4 | **84.8** | **87.8** |

注意EC指标 - Drive-JEPA领先iPad 10个点，领先DriveSuprim 6个点。这就是MTD + Momentum Selection的威力。

### Bench2Drive（closed-loop）

| Method | SR | DS |
|--------|-----|-----|
| iPad | 33.18 | 60.52 |
| DriveTransformer | 35.01 | 63.46 |
| **Drive-JEPA** | **36.82** | **64.52** |

Closed-loop上DS +4 over iPad，证明MTD不只是open-loop metric hacking，真的improve driving quality。

## 七、这套设计为什么work - 核心Intuition

我觉得这paper最deep的insight是：**representation learning和supervision design必须co-design**。

- V-JEPA pretraining提供了好的latent space - encoder能capture spatiotemporal dynamics
- MTD提供了multimodal supervision - model能学到diverse behaviour
- Momentum selection提供了temporal regularization - diverse behaviour仍然smooth

三者缺一不可：
- 只有V-JEPA → no diversity, mode collapse to single trajectory
- 只有MTD → diversity但frame-to-frame jitter严重
- 只有Momentum → smooth但no multimodal reasoning，回到mode collapse

这就像一个three-legged stool - 三个leg互相support，缺一个就倒。

## 八、延伸思考

### 8.1 Simulator as Teacher的哲学

MTD本质上是在说：**simulator能judge trajectory quality，所以可以用作pseudo-teacher**。

这跟RLHF里用reward model当teacher类似 - 你不需要simulator真的"drive"，只需要它能evaluate。这是一个很强的insight，可以extend到其他domain：
- Robotics manipulation: 用physics simulator评估trajectory quality
- Game playing: 用rule-based engine评估move quality
- Dialogue: 用safety classifier评估response quality

### 8.2 Modal Collapse in Imitation Learning的general solution

Drive-JEPA解的是driving的modal collapse，但这问题在所有imitation learning task都存在：
- Human demonstration有multiple valid choices
- L2 regression loss让model collapse到average
- 导致"indecsive" behaviour

MTD的解法是：**找到多个valid targets，分别supervise不同modes**。这在robotics、game AI、dialogue generation都可以用。

### 8.3 V-JEPA vs Pixel-level Generation的scaling efficiency

Table 1的comparison暗示一个scaling law:
- 21M encoder + 20h data → 83.8 PDMS
- 1.1B encoder + 128h data → 86.1 PDMS (Epona)
- 307M encoder + 208h data → 89.0 PDMS (Drive-JEPA)

Epona用了4x参数，0.6x data，反而不如Drive-JEPA。这说明pixel-level generation的scaling efficiency远低于latent prediction。这跟LeCun一直advocate的"JEPA philosophy"完全consistent。

### 8.4 Open questions

1. **Vocabulary size sensitivity**: 8192是怎么定的？论文没给sensitivity analysis
2. **N_pseudo的选择**: Table 6显示N_pseudo=1最好，4个反而略差。这暗示over-diversification可能也有问题，为什么？
3. **Simulator teacher的limitation**: NAVSIM v2是non-reactive log-replay，other agents不会adapt to ego的behaviour。这在dense traffic场景下会有限制
4. **Momentum权重的generalization**: 7:1是NAVSIM v2 specific的，换到其他benchmark需要retune吗？

## 九、一句话总结

**Drive-JEPA = V-JEPA学好的representation + simulator蒸馏多模态trajectory + momentum让selection别jitter**

三个component互相补足，缺一不可。最终在NAVSIM v1/v2和Bench2Drive都拿到SOTA，而且用的compute比之前的video generation methods少很多。

---

**Key reference links:**
- Drive-JEPA code: https://github.com/linhanwang/Drive-JEPA
- V-JEPA 2 paper: https://arxiv.org/abs/2506.09985
- NAVSIM benchmark: https://github.com/autonomousvision/navsim
- iPad baseline: https://arxiv.org/abs/2505.15111
- Hydra-MDP distillation: https://arxiv.org/abs/2406.06978
- Bench2Drive closed-loop: https://github.com/Thinklab-SJTU/Bench2Drive

---

# Drive-JEPA: 深度解析

## 一、Paper的Core Narrative

这篇paper的核心是解决end-to-end autonomous driving中的**两个complementary bottleneck**:

1. **Vision pretraining瓶颈**: 之前world model pretraining给planning带来的收益有限
2. **Supervision瓶颈**: 每个scene只有single human trajectory, 但driving本质上是multimodal的

作者用一个unified framework解决这两个问题:
- 用V-JEPA做large-scale driving video pretraining
- 用multimodal trajectory distillation提供diverse supervision

GitHub repo: https://github.com/linhanwang/Drive-JEPA

## 二、Why V-JEPA Works for Driving (核心Intuition)

### 2.1 之前方法的局限

LAW、World4Drive等latent world model本质是predict feature_{T+1} from feature_T, 它们的问题:
- 容易representation collapse (没有特别的prevent机制)
- 无法scale到large-scale video pretraining
- 仅作为auxiliary loss

而video-generative方法(VaVAM, Epona):
- pixel-level reconstruction计算量巨大
- over-emphasize visual details不relevant to decision making

### 2.2 V-JEPA的核心insight

V-JEPA的核心公式 (公式1):
$$
\min_{\theta, \phi, \Delta_y} \| P_\phi(\Delta_y, E_\theta(x)) - \mathrm{sg}(E_{\bar{\theta}}(y)) \|_1
$$

变量解释:
- $E_\theta$: Student encoder, parameters为$\theta$, 处理masked view $x$
- $P_\phi$: Predictor, parameters为$\phi$, 预测masked位置的representation
- $\Delta_y$: Learnable mask token, 表示被dropped的patch位置
- $\mathrm{sg}(\cdot)$: Stop-gradient operator, 阻断gradient back to target
- $E_{\bar{\theta}}$: Teacher encoder, parameters为$\bar{\theta}$ (EMA of $\theta$), 处理target view $y$
- Loss只在masked positions计算

**Intuition**: V-JEPA在**latent space**做predict, 而不是pixel space, 这样:
- 不会waste capacity去reconstruct irrelevant visual details (sky texture,广告牌等)
- 但是需要predict meaningful dynamics (vehicle motion, lane structure等)
- EMA teacher + stop-gradient = 自然防止collapse (BYOL-style mechanism)

### 2.3 与driving的契合点

Driving video有很强的**spatiotemporal structure**:
- Ego-motion主导画面变化
- 其他agents有可预测的motion pattern
- Road geometry有temporal consistency

V-JEPA的masked prediction正好capture这些dynamics, 而不需要pixel-level generation。

## 三、Data Curation细节

| Source | 类型 | 用途 |
|--------|------|------|
| CoVLA | Vision-Language-Action | 大规模driving video |
| DrivingDojo | Knowledge-enriched driving world model | 交互性scene |
| OpenScene | NuPlan-based | Real-world scenarios |

数据处理:
- 仅用front-view camera (vs. iPad用4 camera)
- 8-frame clips, 512×256 resolution, 2Hz
- 总共208小时 (vs. Epona 128h, LAW/World4Drive ~20h)

**Key insight**: 用single front camera反而比multi-camera setup效果更好, 因为:
- 更aligned to V-JEPA 2的pretraining domain
- 减少了不必要的multi-view fusion complexity
- Front-view已经携带了主要planning信息

## 四、Waypoint-anchored Proposal Generation (架构核心)

### 4.1 Proposal Queries初始化

$$
\mathbf{Q}_0 \in \mathbb{R}^{N_p \times M \times D}
$$

- $N_p = 32$: proposal数量
- $M$: future waypoints数量 (typically 8)
- $D$: feature dimension
- 初始化: learnable positional embedding + ego status feature $\mathbf{e}_t$

Ego status经过linear projection得到 $\bar{\mathbf{e}}_t \in \mathbb{R}^{1 \times D}$, 包括:
- Driving command (left/forward/right)
- Speed
- Acceleration

### 4.2 Iterative Refinement

每次iteration ℓ:
1. **Decode**: $\tilde{\mathbf{W}}_\ell = \text{MLP}(\mathbf{Q}_\ell)$, shape $\mathbb{R}^{N_p \times M \times 3}$, 每个waypoint $(x, y, \psi)$
2. **Aggregate**: 用waypoint位置作为anchor, 通过WADA (Waypoint-anchored Deformable Attention) 从$\mathbf{F}_t$聚合local BEV features
3. **Update**: $\mathbf{Q}_{\ell+1} = \text{MLP}(\text{WADA}(\mathbf{Q}_\ell, \tilde{\mathbf{W}}_\ell, \mathbf{F}_t))$

WADA基于Deformable Attention, 在每个waypoint位置周围sample sparse reference points, 比full cross-attention效率高很多。BEV features通过lift-splat (Philion & Fidler 2020)得到。

### 4.3 Naive Loss的问题

$$
\mathcal{L}_{traj} = \sum_{\ell=0}^{L-1} \lambda^{L-\ell-1} \min_{n \in \{1,...,N_p\}} \| W_t - \tilde{W}_\ell^{(n)} \|_2
$$

- $\lambda = 0.1$: down-weight early iterations, 鼓励coarse-to-fine
- $\min$ over $n$: winner-takes-all (像Social GAN的diversity loss)

**问题**: 只有single human trajectory $W_t$, 所以所有proposals最终会collapse到一个mode附近, 丧失multimodality。

## 五、Multimodal Trajectory Distillation (MTD)

### 5.1 设计思路

这是这篇paper最clever的部分。作者没有用diffusion model (像DiffusionDrive, GoalFlow), 而是用**simulator as teacher**:

1. 构建8192个trajectory vocabulary (k-means on 100k+ trajectories)
2. 对每个training scene, 用rule-based simulator计算所有vocabulary trajectories的EPDMS score
3. 选score > 0.95的high-quality trajectories作为pseudo-teachers $\mathcal{P}_t$

### 5.2 新Loss公式

$$
\sum_{\ell=1}^{L} \lambda^{L-\ell} \Big( \min\| W_t - \tilde{W}_\ell^{(n)} \|_2 + \sum_{P \in \mathcal{P}_t} \min\| P - \tilde{W}_\ell^{(n)} \|_2 \Big)
$$

**关键insight**: 
- 仍然保留human trajectory $W_t$的监督 (第一项)
- 加上多个pseudo-teacher trajectories $P \in \mathcal{P}_t$的监督 (第二项)
- 各自独立做min, 这样不同proposals可以分别fit不同的modes

### 5.3 与Hydra-MDP的对比

Hydra-MDP的hydra-distillation是学习vocabulary上的**scores**, 仍然是fixed vocabulary, 受OOV (out-of-vocabulary)限制。

Drive-JEPA是把vocabulary作为**teacher trajectories**, proposals本身是online生成的continuous vocabulary, 有更好的generalization。

## 六、Momentum-aware Trajectory Selection

### 6.1 为什么需要这一层

MTD带来的副作用: proposals变diverse了, 但帧间consistency变差, 因为每帧可能选到不同mode的proposal, 导致trajectory jitter。

### 6.2 机制

Neural scorer:
$$
\mathbf{Q}_L^- \in \mathbb{R}^{N_p \times M \times D} \xrightarrow{\text{max pool}} \bar{\mathbf{Q}}_L \in \mathbb{R}^{N_p \times D} \xrightarrow{\text{MLP}} S \in \mathbb{R}^{N_p \times 1}
$$

训练loss:
$$
\mathcal{L}_{\text{score}} = \text{BCE}(S, \hat{S})
$$

$\hat{S}$来自simulator-based EPDMS evaluation。

Momentum calibration:
$$
S \gets \frac{7S + S_c}{8}
$$

- $S_c$: 当前proposals与上一帧selected trajectory $\hat{W}_{t-1}$的distortion-based comfort score
- 权重 7:1 来自NAVSIM v2的metric design (HC/EC都强调了cross-frame smoothness)

最终选择:
$$
\hat{W}_t = \tilde{W}_L^{(n^*)}, \quad n^* = \arg\max_{n \in \{1,...,N_p\}} S_n
$$

**Intuition**: 这是一个test-time的trick, 类似于tracking里面的motion model - 上一帧的trajectory应该是当前帧的strong prior, 防止帧间mode switching。

## 七、Auxiliary Tasks (轻量化设计)

作者avoid了computationally intensive的dense BEV segmentation/3D detection, 用了**proposal-centric**的轻量auxiliary:

1. **Proposal-centric mapping**: 
   - 预测每个waypoint的on-road和on-route probabilities
   - $R \in \mathbb{R}^{N_p \times M \times 2}$
   - $\mathcal{L}_{\text{map}} = \text{BCE}(R, \hat{R})$

2. **Proposal-centric collision prediction**:
   - 用log-replay simulation预测collision probability $A_v$
   - $\mathcal{L}_{\text{colli}} = \| A_v - \hat{A}_v \| + 0.1 \text{BCE}(A_v, \hat{A}_v)$

**Total loss**:
$$
\mathcal{L} = \mathcal{L}_{traj} + w_{\text{score}}\mathcal{L}_{\text{score}} + w_{\text{map}}\mathcal{L}_{\text{map}} + w_{\text{colli}}\mathcal{L}_{\text{colli}}
$$

$w_{\text{score}}=1, w_{\text{map}}=2, w_{\text{colli}}=1$ - mapping权重最大, 说明road geometry对planning的importance。

## 八、实验结果深度分析

### 8.1 NAVSIM v1 (Table 2)

Perception-free setting:
| Method | Encoder | Data | PDMS |
|--------|---------|------|------|
| LAW | 21M | ~20h | 83.8 |
| World4Drive | 21M | ~20h | 85.1 |
| Epona | 1.1B | 128h | 86.1 |
| **Ours** | **307M (ViT/L)** | **208h** | **89.0** |

**Key insight**: ViT/L (307M) + 208h数据, 用更低cost超过了1.1B的Epona, 这说明:
- Latent prediction >> pixel reconstruction
- V-JEPA的scaling efficiency远高于video generation models

Full setting (ResNet34):
- Drive-JEPA: 91.5 PDMS
- 仅用Camera, 不用LiDAR, 但超过用LiDAR的DriveSuprim和iPad
- EP (Ego Progress) 88.8最高, 说明driving style更assertive

Full setting (ViT/L):
- Drive-JEPA: 93.3 PDMS (SOTA)
- 注意DriveSuprim的93.5用ViT/L+C+L (camera+LiDAR), Drive-JEPA只用camera

### 8.2 NAVSIM v2 (Table 3)

| Method | Backbone | EPDMS |
|--------|----------|-------|
| iPad | ResNet34 | 84.1 |
| Drive-JEPA | ResNet34 | 85.4 |
| DriveSuprim | ViT/L | 87.1 |
| **Drive-JEPA** | **ViT/L** | **87.8** |

特别值得注意的是**EC (Extended Comfort)**指标:
- iPad ViT/L: 74.6
- DriveSuprim ViT/L: 78.6
- Drive-JEPA ViT/L: 84.8

EC是NAVSIM v2新加的指标, 评估cross-time-step smoothness, Drive-JEPA领先10个点, 说明**MTD + Momentum-aware Selection的组合非常有效**。

### 8.3 Bench2Drive (Table 4)

| Method | Effi. | Comf. | SR | DS |
|--------|-------|-------|-----|-----|
| iPad | 153.83 | 35.51 | 33.18 | 60.52 |
| DriveTransformer | 100.64 | 20.78 | 35.01 | 63.46 |
| **Drive-JEPA** | **157.85** | **30.24** | **36.82** | **64.52** |

**Closed-loop benchmark上的gain**: Drive-JEPA超过iPad 4 DS, 证明MTD在closed-loop setting下确实带来driving quality提升, 不只是open-loop metric hacking。

### 8.4 Ablation (Table 5)

| Config | D (Diversity) | EC | EPDMS |
|--------|--------------|-----|-------|
| Baseline (iPad+V-JEPA2) | 21% | 74.6 | 85.8 |
| + Driving Video Pretraining | 24% | 69.7 | 86.1 |
| + MTD | **40%** | 47.9 | 84.5 |
| + Momentum Selection | 40% | **84.8** | **87.8** |

**Critical insight**: 
- MTD让Diversity从24%升到40% (✓), 但EC从69.7降到47.9 (✗) - 这就是multimodality的代价
- Momentum Selection把EC从47.9拯救到84.8 (+36.9), 这就是为什么这套设计是必要的

### 8.5 Vision Pretraining Comparison (Table 7)

| Vision Encoder | PDMS |
|---------------|------|
| ImageNet (ResNet34) | 76.0 |
| DepthAnything | 76.1 |
| MAE | 83.4 (could not converge fully) |
| DINOv2 | 86.1 |
| SigLIP | 86.1 |
| V-JEPA 2 (直接用) | 86.2 |
| **Ours (V-JEPA + driving pretrain)** | **89.0** |

MAE和DepthAnything收敛困难 - 说明static image pretraining对video task不友好。

## 九、与V-JEPA 2的连接

Drive-JEPA直接使用V-JEPA 2的checkpoints, 论文: https://arxiv.org/abs/2506.09985

V-JEPA 2本身是Meta最新的video self-supervised模型, 在Ego4D等数据集上pretrain。Drive-JEPA的contribution是证明:
1. V-JEPA可以domain transfer到driving (208h data足够)
2. V-JEPA representation对planning task有strong inductive bias
3. Latent prediction远比pixel reconstruction efficient

## 十、相关联想与延伸

### 10.1 与LeCun的JEPA哲学

JEPA系列 (I-JEPA, V-JEPA, V-JEPA 2) 都是LeCun的"predict in latent space, not pixel space"哲学的产物。Drive-JEPA把这个哲学应用到driving, 完美契合, 因为:
- Driving需要semantic abstraction (vehicle/lane/intent), pixel-level detail是noise
- V-JEPA天然做abstraction, 自然sparser signal

### 10.2 与VADv2/Hydra-MDP的vocabulary思路

VADv2: https://arxiv.org/abs/2402.13243
Hydra-MDP: https://arxiv.org/abs/2406.06978

这些方法的核心是discretize action space成fixed vocabulary, 然后score。Drive-JEPA借用vocabulary作为**teacher source**, 但proposals是online生成的, 避免了OOV问题。

### 10.3 与DiffusionDrive/GoalFlow的对比

DiffusionDrive: https://arxiv.org/abs/2501.17049
GoalFlow: https://arxiv.org/abs/2501.18366

Diffusion-based methods理论上可以capture任意distribution, 但是:
- 训练时仍受single trajectory supervision限制
- Inference速度慢 (iterative sampling)

Drive-JEPA用simulator-distilled multi-target supervision代替diffusion的多模态建模, 是一种**simple but effective**的替代方案。

### 10.4 iPad的baseline地位

iPad: https://arxiv.org/abs/2505.15111

iPad是Drive-JEPA的proposal-centric baseline。Drive-JEPA对iPad的改进可以总结为:
1. Vision encoder: ResNet → V-JEPA pretrained ViT/L
2. Supervision: single human → multi pseudo-teacher
3. Selection: neural scorer → momentum-aware

### 10.5 NAVSIM benchmark family

NAVSIM: https://github.com/autonomousvision/navsim
NAVSIM v2: https://arxiv.org/abs/2506.04218

NAVSIM的设计哲学: open-loop evaluation + simulation-based metrics = pseudo-closed-loop assessment。这样既有open-loop的reproducibility, 又有closed-loop的meaningful evaluation。NAVSIM v2进一步加强了rule-compliance和comfort的评估, 让MTD + Momentum Selection的优势凸显。

### 10.6 关于scaling law的implication

Table 1的comparison暗示一个scaling law:
- 21M encoder + 20h data → 83.8 PDMS
- 1.1B encoder + 128h data → 86.1 PDMS
- 307M encoder + 208h data → 89.0 PDMS

V-JEPA的scaling efficiency明显高于video generation, 这意味着未来可以继续scale up:
- 更多driving video (从208h到1000h+)
- 更大ViT (从ViT/L到ViT/H)
- 更多cameras + LiDAR input

### 10.7 Modal Collapse in Imitation Learning

Drive-JEPA解决的modal collapse问题, 是imitation learning的general issue:
- Behavior cloning with L2 loss → mode collapse to mean
- 人类有multiple valid choices, 但BC只能学一种

MTD的解决方案是: 用simulator作为diverse teacher, 因为simulator可以evaluate任意candidate trajectory的quality。这是一种**asymmetric supervision**: simulator能判断"好坏"但不能"生成", 模型学习生成, 用simulator的judgement作为supervision。

### 10.8 Test-time momentum与RL中的similar ideas

Momentum-aware selection让我联想到:
- MPC (Model Predictive Control)中的warm start
- Tracking中的Kalman filter的motion model prior
- RL中的temporal consistency regularization

$$
S \gets \frac{7S + S_c}{8}
$$

这个权重7:1非常ad-hoc, 但作者说明它来自NAVSIM v2的HC+EC权重。这暗示NAVSIM v2已经把comfort的temporal weighting baked in, 这里复用就行。

### 10.9 关于pseudo-simulation的思考

NAVSIM v2用pseudo-simulation (非reactive的log-replay), 所以:
- 不需要训练RL agent
- 可以offline大规模评估
- 但有distribution shift (other agents不react)

Drive-JEPA利用这个pseudo-simulator做teacher, 巧妙地:
- Teacher的limitation (non-reactive) → student学到的是react to log-replayed agents
- Student在closed-loop (Bench2Drive)上仍然work, 说明这种simplification在driving domain够用

### 10.10 未来方向推测

基于这个work的insights, 未来可能的方向:
1. **V-JEPA driving → V-JEPA embodied**: 把这套思路推广到robot manipulation
2. **Multimodal distillation without simulator**: 用RL-trained policy作为teacher
3. **Joint V-JEPA + action conditioning**: 让V-JEPA本身become action-conditioned world model
4. **Test-time trajectory optimization**: 用V-JEPA latent做MPC的cost function

## 十一、Critical Thinking

### 11.1 一些可能的limitation

1. **Simulator teacher quality**: MTD依赖于NAVSIM v2的simulator quality, 如果simulator对某些case判断错误, 会propagate到student
2. **Vocabulary coverage**: 8192 trajectories可能still不够覆盖所有corner cases
3. **Momentum权重tuning**: 7:1是ad-hoc的, 在不同metric system下需要retune
4. **Closed-loop gap**: Bench2Drive的DS 64.52虽然SOTA, 但仍远低于human level, 说明closed-loop还有大空间

### 11.2 一些设计选择的open question

- 为什么是8192 vocabulary size? 论文说"balancing coverage and computational cost", 但没有sensitivity analysis
- 为什么pseudo-teacher trajectories的N_pseudo=1效果最好 (Table 6)? 4个反而略差? 这暗示over-diversification可能也有问题
- 为什么不用multi-view input? 单front camera反而好, 这与BEV-based方法的主流方向相反, 值得深入investigate

### 11.3 公式(2)的形式algebraic含义

$$
\sum_{\ell=1}^{L} \lambda^{L-\ell} \Big( \underbrace{\min\| W_t - \tilde{W}_\ell^{(n)} \|_2}_{\text{fit human}} + \sum_{P \in \mathcal{P}_t} \underbrace{\min\| P - \tilde{W}_\ell^{(n)} \|_2}_{\text{fit pseudo-teacher}} \Big)
$$

注意这里每个$P$单独min over $n$, 这意味着不同$P$可以fit到不同proposals, 实现真正的multimodality。如果改成$\min_n \sum_P \|P - \tilde{W}_\ell^{(n)}\|_2$, 就会变成一个proposal需要fit所有teachers, 失去multimodality。这个细节非常关键。

## 十二、总结性Intuition

Drive-JEPA的设计体现了一个重要principle: **representation learning和supervision design要co-design**。

- V-JEPA pretraining提供**好的latent space**, 让model能capture spatiotemporal dynamics
- MTD提供**multimodal supervision**, 让model能学到diverse behavior
- Momentum selection提供**temporal regularization**, 让diverse behavior仍然smooth

三者缺一不可:
- 只有V-JEPA → no diversity, mode collapse
- 只有MTD → diversity但jitter, EC低
- 只有Momentum → smooth但no multimodal reasoning

这套co-design approach对其他embodied AI task (robotics, navigation)有参考价值。

---

**Reference Links:**
- Drive-JEPA: https://github.com/linhanwang/Drive-JEPA
- V-JEPA (原始): https://arxiv.org/abs/2304.08471
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- NAVSIM: https://github.com/autonomousvision/navsim
- NAVSIM v2 (pseudo-simulation): https://arxiv.org/abs/2506.04218
- Bench2Drive: https://github.com/Thinklab-SJTU/Bench2Drive
- iPad (proposal-centric baseline): https://arxiv.org/abs/2505.15111
- Hydra-MDP (vocabulary distillation): https://arxiv.org/abs/2406.06978
- VADv2 (vocabulary): https://arxiv.org/abs/2402.13243
- DiffusionDrive (diffusion-based): https://arxiv.org/abs/2501.17049
- GoalFlow (goal-guided diffusion): https://arxiv.org/abs/2501.18366
- Epona (autoregressive diffusion world model): https://arxiv.org/abs/2502.08246
- CoVLA dataset: https://github.com/CV-LLM/CoVLA
- DrivingDojo: https://drivingdojo.github.io/
- OpenScene: https://github.com/OpenDriveLab/OpenScene
- Deformable Attention (WADA base): https://arxiv.org/abs/2201.00520
- Lift-Splat-Shoot (BEV): https://arxiv.org/abs/2008.05730
- DINOv2 (vision pretraining baseline): https://arxiv.org/abs/2304.07193
- MAE (pretraining baseline): https://arxiv.org/abs/2111.06377
- CARLA (Bench2Drive simulator): https://carla.org/
- nuPlan (NAVSIM base): https://www.nuscenes.org/nuplan
- Hydra-MDP++ (extended version): https://arxiv.org/abs/2503.12820
- DriveSuprim (SOTA baseline): https://arxiv.org/abs/2506.06659
- World4Drive (latent world model): https://arxiv.org/abs/2507.02122
- LAW (latent world for driving): https://arxiv.org/abs/2406.08486
- VaVIM/VaVAM (video generation): https://arxiv.org/abs/2502.15672
