---
source_pdf: Scaling Cross-Embodied Learning.pdf
paper_sha256: 509826f25f8058b858ad680581a20cd47ecb80c0346b2c27fd42b6eef3482769
processed_at: '2026-08-12T03:22:09-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# CrossFormer 人话版

## 一句话总结

这paper干了一件事：**train一个policy，能同时控制20种不同的robot**——机械臂、双臂、轮式车、四足狗、无人机——一个network全搞定，performance还不输给专门为单一robot训的policy。

项目主页：https://crossformer-model.github.io

---

## 为什么这事难——用一个直觉比喻

想象你要训一个"全能司机"，他得会开轿车、卡车、摩托车、船、飞机。问题是：

- **轿车**有方向盘+油门刹车，**摩托车**有手把+离合，**船**有舵轮+油门推杆，**飞机**有操纵杆+脚舵
- **轿车**前面挡风玻璃看路，**船**360度都看，**飞机**还要看仪表盘
- **轿车**60Hz刷新，**飞机**1000Hz刷新（夸张说法，intuition上）

Prior work怎么解决？**强迫大家都开轿车**——把卡车方向盘拆了装轿车方向盘，把摩托车改成四轮，把飞机固定高度只允许平飞。这就是RT-X、Yang et al.的做法：手动align action space，一次只用一个camera view。

CrossFormer的approach：**就让transformer自己学**。你有什么sensor我吃什么sensor，你要output什么action我给什么head。不align，不裁剪，不妥协。

---

## 他们怎么搞的——一步步拆解

### Step 1: 把所有observation塞进一个sequence

每个robot每个timestep的observation都不一样：

- **WidowX单臂**：1个3rd-person camera image $I \in \mathbb{R}^{H \times W \times 3}$
- **ALOHA双臂**：3个camera（overhead + 2 wrist）+ 14维proprioception $P \in \mathbb{R}^{14}$
- **Go1四足**：只有proprioception $P \in \mathbb{R}^{59}$，没camera
- **LoCoBot导航**：1个egocentric camera

怎么办？**每种observation用专门的tokenizer**：

Image tokenization：
$$I \xrightarrow{\text{ResNet-26}} \text{feature map} \xrightarrow{\text{flatten}} \text{tokens } I^{1:L} \in \mathbb{R}^{L \times 512}$$

其中 $L$ 是flatten后的spatial token数（比如 $7\times7=49$），$512$是token embedding dimension。

Proprioception tokenization：
$$P \xrightarrow{\text{Linear}} P^{1:M} \in \mathbb{R}^{M \times 512}$$

其中 $M$ 是proprioception被切成的token数。

**关键trick**：同类camera共享encoder。3rd-person view的workspace encoder在单臂和双臂之间share，egocentric的navigation encoder在轮式和无人机之间share。这个share是zero-shot Tello transfer能work的原因。

### Step 2: Task specification怎么给

有的task用language说（"把勺子放布上"），有的task用goal image（给一张目标场景的照片）。

- **Language** $l$：用FiLM [52]和image feature融合
  $$\text{FiLM}(I, l) = \gamma(l) \odot I + \beta(l)$$
  其中 $\gamma(l), \beta(l)$ 是language embedding经过MLP产生的scale和shift，$\odot$是element-wise乘法。这就是一种conditional normalization。

- **Goal image** $g$：和current image在channel维度concat
  $$I_{\text{cond}} = [I; g] \in \mathbb{R}^{H \times W \times 6}$$
  然后一起进image encoder。

训练时random mask掉language或goal，所以test time两种都能用。这个设计来自BC-Z [58]和Lynch et al. [56]。

### Step 3: Readout token——这个paper的核心trick

问题来了：一个transformer怎么output不同dimension的action？

答案是**在input sequence里插入special tokens**，叫readout tokens $R$。这些token的位置是fixed的（靠positional embedding），它们的embedding经过transformer后，被送进对应的action head。

完整input sequence长这样：
$$[I_t^{1:L}, P_t^{1:M}, R_t^{1:N}, I_{t+1}^{1:L}, P_{t+1}^{1:M}, R_{t+1}^{1:N}, ..., I_{t+k}^{1:L}, P_{t+k}^{1:M}, R_{t+k}^{1:N}]$$

变量解释：
- $t$：当前timestep
- $k$：observation history length（paper用5）
- $L$：image token数
- $M$：proprioception token数
- $N$：readout token数 = action chunk size

**Attention mask设计**：block-wise causal
- Observation tokens只能attend到**同timestep或更早**的observation tokens
- Readout tokens只能attend到**prior** observation tokens（不能看future）

这个mask让readout token变成"information sink"——它汇集了所有prior observation的信息，然后用来predict action。很像BERT的[CLS] token，但有几个关键区别：

1. 多个readout token（N个，对应chunk size），不是一个
2. 位置固定 → model从positional embedding知道"现在该output哪种action"
3. Causal → 不能看future

### Step 4: Action head——4种action各走各的

| Action head | Dimension | Chunk size | Freq | 用在哪 |
|---|---|---|---|---|
| Single-arm Cartesian | 7 | 4 | 5-15Hz | WidowX, Franka |
| Navigation waypoint | 2 | 4 | 4Hz | LoCoBot, Tello |
| Bimanual joint | 14 | 100 | 20Hz | ALOHA |
| Quadruped joint | 12 | 1 | 20Hz | Go1 |

为啥chunk size差这么多？**因为control frequency差很多**。

直觉：20Hz的ALOHA每0.05秒就得output一个action。如果每step都重新infer policy，compounding error会爆掉。一次predict 100个action（=5秒的action序列），执行起来就平滑了。

公式：
$$\hat{a}_{t:t+c} = \text{ActionHead}(\text{Transformer}(R_t^{1:N}))$$

其中 $c$ 是chunk size，$\hat{a}_{t:t+c} \in \mathbb{R}^{c \times d_a}$，$d_a$是action dimension。

**Loss**：L1 regression
$$\mathcal{L} = \frac{1}{c \cdot d_a} \sum_{i=1}^{c} \sum_{j=1}^{d_a} |\hat{a}_{i,j} - a_{i,j}^{gt}|$$

为什么用L1不用diffusion？我推测：
- Diffusion inference慢，20Hz control压力太大
- L1在ACT [47]上已证明够用bimanual
- 多embodiment mix时，diffusion的noise schedule难调（不同action scale不同）

### Step 5: Transformer backbone

12层decoder-only transformer，8个attention head，token embedding 512，MLP dim 2048。加上4个ResNet-26 image encoder和4个action head，总共130M params。

Context window 2135 tokens = 5个timestep × (image tokens + proprio tokens + readout tokens)。

训练：300K steps，batch 512，TPU V5e-256上47小时。

---

## 数据：900K trajectories，20种embodiment

这是当时最大最diverse的cross-embodied dataset。核心组成：

- **Bridge** (WidowX单臂)：17%
- **GNM** (navigation)：17%
- **ALOHA-multi-task** (双臂)：17%
- **Fractal** (RT-1单臂)：17%
- **Go1-walk** (四足)：8.5%
- **Franka-tabletop** (Franka单臂)：8.5%
- 其他OXE subset加起来约16%

Target dataset up-weight到17%或8.5%，确保evaluation setting表现好。这是hand-tuned的，paper承认是limitation。

---

## 实验结果：没negative transfer就是赢

### 6个real-world setting的平均success rate

| 方法 | 平均成功率 |
|---|---|
| Single-robot dataset (只训target robot数据) | 0.68 |
| Best prior method (每个setting最强的specialist) | 0.51 |
| **CrossFormer** | **0.73** |

**关键观察**：CrossFormer加了900K其他robot的数据，performance**不降反升**。这意味着**no negative transfer**。

最戏剧性的结果：
- **ALOHA双臂**：Single-robot 0.60/0.40，CrossFormer 0.80/0.60 —— 有positive transfer！
- **Tello无人机**：zero-shot（训练数据里没有无人机），CrossFormer 0.88，Yang et al.只有0.30
- **WidowX**：Yang et al.直接0% success（因为Yang一次只用一个camera view，3rd-person view训练不足），CrossFormer 0.25-0.75

### 为什么能beat Yang et al. 3x

Yang et al. [8]的做法是手动align manipulation和navigation的action space，一次只input一个camera view。问题：

1. **Align是脆的**：把wrist camera和egocentric camera "看作一样"是巧合，不是general principle。碰到proprioception-only的quadruped就抓瞎。
2. **单camera view限制表达**：ALOHA有3个camera同时看，Yang只能用1个，信息丢失。
3. **Architecture不够flexible**：硬align导致model capacity被浪费在"适应统一format"上。

CrossFormer让transformer自己学哪个observation对应哪个action，capacity全部用来学task本身。

---

## 核心intuition：为什么这种设计能work

### 1. Sequence-to-sequence是cross-embodied的自然框架

不同embodiment的差异本质是**input/output dimensionality差异**。Transformer天然handle variable-length sequence：
- Missing observation → mask掉对应token
- Different action dim → 不同readout token数 + 不同head

这比align approach更principled，因为不需要人为设计"统一format"。

### 2. Shared encoder促transfer

Workspace encoder同时处理单臂和双臂的3rd-person view → 双臂能leverage单臂的visual representation。
Navigation encoder同时处理LoCoBot和Tello → zero-shot Tello transfer成为可能。

### 3. Readout token的positional prior

Readout token在context window的**固定位置**。这个positional embedding告诉model"现在该output哪种action"——单臂的readout token永远在位置X，双臂的永远在位置Y。Model不需要从observation推断embodiment，positional embedding已经编码了。

### 4. Action chunking解决frequency mismatch

20Hz的bimanual每50ms就要action。如果每step都infer policy，误差累积爆炸。一次predict 100个action = 5秒plan，执行起来平滑。

但quadruped不chunk（只predict 1个action），因为joint dynamics对单步误差敏感，需要replan。

### 5. Zero-shot Tello transfer的深层含义

Tello实验是亮点。训练数据里没有quadcopter，但navigation head能zero-shot控制。这说明：

- **Egocentric navigation的visual representation在embodiment间transfer了**
- **2D waypoint action足够abstract**：高度固定后，"前进+转向"的语义对轮式和飞行式都适用
- **"Embodiment"的概念在navigation setting里其实很模糊**——只要camera view和action semantics类似，model不在乎是轮子还是螺旋桨

这hint了未来的方向：**action abstraction**。如果action能abstract成"move forward X, turn Y"，任何locomotion embodiment都能用同一个head。

---

## Limitations——paper自己承认的

1. **No significant positive transfer yet** —— 主要是"no negative transfer"，真正的cross-embodiment synergy还没出现。我推测原因：embodiment gap太大（quadruped proprioception和单臂visual观测几乎没overlap），data scale per embodiment不够（900K分到20个embodiment，每个平均45K，远不够）。

2. **Hand-picked sampling weights** —— 需要人工tune数据比例。理想情况：model capacity够大，能fit所有数据无需weighting。

3. **Inference speed** —— 130M model控制20Hz bimanual已经吃力。更大model会更慢。未来需要distillation、quantization、或speculative decoding。

4. **No significant positive transfer**这个限制其实最重要。真正positive transfer可能需要：
   - 每个embodiment至少100K+ trajectories
   - 共享的"perceptual" task（比如所有robot都要understand object affordance）
   - 更大model capacity（1B+ params）

---

## 这paper的位置在robot foundation model演进线上

```
RT-1 (single robot, 2022) 
  → RT-2 (VLM transfer, 2023) 
    → RT-X (cross single-arm, 2023) 
      → Octo (flexible single-arm, 2024) 
        → CrossFormer (truly cross-embodied, 2024)
```

同期对比：
- **OpenVLA** [57]: 7B VLA model，只做single-arm，language understanding强但embodiment单一
- **π0** (Physical Intelligence): 更大scale的bimanual，但不是cross-embodied
- **CrossFormer**: 牺牲model size换embodiment diversity

未来可能converge的路线：
- **VLM backbone** + **CrossFormer-style readout token** + **multi-embodiment action head**

---

## 我的两点extra思考

### Readout token vs Diffusion head的trade-off

CrossFormer用L1 regression，简单快。但L1假设action distribution是unimodal的。如果task有multiple valid solution（比如杯子可以从左边抓也可以从右边抓），L1会average out到中间invalid的action。

Diffusion Policy [49]能handle multimodal，但inference慢。未来可能用**flow matching**——比diffusion快，能handle multimodal，已经在π0里用了。

### "Embodiment"这个概念会被action abstraction淘汰

CrossFormer的Tello zero-shot transfer让我觉得，未来"embodiment-specific"会越来越模糊。如果action能被abstract成更高层semantic：

- Locomotion: "move(X, Y, θ)" → 任何locomotion embodiment都用一个head
- Manipulation: "pose(EE_position, gripper)" → 任何arm都用一个head
- Dexterous: "finger_config(joint_angles)" → 任何dexterous hand都用一个head

那时候cross-embodied learning就真的变成"cross-task learning"了，embodiment差异被action abstraction吃掉。

---

## 关键参考文献

1. **CrossFormer项目主页**: https://crossformer-model.github.io
2. **Open X-Embodiment / RT-X**: https://openx-embodiment.github.io
3. **Octo**: https://octo-models.github.io
4. **Yang et al. (Pushing the limits)**: https://arxiv.org/abs/2402.19432
5. **ACT / ALOHA**: https://tonyzhaozh.github.io/aloha
6. **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu
7. **GNM / ViNT**: https://general-navigation-models.github.io
8. **OpenVLA**: https://openvla.github.io
9. **Gato**: https://arxiv.org/abs/2205.06175
10. **RoboCat**: https://arxiv.org/abs/2306.11706
11. **π0 (Physical Intelligence)**: https://www.physicalintelligence.company/blog/pi0
12. **FiLM**: https://arxiv.org/abs/1709.07871
13. **Hindsight Experience Replay**: https://arxiv.org/abs/1707.01495

---

**TL;DR**：CrossFormer证明了，把cross-embodied imitation learning cast成sequence-to-sequence problem，用transformer + readout token + action chunking，就能一个policy控制20种robot，performance还不输specialist。目前主要是"no negative transfer"，positive transfer还要等data scale更大。但这是robot foundation model路线上的重要milestone——证明了flexibility不需要sacrifice performance。

---

# CrossFormer: Cross-Embodied Learning的Scaling尝试

这篇paper来自UC Berkeley的Sergey Levine组和CMU的Sudeep Dasari,核心贡献是提出一个**single transformer policy**能控制20种不同的robot embodiment——从单臂、双臂、轮式导航、四足到无人机——**无需手动align observation或action space**。

项目主页: https://crossformer-model.github.io
arXiv: https://arxiv.org/abs/2408.11812 (CrossFormer)
相关前置工作:
- RT-X / Open X-Embodiment: https://openx-embodiment.github.io
- Octo: https://octo-models.github.io
- GNM: https://general-navigation-models.github.io
- DROID: https://droid-dataset.github.io
- ALOHA / ACT: https://tonyzhaozh.github.io/aloha
- Yang et al. (Pushing the limits): https://arxiv.org/abs/2402.19432

---

## 1. 核心Motivation: 为什么需要Cross-Embodied Policy

Robot learning一直有一个**data scaling bottleneck**——单一robot platform能采集的数据有限。如果能用一个policy训练在多种robot数据上,就能:
1. 跨embodiment共享visual representation (比如不同相机看同样的物体)
2. 共享skills (比如approach、grasp这类primitive)
3. 减少为每个robot设计tune architecture的engineering effort

但cross-embodied training的**核心难点**在于:
- Observation space差异: 有的robot有3个camera + proprioception,有的只有1个egocentric camera,有的只有proprioception (quadruped)
- Action space差异: 7-DoF Cartesian、14-DoF joint、12-DoF quadruped joint、2-DoF navigation waypoint
- Control frequency差异: 从5Hz单臂到20Hz bimanual
- Task specification差异: 有的用language,有的用goal image

**Prior work的妥协**:
- RT-X [5]: 只用single-arm + 3rd-person view + 7-DoF EE position action —— 强制align
- GNM [41] / ViNT [42]: 只用navigation robot + egocentric view + 2D waypoint —— 强制align
- Yang et al. [8]: manipulation和navigation之间手动align action space,一次只能input一个camera view

CrossFormer的突破在于:**完全不align,让transformer自己吸收heterogeneity**。

---

## 2. CrossFormer架构解析

### 2.1 整体设计哲学

CrossFormer把cross-embodied imitation learning cast成**sequence-to-sequence problem**:
- Input: variable observation tokens + task spec tokens
- Output: variable action tokens (via readout tokens)

关键设计选择:
1. **Modality-specific tokenizer** —— 每种observation type有自己的tokenizer
2. **Shared transformer backbone** —— 所有embodiment共用一个decoder-only transformer
3. **Embodiment-specific action head** —— 每类action有独立的projection head
4. **Action chunking** —— 适应不同control frequency

### 2.2 Input Tokenization

给定trajectory $\tau = [(I_t, P_t, a_t), (I_{t+1}, P_{t+1}, a_{t+1}), ...]$,其中:
- $I_t \in \mathbb{R}^{H \times W \times C}$: image observation at time $t$
- $P_t \in \mathbb{R}^{d_p}$: proprioceptive observation (joint position/velocity等)
- $a_t \in \mathbb{R}^{d_a}$: action

定义observation history length $k$,把trajectory切成k-length segments:
$$[I_t, P_t, ..., I_{t+k}, P_{t+k}]$$

**Image tokenization**:
- 用ResNet-26 encoder (ImageNet pretrained)
- 输出feature map → flatten spatial dims → linear project到token embedding size (512)
- **关键**: 同类型的camera view共享encoder weights
  - Workspace image encoder (3rd-person manipulation view)
  - Navigation image encoder (egocentric navigation view)
  - 2个wrist camera encoders (manipulation wrist views)

**Proprioception tokenization**:
- 直接linear project到512-dim token

**Task specification**:
- Language instruction $l$: 通过FiLM [52]与image feature融合
- Goal image $g$: 与current image在channel dimension stack后一起进image encoder
- 训练时random mask其中一种,test time两种都能用

最终input sequence:
$$[I_t^{1:L}, P_t^{1:M}, R_t^{1:N}, ..., I_{t+k}^{1:L}, P_{t+k}^{1:M}, R_{t+k}^{1:N}]$$

其中:
- $L$: image token数 (after flatten)
- $M$: proprioception token数
- $R_t^{1:N}$: N个readout tokens,用于predict action
- $N$ 对应action chunk size

### 2.3 Attention Mask设计

用**block-wise causal attention mask**:
- Observation tokens只能attend到同timestep或更早timestep的observation tokens
- Readout tokens只能attend到prior observation tokens
- 这样readout token的embedding自然作为"query"来predict action

这种设计的好处:
- 时间维度的causality保留
- Readout tokens是"information sink",汇集相关observation信息

### 2.4 Action Head

4个action head对应4类action space:

| Action Head | Dimension | Chunk Size | Freq | Application |
|---|---|---|---|---|
| Single arm Cartesian | 7 (Δxyz + Δrpy + gripper) | 4 | 5-15Hz | WidowX, Franka |
| Navigation waypoint | 2 (Δx, Δy) | 4 | 4Hz | LoCoBot, Tello |
| Bimanual joint | 14 (7 per arm) | 100 | 20Hz | ALOHA |
| Quadruped joint | 12 (3 per leg × 4 legs) | 1 | 20Hz | Go1 |

**Loss**: L1 regression (不是diffusion,不是classification)
$$\mathcal{L} = \frac{1}{N \cdot d_a} \sum_{i=1}^{N} \sum_{j=1}^{d_a} |a_{i,j}^{pred} - a_{i,j}^{gt}|$$

选L1的原因: ACT [47]证明在high-frequency bimanual manipulation上效果好。

### 2.5 为什么Action Chunking重要

对于20Hz的bimanual control,如果每个timestep都重新infer policy,会有:
- **Compounding error**: 每个step的小误差累积
- **Temporal inconsistency**: 相邻action可能跳变,导致jitter

Action chunking一次predict未来100个action (5秒的action),执行时:
- 要么全部执行 (open-loop)
- 要么每k个action re-plan一次 (receding horizon)

公式上,对于timestep $t$,预测:
$$\hat{a}_{t:t+c} = f_\theta(R_t | I_{t-k:t}, P_{t-k:t})$$

其中 $c$ 是chunk size, $f_\theta$ 是transformer + action head。

### 2.6 架构细节与训练超参

| Component | Spec |
|---|---|
| Transformer layers | 12 |
| Attention heads | 8 |
| Token embedding | 512 |
| MLP dim | 2048 |
| Image encoder | ResNet-26 (×4) |
| Total params | 130M |
| Context window | 2135 tokens (5 timesteps) |
| Optimizer | AdamW |
| LR | 3e-4, reciprocal sqrt schedule, 2000 warmup |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| Batch size | 512 |
| Total steps | 300K |
| Training time | 47 hours on TPU V5e-256 |

---

## 3. Training Data: 900K Trajectories across 20 Embodiments

这是当时**最大最diverse的cross-embodied dataset**。数据mix见paper Table 1:

**Target datasets (up-weighted to 17% or 8.5%)**:
- Bridge (WidowX manipulation): 17%
- GNM (navigation): 17%
- ALOHA-multi-task: 17%
- Fractal (RT-1): 17%
- Go1-walk: 8.5% (25分钟,自己采集,RL expert生成)
- Franka-tabletop: 8.5% (200条,自己采集)

**其他OXE subset**:
- Kuka, BC-Z, Language Table, Taco Play, Furniture Bench, Roboturk, Jaco Play, etc.
- DROID: 0.022% (虽然DROID本身很大,但这里比例小,因为target dataset是Franka-tabletop不是DROID)

**Sampling weight的intuition**: 
- Target dataset up-weight确保evaluation setting表现好
- 避免被重复性高的datasets dominate (比如Fractal虽然占17%,但task diversity比某些小dataset低)

**Hindsight goal relabeling** [55]: 训练时uniformly sample未来observation作为goal,这样goal-conditioned能力自然学到。

---

## 4. Evaluation: 6个Real-World Settings

### 4.1 评估场景

1. **WidowX Manipulation** (Bridge setup): 3rd-person camera, 4个task (2 language + 2 goal), 12 trials/task
2. **Franka Manipulation** (DROID setup): 3rd-person camera, 2个language task, 27+12 trials
3. **ALOHA Bimanual**: 3 cameras (1 overhead + 2 wrist), 2个language task, 10 trials/task
4. **LoCoBot Navigation**: 1 camera, 3 skills (path-following, obstacle avoidance, cornering), 6 locations
5. **Go1 Quadruped**: proprioception only ($o_t \in \mathbb{R}^{59}$), walking forward, 25分钟 normalized reward
6. **Tello Quadcopter**: **Zero-shot generalization** (训练数据里没有quadcopter), navigation head输出2D waypoint,固定高度

### 4.2 主要实验结果 (Table 3)

| Embodiment | Task | Single-Robot | Best Prior | CrossFormer |
|---|---|---|---|---|
| WidowX | Spoon on cloth | 0.25 | 0.25 | 0.25 |
| WidowX | Mushroom in pot | 0.00 | 0.17 | 0.25 |
| WidowX | Cloth on saucer | 0.75 | 0.67 | 0.75 |
| WidowX | Carrot on plate | 0.67 | 0.25 | 0.33 |
| Franka | Sweep pinecones | 0.41 | 0.52 | 0.41 |
| Franka | Flip pot upright | 0.83 | 0.67 | 0.83 |
| ALOHA | Uncap pen | 0.60 | 0.70 | **0.80** |
| ALOHA | Cut sushi | 0.40 | 0.30 | **0.60** |
| LoCoBot | Obstacle avoidance | 0.95 | 0.30 | 0.95 |
| LoCoBot | Cornering | 0.95 | 0.85 | 0.95 |
| LoCoBot | Sharp cornering | 0.85 | 0.30 | **0.88** |
| Tello | Cornering | 0.85 | 0.30 | **0.88** |
| Go1 | Walking | 1.0 | N/A | 1.0 |
| **Average** | | **0.68** | **0.51** | **0.73** |

### 4.3 关键Observations

**Q1: 能否match single-robot dataset训练?**
- CrossFormer平均0.73 vs Single-robot 0.68
- 关键: **no negative transfer** —— 加了900K其他robot数据,performance不降反升
- ALOHA上还有明显positive transfer (+0.20)

**Q2: 能否match best prior method?**
- CrossFormer 0.73 vs Best prior 0.51
- 在ALOHA、LoCoBot、Tello上大幅超过prior
- 这说明flexible architecture能更好fit heterogeneous data

**Q3: 与Yang et al. [8]比较 (Figure 6)**

Yang et al.需要手动align action space,一次只能用一个camera view。结果:
- CrossFormer在navigation和manipulation上**3x outperform** Yang et al.
- WidowX上Yang et al.得到**0% success rate** (因为Yang的方法一次只用一个view,3rd-person view训练不充分)
- CrossFormer支持同时用3个camera view (ALOHA设置)

---

## 5. 核心Intuition: 为什么这种设计work

### 5.1 Sequence-to-Sequence是Cross-Embodied的自然框架

不同embodiment的差异本质上是"input/output dimensionality"差异。Transformer天然handle variable-length sequence:
- Missing observation → mask掉对应token
- Different action dimension → 不同readout token数量 + 不同head

对比prior work的align approach:
- RT-X: 强制所有robot用7-DoF EE position —— 但quadruped的12-DoF joint怎么align?
- Yang et al.: manipulation wrist camera和navigation egocentric camera "看起来像"所以align —— 但这是巧合不是general principle

CrossFormer的approach更**principled**: 让model自己learn哪种observation对应哪种action。

### 5.2 Shared Encoder促进Transfer

Workspace image encoder同时处理single-arm和bimanual的3rd-person view —— 这让bimanual能leverage single-arm的visual representation。
Navigation encoder同时处理LoCoBot和Tello的egocentric view —— 这让zero-shot Tello transfer成为可能。

### 5.3 Readout Token是关键trick

Readout token类似BERT的[CLS] token,但有几个关键区别:
1. **多个readout token** (N个,对应action chunk size) —— 不是一个
2. **Positional信息** —— readout token在context window的固定位置,这种positional prior让model知道"现在该predict哪种action"
3. **Attention mask** —— 只能看past,不能看future (causal)

这种设计让一个transformer能输出任意dimension的action,无需修改architecture。

### 5.4 Action Chunking解决Frequency Mismatch

不同embodiment的control frequency差异巨大:
- Single-arm: 5-15Hz
- Navigation: 4Hz
- Bimanual: 20Hz
- Quadruped: 20Hz

如果用同一policy frequency训练,高频robot会因compounding error失败。Action chunking:
- Bimanual predict 100个action (5秒) —— 相当于"规划"
- Quadruped predict 1个action —— 因为joint dynamics对单步误差敏感,不需要chunk
- 单臂predict 4个action —— 中等chunk

---

## 6. Limitations & Future Directions

Paper自己承认的limitations:

1. **No significant positive transfer yet** —— 目前主要是"no negative transfer",真正的cross-embodiment synergy还没出现。Intuition: 数据规模还不够大,embodiment diversity还不够高。

2. **Hand-picked sampling weights** —— 需要人工tune数据比例。理想情况: model capacity足够大,能fit所有数据无需weighting。

3. **Inference speed** —— 130M model控制20Hz bimanual已经吃力,更大model会更慢。未来需要:
   - Model distillation
   - Quantization
   - Speculative decoding for action

4. **Future work方向**:
   - Sub-optimal robot data incorporation
   - Action-free human video (EgoExo4D, Ego4D这类)
   - 更diverse embodiment (humanoid, dexterous hand)

---

## 7. 我的几点思考

### 7.1 这篇paper的位置

CrossFormer在robot foundation model的演进线上:
```
RT-1 (single robot) → RT-2 (VLM transfer) → RT-X (cross single-arm) → Octo (flexible single-arm) → CrossFormer (truly cross-embodied)
```

与同期工作比:
- **OpenVLA** [57]: 7B VLA model,但只做single-arm manipulation
- **π0** (Physical Intelligence): 更大scale的bimanual,但不是cross-embodied
- **CrossFormer**: 牺牲model size换embodiment diversity

### 7.2 为什么No Positive Transfer?

虽然paper希望positive transfer,但实验显示主要是"no negative transfer"。我推测原因:

1. **Embodiment gap太大** —— quadruped的proprioception和单臂的visual观测几乎没overlap
2. **Task skill不共享** —— walking和grasping是不同motor primitive
3. **Data scale per embodiment不够** —— 900K看起来大,但分到20个embodiment,每个平均45K,远不够learn shared representation

真正的positive transfer可能需要:
- 每个embodiment至少100K+ trajectories
- 共享的"perceptual" task (比如所有robot都要understand object affordance)
- 更大model capacity (1B+ params)

### 7.3 Readout Token vs Diffusion Head

CrossFormer用L1 regression,而Diffusion Policy [49]用diffusion。为什么CrossFormer不用diffusion?

推测:
- Diffusion inference慢,20Hz control压力大
- L1在ACT上已证明够用
- 多embodiment mix时,diffusion的noise schedule难tune (不同action scale)

但diffusion的好处是multimodal action distribution。如果未来要handle更complex task (比如multi-solution manipulation),可能需要diffusion或flow matching head。

### 7.4 与VLA路线的对比

当前robot foundation model有两条路线:
1. **VLA路线** (RT-2, OpenVLA, π0): 用pretrained VLM backbone,language understanding强,但action space通常single-arm
2. **Sequence-to-Sequence路线** (CrossFormer, Octo): 从头train transformer,flexible但language understanding弱

CrossFormer的FiLM conditioning远不如VLA的token-level language fusion。未来可能converge:
- 用VLM backbone + CrossFormer-style readout token + multi-embodiment action head

### 7.5 Zero-shot Tello Transfer的Implication

Tello实验是亮点 —— 训练数据里没有quadcopter,但navigation head能zero-shot控制。这说明:
- Egocentric navigation的visual representation在LoCoBot和Tello间transfer了
- 2D waypoint action space足够general (高度固定)
- "Embodiment"的概念在navigation setting里其实很模糊 —— 只要camera view和action semantics类似,model不在乎是轮子还是螺旋桨

这hint了未来的方向: **action abstraction**。如果action能abstract成"move forward X meters, turn Y degrees",任何locomotion embodiment都能用同一个head。

---

## 8. Summary

CrossFormer的核心贡献:
1. **First truly cross-embodied policy** —— 单臂+双臂+轮式+四足+无人机,一个network
2. **No manual alignment** —— 不需要align observation/action space
3. **Scale** —— 900K trajectories, 20 embodiments, 130M params
4. **Match specialist performance** —— 0.73 vs 0.68 single-robot, 0.51 best prior

核心limitation:
1. 主要还是"no negative transfer",真正positive transfer有限
2. Hand-tuned data sampling
3. Inference speed限制frequency

这篇工作对robot foundation model的意义: **证明了cross-embodied learning的flexibility不需要sacrifice performance**。下一个milestone可能是positive transfer的出现 —— 那时robot learning就真的进入"foundation model"时代了。

---

## References (key papers to read next)

1. **Open X-Embodiment / RT-X**: https://openx-embodiment.github.io —— dataset基础
2. **Octo**: https://octo-models.github.io —— CrossFormer的直接前驱
3. **Yang et al. (Pushing the limits)**: https://arxiv.org/abs/2402.19432 —— 主要对比baseline
4. **ACT / ALOHA**: https://tonyzhaozh.github.io/aloha —— bimanual baseline
5. **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu —— action head alternative
6. **GNM / ViNT**: https://general-navigation-models.github.io —— navigation baseline
7. **OpenVLA**: https://openvla.github.io —— VLA路线对比
8. **RoboCat**: https://arxiv.org/abs/2306.11706 —— DeepMind的flexible transformer前驱
9. **Gato (Reed et al.)**: https://arxiv.org/abs/2205.06175 —— sequence-to-sequence robot learning的先驱
10. **π0 (Physical Intelligence)**: https://www.physicalintelligence.company/blog/pi0 —— 同期更大scale的bimanual work
