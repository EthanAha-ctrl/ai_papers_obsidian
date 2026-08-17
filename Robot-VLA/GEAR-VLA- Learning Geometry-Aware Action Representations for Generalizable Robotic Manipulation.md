---
source_pdf: GEAR-VLA- Learning Geometry-Aware Action Representations for Generalizable
  Robotic Manipulation.pdf
paper_sha256: 3478ebb48a2a90e2b98ff9385a37bbcdf4a810d068022d290327efca1aa5a033
processed_at: '2026-08-04T12:57:48-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GEAR-VLA 用人话说

好，我把这篇paper拆成"遇到了什么麻烦 → 怎么搞的 → 为什么work"三段来讲。

---

## 核心矛盾：刷榜猛如虎，落地零杠五

现在的VLA模型有个很尴尬的现象。你拿LIBERO benchmark去测，OpenVLA、π0、π0.5一个个往上刷，98%、99%的success rate，看起来robot manipulation已经solved了。结果你把这些model扔到real world里，换个object颜色、换个background、换个robot，success rate直接跳水。

为什么？作者认为问题出在**representation design**上，具体三个洞：

**第一个洞：action representation没semantic meaning**

现在的VLA怎么做action？拿OpenVLA举例，它把continuous action trajectory量化成discrete tokens，扔进VLM当language token一样predict。π0用的是flow matching，但也是end-to-end让VLM直接输出action。

问题是这些action token本身没有semantic meaning。它只是trajectory的压缩编码。模型学到的是"看到这个pixel pattern → 输出这串token"，本质上在imitate trajectory。它没有学会"这个object的geometry决定了我应该从侧面approach"这种reasoning。

你换个稍微不一样的object，pixel pattern变了，模型就懵了。

**第二个洞：3D信息跟VLM不对齐**

VLM是从image-text pair训练出来的，2D semantic understanding很强，但3D spatial understanding很弱。你让它抓一个occluded的物体，或者判断两个物体之间的距离，它搞不定。

那加3D信息不就行了？没那么简单。DepthVLA加depth，SpatialVLA加3D position encoding，但这些都是当action head的condition用，没真正进VLM的semantic space。你要直接把3D features inject到VLM backbone里，会破坏VLM原本学到的visual-language alignment——因为3D features的distribution跟2D visual tokens的distribution差太远了。

**第三个洞：embodiment信息污染shared representation**

不同robot的DoF不同、kinematics不同、joint layout不同。现有方法要么给每个robot单独的action head（H-RDT），要么用soft prompt区分robot identity（X-VLA）。这两种做法都把"我是哪个robot"这个信息inject到了high-level policy representation里。

后果是什么？你pretrain的时候AgileX数据多、LDT-01数据少，model就会偏向AgileX。你transfer到一个pretraining没见过的新robot，因为representation里entangle了robot identity，transfer效率很低。

---

## GEAR-VLA的三个解法

### 解法一：Coarse-to-fine——先学会想，再学会做

核心intuition：**别一上来就让VLM输出continuous action，先让它学会"thinking about actions"**。

分两步走：

**Step 1: Embodied VLM Pretraining**

这一步VLM不碰continuous action，只学discrete action semantics。数据来源两个：

1. **Robot trajectories** → 用[FAST tokenizer](https://arxiv.org/abs/2501.09747)转成discrete action tokens。FAST本质上是一个压缩效率很高的action tokenizer，把30步action chunk压缩成一串short tokens。

2. **Human/web videos** → 用causal VQ-VAE提取latent action IDs。这步很关键，因为human videos没有robot action labels，但视觉上有action dynamics（物体移动、手在动、contact变化）。VQ-VAE把这些visual dynamics压缩成discrete codes。

这就scale了supervision signal——你不只从robot data学action，还从海量human/web video学action-relevant visual dynamics。

所有这些任务都formulate成autoregressive next-token prediction：
$$\mathcal{L}_{\mathrm{VLM}} = -\sum_i \log p_\theta(y_i | y_{<i}, O, l, s)$$

这里$y_i$可以是text、grounding、planning、FAST action token、或latent action ID。$O$是multi-view observations，$l$是language instruction，$s$是robot state，$y_{<i}$是之前所有tokens。$\theta$是模型参数。

**Step 2: Continuous Action Generation**

VLM学会了discrete action semantics之后，接一个DiT（Diffusion Transformer）action expert来生成continuous action。

关键设计：**DiT不直接访问VLM的full representation，只用latent action tokens的K/V cache**。而且用stop-gradient阻断gradient回流：

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{A,\epsilon,\tau}\left[\|v_\phi(A_\tau, \tau, \mathrm{sg}(h_{\mathrm{la}})) - (A - \epsilon)\|_2^2\right]$$
$$A_\tau = \tau A + (1-\tau)\epsilon$$

拆开讲：
- $A$：target action chunk（30步，30Hz，1秒horizon）
- $\epsilon$：Gaussian noise，跟$A$同维度
- $\tau \in [0,1]$：flow time。$\tau=0$是纯noise，$\tau=1$是纯target。$A_\tau = \tau A + (1-\tau)\epsilon$是两者的linear interpolation
- $h_{\mathrm{la}}$：VLM产生的latent action token的K/V cache
- $\mathrm{sg}(\cdot)$：**stop-gradient**，gradient到这就停，不往VLM传
- $v_\phi$：DiT的velocity prediction network，参数$\phi$。它预测的是从noise指向target的方向向量$(A - \epsilon)$

这就是[Flow Matching](https://arxiv.org/abs/2210.02747) / [Rectified Flow](https://arxiv.org/abs/2209.03003)的标准形式。推理时从$\tau=0$（纯noise）一路积分到$\tau=1$（target action）。

**为什么stop-gradient是关键？**

对比π0——π0是end-to-end的，VLM会接收action loss的gradient。这意味着VLM的representation会被action loss塑造，可能学到更action-relevant features，但也可能破坏原本的semantic understanding。

GEAR-VLA的哲学是**division of labor**：VLM负责"理解场景和action语义"，DiT负责"执行continuous control"。两者通过latent action tokens这个controlled interface通信，但各学各的。

这个思路跟[ControlNet](https://arxiv.org/abs/2302.05543)的zero-convolution、[BLIP-2](https://arxiv.org/abs/2301.12597)的Q-Former类似——用一个trainable bridge连接frozen components，避免互相干扰。但GEAR-VLA更subtle：VLM不是frozen的，它继续接收discrete supervision，只是不被continuous action loss更新。

**Latent Action Tokenizer的细节**

这个component借鉴了[VideoWorld2](https://arxiv.org/abs/2602.00000)和[LAPA](https://openreview.net/forum?id=YLMoZr3XqE)，但有关键区别：

- LAPA：encode initial frame + final frame → 单个latent action。只capture start和end state，中间dynamics丢失
- GEAR-VLA：encode连续video segment → temporally continuous latent code sequence

技术参数：
- Codebook size 16，分成2组每组8 entries
- 每个frame transition提取4个latent action codes
- Video frames sampled at **5 Hz**（robot actions 30 Hz）
- 1秒horizon → 5 frames × 4 codes = 20个latent codes

为什么5 Hz？如果跟robot action一样30 Hz，1秒会产生$30 \times 4 = 120$个latent codes，autoregressive decoding太慢。而且30 Hz的frames之间视觉信息高度redundant，反而weakens latent action learning。5 Hz保留了short-horizon dynamics，同时保持compact。

Codebook只有16个entries看起来很小（对比[VQ-VAE](https://arxiv.org/abs/1711.00937)原始用8192，[DALL-E](https://arxiv.org/abs/2102.12092)用8192）。作者的intuition应该是force compression of essential dynamics——codebook太小，model被迫只encode真正action-relevant的信息，丢弃appearance details。Grouped design（2组×8）可能是hierarchical structure——一组coarse motion category，一组fine parameters。论文没明说，但提到"grouped design is easier to optimize and helps alleviate codebook collapse"。

Ablation（Table 14）确认continuous latent action比LAPA-style好0.6 points（88.7 vs 88.1），比no latent action好1.6 points（88.7 vs 87.1）。

---

### 解法二：Semantic-aligned 3D——冻住2D，慢慢注入3D

核心问题：VLM的2D visual encoder已经跟language representation对齐了，你不能乱动。但3D信息又必须加进来。

GEAR-VLA的做法：

1. **Frozen 2D ViT**（来自[Qwen2.5-VL](https://arxiv.org/abs/2502.13923)）——保留VLM-aligned visual pathway不动
2. **Trainable [VGGT](https://arxiv.org/abs/2503.11651)**——3D spatial encoder，需要adapt到VLA space
3. **Zero-init 3D projector**——训练开始时3D features没影响

公式：
$$Z^{\mathrm{vis}} = [H^{2D}; H^{3D}] W_{\mathrm{vis}} + b, \quad W_{\mathrm{vis}}^{(0)} = [W_{\mathrm{Qwen}}; 0]$$

拆开讲：
- $H^{2D}$：frozen 2D ViT提取的features
- $H^{3D}$：trainable VGGT提取的3D features
- $[H^{2D}; H^{3D}]$：沿feature dimension拼接
- $W_{\mathrm{vis}}$：expanded visual projector，维度$[(d_{2D} + d_{3D}), d_{\mathrm{LLM}}]$。$d_{2D}$、$d_{3D}$是两个encoder的feature维度，$d_{\mathrm{LLM}}$是LLM的hidden dimension
- $W_{\mathrm{vis}}^{(0)}$：projector初始权重
- $W_{\mathrm{Qwen}}$：原始Qwen2.5-VL的projector权重（pretrained的）
- $0$：zero matrix，对应3D block
- $b$：bias

**Zero-init的magic**：训练step 0时，3D block权重全是0，所以$H^{3D} \cdot 0 = 0$。最终$Z^{\mathrm{vis}} = H^{2D} W_{\mathrm{Qwen}} + b$，模型行为跟原始Qwen2.5-VL完全一样。随着训练进行，3D block权重从0逐渐学习，3D features慢慢"渗"进representation。

这避免了unaligned 3D features在训练初期把semantic distribution搞崩。

这个trick跟[ControlNet](https://arxiv.org/abs/2302.05543)的zero-convolution、[LoRA](https://arxiv.org/abs/2106.09685)的zero-init gamma是同一个intuition——**新增的trainable component从identity开始，gradually diverge**。

**Ablation证据（Table 3）非常有说服力**：

| Variant | LIBERO-Plus Avg. | Drop |
|---------|-----------------|------|
| Full model | 88.7 | - |
| w/o VGGT（完全去掉3D） | 85.1 | -3.6 |
| Frozen VGGT（3D encoder不训） | 85.2 | -3.5 |
| w/o Zero-Init（随机初始化3D projector） | 81.9 | **-6.8** |
| Trainable 2D ViT（2D encoder也训） | 86.6 | -2.1 |

最惊人的是**去掉zero-init直接掉6.8 points**。unaligned 3D features在训练初期会严重destabilize learning。Frozen VGGT掉3.5说明固定3D features不够，必须adapt到VLA space。Trainable 2D ViT掉2.1说明破坏VLM-aligned pathway有害。

Figure 7的feature visualization直观展示了这点（20个ImageNet classes, 1000 samples/class的t-SNE）：
- (a) Original ViT：部分semantic separation但class overlap明显
- (b) Unfrozen ViT + Zero-init VGGT：破坏pretrained semantic space，class boundary更模糊
- (c) Frozen ViT + Zero-init VGGT（Ours）：inter-class separation更清晰，intra-class distribution更structured

---

### 解法三：Embodiment Canonicalization——robot差异留在底层

核心思路：**别把"我是哪个robot"这个信息inject到high-level representation里，把它限制在low-level interface**。

**Action side：Relative End-effector Action**

$$\Delta T_{t+i} = (T_t^{ee})^{-1} T_{t+i}^{ee}, \quad i = 1, \ldots, K$$

- $T_t^{ee} \in SE(3)$：当前end-effector pose，在robot base coordinate下。$SE(3)$是6-DoF刚体变换群
- $T_{t+i}^{ee} \in SE(3)$：第$i$步future target pose
- $(T_t^{ee})^{-1}$：当前pose的逆变换
- $\Delta T_{t+i} \in SE(3)$：相对于current pose的relative transform
- $K$：action chunk length（30）

Action chunk：
$$A_t^{\mathrm{rel}} = [\Delta T_{t+1}, \Delta T_{t+2}, \dots, \Delta T_{t+K}, g_{t+1:t+K}]$$

$g_{t+1:t+K}$是gripper开/关序列。

**关键设计**：所有future poses相对于**同一个current pose** $T_t^{ee}$，不是step-wise relative to previous pose。

对比三种action representation的trade-off：

| 方式 | 问题 |
|------|------|
| Absolute action $T_{t+i}^{ee}$ | 依赖robot base coordinate和workspace geometry，换robot就失效 |
| Step-wise delta $(T_{t+i-1}^{ee})^{-1} T_{t+i}^{ee}$ | chunk内累积误差——第一步错了，后续每一步都基于错误的previous pose |
| Relative to current $(T_t^{ee})^{-1} T_{t+i}^{ee}$（Ours） | 每个future pose独立预测，误差不累积；不依赖base coordinate |

这跟[Diffusion Policy](https://arxiv.org/abs/2303.04137)的receding horizon control思路相通——预测future chunk但只执行前几步然后replan，避免误差累积。

**State side：Embodiment-Aware State Projector**

State input：$s_t = \{T_t^{ee}, q_t\}$

- $T_t^{ee}$：end-effector pose
- $q_t$：joint angles

这里有个很巧妙的设计——**end-effector pose和joint angles的pairing implicitly encodes embodiment information**。

为什么？因为同样一个end-effector pose，14-DoF的AgileX和16-DoF的LDT-01的joint angle configuration完全不同。DoF数量、joint layout、kinematic constraints都encoded在这个pairing里。model通过看这个pairing就能推断"我现在在控制什么样的robot"。

每个embodiment $e$有自己的lightweight state projector：
$$z_t^s = f_{\psi_e}^s(s_t)$$

- $f_{\psi_e}^s$：embodiment $e$的state projector，参数$\psi_e$
- $s_t$：raw state
- $z_t^s$：mapped到shared VLA space的state representation

**Transfer到新robot的two-stage adaptation**：
1. **Stage 1**：Freeze VLA backbone，只train新的state projector——让新robot的state distribution对齐到shared representation
2. **Stage 2**：Light end-to-end fine-tuning——微调让整个系统adapt

这跟[LLaVA](https://arxiv.org/abs/2304.08485)的two-stage training（projector pretraining → instruction tuning）思路类似。

**Ablation证据（Table 3）**：

| Variant | Avg. | Drop |
|---------|------|------|
| Full model | 88.7 | - |
| One-stage adaptation（直接end-to-end） | 79.0 | **-9.7** |
| w/ X-VLA Soft Prompt | 85.0 | -3.7 |
| w/o Embodiment-Specific Projector | 86.7 | -2.0 |

One-stage adaptation掉9.7是最严重的degradation——直接fine-tune新robot会破坏shared representation。X-VLA-style soft prompt掉3.7说明inject robot identity into semantic representation有害。

---

## 实验里最值得关注的点

### LIBERO-Plus的zero-shot OOD（Table 9）

| Perturbation | OpenVLA | π0 | π0.5 | ACoT | Ours |
|-------------|---------|-----|------|------|------|
| Camera | 0.8 | 61.0 | 75.8 | 72.6 | **82.6** |
| Robot | 3.5 | 40.8 | 79.4 | 82.6 | **84.1** |
| Language | 23.0 | 63.5 | 83.3 | 87.5 | 82.4 |
| Light | 8.1 | 89.3 | 95.5 | 97.7 | **97.9** |
| Background | 34.8 | 84.1 | 95.0 | 96.5 | 93.1 |
| Noise | 15.2 | 80.1 | 89.6 | 87.8 | **90.0** |
| Layout | 28.5 | 76.4 | 87.0 | 88.1 | **89.4** |

GEAR-VLA在Camera（82.6）和Robot（84.1）perturbation上特别强——这正是3D geometry awareness和embodiment canonicalization直接针对的场景。Language perturbation上不如ACoT（82.4 vs 87.5），因为language understanding不是这篇paper的重点。

### Cross-embodiment transfer到LDT-01

这是最有说服力的实验。LDT-01是16-DoF bimanual robot，**pretraining中完全没有similar counterpart**。

只用200 demos/task做lightweight adaptation，结果：
- π0.5: 73.9%
- ACoT: 77.8%
- GEAR-VLA: **81.0%**

比π0.5高7.1 points。这证明embodiment canonicalization真的实现了transfer to unseen robot。

### Universal Grasping——6,360 trials的大规模验证

| Method | Sparse | Dense | BG/Light | Overall |
|--------|--------|-------|----------|---------|
| π0.5 | 84.1 | 77.3 | 75.9 | 79.1 |
| [DexGraspVLA](https://arxiv.org/abs/2602.11236) | 88.9 | 83.1 | 81.3 | 84.4 |
| GEAR-VLA | **91.7** | **89.7** | **89.0** | **90.1** |

规模：212个unseen objects × 3个settings × 10 trials = 6,360 real-robot trials per method。

关键细节：DexGraspVLA需要persistent target-mask tracking（每帧都要给mask），GEAR-VLA只用first-frame mask。因为Embodied Pretraining里有mask-tracking task，model学会了从first-frame mask推断后续observation中的target object。

按object category看gain最大的：
- **Irregular objects**：+16.8 points over π0.5（86.7 vs 69.9）
- **Tool objects**：+20.0 points（86.7 vs 66.7）
- **Dense scenes**：+12.4 points（89.7 vs 77.3）

Irregular和tool objects上gain最大，说明geometry-aware grounding在non-trivial geometry下优势明显。Dense scenes的gain说明3D spatial understanding helps disambiguate target from distractors。

### Data Efficiency（Table 15）

| Data Ratio | π0 | π0.5 | ACoT | Ours |
|-----------|-----|------|------|------|
| 25% | 33.1 | 55.8 | 59.8 | **69.1** |
| 50% | 49.8 | 69.8 | 64.2 | **76.5** |
| 75% | 56.9 | 74.7 | 71.1 | **80.5** |
| 100% | 64.2 | 80.6 | 77.8 | **85.9** |

25% data时GEAR-VLA比π0.5高13.3 points。低data regime下advantage更大，说明pretrained representation的generalization power——representation学得好，少量demos就能adapt。

---

## 一个practical trick：Attention-Level Modality Dropping

训练时随机drop某些modality的attention weights：
- 0.2 prob：drop wrist-view image tokens的attention
- 0.2 prob：drop robot-state tokens的attention
- 0.2 prob：drop both
- 0.4 prob：keep full attention

**Intuition**：
- Drop wrist → model被迫用head-view global context，improve layout robustness
- Drop state → model被迫rely on visual evidence，improve robustness under robot/camera/lighting变化

这跟[MAE](https://arxiv.org/abs/2111.06377)的random masking、multi-modal learning中的[modality dropout](https://arxiv.org/abs/2205.06168)思路一致——防止model over-rely on某个modality。

Ablation（Table 12）：
- w/o Dropping: 87.7
- Only Drop Wrist: 88.1
- Only Drop State: 88.5
- Drop Both: **88.7**

Drop state的gain比drop wrist大，说明model更容易over-rely on proprioceptive state。

---

## 把三个design串起来看

GEAR-VLA的三个design其实是在解决同一个根本问题的三个层面——**怎么让representation既semantic又geometric又embodiment-agnostic**。

1. **Coarse-to-fine**解决action representation的semantic gap：先学discrete action semantics（从robot trajectories + human videos），再学continuous action execution。Stop-gradient保护VLM的semantic understanding不被action loss破坏。

2. **Semantic-aligned 3D**解决spatial representation的alignment gap：Frozen 2D ViT保semantic，Trainable VGGT加geometry，Zero-init projector控制3D features的注入节奏。

3. **Embodiment canonicalization**解决embodiment representation的transfer gap：Relative SE(3) action消除coordinate dependence，embodiment-specific state projector把robot差异限制在low-level interface。

三个design的ablation drop分别是：
- Discrete Action Learning：-1.6到-2.5
- 3D Geometry Integration：-3.5到-6.8
- Embodiment Canonicalization：-2.0到-9.7

3D和embodiment的design影响最大，说明geometry awareness和cross-embodiment transfer是当前VLA generalization的真正瓶颈。

---

## 我的critique和思考

### Stop-gradient的trade-off

Stop-gradient保护了VLM的semantic understanding，但可能限制了VLM和action expert之间的co-adaptation。π0的end-to-end设计允许VLM representations被action loss塑造，可能学到更action-relevant features。

一个可能改进：用gradient scaling而不是完全stop。比如scale action loss gradient to VLM by 0.1，允许少量co-adaptation但不破坏semantic stability。这跟[ReFL](https://arxiv.org/abs/2310.18491)的直接reward backprop思路相关。

### Latent Action Tokenizer的expressiveness

Codebook size 16很小。虽然ablation确认work，但可能限制了action semantics的expressiveness。可能改进：用residual VQ（[SoundStream](https://arxiv.org/abs/2107.03312) style）或[FSQ](https://arxiv.org/abs/2309.15507)替代VQ，增加expressiveness而不增加codebook size。

### VGGT的computational cost

VGGT是feed-forward 3D model但需要multi-view input。在30Hz real-time control中可能成为bottleneck。论文没报告inference latency。可能需要cache 3D features across timesteps或用更轻量的3D encoder。

### Relative Action的limitation

Relative End-effector Action对大多数manipulation有效，但需要absolute positioning的任务（"把东西放到fixed location"）可能不理想。workspace geometry信息丢失了。

### Grasping vs. Complex Manipulation

212 unseen objects的grasping很impressive，但grasping是简单task（reach + grasp + lift）。更complex manipulation（tool use, assembly, deformable object manipulation）的generalization还需要更大scale验证。

### Training scale的reproducibility

240 H200 GPUs对大多数lab不可行。希望作者release weights（[project page](https://babynabeauty.github.io/gear-vla-p/)）。

---

## 跟其他VLA的关系图

从representation design角度对比：

| 维度 | [OpenVLA](https://openvla.github.io/) | [π0](https://roboticsconference.org/program/papers/10/) | [X-VLA](https://arxiv.org/abs/2510.10274) | [SpatialVLA](https://roboticsconference.org/program/papers/62/) | GEAR-VLA |
|------|---------|---------|-------|------------|----------|
| Action | Tokenized | Flow matching | Tokenized | Tokenized | Coarse-to-fine |
| VLM gradient | End-to-end | End-to-end | End-to-end | End-to-end | Stop-gradient |
| 3D awareness | None | None | None | Position encoding | VGGT + zero-init |
| Embodiment | Robot-specific | Robot-specific | Soft prompts | Robot-specific | Canonicalization |
| Pretraining | VLM only | VLM + robot | VLM + robot | VLM + spatial | VLM + embodied + latent action |

GEAR-VLA在每个维度都做了distinct design choice，而且这些choice在ablation中都被验证有效。

---

## 一句话总结

GEAR-VLA的核心insight：**VLA的generalization瓶颈不在model capacity，而在representation design**。Action需要semantic grounding（不只是trajectory compression），3D需要alignment（不只是raw features注入），embodiment需要canonicalization（不只是soft prompts）。三个representation层面的改进比单纯scale up model更effective。

Related links：
- [Project page](https://babynabeauty.github.io/gear-vla-p/)
- [π0](https://roboticsconference.org/program/papers/10/) | [π0.5](https://proceedings.mlr.press/v305/black25a.html)
- [FAST tokenizer](https://arxiv.org/abs/2501.09747) | [LAPA](https://openreview.net/forum?id=YLMoZr3XqE)
- [VGGT](https://vgg-t.github.io/) | [ControlNet](https://arxiv.org/abs/2302.05543)
- [LIBERO](https://libero-project.github.io/) | [LIBERO-Plus](https://arxiv.org/abs/2510.13626) | [RoboTwin 2.0](https://arxiv.org/abs/2506.18088)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | [Flow Matching](https://arxiv.org/abs/2210.02747)

---

# GEAR-VLA: Geometry-Aware Action Representations for VLA

这篇paper试图解决一个根本问题：**为什么VLA模型在benchmark上刷分很高，但real-world deployment还是不行？** 作者认为根源在于缺乏一个统一的geometry-aware manipulation representation。具体表现为三个gap：action representation gap（tokenized actions导致trajectory imitation而非embodied reasoning）、spatial representation gap（3D features与VLM semantic space不对齐）、embodiment representation gap（robot-specific设计entangle了shared policy）。

GEAR-VLA的解法是三个core design：**Coarse-to-fine action learning**、**Semantic-aligned 3D integration**、**Embodiment canonicalization**。下面我逐个深入讲。

---

## 1. Coarse-to-Fine Action Learning

### 1.1 核心intuition

直接用低层trajectory做supervision训练VLA有个根本问题：VLM会被推向**trajectory fitting**而不是**embodied reasoning**。这跟OpenVLA、RT-2直接把action量化成token的问题类似——token本身没有semantic meaning，只是trajectory的压缩编码。模型学到的是"在这个pixel configuration下输出这个action sequence"，而不是"这个object需要被这样grasp因为它的geometry是这样的"。

GEAR-VLA的思路是**先让VLM学会thinking about actions，再学会producing actions**。分两stage：

**Stage 1: Embodied VLM Pre-training**
- 用FAST-style action tokens提供discrete supervision from robot trajectories
- 同时用causal VQ-VAE latent action IDs从manipulation videos提取high-level action semantics
- 关键insight：latent action IDs可以从human videos、web videos提取，**不需要robot action labels**
- 所有任务formulated为autoregressive token prediction

**Stage 2: Continuous Action Generation**
- 接一个DiT-based action expert
- DiT不直接访问full VLM representation，**只用latent action tokens的K/V cache**
- Flow matching loss训练
- Stop-gradient防止continuous action loss破坏VLM backbone

### 1.2 公式解析

公式(1)是Stage 1的loss：
$$\mathcal{L}_{\mathrm{VLM}} = -\sum_i \log p_\theta(y_i | y_{<i}, O, l, s)$$

变量含义：
- $y_i$：任意target token，包括text、grounding、planning、FAST action、latent action
- $y_{<i}$：之前所有tokens（autoregressive）
- $O$：multi-view observations（图像）
- $l$：language instruction
- $s$：robot state
- $\theta$：模型参数

这就是标准的next-token prediction loss，但target tokens非常diverse——既有language又有action，既有discrete robot actions又有latent visual dynamics。

公式(2)是Stage 2的flow matching loss：
$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{A,\epsilon,\tau}\left[\|v_\phi(A_\tau, \tau, \mathrm{sg}(h_{\mathrm{la}})) - (A - \epsilon)\|_2^2\right]$$
$$A_\tau = \tau A + (1-\tau)\epsilon$$

变量含义：
- $A$：target continuous action chunk（30 steps，1秒horizon，30Hz）
- $\epsilon$：Gaussian noise（与$A$同维度）
- $\tau \in [0,1]$：flow time，0表示纯noise，1表示纯target
- $A_\tau$：noise和target的linear interpolation
- $h_{\mathrm{la}}$：latent action token cache from VLM（VLM前向传播产生的K/V cache）
- $\mathrm{sg}(\cdot)$：**stop-gradient**，阻断gradient从DiT回流到VLM
- $v_\phi$：DiT velocity prediction network，参数为$\phi$

这是**Rectified Flow / Flow Matching**的standard form。$v_\phi$预测velocity $(A - \epsilon)$，即从noise指向target的方向。推理时从$\tau=0$（纯noise）积分到$\tau=1$（target action）。

**最关键的设计是$\mathrm{sg}(h_{\mathrm{la}})$**。这意味着DiT的gradient不会backprop到VLM backbone。对比π0（[Black et al., 2025](https://roboticsconference.org/program/papers/10/)）是end-to-end训练的，VLM会接收action loss的gradient。GEAR-VLA认为这会破坏VLM的semantic understanding。

这个stop-gradient设计让我联想到几个相关工作：
- [ControlNet](https://arxiv.org/abs/2302.05543)的zero-convolution，也是decouple条件信息与主干
- [BLIP-2](https://arxiv.org/abs/2301.12597)的Q-Former，用一个trainable bridge连接frozen visual encoder和frozen LLM
- [LLaVA](https://arxiv.org/abs/2304.08485)的projector设计

但GEAR-VLA的decoupling更subtle——它不是freeze VLM，而是让VLM继续接收discrete supervision（FAST tokens、latent action IDs），只是**不被continuous action loss更新**。这是一种"VLM学action semantics，DiT学action execution"的division of labor。

### 1.3 Causal VQ-VAE Latent Action Tokenizer

这个component借鉴了[VideoWorld2](https://arxiv.org/abs/2602.00000)和[LAPA](https://arxiv.org/abs/2410.11758)（[Ye et al., ICLR 2025](https://openreview.net/forum?id=YLMoZr3XqE)），但有关键改进。

**跟LAPA的区别**：
- LAPA：encode initial state + final state → 单个latent action
- GEAR-VLA：encode continuous video segments → temporally continuous latent code sequence

**技术细节**：
- Codebook size 16，分成2组，每组8 entries
- 每个frame transition提取4个latent action codes
- Grouped design缓解codebook collapse和overfitting
- Video frames sampled at **5 Hz**，robot actions at **30 Hz**
- 1秒horizon → 5 frames × 4 codes = 20 latent codes

为什么5 Hz而不是30 Hz？如果30 Hz，1秒autoregressive inference会产生$30 \times 4 = 120$个latent codes，slows down decoding。而且overly dense frames包含redundant visual information，weakens latent action learning。5 Hz保留了short-horizon visual dynamics同时保持compact code sequence。

这个small codebook（size 16）+ grouping的设计让我想到[SoundStream](https://arxiv.org/abs/2107.03312)和[EnCodec](https://arxiv.org/abs/2210.13438)的residual VQ，但这里是**parallel groups**而不是residual。Grouping的intuition我推测是hierarchical structure——可能一组coarse motion category，一组fine motion parameters。论文没明确说，但提到"grouped design is easier to optimize"。

**Masked cross-attention**防止future information leakage：每个query（对应一个frame）只能attend到current和previous frames的visual features。这保证了latent action extraction是causal的，可以从video online提取。

参考Figure 5和Figure 6的架构。Causal VQ-VAE的decoder在训练时reconstruct后续visual changes，鼓励codes保留action-relevant dynamics（object displacement、contact changes、hand/robot motion、scene-state transitions）。训练后decoder丢弃，只保留encoder + attention + quantizer。

---

## 2. Semantic-Aligned 3D Geometry Integration

### 2.1 核心问题

VLM有strong 2D semantic representations，但limited 3D spatial understanding。现有方法的困境：
- [DepthVLA](https://arxiv.org/abs/2510.13375)：加depth，但depth只是action-head condition，没进入VLM semantic space
- [SpatialVLA](https://arxiv.org/abs/2502.11536)（[Qu et al., RSS 2025](https://roboticsconference.org/program/papers/62/)）：3D position encodings，但not naturally aligned with VLM token space
- 直接inject 3D features到VLM backbone → perturb visual-token distribution

### 2.2 GEAR-VLA的解法

用[VGGT](https://arxiv.org/abs/2503.11651)（[Wang et al., CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.pdf)）作为3D spatial encoder。VGGT是feed-forward 3D model，输入multi-view images，输出3D features。相比COLMAP或单目depth，VGGT exploits multi-view consistency来model scene layout、object shape、spatial relations。

**关键设计**：
1. **Freeze 2D visual encoder**（保留VLM-aligned visual pathway）
2. **Trainable VGGT**（3D encoder需要adapt to VLA representation space）
3. **Zero-init 3D projector**（训练开始时3D features无影响）

公式(3)：
$$Z^{\mathrm{vis}} = [H^{2D}; H^{3D}] W_{\mathrm{vis}} + b, \quad W_{\mathrm{vis}}^{(0)} = [W_{\mathrm{Qwen}}; 0]$$

变量含义：
- $H^{2D}$：2D visual encoder（frozen ViT from Qwen2.5-VL）的features
- $H^{3D}$：3D spatial encoder（VGGT）的features
- $[H^{2D}; H^{3D}]$：沿feature dimension concatenation
- $W_{\mathrm{vis}}$：expanded visual projector，维度为$[(d_{2D} + d_{3D}), d_{\mathrm{LLM}}]$
- $W_{\mathrm{vis}}^{(0)}$：projector的初始化
- $W_{\mathrm{Qwen}}$：原始Qwen2.5-VL的projector权重
- $0$：zero matrix，对应3D block
- $b$：bias
- $Z^{\mathrm{vis}}$：最终输入LLM的visual tokens

**Zero-init是关键trick**。训练step 0时，$H^{3D} \cdot 0 = 0$，所以$Z^{\mathrm{vis}} = H^{2D} W_{\mathrm{Qwen}} + b$，模型行为跟原始Qwen2.5-VL完全一致。随着训练进行，3D block的权重从0逐渐学习，3D features慢慢融入representation。这避免了unaligned 3D features在训练初期破坏semantic distribution。

这个设计跟[ControlNet](https://arxiv.org/abs/2302.05543)的zero-convolution、[LoRA](https://arxiv.org/abs/2106.09685)的zero-init gamma、[AdapterFusion](https://arxiv.org/abs/2106.07808)的zero-init是同样的intuition——**新增的trainable component应该从identity开始，gradually diverge**。

### 2.3 Ablation证据

Table 3的3D Geometry Integration group非常有说服力：

| Variant | Avg. | Δ |
|---------|------|---|
| Full model | 88.7 | - |
| w/o VGGT | 85.1 | -3.6 |
| Frozen VGGT | 85.2 | -3.5 |
| w/o Zero-Init 3D Projector | 81.9 | **-6.8** |
| Trainable 2D ViT | 86.6 | -2.1 |

**最惊人的是w/o Zero-Init掉6.8 points**。这证明unaligned 3D features会严重destabilize learning。Frozen VGGT掉3.5 points说明fixed 3D features不够，必须adapt到VLA space。Trainable 2D ViT掉2.1 points说明破坏VLM-aligned pathway有害。

Figure 7的feature visualization（20 ImageNet classes, 1000 samples/class）直观展示了这一点：
- (a) Original ViT：部分semantic separation，class overlap明显
- (b) Unfrozen ViT + Zero-init VGGT：破坏pretrained semantic space
- (c) Frozen ViT + Zero-init VGGT（Ours）：更清晰inter-class separation，更structured intra-class distribution

---

## 3. Embodiment Canonicalization

### 3.1 核心问题

cross-embodiment transfer的现有方法：
- Robot-specific action heads（[H-RDT](https://arxiv.org/abs/2602.11236)）：entangle robot identity with shared policy
- Embodiment prompts（[X-VLA](https://arxiv.org/abs/2510.10274)）：inject robot identity into high-level representation，weakening transfer

GEAR-VLA的思路：**把robot-specific variation限制在low-level interface**。

### 3.2 Action side: Relative End-effector Action

公式(4)：
$$\Delta T_{t+i} = (T_t^{ee})^{-1} T_{t+i}^{ee}, \quad i = 1, \ldots, K$$

变量含义：
- $T_t^{ee} \in SE(3)$：当前end-effector pose，在robot base coordinate下
- $T_{t+i}^{ee} \in SE(3)$：第$i$个future target end-effector pose
- $(T_t^{ee})^{-1}$：当前pose的逆变换
- $\Delta T_{t+i} \in SE(3)$：相对于current pose的relative transform
- $K$：action chunk length（30 steps）
- $SE(3)$：Special Euclidean group，6-DoF rigid body transformation

公式(5)：
$$A_t^{\mathrm{rel}} = [\Delta T_{t+1}, \Delta T_{t+2}, \dots, \Delta T_{t+K}, g_{t+1:t+K}]$$

变量含义：
- $A_t^{\mathrm{rel}}$：predicted action chunk
- $g_{t+1:t+K}$：gripper command sequence（开/关）

**关键设计**：所有future poses相对于**同一个current pose** $T_t^{ee}$，而不是step-wise relative to previous pose。

对比三种action representation：
1. **Absolute action**：$T_{t+i}^{ee}$ directly。依赖robot base coordinate和workspace geometry，换robot就失效。
2. **Step-wise delta**：$T_{t+i-1}^{ee}{}^{-1} T_{t+i}^{ee}$。会在chunk内累积误差——第一步预测有误差，后续每一步都基于错误的previous pose。
3. **Relative to current**（GEAR-VLA）：$(T_t^{ee})^{-1} T_{t+i}^{ee}$。每个future pose独立预测，误差不在chunk内累积。

这让我想到[Diffusion Policy](https://arxiv.org/abs/2303.04137)（[Chi et al., RSS 2023](https://roboticsconference.org/program/papers/17/)）的receding horizon control——也是预测future chunk但只执行前几步然后replan。GEAR-VLA的relative action进一步消除了coordinate frame dependence。

### 3.3 State side: Embodiment-Aware State Projector

State input：$s_t = \{T_t^{ee}, q_t\}$

变量含义：
- $T_t^{ee}$：end-effector pose
- $q_t$：joint angles

**End-effector pose + joint angles的pairing implicitly captures embodiment information**——DoF数量、joint layout、kinematic constraints都encoded在这个pairing中。比如14-DoF AgileX和16-DoF LDT-01的$q_t$维度不同，这个差异通过state projector处理。

每个embodiment $e$有lightweight state projector：
$$z_t^s = f_{\psi_e}^s(s_t)$$

变量含义：
- $f_{\psi_e}^s$：embodiment $e$的state projector，参数$\psi_e$
- $s_t$：raw state input
- $z_t^s$：mapped state representation in shared VLA space

**Two-stage lightweight adaptation for new robot**：
1. **Stage 1**：Freeze VLA backbone，只train new state projector
2. **Stage 2**：Light end-to-end fine-tuning

这跟[LLaVA](https://arxiv.org/abs/2304.08485)的two-stage training（projector pretraining → instruction tuning）思路类似，但这里是为了adapt new embodiment。

### 3.4 Ablation证据

Table 3的Embodiment Canonicalization group：

| Variant | Avg. | Δ |
|---------|------|---|
| Full model | 88.7 | - |
| One-stage adaptation | 79.0 | **-9.7** |
| w/ X-VLA Soft Prompt | 85.0 | -3.7 |
| w/o Embodiment-Specific Projector | 86.7 | -2.0 |

**One-stage adaptation掉9.7 points是最严重的degradation**。直接end-to-end fine-tune new robot会破坏shared representation。X-VLA-style soft prompt掉3.7 points说明inject robot identity into semantic representation有害。

Table 13进一步确认：
- No Embodiment Canonicalization: 86.9 (-1.8)
- w/ X-VLA Soft Prompt + Embodiment-Aware State + Output-Aware: 85.0 (-3.7)
- Embodiment-Aware State + Output-Aware (w/o Soft Prompt): 88.1 (-0.6)
- w/o Embodiment-Specific State Projector (Output-Aware only): 86.7 (-2.0)
- Ours (Embodiment-Aware State only): 88.7

最优配置是只用Embodiment-Aware State，不加soft prompt，不加output-aware conditioning。

---

## 4. Attention-Level Modality Dropping

这是一个很practical的regularization trick（Table 12）。

**策略**：
- 0.2 probability：drop wrist-view image tokens的attention weights
- 0.2 probability：drop robot-state tokens的attention weights
- 0.2 probability：drop both
- 0.4 probability：keep full attention

**Intuition**：
- Drop wrist → model被迫用head-view global context，improve layout robustness
- Drop state → model被迫rely on visual evidence，improve robustness under robot/camera/lighting/background/noise variations

这个设计让我想到[MAE](https://arxiv.org/abs/2111.06377)的random masking和[Dropout](https://arxiv.org/abs/1207.0580)，但是applied to **attention weights from action-prediction queries to specific modality tokens**。也类似multi-modal learning中的[modality dropout](https://arxiv.org/abs/2205.06168)。

Ablation结果：
- w/o Dropping: 87.7
- Only Drop Wrist: 88.1
- Only Drop State: 88.5
- Drop Both (Ours): 88.7

Drop state的gain比drop wrist大，说明model更容易over-rely on proprioceptive state。

---

## 5. 实验结果的关键Insights

### 5.1 Simulation Benchmarks (Table 1)

| Benchmark | OpenVLA | π0 | π0.5 | ACoT | Ours |
|-----------|---------|-----|------|------|------|
| LIBERO Avg. | 76.5 | 94.4 | 96.9 | 98.5 | **98.7** |
| LIBERO-Plus Avg. | 15.6 | 69.4 | 85.7 | 86.6 | **88.7** |
| RoboTwin-2.0 Clean | 38.3 | 48.4 | 82.7 | 80.1 | **91.1** |
| RoboTwin-2.0 Rand | 26.7 | 26.4 | 76.8 | 78.7 | **89.9** |

**LIBERO已经saturate**，98.7 vs 98.5差别不大。真正有说服力的是：
- **LIBERO-Plus**（zero-shot OOD）：88.7 vs 86.6，+2.1 points。7个perturbation settings（Camera, Robot, Language, Light, Background, Noise, Layout）
- **RoboTwin 2.0**：+11 points over ACoT。这是large-scale multi-task benchmark with domain randomization

Table 9的LIBERO-Plus详细数据很有意思。GEAR-VLA在Camera（82.6）和Robot（84.1）perturbation上特别强，说明3D geometry awareness和embodiment canonicalization确实work。但在Language（82.4）上不如ACoT（87.5），可能因为language understanding不是这篇paper的重点。

### 5.2 Real-World Bimanual Manipulation (Figure 3)

三个contact-rich tasks：T-shirt folding, shorts folding, parcel-label flipping。

- **AgileX (14-DoF dual-arm)**：85.9% avg，每个task 200 demos，测试3个unseen colors
- **LDT-01 (16-DoF bimanual, pretraining-unseen)**：81.0%，比π0.5高7.1 points

**LDT-01的结果最有说服力**——这是一个pretraining中完全没有similar counterpart的embodiment。81.0% success说明embodiment canonicalization真的实现了transfer to unseen robot。

Table 15的数据效率分析也很有价值：
- 25% data: GEAR-VLA 69.1 vs π0.5 55.8
- 50% data: 76.5 vs 69.8
- 75% data: 80.5 vs 74.7
- 100% data: 85.9 vs 80.6

低data regime下advantage更大（25% data时+13.3 points），说明pretrained representation的generalization power。

### 5.3 Universal Grasping Benchmark (Table 2)

这是最impressive的实验。规模：
- 212 unseen objects
- 3 settings: sparse clutter (4-8 bg objects), dense clutter (8-20 bg objects), background/lighting variation
- 每个object 10 trials → 2,120 trials per setting → **6,360 trials total per method**

结果：
- π0.5: 79.1%
- [DexGraspVLA](https://arxiv.org/abs/2602.11236): 84.4%
- GEAR-VLA: **90.1%**

**关键细节**：DexGraspVLA需要persistent target-mask tracking，GEAR-VLA只用first-frame mask。这得益于Embodied Pretraining中的mask-tracking task——model学会了从first-frame mask推断后续observation中的target object。

按object category分析：
- Irregular objects: GEAR-VLA 86.7 vs π0.5 69.9 (+16.8)
- Tool objects: 86.7 vs 66.7 (+20.0)
- Dense scenes: 89.7 vs 77.3 (+12.4)

Irregular和tool objects上gain最大，说明geometry-aware grounding在non-trivial geometry下更强。Dense scenes的gain说明3D spatial understanding helps disambiguate target from distractors。

---

## 6. Training Configuration

- **Hardware**: 240 H200 GPUs
- **Embodied Pretraining**: 350K iterations, batch 8/GPU
- **Continuous Policy Learning**: 700K iterations, batch 4/GPU
- **LIBERO fine-tuning**: 56 H200 GPUs, 12K iterations
- **Optimizer**: AdamW, lr $2 \times 10^{-5}$, weight decay 0, max grad norm 1.0
- **Precision**: bf16 mixed precision
- **Schedule**: constant with 3% warmup
- **Image resolution**: 448
- **Action chunk**: 30 steps @ 30Hz = 1 second horizon

**Data composition**（Table 4, 5）：
- General VQA: 132k samples
- 2D Trajectory: 698k
- 3D Grounding: 632k
- Space Pointing: 625k
- Mask Tracking: 445k
- OXE: 3000h
- AgiBot: 3276h
- Ego4D: 3670h
- Egocentric-10k: 10000h

总数据量非常大，manipulation data约13,000+ hours，human videos约5,000+ hours。

---

## 7. 跟其他VLA的对比

| 维度 | OpenVLA | π0/π0.5 | X-VLA | SpatialVLA | GEAR-VLA |
|------|---------|---------|-------|------------|----------|
| Action representation | Tokenized | Flow matching | Tokenized | Tokenized | Coarse-to-fine |
| VLM gradient | End-to-end | End-to-end | End-to-end | End-to-end | Stop-gradient decoupled |
| 3D awareness | None | None | None | Position encoding | VGGT + zero-init |
| Embodiment handling | Robot-specific | Robot-specific | Soft prompts | Robot-specific | Canonicalization |
| Pretraining | VLM only | VLM + robot | VLM + robot | VLM + spatial | VLM + embodied + latent action |

GEAR-VLA的独特之处在于：
1. **Stop-gradient decoupling**——VLM不被continuous action loss更新
2. **Dual discrete supervision**——FAST tokens + latent action IDs
3. **Zero-init 3D integration**——保护semantic space同时融入geometry
4. **Embodiment canonicalization**——low-level interface处理robot差异

---

## 8. 我的思考和Critique

### 8.1 Stop-gradient的trade-off

Stop-gradient虽然保护了VLM的semantic understanding，但可能限制了VLM和action expert之间的**co-adaptation**。π0的end-to-end设计允许VLM representations被action loss塑造，可能学到更action-relevant features。GEAR-VLA的VLM只接收discrete supervision，可能miss一些fine-grained action-relevant信息。

一个可能改进：用**gradient scaling**而不是完全stop。比如scale action loss gradient to VLM by 0.1，允许少量co-adaptation但不破坏semantic stability。这跟[ReFL](https://arxiv.org/abs/2310.18491)的直接reward backprop思路相关。

### 8.2 Latent Action Tokenizer的expressiveness

Codebook size 16很小。[VQ-VAE](https://arxiv.org/abs/1711.00937)原始工作用8192，[DALL-E](https://arxiv.org/abs/2102.12092)用8192。GEAR-VLA用16可能是为了force compression of essential dynamics，但可能限制了action semantics的expressiveness。不过ablation（Table 14）显示它确实work，连续latent action（5 frames）比LAPA-style（initial/final only）好0.6 points。

可能的改进：用**residual VQ**（[SoundStream](https://arxiv.org/abs/2107.03312) style）增加expressiveness而不增加codebook size。或者用[FSQ](https://arxiv.org/abs/2309.15507)（Finite Scalar Quantization）替代VQ，避免codebook collapse。

### 8.3 VGGT的computational cost

VGGT是feed-forward 3D model但需要multi-view input。在real-time deployment（30Hz control）中，VGGT的前向传播可能成为bottleneck。论文没报告inference latency。可能的改进：用更轻量的3D encoder（如[DUSt3R](https://arxiv.org/abs/2312.14732)的简化版）或cache 3D features across timesteps。

### 8.4 Relative Action的limitation

Relative End-effector Action对大多数manipulation task有效，但对于需要**absolute positioning**的任务（比如"把东西放到fixed location"）可能不理想。这种情况下workspace geometry information丢失了。论文没有讨论这种edge case。

### 8.5 Grasping vs. Complex Manipulation

212 unseen objects的grasping benchmark很impressive，但grasping是相对简单的task——只需要reach + grasp + lift。更complex manipulation（如tool use, assembly, deformable object manipulation）的generalization还需要验证。Real-world bimanual experiments（T-shirt folding等）部分address了这点，但scale远小于grasping benchmark。

### 8.6 Reproducibility

240 H200 GPUs的training scale对大多数lab来说不可行。这限制了reproducibility。希望作者能release modelweights（project page: [https://babynabeauty.github.io/gear-vla-p/](https://babynabeauty.github.io/gear-vla-p/)）。

---

## 9. 相关工作链接

- **π0**: [Black et al., RSS 2025](https://roboticsconference.org/program/papers/10/)
- **π0.5**: [Black et al., CoRL 2025](https://proceedings.mlr.press/v305/black25a.html)
- **OpenVLA**: [Kim et al., CoRL 2024](https://openvla.github.io/)
- **FAST tokenizer**: [Pertsch et al., 2025](https://arxiv.org/abs/2501.09747)
- **LAPA**: [Ye et al., ICLR 2025](https://openreview.net/forum?id=YLMoZr3XqE)
- **VGGT**: [Wang et al., CVPR 2025](https://vgg-t.github.io/)
- **LIBERO**: [Liu et al., NeurIPS 2023](https://libero-project.github.io/)
- **LIBERO-Plus**: [Fei et al., 2025](https://arxiv.org/abs/2510.13626)
- **RoboTwin 2.0**: [Chen et al., 2025](https://arxiv.org/abs/2506.18088)
- **Diffusion Policy**: [Chi et al., RSS 2023](https://diffusion-policy.cs.columbia.edu/)
- **X-VLA**: [Zheng et al., 2025](https://arxiv.org/abs/2510.10274)
- **SpatialVLA**: [Qu et al., RSS 2025](https://roboticsconference.org/program/papers/62/)
- **ControlNet**: [Zhang et al., 2023](https://arxiv.org/abs/2302.05543)
- **Flow Matching**: [Lipman et al., ICLR 2023](https://arxiv.org/abs/2210.02747)
- **Rectified Flow**: [Liu et al., 2022](https://arxiv.org/abs/2209.03003)
- **DexGraspVLA**: [Zhong et al., AAAI 2026](https://arxiv.org/abs/2602.11236)
- **VideoWorld2**: [Ren et al., 2026](https://arxiv.org/abs/2602.00000)

---

## 总结

GEAR-VLA的核心贡献是**一个统一的geometry-aware action representation**，通过三个design实现：

1. **Coarse-to-fine action learning**：先学action semantics（discrete），再学action execution（continuous），用stop-gradient decouple两者。Latent action IDs从action-free videos提取，scale了supervision signal。

2. **Semantic-aligned 3D integration**：Frozen 2D ViT + Trainable VGGT + Zero-init projector。Zero-init是关键——避免unaligned 3D features破坏semantic distribution（ablation显示掉6.8 points）。

3. **Embodiment canonicalization**：Relative SE(3) action + embodiment-specific state projector。把robot差异限制在low-level interface，不inject到shared representation。Two-stage adaptation实现transfer to unseen embodiment。

这三个design的组合产生了strong generalization：LIBERO-Plus zero-shot 88.7%、LDT-01 unseen embodiment 81.0%、212 unseen objects grasping 90.1%（6,360 trials）。Ablation全面支持每个design的贡献，特别是zero-init 3D projector和two-stage embodiment adaptation。

从intuition角度，这篇paper的核心insight是：**VLA的generalization瓶颈不在model capacity，而在representation design**。Action representation需要semantic grounding而不只是trajectory compression；spatial representation需要alignment with VLM semantic space而不只是raw 3D features；embodiment representation需要canonicalization to low-level interface而不只是soft prompts。这三个representation层面的改进比单纯scale up model更effective。
