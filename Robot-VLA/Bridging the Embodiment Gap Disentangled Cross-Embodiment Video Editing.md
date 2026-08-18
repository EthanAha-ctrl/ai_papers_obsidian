---
source_pdf: Bridging the Embodiment Gap Disentangled Cross-Embodiment Video Editing.pdf
paper_sha256: 9c405e0b9c7def41d66de9145ef9c6f128c04ac305355298e1a96cb78d0d7cd6
processed_at: '2026-08-18T02:53:44-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲讲这篇论文

## 一句话版本

Robots 缺数据嘛，想拿 internet 上 mass scale 的 human video 来训，但人手和 robot gripper 长得不一样、动得不一样，直接用不行。这篇 paper 的 trick 就是: 把一段 video 拆成两个独立的 latent code——一个专门 encode "要干什么活"(task)，另一个专门 encode "谁在干"(embodiment)——然后 inference 时把人的 task code 和 robot 的 embodiment code 拼起来，让 frozen video diffusion model 给你 render 出一段 "robot 在干这个活" 的 video。

---

## 痛点在哪

Robot learning 的核心 bottleneck 就是 data。你想 train 一个 generalist robot policy，需要海量 demonstration。但采集 robot demonstration 贵得离谱——每个 teleoperation episode 都要人操控 robot、记录 proprioception、标注 end-effector pose，成本高、diversity 低。相比之下，YouTube 上、Ego4D 里、EgoDex 里有 thousands of hours 的人类第一视角 manipulation video，什么 task 都有，pouring、grasping、stacking、cutting... free 的。

问题是，你不能直接拿人手的 video 去训 robot。原因不止是"长得不一样"这么简单。更本质的是: 人手是 5 个 finger 的 articulated structure，21-DOF；robot gripper 可能是 2-finger parallel jaw，7-DOF；或者 suction cup，1-DOF。它们的 **kinematics** 完全不同，**affordance** 完全不同。你让人去演示 "pick up a cup"，人手会 wrap fingers around 杯子；robot parallel jaw 是侧面夹住。这俩 motion pattern 在视觉上、在 SE(3) trajectory 上都差了十万八千里。

这就是所谓的 **embodiment gap**。

---

## 之前的方法怎么搞的，为什么不 work

### 路线 A: Unified representation

像 EgoMimic ([Kareer et al.](https://arxiv.org/abs/2410.24221))、HAT ([Qiu et al.](https://arxiv.org/abs/2503.13441)) 这种，做法是设计一个 shared action/observation space，让 human data 和 robot data 都 map 到同一个空间里，然后在这个 unified space 里训 policy。

听着挺 make sense，但实际上的 failure mode 是: 这个 shared representation 会 **entangled**。什么意思呢？比如你训一个 VAE 或者 contrastive encoder 来 encode "pouring" task 的 video frame，模型学到的 feature 里既包含 "倾斜容器让液体流出" 这个 task-semantic 信息，也包含 "5 个手指握住容器把手" 这个 human-specific kinematic 信息。这两者耦合在一起，你想 transfer 到 robot 上时，robot policy 会试图模仿那个 5-finger grip pattern，但它根本没有 5 个 finger，就崩了。

### 路线 B: Procedural video editing

像 Phantom ([Lepert et al.](https://arxiv.org/abs/2503.00779))、Masquerade ([Lepert et al.](https://arxiv.org/abs/2508.09976)) 这种，做法是两步走: 先 inpainting 把人手从 video 里擦掉，然后从 simulator 里 render 一个 robot model，animate 上 human motion，overlay 到 video 上。

问题更直观: simulator render 出来的 robot 看起来就是 fake 的。Lighting 对不上、shadow 没有、occlusion 处理不了、物理 interaction 假。最后得到的 "robot video" 既不 photorealistic 也不 temporally coherent，downstream policy 学不到有用的东西。

### 还有路线 C: 直接拿 general-purpose video diffusion model 改

像 VACE ([Jiang et al.](https://arxiv.org/abs/2503.07598)) 这种 video editing model，给它 masked human video + target robot image + text prompt，让它 inpainting 一个 robot 进去。Paper 里试了这个 baseline，结果 VACE 会 **hallucinate** 出一个 humanoid hand！因为它 pre-training 时见过的 "grasping" task 全是人手 grasping，它根本不知道 robot gripper 长啥样、怎么动。

---

## 这篇 paper 的核心 insight

**关键观察**: 一段 demonstration video，不管是谁在执行，都可以 conceptually 分解成两部分信息的乘积:

$$\text{video} \approx f(\text{what to do}, \text{who is doing it})$$

- **"What to do"** = task: 目标是什么、object 怎么 dynamics、interaction sequence 是什么、接触点在哪。这部分应该是 embodiment-agnostic 的——"把杯子从桌上拿起来放到架子上" 这个 task 的语义，不管人手还是 robot gripper 执行，描述应该一样。
- **"Who is doing it"** = embodiment: agent 的 morphology、kinematics、DOF 结构。这部分应该是 task-agnostic 的——一个 robot gripper 的"身份"不因为它在 grasping 还是 pouring 而改变。

如果你能学会把这两个 factor 显式 disentangle 成两个独立的 latent code $z_{task}$ 和 $z_{emb}$，那么 transfer 就 trivial 了: 从 human video 提取 $z_{task}$，从一张 robot 图片提取 $z_{emb}^{robot}$，把它们 concatenate 喂给 generative model，就能 render 出 robot 在执行这个 task 的 video。

**而且 crucially，你 training 时根本不需要 paired human-robot data**（这种数据基本不可能大规模采集）。你只需要 human video 自身做 self-reconstruction，然后用 contrastive objective 强制 disentangle 就行。

---

## 怎么实现 disentanglement 的——直觉版

### Step 1: 定义 task 和 embodiment 各自 encode 什么

**Task representation** $z_{task}$ 是个 multimodal encoding，包含:
- $T$: text description ("grasp the bottle and pour into bowl")
- $M_s$: hand motion trajectory——每帧的 3D position、6D rotation、grip state
- $O_s$: object trajectory——被 manipulation 的 object 的 3D position + 6D rotation 序列

为什么用这种 structural signal 而不直接 encode video frame 呢？直觉上，video frame 里 task 和 embodiment 是 entangled 的——你看到一帧人手握杯子，分不清哪些 pixel 是 "grasping" 的 task 信息，哪些是 "5-finger articulated hand" 的 embodiment 信息。但 hand motion trajectory $M_s$ 已经把"手在空间中怎么动"这个 task-relevant 的部分提取出来了，embodiment 信息被削弱。Object trajectory $O_s$ 更纯——杯子怎么动完全由 task 决定，跟谁在动它无关。

**Embodiment representation** $z_{emb}$ 就是单张 static end-effector image 过 CLIP + shallow Transformer。为什么单张图够？因为 morphology 和 appearance 基本上可以从一张图 infer 出来——你看到一张 robot gripper 的照片，就知道它有几个 finger、长什么样、大概能怎么动。

### Step 2: 强制让 $z_{task}$ 和 $z_{emb}$ 独立

这是最核心的 trick。光靠 architecture 设计（分开 encoder）不能保证 disentanglement，因为 model 可能 find shortcut 让两个 code 互相 leak 信息。

Paper 用了 **CLUB estimator** ([Cheng et al. 2020](https://arxiv.org/abs/2006.12013)) 来 minimize 它们的 mutual information $I(z_{task}; z_{emb})$。

直觉上讲，CLUB 的 idea 是: 训一个 auxiliary neural net $q_\phi(z_{emb} | z_{task})$，试图从 $z_{task}$ 预测 $z_{emb}$。如果 $z_{task}$ 和 $z_{emb}$ 真的 independent，那 $z_{task}$ 里没有任何关于 $z_{emb}$ 的信息，这个 predictor 学不到东西，预测准度等于瞎猜。

CLUB 的 loss 形式:

$$\mathcal{L}_{disentangle} = \mathbb{E}_{p(z_{task}, z_{emb})}[\log q_\phi(z_{emb}|z_{task})] - \mathbb{E}_{p(z_{task})p(z_{emb})}[\log q_\phi(z_{emb}|z_{task})]$$

- 第一项: 在 **真实配对** $(z_{task}, z_{emb})$ 上的 log-likelihood。如果两个 code 有 dependency，这项会高（predictor 能预测准）。
- 第二项: 在 **随机打乱配对** $(z_{task}, z_{emb}^{shuffled})$ 上的 log-likelihood。这是 baseline，predictor 在乱配对上肯定预测不准。

Minimize 这个 loss = 让真实配对的 predictability 逼近随机配对的 predictability = 让 predictor 无法区分真实配对和随机配对 = $z_{task}$ 对 $z_{emb}$ 没有信息。

**Training trick**: 这个 alternating optimization——先 update $q_\phi$ 让 upper bound tight，再 update encoder minimize MI。而且 gradient 只 every 10 step apply 一次，不然 $q_\phi$ 追不上。这个节奏 control 很像 GAN training 里 discriminator 和 generator 的 balance。

### Step 3: 防止 trivial collapse

光 minimize MI 有个 degenerate solution: 让 $z_{task}$ 或 $z_{emb}$ collapse 成常数，那 MI trivially = 0，但 representation 完全没用。

所以 paper 加了 **InfoNCE** ([van den Oord et al. 2018](https://arxiv.org/abs/1807.03748)) 来 maximize intra-space consistency:

$$\mathcal{L}_{contrast} = -\mathbb{E}\left[\log \frac{\exp(\text{sim}(z_i, z_i^+))}{\exp(\text{sim}(z_i, z_i^+)) + \sum_k \exp(\text{sim}(z_i, z_k^-))}\right]$$

- $z_i$: anchor
- $z_i^+$: positive sample，同类（同一 task 或同一 embodiment）
- $z_k^-$: negative samples，异类

这个 loss 把同类 embedding 拉近、异类推远，强制每个 latent space 内部形成 compact cluster。Paper 用两路 InfoNCE:
- $\mathcal{L}_{task\_contrast}$: 同 task 不同 video 的 $z_{task}$ 拉近
- $\mathcal{L}_{emb\_contrast}$: 同 agent 不同 image 的 $z_{emb}$ 拉近

这就形成了一个 **dual structure**:
- Cross-space (CLUB): 推开 task 和 embodiment
- Intra-space (InfoNCE): 在各自空间内聚类

Figure 4 的 t-SNE 很直观地展示了效果: 不同 task 和不同 embodiment 都形成了 distinct cluster，而且 task space 和 embodiment space 之间的 correlation matrix off-diagonal block 接近 0。

---

## 整体 pipeline 的直觉

想象你在用 ControlNet ([Zhang et al.](https://arxiv.org/abs/2302.05543)) 给 Stable Diffusion 加 spatial condition——这里思路类似，但 condition 是 disentangled 的两个 code。

**Architecture**:
- **Frozen backbone**: Wan2.1-VACE-1.3B，一个 latent DiT video diffusion model。这个 model 已经会 generate photorealistic temporally coherent video，paper 不重训它，省 parameter。
- **Trainable Task Encoder**: shallow Transformer，把 (text, hand motion, object trajectory) encode 成 $z_{task}$。用 [CLS] token 输出 fixed-length embedding。
- **Trainable Embodiment Encoder**: frozen CLIP image encoder + shallow Transformer，把 end-effector image encode 成 $z_{emb}$。
- **Trainable Adapter**: 15 个 Transformer block，mirror frozen backbone 的 structure。它 process $(z_{task}, z_{emb})$，然后把 output feature **element-wise add** 到 frozen backbone 对应 block 的 feature 上。

这个 additive injection 让 condition signal 在 multiple feature level 影响 generation，既 parameter-efficient（不动 frozen weight）又 expressive（多层 condition）。

**Training**:
- 输入: human video + 对应的 task signal + end-effector image
- 自 reconstruction: 从 noise 出发，conditioned on $(z_{task}, z_{emb}^{human})$，重建原 human video
- Loss = Flow Matching (重建) + CLUB (disentangle) + 2 路 InfoNCE (聚类)

**Inference** (zero-shot transfer):
- 给一段 new human video → 提取 $z_{task}$
- 给一张 target robot 图片 → 提取 $z_{emb}^{robot}$
- 把两者拼起来 → frozen VACE + adapter → render robot video

由于 training 时这两个 code 是 disentangled 的，inference 时 swap 是合法的——$z_{task}$ 不含 human embodiment 信息，$z_{emb}^{robot}$ 不含 task 信息，组合起来 model 见过的训练分布。

---

## 为什么用 Flow Matching 而不是 DDPM

Paper 用 **Rectified Flow** ([Liu et al. 2022](https://arxiv.org/abs/2209.03003)) 做 generative modeling。直觉上:

DDPM 定义一个 Markov chain $x_0 \to x_1 \to \dots \to x_T$，每步加一点 noise，然后学 reverse process。Noise schedule 是个 hyperparameter，调起来烦，sampling 也慢。

Flow Matching 更直接: 定义一条从 noise $x_0$ 到 data $x_1$ 的 path，学 path 上每点的 velocity。Rectified Flow 的 trick 是用 linear path:

$$x_t = t \cdot x_1 + (1-t) \cdot x_0$$

这条 path 是直线，所以 velocity $v_t = x_1 - x_0$ 是 **constant**，跟 $t$ 无关。这让 velocity field 学起来 super easy，sampling 也 efficient。

Loss:
$$\mathcal{L}_{FM} = \mathbb{E}\left[\|u_\theta(x_t, t, z_{task}, z_{emb}) - (x_1 - x_0)\|^2\right]$$

直觉: 让 model 预测的 velocity 逼近 ground truth constant velocity。

近期 video model 像 CogVideoX ([Yang et al.](https://arxiv.org/abs/2408.06072))、FLUX.1 ([Labs et al.](https://arxiv.org/abs/2506.15742))、VACE 都用 Flow Matching family，这已经是 video generation 的事实 standard。

---

## 实验数据的直觉解读

### Table 1: Quantitative comparison

| Method | FVD↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|---|---|---|---|---|
| VACE | 1575.8 | 0.676 | 12.43 | 0.515 |
| Phantom | 1948.3 | 0.676 | 10.96 | 0.435 |
| **Ours** | **1469.6** | **0.674** | **13.24** | **0.532** |

- **FVD** (Fréchet Video Distance): 衡量 generated video distribution 和 real distribution 的距离，越低越好。Ours 比 VACE 好 ~7%，比 Phantom 好 ~25%。
- **LPIPS**: perceptual distance，越低越像。
- **PSNR/SSIM**: pixel-level 重建质量。

VBench 那一堆 metric 里有个有趣的 trade-off:
- Static-centric metrics (Imaging Quality, Background Consistency, Subject Consistency): VACE 最好。因为它 inpainting prior 强，能保持 scene static part 不变。
- Dynamic/Temporal metrics (Aesthetic Quality, Temporal Style, Motion Smoothness, Overall Consistency): **Ours 最好**。

这个 trade-off 说明: VACE 擅长"静态背景不动"，但学不会"robot 怎么动得对"。Paper 的方法通过 explicit task embedding 把 motion 信息塞进去，所以 dynamic 质量高。Motion Smoothness 0.994 vs 0.993 看着差不多但 consistent，说明 disentangled task representation 让 generation 的 motion 更 coherent。

### Table 2: Ablation

去掉 Dual Contrastive objective 后 FVD 从 1514.9 退化到 1557.8，~3% degradation。看着不大，但 LPIPS 从 0.576 到 0.635，退化明显。Paper 的 Figure 8-13 展示了 t-SNE: 没有 DC，task 和 embodiment feature 会 overlap，cluster 不 compact。

直觉理解: Flow Matching 本身没有 inductive bias 让 latent disentangle。如果不加 contrastive regularization，model 会 find shortcut 把 task 和 embodiment 信息 entangled 在一起 encode（因为这样 reconstruction 更容易），inference 时就 swap 不了了。DC objective 是**强制 structuring latent space**，代价是 reconstruction 稍微难一点，但换来 transferability。

### Table 3: Downstream robotics validation

这个是 money experiment。用 MuJoCo simulation + Unitree H1 humanoid + ACT policy ([Zhao et al.](https://arxiv.org/abs/2304.13705)) 做 grasping task。

| Method | Total Val Loss↓ | Action L1↓ | EEF Pos Loss↓ |
|---|---|---|---|
| HAT | 0.2006 | 0.019 | 0.091 |
| **Ours** | **0.1886** | **0.018** | **0.085** |

流程: paper 先把 human demonstration video 编辑成 robot video，然后用这些 generated robot video 训 policy。对比 baseline 是 HAT——直接在 human+robot unified space 训 policy 的 SOTA。

EEF position loss 从 0.091 降到 0.085，~7% 提升。在 robotics 里，这种 EEF precision 提升是 meaningful 的——意味着 policy 学到的 end-effector trajectory 更准。

直觉: HAT 的 unified representation 不可避免带着 human kinematic bias，policy 学的时候会被"人手怎么动"干扰。Paper 的方法先把 embodiment gap 在 **video generation 阶段** 就 bridge 掉了，policy 看到的是干净的 robot-domain video，学起来就容易多了。

---

## 一些吐槽和 open questions

### 数据 pipeline 的脆弱性

Paper Section 3.5 描述了 object trajectory extraction pipeline: Grounding DINO ([Liu et al.](https://arxiv.org/abs/2303.05499)) detection → SAM2 ([Ravi et al.](https://arxiv.org/abs/2408.00714)) tracking → Video Depth Anything ([Chen et al.](https://arxiv.org/abs/2505.20245)) depth → ICP pose estimation。

这个 pipeline 在 real-world cluttered scene 里会 fragile 得很:
- Detection 失败 → trajectory 整个 missing
- Depth noise → ICP 在 textureless 区域 drift
- 6D rotation 在 symmetric object 上 ambiguous (一个瓶子绕主轴转，外观不变)

Paper 没讨论这些 noise 对 final task representation 的影响。我直觉上觉得这会限制 framework 在 long-tail object 上的 generalization。

### Single image for embodiment 的 limitation

用一张 static image encode $z_{emb}$ 对于 parallel jaw gripper 够用，但对于 dexterous hand（比如 Shadow Hand 24-DOF）可能 information insufficient。一张图很难 capture 所有 joint 的 articulation range。

而且如果是 bimanual robot 或者 mobile manipulator（base + arm），single image 视角受限，怎么 capture 全身 morphology？这可能是为什么 paper 实验只用了相对简单的 gripper。

### Long-horizon task 的 scalability

Paper 实验是 5 秒 81 帧 single manipulation primitive。对于 "make a sandwich" 这种 long-horizon task，需要 hierarchical decomposition——可能要类似 MimicPlay ([Wang et al.](https://arxiv.org/abs/2302.12422)) 那种 high-level plan + low-level controller 的结构。Paper 没探索这个。

### Task representation 的 multimodal dependency

Inference 时，给一个 new human video，要先跑整套 detection + tracking + depth + ICP pipeline 才能 extract $M_s$ 和 $O_s$。这是显著的 inference overhead，部署时不太 practical。能不能 weakly supervised 让 model 直接从 video frame infer 这些 structural signal？这是一个 obvious 的 future work。

### Generalization across gap size

Human hand → parallel jaw gripper 是 small gap，human hand → bimanual robot 是 large gap，human hand → mobile manipulator 是 very large gap。Paper 没系统 explore gap size 对 framework performance 的影响。直觉上 gap 越大，disentanglement 越难——因为 task representation 需要更 abstract 才能 accommodate 不同 embodiment。

---

## Final intuition

如果让我给一个 mental model 来理解这篇 paper:

**想象一个 video 是一道菜的 recipe**。Recipe 包含两部分信息: 食材和做法（task），以及厨具（embodiment）。一个 "用炒锅炒青椒肉丝" 的 recipe，换成平底锅也能做——task 是 "炒青椒肉丝"，embodiment 是 "用什么锅"。

Paper 做的就是: 训一个 encoder 从 recipe 里提取出 "task" 和 "embodiment" 两个独立 code，然后训一个 generative model 接收这两个 code 渲染出 "用平底锅炒青椒肉丝" 的成品 video。

关键 trick 是用 dual contrastive objective 强制这两个 code 真的独立——不然 model 会偷偷把 "炒锅" 的信息 encode 进 task code 里（因为这样 reconstruction 更容易），transfer 时就废了。

CLUB 那个 alternating optimization，直觉上像是: 训一个 "间谍" $q_\phi$ 试图从 task code 偷出 embodiment 信息，再训 encoder 让 "间谍" 偷不到。两边对抗，最后 task code 就真的 embodiment-agnostic 了。

这种 "reframe robotics problem as disentanglement problem + borrow contrastive learning technique" 的 pattern，在 ML research 里非常 productive。类似的 pattern 见过: RT-2 ([Brohan et al.](https://arxiv.org/abs/2307.15818)) 把 robotics policy reframe 成 VLM token prediction；Diffusion Policy ([Chi et al.](https://arxiv.org/abs/2303.04137)) 把 action generation reframe 成 diffusion。

Robotics community 的启示: **generative model 作为 data augmentation tool 比 policy backbone 更 practical**。你不需要 robot action label，只需要 video-level consistency，这正好是 video diffusion 的强项。Paper 走的是这条路，我觉得方向很对。

---

## Web references

- Paper: [Bridging the Embodiment Gap](https://arxiv.org/abs/2503.07598) (VACE)
- [CLUB: Contrastive Log-ratio Upper Bound](https://arxiv.org/abs/2006.12013)
- [CPC / InfoNCE](https://arxiv.org/abs/1807.03748)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
- [HAT: Humanoid Policy ≈ Human Policy](https://arxiv.org/abs/2503.13441)
- [EgoDex](https://arxiv.org/abs/2505.11709)
- [Phantom](https://arxiv.org/abs/2503.00779)
- [Masquerade](https://arxiv.org/abs/2508.09976)
- [EgoMimic](https://arxiv.org/abs/2410.24221)
- [ACT: Action Chunking Transformer](https://arxiv.org/abs/2304.13705)
- [ControlNet](https://arxiv.org/abs/2302.05543)
- [IP-Adapter](https://arxiv.org/abs/2308.06768)
- [Grounding DINO](https://arxiv.org/abs/2303.05499)
- [SAM 2](https://arxiv.org/abs/2408.00714)
- [Video Depth Anything](https://arxiv.org/abs/2505.20245)
- [MimicPlay](https://arxiv.org/abs/2302.12422)
- [LAPA: Latent Action Pretraining](https://arxiv.org/abs/2410.11758)
- [Ego4D](https://arxiv.org/abs/2110.07058)
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)
- [MINE](https://arxiv.org/abs/1801.04062)
- [β-VAE](https://arxiv.org/abs/1606.07336)
- [CogVideoX](https://arxiv.org/abs/2408.06072)
- [FLUX.1 Kontext](https://arxiv.org/abs/2506.15742)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [RT-2](https://arxiv.org/abs/2307.15818)

---

# Bridging the Embodiment Gap: Disentangled Cross-Embodiment Video Editing - 深度解析

## 1. 核心Intuition: 为什么需要Disentanglement?

这篇paper的核心洞见可以从一个观察出发: **internet-scale human video**是一个巨大的潜在robotics数据源, 但存在embodiment gap。这个gap不仅是visual appearance差异, 更本质是 **kinematics** 和 **morphology**差异。

传统方法的失败模式很有启发性:
- **EgoMimic** ([Kareer et al.](https://arxiv.org/abs/2410.24221)), **HAT** ([Qiu et al.](https://arxiv.org/abs/2503.13441)) 这类unified representation方法, 试图用一个shared latent space align human和robot。问题在于task-relevant信息(比如"把杯子拿起来"这个goal + 接触点的物理interaction)会和human-specific kinematic bias(比如人手5个手指的articulated motion)耦合在一起, 导致transferred representation带着human的影子。
- **Phantom** ([Lepert et al.](https://arxiv.org/abs/2503.00779)), **Masquerade** ([Lepert et al.](https://arxiv.org/abs/2508.09976)) 这类procedural方法做"inpaiting human + overlay robot"，问题是overlay的rendered robot model本身不photorealistic, lighting/shadow/occlusion都没法处理。

所以paper的核心假设是: 一个demonstration video可以分解成两个**orthogonal** latent空间:
- $z_{task}$: embodiment-invariant, 编码goal + object dynamics + interaction sequence
- $z_{emb}$: embodiment-specific, 编码morphology + kinematics

这是一个典型的**disentanglement**问题, 但和传统disentanglement (比如 $\beta$-VAE, FactorVAE) 不同的是, 这里直接定义了"什么应该被disentangle"的语义结构, 而不是让模型自己找factor。

---

## 2. 为什么这个特定的Disentanglement设计work? — 从信息论角度看

### 2.1 The Embodiment Gap的本质: 一个Causal Model视角

如果我们把一个demonstration video看作是由以下latent factor生成的:

$$V = f(\text{task}, \text{embodiment}, \text{scene context})$$

那么embodiment gap本质上是: training data里 $(task, embodiment)$ 是confounded的——人类视频里看到"pouring"任务时, embodiment永远是human hand。一个generative model如果直接学$p(V)$, 它无法区分"哪些visual feature来自task, 哪些来自embodiment"。

这就是为什么 **VACE baseline直接做inpainting会失败**——它在pre-training data里看到的grasping都是人手grasping, 所以inference时它会"hallucinate"出humanoid hand (paper里Figure 3的failure case)。

### 2.2 Mutual Information Minimization: 为什么用CLUB?

paper用CLUB ([Cheng et al. 2020](https://arxiv.org/abs/2006.12013))来minimize $I(z_{task}; z_{emb})$。这是关键的技术选择, 值得深挖。

直接minimize MI的难点: $I(X;Y) = \mathbb{E}_{p(x,y)}[\log \frac{p(x|y)}{p(x)}]$, 需要知道$p(x|y)$和$p(x)$, 但我们没有analytical form。

主流MI estimators对比:
- **MINE** (Mutual Information Neural Estimation, [Belghazi et al.](https://arxiv.org/abs/1801.04062)): 用Donsker-Varadhan representation, 给出**lower bound**。适合maximize MI (InfoMax), 但minimize MI时lower bound没用。
- **NWJ** estimator: 也是lower bound。
- **CLUB**: Contrastive Log-ratio Upper Bound, 给出**upper bound**。这正是minimize MI所需要的。

CLUB的核心公式:
$$I(X;Y) \leq \tilde{I}_{CLUB}(X;Y) = \mathbb{E}_{p(x,y)}[\log q_\phi(y|x)] - \mathbb{E}_{p(x)p(y)}[\log q_\phi(y|x)]$$

这里 $q_\phi(y|x)$ 是一个**variational approximation**到 $p(y|x)$, paper里用MLP实现 (Appendix A: 3个linear layer + GELU, 输出mean和log-variance, 用Tanh约束log-variance)。

paper中的公式(1):
$$\mathcal{L}_{disentangle} = \mathbb{E}_{p(z_{task}, z_{emb})}[\log q_\phi(z_{emb}|z_{task})] - \mathbb{E}_{p(z_{task})p(z_{emb})}[\log q_\phi(z_{emb}|z_{task})]$$

变量解释:
- $z_{task}$: task embedding, 从Task Encoder $E_{\theta_{task}}$ 输出
- $z_{emb}$: embodiment embedding, 从Embodiment Encoder $E_{\theta_{emb}}$ 输出
- $q_\phi(z_{emb}|z_{task})$: variational model, 试图predict $z_{emb}$ from $z_{task}$
- 第一项 $\mathbb{E}_{p(z_{task}, z_{emb})}$: 在**真实配对**的joint distribution上取期望
- 第二项 $\mathbb{E}_{p(z_{task})p(z_{emb})}$: 在**随机配对**(marginal product)上取期望

直觉理解: 如果 $z_{task}$ 和 $z_{emb}$ 真的independent, 那么 $p(z_{emb}|z_{task}) = p(z_{emb})$, 此时 $q_\phi(z_{emb}|z_{task})$ 不应该depend on $z_{task}$, 于是两个期望相等, loss=0。minimize这个loss就是让model无法从 $z_{task}$ 推出 $z_{emb}$。

### 2.3 Training Trick: Alternating Optimization

Appendix A的细节很关键: CLUB需要**alternating optimization**:
1. 先更新variational model $q_\phi$ 来maximize log-likelihood of true pairs, 让upper bound tight
2. 再更新encoder parameters来minimize MI

paper在Implementation Details里说: **gradients from $\mathcal{L}_{disentangle}$ applied to main model only once every ten training steps**。这个trick很常见——类似GAN training, 需要保持critic和generator的balance。如果every step都update, $q_\phi$可能没机会追上, upper bound不紧, MI minimization就fail。

---

## 3. Intra-space Consistency: InfoNCE的作用

CLUB只保证 $z_{task}$ 和 $z_{emb}$ 独立, 但**没有保证每个空间内部结构良好**。极端case: 如果 $z_{task}$ 全部collapse成常数, 那 $I(z_{task}; z_{emb}) = 0$ trivially satisfied, 但完全没用。

这就是InfoNCE ([van den Oord et al. 2018](https://arxiv.org/abs/1807.03748))的作用:

$$\mathcal{L}_{contrast} = -\mathbb{E}\left[\log \frac{\exp(sim(z_i, z_i^+))}{\exp(sim(z_i, z_i^+)) + \sum_k \exp(sim(z_i, z_k^-))}\right]$$

变量解释:
- $z_i$: anchor embedding
- $z_i^+$: positive sample, same semantic class (比如same task或者same embodiment)
- $z_k^-$: negative samples, different classes
- $sim(\cdot, \cdot)$: cosine similarity

paper用两路InfoNCE:
- $\mathcal{L}_{task\_contrast}$: positive pairs是**不同video但同一task**的 $z_{task}$, 负样本是不同task。这让 $z_{task}$ 空间按task semantic聚类。
- $\mathcal{L}_{emb\_contrast}$: positive pairs是**同一agent的不同image**的 $z_{emb}$, 负样本是不同agent。

这种设计形成了一个非常漂亮的**dual structure**:
- **Cross-space (CLUB)**: 推开两个空间
- **Intra-space (InfoNCE)**: 把同类拉到一起, 异类推开, 形成compact cluster

Figure 4的t-SNE visualization和correlation heatmap很好地展示了这一点: diagonal block红色(高intra-class correlation), off-diagonal block蓝色(接近0 cross-correlation)。

---

## 4. Architecture: 为什么用Frozen Backbone + Adapter?

### 4.1 VACE作为generative prior

paper用 **Wan2.1-VACE-1.3B** ([Jiang et al. 2025](https://arxiv.org/abs/2503.07598))作为frozen backbone。VACE是一个latent Diffusion Transformer (DiT), 配合3D VAE把video压到latent space, 然后用Video Condition Unit (VCU)处理multimodal信号。

frozen的好处:
1. 保留VACE的video generation prior (scene coherence, photorealism, temporal consistency)
2. Parameter-efficient training (只train encoders + adapter)
3. 避免catastrophic forgetting

### 4.2 Context Adapter Tuning

adapter设计的关键: **mirror frozen backbone的structure**, 然后element-wise add feature到对应block:

$$\text{DiT block output} = \text{Frozen block output} + \text{Adapter block output}(z_{task}, z_{emb})$$

15个Transformer block, initialized with frozen VACE的对应layer (加速convergence)。这种additive injection让conditioning信号在multiple feature level影响generation, 类似ControlNet ([Zhang et al.](https://arxiv.org/abs/2302.05543))和IP-Adapter ([Ye et al.](https://arxiv.org/abs/2308.06768))的思路, 但设计更轻量。

### 4.3 Task Encoder的多模态设计

task representation $z_{task} = E_{\theta_{task}}(T, M_s, O_s)$ 包含三路:
- $T$: text description (用frozen VACE text encoder + trainable Transformer)
- $M_s$: hand motion, per-frame 3D position + 6D rotation + grip state
- $O_s$: object trajectory, 3D position + 6D rotation for all manipulated objects

为什么用multimodal而不是直接encode video frames? 我的理解是:
- Video frames里task和embodiment是**entangled**的, 直接encode很难disentangle
- 而 $M_s$ 和 $O_s$ 是 **structural, lower-level** signal, 更接近"task的本质"——是SE(3)轨迹和interaction pattern
- text $T$ 提供high-level semantic

这是一个**inductive bias很强的设计选择**——通过explicit modality factorization来辅助latent disentanglement。代价是dependency on motion和object pose extraction的quality。

[CLS] token设计: 每个encoder prepend一个learnable token, 通过self-attention聚合global info, 输出fixed-length embedding。这是BERT ([Devlin et al. 2019](https://arxiv.org/abs/1810.04805))的经典trick, 在classification和representation learning里很常见。

### 4.4 Embodiment Encoder: 单张static image

$z_{emb} = E_{\theta_{emb}}(C_s)$, $C_s$ 是static end-effector image。用frozen CLIP image encoder提取patch-level feature, 然后shallow Transformer。

这里一个直觉问题: **single static image足够capture embodiment吗?** paper的假设是morphology和kinematics可以从appearance inference出来。这有一定risk——比如同一个robot从不同角度的image, embedding应该consistent (这正是 $\mathcal{L}_{emb\_contrast}$ 要保证的)。但对于截然不同kinematic的embodiment (e.g. parallel jaw gripper vs dexterous hand vs suction cup), single image确实能encode差异。

---

## 5. Training Objective: Flow Matching with Rectified Flows

paper用 **Rectified Flow** ([Liu et al. 2022](https://arxiv.org/abs/2209.03003))的Flow Matching作为主reconstruction loss, 而非DDPM。这反映了近期video diffusion community的趋势。

### 5.1 Rectified Flow的formulation

定义linear interpolation path:
$$x_t = t \cdot x_1 + (1-t) \cdot x_0$$

变量解释:
- $x_0 \sim \mathcal{N}(0, I)$: 噪声样本, t=0时pure noise
- $x_1$: target video latent (经过3D VAE encoding), t=1时clean
- $t \in [0, 1]$: time step
- $v_t = x_1 - x_0$: ground truth velocity, **constant** along the path

为什么叫"Rectified"? Rectified Flow的核心insight是: 在standard flow/diffusion里, $x_0$和$x_1$之间的path可能curved, 导致velocity $v_t$ depend on $t$, 学起来困难。Rectified Flow通过iterative "rectification"把path拉直, 让 $v_t$ 变成constant, 极大简化了velocity prediction。

Loss:
$$\mathcal{L}_{FM} = \mathbb{E}_{x_0, x_1, t, z_{task}, z_{emb}}\left[\|u_\theta(x_t, t, z_{task}, z_{emb}) - v_t\|^2\right]$$

变量解释:
- $u_\theta$: velocity predictor, 实际上是frozen VACE backbone + trainable adapter
- $v_t = x_1 - x_0$: ground truth constant velocity
- $z_{task}, z_{emb}$: condition, 通过adapter注入

### 5.2 对比: 为什么不用DDPM?

直觉上, DDPM和Flow Matching在理论上可以互相convert, 但实践上:
- DDPM的noise schedule设计是个hyperparameter nightmare
- Flow Matching的linear path让velocity field更smooth, 采样效率更高
- Rectified Flow的straight-line property让training更stable

近期video model如CogVideoX ([Yang et al. 2024](https://arxiv.org/abs/2408.06072)), FLUX.1 ([Labs et al. 2025](https://arxiv.org/abs/2506.15742)), VACE都用了Flow Matching family。这是generative modeling的paradigm shift。

### 5.3 Inference时的composition

$$V_\tau = G(\epsilon, z_{task}, z_{emb}^\tau; A_\psi)$$

这里 $z_{task}$ 来自human video extraction, $z_{emb}^\tau$ 来自target robot image。这个composition是**zero-shot**的——training时只见过self-reconstruction, 但因为disentanglement, 可以**自由组合** task和embodiment。

这是整个framework最elegant的地方: 训练时不需要paired human-robot data (which is essentially impossible to collect at scale), 但因为disentanglement explicit地separate了两个factor, inference时可以swap。

---

## 6. 数据Pipeline的细节: 一个隐藏的engineering feat

paper Section 3.5描述了dataset preparation, 这部分看似boring但其实是paper能work的关键。

### 6.1 数据源

基于 **PH²D** (Physical Human-Humanoid Data, [Qiu et al.](https://arxiv.org/abs/2503.13441)), 是 **EgoDex** ([Hoque et al. 2025](https://arxiv.org/abs/2505.11709))的task-oriented subset。EgoDex本身是large-scale egocentric dexterous manipulation video, 包含high-fidelity hand motion。

但PH²D缺少object 6D pose annotation, 所以paper自己搭pipeline提取。

### 6.2 Object Trajectory Extraction Pipeline

四步流程:

**Step 1: 2D Object Detection & Tracking**
- [Grounding DINO](https://arxiv.org/abs/2303.05499) (Liu et al. 2024b): open-set detection, text prompt指定object
- [SAM 2](https://arxiv.org/abs/2408.00714) (Ravi et al. 2024): mask-based tracking, 给出每帧的segmentation

**Step 2: 2D-to-3D Lifting via Depth**
- [Video Depth Anything](https://arxiv.org/abs/2505.20245) (Chen et al. 2025): 给每帧dense depth map, 构造per-frame 3D point cloud of object

**Step 3: 6D Pose Trajectory via ICP**
- Iterative Closest Point ([Chetverikov et al. 2002](https://ieeexplore.ieee.org/document/1047783)): 在consecutive frame的point cloud之间estimate rigid body transformation
- 输出: per-frame SE(3) pose (3D position + 6D rotation)

**Step 4: Hand/Robot Masking**
- SAM2 segment hand
- morphological dilation kernel稍微expand mask, 确保整个limb被covered (for downstream training/inference)

### 6.3 这个pipeline的脆弱性

直觉上, 这个pipeline的error会propagate:
- Grounding DINO detection failure → 整个trajectory missing
- Video Depth Anything noise → ICP在textureless区域会drift
- 6D rotation估计在symmetric object上ambiguous

paper没有详细讨论这个, 但作为critical reader, 我会wonder: 这些extracted trajectory的noise会不会limit task representation的fidelity? 这可能是为什么paper在simulation experiment里用相对controlled的场景(Grasping Pepsi)。

---

## 7. 实验结果深度分析

### 7.1 Quantitative Comparison (Table 1)

| Method | FVD↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|---|---|---|---|---|
| VACE | 1575.8 | 0.676 | 12.43 | 0.515 |
| Phantom | 1948.3 | 0.676 | 10.96 | 0.435 |
| **Ours** | **1469.6** | 0.674 | **13.24** | **0.532** |

FVD (Fréchet Video Distance, [Unterthiner et al. 2018](https://arxiv.org/abs/1812.01717))是衡量generated video distribution和real video distribution距离的核心metric。Ours比VACE好~7%, 比Phantom好~25%。

VBench的8个metric分两类看:
- **Static-centric** (IQ, BC, SC): VACE最好。这是因为VACE的strong inpainting prior擅长保持scene integrity, 但代价是dynamic质量差
- **Dynamic/Temporal** (AQ, TS, MS, OC): **Ours最好**。这证实了disentanglement学到的task representation能让generation产生更coherent motion

这个**trade-off**很重要: VACE把scene static part处理得好但motion学不到, paper的方法通过explicit task embedding把motion信息"硬塞"进generation, 用MS (Motion Smoothness) 0.994 vs VACE 0.993这种细小但consistent的improvement体现。

### 7.2 Ablation Study (Table 2)

| Method | FVD↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|---|---|---|---|---|
| w/o DC | 1557.8 | 0.635 | 13.42 | 0.489 |
| w/ DC | 1514.9 | 0.576 | 13.53 | 0.498 |

去掉Dual Contrastive objective后FVD退化~3%, LPIPS退化明显。Paper还在Appendix Figure 8-13展示了t-SNE, 完整DC objective给出最compact的cluster。

这个ablation其实有点surprising——DC objective只是auxiliary regularization, 主loss是Flow Matching, 但去掉后影响这么大。我猜测是因为Flow Matching本身没有inductive bias让latent disentangle, 所以representation会"自然"collapse到entangled state, generation时无法swap。

### 7.3 Downstream Robotics Validation (Table 3)

| Method | Total Val Loss↓ | Action L1↓ | EEF Pos Loss↓ |
|---|---|---|---|
| HAT | 0.2006 | 0.019 | 0.091 |
| **Ours** | **0.1886** | **0.018** | **0.085** |

这个experiment最关键——它证明paper的方法不仅是"好看", 还能**actually useful for downstream policy learning**。用 **Unitree H1 humanoid robot** + **MuJoCo** simulation + **ACT policy** ([Zhao et al. 2023](https://arxiv.org/abs/2304.13705)) + ResNet backbone。

paper先生成robot video, 然后用这个video train policy, 比直接用HAT做human-robot alignment更好。EEF (End-Effector) position loss从0.091降到0.085, ~7%的提升, 这个数字在robotics里已经meaningful。

---

## 8. Limitations和Open Questions (paper没明确说, 但我的critical reading)

1. **Object trajectory extraction的noise**: paper依赖ground truth质量的SE(3) object pose, 在real-world cluttered scene里这个pipeline脆弱。是否可以改成weakly-supervised, 让model从video直接infer?

2. **Single static image for embodiment**: 适合standard parallel-jaw gripper, 但对articulated hand (dexterous, multi-finger)可能information insufficient。如果target是[Shadow Hand](https://www.shadow-robot.com/)这种24-DOF hand, 一张image很难capture所有joint。

3. **Task representation依赖multimodal input**: 实际部署时, 给一个新human video, 还需要运行Grounding DINO + SAM2 + Video Depth Anything + ICP这套pipeline才能extract $M_s$ 和 $O_s$。这是显著的inference-time overhead。

4. **Long-horizon task**: paper实验是5秒81帧, 单一manipulation primitive。对于long-horizon task (比如"make a sandwich"包含多个subtask), framework如何scale? 是否需要hierarchical decomposition (类似[MimicPlay](https://arxiv.org/abs/2302.12422))?

5. **Generalization across embodiment gap大小**: human hand → parallel jaw gripper (small gap), human hand → bimanual robot (large gap), human hand → mobile manipulator (very large gap)。Paper没有explore不同gap size的影响。

---

## 9. 相关工作的broad connection

### 9.1 Disentanglement Literature

这个paper的disentanglement思路和经典 $\beta$-VAE ([Higgins et al.](https://arxiv.org/abs/1606.07336)), FactorVAE ([Kim & Mnih](https://arxiv.org/abs/1802.05983))一脉相承, 但更"supervised"——通过contrastive loss指定哪些应该independent。

更接近的是**HSIC** (Hilbert-Schmidt Independence Criterion)和**CCB** (Covariance Constraint), 都是显式minimize dependency。CLUB的优势是neural network based, 可以处理high-dim, complex distribution。

### 9.2 Video Editing / Motion Customization

相关工作:
- [MotionDirector](https://arxiv.org/abs/2310.08365) (Zhao et al. 2024): motion-appearance disentanglement for text-to-video
- [DreamVideo](https://arxiv.org/abs/2308.13884) (Wei et al. 2024a): composing subject + motion
- [DragNUWA](https://arxiv.org/abs/2308.08089), [DragAnything](https://arxiv.org/abs/2402.01458): trajectory-based control
- [MimicMotion](https://arxiv.org/abs/2406.19680): pose-guided human motion video

这些工作的"motion"通常是visual trajectory or pose sequence, 而paper的"task"是更高层semantic + SE(3)轨迹的multimodal encoding。

### 9.3 Learning from Video in Robotics

- [EgoMimic](https://arxiv.org/abs/2410.24221), [HAT](https://arxiv.org/abs/2503.13441): unified representation for cross-embodiment
- [UniSkill](https://arxiv.org/abs/2505.08787): cross-embodiment skill representation
- [MimicPlay](https://arxiv.org/abs/2302.12422): high-level plan from human video + low-level controller
- [Video2Policy](https://arxiv.org/abs/2502.09886), [DreamGen](https://arxiv.org/abs/2505.17972): video-to-simulation pipeline
- [LAPA](https://arxiv.org/abs/2410.11758) (Latent Action Pretraining from Videos): 类似idea, 学latent action然后fine-tune

paper的positioning独特之处: 它**不直接learn policy**, 而是learn一个**video-to-video translation**作为intermediate data augmentation。生成robot video后还要单独train policy。这让它和policy method orthogonal, 可以用任何policy architecture。

### 9.4 Egocentric Datasets

- [Ego4D](https://arxiv.org/abs/2110.07058) (Grauman et al. 2022): 3000小时大规模egocentric video
- [GigaHands](https://arxiv.org/abs/2412.04244) (Fu et al. 2024): massive bimanual hand activity
- [ARCTIC](https://arxiv.org/abs/2204.13662) (Fan et al. 2023): 高精度3D hand-object manipulation
- [HOT3D](https://arxiv.org/abs/2411.19167) (Banerjee et al. 2024): egocentric multi-view hand/object tracking
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864): robotics dataset aggregation

这些数据集提供了大规模human video的基础, 但cross-embodiment alignment仍是open problem。

---

## 10. Final Intuition Summary

如果用一句话抓住paper的essence: **通过explicit dual contrastive objective (CLUB minimize MI + InfoNCE maximize intra-class consistency) 在frozen video diffusion model上learn两个orthogonal latent space, 实现 task 和 embodiment 的disentanglement, 从而允许zero-shot swap实现human-to-robot video editing**。

关键的conceptual贡献是: **把embodiment gap这个robotics-specific problem, 重新formulate成一个disentanglement problem, 然后用contrastive learning + frozen generative prior的组合拳解决**。这种"reframe problem + borrow technique"的pattern在ML research里非常productive。

engineering上的smart choices:
- self-reconstruction (避免paired data bottleneck)
- multimodal task encoding (用structural signal辅助disentanglement)
- frozen backbone + adapter (parameter-efficient + leverage strong prior)
- alternating optimization for CLUB (稳定的MI minimization)

对于robotics community, paper的启示是: **generative model作为data augmentation tool比作为policy backbone更practical**, 因为不需要robot action label, 只需要video-level consistency, 这正好是video diffusion model的强项。

Web References:
- Paper: [arxiv.org/abs/2503.07598](https://arxiv.org/abs/2503.07598) (VACE)
- [arxiv.org/abs/2006.12013](https://arxiv.org/abs/2006.12013) (CLUB)
- [arxiv.org/abs/1807.03748](https://arxiv.org/abs/1807.03748) (CPC/InfoNCE)
- [arxiv.org/abs/2209.03003](https://arxiv.org/abs/2209.03003) (Rectified Flow)
- [arxiv.org/abs/2503.13441](https://arxiv.org/abs/2503.13441) (HAT)
- [arxiv.org/abs/2505.11709](https://arxiv.org/abs/2505.11709) (EgoDex)
- [arxiv.org/abs/2503.00779](https://arxiv.org/abs/2503.00779) (Phantom)
- [arxiv.org/abs/2304.13705](https://arxiv.org/abs/2304.13705) (ACT)
- [arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) (BERT)
- [arxiv.org/abs/2302.05543](https://arxiv.org/abs/2302.05543) (ControlNet)
- [arxiv.org/abs/2308.06768](https://arxiv.org/abs/2308.06768) (IP-Adapter)
- [arxiv.org/abs/2303.05499](https://arxiv.org/abs/2303.05499) (Grounding DINO)
- [arxiv.org/abs/2408.00714](https://arxiv.org/abs/2408.00714) (SAM 2)
- [arxiv.org/abs/2302.12422](https://arxiv.org/abs/2302.12422) (MimicPlay)
- [arxiv.org/abs/2410.11758](https://arxiv.org/abs/2410.11758) (LAPA)
- [arxiv.org/abs/2110.07058](https://arxiv.org/abs/2110.07058) (Ego4D)
- [arxiv.org/abs/2310.08864](https://arxiv.org/abs/2310.08864) (Open X-Embodiment)
- [arxiv.org/abs/1801.04062](https://arxiv.org/abs/1801.04062) (MINE)
- [arxiv.org/abs/1606.07336](https://arxiv.org/abs/1606.07336) (β-VAE)
- [arxiv.org/abs/2408.06072](https://arxiv.org/abs/2408.06072) (CogVideoX)
- [arxiv.org/abs/2506.15742](https://arxiv.org/abs/2506.15742) (FLUX.1 Kontext)
