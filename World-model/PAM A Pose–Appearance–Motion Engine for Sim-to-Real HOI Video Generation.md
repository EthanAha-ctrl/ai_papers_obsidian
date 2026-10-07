---
source_pdf: PAM A Pose–Appearance–Motion Engine for Sim-to-Real HOI Video Generation.pdf
paper_sha256: 6ade76cb9700514461633e168faa0193619788e63f7e9f15b9fd4762022eb334
processed_at: '2026-08-06T01:55:33-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# PAM 用人话说

## 一句话说清楚

这篇 paper 想解决的问题是：**怎么用 AI 批量生成"手抓东西"的视频，而且生成的视频逼真到可以拿来训练其他 AI。**

为什么这件事重要？因为真实世界里收集"手抓东西"的视频太贵了——你要拍视频，还要标注每一帧里手的 21 个关节在哪、物体在哪个位置、手指怎么动的。这种标注成本巨大，所以数据量上不去。如果能用 AI 生成大量逼真的假视频，就能拿这些假数据去训练 hand pose estimator，省掉大量真数据的标注钱。

---

## 之前的 methods 都有什么毛病

作者把之前的工作分成三类，每类都有硬伤：

**第一类：只生成 pose 轨迹**
像 GraspXL 这种，它能告诉你"手在第 0 帧这样、第 10 帧那样"，给你一串 3D 坐标。但是没有画面，没有像素，没有 texture。你拿这个去训练 perception model，得先自己跑一遍 renderer，渲染出来的东西又很丑，不像 real world。

**第二类：只生成单张图**
像 HOIDiffusion 这种，给它一个 mask 或 keypoint，它能生成一张挺好看的手物交互图。但你把多张图串成视频，motion 是不连贯的——手会突然跳、物体会闪。

**第三类：能生成视频，但要求你给第一帧真实图**
像 InterDyn、ManiVideo 这种，你给它完整的 pose 序列 + 第一帧 ground truth 真实图片，它能生成不错的视频。问题是：**sim-to-real 场景下你拿不到第一帧真实图**。simulator 只能给你几何信息（depth、mask、mesh），给不了 photorealistic 的 RGB 图。所以这些方法在 sim-to-real pipeline 里直接卡死。

PAM 的核心 insight 就是：**真正能用的 sim-to-real pipeline，输入只能用 simulator 能给的东西**——初始 hand pose、目标 hand pose、object mesh。就这三样，别的都不给，也能跑通。

---

## PAM 怎么做的：三步走

作者把问题拆成三个独立的 stage，每个 stage 都站在一个已经训练好的 foundation model 肩膀上。这就像搭积木——不自己从零造轮子，而是把最好的轮子组装起来。

### 第一步：Pose Generation（让手先动起来）

输入：初始 hand pose + 目标 hand pose + object mesh
输出：中间每一帧的 hand pose 和 object pose

用的是 GraspXL，一个在 simulation 里用 RL 训练好的 grasping policy。它学会了"怎么从 A 姿态平滑过渡到 B 姿态去抓东西"，而且能 generalize 到不同 object shape。

这一步几乎不花计算资源（0.03 GB memory），但它是整个 pipeline 的几何基础。如果这一步生成的轨迹有物理错误（比如手穿进物体里），后面再好看也白搭——这是作者自己承认的 weakness。

### 第二步：Appearance Generation（生成第一帧好看的图）

这一步是 paper 的核心创新之一。

问题是：simulator 给你的只有 depth map、segmentation mask、object mesh——没有 RGB 图。你得用这些东西"脑补"出一张 photorealistic 的第一帧。

作者用 Flux（一个很强的 image diffusion model）+ ControlNet 来做这件事。但关键问题是：**给 Flux 什么 condition？**

作者的 reasoning 很有意思：
- 只给 depth → 手指头会糊在一起，分不清几个手指
- 只给 segmentation → 知道哪是手哪是物体，但不知道手指具体 pose
- 加上 hand keypoint → 显式告诉模型"这根手指在这、那根手指在那"

三个 condition 互补：depth 给几何，seg 给语义边界，keypoint 给手指细节。Ablation 证明三个一起用效果最好，任何一个单独用都不行。

技术细节：把三个 condition 都 VAE encode 成 latent，channel-wise concat，通过 ControlNet 注入 Flux 的前两层 DiT block。ControlNet 用 zero-conv 初始化——训练开始时 condition 信号是零，慢慢学起来，不会破坏 Flux 原有的生成能力。

还有一个 trick：生成时不是只生成一张第一帧，而是随机 sample 30 张，用 HaMeR（一个 hand pose estimator）检查每张的手 pose 准不准，扔掉最差的 25%。这本质上是 best-of-N with verifier，和 LLM 里的 rejection sampling 一个思路。

### 第三步：Motion Generation（把第一帧动画成视频）

有了第一帧好看图 + 完整 pose 轨迹，下一步就是生成视频。

做法：把 Stage I 的 pose 轨迹每一帧都渲染成 depth + seg + keypoint 三个 condition 序列，然后用 CogVideoX（一个 video diffusion model）+ ControlNet 生成视频。

和 Stage II 一样的 multi-condition 思路，但 inject 进 12 层 DiT（比 Stage II 的 2 层多很多），因为视频需要更强的 condition 信号来保证 temporal consistency——手不能跳、物体不能闪。

训练时有个 trick：每个 condition 以 0.2 概率随机 mask 掉。这样模型不会过度依赖任何一个 modality，当某个 condition 有噪声时（比如 depth 估计不准），模型能用其他 condition 补偿。Ablation 证明这个 masking 让模型在 noisy condition 下 FVD 只从 29.13 升到 30.45，如果不做 masking 会从 28.56 崩到 34.58。

---

## 结果怎么样

### Generation quality

在 DexYCB 数据集上：
- FVD（视频质量）：29.13，比 InterDyn 的 38.83 好不少
- MPJPE（hand pose 精度）：19.37 mm，比 CosHand 的 30.05 mm 好很多
- 分辨率：480×720，比 baseline 的 256×256 / 256×384 高一大截

在 OAKINK2 数据集上也是全面 SOTA。

### 最有说服力的结果：downstream task

作者用 PAM 生成了 3,400 个假视频（207k 帧），拿来增强 SimpleHand（一个 hand pose estimator）的训练数据。

关键发现：**50% 真数据 + PAM 假数据 = 100% 真数据的效果。**

换句话说，用 PAM 生成的假视频，能把真数据的需求量砍一半，性能还不掉。更夸张的是，100% 真数据 + PAM 假数据，比纯 100% 真数据还好一点点（PA-MPJPE 5.3 vs 5.5）。

这才是 paper 真正的价值——不是 FVD 数字多好看，而是假数据真的能省真数据的钱。对于 HOI 这种数据收集成本极高的领域，"真数据需求减半"是实打实的 contribution。

---

## 为什么这么设计，直觉是什么

**为什么要 decouple 成三步，不 end-to-end 训？**

因为 pose、appearance、motion 三个东西的联合分布太难学了。Pose 是 3D 几何，appearance 是 2D texture，motion 是 temporal dynamics——三个不同的 manifold。数据量不够 end-to-end 训出来。

Decouple 之后，每个 stage 都能用一个已经 well-pretrained 的 foundation model：
- Stage I 用 GraspXL（RL 训练的 motion prior）
- Stage II 用 Flux（image diffusion 的 appearance prior）
- Stage III 用 CogVideoX（video diffusion 的 temporal prior）

这是典型的 "stacking pretrained priors" 思路——不自己学一切，而是把最好的 prior 组装起来。

**为什么 multi-condition 比 single-condition 好？**

直觉是：手太复杂了，21 个关节、几十个 DoF。任何单一 signal 都 capture 不了全部信息：
- Depth 给几何形状但手指会糊
- Seg 给语义边界但不知道手指 pose
- Keypoint 给手指位置但不知道全局场景

三个一起用，互相补盲点。Ablation 里"depth + seg 但没有 keypoint"的 MPJPE 是 22.51，加上 keypoint 后变成 19.37——keypoint 对 hand pose 精度提升最直接。

**为什么 random masking 能提升 robustness？**

和 classifier-free guidance 同一个思路。训练时随机丢掉一些 condition，模型被迫学会"缺一个 modality 时用其他 modality 补偿"。推理时如果某个 condition 有噪声，模型不会直接崩掉，而是用其他 condition 兜底。这就像人如果闭上一只眼，另一只眼会补偿——但你得训练过"闭一只眼"才能做好。

---

## 这篇 paper 的局限

1. **Error propagation**：Stage I 的几何错误会传到最终视频。如果 GraspXL 生成的轨迹有手穿物体，最终视频也会手穿物体，再好看也不对。Decoupled pipeline 没有 closed-loop feedback，没法自动 fix 这种错误。

2. **第一帧还要 filter**：用 HaMeR 做 quality gate 说明 Flux + ControlNet 的 hand pose 准确率不够 100%。如果 HaMeR 本身有 bias，这个 filter 会 propagate bias。

3. **没有 end-to-end 训练**：三个 stage 分别训练，stage 之间没有 joint optimization。理论上 joint training 能让 stage 之间 align 得更好，但工程上很难做。

4. **Object 类型有限**：DexYCB 和 OAKINK2 都是 rigid object。如果 object 是 articulatable 的（剪刀、抽屉、订书机），这个 pipeline 还没验证过。

5. **Zero-shot 只有 qualitative**：在 OAKINK2（双手）上 zero-shot 测试只给了图，没给数字。

---

## 更大的 picture

PAM 代表的是一种 sim-to-real 新范式：不直接把 simulation 渲染成 real-looking（domain randomization 那套），也不在 sim 和 real 之间做 domain adaptation，而是把 sim 的 trajectory prior 和 real-world 的 video distribution prior 用 diffusion model 桥接起来。

Sim 给你物理上合理的 motion，diffusion model 给你视觉上逼真的 appearance，两者通过 multi-modal condition 对齐。这个思路比传统的 sim-to-real 更灵活——sim 不需要渲染得好看，只需要 motion 对；diffusion model 不需要懂物理，只需要 appearance 对。

和 Sora 那种 unified world model 相比，PAM 是另一个极端——modular、controllable、可调试，牺牲了 end-to-end 的优雅性换来工程上的可控性。在数据稀缺的 HOI 领域，这种 modular 设计可能更 practical。

最有意思的联想：PAM 的 Stage II rejection sampling（生成 30 个，用 HaMeR 选最好的）本质上是 best-of-N with verifier。这个思路在 LLM 的 RLHF 里已经很成熟——生成多个 response，用 reward model 排序选最好的。PAM 把这个思路用到了视觉生成上，用 hand pose estimator 当 reward model。未来可能会看到更多这种 "generation + verification" 的视觉 pipeline。

---

## References

- PAM project page: https://gasaiyu.github.io/PAM.github.io/
- Flux: https://github.com/black-forest-labs/flux
- ControlNet: https://arxiv.org/abs/2302.05543
- CogVideoX: https://arxiv.org/abs/2408.06072
- GraspXL: https://arxiv.org/abs/2409.18181
- HaMeR: https://arxiv.org/abs/2311.18237
- SimpleHand: https://arxiv.org/abs/2404.04492

---

# PAM: Pose–Appearance–Motion Engine for Sim-to-Real HOI Video Generation — 深度技术解析

Hi Andrej，这篇 paper 来自 PKU + THU + BAAI + SJTU + EIT 的合作，核心 idea 是把 Hand-Object Interaction (HOI) video generation 拆解成 **Pose → Appearance → Motion** 三个 decoupled stage，从而打破 prior work 必须依赖 ground-truth first frame 的 "sim-to-real bottleneck"。我会尽量详细展开技术细节、直觉，以及与相关工作的联想。

Project page: https://gasaiyu.github.io/PAM.github.io/
Flux repo: https://github.com/black-forest-labs/flux
ControlNet paper: https://arxiv.org/abs/2302.05543
CogVideoX: https://arxiv.org/abs/2408.06072
DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
GraspXL (ECCV 2024): https://arxiv.org/abs/2409.18181
DexYCB: https://arxiv.org/abs/2104.04631
OAKINK2: https://arxiv.org/abs/2403.12987
HaMeR: https://arxiv.org/abs/2311.18237
SimpleHand: https://arxiv.org/abs/2404.04492
CoTracker3: https://arxiv.org/abs/2410.11831

---

## 1. 问题动机：为什么 existing HOI generation 是 fragmented 的

作者把当前 HOI generation 的 three disjoint tracks 拆得很清楚（看 Figure 2）：

**(a) Pose-Only Synthesis (e.g., GraspXL [79])**
- 输出：MANO hand pose 轨迹 $\{\mathbf{h}_t, \mathbf{o}_t\}_{t=0}^{T}$
- 缺陷：no pixels — 只能给 downstream 提供 trajectory prior，不能直接用来训练 perception model（除非再跑一遍 renderer）

**(b) Appearance Generation (e.g., HOIDiffusion [83])**
- 输入：mask / 2D keypoint
- 输出：单张 HOI image
- 缺陷：no temporal dynamics — 把这些 image 串起来 motion 是 incoherent 的

**(c) Motion Generation (e.g., InterDyn [3], ManiVideo [48])**
- 输入：full pose sequence + ground-truth first frame
- 缺陷：sim-to-real 时 first frame 拿不到，simulator 只能给 geometry/mask，给不了 photorealistic appearance

PAM 的核心 motivation：**真正能用的 sim-to-real pipeline，必须只用 simulator 能提供的东西（initial pose、target pose、object mesh）就跑得起来**，并且能把 sim 的 trajectory prior 和 real-world appearance distribution 都接进来。

> Intuition：这本质上是把 "pose prior" (来自 RL simulator) 和 "appearance/motion prior" (来自 video diffusion foundation model) 当成两个 complementary 的 prior，用一个 bridge condition 把它们 align 起来。

---

## 2. 三阶段 Pipeline 总览

形式化地，整个 generative model 是一个 mapping：

$$
f_{\boldsymbol{\theta}} : (\mathbf{h}_0, \mathbf{m}, \mathbf{o}_0, \mathbf{h}_T) \rightarrow \{I_t\}_{t=0}^{T} \tag{1}
$$

变量说明：
- $\mathbf{h}_0 \in \mathbb{R}^{51 \times 3}$：initial MANO hand pose（21 个 joint × 3 维坐标，包含 global translation + rotation + joint angles 的参数化）
- $\mathbf{m}$：object mesh，没有 appearance（只有 geometry）
- $\mathbf{o}_0 \in \mathbb{R}^{6}$：initial 6-DoF object pose
- $\mathbf{h}_T \in \mathbb{R}^{51 \times 3}$：target hand pose
- $I_t$：第 $t$ 帧的 RGB image
- $T$：video length（实验中是 49 帧）

三个 constraint：
(i) 视频以 $\mathbf{h}_0$ 起始
(ii) 视频以 $\mathbf{h}_T$ 结束
(iii) 描述一个 temporally-coherent 的 grasp-to-place motion

**为什么 decouple？**
作者在 Section 3.1 引用了 [23] 的说法：jointly modeling pose + appearance + motion 是 high-dimensional spatiotemporal manifold，end-to-end 训不出来（数据量不够、loss 互相打架）。Decouple 之后：
- Pose stage 只学 motion prior（RL 已经做得很好）
- Appearance stage 只学单帧 photorealism（image diffusion 已经成熟）
- Motion stage 只学 temporal coherence + pose-conditioned animation（video diffusion 已经成熟）

每个 stage 都站在一个已经 well-pretrained 的 foundation 上，这是一个典型的 "stacking pretrained priors" 思路。

---

## 3. Stage I: Pose Generation (GraspXL)

这一 stage 在 Figure 3 的左侧，本质是 "调包"，但选 GraspXL [79] 是有讲究的。

### 输入输出
$$
(\mathbf{h}_0, \mathbf{o}_0, \mathbf{m}, \mathbf{h}_T) \xrightarrow{\text{GraspXL}} \{\mathbf{h}_t, \mathbf{o}_t\}_{t=0}^{T}
$$

GraspXL 是一个 RL-based 的 grasping policy，在 simulation 里学了一个 generalizable policy，可以生成 diverse 且 physically plausible 的 hand-object 轨迹，不需要 predefined reference motion。

### 为什么选 GraspXL 而不是 D-Grasp？
Table 7 ablation 给了答案：
| Method (Stage-I) | FVD ↓ | MF ↑ | MPJPE ↓ |
|---|---|---|---|
| D-Grasp | 58.17 | 0.599 | 36.18 |
| GraspXL (Ours) | 49.98 | 0.645 | 30.96 |

GraspXL 在 FVD、MF、MPJPE 三个指标上全面优于 D-Grasp。直觉是：GraspXL 的 trajectory 更平滑、更自然（RL 学的是 policy，可以 generalize 到 diverse object shape），而 D-Grasp 是基于 optimization 的，trajectory 在某些 object 上容易卡住或抖动。

### Section 7.7 的关键 observation：error propagation
作者承认 Stage I 的几何误差（比如 interpenetration 或 missing contact）会 propagate 到 final video，即使视频看起来 photorealistic，物理上可能 implausible。这是 decoupled pipeline 的 inherent 问题 — 没有 closed-loop feedback 从 video diffusion 回到 pose generator。这其实是一个明显的 future work 方向：用 differentiable rendering 或者把 pose generator 也变成 diffusion-based，让 final video loss 能 backprop 到 Stage I。

---

## 4. Stage II: Appearance Generation (Flux + ControlNet)

这一 stage 是 paper 的核心创新之一 — 用 multi-modal condition 控制 Flux 生成第一帧。

### 4.1 为什么需要 multi-modal condition

作者在 "Bridge Conditions for Sim-to-Real HOI Video Synthesis" 小节里有一段很关键的 reasoning：

> Depth + segmentation 这两种 simulator 能直接给的 condition **不足以** capture hand 的高 DoF — 因为 hand 有 21 个 joint、几十个 DoF，单靠 depth map 模糊（手指头会 merge），单靠 seg 看不出 finger 个数和 pose。

所以引入第三个 condition：**hand keypoint sequence**（来自 [83] HOIDiffusion 的 idea），即 2D hand keypoint 图。Table 8 的 ablation 直接验证：
| Hand Representation | FVD ↓ | PSNR ↑ | MPJPE ↓ |
|---|---|---|---|
| Mesh Projection | 29.33 | 30.17 | 36.18 |
| Keypoints (Ours) | 29.13 | 30.05 | 30.96 |

Mesh projection 更容易 self-occlusion（mesh 在 2D 投影会重叠），keypoint 显式表示 finger index 和 pose，对 MPJPE 提升明显（30.96 vs 36.18）。

### 4.2 三种 condition 的具体形式

每个 condition 都是 $H \times W \times 3$ 的 image：
- $D_0$：depth map（由 DepthCrafter [29] 估计，或 simulator 渲染）
- $S_0$：semantic segmentation mask
- $K_0$：hand keypoint image（2D keypoints 画在 image plane 上，类似 OpenPose 那种 skeleton visualization）

**注意**：这三个 condition 都是 $H \times W \times 3$ 的 image，统一通过 VAE encode，而不是用 MLP 处理 2D coordinate。Table 9 ablation：
| Encoder Type | FVD ↓ | PSNR ↑ | MPJPE ↓ |
|---|---|---|---|
| MLP | 31.59 | 30.07 | 21.96 |
| VAE (Ours) | 29.13 | 30.17 | 19.37 |

VAE 保留 local spatial information（keypoint 之间的相对位置），MLP 把 2D coordinate flatten 后丢失了 spatial prior。VAE 在 1,000 个 keypoint image 上的 PSNR 是 40.58，reconstruction error 很低。

### 4.3 ControlNet 注入机制（公式 2）

这是技术细节最密集的部分：

$$
f_l = f_l + \mathcal{Z}(f_l') \tag{2}
$$

变量说明：
- $f_l$：original Flux DiT [50] block 第 $l$ 层的 output feature
- $f_l'$：duplicated DiT block 第 $l$ 层的 output（duplicated block 的 input 是 concatenated conditions）
- $l \in \{0, 1\}$：只 inject 进前两层 DiT（不是所有层都 inject，这是计算量的 trade-off）
- $\mathcal{Z}$：zero-convolution layer，是一个 1×1 convolution，所有参数 init 为 zero

**Zero-conv 的 intuition**（这是 ControlNet 原作 [82] 的精髓）：
- 训练开始时 $\mathcal{Z}(f_l') = 0$，所以 $f_l = f_l + 0 = f_l$，网络行为等价于 original Flux，pretrained 知识完全保留
- 训练过程中 zero-conv 的 weight 慢慢从 0 学起来，condition signal 渐进式注入，不会突然破坏 generation 能力
- 只 update ControlNet 参数（duplicated DiT + zero-conv），original Flux DiT frozen

### 4.4 Latent 编码细节

每个 condition image 通过 VAE encode 到 $\frac{H}{8} \times \frac{W}{8} \times 16$ 的 latent（Flux 用 16 channel latent，和 SD3 一样）。

三个 condition（depth, seg, keypoint）channel-wise concat → $\frac{H}{8} \times \frac{W}{8} \times 48$，注入 duplicated DiT。

**只 inject 进前两层** 的设计很关键：
- 计算量小（只 duplicate 2 层 DiT，而不是全部）
- Foundation model 的 high-level semantic 仍然由 original DiT 主导，condition 只负责 spatial layout guidance
- 这种 "shallow injection" 思路在 ControlNet for SDXL 里也是类似 design choice

### 4.5 第一帧之后还要 filter

Section 6 (Implementation Details) 透露了一个关键 trick：生成时随机 sample 30 个 candidate first frame，然后用 HaMeR [49] 预测 hand keypoint，比较和 ground truth 距离，**discard bottom 25%**。这是 appearance generation 的 self-correction 机制 — Flux + ControlNet 不是 100% 准确的，需要 pose estimator 做 quality gate。

> Intuition：这其实是把 HaMeR 当成 "verifier"，和 LLM 里的 rejection sampling / best-of-N 思路一样。可以联想 RLHF 里的 reward model selection。

---

## 5. Stage III: Motion Generation (CogVideoX + ControlNet)

这一 stage 把 Stage I 的 pose 轨迹渲染成 per-frame condition，再用 video diffusion 生成完整视频。

### 5.1 Condition 的 rasterization

Stage I 输出 $\{\mathbf{h}_t, \mathbf{o}_t\}_{t=0}^{T}$，每一帧都 rasterize 成三个 condition：
- Depth map（per frame）
- Instance-level segmentation mask（per frame，注意是 instance-level，能区分 hand vs object）
- 2D hand keypoint image（per frame）

这些 condition 都从 pose trajectory **直接渲染**得到，几何上是 guaranteed consistent 的。

### 5.2 Video VAE 编码

用 pretrained video VAE encode 成：
$$
\mathbb{R}^{\frac{T+1}{4} \times \frac{H}{8} \times \frac{W}{8} \times 16}
$$

变量说明：
- $\frac{T+1}{4}$：temporal compression factor 4（CogVideoX 的 video VAE 在 time 维度 compress 4 倍）
- $\frac{H}{8}$ 和 $\frac{W}{8}$：spatial compression factor 8
- $16$：latent channel

三个 condition channel-wise concat → $\frac{T+1}{4} \times \frac{H}{8} \times \frac{W}{8} \times 48$，注入 CogVideoX 的 **12 个 duplicate DiT blocks**（比 Stage II 的 2 层多很多，因为 video 需要更强的 condition 信号来 enforce temporal consistency）。

### 5.3 Random masking strategy

每个 cue（depth / seg / keypoint）在训练时以 probability 0.2 被随机 mask 掉。Table 6 的 ablation：
| Settings | FVD ↓ | PSNR ↑ | MPJPE ↓ |
|---|---|---|---|
| 0 Mask Prob + Clean Cond | 28.56 | 30.99 | 19.01 |
| 0 Mask Prob + Noisy Cond | 34.58 | 27.11 | 23.67 |
| 0.2 Mask Prob + Clean Cond (Ours) | 29.13 | 30.17 | 19.37 |
| 0.2 Mask Prob + Noisy Cond | 30.45 | 29.67 | 20.31 |

关键 observation：
- 0 mask + clean: FVD 28.56（最优）
- 0 mask + noisy: FVD 34.58（崩了）
- 0.2 mask + clean: FVD 29.13（稍差于 0 mask clean）
- 0.2 mask + noisy: FVD 30.45（只比 clean 差一点）

**Intuition**：random masking 强制模型不 over-rely 任何一个 modality。当 condition noisy 时，没有 masking 的模型会 overfit 到 noisy signal，而 masking 训练的模型学会了在缺一个 modality 时用其他 modality 补偿。这是一种 **condition-level dropout / data augmentation**，类似于 dropout 在 FC layer 的作用。

> 联想：这跟 classifier-free guidance 的 conditional dropout 思路是同源的，只不过这里是 multi-modal condition 的 dropout，让模型学到 condition 之间的互补关系。

### 5.4 CogVideoX vs SVD backbone ablation

Table 10 很有信息量：
| Backbone | FVD ↓ | PSNR ↑ | MPJPE ↓ |
|---|---|---|---|
| InterDyn (SVD w/ single cond) | 38.83 | 24.86 | 28.15 |
| SVD w/ multi conds | 34.91 | 25.84 | 25.11 |
| Ours (CogVideoX w/ multi conds) | 29.13 | 30.17 | 19.37 |

两个观察：
1. 同样是 SVD backbone，multi-cond 比 single-cond 全面更好（38.83 → 34.91 FVD）→ multi-condition 的增益是 backbone-agnostic 的
2. 同样是 multi-cond，CogVideoX 比 SVD 更好（34.91 → 29.13）→ backbone 本身能力也很关键

CogVideoX 用的是 expert transformer，专门为 text-to-video 设计，temporal attention 机制比 SVD 更强，这是 hand temporal coherence（MPJPE 19.37 vs 25.11）的关键。

---

## 6. Evaluation Metrics 细节

这块我觉得最值得展开，因为 HOI video generation 的 evaluation 本身是个 open problem。

### 6.1 Image Quality: SSIM / LPIPS / PSNR
- SSIM (Structural Similarity)：结构相似度，看 texture
- LPIPS [84]：用 VGG/AlexNet feature 算 perceptual distance，更接近 human perception
- PSNR：pixel-level reconstruction quality

这三个都是 frame-level metric，不能 capture temporal coherence。

### 6.2 FVD (Fréchet Video Distance) [62]
类似 FID，但用 video feature（通常是 I3D 或 VideoNet encoder 提的 spatio-temporal feature），算生成 video set 和 real video set 之间的 Fréchet distance：

$$
\text{FVD} = ||\mu_r - \mu_g||^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

变量说明：
- $\mu_r, \Sigma_r$：real video feature 分布的 mean 和 covariance
- $\mu_g, \Sigma_g$：generated video feature 分布的 mean 和 covariance
- $\text{Tr}$：trace

FVD 是 HOI video 评估的 "gold standard"，但作者用 [60] (StyleGAN-V) 的实现，需要 video 数量足够大才稳定（他们用了 1,600 个 video，每个 49 帧）。

### 6.3 Motion Fidelity (MF) [72]
这个 metric 很有意思 — 不直接比较 pixel，而是比较 **点轨迹的相似度**。

具体做法：
1. 在 generated video 和 GT video 上各 sample 100 个 foreground points（hand / object 上）
2. 用 CoTracker3 [34] 追踪这些点，得到 tracklet
3. 比较轨迹相似度

公式 (3)：
$$
\text{MF} = \frac{1}{|\tilde{\mathcal{T}}|} \sum_{\tilde{\tau} \in \tilde{\mathcal{T}}} \max_{\tau \in \mathcal{T}} \text{corr}(\tau, \tilde{\tau}) + \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \max_{\tilde{\tau} \in \tilde{\mathcal{T}}} \text{corr}(\tau, \tilde{\tau}) \tag{3}
$$

变量说明：
- $\mathcal{T} = \{\tau_1, ..., \tau_T\}$：ground-truth tracklet 集合，$\tau_t \in \mathbb{R}^2$ 是某点在第 $t$ 帧的 2D 位置
- $\tilde{\mathcal{T}} = \{\tilde{\tau}_1, ..., \tilde{\tau}_T\}$：generated tracklet 集合
- 第一项：对每个 generated tracklet，找最相似的 GT tracklet（最坏 case 也有匹配）
- 第二项：对每个 GT tracklet，找最相似的 generated tracklet
- 这是 bidirectional best-match，类似 Hausdorff distance 的思路

公式 (4)：
$$
\text{corr}(\tau, \tilde{\tau}) = \frac{1}{F} \sum_{k=1}^{F} \frac{\mathbf{v}_k \cdot \tilde{\mathbf{v}}_k}{\|\mathbf{v}_k\| \|\tilde{\mathbf{v}}_k\|} \tag{4}
$$

变量说明：
- $F$：总帧数
- $\mathbf{v}_k = (v_k^x, v_k^y)$：GT tracklet 第 $k$ 帧到下一帧的 displacement vector
- $\tilde{\mathbf{v}}_k = (\tilde{v}_k^x, \tilde{v}_k^y)$：generated tracklet 同一时刻的 displacement vector
- 分子是 cosine similarity，衡量运动方向一致性

**Intuition**：MF 衡量的是 "motion direction 一致性"，而不是绝对位置 — 即使生成的 hand 位置偏了，只要 motion pattern 一致（比如都在做 grasp 动作），MF 还是会高。这避免了 pixel-level 对齐的苛刻要求，更适合 generative model 评估。

### 6.4 Hand Pose Accuracy: MPJPE
用 HaMeR [49] 在 generated video 上估计 21 个 hand joint 的 3D 位置，和 GT 比较：

$$
\text{MPJPE} = \frac{1}{21} \sum_{j=1}^{21} \|\hat{\mathbf{p}}_j - \mathbf{p}_j\|_2
$$

变量说明：
- $\hat{\mathbf{p}}_j$：HaMeR 预测的第 $j$ 个 joint 位置（root-aligned）
- $\mathbf{p}_j$：GT 第 $j$ 个 joint 位置（root-aligned）
- 21：MANO hand 的 joint 数量

MPJPE 是 mm 单位，越低越好。这里有个 subtle point：HaMeR 本身有误差，所以 MPJPE 包含两个误差源（生成误差 + HaMeR 估计误差），但作为 relative comparison 还是有效的。

---

## 7. Main Results 详细分析

### 7.1 DexYCB (Table 1)

| Method | Venue | FVD ↓ | MF ↑ | LPIPS ↓ | SSIM ↑ | PSNR ↑ | MPJPE ↓ | Resolution |
|---|---|---|---|---|---|---|---|---|
| CosHand [61] | ECCV'24 | 58.51 | 0.591 | 0.139 | 0.767 | 23.20 | 30.05 | 256×256 |
| InterDyn [3] | CVPR'25 | 38.83 | 0.680 | 0.119 | 0.848 | 24.86 | – | 256×384 |
| ManiVideo [48] | CVPR'25 | – | – | 0.079 | 0.913 | 30.10 | 57.30 | – |
| Ours w/ all | – | **29.13** | **0.712** | **0.069** | **0.914** | 30.17 | **19.37** | 480×720 |

PAM 在 FVD、MF、LPIPS、SSIM、MPJPE 上都是 SOTA，分辨率还是最高的（480×720 vs 256×256/256×384）。

值得注意：
- ManiVideo 的 MPJPE 是 57.30（远高于 PAM 的 19.37），说明 ManiVideo 虽然 PSNR/SSIM 看起来不错（pixel-level），但 hand pose 是错的 — 这正是 occlusion-aware representation 没有显式 hand supervision 的副作用
- CosHand 用 hand mask 作为唯一 condition，几何精度不够（MPJPE 30.05），且 FVD 58.51 — hand mask 信息量太低
- InterDyn 用 hand mask sequence via ControlNet，FVD 38.83 已经不错，但 MPJPE 没报告（应该也不太好）

### 7.2 OAKINK2 (Table 2)

| Method | FVD ↓ | MF ↑ | LPIPS ↓ | SSIM ↑ | PSNR ↑ | MPJPE ↓ |
|---|---|---|---|---|---|---|
| CosHand | 68.76 | 0.651 | 0.156 | 0.765 | 23.84 | 14.49 |
| Ours w/ seg | 48.97 | 0.708 | 0.084 | 0.831 | 25.76 | 9.61 |
| Ours w/ depth | 50.85 | 0.702 | 0.086 | 0.845 | 26.98 | 10.07 |
| Ours w/ hand | 52.41 | 0.671 | 0.113 | 0.838 | 25.66 | 8.01 |
| Ours w/ all | **46.31** | **0.777** | **0.081** | **0.851** | **28.36** | 7.01 |

OAKINK2 是 bimanual（双手）数据集，难度更高。注意几个 interesting point：
- 单独用 hand keypoints，MPJPE 最低（8.01）— 因为显式 hand 监督最强
- 单独用 hand keypoints，FVD 最高（52.41）— 因为缺少全局 scene 信息，appearance 不好
- 三者结合，所有 metric 都最优

这印证了作者的核心 thesis：**local cue (keypoints) + global context (depth, seg) = 最优**。任何一个 alone 都不够。

### 7.3 Ablation on Conditions (Table 3, DexYCB)

| Conditions | FVD ↓ | MF ↑ | LPIPS ↓ | SSIM ↑ | PSNR ↑ | MPJPE ↓ |
|---|---|---|---|---|---|---|
| Seg | 33.23 | 0.695 | 0.077 | 0.900 | 29.27 | 21.14 |
| Depth | 30.00 | 0.703 | 0.070 | 0.906 | 29.15 | 23.16 |
| Hand | 33.41 | 0.713 | 0.086 | 0.901 | 29.07 | 20.70 |
| Depth, Hand | 29.62 | 0.711 | 0.071 | 0.899 | 29.95 | 20.46 |
| Seg, Hand | 29.53 | 0.711 | 0.073 | 0.902 | 29.57 | 19.92 |
| Depth, Seg | 29.32 | 0.712 | 0.071 | 0.906 | 30.60 | 22.51 |
| Depth, Hand, Seg | **29.13** | **0.712** | **0.069** | **0.914** | 30.17 | **19.37** |

观察：
1. Performance 随 condition 数量单调增加 — 多 condition 设计有效
2. **Depth + Seg (no Hand)** 的 MPJPE 是 22.51，比 Hand-only (20.70) 还差 — 说明 depth 和 seg 对 hand 几何捕捉不够
3. **Seg + Hand (no Depth)** MPJPE 是 19.92，比 Depth + Hand (20.46) 略好 — segmentation 帮助 hand/object 边界识别
4. 三者结合 MPJPE 19.37，是全局最优

**Intuition**：depth 提供几何，seg 提供 semantic boundary，hand keypoints 提供 fine-grained finger pose — 三者互补。Depth + Seg 缺 hand 信息 → hand pose 错；Hand only 缺 scene → appearance 差。三者必须一起。

### 7.4 Downstream Task (Table 4) — 最有说服力的结果

用 PAM 生成的 video 作为 SimpleHand [93] 的训练数据增强：

| Setting | PA-MPJPE ↓ | PA-MPVPE ↓ | F-Score@05 ↑ | F-Score@15 ↑ |
|---|---|---|---|---|
| All real data (100%) | 5.5 | 5.5 | 0.7953 | 0.9899 |
| All gen. data | 8.2 | 8.1 | 0.6274 | 0.9626 |
| All gen. + 25% real | 6.1 | 6.0 | 0.7512 | 0.9851 |
| All gen. + 50% real | 5.5 | 5.5 | 0.8001 | 0.9879 |
| All gen. + 75% real | 5.4 | 5.3 | 0.7984 | 0.9899 |
| All gen. + 100% real | 5.3 | 5.3 | 0.8025 | 0.9904 |

关键观察：
- **50% real + synthetic = 100% real baseline**（PA-MPJPE 都是 5.5）— 这是 paper 最强的 claim
- All gen. + 100% real 比 all real 还好（5.3 vs 5.5）— synthetic data 真的能 boost 性能
- All gen. only 性能下降（8.2 vs 5.5）— synthetic data 不能完全替代 real data，但可以作为有效的 augmentation

具体生成了 3,400 个 video，共 207,400 帧，这个数据量约等于 DexYCB s0-split 的 50%（406,888 帧）。所以 PAM 生成的 data density 是很高的。

> Intuition：这其实是 PAM 最实用的价值 — 不需要替代 real data，而是用 synthetic data 把 real data 的 efficiency 提高 2 倍。对于 HOI 这种数据昂贵的领域，这是巨大的实际价值。

### 7.5 Zero-shot Cross-Dataset (Figure 8)

PAM 在 DexYCB（单手）训练，直接在 OAKINK2（双手）做 zero-shot i2v 测试。结果显示 hand pose alignment 还能保持，这是因为：
1. Pretrained video diffusion foundation 提供 general visual prior
2. ControlNet mechanism 保留了 generation capability
3. Multi-condition 是 geometric/semantic 信号，相对 dataset-invariant

但作者没给 zero-shot 的定量结果，只有 qualitative figure。这是 paper 的一个小 weakness。

---

## 8. Implementation Details 关键点

### 8.1 训练 setup
- 8 × NVIDIA 800 GPU（这 GPU 比较少见，估计是 H800 或定制版）
- Batch size: 4 × 8 = 32
- Learning rate: 1e-4
- 8,000 training steps
- AdamW optimizer
- DeepSpeed [55] 训练框架

8,000 step 看起来很少，但 foundation model 已经 pretrained，只需要 finetune ControlNet 部分，所以训练量小是合理的。

### 8.2 Resource Usage (Table 5)

| Resource | Stage I | Stage II | Stage III |
|---|---|---|---|
| Memory (GB) | 0.03 | 41.4 | 30.3 |
| Time (s) | 19.3 | 36.1 | 245.7 |

- Stage I 几乎不耗资源（GraspXL inference 很轻）
- Stage II memory 最高（41.4 GB），因为 Flux 是 12B 参数
- Stage III 时间最长（245.7 s），因为 video diffusion 要 denoise 40+ 帧
- Full pipeline 总时间 301.1 s / 40 帧 — 约 0.13 fps，是 offline data generation 的速度

---

## 9. 我的 critical analysis 与联想

### 9.1 Strengths
1. **Pipeline 设计很 clean**：三个 stage 各司其职，每个 stage 都 stand on 一个 foundation model 的 shoulder（GraspXL on RL, Flux on image diffusion, CogVideoX on video diffusion）
2. **Multi-condition 设计有理论支撑**：depth (geometry) + seg (semantic) + keypoints (fine hand pose) 三个互补维度，ablation 充分验证
3. **Downstream validation 是杀手锏**：50% real + synthetic = 100% real，这个数字很有说服力
4. **Sim-to-real 真的能 deploy**：minimal input (initial pose + target pose + object mesh) 都是 simulator 能直接给的

### 9.2 Weaknesses & Future Work

1. **Error propagation 问题**（Section 7.7 作者自己承认）：Stage I 的 geometric error 不会自动 fix。如果 GraspXL 生成 interpenetration，最终 video 也会出现物理 implausible 的 interaction。这是 decoupled pipeline 的 inherent 问题。
   - **可能的 fix**：用 differentiable rendering 让 video loss backprop 到 Stage I；或者把 Stage I 也换成 diffusion-based，做 closed-loop refinement。

2. **First frame 还是需要 filter**：用 HaMeR 做 quality gate（discard bottom 25%），说明 Flux + ControlNet 的 hand pose 准确率不够。如果 hand pose estimator 本身有 bias，这个 filter 会 propagate bias。
   - **可能的 fix**：训练一个专门针对 PAM 输出的 hand pose verifier，或者用 multi-view consistency check。

3. **没有 end-to-end 训练**：三个 stage 是分别训练的，没有 joint optimization。理论上 joint training 能让 stage 之间更好地 align。
   - **可能的 fix**：用一个 unified diffusion model 同时处理 pose + appearance + motion，类似 Sora 那种 unified world model 思路。但作者在 conclusion 里也提到这是 future work。

4. **Zero-shot evaluation 缺定量**：Figure 8 只给 qualitative，没有 FVD/MPJPE 数字。

5. **Object 类型有限**：DexYCB 和 OAKINK2 都是 rigid object，没有 articulatable object（比如剪刀、抽屉）。Hand-object interaction 的真实场景里 articulatable object 很常见。

### 9.3 与其他工作的联想

1. **Sora / World Models**：PAM 的 decoupled 思路和 Sora 的 unified world model 思路是两个极端。PAM 牺牲了 end-to-end 的优雅性，换来了 modular 的可控性和训练稳定性。在数据稀缺的 HOI 领域，modular 设计可能更 practical。

2. **RLHF / Best-of-N**：Stage II 的 rejection sampling（生成 30 个，discard 25%）本质上是 best-of-N with verifier。这个思路在 LLM 里很成熟，可以联想 RLHF 里的 reward model ranking。

3. **Classifier-Free Guidance**：Stage III 的 random masking (prob 0.2) 是 CFG 的 multi-modal 推广。CFG 是 single condition 的 dropout，这里是 multi-modal condition 的 dropout，让模型学到 condition 之间的互补关系。

4. **Diffusion + RL**：Stage I 用 RL (GraspXL)，Stage II/III 用 diffusion。这其实是 hybrid system，RL 提供 physically plausible trajectory prior，diffusion 提供 photorealistic appearance prior。未来可能看到 RL + diffusion 的更深度融合（比如 diffusion policy for RL）。

5. **Sim-to-Real Transfer**：PAM 是 sim-to-real 的一种新范式 — 不直接把 sim 渲染成 real-looking（domain randomization / domain adaptation），而是把 sim 的 trajectory prior 和 real-world 的 video distribution prior 结合起来。这和 Gen2Real [73] 思路类似，但 PAM 的 condition 设计更精细。

6. **ControlNet for Video**：PAM 把 ControlNet 思路从 image (Flux) 推广到 video (CogVideoX)，用 12 层 duplicate DiT（比 image 的 2 层多很多）。这暗示 video diffusion 对 condition 信号的需求更强，需要更深 injection。

7. **Foundation Model Stacking**：PAM 是一个经典案例 — 把 GraspXL (RL) + Flux (image diffusion) + CogVideoX (video diffusion) + HaMeR (pose estimator) + SimpleHand (downstream) 五个 pretrained model 串起来，每个都有自己的 prior，组合起来解决一个复杂问题。这是 "LLM as orchestrator" 思路在视觉领域的体现。

---

## 10. 总结：这篇 paper 的真正贡献

PAM 的核心 contribution 不是某一个技术点（multi-condition、ControlNet for video 都是已有 idea），而是 **把多个成熟技术组装成一个能跑通的 sim-to-real HOI video generation pipeline**，并且用 downstream task 验证了 synthetic data 的实际价值。

最有说服力的数字：**50% real + PAM synthetic = 100% real**。这不是 metric 上的 SOTA，而是 utility 上的 SOTA — 真正能减少 real data collection 成本的 SOTA。

对于 embodied AI 研究，这种 "synthetic data 价值验证" 比 "FVD 数字" 重要得多。FVD 29.13 vs 38.83 这种数字差异，下游 task 用得到吗？PAM 用 SimpleHand 的实验回答了：**有用**。

---

## References

- PAM project page: https://gasaiyu.github.io/PAM.github.io/
- Flux: https://github.com/black-forest-labs/flux
- ControlNet: https://arxiv.org/abs/2302.05543
- DiT: https://arxiv.org/abs/2212.09748
- CogVideoX: https://arxiv.org/abs/2408.06072
- GraspXL: https://arxiv.org/abs/2409.18181
- DexYCB: https://arxiv.org/abs/2104.04631
- OAKINK2: https://arxiv.org/abs/2403.12987
- HaMeR: https://arxiv.org/abs/2311.18237
- SimpleHand: https://arxiv.org/abs/2404.04492
- CoTracker3: https://arxiv.org/abs/2410.11831
- HOIDiffusion: https://arxiv.org/abs/2312.17658
- InterDyn: https://arxiv.org/abs/2503.07636
- ManiVideo: https://arxiv.org/abs/2501.05058
- CosHand: https://arxiv.org/abs/2409.05988
- DeepSpeed ZeRO: https://arxiv.org/abs/1910.02054
- LPIPS: https://arxiv.org/abs/1801.03924
- FVD: https://arxiv.org/abs/1812.01717
- Motion Fidelity (Space-time diffusion features): https://arxiv.org/abs/2311.16731
