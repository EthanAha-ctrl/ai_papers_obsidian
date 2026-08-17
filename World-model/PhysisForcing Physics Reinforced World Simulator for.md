---
source_pdf: PhysisForcing Physics Reinforced World Simulator for.pdf
paper_sha256: c70000a324019d599d3c3182a36d25744d4e9618eadc85c9ffd1c71e5a6e65e6
processed_at: '2026-08-06T03:36:40-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 PhysisForcing

## 一句话版本

现在的 video generator 生成 robot 操作视频时画面好看但物理不对（gripper 会扭曲、抓的杯子会飘走），这篇 paper 说：**别对所有像素一视同仁地监督，聚焦在 contact 区域，从 pixel 和 semantic 两个层次同时拧紧物理约束**。

## 问题到底是什么

你想象一个 video diffusion model 生成"机器人抓起红苹果放到架子上"的视频。画面看起来很流畅，但你仔细看：
- gripper 中途形状扭曲了（robot deformation）
- 苹果被抓起来后，gripper 和苹果脱开了，苹果自己飘过去（broken contact）
- 推一个东西，东西纹丝不动（anti-gravity / no causality）

这些错误对人类一眼就看出不对，但 model 不知道。因为 model 训练时只学"重建像素"，背景墙、桌面这些 static 像素占了 90%+ 的 loss，contact 区域的物理错误被淹没在 loss 海洋里。

之前大家怎么解决？
- **Geometry-based**：加 depth、keypoint tracking、3D reconstruction。但这些只管 local 的点在动，不管"gripper 抓着苹果"这种 semantic 关系
- **Preference-based**：DPO / GRPO 后训练 alignment，本质是事后纠错，可能为了满足物理把画质弄差
- **Simulator-based**：直接调物理引擎，开销大、不 scale

## 核心洞察：物理合理性是分层的，且是 localized 的

这篇 paper 的关键 insight 我觉得可以归纳成两条原理：

**原理一：hierarchical**
- 低层：每个点的 trajectory 要连续、要 contact-compatible（pixel level）
- 高层：gripper 和 grasped object 要耦合、pushed object 要跟着动（semantic level）

只盯一个层次都不够。只管 pixel 你会得到点都在动但关系乱的画面；只管 semantic 你可能整体关系对但每个点都在抖。

**原理二：region-focused**
物理信息不在背景墙上，不在桌面的 static 像素里。它集中在 gripper、被操作的物体、contact interface、moving regions 这少数地方。把这些 region 单独拎出来狠狠监督，背景就让它自己重建去吧。

这两条原理听起来 obvious，但仔细想想，现有方法都没同时满足。这就是 paper 的切入口。

## 方法怎么做的：三个步骤

### 步骤 1：先找到"哪里有物理戏"

给你一段 reference video，先用 CoTracker3 [https://arxiv.org/abs/2410.11831] 在第一帧 query 一堆点（比如 25×25=625 个 grid），跟踪它们在整段视频里的 trajectory。然后算每个点的累积运动幅度：

$$a_i = \sum_{t=1}^{T-1} \|\mathbf{p}_i^{t+1} - \mathbf{p}_i^t\|_2$$

变量解释：
- $a_i$：点 $i$ 的总位移量
- $\mathbf{p}_i^t$：点 $i$ 在第 $t$ 帧的 2D 位置
- $\|\cdot\|_2$：欧几里得距离

光看 $a_i$ 不行，因为背景里的相机抖动也会让点的 $a_i$ 很大。所以用 Depth-Anything-V2 [https://arxiv.org/abs/2406.09414] 估计第一帧的 depth map，给每个点加权：

$$r_i = \frac{1}{D_0(\mathbf{p}_i^0) + \epsilon}, \quad q_i = a_i \cdot r_i$$

- $D_0(\mathbf{p}_i^0)$：点 $i$ 在第一帧的 depth 值
- $r_i$：foreground 权重，depth 越小（离镜头越近）权重越大
- $\epsilon$：小常数防除零
- $q_i$：physics-informative score，综合考虑"动得多"和"在前景"

然后用 mean score 当自适应阈值筛出 active 点，把它们访问过的像素标记成 1，就得到一个 $T \times H \times W$ 的 spatiotemporal physics mask $\mathbf{M}^{phy}$。

人话版：**用 motion + depth 两个信号找出"哪里正在发生 robot 和 object 的交互"**。

### 步骤 2：Pixel-level 物理对齐

这一步我觉得是 paper 最 elegant 的设计。它没有直接生成 trajectory 然后比对，而是从 DiT 的中间层 feature 里"读出"trajectory，然后用 reference trajectory 监督。

具体怎么做？从 DiT 的一个 middle layer（实验证明 layer 15 最好）取出 hidden feature $\mathbf{H}^l$，过一个小 MLP 得到 $\hat{\mathbf{F}} \in \mathbb{R}^{T \times C \times H \times W}$。

把第一帧的 feature 当 query，其他帧的 feature 当 key：

$$\mathbf{Q} = \hat{\mathbf{F}}_0, \quad \mathbf{K}_t = \hat{\mathbf{F}}_t$$

对每个被 query 的点 $\mathbf{p}_i^0$，在 frame $t$ 上算一个 similarity map：

$$\mathbf{s}_i^t(\mathbf{x}) = \frac{\mathbf{Q}(\mathbf{p}_i^0)^\top \mathbf{K}_t(\mathbf{x})}{\sqrt{C}}$$

- $\mathbf{Q}(\mathbf{p}_i^0) \in \mathbb{R}^C$：query feature 在 $\mathbf{p}_i^0$ 处的值
- $\mathbf{K}_t(\mathbf{x}) \in \mathbb{R}^C$：key feature 在位置 $\mathbf{x}$ 处的值
- $\sqrt{C}$：scaled dot-product attention 的标准 scaling

然后把这个 similarity map softmax 之后对坐标做加权平均：

$$\hat{\mathbf{p}}_i^t = \sum_{\mathbf{x} \in \Omega} \text{Softmax}_\mathbf{x}(\mathbf{s}_i^t(\mathbf{x})) \cdot \mathbf{x}$$

这就是经典的 **soft argmax** trick。attention map 的"重心"就是预测的点位置。好处是 fully differentiable，梯度能 flow 回 DiT feature。

人话版：**让 DiT 的内部 feature 自己学会"我 query 的那个点现在跑到哪里去了"，然后用 CoTracker3 给的 ground truth trajectory 监督它**。

Loss 是 masked MSE：

$$\mathcal{L}_{pix}^{phy} = \frac{1}{|\mathbf{M}^{phy}|} \|\mathbf{M}^{phy} \odot (\mathcal{P}_{pred} - \mathcal{P}_{gt})\|_2^2$$

- $\mathcal{P}_{pred}$：从 DiT feature 读出的 trajectory
- $\mathcal{P}_{gt}$：CoTracker3 从 reference video 提取的 trajectory
- $\mathbf{M}^{phy} \odot$：只在 physics-informative region 上算 loss

这个 loss 直接打击 trajectory discontinuity 这种最 common 的 local failure mode。

### 步骤 3：Semantic-level 物理对齐

光管点的轨迹不够，还得管"区域之间的关系"。比如 gripper 和 grasped object 应该是 coupled 的，pushed object 应该跟着 contact region 动。

怎么衡量这种 relation？用一个 frozen 的 self-supervised video encoder，这里选 V-JEPA 2 [https://arxiv.org/abs/2404.08471] 的 ViT-L/16。V-JEPA 2 是通过预测 masked spatio-temporal feature 学 representation 的，它的 token 自然 capture 了 object-centric 和 interaction-centric 的结构。

具体做法：用 V-JEPA 2 抽 reference video 的 feature $\mathbf{F}^u$，用小 MLP 把 DiT 的 middle layer feature 投影到同空间 $\hat{\mathbf{F}}^u$，两者 resize 到同样的 token grid（$32 \times 16 \times 16$）。然后用之前的 physics mask 选出 $K$ 个 token（最多 512），算两边的 pairwise cosine similarity matrix：

$$\hat{\mathbf{R}}(i,j) = \frac{\hat{\mathbf{F}}_i^{\mathcal{M}} \cdot \hat{\mathbf{F}}_j^{\mathcal{M}}}{\|\hat{\mathbf{F}}_i^{\mathcal{M}}\|_2 \|\hat{\mathbf{F}}_j^{\mathcal{M}}\|_2}, \quad \mathbf{R}(i,j) = \frac{\mathbf{F}_i^{\mathcal{M}} \cdot \mathbf{F}_j^{\mathcal{M}}}{\|\mathbf{F}_i^{\mathcal{M}}\|_2 \|\mathbf{F}_j^{\mathcal{M}}\|_2}$$

- $\hat{\mathbf{R}}(i,j)$：DiT side 第 $i, j$ 个 token 之间的 cosine similarity
- $\mathbf{R}(i,j)$：V-JEPA 2 side 第 $i, j$ 个 token 之间的 cosine similarity
- 都是 $K \times K$ 的矩阵，capture 了 token 之间的 spatio-temporal relation structure

Loss 是两个矩阵的 L1 距离：

$$\mathcal{L}_{sem}^{phy} = \frac{1}{K^2} \sum_{i,j} |\hat{\mathbf{R}}(i,j) - \mathbf{R}(i,j)|$$

人话版：**不强迫 DiT 的 feature 长得像 V-JEPA 2，但强迫 DiT feature 之间的关系结构像 V-JEPA 2 的关系结构**。这是 relational knowledge distillation，比 absolute feature distillation 更鲁棒，因为 cosine similarity 是 rotation-invariant 的，student 不用完全 mimic teacher 的 representation。

为什么这个能 enforce 物理？因为 V-JEPA 2 的 relation matrix 编码了"gripper 和 object 现在是 coupled 状态"这种信息，让 DiT 的 relation matrix 对齐它，等于把这种 interaction semantics 灌进 DiT。

## 两个 Loss 为什么互补

这是 ablation 的核心发现（Table 4）：

| 配置 | R-Bench Avg. |
|------|--------------|
| baseline finetune | 44.8 |
| + pixel loss only | 47.2 |
| + semantic loss only | 46.2 |
| + 两个一起 | 47.5 |

Pixel loss 单独提升更大（+2.4），因为它直接打 trajectory discontinuity 这种最 common 的 local 错误。Semantic loss 单独提升小一点（+1.4），它主要修 global relation 错误比如 broken contact。

两个 loss 打的是**不同 error mode**，所以能叠加。这就像你训练 ResNet，一边加 BN 防 internal covariate shift，一边加 dropout 防 overfitting，它们 orthogonal 所以 1+1>2。

## 为什么 Region Focus 这么重要

Table 5 的 ablation 很说明问题：

| 配置 | Avg. |
|------|------|
| baseline | 44.8 |
| uniform supervision (不 focus) | 46.0 |
| region-focused supervision | 47.5 |

光加 physics loss 但均匀施加到所有 token，提升 +1.2。聚焦到 physics region 之后又多 +1.5。为什么？因为背景 pixel 的 motion 是 0 或者 noise，监督它们要么没信息要么是错的。把 supervision 集中在 contact 区域，等于把 signal-to-noise ratio 大幅提升。

这跟 detection 里 focal loss 的哲学一样：**easy examples（background）会 dominate loss，把它们 down-weight 才能让 hard examples（contact region）真正被学到**。

## 为什么是 Middle Layer

Table 6 测了不同 layer：

| Layer | 10 | 15 | 20 | 25 |
|-------|----|----|----|----|
| Score | 83.9 | 85.2 | 84.1 | 83.2 |

Layer 15（中间）最好。这给一个很重要的 intuition：**representation alignment 要在 backbone 的 "semantic but not finalized" 层做**。

- 太浅（layer 10）：还在 carry low-level appearance，没有 semantic structure 给你对齐
- 太深（layer 25）：已经 specialized 去预测 noise，steer 不动了
- 中间：既有 rich semantic representation 又还没固化，最容易通过 supervision 改变

这跟 BERT 的 middle layer 适合做 transfer learning、CNN 的 middle layer 适合做 feature pyramid 是一个道理。

## MoE Backbone 的 Routing 修改很巧妙

Wan2.2-I2V-A14B 是 MoE 架构，两个 expert 按 noise level 分工：high-noise expert 负责 global layout 和 motion，low-noise expert 负责 high-frequency appearance。

PhysisForcing 怎么 fine-tune？**只 fine-tune high-noise expert，而且训练时让它在整个 noise schedule 上都工作**（跨过原始的 routing boundary）。

intuition：physical structure 在 dynamics-forming stage（高 noise）就被 commit 了，所以 fine-tune high-noise expert 最 effective。但训练时让它跨 boundary 工作，能让它在 low-noise region 也学会 obey physics constraints，这样 inference 时不管被 route 到哪个 expert 都有物理性。

这是一个很 practical 的工程决策，比 fine-tune 整个 MoE 要省显存，又比只 fine-tune 半段要 robust。

## 实验讲了什么故事

三个 benchmark 上 PF-Cosmos（PhysisForcing + Cosmos3-Nano）都拿 best overall：

| Benchmark | PF-Cosmos | 最强 baseline |
|-----------|-----------|---------------|
| R-Bench | 63.8 | Wan2.6 (60.7) |
| PAI-Bench robot | 85.17 | Abot-PhysWorld (84.91) |
| EZS-Bench | 81.08 | Abot-PhysWorld (80.30) |

注意 R-Bench 上 PF-Cosmos 超过了所有 commercial model（Wan2.6, Veo 3.1, Seedance 1.5 Pro），这是相当强的 result。而且 EZS-Bench 是 zero-shot OOD benchmark，能拿最好说明物理 prior 能 generalize。

更让我惊喜的是 **policy learning** 的传导效应（Table 2）：

| Task | Baseline | PhysisForcing | Δ |
|------|----------|---------------|---|
| place_empty_cup | 41.5 | 63.0 | +21.5 |
| press_stapler | 49.0 | 60.0 | +11.0 |
| Average | 68.2 | 72.8 | +4.6 |

把 PhysisForcing 训练的 video model 当 Fast-WAM 的 video backbone，policy 成功率显著上升，尤其 contact-rich 的 place 和 press 任务。这背后 intuition 是：**video backbone 学到的 representation 如果物理一致，policy 从中 extracted 的 action 也更 physically meaningful**。

在 WorldArena 闭环 action planner 测试里，success rate 从 16% 提升到 24%，超过所有 baseline（包括最强的 WoW 20.5%）。

## 整体设计哲学

我把这篇 paper 的 design philosophy 总结成几条：

**1. Training-time 而非 post-hoc**。DPO / GRPO 是在已经训练好的 model 上做 alignment，本质是 correcting。PhysisForcing 在 fine-tuning 阶段就 inject physics supervision，是 preventing。前者可能要 trade visual quality 换 physics，后者因为 physics 是 inductive bias 的一部分，Quality Score 反而保持 competitive（看 Table 7 和 8 的 Quality column）。

**2. 不改变 inference architecture**。所有 auxiliary model（CoTracker3、Depth Anything、V-JEPA 2）只在训练时用，inference 时全部 discard，zero overhead。这是工程上非常重要的属性，意味着可以无缝 plug 进任何现有 video generation pipeline。

**3. Hierarchical + region-focused**。这两个原则其实是 orthogonal 的：pixel/semantic 是"层次"维度，region focus 是"空间"维度。两个维度同时约束，形成了一个 2D 的 supervision 矩阵，既覆盖了 physical plausibility 的不同 level，又避免 background 信号稀释。

**4. Relational distillation 而非 absolute distillation**。Semantic loss 对齐的是 cosine similarity matrix，不是 absolute feature。这让 student model 有自由度去学自己的 representation space，只要 preserve 关系结构即可。这比传统的 feature distillation 更鲁棒。

## 一些延伸联想

写到这里我想跟你聊几个延伸的方向：

**1. 跟 self-supervised learning 的深层 connection**

V-JEPA 2 [https://arxiv.org/abs/2404.08471] 本质是 LeCun 那派 JEPA 思路的 video 版，通过 predict masked feature 学 representation。它为什么 capture 到 interaction-centric structure？因为预测 masked spatio-temporal token 必须理解 object 之间怎么 interact，不然填不出来。这跟 DINO [https://arxiv.org/abs/2508.10104] 的 self-supervised vision encoder 学到 object-centric representation 是一个道理。

PhysisForcing 借用了 V-JEPA 2 的"理解"作为 supervision target，本质是 distill V-JEPA 2 的 world understanding 进 DiT。这跟 VideoRePA [https://arxiv.org/abs/2505.23656] 的 relational alignment 思路是一脉相承的。

**2. Soft argmax 的历史渊源**

Soft argmax 这个 trick 历史悠久，从 spatial soft argmax 在 human pose estimation（"Differentiable Spatial Regression for Pose"）到 PIPs、TAPIR 这些 point tracker 都用。核心好处是 differentiable，让 attention map 能通过 coordinate 误差被监督。

PhysisForcing 把它用在 DiT feature 上，等于让 DiT 的中间层 feature 隐式地学会 correspondence。这跟 correspondence-based methods（如 RAFT、GMFlow）有异曲同工之妙，只不过这里 feature 是在 denoising 过程中动态变化的。

**3. 为什么不直接用 3D physical prior**

一个 natural question：为什么不直接 inject 3D 几何约束？比如用 3D trajectory、multi-view consistency、或者 physics engine 模拟。

我猜原因是 cost 和 scalability。3D supervision 需要准确的 3D annotation（expensive），或者 multi-view setup（restrictive）。PhysisForcing 只需要 monocular video + off-the-shelf 2D tracker + depth estimator，data 准入门槛低很多。而且 2D trajectory 其实已经隐含了很多 3D 信息（接触点、运动连贯性），通过 V-JEPA 2 的 semantic alignment 还能 capture 一部分 3D interaction semantics。

未来如果结合 3D-aware methods（如 TesserAct [https://arxiv.org/abs/2504.20995]）可能更强，但 cost 也会上去。

**4. 这套思路能 generalize 到哪里**

我觉得 PhysisForcing 的 design pattern 不止 robotics。任何生成任务只要满足两个条件就能用：
- 生成内容有"物理 / 结构 prior"
- 这个 prior 是 localized 的，不是 uniform 分布的

比如：
- **自动驾驶 video generation**：focus 在 ego-vehicle 和其他 traffic participant 的 interaction region，用 tracker + HD map 监督
- **Human motion generation**：focus 在 joint 和 contact region，用 motion capture 数据监督
- **Tool use video**：focus 在 tool 和 workpiece 的接触区，用 force/tactile 信号监督

核心 abstraction 是：**找出 prior 信号最集中的 spatiotemporal region，在那里狠狠 supervision，其余地方让 backbone 自己 generalization**。

**5. Long-horizon 是下一个挑战**

当前 trajectory alignment 是 per-point 的，时间维度是 $T$ 帧的 trajectory。但 long-horizon 的物理因果（"先打开抽屉才能拿出东西"）其实是更高层的 causal structure，per-point trajectory 抓不到。这可能需要 temporal hierarchical supervision，或者引入 causal model。

paper 在 Limitations 里也承认当前 open-source backbone 的 long-horizon temporal reasoning 有限。我觉得这是下一阶段的重要方向——能不能用 LLM 提供 causal supervision（"step A 必须在 step B 之前"）来补这个 gap。

## 我会怎么 build on this work

如果让我接着做，我会想几个方向：

**A. 加 force / tactile 信号**。当前只用 visual signal 监督物理，但真正的物理包括力。如果能用 tactile sensor 数据（哪怕少量）作为 contact region 的额外 supervision，可能能 enforce 更 fine-grained 的 physical plausibility。

**B. Action-conditioned version**。当前 method 是 video-conditional，没有 explicit 的 action input。在真正的 world model 里，action 是因果输入。能不能把 action embedding inject 进 DiT，然后让 trajectory alignment 不仅匹配 trajectory 还匹配 action-conditioned trajectory？

**C. Multi-scale region mask**。当前 physics mask 是 binary 的，可以扩展成 soft mask 或者 multi-level mask（contact region 权重最高，manipulated object 次之，其他 motion region 再次）。这可能让 supervision 更精细。

**D. Iterative refinement**。现在是一次性 fine-tune，可以做成 EM-like 的 iterative：先 fine-tune，再在新 model 上重新 extract mask 和 trajectory，再 fine-tune。这样 mask 会越来越准。

## Reference

- Paper: https://dagroup-pku.github.io/PhysisForcing.github.io/
- CoTracker3: https://arxiv.org/abs/2410.11831
- V-JEPA 2: https://arxiv.org/abs/2404.08471
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- VideoRePA: https://arxiv.org/abs/2505.23656
- DINOv3: https://arxiv.org/abs/2508.10104
- Wan: https://arxiv.org/abs/2503.20314
- Cosmos 3: https://arxiv.org/abs/2606.02800
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- WorldArena: https://arxiv.org/abs/2602.08971
- Fast-WAM: https://arxiv.org/abs/2603.16666
- R-Bench: https://arxiv.org/abs/2601.15282
- PAI-Bench: https://arxiv.org/abs/2512.01989
- EZS-Bench / ABot-PhysWorld: https://arxiv.org/abs/2603.23376
- TesserAct: https://arxiv.org/abs/2504.20995

---

总结一句人话：**找到机器人视频中真正发生物理交互的少数区域，在那里从点的运动和区域关系两个层次同时拧紧螺丝，让 video diffusion model 学到的 representation 既视觉流畅又物理合理，而且这个合理性还能传导到下游 policy 提升机器人操作成功率**。

希望这个版本讲得够直观，Andrej。还想聊哪一块（比如 V-JEPA 2 为什么能 capture interaction structure、或者 soft argmax 的 gradient 怎么 flow、或者跟 Dreamer 那类 world model 的对比）我都可以继续展开。

---

# PhysisForcing: Physics Reinforced World Simulator 深度技术讲解

## 一、Big Picture 与 Motivation

这篇 paper 来自 Peking University 和 NVIDIA,核心问题非常清晰:把 video generation model 当作 embodied world simulator 用的时候,生成的视频虽然 visually plausible 但 physically implausible。具体来说,作者发现 physical instability 主要来自两个因素:

1. **Moving objects 的 deformation** —— 比如 gripper 自己扭曲、grasped object 形状漂变
2. **Interacting entities 之间不合理的 spatio-temporal correlations**,尤其在 contact 区域 —— 比如 pushed object 不动、grasped object 飘走

Andrej 你看 Figure 1 里展示的几个典型 failure mode,这些都是当前 video diffusion model 的通病:discontinuous gripper trajectories、object penetration、anti-gravity motion、broken contact。这些错误直接破坏了生成视频作为 visual world simulator 的可靠性。

核心 insight 是: **physical plausibility 在 manipulation video 里是天然 hierarchical 的**:
- **Pixel level**: local motion 应该满足 trajectory continuity、depth consistency、contact-compatible displacement
- **Semantic level**: object relations 应该根据 interaction semantics 演化 —— pushed object 应该移走、grasped object 应该和 gripper 耦合、placed object 应该静止在 support surface 上

同时,physical evidence 高度 localized 在 manipulators、objects、contact areas、moving regions 周围。Uniform supervision over all pixels 会 dilute 这些信号。这引出两个设计原则:**hierarchical** 和 **region-focused**。

## 二、Related Work 的 Positioning

让我梳理一下这个领域的 landscape,帮你建立 mental model:

### 2.1 Embodied video generation 和 world models
- **General-domain video models**: Sora [34], Wan [45], HunyuanVideo [28] —— visual fidelity 强但缺乏 embodied contact dynamics 的 exposure
- **Robot-specific models**: Cosmos [1], DreamGen [24], GR-2 [6], Vidar [15], Unified Video Action Model [30] —— 提升了 task relevance 但用 reconstruction objective 训练,physically critical regions 和 background pixels 被同等对待
- **Video generators as policies**: [23, 31] 直接用 video generator 当 policy
- **Interactive world models**: Genie [5], Genie Envisioner [32]

### 2.2 Physics-aware world simulator 的三个 paradigm
**Geometry-based methods**:
- Depth prediction: Video Depth Anything [9], Depth Anything V2 [48]
- Keypoint tracking: CoTracker [25, 26], AllTracker [22]
- 3D structure: RoboScape [39] (jointly learns temporal depth + keypoint dynamics), RoboDreamer [58] (compositional world models), CTRL-World [18] (controllable generative world model), TesserAct [56] (4D embodied world models)
- 问题: 只 capture local spatial structure, 不 explicit model semantic-level interactions

**Preference-based methods**:
- ABot-PhysWorld [11]: DPO-based training with physics-aware discriminators
- MIND-V [54]: GRPO with Physical Foresight Coherence reward
- 问题: post-hoc correcting 而不是 preventing,可能牺牲 visual quality

**Simulator-based methods**:
- Physics engines [46, 49] 或 test-time selection
- 问题: 计算开销大,scalability 受限

PhysisForcing 的独特之处:**training-time、hierarchical、region-focused**,在 training 时直接 inject physics supervision 而不是 post-hoc 修正。

## 三、Method 深度解析

### 3.1 Physics-informative Region Extraction

这一步的目标是从 reference video 里识别出 robot-object interaction 发生的 regions。整个 pipeline 很 elegant:

**Step 1: Dense trajectory extraction**

用 off-the-shelf point tracker (CoTracker3 [25]) 得到 dense temporal trajectories:
$$\mathcal{P} = \{\mathbf{p}_i^{1:T}\}_{i=1}^N$$

其中:
- $N = H \times W$ 是 query point 数量(在 first frame 上 query 满整个 image grid)
- $\mathbf{p}_i^t \in \mathbb{R}^2$ 是 point $i$ 在 frame $t$ 的 tracked 2D location
- 上标 $1:T$ 表示从 frame 1 到 frame $T$ 的完整 trajectory

**Step 2: Motion magnitude computation**

公式 (1):
$$a_i = \sum_{t=1}^{T-1} \|\mathbf{p}_i^{t+1} - \mathbf{p}_i^t\|_2$$

变量解释:
- $a_i$: point $i$ 的累积运动幅度
- $\mathbf{p}_i^{t+1} - \mathbf{p}_i^t$: 相邻两帧之间的 displacement vector
- $\|\cdot\|_2$: L2 norm,即欧几里得距离
- 求和 over $t$ 从 1 到 $T-1$:累加所有相邻帧对的位移

直觉: $a_i$ 越大表示这个 point 的 local motion 越强。但 motion magnitude alone 会把 background jitter 也高亮出来。

**Step 3: Depth-aware foreground weighting**

公式 (2):
$$r_i = \frac{1}{D_0(\mathbf{p}_i^0) + \epsilon}, \quad q_i = a_i \cdot r_i$$

变量解释:
- $D_0 \in \mathbb{R}^{H \times W}$: first frame 的 estimated depth map(用 Depth-Anything-V2 [48] 的 ViT-L variant)
- $D_0(\mathbf{p}_i^0)$: point $i$ 在 first frame 位置处的 depth value
- $r_i$: foreground weight —— depth 越小(越靠近相机)权重越大,这是 monocular depth 的 relative 性质决定的
- $\epsilon$: small constant for numerical stability(避免除零)
- $q_i$: physics-informative score,combine 了 motion magnitude 和 foreground relevance

直觉: robot-object interaction 通常在 foreground,depth-aware weighting 把 background motion 的干扰压制下去。

**Step 4: Adaptive thresholding 生成 mask**

公式 (3):
$$\mathbf{M}_i^{phy} = \mathbb{I}\left(q_i \geq \frac{1}{N} \sum_{j=1}^N q_j\right)$$

变量解释:
- $\mathbb{I}(\cdot)$: indicator function,条件成立返回 1,否则返回 0
- $\frac{1}{N} \sum_{j=1}^N q_j$: 所有 query points 的 mean score,作为 adaptive threshold
- $\mathbf{M}_i^{phy} \in \{0, 1\}$: trajectory-level 的 binary mask

直觉: 用 mean 作为 threshold 是 adaptive 的,不需要手动调参,适应不同 video 的 motion 分布。

**Step 5: Spatiotemporal mask construction**

公式 (4):
$$\mathbf{M}_t^{phy}(\lfloor \mathbf{p}_i^t \rceil) = 1, \quad \text{if } \mathbf{M}_i^{phy} = 1, \quad t = 1, \ldots, T$$

变量解释:
- $\mathbf{M}_t^{phy} \in \{0, 1\}^{H \times W}$: frame $t$ 的 spatial mask
- $\lfloor \cdot \rceil$: rounding to nearest pixel(把 continuous coordinates 量化到 pixel grid)
- 把 selected trajectories 投影到每个 frame 上,initialize 为 0,被 visited 的 pixel 设为 1

最终得到 $\mathbf{M}^{phy} \in \{0, 1\}^{T \times H \times W}$ 的 spatiotemporal physics mask。

### 3.2 Pixel-level Physics Alignment

这个 module 的核心思想: **用 attention-like 机制从 DiT feature 里 decode 出 point trajectories,然后用 reference trajectories 监督**。这是 paper 里最 elegant 的设计。

**Step 1: DiT feature extraction**

从 denoising transformer 的 intermediate block (middle layer $l$) 提取 hidden feature $\mathbf{H}^l$,用 lightweight MLP $\phi(\cdot)$ refine:
$$\hat{\mathbf{F}} = \text{reshape}(\phi(\mathbf{H}^l)) \in \mathbb{R}^{T \times C \times H \times W}$$

为什么用 middle layer 而不是最后一层?Section 4.4 的 ablation 给出答案:early blocks 还在 carry shallow appearance features,late blocks 已经 specialized for final noise prediction 很难 steer,middle block 是 best trade-off。具体在 Wan2.2-TI2V-5B 上,layer 15 (middle) 拿到 85.2,layer 10 (shallow) 拿到 83.9,layer 25 (deep) 拿到 83.2。

**Step 2: Query-Key 形式化 trajectory reading**

公式 (5):
$$\mathbf{Q} = \hat{\mathbf{F}}_0, \quad \mathbf{K}_t = \hat{\mathbf{F}}_t, \quad t = 1, \ldots, T-1$$

变量解释:
- $\hat{\mathbf{F}}_0$: first frame 的 feature map,作为 query
- $\hat{\mathbf{F}}_t$: frame $t$ 的 feature map,作为 key
- 因为所有 trajectory 都是从 first frame query 的,所以用 first-frame feature 作为 query 是 natural 的

**Step 3: Similarity map computation**

公式 (6):
$$\mathbf{s}_i^t(\mathbf{x}) = \frac{\mathbf{Q}(\mathbf{p}_i^0)^\top \mathbf{K}_t(\mathbf{x})}{\sqrt{C}}, \quad \mathbf{x} \in \Omega$$

变量解释:
- $\mathbf{Q}(\mathbf{p}_i^0) \in \mathbb{R}^C$: query feature 在 point $\mathbf{p}_i^0$ 处的采样值
- $\mathbf{K}_t(\mathbf{x}) \in \mathbb{R}^C$: key feature 在 frame $t$ 位置 $\mathbf{x}$ 处的采样值
- $\sqrt{C}$: scaled dot-product attention 里的 standard scaling
- $\Omega$: spatial grid of size $H \times W$
- $\mathbf{s}_i^t \in \mathbb{R}^{H \times W}$: point $i$ 在 frame $t$ 上对所有 spatial locations 的 similarity map

这就是一个 dense feature matching 的操作 —— 类似 DETR 里的 cross-attention,但 here 是 spatial correspondence。

**Step 4: Coordinate expectation**

公式 (7):
$$\hat{\mathbf{p}}_i^t = \sum_{\mathbf{x} \in \Omega} \text{Softmax}_\mathbf{x}(\mathbf{s}_i^t(\mathbf{x})) \mathbf{x}$$

变量解释:
- $\text{Softmax}_\mathbf{x}$: over spatial dimension 的 softmax
- $\hat{\mathbf{p}}_i^t \in \mathbb{R}^2$: predicted location of point $i$ at frame $t$
- 这是 differentiable 的 —— 用 softmax probability 作为 weights 对 coordinates 做 weighted average,类似 soft attention

直觉: 这是把 attention map 转成 coordinate 的经典 trick,在 PIPs、TAPIR 这些 point tracker 里很常见,叫 "soft argmax" 或 "coordinate expectation"。好处是 fully differentiable,梯度可以 flow back 到 DiT feature。

**Step 5: Masked MSE loss**

公式 (8):
$$\mathcal{L}_{pix}^{phy} = \frac{1}{|\mathbf{M}^{phy}|} \|\mathbf{M}^{phy} \odot (\mathcal{P}_{pred} - \mathcal{P}_{gt})\|_2^2$$

变量解释:
- $\mathcal{P}_{pred} = \{\mathbf{p}_{i, pred}^t\}_{i,t}$: 从 predicted video 里 inferred 的 trajectories
- $\mathcal{P}_{gt} = \{\mathbf{p}_{i, gt}^t\}_{i,t}$: 用 CoTracker3 从 reference video 提取的 trajectories
- $\mathbf{M}^{phy}$: physics-informative mask
- $\odot$: element-wise multiplication (Hadamard product)
- $|\mathbf{M}^{phy}|$: mask 里的有效点数,作为 normalization
- $\|\cdot\|_2^2$: squared L2 norm

直觉: 这个 loss 直接约束 DiT feature 在 physics-informative regions 里的 trajectory continuity,而且通过 soft argmax 把 spatial supervision 转成 coordinate-level supervision,比 pixel-wise reconstruction loss 更 focus 在 motion 一致性上。

### 3.3 Semantic-level Physics Alignment

这个 module 的核心思想: **用 frozen self-supervised video encoder (V-JEPA 2 [3]) 作为 measurement space,把它捕捉到的 inter-region relations transfer 到 DiT 上**。

为什么这样设计?因为 pixel-level trajectory 只能约束 point-wise motion,但 manipulation plausibility 还依赖 region 之间的关系。比如:
- Gripper 和 grasped object 应该 tightly coupled
- Pushed object 应该和 contact region 联合移动

这种 inter-region coupling 自然地 captured 在 self-supervised video encoder 的 pairwise token similarities 里。这其实是借用了 representation alignment 思路,类似 VideoRePA [55] 和 DINOv3 [41] 的 relational alignment。

**Step 1: Dual representation extraction**

公式 (9):
$$\mathbf{F}^u = \Phi_u(\mathcal{V}), \quad \hat{\mathbf{F}}^u = \text{Resize}(\psi(\mathbf{H}^l))$$

变量解释:
- $\Phi_u(\cdot)$: frozen video understanding encoder(用 V-JEPA 2 [3] 的 ViT-L/16 variant,hidden width 1024)
- $\mathcal{V}$: input video
- $\mathbf{F}^u$: encoder 输出的 feature,spatio-temporal token grid 是 $32 \times 16 \times 16$ (tubelet 2, patch 16)
- $\psi(\cdot)$: lightweight MLP,把 DiT feature project 到 encoder 的 representation space
- $\hat{\mathbf{F}}^u$: 经过 resize/interpolation/padding 后 dimensionally aligned with $\mathbf{F}^u$ 的 DiT feature

具体在 Wan2.2-I2V-A14B 上,DiT block 20 (width 5120) 的 hidden feature 被 MLP mapped 到 V-JEPA 2 的 feature space,再 trilinearly resample 到 $32 \times 16 \times 16$ grid,保证 student 和 teacher 的 tokens index-aligned。

**Step 2: Mask-based token selection**

公式 (10):
$$\hat{\mathbf{F}}^{\mathcal{M}} = \{\hat{\mathbf{F}}_{t,n}^u \mid (t,n) \in \mathcal{M}\} \in \mathbb{R}^{K \times C}$$
$$\mathbf{F}^{\mathcal{M}} = \{\mathbf{F}_{t,n}^u \mid (t,n) \in \mathcal{M}\} \in \mathbb{R}^{K \times C}$$

变量解释:
- $\mathcal{M}$: mask-induced token index set(把 physics mask resize 到 token resolution)
- $K$: selected tokens 数量(最多 $K = 512$)
- $C$: aligned feature dimension
- $\hat{\mathbf{F}}^{\mathcal{M}}$: 从 DiT side 选出的 tokens
- $\mathbf{F}^{\mathcal{M}}$: 从 encoder side 选出的 tokens

这里很关键: 只在 physics-informative regions 上做 alignment,不让 background tokens 主导。

**Step 3: Pairwise relational matrices**

公式 (11):
$$\hat{\mathbf{R}}(i,j) = \frac{\hat{\mathbf{F}}_i^{\mathcal{M}} \cdot \hat{\mathbf{F}}_j^{\mathcal{M}}}{\|\hat{\mathbf{F}}_i^{\mathcal{M}}\|_2 \|\hat{\mathbf{F}}_j^{\mathcal{M}}\|_2}$$
$$\mathbf{R}(i,j) = \frac{\mathbf{F}_i^{\mathcal{M}} \cdot \mathbf{F}_j^{\mathcal{M}}}{\|\mathbf{F}_i^{\mathcal{M}}\|_2 \|\mathbf{F}_j^{\mathcal{M}}\|_2}$$

变量解释:
- $\hat{\mathbf{F}}_i^{\mathcal{M}}, \hat{\mathbf{F}}_j^{\mathcal{M}}$: DiT side 的第 $i, j$ 个 token
- $\mathbf{F}_i^{\mathcal{M}}, \mathbf{F}_j^{\mathcal{M}}$: encoder side 的第 $i, j$ 个 token
- 分子: dot product
- 分母: L2 norm 的乘积,等价于 cosine similarity
- $\hat{\mathbf{R}}, \mathbf{R} \in \mathbb{R}^{K \times K}$: pairwise spatio-temporal relation matrices

直觉: cosine similarity matrix 是 rotation-invariant 的,所以只 align relations 而不 align absolute features。这让 student 不需要完全 mimic teacher 的 representation,只需要 preserve 它的 relational structure。

**Step 4: Relational alignment loss**

公式 (12):
$$\mathcal{L}_{sem}^{phy} = \frac{1}{K^2} \sum_{i=1}^K \sum_{j=1}^K |\hat{\mathbf{R}}(i,j) - \mathbf{R}(i,j)|$$

变量解释:
- $K^2$: 总的 pairwise relation 数量,作为 normalization
- $|\cdot|$: absolute value (L1 distance)
- 这个 loss 鼓励 DiT feature 在 physics-informative regions 上的 token-to-token relation structure 匹配 V-JEPA 2 的 relation structure

为什么用 L1 而不是 L2?L1 对 outlier 更 robust,而且 cosine similarity 的范围在 $[-1, 1]$,L1 distance 自然 bounded。

### 3.4 整体 Training Objective

公式 (13):
$$\mathcal{L} = \mathcal{L}_{FM} + \lambda_{pix} \mathcal{L}_{pix}^{phy} + \lambda_{sem} \mathcal{L}_{sem}^{phy}$$

变量解释:
- $\mathcal{L}_{FM}$: standard flow matching loss(video diffusion 的主 loss)
- $\lambda_{pix}, \lambda_{sem}$: 两个 physics loss 的平衡权重
- 关键点: **所有 auxiliary models 只在 training 时用,inference 时全部 discard,zero extra inference cost**

这个设计非常重要 —— 它没有改变 inference 时的 model architecture,只是 fine-tuning 阶段加额外的 supervision signal。

## 四、Backbone 实现细节

paper 在三个 backbone 上做了实验,值得详细讲一下每个的特点:

### 4.1 Wan2.2-I2V-A14B [45]
- Mixture-of-Experts (MoE) image-to-video diffusion transformer
- 两个 ~14B-parameter denoiser experts(总共 27B,每步 active 14B)
- 建在 Wan2.1 3D causal VAE 上,spatio-temporal compression ratio $T \times H \times W = 4 \times 8 \times 8$
- MoE 路由: SNR-based boundary $t_{moe}$ 把 diffusion trajectory 切成两段
  - High-noise expert: $t \geq t_{moe}$,负责 global layout、motion、object configuration
  - Low-noise expert: $t < t_{moe}$,refine high-frequency appearance
- PhysisForcing 只 fine-tune high-noise expert,因为 physical structure 在 dynamics-forming stage 被确定
- Training 时 deviate 原始 MoE routing:把 high-noise expert apply 到 full $t \in [0, T]$ range,每个 step uniform sample $t$,让 fine-tuned expert 学会在整个 denoising trajectory 上 obey physics constraints

### 4.2 Wan2.2-TI2V-5B [45]
- Unified text/image-to-video diffusion transformer
- 单个 ~5B-parameter denoiser,没有 MoE
- 配 new Wan2.2-VAE,compression ratio $T \times H \times W = 4 \times 16 \times 16$,总 reduction 64×
- 加上 patchify layer,effective compression $4 \times 32 \times 32$

### 4.3 Cosmos3-Nano [1]
- ~16B-parameter Mixture-of-Transformers (MoT) video model
- 建在 Qwen3-VL-8B backbone 上(hidden width 4096, 36 transformer blocks)
- 同样用 Wan2.2-VAE,720p resolution,up to 189 frames
- 用 LoRA fine-tune,遵循官方 image-to-video post-training setting

## 五、实验结果分析

### 5.1 R-Bench 结果(主表)

Table 1 给出非常完整的对比。关键数据:

| Model | Avg. | Manipulation | Spatial | Multi-entity | Long-horizon | Reasoning | Single arm | Dual arm | Quadruped | Humanoid |
|-------|------|--------------|---------|--------------|--------------|-----------|------------|----------|------------|----------|
| PF-Cosmos | **63.8** | 56.4 | 65.4 | 45.2 | 58.4 | 47.8 | 68.7 | 69.6 | 69.2 | 68.5 |
| Wan2.6 (commercial) | 60.7 | 54.6 | 65.6 | 47.9 | 51.4 | 53.1 | 66.6 | 68.1 | 72.3 | 66.7 |
| Veo 3.1 | 59.9 | 54.1 | 47.4 | 53.4 | 59.2 | 46.7 | 67.0 | 66.6 | 74.3 | 70.4 |
| PF-Wan | 62.0 | 56.4 | 65.4 | 45.2 | 58.4 | 47.8 | 68.7 | 69.6 | 69.2 | 68.5 |
| Wan2.2-I2V-A14B (base) | 50.7 | 38.1 | 45.4 | 37.3 | 50.1 | 33.0 | 60.8 | 58.2 | 69.0 | 64.8 |
| Wan2.2-I2V-A14B (ft) | 57.9 | 52.3 | 62.8 | 45.2 | 54.5 | 47.8 | 64.2 | 63.5 | 65.6 | 65.3 |

几个 takeaway:
- PF-Cosmos 拿到 best overall (63.8),超越最强 commercial model Wan2.6 (60.7),这是相当 strong 的 result
- PF-Wan 比 base model 提升 +22.3% (50.7 → 62.0),比 vanilla finetune 提升 +7.1% (57.9 → 62.0)
- Cosmos3-Nano variant 提升 +9.2% over base
- 在 Manipulation、Spatial、Multi-entity、Long-horizon 这些 task 上都有 consistent improvement,说明 physical alignment 不是 overfitting 到某个 metric

### 5.2 PAI-Bench 结果

Table 7 给出 full results:

| Model | Quality | Domain | Avg. |
|-------|---------|--------|------|
| PF-Cosmos | 77.08 | **93.26** | **85.17** |
| Abot-PhysWorld | 76.76 | 93.06 | 84.91 |
| Cosmos3-Nano (ft) | 76.52 | 91.54 | 84.03 |
| Wan2.5 | 75.48 | 86.44 | 80.96 |
| PF-Wan | 76.26 | 88.20 | 81.73 |
| Wan2.2-I2V-A14B (ft) | 75.38 | 84.42 | 79.90 |

观察:
- PF-Cosmos 的 Domain Score 93.26 最高(物理语义合理性)
- Quality Score 没有显著下降,说明 physical alignment 不牺牲 visual quality
- vs Abot-PhysWorld 的 DPO 方法,PhysisForcing 在 training-time 就 prevent 错误,而不是 post-hoc correct

### 5.3 EZS-Bench 结果

Table 8:这是一个 training-independent zero-shot benchmark,196 个 unseen robot-task-scene combinations,probe out-of-distribution generalization:

| Model | Quality | Domain | Avg. |
|-------|---------|--------|------|
| PF-Cosmos | 76.95 | **85.20** | **81.08** |
| Abot-PhysWorld | 76.94 | 83.66 | 80.30 |
| Cosmos3-Nano (ft) | 77.42 | 83.16 | 80.29 |
| PF-Wan | 76.58 | 84.49 | 80.54 |
| Wan2.2-I2V-A14B (ft) | 76.12 | 81.95 | 79.04 |

PF-Cosmos 在 OOD setting 下也是 best,Domain Score 提升 +2.04 over vanilla finetune,说明学到的 physical priors 能 generalize。

### 5.4 Policy Learning 结果

这是 paper 的一个重要 extension —— 不仅 evaluate generation quality,还 evaluate downstream policy。

**作为 video backbone for world action modeling**(Table 2):
- 用 PhysisForcing-trained Wan2.2-TI2V-5B plug 进 Fast-WAM [52] 作为 drop-in replacement
- 在 RoboTwin 2.0 [10] 的 6 个 contact-rich tasks 上 evaluate,每个 200 rollouts
- 平均成功率: 68.2% → 72.8% (+4.6%)
- 最大提升在 contact-rich 的 place_empty_cup (41.5% → 63.0%, +21.5%) 和 press_stapler (49.0% → 60.0%, +11.0%)
- 有两个 task 略有下降: shake_bottle (-3.0%) 和 stack_bowls_two (-6.5%) —— 可能因为这些 task 不需要那么 precise 的 contact reasoning,physical constraint 反而限制了 free-form motion

**作为 WorldArena action planner**(Table 3):

| Model | Task 1 | Task 2 | Avg. |
|-------|--------|--------|------|
| + PhysisForcing | **22.0** | **26.0** | **24.0** |
| WoW [12] | 20.0 | 21.0 | 20.5 |
| TesserAct [56] | 1.0 | 35.0 | 18.0 |
| Wan2.2-TI2V-5B (base) | 12.0 | 20.0 | 16.0 |

Closed-loop success rate 从 16.0% 提升到 24.0%,超越所有 baseline,包括最强 WoW (20.5%)。这说明 physically aligned video model 提供 stronger representations for embodied intelligence。

### 5.5 Ablation Studies

**Component ablation** (Table 4):

Wan2.2-TI2V-5B:
| Model | Emb. | Tasks | Avg. |
|-------|------|-------|------|
| baseline (ft) | 56.5 | 35.4 | 44.8 |
| + $\mathcal{L}_{pix}^{phy}$ only | 59.0 | 37.8 | 47.2 |
| + $\mathcal{L}_{sem}^{phy}$ only | 58.4 | 36.5 | 46.2 |
| + PhysisForcing (both) | 58.2 | 38.9 | 47.5 |

Wan2.2-I2V-A14B:
| Model | Emb. | Tasks | Avg. |
|-------|------|-------|------|
| baseline (ft) | 64.7 | 52.5 | 57.9 |
| + $\mathcal{L}_{pix}^{phy}$ only | 67.5 | 55.2 | 60.7 |
| + $\mathcal{L}_{sem}^{phy}$ only | 66.8 | 54.6 | 60.0 |
| + PhysisForcing (both) | 69.0 | 56.3 | 62.0 |

观察:
- 两个 loss 是 **complementary** 的,组合比单独用任一个都好
- $\mathcal{L}_{pix}^{phy}$ 单独提升更大(47.2 vs 46.2),因为它直接 suppress trajectory discontinuity,这是最常见的 local failure mode
- $\mathcal{L}_{sem}^{phy}$ 主要 repair global relational errors 比如 broken contact
- 两个 loss target 不同的 error modes,所以能 stack 起来

**Physics region focus ablation** (Table 5):

| Model | Emb. | Tasks | Avg. |
|-------|------|-------|------|
| baseline (ft) | 56.5 | 35.4 | 44.8 |
| w/o Physics region focus (uniform) | 57.0 | 37.2 | 46.0 |
| w/ Physics region focus | 58.2 | 38.9 | 47.5 |

关键 takeaway:
- 即使 uniform supervision 也有帮助 (44.8 → 46.0),说明 physics loss 本身有价值
- 但 region focus 进一步提升到 47.5,task 维度提升最显著 (35.4 → 38.9)
- 这证实了 paper 的核心论点:background 和 near-static regions 会 dilute physical signal,focus 在 interaction-critical regions 才能 drive task-level correctness

**Alignment layer ablation** (Table 6):

| Layer | 10 | 15 | 20 | 25 |
|-------|-----|-----|-----|-----|
| Robot Domain Score | 83.9 | 85.2 | 84.1 | 83.2 |

Layer 15 (middle) 最好。论文解释:
- Early blocks (layer 10): 还在 carry shallow appearance features,lacks semantic structure for relational alignment
- Late blocks (layer 25): 已经 specialized for final noise prediction,harder to steer
- Middle block (layer 15): 最佳 trade-off,既 rich semantic 又 flexible

这个 ablation 给我们一个 important intuition: **representation alignment 应该在 backbone 的 semantic-but-not-finalized 层做**。

**Training dynamics** (Figure 6):
- 在整个 training 过程中,both losses 都比 vanilla finetune 好
- Full model 始终领先,说明 persistent 而非 transient 的 learning signal
- 在 20k step 达到 peak 85.2 (+4.1),30k 时轻微 overfitting 但仍领先 +3.7

## 六、关键 Intuitions 和延伸思考

### 6.1 为什么 region-focused supervision 如此重要

这个 paper 的一个核心 insight: **physical evidence 高度 localized**。传统的 reconstruction loss 对所有 pixel 同等对待,background 占了 90%+ 的 pixel,所以 loss 信号被 dilute 了。通过 region mask 把 supervision focus 到 interaction-critical regions,等于把 signal-to-noise ratio 大幅提升。

这让我想到类似的思路:
- **Detection 里的 focal loss**: 通过 focusing parameter $\gamma$ 把 loss 集中在 hard examples
- **Segmentation 里的 class imbalance handling**: 不让 background class 主导 loss
- **Attention mechanism 本质上**: 也是在做 region-focused computation

### 6.2 为什么 pixel + semantic 是 hierarchical 互补的

这是 paper 的另一个核心 insight。让我用你熟悉的 deep learning intuition 来解释:

- **Pixel-level trajectory alignment**: 监督 **low-level geometry**,类似 supervising optical flow 或 point tracks。它 catch 的是 trajectory continuity、contact-compatible displacement 这种 per-point 物理性
- **Semantic-level relational alignment**: 监督 **high-level interaction semantics**,类似 contrastive learning 里的 relation structure。它 catch 的是 "grasped object 应该 coupled with gripper"、"pushed object 应该 joint move" 这种 global relational 物理性

两者是 hierarchical 的,因为:
- Pixel-level 关心 individual points 的 motion trajectory
- Semantic-level 关心 regions 之间的关系 structure
- 组合起来 cover 了 physical plausibility 的两个 level

这也呼应了 deep learning 里的 multi-scale supervision 思路,但 here 是 spatial-semantic 的 hierarchy 而不是 spatial 的 multi-scale。

### 6.3 与 Representation Alignment 工作的关联

Semantic-level alignment 的设计跟最近的一些 representation alignment 工作很像:
- **VideoRePA** [55]: Learning physics for video generation through relational alignment with foundation models
- **DINOv3** [41]: self-supervised representation learning
- **V-JEPA 2** [3]: 通过 masked spatio-temporal feature prediction 学 representation,自然 capture object- 和 interaction-centric structure

这个 alignment 思路本质上是 **knowledge distillation** 的一个变种 —— 不是 distill absolute features,而是 distill relational structure。这跟 BYOL、SimSiam 之类的 self-supervised learning 里的 "predictive learning" 有异曲同工之妙。

### 6.4 MoE backbone 的 routing 修改很有意思

Wan2.2-I2V-A14B 的 fine-tuning 策略值得 highlight:
- 原始 MoE 设计:high-noise expert 处理 $t \geq t_{moe}$,low-noise expert 处理 $t < t_{moe}$
- PhysisForcing 修改: fine-tune 时 high-noise expert 在 full $t \in [0, T]$ range 上 work
- intuition: physical structure 在 dynamics-forming stage 被确定,所以 fine-tune high-noise expert 最 effective

这个 routing 修改的 motivation 是 physical structure 在 high-noise stage commit,但实际上 high-noise expert 跨过 boundary 工作,可能让它在 low-noise region 也学会 obey physics constraints。这是一个很 practical 的工程决策。

### 6.5 Training-time vs Post-hoc 的本质区别

PhysisForcing vs preference-based methods (DPO, GRPO) 的本质区别:

- **Post-hoc (ABot-PhysWorld, MIND-V)**: 在已经训练好的 model 上做 alignment,本质是 correcting 而不是 preventing。可能牺牲 visual quality 来满足 physics constraints。
- **Training-time (PhysisForcing)**: 在 fine-tuning 阶段直接 inject physics supervision,让 model 从一开始就 learn physically consistent representations。不需要 trade-off visual quality 和 physics。

这个区别决定了 PhysisForcing 不会损害 Quality Score(看 Table 7 和 Table 8,Quality Score 保持 competitive),而 post-hoc 方法可能在 Domain Score 提升的同时 Quality Score 下降。

### 6.6 Policy Learning 的传导效应

最让我印象深刻的:physical plausibility 提升不仅改善 generation,还 **传导到 downstream policy**。Table 2 显示在 contact-rich tasks 上 policy success rate 显著提升。这背后的 intuition:
- Video backbone 作为 world action model 的 video DiT
- 如果 backbone 学到的 representations 是 physically consistent 的,那 policy 从这些 representations 里 extracted 的 action 也是更 physically meaningful 的
- 在 place_empty_cup 这种需要精确 contact 的 task 上提升 21.5%,说明 physical priors 在 contact-rich scenario 价值最大

这给我们一个重要启示: **为 robotics 训练 video generation model 时,physical plausibility 不仅是 generation quality 问题,是 downstream policy 性能问题**。

## 七、Limitations 和未来方向

paper 自己承认:
- PhysisForcing 是 fine-tuning recipe,inherit backbone 的 capability ceiling
- 当前 open-source video generator (Wan2.2, Cosmos3) 在 world knowledge 和 long-horizon temporal reasoning 上仍然有限
- 随着 stronger video foundation models 出现,PhysisForcing 的 trajectory-level 和 relational physics constraints 会 compound with backbone 能力

我个人的延伸思考:
1. **3D 一致性**: 当前只用 2D trajectory supervision,如果能 extend 到 3D trajectory(用 multi-view 或 learned 3D tracking),可能 capture 更 rich 的 physical structure
2. **Long-horizon**: 2-frame trajectory alignment 可能不足以 capture long-horizon causal relations,需要 temporal hierarchical supervision
3. **Real physics engine integration**: simulator-based methods [46, 49] 用 physics engine 保证 validity 但 overhead 大,可能可以 hybrid —— physics engine 提供 sparse supervision signal,PhysisForcing 提供 dense feature alignment
4. **Action-conditioned generation**: 当前 method 是 unconditional/video-conditional,在 action-conditioned world model 里 physics supervision 应该更自然
5. **Multi-modal supervision**: video encoder 只是 one source of relational structure,如果能 combine language model (描述 interaction semantics) 或者 tactile model (contact force),会更 comprehensive

## 八、Reference Links

- Paper: https://dagroup-pku.github.io/PhysisForcing.github.io/
- Wan: https://arxiv.org/abs/2503.20314
- Cosmos 3: https://arxiv.org/abs/2606.02800
- CoTracker3: https://arxiv.org/abs/2410.11831
- V-JEPA 2: https://arxiv.org/abs/2404.08471
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- ABot-PhysWorld: https://arxiv.org/abs/2603.23376
- VideoRePA: https://arxiv.org/abs/2505.23656
- DINOv3: https://arxiv.org/abs/2508.10104
- WorldArena: https://arxiv.org/abs/2602.08971
- Fast-WAM: https://arxiv.org/abs/2603.16666
- R-Bench: https://arxiv.org/abs/2601.15282
- PAI-Bench: https://arxiv.org/abs/2512.01989
- RoboScape: https://arxiv.org/abs/2506.23135
- CTRL-World: https://arxiv.org/abs/2510.10125
- TesserAct: https://arxiv.org/abs/2504.20995
- MIND-V: https://arxiv.org/abs/2512.06628
- DreamGen: https://arxiv.org/abs/2505.12705
- Vidar: https://arxiv.org/abs/2507.12898

## 九、总结

PhysisForcing 的核心贡献是把 physical plausibility 在 robotic video generation 里 formulate 成 hierarchical + region-focused 的 alignment 问题,然后通过两个 complementary loss 实现:

1. **Pixel-level trajectory alignment**: 用 CoTracker3 的 reference trajectories 通过 soft argmax attention mechanism 监督 DiT feature 的 per-point motion consistency
2. **Semantic-level relational alignment**: 用 frozen V-JEPA 2 作为 measurement space,通过 cosine similarity matrix alignment 监督 inter-region 的 relation structure

整个 framework 是 training-time only,inference zero overhead,这在工程上非常 attractive。实验结果在 R-Bench, PAI-Bench, EZS-Bench 三个 benchmark 上都拿到 best overall(PF-Cosmos),而且在 WorldArena 闭环 success rate 和 downstream policy success 上都有显著提升,证明 physical plausibility 对 embodied intelligence 有 concrete 价值。

核心 intuition 我觉得是两点:(1) **hierarchical supervision** —— pixel 和 semantic 两个 level cover 不同 error modes;(2) **region focus** —— 把 supervision 集中在 interaction-critical regions 而不是 uniform over all pixels。这两个设计原则其实超越了 robotics video generation,可以 generalize 到其他需要 fine-grained physical / structural consistency 的 generation 任务上。

希望这个讲解帮你 build 出 paper 的 mental model,Andrej。如果你对某个 specific component 想深入讨论,比如 soft argmax 的 gradient flow、MoE routing 修改的影响、或者 semantic alignment 跟 V-JEPA 2 self-supervised objective 的深层 connection,我可以进一步展开。
