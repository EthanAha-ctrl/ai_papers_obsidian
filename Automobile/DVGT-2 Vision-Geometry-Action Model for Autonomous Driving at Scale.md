---
source_pdf: DVGT-2 Vision-Geometry-Action Model for Autonomous Driving at Scale.pdf
paper_sha256: 65275a1e768523bc52e5d5a3143dd4d5811ca8880564394f4008ec6bce3d44ca
processed_at: '2026-08-18T07:17:10-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这帮人想让自动驾驶直接从摄像头图片重建出周围世界的 3D 点云，然后基于这个 3D 点云来规划路线，搞了个能实时跑的 streaming 架构。

自动驾驶圈现在有三派在吵架，争的是"中间表示"该用啥。所谓中间表示，就是摄像头看到图片之后，先把它翻译成什么东西，再拿这个翻译结果去做驾驶决策。

**第一派（传统派）**：把图片翻译成"前方有辆车在 30 米，左边有车道线，右前方有个行人"。就是 detection + tracking + mapping 那一套。问题是这个翻译太 lossy——你只保留了几个 box 和线，世界本身的丰富信息全丢了。

**第二派（VLA 派）**：用大语言模型，把图片翻译成自然语言描述，比如"前方道路宽阔，左侧有施工，建议减速"。优点是能 reasoning，缺点是自然语言太粗糙——你说"前方有辆车"，但车到底 29 米还是 31 米？language 表达不了 sub-meter 精度。

**第三派（VGA 派，就是这篇 paper）**：把图片直接翻译成 dense 3D point cloud，每个像素对应一个 3D 点。他们的论点很直接：车在 3D 世界里跑，那最完整的表示就是 3D 几何本身，干嘛要绕一圈用语言或 box？

---

## 前作 DVGT 的痛点

前作 DVGT 已经证明了"用 dense 3D 点云当中间表示"这个思路能 work。但它有个致命问题：**太慢了，没法实时跑**。

为什么慢？因为它是 batch processing——每来一帧新画面，它要把过去 N 帧全部重新跑一遍 transformer。比如现在到了第 10 帧，它要把第 1-10 帧一起塞进模型算；到了第 11 帧，又把 2-11 帧一起算。第 10 帧的 features 在算第 11 帧时被重算了一遍，在第 12 帧时又被重算一遍……大量重复计算。

实测下来 16 帧 8 视角要 1.88 秒一帧。自动驾驶 10Hz 跑，你 1 秒才算一帧，车早撞了。

---

## DVGT-2 的核心招数

DVGT-2 要解决的就是"怎么让 streaming 推理又快又准"。它用了三个互相耦合的 trick，这三个 trick 缺一不可。

### Trick 1：滑动窗口缓存

最朴素的想法是：只保留最近 W 帧的 features 在缓存里，超过 W 就丢掉最老的。这样每帧计算量固定，缓存大小也固定。

但这里有坑。前作 StreamVGGT 也用了缓存，但它的缓存会随时间线性增长——因为它把所有历史帧都留着。DVGT-2 干脆把缓存砍成固定 W=4 帧，FIFO 更新，老的丢了新的进来。

这样每帧计算量恒定，内存恒定，可以无限长 streaming。

### Trick 2：本地坐标系

但 Trick 1 单独用会崩。因为前作的坐标系是 anchor 到第一帧的——所有 3D 点都在"第 0 帧的位置"为原点的坐标系里。你把第 0 帧的 features 丢了，整个坐标系就漂了，后面的点云全乱套。

DVGT-2 的解法很巧妙：**每一帧都预测自己坐标系下的点云**。第 10 帧的点云在"第 10 帧车头位置"为原点的坐标系里，第 11 帧的点云在"第 11 帧车头位置"为原点的坐标系里。然后另外预测一个"第 11 帧相对于第 10 帧的位移和旋转"。

这样缓存里存的 features 就只描述"我那一刻周围的局部几何"，跟全局坐标系无关。你丢掉哪帧的缓存都不影响其他帧的表示。

需要全局点云时，把每帧的 relative pose 串联起来，chain transformation 累积到全局。代价是有累积误差——这是 DVGT-2 在全局位姿指标上不如前作的原因。

### Trick 3：相对时间位置编码

这个 trick 最 subtle 但最关键。

Transformer 需要 positional encoding 告诉模型"我是第几帧"。如果用 absolute encoding，缓存里第 5 帧的 feature 编码了"我是第 5 帧"。下一帧推理时它变成第 4 帧（因为更老的被丢了），encoding 要改，缓存就废了得重算。

DVGT-2 用 relative encoding（MRoPE-I），让 feature 表示"我相对于当前帧差几帧"。这样缓存不动就永远是对的，可以无限复用。

---

## 三个 trick 的耦合关系

这三个 trick 是绑定的：
- 没有 Trick 2，缓存丢了第一帧就坐标系崩溃
- 没有 Trick 3，缓存每次滑动都要重算 positional encoding
- 没有 Trick 1，缓存无限增长还是会 OOM

论文里 Fig. 3 把这个对比画得很清楚：batch 是 O(T²)，full streaming 是 O(T)，他们的 sliding window 是 O(W) 常量。

---

## 架构长啥样

整体流程：

```
多视角图片 → DINOv3 ViT-L 提特征
         → 加上 learnable pose token 和 trajectory token
         → 进 24 层 Geometry Transformer
            每层做三种 attention：
            - 图内 attention（单图细节）
            - 跨视角 attention（8 个摄像头互相对齐）
            - 时序 attention（当前帧 query 历史 4 帧缓存）
         → 三个 head 输出：
            - DPT head 出 dense 3D 点云
            - anchor diffusion head 出 relative ego-pose
            - anchor diffusion head 出未来轨迹
```

总参数 1.8B，训练用了 64 张 H20 跑 10 天。

轨迹 head 用了 DiffusionDrive 的 anchor-based diffusion——预先 cluster 出 20 个典型轨迹模板，diffusion 从模板出发做 truncated denoising。好处是 driving 轨迹是多模态分布（同一场景可能"绕左"或"绕右"都合理），diffusion 能建模这个分布。

---

## 实验结果的故事

### 几何重建

DVGT-2 在局部几何（ray depth）指标上全面 SOTA。这有点反直觉——它明明是 streaming 又是 local coordinate，怎么比 batch + global 还准？

原因正是 local coordinate 的好处：每帧预测自己周围的几何，没有累积位姿误差。而 batch 方法虽然能看全局，但全局坐标系下的点云要靠网络一次性学出来，反而难。

全局点云指标上 DVGT-2 略弱于前作 DVGT——这是累积位姿误差的代价。但这个代价在 driving 任务里可以接受，因为 driving 决策主要看当前帧周围几何。

### 速度

最惊艳的数字：**0.27 秒一帧**，常量内存，可以跑几百帧不崩。前作 DVGT 10 帧 OOM，StreamVGGT 30 帧 OOM。

### 规划

NAVSIM v2 closed-loop 拿到 89.6 EPDMS，超过所有 SOTA。NAVSIM v1 拿到 90.3（fine-tune 版）。nuScenes open-loop collision rate 0.19%，比专门拿 collision 当训练目标的方法还低。

这说明 dense 3D 几何确实让模型隐式学到了"车和环境的物理关系"——不需要显式标"这里会撞"。

### 跨数据集泛化

在 OpenScene/nuScenes/Waymo 上 planning 误差 0.2-0.8 米，但在 KITTI 和 DDAD 上退化到 2 米。原因是 KITTI/DDAD 是高速驾驶场景，轨迹分布跟 OpenScene（训练主体）差太远，anchor-based diffusion head 被 OpenScene 的分布带偏了。

---

## 为什么这个工作有意思

对你（Andrej）来说，这个工作的哲学跟你在 nanoGPT / Eureka Labs 一直强调的思路一致：**别手工设计中间表示，让模型自己学**。

传统自动驾驶把 pipeline 拆成 detection → tracking → prediction → planning，每一步都是人设计的 lossy compression。VGA 说：别拆了，让 transformer 直接从 pixel 学到 metric 3D world，再从这个 world 学 action。dense point cloud 就是"模型的内部表示"，它同时 encode 了 obstacle、free space、static background、dynamic object velocity——所有信息都在里面。

而 streaming 架构解决的是"怎么让这个 representation 在 bounded memory 下持续运行"——跟你讲 attention 时说的"bounded context window"问题同构。他们的答案：local coordinate + relative position encoding，让缓存表示时间不变。

---

## 局限和值得想的方向

1. **全局位姿漂移**：累积 relative pose 会 drift，长序列全局一致性会崩。可能需要 periodic global re-anchoring，类似 SLAM 的 loop closure。

2. **Anchor 分布偏置**：anchor 在 OpenScene 上 cluster，到高速数据集就不好使。可能需要 dataset-aware anchor 或者干脆换成 continuous diffusion。

3. **缺 semantic**：纯几何分不清消防车和卡车，某些特殊场景决策可能吃亏。VGA + VLA 混合可能是未来方向——geometry 给 precision，language 给 reasoning。

4. **MoGe-2 当监督的 dependency**：几何监督来自另一个 foundation model，继承它的 failure mode。如果 MoGe-2 在某种场景（比如夜间、雨雪）不好，DVGT-2 也不好。

5. **跟 world model 的结合**：现在 DVGT-2 是"看当前帧预测当前几何 + 未来轨迹"。如果改成"预测下一帧的几何"，就变成 geometry-first world model 了。DriveVLA-W0 在 NAVSIM 上已经显示 future states 路线很强，DVGT-3 往这个方向走会很自然。

---

## 给你的 takeaway

这个工作本质上是把"3D 几何当自动驾驶的 universal interface"这个理念工程化了。前作证明了理念能 work，这篇证明了能实时跑。三派中间表示之争——sparse perception、language、dense geometry——现在 dense geometry 这派有了能上车的系统。

如果你要在这个基础上做下一步，我会赌两个方向：一是把 trajectory prediction 改成 future pointmap prediction（world model 化），二是把 geometry 和 language 双 path 融合（VGA + VLA）。前者是 representation 升级，后者是 capability 补全。

参考链接还是那些：
- https://wzzheng.net/DVGT-2
- https://github.com/wzzheng/DVGT
- https://arxiv.org/abs/2512.16919（前作 DVGT）

---

# DVGT-2: Vision-Geometry-Action Model 深度解析

 Andrej，这篇 paper 来自清华 + Xiaomi EV + 澳门大学的团队，第一作者 Sicheng Zuo, Zixun Xie, Wenzhao Zheng 共同一作。它是 DVGT 系列的第二代，把 driving 视觉几何从 batch processing 推进到 streaming online inference，并提出了一个新的 paradigm——**VGA (Vision-Geometry-Action)**，对标 VLA。我下面尽量把直觉和工程细节都讲透。

---

## 1. Paradigm 战争：为什么是 VGA 而不是 VLA

Paper 把当前 end-to-end autonomous driving 划成三派，对应 Fig. 2：

| Paradigm | 中间表示 | 监督信号 | 代表方法 |
|---|---|---|---|
| Sparse Perception (Conventional E2E) | bounding boxes, map elements, occupancy voxels | 人工标注的感知任务 | UniAD, VAD, DiffusionDrive |
| VLA | natural language descriptions | language labels + RL | EMMa, Doe-1, AutoVLA, ReCogDrive |
| **VGA (本文)** | dense 3D pointmaps + ego-poses | 几何监督（depth + pose） | DVGT, DVGT-2 |

VGA 的核心论点：vehicle 在 3D 世界里运作，dense metric 3D geometry 是最直接、最完整的信息载体。Language description 太 coarse——你写 "前方有一辆红车在 30 米处左转"，丢掉了 sub-meter 级别几何精度；而 sparse perception 又把世界 information 压成 bounding box，丢掉了 background context 与 free-space geometry。

VGA 的输出形式（公式 4）：

$$
\mathbf{A}_t, \mathbf{P}_{t-T:t}, \mathbf{E}_{t-T:t} = \mathcal{M}_{\text{VGA}}(\mathbf{I}_{t-T:t})
$$

变量含义：
- $\mathbf{I}_{t-T:t}$：从过去第 $t-T$ 帧到当前 $t$ 帧的 multi-view image sequence
- $\mathbf{P}_{t-T:t}$：dense 3D pointmaps，每个像素对应一个 3D 点
- $\mathbf{E}_{t-T:t}$：ego-poses（translation + quaternion）
- $\mathbf{A}_t$：未来 N 步的 ego trajectory（x, y, yaw）

两个 fundamental advantages：
1. **Continuous coordinate space**——pointmap 是 pixel-aligned 连续值，没有 voxel discretization 的 quantization error
2. **Temporal coherence**——multi-frame geometry + pose 一起建模，static structures 和 dynamic motions 都能 capture

参考链接：
- 前作 DVGT: https://arxiv.org/abs/2512.16919
- VGGT (general geometry foundation): https://arxiv.org/abs/2507.13347 是 π³，VGGT 在 https://vgg-t.github.io/
- DUSt3R: https://dust3r.europe.naver.com/

---

## 2. 核心痛点：为什么 DVGT 不能直接 online inference

前作 DVGT 用 batch processing paradigm：

$$
\mathbf{P}_{t-T:t}, \mathbf{E}_{t-T:t} = \mathcal{G}_{\text{batch}}(\mathbf{I}_{t-T:t})
$$

问题在于：当 online 推理时，每来一帧 $t$，模型把整个 sequence $[t-T, t]$ 重新跑一遍，过去 $T-1$ 帧的 features 被反复计算。计算复杂度 $\mathcal{O}(T^2)$，且 frame $t$ 跑完后，frame $t+1$ 又要重新算 frame $t$ 的 features。Table 1 显示 DVGT 在 16 帧 8 视图上要 ~2.28s/frame——根本没法做实时 driving。

StreamVGGT 想缓解这个问题，引入 full-history streaming（公式 6）：

$$
\mathbf{P}_t, \mathbf{E}_t, \mathbf{C}_{t-T:t} = \mathcal{G}_{\text{stream}}([\mathbf{I}_t, \mathbf{C}_{t-T:t-1}])
$$

复杂度从 $\mathcal{O}(T^2)$ 降到 $\mathcal{O}(T)$——但 cache 大小随时间 linearly 增长。Fig. 7 显示 StreamVGGT 在 ~30 帧 OOM。

DVGT-2 的目标是 $\mathcal{O}(W)$ constant per-frame cost，$W$ 是 fixed window size（实验里 $W=4$）。

---

## 3. Sliding-Window Streaming：核心 trick 详解

DVGT-2 的公式 7：

$$
\mathbf{P}_t, \mathbf{E}_t, \mathbf{C}_{t-W+1:t} = \mathcal{G}_{\text{window}}([\mathbf{I}_t, \mathbf{C}_{t-W:t-1}])
$$

要理解这个公式背后的工程创新，要拆成三个 sub-problem：

### 3.1 为什么不能直接套 StreamVGGT 的 cache + 限长

如果你只把 cache 砍成 $W$ 大小，模型就崩了。因为 StreamVGGT 的 cache 里存的是 **global coordinate system anchored to first frame** 的 features——一旦你丢掉 first frame 的 features，整个 coordinate frame 就漂了。

### 3.2 DVGT-2 的解法：解耦 local geometry + relative pose

核心 insight：把预测目标从 "global pointmap in frame 0's coordinate" 改成：
- $\mathbf{P}_t$：在**当前帧 $t$ 的 ego coordinate** 下的 local pointmap
- $\mathbf{E}_t$：**当前帧相对于前一帧 $t-1$** 的 relative ego-pose（7D = 3D translation + 4D quaternion）

这样 cache 里的历史 features 就不再需要 anchor 到某个特定的 global frame——它们只需要表达"相对于它们各自时刻的 local 几何"，时序 attention 只是聚合 temporal context，不需要做全局 coordinate alignment。

global pointmap 重建时通过累积 relative poses 来 chain transformation：
$$
\mathbf{P}^{\text{global}}_t = \mathbf{T}_{t \to 0} \cdot \mathbf{P}_t, \quad \mathbf{T}_{t \to 0} = \prod_{k=1}^{t} \mathbf{T}_{k \to k-1}
$$

这个累积会引入 drift——DVGT-2 在 global pose 上的劣势主要来源于此（见 Section 4.4 的讨论）。

### 3.3 MRoPE-I：让 cache features 时间不变

如果你用 absolute temporal positional encoding，cache 里第 $t-W$ 帧的 feature 编码了 "我是第 $t-W$ 帧"。下一帧 $t+1$ 推理时，原本的 $t-W$ 帧会被丢弃，原来的 $t-W+1$ 帧现在变成"最早"——它的 absolute position 变了，cache 必须重新计算 positional encoding，cache 的复用就崩了。

DVGT-2 用 **MRoPE-I** ([17], arxiv.org/abs/2510.23095) 做 relative temporal positional encoding。Relative encoding 的好处：cache feature 表示"我相对于当前 query 帧的位置"，cache 不动它就永远是正确的 representation，可以无限复用。

这是 sliding-window streaming 能 work 的关键数学基础。

### 3.4 FIFO cache 更新（公式 13）

$$
\mathbf{C}_{t-W+1:t} = \text{FIFO}(\mathbf{C}_{t-W:t-1}, \hat{\mathbf{G}}_t)
$$

$\hat{\mathbf{G}}_t$ 是 geometry transformer 每层的 intermediate features。FIFO 即丢掉 $\mathbf{C}_{t-W}$，append $\hat{\mathbf{G}}_t$。每层 transformer 都维护自己的 cache，所以 $L=24$ 层 × $W=4$ 帧 × per-frame feature dim 构成总 cache size。

---

## 4. DVGT-2 架构详解（对应 Fig. 4, 5）

### 4.1 Overall pipeline

```
Multi-view images I_t (V × H × W × 3)
        ↓
  DINOv3 ViT-L encoder E   (公式 9)
        ↓ F_t^vis (visual tokens)
   + learnable pose tokens F_t^pose
   + learnable trajectory tokens F_t^traj (8 个/frame/view)
        ↓ F_t = [F_t^vis, F_t^pose, F_t^traj] (公式 10)
        ↓
  Geometry Transformer G (24 blocks, 1024 dim, 16 heads)
      × historical cache C_{t-W:t-1}   (公式 11)
        ↓
  G_t^vis, G_t^pose, G_t^traj
        ↓
  H^vis (DPT head)        → P_t (3D pointmaps, V × H × W × 3)
  H^pose (anchor diff.)   → E_t (7D relative ego-pose)
  H^traj (anchor diff.)   → A_t (N × 3 future trajectory)
```

总参数量 ~1.8B。

### 4.2 Geometry Transformer 的 factorized attention

每个 block 三步 sequential attention（继承自 DVGT）：

1. **Intra-View Local Attention**：单张图内部 token interaction，capture fine-grained 局部结构
2. **Cross-View Spatial Attention**：当前帧 $V$ 个 views 之间做 spatial reasoning，建立 surround view 的几何一致
3. **Temporal Causal Attention**：当前帧 tokens 作 query，cache 中 $W$ 帧的 features 作 keys/values，做 temporal aggregation

这种 factorization 把朴素 $\mathcal{O}((VT)^2)$ 复杂度降到 $\mathcal{O}(V^2 T + VTW)$。

### 4.3 Prediction heads 细节

**Visual head**: DPT (Dense Prediction Transformer) head，从 24 层 transformer 的第 4, 11, 17, 23 层取 intermediate tokens 做 multi-scale fusion，输出 dense 3D pointmap $\mathbf{P}_t \in \mathbb{R}^{V \times H \times W \times 3}$。

**Pose / Trajectory heads**: 基于 DiffusionDrive ([41], arxiv.org/abs/2501.17145) 的 anchor-based truncated diffusion。机制是：
1. 预先在 training set 上 cluster 出 20 个 anchors（典型 pose / trajectory 模板）
2. Diffusion process 从 anchor 出发，做 truncated（2 步而非 full steps）denoising
3. 每个 head 由 4 个 self-attention layers（inter-frame interaction）+ 2 个 cross-attention layers（diffusion decoding）组成
4. Trajectory token 还融合了 ego status（velocity, acceleration, driving command）通过 MLP

为什么用 anchor-based diffusion 而不是直接 regression？因为 driving trajectory 是 multi-modal distribution——同一场景可能有"绕左"和"绕右"两个合理选项。Diffusion 能建模这个分布。

### 4.4 训练 stability tricks

- QKNorm ([13])：query/key normalization，防止 attention 爆炸
- LayerScale ([64])：初始值 0.01，让残差路径在初期 dominate，稳定 deep transformer 训练
- Gradient clipping threshold 1.0
- bfloat16 + gradient checkpointing 省显存

---

## 5. 训练流程：两阶段 + NAVSIM fine-tune

**Stage 1: Geometry pretraining**（160K iterations）
- 不开 streaming mechanism
- 只做 dense geometry reconstruction supervision
- 在混合 dataset 上训：nuScenes : OpenScene : Waymo : KITTI : DDAD = 6:77:6:5:6

**Stage 2: VGA training**（80K iterations）
- 开 streaming mechanism
- 加入 trajectory planning supervision
- 同样在混合 dataset 上

**Stage 3: NAVSIM fine-tune**（40K iterations）
- 得到 specialized DVGT-2-NAVSIM
- 固定 8 views + 4 frames，aspect ratio 1.6，batch size 1

数据增强很激进：color jittering, Gaussian blur, grayscale conversion——per-frame 独立做，让模型对 lighting 鲁棒。

**几何监督怎么来？** 跟前作 DVGT 一样，用 depth foundation model **MoGe-2** ([68], arxiv.org/abs/2507.02546) 推 dense depth map，然后 thresholding filter 掉低质量区域。这避免了完全依赖 sparse LiDAR supervision。

训练成本：64 张 H20 GPU × 10 天。

---

## 6. 实验：geometry 重建 vs planning 全维度解析

### 6.1 Geometry reconstruction（Table 1-4, A.3）

OpenScene 上的关键数据：

| Method | Paradigm | Acc ↓ | Comp ↓ | Abs Rel ↓ | δ<1.25 ↑ | Time |
|---|---|---|---|---|---|---|
| VGGT* | Full-Seq | 1.705 | 1.711 | 0.280 | 0.669 | ~5.31s |
| DVGT | Full-Seq | 0.412 | 0.491 | 0.048 | 0.971 | ~1.88s |
| StreamVGGT* | Streaming | 2.209 | 2.060 | 0.303 | 0.620 | ~1.94s |
| Driv3r* | Streaming | 0.884 | 1.693 | 0.188 | 0.740 | ~0.56s |
| **DVGT-2** | **Streaming** | **0.440** | **0.450** | **0.040** | **0.977** | **~0.27s** |

`\*` 表示需要 sparse LiDAR 做 post-alignment 才能恢复 metric scale——DVGT-2 不需要。

关键观察：
1. **Ray depth (Abs Rel)** 在所有 dataset 都是 SOTA。Local pointmap 预测反而比 global 准——因为 model 直接预测当前 ego 系的 geometry，没有累积 pose 误差。
2. **Global point reconstruction (Acc)** 在 OpenScene/Waymo 超过 DVGT，但在 nuScenes/DDAD/KITTI 稍弱——累积 pose drift 的影响。
3. **Latency 0.27s/frame** 是 streaming 方法里最快，且常量。

### 6.2 Inference efficiency（Fig. 7）

- VGGT/DVGT: OOM at ~10 frames (O(T²) memory)
- StreamVGGT: OOM at ~30 frames (O(T) memory)
- DVGT-2: 稳定 ~260ms/frame + 常量 memory，可以无限长 streaming

这是工程上最大的胜利。

### 6.3 Closed-loop planning NAVSIM v1（Table 5）

| Method | Input | Aux Sup | PDMS ↑ |
|---|---|---|---|
| UniAD | C | Map & Box & Mot. & Occ | 83.4 |
| DiffusionDrive | C&L | Map & Box | 88.1 |
| DriveSuprim | C&L | Map & Box | 89.9 |
| AutoVLA | C | Language | 80.5 |
| ReCogDrive | C | Language | 86.8 |
| DriveVLA-W0 | C | Future States | 90.2 |
| AutoVLA† (RL) | C | Language & RL | 89.1 |
| **DVGT-2** | **C** | **Dense Geometry** | **88.6** |
| **DVGT-2-NAVSIM** | **C** | **Dense Geometry** | **90.3** |

DVGT-2-NAVSIM 在 NAVSIM v1 上 SOTA。注意：DVGT-2 只用 camera，不用 LiDAR；只用 dense geometry 监督，没有 language label，没有 RL。

### 6.4 Closed-loop NAVSIM v2（Table 6）

NAVSIM v2 引入 reactive traffic + extended metrics (TL, LK, HC, EC)。DVGT-2-NAVSIM 拿到 **89.6 EPDMS**，超过 DriveVLA-W0 的 86.1、DiffusionDrive 的 83.1、ARTEMIS 的 81.4。这是相当显著的领先。

### 6.5 Open-loop nuScenes（Table 7）

DVGT-2 L2 avg 0.78m，collision rate 0.19% avg。Collision rate 比 SOTA 显著更低——这特别值得注意，因为 nuScenes collision metric 不是 DVGT-2 的训练目标。模型隐式学到了 ego-vehicle 与 environment 的物理 interaction。

### 6.6 Ablation: Window size（Table A.1）

| W | Acc ↓ | Abs Rel ↓ |
|---|---|---|
| 2 | 0.613 | 0.042 |
| 4 | 0.480 | 0.042 |
| 6 | 0.474 | 0.042 |
| 8 | 0.501 | 0.042 |

- Abs Rel 完全不变 → window size 只影响 inter-frame / global geometry，不影响 local geometry
- Acc 在 $W=6$ 达到最优，$W=8$ 反而退化 → 因为更大的 window 让 accumulated pose drift 超过 temporal context 带来的收益
- 默认选 $W=4$ 是 latency / accuracy 的 sweet spot

### 6.7 跨 dataset planning 泛化（Table A.2）

OpenScene (训练主体) L2 0.20m；nuScenes 0.56m；Waymo 0.78m；KITTI 2.12m；DDAD 2.00m。KITTI/DDAD 退化严重，原因是 high-speed driving 的 trajectory 分布偏差，加上 anchor-based diffusion head 被 OpenScene dominant 分布偏置。这是 anchor-based 方法的固有局限。

---

## 7. Intuition 构建：为什么这套设计能 work

### 7.1 Dense geometry 作为 "universal interface"

传统 E2E 把 perception 拆成 detection + tracking + mapping，每个 task 都有人工设计的 supervision。问题是这些 task 之间是 "lossy compression"——detection 丢掉了 free space, mapping 丢掉了 object 速度。VGA 用 dense pointmap 这个 pixel-aligned 表示同时 encode 所有这些信息。

Planner 只需要看 pointmap 序列就能推出：哪里能走、哪里有 obstacle、static vs dynamic、relative velocity。

### 7.2 为什么 streaming + local coordinate 是必然组合

这两个 trick 是 coupled 的，不能单独用：
- 只用 local coordinate 不用 streaming：每帧都要从头跑 transformer，没有 temporal context
- 只用 streaming 不用 local coordinate：cache 依赖 global reference，无限长会崩

DVGT-2 的贡献就是看出这两个问题必须一起解。

### 7.3 Anchor-based diffusion + geometry 的 synergy

Anchor-based diffusion 给 trajectory 提供 multi-modal prior，dense geometry 给 trajectory 提供 obstacle-aware context。两者通过 shared transformer tokens 交互——trajectory tokens 在 transformer 里 attend 到 visual tokens，等于在做"geometry-conditioned trajectory generation"。

### 7.4 为什么 local geometry 反而更准（vs global）

直觉上 global 应该更难——因为要 chain 多个 relative pose。但 DVGT-2 在 ray depth (Abs Rel) 上 SOTA，说明 local 预测本身没有信息损失，loss 来自 pose accumulation。这意味着如果你只关心当前帧的 driving decision（这是 closed-loop 真正关心的），local geometry 完全够用。

### 7.5 VGA vs VLA 的本质区别

VLA 用自然语言作中间表示，优点是 generalization 强，缺点是 discrete + coarse。VGA 用 metric 3D pointmap 作中间表示，优点是 dense + continuous + 精确，缺点是缺少 high-level semantic reasoning（比如"前方有行人正在过马路"这种 scene understanding）。

Paper 没有直接对比 VGA + VLA hybrid 的可能，但这是显然的 next step——dense geometry + language reasoning 双 path。

---

## 8. 局限与开放问题

1. **Global pose drift**：累积 relative pose 导致 long-sequence global consistency 退化。未来可能需要 periodic global re-anchoring 或 loop closure。
2. **Anchor distribution bias**：在 dominant dataset (OpenScene) 上 cluster 的 anchors 在 high-speed dataset (KITTI/DDAD) 上不 work。需要 dataset-aware anchor 或 adaptive anchor。
3. **Window size 是 hyperparameter**：$W=4$ 是工程权衡，理论上应该自适应——比如根据 ego velocity 动态调整。
4. **缺少 semantic understanding**：纯几何不能区分"消防车"和"普通卡车"，可能影响 special scenario 的 decision。这跟 VLA 是互补而非替代关系。
5. **MoGe-2 监督的 dependency**：geometry 监督来自另一个 foundation model，意味着继承了 MoGe-2 的 failure mode。
6. **训练成本**：64 H20 × 10 天 = 15360 GPU-hours，对学术 lab 不友好。

---

## 9. 关键参考链接

- Paper 主页 / 代码: https://wzzheng.net/DVGT-2
- DVGT (前作): https://arxiv.org/abs/2512.16919
- Large Driving Models repo: https://github.com/wzzheng/LDM
- DVGT 系列代码: https://github.com/wzzheng/DVGT
- VGGT: https://vgg-t.github.io/
- π³: https://arxiv.org/abs/2507.13347
- DUSt3R: https://dust3r.europe.naver.com/
- StreamVGGT: https://arxiv.org/abs/2507.11539
- Driv3r: https://arxiv.org/abs/2412.06777
- CUT3R: https://cut3r.github.io/
- DiffusionDrive: https://arxiv.org/abs/2501.17145
- DINOv3: https://arxiv.org/abs/2508.10104
- MoGe-2: https://arxiv.org/abs/2507.02546
- MRoPE-I: https://arxiv.org/abs/2510.23095
- NAVSIM: https://github.com/autonomousvision/navsim
- nuScenes: https://www.nuscenes.org/
- OpenScene: https://github.com/OpenDriveLab/OpenScene
- Waymo Open Dataset: https://waymo.com/open/
- KITTI: http://www.cvlibs.net/datasets/kitti/
- DDAD: https://github.com/TRAILab/DDAD
- UniAD: https://github.com/OpenDriveLab/UniAD
- VAD: https://github.com/hustvl/VAD

---

## 10. 给你 (Andrej) 的 personal note

这个工作跟你最近在 Eureka Labs / 讲 micrograd / nanoGPT 系列时一直强调的 "let the model learn the representations, don't hand-engineer them" 思路高度一致。VGA 的核心论点其实就是：driving 不应该被人为拆成 detection / tracking / mapping / planning，而应该让 transformer 直接从 pixel 学到 metric 3D world，再从这个 world 学 action。

而 streaming + sliding window + local coordinate + relative pose 这套组合，本质上是解决 "online inference + constant memory" 的工程难题——跟你讲 attention 时常说的 "信息瓶颈" 思路一致：cache 就是 bounded context window，问题是当 window 滑动时怎么保持 representation 的 invariance，他们的答案是 MRoPE-I + local coordinate system。

我很想看到这个 model 跟 world model / video prediction 路线（Sora-style）的结合——dense pointmap 实际上就是一个 4D world model 的 implicit representation，未来如果 DVGT 系列演化成"预测下一帧的 pointmap + 自己的 trajectory"，就真正是 geometry-first world model for driving 了。DriveVLA-W0 ([34], arxiv.org/abs/2510.12796) 在 Table 5/6 里已经显示 "Future States" 路线很强，DVGT-3 如果把 trajectory 改成 conditional generation of future pointmaps，会很有意思。

希望对你 build intuition 有帮助。
