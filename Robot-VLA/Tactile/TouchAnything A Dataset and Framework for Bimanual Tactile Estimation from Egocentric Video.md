---
source_pdf: TouchAnything A Dataset and Framework for Bimanual Tactile Estimation
  from Egocentric Video.pdf
paper_sha256: 94afb180cd98238a1f10425dbe66878b0af325b81024e57793d287d6903f3d35
processed_at: '2026-08-12T16:37:08-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 TouchAnything

## 一句话总结

**用手上的摄像头看手怎么抓东西，然后猜出手掌跟物体之间压了多大力。**

## 为什么这事有意思

机器人搞 manipulation，最头疼的就是 "接触"。视觉能看到手在哪儿，看不到手跟物体之间那层接触面——压了没有、压多重、压在哪个指尖。真要装 tactile sensor，又贵又笨重，没法大规模部署。

所以他们就想了个办法：既然人有头戴摄像头，再给手腕上各绑一个小 fisheye 摄像头，这三路视频同步拍，再加上手套里的 pressure sensor 记录真实压力。训练一个 model，让它学 "看到画面 → 猜压力图"。训练好之后，推理时只要给视频就能出压力，不用真 tactile sensor。

## 关键 insight

**头戴摄像头看不到手心。** 你抓个杯子，从头戴视角看，手把杯子包住了，接触面全被手挡住。无论 model 多强，看不到就是看不到，这是信息缺失，不是模型不够好。

手腕上的 fisheye 摄像头刚好补上这个洞——它直接拍手心方向，接触面看得清清楚楚。

实验数据也印证：加了 wrist view，Contact IoU 从 0.48 涨到 0.50，Volumetric IoU 从 0.43 涨到 0.46。单加一个 wrist 就基本够了，加第二个边际收益很小，因为 fisheye 视野广，常常一只手的摄像头能顺带拍到另一只手。

## Dataset 怎么搭的

208 个 task、1891 个 episode、20 小时、210 万帧，室内室外都覆盖。每个 actor 戴一整套装备：
- 头上一个 RGB camera
- 两手腕各一个 fisheye RGB camera
- Rokoko 动捕手套记 42 个 joint 位置
- 自制 pressure glove，每只手 16×16 压力阵列
- 三个 HTC Vive Tracker 做全局定位

30Hz 同步，frame-level 对齐。

有个细节值得注意：手套上 256 个 sensor 物理上不是规整 16×16 排列，散布在手指和掌心不同位置。他们 remap 到 21×21 的 "手形网格"，让 model 输出直接是 hand-shaped pressure distribution，spatial prior 天然就在里面了。

## Model 怎么设计的

架构很直接，四步：

1. **Shared backbone**：三个 view 都过同一个 frozen DINOv2-ViT-B/14，加 view embedding 区分身份。共享参数省显存，DINOv2 的 visual prior 不用重学。

2. **Cross-view fusion**：每个 view 先 pool 出一个 summary token，在几个 summary 之间跑个轻量 transformer 交换信息。然后 gated fusion 加权求和——哪个 view 可信就给哪个 view 大权重。

3. **Pose-vision cross-attention**：这是最聪明的一步。42 个 hand joint 各自作为 query，visual patch 作为 key/value。意思是 "每个 joint 去找它对应的视觉区域"。右手食指指尖的 query 会 attend 到指尖附近的 visual patch，把视觉信息 attach 到那个 joint 上。这样 tactile prediction 就是 spatially grounded 的。

4. **Decoder**：42 个 joint feature 拆成左右手各 21 个，各自 MLP + sigmoid 出 21×21 pressure map。

## 训练 trick

两个关键 trick：

**View dropout**：训练时 ego view 永远保留，两个 wrist view 各以 0.3 概率随机 drop。这样 model 见过所有 4 种 view 组合，部署时给任何子集都能跑。Ablation 很震撼——不 dropout 的话，缺 wrist view 时性能掉 27%；dropout 后只掉 5.8%。

**Contact-region reweighting**：pressure map 极度 sparse，大部分像素是 0。不加权 model 直接学全 0 输出。所以 pressure > 0.1 的像素 loss 权重 ×3。这是 tactile 任务特有的——比 image segmentation 的 sparsity 严重得多。

## 结果怎么样

Overall，seen object 上 Volumetric IoU 0.43 → 0.46，unseen 上 0.37 → 0.39。提升有但不夸张。Office scenario 提升最大（+8.8%），因为打键盘、刷卡这类 precision 操作 contact region 小、occlusion 严重，wrist view 帮助最大。

Data scaling 没饱和——从 25% 数据加到 100%，性能一路涨。说明 vision-to-touch 还是 data-hungry，scale 还能继续推。

## Limitation

- 全部数据戴 glove 采集，model 可能学到 "手套外观 → 压力" 的 spurious correlation，bare-hand 泛化存疑
- 失败案例：黑色手套碰黑色短裤，视觉低对比度，model 误判有 contact。说明 model 高度依赖 appearance cue，缺 depth/segmentation 的 disambiguation
- 只做了 tactile estimation，没接下游 robot policy task

## 为什么这事值得关注

跟 LLM 的 scaling 思路一致：**大规模便宜 modality（vision）+ 少量 paired 昂贵 modality（tactile）**，训练一个 cross-modal predictor。如果这条路 work，internet-scale 的 ego video 都能被自动标注 tactile label，embodied foundation model 就能获得 "物理接地" 的 contact understanding，不用真的给每个 robot 装 tactile sensor。

跟 sim2real 也有关系——真 tactile sensor 在 sim 里能拿到 ground truth，real 上噪声大易坏。TouchAnything 是 "real2real tactile synthesis"，绕开 sensor 硬件，直接从 vision 生成 tactile supervision。这条路更 scalable，因为 human ego video 几乎无限。

Project page: <https://jianyi2004.github.io/TouchAnything-Website/>

---

# TouchAnything: 从 Egocentric Video 推断 Bimanual Tactile 信号

## 1. Paper 的核心 Motivation 与 Intuition

这篇 paper 来自 Harbin Institute of Technology (Shenzhen)、Meituan Academy of Robotics、Tsinghua Shenzhen International Graduate School，submission date 2026年5月14日。它要解决的核心问题非常直觉：**能不能从纯粹的视觉 observation 推断出人手与物体接触时的压力分布图 (pressure map)**，从而为大规模 egocentric video 数据提供 tactile supervision，而避免大规模部署昂贵的 tactile sensor 硬件。

Vision-to-touch prediction 的 bottleneck 在于 data 稀缺与 view occlusion。现有 dataset 比如 GRAB、ContactDB、ARCTIC、OakInk、DexYCB、ActionSense、EgoPressure、OpenTouch、EgoDex，各自缺少以下属性中的若干：
- **In-the-wild** 采集环境
- **Bimanual** 双手交互
- **Multi-view** 特别是 **wrist-mounted** 视角
- **Real dense pressure**（区别于 analytical 或 thermal 估算）
- 大规模 object 与 task diversity

EgoTouch 的贡献正是把上述所有属性同时拼齐：208 tasks、1891 episodes、2.1M frames、~1000 objects、head-mounted + 双 wrist-mounted RGB + bimanual 3D hand pose (42 joints) + 真实 pressure glove 的 16×16 array remap 到 21×21 hand-shaped grid。

Project page: <https://jianyi2004.github.io/TouchAnything-Website/>

---

## 2. 为什么 Wrist View 是关键 —— Occlusion 分析

Egocentric vision-to-touch 的根本困难在于 **palmar contact region 在 head-mounted camera 视角下经常被 hand 自身或 object 完全遮挡**。无论模型 capacity 多大，如果 contact interface 没有被任何一个 camera 直接观察到，那么 prediction 本质上是 ill-posed 的——只能从 hand pose、object 类别、motion 模式等 indirect cue 推断。

作者引入 **dual wrist-mounted fisheye camera**，直接拍摄手与物体接触界面。这给出了 complementary viewpoint，并在实验中验证：仅加一个 wrist view 就把 Contact IoU 从 0.4792 提到 0.5030（+5.0% relative），Volumetric IoU 从 0.4311 提到 0.4575（+6.1% relative）。加第二个 wrist view 提升已经很小，说明 **fisheye 单 wrist 已覆盖大部分 opposite-hand 的细节**。

这点对 build intuition 很重要：vision-to-touch 的瓶颈不是 "model 不够强"，而是 "input information 本身不够"。wrist view 把 missing evidence 补回来，prediction 才能从 ill-posed 变成 well-posed。

---

## 3. EgoTouch Dataset 细节

### 3.1 Hardware Stack

采集系统是一个 wearable 多模态同步套件：
- **Head-mounted RGB camera**：640×480×3，global egocentric context
- **Dual wrist-mounted fisheye RGB camera**：640×480×3，close-up contact
- **Rokoko motion-capture gloves**：21 joints/hand × 3D，bimanual 共 42 joints
- **Custom pressure-sensing gloves**：每只手 16×16 = 256 channel 8-bit 压力 vector + IMU quaternion
- **HTC Vive Trackers**：3 个 6-DoF tracker，分别绑在 chest、left_wrist、right_wrist，提供全局定位
- 软件 synchronization 通过 30Hz timer + latest-valid snapshot 实现

### 3.2 Tactile Grid Remapping（关键 preprocessing）

原始 pressure glove 的 256 个 sensor 物理上**不是规则的 16×16 image grid**——它们散布在手指、palm、bending location 等不同位置。直接当作 16×16 image 训练会丢失 hand 形状的 spatial prior。

作者用一个 hand-specific JSON 文件定义 mapping：每个 key 是 target grid 坐标 (r, c)，value 是 raw sensor index。这样得到一个 **21×21 hand-shaped sparse grid**，invalid 位置标记为 NaN。right hand 做水平 mirror，使左右手 share 同一个 canonical 坐标系。

预处理还包括：
- **Baseline subtraction**：若首帧被判定 contact-free（通过 manual annotation 或 low-pressure threshold），减掉首帧 baseline 消除 static sensor bias
- **Broken column repair**：右手 tactile grid 中已知损坏列通过相邻有效列插值修复
- **Separate normalization**：tactile sensor 与 bending sensor 分开 normalize，避免 bending 高值压缩 contact 动态范围

这个 remapping 步骤直觉上很重要：**模型输出直接是 hand-shaped pressure distribution**，而不是 arbitrary sensor vector，网络学到的 spatial pattern 直接对应物理 hand anatomy。

### 3.3 Task Taxonomy

208 tasks 分为 5 个 environment category：
- **Home**：开容器、推拉、按开关、擦表面、折衣服
- **Workbench**：工具操作（锯、钻、磨、切、夹、precision assembly）
- **Office**：刷卡片、打键盘、操作文具
- **Retail**：捏产品、按包装、折商品、开袋
- **Outdoor**：球类、球拍、户外动态交互

这种 diversity 让模型不能简单 memorize 单一 contact 模式，必须学到 cross-task、cross-object 的 generalizable vision-to-touch mapping。

---

## 4. TouchAnything 架构解析

架构分四块：**Shared Visual Encoder + View Embedding → Cross-View Attention + Gated Fusion → Temporal Transformer + Pose-Vision Cross-Attention → Joint-Level Tactile Decoder**。下面逐步拆解公式。

### 4.1 Shared Backbone with View Embedding

每个 view $V_v$ 通过 **frozen DINOv2-ViT-B/14** 编码，得到 $N=256$ 个 patch token，维度 $D=768$。加一个 learnable view embedding $\mathbf{e}_v \in \mathbb{R}^D$ 区分视角身份：

$$
\mathbf{F}_v = \text{DINOv2}(V_v) + \mathbf{e}_v, \quad \mathbf{F}_v \in \mathbb{R}^{T \times N \times D}
$$

- $V_v$：第 $v$ 个 view 的 RGB 帧（$v \in \{ego, wL, wR\}$）
- $T$：clip 长度（=8 frames）
- $N$：patch token 数量（224×224 输入，14×14 patch size，N=256）
- $D$：feature 维度（768）
- $\mathbf{e}_v$：view-specific learnable embedding，区分 ego/wL/wR

**共享 backbone 参数** 把参数量从 $3 \times 86M$ 降到 $86M + 3 \times 768$，efficiency 和 generalization 双赢。DINOv2 提供的 strong visual prior 是关键——避免从 scratch 学习 visual feature，而把训练 budget 集中在 tactile-specific 头部。

### 4.2 Cross-View Attention（轻量化设计）

直接对所有 $N \times |\mathcal{V}|$ patch token 做 cross-attention 计算量太大。作者先 **global average pooling** 得到每个 view 的 summary token，再在 $|\mathcal{V}|$ 个 summary token 上跑一个 lightweight transformer：

$$
\mathbf{s}_v = \text{MeanPool}(\mathbf{F}_v), \quad [\hat{\mathbf{s}}_1, \ldots, \hat{\mathbf{s}}_{|\mathcal{V}|}] = \text{CrossViewTransformer}([\mathbf{s}_1, \ldots, \mathbf{s}_{|\mathcal{V}|}])
$$

- $\mathbf{s}_v$：view $v$ 的 pooled summary，$\mathbb{R}^{T \times D}$
- CrossViewTransformer：standard multi-head self-attention 在 view 维度上跑

这一步让 wrist view 的 summary 信息能流到 ego view 的 summary 上，反之亦然，实现 "view-level 的互补信息交换"。计算成本只有 $O(|\mathcal{V}|^2 \cdot D)$，可忽略。

### 4.3 Gated View Fusion

不同 view 的可信度不同（比如 wrist view 可能被物体本身遮挡）。作者用 gating network 学 view-dependent weight：

$$
w_v = \text{softmax}(\text{MLP}(\hat{\mathbf{s}}_v)), \quad \mathbf{F}^{\text{fused}} = \sum_{v \in \mathcal{V}} w_v \cdot \mathbf{F}_v
$$

- $w_v$：view $v$ 的 fusion weight（softmax 归一化）
- $\mathbf{F}^{\text{fused}} \in \mathbb{R}^{T \times N \times D}$：fused feature，shape 与单 view 一致

Gating 的直觉是：当 wrist view 被 occluded 或不可靠时，weight 自动下降，让 ego view 主导；反之当 wrist view 直接看到 contact 时，weight 上升。这给 model 在 missing view 下 graceful degradation 的能力。

### 4.4 Temporal Transformer

Manipulation 是动态过程，temporal context 极重要（grasping → squeezing → releasing）。fused feature 跑一个 windowed temporal transformer：

$$
\mathbf{H} = \text{TemporalTransformer}(\mathbf{F}^{\text{fused}}), \quad \mathbf{H} \in \mathbb{R}^{T \times N \times D}
$$

### 4.5 Pose-Vision Cross-Attention Fusion

这一步是 architecture 的精髓。Bimanual hand pose $\mathbf{P} \in \mathbb{R}^{T \times 42 \times 3}$ 通过 pose encoder 得到 per-joint feature $\mathbf{G} \in \mathbb{R}^{T \times 42 \times D}$。然后每个 **joint token 作为 query**，**visual patch token 作为 key/value**，做 cross-attention：

$$
\mathbf{Z} = \text{CrossAttn}(Q=\mathbf{G}, K=\mathbf{H}, V=\mathbf{H}), \quad \mathbf{Z} \in \mathbb{R}^{T \times 42 \times D}
$$

- $Q = \mathbf{G}$：42 个 joint token，每个代表一个 hand joint
- $K = V = \mathbf{H}$：256 个 visual patch token
- 输出 $\mathbf{Z}$：每个 joint 对应的、与最相关 visual region 融合后的 feature

直觉上这是 **spatially grounded reasoning**：第 $i$ 个 joint（比如右手食指指尖）通过 attention 找到它对应的 visual patch（指尖附近的 RGB 区域），把视觉信息 attach 到该 joint。这比单纯 concatenate pose 与 visual feature 强很多——它显式建模了 "哪个 joint 看到了什么"。

### 4.6 Joint-Level Tactile Decoder

42 个 joint feature 拆成 left (1–21) 和 right (22–42) 两组，**每只手独立 decode 成 21×21 pressure map**：

$$
\hat{\mathbf{M}}_t^{left} = \sigma(\text{MLP}(\mathbf{Z}_t^{left})) \in [0,1]^{21 \times 21}, \quad \hat{\mathbf{M}}_t^{right} = \sigma(\text{MLP}(\mathbf{Z}_t^{right})) \in [0,1]^{21 \times 21}
$$

- $\sigma$：sigmoid，把输出压到 [0, 1]（normalized pressure）
- MLP：joint feature → 441-dim → reshape 成 21×21

注意这里 **每只手独立 decode**，bimanual coordination 主要在前面 fusion 与 temporal transformer 层面建模，decoder 层面是单手 task。这是合理的，因为 left/right pressure map 在物理上确实是独立量。

### 4.7 View Dropout Training

部署时 wrist camera 可能没有，所以训练时需要让 model 学会 "missing view 下也能工作"。策略：

- **Ego view 永远保留**
- 每个 wrist view 独立以 probability $p=0.3$ drop
- 训练时随机出现 4 种配置：Ego only / Ego+wL / Ego+wR / All three

这让 model 不依赖 fixed camera config，inference 时给任何 view subset 都能跑。这个 trick 在 multi-view learning 里类似 **DropPath / DropView**，但这里动机是 deployment robustness 而不仅是 regularization。

### 4.8 Training Objective

$$
\mathcal{L} = \lambda_{mse}\mathcal{L}_{MSE} + \lambda_{l1}\mathcal{L}_{L1} + \lambda_{tv}\mathcal{L}_{TV}(\hat{\mathbf{M}})
$$

- $\mathcal{L}_{MSE}$：pixel-wise mean squared error
- $\mathcal{L}_{L1}$：pixel-wise L1 loss
- $\mathcal{L}_{TV}$：total variation，鼓励 spatial smoothness
- 权重：$\lambda_{mse}=1.0$，$\lambda_{l1}=0.5$，$\lambda_{tv}=0.01$
- **Contact-region 加权**：pressure > 0.1 的像素 loss 权重 ×3.0，防止 model 学到 trivial all-zero prediction（因为 tactile map 极度 sparse，大部分像素是 0）

这个 reweighting 是 tactile 任务特有的 trick——vision-to-touch 的 sparsity 比 segmentation 严重得多，不加权 model 会塌缩到全 0 输出。

---

## 5. 评估 Metric 详解

### 5.1 Temporal Accuracy

只看每帧 "是否有任何 contact"（二值），prediction 与 ground truth 一致即正确。**不关心 contact 位置**，只关心 onset/offset 时刻。

### 5.2 Contact IoU

二值化 pressure map（threshold $\tau$）后算 IoU：

$$
\text{IoU}_{contact} = \frac{|M_{gt} \cap M_{pred}|}{|M_{gt} \cup M_{pred}|}
$$

不关心 pressure magnitude，只看 contact 区域 spatial 重合度。是 Volumetric IoU 的 upper bound。

### 5.3 Volumetric IoU

把 2D pressure image 转成 3D "pressure volume"——高度等于该像素压力值。然后算 volume-level IoU：

$$
\text{IoU}_{vol} = \frac{\sum_{i,j} \min(P_{i,j}, \hat{P}_{i,j})}{\sum_{i,j} \max(P_{i,j}, \hat{P}_{i,j})}
$$

- $P_{i,j}$：ground truth 在 pixel $(i,j)$ 的 normalized pressure
- $\hat{P}_{i,j}$：prediction 在 pixel $(i,j)$ 的 normalized pressure
- 分子：intersection（min of two values，取较小者求和）
- 分母：union（max of two values，取较大者求和）

这个 metric 同时考虑 spatial 位置和 pressure magnitude。如果 prediction 在正确位置但 magnitude 偏小，intersection 会被 min 截断，分母的 max 又会惩罚 underestimation。是 vision-to-touch 文献的标准 metric（来自 PressureVision）。

### 5.4 MAE

每个 pixel 的 mean absolute error，由于大部分像素是 0，MAE 天然很小，**主要看 non-trivial 区域**。

---

## 6. 主实验结果

Table 2 是核心结果，覆盖 5 个 environment scenario + Overall。

### 6.1 Overall 结果

| Method | T.Acc | C.IoU | V.IoU | MAE |
|---|---|---|---|---|
| **Seen Objects** | | | | |
| Ego-only | 0.8393 | 0.4792 | 0.4311 | 0.0456 |
| Ego + wL | 0.8567 ↑2.1% | 0.5030 ↑5.0% | 0.4575 ↑6.1% | 0.0437 ↓4.2% |
| Ego + wR | 0.8561 ↑2.0% | 0.5024 ↑4.8% | 0.4572 ↑6.1% | 0.0437 ↓4.2% |
| Ego + wL + wR | 0.8566 ↑2.1% | 0.5030 ↑5.0% | 0.4575 ↑6.1% | 0.0436 ↓4.4% |
| **Unseen Objects** | | | | |
| Ego-only | 0.8271 | 0.4396 | 0.3743 | 0.0615 |
| Ego + wL + wR | 0.8347 ↑0.9% | 0.4496 ↑2.3% | 0.3852 ↑2.9% | 0.0601 ↓2.3% |

**关键观察**：
1. **加 wrist view 主要提升 C.IoU 和 V.IoU**，对 T.Acc 提升小。直觉上：wrist view 帮 model 找到 contact 在哪里、压力多大，对 "是否 contact" 帮助小（这本来就靠 hand pose 大致能推断）。
2. **单 wrist view 已经接近双 wrist**：fisheye camera FOV 大，常常能拍到对侧手，所以加第二个 wrist 边际收益小。
3. **Unseen object 上提升更小**：模型在 unseen object 上更依赖 hand pose 等 object-agnostic cue，wrist view 的 object-specific 信息泛化性差一点。

### 6.2 各 Scenario 对比

- **Office**：C.IoU 0.4918 → 0.5256（+6.9%），提升最大。Office 任务多 precision 操作，contact region 小且容易被 hand 遮挡，wrist view 价值最高。
- **Retail**：T.Acc +5.2%，提升最大。Retail 多 dynamic 抓握、捏压，temporal contact onset 判定在 ego view 下困难（物体表面被手包住）。
- **Workbench**：Unseen object 上 T.Acc 反而下降 6.9%。Workbench 工具种类多，wrist view 在 unseen tool 上可能引入 spurious cue，这是 interesting 的 negative finding。
- **Outdoor**：seen object V.IoU 0.4769 → 0.5002，unseen 上 T.Acc 略降 1.2%。Outdoor 物体多样性极高（球、球拍等），wrist view 泛化困难。

---

## 7. Ablation Studies

### 7.1 View Dropout 的影响

| Training | Ego-only C.IoU / V.IoU / ΔV | Ego+wL/wR C.IoU / V.IoU / ΔV | All views C.IoU / V.IoU |
|---|---|---|---|
| No dropout | 0.3623 / 0.3233 / **-27.20%** | 0.4497 / 0.4073 / -8.29% | 0.4883 / 0.4441 |
| w/ dropout | 0.4792 / 0.4311 / **-5.78%** | 0.5027 / 0.4573 / -0.04% | 0.5030 / 0.4575 |

**没有 dropout 时**，model 高度依赖 wrist view，只给 ego view 时 V.IoU 掉 27.20%（catastrophic）。
**有 dropout 时**，model 学到 view-agnostic 的 robust representation，ego-only 掉幅只有 5.78%，且 all-view 性能与 no-dropout 持平甚至更好。

这个 ablation 强烈说明：**multi-view model 必须 view dropout 训练**才能在 deployment 时 graceful degrade。否则 model 隐式学到 "等所有 view 都来再做判断"，missing view 时性能崩盘。

### 7.2 Data Scaling

Figure 6 显示从 25% → 50% → 75% → 100% 数据量，C.IoU 和 V.IoU 持续上升且**未饱和**。这暗示 vision-to-touch 是 data-hungry task，scale 还能进一步推高。这跟 LLM 的 scaling law 类似——representation 复杂度足够大，数据量是 bottleneck。

---

## 8. Failure Cases 与 Limitations

Figure 12 给了一个典型 failure：fold black shorts 时，**黑色手套与黑色短裤在 ego view 下低对比度**，model 在 first frame（实际无 contact）误预测 left hand 有 contact。这是 model 学到 "视觉 proximity + occlusion → contact" 关联后，被 color similarity 误导。

启示：
1. Model 高度依赖 visual appearance cue，缺乏 explicit depth / segmentation 的 disambiguation
2. First frame 没有 temporal prior 可用，更容易 hallucinate
3. Color augmentation 可能缓解，但根本解法还是 multi-modal（depth、tactile sensor fallback）

其他 limitations：
- 全部数据戴 tactile glove 采集，**存在 glove-specific appearance bias**，难泛化到 bare-hand
- Data scaling 未饱和，需要更大规模、更 diverse 的采集
- 当前 benchmark 仅做 tactile estimation，未触及下游 task（grasp stability、affordance、world model）

---

## 9. 与相关工作的关系

### 9.1 Egocentric Hand-Object Datasets 谱系

- **Ego4D** (CVPR 2022) <https://ego4d-data.org/>：大规模 ego video，无 hand pose、无 tactile
- **EPIC-KITCHENS** (ECCV 2018) <https://epic-kitchens.org/>：厨房活动，无 dense tactile
- **EgoDex** (2026) <https://arxiv.org/abs/2505.11709>：194 tasks 双手 dexterous，3D hand/finger tracking，单 view，无 tactile
- **EgoPressure** (2024) <https://arxiv.org/abs/2405.05288>：ego + real pressure，但 single-hand、surface-only
- **OpenTouch** (2025) <https://arxiv.org/abs/2512.16842>：in-the-wild full-hand tactile，single-hand，single view
- **ARCTIC** (CVPR 2023) <https://arctic.is.tue.mpg.de/>：bimanual + analytical contact，无 real pressure
- **GRAB** (ECCV 2020) <https://grab.is.tue.mpg.de/>：whole-body grasp，analytical contact
- **OakInk** (CVPR 2022) <https://oakink.net/>：large-scale HOI knowledge，single-hand
- **DexYCB** (CVPR 2021) <https://dex-ycb.github.io/>： grasping benchmark，analytical contact
- **HOT3D** (CVPR 2025) <https://openaccess.thecvf.com/content/CVPR2025/html/Banerjee_HOT3D_Hand_and_Object_Tracking_in_3D_from_Egocentric_Multi-View_CVPR_2025_paper.html>：ego multi-view hand/object tracking，无 tactile
- **Ego-Exo4D** (CVPR 2024) <https://ego-exo4d-data.org/>：ego + exo cross-view，无 dense tactile

EgoTouch 是第一个同时具备 multi-view (含 wrist)、bimanual、real dense pressure、in-the-wild 的 dataset。

### 9.2 Vision-to-Touch 方法谱系

- **VisGel** (CVPR 2019) <https://visgel.csail.mit.edu/>：cross-modal vision-touch learning，paired GelSight
- **Touching a NeRF** (arXiv 2023) <https://arxiv.org/abs/2304.12828>：从 NeRF 合成 tactile
- **PressureVision** (ECCV 2022) <https://pressurevision.cs.columbia.edu/>：单 RGB 图像预测 hand pressure map
- **Touch and Go** (NeurIPS 2022) <https://touchandgo.csail.mit.edu/>：human-collected vision + touch paired data
- **GelSight** (Sensors 2017) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5750697/>：高分辨率 robot tactile sensor
- **DIGIT** (RA-L 2020) <https://digit.mit.edu/>：compact tactile sensor for in-hand manipulation

TouchAnything 的差异化点：multi-view + bimanual + egocentric deployment 场景，且显式建模 view dropout。

### 9.3 与 Robot Policy Learning 的潜在连接

EgoTouch 的 long-term vision 是为 robot policy learning 提供 tactile supervision。相关 recent work：
- **DreamDojo** (2026) <https://arxiv.org/abs/2602.06949>：从大规模 human video 学 generalist robot world model
- **EgoVLA** (2025) <https://arxiv.org/abs/2507.12440>：从 ego human video 学 VLA
- **EgoScale** (2026) <https://arxiv.org/abs/2602.16710>：scaling dexterous manipulation from ego data
- **UniDex** (2026) <https://arxiv.org/abs/2603.22264>：universal dexterous hand control from ego video
- **Being-H0.5** (2026) <https://arxiv.org/abs/2601.12993>：cross-embodiment generalization

这些工作都缺乏 tactile supervision。TouchAnything 可以作为 **pseudo-tactile labeler**，给大规模 ego video 自动生成 pressure map annotation，下游 robot policy 学习就能用 vision + pseudo-tactile 联合训练。

### 9.4 Tactile for Robot Manipulation

- **TacGNN** (RA-L 2023) <https://arxiv.org/abs/2302.14365>：tactile-based in-hand manipulation with hierarchical GNN
- **Attention for Robot Touch** (IROS 2023) <https://arxiv.org/abs/2305.16549>：tactile saliency prediction for sim-to-real

这些方法假设有真实 tactile sensor，TouchAnything 则在 "没有 sensor 也能预测 tactile" 的方向上推进。

---

## 10. 直觉总结与开放问题

### 10.1 关键 Insight

1. **Vision-to-touch 的 bottleneck 是 observation 不全**，model capacity 在 occlusion 面前无能为力。Wrist view 是信息补全的关键。
2. **Single wrist view 已捕获大部分互补信息**，因为 fisheye FOV 大。部署时一个 wrist camera 是 cost-effective 的 sweet spot。
3. **View dropout 是 multi-view model 的必须训练策略**，否则 deployment 时 missing view 会 catastrophic。
4. **Hand pose 是 spatially grounded reasoning 的 anchor**，joint-level cross-attention 让 visual 信息精确 attached 到对应 anatomical 位置。
5. **Tactile map 的 sparsity 需要 contact-region reweighting**，否则 model 塌缩到 all-zero trivial solution。
6. **Data scaling 未饱和**，vision-to-touch 还在 scaling regime。

### 10.2 开放问题

1. **Bare-hand 泛化**：当前训练数据全戴 glove，model 学到 glove appearance 与 tactile 的关联。Glove-to-bare-hand retargeting 怎么做？可能需要 GAN-based appearance translation 或 domain randomization。
2. **Cross-embodiment transfer**：human hand → robot end-effector 的 tactile distribution 完全不同，怎么 transfer？可能需要 learned retargeting network 或 simulation-based adaptation。
3. **Active tactile sensing**：当前是 passive prediction，能不能闭环——prediction 不确定时主动调整 wrist camera 角度补足信息？
4. **Tactile world model**：把 predicted pressure 作为 latent state 输入到 world model，能否改善 robot policy 的 contact-rich manipulation？
5. **Multi-finger force closure 评估**：predicted pressure map 能否直接用来评估 grasp stability？这需要 pressure → wrench 的 forward model。
6. **Failure mode 分析**：低对比度场景下 model hallucinate contact，能否引入 explicit depth 或 hand-object segmentation 来 disambiguate？
7. **Self-supervised refinement**：少量真实 tactile data + 大量 pseudo-label，能否做 semi-supervised refinement？

### 10.3 与 LLM/World Model 范式的联想

EgoTouch + TouchAnything 实质是在做 **"visual encoder + tactile decoder" 的 modality translation**，与 LLM 中 "text encoder + vision decoder" 的多模态生成有结构相似性。可以想象未来一个 unified embodied foundation model：
- 输入：multi-view video + audio + IMU + hand pose
- 输出：text action description + predicted tactile map + next-frame video prediction + audio
- 训练：自监督 + 少量 paired tactile supervision

Tactile 作为 "物理 grounded" 的 modality，能显著提升 world model 对 contact dynamics 的建模精度。当前 video world model（Sora 类）对 contact 物体经常出现穿透、漂浮，正是因为缺乏 contact supervision。TouchAnything 的 pseudo-tactile label 可能成为 video world model 的物理 grounding 信号。

---

## 11. 与 Sim2Real 的关联

tactile sensor 在 sim2real 里是个老大难：simulation 里可以读 ground truth contact force，但 real robot 上 tactile sensor 噪声大、易坏。TouchAnything 反过来——**用 vision 给 real 数据补 tactile**，相当于 "real2real tactile synthesis"。这条路可能比 sim2real 更 scalable，因为 human ego video 是几乎无限的数据源。

相关工作：
- **RoboSense** (CVPR 2025) <https://openaccess.thecvf.com/content/CVPR2025/html/Su_RoboSense_Large-scale_Dataset_and_Benchmark_for_Egocentric_Robot_Perception_and_CVPR_2025_paper.html>：ego robot perception dataset
- **Tactile sim2real survey** (Yamaguchi & Atkeson 2019) <https://arxiv.org/abs/1907.09827>：can we turn tactile sensing into vision?

### 11.1 与 ActionSense 的对比

ActionSense (NeurIPS 2022) <https://actionsense.github.io/> 是另一个用 wearable sensor 记录 kitchen 活动的 dataset，包含 pressure glove，但 single-view、室内、glove-only。EgoTouch 把它扩展到 multi-view + bimanual + in-the-wild。

---

## 12. 总结

EgoTouch + TouchAnything 在 vision-to-touch 方向迈出了关键一步：

- **Dataset**：首次同时具备 multi-view (含 wrist)、bimanual、real dense pressure、in-the-wild、千 object-scale
- **Method**：cross-view attention + gated fusion + pose-vision cross-attention + view dropout，一个简洁但有效的 multi-view baseline
- **Benchmark**：5 个 environment scenario × seen/unseen × 4 个 view 配置，提供 systematic 评估 protocol
- **Future**：glove-to-bare-hand、cross-embodiment、tactile world model、grasp stability，下一步空间巨大

对 Karpathy 你来说，这个工作直觉上很关键的一点是：**tactile 是 embodied intelligence 的 "missing modality"**，而大规模采集真 tactile 不可行。TouchAnything 把问题转化为 "from scalable vision data infer tactile"，与 LLM 用大量 text + 少量 paired modality 的 scaling 思路完全一致。如果这条路 work，意味着我们可以用 internet-scale ego video 训练出有 tactile understanding 的 embodied foundation model，sim2real 的 contact bottleneck 也可能被绕过。

Project page: <https://jianyi2004.github.io/TouchAnything-Website/>
Correspondence: shuoyang@hit.edu.cn
