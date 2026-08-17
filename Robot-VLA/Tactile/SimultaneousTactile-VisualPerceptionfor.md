---
source_pdf: SimultaneousTactile-VisualPerceptionfor.pdf
paper_sha256: 386741db69c035e3c3e506777ef573b213f2a84ddfe682f06d14254ff89a51c1
processed_at: '2026-08-12T06:57:12-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 TacThru

## 一句话概括

给 robot gripper 装一个**又能看又能摸**的 finger，看和摸同时进行，然后让 Diffusion Policy 自己学会什么时候该看、什么时候该摸。

---

## 为什么需要这个东西

想象你闭着眼睛去抓一个塑料瓶。你能看见瓶子大概在哪（wrist camera 或者第三人称 camera），但 gripper 一合上，你的眼睛就废了——finger 把瓶子挡住了。接下来你要靠 touch 来判断：抓到了没有？抓歪了没有？瓶子在滑吗？

反过来，你只用 tactile sensor（比如 GelSight）。finger 没碰到东西之前，sensor 什么信号都没有。approach 阶段你是瞎的，只能靠 prior position 估计着往前伸。

**问题就在这**：vision 在 pre-contact 阶段好用但 contact 之后瞎掉，tactile 在 contact 之后好用但 pre-contact 阶段瞎掉。中间有个断裂带。

有人做过 STS (See-Through-Skin) sensor，就是让 tactile sensor 里面的相机透过 elastomer 同时看外面。但之前的做法很笨——要么用灯光切换（先开内灯拍 marker，再开外灯拍环境），要么用机械结构动 elastomer。这就引入了延迟和控制复杂度，而且切换时机全靠 hand-crafted rule。

TacThru 的核心 insight 很简单：**别切了，同时来**。

---

## TacThru 怎么做到同时

三个工程决策：

### 1. Elastomer 全透明

传统 GelSight 的 elastomer 里面有一层 opaque reflective coating（通常是铝镜或者彩色涂层），作用是让内部 LED 光反射回来，形成 high-contrast 背景，方便 marker detection 和 photometric stereo 算 depth。

TacThru 把这层 coating 直接去掉了，elastomer 全透明。内部相机就直接透过 elastomer 看到外面的 world。

代价：你失去了 GelSight 那种 fine surface geometry reconstruction（用 RGB 通道推 normal 再积分回 depth）。但作者认为 manipulation policy 其实在大多数场景下不需要 sub-mm 的 surface 重建，更需要的是"contact 有没有发生"和"外面的环境长什么样"。

### 2. LED 一直开着

24 个 LED 灯泡围着 elastomer，一直亮，不切。加上 diffusion film 把光打散，避免 specular reflection。

这样一帧图像里同时包含：
- 外部 world 的 appearance（透过透明 elastomer 看到的）
- Marker 的位置（elastomer 表面变形导致 marker 位移）
- Contact 时的亮度变化（refractive index mismatch 消失导致的光学变化）

一帧图像 = visual + tactile，120Hz。

### 3. Keyline Markers（最有意思的 trick）

这是 transparent elastomer 带来的 side effect：传统黑点 marker 在黑色背景上直接消失了（黑对黑看不见）。

作者的解法：**画两个同心圆**，外面白色，里面黑色。

- 白色外圈在 dark 背景上可见
- 黑色内圈在 light 背景上可见
- 两者交界的 edge 在**任何背景**上都有 contrast

这就是 keyline marker。64 个 marker 铺在 40×40mm 的 elastomer 上，间距 3.5mm。

### 4. Kalman Filter 跟踪 marker

Keyline 解决了"看不见"的问题，但还有"看错了"的问题——外部 world 里的 blob-like 结构（文字、阴影、纹理）会被 blob detector 误检为 marker。

作者的解法是给每个 marker 跑一个 Kalman filter：

- **Predict**: 假设 marker 不动（Random Walk），uncertainty 累积
- **Measure**: blob detector 检出候选点，找离 prediction 最近的那个当 measurement
- **Update**: 用 Kalman Gain 加权 predict 和 measure
- **Reject**: 如果 measurement 离 prediction 太远（>10px），直接拒绝，保持上一帧的 estimate，让 uncertainty 继续累积等下一帧

这套东西跑在 CPU 上，64 个 marker 全部跟踪，平均 6.08ms 一帧，支持 120Hz。bottleneck 反而是 image read-in（UVC 协议 + OpenCV blocking），tracking 本身很快。

---

## 怎么接到 learning 里

这部分是工程整合，没有太多 novelty，但做得很扎实：

1. **Data collection**: 改造 UMI handheld device，把 standard finger 换成 TacThru finger。用 HTC Vive Tracker 跟踪 pose（替代 UMI 原版的 SLAM，更稳）。30FPS 采集，存 Zarr format。

2. **Robot end-effector**: 自制 gripper mirror data collector 的 body，finger width 由 servo-electric cylinder 控制（~$280）。

3. **Policy**: Transformer-based Diffusion Policy。四路 observation：
   - Wrist camera → DINOv2 ViT-B encode
   - TacThru image → DINOv2 ViT-S encode
   - Marker deviation → MLP encode
   - Proprioception (end-effector pose + gripper width) → MLP encode
   
   四路 token concat，加 learnable modality embedding（让 Transformer 知道哪个 token 来自哪个 modality），加 positional embedding，condition 给 Diffusion Policy denoise 出 16 步 action chunk，执行中间 3-8 步（receding horizon）。

关键：Transformer 的 self-attention 让 policy **自己学**什么时候该 attend visual、什么时候该 attend tactile。不需要写 rule。

---

## 实验说明了什么

五个任务，四组 baseline：

| | PickBottle | PullTissue | SortBolt | HangScissors | InsertCap |
|---|---|---|---|---|---|
| 核心挑战 | 基础验证 | 薄纸 tactile 检测不到 | 区分 M12 螺栓颜色+形状 | 判断挂钩成功 | mm 级 insertion |
| TT-M (TacThru+marker) | ~95% | high | ~80% | high | high |
| TT (TacThru image only) | ~95% | high | ~80% | lower | high |
| GS-M (GelSight+marker) | ~95% | **low** | **confused** | high | medium |
| Wrist (vision only) | ~95% | **low** | **low** | **low** | medium |

平均：TT-M 85.5%，GS-M 66.3%，Wrist 55.4%。

### 四个关键发现

**PullTissue**：薄纸产生的 force 太小，GelSight 检测不到 contact。但 TacThru 透过透明 elastomer 直接看到 tissue 在 finger 间的位置。tissue 滑了，TT-M 立刻检测到位移触发 retry，GS-M 和 Wrist 都瞎着继续 pull。

**SortBolt**：M12×25 螺栓，A 是 button head 黑色，B 是 socket head 银色，C 是 socket head 黑色。Wrist camera 在 mounted distance 上 resolve 不了头部几何。GelSight tactile 只感知几何，分不清 B 和 C（形状一样颜色不同）。TacThru 的 close-proximity visual 能看清颜色和细微几何。t-SNE 可视化 DINOv2 embedding：TacThru 三个 cluster 分得很开，GelSight 的 B 和 C 重叠。

**HangScissors**：挂钩成功与否，wrist camera 因 occlusion 看不清。Marker displacement 提供 tactile evidence。这验证了 explicit marker tracking 的必要性——TT（不 track marker）在这个任务上比 TT-M 差。

**InsertCap（最核心的发现）**：mm 级 insertion。20 次 trial 里 15 次 cap-mount interface 可见，policy 用 visual servoing 直接对齐；5 次 grasp 导致 occlusion，policy **无缝切换**到 tactile-based insertion，用 marker displacement 检测 contact 和 guide alignment。这种 adaptive behavior **没有 rule-based 编程**，完全从 demonstration 学出来。

InsertCap 这个结果其实是整篇 paper 的 thesis 的实证：simultaneous multimodal perception + Transformer attention = policy 自己学会 adaptive strategy。sequential sensing 配合 rule 切换做不到这件事，或者需要大量 hand-tuned heuristic。

---

## DINOv2 在 TacThru image 上的 surprise

TacThru image 理论上有很大 domain gap：marker 覆盖、elastomer 光学畸变、contact 变形。但 DINOv2（在 LVD-142M 上自监督预训练）zero-shot transfer 表现很好。PCA 可视化 patch token 显示 DINOv2 能清晰区分 markers、manipulated objects、background。这意味着**不需要为 TacThru 训练专门 encoder**，直接用预训练 visual encoder 就 work，大幅降低 implementation barrier。

---

## Limitations

1. Elastomer 极端 load / sharp indentation 下会 delamination
2. 高强度环境光干扰 camera auto-exposure，marker contrast 下降
3. 放弃了 depth reconstruction，需要 sub-mm geometry 重建的任务不适用
4. 未来方向：大规模数据 + tactile simulation (Taccel) 预训练 specialized encoder，探索 dexterous tasks

---

## 这篇 paper 的位置

Sensor 端：STS sensor 谱系里走"透明 + 持续照明 + keyline + Kalman"这条线，比 mode-switching STS 多了 simultaneity，比 stereo STS 少了 depth 但多了 simplicity 和 throughput。

Learning 端：第一个把 STS 完整接入 Diffusion Policy + UMI pipeline 的工作。核心 contribution 不是 sensor 本身有多新，而是证明了 **simultaneous multimodal perception + Transformer attention 能让 policy 自适应地学 modality 切换策略**，不需要 rule。两端协同设计才是 85.5% vs 55.4%/66.3% 差距的根源。

---

# TacThru: Simultaneous Tactile-Visual Perception 深度解析

这篇 paper 来自 Yixin Zhu 组 (BIGAI)，核心想做一件事：让 robot manipulation 既看到接触前的环境，又感知接触时的力，并且把这两路信号同时喂给 modern imitation learning framework (Diffusion Policy)。我从 design intuition、sensor physics、tracking algorithm、policy architecture、experimental findings 五个层面拆解。

---

## I. Motivation: 为什么 manipulation 需要"同时" tactile + visual

经典 manipulation 感知有两条断裂带：

1. **Vision 的盲区**：第三人称 / wrist camera 在 pre-contact 阶段很好，但一旦 gripper 闭合接近 object，object 被 finger 遮挡，精度要求最高的 contact-rich 阶段反而看不见。
2. **Tactile 的盲区**：VBTS (GelSight, DIGIT, 9DTact, MiniTac 等) 只在 contact 之后才出信号。approach 阶段是黑的，slip 检测、fine alignment 都只能 reactive，没法 proactive。

STS (See-Through-Skin) sensors 想把这两条带打通：让 VBTS 内部的相机透过 elastomer 同时看到外部 world 和 marker deformation。现有方案 (FingerSTS, Stereotac, SpecTac, CompdVision, Look-To-Touch 等) 大多通过 **illumination switching** 或 **mechanical actuation** 在两种 mode 之间切，引入两个 cost：
- 时序错位（actuation 需要 second-level delay）
- 距离估计 mm-level error（基于 reflective coating 的几何深度估计在透明 coating 下不成立）

TacThru 的关键 insight：**放弃 fine depth reconstruction，直接用 transparent elastomer + 持续照明换 continuous visual perception**。这是一种 sensing-level 的 trade-off，把"高精度深度重建"换成"全时序的视觉 + 触觉同步流"。

参考：
- FingerSTS: https://arxiv.org/abs/2204.06064
- Stereotac: https://arxiv.org/abs/2308.06826  
- UMI: https://arxiv.org/abs/2402.10329
- GelSight: https://arxiv.org/abs/1708.00922

---

## II. Sensor 设计：三层工程决策

### A. Transparent Elastomer + Persistent Illumination

把传统 VBTS 的 reflective opaque coating 整个换成 fully transparent elastomer (例如透明 silicone)。照明系统是 **24-bulb LED array** (RGBW 用于兼容标准 VBTS 流水线，white-only 用于最高视觉清晰度)，外加：
- LED 上的 **diffusion film**
- Acrylic plate 两侧的 **diffusion film**

作用是消除 specular reflection，让光均匀漫射。这就解决了"半透明涂层在强光下产生 hot spots"的问题。

**Contact detection 的两条 backup 通路**：
1. **Light reflection change at contact interface**：当 elastomer 接触物体时，refractive index mismatch 消失 (从 air-silicone 界面变成 object-silicone 界面)，亮度分布变化。这是 TIRGel / TacTip 类 sensor 共用的原理。
2. **Marker divergence from elastomer deformation**：shear force 导致 elastomer 表面发生 tangential displacement，markers 跟着移动，从 displacement vector 推 shear 力。

注意：**没有 depth reconstruction**。传统 GelSight 利用 colored photometric stereo 从 RGB 通道推 surface normal 再积分回 depth，这里因为涂层透明 + 外部光污染，这条路放弃了。Paper §III-A 明确说："explicitly prioritizes global contact state and visual context at the fingertips over fine-grained surface geometry reconstruction"。

### B. Keyline Markers

这是 sensor 层最有意思的设计。透明 elastomer 引入两个 marker detection 的 fundamental problem：

**(i) Degraded detectability**: 传统 solid markers (黑点) 在黑色背景上消失。GelSight 类 sensor 用 RGB 编码 marker 因为 coating 提供 stable background contrast，这里不行。

**(ii) Noisy detections**: 外部 world 里的任何 blob-like 结构 (text、shadow、纹理) 都会被 blob detector 误检为 marker。

Keyline marker 的设计：**两个同心圆**：
- Inner circle: 黑色, $r_{\mathrm{in}} = 0.6 \mathrm{mm}$
- Outer circle: 白色, $r_{\mathrm{out}} = 1.0 \mathrm{mm}$

关键直觉：白色外圈在 dark 背景上可见，黑色内圈在 light 背景上可见，**两者交界处 (keyline edge) 在任何背景上都形成高对比 edge**。这是一个图像处理层面的 invariance trick，类似于 zero-crossing edge detection (Marr-Hildreth) 的思路——edge 的可检测性高于 region。

参数：
- $N_m = 64$ markers
- Spacing $d_{\mathrm{marker}} = 3.5 \mathrm{mm}$
- Pixel spacing $\tilde{d}_{\mathrm{marker}} = 51 \mathrm{px}$
- Elastomer: $40 \times 40 \mathrm{mm}$

设计约束：
1. **Camera focus distance** 必须同时满足 marker 检测和 visual perception
2. **Marker size** 平衡 detectability 和 visual occlusion (markers 太大就遮挡透明 visual)
3. **Marker spacing > max marker deviation**：防止 tracking ambiguity (两个 marker 因 deformation 互相靠近导致 data association 错乱)

### C. Robust Marker Tracking via Kalman Filter

这是 paper 里数学最完整的部分。问题：keyline 解决了 detectability，但 noise 和 large deformation 还会导致 false positive / mis-match。

**State-space model** (Eq. 1):
$$x_t = A_t x_{t-1} + w_t, \quad z_t = H_t x_t + v_t$$

变量：
- $x_t \in \mathbb{R}^2$：marker 在 image plane 上的真实 position (latent state)
- $z_t \in \mathbb{R}^2$：blob detector 输出的 measured position (observation)
- $A_t$：state transition matrix (这里取 Random Walk: $A_t = \mathbb{I}_2$)
- $H_t$：observation matrix (这里取 direct observation: $H_t = \mathbb{I}_2$)
- $w_t \sim \mathcal{N}(0, Q)$：process noise
- $v_t \sim \mathcal{N}(0, R)$：measurement noise
- $Q = \sigma_w^2 \mathbb{I}_2$, $R = \sigma_v^2 \mathbb{I}_2$

Random Walk model 的物理含义：marker 的瞬时位移无法预测 (取决于 contact 何时何地发生)，只能假设它停在上一个估计位置附近，方差随时间累积。

**Predict step** (Eq. 2-3):
$$\tilde{x}_t = A_t \hat{x}_{t-1} = \hat{x}_{t-1}$$
$$\tilde{P}_t = A_t \hat{P}_{t-1} A_t^T + Q = \hat{P}_{t-1} + Q$$

- $\tilde{x}_t$：prior state estimate (predicted)
- $\hat{x}_{t-1}$：posterior state estimate from previous step
- $\tilde{P}_t$：prior covariance (uncertainty 增加了 $Q$)
- $\hat{P}_{t-1}$：posterior covariance from previous step

**Kalman Gain** (Eq. 4):
$$K_t = \tilde{P}_t H_t^T (H_t \tilde{P}_t H_t^T + R)^{-1} = \tilde{P}_t (\tilde{P}_t + R)^{-1}$$

直觉：$K_t$ 衡量"多大程度信任 measurement 而非 prior"。当 measurement noise $R$ 大，$K_t \to 0$ (信任 prior)；当 prior uncertainty $\tilde{P}_t$ 大，$K_t \to 1$ (信任 measurement)。

**Update step** (Eq. 5-6):
$$\hat{x}_t = \tilde{x}_t + K_t (z_t - H_t \tilde{x}_t) = \tilde{x}_t + K_t (z_t - \tilde{x}_t)$$
$$\hat{P}_t = (\mathbb{I}_2 - K_t H_t) \tilde{P}_t = (\mathbb{I}_2 - K_t) \tilde{P}_t$$

- $(z_t - \tilde{x}_t)$：innovation / residual，即 measurement 和 prediction 的差
- $\hat{P}_t$：posterior covariance，比 prior 减小 (因为测量降低了 uncertainty)

**Initialization**: $\hat{x}_0, \hat{P}_0 = \epsilon \mathbb{I}_2$ with $\epsilon = 10^{-3}$，低 uncertainty 表示初始位置已知 (marker 网格是预先定义的)。

**Measurement acquisition pipeline**:
a) Grayscale: $I_{\mathrm{gs},t} \gets \mathrm{Grayscale}(I_t)$
b) Thresholding: $I_{\mathrm{bin},t}(u,v) = \mathbb{1}(I_{\mathrm{gs},t}^{(u,v)} > \tau_i)$
   - $\tau_i = 0.784$ (= 200 in UINT8)：高阈值，只保留 bright 区域 (即白色外圈)
   - $\mathbb{1}(\cdot)$：indicator function
   - $(u,v)$：pixel coordinate
c) Blob Detection: OpenCV SimpleBlobDetector
   - 输出: $Z_t = \{z\}$
d) Data association: $z_t = \arg\min_{z \in Z_t} \|z - \hat{x}_{t-1}\|$
   - 最近邻匹配，利用 prior 估计排除远处 noise

**Motion continuity constraint** (reject 离谱 measurement):
$$\text{if } \|z_t - \hat{x}_{t-1}\| > \tau_z, \text{ then } K_t := 0, \hat{x}_t := \hat{x}_{t-1}, \hat{P}_t := \hat{P}_{t-1} + Q$$

直觉：如果 measurement 跳得太远 (超过 $\tau_z = 10 \mathrm{px}$)，几乎肯定是 mis-association，干脆忽略它，让 uncertainty 累积，等下一帧再来。

$\tau_z$ 的设计约束：$\tau_z$ 必须 ≤ $\frac{\tilde{d}_{\mathrm{marker}}}{2}$ (半个 marker 间距)，否则可能匹配到隔壁 marker。

**参数标定**：
- $\sigma_v = 0.42 \mathrm{px}$：通过静止状态下的 sensor recording 标定 (纯 measurement noise)
- $\eta = \sigma_v / \sigma_w \approx 4$，因此 $\sigma_w = 0.11 \mathrm{px}$：经验调参，平衡 responsiveness vs stability

η = 4 的物理含义：measurement 比 process 更可信 4 倍。Process noise 不能太小，否则 Kalman gain 太低，对 fast contact 响应慢；不能太大，否则 jitter 严重。

### D. 性能数据

- 平均处理延迟：**6.08 ms** (CPU only, AMD Ryzen 9 5950X)
- 支持 120Hz operation
- 在 Fig. 4 的时间分解里，**image read-in** 才是 bottleneck (UVC protocol + OpenCV VideoCapture.retrieve() blocking)，undistortion 和 marker tracking 都很快

对比实验 (Fig. 3, 抓 plastic bottle with 复杂黑白文字)：
- Solid markers: frequent missed detections
- Keyline only (no filter): 检出 > 64 个 (false positives)
- Keyline + Kalman: 稳定跟踪全部 64 个 markers

---

## III. TacThru-UMI: 把 sensor 接入 imitation learning

### A. Hardware Setup

- **Data collector**: 改造 UMI handheld device，把 standard finger 换成 TacThru finger (extending linkages)
- **Robot end-effector**: 自制 low-cost gripper，mirror data collector 的 body，finger width 由 Inspire LAS30-021D servo-electric cylinder 控制 (~$280)
- **Pose tracking**: 用 HTC Vive Tracker 替代 UMI 原版的 SLAM-based tracking (提高 success rate)
- **Data frequency**: 30 FPS (sensor 支持 120Hz 但 30 已足够 [37, 43])
- **Storage**: Zarr format，所有数据流同步到 wrist-camera timestamps

### B. Policy Architecture (Transformer-based Diffusion Policy)

Observations at timestep $t$:
- Wrist-camera frames: $\mathbf{I}_w^t := \{I_w^i\}_{i=t-n_w^{\mathrm{obs}}}^{t-1}$
- Sensor frames: $\mathbf{I}_s^t := \{I_s^i\}_{i=t-n_s^{\mathrm{obs}}}^{t-1}$
- Marker deviations: $\Delta\mathbf{x}^t := \{\Delta x^{i,j}, j=1,\ldots,N_m\}_{i=t-n_e^{\mathrm{obs}}}^{t-1}$
- Proprioception: $\mathbf{s}^t := \{s^i\}_{i=t-n_v^{\mathrm{obs}}}^{t-1}$ (end-effector pose + gripper width, relative coordinates)

Encoders:
- DINOv2 ViT-B for wrist camera
- DINOv2 ViT-S for TacThru frames (14×14 patch size → 每张图 196 tokens 左右)
- Dedicated MLPs for marker deviations 和 proprioception

Token encoding (Eq. 7-10):
$$\mathbf{z}_w = \{\mathrm{DINO}_w(I) + z_w \mid I \in \mathbf{I}_w^t\}$$
$$\mathbf{z}_s = \{\mathrm{DINO}_s(I) + z_s \mid I \in \mathbf{I}_s^t\}$$
$$\mathbf{z}_x = \{\mathrm{MLP}_x(\Delta x) + z_x \mid \Delta x \in \Delta\mathbf{x}^t\}$$
$$\mathbf{z}_p = \{\mathrm{MLP}_p(s) + z_p \mid s \in \mathbf{s}^t\}$$

变量解释：
- $\mathbf{z}_w, \mathbf{z}_s, \mathbf{z}_x, \mathbf{z}_p$：四种 modality 的 token sequences
- $z_w, z_s, z_x, z_p$：learnable modality embeddings (类似 ViT 的 class token / BERT 的 segment embedding)，让 Transformer 区分 token 来自哪个 modality
- DINOv2 输出是 patch-level feature，每个 patch 一个 token

这些 token concat 起来 + positional embedding 作为 condition，喂给 Diffusion Policy $\pi_\theta$：

$$\mathbf{a} = \{a^i\}_{i=t}^{t+T_a-1} \sim \pi_\theta(\mathbf{a} \mid \mathbf{z}_w, \mathbf{z}_s, \mathbf{z}_x, \mathbf{z}_p)$$

- $\mathbf{a}$：action chunk (序列)
- $a^i$：第 $i$ 步 action (relative end-effector pose + gripper width)
- $T_a = 16$：prediction horizon (predict 16 步)
- 执行：前 $L_a$ 步 ($L_a \leq T_a$，实际 steps 3-8) 发给 robot controller

Diffusion Policy 的核心：把 action generation 看成 denoising process。从 Gaussian noise $\mathbf{a}_K \sim \mathcal{N}(0, I)$ 出发，通过 $K$ 步去噪得到 $\mathbf{a}_0 = \mathbf{a}$，每步：
$$\mathbf{a}_{k-1} = \frac{1}{\sqrt{\alpha_k}} \left( \mathbf{a}_k - \frac{1-\alpha_k}{\sqrt{1-\bar{\alpha}_k}} \epsilon_\theta(\mathbf{a}_k, k, \mathbf{z}_w, \mathbf{z}_s, \mathbf{z}_x, \mathbf{z}_p) \right) + \sigma_k \xi$$

(这是 DDPM 形式；Diffusion Policy 实际用 DDIM variant)

Transformer 部分：condition tokens 通过 cross-attention 或 concat 进 denoising network。Transformer 的 self-attention 让 policy **动态决定哪些 modality 在当前 context 下更重要**——这是 §V-C 里 InsertCap 任务展现 adaptive behavior 的根本机制。

参考：
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- DINOv2: https://arxiv.org/abs/2304.07193
- ACT / Action Chunking: https://arxiv.org/abs/2304.13705

### C. Training Setup

- Observation horizons: $n_w^{\mathrm{obs}} = 1, n_s^{\mathrm{obs}} = 1, n_p^{\mathrm{obs}} = 2$
- Predict: $T_a = 16$ action steps
- Execute: steps 3-8 (执行中间一段，receding horizon control)
- 150 epochs
- AdamW optimizer, lr = $3 \times 10^{-4}$, one-cycle scheduler
- 每个 checkpoint 20 rollouts (SortBolt 24 rollouts, 每个 bolt 8)

---

## IV. Experimental Findings: 5 个任务，4 个 baseline

### A. Task Suite

| Task | Type | 主要挑战 | 关键 modality |
|------|------|---------|--------------|
| PickBottle | Pick-and-place | 基础验证 | Vision + Tactile |
| PullTissue | Thin/soft object | 薄纸几乎不产生 tactile force | Vision (透过 transparent 看) |
| SortBolt | Visual discrimination | M12×25 bolts 头部几何 + 颜色细微差异 | STS visual |
| HangScissors | Tactile discrimination | 挂钩是否成功需 tactile 反馈 | Marker displacement |
| InsertCap | Multimodal fusion | mm-level precision alignment | Vision + Tactile (adaptive) |

### B. Policy Variants

公平对比设计：gripper 一边装 TacThru，另一边装 GelSight，确保 training trajectory 完全相同。所有 policy 都包含 wrist camera + proprioception 作为 baseline。

- **TT-M**: TacThru images + marker deviations (完整版)
- **TT**: TacThru images only (marker 可见但不显式 track，作为 ablation，看 explicit marker tracking 是否必要)
- **GS-M**: GelSight images (rectified by idle image for isolating contact) + marker deviations (tactile baseline)
- **Wrist**: Wrist camera only (vision baseline)

### C. Quantitative Results (Fig. 7)

| Policy | PickBottle | PullTissue | SortBolt | HangScissors | InsertCap | Avg |
|--------|-----------|-----------|---------|-------------|----------|-----|
| TT-M | ~95% | high | ~80% | high | high | **85.5%** |
| TT | ~95% | high | ~80% | lower | high | slightly lower |
| GS-M | ~95% | low | confused | high | medium | **66.3%** |
| Wrist | ~95% | low | low | low | medium | **55.4%** |

TacThru-UMI 相对 vision baseline 提升 **1.54×**，相对 tactile baseline 提升 **1.29×**。

### D. 关键定性发现

**1. PullTissue: Tactile sensor 的根本局限**

薄纸产生的 pressure 太小，piezoresistive / VBTS 都检测不到 contact。TacThru 通过 transparent visual 直接看到 tissue 在 finger 间的位置。当 tissue slippage 发生时：
- GS-M (GelSight): 无法感知，gripper 持续 pull 但 tissue 已掉
- Wrist: wrist camera 看不到 finger 间，无法感知
- TT-M: **立即检测 displacement，触发 retry**

这是 STS 的独特优势——把"force-based tactile"换成"vision-based proximity + tactile"，扩展了 manipulable object 的范围。

**2. SortBolt: Visual discrimination 的极限**

M12×25 bolts，三种：
- A: button head, 黑色
- B: socket head, 银色
- C: socket head, 黑色

挑战：
- Wrist camera 在 mounted distance 上无法 resolve bolt head 几何
- Tactile 无法区分几何相同但颜色不同的 B vs C

TacThru 的 close-proximity view 能 capture fine geometric features + 颜色。Fig. 9a 的 confusion matrix 显示：
- TT-M, TT: ~80% accuracy，三种 bolt 都分得清
- GS-M: B 和 C 完全混淆 (因为 tactile 只感知几何，不感知颜色)

Fig. 9b 的 t-SNE 可视化 DINOv2 CLS embedding：TacThru 三个 cluster 分得很开，GelSight 的 B 和 C 重叠在一起。

**3. HangScissors: Tactile 决策时机**

挂钩成功 vs 失败，wrist camera 无法可靠判断 (2D perception + occlusion)。Marker displacement 提供 tactile evidence：成功挂上时会有特定的 force pattern，policy 据此决定何时 release gripper。这验证了 §IV-C 里 explicit marker tracking 的必要性 (TT > TT-M 在这个任务上较差，因为 TT 没有显式 marker signal，只能靠 raw image 隐式推断)。

**4. InsertCap: Adaptive Multimodal Strategy (最有意思的发现)**

mm-level precision insertion。两种情况：
- (15/20 trials) Cap-mount interface visible → policy 用 **visual servoing** 直接对齐 visual features
- (5/20 trials) Grasp degraded visual perception (occlusion / cap 倾斜) → policy **seamlessly 切换到 tactile-based insertion**，用 marker displacement pattern 检测 contact 和 guide alignment

这种 adaptive behavior **没有任何 rule-based programming**，完全从 demonstration data 学出来。机制是 Transformer self-attention：当 visual token 信息量低 (attention weight 自然下降)，policy 自动 weight tactile token 更高。这是 simultaneous multimodal perception 相对 sequential 的根本优势——sequential mode 必须预先决定何时切换，simultaneous 让 policy 自己学 attention distribution。

### E. DINOv2 在 TacThru image 上的 Generalization (Fig. 10)

理论上 TacThru image 有很大 domain gap：
- Markers 覆盖在 image 上
- Transparent elastomer 引入 optical distortion
- Contact 时有 deformation artifacts

但 DINOv2 (在 LVD-142M 上自监督预训练) 表现出强 zero-shot transfer。PCA on patch tokens 显示 DINOv2 能清晰区分 markers、manipulated objects、background elements (opposing finger)。这降低了 implementation barrier——**不需要为 TacThru 训练专门 encoder**，直接用预训练 visual encoder 就 work。

参考 DINOv2 zero-shot transfer: https://arxiv.org/abs/2304.07193

---

## V. Limitations & Future Work

1. **Elastomer delamination**: 极端 load / sharp indentation 下会损坏。需要 reinforced surface materials (e.g., stretchable protective layers [64] PolyTouch)。
2. **Ambient illumination 干扰**: 高强度环境光干扰 camera auto-exposure，降低 marker contrast。可以通过降低 $\tau_i$ 手动缓解，但 adaptive exposure control 是 future work。
3. **放弃 depth reconstruction**: TacThru 不提供 GelSight 那种 fine surface geometry。对于需要 sub-mm geometry 重建的任务 (e.g., texture recognition)，传统 VBTS 可能更合适。
4. **Future**: 大规模数据收集 + synthetic tactile simulation (Taccel [65], Sim2Real [50, 51]) 预训练 specialized encoder，探索 dexterous tasks。

参考：
- Taccel: https://arxiv.org/abs/2506.06888 (大概)
- TacMan: https://arxiv.org/abs/2403.07840
- PolyTouch: https://arxiv.org/abs/2412.02132

---

## VI. Build Intuition: 这篇 paper 在大图里的位置

Manipulation sensing 的光谱：
- **Global vision** (3rd-person camera): context-rich, occlusion-blind
- **Wrist vision** (eye-in-hand): local, still occluded at contact
- **Proximity sensors** (capacitive, ultrasonic, IR): pre-contact gap, sparse
- **VBTS** (GelSight, DIGIT): contact-rich, post-contact only
- **STS** (FingerSTS, Stereotac, TacThru): pre-contact + contact unified

TacThru 在 STS 子类里属于 **"透明 + 持续照明 + keyline marker + Kalman tracking"** 这条路线，相比 mode-switching STS 多了 simultaneity，相比 stereo STS 少了 depth 但多了 simplicity 和 120Hz throughput。

Learning 层面：这是 **第一个把 STS sensor 完整接入 Diffusion Policy + UMI 数据收集 pipeline** 的工作 (前作 [24] Multimodal imitation 用的是 hand-crafted controller)。核心 contribution 是证明 simultaneous multimodal perception + Transformer attention 能让 policy **自适应地** 在 modality 之间切换，无需 rule-based 编程。

潜在联想方向：
- 与 TacMan / TacMan-Turbo 的 articulated object manipulation 结合：tactile-informed prior 在 transparent visual 下可能更鲁棒
- 与 Taccel GPU tactile simulation 结合：sim2real 预训练 TacThru-specific encoder
- 与 in-hand rotation (Qi et al. CoRL 2023) 结合：simultaneous visual-tactile 可能解锁更复杂的 in-hand reorientation
- 与 Vitamin / ViTacFormer 等 visuo-tactile policy 架构对比：cross-modal representation learning 是否比 simple token concat 更好

整体看，TacThru 是 sensor design + learning framework 协同设计的范例：sensor 端通过 transparent + keyline + Kalman 把 multimodal 信号"同时"提供出来，learning 端通过 Transformer attention 自然地"消费"这种 simultaneity。两端缺一不可——sequential sensor 配 Transformer 也能学切换，但需要更多数据；simultaneous sensor 配 rule-based controller 也能 work，但无法 adaptive。这个组合才是 85.5% vs 55.4%/66.3% 差距的根源。
