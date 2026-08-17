---
source_pdf: WARPED Wrist-Aligned Rendering for Robot Policy.pdf
paper_sha256: a150fc7bed842cf9be658794b74a47c4a346a7b2ae89e37a542882ad6459e29c
processed_at: '2026-08-13T03:39:03-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# WARPED 用人话说一遍

## 这事到底难在哪

你想训一个机械臂 policy，最简单直接的办法是 teleop：人拿个 VR 手柄或者 3D mouse 控制机器人，录几百段 trajectory，喂给 diffusion policy。问题是 teleop 慢得要死，录 30 段要半小时，动作还不一定自然——让人用 VR 手柄做一个 90 度旋转盒子，人手会抖，VR 手柄没有触觉反馈，录出来的 trajectory 本身就不太干净。

那直接录人手操作视频不就行了？人手抓东西快、自然、丝滑，30 段视频 3 分钟就录完。但有两个 gap：

**第一个 gap 叫 embodiment gap**。人手 21 个自由度，机器人 gripper 就 1 个自由度（开/合）。你录的是人手怎么动，机器人得执行的是 gripper 怎么动，这俩 motion space 完全不一样。

**第二个 gap 叫 observation gap**。你头戴 GoPro 录的是 egocentric view（从头顶往下看手），但机器人部署时相机装在 wrist 上，看的是 gripper 正前方的画面。policy 训的时候见的是 A 视角，部署时遇到的是 B 视角，直接崩。

之前的工作怎么解决这俩 gap？要么上多视角相机（RGB-D，多目），要么搞定制硬件（UMI 那个手持夹爪、DexCap 的 mocap 手套），要么训一个 generative model 把人手画面"翻译"成 robot 画面（RwoR 那种）。每种方案都有痛点：多视角相机谁家里没有，定制硬件又贵又难推广，generative model 不稳且没开源。

WARPED 想说：**就一个头戴 GoPro，啥别的都不要，能不能从人手视频直接产出 robot wrist-view 图像 + robot trajectory？** 能。

---

## 核心思路：把人手当"几何探针"

单目视频最大的问题是什么？是 depth。你从一段 2D 视频里想知道物体 3D pose，本身是 underconstrained 的。物体可以近一点小一点，也可以远一点大一点，投影到画面上长得一样。

WARPED 的核心 insight 是：**当人手抓起物体那一刻，手和物体之间就建立了一堆物理约束**——

- 手的指尖必须贴着物体表面（contact loss）
- 手不能穿进物体内部（collision loss）
- 抓住之后，手和物体的相对位置在移动过程中应该不变（stable grasp loss）
- 手不能穿桌子（scene TSDF loss）

这些约束加起来，就把物体的 6D pose 钉死了。你只要能 track 住手的 pose（用 HAMER 这个现成的 3D hand reconstruction 模型），再用手和物体的交互关系反推物体 pose，单目就够用。

这跟用 FoundationPose 那种纯 object tracker 的根本区别在于：FoundationPose 只看物体本身，物体被手挡住一半它就懵了；WARPED 在物体被手挡住的时候，可以从手的运动推物体在动什么。实验里 Can on Plate 任务，FoundationPose 2/20，WARPED 17/20，差距就是从这里来的。

---

## 整个 pipeline 像拍电影重剪

你可以把 WARPED 想成一个"电影重剪"系统：

**第一步：扫场景**。你拿 GoPro 在空桌面上方扫一圈，1 分钟。系统用 SfM 重建出桌面的 3D 几何，再训一个 Gaussian Splat（就是那种用一堆 3D 高斯椭球拟合场景的神经表示，渲染快、显式、可编辑）。

**第二步：录人手 demo**。戴头盔，自然地做任务 30 次。3 分钟。

**第三步：从 demo 里反解 3D 几何**。这是整个 pipeline 最重的一步：
- 用 HAMER 出每帧的手 pose 初始估计，再 refine（先粗调全局位姿，再细调关节角，避免优化掉进 local minima）
- 用 Grounding DINO + SAM2 + SAM3D + MegaPose 出物体的初始 mesh 和 pose
- 然后做那个 hand-object joint optimization：所有帧的手 pose 参数和物体 pose 参数一起优化，loss 包括 mask 对齐、depth 对齐、DINOv2 feature 对齐、contact 约束、collision 约束、stable grasp 约束、scene TSDF 约束等。这一步大概每段 demo 跑 4 分钟。

**第四步：retarget 到 robot**。人手映射到 gripper 有个简单规则：拇指尖和食指尖的中点对应 gripper 的 TCP，拇指根和食指根的中点对应 gripper base。接触之前 gripper 张开，trajectory 要做一个小优化防止撞物体（funnel loss，越接近接触时刻越严格贴近演示动作）；接触时刻找 50 个 contact point 把 gripper pose 和 width 钉死；接触之后 gripper 跟着物体刚性运动，公式就是 $\mathbf{T}^{ee}_t = \mathbf{T}^{ee}_{t_s} (\mathbf{T}^{obj}_{t_s})^{-1} \mathbf{T}^{obj}_t$，意思就是"物体怎么动，gripper 就怎么动"。

**第五步：重新"拍摄"**。这一步是 magic 所在。你有了 scene 的 Gaussian Splat、物体的 Gaussian Splat、gripper 的 3D 模型，你知道每帧 gripper 在哪、物体在哪，那就用 Gaussian Splatting 渲染器，从 wrist-camera 的视角，把整个场景重新渲染出来。渲染出来的是一张 photorealistic 的图，就跟真 robot 拍的一样。30 段 demo 每段 100 帧 = 3000 张虚拟 wrist-view 图像 + 对应 action，这就成了一个 robot dataset。

**第六步：augmentation**。Gaussian Splat 显式表示的好处是可以随便改：换物体贴图、随机移动物体位置、随机化初始 gripper pose、随机缩放场景、扰动 wrist-camera 内参外参。每段 demo aug 10 次，30 段变 300 段。这一步对 sim-to-real 至关重要——Table I 里 no aug 的 WARPED 在 Rotate Box / Bottle / Wipe Brush 上全是 0/20，加了 aug 直接跳到 17-20/20。

**第七步：训 diffusion policy**。拿这堆虚拟 wrist-view 图像 + action，喂给标准的 diffusion policy（UMI 那套，CLIP ViT-B/16 做 vision encoder，50 步 denoising，AdamW，120 epochs）。输入图像加点 Gaussian noise 弥补渲染图和真实图的 gap。

---

## 为什么这套能 work

我觉得有三个关键设计让 WARPED 能 work：

**1. Hand-object joint optimization 把单目的 underconstrained 问题变 constrained 了**

单目视频里物体 pose 本身是模糊的，但加上"手在抓物体"这个事实，contact / collision / stable grasp 这些物理约束就把 pose 钉死了。这相当于把人手当成一个"已知运动学结构的探针"，手的 motion 给物体的 motion 提供了 relative 约束。

**2. Gaussian Splatting 让"wrist 重拍"变得可行**

同样的思路如果用 NeRF，每段 demo 渲染几千张图要跑半天。G.S. 训练快、渲染快、显式可控（能随便改物体位置/朝向/贴图），每次 augmentation 都要重渲染整个序列，工程上只有 G.S. 顶得住。而且 G.S. 还能渲染 fisheye（用 3DGUT 那个 undulating toroidal camera model），匹配真实 GoPro Max Lens Mod 的鱼眼镜头。

**3. Augmentation 摊平了 sim-to-real gap**

渲染图再真实也不是真的，和真实 GoPro 图像之间有 distribution gap。你直接拿渲染图训 policy 部署到真实 robot 会崩（no aug 的 0/20 就是证据）。Augmentation 的本质是"把 gap 摊开"——与其让 policy 见到一个固定的渲染分布然后 deployment 时遇到一个固定的真实分布，不如让 policy 见到各种乱七八糟的变体，部署时真实图像落在 policy 见过的某个变体附近就行。这个 trick 在 SplatSim、Rovi-Aug、NeRF-Aug 里都验证过，WARPED 把它用到了极致。

---

## 实验结果说人话

**主表（Table I）**：
- Rotate Box：WARPED 20/20，teleop 16/20。WARPED 反超，因为人手做 90 度旋转比 VR 手柄自然多了。
- Pour Mug / Bottle / Can：WARPED 17-18/20，teleop 16-19/20，打平。
- Wipe Brush：WARPED 11/20，最差。因为 brush 太小又平躺，单目 pose estimation 本身就难。

**Co-training**：15 段 WARPED + 15 段 teleop，4/5 任务打平或超过纯 teleop。WARPED 可以当 teleop 的加速器用。

**Novel object（Table II）**：WARPED 在 Rotate Box 和 Bottle 上明显胜过 teleop（10/10 vs 8/10, 8/10 vs 4/10）。因为 retexturing + scaling 的 aug 让 policy 学到更通用的 object representation，teleop 30 段就一个物体，遇到 novel 物体就崩。

**vs UMI（Table IV）**：
- Can on Plate：UMI 9+10/10，WARPED 8+9/10，UMI 略胜。圆柱形罐子视觉太一致，UMI 抓得准。
- Rotate Box：UMI 2+0/10，WARPED 10+10/10。**血洗**。UMI 手持夹爪做 90 度旋转容易打滑，人手天然适合做这种连续旋转。这实验我觉得最能说明 human demo 的价值。

**Data collection time（Table III）**：WARPED 3-5 min/task，teleop 15-32 min/task，5-8x speedup。但 WARPED 还有 offline processing 每 demo 7 min（主要是 hand-object optimization），全自动可并行。

---

## 这工作的真正价值

表面卖点是"5-8x faster data collection"，但我觉得真正价值是"**data collection 完全脱离 robot hardware**"。

你想，UMI 已经够 low-cost 了，但你还是要拖一个夹爪到处走，夹爪跟 robot 部署时的 gripper 还有机械差异。WARPED 只要一个 GoPro + 一个头盔，你在家都能录 demo。录完之后，整套 pipeline（hamer + sam2 + megapose + gaussian splatting + diffusion policy）都是开源组件拼起来的，跑在一张 3090 上。

这对小 lab、对农业 [47]、对定制任务场景的意义很大。你想给一个新任务训 policy，不用买 UMI 夹爪、不用搭 multi-view rig、不用训 generative model，戴个头盔录半小时，跑一晚上 pipeline，第二天就能 deploy。

**Limitations 也很明确**：只支持 rigid object、quasi-static scene、单物体操作、物体完全被手挡住时崩、小物体平躺时 pose estimation 难。这些 limitations 本质上都来自单目 + rigid 假设，扩展到 articulated object / deformable / dynamic scene 需要新的几何表示（dynamic Gaussian Splatting、articulated G.S. 等）。

---

## 我觉得最值得 follow 的方向

1. **Bimanual**：MANO 本来就是双手模型，HAMER 也支持双手。把 pipeline 扩到双手 demo + 双 gripper，bimanual manipulation 的 data 采集瓶颈就破了。
2. **Online refinement loop**：policy 部署后收集失败案例，用 WARPED pipeline 把这些失败视频也变成训练数据，闭环 self-improvement。
3. **替换 SfM + G.S. init 为 feed-forward 3D foundation model**：VGGT [95] 这种 feed-forward 3D transformer 可能直接从视频出 3D 表示，省掉 SfM 和 G.S. 训练那 8 分钟。
4. **Hand-object optimization 蒸馏成 feed-forward tracker**：现在 4 分钟优化一段 demo，如果能蒸馏成一个网络直接 forward 出 hand+object pose，整个 pipeline 就能走向 realtime。
5. **Cross-embodiment**：让 WARPED 同时产出多种 gripper（parallel jaw / suction / 3-finger）的 trajectory，policy 学一个 conditioned on gripper type 的 conditional policy，一次 demo 训多 robot。

总之 WARPED 是个工程整合度极高的工作，每个组件都不是新的，但组合方式很巧。核心 insight 是"hand-object interaction 是单目场景的几何锚点"，这个想法我觉得可以延伸到很多 beyond tabletop 的场景。

---

# WARPED: 从 egocentric human demo 到 wrist-view robot policy 的全流程拆解

## 1. 这篇 paper 在解决什么问题

Imitation learning 的瓶颈在 data。Teleoperation 慢、贵、精细动作难做（Rotate Box 任务里 teleop 只拿到 16/20 就是因为连续旋转很难用 VR controller 稳定控制）。Human demonstration 快、自然，但有两个 gap：
- **Embodiment gap**：人手有 21 DoF，机器人 gripper 只有 1 DoF（开合），关节空间完全不同。
- **Observation gap**：人 demo 是头戴相机拍的 egocentric view，robot 部署时通常是 wrist-mounted camera，视角差异巨大。

之前的工作要 bridge 这两个 gap，往往依赖多视角相机、深度传感器、定制硬件（UMI 的 gripper、DexCap 的 mocap 手套等），或者训练专用 generative model（RwoR [36]）。WARPED 的 selling point 是：**只用一个头戴 monocular RGB camera，端到端产出 wrist-view 图像 + robot trajectory**，直接喂给 diffusion policy。

论文链接：https://arxiv.org/abs/2507.20772 (推测，从作者和标题推断；正式版可能还没上)
作者主页（George Kantor group）：https://www.cmu.edu/roboticsinstitute/
相关 GitHub（暂未发布）：可关注 Harry Freeman / CMU RI 的 repo

---

## 2. Pipeline 全景

整个 pipeline 五阶段，对应 Fig. 2：

```
[Scene scan] → SfM + LightGlue → scene Gaussian Splat
[Human demo (egocentric)] → HLoc 定位 → SpatialTrackerV2 depth
                          → HAMER hand init → SAM2/SAM3D/MegaPose object init
                          → Hand-Object Joint Optimization (核心)
                          → Hand→EE retargeting + 3DGUT wrist-view rendering
                          → Diffusion policy training
```

Intuition 上：把人手 demo 当作"几何探针"。人手抓物体时，物体 pose 和手 pose 互为约束，可以从单目视频里反解出 6D 轨迹。然后把这个轨迹 retarget 到 robot gripper，并从 wrist-camera 视角"重新拍摄"出来。

---

## 3. Stage 1: Data Collection

### 3.1 静态场景扫描
先在没物体的状态下扫一段 workspace 视频（< 1 min），用 SfM（LightGlue [60] 做特征匹配）恢复 sparse 3D 点云 + camera poses。这个 sparse 重建后续用来做：
- 初始化 scene-level Gaussian Splat
- 提供 SfM 2D-3D correspondences 给 demo frames 做 localization

### 3.2 Demo 录制
GoPro Hero9 装头盔上，30 Hz，pinhole 模型。每个任务录 30 段 demo。Table III 显示每段 demo + scan 总共只要 3-5 min，而 teleop 要 15-32 min，**5-8x speedup**。

Intuition：UMI 的 scan 步骤是从这里借鉴的，但 UMI 之后还要拖一个夹爪到处走，WARPED 只需要一个头盔。

---

## 4. Stage 2: Interactive Scene Initialization

### 4.1 Demo frame localization & depth
用 HLoc [83] 把每帧 demo 注册到 SfM 模型，得到 per-frame camera pose。SpatialTrackerV2 [100] 出 monocular depth maps $\hat{\mathcal{D}}$。

### 4.2 Scene-level scale alignment
SfM 有 gauge ambiguity（global scale 不可观），而 monocular depth 也有未知 scale。两者之间用 affine 对齐：

$$z^{sfm} \approx A + B z^{pred}$$

- $z^{sfm}$: SfM 3D 点在 camera frame 下的 z 值
- $z^{pred}$: SpatialTrackerV2 预测深度
- $A, B$: offset 和 scale

求解用 Huber loss（对 outlier 鲁棒）的非线性最小二乘（Eq. 17）。如果场景里放了 fiducial marker（已知尺寸），再二次校正 scale。

Intuition：单目深度只有相对深度对，SfM 只有结构对但 scale 漂浮，两者一拼刚好能拿到 metric scale。

### 4.3 Hand pose initialization
HAMER [74] 是 transformer-based 3D hand reconstruction，per-frame 出 MANO 参数 $\theta, \beta$。但 HAMER 没用 temporal info，会抖。两阶段 refine：

**Stage 1 (Coarse)**：固定 $\theta$，只优化 global rotation $\mathbf{R}^{hand}$、translation $\mathbf{t}^{hand}$、shape $\beta$，loss 是 2D keypoint + smoothness。

$$\mathcal{L}_{smooth} = \sum_t \sum_{\mathbf{v}^h \in V^{hand}} \|\mathbf{v}_t^h - \mathbf{v}_{t-1}^h\|_2^2$$

**Stage 2 (Fine)**：放开 $\theta$，加 depth supervision $\mathcal{L}_{\mathcal{D}_{hand}}$（和后面 Eq. 9 一样）。

Intuition：先把粗的全局位姿锁定再加细节，否则优化容易掉到 local minima（这是 hand pose 优化里的经典 trick，参考 [35, 73]）。

### 4.4 Object pose initialization
- **Grounding DINO** [61] 用文本描述检测物体（first frame）
- **SAM2** [79] 分割并 propagate mask 到整个 sequence
- **SAM3D** [91] 从 mask 重建 mesh
- 但 SAM3D 自己产出的 Gaussian Splat 质量不够，作者用 mesh 渲染多视角图像，再训自己的 object Gaussian Splat
- **MegaPose** [50] 用 mesh + first frame mask 给初始 6D pose

然后估计 contact start frame $\tilde{t}_s$（mask 之间 overlap 阈值），在 contact 之前的帧优化 object pose + scale，假设物体静止：

$$\min_{\mathbf{R}_0^{obj}, \mathbf{t}_0^{obj}, s^{obj}} \lambda_{\mathcal{M}_{obj}} \mathcal{L}_{\mathcal{M}_{obj}} + \lambda_{\mathcal{D}_{obj}} \mathcal{L}_{\mathcal{D}_{obj}}$$

object mesh 顶点变为 $V^{obj} = s^{obj} V^{obj*}$（$V^{obj*}$ 是 SAM3D 输出的 canonical mesh）。

---

## 5. Stage 3: Hand-Object Optimization（核心创新）

这是 paper 的技术核心。两个 sub-stage：

### 5.1 Per-frame Object Pose Estimation
对每帧用 differentiable rasterizer $\mathcal{R}$（nvdiffrast [51]）渲染 object 的 RGB、mask、depth：

$$\mathcal{T}_t^{obj}, \mathcal{M}_t^{obj}, \mathcal{D}_t^{obj} = \mathcal{R}(\mathbf{R}_t^{obj} V^{obj} + \mathbf{t}_t^{obj}, F^{obj})$$

- $\mathbf{R}_t^{obj}, \mathbf{t}_t^{obj}$: 当前帧 object 的 rotation (3×3) 和 translation (3,)
- $V^{obj}, F^{obj}$: object mesh 顶点和面

三个 loss：

**Occlusion-aware mask loss** (Eq. 2)：
$$\mathcal{L}_{\mathcal{M}_{obj}} = \|(\mathcal{M}_t^{obj} - \hat{\mathcal{M}}_t^{obj}) \odot (1 - \hat{\mathcal{M}}_t^{hand})\|$$

- $\hat{\mathcal{M}}_t^{obj}$: SAM2 预测的 object mask (ground truth)
- $\hat{\mathcal{M}}_t^{hand}$: 预测的 hand mask
- $\odot$: 逐元素乘
- $(1 - \hat{\mathcal{M}}_t^{hand})$ 是个"非手区域"权重，意思是"被手遮挡的部分不算 mask loss 的错"

Intuition：当手盖住物体一部分时，SAM2 也看不清，没必要 penalize 渲染 mask 和这个不准的 SAM2 mask 的差异。

**Depth loss** (Eq. 3)：
$$\mathcal{L}_{\mathcal{D}_{obj}} = \|(\mathcal{D}_t^{obj} - \hat{\mathcal{D}}_t^{obj}) \odot (1 - \hat{\mathcal{M}}_t^{hand})\|_2^2$$

Intuition：单纯 mask loss 有歧义——很多 pose 都能渲染出相同 mask（比如一个 box 正面和侧面在某些视角下 mask 形状一样）。Depth 把 z 信息塞进来消歧。同样在 hand 区域不计 loss。

**DINOv2 feature loss** (Eq. 4)：
$$\mathcal{L}_{DINO} = \frac{1}{P}\sum_{p=1}^{P}\left(1 - \frac{\mathcal{F}_{t,p} \cdot \hat{\mathcal{F}}_{t,p}}{\|\mathcal{F}_{t,p}\|_2 \|\hat{\mathcal{F}}_{t,p}\|_2}\right)$$

- $\mathcal{F}_t = \mathcal{G}(\mathcal{T}_t)$: 渲染图过 DINOv2 ViT-S 第 9 层特征
- $\hat{\mathcal{F}}_t = \mathcal{G}(\hat{\mathcal{T}}_t \odot \mathcal{M}_t^{obj})$: 原图 mask 后过 DINOv2
- $P$: 像素数
- cosine distance 的形式

Intuition：DINOv2 feature 对视角、光照相对鲁棒，能提供 texture-level supervision，弥补 monocular depth 的噪声。这是 NVR-style 重建里很关键的 trick（参考 NeRF-III、GARField 等）。

最终优化：
$$\min_{\mathbf{R}_t^{obj}, \mathbf{t}_t^{obj}} \lambda_{\mathcal{M}_{obj}} \mathcal{L}_{\mathcal{M}_{obj}} + \lambda_{\mathcal{D}_{obj}} \mathcal{L}_{\mathcal{D}_{obj}} + \lambda_{DINO} \mathcal{L}_{DINO}$$

每帧独立优化，用前一帧结果做 init。然后通过 translation/rotation 变化阈值检测 contact start $t_s$ 和 contact end $t_e$（Eq. 22, 23），连续 $m$ 帧超过阈值才算 $t_s$，连续 $m$ 帧低于阈值才算 $t_e$，这是为了滤掉抖动。

### 5.2 Joint Hand-Object Refinement（关键）
独立估计容易错，因为 hand 和 object 互为遮挡互为约束。Joint optimization 同时优化：
- Object: $\Theta^{obj} = \{\mathbf{R}^{obj}, \mathbf{t}^{obj}, s^{obj}\}$（所有帧 + 全局 scale）
- Hand: $\Theta^{hand} = \{\bar{\mathbf{R}}^{hand}, \mathbf{t}^{hand}, \theta, \beta\}$（MANO 参数，shape $\beta$ 全序列共享）

约束：object 在 $t \le t_s$ 和 $t \ge t_e$ 时静止。

Loss 组合：

**(1) Mutual occlusion-aware mask loss** (Eq. 8)：
$$\mathcal{L}_{\mathcal{M}_{obj}} = \|(\mathcal{M}^{obj} - \hat{\mathcal{M}}^{obj}) \odot (1 - \hat{\mathcal{M}}^{hand})\|$$
$$\mathcal{L}_{\mathcal{M}_{hand}} = \|(\mathcal{M}^{hand} - \hat{\mathcal{M}}^{hand}) \odot (1 - \hat{\mathcal{M}}^{obj})\|$$

两个 mask 互相 mask，互不干扰对方的"盲区"。

**(2) Depth loss** (Eq. 9)：同上，object 和 hand 各一份。

**(3) Contact loss** (Eq. 10)：
$$\mathcal{L}_{contact}(t) = \begin{cases} \sum_{\mathbf{v}^\tau \in V^{tip}} \min_{\mathbf{v}^o \in V^{obj}} \|\mathbf{v}_t^\tau - \mathbf{v}_t^o\|_2^2, & t_s \le t \le t_e \\ 0, & \text{otherwise}\end{cases}$$

- $V^{tip} \subset V^{hand}$: 经常接触物体的手部顶点（thumb + index fingertip 等，Fig. 4a）
- $\mathbf{v}_t^\tau, \mathbf{v}_t^o$: 时刻 $t$ 的手顶点和 object 顶点位置

Intuition：接触期间，fingertip 应该贴近 object 表面。这是把"接触"这个物理事实塞进优化的方式。参考了 Hasson et al. [34] 的 PHOSA 工作。

**(4) Collision loss** (Eq. 11)：
$$\mathcal{L}_{col} = \sum_{\mathbf{v}^h \in V^{hand}} \Phi^{obj}(\mathbf{v}^h)$$

- $\Phi^{obj}(\cdot)$: object mesh 的 Truncated Signed Distance Field (TSDF)，物体内部为负、外部为正、表面附近为 0
- 只对落在 object 内部（$\Phi < 0$）的 hand vertex 惩罚

Intuition：手不能穿物体。TSDF 是把"不可穿透"变成可微分约束的标准方法。

**(5) Stable grasp loss** (Eq. 12)：
$$\mathcal{L}_{sg} = \sum_{\mathbf{v}^\tau} \sum_{\mathbf{v}^o} \sum_{n=t_s}^{t_e} \sum_{m=t_s}^{t_e} \|d_n^{o\tau} - d_m^{o\tau}\|$$
$$d_n^{o\tau} := \|\mathbf{v}_n^\tau - \mathbf{v}_n^o\|_2^2$$

- $d_n^{o\tau}$: 时刻 $n$ 时 fingertip vertex $\tau$ 到 object vertex $o$ 的距离平方
- 整个 contact 期间，fingertip 到对应 object vertex 的距离应保持稳定（grasp 不松）

Intuition：接触期间物体跟着手走，相对距离不变。这个 loss 强制实施"rigid grasp"假设，让后续 EE pose 可以直接 copy object motion。

**(6) Auxiliary losses**：
- **Scene TSDF loss** (Eq. 24)：手和物体不能穿桌子
- **Resting-on-scene loss** (Eq. 25)：非 contact 期间物体要贴着桌面（防止 floating artifact）
- **2D projection loss**：手 projected vertex 对齐 HAMER 2D keypoint
- **Hand pose regularization** (Eq. 26)：$\mathcal{L}_{hp} = \|\theta - \hat{\theta}\|$，防止 MANO pose 飘到不合理区域

总优化（Eq. 13）：
$$\min_\Theta \lambda_{\mathcal{M}}\mathcal{L}_{\mathcal{M}} + \lambda_{\mathcal{D}}\mathcal{L}_{\mathcal{D}} + \lambda_c\mathcal{L}_{contact} + \lambda_{col}\mathcal{L}_{col} + \lambda_{sg}\mathcal{L}_{sg} + \mathcal{L}_{aux}$$

$\Theta = \{\Theta^{obj}, \Theta^{hand}\}$，**所有帧同时优化**。

**为什么这个 joint optimization 比 FoundationPose 强？**
实验对比（Table I 旁边讨论）：FoundationPose [96] 在 Can on Plate 上只有 2/20，WARPED 是 17/20。原因是 FoundationPose 是 single-object tracker，没有 hand 信息，物体小且被手遮挡就崩。Hand-object joint optimization 用 hand 的 motion 约束补全了 occlusion 期间的 object pose（接触期间 object 跟 hand 走）。

---

## 6. Stage 4: Retargeting & Rendering

### 6.1 Hand → End-effector mapping (Appendix F)
Fig. 10：
- **TCP** = thumb tip 和 index tip 的中点 $\mathrm{H}_{tcp} = (\mathrm{Th}_{tip} + \mathrm{Ind}_{tip})/2$
- **Base** = thumb MCP 和 index MCP 的中点 $\mathrm{H}_{base} = (\mathrm{Th}_{mcp} + \mathrm{Ind}_{mcp})/2$
- **Gripper axis** = thumb tip → index tip 方向向量

xArm G1 gripper 的 TCP 和 base 之间距离是固定的，所以 hand → EE 的 mapping 不是 1:1，是把"两指尖中点"和"两指根中点"对齐到 gripper 的对应位置。

### 6.2 Pre-contact trajectory optimization (Eq. 14, 28-31)
$$\min_{\mathbf{T}_{t<t_s}^{ee}} \lambda_{funnel}\mathcal{L}_{funnel} + \lambda_{col}\mathcal{L}_{col} + \lambda_{smooth}\mathcal{L}_{smooth}$$

- **Funnel loss** (Eq. 28)：$\sum_t w_t \|\mathbf{t}_t^{ee} - \hat{\mathbf{t}}_t^{ee}\|_2^2$
- **Weights** (Eq. 29)：$w_t = w_{min} + (w_{max} - w_{min})(t/(T-1))^3$，三次方增长，越接近 contact 越严格
- **Collision loss** (Eq. 30)：用 object TSDF，penalize EE vertex 穿物体
- **Smoothness loss** (Eq. 31)：translation + rotation（log map 形式）的相邻帧差

Intuition：人手 retarget 到 gripper 后，gripper 比手"宽"，可能撞物体。Funnel loss 让 trajectory 在 contact 前必须收回到接近"演示动作"，但允许早期偏移以避开 collision。三次方权重是 trick：早期松、晚期紧，避免 collision 时只能改早期，晚期已经锁死接近演示。

### 6.3 Contact grasp refinement (Eq. 32-33)
在 $t_s$ 帧，找 50 个 contact points（thumb 侧 + index 侧），优化 EE pose 和 gripper width：
$$\min_{\mathbf{T}_{t_s}^{ee}, g_{t_s}} \lambda_{contact}\mathcal{L}_{contact} + \lambda_{width}\mathcal{L}_{width} + \lambda_{col}\mathcal{L}_{col}$$
- $\mathcal{L}_{contact} = \sum_{\mathbf{v}^{ee}} \min_{\mathbf{v}^c} \|\mathbf{v}^{ee} - \mathbf{v}^c\|_2^2$：gripper 顶点贴近 contact points
- $\mathcal{L}_{width} = g_{t_s}$：gripper 尽量闭合（别太开）
- $\mathcal{L}_{col}$：gripper 不穿物体

### 6.4 Contact 期间 EE 跟随 object (Eq. 15)
$$\mathbf{T}_t^{ee} = \mathbf{T}_{t_s}^{ee} (\mathbf{T}_{t_s}^{obj})^{-1} \mathbf{T}_t^{obj}$$

- $\mathbf{T}_{t_s}^{ee}, \mathbf{T}_{t_s}^{obj}$: contact 开始时的 EE 和 object pose
- $\mathbf{T}_t^{obj}$: 当前帧 object pose
- $(\mathbf{T}_{t_s}^{obj})^{-1} \mathbf{T}_t^{obj}$: object 相对 contact frame 的相对运动
- 左乘 $\mathbf{T}_{t_s}^{ee}$：把这个相对运动施加到 EE 上

这是 rigid grasp 假设的直接体现：EE 相对 object 在 contact 时锁定，之后 EE pose = initial EE pose × relative object motion。

### 6.5 Wrist-view rendering
用 Nerfstudio 的 3DGUT [99] 实现（Gaussian Splatting + Undulating Toroidal camera model）渲染 fisheye 图像（GoPro Max Lens Mod 是鱼眼）。三个 Gaussian Splat 组合：
1. Scene（静态桌面）
2. Object（per-frame pose）
3. End-effector（gripper 模型 + pose）

**为什么 Gaussian Splatting 不用 NeRF？** 因为 G.S. 渲染快、训练快、显式表达，每次 augmentation 都要重渲染很多帧，G.S. 是工程上唯一可行的选择。

---

## 7. Stage 5: Policy Training

- **Diffusion policy**（UMI [17] 实现，参考 https://diffusion-policy.cs.columbia.edu/）
- Vision encoder: CLIP ViT-B/16 (image 224×224)
- 50 denoising steps
- AdamW, lr 3e-4, cosine decay, 120 epochs, batch 64×4 GPU
- Input: 2 wrist images (Img-H=2) + 2 proprioception frames
- Proprio: relative EEF xyz + 6D rotation + binary gripper
- Output: action chunk (8-12 steps)

### 7.1 Sim-to-real bridging
对 input images 加 Gaussian noise（参考 [10, 78]）。Rendered 图像和真实 GoPro 图像之间有 domain gap，noise injection 让 policy 学到对图像扰动鲁棒的表示。

### 7.2 Data Augmentation（关键！）
Fig. 6 展示五种：
1. **Retexturing** object mesh（用 Trellis 重新生成 texture）
2. **Random object translation** in scene
3. **Random initial gripper pose**
4. **Random scene scaling**
5. **Perturb wrist-camera intrinsics/extrinsics**

每个 demo augment 10 次，30 demos → 300 effective trajectories。

**为什么 aug 这么重要？** Table I 里 "WARPED (no aug)" 在 Rotate Box 上 0/20、Bottle 0/20、Wipe Brush 0/20，Can on Plate 8/20 → 17/20（翻倍多）。Augmentation 把 rendered 图像和真实图像之间的 distribution gap 给"摊开"了，让 policy 看到各种变体，部署时不容易因为小差异崩。

这点和 UMI 对比是个关键差异：UMI 没有这种 augmentation 框架，所以 30 demos 就是 30 demos，而 WARPED 30 demos × 10 aug = 300。这也是 WARPED 在 Rotate Box 上完胜 UMI（10/10 vs 2/10）的原因之一。

---

## 8. 实验结果深入解读

### 8.1 主表（Table I）

| Method | Rotate | Pour | Bottle | Wipe | Can |
|---|---|---|---|---|---|
| Teleop | 16/20 | 19/20 | 16/20 | 15/20 | 19/20 |
| Alter [36] | 7/20 | 3/20 | 0/20 | 0/20 | 8/20 |
| WARPED no aug | 0/20 | 17/20 | 0/20 | 0/20 | 8/20 |
| WARPED bg distractor | 18/20 | 15/20 | 17/20 | 9/20 | 17/20 |
| WARPED | **20/20** | 18/20 | 17/20 | 11/20 | 17/20 |
| Teleop + WARPED | 19/20 | **20/20** | 17/20 | 11/20 | **20/20** |

观察：
- **Rotate Box**：WARPED 比 teleop 好（20 vs 16）。Teleop 做 90° 旋转难，人手自然做。这验证了 human demo 在精细 motion 上的优势。
- **Wipe Brush**：WARPED 最差（11/20）。原因 paper 给了：brush 小且平躺在桌面，pose estimation 难，retargeting 难。这也是 WARPED 的 inherent limitation：单目对小物体、低对比度物体的 6D pose 容易崩。
- **Co-training**：4/5 任务上达到或超过 teleop，证明 WARPED 数据可以无缝补充 teleop。

### 8.2 Novel object generalization（Table II）
WARPED 在 Rotate Box 和 Bottle 上明显胜过 teleop（10/10 vs 8/10, 8/10 vs 4/10）。这要归功于 retexturing + scaling augmentation 让 policy 学到更 general 的 object representation。Teleop 没有 augmentation，30 demos 都是一个物体，novel object 一来就崩。

### 8.3 OOD scene（Can on Plate）
50 demos across 20 scenes，eval on 4 unseen scenes，16/20 成功。作者说 "performance could be improved by integrating scene-level augmentation"，暗指 scene Gaussian Splat 的 augmentation 可以加。

### 8.4 Background distractors（Fig. 9）
在 Can on Plate 和 Bottle 上几乎无影响（17/20 vs 17/20；17/20 vs 17/20），Pour Mug 掉 3 个（18→15），Rotate 掉 2 个（20→18），Wipe 掉 2 个。结果相当 robust，因为 wrist camera 视野窄，背景 distractor 多在视野边缘。

### 8.5 vs UMI（Table IV）
- **Can on Plate**：UMI 9/10+10/10 (训练+novel)，WARPED 8/10+9/10。差不多，UMI 略胜（因为 can 圆柱体视觉一致性好，UMI 抓得准）。
- **Rotate Box**：UMI 2/10+0/10，WARPED 10/10+10/10。**完全碾压**。原因是 UMI 用一个 handheld gripper 模拟 end-effector，做 90° 旋转时容易打滑；人手天然适合做这种连续旋转，加上 WARPED 的 object retexturing augmentation 让 policy 泛化到 novel box。

### 8.6 vs FoundationPose
- Rotate Box: WARPED 17/20 vs FoundationPose 11/20
- Can on Plate: WARPED 17/20 vs FoundationPose 2/20

FoundationPose 在小物体 + 手遮挡场景崩，hand-object joint optimization 利用了 hand 的 motion 信息补全 occlusion。

### 8.7 Data collection efficiency（Table III）
WARPED 全部任务 3-5 min，teleop 15-32 min。**5-8x speedup**。但要注意 WARPED 还有 offline processing time（Table VII）：每 demo 处理约 7 min（CPU AMD 7950X + RTX 3090），但这部分全自动、可并行，不需要人在场。

### 8.8 Processing time breakdown（Table VII）
- Scene reconstruction: 8:20 (一次性)
- Object mesh reconstruction: 1:48 (一次性)
- Per demo: 6:57
  - Localization: 33s
  - Depth prediction: 20s
  - Hand init: 45s
  - Object init: 48s
  - Hand-object optimization: 3:52（占大头）
  - Retargeting + rendering: 33s
- Per augmentation: 18s（aug 10 次也就 3 min）

主要时间在 hand-object joint optimization，这是可优化空间最大的部分（GPU 加速、batch 优化、蒸馏等）。

---

## 9. Related work 对比

### 9.1 View synthesis 方向
- **WristWorld** [76]：用 VGGT 从 third-person 输入生成 wrist view。需要 third-person camera，WARPED 只要 egocentric。
- **Imagination at Inference** [21]：fine-tune ZeroNVS 在 inference 时合 wrist view。要训 generative model，WARPED 用 explicit geometry 不用 generative model。
- **RwoR** [36]：手戴 GoPro 录 demo，用 generative model 把 wrist-hand view 转成 robot wrist view。还是依赖 generative model，没开源。

### 9.2 Cross-embodiment imitation
- **Phantom** [56] / **Masquerade** [55]：用 inpainting 把人手换成 robot gripper。简单但失去 3D 几何，难处理 occlusion。
- **Mirage** [13] / **ROVI-Aug** [14]：cross-painting 思路。
- **Ditto** [37] / **Track2Act** [6]：trajectory transformation。

### 9.3 Data augmentation via neural rendering
- **Real2Render2Real** [109]：用 Gaussian Splatting 把少量 demo 增广，需要 robot hardware 录初始 demo。
- **NeRF-Aug** [121] / **RVT-Aug** [104]：类似思路。
- **SplatSim** [78]：zero-shot sim2real with G.S.

WARPED 的独特之处：**完全不需要 robot hardware 录任何数据**，从人 demo 端到端到 policy。

### 9.4 Egocentric demo learning
- **EgoMimic** [43] / **EgoZero** [62] / **EgoMI** [110]：从智能眼镜或 egocentric camera 学 policy。EgoZero 直接用智能眼镜出 action，但限于桌面简单任务；EgoMimic 用 egocentric 做 high-level plan，仍要 teleop 数据训练 low-level。
- **UMI** [17]：手持夹爪录 demo。WARPED 借鉴了 UMI 的 scan 流程和 diffusion policy 架构，但去掉了手持夹爪这一硬件需求。

---

## 10. Limitations 和未来方向

### 明确列出的 limitations
1. **Rigid object only**：articulated / deformable 不支持。作者提到可以借鉴 [46, 109] 的 DINOv2 feature tracking for articulated motion，以及 Deformable G.S. [26, 105]。
2. **Quasi-static scene**：背景和未操作物体必须静止。Dynamic G.S. [97, 105] 可能是方向。
3. **完全 occlusion 时失败**：物体被手完全挡住时，hand-object optimization 也救不回来。
4. **小物体 + 平躺姿态难**：Wipe Brush 任务的失败率（11/20）就是证据。

### 没明说但我看出的 limitations
1. **MANO 模型假设**：HAMER 和 MANO 都假设手是标准人手形态，戴手套或者手部畸形会崩。
2. **Single object per demo**：Can on Plate 是少数 dual object 任务，且 plate 是静态的。如果任务涉及多物体动态交互（比如 stacking blocks），pipeline 要扩展。
3. **Gripper 类型固定**：xArm G1 是 parallel jaw，retarget 到 suction gripper 或 multi-finger hand 需要重新设计 mapping。
4. **Image resolution 224×224**：低分辨率可能限制精细任务。但 paper 用 CLIP ViT-B/16 是标准做法。
5. **Photo-realism gap**：Gaussian Splatting 渲染再好也不是真的，sim-to-real noise injection 只能 part mitigate。论文里 co-training 比 pure WARPED 还好，说明 rendered 图像和真实图像之间确实有不可忽略的 gap。

### 未来工作联想
- **Bimanual manipulation**：现在单手 demo + 单 gripper，扩展到双手需要 dual-hand tracking + dual-EE retargeting。MANO 本来就是双手模型，HAMER 也支持。
- **Tactile sensor integration**：wrist camera 看不到 contact 区域，触觉传感器可以补 contact 期间的 force 信息。
- **Long-horizon task**：现在每段 demo 是一个 atomic skill，可以加 hierarchical policy / VLM 把多个 WARPED-trained skills 串起来。
- **In-the-wild scene scaling**：现在限定 tabletop，去到 kitchen / agriculture [47] 等场景需要 dynamic scene reconstruction + 多物体 tracking。
- **Online refinement**：policy 部署后收集失败案例，用 WARPED pipeline 把失败 demo 也变成训练数据，loop 起来。
- **Foundation model integration**：把 SAM2 + HAMER + DINOv2 + MegaPose 换成更新更强的（VGGT [95]、SAM3D、CuMo 等）可能进一步降 processing time。

---

## 11. 几个我特别注意的工程细节

### 11.1 SAM3D mesh vs 自己重建 Gaussian Splat
作者在 Sec. III-C3 特别提了一句：SAM3D 自己产出的 Gaussian Splat 不够好，他们用 SAM3D 的 mesh 渲染多视角图像，再训自己的 object Gaussian Splat。这是个很有意思的工程观察——SAM3D 的 G.S. representation 优化目标可能是 generic 重建，而 wrist-view rendering 需要在特定视角范围内高质量，所以从 mesh re-render 再训 G.S. 反而效果更好。这也暗示 SAM3D 的 G.S. 输出可能存在 view-dependent artifacts。

### 11.2 Three-stage scale alignment
Scene SfM (scale-free) → SpatialTrackerV2 depth (scale-free) → alignment via SfM 2D-3D correspondences → optional fiducial marker absolute scale。这个 multi-stage 对齐是单目系统的必修课，UMI 也用类似思路但用 fiducial marker 为主。WARPED 让 fiducial 变 optional 是为了 usability。

### 11.3 Two-stage hand pose optimization
Coarse (固定 θ) → Fine (放开 θ + depth supervision)。这种"先粗后细"的优化策略在 pose estimation 里很经典（SMPLify 也是先全局后细节），避免高维参数空间同时优化掉进 local minima。

### 11.4 Funnel loss 三次方权重
$w_t = w_{min} + (w_{max} - w_{min})(t/(T-1))^3$，三次方增长不是线性的。早期 t/T 接近 0，权重接近 $w_{min}$，trajectory 可以偏离演示；晚期 t/T 接近 1，权重接近 $w_{max}$，必须贴近演示。这种"晚期约束强"的设计是为了让 trajectory 在 contact 前一刻收敛到正确位置，方便后续 contact grasp refinement。参考自 Pan et al. One Demo is Worth a Thousand Trajectories [70]。

### 11.5 Wrist-camera perturbation augmentation
随机扰动 wrist-camera intrinsics/extrinsics。这是非常关键的 augmentation：渲染时 wrist camera 是虚拟的，pose 精确；部署时真实 wrist camera 安装有误差，intrinsics 也不完全一致。Perturb 这些参数让 policy 对 camera mounting variation 鲁棒。这个 trick 在 Rovi-Aug [14]、SplatSim [78] 里也用过。

---

## 12. 一些 web links 参考

- **WARPED 项目（推测）**：作者 Harry Freeman 在 CMU RI，关注 https://www.cs.cmu.edu/~hfreeman/ 或 CMU RI publications
- **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu/
- **UMI (Universal Manipulation Interface)**: https://universal-manipulation-interface.github.io/
- **Gaussian Splatting 原始 paper**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **Nerfstudio (3DGUT 实现)**: https://nerfstudio.io/
- **HAMER (3D hand reconstruction)**: https://geopavlakos.github.io/hamer/
- **MANO hand model**: https://mano.is.tuebingen.mpg.de/
- **SAM2**: https://github.com/facebookresearch/sam2
- **Grounding DINO**: https://github.com/IDEA-Research/GroundingDINO
- **MegaPose**: https://github.com/megapose6d/megapose6d
- **FoundationPose**: https://github.com/Project-Splinter/FoundationPose
- **SpatialTrackerV2**: https://github.com/xdspacelab/SpatialTracker-V2
- **DINOv2**: https://github.com/facebookresearch/dinov2
- **VGGT (相关 view synthesis 方向)**: https://vgg-t.github.io/
- **LightGlue**: https://github.com/cvg/LightGlue
- **HLoc**: https://github.com/cvg/Hierarchical-Localization
- **Open X-Embodiment**: https://robotics-transformer-x.github.io/
- **π0 (VLA model, 相关 imitation learning 大方向)**: https://arxiv.org/abs/2410.24164
- **Phantom (类似工作, Stanford)**: https://arxiv.org/abs/2503.00779
- **Masquerade**: https://arxiv.org/abs/2508.09976
- **RwoR**: https://arxiv.org/abs/2507.03930
- **Real2Render2Real**: https://r2r2.github.io/
- **EgoMimic**: https://egomimic.github.io/
- **EgoZero**: https://arxiv.org/abs/2505.20290
- **WristWorld**: https://arxiv.org/abs/2510.07313
- **Imagination at Inference**: https://arxiv.org/abs/2509.15717
- **SplatSim**: https://splatsim.github.io/
- **One Demo is Worth a Thousand Trajectories (funnel loss 来源)**: https://arxiv.org/abs/2510.01607 或者 follow Chuer Pan 的工作

---

## 13. 总体评价

WARPED 是一个工程整合度非常高的工作，把 SfM + monocular depth + hand pose + object pose + Gaussian Splatting + differentiable rendering + diffusion policy 全部串起来，每个组件都不是新发明的，但组合方式很巧妙。真正的 contribution 在于：

1. **用 hand-object interaction 作为单目场景的几何锚点**——这是 paper 最核心的 insight。单目本身 underconstrained，但 hand 和 object 之间的 contact、collision、stable grasp 约束把 DOF 大大收紧。
2. **从 rendered wrist view 直接训 policy**——bypass 了 generative model 的不稳定，用 explicit geometry 保证 consistency。
3. **Augmentation 策略的 design**——把 G.S. 的 explicit representation 优势用足，每种 aug 都有 sim-to-real 的 motivation。

**最值得 follow 的方向**：
- 把这套 pipeline 用在 bimanual + articulated object（drawer、cabinet、scissors）
- 把 hand-object optimization 蒸馏成 real-time tracker，让 pipeline 从 offline 走向 online
- 把 VGGT [95] 或更前沿的 feed-forward 3D foundation model 替换 SfM + Gaussian Splatting，可能省掉很多 init 步骤
- Cross-embodiment：让 WARPED 直接产出多种 gripper（parallel jaw + suction + 3-finger）的 trajectory，policy 学一个 conditional on gripper type

paper 的 sell 是"5-8x faster data collection"，但我觉得真正的价值是"zero robot hardware during data collection"——你只要一个 GoPro 和一个头盔，就能从零开始训一个 tabletop manipulation policy。如果未来扩展到 mobile manipulation 或者 bimanual，这个 reduction in barrier 意义很大。
