---
source_pdf: SpatialVID.pdf
paper_sha256: 1da3a57c0cafab7a2c36b721f21c873a29204c5d94ef959864d89a295d0df672
processed_at: '2026-08-12T09:44:33-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SpatialVID 的人话版本

Andrej，我用最直白的方式再讲一遍，重点讲清楚**它为什么这么做、为什么这么设计、为什么 work**。

---

## 一句话总结

**从 YouTube 抓了 2 万多小时视频，用一套 pipeline 把它们变成带 camera pose + depth + 精致 caption 的训练数据，给 spatial intelligence / world simulator 这个方向铺数据基础。**

核心 idea 很简单：**互联网上 video 海量，但带 3D 标注的 video 极少**。那我们不去手动标，而是用现有的几何估计 model（MegaSaM）批量"反向标"，再用 VLM+LLM 把语义补齐。这是一条 data factory 流水线，product 就是训练数据。

---

## 为什么这事值得做？

现在的 spatial intelligence 领域，data 处在一个尴尬的 split state：

**左边的 video dataset**（Panda-70M、MiraData）：几千万 clip，semantic 丰富，但是 **zero geometry**。没有 camera pose，没有 depth。模型只能从 pixel 里 implicit 学"这是不是 3D 一致的"，所以 Sora 这类模型生成视频时候镜头一动就破功。

**右边的 spatial dataset**（CO3Dv2、RealEstate10K、ScanNet）：geometry 精确，但是 **scale 小、scene 静态、要么是 synthetic**。RealEstate10K 8 万 clip 听起来还行，但都是从 Zillow 房产视频来的，全是 indoor walkthrough，scene 类型极度单一。CO3Dv2 是 object-centric 的转圈视频。ScanNet 是室内 SLAM 扫描。

**中间是空的**：real-world、dynamic、scale 大、既有 geometry 又有 semantic 的 dataset 不存在。

SpatialVID 就是来填这个洞的。Table 1 里一目了然：之前最大的 real dynamic + pose + depth 是 Stereo4d（11 万 clip），SpatialVID 直接干到 **270 万 clip**，scale 拉大 25 倍。

参考 Stereo4d: https://arxiv.org/abs/2412.09621

---

## 整个 pipeline 用大白话讲

```
YouTube (21,789 小时)
   ↓ 手动筛 + keyword query (walk/tour/drone)
33,443 个 raw video
   ↓ PySceneDetect 切成 3-15s clip + 编码标准化
700 万+ raw clip
   ↓ 四道 filter（美学/亮度/文字/运动）
270 万 clip = SpatialVID (7,089 小时)
   ↓ 类别 + motion 平衡采样
37 万 clip = SpatialVID-HQ (1,111 小时)
```

每一步都有它的"小心机"：

### 第一步：为什么从 YouTube 自己抓，而不是用现成的 Panda-70M？

作者做了一个很 Karpathy-style 的实验：把 Panda-70M 的 validation split 喂进自己的 pipeline，**只有 10% 能过 quality gate**。原因：Panda-70M 大量是 static camera、flickering artifacts、caption 没有运动描述。

所以只能自己从 YouTube 抓，用 `walk`、`tour`、`drone` 这类 keyword query 天然 filter 出 motion-rich video。**这是 motion-first curation**，先把"有 motion"这个先验写进 data source。

### 第二步：四道 filter 是什么意思？

**Aesthetics**：用 LAION-5B 训的 CLIP+MLP 打分（0-10），低于 4 分扔掉。就是过滤掉丑的、糊的、low-effort 的。

**Luminance**：用 Rec. 709 公式算每帧亮度：

$$L = 0.2126 R + 0.7152 G + 0.0722 B$$

- $R, G, B$：red / green / blue channel 值（8-bit 0-255 或归一化 0-1）
- 三个系数是 ITU-R BT.709 standard 定义的人眼对三色感知权重
- green 权重最大（0.7152）因为人眼对绿色最敏感
- 阈值 [20, 140]——非常保守，只留亮度区间中间 47% 的 clip

**OCR**：用 PaddleOCR 检测文字区域，文字占比 > 30% 扔掉。滤掉新闻、documentary、带大字幕的 Vlog。

**Motion**：用 VMAF（Netflix 的 perceptual video quality metric，FFmpeg 自带）作为 motion 强度 proxy。VMAF 本来是测画质，但作者挪用为 motion score（2.0-14.0 区间有效）。

这四道 filter 加起来，把 700 万砍到 270 万。**留下的都是"好看 + 不太亮不太暗 + 没字幕 + 有 motion"的 clip**。

### 第三步：几何标注——这是 paper 最技术的地方

用 **MegaSaM** 做 camera pose 估计。MegaSaM 是 Google 2024 年的工作，专门处理 casual dynamic video（手抖、物体动、非 professional 拍摄）。作者为什么选它？

对比实验（Fig. 10）说得很清楚：
- DROID-SLAM：dynamic 场景容易跑飞
- COLMAP：feature-sparse 场景直接 fail
- Fast3R：快但 robustness 不够
- MonST3R：准但慢
- VGGT：feature-sparse 场景不稳
- **MegaSaM：accuracy + speed + robustness 最好的平衡**

但是 MegaSaM 原版有几个问题，作者做了三处改造：

**改造 1：换 depth module**
原版依赖 external monocular depth，作者换成 **UniDepth v2 + Depth Anything v2** 组合：
- UniDepth v2 做 metric depth（绝对尺度）
- Depth Anything v2 做 dense relative depth（边缘细节）
两者互补，在 dominant moving object、collinear motion、varying focal length 这些 corner case 上 robustness 大幅提升。

**改造 2：dynamic mask 用 SAM2**
传统做法（光流阈值、motion probability）对复杂场景失效。作者做法：
1. adaptive thresholding + contour detection 找 candidate region
2. 从 region 里 sample anchor point
3. 把 anchor point 喂给 SAM2 当 prompt
4. SAM2 输出 video-level mask（时间连贯）

这其实是个工程 trick：用 SAM2 的 zero-shot 能力，把 mask 提取从"threshold 调参"变成"prompt engineering"。

**改造 3：物理性 sanity check**
用 acceleration-based detector 抓 camera 突变——如果相邻帧加速度异常，说明 pose 估计错了，直接剔 clip。

参考 MegaSaM: https://megasam-project.github.io/
参考 UniDepth v2: https://github.com/spapathanasiou/UniDepth
参考 SAM2: https://sam2.metademolab.com/

### 第四步：三个 motion 统计量——把 trajectory 数值化

这部分是 paper 一个小亮点。为了让 camera trajectory 可比较、可筛选，定义三个 metric：

**MoveDist**（走了多远）：
$$\text{MoveDist} = \sum_{t=1}^{N-1} \| \mathbf{c}_{t+1} - \mathbf{c}_t \|_2$$

- $\mathbf{c}_t \in \mathbb{R}^3$：第 $t$ 帧相机中心在 world 坐标下的位置
- $N$：总帧数
- 直觉：相机走过的总路径长度，相当于里程表

**RotAngle**（转了多少角度）：
$$\text{RotAngle} = \sum_{t=1}^{N-1} \arccos\left(\frac{\text{tr}(\mathbf{R}_t^\top \mathbf{R}_{t+1}) - 1}{2}\right)$$

- $\mathbf{R}_t \in SO(3)$：第 $t$ 帧的 rotation matrix
- 内层那个公式：旋转矩阵 trace = $1 + 2\cos\theta$，所以反解出 $\theta$（Rodrigues formula）
- 直觉：相邻帧旋转角累加，刻画视角变化幅度

**TrajTurns**（拐了几次弯）：
- 把每帧 orientation 相对 start-end 参考方向算角度
- 数序列里的 local extrema 个数
- 直觉：直线 = 0 turn，S 形 = 多个 turn，衡量 trajectory 复杂度

为什么这三个 metric 重要？因为 Section 4.2 的 Panda-70M 对比里，**Panda-70M 在 RotAngle 和 MoveDist 上 80%+ 都堆在 0 附近**——意味着 Panda-70M 大部分 clip 是死的，相机根本没动。SpatialVID 把这个分布拉宽，HQ subset 进一步平衡。

### 第五步：Motion Instruction——把 trajectory 转成电影术语

这部分 Karpathy 你应该会有共鸣，本质是 **action tokenization**。

流程：
1. 提取相邻帧的 relative translation + rotation
2. temporal smoothing 去 jitter
3. magnitude thresholding——只在变化超过阈值时才生成 instruction，避免噪声
4. map 到 cinematographic vocabulary：
   - `dolly in` = 前推
   - `pan left` = 水平左转
   - `truck right` = 右平移
   - `tilt up` = 仰角
   - `roll` = 翻滚

Fig. 13 用键盘 icon 可视化：
- 左 cluster：translation（W/S=前后, A/D=左右, ↑/↓=上下）
- 右 cluster：rotation（∧/∨=pitch, </>=yaw, ⟲/⟳=roll）

这个 design 的好处是 **可解释 + 可作为 supervision target**。比如 Hunyuan-GameCraft 这种 interactive generation，user 输入"dolly in"，model 直接对应到一个 action token。它就是 video 生成领域的 " Atari joystick interface"。

参考 CameraBench: https://arxiv.org/abs/2504.15376

---

## 最精彩的设计：Structured Caption Pipeline

Karpathy 你之前在 podcast 里说过 VLM 在 spatial reasoning 上很差。这篇 paper 直接 acknowledge 这个问题，并给了一个工程解法。

**问题**：让 Gemini-2.0-Flash 直接看视频描述 camera motion，它经常搞反方向（Fig. 4 例子：实际向左走，VLM 说"right"）。这是因为 VLM 训练数据里 spatial reasoning 信号弱，加上单帧图像无法可靠恢复 3D motion。

**解法**：两阶段 pipeline，**VLM 做 perception，LLM 做 reasoning + correction**。

**Stage 1: Visual Parsing (Gemini-2.0-Flash)**
- 输入：1 fps 采样的 frame sequence
- Prompt 要求输出两段：
  1. Camera Motion Caption（50-100 词）：用 cinematography term 描述
  2. Scene Description（~100 词）：subject + environment + lighting + mood

prompt 的关键设计（Fig. 11）：
- 强调"Do NOT assume the camera starts static"——避免 VLM 偷懒说"static camera"
- 要求"describe motion state transitions, not frame-by-frame repetition"——避免 VLM 把每帧重复说一遍
- 限定字数（50-100 / ~100）——避免 VLM 啰叽歪歪

**Stage 2: Language Refinement (Qwen3-30B-A3B)**
- 输入：Stage 1 的 caption + **calibrated camera pose 数值**
- 输出三个层级 caption：
  1. **OptCamMotion**（50.3 词 avg）：machine-friendly kinematic instruction
  2. **SceneSummary**（28.6 词 avg）：compact context
  3. **ShotImmersion**（89.7 词 avg）：immersive narrative

**核心 insight**：camera pose 作为 LLM 的 conditioning signal。LLM 不直接看 frame，它读 VLM caption + 一串数字（pose），然后纠正方向错误。这是 asymmetric 设计：VLM 擅长 visual parsing（识别物体、描述场景），LLM 擅长 numerical reasoning + long-text coherence，分工明确。

Fig. 4 的 example 很直观：VLM 说"right"，LLM 看到 pose 里 translation 是 negative x，就改成"left"。这是个**用 symbolic signal 修正 neural perception**的经典 pattern。

参考 Qwen3: https://arxiv.org/abs/2505.09388
参考 Gemini 2.0 Flash: https://deepmind.google/technologies/gemini/

---

## 为什么要有 HQ subset？

全集 SpatialVID 270 万 clip，但分布不均衡（Fig. 14 左侧 donut chart：forward motion 占比过大）。HQ subset 37 万 clip 是怎么来的？

- **更紧的 quality threshold**：4 道 filter 阈值再收紧
- **类别 + motion balance sampling**：让 forward / backward / lateral / rotation 各类 motion 占比均衡

Fig. 5 是 paper 的"杀手锏 evidence"——6 张分布图对比 Panda-70M-test / SpatialVID / SpatialVID-HQ：

| 维度 | Panda-70M 长什么样 | SpatialVID 长什么样 | HQ 长什么样 |
|---|---|---|---|
| Aesthetics | 长尾偏左（质量差） | 钟形偏右 | 更窄的钟形偏右 |
| Luminance | 双峰（暗+亮都有） | 单峰钟形 | 更窄单峰 |
| Motion | 大堆 low motion | 偏右分布 | 更偏右 |
| RotAngle | 80%+ 在 0 附近 | 均匀 | 均衡 |
| TrajTurns | 几乎全 0 | 有曲线比例 | 曲线更多 |
| MoveDist | 80%+ 在 0 附近 | 宽分布 | 更宽 |

**直觉**：Panda-70M 对 spatial task 来说大部分 clip 是"死的"。SpatialVID 天然 dynamic（因为 keyword 是 walk/tour/drone），HQ 进一步把 tail 削掉，是训练友好的子集。

这其实是 Karpathy 你常说的 "data quality >> data quantity" 的 verification——全集 270 万很 impressive，但训练只靠 HQ 37 万就够，说明 curator 比 collector 更重要。

---

## 三个 Validation Experiment

数据集 paper 必须证明 dataset 有用。作者选了三个互补的 task：

### Experiment 1: Camera-Controlled Video Generation

**核心**：用 camera pose 作为 control signal，让 video diffusion model 生成对应视角的视频。

**Architecture**（ReCamMaster 风格 + Wan 2.2 backbone）：
- Base：Wan 2.2 DiT-5B（开源 SOTA video diffusion）
- Text encoder：T5
- Camera injection：每帧 3×4 extrinsic matrix（12 参数）通过 learnable linear layer 投到 hidden dim，然后通过 per-block identity-init projector 注入每个 transformer block

数学化（paper 没明写，我推断的）：
$$\mathbf{f}_c^{(t)} = E_c \cdot \text{vec}(\mathbf{P}_t)$$
$$\mathbf{z}^{(l)} \leftarrow \mathbf{z}^{(l)} + W_{\text{proj}}^{(l)} \mathbf{f}_c^{(t)}$$

- $\mathbf{P}_t \in \mathbb{R}^{3\times4}$：第 $t$ 帧 extrinsic（rotation | translation）
- $\text{vec}(\cdot)$：12D flatten
- $E_c \in \mathbb{R}^{12 \times d}$：camera encoder
- $W_{\text{proj}}^{(l)} \in \mathbb{R}^{d \times d}$：第 $l$ 层 projector，init 为 identity 保护 pretrain feature scale

**Training**：
- 382×480 resolution, 81 frames
- 20K steps, batch 32
- AdamW lr $10^{-5}$, cosine decay, 2K warmup
- 32 H20 GPUs × 2 天
- 三个 training data 对比：RE10K / Sekai-Real / SpatialVID-HQ

**Metrics**：
- TransErr / RotErr / CamMC：camera accuracy，用 MegaSaM 反估生成视频的 pose 和 GT 比
- CLIP-T：frame-text similarity
- CLIP-F：inter-frame temporal consistency
- VBench 5 个指标

**结果**（Table 2，9 个 cell）：

| Benchmark | Best Train Data | TransErr↓ | RotErr↓ | CamMC↓ | CLIP-T↑ |
|---|---|---|---|---|---|
| RE10K test | **SpatialVID-HQ** | 7.42 | 0.99 | 7.72 | 30.54 |
| Sekai test | **SpatialVID-HQ** | 6.04 | 1.43 | 6.70 | 35.19 |
| SpatialVID test | **SpatialVID-HQ** | 4.33 | 3.81 | 7.57 | 30.26 |

SpatialVID-HQ 训练的 model 在三个 benchmark 全胜，包括 RE10K 自己的 test set——说明 SpatialVID 的 pose annotation 质量高，model 学到的 camera representation transferable。

**注意**：SpatialVID test 上的 RotErr=3.81 看起来比 RE10K test 的 0.99 大很多。这不是 model 变差，是 benchmark 更难——SpatialVID test 是 real-world dynamic + handheld jitter，RE10K test 是 indoor 静态 + smooth trajectory。绝对数字不能跨 benchmark 比，要看相对排名。

参考 Wan 2.2: https://arxiv.org/abs/2503.20314
参考 ReCamMaster: https://arxiv.org/abs/2503.11647

### Experiment 2: Novel View Synthesis

**核心**：GS-LRM（Large Reconstruction Model + 3D Gaussian Splatting），2 个 reference view 生成 4 个 target view。

**Loss**：
$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{mse}} + \lambda_2 \mathcal{L}_{\text{lpips}} + \lambda_3 \mathcal{L}_{\text{reg}}$$

- $\mathcal{L}_{\text{mse}}$：pixel-wise RGB MSE
- $\mathcal{L}_{\text{lpips}}$：perceptual loss，VGG feature 距离
- $\mathcal{L}_{\text{reg}}$：depth smoothness，惩罚深度突变
- $\lambda_1 = 1.0, \lambda_2 = 0.5, \lambda_3 = 0.25$

两阶段训练：180×320 / 15K steps → 360×640 / 45K steps，total 60K steps，batch 32，AdamW lr $2 \times 10^{-5}$。

**结果**（Table 4）：

| Train Data | DL3DV PSNR↑ | DL3DV LPIPS↓ | SpatialVID PSNR↑ | SpatialVID LPIPS↓ |
|---|---|---|---|---|
| RE10K | 27.01 | 0.132 | 24.13 | 0.222 |
| SpatialVID | **27.80** | **0.116** | **24.97** | **0.203** |

SpatialVID 训练的 model 在 DL3DV（outdoor 为主）上也比 RE10K 训练的好。证明 SpatialVID 的 scene diversity 帮 model 学到了更通用的 geometric prior，不是 overfit 到自己 domain。

参考 GS-LRM: https://katoken-klg.github.io/2024/05/23/gs-lrm.html
参考 DL3DV: https://arxiv.org/abs/2310.13342

### Experiment 3: Geometric Prediction

**核心**：fine-tune 现有 SOTA pose estimator（CUT3R / VGGT），看 SpatialVID 能不能 boost 它们。

**Metrics**：ATE / RPE-trans / RPE-rot，在 Sintel / TUM-dynamics / Dycheck 三个 dynamic benchmark 上测。

**结果**（Table 3）：

CUT3R fine-tune 后：
- TUM-dynamics: ATE 0.049 → 0.040（**19% 提升**）
- TUM-dynamics: RPE-rot 0.449 → 0.395（12% 提升）

VGGT fine-tune 后：
- TUM-dynamics: ATE 0.015 → 0.013
- TUM-dynamics: RPE-rot 0.352 → 0.312

**直觉**：CUT3R / VGGT 主要在 synthetic data 上训练，fine-tune on SpatialVID（real-world dynamic）后，在 real-world dynamic benchmark 上提升最明显。证明 SpatialVID 补上了 real-world dynamic 这个 distribution gap。

参考 CUT3R: https://arxiv.org/abs/2502.14874
参考 VGGT: https://vgg-t.github.io/

---

## 一些更深层的思考

### 1. Data Scaling Law 在 spatial intelligence 上的验证

3D community 一直困在小数据集。这篇 paper 把规模推到百万级，本质上是在说：**spatial intelligence 的下一步 scaling，靠 in-the-wild video 的 distillation pipeline，而不是更精致的 synthetic rendering**。这条路才是 scale 得动的。

### 2. MegaSaM 是 backbone，但不是 ceiling

作者自己也说 ViPE（更新的 video pose estimator）会替代 MegaSaM。这意味着 dataset 的 pose annotation 部分会随着 estimator 进步自动变好——**dataset 本身是 "snapshot of current SOTA"，未来可以 re-annotate**。这跟 model 的 scaling 是平行的两条线。

### 3. Caption 的 VLM+LLM 两阶段是个 template

这种 "VLM 做 perception，LLM 做 reasoning + correction" 的 pattern，在 spatial / temporal reasoning 上几乎是必然方向。特别是引入 camera pose 作为 LLM 的 conditioning signal——**用 symbolic signal 修正 neural perception 的错误**，这是个聪明的工程选择，也是未来 VLM4D 类工作的 template。

参考 VLM4D: https://arxiv.org/abs/2505.04208（推测）

### 4. 与 Sekai 的关系

Sekai（arxiv 2506.15675）是同期工作，也做大规模 web video + spatial annotation。Sekai 在 scale 上更大，SpatialVID 在 dynamic scene + structured caption 上更完整。两个 dataset 互补，预示 2025 下半年 spatial-aware video generation 会有 breakthrough。

参考 Sekai: https://arxiv.org/abs/2506.15675

### 5. World Model 的视角

Karpathy 你说过"world model = simulator"。SpatialVID 直接对应这个 vision：camera pose + depth + structured motion instruction 构成一个"video as state sequence"的训练数据。下游 model 可以学 "given state + action (motion) → next state (next frame)"。这其实就是 Cosmos / Genie3 的训练 paradigm。

参考 Cosmos: https://arxiv.org/abs/2501.03575

### 6. 可能的 next step（paper 没做）

- **Dynamic object 6-DoF tracking**：现在只有 dynamic mask，没 track individual object
- **Long video > 15s**：clip 切成 3-15s，长视频连贯性没用
- **4D reconstruction**：time-varying geometry 还没有
- **Close the loop**：用 SpatialVID 训练的 model 反过来 generate synthetic data augment

### 7. 与 MineDojo 的类比

在 embodied agent 领域，data-centric breakthrough 都是从 web mining 开始。**SpatialVID 之于 spatial intelligence，就像 MineDojo 之于 Minecraft agent**——都是从互联网"distill"训练数据，让 scale 起来。

参考 MineDojo: https://minedojo.github.io/

---

## Useful Links 一览

| 类别 | 链接 |
|---|---|
| MegaSaM | https://megasam-project.github.io/ |
| UniDepth v2 | https://github.com/spapathanasiou/UniDepth |
| Depth Anything v2 | https://depth-anything-v2.github.io/ |
| SAM2 | https://sam2.metademolab.com/ |
| CameraBench | https://arxiv.org/abs/2504.15376 |
| Sekai | https://arxiv.org/abs/2506.15675 |
| Wan 2.2 | https://arxiv.org/abs/2503.20314 |
| ReCamMaster | https://arxiv.org/abs/2503.11647 |
| CameraCtrl | https://arxiv.org/abs/2404.02101 |
| GS-LRM | https://katoken-klg.github.io/2024/05/23/gs-lrm.html |
| CUT3R | https://arxiv.org/abs/2502.14874 |
| VGGT | https://vgg-t.github.io/ |
| Panda-70M | https://arxiv.org/abs/2402.19479 |
| RealEstate10K | https://arxiv.org/abs/1805.09817 |
| DL3DV | https://arxiv.org/abs/2310.13342 |
| Cosmos | https://arxiv.org/abs/2501.03575 |
| Qwen3 | https://arxiv.org/abs/2505.09388 |
| VBench | https://vchitect.github.io/VBench-project/ |
| Stereo4d | https://arxiv.org/abs/2412.09621 |
| CO3Dv2 | https://commonobjectsin3d.github.io/ |
| MineDojo | https://minedojo.github.io/ |

---

**最后一句话 build intuition**：SpatialVID 干的事，相当于把 YouTube 上随手可得的"raw 2D video" 通过一个 MegaSaM + SAM2 + VLM + LLM 的工厂流水线，蒸馏成"带 3D 标注 + 精致 caption 的训练 fuel"，让 spatial intelligence / world simulator 这个方向的 model 终于有足够大、足够 diverse、足够 accurate 的数据来 scale。它本身没提出新 model，但给了 community 一个能撑起下一代 world simulator 的 data substrate。在 data-centric AI 这条路上，这是一篇扎实的 infrastructure 工作。

---

# SpatialVID: 从 Web Video 到 Spatial Intelligence 的 Data Engine

Andrej, 这篇 paper 读完我的第一直觉是：**这是一个 data-centric 的工作，而不是 model-centric 的工作**。核心 contribution 是把互联网上随手可得的 video "蒸馏" 成带有 dense 3D geometric annotation 的训练数据，填补了 video dataset（rich semantics, no geometry）和 spatial dataset（precise geometry, no scale）之间的鸿沟。下面我从 motivation、pipeline、关键设计、实验验证四个层次拆解。

---

## 1. Motivation: 为什么现在的 Dataset 都不够用？

Karpathy 你自己常说 "the bitter lesson"，这里其实是一个 data bottleneck 的典型例子。Paper 在 Section 1 精准指出了当前 data 的两难：

- **Large-scale video dataset**（Panda-70M, MiraData, OpenVid-1M）有 rich semantics 但是 **zero geometry**——模型必须 implicit 从 pixel 推 spatial relations，所以 Sora 类 world simulator 在物理一致性上一直是软肋。
- **Spatial dataset**（CO3Dv2, RealEstate10K, TartanAir, ScanNet）有 accurate camera pose 和 depth，但是 **scale 小、scene 静态、或 synthetic**——object-centric、indoor-only、或者 rendering pipeline 工程开销巨大。

Table 1 是关键 evidence：之前最大的 real-world dynamic dataset 是 Stereo4d（110K clips, 10M frames, fisheye domain），Dynpose100k 只有 camera pose 没有 depth。**SpatialVID 直接把规模推到 2.71M clips / 127.6M frames，并且同时有 C. + D. + Structured Caption**，比 Stereo4d 大 ~25×。

这里有个非常 Karpathy-style 的 insight：作者先把 Panda-70M 的 validation split 喂进自己的 pipeline，**只保留 10% 的 clip 通过 quality 标准**——这其实是在说，现成的大 video dataset 大部分都是 "static viewpoint、flickering、没 motion description"，对 spatial 任务几乎不可用。这就是为什么他们必须从 YouTube 自己 re-collect 21,789 小时的 raw video。

参考：
- Panda-70M: https://arxiv.org/abs/2402.19479
- RealEstate10K: https://arxiv.org/abs/1805.09817
- Stereo4d: https://arxiv.org/abs/2412.09621
- CO3Dv2: https://commonobjectsin3d.github.io/

---

## 2. Curation Pipeline: 三段式 Data Engine

整个 pipeline（Fig. 2）的设计哲学是 **hierarchical filtering + dual annotation**，可以理解为一个 funnel：

```
33,443 YouTube videos (21,789 hours)
    → preprocessing (PySceneDetect + H.265/720P) → 7M+ raw clips
    → 4-axis quality filter → 2.71M clips (SpatialVID, 7,089 hrs)
    → motion/category balancing → SpatialVID-HQ (0.37M clips, 1,111 hrs)
```

### 2.1 Preprocessing 细节

PySceneDetect 被改造过：作者调整 sensitivity threshold + 用 interval-based multi-frame comparison 来处理 fade transition。这其实是个工程 trick——传统的 cut detector 对 hard cut 很准，但对 slow fade（aesthetic transition）容易漏检或者误判。最终所有 clip 标准化到 H.265 MP4 / 1280×720。

### 2.2 四轴 Filtering（Appendix A.1）

每个 clip 用 first/middle/last 三帧采样评估（节省 compute）：

| Filter | 工具 | 阈值 | 直觉 |
|---|---|---|---|
| Aesthetics | CLIP+MLP predictor (LAION-5B 训练) | avg score < 4.0 丢弃 | 过滤 low-quality 内容 |
| Luminance | 公式 $L = 0.2126 R + 0.7152 G + 0.0722 B$ | 落在 [20, 140] 之外丢弃 | Rec. 709 luma 系数，避免 over/under-exposed |
| OCR | PaddleOCR 3.0 | text area > 30% 丢弃 | 去掉新闻、documentary 之类 watermark-heavy clip |
| Motion | VMAF (FFmpeg 集成) | score 2.0–14.0 | VMAF 是 Netflix 的 perceptual quality metric，作者挪用为 motion 强度 proxy |

公式讲解：
$$L = 0.2126 R + 0.7152 G + 0.0722 B$$

- $L$：该像素的 luma（亮度）
- $R, G, B$：分别是 sRGB 空间下 red / green / blue channel 的归一化值（0–1 或 0–255 都可以，关键是用 Rec. 709 加权）
- 系数来自 ITU-R BT.709 standard：人眼对 green 最敏感，所以 G 权重最大；red 次之；blue 几乎可忽略

这个阈值的 [20, 140] 范围对应 8-bit 强度空间，非常保守——大约只保留中间 47% 的亮度区间。这种 aggressive filtering 说明作者对训练数据 quality 极其挑剔，宁可漏掉一些 borderline clip 也不要 noise。

### 2.3 Geometry Annotation（Section 3.3 + Appendix A.2）

这是整个 paper 技术上最 interesting 的部分。Base engine 是 **MegaSaM**（Google 提出的 casual dynamic video SfM），但作者做了三处关键改造：

1. **Depth module 替换**：原版 MegaSaM 依赖 external monocular depth，作者换上 **UniDepth v2 + Depth Anything v2**。这两个 model 一个做 metric depth（UniDepth v2），一个做 relative dense depth（Depth Anything v2），互补——paper 提到在 dominant moving object、collinear motion、varying focal length 这些 corner case 上 robustness 显著提升。
2. **Dynamic mask**：先用 adaptive thresholding + contour detection 得到 candidate region，sample anchor point，再用 **SAM2** prompt 提取 mask。然后计算 dynamic ratio（每帧动态区域占比）。
3. **物理性 sanity check**：用 acceleration-based detector 抓 abrupt、non-physical motion fluctuation，把这些 clip 直接剔除。

参考：
- MegaSaM: https://arxiv.org/abs/2411.16606（实际 paper 在 arxiv 上）
- UniDepth v2: https://github.com/spapathanasiou/UniDepth
- Depth Anything v2: https://arxiv.org/abs/2406.09414
- SAM2: https://arxiv.org/abs/2408.00714

#### 三个 Camera Motion Statistic（这是 paper 的一个小亮点）

为了把 camera trajectory 数值化、可比较，作者引入三个 metric：

**MoveDist**（位移总长）：
$$\text{MoveDist} = \sum_{t=1}^{N-1} \| \mathbf{c}_{t+1} - \mathbf{c}_t \|_2$$

- $\mathbf{c}_t \in \mathbb{R}^3$：第 $t$ 帧的 camera center 在 world coordinate 下的位置
- $N$：clip 总帧数
- 直觉：camera 走过的 total path length，类似 odometry 的累计里程

**RotAngle**（累计旋转角）：
$$\text{RotAngle} = \sum_{t=1}^{N-1} \arccos\left(\frac{\text{tr}(\mathbf{R}_{t}^\top \mathbf{R}_{t+1}) - 1}{2}\right)$$

- $\mathbf{R}_t \in SO(3)$：第 $t$ 帧的 camera rotation matrix
- 中间那项是 Rodrigues 公式 / 旋转矩阵的 trace-based angle formula（$1 + 2\cos\theta$ → $\theta$）
- 直觉：所有相邻帧之间旋转角的总和，刻画 viewpoint 变化幅度

**TrajTurns**（轨迹转弯数）：
- 定义为：相对于 start-end reference direction 的 orientation angle序列中 local extrema 的个数
- 直觉：camera 走直线 = 0 turns；走 S 形 = 多个 turns。这衡量 trajectory 复杂度

这三个 metric 在 Section 4.2 的 Panda-70M 对比中起到决定性作用：**Panda-70M 的 TrajTurns 直方图 80%+ 集中在 0**——意味着大多数 clip 是 static 或者几乎不动，根本没法做 reconstruction。SpatialVID-HQ 故意提升了 curved/turning trajectory 的比例，这是有意识的 data curation。

### 2.4 Motion Instruction Decomposition（Section 3.4）

这部分 Karpathy 你应该会很感兴趣——它本质上是把 camera trajectory **tokenize** 成 cinematographic vocabulary。

流程：
1. 从 estimated pose sequence 提取相邻帧之间的 relative translation + relative rotation
2. temporal smoothing filter（细节没说，猜测是 moving average 或 Savitzky-Golay）去 jitter
3. **magnitude-based thresholding**：只在 pose variation 超过 threshold 时才生成 instruction，避免 trivial movement 噪声
4. 把 motion signal map 到 **CameraBench 风格的 controlled vocabulary**：`dolly in`（前推）、`pan left`（水平左转）、`truck right`（右平移）等等

Fig. 13 给出了 icon system：
- 左 cluster：translation（W/S=前后, A/D=左右, ↑/↓=垂直）
- 右 cluster：rotation（∧/∨=pitch, </>=yaw, ⟲/⟳=roll）

这种设计的好处是 **可解释 + 可作为 supervision target**——你可以直接 fine-tune 一个 VLM 让它输出 motion token，或者作为 Hunyuan-GameCraft 类 interactive generation 的 action interface。

参考 CameraBench: https://arxiv.org/abs/2504.15376

### 2.5 Structured Caption Pipeline（Section 3.5 + Fig. 3）—— 这是 Karpathy 视角最精彩的设计

这里作者很诚实地指出一个 limitation：**VLM 在 spatial reasoning 上很差**。Gemini 2.0 Flash 这种级别的 VLM 也经常搞反 motion direction（Fig. 4 给的例子：VLM 说 "right"，实际是 "left"）。

他们的解法是两阶段：

**Stage 1: Visual Parsing**（Gemini-2.0-Flash）
- 输入：1 fps 采样的 frame sequence
- Prompt（Fig. 11）要求输出两段：
  1. **Camera Motion Caption**（50–100 词）：用 cinematography term 描述 motion trajectory
  2. **Scene Description**（~100 词）：主客体 + 环境 + 光照 + 氛围

**Stage 2: Language Refinement**（Qwen3-30B-A3B）
- 输入：Stage 1 输出 + **calibrated camera poses**
- 输出三个层级 caption：
  1. **OptCamMotion**（avg 50.3 词）：machine-friendly kinematic instruction
  2. **SceneSummary**（avg 28.6 词）：compact context
  3. **ShotImmersion**（avg 89.7 词）：immersive narrative

关键 insight：**camera pose 作为 LLM refinement 的 conditioning signal**。LLM 不直接看 frame，它读 VLM 的 caption + camera pose 数值，然后纠正方向错误。这是一个 asymmetric 设计：VLM 擅长 visual parsing，LLM 擅长长文本 coherence + numerical reasoning，分工明确。这种 "geometry-grounded caption" 是 paper 的方法论亮点。

参考：
- Gemini 2.0 Flash: https://deepmind.google/technologies/gemini/
- Qwen3: https://arxiv.org/abs/2505.09388

---

## 3. Dataset Statistics: 为什么 SpatialVID-HQ 是必要的

SpatialVID 全集 2.71M clip 看起来 impressive 但是分布不均衡（Fig. 14 左侧 donut chart）。HQ subset 是用更紧的阈值 + 类别平衡 sampling 出来的 0.37M clip。

从 Fig. 5 的 6 张 distribution 图可以读出几个关键 takeaway：

| Metric | Panda-70M-test | SpatialVID | SpatialVID-HQ |
|---|---|---|---|
| Aesthetics | 长尾偏左（低质量多） | 紧凑偏右 | 更紧凑偏右 |
| Luminance | 双峰（暗 + 亮都有） | 钟形单峰 | 钟形单峰更窄 |
| Motion | 大量 low motion | 偏右 | 更偏右 |
| RotAngle | 80%+ 在 0 附近 | 均匀分布 | 平衡分布 |
| TrajTurns | 几乎全 0 | 有曲线比例 | 曲线比例更高 |
| MoveDist | 80%+ 在 0 附近 | 宽分布 | 更宽 |

**直觉解读**：Panda-70M 是个 generic video dataset，对 spatial task 来说大部分 clip 是 "死" 的（static camera）。SpatialVID 是从 raw video 起就 manual screen + keyword query（walk/tour/drone），所以 motion 分布天然偏 dynamic；HQ 进一步把 tail 削掉，更训练友好。

---

## 4. Validation Experiments: 三个 Task，三种证据

Paper 选了三个互补的 downstream task 来 validate dataset quality。这其实是数据集 paper 的标准做法，但作者选得很巧：

### 4.1 Camera-Controlled Video Generation（Section 5.1）

**Baseline**：ReCamMaster 的 camera injection 机制 + Wan 2.2 DiT backbone + T5 text encoder。核心机制（Section C.1）：

每帧的 3×4 camera extrinsic matrix（12 个参数）通过一个 learnable linear layer $E_c \in \mathbb{R}^{12 \times d}$ 投影到 video token 维度 $d$。然后通过 per-block projector（初始化为 identity）和 visual token 融合，注入到每个 transformer block。

公式化（我的理解，paper 没明写）：
$$\mathbf{f}_c^{(t)} = E_c \cdot \text{vec}(\mathbf{P}_t) \in \mathbb{R}^d$$
$$\mathbf{z}_t^{(l)} \leftarrow \mathbf{z}_t^{(l)} + W_{\text{proj}}^{(l)} \cdot \mathbf{f}_c^{(t)}$$

- $\mathbf{P}_t \in \mathbb{R}^{3 \times 4}$：第 $t$ 帧 extrinsic matrix（R | t，rotation + translation）
- $\text{vec}(\cdot)$：matrix → 12D vector
- $E_c$：把 12D 投到 hidden dim
- $W_{\text{proj}}^{(l)}$：第 $l$ 层 transformer block 的 projector，初始化为 identity 以保留 pretrain feature scale

**Training setting**（Section C.1）：
- Resolution: 382 × 480
- Sequence length: 81 frames
- 20K steps, global batch 32
- AdamW, lr $10^{-5}$, cosine decay + 2K warmup
- 32 H20 GPUs × 2 天

**Metrics**：
- **TransErr** / **RotErr** / **CamMC**：camera accuracy，用 MegaSaM 估计生成 video 的 pose，和 GT 比较
- **CLIP-T**：frame-text similarity
- **CLIP-F**：inter-frame temporal consistency
- VBench metrics（subject consistency, background consistency, motion smoothness, aesthetic, imaging quality）

**Table 2 结果分析**（核心 takeaway）：

三个 training data（RE10K / Sekai-Real / SpatialVID-HQ）× 三个 benchmark（RE10K / Sekai / SpatialVID）= 9 个 cell。SpatialVID-HQ 在所有 3 个 benchmark 上的 Camera Accuracy 都是最好：

| Benchmark | Best Training Data | TransErr↓ | RotErr↓ | CamMC↓ |
|---|---|---|---|---|
| RE10K | SpatialVID-HQ | 7.42 | 0.99 | 7.72 |
| Sekai | SpatialVID-HQ | 6.04 | 1.43 | 6.70 |
| SpatialVID | SpatialVID-HQ | 4.33 | 3.81 | 7.57 |

注意 SpatialVID-HQ 在自己的 benchmark 上 RotErr=3.81 看似比 RE10K 上的 0.99 差很多——这是因为 SpatialVID benchmark 包含更复杂的 dynamic + handheld jitter trajectory，绝对数值高不代表 model 差，反而证明 benchmark 更 challenging。Sekai 和 SpatialVID 都是 dynamic real-world，RE10K 是静态 indoor 为主的合成轨迹测试集，所以 RE10K 的数字看起来漂亮但实际 generalization 价值低。

**Karpathy 视角的 takeaway**：这个实验直接验证了 dataset 的 camera pose annotation 质量。如果 pose 估计错，model 学到的 control signal 就是 garbage，TransErr 会爆炸。SpatialVID-HQ 在三个 benchmark 都赢，说明 pose 估计不仅在自己 domain 准，还学到了 transferable camera representation。

参考：
- Wan 2.2: https://arxiv.org/abs/2503.20314
- ReCamMaster: https://arxiv.org/abs/2503.11647
- CameraCtrl: https://arxiv.org/abs/2404.02101
- VBench: https://arxiv.org/abs/2311.16122

### 4.2 Novel View Synthesis（Section 5.2）

**Baseline**：GS-LRM（Large Reconstruction Model + 3D Gaussian Splatting）。两阶段训练（Section C.1）：
- Stage 1: 180×320, 15K steps
- Stage 2: 360×640, 45K steps
- 总 60K steps, batch 32, AdamW, lr $2 \times 10^{-5}$, cosine + 2K warmup

**Loss function**:
$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{mse}} + \lambda_2 \mathcal{L}_{\text{lpips}} + \lambda_3 \mathcal{L}_{\text{reg}}$$

- $\mathcal{L}_{\text{mse}}$：pixel-wise reconstruction loss（per-pixel RGB MSE）
- $\mathcal{L}_{\text{lpips}}$：perceptual loss（LPIPS，VGG features 之间距离）
- $\mathcal{L}_{\text{reg}}$：depth smoothness regularization（penalize abrupt depth discontinuity）
- $\lambda_1 = 1.0$, $\lambda_2 = 0.5$, $\lambda_3 = 0.25$

实验设置：2 reference view + 4 target view supervision，60K clip 训练（SpatialVID 和 RE10K 等量），在 DL3DV（500 seq）和 SpatialVID（500 seq）测试。

**Table 4 结果**：

| Training Data | DL3DV PSNR↑ | DL3DV SSIM↑ | DL3DV LPIPS↓ | SpatialVID PSNR↑ | SpatialVID SSIM↑ | SpatialVID LPIPS↓ |
|---|---|---|---|---|---|---|
| RE10K | 27.01 | 0.889 | 0.132 | 24.13 | 0.774 | 0.222 |
| SpatialVID | 27.80 | 0.892 | 0.116 | 24.97 | 0.790 | 0.203 |

SpatialVID 训练的 model 在 DL3DV（outdoor 为主）上也比 RE10K 训练的好——证明 SpatialVID 的 scene diversity 帮 model 学到了更 robust 的几何 prior，不只是 overfit 到自己的 domain。

参考：
- GS-LRM: https://katoken-klg.github.io/2024/05/23/gs-lrm.html
- DL3DV: https://arxiv.org/abs/2310.13342

### 4.3 Geometric Prediction（Section 5.3）

这是验证 dataset 对已有 SOTA 几何估计 model 的 fine-tuning 增益。

**Baselines**: CUT3R（continuous 3D perception with persistent state）+ VGGT（Visual Geometry Grounded Transformer）

**Metrics**: ATE（Absolute Trajectory Error）/ RPE-trans / RPE-rot，在 Sintel / TUM-dynamics / Dycheck 三个 dynamic benchmark 上测。

**Table 3 结果**：

CUT3R fine-tune 后：
- Sintel: ATE 0.210→0.210, RPE-trans 0.070→0.069, RPE-rot 0.637→0.619（小幅提升）
- TUM-dynamics: ATE 0.049→0.040（**19% 提升**）, RPE-trans 0.015→0.013, RPE-rot 0.449→0.395
- Dycheck: ATE 0.020→0.019, RPE-rot 1.275→1.184

VGGT fine-tune 后：
- Sintel: ATE 0.134→0.148（轻微回退，可能 noise）, RPE-rot 0.501→0.462
- TUM-dynamics: ATE 0.015→0.013, RPE-rot 0.352→0.312
- Dycheck: 几乎饱和（VGGT 已经太强了）

**直觉解读**：CUT3R / VGGT 主要在 synthetic data 上训练，fine-tune on SpatialVID（real-world dynamic）后，在 TUM-dynamics 这种 real-world dynamic benchmark 上有最显著的提升。这证明 SpatialVID 补上了 real-world dynamic 这个 distribution gap。

参考：
- CUT3R: https://arxiv.org/abs/2502.14874
- VGGT: https://vgg-t.github.io/

---

## 5. Limitation & 我的思考

Paper Section 6 自己列了 limitation：
- 继承 MegaSaM 的 failure mode（dominant moving object、varying focal length、radial distortion）
- camera pose 在某些场景下 non-metric（即尺度不确定）
- mask 来自 motion probability，复杂场景 suboptimal
- 未来用 ViPE（更新的 video pose estimator）能改善

**Karpathy 视角的几点 deeper insight**：

1. **Data Scaling Law 在 spatial intelligence 上的验证**：之前 3D community 一直困在小数据集，这篇文章把规模推到百万级 clip，本质上是在说 "spatial intelligence 的下一步 scaling，靠的是 in-the-wild video 的 distillation pipeline，而不是更精致的 synthetic rendering"。

2. **MegaSaM 是 backbone 但不是 ceiling**：作者自己也说 ViPE 会替代。这意味着这个 dataset 在 pose annotation 部分会随着 estimator 进步而变好——dataset 本身是 "snapshot of current SOTA"，未来可以 re-annotate。

3. **Caption 的 two-stage VLM+LLM 设计是个 template**：这种 "VLM 做 perception，LLM 做 reasoning + correction" 的 pattern 在 spatial / temporal reasoning 上几乎是必然方向。特别是引入 camera pose 作为 LLM 的 conditioning signal，是一个聪明的工程选择。

4. **HQ subset 的 importance**：全集 2.71M 看着 impressive，但训练用 HQ 0.37M 就够了——这其实是 Karpathy 你自己常说的 "data quality >> data quantity" 的一个 verification。Fig. 5 的 distribution 对比是 paper 的"杀手锏"。

5. **下一步的开放问题**：
   - Dynamic object 的 6-DoF tracking 还没做（只有 dynamic mask）
   - Long video（>15s）的连贯性没用上
   - 4D reconstruction（time-varying geometry）还没有
   - 是否能 close the loop：用 SpatialVID 训练的 model 反过来 generate synthetic data 再 augment？

6. **与 concurrent work 的关系**：Sekai（arxiv 2506.15675）几乎是同期工作，也做大规模 web video + spatial annotation。SpatialVID 在 dynamic scene + structured caption 上更完整，Sekai 在 scale 上更大。这两个 dataset 互相补足，预示 2025 下半年 spatial-aware video generation 会有 breakthrough。

---

## 6. 一些可能的相关联想（Hallucination 区，但应该有用）

- **Karpathy 你提过的 "world model = simulator"**：SpatialVID 直接对应这个 vision——camera pose + depth + structured motion instruction 构成了一个 "video as a state sequence" 的训练数据，下游 model 可以学习 "given state + action (motion) → next state (next frame)"。这其实就是 Cosmos / Genie3 的训练 paradigm。
- **与 ELLIE / MineDojo 类比**：在 embodied agent 领域，data-centric breakthrough 都是从 web mining 开始。SpatialVID 之于 spatial intelligence，就像 MineDojo 之于 Minecraft agent。
- **可能的 negative result 没报告**：fine-tune 在 SpatialVID 上是否会让 model 在静态 scene（如 RE10K 测试）上变差？Table 2 的 RE10K benchmark 上 SpatialVID-HQ 训练还是最好，说明没有 negative transfer，这点不容易。
- **关于 VLM4D / 3D-LLM-Mem 的联系**：SpatialVID 的 structured caption 可以直接作为这些 spatial-aware VLM 的训练数据，让 VLM 真正具备 3D grounding。

---

## 7. Useful Links 汇总

| 类别 | 链接 |
|---|---|
| Paper | （SpatialVID arxiv，未直接给出，推测在 arxiv 2025 末）|
| MegaSaM | https://megasam-project.github.io/ |
| UniDepth v2 | https://github.com/spapathanasiou/UniDepth |
| Depth Anything v2 | https://depth-anything-v2.github.io/ |
| SAM2 | https://sam2.metademolab.com/ |
| CameraBench | https://arxiv.org/abs/2504.15376 |
| Sekai (concurrent) | https://arxiv.org/abs/2506.15675 |
| Wan 2.2 | https://arxiv.org/abs/2503.20314 |
| ReCamMaster | https://arxiv.org/abs/2503.11647 |
| GS-LRM | https://katoken-klg.github.io/2024/05/23/gs-lrm.html |
| CUT3R | https://arxiv.org/abs/2502.14874 |
| VGGT | https://vgg-t.github.io/ |
| Panda-70M | https://arxiv.org/abs/2402.19479 |
| DL3DV | https://arxiv.org/abs/2310.13342 |
| Cosmos World Foundation | https://arxiv.org/abs/2501.03575 |
| Genie 3 | （Google DeepMind, 2025） |
| Qwen3 | https://arxiv.org/abs/2505.09388 |
| VBench | https://vchitect.github.io/VBench-project/ |

---

**总结一句 build intuition 的话**：SpatialVID 的核心贡献是把 video data 的 "几何盲区" 用一个 MegaSaM-based + VLM+LLM 的蒸馏 pipeline 一次性补上，把百万级 real-world dynamic video 变成 spatial intelligence model 的可训练燃料。它不是新 model，是给 community 提供了一个能撑起下一代 world simulator 的 data substrate——而这恰恰是 Karpathy 你一直在强调的 "scale + quality" 的 data 路线。
