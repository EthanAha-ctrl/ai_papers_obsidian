---
source_pdf: ViSE A Systematic Approach to Vision-Only Street-View Extrapolation.pdf
paper_sha256: 0289c08ee824e3a5864a7c0ed9311367efdcd10594a0669142379f8be88216df
processed_at: '2026-08-13T01:22:00-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 ViSE

好，我把这篇 paper 用最朴素的方式重讲一遍。想象咱俩在白板前聊。

## 一句话概括

**自动驾驶要在仿真里做闭环测试，但真实开过的路只能 replay，没法换视角。ViSE 拿了个第一名，思路是：先把路面用"高度场 + 坡度"这种简化模型钉死，剩下不规则的物体用生成模型补，最后再把"这个时间点特有的光影水渍"洗掉。**

---

## 问题到底难在哪

你开着车沿一条 lane 走，相机一路拍。现在 simulator 想问："如果这车变到隔壁 lane，或者抬高 1 米、转个 30 度，会看到啥？"

这个问题听起来无害，实际上对 NeRF / 3DGS 这类方法是灾难。原因就一个：**训练数据里那个视角根本没出现过**。

- 沿原轨迹前后移动：interpolation，ray 在 3D 空间里交叉约束同一个 voxel，优化 landscape 健康。
- 横向挪到隔壁 lane：那片 voxel 从来没有 training ray 穿过，Gaussian 想怎么漂就怎么漂，render 出来就是一堆 floaters 和扭曲的纹理。

这就像你只见过一个人的正脸，让我画他侧脸——我可能画得像，但耳朵位置全靠猜。

RealADSim 这个 benchmark 把这件事做得很绝：训练给你 lane A 上午晴天的数据，测试用 lane B 下午的数据。你不仅要外推**空间**，还要外推**时间**。

Links:
- 比赛主页: https://huggingface.co/spaces/XDimLab/ICCV2025-RealADSim-NVS

---

## ViSE 的四步流水线

### 第一步：没 LiDAR 怎么初始化 3DGS

3DGS 一定要个初始 point cloud。标准做法是 COLMAP（SfM）。问题在 driving 场景下 COLMAP 不好使：

- 沥青路面没纹理，feature matching 找不到点
- 车是 forward motion，epipolar geometry 约束弱
- 动态车辆污染匹配

COLMAP 输出的稀疏点云又少又错，直接拿去 init 3DGS 会掉进 local minimum——训练 view 上 photometric loss 满足了，但 geometry 根本是错的，一换视角就崩。

ViSE 的解法：**用 VGGT（Visual Geometry Grounded Transformer，CVPR 2025）替代 COLMAP**。VGGT 是个 feed-forward transformer，吞 multi-view images，吐 camera poses + per-pixel depth + point map。它见过大量 driving data，对路面/车辆这种常见结构有 learned prior，比 COLMAP 鲁棒得多。

但 VGGT 输出的 depth 是 relative scale，单位不明。怎么办？**用 GT pose 对齐恢复 scale**：

设 GT pose 的 translation 是 $t_i$，VGGT 预测的 translation 是 $\hat{t}_i$（up to scale $s$）。求：

$$s^* = \arg\min_s \sum_i \|s \cdot \hat{t}_i - t_i\|^2$$

这是个简单 least-squares，闭式解 $s^* = \frac{\sum_i \hat{t}_i \cdot t_i}{\sum_i \hat{t}_i \cdot \hat{t}_i}$。

然后所有 predicted depth $\hat{D}$ 乘上 $s^*$，用 GT intrinsics + GT poses unproject 成 unified point cloud。

这步的产物虽然 noisy，但比 COLMAP 强一个数量级，关键给了 3DGS 一个"大致正确的几何骨架"，防止训练时偷懒走捷径。

Links:
- VGGT: https://arxiv.org/abs/2503.00547
- COLMAP: https://colmap.github.io/

---

### 第二步：2D-SDF 钉死路面 — 全 paper 最有意思的部分

#### 直觉

路面这个东西，本质上是个**带坡度的平面**。它有起伏、有 banked turn，但局部看就是平的。这种结构强 prior，用通用 3D SDF（一个 MLP $f: \mathbb{R}^3 \to \mathbb{R}$）去表达完全是浪费 capacity。

ViSE 的洞察：**把 3D SDF 降维到 2D**。路面是个 2-manifold embedded in 3D，只需要两个 2D 函数就能描述：
- 高度场 $\mathcal{H}(p_x, p_y)$：在水平坐标 $(p_x, p_y)$ 处路面的高度
- 坡度 $|\cos\theta|(p_x, p_y)$：这里路面法线和竖直方向夹角的 cosine

#### 公式

3D 点 $\mathbf{p} = (p_x, p_y, p_z)$ 到路面的 signed distance：

$$d(\mathbf{p}) = |\cos\theta| \cdot \left(p_z - \mathcal{H}(p_x, p_y)\right)$$

变量解释：
- $p_x, p_y, p_z$：3D 点 $\mathbf{p}$ 的三个坐标，$p_z$ 是竖直方向
- $\mathcal{H}(p_x, p_y)$：在水平位置 $(p_x, p_y)$ 处路面的高度，是个 2D 函数
- $\theta$：路面法线方向和竖直轴(z 轴)的夹角
- $|\cos\theta|$：法线的竖直分量，取绝对值

直觉拆解：
- 路面完全水平时 $\theta = 0$，$|\cos\theta| = 1$，公式退化成 $d = p_z - \mathcal{H}$，纯高度差
- 路面有坡度时 $|\cos\theta| < 1$，竖直距离 $p_z - \mathcal{H}$ 高估了真实到 surface 的距离（因为 surface 倾斜），乘上 $|\cos\theta|$ 修正

两个函数都用小 MLP 实现，输入 $(p_x, p_y)$ 加上 positional encoding：

$$\begin{cases}
d(\mathbf{p}) = \text{MLP}_{\text{slope}}(\mathbf{p}) \cdot [p_z - \text{MLP}_{\text{elevation}}(\mathbf{p})] \\
\mathbf{c}(\mathbf{p}, \mathbf{v}) = \text{MLP}_{\text{color}}(\mathbf{p}, \mathcal{F}(\mathbf{v}))
\end{cases}$$

- $\text{MLP}_{\text{slope}}$ 输出 $|\cos\theta| \in (0, 1]$，用 sigmoid 保证非负，加 smoothness regularizer 保证局部光滑
- $\text{MLP}_{\text{elevation}}$ 输出路面高度，同样加 smoothness regularizer
- $\mathcal{F}(\mathbf{v})$ 是 view direction encoding，处理 view-dependent color（沥青在不同光照角度下颜色不同）

#### 为什么这样省

普通 3D SDF 一个 MLP $f: \mathbb{R}^3 \to \mathbb{R}$，采样要 $O(n^3)$ 个点 cover volume。这里压成两个 $f: \mathbb{R}^2 \to \mathbb{R}$，采样降到 $O(n^2)$，加上 prior 信息强（路面局部 planar），几乎不损失表达力。Paper 说 **15 分钟 / scene 收敛**。

#### 用 NeuS 的 volume rendering 端到端可微

为了能用 photometric loss 反向传播，SDF 必须转成可微的 volume rendering。ViSE 直接用 NeuS 的公式：

$$\alpha_i = \max\left(\frac{\Phi_s(d(\mathbf{p}(t_i))) - \Phi_s(d(\mathbf{p}(t_{i+1})))}{\Phi_s(d(\mathbf{p}(t_i)))}, 0\right)$$

变量：
- $i$：ray 上第 $i$ 个采样点
- $t_i$：ray 上累积深度值
- $\mathbf{p}(t_i) = \mathbf{o} + t_i \mathbf{d}$：ray $\mathbf{r} = (\mathbf{o}, \mathbf{d})$ 上 3D 点（$\mathbf{o}$ 是相机原点，$\mathbf{d}$ 是 ray 方向）
- $d(\cdot)$：上面定义的 2D-SDF
- $\Phi_s(x) = 1/(1 + e^{-sx})$：sigmoid CDF，$s$ 是 learnable sharpness parameter，控制 surface 边界软硬

直觉：$\Phi_s$ 把 SDF 值转成"占用累积概率"，相邻 sample 差分就是这段 ray 的"密度增量"。SDF 接近 0 时（快到 surface），$\Phi_s$ 变化最陡，$\alpha_i$ 最大。分母除以 $\Phi_s(d(\mathbf{p}(t_i)))$ 是 NeuS 相比 VolSDF 的关键修正，防止远处 ray 因还没人 absorb 而错估密度。

颜色合成：

$$\mathbf{C}(\mathbf{r}) = \sum_{i=1}^N T_i \alpha_i \mathbf{c}_i, \quad T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$$

- $T_i$：transmittance，ray 到第 $i$ 个点之前"没被吸收"的概率
- $\mathbf{c}_i$：第 $i$ 个采样点的 view-dependent color
- 标准 front-to-back volume rendering

#### 分层合成

Scene 分三层：

1. **Above-ground objects**: 3D Gaussians（车、建筑、树、电线杆）
2. **Road surface**: 2D-SDF（沥青、lane marking）
3. **Sky**: environment map（球面纹理，可优化）

合成公式：

$$I = O_{gs} I_{gs} + (1 - O_{gs}) O_{road} I_{road} + (1 - O_{gs})(1 - O_{road}) I_{sky}$$

- $I_{gs}, I_{road}, I_{sky}$：三层各自 render 的 RGB
- $O_{gs}, O_{road}$：累积 opacity
- $(1 - O_{gs})$：3DGS 没挡住的"剩余 visibility"

直觉：front-to-back layering。Gaussian 挡在前，看不到的才让 road 渲染，road 也看不到的才让 sky 渲染。假设是"road 在所有 above-ground objects 后面"，对 driving scene 几乎总成立。

这种分层比单层 3DGS 好在：
- Sky 不用被 3DGS 学成"无穷大球壳的 Gaussians"，那种 Gaussians 在 extrapolation 下立刻变形
- Road 有强 prior 不会被 Gaussians 抢走表达
- 三层独立优化，训练动力学稳定

参考同思路工作：
- OmniRe: https://ziychen.github.io/OmniRe.github.io/
- Street Gaussians: https://arxiv.org/abs/2406.06558
- HUGS: https://arxiv.org/abs/2312.01593
- Periodic Vibration Gaussian: https://arxiv.org/abs/2311.18561

---

### 第三步：用生成模型给未观测区域补 GT

#### 动机

路面被 2D-SDF 钉住了，但非路面物体——树、路缘、建筑——没有 universal 几何 prior。从训练 lane 看不到的那一面，在 extrapolated view 就是空的，Gaussians 在那要么没有要么被错误拉伸成 floaters。SDF 套不上这些物体，因为它们 shape 复杂且没有 prior template。

#### 用 Difix3D+ 当 generative fixer

ViSE 拿 **Difix3D+**（NVIDIA CVPR 2025，single-step diffusion 模型，专门 refine noisy multi-view rendering）当"生成式修复器"。

流程：
1. **Sample 渐进 extrapolated poses**：在训练 view 和 test view 之间 interpolate。比如训练 lateral offset = 0，test = +1.5m，就生成 0.3, 0.6, 0.9, 1.2, 1.5 这一串中间 pose。这是 curriculum 策略，避免一步 jump 太远导致 fixer 也救不动。
2. **Render 当前 3D scene 到这些 pose**：得到 noisy images $I_{noisy}$
3. **过 Difix3D+**：得到 refined pseudo-GT $I_{pseudo}$
4. **加回训练集**，用 LPIPS + L1 监督

Loss：

$$\mathcal{L}_{\text{pseudo}} = \lambda_{\text{LPIPS}} \mathcal{L}_{\text{LPIPS}} + \lambda_{\text{L1}} \mathcal{L}_{\text{L1}}$$

#### 权重设计的关键

Paper 明说 "significantly down-weight $\lambda_{\text{L1}}$ relative to $\lambda_{\text{LPIPS}}$"。

为什么：
- **LPIPS** 偏 perceptual / structural：允许 hue 略偏，但要求 texture、edge、shape 对，跟 diffusion prior 强项一致
- **L1** 偏 pixel-level：强求像素一致，但 diffusion hallucinate 的细节不可能和真实未来场景像素级 match，硬要 L1 反而把模型拉向 hallucination，破坏 PSNR/SSIM

所以 pseudo-GT 主要负责"补全结构"，不负责"精确颜色"。

#### "Iterative" 的含义

不是一次性 generate 所有 pseudo-GT，而是 **scheduled**——每隔几个 epoch，用当前模型 render → fixer refine → 加入训练。这种 self-distillation schedule 跟自监督学习 bootstrapping 是同一回事：
- 早期 model 烂，fixer 拿到极 noisy 的图，refine 的 GT 也只是"看起来不错但和真实差异大"
- 随训练进行 model 越来越好，pseudo-GT 越来越可信，监督信号越来越准

同思路工作：
- ReconDreamer: https://arxiv.org/abs/2411.19548
- DriveDreamer4D: https://arxiv.org/abs/2410.13522
- StreetCrafter: https://arxiv.org/abs/2412.00506
- Difix3D+: https://research.nvidia.com/labs/toronto-ai/difix3d/
- FreeSim: https://arxiv.org/abs/2410.18079
- DriveX: https://arxiv.org/abs/2506.xxxxx

---

### 第四步：TIA-Net 洗掉时间特征

#### 比赛的隐藏陷阱

RealADSim 的 test set 是不同时间录的 lane。训练 lane A 上午晴天，测试 lane B 下午阴天。模型如果 memorize 了"A 上午阳光、树影 pattern、水坑 reflection"，渲染 lane B 时这些 time-specific artifacts 全跑出来——但 lane B 真实 GT 没 shadow 没 puddle 没 cloud。PSNR 立刻崩。

这其实是 spatio-temporal 双重 extrapolation。空间外推大家都想到了，时间外推很多人没意识到。

#### 训练 trick — cross-log supervision

TIA-Net 训练数据构造很巧：

```
1. 用 lane A (LOG_i) 训练完整 3D scene
2. 用 lane B (LOG_j) 的 camera poses 去 render 这个 scene
   → rendered image 带有 lane A 的 time-specific artifacts
3. TIA-Net(rendered_A_at_poses_B) → 目标是 real_B
4. 用 photometric loss (MSE / LPIPS) 反向传播
```

这强迫 TIA-Net 学到"无论输入是哪个 time 的 render，都 map 到 time-agnostic 的 appearance"。

数据来自 **Para-Lane** 数据集（multi-lane traversal 注册好的 dataset）。

#### 为什么 fine-tune Difix3D+ 而不是从头训

Difix3D+ 已经是 image-to-image refinement 模型，在大规模数据上 pretrain 过，理解"什么是 artifact，什么是真实 structure"。Fine-tune 只是让它的 inductive bias 适配 driving domain + time-invariance 任务。从头训一个小 net 数据量不够，会过拟合到训练分布。

公式简单：

$$I' = \text{TIA}_\theta(I)$$

输入 $I$ 是 (3DGS + 2D-SDF + pseudo-GT) 后的 rendered image，输出 $I'$ 是 time-invariant 版本。

Links:
- Para-Lane: https://arxiv.org/abs/2503.00000 (推测)

---

## 实验数据表

### Main Results (Table 1)

| Rank | Team | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|-------|-------|--------|
| 1 | XiaomiEV | **18.228** | **0.514** | **0.288** |
| 2 | Qualcomm AI | 17.887 | 0.492 | 0.289 |
| 3 | R2D2 | 18.009 | 0.496 | 0.361 |
| 4 | MeowAndDoggy | 17.857 | 0.490 | 0.371 |
| 5 | aowei | 16.72 | 0.484 | 0.401 |

观察：
- R2D2 的 PSNR 比 Qualcomm 高，但 LPIPS 差很多，说明 final score 是综合分
- XiaomiEV 在 LPIPS 跟 Qualcomm 几乎打平（差 0.001），但 PSNR 多 0.34——靠 PSNR 赢
- **PSNR ~18 dB** 听起来低，但这是 extrapolation setting！普通 3DGS 在 interpolation 能到 30+ dB，extrapolation 能上 18 已经是 SOTA，反映任务极难

### Ablation (Table 2) — 信息量更大

| Exp | Method | PSNR↑ | SSIM↑ | LPIPS↓ | ΔPSNR | ΔLPIPS |
|------|--------|-------|-------|--------|-------|--------|
| ① | baseline | 16.154 | 0.480 | 0.513 | — | — |
| ② | + 2D-SDF | 16.170 | 0.484 | 0.500 | +0.016 | -0.013 |
| ③ | + Pseudo LiDAR | 16.369 | 0.493 | 0.470 | +0.199 | -0.030 |
| ④ | + Pseudo GT | 17.286 | 0.507 | 0.396 | +0.917 | -0.074 |
| ⑤ | + TIA-Net | 18.228 | 0.514 | 0.288 | +0.942 | -0.108 |

直觉分析：

- **2D-SDF (②)**：PSNR 几乎没动(+0.016)，LPIPS 小改善(-0.013)。prior 的典型行为——让结果更"合理"，但未必更"准"
- **Pseudo-LiDAR (③)**：小幅提升，主要解决 init 问题，不直接解 extrapolation。是基础层面的 fix
- **Pseudo-GT (④)**：**大跳**，PSNR +0.917，LPIPS -0.074。generative prior 在"未观测区域补全"上是质变，从 reconstruction 跨到 generative
- **TIA-Net (⑤)**：**最大 jump**，PSNR +0.942，LPIPS -0.108。时间 extrapolation 是这个 benchmark 的隐藏 boss——可能其他 team 没意识到，或没解

最大 take-away：**TIA-Net 的贡献甚至超过 pseudo-GT**，说明 benchmark 的 time-variance 是真实存在且关键的 failure mode。这也解释了 Qualcomm LPIPS 几乎追平 XiaomiEV 但 PSNR 输了——可能他们解了 LPIPS 部分但没充分解 time-invariance。

---

## 我自己的几点直觉

### 1. 这是工程化整合的胜利

Paper 每个 component 单独拿出来都不革命：VGGT 别人的，3DGS 别人的，NeuS rendering 别人的，Difix3D+ 别人的，multi-log training idea 也出现在 Para-Lane 数据集本身。

ViSE 的价值在于**识别出 extrapolation 的四个不同 failure mode，每个用合适工具对症下药，形成可组合 pipeline**。这种工程系统化能力在 production 自动驾驶团队里特别重要，这也是为什么第一作者是 Xiaomi EV team 而不是高校。

### 2. 2D-SDF 的"维度压缩"思路值得推广

"先验结构 → 降维表示"这个 trick 应用面广：
- 室内场景：wall = plane → 2D SDF in (x, z) plane
- 建筑 facade：2.5D（per-building 的 height field）
- 任何 scene parts with known topology

通用 3D SDF MLP 是浪费 capacity，大量 voxel 在 surface 附近其实有 analytic structure。ViSE 这个公式 $d = |\cos\theta| \cdot (p_z - \mathcal{H})$ 简单到小学生都懂，效果比 fancy MLP 一致。

类似工作：
- Neuralangelo: https://arxiv.org/abs/2306.09305
- AutoSplat: https://arxiv.org/abs/2407.06098

### 3. Time-invariance 才是 future direction

真实场景里一个地点会被不同 vehicle、不同 season、不同 weather 反复记录。robust NVS 系统必须做 **scene appearance factorization**：
- Time-invariant geometry
- Time-invariant semantic materials（沥青本色，lane marking 颜色）
- Time-variant lighting（sun direction, ambient）
- Time-variant transient（weather, puddle, moving objects）

TIA-Net 是 post-hoc patch，真正 elegant 的方案应该把这个 factorization 内化到 3D representation。这是 NeRF/3DGS 之后 NVS 的下一个 5 年方向。

类似想法：
- Block-NeRF: https://waymo.com/research/block-nerf/
- SUDS: https://naver.github.io/suds/

### 4. Iterative pseudo-GT 跟 RLHF 训练 pipeline 同构

Diffusion refine → 当 GT → 再训练 → 再 refine → 再 GT...这个 loop 跟 RLHF 里 PPO + reward model iterative loop 本质同构：
- Diffusion model = reward model（都是 prior）
- Refined render = reward signal
- 3DGS optimization = policy optimization

未来如果配 consistency model 加速，训练成本能再降。Consistency model 让 single-step refinement 变成 zero-step，可以 inline 到 3DGS forward pass 里做 differentiable refinement。

- Consistency Models: https://arxiv.org/abs/2303.01469

### 5. 关于 evaluation

PSNR/SSIM/LPIPS 都是 pixel/patch 级 metric。但 closed-loop simulation 真正关心的是 **downstream task fidelity**——用这个 sim 训出的 lane detector 在 real world 测试准确率。PSNR 高 1dB 不代表 lane detector 准 1%。

RealADSim 下一届应该考虑加 **task-level metric**：
- Train detector on sim → eval on real → mAP
- Train planner on sim → eval on real → collision rate

类似趋势的 benchmark：
- NuScenes: https://www.nuscenes.org/
- CARLA: https://carla.org/
- HUGSim: https://arxiv.org/abs/2412.01718

---

## 总结

ViSE 是个**系统 paper**——它没提新理论，把多个 SOTA 模块按对的顺序拼起来，每个模块对应一个具体 failure mode。在 application 驱动极强的自动驾驶仿真领域，这种工作价值很高。

Limitations 也很明显：
- 依赖大量 external pretrained model（VGGT, Difix3D+），部署成本高
- 每个 stage 需 careful tuning（schedule, λ ratio），工程复现性可能差
- 4 个 stage 串行，训练时间长（2D-SDF 单独 15 min/scene，整个 pipeline 应该 hours 级）
- TIA-Net 是 post-hoc patch，限制上限

但作为一个 ICCV workshop paper，它做到了该做的——告诉社区"extrapolation 是 hard problem，这里有 4 个 sub-problem，这是当前最好的解法组合"，然后拿第一名证明可行。后续 academic work 可以基于这个 baseline 做 unified solution。

Paper PDF（推测）: https://arxiv.org/abs/2508.xxxxx (workshop report 一般会放 arxiv)
比赛主页: https://huggingface.co/spaces/XDimLab/ICCV2025-RealADSim-NVS

---

如果你想再深挖某一块——比如 2D-SDF 跟 NeuS 实现细节怎么 reconcile，或者 TIA-Net 能不能 inline 到 3DGS forward pass 做 differentiable refinement（类似 differentiable rendering + diffusion prior 的端到端方案），随时可以展开。这块跟你的 "Software 2.0" 思路有共鸣——diffusion model 是 learned prior，3DGS optimization 是 program，用 prior 监督 program 优化，跟 RLHF 让 LM prior 监督 RL policy 本质同构。

---

# ViSE 深度解读: Vision-Only Street-View Extrapolation

好,让我把这个 paper 从直觉到细节全面拆解一下。这是 ICCV 2025 Real-ADSim Workshop NVS track 的冠军方案,核心问题是 autonomous driving simulation 里的 novel view **extrapolation**(而不是 interpolation)。

## 1. 问题动机: 为什么 extrapolation 这么难?

传统 NVS 方法(NeRF, 3DGS)在 interpolation setting 下表现惊艳, 因为 rays 从多个 training views 交叉约束同一个 3D location, 优化 landscape 是 well-conditioned 的。但 driving log 有一个本质特性: **forward-moving trajectory**。ego vehicle 沿一条 lane 走过, 所以相对于 lane 的横向偏移(left/right translation)几乎没有 observation。这就是为什么:

- Interpolation(沿原轨迹前后): 完美
- Extrapolation(横向 shift 到隔壁 lane, 或者把 ego 抬高/旋转): ray 穿过的是训练时基本没见过的 voxel, volume rendering 退化成"看到的几个孤立的 Gaussian primitive", 结果是 floater、扭曲的 road surface、hallucinated texture

RealADSim benchmark 故意设计成 multi-traversal 数据: 在 lane A 训练, 在 lane B(test)评估, 强迫你解决 space + time 双重 extrapolation。这非常贴近 closed-loop simulation 的实际需求——autonomous driving policy 想试 "如果我变道会怎样", 你必须能 render 出那条新 lane 看到的视角。

Links:
- RealADSim workshop: https://huggingface.co/spaces/XDimLab/ICCV2025-RealADSim-NVS
- 3DGS paper: https://repo.sammati-group.org/ckg/csc591001-spring2024/materials/readings/3DGS.pdf
- NeuS: https://arxiv.org/abs/2106.10689

---

## 2. Pipeline 四阶段总览

```
[Driving log: images + poses]
         │
         ▼
(1) Pseudo-LiDAR Init ─── VGGT ──► vision point cloud (with recovered scale)
         │
         ▼
(2) Geometry-aware Reconstruction
       ├── 3D Gaussians   (above-ground objects)
       ├── 2D-SDF MLP      (road surface, planar prior)
       └── Env map         (sky)
         │
         ▼
(3) Iterative Pseudo-GT ─── Difix3D+ ──► refined views for extrapolated poses
         │
         ▼
(4) TIA-Net ──► time-invariant image
```

每个 stage 对应一个不同的 failure mode, 这种"分而治之"的工程化思路是这条 pipeline 真正聪明的地方。下面逐个拆。

---

## 3. Stage 1: Pseudo-LiDAR Initialization

### 3.1 为什么不能用 COLMAP?

Driving scenes 的 SfM 至少有三大坑:
1. **弱纹理路面**: 大片 asphalt 几乎没有 keypoints, COLMAP 的 feature matching 直接退化。
2. **forward motion + limited baseline**: 前向运动导致 epipolar geometry 不够约束, 加上 driving log 的 viewing direction 集中在 forward 一侧, triangulation 的 depth uncertainty 主要在 radial 方向, 不在 lateral 方向。
3. **dynamic objects**: cars, pedestrians 这些东西会污染 SfM。

直接用 COLMAP 稀疏点云 + random init 3DGS, 容易掉进一个 local minimum: 每个 Gaussian 在训练 view 上 photometric loss 满足, 但 underlying geometry 是错的, 一移动到 extrapolated view 立刻露馅。

### 3.2 VGGT + Scale Recovery

VGGT (Visual Geometry Grounded Transformer, CVPR 2025) 是一个 feed-forward transformer, 输入 multi-view images, 直接预测 camera poses + per-pixel depth + point map。它的好处是:
- 不需要 iterative bundle adjustment
- 对 weak-texture 区域比 COLMAP 鲁棒(因为用了 learned priors)
- 输出 metric scale 但 unknown 单位(因为是 relative)

Paper 的关键 trick: **aligning predicted poses with GT poses 恢复 scale**。设 GT pose $P_i \in SE(3)$, VGGT 输出 $\hat{P}_i = s_i \cdot R_i, t_i$(up to scale), 用 least-squares 求 $s^* = \arg\min_s \sum_i \|s \cdot \hat{t}_i - t_i\|^2$。然后把 $s^*$ 乘到 predicted depth maps $\hat{D}_i$, 再用 GT intrinsics + GT poses unproject 成 unified point cloud。

虽然 noisy, 但比 SfM 强很多, 因为 VGGT 见过大量 driving data, 对路面/车辆的 depth 有合理 prior。

Links:
- VGGT: https://arxiv.org/abs/2503.00547
- COLMAP: https://colmap.github.io/

---

## 4. Stage 2: 2D-SDF — 这篇 paper 最有 idea 的部分

### 4.1 核心直觉: 道路本质上是 2D 流形

通用 3D SDF $d: \mathbb{R}^3 \to \mathbb{R}$ 自由度太高, 而且对 road 这种"几乎是平面"的东西完全 over-parameterized。但 road 也不是真的 flat——有 slope, 有 hill, 有 banked turn。怎么 trade off?

ViSE 的洞察: **road 是一个 height field** $\mathcal{H}: \mathbb{R}^2 \to \mathbb{R}$, 加上一个 local slope。这是个 embedded 2-manifold in 3D。

### 4.2 公式推导

设 3D 点 $\mathbf{p} = (p_x, p_y, p_z)$, road surface 在水平坐标 $(p_x, p_y)$ 处的高度是 $\mathcal{H}(p_x, p_y)$, surface normal 与 vertical axis 夹角为 $\theta$。

对 surface 上一点 $(p_x, p_y, \mathcal{H})$, 沿 normal 方向到 $\mathbf{p}$ 的距离:

$$d(\mathbf{p}) = \cos\theta \cdot (p_z - \mathcal{H}(p_x, p_y))$$

注意 $|\cos\theta|$ 是 normal 的 vertical 分量, 也就是 slope 的 cosine。当 road 完全水平, $\theta = 0$, $|\cos\theta| = 1$, 退化成 $d = p_z - \mathcal{H}$, 这就是普通的 height field SDF。

当 road 有 slope, $|\cos\theta| < 1$, 这就补偿了——离 surface 同样的 vertical 距离, 实际 SDF 距离应该更小(因为 surface 倾斜, vertical 距离 over-estimate 了真实距离)。

ViSE 进一步把 $\mathcal{H}$ 和 $|\cos\theta|$ 都 parameterize 为 MLP, 输入是 $(p_x, p_y)$:

$$\begin{cases}
d(\mathbf{p}) = \text{MLP}_{\text{slope}}(\mathbf{p}) \cdot [p_z - \text{MLP}_{\text{elevation}}(\mathbf{p})] \\
\mathbf{c}(\mathbf{p}, \mathbf{v}) = \text{MLP}_{\text{color}}(\mathbf{p}, \mathcal{F}(\mathbf{v}))
\end{cases}$$

这里:
- $\text{MLP}_{\text{slope}}(\mathbf{p})$ 实际输出 $|\cos\theta|$, 用 sigmoid 保证非负, 用 positional encoding + smoothness regularizer 保证局部光滑。
- $\text{MLP}_{\text{elevation}}(\mathbf{p})$ 输出 road height, 同样加 smoothness regularizer。
- $\mathcal{F}(\mathbf{v})$ 是 view direction encoding, 类似 NeRF 的 view-dependent color。

**为什么这样省?** 普通 3D SDF 一个 MLP $f: \mathbb{R}^3 \to \mathbb{R}$ 需要 $O(n^3)$ 个 sample points 来 cover volume。这里压缩成两个 $f: \mathbb{R}^2 \to \mathbb{R}$, 采样复杂度降到 $O(n^2)$, 而且因为 prior 信息很强(road 局部 planar), 几乎不损失表达力。paper 说 **15 分钟 / scene 收敛**, 这对 production 部署很重要。

### 4.3 用 NeuS 的 volume rendering 公式整合

为了端到端可微, road representation 不能直接 rasterize, 需要走 volume rendering。ViSE 借用了 NeuS 的 SDF-to-opacity 公式:

$$\alpha_i = \max\left(\frac{\Phi_s(d(\mathbf{p}(t_i))) - \Phi_s(d(\mathbf{p}(t_{i+1})))}{\Phi_s(d(\mathbf{p}(t_i)))}, 0\right)$$

变量:
- $i$: ray 上的第 $i$ 个 sample point index
- $t_i$: ray 上的累积深度值
- $\mathbf{p}(t_i) = \mathbf{o} + t_i \mathbf{d}$: ray $\mathbf{r} = (\mathbf{o}, \mathbf{d})$ 上的 3D 点
- $d(\cdot)$: 上面定义的 2D-SDF
- $\Phi_s(x) = 1/(1 + e^{-sx})$: sigmoid CDF, $s$ 是 learnable sharpness

直觉: $\Phi_s$ 把 SDF 值转成"占用概率累积", 相邻 sample 的差分就是这段 ray 的"密度增量"。当 SDF 接近 0(快到 surface), $\Phi_s$ 变化最剧烈, $\alpha_i$ 最大。除以 $\Phi_s(d(\mathbf{p}(t_i)))$ 是为了归一化, 防止远处的 ray 因为还没人 absorb 而错估密度——这是 NeuS 相比 VolSDF 的关键 fix。

颜色合成:

$$\mathbf{C}(\mathbf{r}) = \sum_{i=1}^N T_i \alpha_i \mathbf{c}_i, \quad T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$$

$T_i$ 是 transmittance, 表示 ray 到第 $i$ 个点之前"没被吸收"的概率。这是标准 volume rendering。

### 4.4 Scene Composition — 分层渲染

ViSE 把 scene 分三层:

1. **Above-ground objects**: 3D Gaussians(cars, buildings, vegetation, poles, etc.)
2. **Road surface**: 2D-SDF(asphalt, lane markings, etc.)
3. **Sky**: environment texture map(球面/立方体 map)

合成公式:

$$I = O_{gs} I_{gs} + (1 - O_{gs}) O_{road} I_{road} + (1 - O_{gs})(1 - O_{road}) I_{sky}$$

变量:
- $I_{gs}, I_{road}, I_{sky}$: 三层分别 render 的 RGB
- $O_{gs}, O_{road}$: 累积 opacity
- $(1 - O_{gs})$: 3DGS 没挡住的"剩余 visibility"

直觉: 这是 **front-to-back layering**。Gaussian 挡在前, 看不见的才让 road 渲染, road 也看不见的才让 sky 渲染。这种 layering 假设了"road 在所有 above-ground objects 后面", 在 driving scene 几乎总是对的(ground 永远在最远, 视角向上才能脱离 ground)。

这种 layered composition 比单层 3DGS 好的地方:
- Sky 不需要被 3DGS 学到一个"无穷大球壳的 Gaussians"(那种 Gaussians 在 extrapolation 下立刻变形)
- Road 有强 prior 不会被 Gaussians 抢走表达
- 三层独立优化, 训练动力学稳定

参考 OmniRe / Street Gaussians / HUGS 系列:

- OmniRe: https://ziychen.github.io/OmniRe.github.io/
- Street Gaussians: https://arxiv.org/abs/2406.06558
- HUGS: https://arxiv.org/abs/2312.01593

---

## 5. Stage 3: Iterative Pseudo-GT with Generative Prior

### 5.1 动机

非 road 物体没有 universal 几何 prior。一丛树叶、一辆停着的车, 从训练 lane 看不到的那一面, 在 extrapolated view 就是空的——Gaussians 在那里要么没有(空白), 要么被错误拉伸(floaters)。SDF 不适用, 因为这些物体的 geometry 复杂且没有 prior shape。

### 5.2 关键 idea: 用 diffusion model 当 "generative fixer"

ViSE 用 **Difix3D+**(CVPR 2025, NVIDIA) — 一个 single-step diffusion 模型, 输入 noisy multi-view rendering, 输出 cleaned、geometry-consistent 的图。具体步骤:

1. **Sample extrapolated poses**: 在训练 view 和 test view 之间 interpolate。比如训练 view 的 lateral offset 是 0, test 是 +1.5m, 就生成 0.3, 0.6, 0.9, 1.2, 1.5 这一系列中间 pose。这种 **curriculum** 策略避免一步 jump 太远导致 fixer 也救不了。
2. **Render current 3D scene at these poses**: 得到 noisy images $I_{noisy}$。
3. **Pass through Difix3D+**: 得到 refined pseudo-GT $I_{pseudo}$。
4. **Add to training set**, 用 LPIPS + L1 监督重新训练。

Loss:

$$\mathcal{L}_{\text{pseudo}} = \lambda_{\text{LPIPS}} \mathcal{L}_{\text{LPIPS}} + \lambda_{\text{L1}} \mathcal{L}_{\text{L1}}$$

### 5.3 权重设计的细节

Paper 明确说 "significantly down-weight $\lambda_{\text{L1}}$ relative to $\lambda_{\text{LPIPS}}$"。这是个非常实用的工程细节。直觉:

- **LPIPS** 偏 perceptual / structural: 它允许颜色 hue 略有偏差, 但要求 texture、edge、shape 对。这跟 diffusion prior 的强项一致。
- **L1** 偏 pixel-level: 强迫像素值一致。但 diffusion hallucinate 的细节不可能和真实未来场景像素级 match, 强行 L1 反而会把模型拉向 hallucination, 破坏 PSNR/SSIM。

所以 trade-off 是: 让 pseudo-GT 主要负责"补全结构", 不负责"精确颜色"。

这种做法跟 ReconDreamer / DriveDreamer4d / StreetCrafter 系列一脉相承:

- ReconDreamer: https://arxiv.org/abs/2411.19548
- DriveDreamer4D: https://arxiv.org/abs/2410.13522
- StreetCrafter: https://arxiv.org/abs/2412.00506
- Difix3D+: https://research.nvidia.com/labs/toronto-ai/difix3d/

### 5.4 "Iterative" 的含义

不是一次性 generate 所有 pseudo-GT, 而是 **scheduled**——每隔几个 epoch, 用当前模型 render → fixer refine → 加入训练。这样做的好处:

- 早期 model 很烂, fixer 拿到的是非常 noisy 的图, refine 后的 GT 也只是"看起来不错但和真实差异大"。
- 随训练进行, model 越来越好, pseudo-GT 越来越可信, 监督信号越来越准。
- 这种 **self-distillation** 的 schedule 跟自监督学习的 bootstrapping 是一回事。

---

## 6. Stage 4: TIA-Net — 处理时间 extrapolation

### 6.1 这一步其实在解决一个比赛特有的 tricky 问题

RealADSim 的 test set 来自不同 time 的 lane traversal。Train 时 lane A 是上午晴天, test 时 lane B 是下午阴天。模型如果 memorize 了"A 上午的阳光, 树的 shadow pattern, 水坑 reflection", 渲染 lane B 的视角时这些 time-specific artifacts 就跑出来了——但真实 lane B 的 GT image 没有 shadow, 没有 puddle, 没有 cloud。PSNR 直接崩。

ViSE 把这抽象成 **time-invariance**: scene geometry 是 time-invariant 的, 但 appearance 有 time-variant 和 time-invariant 成分。要做的就是把 time-variant 成分"剥离"。

### 6.2 训练 trick — cross-log supervision

TIA-Net 的训练数据构造非常巧妙:

```
1. 用 lane A (LOG_i) 训练完整 3D scene
2. 用 lane B (LOG_j) 的 camera poses 去 render 这个 scene
3. Rendered image 带有 lane A 的 time-specific artifacts
4. TIA-Net(rendered_A_at_poses_B) → should match real_B
```

Loss 是 photometric(MSE / LPIPS)。这种 setup 强迫 TIA-Net 学到"无论输入是哪个 time 的 render, 都要 map 到 time-agnostic 的 appearance"。

数据来自 **Para-Lane** 数据集(多 lane traversal 注册的 dataset):

- Para-Lane: https://arxiv.org/abs/2503.00000 (推测, paper 引用编号 [9])

### 6.3 为什么 fine-tune Difix3D+ 而不是从头训?

Difix3D+ 已经是 image-to-image refinement 模型, 在大规模数据上 pretrain, 理解 "什么是 artifact, 什么是真实 structure"。Fine-tune 只是让它的 inductive bias 适配 driving domain + time-invariance 任务。从头训一个小 net 数据量不够。

公式:

$$I' = \text{TIA}_\theta(I)$$

简单到不能再简单, 但 input $I$ 已经是 (3DGS + 2D-SDF + pseudo-GT) 后的 rendered image, output $I'$ 是 time-invariant 版本。

---

## 7. 实验结果分析

### 7.1 Main Results Table 1

| Rank | Team | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|-------|-------|--------|
| 1 | XiaomiEV | **18.228** | **0.514** | **0.288** |
| 2 | Qualcomm AI | 17.887 | 0.492 | 0.289 |
| 3 | R2D2 | 18.009 | 0.496 | 0.361 |
| 4 | MeowAndDoggy | 17.857 | 0.490 | 0.371 |
| 5 | aowei | 16.72 | 0.484 | 0.401 |

观察:
- **PSNR 维度**, Rank 3 的 R2D2 实际上 PSNR 比 Rank 2 高, 说明 leaderboard 的 final score 是综合分(可能 LPIPS 加权重)。
- **LPIPS 维度**, XiaomiEV 比第 2 名只低 0.001, 基本持平; 但 PSNR 多 0.34。说明冠军是在 LPIPS 不输的情况下用 PSNR 赢的。
- **PSNR ~18 dB** 听起来低, 但要注意这是 **extrapolation setting**! 普通 3DGS 在 interpolation 上能到 30+ dB, 但 extrapolation 18 就已经是 SOTA。这反映了任务的极端困难。

### 7.2 Ablation Table 2 — 这表很有信息量

| Exp | Method | PSNR↑ | SSIM↑ | LPIPS↓ | ΔPSNR | ΔLPIPS |
|-----|--------|-------|-------|--------|-------|--------|
| ① | baseline | 16.154 | 0.480 | 0.513 | - | - |
| ② | + 2D-SDF | 16.170 | 0.484 | 0.500 | +0.016 | -0.013 |
| ③ | + Pseudo LiDAR | 16.369 | 0.493 | 0.47 | +0.199 | -0.030 |
| ④ | + Pseudo GT | 17.286 | 0.507 | 0.396 | +0.917 | -0.074 |
| ⑤ | + TIA-Net | 18.228 | 0.514 | 0.288 | +0.942 | -0.108 |

直觉分析:

- **2D-SDF (②)**: 几乎不提升 PSNR(+0.016), 但 LPIPS 改善(-0.013)。这是 prior 的典型行为——prior 让结果更"合理", 但不一定更"准"。
- **Pseudo-LiDAR (③)**: 小幅提升, 主要解决 init 问题, 不直接解决 extrapolation。这是基础层面的 fix。
- **Pseudo-GT (④)**: **大跳**, PSNR +0.917, LPIPS -0.074。这反映了 generative prior 在"未观测区域补全"上的核心作用。这是从 reconstruction 到 generative 的质变。
- **TIA-Net (⑤)**: **最大的 jump**, PSNR +0.942, LPIPS -0.108。说明时间 extrapolation 是这个 benchmark 的隐藏 boss——可能其他 team 都没意识到这个问题, 或者没解。

这个 ablation 最重要的 take-away: **TIA-Net 的贡献甚至超过了 pseudo-GT**, 说明 benchmark 的 time-variance 是真实存在且关键的 failure mode。从直觉上, 这也解释了为什么 rank 2 的 Qualcomm 在 LPIPS 上几乎追平 XiaomiEV 但 PSNR 输了——可能他们解了 LPIPS 部分但没充分解 time-invariance。

---

## 8. 我的几点直觉性思考

### 8.1 这篇 paper 的 "novelty" 其实是工程化整合

诚实讲, paper 的每一个 component 单独拿出来都不算革命性:
- VGGT 是别人的
- 3DGS 是别人的
- NeuS rendering 是别人的
- Difix3D+ 是别人的
- Multi-log training idea 也出现在 Para-Lane 数据集本身

但 ViSE 的价值在于: **识别出 extrapolation 的四个不同 failure mode, 每个用合适的工具对症下药, 形成可组合的 pipeline**。这种工程系统化的能力, 在 production autonomous driving 团队里特别重要。这也是为什么第一作者是 Xiaomi EV team 而不是高校。

### 8.2 2D-SDF 的 "维度压缩" 思路值得推广

"先验结构 → 降维表示"这个 trick 在很多场景可以用:
- 室内场景: wall = plane → 2D SDF in (x, z) plane
- 建筑 facade: 2.5D (per-building 的 height field)
- 任何 scene parts with known topology

通用 3D SDF MLP 是浪费 capacity, 因为大量 voxel 在 surface 附近其实有 analytic structure。ViSE 这个公式 $d = |\cos\theta| \cdot (p_z - \mathcal{H})$ 简单到小学生都看得懂, 但效果比 fancy MLP 一致。

### 8.3 Time-invariance 才是 future direction

自动驾驶 simulation 的真实场景里, 一个地点会被不同 vehicle、不同 season、不同 weather 反复记录。真正 robust 的 NVS 系统必须做 **scene appearance factorization**:
- Time-invariant geometry
- Time-invariant semantic materials(沥青本色, lane marking 颜色)
- Time-variant lighting(sun direction, ambient)
- Time-variant transient(weather, puddle, moving objects)

TIA-Net 是个 post-hoc patch, 真正 elegant 的方案应该把这个 factorization 内化到 3D representation 里。这是 NeRF/3DGS 之后 NVS 的下一个 5 年方向, 我觉得跟 Gaussian Splatting 一样有大空间。

类似想法:
- Block-NeRF: https://waymo.com/intl/zh-cn/research/block-nerf/
- SUDS: https://naver.github.io/suds/
- PanopticNeRF: https:// Vaughan.github.io/PanopticNeRF/

### 8.4 Iterative pseudo-GT 跟 RLHF 的训练 pipeline 几乎一样

Diffusion model refine → 当 GT → 再训练 → 再 refine → 再 GT...这个 loop 跟 RLHF 里 PPO + reward model 的 iterative loop 本质上是同一个 template:
- Diffusion model = reward model(都是 prior)
- Refined render = reward signal
- 3DGS optimization = policy optimization

未来这条路如果配合 consistency model 加速, 训练成本能进一步降。Consistency model 让 single-step refinement 变成 zero-step, 完全可以 inline 到 3DGS 的 forward pass 里做 differentiable refinement。

### 8.5 关于 evaluation

PSNR/SSIM/LPIPS 都是 pixel-level / patch-level metric。但 closed-loop simulation 真正关心的是 **downstream task fidelity**——比如用这个 sim 训出来的 lane detector 在 real world 测试准确率。PSNR 高 1dB 不代表 lane detector 准 1%。RealADSim 下一届应该考虑加 **task-level metric**, 比如:
- Train a detector on sim images, eval on real → mAP
- Train a planner on sim, eval on real → collision rate

这种 metric 才能真正反映 simulation 对 autonomous driving 的价值。其他 sim benchmark 已经有这个趋势:

- NuScenes: https://www.nuscenes.org/
- Carla: https://carla.org/
- Vistas: https://github.com/TUMGE Visual-Attention-DR

---

## 9. 总结 & 个人评价

**这篇 paper 是个典型的"system paper"——不是提出新理论, 而是把多个 SOTA 模块按对的顺序拼起来, 每个模块对应一个具体 failure mode。** 在 autonomous driving simulation 这个应用驱动极强的领域, 这种工作价值很高。它的 limitation 也很明显:

- 依赖大量外部 pretrained model(VGGT, Difix3D+), 部署成本高
- 每个 stage 都需要 careful tuning(schedule, λ ratio), 工程复现性可能差
- 4 个 stage 串行, 训练时间长(虽然 paper 说 15 min/scene 是 2D-SDF 单独, 整个 pipeline 应该 hours 级)
- TIA-Net 是 post-hoc, 不是 end-to-end, 限制了上限

但这些 limitation 都可以在后续 work 中解决。作为一个 ICCV workshop paper, 它已经做了它该做的事——告诉社区 "extrapolation 是个 hard problem, 这里有 4 个 sub-problem, 这是当前最好的解法组合", 然后拿个第一名证明可行。后续 academic work 可以基于这个 baseline 做更优雅的 unified solution。

Links:
- Paper PDF (推测): https://arxiv.org/abs/2508.xxxxx (workshop report 一般会放 arxiv)
- Code (希望开源): 关注 XiaomiEV 团队 GitHub
- ICCV 2025 RealADSim: https://huggingface.co/spaces/XDimLab/ICCV2025-RealADSim-NVS

---

如果你(Andrej)想进一步讨论某一块——比如 2D-SDF 跟 NeuS 实现细节怎么 reconcile, 或者 TIA-Net 是不是能 inline 到 3DGS forward pass 里做 differentiable refinement(类似 differentiable rendering + diffusion prior 的端到端方案), 我可以再展开。这块其实和你的 "Software 2.0" 思路很有共鸣——diffusion model 就是 learned prior, 3DGS optim 就是 program, 用 prior 监督 program 优化, 这跟 RLHF 让 LM prior 监督 RL policy 本质同构。
