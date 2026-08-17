---
source_pdf: WRISTWORLD.pdf
paper_sha256: d5f73fdc9be5a2ddaba8a05b6c65bce7fe365f0eca8c48f93a71b73e99cc7c8e
processed_at: '2026-08-13T06:19:03-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 WristWorld

## 一句话版本

机器人练动作，第三人称视角的视频一抓一大把，手腕上摄像头的视频却很少。这篇 paper 做的事：**光看第三人称视频，就能"脑补"出手腕视角该看到什么**，然后拿这个假视频去训练 VLA，效果还真能涨。

---

## 问题出在哪

想象你学抓东西。有人给你看一百段"别人从旁边拍你手"的视频，但几乎没有"你眼睛贴在手腕上看"的视频。哪个对学精细操作更有用？当然是手腕视角——你能看到手指和物体接触的那一瞬间。

Droid 有 76k 条轨迹，Open X-Embodiment 更大，但 wrist view 的覆盖率很低。采集 wrist view 要额外装 camera、标定、同步，成本高。于是数据严重不对称：anchor view（第三人称，external camera）一堆，wrist view（第一人称，egocentric）稀缺。

现有 world model 想生成 wrist view？基本都需要你先给一帧 wrist view 的图当 condition（SVD、Cosmos-Predict2、WoW 14B 都这样）。问题来了——你本来就是因为没有 wrist view 才想生成，结果生成又需要 wrist view，死循环。

WristWorld 打破这个循环：**纯 anchor view 输入，零 wrist view 输入，输出 wrist view video**。

参考：
- Droid dataset: https://droid-dataset.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- Calvin benchmark: https://calvinrobot.github.io/

---

## 核心直觉：用几何当"骨架"，用 diffusion 当"皮肤"

这篇 paper 最本质的 idea，用比喻讲：

你想画一张从手腕看出去的画，但你只看过第三人称。你脑子里其实知道两件事：
1. **房间长什么样**（桌子在哪、物体在哪、机械臂怎么动的）——这是几何
2. **东西看起来是什么质感**（颜色、纹理、光影）——这是外观

WristWorld 把这两件事拆开做：

**第一步（Reconstruction）**：先用 VGGT 这种 visual geometry model，从多个 anchor view 把场景的 3D 结构和 wrist camera 的位姿算出来。然后把这个 3D 点云"投影"到 wrist 视角，得到一张稀疏的、只有几何骨架的 condition map。这张 map 告诉生成模型："手腕视角下，物体的大致轮廓和位置应该长这样。"

**第二步（Generation）**：用 video diffusion model（基于 Wan 1.3B DiT），在 condition map 的几何约束下，加上 CLIP 从 anchor view 提取的语义信息，生成有纹理、有细节、时间连贯的 wrist view video。

这就像画素描先打骨架再上色。骨架保证空间对，颜色保证内容对。

---

## 两个关键创新，用大白话讲

### 创新 1：Wrist Head——怎么猜手腕相机在哪

VGGT 原本能从多视角图像预测 3D 点云、camera pose、dense matching。WristWorld 给它加了个"wrist head"——一个小 transformer decoder，吃 VGGT 的 fused feature，吐出 wrist camera 的旋转 $\mathbf{R}_w$ 和平移 $\mathbf{T}_w$。

直觉：即使没见过 wrist view 图像，anchor views 之间的几何关系 + 机械臂的运动模式，已经隐含了"手腕此刻在哪、朝哪看"的信息。wrist head 就是把这个隐含信息 decode 出来。

### 创新 2：SPC Loss——没有 ground truth 怎么监督

这是最聪明的地方。问题：我们没 wrist camera 的真实 pose，也没 depth，怎么训练 wrist head？

WristWorld 的招：VGGT 能给我们两样东西——
1. anchor view 和 wrist view 之间的 dense 2D-2D 点匹配（哪个点对应哪个点）
2. 从 anchor view 重建的 3D 点云

逻辑链：
- anchor view 里有个像素 $\mathbf{u}_q^j$，它对应 3D 空间里的点 $\hat{\mathbf{y}}_j$
- VGGT 的 matching head 说：这个 $\mathbf{u}_q^j$ 在 wrist view 里应该对应到像素 $\hat{\mathbf{u}}_w^j$
- 我们的 wrist head 预测了一个 wrist pose $(\mathbf{R}_w, \mathbf{T}_w)$
- 用这个 pose 把 $\hat{\mathbf{y}}_j$ 投影到 wrist view，得到 $\mathbf{u}_w^{\prime j}$
- 如果 pose 对，$\mathbf{u}_w^{\prime j}$ 应该和 $\hat{\mathbf{u}}_w^j$ 重合

SPC loss 就是让它们重合。具体分两部分：
- 前面的点（$z_j > 0$，相机前方）：算 reprojection MSE
- 后面的点（$z_j < 0$，相机后方，不合理）：惩罚，逼它跑到前面来

公式：
$$\mathcal{L}_u = \frac{1}{|S_{\mathrm{front}}|} \sum \mathrm{MSE}(\mathbf{u}_w^{\prime j}, \hat{\mathbf{u}}_w^j)$$
$$\mathcal{L}_{\mathrm{depth}} = -\frac{1}{|S_{\mathrm{back}}|} \sum z_j$$
$$\mathcal{L}_{\mathrm{proj}} = \lambda_u \mathcal{L}_u + \lambda_{\mathrm{depth}} \mathcal{L}_{\mathrm{depth}}$$

$z_j$ 是 3D 点在 wrist camera 坐标系下的深度值。背面的点 $z_j < 0$，加个负号 $-z_j > 0$，loss 会推 $z_j$ 往正方向走，避免 pose 退化到把场景甩到相机背后的 trivial 解。

一句话：**用 VGGT 自己的 matching 结果当伪标签，反过来监督 wrist pose，不需要任何真实 wrist 标注**。

---

## Generation 阶段：两种 condition 打配合

Diffusion model 生成视频时，给它两种 guidance：

**Dense condition（像素级）**：wrist view projection 的 condition map 经 VAE 编码成 latent $\mathbf{z}_c^t$，和 noisy wrist latent $\mathbf{z}_w^t$ 在 channel 维拼接：
$$\mathbf{z}^t = [\mathbf{z}_w^t; \mathbf{z}_c^t]$$
这样每个像素位置都有几何 hint，DiT 去噪时每一步都能看到"这里应该有个物体边缘"。

**Sparse condition（全局语义）**：每个 anchor view 每帧过 CLIP image encoder，得到 embedding $\mathbf{e}_{t,i}$，再加 temporal embedding $\mathbf{p}_{\mathrm{temporal}}$ 和 view identity embedding $\mathbf{p}_{\mathrm{view}}$，和 text embedding 一起作为 condition token 注入 DiT。

$$\mathbf{c} = [\tilde{\mathbf{E}}_{\mathrm{clip}} + \mathbf{p}_{\mathrm{temporal}}^{1:T} + \mathbf{p}_{\mathrm{view}}^{1:N}; \tilde{\mathbf{E}}_{\mathrm{text}} + \mathbf{p}_{\mathrm{text}}]$$

Dense 保证空间对齐，sparse 保证内容正确（别把杯子生成成方块）。这跟 ControlNet 的思路很像——ControlNet 也是 dense condition（canny、depth map）+ text prompt 的组合。

参考 ControlNet: https://arxiv.org/abs/2302.05543

---

## 实验结果：到底好不好用

### 视频生成质量（Table 1）

最狠的对比：WoW 14B 用了 wrist first frame（更强 condition），FVD 在 Droid 上 935；WristWorld 不用 first frame，FVD 421。Franka 上更夸张，231 vs 985。

LPIPS、SSIM、PSNR 全面碾压。说明不光视频流畅，单帧质量也更好。

### VLA 下游任务（Table 2, Calvin）

VPP 加了 WristWorld 生成的 wrist view 当训练数据：
- 平均任务完成长度 3.67 → 3.81（+3.81%）
- 5 个连续任务全完成率 55.4% → 60.4%（+5%）
- 缩小了 anchor-wrist 性能 gap 的 42.4%

long-horizon 任务提升最大，符合直觉——越长越精细的任务，wrist view 越值钱。

### 真实机器人 Franka Panda（Table 3）

三个任务平均成功率：anchor only 37.8% → 加生成 wrist view 53.3%。单任务最高涨 20 个点。生成的假数据真能帮真机器人干活。

### Plug-and-Play（Table 5）

把 WristWorld 当插件接在 Cosmos 或 WoW 14B 后面：原本它们需要 wrist first frame，接上 WristWorld 后只需 left view first frame，FVD 从 1156/985 降到 467/455。说明 WristWorld 是个通用的 multi-view 扩展器。

### 消融（Table 4）

- 去掉 wrist projection：FVD 421 → 3091，崩了。几何骨架是命根子。
- 去掉 SPC loss：FVD 421 → 790。pose 不准，projection map 就歪，生成跟着歪。
- 去掉 CLIP：FVD 421 → 474。语义丢了，小物体容易糊。

三个组件缺一不可，但 projection 最关键。

---

## 我的直觉与联想

**1. 这本质是 "geometry as inductive bias for generation"**

纯 generative model（diffusion）在极端视角变换下会崩，因为训练分布里没这种 mapping。VGGT 注入的几何先验相当于告诉 diffusion model："别瞎猜空间关系，我已经算好了。"这跟 Sora 试图学世界物理但经常翻车形成对比——Sora 纯靠 scale 硬学物理规律，WristWorld 是显式把物理/几何结构塞进去。

Sora 参考: https://openai.com/sora/

**2. "伪标签自监督"的套路很经典**

SPC loss 的思路——用 model A（VGGT matching）的输出当伪标签监督 model B（wrist head）——在 self-supervised learning 里到处都是。Noisy Student、DINO、MAE 都有类似味道。关键在于 model A 的输出要够准、够 dense，而 VGGT 恰好满足。

DINO: https://arxiv.org/abs/2104.14294
MAE: https://arxiv.org/abs/2111.06377

**3. Dense + Sparse conditioning 是 video gen 的通用 pattern**

ControlNet 用 dense spatial condition + text。WristWorld 用 dense projection latent + CLIP token。EnerVerse 用 Gaussian splatting + text。这个 pattern 会越来越普遍——纯 text-to-video 不够 controllable，加一路 dense spatial condition 是正解。

EnerVerse: https://arxiv.org/abs/2501.01895

**4. 数据增强 > 模型改进？**

Calvin 上 VPP 不改架构、不加 loss、不换 backbone，只多了一路生成 wrist view 当训练数据，就涨了 3.81%。这暗示当前 VLA 的瓶颈可能不在模型容量，而在数据的多视角覆盖。如果把 Open X-Embodiment 全量跑一遍 WristWorld，给所有轨迹补 wrist view，VLA 社区可能迎来一波 "synthetic data scaling"。

**5. 局限与风险**

- VGGT 对透明/反光/小物体重建差，condition map 会糊，生成跟着崩
- SPC loss 依赖 VGGT matching 质量，anchor-wrist 视角差太大时 matching 噪声大
- 动态场景下 per-frame point cloud 时间一致性可能不够，paper 说 4D 但 condition map 是逐帧独立投影的，没有显式 4D motion modeling
- Wan 1.3B 的 prior 有限，分布外物体 generalization 存疑
- 生成的 wrist view 再拿去训 VLA，万一有 systematic bias（比如物体形状系统性地偏一点），VLA 会学到错误的手眼协调

**6. 更大的图景**

这篇 paper 在暗示一个趋势：**未来的 world model 不会是纯 generative 黑盒，而是 geometric reasoning + generative synthesis 的 hybrid**。VGGT 这类 visual geometry model 已经足够强，可以作为 generative model 的 "geometry brain"。这跟你在 YC LLM 讲座里强调的 "model needs to understand world structure, not just predict tokens" 思路一致——纯 next-token prediction 学不会空间推理，得显式注入结构。

Karpathy 的 world model 观点参考: https://www.youtube.com/watch?v=VAM3dDXB7RA

---

## 一句话收尾

WristWorld 干的事：拿现成的 geometry model（VGGT）当骨架，拿现成的 video diffusion model（Wan）当皮肤，用个聪明的 SPC loss 把两者缝起来，就能从第三人称视频凭空生成第一人称手腕视频，还能反哺 VLA 训练。工程上不复杂，idea 上很干净——**geometry is all you need (for cross-view generation)**。

---

# WristWorld: 从 Anchor Views 生成 Wrist-View 视频的 4D World Model

## 1. 核心问题与动机

这篇 paper 解决的是 robotic manipulation 中一个很实际的数据不对称问题：大规模 robot datasets（如 Droid, Open X-Embodiment）通常只有丰富的 third-person / anchor views，但缺少 wrist-mounted camera 的 egocentric views。而 wrist view 对 VLA (Vision-Language-Action) 模型极为关键，因为它直接捕捉 fine-grained hand-object interaction，是 precise manipulation 的基础。

关键矛盾在于：现有的 video world models（SVD, Wan, Cosmos-Predict2 等）如果要生成 wrist view，通常需要一个 wrist-view 的 first frame 作为 condition。这就形成死循环——你想要 wrist 数据来训练，但生成 wrist 数据又需要 wrist 数据。WristWorld 的突破点是：**完全从 anchor views 出发，无需任何 wrist-view 输入**，就能合成几何和时间一致性的 wrist-view video。

作者从 VGGT (Visual Geometry Grounded Transformer) 这类 visual geometry models 中获得灵感——这类模型能从多视角 RGB 直接预测 3D point cloud、camera pose、dense correspondences，提供了跨极端视角变换所需的几何先验。相关参考：
- VGGT: https://vgg-t.github.io/
- Droid dataset: https://droid-dataset.github.io/
- Calvin benchmark: https://calvinrobot.github.io/

---

## 2. 方法架构：两阶段 4D Generative World Model

整个 pipeline 分为 Reconstruction 和 Generation 两阶段，核心思想是「先用几何重建把 wrist camera pose 和结构 condition map 算出来，再用 video diffusion 在这个几何骨架上生成真实纹理」。

### 2.1 Preliminary：Video Diffusion 与 VGGT

**Video Diffusion** 的 forward/reverse 过程。视频 $\mathbf{X} = \{x^t\}_{t=1}^T$ 经 video VAE 压缩成 latent $\mathbf{Z}_0 \in \mathbb{R}^{T \times C \times H \times W}$，其中 $T$ 是帧数，$C$ 是 latent channel 数，$H, W$ 是空间分辨率。训练目标：

$$\mathcal{L}_{\mathrm{diff}} = \mathbb{E}_{\mathbf{Z}_0, \epsilon, \tau} \|\epsilon - \epsilon_\theta(\mathbf{Z}_\tau, \tau \mid \mathbf{c})\|_2^2$$

这里 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ 是采样噪声，$\tau$ 是 diffusion timestep，$\mathbf{Z}_\tau$ 是 $\mathbf{Z}_0$ 在第 $\tau$ 步加噪后的 latent，$\mathbf{c}$ 是 condition（text / image embedding），$\epsilon_\theta$ 是要学习的 denoising network（DiT）。

**VGGT** 给定多视角图像，输出 fused feature $\mathbf{F}$ 以及 3D 量（point cloud, depth, correspondences）。对于 query 点 $\mathbf{u}_q^j$（$j$ 是 query index，$j=1,\dots,M$，$M$ 是采样点数），matching head 预测其他 view $I_i$（$i=1,\dots,N$，$N$ 是 anchor view 数）中的对应点 $\hat{\mathbf{u}}_i^j$，构成 correspondence set：

$$\mathcal{C} = \{(\mathbf{u}_q^j, \hat{\mathbf{u}}_i^j)\}_{i=1,\dots,N}^{j=1,\dots,M}$$

针孔相机模型下，3D 点 $\hat{\mathbf{y}} \in \mathbb{R}^3$ 投影到像素 $\mathbf{u} \in \mathbb{R}^2$：

$$\mathbf{u} = \Pi(\mathbf{K}, \mathbf{R}, \mathbf{T}, \hat{\mathbf{y}})$$

其中 $\mathbf{K}$ 是 intrinsics，$\mathbf{R} \in SO(3)$ 是 rotation，$\mathbf{T} \in \mathbb{R}^3$ 是 translation，$\Pi(\cdot)$ 是 projection function（含 perspective division）。

### 2.2 Reconstruction Stage：Wrist Head + SPC Loss

**Wrist Head Design**：在 VGGT 的 aggregated multi-view feature $\mathbf{F}$ 基础上，引入一组 learnable wrist queries $\mathbf{q}_w$，通过 transformer decoder 做 cross-attention，回归 wrist camera extrinsics：

$$(\mathbf{R}_w, \mathbf{T}_w) = \mathrm{WristHead}(\mathbf{F}, \mathbf{q}_w)$$

其中 $\mathbf{R}_w \in SO(3)$ 是 wrist 相机旋转，$\mathbf{T}_w \in \mathbb{R}^3$ 是 wrist 相机平移。wrist head 实现为 3 层 transformer decoder，8 个 attention heads，embedding dim 1024。关键 insight 是：即使训练时没有 wrist-view 图像，anchor views 之间的几何关系 + 机器人 forward kinematics 隐含的 wrist 运动模式，足以让 model 学会预测 wrist pose。

**Spatial Projection Consistency (SPC) Loss**——这是本文最有意思的设计。问题在于：我们没有 wrist camera 的 ground-truth extrinsics 或 depth 来直接监督 $\mathbf{R}_w, \mathbf{T}_w$。但 VGGT 能给我们：
1. anchor-wrist 的 dense 2D-2D correspondences $\mathcal{C} = \{(\mathbf{u}_q^j, \hat{\mathbf{u}}_w^j)\}_{j=1}^M$（注意这里 $\hat{\mathbf{u}}_w^j$ 是 VGGT matching head 预测的 wrist view 中的对应像素，下标 $w$ 表示 wrist view，上标 $j$ 是 point index）
2. 从 anchor views 重建的 3D point cloud $\mathcal{V} = \{\hat{\mathbf{y}}_k\}$

把 anchor 像素 $\mathbf{u}_q^j$ 关联到其 3D 点 $\hat{\mathbf{y}}_j$，得到 3D-2D pairs：

$$\mathcal{T} = \{(\hat{\mathbf{y}}_j, \hat{\mathbf{u}}_w^j)\}_{j=1}^M$$

然后用预测的 wrist pose $(\mathbf{R}_w, \mathbf{T}_w)$ 把 3D 点投影到 wrist view：

$$\mathbf{u}_w^{\prime j} = \Pi(\mathbf{K}, \mathbf{R}_w, \mathbf{T}_w, \hat{\mathbf{y}}_j)$$

$\mathbf{u}_w^{\prime j}$ 是模型预测 pose 下的投影像素，$\hat{\mathbf{u}}_w^j$ 是 VGGT 给的「应该是」的对应像素。两者应该一致。

根据投影后深度 $z_j$（wrist camera 坐标系下的 z 值）的正负，把点分成两组：
- $S_{\mathrm{front}} = \{\hat{\mathbf{y}}_j \mid z_j > 0\}$：相机前方的点
- $S_{\mathrm{back}} = \{\hat{\mathbf{y}}_j \mid z_j < 0\}$：相机后方的点（不合理，应被惩罚）

SPC loss：

$$\mathcal{L}_u = \frac{1}{|S_{\mathrm{front}}|} \sum_{\hat{\mathbf{y}}_j \in S_{\mathrm{front}}} \mathrm{MSE}(\mathbf{u}_w^{\prime j}, \hat{\mathbf{u}}_w^j)$$

$$\mathcal{L}_{\mathrm{depth}} = -\frac{1}{|S_{\mathrm{back}}|} \sum_{\hat{\mathbf{y}}_j \in S_{\mathrm{back}}} z_j$$

$$\mathcal{L}_{\mathrm{proj}} = \lambda_u \mathcal{L}_u + \lambda_{\mathrm{depth}} \mathcal{L}_{\mathrm{depth}}$$

Intuition：$\mathcal{L}_u$ 是标准的 reprojection error——前面可见点的投影要和 VGGT 匹配的像素对齐；$\mathcal{L}_{\mathrm{depth}}$ 是个 trick——背面的点 $z_j < 0$，加负号后 $-z_j > 0$，loss 会推这些点的 $z_j$ 变大（朝 0 或正方向），迫使他们跑到相机前方，避免 pose 退化到把场景甩到身后的 trivial 解。这个设计避免了需要 depth GT 的问题，纯 RGB correspondences 就能监督 wrist pose。

**Condition Map Generation**：用估计的 wrist poses 把重建的 3D scene（per-frame point cloud）投影到 wrist image plane，得到一串 temporally aligned 的 condition maps。这些 map 是稀疏的、结构性的（只有点云投影的像素），但携带了 wrist view 的几何骨架。

### 2.3 Generation Stage：DiT + Projection + CLIP Semantics

基于 Wan 1.3B DiT（text-to-video 预训练），做两处修改：

**1. Latent 拼接**：把 wrist-view projection $\mathbf{C}^t$ 经 VAE 编码成 $\mathbf{z}_c^t$，和 noisy wrist latent $\mathbf{z}_w^t$ 在 channel 维拼接：

$$\mathbf{z}^t = [\mathbf{z}_w^t; \mathbf{z}_c^t]$$

输入从 $(T, C, H, W)$ 变成 $(T, 2C, H, W)$，patch embedding 的 in-channels 从标准 16 扩展到 32（2×2 conv patch）。这样 geometry condition 是 pixel-aligned 的，直接注入到去噪过程的每一步。

**2. CLIP-Encoded Anchor Semantics**：condition map 只是点云投影，可能丢掉小物体、模糊区域的全局语义。所以每个 anchor view 每帧过 CLIP image encoder：

$$\mathbf{E}_{\mathrm{clip}} = \{\mathbf{e}_{t,i}\}_{i=1,\dots,N; t=1,\dots,T} \in \mathbb{R}^{(NT) \times d_c}$$

$\mathbf{e}_{t,i}$ 是第 $t$ 帧、第 $i$ 个 anchor view 的 CLIP embedding，$d_c=512$。text prompt 也编码，两者投影到 shared conditioning space：

$$\tilde{\mathbf{E}}_{\mathrm{clip}} = W_c \mathbf{E}_{\mathrm{clip}}, \quad \tilde{\mathbf{E}}_{\mathrm{text}} = W_t \mathbf{E}_{\mathrm{text}}$$

最终 condition tokens：

$$\mathbf{c} = [\tilde{\mathbf{E}}_{\mathrm{clip}} + \mathbf{p}_{\mathrm{temporal}}^{1:T} + \mathbf{p}_{\mathrm{view}}^{1:N}; \tilde{\mathbf{E}}_{\mathrm{text}} + \mathbf{p}_{\mathrm{text}}]$$

$\mathbf{p}_{\mathrm{temporal}}^{1:T}$ 是 temporal positional embedding（编码帧序），$\mathbf{p}_{\mathrm{view}}^{1:N}$ 是 view-identity embedding（区分 N 个 anchor view），$\mathbf{p}_{\mathrm{text}}$ 是 text token positional embedding。token 维度拼接，最多 512 个 condition tokens。CFG=5.0。

这里有两层条件：pixel-aligned 的 geometry（projection latent concat）提供空间结构，token-level 的 CLIP+text 提供全局语义。这种「dense + sparse」双路 conditioning 在 video generation 里是常见且有效的设计模式。

---

## 3. 实验数据解读

### 3.1 Video Generation 定量对比（Table 1）

| Method | Wrist First Frame | Droid FVD↓ | Droid LPIPS↓ | Droid SSIM↑ | Droid PSNR↑ | Franka FVD↓ | Franka LPIPS↓ | Franka SSIM↑ | Franka PSNR↑ |
|---|---|---|---|---|---|---|---|---|---|
| VGGT | × | — | 0.74 | 0.28 | 9.56 | — | 0.73 | 0.49 | 12.05 |
| Pix2Pix | × | — | 0.55 | 0.58 | 12.81 | — | 0.58 | 0.71 | 15.60 |
| WoW 1.3B | × | 1142.15 | 0.61 | 0.46 | 10.08 | 1944.59 | 0.72 | 0.53 | 10.45 |
| SVD | ✓ | 2005.44 | 0.56 | 0.50 | 11.12 | 1354.56 | 0.60 | 0.68 | 14.10 |
| Cosmos-Predict2 | ✓ | 1990.72 | 0.51 | 0.56 | 12.74 | 1156.69 | 0.65 | 0.67 | 12.59 |
| WoW 14B | ✓ | 935.03 | 0.53 | 0.54 | 11.98 | 985.99 | 0.59 | 0.68 | 13.93 |
| **Ours** | **×** | **421.10** | **0.39** | **0.64** | **14.78** | **231.43** | **0.33** | **0.78** | **17.84** |

关键观察：
- **FVD 大幅领先**：Droid 上 421 vs WoW 14B 的 935，Franka 上 231 vs 985。FVD 衡量 temporal coherence 和视频分布距离，说明生成视频的时间一致性远超 baseline。即使 baseline 用了 wrist first frame（更强 condition），WristWorld 不用 first frame 还更好。
- **LPIPS/SSIM/PSNR 全面领先**：说明单帧 perceptual quality 和结构保真度都更好。
- **不需要 first frame 是本质优势**：黄色行（SVD, Cosmos, WoW 14B）需要 wrist first frame，意味着它们无法用于「anchor-only 数据增强」场景。绿色行（VGGT, Pix2Pix, WoW 1.3B）不需要但质量差。WristWorld 是唯一同时满足「无 first frame」+「高质量」的方法。

### 3.2 VLA 下游性能（Table 2, Calvin）

用 VPP (Video Prediction Policy) 作为 VLA backbone，加 WristWorld 生成的 wrist view 作为额外输入：

| Method | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | Avg. Len |
|---|---|---|---|---|---|---|
| VPP (anchor only) | 91.2% | 82.2% | 73.2% | 65.2% | 55.4% | 3.67 |
| VPP + Ours | 92.9% ↑1.7 | 84.2% ↑2.0 | 75.4% ↑2.2 | 67.6% ↑2.4 | 60.4% ↑5.0 | 3.81 ↑0.14 |

Avg. Len 提升 3.81%，5/5（完整完成 5 个连续任务）提升 5%。注意 long-horizon（5/5）提升最大，说明 wrist view 在精细、长时序任务中价值更高。对照「Anchor + Wrist（真实 GT）」的 upper bound，WristWorld 缩小了 anchor-wrist 性能 gap 的 42.4%。

### 3.3 真实机器人 Franka Panda（Table 3）

| Inputs | Close upper drawer | Pick bread → drawer | Pick milk | Mean |
|---|---|---|---|---|
| Anchor + Wrist (GT) | 80.0% | 73.3% | 46.7% | 66.7% |
| Anchor only | 60.0% | 40.0% | 13.3% | 37.8% |
| Anchor + Ours Gen | 73.3% ↑13.3 | 53.3% ↑13.3 | 33.3% ↑20.0 | 53.3% ↑15.5 |

生成的 wrist view 让 mean success 从 37.8% 跳到 53.3%，单任务最高提升 20%。这证明生成数据是「有用信号」而非噪声。

### 3.4 Plug-and-Play 扩展单视角 World Model（Table 5）

| Method | Input | FVD↓ | LPIPS↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|---|---|
| Cosmos (alone) | Wrist first frame | 1156.69 | 0.65 | 0.67 | 12.59 |
| WoW 14B (alone) | Wrist first frame | 985.99 | 0.59 | 0.68 | 13.93 |
| **Ours + Cosmos** | Left view first frame | 467.19 ↓689.50 | 0.58 ↓0.07 | 0.70 ↑0.03 | 14.66 ↑2.07 |
| **Ours + WoW 14B** | Left view first frame | 455.57 ↓530.42 | 0.57 ↓0.02 | 0.71 ↑0.03 | 14.60 ↑0.67 |

思路：先用 single-view world model（Cosmos/WoW）从 left view 生成 anchor rollout，再用 WristWorld 把 anchor rollout 转成 wrist video。这样原本需要 wrist first frame 的 model，现在只需 left view first frame。FVD 暴跌，说明 WristWorld 是个通用的「multi-view 扩展插件」。

### 3.5 消融实验（Table 4）

| Wrist Projection | Ext CLIP | SPC Loss | FVD↓ | LPIPS↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|---|---|---|
| × | ✓ | × | 3091.74 | 0.74 | 0.55 | 10.42 |
| ✓ | ✓ | × | 790.10 | 0.59 | 0.47 | 10.75 |
| ✓ | × | ✓ | 474.32 | 0.44 | 0.61 | 13.67 |
| ✓ | ✓ | ✓ | 421.10 | 0.39 | 0.64 | 14.78 |

- **Wrist Projection 最关键**：去掉后 FVD 从 421 飙到 3091，LPIPS 从 0.39 恶化到 0.74。没有 geometry condition，DiT 完全不知道往哪生成。
- **SPC Loss 很重要**：有 projection 但没 SPC，FVD 790 → 421，LPIPS 0.59 → 0.39。SPC 让 projection map 更准，直接提升下游生成质量。
- **CLIP Semantics 有辅助**：去掉 CLIP，FVD 474 → 421，PSNR 13.67 → 14.78。CLIP 补全了 projection 丢失的全局语义。

---

## 4. 训练细节与超参

**Reconstruction Stage**（Table 6）：
- Backbone: VGGT-1B（frozen），只训练 wrist head
- Image size: 518×518
- Optimizer: AdamW, weight decay 0.05, lr 2e-5 cosine decay
- Batch size: 4/GPU × grad accum 3
- Hardware: 8×A800, ~12h pretrain + 6h finetune

**Generation Stage**（Table 7）：
- Backbone: Wan 1.3B DiT（text-to-video 预训练）
- Resolution: 640×480, latent scale 1/8
- LoRA: rank 4, α=4, targets {q,k,v,o,ffn}
- Condition tokens: 512 (CLIP + text + temporal/view)
- CFG: 5.0
- Optimizer: AdamW, lr 1e-5
- Hardware: 8×A800, ~24h pretrain + 12h finetune

注意用了 LoRA 微调 Wan DiT，这是参数高效的做法，避免全量微调破坏预训练 video prior。

---

## 5. Intuition 与我的思考

从 Karpathy 你经常强调的「world model 需要可预测的、结构化的 latent」视角看，这篇工作有个很漂亮的分解：

1. **Geometry 作为 inductive bias**：纯 end-to-end 的 video diffusion（如 Wan/SVD）在 anchor→wrist 这种极端视角变换下会失败，因为训练分布里几乎没有这种 cross-view mapping。VGGT 提供的几何先验相当于注入了一个强结构约束——wrist view 不是凭空想象，而是 anchor view 几何的确定性变换。SPC loss 进一步把这个约束变成可微的监督信号。

2. **两阶段解耦的可解释性**：Reconstruction 负责「where」（camera pose + point cloud structure），Generation 负责「what」（纹理、语义）。这比一个黑盒 model 直接 anchor→wrist 更可控。condition map 可视化（Figure 4）能直接看到几何对齐质量，便于 debug。

3. **Dense + Sparse Conditioning 的互补**：Projection latent 是 dense、pixel-aligned 的（每个像素都有几何 hint），CLIP token 是 sparse、global 的（捕获物体类别、场景语义）。前者保证空间一致性，后者保证内容正确性。这跟 ControlNet 用 dense condition + text 用 sparse condition 的思路一致。

4. **可能的局限/hallucination 联想**：
   - Point cloud 投影的 condition map 是稀疏的，对于小物体、透明物体、反光物体（VGGT 重建差的地方）会失效。
   - SPC loss 依赖 VGGT 的 2D-2D correspondence 质量，如果 anchor-wrist 视角差太大，matching 本身会噪声大。
   - 动态场景（articulated arm 大幅运动）下，per-frame point cloud 的时间一致性可能不够，虽然 paper 声称 4D，但 condition map 是逐帧独立投影的。
   - 生成质量依赖 Wan 1.3B 的 prior，对于训练分布外的新物体/新场景可能 generalization 有限。

5. **与相关工作的联系**：
   - EnerVerse (https://arxiv.org/abs/2501.01895) 也做 4D + multi-view diffusion，但用 Gaussian splatting，需要 wrist view 初始化。
   - Tesseract (https://arxiv.org/abs/2504.20995) 做 4D embodied world model，思路类似但场景不同。
   - GR-1/GR-2 (https://arxiv.org/abs/2410.06158) 是 VLA 侧的工作，用大规模 video pretraining，WristWorld 可以作为它的数据增强工具。
   - Exocentric-to-Egocentric video generation (Liu et al., NeurIPS 2024, https://arxiv.org/abs/2311.13681) 做人际视角转换，但不是 robotics 场景。

6. **更深层 insight**：这篇 paper 本质上是在说——「visual geometry model (VGGT) 已经学到了足够强的 3D 理解，可以作为 video diffusion model 的 geometry brain」。这预示着一个趋势：未来 world model 不会是纯 generative 的，而是 geometric reasoning + generative synthesis 的组合。这跟你之前讲的「LLM 需要 world model 来做 long-horizon planning」的论点一致——纯 next-token prediction 不够，需要结构化的空间理解。

---

## 6. 总结

WristWorld 的贡献可以概括为：
1. **首次**实现 anchor-only → wrist-view video 生成，无需 wrist first frame
2. **SPC loss** 是核心创新——用纯 RGB correspondences 监督 wrist pose，避免 depth/extrinsics GT 依赖
3. **两阶段 + dense/sparse 双 condition** 设计，几何与语义互补
4. **plug-and-play** 可扩展任意 single-view world model 到 multi-view

对 VLA 社区的意义：提供了一种 scalable 的数据增强路径，把海量 anchor-view dataset（Droid 76k, Open X-Embodiment）自动转成 multi-view dataset，无需额外传感器采集。Calvin 上 3.81% avg length 提升 + 42.4% gap 缩小，证明生成数据是 effective supervision。

Project page 和 code 链接在 paper abstract 里有标注（§ Code），建议关注后续 release。
