---
source_pdf: HY-World 2.0 A Multi-Modal World Model for Reconstructing, Generating,
  and Simulating 3D Worlds.pdf
paper_sha256: 6a92b680015f1318de716eb64fe3a7fccccd3cc18d835a4ab845deafc5f1973b
processed_at: '2026-08-05T08:42:01-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HY-World 2.0 大白话版

## 这玩意儿到底干啥的

一句话:你给它一句话或者一张图,它给你造一个**能走进去逛的 3D 世界**。你给它一堆照片或者视频,它也能帮你把真实场景的 3D 结构重建出来。两条路用同一套底层引擎。

打个比方,以前 world generation 和 reconstruction 是两个工种——一个负责"编造",一个负责"测绘"。HY-World 2.0 把这俩活儿合并成一个岗位,还让一个人把两件事都干好。

项目地址:https://3d-models.hunyuan.tencent.com/world/
代码:https://github.com/Tencent-Hunyuan/HY-World-2.0

---

## 整个流程像拍电影分四步走

### 第一步:HY-Pano 2.0 — 先拍一张 360° 全景

你给它一句话比如"现代极简风的开放式办公区",或者一张普通视角的照片,它先给你生成一张完整的 $360° \times 180°$ 的 panorama。这张全景图就是整个 3D 世界的"种子"。

**老版本怎么干的**:HY-World 1.0 用 explicit camera intrinsic 做 perspective→ERP 的几何 warping,需要知道 focal length、FoV 这些 metadata。问题是你随便给一张网图,这些信息基本都没有,warpping 就会扭曲。

**新版本怎么干的**:HY-Pano 2.0 直接放弃 explicit camera prior,用 [MMDiT](https://arxiv.org/abs/2212.09748)(Multi-Modal Diffusion Transformer)把 conditional image 的 latent 和 panoramic noise latent concat 成一串 token,让 self-attention 自己学 perspective 到 ERP 的隐式映射。说白了就是"我不告诉你怎么投,你自己看着学"。

**边界接缝问题**:ERP 图左右边界在 $360°$ wrap 处会有明显接缝。两层处理:
1. Latent 层:对 feature 做 circular padding,去噪时强制周期边界
2. Pixel 层:decode 到像素后沿 equirectangular edge 做线性 blending

$$I_{\text{final}}(x, y) = (1 - w(x)) \cdot I_{\text{decode}}(x, y) + w(x) \cdot I_{\text{decode}}(x \bmod W, y)$$

$w(x)$ 在 edge 区域从 0 平滑过渡到 1,让左右边界像素互相混色消除接缝。

**数据**:real-world panorama + Unreal Engine synthetic 渲染混着用,严格过滤 stitching artifact 和露出的相机设备。

效果看 Table 4:T2P 上 CLIP-T 0.258 领先,I2P 上 5 个指标全第一。Q-Align quality score 比 HY-World 1.0 从 3.317 涨到 4.026,提升明显。

---

### 第二步:WorldNav — 规划相机怎么走

有了 panorama 当种子,接下来要决定"相机要去哪些地方拍什么样的画面"。这一步本质是给后面 view generation 准备 camera trajectory。

**先做 scene parsing**:
- 用 [MoGe2](https://arxiv.org/abs/2507.02546) 估单目深度,再用 LSMR (Least-Squares Minimal Residual) 在 42 个 perspective 子视图之间对齐拼成 panoramic point cloud $P^{\text{pan}}$。HY-World 1.0 只用 12 views,这版提到 42 views + GPU 加速 solver。
- [Qwen3-VL](https://arxiv.org/abs/2505.09388) 识别关键 landmark 和 obstacle,[SAM3](https://arxiv.org/abs/2511.16719) 出 2D semantic mask,再投影到 3D。
- [Recast Navigation](https://github.com/recastnavigation/recastnavigation) 生成 NavMesh,用 ray-casting 修正地面、KD-Tree erosion、bridge polygon 连接孤立区域。

**五种 trajectory 模式**(Figure 5,Table 1):

| 类型 | 数量上限 | 干啥的 |
|---|---|---|
| Regular | 9 | 在 panorama 中心把全景切成 3 个 120° 视图,每个绕中心做 +45° pitch + ±120° azimuth 轨道,再加 +60° azimuth 给 aerial 视角 |
| Surrounding | 5 | 围绕最重要的物体环绕,radius 根据物体大小自适应,72 候选节点 ray-casting 过筛,Dijkstra 连接 |
| Recon-Aware | 10 | 专攻 under-observed 区域(mesh 中 stretched sharp face),NMS 提取代表 center,iterative orbiting |
| Wandering | 3 | NavMesh 按 8 个 angular sector 切,Dijkstra distance field 找最远点,适合 corridor、street 这种窄空间 |
| Aerial | 8 | 在 surrounding/wandering 上叠加 +45° pitch,动态降低防碰撞 |

加起来最多 35 条轨迹。

**直觉**:这五种模式覆盖了"全方位概览、重点物体环绕、补盲区、走到边界、俯视"这五种典型探索行为。Figure 19 的消融展示只用 panoramic view 训练 3DGS 全是空洞,逐步加 trajectory 逐步完善。Aerial 提供 BEV (Bird's Eye View) 观测提升视角切换自由度。

---

### 第三步:WorldStereo 2.0 — 沿轨迹生成新视角

这是 paper 最硬核的部分。任务:沿着上一步规划的 trajectory 生成一系列 keyframe,保证跨轨迹一致,后续能拿去做 3D 重建。

#### 核心改造 1:Keyframe-VAE 替换 Video-VAE

**问题**:标准 video diffusion([HunyuanVideo](https://arxiv.org/abs/2412.03603)、[Wan](https://arxiv.org/abs/2503.20314)、[CogVideoX](https://arxiv.org/abs/2408.06072))用 Video-VAE 做 spatio-temporal 压缩。相机一快动,Video-VAE 编码就出 motion blur 和 geometric distortion,直接毁掉下游 3D 重建。Figure 8 对比很扎眼。

**思路**:借鉴 [FlashWorld](https://arxiv.org/abs/2510.13678)。承认一个事实——对于 3D 重建,**viewpoint coverage 比 frame continuity 重要**。中间那些冗余帧只是消耗 token budget 没啥用。

**做法**:对每个 keyframe $V_i$ 独立用 causal-padding image encoder 编码:

$$F_i \in \mathbb{R}^{1 \times \frac{H}{8} \times \frac{W}{8} \times C}$$

只做 8× spatial 压缩,不做时间压缩。$T_{kf} \ll T_{vid}$,帧数远少但通过加大 sampling interval 维持 viewpoint coverage。每个 keyframe 独立处理,VAE 编码解码天然 parallelizable。

**Table 7 消融最有意思的一行**:freeze cross-attention + FFN 后 RotErr 0.492 最低、User Quality 64.39% 最高。Full training 虽然 visual metric 高但 camera precision 下降,因为 overfitting 导致 image style drift。这种"选择性 freeze"是性能与泛化的最佳平衡点,很实战的工程 insight。

#### 核心改造 2:Explicit Camera Control

两条路并行:
1. **Plücker rays**:每条 ray 6 维 (origin $o \in \mathbb{R}^3$ + direction $d \in \mathbb{R}^3$),编码像素级 camera ray
2. **Point cloud**:从 reference view 提取 $P^{\text{ref}}$,warp 到 target view:

$$P_i^{\text{tar}}(x) \simeq R_i^{cw} D(x) K_i^{-1} \hat{x}$$

- $R_i^{cw}$:target view $i$ 的 camera-to-world 旋转
- $K_i$:intrinsic matrix
- $D(\cdot)$:[MoGe2](https://arxiv.org/abs/2507.02546) 估的 monocular depth
- $\hat{x}$:像素 homogeneous 坐标

Warp 后 render 成 view-wise keyframe,再过 Keyframe-VAE 编码。

#### 核心改造 3:Memory 机制 — GGM + SSM++

跨轨迹一致性怎么保?两套互补 memory:

**Global-Geometric Memory (GGM)**:用扩展的全局点云做"粗"结构 prior:

$$P^{\text{glo}} = [P^{\text{ref}}, \hat{P}] \in \mathbb{R}^{(N+\hat{N}) \times 3}$$

$P^{\text{ref}}$ 是 reference view 点云,$\hat{P}$ 是从 $T_g$ 个 novel view 随机采样的额外点。Inference 时直接用 $P^{\text{pan}}$ 当 global guidance 覆盖 360°。

**数据增强**(模拟 imperfect depth):50% 双线性下采样模拟"depth bleeding",10% Gaussian filter 造 floater,50% real-world 保留原始 noise。**关键经验**:像 [Neoverse](https://arxiv.org/abs/2601.00393) 那种 aggressive point cloud distortion 对 GGM 反而有害,过度削弱 guidance 导致跨视频几何不一致。

**Improved Spatial-Stereo Memory (SSM++)**:用 retrieved image 做"细"对应。这是 SSM 的重大升级:

**(a) 不再独立 branch**,retrieved keyframe 直接进 main DiT。

**(b) RoPE 修改**(Figure 11):target 和它的 retrieved reference 沿水平轴拼接,width 变 2W,retrieved view 继承配对 target 的 temporal index。

**这个 spatial stitching 是核心设计**。Table 8 的 A* vs A 对比很有说服力——把 retrieved 通过 temporal concatenation 集成(传统 long video generation 那种)所有指标严重退化。Spatial stitching 让 attention 在 target-retrieval pair 内自然建立 correspondence,这跟传统 [stereo matching](https://www.science.org/doi/10.1126/science.194.4262.283) 的 insight 一致——对应关系在空间维度建立才对。

**(c) Selective retrieval**:不强制每帧 retrieve,只取最相关 $T_r$ 个 keyframe($T_r < T_{kf}$),降冗余。

**(d) Full attention** 替代 restricted attention(除 cross-attention),让模型学所有 target+retrieved 的全局 context。

**(e) Implicit camera embedding**:7 维 pose (quaternion 4D + translation 3D) → 3-layer MLP → camera token,zero-init 加到 feature。替代 WorldStereo 1.0 的 explicit pointmap guidance,更灵活。

**Memory bank**:初始用 panorama 切的 perspective view,生成 keyframe 增量加入。Retrieval 基于 3D FoV similarity,上限 $T_r$ keyframe。

**训练数据**(Figure 10):
- 现有 multi-view 数据:temporally misaligned retrieval,30%-90% temporal overlap,部分 frame 在 target trajectory 之外增加难度
- Synthetic UE 数据:multi-trajectory retrieval,基于 3D FoV similarity 从 alternative trajectory 选

#### 核心改造 4:DMD 蒸馏加速

基于 [Distribution Matching Distillation](https://arxiv.org/abs/2405.14867),通过近似 KL divergence 蒸馏 4-step student:

$$\nabla \mathcal{L}_{\text{DMD}} = -\mathbb{E}_t \left( \int (s_{\text{real}}(x_t, t) - s_{\text{fake}}(x_t, t)) \frac{dx_t}{d\theta} dz \right)$$

- $x = G_\theta(z)$:student 给定 noise $z$ 的生成
- $s_{\text{real}}$:frozen real score
- $s_{\text{fake}}$:trainable fake score,每次 $G_\theta$ 更新训 5 次
- Stochastic gradient truncation 稳定训练
- 省略 GAN loss(影响不显著还减速)

**关键进步**:WorldStereo 1.0 只在 camera control task 蒸馏且 freeze memory branch(annotated memory data 短缺),2.0 借 SSM++ 灵活 + UE 数据丰富,实现 memory-based full fine-tuning 蒸馏。**有意思的是 DMD 后 T&T AUC 从 58.19 升到 60.09**,蒸馏反而提升——说明 full-step diffusion 在某些 metric 上有冗余甚至噪声,few-step 起到正则化效果。

#### Table 8 消融故事

- Camera only baseline: PSNR 16.13
- +GGM+SSM++(A): 20.94,大涨
- +Trainable FFN(B): 21.56,camera error 也降
- +Point cloud aug(C): 21.36,小损 clean metric 但提 robustness
- +Reference aug(D): 20.86
- +Camera embedding(E): 21.06
- **A* (temporal concat SSM): 19.83**——大幅退化,验证 spatial stitching 核心价值
- +Doubled batch(F): 21.63,稳定训练
- **+DMD(G): 21.84,最优**

---

## 第四步:WorldMirror 2.0 — 重建出 3D 几何

生成完 keyframe,要拿来做 3D 重建。WorldMirror 2.0 既是 generation pipeline 的 Stage IV backbone,也是独立 reconstruction foundation model 服务 dense multi-view input。基于 [WorldMirror 1.0](https://arxiv.org/abs/2510.10726),三大改进:

### 改进 1:Normalized Position Encoding

**问题**:WorldMirror 1.0 用标准 [RoPE](https://arxiv.org/abs/2104.09864),patch 用 absolute integer index $(i, j)$。测试分辨率高于训练就 position extrapolation,低于训练就 distribution shift。Table 12 显示 WorldMirror 1.0 high resolution 时 camera AUC@30 从 86.13 暴跌到 66.29,NVS PSNR 从 21.34 崩到 17.78。

**做法**:借鉴 [DINOv3](https://arxiv.org/abs/2508.10104),把 absolute integer 换成 normalized 坐标,所有 patch 映射到 $[-1, 1]$:

$$\hat{x}_i = \frac{2i + 1}{H_p} - 1, \quad \hat{y}_j = \frac{2j + 1}{W_p} - 1$$

- $i, j$: patch grid index
- $H_p = H/p$, $W_p = W/p$: patch grid 尺寸
- $+1$ offset 保证 pixel-center alignment,防边界塌缩

**核心 insight**:把 extrapolation 转 interpolation。8-patch 训练 grid 是 $[0, 7]$,16-patch inference 时 $[8, 15]$ 全 out-of-distribution;normalized 后两者都在 $[-1, 1]$,inference 只是更密采样。Figure 13 验证 normalized RoPE 跨分辨率 cosine similarity > 0.95,标准 RoPE 严重退化。

### 改进 2:Depth-to-Normal Loss 显式耦合

**问题**:1.0 中 depth 和 normal head 独立监督。Real-world 数据 depth annotation 噪声大,monocular depth pseudo label 多视图不一致。

**做法**:用 normal pseudo-label 反向监督 depth。把预测 depth back-project 成 3D 点再算 cross product 得 normal:

$$\tilde{N}_i(x) = \text{normalize}\left(\frac{\partial P_i}{\partial u} \times \frac{\partial P_i}{\partial v}\right), \quad P_i = K^{-1} \hat{D}_i \cdot [u, v, 1]^\top$$

- $K$:camera intrinsic
- $\hat{D}_i$:预测 depth
- $[u, v, 1]^\top$:pixel homogeneous
- 偏导用四方向 finite difference 近似

Loss 是推导 normal 与 target normal 的 angular error:

$$\mathcal{L}_{d2n} = \frac{1}{|\mathcal{V}|} \sum_{x \in \mathcal{V}} \arccos\left(\frac{\tilde{N}_i \cdot \hat{N}_i}{\|\tilde{N}_i\| \|\hat{N}_i\|}\right)$$

**妙处**:不依赖完整 depth GT,通过 normal pseudo-label 间接监督 depth。Surface normal 描述局部 orientation 不需要全局 metric consistency,在 multi-view 设置下天然更鲁棒。Synthetic 数据用 GT depth 推 normal,real-world 用 monocular normal teacher 预测的 pseudo normal。

### 改进 3:Depth Mask Prediction Head

**问题**:1.0 用 learned confidence weight 调 loss 但 inference 时没显式 per-pixel validity,下游只能用启发式 threshold。

**做法**:加 dedicated head 输出 per-pixel validity logit $\hat{m}(x)$,BCE loss 训练:

$$\mathcal{L}_{\text{mask}} = -\frac{1}{|\mathcal{M}|} \sum_{x \in \mathcal{M}} [m^*(x) \log \sigma(\hat{m}(x)) + (1 - m^*(x)) \log(1 - \sigma(\hat{m}(x)))]$$

Synthetic GT mask 来自 rendering pipeline,real-world 用 extreme depth/discontinuity/sky 识别 pseudo label。Inference 时输出 mask 让下游选择性 filter。

### 训练策略改造

**Token-budget-first batch sizing**:固定每 GPU token budget $T_{\max}$(如 25000),先采样 resolution 算 per-image token count $t = \frac{H}{p} \times \frac{W}{p}$,再推导最大 view 数:

$$N_{\max} = \min\left(N_{\text{cap}}, \left\lfloor \frac{T_{\max}}{t} \right\rfloor\right)$$

$N_{\text{cap}}$ 是架构上限(如 48)。这种"反过来"的策略让 GPU memory 利用率接近满载,消除 OOM 错误,不管采何种 resolution。本质上是把硬件约束作为 first-class 设计目标,值得推广。

**三阶段 curriculum**:
1. 所有 geometry head 用 native annotation 训练,无 pseudo-label
2. 加 $\mathcal{L}_{d2n}$,增 synthetic data 比例提精度
3. 冻结 backbone+geometry head,用 depth head 权重初始化训 3DGS head

### 推理效率

三种互补策略:
- **Token-level SP**:Transformer backbone 的 token sequence 跨 GPU 分区,attention 层 All-to-All 重分配
- **Frame-level SP**:DPT decoder conv 层按帧分区
- **BF16 mixed-precision**:大部分参数 BF16,precision-critical 模块保留 FP32
- **FSDP**:每个 Transformer block 和 DPT head 作为独立 FSDP unit shard 参数

Table 14:4 GPU + SP + BF16 + FSDP,128 views 推理 5.60s / 42.71GB,256 views 17.52s / 78.78GB。baseline 单 GPU 256 views 直接 OOM。

### WorldMirror 2.0 效果

Table 11:7-Scenes Acc.@M 从 1.0 的 0.043 降到 2.0 的 0.033,@H 从 0.079 降到 0.037(差距大),+all priors 后 @H 0.012。

Table 12:Camera AUC@30 @H 从 66.29 升到 86.89(20 分提升),NVS PSNR @H 从 17.78 稳到 19.98。

Table 13:Surface normal 在 ScanNet、NYUv2、iBims-1 都最佳,ScanNet mean error 从 1.0 @H 17.6 降到 2.0 @H 12.5。

---

## 第五步:World Composition — 拼成最终 3D 世界

### Point Cloud Expansion

WorldMirror 2.0 估 per-frame depth 和 normal:

$$\{D_i^m, N_i^m\}_{i=1}^{T_{ex}'} = \Phi(\{V_j, C_j\}_{j=1}^{T_{pan}}, \{V_i, C_i\}_{i=1}^{T_{ex}'})$$

$\Phi$ 是 WorldMirror 2.0 网络,输入是 panorama 切的 perspective view + 生成的 keyframe subset。

### Depth Alignment

WorldMirror depth 存在 scale ambiguity,未对齐 $P^{\text{pan}}$ 世界坐标。渲染 $P^{\text{pan}}$ 得 sparse guidance depth $D_i^g$,做 RANSAC linear alignment:

$$D_i^a = \gamma_i D_i^m + \beta_i$$

只在 reliability mask $M_i$ 定义的有效区域估 $\gamma_i, \beta_i$:

$$M_i = M_i^m \cap M_i^g \cap M_i^n \cap M_i^p \cap \overline{M_i^{sky}}$$

- $M_i^m$:WorldMirror confidence 有效区
- $M_i^g$:panoramic guidance 有效区
- $M_i^n$:normal consistency mask,排除 normal 角度偏差 > 90°
- $M_i^p$:percentile statistical filter 排 outlier
- $\overline{M_i^{sky}}$:SAM3 video mode 识别的非 sky mask

**Outlier detection**:设 $Q=9$ 个 anchor depth 均匀分布,每帧算变换后 anchor 值 $\mathcal{V}_{i,q} = \gamma_i A_q + \beta_i$,最大相对偏差超 90% percentile 的 $(\gamma_i, \beta_i)$ 视为 outlier,用同序列最近 inlier 替换。整序列 outlier 则全丢。

最终 $\tilde{P} = \text{voxel\_downsample}(P^{\text{pan}} \cup P^{\text{ex}})$。

### 3DGS 优化

**Initialization**:每个 3D Gaussian 参数化 opacity $\sigma_k \in [0,1]$、center $\mu_k \in \mathbb{R}^3$、covariance $\Sigma_k = R_k S_k S_k^\top R_k^\top$、view-independent RGB $c_k \in \mathbb{R}^3$(放弃 Spherical Harmonics,生成场景无显著 view-dependent effect)。

**MaskGaussian 解决 densification dilemma**:

Dilemma:
- 不 densify:点云不均,sky 等低频区冗余 Gaussian 拖慢渲染,高频区细节不足
- 标准 growth:恢复细节但 sky 区产生严重 floater

解法:
1. 初始点云分 $\tilde{P}_{\text{sky}}$ 和 $\tilde{P}_{\text{scene}}$
2. 标准 growth 只应用于 $\tilde{P}_{\text{scene}}$,严格防 sky 出 floater
3. 集成 [MaskGaussian](https://arxiv.org/abs/2412.13678):每个 Gaussian 存在性建模为概率实体,binary mask $M_k \in \{0, 1\}$ 通过 [Gumbel-Softmax](https://arxiv.org/abs/1611.01144) 从 learnable logits 采样

修改的 rendering:

$$c(x) = \sum_{k=1}^N M_k c_k \sigma_k T_k, \quad T_{k+1} = T_k(1 - M_k \sigma_k)$$

- $M_k = 0$ 时颜色贡献可忽略且不消耗 transmittance
- Gumbel-Softmax relaxation 允许 backward gradient 传递,动态评估重要性

Sparsity regularization:

$$\mathcal{L}_{\text{mask}} = \lambda_m \left(\frac{1}{N} \sum_{k=1}^N M_k\right)^2$$

Activation 持续接近零的 Gaussian 永久 prune。

**Losses**:

$$\mathcal{L}_{\text{color}} = (1 - \lambda_{c1}) \mathcal{L}_1 + \lambda_{c1} \text{SSIM} + \lambda_{c2} \text{LPIPS}$$

$$\mathcal{L}_{\text{geo}} = \lambda_d \mathcal{L}_1(\hat{D}_i, D_i^a) + \lambda_n (1 - \cos(\hat{N}_i, N_i))$$

- Depth supervision 稀疏应用于部分对齐 depth map
- Normal supervision 由 [MoGe2](https://arxiv.org/abs/2507.02546) 估,alignment-free,应用于所有帧

$$\mathcal{L}_{\text{GS}} = \mathcal{L}_{\text{color}} + \mathcal{L}_{\text{geo}} + \mathcal{L}_{\text{reg}} + \mathcal{L}_{\text{mask}}$$

**Mesh extraction**:渲染所有 training view 的 RGB+depth 集成 TSDF volume,[marching cubes](https://en.wikipedia.org/wiki/Marching_cubes) 提取 mesh,移除小 disconnected component + simplification 抑 floater。

### Table 9 3DGS 消融

| Config | GS Number | PSNR↑ |
|---|---|---|
| Baseline (6M Gaussians) | 6.000M | 25.176 |
| + Voxel downsample | 1.000M | 24.504 (掉 0.68dB) |
| + Adaptive densify | 5.254M | 25.158 (恢复但膨胀) |
| + MaskGaussian | 1.383M | 25.017 (减 73.7% 只损 0.14dB) |
| + Non-sky densify | 1.381M | 25.023 |

完整配置 Gaussian 数从 6M 降到 1.381M,减 77%,PSNR 只损 0.15dB。

---

## 总效果

### 跟闭源 [Marble](https://marble.worldlabs.ai/) 比

Figure 23 同 panorama input:HY-World 2.0 严格 adheres 输入,Marble 偏离导致 fidelity 下降。Figure 24 同 perspective input:HY-World 2.0 更好保持输入 view,3DGS completeness 更优。在 fence、car、furniture、mountain、arcade machine 等 large viewpoint change 下,Marble 严重 blur 和 geometric missing,HY-World 2.0 保持结构完整和纹理平滑。

### Runtime

Table 10:完整 3D world 生成 ~10 分钟 (NVIDIA H20):
- Panorama 15s
- Trajectory plan 182s
- World expansion 286s
- Recon+Align 102s
- 3DGS 127s
- Total 712s

优化技术:Sequence Parallelism 跨所有 stage、[SageAttention2](https://arxiv.org/abs/2410.02367)、FP8 mixed-precision、step caching。

### WorldLens 渲染平台

虽然 paper 细节少,但特性包括:engine-agnostic 架构、automatic IBL lighting、efficient collision detection、training-rendering co-design、支持 character 的交互式探索。提取的 mesh 作为 collision proxy,支持实时物理反馈,为 game、VR、embodied AI 提供下游基础。

---

## 这工作核心 insight 提炼

### 1. Generation 和 Reconstruction 本就该统一

Reconstruction 是 generation 的必要组件,不是独立模块。WorldMirror 2.0 既独立服务 dense input,又嵌入 generation pipeline Stage IV 当 backbone。生成场景获得 reconstruction 级几何精度,reconstruction 模型从 generation 数据受益。

### 2. Keyframe Latent Space 是对的抽象

对 3D 重建,viewpoint coverage 比 frame continuity 重要。冗余中间帧只是消耗 token budget。Keyframe-VAE 只做 spatial 压缩不做 temporal 压缩,保持高频细节,这与 [FlashWorld](https://arxiv.org/abs/2510.13678) 思路一致。Table 7 的 freeze cross-attn+FFN 配置达到性能与泛化最佳平衡——full training overfit 导致 style drift 反而损 camera precision。

### 3. Spatial Stitching > Temporal Concatenation

Table 8 的 A* vs A 是最有说服力的对比。Retrieved reference 通过 temporal concat 集成所有指标严重退化。Spatial stitching 让 attention 在 target-retrieval pair 内自然建立 correspondence,跟传统 stereo matching 的 insight 一致。这是把 2D stereo 的几何 prior 编码进 attention pattern 的巧妙做法。

### 4. Position Encoding 归一化的艺术

Normalized RoPE 把 extrapolation 转 interpolation。8-patch 训练 $[0,7]$,16-patch inference $[8,15]$ 全 OOD;normalized 后两者都在 $[-1,1]$,inference 只是更密采样。跟 [DINOv3](https://arxiv.org/abs/2508.10104) 思路一致。神经网络在 fixed-range 上泛化更好,这是普适 insight。

### 5. Depth-Normal 耦合监督

不依赖完整 depth GT,通过 normal pseudo-label 间接监督 depth。Surface normal 描述局部 orientation 不需要全局 metric consistency,在 multi-view 下天然更鲁棒。这是把"什么 supervision 容易获得且 reliable"和"什么 task 需要监督"做精准匹配的好例子。

### 6. MaskGaussian 概率建模

Gaussian 存在性建模为概率实体而非 hard pruning,Gumbel-Softmax relaxation 让 backward gradient 通过,动态评估重要性。Sparsity reg 鼓励低概率 Gaussian 永久 prune。这避免了传统 hard pruning 的信息丢失,让模型自己学会哪些 Gaussian 该留。

### 7. Token Budget 作为 First-Class 约束

WorldMirror 2.0 的 token-budget-first 训练策略:固定 $T_{\max}$,让 resolution 和 view count 自适应。本质是把硬件约束作为 first-class 设计目标,而非事后 patch。这种思路值得在 large model 训练中推广。

### 8. Sky 与 Scene 分离处理

3DGS 训练中分离 sky 和 scene subset,growth 只应用于 scene,严格防 sky 出 floater。Sky 区域没 depth supervision 且高频信息有限,任何 densification 都纯冗余。这是 domain-specific 的深刻理解。

### 9. Implicit vs Explicit Prior 的精准权衡

HY-Pano 2.0 放弃 explicit camera intrinsic 转 implicit attention learning,WorldStereo 2.0 在 camera control 保留 explicit Plücker rays + point cloud。这种不一致反映任务特性:panorama generation 中 metadata 不可靠 implicit 更鲁棒;view synthesis 中 camera precision 是硬约束 explicit 不可省。

### 10. Distillation 不损反增益

DMD 蒸馏后 T&T AUC 从 58.19 升到 60.09。Full-step diffusion 在某些 metric 上有冗余甚至噪声,few-step 起到正则化效果。这跟 distillation 在其他领域的观察一致——多步优化容易过拟合 training distribution 的 artifact。

---

## 这工作在开源生态里的位置

开源 world model 这个赛道近期很热:[Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)、[Yume-1.5](https://arxiv.org/abs/2512.22096)、[Robbyant](https://arxiv.org/abs/2601.20540) 都在推进。HY-World 2.0 的差异化在于:
1. 统一 generation 和 reconstruction,一个 framework 两个 task
2. 四阶段 pipeline 设计清晰,每阶段独立可改进
3. 完整开源 model weights + code + 技术细节
4. 与闭源 [Marble](https://marble.worldlabs.ai/) 竞争力相当

这种"system paper"——单组件可能不全是 SOTA,但系统集成和工程优化极强——是工业界研究的典型范式。10 分钟端到端生成可探索 3DGS 在开源社区是显著突破,对 embodied AI、robotics simulation、game development 这些下游应用是直接可用的基础设施。

参考链接汇总:
- HY-World 2.0 Project: https://3d-models.hunyuan.tencent.com/world/
- GitHub: https://github.com/Tencent-Hunyuan/HY-World-2.0
- Scene to 3D Demo: https://3d.hunyuan.tencent.com/sceneTo3D
- WorldMirror: https://arxiv.org/abs/2510.10726
- Marble: https://marble.worldlabs.ai/
- FlashWorld: https://arxiv.org/abs/2510.13678
- DiT360: https://arxiv.org/abs/2510.11712
- Matrix3D: https://arxiv.org/abs/2508.08086
- GenEx: https://arxiv.org/abs/2412.09624
- VGGT: https://arxiv.org/abs/2503.21851
- π3: https://arxiv.org/abs/2507.13347
- MapAnything: https://arxiv.org/abs/2509.13414
- MoGe2: https://arxiv.org/abs/2507.02546
- SAM3: https://arxiv.org/abs/2511.16719
- Qwen3-VL: https://arxiv.org/abs/2505.09388
- Q-Align: https://arxiv.org/abs/2312.17090
- MaskGaussian: https://arxiv.org/abs/2412.13678
- DMD: https://arxiv.org/abs/2405.14867
- DINOv3: https://arxiv.org/abs/2508.10104
- RoPE: https://arxiv.org/abs/2104.09864
- Gumbel-Softmax: https://arxiv.org/abs/1611.01144
- SageAttention2: https://arxiv.org/abs/2410.02367
- 3DGS: https://repo.samuelgarcia.delarava.eu/inria_graffiti/2023_sigg_asia_3D_Gaussian_Splatting.pdf
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- Wan: https://arxiv.org/abs/2503.20314
- CogVideoX: https://arxiv.org/abs/2408.06072
- Recast Navigation: https://github.com/recastnavigation/recastnavigation
- Tanks-and-Temples: https://www.tanksandtemples.org/
- MipNeRF360: https://arxiv.org/abs/2111.05505
- Gen3C: https://arxiv.org/abs/2412.16960
- Lyra: https://arxiv.org/abs/2509.19296
- SEVA: https://arxiv.org/abs/2503.xxxxx
- DepthAnything3: https://arxiv.org/abs/2511.10647
- Fast3R: https://arxiv.org/abs/2501.13928
- CUT3R: https://arxiv.org/abs/2503.17910
- FLARE: https://arxiv.org/abs/2503.xxxxx
- LeftRefill: https://arxiv.org/abs/2403.15812
- Neoverse: https://arxiv.org/abs/2601.00393
- WorldPlay: https://arxiv.org/abs/2512.14614
- WorldCompass: https://arxiv.org/abs/2602.09022
- video2world: https://arxiv.org/abs/2603.16736

---

# HY-World 2.0: 多模态世界模型深度技术解析

## 1. 整体定位与设计哲学

HY-World 2.0 由 Tencent Hunyuan 团队提出,核心目标在于**统一 3D 世界生成（world generation）与重建（world reconstruction）**这两个长期割裂的范式。传统上,generative 方法（如 [Matrix3D](https://arxiv.org/abs/2508.08086)、[GenEx](https://arxiv.org/abs/2412.09624)）从 sparse input 合成可探索场景但缺乏几何精度;reconstruction 方法（如 [VGGT](https://arxiv.org/abs/2503.21851)、[π3](https://arxiv.org/abs/2507.13347)、[MapAnything](https://arxiv.org/abs/2509.13414)）从 dense multi-view 恢复精确 3D 结构但缺少生成先验来 hallucinate 不可见区域。HY-World 2.0 通过一个**四阶段 pipeline** 把这两条路径融合,输出 navigable 3D Gaussian Splatting (3DGS) 场景。

整体 pipeline 形式化上可以理解为一个条件映射:

$$f_{\text{HY-World}}: \mathcal{M}_{\text{in}} \rightarrow \mathcal{G}_{\text{3DGS}}$$

其中 $\mathcal{M}_{\text{in}} \in \{\text{text}, I_{\text{single}}, \{I_i\}_{\text{multi-view}}, V_{\text{video}}\}$。当 input 稀疏时执行 generation branch,当 input 为 multi-view/video 时执行 reconstruction branch,而 reconstruction module（WorldMirror 2.0）同时嵌入在 generation pipeline 的 Stage IV 中作为几何 backbone。

参考链接:
- Project page: https://3d-models.hunyuan.tencent.com/world/
- GitHub: https://github.com/Tencent-Hunyuan/HY-World-2.0
- 前作 HY-World 1.0: https://arxiv.org/abs/2503.20191 (相关 hunyuanworld 系列)

---

## 2. Stage I — Panorama Generation: HY-Pano 2.0

### 2.1 任务与动机

Panorama 提供完整的 $360° \times 180°$ FoV,作为后续 trajectory planning 与世界扩展的"种子"。HY-World 1.0 依赖 explicit camera intrinsic 来做 perspective→ERP (Equirectangular Projection) 的几何 warping,但实际场景中 metadata 常缺失或不准,导致投影畸变。

### 2.2 模型架构:Geometry-Free 的 MMDiT

HY-Pano 2.0 放弃显式相机先验,采用 **Multi-Modal Diffusion Transformer (MMDiT)**,把 conditional image latent 与 panoramic noise latent 直接在 token 序列上 concat,让 self-attention 自动学习 perspective-to-ERP 的隐式空间映射。这是一个**纯数据驱动**的策略:

$$\text{Tokens} = [\text{Concat}(\text{Enc}(I^{\text{cond}}), \text{Enc}(z_T^{\text{pan}}))]$$

其中 $z_T^{\text{pan}} \sim \mathcal{N}(0, I)$ 是 panoramic 噪声 latent,$\text{Enc}(\cdot)$ 为 VAE encoder。MMDiT 的 self-attention 在统一 latent 空间内建立 spatial correspondence,这本质上是把 geometry 问题转化为 attention pattern 学习问题,与 [DiT360](https://arxiv.org/abs/2510.11712) 思路相似但去除了 camera metadata 依赖。

### 2.3 Circular Padding + Pixel Blending 处理 ERP 边界不连续

ERP 的左右边界在 $360°$ wrap-around 处通常有 seam。HY-Pano 2.0 用两级融合策略:

1. **Latent level**: 对 latent feature 应用 circular padding,在去噪过程中强制周期边界条件。
2. **Pixel level**: 解码到 pixel 空间后,沿 equirectangular edge 做线性 pixel blending。

公式上可以写为:

$$I^{\text{pan}}_{\text{final}}(x, y) = (1 - w(x)) \cdot I^{\text{decode}}(x, y) + w(x) \cdot I^{\text{decode}}(x \bmod W, y)$$

其中 $w(x)$ 是线性权重,在 edge 区域从 0 平滑过渡到 1。

### 2.4 数据策略

混合 real-world panorama 与 Unreal Engine (UE) synthetic 渲染:
- Real-world: 注入真实光照、纹理、结构先验
- Synthetic: 提供精确几何 label 和多样场景配置
- 严格过滤 stitching artifacts 与 exposed camera equipment

### 2.5 实验结果（Table 4）

| Metric (T2P) | DiT360 | Matrix3D | HY-World 1.0 | **HY-Pano 2.0** |
|---|---|---|---|---|
| CLIP-T ↑ | 0.248 | 0.238 | 0.250 | **0.258** |
| Q-Align Qual (Persp) ↑ | 3.788 | 2.983 | 3.992 | **4.103** |
| Q-Align Aes (Equi) ↑ | 4.072 | 3.880 | 4.186 | **4.247** |

| Metric (I2P) | CubeDiff | GenEx | HY-World 1.0 | **HY-Pano 2.0** |
|---|---|---|---|---|
| CLIP-I ↑ | 0.828 | 0.831 | 0.831 | **0.844** |
| Q-Align Qual (Persp) ↑ | 2.938 | 2.917 | 3.317 | **4.026** |

HY-Pano 2.0 在 5/5 的 I2P 指标上排名第一,T2P 上多数指标领先。Q-Align 评分采用 [Q-Align](https://arxiv.org/abs/2312.17090)（基于 large multimodal model 与人类评分对齐）。

---

## 3. Stage II — Trajectory Planning: WorldNav

### 3.1 Scene Parsing

在 panorama 基础上,WorldNav 需要提取:
- **Panoramic point cloud** $P^{\text{pan}}$: 用 [MoGe2](https://arxiv.org/abs/2507.02546) 估计单目深度,通过 Least-Squares Minimal Residual (LSMR) 在 perspective 子视图间对齐。从 HY-World 1.0 的 12 views 增加到 42 views 以提升几何质量,GPU 加速 LSMR solver。
- **Semantic landmarks**: [Qwen3-VL](https://arxiv.org/abs/2505.09388) 识别关键空间地标与障碍;[SAM3](https://arxiv.org/abs/2511.16719) 生成 2D 语义 mask,再 localize 到 3D。
- **NavMesh**: [Recast Navigation](https://github.com/recastnavigation/recastnavigation) 生成可通行区域,通过 ray-casting 修正地面、KD-Tree 加速 boundary erosion、合成 bridge polygons 连接孤立区域。

### 3.2 五种 Heuristic 轨迹模式

WorldNav 设计 5 种 trajectory mode,从 panorama 中心出发覆盖 diverse viewpoints 同时保持 collision-free:

1. **Regular Trajectories**: 将 panorama 均分为 3 个 120° FoV-x perspective view,对每个 view 在 median depth 处定义 orbital target,先做 +45° pitch 旋转再做 ±120° azimuth offset,额外 +60° azimuth 提供 aerial 视角。Ray-casting 防止相机 clipping。

2. **Surrounding Trajectories**: 围绕最重要物体环绕。Orbit radius 根据物体 3D size 自适应,沿理想圆周采样 72 个候选节点,经 ray-casting 验证后用 bidirectional greedy search 连成 arc。Tail pruning 删除偏离方向的尾端,Dijkstra 算法连接起点到弧的最近端点。

3. **Reconstruct-Aware Trajectories**: 针对 under-observed 区域(panoramic mesh 中表现为 stretched sharp faces)。检测 aspect ratio 超过阈值的 mesh face,用 NMS (Non-Maximum Suppression) 提取代表性 cluster center,与最近 semantic landmark 关联。在节点周围生成 candidate viewpoint,选择 viewing angle 对齐缺失区域的端点。Iterative orbiting trajectory 保持 fixed gaze。

4. **Wandering Trajectories**: 模拟自主 agent 探索,目标为 NavMesh 内最远可达点。将 NavMesh 按 8 个均匀 angular sector 分割,在每个 sector 内用 Dijkstra distance field 找最远 node。适合 narrow 环境(streets、corridors)。

5. **Aerial Trajectories**: 在 surrounding 和 wandering 基础上叠加 +45° upward pitch。Pitch 角度动态降低以防止与 panoramic mesh 碰撞。

| Trajectory Type | Max Number | Attached to Object | Iterative |
|---|---|---|---|
| Regular | 9 | ✗ | ✗ |
| Surrounding | 5 | ✓ | ✗ |
| Recon-Aware | 10 | ✓ | ✓ |
| Wandering | 3 | ✗ | ✗ |
| Aerial | 8 | – | – |
| **Total** | **35** | – | – |

### 3.3 实验消融（Figure 19）

只用 panoramic views 训练 3DGS → 大量几何空洞;逐步加入 Regular → Surround+Recon → Wandering → Aerial,scene completeness 逐步提升。Aerial 额外提供 BEV (Bird's-Eye View) 观测,提升 3D world viewpoint 切换自由度。

---

## 4. Stage III — World Expansion: WorldStereo 2.0

这是 HY-World 2.0 最核心的技术创新之一。WorldStereo 2.0 是 WorldStereo 1.0（[Tencent WorldStereo](https://arxiv.org/abs/2603.xxxxx)）的升级,核心思想是在 **keyframe latent space** 而非 video latent space 上做 camera-guided 生成,辅以 memory mechanism 保证多轨迹一致性。

### 4.1 Keyframe-VAE:从 spatio-temporal 压缩到 spatial-only 压缩

#### 动机

标准 video diffusion（如 [HunyuanVideo](https://arxiv.org/abs/2412.03603)、[Wan](https://arxiv.org/abs/2503.20314)、[CogVideoX](https://arxiv.org/abs/2408.06072)）使用 Video-VAE 同时做空间和时空压缩。这种 spatio-temporal compression 在相机快速运动时会产生严重质量退化——motion blur、几何畸变——直接破坏下游 3D 重建。Figure 8 的对比非常直观。

#### 方法

借鉴 [FlashWorld](https://arxiv.org/abs/2510.13678),WorldStereo 2.0 提出 **Keyframe-VAE**:对每个 keyframe $V_i \in \mathbb{R}^{1 \times H \times W \times 3}$ 独立应用 causal-padding image encoder,得到 latent:

$$\{F_i\}_{i=1}^{1+T_{kf}} \in \mathbb{R}^{1 \times \frac{H}{8} \times \frac{W}{8} \times C}$$

其中:
- $H, W$: frame 高宽
- $C$: latent channel
- $T_{kf}$: keyframe 数量,满足 $T_{kf} \ll T_{vid}$（远小于标准视频帧数）
- $\frac{H}{8} \times \frac{W}{8}$: 8× 空间压缩,无时间维度压缩

关键 insight: 同一 token length 下,keyframe latent 包含更少帧,但通过增大 keyframe sampling interval 维持相同 viewpoint coverage。由于 Keyframe-VAE 保留 image-level 高频细节,viewpoint 大幅变化时 fidelity 显著优于 Video-VAE。同时独立处理每个 keyframe,VAE 编码解码天然 parallelizable。

#### Table 7 消融验证

| Frozen Parts | VAE Type | RotErr↓ | TransErr↓ | ATE↓ | User Camera↑ | User Quality↑ |
|---|---|---|---|---|---|---|
| Main DiT (baseline) | Video-VAE | 0.762 | 1.245 | 2.141 | 84.85% | 46.46% |
| Main DiT | Keyframe-VAE | 0.768 | 1.149 | 2.027 | – | – |
| None (full train) | Keyframe-VAE | 0.578 | 1.115 | 2.245 | 93.81% | 60.61% |
| Cross-Attn | Keyframe-VAE | 0.684 | 1.243 | 2.111 | 93.13% | 60.95% |
| **Cross-Attn + FFN** | **Keyframe-VAE** | **0.492** | **0.968** | **1.768** | 92.44% | **64.39%** |

关键观察:full training 虽然最大化 visual metric 但导致 camera precision 下降（overfitting 引起 image style drift）。Freeze cross-attention + FFN 在 camera precision 与 visual quality 之间取得最佳平衡——RotErr 0.492 最低,User Quality 64.39% 最高。

### 4.2 Explicit Camera Control:Plücker Rays + Point Clouds

WorldStereo 2.0 在 domain-adaption 阶段引入双相机引导:

1. **Plücker rays** $L = (o, d)$:每条 ray 由 origin $o \in \mathbb{R}^3$ 和 direction $d \in \mathbb{R}^3$ 表示,6 维,编码像素级 camera ray。
2. **Point clouds**:从 reference view $I^{\text{ref}}$ 提取的 $P^{\text{ref}} \in \mathbb{R}^{N \times 3}$,warp 到每个 target view:

$$P_i^{\text{tar}}(x) \simeq R_i^{cw} D(x) K_i^{-1} \hat{x} \quad (1)$$

变量解释:
- $P_i^{\text{tar}}(x)$: target view $i$ 在像素 $x$ 处的 3D 点
- $R_i^{cw} \in \mathbb{R}^{3 \times 3}$: target view $i$ 的 camera-to-world 旋转矩阵
- $K_i \in \mathbb{R}^{3 \times 3}$: target view $i$ 的 camera intrinsic matrix
- $D(\cdot)$: 在 reference view 上由 [MoGe2](https://arxiv.org/abs/2507.02546) 估计的 monocular depth
- $\hat{x} \in \mathbb{R}^4$: 像素 $x$ 的 homogeneous 坐标
- $\simeq$: projective 等价(差一个 scale factor)

Warping 后的 point cloud 渲染成 view-wise keyframe,经 Keyframe-VAE 编码为 latent。相比 [Uni3C](https://arxiv.org/abs/2503.xxxxx) 只训练 control branch,WorldStereo 2.0 同时 fine-tune DiT backbone 子集（freeze cross-attention 和 feed-forward）以更好匹配 keyframe latent 空间。

### 4.3 Memory Mechanism:GGM + SSM++

#### 4.3.1 Global-Geometric Memory (GGM)

GGM 用扩展的全局点云作为 3D prior 强制多轨迹几何一致性。在 mid-training 阶段使用:

$$P^{\text{glo}} = [P^{\text{ref}}, \hat{P}] \in \mathbb{R}^{(N+\hat{N}) \times 3} \quad (2)$$

其中:
- $P^{\text{ref}} \in \mathbb{R}^{N \times 3}$: reference view 点云
- $\hat{P} \in \mathbb{R}^{\hat{N} \times 3}$: 从 $T_g$ 个 novel view 随机采样的额外点云

Inference 时直接用 panoramic point cloud $P^{\text{pan}}$ 作为 global guidance,覆盖 360° 视角。

**数据增强**(防止过拟合 imperfect point cloud):
- 50% 样本: 双线性下采样 depth 模拟"depth bleeding" artifact
- 10% 样本: Gaussian filter 制造 artificial floaters
- 50% real-world 样本: 保留原始 noise 不做过滤

**关键经验**: aggressive point cloud distortion(如 [Neoverse](https://arxiv.org/abs/2601.00393) 那种)对 GGM 反而有害,过度削弱几何 guidance 导致跨视频几何不一致。

#### 4.3.2 Improved Spatial-Stereo Memory (SSM++)

SSM++ 是 SSM 的重大升级,灵感来自传统 [stereo matching](https://www.science.org/doi/10.1126/science.194.4262.283) 和 reference-based inpainting([LeftRefill](https://arxiv.org/abs/2403.15812))。核心创新:

**(a) 集成方式变更**:放弃 WorldStereo 1.0 的独立 memory branch,直接把 retrieved keyframe 嵌入 main DiT branch。

**(b) RoPE 修改**(Figure 11):Target frame 与其 retrieved reference view 沿 horizontal axis 空间拼接(width 变为 2W),关键的是每个 retrieved view 继承其配对 target frame 的 temporal index,送入 main DiT。

公式上,对每个 target frame $V_i^t$ 和 retrieved reference $V_i^r$:

$$\text{Token}_i = \text{Concat}_{\text{spatial}}(\text{Enc}(V_i^t), \text{Enc}(V_i^r)) \in \mathbb{R}^{\frac{H}{8} \times \frac{2W}{8} \times C}$$

RoPE 修改后,retrieved view 在 temporal 维度上与 target 共享 index,在 spatial 维度上保持独立 2D 位置编码。这种设计让 attention 自然在"target-retrieval pair"内建立 correspondence。

**(c) Selective Retrieval**:不再像 WorldStereo 1.0 强制每帧都 retrieve,只 retrieve 最相关的 $T_r$ 个 keyframe($T_r < T_{kf}$),大幅降低冗余计算和 memory overhead。

**(d) Full Attention 替代 Restricted Attention**:mid-training 阶段移除 attention receptive field 限制(除 cross-attention),让模型通过 full self-attention 学习所有 target 和 retrieved feature 的全局 context。

**(e) Implicit Camera Embedding 替代 Explicit Pointmap Guidance**:用 7 维相机 pose 向量(quaternion 4D + translation 3D)替代 pointmap,经 3-layer MLP 编码为 camera token,通过 zero-initialization 加到 target 和 retrieved keyframe feature。

**Memory Bank 与 Retrieval 策略**:
- 初始 memory bank: 输入 panorama 切分的 perspective views
- 增量更新: 生成的 keyframe 逐步加入 memory bank
- 存储: RGB image + camera parameter
- Retrieval: 基于 3D FoV similarity,上限 $T_r$ keyframe

**训练数据构造**(Figure 10):
- 现有 multi-view 数据: temporally misaligned retrieval,从 retrieval trajectory 随机选 30%-90% temporal overlap 的 frame,部分 frame 在 target trajectory 之外,增加训练难度提升 robustness
- Synthetic UE 数据: multi-trajectory retrieval,基于 3D FoV similarity 从 alternative trajectory 选最相关 frame

#### 4.3.3 SSM++ 数据增强

- 对 retrieved frame: random motion blur + color jitter
- Random crop target 和 retrieved image 模拟不同 visibility range 和 FoV overlap

### 4.4 Post-Train: Distribution Matching Distillation (DMD)

加速推理。基于 [DMD](https://arxiv.org/abs/2405.14867)(Variational Score Distillation 的扩展),通过近似 KL divergence 蒸馏 few-step student:

$$\nabla \mathcal{L}_{\text{DMD}} = -\mathbb{E}_t \left( \int \left( s_{\text{real}}(x_t, t) - s_{\text{fake}}(x_t, t) \right) \frac{dx_t}{d\theta} dz \right) \quad (3)$$

变量解释:
- $x = G_\theta(z)$: student generator $G_\theta$ 给定 random Gaussian noise $z \sim \mathcal{N}(0, I)$ 的生成
- $t \sim \mathcal{U}(0, 1)$: 时间步 uniform 采样
- $x_t \sim q_t(x_t | x, t)$: forward diffusion 过程
- $s_{\text{real}}(x_t, t)$: frozen real score function
- $s_{\text{fake}}(x_t, t)$: trainable fake score function
- $\frac{dx_t}{d\theta}$: 通过 generator 的梯度回传

**训练细节**:
- $G_\theta, s_{\text{real}}, s_{\text{fake}}$ 都从 mid-training 后的 VDM 初始化
- $s_{\text{real}}$ frozen,$G_\theta$ 和 $s_{\text{fake}}$ fully trainable
- $s_{\text{fake}}$ 每次更新 $G_\theta$ 训练 5 次
- Stochastic gradient truncation 稳定训练
- 省略 GAN loss(影响不显著且大幅减速)
- 蒸馏为 4-step DiT

**关键进步**:相比 WorldStereo 1.0 只在 camera control 任务蒸馏并 freeze memory branch(因 annotated memory data 短缺),WorldStereo 2.0 借助灵活的 explicit-guidance-free SSM++ 和丰富 UE 数据,实现 memory-based training 的 full fine-tuning 蒸馏,同时增强 camera control 和 memory capability。

### 4.5 实验结果

#### Table 5: Single-View Scene Reconstruction

在 [Tanks-and-Temples](https://www.tanksandtemples.org/) 和 [MipNeRF360](https://arxiv.org/abs/2111.05505) 上,WorldStereo 2.0 在 F1-Score 和 AUC 上都超越所有 video-based 和 3D-based 竞争者:

| Methods | T&T F1↑ | T&T AUC↑ | MipNeRF360 F1↑ | MipNeRF360 AUC↑ |
|---|---|---|---|---|
| SEVA | 36.73 | 51.03 | 28.75 | 46.81 |
| Gen3C | 31.24 | 42.44 | 35.26 | 52.10 |
| Lyra | 32.54 | 43.05 | 36.05 | 49.89 |
| FlashWorld | 22.29 | 30.45 | 42.60 | 53.86 |
| **WorldStereo 2.0** | 41.43 | 58.19 | 51.27 | **65.79** |
| WorldStereo 2.0 (DMD) | 43.16 | **60.09** | 50.52 | 65.64 |

DMD 蒸馏版本甚至在 T&T AUC 上略优于非蒸馏版本(58.19→60.09),说明蒸馏过程起到了正则化效果。

#### Table 6: Camera Control Capability

| Method | RotErr↓ | TransErr↓ | ATE↓ | Q-Align↑ |
|---|---|---|---|---|
| SEVA | 1.690 | 1.578 | 2.879 | 3.232 |
| Gen3C | 0.944 | 1.580 | 2.789 | 3.353 |
| WorldPlay | 3.481 | 1.288 | 2.722 | 3.628 |
| WorldStereo 1.0* | 0.762 | 1.245 | 2.141 | 4.149 |
| **WorldStereo 2.0*** | **0.492** | **0.968** | **1.768** | **4.205** |

WorldStereo 2.0 在所有 camera metric 上达到最低 error,Q-Align 视觉质量最高。

#### Table 8: Memory and Distillation 消融

| Config | PSNR↑ | SSIM↑ | LPIPS↓ | PSNR_m↑ | RotErr↓ | TransErr↓ |
|---|---|---|---|---|---|---|
| Baseline (camera only) | 16.13 | 0.474 | 0.349 | 28.81 | 0.396 | 0.053 |
| A: GGM + SSM++ | 20.94 | 0.640 | 0.170 | 30.27 | 0.407 | 0.047 |
| B: + Trainable FFN | 21.56 | 0.667 | 0.162 | 30.44 | 0.351 | 0.036 |
| C: + Pointcloud aug | 21.36 | 0.632 | 0.163 | 30.72 | 0.360 | 0.050 |
| D: + Reference aug | 20.86 | 0.639 | 0.165 | 30.66 | 0.322 | 0.049 |
| E: + Camera embedding | 21.06 | 0.639 | 0.164 | 30.58 | 0.329 | 0.042 |
| A*: Temporal-concat SSM | 19.83 | 0.581 | 0.219 | 29.77 | 0.545 | 0.087 |
| F: + Doubled batch | 21.63 | 0.669 | 0.156 | 30.76 | 0.296 | 0.036 |
| **G: + DMD distillation** | **21.84** | **0.669** | 0.165 | **30.93** | 0.316 | 0.052 |

关键发现:
- GGM + SSM++ (A) 带来 PSNR 从 16.13 → 20.94 的巨大提升
- Trainable FFN (B) 显著降低 camera error
- 用 temporal concatenation 替代 spatial stitching (A*) 严重退化所有指标,验证了 spatial stitching 设计的核心价值
- Doubled batch (F) 稳定训练
- DMD distillation (G) 进一步提升 PSNR 和 consistency

---

## 5. Stage IV — World Reconstruction: WorldMirror 2.0

WorldMirror 2.0 是 HY-World 2.0 的几何 backbone,既是 standalone reconstruction foundation model,也嵌入 generation pipeline 的 Stage IV。它基于 [WorldMirror 1.0](https://arxiv.org/abs/2510.10726),针对三大限制改进:
1. 非 training resolution 退化
2. Depth-normal 缺乏 explicit coupling
3. 大量 view 时 memory/latency 爆炸

### 5.1 Any-Modal Tokenization 回顾

WorldMirror 1.0 核心设计:把所有 input modalities(image、camera pose、intrinsic、depth)token 化为统一 sequence,Transformer backbone + DPT decoder heads([VGGT](https://arxiv.org/abs/2503.21851))一次性输出 point map、depth、normal、camera、3DGS attribute。训练时每个 prior modality 以 0.5 概率独立 drop,实现 inference 时的灵活 prior injection。

两阶段 curriculum:
- Phase 1: geometry heads (point map、depth、camera、normal) 联合训练
- Phase 2: 冻结 geometry 参数,只训练 3D Gaussian head

### 5.2 模型架构改进

#### 5.2.1 Normalized Position Encoding

**问题**: WorldMirror 1.0 用标准 [RoPE](https://arxiv.org/abs/2104.09864),每个 patch 用 absolute integer grid index $(i, j) \in \{0, ..., H_p - 1\} \times \{0, ..., W_p - 1\}$。这导致:测试分辨率高于训练时,patch index 超出训练范围(position extrapolation);测试分辨率低于训练时,index space 未充分利用(distribution shift)。

**方法**: 借鉴 [DINOv3](https://arxiv.org/abs/2508.10104),将 absolute integer 坐标替换为 normalized 坐标,所有 patch position 映射到固定 $[-1, 1]$ 范围:

$$\hat{x}_i = \frac{2i + 1}{H_p} - 1, \quad \hat{y}_j = \frac{2j + 1}{W_p} - 1 \quad (4)$$

变量解释:
- $\hat{x}_i, \hat{y}_j \in [-1, 1]$: normalized 坐标
- $i \in \{0, ..., H_p - 1\}$, $j \in \{0, ..., W_p - 1\}$: patch grid index
- $H_p = H/p$, $W_p = W/p$: patch grid 尺寸,patch size 为 $p$
- $+1$ offset 在分子:确保 pixel-center alignment,防止边界 patch 塌缩到 $\pm 1$

高度和宽度独立归一化,保留 aspect ratio 信息,泛化到非 square input。

**核心 insight**:将 resolution extrapolation 转化为 interpolation。标准 RoPE 下 8-patch 训练 grid 占 integer index $[0, 7]$,16-patch inference 时 $[8, 15]$ 完全 out-of-distribution;normalized RoPE 把两者都映射到 $[-1, 1]$,inference 坐标只是同一范围的更密采样。

Figure 13 验证:
- Normalized RoPE 跨分辨率 cosine similarity > 0.95
- 标准 RoPE 显著退化
- Normalized RoPE 编码值 mean/std 稳定,标准 RoPE 出现 systematic mean drift

#### 5.2.2 Explicit Normal Supervision for Depth

**问题**: WorldMirror 1.0 中 depth 和 normal head 独立监督,无显式几何耦合。Real-world multi-view dataset 含噪声或不完整 depth annotation,monocular depth pseudo label 存在 multi-view 不一致。

**方法**: 引入 depth-to-normal loss $\mathcal{L}_{d2n}$,将预测 depth 通过 back-projection 和 cross product 转换为 surface normal,与 normal target 监督:

$$\tilde{N}_i(x) = \text{normalize}\left(\frac{\partial P_i}{\partial u} \times \frac{\partial P_i}{\partial v}\right), \quad P_i = K^{-1} \hat{D}_i \cdot [u, v, 1]^\top \quad (5)$$

变量解释:
- $\tilde{N}_i(x)$: 在像素 $x$ 处由预测 depth 推导的 normal
- $P_i$: back-projected 3D 点
- $K \in \mathbb{R}^{3 \times 3}$: camera intrinsic matrix
- $\hat{D}_i$: 预测 depth map
- $[u, v, 1]^\top$: 像素 $(u, v)$ 的 homogeneous 坐标
- $\frac{\partial P_i}{\partial u}, \frac{\partial P_i}{\partial v}$: 偏导,用四方向 finite difference 近似
- $\times$: cross product
- $\text{normalize}(\cdot)$: L2 归一化

Loss 定义为推导 normal 与目标 normal 的 angular error:

$$\mathcal{L}_{d2n} = \frac{1}{|\mathcal{V}|} \sum_{x \in \mathcal{V}} \arccos\left(\frac{\tilde{N}_i(x) \cdot \hat{N}_i(x)}{\|\tilde{N}_i(x)\| \|\hat{N}_i(x)\|}\right) \quad (6)$$

变量解释:
- $\mathcal{V}$: valid pixel 集合
- $\hat{N}_i$: normal supervision target
- $\arccos(\cdot)$: 反余弦,得到 angular error

**Normal target 选择**:
- Synthetic dataset: $\hat{N}_i$ 由 ground-truth depth 应用相同 depth-to-normal 变换得到,提供 clean、multi-view consistent supervision
- Real-world dataset: $\hat{N}_i$ 由 monocular normal estimation teacher model 预测的 pseudo normal,提供 dense 可靠 surface orientation supervision

Surface normal 描述局部 orientation 不需要全局 metric consistency,因此比 depth pseudo label 在 multi-view 设置下天然更鲁棒。

#### 5.2.3 Depth Mask Prediction Head

**问题**: WorldMirror 1.0 用 learned confidence weight 调制 training loss 但不在 inference 时输出 explicit per-pixel validity prediction,下游应用只能用启发式 threshold。

**方法**: 新增 dedicated depth mask prediction head,输出 per-pixel validity logit $\hat{m}(x)$,用 binary cross-entropy 训练:

$$\mathcal{L}_{\text{mask}} = -\frac{1}{|\mathcal{M}|} \sum_{x \in \mathcal{M}} \left[ m^*(x) \log \sigma(\hat{m}(x)) + (1 - m^*(x)) \log(1 - \sigma(\hat{m}(x))) \right] \quad (7)$$

变量解释:
- $m^*(x) \in \{0, 1\}$: ground-truth validity label
- $\hat{m}(x)$: 预测的 validity logit
- $\sigma(\cdot)$: sigmoid 函数
- $\mathcal{M}$: 已知 validity 的 pixel 集合

**Ground-truth 来源**:
- Synthetic dataset: rendering pipeline 中精确已知的 invalid region
- Real-world dataset: 通过 extreme depth value、depth discontinuity、sky region 识别 pseudo label

Inference 时输出 mask 让下游应用可选择性 filter invalid pixel,提升 point cloud fusion 和 3D 重建鲁棒性。

### 5.3 数据改进

1. 加入高质量 UE synthetic 渲染:pixel-accurate ground-truth geometry,多样室内外环境
2. Real-world dataset 的 normal-only pseudo-label 增强:用 monocular normal estimation teacher model 预测 dense normal 作为 pseudo supervision,既直接监督 normal head,又通过 $\mathcal{L}_{d2n}$ 间接监督 depth head

### 5.4 Inference Efficiency 改进

1. **Token-level Sequence Parallelism (SP)**: Transformer backbone input token sequence 跨 GPU 分区,attention layer 通过 All-to-All collectives 重分配
2. **Frame-level SP**: DPT decoder head 的 conv layer 独立操作 per-view feature map,按帧分区
3. **Mixed-precision**: 大部分参数 cast 到 BF16,小部分 precision-critical 模块保留 FP32,内存减半精度损失可忽略
4. **FSDP (Fully Sharded Data Parallelism)**: 每个 Transformer block 和 DPT head 作为独立 FSDP unit 跨 GPU shard 参数

三种策略互补:SP 分布计算和 activation memory,mixed-precision 降低 per-element cost,FSDP shard weight memory。

### 5.5 Training Strategy 改进

#### Token-based Dynamic Batch Sizing

WorldMirror 1.0 独立采样 per-image resolution 和 view count,导致 GPU memory 必须满足 worst-case joint maximum,实际大多配置远低于此 ceiling,memory 利用率低。

WorldMirror 2.0 用 token-budget-first 策略:固定每 GPU 最大 token budget $T_{\max}$(如 25,000 tokens)。每次迭代先采样 per-image resolution(pixel count 50K-500K)和 aspect ratio,计算 per-image token count $t = \frac{H}{p} \times \frac{W}{p}$,再推导最大 view 数:

$$N_{\max} = \min\left(N_{\text{cap}}, \left\lfloor \frac{T_{\max}}{t} \right\rfloor\right) \quad (8)$$

变量解释:
- $N_{\max}$: 当前迭代最大 view 数
- $N_{\text{cap}}$: 架构 view-count 上限(如 48)
- $T_{\max}$: 每 GPU token budget
- $t$: per-image token count

实际 view count 从 $[N_{\min}, N_{\max}]$ 均匀采样。当采样 view 数小于 $N_{\max}$ 时,多个 sample 打包到同一 GPU 填满 token budget:

$$T_{\text{total}} = N \times \frac{H}{p} \times \frac{W}{p} \leq T_{\max} \quad (9)$$

$N$ 是单 GPU 上所有 image 总数(包括多个 sample)。这保证每 GPU 紧密 bounded token count,无论采样何种 resolution 都达到近乎满 GPU memory 利用,消除 OOM 错误。

#### 三阶段 Curriculum

| Stage | 内容 |
|---|---|
| 1 | 所有 geometry head 用 native annotation 训练,无 pseudo-label 增强,无 $\mathcal{L}_{d2n}$ |
| 2 | 引入 $\mathcal{L}_{d2n}$,显著增加 synthetic data 比例提升几何精度 |
| 3 | 冻结 backbone 和所有 geometry head,只用 depth head 权重初始化训练 3DGS head |

### 5.6 实验结果

#### Table 11: Point Map Reconstruction

在 7-Scenes、NRGBD、DTU 上,WorldMirror 2.0 在每个分辨率都优于 1.0:
- 7-Scenes Mean Acc.@M: 1.0 0.043 → 2.0 0.033
- 7-Scenes Mean Acc.@H: 1.0 0.079 → 2.0 0.037(差距更大)
- 加 all priors 后 7-Scenes Acc.@H: 1.0 0.042 → 2.0 0.012

#### Table 12: Camera Pose + Depth + NVS

WorldMirror 2.0 在每个分辨率都优于 1.0:
- Camera AUC@30: 1.0 H 66.29 → 2.0 H 86.89(20 分提升)
- Depth AbsRel: 1.0 H 0.195 → 2.0 H 0.162
- NVS PSNR: 1.0 H 17.78(collapse)→ 2.0 H 19.98(稳定)

#### Table 13: Surface Normal

WorldMirror 2.0 在三个 benchmark 都取得最佳:
- ScanNet mean error: 1.0 H 17.6 → 2.0 H 12.5
- NYUv2 mean error: 1.0 M 15.1 → 2.0 M 13.9
- iBims-1 mean error: 1.0 M 16.6 → 2.0 M 14.2

#### Table 14: Inference Efficiency

| Configuration | #GPUs | 128 views Mem/Time | 256 views Mem/Time |
|---|---|---|---|
| Baseline (FP32) | 1 | 59.26GB / 18.00s | OOM |
| + BF16 | 1 | 41.73GB / 16.96s | 75.05GB / 56.96s |
| + SP (×4) | 4 | 61.53GB / 6.27s | OOM |
| + SP + BF16 + FSDP (×4) | 4 | **42.71GB / 5.60s** | **78.78GB / 17.52s** |

完整配置 4 GPU 上 256 views 推理只需 17.52s,比 baseline 32 views 18.00s 还快。

---

## 6. Stage IV — World Composition

### 6.1 Point Cloud Expansion

WorldMirror 2.0 估计 per-frame depth 和 normal:

$$\{D_i^m, N_i^m\}_{i=1}^{T_{ex}'} = \Phi\left(\{V_j, C_j\}_{j=1}^{T_{pan}}, \{V_i, C_i\}_{i=1}^{T_{ex}'}\right) \quad (10)$$

变量解释:
- $\Phi(\cdot)$: WorldMirror 2.0 网络
- $\{V_j, C_j\}_{j=1}^{T_{pan}}$: 从初始 panorama 切分的 perspective view 和对应 camera parameter
- $\{V_i, C_i\}_{i=1}^{T_{ex}'}$: 生成的 keyframe 子集和对应 camera parameter
- $D_i^m, N_i^m$: 预测的 depth 和 normal map

### 6.2 Depth Alignment

WorldMirror depth $D_i^m$ 存在 scale ambiguity,未对齐到 panoramic point cloud $P^{pan}$ 的世界坐标系。通过渲染 $P^{pan}$ 得到 sparse guidance depth $D_i^g$,做 alignment:

$$D_i^a = \varphi_{\text{align}}(D_i^m, D_i^g, M_i) \quad (11)$$

Reliability mask 定义为多个 mask 的交集:

$$M_i = M_i^m \cap M_i^g \cap M_i^n \cap M_i^p \cap \overline{M_i^{sky}} \quad (12)$$

变量解释:
- $M_i^m$: WorldMirror confidence 有效投影区域(edge floater 移除)
- $M_i^g$: panoramic guidance 有效投影区域
- $M_i^n$: normal consistency mask,排除 WorldMirror normal $N_i^m$ 与推导 panoramic normal $N_i^g$ 角度偏差超 90° 的区域
- $M_i^p$: percentile-based statistical filter,排除相对 depth discrepancy 显著的 outlier
- $\overline{M_i^{sky}}$: 由 [SAM3](https://arxiv.org/abs/2511.16719) video mode 识别的非 sky mask

**RANSAC-based linear alignment**:

$$D_i^a = \gamma_i D_i^m + \beta_i$$

在 $M_i$ 定义的有效区域上估计 scale $\gamma_i$ 和 shift $\beta_i$。由于 WorldMirror 2.0 初始 depth 质量高,per-frame linear alignment 已足够,无需复杂非线性 refinement。

**Outlier Detection**:设 $Q = 9$ 个 anchor depth $\{A_q\}_{q=1}^Q$ 均匀分布在场景 depth 范围内。每帧 $i$ 计算变换后 anchor 值 $\mathcal{V}_{i,q} = \gamma_i A_q + \beta_i$。最大相对偏差:

$$\mathcal{V}_i^{\max} = \max_q \left(\left|\frac{\mathcal{V}_{i,q} - \hat{\mathcal{V}}_q}{\hat{\mathcal{V}}_q}\right|\right), \quad \hat{\mathcal{V}}_q = \text{median}_{j \in \{1, ..., T_{ex}'\}} (\mathcal{V}_{j,q}) \quad (13)$$

变量解释:
- $\hat{\mathcal{V}}_q$: anchor $q$ 在所有帧的中位变换值
- 超过 90% percentile 的 $(\gamma_i, \beta_i)$ 视为 outlier,用同视频序列内最近 inlier 替换
- 整序列为 outlier 时丢弃所有 depth map

最终扩展点云: $\tilde{P} = \text{voxel\_downsample}(P^{pan} \cup P^{ex})$

### 6.3 3D Gaussian Splatting 优化

#### Initialization

每个 3D Gaussian 参数化:
- Opacity $\sigma_k \in [0, 1]$
- Center $\mu_k \in \mathbb{R}^3$
- 3D covariance $\Sigma_k \in \mathbb{R}^{3 \times 3}$,分解为 scaling matrix $S_k$ 和 rotation matrix $R_k$:$\Sigma_k = R_k S_k S_k^\top R_k^\top$
- View-independent RGB color $c_k \in \mathbb{R}^3$(放弃 Spherical Harmonics,因生成场景无显著 view-dependent effect)

#### MaskGaussian:概率性 mask 解决 densification dilemma

**Dilemma**:
- 不做 densification:点云分布不均,sky 等低频区域冗余 Gaussian 拖慢渲染,而高频区域细节不足
- 标准 growth strategy(densification):恢复高频细节但引入严重 floater(主要来自 sky 区域)

**解决方案**:
1. 分割初始点云为 sky 和 scene subset:$\tilde{P} = \tilde{P}_{\text{sky}} \cup \tilde{P}_{\text{scene}}$
2. 标准 growth strategy 只应用于 $\tilde{P}_{\text{scene}}$,严格防止 sky 产生 floater
3. 集成 [MaskGaussian](https://arxiv.org/abs/2412.13678):每个 Gaussian 存在性建模为概率实体,二值 mask $M_k \in \{0, 1\}$ 通过 [Gumbel-Softmax](https://arxiv.org/abs/1611.01144) 从 learnable mask logits 采样

修改的 rendering 公式:

$$c(x) = \sum_{k=1}^N M_k c_k \sigma_k T_k, \quad T_{k+1} = T_k(1 - M_k \sigma_k) \quad (14)$$

变量解释:
- $c(x)$: 像素 $x$ 的渲染颜色
- $M_k$: 第 $k$ 个 Gaussian 的 binary mask
- $c_k, \sigma_k$: 第 $k$ 个 Gaussian 的颜色和 opacity
- $T_k$: 累积 transmittance(深度顺序),$T_1 = 1$
- $M_k = 0$ 时该 Gaussian 颜色贡献可忽略且不消耗 transmittance,但 Gumbel-Softmax relaxation 允许 backward gradient 传递

Sparsity regularization:

$$\mathcal{L}_{\text{mask}} = \lambda_m \left(\frac{1}{N} \sum_{k=1}^N M_k\right)^2 \quad (15)$$

训练中 activation probability 持续接近零的 Gaussian 永久 prune。

#### Optimization Losses

Photometric loss:

$$\mathcal{L}_{\text{color}} = (1 - \lambda_{c1}) \mathcal{L}_1(\hat{I}_i, I_i) + \lambda_{c1} \text{SSIM}(\hat{I}_i, I_i) + \lambda_{c2} \text{LPIPS}(\hat{I}_i, I_i) \quad (16)$$

Geometric loss:

$$\mathcal{L}_{\text{geo}} = \lambda_d \mathcal{L}_1(\hat{D}_i, D_i^a) + \lambda_n (1 - \cos(\hat{N}_i, N_i)) \quad (17)$$

变量解释:
- $\hat{I}_i, \hat{D}_i, \hat{N}_i$: 3DGS 渲染的 RGB、depth 和推导 normal
- $I_i, D_i^a, N_i$: ground truth image、aligned depth、normal
- Depth supervision 稀疏应用于部分对齐 depth map
- Normal supervision 由 [MoGe2](https://arxiv.org/abs/2507.02546) 估计,alignment-free,应用于所有帧

总 loss:

$$\mathcal{L}_{\text{GS}} = \mathcal{L}_{\text{color}} + \mathcal{L}_{\text{geo}} + \mathcal{L}_{\text{reg}} + \mathcal{L}_{\text{mask}} \quad (18)$$

#### Mesh Extraction

为支持 collision detection 和 physics simulation,从 3DGS 提取 mesh:渲染 RGB 和 depth 集成到 TSDF volume,通过 [marching cubes](https://en.wikipedia.org/wiki/Marching_cubes) 提取。移除小 disconnected component,mesh simplification 抑制 floater。

### 6.4 实验结果

#### Table 9: 3DGS 消融

| Voxel Downsample | Adaptive Densification | MaskGaussian | GS Number | PSNR↑ | LPIPS↓ |
|---|---|---|---|---|---|
| ✗ | ✗ | ✗ | 6.000M | 25.176 | 0.209 |
| ✓ | ✗ | ✗ | 1.000M | 24.504 | 0.276 |
| ✓ | ✓ | ✗ | 5.254M | 25.158 | 0.210 |
| ✓ | ✓ | ✓ | 1.383M | 25.017 | 0.216 |
| ✓ | ✓ (non-sky) | ✓ | 1.381M | 25.023 | 0.215 |

完整配置:Gaussian 数量从 6M 降到 1.381M(减少 77%),PSNR 只损失 0.15dB,LPIPS 损失 0.006。

---

## 7. 整体对比与 Runtime

### 7.1 与闭源 Marble 对比（Figure 23, 24）

与 [Marble](https://marble.worldlabs.ai/) (World Labs 闭源商业产品) 对比:
- 同 panorama input: HY-World 2.0 严格 adheres 输入,Marble 偏离导致 fidelity 下降
- 同 perspective input: HY-World 2.0 更好保持输入 view,3DGS completeness 更优
- 在 fence、car、furniture、mountain、arcade machine 等 large viewpoint change 下,Marble 出现严重 blur 和 geometric missing,HY-World 2.0 保持结构完整和纹理平滑

### 7.2 Runtime 分析（Table 10）

| Stage | Panorama | Trajectory Plan | World Expansion | Recon+Align | 3DGS | Total |
|---|---|---|---|---|---|---|
| Time (sec) | 15s | 182s | 286s | 102s | 127s | **712s** |

完整 3D world 生成只需 ~10 分钟(NVIDIA H20 GPU)。优化技术包括:
- Sequence Parallelism 跨所有 inference stage
- [SageAttention2](https://arxiv.org/abs/2410.02367)
- FP8 mixed-precision inference
- Step caching mechanism

---

## 8. WorldLens: 高性能 3DGS 渲染平台

论文还介绍了 WorldLens,虽然技术细节较少,但特性包括:
- Engine-agnostic 架构
- Automatic IBL (Image-Based Lighting)
- Efficient collision detection
- Training-rendering co-design
- 支持 character 的交互式 3D world 探索

提取的 mesh 作为 collision proxy,支持实时物理反馈和空间交互,为 game、VR、embodied AI 提供下游应用基础。

---

## 9. 核心创新直觉总结

### 9.1 Generation 与 Reconstruction 的统一

HY-World 2.0 的核心 insight: reconstruction 能力是 generation 的必要组件,而非独立模块。WorldMirror 2.0 既独立作为 reconstruction foundation model 服务 dense multi-view input,又嵌入 generation pipeline Stage IV 作为几何 backbone。这种设计让生成场景获得 reconstruction 级别的几何精度,同时让 reconstruction 模型从 generation 数据中受益。

### 9.2 Keyframe Latent Space 的设计哲学

放弃 spatio-temporal Video-VAE 转向 spatial-only Keyframe-VAE,本质上是承认对于 3D 重建任务,**viewpoint coverage 重要性高于 frame continuity**。冗余的中间帧只是消耗 token budget,关键 keyframe 间的高保真度更重要。这与 [FlashWorld](https://arxiv.org/abs/2510.13678) 思路一致。

### 9.3 Spatial Stitching 优于 Temporal Concatenation

Table 8 的 A* vs A 对比非常有说服力:把 retrieved reference 通过 temporal concatenation 集成(像传统 long video generation 那样)会导致严重性能退化。Spatial stitching 让 attention 在 target-retrieval pair 内自然建立 correspondence,这与 stereo matching 的传统 insight 一致:对应关系应在空间维度建立。

### 9.4 Position Encoding 的归一化艺术

Normalized RoPE 的核心 insight:把 absolute integer 坐标的 extrapolation 问题转化为 normalized 坐标的 interpolation 问题。这与 [DINOv3](https://arxiv.org/abs/2508.10104) 的设计思路一致,本质上是承认神经网络在 fixed-range 上泛化更好。

### 9.5 Depth-Normal 耦合监督

Depth-to-normal loss 的妙处:不依赖完整 depth ground truth(许多 real-world dataset 没有),而通过 normal pseudo-label 间接监督 depth。Surface normal 描述局部 orientation 不需要全局 metric consistency,在 multi-view 设置下天然更鲁棒,这是一个非常实用的工程 insight。

### 9.6 MaskGaussian 的概率性建模

把 Gaussian 存在性建模为概率实体而非 hard pruning,允许 backward gradient 通过 Gumbel-Softmax relaxation 传递,实现动态重要性评估。Sparsity regularization 鼓励低概率 Gaussian 永久 prune,既减少冗余又保留高频细节。

---

## 10. 参考链接汇总

**核心论文/项目**:
- HY-World 2.0 Project: https://3d-models.hunyuan.tencent.com/world/
- GitHub: https://github.com/Tencent-Hunyuan/HY-World-2.0
- Scene to 3D Demo: https://3d.hunyuan.tencent.com/sceneTo3D
- WorldMirror: https://arxiv.org/abs/2510.10726
- Marble (闭源对比): https://marble.worldlabs.ai/

**对比方法**:
- [DiT360](https://arxiv.org/abs/2510.11712)
- [Matrix3D](https://arxiv.org/abs/2508.08086)
- [CubeDiff](https://arxiv.org/abs/2412.17904)
- [GenEx](https://arxiv.org/abs/2412.09624)
- [FlashWorld](https://arxiv.org/abs/2510.13678)
- [SEVA (Stable Virtual Camera)](https://arxiv.org/abs/2503.xxxxx)
- [Gen3C](https://arxiv.org/abs/2412.16960)
- [Lyra](https://arxiv.org/abs/2509.19296)
- [VGGT](https://arxiv.org/abs/2503.21851)
- [π3](https://arxiv.org/abs/2507.13347)
- [MapAnything](https://arxiv.org/abs/2509.13414)
- [DepthAnything3](https://arxiv.org/abs/2511.10647)
- [Fast3R](https://arxiv.org/abs/2501.13928)
- [CUT3R](https://arxiv.org/abs/2503.17910)
- [FLARE](https://arxiv.org/abs/2503.xxxxx)

**基础组件**:
- [MoGe2](https://arxiv.org/abs/2507.02546)
- [SAM3](https://arxiv.org/abs/2511.16719)
- [Qwen3-VL](https://arxiv.org/abs/2505.09388)
- [Q-Align](https://arxiv.org/abs/2312.17090)
- [3D Gaussian Splatting](https://repo.samuelgarcia.delarava.eu/inria_graffiti/2023_sigg_asia_3D_Gaussian_Splatting.pdf)
- [MaskGaussian](https://arxiv.org/abs/2412.13678)
- [Gumbel-Softmax](https://arxiv.org/abs/1611.01144)
- [RoPE (RoFormer)](https://arxiv.org/abs/2104.09864)
- [DINOv3](https://arxiv.org/abs/2508.10104)
- [DMD](https://arxiv.org/abs/2405.14867)
- [SageAttention2](https://arxiv.org/abs/2410.02367)
- [HunyuanVideo](https://arxiv.org/abs/2412.03603)
- [Wan](https://arxiv.org/abs/2503.20314)
- [CogVideoX](https://arxiv.org/abs/2408.06072)
- [Recast Navigation](https://github.com/recastnavigation/recastnavigation)
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
- [Tanks-and-Temples](https://www.tanksandtemples.org/)
- [MipNeRF360](https://arxiv.org/abs/2111.05505)
- [LeftRefill](https://arxiv.org/abs/2403.15812)

---

## 11. 个人直觉与思考

从 Karpathy 的视角看,这篇工作有几个值得深思的设计:

**1. Asymmetric Memory 设计**: GGM 用 point cloud 做"粗"全局结构,SSM++ 用 retrieved image 做"细"局部对应。这种 coarse-to-fine memory hierarchy 与人类视觉系统的 fovea + peripheral 结构有相似之处——全局感知用稀疏几何,细节感知用高保真图像。

**2. Token Budget 作为 First-Class 训练目标**: WorldMirror 2.0 的 token-budget-first 训练策略是一个非常工程化的 insight。传统做法是先定 resolution 再看 view,导致 GPU memory 利用率低。反过来固定 token budget 让 resolution 和 view count 自适应,本质上把硬件约束作为 first-class 设计目标,这种思路值得在更多 large model 训练中推广。

**3. Distillation 不损反增益**: DMD 蒸馏后 WorldStereo 2.0 在 T&T AUC 上从 58.19 提升到 60.09。这说明 full-step diffusion model 在某些 metric 上有冗余甚至噪声,few-step distillation 起到正则化作用。这与我在 distillation 方面的直觉一致——多步 diffusion 优化过程中容易过拟合 training distribution 的某些 artifact。

**4. Sky 与 Scene 分离处理**: 在 3DGS 训练中分离 sky 和 scene subset,只对 scene 做 densification,严格防止 sky 产生 floater。这是一个非常 domain-specific 的工程 insight,反映了对生成场景结构的深刻理解——sky 区域没有 depth supervision 且高频信息有限,任何 densification 都是纯冗余。

**5. Implicit vs Explicit Camera Prior 的权衡**: HY-Pano 2.0 放弃 explicit camera intrinsic 转向 implicit attention learning,而 WorldStereo 2.0 在 camera control 中保留 explicit Plücker rays + point cloud。这种不一致恰恰反映了任务特性:panorama generation 中 metadata 不可靠,implicit 学习更鲁棒;view synthesis 中 camera precision 是硬约束,explicit guidance 不可省。这是对不同任务"何时该隐式何时该显式"的精准把握。

整体而言,HY-World 2.0 是一个典型的"system paper"——单个组件可能不是 SOTA,但系统集成和工程优化达到极高水平。完整 pipeline 从 text/image 到可探索 3DGS 只需 10 分钟,在开源社区中是显著突破,与闭源 Marble 的对比也展示了竞争力。这种把 generation 先验和 reconstruction 精度统一的能力,正是 embodied AI、robotics simulation、game development 等下游应用所急需的。
