---
source_pdf: WorldExplorer.pdf
paper_sha256: de026c2dd0d048f1f54a85702da088f65af8ff229ca96370703719c45c0ad872
processed_at: '2026-08-13T05:40:35-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 WorldExplorer

## 一句话

从一句话描述生成一个能进去逛的 3D 房子，逛的时候不掉画质。之前要么好看但走不动，要么能走但越走越糊，这篇两边都搞定了。

---

## 问题到底难在哪

你想从 "深海发光水母 hive 公寓" 这种 prompt 做出一个 3D 场景，让人能走进去、绕到沙发背后、抬头看天花板。三条老路都有毛病：

**第一条：生成 360 全景图，lift 成 3D**  
DreamScene360 / LayerPano3D 干这事。你站在房间正中间看一圈很漂亮，但只要你往前走两步或者绕到物体背面——完了，背面从来没被看见过，inpaint 出来全是扭曲的。本质就是只拍了一张球面照片，没多视角信息。

**第二条：一张图一张图往外面"长"**  
Text2Room / WonderWorld 这类，用 depth 估计 + inpainting 一帧帧往外推。问题：depth 估错一帧，后面所有帧都基于这个错的位置继续画，物体越画越拉长，最后全是面条状的椅子腿。**误差累积**。

**第三条：用 video diffusion 拍一段绕圈视频**  
FlexWorld / SEVA 用 camera-guided video diffusion 生成连续帧。局部看挺好，但绕一大圈回到原点，模型已经忘了原来那张桌子长啥样了——会出现"两张桌子叠在一起"的诡异画面。**长程记忆崩了**，catastrophic forgetting。

WorldExplorer 的核心 insight 就是把这三条的优点拼起来，避开各自缺点。

---

## 怎么做的：三个阶段

### 阶段一：先搭骨架（8 张全景图）

不用单一 panorama generator（它倾向于生成"全是客厅"那种单调场景），而是用 Flux 独立生成 4 张图，每张描述不同房间：厨房、客厅、卧室、办公室。4 张图相机原点都在 (0,0,0)，朝向 0°/90°/180°/270° 外看，形成一个十字布局。

然后用 Depth Anything V2 估 depth，把这 4 张图反投影成 4 团 point cloud。再在 45°/135°/225°/315° 这 4 个对角方向定义新相机，把已有 point cloud splat 过去，没覆盖的地方 inpaint 出来——得到 8 张全景图当 scaffold。

**人话**：先把"世界的四分之一象限"用 4 张图定下来，剩下 4 个对角空缺用已有信息投影+补画填上。这样最终有 8 张图覆盖 360°，而且 4 个区域可以风格完全不同（室内厨房和室外森林都能拼一起）。

这 8 张图后面永远作为"全局锚点"出现，模型每生成一段新视频都看着这 8 张图，不会跑偏。

### 阶段二：从骨架出发，生成 32 条小视频

这是 paper 的核心。从 8 张全景图每张出发，生成 4 条预定义 trajectory：

1. **zoom in**：沿一条直线往前走（44 帧）
2. **rotate left**：往左绕 134 帧
3. **rotate right**：往右绕 134 帧
4. **elevate**：往上抬头 44 帧

$8 \times 4 = 32$ 条 video，每条平均 70 帧，最后攒出 ~2250 张图。

每条 trajectory 内部按 21 帧 batch 生成（继承 SEVA 的 sliding window）。每个 batch 需要 13 帧 conditioning，这 13 帧怎么选是关键创新——

#### Scene Memory：13 帧 context 怎么填

- **8 帧**：永远是阶段一的 8 张全景图。这相当于 system prompt，保证模型每一步都"记得"世界全貌。
- **5 帧**：从所有之前生成过的帧里，按**相机朝向相似度**检索 top-5。为什么用 rotation 不用 translation？因为朝向一样看到的东西基本一样，translation 近但朝向不同看的是完全不同的墙。

trajectory 生成完后，把第 3 帧之后的图加进 memory（前 2 帧太接近初始全景图，加进去冗余）。

**人话类比**：这跟 LLM 的 RAG 一模一样。8 张全景图是 system prompt（永远在 context 里），5 张检索是 RAG 拉出来的相关历史 token。模型一边生成一边往外部 memory 里写新内容，下次生成再检索。

#### Collision Detection：别撞墙

trajectory 是 a-priori 定死的，不管生成出什么内容都按这条路走，肯定会撞墙。怎么办？

生成完整个 trajectory 之后，用 Video Depth Anything 算每帧 normalized depth。取图像中心 20% 区域（115×115 像素）算平均 depth $\bar{d}_t$。从第一个满足 $\bar{d}_{t^*} < 0.4$ 的帧开始，把后面所有帧丢弃。

**人话**：中心区域深度突然变浅，说明相机已经怼到墙上了。用中心 crop 不用全图，因为图像边缘就算贴墙也会有低 depth，会误判。0.4 是经验阈值。

为什么这种简单 hard threshold 能用？因为 video diffusion prior 已经把帧间 smoothness 处理得不错了，depth 信号相对干净。

### 阶段三：3D Gaussian Splatting 重建

2250 张图怎么 fuse 成一个 3D 场景？传统 COLMAP 跑 1K+ 张图又慢又容易炸，改用 VGGT——一个前馈 transformer，一次性吃多张图吐出 point cloud + camera pose。

VGGT 在自己坐标系预测，要 align 到我们的已知 pose：

1. **Rigid transform**：用第 1 张图对齐，$\pi_t = \pi_1 \hat{\pi}_1^{-1}$
2. **Scale**：用所有相机中心的 bounding box 周长做尺度比，$\mathbf{s} = \text{hull}(\pi) / \text{hull}(\hat{\pi})$
3. **Transform**：$\mathcal{P}_{GS} = \mathbf{s} \cdot \pi_t \cdot \hat{\mathcal{P}}_{GS}$

point cloud 下采样到 200K 点作为 3DGS init，然后标准 3DGS 优化：

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left( \lambda_1 |\hat{\mathbf{I}}_i - \mathbf{I}_i| + \lambda_2 (1 - \mathrm{SSIM}(\hat{\mathbf{I}}_i, \mathbf{I}_i)) \right)$$

- $\hat{\mathbf{I}}_i$：3DGS 渲染出来的图
- $\mathbf{I}_i$：阶段二生成的图当 ground truth
- $\lambda_1 = 0.8, \lambda_2 = 0.2$：L1 + SSIM 的常规配比

**人话**：3DGS 优化在这里其实是"投票机"。32 条 video 之间有小不一致（同一面墙被两条 trajectory 看到但亮度略不同），3DGS 通过 photometric loss 把它们平均成一个一致的 3D 表征。本质上是 ensemble denoising。

---

## 结果：到底有多好

User study n=64，满分 5 分：

| 指标 | 我们的 | 最强 baseline |
|---|---|---|
| Perceptual Quality | 4.04 | 3.01 (DreamScene360) |
| 3D Consistency | 4.02 | 3.04 (DreamScene360) |
| CLIP Score | 25.94 | 24.37 (Text2Room) |

PQ 和 3DC 几乎是满分，吊打所有 baseline。但 Inception Score 没拿最高（2.27 vs FlexWorld 2.31），说明 2D 单帧画质和 baseline 持平，主要赢在 3D 一致性。

时间成本：7 小时一个场景（RTX 3090），生成 ~2250 帧。Per-frame ~14 秒，跟 baseline 持平，只是因为生成 frame 多所以总时间长。

---

## Ablation：哪个组件最重要

1. **8 张全景图砍成 1 张**：场景变小、不 diverse，因为模型要 from-scratch hallucinate 新区域，要么糊（低 guidance）要么过饱和（高 guidance）
2. **砍成 4 张**：场景大但跨 trajectory 不一致，3D 重建出 floating artifacts
3. **去掉 scene memory**：3DC 从 4.02 掉到 2.36——每条 trajectory 各画各的，连物体类型都对不上
4. **去掉 collision detection**：撞墙区域边缘 fuzzy。但只有撞到的那条 trajectory 受影响，其他 trajectory 不受影响——这是 iterative 设计的鲁棒性体现

---

## 这套设计本质上在干啥

我觉得最深的洞察是：**video diffusion 是个隐式 world model，但它的"工作记忆"很短**。你让它一次绕一大圈，它会忘；你让它一段一段绕，每段开始时把全局地图摆桌上（8 张全景图）+ 翻一下之前拍过的相关照片（5 张 retrieval），它就能保持一致。

这跟 LLM 的 prompt engineering 是同构的：
- 8 张全景图 = system prompt（全局上下文）
- 5 张 retrieval = RAG（按相关性从历史里拉 token）
- 多段 trajectory = chain-of-thought 分段生成（避免长程生成 forgetting）
- 3DGS 优化 = test-time consensus（多 sample 投票去噪）
- Collision detection = step-level filter（bad step → bad trajectory，砍掉）

所以这篇 paper 本质是把 LLM 时代摸索出来的 context engineering 最佳实践移植到了 3D scene generation。

---

## 还差什么

1. **7 小时太久**：interactive world generation 需要秒级，streaming video diffusion 能解决
2. **预定义 trajectory 是死板的**：4 种 trajectory 类型相当于 hardcoded exploration policy。未来该用 active SLAM 那种"哪里没看过就去看哪里"的 policy
3. **物体背面经常没覆盖**：Fig. 10 里扶手椅背面贴墙了，没 trajectory 能看到，3D 重建就退化。需要 adaptive trajectory
4. **RGB space diffusion → baked lighting**：没法 relight / 改 material。接入 IntrinsiX 那种 PBR 生成能解决
5. **Flickering 仍然有**：video diffusion 的高频抖动 → 3DGS blur。可以引入 RobustNeRF 那种 robust loss
6. **Memory retrieval 用 rotation similarity 是手工 metric**：换成 CLIP/DINO image embedding 做语义检索更聪明

---

## 我的整体判断

这是 text-to-3D-scene 这条线目前最 solid 的一步——第一次做到"真正能进去逛"的生成场景。前作要么好看但走不动，要么能动但糊。这篇通过 decomposition + global anchor + retrieval + consensus 这套组合拳把两条短板都补上了。

但离"interactive world generation"（秒级、可控、可编辑、4D、relightable）还有 2-3 代工作的距离。预定义 trajectory、baked material、7 小时 runtime 这三个是主要 adoption barrier。

我感觉接下来 6-12 个月会出现：
1. streaming video diffusion 版本，把生成降到秒级
2. 用 VLM 做 next-best-view exploration policy 替代预定义 trajectory
3. PBR material 生成接入，解锁 relighting
4. Dynamic 3DGS + time-aware video diffusion，生成 4D 世界

总之这篇值得读，不是因为方法多 fancy（其实组件都很朴素），而是这个组合设计——把 decomposition + global anchor + retrieval 这套范式在 3D 上验证有效——是个有迁移价值的 insight。

---

# WorldExplorer: Text-to-Fully-Navigable 3D Scenes via Iterative Video Diffusion

## 1. 一句话定位

这是 TUM 的 Matthias Nießner 实验室(SIGGRAPH Asia 2025)的一篇 paper,核心贡献是用 camera-guided video diffusion model 以"**autoregressive multi-trajectory + scene memory**"的方式,从 text 生成可以 360° 自由导航(走出 panorama 中心、绕到物体背面)的 3D Gaussian Splatting scene。前作 SEVA / FlexWorld / ViewCrafter 都在 video diffusion 上做到局部多视角,但在 large camera motion 下会有 flickering / duplication / stretching;这篇的核心 insight 是把"single long trajectory"拆成"多个短而 discontinuous 的 predefined trajectory",再通过 memory 把它们绑成一致的世界。

Paper link: <https://doi.org/10.1145/3757377.3763946>
Project page(估计): <https://niessnerlab.org/projects/schneider2025worldexplorer.html>
arXiv 预印本相关: <https://arxiv.org/abs/2505.18414>(SEVA 主干,Zhou et al. 2025)

---

## 2. Motivation:为什么前作都还不够

text-to-3D scene 这一脉有几条线:

1. **Panorama-to-3D**: DreamScene360 [Zhou et al. 2024] <https://dreamscene360.github.io/>、LayerPano3D [Yang et al. 2024b] <https://arxiv.org/abs/2408.13252>。先 text-to-panorama 再 lift 成 3DGS。问题是 panorama 中心视角漂亮,**一旦偏离中心就有 occlusion + distortion**(因为背面没观测,只能 inpaint)。

2. **Iterative render-refine-repeat (T2I lifting)**: Text2Room [Höllein et al. 2023] <https://arxiv.org/abs/2303.12989>、WonderWorld [Yu et al. 2024a] <https://wonderworld-2024.github.io/>、Realm Dreamer [Shriram et al. 2024] <https://arxiv.org/abs/2404.07199>。逐帧 unproject + monocular depth + inpainting。问题是 **error accumulation**——depth 估错一帧后续几何就一直 stretch,物体被拉成面条状。

3. **Video diffusion as scene generator**: FlexWorld [Chen et al. 2025b] <https://arxiv.org/abs/2503.13265>、SEVA [Zhou et al. 2025] <https://stablevirtualcamera.github.io/>、Gen3C [Ren et al. 2025] <https://arxiv.org/abs/2503.03751>、ViewCrafter [Yu et al. 2024c] <https://arxiv.org/abs/2409.02048>。Video diffusion 是个隐式 world model,沿 camera path 生成连续帧能避免 stretching。但**单条长 trajectory 有 catastrophic forgetting**:你绕一圈回到原点,模型已经忘了原来桌子长啥样,会出现"叠加的多张桌子"这种 duplication artifact(见 paper Fig. 9)。

WorldExplorer 把 (1)(2)(3) 的优点拼起来:用 panorama 当全局 scaffold(解决 forgetting),用 video diffusion 当局部几何 prior(解决 stretching),用 iterative + scene memory 把它们绑成一致(解决 cross-trajectory inconsistency)。

---

## 3. Method:三阶段管线详解

### Stage 1: Panorama Initialization(8 张图的 scene scaffold)

**关键创新**:不是用单一 panorama generator(它倾向于生成单一场景类型,比如"全部是 living room"),而是用 Flux [Black Forest Labs 2023] <https://github.com/black-forest-labs/flux> 独立生成 4 张图,每张描述不同区域(e.g. kitchen / living room / bedroom / office),让用户能用 prompt 控制 4 个房间类型组合。

**Camera pose 定义**(Eq. 1):

$$R_i = R_y(\theta_i), \quad \theta_i \in \{0°, 90°, 180°, 270°\}, \quad t_i = 0$$

- $R_y(\theta)$:绕全局 $y$ 轴(yaw)旋转角度 $\theta$ 的 rotation matrix ∈ SO(3)
- $i \in \{1,2,3,4\}$:第 i 张初始图
- $t_i = 0$:4 张图共享同一原点,只是朝向不同
- 这种"cross 形"布局让 4 个 prompt 自然对应 4 个 explorable region

**Unproject 成 point cloud**:

$$\mathcal{P}_i = R_i \, K^{-1} [u, v, 1]^\top \cdot \mathbf{D}_i(u,v)$$

- $K$:pinhole camera intrinsics,FOV = 60°
- $[u,v,1]^\top$:pixel 坐标齐次化
- $K^{-1}[u,v,1]^\top$:把 pixel ray 方向从 image plane 反投影到 camera space(单位向量)
- $R_i$ 乘到 camera space 方向上:转到 world space 方向
- $\mathbf{D}_i(u,v)$:Depth Anything V2 [Yang et al. 2024a] <https://arxiv.org/abs/2406.09414> 估的 depth,作为长度 scale
- 最终 $\mathcal{P}_i \in \mathbb{R}^{3}$ 是 world space 3D 点

**补成 8 张 panorama**:再定义 4 个对角方向的 pose(45°/135°/225°/315°),把前面 4 张图的 point cloud 用 PyTorch3D [Ravi et al. 2020] / SynSin [Wiles et al. 2020] 的 point splatting 投影到新视角,然后 inpaint 未观测区域。结果:8 张 panorama scaffold 作为后续的"全局坐标系锚"。

直觉:**这步把"世界的骨架"先建起来,后续 video diffusion 只需要 inpaint 而不是 from-scratch hallucinate**——这降低了 video diffusion 的负担(参见 ablation,只用 1 张图时 scene 显著降质)。

### Stage 2: Iterative Video Trajectory Generation(核心)

**总体框架**:对 8 张 panorama 中的每张(其实是 4 张原始 + 4 张 inpainted),都生成 4 条 predefined trajectory:zoom-in(1)、rotate-left(2)、rotate-right(3)、elevate-up(4)。trajectory 数固定为 $4 \times 8 = 32$。每条 trajectory 长度:trajectory (1)(4) 是 44 帧,(2)(3) 是 134 帧(因为旋转类需要更多帧才能环绕一圈)。最终 ~1.5K-2.5K frames per scene,分辨率 576×576。

**为什么不用单条长 trajectory?**
- 单条长 trajectory 难以 a-priori 定义覆盖整个 scene 的路径(场景内容未知)
- 长程 memory 在 video diffusion 里仍很差(SEVA 在远端会有 flickering 和 duplication)
- Collision 处理变难:前面撞墙了后面无法恢复

**Trajectory 生成是 autoregressive 的 batch**:每条 trajectory 内部按 21 帧 batch 生成(类似 SEVA 的 sliding window)。每 batch 的 conditioning 是 13 帧 = 8 panorama + 5 retrieval。

#### 3.2.1 Camera-guided video diffusion 的形式

video diffusion model 建模:

$$p_\theta(\mathbf{I}^{tgt} | \mathbf{c}), \quad \mathbf{c} = (\mathbf{I}^{src}, \pi^{src}, \pi^{tgt})$$

- $\mathbf{I}^{tgt}$:target image set(要生成的 21 帧 batch)
- $\mathbf{I}^{src}$:source / conditioning image set(13 帧)
- $\pi^{src}, \pi^{tgt}$:对应 camera pose,编码成 **Plücker embeddings** [Liang et al. 2024] <https://arxiv.org/abs/2412.12091>(每条 ray 用 6D origin+direction 表示,这是 manifold-friendly 的 pose 表示)
- $p_\theta$ 用 flow-matching [Lipman et al. 2022] <https://arxiv.org/abs/2210.02747> 训练的 DiT 主干(具体用 SEVA 的 checkpoint)

公式细节:

$$p_\theta(\mathbf{I}^{tgt} | \mathbf{c}) = \int_\Omega p_\phi(\mathbf{I}_{0:T}^{tgt} | \mathbf{c}) \, d\mathbf{I}_{1:T}^{tgt}$$

- $\mathbf{I}_{0:T}^{tgt}$:diffusion 的 latent trajectory,$\mathbf{I}_0$ 是 clean,$\mathbf{I}_{1:T}$ 是加噪版本
- $p_\phi$ 是 denoising distribution
- $\Omega$:latent space
- 这是标准 DDPM [Ho et al. 2020] 的 marginal likelihood 形式

#### 3.2.2 Scene Memory:跨 trajectory 的 conditioning

这是 paper 的核心创新。每个 21 帧 batch 的 13 个 conditioning slot 划分:

- **Slot 1-8(固定)**:8 张 panorama scaffold。永远在 context 里 → 模型始终知道全局布局。
- **Slot 9-13(动态)**:基于 **rotational similarity** 从所有 previously generated frames 中检索 top-5 nearest camera poses。

**为什么用 rotation 而不是 translation?** 这是个 subtle design choice:
- 相机朝向决定视野内容
- translation 近但朝向不同 → 看到完全不同的物体
- translation 远但朝向相同 → 至少看到同一面墙
- 检索 metric 用 rotation distance 更鲁棒

**Anti-degenerate trick**:剔除和当前 trajectory 初始图直接相对(180° yaw 差)的相机——因为这种 view 通常看到相反区域,不相关。

**Memory update**:trajectory 生成完后,把**第 3 帧之后**的图像加入 memory(前 2 帧太接近 panorama,放进 memory 会冗余)。

直觉对比:这非常像 LLM 的 **in-context learning + retrieval**。固定 8 张 panorama 像 system prompt / global context,动态 5 张像 RAG 检索的相关 history token。和 RETRO [Borgeaud et al. 2022] / Memorizing Transformer [Wu et al. 2022] 的思路本质一致——把全 history 当外部 database,query-time 检索相关片段。

#### 3.2.3 Collision Detection:防止 trajectory 撞墙

因为 trajectory 是 a-priori 定义(不管生成什么内容都用同样路径),需要 dynamic adjustment。

**Mask 定义**(Eq. 2):

$$\mathbf{M}(u,v) = \begin{cases} 1 & \text{if } u \in [\frac{W-w}{2}, \frac{W+w}{2}), v \in [\frac{H-h}{2}, \frac{H+h}{2}) \\ 0 & \text{otherwise} \end{cases}$$

- $W, H$:图像宽高(576)
- $w, h$:center crop 尺寸,设为图像分辨率 20%(即 ~115×115 像素区域)
- 关键:用 center crop 而不是全图,因为图像边缘即使贴近墙也会出现低 depth,会误判 trajectory collision。Center crop 是 conservative 的 collision indicator。

**Masked average depth**(Eq. 3):

$$\bar{d}_t = \frac{\sum_{u,v} \mathbf{M}(u,v) \cdot \hat{\mathbf{D}}_t(u,v)}{\sum_{u,v} \mathbf{M}(u,v)}$$

- $\hat{\mathbf{D}}_t \in [0,1]^{N \times H \times W}$:Video Depth Anything [Chen et al. 2025a] <https://arxiv.org/abs/2501.12375> 估的 normalized depth
- $t \in \{1, \ldots, N\}$:frame index

**Discard rule**:从第一个满足 $\bar{d}_{t^*} < 0.4$ 的 frame $t^*$ 开始,丢弃之后所有帧。

直觉:normalized depth 0.4 大约对应中等距离;center crop 平均深度突然降低,意味着相机已经走到墙/物体前。这种 hard threshold 看起来 brittle,但 video diffusion prior 已经把帧间 smoothness 处理好了,所以有效。让我联想到 chain-of-thought 早期工作中的"step-level filter"。

### Stage 3: 3D Gaussian Splatting Optimization

为什么不用 COLMAP [Schonberger & Frahm 2016] <https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf>? 因为 1K+ 张图 COLMAP 慢且不稳。改用 **VGGT** [Wang et al. 2025a] <https://arxiv.org/abs/2503.11651>(Visual Geometry Grounded Transformer,前馈式 SfM)。

**Coordinate alignment**:VGGT 在自己的 world coord 预测 point cloud $\hat{\mathcal{P}}_{GS}$ 和 camera pose。需要 align 到 known pose。

**Rigid transform**(用第一张图):
$$\pi_t = \pi_1 \hat{\pi}_1^{-1}$$
- $\pi_1$:第 1 张图的已知 camera pose(WorldExplorer 体系下)
- $\hat{\pi}_1$:VGGT 预测的第 1 张图 camera pose(VGGT 体系下)
- $\pi_t$:从 VGGT 体系到 WorldExplorer 体系的 rigid transform

**Scaling**:
$$\mathbf{s} = \frac{\text{hull}(\{\pi_i\}_{i=1}^N)}{\text{hull}(\{\hat{\pi}_i\}_{i=1}^N)}$$
- $\text{hull}(\cdot)$:所有 camera center 的 bounding box 周长
- 这是用 camera 中心 bounding box 做尺度归一化,因为两套坐标系尺度可能不一致

**Final transform**:
$$\mathcal{P}_{GS} = \mathbf{s} \cdot \pi_t \cdot \hat{\mathcal{P}}_{GS}$$
- 先 rigid transform 再 scale

**3DGS optimization loss**(Eq. 4,标准 Kerbl et al. 2023 [https://repo.sam.labtwin.org/3dgs]):

$$\mathcal{L}(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^{N} \left( \lambda_1 |\hat{\mathbf{I}}_i - \mathbf{I}_i| + \lambda_2 (1 - \mathrm{SSIM}(\hat{\mathbf{I}}_i, \mathbf{I}_i)) \right)$$

- $\hat{\mathbf{I}}_i$:3DGS rasterization 渲染出的 image
- $\mathbf{I}_i$:Stage 2 生成的 frame(作为 ground truth)
- $\lambda_1, \lambda_2$:loss weights(3DGS 默认 $\lambda_1 = 0.8, \lambda_2 = 0.2$)
- $|\cdot|$:L1 photometric loss
- $1 - \mathrm{SSIM}$:structural similarity 补充(纯 L1 会偏 blurry)

**VGGT point cloud 下采样到 200K** 作为 3DGS init。这个数是个 trade-off:太密训练慢,太稀几何细节丢。

直觉:**3DGS optimization 在这里充当 cross-trajectory consensus mechanism**。多个 video 之间有 small inconsistency,3DGS 通过 photometric loss 把它们"平均"成一个 consistent 3D 表征。这本质上是 Bayesian model averaging / denoising ensemble。

---

## 4. Experiment Results

### Implementation Details

- **Hardware**: single RTX 3090
- **总运行时间**: ~7 hours
- **Stage breakdown**:
  - Stage 1(panorama):~5 min
  - Stage 2(video diffusion):~10 min/trajectory × 32 = ~5.3 h ← 主要 bottleneck
  - Stage 3(3DGS optimization):~11 min
- **生成 frame 数**: 1.5K–2.5K per scene @ 576×576

### Quantitative Results(Table 1)

| Method | CS↑ | IS↑ | PQ↑ | 3DC↑ |
|---|---|---|---|---|
| DreamScene360 | 23.51 | 2.06 | 3.01 | 3.04 |
| LayerPano3D | 24.30 | 2.04 | 2.16 | 2.14 |
| Text2Room | 24.37 | 1.92 | 2.68 | 2.32 |
| WonderWorld | 19.74 | 2.23 | 2.34 | 1.75 |
| FlexWorld | 21.61 | 2.31 | 2.89 | 2.71 |
| SEVA | 21.63 | 2.13 | 2.51 | 1.91 |
| Ours (1 image) | 21.42 | 1.88 | 2.42 | 2.03 |
| Ours (4 images) | 25.65 | 2.29 | 2.48 | 2.14 |
| Ours (w/o scene-mem) | 25.91 | 2.21 | 2.61 | 2.36 |
| Ours (w/o coll-det) | 25.37 | 1.89 | 3.77 | 3.42 |
| **Ours (full)** | **25.94** | 2.27 | **4.04** | **4.02** |

**关键观察**:
- **CLIP Score 25.94 最高** → panorama scaffold 让模型不必 from-scratch hallucinate,prompt alignment 更强
- **PQ 4.04 / 3DC 4.02(满分 5)**:user study n=64, 2432 datapoints。显著优于所有 baselines
- **Inception Score 没拿最高**:说明生成的 image sharpness 和 baselines 持平,主要赢在 3D 一致性而非 2D quality
- **w/o scene-mem 的 3DC 只有 2.36**:验证 scene memory 是关键创新
- **w/o coll-det 的 PQ/3DC 下降明显**:collision 会造成 blurry 边缘

### Runtime/Memory(Table 2)

| Method | Total runtime | Generated frames | Runtime per frame | Memory |
|---|---|---|---|---|
| DreamScene360 | 0.5h | one panorama | N/A | 16GB |
| LayerPano3D | 0.2h | one panorama | N/A | 5GB |
| Text2Room | 0.9h | 200 | 16.2s | 15GB |
| WonderWorld | user-specific | – | ~10s | 45GB |
| FlexWorld | 0.5h | 144 | 12.9s | 48GB |
| SEVA | 0.9h | 270 | 13.5s | 22GB |
| **Ours** | **7h** | **2251** | **14.1s** | **22GB** |

**Per-frame runtime comparable**(~14s),主要因为生成 frame 数多(2251 vs 144-270),所以总时长长。本质上是用更多 sample 换取 3D completeness。

### Ablation Insight(Fig. 5)

1. **Panorama init 数量**:
   - 1 image → scene 小且不够 diverse,模型要 from-scratch hallucinate 新区域,要么 oversimplified(低 guidance),要么 oversaturated(高 guidance)
   - 4 images → 大场景但 cross-trajectory 不一致,3DGS blur + floating artifacts
   - 8 images → sweet spot

2. **Scene memory**:删了的话每个 trajectory 只看局部,生成不同物体类型 → 严重 3D 不一致 → blur

3. **Collision detection**:删了 trajectory 会撞墙,撞墙区域边缘 fuzzy。但因为 trajectory 之间 disjoint,只影响撞墙的 trajectory,其它 trajectory 不受影响——这是 iterative 设计的 robustness 体现

---

## 5. Limitations

1. **High-frequency flickering**:video diffusion 本身的 frame-to-frame flickering → 3DGS blur(尤其远端)
2. **Pre-defined trajectory 不全 cover**:Fig. 10 中扶手椅背面没被任何 trajectory 看到 → 退化渲染
3. **Baked lighting/material**:RGB-space diffusion → 无法 relight / re-material
4. **Fixed layout**:4 房间 cross 形限制了 scene topology,虽然 Fig. 11 展示可以通过把 "dining room" / "bedroom" 换成 "wall" 生成 L-shape

---

## 6. 与 LLM 训练的直觉关联(build your intuition)

这篇 paper 的设计哲学其实和现代 LLM 的几个核心 trick 高度同构:

### 6.1 Scene Memory ≈ KV-cache + Retrieval

固定的 8 张 panorama 像 system prompt / global context(永远在 KV-cache 里),动态的 5 张 retrieval 像 RAG 检索的 history token。这个 13-slot context window 是手工设计的 prompt template。可以想象 next-gen 用 learnable retrieval(e.g., CLIP image embedding 做 similarity)替代 rotation similarity。

### 6.2 Multi-trajectory ≈ Chain-of-Thought decomposition

单条 long trajectory 类似"一步到位生成全文",会 catastrophic forgetting。Multi-trajectory 类似"分章节生成 + 全局 outline 锚定"。这种 decomposition + global anchor 的范式在 long-form generation 里被反复验证有效。

### 6.3 Collision detection ≈ Reward shaping / step-level filter

hard threshold $\bar{d}_t < 0.4$ 是一个 step-level filter,把 degenerate 早期砍掉。这跟 RLHF / process supervision 中"bad step → bad trajectory"的思路一致。

### 6.4 3DGS optimization ≈ Test-time consensus

多个不一致的 video 通过 3DGS photometric loss "投票"出一致 3D 表征。这跟 ensemble / self-consistency 的 test-time scaling 思想一致。

### 6.5 Video diffusion 是隐式 world model

SEVA / Stable Virtual Camera 这类 camera-conditioned video diffusion 本质是被蒸馏出来的 visual world model。把 world model 转成显式 3DGS scene 是一种 **explicit memory externalization**——生成的 scene 像 episodic memory,future queries 可以通过 rasterization "检索"。

参考 Sora-style world simulator 讨论:<https://openai.com/research/video-generation-models-as-world-simulators>

---

## 7. 相关工作谱系

### 7.1 Score Distillation 谱系(text-to-3D object)
- DreamFusion [Poole et al. 2023] <https://dreamfusion3d.github.io/>: SDS for single object
- ProlificDreamer [Wang et al. 2023] <https://prolificdreamer.github.io/>: VSD 改进 SDS
- Fantasia3D [Chen et al. 2023] <https://fantasia3d.github.io/>: disentangle geometry/appearance

### 7.2 Multi-view Diffusion 谱系
- MVDiffusion [Tang et al. 2023] <https://arxiv.org/abs/2307.01097>
- CAT3D [Gao et al. 2024] <https://cat3d.github.io/>: multi-view diffusion + 3DGS
- ViewDiff [Höllein et al. 2024] <https://viewdiff.github.io/>
- Bolt3D [Szymanowicz et al. 2025] <https://arxiv.org/abs/2503.14445>: 3D in seconds

### 7.3 Iterative scene generation 谱系
- Infinite Nature [Liu et al. 2021] <https://infinite-nature.github.io/>: perpetual view generation 单图
- WonderJourney [Yu et al. 2024b] <https://wonderjourney2024.github.io/>
- Text2Room [Höllein et al. 2023] <https://niessnerlab.org/projects/hollein2023text2room.html>
- Realm Dreamer [Shriram et al. 2024] <https://arxiv.org/abs/2404.07199>
- WonderWorld [Yu et al. 2024a] <https://wonderworld-2024.github.io/>

### 7.4 Panorama-to-3D 谱系
- DreamScene360 [Zhou et al. 2024] <https://dreamscene360.github.io/>
- LayerPano3D [Yang et al. 2024b] <https://layerpano3d.github.io/>
- Perf [Wang et al. 2024] <https://arxiv.org/abs/2408.07015>
- SceneDreamer360 [Li et al. 2024a] <https://arxiv.org/abs/2408.13711>

### 7.5 Camera-guided video diffusion 主干
- CameraCtrl [He et al. 2024] <https://arxiv.org/abs/2404.02101>
- CameraCtrl II [He et al. 2025] <https://arxiv.org/abs/2503.10592>
- DimensionX [Sun et al. 2024] <https://ali-vilab.github.io/DimensionX/>
- ViewCrafter [Yu et al. 2024c] <https://arxiv.org/abs/2409.02048>
- SEVA [Zhou et al. 2025] <https://stablevirtualcamera.github.io/> ← WorldExplorer 用的主干
- Gen3C [Ren et al. 2025] <https://gen3c.github.io/>: 3D-informed video generation
- History-Guided Video Diffusion [Song et al. 2025] <https://arxiv.org/abs/2502.06764>

### 7.6 3D Reconstruction 工具
- 3D Gaussian Splatting [Kerbl et al. 2023] <https://repo.sam.labtwin.org/3dgs>
- VGGT [Wang et al. 2025a] <https://vgg-t.github.io/>
- Depth Anything V2 [Yang et al. 2024a] <https://depth-anything-v2.github.io/>
- Video Depth Anything [Chen et al. 2025a] <https://arxiv.org/abs/2501.12375>
- Flux [Black Forest Labs 2023] <https://blackforestlabs.ai/>

### 7.7 Future 方向参考
- IntrinsiX [Kocsis et al. 2025] <https://arxiv.org/abs/2504.01008>: PBR material generation(解耦 albedo/roughness/metallic)
- Marigold [Ke et al. 2025] <https://arxiv.org/abs/2505.09358>: diffusion-based dense prediction
- Bayes' Rays [Goli et al. 2024] <https://arxiv.org/abs/2403.10166>: NeRF uncertainty
- RobustNeRF [Sabour et al. 2023] <https://robustnerf.github.io/>: distractor-resistant NeRF

---

## 8. 我的几点批评 / 联想

1. **Scaling 方向**:32 trajectory × ~70 frames = 2.2K。如果 scaling 到 256 trajectory(覆盖更密视角),memory retrieval 会成瓶颈。可以引入 hierarchical memory(e.g., voxel-grid 索引 retrieved frames)。

2. **Pre-defined trajectory 是 brittle prior**:4 种 trajectory 类型本质是 hardcoded exploration policy。未来可以引入 curiosity-driven / coverage-maximizing exploration policy(RL 学习"下一步生成哪里"),就像 active SLAM。

3. **3DGS 作为 denoising target 的局限**:3DGS optimization 收敛速度依赖 init point cloud 质量。VGGT 失败场景(低 texture 区域)会直接传递到 3DGS。可以引入 bundle-adjustment refinement。

4. **Baked lighting 是深层限制**:RGB-space diffusion 决定了所有 light/material 都 baked in。IntrinsiX [Kocsis et al. 2025] 显示可以从 image prior 恢复 PBR material,这条线接入后会解锁 relighting。

5. **4D extension**:video diffusion 本身时序信息已经 rich。要生成 dynamic scene(行人走动、烟雾),需要 Dynamic 3DGS [Yang et al. 2024] <https://dynamic3dgaussians.github.io/> + time-aware video diffusion。这是 SIGGRAPH 2026 大概率会出现的工作。

6. **Failure mode 仍然可见**:Fig. 10 的 floating artifacts 表明 video diffusion 的 flickering 没完全被 3DGS 解决。可以考虑 robust loss(Sabour et al. 2023)或 uncertainty weighting(Goli et al. 2024)。

7. **Memory retrieval metric 可升级**:rotation similarity 是手工 metric。用 CLIP/DINO image embedding 做 visual similarity 检索更语义化——找到视觉上相关的历史帧,而非仅仅朝向相似。

8. **与 SDXL/Flux 的 prompt engineering**:8 张 panorama 的 prompt 是用户写。可以让 LLM 自动 decompose 复杂 scene 成多 room prompt(像 Cosmic Jellyfish Hive 这种 fantasy prompt)。

9. **Interactive editing missing**:目前是 text → 静态 scene。如果允许用户在生成过程中"指定这里放一张沙发"会非常有用,类似 ControlNet [Zhang et al. 2023a] <https://lllyasviel.github.io/ControlNet/> 在 3D 上的扩展。

10. **Real-time generation**:7h 总时长是主要 adoption barrier。Streaming video diffusion(Streaming T2V, LTX-Video)能把 trajectory generation 降到秒级,那就能做 interactive world generation。

---

## 9. 总结直觉

WorldExplorer 的核心 insight:**当 video diffusion model 作为 world prior 时,与其逼它在一条 long trajectory 上 remember 整个世界(它会 forget),不如把世界拆成多个 local exploration episodes,每个 episode 都用 global scaffold 锚定 + retrieval 补充 history**。这本质上是把 LLM context engineering 的最佳实践(video 模型层面的 RAG + global prompt + decomposed generation)移植到 3D scene generation。3DGS optimization 在最后充当 consensus / denoising 角色,把多个 episode 的不一致 fuse 成单一 coherent 3D 表征。

最终这是迈向 "text → fully navigable immersive 3D world" 的扎实一步,但离真正的 interactive world generation(秒级、可控、可编辑、4D、relightable)还有 ~2-3 代工作的距离。

相关 reading list 推荐:
- SEVA(stable virtual camera)<https://stablevirtualcamera.github.io/>
- Gen3C <https://gen3c.github.io/>
- CAT3D <https://cat3d.github.io/>
- WonderWorld <https://wonderworld-2024.github.io/>
- 3DGS 原始 paper <https://repo.sam.labtwin.org/3dgs>
- VGGT <https://vgg-t.github.io/>
- Depth Anything V2 <https://depth-anything-v2.github.io/>
