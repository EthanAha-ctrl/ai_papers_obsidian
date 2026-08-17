---
source_pdf: HDGS.pdf
paper_sha256: b54de130ca84f3f649c7ec754bfd71f6c0f80aea87072e10cf35cad3261ed6ad
processed_at: '2026-08-04T23:32:36-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HDGS 用人话说

## 一句话概括

想象你用一堆小扑克牌拼出一个 3D 场景，每张牌上画着颜色。2DGS 就是这么干的。HDGS 做了四件事：**给每张牌贴一张高清小贴纸**、**每条视线单独排一次序**、**用数学判断哪些牌是多余的可以扔掉**、**把每个像素当成一个漏斗而不是一条线来采样**。

---

## 用扑克牌类比讲清楚

### 场景重构 = 拼扑克牌

2DGS 把 3D 世界表示成几十万张小扑克牌（surfel），每张牌有：
- 位置（在哪）
- 朝向（朝哪边）
- 大小（多大）
- 颜色（SH 参数）
- 透明度

渲染一张图就是：从相机发出射线，看射线撞到哪些牌，按距离从近到远 alpha-blend 起来。

### 痛点 1：贴脸看的时候糊

你靠近一个物体，原本一张扑克牌上只有一个颜色，但物体表面其实有布料的纹路、纸的纤维、书本的文字。**一张牌一个颜色根本装不下这些细节**。要让 2DGS 表达这些，就得堆几十万张牌，贵且慢。

HDGS 的解法：**给每张牌贴一张小贴纸**。大牌子贴大贴纸，小牌子贴小贴纸。牌子只管几何位置，贴纸管外观细节。这就是 per-surfel texture map。

公式上，渲染颜色从原来的 $\mathbf{c}(\mathbf{d})$（一组 SH 系数）变成：

$$
\hat{\mathbf{C}}(\mathbf{x}) = \mathbf{C}[u,v] + \mathbf{SH}(\mathbf{d})
$$

- $\mathbf{C}[u,v]$：从贴纸 bilinear 采样得到的 base color
- $\mathbf{SH}(\mathbf{d})$：view-dependent 高阶项
- $u, v$：射线撞到牌子的局部坐标

牌子大就配大贴纸：$U = \lceil T \cdot s_u \rceil$，$V = \lceil T \cdot s_v \rceil$。$T$ 是密度系数，$s_u, s_v$ 是牌子的尺度。

### 痛点 2：换视角就闪烁

原 2DGS 渲染前会做一次"全局排序"：所有牌按中心点深度排一次，然后整张图所有像素都按这个顺序 blend。问题是**两张倾斜的牌在不同视角下中心点深度的相对顺序会变**。视角一变，blend 顺序变了，画面就 pop 一下。这就是 popping artifact。

HDGS 的解法：**每个像素单独排一次序**。用 k-buffer：每条射线维护一个大小为 k 的有序 buffer，遇到新牌子就 insertion sort 进去。论文用 $k=24$（合成场景）/ $k=16$（真实场景）。

这跟 StopThePop [29] 思路一致。公式上用 order error 量化（Eq. 17）：

$$
\epsilon = \sum_i (z_i - z_{i+1})\cdot\mathbb{1}[z_{i+1} < z_i]
$$

- $z_i$：第 $i$ 张牌的相交深度
- $\mathbb{1}[z_{i+1}<z_i]$：若后一张牌反而更近，indicator = 1

实测 Table 4：2DGS 的 $\epsilon_{test} = 0.2713$，HDGS 是 $0.0021$。**降了两个数量级**，popping 几乎消失。

### 痛点 3：牌太多导致排序慢

per-ray sort 虽然好，但每个像素都要维护 k-buffer，牌越多越慢。能不能扔掉一些牌？

扔掉哪些？**扔掉那些"动一下对画面影响很小的牌"**。这就是 Fisher information pruning。

直觉：一张牌的 position $x_i$ 和 scale $s_i$ 稍微改一下，渲染图变化大不大？变化小说明这张牌"看不清"，剪掉无所谓。数学上就是算 rendering image 对该牌参数的 Jacobian 外积：

$$
U_i = \left\|\log\!\left(\mathrm{diag}\!\left(\sum_\phi \nabla_{x_i, s_i}\mathbf{I}_{\mathcal{G}_i}\,\nabla_{x_i, s_i}\mathbf{I}_{\mathcal{G}_i}^\top\right)\right)\right\|_1
$$

- $\nabla_{x_i, s_i}\mathbf{I}_{\mathcal{G}_i}$：image 对该牌 position+scale 的 Jacobian，shape (像素×3, 5)
- 外积得到 5×5 Fisher sub-matrix
- 取 diag 得 5 个 sensitivity 值
- log 让乘性变加性
- L1 norm 加起来

剪掉 $U_i$ 最小的 10%。

漂亮的地方：**这个 score 只依赖 camera poses，不需要 GT image**。因为推导时假设 model 已 converged，loss residual→0，Hessian 第二项 vanish（Eq. 9→10）。这是个 data-free uncertainty 估计。

Table 8 ablation：剪 50% 都只掉 0.22 dB PSNR。2DGS 默认训练完确实有大量冗余牌。

### 痛点 4：缩远看就 aliasing

你训练时是 800×800，测试时缩到 1/8 = 100×100，画面全是 moiré 条纹和锯齿。为什么？因为每个像素其实代表 3D 空间里的一块体积（frustum），原 2DGS 只用一条 center ray 采样，相当于在那一块体积里 random sample 一个点，高频信号被随机采样就会 aliasing。

3DGS 系的方法是加 post-process filter（Mip-Splatting [42] 分析投影后 Gaussian 协方差做 2D Mip filter）。但 2DGS 的精确投影不再保持 Gaussian 分布，这套方法套不上。

HDGS 的解法：**把每个像素当成一个漏斗，在漏斗里采 5 个点平均**。1 个中心点 + 4 个角点，权重都是 0.2：

$$
\hat{G}(\mathbf{x}) = w_m\,G(\mathbf{x}) + \sum_{i,j \in \{1,-1\}} w_c\,G\!\left(\mathbf{x} + (i\delta_y, j\delta_x)\right)
$$

- $\mathbf{x}$：像素中心
- $\delta_x, \delta_y$：像素的半宽半高
- $w_m = w_c = 0.2$：采样权重

聪明的实现：**不是发 5 条独立 subpixel ray**（那会 5× 计算量），而是只对 center ray 已经击中的那张牌子，额外算 4 个角点对应的 $(u, v)$ 交点，再 query 4 次 Gaussian。ray-surfel intersection 是 closed-form，开销很小。

远距离时 frustum 在 3D 中发散越大，5 个点在 $(u,v)$ 上拉得越开，Gaussian 权重衰减越多，**平均的效果就是低通滤波**。这跟 Mip-NeRF [1] 的 cone casting 是同一个思想，只是搬到 explicit surfel 上。

效果（Table 1，NeRF synthetic 1/8 res）：
- 2DGS: 16.65 dB
- HDGS: 23.98 dB
- **提升 7.33 dB**

---

## 整套 pipeline 的"先后顺序"很关键

HDGS 是 coarse-to-fine 两阶段：

**Stage 1**：原版 2DGS 训练（单色牌），训练完用 Fisher 公式剪 10%。

**Stage 2**：给剩下的牌分配 texture map，用更高分辨率图 fine-tune。**关键：freeze 牌子的位置、朝向、大小**，只优化贴纸。为什么？因为贴纸的索引依赖 $(u,v)$，一旦牌子动了，老的 $(u,v)$ 不再对应老的贴纸像素，就乱套了。这就像你已经画好一张贴纸贴在牌子上，牌子一动贴纸就得重新对位。

总 loss（Eq. 15）：

$$
L = L_c + \lambda_n L_n + \lambda_d L_d
$$

- $L_c$：appearance（L1 + SSIM）
- $L_n$：normal consistency（Eq. 7）
- $L_d$：depth distortion（Eq. 6）

---

## 用 Karpathy 风格的 intuition 总结

我看完这篇 paper 的第一反应是：**这四个模块其实对应 rendering pipeline 的四个独立 bottleneck，作者一个一个打补丁**：

| Bottleneck | 现象 | HDGS 模块 | 数学工具 |
|---|---|---|---|
| 容量不足 | close-up 糊 | per-surfel texture | bilinear sampling |
| 视角一致性 | popping | per-ray k-buffer sort | order error ε |
| 参数冗余 | sort 慢 | Fisher pruning | Fisher information |
| 采样不足 | 缩远 aliasing | frustum 5-sample | Nyquist / cone casting |

四件事互相支撑：
- texture 让单牌能装高频 → 但 texture 对 view 敏感会 popping → 必须 sort
- sort 让每像素排序代价高 → 必须 prune 降牌数
- prune 用 Fisher → 只动 geometry 不动 appearance，texture 不受影响
- frustum 解决缩远，与前三件正交

这四件事**单独做哪个都不够**。GSTex [30] 只做 texture 不做 sort，复杂场景就崩。StopThePop [29] 只做 sort 不做 texture，appearance 拟合掉点。Mip-Splatting [42] 只做 anti-aliasing 不做 texture，close-up 仍糊。HDGS 把四件事打包才取得 SOTA。

更深层 intuition：这四件事其实对应**信号处理 + 概率推断 + 数值积分**三个数学领域的经典工具：
- Frustum sampling = numerical integration（Simpson's rule / midpoint rule）
- Per-ray sort = order statistics
- Fisher pruning = Laplace approximation / posterior sensitivity
- Texture map = adaptive basis expansion

作者没发明任何新数学，全是在 2DGS 框架上把经典工具拼接起来。这是好的 engineering taste — 用对的工具解对应的问题，而不是用一个 fancy framework 硬套。

---

## 联想到的其他领域

### 与 deep learning training 的类比

HDGS 的 two-stage pipeline 跟 LLM training 异曲同工：
- Stage 1 = pretrain（学 geometry / 单色）
- Stage 2 = SFT（学 appearance / texture）
- Fisher pruning = pruning pretrain model（如 Movement Pruning, SparseGPT）
- Freeze geometry in Stage 2 = freeze backbone in adapter tuning

这反映了一个普遍原则：**当优化目标有强耦合时，分阶段解耦比 joint optimize 更稳**。

参考：
- SparseGPT: https://arxiv.org/abs/2301.00774
- LoRA: https://arxiv.org/abs/2106.09685

### 与 photography 的类比

Frustum sampling 跟数码相机的 demosaicing 是同源问题 — 都是"每个像素接收一片光，如何积分"。手机摄影的 anti-aliasing filter（OLPF）就是在 sensor 前加一层低通光学滤波器，HDGS 是在算法层做同样的事。

参考：
- Anti-aliasing filter in camera: https://en.wikipedia.org/wiki/Anti-aliasing_filter

### 与点云压缩的类比

Fisher pruning 跟点云里的"基于 geometric uncertainty 的 downsample"思路一致。PCL 库里有 StatisticalOutlierRemoval，核心也是基于邻域分布判断点的"可信度"。HDGS 用 Fisher 信息把这套搬到可微渲染里。

参考：
- PCL Statistical Outlier Removal: https://pointclouds.org/documentation/group__sample__consensus.html

### 与 sparse voxel rendering 的类比

per-surfel texture 思路其实接近 sparse voxel octree 的 brick — 每个 brick 存一个 small 3D texture。HDGS 是 2D 版本：每张 surfel 存 2D texture。NVIDIA 的 NVSG、GigaVoxels 是这个 family 的早期工作。

参考：
- GigaVoxels: https://maverick.inria.fr/Publications/2009/CNLE09/
- Sparse Voxel Octree: https://research.nvidia.com/publication/2010-02_Sparse-Voxel-Octree

---

## 我个人觉得 elegant 与不 elegant 的地方

**Elegant**：
- Fisher pruning 只用 camera pose 不用 GT image — 数学上漂亮
- Texture 大小随 surfel scale 自适应 — 自然 resolution-adaptive
- Frustum 5 sample 复用 center ray intersection — 开销小
- Two-stage freeze geometry — 避免 texture indexing 错位

**不 elegant**：
- $T$ 这个 texture density 超参需要 per-dataset 调（1000/400/200/4000）— 暗示 texture 分配机制还不够自适应
- $k=24$ 硬选 — 没有理论指导
- $\delta_x, \delta_y$ 假设 square pixel — 对 anisotropic pixel footprint 不友好
- Stage 2 必须 freeze geometry — 限制场景动态性

如果让我改进，我会想做：
1. **Texture resolution 用 Fisher 自动决定** — 高 Fisher 的牌分配大 texture
2. **Texture map 用 multi-plane 表达 view-dependent** — 取代 SH 高阶项
3. **Frustum sampling 用 importance sampling** — 远距离时多采样几个点
4. **Stage 2 geometry 不 freeze 而是 LoRA-style low-rank update** — 允许小幅 geometry 调整但不动主结构

---

## 几个值得读的延伸

| 主题 | Paper | Link |
|---|---|---|
| 2DGS 基础 | 2D Gaussian Splatting (Huang et al. SIGGRAPH 2024) | https://github.com/hbb1/2d-gaussian-splatting |
| Popping 解决 | StopThePop (Radl et al. 2024) | https://github.com/lRaichu/StopThePop |
| GS anti-aliasing | Mip-Splatting (Yu et al. CVPR 2024) | https://github.com/autonomousvision/mip-splatting |
| Per-primitive texture | GSTex (Rong et al. 2024) | https://arxiv.org/abs/2409.12954 |
| Fisher pruning | PUP 3D-GS (Hanson et al. 2024) | https://arxiv.org/abs/2406.10219 |
| Fisher active learning | FisherRF (Jiang et al. 2023) | https://arxiv.org/abs/2311.17874 |
| NeRF anti-aliasing | Mip-NeRF 360 (Barron et al. CVPR 2022) | https://jonbarron.info/mipnerf360/ |
| EWA splatting | Zwicker et al. 2001 | https://www.cs.umd.edu/~zwicker/publications/EWAVolSplatting.pdf |
| Light field 渲染 | Levoy & Hanrahan 1996 | https://graphics.stanford.edu/papers/light/ |

---

总结一句人话：**HDGS 是 2DGS 的"高配版"，给每张牌贴贴纸解决细节、给每条视线单独排序解决闪烁、用 Fisher 信息扔冗余牌解决速度、用漏斗采样解决缩远锯齿。四件事各管一摊，合起来就 SOTA。**

---

# HDGS: Textured 2D Gaussian Splatting for Enhanced Scene Rendering 深度解析

## 一、Paper 整体定位与核心 contribution

HDGS 由 UPenn 的 Kostas Daniilidis 团队提出（含 Lingjie Liu），针对 2D Gaussian Splatting (2DGS) [14] 在 **arbitrary viewpoint / arbitrary resolution** 渲染时的两大痛点：**close-up 渲染细节不足** 与 **远距离/低分辨率 aliasing**，提出了一套**包含四个互补模块**的 coarse-to-fine pipeline。

四个核心模块：
1. **Frustum-based Sampling**：把每个 pixel 看作一个 light frustum 而不是一条 ray，通过 5 个 ray-surfel intersection 取平均，作为 anti-aliasing 的低通滤波。
2. **Per-Ray Sorting (k-buffer)**：替代 2DGS 的 per-view global sort，消除 popping artifacts。
3. **Fisher Information-based Pruning**：基于 Fisher 信息矩阵的对角元 L1 norm 量化每个 surfel 的几何 uncertainty，剪掉 sensitivity 低的 surfel。
4. **Per-Surfel Texture Map**：给每个 surfel 配一张可优化的小 texture map，把 appearance 和 geometry 解耦。

总体 motivation：在 2DGS 里，**geometry 和 appearance 是纠缠的**，要表达高频 appearance 就需要堆 primitive，但堆多了 popping 严重，远距离时 aliasing 也严重。HDGS 通过"用少量大 surfel 承载 geometry + 每个 surfel 贴一张 texture 表达 appearance"来打破这个 trade-off。

参考链接：
- 2DGS: https://github.com/hbb1/2d-gaussian-splatting
- StopThePop: https://github.com/lRaichu/StopThePop
- Mip-Splatting: https://github.com/autonomousvision/mip-splatting
- 3DGS: https://github.com/graphdeco-inria/gaussian-splatting
- GSTex: https://arxiv.org/abs/2409.12954
- PUP 3D-GS: https://arxiv.org/abs/2406.10219
- Zip-NeRF: https://jonbarron.info/zipnerf/
- Mip-NeRF 360: https://jonbarron.info/mipnerf360/
- NeRF: https://www.matthewtancik.com/nerf

---

## 二、Preliminaries 复习：2DGS 的精确投影

理解 HDGS 必须先吃透 2DGS 的 ray-surfel 精确相交。

### 2D Gaussian Surfel 参数化

每个 surfel 由：
- 两个 tangent vector $\mathbf{t}_u, \mathbf{t}_v$（切向主轴）
- 两个 scaling $s_u, s_v$
- center $\mathbf{p}$
- opacity $\alpha$
- SH color参数 $\mathbf{c}$
- 法向 $\mathbf{n} = \pm(\mathbf{t}_u \times \mathbf{t}_v)$

定义 surfel-to-world 变换矩阵 $\mathbf{H}$（Eq. 2）：

$$
\mathbf{H} = \begin{bmatrix} s_u\mathbf{t}_u & s_v\mathbf{t}_v & \mathbf{0} & \mathbf{p} \\ \mathbf{0} & \mathbf{0} & 0 & 1 \end{bmatrix}
$$

- 上标/下标含义：$s_u$ 表示沿 $\mathbf{t}_u$ 方向的尺度，列向量 $\mathbf{p}$ 是平移部分。
- 注意第三列是 $\mathbf{0}$ — surfel 是 2D（没有沿法向的 extent）。

### 精确 ray-surfel intersection (Eq. 1)

$$
\mathbf{x} = (xz, yz, z, z)^\top = \mathbf{P}\mathbf{H}(u, v, 1, 1)^\top
$$

变量含义：
- $\mathbf{x}$：homogeneous pixel coordinate，前 3 项是 world ray 上的交点（除以 z 即得 NDC 坐标 $(x,y)$）
- $u, v$：surfel local coordinate（沿 $\mathbf{t}_u, \mathbf{t}_v$ 的有符号距离）
- $\mathbf{P}$：相机投影矩阵 $4\times 4$
- $\mathbf{H}$：surfel-to-world $4\times 4$
- $(u,v,1,1)^\top$：surfel local point 的 homogeneous 表示

这意味着 2DGS **不**像 3DGS 那样把 center 投到 image 后用 local Gaussian covariance 近似 — 它**显式求解 ray-surfel 相交的 $(u,v)$**，再查询 normalized Gaussian：

$$
G(\mathbf{x}) = \exp\!\left(-\frac{u^2+v^2}{2}\right)
$$

### Alpha-blending accumulation (Eq. 3-5)

$$
\mathbf{C}(\mathbf{x}) = \sum_i \mathbf{c}_i(\mathbf{d})\,\alpha_i\,G_i(\mathbf{x})\,T_i
$$

$$
D(\mathbf{x}) = \sum_i z_i(\mathbf{x})\,\alpha_i\,G_i(\mathbf{x})\,T_i
$$

$$
\mathbf{N}(\mathbf{x}) = \sum_i \mathbf{n}_i\,\alpha_i\,G_i(\mathbf{x})\,T_i
$$

其中 $T_i = \prod_{j<i}(1-\alpha_j G_j)$ 是 transmittance。

### 几何正则 (Eq. 6-7)

$$
L_d = \sum_{i,j}\alpha_i G_i T_i \alpha_j G_j T_j (z_i - z_j)^2
$$

$$
L_n = \sum_i \alpha_i G_i T_i (1 - \mathbf{n}_i \mathbf{N}_d)
$$

- $L_d$：depth distortion loss，鼓励所有相交 surfel 沿 ray 集中在同一深度（避免多个 surfel 互相重叠）。
- $L_n$：normal consistency，$\mathbf{N}_d$ 是由 depth map gradient 反算的 normal，把每个 surfel 的 normal 拉到与 depth-derived normal 一致。

这是 2DGS 拿到 sharp surface 的关键。

---

## 三、Module 1: Frustum-based Sampling

### 3.1 Motivation 与 Nyquist 视角

Aliasing 的本质是 **采样频率 < 2× 信号频率**。在 rendering 中：
- 采样频率 = pixel resolution（pixel 越密，采样频率越高）
- 信号频率 = object 投影到 image plane 后的 appearance 复杂度

远距离的物体由于 perspective projection，**pixel footprint 在 3D 中覆盖的体积更大**，但 appearance 细节本身没变。如果还是用单 ray（pixel center）采样，相当于做了一个 point sample，远距离时这个 point sample 会随机落在高频 appearance 上，产生 moiré/jaggies。

3DGS 的 anti-aliasing 思路（Mip-Splatting [42], Multi-Scale GS [40]）：分析 projection 后的 Gaussian 协方差，加 post-process filter。但 2DGS 的 projection **不再保持 Gaussian 分布**（因为精确相交导致非线性），这套方法不能直接套用。

### 3.2 Frustum 5-sample 设计 (Eq. 11)

$$
\hat{G}(\mathbf{x}) = w_m\,G(\mathbf{x}) + \sum_{i,j \in \{1,-1\}} w_c\,G\!\left(\mathbf{x} + (i\delta_y, j\delta_x)\right)
$$

变量含义：
- $\mathbf{x}$：pixel center
- $\delta_x, \delta_y$：pixel 的 half-width / half-height
- $(i,j) \in \{(-1,-1),(-1,1),(1,-1),(1,1)\}$：4 个 corner offset
- $w_m$：center 权重（论文用 0.2）
- $w_c$：corner 权重（论文用 0.2）

关键设计 trick：**这 5 个 sample 不是 5 条独立 subpixel ray**（那样会 5× 计算量），而是只对**已经被 center ray 击中的同一个 surfel** 计算 4 个 corner ray 的 intersection $(u,v)$，再 query 5 次 Gaussian。由于 ray-surfel intersection 是 closed-form，开销很小。

### 3.3 为什么这等价于低通滤波

参考 Mip-NeRF 的 cone casting 思想：远距离时，pixel footprint 对应的 frustum 在 3D 中发散得越大。对一个 surfel 而言，corner sample 在 $(u,v)$ 坐标里离 center 越远，$G$ 值越小。**平均 5 个 sample 的效果是把高频信号"抹平"**。

数学上：对一个 $h\times h$ 的 pixel footprint，integrate Gaussian：

$$
\int_{-h/2}^{h/2}\int_{-h/2}^{h/2} f(x,y)\,dxdy
$$

这个 integral 本身就是一个低通滤波（Gaussian 当 window）。

### 3.4 Supplementary 里的误差分析（Section 7）

论文 supplement 推导了不同 sampling 策略的 Taylor 展开误差阶：

**4 corner point**（Eq. 20，每点权重 $1/4$）：
$$
\frac{h^2}{4}\sum_{\text{corners}} f = \iint_P f\,dxdy + \frac{h^4 f''_{xx}(0,0)}{12} + \frac{h^4 f''_{yy}(0,0)}{12} + O(h^6)
$$
误差是 $O(h^4)$。

**Center + 4 corner**（Eq. 22，center 权重 $2/3$，corner 权重 $1/12$）：
$$
\frac{h^2}{12}\left(8f(0,0) + \sum_{\text{corners}} f\right) = \iint_P f\,dxdy + O(h^6)
$$
理论上误差 $O(h^6)$，更精确。

但 Table 9 显示：在 **full resolution** 上 vanilla average（论文 main paper 用的）反而最好（PSNR 33.46 vs Center 33.26）；只有在 reduced resolution 时 Center 更优。论文解释为"理论误差不总是 hold"。

Intuition：5-point stencil 等价于 2D Simpson's rule，需要 function 在 footprint 内 smooth。但 Gaussian 在 cutoff $r=4.5$ 处 hard truncate（实际 GPU 实现会 cutoff），corner 处不连续，导致高阶 stencil 不一定赢。

---

## 四、Module 2: Per-Ray Sorting

### 4.1 Popping artifact 的根源

3DGS / 2DGS rasterizer 的排序逻辑：
1. 对每个 view，所有 primitive 按 **center depth** 做一次 global radix sort
2. alpha blending 时，所有 pixel ray 都用同一个 sort 顺序遍历 primitive

问题：**同一个 3D 区域在不同 view 下的 center depth 排序可能不同**。例如两个 surfel 在 view A 中 surfel-1 的 center 更近，但在 view B 中 surfel-2 的 center 更近（因为两个 surfel 倾斜不同）。alpha blending 是顺序敏感的，order 一变就 popping。

Figure 2 右侧举例：surfel $p_2$ 的 center 比较靠近屏幕，但实际相交深度 $z_1 < z_2$。global sort 会用 center depth 给错顺序，per-ray sort 用 $z_1, z_2$ 给对顺序。

### 4.2 实现：k-buffer + per-tile

HDGS 借用 StopThePop [29] 的思路：
- **Per-tile sort**：把 image 切成 8×8 tile，每个 tile 用 tile-center 像素的 depth 做 radix sort。这是 GPU-friendly 的近似。
- **Per-ray k-buffer**：每条 ray 维护一个 size-k 的 sorted buffer，遇到新 surfel 时做 insertion sort。论文用 $k=24$（synthetic）/ $k=16$（real）。

k-buffer 的内存代价：$H\times W\times k\times (\text{per-surfel data})$。对 800×800×24×(float depth + primitive_id)，大约 60MB，L40 可以接受。

### 4.3 Sorting Error 量化 (Eq. 17)

StopThePop [29] 提出的 per-ray depth order error：

$$
\epsilon = \sum_i (z_i - z_{i+1})\cdot\mathbb{1}[z_{i+1} < z_i]
$$

变量：
- $z_i$：第 $i$ 个 surfel 在 ray 上的 blending depth
- $\mathbb{1}[z_{i+1}<z_i]$：indicator，若前后顺序颠倒则为 1

Table 4 实测数据：

| Method | Train ε | Test ε |
|---|---|---|
| 2DGS | 0.2392 | 0.2713 |
| Ours w/o prune | 0.0023 | 0.0024 |
| Ours w/ prune | 0.0021 | 0.0021 |

**Popping 几乎消除了两个量级**。注意 prune 还进一步降低了 error，因为更少 surfel = 更少 order 冲突机会。

---

## 五、Module 3: Fisher Information-based Pruning

### 5.1 Fisher Information 直觉

Pruning 的核心问题：**剪掉谁？**

不剪：留着会拖慢 per-ray sort，且增加 popping 概率。
乱剪：可能剪掉高频 detail surfel。

Fisher 信息提供了一个有原则的判据。给定 L2 loss：

$$
L_2 = \frac{1}{2}\sum_{\phi \in P_{gt}} \|\mathbf{I}_\mathcal{G}(\phi) - \mathbf{I}_{gt}\|_2^2
$$

变量：
- $\phi$：camera pose
- $P_{gt}$：所有输入 camera poses
- $\mathbf{I}_\mathcal{G}(\phi)$：用 Gaussian scene $\mathcal{G}$ 渲染的 image
- $\mathbf{I}_{gt}$：ground truth image

Hessian（Eq. 9）：

$$
\nabla_\mathcal{G}^2 L_2 = \sum_\phi \nabla_\mathcal{G}\mathbf{I}_\mathcal{G}(\phi)\nabla_\mathcal{G}\mathbf{I}_\mathcal{G}(\phi)^\top + (\mathbf{I}_\mathcal{G}(\phi)-\mathbf{I}_{gt})\nabla_\mathcal{G}^2\mathbf{I}_\mathcal{G}(\phi)
$$

- 第一项：Jacobian 外积（gradient of rendered image w.r.t. Gaussian parameters）
- 第二项：residual × Hessian

**当 model converged 时 residual→0，第二项消失**，剩 Fisher 近似（Eq. 10）：

$$
\nabla_\mathcal{G}^2 L_2 \approx \sum_\phi \nabla_\mathcal{G}\mathbf{I}_\mathcal{G}\nabla_\mathcal{G}\mathbf{I}_\mathcal{G}^\top
$$

直觉：Fisher 信息衡量"参数变化能引起多大 render 变化"。**Render 对这个 surfel 不敏感 = Fisher 小 = 可剪**。

### 5.2 Per-surfel Sensitivity Score (Eq. 12)

$$
U_i = \left\|\log\!\left(\mathrm{diag}\!\left(\sum_{\phi \in P_{gt}} \nabla_{x_i, s_i}\mathbf{I}_{\mathcal{G}_i}\,\nabla_{x_i, s_i}\mathbf{I}_{\mathcal{G}_i}^\top\right)\right)\right\|_1
$$

变量含义：
- $x_i$：surfel $i$ 的 position 参数（3D 坐标）
- $s_i$：surfel $i$ 的 scale 参数（$s_u, s_v$）
- $\mathbf{I}_{\mathcal{G}_i}$：surfel $i$ 对 rendered image 的贡献
- $\nabla_{x_i, s_i}\mathbf{I}_{\mathcal{G}_i}$：Jacobian，shape 是 (像素数×3, 5)（位置3 + scale2）
- $\nabla\mathbf{I}\nabla\mathbf{I}^\top$：5×5 的 Fisher sub-matrix
- $\mathrm{diag}(\cdot)$：取对角元（5 个）
- $\log(\cdot)$：element-wise log（让数值范围可比较，同时把"乘性 sensitivity"变"加性"）
- $\|\cdot\|_1$：L1 norm，把 5 个对角元加起来

只考虑 $x_i, s_i$（位置 + scale），**不考虑 rotation、opacity、SH**，因为 position/scale 是几何 uncertainty 的主要来源（多视角的 triangulation uncertainty 主要在 position 上）。

Pruning 策略：剪掉 $U_i$ 最小的 10% surfel。

### 5.3 Pruning 比例 ablation (Table 8)

| Pruned % | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 10% (Ours) | 33.46 | 0.968 | 0.030 |
| 20% | 33.44 | 0.968 | 0.030 |
| 30% | 33.44 | 0.968 | 0.030 |
| 40% | 33.38 | 0.967 | 0.031 |
| 50% | 33.24 | 0.967 | 0.032 |

剪 50% 都还能 33.24 PSNR，几乎不掉点。这反映了 2DGS 默认有大量 redundant surfel。

---

## 六、Module 4: Per-Surfel Texture Map

### 6.1 设计动机

per-ray sort 解决 popping 但牺牲了 fitting 性能（StopThePop [29] 已经观察到这个 trade-off）。原因：sort 约束使优化空间变窄，加上 adaptive density control 倾向于 prune 高频低 opacity 的 surfel，导致 appearance 拟合不充分。

解决思路：**给每个 surfel 一张小 texture map**，把"几何 = surfel 位置/朝向/scale"与"appearance = texture map 像素值"解耦。

### 6.2 Texture Map 参数化

给定 surfel 的 scaling $(s_u, s_v)$，分配 texture map $\mathbf{C}[u,v]$ 大小 $(U, V, c)$：

$$
U = \lceil T \cdot s_u \rceil, \quad V = \lceil T \cdot s_v \rceil
$$

变量：
- $T$：texture density hyperparameter
  - NeRF synthetic: $T=1000$
  - Mip-NeRF 360 indoor: $T=400$
  - Mip-NeRF 360 outdoor: $T=200$
  - texture-rich dataset: $T=4000$
- $c$：color channel 数
- $U, V$：texture map 的长宽

Intuition：**texture 分辨率与 surfel 大小成正比**。大 surfel 配大 texture（捕获更多 appearance detail），小 surfel 配小 texture（省内存）。这是非常聪明的"resolution-adaptive"设计。

### 6.3 索引与采样

Ray-surfel intersection 得到 surfel local coord $(u, v)$，cutoff range $r=4.5$。索引坐标：

$$
\left(\frac{u+r}{2r}U,\ \frac{v+r}{2r}V\right)
$$

变量：
- $u, v \in [-r, r]$：surfel local coordinate（Gaussian 的 ±4.5σ 范围）
- $\frac{u+r}{2r}$：归一化到 $[0, 1]$
- 乘以 $U, V$：得到 texture map 像素坐标

**Bilinear sampling** texture map 得到 0-th SH component（即 base color）。

### 6.4 View Direction 精确化

这是个 subtle 但重要的点。原 2DGS 用：

$$
\mathbf{d}_{approx} = \mathbf{p} - \mathbf{O}
$$

- $\mathbf{p}$：surfel center
- $\mathbf{O}$：camera center

但 HDGS 用：

$$
\mathbf{d}_{precise} = \mathbf{H}(u,v,1)^\top - \mathbf{O}
$$

即"camera 到**精确相交点**"的方向，而不是"camera 到 surfel center"。

为什么重要？对于 large surfel，相交点可能离 center 很远，view direction 差异会让 SH 高阶项算错，导致 specular highlight 偏移。

### 6.5 渲染方程 (Eq. 13-14)

$$
\hat{\mathbf{C}}(\mathbf{x}) = \mathbf{C}[u,v] + \mathbf{SH}(\mathbf{d})
$$

$$
\mathbf{C}(\mathbf{x}) = \sum_i \hat{\mathbf{C}}(\mathbf{x})_i\,\alpha_i\,\hat{G}_i(\mathbf{x})\,T_i
$$

变量：
- $\mathbf{C}[u,v]$：从 texture map bilinear sample 得到的 base color
- $\mathbf{SH}(\mathbf{d})$：view-dependent high-order SH
- $\hat{G}_i(\mathbf{x})$：frustum-averaged Gaussian density (Eq. 11)
- $T_i$：transmittance
- 下标 $i$：按 per-ray sorted order

注意 $\hat{G}$ 用了 frustum sampling，$\hat{\mathbf{C}}$ 用了 texture map + SH，叠加效应。

---

## 七、Two-Stage Optimization Pipeline

### Stage 1: 单色 primitive + Fisher pruning

直接用原 2DGS 训练（单色 surfel），训练完毕后做 Fisher pruning 剪 10%。

### Stage 2: 分配 texture map + fine-tune

- 给每个剩余 surfel 分配 texture map
- 用 on-demand **higher-resolution images** fine-tune
- **Freeze gradient** of Gaussian centers, rotations, scales — 因为 high-frequency rendering 对 geometry 修改敏感，一旦动 geometry，texture indexing 就错位。

Loss（Eq. 15）：

$$
L = L_c + \lambda_n L_n + \lambda_d L_d
$$

- $L_c$：appearance reconstruction loss（L1 + SSIM 组合）
- $L_n, L_d$：geometric regularization（Eq. 6, 7）

为什么 freeze geometry？因为 texture sampling 是"以 $(u,v)$ 为索引"，一旦 surfel 位置/方向变了，老的 $(u,v)$ 就不再对应老的 texture 像素，会冲掉已学到的 detail。这其实是个相当 practical 的"freeze then fine-tune texture only"思路，类似 deferred shading 的精神。

---

## 八、Experiment 结果深度分析

### 8.1 Reduced Resolution Rendering (Table 1)

最显著的结果。NeRF synthetic 上的 4 个分辨率：

| Method | Full | 1/2 | 1/4 | 1/8 | Avg |
|---|---|---|---|---|---|
| 3DGS | 33.33 | 26.95 | 21.38 | 17.69 | 24.84 |
| MipSplatting | 33.36 | 34.00 | 31.85 | 28.67 | 31.97 |
| 2DGS | 32.67 | 27.19 | 20.57 | 16.65 | 24.27 |
| **Ours** | **33.46** | 32.16 | 28.18 | 23.98 | 29.45 |

关键观察：
- **Full res**：HDGS 略胜 2DGS（33.46 vs 32.67），不及 MipSplatting（33.36），但远超 3DGS。
- **1/8 res**：HDGS 比 2DGS 高 7.33 dB（23.98 vs 16.65），差距巨大！这是 frustum sampling 的功劳。
- 但 HDGS 在 1/2 / 1/4 仍不及 MipSplatting，因为 MipSplatting 用了 2D Mip filter（理论上更优），HDGS 是 5-sample 离散近似。

### 8.2 与 StopThePop / GSTex 对比 (Table 2, 12)

NeRF synthetic 上各方法 PSNR：

| Method | PSNR |
|---|---|
| 3DGS | 33.33 |
| StopThePop | 33.57 |
| PGSR | 31.87 |
| MipSplatting | 33.36 |
| 2DGS | 32.67 |
| GSTex | 33.25 |
| **Ours** | 33.46 |

HDGS 介于 3DGS 和 StopThePop 之间。但注意 StopThePop 是 3DGS-based（不优化几何），HDGS 是 2DGS-based（同时优化几何）。看 LPIPS：GSTex 0.024（最好）, StopThePop/Ours 0.030, 2DGS 0.035。GSTex [30] 在 LPIPS 上更优，但 HDGS 在 PSNR / 几何上更平衡。

### 8.3 Texture-rich Dataset (Table 3)

这是论文自建的高频细节数据集：

| Method | PSNR | SSIM | LPIPS | $\|HF\|$ |
|---|---|---|---|---|
| 3DGS | 22.40 | 0.602 | 0.461 | 0.304 |
| 2DGS | 22.11 | 0.627 | 0.460 | 0.342 |
| **Ours** | 22.72 | 0.607 | 0.350 | 0.434 |
| Ours w/o texture | 22.58 | 0.624 | 0.455 | 0.345 |

**$\|HF\|$** 是论文新提出的高频保真度度量（Eq. 16）：

$$
\|HF\| = \frac{DCT_{AC}(\hat{\mathbf{I}}) \cdot DCT_{AC}(\mathbf{I})}{\|DCT_{AC}(\hat{\mathbf{I}})\|_2\,\|DCT_{AC}(\mathbf{I})\|_2}
$$

变量：
- $DCT_{AC}$：Discrete Cosine Transform 的 AC components（即去掉 DC/直流分量后所有频率分量）
- $\hat{\mathbf{I}}, \mathbf{I}$：rendered / ground truth image
- 分子：cosine similarity
- 分母：L2 norm 归一化

这本质上是 **rendered image 与 GT 在频域 cosine 相似度**（剔除亮度均值）。HDGS 0.434 vs 2DGS 0.342 — texture map 把高频保留得显著更好。

**SSIM 反而略降**（0.607 vs 0.627），因为 SSIM 对 structure 敏感，而 texture map 改变了"表面 look"导致结构特征略变。

### 8.4 Geometry Quality (Table 5, 11)

DTU 上的 Chamfer Distance：

| Method | Mean CD |
|---|---|
| 2DGS | 0.74 |
| **Ours** | 0.75 |

几乎相当（差 0.01 mm）。HDGS 没有在几何上输给 2DGS，证明 Fisher pruning + texture disentangle 不伤几何。Per-scene Table 11 显示个别场景 HDGS 反而更好（如 scan 65: 0.91 vs 0.86）。

### 8.5 Ablation (Table 6, 13)

NeRF synthetic 上各 ablation 的 PSNR：

| Variant | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Full | 33.46 | 0.968 | 0.030 |
| w/o texture | 33.28 | 0.968 | 0.032 |
| w/o prune | 33.40 | 0.967 | 0.031 |
| **w/o sorting** | **31.99** | **0.955** | **0.048** |
| w/o frustum | 33.27 | 0.967 | 0.031 |

最关键的是 **w/o sorting 掉 1.47 dB**！这证明 per-ray sort 是整个 pipeline 的基石。没有 sort，per-surfel texture 在 novel view 上一变 view 就乱跳（popping 严重）。

w/o texture 只掉 0.18 dB，但 Table 7 在 texture-rich 数据集上掉更多（22.72 → 22.58，$\|HF\|$ 从 0.434 → 0.345），说明 **texture 在高频场景才显著有效**。

w/o prune 几乎不掉（甚至略升 33.40 vs 33.46），但 prune 的好处在 efficiency（少 surfel = 快 sort）。

### 8.6 Primitive 数量

论文报告：
- NeRF synthetic: HDGS $1.3\times 10^5$ vs 2DGS $1.4\times 10^5$ vs MipSplatting $3.0\times 10^5$
- Mip-NeRF 360: HDGS $2.6\times 10^6$ vs 2DGS $2.2\times 10^6$ vs MipSplatting $4.2\times 10^6$

HDGS 比 2DGS 多一点点 primitive（Mip-NeRF 360 上多 18%），但远少于 MipSplatting。这是 texture map 带来的额外参数。

---

## 九、与相关工作的联系

### 9.1 Mip-NeRF / Zip-NeRF 的 lineage

HDGS 的 frustum sampling **直接灵感来自 Mip-NeRF [1] 的 cone casting**。Mip-NeRF 把每个 pixel 视作 cone，integrate cone 内的 radiance。Zip-NeRF [3] 进一步用 multi-sampling 在 hash grid 上实现 anti-aliasing。Rip-NeRF [22] 用 Ripmap-encoded Platonic solid 实现 anti-aliasing。

HDGS 是把这套思想移植到 2D Gaussian Splatting，因为 2DGS 的精确投影不能用 Mip-Splatting 那种 covariance-based post-process filter，所以必须从 sampling 端解决。

### 9.2 与 GSTex [30] 的对比

GSTex 也是 2DGS + per-primitive texture，但 GSTex：
- 不解决 aliasing（HDGS 加 frustum）
- 不做 Fisher pruning（HDGS 做）
- per-view sort（HDGS 用 per-ray sort）

所以 GSTex 在简单场景能跑（LPIPS 0.024），但在复杂场景 + novel view 下 popping 严重。HDGS 是 GSTex 的"鲁棒化版本"。

### 9.3 与 Texture-GS [38] 的对比

Texture-GS [38] 假设 Gaussians 在拓扑上类似球面，对 spherical topology 场景 OK，对复杂场景失败。HDGS 利用了 2DGS 的 surfel 性质，对 arbitrary topology 都能 work。

### 9.4 与 Sort-Free Blending [12] 的对比

SortFreeGS [12] 提出用 weighted sum rendering 取代 alpha blending，避免 sort。理论上 elegant，但 weighted sum 不是体渲染严格形式，可能 loss 物理意义。HDGS 选择更保守的 per-ray sort，保留 alpha blending。

### 9.5 与 Light Field / Lumigraph 的联系

HDGS 的"大 surfel + per-surfel texture"在思想上接近 **Lumigraph / Light Field Rendering**（Levoy & Hanrahan 1996, Gortler et al. 1996）— 用 2D patch + 每 patch 一组方向 color 表达光场。HDGS 可以看作可微分的 light field。

参考：
- Lumigraph: https://www.cs.princeton.edu/~smr/papers/lumigraph/lumigraph.pdf
- Light Field Rendering: https://graphics.stanford.edu/papers/light/

### 9.6 与 Point-Based Rendering 的联系

HDGS 接续了 point-based rendering (PBR) 的传统 — surfel 就是 point splat 的演化版。Zwicker et al. 2001 的 EWA volume splatting 是这个 lineage 的奠基工作，HDGS 的 frustum sampling 本质上是 EWA filter 的 2D 简化版。

参考：
- EWA Splatting: https://www.cs.umd.edu/~zwicker/publications/EWAVolSplatting.pdf
- Surfel Rendering (Pfister et al. 2000): https://lig.espci.fr/~biblio/labo/ENSG/SIGGRAPH2000/course39/surfels.pdf

---

## 十、Limitations 与开放问题

论文承认：**per-ray sort + texture indexing 的计算开销高于 baseline**，scalability 受限。

我额外想到的：
1. **Texture map 的初始化**：论文用 background color 初始化。如果初始化不好，Stage 2 fine-tune 慢。可以考虑预填充 Stage 1 学到的 SH 0-th coeff 作为 initialization。
2. **Texture map 与 view-dependent SH 的耦合**：HDGS 把 texture map 作为 base color，SH 作为高阶项。但 texture map 也可以做成 view-dependent（如 multi-plane texture），表达更复杂的 BRDF。GaussianShader [18] 走的就是这条路。
3. **Frustum sampling 的尺度自适应**：现在固定 $w_m = w_c = 0.2$。可以考虑 view-distance-adaptive 权重（远距离时 corner 权重更高，相当于更强低通）。
4. **Per-ray k-buffer 的 k 选择**：$k=24$ 对场景复杂度敏感。对 Mip-NeRF 360 outdoor（复杂背景）可能需要更大 k，否则会深度排序冲突。
5. **与 Mip-Splatting 的集成**：HDGS 的 frustum sampling 是离散 5-sample，Mip-Splatting 的 2D Mip filter 是解析的。把 Mip-Splatting 思想搬到 2DGS（解决精确投影后的非 Gaussian 分布问题）是个 open problem。
6. **3DGS 上的扩展**：HDGS 完全基于 2DGS。在 3DGS 上能否做 per-Gaussian texture？Texture-GS [38] 尝试过但假设球形拓扑。General 3DGS texture 是 open problem（可能需要 mesh extraction + UV unwrap）。

---

## 十一、对我（Karpathy 视角）的 Intuition 总结

回到 first principles 看 HDGS：

**Rendering = sampling + accumulation**。
- Sampling 端：HDGS 用 frustum（5 sample）做 anti-aliasing，本质是 Nyquist-aware 增加采样率，但巧妙地复用 surfel-level intersection，开销小。
- Accumulation 端：HDGS 用 per-ray sort 解决 view inconsistency，用 texture map 把 appearance capacity 从"primitive 数量"解耦到"per-primitive texture size"，用 Fisher pruning 控制总参数量。

这套设计的精髓在于 **coarse-to-fine**：
1. 先用 2DGS 训练 geometry（coarse）
2. 用 Fisher 信息 identify 几何冗余 surfel，prune
3. 给保留的 surfel 分配 texture capacity（fine）
4. Fine-tune 时 freeze geometry，只优化 appearance

这跟 LLM 的 pretrain → SFT → RLHF 阶段化思路异曲同工 — 不同阶段做不同事，避免同时优化多个耦合目标导致互相干扰。

最让我惊艳的是 **Fisher pruning 的 elegance**：它利用了"converged model 的 Hessian 第二项 vanish"这个数学事实，把"参数 uncertainty 估计"转化为"rendering Jacobian 的外积"，完全不需要 ground truth image — 只需要 camera poses。这是个 beautiful 的 "data-free uncertainty quantification"。

更深入一层：Fisher 信息 = expected Fisher information = KL divergence 二阶展开。剪掉 Fisher 小的 surfel 等价于"剪掉对 posterior 影响小的 latent variable"。这跟 Bayesian deep learning 里的 Laplace approximation 是同源思想。PUP 3D-GS [11] 和 FisherRF [17] 是同一 family 工作。

参考：
- FisherRF: https://arxiv.org/abs/2311.17874
- Laplace Approximation in NNs: https://arxiv.org/abs/2106.10686

---

## 十二、可能引用的 web 资源汇总

| Resource | Link |
|---|---|
| 2DGS (Huang et al. SIGGRAPH 2024) | https://github.com/hbb1/2d-gaussian-splatting |
| 3DGS (Kerbl et al. ACM TOG 2023) | https://github.com/graphdeco-inria/gaussian-splatting |
| StopThePop (Radl et al. 2024) | https://github.com/lRaichu/StopThePop |
| Mip-Splatting (Yu et al. CVPR 2024) | https://github.com/autonomousvision/mip-splatting |
| GSTex (Rong et al. 2024) | https://arxiv.org/abs/2409.12954 |
| PUP 3D-GS (Hanson et al. 2024) | https://arxiv.org/abs/2406.10219 |
| Mip-NeRF 360 (Barron et al. CVPR 2022) | https://jonbarron.info/mipnerf360/ |
| Zip-NeRF (Barron et al. ICCV 2023) | https://jonbarron.info/zipnerf/ |
| Rip-NeRF (Liu et al. SIGGRAPH 2024) | https://rng-3.github.io/RipNeRF/ |
| NeRF (Mildenhall et al. ECCV 2020) | https://www.matthewtancik.com/nerf |
| Texture-GS (Xu et al. ECCV 2024) | https://arxiv.org/abs/2403.17829 |
| GaussianShader (Jiang et al. CVPR 2024) | https://asparagus15.github.io/GaussianShader/ |
| EWA Splatting (Zwicker et al. 2001) | https://www.cs.umd.edu/~zwicker/publications/EWAVolSplatting.pdf |
| Light Field Rendering (Levoy & Hanrahan 1996) | https://graphics.stanford.edu/papers/light/ |
| Lumigraph (Gortler et al. 1996) | https://www.cs.princeton.edu/~smr/papers/lumigraph/lumigraph.pdf |
| SortFreeGS (Hou et al. 2024) | https://arxiv.org/abs/2412.01042 |
| FisherRF (Jiang et al. 2023) | https://arxiv.org/abs/2311.17874 |
| PGSR (Chen et al. 2024) | https://arxiv.org/abs/2406.06521 |
| DTU Dataset | https://rovdymusic.com/dtu |
| K-buffer (Bavoil et al. 2007) | https://dl.acm.org/doi/10.1145/1232500.1232520 |

---

如果 Karpathu 你想进一步深挖某个方向，我推荐：
- **Fisher information 路线**：从 PUP 3D-GS → FisherRF → Laplace Neural Network 这条线，能看到 pruning / active learning / uncertainty estimation 的统一数学框架。
- **Anti-aliasing 路线**：Mip-NeRF → Zip-NeRF → Mip-Splatting → HDGS → Rip-NeRF，能看到 NeRF 系与 GS 系的 anti-aliasing 思想演化。
- **Texture disentangle 路线**：NeRF-Texture → NeuMesh → Texture-GS → GSTex → HDGS，能看到从 implicit field 到 explicit primitive 的 texture 解耦演化。
