---
source_pdf: Content-Aware Texturing for Gaussian Splatting.pdf
paper_sha256: bb73473c38165a507555aabedcc23b47169a436044829573097ebdfbefa1407b
processed_at: '2026-08-03T17:04:47-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇 paper 在干啥

## 一句话

给 2D Gaussian Splatting 的每个小圆盘贴上一张可大可小的"贴纸"，贴纸分辨率根据这块地方内容有多复杂自动伸缩——墙皮花纹复杂就贴精细图，墙面平整就贴粗图，省着用参数。

---

## 为什么要贴纸

标准 2DGS 一个 Gaussian = 一个颜色点。想表达高频花纹？只能把一整块区域切成成千上万个极小的 Gaussian，每个记一个平均色。问题是这堆小 Gaussian 的 position、scale、rotation、opacity 全在重复表示同一个平面——geometry 上完全冗余，只是为了让 appearance 密度够高。

传统 CG 三十年前就解决了这事：mesh 负责低频 shape，texture 负责高频 appearance，各管各的。但 GS 是优化出来的，没有 UV map，没法预先画 atlas。所以问题变成：怎么把 texture "挂"到优化出来的 Gaussian disc 上，并且让它自己长成合适的分辨率。

---

## 核心设计 1：贴纸的格子大小钉在 world space

最直觉的做法是把 texture 坐标绑在 Gaussian 的局部坐标系——Gaussian 缩放，贴纸跟着缩放。这听上去自然，实际上是个坑：optimizer 一调 scale，整个 texture 内容被拉伸变形，已学好的花纹全乱了，容易卡在 local minimum。

论文的 fix 很简单但很关键：texel size 用 world space 单位（比如 0.01 米一格），一旦设定就不变。Gaussian 长大？只是露出更多格子，已经画上去的内容原封不动。Gaussian 缩小？格子里画的东西还是那个东西。

这把"texture 内容"和"primitive shape"两个优化空间彻底解耦，optimizer 可以独立探索，不会互相干扰。

代价是 texture 分辨率变成动态的——primitive 大了需要更多格子，小了需要更少。所以得实现一个"锯齿状 tensor"存不同大小的 texture，每 100 iter 重新分配内存。

---

## 核心设计 2：贴纸分辨率自己长

给每个 Gaussian 一个 ratio 叫 $t_2p_r$（texel-to-pixel ratio），只能取 2 的幂。ratio=1 意味着 texel 和训练相机像素一样大（最精细）；ratio=8 意味着 texel 是像素 8 倍大（很粗，参数量 1/64）。

训练过程中每 250 iter 做两件事：

**Downscale（让它变粗）**：对 texture 做一次 lowpass filter，和原图比误差。误差小？说明这地方本来就没啥高频内容，texture 白存那么精细，把 ratio 翻倍，参数砍掉 3/4。

**Upscale（让它变细）**：算每个 primitive 的渲染误差，取 top 10% 误差最大的，把它们的 ratio 减半，参数涨 4 倍，给它们更多分辨率去拟合细节。

就这么一粗一细反复调，最后 scene 上自动长出一幅"容量地图"——花纹复杂的桌布 ratio 接近 1，光秃秃的白墙 ratio 可能到 8 或 16。

---

## 核心设计 3：什么时候该加 Gaussian 而不是加 texel

Upscale 给 texture 加到极限后，误差还大？说明问题不在 appearance 分辨率不够，而是 geometry 本身没建对——比如边缘没对齐、有个洞、primitive 被拉得过长。这时该加 primitive 数量，不该继续加 texel。

具体规则：误差 top 10% 的 primitive 里，如果 texture 分辨率已经超过阈值（比如 64×64），就把它 split 成两半——各向中心 ±1σ 偏移，scale 减半，texture 也减半，opacity 调一下保持 blending 连续。新 primitive 的 texture 从老的对应位置 bilinear sample 继承下来。

为什么不用 3DGS 经典的 clone？因为叠加多个 textured primitive 会让 optimizer 不稳定——几张贴纸互相抢颜色，参数又费。split 比 clone 干净。

这套 upscale → split 的流水线形成闭环：先拼命给 texture 加容量，加到天花板还不行就 split primitive，split 完新 primitive 各自 texture 更小，继续优化。

---

## Loss 里有个聪明的小 trick

texture 学的是"offset"叠加在 SH 颜色之上。如果不约束，texture 可能学出一大片非零值，但 alpha blending 一平均又平了，白白浪费参数。

所以加了 sparsity loss，L1 惩罚 texel 值，推大部分 texel 趋零。效果是 SH 学 base color，texture 只在真正需要高频细节的地方冒出小扰动。这也方便后期 K-means 压缩——大部分值接近 0，聚类效率高。

---

## 结果怎么样

默认设置下，DeepBlending 数据集上 PSNR 30.03，比 2DGS baseline 高 0.5 dB，比 BBSplat 高 0.8 dB，但 primitive 数只有 2DGS 的 1/6.5，总参数还略低。

把竞品 BBSplat 和 GStex 强行调到相同参数预算，它们 PSNR 掉 1.1–1.7 dB，因为固定 16×16 分辨率或启发式分配本质上不如 content-aware 分配高效。

最漂亮的是 Table 3：固定 primitive 数从 40K 涨到 160K，Tanks&Temples 上 texel 总量反而从 15.4M 降到 10.5M。更多 primitive → 每个 primitive 覆盖更局部更简单内容 → 不需要那么密的 texture。geometry 和 appearance 自动平衡，不用人调。

---

## 跟你熟悉的东西连一下

这套思路本质上是 **把 capacity allocation 当成一阶优化对象**。之前 GS 系列把 densification 当成训练循环里的一个固定 schedule（梯度大的就 clone/split），这里把 texel size 也变成 error-driven 的可调度量，两类自由度各有各的 error signal 驱动，类似 mixture-of-experts 里 router 学着把容量分配到需要的地方。

SH 学低频 base、texture 学高频 sparse offset、sparsity loss 强制分工——这个频段分离和 ResNet 的 residual learning、VQ-VAE 的 codebook 学残差、Laplacian pyramid 显式分频段是一脉相承的直觉。

把 texel size 钉在 world space 而不是 canonical space，这个改动小但意义深。类似 NeRF 的 positional encoding 把"频率"和"坐标"解耦——这里把"texture 像素密度"和"primitive 几何缩放"解耦。解耦之后 optimizer 才不会在两个子空间之间来回踩坑。

---

## Limitations

没赶上 3DGS 的 NVS 质量——所有 2DGS-based 方法都这样，因为 2D surfel 对 view-dependent 效果（高光反射）表达力弱。没有 anti-aliasing，远离训练视角会 alias。没用 GPU texture hardware，custom CUDA per-ray query 有 overhead，但 WebGL 端如果能用 hardware sampling 会很赚。

3DGS ellipsoid + 3D texture（voxel 或 hash grid）是开放问题，[Textured-GS](https://arxiv.org/abs/2410.03708) 试了 2D plane 挂 3D primitive，但没充分分析。

---

## 我的直觉 takeaway

这篇 paper 最大的贡献是给了 GS 一个显式的 **frequency axis**。3DGS 原始设计里频率是隐含在 primitive 密度里的——细节多就多放点。这里把"点数"和"每点的纹理频率"拆成两个独立维度，各有各的 error signal，各有各的 grow/shrink 机制。

如果要继续推：texel size 能不能做成连续可微而非 2 的幂离散？能不能跟 [Scaffold-GS](https://arxiv.org/abs/2312.00109) 的 anchor + neural Gaussian 结合，anchor 当低频容器、texture 当高频细节？甚至能不能把这套 content-aware 思路搬上 video，让 texel resolution 随时间也自适应？这些都是 open 的好方向。

project page / code 暂时没在 arXiv 找到 official release，可能在 Inria graphdeco 后续放出。

---

# Content-Aware Texturing for Gaussian Splatting 深度讲解

## 1. 出发点：3DGS/2DGS 的容量错配问题

要 build intuition，先看标准 2D Gaussian Splatting 一个被忽视的"细粒度耦合"：**每个 surfel 只携带一组 SH 系数**（48 个标量用于 3 阶球谐），即 1 primitive → 1 color sample。这意味着当 appearance 频率高于 geometry 频率时（比如一面几何上平整的墙，但贴满了高频砖纹），唯一办法是把墙分成几千个小 primitive，每个用自己单独的 SH 拟合一小块平均颜色。geometry 信息被无谓地"复制"——位置、scale、rotation、opacity 全都在重复表示同一个平面。

传统 CG 早就给出了答案：**texture mapping**。把 geometry（mesh/三角形）和 appearance（UV texture）解耦，geometry 简单 + texture 高频。问题是：3DGS 是优化出来的而非手工建模，UV 参数化未知，怎么"挂"texture 到 Gaussian 上？这是论文要解的核心问题。

并发工作的几条路径都各有缺陷：
- **BBSplat** [arXiv:2411.08508](https://arxiv.org/abs/2411.08508)：每个 primitive 固定 16×16 texture，且 texture 定义在 canonical 空间，随 primitive 一起 scale → 会出现 stretching artifact。
- **GStex** (WACV 2025): https://openaccess.thecvf.com/content/WACV2025/html/Rong_GStex_Per-Primitive_Texturing_of_2D_Gaussian_Splatting_for_Decoupled_WACV_2025_paper.html ：基于已收敛的 2DGS 点云起步，固定 texel budget 启发式分配 → 点云本身没为 texture 优化，预算分配不能动态调整。
- **SuperGaussians** [arXiv:2404.23710](https://arxiv.org/abs/2404.23710)：固定纹理分辨率，可选 small NN 表示纹理。
- **Textured-GS** [arXiv:2410.03708](https://arxiv.org/abs/2410.03708)：3DGS 基础 + 固定分辨率纹理。

论文要做的事：**让 texture 在 optimization 过程中内容感知地伸缩** —— 高频区给小 texel，低频区给大 texel；并把"primitive 数量"和"texel 总量"作为两类自由度独立调度。

---

## 2. 基础背景：2D Gaussian Splatting 复盘

每个 2D surfel 参数集 $A = \{\mu, \sigma, \mathbf{q}, o, \mathbf{SH}\}$：

| 符号 | 含义 | 维度 |
|---|---|---|
| $\mu$ | primitive center（world space 位置） | $\mathbb{R}^3$ |
| $\sigma$ | 两主轴 scale | $\mathbb{R}^2_+$ |
| $\mathbf{q}$ | quaternion 表示旋转 $R$ | $\mathbb{R}^4$ |
| $o$ | opacity | scalar |
| $\mathbf{SH}$ | 球谐系数 → 视角相关颜色 $\mathbf{c}(\mathbf{d})$ | 48 维 (3阶) |

法向量 $\mathbf{n}$ 是 $R$ 的第三列（disc 法线方向）。

光线-primitive 相交（Eq. 1）：给定光线 $\mathbf{r} = \mathbf{r}_0 + t\mathbf{d}$，相交点参数：

$$
t = \frac{\mathbf{n}\cdot(\boldsymbol{\mu} - \mathbf{r}_0)}{\mathbf{n}\cdot\mathbf{d}}, \quad \mathbf{p} = \mathbf{r}_0 + t\mathbf{d}
$$

分母 $\mathbf{n}\cdot\mathbf{d}$ 即光线与 disc 法线夹角的余弦；分子 $\mathbf{n}\cdot(\boldsymbol{\mu}-\mathbf{r}_0)$ 是 primitive center 沿法线方向的投影距离。

Alpha compositing（Eq. 2-3）：

$$
\mathbf{C}(\mathbf{r}) = \sum_i w_i(\mathbf{p}_i)\, \mathbf{c}_i(\mathbf{d}), \quad
w_i(\mathbf{p}_i) = T_i\, o_i\, G_i(\mathbf{p}_i), \quad
T_i = \prod_{j=1}^{i-1}(1 - o_j G_j(\mathbf{p}_j))
$$

其中 $T_i$ 是 transmittance（前面所有 primitive 的透射率乘积），$G(\mathbf{x}) = e^{-\frac12 \mathbf{x}^T\mathbf{x}}$ 是 Gaussian falloff。

论文使用四套坐标系方便不同操作：
- $\mathbf{p}^w = \mathbf{p}$：world space 原始交点
- $\mathbf{p}^{w_0} = \mathbf{p}^w - \boldsymbol{\mu}$：world space 平移到 primitive 中心
- $\mathbf{p}^l = R^{-1}\mathbf{p}^{w_0}$：local axis-aligned 坐标系（去掉旋转）
- $\mathbf{p}^c = S^{-1}\mathbf{p}^l$，$S=\mathrm{diag}(\sigma)$：canonical normalized space（去掉 scale，落在单位 disc 内）

下标 i 表示第 i 个 primitive，上标 w/l/c 表示坐标系。

---

## 3. Texture 表示的核心设计：World-fixed Texel Size

### 3.1 Naive UV mapping（为什么不工作）

最直觉的做法（Eq. 5）：把 canonical 空间坐标线性映射到纹理 UV：

$$
\mathbf{u} = \left(\frac{\mathbf{p}^c}{2 s_i} + 0.5\right) \cdot T_{\mathrm{res}}
$$

- $\mathbf{p}^c$：canonical normalized 坐标，范围大致 $[-1,1]$
- $s_i$：texture 在 Gaussian 标准差单位下的覆盖范围（典型 $s_i = 3$，覆盖 $\pm 3\sigma$）
- $T_{\mathrm{res}}$：texture 分辨率（texel 数）

问题：texture 坐标绑死在 **canonical 空间**，当 optimizer 调整 $\sigma$ 时，primitive 缩放，但 UV 坐标系跟着缩放 → 已学到的纹理被拉伸/压缩（Fig. 2 上排）。SH 和 texture 优化耦合，容易卡在 local minima，视觉上出现 stretching blur。

### 3.2 关键改动：把 texel size 钉在 world space（Eq. 6）

$$
\mathbf{u} = \frac{\mathbf{p}^l}{k_i} + T_{\mathrm{offset}}
$$

- $\mathbf{p}^l$：local axis-aligned 坐标（已去旋转，但未去 scale），单位是 world space 单位
- $k_i$：**world space texel size**（米/单位长度），不随 $\sigma$ 变化
- $T_{\mathrm{offset}}$：texture 中心偏移

直觉：texel 现在像一张铺在 scene 上的"绝对坐标网"，primitive 增大只是露出更多网格，已写入的内容保持不动。这就把"texture 内容"和"primitive shape"参数彻底解耦。

代价：texture 分辨率 $T_{\mathrm{res}}$ 不再固定。当 $\sigma$ 增大，覆盖面积变大 → 需要更多 texel（因为 $k_i$ 固定）。论文必须实现动态分配/释放 texture memory 的机制（jagged tensor + 每 100 iter 重分配）。

### 3.3 两个约束原则

1. **频率约束**：texel 投影到最近训练 camera 后的尺寸 ≥ 该 camera 的像素尺寸。否则 texture 频率超过 Nyquist，浪费且 alias。最小 texel size：

$$
k_{\min}^p = \text{最近 camera 像素的反投影 world-space 尺寸}
$$

2. **内容自适应**：高频外观给小 $k_i$（多 texel），低频给大 $k_i$（少 texel）。

---

## 4. Progressive Adaptive Texel Size：Downscale / Upscale

引入 **texel-to-pixel ratio** $t_2p_r$（取 $2$ 的幂次方便每次 resize 因子为 2）：

$$
k_i = k_{\min}^p \cdot t_2p_r \quad \text{(Eq. 7)}
$$

直觉：$t_2p_r=1$ 表示 texel 与像素同尺寸（最高有效分辨率）；$t_2p_r=8$ 表示 texel 是像素 8 倍大（粗化 8 倍，texel 总数减为 1/64）。

### 4.1 Downscale：低频区域放大 texel

对当前 texture 应用 lowpass filter 得到 $T_{\mathrm{lowpass}}$，与原 texture $T_{\mathrm{orig}}$ 比较，按 Gaussian 权重累积误差（Eq. 8）：

$$
\mathcal{E}_d = \frac{1}{\sum_{\mathbf{p}} G(\mathbf{p})} \sum_{\mathbf{p}} G(\mathbf{p}) \left( T_{\mathrm{orig}}(\mathbf{p}) - T_{\mathrm{lowpass}}(\mathbf{p}) \right)
$$

- $G(\mathbf{p})$：以 primitive 中心为原点的 Gaussian 权重，让 primitive 中心区域误差占主导（边缘 Gaussian 权重低）
- $\sum_{\mathbf{p}} G(\mathbf{p})$：归一化项

如果 $\mathcal{E}_d < \tau_{\mathrm{ds}}$（阈值），说明 lowpass 几乎无损，$t_2p_r \leftarrow 2 \cdot t_2p_r$，texture 参数量 $\times \frac14$（每轴减半，面积减 1/4）。

### 4.2 Upscale：高频区域缩小 texel

基于 [Rota Bulò et al., ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-73036-8_20) 的 per-primitive error 思路。

对单视角 $\pi$，primitive $i$ 的误差（Eq. 9）：

$$
E_i^\pi = \sum_{\mathbf{r} \in \mathcal{P}_i} \mathcal{E}_\pi(\mathbf{r})\, w_i^\pi(\mathbf{r})
$$

- $\mathcal{P}_i$：primitive $i$ 覆盖的所有像素
- $\mathcal{E}_\pi(\mathbf{r})$：视角 $\pi$ 的 RGB error image 在像素 $\mathbf{r}$ 处的值
- $w_i^\pi(\mathbf{r})$：primitive $i$ 在该像素的 blending weight

多视角聚合（Eq. 10）—— 与原作取 max 不同，这里做**贡献加权平均**：

$$
E_i = \frac{\sum_{\pi \in \Pi} E_i^\pi\, \overline{w_i^\pi}}{\sum_{\pi \in \Pi} \overline{w_i^\pi}}, \quad
\overline{w_i^\pi} = \sum_{\mathbf{r} \in \mathcal{P}_i} w_i^\pi(\mathbf{r})
$$

- $\Pi$：所有训练视角
- $\overline{w_i^\pi}$：primitive $i$ 在视角 $\pi$ 中的总贡献

取 error top 10% 的 primitive，$t_2p_r \leftarrow t_2p_r / 2$，texture 参数量 $\times 4$。

### 4.3 渐进调度

每 250 iter 跑一次 downscale + upscale 检查，直到 25k iter。低频区域 texel 越变越大，高频区域越变越细，整个 scene 的"容量地图"自适应成型（Fig. 1 右侧面板，颜色编码 texel-to-pixel ratio）。

---

## 5. Resolution-aware Splitting：区分 Appearance Error vs Geometry Error

Upscale 把 appearance 容量加到极限后仍误差大，说明问题不在 texture 频率，而在 **geometry 本身没建模对**（如边缘没对齐、洞、拉长的 primitive）。这时不能再加 texel，应该加 primitive。

### 5.1 为什么不用 clone？

3DGS-MCMC 的 clone/densify 思路是把高梯度区域 primitive 复制一份。但这里叠加多个 textured primitive 会让 optimization 不稳定——纹理间互相"争"颜色，且参数浪费严重。论文只采用 **split**。

### 5.2 Split 规则

候选：error top 10% 中，texture resolution 超过阈值 $\tau_{\mathrm{tr}}$（初始 64，渐进降到 32）的 primitive。

对每个 axis 独立判断（Fig. 4）：若该 axis 上 resolution > $\tau_{\mathrm{tr}}$，沿该 axis 切成两半：

- 新 primitive 各自向 $\pm 1\sigma$ 偏移
- scale 减半 → 覆盖原 area 的 1/4（两 axis 都切的话）
- texture 分辨率减半（因 $k_i$ 不变，primitive 缩小，所需 texel 减少）
- opacity 乘以 $G(1)$（保持边缘 blending 不变）
- 新 primitive 的 texture 通过原 texture 在对应位置 bilinear sample 得到（继承高频细节）

### 5.3 与 upscale 的协同

- Upscale 解决 appearance error（加 texel）
- Split 解决 geometry error（加 primitive）
- 流程（Fig. 5 / Alg. 1）：每 250 iter 计算 $E_i$ → 若 texture res > $\tau_{\mathrm{tr}}$ 且 $E_i$ top 90% → split；否则若 $t_2p_r > 1$ 且 $E_i$ top 90% → upscale；否则若 $\mathcal{E}_d < \tau_{\mathrm{ds}}$ → downscale

这是一个稳定的反馈环：geometry 不够 → texture 先被 upscale 到极限 → split 触发 → 新 primitive 各自 texture 更小 → 继续 optimize。

---

## 6. Loss 设计

总损失（Eq. 15）：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{RGB}} + \mathcal{L}_{\mathrm{texture}} + \mathcal{L}_{\mathrm{opacity}}
$$

### 6.1 RGB loss（Eq. 11）

$$
\mathcal{L}_{\mathrm{RGB}} = (1 - \lambda_{\mathrm{SSIM}})\mathcal{L}_1 + \lambda_{\mathrm{SSIM}} \mathcal{L}_{\mathrm{SSIM}}, \quad \lambda_{\mathrm{SSIM}} = 0.2
$$

### 6.2 Texture sparsity loss（Eq. 12, 13）

核心 insight：让 SH 学 base color（view-dependent 平均色），texture 只学高频 offset。如果 texture 自由学，可能学出大范围非零值，alpha blending 后反而被平滑掉，浪费参数。

texel 激活约束在 $[-1, 1]$：

$$
\mathbf{c}^{T_i} = 2\sigma(\mathbf{c}^{\prime T_i}) - 1
$$

- $\mathbf{c}^{\prime T_i}$：未激活 texture feature（优化变量）
- $\sigma(\cdot)$：sigmoid
- $2\sigma - 1$ 把 $(0,1)$ 映射到 $(-1,1)$

sparsity loss：

$$
\mathcal{L}_{\mathrm{texture}} = \lambda_{\mathrm{texture}} \sum_i |\mathbf{c}_i^{T_i}|
$$

- $|\cdot|$ 对每个 texel 每个通道求 L1
- 推动大部分 texel 趋零，只在必要位置出现"扰动"

效果：训练完 texture 大部分区域接近 0，少数位置有 sharp offset，便于后期 K-means 压缩。

### 6.3 Opacity regularization（Eq. 14）

$$
\mathcal{L}_{\mathrm{opacity}} = \lambda_{\mathrm{opacity}} \frac{1}{N}\sum_i^N \mathbf{o}_i
$$

推动低贡献 primitive 的 opacity → 0，自然 prune 掉冗余 geometry。继承自 [Papantonakis et al., 2024](https://dl.acm.org/doi/10.1145/3651282) 和 3DGS-MCMC。

---

## 7. 实现细节

- 3DGS codebase + 2DGS primitive（用 Eq. 1 在 camera space 直接求交，不用 2DGS 原版的三平面法）
- **Jagged tensor**：各 primitive texture 分辨率不同，自定义数据结构存储；每 100 iter 动态分配/释放
- Texture 覆盖范围 $\pm 3\sigma$；硬上限 256 texel/轴
- 超出已分配区域 → zero padding（因为是 offset，zero 即等于 SH 颜色，自然过渡）
- $t_2p_r$ 下限设 2（sub-texel 细节可由 alpha blending + primitive overlap 恢复，无需更高分辨率）
- 训练时长 1.5–2× 2DGS，渲染慢 25%
- 压缩：texel 值经过 sigmoid 限幅 → 适合 K-means 聚类压缩

---

## 8. 实验结果解析

### 8.1 Table 1：默认设置全量对比

| 方法 | DeepBlending SSIM/PSNR/LPIPS | Points | Texels | Params | FPS |
|---|---|---|---|---|---|
| 3DGS-MCMC | 0.903 / 29.81 / 0.311 | 1323K | 0 | 78.1M | 265 |
| 2DGS* | 0.899 / 29.52 / 0.324 | 1444K | 0 | 83.8M | 96 |
| BBSplat | 0.898 / 29.25 / 0.318 | 160K | 41.0M | 173.0M | 27 |
| GSTex | 0.906 / 29.63 / 0.323 | 1503K | 10.0M | 117.2M | 21 |
| **Ours** | **0.907 / 30.03 / 0.303** | **222K** | **21.6M** | **78.1M** | 70 |

观察：
- Ours 在 DeepBlending 三项指标都最优，primitive 数比 2DGS* 少 6.5×，参数总量反而略低
- BBSplat primitive 少（160K）但 texel 多（41M），总参数是 Ours 的 2.2× —— 因为固定 16×16 比例对低频区域严重浪费
- GSTex 在 primitive 数上"继承"了 2DGS 点云的 1.5M，texture 利用率低
- Mip-NeRF-360 上 Ours 略低于 2DGS* SSIM 0.795 vs 0.801，但 LPIPS 更优 0.263 vs 0.282，参数相当
- Tanks&Temples 上 Ours FPS 121 远超其它 texturing 方法（BBSplat 38, GSTex 20）

### 8.2 Table 2：固定参数预算下的对比

把 BBSplat 和 GSTex 调到与 Ours 相同的参数预算，结果明显退化：

| DeepBlending | SSIM | PSNR | LPIPS |
|---|---|---|---|
| BBSplat | 0.895 | 28.93 | 0.332 |
| GSTex | 0.896 | 28.29 | 0.354 |
| **Ours** | **0.907** | **30.03** | **0.303** |

PSNR 差 1.1–1.7 dB。证明 content-aware 分配比固定/启发式分配本质上更高效。

### 8.3 Table 3：固定 primitive budget 扫描

DeepBlending 上 primitive 从 40K 增到 160K：
- SSIM: 0.890 → 0.905
- Texels: 13.2M → 19.9M（几乎线性，因为更多 primitive 覆盖更多面积）
- Params: 41.9M → 69.1M

Tanks&Temples：primitive 增多反而 texels **下降** 15.4M → 10.5M。这是 content-aware 的精髓——更多 primitive 让每个 primitive 表达更局部更简单内容，texture 不必那么密。geometry/appearance 之间自动平衡。

### 8.4 Table 4：超参数敏感度

| 变量 | 效果 |
|---|---|
| $\tau_{\mathrm{ds}}$ 调高 | downscale 更激进 → texels 大幅减少（21.6M→7.9M in DeepBlending），质量略降 |
| $\tau_{\mathrm{tr}}$ 调低 | split 更易触发 → primitives 增多（222K→316K），quality 提升，但训练慢 |

### 8.5 Table 5：单 scene 统计

最大：stump（户外高频植被）317.6M params, 1270MB；最小：train 31.8M params, 127MB。Texels/primitive 范围 30.8（bonsai）到 450.5（bicycle）——这正体现了 content-aware 的自适应范围。

---

## 9. 与你 (Karpathy) 思维框架的几个连接点

1. **Capacity allocation as a control problem**：这套 system 类似于 neural network 中"在哪里花参数"的问题。NeRF 用一个统一 MLP 把所有容量等价分配；3DGS 用 spatially-adaptive densification；这里进一步把"参数密度"分成两类（primitive 数 × texel 数），每类有独立 feedback signal（geometry error vs appearance error）。本质是用两个误差信号驱动两类参数的 growth/pruning，类似 mixture-of-experts 的 routing。

2. **Decoupling frequency bands**：SH 学低频 base color（球谐本身就是低阶基函数），texture 学高频 offset，sparsity loss 强制分工。这非常像 ResNet 的 residual learning、或 VQ-VAE 中 codebook 学高频残差。其实更类似 image pyramid / Laplacian pyramid 显式分离频段。

3. **Coordinate system matters**：把 texel size 钉在 world space 而非 canonical space 这个改动看似小，却是论文最深的 insight。类似 NeRF 用 positional encoding 把"频率"和"坐标"解耦——这里把"texture 像素密度"和"primitive 几何缩放"解耦。一旦解耦，optimizer 才能独立探索两个子空间，避免 local minima。

4. **Why not 3D texture / hash grid?**：论文 limitation 讨论了 3D primitive 配 texture 的开放问题——是用 2D plane texture 还是 3D voxel/hash grid？[Textured-GS](https://arxiv.org/abs/2410.03708) 选了 2D plane。这里选 2D 是因为 surfel 本身就是 2D disc，texture 直接挂上即可。但 3DGS 的 ellipsoid 没有天然 2D 参数化（需要切 uv atlas），这正是 2DGS 的天然优势。

5. **Progressive growing 的 echo**：texel size 从粗到细 upscale，类似 Progressive Growing of GANs（Karras et al. 2018）——先学低频内容，再逐步加入高频容量，避免早期 optimizer 在高频空间迷失。这里 downscale 还提供了"反悔机制"：如果某区域实际只需低频，把已 upscale 的 texture 收回，省参数。

---

## 10. Limitations 与未来方向

1. **没赶上 3DGS 的 NVS 质量**：所有 2DGS-based 方法（包括本作）在 NVS 上略低于 3DGS，因为 2D surfel 对 view-dependent 效果（高光、反射）表达力弱。这是 2D vs 3D primitive 的根本 trade-off。

2. **无 anti-aliasing**：$t_2p_r \ge 2$ 保证训练 view 附近不 alias，但远离 convex hull 后会 alias。完整方案应类似 [Mip-Splatting](https://arxiv.org/abs/2311.16493) 的 2D Mip filter。

3. **未利用 GPU texture hardware**：当前是 custom CUDA 实现，per-ray texture query 有 overhead。WebGL 上的 [splat viewer](https://github.com/antimatter15/splat) 若能直接用 hardware texture sampling 会大幅加速。

4. **3D texture 扩展**：能否把这套 content-aware 思路搬到 3DGS ellipsoid + 3D voxel / hash grid texture 上？hash collision 与 adaptive resolution 的交互是关键难题，[Instant-NGP](https://arxiv.org/abs/2201.05989) 已有 multi-resolution hash 经验可借鉴。

5. **Compression**：sigmoid 限幅 + K-means 是简单 first step，更激进可结合 [3DGS.zip 综述](https://arxiv.org/abs/2407.09510) 中的 vector quantization、spherical harmonic prune、codec 等。

---

## 11. 个人 takeaway

这篇论文真正的贡献不是"给 GS 加 texture"（这个想法很多人都有），而是把 **capacity allocation 提升为一阶优化对象**：
- 之前的 texturing GS：固定纹理分辨率（BBSplat 16×16、SuperGaussians），或一次性启发式分配（GStex）
- 这篇：texel size 作为可优化自由度，由 error signal 驱动 2 的幂次递增/递减

这让我想起 [Mip-NeRF 360](https://arxiv.org/abs/2111.12055) 在 NeRF 上做 multiscale 的思想：**显式对频率资源建模比让单一表示"自己学出来"更可控**。3DGS 缺少一个显式的频率轴，本作通过 texel size 给它加上了。

如果继续推这条路：texel size 是否可以是连续可微的（而非 2 的幂次离散）？dynamic texture resolution 是否可以与 [Scaffold-GS](https://arxiv.org/abs/2312.00109) 的 anchor + neural Gaussian 结合？把 anchor 当作"低频 geometry + 高频 texture"的容器，texture 用本作的 content-aware 方式，可能得到一个非常紧凑且高频表达力强的 hybrid。

project page / code（如公开）：未在 arXiv 找到 official project page；论文列在 Inria NERPHYS ERC 项目下（https://project.inria.fr/nerphys/）。Inria/graphdeco-quantitative 仓库可能后续放出。
