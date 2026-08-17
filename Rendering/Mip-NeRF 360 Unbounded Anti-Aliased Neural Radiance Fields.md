---
source_pdf: Mip-NeRF 360 Unbounded Anti-Aliased Neural Radiance Fields.pdf
paper_sha256: 256e36b546a7a6d985341225c6cc8f3745092f8f4733eaea2b0ea942d9210659
processed_at: '2026-08-05T18:49:25-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej, 没问题，咱们抛开那些生硬的学术八股文，直接用大白话和底层直觉来过一遍这篇 paper。

Mip-NeRF 360 要解决的核心痛点非常明确：以前的 NeRF 和 Mip-NeRF 只能玩转“室内小物件”或者“单一朝向的前向场景”。一旦你拿着相机绕着一棵树、一栋楼转 360 度拍，而且背景还有无限远的远山和天空，模型就彻底傻眼了。表现就是：背景模糊一片、前景细节丢失、空中飘着一堆半透明的“幽灵”伪影，而且训练慢得让人绝望。

这篇 paper 就是针对这三个致命伤，开了三剂猛药。

### 1. Parameterization: 如何把“无限大”的世界塞进“有限大”的脑子？

**直觉：**
NeRF 的 MLP 就像一个容量有限的记忆体。在 360 度场景中，近处的一朵花和远处的一座山在图像上可能占据同样的像素大小，但在真实的 3D 空间里，山的体积比花大亿万倍。如果按照真实的 Euclidean 坐标去喂给 MLP，MLP 会把所有的 capacity 都浪费在远处的空白背景上，近处的花反而学不清楚。

我们需要一种“失真”的映射，让近处的东西占据更多的网络容量，远处的东西被压缩。原版 NeRF 针对“一直往前看”的场景用了 NDC (Normalized Device Coordinates)，本质上是沿着 z 轴做 disparity (深度的倒数) 变换。但 360 度场景四面八方都是无限的，NDC 就失效了。

**技术细节：Contraction 与 Kalman 滤波**
论文提出了一种叫 `contract` 的空间扭曲函数：
$$
\mathrm{contract}(\mathbf{x}) = \begin{cases} 
\mathbf{x} & \|\mathbf{x}\| \leq 1 \\ 
\left(2 - \frac{1}{\|\mathbf{x}\|}\right) \left(\frac{\mathbf{x}}{\|\mathbf{x}\|}\right) & \|\mathbf{x}\| > 1 
\end{cases}
$$
*   **$\mathbf{x}$**: Euclidean 空间中的 3D 坐标点。
*   **$\|\mathbf{x}\|$**: 该点到场景中心（通常设为相机环绕中心）的欧几里得距离。
*   **直觉**: 这就像一个黑洞。在半径为 1 的单位球内部，空间保持原样不变。一旦你走出单位球，你的方向 $\frac{\mathbf{x}}{\|\mathbf{x}\|}$ 保持不变，但你的半径被压缩成了 $2 - \frac{1}{\|\mathbf{x}\|}$。当你离中心 1 米时，你在边界 1 上；当你离中心 10000 米时，你在 1.9999 上；当你无限远时，你被死死压在半径为 2 的球面上。无限大的宇宙被完美塞进了一个半径为 2 的球里。

但是，Mip-NeRF 处理的不是一个点，而是一个由圆锥台近似出来的 3D Gaussian $(\mu, \Sigma)$。对单个点做非线性变换容易，对一个 Gaussian 做非线性变换，出来的就不再是 Gaussian 了。

论文极其聪明地借用了控制论里的 Extended Kalman Filter (EKF) 思想：局部线性化。
$$
f(\mu, \Sigma) = \left( f(\mu), \mathbf{J}_f(\mu) \Sigma \mathbf{J}_f(\mu)^{\mathrm{T}} \right)
$$
*   **$\mu$**: Gaussian 的中心均值。
*   **$\Sigma$**: Gaussian 的协方差矩阵，代表这个椭球的形状和大小。
*   **$\mathbf{J}_f(\mu)$**: 收缩函数 $f$ 在点 $\mu$ 处的 Jacobian 矩阵。
*   **直觉**: 既然非线性变换会破坏 Gaussian 性质，那我们就在均值点附近做个一阶泰勒展开。新的均值直接变换过去 $f(\mu)$，新的协方差用 Jacobian 矩阵左右乘一下，相当于做了一个局部的线性旋转拉伸。这样我们就能在 contracted space 里继续算 Integrated Positional Encoding (IPE) 了。

**配套的射线采样：**
光把空间压扁还不够，射线上采样的间距也要改。论文定义了归一化距离 $s$：
$$
s \triangleq \frac{g(t) - g(t_n)}{g(t_f) - g(t_n)}
$$
如果令 $g(t) = 1/t$ (disparity)，那么在 $s \in [0, 1]$ 空间里均匀采样，等价于在真实的深度 $t$ 空间里按视差均匀采样。近处密密麻麻采一堆，远处稀稀拉拉采几个。这与 `contract` 空间形成了绝佳的配合：在 contracted space 里，射线看起来就像是被均匀分割的一样，MLP 看到的世界非常和谐。

---

### 2. Efficiency: 别让大网络干粗活

**直觉：**
原版 Mip-NeRF 有个很败家的操作：Coarse-to-Fine。先用一个 MLP 粗略估一遍权重 $w^c$，然后根据 $w^c$ 重新采样，再用同一个 MLP 算一遍 fine 权重 $w^f$。这个 Coarse 渲染的结果根本不会输出到最终图像里，纯粹是为了给 Fine 指路。这就好比你请了一个年薪百万的资深架构师去写初版的 CRUD 代码，纯属浪费 capacity 和算力。

**技术细节：Proposal MLP 与 Online Distillation**
论文把一个大 MLP 拆成了两个：
1.  **Proposal MLP ($\Theta_{\mathrm{prop}}$)**: 4 层，256 宽度。只输出 density，不输出 color。廉价，快速，被评估多次（论文中是 2 次，各 64 个 samples）。相当于实习生，专门负责画草图，定大致边界。
2.  **NeRF MLP ($\Theta_{\mathrm{NeRF}}$)**: 8 层，1024 宽度。输出 density 和 color。昂贵，慢速，只评估一次（32 个 samples）。负责精修出图。

**核心的 Loss 设计：**
如何让实习生（Proposal）向架构师学习？论文设计了一个基于直方图上界的 loss：
$$
\mathcal{L}_{\mathrm{prop}} = \sum_i \frac{1}{w_i} \max\left(0, w_i - \operatorname{bound}(\hat{\mathbf{t}}, \hat{\mathbf{w}}, T_i)\right)^2
$$
*   **$w_i$**: NeRF MLP 在第 $i$ 个小区间产生的精确权重。
*   **$\hat{w}$**: Proposal MLP 产生的粗略权重。
*   **$\operatorname{bound}$**: 算出 Proposal MLP 中所有覆盖到第 $i$ 个区间的粗略 bin 的权重之和。
*   **直觉**: 这个 loss 是 asymmetric 的。它只惩罚 Proposal MLP “低估”了 NeRF MLP 的情况。如果 NeRF 在某个深度发现了一堵墙（$w_i$ 很大），那么 Proposal MLP 在这个深度附近的粗略 bin 的总和，**必须大于等于** $w_i$。如果小于，说明 Proposal 漏掉了关键物体，就要被惩罚。如果 Proposal 高估了（总和大于 $w_i$），没关系，大不了 Fine 采样时多采几个空点而已，无伤大雅。

在反向传播时，对 NeRF MLP 的输出 $w$ 强行打上 `stop-gradient`。逻辑很清晰：架构师好好画图，实习生去模仿架构师。绝不允许架构师为了迁就实习生而故意把图画烂。

---

### 3. Ambiguity: 消灭“Floaters”和“背景塌陷”

**直觉：**
NeRF 的方程存在巨大的 ill-posedness。如果模型懒得学远处那棵树的复杂纹理，它完全可以作弊：在相机正前方不到 1 米的地方，悬浮一团半透明的密度云，颜色调成树的绿色。从训练视角看，像素对得严丝合缝；换个视角看，就是一团恶心的 floater。这就是 Background Collapse。同理，空中也会飘满解释某些视角漏洞的 floaters。

原版 NeRF 用在 density 里加高斯噪声的方法逼着 density 二值化，但这招在 360 场景里不够用了。

**技术细节：Distortion Loss**
论文提出了一个极其优雅的一维正则化项：
$$
\mathcal{L}_{\mathrm{dist}}(\mathbf{s}, \mathbf{w}) = \sum_{i,j} w_i w_j \left|\frac{s_i + s_{i+1}}{2} - \frac{s_j + s_{j+1}}{2}\right| + \frac{1}{3} \sum_i w_i^2 (s_{i+1} - s_i)
$$
*   **$\mathbf{s}$**: 归一化后的射线距离（使用 $s$ 空间是为了让远近一视同仁，避免远处的 loss 权重过大）。
*   **$w_i, w_j$**: 第 $i$ 和第 $j$ 个区间的权重。
*   **$s_i, s_{i+1}$**: 第 $i$ 个区间的起点和终点。
*   **直觉**: 这个公式本质上是计算射线上一维分布的“转动惯量”。
    *   第一项 $\sum_{i,j} w_i w_j |mid_i - mid_j|$: 惩罚任意两个有权重的区间之间的距离。如果你有多个权重峰值（floaters），它们彼此之间有距离，这个 loss 就会巨大。要把这个 loss 降下来，所有的权重必须往一个点靠拢。
    *   第二项 $\frac{1}{3} \sum_i w_i^2 (s_{i+1} - s_i)$: 惩罚单个区间的宽度。如果某个区间的权重很大，它必须极窄（表面必须 sharp）。
    *   合在一起：这条射线上的密度分布，要么什么都没有（全空），要么必须像一根针一样（Dirac delta），极其尖锐且集中在一个点上。

这个 regularizer 极其有效。看论文的 Table 2，去掉 $\mathcal{L}_{\mathrm{dist}}$，PSNR 只掉了一点点（24.37 -> 24.41），但视觉上 floaters 满天飞。它主要是在规范几何，而纯粹提升像素拟合度。

---

### 4. 容易被忽略的隐藏技巧

**Off-Axis Positional Encoding (离轴位置编码):**
原版 Mip-NeRF 的 IPE 只取协方差矩阵 $\Sigma$ 的对角线，这意味着它分不清一个“横着的扁椭圆”和一个“竖着的扁椭圆”。在 Euclidean 空间里这无所谓，但在经过 `contract` 空间扭曲后，远处的 Gaussian 会被极度拉长成各向异性，方向信息就变得至关重要了。
论文引入了一个基于二十面体顶点构造的 skinny matrix $P$，用它来捕捉 $\Sigma$ 的离轴信息。计算量增加极少，但极大提升了远处各向异性特征的分辨能力。

**Histogram Resampling Fix:**
原版 Mip-NeRF 从 coarse histogram 采样新的区间端点时，直接用采到的点做端点。这会导致区间无法覆盖到原始峰值的边缘，产生侵蚀效应。Mip-NeRF 360 改为采样后取相邻点的中点作为新区间端点。一个小小的改动，消除了讨厌的锯齿。

---

### 实验数据的直觉印证

看 Table 1，对比 baseline：
*   **Mip-NeRF**: 0.7M params, 3.17 hours, PSNR 24.04。
*   **Mip-NeRF (Bigger MLP)**: 强行把网络放大到 9.0M params, 22.71 hours, PSNR 26.19。这证明纯粹堆参数收益递减，训练极慢。
*   **Our Model (Mip-NeRF 360)**: 9.9M params (和 Bigger 差不多), 6.89 hours (只比原版慢 2 倍), PSNR 27.69。
**结论：** Online Distillation 架构完美解决了大容量与计算效率的矛盾。

看 Table 2 的消融实验：
*   去掉 Proposal Loss ($\mathcal{L}_{\mathrm{prop}}$): PSNR 暴跌到 20.49。实习生没人管就彻底乱来。
*   用 NeRF 原版的噪声注入替代 Distortion Loss: PSNR 24.00，视觉效果一塌糊涂。证明老办法管不了新场景。

### 相关联想与启发

Mip-NeRF 360 的这套 Proposal MLP + Online Distillation 机制，彻底改变了 NeRF 的训练范式。它把“用一个大网络做所有事”变成了“用小网络做重要性采样，用大网络做精细渲染”。这个思想直接启发了后来大红大紫的 **Instant-NGP** 和 **Zip-NeRF**，它们都采用了类似的 proposal sampler 架构来加速训练。

**Web Links Reference:**
*   **Mip-NeRF 360 Project Page:** [https://jonbarron.info/mipnerf360/](https://jonbarron.info/mipnerf360/)
*   **Mip-NeRF (前作):** [https://jonbarron.info/mipnerf/](https://jonbarron.info/mipnerf/)
*   **NeRF (原论文):** [https://www.matthewtancik.com/nerf](https://www.matthewtancik.com/nerf)
*   **NeRF++ (Unbounded 场景另一思路):** [https://github.com/Kai-46/nerfplusplus](https://github.com/Kai-46/nerfplusplus)
*   **DONeRF (Depth Oracle Network, 类似 Proposal 思想):** [https://research.nvidia.com/publications/2021-06_DONeRF](https://research.nvidia.com/publications/2021-06_DONeRF)

---

Andrej, 很高兴能与你深入探讨这篇 Mip-NeRF 360。作为将 NeRF 发扬光大的关键人物，你对基于坐标的 MLP 表示 3D 场景的局限性有着深刻的直觉。Mip-NeRF 360 这篇 paper 的核心目的，就是解决当相机在 360 度无界场景中自由运动时，原始 NeRF 和 Mip-NeRF 所遭遇的 capacity 瓶颈、采样效率低下 以及 ill-posedness 导致的 artifact。

我会按照论文的逻辑，结合底层数学和几何直觉，为你详细拆解三大核心创新点：Scene Parameterization, Online Distillation, 以及 Distortion Regularizer。同时我会尽量发散，涵盖相关的技术联想。

---

### 1. Parameterization: 从 Euclidean Space 到 Contraction Space

在原始 NeRF 中，针对前向场景使用了 NDC (Normalized Device Coordinates) 来将无限远的 z 轴压缩到 $[-1, 1]$ 之间。NDC 的本质是按照 disparity (视差，即深度的倒数) 进行分配，从而让远处的背景占据较少的 MLP capacity。但在 360 度场景中，相机朝向四面八方，单纯在 z 轴上做 NDC 是无效的。

Mip-NeRF 360 提出了一种基于空间的非线性 contraction 机制。

**公式解析与直觉:**

定义 contraction 函数 $f(\mathbf{x})$ (对应论文 Eq. 10):
$$
\mathrm{contract}(\mathbf{x}) = \begin{cases} 
\mathbf{x} & \|\mathbf{x}\| \leq 1 \\ 
\left(2 - \frac{1}{\|\mathbf{x}\|}\right) \left(\frac{\mathbf{x}}{\|\mathbf{x}\|}\right) & \|\mathbf{x}\| > 1 
\end{cases}
$$
*   **变量解释:** $\mathbf{x}$ 是 Euclidean 空间中的 3D 坐标。$\|\mathbf{x}\|$ 是该点到原点(通常设为相机环绕的中心)的距离。
*   **直觉:** 对于半径 $\le 1$ 的单位球内部，空间保持原状(线性)；对于单位球外部，方向向量 $\frac{\mathbf{x}}{\|\mathbf{x}\|}$ 保持不变，但半径被映射为 $2 - \frac{1}{\|\mathbf{x}\|}$。当 $\|\mathbf{x}\| \to 1$ 时，半径趋于 $1$；当 $\|\mathbf{x}\| \to \infty$ 时，半径趋于 $2$。这样就把无穷大的空间压缩到了一个半径为 2 的球体内。这与 NeRF++ 的 inverse sphere 思想类似，但 Mip-NeRF 360 提出了一种平滑的、可微的统一映射，避免了内外两个 MLP 拼接导致的 seam (接缝) 问题。

**Kalman Filter 般的 Gaussian Reparameterization:**

Mip-NeRF 的核心在于它不处理点，而是处理 conical frustum (圆锥台) 近似出的 3D Gaussian $(\mu, \Sigma)$。当我们对空间施加非线性变换 $f(\mathbf{x})$ 时，如何变换 Gaussian？

论文使用了 Extended Kalman Filter (EKF) 中的线性化技术 (对应论文 Eq. 8, 9):
$$
f(\mathbf{x}) \approx f(\mu) + \mathbf{J}_f(\mu)(\mathbf{x} - \mu)
$$
$$
f(\mu, \Sigma) = \left( f(\mu), \mathbf{J}_f(\mu) \Sigma \mathbf{J}_f(\mu)^{\mathrm{T}} \right)
$$
*   **变量解释:** $\mu$ 是 Gaussian 的均值，$\Sigma$ 是协方差矩阵。$\mathbf{J}_f(\mu)$ 是非线性函数 $f$ (即 contract) 在 $\mu$ 处的 Jacobian 矩阵。
*   **直觉:** 非线性变换会破坏 Gaussian 的性质。我们只能在局部用线性近似。变换后的均值直接取 $f(\mu)$，而变换后的协方差则通过 Jacobian 矩阵进行线性旋转变换。这使得我们可以在 contracted space 中计算 Integrated Positional Encoding (IPE)。

**Disparity-based Ray Sampling:**

除了空间坐标，沿射线的距离 $t$ 也需要重新参数化。论文定义了从 $t$ 到 normalized distance $s \in [0, 1]$ 的映射 (对应论文 Eq. 11):
$$
s \triangleq \frac{g(t) - g(t_n)}{g(t_f) - g(t_n)}, \quad t \triangleq g^{-1}(s \cdot g(t_f) + (1-s) \cdot g(t_n))
$$
*   **变量解释:** $t_n, t_f$ 是 near/far plane。$g(\cdot)$ 是可逆标量函数。如果设 $g(x) = 1/x$，则 $t$ 空间中的均匀采样对应于 disparity 空间中的均匀采样。
*   **直觉:** 在 360 场景中，远处的内容在图像上占比极小，如果用线性 $t$ 采样，大量 sample points 会浪费在空旷的背景上。使用 disparity 采样使得近处密集、远处稀疏，恰好与 contraction 空间形成对偶，配合起来让 MLP 在 contracted 空间内看到的是各向同性的采样分布。

---

### 2. Efficiency: Proposal MLP 与 Online Distillation

NeRF 和 Mip-NeRF 使用 coarse-to-fine 策略：先用 uniform 采样得到 coarse weight $w^c$，再根据 $w^c$ 重新采样得到 fine samples。这里最大的 wastage 在于：coarse MLP 的渲染结果并不参与最终图像输出，但其网络结构和 fine MLP 一样大，消耗了大量 capacity 和计算资源。

Mip-NeRF 360 借鉴了知识蒸馏的思想，但将其做成了 "online" 的形式。

**架构设计:**
*   **Proposal MLP $\Theta_{\mathrm{prop}}$:** 体积小 (4 层, 256 hidden units)。只输出 density $\tau$，不输出 color。被多次评估 (论文中是 2 次，每次 64 个 samples)，产生 proposal weights $\hat{w}$。
*   **NeRF MLP $\Theta_{\mathrm{NeRF}}$:** 体积大 (8 层, 1024 hidden units)。输出 density $\tau$ 和 color $c$。只评估一次 (32 个 samples)，产生最终 weights $w$ 和颜色 $c$。

**Supervision 机制与 Loss 设计:**

Proposal MLP 不直接监督图像渲染，它的任务是 "包住" NeRF MLP 产生的 weights 直方图。这引出了一个很有趣的 histogram bound loss (对应论文 Eq. 12, 13):

$$
\operatorname{bound}(\hat{\mathbf{t}}, \hat{\mathbf{w}}, T) = \sum_{j: T \cap \hat{T}_j \neq \emptyset} \hat{w}_j
$$
$$
\mathcal{L}_{\mathrm{prop}}(\mathbf{t}, \mathbf{w}, \hat{\mathbf{t}}, \hat{\mathbf{w}}) = \sum_i \frac{1}{w_i} \max\left(0, w_i - \operatorname{bound}(\hat{\mathbf{t}}, \hat{\mathbf{w}}, T_i)\right)^2
$$

*   **变量解释:** $T_i$ 是 NeRF MLP 产生的第 $i$ 个区间。$\operatorname{bound}$ 函数计算 Proposal MLP 中所有与 $T_i$ 有交集的区间 $\hat{T}_j$ 的权重 $\hat{w}_j$ 之和。
*   **直觉:** 为什么是上界？因为 Proposal MLP 是粗略的，它的直方图 bins 必然比 NeRF MLP 宽。如果 NeRF 在某个小区间 $T_i$ 内有很大的 weight $w_i$，那么 Proposal MLP 中所有覆盖到 $T_i$ 的粗 bins 的权重之和，**必须大于等于** $w_i$。如果小于，说明 Proposal 漏掉了重要内容，需要被惩罚。这个 loss 是 asymmetric 的，因为 Proposal MLP 的权重高估是无害的(最多就是采样时多采一些空区域)，低估则是致命的(漏掉表面)。
*   **Stop-gradient:** 计算 $\mathcal{L}_{\mathrm{prop}}$ 时，NeRF MLP 输出的 $(\mathbf{t}, \mathbf{w})$ 被 stop-gradient。NeRF MLP 负责 "lead" (通过 photometric loss 学习真实结构)，Proposal MLP 负责 "follow" (模仿 NeRF 的分布)。这避免了 NeRF MLP 为了让 Proposal MLP 容易学习而退化自己的表示。

---

### 3. Ambiguity: Distortion Regularizer 解决 Floaters 与 Background Collapse

NeRF 的 ill-posedness 会导致两种常见的 artifact:
1.  **Floaters:** 悬浮在空中的半透明密度云。因为在某些训练视角下，这些悬浮物恰好能填补背景的像素漏洞。
2.  **Background Collapse:** 远处的背景被错误地建模为靠近相机的半透明层。这通常是因为 MLP 的有限 capacity 无法表示无限远的细节，从而贪心地"拉近"表面来拟合像素。

原始 NeRF 通过在 density 输出前注入 Gaussian noise 来鼓励 binary opacity (要么完全透明，要么完全不透明)，但这对于复杂场景不够。Mip-NeRF 360 提出了基于 interval 距离的 distortion loss (对应论文 Eq. 14, 15):

$$
\mathcal{L}_{\mathrm{dist}}(\mathbf{s}, \mathbf{w}) = \iint_{-\infty}^{\infty} \mathbf{w_s}(u) \mathbf{w_s}(v) |u - v| d_u d_v
$$
展开后的计算形式:
$$
\mathcal{L}_{\mathrm{dist}}(\mathbf{s}, \mathbf{w}) = \sum_{i,j} w_i w_j \left|\frac{s_i + s_{i+1}}{2} - \frac{s_j + s_{j+1}}{2}\right| + \frac{1}{3} \sum_i w_i^2 (s_{i+1} - s_i)
$$

*   **变量解释:** $\mathbf{s}$ 是 normalized ray distances (用 $s$ 而不是 $t$ 是为了平衡远近权重)。$\mathbf{w_s}(u)$ 是通过 $(\mathbf{s}, \mathbf{w})$ 定义的 step function。$w_i, w_j$ 是第 $i, j$ 个区间的 alpha compositing weights。$s_i, s_{i+1}$ 是第 $i$ 个区间的端点。
*   **直觉:** 这个公式本质上是计算整个射线上一维分布的 "moment of inertia" (转动惯量) 或者类似于 k-means 中的 distortion measure。
    *   第一项 $\sum_{i,j} w_i w_j |mid_i - mid_j|$ 惩罚两两区间中点之间的距离。如果 weight 分散在不同的深度，这个值会很大。为了最小化它，所有的 weight 必须在 $s$ 空间中尽量靠拢。
    *   第二项 $\frac{1}{3} \sum_i w_i^2 (s_{i+1} - s_i)$ 惩罚每个区间的宽度。如果某个区间的 weight 很大，它的宽度必须很窄(即表面必须 sharp)。
*   **效果:** 这个 regularizer 强制射线上的密度分布尽可能像 Dirac delta function(单点脉冲)。如果射线没击中任何物体，最优解是 $w=0$；如果击中了，最优解是在表面处有一个极窄的尖峰。完美消除了 floaters(离散的多个尖峰)和 background collapse(宽泛的半透明区域)。

---

### 4. 细节与联想: Off-Axis IPE 与实现 Trick

除了三大主轴，论文的附录里藏着一些极具工程价值的技术点:

**Off-Axis Positional Encoding:**
Mip-NeRF 的 IPE 使用 identity matrix 作为基，只提取 $\Sigma$ 的对角线。但在 contraction space 中，由于 Jacobian 的扭曲，Gaussian 不可避免地变成强烈的 anisotropic (各向异性)。对角线 IPE 会丢失方向信息。论文 Appendix A 引入了一个基于 icosahedron (二十面体) 顶点的 skinny matrix $\mathbf{P}$。通过计算 $\operatorname{diag}(\mathbf{P}\Sigma\mathbf{P}^{\mathrm{T}})$，可以在不过度增加计算量的前提下，捕捉到 Gaussian 的方向特征。这非常像 Random Fourier Features 的思想。

**Histogram Resampling 修正:**
原版 Mip-NeRF 从 coarse histogram 采样 $n+1$ 个点作为 fine interval 的端点。这会 "erode" (腐蚀) 直方图的边缘。Mip-NeRF 360 改为采样 $n+1$ 个点，然后取相邻点的 midpoints 作为 $n$ 个区间的端点。这是一个很小但很关键的 anti-aliasing 改进。

**Annealing 与 Dilation:**
训练初期，Proposal MLP 的输出 $\hat{w}$ 很不稳定。论文使用 Schlick's bias 函数对 $\hat{w}$ 做退火: $\hat{w}_n \propto \hat{w}^{\frac{bn/N}{(b-1)n/N+1}}$。初期使其趋向均匀分布，后期恢复真实分布。此外，对 proposal histogram 进行 dilation (膨胀，类似 max-pooling)，可以缓解 Proposal MLP 只在特定视角有效导致的 rotational aliasing。

---

### 5. Results 与实验数据直觉

在定量指标上 (Table 1):
*   相比 Mip-NeRF (PSNR 24.04)，Mip-NeRF 360 达到了 27.69 (MSE 下降 57%)。
*   训练时间从 3.17 小时增加到 6.89 小时 (仅 2x 放缓)，但模型 capacity 是 9.9M params vs 0.7M params (15x 增加)。
*   如果强行把 Mip-NeRF 的 MLP 变大 (9.0M params)，训练时间暴涨到 22.71 小时，但 PSNR 只能达到 26.19。这证明了 Proposal MLP 架构的巨大成功。

在消融实验中 (Table 2):
*   去掉 $\mathcal{L}_{\mathrm{prop}}$，PSNR 从 24.37 暴跌到 20.49，因为 Proposal MLP 没有监督会随机发散。
*   去掉 $\mathcal{L}_{\mathrm{dist}}$，PSNR 几乎不变 (24.41)，但视觉上会出现严重的 floaters。这印证了 distortion loss 主要是在解决 ill-posedness artifact，而不是提升像素拟合能力。

---

### Web Links Reference
*   **Mip-NeRF 360 Project Page:** [https://jonbarron.info/mipnerf360/](https://jonbarron.info/mipnerf360/)
*   **Mip-NeRF (前作):** [https://jonbarron.info/mipnerf/](https://jonbarron.info/mipnerf/)
*   **NeRF (原论文):** [https://www.matthewtancik.com/nerf](https://www.matthewtancik.com/nerf)
*   **NeRF++ (Unbounded 场景另一思路):** [https://github.com/Kai-46/nerfplusplus](https://github.com/Kai-46/nerfplusplus)
*   **DONeRF (Depth Oracle Network, 类似 Proposal 思想):** [https://research.nvidia.com/publications/2021-06_DONeRF](https://research.nvidia.com/publications/2021-06_DONeRF)

总结来说，Mip-NeRF 360 是一篇将数学、物理直觉与深度学习架构结合得极其精妙的 paper。它没有盲目堆砌网络参数，而是深挖了 NeRF 采样与渲染过程中的 information flow bottleneck。Proposal MLP 的 online distillation 机制实际上启发了后来诸如 Instant-NGP 中 proposal network sampler 的设计，而 distortion loss 也成为了后续诸多 NeRF 变体处理 ill-posedness 的标准配置。
