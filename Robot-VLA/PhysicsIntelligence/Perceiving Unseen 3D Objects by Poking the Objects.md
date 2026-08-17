---
source_pdf: Perceiving Unseen 3D Objects by Poking the Objects.pdf
paper_sha256: ca01ace7fbc20effe5a064f6e627d796f6fcfce8d9ecde04bdc6108899b7dbf1
processed_at: '2026-08-06T02:41:28-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 paper

## 一句话概括

**robot 进新房间，啥也不认识，怎么办？戳一戳，戳完就认识了。**

---

## 场景想象

想象你被蒙着眼睛带进一个陌生房间，摘下眼罩后看到桌上摆着几个东西。你不知道它们叫什么、是什么材质、能不能动。你会怎么做？

最自然的反应就是 **伸手戳一下**。

戳完你立刻知道了三件事：
1. 这个东西能动 → 它是个独立 object，不是桌子的一部分
2. 它翻了个面 → 你看到了之前看不到的背面
3. 它摔倒了 → 你看到了不同角度的样子

这篇 paper 就是让 robot 干这个事。

---

## 传统方法为什么不行

以前让 robot 认东西，两条路：

**第一条路**：提前给它一个 3D model 库，"这是杯子，这是猫，这是鸭子"。robot 拿新图像去匹配。问题是你不可能预知所有物体，进了新环境就傻眼了。

**第二条路**：喂几万张标注好的图像训练神经网络。问题是标注数据太贵，而且只认得训练过的类别，没见过的还是不认识。

两条路都死在 **"unseen"** 这个词上。

---

## 这篇 paper 的思路

非常简单，分三步：

### Step 1: 戳

robot 看到桌上有一堆东西，先 guess 哪些可能是 object（用 point cloud 聚类，假设物体都放在桌面上）。然后机械臂挨个戳。

戳得动的 → 是 object，继续戳，转一圈戳 4 次，相机录下整个过程。

戳不动的 → 不是 object，丢掉。

### Step 2: 建

戳的过程中相机拍了一堆 RGB-D 图像。这些图像天然是 multi-view 的，因为物体被戳得转来转去。

然后用 NeRF 把这些多视角图像合成一个 3D model。这步就是 reconstruction。

### Step 3: 记

建好 3D model 后，用这个 model 渲染出一万张不同角度的假图像，拿这些假图像训练一个 pose estimation 网络（PVNet）。

下次再看到这个物体，网络直接一步前向传播就知道它在哪、朝哪个方向。

---

## 为什么戳比抓好

你可能会问，为什么不直接抓起来看？

作者列了几个原因，用人话说：

1. **抓需要训练**：训练抓取的网络只见过训练集里的物体，新物体可能抓不住甚至抓碎
2. **有些东西太大抓不动**：比如一个大纸箱
3. **抓会挡住物体**：手一抓，一半被遮住了，重建出来缺一块

而戳几乎没这些问题：什么都能戳，戳完不挡，还能筛选掉那些戳不动的"假物体"（比如粘在桌上的东西）。

---

## 技术核心的"人话版"

### Decomposed NeRF 是啥意思

场景里有 background（桌子、墙壁）和 foreground（猫、鸭子、纸盒）。如果用一个大 NeRF 表示整个场景，那物体一动整个场景就变了，NeRF 搞不定。

所以拆开：background 一个 NeRF（静态的，不用太精确），每个 object 一个 VolSDF（动态的，需要精确几何）。

每个 object 有自己的 pose $\xi_t^k$，表示在第 t 帧时它在哪。object 本身的 shape 在它自己的 local coordinate 里是固定的，只是被 pose 搬来搬去。

### VolSDF 和 NeRF 的区别

NeRF 用 density 表示"这个点有多不透明"，但没有 surface 的概念，重建出来的 mesh 往往 fuzzy、有 floaters。

VolSDF 用 signed distance 表示"这个点离表面有多远"，有明确的 surface（distance = 0 的地方），几何质量高得多。

用人话：NeRF 像画水彩，边界模糊；VolSDF 像素描，轮廓清晰。

### Sparsity loss 在干嘛

物体放在桌上，contact 区域很难分清谁是物体谁是桌子。特别是桌子没 texture 的时候，network 搞不清是物体在动还是桌子在动。

sparsity loss 的意思是：**"background 表面后面不许有物体"**。

具体做法是找到 background 表面的 depth $z_m$，凡是 depth 比 $z_m$ 还大的点（即 background 后面的点），强抑制 object 的 density。

用人话：告诉网络"桌子后面不会有猫，别在那瞎填"。

这个 loss 的权重 $w_4 = 2 \times 10^{-5}$ 特别小，因为只是个 auxiliary hint，不能喧宾夺主。

### Stage-wise training 为什么必要

如果一上来就 joint optimize pose 和 geometry，pose 还没估准，geometry 就跟着乱跑；geometry 乱跑，pose 更估不准。死循环。

所以分三步走：
1. 先不管物体，把 background 建好（10k iterations）
2. 固定 pose，先把 object 的 initial shape 建出来（10k iterations）
3. 再 joint refine 所有东西（50k iterations）

用人话：先各自练好基本功，再合练。

---

## 实验效果用人话说

### Pose accuracy

cat 和 duck 的 rotation error 从 ~14° 降到 ~4°，max error 从 ~30° 降到 ~8°。

为什么提升这么大？因为 MaskFusion（baseline）是逐帧用 ICP tracking，error 会 accumulate，越到后面越歪。这篇 paper 是所有帧 joint optimize，globally consistent，不会越走越偏。

用人话：MaskFusion 像"走一步看一步"，走着走着就歪了；这篇 paper 像"走完回头看全程 GPS 轨迹重新校正"。

### Geometry quality

Chamfer Distance 降了 3-5 倍，Normal Consistency 从 0.58 提升到 0.85。

Normal Consistency 高意味着 surface normal 准确，这对 grasping 很关键 —— 你得知道物体表面朝哪才能正确下夹爪。

### Ablation

去掉 stage-wise training → error 暴增 4 倍（4° → 18°）
去掉 sparsity loss → 物体和背景粘成一坨，分不开
去掉 foreground sampling → optimization 抓不住重点，结果粗糙

---

## Memorization 那步在干嘛

重建完 3D model 后，问题变成：下次看到这个物体怎么认出来？

最直接的办法：用重建的 model 渲染一万张假图像（不同角度、不同背景），训练 PVNet。

用人话：**"我建了你的 3D 模型，然后合成一堆你在各种角度的照片，拿这些照片训练一个识别网络。下次再见到你，网络一眼就认出来。"**

这步的聪明之处在于 **完全不需要人工标注** —— training data 全是 render 出来的，background 从 ScanNet 随机选，让网络有 generalization 能力。

---

## Grasping 怎么用

有了 reconstructed model + pose estimation network，grasping 就是拼 transformation：

$$T_{gb} = T_{go} \cdot T_{oc} \cdot T_{cb}$$

- $T_{go}$：gripper 相对 object 的 pose（Graspit! 在 3D model 上 planning）
- $T_{oc}$：object 相对 camera 的 pose（PVNet 估计）
- $T_{cb}$：camera 相对 robot base 的 pose（hand-eye calibration）

三个 matrix 乘起来就是 gripper 相对 robot base 的 pose，机械臂照着走就行。

用人话：**"我知道物体长啥样 → 我能在模型上规划怎么夹；我知道物体在哪 → 我能让机械臂找对位置。两个信息一拼，搞定。"**

---

## 局限性

作者自己承认：

1. **glossy/transparent 表面 depth 扫不准** → 像 glass cup 这种就难搞
2. **慢** → reconstruction 几十分钟，memorization 训练 PVNet 也要时间

我补充几个：

3. **fragile objects** → 戳一下碎了怎么办
4. **heavy objects** → 戳不动就当成"非物体"丢了
5. **tethered/fixed objects** → 比如插在插座上的充电线，戳不动但确实是物体
6. **scalability** → 3 个物体还行，30 个物体要戳 120 次，时间成本线性增长

---

## 和最新工作的联系

这篇是 2022 年的工作，现在看可以有很多升级：

- NeRF → **3D Gaussian Splatting**：速度快 100 倍，质量相当
- PVNet + render 训练 → **OnePose**：一个 reference video 就行，不用训练
- Graspit! analytic planning → **Diffusion Policy**：learned grasping，更 general
- Poking heuristic → **LLM-guided poking**：用大模型 reasoning 决定戳哪里

---

## 最核心的 intuition

如果只记住一件事，记住这个：

**当你不知道一个东西是什么的时候，碰它一下。它动了就是 object，动了的过程就是 data，data 就是 model，model 就是 knowledge。**

这篇 paper 把 active perception 的哲学用一个最简单的 action（poking）实现了，然后串起了一条完整的 pipeline：**action → data → model → recognition → application**。

这个 pipeline 的美妙之处在于 **完全 unsupervised**，不需要人类标注，不需要预知 model，不需要训练数据集。robot 自己探索、自己建模、自己记住。这是迈向真正 autonomous robot 的一小步，但方向很对。

---

# Perceiving Unseen 3D Objects by Poking the Objects - 深度解析

## 一、核心 motivation 与 high-level intuition

这篇 paper 解决的核心问题是：**robot 进入一个全新的环境，如何对从未见过的 3D objects 进行感知、重建并记住它们**。传统方法要么依赖预知的 object models（model-based），要么依赖大量标注数据（learning-based），都难以泛化到 unseen objects。

作者的核心 insight 是模仿人类的行为：**通过 interaction 来打破 ambiguity**。人类看到一个静止的物体时，往往难以判断它是否是独立物体、是否能移动、boundary 在哪里；但轻轻戳一下，所有这些 ambiguity 瞬间消失 —— 能动的就是独立物体，运动产生的 multi-view observations 自然可以用于重建。

这背后的哲学类似于 **active perception**：perception 和 action 不应该分离，action 本身是 perception 的信息源。poking 这个 action 同时完成了三件事：
1. **discovery**：验证 object proposal 是否真的是独立可移动物体
2. **multi-view data collection**：通过物体运动产生多视角观测
3. **occlusion reduction**：poking 可以把被遮挡的物体暴露出来

项目主页：https://zju3dv.github.io/poking_perception/

---

## 二、Poking 机制的设计细节

### 2.1 Object proposal generation

由于 poking 搜索空间无限大，作者先用 geometric prior 来 narrow down 候选区域：
- **plane segmentation**：假设物体放在平面上
- **point cloud clustering**：对平面上方的 point cloud 进行聚类得到 object proposals

这一步类似于 object detection 中的 **Region Proposal Network (RPN)** 思想 —— 先粗筛再精检。

### 2.2 Poking trajectory design

poking 的 trajectory 设计原则：
1. 确保 object 被从足够多的 viewpoints 观测到
2. 避免 robot arm 造成 occlusion
3. 通过 clockwise direction 的多次 poking iteration 实现

每个 object 执行 **4 次 poking action**（empirically enough for sufficient view coverage）。

### 2.3 为什么不用 grasping？

作者的 discussion 部分非常有 intuition，对比了 grasping-based discovery 的局限：
- learning-based grasp detection 受限于 training domain，对 unseen objects 可能 fail 甚至损坏 fragile objects
- 某些 objects 太大无法 grasp
- grasping 会 occlude object，导致 reconstruction 不完整

而 poking 的优势：
- 对 object category 和 size 无限制
- occlusion 极小（只是 arm 瞬间接触）
- 能 prune immovable proposals

这个设计选择让我联想到 **minimalism in robotics** —— 用最简单、最通用的 action 来获取 maximum information。

---

## 三、Decomposed Neural Radiance Fields - 技术核心

这是论文最 technical 的部分，也是真正有意思的地方。让我深入讲解。

### 3.1 为什么需要 decomposed representation？

场景中包含 background + multiple moving objects。单个 NeRF 只能表示静态场景，因此需要把场景 decompose 成多个 sub-field：
- $F_{\Theta}^{b}$：background NeRF
- $F_{\Theta}^{k}$：第 k 个 object 的 VolSDF
- $\xi_{t}^{k} \in \mathfrak{se}(3)$：第 k 个 object 在 frame t 的 pose

### 3.2 为什么 object 用 VolSDF 而不是 NeRF？

这是关键设计选择。NeRF 用 density $\sigma$ 表示 occupancy，但 **没有 surface constraint**，导致重建的几何质量差（floaters、模糊边界）。VolSDF 用 **Signed Distance Function (SDF)** 表示表面，天然有几何约束。

公式 (3) 和 (4) 展示了 SDF 到 density 的转换：

$$\sigma(\mathbf{x})^k = \begin{cases} 
\frac{1}{\beta}\left(1 - \frac{1}{2}\exp\left(\frac{d(\mathbf{x})^k}{\beta}\right)\right) & \text{if } d(\mathbf{x})^k < 0 \\
\frac{1}{2\beta}\exp\left(-\frac{d(\mathbf{x})^k}{\beta}\right) & \text{if } d(\mathbf{x})^k \geq 0 
\end{cases}$$

变量解释：
- $d(\mathbf{x})^k$：point x 到第 k 个 object 表面的 signed distance（负值表示在物体内部，正值表示在外部）
- $\beta$：learnable temperature parameter，控制 density 分布的 sharpness
- $\sigma(\mathbf{x})^k$：转换后的 volume density

这个转换的 intuition：
- 当 $d < 0$（在物体内）：density 接近 $1/\beta$（高密度）
- 当 $d = 0$（在表面）：density $= 1/(2\beta)$
- 当 $d > 0$（在物体外）：density 按 $\exp(-d/\beta)$ 衰减

**$\beta$ 越小，surface 越尖锐**，随着训练进行 $\beta$ 会逐渐减小，surface 越来越清晰。这是 Laplacian CDF 的变体，源自 VolSDF [31]。

### 3.3 Coordinate transformation

公式 (3) 中有个关键变换：$\mathbf{x_o} = (\xi_t^k)^{-1}\mathbf{x}$

这把 world coordinate 的 point x 变换到 object 的 canonical coordinate。**这是处理动态物体的核心**：object 在运动，但 object 的 intrinsic geometry 在 canonical frame 中是静态的。通过这个变换，不同 frame 中同一 object 的不同位置都能 share 同一个 VolSDF 网络。

这和 **NeRF-W 的 appearance embedding**、**NeRF for dynamic scenes 的 deformation field** 思想类似，但这里用的是 rigid transformation（更简单、更精确）。

### 3.4 Composite volume rendering

公式 (5) 和 (6) 是 composite rendering：

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \left(\alpha_i^b \mathbf{c}_i^b + \sum_{k=1}^{K} \alpha_i^k \mathbf{c}_i^k\right)$$

$$\hat{D}(\mathbf{r}) = \sum_{i=1}^{N} T_i \alpha_i \mathbf{d}_i$$

变量解释：
- $T_i = \exp\left(-\sum_{j=1}^{i-1}\bar{\sigma}_j \delta_j\right)$：accumulated transmittance（光线到达 point i 时剩余的能量）
- $\bar{\sigma}_i = \sigma_i^b + \sum_{k=1}^K \sigma_i^k$：所有 radiance fields 的 composed density
- $\alpha_i = 1 - \exp(-\bar{\sigma}_i \delta_i)$：point i 处的 opacity
- $\alpha_i^k = \frac{\sigma_i^k}{\bar{\sigma}_i}\alpha_i$：第 k 个 object 在 point i 处的相对 opacity 贡献
- $\alpha_i^b = \frac{\sigma_i^b}{\bar{\sigma}_i}\alpha_i$：background 的相对 opacity 贡献

**核心 intuition**：多个 radiance fields 的 density 是 additive 的，opacity 按 density 比例分配。这意味着如果某个 point 同时被 object 和 background claim，它们的 color 会按 density ratio 混合。这种设计允许 network 隐式地 learn segmentation mask —— 哪个 field 在哪个 point 应该有高 density，完全由数据驱动。

这个思想来自 **Neural Scene Graphs [34]** 和 **STaR [37]**，但本文把它和 SDF-based representation 结合，获得更好的几何质量。

### 3.5 为什么 background 用 NeRF 而 object 用 VolSDF？

这是一个 asymmetry 的设计选择，背后的 intuition：
- **Background**：静态、不需要精确几何、只需要 photometric consistency → NeRF 足够
- **Object**：需要精确几何用于 downstream grasping、需要 surface normal 用于 grasp planning → VolSDF 必要

这种 asymmetric design 在工程上很聪明 —— 把 "好钢用在刀刃上"。

---

## 四、Loss Functions 的设计 intuition

### 4.1 基础 loss

$$\mathcal{L}_c = \|\hat{C}(\mathbf{r}) - C(\mathbf{r})\|, \quad \mathcal{L}_d = \|\hat{D}(\mathbf{r}) - D(\mathbf{r})\|$$

color loss 和 depth loss 都用 1-norm（L1），比 L2 更 robust to outliers。

### 4.2 Eikonal loss

$$\mathcal{L}_{sdf} = \mathbb{E}_z(\|\nabla d(z)\| - 1)^2$$

这是 SDF 的 **正则化约束**：SDF 的 gradient magnitude 必须等于 1（这是 signed distance function 的数学定义）。变量 $z$ 是随机采样的 3D points，$\nabla d(z)$ 是 SDF 在点 z 处的 spatial gradient。

没有这个 loss，network 可能 learn 出一个不是真正 SDF 的 function（只是一个 arbitrary scalar field），导致 marching cubes 提取的 mesh 质量差。这个 loss 来自 IGR (Implicit Geometric Regularization) [39]。

### 4.3 Sparsity loss - 最关键的 design

这是论文的 **核心创新** 之一：

$$\mathcal{L}_{sp} = w_{sp}|1 - \exp(-\sigma_i)|$$

其中：
$$w_{sp} = \exp(-\mathbf{w} \cdot \max(z_m - z_i, 0))$$

变量解释：
- $\sigma_i$：point $x_i$ 处 object VolSDF 的 density
- $z_i$：point $x_i$ 的 depth
- $z_m = \max_t\{D_r^t\}$：ray r 在所有 frames 中的 maximum depth（即 background surface 的 depth）
- $\mathbf{w}$：weight decay parameter（设为 200）

**Intuition 解析**：

这个 loss 的目的是 **suppress object VolSDF 在 background 表面及更远区域的 density**。为什么需要这个？

问题：object 和 background 在 contact 区域（物体放在桌面上）难以分解，特别是 textureless background 有 motion ambiguity（不知道是物体在动还是 background 在动）。

解决方案的设计非常巧妙：
- $z_m - z_i > 0$：point 在 background 表面前方 → $w_{sp}$ 小 → sparsity loss 权重小（不抑制，因为可能是 object）
- $z_m - z_i < 0$：point 在 background 表面后方 → $\max(z_m - z_i, 0) = 0$ → $w_{sp} = 1$ → sparsity loss 权重大（强抑制）

这个设计让我想起 **PlenOctrees [40]** 中的 sparsity prior，但本文把它和 background depth 结合，做成 **depth-aware sparsity**，非常 elegant。

### 4.4 Total loss

$$\mathcal{L} = w_1\mathcal{L}_c + w_2\mathcal{L}_d + w_3\mathcal{L}_{sdf} + w_4\mathcal{L}_{sp}$$

权重设置：$w_1=1, w_2=1, w_3=0.1, w_4=2\times10^{-5}$

注意 $w_4$ 非常小，因为 sparsity loss 是 auxiliary regularizer，不能主导 optimization。

---

## 五、Sampling Strategy 和 Training Strategy

### 5.1 Foreground sampling

$$N_r \text{ pixels: } N/2 \text{ within object mask} + N/2 \text{ over entire image}$$

**Intuition**：object region 相对整个 image 很小，如果 uniform sampling，大部分 rays 都浪费在 background 上。但也不能只 sample object region，否则 background NeRF 训练不好。50-50 是一个平衡。

这和 **NeRF 的 coarse-to-fine sampling**、**Mip-NeRF 的 cone sampling** 思想不同 —— 这里是 spatial importance sampling。

### 5.2 Robot mask exclusion

Poking 过程中 robot arm 和 object 接触，难以分解。作者的解决方案很 pragmatic：**直接用 robot arm 的 URDF model render 出 mask，不 sample 这些 pixels**。

这是 robotics + vision 结合的优势 —— 你知道 robot 的精确 kinematics，可以利用这个 prior。

### 5.3 Stage-wise training

三阶段训练策略避免 local optima：

**Stage 1**（10k iterations）：只训练 background NeRF，sample outside robot mask + object mask
- 目的：建立 clean background model

**Stage 2**（10k iterations）：只训练 object VolSDF，fix object pose
- 目的：在已知 pose 下建立 object 的 initial geometry

**Stage 3**（50k iterations）：joint optimize everything
- 目的：fine-tune 所有参数

**Intuition**：joint optimization from scratch 容易陷入 local optima（pose 和 geometry 互相干扰）。先 establish 各自的 initial estimate，再 joint refine，类似于 **coordinate descent** 的思想。

### 5.4 Initialization

- **Object mask**：optical flow norm > threshold 的 pixels
- **Object pose**：scene flow within mask + Least-Squares estimation + ICP refinement

这个 initialization 利用了 **motion cue** —— 运动的 pixels 就是 object，scene flow 直接给出 motion field，可以 regression 出 rigid pose。

---

## 六、Memorization - 从 reconstruction 到 recognition

### 6.1 PVNet 训练

重建完 object model 后，如何让 robot 在新图像中 recognize 它？作者用 **PVNet [4]** 作为 example：

1. 从 reconstructed model render 10000 张 training images
2. Object poses 采样在 30 个 semi-spheres 上（不同距离）
3. Background 用 ScanNet [43] 的真实场景图像
4. 同时用 recorded video 中的真实 frames 训练（增加 generalization）

PVNet 的核心：pixel-wise voting 预测 2D keypoints，然后用 PnP [42] 解出 6DoF pose。

### 6.2 Inference refinement

推理时用 ICP refine pose：align reconstructed model 和 depth image backprojected 的 point cloud。

**Intuition**：PVNet 给出 RGB-based 的 pose estimate，ICP 用 depth 信息 refine。这是 **hybrid RGB-D pose estimation** 的经典做法。

参考链接：
- PVNet: https://github.com/zju3dv/pvnet
- OnePose (后续工作，无需 CAD model): https://github.com/zju3dv/OnePose

---

## 七、实验结果深度分析

### 7.1 Object pose accuracy（Table I）

| Object | Method | Rotation (deg) | Translation (cm) |
|--------|--------|---------------|------------------|
| cat | MF | 11.914 / 30.074 | 1.676 / 4.684 |
| cat | Ours | 4.391 / 8.003 | 0.452 / 1.168 |
| duck | MF | 14.144 / 31.871 | 3.728 / 8.212 |
| duck | Ours | 4.070 / 12.743 | 1.116 / 3.388 |

**Key observations**：
- cat 和 duck（texture-poor objects）提升最大，rotation error 从 ~14° 降到 ~4°
- box（textured）提升较小，因为 ICP 在 textured object 上本身就 work 得好
- Maximum error 改善显著（cat: 30° → 8°），说明 **global consistency** 好

**Intuition**：MaskFusion 用 ICP 逐帧 tracking，error 会 accumulate。本文 joint optimize 所有 frames 的 pose，globally consistent，error 不会 accumulate。

### 7.2 3D geometry quality（Table II）

| Object | Method | C.D. ↓ | F-score ↑ | N.C. ↑ | Mask IoU ↑ |
|--------|--------|--------|-----------|--------|------------|
| cat | MF | 0.173 | 0.836 | 0.579 | 0.708 |
| cat | Ours | 0.051 | 0.926 | 0.818 | 0.839 |
| duck | MF | 0.177 | 0.812 | 0.587 | 0.674 |
| duck | Ours | 0.035 | 0.963 | 0.854 | 0.771 |

- **Chamfer Distance** 降 3-5 倍：geometry 更精确
- **Normal Consistency** 从 ~0.58 提升到 ~0.85：surface normal 更准确，这对 grasping 很关键
- **Mask IoU** 提升约 0.1：segmentation 更准

### 7.3 Ablation study（Table III, Fig. 7）

| Config | Rotation (deg) | Translation (cm) |
|--------|---------------|------------------|
| Full | 4.390 / 8.003 | 0.452 / 1.168 |
| w/o stage-wise | 18.421 / 44.382 | 3.820 / 9.000 |
| w/o foreground sampling | 9.417 / 30.385 | 1.056 / 4.160 |

**Key insight**：stage-wise training 最关键（去掉后 error 增加 4 倍），说明 **initialization 对 joint optimization 至关重要**。这与 NeRF 社区中 "BARF [35] 需要 coarse-to-fine pose regularization" 的发现一致。

Sparsity loss 的 visualization（Fig. 7b）显示，没有它 object 和 background 无法 decompose，验证了 motion ambiguity 的问题确实存在。

---

## 八、Grasping Application

公式 (12) 展示了 grasping pipeline：

$$T_{gb} = T_{go}T_{oc}T_{cb}$$

- $T_{go}$：gripper-to-object pose（由 Graspit! [16] 在 reconstructed model 上 planning）
- $T_{oc}$：object-to-camera pose（由 trained PVNet 估计）
- $T_{cb}$：camera-to-base pose（由 hand-eye calibration 获得）

**Intuition**：这是典型的 **pose-based grasping** pipeline。vs. end-to-end grasping（如 GraspNet [24]），pose-based 的优势是可以 leverage analytic grasp planning，对 object geometry 有 reasoning。

---

## 九、Limitations 和我的思考

作者提到的 limitations：
1. **Depth sensing 对 glossy/transparent surfaces 不好** → 影响 pose initialization 和 depth supervision
2. **Reconstruction 和 memorization 耗时长** → 可用 Instant-NGP [46]、Plenoxels [48]、OnePose [49] 加速

我自己的 additional thoughts：

### 9.1 Poking 的局限性
- **Fragile objects**：poking 可能损坏
- **Heavy objects**：poking 推不动
- **Tethered objects**：poking 不会产生 motion

### 9.2 Generalization 的边界
论文只在 3 个 objects 上测试，如何 scale 到几十个 objects？poking 时间会线性增长。

### 9.3 与最新方法的联想

这篇 paper 是 2022-2023 的工作，和很多最新方向相关：

1. **Gaussian Splatting [1]**：可以替代 NeRF 做重建，速度快 100x，质量相当
   - 参考：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

2. **OnePose [49] / OnePose++ [50]**：不需要 CAD model 的 one-shot pose estimation，可以替代 PVNet + render 训练的流程
   - 参考：https://zju3dv.github.io/onepose_plus_plus/

3. **Foundation models for pose estimation**：如 SAM + DINO 可以提供更好的 mask initialization

4. **Diffusion models for grasp generation**：如 Diffusion Policy [2]，可以替代 Graspit! 的 analytic planning
   - 参考：https://diffusion-policy.cs.columbia.edu/

5. **Interactive perception with large language models**：如 RT-2 [3]、VoxPoser [4]，可以用 LLM reasoning 来指导 poking strategy
   - RT-2: https://robotics-transformer2.github.io/
   - VoxPoser: https://voxposer.github.io/

6. **Category-level neural radiance fields**：如 NeuMesh、ObjectNeRF，可以结合 category prior 加速重建

---

## 十、与 Karpathy 你自己的工作的联想

Andrej，这篇 paper 和你的一些观点有有趣联系：

### 10.1 "Software 2.0" 视角
这篇 paper 的 memorization 阶段就是典型的 Software 2.0：用 neural network 学习从 image 到 pose 的 mapping，training data 由重建模型 render 生成。这和你的 "data as program" 思想一致。

### 10.2 "Micrograd" 和 implicit differentiation
论文中 joint optimize pose 和 NeRF parameters，涉及通过 volume rendering 的 backprop。如果你想从第一性原理理解，可以参考你的 micrograd 思路 —— 其实就是 chain rule through sampling and rendering。

### 10.3 "Recipe for training neural networks" 
论文的 stage-wise training、loss weighting、sampling strategy 都是你的 recipe 中提到的 "engineering matters" 的体现。特别是 sparsity loss 的设计，展示了 **inductive bias 的重要性** —— 纯粹的 end-to-end learning 不够，需要 domain knowledge 注入。

### 10.4 神经网络作为世界模型
这和你的 "world models" 讲座 [5] 相关 —— NeRF 本质上是一个 **scene-level world model**，可以 render、可以 query geometry。poking 则是 **active data collection for world model building**。
- 参考：https://worldmodels.github.io/

---

## 十一、技术细节的进一步展开

### 11.1 Scene flow for pose initialization

论文用 scene flow + Least-Squares 估计 initial pose。具体来说：

给定 object mask 内的 scene flow field $\{(\mathbf{p}_i, \mathbf{f}_i)\}_{i=1}^M$（3D points 和对应的 motion vectors），rigid transform $(R, \mathbf{t})$ 满足：

$$\mathbf{p}_i' = R\mathbf{p}_i + \mathbf{t} = \mathbf{p}_i + \mathbf{f}_i$$

Least-Squares 解：
$$\min_{R, \mathbf{t}} \sum_i \|R\mathbf{p}_i + \mathbf{t} - \mathbf{p}_i - \mathbf{f}_i\|^2$$

这是经典的 **point set registration** 问题，可以用 SVD closed-form 解（Arun et al. 1987）。

### 11.2 Marching cubes for mesh extraction

训练完 VolSDF 后，用 Marching Cubes [41] 从 SDF 提取 mesh。算法在 voxel grid 上 march，对每个 voxel 根据 8 个 corner 的 SDF 值判断 surface 是否经过该 voxel，并用 lookup table 确定 triangulation topology。

### 11.3 PnP for pose estimation

PVNet 用 PnP (Perspective-n-Point) 从 2D-3D correspondences 解 6DoF pose。给定 n 个 3D points $\{\mathbf{X}_i\}$ 和对应 2D projections $\{\mathbf{x}_i\}$，求 camera pose $(R, \mathbf{t})$ 使得：

$$\mathbf{x}_i = \pi(K(R\mathbf{X}_i + \mathbf{t}))$$

其中 $\pi$ 是 perspective projection，$K$ 是 intrinsic matrix。EPnP [42] 是 $O(n)$ 复杂度的经典解法。

---

## 十二、架构图解析（Fig. 1 和 Fig. 2）

### Fig. 1 - System overview

```
Scene with unseen objects
    ↓
Poking process (robot arm + RGB-D camera)
    ↓
Multi-view observations (RGB-D video)
    ↓
Reconstruction (decomposed NeRF + VolSDF)
    ↓
Reconstructed 3D object models
    ↓
Memorization (render training images → train PVNet)
    ↓
Neural network for recognition on new test images
```

### Fig. 2 - PVNet training pipeline

```
Reconstructed object model
    ↓
Render at various poses + ScanNet backgrounds
    ↓
Synthesized training images (10000 images)
    + Recorded video frames
    ↓
Train PVNet
    ↓
Trained pose estimator
```

---

## 十三、相关 paper 的 deeper dive

如果你想深入理解这篇 paper 的技术基础，推荐阅读：

1. **NeRF** [30]: https://arxiv.org/abs/2003.08934
   - 理解 volume rendering 的数学基础

2. **VolSDF** [31]: https://arxiv.org/abs/2106.12086
   - 理解 SDF 到 density 的转换

3. **NeuS** [32]: https://arxiv.org/abs/2106.10689
   - 另一种 SDF-based volume rendering，和 VolSDF 对比

4. **STaR** [37]: https://arxiv.org/abs/2101.08400
   - 本文 decomposed NeRF 的灵感来源

5. **Neural Scene Graphs** [34]: https://arxiv.org/abs/2103.01364
   - Dynamic scene decomposition

6. **DensePhysNet** [12]: https://arxiv.org/abs/1906.03853
   - Interactive perception 的前作

7. **OnePose** [49]: https://arxiv.org/abs/2111.03844
   - 无需 CAD model 的 pose estimation，可以替代本文的 memorization

8. **GraspNet-1Billion** [24]: https://arxiv.org/abs/2003.08589
   - End-to-end grasping 的 baseline

9. **Interactive Perception review** [8]: https://arxiv.org/abs/1708.00528
   - Bohg et al. 的综述，理解 interaction-based perception 的全貌

10. **BARF** [35]: https://arxiv.org/abs/2103.01332
    - Bundle-adjusting NeRF，理解 pose-NeRF joint optimization

---

## 十四、总结：build your intuition

这篇 paper 的核心 insight 可以浓缩为：

**"Action is perception's best friend when labels are scarce."**

具体的技术 intuition：
1. **Poking 是 information-dense action**：一次 poking 同时回答 "is this an object?" + "what are its multi-view observations?"
2. **Decomposed NeRF 让 segmentation 隐式可学**：通过 density ratio 分配 color，network 自己决定 boundary
3. **SDF for objects, density for background**：asymmetric representation，把 precision 留给需要的地方
4. **Sparsity loss 解决 contact ambiguity**：depth-aware prior 告诉 network "background 后面不要 stuff"
5. **Stage-wise training 避免 local optima**：先独立 estimate 再 joint refine

这篇 paper 在 2022 年的时间点是 **NeRF-based robotics** 的早期探索，后续 Instant-NGP、3D Gaussian Splatting、OnePose 等工作让这个方向越来越 practical。如果你想 reproduce 或 extend，建议从 Gaussian Splatting 替代 NeRF 开始，能获得 10-100x 加速。

希望这个 detailed walkthrough 帮你 build 起对这个方向的 intuition！如果想深入某个 specific 方面（比如 NeRF 的 math、ICP 的实现、或 Graspit! 的 planning），我可以继续展开。
