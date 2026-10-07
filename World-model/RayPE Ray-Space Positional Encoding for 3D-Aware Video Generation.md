---
source_pdf: RayPE Ray-Space Positional Encoding for 3D-Aware Video Generation.pdf
paper_sha256: 7f47f35a63cb8960a3e475d5990a3c1740030e3614bcc047ba671938bbbdd53a
processed_at: '2026-08-11T20:57:46-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RayPE 用人话说

## 1. 一句话概括

现有 video diffusion model 的 positional encoding 只告诉你"这个 token 在第几帧第几个像素"，但完全不知道"这个 token 在 3D 空间里对应的是哪条光线"。RayPE 把每条光线的几何信息直接塞进 attention 的 dot product 里，让模型天生就懂 3D 几何。

---

## 2. 问题出在哪？用生活类比

想象你看一段视频，两帧画面都拍到了同一把椅子。人一眼就知道"这是同一把椅子，只是视角不同"。但现有的 video diffusion model 不知道这件事。

它的 positional encoding 长这样：$(u, v, t)$，意思是"第 $t$ 帧的第 $v$ 行第 $u$ 列那个 patch"。

这就像你给快递柜编号——A1、A2、B1、B2，告诉模型"这个 token 在格子 A1"。但格子编号跟快递内容毫无关系。模型看到 A1 格子的椅子和 B3 格子的椅子，光从位置编号上完全猜不出"这俩是同一把椅子"。

模型只能靠 pixel content 硬猜：A1 那个 patch 长得像椅子腿，B3 那个也长得像椅子腿，颜色纹理相似，可能 attend 一下。但这是很笨的办法，因为：
- 视角变了，椅子腿看起来不一样
- 光照变了，颜色不一样
- 遮挡、模糊、运动模糊都会干扰

人脑不这么干。人脑天然知道"两帧画面里那条视线在 3D 空间里相交于同一点，所以它们看的是同一个东西"。这就是 epipolar geometry 的直觉。

RayPE 想干的事就是把这个 3D 几何直觉直接焊进 attention 机制里。

---

## 3. 核心 insight：一个美妙的代数巧合

### 3.1 先说说 attention 在干啥

Attention 的核心就是一个 dot product：

$$\langle q_i, k_j \rangle = q_i^\top k_j$$

这个 dot product 对 $q$ 线性，对 $k$ 也线性。数学上叫 **bilinear form**。

### 3.2 再说说 Plücker 坐标

3D 空间里一条光线（ray）可以用 6 个数描述：

$$\mathbf{r} = (\mathbf{d}, \mathbf{m})$$

- $\mathbf{d} \in \mathbb{R}^3$：光线的方向（单位向量）
- $\mathbf{m} = \mathbf{o} \times \mathbf{d} \in \mathbb{R}^3$：光线的"moment"，其中 $\mathbf{o}$ 是相机光心（光线上任一点都行）

直觉上 $\mathbf{m}$ 编码的是"光线离原点有多远、朝哪个方向偏"。两条平行的光线方向相同，但 $\mathbf{m}$ 不同，因为它们离原点距离不同。

### 3.3 Plücker reciprocal product：两条光线的关系

两条光线 $\mathbf{r}_i, \mathbf{r}_j$ 之间有个经典的几何量叫 reciprocal product：

$$\langle \mathbf{r}_i, \mathbf{r}_j \rangle_{\text{Pl}} = \mathbf{d}_i \cdot \mathbf{m}_j + \mathbf{d}_j \cdot \mathbf{m}_i$$

这个量有个超级好的性质：

- **双线性**：对 $\mathbf{r}_i$ 线性，对 $\mathbf{r}_j$ 也线性。**跟 attention 的 dot product 是同一种代数形状！**
- **共面判据**：$\langle \mathbf{r}_i, \mathbf{r}_j \rangle_{\text{Pl}} = 0$ 当且仅当两条光线共面（相交、平行或重合）。如果两条光线观测同一个 3D 点，它们必然相交，所以 reciprocal product = 0。
- **SE(3) 不变性**：随便怎么旋转平移整个场景，这个量不变。它只依赖两条光线的相对几何关系。

### 3.4 顿悟时刻

Paper 的核心 insight 就一句话：

> Attention score 是 bilinear form。Plücker reciprocal product 也是 bilinear form。它们形状一样。那能不能把 ray 几何塞进 attention score 里？

答案是可以。这就是整个 RayPE 的出发点。

---

## 4. 怎么塞？Q/K Flip 的妙处

### 4.1 朴素方案为什么不行

假设我想让 $\langle q_i, k_j \rangle$ 包含 reciprocal product $\mathbf{d}_i \cdot \mathbf{m}_j + \mathbf{m}_i \cdot \mathbf{d}_j$。

第一反应：$q_i$ 放 $(\mathbf{d}_i, \mathbf{m}_i)$，$k_j$ 放 $(\mathbf{d}_j, \mathbf{m}_j)$。

但 dot product 算出来是 $\mathbf{d}_i \cdot \mathbf{d}_j + \mathbf{m}_i \cdot \mathbf{m}_j$。这是普通 Euclidean 内积，不是 reciprocal product。完全不对。

### 4.2 Flip 方案

如果把 key 侧的 $\mathbf{d}$ 和 $\mathbf{m}$ 顺序交换一下：

- $q_i$ 放 $(\mathbf{d}_i, \mathbf{m}_i)$
- $k_j$ 放 $(\mathbf{m}_j, \mathbf{d}_j)$ ← 注意这里是 $\mathbf{m}$ 在前 $\mathbf{d}$ 在后

那 dot product 就是：

$$\mathbf{d}_i \cdot \mathbf{m}_j + \mathbf{m}_i \cdot \mathbf{d}_j = \langle \mathbf{r}_i, \mathbf{r}_j \rangle_{\text{Pl}}$$

正好是 reciprocal product！

这就是所谓的 **Q/K flip**。query 用正常顺序，key 把 (d, m) 两块交换。

### 4.3 这个 flip 是什么意思？

直觉上，reciprocal product 是个"交叉"内积（$\mathbf{d}$ 配 $\mathbf{m}$），不是"同类"内积（$\mathbf{d}$ 配 $\mathbf{d}$）。Flip 就是制造这种交叉配对。

更深一层：flip 选了一个 **canonical basis**，让对称配置 $E_q = E_k = I_6$ 恰好对应教科书上的 reciprocal product。这样初始化时模型天生就有正确的几何先验，学到的 $M = E_q^\top E_k$ 偏离 $I_6$ 的程度还能直接读出来，看模型"学偏了多少"。

注意 flip 本身不增加模型表达能力——任何 flip 后能表达的 bilinear form，no-flip 也能表达（只是参数换了一下）。Flip 的价值纯粹是 **interpretability 和 init prior**。

---

## 5. 完整公式（带变量解释）

每个 token $i$ 的 query 和 key 变成：

$$q_i = \underbrace{R(u_i, v_i, t_i) W_Q x_i}_{\text{原 pretrained 部分}} + \underbrace{E_q(\mathbf{d}_i, \mathbf{m}_i)}_{\text{geometry 注入}}$$

$$k_j = \underbrace{R(u_j, v_j, t_j) W_K x_j}_{\text{原 pretrained 部分}} + \underbrace{E_k(\mathbf{m}_j, \mathbf{d}_j)}_{\text{geometry 注入，flip！}}$$

变量含义：
- $x_i \in \mathbb{R}^d$：token $i$ 的输入特征
- $W_Q, W_K \in \mathbb{R}^{d \times d}$：pretrained 的 projection（不动它）
- $R(u, v, t)$：标准 3D RoPE 旋转（不动它）
- $E_q, E_k \in \mathbb{R}^{d \times 6}$：**新增的可学习参数**，把 6D Plücker 坐标投影到 $d$ 维
- $(\mathbf{d}_i, \mathbf{m}_i)$：token $i$ 对应光线的 Plücker 坐标，从 camera pose 算出来
- $(\mathbf{m}_j, \mathbf{d}_j)$：key 侧做了 flip

展开 attention score，会得到四项：

$$\langle q_i, k_j \rangle = \underbrace{\langle \tilde{W}_Q x_i, \tilde{W}_K x_j \rangle}_{(A) \text{ content}} + \underbrace{\langle \tilde{W}_Q x_i, E_k \tilde{\mathbf{r}}_j \rangle}_{(B) \text{ content}\to\text{geom}} + \underbrace{\langle E_q \mathbf{r}_i, \tilde{W}_K x_j \rangle}_{(C) \text{ geom}\to\text{content}} + \underbrace{\langle E_q \mathbf{r}_i, E_k \tilde{\mathbf{r}}_j \rangle}_{(D) \text{ geometry}}$$

人话解释这四项：
- **(A)**：原始 content attention，pretrained 模型本来就有的
- **(B) + (C)**：content 和 geometry 的耦合。比如"天空 content 偏好朝上的 ray"，"前景 content 偏好离得近的 ray"
- **(D)**：纯几何项。初始化时就是 Plücker reciprocal product，给模型"共面 ray 该 attend"的先验

---

## 6. 为什么不能直接用？NGI 来救场

### 6.1 问题一：Scale 乱七八糟

Plücker moment $\mathbf{m} = \mathbf{o} \times \mathbf{d}$ 随 camera 平移线性变化。相机移动 1 米 vs 移动 100 米，$\mathbf{m}$ 差 100 倍。

不同数据集的 scale convention 完全不同：
- **RealEstate10K**：COLMAP SfM 归一化，场景体积压到单位
- **DL3DV**：另一种 SfM 尺度
- **OmniWorld**：DROID-SLAM 内部归一化，跟 metric 没关系
- **PanShot**：合成数据，ground-truth by construction

直接用 raw $\mathbf{m}$ 的话，不同 batch 之间 geometry channel 的 magnitude 会剧烈跳变，优化根本稳不下来。

### 6.2 问题二：跟 content branch 不匹配

现代 video DiT 都用 QKNorm（RMSNorm）归一化 query/key，控制 content branch 的 magnitude。直接加一个没归一化的 geometry 项，两边尺度对不上，一个 scalar $\alpha$ 没法同时平衡 per-clip scale 和 geometry-content 的局部关系。

### 6.3 NGI 三步走

**Step 1：拆开 direction 和 magnitude**

$$\hat{\mathbf{m}} = \mathbf{m} / \max(\|\mathbf{m}\|, \epsilon), \quad s = \log \max(\|\mathbf{m}\|, \epsilon)$$

- $\hat{\mathbf{m}}$：归一化后的 moment，scale-invariant，全局 rescale 也不变
- $s$：log-magnitude，保留绝对距离信息
- $\epsilon = 10^{-6}$：防止近 static camera 除零

实际输入是 7D：$f^{(q)} = (\mathbf{d}, \hat{\mathbf{m}}, s)$，$f^{(k)} = (\hat{\mathbf{m}}, \mathbf{d}, s)$（注意 flip）

**Step 2：Scale-aware gate**

$$g_i = \sigma(G_s(s_i)) \in (0, 1)^d$$

- $G_s$：两层 MLP，输入 log-magnitude $s_i$
- 输出 $d$ 维 sigmoid gate
- 初始化让 $g \approx 0.5$ 当 $s = 0$
- 直觉：模型根据 pose scale 动态调 geometry branch 强度，而非一刀切

**Step 3：Normalize + Inject**

$$q_i = \text{QKNorm}(\tilde{W}_Q x_i) + \alpha \cdot g_i \odot N_q(E_q f_i^{(q)})$$

- $N_q, N_k$：新增的 learnable RMSNorm，跟 content branch 的 QKNorm 对称
- $\alpha$：scalar，初始化为 0，控制整体 geometry/content 平衡
- $\alpha = 0$ 时网络完全等于 pretrained DiT（zero init，安全启动）

### 6.4 Log-scale augmentation

训练时随机扰动 $s$：

$$\tilde{s}_i = s_i + \delta, \quad \delta \sim \text{Uniform}(-1.2, 1.6), \text{ prob. } 0.3$$

$\delta$ 在一个 clip 内共享。模拟全局 rescale 但不改监督信号。关键：只有 gate 用 $\tilde{s}$，projection 用真实 $s$，不污染几何信息。

---

## 7. 实验数据，用大白话解读

### 7.1 主对比（5B backbone，RealEstate10K 测试集）

| Method | CLIP↑ | RotErr↓ | TransErr↓ | FVD↓ |
|--------|-------|---------|-----------|------|
| CameraCtrl | 25.04 | 0.152 | 1.292 | 824.37 |
| ReCamMaster | 24.97 | 0.131 | 1.226 | 874.30 |
| ReRoPE | 25.20 | 0.137 | 1.109 | 684.57 |
| UCPE | 25.39 | 0.113 | 0.856 | 703.41 |
| **RayPE** | **26.05** | **0.085** | **0.751** | **543.17** |

人话：
- **CLIP**（帧间相似度）：RayPE 最高，说明生成质量好
- **RotErr / TransErr**（相机轨迹误差）：RayPE 最低，说明相机控制最准
- **FVD**（视频分布距离）：RayPE 最低（543 vs 其他 700+），说明整体视频质量大幅领先

14B backbone 上 RayPE 的 FVD 是 280，其他方法 500+，几乎减半。模型越大 RayPE 优势越明显。

### 7.2 Component ablation（最关键的一张表）

| Variant | FVD↓ | 解读 |
|---------|------|------|
| Full RayPE | 543 | 完整模型 |
| w/o Q/K flip | 580 | 小幅下降，证实 flip 只是 reparametrization |
| w/o NGI | 560 | 中等下降，NGI 整体有用 |
| w/o PE RMSNorm | 601 | 较大下降，归一化必须跟 content branch 对齐 |
| w/o zero init | 635 | 大幅下降，打破 pretrained equilibrium 代价高 |
| with V transform | 724 | 巨大下降，geometry 适合当 attention bias 不适合当 value |
| w/o (B)+(C) content↔geom | 695 | 大幅下降，cross-terms 是关键 |
| **w/o (D) geom↔geom** | **733** | **最大下降**，纯几何项是核心先验 |

人话解读：

1. **去掉 (D) 纯几何项**：FVD 从 543 飙到 733。这是最大的下降。说明 Plücker reciprocal product 这个几何先验是整个方法的灵魂，拿掉就废了。

2. **去掉 (B)+(C) content-geometry 耦合**：FVD 升到 695。这个耦合让模型学"某类 content 何时该依赖某类 geometry"——比如"天空 token 偏好朝上的 ray"。这是其他方法（V-side injection、cross-attention、AdaLN、rotation-based PE）**结构上无法表达**的，因为它们把 geometry 放在 attention 之外。

3. **把 geometry 也塞进 V**：FVD 飙到 724。说明 ray geometry 当 attention bias 最有效，当 value channel feature 反而有害。

4. **去掉 zero init**：FVD 升到 635。说明从 pretrained 模型平滑启动很重要，硬来打破已有 equilibrium 代价大。

5. **去掉 Q/K flip**：FVD 升到 580，下降最小。印证理论：flip 只是 reparametrization，不改变函数空间。它的价值是 init prior 和 interpretability。

### 7.3 数据混合效应

| Training mixture | FVD↓ |
|------------------|------|
| RE10K only | 586 |
| + DL3DV | 602（暂时略升） |
| + PanShot | 550 |
| + OmniWorld (full) | 543 |

随着更 scale-heterogeneous 的数据加入（尤其 OmniWorld 的 DROID-SLAM poses），RayPE 持续受益。这正是 NGI 的设计目标——让方法能吃下混合 scale 的数据。

---

## 8. 跟其他方法比，到底好在哪？

### 8.1 Adapter-based 方法（CameraCtrl、MotionCtrl、CamCo）

这些方法把 camera 信息塞进 attention **外部**的模块里，比如 ControlNet-style encoder、per-frame token、cross-attention stream。attention dot product 本身还是纯 content-based。

人话：它们让模型"看一眼 camera 信息然后自己琢磨"，attention 内部没有几何先验。

RayPE 直接把几何塞进 dot product，模型**算 attention 的时候就在用几何**。

### 8.2 Rendering-conditioned 方法（ViewCrafter、TrajectoryCrafter、GS-DiT）

这些方法先建 3D proxy（point cloud、Gaussian），然后渲染出来当 pixel-aligned condition。

人话：它们用渲染结果当"答案参考图"，绕了一圈。

RayPE 不需要渲染，直接用 camera pose 算 ray 就行，轻量得多。

### 8.3 其他 camera-aware PE（PRoPE、UCPE、ReRoPE）

这些方法 **multiplicative** 地修改 RoPE——替换或分裂 RoPE 的某些频率 band。三个问题：

1. **破坏 pretrained 位置结构**：RoPE 是 pretrained 模型已经学好的，你改它等于推翻重来
2. **用 reduced parameterization**：rotation angle、projective matrix 这些，没利用 dot product 的 bilinear 结构
3. **不处理 scale heterogeneity**：有的丢掉绝对 translation，有的固定归一化

RayPE 三个都解决：additive 保留 RoPE、bilinear 结构用满、NGI 处理 scale。

---

## 9. 为什么这个方法 work？三个直觉

### 9.1 代数同构不是巧合

Attention score 是 bilinear form。Plücker reciprocal product 也是 bilinear form。它们形状一样，所以 ray 几何可以"嵌入"attention 而不需要额外计算通道。这个 embedding 是 **canonical 的**——对称配置下直接对应 Klein form。

### 9.2 几何先验当 inductive bias

初始化时 (D) 项就是 reciprocal product，模型 step 0 就有"共面 ray 该 attend"的先验。这比让模型从 pixel content 隐式学 epipolar geometry 高效太多。

类比：你教小孩认椅子，告诉他"椅子有四条腿、一个座、一个靠背"（先验），比让他看几万张椅子图片自己悟出来快得多。

### 9.3 Content-Geometry 耦合是独有结构

(B)+(C) cross-terms 让 content 和 geometry 互相调制。这是 V-side injection、cross-attention、AdaLN、rotation-based PE 结构上无法表达的，因为它们把 geometry 放在 attention 之外。

Ablation 显示这个耦合贡献巨大（去掉 FVD 升 150+）。这是 RayPE 的"独特武器"。

---

## 10. 可能的联想与延伸

### 10.1 跟 Epipolar Attention 的关系

Epipolar Transformer (https://arxiv.org/abs/2107.10133) 显式 mask attention 沿 epipolar line。RayPE 隐式做类似事：共面 ray 有高 attention score（reciprocal product 接近 0）。但 RayPE 是 soft、可学习的，不是 hard mask。

### 10.2 跟 3D Gaussian Splatting 的联系

Plücker 坐标在 differentiable rendering (https://arxiv.org/abs/2308.14737) 里也用于 ray-line 距离计算。RayPE 把同样表示引入 generative transformer，是 3D 表示和生成模型之间的桥梁。

### 10.3 Multi-View Diffusion 的自然扩展

MVDream (https://arxiv.org/abs/2308.16512) 这类 multi-view diffusion 天然适合 RayPE：每个 view 的 token 都有自己的 Plücker ray，cross-view attention 自动获得几何先验。

### 10.4 World Model 视角

如果 video diffusion 是 world model 的一种形式 (https://worldmodels.github.io)，RayPE 注入的 ray geometry 可能是从 2D content reasoning 走向 3D scene reasoning 的关键一步。

### 10.5 Geometric Algebra 的深层联系

Plücker 坐标和 Klein form 是 Grassmann-Cayley algebra 的特例。Attention 的 bilinear form 跟 geometric algebra 的 inner product 结构相似。更深联系可能通过 Clifford algebra attention (https://arxiv.org/abs/2402.15410) 探索。

### 10.6 动态场景的挑战

Plücker ray 假设静态场景。动态物体（人、车）的 ray-geometry 关系更复杂——同一 3D 点在不同帧可能移动了。未来可能需要 4D Plücker（加时间维度）或 scene flow aware 的扩展。

### 10.7 跟 NeRF 的思想连续性

NeRF (https://arxiv.org/abs/2003.08934) 用 ray 作为 query 的基本单元。RayPE 把 ray 作为 attention 的基本单元。都是从 2D pixel 思维走向 3D ray 思维。PlückerGAN (https://arxiv.org/abs/2211.16487)、Ray Conditioning (https://arxiv.org/abs/2311.17084) 也有类似思想。

### 10.8 Orbit / Dolly / Pan 的自动编码

Camera trajectory 有结构化模式：orbit、dolly、pan 等。RayPE 的 Plücker 表示自动编码这些——orbit 对应 $\mathbf{m}$ 方向固定、$\mathbf{d}$ 旋转；dolly 对应 $\mathbf{o}$ 沿 $\mathbf{d}$ 平移。这可能解释为什么 RayPE 在 OOD（艺术画、电影 still）上泛化好——几何先验不依赖 content domain。

---

## 11. 一句话总结

RayPE 发现 attention score 和 Plücker reciprocal product 都是 bilinear form，于是把 6D ray 坐标 additively 注入 query/key（配一个 flip 让对称 init 对应 Klein form），用 NGI 处理 scale heterogeneity，让 video diffusion model 天生懂 3D 几何，几乎零成本（<0.1% 参数）就能大幅提升 camera controllability 和 video quality。

核心 takeaway：**当两个看似不同的结构共享同一代数形状时，往往存在 canonical 的嵌入方式让一个结构"活"在另一个内部**。这是 math-driven architecture design 的典范。

---

## 参考链接

**论文与项目**：
- RayPE Project: https://raype-project.github.io

**Baselines**：
- CameraCtrl: https://openreview.net/forum?id=Nw8r7Hc6t5
- ReCamMaster: https://arxiv.org/abs/2503.11647
- UCPE: https://arxiv.org/abs/2512.07237
- ReRoPE: https://arxiv.org/abs/2602.08068
- PRoPE: https://arxiv.org/abs/2507.10496
- ViewCrafter: https://arxiv.org/abs/2409.02048
- MotionCtrl: https://dl.acm.org/doi/10.1145/3658167.3658181
- CamCo: https://arxiv.org/abs/2406.02509

**背景**：
- Wan2.2: https://arxiv.org/abs/2503.20314
- RoPE: https://arxiv.org/abs/2104.09864
- Plücker coordinates: https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates
- QKNorm: https://arxiv.org/abs/2010.04245
- RMSNorm: https://arxiv.org/abs/1910.07467
- DROID-SLAM: https://arxiv.org/abs/2108.10869
- RealEstate10K: https://arxiv.org/abs/1805.06191

**延伸**：
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.14737
- MVDream: https://arxiv.org/abs/2308.16512
- Epipolar Transformer: https://arxiv.org/abs/2107.10133
- Clifford algebra attention: https://arxiv.org/abs/2402.15410
- NeRF: https://arxiv.org/abs/2003.08934
- PlückerGAN: https://arxiv.org/abs/2211.16487
- World Models: https://worldmodels.github.io

---

# RayPE: Ray-Space Positional Encoding for 3D-Aware Video Generation 深度讲解

## 1. 核心问题与 Motivation

现代 video diffusion transformer（如 Wan、CogVideoX、HunyuanVideo、MovieGen）都采用类似的架构范式：将 latent video frame patchify 成一个 $(u, v, t)$ 的 token grid，然后通过 factorized 3D RoPE 注入位置信息。

**关键观察**：$(u, v, t)$ 只是 camera sampling grid 上的整数索引，它告诉你 token 在 2D 像素格和时间轴上的位置，但完全没有告诉你这个 token 对应的 3D ray 指向哪里。两个 token 可能从不同视角观测同一物理表面，但它们的位置编码没有任何几何关系，模型必须完全从 pixel content 恢复 3D 对应。

这就是 RayPE 要解决的核心问题：**能否让 camera ray 的几何信息直接活在 attention dot product 内部，让 3D 信息通过承载位置和 content 的同一通道进入模型？**

参考：Wan2.2 (https://arxiv.org/abs/2503.20314)、RoPE (https://arxiv.org/abs/2104.09864)、3D RoPE for ViT (https://arxiv.org/abs/2403.13276)

---

## 2. 核心洞察：Plücker Reciprocal Product 与 Attention Score 的代数同构

这是整篇论文的灵魂。让我详细推导。

### 2.1 Plücker 坐标

3D 中的一条 ray（有向直线）用 6D Plücker 坐标 $(\mathbf{d}, \mathbf{m}) \in \mathbb{R}^6$ 表示：

- $\mathbf{d} \in \mathbb{R}^3$：ray 的单位方向向量，$\|\mathbf{d}\| = 1$
- $\mathbf{m} = \mathbf{o} \times \mathbf{d} \in \mathbb{R}^3$：Plücker moment，其中 $\mathbf{o}$ 是 ray 上任一点（通常取 camera origin）
- 约束：$\mathbf{d} \cdot \mathbf{m} = 0$（因为 $\mathbf{d} \cdot (\mathbf{o} \times \mathbf{d}) = 0$，标量三重积中两个向量相同）

参考：Plücker coordinates (https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates)、Hartley & Zisserman *Multiple View Geometry*

### 2.2 Plücker Reciprocal Product（Klein Form）

定义两个 ray $\mathbf{r}_i = (\mathbf{d}_i, \mathbf{m}_i)$ 和 $\mathbf{r}_j = (\mathbf{d}_j, \mathbf{m}_j)$ 的 reciprocal product：

$$\langle \mathbf{r}_i, \mathbf{r}_j \rangle_{\text{Pl}} \equiv \mathbf{d}_i \cdot \mathbf{m}_j + \mathbf{d}_j \cdot \mathbf{m}_i = \mathbf{r}_i^\top J \mathbf{r}_j$$

其中 $J = \begin{pmatrix} \mathbf{0}_3 & I_3 \\ I_3 & \mathbf{0}_3 \end{pmatrix}$ 是辛形式矩阵。

**三个关键性质**：

**(1) 双线性**：对每个 ray 分别线性。展开验证：
$$\langle \alpha \mathbf{r}_i + \beta \mathbf{r}_i', \mathbf{r}_j \rangle_{\text{Pl}} = \alpha \langle \mathbf{r}_i, \mathbf{r}_j \rangle_{\text{Pl}} + \beta \langle \mathbf{r}_i', \mathbf{r}_j \rangle_{\text{Pl}}$$

**(2) 共面判据**：$\langle \mathbf{r}_i, \mathbf{r}_j \rangle_{\text{Pl}} = 0$ 当且仅当两 ray 共面（相交、平行或重合）。直觉：两个 ray 如果观测同一 3D 点，它们必相交于该点，因此 reciprocal product 为 0。

**(3) SE(3) 不变性**：对刚体变换 $g = (R, \mathbf{t}) \in \text{SE}(3)$，作用在 ray 上为 $g \cdot (\mathbf{d}, \mathbf{m}) = (R\mathbf{d}, R\mathbf{m} + \mathbf{t} \times R\mathbf{d})$，则 $\langle g \cdot \mathbf{r}_i, g \cdot \mathbf{r}_j \rangle_{\text{Pl}} = \langle \mathbf{r}_i, \mathbf{r}_j \rangle_{\text{Pl}}$。证明依赖标量三重积的反对称性：$R\mathbf{d}_i \cdot (\mathbf{t} \times R\mathbf{d}_j) = -R\mathbf{d}_j \cdot (\mathbf{t} \times R\mathbf{d}_i)$ 两项抵消。

### 2.3 代数同构

Attention score $\langle q_i, k_j \rangle$ 是 bilinear 形式（对 $q$ 和 $k$ 分别线性）。

Plücker reciprocal product $\langle \mathbf{r}_i, \mathbf{r}_j \rangle_{\text{Pl}}$ 也是 bilinear 形式（对 $\mathbf{r}_i$ 和 $\mathbf{r}_j$ 分别线性）。

**这两者具有完全相同的代数 shape**。这就是 RayPE 的核心：能否把 ray 几何关系编码进 attention score，让 3D 信息通过原本承载位置和 content 的同一通道进入？

---

## 3. Plücker Flip Positional Encoding：核心设计

### 3.1 为什么需要 Q/K Flip？

设想我们想让 $\langle q_i, k_j \rangle$ 包含 $\mathbf{d}_i \cdot \mathbf{m}_j + \mathbf{m}_i \cdot \mathbf{d}_j$（reciprocal product）。

**朴素方案**：$q_i$ 含 $(\mathbf{d}_i, \mathbf{m}_i)$，$k_j$ 含 $(\mathbf{d}_j, \mathbf{m}_j)$
$$q_i \cdot k_j = \mathbf{d}_i \cdot \mathbf{d}_j + \mathbf{m}_i \cdot \mathbf{m}_j$$
这是 Euclidean 内积，**不是** reciprocal product！

**Flip 方案**：$q_i$ 含 $(\mathbf{d}_i, \mathbf{m}_i)$，$k_j$ 含 $(\mathbf{m}_j, \mathbf{d}_j)$（交换 $\mathbf{d}$ 和 $\mathbf{m}$）
$$q_i \cdot k_j = \mathbf{d}_i \cdot \mathbf{m}_j + \mathbf{m}_i \cdot \mathbf{d}_j = \langle \mathbf{r}_i, \mathbf{r}_j \rangle_{\text{Pl}} \checkmark$$

这就是 flip 的妙处：通过交换 key 侧的 (d, m) 顺序，使对称配置 $E_q = E_k = I_6$ 恰好对应教科书上的 Plücker reciprocal product。

### 3.2 形式化定义

公式 (7)-(8)：
$$q_i = R(u_i, v_i, t_i) W_Q x_i + E_q(\mathbf{d}_i, \mathbf{m}_i)$$
$$k_j = R(u_j, v_j, t_j) W_K x_j + E_k(\mathbf{m}_j, \mathbf{d}_j)$$

变量解释：
- $x_i \in \mathbb{R}^d$：token $i$ 的输入特征
- $W_Q, W_K \in \mathbb{R}^{d \times d}$：pretrained query/key projection
- $R(u, v, t)$：标准 3D RoPE 旋转，block-diagonal 结构，对每对 (q, k) 旋转使其 dot product 依赖 $(u_i - u_j, v_i - v_j, t_i - t_j)$
- $E_q, E_k \in \mathbb{R}^{d \times 6}$：可学习的 ray projection（新增参数）
- $\mathbf{d}_i, \mathbf{m}_i$：token $i$ 对应 camera ray 的 Plücker 坐标

### 3.3 Attention Score 的四项分解

展开 $\langle q_i, k_j \rangle$（公式 9）：

$$\langle q_i, k_j \rangle = \underbrace{\langle \tilde{W}_Q x_i, \tilde{W}_K x_j \rangle}_{(A) \text{ content}} + \underbrace{\langle \tilde{W}_Q x_i, E_k \tilde{\mathbf{r}}_j \rangle}_{(B) \text{ content}\to\text{geom}} + \underbrace{\langle E_q \mathbf{r}_i, \tilde{W}_K x_j \rangle}_{(C) \text{ geom}\to\text{content}} + \underbrace{\langle E_q \mathbf{r}_i, E_k \tilde{\mathbf{r}}_j \rangle}_{(D) \text{ geometry}}$$

其中 $\tilde{\mathbf{r}}_j = (\mathbf{m}_j, \mathbf{d}_j)$ 是翻转后的坐标，$\tilde{W}_Q, \tilde{W}_K$ 表示经过 RoPE 旋转后的 projection。

**四项的物理意义**：
- **(A)**：原始 content attention，pretrained 模型的主力
- **(B)+(C)**：content 与 geometry 的耦合，让模型学习"某类 content 何时应该依赖某类 geometry"
- **(D)**：纯几何项，$\mathbf{r}_i^\top M \tilde{\mathbf{r}}_j$，其中 $M = E_q^\top E_k \in \mathbb{R}^{6 \times 6}$

### 3.4 Proposition 1：Flip 作为 Canonical Basis

**命题**：当 $E_q = E_k = I_6$（即 $M = I_6$）时，(D) 项恰好等于 Plücker reciprocal product：
$$(D)|_{M=I_6} = \mathbf{d}_i \cdot \mathbf{m}_j + \mathbf{m}_i \cdot \mathbf{d}_j = \langle \mathbf{r}_i, \mathbf{r}_j \rangle_{\text{Pl}}$$

**附录 A.4 的更细分解**：将 $M$ 写成 $2 \times 2$ block：$M = \begin{pmatrix} A & B \\ C & D \end{pmatrix}$，每个 block 是 $3 \times 3$。

$$(D) = \mathbf{d}_i^\top A \mathbf{m}_j + \mathbf{d}_i^\top B \mathbf{d}_j + \mathbf{m}_i^\top C \mathbf{m}_j + \mathbf{m}_i^\top D \mathbf{d}_j$$

- $A = D = I_3, B = C = \mathbf{0}_3$ 时塌缩为 reciprocal product
- **off-diagonal blocks** $A, D$ 控制 reciprocal-product-like 部分
- **diagonal blocks** $B, C$ 控制 direction-direction 和 moment-moment 耦合（经典 reciprocal product 中不存在）

**重要澄清（附录 A.5）**：flip 并**不扩大**模型能表达的函数空间。对任意 learnable $E_k$，flip 形式与 no-flip 形式只差一个置换矩阵 $P = \begin{pmatrix} 0 & I_3 \\ I_3 & 0 \end{pmatrix}$：$E_k^{\text{flip}} = E_k^{\text{no-flip}} P$。所以 $\|M - I_6\|$ 可以直接读取学习到的 bilinear bias 偏离 ray-incidence 形式的程度——这是一个 **post-hoc interpretability** 的便利。

---

## 4. Normalize-Gate-Inject (NGI)：工程化的关键

直接用 raw Plücker 坐标会遇到两个严重问题。

### 4.1 问题一：Scale Heterogeneity

Plücker moment $\mathbf{m} = \mathbf{o} \times \mathbf{d}$ 随 camera translation 线性缩放。如果所有 camera 位置 rescale $s$ 倍：$\mathbf{o} \to s\mathbf{o}$，则 $\mathbf{m} \to s\mathbf{m}$，reciprocal product 也 $\to s \cdot \langle \cdot, \cdot \rangle_{\text{Pl}}$（公式 6）。

不同数据集的 scale convention 差异巨大：
- **RE10K**：COLMAP SfM 归一化（场景体积归一）
- **DL3DV**：场景级 SfM 尺度
- **DROID-SLAM**（OmniWorld 用）：内部归一化尺度，与 metric 或 SfM 无关
- **PanShot**：合成数据，ground-truth by construction

直接用 raw $\mathbf{m}$ 会让 geometry channel 在不同 batch 间 magnitude 剧烈波动，优化不稳定。

### 4.2 问题二：Magnitude Imbalance

现代 video DiT 都用 QKNorm（RMSNorm）归一化 $q, k$，控制 content branch 的 magnitude。但直接加未归一化的 $E\mathbf{r}$ 不尊重这个归一化。单靠一个 global scalar $\alpha$ 也太粗糙，无法同时处理 per-clip pose scale 和 geometry-content 的局部平衡。

### 4.3 NGI 三步走

**Step 1: Direction/Magnitude Decomposition**（公式 11）

$$\hat{\mathbf{m}} = \mathbf{m} / \max(\|\mathbf{m}\|, \epsilon), \quad s = \log \max(\|\mathbf{m}\|, \epsilon)$$

- $\hat{\mathbf{m}}$：归一化 moment，scale-invariant（全局 rescale $\mathbf{o} \to s\mathbf{o}$ 时 $\hat{\mathbf{m}}$ 不变）
- $s$：log-magnitude，保留绝对距离信息作为单独 scalar
- $\epsilon = 10^{-6}$：数值安全，避免近 static camera 时除零
- 7D 几何输入：
  - $f^{(q)} = (\mathbf{d}, \hat{\mathbf{m}}, s) \in \mathbb{R}^7$
  - $f^{(k)} = (\hat{\mathbf{m}}, \mathbf{d}, s) \in \mathbb{R}^7$（注意 flip：d 和 $\hat{\mathbf{m}}$ 交换，但 $s$ 保持在末尾）

**Step 2: Scale-Aware Gating**（公式 12）

$$g_i = \sigma(G_s(s_i)) \in (0, 1)^d$$

- $G_s$：两层 MLP，输入 log-magnitude $s_i$，输出 $d$ 维
- $\sigma$：sigmoid
- bias 初始化使 $g \approx 0.5$ 当 $s = 0$
- 直觉：模型根据 pose scale 动态调整 geometry branch 强度，而非一刀切

**Step 3: Normalize and Inject**（公式 13-15）

$$\text{pe}_i^q = g_i \odot N_q(E_q f_i^{(q)})$$
$$\text{pe}_j^k = g_j \odot N_k(E_k f_j^{(k)})$$

$$q_i = \text{QKNorm}(\tilde{W}_Q x_i) + \alpha \cdot \text{pe}_i^q$$
$$k_j = \text{QKNorm}(\tilde{W}_K x_j) + \alpha \cdot \text{pe}_j^k$$

- $N_q, N_k$：per-layer learnable RMSNorm，与 content branch 的 QKNorm 对称
- $\odot$：element-wise 乘
- $\alpha$：scalar，初始化为 0（zero init，确保 step 0 时网络完全等于 pretrained DiT）
- 由于两侧都归一化，$\alpha$ 在不同 layer 和不同训练样本间有 consistent 含义

### 4.4 Log-Scale Augmentation（公式 16）

$$\tilde{s}_i = s_i + \delta, \quad \delta \sim \text{Uniform}(-1.2, 1.6), \text{ with prob. } 0.3$$

- $\delta$ 在一个 clip 内所有 token 共享
- 模拟全局 rescaling $\mathbf{o} \to e^\delta \mathbf{o}$ 而不改变监督信号
- **关键**：只有 gate 接收 $\tilde{s}_i$，projection $E_q f^{(q)}$ 仍用真实 $s_i$
- 直觉：训练模型对 scale shift 鲁棒，但不污染真实的几何信息

---

## 5. 架构总览（Algorithm 1 解读）

每个 attention layer 的完整流程：

```
1. Trajectory rescaling: o_t ← (η_ds / max(z_n, ε)) · o_t
   （per-dataset scale factor η，per-clip near-depth z_n）
2. Compute per-token rays {(d_i, m_i)} from {T_t, K_t} 和 latent patch grid
3. Sample noise ε ~ N(0, I), timestep τ, build noisy latent z_τ
4. For each DiT block ℓ:
   a. q, k, v ← W_Q h, W_K h, W_V h  （pretrained projections）
   b. q, k ← QKNorm(q), QKNorm(k)  （标准 QKNorm）
   c. q, k ← R(u,v,t)·q, R(u,v,t)·k  （标准 3D RoPE，未修改！）
   d. f^(q) ← (d, m̂, s), f^(k) ← (m̂, d, s)  （Q/K flip）
   e. g ← σ(G_s(ŝ))  （scale gate）
   f. q ← q + α · g ⊙ N_q(E_q f^(q))  （geometry injection）
   g. k ← k + α · g ⊙ N_k(E_k f^(k))
   h. h ← h + Attn(q, k, v) W_O
   i. h ← h + CrossAttn(h, c) + FFN(h)
```

**关键设计要点**：
- **Additive**：原 RoPE 完全保留，只是 Q/K 上加一个 bias 项
- **Zero-init**：α = 0, PE RMSNorm 权重 = 1, gate bias = 0, E_q/E_k output layer = 0
- **Per-layer 独立**：每个 attention layer 有自己的 $E_q, E_k, N_q, N_k, G_s$
- **参数量**：< 0.1% of 5B backbone

---

## 6. 实验数据深度解读

### 6.1 主对比（Table 1）

**Wan-2.2 5B Scale**：

| Method | CLIP↑ | RotErr↓ | TransErr↓ | CamMC↓ | ATE↓ | FVD↓ | FVDc↓ | FID↓ |
|--------|-------|---------|-----------|--------|------|------|-------|------|
| CameraCtrl | 25.04 | 0.152 | 1.292 | 1.355 | 1.501 | 824.37 | 805.62 | 64.08 |
| ReCamMaster | 24.97 | 0.131 | 1.226 | 1.279 | 1.460 | 874.30 | 890.52 | 62.53 |
| ReRoPE | 25.20 | 0.137 | 1.109 | 1.215 | 1.350 | 684.57 | 650.31 | 60.77 |
| UCPE | 25.39 | 0.113 | 0.856 | 0.909 | 0.990 | 703.41 | 755.83 | 61.50 |
| **RayPE** | **26.05** | **0.085** | **0.751** | **0.802** | **0.884** | **543.17** | **588.62** | **57.83** |

**Wan-2.2 14B Scale**：

| Method | CLIP↑ | RotErr↓ | TransErr↓ | CamMC↓ | ATE↓ | FVD↓ | FVDc↓ | FID↓ |
|--------|-------|---------|-----------|--------|------|------|-------|------|
| UCPE | 25.72 | 0.082 | 0.693 | 0.760 | 0.788 | 529.42 | 558.70 | 54.75 |
| **RayPE** | **26.30** | **0.058** | **0.517** | **0.530** | **0.605** | **280.17** | **354.52** | **41.01** |

**观察**：
1. RayPE 在所有 8 个指标上都最好
2. 14B scale 上 FVD 从 ~530 降到 280，几乎减半——scale 上去后收益更大
3. Pose error（RotErr, TransErr, CamMC, ATE）大幅改善，说明 camera controllability 显著提升
4. Quality（CLIP）和 distribution（FVD, FID）也改善——几何信息帮助了整体生成质量

### 6.2 评估协议的设计选择

作者用 **raw unscaled ViPE trajectories** 而非传统的 rescaled protocol。

**为什么？**
- Rescaled 会掩盖 degenerate solution：只产生微小全局运动的模型，rescale 后看起来 shape error 很小
- Raw 保持绝对 scale fidelity，真实反映模型是否尊重用户指定的 scale
- Rescaled 单位是 per-clip 的 GT norm，难以跨数据集比较

附录 Table 5 也提供了 rescaled 版本，RayPE 仍然全面领先。

### 6.3 Camera-Aware PE Design Space（Table 2）

对比四种 FreqSplit-RoPE 变体（multiplicative 替换 RoPE 低频部分）vs RayPE：

| PE design | CLIP↑ | RotErr↓ | TransErr↓ | FVD↓ |
|-----------|-------|---------|-----------|------|
| FreqSplit + 4×4 Proj. | 25.09 | 0.175 | 1.227 | 675.03 |
| FreqSplit + Camera-UV | 25.39 | 0.141 | 1.205 | 685.20 |
| FreqSplit + Plücker-RoPE | 25.61 | 0.159 | 1.018 | 671.45 |
| FreqSplit + 4DOF Plücker | 25.52 | 0.137 | 0.976 | 638.31 |
| **RayPE** | **26.05** | **0.085** | **0.751** | **543.17** |

**结论**：additive Plücker formulation 显著优于 multiplicative-RoPE 家族，RotErr 几乎减半，FVD 降 ~100。

### 6.4 Component Ablation（Table 3）——最重要的实验

| Variant | CLIP↑ | RotErr↓ | CamMC↓ | ATE↓ | FVD↓ |
|---------|-------|---------|--------|------|------|
| Full RayPE | 26.05 | 0.085 | 0.802 | 0.884 | 543.17 |
| w/o Q/K flip | 25.93 | 0.091 | 0.817 | 0.929 | 579.83 |
| w/o NGI | 25.86 | 0.086 | 0.865 | 0.933 | 560.37 |
| w/o PE RMSNorm | 25.62 | 0.090 | 0.874 | 0.913 | 601.39 |
| w/o log-scale aug. | 25.94 | 0.083 | 0.823 | 0.890 | 557.92 |
| w/o zero init | 25.68 | 0.86 | 0.837 | 0.899 | 635.42 |
| with V transform | 25.57 | 0.090 | 0.855 | 0.949 | 724.10 |
| w/o (B)+(C) content↔geom | 25.21 | 0.135 | 1.174 | 1.358 | 695.41 |
| **w/o (D) geom↔geom** | **24.08** | **0.159** | **1.382** | **1.440** | **733.08** |

**关键发现**：

1. **w/o (D) geometry↔geometry**：FVD 从 543 升到 733，**最大下降**。纯几何项（Klein form matching）是核心 geometric prior，不可替代。

2. **w/o (B)+(C) content↔geometry**：FVD 升到 695。Cross-terms 让 content 与 geometry 耦合（"sky token 偏好向上 ray"、"foreground token 偏好近 ray"），这种耦合是 V-side、cross-attention、AdaLN、rotation-based ray PE **结构上无法表达**的。

3. **with V transform**：FVD 升到 724。说明 ray geometry 更适合作为 **attention bias** 而非 value channel feature。

4. **w/o zero init**：FVD 升到 635。打破 pretrained equilibrium 代价很高，即使最终能收敛。

5. **w/o PE RMSNorm**：FVD 升到 601。几何 magnitude 必须与 content branch 匹配。

6. **w/o Q/K flip**：FVD 升到 580，变化最小——证实了 Proposition 1 的理论预测（flip 只是 reparametrization）。

### 6.5 数据混合效应（Table 4）

| Training mixture | CLIP↑ | RotErr↓ | CamMC↓ | ATE↓ | FVD↓ |
|------------------|-------|---------|--------|------|------|
| RE10K only | 25.58 | 0.091 | 0.897 | 0.983 | 586.45 |
| + DL3DV | 25.65 | 0.089 | 0.872 | 0.960 | 602.13 |
| + PanShot | 25.93 | 0.083 | 0.836 | 0.905 | 550.19 |
| + OmniWorld (full) | 26.05 | 0.085 | 0.802 | 0.884 | 543.17 |

**直觉**：随着更 scale-heterogeneous 的数据加入（特别是 OmniWorld 的 DROID-SLAM poses），RayPE 持续受益——这正是 NGI 的设计目标。

---

## 7. 与相关工作对比

### 7.1 Camera-Conditioned Video Generation 三大流派

**Adapter-based**：
- MotionCtrl (https://dl.acm.org/doi/10.1145/3658167.3658181)：extrinsics 编码成 per-frame token 加到 temporal attention
- CameraCtrl (https://openreview.net/forum?id=Nw8r7Hc6t5)：rasterize per-pixel Plücker maps，ControlNet-style encoder
- CamCo (https://arxiv.org/abs/2406.02509)：concatenate Plücker embeddings at input channel + epipolar-constrained cross-view attention
- AC3D, VD3D：cross-attention adapters / LoRA

**Rendering-conditioned**：
- ViewCrafter (https://arxiv.org/abs/2409.02048)：lift input 到 point cloud，rendered views 作为 pixel-aligned condition
- TrajectoryCrafter (https://arxiv.org/abs/2503.13136)：rendered point-cloud video + source video 双流
- GS-DiT (https://arxiv.org/abs/2501.02690)：tracked 3D Gaussians 替代轨迹

**Implicit cross-view**：
- ReCamMaster (https://arxiv.org/abs/2503.11647)：concatenate source/target tokens along frame dim，lightweight camera encoder

**RayPE 的不同**：所有上述方法都把 camera geometry 当作 attention **外部**处理的 content，attention dot product 仍然是纯 content-based。RayPE 把 ray geometry 直接放进 Q/K 作为 positional encoding，让 attention score 本身携带几何信息，**无需任何 adapter 或 rendering proxy**。

### 7.2 Camera-Aware Positional Encoding

- **PRoPE** (https://arxiv.org/abs/2507.10496)：multiplicative relative-projective rotation 替换 temporal RoPE
- **UCPE** (https://arxiv.org/abs/2512.07237)：spatial adapter 调制 RoPE frequencies
- **ReRoPE** (https://arxiv.org/abs/2602.08068)：repurpose 低频 temporal RoPE bands 为 relative camera pose

**三个共同局限**：
1. **Multiplicative**：替换或分裂原 RoPE，破坏 pretrained 位置结构
2. **Reduced parameterization**：rotation angles, projective matrices，未利用 dot product 的 bilinear 结构
3. **不处理 scale heterogeneity**：PRoPE/ReRoPE 丢弃绝对 translation，UCPE 用固定归一化

RayPE 三个都解决：additive injection 保留 RoPE，canonical Q/K flip basis 实现 Klein form，direction/magnitude decoupling + learned scale-aware gate。

---

## 8. Intuition 总结

### 8.1 为什么这个方法有效？

**核心 algebraic insight**：Plücker reciprocal product 和 attention score 都是 bilinear form。这个代数同构不是巧合——它意味着 ray 几何关系可以"嵌入"attention score 而不需要额外的计算通道。

**几何先验作为 inductive bias**：(D) 项在初始化时就是 Klein form，模型 step 0 就有"共面 ray 应该 attend"的几何先验。这比让模型从 pixel content 隐式学习 epipolar geometry 高效得多。

**Content-Geometry 耦合的独特性**：(B)+(C) cross-terms 是 RayPE 独有的结构。V-side injection、cross-attention、AdaLN、rotation-based PE 都无法表达"某类 content 偏好某类 ray"这种耦合。Ablation 显示这是性能的重要组成部分。

### 8.2 为什么 additive 优于 multiplicative？

- **保留 pretrained 知识**：multiplicative 替换 RoPE 破坏 pretrained 位置结构
- **零初始化稳定启动**：α = 0 时网络完全等于 pretrained DiT
- **可解释的 attention 分解**：四项分解 (A)(B)(C)(D) 各有明确物理意义

### 8.3 为什么 NGI 必要？

**Scale heterogeneity 是真实数据的核心挑战**。不同数据集的 translation scale 差异可达数量级（SfM normalized vs SLAM internal vs metric）。Direction/magnitude decomposition 把 scale-invariant 部分（direction）和 absolute scale（log-magnitude）分开，gate 让模型自适应。

**PE RMSNorm 对齐 content branch**：content branch 有 QKNorm，geometry branch 也必须有对称的 RMSNorm，否则 α 无法在 layer 间和样本间保持 consistent 含义。

### 8.4 与 Multi-View Geometry 的深层联系

Plücker reciprocal product 是 line geometry 的 Klein form，SE(3) 不变。这意味着 RayPE 的几何项自动满足：
- **视角不变性**：同一对 3D ray 在任意 world frame 下 reciprocal product 相同
- **共面判据**：观测同一 3D 点的两 ray 必共面，reciprocal product = 0

这给了模型一个**内置的 epipolar geometry 先验**，无需从数据学习。

---

## 9. 可能的延伸联想

### 9.1 与 Epipolar Attention 的关系
Epipolar Transformer (https://arxiv.org/abs/2107.10133) 显式约束 attention 沿 epipolar line。RayPE 通过 reciprocal product 隐式实现类似约束：共面 ray（包括 epipolar 线上的 ray 对）有高 attention score（reciprocal product 接近 0）。但 RayPE 是 soft 的、可学习的，而非 hard mask。

### 9.2 与 3D Gaussian Splatting 的联系
Plücker coordinates 在不同iable rendering (如 3DGS, https://arxiv.org/abs/2308.14737) 中也用于 ray-line 距离计算。RayPE 把同样的几何表示引入 generative transformer，是 3D 表示与生成模型的桥梁。

### 9.3 与 Implicit Neural Representations
Plücker coordinates 是 ray 的 **canonical representation**，SE(3) 等变。这与 NeRF 的 ray formulation、最近的工作如 PlückerGAN (https://arxiv.org/abs/2211.16487)、Ray Conditioning (https://arxiv.org/abs/2311.17084) 有思想上的连续性。

### 9.4 扩展到 Multi-View Diffusion
RayPE 天然适合 multi-view 设置：每个 view 的 token 都有自己的 Plücker ray，cross-view attention 自动获得 geometric prior。这可能是 multi-view diffusion (如 MVDream, https://arxiv.org/abs/2308.16512) 的自然扩展。

### 9.5 与 Geometric Algebra 的联系
Plücker coordinates 和 Klein form 是 Grassmann-Cayley algebra 的特例。Attention 的 bilinear form 与 geometric algebra 的 inner product 有结构相似性。更深的联系可能通过 Clifford algebra attention (https://arxiv.org/abs/2402.15410) 探索。

### 9.6 World Model 的视角
RayPE 提供了一个"3D-aware attention"的范式。如果 video diffusion 是 world model 的一种形式 (https://worldmodels.github.io)，那么 RayPE 注入的 ray geometry 可能是 world model 从 2D content 推理走向 3D scene reasoning 的关键一步。

### 9.7 与 Orbit Embeddings 的关系
Camera trajectory 通常有 orbit, dolly, pan 等结构化模式。RayPE 的 Plücker 表示自动编码这些：orbit 对应 $\mathbf{m}$ 方向固定、$\mathbf{d}$ 旋转；dolly 对应 $\mathbf{o}$ 沿 $\mathbf{d}$ 平移。这可能是为什么 RayPE 在 OOD（艺术画、电影 still）上泛化好——几何先验不依赖 content domain。

### 9.8 与 Diffusion 3D Consistency 的理论联系
跨帧 3D consistency 是 video diffusion 的难题。RayPE 通过几何先验让 attention 自动倾向"观测同一 3D 表面的 token"，这可能是 implicit 的 cross-frame consistency constraint。理论分析可能与 3D-aware score distillation (https://arxiv.org/abs/2305.12979) 相关。

---

## 10. 局限与未来方向

论文未充分讨论的：
1. **Dynamic scene**：Plücker ray 假设静态场景，dynamic object 的 ray-geometry 关系更复杂
2. **Non-pinhole camera**：当前假设 pinhole model，fisheye / omnidirectional 需要扩展
3. **Long video**：当前 81 帧，长视频的 camera trajectory 累积误差
4. **Interaction with text conditioning**：camera 和 text 信号如何平衡未深入分析
5. **Theoretical analysis of learned $M$**：$\|M - I_6\|$ 的演化轨迹未可视化

未来可能方向：
- 4D Plücker（时间维度扩展）处理 dynamic scene
- 与 3D representation（Gaussian, mesh）联合训练
- Physics-aware ray encoding（反射、折射 ray）

---

## 参考链接汇总

**论文与 Project**：
- RayPE Project: https://raype-project.github.io
- Wan2.2: https://arxiv.org/abs/2503.20314

**Baselines**：
- CameraCtrl: https://openreview.net/forum?id=Nw8r7Hc6t5
- ReCamMaster: https://arxiv.org/abs/2503.11647
- UCPE: https://arxiv.org/abs/2512.07237
- ReRoPE: https://arxiv.org/abs/2602.08068
- PRoPE: https://arxiv.org/abs/2507.10496
- ViewCrafter: https://arxiv.org/abs/2409.02048
- TrajectoryCrafter: https://arxiv.org/abs/2503.13136
- GS-DiT: https://arxiv.org/abs/2501.02690
- MotionCtrl: https://dl.acm.org/doi/10.1145/3658167.3658181
- CamCo: https://arxiv.org/abs/2406.02509

**背景技术**：
- RoPE: https://arxiv.org/abs/2104.09864
- 3D RoPE for ViT: https://arxiv.org/abs/2403.13276
- Plücker coordinates: https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates
- Hartley & Zisserman: https://www.cambridge.org/core/books/multiple-view-geometry-in-computer-vision/2B7C5D9F26D32D40C7A6C23A5C9A8A0A
- QKNorm: https://arxiv.org/abs/2010.04245
- RMSNorm: https://arxiv.org/abs/1910.07467

**相关延伸**：
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.14737
- MVDream: https://arxiv.org/abs/2308.16512
- Epipolar Transformer: https://arxiv.org/abs/2107.10133
- Clifford algebra attention: https://arxiv.org/abs/2402.15410
- DROID-SLAM: https://arxiv.org/abs/2108.10869
- RealEstate10K: https://arxiv.org/abs/1805.06191
- DL3DV: https://arxiv.org/abs/2312.16236

---

这篇论文的优雅之处在于：它发现了一个深刻的**代数同构**（Plücker reciprocal product 与 attention score 都是 bilinear form），并把这个同构转化为一个几乎零成本的工程实现（additive injection + Q/K flip + NGI），既保留了 pretrained 模型的能力，又注入了强几何先验。这种"数学洞察驱动工程实现"的风格，是高质量 research 的典范。对 build intuition 而言，最值得内化的是：**当两个看似不同的结构共享同一代数 shape 时，往往存在一个 canonical 的嵌入方式，让一个结构"活"在另一个内部**。
