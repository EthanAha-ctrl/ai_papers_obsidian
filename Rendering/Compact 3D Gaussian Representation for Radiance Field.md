---
source_pdf: Compact 3D Gaussian Representation for Radiance Field.pdf
paper_sha256: c7b9e5b8362dc60e9ae6910472b11c34993dc7b756247e36cbf85ae5fad297ac
processed_at: '2026-08-03T16:37:24-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版

## 这篇 paper 在干啥

3DGS 厉害是厉害 —— 渲染快、质量高、训练也不慢。但它有个让人头大的毛病：**太胖了**。一个 real-world scene 随随便便几百 MB 到 1 GB，你想塞手机上跑？想多了。

为什么这么胖？因为 3DGS 用几百万个小椭球（Gaussian）拼出一个场景，每个椭球要存 59 个 float：position 3 个、opacity 1 个、scale 3 个、rotation 4 个，剩下 **48 个全给 spherical harmonics 存颜色**。你看，光是颜色就吃掉 81% 的参数预算。这就好比你开个公司，员工名片上印了 48 个头衔，实际干活就一个。

这篇 paper 的核心观察就三句话：

**第一，很多 Gaussian 干脆就是废物。** 3DGS 训练时疯狂 densify，clone 又 split，最后留下一堆小到看不见、opacity 又低的 Gaussian，对渲染几乎没贡献，但照样占内存。

**第二，颜色有空间冗余。** 一面墙上的 Gaussians 颜色都差不多，但 3DGS 偏要每个 Gaussian 自己存一份 SH 系数，跟邻居互不搭理。

**第三，几何形状更没多样性。** 几百万个 Gaussians，scale 和 rotation 翻来覆去就那几种组合。本质上是 low-dimensional signal，却用 high-dimensional storage。

## 三个招数

**招数一：学会剪枝，而且剪得有讲究。**

3DGS 自己也有 pruning —— 把 opacity 太小的删掉。但作者发现这不够。一个 Gaussian 体积特别小的话，opacity 再高也没用，因为投影到屏幕上就一个像素都不够填。所以作者加了个 learnable mask，**同时看 volume 和 opacity**。每个 Gaussian 旁边挂一个 learnable parameter $m_n$，sigmoid 之后跟 threshold 比，决定这个 Gaussian 活不活。前向时是 hard 0/1（要么在要么不在），反向时用 straight-through estimator 让 $m_n$ 能训。再加个 sparsity loss 推着所有 $m_n$ 往下走，逼着模型自己挑出哪些 Gaussian 不重要。

这个 mask 还同时作用在 scale 和 opacity 上 —— scale 被 mask 成 0 的话，Gaussian 直接塌缩成一个点，连 rasterization 都懒得处理它。比单剪 opacity 更彻底。

结果：Gaussians 数量砍掉 2.4 倍，PSNR 居然不掉甚至略涨（剪掉的本来就是 overfit 噪声 Gaussian）。

**招数二：颜色不存了，去查表。**

SH 那 48 个参数太奢侈。作者换了个思路：用一个 **hash grid + 小 MLP** 来 represent 颜色。Gaussian 不再自己存颜色，它拿着自己的 position 去问 hash grid："哥们，我这位置啥颜色？" Hash grid 答一句，再根据 view direction 调一下，输出 RGB。

好处是啥？**相邻的 Gaussians 自然共享同一组 hash bins**，等于自动做了 spatial smoothing。一面墙上 1000 个 Gaussians 以前要存 1000 份 SH，现在共享一份 grid features。

为啥用 hash grid 不用别的？因为 hash grid（Instant NGP 那套）compact 又快，而且 hash collision 在 smooth signal（颜色就是 smooth 的）上基本无害。论文还做了 ablation：用 hash grid 同时去 represent opacity、scale、rotation，结果 PSNR 崩到 9.3 —— 因为那些 attribute 是 discontinuous 的，hash collision 直接灾难。**只有颜色适合 grid，几何不适合**。这是这篇 paper 设计选择的关键 insight。

**招数三：几何用 codebook。**

既然几何形状就那几种，那就建个"几何字典"。论文用 **Residual Vector Quantization (R-VQ)**：6 个 stage，每个 stage 64 个 code。第一个 stage 粗匹配，后面 stage 修 residual，级联逼近。最终每个 Gaussian 只存 6 个 index（每个 6 bits，共 36 bits），就替代了原来 7 个 float 的 scale+rotation。

直觉上 R-VQ 像 learned wavelet decomposition —— 第一个 stage 抓 low-frequency 几何，后面的 stage 抓 high-frequency 修正。论文可视化代码本 norm 逐 stage 衰减，从 0.97 跌到 0.007，正好印证这个 multi-scale 结构。

为啥 R-VQ 不一上来就训？因为早期 Gaussian 位置和几何还在大幅调整，codebook 容易陷入局部最优。所以论文前 29K iter 用 raw scale/rotation，最后 1K iter 才开 R-VQ，用 K-means 初始化 codebook。

## 效果

Mip-NeRF 360：storage 从 746 MB 砍到 29 MB（25x），PSNR 只掉 0.13 dB，FPS 还涨了点。

Deep Blending：storage 28x 压缩，**PSNR 反超 3DGS 0.38 dB**，FPS 快 32%。这是这篇 paper 最骚的结果 —— 压缩完质量反而更高，因为剪掉的冗余 Gaussians 本来就是在 overfit training view 的噪声。

NeRF-Synthetic：storage 26x 压缩，FPS 1.5x。

Post-processing 再加 buff：8-bit quantization + Huffman encoding + hash grid pruning，进一步砍 40%。

## 我的 takeaway

这篇 paper 真正教我的不是那三个技术 trick —— mask、codebook、hash grid 都不是新东西。**它的精髓在于"对症下药"**：

- Existence 这种 binary 信号 → learnable mask（稀疏 prior）
- Geometry 这种 low-DoF 离散信号 → codebook（template prior）
- Color 这种 spatially smooth 信号 → hash grid（continuity prior）

三个 attribute，三个完全不同的 representation，因为它们的 statistical property 完全不同。**与其一刀切 quantize，不如分析每个 attribute 的特性选最优 encoder**。这个思路比单纯堆 quantization 高明多了，也是后续 3DGS 压缩工作反复借鉴的核心思想。

一句话总结：**3DGS 不是参数太多，是没把参数用在该用的地方**。这篇 paper 把每个 attribute 的 representation 重新选了一遍，自然就瘦下来了。

---

# Compact 3D Gaussian Representation for Radiance Field 深度解析

## 1. Paper 全局动机：3DGS 的"参数肥胖症"

3DGS [Kerbl et al. 2023] 用一组 anisotropic 3D Gaussians 表示场景，每个 Gaussian 存储 59 个参数：position (3) + opacity (1) + scale (3) + rotation quaternion (4) + **spherical harmonics (48, max 3 degree)**。SH 占了 81% 的参数量。一个 realistic scene 动辄几百万个 Gaussians，单场景 >1GB。这跟 NeRF 的几 MB 模型相比，存储上完全不是一个量级。

Paper 的核心 thesis：3DGS 里面存在两层冗余 —— (1) Gaussian 数量冗余（densification 产生了大量小体积低贡献的 Gaussians），(2) attribute 冗余（相邻 Gaussians 颜色相似、几何形状相似）。两者分别处理。

**Reference**: 
- 3DGS project: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- 3DGS arXiv: https://arxiv.org/abs/2308.14737
- This paper project page: https://maincold2.github.io/c3dgs/
- This paper arXiv: https://arxiv.org/abs/2311.13681

---

## 2. 三大技术组件全解析

### 2.1 Learnable Volume Mask (Section 3.1)

**核心 insight**: 3DGS 的 opacity-based pruning 不够狠。作者 empirical 发现把 Gaussian 数量砍掉 2.42x，PSNR 几乎不掉（Figure 3）。但光看 opacity 不够 —— 一个小体积高 opacity 的 Gaussian 对渲染贡献也很小，因为它的体积小，投影到屏幕上的 footprint 极小。

**公式 1-2 (Straight-Through Estimator Masking)**:
$$
M_n = \mathrm{sg}(\mathbb{1}[\sigma(m_n) > \epsilon] - \sigma(m_n)) + \sigma(m_n)
$$

变量拆解：
- $n \in \{1, \dots, N\}$: Gaussian 索引，$N$ 随 densification 变化
- $m_n \in \mathbb{R}$: 每个 Gaussian 附带的 learnable mask logit
- $\sigma(\cdot)$: sigmoid function，把 logit 映射到 $(0,1)$
- $\epsilon$: mask threshold（论文用 0.5 隐式，paper 没明说但 straight-through 默认）
- $\mathbb{1}[\cdot]$: indicator function，返回 0 或 1
- $\mathrm{sg}(\cdot)$: stop-gradient operator，前向透传、反向梯度置零
- $M_n \in \{0, 1\}$: 最终 binary mask

**怎么理解 STE 在这里的作用**：前向时 $M_n = \mathbb{1}[\sigma(m_n) > \epsilon]$（hard 0/1），反向时 $\frac{\partial M_n}{\partial m_n} = \sigma'(m_n)$（soft gradient）。这样 $m_n$ 能被 SGD 推动。

$$
\hat{s}_n = M_n \cdot s_n, \quad \hat{o}_n = M_n \cdot o_n
$$

- $s_n \in \mathbb{R}_+^{3}$: Gaussian 的 scale（3D 各向异性 scale）
- $o_n \in [0, 1]$: opacity
- $\hat{s}_n, \hat{o}_n$: masked 后的 effective 属性

**关键 insight (build intuition)**: 当 $M_n = 0$ 时 $\hat{s}_n = 0$，意味着这个 Gaussian 被压成一个体积为 0 的点，渲染时 contribution 完全消失（α-blending 公式里 $T_i \alpha_i C_i$ 中 $\alpha_i \to 0$）。这比单纯把 opacity 置 0 更直接 —— 因为 scale→0 会让 3D Gaussian 在 splat 时投影面积趋于 0，从 rasterization pipeline 早期就剔除掉，连 fragment shader 都不用跑。

**公式 3 (Sparsity Loss)**:
$$
L_m = \frac{1}{N}\sum_{n=1}^{N} \sigma(m_n)
$$

- 这是把所有 mask logits 的 sigmoid 平均值当作正则项。直觉：mask logit 越大，对应 Gaussian 越想"活着"，loss 越鼓励 mask logit 整体往下推，最终大量 $m_n$ 跌破 $\epsilon$，对应 Gaussian 被剪掉。
- $\lambda_m$ 控制 aggressiveness：real scenes 用 $5 \times 10^{-4}$，synthetic 用 $4 \times 10^{-3}$。Synthetic 场景几何简单，可以更狠剪。

**对比 3DGS 原生 pruning**：3DGS 每 100 iter 把 opacity 设成 0.005，过段时间剪 opacity 仍很小的。这只看 transparency，不看 volume。Paper 的 Table 6 ablation 明确显示：volume-only mask (508K Gaussians, PSNR 31.89) > opacity-only mask (629K, 31.86) > 两者结合 (601K, 32.08)。**两者结合才是最优**。

**Figure 3 的训练动力学**：3DGS 在 15K iter 后停止 densify，Gaussians 数量稳定在 ~1.2M。Ours 在整个 30K iter 训练中持续 mask，最终稳定在 ~600K。意味着我们一边 densify 一边 mask，始终保持 lean state。

---

### 2.2 Geometry Codebook via Residual Vector Quantization (Section 3.2)

**核心 insight**: 场景由很多小 Gaussians 组成，每个 Gaussian 不需要 express 太多几何多样性 —— 大量 Gaussians 共享类似的 (scale, rotation) 组合。Codebook 是 natural fit。

**为什么不直接 VQ 而用 R-VQ**？Naive VQ 要么 codebook 大（GPU memory 爆炸），要么 quantization error 大。R-VQ [SoundStream, Zeghidour et al. 2021] 把量化误差残差再喂到下一 stage 的 codebook，级联逼近。

**公式 4-5 (R-VQ Forward)**:
$$
\hat{r}_n^l = \sum_{k=1}^{l} \mathcal{Z}^k[i^k], \quad l \in \{1, \dots, L\}
$$

- $r_n \in \mathbb{R}^{4}$: 第 $n$ 个 Gaussian 的 rotation quaternion（4D）
- $\hat{r}_n^l \in \mathbb{R}^{4}$: 经过 $l$ 个 stage 量化后的 cumulative 估计
- $L$: 总 stage 数，paper 中 $L = 6$
- $\mathcal{Z}^k \in \mathbb{R}^{C \times 4}$: 第 $k$ stage 的 codebook，$C = 64$（每个 codebook 64 个 4D vectors）
- $i_n^k \in \{0, \dots, C-1\}$: 第 $n$ 个 Gaussian 在第 $k$ stage 选中的 codebook index

$$
i_n^l = \arg\min_k \|\mathcal{Z}^l[k] - (r_n - \hat{r}_n^{l-1})\|_2^2, \quad \hat{r}_n^0 = \vec{0}
$$

**Build intuition**: 
- Stage 1: 找 codebook 里最接近 $r_n$ 的 code（coarse approximation）
- Stage 2: 量化 residual $r_n - \hat{r}_n^1$（fine correction）
- 以此类推，6 stage 级联，每 stage 64 个 options，等效 codebook size 是 $64^6 \approx 6.8 \times 10^{10}$，但实际存储只用 $6 \times 64 \times 4 \times 4\text{bytes} = 6\text{KB}$ 的 codebook 加上每 Gaussian $6 \times \log_2 64 = 36$ bits 的 indices。

**公式 6 (Commitment + Codebook Loss)**:
$$
L_r = \frac{1}{NC}\sum_{k=1}^{L}\sum_{n=1}^{N}\|\mathrm{sg}[r_n - \hat{r}_n^{k-1}] - \mathcal{Z}^k[i_n^k]\|_2^2
$$

- $\mathrm{sg}[\cdot]$: stop-gradient on input（residual），让 codebook vectors 主动靠近 residuals
- 这是 VQ-VAE 经典 commitment loss 的多 stage 推广
- Paper 还提到 EMA codebook update，但 paper 文本里没明写，应该是省略了
- 同样的 loss $L_s$ 应用到 scale $s_n \in \mathbb{R}^3_+$（注意 scale 是非负的，量化时小心）

**训练策略细节**：只在最后 1K iterations 应用 R-VQ（前面 29K iter 直接用 raw rotation/scale），避免早期训练陷入 codebook 局部最优。K-means 初始化 codebook。

**Figure 7-(b) 的可视化非常有 insight**：
- Stage 1: 平均 code norm = 0.9752（codes 幅值大，coarse 几何）
- Stage 2: 0.1893
- Stage 3: 0.0765
- Stage 5: 0.0156
- Stage 6: 0.0074

后期 stages 处理越来越小的残差，呈现 **multi-scale 分解**，类似 wavelet decomposition 的 coarse-to-fine 结构。这就是为什么 R-VQ 比单层 VQ 表达力强 —— 它自然地分离了 low-frequency 和 high-frequency 几何。

**Reference**:
- SoundStream (R-VQ origin): https://arxiv.org/abs/2106.06969
- VQ-VAE: https://arxiv.org/abs/1711.00937

---

### 2.3 Compact View-dependent Color via Hash Grid (Section 3.3)

**核心 insight**: SH 48 params per Gaussian 是冗余重灾区。相邻 Gaussians 颜色高度相关（同一表面），但 3DGS 把每个 Gaussian 当独立 SH 个体。空间相关性被浪费了。Grid-based neural field (Instant NGP) 恰好擅长 encode 空间连续的 signal。

**公式 7-8 (Neural Field Color)**:
$$
c_n(d) = f(\mathrm{contract}(p_n), d; \theta)
$$

- $c_n(d) \in \mathbb{R}^3$: RGB color
- $d \in \mathbb{R}^3$: view direction（从 camera center 到 Gaussian 中心）
- $p_n \in \mathbb{R}^3$: Gaussian 的 3D position
- $\theta$: hash grid + MLP 的参数
- $f(\cdot ; \theta)$: neural field，先 hash grid lookup 再 tiny MLP

$$
\mathrm{contract}(p_n) = \begin{cases} p_n & \|p_n\| \leq 1 \\ \left(2 - \frac{1}{\|p_n\|}\right)\left(\frac{p_n}{\|p_n\|}\right) & \|p_n\| > 1 \end{cases}
$$

- 这是 Mip-NeRF 360 [Barron et al. 2022] 的 contraction，把 unbounded scene 压到 unit ball 内
- $\|p_n\| > 1$ 时：在球外，沿径向压缩，$(2 - 1/r) \cdot (p/r)$ —— $r \to 1^+$ 时连续过渡到 $r = 1$，$r \to \infty$ 时 $|contract(p)| \to 2$，整个 $\mathbb{R}^3$ 被压到半径 2 的球内
- 这是 **必须的**，因为 hash grid size 有限，必须先 bounded 化

**Hash Grid 架构细节**:
- 16 个 resolution levels，从 16 到 4096（几何级数）
- 每 level 2-channel features（合计 32 channels）
- Max hashmap size: $2^{19}$ for real scenes, $2^{16}$ for synthetic
- 后接 2-layer 64-channel MLP

**初始 color 用 0-degree SH**: 不是直接 random init，而是用 3-channel constant color 初始化（0-degree SH = DC component，等价于 view-independent base color）。Paper 报告这比直接用 RGB 表示稍好。

**Build intuition**: 邻近 Gaussians 共享同一组 hash grid bins（space hashing 的 local coherence），自然产生 spatial smoothing。一个 hash bin 被多个 Gaussians 引用，等效于 "color template sharing"。这其实跟 codebook 的 share 思想是一致的，只是 hash grid 是 implicit 的、空间自适应的 template。

**为什么不用 I-NGP 表示所有 attributes**？Paper Table 7 给出 ablation：用 I-NGP 同时表示 opacity + scale + rotation + color，PSNR 崩到 9.3（vs 32.2 baseline）。原因：opacity/scale/rotation 是 **高频且 discontinuous** 的 attribute（不同 surface 上的 Gaussians 几何差异大），hash grid 在 discontinuity 处 hash collision 严重。Color 是低频 smooth signal，适合 grid。

**Reference**:
- Instant NGP: https://nvlabs.github.io/instant-ngp/
- Mip-NeRF 360: https://jonbarron.info/mipnerf360/

---

## 3. 总 Loss 与训练流程

**公式 9**:
$$
L = L_{ren} + \lambda_m L_m + L_r + L_s
$$

- $L_{ren}$: L1 + SSIM 加权（同 3DGS）
- $\lambda_m$: real $5e^{-4}$, synthetic $4e^{-3}$
- $L_r, L_s$: 只在 last 1K iter 开启，避免 codebook 早期干扰主训练
- 总训练 30K iter

**关键流程顺序**:
1. 前 29K iter：用 raw scale/rotation 训练 Gaussians + mask + neural field color
2. 最后 1K iter：开启 R-VQ，K-means 初始化 codebook，commitment loss 启动

**Reference**: 
- SSIM: https://en.wikipedia.org/wiki/Structural_similarity

---

## 4. Post-Processing Pipeline (Section 4.1)

简单但有效的后处理：
1. **8-bit min-max quantization** for opacity 和 hash grid params
2. **Hash grid pruning**: 删 value < 0.1 的 hash entry（I-NGP hash table 本身稀疏）
3. **Huffman encoding** [Huffman 1952] for quantized opacity + hash params + R-VQ indices

**Table 5 数据拆解** (Mip-NeRF 360 avg):
| Attribute | 3DGS (32f) | Ours (16f) | Ours+PP |
|---|---|---|---|
| Position | 37.9 MB | 8.3 | 8.3 |
| Opacity | 12.6 | 2.8 | 1.2 (8b+Huffman) |
| Scale | 37.9 | 6.3 | 5.9 |
| Rotation | 50.6 | 6.3 | 6.2 |
| Color (SH/Hash) | 606.9 (SH) | 25.2 (Hash 16f) + 0.016 (MLP) | 7.4 (8b+prune+Huffman) + 0.016 |
| **Total** | **746 MB** | **48.8 MB** | **29.1 MB** |

**直觉**: 
- 最大 win 来自 color attribute：606.9 → 25.2 MB（24x reduction），单单这一项就追平了 3DGS 的总瓶颈
- Position 用 16-bit half tensor 已经够精度
- Scale + Rotation 用 R-VQ 编码，原 88.5 MB → 12.6 MB（7x reduction）
- Post-processing 进一步把 hash grid 从 25.2 压到 7.4 MB（hash grid 本身稀疏，prune 后 Huffman 极有效）

**Reference**:
- Huffman 1952 paper: https://ieeexplore.ieee.org/document/405111

---

## 5. 实验结果详读

### 5.1 Mip-NeRF 360 (Table 1)

| Method | PSNR | FPS | Storage |
|---|---|---|---|
| 3DGS | 27.21 | 134 | 734 MB |
| 3DGS* | 27.46 | 120 | 746 MB |
| Ours | 27.08 | 128 | 48.8 MB |
| Ours+PP | 27.03 | - | 29.1 MB |

**Storage 25x reduction, PSNR 掉 0.13, FPS 持平**。这 trade-off 极佳。

### 5.2 Deep Blending (Table 2) - **这里 paper 真正胜过 3DGS**

| Method | PSNR | FPS | Storage |
|---|---|---|---|
| 3DGS | 29.41 | 137 | 676 MB |
| Ours | **29.79** | **181** | 43.2 MB |
| Ours+PP | 29.73 | - | 23.8 MB |

**PSNR 反超 3DGS（+0.38 dB），FPS 快 32%，Storage 28x reduction**。这是 paper 的 strong evidence：masking 不仅不损失质量，反而可能提升质量 —— 因为剪掉冗余 Gaussians 减少了 overfit noise（很多冗余 Gaussians 在拟合 training view 的 noise，剪掉后泛化更好）。

### 5.3 NeRF-Synthetic (Table 3)

| Method | PSNR | Storage | FPS |
|---|---|---|---|
| 3DGS | 33.32 | 68.1 MB | 359 |
| Ours | 33.33 | 5.55 MB (0.08x) | 545 (1.52x) |
| Ours+PP | 32.88 | 2.67 MB (0.04x) | - |

Storage 26x reduction，FPS 1.52x。

### 5.4 Per-scene Analysis (Table 9, Mip-NeRF 360)

| Scene | 3DGS #Gauss | Ours #Gauss | Reduction |
|---|---|---|---|
| bicycle | 5.72M | 2.22M | 2.57x |
| garden | 5.64M | 2.21M | 2.55x |
| bonsai | 1.25M | 0.60M | 2.08x |
| room | 1.48M | 0.53M | 2.80x |

Outdoor scenes (bicycle, garden) reduction 比 indoor 大（outdoor 有更多 sky/ground 冗余 Gaussians）。

---

## 6. 关键 Ablation Tables 解析

### 6.1 Table 4 - 完整 ablation (Playroom + Bonsai)

| M | C | G | H | PSNR | #Gauss | Storage | FPS |
|---|---|---|---|---|---|---|---|
|  |  |  |  | 29.87 (Play) | 2.34M | 553 MB | 154 |
| ✓ |  |  |  | 29.91 | 967K | 228 MB | 254 |
| ✓ | ✓ |  |  | 30.33 | 770K | 59 MB | 210 |
| ✓ | ✓ | ✓ |  | 30.33 | 761K | 44 MB | 204 |
| ✓ | ✓ | ✓ | ✓ | 30.32 | 778K | 38 MB | 206 |

**逐 component 收益拆解**（Playroom）：
- **Mask only**: 2.34M → 967K Gaussians（2.4x reduction），Storage 553→228 MB（主要来自 Gaussians 数量减少带来的所有 attribute 同比缩），PSNR +0.04，FPS +100
- **+ Color**: Storage 228→59 MB（4x reduction），#Gauss 也略降（因 mask 在 color 部分 hash 共享使更多 Gaussians 可剪），PSNR +0.42
- **+ Geometry Codebook**: 59→44 MB（1.34x），PSNR 不变
- **+ Half tensor**: 44→38 MB（1.16x）

**Build intuition**: 
- Masking 是 Gaussians 数量层面的瘦身
- Color 是 attribute 维度的瘦身（最大头）
- Codebook 是 geometry attribute 维度的瘦身
- Half tensor 是精度维度的瘦身
- 四者正交且互补

### 6.2 Table 7 - I-NGP 表示什么 attribute 的 ablation (Bonsai)

| Opa | Sca | Rot | Col | PSNR | #Gaussians |
|---|---|---|---|---|---|
|  |  |  |  | 32.2 (3DGS) | 1245K |
|  |  |  | ✓ | 32.3 | 1178K |
| ✓ |  |  | ✓ | 9.3 | 666K |
| ✓ | ✓ |  | ✓ | 9.3 | 559K |
| ✓ | ✓ | ✓ | ✓ | 25.9 / 27.37 | 1692K |

**用 I-NGP 表示 opacity/scale/rotation 直接崩溃**（PSNR 9.3）。Hash collision 在 discontinuous signals 上 catastrophic。但只表示 color 完美（PSNR 32.3，甚至超过 3DGS 32.2）。**这验证了 paper 的核心 design choice**：geometry 用 codebook（discrete、template-based），color 用 hash grid（continuous、spatially coherent）。

### 6.3 Table 8 - GPU Memory (推理时)

| Scene | 3DGS Mem | Ours Mem |
|---|---|---|
| bicycle | 9.4 GB | 7.6 GB |
| bonsai | 8.7 GB | 8.3 GB |
| playroom | 6.4 GB | 5.6 GB |

GPU memory 也有 1-2 GB 节省（推理时）。

---

## 7. 与 3DGS 训练动力学对比

3DGS 训练在 ~15K iter 停止 densification，之后 Gaussians 数量固定。Paper 的方法持续 mask 整个 30K iter，Gaussians 数量持续下降。

**为什么持续 masking 好**？因为场景中 distant background、floater、redundant Gaussians 在早期 densification 阶段还在贡献 training view 的 reconstruction loss，过早 mask 会损失质量。后期这些 Gaussians 已经"被替代"（其他 Gaussians 接管了它们的 contribution），mask 掉完全 free。

---

## 8. Inference Speedup 来源 (Section 7)

Paper Appendix 提到三方面 inference speedup：

1. **Gaussians 数量减少**：rasterization 是 $O(N)$ 算法，N 小直接快
2. **Hash grid features precompute**：grid features 不依赖 view direction（只有 MLP 后段依赖 $d$），可预先 cache。Test 时只需 MLP forward
3. **Codebook lookup 预 index**：test 时直接查表，无需 argmin search

第二、三点是工程优化，paper 没详细展开代码层面，但很重要 —— 没这些，neural field + R-VQ 反而比 3DGS raw attribute 慢。

---

## 9. Paper 的局限 / 可改进点

1. **Training time 反而增加**（33m vs 24m on Mip-NeRF 360），因为 neural field forward + R-VQ search 开销
2. **GPU memory saving 不显著**（Table 8），主要 win 是 storage 不是 GPU memory。原因：hash grid + codebook 还是要常驻 GPU
3. **PSNR 略掉**（Mip-NeRF 360 上掉 0.13 dB，主要是 high-frequency detail 受损），Deep Blending 反超属例外
4. **Codebook 在极端低 bitrate 下崩**（Figure 8 R-D curve）：当 R-VQ stages 极少时 PSNR 急剧下降。Codebook 表达力有 sharp cliff
5. **没考虑 dynamic scene 扩展**

**Reference**:
- EAGLES (后续改进 3DGS compression): https://arxiv.org/abs/2312.04564
- LightGaussian: https://arxiv.org/abs/2311.12945
- Gaussian Surfels: https://arxiv.org/abs/2311.04689

---

## 10. 跟同期/后续工作对比

| Method | Approach | Compression | PSNR Drop |
|---|---|---|---|
| **This paper (C3DGS)** | Mask + Hash + R-VQ | 25-28x | -0.1 to +0.4 |
| LightGaussian [2023] | SH pruning + quantization + perceptual distill | 10-20x | ~0 |
| EAGLES [2023] | Anchor-based Gaussians | ~10x | -0.1 |
| Mini-Splatting [2024] | Importance-based sampling | 10-30x | -0.1 |
| HAC [2024] | Hash-grid + entropy-constrained | 40-100x | -0.2 |

C3DGS 是最早系统性组合 mask + grid color + codebook 的 paper 之一，后续工作（HAC 等）在其基础上进一步用 entropy coding + 更激进的 anchor-based representation。

**Reference**:
- HAC (Hierarchical Compression): https://arxiv.org/abs/2403.11330
- Mini-Splatting: https://arxiv.org/abs/2401.15044
- Survey on 3DGS compression: https://arxiv.org/abs/2407.08766

---

## 11. 直觉性总结（Karpathy-style）

整篇 paper 其实是**信号处理 + 神经渲染**的 cross-pollination：

- **Volume masking**: 把 mask 当作稀疏 prior 引入，类似 model pruning 但不依赖 magnitude heuristic。同时 scale→0 + opacity→0 的 dual mask 比 单 opacity 更 principled —— 因为它从 **contribution** 而非 **attribute** 角度判断冗余
- **R-VQ for geometry**: 一个 scene 由 millions of Gaussians 构成，但 (scale, rotation) 的 **intrinsic degrees of freedom** 远低于 Gaussians 数量。Codebook 恰好是低-DoF signal 的最优 encoder。R-VQ 是 multi-resolution VQ，类似 learned wavelet decomposition
- **Hash grid for color**: Color 是 **spatially smooth** signal，hash grid 是 sparse smooth signal 的最优 adaptive discretization。3DGS 用 SH 是 per-point redundancy，hash grid 用 spatial coherence 消除它

**核心设计哲学**: 不同 attribute 用不同的 **representation prior**：
- Geometry: discrete codebook（attributes 是离散 template 的混合）
- Color: continuous hash grid（attributes 空间连续）
- Existence: learnable mask（稀疏 prior）

这是 paper 真正贡献 —— 它不是简单把所有 attribute 都 quantize，而是分析每个 attribute 的 **statistical property** 选最优 representation。这种"对症下药"思路在后续 3DGS 压缩工作中被反复借鉴。

---

## 12. Personal Thoughts & Extensions

如果让我（hypothetically speaking）改这篇 paper：

1. **Mask strategy 升级**: 现在 mask 是 per-Gaussian independent。考虑 **structured pruning**（按 region/voxel 整组 mask），减少 bits per mask decision
2. **Codebook EMA**: 用 EMA update codebook 而非 gradient descent（VQ-VAE 经典做法），避免 commitment loss 的不稳定
3. **Joint R-D optimization**: 当前 post-processing 是 detached pipeline。端到端 entropy-constrained 训练（类似 learned image compression）能进一步压
4. **Color field 也用 codebook**: Hash grid 仍有 25 MB（占总量一半），可以用 **codebook of color templates** 或 **low-rank tensor decomposition**
5. **End-to-end rate-distortion loss**: 把 storage 作为 constraint 加进 loss，而非 multi-stage post-processing

**Reference - Recent follow-ups**:
- CompGS: https://arxiv.org/abs/2405.18000
- RadSplat: https://arxiv.org/abs/2403.13534
- 3DGS 综述: https://arxiv.org/abs/2405.11110

---

## 13. 实现细节补充 (Appendix 6)

- Codebook size $C = 64$, stages $L = 6$
- Hash grid: 16 resolutions from 16 to 4096, 2-channel features per level
- MLP: 2 layers, 64 hidden channels
- Real scene: max hash map size $2^{19}$, $\lambda_m = 5 \times 10^{-4}$, mask lr $10^{-2}$, neural field lr $10^{-2}$，neural field lr decay 在 5K, 15K, 25K iter（factor 0.33）
- Synthetic scene: max hash map size $2^{16}$, $\lambda_m = 4 \times 10^{-3}$, mask lr $10^{-3}$, neural field lr $10^{-3}$，decay 在 25K

**Synthetic scene 用更小 hash map** —— 因为 synthetic scenes 物体尺度小、bounded。

---

## 14. 总结一行

这篇 paper 的精髓在于：**对 3DGS 的不同 attribute 用不同的统计 prior 来选 representation** —— existence 用 learnable mask（稀疏）、geometry 用 R-VQ codebook（离散 template）、color 用 hash grid（空间平滑），三者正交组合加 post-processing 实现 25-28x storage reduction 同时质量几乎不损甚至略升，在 Deep Blending 上 SOTA。

**Reference Repo**:
- Official repo: https://github.com/maincold2/c3dgs
- 3DGS repo: https://github.com/graphdeco-inria/gaussian-splatting
- Instant NGP repo: https://github.com/NVlabs/instant-ngp
