---
source_pdf: Quantile Rendering.pdf
paper_sha256: 1f5290dcaac00d229bbbe82941dffc53b704924032a31d970ab29508b01ff5d4
processed_at: '2026-08-06T07:29:30-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Quantile Rendering — 用人话讲

## 一句话总结

渲染 512 维 CLIP feature 太慢了，作者发现沿着一条 ray，真正"有话语权"的 Gaussian 其实很少，所以只挑 K 个"代表"来算，速度提升 43 倍，精度几乎不掉。

---

## 问题到底出在哪

3D-GS 渲染一个 pixel，就是沿 ray 把一堆 Gaussian 做 alpha-blending：

```
pixel_color = Σ (w_i × c_i)
```

这里 `c_i` 是 3 维 RGB，所以即便一条 ray 有 100 个 Gaussian，也就是 300 次乘法，GPU 秒秒钟搞定。

但你想渲染 CLIP feature 时，`c_i` 变成了 512 维向量。100 个 Gaussian × 512 维 = 51200 次乘法，per pixel。一张 1600×1063 的图有 170 万 pixel，那就得算 870 亿次乘法。Table 5 里 LangSplat 用 for-loop 渲染 512-D feature 只有 **0.65 FPS**，基本没法用。

之前的 workaround 是把 512 维压成 3 维（LangSplat）或 6 维（OpenGaussian），用 codebook 或 autoencoder。但这会丢信息，segmentation 质量下降。作者的核心 claim 是：**你不需要压缩 feature，你需要的是减少采样**。

---

## 核心 intuition：大部分 Gaussian 都是"凑数的"

想象你站在一条 ray 上往前看。你会遇到一堆 Gaussian，但它们对最终 pixel 的贡献 $w_i = T_i \alpha'_i$ 分布极度不均匀：

- 前面几个 opaque 的 Gaussian 把 transmittance $T$ 从 1 砸到 0.1
- 后面 80 个 Gaussian 的 $T$ 已经接近 0，它们的 $w_i$ 微乎其微
- 但 volume rendering 依然老老实实把它们全算一遍

作者的关键观察：**transmittance $T$ 从 1 单调递减到 0，这条曲线的"信息量"集中在少数几个 drop 巨大的点上**。与其在 spatial domain（沿 ray 距离 $t$）均匀采样，不如在 transmittance domain（$T$ 值）上均匀采样。

---

## Q-Render 怎么做：transmittance domain 上的 Riemann sum

### 数学换元

标准 volume rendering 的连续形式：

$$C_{\text{vol}} = \int_0^\infty c(t) \sigma(t) T(t) \, dt$$

- $t$: 沿 ray 的距离
- $\sigma(t)$: volume density
- $T(t) = \exp(-\int_0^t \sigma(s) ds)$: transmittance
- $c(t)$: 该位置的颜色/feature

关键 substitution：令 $u = T(t)$，则 $du = -\sigma(t) T(t) dt$，积分变成：

$$C_{\text{vol}} = \int_0^1 c(u) \, du$$

这个变换告诉你：**volume rendering 本质上是在 transmittance 区间 [0,1] 上积分 feature 函数 $c(u)$**。

### Q-Render 的做法

把 [0,1] 均匀分成 K+1 段：

```
0 = u_{K+1} < u_K < ... < u_1 < u_0 = 1
```

每段取右端点的 Gaussian 作为"代表"，做 Right Riemann Sum：

$$C_Q = \sum_{k=1}^{K+1} c(u_k) \cdot \Delta u$$

其中 $\Delta u = \frac{1}{K+1}$。

**直觉**：你不再关心"沿 ray 第几米处有什么 Gaussian"，而是关心"transmittance 从 1 降到 0.9 时遇到了谁，从 0.9 降到 0.8 时遇到了谁"。这些"谁"就是 Quantile Gaussian，只有 K 个。

### Algorithm 1 逐行解读

```
输入：3D Gaussians G, features F, K, ray上的Gaussian索引I
初始化：T=1 (累积transmittance), T_Q=1 (Quantile的transmittance), f_Q=0, k=0

for 每个Gaussian i (按深度排序):
    T_test = T × (1 - α'_i)          # 试探：过了这个Gaussian后T变成多少
    
    if T_test < 1 - (k+1)/(K+1):     # 关键判断：T跌穿了第k个quantile边界
        k += 1
        w_Q = T_Q × α'_i             # 这个Gaussian的Quantile权重
        f_Q += w_Q × f_i             # 累加feature
        T_Q *= (1 - α'_i)            # 更新Quantile transmittance
        
        while T_test < 1 - (k+1)/(K+1):  # 可能一个Gaussian跨多个quantile
            k += 1
        end
    end
    
    if T_test < 1/(K+1):             # T已经接近0，提前终止
        break
    end
    
    T = T_test                        # 更新真实transmittance
end

f_Q_normalized = f_Q / (1 - T_Q)     # 归一化：补偿未采完的transmittance
return f_Q_normalized
```

**几个关键细节**：

1. **为什么判断条件是 `T_test < 1 - (k+1)/(K+1)`**：quantile 边界是 $1, \frac{K}{K+1}, \frac{K-1}{K+1}, ..., \frac{1}{K+1}, 0$。当 T 跌破某个边界，说明这个 Gaussian 跨越了该 quantile，就选它。

2. **while 循环处理一个 Gaussian 跨多个 quantile 的情况**：如果一个 Gaussian 特别大特别 opaque，它的 $\alpha'_i$ 很大，T 一下从 0.9 跳到 0.3，跨越了 6 个 quantile，那 k 要连加 6 次。但这个 Gaussian 只被采样一次（用它的 $w_Q$ 算一次）。

3. **归一化 `f_Q / (1 - T_Q)`**：因为只采了 K 个 Gaussian，最后的 $T_Q$ 可能还没到 0（还剩一些 transmittance 没用完）。归一化相当于把"没采到的部分"按比例补偿回来，让结果逼近完整 volume rendering 的 $\sum w_i = 1$。

---

## 复杂度对比（Table 1）

| 方法 | 复杂度 | 说明 |
|------|--------|------|
| V-Render (原始) | $\mathcal{O}(NC)$ | N个Gaussian × C维feature，全算 |
| top-K (Dr.Splat) | $\mathcal{O}(N\log K + KC)$ | 要先排序再选top-K，排序有log因子 |
| Q-Render (本文) | $\mathcal{O}(N + KC)$ | 遍历N个Gaussian只算transmittance(标量)，K个Quantile才算feature(向量) |

**关键区别**：Q-Render 遍历所有 N 个 Gaussian 时只做标量运算（算 T），只有被选中的 K 个才做向量运算（乘 C 维 feature）。top-K 要排序，多了 $\log K$ 因子，而且 Figure 3 显示 top-K 的 transmittance 分布和原始 V-Render 差距大，Q-Render 更接近。

---

## GS-Net：不只是 renderer，还有 generalizable 的 feature predictor

### 为什么要 neural network

之前的方法（LangSplat, OpenGaussian, Dr.Splat）都是 **per-scene optimization**：每个场景单独训一组 3D-GS + feature，换个场景就得重训。作者想做一个 **generalizable** 的网络：输入任意场景的 3D-GS，直接输出每个 Gaussian 的 CLIP feature。

### 架构

```
优化好的3D-GS → voxelization → 3D神经网络 → de-voxelization → Gaussian features → Q-Render → feature map
```

具体：
1. **Voxelization**：把每个 Gaussian 的 center $\mu$ 采样到 sparse voxel grid（SplatFormer 的做法），grid size 5cm 效果最好（Table 4）
2. **3D backbone**：试了四种（Table 10）
   - MinkUNet（sparse convolution）：效果最好，50.75 mIoU
   - PTv3（Point Transformer V3）：48.99 mIoU，但容易过拟合
   - PointNet++：39.42 mIoU，不行
   - PointNeXT：37.89 mIoU，更不行
3. **De-voxelization**：把 voxel feature 还原回 per-Gaussian feature
4. **Q-Render**：渲染成 2D feature map

### 为什么 voxel-based 比 point-based 好（Table 10）

3D-GS 的 Gaussian 在物体表面密集聚集，point-based 方法用 k-NN 聚合邻域，receptive field 受限于局部点密度，抓不到 broad context。Voxel-based 用 sparse convolution，对密度变化更 robust。

### Training loss（Eq.2）

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(\tilde{\mathbf{f}}^Q, \mathbf{f}_i^{\text{CLIP}}))}{\sum_{i \neq j} \exp(\text{sim}(\tilde{\mathbf{f}}^Q, \mathbf{f}_j^{\text{CLIP}}))}$$

- $\tilde{\mathbf{f}}^Q$: Q-Render 渲染出的 feature
- $\mathbf{f}_i^{\text{CLIP}}$: Grounded-SAM2 提取的 mask 对应的 CLIP feature
- $\text{sim}(\cdot, \cdot)$: cosine similarity
- 这是一个 **InfoNCE / contrastive loss**：让渲染 feature 和正确 CLIP feature 的相似度高于其他所有 CLIP feature

---

## 实验结果

### ScanNet OVS（Table 2）

| Method | Per-scene? | 19-class mIoU |
|--------|-----------|---------------|
| LangSplat | ✓ | 1.47 |
| OpenGaussian | ✓ | 22.60 |
| Dr.Splat | ✓ | 23.21 |
| **GS-Mink (ours)** | ✗ (generalizable) | **50.75** |
| GS-PTv3 (ours) | ✗ | 48.99 |

GS-Net 是 generalizable 的（训练时没见过测试场景），但 mIoU 翻倍。即便 overfit 单场景，还能再提 12%。

### 速度（Table 5）

| Method | Feature dim | FPS |
|--------|-------------|-----|
| LangSplat | 3 | 112.12 |
| LangSplat | 512† (for-loop) | 0.65 |
| OpenGaussian | 6 | 71.13 |
| OpenGaussian | 512† | 0.83 |
| **GS-Mink (ours)** | **512** | **28.42** |

**28.42 / 0.65 ≈ 43.7× 加速**，而且用的是完整 512-D feature，没压缩。

### K 的 ablation（Table 6, Figure 6）

- K=5: mIoU 39.16
- K=10: mIoU 42.18（开始 converge）
- K=40: mIoU 45.81（最优）
- K=50: mIoU 45.71（饱和）

有趣的是：K=40 的 Q-Render 比 V-Render（用全部 Gaussian）的 mIoU 还高一点（50.75 vs 49.02，Table 11）。作者猜测是因为 3D-GS 本身有 noise（floaters、local minima），Q-Render 的稀疏采样反而起到了 **denoising** 效果。

### Robustness（Table 12）

给 opacity 加 Gaussian noise：
- noise scale 0.5: mIoU 几乎不变（50.18）
- noise scale 1.0: 降到 47.13
- noise scale 4.0: 崩到 16.12

说明 Q-Render 对 moderate noise 鲁棒，但严重几何扭曲会打垮它。

---

## 理论分析（Appendix C）

### 收敛率

Q-Render 是 transmittance domain 上的 Right Riemann Sum，误差 bound：

$$|C_{\text{vol}} - \tilde{C}_Q| \leq \frac{M}{2K}$$

- $M$: $c'(u)$ 的上界，即 feature 函数在 transmittance domain 的最大变化率
- $K$: Quantile 数量
- 收敛率 $\mathcal{O}(1/K)$，线性收敛

**直觉**：feature 在 transmittance domain 越平滑（$M$ 小），K 可以越小。如果 feature 突变剧烈（比如物体边界），需要更大 K。但实际上 K=10 就基本 converge 了。

### 归一化的作用

$$\tilde{C}_Q = \frac{C_Q}{1 - T_Q}$$

因为 $T_Q \leq \frac{1}{K+1}$，所以归一化因子 $\frac{1}{1-T_Q} \leq \frac{K+1}{K}$，当 K 大时趋近 1。归一化补偿了"没采到的尾部 transmittance"，让结果更逼近完整 volume rendering。

---

## 我的联想和思考

### 1. 和 NeRF 采样策略的对比

NeRF 用 stratified sampling + hierarchical coarse-to-fine，在 **spatial domain** 采样。Q-Render 在 **transmittance domain** 采样。本质上都是 importance sampling，但 transmittance domain 更直接地对应了"对最终 pixel 的贡献"。

Mip-NeRF 360 ([Barron et al., 2022](https://arxiv.org/abs/2111.12077)) 也做了类似的 integration domain 变换，但在 cone tracing 框架下。Q-Render 更简洁。

### 2. 和 DVGO / Plenoxels 的关系

DVGO ([Sun et al., 2022](https://arxiv.org/abs/2111.11215)) 和 Plenoxels ([Fridovich-Keil et al., 2022](https://arxiv.org/abs/2112.05131)) 也是 dense grid + alpha blending，它们的 rendering 和 3D-GS 一样面临 high-dim feature 的计算瓶颈。Q-Render 可以直接 plug-in。

### 3. 和 Dr.Splat top-K 的本质区别

Dr.Splat ([Jun-Seong et al., 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_Dr._Splat_Directly_Referring_3D_Gaussian_Splatting_via_Direct_Language_CVPR_2025_paper.pdf)) 选 top-K 是按 $\alpha'_i$ 排序，选最大的 K 个。问题是：
- 排序有 $\mathcal{O}(N\log K)$ 复杂度
- $\alpha'_i$ 大不代表 $w_i = T_i \alpha'_i$ 大（前面的 Gaussian $T$ 大但 $\alpha'$ 可能小，后面的反过来）
- Figure 3 显示 top-K 的 transmittance profile 和 V-Render 差距明显

Q-Render 的采样天然按 transmittance 分布来，不需要排序，且 profile 更接近 V-Render。

### 4. 和 point-based rendering 的联系

3DGS-based point rendering（如 [Pulsar](https://github.com/torch-points3d/torch-points3d)）也是 sort + alpha blend。Q-Render 的 transmittance-quantile 思路应该也适用，可能需要 minor adaptation。

### 5. 可以扩展的方向

- **Dynamic K**：作者试了 Learned-K 和 Stratified-K（Appendix E.1），但都要 two-pass，FPS 减半。如果能做到 single-pass adaptive K 会很 powerful。
- **RGB rendering 也能用**：Appendix E.6 显示 Q-Render 用于 RGB 只掉 0.几 PSNR，但 FPS 提升。对 4DGS / dynamic GS 这种 Gaussian 数量爆炸的场景可能很有用。
- **和 3DGS 原生 rasterizer 集成**：目前 Q-Render 是在 Python/CUDA 层面做的，如果能 merge 进 3DGS 的 tile-based rasterizer kernel，效率还能再提。

### 6. 和 ViT token pruning 的类比

Q-Render 让我想到 ViT 里的 token pruning / merging（如 [ToMe](https://github.com/facebookresearch/ToMe)）：都是"不是所有 element 都重要，只保留有贡献的"。ViT 按 attention score prunning，Q-Render 按 transmittance drop pruning，思路异曲同工。

### 7. Limitations 的诚实度

作者很诚实地承认了三个 limitation：
1. 固定 K 不够 adaptive
2. 依赖 per-scene optimized 3D-GS（虽然有 DepthSplat / WorldMirror 等generalizable GS 出现）
3. 依赖 voxel-based backbone（point-based 不行）

这种诚实度在 paper 里挺少见的，加分。

---

## 总结

Q-Render 的 beauty 在于：它把 volume rendering 从一个"沿 ray 空间积分"的问题，通过 $u = T(t)$ 换元变成"沿 transmittance 积分"的问题，然后做最朴素的 Right Riemann Sum。没有花哨的 learned sampling，没有 codebook compression，就是一个干净的数学变换 + 算法实现。

对于 high-dim feature rendering，这个变换的价值巨大：把 $\mathcal{O}(NC)$ 降到 $\mathcal{O}(N+KC)$，当 $C=512, K=40, N=100$ 时，从 51200 次乘法降到 21200 次（其中 20000 次是标量 transmittance 计算，只有 40×512=20480 次是向量乘法），实际 43.7× 加速。

Paper 链接（推测，基于作者和标题）：
- arXiv: 搜索 "Quantile Rendering Jeong Sun Wang Cho Choe NVIDIA POSTECH"
- 代码：作者说会公开
- 相关：[3D-GS original paper](https://repo.samgraph.com/3dgs), [SplatFormer](https://arxiv.org/abs/2411.06390), [OpenGaussian](https://arxiv.org/abs/2406.02058), [Dr.Splat](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_Dr._Splat_Directly_Referring_3D_Gaussian_Splatting_via_Direct_Language_CVPR_2025_paper.pdf)

---

# Quantile Rendering: 给 3D Gaussian Splatting 高维特征渲染的"采样重构"

你好 Andrej！这篇 paper 的核心 idea 其实非常 elegant，我读完之后最大的 intuition 是：**作者把 volume rendering 重新理解为 transmittance domain 上的 Riemann sum，然后做了一个非常自然的"换元 + 等距划分"**，把原来在 spatial domain 上密集采样的 O(NC) 操作，变成了在 transmittance domain 上稀疏采样的 O(N+KC) 操作。这个变换之所以能成立，本质上是因为 3D-GS 的 transmittance $T$ 沿着 ray 是 **单调递减** 的，所以可以做一个干净的 substitution $u = T(t)$。

下面我一层一层 build 你的 intuition。

---

## 1. 为什么 high-dim feature rendering 是个 problem

3D-GS 原始的 volume rendering 公式（paper Eq.1）：

$$\tilde{\mathbf{C}}[\vec{p}] = \sum_{i \in S_r} w_i \mathbf{c}_i, \quad w_i = T_i \alpha'_i, \quad T_i = \prod_{j \in S_{r,i-1}} (1 - \alpha'_j)$$

变量含义：
- $\vec{p}$: 一条 camera ray
- $S_r$: 沿 ray 按深度排序后的 Gaussian 索引序列
- $\alpha'_i = \alpha_i \cdot \mathcal{G}(\mu_i, \Sigma_i)(u)$: 第 $i$ 个 Gaussian 在 pixel $u$ 处的 effective opacity（opacity × 2D projected Gaussian 值）
- $T_i$: 到第 $i$ 个 Gaussian 之前的累积 transmittance（剩余"光"的比例）
- $w_i$: 第 $i$ 个 Gaussian 对最终 pixel 的贡献权重
- $\mathbf{c}_i$: 第 $i$ 个 Gaussian 的 SH color

问题：当你渲染 RGB（3-D）时，这个 $\sum w_i \mathbf{c}_i$ 只是把 3 维向量做加权平均，cost 不大。但是当你想渲染 512-D CLIP feature 时，每个 Gaussian 要存一个 $\mathbf{f}_i
