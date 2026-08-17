---
source_pdf: CITYGAUSSIANV2.pdf
paper_sha256: 110c304fb1c05d35b8ff956ed6cf19a83f5cbfb0e303a5401ceb9cafb645e2f7
processed_at: '2026-08-03T15:38:05-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 CityGaussianV2

## 一句话版本

这 paper 就是把 2DGS 这个在小物件上效果很好的 surface reconstruction 方法，通过几个工程 trick 硬塞进 CityGaussian 的大规模并行训练 pipeline 里，同时砍掉了一半显存和 10 倍存储，还顺手搞了个靠谱的几何评估 benchmark。

## 先 build 一个 intuition

你想象一下你在用乐高积木拼一个城市模型。3DGS 是一堆小方块乱堆，颜色对了但形状乱。2DGS 把每个小方块压成小圆盘（surfel），贴在表面上，所以几何更准。问题是城市太大了，一张 GPU 卡装不下，得把城市切成一块一块分别拼，最后再拼起来（这就是 CityGaussian 的思路）。

但 2DGS 直接这么干会出三个 bug：

1. **拼得慢**：早期训练出来总是糊的，看不出细节
2. **积木爆炸**：某些被压扁的圆盘在远视角下投影小于一个像素，反而被反复 clone，数量指数级增长直到 OOM
3. **流程太重**：原 CityGaussian 训练完还要做 pruning + distillation，又花 4 万 iter，太慢

这 paper 就是挨个解决这三个 bug。

## Bug 1：为什么 2DGS 早期训练总是糊的？

### 现象

Fig.10 里能看到，训练 7000 iter 的时候，3DGS 已经有点样子了，2DGS 还是一团糊。这个很反直觉——2DGS 几何更准，凭什么收敛更慢？

### 诊断

3DGS 的 loss 是 $\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{SSIM}$，densification（决定哪些地方要 clone/split 出新 Gaussian）用的是 total loss 对 position 的 gradient。

关键 insight：**L1 loss 对 blur 不敏感，SSIM 对 blur 极度敏感**。这个事 [Wang 2004 SSIM 原论文](https://ieeexplore.ieee.org/document/1288389) 早就讲过。你把一张图 blur 一下，L1 误差变化不大，SSIM 直接跳水。

2DGS 因为把 3D 压成 2D，早期容易出现大面积、低 opacity 的"摊大饼"surfels。这种摊大饼在 L1 loss 下误差小，所以 total loss gradient 也不大，optimizer 觉得"差不多得了"，不去 densify 这些区域。结果就是这些区域一直糊着，convergence 卡住。

### 解法：DGD (Decomposed-Gradient Densification)

公式 2 长这样：

$$\nabla_{densify} = \max\left(\omega \times \frac{|\nabla \mathcal{L}|_{avg}}{|\nabla \mathcal{L}_{SSIM}|_{avg}}, 1\right) \times \nabla \mathcal{L}_{SSIM}$$

人话翻译：
- $\nabla \mathcal{L}_{SSIM}$：只用 SSIM loss 算出来的 position gradient
- $|\nabla \mathcal{L}|_{avg}$ / $|\nabla \mathcal{L}_{SSIM}|_{avg}$：batch 里 total loss gradient 平均大小 / SSIM loss gradient 平均大小
- $\omega = 0.9$：一个稍微保守一点的系数
- $\max(\cdot, 1)$：保证 scale factor 至少是 1，别把 SSIM gradient 弄得更小

**核心 idea**：densification 这个事，干脆只听 SSIM 的。SSIM 觉得哪里结构不对，就在哪里加 Gaussians。L1 觉得哪里颜色不对，那是 parameter optimization 的事，跟 capacity allocation 没关系。

为什么要乘那个 scaling factor？因为原版 3DGS 的 densification threshold（默认 0.0002）是针对 total loss gradient 调的。SSIM gradient 的 magnitude 跟 total loss 不在一个量级，直接换 SSIM 的话 threshold 就失效了。所以用一个 ratio 把 SSIM gradient 自动 rescale 到 total loss 的量级，原 threshold 还能继续用。

### 实验验证

Tab.6 是最干净的 ablation：

| Densification gradient 来源 | PSNR | SSIM | F1 |
|---|---|---|---|
| SSIM + RGB + Normal | 21.18 | 0.636 | 0.401 |
| 只用 SSIM | **22.24** | **0.674** | **0.419** |
| 只用 SSIM + Depth | 22.21 | 0.674 | 0.429 |

加入 RGB gradient **反而下降 1 个 PSNR**。这就证明了 L1 gradient 在 densification 里是有害的，会 clone/split 那些 photometric error 高但结构已经 OK 的地方，浪费 Gaussian 容量。

### Intuition 总结

这件事的本质是 **decoupling**：loss function 是为了 parameter 优化（让颜色对），densification 是为了 capacity 分配（让结构对）。两者用同一个 metric 其实是 historical accident。DGD 把这两个事情解耦，densification 专心搞结构，parameter 优化专心搞颜色。

## Bug 2：为什么 Gaussian 数量会爆炸？

### 现象

Fig.3 右图：naive 2DGS 在 parallel tuning 阶段，Gaussian 数量从 ~10M 一路指数涨到 OOM。CityGaussianV2 加了 Elongation Filter 后数量稳得很。

### 诊断

2DGS 的 surfel 有两个 tangent scaling $s_{n,u}$ 和 $s_{n,v}$。如果一个 surfel 被压得很扁（一个方向大、一个方向小），从远处看投影面积就小于一个 pixel。

2DGS 用了 [Mip-Splatting](https://arxiv.org/abs/2312.00257) 的 anti-aliasing trick：投影小于一个 pixel 时，用 fixed value 替换 covariance，避免 aliasing。副作用是这些 surfel 的 covariance 被"冻结"了，optimizer 没法通过 gradient 调整它们的 scale 和 rotation。

但它们还参与 alpha blending，opacity 高的时候位置稍微动一下，像素颜色就大变 → position gradient 巨大 → 触发 densification → split 出来的子 surfel 继承同样 degenerated shape → 继续 freeze → 继续 high gradient → 继续 split...

在 single GPU 全数据训练时这事还好，因为每个 surfel 平均被很多 view 看到，gradient 是平均的。但 block-wise parallel tuning 里每个 block 只分到一部分 view，distant views 频繁出现 → degenerated points 反复被刺激 → 爆炸。

Fig.3 左图就是这个观察的可视化：高 gradient 点集 和 extreme elongation 点集 **高度重叠**。

### 解法：Elongation Filter

定义 elongation rate：

$$\eta_n = \frac{\min(s_{n,u}, s_{n,v})}{\max(s_{n,u}, s_{n,v})}$$

人话：两个 tangent scaling 的比值，越小说明越扁。$\eta_n = 1$ 是完美圆盘，$\eta_n = 0$ 是退化成一条线。

**做法**：densification 之前，凡是 $\eta_n$ 低于某个 threshold 的 surfel，直接 skip，不参与 clone/split。

就这么简单。一行 if 语句搞定。

### 为什么这么 simple 的 trick 能 work？

因为它精准命中了 failure mode 的因果链：
1. Elongated surfel → 投影 < 1 pixel → covariance 被 freeze
2. Freeze 后无法通过 gradient 修正 shape
3. 但 opacity 高 → 微小位移 → 像素大变 → 高 gradient
4. 高 gradient → 触发 split → 子 surfel 继承 degenerate shape
5. 子 surfel 重复 1-4 → 指数爆炸

在步骤 4 这里切断因果链就行。Filter 不修改 rasterizer，不修改 loss，不修改 optimizer，只在 densification 这一步堵漏洞。非常 surgical 的工程思路。

### 实验验证

Tab.2 第二行：+ Elongation Filter 后 PSNR 21.12 → 21.18，F1 0.410 → 0.411。**pretraining 阶段几乎没影响**。

为什么不影响？因为 pretraining 时全数据训练，degenerated points 不会被 distant views 反复刺激，filter 主要在 parallel tuning 阶段发挥作用。这正是 Fig.3 右图显示的：pretraining 阶段数量稳定，tuning 阶段才开始爆炸。

## Bug 3：Pipeline 太重怎么砍？

### 原版 CityGaussian 的流程

看 Fig.4 上半部分虚线：
1. Pretrain coarse model on full data
2. Block partition + parallel tuning
3. **Post-pruning**（30K iter，用 [LightGaussian](https://arxiv.org/abs/2311.17245) 的策略）
4. **Distillation**（10K iter，SH degree 3→2）
5. Vectree quantization

步骤 3+4 多花 4 万 iter，巨慢。

### CityGaussianV2 的优化

#### 优化 1：From scratch 用 SH degree 2

3DGS 默认 SH degree 3，每个 Gaussian 存 48 维 SH feature。SH degree 2 只要 27 维。

原 CityGaussian 用 degree 3 训完，再用 distillation 降到 degree 2。CityGaussianV2 直接从头用 degree 2，省掉 distillation 整个步骤。

代价：rendering quality 微降（Tab.2 对比 +SH Degree=2 那行：PSNR 23.57 → 23.49，下降 0.08）。可接受。

收益：memory 14.2GB（省 50%+），storage 1.29GB，速度 +4.2 FPS。

#### 优化 2：Trimming 集成到 tuning 阶段

原版 pruning 是训练完单独跑 30K iter。CityGaussianV2 借鉴 [TrimGS](https://arxiv.org/abs/2406.07499) 的 contribution 概念，在每个 block 的 tuning 阶段定期 trim。

公式 3 定义 single-view contribution：

$$\mathbf{C}_{n,k} = \frac{1}{|\mathbb{P}_k|} \sum_{p \in \mathbb{P}_k} (\alpha_n)^{\gamma} \left(\prod_{j=1}^{n(p)-1}(1-\alpha_j)\right)^{(1-\gamma)}$$

人话翻译：
- $\mathbb{P}_k$：第 $n$ 个 Gaussian 在第 $k$ 个 view 的 2D 投影区域
- $\alpha_n$：该 Gaussian 的 opacity
- $n(p)$：在穿过 pixel $p$ 的 ray 上，该 Gaussian 的 depth 排序位置
- $\prod_{j=1}^{n(p)-1}(1-\alpha_j)$：前面所有 Gaussian 的 transmittance，前置遮挡多则这个值小
- $\gamma = 0.5$：balance opacity 直接贡献 和 被遮挡程度的权重

公式 4 是 multi-view average：把一个 Gaussian 在 block 内所有 view 上的 contribution 取平均。

**关键差异**：原 TrimGS 用 fixed threshold，CityGaussianV2 用 **percentile-based threshold**（比如 pruning ratio 0.025 或 0.1，把 contribution 最低的百分之几干掉）。percentile 的好处是自适应不同 scene 的分布。

收益：Tab.2 对比 +Trim vs +Prune：
- Trim (percentile): PSNR 23.57, F1 0.477, #GS 8.07M
- Prune (LightGaussian): PSNR 23.46, F1 0.472, #GS 10.3M

Trim 又快又好。LightGaussian 的 importance score 在 large-scale 下不如 percentile 鲁棒。

#### 优化 3：Vectree Quantization on 2DGS

codebook size 8192，quantization ratio 0.4。least important Gaussians 的 SH 用 vector quantization 压，其他 attribute 用 float16。

Tab.2 +VQ 那行：storage 从 1.29GB → **0.44GB**，10x 压缩。memory 不变（VQ 是后处理）。

### 整体效果对比

Tab.2 最底下两行 vs CityGaussian：

| 方法 | Time | Memory | Storage | F1 |
|---|---|---|---|---|
| CityGaussian | 254 min | 31.5 GB | 0.60 GB | 0.450 |
| Ours-s (small) | 181 min | 14.2 GB | 0.44 GB | 0.465 |
| Ours-t (tiny, 7k pretrain) | 115 min | 11.5 GB | 0.29 GB | 0.461 |

Ours-s：时间省 29%，memory 省 55%，storage 省 27%，F1 还高 0.015。
Ours-t：时间省 55%，memory 省 64%，storage 省 52%，F1 还高 0.011。

小模型 Ours-t 0.29GB 能塞手机或 VR headset 里。

## 顺手搞的 Geometry Benchmark

### 问题

[GauU-Scene](https://arxiv.org/abs/2410.18942) 是第一个 large-scale geometry benchmark，但它的 evaluation protocol 有两个坑：

**坑 1：boundary effect**。场景边界区域观测不足，geometry 估计天然不准。把边界也算进 F1 score → 误差大、不稳定，error bar 长度 0.1。

**坑 2：不公平的 surface point 提取**。NeRF-based 方法从 depth map 提点，3DGS 从 Gaussian mean 提点。两种提取方式根本不可比。

### 解法：TnT-style Protocol

借鉴 [Tanks and Temples](https://wwwtanksandtemples.org/) 的思路：

1. **统一提取方式**：所有方法先 extract mesh，再从 mesh surface 均匀采样点。注意是均匀采样，不是 vertex + face center（TnT 原版做法会低估大三角形误差）。

2. **Visibility-based crop volume estimation**：
   - 把 GT point cloud 初始化成 3DGS field
   - 用 GS rasterizer 跑所有 training view，统计每个点的可见频率 $\tau_j$
   - $\tau_j$ 低于 threshold 的点剔除
   - 剩下点用 alpha shape 算 ground plane bound + min/max height
   - 整个过程 < 1 minute

3. **Downsampling**：voxel size 0.35m，距离 threshold 0.3m-0.6m。

### 效果

Error bar 从 0.1 降到 0.003，30 倍稳定性提升。这才让 SOTA 比较有了 statistical meaning。

## 整体 Pipeline 图解

Fig.2 画出了三个 trick 的协同关系：

```
2D Gaussians
    ↓ camera projection
Rasterizer
    ↓
Rendered image + depth
    ↓
Loss: RGB + SSIM + Normal + Depth
    ↓ ↓ ↓ ↓
    │ │ │ └→ Depth Anything V2 监督（exponential decay weight）
    │ │ └──→ Normal loss（7k iter 后激活，weight 降到 1/4）
    │ └────→ SSIM loss → DGD → densification gradient（rescaled）
    └──────→ RGB loss → parameter optimization（不参与 densification）
    ↓
Densification
    ↓
Elongation Filter（skip η_n < threshold 的 surfels）
    ↓
Clone / Split
```

三者形成 feedback loop：
- **Depth prior** 提供早期 global structure guidance
- **DGD** 让 blurry region 被 identify 并 densify
- **Elongation Filter** 防止 densification 被 degenerated points 绑架

## 实验数据再过一遍

### Tab.1 主表

GauU-Scene 上：
- NeuS / Neuralangelo：NaN 或 FAIL（NeRF 在大场景直接挂）
- SuGaR：F1 0.377（mesh 太大 memory 爆炸）
- GOF：F1 0.374（生成 shell-like mesh 难处理）
- 2DGS：F1 0.491（单 GPU 能跑但没 scale up）
- CityGS：F1 0.407（rendering 强但 geometry 弱）
- **Ours：F1 0.501**（geometry SOTA）

MatrixCity-Aerial：
- SuGaR / GOF：OOM / FAIL
- 2DGS：F1 0.270
- CityGS：F1 0.462
- **Ours：F1 0.556**（比 CityGS 高 0.094，比 2DGS 高 2 倍）

MatrixCity-Street：
- CityGS：F1 0.401
- **Ours：F1 0.503**

### Tab.2 消融的 takeaways

逐行讲已经在上文做过了，这里再强调三个 non-obvious 的点：

1. **DGD 是 single most impactful trick**：单独加 DGD 提升 1.12 PSNR + 0.019 F1。其他 trick 都是 scale up / efficiency，DGD 是 quality 的核心。

2. **7k pretrain 失败实验的反直觉启示**：用 7k iter 的 2DGS 做 partition → PSNR 22.68, F1 0.456（远低于 ours-t 的 23.17/0.461）。这说明 block partition 的 quality 严重依赖 pretrain model 的 convergence。DGD 让 7k iter 时 model 已经 well-converged，partition 才有意义。DGD 对整个 pipeline 有 multiplier effect。

3. **SH degree 2 vs 3 几乎无损**：Tab.2 +SH Degree=2 那行 PSNR 23.57 → 23.49，F1 0.477 → 0.474。drop 微乎其微。所以 distillation 这个步骤在 large-scale 场景下完全是浪费时间——degree 3 的 SH 在大场景下没用上 view-dependent 的复杂度。

## 一些个人联想

### 跟其他工作的关系

- **DGD 跟 [AbsGS](https://arxiv.org/abs/2404.06109) 的思路相反**：AbsGS 觉得 3DGS 的 densification threshold 用绝对值不合理，改用 relative threshold。DGD 觉得 densification 用 total loss gradient 不合理，改用 SSIM gradient。两者都是对原版 3DGS densification 机制的反思，但切入角度不同。

- **Elongation Filter 跟 [TrimGS](https://arxiv.org/abs/2406.07499) 互补**：TrimGS 用 contribution 来 prune，Elongation Filter 用 shape 来 prevent densification。一个治存量，一个治增量。两者结合才能在 large-scale 下稳定。

- **Pipeline 优化跟 [Compact3D](https://arxiv.org/abs/2311.18159) 类似**：都是从 from-scratch 用低 SH degree + VQ 来省 storage。区别在 CityGaussianV2 还做了并行训练层面的优化。

### 如果你要自己实现

我会这么干：
1. **DGD 直接抄**：公式 2 实现就 10 行代码，threshold 都不用改。这个 trick 在任何 GS 变体上应该都 work，因为本质是 decouple loss 和 densification。
2. **Elongation Filter 也直接抄**：就一行 if 语句。threshold 可以从 0.1 开始调。
3. **SH degree 2 from scratch**：除非你的场景有非常强的 view-dependent 效果（水面、玻璃），否则 degree 2 够用，degree 3 是浪费。
4. **Trimming 用 percentile**：固定 threshold 不鲁棒，percentile 自适应不同 scene。
5. **Depth supervision 用 [Depth-Anything V2](https://arxiv.org/abs/2406.09414)**：别自己训 monocular depth model，直接用人家预训练好的，记得 weight exponential decay。

### 这 paper 没解决的问题

1. **TSDF mesh extraction**：thin structures 还是会丢。GOF 的 marching tetrahedra 可能更好但贵。这块还有空间。

2. **Rendering speed**：2DGS rasterizer 比 3DGS 慢，所以 Ours-s 才 34.5 FPS，CityGS 能到 58.8 FPS。要上 [FlashGS](https://arxiv.org/abs/2408.07967) 这种 rasterizer 优化或者 [HierarchicalGS](https://arxiv.org/abs/2406.12080) / [OctreeGS](https://arxiv.org/abs/2403.17898) 的 LoD 才行。

3. **Dynamic scene**：整个 paper 都是 static scene。大规模动态场景（比如城市交通）是另一个量级的问题。

4. **Block partition 的自动化**：现在还是手动切 block，怎么根据 scene structure 自动 partition 是个开放问题。

## 相关 Links 汇总

主要 paper：
- [CityGaussianV2 project page](https://dekuliutesla.github.io/CityGaussianV2/)
- [3DGS (Kerbl et al., 2023)](https://repo.sam.lum.is/3dgaussian-splatting/)
- [2DGS (Huang et al., 2024)](https://arxiv.org/abs/2403.17888)
- [CityGaussian V1 (Liu et al., 2024)](https://arxiv.org/abs/2404.01133)
- [TrimGS (Fan et al., 2024)](https://arxiv.org/abs/2406.07499)
- [GOF (Yu et al., 2024)](https://arxiv.org/abs/2404.10772)
- [SuGaR (Guedon & Lepetit, 2024)](https://arxiv.org/abs/2311.12775)
- [Depth-Anything V2](https://arxiv.org/abs/2406.09414)
- [LightGaussian](https://arxiv.org/abs/2311.17245)
- [Mip-Splatting](https://arxiv.org/abs/2312.00257)

Benchmark & Dataset：
- [GauU-Scene](https://arxiv.org/abs/2410.18942)
- [Tanks and Temples](https://wwwtanksandtemples.org/)
- [MatrixCity](https://github.com/city-super/MatriCity)

大规模 GS：
- [VastGaussian](https://arxiv.org/abs/2402.04332)
- [HierarchicalGS](https://arxiv.org/abs/2406.12080)
- [OctreeGS](https://arxiv.org/abs/2403.17898)
- [DoGaussian](https://arxiv.org/abs/2405.13943)
- [GrendelGS](https://arxiv.org/abs/2406.18533)
- [FlashGS](https://arxiv.org/abs/2408.07967)

辅助：
- [SSIM (Wang et al., 2004)](https://ieeexplore.ieee.org/document/1288389)
- [AbsGS (Bulo et al., 2024)](https://arxiv.org/abs/2404.06109)
- [Compact3D](https://arxiv.org/abs/2311.18159)
- [NeuS](https://arxiv.org/abs/2106.10689)
- [Neuralangelo](https://research.nvidia.com/labs/toronto-ai/neuralangelo/)
- [Depth-supervised NeRF](https://arxiv.org/abs/2107.02791)

## 最后一句

这 paper 的优雅之处在于它没发明新理论，全是对 2DGS 在大场景下 failure mode 的精准诊断 + 对症下药。三个 trick 各自简单，组合起来把 2DGS 从单物件方法推到了城市级几何重建 SOTA，还顺手把 memory 和 storage 砍了 50%+。对要 build production GS pipeline 的人，这 paper 几乎每个 trick 都可以直接抄。

---

# CityGaussianV2 深度技术解析

## 一、Motivation 与 Problem Setting

大规模场景重建长期面临 **scalability** 与 **geometric accuracy** 之间的 trade-off。3DGS ([Kerbl et al., 2023](https://dl.acm.org/doi/10.1145/3588432.3588446)) 通过 anisotropic ellipsoids 实现了 real-time rendering，但 unstructured 性质导致 surface estimation 存在 ambiguity。2DGS ([Huang et al., 2024](https://arxiv.org/abs/2403.17888)) 将 3D ellipsoid collapse 成 oriented disk (surfel)，将 covariance 从 3×3 降为 2D tangent space，但作者发现直接 scale up 2DGS 会遇到 **三个核心问题**：

1. **Convergence slowdown**：早期训练阶段 2DGS 比 3DGS 更容易产生 blurry reconstruction（见原文 Fig.10），原因是 L1 RGB loss 对 blurriness 不敏感，而 densification 的 position gradient 主要来自 L1 loss。
2. **Gaussian count explosion**：2DGS degeneration 后某些 elongated surfels 在远视角下投影面积小于一个 pixel，anti-aliased low-pass filter 会用 fixed value 替换其 covariance，导致这些点无法获得有效 gradient 来调整 scale/rotation。在 block-wise parallel tuning 中，distant views 频繁出现，degenerated points 累积高 gradient 触发指数级 densification，最终 OOM。
3. **Pipeline overhead**：原版 CityGaussian 的 post-pruning（30K iter）+ distillation（10K iter）耗时严重。

## 二、Methodology 详解

### 2.1 Preliminary：3DGS rendering equation

3DGS 用一组 ellipsoid $\mathbf{G}_N = \{\bar{G}_n | n=1,...,N\}$ 表示场景，每个 Gaussian 的属性包括 center $\boldsymbol{\mu}_n \in \mathbb{R}^{3\times1}$、covariance $\boldsymbol{\Sigma}_n \in \mathbb{R}^{3\times3}$、opacity $\sigma_n \in [0,1]$、SH features $f_n \in \mathbb{R}^{3\times16}$。Covariance 分解为 rotation $\mathbf{R}_n$ 与 scaling $\mathbf{S}_n$：

$$\boldsymbol{\Sigma}_n = \mathbf{R}_n \mathbf{S}_n \mathbf{S}_n^T \mathbf{R}_n^T$$

对 pixel $p$，color $c_p$ 由 alpha blending 得到（**公式1**）：

$$c_p = \sum_{i \in \gamma(p)} c_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$$

$$\alpha_i = \sigma_i \cdot \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1}(\mathbf{x}-\boldsymbol{\mu}_i)\right)$$

变量解释：
- $\gamma(p)$：穿过 pixel $p$ 的 ray 上覆盖的所有 Gaussians
- $c_i$：第 $i$ 个 Gaussian 的 view-dependent color（由 SH 计算）
- $\alpha_i$：第 $i$ 个 Gaussian 在 query point $\mathbf{x}$ 处的 opacity contribution
- $\sigma_i$：learned opacity
- 指数项：标准 Gaussian probability density，$\boldsymbol{\Sigma}_i^{-1}$ 是 inverse covariance
- $\prod_{j=1}^{i-1}(1-\alpha_j)$：front-to-back compositing 的 transmittance

3DGS 的 densification 由 position gradient 触发：$\nabla_{densify} = \partial \mathcal{L} / \partial \boldsymbol{\mu}_n$，超过 threshold 的 Gaussians 被 clone 或 split。

### 2.2 Depth Supervision via Depth-Anything V2

作者用 [Depth-Anything-V2](https://arxiv.org/abs/2406.09414) 估计 monocular inverse depth $\hat{D}_k$，再 align 到 dataset scale 得到 $D_k$。Loss 为：

$$\mathcal{L}_{Depth} = |\hat{D}_k - D_k|$$

关键 trick：loss weight $\alpha$ 随训练 **exponentially decay**（从 0.5 → 0.0025），因为 mono depth 在 fine detail 上有 noise，早期提供 global structure guidance，后期让 photometric loss 接管 fine-tuning。这与 [Depth-supervised NeRF](https://arxiv.org/abs/2107.02791) 的思路类似，但应用到 GS 上的 novelty 在于与 DGD 的协同作用。

### 2.3 Elongation Filter：防止 Gaussian Count Explosion

这是 paper 中最 intuitive 的 trick。定义 elongation rate：

$$\eta_n = \frac{\min(s_{n,u}, s_{n,v})}{\max(s_{n,u}, s_{n,v})}$$

其中 $s_{n,u}, s_{n,v}$ 是 2D surfel 的两个 tangent scaling。当 $\eta_n < \text{threshold}$（极度 elongated，像细沙粒），在 densification 前 **直接 skip** 这些 Gaussians。

**Intuition 构建**：Fig.3 左图揭示了一个非常重要的几何观察——高 gradient 点集合 与 extreme elongation 点集合 **高度重合**。原因：
- Elongated surfel 远距离投影 < 1 pixel
- Anti-aliasing low-pass filter 用 fixed covariance 替换
- 这些 "frozen" Gaussians 无法通过 gradient 修正 shape
- 但它们仍参与 alpha blending，opacity 高时移动一点点 → 像素颜色巨变 → gradient 极大
- 在 block-wise tuning 中 distant views 反复出现 → gradient 累积 → 触发 split
- Split 出来的子点继承相同 degenerated shape → 无限繁殖

这个 filter 让 Gaussian count evolution 从指数增长变成类似 CityGaussian 的稳定曲线（Fig.3 右图）。

### 2.4 Decomposed-Gradient-based Densification (DGD)

**核心 idea**：naive 2DGS 在 early stage 容易 blurry（Fig.10），原因是 L1 RGB loss 对 blurriness 几乎不敏感（[Wang et al., 2004](https://ieeexplore.ieee.org/document/1288389) 早就指出 SSIM 才捕获 structural degradation）。而原版 3DGS 的 densification gradient 来自 total loss $\nabla \mathcal{L}$，L1 在 total loss 中占主导 → blurry surfels 不会被 densify → 无法 converge。

DGD 公式（**公式2**）：

$$\nabla_{densify} = \max\left(\omega \times \frac{|\nabla \mathcal{L}|_{avg}}{|\nabla \mathcal{L}_{SSIM}|_{avg}}, 1\right) \times \nabla \mathcal{L}_{SSIM}$$

变量解释：
- $\nabla \mathcal{L}_{SSIM}$：SSIM loss 对 Gaussian position 的 gradient
- $|\nabla \mathcal{L}|_{avg}$、$|\nabla \mathcal{L}_{SSIM}|_{avg}$：batch 内平均 gradient norm
- $\omega = 0.9$：scaling constant
- $\max(\cdot, 1)$：保证 scaled SSIM gradient magnitude 不低于原始 SSIM gradient

**Intuition**：
1. SSIM 对 blurriness 敏感 → blurry surfels 在 SSIM gradient 上信号强
2. Scaling factor $\omega \cdot |\nabla \mathcal{L}|_{avg} / |\nabla \mathcal{L}_{SSIM}|_{avg}$ 把 SSIM gradient magnitude 拉到与 total loss 相同量级，避免原 threshold 设定失效
3. 只用 SSIM gradient 来 densify，相当于告诉 optimizer："structural inconsistency 的地方才需要更多 Gaussians"

Tab.6 的 ablation 极具说服力：
- 用 SSIM + RGB + Normal gradient：PSNR 21.18, SSIM 0.636
- 只用 SSIM gradient：PSNR 22.24, SSIM 0.674
- 加入 RGB gradient 反而 **下降** 1.06 PSNR

这说明 L1 gradient 在 densification 中是 actively harmful 的——它会 clone/split 那些 photometric error 高但 structurally 已经 OK 的区域，浪费 capacity。

### 2.5 Parallel Training Pipeline 优化

原 CityGaussian pipeline 的问题：
- Pre-train coarse model on full data
- Block partition + parallel tuning
- Post-pruning (30K iter, LightGaussian strategy)
- Distillation (10K iter, SH degree 3→2)
- Vectree quantization

CityGaussianV2 的改动（见 Fig.4）：
1. **From scratch 用 SH degree 2**：省去 distillation，SH feature dim 从 48 降到 27，整个 pipeline memory/storage 直接减少 ~44%。
2. **Contribution-based pruning during tuning**：参考 [TrimGS](https://arxiv.org/abs/2406.07499)，定义 single-view contribution（**公式3**）：

$$\mathbf{C}_{n,k} = \frac{1}{|\mathbb{P}_k|} \sum_{p \in \mathbb{P}_k} (\alpha_n)^{\gamma} \left(\prod_{j=1}^{n(p)-1}(1-\alpha_j)\right)^{(1-\gamma)}$$

变量：
- $\mathbb{P}_k$：第 $n$ 个 Gaussian 在第 $k$ 个 view 下的 2D projection region
- $n(p)$：第 $n$ 个 Gaussian 在穿过 $p$ 的 ray 上的 depth-sorted order
- $\gamma = 0.5$：weight balancing opacity 与 transmittance contribution
- $(\alpha_n)^{\gamma}$：直接 opacity 贡献
- $\left(\prod(1-\alpha_j)\right)^{1-\gamma}$：occlusion 因子，前置 Gaussian 多则该 Gaussian 贡献小

Multi-view average（**公式4**）：

$$\mathbf{C}_n = \frac{1}{|\mathbb{V}_m|} \sum_{k \in \mathbb{V}_m} \mathbf{C}_{n,k}$$

$\mathbb{V}_m$：分配给第 $m$ 个 block 的 image set。

**关键差异**：原 TrimGS 用 fixed threshold，CityGaussianV2 用 **percentile-based threshold**（pruning ratio = 0.025 或 0.1），更鲁棒地适应不同 scene 的 contribution 分布。

3. **Vectree quantization**：codebook size 8192，quantization ratio 0.4。least important Gaussians 的 SH 用 vector quantization，其余 attributes 用 float16。

### 2.6 Geometry Evaluation Protocol

这是 paper 中一个被低估的 contribution。现有 [GauU-Scene](https://arxiv.org/abs/2410.18942) 的 evaluation protocol 有两个问题：
1. 边界区域 observation 不足 → F1 score 不稳定，error bar 长度可达 0.1
2. NeRF-based 方法从 depth map 提取点，3DGS 从 Gaussian mean 提取 → 不公平

CityGaussianV2 借鉴 [Tanks and Temples](https://wwwtanksandtemples.org/) protocol：
- 对所有方法先 extract mesh，再从 surface **均匀采样**（而非 vertex + face center，避免大三角形 bias）
- **Visibility-based crop volume estimation**：把 GT point cloud 当成 Gaussians，用 GS rasterizer 跑 visibility count，剔除观测频率低的点，再用 alpha shape 计算 ground plane bound + min/max height
- 整个过程 < 1 minute，error bar 从 0.1 降到 0.003

## 三、实验数据深度解读

### 3.1 与 SOTA 比较 (Tab. 1)

关键观察：
- **NeRF-based methods (NeuS, Neuralangelo)** 在 large-scale 全部失败（NaN 或 FAIL），原因是 MLP 在 sparse supervision 下 poor convergence，且 10+ hours 训练
- **SuGaR** 在 MatrixCity-Aerial OOM（mesh 在大场景下 memory 爆炸）
- **GOF** 生成近地 shell-like mesh 难以移除（Fig.8），且 Russian Building 上 OOM
- **2DGS** 单 GPU 能跑但几何质量中等（GauU-Scene F1=0.491）
- **CityGS** rendering 强（PSNR 24.75）但几何弱（F1=0.407）
- **CityGaussianV2** 几何 SOTA（F1=0.501, 0.556, 0.503），rendering 与 CityGS on par

特别值得注意的是 MatrixCity-Aerial：CityGaussianV2 F1=0.556 vs CityGS 0.462 vs 2DGS 0.270。这个 2x 的提升来自三个因素叠加：DGD 加速收敛 + Elongation Filter 防止爆炸 + Depth supervision。

### 3.2 消融研究 (Tab. 2)

最 informative 的实验。逐行分析：

**Pretraining 阶段**（baseline 2DGS, PSNR 21.12, F1 0.410）：
- + Elongation Filter: 21.18 / 0.411 — 几乎无影响（验证 pretraining 阶段 degeneration 不严重）
- + DGD: **22.24 / 0.429** — PSNR +1.12, F1 +0.019，单 trick 巨大提升
- + Depth Regression: 22.22 / **0.438** — rendering 几乎不变，几何 +0.009（depth prior 主要帮 geometry）

**Parallel tuning 阶段**：
- + Parallel Tuning: 23.50 / 0.471，#GS 从 9.67M 飙到 19.3M，memory 31.5GB（爆炸！）
- + Trim (Ours-b): 23.57 / 0.477，#GS 8.07M，storage 1.9GB（trim 拯救了爆炸）
- + Prune (LightGaussian): 23.46 / 0.472 — 比 trim 略差，验证 percentile-based 优于 fixed threshold
- + SH Degree=2: 23.49 / 0.474，storage 1.29GB，memory 14.2GB（distillation 不需要）
- + VQ (Ours-s): 23.46 / 0.465，**storage 0.44GB**（10x 压缩），memory 14.2GB
- + 7k pretrain (Ours-t): 23.17 / 0.461，时间 115 min（vs CityGS 254 min，**省 55%**），storage 0.29GB

**vs CityGaussian**：CityGS 254 min / 0.6GB / F1=0.450 vs Ours-s 181 min / 0.44GB / F1=0.465。**全面碾压**。

### 3.3 为什么 DGD 这么 effective？

回到公式 2 的 intuition：2DGS 的 surfel 表示中，blurry artifact 表现为大面积、低 opacity 的 surfel 摊平覆盖一块区域。L1 loss 对此**不敏感**（颜色平均误差小），SSIM loss 则**极度敏感**（structural similarity 被 blur 严重破坏）。

原 3DGS 的 densification threshold（典型 $\tau = 0.0002$）针对 total loss gradient 调优，包含 L1 主导项。直接换用 SSIM gradient 时 magnitude 不匹配，threshold 失效。DGD 的 scaling factor $\omega \cdot |\nabla \mathcal{L}|_{avg}/|\nabla \mathcal{L}_{SSIM}|_{avg}$ 自动把这个 magnitude 校准回 total loss 量级，使原 threshold 仍然适用。

这是一个非常 elegant 的工程 insight：不修改 threshold，不修改 loss weight，只改 densification 的 gradient source + 自动 rescale。

### 3.4 7k Pretrain 失败实验的启示

Tab.2 倒数第二行：用 7k iter 的 2DGS partition → PSNR 22.68, F1 0.456（远低于 ours-t 的 23.17/0.461）。

**这告诉我们什么？** Block partition 的 quality 严重依赖 pretrain model 的 convergence。7k iter 时 2DGS 还在 blurry state（见 Fig.10），此时做 partition 会让 block 边界穿过 under-reconstructed 区域。DGD 让 7k iter 时 model 已经 well-converged，partition 才有意义。

这个 ablation 间接证明了 DGD 对整个 pipeline 的 multiplier effect。

## 四、架构图解析 (Fig. 2)

Fig.2 展示 optimization mechanism 的三个 component 如何协同：

1. **2D Gaussians** → camera projection → rasterizer → rendered image
2. **Loss calculation**：RGB loss + SSIM loss + Normal loss + Depth loss
3. **DGD**：只取 SSIM loss 的 position gradient → rescale → 用于 densification
4. **Elongation Filter**：在 densification 前过滤 $\eta_n < \tau$ 的 surfels
5. **Depth Regression**：用 Depth-Anything V2 提供的 inverse depth 监督 rendered depth

三者形成 feedback loop：
- Depth prior 提供 global structure guidance（前期强、后期弱）
- DGD 让 blurry region 被 identify 并 densify
- Elongation Filter 防止 densification 被 degenerated points 绑架

## 五、Limitations 与 Future Directions

作者在 Section D 诚实讨论了局限：
1. **TSDF mesh extraction** 的 classic tradeoff：thin structures 完整性 vs. quality。GOF 的 marching tetrahedra 可能更好但计算贵。
2. **Rendering speed** 仍落后 CityGS（31-43 FPS vs 58.8 FPS），因为 2DGS rasterizer 比 3DGS 慢，未来需要 [FlashGS](https://arxiv.org/abs/2408.07967) 风格的优化或 [HierarchicalGS](https://arxiv.org/abs/2406.12080) / [OctreeGS](https://arxiv.org/abs/2403.17898) 的 LoD。

## 六、对大规模场景重建的 Intuition 总结

这篇 paper 给我几个深层 insight：

1. **Loss 设计与 densification 的 decoupling**：3DGS 原版默认 densification 用 total loss gradient，这其实是 historical accident。Loss 是为了 parameter optimization，densification 是 capacity allocation。两者 metric 应该不同——前者要 photometric accuracy，后者要 structural coverage。DGD 把这个 decoupling 做到了极致。

2. **Degeneration 的诊断与 mitigation**：2D surfel 在远距离投影 < 1 pixel 时 anti-aliasing 把 covariance "冻结"，这是 [Mip-Splatting](https://arxiv.org/abs/2312.00257) 的 anti-aliasing 机制的副作用。CityGaussianV2 没有修改 rasterizer，而是用 Elongation Filter 在 densification 环节堵漏洞——非常实用的工程思路。

3. **Pipeline 的端到端 co-design**：From-scratch SH degree 2 + integrated trimming + VQ 的组合，让 storage 从 4.6GB 降到 0.44GB（**10x**），memory 从 31.5GB 降到 14.2GB（**2.2x**），time 从 195min 降到 115min（**1.7x**）。这种 system-level 优化比单点算法改进更稀缺。

4. **Evaluation protocol 即 contribution**：在 large-scale geometry 评估领域，没有 reliable benchmark 是阻碍进展的根本原因。TnT-style + visibility-based crop 把 error bar 从 0.1 降到 0.003，让后续工作的比较有了 statistical meaning。

## 七、相关 Links

- **Project page**: [https://dekuliutesla.github.io/CityGaussianV2/](https://dekuliutesla.github.io/CityGaussianV2/)
- **3DGS original**: [Kerbl et al., 2023](https://repo.sam.lum.is/3dgaussian-splatting/)
- **2DGS**: [Huang et al., 2024](https://arxiv.org/abs/2403.17888)
- **CityGaussian (V1)**: [Liu et al., 2024](https://arxiv.org/abs/2404.01133)
- **Depth-Anything V2**: [Yang et al., 2024](https://arxiv.org/abs/2406.09414)
- **TrimGS**: [Fan et al., 2024](https://arxiv.org/abs/2406.07499)
- **GOF (Gaussian Opacity Fields)**: [Yu et al., 2024c](https://arxiv.org/abs/2404.10772)
- **SuGaR**: [Guedon & Lepetit, 2024](https://arxiv.org/abs/2311.12775)
- **GauU-Scene**: [Xiong et al., 2024](https://arxiv.org/abs/2410.18942)
- **Tanks and Temples**: [Knapitsch et al., 2017](https://wwwtanksandtemples.org/)
- **VastGaussian**: [Lin et al., 2024](https://arxiv.org/abs/2402.04332)
- **LightGaussian**: [Fan et al., 2023](https://arxiv.org/abs/2311.17245)
- **Mip-Splatting**: [Yu et al., 2024b](https://arxiv.org/abs/2312.00257)
- **MatrixCity**: [Li et al., 2023a](https://arxiv.org/abs/2304.07136)
- **FlashGS**: [Feng et al., 2024](https://arxiv.org/abs/2408.07967)
- **HierarchicalGS**: [Kerbl et al., 2024](https://arxiv.org/abs/2406.12080)
- **OctreeGS**: [Ren et al., 2024](https://arxiv.org/abs/2403.17898)
- **DoGaussian**: [Chen & Lee, 2024](https://arxiv.org/abs/2405.13943)
- **GrendelGS**: [Zhao et al., 2024](https://arxiv.org/abs/2406.18533)

## 八、最终 Thoughts

CityGaussianV2 的 elegance 在于它**没有发明新理论**，而是对 2DGS 在 large-scale 下的 failure modes 做了精准诊断 + 对症下药。DGD 解决 convergence，Elongation Filter 解决 scalability，SH degree 2 + integrated trim 解决 pipeline overhead，visibility-based TnT protocol 解决 evaluation。

这种 engineering-heavy 的 paper 在工业落地角度价值极高——把 31.5GB memory 降到 14.2GB，把 4.6GB storage 降到 0.44GB，让 2DGS 第一次真正能跑在城市级场景上。对想要 build production-grade GS pipeline 的人来说，这篇 paper 几乎每一个 trick 都值得直接采纳。
