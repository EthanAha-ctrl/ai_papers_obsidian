---
source_pdf: ONSCALINGUP3DGAUSSIANSPLATTINGTRAINING.pdf
paper_sha256: 6aa1d42ea66e2b86efd979689f23addb4cd093b23cb358c2175b8762cf943353
processed_at: '2026-08-05T23:58:50-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Grendel

好，咱们把这篇 paper 拆开揉碎了聊。

---

## 一句话说清楚这 paper 干了啥

3D Gaussian Splatting（3DGS）是个特别火的 3D 重建方法，但它有个要命的毛病：**只能在一张 GPU 上跑**。一张 A100 40GB 最多塞下 1100 万个 Gaussians，但实验证明你给它 4000 万个，PSNR 还能涨 1 个 dB。所以问题变成了——**怎么把 3DGS 训练拆到多张 GPU 上**。

这听起来像是个 trivial 的问题（DDP 嘛，谁不会），但其实特别 tricky，因为 3DGS 的 computation pattern 跟 neural network 完全不是一个物种。

---

## 为什么不能直接抄 DDP / FSDP 的作业

你想，我们训 LLM 的时候，一张卡算一个 batch 的 forward，然后 all-reduce gradients，多简单。因为 LLM 的 computation 是 **dense matrix multiply**，每个 parameter 都被每次 forward 碰一遍，workload 天然 balanced。

但 3DGS 不是这样。它的一个 training iteration 长这样：

```
Gaussians (几千万个) ──transform──> projected Gaussians ──render──> Image ──loss──> gradients
```

注意这里有两个完全不同的 parallelism axis：

- **Transform step**：对每个 Gaussian 独立算它在 screen 上的投影。这是 **per-Gaussian parallel**，你按 Gaussian 切分就行。
- **Render step**：对每个 pixel，找出所有覆盖它的 Gaussians，按 depth 排序做 alpha blending。这是 **per-pixel parallel**，你得按 pixel 切分。

所以问题来了：transform 阶段你按 Gaussian 分，render 阶段你按 pixel 分，**中间必须 shuffle data**。这跟 FSDP 的 dense all-gather 完全不同。

更恶心的是，**一个 pixel 被哪些 Gaussians 覆盖是 dynamic 的**——取决于当前 camera pose 和 Gaussian 的 position/radius，而且训练过程中 Gaussian 还在不断 clone/split/prune。你没法预先确定 partition。

---

## 第一个 key insight：Spatial Locality 救了你

如果你仔细看 3DGS 训练中的 Gaussians，**90% 的 Gaussian 的 radius 小于 image width 的 2%**（Figure 3）。也就是说，一个 Gaussian 只影响 image 中很小一块区域。

这意味着什么？意味着当你从 Gaussian-wise partition 切到 pixel-wise partition 的时候，**每个 GPU 只需要拿一小撮 Gaussians**，不是全部。这叫 **sparse all-to-all**。

对比一下 FSDP：FSDP 在每次 forward 前做 dense all-gather，把所有 weights 拿到每张卡。Grendel 不需要——每个 GPU 只拉自己负责的 pixel region 真正会被 intersect 到的那些 Gaussians。通信量直接降一个数量级。

**Intuition**：3DGS 的 Gaussians 就像一堆小灯泡，每个灯泡只照亮屏幕一小块。你不需要把所有灯泡都搬到每个屏幕区域，只需要搬照得到的那几个。

---

## 第二个 key insight：Workload 是 dynamic imbalanced 的

你看 Figure 4 那张 heatmap：同一张 image 里，天空区域只有几个 Gaussians 覆盖（远、简单），但前景的人物/建筑可能有几百个 Gaussians 叠在一起（近、复杂）。如果你把 image 均匀切成 4 块分给 4 个 GPU，**有的 GPU 累死，有的 GPU 闲死**。

而且这个 imbalance 会随训练变化——训练早期全局结构先建好，后期细节区域才 densify。所以 **static partition 一定 suboptimal**。

Grendel 的解法特别朴素（Algorithm 1）：

1. 前几个 epoch 记录每个 $16 \times 16$ pixel block 的 rendering time
2. 假设相邻 epoch 间 workload 变化缓慢
3. 用 cumulative sum + binary search 把 blocks 按 **累计 cost** 而非 **block 数量** 均分给 GPU

就是个贪心的 prefix sum partition，$O(G \log B)$ 复杂度，几乎零开销。

**人话**：与其让每个 GPU 拿一样多的 blocks（但实际活儿不一样多），不如让每个 GPU 拿一样多的"预计活儿"。天空区域多分点 blocks（反正每个 block 活儿少），前景区域少分点 blocks（每个 block 活儿多）。

---

## 第三个 key insight：Batched Training 的 Scaling Rule

这部分我觉得是 paper 里最 elegant 的。

### 问题

单 GPU 3DGS 用 batch size = 1（一次一个 view）。多 GPU 你自然想用大 batch（比如 32），否则 GPU utilization 上不去。但大 batch 不调 learning rate 会训崩或训不动——这是 deep learning 的老问题了。

对于 SGD，Goyal et al. (2017) 的经典 rule 是 **linear scaling**：$\lambda' = \lambda \times b$。
对于 Adam，Malladi et al. (2022) 推出 **sqrt scaling**：$\lambda' = \lambda \times \sqrt{b}$。

但 3DGS 有个特殊性：**它的 gradients 极其 sparse**。一个 Gaussian 只会被"看到"它的那些 view 产生非零 gradient。所以如果你随机采 $b$ 个 view，这些 view 之间的 gradient 几乎不重叠——**近似 independent**。

### 推导（人话版）

假设 $g_k$ 是 view $k$ 上的 gradient，不同 view 的 gradient 独立零均值。

**Batch size = 1 的 Adam update（无 momentum）**：

$$
\Delta^{\{k\}} = \frac{g_k}{\sqrt{\mathbb{E}[\bar{g}^2] \cdot |V|}}
$$

这里 $|V|$ 是所有 view 数，$\bar{g}$ 是 full-batch mean gradient。分母是 Adam 的二阶矩估计（开方后）。

**Batch size = $b$ 的 Adam update（无 momentum）**：

$$
\Delta^{\{B\}} = \frac{\sum_{k \in B} g_k / b}{\sqrt{\mathbb{E}[\bar{g}^2] \cdot |V| / b}} = \frac{1}{\sqrt{b}} \cdot \frac{\sum_{k \in B} g_k}{\sqrt{\mathbb{E}[\bar{g}^2] \cdot |V|}}
$$

关键在最后那个 $\frac{1}{\sqrt{b}}$。为什么？因为分子是 gradient 的 **mean**（除以 $b$），但分母是 gradient 的 **second moment**（平方后开方），second moment 只除以 $\sqrt{b}$ 而不是 $b$。

所以 $\Delta^{\{B\}} = \frac{1}{\sqrt{b}} \sum_{k \in B} \Delta^{\{k\}}$。

要让 batch $b$ 的一步等价于 $b$ 个 batch-1 步，**learning rate 乘 $\sqrt{b}$**：

$$
\lambda' = \lambda \times \sqrt{\text{batch\_size}}
$$

**Intuition**：Adam 的 denominator 是 $\sqrt{\text{second moment}}$。gradient mean 之后，分子线性变小，但 denominator 只按 $\sqrt{b}$ 变小（因为是平方再开方），所以 update 被 $\frac{1}{\sqrt{b}}$ 缩了。补回来就是乘 $\sqrt{b}$。

### Momentum 怎么办

Adam 还有 momentum（EMA of gradients and squared gradients）。如果你 batch 变大，每个 step "看到" $b$ 个 view，相当于 batch-1 训练走了 $b$ 步。EMA 系数 $\beta$ 在 $b$ 步后衰减为 $\beta^b$，所以 batch $b$ 时 momentum 系数应该用：

$$
\beta' = \beta^{\text{batch\_size}}
$$

这样 EMA 的 effective horizon 保持一致。这个 trick 来自 Busbridge et al. (2023)。

### Empirical 验证（Figure 6, 12）

Figure 6：在 Rubble 上测 gradient 的 inverse variance，随 batch size **线性增长**到 b=32 左右 plateau。线性增长正是 independent gradients 的 signature（variance $\propto 1/b$，inverse variance $\propto b$）。Plateau 说明 batch 太大后 gradients 开始 correlated（相似 view 的 gradient 重叠）。

Figure 12：用 batch-1 训到 15K iter，reset optimizer state，切换到不同 batch size，比较 update 跟 batch-1 baseline 的 cosine similarity。**只有 sqrt LR scaling + exponential momentum scaling 同时用，cosine similarity 才保持高**。constant LR 太小，linear LR 太大，不调 momentum 也不行。

---

## 实验里最 striking 的数字

### Gaussian 数量 vs 质量（Table 4, Figure 11）

Rubble scene（4K 分辨率）：

| Gaussians | PSNR | 备注 |
|-----------|------|------|
| 2.1M | 24.84 | |
| 11.2M | 26.28 | **单 GPU 极限** |
| 40.4M | **27.28** | 16 GPU |

从单 GPU 极限到 16 GPU，PSNR **+1 dB**。在 3DGS 这个领域，1 dB 是很大的提升。而且曲线没饱和，说明还能继续涨。

**这就是这篇 paper 存在的核心理由**：单 GPU 装不下足够 Gaussians 来达到 quality saturation，你必须 distribute。

### vs CityGaussian（Table 2）

CityGaussian 是 divide-and-conquer 方法：先 coarse train，切 scene 成 cells，每个 cell 独立 train，最后 merge。流程繁琐，每步要 tune。

Grendel 在 Rubble 上：PSNR 27.39 vs 25.40（CityGS 200K），**+2 dB**，时间 0.85h vs 2.18h，**2.6x 快**。

而且 Grendel 用起来跟原版 3DGS 一样——你指定 GPU 数就行，不用搞 coarse training / partition / merge 那套。

### 训练速度（Table 8）

Train scene（小 scene）：16 GPU bsz=32，30K images，**2 分 43 秒**。号称 SOTA training speed。

---

## 一些 system 细节值得品味

### Densification 是 local 的

Clone/split/prune 的决策基于单个 Gaussian 的 position variance 和 scale threshold，完全是 per-Gaussian 的。所以 Grendel 让每个 GPU 对自己的 Gaussians **独立做 densification**，不需要 cross-GPU 通信。

但问题：不同 region densify 速率不同，会导致 Gaussian 分布不均。所以定期 redistribute Gaussians。

有意思的是，paper 发现 **random redistribution 反而最快**——虽然 total communication volume 不是最优，但 NCCL all-to-all 偏好 uniform send/recv volume，random 让每个 GPU 的通信量均匀。这是个 system-level 的 counterintuitive 发现。

### Z-Buffer 是内存大头

Z-buffer 存每个 pixel 的 intersecting Gaussian indices。由于一个 Gaussian 会覆盖多个 pixels，Z-buffer 总量 > Gaussians 数 × pixels 数。这是 high-res / large-batch 训练 OOM 的主因。

这也解释了为什么 batch size 增大会显著减少能容纳的 Gaussian 数（Table 9）：bsz=1 时 16 GPU 能装 230M Gaussians，bsz=16 时只能装 75M。Activation memory 随 batch 线性涨。

### Forward vs Backward 的 imbalance 不同

- Forward 复杂度 $\propto$ ray intersect 的 Gaussian 数
- Backward 复杂度 $\propto$ 真正贡献到 pixel color 的 Gaussian 数（opacity saturation 前的）

所以 forward 和 backward 的热点区域不一样。Load balancer 需要综合考虑 forward + backward time。

---

## 我（假装 Karpathy 视角）的看法

这篇 paper 最让我欣赏的是它 **没有强行套用 DNN distributed training 的模板**。很多人遇到"训 3DGS 太慢"的第一反应是"上 DDP / FSDP"，但 3DGS 的 computation pattern（sparse、irregular、dynamic、mixed parallelism）跟 dense NN 完全不同，硬套会出问题。

Grendel 的做法是先理解 3DGS 的 workload 本质：

1. **Mixed parallelism** → 不同 stage 不同 partition axis，中间 sparse all-to-all
2. **Spatial locality** → sparse all-to-all 可行（通信量正比于实际 intersect 而非总数）
3. **Dynamic workload** → adaptive load balancing（history-based cost estimation）

这三条 observation 每一条都很朴素，但组合在一起就是 3DGS 专属的 distributed system design。这比"把 3DGS 塞进 PyTorch DDP"高明得多。

Optimization 部分也很漂亮。Independent gradients hypothesis 对 3DGS 特别成立（因为 gradient 极 sparse），所以 sqrt scaling + exponential momentum scaling 是自然的。推导干净，empirical validation 也有说服力。

**唯一让我觉得可以 push 的**：batch size 只到 32，Figure 6 显示 independent gradients 在 b=32 已经 plateau。如果要上 b=128 或 b=256（像 LLM 那样），可能需要新的 scaling rule 或者 gradient compression。这是个好的 future direction。

另外一个联想：3DGS 的 densification 机制本质上是一种 **dynamic sparse growth**，这跟 neural network 的 pruning/growing（比如 Lottery Ticket、RigL）有神似之处。如果未来 3DGS 变体引入 cross-Gaussian attention 或 global reasoning（比如 Scaffold-GS Lu et al., 2024 的 structure），distributed design 会更复杂——可能需要引入 tensor parallelism 的思路。但现阶段 pure explicit representation 的 3DGS，Grendel 这套设计是恰到好处的。

---

## 相关链接汇总

**核心**:
- Paper: https://arxiv.org/abs/2406.18533
- Code: https://github.com/nyu-systems/Grendel-GS
- 原版 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

**对比方法**:
- CityGaussian: https://arxiv.org/abs/2404.16873
- DOGS: https://arxiv.org/abs/2405.13943
- RetinaGS: https://arxiv.org/abs/2406.11836
- VastGaussian: https://arxiv.org/abs/2402.17427
- Hierarchical 3DGS: https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/
- OctreeGS: https://arxiv.org/abs/2403.17898
- Scaffold-GS: https://arxiv.org/abs/2312.00109

**Scaling rule 理论**:
- SGD linear scaling (Goyal): https://arxiv.org/abs/1706.02677
- Adam SDE scaling (Malladi): https://arxiv.org/abs/2206.01729
- EMA scaling (Busbridge): https://openreview.net/forum?id=DkeeXVdQyu
- LARS/LAMB (You): https://arxiv.org/abs/1904.00962
- Large batch generalization gap (Keskar): https://openreview.net/forum?id=H1oyRlYgg

**分布式训练系统**:
- FSDP: https://arxiv.org/abs/2304.11277
- ZeRO: https://arxiv.org/abs/1910.02054
- Megatron-LM: https://arxiv.org/abs/1909.08053
- Alpa: https://arxiv.org/abs/2201.12019
- PyTorch DDP: https://arxiv.org/abs/2006.15704

**数据集**:
- Rubble / Mega-NeRF: https://arxiv.org/abs/2112.10703
- MatrixCity: http://city-dataset.com/
- Mip-NeRF 360: https://jonbarron.info/mipnerf360/
- Tanks & Temples: https://tanksandtemples.org/
- DeepBlending: https://github.com/weiwei-performing/DeepBlending

**硬件**:
- Perlmutter (NERSC): https://docs.nersc.gov/systems/perlmutter/architecture/

---

TL;DR：3DGS 单卡训不动大 scene，因为 Gaussians 太多装不下。Grendel 把 Gaussians 拆到多卡，用 sparse all-to-all 在 Gaussian-wise 和 pixel-wise partition 间切换，用 history-based load balancing 处理 dynamic imbalance，用 sqrt LR + exponential momentum scaling 处理 batched training。结果是 4K Rubble scene 上 PSNR 从 26.28（单卡极限）涨到 27.28（16 卡 40M Gaussians），而且比 CityGaussian 快 3x 还更简单。Core insight：3DGS 不是 neural network，别硬套 DDP/FSDP，要针对它的 sparse/dynamic/mixed-parallelism pattern 专门设计。

---

# Grendel: Scaling Up 3D Gaussian Splatting Training — 详细讲解

## 1. 背景与动机

### 1.1 3DGS 的崛起与瓶颈

3D Gaussian Splatting (3DGS) 由 Kerbl et al. (2023) 提出，相比 NeRF (Mildenhall et al., 2020) 在 training speed 和 rendering speed 上都有显著优势。3DGS 用一组 anisotropic 3D Gaussians 显式表示场景，每个 Gaussian 有四类 learnable parameters：

- **Position** $x_i \in \mathbb{R}^3$：Gaussian 在 3D 空间中的中心位置
- **Shape**：通过 scaling vector $s_i \in \mathbb{R}^3$ 和 rotation quaternion $q_i \in \mathbb{R}^4$ 构造 covariance matrix
- **Opacity** $\alpha_i \in \mathbb{R}$：不透明度
- **Spherical Harmonics** $sh_i \in \mathbb{R}^{48}$：48 维系数，用于 view-dependent color（一般是 3 阶 SH，$16 \times 3 = 48$）

**问题在于**：传统 3DGS 只能在单 GPU 上训练。以 Rubble dataset (Turki et al., 2022) 为例，4K 分辨率（$4591 \times 3436$），1657 张图像。一个 A100 40GB 最多只能容纳 11.2M Gaussians，这远未达到 quality saturation point。从 Figure 11 可以看到，Gaussian 数量越多 PSNR/SSIM 越好，LPIPS 越低，且没有饱和迹象。

### 1.2 为什么不能直接用 DNN 分布训练框架？

这是这篇 paper 最关键的 insight 之一。现有的 Megatron-LM (Shoeybi et al., 2020)、ZeRO (Rajbhandari et al., 2020)、PyTorch FSDP (Zhao et al., 2023)、Alpa (Zheng et al., 2022) 等框架都假设 **consistent and balanced workload with regular dense tensor operations**。但 3DGS 是基于 explicit point cloud 而非 neural network，computation pipeline 有三个本质特征：

1. **Mixed parallelism**：不同阶段需要不同的 partition axis
2. **Spatial locality**：每个 Gaussian 只影响 image 中一小块 contiguous 区域
3. **Dynamic and unbalanced workload**：load 在空间和时间上都剧烈变化

所以直接套用 FSDP/ZeRO 的策略会失效。

---

## 2. System Design：Grendel 的核心架构

### 2.1 Mixed Parallelism（混合并行）

3DGS 一个 training iteration 包含四步：
1. **Gaussian transformation**：对每个 Gaussian 独立计算 screen-space position $x_{v,i} \in \mathbb{R}^2}$、depth $\text{depth}_{v,i}$、footprint radius $\text{radius}_{v,i}$、view-dependent color $c_{v,i}$。这一步是 **Gaussian-wise parallel**。
2. **Image rendering**：对每个 pixel，找出所有 intersecting Gaussians，按 depth 排序后 alpha-compositing。这一步是 **Pixel-wise parallel**。
3. **Loss calculation**：L1 loss 和 SSIM loss 都是 per-pixel 计算。**Pixel-wise parallel**。
4. **Backpropagation**：rendering backward 是 pixel-wise，Gaussian transformation backward 是 Gaussian-wise。

Grendel 的设计是：
- **Gaussian transformation (forward + backward)**：用 **Gaussian-wise distribution**，将 Gaussians（包括 parameters、gradients、optimizer states）均匀切分到各个 GPU。
- **Rendering + Loss (forward + backward)**：用 **Pixel-wise distribution**，将 image 分成 $16 \times 16$ 的 block，把 contiguous block 分配给不同 GPU。

### 2.2 Sparse All-to-All Communication

切换 Gaussian-wise 和 pixel-wise 之间需要 transfer Gaussians。这里的关键 observation 是 **spatial locality**：

从 Figure 3 可以看到，在 Rubble、Bicycle、Train 三个数据集中，**90% 的 Gaussians 的 radius < image width 的 2%**。也就是说，一个 Gaussian 只影响 image 中很小一块区域。

因此 Grendel 不像 FSDP 那样做 dense all-gather，而是做 **sparse all-to-all**：

1. 每个 GPU 先基于本地 pixel partition 决定哪些 Gaussians 与之 intersect（Figure 5）
2. 然后通过 sparse all-to-all 只 transfer 这些 intersecting Gaussians
3. Backward pass 时做反向的 sparse all-to-all 传 gradients

这与 FSDP 的两个本质区别（paper §3.1 末段强调）：
- FSDP 的 weight sharding 只是为了 storage，computation 还是要 gather 全部 weights；Grendel 的 Gaussian-wise distribution 既为了 storage 也为了 computation（Gaussian transformation 直接在 local shard 上算）。
- FSDP 用 dense all-gather；Grendel 用 sparse all-to-all，通信量正比于实际 intersect 的 Gaussians 数。

### 2.3 Iterative Workload Rebalancing

#### 2.3.1 Pixel-wise Rebalancing

由于 dynamic workload（Figure 4 显示同一张 image 不同 tile 的 Gaussian 数量差异巨大），static partitioning 会严重 load imbalance。

Grendel 的策略：
1. 在前几个 epoch 之后开始记录每个 $16 \times 16$ pixel block 的 rendering time
2. 假设相邻 epoch 间 rendering time 变化平滑（因为 scene 训练中变化缓慢）
3. 用 historical rendering time 估计当前 epoch 各 block 的 cost
4. 用 **Algorithm 1** 计算 division points，把 blocks 按累计 cost 均分到各 GPU

**Algorithm 1 解析**：
```
输入：B (pixel blocks 数), G (GPU 数), ET (Estimated runtime per block)
1: CT ← cumsum(ET)              # CT[i] = sum(ET[0..i])，即累计 cost
2: ET_gpu ← CT[B-1] / G         # 每个 GPU 应该承担的总 cost
3: TH ← arange(0, G) · ET_gpu   # G 个 threshold: 0, ET_gpu, 2*ET_gpu, ..., (G-1)*ET_gpu
4: DP ← searchsorted(CT, TH)    # 在 CT 中二分查找 threshold 位置，得到 G-1 个 division points
5: return DP
```

`torch.searchsorted` 是 monotonic sequence 上的二分查找，复杂度 $O(G \log B)$，非常快。这样 GPU 0 拿到 $[0, DP_1)$ 的 blocks，GPU 1 拿到 $[DP_1, DP_2)$，依此类推。每个 GPU 拿到的总 cost 大致相等。

#### 2.3.2 Gaussian-wise Rebalancing

Densification 过程（clone + split）会让不同区域的 Gaussian 数量增长不均，所以每隔几次 densification 后要重新 redistribute Gaussians 以保持均匀。

---

## 3. Batched Training 与 Hyperparameter Scaling

这是 paper 的另一个核心贡献，也是 Karpathy 你最可能感兴趣的 optimization dynamics 部分。

### 3.1 为什么需要 batched training？

单 GPU 3DGS 用 batch size = 1（一次一个 view）足够了，但在分布式系统中这会导致 GPU utilization 低（每次只有少数 GPU 工作）。Grendel 支持 batch size 最大到 32（甚至 64）。但增大 batch size 不调 hyperparameter 会导致 unstable / inefficient training（Goyal et al., 2017; Qiao et al., 2021）。

### 3.2 Independent Gradients Hypothesis（核心假设）

**假设**：不同 camera view 上算出的 gradients 是独立的。形式化地，记 $g_k$ 为某个 parameter 在 view $k$ 上的 gradient，$\bar{g} = \frac{\sum_{j \in V} g_j}{|V|}$ 为 full-batch gradient，假设 $\mathbb{E}[g_k] = 0$，则：

$$
\text{Cov}(g_k, g_j) = \mathbb{E}[g_k g_j] = \begin{cases} 0 & k \neq j \\ \mathbb{E}[g_k^2] & k = j \end{cases}
$$

**为什么这个假设对 3DGS 大致成立？** 因为 3DGS 的 gradient 对每个 parameter 都非常 sparse——只有真正"看到"这个 Gaussian 的 view 才会对其产生非零 gradient。所以 random 采样的不同 view 之间的 gradient 重叠很少，近似独立。

Figure 6 是关键的 empirical evidence：在 Rubble 数据集上，对 diffuse color parameters 的 gradient，**inverse variance 随 batch size 线性增长**，到某个点后 plateau。线性增长正是 independent gradients 的特征——若 $g_k$ 独立同分布，方差 $\sigma^2$，则 batch 平均的 variance 为 $\sigma^2/b$，inverse variance 为 $b/\sigma^2$，线性于 $b$。

### 3.3 Square-Root Learning Rate Scaling 的推导

这是 paper 最 elegant 的部分。我们要找一个 scaling rule，使得 batch size $b$ 的一个 Adam step 等价于 $b$ 个 batch-size-1 的 Adam step。

**Step 1: Batch-size-1 Adam update（无 momentum）**

在 view $k$ 上，Adam 不带 momentum 的 update 为：

$$
\Delta^{\{k\}} = \frac{g_k}{\sqrt{\mathbb{E}\left[\mathbb{E}_{j \in V}[g_j^2]\right]}}
$$

由独立假设：
$$
\mathbb{E}_{j \in V}[g_j^2] = \frac{1}{|V|}\sum_{j \in V} g_j^2 = \frac{1}{|V|} \cdot |V| \cdot \bar{g}^2 \cdot |V| / |V| = |V| \bar{g}^2
$$

（这里 paper 推导有点跳跃，核心是 $\sum g_j^2 = |V| \bar{g}^2$ 在独立零均值假设下的期望）

所以：

$$
\Delta^{\{k\}} = \frac{g_k}{\sqrt{|V| \mathbb{E}[\bar{g}^2]}} = \frac{g_k}{\sqrt{|V|} \sqrt{\mathbb{E}[\bar{g}^2]}}
$$

**Step 2: Batch size $b$ 的 Adam update（无 momentum）**

在 batch $B \subseteq V$，$|B| = b$ 上：

$$
\Delta^{\{B\}} = \frac{\sum_{k \in B} g_k / b}{\sqrt{\mathbb{E}\left[\mathbb{E}_{B' \subseteq V}\left[(\sum_{j \in B'} g_j / b)^2\right]\right]}}
$$

由独立假设：
$$
\mathbb{E}_{B' \subseteq V}\left[(\sum_{j \in B'} g_j / b)^2\right] = \frac{1}{b^2} \cdot b \cdot \mathbb{E}[g^2] \cdot |V|/b \cdot b = \frac{|V|}{b} \bar{g}^2
$$

（这里 $b$ 个 independent gradients，方差加和，平均后除以 $b^2$，再考虑 sampling）

所以：

$$
\Delta^{\{B\}} = \frac{\sum_{k \in B} g_k / b}{\sqrt{\frac{|V|}{b} \mathbb{E}[\bar{g}^2]}} = \frac{\sum_{k \in B} g_k / b}{\sqrt{|V|/b} \sqrt{\mathbb{E}[\bar{g}^2]}} = \frac{1}{\sqrt{b}} \cdot \frac{\sum_{k \in B} g_k}{\sqrt{|V|} \sqrt{\mathbb{E}[\bar{g}^2]}}
$$

**Step 3: 比较**

注意 $\sum_{k \in B} \Delta^{\{k\}} = \sum_{k \in B} \frac{g_k}{\sqrt{|V|}\sqrt{\mathbb{E}[\bar{g}^2]}}$，所以：

$$
\Delta^{\{B\}} = \frac{1}{\sqrt{b}} \sum_{k \in B} \Delta^{\{k\}}
$$

要让 batch update 等于 $b$ 个 single updates 之和，需要乘以 $\sqrt{b}$，即 **learning rate scaling**：

$$
\boxed{\lambda' = \lambda \times \sqrt{\text{batch\_size}}} \quad \text{(Eq 1)}
$$

这跟 Malladi et al. (2022) 对 Adam 的推导一致，但比 SGD 的 linear scaling (Goyal et al., 2017) 弱。

**Intuition**：Adam 的 denominator 是 gradient 的二阶矩估计。batch 内 gradient 平均后，分子线性减小（除以 $b$），但 denominator 的二阶矩只按 $\sqrt{b}$ 减小（因为是平方再开方），所以 update 整体被 $\frac{1}{\sqrt{b}}$ 缩小。要补偿，learning rate 要乘 $\sqrt{b}$。

### 3.4 Exponential Momentum Scaling

公式 (Eq 2)：

$$
\boxed{\beta_1' = \beta_1^{\text{batch\_size}}, \quad \beta_2' = \beta_2^{\text{batch\_size}}}
$$

这个 rule 来自 Busbridge et al. (2023)。直觉是：batch size 越大，每个 step "看到" 的 view 越多，相当于 batch-size-1 训练中已经做了 $b$ 步。EMA 系数 $\beta$ 在 $b$ 步后的衰减是 $\beta^b$，所以 batch $b$ 训练时 momentum 衰减应该用 $\beta^b$ 来匹配 effective horizon。

**关键区别**：linear scaling rule（Goyal et al., 2017）是 SGD 的；Malladi 的 sqrt LR scaling 没配 momentum scaling；Grendel 是 **同时** sqrt LR scaling + exponential momentum scaling，且专为 3DGS 的 Adam 调优。

### 3.5 Empirical Validation（Figure 12）

实验设计很巧妙：
1. 在 Rubble 上用 batch size=1 训练到 15K iterations
2. Reset Adam optimizer states
3. 切换到不同 batch size (4, 16, 32)，应用不同的 scaling rule
4. 比较 weight update 与 batch-size=1 baseline 的 cosine similarity 和 norm

**Figure 12a**：固定 exponential momentum scaling，比较 LR scaling ∈ {constant, sqrt, linear}。只有 **sqrt** 在不同 batch size 下保持高 cosine similarity 和稳定的 update norm。constant scaling 让 update 太小（underfit），linear scaling 让 update 太大（unstable）。

**Figure 12b**：固定 sqrt LR scaling，比较 momentum scaling ∈ {unchanged, exponential}。**Exponential** 显著保持更高 cosine similarity。

---

## 4. 实验结果

### 4.1 数据集与 Setup

- **Large-scale**：Rubble (4K, 1657 imgs), MatrixCity Block_All (1080P, 5620 imgs)
- **Small-scale**：Mip-NeRF 360 (1080P, 9 scenes), Tanks & Temples (~1K, 2 scenes), DeepBlending (~1K, 2 scenes)
- **Hardware**：Perlmutter cluster, 每 node 4×A100 40GB, NVLink 25GB/s/direction, 200Gbps Slingshot inter-node

### 4.2 Throughput Scaling（Table 7, 8；Figure 7, 8, 9, 10）

**Rubble (Table 7)**：
- 4 GPU bsz=1: 5.55 img/s
- 4 GPU bsz=4: 7.28 img/s
- 16 GPU bsz=32: 25.18 img/s
- 32 GPU bsz=64: 38.03 img/s（~6.8x speedup vs 4 GPU bsz=1）

**Train (Table 8)**：
- 1 GPU bsz=1: 34.72 img/s, PSNR 21.84
- 16 GPU bsz=32: 185.19 img/s, PSNR 21.76
- **16 GPU bsz=16 训练 30K images 仅需 2 min 42.97 s**（号称 SOTA training speed）

**Mip-NeRF 360 + TT & DB (Table 3, Figure 9)**：4 GPU bsz=4 vs 1 GPU bsz=1，throughput 3-4x，PSNR 几乎一致（部分 scene 略降，部分略升）。

### 4.3 Gaussian Quantity vs Quality（Figure 11, Table 4, 5, 6）

这是 **paper 最重要的 takeaway**：**Gaussian 数量直接决定 reconstruction quality，且单 GPU 装不下足够 Gaussian**。

**Rubble (Table 4)**：
| EXP | n3dgs | PSNR | SSIM | LPIPS |
|-----|-------|------|------|-------|
| EXP1 | 2.11M | 24.84 | 0.70 | 0.48 |
| EXP4 | 11.17M | 26.28 | 0.78 | 0.37 |（单 GPU 上限）
| EXP8 | **40.40M** | **27.28** | 0.82 | 0.29 |（16 GPU）

从 11.17M (单 GPU 极限) 到 40.40M (16 GPU)，PSNR 从 26.28 → 27.28，**+1 dB**。这在 3DGS 论文中已经是显著提升。

**MatrixCity Block_All (Table 5)**：从 1.5M (24.41) 到 30M (26.96)，PSNR +2.55 dB。

**Bicycle (Table 6)**：从 2.2M (24.09) 到 9.6M (24.85)，PSNR +0.76 dB。

### 4.4 与 CityGaussian 对比（Table 2, 10）

CityGaussian (Liu et al., 2024) 是 divide-and-conquer 方法：先 coarse training，然后 partition scene 成 cells，每个 cell 独立训练，最后 merge point cloud。

**4 GPU, 200K images**：
| Method | Rubble PSNR | Time | Building PSNR | Time | MatrixCity PSNR | Time |
|--------|------------|------|---------------|------|-----------------|------|
| CityGS Official | 25.88 | 2.88h | 22.14 | 4.57h | 27.41 | 8.25h |
| CityGS 200K | 25.40 | 2.18h | 20.32 | 2.22h | 23.68 | 3.60h |
| **Grendel 200K** | **27.39** | **0.85h** | **22.69** | **0.90h** | 27.33 | **1.22h** |

Grendel 在 Rubble 上 PSNR 高 1.5 dB，时间快 3.4x；MatrixCity 上 PSNR 相当但时间快 6.8x。

**Table 10 的 Time Decomposition** 很说明问题：CityGaussian 有 coarse training、data partition、per-cell training、merge 四步，每步都要单独 tune；Grendel 就是单步 distributed training，使用复杂度跟原版 3DGS 一样。

### 4.5 Memory Scaling（Table 9, Figure 14）

Rubble 上不同 GPU 数能容纳的 Gaussian 数（bsz=1/4/16）：
- 1 GPU: 12.71M (bsz=1), 7.10M (bsz=4), OOM (bsz=16)
- 16 GPU: 230.41M (bsz=1), 169.37M (bsz=4), 74.98M (bsz=16)
- 32 GPU: 354.46M (bsz=1), 313.10M (bsz=4), 150.21M (bsz=16)

**Linear scaling**：GPU 数翻倍，可容纳 Gaussian 数近似翻倍。Batch size 增大会占用 activation memory（Z-buffer 等），所以 bsz=16 时能装的 Gaussian 比 bsz=1 少。

### 4.6 Render Speedup（Table 11）

Inference 也受益于多 GPU：4 GPU vs 1 GPU speedup 1.88x-2.63x。

---

## 5. 一些细节与 Intuition

### 5.1 Z-Buffer 内存压力（Appendix A.2）

Z-buffer 存储每个 pixel 的 intersecting Gaussian indices。由于一个 Gaussian 会投射到其 footprint 内的多个 pixels，所有 pixels 的 Z-buffer 总和大于 Gaussians 数 × pixels 数。这是 high-res / large-scene / large-batch 训练时 OOM 的主要原因。

**Intuition**：这就是为什么 batch size 增大会显著减少能容纳的 Gaussian 数——activation memory（Z-buffer 等）随 batch size 线性增长。

### 5.2 Densification 在 distributed 下的处理（Appendix A.3）

Densification（clone/split/prune）是 per-Gaussian 的决策，基于 position variance 和 scale threshold。所以 Grendel 在 **local GPU 上独立执行 densification**——每个 GPU 对自己 shard 的 Gaussians 做 clone/split/prune，不需要 cross-GPU 通信。这很优雅。

但问题来了：clone/split 会让不同 GPU 的 Gaussian 数量失衡，所以才有 §3.2.2 的 periodic Gaussian redistribution。

### 5.3 Forward vs Backward 的 Unbalance 不同（Appendix A.4）

- **Forward render** 复杂度正比于 ray intersect 的 Gaussian 数
- **Backward render** 复杂度正比于 **真正贡献到 pixel color 的 Gaussian 数**（opacity saturation 前的那些）

这意味着 forward 和 backward 的 load imbalance pattern 不同，load balancer 需要综合考虑。

### 5.4 Block Size 16×16 的 trade-off（Appendix B.1）

Block 是 scheduling granularity。太小：scheduler overhead 大；太大：单个 block 内 workload 不均无法 redistribute。16×16 = 256 pixels 是经验值。

### 5.5 Gaussian Redistribution 策略（Appendix B.2）

有趣的是，paper 发现 **random redistribution 反而最快**，虽然 communication volume 不是最优。因为 NCCL all-to-all 偏好 uniform send/recv volume。如果用 custom communication primitive 只关心 total volume，可能需要不同策略。这是 system-level 的 trade-off。

---

## 6. 与 Related Work 的对比

### 6.1 vs FSDP (Zhao et al., 2023)

| 维度 | FSDP | Grendel |
|------|------|---------|
| Sharding 对象 | Neural network weights | 3D Gaussians |
| Sharding 目的 | Storage only | Storage + Computation |
| Communication | Dense all-gather | Sparse all-to-all |
| Workload | Static, balanced | Dynamic, imbalanced |
| Load balancing | 不需要 | Adaptive, history-based |

### 6.2 vs DOGS (Chen & Lee, 2024)

DOGS 用 ADMM 做 distributed optimization，每 100 iterations average boundary Gaussians。允许 asynchronous training，但 convergence rate 可能受影响。Grendel 保持原版 3DGS 算法，convergence 特性不变。

### 6.3 vs RetinaGS (Li et al., 2024a)

RetinaGS 每个 GPU 用 local Gaussian partition 渲染整张 image，然后 merge 输出。**Redundant computation**——很多 GPU 渲染了 opacity saturation 之外的 pixels。Grendel 通过 pixel partition 避免了这个问题。

### 6.4 vs VastGaussian / CityGaussian / Hierarchical Gaussian

这些都是 divide-and-conquer 算法层面创新，把大 scene 切成小 region 分别训练。Grendel 是 system-level parallelization，**complementary**——这些方法的 coarse training step 可以用 Grendel 加速。

### 6.5 vs Mega-NeRF (Turki et al., 2022) / NeRF-XL (Li et al., 2024b)

这些是 NeRF 的 distributed 方法，由于 NeRF 是 dense NN computation，不直接适用于 3DGS 的 sparse irregular pattern。

### 6.6 vs LLM Large Batch Training

- SGD: linear LR scaling (Goyal et al., 2017) + warmup
- Adam: sqrt LR scaling (Malladi et al., 2022; Granziol et al., 2022)
- Layer-wise: LARS/LAMB (You et al., 2020; Ginsburg et al., 2018)

Grendel 的创新在于：**同时** sqrt LR scaling + exponential momentum scaling，且不依赖 neural network 的 layer-wise structure，专门为 3DGS 的 gradient sparsity 设计。

---

## 7. 局限与可能的延伸

### 7.1 局限

1. **Sparse all-to-all 依赖 spatial locality**：如果未来 3DGS 变体用大 radius Gaussians（比如 global illumination Gaussian），locality 假设失效，通信量爆炸。
2. **NCCL all-to-all 的 uniform volume 偏好**：限制了 redistribution 策略，custom communication primitive 可能更优。
3. **Densification 仍是 local 决策**：如果某 GPU 的 region 训练完成后不再 densify，而其他 GPU 的 region 持续 densify，会导致严重 imbalance。Periodic redistribution 缓解但不根治。
4. **Batch size 32 上限**：Figure 6 显示 independent gradients hypothesis 在 b=32 附近开始 plateau，更大 batch 可能需要新 scaling rule。
5. **No pipeline parallelism**：Grendel 只用 data parallel + Gaussian/Pixel sharding，没有探索 pipeline（像 GPipe/Megatron）。但 3DGS 的 iteration 内 stages 有 dependency，pipeline 收益有限。

### 7.2 可能的延伸

1. **Hierarchical 3DGS + Grendel**：Hierarchical Gaussian (Kerbl et al., 2024) 的 coarse-to-fine 训练可以用 Grendel 加速 coarse stage。
2. **Differentiable rendering 的其他形式**：Mip-Splatting、2DGS、4DGS 都可以套用 Grendel 的 mixed parallelism 框架。
3. **Heterogeneous cluster**：当前假设 uniform GPU，未来可以扩展到 heterogeneous cluster 的 load balancing。
4. **Overlap communication and computation**：sparse all-to-all 与 Gaussian transformation 的 overlap 可以进一步隐藏通信。
5. **Gradient compression**：既然 gradients sparse，可以结合 gradient compression (Top-K, PowerSGD) 进一步减少通信。
6. **Online distillation for large batch**：当 batch size 超过 independent gradients hypothesis 的有效范围，可以用 online distillation 缓解 generalization gap。

---

## 8. Reference Links

- **Paper**: https://arxiv.org/abs/2406.18533 (Grendel-GS)
- **Code**: https://github.com/nyu-systems/Grendel-GS
- **3DGS original**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ (Kerbl et al., 2023)
- **CityGaussian**: https://arxiv.org/abs/2404.16873 (Liu et al., 2024)
- **DOGS**: https://arxiv.org/abs/2405.13943 (Chen & Lee, 2024)
- **RetinaGS**: https://arxiv.org/abs/2406.11836 (Li et al., 2024a)
- **Mega-NeRF (Rubble dataset)**: https://arxiv.org/abs/2112.10703 (Turki et al., 2022)
- **MatrixCity**: http://city-dataset.com/ (Li et al., 2023)
- **Mip-NeRF 360**: https://jonbarron.info/mipnerf360/ (Barron et al., 2022)
- **Tanks & Temples**: https://tanksandtemples.org/ (Knapitsch et al., 2017)
- **PyTorch FSDP**: https://arxiv.org/abs/2304.11277 (Zhao et al., 2023)
- **ZeRO**: https://arxiv.org/abs/1910.02054 (Rajbhandari et al., 2020)
- **Megatron-LM**: https://arxiv.org/abs/1909.08053 (Shoeybi et al., 2020)
- **SGD Linear Scaling**: https://arxiv.org/abs/1706.02677 (Goyal et al., 2017)
- **Adam SDE Scaling**: https://arxiv.org/abs/2206.01729 (Malladi et al., 2022)
- **EMA Scaling**: https://openreview.net/forum?id=DkeeXVdQyu (Busbridge et al., 2023)
- **LARS/LAMB**: https://arxiv.org/abs/1904.00962 (You et al., 2020)
- **Hierarchical 3DGS**: https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/ (Kerbl et al., 2024)
- **OctreeGS**: https://arxiv.org/abs/2403.17898 (Ren et al., 2024)
- **VastGaussian**: https://arxiv.org/abs/2402.17427 (Lin et al., 2024)
- **Scaffold-GS**: https://arxiv.org/abs/2312.00109 (Lu et al., 2024)
- **Perlmutter cluster**: https://docs.nersc.gov/systems/perlmutter/architecture/

---

## 9. 最终 Intuition 总结

Grendel 的核心 insight 是 **3DGS 不是 neural network，不能套用 DNN 分布训练范式**。它的三个本质特征（mixed parallelism、spatial locality、dynamic workload）决定了需要全新的 distributed system design：

1. **Mixed parallelism** 要求不同 stage 用不同 partition axis，stage 之间需要 data shuffle。
2. **Spatial locality** 让 sparse all-to-all 成为可能，通信量正比于实际 intersect 的 Gaussians 而非总数。
3. **Dynamic workload** 要求 adaptive load balancing，用历史 runtime 估计当前 cost。

Batched training 的 hyperparameter scaling 则是 optimization dynamics 层面的贡献。Independent gradients hypothesis 让 Adam 的 sqrt LR scaling + exponential momentum scaling 成为自然选择。Empirical evidence (Figure 6, 12) 验证了 hypothesis 和 scaling rule 的有效性。

最 striking 的实验结果是：**单 GPU 上 3DGS 远未达到 quality saturation**——从 11M 到 40M Gaussians，PSNR +1 dB，这在大 scene 高 resolution 场景下意义重大。Grendel 让这种 scaling 成为可能。

如果让我（Karpathy 视角）评论：这篇 paper 是 system + optimization theory 的优雅结合。System 部分是 classic distributed systems design，抓住 workload pattern 的本质特征做针对性优化；Optimization 部分是 classic optimization dynamics 分析，从 independent gradients hypothesis 推导 scaling rule 并 empirical validation。两者结合，让 3DGS 从 single-GPU toy 走向 multi-GPU production-grade。它也提示我们：**不同 model class（NN vs. explicit representation）需要不同的 distributed training methodology**，不能简单复用。
