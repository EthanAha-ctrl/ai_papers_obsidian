---
source_pdf: FlashSplat 2D to 3D Gaussian Splatting Segmentation Solved Optimally.pdf
paper_sha256: bbb4bb832c75cb46b3026f4155201d419bb1bb94302912d9e701780a295afa9c
processed_at: '2026-08-04T08:56:40-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 FlashSplat

---

## 一句话概括

**别人用 gradient descent 硬训几万轮把 2D mask 升到 3D，这篇 paper 发现其实可以列个 linear equation 一步算出来，30 秒搞定。**

---

## 问题场景

想象你拍了一个房间的几十张照片，用 3D Gaussian Splatting 把它重建出来——每个 Gaussian 就像一个半透明的小椭球，叠加在一起渲染出照片。现在你用 SAM 给每张照片标了 mask：这张桌子在这，那把椅子在那。你想知道：**这些 2D mask 背后，每个 3D Gaussian 到底属于桌子还是椅子？**

SAGA 和 Gaussian Grouping 的做法是给每个 Gaussian 学一个 feature，用 backprop 训 30,000 steps，让它 feature 跟 2D mask 对齐。慢，还容易卡在 local optimum。

---

## 关键观察

3D-GS 渲染一个 pixel 的公式是：

$$X = \sum_i x_i \cdot \alpha_i \cdot T_i$$

这里 $x_i$ 是你想渲染的属性，$\alpha_i$ 是这个 Gaussian 在该 pixel 上的 alpha 值，$T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$ 是前面所有 Gaussian 没挡住光的比例。

**一旦 3D scene 重建好了，$\alpha_i$ 和 $T_i$ 就跟 label 无关，是固定的常数。**

你想渲染的不是 color，而是 label $P_i \in \{0, 1\}$。所以渲染出来的 mask value 是：

$$\text{rendered mask} = \sum_i P_i \cdot \alpha_i \cdot T_i$$

这就是一个关于 $\{P_i\}$ 的 **linear combination**。你的 loss 是 rendered mask 跟 ground truth 2D mask 的 L1 距离。linear loss + linear constraint = **linear programming**。

---

## 求解过程

目标函数展开后（用 indicator function 拆开绝对值）变成：

$$\mathcal{F} = C + \sum_i P_i \cdot (A_0^i - A_1^i)$$

- $A_0^i$ = 这个 Gaussian 在所有 view、所有 background pixel 上的 $\alpha_i T_i$ 贡献总和
- $A_1^i$ = 这个 Gaussian 在所有 view、所有 foreground pixel 上的 $\alpha_i T_i$ 贡献总和
- $C$ 是常数

要 minimize $\mathcal{F}$，每个 $P_i$ 独立决策：
- 如果 $A_0^i > A_1^i$，说明这个 Gaussian 主要被看到在 background → 标 0
- 如果 $A_1^i > A_0^i$，说明它主要被看到在 foreground → 标 1

就是一个 **weighted majority vote**。没有 iteration，没有 backprop，一个 arg max 完事。

---

## 物理直觉

$A_1^i$ 问的是："这个 Gaussian 在所有照片的 foreground 区域里，被像素们'看见'了多少？" $A_0^i$ 问的是同样的问题对 background。谁的投票权重大就归谁。

这个投票权重不是简单数次数，而是 $\alpha_i T_i$——考虑了这个 Gaussian 的 opacity（不透明度）、它在 pixel 上的 footprint（投影面积）、以及前面有没有被挡住（transmittance）。一个半透明的 Gaussian 对 pixel 的贡献很小，它的"票"就很轻。

---

## 为什么要加 background bias γ

SAM 的 mask 有噪声——经常把背景误判成前景。直接 majority vote 会把这些 noise 带进 3D segmentation，让物体边缘有很多杂散 Gaussian。

解决方法很朴素：给 background 的权重加一个 bias $\gamma$：

$$P_i = \arg\max\{ \bar{A}_0 + \gamma, \bar{A}_1 \}$$

（先 L1 normalize 再加 bias）

- $\gamma > 0$：偏向标 background，压制 foreground 噪声
- $\gamma < 0$：偏向标 foreground，用于 object removal 时让背景更干净

因为 $\{A_e\}$ 已经算好了，调整 $\gamma$ 就是改一个数再 arg max，**interactive 的，毫秒级响应**。用户可以拖 slider 实时看 segmentation 变化。

---

## Multi-instance 怎么办

如果场景里有桌子、椅子、灯泡好几个物体，能不能直接用 $P_i \in \{0,1,2,...,E-1\}$ 做 multi-label arg max？

不行。因为 label 之间是 exchangeable 的——你把所有 0 和 1 互换，loss 不变。而且 3D-GS 的 Gaussian 不互斥，一个 Gaussian 可能同时属于桌子和椅子的边界区域。

作者的做法是 **one-vs-rest**：对每个物体 $t$，把其他所有物体当作 background，做一次 binary segmentation：

$$P_i^t = \arg\max\{ A_{others} + \gamma, A_t \}$$

$A_{others} = \sum_{e \neq t} A_e$。只需要算一次 $\{A_e\}$，然后对每个 instance 跑一遍 arg max。当 $\gamma < 0$ 时，不同 instance 的 Gaussian subset 可以 overlap，这反而符合 3D-GS 的真实物理性质。

---

## 新视角 mask 渲染

有了 3D label，怎么渲染一个没见过的视角的 mask？

**Binary**：只渲染 $P_i = 1$ 的 Gaussian，累加 alpha 值 $\rho_{jk}$，超过 threshold $\tau$ 就是前景：

$$\hat{M}_{jk} = \mathbb{1}[\rho_{jk} > \tau]$$

**Scene**：多个 instance 都超过 threshold 怎么办？用 depth guidance——选 depth 最小（离相机最近）的那个 instance。

Paper 也承认 3D-GS 没有严格的几何监督，depth 不够精确，scene rendering 边界仍然可能模糊。

---

## 实验数据

### 精度（NVOS dataset）

| Method | mIoU | mAcc |
|---|---|---|
| NVOS (NeRF-based) | 39.4 | 73.6 |
| SAGA (3D-GS + feature field) | 90.9 | 98.3 |
| **FlashSplat** | **91.8** | **98.6** |

比 SAGA 略好，但 SAGA 要训 30,000 iterations。

### 速度（Figurines scene, A6000）

| Method | 优化时间 | 优化步数 | 单次分割 | 显存峰值 |
|---|---|---|---|---|
| SAGA | 18 min | 30,000 | 0.5 s | 15 GB |
| Gaussian Grouping | 37 min | 30,000 | 0.3 s | 34 GB |
| **FlashSplat** | **26 s** | **1** | **0.4 ms** | **8 GB** |

**50 倍加速，一半显存**。

### Few-shot 能力

只要 1/8 的 view 有 mask 就能产出像样的 3D segmentation。因为不需要 iterative optimization 来拟合 feature field，10% 监督就够。

---

## 为什么这个工作有意思

从你 Karpathy 一直强调的 first principles 角度看，这篇 paper 的价值在于：**它停下来问了"这个问题本质是什么结构"，而不是上来就堆 network**。

3D-GS 的 alpha blending 是 linear 的，这一直都在那里——SAGA、Gaussian Grouping 的人没看到，或者看到了没敢相信能 closed-form 求解。FlashSplat 把这个线性性 exploit 到极致：linear objective + linear constraint + 独立 per-Gaussian 决策 = closed-form。

这跟 Plenoxels 之于 NeRF 的关系类似——NeRF 不一定需要 MLP，3D-GS segmentation 不一定需要 gradient descent。

而且这个 paradigm 不局限于 segmentation。任何 3D-GS 上"从 2D 监督 lift 到 3D 属性"的任务，只要渲染对属性是线性的，都可以这么搞。比如 depth supervision、semantic feature distillation、material estimation，理论上都能 reformulate 成 LP。

---

## 局限

1. **需要 traverse 所有 mask pixel**，超大规模场景可能慢。
2. **Depth guidance 不够精确**，3D-GS 本身几何不严格。
3. **依赖 alpha blending 的线性性**——换成其他 rendering 方式不一定成立。
4. **Mask association 还是依赖外部 model**（SAM + video tracker），不是 end-to-end。

---

## Code 结构

实现非常简洁：
1. CUDA kernel 里跑 rasterization，但只算 $\alpha_i T_i$，不算 color。对每个 pixel，根据它的 mask label $e$，把 $\alpha_i T_i$ 累加到 $A_e[i]$ 里。用 atomic operation 处理并发。
2. 算完 $\{A_e\}$ 后（~26s），arg max 就 1ms。
3. $\gamma$ 调整不需要重新 rasterize，改个数再 arg max 就行。

PyTorch 实现在 supplementary Listing 1.1，核心逻辑不到 20 行。

---

## 最终 takeaway

**当你的 loss function 关于优化变量是线性的，且变量之间独立，那就 closed-form 求解。别用 gradient descent 硬训。**

听起来 trivial，但 SAGA 和 Gaussian Grouping 这两个 baseline 都是 2023 年的工作，都用了 deep learning 的思路。FlashSplat 2024 年才发现这个 trivial 解。这本身就是个 lesson：**先看 problem structure，再选 method**。

Paper link: [https://arxiv.org/abs/2406.12327](https://arxiv.org/abs/2406.12327)
Code: [https://github.com/florinshen/FlashSplat](https://github.com/florinshen/FlashSplat)

---

## 相关联想

- **Plenoxels** [[Fridovich-Keil et al. CVPR 2022]](https://arxiv.org/abs/2111.11288): 把 NeRF 的 MLP 换成 sparse voxel grid + trilinear interpolation，发现完全不需要 neural network 也能做 view synthesis。跟 FlashSplat 异曲同工——都是发现"deep learning 其实是多余的"。

- **Gaussian Splatting SLAM** [[Matsuki et al. 2024]](https://arxiv.org/abs/2311.13000): 用 explicit Gaussians 做 camera tracking，也能避开隐式 representation 的 optimization 难题。

- **GraphCut Segmentation** [[Boykov & Jolly ICCV 2001]](https://arxiv.org/abs/1105.5530): 经典 graph cut 方法做 image segmentation，能量函数有 pairwise term 需要 min-cut 求解。FlashSplat 之所以能 closed-form 是因为没有 pairwise term，Gaussian 之间不直接交互——这跟 alpha blending 的物理特性有关。

- **Alpha Blending as Linear Operator**: 这个观察其实在 volume rendering 文献里早就隐含，比如 [[Max 1995]](https://www.cs.duke.edu/courses/cps296.2/spring03/papers/max95.pdf) 的 optical models 论文。但在 3D-GS segmentation context 下被第一次 exploit 成 algorithmic advantage。

- **LP relaxation for discrete optimization**: 传统 ILP 通常 NP-hard，需要 LP relaxation + rounding。这里因为每个 $P_i$ 独立决策（objective 是 separable 的），ILP 直接有 closed-form integer solution，非常罕见。

---

# FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally 深度解析

Paper link: https://arxiv.org/abs/2406.12327
Code: https://github.com/florinshen/FlashSplat

---

## 1. 核心问题与动机

3D Gaussian Splatting (3D-GS) 通过 explicit 的 3D Gaussians 集合 $\{G_i\}$ 重建场景，每个 Gaussian 参数化为 $G_i = \{m_i, q_i, s_i, o_i, c_i\}$：
- $m_i \in \mathbb{R}^3$: center position
- $q_i \in \mathbb{R}^4$: rotation quaternion
- $s_i \in \mathbb{R}^3$: scale
- $o_i \in \mathbb{R}$: opacity
- $c_i \in \mathbb{R}^{48}$: 3-order spherical harmonics for view-dependent color

问题定义：给定重建好的 3D-GS 场景与多视角的 2D binary mask $\{M^v\}$，为每个 3D Gaussian $G_i$ 分配 label $P_i \in \{0,1\}$。

先前工作 SAGA [[3]](https://arxiv.org/abs/2312.00860) 与 Gaussian Grouping [[54]](https://arxiv.org/abs/2312.00732) 都采用 iterative gradient descent 训练额外的 feature field，需要 30,000 iterations 才能收敛，并且容易陷入 local optimum。

---

## 2. 关键洞察：Rendering 关于 Label 的线性性

### 2.1 Alpha Blending 回顾

3D-GS 的 tile-based rasterization 对每个 pixel 的属性 $X$ 通过 alpha composition 混合：

$$X = \sum_{i \in \{G_i\}_B} x_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j) = \sum_{i} x_i \alpha_i T_i \tag{1}$$

各变量含义：
- $x_i$: pixel-space 的 property（color、depth，或本 paper 中的 label $P_i$）
- $\alpha_i$: 当 pixel 落在 projected 2D Gaussian $(m_i^{2D}, \Sigma_i^{2D})$ 上时的 alpha 值，等于 $o_i$ 乘以 pixel 在 2D Gaussian 分布中的概率
- $T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$: transmittance，前面 $i-1$ 个 Gaussian 未吸收的光的比例
- $B$: 当前 tile 共享的 Gaussian 子集

### 2.2 线性化的关键

**一旦 $\{G_i\}$ 重建完成，所有 $\alpha_i$ 与 $T_i$ 都变成 constants**。所以渲染函数对 blending property $x_i$（即 label $P_i$）来说是 **purely linear**：

$$\mathcal{R}(\{G_i\}, \{P_i\}) = \sum_i P_i \cdot \alpha_i T_i$$

这个 observation 极其优雅——它把一个看似需要 iterative optimization 的问题降维成了 linear programming。

---

## 3. ILP Formulation 与闭式解

### 3.1 目标函数

最小化渲染 mask 与给定 2D mask 的 L1 误差：

$$\min_{\{P_i\}} \mathcal{F} = \sum_{v \in L} \sum_{M_{jk}^v \in M^v} \left| \sum_i P_i \alpha_i T_i - M_{jk}^v \right| \tag{2}$$

subject to $P_i \in \{0, 1\}$

### 3.2 Lemma 1: Alpha Composition 的有界性

由于 alpha composition 中 light 只能被吸收不能创造：

$$0 \leq \sum_i P_i \alpha_i T_i \leq \sum_i \alpha_i T_i \leq 1 \tag{3}$$

即 accumulated contribution 的上界为 1（initial light intensity normalized 为 1）。这个 lemma 保证可以去除绝对值符号，将目标函数改写。

### 3.3 关键推导：从绝对值到线性表达式

利用 $0 \leq \sum P_i \alpha_i T_i \leq 1$ 与 indicator function $\mathbb{I}(M_{jk}^v, n)$（当 $M_{jk}^v = n$ 时为 1，否则为 0），将式 (2) 展开为：

$$\mathcal{F} = C + \sum_i P_i (A_0^i - A_1^i) \tag{6}$$

其中：
- $C = \sum_{v,j,k} M_{jk}^v$: constant
- $A_0^i = \sum_{v,j,k} \alpha_i T_i \cdot \mathbb{I}(M_{jk}^v, 0)$: Gaussian $G_i$ 对所有 background pixel 的总贡献（加权累计）
- $A_1^i = \sum_{v,j,k} \alpha_i T_i \cdot \mathbb{I}(M_{jk}^v, 1)$: Gaussian $G_i$ 对所有 foreground pixel 的总贡献

**Intuition**：$A_0^i$ 度量 Gaussian $i$ 在所有 view 中"贡献给 background 的总透光量"，$A_1^i$ 度量它"贡献给 foreground 的总透光量"。

### 3.4 Weighted Majority Vote 解

最小化 $\mathcal{F} = C + \sum_i P_i(A_0^i - A_1^i)$ 关于每个 $P_i$ 是独立的：

$$P_i = \arg\max_n A_n, \quad n \in \{0, 1\} \tag{7}$$

- 若 $A_0^i > A_1^i$（即该 Gaussian 更多贡献给 background pixel），则 $P_i = 0$
- 若 $A_1^i > A_0^i$，则 $P_i = 1$

这就是 **globally optimal closed-form solution**，不需要任何迭代！

---

## 4. Background Bias: 抗噪声机制

### 4.1 问题

实际 2D mask 由 SAM [[20]](https://arxiv.org/abs/2304.02643) 等模型预测，存在噪声——比如背景区域被误判为 foreground。直接 majority vote 会把 noise-induced 的 background Gaussian 错误标为 foreground，导致 3D 物体边缘有尖锐噪点。

### 4.2 Softened Assignment

引入 L1 normalization 与 bias：

$$\bar{A}_e = \frac{A_e}{\sum_t A_t}, \quad \hat{A}_0 = \bar{A}_0 + \gamma$$

$$P_i = \arg\max_n \{\hat{A}_0, \bar{A}_1\}$$

$\gamma \in [-1, 1]$ 的作用：
- $\gamma > 0$: 偏向 background，压制 foreground noise（图 2b）
- $\gamma < 0$: 偏向 foreground，清理 background holes（图 2c，用于 object removal 时保留干净前景）
- $\gamma = 0$: 标准 majority vote

Table 3 ablation（truck 场景）显示 $\gamma = 0.4$ 时 mIoU 最优 94.2%，说明 SAM mask 存在系统性 background→foreground 误判噪声。

---

## 5. Binary → Scene Segmentation 扩展

### 5.1 难点

3D Gaussians 具有非互斥性——一个 Gaussian 可能同时贡献给多个 instance。Fig. 1 中 pixel $u_1, u_2$ 属于不同 instance 但共享同一 Gaussian。若直接用 $P_i \in \{0,1,...,E-1\}$（E 为 instance 总数），label 之间可交换（exchangeable），导致 ILP 无法得到 global optimum。

### 5.2 解决方案：One-vs-Rest Binary Decomposition

将 multi-instance segmentation 分解为 E 个 binary segmentation：

$$P_i = \arg\max_n A_n, \quad n \in \{0, t\} \tag{8}$$

$$A_t = \sum_{v,j,k} \alpha_i T_i \mathbb{I}(M_{jk}^v, t), \quad A_0 = A_{others} = \sum_{e \neq t} \sum_{v,j,k} \alpha_i T_i \mathbb{I}(M_{jk}^v, e)$$

只需累积一次 $\{A_e\}$，对每个 instance $t$ 做 arg max 即可。Listing 1.1 给出 PyTorch 实现：

```python
def multi_instance_opt(all_contrib, gamma=0.):
    all_contrib_sum = all_contrib.sum(dim=0)
    all_obj_labels = torch.zeros_like(all_contrib)
    for obj_idx, obj_contrib in enumerate(all_contrib):
        other_contrib = all_contrib_sum - obj_contrib
        obj_contrib = torch.stack([other_contrib, obj_contrib])
        obj_contrib = F.normalize(obj_contrib, dim=0, p=1)
        obj_contrib[0, :] += gamma  # bias "others"
        obj_label = torch.argmax(obj_contrib, dim=0)
        all_obj_labels[obj_idx] = obj_label
    return all_obj_labels
```

**注意**：当 $\gamma < 0$，不同 instance 的 Gaussian 子集可能 overlap，反映了 3D-GS 内在的非互斥性。

---

## 6. Depth-guided Novel View Mask Rendering

### 6.1 Binary 情形

只渲染 foreground Gaussians（$P_i = 1$），得到 per-pixel accumulated alpha $\rho_{jk}$，再量化：

$$\hat{M}_{jk}^v = \mathbb{Q}(\rho_{jk}, \tau), \quad \mathbb{Q}(\rho, \tau) = \begin{cases} 1, & \rho > \tau \\ 0, & \rho \leq \tau \end{cases}$$

$\tau$ 为预设阈值。Fig. 10 显示：没有 quantization，因 Gaussian 半透明，背景 Gaussians 也会影响 alpha blending，产生大量 holes。

### 6.2 Scene 情形：Ambiguity 处理

由于 instance 之间 Gaussian overlap，渲染后同一 pixel 可能多个 instance 都满足 $\mathbb{Q}(\rho_{jk}^e, \tau) = 1$。引入 depth guidance：

$$\hat{M}_{jk} = \arg\min_{e: \mathbb{Q}(\rho_{jk}^e, \tau)=1} D_{jk}^e$$

选择 depth 最小（即最靠近 camera）的 instance 作为最终 label。Limitation section 提到 3D-GS 缺乏 explicit geometry supervision，depth 不够精确，scene mask rendering 仍可能 ambiguous。

---

## 7. 实验数据深度分析

### 7.1 NVOS 量化对比 (Table 1)

| Method | mIoU (%) ↑ | mAcc (%) ↑ |
|---|---|---|
| NVOS [[37]](https://arxiv.org/abs/2204.13266) | 39.4 | 73.6 |
| ISRF [[12]](https://arxiv.org/abs/2212.13545) | 70.1 | 92.0 |
| SGISRF [[45]](https://arxiv.org/abs/2305.16900) | 83.8 | 96.4 |
| SA3D [[4]](https://arxiv.org/abs/2304.12301) | 90.3 | 98.2 |
| SAGA [[3]](https://arxiv.org/abs/2312.00860) | 90.9 | 98.3 |
| **FlashSplat** | **91.8** | **98.6** |

NVOS dataset [[37]](https://arxiv.org/abs/2204.13266) 基于 LLFF [[29]](https://arxiv.org/abs/1905.00824)，8 个 forward-facing scene，提供 reference + target view mask。Pipeline：reference mask 上采样 point prompts → 传播到其他 views → SAM 生成 mask → FlashSplat binary segmentation → target view 渲染 mask（$\tau = 0.1$）。

**FlashSplat 比 SAGA 略好（91.8 vs 90.9 mIoU），更重要的是无需训练 feature field**。

### 7.2 计算成本 (Table 2)

| Method | Extra Time | Optimization Steps | Seg Time | Peak Memory |
|---|---|---|---|---|
| SAGA | 18 min | 30,000 | 0.5 s | 15G |
| Gaussian Grouping [[54]](https://arxiv.org/abs/2312.00732) | 37 min | 30,000 | 0.3 s | 34G |
| **FlashSplat** | **26 s** | **1** | **0.4 ms** | **8G** |

**~50× speedup**，**~50% memory reduction**。关键原因：
- 只需一次 rasterization 累积 $\{A_e\}$（26s），无需 gradient descent
- Arg max 在 1ms 内完成
- $\gamma$ 调整无需重新计算 $A_e$，实现 interactive

### 7.3 Few-shot Segmentation 能力 (Fig. 7)

由于不需要 iterative optimization，FlashSplat 在 1/8 视角 mask 下仍能产生 decent 分割结果，对应 paper abstract 中提到的 "10% of masked 2D views"。这是 gradient-based method 难以做到的——它们需要充分监督才能训练 feature field。

---

## 8. Intuition Building：为什么这个方法能 work

### 8.1 线性性的几何本质

3D-GS 的 alpha blending 是一个 weighted sum，权重由场景几何（depth order、projected footprint、opacity）决定。这些几何信息一旦确定，与 label 无关——这正是线性性的来源。

对比 NeRF-based 方法 [[37, 12, 45, 4]](https://arxiv.org/abs/2204.13266)，它们用隐式 MLP 编码颜色与语义，需要端到端 backprop，无法把语义部分剥离出来做 linear solve。

### 8.2 Majority Vote 的物理直觉

$A_0^i, A_1^i$ 本质上衡量了 Gaussian $i$ 在所有 view 的所有 pixel 上的"投票权重"——这个 Gaussian 在多大程度上"被看到属于 background / foreground"。Weight 不是 uniform 的，而是 $\alpha_i T_i$，即该 Gaussian 对该 pixel 的实际视觉贡献（按 visibility 与 footprint 加权）。

这比 SAGS [[16]](https://arxiv.org/abs/2401.17857) 的简单 projection-based assignment（只看 Gaussian center 是否落在 mask 内）更合理，因为它考虑了 Gaussian 的 spatial extent 与 occlusion。

### 8.3 与 Graph Cut / MRF 的对比

传统 multi-label segmentation 常用 MRF / graph cut 求解 pairwise energy，但 3D-GS 的问题结构允许 decompose 到 per-Gaussian independent decision——这是 alpha blending 可加性的恩赐。没有 pairwise term 需要优化（Gaussian 之间不直接交互），所以 closed-form 成立。

---

## 9. Limitations 与未来方向

Paper Section 9 列出几个 limitation：

1. **Scalability**：需要 traverse 所有 mask pixel，对超大规模场景可能 cost 高。
2. **Depth guidance ambiguity**：3D-GS 缺乏 explicit geometry supervision，scene mask rendering 在 occlusion boundary 可能模糊。
3. **未利用几何约束**：与 SDF / NeuS 等相比，3D-GS 的 Gaussian 不严格对应 surface，导致分割边缘可能 fuzz。

潜在改进方向：
- 结合 [[surface Gaussians]](https://arxiv.org/abs/2311.12983) 等方法引入几何正则
- 用 hierarchical data structure 加速 $\{A_e\}$ 累积
- 引入 SAM 视频版 [[Segment-and-Track Anything]](https://arxiv.org/abs/2305.06558) 提升 mask association 质量

---

## 10. 对 3D-GS 生态的影响

FlashSplat 揭示了一个更广泛的方法论：**3D-GS 的 explicit representation 使得许多看似需要 deep learning 的任务可以 reformulate 为 closed-form optimization**。这与近期 trend 一致：

- [[GS-based SLAM]](https://arxiv.org/abs/2311.13000) 利用 explicit Gaussians 做 tracking
- [[GS-based editing]](https://arxiv.org/abs/2311.16684) 直接操作 Gaussian 属性
- [[GS-based physics simulation]](https://arxiv.org/abs/2311.17680) 把 Gaussians 当作 particles

这种 explicit-first paradigm 与 NeRF-based implicit methods 的 contrast 越来越清晰：前者 trade off 表示紧凑性换取可解释性、可编辑性、与可解性。

---

## 11. 个人思考

FlashSplat 的 elegance 在于它发现了一个被前人忽视的简单事实：**alpha blending 在 label space 是线性的**。这种 discovery 类似于 [[Plenoxels]](https://arxiv.org/abs/2111.11288) 发现 NeRF 不需要 MLP 也能 work——表面看是工程 trick，实则揭示 problem structure 的本质。

从 Karpathy 你一向推崇的 "first principles thinking" 角度看，这篇 paper 是一个范例：不要急着套 deep learning，先检查目标函数的结构是否有可利用的性质。Paper 的整个推导从 Eq 2 到 Eq 7 只用了 elementary math，却达成了 prior work 30,000 iterations 都未必达到的 global optimum。

值得注意的是，这个方法的成功也依赖于 3D-GS representation 的特殊性质。如果换成 voxel grid 或 point cloud 的渲染方式（不基于 alpha composition），这种 linear decomposition 不一定成立。所以它既是 algorithm 的胜利，也是 representation 的胜利。

---

## 关键 References

- [3D Gaussian Splatting (Kerbl et al. 2023)](https://arxiv.org/abs/2308.14737)
- [SAGA: Segment Any 3D Gaussians](https://arxiv.org/abs/2312.00860)
- [Gaussian Grouping](https://arxiv.org/abs/2312.00732)
- [SAGS: Semantic Anything in 3D Gaussians](https://arxiv.org/abs/2401.17857)
- [SAM: Segment Anything](https://arxiv.org/abs/2304.02643)
- [LERF: Language Embedded Radiance Fields](https://arxiv.org/abs/2303.09553)
- [MIP-NeRF 360](https://arxiv.org/abs/2111.12077)
- [NVOS: Neural Volumetric Object Selection](https://arxiv.org/abs/2204.13266)
- [Tanks and Temples](https://www.tanksandtemples.org/)
- [LLFF](https://arxiv.org/abs/1905.00824)
- [LaMa Inpainting](https://arxiv.org/abs/2109.07161)
- [Grounding DINO](https://arxiv.org/abs/2303.05499)
- [Tracking Anything (TAM)](https://arxiv.org/abs/2305.06558)
