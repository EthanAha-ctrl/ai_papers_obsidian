# SAM2 微调全流程深度解析

这篇文章系统性地介绍了如何在仅60行代码内对 Meta 的 **SAM2 (Segment Anything 2)** 进行微调，使其从"通用分割器"适配到特定领域（如实验室场景中的材料和液体分割）。下面我从第一性原理出发，逐层拆解每个技术细节。

---

## 1. 为什么需要微调 SAM2？——从第一性原理出发

SAM2 的核心承诺是 **"分割一切"**，其训练数据规模惊人：**1100万张图像 + 110亿个掩码**。但从信息论角度看：

$$H(\text{通用数据}) \gg H(\text{特定领域数据})$$

SAM2 的模型容量 $C$ 是有限的，训练时参数会优先拟合高频出现的通用物体分布 $P_{\text{general}}(x)$，而对低频的特定领域分布 $P_{\text{domain}}(x)$ 覆盖不足。因此：

$$P_{\text{SAM2}}(y|x) \approx P_{\text{general}}(y|x) \neq P_{\text{domain}}(y|x)$$

微调的本质是 **在保持已学通用表征的前提下，将模型分布从 $P_{\text{general}}$ 向 $P_{\text{domain}}$ 偏移**，这比从随机初始化开始学习所需的标注数据量少几个数量级（迁移学习的信息论优势）。

> 📖 参考：[Transfer Learning Survey (Pan & Yang, 2010)](https://ieeexplore.ieee.org/abstract/10.1109/tkde.2009.191)

---

## 2. SAM2 架构深度解析

SAM2 由三个核心组件构成，形成一条清晰的信息流：

```
输入图像 ──→ [图像编码器] ──→ image_embeddings
                                        ↓
输入点  ──→ [提示编码器]  ──→ sparse_embeddings + dense_embeddings
                                        ↓
                              [掩码解码器] ──→ 分割掩码 + 分数
```

### 2.1 图像编码器

- **架构**：基于 **ViT (Vision Transformer)** 的 Hiera 变体
- **作用**：将 $H \times W \times 3$ 的输入图像编码为紧凑的 embedding tensor
- **特点**：这是网络中最大的组件，参数量占主导
- **微调策略**：**冻结不训练**——因为：
  1. 已经提供了良好的通用视觉表征
  2. 训练成本极高（大模型的梯度计算和内存开销巨大）
  3. 容易过拟合（在小数据集上微调大模型的风险）

Hiera 架构的核心思想是层级化的多尺度特征提取，其数学表达为：

$$\mathbf{f}_l = \text{HieraBlock}_l(\mathbf{f}_{l-1}), \quad l = 1, 2, \ldots, L$$

其中 $\mathbf{f}_l \in \mathbb{R}^{h_l \times w_l \times d_l}$ 是第 $l$ 层的特征图，$h_l, w_l$ 随层级递减，$d_l$ 递增。

### 2.2 提示编码器

- **输入**：点坐标 $(x, y)$ + 标签 $l \in \{0, 1\}$（前景/背景）
- **处理流程**：
  1. 将坐标归一化到 $[0, 1]$：$\hat{x} = x / W, \quad \hat{y} = y / H$
  2. 通过位置编码映射为高维向量
  3. 加上标签的 learned embedding

$$\mathbf{e}_{\text{point}} = \text{PE}(\hat{x}, \hat{y}) + \mathbf{e}_{\text{label}}$$

其中 $\text{PE}(\cdot)$ 是 positional encoding，$\mathbf{e}_{\text{label}} \in \mathbb{R}^d$ 是可学习的标签嵌入。

- **输出**：
  - `sparse_embeddings`：点对应的稀疏嵌入（形状 $B \times N_{\text{points}} \times d$）
  - `dense_embeddings`：无点输入时的默认密集嵌入

- **微调策略**：**可训练**——轻量级组件，训练成本低

### 2.3 掩码解码器

- **架构**：基于 **Transformer Decoder** 的交叉注意力机制
- **输入**：
  - `image_embeddings`：来自图像编码器
  - `sparse_prompt_embeddings`：来自提示编码器
  - `dense_prompt_embeddings`：来自提示编码器
  - `image_pe`：图像位置编码
  - `high_res_features`：来自图像编码器的高分辨率特征（SAM2 新增！）

- **输出**：
  - `low_res_masks`：低分辨率掩码 logits（通常为 $256 \times 256$）
  - `prd_scores`：每个掩码的质量分数

掩码解码器的核心计算可表示为：

$$\mathbf{q}_0 = \text{MLP}(\mathbf{e}_{\text{sparse}})$$
$$\mathbf{q}_l = \text{CrossAttn}(\mathbf{q}_{l-1}, \mathbf{f}_{\text{image}}) + \text{SelfAttn}(\mathbf{q}_{l-1})$$
$$\mathbf{m}_{\text{logits}} = \text{ConvTranspose}(\text{MLP}(\mathbf{q}_L) \oplus \mathbf{f}_{\text{high\_res}})$$

其中 $\oplus$ 表示特征拼接/融合，$\mathbf{f}_{\text{high\_res}}$ 是 SAM2 特有的高分辨率特征注入。

> 📖 参考：[SAM2 Paper (Ravi et al., 2024)](https://ai.meta.com/research/publications/segment-anything-2/)

---

## 3. 数据准备——LabPics1 数据集

### 3.1 数据集概况

LabPics1 专注于 **实验室场景中的材料和液体分割**，这与 SAM2 训练数据中的日常物体分布有显著差异，是验证微调效果的理想场景。

> 📖 数据集链接：[LabPicsV1 on Zenodo](https://zenodo.org/records/3697452)

### 3.2 数据读取器设计

数据读取器的核心任务是生成训练三元组 $(I, \mathcal{M}, \mathcal{P})$：

- $I \in \mathbb{R}^{H \times W \times 3}$：RGB 图像
- $\mathcal{M} = \{M_1, M_2, \ldots, M_K\}$：$K$ 个二值掩码，$M_k \in \{0, 1\}^{H \times W}$
- $\mathcal{P} = \{p_1, p_2, \ldots, p_K\}$：每个掩码内的一个随机点，$p_k = (x_k, y_k)$ 满足 $M_k(y_k, x_k) = 1$

#### 关键技术细节

**① 图像缩放**

```python
r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])
```

缩放因子 $r$ 保证图像最长边不超过 1024，这是 SAM2 输入的要求。数学上：

$$r = \min\left(\frac{1024}{W}, \frac{1024}{H}\right)$$

$$\hat{W} = \lfloor W \cdot r \rfloor, \quad \hat{H} = \lfloor H \cdot r \rfloor$$

**② 标注图缩放——最近邻插值**

```python
ann_map = cv2.resize(ann_map, ..., interpolation=cv2.INTER_NEAREST)
```

这是 **关键细节**！标注图中每个像素值是分割索引 ID，如果使用双线性插值（默认），会在两个不同 ID 之间产生不存在的新值，破坏标注语义。最近邻插值确保：

$$\forall (i,j): \hat{A}(i,j) \in \{A(i',j') | (i',j') \in \text{neighborhood}(i,j)\}$$

**③ 材料和容器标注合并**

```python
mat_map = ann_map[:,:,0]  # 材料通道
ves_map = ann_map[:,:,2]  # 容器通道
mat_map[mat_map==0] = ves_map[mat_map==0] * (mat_map.max()+1)
```

逻辑：对于材料通道中值为 0（未标注）的像素，用容器通道的值填充，但需要偏移以避免 ID 冲突。数学表达：

$$M_{\text{merged}}(i,j) = \begin{cases} M_{\text{mat}}(i,j) & \text{if } M_{\text{mat}}(i,j) > 0 \\ M_{\text{ves}}(i,j) \times (\max(M_{\text{mat}}) + 1) & \text{if } M_{\text{mat}}(i,j) = 0 \end{cases}$$

**④ 随机点采样**

```python
coords = np.argwhere(mask > 0)  # 掩码内所有坐标
yx = np.array(coords[np.random.randint(len(coords))])  # 随机选一个
```

这实现了 **均匀随机采样**：$p_k \sim \text{Uniform}(\{(i,j) | M_k(i,j)=1\})$。每个训练迭代中，每个掩码只提供一个点作为 prompt，这模拟了 SAM 的核心使用模式——单点分割。

---

## 4. 训练配置与策略

### 4.1 可训练参数的选择

```python
predictor.model.sam_mask_decoder.train(True)   # 掩码解码器：可训练
predictor.model.sam_prompt_encoder.train(True)  # 提示编码器：可训练
# 图像编码器：默认不训练（冻结）
```

| 组件 | 参数量级 | 是否训练 | 原因 |
|------|---------|---------|------|
| 图像编码器 | ~数十M | ❌ 冻结 | 已学良好表征，训练成本高 |
| 提示编码器 | ~数百K | ✅ 训练 | 轻量，需要适配领域特定的点分布 |
| 掩码解码器 | ~数M | ✅ 训练 | 核心适配目标，需要学习领域特定的掩码生成 |

### 4.2 优化器选择

```python
optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
```

- **AdamW** 而非 Adam：解耦权重衰减，更好地正则化

$$\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

其中：
- $\hat{m}_t$：一阶矩估计（动量）
- $\hat{v}_t$：二阶矩估计
- $\lambda = 4 \times 10^{-5}$：权重衰减系数
- $\eta = 10^{-5}$：学习率（非常小，因为是微调）

- **学习率 $1 \times 10^{-5}$**：微调经典选择，避免灾难性遗忘
- **权重衰减 $4 \times 10^{-5}$**：轻量级 L2 正则化

### 4.3 混合精度训练 (AMP)

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    ...
```

混合精度的核心思想：前向传播用 **FP16/BF16**（省内存、加速），梯度更新用 **FP32**（保持精度）。

$$\text{loss}_{\text{scaled}} = \text{loss} \times S$$
$$\text{grads}_{\text{FP32}} = \frac{\text{grads}_{\text{FP16}}}{S}$$

其中 $S$ 是动态缩放因子，防止 FP16 下的梯度下溢。

> 📖 参考：[Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740)

---

## 5. 前向传播详解

### 5.1 图像编码

```python
predictor.set_image(image)
```

此步骤运行 Hiera 编码器，将图像编码为：
- `predictor._features["image_embed"][-1]`：低分辨率图像嵌入，形状约 $1 \times 256 \times 64 \times 64$
- `predictor._features["high_res_feats"]`：多级高分辨率特征（SAM2 新增！用于更精细的掩码生成）

### 5.2 Prompt 编码

```python
mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
    input_point, input_label, box=None, mask_logits=None, normalize_coords=True
)
sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
    points=(unnorm_coords, labels), boxes=None, masks=None
)
```

- `_prep_prompts`：对输入点坐标做归一化（normalize_coords=True），将像素坐标映射到 $[0,1]$
- `sam_prompt_encoder`：将归一化点坐标 + 标签编码为稀疏/密集嵌入

### 5.3 掩码解码

```python
low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=True,
    repeat_image=batched_mode,
    high_res_features=high_res_features,
)
```

关键参数解析：

| 参数 | 含义 |
|------|------|
| `image_embeddings` | 图像编码器输出的主特征图 |
| `image_pe` | 图像位置编码 |
| `sparse_prompt_embeddings` | 点的稀疏嵌入 |
| `dense_prompt_embeddings` | 密集提示嵌入（无点输入时的默认） |
| `multimask_output=True` | 每个点输出3个掩码候选（歧义消除机制） |
| `repeat_image=batched_mode` | 多点时是否复制图像特征 |
| `high_res_features` | SAM2特有的高分辨率特征注入 |

### 5.4 后处理——上采样

```python
prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
```

将低分辨率掩码 logits（如 $256 \times 256$）双线性上采样回原始图像分辨率 $(H, W)$。

---

## 6. 损失函数深度解析

### 6.1 分割损失——二元交叉熵 (BCE)

```python
prd_mask = torch.sigmoid(prd_masks[:, 0])  # logits → 概率
gt_mask = torch.tensor(mask).cuda()         # GT 掩码

seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) 
            - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()
```

**完整数学形式**：

$$\mathcal{L}_{\text{seg}} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\hat{y}_i + \epsilon) + (1-y_i)\log(1-\hat{y}_i + \epsilon)\right]$$

其中：
- $y_i \in \{0, 1\}$：GT 掩码中像素 $i$ 的值
- $\hat{y}_i = \sigma(z_i)$：预测 logits 经 sigmoid 后的概率
- $\sigma(z) = \frac{1}{1+e^{-z}}$：sigmoid 函数
- $\epsilon = 10^{-5}$：数值稳定项，防止 $\log(0)$
- $N = H \times W$：像素总数

**为什么取 `prd_masks[:, 0]`？** 因为 `multimask_output=True` 会为每个点生成3个候选掩码，索引0对应最高置信度的掩码。

**为什么手动实现而非 `F.binary_cross_entropy`？** 手动实现更灵活，可以精确控制 $\epsilon$ 和计算图，但在功能上等价。

### 6.2 分数损失——IoU 监督

首先计算 GT 和预测之间的 **Intersection over Union (IoU)**：

```python
inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
```

$$\text{IoU} = \frac{|M_{\text{gt}} \cap M_{\text{prd}}|}{|M_{\text{gt}} \cup M_{\text{prd}}|} = \frac{\text{inter}}{\text{gt\_sum} + \text{prd\_sum} - \text{inter}}$$

其中：
- $M_{\text{gt}}$：二值 GT 掩码
- $M_{\text{prd}} = \mathbb{1}[\hat{y} > 0.5]$：预测概率经阈值0.5二值化后的掩码
- $\text{inter} = \sum_i M_{\text{gt},i} \cdot M_{\text{prd},i}$：交集像素数

然后分数损失为 **L1 损失**：

$$\mathcal{L}_{\text{score}} = \frac{1}{K}\sum_{k=1}^{K}|\text{prd\_scores}_k - \text{IoU}_k|$$

这个损失训练网络预测自己的分割质量，使得推理时可以筛选高质量掩码。

### 6.3 总损失

$$\mathcal{L} = \mathcal{L}_{\text{seg}} + 0.05 \cdot \mathcal{L}_{\text{score}}$$

分数损失的权重 $0.05$ 远小于1，因为：
1. 分数预测是辅助任务，主任务是分割质量
2. 分数损失的梯度可能不稳定（IoU 不可微，阈值操作截断梯度）
3. 过大的分数损失权重会干扰掩码学习的优化方向

> 📖 参考：[Multi-task Learning Using Uncertainty to Weigh Losses (Kendall et al., 2018)](https://arxiv.org/abs/1705.07115)

---

## 7. 训练动态分析

```python
mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
```

这是 **指数移动平均 (EMA)** 跟踪 IoU：

$$\mu_t = 0.99 \cdot \mu_{t-1} + 0.01 \cdot \overline{\text{IoU}_t}$$

其中 $\overline{\text{IoU}_t}$ 是第 $t$ 步所有掩码 IoU 的均值。平滑系数 $\alpha = 0.01$ 对应约 $1/\alpha = 100$ 步的有效窗口。

文章指出约 **25,000步** 后可见显著改善，这给出了一些收敛的直觉。

---

## 8. 推理流程详解

推理的目标：给定图像 + 感兴趣区域的粗掩码 → 生成完整的分割图。

### 8.1 点采样策略

```python
num_samples = 30  # 在目标区域内采样30个点
```

在粗掩码 $M_{\text{roi}}$ 内均匀随机采样 30 个点：

$$\{p_1, p_2, \ldots, p_{30}\} \sim \text{Uniform}(\{(i,j) | M_{\text{roi}}(i,j) > 0\})$$

### 8.2 多点预测

```python
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=np.ones([input_points.shape[0], 1])
)
```

每个点独立预测3个候选掩码，取第一个（最高置信度），共得到30个掩码。

### 8.3 掩码拼接——贪心合并算法

这是推理中最有趣的部分。问题：30个掩码之间可能重叠、冲突，如何合并成一致的全图分割？

**算法**：

1. **按分数排序**：高置信度的掩码优先处理
2. **贪心放置**：逐个掩码尝试加入分割图
3. **冲突检测**：如果新掩码与已占用区域的重叠率超过15%，则跳过

```python
if (mask * occupancy_mask).sum() / mask.sum() > 0.15: continue
```

数学上，重叠率定义为：

$$\text{overlap\_ratio} = \frac{|M_{\text{new}} \cap M_{\text{occupied}}|}{|M_{\text{new}}|}$$

只有当 $\text{overlap\_ratio} < 0.15$ 时才接受该掩码。

4. **去重后写入**：

```python
mask[occupancy_mask] = 0      # 去掉与已占用区域重叠的部分
seg_map[mask] = i + 1          # 写入分割索引
occupancy_mask[mask] = 1       # 标记为已占用
```

这个贪心算法的时间复杂度为 $O(N \cdot H \cdot W)$，其中 $N=30$ 是掩码数量。

**为什么阈值选15%？** 这是一个经验值：
- 太低（如5%）：会过度拒绝有效掩码
- 太高（如50%）：会接受大量重叠掩码，破坏分割一致性

> 📖 类似思路参考：[PointRend (Kirillov et al., 2020)](https://arxiv.org/abs/1912.08193)

---

## 9. 与 SAM1 的关键差异

| 特性 | SAM1 | SAM2 |
|------|------|------|
| 图像编码器 | ViT-H (632M params) | Hiera (多尺寸) |
| 高分辨率特征 | 无 | ✅ `high_res_features` 注入解码器 |
| 视频支持 | ❌ | ✅ Memory attention |
| 掩码解码器输入 | 仅 image_embed | image_embed + high_res_features |
| 位置编码 | 标准 | 改进的 dense PE |

SAM2 的 `high_res_features` 是一个重要改进：它将图像编码器中间层的特征注入掩码解码器，显著提升了分割边界的精度。

---

## 10. 潜在改进方向

### 10.1 损失函数改进

- **Dice Loss** 替代/补充 BCE：对小目标更友好

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2|M_{\text{gt}} \cap M_{\text{prd}}| + \epsilon}{|M_{\text{gt}}| + |M_{\text{prd}}| + \epsilon}$$

- **Focal Loss**：处理前景/背景的严重不平衡

$$\mathcal{L}_{\text{Focal}} = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

### 10.2 训练策略改进

- **LoRA / Adapter**：冻结全部参数，只在关键层插入低秩适配器，更高效
- **渐进式解冻**：先训练 mask decoder，再逐步解冻 image encoder 的后几层
- **数据增强**：颜色抖动、随机裁剪、水平翻转等

### 10.3 推理改进

- **非极大值抑制 (NMS)**：替代贪心合并，更系统化地处理重叠
- **点采样策略**：网格采样 + 随机采样混合，提高覆盖均匀性

> 📖 参考：[LoRA (Hu et al., 2022)](https://arxiv.org/abs/2106.09685), [Focal Loss (Lin et al., 2017)](https://arxiv.org/abs/1708.02002)

---

## 11. 完整代码流程图

```
┌─────────────────────────────────────────────────────┐
│                    训练流程                          │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. 加载数据 (image, masks, points, labels)         │
│     ↓                                               │
│  2. predictor.set_image(image)   ← 图像编码器(冻结) │
│     ↓                                               │
│  3. _prep_prompts() + sam_prompt_encoder()          │
│     ↓                            ← 提示编码器(训练) │
│  4. sam_mask_decoder()           ← 掩码解码器(训练) │
│     ↓                                               │
│  5. postprocess_masks()  →  prd_masks (全分辨率)    │
│     ↓                                               │
│  6. 计算 L_seg (BCE) + L_score (L1×0.05)           │
│     ↓                                               │
│  7. 反向传播 + AdamW 更新                            │
│     ↓                                               │
│  8. 每1000步保存模型                                 │
│                                                     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                    推理流程                          │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. 加载微调模型 + bfloat16                         │
│     ↓                                               │
│  2. 读取图像 + ROI粗掩码                            │
│     ↓                                               │
│  3. 在ROI内采样30个随机点                            │
│     ↓                                               │
│  4. predictor.predict() → 30个掩码 + 分数           │
│     ↓                                               │
│  5. 按分数排序掩码（高→低）                          │
│     ↓                                               │
│  6. 贪心合并（重叠>15%则跳过）                       │
│     ↓                                               │
│  7. 输出分割图                                      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 12. 核心直觉总结

1. **冻结编码器，微调解码器** — 这是迁移学习的黄金法则：通用表征已经很强大，只需要适配输出层
2. **单点 prompt 是核心** — SAM 的设计哲学是通过最简单的输入（一个点）获得最完整的分割
3. **IoU 作为自监督信号** — 网络学会评估自己的分割质量，这对推理时的掩码筛选至关重要
4. **贪心合并 > 复杂优化** — 简单的按分数排序 + 重叠阈值就能获得不错的一致分割图
5. **60行代码足以微调** — 体现了 SAM 架构设计和 API 抽象的优秀，大大降低了领域适配的门槛

这篇文章本质上展示了 **基础模型 + 少量微调** 的范式如何极大地降低特定领域分割任务的工程复杂度，是 Foundation Model 时代的一个经典范例。