# DETR: End-to-End Object Detection with Transformers 论文详解

- 机构: Facebook AI Research
- 发表: arXiv:2005.12872 (2020)
- 代码: https://github.com/facebookresearch/detr

## 一、研究背景与动机

### 1.1 传统目标检测的问题
在DETR出现之前，主流目标检测器（如Faster R-CNN、YOLO等）都包含大量手工设计的组件：

- **Anchor生成**：需要预设anchor boxes的尺度和宽高比
- **NMS (Non-Maximum Suppression)**：后处理步骤用于去除重复检测
- **Anchor分配规则**：通过启发式算法将ground truth分配给anchors
- **多阶段pipeline**：提案生成、分类、回归等多个独立的步骤

这些组件引入了大量任务相关的先验知识，使得整个检测pipeline复杂且难以优化。

### 1.2 核心思想
DETR将目标检测重新定义为**直接的集合预测问题**（direct set prediction problem）：
- 端到端训练，无需手工设计的anchors或NMS
- 使用Transformer的encoder-decoder架构
- 通过二分图匹配（bipartite matching）实现唯一的预测-真值对应

## 二、DETR核心架构

### 2.1 整体架构图解

```
输入图像 (H×W×3)
    ↓
CNN Backbone (ResNet-50/101)
    ↓ Feature Map (C×H/32×W/32)
    ↓ 1×1卷积 (降维到d)
    ↓ 展平并添加位置编码
    ↓
┌─────────────────────────┐
│  Transformer Encoder    │
│   (Self-Attention)     │
│   全局场景理解          │
└─────────────────────────┘
    ↓ Memory (H×W个token)
    ↓
┌─────────────────────────┐
│  Transformer Decoder    │
│   (N个object queries)   │
│   自注意力 + 交叉注意力 │
└─────────────────────────┘
    ↓ N个输出嵌入 (N=100)
    ↓
┌────────────────┬────────────────┐
│   FFN for      │   FFN for      │
│  Classification│  Bounding Box  │
│  (C+1 classes) │  (4 coordinates)│
└────────────────┴────────────────┘
```

### 2.2 数学公式详解

#### 2.2.1 二分图匹配 Loss

DETR的核心创新是使用二分图匹配来解决集合预测问题。给定：
- Ground truth集合: y = {(c_i, b_i)} for i=1,...,n
- 预测集合: ŷ = {(ĉ_i, b̂_i)} for i=1,...,N (N >> n)

其中c_i是类别，b_i是边界框坐标。

**第一步：寻找最优匹配**

使用Hungarian算法找到最优排列σ：

$$
\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^N \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})
$$

匹配成本包含分类和边界框两部分：

$$
\mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)}) = -\mathbb{1}_{\{c_i \neq \varnothing\}} \hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\sigma(i)})
$$

**第二步：Hungarian Loss**

在最优匹配基础上计算最终损失：

$$
\mathcal{L}_{\text{Hungarian}}(y, \hat{y}) = \sum_{i=1}^N \left[-\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)})\right]
$$

特殊处理：当c_i = ∅（无对象）时，对数概率项会被除以10来平衡类别不平衡。

#### 2.2.2 边界框 Loss

DETR直接预测绝对边界框坐标，使用组合损失：

$$
\mathcal{L}_{\text{box}}(b_i, \hat{b}_i) = \lambda_{\text{iou}} \mathcal{L}_{\text{iou}}(b_i, \hat{b}_i) + \lambda_{\text{L1}} \|b_i - \hat{b}_i\|_1
$$

其中**Generalized IoU Loss**定义为：

$$
\mathcal{L}_{\text{GIoU}}(b, \hat{b}) = 1 - \left(\frac{|b \cap \hat{b}|}{|b \cup \hat{b}|} - \frac{|B(b, \hat{b}) \setminus b \cup \hat{b}|}{|B(b, \hat{b}|}\right)
$$

B(b, b̂)是包含两个边界框的最小外接矩形。GIoU通过考虑非重叠区域，解决了传统IoU在边界框不重叠时梯度为0的问题。

**超参数**：
- λ_L1 = 5
- λ_iou = 2

#### 2.2.3 Multi-Head Attention 机制

DETR使用标准的Transformer注意力机制。对于单个注意力头：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d'}}\right)V
$$

查询Q、键K、值V的计算（包含位置编码）：

$$
[Q; K; V] = [T_1'(X_q + P_q); T_2'(X_{kv} + P_{kv}); T_3'X_{kv}]
$$

其中：
- T', T_1', T_2', T_3'是可学习的投影矩阵
- P_q, P_{kv}是位置编码
- d' = d/M是每个头的维度

### 2.3 关键组件

#### 2.3.1 Object Queries
DETR使用N个学习的位置嵌入作为object queries（N=100），这是解码器的输入。每个query学习关注特定区域和尺度的对象。

**特性**：
- 并行解码（非自回归）
- 通过训练自动差异化
- 可学习的位置编码

#### 2.3.2 位置编码

DETR使用两类位置编码：

**空间位置编码**：用于encoder，将2D位置信息编码为：

$$
PE_{(x,y)}^{2i} = \sin(x/10000^{2i/d})
$$
$$
PE_{(x,y)}^{2i+1} = \cos(x/10000^{2i/d})
$$

类似地处理y坐标和索引奇偶。

**输出位置编码**：object queries，与空间编码独立学习。

#### 2.3.3 辅助解码 Loss

在训练过程中，每个decoder层都输出预测并计算loss，帮助模型学习：

$$
\mathcal{L}_{\text{total}} = \sum_{l=1}^{L} \mathcal{L}_{\text{Hungarian}}^{(l)}
$$

其中L是decoder层数（L=6）。

## 三、实验结果与数据

### 3.1 COCO数据集实验结果

**表1：与Faster R-CNN的对比**

| Model | GFLOPS/FPS | #params | AP | AP50 | AP75 | APS | APM | APL |
|-------|-----------|---------|----|----|----|-----|-----|-----|
| Faster RCNN-FPN | 180/26 | 42M | 40.2 | 61.0 | 43.8 | 24.2 | 43.5 | 52.0 |
| Faster RCNN-R101-FPN | 246/20 | 60M | 42.0 | 62.5 | 45.9 | 25.2 | 45.6 | 54.6 |
| Faster RCNN-FPN+ | 180/26 | 42M | 42.0 | 62.1 | 45.5 | 26.6 | 45.4 | 53.4 |
| Faster RCNN-R101-FPN+ | 246/20 | 60M | 44.0 | 63.9 | 47.8 | 27.2 | 48.1 | 56.0 |
| **DETR** | 86/28 | 41M | 42.0 | 62.4 | 44.2 | 20.5 | 45.8 | **61.1** |
| **DETR-DC5** | 187/12 | 41M | 43.3 | 63.1 | 45.9 | 22.5 | 47.3 | **61.1** |
| **DETR-R101** | 152/20 | 60M | 43.5 | 63.8 | 46.4 | 21.9 | 48.0 | **61.8** |
| **DETR-DC5-R101** | 253/10 | 60M | 44.9 | 64.7 | 47.7 | 23.7 | 49.5 | **62.3** |

**关键发现**：
- DETR在小目标性能(APS)较差（-5.5 AP），但在大目标(APL)上显著优于Faster R-CNN（+7.8 AP）
- FLOPS效率更高：DETR比Faster R-CNN-FPN少约50%的计算量
- 推理速度相当：28 FPS vs 26 FPS

### 3.2 消融实验

**表2：Encoder层数影响**

| #layers | GFLOPS/FPS | #params | AP | AP50 | APS | APM | APL |
|---------|-----------|---------|----|----|-----|-----|-----|
| 0 | 76/28 | 33.4M | 36.7 | 57.4 | 16.8 | 39.6 | 54.2 |
| 3 | 81/25 | 37.4M | 40.1 | 60.6 | 18.5 | 43.8 | 58.6 |
| 6 | 86/23 | 41.3M | 40.6 | 61.6 | 19.9 | 44.3 | 60.2 |
| 12 | 95/20 | 49.2M | 41.6 | 62.1 | 19.8 | 44.9 | 61.9 |

发现：Encoder至关重要，移除encoder导致AP下降3.9点，尤其是大目标下降6.0 AP。这是因为encoder通过全局注意力分离了不同的实例。

**表3：位置编码消融**

| spatial pos. enc. | output pos. enc. | AP | ΔAP | AP50 | ΔAP50 |
|------------------|-------------------|----|-----|----|----|
| none | learned at input | 32.8 | -7.8 | 55.2 | -6.5 |
| sine at input | learned at input | 39.2 | -1.4 | 60.0 | -1.6 |
| learned at attn. | learned at attn. | 39.6 | -1.0 | 60.7 | -0.9 |
| none | sine at attn. | 39.3 | -1.3 | 60.3 | -1.4 |
| sine at attn. | learned at attn. | **40.6** | - | **61.6** | - |

关键：空间位置编码对性能至关重要，但只需在decoder中添加即可（-1.3 AP略差）。

**表4：Loss组件消融**

| class | ℓ1 | GIoU | AP | ΔAP | AP50 | APS | APM | APL |
|-------|----|----|----|----|----|-----|-----|-----|
| ✓ | ✓ | | 35.8 | -4.8 | 57.3 | 13.7 | 39.8 | 57.9 |
| ✓ | | ✓ | 39.9 | -0.7 | 61.6 | 19.9 | 43.2 | 57.9 |
| ✓ | ✓ | ✓ | 40.6 | - | 61.6 | 19.9 | 44.3 | 60.2 |

GIoU loss贡献了大部分性能改善，ℓ1 loss提供额外的中等和大目标提升。

### 3.3 Decoder层数分析

图4显示AP和AP50随decoder层的变化：
- 第一层：AP ~28.4, AP50 ~46.7
- 第二层：AP ~31.1
- ...
- 第六层：AP ~36.6

每一层都持续改善，总计+8.2 AP improvement，证明了深度解码器的价值。NMS在早期层有益，但后期层反而可能损害性能（移除真阳性）。

**NMS效果**：
- 第1层：NMS提升+1.1 AP（减少重复检测）
- 第6层：NMS降低-0.4 AP（移除真阳性）

这证明了Self-attention在抑制重复预测方面的作用。

### 3.4 Panoptic Segmentation扩展

**表5：Panoptic分割结果**

| Model | Backbone | PQ | SQ | RQ | PQ^th | SQ^th | RQ^th | PQ^st | SQ^st | RQ^st | AP |
|-------|---------|----|----|----|-----|-----|-----|-----|-----|-----|----|
| PanopticFPN++ | R50 | 42.4 | 79.3 | 51.6 | 49.2 | 82.4 | 58.8 | 32.3 | 74.8 | 40.6 | 37.7 |
| UPSnet | R50 | 42.5 | 78.0 | 52.5 | 48.6 | 79.4 | 59.6 | 33.4 | 75.9 | 41.7 | 34.3 |
| **DETR** | R50 | **43.4** | 79.3 | **53.8** | 48.2 | 79.8 | 59.5 | **36.3** | **78.5** | **45.3** | 31.1 |
| **DETR-R101** | R101 | **45.1** | 79.9 | **55.5** | 50.5 | 80.9 | 61.7 | **37.0** | **78.5** | **46.0** | 33.0 |
| PanopticFPN++ | R101 | 44.1 | 79.5 | 53.3 | 51.0 | 83.2 | 60.6 | 33.6 | 74.0 | 42.1 | 39.7 |

DETR在panoptic分割中尤其优势在Stuff类别（PQ^st），归功于全局推理能力。

## 四、计算复杂度分析

**每层Self-Attention复杂度**：

$$
\mathcal{O}_{\text{encoder}} = \mathcal{O}(d^2HW + d(HW)^2)
$$

$$
\mathcal{O}_{\text{decoder-self}} = \mathcal{O}(d^2N + dN^2)
$$

$$
\mathcal{O}_{\text{cross-attn}} = \mathcal{O}(d^2(N+HW) + dNHW)
$$

当HW ≈ 2500 (50×50特征图)，N = 100：
- Encoder自注意力占主导
- Decoder代价很小，因为N << HW

## 五、训练策略

**关键超参数**：
- 优化器：AdamW
- Transformer学习率：10^-4
- Backbone学习率：10^-5
- 权重衰减：10^-4
- 梯度裁剪：0.1
- Dropout：0.1

**训练长度**：
- 标准训练：300 epochs（+1.5 AP vs短训练）
- 长训练：500 epochs（对比Faster R-CNN）
- 辅助loss：每个decoder层都计算

**数据增强**：
- 短边缩放：480-800像素
- 长边限制：1333像素
- 随机裁剪（概率0.5）：+1 AP
- 0-padding保证batch维度一致

**硬件需求**：
- 16个V100 GPU
- Batch size = 64 (4 images/GPU)
- 300 epochs约3天

## 六、代码实现

论文提供了一个简洁的PyTorch实现（<50行推理代码）：

```python
class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads, 
                 num_encoder_layers, num_decoder_layers):
        super().__init__()
        # Backbone: ResNet-50
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        
        # Transformer
        self.transformer = nn.Transformer(hidden_dim, nheads,
                                          num_encoder_layers, num_decoder_layers)
        
        # 预测头
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        
        # 可学习嵌入
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))  # object queries
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
    
    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        
        # 构建位置编码
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        
        # Transformer forward
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
                            self.query_pos.unsqueeze(1))
        
        return self.linear_class(h), self.linear_bbox(h).sigmoid()
```

## 七、优势与局限性

### 7.1 优势

1. **端到端训练**：无需anchors、proposal生成、NMS等手工组件
2. **概念简洁**：标准CNN + Transformer + FFN，易于理解和复现
3. **全局推理**：Transformer Self-attention建模全局上下文
4. **大目标优势**：APL显著优于传统方法（+7.8 AP）
5. **易于扩展**：统一处理thing和stuff类别，可直接应用于panoptic分割
6. **并行推理**：100个objects并行解码
7. **性能竞争力**：与高度优化的Faster R-CNN comparable

### 7.2 局限性

1. **小目标性能**：APS显著低于Faster R-CNN（-5.5 AP）
   - 原因：1/32下采样率丢失细节
   - 解决：DETR-DC5通过dilated convolution提升小目标性能

2. **训练需求**：
   - 需要更长的训练schedule（300-500 epochs vs 3x schedule for Faster R-CNN）
   - 需要大量GPU资源

3. **最大检测数量**：N=100的上限，当物体数量接近100时性能下降
   - 当100个物体时，平均只检测30个
   - 原因：训练数据很少有这么密集的场景

4. **收敛速度**：初期性能较差，需要长时间训练

## 八、技术洞察与可视化

### 8.1 Attention可视化

**Encoder Self-Attention**：
- 专注于分离图像中的实例
- 利用全局上下文区分不同对象
- 不同位置关注不同实例的边界

**Decoder Attention**：
- 相对局部，主要关注物体的极端部分
- 如头部、四肢等关键区域
- Encoder已经分离实例后，Decoder只需关注特征边界

### 8.2 Object Queries特性

通过对100个query slots的分析发现：
- 每个slot学习专门化的预测
- 多种工作模式：不同区域、不同大小的目标
- 所有slot都有预测大图像范围框的模式
- 与COCO数据集的目标分布相关

### 8.3 泛化能力

实验显示DETR具有良好的外分布泛化：
- 在从未见过24个长颈鹿的合成图像上，成功检测所有24个
- 不需要特定的类别专门化
- 证明了object queries的泛化性

## 九、影响与后续工作

DETR开辟了目标检测的新方向，催生了大量后续研究：

### 主要变体：
- **Deformable DETR** (2021)：引入可变形注意力，加速收敛，改善小目标性能
- **Conditional DETR**：条件空间查询，加速收敛
- **DAB-DETR**：动态锚框更新
- **DN-DETR**：去噪训练
- **H-DETR**：分层检测
- **RT-DETR**：实时DETR

### 在其他任务应用：
- 实例分割（MaskFormer）
- 视频目标检测
- 多目标跟踪
- 3D目标检测
- 医学图像分析

### 技术影响：
- 集合预测成为新范式
- Hungarian matching成为标配
- 证明Vision Transformer的潜力
- 推动注意力机制在视觉任务的研究

## 十、总结

DETR是目标检测领域的里程碑工作，通过以下创新：

1. **将目标检测视为集合预测问题**，使用二分图匹配解决
2. **引入Transformer架构**，替代传统复杂的检测pipeline
3. **端到端训练**，移除所有手工设计的组件
4. **统一的框架**，可扩展到panoptic分割

虽然在某些方面（如小目标性能、训练效率）仍有局限，但DETR为社区提供了新的思考方向和技术路径。其简洁的架构和强大的扩展性使其成为后续大量研究的基础，推动了目标检测向更简洁、更统一的方向发展。

---

## 参考资料

1. **论文**: https://arxiv.org/abs/2005.12872
2. **官方代码**: https://github.com/facebookresearch/detr
3. **相关博客**:
   - https://towardsdatascience.com/detr-end-to-end-object-detection-with-transformers-4f6093c7a753
   - https://distill.pub/2020/circuits/zoom-in/
4. **后续工作**:
   - Deformable DETR: https://arxiv.org/abs/2010.04159
   - Conditional DETR: https://arxiv.org/abs/2108.06152
   - DAB-DETR: https://arxiv.org/abs/2201.12329
5. **COCO数据集**: https://cocodataset.org/
6. **Transformer论文**: https://arxiv.org/abs/1706.03762