# Niantic 的 Large Geospatial Model (LGM)：空间智能的下一个前沿

这篇文章由 Niantic 的 Eric Brachmann 和 Victor Adrian Prisacariu 撰写，阐述了一个雄心勃勃的愿景：**从数千万个局部神经地图出发，构建一个全球性的大型地理空间模型，赋予机器类似人类的"空间理解"能力**。下面我从第一性原理出发，逐层拆解其技术内核。

---

## 一、核心问题的第一性原理分析

### 1.1 问题的本质：从局部观测到全局推理的鸿沟

人类的空间智能建立在一个基本事实之上：

> **世界具有结构性先验（structural priors）**——物体由固体物质构成，有正面和背面；人造结构遵循对称、布局等文化规则；外观随时间和季节变化。

这意味着：
- 人类只需看到教堂的正面，就能**推断**背面大概长什么样（因为见过大量教堂的先验知识）
- 机器的局部模型只能**插值**于已见视角，无法**外推**到未见视角

这本质上是一个 **few-shot/zero-shot spatial reasoning** 问题，与 LLM 中从训练语料学习语言先验、然后泛化到新文本的范式高度同构。

### 1.2 LLM → LGM 的类比框架

| 维度 | LLM | LGM |
|------|-----|-----|
| **原始数据** | 互联网文本（billions of tokens） | 锚定地理位置的图像（billions of geo-anchored images） |
| **核心表征** | 离散 token 的语义嵌入 | 3D 场景的几何+外观+文化嵌入 |
| **推理能力** | 语言的补全与生成 | 空间的补全与想象（从新角度"看到"场景） |
| **度量约束** | 无（文本无物理尺度） | **有（metric space，厘米级精度）** |
| **泛化机制** | 全局插值 → 局部外推 | 全局插值 → 局部外推 |

关键差异在于 **metric quality**：LGM 不是生成无尺度的 3D asset，而是绑定到公制空间的"下一代地图"。这使得 LGM 的输出不仅是视觉上合理的，更是**物理上精确的**。

---

## 二、技术栈深度解析

### 2.1 Visual Positioning System (VPS) —— 当前的生产系统

VPS 的核心 pipeline：

```
用户扫描 → 多视角图像 + 位置信息 → Structure from Motion (SfM) → 经典3D视觉地图
                                              ↓
                                    神经网络隐式编码 → Neural Map
                                              ↓
                              单张查询图像 → 厘米级6DoF定位
```

**关键数据指标：**
- **10M** scanned locations globally
- **1M+** activated VPS locations
- **~1M** fresh scans per week（每条 scan 含数百张离散图像）
- **50M+** neural networks trained（多个网络可对应同一地点）
- **150T+** total parameters（所有网络参数之和）

### 2.2 ACE (2023) & ACE Zero (2024) —— 神经地图的核心架构

这两篇论文是实现 "neural map" 的关键技术，其核心思想是：

> **用神经网络的可学习参数隐式编码一个地点的3D地图，而非使用经典的3D数据结构（如点云、网格）。**

#### ACE (Accelerated Coordinate Encoding) 的核心公式

ACE 的目标是从单张查询图像 $\mathbf{I}_q$ 预测相机的6DoF位姿 $(\mathbf{R}, \mathbf{t})$，其中 $\mathbf{R} \in SO(3)$ 是旋转矩阵，$\mathbf{t} \in \mathbb{R}^3$ 是平移向量。

其核心是一个 scene coordinate regression 网络 $f_\theta$：

$$\mathbf{y}(u,v) = f_\theta(\mathbf{I}_q, u, v)$$

其中：
- $(u,v)$ 是图像中的像素坐标
- $\mathbf{y}(u,v) \in \mathbb{R}^3$ 是该像素对应的3D场景坐标
- $\theta$ 是该地点专属的网络参数

给定足够多的 scene coordinate 预测，通过 RANSAC + PnP（Perspective-n-Point）即可求解相机位姿：

$$\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \left\| \mathbf{R} \mathbf{y}_i + \mathbf{t} - \mathbf{X}_i \right\|^2$$

其中 $\mathbf{X}_i$ 是像素 $i$ 的射线方向在相机坐标系中的投影。

**ACE 的关键创新**：将数千张 mapping images 压缩成一个轻量级神经表征，训练速度极快（几分钟内完成一个地点的映射）。

#### ACE Zero (2024) 的改进

ACE Zero 进一步消除了对已知相机位姿的依赖（即不需要 SfM 预处理），实现了 **zero-calibration** 的神经地图构建。这意味着：

$$\{\theta_{\text{map}}\} = \text{Train}_{\text{ACE Zero}}(\{\mathbf{I}_k\}_{k=1}^{N})$$

其中输入仅为图像集合 $\{\mathbf{I}_k\}$，无需预先计算的位姿标签。

### 2.3 MicKey (2024) —— LGM 的雏形

MicKey 是 LGM 概念的 proof-of-concept，解决的核心问题是：

> **给定两张来自截然不同视角的图像，预测它们之间的相对位姿。**

#### MicKey 的技术细节

给定图像 $\mathbf{I}_1$ 和 $\mathbf{I}_2$，MicKey 预测相对位姿 $(\mathbf{R}_{12}, \mathbf{t}_{12})$：

$$({\mathbf{R}_{12}}, \mathbf{t}_{12}) = g_\phi(\mathbf{I}_1, \mathbf{I}_2)$$

其架构的关键要素：

1. **Feature Extraction**：使用共享的 backbone（如 ResNet/ViT）提取两张图像的特征图
2. **Cross-Attention Matching**：通过 cross-attention 机制建立图像间的对应关系
3. **Pose Regression**：从匹配特征回归相对位姿

MicKey 能处理 **极端视角变化**（包括对向拍摄，即 $\sim 180°$ 的视角差异），这暗示模型学到了超越局部特征匹配的**场景级语义理解**。

**局限性**：
- 仅支持双视图输入（2-view only）
- 训练数据量相对较小
- 但已证明：即使有限数据，模型也能学到跨视角的几何先验

---

## 三、从局部到全局：LGM 的架构设想

### 3.1 当前系统的局限

当前的 VPS 是一个 **分布式局部模型集合**：

$$\text{VPS} = \{f_{\theta_1}^{\text{loc}_1}, f_{\theta_2}^{\text{loc}_2}, \ldots, f_{\theta_M}^{\text{loc}_M}\}$$

其中 $M > 50\text{M}$，每个 $f_{\theta_i}^{\text{loc}_i}$ 是一个独立的局部神经地图。

**核心失效模式**：当查询图像的视角超出局部模型的训练分布时，定位失败。即：

$$P(\text{success} | \mathbf{I}_q, f_{\theta_i}^{\text{loc}_i}) \approx 0 \quad \text{if } \mathbf{I}_q \notin \text{support}(\text{training views of loc}_i)$$

### 3.2 LGM 的架构假设

文章暗示（但未明确给出）的 LGM 架构可能如下：

$$f_{\Theta}^{\text{global}} = \text{Aggregate}\left(\{f_{\theta_i}^{\text{loc}_i}\}_{i=1}^{M}\right)$$

其中 $\Theta$ 是全局模型参数，它蒸馏了所有局部模型的共同知识。

**具体可能的实现路径**：

#### 路径 A：知识蒸馏

$$\mathcal{L}_{\text{distill}} = \sum_{i} \mathbb{E}_{\mathbf{I} \sim \mathcal{D}_i} \left[ \mathcal{L}\left(f_{\Theta}^{\text{global}}(\mathbf{I}), f_{\theta_i}^{\text{loc}_i}(\mathbf{I})\right) \right]$$

全局模型通过模仿所有局部模型的输出进行训练。

#### 路径 B：条件生成

$$f_{\Theta}^{\text{global}}(\mathbf{I}_q, \mathbf{c}_{\text{geo}}) \rightarrow (\mathbf{R}, \mathbf{t}, \{\mathbf{y}(u,v)\})$$

其中 $\mathbf{c}_{\text{geo}}$ 是地理上下文编码（如 GPS 坐标、区域类型标签），全局模型根据地理上下文条件化生成场景表征。

#### 路径 C：层级架构

```
Global LGM (语义/结构先验)
    ↕ 通信
Local Neural Maps (精细几何/外观)
```

全局模型提供 "church 的背面大概长什么样" 的先验，局部模型提供 "这个特定教堂的精确几何" 的细节。

### 3.3 核心原则：全局插值 → 局部外推

这是文章最深刻的一句话：

> **"The LGM extrapolates locally by interpolating globally."**

数学直觉：

- **全局插值**：在"教堂"这个概念空间中，通过见过数千个教堂，LGM 学到了教堂的统计分布 $P(\text{church appearance} | \text{viewpoint}, \text{region})$
- **局部外推**：对于某个只映射了正面的特定教堂，LGM 利用全局先验推断其背面

$$P(\text{back of church}_A) = \int P(\text{back} | \text{front}, \theta_{\text{global}}) \, d\theta_{\text{global}}$$

这与 LLM 的工作方式完全类比：LLM 通过在大量文本上插值学到了语言的统计结构，从而能对新 prompt 进行外推生成。

---

## 四、数据飞轮与竞争壁垒

### 4.1 Niantic 的数据优势

```
用户玩游戏/扫描 → 每周1M+新扫描 → 训练局部模型 → 提升VPS精度 → 更好的AR体验 → 更多用户 → 更多扫描
```

这个飞轮的关键在于：
- 数据是 **行人视角**（pedestrian perspective），包含车无法到达的地方
- 数据具有 **时序多样性**（不同时间、季节、天气）
- 数据具有 **地理锚定**（精确的地理位置信息）

### 4.2 数据规模对比

| 数据源 | 规模 | 视角 | 地理锚定 |
|--------|------|------|----------|
| Google Street View | ~220B images | 车载视角（街道） | 有 |
| Niantic VPS | ~10M locations, 每周1M scans | 行人视角（室内+室外） | 有 |
| MegaDepth | ~100 scenes | 游客视角 | 有（有限） |
| BLINDS | 规模有限 | 多样 | 部分有 |

Niantic 的独特优势在于 **行人视角 + 全球覆盖 + 持续增长**，这是自动驾驶公司或互联网公司难以复制的。

---

## 五、Foundation Models 生态系统

### 5.1 互补架构

文章描绘了三种 foundation model 的协同：

```
┌──────────────────────────────────────────────────┐
│              Foundation Model 生态系统              │
│                                                    │
│   LLM ←→ Multimodal Model ←→ LGM                 │
│    │           │                    │              │
│  语言推理    视觉-语言对齐      空间-几何推理        │
│    │           │                    │              │
│  "这是什么"  "看起来像教堂"   "背面应该是这样"     │
│    │           │                    │              │
│    └───────────┴────────────────────┘              │
│                    ↓                               │
│         统一的空间计算操作系统                        │
└──────────────────────────────────────────────────┘
```

### 5.2 具体交互场景

1. **LLM + LGM**：用户问 "附近有什么有趣的建筑？" → LLM 理解语言 → LGM 提供空间上下文 → 返回空间感知的回答
2. **Multimodal + LGM**：用户拍一张照片 → Multimodal model 识别内容 → LGM 定位 + 推断未见图 → AR 叠加
3. **LGM → Generation**：LGM 的 scene-level features 可用于 3D 场景的补全、编辑和生成

---

## 六、应用场景深度展开

### 6.1 近期应用

| 应用 | 技术需求 | 当前成熟度 |
|------|----------|-----------|
| Pokémon Playgrounds | 厘米级 VPS 定位 | ✅ 已上线 |
| AR 导航 | VPS + 路径规划 | 🔬 实验中 |
| 空间问答 | LLM + LGM | 🔬 研究中 |

### 6.2 中期应用

- **AR 眼镜（音频/2D显示）**：LGM 引导用户，回答空间相关问题
- **3D 场景补全**：LGM 生成/完成场景的3D表征
- **物流**：精确的空间理解优化最后一公里配送

### 6.3 远期应用

- **自主系统**：机器人导航需精确的空间理解
- **空间规划与设计**：LGM 提供城市级3D语义地图
- **远程协作**：共享空间感知的虚拟会议

---

## 七、技术挑战与开放问题

### 7.1 模型架构挑战

1. **如何从50M个独立局部模型蒸馏到一个全局模型？**
   - 直接合并150T参数不可行
   - 需要高效的参数共享和蒸馏策略

2. **如何处理地理空间的连续性？**
   - 相邻地点的模型应共享信息
   - 需要设计空间感知的模型架构（类似 NeRF 中的空间哈希编码）

3. **如何平衡全局先验与局部精度？**
   - 全局先验可能引入幻觉（如误判特定教堂的独特结构）
   - 需要不确定性量化机制

### 7.2 数据挑战

1. **数据分布极度不均匀**：城市区域密集，乡村稀疏
2. **时变场景**：建筑翻新、季节变化、施工
3. **隐私与合规**：用户扫描可能包含敏感信息

### 7.3 评估挑战

- 如何定义 LGM 的"空间智能"评估指标？
- 现有的视觉定位 benchmark（如 Cambridge Landmarks, 7-Scenes）规模太小
- 需要新的大规模、多视角、跨地理位置的评估框架

---

## 八、与相关工作的关系

| 工作 | 与 LGM 的关系 |
|------|-------------|
| **NeRF / 3D Gaussian Splatting** | 单场景的隐式表征，LGM 的局部组件的底层技术 |
| **LERF (Language-Embedded Radiance Fields)** | 将语言嵌入3D场，LGM 可借鉴其语义-几何融合方式 |
| **Contrastive Language-Image Pretraining (CLIP)** | 视觉-语言对齐，multimodal model 的基础 |
| **DUSt3R (2024)** | 无需相机参数的3D重建，与 ACE Zero 思路类似 |
| **MASt3R (2024)** | DUSt3R 的匹配增强版，与 MicKey 目标类似 |
| **GeoLRM (2024)** | Google 的地理3D重建，LGM 的潜在竞品 |
| **InstantSplat (2024)** | 快速 sparse-view 3D重建，与 ACE 速度优化目标类似 |

---

## 九、关键论文索引

1. **ACE (2023)** - Brachmann et al., "Accelerated Coordinate Encoding: Learning to Relocalize in Minutes using RGB and Poses", CVPR 2023. [arXiv:2305.14059](https://arxiv.org/abs/2305.14059)
2. **ACE Zero (2024)** - Brachmann et al., "ACE Zero: Visual Localization Without Any Prior Views", CVPR 2024. [arXiv:2312.00435](https://arxiv.org/abs/2312.00435)
3. **MicKey (2024)** - Brachmann et al., "MicKey: Matching and Keypoint Detection for Camera Pose Estimation", CVPR 2024. [arXiv:2404.06437](https://arxiv.org/abs/2404.06437)
4. **DUSt3R (2024)** - Wang et al., "DUSt3R: Geometric 3D Vision Made Easy", CVPR 2024. [arXiv:2312.14132](https://arxiv.org/abs/2312.14132)
5. **MASt3R (2024)** - Leroy et al., "Grounding Image Matching in 3D with MASt3R", arXiv 2024. [arXiv:2406.09756](https://arxiv.org/abs/2406.09756)
6. **GeoLRM (2024)** - Tang et al., "GeoLRM: Geometry-Aware Large Reconstruction Model", arXiv 2024. [arXiv:2406.15325](https://arxiv.org/abs/2406.15325)

---

## 十、总结：LGM 的核心洞察

LGM 的本质洞察可以用一个公式概括：

$$\underbrace{P(\text{scene}_{\text{novel view}} | \text{scene}_{\text{observed}})}_{\text{局部外推}} = \int \underbrace{P(\text{scene}_{\text{novel view}} | \text{scene}_{\text{observed}}, \theta_{\text{global}})}_{\text{条件生成}} \cdot \underbrace{P(\theta_{\text{global}} | \{\text{scenes}_{\text{all}}\})}_{\text{全局先验}} \, d\theta_{\text{global}}$$

- **全局先验** 来自于在数百万地点上的训练
- **条件生成** 利用了特定地点的观测数据
- **局部外推** 是两者的结合，赋予机器"从未见过的角度想象一个地方"的能力

这正是从 **perception**（感知）到 **spatial intelligence**（空间智能）的跃迁——也是 Niantic 所说的 **"空间智能是 AI 的下一个前沿"** 的技术根基。