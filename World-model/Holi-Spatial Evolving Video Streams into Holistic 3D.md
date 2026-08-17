---
source_pdf: Holi-Spatial Evolving Video Streams into Holistic 3D.pdf
paper_sha256: 78848dcfd48b9bd788523b9315fc1984d8fe94b439b66e8ed5483a1a6f194972
processed_at: '2026-08-04T23:54:05-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Holi-Spatial

Andrej, 咱们抛开那些公式和术语, 用大白话重新捋一遍这篇paper。

---

## 这篇paper到底在干嘛?

想象你要训练一个AI, 让它能理解3D世界——比如看一段房间的视频, 能知道"沙发在左边, 茶几在沙发前面, 沙发是米色的, 大概两米长"。

问题来了: 训练这种AI需要大量标注好的3D数据。但现成的3D数据集少得可怜, ScanNet这种最经典的, 也就1500个房间, 50个物体类别。50个类别? 你家厨房的调味料都不止50种。而且都是人工标注的, 边界还经常标得毛毛糙糙。

那怎么办? 这帮人说: **我们干脆自己造一个自动标注机器, 喂进去视频, 吐出来3D标注, 不用一个人工标注员。**

而且他们赌的是: 2025年了, 各种AI工具(Depth-Anything-V3估深度、SAM3分割、Gemini/Qwen3看图说话)已经够强了, 把它们像乐高积木一样拼起来, 标注质量能超过人工。

结果呢? ScanNet++上3D物体检测, 之前最好的方法AP25是12.2, 他们做到了81.06。差不多是**7倍的差距**。

---

## 整个pipeline就像造一辆车, 分三个车间

### 车间一: 把几何搞准 (Geometric Optimization)

你拿一段视频进去, 先要搞清楚这个房间在3D空间里长什么样。

第一步, 用COLMAP这种老工具算出每个画面的相机位置和朝向。然后, 用Depth-Anything-V3给每一帧估一个深度图。

但问题来了: Depth-Anything-V3是单帧估的, frame A估的深度和frame B估的深度, 放到3D空间里对不上, 会有"鬼影"——同一个物体在不同帧里的位置飘了。

怎么办? 他们用3D Gaussian Splatting (3DGS)来"拉齐"这些深度。3DGS本来是个渲染工具, 这里被当成一个"多视角几何约束器"——通过可微渲染, 强制不同帧的深度互相consistent, 同时把那些漂浮在半空中的"飞点"push到实际表面上。

**这步有多重要?** 他们做了ablation: 不做这步, 后续3D物体检测的precision只有0.13; 做了, 直接跳到0.81。6倍的提升。这是整个pipeline的地基。

类比一下: 就像你拍照, 每张照片都测距, 但每张照片的测距都有误差。如果只是单独看每张, 误差累积起来, 把照片拼成3D模型就乱套了。3DGS的作用相当于"多张照片互相校对", 把误差消掉。

---

### 车间二: 把物体认出来 (Image-level Perception)

几何搞准了, 接下来要认物体。

#### 一个小trick: "记忆本"

最朴素的做法是: 对每一帧, 用Gemini3-Pro生成caption, 然后让SAM3分割。但这有个问题——同一个沙发, frame 1里叫"wooden chair", frame 5里叫"dining chair", 后面就没法merge了。

他们的解法特别简单: 维护一个"记忆本" $\mathcal{M}_t$, 把之前所有帧认出来的类别都记下来, 每次给VLM看新帧时, 把记忆本也一起塞给它, 说"你优先用这些已有的名字"。

$$\mathcal{M}_t = \mathcal{M}_{t-1} \cup \text{Extract}(I_t)$$

就这么一个set union, 跨帧的语义一致性就保住了。没什么高深的, 但很实用。

#### 把2D mask变3D box的麻烦事

SAM3给你2D mask, 你要把它变成3D bounding box。直觉上很简单——把mask里的每个像素, 用深度反投影回3D, 然后fit一个box就行。

但实际上有两个坑:
1. **2D层面的坑**: SAM3的mask在物体边缘经常对不齐, 会有几个像素跑到背景里
2. **3D层面的坑**: 物体边缘深度跳变处, 会有"飞点"飘到前面或后面

他们的处理方式很"保守": 先把mask边缘往里腐蚀一圈, 只用中间靠谱的部分; 再用mesh深度当参考, 把不一致的像素过滤掉; 最后在干净的点云上fit box。

**宁可少用一些像素, 也要保证box不被飞点拉歪。** 后面的车间会把同一个物体的多个碎片merge回来。

---

### 车间三: 把碎片拼成完整物体 (Scene-level Refinement)

这是这篇paper最巧妙的地方。

#### 先merge

同一个沙发, 从不同角度拍, SAM3可能把它切成3块。在3D空间里, 这3块的box会overlap。如果两个box类别一样、3D IoU超过0.2, 就合并。

阈值0.2选得挺低——因为partial observation时, 不同角度看到的box overlap可能不大, 但确实是同一个物体。

#### 再验证 (最有意思的部分)

合并完, 每个物体有一个confidence score。最naive的做法是设个阈值, 比如高于0.85就留, 低于就扔。

但这个阈值会陷入两难:
- 阈值高 → 留下来的准, 但很多难认的物体(被遮挡的、远处的)就被扔了 → recall低
- 阈值低 → recall高, 但混进来一堆误分类的 → precision低

他们的解法是**三段式**:
- score ≥ 0.9 → 直接留
- score < 0.8 → 直接扔
- 0.8 ≤ score < 0.9 → **叫一个VLM agent来二次判断**

这个agent不是简单看一眼, 它有工具:
- **zoom-in tool**: 把那个区域放大了看细节
- **SAM3 re-segmentation tool**: 在放大图上重新分割验证

相当于把"灰区"的决策从"看一个scalar数字"升级成"看zoom-in后的细节+重新分割验证"。

**ablation数据**: 只用confidence filter, recall从0.74掉到0.69(扔掉了一些难认的真物体); 加上agent, recall回到0.89(把那些被误杀的真物体救回来), precision保持0.81。

用Karpathy你可能喜欢的说法: 这是把system 1的快速直觉判断和system 2的慢速推理判断结合起来。简单的事用system 1快速过, 难判断的事用system 2仔细想。

---

## 最终产出: Holi-Spatial-4M

喂进去ScanNet、ScanNet++、DL3DV的视频, 产出来:

- 12000个优化好的3DGS场景
- 130万个2D mask
- 32万个3D bounding box
- 32万个instance caption
- 120万个3D grounding对
- 125万个spatial QA对

**总共400多万条标注, 全自动, 零人工。**

QA对分两类:
- **相机视角的**(egocentric): 相机怎么转、往哪移、移了多远
- **物体视角的**(allocentric): 物体A离物体B多远、在什么方向、多大

---

## 用这个数据集finetune VLM, 效果怎么样?

他们finetune了Qwen3-VL的2B和8B版本, 1个epoch, 1024 batch size, 32张H800跑。

### 空间推理QA (MMSI-Bench, MindCube)

| 模型 | MMSI-Bench | MindCube |
|---|---|---|
| Qwen3-VL-8B原版 | 31.1 | 29.4 |
| Qwen3-VL-8B + Holi-Spatial | 32.6 | 49.1 |

MindCube上从29.4跳到49.1, 提升了快20个点。这个任务考的是"空间心象"——给你几个视角, 你得能在脑子里构建3D模型。用他们的QA数据训完, 这个能力涨了一大截。

### 3D grounding (ScanNet++)

| 方法 | AP50 |
|---|---|
| Qwen3-VL-8B原版 | 13.50 |
| Qwen3-VL-8B + Holi-Spatial | 27.98 |

AP50翻了一倍多。之前的模型有"视角偏见"——训练数据大多是单视角或锚点视角, 换个角度或深度就找不着物体了。

---

## 几个直白的洞察

### 1. 3DGS在这里根本不是渲染工具

大家提到3DGS想到的是"炫酷的实时渲染"。这篇paper里, 3DGS的角色是"多视角几何校准器"。它最重要的产出是**多视角一致的深度图**, 而不是好看的图片。

这是把3DGS当"正则化工具"用的思路, 挺聪明的。

### 2. "工具组合"为什么能打败端到端模型?

端到端模型把所有能力都压在一个网络里, error是耦合的——一个地方出错, 连带其他地方也错。

这个pipeline是模块化的: Depth用DA3(在几十亿单目视频上训的), Seg用SAM3(在十亿mask上训的), Caption用Gemini/Qwen3(在全网图文上训的)。每个模块都利用了自己最擅长的数据分布, error互相独立, 在交集处自然suppress。

而且每个模块的输出都能单独inspect和debug, 端到端模型做不到。

### 3. 那个VLM agent验证环节, 本质是cascaded classifier的思想升级版

传统cascaded classifier是用"快分类器先过一遍, 难的用慢分类器"。这里把"慢分类器"换成了"带工具的VLM agent"——它不只是再判一次, 而是zoom-in看细节、重新分割验证, 利用了LLM的语义推理能力。

这思路可以泛化: 任何confidence灰区的决策, 都可以用agentic的方式做二次验证。

### 4. "记忆本"那个trick看似简单, 但解决了一个真问题

跨帧语义漂移是video理解里很常见但容易被忽视的问题。用一个set union来维护label consistency, 朴素但有效。比直接用CLIP做feature matching轻量得多。

---

## 一句话总结

这篇paper的核心不是发明了什么新architecture, 而是证明了: **2025年了, 把一堆SOTA foundation model像管道一样串起来, 配上几何优化和agentic验证, 自动产生的3D标注能比人工标注质量更高、规模更大、类别更丰富。** 然后用这批数据finetune VLM, 空间理解能力直接起飞。

工程哲学就是: data flywheel转起来, 后面就是赢者通吃。

---

参考链接:
- 项目主页: https://visionary-laboratory.github.io/holi-spatial/
- 代码: https://github.com/Visionary-Laboratory/Holi-Spatial
- 联系: zhongzhihang95@gmail.com

---

# Holi-Spatial 深度技术讲解

Andrej, 这篇paper的核心 contribution 在于它把 spatial intelligence 的 data bottleneck 重新 frame 成一个 **engineering systems problem**——通过 systematic composition 多个 SOTA foundation models (Depth-Anything-V3, Gemini3-Pro, SAM3, Qwen3-VL) 加 per-scene 3DGS optimization, 实现了 raw video → holistic 3D annotations 的 fully automatic pipeline。下面我分层拆解 build 你的 intuition。

---

## 1. Motivation 与核心 Insight

**问题根源**: 现有 spatial intelligence benchmark (如 ScanNet [28], ScanNet++ [27], SenseNova-SI-800K [37], VST-4M [26]) 都依赖一个很小的 manually annotated 3D scan pool (例如 ScanNet 只有 ~1500 个 scenes, 50 个 category)。这个 narrow data pool 造成 generalization 差, domain gap 大。

**核心 insight**: 与其继续 manual annotation, 不如 build 一个 **positive data flywheel**——用 AI tools 自动产生比 human 更精细的 3D annotation。paper 的 Figure 2 直接 evidence 这一点: 他们的 refinement mask 比 ScanNet 官方 annotation 更 sharp, boundary 更 clean。

**为什么这是可行的 (2025 的时间窗)**: 
- DA3 [12] 提供 metric-aware monocular depth prior
- SAM3 [29] 支持 open-vocabulary prompt-driven segmentation
- Gemini3-Pro / Qwen3-VL [13] 提供 open-world semantic reasoning
- 3DGS [38] 提供 differentiable scene representation 做 multi-view consistency regularization

paper 的 bet 是: 这些 tools 组合起来, error 互相 suppress, 可以达到甚至超过 human annotation quality。

参考链接:
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/  
- DA3: https://arxiv.org/abs/2511.10647  
- SAM3: https://arxiv.org/abs/2511.16719  
- ScanNet: http://www.scan-net.org/  
- ScanNet++: https://scannetpp.github.io/

---

## 2. Pipeline 三阶段架构详解

Pipeline 设计遵循一个 **coarse-to-fine, geometry-then-semantic** 的 philosophy, 共三 stages:

### Stage 1: Geometric Optimization (Section 3.1)

**目标**: 把 noisy monocular depth prior 转成 multi-view consistent 3D structure。

**步骤**:
1. 用 Structure-from-Motion (SfM, COLMAP [45]) 估计 camera intrinsics $\mathbf{K}$ 和 extrinsics $\{\Pi_t\}$
2. 用 DA3 在每帧 $I_t$ 上预测 dense depth $\hat{D}_t$
3. 用 $\hat{D}_t$ 初始化 3DGS scene
4. 在 3DGS optimization 中加入 geometric regularization (借鉴 PGSR [44], 2DGS [51], Gaussian Opacity Fields [50], RaDe-GS [48], CityGS-X [52]) 强制 multi-view depth consistency

**直觉**: DA3 是 per-frame prediction, 不同 frame 的 depth 之间 scale / offset 不完全 consistent。直接 back-project 会产生 ghosting (Figure 7 中 LangSplat / M3-Spatial 的可视化就是 evidence)。3DGS 的作用相当于一个 **multi-view geometric bottleneck**: 通过可微 rendering 拉齐不同 view 之间的 depth, 同时 surface reconstruction regularization (planar constraint 等) 把 floaters push 到实际 surface 上。

**关键 ablation (Table 5 ID.1 vs ID.2)**:
| Setting | $P_{25}$ | $R_{25}$ |
|---|---|---|
| DA3 depth only | 0.13 | 0.31 |
| DA3 + 3DGS refine | **0.81** | **0.89** |

这个 lift 巨大 (Precision 提升约 6x), 说明 multi-view consistency 是整个 pipeline 的 bottleneck。**没有这一步, 后续 3D merge 会全部 collapse** (Figure 12 直接展示: 不同 object 由于 depth ghosting 会被错误 cluster 成一个 instance)。

---

### Stage 2: Image-level Perception (Section 3.2)

**步骤**:

**(a) 动态 class-label memory 机制**

从 raw video 中 uniform sample keyframes $\mathcal{J} = \{I_1, \ldots, I_T\}$。对每帧 $I_t$, 用 Gemini3-Pro 生成 caption, 但维护一个 cross-frame memory:

$$\mathcal{M}_t = \mathcal{M}_{t-1} \cup \text{Extract}(I_t)$$

变量解释:
- $\mathcal{M}_t$: 时刻 $t$ 的累积 category label set
- $\mathcal{M}_{t-1}$: 之前所有 frame 累积的 label set
- $\text{Extract}(I_t)$: 从当前 frame caption 中提取的 object categories
- $\cup$: set union, 保留 emerging new categories

**直觉**: 这是 cross-frame semantic consistency 的 anchor。VLM 单帧 prediction 会产生 synonym drift (例如同一把椅子在 frame 1 叫 "wooden chair", frame 5 叫 "dining chair")。memory 把历史 label 当作 in-context prompt 强制 VLM reuse 既有 label, 让后续 SAM3 grounding 跨 frame 命名 consistent。

**(b) SAM3 open-vocabulary instance segmentation**

SAM3 接收 $\mathcal{M}_t$ 作为 text prompt, 在 frame $I_t$ 上输出:
$$O_t = \{(M_k, s_k)\}_{k=1}^{N}$$
- $M_k$: binary mask $\in \{0, 1\}^{H \times W}$
- $s_k$: confidence score $\in [0, 1]$
- $N$: 该 frame 的 instance 数

**(c) 2D-to-3D back-projection (Figure 4 的四步处理)**

数学上, 对 mask $M_k$ 中每个像素 $\mathbf{u} = (u, v)$:

$$\mathbf{P} = D_t(\mathbf{u}) \cdot \mathbf{K}^{-1} \tilde{\mathbf{u}}$$

变量解释:
- $\mathbf{P} \in \mathbb{R}^3$: 3D point in camera coordinate
- $D_t(\mathbf{u})$: refined depth value at pixel $\mathbf{u}$
- $\mathbf{K} \in \mathbb{R}^{3\times3}$: camera intrinsic matrix (fx, fy, cx, cy encoded)
- $\tilde{\mathbf{u}} = [u, v, 1]^\top$: homogeneous pixel coordinate
- $\mathbf{K}^{-1}$: 反投影变换 (本质是把 pixel ray 映射回 3D unit ray, 再乘以 depth 得到 actual 3D point)

但**直接 fit OBB 会有问题**, paper 指出两个 noise source:
- **2D-level**: SAM3 mask boundary 在 object contour 附近 misalign
- **3D-level**: depth discontinuity 处产生 outlier points (floaters)

paper 的四步 filtering (Figure 4):
1. **Initial depth**: combine 3DGS rendering + SAM3 mask
2. **Mask erosion**: 对 mask contour 做 morphological erosion, 只保留 interior 高置信区域
3. **Mesh-guided depth filtering**: 用 multi-view consistent mesh depth 作为 guide, filter 不一致 pixel
4. **OBB estimation**: 在 refined point cloud 上 fit OBB, 同时保留 original 2D mask + confidence + source image index

**直觉**: 这一步是 **conservative precision-first**。宁可少用一些 pixel, 也要保证 fit 出来的 OBB 不被 outlier 拉歪。后续 Scene-level Refinement 会再做 merge 来 recover completeness。

**(d) Floor-aligned OBB post-processing (Figure 5)**

per-instance OBB 的 roll/pitch 可能 inconsistent (depth noise + partial observation 造成)。paper 用 floor detection 推 global up-axis, 然后 yaw-lock + PCA fallback 把所有 OBB 重 align 到 floor-consistent frame。这是为了后续 3D IoU 计算的 axis alignment。

---

### Stage 3: Scene-level Refinement (Section 3.3)

这是 paper 最 interesting 的部分。从 2D-to-3D lifting 出来的 proposal 集合:

$$\mathcal{P}_{\text{init}} = \{(B_i, c_i, s_i)\}_{i=1}^{M}$$

- $B_i \in \mathbb{R}^7$: 3D OBB parameters (center 3D + size 3D + yaw 1D)
- $c_i$: semantic category
- $s_i$: confidence score
- $M$: 总 proposal 数

**(a) Multi-view Merge**

对每对 proposal $p_i, p_j$, 如果满足:

$$c_i = c_j \quad \wedge \quad \text{IoU}_{3D}(B_i, B_j) > \tau_{\text{merge}}$$

变量解释:
- $c_i = c_j$: 同 semantic category
- $\text{IoU}_{3D}(B_i, B_j)$: 两个 3D OBB 的 intersection-over-union volume
- $\tau_{\text{merge}} = 0.2$: 阈值 (paper Section 3.3 设为 0.2)

则 merge。merge 后的 instance 取 max confidence:
$$s_k = \max(s_i, s_j)$$

并保留产生 max confidence 的那一帧 image index $t^*$ (作为后续 caption 的 canonical view)。

**直觉**: SAM3 单帧会把同一个 object (尤其 large object 如 sofa) 切成多个 fragment。3D 空间中按 IoU clustering 可以把 cross-view 看到同一物体的 fragments 合并。**关键是阈值 0.2 比较低**: 因为 partial observation 时, 不同 view 的 OBB 可能 overlap 不大但确实是同一物体。

**(b) Floor-aligned post-processing (Figure 5)**

如前所述, 对 merged 后的 OBB 做 global gravity alignment。

**(c) Tri-level Confidence Gating (核心创新点)**

对每个 merged proposal $p_k$:

$$\text{Action}(p_k) = \begin{cases} 
\text{keep}, & s_k \geq \tau_{\text{high}} \\
\text{discard}, & s_k < \tau_{\text{low}} \\
\text{verify (VLM agent)}, & \tau_{\text{low}} \leq s_k < \tau_{\text{high}}
\end{cases}$$

阈值: $\tau_{\text{high}} = 0.9$, $\tau_{\text{low}} = 0.8$。

对 verify band 的 proposal, 调用 VLM agent, 该 agent 装备两个 tool:
- **image zoom-in tool**: 裁剪 canonical view 中的 instance 区域 zoom in 给 VLM
- **SAM3 re-segmentation tool**: 在 zoom-in image 上重新跑 SAM3 验证 mask quality

VLM 重新 score $s_k'$, 如果 $s_k' \geq \tau_{\text{high}}$ 则保留, 否则 discard。

**直觉 (关键)**: 一个简单的 binary threshold (e.g., 0.85 cut-off) 会陷入 precision-recall dilemma:
- 阈值高 → precision ↑ 但 recall ↓ (丢掉 hard positive 例如 occluded object)
- 阈值低 → recall ↑ 但 precision ↓ (false positive 例如背景被误分类)

paper 把 score 在 [0.8, 0.9] 的 "灰区" 用 VLM agent 做 fine-grained 二次判断。agent 可以用 zoom-in tool 看 detail, 用 SAM3 re-segmentation 验证 mask alignment。**这是 system 2 reasoning 替代 system 1 heuristic 的思路**。

Ablation Table 5 直接 evidence:

| ID | Conf Filter | Agent Recall | $P_{25}$ | $R_{25}$ |
|---|---|---|---|---|
| 3 | ✗ | ✗ | 0.35 | 0.74 |
| 4 | ✓ | ✗ | 0.67 | 0.69 |
| 5 | ✓ | ✓ | **0.81** | **0.89** |

只加 confidence filter (ID.3→4): P 从 0.35→0.67 (precision 翻倍), R 从 0.74→0.69 (recall 降)。加 agent (ID.4→5): R 从 0.69→0.89 (恢复 +20%), P 维持 0.81。**两者互补**: filter 拿 precision, agent 救 recall。

**(d) Dense semantic annotation 生成**

对最终 $\mathcal{P}_{\text{final}}$, 用 Qwen3-VL-30B 在 canonical view $I_k^*$ 上生成 fine-grained caption, 然后用 template 合成 spatial QA pairs。

---

## 3. Holi-Spatial-4M Dataset 统计

数据源: ScanNet [28] + ScanNet++ [27] + DL3DV-10K [31]

| Modality | 数量 |
|---|---|
| Optimized 3DGS scenes | 12K |
| 2D instance masks | 1.3M |
| 3D bounding boxes | 320K |
| Instance captions | 320K |
| 3D grounding pairs | 1.2M |
| Spatial QA pairs | 1.25M |
| **Total annotations** | **4M+** |

**QA taxonomy** (Figure 6(3)):
- **Camera-centric** (egocentric): camera rotation, movement direction, movement distance
- **Object-centric** (allocentric): 
  - object-to-object distance
  - object-to-object direction (local / global frame)
  - camera-object direction / distance
  - object measurement (size)

总计 10 类 QA (Figure 16 给出 example)。

**Open-vocab coverage**: 用 word cloud (Figure 6(1)) 展示 long-tailed categories。比起 ScanNet 的 50 类 closed-set, 这是数量级的提升。

---

## 4. Experiment 详细分析

### 4.1 Framework Evaluation (Table 2)

paper 在 ScanNet, ScanNet++, DL3DV 上各 sample 10 scenes 做 manual GT 重新 annotation (因为原 GT 是 closed-vocab)。

**3D Object Detection** (AP@25, AP@50):

ScanNet++ 上:
| Method | AP25 | AP50 |
|---|---|---|
| SpatialLM | 9.11 | 6.23 |
| LLaVA-3D | 12.2 | 4.80 |
| SceneScript | 9.86 | 4.42 |
| **Holi-Spatial** | **81.06** | **70.05** |

**这是 ~10x 的 gap**。直觉上: 3D-VLM methods 是 single-view (or anchor view) 输入 + 单次 prediction; Holi-Spatial 是 multi-view lift + 3DGS refine + cross-view merge + VLM verify, 把所有 SOTA tool 串起来, 输入信息更丰富。

**Depth F1**:

ScanNet++ 上:
| Method | F1 |
|---|---|
| LangSplat | 0.21 |
| M3-Spatial | 0.39 |
| **Holi-Spatial** | **0.89** |

Figure 7 可视化展示 Holi-Spatial 的 point cloud 几乎没有 ghosting, 而 baseline 严重。

**2D Segmentation IoU**:
| Method | ScanNet++ IoU |
|---|---|
| SAM3 | 0.50 |
| SA2VA | 0.25 |
| **Holi-Spatial** | **0.64** |

paper 解释 (Section 5.1): SAM3 在单帧上 miss distant mirror, 但 Holi-Spatial 利用 multi-view 信息补全。

---

### 4.2 VLM Finetuning Evaluation

#### Spatial Reasoning QA (Table 3)

在 Holi-Spatial-4M 的 1.2M QA pairs 上 finetune Qwen3-VL, 1 epoch, batch size 1024, 32× H800 GPU。

| Model | MMSI-Bench | MindCube |
|---|---|---|
| VST-SFT-7B [26] | 32.0 | 39.7 |
| Cambrian-S-7B [3] | 25.8 | 39.6 |
| Intern3-VL-8B [57] | 28.0 | 41.5 |
| Qwen3-VL-2B [13] | 26.1 | 33.5 |
| Qwen3-VL-2B + Ours | 27.6 | **44.0** |
| Qwen3-VL-8B [13] | 31.1 | 29.4 |
| **Qwen3-VL-8B + Ours** | **32.6** | **49.1** |

MindCube 上 Qwen3-VL-8B + Ours 比 baseline 提升 ~20 个点 (29.4→49.1), 这个 lift 极大。MMSI-Bench 上 8B 模型 +1.5, 改善相对 modest, 可能因为 MMSI-Bench 已经 saturated (baseline 31.1 已经很高)。

#### 3D Grounding (Table 4)

在 Holi-Spatial-4M 的 1.2M grounding pairs 上 finetune Qwen3-VL-8B, ScanNet++ 评估:

| Method | AP15 | AP25 | AP50 |
|---|---|---|---|
| VST-7B-SFT [26] | 17.29 | 14.50 | 11.20 |
| Qwen3-VL-8B [13] | 19.82 | 16.80 | 13.50 |
| **Qwen3-VL-8B + Ours** | **35.52** | **31.94** | **27.98** |

AP50 提升 14.48 点。Figure 11 展示 baseline model 有 viewpoint bias (训练在 anchor view), 跨 view 和 depth 都 fail; finetune 后 spatial localization 明显改善。

---

### 4.3 Ablation Study (Section 5.3 + Table 5)

paper 把 pipeline 拆成 Step 1 (Geometric Optimization) 和 Step 3 (Scene-Level Refinement) 两部分做 ablation:

**Step 1**: DA3 depth vs DA3 + 3DGS refine
| ID | DA3 Depth | 3DGS Training | $P_{25}$ | $R_{25}$ |
|---|---|---|---|---|
| 1 | ✓ | ✗ | 0.13 | 0.31 |
| 2 | ✓ | ✓ | 0.81 | 0.89 |

这是 **pipeline 中最 critical 的一步**。没有 3DGS refine, P/R 都很低, 整个 downstream 都 collapse。

**Step 3**: Confidence filter + Agent recall (见上文 Table 5 ID.3-5)

Figure 10 给出 stage-wise 可视化, paper 用 shelf/curtain case 说明 DA3 直接 depth 的 ghosting 把一个 instance 切成多个; confidence filter 把 misclassified vending machine 滤掉; agent recall 把 hair dryer / cart 这些 hard positive 救回来。

---

## 5. Algorithm 1 完整伪代码 (Appendix D)

paper 在 Appendix D 给出完整 algorithm, 三个阶段:

```
Input: keyframes {I_t}, camera params {Π_t}, refined depth {D_t}, per-frame label sets {L_t}
       SAM3(I_t, ℓ) → {(m, s)}  // mask + confidence
       thresholds: τ_iou, τ_low < τ_high

Stage 1: Lift 2D instances to 3D candidates (per label)
  for each frame t:
    for each label ℓ ∈ L_t:
      SAM3 → {(m, s)}
      for each (m, s):
        P = BackProject(m, D_t, Π_t)
        B = BBox(P)
        C_ℓ ← C_ℓ ∪ {(P, B, s, t, m)}

Stage 2: Multi-view merging within each label (3D IoU)
  for each label ℓ with C_ℓ ≠ ∅:
    G_ℓ = MergeByIoU3D(C_ℓ, τ_iou)

Stage 3: Confidence gating and VLM-based verification
  for each merged group g with canonical view c*:
    s* = max confidence in group
    if s* ≥ τ_high: keep
    elif s* < τ_low: discard
    else: VLMVerify(I_t*, m*, ℓ) → keep if true
  return O
```

---

## 6. Limitations & Impact (Section 6 + Impact Statement)

**Limitations**:
1. Pipeline 依赖 multiple upstream components, 每一个都可能 fail (challenging videos: limited viewpoints, motion blur, heavy occlusion, dynamic objects)
2. Per-scene optimization 计算开销大
3. Open-vocab semantic labeling 继承 foundation model 的 biases / errors

**Future work**:
1. Adaptive early stopping
2. Better confidence-based validation
3. Expand 到 broader domains 和 longer video contexts
4. Stronger benchmarks for holistic 3D spatial understanding

**伦理考量**: Pipeline 可以 reconstruct personal spaces, 有隐私风险。建议 consent + data governance。

---

## 7. Build Intuition: 几个深层思考

### 7.1 为什么 "tools composition" 能超过 single end-to-end model?

paper 的 evidence 是 AP50 提升 64% / 10x。深层原因:
- **Modular inductive bias**: 每个子任务 (depth, segmentation, caption) 都有专属的 SOTA model, 各自 exploit 不同的 data distribution (depth 用大规模 monocular video, SAM3 用 billion-mask data, VLM 用 web text-image pairs)
- **Error decorrelation**: end-to-end model error 是 coupled 的; modular pipeline 的 error 在每个 stage 互相 suppress (e.g., VLM 给 right label, SAM3 给 right mask, 3DGS 给 right depth, 三者 intersect 就 high precision)
- **Verifiability**: 每个 stage 输出可 inspect, 便于 debug 和 ablation

**这是 system 2 的 engineering philosophy**,  类似 AlphaProof 把 prover + LLM 组合的方式。

### 7.2 为什么 VLM agent 比单一 confidence threshold 强?

直觉: confidence score 是 SAM3 输出的 **scalar summary**, 丢掉了 spatial / contextual 信息。VLM agent 可以重新 look at the image, 用 zoom-in tool 看 detail, 用 SAM3 re-segmentation 验证 mask shape, 这些都是 scalar confidence 无法 capture 的。

这本质是 **delay decision to higher-resolution processor** 的思想, 类似 cascaded classifiers, 但用 LLM agent 做 semantic-aware cascade。

### 7.3 3DGS 在这里到底干了什么?

直觉上, 3DGS 不只是 rendering 工具, 它是 **multi-view geometric regularizer**。具体地:
- 提供 differentiable scene representation
- 通过 photometric loss (RGB rendering) 拉齐不同 view 的 geometry
- 通过 surface regularization (planar / normal constraint) 把 floaters push 到 surface
- Output 是 multi-view consistent depth $D_t$ at any view

paper 在 Table 5 中 evidence: 这一步的 P/R 提升最大 (0.13/0.31 → 0.81/0.89)。**整个 pipeline 的 backbone 是 3DGS**, 其他 components 都是给它服务。

参考链接:
- PGSR (surface reconstruction GS): https://arxiv.org/abs/2406.06521
- 2DGS: https://arxiv.org/abs/2403.17888
- Gaussian Opacity Fields: https://arxiv.org/abs/2404.10772

### 7.4 与相关工作的定位

Table 1 给出 capability comparison:
- 2D-VLM (SAM3, SA2VA): 只支持 2D seg
- 3D-VLM (SpatialLM, LLaVA-3D, SceneScript): 需要 point cloud input
- 3DGS-based (M3-Spatial, LangSplat): 只能 per-scene optimize,  speed 差
- **Holi-Spatial**: 唯一同时支持 depth + 2D seg + 3D det + grounding + spatial QA, 且输入只需 images (不需要 point cloud)

这意味着 Holi-Spatial 可以 scale 到 web video, 不需要 specialized 3D sensors。

---

## 8. 可能的扩展与潜在联想

1. **Video-scale extension**: 当前 paper 在 ScanNet / ScanNet++ / DL3DV 上 (相对短 video), 扩展到 YouTube-scale long video 会遇到: memory explosion, dynamic objects (moving people / vehicles), domain shift (lens distortion, lighting)
2. **Action annotation**: 当前 QA 只覆盖 spatial reasoning, 没覆盖 temporal action。可结合 action recognition model 扩展
3. **4D reconstruction**: dynamic scene (non-rigid deformation) 是 3DGS 的 frontier,  paper 的 pipeline 假设 static scene
4. **Robotics downstream**: Section 1 提到 robotics manipulation [21, 22] 和 navigation [16] 是 target application。Holi-Spatial-4M 直接作为 VLA training data 是一个 obvious next step
5. **Self-distillation flywheel**: paper 提到 "positive data flywheel" 但没 implement。可以想象: 用 Holi-Spatial-4M finetune VLM → 更强的 VLM 替换 pipeline 中的 Gemini3-Pro → 生成更高质量 dataset → iteratively improve

---

## 9. 总结

Holi-Spatial 的核心 contribution 不是新 architecture, 而是 **system composition + multi-view refinement + agent verification** 的工程哲学。它把 spatial intelligence 的 bottleneck 从 "data scarcity" 转化成 "pipeline engineering", 用 4M 自动生成的高质量 annotation (远超 human annotation quality) 来 finetune VLM。

最有意思的几个 insight:
1. **3DGS as multi-view depth regularizer** (而不是 rendering tool) — Table 5 evidence
2. **Memory-based cross-frame label consistency** — 简单但 effective
3. **Tri-level confidence gating + VLM agent** — system 2 替代 heuristic threshold
4. **Coarse-to-fine pipeline design** — 先 geometric precision, 再 semantic granularity

最终结果在 ScanNet++ 上 AP25 从 12.2 提升到 81.06, 这是数量级 jump, 说明 modular composition 在当前 foundation model era 是 highly competitive strategy。

参考:
- Project page: https://visionary-laboratory.github.io/holi-spatial/
- Code: https://github.com/Visionary-Laboratory/Holi-Spatial
- Contact: zhongzhihang95@gmail.com
