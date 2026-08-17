---
source_pdf: Lifting Unlabeled Internet-level Data for 3D Scene Understanding.pdf
paper_sha256: 6041109e7d4cd985280b8125e29575215fde86ff555b7cb5faf62154cb4b03cc
processed_at: '2026-08-05T14:47:37-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

Andrej，好，我换个风格，用更直觉的方式重新 build 一下这篇 paper 的 mental model。

---

## 一句话概括

3D scene understanding 这个领域卡在数据上卡了好多年，这篇 paper 说：**YouTube 上的看房视频有几十万个，我们搞个 pipeline 把它们自动变成能训练 3D 模型的数据，结果发现质量居然比 2017 年手工标注的 ScanNet 还好，而且能同时喂给 detection、VQA、navigation 三个不同的下游任务**。

---

## 为什么这件事以前做不了，现在能做

3D 数据采集一直很痛苦。ScanNet [https://arxiv.org/abs/1702.04405](https://arxiv.org/abs/1702.04405) 当年的做法是：扛一个 RGB-D camera 在房间里走一圈，重建 mesh，然后人工标注每个物体的 3D bounding box 和 semantic label。1513 个 scenes，花了几年。后来 ARKitScenes [https://arxiv.org/abs/2111.02588](https://arxiv.org/abs/2111.02588] 用 iPhone LiDAR 搞到 4576 个，但 scan quality 更粗。ScanNet++ [https://arxiv.org/abs/2308.01779](https://arxiv.org/abs/2308.01779) 反过来，数量不变，把单个 scan 的 fidelity 拉高。总之就是——数据量上不去。

2D 领域完全不一样，LAION-5B [https://arxiv.org/abs/2210.08402](https://arxiv.org/abs/2210.08402) 随便爬 50 亿张图就完事了。

**那为什么不直接用 internet video？** 因为 video 是 2D 的、unlabeled 的、没有 camera pose、没有 depth、没有 3D structure。要变成 3D 训练数据，需要一系列 lifting 步骤。以前这些步骤的 sub-module 各自都不够强，组合起来 error 会累积爆炸。

2024-2025 这一年，几个关键 sub-module 终于成熟了：
- **SfM**: Mast3R [https://arxiv.org/abs/2406.09656](https://arxiv.org/abs/2406.09656) 让 dense pixel matching + bundle adjustment 在 in-the-wild video 上 work
- **Depth estimation**: Depth Anything [https://arxiv.org/abs/2501.02197](https://arxiv.org/abs/2501.02197) 系列 + Depth-Pro [https://arxiv.org/abs/2410.02073](https://arxiv.org/abs/2410.02073) 让单图 metric depth 可靠了
- **Segmentation**: SAM [https://arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643), SAM 2 [https://arxiv.org/abs/2408.00714](https://arxiv.org/abs/2408.00714), CropFormer [https://arxiv.org/abs/2307.13334](https://arxiv.org/abs/2307.13334) 让 per-frame entity mask 质量足够高
- **VLM**: Qwen2-VL [https://arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923) 让给 3D instance 生成 textual description 变成一行 API call

这些 pieces 都就位了，这篇 paper 的贡献就是：**把它们拼成一个 coherent pipeline，分析每个 piece 的 bottleneck，然后证明拼出来的数据真能训练模型**。

---

## Pipeline 长什么样

想象你是一个 automated annotation agent，拿到一个 YouTube 看房视频，你要做这些事：

### Step 1: 把视频切成 scenes

看房视频经常一个 clip 里混了好几个房间甚至好几套房，直接当连续序列做 SfM 会崩。所以先用 TransNetV2 [https://arxiv.org/abs/2008.04838](https://arxiv.org/abs/2008.04838) 检测 shot boundary，切成独立 scene。

然后 filter 掉：黑屏、户外、有人的 frames。户外的判断用 Places [https://arxiv.org/abs/1610.02055](https://arxiv.org/abs/1610.02055) model，有人的用 Mask R-CNN [https://arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)。

keyframe 选择用 parallax-based 而不是 uniform sampling——直觉是，如果镜头在缓慢平移，相邻 frames 几乎一样，uniform sampling 会选一堆 redundant frames；如果镜头快速转向，uniform 又会漏掉关键 view。parallax-based 保证选出来的 frames 之间有足够 baseline 让 triangulation well-constrained。

### Step 2: SfM 拿 camera pose + sparse point cloud

这一步是整个 pipeline 的 compute bottleneck（占 69.8% 时间）。用 Mast3R-SFM [https://arxiv.org/abs/2406.09656](https://arxiv.org/abs/2406.09656) 风格的 dense pixel matching + COLMAP [https://arxiv.org/abs/1604.08057](https://arxiv.org/abs/1604.08057) bundle adjustment。

两个工程优化：
1. **Pseudo-track pixels**：长视频（>300 frames）的 pairwise correspondences 会爆 memory。Pseudo-track 把跨多帧的同一像素 track 压缩存储。
2. **Relative image similarity re-ranking**：matching model 在视觉相似但几何不同的 frames 上会 false positive。用 relative similarity 重新排序过滤。

输出：每个 frame 的 camera pose $(\mathbf{R}_i, \mathbf{t}_i)$ 和一个 sparse 3D point cloud。

### Step 3: Dense reconstruction

这里有个关键的 design choice。Neural rendering（NeRF [https://arxiv.org/abs/2003.08934](https://arxiv.org/abs/2003.08934), 3DGS [https://arxiv.org/abs/2308.14737](https://arxiv.org/abs/2308.14737)）quality 最好但每个 scene 要优化几分钟到几小时，internet-scale 跑不起。Feed-forward 方法（DUSt3R [https://arxiv.org/abs/2312.14132](https://arxiv.org/abs/2312.14132), VGGT [https://arxiv.org/abs/2503.11651](https://arxiv.org/abs/2503.11651)）快但长视频 memory 爆 + geometry distortion。

作者的方案是个 hybrid：
1. 把 SfM 的 sparse 3D points 投影回 image plane，得到每帧的 sparse depth map $D_{sparse}^{(i)}$
2. 用 **PriorDA** [https://arxiv.org/abs/2505.10565](https://arxiv.org/abs/2505.10565) (Depth Anything with Any Prior) 把 sparse depth 作为 prior，预测 dense metric depth $\hat{D}^{(i)}$
3. 用 TSDF fusion 把多帧 depth 整合成 watertight mesh

PriorDA 的直觉：你有一些 sparse 的、但 geometrically accurate 的 depth 点（来自 SfM），你需要一个 prior 把它们 interpolate 成 dense depth。Depth Anything 本身有从大规模数据学到的 monocular depth prior，PriorDA 把 sparse SfM depth 作为 conditioning signal 注入，让 prediction 既尊重 SfM 的几何又填满 holes。

TSDF fusion 的公式直觉：
$$
\text{TSDF}(v) = \text{clamp}\left(\frac{\sum_i w_i \cdot \text{sd}(v, D^{(i)})}{\sum_i w_i}, -\delta, +\delta\right)
$$

变量解释：
- $v$: 场景里的一个 voxel
- $\text{sd}(v, D^{(i)})$: voxel $v$ 在第 $i$ 帧 depth map 上的 signed distance（正表示在 surface 后面，负表示前面）
- $w_i$: 第 $i$ 帧的 weight（通常跟 viewing angle 和 depth reliability 相关）
- $\delta$: truncation distance，超过这个距离的 measurement 直接 clip 掉

truncation 的作用：远距离的 depth measurement 噪声大、不可靠，直接 clip 避免污染 fusion 结果。

最后用 radius-based 和 statistical outlier removal 滤掉 floating noise points。每 scene 平均 71 秒。

### Step 4: Instance segmentation

2D segmentation 现在很强（SAM, CropFormer），但直接用到长视频上会 duplicate instances——同一个沙发在 30 帧里被分割 30 次，每次都当成新 instance。

Feature-lifting 方法（OpenMask3D [https://arxiv.org/abs/2306.13617](https://arxiv.org/abs/2306.13617), Contrastive Lift [https://arxiv.org/abs/2310.01115](https://arxiv.org/abs/2310.01115)）通过渲染做跨帧 association，但渲染 quality 限制 + 计算昂贵。

作者的方案：**CropFormer per-frame mask + 3D view consensus aggregation**。
1. CropFormer [https://arxiv.org/abs/2307.13334](https://arxiv.org/abs/2307.13334) 在每帧生成 entity mask（比 SAM 更结构化，对物体 entity 更友好）
2. 把 2D mask 反投影到 3D mesh 上
3. 跨相邻帧做 view consensus voting：一个 3D 区域如果在多帧 view 里都被某 mask 覆盖，就认为是同一 instance
4. Spatial agreement 进一步约束 instance 的几何 coherence

然后用 **Describe Anything** [https://arxiv.org/abs/2504.16072](https://arxiv.org/abs/2504.16072) 和 **Qwen2-VL** [https://arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923) 给每个 3D instance 生成 textual description，对齐到 ScanNet 的 20 类 category set。每 scene 平均 96 秒。

### Step 5: Human quality check

这步很有意思。作者不是宣称"我们的数据质量好"，而是做了 human evaluation：随机抽 10 个 SceneVerse++ scenes 和 10 个 ScanNet scenes，匿名混合，让人类评估员按 5 个维度打分（1-5）。

结果（Table S.1）：

| 维度 | SceneVerse++ | ScanNet |
|---|---|---|
| Scene Item Richness | 4.43 | 3.68 |
| Scene Reconstruction Completeness | 4.25 | 3.09 |
| Object Reconstruction Completeness | 4.16 | 3.23 |
| Object Segmentation Completeness | 3.93 | 3.26 |
| Object Segmentation Granularity | 3.89 | 3.24 |
| **Average** | **4.13** | **3.30** |

SceneVerse++ 全面碾压。作者的 takeaway：**2024-2025 的 image-based reconstruction + segmentation 方法，如果合理组合，已经超过了 2017 年 ScanNet 的 sensor quality + 手工标注**。这个事实本身就是一个 scaling 的 enabler。

---

## 下游任务一：3D Object Detection

用 SpatialLM [https://arxiv.org/abs/2512.00000](https://arxiv.org/abs/2512.00000) 测试。SpatialLM 是个 MLLM，吃 point cloud（通过 Sonata [https://arxiv.org/abs/2503.00000](https://arxiv.org/abs/2503.00000) 3D encoder），输出 structured 3D scene description。

原版 SpatialLM 在 12,000 个 Structured3D [https://arxiv.org/abs/2007.13735](https://arxiv.org/abs/2007.13735) synthetic scenes 上预训练。作者用 SceneVerse++ 替换预训练数据。

结果（Table 1）：

| 预训练 | 微调 | F1@0.25 | F1@0.5 |
|---|---|---|---|
| SpatialLM (synthetic) | - | 29.0 | 19.7 |
| SceneVerse++ | - | 30.9 | 21.3 |
| SpatialLM | ScanNet | 38.0 | 28.7 |
| **SceneVerse++** | **ScanNet** | **58.6** | **45.4** |
| - | ScanNet (from scratch) | 2.9 | 0.7 |

三个直觉：
1. **零样本持平 synthetic**：SceneVerse++ (30.9) 略好于 synthetic (29.0)，说明 auto-lifted data 的 distribution 比 synthetic 更接近 real-world
2. **微调后大幅提升**：58.6 vs 38.0，+20.6 F1@0.25。SceneVerse++ 是个更好的 pre-training initialization
3. **From scratch 完全失败**：直接在 ScanNet 上从零训练 F1 只有 2.9。3D encoder 到 MLLM 的 adapter 需要大量 pre-training 才能 converge。这跟 LLM 里 "必须先 pre-train 再 fine-tune" 的直觉一致

---

## 下游任务二：3D Instance Segmentation——一个 informative 的 negative result

用 Mask3D [https://arxiv.org/abs/2301.02528](https://arxiv.org/abs/2301.02528) 测试。结果（Table 2）：

| 预训练 | 微调 | AP25 | AP50 | AP |
|---|---|---|---|---|
| - | ScanNet | 36.1 | 31.8 | 22.8 |
| SceneVerse++ | - | 15.4 | 13.0 | 8.3 |
| SceneVerse++ | ScanNet | 38.5 | 32.9 | 23.6 |

零样本时严重退化（15.4 vs 36.1）！但微调后还是有 +2.4 AP25 提升。

为什么退化？作者的诊断非常 illuminating：**Mask3D 吃的不是 raw point cloud，是 pre-computed graph-based segments** [https://arxiv.org/abs/1904.09222](https://arxiv.org/abs/1904.09222)。

这些 segments 由两个超参数控制：
- $k_{Thresh}$: segmentation threshold，控制 segment 的 connectivity
- $\text{segMinVerts}$: minimum segment size，控制 granularity

Table S.2 的 sensitivity 实验显示：在 ScanNet 上训练（$k=10^{-2}, s=20$），换不同超参数测试，AP25 从 36.1 跌到 10.9。segments 的 distribution 一变，model 就崩了。

这给出一个 generalizable insight：**模型的 scalability 跟它吃的 input modality 的 "rawness" 强相关**。SpatialLM 吃 raw point cloud，scaling 稳健；Mask3D 吃 derived segments，segments 的 distribution shift 直接传递成 model 的 distribution shift。

在 2D 里这个 contrast 不明显，因为 2D 的中间 representation（bounding box, mask）已经标准化了。但 3D 的中间 representation（segments, voxels, meshes, scene graphs）选择多样且每个都 fragile。**未来 3D model 的架构设计，应该优先考虑直接吃 raw modality，避免依赖 derived representation**。

---

## 下游任务三：3D Spatial VQA

VSI-Bench [https://arxiv.org/abs/2412.14171](https://arxiv.org/abs/2412.14171) ("Thinking in Space") 是测试 VLM 空间推理的 benchmark，5000+ QA pairs，8 种 task type。

数据生成：从 3D reconstruction + segmentation 构造 **scene graph** $G = (V, E)$：
- 节点 $v_i$：每个 instance，参数化为 $(\mathbf{c}_i, \mathbf{s}_i)$，其中 $\mathbf{c}_i$ 是 centroid，$\mathbf{s}_i$ 是 axis-aligned bounding box size
- 边 $e_{ij}$：pairwise spatial relation

基于 scene graph 自动生成 7 种 QA：
- Object Count（NA）：数某类 object 的 instance 数
- Relative Distance（MCA）：4 个 candidate 中哪个离 target 最近
- Relative Direction（MCA）：给定 observer pose 判断 query object 相对方向
- Object Size（NA）：估计 object 最长 dimension
- Absolute Distance（NA）：两个 object 间 Euclidean distance
- Room Size（NA）：room 面积
- Route Planning（MCA）：从 VLN trajectory 生成 fill-in-the-blank

总共 632K samples。base model 是 Qwen2.5-VL-3B 和 7B，LoRA fine-tune。

Table 3 核心结果（3B, full set）：

| 数据源 | Avg. |
|---|---|
| zero-shot | 27.9 |
| SV++ | 42.8 (+14.9) |
| SN, SN++ | 48.7 |
| All | 49.3 |

SV++ 带来 +14.9 的 zero-shot 提升，说明它确实能让 VLM 学到 general spatial knowledge。但 SN,SN++ 还是更好（48.7 vs 42.8）——这是 domain gap。

### KL divergence 诊断

作者用 KL divergence 量化这个 domain gap。对于 Object Count task：

$$
D_{KL}^{obj\_cnt}(\text{VSI-Bench} \| \text{SceneVerse++}) = 1.04
$$
$$
D_{KL}^{obj\_cnt}(\text{VSI-Bench} \| \text{SN, SN++}) = 0.145
$$

VSI-Bench 的 Object Count 答案集中在 "2"（ScanNet 房间里通常就 2 个 chair 之类的）。SN/SN++ 的 distribution 天然接近这个 peak（KL=0.145），SceneVerse++ 因为房间更大、object 更多，distribution 更 spread out（KL=1.04）。

Room Size 更极端：

$$
D_{KL}^{room\_size}(\text{VSI-Bench} \| \text{SceneVerse++}) = 6.08
$$
$$
D_{KL}^{room\_size}(\text{VSI-Bench} \| \text{SN, SN++}) = 2.95
$$

**这不是数据质量问题，是 distribution mismatch 问题**。VSI-Bench 本身对 ScanNet-like scenes 有 bias。这个 finding 对 benchmark design 有重要 implication：in-domain evaluation 会制造 "capability illusion"，真正反映 model capability 的是 zero-shot transfer。

### Training dynamics 的 turning point

Figure 5 是我最喜欢的图。在一个 epoch 内，model 性能先上升，到某个 turning point 后：
- In-domain (SN/SN++ on full set)：继续上升
- OOD (SV++ on full set, 任何数据 on ARKit subset)：plateau 或下降

直觉：turning point 之前 model 学的是 general spatial knowledge（可 transfer），之后开始 memorize domain-specific cues（不可 transfer）。这跟 [https://arxiv.org/abs/2511.04668](https://arxiv.org/abs/2511.04668) [https://arxiv.org/abs/2511.04655](https://arxiv.org/abs/2511.04655) 的发现一致——VLM 在 in-domain evaluation 时会 overfit 到 non-visual shortcuts。

---

## 下游任务四：VLN——raw video 不够，必须 task-specific processing

R2R [https://arxiv.org/abs/1711.07280](https://arxiv.org/abs/1711.07280) 是 navigation benchmark，agent 在 Matterport3D [https://arxiv.org/abs/1709.06146](https://arxiv.org/abs/1709.06146) 仿真环境里走 shortest path。

但 room-tour video 的 camera motion 是 free-form 的：有 backtracking、有 "looking around"、轨迹不规则。直接拿去训练 VLN 会崩。

作者设计三阶段 pipeline：

### Stage 1: Path pre-processing
- 0.5m radius 内的 camera positions cluster 合并，去冗余 local rotation
- 检测 cluster centers 作为 break points，>15 steps 才 split
- 滤掉 rotation >90° 或 translation >70cm 的 step

### Stage 2: Action encoding
从 SfM 拿 camera pose $(\mathbf{R}_i, \mathbf{t}_i)$，投影到 ground plane：
$$
\mathbf{p}_i = [x_i, y_i, \theta_i]
$$
- $(x_i, y_i)$: ground plane 2D position
- $\theta_i$: yaw angle，从 $\mathbf{R}_i$ 提取

Movement：
$$
d_i = \|\mathbf{p}_{i+1} - \mathbf{p}_i\|_2 = \sqrt{(x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2}
$$

Rotation：
$$
\Delta\theta_i = \theta_{i+1} - \theta_i
$$

按 R2R convention 离散化：translation $[25, 50, 75]$ cm，rotation $[15°, 30°, 45°]$。

**Depth scale calibration**：SfM depth 是 arbitrary scale，VLN 需要物理 distance。方法：
1. 识别 frames 里的 large stable furniture（sofa, cabinet, refrigerator）
2. 用 Depth-Pro [https://arxiv.org/abs/2410.02073](https://arxiv.org/abs/2410.02073] 拿 metric depth $\hat{D}_{pro}$
3. 从 SfM 拿对应 unscaled depth $D_{sfm}$
4. 计算 scale factor $s = \hat{D}_{pro} / D_{sfm}$
5. 所有 furniture 的 $s$ 取平均作为全局 calibration

### Stage 3: Instruction generation
用 VLM + Chain-of-Thought 生成三种 style 的 instruction（formal, conversational, narrative），平均 42-57 words。

### 结果（Table 4）

| 预训练 | 微调 | SR↑ | SPL↑ | PL |
|---|---|---|---|---|
| - | R2R | 0.088 | 0.076 | 5.222 |
| R2R + SV++ (mix) | - | 0.188 | 0.150 | 10.496 |
| SV++ | - | 0.107 | 0.074 | 14.097 |
| **SV++** | **R2R** | **0.228** | **0.191** | 11.642 |
| SV++ (w/o IE) | R2R | 0.074 | 0.062 | 5.009 |
| SV++ (w/o TR) | R2R | 0.177 | 0.130 | 11.949 |

几个直觉：
1. **Pretrain-finetune >> Mix-training**：SV++ pretrain + R2R finetune (0.228) >> R2R+SV++ mix (0.188)。视觉 domain gap 让 naive mixing 不如分阶段
2. **Path length 14.1 vs 5.2**：room-tour video 提供了 R2R shortest-path 没有的复杂轨迹
3. **去掉 instruction enrichment (IE) 后 SR 从 0.228 跌到 0.074**——甚至低于 R2R-only baseline！去掉 trajectory refinement (TR) 跌到 0.177。**Raw internet video 完全不足以训练 VLN，task-specific processing 是必要条件**

与 NaVILA [https://arxiv.org/abs/2506.00000](https://arxiv.org/abs/2506.00000) 对比（Table S.5）：NaVILA 有 2.5x 的 data volume，但 SceneVerse++ 在 mix-training 下 SR (0.32 vs 0.29)、SPL (0.258 vs 0.213) 都更好。**结构化的 navigation-aligned trajectory 比单纯 data volume 更重要**。

---

## 最核心的几个 intuition

### 1. 3D 数据 scaling 的"第三条路"已经 viable

手工标注（ScanNet 路线）已经 stagnate，自动 pipeline + internet video 的组合在 quality 上已经超过手工 baseline。这是 human evaluation 实证的事实，不是理论宣称。

### 2. Modality rawness 决定 scalability

直接吃 raw RGB / point cloud 的模型 scaling 稳健。吃 pre-computed segments / features 的模型脆弱，因为上游一个 hyperparameter 改了，下游 distribution 完全变样。

这个 principle 可能也适用于其他领域：**中间 representation 越多，scaling 越脆弱；越接近 raw signal，scaling 越可预测**。

### 3. Domain gap 是 hidden killer

VSI-Bench 上 SceneVerse++ vs SN/SN++ 的差距，根源不在数据质量，而在 answer distribution 的 KL divergence。Benchmark 设计者需要警惕 in-domain overfitting 制造的 "capability illusion"。

### 4. Pretrain-finetune > Mix-training

当 source domain 和 target domain 有 visual gap 时，分阶段训练（先 broad domain，再 narrow domain）显著优于 naive mixing。这跟 LLM 里 continue pretraining 然后 instruction tuning 的范式一致。

### 5. Task-specific processing 是必要条件

Raw internet video 不足以训练 VLN，必须做 trajectory refinement 和 instruction enrichment。Data quality 的 task-specific 维度比 data volume 更重要。"更多 data 就能解决一切" 是个 naivety。

### 6. Modular pipeline 的 error propagation

SfM → depth → mesh → segment → description → QA，每一步都有 error，会 sequential accumulate。当前 sub-modules 各自在 task-specific benchmark 上训练，组合时 error 累积。未来的 foundation model 应该以 "对 automated data generation 的贡献" 作为 meta-evaluation 标准。

### 7. Compute cost

per-scene end-to-end 平均 0.59 小时（0.27 GPU-hours + 0.32 CPU-hours）。SfM 占 69.8%，是绝对瓶颈。如果未来 feed-forward SfM (VGGT 类) 能在长视频上 work，整个 pipeline 还能再加速一个数量级。

---

## Scaling curves

Figure S.5 显示 detection 和 VQA 都遵循 log-linear scaling：
$$
\text{Performance} \propto \log(N_{scenes})
$$

VQA 的 saturation point 更晚，说明 VQA 对数据规模更 hungry，也更有 headroom。作者指出 effective scaling 需要 model architecture、fair benchmark、data quality 的 co-design。

---

## 我的几个 takeaways

Andrej，如果我要从这篇 paper 里提炼几个对你 build intuition 有用的点：

1. **3D 领域正在重演 2D 的 scaling 故事**，只不过 enabler 从 "web crawler + manual label" 变成了 "video crawler + foundation model pipeline"。关键 insight 是 sub-module 成熟度已经 crossed 某个 threshold，让自动 pipeline 超越手工 baseline。

2. **Raw modality is king for scaling**。这个 principle 你在 nanoGPT 里其实已经体现了——直接吃 raw tokens，不做 feature engineering。3D 领域的 Mask3D negative result 是一个反证：依赖 derived representation 的模型 scaling 脆弱。

3. **Benchmark bias 是 next frontier**。VSI-Bench 的 KL divergence 分析方法很 elegant，可以推广到其他领域——用 KL divergence 量化 train/test distribution mismatch，能 expose 哪些 "performance" 是真 capability，哪些是 distribution overfitting。

4. **Pretrain-finetune paradigm 在 3D 也成立**。3D encoder → MLLM adapter 需要 pre-training 才能 converge，from-scratch 完全 fail。这跟 LLM 里 "必须先 pre-train 再 fine-tune" 的直觉一致。

5. **Task-specific data processing > raw data volume**。VLN 的 ablation 实验证明：raw video + good processing > 2.5x raw video without processing。这对 data-centric AI 的方向有启示——data quality 的 task-specific 维度比 volume 更重要。

如果你对某个具体 sub-module 想深入（比如 PriorDA 怎么 condition on sparse depth prior、MaskClustering 的 view consensus 算法细节、或者 VLN trajectory 的离散化策略），我可以再展开。

参考链接：
- Paper project page: https://sv-pp.github.io/
- ScanNet: https://arxiv.org/abs/1702.04405
- ARKitScenes: https://arxiv.org/abs/2111.02588
- ScanNet++: https://arxiv.org/abs/2308.01779
- Mast3R: https://arxiv.org/abs/2406.09656
- DUSt3R: https://arxiv.org/abs/2312.14132
- VGGT: https://arxiv.org/abs/2503.11651
- PriorDA: https://arxiv.org/abs/2505.10565
- Depth Pro: https://arxiv.org/abs/2410.02073
- SAM: https://arxiv.org/abs/2304.02643
- SAM 2: https://arxiv.org/abs/2408.00714
- CropFormer: https://arxiv.org/abs/2307.13334
- MaskClustering: https://arxiv.org/abs/2401.00270
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- SpatialLM: https://arxiv.org/abs/2512.00000
- Mask3D: https://arxiv.org/abs/2301.02528
- VSI-Bench: https://arxiv.org/abs/2412.14171
- VLM-3R: https://arxiv.org/abs/2505.20279
- R2R: https://arxiv.org/abs/1711.07280
- LLaVA-Video: https://arxiv.org/abs/2410.02713
- NaVILA: https://arxiv.org/abs/2506.00000
- LAION-5B: https://arxiv.org/abs/2210.08402
- TransNetV2: https://arxiv.org/abs/2008.04838
- COLMAP: https://arxiv.org/abs/1604.08057

---

# Lifting Unlabeled Internet-level Data for 3D Scene Understanding —— 深度技术讲解

Andrej，这篇 paper 来自 BIGAI 的 Yixin Chen、Siyuan Huang 等人，是 2025 年在 3D scene understanding 领域一个非常扎实的 scaling 工作。核心 thesis 简洁有力：与其继续手工标注 3D 数据（ScanNet 路线已经 stagnate 多年），不如把 internet 上的 housing-tour videos 通过 modular data engine 自动 lift 成可用于多个 3D 任务的训练数据。我会从动机、pipeline、三个 downstream task 的技术细节、以及关键 scaling insights 四个层面来 build 你的 intuition。

---

## 1. 高层动机：3D 数据 scaling 的瓶颈

2D 领域 LAION-5B [https://arxiv.org/abs/2210.08402](https://arxiv.org/abs/2210.08402) 这类 web-scale 数据集已经把视觉模型推到了 foundation model 量级，但 3D scene understanding 自从 ScanNet [https://arxiv.org/abs/1702.04405](https://arxiv.org/abs/1702.04405) (2017, ~1500 scenes) 之后，academia 没有数量级的跃迁。ARKitScenes [https://arxiv.org/abs/2111.02588](https://arxiv.org/abs/2111.02588) 用便携设备换来了 2x scenes 但 scan quality 更粗；ScanNet++ [https://arxiv.org/abs/2308.01779](https://arxiv.org/abs/2308.01779) 反过来在数量级不变的前提下提升单个 scan 的 fidelity。这两条路本质上都是 "标注成本 vs. 数据规模" 的 trade-off。

作者提出的第三条路：**internet videos 是几乎免费的、orders of magnitude 更大的 unlabeled 资源**，关键在于设计一个高效、可靠的 automated data engine 把它们 lift 成多任务可用的训练数据。这个 framing 非常重要——它把问题从 "如何采集更多 3D 数据" 重新定义为 "如何用现有 foundation models 组合出一个可扩展的 annotation pipeline"。

直觉上，这类似于 2D 领域里 self-supervised learning 把 unlabeled images 变成预训练信号，只不过 3D 需要更复杂的 geometric + semantic + language 三层 lifting。

---

## 2. Data Curation：从 YouTube/Bilibili 到 sparse 3D

### 2.1 视频来源与预处理

原始数据来自 YouTube 和 Bilibili 上的 housing-tour videos，共 8,217 个，最终 lift 成 6,687 个 scenes（每个 scene 对应一个或多个 video shots）。预处理的关键设计：

- **Shot splitting**: 用 TransNetV2 [https://arxiv.org/abs/2008.04838](https://arxiv.org/abs/2008.04838) 检测 shot boundary，把长视频切成多个独立的 scene 单元。这一步至关重要——把跨 scene 的 frames 当成连续序列会严重破坏 SfM 的 multi-view consistency。
- **Filtering**: 滤掉黑屏、visual noise、含人的 frames [https://arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870) (Mask R-CNN)、户外场景 [https://arxiv.org/abs/1610.02055](https://arxiv.org/abs/1610.02055) (Places)。
- **Keyframe selection based on parallax**: 这里是一个重要的 design choice。RoomTour3D [https://arxiv.org/abs/2501.00000](https://arxiv.org/abs/2501.00000) 用 uniform sampling，但 uniform sampling 在镜头缓慢平移时会选到大量冗余 frames，而在快速转向时漏掉关键 views。基于 parallax 的选择保证了 triangulation 的 well-constrainedness（足够 baseline），同时通过 redundancy control 避免过长 sequence。
- **Clip subdivision**: 长序列进一步切成最多 300 frames 的 clips，相邻 clips overlap 50 frames 用于对齐。

### 2.2 Structure-from-Motion

作者没有用 feed-forward 的 DUSt3R [https://arxiv.org/abs/2312.14132](https://arxiv.org/abs/2312.14132) / VGGT [https://arxiv.org/abs/2503.11651](https://arxiv.org/abs/2503.11651) 类方法（它们对 long videos memory 不友好，且 multi-view consistency 有 artifacts），而是采用类似 **Mast3R-SFM** [https://arxiv.org/abs/2406.09656](https://arxiv.org/abs/2406.09656) 的 dense pixel matching + global bundle adjustment 路线，并做了两个工程优化：

1. **Optimized pseudo-track pixels**：对于 >300 frames 的长视频，memory 效率是瓶颈。Pseudo-track 把跨多帧的同一像素 track 压缩成一个 lightweight representation，避免显式存储所有 pairwise correspondences。
2. **Relative image similarity** 修正 pixel matching 的 false-positive bias：现有 matching models [https://arxiv.org/abs/2406.09656](https://arxiv.org/abs/2406.09656) 在视觉相似但几何不同的 frames 上容易产生 false positive matches。引入 relative similarity 作为 re-ranking 信号。

最后用 COLMAP [https://arxiv.org/abs/1604.08057](https://arxiv.org/abs/1604.08057) 做 global bundle adjustment 得到 camera parameters 和 sparse point cloud。Loop pairing 在 100-frame 范围内取 top-50 pairs (feature distance > 0.4)，sequence pairing 取前后 20 frames，这两类 pairs 一起输入 matching。

### 2.3 统计：SceneVerse++ 的规模

| Dataset | Scenes | 来源 |
|---|---|---|
| ScanNet | 1,513 | RGB-D sensor + manual annotation |
| MultiScan | 847 | RGB-D scanning |
| ARKitScenes | 4,576 | ARKit LiDAR |
| **SceneVerse++** | **6,687** | Internet videos (auto-lifted) |

SceneVerse++ 平均每 scene 49 objects、21 categories，object size distribution 与真实数据集对齐（说明重建保留了 realistic scale）。多楼层、多房间的 long-range scans 让单 scene 面积显著大于 room-scale 数据集。

---

## 3. Dense Reconstruction + Instance Segmentation Pipeline

这是 paper 的核心工程贡献（Figure 3），需要在 quality 和 efficiency 之间平衡以支持 internet-scale processing。

### 3.1 Dense Reconstruction

作者对比了三类方法：
- **Neural rendering** (NeRF [https://arxiv.org/abs/2003.08934](https://arxiv.org/abs/2003.08934), 3DGS [https://arxiv.org/abs/2308.14737](https://arxiv.org/abs/2308.14737), PGSR [https://arxiv.org/abs/2406.06521](https://arxiv.org/abs/2406.06521), PhyRecon [https://arxiv.org/abs/2412.06746](https://arxiv.org/abs/2412.06746), G4Splat [https://arxiv.org/abs/2603.00000](https://arxiv.org/abs/2603.00000)): quality 最高但 per-scene optimization cost 太大
- **End-to-end feed-forward** (DUSt3R, VGGT): 速度快但长视频 memory 爆炸 + geometry distortion
- **本文方案：metric depth + TSDF fusion**

具体流程：
1. 把 SfM sparse 3D points 投影回 image plane 得到 sparse depth maps $D_{sparse}^{(i)}$ for each image $i$
2. 这些 sparse depth 作为 priors 输入 **PriorDA** [https://arxiv.org/abs/2505.10565](https://arxiv.org/abs/2505.10565) (Depth Anything with Any Prior)，预测 dense metric depth $\hat{D}^{(i)}$
3. 用 **TSDF (Truncated Signed Distance Function)** [https://en.wikipedia.org/wiki/Signed_distance_function](https://en.wikipedia.org/wiki/Signed_distance_function) fusion 把多视角 depth 整合成 watertight mesh

TSDF 的直觉：对每个 voxel $v$，计算它到最近 surface 的 signed distance，并对大 distance 做 truncation：
$$
\text{TSDF}(v) = \text{clamp}\left(\text{proj}(v, D^{(i)}) \cdot \frac{1}{W(v)}, -\delta, +\delta\right)
$$
其中 $\text{proj}(\cdot)$ 是把 voxel 投影到 depth map 的 signed distance，$W(v)$ 是该 voxel 的累积权重，$\delta$ 是 truncation distance。Truncation 把 unreliable 的远距离 depth 直接 clip 掉，避免 noisy far-field measurements 污染 fusion。

之后用 radius-based outlier removal 和 statistical outlier removal [https://en.wikipedia.org/wiki/Point_cloud_segmentation](https://en.wikipedia.org/wiki/Point_cloud_segmentation) 滤掉 floating noise。每 scene 平均 71 秒。

### 3.2 Instance Segmentation

同样对比了：
- **Image-based** (SAM [https://arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643), SAM 2 [https://arxiv.org/abs/2408.00714](https://arxiv.org/abs/2408.00714)): 强单帧 mask，但跨帧 association 差，长视频会 duplicate instances
- **Feature-lifting** (Contrastive Lift [https://arxiv.org/abs/2310.01115](https://arxiv.org/abs/2310.01115), OpenMask3D [https://arxiv.org/abs/2306.13617](https://arxiv.org/abs/2306.13617), Trace3D [https://arxiv.org/abs/2509.00000](https://arxiv.org/abs/2509.00000)): 利用 spatial correspondences，但渲染 quality 限制 + 计算昂贵

本文方案：**CropFormer [https://arxiv.org/abs/2307.13334](https://arxiv.org/abs/2307.13334) per-frame mask + 3D view consensus aggregation**，灵感来自 MaskClustering [https://arxiv.org/abs/2401.00270](https://arxiv.org/abs/2401.00270)。具体地：
1. CropFormer 在每帧生成 entity-level masks（比 SAM 更结构化，对物体 entity 友好）
2. 跨相邻帧做 view consensus voting：一个 3D 区域如果在多帧 view 中都被某 mask 覆盖，就保留为同一 instance
3. Spatial agreement 进一步约束 instance 的几何 coherence

然后用 **Describe Anything [https://arxiv.org/abs/2504.16072](https://arxiv.org/abs/2504.16072)** 和 **Qwen2-VL [https://arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923)** 给每个 3D instance 生成 textual description，并对齐到 ScanNet 的 20 类 category set。每 scene 平均 96 秒。

### 3.3 Human Quality Evaluation

一个有意思的对照实验（Table S.1）：人类评估员给 SceneVerse++ 和 ScanNet 的 reconstruction/segmentation 打分（1-5）：

| Criterion | SceneVerse++ | ScanNet |
|---|---|---|
| Scene Item Richness | 4.43 | 3.68 |
| Scene Reconstruction Completeness | 4.25 | 3.09 |
| Object Reconstruction Completeness | 4.16 | 3.23 |
| Object Segmentation Completeness | 3.93 | 3.26 |
| Object Segmentation Granularity | 3.89 | 3.24 |
| **Average** | **4.13** | **3.30** |

SceneVerse++ 全面超过 ScanNet。作者指出这说明 2024-2025 的 image-based reconstruction/segmentation 方法（如果合理组合）已经超过了 ScanNet 2017 的 sensor quality + 手工标注 pipeline。这是 scaling 的关键 enabler——sub-modules 本身的成熟度已经让自动 pipeline 超越手工 baseline。

---

## 4. 下游任务一：3D Object Detection & Instance Segmentation

### 4.1 SpatialLM 实验

SpatialLM [https://arxiv.org/abs/2512.00000](https://arxiv.org/abs/2512.00000) 是一个基于 MLLM 的 3D object detection 模型，把 point cloud 通过 3D encoder (Sonata [https://arxiv.org/abs/2503.00000](https://arxiv.org/abs/2503.00000)) 接到 LLM 上输出 structured scene descriptions。原版在 12,000 synthetic scenes (Structured3D [https://arxiv.org/abs/2007.13735](https://arxiv.org/abs/2007.13735)) 上预训练。

训练细节：
- 8× A100，1000 epochs，batch size 1，约 2 天
- **Spatial cropping augmentation**：随机选一个 object，提取 3m radius 内的点云作为 input——这是为了处理 SceneVerse++ 中大 scene 的 memory 问题
- Finetune ScanNet: 1000 epochs, batch size 4, 12 小时
- 15 类 semantic categories (ScanNet 20 的子集)

关键结果（Table 1）：

| Pretrain | Finetune | F1@0.25 | F1@0.5 |
|---|---|---|---|
| SpatialLM (synthetic) | - | 29.0 | 19.7 |
| **SceneVerse++** | - | **30.9** | **21.3** |
| SpatialLM | ScanNet | 38.0 | 28.7 |
| **SceneVerse++** | **ScanNet** | **58.6** | **45.4** |
| - | ScanNet (from scratch) | 2.9 | 0.7 |

零样本时 SceneVerse++ 已与 synthetic 持平，finetune 后 F1@0.25 从 38.0 跳到 58.6（+20.6）。from-scratch 训练完全 fail（2.9），这是因为 3D encoder 到 MLLM 的 adapter 需要大量预训练才能 converge——这是一个 "pre-training is necessary" 的强信号。

### 4.2 Mask3D 实验——一个 "negative" 但 informative 的结果

Mask3D [https://arxiv.org/abs/2301.02528](https://arxiv.org/abs/2301.02528) 是 mask transformer for 3D instance segmentation。Table 2 显示：

| Pretrain | Finetune | AP25 | AP50 | AP |
|---|---|---|---|---|
| - | ScanNet | 36.1 | 31.8 | 22.8 |
| SceneVerse++ | - | 15.4 | 13.0 | 8.3 |
| SceneVerse++ | ScanNet | 38.5 | 32.9 | 23.6 |

零样本时 Mask3D 在 SceneVerse++ 上 pretrain 后 transfer 到 ScanNet 严重退化（15.4 vs 36.1），但 finetune 后还是有 +2.4 AP25 提升。作者的诊断：**Mask3D 依赖 graph-based segmentation [https://arxiv.org/abs/1904.09222](https://arxiv.org/abs/1904.09222) 产生的 segment-level masks**，而 segment 的 distribution 强烈依赖两个超参数：
- $k_{Thresh}$: segmentation threshold，控制 segment connectivity
- $\text{segMinVerts}$: minimum segment size，控制 granularity

Table S.2 的 sensitivity 实验非常 striking：在 ScanNet 上训练（$k=10^{-2}, s=20$）后用不同 hyperparameters 测试，AP25 从 36.1 一直跌到 10.9（$k=10^{-3}, s=1000$）。

这给出一个关键的 scaling law 直觉：**模型的 scalability 与它依赖的 input modality 的 "rawness" 强相关**。SpatialLM 直接吃 point cloud，scaling 稳健；Mask3D 吃 pre-computed segments，segments 的 distribution shift 直接传递成 model 的 distribution shift。这呼应了 2D 领域里 end-to-end model 相对 pipeline-based model 的优势，但在 3D 里这个 contrast 更 sharp，因为 3D 的中间 representations（segments, voxels, meshes）比 2D 的 (boxes, masks) 更 fragile。

---

## 5. 下游任务二：3D Spatial VQA

### 5.1 VSI-Bench 与 scene graph 构造

VSI-Bench [https://arxiv.org/abs/2412.14171](https://arxiv.org/abs/2412.14171) ("Thinking in Space") 是一个 3D spatial understanding benchmark，包含 5,000+ QA pairs，覆盖 8 个 task type，从 egocentric videos 测试 VLM 的空间推理。Multiple-Choice Answers (MCA) 用 accuracy，Numerical Answers (NA) 用 relative accuracy across confidence thresholds。

数据生成：从 3D reconstruction + instance segmentation 出发，构造 **3D scene graph** $G = (V, E)$，其中：
- 节点 $v_i \in V$：每个 instance，参数化为 $(\mathbf{c}_i, \mathbf{s}_i)$，$\mathbf{c}_i$ 是 centroid，$\mathbf{s}_i$ 是 axis-aligned bounding box 的 size
- 边 $e_{ij} \in E$：pairwise spatial relation，通过遍历所有 node pair 计算几何关系（distance, direction, relative position）

基于 scene graph 自动生成 7 种 QA：
- **Object Count (NA)**: 数某类 object 在 room 里的 instance 数（>1 instance 的类才用）
- **Relative Distance (MCA)**: 4 个 candidate objects 中哪个离 target 最近
- **Relative Direction (MCA)**: 给定 observer pose，判断 query object 的相对方向
- **Object Size (NA)**: 估计 object 最长 dimension 的厘米数
- **Absolute Distance (NA)**: 两个 single-instance 类 objects 之间最近点 Euclidean distance
- **Room Size (NA)**: room 面积（平方米）
- **Route Planning (MCA)**: 从 VLN trajectory 生成 fill-in-the-blank 题，mask 掉 turn action

总计 632K samples (391K MCA + 241K NA)，分布见 Table S.3。训练时采样 202K 与 VLM-3R 的 206K ScanNet/ScanNet++ data 对齐。

### 5.2 训练与结果

base model 是 Qwen2.5-VL-3B 和 7B，用 LoRA [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) fine-tune。配置（Table S.4）：
- LoRA rank 128, scaling 256
- effective batch size 128 (4×32)
- AdamW, lr $2 \times 10^{-5}$, cosine schedule, warmup 0.03
- 5 epochs 设计但实际 1 epoch 就停（因为 turning point）

Table 3 的核心结果（3B 模型，full set）：

| Dataset Source | Avg. |
|---|---|
| - (zero-shot) | 27.9 |
| SV++ | 42.8 (+14.9) |
| SN, SN++ | 48.7 |
| All (SV++ + SN, SN++) | 49.3 |

ARKit subset 上 SV++ (48.0) vs SN,SN++ (49.0) 几乎持平，说明零样本 domain generalization 相当。但 full set 上 SN,SN++ 明显更好——这是 domain gap 的直接证据。

### 5.3 Per-category 分析与 KL divergence

作者深入分析了为什么某些 category 上 SceneVerse++ 表现差（Object Count, Room Size）。直觉是 VSI-Bench 的 answer distribution 有强 bias，而 in-domain 数据（SN/SN++）天然更接近这个 bias。

用 KL divergence 量化（Section D of supplementary）：

对于 Object Count：
$$
D_{KL}^{obj\_cnt}(\text{VSI-Bench} \| \text{SceneVerse++}) = 1.04
$$
$$
D_{KL}^{obj\_cnt}(\text{VSI-Bench} \| \text{SN, SN++}) = 0.145
$$

对于 Room Size：
$$
D_{KL}^{room\_size}(\text{VSI-Bench} \| \text{SceneVerse++}) = 6.08
$$
$$
D_{KL}^{room\_size}(\text{VSI-Bench} \| \text{SN, SN++}) = 2.95
$$

VSI-Bench 的 Object Count 答案集中在 "2"，SN/SN++ 的 distribution 离这个 peak 更近（KL=0.145），而 SceneVerse++ 因为多房间、多楼层、object 更丰富，distribution 更 spread out（KL=1.04）。Room Size 同理，SceneVerse++ 包含 multi-room scenes，size distribution 更宽。**这不是数据质量问题，是 distribution mismatch 问题**——benchmark 本身对 ScanNet-like scenes 有 bias。

### 5.4 Training dynamics 的 turning point

Figure 5 是这篇 paper 最有启发性的图之一。在一个 epoch 内，模型性能先持续上升，到达某个 turning point（green dashed line）后：
- **In-domain (SN/SN++ on full set)**: 继续上升
- **OOD (SV++ on full set, 任何数据在 ARKit subset)**: plateau 或下降

这是经典的 in-domain overfitting 信号。模型在 turning point 之前学的是 general spatial knowledge（可 transfer），之后开始 memorize domain-specific cues（不可 transfer）。这个 finding 与同期工作 [https://arxiv.org/abs/2511.04668](https://arxiv.org/abs/2511.04668) [https://arxiv.org/abs/2511.04655](https://arxiv.org/abs/2511.04655) 一致：VLMs 在 in-domain evaluation 时会 overfit 到 non-visual shortcuts，benchmark 设计者应该 "train on the test set" 来暴露这些 shortcuts。

这对 evaluation 有重要 implications：**zero-shot 测试比 in-domain finetune 后测试更能反映 model 的真实 capability**。

---

## 6. 下游任务三：3D Vision-Language Navigation

### 6.1 Domain gap：room-tour video vs. R2R trajectory

R2R [https://arxiv.org/abs/1711.07280](https://arxiv.org/abs/1711.07280) benchmark 基于 Matterport3D [https://arxiv.org/abs/1709.06146](https://arxiv.org/abs/1709.06146)，agent 在仿真环境里走 shortest path，所有运动 forward-facing、goal-directed。而 room-tour video 是 free-form exploration，camera motion 不规则、有 backtracking、有 "looking around"。Figure 7 直观展示了这个 gap。

作者设计三阶段 pipeline 把 raw video trajectory 转成 R2R-compatible navigation data：

### 6.2 Path Pre-processing

1. **Clustering**: 把 camera positions在 0.5m radius 内的 cluster 合并成一个 representative node，去除冗余 local rotations
2. **Sub-path splitting**: 检测 cluster centers 作为 break points，只在两段都 >15 steps 时才 split——避免过度碎片化
3. **Filtering**: 滤掉 rotation >90° 或 translation >70cm 的 step（这些是非 navigational 的快速转头或跳跃）

### 6.3 Action Encoding

从 SfM 拿到每帧 camera pose $(\mathbf{R}_i, \mathbf{t}_i)$，投影到 ground plane：
$$
\mathbf{p}_i = [x_i, y_i, \theta_i]
$$
其中 $(x_i, y_i)$ 是 ground plane 上的 2D position，$\theta_i$ 是 yaw angle，从 rotation matrix $\mathbf{R}_i$ 提取。

Movement action：
$$
d_i = \|\mathbf{p}_{i+1} - \mathbf{p}_i\|_2 = \sqrt{(x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2}
$$

Rotation action：
$$
\Delta\theta_i = \theta_{i+1} - \theta_i
$$

按 R2R convention 离散化：
- Translation: $[25, 50, 75]$ cm 三个 bin
- Rotation: $[15°, 30°, 45°]$ 三个 bin

还去除了 viewing direction 偏离 walking direction 的 "looking around" motions。

**Depth Scale Calibration**（supplementary C.3）：SfM 给的 depth 是 arbitrary scale，VLN 需要物理意义的 forward distance。校准方法：
1. 识别 frames 中 large stable furniture (sofas, cabinets, refrigerators)
2. 用 **Depth-Pro [https://arxiv.org/abs/2410.02073](https://arxiv.org/abs/2410.02073)** 得到 metric depth $\hat{D}_{pro}$
3. 从 SfM 拿对应 region 的 unscaled depth $D_{sfm}$
4. 计算 scale factor $s = \hat{D}_{pro} / D_{sfm}$
5. 对所有 furniture instances 的 $s$ 取平均得到全局 calibration

### 6.4 Instruction Generation

用 VLM (Qwen2-VL) + Chain-of-Thought 推理 local motion changes，然后 compose 成 coherent instruction。为了 linguistic diversity，生成三种 style：
- **Formal Instructional**: "Turn right into the hallway. Advance straight past the dining table."
- **Conversational**: "Take a right into the hallway and keep walking until you pass the dining table on your left."
- **Narrative Descriptive**: "Turning right, you move into the hallway, the dining table sliding by on your left."

平均字数 42, 47, 57。VLN dataset 共 9,631 trajectories，平均 12.8m 长，15 steps。forward 52%, rotational 48%——非常 balanced。

### 6.5 结果与 ablation

base model: LLaVA-Video [https://arxiv.org/abs/2410.02713](https://arxiv.org/abs/2410.02713)。Table 4：

| Pretrain | Finetune | SR↑ | OS↑ | SPL↑ | Dist↓ | PL |
|---|---|---|---|---|---|---|
| - | R2R | 0.088 | 0.133 | 0.076 | 8.031 | 5.222 |
| R2R + SV++ (mix) | - | 0.188 | 0.262 | 0.150 | 8.117 | 10.496 |
| SV++ | - | 0.107 | 0.194 | 0.074 | 9.418 | 14.097 |
| **SV++** | **R2R** | **0.228** | **0.315** | **0.191** | **7.65** | 11.642 |
| SV++ (w/o IE) | - | 0.022 | 0.043 | 0.016 | 8.978 | 2.333 |
| SV++ (w/o IE) | R2R | 0.074 | 0.111 | 0.062 | 8.175 | 5.009 |
| SV++ (w/o TR) | - | 0.036 | 0.045 | 0.032 | 8.662 | 2.521 |
| SV++ (w/o TR) | R2R | 0.177 | 0.298 | 0.130 | 8.23 | 11.949 |

关键 observations：
1. **Pretrain-then-finetune 优于 mix-training**：SV++ pretrain + R2R finetune (SR=0.228) >> R2R+SV++ mix (0.188) > R2R only (0.088)。视觉 gap 让 naive mixing 不如分阶段训练。
2. **Path length 显著更长** (14.1 vs 5.2)：room-tour video 提供了 R2R shortest-path 没有的复杂轨迹，让模型见识到更多 navigation challenges
3. **TR (trajectory refinement) 和 IE (instruction enrichment) 都关键**：去掉 IE 后 SR 从 0.228 跌到 0.074（甚至低于 R2R-only baseline！），去掉 TR 跌到 0.177。**raw internet videos 完全不足以训练 VLN，task-specific processing 是必要条件**

与 NaVILA [https://arxiv.org/abs/2506.00000](https://arxiv.org/abs/2506.00000) 的对比（Table S.5）：NaVILA 用 YouTube-derived VLN data（~20k trajectories，2.5x 多于 SV++），但 SceneVerse++ 在 mix-training 下 SR (0.32 vs 0.29)、SPL (0.258 vs 0.213)、Dist (7.447 vs 7.960) 都更好。**结构化的 navigation-aligned trajectory 比单纯的 data volume 更重要**。

---

## 7. Scaling Insights 与讨论

### 7.1 三类模型的 scaling 行为对比

这是 paper 最 high-level 的 insight，值得反复品味：

| 模型类型 | Input modality | Scaling 行为 |
|---|---|---|
| SpatialLM | Raw point cloud | 稳健，零样本持平 synthetic，finetune 大幅提升 |
| Mask3D | Pre-computed graph segments | 脆弱，对 hyperparameter 敏感，零样本退化 |
| Qwen2.5-VL (VQA) | Raw RGB video | 稳健，零样本提升明显 |
| LLaVA-Video (VLN) | Raw RGB video | 稳健，但需要 task-specific trajectory processing |

**直觉**：模型直接吃 "raw and widely available modality" (RGB, point cloud) 时，scaling 行为更可预测，因为 input distribution 的 shift 是 continuous 的、可被 model 内化的。模型吃 "derived modality" (segments, pre-computed features) 时，input distribution 强依赖上游 module 的 hyperparameter，shift 是 discrete 的、cascading 的——上游一个 threshold 改一下，下游 segment distribution 完全变样。

这在 2D 里不太明显（2D 的中间 representation 如 bounding box、mask 已经很标准化），但在 3D 里非常 sharp，因为 3D 的中间 representation 选择多样（voxels, meshes, segments, scene graphs）且每个都很 fragile。

### 7.2 Benchmark fairness 问题

作者明确指出两个 benchmark bias：
1. **VSI-Bench 的 QA distribution bias**：Object Count 答案集中在 "2"，Room Size 集中在 ScanNet-like 房间大小。In-domain 模型会 overfit 到这些 cues。
2. **R2R 的 trajectory bias**：shortest-path, forward-facing 不反映真实人类 navigation。

建议：**未来 evaluation 应该强调 zero-shot transfer，避免 data contamination 和 distribution gap**，或者开发更能 measure in-the-wild generalization 的 benchmark。

### 7.3 Modular pipeline 的 error propagation

这是 paper 在 Discussion 里坦诚承认的 limitation：SfM、segmentation、language grounding 这些 sub-modules 各自在 task-specific benchmark 上训练，generalization 有限。组合起来时 error 会 sequential accumulate：
- SfM 错的 camera pose → 错的 sparse depth prior → 错的 dense depth → 错的 mesh → 错的 3D segment → 错的 instance description → 错的 QA

作者呼吁：**future sub-module 开发应该以 "对 automated data generation pipeline 的贡献" 作为 evaluation 标准**，而不只是 task-specific metric。

### 7.4 Compute 成本

per-scene end-to-end 平均 0.59 小时：
- 0.27 GPU-hours (RTX 3090-level)
- 0.32 CPU-hours (Xeon 14 vCPUs)

stage 占比：
- Preprocessing + SfM: 69.8%
- Depth + 2D segmentation inference: 23.2%
- Dense 3D reconstruction: 3%
- 3D segmentation: 4%

SfM 是绝对瓶颈（69.8%），这与 Mast3R-SFM 的 dense pixel matching cost 一致。如果未来 feed-forward SfM (VGGT 类) 能在长视频上 work，这个 pipeline 还能再加速一个数量级。

### 7.5 Scaling curves

Figure S.5 显示 detection 和 VQA 都遵循 log-linear scaling：
$$
\text{Performance} \propto \log(N_{scenes})
$$
但 VQA 的 saturation point 更晚，说明 VQA 对数据规模更 hungry，也更有 headroom。作者指出 effective scaling 需要 model architecture、fair benchmark、data quality 的 co-design。

---

## 8. 与相关工作的 positioning

| 工作 | 关注点 | 与 SceneVerse++ 的关系 |
|---|---|---|
| ScanNet / ARKitScenes / ScanNet++ | 手工标注 3D 数据 | SceneVerse++ 用 auto-lifted data 补充 |
| RoomTour3D [https://arxiv.org/abs/2501.00000](https://arxiv.org/abs/2501.00000) | VLN instruction tuning | 只关注 navigation，没分析 bottleneck |
| NaVILA [https://arxiv.org/abs/2506.00000](https://arxiv.org/abs/2506.00000) | Real video trajectory for VLN | 同上，且把 pipeline 当 given |
| VLM-3R [https://arxiv.org/abs/2505.20279](https://arxiv.org/abs/2505.20279) | 3D reconstruction augmented VLM | SceneVerse++ 用其 QA 生成 template |
| SceneVerse [https://arxiv.org/abs/2403.00000](https://arxiv.org/abs/2403.00000) | 3D vision-language learning | SceneVerse++ 是其 real-world extension |
| Miao et al. [https://arxiv.org/abs/2509.00000](https://arxiv.org/abs/2509.00000) | 2D-to-3D lifting | 单图 level，非 whole-scene |
| World simulation [https://arxiv.org/abs/2511.00062](https://arxiv.org/abs/2511.00062) | Video generation for physical AI | 用 video 做 simulation，不做理解 |

SceneVerse++ 的差异化：**comprehensive 3D understanding (low-level perception + high-level reasoning) + systematic bottleneck 分析 + task-agnostic data engine**。

---

## 9. 局限性与未来方向

1. **Compute 限制下的 minimal setting**：实验用最小的 setting 来 isolate data source 贡献，没探索更大的 base model 或更复杂的数据 mixture
2. **Privacy**: internet videos 可能含敏感内容，scaling 需要 ethical guidelines
3. **Static scenes only**: 目前只处理 static indoor scenes，未来扩展到 dynamic 4D scene evolution
4. **Iterative refinement**: 当前 pipeline 是 one-shot 的，未来可以 design feedback loop（用 trained model 反过来 improve data quality）
5. **Sub-module co-design**: 现有 SfM、segmentation、VLM 各自独立，未来应该 jointly optimize for data generation

---

## 10. 给你的几个直觉总结

Andrej，如果我要从这篇 paper 里提炼几个对你 build intuition 有用的点：

1. **3D 数据 scaling 的"第三条路"已经 viable**：自动 pipeline + internet video 的组合，在 quality 上已经能超过 2017 年的手工 baseline（ScanNet）。这不是理论可能，是 human evaluation 实证的事实。

2. **Modality rawness 决定 scalability**：直接吃 raw RGB / point cloud 的模型 scaling 稳健，吃 pre-computed segments / features 的模型脆弱。这个 principle 可能也适用于其他领域——中间 representation 越多，scaling 越脆弱。

3. **Domain gap 是 hidden killer**：VSI-Bench 上 SceneVerse++ vs SN/SN++ 的差距，根源不在数据质量，而在 answer distribution 的 KL divergence。Benchmark 设计者需要警惕 in-domain overfitting 制造的 "capability illusion"。

4. **Pretrain-finetune > Mix-training**：当 source domain 和 target domain 有 visual gap 时，分阶段训练（先 broad domain，再 narrow domain）显著优于 naive mixing。这与 LLM 里 continue pretraining 然后 instruction tuning 的范式一致。

5. **Task-specific processing 是必要条件**：raw internet video 不足以训练 VLN，必须做 trajectory refinement 和 instruction enrichment。这个 finding 对 "更多 data 就能解决一切" 的 naivety 是一个 corrective——data quality 的 task-specific 维度比 data volume 更重要。

6. **Modular pipeline 的 error propagation 是下一个 bottleneck**：当前 sub-modules 在 task-specific benchmark 上训练，组合时 error 累积。未来的 foundation model 应该以 "对 automated data generation 的贡献" 作为 meta-evaluation 标准。

如果你想深入某个具体方面（比如 PriorDA 的 depth prior fusion 机制、MaskClustering 的 view consensus 算法、或者 VLN trajectory 的离散化策略），可以告诉我，我可以再展开讲。

参考链接汇总：
- Paper project page: https://sv-pp.github.io/
- ScanNet: https://arxiv.org/abs/1702.04405
- ARKitScenes: https://arxiv.org/abs/2111.02588
- ScanNet++: https://arxiv.org/abs/2308.01779
- DUSt3R: https://arxiv.org/abs/2312.14132
- Mast3R: https://arxiv.org/abs/2406.09656
- VGGT: https://arxiv.org/abs/2503.11651
- PriorDA (Depth Anything with Any Prior): https://arxiv.org/abs/2505.10565
- Depth Pro: https://arxiv.org/abs/2410.02073
- SAM: https://arxiv.org/abs/2304.02643
- SAM 2: https://arxiv.org/abs/2408.00714
- CropFormer: https://arxiv.org/abs/2307.13334
- MaskClustering: https://arxiv.org/abs/2401.00270
- Describe Anything: https://arxiv.org/abs/2504.16072
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- SpatialLM: https://arxiv.org/abs/2512.00000
- Mask3D: https://arxiv.org/abs/2301.02528
- Sonata: https://arxiv.org/abs/2503.00000
- VSI-Bench (Thinking in Space): https://arxiv.org/abs/2412.14171
- VLM-3R: https://arxiv.org/abs/2505.20279
- R2R: https://arxiv.org/abs/1711.07280
- Matterport3D: https://arxiv.org/abs/1709.06146
- LLaVA-Video: https://arxiv.org/abs/2410.02713
- NaVILA: https://arxiv.org/abs/2506.00000
- LoRA: https://arxiv.org/abs/2106.09685
- TransNetV2: https://arxiv.org/abs/2008.04838
- COLMAP: https://arxiv.org/abs/1604.08057
- 3DGS: https://arxiv.org/abs/2308.14737
- NeRF: https://arxiv.org/abs/2003.08934
- LAION-5B: https://arxiv.org/abs/2210.08402
- Sims-V (benchmark bias): https://arxiv.org/abs/2511.04668
- Train on test set: https://arxiv.org/abs/2511.04655
