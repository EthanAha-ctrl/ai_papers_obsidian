---
source_pdf: AnyGrasp Robust and Efficient Grasp Perception in.pdf
paper_sha256: 57eaa4eb9756c351d0ffeb0dbdfdf8ffd04d178cabb6de3f8d39ec201f250de7
processed_at: '2026-07-18T08:11:36-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AnyGrasp: 一个 Spatial-Temporal Unified Grasp Perception 系统

Karpathy 你好！这篇 paper 我仔细看过了。这是一个非常 engineering-friendly 的工作，来自 SJTU Cewu Lu 团队（也是 GraspNet-1Billion 的作者），核心思路是把 grasping perception 当作一个 dense prediction problem 在 spatial 和 temporal 两个 domain 上同时优化。我下面用更细的粒度来讲，build your intuition。

---

## 1. Paper 的 Motivation：人类 grasping 的四个属性

作者 introspect 人类 grasping 系统的四个关键属性：
- **Prompt**: 视觉处理 < 100ms
- **Accurate**: 7-DoF dense prediction
- **Flexible**: 任何 object
- **Continuous**: 空间上 dense，时间上 smooth（可以抓 moving objects）

之前的工作（sampling-evaluation 类，比如 [ten Pas et al.](https://arxiv.org/abs/1704.02443)、[6-DoF GraspNet](https://arxiv.org/abs/1905.10520)）通常 trade off 计算时间和 grasp 数量，几秒只能生成几十个 grasp。AnyGrasp 走 end-to-end 路线，单次 forward pass 100ms 输出 dense grasp。

---

## 2. Problem Formulation 详解

### 2.1 7-DoF Grasp 表示

$$\mathcal{G} = [\mathbf{R}, \mathbf{t}, w]$$

变量含义：
- $\mathbf{R} \in \mathbb{R}^{3\times3}$: gripper 的 rotation matrix（SO(3)），3 个 DoF
- $\mathbf{t} \in \mathbb{R}^{3\times1}$: grasp center translation，3 个 DoF
- $w \in \mathbb{R}$: gripper 最小闭合宽度，1 个 DoF

总共 7-DoF，比传统 4-DoF rectangle representation [Lenz et al.](https://arxiv.org/abs/1301.3592) 多出来的是完整的 SE(3) rotation（允许 plate 边缘 grasp 这类）。

### 2.2 Spatial Objective

$$\{\mathcal{G}_1^*, ..., \mathcal{G}_n^*\} = \arg\max_{|\mathbf{G}|=n} \sum_{\mathcal{G}_i \in \mathbf{G}} \text{Prob}(s=1 | \mathcal{E}, \mathcal{P}, \mathcal{G}_i)$$

这里 $s$ 是 binary success indicator，$\mathcal{E}$ 是 environment（robot+objects），$\mathcal{P}$ 是 partial-view point cloud。这是一个 set prediction 问题，目标是让 top-n grasp 的 expected success rate 最大化。

### 2.3 Temporal Objective

引入时间维度后，加了一个 consistency constraint：

$$\text{dist}(\mathcal{G}_k^t, \mathcal{G}_k^{t-1} | \mathcal{E}^t, \mathcal{E}^{t-1}) \leq \delta$$

**关键直觉**：这个 distance 是在 **object coordinate system** 下算的，不是 camera coordinate。这意味着即使相机视角变了，只要 grasp 在 object 上的位置没变，就视为同一个 grasp。这是 dynamic grasping 的核心——传统方法只保证 image plane 上的 small distance，物体一转就丢。

---

## 3. Architecture 拆解

整个 system 分两个 module：

### 3.1 Geometry Processing Module（基于 GSNet [Wang et al. ICCV 2021](https://arxiv.org/abs/2108.05850)）

GSNet 的核心 idea 是 **Graspness Discovery**——不是直接在所有点上预测 grasp，而是先预测每个点 "可 grasp" 的概率，然后在 graspable 区域上采样 seed。

Pipeline：
1. **Backbone** (PointNet++/sparse conv)：对每个 point 提取 geometric feature
2. **First MLP block**：输出 objectness mask + graspable heatmap
3. **Graspable FPS**：根据 heatmap 采样 M=1024 个 seed points
4. **Second MLP**：对每个 seed 预测 300 个 view scores，选 best view
5. **Cylinder Grouping**：沿 view 方向 cylinder space 内 group 局部特征
6. **Third MLP**：预测 12 in-plane rotations × 5 approach depths = 60 grasp poses 的 scores 和 widths

**关键修改**：原 GSNet 是 12×4×2 outputs（12 angles × 4 depths × {score, width}），AnyGrasp 改成 12×5×2 + 12，新增的：
- 第 5 个 approach depth 是 0.5cm（为了 grasp 小物体）
- 额外 12 个 stable scores（每个 in-plane angle 一个，跨 depth 共享）

### 3.2 Stable Score 的物理直觉

这是这篇 paper 一个很 elegant 的贡献。基于 [Baldauf & Deubel 2010](https://pubmed.ncbi.nlm.nih.gov/20304546/) 的认知科学发现：人类在 grasping preparation 阶段 visual attention 会偏向 object 的 center of gravity。

定义：假设 gripper 抓到 object 后会移动到垂直 pose（与 gravity 平行）。stable score 就是 gripper plane 到 object COG 的 **normalized perpendicular distance**。

物理意义：如果 grasp 点离 COG 远，重力臂大，object 容易在 transport 过程中 rotate/slippage。如果离 COG 近，重力矩小，更稳定。

数学表达（我推测）：
- 设 COG 在 gripper frame 下坐标为 $\mathbf{c} = (c_x, c_y, c_z)$
- Gripper plane 是 $z=0$ 平面
- Perpendicular distance $= |c_z|$
- Normalized：$s_{\text{stable}} = |c_z| / \max_{\text{all grasps on object}} |c_z|$

推理时 score = grasp_score × (1 - stable_score)，相当于 penalize 离 COG 远的 grasp。

### 3.3 Temporal Association Module

这是 paper 的真正 novelty。要 track 一个 grasp 在 dynamic scene 中的轨迹，他们没有走 6D pose tracking（[BundleTrack](https://arxiv.org/abs/2102.05500)）路线，而是直接 track **grasp pose** 本身。

Feature 构造（每个 grasp 一个 256-dim feature vector）：
1. **Seed feature**: 来自 backbone，几何信息
2. **Grasp feature**: 来自 grasp prediction 前的中间层
3. **Color feature**: 用 cylinder grouping 把 K=16 个 RGB points 沿 grasp 方向 gather，过 MLP+pooling
4. **Grasp pose**: 9 (rotation matrix) + 3 (translation) = 12 维

Concat 后过 MLP → 256-dim feature $\mathbf{f}$

**Correspondence score**:
$$s_{\text{corres}}(\mathcal{G}_1, \mathcal{G}_2) = \frac{\mathbf{f}_1 \cdot \mathbf{f}_2}{\|\mathbf{f}_1\| \cdot \|\mathbf{f}_2\|}$$

就是 cosine similarity。M×M matrix 就是 correspondence matrix。

### 3.4 Distance Metric for Grasp Pairs

两个 grasp 在 object frame 下的距离：

$$\Delta \mathbf{R} = \arccos \frac{\text{trace}(\mathbf{R}_1^\top \mathbf{R}_2) - 1}{2}$$

这是 SO(3) 上的 geodesic distance，$\text{trace}(\mathbf{R}_1^\top \mathbf{R}_2)$ 等于 $1 + 2\cos\theta$，其中 $\theta$ 是相对旋转角。

$$\Delta \mathbf{t} = \|\mathbf{t}_1 - \mathbf{t}_2\|$$

欧氏距离。

总距离：
$$d(\mathcal{G}_1, \mathcal{G}_2) = \frac{\Delta \mathbf{t}}{w_{\max}} + \gamma \frac{\Delta \mathbf{R}}{\pi}$$

变量：
- $w_{\max} = 0.01$m（gripper 最大宽度，作为 translation 的 normalization）
- $\gamma = 0.1$（rotation vs translation 的 balance weight）
- $\Delta \mathbf{R}/\pi \in [0, 1]$：rotation 归一化到 [0,1]
- $\Delta \mathbf{t}/w_{\max}$：translation 用 gripper width 归一化

如果 $d \leq \sigma = 0.1$，认为两个 grasp 是 "same class"（同一 grasp 在不同帧的对应）。

### 3.5 Loss Function: Supervised Contrastive Learning

用的是 [SupCon (Khosla et al.)](https://arxiv.org/abs/2004.11362)：

$$L = \sum_{\mathcal{G}_i^1 \in \mathbf{G}_1} \frac{-1}{|\mathbf{P}(i)|} \sum_{\mathcal{G}_k^2 \in P(i)} \log \frac{\exp(s_{\text{corres}}(\mathcal{G}_i^1, \mathcal{G}_k^2)/\tau)}{\sum_{\mathcal{G}_j^2 \in \mathbf{G}_2} \exp(s_{\text{corres}}(\mathcal{G}_i^1, \mathcal{G}_j^2)/\tau)}$$

变量：
- $\mathbf{P}(i) = \{\mathcal{G}_k^2 \in \mathbf{G}_2 | d(\mathcal{G}_i^1, \mathcal{G}_k^2) \leq \sigma\}$：second frame 中和 $\mathcal{G}_i^1$ 同 class 的 grasp 集合
- $|\mathbf{P}(i)|$：这个集合的大小
- $\tau = 0.1$：temperature，控制 similarity 分布的 sharpness

直觉：让 same-class grasp pair 的 feature 靠近，不同 class 的远离。这比 simple triplet loss 优势在于 **many positives** 一起拉，更 stable。

---

## 4. Dataset Design 的 Insights

这部分我觉得是 paper 最 valuable 的 contribution 之一，对社区非常有指导意义。

### 4.1 Real vs Simulation

作者做了 controlled experiment：用 PyRender 渲染同一个 dataset（因为 GraspNet-1Billion 有 mesh + 6D pose annotation），加 Gaussian noise 做 sim-to-real，结果：
- 在 GraspNet-1Billion benchmark 上：sim w/ noise 比 real 差很多
- 在 real robot bin picking 上：sim w/ noise 从 93.3% 掉到（图中所示）大约 70%+

**直觉**：low-cost depth camera (RealSense D415/D435) 的 noise 分布不是 Gaussian，是 structured noise（depth deviation map 显示有 spatial correlation）。Gaussian noise augmentation 无法 capture 真实 sensor 的 noise pattern。

### 4.2 Dense Annotation vs Image Amount

作者把训练数据按三个维度 downsample：
- Grasp pose density per object (10M → 1M / 200K)
- Image amount
- Scene amount

结果：
- Downsample pose 10× ≈ Downsample image 10×（performance 下降相当）
- Downsample scene 10× 下降更严重
- Downsample scene 50×（只剩 2 个 scene）直接不收敛

**直觉**：scene diversity > grasp annotation density > image amount。这跟 ImageNet 的经验一致——class 多比 image 多重要。但这里 grasp annotation 几乎免费（analytic antipodal score 自动算），所以 dense annotation 是 free lunch。

### 4.3 144 Objects 的反直觉发现

之前的 dataset 越来越大：DexNet 1000+ objects, EGAD 2000+, ACRONYM 8000+。AnyGrasp 只用 144 objects 就达到 human-level performance。

**直觉**：grasping 的难度不在 object shape 的 long-tail，而在 sensor noise + scene clutter 的 combination。144 个 carefully selected real objects（每个有 dense annotation + 256 viewpoints）> 8000 个 simulated objects。

---

## 5. 实验数据深度分析

### 5.1 Main Results (Table I)

| Object | Dex. | Any. | Human |
|--------|------|------|-------|
| Hardware | 59.3 | 81.5 | 91.4 |
| Snack | 52.3 | 100.0 | 93.9 |
| Ragdoll | 87.4 | 100.0 | 96.6 |
| Toy | 72.8 | 93.1 | 91.8 |
| Household | 64.6 | 85.5 | 94.4 |
| **All** | **72.2** | **93.3** | **93.9** |

**Attempt-centric** vs **Object-centric** 两个 metric：
- Attempt-centric = successful attempts / total attempts
- Object-centric = successful objects / total objects（更宽松）

AnyGrasp 在 attempt-centric 上 93.3% ≈ Human 93.9%，object-centric 99.8% ≈ 100%。

注意 Snack 和 Ragdoll 上 AnyGrasp **超过** human（100% vs 93.9%/96.6%），我推测是因为这些物体几何上 "easy" 但 human 在 open-loop（无 tactile feedback）条件下不稳。

### 5.2 MPPH (Mean Picks Per Hour)

AnyGrasp: **900+ MPPH** (single UR5 arm)
DexNet 4.0: 300 MPPH (dual-arm system)
Human: 1000-1200 MPPH average, 1500 max

**Perception time**: 100ms grasp prediction + 80ms post-processing = <200ms total decision time。瓶颈是 robot motion（UR5 max 1m/s）+ gripper close time。

### 5.3 Fish-Catching Dynamic Experiment

最 impressive 的 demo。在 fish tank 里抓 8 个 swimming robot fish，5 次实验平均 **75.5% success rate**。

挑战：
1. 水下摩擦小，gripper 容易 slip
2. Fish 很小，pose error tolerance 低
3. Fish 一直动（vs human handover 时 human 会 stabilize）
4. 水下点云 noise 大（光的反射折射）

Failure analysis (5 类)：
1. Slip away (good pose but low friction) ~50%
2. Future pose 预测过远
3. Grasp quality 不好，finger 推开 fish
4. Future pose 预测过近
5. Correspondence switch（两条相似 fish 时 tracking 混淆）

vs Heuristic baseline (track nearest grasp): 62.5%，且慢 12.7%。

### 5.4 Algorithm 1 解析：Dynamic Grasping 的核心 trick

```
Pre-grasp servoing with future prediction:
- Buffer length 10
- Future pose = last pose + momentum from buffer
- Servo to pre-grasp pose = future_pose translated 3.5cm back along z
- Trigger grasp when: 3D dist < 5.5cm AND 2D dist < 2cm AND angle < 20°
```

**直觉**：不能直接 servo 到 grasp pose，因为 gripper closing 有延迟，fish 一直在动。所以预测 future pose，servo 到 future pose 之前 3.5cm 处，等 fish 自己 "游进" grasp range。

---

## 6. Post-processing 的两个 trick

### 6.1 Collision Detection

虽然 network 隐式学了 obstacle awareness，但是 soft constraint。对 top-100 grasp 做 hard collision check：把 gripper 简化成 3 个 cube，检查 partial-view point cloud 里有没有 point 在 cube 内。

### 6.2 Gripper Centering

观察：如果两个 fingertip 不同时接触 object surface，先接触的 finger 会推开 object。GraspNet-1Billion 的 annotation 没有保证 fingertip 到 object 的等距性。

修复：找到 gripper space 内最外侧的 point 作为 contact point，沿 fingertip 连线方向 translate gripper，使两 fingertip 到 contact point 的距离相等。

**直觉**：这其实是在 "重定义" grasp center，让它对齐 object 的几何 center（在 gripper 方向上），而不是用 dataset 给的 analytic center。

---

## 7. Limitations & Open Questions

1. **Closed-loop under occlusion**: 当 gripper 接近 object 时会 occlude target，visual perception 失效。Missing piece 是 tactile sensing。
2. **Transparent / black objects**: depth sensor 失效，需要 [ClearGrasp](https://arxiv.org/abs/1911.04498) 或 [TransCG](https://arxiv.org/abs/2202.08471) 这类方法补 depth。
3. **Deformable objects**: 文中 Fig. 17 展示了 grasp tracking 在 deformable tube 上能 work，但是是 textureless + 噪声的 special case。一般 deformable object 的 grasp 还是 open problem。
4. **Failure on adversarial objects**: EGAD + DexNet 2.0 adversarial set 上 success rate 明显下降，因为 paper 的 system 是 open-loop（无 trial-and-error feedback），human 会从失败中调整。

---

## 8. 我对这篇 paper 的 intuition 总结

**最核心的 insight**：把 grasping 当作 **dense spatial-temporal prediction** 而不是 sampling + evaluation。这样：
- 速度：单次 forward pass vs 几百次 sample-eval
- Density：scene 任何位置都有 grasp 候选
- Temporal consistency：dense prediction + feature matching 天然 enable tracking

**第二个 insight**：**Real data > Sim data**，即使只有 144 objects。这挑战了 grasping community "more objects is better" 的迷信。Key 是 dense annotation + scene diversity + real sensor noise。

**第三个 insight**：**Grasp tracking > Object 6D pose tracking** for dynamic grasping。因为：
- 不需要 object model
- 对 deformable / partial view 更 robust
- 直接 optimize grasp-centric objective

**第四个 insight**：**Stable score (COG awareness)** 是一个 cheap but effective physics prior，把认知科学的发现（human attention bias to COG）encode 进 network。

---

## References

- [AnyGrasp SDK GitHub](https://github.com/graspnet/anygrasp_sdk)
- [GraspNet-1Billion (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf)
- [GSNet / Graspness Discovery (ICCV 2021)](https://arxiv.org/abs/2108.05850)
- [Supervised Contrastive Learning (NeurIPS 2020)](https://arxiv.org/abs/2004.11362)
- [DexNet 4.0 (Science Robotics 2019)](https://www.science.org/doi/10.1126/scirobotics.aav4944)
- [BundleTrack 6D Pose Tracking (IROS 2021)](https://arxiv.org/abs/2102.05500)
- [6-DoF GraspNet (ICCV 2019)](https://arxiv.org/abs/1905.10520)
- [ten Pas et al. Grasp Pose Detection in Point Clouds (IJRR 2017)](https://arxiv.org/abs/1704.02443)
- [ClearGrasp (ICRA 2020)](https://arxiv.org/abs/1911.04498)
- [TransCG (ICRA 2022)](https://arxiv.org/abs/2202.08471)
- [EGAD Dataset (RA-L 2020)](https://arxiv.org/abs/2003.01314)
- [ACRONYM Dataset (ICRA 2021)](https://arxiv.org/abs/2011.09274)
- [Baldauf & Deubel, Attentional Landscapes (Vision Research 2010)](https://pubmed.ncbi.nlm.nih.gov/20304546/)
- [GraspNet 官方网站](https://graspnet.net/)

---

如果你想 dive deeper，我特别推荐看 [GSNet](https://arxiv.org/abs/2108.05850) 的 graspness formulation，因为 AnyGrasp 的 spatial module 完全 build on 它。另外 [SupCon paper](https://arxiv.org/abs/2004.11362) 值得重读，temporal module 的 loss 设计直接借用，但 positive set 的构造（基于 grasp pair distance）是 novel 的应用。
