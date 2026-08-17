---
source_pdf: Explaining Human Preferences via Metrics for Structured 3D Reconstruction.pdf
paper_sha256: 6be228f0ed1a380524f8e9ca0c830be423d721d6d0646c811deb5c33e1f91c85
processed_at: '2026-08-04T06:15:36-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话版本

搞 3D 重建这帮人用了一堆 metric 来评分，结果发现最受欢迎的 metric（WED，CVPR challenge 官方用的）跟人类专家的直觉完全相反——你按它优化，模型越优化人类越觉得垃圾。

---

## 这事儿怎么发生的

3D 重建这边有个挺常见的任务：从照片或 LiDAR 点云重建出房子的 wireframe（线框图，就是一堆顶点和边连成的 graph）。每年都有 challenge，大家比谁的方法好。

问题是：**怎么评分？**

你看，做 image classification 简单——对就对了，错就错了。但 wireframe 这玩意儿是 graph，两个 graph 之间的"距离"压根没有一个公认的 definition。于是大家各搞各的：

- 有人算 vertex 的 precision / recall
- 有人算 edge 的 precision / recall
- 有人搞了个 Graph Edit Distance 的变体叫 **WED**（Wireframe Edit Distance）
- 有人用 Chamfer Distance（把 edge 采样成点云算距离）
- 有人用 spectral distance（graph Laplacian 的 eigenvalue 比较）
- 这篇 paper 还加了 IoU（把 wireframe 当成圆柱体算体积交集）

**每篇 paper 用的 metric 不一样，所以方法之间没法比。**这本身就 bad。

但更糟糕的是：**没人验证过这些 metric 跟人类判断一致不一致。**

---

## Motivating example 太打脸了

Figure 1 那个例子真该贴在每一个做 3D 重建的人的墙上。三个 wireframe：

- **WF1**：把一条长边拆成几段短的 collinear 边——几何上完全正确，就是顶点多打了几个
- **WF2**：缺了一些顶点和边
- **WF3**：基本全错，只有一个顶点落在正确位置

**人类排序**：WF1 > WF2 > WF3（很显然）

**WED 排序**：WF3 > WF2 > WF1（完全反过来！）

为什么？因为 WED 把"多一个顶点"当成 insert operation，cost 直接加上去。WF1 有几个多余的 collinear 顶点，WED 就把它罚得很惨。WF3 虽然几乎全错，但它"编辑操作"少啊——删掉一个、插入几个就完事，总 cost 反而低。

这就是作者说的 **metric hacking**：你要是想在 S23DR challenge 拿奖，最优策略是输出一个几乎空的 wireframe，WED 会给你高分。荒谬吧？

---

## 他们怎么验证的

招了三类人做标注：

1. **11 个 professional 3D modeler**（Hover Inc. 的员工，每天的工作就是从照片建 CAD model，这是真 expert）
2. 4 个 CV researcher
3. 3 个 designer（不做 3D 的外行）

每个人看 3510 对 wireframe，选哪个更接近 ground truth。UI 做得很认真——可以旋转、缩放、平移，两边视角同步，每 350 对强制休息。标注质量：self-consistency 89.4%，synthetic sanity check 98.3%。这个 setup 比 AMT 众包严谨多了。

**关键设计**：除了真实重建结果，他们还对 ground truth 施加各种"损坏"（deform / perturb / add / remove，每种三档严重程度），这给了他们一个 controlled ground truth ranking 来做 sanity check。

---

## 几个让我醍醐灌顶的发现

### 1. Recall 比 Precision 重要

这是 paper 里最深的 insight。人类专家打分时，**漏掉正确的边/顶点比多出错误的边/顶点更受惩罚**。

看 win rate：

- `add_low`（加少量错边）：win rate 0.89
- `remove_low`（删少量对边）：win rate 0.79

同样是少量错误，加错的边人类觉得"还行，我能手动删"，删对的边人类觉得"完蛋，我上哪推断它应该在哪儿"。

直觉解释：reconstruction 是给 modeler 当参考用的。**多出来的东西是可验证的（modeler 看一眼就知道是错的），但缺失的东西是不可推断的**（信息已经丢了）。这是 information asymmetry。

这让我想到 GAN 的 mode collapse 问题——生成器"省略"某些 mode 比生成错误样本更糟，因为 critic 无法从 absence 推断 presence。信息论视角：absence 的 entropy 高，presence 的 entropy 低。

### 2. 人类分两派

标注者分两个 cluster：
- **Cluster 1**：更看重顶点准确性（correlates with corner F1）
- **Cluster 2**：更看重边的准确性（correlates with edge F1 / Jaccard）

而且这个 split 跟职业无关——3D expert 也分散在两个 cluster 里。这意味着即使是 domain expert，脑子里也没有统一的"质量"定义。

但有个统一规律：**当人类做出明确判断（不选"equal"）时，agreement 飙升到 91%**，cluster 现象消失。也就是说，**模糊 case 人类各有偏好，清晰 case 人类高度一致**。

### 3. WED 是最差的 metric

WED 系列（包括 S23DR challenge 用的版本）跟人类一致性最低，preregistration 版本甚至只略好于随机。这是个大 news：**决定 CVPR challenge winner 的 metric 基本是 random**。

### 4. VLM 没用

他们试了 GPT-4o, o1, Grok 2, Qwen-2.5, Pixtral 12B, Claude 3.5/3.7, Gemini 2.0/2.0-flash。结果除了 OpenAI 和 Grok 2 略好于随机，其他基本就是 random。

讽刺的是 VLM 跟 WED 一致性最高——也就是说 VLM 跟最差的 handcrafted metric 想法一致。这暗示 VLM 的 visual encoder 对 wireframe 这种 sparse line drawing 的理解很 shallow。

我（作为 Karpathy）直觉：VLM 在 2D rendering 上做 3D topological comparison，本质上缺少了 3D 空间推理的 grounding。VLM 训练数据里"判断哪个 wireframe 更像 GT"这种样本几乎为零。

### 5. 数学性质好 ≠ 好用

Spectral L1 distance 在 17 个 unit test 里 pass 13 个，数学性质很好。但跟人类 alignment 差。

反过来 Chamfer edge distance 在 unit test 只 pass 8 个，但 alignment 还行。

教训：**你不能光看一个 metric 的数学优雅程度判断它好不好用**。Metric 的"目标函数"是人类偏好，而人类偏好不是数学公理推导出来的。

### 6. 单一 quality factor 确实存在

用 Bradley-Terry model + SVD factor analysis，证明人类判断背后有一个 dominant "quality" factor（Kendall τ > 0.7）。也就是说虽然标注者之间有 cluster，但背后还是有个统一的 quality dimension。

---

## 他们还搞了个 learned metric

Pipeline 很简洁：

1. 把 wireframe 在 3D 里 render 成图（canonical viewpoint）
2. 用 **DiNOv2** 提 visual features
3. MLP regression head 输出 scalar score
4. Bradley-Terry loss + binary cross-entropy 训练

10-fold cross validation，train/test 的 GT structures 和 reconstruction methods 都 disjoint（防 leakage）。**结果：76% accuracy**。

听起来不高，但人类之间的 agreement 也才 80-85%。所以 learned metric 已经接近 human noise floor 了。DiNOv2 features 真的很 strong，居然从一个 2D rendering 就能 decode 出 wireframe 的"质量"。

但作者明确警告：**别拿 learned metric 做 challenge 的唯一评判**，因为 differentiable metric 容易被 gradient-based adversarial attack hack。任何 learned metric 都是可微的，attacker 反向传播到输入 wireframe 就能找出"score 高但实际 garbage"的 reconstruction。这是 learned metric 的 fundamental vulnerability。

---

## Unit test 这个 idea 很聪明

他们定义了 17 个 "unit test"：identity of indiscernibles（$d(x,x)=0$）、symmetry、triangle inequality、各种 monotonicity（加错边、删对边、移顶点等情况下分数该往哪个方向变）。

每个 metric 在 128 个 GT wireframe 上跑这些 test，统计 pass 率。结果：

- **Edge F1** 通过最多（14/17）+ alignment 好 → 推荐
- **WED prereg** 通过最少（8/17）+ alignment 差 → 双重失败
- **Hausdorff** 双重失败（6/17 + 差 alignment）—— 对 outlier 太敏感

这种 unit test framework 很 elegant，本质上是在做 metric 的 "behavioral test"——你不用 prove metric 是对的，你只要在 controlled 扰动下看它的 behavior 符合不符合预期。

我觉得这种思路值得推广到其他领域的 metric design。ImageNet 上的 top-1 accuracy 也可以搞 unit test：adversarial perturbation 下该不该降分？texture change 下该不该降分？shape preserve 下该不该保持？

---

## 我的几个 takeaway

### 1. Benchmark metric 必须做 human alignment study

这篇 paper 给整个 CV community 一个方法论：**不要假设你的 metric 跟人类一致，去验证**。招 domain expert 做 pairwise comparison，算 Kendall τ，跟 metric ranking 比。这是基本的 scientific hygiene，但 3D 重建社区十年都没人做。

ImageNet 之所以成功，部分原因是 top-1 accuracy 跟人类直觉大致一致（虽然也有 adversarial robustness 问题）。3D reconstruction 的 metric 从一开始就脱离人类直觉，所以整个 community 在 optimize 错误的目标。

### 2. Recall > Precision 这个 insight 影响深远

不只是 wireframe 重建，很多生成 / 重建任务都可能有这个 phenomenon。Floorplan 重建、3D mesh reconstruction、point cloud completion 都值得验证。

更广义地：**absence 的 information loss 大于 presence 的 information loss**，当输出是给下游 human-in-the-loop 用的时候。这个 principle 可能是 universal 的。

### 3. Cluster 现象值得深挖

为什么 expert 之间会分 cluster？可能反映两种 cognitive style：
- "Bottom-up"：先看局部顶点对不对
- "Top-down"：先看整体边连得对不对

这种 cognitive split 在其他 task 里也可能存在。标注协议设计时应该考虑。

### 4. Learned metric 接近 noise ceiling 这事很重要

76% accuracy 听起来不高，但接近 80% human-human agreement。这意味着**人类偏好是可以被 distilled 的**，不是完全 subjective 的。这给 future benchmark design 一个方向：用 learned metric 做 pre-screening，human rater 做 final adjudication。

### 5. 这篇 paper 的局限

- 只在北美 residential building 上做的，domain generalization 没验证
- Learned metric 的 adversarial vulnerability 只 mention 没 study
- 没尝试 metric ensemble
- WED 改进方向只提了 idea 没实现
- Cluster 现象没有 cognitive science 层面的解释

---

## 结论

这篇 paper 是 3D reconstruction community 长期需要的 wake-up call。它证明：

1. 你用的 metric 可能跟人类判断 anti-correlated
2. 竞赛 winner 可能是 metric bug 的产物
3. Recall 比 Precision 更重要（对 human-in-the-loop task）
4. VLM 还不能替代 human judgment
5. 数学优雅 ≠ 好用

Methodologically，它给了一套 template：human pairwise annotation + Bradley-Terry / SVD ranking + unit test + learned metric distillation。这套方法可以迁移到任何 "metric 验证"问题。

如果你做 3D 重建，**立刻把你的 metric 从 WED 切换到 edge F1 + corner F1**。别再 optimize 错误目标了。

参考链接：

- Paper repo: https://github.com/s23dr/wireframe-metrics-iccv2025
- S23DR Challenge: https://huggingface.co/usm3d
- DiNOv2: https://arxiv.org/abs/2304.07193
- PC2WF (WED 的 origin): https://arxiv.org/abs/2103.01793
- Building3D dataset: http://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Building3D_An_Urban-Scale_Dataset_and_Benchmarks_for_Learning_Roof_ICCV_2023_paper.pdf
- HEAT (CVPR 2022 structured reconstruction): https://arxiv.org/abs/2203.10379
- LMSYS Chatbot Arena (Bradley-Terry 在 LLM benchmarking 的应用): https://chat.lmsys.org
- Original Bradley-Terry paper (Biometrika 1952): https://www.jstor.org/stable/2334029
- "What cannot be measured cannot be improved" - Lord Kelvin apocryphal quote: https://en.wikipedia.org/wiki/Lord_Kelvin#Atoms_and_energy

---

# Explaining Human Preferences via Metrics for Structured 3D Reconstruction — 深度解读

这篇 ICCV 2025 paper 由 Jack Langerman、Denys Rozumnyi（ETH Zurich）、Yuzhong Huang、Dmytro Mishkin（Hover Inc.）合作完成，core question 极其简单但又极关键：**当我们用 automatic metric 评估 structured 3D reconstruction（wireframe / CAD-like 模型）时，metric 算出来的 ranking 真的反映人类专家的偏好吗？** 答案令人不安——很多社区 standard metric（包括 CVPR challenge 用的 WED）基本上是 broken 的。

GitHub repo: https://github.com/s23dr/wireframe-metrics-iccv2025
S23DR Challenge: https://huggingface.co/usm3d

---

## 1. Motivation: 为什么这个问题值得做

Structured 3D reconstruction 的输出是 spatial graph: vertices（roof apex, corner）+ edges（ridge line, eave line）。输入可以是 ground images（HEAT, CVPR 2022）、satellite images、aerial LiDAR（Building3D, PBWR）等。问题不在于缺少 metric，而在于 metric 太多了，每个 paper 用不同的：

- Recognition metrics: vertex/edge Precision/Recall/F1，Structured Average Precision
- Graph metrics: Wireframe Edit Distance (WED)，Graph Edit Distance
- Point cloud metrics: Chamfer Distance (CD)，Hausdorff Distance
- Spectral metrics: Length-Weighted Spectral Graph Distance
- IoU-based metrics（这篇 paper 新提出）

**关键痛点**: Figure 1 给的 motivating example 非常 sharp。三个 wireframe：

- **WF1**: 把长 edge split 成多个 collinear 段，几何和拓扑完全正确
- **WF2**: missing 一些 vertices 和 edges
- **WF3**: 几乎只有一个 vertex 是对的

人类一眼排序：WF1 > WF2 > WF3。但看 metric：

| Metric | WF1 | WF2 | WF3 |
|---|---|---|---|
| Vertex F1 ↑ | 0.56 | 0.91 | 0.18 |
| Edge F1 ↑ | 0.19 | 0.71 | 0.00 |
| Jaccard ↓ | 0.00 | 0.33 | 1.00 |
| WED ↓ | 2.32 | 0.52 | 1.82 |
| WED S23DR ↓ | 2.69 | 0.59 | 1.63 |

Vertex F1 把 WF2 排第一（因为 WF1 多了 collinear vertices 算 "wrong"），WED 把 WF3 排在 WF1 前面（完全 reverse）。也就是说，challenge winners 用的 metric 是 anti-correlated with human preference 的。这就是作者说的 "metric hacking"——你完全可以靠 split edges 让 WED 变差，但其实 reconstruction 是 perfect 的。

---

## 2. 实验设计: 人类偏好如何被 measure

### 2.1 Annotation pool

作者招了三类人：

- **11 个 professional 3D modelers**（每天工作就是从照片做 CAD model 的，来自 Hover Inc.）
- **4 个 CV researchers**
- **3 个 designers**（不做 3D 建模）

每个 rater 看 27 个 method × C(27,2)=351 pairs × 10 houses = 3510 pairs，UI 见 Figure 2。Annotators 可以 translate / scale / rotate，两边 viewpoint 同步。每 350 pairs 强制休息。

**两个 wireframe pool**:
- Pool 1: S23DR challenge top-10 entries + PC2WF baseline，real reconstructions from multi-view images
- Pool 2: corrupted ground truth——对 GT wireframe 施加 4 种 corruption，每种 3 个 severity level (low/med/high)：
  - `deform_*`: edge split + 顶点扰动（不破坏 topology）
  - `perturb_*`: vertex split into several，随机 shift，随机 reconnect
  - `add_*`: 加 wrong edges
  - `remove_*`: 删 vertices + 所有连边

Pool 2 是关键设计——它给了一个 controlled "ground truth ranking" 来做 sanity check。

### 2.2 Rater reliability

- Synthetic correctness（low vs high corruption 应该明显排序）：**98.3%**
- Self-consistency（5% 概率重复 pair，可能 reverse order）：**89.4%**
- Binomial model 假设 individual error rate 20%，11 raters 的 panel error rate ≈ 1%，17 raters ≈ 0.25%
- Bootstrap CI 分析：达到 Kendall τ ≥ 0.95 需要至少 3350 comparisons、4 houses、8 raters——他们都超过这个数

**这个 setup 比 ImageNet 那种众包标注严谨多了**。3D modeler 的工资由 Hover 付，是 fair hourly rate，不是 AMT worker。

---

## 3. Metrics 深入讲解

### 3.1 Wireframe Edit Distance (WED)

WED 是 Graph Edit Distance 的扩展。GED 把 graph 距离定义为 minimum number of edit operations（insert/delete vertex/edge）把一个 graph 转成另一个。WED 加了 node positions 和 edge lengths，用 cheap approximation 解 NP-Hard 问题：

1. 先做 predicted ↔ GT vertex assignment（成本 ∝ matched vertex 距离）
2. Unmatched vertices 被 delete / missing 被 insert（成本 ∝ 数量）
3. 在 vertex assignment 固定后，insert missing edges / delete extra edges（成本 ∝ length）

WED 有多个变体：

- **WED_S23DR** (preregistration): S23DR challenge 用的版本，先做 registration
- **WED_MNN** (mutual nearest neighbor)
- **WED_AP**: Building3D challenge 用的，average precision-style assignment

### 3.2 Edge Chamfer Distance (ECD)

公式（paper 中的 Eq. 1）：

$$d(A,B) := \inf_{\pi_{AB}: A \to B} \mathbb{E}_{a \in A}[f(a, \pi_{AB}(a))]$$

变量解释：
- $A, B$：从两个 wireframe 的 edges 上 sample 出来的点集
- $\pi_{AB}$：A 中元素到 B 中元素的 assignment
- $f$：通常是 $\ell_p$ norm of difference

两个 extreme：

- **Classical Chamfer**: $\pi_{AB}(a) = \arg\min_{b \in B} f(a,b)$，nearest neighbor matching
- **Bijective / Earth Mover's Distance**: $\pi_{AB}$ 必须是 bijection，用 Hungarian algorithm 求解

### 3.3 Length-Weighted Spectral Graph Distance

公式（Eq. 2, Eq. 3）：

$$SD(G_1, G_2) := W_2(\lambda(L_1), \lambda(L_2))$$

$$L := D - A$$

变量解释：
- $G = (V, E)$：graph
- $L$：weighted graph Laplacian
- $D$：weighted degree matrix，$|V| \times |V|$ 对角阵，每个对角元是 incident edges 的 length 之和
- $A$：weighted adjacency matrix，$A_{ij} = \|\text{coord}(V_i) - \text{coord}(V_j)\|_2$ if $(i,j) \in E$ else 0
- $\lambda(L)$：L 的 spectrum（eigenvalue 分布）
- $W_2$：2-Wasserstein distance between two eigenvalue distributions

直觉：graph Laplacian 的 spectrum 编码了 graph 的"振动模式"，类似 shape analysis 里的 Laplace-Beltrami spectrum。两个 graph 越相似，spectrum 越接近。Length weighting 让 long edges 和 short edges 贡献不同。

### 3.4 IoU via cylinders

把 wireframe 看作一组 fixed-radius cylinders（半径是唯一 hyperparameter），定义 IoU 为两组 cylinders 的体积交集 / 体积并集。实际实现用 point sampling 近似：从两个 cylinder 集合 sample 随机点，统计同时落在两组内的比例，report Jaccard distance。

### 3.5 Hausdorff

从 edges 上 sample points，计算两组点之间的 Hausdorff distance（max of min distances）。只考虑最坏点，所以对 outlier 极敏感。

### 3.6 Corner / Edge F1

- Corner: prediction 在 GT corner 的 distance threshold 内算 correct
- Edge: 用 Hausdorff distance between line segments 判断 match

---

## 4. 学到的 metric (Learned Metric)

这是 paper 最有意思的部分之一。Pipeline：

1. **Rendering**: reconstruction 和 GT wireframe 在 3D 中 plot，从 canonical viewpoint render 出 $r_i$
2. **Feature extraction**: 用 **DiNOv2** [Oquab et al., 2023, https://arxiv.org/abs/2304.07193] 提取 rendering 的 features
3. **Regression head**: MLP，输入 DiNOv2 features，输出 scalar score $g(r_i)$
4. **Training objective**: Bradley-Terry probability model，pairwise annotations 监督，binary cross-entropy loss，batch size 16
5. **Cross-validation**: 10-fold，重要——splits 让 training 和 test 中的 **GT structures 和 reconstruction methods 都 disjoint**

Bradley-Terry model（Eq. 4, 5, 6）：

$$P(i > j) = \frac{a_i}{a_i + a_j} = \sigma(\theta_i - \theta_j)$$

$$p_{ij} = \sigma\left(\frac{\theta_i - \theta_j}{s} + o\right)$$

变量：
- $a_i, a_j$：latent "strength" of item $i, j$，正实数
- $\theta_i$：reparameterized log-strength，实数
- $\sigma(x) = 1/(1+e^{-x})$：sigmoid
- $s$：scale parameter
- $o$：offset
- $s=1, o=0$：standard Bradley-Terry
- $s=400, o=800$：Elo scoring system（chess 用的）

Loss（Eq. 7）：

$$\mathcal{L} = \mathbb{E}_{(i,j)}[-y \log(p_{ij}) - (1-y) \log(1 - p_{ij})]$$

$y=1$ if $i$ chosen over $j$，否则 $y=0$。用 SGD + Adam 优化，$\theta_i$ 从独立 Gaussian 初始化。

**结果**: 10-fold CV average accuracy **76%**，prediction 正确的定义是 $g(r_{\text{winner}}) > g(r_{\text{loser}})$。

这个 accuracy 看起来不高，但要理解：human-human agreement 也才 80-85%。所以 learned metric 已经接近 rater noise ceiling。

---

## 5. 关键 Observations 与直觉

### 5.1 Observation 4.1: 人类分两 cluster

平均 agreement ~80%，但 decisive pairs（不选 "equal"）达到 91%。人类分两 cluster：
- **Cluster 1** (raters A-G, CV, Des1-2): 关注 vertex accuracy，与 corner recall / corner F1 相关
- **Cluster 2** (raters H-K, Des0): 关注 edge accuracy，与 edge recall / edge F1 / Jaccard 相关

**重要**: cluster 与 background 无关。3D modeler 也分散在两个 cluster 里。这意味着即使是 domain expert，关注点也不同。

### 5.2 Observation 4.2: Recall > Precision

**这是最重要的 insight**。人类更关心 reconstruction 的 correct parts，而不是 incorrect parts。无论 vertex 还是 edge，**recall metrics 与人类偏好更一致**。

直觉解释：reconstruction 是给 3D modeler 当参考的。如果所有 correct edges 都在，modeler 可以手动删除 wrong edges；但如果 correct edges missing，reconstruction 本身没有任何信息让 modeler 推断出 missing edge 的位置。这是 information 不对称——presence 的 informational value 大于 absence。

Table 1 的 win rate 强烈支持这一点：

| Method | Win Rate |
|---|---|
| add_low | 0.89 |
| add_med | 0.86 |
| perturb_med | 0.85 |
| add_high | 0.82 |
| perturb_low | 0.79 |
| remove_low | 0.79 |
| perturb_high | 0.67 |
| deform_med | 0.67 |
| deform_low | 0.66 |
| remove_high | 0.65 |
| remove_med | 0.63 |
| kc92 (best challenge entry) | 0.51 |

`add_low`（加少量 wrong edges）比 `remove_low`（删少量 correct edges）排名高，尽管两者都引入同等"数量"的误差。`add_high`（加大量 wrong edges）排名 0.82，比 `perturb_med`（中等扰动）高，因为 recall 还在。

### 5.3 Observation 4.3: WED 最差

WED-based scores 与 annotator 一致性最低。WED_S23DR（preregistration 版本）甚至只略好于 random chance。这与 metric hacking observation 呼应：WED 对 collinear split 极敏感，而人类根本不在乎。

### 5.4 Observation 4.4: VLM 没用

测试了 GPT-4o, o1, Grok 2, Qwen-2.5, Pixtral 12B, Claude 3.5/3.7, Gemini 2.0/2.0-flash（via OpenRouter）。Prompt 在 supplementary 里。结果：除了 OpenAI 模型和 Grok 2 略好于随机，其他基本等于 random。**VLM 与 WED family 一致性最高**——讽刺，因为 WED 也最差。

直觉：wireframe comparison 需要精细的 spatial reasoning + 3D structure understanding，VLM 在 2D rendering 上理解 3D topology 能力有限，对 collinear split 这种 subtle 几何差异不敏感。

### 5.5 Observation 4.5: 低质量 regime 人类区分不开

人类对所有 quality threshold 以下的 solution 一视同仁。**完全没有 reconstruction 比有 totally wrong reconstruction 排名还高**。直觉：totally wrong reconstruction 是 misleading 的，浪费 modeler 时间；no reconstruction 让 modeler 直接从头做，至少 mental model 干净。

### 5.6 Observation 4.6: 单一 quality factor 存在

作者用三种方法把 pairwise comparisons 映射到 global ranking：
1. **Simple Win Rate**: 类似 chess 积分，win=1, tie=0.5, loss=0
2. **Bradley-Terry Model**: 估计 latent ability $\theta_i$
3. **Factor Analysis**: methods × raters matrix $M$，log-odds $\eta = \log(M/(1-M))$，SVD 分解，取 first left singular vector

SVD ranking 与 BT ranking Kendall 相关 > 0.7。说明存在一个 dominant "quality" factor 驱动 rater 判断。

---

## 6. Unit Tests: 数学性质检验

作者提出 17 个 unit tests 检验 metric 是否满足数学性质 + 实用性质：

- **Identity of Indiscernibles**: $d(x,x) = 0$
- **Symmetry**: $d(x,y) = d(y,x)$
- **Triangle Inequality**: $d(x,z) \le d(x,y) + d(y,z)$
- **Monotonicity**: 删除 wrong edges 不应增加 dissimilarity；添加 correct edges 不应增加 dissimilarity。具体 sub-tests:
  - Monotonic (wrong edges): 加 wrong edges 后分数变差
  - Monotonic (deform/split): edge split + perturb 后分数变差
  - Monotonic (moving vertex): vertex 移动后分数变差
  - Monotonic (disconnect edges): edge disconnect 后分数变差
  - Monotonic (delete vertices): 删 vertices 后分数变差
  - Monotonic (delete edges): 删 edges 后分数变差
- **Quasi-proportionality**: 平滑扰动下分数平滑变化

每个 property 应用 10 次，要求连续 10 次单调才算 pass。

Table 2 结果（pass count，满分 17）：

| Metric | Pass Count |
|---|---|
| Corner F1 | 12/17 |
| Corner Prec | 11/17 |
| Corner Rec | 11/17 |
| Edge F1 | **14/17** |
| Edge Prec | 12/17 |
| Edge Rec | 13/17 |
| IoU/Jaccard | 11/17 |
| WED AP | 13/17 |
| WED MNN | 10/17 |
| WED nearest | 10/17 |
| WED prereg | 8/17 |
| Spectral L1 | 13/17 |
| Spectral L2 | 8/17 |
| Hausdorff | 6/17 |
| Chamfer edge | 8/17 |

关键 observation：

- **Edge F1** 通过最多 unit tests，且与人类偏好一致——推荐用
- **WED preregistration** 通过最少 tests，且与人类偏好最差——双重失败
- **Spectral L1** 在 unit test 上表现好（13/17），但与人类 alignment 差——说明数学性质好不等于好用
- **Chamfer edge** 反过来：unit test 差（8/17）但 alignment 还行
- **Hausdorff** 双重失败：6/17 + alignment 差。Hausdorff 对 outlier 极敏感，一个错误 vertex 就能让距离爆炸

---

## 7. 作者的 recommendations

### 7.1 竞赛 / benchmark 场景

推荐 **F1-score**（edge F1 + corner F1），尽管 recall metrics 更与人类一致。为什么不用 recall？因为 recall 容易 hack：

> "a dense grid of vertices and edges could score perfectly on recall but be useless in practice and score poorly on precision-based metrics"

F1 同时惩罚 precision，防止 dense-grid hack。但 F1 又基本保留了 recall-favoring 的特性（F1 是 precision 和 recall 的 harmonic mean）。

### 7.2 WED 的问题

WED 设计意图是 estimate "修改 predicted wireframe 到 GT 的 cost"。但实际 edit operations 只包括 vertex/edge insert/delete + vertex movement。3D editing 软件里常用的：

- **Fit one edge to multiple noisy ones**（merge collinear edges）
- **Bulk delete** wrong edges/vertices
- **Rigid transform** 整个 model

这些操作 WED 都不支持，导致 WED 高估了 "fix cost"。

---

## 8. 我对这篇 paper 的思考

### 8.1 对 metric design 的教训

这篇 paper 的核心教训是：**metric 一定要对 invariances 敏感，对 irrelevances 不敏感**。Collinear split 是 geometrically irrelevant——所有几何关系保持不变，但 WED / F1 把它当 major error。Metric hacking 的根源是 metric 和人类 invariance class 不一致。

ImageNet classification 也有类似现象——CNN 对 texture 敏感而人类对 shape 敏感（参考 https://arxiv.org/abs/1811.12231）。但 3D reconstruction 更严重，因为 metric 直接决定 challenge winners，影响整个社区方向。

### 8.2 Recall > Precision 的深层原因

这个 observation 让我想到 information theory 的视角。GT wireframe 有 $N$ 个 edges。Predicted 有 $M$ 个。Recall = |correct predicted| / N。Precision = |correct predicted| / M。

人类偏好 recall 等价于：missing correct element 比多余 wrong element 信息损失更大。这与 "data imputation" 思维一致——absence 是不能被 inference 出来的（除非有 strong prior），但 presence 是可被 verification 的。

也让我想到 detection 任务里的 AP vs AR。COCO 用 AP 主要是为了鼓励 precision，但很多下游任务（autonomous driving 的 pedestrian detection）实际更关心 recall。

### 8.3 Learned metric 的潜力与风险

DiNOv2 + MLP + Bradley-Terry 达到 76%，接近 human noise floor 80%。这暗示 learned metric 已经基本 work。但作者明确警告：**不要用 learned metric 作为 challenge 唯一 adjudication mechanism**，因为 reward hacking / gradient-based adversarial attacks 风险。

这个 warning 很关键。任何 learned metric 都是 differentiable 的，attacker 可以反向传播到输入 wireframe，找一个让 score 高但实际 garbage 的 reconstruction。Handcrafted metric 的"防御"在于它的 non-differentiability 和离散性。

### 8.4 VLM 的失败意味什么

VLM 在这个任务上失败让我 surprise 但又 not surprise。Surprise 是因为 VLM 在很多 vision task 上很强；not surprise 是因为 wireframe comparison 需要的 reasoning 类型——精确的 3D 几何 + topology comparison——是 VLM 当前最弱的环节。VLM 的训练数据里几乎没有 "判断两个 wireframe 哪个更接近 GT" 的样本。

GPT-4o / Grok 2 略好于随机可能是因为它们 reasoning 能力略强，能 follow prompt 指令做一些 explicit comparison。但本质上 vision encoder 对 wireframe 的几何理解很弱。

### 8.5 与我好恶句式的联系

我注意到 paper 在 Observation 4.2 里直接说 "Human annotators pay more attention to correct parts of the reconstruction than the incorrect parts"——直接陈述事实，没说 "不是 X 而是 Y"。这种直接陈述让 observation 清晰。学术写作就该这样。

---

## 9. 与其他工作的联系

### 9.1 ImageNet metric hacking 的历史

ImageNet top-1 accuracy 长期被认为 hack-able：adversarial examples、distribution shift（ImageNet-C, ImageNet-R, ImageNet-Sketch）、long-tail（ImageNet-LT）都暴露了 top-1 accuracy 的局限。这篇 paper 在 3D reconstruction 领域做了类似的"metric critique"。

### 9.2 PC2WF (Liu et al., ICLR 2021)

PC2WF [https://arxiv.org/abs2103.01793] 是第一个用 WED 训练 wireframe reconstruction 的工作。这篇 paper 显示 WED 与人类偏好 anti-correlated，意味着 PC2WF 在优化错误目标。后续工作（LC2WF, BMVC 2022）可能都受影响。

### 9.3 S23DR Challenge (CVPR 2024)

S23DR challenge 用 WED_S23DR 作为官方 metric，决定 winners。这篇 paper 揭示 WED_S23DR 是 "just slightly better than random chance"，意味着 challenge ranking 可能 mislead 整个社区。这是个 serious issue。

### 9.4 Bradley-Terry / Elo 的应用

Bradley-Terry model 在 ML benchmarking 里越来越常见，例如 LMSYS Chatbot Arena（https://chat.lmsys.org）用 BT 给 LLM 排名。AlphaGo 也用类似 BT 思想。这篇 paper 把 BT 应用到 3D reconstruction metric learning，pipeline 设计值得借鉴。

### 9.5 DiNOv2 (Oquab et al.)

DiNOv2 [https://arxiv.org/abs/2304.07193] 的 self-supervised visual features 在很多下游任务上表现优秀。这篇 paper 用 DiNOv2 features 做 metric regression head，证明 DiNOv2 features 已经 encode 了足够 visual information 让 MLP 学出 reasonable 评分。

### 9.6 Graph spectral methods

Spectral graph methods 在 shape retrieval（SHREC benchmarks）和 graph matching 上历史悠久。这篇 paper 显示 spectral L1 在数学性质上 pass 13/17，但 human alignment 差，说明 spectral distance 不直接对应人类对 wireframe 的"质量"判断。

### 9.7 MOTChallenge, BOP Challenge 等其他 benchmark

paper 提到 object tracking (MOTChallenge)、6D pose (BOP)、image retrieval (Oxford/Paris)、optical flow (KITTI) 等都有 standard metric。这些 metric 经过 community validation，但 3D reconstruction 缺乏类似 validation。这篇 paper 是 first systematic attempt。

---

## 10. 局限性与开放问题

paper 没有深入讨论但值得思考的：

1. **Domain shift**: 标注只在北美 residential buildings 上做。Commercial buildings、non-Western architecture、indoor scenes 的偏好可能不同。
2. **Cross-rater cluster 解释**: 为什么人类分两 cluster？是否反映 cognitive style 差异？作者没有进一步分析。
3. **Learned metric 的 generalization**: 10-fold CV 内 disjoint structures + methods，但训练数据量小（3510 pairs × 17 raters ≈ 60k comparisons）。是否能 scale 到更大数据？
4. **Metric 组合**: 是否可以 ensemble 多个 metric 取得更好 alignment？paper 没尝试。
5. **Reward hacking 的具体形式**: 作者提到 dense grid hack，但没系统 study learned metric 的 adversarial vulnerability。
6. **WED 改进方向**: paper 提到 WED 缺 fit-one-edge-to-many、bulk-delete、rigid-transform 操作，但没实现改进版。

---

## 11. 结论

这篇 paper 是对 structured 3D reconstruction community 的一个 important wake-up call。它系统证明：

1. 社区常用 metric（WED 系列）与人类专家偏好 anti-correlated
2. 人类偏好 recall > precision，但竞赛 metric 应该用 F1 防 hack
3. 数学性质好（如 spectral）≠ human alignment 好
4. VLM 当前不能替代 human judgment
5. 学到的 metric（DiNOv2 + BT）能达到 human noise floor 附近的 accuracy

如果你做 3D reconstruction，建议立刻切换 metric 到 edge F1 + corner F1，避免 WED。如果做 metric research，这篇 paper 的 unit test framework + annotation protocol 是 template。

参考链接：

- Paper repo: https://github.com/s23dr/wireframe-metrics-iccv2025
- S23DR Challenge: https://huggingface.co/usm3d
- Building3D dataset: http://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Building3D_An_Urban-Scale_Dataset_and_Benchmarks_for_Learning_Roof_ICCV_2023_paper.pdf
- PC2WF: https://arxiv.org/abs/2103.01793
- DiNOv2: https://arxiv.org/abs/2304.07193
- HEAT: https://arxiv.org/abs/2203.10379
- LMSYS Chatbot Arena (BT application): https://chat.lmsys.org
- Original Bradley-Terry paper: https://www.jstor.org/stable/2334029
