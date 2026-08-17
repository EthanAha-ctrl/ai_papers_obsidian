---
source_pdf: Fast Enough to Act Spatio-Temporal Visual Token Merging for Low-Latency.pdf
paper_sha256: f465f008b8fd8e6607d3293da9eea3936576b06f7b15433564f98896bcd22986
processed_at: '2026-08-04T06:52:16-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 ST-Merge

## 一、这 paper 在解决什么问题

想象你在操控一个机械臂做精细任务，比如把 AirPods 放到橘子旁边。你眼睛看到画面，大脑决定动作，手去执行。

问题是：**现在的 VLA 模型 "想" 太慢了**。

- π0.5 在 244×244 低分辨率下要 241ms 一步
- 在 1024×1024 高分辨率下要 2614ms（2.6秒）一步

2.6 秒一步是什么概念？机械臂早就撞到东西了或者位置已经变了。机器人需要 10-30Hz 的控制频率，也就是 30-100ms 一步。

**为什么这么慢？** 因为 attention 是 $O(n^2)$ 复杂度。1024×1024 分辨率会产生海量的 vision token，数量一爆炸，attention 计算量就平方级爆炸。

**现在的 workaround 是什么？** 直接把分辨率降到 244×244。这就像让你戴老花镜做显微手术——能做但看不清细节。对于 millimeter 级精度的 dexterous manipulation，这是致命的。

所以 dilemma 就是：**高分辨率看得清但太慢，低分辨率快但看不清**。

## 二、ST-Merge 的 core idea

作者的 insight 很朴素：**视频里大部分像素是冗余的**。

想想机械臂的摄像头画面：
- 桌子的背景一直不变，几百个 token 都在编码同样的桌面纹理
- 机械臂在动，但大部分时间只有一小部分区域有变化
- 相邻几帧之间，90% 的内容是重复的

那为什么要把所有这些 redundant token 都喂给 attention 去算？**在 vision encoder 的浅层就把它们合并掉**，这样后续每一层的计算量都省了。

这个 idea 本身不新，ToMe (2022) 就做过 token merging。但 ST-Merge 做了三个关键的改进让它真正能在 robotics 上用。

## 三、三个关键改进

### 改进 1：Multi-queue 并行匹配

ToMe 的做法是把 token 分成两组 A 和 B，A 组的每个 token 去找 B 组里最相似的配对合并。问题：这是串行的二分图匹配，在 video 这种 token 量大的场景下本身就慢。

ST-Merge 的做法：分成 K 组，1 个 anchor 组 + K-1 个 candidate 组。每个 anchor token 可以同时从 K-1 个 candidate 组里各找最好的配对。

打个比方：ToMe 是让你在一个房间里找对象，只能一对一。ST-Merge 是开了 K-1 个候选池，你可以同时从所有池子里挑最合适的，效率上去了。

### 改进 2：Spatio-temporal 位置约束

这是最关键的改进。ToMe 只看 token 的 semantic similarity，不管 spatial 位置。结果可能是：第一帧左上角的桌面 token 和第十帧右下角的桌面 token 被合并了——因为它们语义都是 "桌面"。

但在 robotics 里这要命：**位置信息就是 action 信息**。你把左上和右下的 token 合并了，模型怎么知道机械臂该往左还是往右？

ST-Merge 给每个 token 构造一个 3D 坐标 $(t, y, x)$——时间、行、列。然后用 Gaussian kernel 算两个 token 的位置权重：

$$w_{ij} = \exp\left(-\frac{\|p_i - p_j\|^2}{2\sigma^2}\right)$$

直觉上：离得近的 token 权重接近 1，离得远的权重指数衰减到接近 0。这样匹配的时候只会找 spatio-temporal 邻居，不会跨区域乱合并。

最终的 matching score 是 semantic similarity 乘以 position weight：
$$S'_{ij} = (a_i^\top c_j) \cdot w_{ij}$$

既要语义相似，又要位置相邻，才能合并。

### 改进 3：Post-merge 位置矫正

这个最 subtle 但对 robotics 最重要。

背景：Qwen2.5-VL 用 RoPE (Rotary Position Embedding) 编码位置。RoPE 的核心性质是 attention 只依赖 **relative position**——query 在位置 $m$，key 在位置 $n$，attention score 是 $q^T R_{n-m} k$，只和 $n-m$ 有关。

当你把多个 token 合并成一个，如果你直接继承其中一个 source token 的位置，那它和其他 token 的 relative position 关系就乱了。想象你把位置 3, 4, 5 三个 token 合并成一个，新 token 的位置应该是多少？

ST-Merge 的做法：把合并后 token 的 RoPE sin/cos embedding 设为 source token 的 weighted average：
$$\cos_{\text{new}} = \sum_j \hat{S}_{ij} \cdot \cos_{\text{ori},j}$$

数学上这等价于新 token 的 "effective position" 是 source token 位置的 **weighted centroid**（重心）。

直觉：如果你把相邻几个 patch 合并成一个，新 token 的 "代表位置" 就是这些 patch 的几何重心。这在几何上是 consistent 的——合并后的 token 代表的就是那一片区域的 "中心点"。

最后要 renormalize 保证 $\cos^2 + \sin^2 = 1$，不然 RoPE 的 rotation 性质被破坏。

## 四、实验结果有多好

### Video QA 任务（Qwen2.5-VL）

| 方法 | FLOPs | 延迟 | 准确率 |
|------|-------|------|--------|
| 原版 | 55 TFLOPs | 1043ms | 58.73% |
| ST-Merge | 28 TFLOPs | 415ms | 57.68% |

**延迟砍掉 60%，准确率只掉 1%**。这是个很甜的 tradeoff。

### LIBERO VLA 任务（π0.5）

最惊人的结果在 1024×1024 分辨率：

| 方法 | 1024 延迟 | 1024 准确率 |
|------|-----------|-------------|
| 原版 | 2614ms | 97.50% |
| ST-Merge | 312ms | 97.50% |

**8.3× 加速，准确率零损失**。

为什么高分辨率下效果这么好？因为高分辨率 = 更多冗余 = token merging 更有效。背景里那一大片重复的桌面纹理，merge 掉完全不影响 action 相关的信息。

这正好破解了 latency-resolution dilemma：**高分辨率反而更适合 ST-Merge**。

### 真实机器人（SO-ARM101）

| 方法 | Place | Put | Move | Thinking Time |
|------|-------|-----|------|---------------|
| 原版 | 80% | 70% | 50% | 582ms |
| ST-Merge | 80% | 75% | 55% | 191ms |

Thinking time 从 582ms 砍到 191ms。更意外的是 **Put 和 Move task 的成功率反而上升了**。

为什么？因为 closed-loop control 里 latency 本身就是 accuracy 的一部分。机械臂在 alignment phase 需要高频视觉反馈，如果感知滞后，动作就会 pause 或 oscillate，导致抓取失败。ST-Merge 让感知跟上动作，执行更稳定，成功率反而提高。

## 五、Ablation 教会我们什么

最有教育意义的 ablation：

| 加了什么 | FLOPs | 延迟 | 准确率 |
|----------|-------|------|--------|
| Baseline | 55 | 1044ms | 58.73% |
| 只加 ToMe | 42 | 849ms | 53.41% |
| + Queue 并行 | 28 | 381ms | 53.78% |
| + 位置约束 | 28 | 401ms | 56.90% |
| + 位置矫正 | 28 | 415ms | 57.68% |

读法：
1. **只做 token merging**：速度上去了，准确率崩了 5.3%。说明 video 里乱合并会丢关键信息。
2. **加 multi-queue**：延迟大幅下降（849→381ms），但准确率没恢复。说明 speed 不等于 quality。
3. **加 spatial 约束**：准确率从 53.78% 跳到 56.90%。**这是 accuracy 的核心**——spatial constraint 让合并变得 "聪明"。
4. **加 positional correction**：再涨 0.78%。看似小，但在 millimeter 级 robotics 任务上这点 precision 很关键。

**Intuition**: Speed 来自 queue 并行，accuracy 来自 spatial 约束，precision 来自 positional correction。三者缺一不可。

## 六、我的 takeaways

1. **Token reduction 的位置比数量更重要**：在 vision encoder 浅层 reduce 一点点，比在 LLM decoder reduce 很多都有效。因为 $O(n^2)$ 的 attention 让 early reduction 的 saving 复合放大。

2. **高分辨率场景反而是 token merging 的 sweet spot**：这反直觉但合理——分辨率越高，冗余越多，merge 空间越大。ST-Merge 反而让 "高分辨率 + 低延迟" 不再矛盾。

3. **Geometric consistency 是 robotics 的硬约束**：普通 VLM 丢点 spatial 信息无所谓，robotics 要 millimeter precision，positional encoding 的任何 distortion 都直接变成 action error。Post-merge positional correction 这个细节是 paper 的 secret sauce。

4. **Closed-loop 场景里 speed 就是 accuracy**：在 offline 任务上 latency 和 accuracy 是两个独立 metric，但在 closed-loop robotics 上 latency 直接影响 temporal alignment，从而影响 success rate。Table IV 里 Put/Move task 成功率上升就是实证。

5. **Training-free 的代价**：ST-Merge 不需要训练，plug-and-play，但 matching 的 hyperparameter ($\sigma$, merge ratio $r$, queue 数 $K$) 是固定的。理想情况下应该 data-driven，根据 scene 动态调整。这是 future work 的空间。

## 七、什么场景该用 ST-Merge

- ✅ High-resolution VLA deployment（1024×1024 + real-time control）
- ✅ Video understanding with long streams（大量帧间冗余）
- ✅ Robot manipulation 需要 spatial precision 的任务
- ⚠️ Low-resolution 场景要小心，paper 显示 244×244 下会掉 1.67% accuracy
- ⚠️ Safety-critical 场景需要 fallback，因为 token merging 可能 merge 掉小障碍物

## 八、一句话总结

**别 brute-force 降分辨率，把冗余的 spatio-temporal token 智能合并掉，高分辨率和低延迟可以兼得。**

Code: https://github.com/Junzhou-Chen/ST_Merge

---

# ST-Merge: Spatio-Temporal Visual Token Merging for Low-Latency Robotic VLMs/VLAs 深度解析

## 一、Paper 的核心 Intuition

这篇 paper 解决的是 robotic VLM/VLA 部署中一个特别 concrete 的痛点：**latency-resolution dilemma**。

考虑一个 physical robot 要做 closed-loop control，它需要以 ~10-30Hz 的频率刷新动作。但是 SOTA 的 VLA 比如π0.5 在 244×244 三视角输入下推理就要 241ms，在 1024×1024 下要 2614ms，完全没法 real-time。问题根源在于 attention 的 $O(n^2)$ complexity，而 vision token 在高分辨率+多帧场景下数量爆炸。

作者的 core insight 是：**视觉 token 在 spatio-temporal 维度上是高度 redundant 的**，特别是 static background regions 和 consecutive frames 中重复的 visual content。如果能把这些 redundant tokens 在 vision encoder 的 shallow layer 就合并掉，那后续所有层的计算量都会受益（compounding effect）。

这和 ToMe (Token Merging, Bolya et al. 2022, https://arxiv.org/abs/2210.09461) 的 idea 类似，但是 ST-Merge 做了三个关键的 engineering 改进：

1. **Multi-queue parallel matching** 代替 bipartite matching，实现 $O(n)$ 复杂度
2. **Explicit 3D spatio-temporal coordinates** 作为 matching prior，避免 cross-region 误合并
3. **Post-merge positional correction**：重新计算 RoPE 的 sin/cos embedding，保证 geometric consistency

## 二、Method 的技术细节

### 2.1 Position-Aware Distance Computation

对每个 vision token $i$，构造一个 3D spatio-temporal coordinate：
$$p_i = (t_i, y_i, x_i)$$

其中：
- $t_i$：token 所属的 frame index（time 维度）
- $y_i$：在 feature grid 中的 row index（spatial height 维度）  
- $x_i$：在 feature grid 中的 column index（spatial width 维度）

然后用 Gaussian kernel 计算两个 token 之间的 position weight：
$$w_{ij} = \exp\left(-\frac{\|p_i - p_j\|^2}{2\sigma^2}\right)$$

变量含义：
- $w_{ij}$：token $i$ 和 token $j$ 的 spatio-temporal neighborhood weight，值在 $(0, 1]$ 之间
- $\|p_i - p_j\|^2$：两个 token 在 3D spatio-temporal space 的 Euclidean distance squared
- $\sigma$：Gaussian kernel 的 bandwidth，控制 neighborhood 的大小。$\sigma$ 大 → 越多 tokens 被视为邻居，$\sigma$ 小 → 只考虑很近的 token

**Intuition**: 这个 weight 告诉 matching 算法 "你只应该考虑在空间和时间上都靠近的 token 去合并"，避免把 video 里第一帧左上角的 token 和第十帧右下角的 token 合并（即使它们 semantic 相似）。这对 robotics 特别重要，因为 spatial localization 的精度直接决定 action 的精度。

**Multi-queue partition**: 传统 ToMe 把 token 分成两个 set $G_1, G_2$ 做 bipartite matching，这样同一个 set 内的 token 不能互相 merge。ST-Merge 把 token 分成 $K$ 个 sub-queue：$G_1$ 是 anchor queue，$G_{2:K}$ 是 candidate queues。每个 anchor token 可以从 $K-1$ 个 candidate queue 中各选一个最相似的，实现 one-to-many parallel merge。

### 2.2 Weighted Spatio-Temporal Matching

给定 anchor array $G_1 = \{a_i\}$ 和 candidate array $G_{2:K} = \{c_j\}$，计算 weighted similarity：
$$S'_{ij} = (a_i^\top c_j) \cdot w_{ij}$$

其中：
- $a_i^\top c_j$：anchor token $a_i$ 和 candidate token $c_j$ 的 key vector dot product（cosine similarity 形式），表示 semantic similarity
- $w_{ij}$：前面算出的 spatio-temporal neighborhood weight
- $S'_{ij}$：final weighted similarity score

然后对每个 anchor $a_i$，选最佳的 candidate：
$$b_{j^*} = \arg\max_j S'_{ij}, \quad s_i = \max_j S'_{ij}$$

- $b_{j^*}$：anchor $a_i$ 的 best matching target token
- $s_i$：对应的 best similarity score

接着把所有 anchor 按 $s_i$ 降序排列，选 top-$r$ 个 high-confidence pair 做 merge。这个 r 控制了 token reduction 的 aggressive 程度。

**Intuition**: 这个 matching 是 "soft bipartite matching"，因为允许同一个 target 接收多个 source token，而且通过 multi-queue parallel 把复杂度从 ToMe 的二分图匹配降到 $O(n)$。

### 2.3 Token Merging and Feature Aggregation

把 source token $a_i$ merge 到 target token $b_{j^*}$ 里：
$$v_{b_{j^*}} \leftarrow \frac{s_{b_{j^*}} v_{b_{j^*}} + s_{a_i} v_{a_i}}{s_{b_{j^*}} + s_{a_i}}, \quad s_{b_{j^*}} \leftarrow s_{b_{j^*}} + s_{a_i}$$

变量：
- $v_{b_{j^*}}$：target token 的 feature vector，做 size-weighted average update
- $v_{a_i}$：source token 的 feature vector  
- $s_{b_{j^*}}$：target token 累积的 weight（初始是它自己的 similarity score）
- $s_{a_i}$：source token 的 similarity score

这个 formulation 类似 running mean，每次 merge 后 $s$ 累加，下一个 token merge 进来时会按累计 weight 加权，保证 feature magnitude 不被过度稀释。

### 2.4 Post-Merge Positional Correction（关键创新）

这部分是 paper 最有 insight 的细节。背景：Qwen2.5-VL 用 RoPE (Rotary Position Embedding, Su et al. 2024, https://arxiv.org/abs/2104.09864) 对 query 和 key 做 rotation：

$$q_m = R_m q, \quad k_n = R_n k$$

其中 $R_m, R_n$ 是基于 position $m, n$ 的 rotation matrix。Attention score：
$$\text{Attention}(q_m, k_n) = \text{Re}\left((R_m q)^* (R_n k)\right) = \text{Re}\left(q^* R_{n-m} k\right)$$

这暗示了 attention 只依赖于 **relative position** $n-m$。当多个 token merge 成一个，如果直接继承其中一个 source token 的 position，那 relative position 关系就被破坏了。

ST-Merge 的解决方案：
1. 记录每个 merged token $i$ 包含的 source token $j$ 的 contribution weight $S_{ij}$
2. Normalize: $\hat{S}_{ij} = \frac{S_{ij}}{\sum_j S_{ij}}$（这里 paper 公式有 typo，应该是 $\sum_j$ 而不是 $\sum_i$，因为每个 merged token $i$ 内部 normalize）
3. 对 sin/cos embedding 做 weighted sum：
$$\cos_{\text{new},i} = \sum_j \hat{S}_{ij} \cdot \cos_{\text{ori},j}, \quad \sin_{\text{new},i} = \sum_j \hat{S}_{ij} \cdot \sin_{\text{ori},j}$$
4. Renormalize $(\cos_{\text{new},i}, \sin_{\text{new},i})$ 保证 rotation invariance

**Intuition**: 这里有个 subtle 的数学问题。RoPE 的 rotation 是 $R_m = \text{diag}(e^{im\theta_1}, e^{im\theta_2}, ...)$，对应到每个 dim 是 $\cos(m\theta_d) + i\sin(m\theta_d)$。如果我们对两个 position $m_1, m_2$ 的 sin/cos 做 linear interpolation，得到的是 $\cos(\bar{m}\theta_d) + i\sin(\bar{m}\theta_d)$ 其中 $\bar{m} = \hat{S}_{i1} m_1 + \hat{S}_{i2} m_2$（weighted average position）。

这意味着 merge 后的 token 的 effective position 是其 source token 的 **weighted centroid**。这是 geometrically consistent 的：如果几个 spatially adjacent patch merge 成一个，新 token 的 "代表位置" 就是这些 patch 位置的重心。这对 robotics 的 spatial reasoning 至关重要。

但 weighted sum 会让 $\cos^2 + \sin^2 \neq 1$，所以需要 renormalize。这个 step 保留了 RoPE 的 rotation 性质。

### 2.5 Deployment Strategy

FLOPs 节省公式：
$$\text{FLOPs}_{\text{saved}} = \sum_{k=l+1}^{L} \left[\alpha_k \left(N_{k,\text{orig}}^2 - N_{k,\text{red}}^2\right)\right]$$

变量：
- $l$：插入 ST-Merge 的 layer index
- $L$：vision encoder 总层数
- $\alpha_k$：第 $k$ 层的 attention 计算系数（与 hidden dim 相关）
- $N_{k,\text{orig}}$：原始 token 数
- $N_{k,\text{red}}$：reduce 后的 token 数

**Intuition**: 这个公式展示了 "early merge" 的 compounding benefit。在 layer $l$ merge 掉 tokens 之后，layer $l+1, l+2, ..., L$ 的 attention 计算都按 $N^2$ 减少。所以越早 merge，节省越多。这也是为什么 ST-Merge 在 vision encoder shallow layer 插入，而不是在 LLM decoder 里（FastV, VTW 那种做法）。

**Qwen2.5-VL vs π0.5 的差异**：
- Qwen2.5-VL：windowed attention 为主，少数 global attention layer。ST-Merge 只插入在 shallow global attention layers，因为 windowed attention 下跨 window 的 token similarity 不可比。
- π0.5：visual encoder 基于 SigLIP，全用 global attention，没有 RoPE。所以不需要 positional correction，只做 spatial neighborhood（无 temporal），单帧多视角。

## 三、实验数据深度解读

### 3.1 Video QA on Qwen2.5-VL（Table I）

| Method | TFLOPs | Speed (ms) | Robot Obj | Temporal | Physics | Avg |
|---|---|---|---|---|---|---|
| Baseline | 55.07 (100%) | 1043.61 | 58.55 | 63.12 | 54.53 | 58.73 |
| ToMe | 41.58 (75.5%) | 848.52 | 53.25 | 58.00 | 49.00 | 53.41 |
| FastV | 46.96 (85.3%) | 719.10 | 54.60 | 60.25 | 52.20 | 55.68 |
| TempMe | 36.79 (66.8%) | 605.77 | 54.55 | 61.70 | 49.80 | 55.35 |
| **ST-Merge** | **28.11 (51.0%)** | **414.90** | 56.55 | 62.82 | 53.69 | **57.68** |

观察：
- ST-Merge 把 FLOPs 砍掉一半（51.04% of baseline），latency 砍掉 60%（从 1043ms 到 415ms）
- Accuracy 只掉 1.05%（58.73% → 57.68%）
- FastV 虽然 latency 不错但 FLOPs 没降多少（因为它只 optimize LLM decoder 不动 vision encoder）
- TempMe 在 FLOPs 上接近 ST-Merge，但 latency 反而高，因为它用 bipartite matching 在 scale 上有 overhead

**Intuition**: 在 video QA 上，temporal reasoning 任务 ST-Merge 几乎无损（62.82 vs 63.12），这归功于 explicit spatio-temporal modeling。Physics 任务掉得稍多（54.53→53.69），因为 physical understanding 可能依赖 fine-grained visual detail。

### 3.2 Ablation Study（Table III）——最有教育意义

| Method | TFLOPs | Speed (ms) | Acc |
|---|---|---|---|
| Baseline | 55.07 | 1043.61 | 58.73% |
| + ToMe only | 41.58 | 848.52 | 53.41% |
| + Queue Partitioning | 27.90 | 381.22 | 53.78% |
| + Spatio Information | 28.01 | 401.22 | 56.90% |
| + Positional Correction | 28.11 | 414.90 | 57.68% |

关键观察：
1. **ToMe only**：FLOPs 降了但 accuracy 暴跌 5.3%。说明 vanilla token merging 在 video 上失效，因为不区分 frame。
2. **+ Queue Partitioning**：FLOPs 从 41.58 → 27.90（多 queue 允许更多 merge），latency 从 848 → 381ms（parallel 加速明显），但 accuracy 只涨 0.37%。证明 speed↑ 但 quality 没保证。
3. **+ Spatio Information**：accuracy 从 53.78% → 56.90%，涨 3.1%。这是最关键的 step，证明 spatial constraint 是 quality 的核心。FLOPs 几乎不变。
4. **+ Positional Correction**：再涨 0.78%。这 step 的 ROI 看似不大，但作者强调在 robotic dexterous manipulation 上 millimeter 级精度很重要。

**Intuition**: 这个 ablation 告诉我们 token reduction 的 "free lunch" 在哪：Queue partitioning 给 speed，spatial constraint 给 accuracy，positional correction 给 precision。

### 3.3 LIBERO VLA（Table II）

| Method | 244×244 Speed | 244 Acc | 512 Speed | 512 Acc | 1024 Speed | 1024 Acc |
|---|---|---|---|---|---|---|
| π0.5 Baseline | 241.21 | 95.83% | 607.14 | 96.67% | 2614.56 | 97.50% |
| + ST-Merge | 141.55 | 94.16% | 194.32 | 94.16% | 312.11 | 97.50% |

惊人的观察：
- **1024×1024 下 8.3× speedup（2614ms → 312ms），accuracy 完全不变（97.50%）**
- 高分辨率下 ST-Merge 是 "free lunch"：背景 redundancy 极多，merge 掉不影响 action relevant info
- 低分辨率下（244×244）有 1.67% accuracy 下降，因为信息已经稀疏，进一步 merge 损失关键细节

**Intuition**: 这给出了 ST-Merge 的 sweet spot——**resolution 越高，redundancy 越多，ST-Merge 越有效**。这正好解决 "VLA 要 high resolution 但 latency 爆炸" 的痛点。1024×1024 让 VLA 能看清 millimeter 级 detail，同时 ST-Merge 让 latency 可接受。

### 3.4 Real-World SO-ARM101（Table IV）

| Method | Place | Put | Move | Thinking Time |
|---|---|---|---|---|
| w/o ST-Merge | 80% | 70% | 50% | 582.14ms |
| w/ ST-Merge | 80% | 75% | 55% | 191.22ms |

- Thinking time 砍 67%（582→191ms）
- Put 和 Move task 的 success rate 居然 **上升** 了 5%！

**Intuition**: 为什么 accuracy 反而上升？作者的解释很 reasonable：closed-loop control 的 latency 直接影响 temporal alignment between perception 和 action。低频决策导致视觉反馈滞后，机械臂在 alignment phase 出现 pause 或 oscillation。ST-Merge 把 latency 砍下来，恢复了 high-frequency closed-loop control 的能力，更稳定的 decision-execution link 反而提升了 success rate。

这是一个很 deep 的 insight：**在 robotic closed-loop 场景，inference speed 本身就是 accuracy 的一部分**。

## 四、和 Related Work 的关系

### 4.1 Token Reduction 谱系

1. **Pruning 类** (FastV https://arxiv.org/abs/2403.06764, TopV, Learned Token Pruning)：基于 importance score 砍掉 token。问题：可能丢掉关键信息，且多在 LLM decoder 之后做。
2. **Merging 类** (ToMe https://arxiv.org/abs/2210.09461, PuMer https://arxiv.org/abs/2305.17530)：合并相似 token。问题：通常不 modeling spatial/temporal structure。
3. **Video-specific** (TempMe https://arxiv.org/abs/2409.01156, FrameFusion https://arxiv.org/abs/2410.23782)：引入 temporal 维度，但用 bipartite matching 复杂度高。

ST-Merge 的 positioning 是 **"first to merge inside vision encoder shallow layers with explicit spatio-temporal RoPE-aware correction"**。

### 4.2 VLA Latency 问题

参考 OpenVLA (https://arxiv.org/abs/2406.09246)、π0.5 (https://arxiv.org/abs/2504.16054)、SmolVLA (https://arxiv.org/abs/2506.01844)。这些 model 都 struggle with latency。π0.5 用 244×244 是 brute-force downsample 的妥协。ST-Merge 给出 third way：keep high resolution, prune redundancy smartly。

## 五、Critical Thoughts 和 Open Questions

### 5.1 Method 层面的疑问

1. **σ 的选择**: paper 没明说 σ 怎么设。如果是 hand-tuned hyperparameter，跨场景泛化可能有问题。理想情况下应该 data-driven。
2. **Gaussian kernel 的必要性**: 为什么不直接 hard mask 只在 k-neighborhood 内 matching？Gaussian 给 soft prior 但计算开销更高。
3. **Multi-queue 的 K 怎么选**: paper 没详述。K 太小退化成 bipartite，K 太大 anchor-candidate 比例失衡。
4. **Positional Correction 的数学严谨性**: Renormalize $(\cos, \sin)$ 后，effective position 不再严格等于 weighted centroid，而是 weighted centroid 附近的一个 unit vector 方向。这在小 distortion 下近似 OK，但大 merge 比例下可能有偏差。
5. **Renormalize 的细节**: 公式 (5) 的 normalize 分母写的是 $\sum_i S_{ij}$ 而不是 $\sum_j S_{ij}$，这应该是 typo。逻辑上应该是每个 merged token 内部把 source weights normalize 到 1。

### 5.2 Experimental 层面

1. **只在 Qwen2.5-VL 和 π0.5 上测**：泛化到 InternVL (https://arxiv.org/abs/2312.14238)、LLaVA-OneVision (https://arxiv.org/abs/2408.03326) 等其他 VLM 未知。不同 model 的 RoPE 实现细节不同，positional correction 的 transferability 不清楚。
2. **LIBERO 是 simulation benchmark**：simulator 的 visual redundancy 可能比 real world 更明显（uniform background）。real-world 只有 3 个 task 100 trajectories，scale 不够。
3. **和 quantization, distillation 的 orthogonal 性**：没测 ST-Merge + INT8 quantization 的叠加效果。理论上应该 orthogonal。
4. **Long-horizon task**：没测 task 持续 30s+ 的情况，token merging 在长 video 上的 error accumulation 未知。

### 5.3 Robotics 系统层面

1. **Action distribution shift**: token merging 改变了 visual representation，action head 是否需要 fine-tune 来适应？paper 没在 π0.5 上 fine-tune vision encoder，但 LIBERO 上 fine-tune 50K steps。这 fine-tune 是否 compensates 了 ST-Merge 的副作用？需要 ablation。
2. **Safety-critical 场景**: paper 强调 dexterous manipulation，但 safety-critical 场景下 token merging 可能 merge 掉关键 small obstacle，导致 collision。需要 fallback mechanism。
3. **Cerebrum-cerebellum architecture**: π0.5 用了 cerebrum (high-level) + cerebellum (low-level) 架构，paper 只 optimize cerebrum。如果 cerebellum 也处理视觉，latency bottleneck 可能转移。

### 5.4 我的 Intuition 升级

读完这篇 paper 我对 token reduction 的理解升级了几点：

1. **Token reduction 的 "where" 比 "how much" 更重要**: 在 vision encoder shallow layer reduce 1% 比在 LLM decoder reduce 10% 的 end-to-end speedup 还大。因为 attention 是 $O(n^2)$，early reduction 让 saving compound。

2. **Geometric consistency 是 robotics 的硬约束**: 一般 VLM task 上 token merging 丢点信息没事，但 robotics 要 millimeter precision，positional encoding 的 distortion 直接 propagate 到 action error。Post-merge positional correction 这个细节看似小，实则是 paper 的 secret sauce。

3. **Resolution-latency 不是 zero-sum tradeoff**: 传统认为要么 low res fast, 要么 high res slow。ST-Merge 揭示 high res 下 redundancy 多，正是 token merging 的 sweet spot，让 high res 也能 fast。这反直觉但合理。

4. **Closed-loop speed 是 accuracy 的一部分**: 在 offline QA 上 latency 和 accuracy 是两个 metric，但在 closed-loop robotics 上 latency 直接影响 action 的 temporal alignment，从而影响 success rate。Table IV 的 Put/Move task accuracy 上升是这个 insight 的实证。

## 六、可能延伸的方向

### 6.1 学习式 token merging

ST-Merge 是 training-free 的好处是 plug-and-play，但坏处是 matching 的 hyperparameter (σ, r, K) 是固定的。可以想象一个 learned merging policy，根据当前 scene 动态调整。比如 dynamic scene 用更少 merge，static scene 用更多 merge。这个 idea 类似 AdaViT (https://arxiv.org/abs/2112.01619) 的 adaptive computation。

### 6.2 和 Test-Time Training 结合

如果 deployment 时遇到 OOD scene，token merging 的 matching 可能失效。可以用 test-time adaptation 微调 σ 等参数，比如 minimize 一个 self-supervised consistency loss between merged 和 unmerged representations。

### 6.3 Multi-modal token merging

paper 只 merge vision tokens。但 instruction 里也有冗余（"put the cup on the table" 中 "the" 之类）。可以想象 cross-modal merging，比如把 "the red cup" 的 text token 和 vision 中对应 cup 的 token 一起 merge 成一个 grounded representation。这和 PaLI-X (https://arxiv.org/abs/2206.07169) 等 model 的 cross-modal attention 思路呼应。

### 6.4 Merging for Long-horizon Reasoning

Long-horizon robotic task（100+ steps）下，每步都做 token merging 会让 history representation 越来越 compressed。可以想象 hierarchical merging：short-term preserve detail，long-term compress to abstract scene graph。这个 idea 和 1X World Model (https://wayve.ai/thinking/world-models) 等的 hierarchical representation 思路呼应。

### 6.5 Merging + Diffusion Policy

π0.5 用 flow matching (diffusion) 做 action generation。Diffusion 的 iterative denoising 本身计算昂贵。如果 token merging 让 vision encoder 加速 8×，那 diffusion step 数也能相应增加而总 latency 不变，可能改善 action distribution 的 modeling。这个 orthogonal direction 值得 explore。

## 七、Summary

ST-Merge 是一篇 engineering-heavy 但 insight 清晰的 paper。它的 contribution 不在 algorithm novelty（token merging 不是新概念），而在于：

1. 把 token reduction 的战场从 LLM decoder 转移到 vision encoder shallow layer
2. 把 spatial-temporal geometric consistency 显式编码进 matching 和 positional embedding
3. 在真实 VLA deployment 上验证 "high resolution + low latency" 可以兼得

对 robotics VLA 领域，这篇 paper 给出了一个 actionable recipe：不要 brute-force downsample，而是 spatio-temporally merge。这对 future 的 high-precision dexterous manipulation model 部署有直接 reference value。

对更广的 VLM community，paper 提出的 post-merge positional correction 这个 idea 可能 transfer 到其他用 RoPE 的 model（DeepSeek-V3 https://arxiv.org/abs/2412.19437、Qwen3 https://arxiv.org/abs/2505.09388 等）的 token reduction 工作。

Code repo: https://github.com/Junzhou-Chen/ST_Merge

参考 papers:
- ToMe: https://arxiv.org/abs/2210.09461
- FastV: https://arxiv.org/abs/2403.06764
- PuMer: https://arxiv.org/abs/2305.17530
- TempMe: https://arxiv.org/abs/2409.01156
- FrameFusion: https://arxiv.org/abs/2410.23782
- RoPE/RoFormer: https://arxiv.org/abs/2104.09864
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- LIBERO: https://arxiv.org/abs/2306.03310
- SmolVLA: https://arxiv.org/abs/2506.01844
- LLaVA-PruMerge: https://arxiv.org/abs/2403.15388
- InternVL: https://arxiv.org/abs/2312.14238
- LLaVA-OneVision: https://arxiv.org/abs/2408.03326
- FlashAttention: https://arxiv.org/abs/2205.14135
- MVP benchmark: https://arxiv.org/abs/2503.07168
- LeRobot: https://github.com/huggingface/lerobot
