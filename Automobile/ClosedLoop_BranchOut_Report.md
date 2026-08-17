---
source_pdf: ClosedLoop_BranchOut_Report.pdf
paper_sha256: 7e4e2b65125f0281cd2956d56bcae4409a30036f67aeab5d97880040a03b9016
processed_at: '2026-08-03T16:02:04-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的 Karpathy，咱们用大白话把这篇 paper 揉碎了聊聊。我会尽量把那些学术黑话翻译成工程师能直接 get 的直觉，同时保留核心的技术细节。

说白了，这篇 paper 的核心故事非常简单：**当前 autonomous driving 的 planner 都在“抄标准答案”，而且抄得太过分，以至于到了真实世界（closed-loop）就抓瞎。为了解决这个问题，作者给模型装了一个“多选题”机制，强行让它学会人类开车的多种可能性。**

---

### 1. 痛点：自动驾驶的“抄答案”困境与 L2 的暴政

现在的 end-to-end planner（比如 UniAD, VAD）是怎么训练的呢？给它一段场景，让它预测未来 3 秒的 trajectory，然后用 L2 loss（就是简单的欧氏距离）去跟 ground truth 对齐。

问题在于，**开车这件事本质上是多模态的**。比如前面有个减速带，我可以往左打一点绕过去，也可以往右打一点，也可以直接压过去。这三个选择都是“人类合理驾驶”。但在 nuScenes 数据集里，采集车当时只走了一条路，所以 ground truth 只有一条。

如果你用 L2 loss 去训练模型，模型为了让 loss 最小，就会去预测所有可能路径的“平均值”。你可以想象，把往左绕和往右绕的轨迹平均一下，得到的就是直直撞上去。所以 L2 这种 unimodal 的 metric，实际上在狠狠惩罚那些“合理但与 GT 不同”的预测。

这篇 paper 的一个重磅发现是：他们找来 40 个人，戴着 VR 头显拿着 Logitech 手柄在 HUGSIM 仿真器里开 nuScenes 的场景。结果发现，同一场景下人类的轨迹差异（minL2）只有 0.79 米，而目前最强的模型预测误差都在 1.4 米以上。这证明了**人类的驾驶行为分布是非常丰富的，单一 GT 根本代表不了这种分布**。

---

### 2. Diffusion 的局限与 GMM Head 的降维打击

既然要建模多模态分布，很多人第一反应是用 **Diffusion model**。Diffusion 理论上能拟合任何分布，但在实际训练中，它极其容易发生 **mode collapse**（模式崩溃）。也就是说，它倾向于在最常见的那条轨迹周围生成一点微小的抖动，而完全忽略其他合法的驾驶选项（比如转弯时，它只敢稍稍偏一点，不敢真转大弯）。

之前有个工作叫 DiffusionDrive，试图用 K-Means 预先聚类出一些 anchor 来引导 diffusion。但作者觉得这太脱裤子放屁了：为什么我们要先 offline 聚类，而不是让网络自己端到端地把多模态结构学出来？

于是作者提出了 **BranchOut**。它的核心 insight 是：**与其指望 diffusion 隐式地学出多模态，不如在网络的 head 处直接加上显式的 GMM（Gaussian Mixture Model）结构**。

---

### 3. 架构细节拆解（带公式直觉）

整个模型分为三块，非常清晰：

#### A. Scene Encoder（场景编码器）
直接复用了 VAD-Tiny 的架构。输入是 6 个视角的 camera 图像和 map。输出两组 token：
*   $\mathbf{P}_{\text{agent}}$：周围车辆、行人的 embedding
*   $\mathbf{P}_{\text{map}}$：车道线、红绿灯等 map 元素的 embedding
**这里有个反直觉的设计**：作者坚决不输入 ego status（自车的速度、加速度、历史轨迹）。因为有篇 paper（Ego-MLP）证明，模型一旦看到 ego status，就会走捷径，直接用当前速度乘以时间去外推未来轨迹，完全忽略图像里的障碍物。这在 open-loop 评价指标上能刷很高分，但在 closed-loop 里一旦自车状态发生漂移，立马撞车。

#### B. Scene-Aware Diffusion Transformer（带场景条件的 Diffusion 去噪器）
这里用 diffusion 来生成 trajectory。训练时的前向加噪公式是：

$$\mathbf{X}_{\text{ego}}^{(t)} = \sqrt{\alpha(t)} \cdot \mathbf{Y}_{\text{ego}} + \sqrt{1 - \alpha(t)} \cdot \mathbf{z}$$

*   $\mathbf{Y}_{\text{ego}} \in \mathbb{R}^{T_f \times 2}$：真实的未来轨迹，$T_f = 6$ 代表未来 3 秒（每 0.5 秒一个点），2 代表 BEV 坐标 $(x, y)$。
*   $t \sim \mathcal{U}(0,1)$：采样的时间步。
*   $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$：标准高斯噪声。
*   $\alpha(t) = 1 - \sigma^2(t)$：信号保留比例。$t$ 越大，$\alpha(t)$ 越小，轨迹被噪声淹没得越厉害。

然后网络通过 Cross-Attention 把场景信息注入进去：
$$\mathbf{P} = [\text{MHCA}(\mathbf{P}, \mathbf{P}_{\text{agent}}, \mathbf{P}_{\text{agent}}), \text{MHCA}(\mathbf{P}, \mathbf{P}_{\text{map}}, \mathbf{P}_{\text{map}})]$$
这里的 Query 是带噪的轨迹 token，Key 和 Value 分别是 agent 和 map token。这就让轨迹在去噪时能“看到”周围的障碍物。

#### C. Branched GMM Head（核心贡献：分支高斯混合头）
这是全篇最巧妙的地方。网络最后输出时，根据 high-level command $c \in \{\text{Left, Straight, Right}\}$ 走完全不同的 MLP 分支。

每个分支输出 $K$ 个候选轨迹及其概率：
$$\mathcal{G}(\mathbf{P}) = \{(\mu_k^m, \pi_k^m)\}_{k=1}^K$$
*   $\mu_k^m \in \mathbb{R}^{T_f \times 2}$：第 $m$ 个 command 分支下，第 $k$ 条候选轨迹的均值。
*   $\pi_k^m$：这条轨迹的 mixture weight（概率权重，经过 softmax）。

**为什么用 hard branch？** 因为 Left 和 Right 的 gradient 如果共享一个 head，会互相平均掉。强行把路由做死，模型就必须给 Left 分配专门去学左转轨迹的 capacity。

**推理时的 trick**：虽然输出了 K 条轨迹，但在 closed-loop 跑的时候，模型只选 $\pi_k^m$ 最大的那一条（argmax）。这保证了行为的确定性，避免 controller 因为每次采样的轨迹不同而震荡。更狠的是，他们发现 **用 1 步 DPM-Solver 就够了**，多跑几步 diffusion 反而因为噪声漂移导致效果变差。

---

### 4. Loss 设计：软硬兼施

Loss 函数也很有意思，结合了 deterministic 和 probabilistic：
$$\mathcal{L} = \mathcal{L}_{\text{plan}} + \lambda_{\text{NLL}} \mathcal{L}_{\text{NLL}} + \lambda_c \mathcal{L}_{\text{constraints}}$$

1.  $\mathcal{L}_{\text{plan}}$：标准的 diffusion MSE loss，保证模型能把轨迹复原出来。
2.  $\mathcal{L}_{\text{NLL}}$：Negative Log-Likelihood loss。公式大概是 $-\log \sum_{k=1}^K \pi_k \cdot \mathcal{N}(\mathbf{Y}_{\text{ego}} | \mu_k, \Sigma_k)$。这个 loss 的作用是：**强迫 K 条轨迹里，至少有一条得非常贴近真实 GT**。这就防止了 GMM 头输出的 K 条轨迹都偏离目标。
3.  $\mathcal{L}_{\text{constraints}}$：VAD 提出的安全损失，惩罚轨迹撞墙或撞车。

---

### 5. 实验里的反直觉洞察

这篇 paper 的实验数据极其扎实，充满了反直觉的结论：

**洞察一：GMM 结构比 Diffusion 本身更重要。**
作者做了消融实验：
*   去掉 GMM，只留 Diffusion：NLL（负对数似然）是 4.11。
*   去掉 Diffusion，只留 GMM：NLL 降到了 3.80。
*   两个都加上：NLL 最低，3.72。
这说明，**显式的结构化 head 在捕捉多模态上，比隐式的生成式模型更管用**。Diffusion 负责平滑采样，GMM 负责 mode 定位，两者互补。

**洞察二：Open-loop 排名和 Closed-loop 排名完全是两码事。**
在 nuScenes 标准 open-loop 测试里，UniAD 的 L2 误差（1.65m）比 VAD-Base（1.69m）好。但是，一旦用多模态 metric（Frechet distance，计算预测轨迹集与人类真实轨迹集的最小距离）去评估，VAD-Base 反超 UniAD。
更夸张的是在 HUGSIM closed-loop 仿真里：
*   VAD-Tiny 的 HD-Score 只有 **0.11**
*   BranchOut 的 HD-Score 达到了 **0.47**（提升了 4

---

# BranchOut: GMM-based Diffusion Planner 详解

这篇 paper 来自 Boston University 的 Eshed Ohn-Bar 团队，在 RealADSim closed-loop 比赛中拿到第三名（academic team 里第一名），核心思想非常简洁：**用 branched GMM head + diffusion，让 planner 显式建模人类驾驶的多模态决策分布**。我把它当作一个 case study，从 intuition → formulation → architecture → experiments 逐层拆开。

---

## 1. Motivation & Core Intuition

### 现状问题
当前 SOTA motion planner 大多是 **deterministic**（UniAD、VAD、Transfuser 等），它们在 single ground-truth L2 评价下表现良好，但有几个深层问题：

- **L2 评价的偏置**：nuScenes 只有单条 GT trajectory，如果一个 model 预测出一条 "合理但与 GT 不同的轨迹"（比如稍微宽一点的左转），L2 会狠狠惩罚它。这导致 model 趋向于 overfit 到 GT 的具体几何形状，而 ignore 了 human driving 的 inherent multimodality。
- **Diffusion 的 mode collapse**：纯 diffusion 容易在 dominant mode 周围生成小 variations，而忽略次优 mode。DiffusionDrive 用 K-Means pre-clustered anchors 缓解这个问题，但 anchors 是 offline 聚类的，与 model 联合优化脱节。
- **Closed-loop 与 Open-loop 的 gap**：很多 open-loop 表现好的 model 在 closed-loop 中崩盘（VAD-Tiny 在 HUGSIM 只有 0.11 HD-Score），说明预测分布不够 "真实"。

### BranchOut 的核心 insight
作者提出：**与其依赖 diffusion 本身的隐式 multimodality，不如显式在 head 上加 GMM 结构，让 model 直接输出 K 个 (mean, weight) 对**。同时用 high-level command（Left/Straight/Right）作为 hard branch，强制不同 maneuver 走不同的 head。这种 design 比 DiffusionDrive 的 anchor-based 引导更 "end-to-end"，因为 GMM 的 mixture weights 是 learnable 的，与 backbone joint optimize。

---

## 2. Architecture 拆解

整个 pipeline 分三个模块：

```
[Multi-view cameras + Map] 
        ↓
  Scene Encoder F (VAD-Tiny based)
        ↓
  P_agent, P_map (scene embeddings)
        ↓
  Diffusion Transformer Denoiser D (cross-attention with scene)
        ↓
  Branched GMM Head G (selected by command c)
        ↓
  {(μ_k^m, π_k^m)}_{k=1}^K  →  Ŷ (best trajectory)
```

### 2.1 Scene Encoder F
直接复用 **VAD-Tiny 的 encoder**，输出两类 token：
- $\mathbf{P}_{\text{agent}} \in \mathbb{R}^{N_a \times N_d}$：agent embeddings（其他车辆、行人）
- $\mathbf{P}_{\text{map}} \in \mathbb{R}^{N_m \times N_d}$：map embeddings（lane、traffic light 等）

这里 $N_a$ 是 agent 数量，$N_m$ 是 map element 数量，$N_d$ 是 embedding dimension。

**关键 design choice**：**不输入 ego status / ego past trajectory**（与 Ego-MLP 的发现相反）。作者认为 ego history 会让 model 学到 shortcut，而非真正理解 scene。这点在 closed-loop 中可能更 robust，因为 closed-loop error accumulation 会放大 ego-state shortcut 的 bias。

### 2.2 Diffusion Transformer Denoiser D

#### Forward Process（训练时）
公式 (1)：

$$\mathbf{X}_{\text{ego}}^{(t)} = \sqrt{\alpha(t)} \cdot \mathbf{Y}_{\text{ego}} + \sqrt{1 - \alpha(t)} \cdot \mathbf{z}$$

变量含义：
- $\mathbf{Y}_{\text{ego}} \in \mathbb{R}^{T_f \times 2}$：ground-truth 未来 trajectory，$T_f = 6$（3秒，每 0.5s 一个 waypoint，BEV 坐标 $(x, y)$）
- $t \sim \mathcal{U}(0, 1)$：diffusion timestep
- $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$：标准 Gaussian noise
- $\alpha(t) = 1 - \sigma^2(t)$：signal retention ratio，$\sigma(t)$ 是 noise schedule
- $\mathbf{X}_{\text{ego}}^{(t)} \in \mathbb{R}^{M \times T_f \times 2}$：扰动后的 trajectory，$M = 3$ 对应三个 command 的 parallel noisy trajectories

**注意一个细节**：这里 $M=3$ 的 dimension 是 batch-like 的，即三个 command 对应的 trajectory **同时**被加噪并送入 denoiser，而非分别处理。这让 model 在一次 forward 中就能 reason across maneuver modes。

#### Denoiser 的 scene conditioning
公式 (2)：

$$\mathbf{P} = [\text{MHCA}(\mathbf{P}, \mathbf{P}_{\text{agent}}, \mathbf{P}_{\text{agent}}), \text{MHCA}(\mathbf{P}, \mathbf{P}_{\text{map}}, \mathbf{P}_{\text{map}})]$$

变量：
- $\mathbf{P} \in \mathbb{R}^{M \times N_p}$：noisy trajectory 经过 linear projection 后的 embedding
- $\text{MHCA}(q, k, v)$：multi-head cross-attention，queries 来自 ego trajectory tokens，keys/values 来自 scene tokens
- 输出 $[\cdot, \cdot]$ 是 concatenation，把 agent-conditioned 和 map-conditioned features 合在一起

这是一个很标准的 cross-attention conditioning，跟 TransFuser 的 sensor fusion 思路类似。

### 2.3 Branched GMM Head G（核心 contribution）

公式 (3)：

$$\mathcal{G}(\mathbf{P}) = \{(\mu_k^m, \pi_k^m)\}_{k=1}^K$$

变量：
- $\mu_k^m \in \mathbb{R}^{T_f \times 2}$：第 $m$-th command branch 下第 $k$-th Gaussian component 的 mean trajectory
- $\pi_k^m$：对应 mixture weight（softmax normalized）
- $K$：每个 branch 的 GMM component 数量
- $m \in \{\text{Left, Straight, Right}\}$：由 high-level command $c$ 选定

**Branched structure 的具体形式**：

$$\bar{\mu}^m = \text{MLP}_{\mu}^m(\mathbf{P}) \in \mathbb{R}^{K \times T_f \times 2}$$
$$\pi^m = \text{MLP}_{\pi}^m(\mathbf{P}) \in \mathbb{R}^K$$

即**每个 command 有两套独立的 MLP**：一套输出 $K$ 条候选 trajectory 的均值，一套输出 $K$ 个 mixture weight。这相当于在 head 上做了 **hard routing**——Left 的 maneuver 只走 Left 的 MLP，不与 Right 的 MLP share weights。

这种 design 的好处：
1. **Gradient isolation**：不同 maneuver 的 gradient 不会通过 shared head 互相干扰
2. **Capacity allocation**：每个 maneuver 有独立的参数 budget 来精细刻画其 sub-multimodality（比如 "Right turn tight" vs "Right turn wide"）
3. **Sample efficiency**：相比 shared head + command embedding 的 conditioning，hard branch 更容易学

从 ablation（Table 1）验证：
- BranchOut w/ Shared Head: L2 0.87, Frechet 2.41
- BranchOut (branched): L2 0.83, Frechet 2.29

确实 branched 略好，但 gain 不算巨大（~5%）。说明 branch 的主要价值在 closed-loop 中体现（route completion 大幅提升）。

---

## 3. Training Loss

公式 (4)：

$$\mathcal{L} = \mathcal{L}_{\text{plan}} + \lambda_{\text{NLL}} \mathcal{L}_{\text{NLL}} + \lambda_c \mathcal{L}_{\text{constraints}}$$

三部分：
1. **$\mathcal{L}_{\text{plan}}$**：标准 diffusion 的 denoising loss（在 noise $\mathbf{z}$ 上的 MSE），公式上等价于 $\|\hat{\mathbf{Y}} - \mathbf{Y}_{\text{ego}}\|^2$
2. **$\mathcal{L}_{\text{NLL}}$**：对 GMM 分布的 negative log-likelihood
   $$\mathcal{L}_{\text{NLL}} = -\log \sum_{k=1}^K \pi_k \cdot \mathcal{N}(\mathbf{Y}_{\text{ego}} | \mu_k, \Sigma_k)$$
   这个 loss 强制 GMM 的某个 component 要在 GT 附近有 high density，避免 mode collapse 到一个点
3. **$\mathcal{L}_{\text{constraints}}$**：safety constraint loss（来自 VAD），惩罚 ego 与 agent / map boundary 的 collision

$\lambda_{\text{NLL}} = \lambda_c = 0.1$，weight 不大，说明 plan reconstruction loss 仍是主导，NLL 起到 "shape the distribution" 的辅助作用。

---

## 4. Inference 细节

- 初始化：$\mathbf{X}_{\text{ego}}^{(1)} \sim \mathcal{N}(0, \mathbf{I})$
- Solver：**single-step DPM-Solver++** [Lu et al., NeurIPS 2022 / Machine Intelligence Research 2025]
- Denoising steps：从 Table 2 看，**1 步就够**（L2 0.83），更多步反而略差（10 步 L2 0.86）。这说明 GMM head 已经把 multimodality 撑住了，diffusion 的 iterative refinement 反而引入 noise drift。

**最终输出选择**：从 K 个 $(\mu_k^m, \pi_k^m)$ 中选 $\arg\max_k \pi_k^m$ 作为 $\hat{\mathbf{Y}}$。

这是 inference 时的一个重要 trick——**不用 sampling，用 argmax**。这意味着 model 实际上在做 "select best mode" 而非 "sample from distribution"。这让 closed-loop 行为更 deterministic、更 reproducible，但也牺牲了一些 distribution diversity 的优势。可能 sampling 在 closed-loop 中会导致行为不稳定（每次 trajectory 不同，controller 难以 track）。

---

## 5. 实验数据深度解读

### 5.1 Open-loop nuScenes（Table 1）

| Method | Params (M) | L2 3s ↓ | Frechet ↓ | NLL ↓ | Speed JSD ↓ |
|---|---|---|---|---|---|
| UniAD | 55.7 | 1.65 | 2.60 | 10.86 | 0.45 |
| VAD-Tiny | 39.6 | 1.76 | 2.65 | 7.22 | 0.43 |
| VAD-Base | 58.1 | 1.69 | **2.50** | 7.72 | 0.41 |
| DiffusionDrive | 60.0 | 1.58 | 2.41 | 3.95 | 0.39 |
| **BranchOut** | **41.9** | **1.41** | **2.29** | **3.72** | **0.36** |
| BranchOut w/ EgoHistory | 42.4 | 1.30 | 2.25 | 3.74 | 0.35 |

**关键观察**：

1. **L2 vs Frechet 的反转**：UniAD 在 L2 上比 VAD-Base 好 4.1%（1.65 vs 1.69），但在 Frechet 上 VAD-Base 反超（2.50 vs 2.60）。这说明 UniAD 过拟合 single GT，预测的轨迹 "对但单一"；VAD-Base 的预测虽然偏离 GT，但属于 "合理人类行为"。

2. **BranchOut 全面领先**：在所有 distribution-based metrics（Frechet、NLL、JSD）上都是 SOTA，且只用 41.9M params（VAD-Base 58.1M，DiffusionDrive 60M）。

3. **Ablation 的洞察**：
   - w/o GMM（保留 diffusion + 单 head）：L2 0.90, Frechet 2.43, NLL 4.11
   - w/o Diffusion（保留 GMM，去掉 diffusion）：L2 0.87, Frechet 2.35, NLL 3.80
   - Full BranchOut：L2 0.83, Frechet 2.29, NLL 3.72
   
   **GMM 对 multimodality 的贡献 > diffusion**！这是一个反直觉但很重要的结论。Diffusion 提供 sample diversity，但 GMM head 提供显式的 mode structure。两者叠加 gain 是 complementary 的（从 0.87 → 0.83，再改善 Frechet 2.35 → 2.29）。

### 5.2 Closed-loop HUGSIM（Table 3）

| Method | NC ↑ | DAC ↑ | TTC ↑ | COM ↑ | Rc ↑ | HD-Score ↑ |
|---|---|---|---|---|---|---|
| UniAD | 0.70 | 0.95 | 0.58 | 0.81 | 0.34 | 0.25 |
| VAD-Tiny | 0.44 | 0.80 | 0.34 | 1.00 | 0.32 | 0.11 |
| VAD-Base | 0.56 | 0.87 | 0.43 | 1.00 | 0.28 | 0.14 |
| DiffusionDrive | 0.56 | 0.67 | 0.48 | 0.80 | 0.24 | 0.10 |
| **BranchOut** | **0.76** | **0.99** | **0.69** | **1.00** | **0.58** | **0.47** |

**关键观察**：

1. **BranchOut 的 HD-Score 是 VAD-Tiny 的 4.3 倍**（0.47 vs 0.11），是 UniAD 的 1.88 倍。这是一个 dramatic 的 closed-loop 提升。
2. **Route Completion $R_c$ 从 0.34 → 0.58**（+70.5%）：这是 closed-loop 中真正完成 route 的比例。DiffusionDrive 只有 0.24，说明 anchor-based diffusion 在 closed-loop 中反而更差。
3. **NC（No Collision）0.76**：远超 VAD-Tiny 的 0.44 和 DiffusionDrive 的 0.56。GMM 的 multimodal reasoning 让 model 在面对 dynamic agent 时能 "考虑多种可能" 并选择最安全的。
4. **DAC（Drivable Area Compliance）0.99**：几乎不驶出可行驶区域。

### 5.3 RealADSim Leaderboard（Table 4）

| Rank | Team | Rc ↑ | HD-Score ↑ |
|---|---|---|---|
| 1 | UT/NV | 0.5905 | 0.4190 |
| 2 | NVIDIA | 0.4601 | 0.4012 |
| 3 | **BranchOut** | 0.3950 | 0.3016 |
| 4 | ReturnO_o | 0.2822 | 0.2303 |

BranchOut 是 academic team，只用 nuScenes 训练，没做任何 special tricks，就拿到第三。第一名 UT/NV 和第二名 NVIDIA 都是 industry team，大概率有更多 data 和 compute。这个结果说明 **multimodal modeling 本身的 design 比 scale 更重要**——这也是作者的核心论点。

### 5.4 Multimodal Human Benchmark（论文 4 节）

这是这篇 paper 最有意思的 contribution 之一。作者请了 40 个 participant，用 Logitech controller + VR headset 在 HUGSIM 的 photorealistic sim 中 driving nuScenes 场景，收集 human driving trajectories。

发现：
- 人与人的 minL2 是 **0.79m 3s**（即同一场景下不同人的最佳匹配轨迹差异）
- 这个数字 **比所有 SOTA model 都低**（BranchOut 是 1.41m）
- 说明 human driving 本身就是 multimodal 的，single GT 严重 underrepresent 了合理行为空间

这个 benchmark 让作者能计算：
- **Frechet distance**：pred set 与 human trajectory set 之间的 min Frechet
- **NLL**：pred distribution 在 human trajectory 上的 likelihood
- **JSD**：pred speed distribution 与 human speed distribution 的 Jensen-Shannon divergence

这套 metric 的意义：**它在 evaluation 层面解决了 "合理但不同" 的 credit assignment 问题**。一个 model 预测的轨迹如果落在 human trajectory 的 support 内，应该得到奖励，而不是 L2 惩罚。

---

## 6. 关于 EgoStatus 的 Ablation

Table 1 最后两行：
- BranchOut w/ EgoStatus: L2 0.75, Frechet 2.35, NLL 3.79
- BranchOut w/ EgoHistory: L2 0.74, Frechet 2.25, NLL 3.74

加上 ego status / history 后 open-loop L2 进一步降低（0.83 → 0.75）。这与 Ego-MLP [Li et al., CVPR 2024] 的发现一致：**ego status 包含大量 shortcut 信息**（速度、加速度、heading 直接决定了未来 1s 的轨迹）。

但作者主 model 故意不加 ego status，原因可能是：
1. Open-loop L2 提升是 shortcut 带来的虚假提升
2. Closed-loop 中 ego status 来自 model 自己的过去 prediction，error 会 compound
3. 为了 generalization（HUGSIM 上 sim 的 ego status 可能与 nuScenes 不一致）

这与 Jaeger et al. 的 "Hidden Biases of End-to-End Driving Models" [ICCV 2023] 的结论一致：ego status 是 open-loop 评价的作弊路径。

---

## 7. 我的 Intuition 总结

把这篇 paper 当作 teaching material，我觉得有几个 takeaways 值得 build intuition：

### Intuition 1: Diffusion 不自动给你 multimodality
Diffusion 在理论上是 multimodal 的（可以 sample 任意 mode），但实际训练中 mode collapse 严重。原因：
- L2 loss 倾向 dominant mode
- 有限的 training steps 和 noise schedule 让 model 偏向高 density region
- 没有 explicit 的 mode structure 约束

GMM head 把 multimodality **显式 parameterize**：K 个 mean + K 个 weight，每个 mean 是一个 mode 的 anchor。这比依赖 diffusion 的 implicit multimodality 更 sample-efficient、更可控。

### Intuition 2: Hard routing > Soft conditioning
用 command 做 hard branch 选择（只激活对应 MLP）比把 command 作为 embedding 输入到 shared MLP 更好。原因：
- Hard routing 让每个 maneuver 有 dedicated capacity
- 避免 different maneuver 的 gradient 互相 average
- 类似 Mixture of Experts 的思想，但 expert 选择是 deterministic 的（来自上游 command）

### Intuition 3: Evaluation 决定了你优化的是什么
Single GT L2 优化的是 "predict the exact GT"。Multimodal Frechet 优化的是 "predict within the support of human behavior"。这两个目标在 closed-loop 中表现截然不同——前者过拟合 geometric detail，后者学到 behavioral diversity。

作者用 40-person human benchmark 这个 "重武器" 证明了 evaluation 的重要性。这跟 ImageNet → CLIP 的 evolution 类似：从 single-label classification 到 contrastive learning，evaluation 范式变了，model 能力也跃迁了。

### Intuition 4: Closed-loop 是 ultimate test
Open-loop L2 排名（UniAD > VAD-Base > BranchOut w/o EgoHistory）和 closed-loop HD-Score 排名（BranchOut >> UniAD > VAD-Base）完全不一致。这说明 open-loop metric 与真实 driving 性能 weakly correlated。

未来 autonomous driving 的研究应该以 closed-loop 为主，open-loop 为辅。HUGSIM 这样的 photorealistic closed-loop sim 是关键 infrastructure。

---

## 8. 局限性与可能的 follow-up

1. **Single-step inference 用 argmax**：牺牲了 distribution sampling 的好处。可以试试 stochastic sampling + smoothing（如 EMA trajectory）来平衡 diversity 和 stability。
2. **M = 3 commands** 太粗：实际 driving 有更细的 maneuver（lane change left/right, yield, proceed, merge）。可以扩展到 M = 10+ 的 fine-grained maneuver。
3. **GMM isotropic assumption**：每个 Gaussian component 假设是对角协方差，可能不够 expressive。可以试 full covariance 或 normalizing flow 作为 component。
4. **Only nuScenes training**：跨 dataset generalization（HUGSIM 含 KITTI-360/Waymo/PandaSet）已经不错，但加上 more data 应该能进一步提升 leaderboard rank。
5. **No reactive agent modeling**：当前 model 只预测 ego，没建模 agent 的 reactive behavior。在 closed-loop 中，agent 会对 ego 行为做出反应，这需要 multi-agent game-theoretic modeling（如 GameFormer）。

---

## 9. References & 进一步阅读

- **VAD** (基础架构): https://arxiv.org/abs/2303.12077
- **DiffusionDrive** (直接 baseline): https://arxiv.org/abs/2410.02983
- **Diffusion Planner** (nuPlan 上的 diffusion baseline): https://arxiv.org/abs/2501.15564
- **DPM-Solver++** (单步采样): https://arxiv.org/abs/2211.01095
- **HUGSIM** (closed-loop benchmark): https://arxiv.org/abs/2410.15766
- **Ego-MLP** (ego status shortcut 分析): https://arxiv.org/abs/2312.03019
- **Hidden Biases of End-to-End Driving Models** (ego status bias): https://arxiv.org/abs/2307.07962
- **UniAD** (经典 baseline): https://arxiv.org/abs/2212.10156
- **RealADSim Leaderboard**: https://realadsim.net/
- **Diffusion Policy** (robotics 中的 diffusion policy): https://arxiv.org/abs/2305.20070
- **Motion Diversification Networks** (同一作者组前作): https://arxiv.org/abs/2404.09976

---

## 10. 一句话总结

BranchOut 的核心贡献是用一个 **explicit GMM head + command-based hard branch** 让 diffusion planner 真正学到 human driving 的 multimodal decision distribution，并通过一个 **40-person human benchmark** 揭示了 single-GT L2 评价的盲点。它的成功证明：在 autonomous driving 的 end-to-end planning 中，**model 的输出结构（head design）和 evaluation protocol 比单纯的 model scale 更重要**。这是一个对 academic researcher 非常友好的结论——你不需要 NVIDIA 那样的 compute，只需要正确的 inductive bias。
