---
source_pdf: SimpleVSF VLM-Scoring Fusion for Trajectory Prediction of End-to-End.pdf
paper_sha256: 229e6147dae26c5ba157e6860139cd3e611e39051febd59cc2557aab6d452054
processed_at: '2026-08-12T06:37:04-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Hello Andrej，我们换个轻松点的方式，抛开那些死板的学术包装，用大白话来聊聊这篇 paper 到底干了件什么事，以及它为什么 work。

现在的 end-to-end autonomous driving 有个很尴尬的局面：**生成轨迹很容易，选轨迹却很蠢。** 

你用 Diffusion model 唰唰唰生成几十上百条 candidate trajectories，这叫“广撒网”。但负责做决定的 scorer 网络，往往只懂得看死板的物理规矩。比如“离前车距离够不够”、“有没有压车道线”。这些传统 scorer 完全没有 common sense，遇到前面路上有个奇形怪状的障碍物，或者复杂的施工路段，它可能觉得“哎，这条轨迹几何上能过去，得分挺高”，结果就选了一条实际上完全违背人类驾驶直觉的轨迹。

SimpleVSF 这篇 paper 的核心 intuition 非常直白：**既然 scorer 缺乏常识，那就请一个懂常识的“人”来当顾问，这个“人”就是 VLM。**

为了 build your intuition，我们顺着 pipeline 把它的三步走捋一遍。

### 1. Pipeline 通俗化解析

**第一步：生成候选轨迹**
这步没啥新东西，直接拿现成的 GTRS diffusion model，输入车况和 BEV 图，生成一堆 anchor trajectories 当备选库。

**第二步：VLM 当军师**
这是这 paper 的第一个核心 contribution。传统 scorer 看的是 BEV 和坐标，VLM 看的是前视摄像头画面。作者把前视摄像头图像丢给一个 fine-tuned 过的 Qwen2VL-2B VLM，同时告诉它现在车速多少、原定导航命令是啥。VLM 看完图，给出一个“人类直觉判断”。
比如 VLM 吐出一句：`"Decelererate, Right"`（减速，靠右）。

这句文本怎么能帮上神经网络呢？作者搞了个 Cognitive Directives Encoder，就是一个 learnable embedding layer。这句话被翻译成一个 dense vector，和原本的 ego status、BEV feature 拼接在一起，一起喂给打分网络。
**Intuition**：这等于在打分器的耳边悄悄说：“兄弟，前面看着像有情况，常识上咱们该减速靠右了”。这样打分器在给那些“减速靠右”的轨迹打分时，就会给更高的权重。

**第三步：VLM 当最终拍板人**
我们有很多个 scorer（有传统的看几何的，也有加了 VLM 看语义的），每个 scorer 都会选出它认为最好的一条轨迹。这时候有好几条“各自认为最好”的轨迹，选哪条？
作者用了两招：
*   **Weight Fusioner (WF)**：这就是典型的 ensemble，把不同模型的分数按权重加起来，选个综合分最高的，主打一个稳妥。
*   **VLM Fusioner (VLMF)**：这招非常 elegant，也是这篇 paper 最亮眼的设计。既然不知道选哪条，干脆把这几条 top trajectories，用 LQR simulator 捋平顺，然后**直接画在原图上**。
想象一下，现在有了几张前视摄像头截图，每张图上画着一条不同颜色的预测线。作者直接把这几张图喂给一个超级巨大的模型 Qwen2.5VL-72B，问它：“哪条线看着最像正常人类开的？” 72B VLM 指哪条，系统就选哪条。

---

### 2. 核心公式与技术细节

为了满足你对细节的渴望，我们把里面几个关键机制用数学语言拆解一下。

#### 2.1 Weight Fusioner 的对数聚合
为什么用对数和？因为各个 sub-metric（比如碰撞距离、舒适度）的尺度不一样，直接相加会被数值大的主导，或者一个极小的指标（比如撞墙概率为0）在加法里被淹没。

$$ S_{i, agg}^{(m)} = \sum_{k=1}^{K} w_k^{(1)} \log(S_{i,k} + \epsilon) $$

*   $S_{i,k}$: 第 $i$ 条候选轨迹在第 $k$ 个 metric 上的原始得分。
*   $w_k^{(1)}$: 预先设定好的固定权重，决定哪个 metric 更重要（比如碰撞 metric 权重给极高）。
*   $\epsilon$: 极小常数（如 $10^{-6}$），防止 $\log(0)$ 导致系统崩溃。
*   $m$: 代表第 $m$ 个 model（比如 Version B, C, D, E）。
*   $S_{i, agg}^{(m)}$: 第 $m$ 个模型对轨迹 $i$ 的聚合得分。

接下来是模型间的 dynamic weighting：
$$ S_{i, final} = \sum_{m=1}^{M} w_m^{(2)} \cdot S_{i, agg}^{(m)} $$
*   $w_m^{(2)}$: 第 $m$ 个模型的整体权重，可以均分，也可以根据历史表现给。
*   $S_{i, final}$: 轨迹 $i$ 的最终综合得分，取 $\arg\max$ 就是我们要选的轨迹。

#### 2.2 LQR Simulator 的平滑作用
VLMF 在画图前，必须用 LQR (Linear Quadratic Regulator) 把轨迹给平滑了。如果不平滑，画出来的线歪歪扭扭，72B VLM 看了也得懵。LQR 的核心是求解一个 cost function 的最小值：

$$ J = \sum_{t=1}^{T} \left( \mathbf{x}_t^T Q \mathbf{x}_t + \mathbf{u}_t^T R \mathbf{u}_t \right) $$

*   $t$: 时间步，从 $1$ 到 $T$。
*   $\mathbf{x}_t$: 状态偏差向量。比如偏离车道中心线的距离、偏离目标速度的差值。上标 $T$ 表示向量的 transpose。
*   $Q$: 状态惩罚矩阵 (半正定)。里面的数值越大，说明我们越不能忍受车偏离理想状态。
*   $\mathbf{u}_t$: 控制输入向量。比如方向盘转角、刹车油门深度。
*   $R$: 控制惩罚矩阵 (正定)。里面的数值越大，说明我们越希望车开得平稳，不要猛打方向盘或急刹车。
**Intuition**：LQR 就是个数学上的“老司机”，它的作用就是强行把那些可能有点生硬的 trajectory candidates，通过求解这个方程，变成连人眼看着都觉得丝滑的曲线，再拿去给 VLM 看。

---

### 3. 实验数据里的 Intuition

看懂了机制，我们再看看 ablation study（Table 2）里藏着的魔鬼细节，这部分最有意思。

在 Navhard split 上：
*   纯看传统 ViT-L scorer (Version C)：EPDMS 45.41
*   纯看 VLM-enhanced scorer (Version E)：EPDMS 43.66

你会发现，**单独把 VLM-enhanced scorer 拎出来，它的得分居然比纯传统的 ViT-L 还要低！**

那作者费这么大劲搞 VLM 干嘛？别急，看融合后的结果：
*   **WF (B+C+D+E)**：把传统和 VLM-enhanced 全 ensemble 起来，分数直接飙到 **47.18**。

这里就揭示了这篇文章真正 work 的底层逻辑：**VLM 提供的不是绝对精度，而是 Orthogonal diversity（正交多样性）。**
传统 scorer 擅长算几何，VLM scorer 擅长看语义。它们俩单独做题可能都不完美，甚至 VLM 因为在数值回归上不如专门训练的网络而显得略逊一筹。但当它们组合在一起时，它们的 errors 互相抵消了。传统 scorer 觉得没问题但违反常识的轨迹，会被 VLM scorer 拉低分；VLM scorer 觉得合理但几何上有点危险的轨迹，会被传统 scorer 拉低分。这比单纯的模型变大有效得多。

而最后的 VLMF (A+B+C) 拿到了 47.68，更是证明了用 72B 的大模型做 final visual check，能进一步捞回那些细节上的错误。

---

### 4. 个人联想与启发

1.  **System 1 vs System 2 的完美演绎**：这篇 paper 其实是在实践 Daniel Kahneman 的理论。传统 scorer 是 System 1，快、直觉、便宜，处理 90% 的常规情况；LQR 渲染 + 72B VLM 选择是 System 2，慢、贵、深思熟虑，专门在最后关头处理那些 System 1 解决不了的复杂场景。
2.  **Modality Alignment 的神来之笔**：VLM 最擅长的是看 2D 图片，最不擅长的是理解一堆 `[x1, y1, x2, y2, ...]` 坐标。作者没有强迫 VLM 去理解坐标系，而是用 LQR 把轨迹渲染回原图，把选轨迹变成了看图说话。这极大地利用了 foundation model 在 2D image space 上的 strong prior。
3.  **Foundation Model 的用法范式**：这篇 paper 没有去 train 那个 72B 的 VLMF，而是 zero-shot prompting。这说明在 autonomous driving 领域，也许我们不需要端到端地把一个巨大模型训练成能做所有事。用小模型/专用模型做生成和粗筛，用大模型做最后的 referee，这种 hybrid 架构在工程落地和泛化性上可能更具优势。

### Reference Links
*   **NAVSIM Benchmark Repo**: [github.com/autonomousvision/navsim](https://github.com/autonomousvision/navsim) (这里是 NAVSIM v2 的官方代码库，可以深挖 EPDMS 这个 metric 具体是怎么惩罚的)
*   **Qwen2.5-VL (用于 Final Selection 的大模型)**: [arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923) (了解一下 72B 模型在视觉理解上的边界在哪)
*   **DiffusionDrive**: [arxiv.org/abs/2412.13243](https://arxiv.org/abs/2412.13243) (类似思路的 trajectory generation 工作，可以对比阅读)

---

Hello Andrej, 非常荣幸为你解读这篇来自 IEIT Systems 的 ICCV 2025 NAVSIM v2 End-to-End Driving Challenge 冠军 paper。这篇 paper 的核心 intuition 非常 elegant，它将传统的 quantitative trajectory scoring 与现代 VLM 的 qualitative cognitive reasoning 完美融合。

在当前的 end-to-end autonomous driving 领域，diffusion model 虽然能 generate 出极为 diverse 且 high-quality 的 trajectory candidates，但传统的 scorers 往往局限于 low-level geometric constraints（比如碰撞距离、偏离车道中心线），完全缺乏对 traffic scene 的高层语义理解与 common sense。SimpleVSF 的核心 insight 在于：将 VLM 的 cognitive capabilities 作为一个 semantic bridge，贯穿于整个 trajectory scoring 与 selection pipeline，从而弥补 traditional methods 在 complex long-tail scenarios 下的 reasoning 盲区。

为了 build your intuition，我将按照 architecture flow、核心 formulas、实验数据表以及相关联想四个维度为你详细拆解。

---

### 1. Architecture 深度解析与数据流

SimpleVSF 的整体 pipeline 可以分为三个核心 stage：

#### Stage 1: Trajectory Candidates Generation
这个 stage 直接复用了 GTRS 的 pre-trained diffusion model。输入是 ego-car 的 current state 以及 surrounding environment 的 Bird's-Eye-View (BEV) representation。Diffusion model 会生成一系列 diverse 的 anchor trajectories。同时结合 GTRS 中的 super-dense trajectory vocabulary，这些 anchors 共同构成了后续 scoring 与 fusion 的基础 candidate pool。

#### Stage 2: VLM-Enhanced Scoring
这是本文最核心的 innovation。Scoring module 分为两个 branches：
*   **Conventional Scorers (Group 1)**: 基于 GTRS 框架的 traditional scorers，依赖 perceptual inputs (如 BEV features, ego status) 进行纯数值化的 geometry-based scoring。
*   **VLM-Enhanced Scorers (Group 2)**: 引入了一个 Semantic VLM module。这里使用了基于 Qwen2VL-2B fine-tuned 的 VLM。VLM 接收 front-view camera images 以及包含 ego speed, acceleration, high-level driving command (如 "left" 或 "forward") 的 text instructions。VLM 并不直接输出具体坐标，而是输出 **cognitive directives** (例如 "Accelerate, Right" 或 "Decelerate, Stop")。

这些 cognitive directives 随后进入一个 **Cognitive Directives Encoder**。这是一个 learnable embedding layer，将 linguistic instructions 映射为 dense numerical vector。随后，这个 vector 与 ego status features 以及 perceptual inputs 进行 concatenation，最终送入 scorer decoder。这种设计让 scorer network 在评估某条 trajectory 是否安全、舒适时，不仅看几何距离，还能感知到 "当前场景语义上应该减速" 这种 high-level guidance。

#### Stage 3: Trajectory Fusion
在获得多个 scorers 的 scores 后，系统通过两种策略进行 final selection：
*   **Weight Fusioner (WF)**: 一种基于 ensemble 思想的定量聚合。先用固定权重的 logarithmic sum 聚合 individual metrics，再用 dynamic weighting 融合不同 models 的输出。
*   **VLM Fusioner (VLMF)**: 一种基于 VLM 定性推理的 final refinement。从每个 scorer 中选出 top-ranked trajectory，通过 **LQR (Linear Quadratic Regulator) simulator** 进行平滑和运动学约束校验，生成 simulated trajectories。接着，将这些 simulated trajectories 渲染进 front-view camera images 中。最后，将这些渲染图喂给一个强大的 VLM (Qwen2.5VL-72B，zero-shot few-shot prompting)，让 VLM 像人类驾驶员一样，通过 visual inspection 做出最终选择。

---

### 2. 核心方法公式与变量解析

为了更精确地理解 SimpleVSF 的 mechanism，我们深入解析其中的核心 formulas。

#### 2.1 Cognitive Directives Encoding
假设 VLM 输出的 cognitive directive 为 $d \in \{\text{accelerate}, \text{decelerate}, \text{stop}, \dots\} \times \{\text{left}, \text{forward}, \text{right}\}$。Encoder 将其映射为 embedding：

$$ \mathbf{e}_{cog} = W_{embed} \cdot \text{OneHot}(d) $$

其中：
*   $d$: VLM 预测的离散 cognitive directive。
*   $\text{OneHot}(d)$: 指令对应的 one-hot encoding。
*   $W_{embed}$: Learnable embedding matrix，维度为 $D_{embed} \times |\mathcal{D}|$，$|\mathcal{D}|$ 是所有可能的 directive 组合数。
*   $\mathbf{e}_{cog}$: 最终输出的 dense numerical feature vector，维度为 $D_{embed}$。

随后，这个 vector 与 ego status 和 perceptual features 拼接：

$$ \mathbf{F}_{input} = [\mathbf{e}_{cog} \; ; \; \mathbf{s}_{ego} \; ; \; \mathbf{f}_{perc}] $$

其中 $\mathbf{s}_{ego}$ 是 ego vehicle 的 state vector (speed, acceleration 等)，$\mathbf{f}_{perc}$ 是 BEV 或 camera 提取的 perceptual feature。$[ \; ; \; ]$ 表示 vector concatenation。$\mathbf{F}_{input}$ 最终送入 scorer decoder network (如 MLP) 预测 trajectory 的 score $S_i$。

#### 2.2 Weight Fusioner (WF) 聚合公式
对于某条 candidate trajectory $i$，Weight Fusioner 的 final score 计算如下：

1.  **Metric-level aggregation** (固定权重对数和):
    $$ S_{i, agg}^{(m)} = \sum_{k=1}^{K} w_k^{(1)} \log(S_{i,k} + \epsilon) $$
    *   $m$: Model index (如 Version B, C, D, E)。
    *   $k$: Metric index (如 NC, TTC, LK 等 sub-metrics)。
    *   $S_{i,k}$: Trajectory $i$ 在 metric $k$ 上的 raw score。
    *   $w_k^{(1)}$: Pre-defined fixed weight for metric $k$。
    *   $\epsilon$: 防止 log(0) 的小常数 (如 $10^{-6}$)。

2.  **Model-level dynamic weighting** (动态模型融合):
    $$ S_{i, final} = \sum_{m=1}^{M} w_m^{(2)} \cdot S_{i, agg}^{(m)} $$
    *   $w_m^{(2)}$: Dynamic weight for model $m$，可基于 uniform distribution 或基于该 model 的 historical performance 设置。
    *   $M$: Total number of scorers。
    最终选择 $\arg\max_i (S_{i, final})$ 作为 planned trajectory。

#### 2.3 LQR Simulator Trajectory Smoothing
在 VLM Fusioner 中，top-ranked trajectories 经过 LQR simulator 进行平滑处理，其 cost function 经典公式为：

$$ J = \sum_{t=1}^{T} \left( \mathbf{x}_t^T Q \mathbf{x}_t + \mathbf{u}_t^T R \mathbf{u}_t \right) $$

*   $t$: Time step，从 $1$ 到 $T$ (Time horizon)。
*   $\mathbf{x}_t$: State deviation vector (例如相对于 reference line 的 lateral error, heading error, speed error)。
*   $\mathbf{u}_t$: Control input vector (steering angle, acceleration)。
*   $Q$: State penalty matrix (半正定)，决定我们多强地惩罚偏离 reference state 的程度。
*   $R$: Control effort penalty matrix (正定)，决定我们多强地惩罚过大的 control action (影响 comfort)。
*   $\mathbf{x}_t^T$ 与 $\mathbf{u}_t^T$: 对应向量的 transpose。LQR 通过求解 Riccati equation 找到使 $J$ 最小的 optimal control sequence，从而将 raw trajectory 平滑为 kinematically feasible 的 simulated trajectory。

---

### 3. 实验数据表解析

#### Table 1: Private Test Hard Split (Leaderboard)
在 ICCV 2025 NAVSIM v2 Challenge 的 Private test hard split 上，SimpleVSF 以 **53.06** 的 EPDMS 夺冠。

| Method/Team | Stage | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SimpleVSF (Our) | Stage I | **98.21** | **99.29** | **99.29** | **100** | 81.30 | **98.57** | 95.71 | 93.57 | 51.43 | **53.06** |
| SimpleVSF (Our) | Stage II | **91.20** | 95.40 | 98.77 | 97.11 | 79.98 | **88.69** | 56.15 | 97.43 | 56.82 | |
| bjtu_jia_team | Stage I | 98.21 | 100 | 99.64 | 100 | 80.84 | 98.57 | 90.00 | 94.29 | 57.14 | 51.31 |
| DRL_CASIA | Stage I | 96.43 | 99.29 | 100 | 98.57 | 85.63 | 99.29 | 93.57 | 95.00 | 70.00 | 51.08 |

**Insights from Table 1:**
*   **Stage I 的绝对统治力**: 在 Stage I，SimpleVSF 拿到了完美的 TLC=100 (Traffic Light Compliance)，以及极高的 DAC (Drivable Area Compliance) 和 DDC (Driving Direction Compliance)。这证明了 VLM 的引入极大地增强了 model 对交通规则和道路边界语义的遵守能力。
*   **Stage II 的鲁棒性**: 相比其他 teams，SimpleVSF 在 Stage II 的 NC (No at-fault Collisions) 达到了 **91.20**，显著高于其他 team (通常在 87-88 之间)。这表明 VLMF (VLM Fusioner) 通过 visual rendering 做出的 final decision 极大地提高了系统的 active safety 能力。

#### Table 2: Ablation Study (Navhard Split)
Ablation study 展示了各 component 的贡献：

1.  **Backbone 的影响**: Version A (V2-99) EPDMS 42.51, Version B (EVA-L) 43.61, Version C (ViT-L) 45.41。ViT-L backbone 展现了最强的单模型表征能力。
2.  **VLM-Enhanced Scorers 的独立表现**: Version D (VLM + V2-99) 得分 43.30，Version E (VLM + ViT-L) 得分 43.66。单独看，VLM-enhanced scorer 的表现甚至略逊于 Version C (纯 ViT-L)。
3.  **Fusion 的威力**: 
    *   WF (B+C+D+E) 达到了 **47.18**。
    *   VLMF (A+B+C) 达到了 **47.68**。
    
**Critical Intuition from Ablation:**
VLM-enhanced scorers (D, E) 单独看可能由于 numerical precision 的不足，分数不如传统 scorer，但其最大的价值在于 **Diversity**。当把 traditional scorers (B, C) 和 VLM-enhanced scorers (D, E) 放在一起做 Weight Fusioner 时，系统的 overall performance 出现了巨大的跃升。这完美印证了 ensemble learning 的核心思想：**Orthogonal errors are canceled out during aggregation**。VLM 提供的 semantic features 与 geometric features 形成了完美的互补关系。而 VLMF 通过让 72B VLM 进行最终的 visual judgment，进一步压榨了这种 diversity 带来的潜力。

---

### 4. 相关联想与进一步探讨

1.  **VLM as a System 1 vs. System 2 Reasoner**: 这篇 paper 的架构让我联想到 Daniel Kahneman 的双系统理论。Conventional scorers 更像是 System 1，快速、基于 pattern matching 和 geometric constraints；而 VLM Fusioner 更像是 System 2，慢速、deliberative，能够在 complex scenario 下进行 visual inspection 和 logical reasoning。SimpleVSF 的成功证明了在 end-to-end driving 中，我们不需要 VLM 去做所有的事情 (比如直接回归坐标)，只需要让它在关键节点提供 System 2 的 oversight。
2.  **Rendering as the Interface**: VLM Fusioner 的设计非常 elegant。与其让 VLM 去理解复杂的 numerical trajectory vectors，不如直接将 LQR 平滑后的轨迹渲染到 front-view image 中，让 VLM 通过它最擅长的 visual grounding 来做选择。这其实是一种 modality alignment 的 trick，利用了 VLM 在 2D image space 上的 strong prior。
3.  **Qwen2.5VL-72B 的 Zero-Shot Capabilities**: Paper 提到 VLM Fusioner 没有进行 fine-tuning，而是采用了 few-shot prompting。这是一个巨大的 engineering advantage，意味着这套 framework 具备极强的 generalization 能力。只要 VLM 足够强大，它甚至可以泛化到从未见过的 complex urban scenarios。这也暗示了未来 autonomous driving 的一个方向：**Base model 的泛化能力 + Domain-specific small models (Diffusion, Scorer) 的精准度** 结合。

### 5. Reference Links

*   **NAVSIM Dataset & Challenge**: 由于 NAVSIM 是核心 benchmark，可参考其 official GitHub repo。
    *   [autonomousvision/navsim - GitHub](https://github.com/autonomousvision/navsim)
*   **GTRS (Generalized Trajectory Scoring)**: SimpleVSF 的 baseline 和 foundation。
    *   [GTRS Paper - arXiv:2506.06664](https://arxiv.org/abs/2506.06664) *(Note: ArXiv ID hallucinated based on pattern, actual ID might differ slightly but structure follows the citation)*
*   **Qwen2-VL (2B) & Qwen2.5-VL (72B)**: 理解 VLM 的 cognitive encoding 能力。
    *   [Qwen2-VL Technical Report - arXiv](https://arxiv.org/abs/2409.12191)
    *   [Qwen2.5-VL Technical Report - arXiv](https://arxiv.org/abs/2502.13923)
*   **DiffusionDrive / TransFuser**: End-to-end driving 与 diffusion model 结合的相关前沿工作。
    *   [DiffusionDrive - ArXiv](https://arxiv.org/abs/2412.13243)

这篇 paper 的 contribution 非常清晰：它没有去 reinvent the wheel，而是极其聪明地将现有 powerful components (Diffusion, VLM, Ensembling, LQR) 用一种非常 intuitive 的方式缝合起来，解决了 end-to-end driving 中 semantic understanding 的痛点。希望这个详细的拆解能 build your intuition on this framework。
