---
source_pdf: That Chip Has Sailed.pdf
paper_sha256: 1a169408ccab9e32e92db6e0fd135d6cc4cb077ff312a30bff713df633221e18
processed_at: '2026-08-12T13:48:50-07:00'
target_folder: AI芯片
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这场"芯片设计 AI 之争"

## 这场架是怎么起来的

2021 年，Google DeepMind 的 Azalia Mirhoseini、Anna Goldie 和 Jeff Dean 在 *Nature* 发了篇 paper [30]，叫 AlphaChip。核心 claim：用 deep RL 做 chip macro placement，能超过人类专家，已经用在 Google TPU 生产里了。

EDA 圈子立刻炸锅。Chip placement 是 EDA 几十年的硬骨头，全是启发式方法（simulated annealing、force-directed、analytical solver 比如 RePlAce [13]）。突然来一群 ML 人说"我比你强"，老派 EDA 学者天然不舒服。

领头质疑的是 Igor Markov，密歇根大学教授，EDA 圈大佬。他 2024 年 11 月在 *Communications of the ACM* 发了篇文章 [27]，叫 "Reevaluating Google's RL for IC Macro Placement"，包装成 "meta-analysis"，核心指控：AlphaChip 的结果不可复现、疑似 cherry-picking、疑似 data leakage、疑似造假。

Markov 的"meta-analysis"合并三篇东西：

1. AlphaChip 原始 Nature paper
2. Cheng et al. ISPD 2023 [9] — UCSD 的 Chung-Kuan Cheng 和 UT Austin 的 Andrew Kahng 带头的复现尝试
3. 一份匿名 PDF [3]，标题 "Stronger baselines..."，被说成 "Google Team 2 的独立评估"，**但 Markov 自己其实是 co-author，没披露**

Goldie 等人这篇 "That Chip Has Sailed" 就是怼回去的。标题是双关："the ship has sailed" — 船已经开了，你们质疑得太晚了，AlphaChip 早已大规模部署。

## AlphaChip 到底做了啥（人话版）

Chip placement 就是把一堆 macros（SRAM、IO block、ASIC subsystem）摆到芯片 canvas 上，目标线短、不堵、密度均匀。

传统做法是 numerical optimization：解一个连续 relaxation，再 round 到 grid。RePlAce [13] 就是这类，基于 force-directed 模拟，把 net 拉成弹簧，让节点受力平衡。

AlphaChip 换了个思路：**把 placement 看成 sequential decision-making**。一个一个 macro 往 canvas 上放，每放一个就是一个 action，放完所有 macro 算一次 reward，reward 是负的 proxy cost：

$$R = -\bigl(\alpha \cdot \text{HPWL} + \beta \cdot \text{Cong} + \gamma \cdot \text{Density}\bigr)$$

- HPWL：Half-Perimeter Wirelength，所有 net bounding box 半周长之和，估线长
- Cong：routing congestion 估计
- Density：局部密度惩罚
- $\alpha, \beta, \gamma$ 是权重

Policy network 是个 hybrid：
- **Edge-GNN** 编码 netlist 拓扑（节点是 macro / standard cell cluster，边是 net）
- **CNN** 编码当前 canvas 的 density / congestion 图像
- **Policy head** 输出"下一个 macro 放到哪个 grid cell"的 softmax 分布
- **Value head** 估 expected return

训练用 PPO，分布式收集 experience。

## Pre-training 是 AlphaChip 的灵魂

这是整场争论的核心。AlphaChip 在 20 个 TPU block 上 pre-train policy，再 fine-tune 或 zero-shot 到新 block。

这个 pre-training 学到的是什么？我给你 build 个 intuition。

GNN 在多个 netlist 上跑 message passing，每层更新节点 embedding：

$$h_v^{(l+1)} = \text{MLP}_{\text{self}}(h_v^{(l)}) + \sum_{u \in \mathcal{N}(v)} \text{MLP}_{\text{msg}}(h_u^{(l)}, h_v^{(l)}, e_{uv})$$

经过 $L$ 层后，节点 $v$ 的 embedding $h_v$ 包含了它 $L$-hop 邻域的结构信息。经过 20 个 block 训练，GNN 权重编码了 chip 设计里常见的拓扑 motif：

- **Pipeline 链**：stage 之间串联，要摆成一条线
- **Memory bank 阵列**：SRAM bank 紧挨着 controller
- **Crossbar / NoC**：高度对称的 router 拓扑
- **Clock tree root**：fan-out 最大的节点
- **Datapath**：宽总线、并行 unit

这些 motif 在不同 chip block 里反复出现。pre-trained GNN "认得"这些 pattern，policy head 可以直接利用学过的 placement 模式。

这跟 LLM 在 web-scale text 上 pre-train 学到 syntactic + semantic prior 是同一个 paradigm。"P" in GPT = "pre-trained"。AlphaGo [14] 在百万棋谱上 pre-train，才打得过人类。

Nature Figure 4（本文 Figure 3）量化了这个 gap：在 Ariane RISC-V 上，random-init policy 要 **48 小时**才能达到 pre-trained policy **6 小时**的水平。8× 加速。

## Markov 这边的核心指控

Markov 的"meta-analysis"指控主要有：

1. **不可复现**：引用 Cheng et al. 说"跑了 AlphaChip，没看出优势"
2. **Cherry-picking**：怀疑 Goldie 等人选了对自己有利的 block
3. **Data leakage**：怀疑 train / test block 之间有信息泄漏
4. **Misreporting**：怀疑数据造假
5. **TPU 部署是 dogfooding**：怀疑 Google 为了撑 paper，强行用劣质 AlphaChip placement

最后这一条特别离谱。Google TPU 是数十亿美元项目，支撑 Google Cloud 和整个 AI 业务。为了撑一篇 Nature paper 而故意用劣质 placement，逻辑上说不通。

## Goldie 等人怎么怼回去：Cheng et al. 根本没按方法跑

这是论文技术含量最高的部分。Cheng et al. 声称复现了 AlphaChip，但实际设置跟 Nature paper 差了一大截：

| 设置 | Nature (AlphaChip) | Cheng et al. |
|---|---|---|
| Pre-training | 48h on 20 blocks | **0h，完全没 pre-train** |
| RL collectors | 512 | 26（少 20×） |
| GPUs | 16 | 8（少 2×） |
| 训练到收敛 | 是 | **否，160k-350k steps 就停** |
| Tech node | sub-7nm | 45nm, 12nm |

**用人话讲**：

Cheng et al. 拿一个没 pre-train、用了 1/20 算力、没训完的"AlphaChip"，跑在跟 Google 完全不同工艺节点的老芯片上，然后说"AlphaChip 不行"。这就像你拿一个没 pre-train 的 GPT、用 1/20 的 GPU 跑、训了一半，然后说"GPT 不行"。

Cheng et al. 还借口说 Google 的 open-source repo [21] "不支持 pre-training"。Goldie 等人反驳：pre-training 就是简单地在多个 example 上跑训练，repo 一直支持。

更糟糕的是 Cheng et al. 自己的 Tensorboard（Figure 5）显示：

- **MemPool-NG45**：100k 步开始 loss 跳，250k 步停止 → 没收敛
- **Ariane-GF12**：130k 步 loss 跳，350k 步停止 → 没收敛
- **BlackParrot-GF12**：160k 步还在下降 → 过早停止
- **MemPool-GF12**：250k 步还在下降 → 过早停止
- **Ariane-NG45 / BlackParrot-NG45**：连 Tensorboard 都没给

RL 训练中 loss 暂时跳一下是 policy 改进时 value function 重调整的常见现象。但 Cheng et al. 既没训到 plateau，也没给足 sample。结果差是意料之中。

## 关于工艺节点的不可比性

Nature 用 sub-7nm TPU block。Cheng et al. 用 45nm（NG45）和 12nm（GF12）。

物理意义：sub-10nm 之后用 multiple patterning [15, 38]（SAQP、LELELE 等），routing congestion 在较低 density 就出现。45nm/12nm 的 congestion profile 完全不同。

AlphaChip 的 reward 里 $\beta$（congestion 权重）和 $\gamma$（density 权重）是为 7nm 调的。在 45nm 上不调参，reward 信号就不对，policy 自然学歪。

Cheng et al. 拒绝提供 NG45 的 synthesized netlists（10+ 次请求被拒），导致 Google 都没法复现他们的结果。AutoDMP [2] 在同样 NG45 block 上跑过，结果跟 Cheng et al. Table 1 不一致，进一步暴露问题。

## 关于 ablation 的设计错误

Cheng et al. 做了个"ablation"，声称 AlphaChip 的 RL agent 偷偷用了 physical synthesis 给的 initial placement 信息。

他们做法：把所有 standard cell 堆在 canvas 左下角，再跑 cluster rebalancing。结果性能崩。Cheng et al. 据此说"RL 利用了 initial placement"。

**用人话讲**：这个 ablation 设计有 confounding。他们同时改了两件事：
1. 是否使用 initial placement
2. cluster 质量（因为堆在角落导致 hMETIS 聚类 degenerate）

性能崩到底是哪个原因？Cheng et al. 直接归咎于 (1)，但可能是 (2)。

Goldie 等人做了正确 ablation：直接 skip cluster rebalancing，把 hMETIS 的 `UBFactor` 参数调到最严格（`UBFactor=1`），让 hMETIS 自己生成平衡 cluster [23]。结果（Table 2）：

| TPU-v6 Block | Wirelength | WNS | TNS | Density | Cong (H) | Cong (V) |
|---|---|---|---|---|---|---|
| **with** initial placement | 5,176 | -0.046 | -2.466 | 23.830 | 0.01 | 0.01 |
| **without** initial placement | 5,133 | -0.048 | -2.583 | 23.827 | 0.01 | 0.01 |

**Wirelength 反而更好**（5,133 < 5,176），其他指标几乎一样。证明 RL agent 根本没用 initial placement——它的输入只有 GNN node embedding 和 canvas density map，拿不到 initial placement。

Ablation 的核心原则：只改一个变量，控制其他不变。Cheng et al. 违反了这条。

## 关于 proxy 与 final metric 的相关性

Cheng et al. 说 AlphaChip 的 proxy cost 跟 final metric 不相关。但他们自己的 Table 2（Figure 6）显示：

| Proxy 元素 | Std Cell Area | rWL | Power | WNS | TNS |
|---|---|---|---|---|---|
| Wirelength | -0.221 | **+0.317** | -0.144 | +0.163 | **+0.317** |
| Congestion | -0.029 | +0.086 | -0.010 | +0.105 | -0.048 |
| Density | +0.096 | +0.230 | +0.096 | **+0.268** | +0.077 |
| **Overall Proxy** | -0.010 | **+0.257** | +0.048 | **+0.200** | +0.048 |

除了 Std Cell Area（被当 hard constraint 不优化），其他全是**正相关**。Wirelength proxy 跟 rWL +0.317，跟 TNS +0.317，说明 proxy 确实反映真实线长。

Proxy 弱相关是 RL 训练信号的常态。LLM 训练的 next-token cross-entropy 跟下游 benchmark 也是弱相关，但训练信号有效。AutoDMP [2] 用类似 proxy，确认与 final metric 相关。

Cheng et al. 还做了几个奇怪选择：
- 只报告 proxy < 0.9 的样本相关（无理由，排除大部分 result）
- 只在一个 45nm test case（Ariane-NG45）做 study，不能代表 7nm 以下
- 没调 congestion/density 权重适应老 node

## Markov 那份"匿名 PDF"的问题

Markov "meta-analysis" 第二个 source 是份匿名 PDF [3]，标题 "Stronger baselines for evaluating deep RL in chip placement"，没作者列表，被说成 "Google Team 2 的独立评估"。

实际情况（Goldie 等人披露）：
- Markov 自己是 co-author，但没披露
- 这份 PDF 从未发表
- 2022 年 Google 独立委员会 review 后拒绝发表，理由："the claims and conclusions in the draft are not scientifically backed by the experiments" [33]
- 委员会发现：AlphaChip 在原数据集上的结果被独立复现了，**反而是 Markov et al. 的 RL 结果无法复现**
- Goldie 等人给委员会提供了一行脚本，生成的 RL 结果比 Markov et al. 报告的好，甚至超过他们自己"加强版"的 simulated annealing baseline

换句话说，Markov 拿一份自己 co-author 但被 Google 拒绝发表的、不能复现的 PDF，当独立证据用。

## "Whistleblower" 的真相

Markov 引用一个 Google 内部 "whistleblower" 支持造假指控。但加州 Santa Clara County Superior Court 公开文件 (Case 22CV398683) [24] 显示，该人 2022 年 6 月 29 日的 declaration 里自己说：

> "he stated that he suspected that the research being conducted by Goldie and Mirhoseini was fraudulent, **but also stated that he did not have evidence to support his suspicion of fraud**"

他自己都说没证据。这其实是一桩 Google 内部劳动纠纷延伸出来的，跟学术 fraud 没关系。

## Nature 怎么处理的

时间线：
- 2023-09：Nature 公布 Editor's note，开始调查 + 二次 peer review
- 2024-04：Nature 完成 investigation，结论："the best way forward is to publish an update to the paper in the form of an **Addendum** (not a 'Correction', as we have established that there is little that actually needs correcting)" [44]
- 2024-09：Nature 发表 Addendum [20]，移除 Editor's note

注意 Nature 用 "Addendum" 不是 "Correction"。Correction 是说原文有错要改。Addendum 是补充说明，原文基本无错。Markov 在 2024 年 11 月 CACM 重发同样指控，相当于无视 Nature 调查结论。

## AlphaChip 真的部署了吗

Goldie 等人给出 production 时间线：

| 时间 | 部署 |
|---|---|
| 2020-08 | **10 个** AlphaChip layouts tape-out in TPU v5e |
| 2021-09 | **15 个** AlphaChip layouts tape-out in TPU v5p |
| 2022-10 | **25 个** AlphaChip layouts tape-out in Trillium |
| 2024-03 | **7 个** AlphaChip layouts in Google Axion (ARM CPU) |
| 2024-09 | MediaTek SVP 宣布采用 AlphaChip [19] |

Figure 1 显示每一代 TPU 中 AlphaChip 占比上升，相对人类专家的优势 margin 扩大。TPU 是数十亿美元项目，支撑 Google Cloud 和 AI 业务。Markov 说 Google "为了撑 paper 故意用劣质 placement" — 这个 conspiracy 假设非常荒谬。

## 这场争论的本质：方法论之争

往深里看，这场架其实是 **EDA 启发式派 vs. ML-based 派** 的领地之争。

Markov、Cheng、Kahng 这些人几十年都在做 EDA 算法。RePlAce [13] 就是 Kahng 团队做的。他们的方法论核心：**numerical optimization + domain-specific heuristic**。每个问题要人手设计 cost function、调参、加 special case。

AlphaChip 换 paradigm：**learning-based**。policy 从数据里学，不需要人手设计 heuristic。pre-training 让知识可迁移。

这跟当年 deep learning 在 computer vision 替代 hand-crafted feature 是同一类范式之争。feature engineering 派一开始也不信 neural net 能超过精心设计的 SIFT/HOG。后来 AlexNet 在 ImageNet 上把所有人打服。

EDA 圈现在正经历类似 shift。Synopsys DSO.ai、Cadence Cerebrus 都在做 RL for EDA [8, 12, 29, 34]。MediaTek 2024 年 9 月公开采用 AlphaChip [19]。学术界 follow-up 一大堆：ChiPFormer [39]、MaskPlace [40]、RL-CCD [41]、AutoDMP [2]。

Markov 的质疑某种程度上是 EDA 老派面对新 paradigm 的"本能防御"。但质疑要拿证据，要按方法跑。Cheng et al. 那种"跑了一半、没 pre-train、少 20× 算力、老 node、不调参"的"复现"，在 ML 标准下根本不算复现。

## 给 ML 研究者的几个 takeaway

1. **RL 复现比监督学习难得多**。compute、pre-training、convergence 三件事全都要对齐。任何"复现失败"的工作，先检查这三件事。Cheng et al. 三件事全没对齐。

2. **Pre-training 在 combinatorial optimization 上是真的 work**。AlphaChip 的 transfer learning 来自 GNN 在多个 netlist 上学到的可迁移 representation。这跟 LLM、AlphaGo 是同一 paradigm。

3. **Ablation 设计要 isolate variable**。Cheng et al. 的 initial-placement ablation 因为 confound cluster 质量，结论无效。Table 2 才是正确 ablation — 只改 IP usage，结果无差异。

4. **Proxy 与 final metric 不必强相关**。只要符号对、方向对，弱 proxy 也能驱动 RL 学到有用 policy。LLM 训练也是这样。

5. **Meta-analysis 的伦理底线**。Markov 把自己 co-author 的未发表 PDF 当独立证据，且不披露。这在学术伦理上严重越界。

6. **Institutional 路径**。Nature 二次 peer review 是 institutional 解决机制，结果是支持 AlphaChip。Markov 在 CACM 重发同样指控，绕开 institutional 路径。

7. **Production deployment 是最强证据**。一篇 paper 可以质疑，但 AlphaChip 在三代 TPU + Axion CPU + MediaTek 部署，是几十亿美元项目的实际生产证据。这种规模的 deployment 不可能为撑一篇 paper 而做。

## 相关链接

- 原始 Nature paper：https://www.nature.com/articles/s41586-021-03525-w
- Nature Addendum (2024-09)：https://www.nature.com/articles/s41586-021-03525-w
- Open-source Circuit Training repo：https://github.com/google_research/circuit_training
- 2020 arXiv preprint：https://arxiv.org/abs/2004.10746
- Cheng et al. ISPD 2023：https://arxiv.org/abs/2306.14167
- Markov CACM 2024：https://cacm.acm.org/research/reevaluating-googles-reinforcement-learning-for-ic-macro-placement/
- AutoDMP ISPD 2023：https://dl.acm.org/doi/10.1145/3563758
- DREAMPlace ICCAD 2021：https://arxiv.org/abs/2104.12317
- RePlAce TCAD 2019：https://ieeexplore.ieee.org/document/8594509
- AlphaGo Nature 2016：https://www.nature.com/articles/nature16961
- Google Court Declaration (22CV398683)：https://portal.scscourt.org/search
- DeepMind blog "How AlphaChip transformed chip design"：https://deepmind.google/discover/blog/how-alphachip-transformed-computer-chip-design/
- TILOS MacroPlacement repo：https://github.com/TILOS-AI-Institute/MacroPlacement
- Ariane RISC-V (现 CVA6)：https://github.com/openhwgroup/cva6
- ISPD 2015 benchmark：https://www.ispd.cc/contests/
- Synopsys DSO.ai：https://www.synopsys.com/ai/what-is-ai-chip-design.html
- Cadence RL for EDA：https://www.cadence.com/en_US/home/explore/reinforcement-learning.html

## 最后一句

"That Chip Has Sailed" 这个标题已经说明态度了：船开了，你们质疑晚了。AlphaChip 在 TPU v5e、v5p、Trillium、Axion、MediaTek 都已部署，学界 follow-up 一大堆。Markov 这场质疑在方法学上站不住脚，在伦理上有问题（不披露 co-author 身份），在 institutional 路径上被 Nature 二次 review 否决。

EDA 圈正在经历 ML paradigm shift，跟当年 CV 圈经历的一样。老派会本能抵抗，但最终看 result。Result 现在很清楚：AlphaChip work，deployment 说话。

---

# That Chip Has Sailed — Paper Deep Dive

## 一、这场争论的全景：AlphaChip vs. Markov 的"meta-analysis"

这篇 paper 是 Google DeepMind 的 Anna Goldie、Azalia Mirhoseini 和 Jeff Dean 三人针对 Igor Markov（密歇根大学教授、EDA 领域权威）2024 年 11 月发表在 *Communications of the ACM* 的 "meta-analysis" 文章所做的反击。Markov 的文章质疑他们 2021 年发表于 *Nature* 的 AlphaChip 工作（"A Graph Placement Methodology for Fast Chip Design" [30]）。

Markov 的 "meta-analysis" 本质上合并了三篇内容：

1. **Mirhoseini et al. (Nature 2021)** — 原始 AlphaChip paper，peer-reviewed。
2. **Cheng et al. (ISPD 2023)** — 一篇 invited（非 peer-reviewed）的复现尝试 [9]，作者为 Chung-Kuan Cheng、Andrew B. Kahng 等 UCSD/UT Austin 学者。
3. **Markov et al.（未发表 PDF）** — 一份匿名的、标题为 "Stronger baselines for evaluating deep reinforcement learning in chip placement" 的 PDF [3]，被描述为 "Google Team 2" 的独立评估，但 Markov 本人其实是 co-author，并未披露 [33]。

Google 这边的核心论点：**Markov 的两份"反证"都不构成有效的科学反驳**——Cheng et al. 完全没有按 Nature 论文的方法跑（无 pre-training、少 20× 算力、未训练到收敛、不具代表性的 45nm/12nm node），而 Markov et al. 的内部 PDF 在 2022 年就被 Google 独立委员会认定"the claims and conclusions in the draft are not scientifically backed by the experiments" [33]，且其 RL 结果无法被复现，反而是 AlphaChip 原数据集上的结果被独立复现了。

## 二、AlphaChip 的方法回顾（为 build intuition）

AlphaChip 把 chip placement 形式化为一个 sequential decision-making 问题，用 deep RL 求解。先复述关键组件，便于后面分析 Cheng et al. 哪里"歪"了。

### 2.1 Problem formulation

给定一个 netlist $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，其中：
- $\mathcal{V} = \mathcal{V}_m \cup \mathcal{V}_s$：节点包含 macros（$\mathcal{V}_m$，大块如 SRAM、IO、ASIC blocks）和 standard cell clusters（$\mathcal{V}_s$，把 standard cell 用 hMETIS [23] 聚类后的虚拟节点）。
- $\mathcal{E}$：net 连接，每条边 $e_{uv}$ 表示 node $u$ 与 node $v$ 之间有信号连接。

目标：将每个 macro 放到 canvas grid 的某 cell 上，最小化 proxy cost：
$$\text{cost}(\pi) = \alpha \cdot \text{HPWL}(\pi) + \beta \cdot \text{Cong}(\pi) + \gamma \cdot \text{Density}(\pi)$$

其中：
- $\pi: \mathcal{V}_m \to \text{grid cell}$ 是放置策略
- HPWL 是 Half-Perimeter Wirelength，bounding box 半周长估计
- $\text{Cong}$ 是 congestion 估计（基于 routing demand/supply）
- $\text{Density}$ 是局部密度，惩罚过密区域
- $\alpha, \beta, \gamma$ 是权重超参

Reward 定义为 $R = -\text{cost}$，episode 末尾给一次 sparse reward。

### 2.2 Policy network 架构

AlphaChip policy 是个 hybrid network：

1. **Graph embedding (Edge-GNN)**：对 netlist 做 message passing。节点 $v$ 在第 $l$ 层的 embedding 更新：
$$h_v^{(l+1)} = \text{MLP}_{\text{self}}\!\left(h_v^{(l)}\right) + \sum_{u \in \mathcal{N}(v)} \text{MLP}_{\text{msg}}\!\left(h_u^{(l)}, h_v^{(l)}, e_{uv}\right)$$
经过 $L$ 层后得到 $\{h_v\}_{v \in \mathcal{V}}$，再 mean-pool 得到 graph-level embedding $h_{\mathcal{G}}$。

2. **State embedding (CNN)**：当前 canvas 上 density / congestion 图像用 CNN 编码为 $h_{\text{canvas}}$。

3. **Current macro embedding**：当前要放的 macro $v_t$ 的 node embedding $h_{v_t}$。

4. **Policy head**：
$$\pi_\theta(a_t \mid s_t) = \text{softmax}\!\left(W_{\text{out}} \cdot [h_{\mathcal{G}}; h_{\text{canvas}}; h_{v_t}] + b\right)$$
其中 $a_t$ 是放置到哪个 grid cell。

5. **Value head**：$V_\phi(s_t) = \text{MLP}([h_{\mathcal{G}}; h_{\text{canvas}}])$，输出 expected return。

训练用 PPO，多 GPU 分布式数据收集。

### 2.3 Pre-training 的关键

AlphaChip 最大的卖点之一：在 20 个 TPU block 上 pre-train policy，再 zero-shot 或 fine-tune 到新 block。Nature 论文中"pretrain" 一词出现 37 次。这是 learning-based 方法相对于 RePlAce [13] 等启发式方法的核心优势——**能从历史经验中学习可迁移的 placement knowledge**。

类比（论文里用的）：评估一个没见过 Go 棋谱的 AlphaGo [14]，然后说 AlphaGo 不强。或者评估一个没在 web-scale text 上 pre-train 的 GPT，然后说它语言能力不行——"P" in GPT 就是 "pre-trained"。

## 三、Cheng et al. 复现中的核心错误

Google 这边列了 5 个主要 methodological 偏差。这是整篇 paper 技术含量最高的部分。

### 3.1 完全没有 pre-training

| 设置 | Nature (AlphaChip) | Cheng et al. (ISPD 2023) |
|---|---|---|
| Pre-training | 48 hours on 20 blocks | **0 hours, no training data** |
| RL collectors | 512 | 26 (20× 少) |
| GPUs | 16 | 8 (2× 少) |
| Training | 48h，训练到收敛 | 160k–350k steps，**未收敛** |
| Tech node | sub-7nm | 45nm, 12nm |

Cheng et al. 借口是 open-source repo [21] "不支持 pre-training"。Goldie 等人反驳：pre-training 就是简单地在多个 example 上跑训练，repo 一直支持。Figure 3（Nature Figure 4 复现）显示，在 Ariane RISC-V 上：
- 预训练 policy 6 小时达到的 quality，random-init policy 要 **48 小时**才能逼近
- 这是一个 8× 的 wall-clock 加速

Intuition：pre-trained GNN 学到了 chip block 中常见的拓扑 motif（pipeline、crossbar、memory bank 阵列、control tree 等），这些 motif 在不同 block 间是 share 的，于是 policy 可以"识别"出新 block 的子结构并应用学过的 placement 模式。这就是 in-context / transfer learning 在 combinatorial optimization 上的体现，跟 LLM 学到 "syntactic + semantic prior" 是一类现象。

### 3.2 算力 20× 缩水 + 未训练到收敛

Figure 4 是从 follow-up paper [42] 复现的 GPU scaling 图：
- 左图：placement return（higher is better）随 GPU 数增加而提升；GPU=8 拿到最好的 -1.07 return，GPU=2 拿不到这个值。
- 右图：达到给定 return 所需时间随 GPU 数下降。

Cheng et al. 的 Tensorboard（Figure 5）显示：
- **MemPool-NG45**：~100k 步开始 divergence（loss 跳起来），250k 步停止。明显没收敛。
- **Ariane-GF12**：~130k 步 divergence，350k 步停止。明显没收敛。
- **BlackParrot-GF12**：160k 步仍在下降，被过早停止。
- **MemPool-GF12**：250k 步仍在下降，被过早停止。
- **Ariane-NG45 / BlackParrot-NG45**：连 Tensorboard 都没提供。

RL 训练中 "loss 看似 divergence" 不一定是真发散——policy 改进时 value function loss 暂时上升是常见现象。但 Cheng et al. 既没有训练到 plateau，也没有提供足够的 sample，结果"理所当然"差。这违反 ML 的基本实践 [1]。

### 3.3 测试用例：45nm/12nm 不可比

Nature 用 TPU sub-7nm node。Cheng et al. 用 NG45（45nm）和 GF12（12nm）。

物理意义上，sub-10nm 之后用 multiple patterning [15, 38]（SAQP、LELELE 等），routing congestion 在较低 density 下就出现。45nm/12nm 的 congestion profile 完全不同，需要调整 reward 的 $\beta / \gamma$ 权重。AlphaChip 在老 node 上没有调参，结果当然不优。

更进一步，Cheng et al. 拒绝提供 NG45 的 synthesized netlists（10+ 次请求被拒），导致连 Google 都无法复现他们的结果。AutoDMP [2] 在同样的 NG45 block 上跑过，结果与 Cheng et al. Table 1 不一致，进一步暴露 reproducibility 问题。

### 3.4 "Massive reimplementation" 引入 bug

Cheng et al. 没用 Google 的 open-source repo [21]，而是 "massive reimplementation" 了整个 pipeline。Goldie 等人指出 repo 在 2022 年 6 月就开源了，独立团队 TF-Agents 已经做过一次独立复现。

更糟糕的是，他们还 reverse-engineer 两个 binary function（proxy cost 和 force-directed standard cell placer）。Goldie 等人建议社区改用 DREAMPlace [26]（GPU 加速的 placement，MLCAD 2021 [22] 已推荐）来替换 FD placer。

## 四、关于 ablation 的设计正确性

Cheng et al. 做了一个"ablation"，声称 AlphaChip 的 RL agent 偷偷利用了 physical synthesis 给出的 initial placement 信息。他们的"ablation"做法：把所有 standard cell 堆在 canvas 的 lower-left corner，再跑 cluster rebalancing。

结果：因为 initial placement 完全 degenerate，hMETIS 聚类出的 cluster 也 degenerate，性能当然崩。Cheng et al. 据此说"RL 利用了 initial placement"。

Goldie 等人指出这是错误的 ablation 设计：**confound 了"是否使用 initial placement"和"是否产生 degenerate cluster"**两个变量。

正确 ablation：直接 skip cluster rebalancing step，用 hMETIS 的 `UBFactor=1`（最严格的平衡约束 [23]），让 hMETIS 自己生成平衡 cluster。Table 2 显示这样做的结果：

| TPU-v6 Block | Wirelength | WNS | TNS | Density | Cong (H) | Cong (V) |
|---|---|---|---|---|---|---|
| Clustering **with** initial placement | 5,176 | -0.046 | -2.466 | 23.830 | 0.01 | 0.01 |
| Clustering **without** initial placement | 5,133 | -0.048 | -2.583 | 23.827 | 0.01 | 0.01 |

**Wirelength 反而更好**（5,133 < 5,176），WNS / TNS / Density 基本相同。证明 RL agent 完全没用到 initial placement——因为它根本拿不到 initial placement，policy 输入只有 GNN node embedding 和 canvas density map。

Intuition：Ablation 的核心原则是只变动你想测试的那一个变量，控制其他所有变量不变。Cheng et al. 在改变"是否使用 IP"的同时，被动改变了"cluster 质量"，导致结论完全不可信。这是 ML 实验中常见的 confounding 错误。

## 五、关于 proxy 与 final metric 的相关性

Cheng et al. 的 Table 2（Figure 6 复现）显示 proxy cost 与 final metrics 的 Pearson 相关系数：

| Proxy 元素 | Std Cell Area | rWL | Power | WNS | TNS |
|---|---|---|---|---|---|
| Wirelength | -0.221 | **+0.317** | -0.144 | +0.163 | **+0.317** |
| Congestion | -0.029 | +0.086 | -0.010 | +0.105 | -0.048 |
| Density | +0.096 | +0.230 | +0.096 | **+0.268** | +0.077 |
| **Overall Proxy** | -0.010 | **+0.257** | +0.048 | **+0.200** | +0.048 |

几个观察：
1. Overall proxy 与 rWL 相关 +0.257，与 WNS +0.200，与 TNS +0.048。**全部为正**（除了 Std Cell Area，因为它被当作 hard constraint 不优化）。
2. Wirelength proxy 与 rWL、TNS 都是 +0.317，说明 wirelength 估计确实反映了 routing 后的真实线长。
3. 相关性"弱但正"是 RL 训练信号的特征。

类比：LLM 训练的 next-token cross-entropy loss 与下游 benchmark 性能也只弱相关，但训练信号是有效的。Goldie 等人引用了 AutoDMP [2]：用类似的 proxy cost，**确实**与 final metric 相关。

Cheng et al. 还有几个 surprising 的选择：
- 只报告 proxy < 0.9 的样本相关（无理由），这把大部分 result 排除掉了
- 只在一个 45nm test case（Ariane-NG45）做这个 study，不能代表 7nm 以下情况
- 没有调 congestion/density 权重以适应老 node

## 六、关于 "Google engineers confirmed" 的错误声明

Cheng et al. 在 Acknowledgments 里暗示 Google 工程师验证过他们的技术正确性。Goldie 等人澄清：那些 Google 工程师只是确认能在 open-source repo 的 quick-start guide 上从零训练单个 test case（Ariane）——这是 install check，不是方法复现。

这些工程师实际提过 concern：compute 用量太少、proxy weight 未调，但 Cheng et al. 未采纳。Acknowledgments 还列了 Nature 论文 corresponding authors，但他们事先根本不知道有这篇 paper。

## 七、"Whistleblower" 的真相

Markov 引用一个 Google 内部 "whistleblower" 支持 fraud 嫌疑。但根据加州 Santa Clara County Superior Court 公开文件 (Case No. 22CV398683) [24]，该人在 2022 年 6 月 29 日的 declaration 中承认：

> "he stated that he suspected that the research being conducted by Goldie and Mirhoseini was fraudulent, but also stated that he did not have evidence to support his suspicion of fraud"

也就是说，他自己都说没有证据。

## 八、Nature 的独立调查结论

时间线：
- 2023-09：Nature 公布 Editor's note，开始调查并启动二次 peer review
- 2024-04：Nature 完成 investigation，认定"the best way forward is to publish an update to the paper in the form of an Addendum (not a 'Correction', as we have established that there is little that actually needs correcting)" [44]
- 2024-09：Nature 发表 Addendum [20]，移除 Editor's note

注意 Nature 用 "Addendum" 而非 "Correction"，意味着原始论文基本无错。Markov 在 11 月再次发表同样的 meta-analysis，相当于无视 Nature 的调查结论。

## 九、AlphaChip 的真实部署证据

论文给出 production 部署时间线：

| 时间 | 部署 |
|---|---|
| 2020-04 | arXiv preprint |
| 2020-08 | **10 个** AlphaChip layouts tape-out in TPU v5e |
| 2021-06 | Nature 论文发表 |
| 2021-09 | **15 个** AlphaChip layouts tape-out in TPU v5p |
| 2022-01–07 | open-source AlphaChip（含独立复现）|
| 2022-02 | Google 独立委员会拒绝 Markov et al. 发表 [33] |
| 2022-10 | **25 个** AlphaChip layouts tape-out in Trillium |
| 2024-03 | **7 个** AlphaChip layouts in Google Axion (ARM CPU) |
| 2024-09 | MediaTek SVP 宣布采用 AlphaChip [19] |

Figure 1 显示每一代 TPU 中 AlphaChip 占比上升，相对人类专家的优势 margin 扩大。Google 不可能"为了一篇 paper"在价值数十亿美元、关键支撑 cloud 和 AI 业务的 TPU 项目上用劣质 placement——这种 conspiracy 假设非常荒谬。

## 十、给 ML 研究者的几个 intuition takeaway

1. **Pre-training 在 combinatorial optimization 中是真的 game-changer**：AlphaChip 的 transfer learning 来自 GNN 在不同 netlist 上学到的可迁移 representation。这跟 LLM、AlphaGo [14] 是同一个 paradigm——把"经验"以网络权重形式编码。

2. **Reproducibility 在 RL 中比监督学习难**：RL 的结果对 hyperparameter、sample 数、convergence 极敏感。"我跑了你的方法但没用你的算力"不是 reproducibility，是 misrepresentation。

3. **Ablation 的设计需要 isolate variable**：Cheng et al. 的 initial-placement ablation 因为 confounding cluster 质量，结论无效。Table 2 才是正确的 ablation——只改 IP usage，结果没差异。

4. **Proxy 与 final metric 的相关性不必强**：只要符号正确、方向正确，弱 proxy 也能驱动 RL 学习到有用的 policy。LLM 也是这样。

5. **Meta-analysis 的伦理**：Markov 把自己 co-authored 的未发表 PDF 当作独立证据，且未披露作者身份。这是严重的学术伦理问题。

6. **学术争议的 institutional 路径**：Nature 的二次 peer review 是 institutional 解决机制，结果是支持 AlphaChip。继续在 CACM 重发同样指控绕开了 institutional 路径。

## 十一、参考链接

- 原始 Nature paper（AlphaChip）：https://www.nature.com/articles/s41586-021-03525-w
- Nature Addendum (2024-09)：https://www.nature.com/articles/s41586-021-03525-w
- Open-source repo (Circuit Training)：https://github.com/google_research/circuit_training
- Nature 2020 arXiv preprint：https://arxiv.org/abs/2004.10746
- Cheng et al. ISPD 2023 paper：https://arxiv.org/abs/2306.14167
- Markov CACM 2024 article：https://cacm.acm.org/research/reevaluating-googles-reinforcement-learning-for-ic-macro-placement/
- AutoDMP (ISPD 2023)：https://dl.acm.org/doi/10.1145/3563758
- DREAMPlace (ICCAD 2021)：https://arxiv.org/abs/2104.12317
- RePlAce (TCAD 2019)：https://ieeexplore.ieee.org/document/8594509
- AlphaGo (Nature 2016)：https://www.nature.com/articles/nature16961
- Google Court Declaration (22CV398683)：https://portal.scscourt.org/search
- DeepMind blog "How AlphaChip transformed computer chip design"：https://deepmind.google/discover/blog/how-alphachip-transformed-computer-chip-design/
- TILOS MacroPlacement repo (Cheng et al. project)：https://github.com/TILOS-AI-Institute/MacroPlacement
- ISPD 2015 contest benchmark：https://www.ispd.cc/contests/
- Ariane RISC-V CPU：https://github.com/openhwgroup/cva6 (now CVA6)

## 十二、对 AlphaChip 工作的延续影响

AlphaChip 之后学界有一大批 follow-up，论文中引用了 [5, 7, 8, 10, 12, 17, 18, 29, 34, 37, 39, 40, 41, 43]，包括：

- **ChiPFormer** [39]：用 Offline Decision Transformer 做 placement，PMLR 2023
- **MaskPlace** [40]：visual representation learning 加速 placement，NeurIPS 2022
- **RL-CCD** [41]：用 attention-based self-supervised RL 做 concurrent clock and data optimization，DAC 2023
- **AutoDMP** [2]：基于 DREAMPlace 的 automated macro placement，ISPD 2023
- **Floorplan like Human Experts** [5]：RL floorplanning，DATE 2024
- **Macro Placement by Wire-Mask-Guided Black-Box Optimization** [43]：NeurIPS 2024

工业界：Synopsys DSO.ai、Cadence Cerebrus 都在做 RL for EDA。MediaTek 2024 年 9 月正式宣布采用 AlphaChip 加速其先进工艺芯片开发 [19]。

## 总结

"That Chip Has Sailed" 这个标题本身就是双关："the ship has sailed"（船已经开走了）暗示"质疑已经太晚了，AlphaChip 已经被广泛部署和验证"。这场争议本质上是 EDA 启发式派 vs. ML-based 派的方法论之争。Goldie 等人通过详尽的方法学对比（pre-training 缺失、算力缩水、未收敛、不可比 node、错误 ablation、错误披露），把 Cheng et al. 和 Markov 的质疑拆解得相当彻底。

从 ML research 角度看，这件事最大的教训是：**RL 复现不能像监督学习那样宽松**——compute、pre-training、convergence 三个维度都关键。任何声称"复现失败"的工作都需要先确认这三个维度是否对齐，再下结论。
